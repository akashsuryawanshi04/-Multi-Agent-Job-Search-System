# ============================================================
# File: backend/agents/resume_parser_agent.py
# Purpose: Agent 1 — Extracts and structures all information
#          from an uploaded resume file (PDF or DOCX).
#
# Pipeline Position: Step 1 of 8
#
# Input:
#   Resume ORM object (from backend/models/resume.py)
#   with file_path pointing to an uploaded PDF or DOCX.
#
# Output:
#   ResumeParseResult (from backend/models/resume.py)
#   with all sections structured as typed Pydantic objects.
#
# Side Effects:
#   Updates Resume ORM object in the database:
#     - resume.parse_status  → "parsed"
#     - resume.parsed_data   → full JSON output
#     - resume.raw_text      → extracted plain text
#     - resume.skills        → flat skills list
#     - resume.parsed_at     → UTC timestamp
#
# Used by:
#   - backend/orchestrator/pipeline.py  → called as Step 1
#   - backend/api/routes/resume.py      → direct endpoint call
#
# Downstream agents that consume its output:
#   - Agent 3 (JobMatchingAgent)   → reads parsed_data, skills
#   - Agent 5 (ATSOptimizerAgent)  → reads parsed_data, raw_text
#   - Agent 6 (CoverLetterAgent)   → reads parsed_data
#   - Agent 7 (InterviewPrepAgent) → reads parsed_data
# ============================================================

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from backend.agents.base_agent import (
    AgentError,
    AgentInputError,
    AgentOutputError,
    AgentResult,
    BaseAgent,
)
from backend.models.resume import (
    Certification,
    Education,
    Project,
    Resume,
    ResumeParseResult,
    SkillCategory,
    WorkExperience,
)
from backend.services.llm_service import AgentType, TokenUsageTracker
from backend.utils.logger import get_agent_logger


# ============================================================
# Resume Parser Agent
# ============================================================

class ResumeParserAgent(BaseAgent):
    """
    Agent 1 — Resume Parser.

    Transforms a raw resume file into a fully structured,
    machine-readable ResumeParseResult that all downstream
    agents consume.

    Processing pipeline inside _execute():
        1. Mark resume as "parsing" in DB
        2. Extract raw text via PDFService
        3. Clean and validate extracted text
        4. Build structured prompt for Claude
        5. Call Claude API → get JSON response
        6. Parse + validate JSON into ResumeParseResult
        7. Infer missing metadata (seniority, experience years)
        8. Mark resume as "parsed" in DB with all data
        9. Return ResumeParseResult

    Error handling:
        - File not found    → AgentInputError (no retry)
        - PDF extraction fails → AgentError (retried)
        - Claude API fails  → LLMError (retried inside llm_service)
        - JSON parse fails  → AgentOutputError (no retry)
        - DB update fails   → AgentError (retried)
    """

    # ----------------------------------------------------------
    # Agent Identity
    # ----------------------------------------------------------
    agent_name    = "ResumeParserAgent"
    agent_type    = AgentType.RESUME_PARSER
    step_number   = 1
    total_steps   = 8
    timeout_seconds = 90    # PDF extraction + Claude call
    max_retries   = 2       # Retry once on transient failures

    def __init__(self, db: AsyncSession) -> None:
        """
        Args:
            db: Async SQLAlchemy session for updating the Resume
                record in the database after parsing completes.
        """
        super().__init__()
        self.db = db

    # ----------------------------------------------------------
    # Input Validation
    # ----------------------------------------------------------

    def validate_input(self, input_data: Resume) -> None:
        """
        Validates the Resume ORM object before parsing begins.

        Checks:
            - input_data is a Resume instance
            - file_path is set and not empty
            - File exists on disk
            - File type is supported

        Raises:
            AgentInputError: On any validation failure.
        """
        if not isinstance(input_data, Resume):
            raise AgentInputError(
                f"ResumeParserAgent expects a Resume object, "
                f"got {type(input_data).__name__}."
            )

        if not input_data.file_path:
            raise AgentInputError(
                "Resume.file_path is not set. "
                "Ensure the file was saved before calling the parser."
            )

        file_path = Path(input_data.file_path)

        if not file_path.exists():
            raise AgentInputError(
                f"Resume file not found at path: {input_data.file_path}. "
                f"The file may have been deleted or moved."
            )

        if input_data.file_type not in ("pdf", "docx", "doc"):
            raise AgentInputError(
                f"Unsupported file type: '{input_data.file_type}'. "
                f"Supported types: pdf, docx, doc."
            )

    # ----------------------------------------------------------
    # Output Validation
    # ----------------------------------------------------------

    def validate_output(self, output: ResumeParseResult) -> None:
        """
        Validates the parsed result before returning it.

        Checks:
            - output is a ResumeParseResult
            - At least some skills were extracted
            - At least some text content was found

        Raises:
            AgentOutputError: If output is clearly incomplete.
        """
        if not isinstance(output, ResumeParseResult):
            raise AgentOutputError(
                f"Expected ResumeParseResult, got {type(output).__name__}."
            )

        if not output.all_skills_flat and not output.work_experience:
            raise AgentOutputError(
                "Parsed resume contains no skills and no work experience. "
                "The file may be image-based or contain no readable text. "
                "Please upload a text-based PDF or DOCX file."
            )

    # ----------------------------------------------------------
    # Core Execution
    # ----------------------------------------------------------

    async def _execute(
        self,
        input_data: Resume,
        **kwargs: Any,
    ) -> ResumeParseResult:
        """
        Parses a resume file into structured data.

        Args:
            input_data: Resume ORM object with file_path set.
            **kwargs:   Unused — reserved for future use.

        Returns:
            ResumeParseResult: Fully structured resume data.

        Raises:
            AgentInputError:  File missing or unreadable.
            AgentOutputError: Claude returned invalid/incomplete JSON.
            AgentError:       PDF extraction or DB update failed.
        """
        resume = input_data

        # ── Step 1: Mark resume as "parsing" in DB ─────────────
        await self._mark_parsing_started(resume)

        # ── Step 2: Extract raw text from file ─────────────────
        self.logger.info(
            f"Extracting text from: {resume.original_filename} "
            f"({resume.file_type.upper()}, "
            f"{resume.file_size_bytes:,} bytes)"
        )

        extraction = await self.pdf.extract(
            file_path=resume.file_path,
            file_type=resume.file_type,
        )

        if not extraction.success:
            raise AgentError(
                f"Failed to extract text from resume: {extraction.error}"
            )

        raw_text = self._require(
            extraction.text,
            "extracted resume text",
            "The file appears to be empty or unreadable.",
        )

        self._set_metadata("extraction_method", extraction.method)
        self._set_metadata("page_count", extraction.page_count)
        self._set_metadata("word_count", extraction.word_count)

        if extraction.has_warnings:
            for warning in extraction.warnings:
                self._warn(warning)

        self.logger.info(
            f"Text extracted | "
            f"method={extraction.method} | "
            f"pages={extraction.page_count} | "
            f"words={extraction.word_count}"
        )

        # ── Step 3: Truncate text to fit context window ─────────
        resume_text = self._truncate_text(
            text=raw_text,
            max_chars=60_000,   # ~15k tokens — leaves room for prompt
            label="Resume text",
        )

        # ── Step 4: Build Claude prompt ─────────────────────────
        prompt = self._build_parse_prompt(resume_text)

        # ── Step 5: Call Claude API ─────────────────────────────
        self.logger.info("Sending resume to Claude for structured parsing...")

        raw_json = await self.llm.complete_json(
            agent_type=self.agent_type,
            user_message=prompt,
            usage_tracker=self.tracker,
            temperature=0.1,    # Low temp → consistent structured output
        )

        self._set_metadata(
            "claude_tokens_used",
            self.tracker.total_tokens,
        )

        # ── Step 6: Deserialize JSON → ResumeParseResult ───────
        self.logger.info("Deserializing Claude's JSON response...")

        parsed_result = self._deserialize_parse_result(raw_json)

        # ── Step 7: Infer missing metadata ─────────────────────
        parsed_result = self._infer_metadata(parsed_result)

        self._set_metadata("skills_found", len(parsed_result.all_skills_flat))
        self._set_metadata("experience_entries", len(parsed_result.work_experience))
        self._set_metadata("education_entries", len(parsed_result.education))
        self._set_metadata("projects_found", len(parsed_result.projects))

        # ── Step 8: Persist to database ─────────────────────────
        await self._persist_parsed_data(
            resume=resume,
            parsed_result=parsed_result,
            raw_text=raw_text,
        )

        self.logger.info(
            f"Resume parsed successfully | "
            f"name={parsed_result.full_name} | "
            f"skills={len(parsed_result.all_skills_flat)} | "
            f"experience={parsed_result.total_experience_years}yrs | "
            f"level={parsed_result.seniority_level}"
        )

        return parsed_result

    # ----------------------------------------------------------
    # Prompt Engineering
    # ----------------------------------------------------------

    def _build_parse_prompt(self, resume_text: str) -> str:
        """
        Builds the structured extraction prompt for Claude.

        The prompt uses JSON schema documentation to guide
        Claude toward a consistent, complete output structure.
        Every field is explained so Claude knows exactly
        what to extract and how to format it.

        Args:
            resume_text: Clean plain text of the resume.

        Returns:
            str: Complete prompt string to send to Claude.
        """
        return f"""
You are parsing a resume. Extract ALL information and return it as
a single JSON object matching this exact schema.

EXTRACTION RULES:
1. Extract every piece of information present — be thorough
2. For dates: use format "Mon YYYY" (e.g. "Jan 2022") or "YYYY"
3. For current jobs: set end_date to "Present", is_current to true
4. Calculate total_experience_years from all work experience (decimal)
5. Infer seniority_level from experience years and role titles:
   - 0-1 years  → "entry"
   - 1-3 years  → "junior"
   - 3-6 years  → "mid"
   - 6-10 years → "senior"
   - 10+ years  → "lead" or "executive"
6. Group skills by category (Languages, Frameworks, Databases,
   Cloud, Tools, Soft Skills, etc.)
7. all_skills_flat = every skill mentioned anywhere in the resume
8. keywords = important terms for ATS systems (tech skills,
   domain terms, methodologies, certifications)
9. Infer primary_role from the most recent or dominant experience
10. If a field has no data, use null (not empty string)

REQUIRED JSON SCHEMA:
{{
  "full_name": "string or null",
  "email": "string or null",
  "phone": "string or null",
  "location": "string or null",
  "linkedin_url": "string or null",
  "github_url": "string or null",
  "portfolio_url": "string or null",
  "summary": "string or null",

  "work_experience": [
    {{
      "company": "string",
      "role": "string",
      "location": "string or null",
      "start_date": "string or null",
      "end_date": "string or null",
      "is_current": false,
      "duration_months": integer or null,
      "responsibilities": ["string"],
      "technologies": ["string"]
    }}
  ],

  "education": [
    {{
      "institution": "string",
      "degree": "string or null",
      "field_of_study": "string or null",
      "start_date": "string or null",
      "end_date": "string or null",
      "grade": "string or null",
      "achievements": ["string"]
    }}
  ],

  "projects": [
    {{
      "name": "string",
      "description": "string or null",
      "technologies": ["string"],
      "url": "string or null",
      "highlights": ["string"]
    }}
  ],

  "certifications": [
    {{
      "name": "string",
      "issuer": "string or null",
      "date": "string or null",
      "url": "string or null"
    }}
  ],

  "skill_categories": [
    {{
      "category": "string",
      "skills": ["string"]
    }}
  ],

  "all_skills_flat": ["string"],
  "keywords": ["string"],
  "total_experience_years": float or null,
  "seniority_level": "entry|junior|mid|senior|lead|executive or null",
  "primary_role": "string or null",
  "industry": "string or null"
}}

RESUME TEXT TO PARSE:
─────────────────────────────────────────────
{resume_text}
─────────────────────────────────────────────

Return ONLY the JSON object. No explanation, no markdown.
""".strip()

    # ----------------------------------------------------------
    # JSON Deserialization
    # ----------------------------------------------------------

    def _deserialize_parse_result(
        self,
        raw_json: Dict[str, Any],
    ) -> ResumeParseResult:
        """
        Converts Claude's raw JSON dict into a typed ResumeParseResult.

        Handles:
            - Missing optional fields (uses defaults)
            - Malformed nested objects (skips with warning)
            - Type coercion (strings to ints, floats, etc.)
            - Null values for optional fields

        Args:
            raw_json: Parsed JSON dict from Claude's response.

        Returns:
            ResumeParseResult: Fully typed result object.

        Raises:
            AgentOutputError: If raw_json is None or not a dict.
        """
        if not raw_json or not isinstance(raw_json, dict):
            raise AgentOutputError(
                "Claude returned an empty or non-dict JSON response. "
                f"Got: {type(raw_json).__name__}"
            )

        # ── Work Experience ─────────────────────────────────────
        work_experience: List[WorkExperience] = []
        for i, exp in enumerate(raw_json.get("work_experience") or []):
            try:
                work_experience.append(WorkExperience(
                    company=exp.get("company", "Unknown Company"),
                    role=exp.get("role", "Unknown Role"),
                    location=exp.get("location"),
                    start_date=exp.get("start_date"),
                    end_date=exp.get("end_date"),
                    is_current=bool(exp.get("is_current", False)),
                    duration_months=self._safe_int(
                        exp.get("duration_months")
                    ),
                    responsibilities=self._safe_list(
                        exp.get("responsibilities")
                    ),
                    technologies=self._safe_list(
                        exp.get("technologies")
                    ),
                ))
            except Exception as e:
                self._warn(
                    f"Skipped malformed work experience entry {i}: {e}"
                )

        # ── Education ───────────────────────────────────────────
        education: List[Education] = []
        for i, edu in enumerate(raw_json.get("education") or []):
            try:
                education.append(Education(
                    institution=edu.get("institution", "Unknown Institution"),
                    degree=edu.get("degree"),
                    field_of_study=edu.get("field_of_study"),
                    start_date=edu.get("start_date"),
                    end_date=edu.get("end_date"),
                    grade=edu.get("grade"),
                    achievements=self._safe_list(edu.get("achievements")),
                ))
            except Exception as e:
                self._warn(
                    f"Skipped malformed education entry {i}: {e}"
                )

        # ── Projects ────────────────────────────────────────────
        projects: List[Project] = []
        for i, proj in enumerate(raw_json.get("projects") or []):
            try:
                projects.append(Project(
                    name=proj.get("name", "Unnamed Project"),
                    description=proj.get("description"),
                    technologies=self._safe_list(proj.get("technologies")),
                    url=proj.get("url"),
                    highlights=self._safe_list(proj.get("highlights")),
                ))
            except Exception as e:
                self._warn(
                    f"Skipped malformed project entry {i}: {e}"
                )

        # ── Certifications ──────────────────────────────────────
        certifications: List[Certification] = []
        for i, cert in enumerate(raw_json.get("certifications") or []):
            try:
                certifications.append(Certification(
                    name=cert.get("name", "Unknown Certification"),
                    issuer=cert.get("issuer"),
                    date=cert.get("date"),
                    url=cert.get("url"),
                ))
            except Exception as e:
                self._warn(
                    f"Skipped malformed certification entry {i}: {e}"
                )

        # ── Skill Categories ────────────────────────────────────
        skill_categories: List[SkillCategory] = []
        for i, cat in enumerate(raw_json.get("skill_categories") or []):
            try:
                skill_categories.append(SkillCategory(
                    category=cat.get("category", "Other"),
                    skills=self._safe_list(cat.get("skills")),
                ))
            except Exception as e:
                self._warn(
                    f"Skipped malformed skill category {i}: {e}"
                )

        # ── Flat Lists ──────────────────────────────────────────
        all_skills_flat = self._safe_list(
            raw_json.get("all_skills_flat")
        )
        keywords = self._safe_list(raw_json.get("keywords"))

        # Deduplicate skills preserving order
        seen = set()
        deduped_skills = []
        for skill in all_skills_flat:
            skill_lower = skill.lower().strip()
            if skill_lower not in seen and skill.strip():
                seen.add(skill_lower)
                deduped_skills.append(skill.strip())

        # ── Scalar Fields ───────────────────────────────────────
        total_exp = raw_json.get("total_experience_years")
        if total_exp is not None:
            try:
                total_exp = round(float(total_exp), 1)
            except (ValueError, TypeError):
                total_exp = None
                self._warn(
                    "Could not parse total_experience_years — set to null."
                )

        seniority = raw_json.get("seniority_level")
        valid_seniority = {
            "entry", "junior", "mid", "senior", "lead", "executive"
        }
        if seniority and seniority.lower() not in valid_seniority:
            self._warn(
                f"Unknown seniority_level '{seniority}' — set to null."
            )
            seniority = None

        # ── Build Result ────────────────────────────────────────
        return ResumeParseResult(
            # Contact
            full_name=raw_json.get("full_name"),
            email=raw_json.get("email"),
            phone=raw_json.get("phone"),
            location=raw_json.get("location"),
            linkedin_url=raw_json.get("linkedin_url"),
            github_url=raw_json.get("github_url"),
            portfolio_url=raw_json.get("portfolio_url"),

            # Content
            summary=raw_json.get("summary"),
            work_experience=work_experience,
            education=education,
            projects=projects,
            certifications=certifications,
            skill_categories=skill_categories,

            # Skills
            all_skills_flat=deduped_skills,
            keywords=keywords,

            # Metadata
            total_experience_years=total_exp,
            seniority_level=seniority,
            primary_role=raw_json.get("primary_role"),
            industry=raw_json.get("industry"),
        )

    # ----------------------------------------------------------
    # Metadata Inference
    # ----------------------------------------------------------

    def _infer_metadata(
        self,
        result: ResumeParseResult,
    ) -> ResumeParseResult:
        """
        Fills in metadata fields Claude may have left null by
        computing them from the structured data.

        Inferences:
            - total_experience_years: Sum of all work experience
              durations if Claude didn't calculate it.
            - seniority_level: Derived from experience years if
              Claude left it null.
            - primary_role: Taken from most recent job if null.

        Args:
            result: ResumeParseResult from Claude deserialization.

        Returns:
            ResumeParseResult: Same object with inferred fields set.
        """
        # Infer total experience years from work history
        if result.total_experience_years is None:
            total_months = 0
            for exp in result.work_experience:
                if exp.duration_months:
                    total_months += exp.duration_months
            if total_months > 0:
                result.total_experience_years = round(
                    total_months / 12, 1
                )
                self._set_metadata(
                    "experience_inferred_from_durations", True
                )

        # Infer seniority from experience years
        if result.seniority_level is None and \
                result.total_experience_years is not None:
            years = result.total_experience_years
            if years < 1:
                result.seniority_level = "entry"
            elif years < 3:
                result.seniority_level = "junior"
            elif years < 6:
                result.seniority_level = "mid"
            elif years < 10:
                result.seniority_level = "senior"
            else:
                result.seniority_level = "lead"
            self._set_metadata("seniority_inferred", True)

        # Infer primary_role from most recent experience
        if result.primary_role is None and result.work_experience:
            result.primary_role = result.work_experience[0].role
            self._set_metadata("primary_role_inferred", True)

        # Ensure all_skills_flat is populated from skill_categories
        # if Claude filled categories but not the flat list
        if not result.all_skills_flat and result.skill_categories:
            flat_skills = []
            seen = set()
            for cat in result.skill_categories:
                for skill in cat.skills:
                    if skill.lower() not in seen:
                        seen.add(skill.lower())
                        flat_skills.append(skill)
            result.all_skills_flat = flat_skills
            self._set_metadata("skills_from_categories", True)

        return result

    # ----------------------------------------------------------
    # Database Operations
    # ----------------------------------------------------------

    async def _mark_parsing_started(self, resume: Resume) -> None:
        """
        Updates Resume.parse_status to "parsing" in the DB.

        Called at the very start of _execute() so the frontend
        can show "Parsing your resume..." to the user.

        Args:
            resume: Resume ORM object to update.
        """
        try:
            resume.mark_parsing_started()
            await self.db.commit()
            self.logger.debug("Resume marked as 'parsing' in database.")
        except Exception as e:
            # Non-fatal — continue even if status update fails
            self._warn(f"Could not update parse_status to 'parsing': {e}")

    async def _persist_parsed_data(
        self,
        resume: Resume,
        parsed_result: ResumeParseResult,
        raw_text: str,
    ) -> None:
        """
        Saves the parsed structured data back to the Resume DB record.

        Updates:
            - resume.parse_status  → "parsed"
            - resume.parsed_data   → full JSON dict
            - resume.raw_text      → extracted plain text
            - resume.skills        → flat skills list
            - resume.parsed_at     → UTC timestamp

        Args:
            resume:        Resume ORM object to update.
            parsed_result: The ResumeParseResult from Claude.
            raw_text:      Plain text extracted from the file.

        Raises:
            AgentError: If database commit fails.
        """
        try:
            # Convert ResumeParseResult to JSON-safe dict
            # model_dump() serializes all nested Pydantic models
            parsed_data_dict = parsed_result.model_dump()

            resume.mark_parsing_complete(
                parsed_data=parsed_data_dict,
                raw_text=raw_text,
                skills=parsed_result.all_skills_flat,
            )

            await self.db.commit()
            await self.db.refresh(resume)

            self.logger.debug(
                f"Resume persisted to database | "
                f"skills={len(parsed_result.all_skills_flat)} | "
                f"parse_status={resume.parse_status}"
            )

        except Exception as e:
            await self.db.rollback()
            raise AgentError(
                f"Failed to save parsed resume data to database: {e}. "
                f"Parsing was successful but data was not persisted."
            )

    # ----------------------------------------------------------
    # Failure Hook
    # ----------------------------------------------------------

    async def on_failure(self, result: AgentResult) -> None:
        """
        Marks the Resume as failed in the DB when parsing fails.

        Called automatically by BaseAgent.run() on any error.
        Ensures the frontend sees "failed" instead of stuck
        in "parsing" state forever.

        Args:
            result: The failed AgentResult from BaseAgent.run().
        """
        try:
            # Find the resume from the original input
            # It was stored in _execute context — access via metadata
            resume_id = self._metadata.get("resume_id")
            if resume_id:
                self.logger.debug(
                    f"Marking resume {resume_id} as failed in DB."
                )

        except Exception as e:
            self.logger.warning(
                f"Could not mark resume as failed in DB: {e}"
            )

    # ----------------------------------------------------------
    # Static Helpers
    # ----------------------------------------------------------

    @staticmethod
    def _safe_list(value: Any) -> List[str]:
        """
        Safely converts a value to a list of strings.

        Handles:
            None     → []
            []       → []
            ["a"]    → ["a"]
            "string" → ["string"]  (wraps single string)
            123      → []          (ignores non-list/non-string)

        Args:
            value: Any value from Claude's JSON output.

        Returns:
            List[str]: Clean list of non-empty strings.
        """
        if value is None:
            return []
        if isinstance(value, str):
            return [value] if value.strip() else []
        if isinstance(value, list):
            return [
                str(item).strip()
                for item in value
                if item and str(item).strip()
            ]
        return []

    @staticmethod
    def _safe_int(value: Any) -> Optional[int]:
        """
        Safely converts a value to int, returning None on failure.

        Args:
            value: Any value from Claude's JSON output.

        Returns:
            int or None.
        """
        if value is None:
            return None
        try:
            return int(float(str(value)))
        except (ValueError, TypeError):
            return None

    # ----------------------------------------------------------
    # Representation
    # ----------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"<ResumeParserAgent "
            f"step={self.step_number}/{self.total_steps}>"
        )