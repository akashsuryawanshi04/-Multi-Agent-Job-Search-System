# ============================================================
# File: backend/agents/job_search_agent.py
# Purpose: Agent 2 — Searches job boards for relevant listings
#          and persists them to the database.
#
# Pipeline Position: Step 2 of 8
#
# Input:
#   JobSearchInput dataclass containing:
#     - user_id:        UUID of the searching user
#     - resume_id:      UUID of the active resume
#     - parsed_resume:  ResumeParseResult from Agent 1
#     - search_params:  JobSearchRequest (role, location, etc.)
#
# Output:
#   JobSearchOutput dataclass containing:
#     - jobs:           List[Job] ORM objects (saved to DB)
#     - job_matches:    List[JobMatch] ORM objects (saved to DB)
#     - total_found:    Total jobs discovered
#     - sources_used:   Which platforms returned results
#
# Side Effects:
#   - Creates Job rows in the database (skips duplicates)
#   - Creates JobMatch rows linking jobs to user+resume
#   - Sets JobMatch.is_recommended = False (Agent 3 scores them)
#
# Used by:
#   - backend/orchestrator/pipeline.py → called as Step 2
#
# Downstream agents that consume its output:
#   - Agent 3 (JobMatchingAgent) → scores each JobMatch
# ============================================================

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.agents.base_agent import (
    AgentError,
    AgentInputError,
    AgentResult,
    BaseAgent,
)
from backend.models.job import Job, JobCreate, JobMatch, JobSearchRequest
from backend.models.resume import ResumeParseResult
from backend.services.llm_service import AgentType, TokenUsageTracker
from backend.services.scraper_service import (
    ScraperService,
    SearchParams,
    scraper_service,
)


# ============================================================
# Input / Output Data Classes
# ============================================================

@dataclass
class JobSearchInput:
    """
    Input data for Agent 2 (Job Search Agent).

    Constructed by the orchestrator from the pipeline context
    after Agent 1 completes.

    Attributes:
        user_id:        UUID of the user running the pipeline.
        resume_id:      UUID of the resume to link job matches to.
        parsed_resume:  Structured resume data from Agent 1.
        search_request: User's job search preferences.
    """
    user_id:        uuid.UUID
    resume_id:      uuid.UUID
    parsed_resume:  ResumeParseResult
    search_request: JobSearchRequest


@dataclass
class JobSearchOutput:
    """
    Output data from Agent 2 (Job Search Agent).

    Passed directly to Agent 3 as input.

    Attributes:
        jobs:         Job ORM objects saved to the database.
        job_matches:  JobMatch ORM objects linking jobs to user.
        total_found:  Total jobs the API reported finding.
        sources_used: Job board platforms that returned results.
        new_jobs:     Count of newly created Job rows.
        existing_jobs:Count of Job rows that already existed in DB.
    """
    jobs:          List[Job]
    job_matches:   List[JobMatch]
    total_found:   int
    sources_used:  List[str]
    new_jobs:      int       = 0
    existing_jobs: int       = 0

    @property
    def total_jobs(self) -> int:
        """Total jobs available for matching."""
        return len(self.jobs)


# ============================================================
# Job Search Agent
# ============================================================

class JobSearchAgent(BaseAgent):
    """
    Agent 2 — Job Search Agent.

    Discovers relevant job listings from multiple job boards
    and persists them to the database ready for scoring.

    Processing pipeline inside _execute():
        1. Build optimized search query from resume + preferences
        2. Call ScraperService to fetch live job listings
        3. For each fetched job:
           a. Check if job already exists in DB (by source + ID)
           b. Save new jobs to DB
           c. Create JobMatch row linking job → user + resume
        4. Return JobSearchOutput with all jobs + matches

    Search query optimization:
        The agent intelligently combines the user's target role
        with their top skills to build a more targeted query:

        User wants: "Backend Engineer"
        Top skills: ["Python", "FastAPI", "PostgreSQL"]
        → Query: "Backend Engineer Python FastAPI"

        This produces more relevant results than just the role title.

    Deduplication:
        Before saving, each job is checked against the DB using
        (source, external_job_id). If it exists:
        - The existing Job row is reused
        - A new JobMatch is still created for this user
        This prevents duplicate job listings while allowing
        multiple users to match against the same job.
    """

    # ----------------------------------------------------------
    # Agent Identity
    # ----------------------------------------------------------
    agent_name      = "JobSearchAgent"
    agent_type      = AgentType.JOB_SEARCH
    step_number     = 2
    total_steps     = 8
    timeout_seconds = 120    # Scraping can be slow — allow 2 minutes
    max_retries     = 2

    def __init__(self, db: AsyncSession) -> None:
        """
        Args:
            db: Async SQLAlchemy session for reading/writing jobs
                and job matches to the database.
        """
        super().__init__()
        self.db = db
        self.scraper: ScraperService = scraper_service

    # ----------------------------------------------------------
    # Input Validation
    # ----------------------------------------------------------

    def validate_input(self, input_data: JobSearchInput) -> None:
        """
        Validates the JobSearchInput before searching begins.

        Checks:
            - input_data is a JobSearchInput instance
            - user_id and resume_id are valid UUIDs
            - parsed_resume is a ResumeParseResult
            - search_request has a non-empty role

        Raises:
            AgentInputError: On any validation failure.
        """
        if not isinstance(input_data, JobSearchInput):
            raise AgentInputError(
                f"JobSearchAgent expects JobSearchInput, "
                f"got {type(input_data).__name__}."
            )

        if not input_data.user_id:
            raise AgentInputError("user_id is required.")

        if not input_data.resume_id:
            raise AgentInputError("resume_id is required.")

        if not isinstance(input_data.parsed_resume, ResumeParseResult):
            raise AgentInputError(
                "parsed_resume must be a ResumeParseResult. "
                "Ensure Agent 1 ran successfully before Agent 2."
            )

        if not input_data.search_request.role:
            raise AgentInputError(
                "search_request.role is required. "
                "The user must specify a target job title."
            )

    # ----------------------------------------------------------
    # Output Validation
    # ----------------------------------------------------------

    def validate_output(self, output: JobSearchOutput) -> None:
        """
        Validates Agent 2's output before passing to Agent 3.

        A warning (not error) is issued if no jobs were found —
        the pipeline can continue with 0 jobs (Agent 3 will
        simply have nothing to score).

        Args:
            output: JobSearchOutput from _execute().
        """
        if not isinstance(output, JobSearchOutput):
            raise AgentInputError(
                f"Expected JobSearchOutput, got {type(output).__name__}."
            )

        if output.total_jobs == 0:
            self._warn(
                "No jobs were found matching the search criteria. "
                "Try broadening the search role, location, or "
                "ensure RAPIDAPI_KEY is configured in .env"
            )

    # ----------------------------------------------------------
    # Core Execution
    # ----------------------------------------------------------

    async def _execute(
        self,
        input_data: JobSearchInput,
        **kwargs: Any,
    ) -> JobSearchOutput:
        """
        Searches for jobs and saves them to the database.

        Args:
            input_data: JobSearchInput with user preferences.
            **kwargs:   Unused — reserved for future use.

        Returns:
            JobSearchOutput: All found jobs with DB-persisted rows.

        Raises:
            AgentError: If scraping and DB operations both fail.
        """
        user_id       = input_data.user_id
        resume_id     = input_data.resume_id
        parsed_resume = input_data.parsed_resume
        search_req    = input_data.search_request

        # ── Step 1: Build optimized search query ───────────────
        search_params = self._build_search_params(
            search_request=search_req,
            parsed_resume=parsed_resume,
        )

        self.logger.info(
            f"Starting job search | "
            f"query='{search_params.to_query_string()}' | "
            f"location='{search_params.location}' | "
            f"max_results={search_params.max_results}"
        )

        self._set_metadata("search_query", search_params.to_query_string())
        self._set_metadata("search_location", search_params.location)
        self._set_metadata("max_results", search_params.max_results)

        # ── Step 2: Fetch jobs from API ─────────────────────────
        scraper_result = await self.scraper.search_jobs(search_params)

        if scraper_result.has_errors:
            for error in scraper_result.errors:
                self._warn(f"Scraper warning: {error}")

        self.logger.info(
            f"Scraper returned {scraper_result.jobs_count} jobs | "
            f"sources={scraper_result.sources_used} | "
            f"duration={scraper_result.duration_ms}ms"
        )

        self._set_metadata("scraper_jobs_found", scraper_result.jobs_count)
        self._set_metadata("sources_used", scraper_result.sources_used)
        self._set_metadata("scraper_duration_ms", scraper_result.duration_ms)

        # ── Step 3: Persist jobs + create job matches ───────────
        jobs, job_matches, new_count, existing_count = (
            await self._persist_jobs_and_matches(
                job_creates=scraper_result.jobs,
                user_id=user_id,
                resume_id=resume_id,
            )
        )

        self._set_metadata("new_jobs_created", new_count)
        self._set_metadata("existing_jobs_reused", existing_count)
        self._set_metadata("job_matches_created", len(job_matches))

        self.logger.info(
            f"Jobs persisted | "
            f"new={new_count} | "
            f"existing={existing_count} | "
            f"matches_created={len(job_matches)}"
        )

        return JobSearchOutput(
            jobs=jobs,
            job_matches=job_matches,
            total_found=scraper_result.total_found,
            sources_used=scraper_result.sources_used,
            new_jobs=new_count,
            existing_jobs=existing_count,
        )

    # ----------------------------------------------------------
    # Search Query Builder
    # ----------------------------------------------------------

    def _build_search_params(
        self,
        search_request: JobSearchRequest,
        parsed_resume: ResumeParseResult,
    ) -> SearchParams:
        """
        Builds an optimized SearchParams from user preferences
        and resume data.

        Query optimization strategy:
            Base:  User's target role (e.g. "Backend Engineer")
            + Top: 2-3 most relevant skills from resume
            Result: "Backend Engineer Python FastAPI"

        Skills are chosen by relevance to the role:
            - Programming languages first
            - Then frameworks matching the role type
            - Capped at 2-3 to avoid over-specificity

        Location handling:
            - Uses search_request.location if set
            - Falls back to parsed_resume.location
            - Falls back to None (global search)

        Args:
            search_request: User's search preferences.
            parsed_resume:  Parsed resume data from Agent 1.

        Returns:
            SearchParams: Optimized parameters for the scraper.
        """
        role = search_request.role

        # Build skill-augmented query
        top_skills = self._select_top_skills_for_query(
            role=role,
            resume_skills=parsed_resume.all_skills_flat,
            max_skills=2,
        )

        if top_skills:
            query = f"{role} {' '.join(top_skills)}"
        else:
            query = role

        # Determine location
        location = (
            search_request.location
            or parsed_resume.location
            or None
        )

        # Determine experience context
        experience_years = (
            search_request.experience_years
            or (
                int(parsed_resume.total_experience_years)
                if parsed_resume.total_experience_years
                else None
            )
        )

        # Determine country from location
        country = self._infer_country(location)

        self.logger.debug(
            f"Search params built | "
            f"query='{query}' | "
            f"location='{location}' | "
            f"country='{country}' | "
            f"experience={experience_years}yrs"
        )

        return SearchParams(
            query=query,
            location=location,
            experience_years=experience_years,
            work_mode=search_request.work_mode,
            job_type=search_request.job_type,
            max_results=search_request.max_results or 20,
            date_posted="month",
            country=country,
        )

    def _select_top_skills_for_query(
        self,
        role: str,
        resume_skills: List[str],
        max_skills: int = 2,
    ) -> List[str]:
        """
        Selects the most relevant skills to append to the search query.

        Strategy:
            1. Identify the role category (backend, frontend, data, etc.)
            2. Score each resume skill by relevance to that category
            3. Return top N skills by relevance score

        This improves search relevance significantly:
            Without: "Backend Engineer" → too broad
            With:    "Backend Engineer Python FastAPI" → targeted

        Args:
            role:          Target job role from search request.
            resume_skills: All skills from parsed resume.
            max_skills:    Maximum skills to include in query.

        Returns:
            List[str]: Top skills to append to query.
        """
        if not resume_skills:
            return []

        role_lower = role.lower()

        # Define skill priorities by role category
        priority_skills: Dict[str, List[str]] = {
            "backend": [
                "python", "java", "golang", "node.js", "fastapi",
                "django", "spring", "express", "postgresql", "redis",
            ],
            "frontend": [
                "react", "angular", "vue", "typescript", "javascript",
                "next.js", "tailwind", "html", "css",
            ],
            "fullstack": [
                "react", "python", "node.js", "typescript",
                "postgresql", "mongodb", "docker",
            ],
            "data": [
                "python", "sql", "pandas", "spark", "tensorflow",
                "pytorch", "scikit-learn", "tableau", "airflow",
            ],
            "devops": [
                "kubernetes", "docker", "terraform", "aws",
                "jenkins", "ansible", "linux", "ci/cd",
            ],
            "mobile": [
                "swift", "kotlin", "react native", "flutter",
                "android", "ios",
            ],
            "ml": [
                "python", "tensorflow", "pytorch", "scikit-learn",
                "mlflow", "pandas", "numpy", "transformers",
            ],
        }

        # Detect role category
        category = "backend"   # Default
        for cat in priority_skills:
            if cat in role_lower:
                category = cat
                break

        # Check for specific keywords
        if any(w in role_lower for w in ["machine learning", "ml", "ai", "nlp"]):
            category = "ml"
        elif any(w in role_lower for w in ["data scientist", "data analyst", "analytics"]):
            category = "data"
        elif any(w in role_lower for w in ["devops", "sre", "infrastructure", "platform"]):
            category = "devops"
        elif any(w in role_lower for w in ["mobile", "ios", "android"]):
            category = "mobile"
        elif any(w in role_lower for w in ["frontend", "front-end", "ui"]):
            category = "frontend"
        elif any(w in role_lower for w in ["fullstack", "full-stack", "full stack"]):
            category = "fullstack"

        priority = priority_skills.get(category, priority_skills["backend"])

        # Score resume skills by position in priority list
        scored: List[Tuple[str, int]] = []
        resume_lower = {s.lower(): s for s in resume_skills}

        for skill_lower, skill_original in resume_lower.items():
            if skill_lower in priority:
                score = len(priority) - priority.index(skill_lower)
                scored.append((skill_original, score))

        # Sort by score descending
        scored.sort(key=lambda x: x[1], reverse=True)

        return [skill for skill, _ in scored[:max_skills]]

    def _infer_country(
        self,
        location: Optional[str],
    ) -> str:
        """
        Infers the country code from a location string.

        Used to filter job search results by country.
        Defaults to "in" (India) if no location is given.

        Args:
            location: Location string e.g. "Bangalore" or "Remote".

        Returns:
            str: Two-letter country code for JSearch API.
        """
        if not location:
            return "in"   # Default: India

        location_lower = location.lower()

        country_map = {
            # India
            "india": "in", "bangalore": "in", "bengaluru": "in",
            "mumbai": "in", "delhi": "in", "hyderabad": "in",
            "pune": "in", "chennai": "in", "kolkata": "in",
            "noida": "in", "gurgaon": "in", "gurugram": "in",

            # US
            "usa": "us", "united states": "us", "new york": "us",
            "san francisco": "us", "seattle": "us", "austin": "us",
            "chicago": "us", "boston": "us", "los angeles": "us",

            # UK
            "uk": "gb", "united kingdom": "gb", "london": "gb",
            "manchester": "gb", "birmingham": "gb",

            # Others
            "canada": "ca", "toronto": "ca", "vancouver": "ca",
            "australia": "au", "sydney": "au", "melbourne": "au",
            "germany": "de", "berlin": "de", "munich": "de",
            "singapore": "sg", "dubai": "ae", "uae": "ae",
        }

        for keyword, code in country_map.items():
            if keyword in location_lower:
                return code

        return "in"   # Default to India

    # ----------------------------------------------------------
    # Database Operations
    # ----------------------------------------------------------

    async def _persist_jobs_and_matches(
        self,
        job_creates: List[JobCreate],
        user_id: uuid.UUID,
        resume_id: uuid.UUID,
    ) -> Tuple[List[Job], List[JobMatch], int, int]:
        """
        Saves jobs to DB and creates JobMatch rows.

        For each JobCreate:
            1. Check if (source, external_job_id) already exists in DB
            2. If new: INSERT into jobs table
            3. If existing: load the existing Job row
            4. Check if a JobMatch already exists for (user, job, resume)
            5. If not: INSERT into job_matches table

        All operations are batched in a single transaction.

        Args:
            job_creates:  List of normalized JobCreate objects.
            user_id:      UUID of the user to link matches to.
            resume_id:    UUID of the resume to link matches to.

        Returns:
            Tuple of:
                List[Job]:      All Job ORM objects (new + existing).
                List[JobMatch]: All created JobMatch ORM objects.
                int:            Count of newly created Job rows.
                int:            Count of reused existing Job rows.

        Raises:
            AgentError: If the database transaction fails.
        """
        all_jobs:    List[Job]      = []
        all_matches: List[JobMatch] = []
        new_count      = 0
        existing_count = 0

        try:
            for job_create in job_creates:

                # ── Get or create Job row ───────────────────────
                job, is_new = await self._get_or_create_job(job_create)

                if is_new:
                    new_count += 1
                else:
                    existing_count += 1

                all_jobs.append(job)

                # ── Get or create JobMatch row ──────────────────
                job_match = await self._get_or_create_job_match(
                    job=job,
                    user_id=user_id,
                    resume_id=resume_id,
                )

                if job_match:
                    all_matches.append(job_match)

            # Commit all inserts in one transaction
            await self.db.commit()

            # Refresh all objects to load DB-generated fields
            for job in all_jobs:
                await self.db.refresh(job)
            for match in all_matches:
                await self.db.refresh(match)

            return all_jobs, all_matches, new_count, existing_count

        except Exception as e:
            await self.db.rollback()
            raise AgentError(
                f"Failed to persist jobs to database: {e}. "
                f"Attempted to save {len(job_creates)} jobs."
            )

    async def _get_or_create_job(
        self,
        job_create: JobCreate,
    ) -> Tuple[Job, bool]:
        """
        Fetches an existing Job from DB or creates a new one.

        Uniqueness is determined by (source, external_job_id).
        If external_job_id is None, falls back to (company, title).

        Args:
            job_create: Normalized job data to save.

        Returns:
            Tuple[Job, bool]:
                Job ORM object and True if newly created.
        """
        # Try to find existing by (source, external_job_id)
        if job_create.source and job_create.external_job_id:
            existing = await self.db.execute(
                select(Job).where(
                    Job.source == job_create.source,
                    Job.external_job_id == job_create.external_job_id,
                )
            )
            existing_job = existing.scalar_one_or_none()

            if existing_job:
                self.logger.debug(
                    f"Reusing existing job: "
                    f"'{existing_job.title}' @ {existing_job.company_name}"
                )
                return existing_job, False

        # Fallback: find by (company_name, title)
        if not job_create.external_job_id:
            existing = await self.db.execute(
                select(Job).where(
                    Job.company_name == job_create.company_name,
                    Job.title == job_create.title,
                )
            )
            existing_job = existing.scalar_one_or_none()

            if existing_job:
                return existing_job, False

        # Create new Job row
        new_job = Job(
            source=job_create.source,
            external_job_id=job_create.external_job_id,
            job_url=job_create.job_url,
            title=job_create.title,
            company_name=job_create.company_name,
            company_logo_url=job_create.company_logo_url,
            company_website=job_create.company_website,
            industry=job_create.industry,
            location=job_create.location,
            country=job_create.country,
            work_mode=job_create.work_mode,
            job_description=job_create.job_description,
            responsibilities=job_create.responsibilities,
            requirements=job_create.requirements,
            required_skills=job_create.required_skills,
            preferred_skills=job_create.preferred_skills,
            keywords=job_create.keywords,
            experience_min_years=job_create.experience_min_years,
            experience_max_years=job_create.experience_max_years,
            experience_level=job_create.experience_level,
            education_required=job_create.education_required,
            salary_min=job_create.salary_min,
            salary_max=job_create.salary_max,
            salary_currency=job_create.salary_currency,
            salary_period=job_create.salary_period,
            salary_display=job_create.salary_display,
            job_type=job_create.job_type,
            posted_at=job_create.posted_at,
            is_active=True,
        )

        self.db.add(new_job)

        # Flush to get the generated ID without committing
        await self.db.flush()

        self.logger.debug(
            f"Created new job: "
            f"'{new_job.title}' @ {new_job.company_name} "
            f"[{new_job.source}]"
        )

        return new_job, True

    async def _get_or_create_job_match(
        self,
        job: Job,
        user_id: uuid.UUID,
        resume_id: uuid.UUID,
    ) -> Optional[JobMatch]:
        """
        Fetches an existing JobMatch or creates a new one.

        A JobMatch uniquely identifies the combination of
        (user_id, job_id, resume_id). If the user runs the
        pipeline twice with the same resume + job, the existing
        match is returned rather than creating a duplicate.

        The new JobMatch is created with:
            overall_score = 0.0 (Agent 3 will fill this in)
            is_recommended = False (Agent 3 will update this)

        Args:
            job:       Job ORM object to link.
            user_id:   User who is searching.
            resume_id: Resume used for matching.

        Returns:
            JobMatch ORM object (new or existing), or None on error.
        """
        # Check for existing match
        existing = await self.db.execute(
            select(JobMatch).where(
                JobMatch.user_id  == user_id,
                JobMatch.job_id   == job.id,
                JobMatch.resume_id == resume_id,
            )
        )
        existing_match = existing.scalar_one_or_none()

        if existing_match:
            self.logger.debug(
                f"Reusing existing JobMatch for job: {job.title}"
            )
            return existing_match

        # Create new JobMatch — scores filled by Agent 3
        new_match = JobMatch(
            user_id=user_id,
            job_id=job.id,
            resume_id=resume_id,
            overall_score=0.0,
            is_recommended=False,
            ats_generated=False,
            cover_letter_generated=False,
            interview_prep_generated=False,
        )

        self.db.add(new_match)
        await self.db.flush()

        return new_match

    # ----------------------------------------------------------
    # Post-Run Hooks
    # ----------------------------------------------------------

    async def on_success(self, result: AgentResult) -> None:
        """
        Logs a summary after successful job search.

        Args:
            result: Successful AgentResult from BaseAgent.run().
        """
        if result.output:
            output: JobSearchOutput = result.output
            self.logger.info(
                f"Job search summary | "
                f"total_jobs={output.total_jobs} | "
                f"new={output.new_jobs} | "
                f"existing={output.existing_jobs} | "
                f"matches={len(output.job_matches)} | "
                f"sources={output.sources_used}"
            )

    async def on_failure(self, result: AgentResult) -> None:
        """
        Handles cleanup if job search fails.

        Job search failure is non-fatal for the pipeline —
        the orchestrator can proceed with 0 jobs or
        use cached results from a previous run.

        Args:
            result: Failed AgentResult from BaseAgent.run().
        """
        self.logger.warning(
            f"Job search failed — pipeline will continue with 0 jobs. "
            f"Error: {result.error}"
        )

    # ----------------------------------------------------------
    # Representation
    # ----------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"<JobSearchAgent "
            f"step={self.step_number}/{self.total_steps}>"
        )