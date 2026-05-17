# ============================================================
# File: backend/agents/job_matching_agent.py
# Purpose: Agent 3 — Scores and ranks job matches against
#          the candidate's resume using semantic similarity,
#          keyword overlap, and skill matching.
#
# Pipeline Position: Step 3 of 8
#
# Input:
#   JobMatchingInput dataclass containing:
#     - parsed_resume:  ResumeParseResult from Agent 1
#     - job_matches:    List[JobMatch] stubs from Agent 2
#     - jobs:           List[Job] ORM objects from Agent 2
#
# Output:
#   JobMatchingOutput dataclass containing:
#     - ranked_matches: List[JobMatch] sorted by score desc
#     - recommended:    List[JobMatch] above MIN_MATCH_SCORE
#     - score_summary:  Score statistics for the UI
#
# Side Effects:
#   Updates every JobMatch row in the database:
#     - overall_score
#     - skill_match_score
#     - keyword_match_score
#     - semantic_score
#     - matched_skills
#     - missing_skills
#     - bonus_skills
#     - rank
#     - is_recommended
#
# Used by:
#   - backend/orchestrator/pipeline.py → called as Step 3
#
# Downstream agents that consume its output:
#   - Agent 4 (SkillGapAgent)     → reads missing_skills
#   - Agent 5 (ATSOptimizerAgent) → reads top job matches
#   - Agent 6 (CoverLetterAgent)  → reads top job matches
#   - Agent 7 (InterviewPrepAgent)→ reads top job matches
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
from backend.config import settings
from backend.models.job import Job, JobMatch
from backend.models.resume import ResumeParseResult
from backend.services.embedding_service import (
    BatchMatchResult,
    SimilarityResult,
)
from backend.services.llm_service import AgentType


# ============================================================
# Input / Output Data Classes
# ============================================================

@dataclass
class JobMatchingInput:
    """
    Input data for Agent 3 (Job Matching Agent).

    Constructed by the orchestrator from Agent 2's output.

    Attributes:
        parsed_resume: Structured resume data from Agent 1.
                       Provides raw_text, skills, experience level.
        job_matches:   JobMatch stubs from Agent 2 with score=0.0.
                       Agent 3 fills in the real scores.
        jobs:          Corresponding Job ORM objects.
                       Must be same length and order as job_matches.
        resume_raw_text: Full plain text of the resume for
                         semantic embedding. Taken from Resume.raw_text.
    """
    parsed_resume:   ResumeParseResult
    job_matches:     List[JobMatch]
    jobs:            List[Job]
    resume_raw_text: str


@dataclass
class ScoreSummary:
    """
    Statistical summary of all match scores.
    Displayed in the UI as a score overview card.

    Attributes:
        total_jobs:       Total jobs scored.
        recommended_count:Jobs above MIN_MATCH_SCORE threshold.
        avg_score:        Mean overall score across all jobs.
        highest_score:    Best match score found.
        lowest_score:     Worst match score found.
        score_distribution: Count of jobs per score band.
    """
    total_jobs:        int
    recommended_count: int
    avg_score:         float
    highest_score:     float
    lowest_score:      float
    score_distribution: Dict[str, int]   # "excellent|good|fair|low" → count

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_jobs":         self.total_jobs,
            "recommended_count":  self.recommended_count,
            "avg_score":          round(self.avg_score * 100, 1),
            "highest_score":      round(self.highest_score * 100, 1),
            "lowest_score":       round(self.lowest_score * 100, 1),
            "score_distribution": self.score_distribution,
        }


@dataclass
class JobMatchingOutput:
    """
    Output data from Agent 3 (Job Matching Agent).

    Attributes:
        ranked_matches:  All JobMatch objects sorted by rank (1=best).
        recommended:     Subset of ranked_matches above threshold.
        score_summary:   Statistical overview of all scores.
        batch_duration_ms: How long batch scoring took.
    """
    ranked_matches:    List[JobMatch]
    recommended:       List[JobMatch]
    score_summary:     ScoreSummary
    batch_duration_ms: float

    @property
    def top_match(self) -> Optional[JobMatch]:
        """Returns the highest-ranked job match."""
        return self.ranked_matches[0] if self.ranked_matches else None

    @property
    def has_recommendations(self) -> bool:
        """True if any jobs cleared the minimum score threshold."""
        return len(self.recommended) > 0


# ============================================================
# Job Matching Agent
# ============================================================

class JobMatchingAgent(BaseAgent):
    """
    Agent 3 — Job Matching Agent.

    Scores every job from Agent 2 against the candidate's
    resume using three complementary similarity methods,
    then ranks and filters the results.

    Three scoring dimensions:
        1. Semantic (50% weight):
           sentence-transformers embeddings capture meaning.
           "Python developer" ≈ "software engineer using Python"
           Even without identical words.

        2. Keyword (25% weight):
           TF-IDF measures exact term overlap.
           ATS systems care about exact keyword matches.
           "FastAPI" must appear in both resume and JD.

        3. Skill (25% weight):
           Direct skill list comparison with alias expansion.
           Handles "JS" = "JavaScript", "k8s" = "Kubernetes".

    Combined score drives:
        - Job ranking (rank=1 is best match)
        - Recommendation flag (score >= MIN_MATCH_SCORE)
        - Skill gap identification (missing_skills)

    Batch processing:
        All jobs are scored in ONE batch call to embedding_service.
        The resume embedding is computed once and reused for all
        job comparisons — significantly faster than sequential calls.
    """

    # ----------------------------------------------------------
    # Agent Identity
    # ----------------------------------------------------------
    agent_name      = "JobMatchingAgent"
    agent_type      = AgentType.JOB_MATCHING
    step_number     = 3
    total_steps     = 8
    timeout_seconds = 180    # Batch embedding can take time for 50 jobs
    max_retries     = 2

    def __init__(self, db: AsyncSession) -> None:
        """
        Args:
            db: Async SQLAlchemy session for updating JobMatch scores.
        """
        super().__init__()
        self.db = db
        self._min_score = settings.min_match_score

    # ----------------------------------------------------------
    # Input Validation
    # ----------------------------------------------------------

    def validate_input(self, input_data: JobMatchingInput) -> None:
        """
        Validates input before scoring begins.

        Raises:
            AgentInputError: If input is missing or malformed.
        """
        if not isinstance(input_data, JobMatchingInput):
            raise AgentInputError(
                f"JobMatchingAgent expects JobMatchingInput, "
                f"got {type(input_data).__name__}."
            )

        if not isinstance(input_data.parsed_resume, ResumeParseResult):
            raise AgentInputError(
                "parsed_resume must be a ResumeParseResult. "
                "Ensure Agent 1 ran successfully."
            )

        if not input_data.resume_raw_text:
            raise AgentInputError(
                "resume_raw_text is empty. "
                "Cannot compute semantic similarity without resume text."
            )

        if len(input_data.job_matches) != len(input_data.jobs):
            raise AgentInputError(
                f"job_matches ({len(input_data.job_matches)}) and "
                f"jobs ({len(input_data.jobs)}) must be the same length."
            )

        if not input_data.job_matches:
            self._warn(
                "No job matches provided — "
                "Agent 2 may have found no jobs. "
                "Scoring will be skipped."
            )

    # ----------------------------------------------------------
    # Core Execution
    # ----------------------------------------------------------

    async def _execute(
        self,
        input_data: JobMatchingInput,
        **kwargs: Any,
    ) -> JobMatchingOutput:
        """
        Scores, ranks, and persists match results for all jobs.

        Args:
            input_data: JobMatchingInput with resume + job stubs.
            **kwargs:   Unused — reserved for future use.

        Returns:
            JobMatchingOutput: Ranked matches with updated scores.

        Raises:
            AgentError: If batch scoring or DB update fails.
        """
        parsed_resume   = input_data.parsed_resume
        job_matches     = input_data.job_matches
        jobs            = input_data.jobs
        resume_text     = input_data.resume_raw_text

        # Early return if no jobs to score
        if not jobs:
            self.logger.info("No jobs to score — skipping matching step.")
            return JobMatchingOutput(
                ranked_matches=[],
                recommended=[],
                score_summary=ScoreSummary(
                    total_jobs=0,
                    recommended_count=0,
                    avg_score=0.0,
                    highest_score=0.0,
                    lowest_score=0.0,
                    score_distribution={
                        "excellent": 0, "good": 0,
                        "fair": 0, "low": 0,
                    },
                ),
                batch_duration_ms=0.0,
            )

        self.logger.info(
            f"Starting batch job matching | "
            f"jobs={len(jobs)} | "
            f"resume_skills={len(parsed_resume.all_skills_flat)} | "
            f"threshold={self._min_score}"
        )

        # ── Step 1: Prepare batch inputs ────────────────────────
        job_descriptions, req_skills_list, pref_skills_list = (
            self._prepare_batch_inputs(jobs)
        )

        self._set_metadata("jobs_to_score", len(jobs))
        self._set_metadata("min_score_threshold", self._min_score)

        # ── Step 2: Batch similarity scoring ───────────────────
        self.logger.info(
            "Running batch semantic + keyword + skill scoring..."
        )

        try:
            batch_result: BatchMatchResult = (
                await self.embeddings.batch_match(
                    resume_text=resume_text,
                    job_descriptions=job_descriptions,
                    resume_skills=parsed_resume.all_skills_flat,
                    jobs_required_skills=req_skills_list,
                    jobs_preferred_skills=pref_skills_list,
                )
            )
        except Exception as e:
            raise AgentError(
                f"Batch embedding scoring failed: {e}. "
                f"Ensure sentence-transformers model is loaded."
            )

        self._set_metadata(
            "batch_duration_ms",
            round(batch_result.total_duration_ms, 2),
        )
        self._set_metadata(
            "cache_stats",
            self.embeddings.cache_stats,
        )

        self.logger.info(
            f"Batch scoring complete | "
            f"duration={batch_result.total_duration_ms:.0f}ms | "
            f"avg={batch_result.total_duration_ms/len(jobs):.1f}ms/job"
        )

        # ── Step 3: Map scores back to JobMatch objects ─────────
        scored_matches = self._apply_scores_to_matches(
            job_matches=job_matches,
            jobs=jobs,
            batch_result=batch_result,
        )

        # ── Step 4: Rank all matches ────────────────────────────
        ranked_matches = self._rank_matches(scored_matches)

        # ── Step 5: Filter recommended jobs ────────────────────
        recommended = [
            m for m in ranked_matches
            if m.overall_score >= self._min_score
        ]

        self.logger.info(
            f"Ranking complete | "
            f"total={len(ranked_matches)} | "
            f"recommended={len(recommended)} | "
            f"threshold={self._min_score}"
        )

        # ── Step 6: Compute score summary ──────────────────────
        score_summary = self._compute_score_summary(
            ranked_matches=ranked_matches,
            recommended_count=len(recommended),
        )

        # ── Step 7: Persist scores to database ─────────────────
        await self._persist_scores(ranked_matches)

        self._set_metadata(
            "recommended_count",
            len(recommended),
        )
        self._set_metadata(
            "avg_score_pct",
            score_summary.avg_score * 100,
        )
        self._set_metadata(
            "top_score_pct",
            score_summary.highest_score * 100,
        )

        return JobMatchingOutput(
            ranked_matches=ranked_matches,
            recommended=recommended,
            score_summary=score_summary,
            batch_duration_ms=batch_result.total_duration_ms,
        )

    # ----------------------------------------------------------
    # Batch Input Preparation
    # ----------------------------------------------------------

    def _prepare_batch_inputs(
        self,
        jobs: List[Job],
    ) -> Tuple[List[str], List[List[str]], List[List[str]]]:
        """
        Extracts parallel lists of inputs for batch_match().

        batch_match() expects:
            - job_descriptions[i]    → text for job i
            - req_skills_list[i]     → required skills for job i
            - pref_skills_list[i]    → preferred skills for job i

        All three lists are the same length and same order as jobs[].

        Also handles missing job descriptions by building a
        synthetic description from the structured fields —
        some jobs have rich highlights but no long-form description.

        Args:
            jobs: List of Job ORM objects.

        Returns:
            Tuple of three parallel lists:
                job_descriptions, req_skills_list, pref_skills_list
        """
        job_descriptions: List[str]       = []
        req_skills_list:  List[List[str]] = []
        pref_skills_list: List[List[str]] = []

        missing_jd_count = 0

        for job in jobs:
            # Build job description text
            jd_text = self._build_job_text(job)
            if not jd_text:
                missing_jd_count += 1
            job_descriptions.append(jd_text or "")

            # Required skills
            req_skills = job.required_skills or []
            req_skills_list.append(req_skills)

            # Preferred skills
            pref_skills = job.preferred_skills or []
            pref_skills_list.append(pref_skills)

        if missing_jd_count > 0:
            self._warn(
                f"{missing_jd_count} jobs have no description. "
                f"Scoring will rely on skill matching only for these."
            )

        return job_descriptions, req_skills_list, pref_skills_list

    def _build_job_text(self, job: Job) -> str:
        """
        Builds a rich text representation of a job for embedding.

        Combines multiple fields into one text block to give the
        embedding model the best possible context. Jobs often have
        structured fields (title, skills) even when the full
        description is missing.

        Strategy:
            1. Use job_description if available (best signal)
            2. Supplement with responsibilities + requirements
            3. Append required skills as text
            4. Include title + company for context

        Args:
            job: Job ORM object.

        Returns:
            str: Combined text representation of the job.
        """
        parts: List[str] = []

        # Title + company as context anchor
        parts.append(f"Job Title: {job.title}")
        parts.append(f"Company: {job.company_name}")

        if job.experience_level:
            parts.append(f"Level: {job.experience_level}")

        # Main description — truncated to prevent context overflow
        if job.job_description:
            jd_truncated = job.job_description[:6000]
            parts.append(f"Description:\n{jd_truncated}")

        # Responsibilities
        if job.responsibilities:
            resp_text = "\n".join(
                f"- {r}" for r in job.responsibilities[:10]
            )
            parts.append(f"Responsibilities:\n{resp_text}")

        # Requirements
        if job.requirements:
            req_text = "\n".join(
                f"- {r}" for r in job.requirements[:10]
            )
            parts.append(f"Requirements:\n{req_text}")

        # Required skills as plain text (boosts keyword matching)
        if job.required_skills:
            skills_text = ", ".join(job.required_skills[:20])
            parts.append(f"Required Skills: {skills_text}")

        if job.preferred_skills:
            pref_text = ", ".join(job.preferred_skills[:10])
            parts.append(f"Preferred Skills: {pref_text}")

        return "\n\n".join(filter(None, parts))

    # ----------------------------------------------------------
    # Score Application
    # ----------------------------------------------------------

    def _apply_scores_to_matches(
        self,
        job_matches: List[JobMatch],
        jobs: List[Job],
        batch_result: BatchMatchResult,
    ) -> List[JobMatch]:
        """
        Applies SimilarityResult scores to JobMatch ORM objects.

        Maps each batch result back to its corresponding JobMatch
        using the same index ordering established in _prepare_batch_inputs.

        Also uses Claude to generate a brief analysis for the top
        matches — explaining WHY the match is good or bad.

        Args:
            job_matches:  JobMatch stubs with score=0.0.
            jobs:         Corresponding Job objects (same order).
            batch_result: BatchMatchResult from embedding_service.

        Returns:
            List[JobMatch]: Updated JobMatch objects with all scores.
        """
        results = batch_result.results

        for i, (job_match, job) in enumerate(zip(job_matches, jobs)):
            if i >= len(results):
                self.logger.warning(
                    f"No score result for job index {i}: {job.title}"
                )
                continue

            sim: SimilarityResult = results[i]

            # Apply all scores to JobMatch
            job_match.overall_score       = sim.overall_score
            job_match.skill_match_score   = sim.skill_score
            job_match.keyword_match_score = sim.keyword_score
            job_match.semantic_score      = sim.semantic_score

            # Apply skill breakdown
            job_match.matched_skills = sim.matched_skills
            job_match.missing_skills = sim.missing_skills
            job_match.bonus_skills   = sim.bonus_skills

            # Store full similarity analysis as JSON
            job_match.skill_gap_analysis = sim.to_dict()

            # Set recommendation flag
            job_match.is_recommended = (
                sim.overall_score >= self._min_score
            )

            self.logger.debug(
                f"Scored: '{job.title}' @ {job.company_name} | "
                f"overall={sim.overall_score_percent}% | "
                f"semantic={round(sim.semantic_score*100,1)}% | "
                f"keyword={round(sim.keyword_score*100,1)}% | "
                f"skill={round(sim.skill_score*100,1)}% | "
                f"matched={len(sim.matched_skills)} | "
                f"missing={len(sim.missing_skills)}"
            )

        return job_matches

    # ----------------------------------------------------------
    # Ranking
    # ----------------------------------------------------------

    def _rank_matches(
        self,
        job_matches: List[JobMatch],
    ) -> List[JobMatch]:
        """
        Sorts JobMatch objects by overall_score descending
        and assigns rank values (1 = best match).

        Tiebreaker order (when scores are equal):
            1. More matched skills → higher rank
            2. Fewer missing skills → higher rank
            3. More recent posting → higher rank (not yet tracked)

        Args:
            job_matches: Unranked list of scored JobMatch objects.

        Returns:
            List[JobMatch]: Same objects with .rank set, sorted best-first.
        """
        # Sort with tiebreakers
        sorted_matches = sorted(
            job_matches,
            key=lambda m: (
                m.overall_score,
                len(m.matched_skills or []),
                -len(m.missing_skills or []),
            ),
            reverse=True,
        )

        # Assign ranks (1-based)
        for rank, match in enumerate(sorted_matches, start=1):
            match.rank = rank

        return sorted_matches

    # ----------------------------------------------------------
    # Score Summary
    # ----------------------------------------------------------

    def _compute_score_summary(
        self,
        ranked_matches: List[JobMatch],
        recommended_count: int,
    ) -> ScoreSummary:
        """
        Computes statistical summary of all match scores.

        Score bands:
            Excellent: >= 80%
            Good:      >= 60% and < 80%
            Fair:      >= 40% and < 60%
            Low:       < 40%

        Args:
            ranked_matches:    All matches sorted by rank.
            recommended_count: Count above MIN_MATCH_SCORE.

        Returns:
            ScoreSummary: Statistics for the UI dashboard.
        """
        if not ranked_matches:
            return ScoreSummary(
                total_jobs=0,
                recommended_count=0,
                avg_score=0.0,
                highest_score=0.0,
                lowest_score=0.0,
                score_distribution={
                    "excellent": 0, "good": 0,
                    "fair": 0,      "low": 0,
                },
            )

        scores = [m.overall_score for m in ranked_matches]

        # Score band distribution
        distribution = {
            "excellent": sum(1 for s in scores if s >= 0.80),
            "good":      sum(1 for s in scores if 0.60 <= s < 0.80),
            "fair":      sum(1 for s in scores if 0.40 <= s < 0.60),
            "low":       sum(1 for s in scores if s < 0.40),
        }

        return ScoreSummary(
            total_jobs=len(ranked_matches),
            recommended_count=recommended_count,
            avg_score=round(sum(scores) / len(scores), 4),
            highest_score=round(max(scores), 4),
            lowest_score=round(min(scores), 4),
            score_distribution=distribution,
        )

    # ----------------------------------------------------------
    # Database Persistence
    # ----------------------------------------------------------

    async def _persist_scores(
        self,
        ranked_matches: List[JobMatch],
    ) -> None:
        """
        Saves all updated JobMatch scores to the database.

        Uses a single transaction for all updates — either all
        scores are saved or none are (atomic operation).

        Updates per JobMatch:
            - overall_score
            - skill_match_score
            - keyword_match_score
            - semantic_score
            - matched_skills
            - missing_skills
            - bonus_skills
            - skill_gap_analysis
            - rank
            - is_recommended

        Args:
            ranked_matches: All JobMatch objects with scores applied.

        Raises:
            AgentError: If the database transaction fails.
        """
        self.logger.info(
            f"Persisting {len(ranked_matches)} match scores to database..."
        )

        try:
            # Add all modified objects to the session
            for match in ranked_matches:
                # SQLAlchemy tracks changes automatically for
                # objects already in the session — just merge
                # to ensure they're attached to this session
                await self.db.merge(match)

            await self.db.commit()

            self.logger.info(
                f"Match scores persisted | "
                f"count={len(ranked_matches)}"
            )

        except Exception as e:
            await self.db.rollback()
            raise AgentError(
                f"Failed to save match scores to database: {e}. "
                f"Scoring was successful but results were not persisted."
            )

    # ----------------------------------------------------------
    # Claude Enhancement (optional — enriches top matches)
    # ----------------------------------------------------------

    async def _enrich_top_matches_with_claude(
        self,
        top_matches: List[JobMatch],
        jobs: List[Job],
        parsed_resume: ResumeParseResult,
    ) -> None:
        """
        Uses Claude to generate a brief match analysis for the
        top N matches (default top 5).

        This is OPTIONAL — called only if the pipeline has budget
        for extra Claude API calls. Adds a human-readable
        explanation of why each job is a good or poor match.

        The analysis is stored in JobMatch.skill_gap_analysis
        under the key "claude_analysis".

        Args:
            top_matches:    Top ranked JobMatch objects.
            jobs:           Corresponding Job objects.
            parsed_resume:  Resume data for context.
        """
        # Build a lookup of job_id → Job
        job_lookup: Dict[uuid.UUID, Job] = {
            j.id: j for j in jobs
        }

        for match in top_matches[:5]:   # Only top 5
            job = job_lookup.get(match.job_id)
            if not job:
                continue

            try:
                prompt = self._build_analysis_prompt(
                    match=match,
                    job=job,
                    parsed_resume=parsed_resume,
                )

                analysis = await self.llm.complete_json(
                    agent_type=self.agent_type,
                    user_message=prompt,
                    usage_tracker=self.tracker,
                    temperature=0.3,
                )

                # Merge into existing skill_gap_analysis dict
                if match.skill_gap_analysis and analysis:
                    match.skill_gap_analysis["claude_analysis"] = analysis

            except Exception as e:
                self.logger.warning(
                    f"Claude enrichment failed for job "
                    f"'{job.title}': {e} — skipping."
                )

    def _build_analysis_prompt(
        self,
        match: JobMatch,
        job: Job,
        parsed_resume: ResumeParseResult,
    ) -> str:
        """
        Builds a concise analysis prompt for Claude.

        Asks Claude to explain the match in 2-3 sentences and
        provide 3 specific tips to improve the application.

        Args:
            match:         Scored JobMatch with skill breakdown.
            job:           The job being analyzed.
            parsed_resume: Resume for candidate context.

        Returns:
            str: Prompt for Claude.
        """
        matched  = ", ".join((match.matched_skills or [])[:8])
        missing  = ", ".join((match.missing_skills or [])[:8])
        top_role = parsed_resume.latest_role or "professional"
        exp_yrs  = parsed_resume.total_experience_years or 0

        return f"""
Analyze this job match and provide actionable insights.

CANDIDATE:
  - Current Role: {top_role}
  - Experience: {exp_yrs} years
  - Matched Skills: {matched or "None identified"}
  - Missing Skills: {missing or "None identified"}

JOB:
  - Title: {job.title}
  - Company: {job.company_name}
  - Location: {job.location or "Not specified"}
  - Level: {job.experience_level or "Not specified"}

MATCH SCORE: {match.overall_score * 100:.1f}%

Return JSON with this exact structure:
{{
  "match_summary": "2-sentence explanation of why this is a good/poor match",
  "biggest_strength": "The single strongest alignment between candidate and role",
  "biggest_gap": "The single most important missing qualification",
  "tips": [
    "Specific tip 1 to improve chances",
    "Specific tip 2 to improve chances",
    "Specific tip 3 to improve chances"
  ],
  "apply_recommendation": "strong_yes | yes | maybe | no"
}}
""".strip()

    # ----------------------------------------------------------
    # Post-Run Hooks
    # ----------------------------------------------------------

    async def on_success(self, result: AgentResult) -> None:
        """
        Logs match summary after successful scoring.

        Args:
            result: Successful AgentResult.
        """
        if result.output:
            output: JobMatchingOutput = result.output
            summary = output.score_summary
            self.logger.info(
                f"Matching summary | "
                f"total={summary.total_jobs} | "
                f"recommended={summary.recommended_count} | "
                f"avg={round(summary.avg_score*100,1)}% | "
                f"top={round(summary.highest_score*100,1)}% | "
                f"distribution={summary.score_distribution}"
            )

    async def on_failure(self, result: AgentResult) -> None:
        """
        Handles scoring failures.

        Match scoring failure means downstream agents (4-7) cannot
        run because they need ranked jobs. The orchestrator will
        abort the pipeline or skip to Agent 8.

        Args:
            result: Failed AgentResult.
        """
        self.logger.error(
            f"Job matching failed — downstream agents 4-7 cannot run. "
            f"Error: {result.error}"
        )

    # ----------------------------------------------------------
    # Representation
    # ----------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"<JobMatchingAgent "
            f"step={self.step_number}/{self.total_steps} "
            f"threshold={self._min_score}>"
        )