# ============================================================
# File: backend/models/job.py
# Purpose: Job listing SQLAlchemy ORM model + Pydantic schemas.
#
#   - Stores job listings fetched by Agent 2 (Job Search)
#   - Stores match scores calculated by Agent 3 (Job Matching)
#   - Stores skill gap data from Agent 4
#   - Acts as input for Agents 5, 6, 7
#   - Tracks application status via Agent 8
#
# Used by:
#   - backend/db/database.py                  → imported in init_db()
#   - backend/agents/job_search_agent.py      → creates Job rows
#   - backend/agents/job_matching_agent.py    → updates match scores
#   - backend/agents/skill_gap_agent.py       → reads required_skills
#   - backend/agents/ats_optimizer_agent.py   → reads job_description
#   - backend/agents/cover_letter_agent.py    → reads full job details
#   - backend/agents/interview_prep_agent.py  → reads job details
#   - backend/models/application.py           → foreign key here
#   - backend/db/repositories/job_repo.py     → CRUD operations
#   - backend/api/routes/jobs.py              → search + fetch endpoints
# ============================================================

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, HttpUrl, field_validator
from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.db.database import Base


# ============================================================
# SQLAlchemy ORM Model — Job
# ============================================================

class Job(Base):
    """
    Job table — stores job listings and their match analysis.

    Table name: jobs

    A Job row is created when Agent 2 finds a listing.
    It is then enriched by Agent 3 with match scores.

    One job can appear in multiple users' searches, so the
    match data (score, missing skills) is stored per-user
    in the JobMatch model below — keeping the core Job
    record reusable across users.

    Unique constraint on (source, external_job_id) prevents
    duplicate scraping of the same listing.
    """

    __tablename__ = "jobs"

    # Prevent duplicate job listings from the same source
    __table_args__ = (
        UniqueConstraint(
            "source",
            "external_job_id",
            name="uq_jobs_source_external_id",
        ),
    )

    # ----------------------------------------------------------
    # Primary Key
    # ----------------------------------------------------------
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True,
        comment="Unique job identifier (UUID v4)",
    )

    # ----------------------------------------------------------
    # Source Tracking
    # ----------------------------------------------------------
    source: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True,
        comment=(
            "Platform where job was found: "
            "'linkedin' | 'naukri' | 'indeed' | "
            "'internshala' | 'jsearch_api' | 'manual'"
        ),
    )

    external_job_id: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
        index=True,
        comment="Original job ID on the source platform",
    )

    job_url: Mapped[Optional[str]] = mapped_column(
        String(2000),
        nullable=True,
        comment="Direct URL to the job listing",
    )

    # ----------------------------------------------------------
    # Core Job Details
    # ----------------------------------------------------------
    title: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
        index=True,
        comment="Job title e.g. 'Senior Backend Engineer'",
    )

    company_name: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
        index=True,
        comment="Hiring company name",
    )

    company_logo_url: Mapped[Optional[str]] = mapped_column(
        String(2000),
        nullable=True,
        comment="URL to company logo image",
    )

    company_website: Mapped[Optional[str]] = mapped_column(
        String(2000),
        nullable=True,
        comment="Company website URL",
    )

    company_size: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        comment="Company size e.g. '1-10' | '51-200' | '1000+'",
    )

    industry: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        comment="Industry sector e.g. 'FinTech', 'Healthcare', 'SaaS'",
    )

    # ----------------------------------------------------------
    # Location & Work Mode
    # ----------------------------------------------------------
    location: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
        index=True,
        comment="Job location e.g. 'Bangalore, India' or 'Remote'",
    )

    country: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        comment="Country code or name e.g. 'IN', 'US', 'India'",
    )

    work_mode: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        comment="Work mode: 'remote' | 'onsite' | 'hybrid'",
    )

    # ----------------------------------------------------------
    # Job Description & Requirements
    # ----------------------------------------------------------
    job_description: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Full job description text — used by all downstream agents",
    )

    responsibilities: Mapped[Optional[List[str]]] = mapped_column(
        JSON,
        nullable=True,
        comment="Parsed list of job responsibilities",
    )

    requirements: Mapped[Optional[List[str]]] = mapped_column(
        JSON,
        nullable=True,
        comment="Parsed list of job requirements / qualifications",
    )

    # ----------------------------------------------------------
    # Skills & Keywords
    # ----------------------------------------------------------
    required_skills: Mapped[Optional[List[str]]] = mapped_column(
        JSON,
        nullable=True,
        comment="Skills explicitly required in the job description",
    )

    preferred_skills: Mapped[Optional[List[str]]] = mapped_column(
        JSON,
        nullable=True,
        comment="Skills mentioned as 'nice to have' or 'preferred'",
    )

    keywords: Mapped[Optional[List[str]]] = mapped_column(
        JSON,
        nullable=True,
        comment="ATS keywords extracted from job description",
    )

    # ----------------------------------------------------------
    # Experience & Education Requirements
    # ----------------------------------------------------------
    experience_min_years: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="Minimum years of experience required",
    )

    experience_max_years: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="Maximum years of experience (upper bound of range)",
    )

    experience_level: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        comment=(
            "Experience level: "
            "'internship' | 'entry' | 'junior' | "
            "'mid' | 'senior' | 'lead' | 'executive'"
        ),
    )

    education_required: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        comment="Required education e.g. 'B.Tech in CS' or 'Any Graduate'",
    )

    # ----------------------------------------------------------
    # Salary Information
    # ----------------------------------------------------------
    salary_min: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Minimum salary (in currency units from salary_currency)",
    )

    salary_max: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Maximum salary",
    )

    salary_currency: Mapped[Optional[str]] = mapped_column(
        String(10),
        nullable=True,
        comment="Salary currency code e.g. 'INR', 'USD', 'EUR'",
    )

    salary_period: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        comment="Salary period: 'annual' | 'monthly' | 'hourly'",
    )

    salary_display: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        comment="Human-readable salary string e.g. '15-25 LPA'",
    )

    # ----------------------------------------------------------
    # Job Metadata
    # ----------------------------------------------------------
    job_type: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        comment=(
            "Employment type: "
            "'full_time' | 'part_time' | "
            "'contract' | 'internship' | 'freelance'"
        ),
    )

    department: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        comment="Department e.g. 'Engineering', 'Product', 'Data Science'",
    )

    benefits: Mapped[Optional[List[str]]] = mapped_column(
        JSON,
        nullable=True,
        comment="List of benefits e.g. ['Health Insurance', 'Stock Options']",
    )

    application_deadline: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Application closing date if mentioned",
    )

    posted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        index=True,
        comment="When the job was posted on the source platform",
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        comment="False = job listing has expired or been removed",
    )

    # ----------------------------------------------------------
    # Timestamps
    # ----------------------------------------------------------
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="When this record was created in our DB (UTC)",
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        comment="When this record was last updated (UTC)",
    )

    # ----------------------------------------------------------
    # ORM Relationships
    # ----------------------------------------------------------
    job_matches: Mapped[list["JobMatch"]] = relationship(
        "JobMatch",
        back_populates="job",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    applications: Mapped[list["Application"]] = relationship(  # noqa: F821
        "Application",
        back_populates="job",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    # ----------------------------------------------------------
    # Helper Properties
    # ----------------------------------------------------------

    @property
    def salary_range_display(self) -> str:
        """
        Returns a formatted salary range string.

        Examples:
            "₹15L - ₹25L per year"
            "$80k - $120k per year"
            "Salary not disclosed"
        """
        if not self.salary_min and not self.salary_max:
            return self.salary_display or "Salary not disclosed"

        currency_symbols = {
            "INR": "₹", "USD": "$",
            "EUR": "€", "GBP": "£",
        }
        symbol = currency_symbols.get(
            self.salary_currency or "", self.salary_currency or ""
        )

        def _format(amount: float) -> str:
            if self.salary_currency == "INR" and amount >= 100000:
                return f"{amount / 100000:.1f}L"
            if amount >= 1000:
                return f"{amount / 1000:.0f}k"
            return str(int(amount))

        period = self.salary_period or "annual"
        min_str = f"{symbol}{_format(self.salary_min)}" if self.salary_min else ""
        max_str = f"{symbol}{_format(self.salary_max)}" if self.salary_max else ""

        if min_str and max_str:
            return f"{min_str} - {max_str} per {period}"
        return min_str or max_str or "Salary not disclosed"

    @property
    def experience_range_display(self) -> str:
        """Returns human-readable experience range."""
        if self.experience_min_years is None and self.experience_max_years is None:
            return "Experience not specified"
        if self.experience_min_years is not None and self.experience_max_years is not None:
            return f"{self.experience_min_years}-{self.experience_max_years} years"
        if self.experience_min_years is not None:
            return f"{self.experience_min_years}+ years"
        return f"Up to {self.experience_max_years} years"

    @property
    def all_skills(self) -> List[str]:
        """
        Combines required + preferred skills into one deduplicated list.
        Used by Agent 3 for comprehensive matching.
        """
        skill_set = set()
        if self.required_skills:
            skill_set.update(self.required_skills)
        if self.preferred_skills:
            skill_set.update(self.preferred_skills)
        return sorted(skill_set)

    def __repr__(self) -> str:
        return (
            f"<Job id={self.id} "
            f"title='{self.title}' "
            f"company='{self.company_name}' "
            f"source='{self.source}'>"
        )


# ============================================================
# SQLAlchemy ORM Model — JobMatch
# ============================================================

class JobMatch(Base):
    """
    JobMatch table — stores per-user match analysis for each job.

    Table name: job_matches

    Why separate from Job?
        The same job listing can be matched against different
        users' resumes with different scores. Keeping match
        data separate lets us reuse Job rows across users
        without duplicating the full listing.

    Created by: Agent 3 (Job Matching Agent)
    Read by:    Agent 4 (Skill Gap), Agent 5 (ATS),
                Agent 6 (Cover Letter), Agent 7 (Interview Prep)
    """

    __tablename__ = "job_matches"

    # Prevent duplicate match records for same user+job+resume combo
    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "job_id",
            "resume_id",
            name="uq_job_matches_user_job_resume",
        ),
    )

    # ----------------------------------------------------------
    # Primary Key
    # ----------------------------------------------------------
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True,
    )

    # ----------------------------------------------------------
    # Foreign Keys
    # ----------------------------------------------------------
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="User whose resume was matched",
    )

    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Job that was matched against",
    )

    resume_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("resumes.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Resume used for this match calculation",
    )

    # ----------------------------------------------------------
    # Match Scores (Agent 3 Output)
    # ----------------------------------------------------------
    overall_score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        comment="Overall match score 0.0-1.0 (displayed as percentage)",
    )

    skill_match_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Skills similarity score 0.0-1.0",
    )

    experience_match_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="Experience level match score 0.0-1.0",
    )

    keyword_match_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="ATS keyword overlap score 0.0-1.0",
    )

    semantic_score: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment=(
            "Semantic similarity score from sentence-transformers "
            "embeddings 0.0-1.0"
        ),
    )

    # ----------------------------------------------------------
    # Skill Gap Analysis (Agent 4 Output)
    # ----------------------------------------------------------
    matched_skills: Mapped[Optional[List[str]]] = mapped_column(
        JSON,
        nullable=True,
        comment="Skills the user has that match the job requirements",
    )

    missing_skills: Mapped[Optional[List[str]]] = mapped_column(
        JSON,
        nullable=True,
        comment="Required skills the user is missing",
    )

    bonus_skills: Mapped[Optional[List[str]]] = mapped_column(
        JSON,
        nullable=True,
        comment="User skills not required but beneficial for this role",
    )

    skill_gap_analysis: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        nullable=True,
        comment="Full skill gap output from Agent 4 (JSON)",
    )

    # ----------------------------------------------------------
    # Ranking
    # ----------------------------------------------------------
    rank: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        index=True,
        comment="Rank among all matched jobs for this user (1 = best)",
    )

    is_recommended: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        comment="True if score exceeds MIN_MATCH_SCORE threshold",
    )

    # ----------------------------------------------------------
    # Agent Processing Flags
    # ----------------------------------------------------------
    ats_generated: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        comment="True if Agent 5 has generated ATS resume for this job",
    )

    cover_letter_generated: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        comment="True if Agent 6 has generated cover letter for this job",
    )

    interview_prep_generated: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        comment="True if Agent 7 has generated interview kit for this job",
    )

    # ----------------------------------------------------------
    # Timestamps
    # ----------------------------------------------------------
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # ----------------------------------------------------------
    # ORM Relationships
    # ----------------------------------------------------------
    job: Mapped["Job"] = relationship(
        "Job",
        back_populates="job_matches",
        lazy="selectin",
    )

    # ----------------------------------------------------------
    # Helper Properties
    # ----------------------------------------------------------

    @property
    def overall_score_percent(self) -> float:
        """Returns overall score as a 0-100 percentage."""
        return round(self.overall_score * 100, 1)

    @property
    def match_label(self) -> str:
        """
        Returns a human-readable match quality label.

        Labels:
            >= 80% → "Excellent Match"
            >= 60% → "Good Match"
            >= 40% → "Fair Match"
            <  40% → "Low Match"
        """
        score = self.overall_score_percent
        if score >= 80:
            return "Excellent Match"
        elif score >= 60:
            return "Good Match"
        elif score >= 40:
            return "Fair Match"
        return "Low Match"

    @property
    def missing_skills_count(self) -> int:
        """Number of required skills the user is missing."""
        return len(self.missing_skills) if self.missing_skills else 0

    @property
    def matched_skills_count(self) -> int:
        """Number of required skills the user already has."""
        return len(self.matched_skills) if self.matched_skills else 0

    def __repr__(self) -> str:
        return (
            f"<JobMatch id={self.id} "
            f"score={self.overall_score_percent}% "
            f"rank={self.rank} "
            f"label='{self.match_label}'>"
        )


# ============================================================
# Pydantic Schemas — Job
# ============================================================

class JobBase(BaseModel):
    """Shared fields for job schemas."""

    title: str = Field(..., min_length=1, max_length=500)
    company_name: str = Field(..., min_length=1, max_length=500)
    location: Optional[str] = Field(None, max_length=500)
    work_mode: Optional[str] = Field(None)
    job_type: Optional[str] = Field(None)
    experience_level: Optional[str] = Field(None)
    salary_display: Optional[str] = Field(None)


class JobCreate(JobBase):
    """
    Schema used by Agent 2 to create a new job record.
    All fields Agent 2 can populate from scraping or API.
    """

    source: str = Field(..., description="Source platform identifier")
    external_job_id: Optional[str] = None
    job_url: Optional[str] = None
    company_logo_url: Optional[str] = None
    company_website: Optional[str] = None
    company_size: Optional[str] = None
    industry: Optional[str] = None
    country: Optional[str] = None
    job_description: Optional[str] = None
    responsibilities: Optional[List[str]] = None
    requirements: Optional[List[str]] = None
    required_skills: Optional[List[str]] = None
    preferred_skills: Optional[List[str]] = None
    keywords: Optional[List[str]] = None
    experience_min_years: Optional[int] = None
    experience_max_years: Optional[int] = None
    education_required: Optional[str] = None
    salary_min: Optional[float] = None
    salary_max: Optional[float] = None
    salary_currency: Optional[str] = None
    salary_period: Optional[str] = None
    department: Optional[str] = None
    benefits: Optional[List[str]] = None
    posted_at: Optional[datetime] = None

    @field_validator("work_mode")
    @classmethod
    def validate_work_mode(cls, v: Optional[str]) -> Optional[str]:
        """Normalizes work mode to a standard value."""
        if v is None:
            return v
        valid = {"remote", "onsite", "hybrid"}
        normalized = v.lower().strip()
        if normalized not in valid:
            return None     # Unknown modes stored as NULL
        return normalized

    @field_validator("job_type")
    @classmethod
    def validate_job_type(cls, v: Optional[str]) -> Optional[str]:
        """Normalizes job type to a standard value."""
        if v is None:
            return v
        mapping = {
            "full-time": "full_time",
            "fulltime": "full_time",
            "full time": "full_time",
            "part-time": "part_time",
            "parttime": "part_time",
            "part time": "part_time",
            "internship": "internship",
            "intern": "internship",
            "contract": "contract",
            "freelance": "freelance",
        }
        return mapping.get(v.lower().strip(), v.lower())


class JobResponse(JobBase):
    """
    Full job response returned by GET /jobs/{id}.
    Includes all parsed fields and computed properties.
    """

    id: uuid.UUID
    source: str
    job_url: Optional[str] = None
    company_logo_url: Optional[str] = None
    company_website: Optional[str] = None
    company_size: Optional[str] = None
    industry: Optional[str] = None
    country: Optional[str] = None
    job_description: Optional[str] = None
    responsibilities: Optional[List[str]] = None
    requirements: Optional[List[str]] = None
    required_skills: Optional[List[str]] = None
    preferred_skills: Optional[List[str]] = None
    keywords: Optional[List[str]] = None
    experience_min_years: Optional[int] = None
    experience_max_years: Optional[int] = None
    education_required: Optional[str] = None
    salary_min: Optional[float] = None
    salary_max: Optional[float] = None
    salary_currency: Optional[str] = None
    salary_period: Optional[str] = None
    salary_range_display: str
    experience_range_display: str
    department: Optional[str] = None
    benefits: Optional[List[str]] = None
    posted_at: Optional[datetime] = None
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class JobMatchResponse(BaseModel):
    """
    Combined job + match score response.
    Returned by GET /jobs/matches — the ranked job list page.
    This is the primary response the frontend job cards display.
    """

    # Match metadata
    match_id: uuid.UUID
    rank: Optional[int]
    overall_score: float
    overall_score_percent: float
    match_label: str
    skill_match_score: Optional[float]
    experience_match_score: Optional[float]
    keyword_match_score: Optional[float]
    semantic_score: Optional[float]
    is_recommended: bool

    # Skill breakdown
    matched_skills: Optional[List[str]]
    missing_skills: Optional[List[str]]
    bonus_skills: Optional[List[str]]
    matched_skills_count: int
    missing_skills_count: int

    # Agent completion flags
    ats_generated: bool
    cover_letter_generated: bool
    interview_prep_generated: bool

    # Core job details (nested)
    job: JobResponse

    model_config = {"from_attributes": True}


class JobSearchRequest(BaseModel):
    """
    Request body for POST /jobs/search.
    Sent by the frontend when the user starts a pipeline run.
    Agent 2 uses these parameters to search job boards.
    """

    role: str = Field(
        ...,
        min_length=2,
        max_length=255,
        description="Target job title to search for",
        examples=["Backend Engineer", "Data Scientist", "Product Manager"],
    )
    location: Optional[str] = Field(
        None,
        max_length=255,
        description="Preferred location or 'Remote'",
        examples=["Bangalore", "Remote", "Mumbai"],
    )
    experience_years: Optional[int] = Field(
        None,
        ge=0,
        le=60,
        description="Years of experience to filter by",
    )
    work_mode: Optional[str] = Field(
        None,
        description="Preferred work mode: 'remote' | 'onsite' | 'hybrid'",
    )
    job_type: Optional[str] = Field(
        None,
        description="Employment type: 'full_time' | 'internship' | 'contract'",
    )
    max_results: Optional[int] = Field(
        default=20,
        ge=1,
        le=50,
        description="Maximum number of jobs to search for",
    )
    sources: Optional[List[str]] = Field(
        default=None,
        description=(
            "Specific platforms to search. "
            "Defaults to all available sources."
        ),
        examples=[["linkedin", "naukri"]],
    )
    resume_id: Optional[uuid.UUID] = Field(
        None,
        description=(
            "Resume ID to use for matching. "
            "Defaults to user's active resume."
        ),
    )


class JobSearchFilters(BaseModel):
    """
    Query filters for GET /jobs — listing and filtering matched jobs.
    All fields optional — omitting a field means no filter applied.
    """

    min_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Minimum match score filter",
    )
    work_mode: Optional[str] = None
    job_type: Optional[str] = None
    experience_level: Optional[str] = None
    location: Optional[str] = None
    source: Optional[str] = None
    is_recommended: Optional[bool] = None
    sort_by: Optional[str] = Field(
        default="rank",
        description="Sort field: 'rank' | 'score' | 'posted_at' | 'salary'",
    )
    sort_order: Optional[str] = Field(
        default="asc",
        description="Sort direction: 'asc' | 'desc'",
    )
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=100)


class JobListResponse(BaseModel):
    """
    Paginated list of matched jobs.
    Returned by GET /jobs/matches.
    """

    total: int = Field(..., description="Total number of matched jobs")
    page: int
    page_size: int
    total_pages: int
    jobs: List[JobMatchResponse]

    model_config = {"from_attributes": True}