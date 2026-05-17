# ============================================================
# File: backend/models/resume.py
# Purpose: Resume SQLAlchemy ORM model + Pydantic schemas.
#
#   - Stores uploaded resume file path
#   - Stores parsed structured data from Agent 1
#   - Acts as input for Agents 3, 5, 6, 7
#   - Tracks ATS optimization status
#   - One user can have multiple resume versions
#
# Used by:
#   - backend/db/database.py              → imported in init_db()
#   - backend/agents/resume_parser_agent.py → creates Resume rows
#   - backend/agents/ats_optimizer_agent.py → updates ats_resume
#   - backend/agents/job_matching_agent.py  → reads parsed_data
#   - backend/agents/cover_letter_agent.py  → reads parsed_data
#   - backend/agents/interview_prep_agent.py→ reads parsed_data
#   - backend/db/repositories/resume_repo.py→ CRUD operations
#   - backend/api/routes/resume.py          → upload + fetch endpoints
# ============================================================

import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.db.database import Base


# ============================================================
# SQLAlchemy ORM Model
# ============================================================

class Resume(Base):
    """
    Resume table — stores uploaded resumes and their parsed data.

    Table name: resumes

    Lifecycle:
        1. User uploads PDF/DOCX → file saved → Resume row created
           (status = "uploaded")

        2. Agent 1 parses the file → structured data stored in
           parsed_data JSON column
           (status = "parsed")

        3. Agent 5 rewrites resume → ats_optimized_text stored
           (ats_status = "optimized")

        4. User can have multiple resumes — active_resume flag
           marks the one used by default in the pipeline.

    JSON Columns:
        parsed_data        → Full structured output from Agent 1
        skills             → Flat list of skill strings (for fast querying)
        ats_keywords       → Keywords injected by Agent 5
    """

    __tablename__ = "resumes"

    # ----------------------------------------------------------
    # Primary Key
    # ----------------------------------------------------------
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True,
        comment="Unique resume identifier (UUID v4)",
    )

    # ----------------------------------------------------------
    # Foreign Key → User
    # ----------------------------------------------------------
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Owner of this resume",
    )

    # ----------------------------------------------------------
    # File Storage
    # ----------------------------------------------------------
    original_filename: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
        comment="Original uploaded filename e.g. 'Rahul_Resume.pdf'",
    )

    file_path: Mapped[str] = mapped_column(
        String(1000),
        nullable=False,
        comment="Absolute path to stored file on disk",
    )

    file_type: Mapped[str] = mapped_column(
        String(10),
        nullable=False,
        comment="File extension: 'pdf', 'docx', or 'doc'",
    )

    file_size_bytes: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="File size in bytes",
    )

    # ----------------------------------------------------------
    # Raw Text Content
    # ----------------------------------------------------------
    raw_text: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Full plain text extracted from the resume file",
    )

    # ----------------------------------------------------------
    # Parsed Structured Data (Agent 1 Output)
    # ----------------------------------------------------------
    # parsed_data stores the complete structured output from
    # the Resume Parser Agent as a JSON object.
    # Schema is defined by ResumeParseResult Pydantic model below.

    parsed_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        nullable=True,
        comment="Full structured resume data from Agent 1 (JSON)",
    )

    # Flat skills list extracted for fast DB querying and filtering
    # Example: ["Python", "FastAPI", "PostgreSQL", "Docker"]
    skills: Mapped[Optional[List[str]]] = mapped_column(
        JSON,
        nullable=True,
        comment="Flat list of skills for fast lookup",
    )

    # ----------------------------------------------------------
    # ATS Optimization (Agent 5 Output)
    # ----------------------------------------------------------
    ats_optimized_text: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Full ATS-optimized resume text from Agent 5",
    )

    ats_keywords: Mapped[Optional[List[str]]] = mapped_column(
        JSON,
        nullable=True,
        comment="Keywords injected by ATS Optimizer Agent",
    )

    ats_score_before: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="ATS compatibility score before optimization (0.0-1.0)",
    )

    ats_score_after: Mapped[Optional[float]] = mapped_column(
        Float,
        nullable=True,
        comment="ATS compatibility score after optimization (0.0-1.0)",
    )

    # ----------------------------------------------------------
    # Resume Metadata
    # ----------------------------------------------------------
    title: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        comment="User-defined label e.g. 'Backend Dev Resume v2'",
    )

    target_role: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        comment="Role this resume is tailored for",
    )

    # Whether this is the default resume used in pipeline runs
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        comment="True = this resume is selected for pipeline runs",
    )

    # ----------------------------------------------------------
    # Processing Status
    # ----------------------------------------------------------
    # Tracks which stage of the pipeline this resume has been through

    parse_status: Mapped[str] = mapped_column(
        String(50),
        default="uploaded",
        nullable=False,
        comment=(
            "Parse status: "
            "'uploaded' | 'parsing' | 'parsed' | 'failed'"
        ),
    )

    ats_status: Mapped[str] = mapped_column(
        String(50),
        default="pending",
        nullable=False,
        comment=(
            "ATS status: "
            "'pending' | 'optimizing' | 'optimized' | 'failed'"
        ),
    )

    parse_error: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Error message if parsing failed — for debugging",
    )

    # ----------------------------------------------------------
    # Timestamps
    # ----------------------------------------------------------
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="When the resume was uploaded (UTC)",
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        comment="When the resume was last modified (UTC)",
    )

    parsed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="When Agent 1 successfully parsed this resume (UTC)",
    )

    ats_optimized_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="When Agent 5 optimized this resume (UTC)",
    )

    # ----------------------------------------------------------
    # ORM Relationships
    # ----------------------------------------------------------
    user: Mapped["User"] = relationship(  # noqa: F821
        "User",
        back_populates="resumes",
        lazy="selectin",
    )

    # ----------------------------------------------------------
    # Helper Methods
    # ----------------------------------------------------------

    @property
    def is_parsed(self) -> bool:
        """True if Agent 1 has successfully parsed this resume."""
        return self.parse_status == "parsed" and self.parsed_data is not None

    @property
    def is_ats_optimized(self) -> bool:
        """True if Agent 5 has optimized this resume."""
        return self.ats_status == "optimized" and self.ats_optimized_text is not None

    @property
    def skills_count(self) -> int:
        """Number of skills extracted from this resume."""
        return len(self.skills) if self.skills else 0

    @property
    def ats_improvement(self) -> Optional[float]:
        """
        Percentage improvement in ATS score after optimization.
        Returns None if optimization hasn't happened yet.
        """
        if self.ats_score_before is not None and self.ats_score_after is not None:
            if self.ats_score_before == 0:
                return 100.0
            improvement = (
                (self.ats_score_after - self.ats_score_before)
                / self.ats_score_before
                * 100
            )
            return round(improvement, 2)
        return None

    def mark_parsing_started(self) -> None:
        """Call when Agent 1 starts processing this resume."""
        self.parse_status = "parsing"

    def mark_parsing_complete(
        self,
        parsed_data: Dict[str, Any],
        raw_text: str,
        skills: List[str],
    ) -> None:
        """
        Call when Agent 1 successfully finishes parsing.

        Args:
            parsed_data: Full structured JSON output from Agent 1.
            raw_text:    Plain text extracted from the resume file.
            skills:      Flat list of skill strings for fast lookup.
        """
        from datetime import timezone
        self.parse_status = "parsed"
        self.parsed_data = parsed_data
        self.raw_text = raw_text
        self.skills = skills
        self.parsed_at = datetime.now(timezone.utc)
        self.parse_error = None

    def mark_parsing_failed(self, error: str) -> None:
        """Call when Agent 1 encounters an unrecoverable error."""
        self.parse_status = "failed"
        self.parse_error = error

    def mark_ats_optimized(
        self,
        optimized_text: str,
        keywords: List[str],
        score_before: float,
        score_after: float,
    ) -> None:
        """
        Call when Agent 5 successfully optimizes this resume.

        Args:
            optimized_text: Full rewritten resume text.
            keywords:       List of ATS keywords injected.
            score_before:   ATS score before optimization.
            score_after:    ATS score after optimization.
        """
        from datetime import timezone
        self.ats_status = "optimized"
        self.ats_optimized_text = optimized_text
        self.ats_keywords = keywords
        self.ats_score_before = score_before
        self.ats_score_after = score_after
        self.ats_optimized_at = datetime.now(timezone.utc)

    def __repr__(self) -> str:
        return (
            f"<Resume id={self.id} "
            f"user_id={self.user_id} "
            f"status={self.parse_status} "
            f"file={self.original_filename}>"
        )


# ============================================================
# Pydantic Schemas — Nested Parsed Data Structure
# ============================================================
# These schemas define the exact JSON structure stored in the
# parsed_data column. Agent 1 outputs a ResumeParseResult
# which gets stored as JSON. All downstream agents
# deserialize this JSON back into these Pydantic models.

class WorkExperience(BaseModel):
    """Single work experience entry from parsed resume."""

    company: str = Field(..., description="Company or organization name")
    role: str = Field(..., description="Job title or designation")
    location: Optional[str] = Field(None, description="City/Country or 'Remote'")
    start_date: Optional[str] = Field(
        None,
        description="Start date as string e.g. 'Jan 2022' or '2022-01'",
    )
    end_date: Optional[str] = Field(
        None,
        description="End date or 'Present' if currently working here",
    )
    is_current: bool = Field(
        default=False,
        description="True if this is the current job",
    )
    duration_months: Optional[int] = Field(
        None,
        description="Calculated duration in months",
    )
    responsibilities: List[str] = Field(
        default_factory=list,
        description="List of bullet point responsibilities/achievements",
    )
    technologies: List[str] = Field(
        default_factory=list,
        description="Technologies used in this role",
    )


class Education(BaseModel):
    """Single education entry from parsed resume."""

    institution: str = Field(..., description="University or college name")
    degree: Optional[str] = Field(
        None,
        description="Degree type e.g. 'B.Tech', 'M.Sc', 'MBA'",
    )
    field_of_study: Optional[str] = Field(
        None,
        description="Major or field e.g. 'Computer Science'",
    )
    start_date: Optional[str] = Field(None, description="Start year or date")
    end_date: Optional[str] = Field(
        None,
        description="Graduation year or 'Present'",
    )
    grade: Optional[str] = Field(
        None,
        description="GPA, percentage, or grade e.g. '8.5 CGPA' or '85%'",
    )
    achievements: List[str] = Field(
        default_factory=list,
        description="Honors, awards, or relevant coursework",
    )


class Project(BaseModel):
    """Single project entry from parsed resume."""

    name: str = Field(..., description="Project name")
    description: Optional[str] = Field(
        None,
        description="What the project does — 1-2 sentences",
    )
    technologies: List[str] = Field(
        default_factory=list,
        description="Technologies, frameworks, and tools used",
    )
    url: Optional[str] = Field(
        None,
        description="GitHub link, live demo URL, or paper link",
    )
    highlights: List[str] = Field(
        default_factory=list,
        description="Key achievements or metrics from this project",
    )


class Certification(BaseModel):
    """Single certification or course from parsed resume."""

    name: str = Field(..., description="Certification or course name")
    issuer: Optional[str] = Field(
        None,
        description="Issuing organization e.g. 'AWS', 'Google', 'Coursera'",
    )
    date: Optional[str] = Field(
        None,
        description="Date earned or expiry date",
    )
    url: Optional[str] = Field(None, description="Verification link")


class SkillCategory(BaseModel):
    """
    Skills grouped by category.
    Example: { category: "Backend", skills: ["FastAPI", "Django"] }
    """

    category: str = Field(
        ...,
        description="Skill group name e.g. 'Programming Languages', 'Frameworks'",
    )
    skills: List[str] = Field(
        ...,
        description="List of skills in this category",
    )


class ResumeParseResult(BaseModel):
    """
    Complete structured output from Agent 1 (Resume Parser).

    This is stored as JSON in Resume.parsed_data column.
    All downstream agents deserialize this model from that column.

    Usage in agents:
        parsed = ResumeParseResult(**resume.parsed_data)
        all_skills = parsed.all_skills
        total_exp = parsed.total_experience_years
    """

    # Contact Information
    full_name: Optional[str] = Field(None, description="Candidate full name")
    email: Optional[str] = Field(None, description="Contact email")
    phone: Optional[str] = Field(None, description="Contact phone number")
    location: Optional[str] = Field(
        None,
        description="Current city/country from resume",
    )
    linkedin_url: Optional[str] = Field(None, description="LinkedIn URL if present")
    github_url: Optional[str] = Field(None, description="GitHub URL if present")
    portfolio_url: Optional[str] = Field(
        None,
        description="Portfolio or website URL if present",
    )

    # Professional Summary
    summary: Optional[str] = Field(
        None,
        description="Professional summary or objective statement",
    )

    # Core Sections
    work_experience: List[WorkExperience] = Field(
        default_factory=list,
        description="Work experience entries in reverse chronological order",
    )
    education: List[Education] = Field(
        default_factory=list,
        description="Education entries in reverse chronological order",
    )
    projects: List[Project] = Field(
        default_factory=list,
        description="Personal or professional projects",
    )
    certifications: List[Certification] = Field(
        default_factory=list,
        description="Certifications, licenses, and courses",
    )

    # Skills
    skill_categories: List[SkillCategory] = Field(
        default_factory=list,
        description="Skills grouped by category",
    )
    all_skills_flat: List[str] = Field(
        default_factory=list,
        description="All skills as a single flat list for quick matching",
    )

    # Keywords
    keywords: List[str] = Field(
        default_factory=list,
        description="Important keywords extracted for ATS matching",
    )

    # Computed Metadata
    total_experience_years: Optional[float] = Field(
        None,
        description="Total professional experience in years (calculated)",
    )
    seniority_level: Optional[str] = Field(
        None,
        description=(
            "Inferred seniority: "
            "'entry' | 'junior' | 'mid' | 'senior' | 'lead' | 'executive'"
        ),
    )
    primary_role: Optional[str] = Field(
        None,
        description="Inferred primary job title e.g. 'Backend Engineer'",
    )
    industry: Optional[str] = Field(
        None,
        description="Inferred industry e.g. 'Software', 'Finance', 'Healthcare'",
    )

    @property
    def all_technologies(self) -> List[str]:
        """
        Returns all technologies mentioned across work experience
        and projects — deduplicated. Used by Job Matching Agent.
        """
        tech_set = set(self.all_skills_flat)
        for exp in self.work_experience:
            tech_set.update(exp.technologies)
        for project in self.projects:
            tech_set.update(project.technologies)
        return sorted(tech_set)

    @property
    def latest_role(self) -> Optional[str]:
        """Returns the most recent job title from work experience."""
        if self.work_experience:
            return self.work_experience[0].role
        return None

    @property
    def latest_company(self) -> Optional[str]:
        """Returns the most recent company name."""
        if self.work_experience:
            return self.work_experience[0].company
        return None


# ============================================================
# Pydantic API Schemas
# ============================================================

class ResumeUploadResponse(BaseModel):
    """Returned immediately after a resume file is uploaded."""

    resume_id: uuid.UUID
    original_filename: str
    file_size_bytes: int
    file_type: str
    parse_status: str
    message: str = Field(
        default="Resume uploaded successfully. Parsing will begin shortly."
    )

    model_config = {"from_attributes": True}


class ResumeStatusResponse(BaseModel):
    """
    Returned when polling resume parse + ATS status.
    Frontend polls GET /resume/{id}/status until parse_status = 'parsed'.
    """

    resume_id: uuid.UUID
    parse_status: str
    ats_status: str
    is_parsed: bool
    is_ats_optimized: bool
    skills_count: int
    ats_improvement: Optional[float] = None
    parse_error: Optional[str] = None

    model_config = {"from_attributes": True}


class ResumeResponse(BaseModel):
    """
    Full resume response — includes all parsed data.
    Returned by GET /resume/{id}.
    """

    id: uuid.UUID
    user_id: uuid.UUID
    original_filename: str
    file_type: str
    file_size_bytes: int
    title: Optional[str]
    target_role: Optional[str]
    is_active: bool
    parse_status: str
    ats_status: str
    skills: Optional[List[str]]
    ats_keywords: Optional[List[str]]
    ats_score_before: Optional[float]
    ats_score_after: Optional[float]
    ats_improvement: Optional[float]
    parsed_data: Optional[Dict[str, Any]]
    created_at: datetime
    updated_at: datetime
    parsed_at: Optional[datetime]
    ats_optimized_at: Optional[datetime]

    model_config = {"from_attributes": True}


class ResumeListItem(BaseModel):
    """
    Compact resume info for list views.
    Returned by GET /resume/ (list all user resumes).
    """

    id: uuid.UUID
    original_filename: str
    title: Optional[str]
    target_role: Optional[str]
    is_active: bool
    parse_status: str
    ats_status: str
    skills_count: int
    created_at: datetime

    model_config = {"from_attributes": True}


class ResumeUpdate(BaseModel):
    """
    Schema for PATCH /resume/{id} — update metadata only.
    Does not re-trigger parsing.
    """

    title: Optional[str] = Field(None, max_length=255)
    target_role: Optional[str] = Field(None, max_length=255)
    is_active: Optional[bool] = None