# ============================================================
# File: backend/models/application.py
# Purpose: Application Tracker SQLAlchemy ORM model + Pydantic
#          schemas for Agent 8 (Application Tracker Agent).
#
#   - Tracks every job application the user submits
#   - Stores full application lifecycle status
#   - Stores interview rounds, dates, and notes
#   - Links generated cover letters and ATS resumes to jobs
#   - Powers the dashboard + tracking UI
#
# Used by:
#   - backend/db/database.py                   → imported in init_db()
#   - backend/agents/tracker_agent.py          → creates + updates rows
#   - backend/db/repositories/tracker_repo.py  → CRUD operations
#   - backend/api/routes/tracker.py            → dashboard endpoints
#   - backend/models/user.py                   → User.applications rel
#   - backend/models/job.py                    → Job.applications rel
# ============================================================

import uuid
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator
from sqlalchemy import (
    Boolean,
    Date,
    DateTime,
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
# Application Status Constants
# ============================================================
# These are the valid lifecycle states for an application.
# Status flows forward — never backward (except 'withdrawn').
#
# Flow:
#   saved → applied → screening → interviewing
#            → technical_round → hr_round
#            → offered → accepted / rejected / withdrawn

class ApplicationStatus:
    """
    Valid application status values.
    Used as constants throughout the codebase — avoids typos.
    """
    SAVED          = "saved"           # User bookmarked — not yet applied
    APPLIED        = "applied"         # Application submitted
    SCREENING      = "screening"       # Initial HR/recruiter screening
    INTERVIEWING   = "interviewing"    # In interview process (general)
    TECHNICAL      = "technical_round" # Technical interview round
    HR_ROUND       = "hr_round"        # Final HR discussion
    OFFERED        = "offered"         # Offer letter received
    ACCEPTED       = "accepted"        # User accepted the offer
    REJECTED       = "rejected"        # Application rejected
    WITHDRAWN      = "withdrawn"       # User withdrew application
    NO_RESPONSE    = "no_response"     # No reply after application

    ALL_STATUSES = [
        SAVED, APPLIED, SCREENING, INTERVIEWING,
        TECHNICAL, HR_ROUND, OFFERED, ACCEPTED,
        REJECTED, WITHDRAWN, NO_RESPONSE,
    ]

    ACTIVE_STATUSES = [
        SAVED, APPLIED, SCREENING,
        INTERVIEWING, TECHNICAL, HR_ROUND, OFFERED,
    ]

    CLOSED_STATUSES = [
        ACCEPTED, REJECTED, WITHDRAWN, NO_RESPONSE,
    ]

    POSITIVE_STATUSES = [OFFERED, ACCEPTED]
    NEGATIVE_STATUSES = [REJECTED, WITHDRAWN, NO_RESPONSE]


# ============================================================
# SQLAlchemy ORM Model — InterviewRound
# ============================================================

class InterviewRound(Base):
    """
    InterviewRound table — tracks individual interview rounds.

    Table name: interview_rounds

    One application can have multiple interview rounds.
    Each round stores type, date, interviewer, outcome and notes.

    Examples:
        Round 1: Online Assessment    → Passed
        Round 2: Technical Interview  → Passed
        Round 3: System Design        → Passed
        Round 4: HR Discussion        → Offered
    """

    __tablename__ = "interview_rounds"

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
    # Foreign Key → Application
    # ----------------------------------------------------------
    application_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("applications.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="Application this round belongs to",
    )

    # ----------------------------------------------------------
    # Round Details
    # ----------------------------------------------------------
    round_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="Round sequence number (1 = first round)",
    )

    round_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        comment=(
            "Type of interview: "
            "'online_assessment' | 'phone_screen' | "
            "'technical' | 'system_design' | "
            "'behavioral' | 'hr' | 'managerial' | 'final'"
        ),
    )

    scheduled_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Scheduled interview date and time (UTC)",
    )

    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="When the interview was actually held (UTC)",
    )

    duration_minutes: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="Interview duration in minutes",
    )

    interviewer_name: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        comment="Name of the interviewer(s)",
    )

    interviewer_linkedin: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
        comment="LinkedIn URL of interviewer (for research)",
    )

    platform: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        comment=(
            "Interview platform: "
            "'google_meet' | 'zoom' | 'teams' | "
            "'phone' | 'in_person' | 'hackerrank' | 'other'"
        ),
    )

    meeting_link: Mapped[Optional[str]] = mapped_column(
        String(2000),
        nullable=True,
        comment="Video call or test platform link",
    )

    # ----------------------------------------------------------
    # Outcome
    # ----------------------------------------------------------
    outcome: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        comment="Result: 'passed' | 'failed' | 'pending' | 'cancelled'",
    )

    feedback: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Interviewer feedback or self-assessment notes",
    )

    questions_asked: Mapped[Optional[List[str]]] = mapped_column(
        JSON,
        nullable=True,
        comment="Questions asked during this round (for future prep)",
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
    # ORM Relationship
    # ----------------------------------------------------------
    application: Mapped["Application"] = relationship(
        "Application",
        back_populates="interview_rounds",
        lazy="selectin",
    )

    # ----------------------------------------------------------
    # Helper Properties
    # ----------------------------------------------------------

    @property
    def is_upcoming(self) -> bool:
        """True if the interview is scheduled but not yet completed."""
        if self.scheduled_at and not self.completed_at:
            return self.scheduled_at > datetime.now(self.scheduled_at.tzinfo)
        return False

    @property
    def is_passed(self) -> bool:
        """True if this round was passed."""
        return self.outcome == "passed"

    def __repr__(self) -> str:
        return (
            f"<InterviewRound id={self.id} "
            f"type={self.round_type} "
            f"round={self.round_number} "
            f"outcome={self.outcome}>"
        )


# ============================================================
# SQLAlchemy ORM Model — Application
# ============================================================

class Application(Base):
    """
    Application table — tracks every job application.

    Table name: applications

    Lifecycle:
        1. User views a matched job → Application created
           (status = 'saved')

        2. User applies to the job → status = 'applied'

        3. Agent 8 / user updates status as process progresses

        4. Final status: 'accepted' | 'rejected' | 'withdrawn'

    Stores:
        - Links to generated cover letter and ATS resume
        - All interview rounds via relationship
        - Timeline of status changes
        - Offer details if applicable
        - Personal notes
    """

    __tablename__ = "applications"

    # ----------------------------------------------------------
    # Primary Key
    # ----------------------------------------------------------
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True,
        comment="Unique application identifier (UUID v4)",
    )

    # ----------------------------------------------------------
    # Foreign Keys
    # ----------------------------------------------------------
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="User who submitted this application",
    )

    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("jobs.id", ondelete="SET NULL"),
        nullable=True,       # SET NULL if job is deleted
        index=True,
        comment="Job this application is for",
    )

    resume_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("resumes.id", ondelete="SET NULL"),
        nullable=True,
        comment="Resume used for this application",
    )

    # ----------------------------------------------------------
    # Application Status
    # ----------------------------------------------------------
    status: Mapped[str] = mapped_column(
        String(50),
        default=ApplicationStatus.SAVED,
        nullable=False,
        index=True,
        comment="Current application status — see ApplicationStatus",
    )

    # Stores the full history of status changes with timestamps
    # Format: [{"status": "applied", "changed_at": "2024-01-15T10:00:00Z"}]
    status_history: Mapped[Optional[List[Dict[str, Any]]]] = mapped_column(
        JSON,
        nullable=True,
        comment="Ordered list of all status changes with timestamps",
    )

    # ----------------------------------------------------------
    # Application Details
    # ----------------------------------------------------------
    applied_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="When the application was submitted (UTC)",
    )

    application_url: Mapped[Optional[str]] = mapped_column(
        String(2000),
        nullable=True,
        comment="URL where the application was submitted",
    )

    application_platform: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        comment=(
            "Platform used to apply: "
            "'linkedin' | 'naukri' | 'indeed' | "
            "'company_website' | 'email' | 'referral' | 'other'"
        ),
    )

    referral_name: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        comment="Name of person who referred (if applied via referral)",
    )

    # ----------------------------------------------------------
    # Generated Content Links
    # ----------------------------------------------------------
    # Paths to AI-generated documents for this specific application

    cover_letter_text: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Generated cover letter text from Agent 6",
    )

    ats_resume_text: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Job-specific ATS resume from Agent 5",
    )

    cover_letter_generated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="When Agent 6 generated the cover letter",
    )

    ats_resume_generated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="When Agent 5 generated the tailored resume",
    )

    # ----------------------------------------------------------
    # Follow-up Tracking
    # ----------------------------------------------------------
    follow_up_date: Mapped[Optional[date]] = mapped_column(
        Date,
        nullable=True,
        comment="Reminder date to follow up on this application",
    )

    last_follow_up_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="When the user last sent a follow-up message",
    )

    follow_up_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        comment="Number of follow-up messages sent",
    )

    # ----------------------------------------------------------
    # Offer Details (populated when status = 'offered')
    # ----------------------------------------------------------
    offer_salary: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        comment="Offered salary/compensation package",
    )

    offer_received_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="When the offer was received",
    )

    offer_deadline: Mapped[Optional[date]] = mapped_column(
        Date,
        nullable=True,
        comment="Deadline to accept or decline the offer",
    )

    offer_details: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        nullable=True,
        comment=(
            "Full offer package details: "
            "salary, equity, bonus, benefits, start date, etc."
        ),
    )

    # ----------------------------------------------------------
    # Interview Preparation
    # ----------------------------------------------------------
    interview_prep_data: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        nullable=True,
        comment="Generated interview kit from Agent 7 (JSON)",
    )

    interview_prep_generated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="When Agent 7 generated the interview prep",
    )

    # ----------------------------------------------------------
    # User Notes
    # ----------------------------------------------------------
    notes: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Personal notes about this application",
    )

    rejection_reason: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Reason for rejection if provided by company",
    )

    # ----------------------------------------------------------
    # Priority & Bookmarking
    # ----------------------------------------------------------
    priority: Mapped[int] = mapped_column(
        Integer,
        default=2,
        nullable=False,
        comment="User priority: 1=High, 2=Medium, 3=Low",
    )

    is_starred: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        comment="True if user starred/bookmarked this application",
    )

    # ----------------------------------------------------------
    # Timestamps
    # ----------------------------------------------------------
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="When this application record was created (UTC)",
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        comment="When this record was last updated (UTC)",
    )

    closed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment=(
            "When application reached a final state "
            "(accepted/rejected/withdrawn) (UTC)"
        ),
    )

    # ----------------------------------------------------------
    # ORM Relationships
    # ----------------------------------------------------------
    user: Mapped["User"] = relationship(  # noqa: F821
        "User",
        back_populates="applications",
        lazy="selectin",
    )

    job: Mapped[Optional["Job"]] = relationship(  # noqa: F821
        "Job",
        back_populates="applications",
        lazy="selectin",
    )

    interview_rounds: Mapped[List["InterviewRound"]] = relationship(
        "InterviewRound",
        back_populates="application",
        cascade="all, delete-orphan",
        order_by="InterviewRound.round_number",
        lazy="selectin",
    )

    # ----------------------------------------------------------
    # Helper Properties
    # ----------------------------------------------------------

    @property
    def is_active(self) -> bool:
        """True if application is still in progress."""
        return self.status in ApplicationStatus.ACTIVE_STATUSES

    @property
    def is_closed(self) -> bool:
        """True if application has reached a final state."""
        return self.status in ApplicationStatus.CLOSED_STATUSES

    @property
    def is_positive_outcome(self) -> bool:
        """True if application resulted in an offer or acceptance."""
        return self.status in ApplicationStatus.POSITIVE_STATUSES

    @property
    def days_since_applied(self) -> Optional[int]:
        """
        Returns number of days since application was submitted.
        Returns None if application hasn't been submitted yet.
        """
        if not self.applied_at:
            return None
        delta = datetime.now(self.applied_at.tzinfo) - self.applied_at
        return delta.days

    @property
    def upcoming_interviews(self) -> List["InterviewRound"]:
        """Returns list of upcoming (not yet completed) interview rounds."""
        return [r for r in self.interview_rounds if r.is_upcoming]

    @property
    def completed_rounds_count(self) -> int:
        """Number of interview rounds already completed."""
        return sum(1 for r in self.interview_rounds if r.completed_at)

    @property
    def has_cover_letter(self) -> bool:
        """True if a cover letter has been generated."""
        return bool(self.cover_letter_text)

    @property
    def has_ats_resume(self) -> bool:
        """True if a tailored ATS resume has been generated."""
        return bool(self.ats_resume_text)

    @property
    def has_interview_prep(self) -> bool:
        """True if Agent 7 has generated interview preparation."""
        return bool(self.interview_prep_data)

    # ----------------------------------------------------------
    # State Machine Methods
    # ----------------------------------------------------------

    def update_status(self, new_status: str) -> None:
        """
        Updates application status and appends to status history.

        Automatically:
        - Records timestamp of the change
        - Sets applied_at when status moves to 'applied'
        - Sets closed_at when status reaches a final state

        Args:
            new_status: One of ApplicationStatus constants.

        Raises:
            ValueError: If new_status is not a valid status value.
        """
        if new_status not in ApplicationStatus.ALL_STATUSES:
            raise ValueError(
                f"Invalid status '{new_status}'. "
                f"Valid values: {ApplicationStatus.ALL_STATUSES}"
            )

        from datetime import timezone
        now = datetime.now(timezone.utc)

        # Initialize history list if first update
        if not self.status_history:
            self.status_history = []

        # Append change to history
        self.status_history.append({
            "status": new_status,
            "changed_at": now.isoformat(),
            "previous_status": self.status,
        })

        # Update main status field
        self.status = new_status

        # Set applied_at timestamp on first application submission
        if new_status == ApplicationStatus.APPLIED and not self.applied_at:
            self.applied_at = now

        # Set closed_at when application reaches terminal state
        if new_status in ApplicationStatus.CLOSED_STATUSES:
            self.closed_at = now

    def add_interview_round(
        self,
        round_type: str,
        scheduled_at: Optional[datetime] = None,
        platform: Optional[str] = None,
        interviewer_name: Optional[str] = None,
    ) -> "InterviewRound":
        """
        Creates and appends a new interview round to this application.

        Automatically assigns the correct round number based on
        existing rounds.

        Args:
            round_type:       Type of interview round.
            scheduled_at:     Scheduled datetime (UTC).
            platform:         Interview platform.
            interviewer_name: Name of the interviewer.

        Returns:
            InterviewRound: The newly created round object.
        """
        next_round_number = len(self.interview_rounds) + 1

        new_round = InterviewRound(
            application_id=self.id,
            round_number=next_round_number,
            round_type=round_type,
            scheduled_at=scheduled_at,
            platform=platform,
            interviewer_name=interviewer_name,
            outcome="pending",
        )

        self.interview_rounds.append(new_round)

        # Auto-update application status to 'interviewing'
        if self.status == ApplicationStatus.APPLIED:
            self.update_status(ApplicationStatus.INTERVIEWING)

        return new_round

    def set_offer(
        self,
        salary: Optional[str] = None,
        offer_details: Optional[Dict[str, Any]] = None,
        deadline: Optional[date] = None,
    ) -> None:
        """
        Records an offer received for this application.

        Args:
            salary:        Offered salary string e.g. "25 LPA".
            offer_details: Full offer package as JSON dict.
            deadline:      Date to accept/decline by.
        """
        from datetime import timezone
        self.offer_salary = salary
        self.offer_details = offer_details
        self.offer_deadline = deadline
        self.offer_received_at = datetime.now(timezone.utc)
        self.update_status(ApplicationStatus.OFFERED)

    def __repr__(self) -> str:
        return (
            f"<Application id={self.id} "
            f"user_id={self.user_id} "
            f"status='{self.status}' "
            f"priority={self.priority}>"
        )


# ============================================================
# Pydantic Schemas — InterviewRound
# ============================================================

class InterviewRoundCreate(BaseModel):
    """Schema to add a new interview round to an application."""

    round_type: str = Field(
        ...,
        description=(
            "Interview type: 'online_assessment' | 'phone_screen' | "
            "'technical' | 'system_design' | 'behavioral' | "
            "'hr' | 'managerial' | 'final'"
        ),
    )
    scheduled_at: Optional[datetime] = Field(
        None,
        description="Scheduled date and time (UTC)",
    )
    duration_minutes: Optional[int] = Field(
        None, ge=5, le=480,
        description="Expected duration in minutes",
    )
    interviewer_name: Optional[str] = Field(None, max_length=255)
    interviewer_linkedin: Optional[str] = Field(None, max_length=500)
    platform: Optional[str] = Field(None, max_length=100)
    meeting_link: Optional[str] = Field(None, max_length=2000)


class InterviewRoundUpdate(BaseModel):
    """Schema to update an existing interview round (e.g. after completion)."""

    scheduled_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_minutes: Optional[int] = Field(None, ge=5, le=480)
    interviewer_name: Optional[str] = Field(None, max_length=255)
    platform: Optional[str] = Field(None, max_length=100)
    meeting_link: Optional[str] = Field(None, max_length=2000)
    outcome: Optional[str] = Field(
        None,
        description="Result: 'passed' | 'failed' | 'pending' | 'cancelled'",
    )
    feedback: Optional[str] = None
    questions_asked: Optional[List[str]] = None

    @field_validator("outcome")
    @classmethod
    def validate_outcome(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        valid = {"passed", "failed", "pending", "cancelled"}
        if v.lower() not in valid:
            raise ValueError(f"outcome must be one of: {valid}")
        return v.lower()


class InterviewRoundResponse(BaseModel):
    """Full interview round response."""

    id: uuid.UUID
    application_id: uuid.UUID
    round_number: int
    round_type: str
    scheduled_at: Optional[datetime]
    completed_at: Optional[datetime]
    duration_minutes: Optional[int]
    interviewer_name: Optional[str]
    interviewer_linkedin: Optional[str]
    platform: Optional[str]
    meeting_link: Optional[str]
    outcome: Optional[str]
    feedback: Optional[str]
    questions_asked: Optional[List[str]]
    is_upcoming: bool
    is_passed: bool
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


# ============================================================
# Pydantic Schemas — Application
# ============================================================

class ApplicationCreate(BaseModel):
    """
    Schema for creating a new application.
    Called by Agent 8 or directly by the user.
    """

    job_id: uuid.UUID = Field(..., description="Job to apply for")
    resume_id: Optional[uuid.UUID] = Field(
        None,
        description="Resume to attach — defaults to active resume",
    )
    status: str = Field(
        default=ApplicationStatus.SAVED,
        description="Initial status — usually 'saved' or 'applied'",
    )
    application_url: Optional[str] = Field(None, max_length=2000)
    application_platform: Optional[str] = Field(None, max_length=100)
    referral_name: Optional[str] = Field(None, max_length=255)
    notes: Optional[str] = None
    priority: int = Field(
        default=2,
        ge=1,
        le=3,
        description="Priority: 1=High, 2=Medium, 3=Low",
    )

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        if v not in ApplicationStatus.ALL_STATUSES:
            raise ValueError(
                f"Invalid status '{v}'. "
                f"Valid: {ApplicationStatus.ALL_STATUSES}"
            )
        return v


class ApplicationUpdate(BaseModel):
    """
    Schema for updating an application.
    All fields optional — only provided fields are updated.
    """

    status: Optional[str] = None
    application_url: Optional[str] = Field(None, max_length=2000)
    application_platform: Optional[str] = Field(None, max_length=100)
    referral_name: Optional[str] = Field(None, max_length=255)
    notes: Optional[str] = None
    rejection_reason: Optional[str] = None
    follow_up_date: Optional[date] = None
    priority: Optional[int] = Field(None, ge=1, le=3)
    is_starred: Optional[bool] = None
    offer_salary: Optional[str] = Field(None, max_length=255)
    offer_deadline: Optional[date] = None
    offer_details: Optional[Dict[str, Any]] = None

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        if v not in ApplicationStatus.ALL_STATUSES:
            raise ValueError(
                f"Invalid status '{v}'. "
                f"Valid: {ApplicationStatus.ALL_STATUSES}"
            )
        return v


class ApplicationResponse(BaseModel):
    """
    Full application response — includes all fields and relationships.
    Returned by GET /tracker/{id}.
    """

    id: uuid.UUID
    user_id: uuid.UUID
    job_id: Optional[uuid.UUID]
    resume_id: Optional[uuid.UUID]
    status: str
    status_history: Optional[List[Dict[str, Any]]]
    applied_at: Optional[datetime]
    application_url: Optional[str]
    application_platform: Optional[str]
    referral_name: Optional[str]
    cover_letter_text: Optional[str]
    ats_resume_text: Optional[str]
    cover_letter_generated_at: Optional[datetime]
    ats_resume_generated_at: Optional[datetime]
    interview_prep_data: Optional[Dict[str, Any]]
    interview_prep_generated_at: Optional[datetime]
    follow_up_date: Optional[date]
    last_follow_up_at: Optional[datetime]
    follow_up_count: int
    offer_salary: Optional[str]
    offer_received_at: Optional[datetime]
    offer_deadline: Optional[date]
    offer_details: Optional[Dict[str, Any]]
    notes: Optional[str]
    rejection_reason: Optional[str]
    priority: int
    is_starred: bool
    is_active: bool
    is_closed: bool
    is_positive_outcome: bool
    has_cover_letter: bool
    has_ats_resume: bool
    has_interview_prep: bool
    days_since_applied: Optional[int]
    completed_rounds_count: int
    interview_rounds: List[InterviewRoundResponse]
    created_at: datetime
    updated_at: datetime
    closed_at: Optional[datetime]

    model_config = {"from_attributes": True}


class ApplicationListItem(BaseModel):
    """
    Compact application info for dashboard list view.
    Returned by GET /tracker/ — shows all applications.
    """

    id: uuid.UUID
    job_id: Optional[uuid.UUID]
    status: str
    priority: int
    is_starred: bool
    is_active: bool
    applied_at: Optional[datetime]
    days_since_applied: Optional[int]
    follow_up_date: Optional[date]
    has_cover_letter: bool
    has_interview_prep: bool
    completed_rounds_count: int
    upcoming_interview_count: int
    created_at: datetime

    model_config = {"from_attributes": True}


class ApplicationStats(BaseModel):
    """
    Aggregated statistics for the user's application dashboard.
    Returned by GET /tracker/stats.
    Powers the dashboard summary cards.
    """

    total_applications: int = Field(..., description="Total applications created")
    active_applications: int = Field(..., description="Currently in-progress")
    saved_count: int = Field(..., description="Bookmarked but not applied")
    applied_count: int = Field(..., description="Applications submitted")
    interviewing_count: int = Field(..., description="In interview process")
    offered_count: int = Field(..., description="Offers received")
    accepted_count: int = Field(..., description="Offers accepted")
    rejected_count: int = Field(..., description="Applications rejected")
    withdrawn_count: int = Field(..., description="Applications withdrawn")
    no_response_count: int = Field(..., description="No response received")
    response_rate: float = Field(
        ...,
        description=(
            "Percentage of applications that got a response "
            "(screening or beyond)"
        ),
    )
    offer_rate: float = Field(
        ...,
        description="Percentage of applications that resulted in an offer",
    )
    upcoming_interviews: int = Field(
        ...,
        description="Number of interviews scheduled in the future",
    )
    avg_days_to_response: Optional[float] = Field(
        None,
        description="Average days from application to first response",
    )

    model_config = {"from_attributes": True}


class TrackerDashboardResponse(BaseModel):
    """
    Complete tracker dashboard response.
    Returned by GET /tracker/dashboard.
    Combines stats + recent applications + upcoming interviews.
    """

    stats: ApplicationStats
    recent_applications: List[ApplicationListItem]
    upcoming_interviews: List[InterviewRoundResponse]
    starred_applications: List[ApplicationListItem]