# ============================================================
# File: backend/models/user.py
# Purpose: User SQLAlchemy ORM model + Pydantic schemas.
#          Root entity — all other models belong to a User.
#
# Used by:
#   - backend/db/database.py          → imported in init_db()
#   - backend/models/resume.py        → User.id foreign key
#   - backend/models/application.py   → User.id foreign key
#   - backend/db/repositories/        → user_repo.py CRUD
#   - backend/api/routes/auth.py      → register, login, profile
#   - backend/api/middleware.py       → JWT token validation
# ============================================================

import uuid
from datetime import datetime, timezone
from typing import Optional

from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr, Field, field_validator
from sqlalchemy import (
    Boolean,
    DateTime,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.db.database import Base


# ============================================================
# Password Hashing Context
# ============================================================
# bcrypt is the industry standard for password hashing.
# CryptContext handles hashing + verification + scheme upgrades.
# deprecated="auto" means old hashes are auto-upgraded on login.

_pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__rounds=12,      # Cost factor — higher = slower = more secure
)


def hash_password(plain_password: str) -> str:
    """
    Hashes a plain-text password using bcrypt.

    Args:
        plain_password: The raw password string from the user.

    Returns:
        str: The bcrypt hash to store in the database.

    Usage:
        hashed = hash_password("mySecret123")
    """
    return _pwd_context.hash(plain_password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verifies a plain-text password against a stored bcrypt hash.

    Args:
        plain_password:  The raw password submitted by the user.
        hashed_password: The bcrypt hash stored in the database.

    Returns:
        bool: True if the password matches, False otherwise.

    Usage:
        is_valid = verify_password("mySecret123", user.hashed_password)
    """
    return _pwd_context.verify(plain_password, hashed_password)


# ============================================================
# SQLAlchemy ORM Model
# ============================================================

class User(Base):
    """
    User table — stores all registered users.

    Table name: users

    Relationships:
        - resumes       → List[Resume]      (one user → many resumes)
        - applications  → List[Application] (one user → many applications)

    All timestamps are stored in UTC.
    UUID primary key avoids sequential ID guessing attacks.
    """

    __tablename__ = "users"

    # ----------------------------------------------------------
    # Primary Key
    # ----------------------------------------------------------
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
        index=True,
        comment="Unique user identifier (UUID v4)",
    )

    # ----------------------------------------------------------
    # Identity Fields
    # ----------------------------------------------------------
    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
        index=True,
        comment="User email address — used for login",
    )

    full_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="User's full display name",
    )

    hashed_password: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="bcrypt hashed password — never store plain text",
    )

    # ----------------------------------------------------------
    # Profile Fields
    # ----------------------------------------------------------
    phone: Mapped[Optional[str]] = mapped_column(
        String(20),
        nullable=True,
        comment="Optional phone number for contact",
    )

    location: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        comment="User's current city/country e.g. 'Pune, India'",
    )

    linkedin_url: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
        comment="LinkedIn profile URL",
    )

    github_url: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
        comment="GitHub profile URL",
    )

    portfolio_url: Mapped[Optional[str]] = mapped_column(
        String(500),
        nullable=True,
        comment="Personal portfolio or website URL",
    )

    bio: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="Short professional bio or summary",
    )

    # ----------------------------------------------------------
    # Job Preferences (used by Job Search Agent)
    # ----------------------------------------------------------
    preferred_role: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        comment="Target job title e.g. 'Backend Engineer'",
    )

    preferred_location: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        comment="Preferred job location e.g. 'Remote' or 'Bangalore'",
    )

    experience_years: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="Total years of professional experience",
    )

    expected_salary: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        comment="Expected salary range e.g. '15-20 LPA'",
    )

    # ----------------------------------------------------------
    # Account Status
    # ----------------------------------------------------------
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        comment="False = account deactivated or banned",
    )

    is_verified: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        comment="True = email has been verified",
    )

    is_superuser: Mapped[bool] = mapped_column(
        Boolean,
        default=False,
        nullable=False,
        comment="True = admin access to all data",
    )

    # ----------------------------------------------------------
    # Timestamps (all UTC)
    # ----------------------------------------------------------
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="When the account was created (UTC)",
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
        comment="When the account was last updated (UTC)",
    )

    last_login_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        comment="Timestamp of most recent login (UTC)",
    )

    # ----------------------------------------------------------
    # ORM Relationships
    # ----------------------------------------------------------
    # lazy="selectin" means SQLAlchemy automatically loads
    # related records with a SELECT IN query — works well with
    # async sessions (avoids the lazy loading async problem).

    resumes: Mapped[list["Resume"]] = relationship(  # noqa: F821
        "Resume",
        back_populates="user",
        cascade="all, delete-orphan",   # Deleting user deletes their resumes
        lazy="selectin",
    )

    applications: Mapped[list["Application"]] = relationship(  # noqa: F821
        "Application",
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    # ----------------------------------------------------------
    # Helper Methods
    # ----------------------------------------------------------

    def set_password(self, plain_password: str) -> None:
        """
        Hashes and stores a new password.

        Usage:
            user.set_password("newSecret123")
            await db.commit()
        """
        self.hashed_password = hash_password(plain_password)

    def check_password(self, plain_password: str) -> bool:
        """
        Verifies a submitted password against the stored hash.

        Usage:
            if not user.check_password(submitted_password):
                raise HTTPException(401, "Invalid credentials")
        """
        return verify_password(plain_password, self.hashed_password)

    def update_last_login(self) -> None:
        """
        Updates the last_login_at timestamp to now (UTC).
        Call this after successful authentication.
        """
        self.last_login_at = datetime.now(timezone.utc)

    def __repr__(self) -> str:
        return (
            f"<User id={self.id} "
            f"email={self.email} "
            f"active={self.is_active}>"
        )


# ============================================================
# Pydantic Schemas — Request / Response validation
# ============================================================
# These are separate from the ORM model.
# FastAPI uses these to validate incoming JSON and
# serialize outgoing responses.

class UserBase(BaseModel):
    """Shared fields used across multiple schemas."""

    email: EmailStr = Field(
        ...,
        description="Valid email address",
        examples=["user@example.com"],
    )
    full_name: str = Field(
        ...,
        min_length=2,
        max_length=255,
        description="Full display name",
        examples=["Rahul Sharma"],
    )


class UserCreate(UserBase):
    """
    Schema for POST /auth/register request body.
    Validates password strength before hashing.
    """

    password: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="Password — minimum 8 characters",
        examples=["StrongPass123!"],
    )
    confirm_password: str = Field(
        ...,
        description="Must match password field",
    )

    @field_validator("password")
    @classmethod
    def validate_password_strength(cls, v: str) -> str:
        """
        Enforces basic password strength rules:
        - At least 8 characters (handled by min_length)
        - At least one uppercase letter
        - At least one digit
        """
        if not any(c.isupper() for c in v):
            raise ValueError(
                "Password must contain at least one uppercase letter."
            )
        if not any(c.isdigit() for c in v):
            raise ValueError(
                "Password must contain at least one number."
            )
        return v

    @field_validator("confirm_password")
    @classmethod
    def passwords_must_match(cls, v: str, info) -> str:
        """Ensures confirm_password matches password."""
        if "password" in info.data and v != info.data["password"]:
            raise ValueError("Passwords do not match.")
        return v


class UserUpdate(BaseModel):
    """
    Schema for PATCH /users/me request body.
    All fields optional — only provided fields are updated.
    """

    full_name: Optional[str] = Field(
        default=None, min_length=2, max_length=255
    )
    phone: Optional[str] = Field(
        default=None, max_length=20
    )
    location: Optional[str] = Field(
        default=None, max_length=255
    )
    linkedin_url: Optional[str] = Field(
        default=None, max_length=500
    )
    github_url: Optional[str] = Field(
        default=None, max_length=500
    )
    portfolio_url: Optional[str] = Field(
        default=None, max_length=500
    )
    bio: Optional[str] = Field(
        default=None, max_length=2000
    )
    preferred_role: Optional[str] = Field(
        default=None, max_length=255
    )
    preferred_location: Optional[str] = Field(
        default=None, max_length=255
    )
    experience_years: Optional[int] = Field(
        default=None, ge=0, le=60
    )
    expected_salary: Optional[str] = Field(
        default=None, max_length=100
    )


class UserResponse(UserBase):
    """
    Schema for API responses — never exposes password hash.
    Returned by login, register, and profile endpoints.
    """

    id: uuid.UUID
    phone: Optional[str] = None
    location: Optional[str] = None
    linkedin_url: Optional[str] = None
    github_url: Optional[str] = None
    portfolio_url: Optional[str] = None
    bio: Optional[str] = None
    preferred_role: Optional[str] = None
    preferred_location: Optional[str] = None
    experience_years: Optional[int] = None
    expected_salary: Optional[str] = None
    is_active: bool
    is_verified: bool
    created_at: datetime
    updated_at: datetime
    last_login_at: Optional[datetime] = None

    model_config = {
        # Allows Pydantic to read attributes from SQLAlchemy ORM objects
        # instead of only from plain dictionaries
        "from_attributes": True
    }


class UserLogin(BaseModel):
    """
    Schema for POST /auth/login request body.
    """

    email: EmailStr = Field(
        ...,
        description="Registered email address",
    )
    password: str = Field(
        ...,
        min_length=1,
        description="Account password",
    )


class TokenResponse(BaseModel):
    """
    Schema for JWT token response after successful login.
    Returned by POST /auth/login and POST /auth/refresh.
    """

    access_token: str = Field(
        ...,
        description="JWT access token — include in Authorization header",
    )
    refresh_token: str = Field(
        ...,
        description="JWT refresh token — use to get new access token",
    )
    token_type: str = Field(
        default="bearer",
        description="Always 'bearer' — used in Authorization header",
    )
    expires_in: int = Field(
        ...,
        description="Access token lifetime in seconds",
    )

    model_config = {"from_attributes": True}


class PasswordChange(BaseModel):
    """
    Schema for POST /auth/change-password request body.
    """

    current_password: str = Field(
        ...,
        description="Current account password for verification",
    )
    new_password: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="New password — minimum 8 characters",
    )
    confirm_new_password: str = Field(
        ...,
        description="Must match new_password",
    )

    @field_validator("confirm_new_password")
    @classmethod
    def new_passwords_must_match(cls, v: str, info) -> str:
        if "new_password" in info.data and v != info.data["new_password"]:
            raise ValueError("New passwords do not match.")
        return v