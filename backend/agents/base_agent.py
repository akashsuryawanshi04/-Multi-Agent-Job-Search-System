# ============================================================
# File: backend/agents/base_agent.py
# Purpose: Abstract base class for all 8 pipeline agents.
#          Provides shared lifecycle, logging, error handling,
#          service access, timing, and token tracking.
#
# All agents inherit from BaseAgent:
#   - ResumeParserAgent   (Agent 1)
#   - JobSearchAgent      (Agent 2)
#   - JobMatchingAgent    (Agent 3)
#   - SkillGapAgent       (Agent 4)
#   - ATSOptimizerAgent   (Agent 5)
#   - CoverLetterAgent    (Agent 6)
#   - InterviewPrepAgent  (Agent 7)
#   - TrackerAgent        (Agent 8)
#
# Pattern:
#   Each agent MUST implement _execute() — the core AI logic.
#   BaseAgent.run() wraps _execute() with logging, timing,
#   error handling, and pipeline step reporting.
# ============================================================

import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Generic, List, Optional, TypeVar

from backend.services.embedding_service import (
    EmbeddingService,
    embedding_service,
)
from backend.services.llm_service import (
    AgentType,
    LLMError,
    LLMService,
    TokenUsageTracker,
    llm_service,
)
from backend.services.pdf_service import PDFService, pdf_service
from backend.utils.logger import get_agent_logger, log_pipeline_step

# Generic type variable for agent input and output
InputT  = TypeVar("InputT")
OutputT = TypeVar("OutputT")


# ============================================================
# Agent Result Wrapper
# ============================================================

@dataclass
class AgentResult(Generic[OutputT]):
    """
    Standardized result wrapper returned by every agent's run().

    Every agent returns AgentResult — not raw data — so the
    orchestrator always has a consistent interface regardless
    of which agent ran.

    Attributes:
        success:      True if agent completed without error.
        output:       The agent's actual output data (typed).
        agent_name:   Human-readable agent identifier.
        agent_type:   AgentType enum value for this agent.
        duration_ms:  Wall-clock time taken to run (ms).
        token_usage:  Claude API token usage summary.
        error:        Error message if success=False.
        error_type:   Exception class name for error classification.
        warnings:     Non-fatal issues encountered.
        metadata:     Agent-specific extra info for debugging.
        started_at:   UTC timestamp when agent started.
        completed_at: UTC timestamp when agent finished.
    """
    success:      bool
    output:       Optional[OutputT]
    agent_name:   str
    agent_type:   AgentType
    duration_ms:  float
    token_usage:  Dict[str, Any]
    error:        Optional[str]        = None
    error_type:   Optional[str]        = None
    warnings:     List[str]            = field(default_factory=list)
    metadata:     Dict[str, Any]       = field(default_factory=dict)
    started_at:   Optional[datetime]   = None
    completed_at: Optional[datetime]   = None

    @property
    def duration_seconds(self) -> float:
        """Duration in seconds (rounded to 2 decimal places)."""
        return round(self.duration_ms / 1000, 2)

    @property
    def is_failed(self) -> bool:
        """True if the agent did not complete successfully."""
        return not self.success

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the result for API responses and logging.
        Excludes the raw output (too large for logs).
        """
        return {
            "success":      self.success,
            "agent_name":   self.agent_name,
            "agent_type":   self.agent_type.value,
            "duration_ms":  round(self.duration_ms, 2),
            "duration_s":   self.duration_seconds,
            "token_usage":  self.token_usage,
            "error":        self.error,
            "error_type":   self.error_type,
            "warnings":     self.warnings,
            "metadata":     self.metadata,
            "started_at":   self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }

    def __repr__(self) -> str:
        status = "✓" if self.success else "✗"
        return (
            f"<AgentResult {status} "
            f"agent={self.agent_name} "
            f"duration={self.duration_seconds}s "
            f"tokens={self.token_usage.get('total_tokens', 0)}>"
        )


# ============================================================
# Agent Exceptions
# ============================================================

class AgentError(Exception):
    """
    Base exception for all agent-level errors.

    Raised inside _execute() when the agent encounters an
    unrecoverable error. BaseAgent.run() catches this and
    returns AgentResult(success=False, error=...).

    Usage in agents:
        raise AgentError("Resume file is empty — cannot parse.")
    """
    pass


class AgentInputError(AgentError):
    """
    Raised when agent input is invalid or missing.

    Examples:
        - Resume file path is None
        - Job description is empty string
        - Required field missing from parsed data
    """
    pass


class AgentOutputError(AgentError):
    """
    Raised when agent output is invalid or incomplete.

    Examples:
        - Claude returned JSON missing required fields
        - Similarity score is outside [0, 1]
        - Generated cover letter is empty
    """
    pass


class AgentTimeoutError(AgentError):
    """
    Raised when agent execution exceeds the time limit.
    """
    pass


# ============================================================
# Base Agent Class
# ============================================================

class BaseAgent(ABC):
    """
    Abstract base class for all 8 Job Search AI pipeline agents.

    Subclasses MUST implement:
        _execute(input_data, **kwargs) → OutputT

    Subclasses MAY override:
        validate_input(input_data)  → raise AgentInputError if invalid
        validate_output(output)     → raise AgentOutputError if invalid
        on_success(result)          → hook called after successful run
        on_failure(result)          → hook called after failed run

    Provided automatically (do NOT override):
        run(input_data, **kwargs)   → AgentResult[OutputT]
        llm                         → LLMService singleton
        pdf                         → PDFService singleton
        embeddings                  → EmbeddingService singleton
        logger                      → Agent-specific loguru logger
        tracker                     → TokenUsageTracker for this run

    Example subclass:
        class ResumeParserAgent(BaseAgent):
            agent_name = "ResumeParserAgent"
            agent_type = AgentType.RESUME_PARSER
            step_number = 1
            total_steps = 8

            async def _execute(
                self,
                input_data: Resume,
                **kwargs,
            ) -> ResumeParseResult:
                text = await self.pdf.extract(input_data.file_path)
                data = await self.llm.complete_json(
                    agent_type=self.agent_type,
                    user_message=f"Parse:\\n{text.text}",
                    usage_tracker=self.tracker,
                )
                return ResumeParseResult(**data)
    """

    # ----------------------------------------------------------
    # Class-level attributes — override in each subclass
    # ----------------------------------------------------------

    # Human-readable name used in logs and responses
    agent_name: str = "BaseAgent"

    # AgentType enum value — used for Claude system prompt selection
    agent_type: AgentType = AgentType.GENERAL

    # Position in the pipeline (1-8) — used for step logging
    step_number: int = 0

    # Total steps in the full pipeline — used for step logging
    total_steps: int = 8

    # Maximum time (seconds) the agent is allowed to run
    # Agents that exceed this raise AgentTimeoutError
    timeout_seconds: int = 120

    # Number of times to retry _execute() on AgentError
    # (not LLMError — those are retried inside llm_service)
    max_retries: int = 1

    # ----------------------------------------------------------
    # Constructor
    # ----------------------------------------------------------

    def __init__(self) -> None:
        """
        Initializes the agent with shared service references
        and a fresh logger bound to this agent's name.

        Services are module-level singletons — not recreated
        per agent instance. All agents share the same LLM client,
        PDF extractor, and embedding model.
        """
        # Agent-specific logger — every log line includes agent name
        self.logger = get_agent_logger(self.agent_name)

        # Shared service singletons
        self.llm:        LLMService        = llm_service
        self.pdf:        PDFService        = pdf_service
        self.embeddings: EmbeddingService  = embedding_service

        # Fresh token tracker per agent instance
        # If running in a pipeline, the orchestrator passes its
        # shared tracker via run(usage_tracker=...)
        self.tracker: TokenUsageTracker = TokenUsageTracker()

        # Warnings accumulated during _execute()
        # Agents call self._warn("message") to add non-fatal issues
        self._warnings: List[str] = []

        # Extra metadata added by subclasses for debugging
        self._metadata: Dict[str, Any] = {}

        self.logger.debug(
            f"Agent initialized | "
            f"step={self.step_number}/{self.total_steps} | "
            f"timeout={self.timeout_seconds}s"
        )

    # ----------------------------------------------------------
    # Public Interface — run()
    # ----------------------------------------------------------

    async def run(
        self,
        input_data: Any,
        usage_tracker: Optional[TokenUsageTracker] = None,
        **kwargs: Any,
    ) -> AgentResult:
        """
        Executes the agent with full lifecycle management.

        This is the ONLY method the orchestrator calls.
        Never call _execute() directly from outside the agent.

        Lifecycle:
            1. Reset state (warnings, metadata)
            2. Log pipeline step start
            3. Validate input
            4. Run _execute() with timeout + retry
            5. Validate output
            6. Log pipeline step completion
            7. Call on_success() hook
            8. Return AgentResult(success=True, output=...)

            On any error:
            → Log pipeline step failure
            → Call on_failure() hook
            → Return AgentResult(success=False, error=...)

        Args:
            input_data:    The data this agent processes.
                           Type depends on the specific agent.
            usage_tracker: Shared TokenUsageTracker from the
                           orchestrator. If provided, this agent's
                           token usage is recorded into it.
            **kwargs:      Additional keyword arguments passed
                           through to _execute().

        Returns:
            AgentResult: Always returns — never raises.
            Check result.success before using result.output.

        Usage by orchestrator:
            result = await agent.run(
                input_data=resume,
                usage_tracker=pipeline_tracker,
                job_preferences=prefs,
            )
            if result.is_failed:
                logger.error(result.error)
                pipeline.abort(result)
            else:
                next_input = result.output
        """
        # Reset per-run state
        self._warnings = []
        self._metadata = {}

        # Use shared tracker if provided, else use own tracker
        if usage_tracker:
            self.tracker = usage_tracker

        # Record start time
        started_at = datetime.now(timezone.utc)
        start_perf = time.perf_counter()

        # Log pipeline step start
        log_pipeline_step(
            step=self.step_number,
            total=self.total_steps,
            agent_name=self.agent_name,
            status="started",
        )

        self.logger.info(
            f"Agent starting | "
            f"input_type={type(input_data).__name__}"
        )

        try:
            # ── Step 1: Validate input ──────────────────────────
            self.validate_input(input_data)

            # ── Step 2: Execute with timeout + retry ───────────
            output = await self._run_with_timeout_and_retry(
                input_data=input_data,
                **kwargs,
            )

            # ── Step 3: Validate output ─────────────────────────
            self.validate_output(output)

            # ── Step 4: Build success result ───────────────────
            duration_ms = (time.perf_counter() - start_perf) * 1000
            completed_at = datetime.now(timezone.utc)

            result: AgentResult = AgentResult(
                success=True,
                output=output,
                agent_name=self.agent_name,
                agent_type=self.agent_type,
                duration_ms=round(duration_ms, 2),
                token_usage=self.tracker.summary(),
                warnings=list(self._warnings),
                metadata=dict(self._metadata),
                started_at=started_at,
                completed_at=completed_at,
            )

            # Log success
            log_pipeline_step(
                step=self.step_number,
                total=self.total_steps,
                agent_name=self.agent_name,
                status="completed",
                detail=(
                    f"duration={result.duration_seconds}s | "
                    f"tokens={self.tracker.total_tokens}"
                ),
            )

            self.logger.info(
                f"Agent completed successfully | "
                f"duration={result.duration_seconds}s | "
                f"tokens={self.tracker.total_tokens} | "
                f"warnings={len(self._warnings)}"
            )

            # Call success hook (subclasses may override)
            await self.on_success(result)

            return result

        except AgentInputError as e:
            return await self._build_failure_result(
                error=str(e),
                error_type="AgentInputError",
                start_perf=start_perf,
                started_at=started_at,
            )

        except AgentOutputError as e:
            return await self._build_failure_result(
                error=str(e),
                error_type="AgentOutputError",
                start_perf=start_perf,
                started_at=started_at,
            )

        except AgentTimeoutError as e:
            return await self._build_failure_result(
                error=str(e),
                error_type="AgentTimeoutError",
                start_perf=start_perf,
                started_at=started_at,
            )

        except AgentError as e:
            return await self._build_failure_result(
                error=str(e),
                error_type="AgentError",
                start_perf=start_perf,
                started_at=started_at,
            )

        except LLMError as e:
            # LLM errors bubble up from llm_service after all retries
            return await self._build_failure_result(
                error=f"Claude API error: {str(e)}",
                error_type="LLMError",
                start_perf=start_perf,
                started_at=started_at,
            )

        except Exception as e:
            # Catch-all — unexpected errors should never crash pipeline
            tb = traceback.format_exc()
            self.logger.error(
                f"Unexpected error in agent: "
                f"{type(e).__name__}: {e}\n{tb}"
            )
            return await self._build_failure_result(
                error=f"Unexpected error: {type(e).__name__}: {str(e)}",
                error_type=type(e).__name__,
                start_perf=start_perf,
                started_at=started_at,
            )

    # ----------------------------------------------------------
    # Abstract Method — implement in each agent
    # ----------------------------------------------------------

    @abstractmethod
    async def _execute(
        self,
        input_data: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Core agent logic — must be implemented by every subclass.

        This is where each agent does its actual work:
            - Agent 1: Parse resume with pdf_service + llm
            - Agent 2: Search jobs with scraper_service
            - Agent 3: Score matches with embedding_service
            - Agent 4: Analyze skill gaps with llm
            - Agent 5: Optimize ATS resume with llm
            - Agent 6: Generate cover letter with llm
            - Agent 7: Generate interview prep with llm
            - Agent 8: Update application tracker with db

        Rules for implementing _execute():
            ✓ Use self.llm for all Claude API calls
            ✓ Use self.pdf for all file extraction
            ✓ Use self.embeddings for all similarity scoring
            ✓ Use self.logger for all logging
            ✓ Use self.tracker for token usage tracking
            ✓ Call self._warn() for non-fatal issues
            ✓ Call self._set_metadata() to attach debug info
            ✓ Raise AgentError (or subclass) on failure
            ✗ Never raise generic Exception — always AgentError
            ✗ Never call run() recursively

        Args:
            input_data: The agent's input. Type varies per agent.
            **kwargs:   Additional arguments from run().

        Returns:
            The agent's output. Type varies per agent.

        Raises:
            AgentInputError:  Bad input data.
            AgentOutputError: Bad output from AI or processing.
            AgentError:       Any other unrecoverable failure.
        """
        ...

    # ----------------------------------------------------------
    # Hooks — override in subclasses if needed
    # ----------------------------------------------------------

    def validate_input(self, input_data: Any) -> None:
        """
        Validates the agent's input before _execute() is called.

        Default: passes all input without validation.
        Override to add agent-specific input checks.

        Args:
            input_data: The input to validate.

        Raises:
            AgentInputError: If input is invalid.

        Example override:
            def validate_input(self, input_data: Resume) -> None:
                if not input_data.file_path:
                    raise AgentInputError(
                        "Resume file_path is required."
                    )
                if not Path(input_data.file_path).exists():
                    raise AgentInputError(
                        f"Resume file not found: {input_data.file_path}"
                    )
        """
        pass

    def validate_output(self, output: Any) -> None:
        """
        Validates the agent's output after _execute() returns.

        Default: passes all output without validation.
        Override to add agent-specific output checks.

        Args:
            output: The output from _execute() to validate.

        Raises:
            AgentOutputError: If output is invalid.

        Example override:
            def validate_output(self, output: ResumeParseResult) -> None:
                if not output.all_skills_flat:
                    raise AgentOutputError(
                        "Parsed resume has no skills — "
                        "parsing may have failed silently."
                    )
        """
        pass

    async def on_success(self, result: AgentResult) -> None:
        """
        Hook called after a successful run().

        Default: does nothing.
        Override to add post-success actions like:
            - Sending WebSocket notifications
            - Updating database status fields
            - Triggering downstream agents

        Args:
            result: The completed AgentResult.
        """
        pass

    async def on_failure(self, result: AgentResult) -> None:
        """
        Hook called after a failed run().

        Default: does nothing.
        Override to add failure actions like:
            - Marking DB records as failed
            - Sending error notifications
            - Triggering fallback logic

        Args:
            result: The failed AgentResult (success=False).
        """
        pass

    # ----------------------------------------------------------
    # Protected Helpers — use inside _execute()
    # ----------------------------------------------------------

    def _warn(self, message: str) -> None:
        """
        Adds a non-fatal warning to the current run's warnings list.

        Warnings are included in AgentResult.warnings and logged
        at WARNING level. They do NOT cause run() to fail.

        Use when:
            - Something is unusual but recoverable
            - Data is incomplete but processable
            - A fallback path was taken

        Args:
            message: Warning description.

        Usage:
            if not job.salary_display:
                self._warn("Job listing has no salary information.")
        """
        self._warnings.append(message)
        self.logger.warning(f"[Warning] {message}")

    def _set_metadata(self, key: str, value: Any) -> None:
        """
        Attaches key-value metadata to the current run.

        Metadata is included in AgentResult.metadata and appears
        in logs. Useful for debugging and performance analysis.

        Use for:
            - Recording counts (jobs_found, skills_matched)
            - Recording intermediate scores
            - Flagging which code path was taken

        Args:
            key:   Metadata key (short snake_case string).
            value: Any JSON-serializable value.

        Usage:
            self._set_metadata("jobs_found", len(jobs))
            self._set_metadata("extraction_method", "pdfplumber")
            self._set_metadata("cache_hit", True)
        """
        self._metadata[key] = value
        self.logger.debug(f"[Metadata] {key}={value!r}")

    def _require(
        self,
        value: Any,
        field_name: str,
        context: str = "",
    ) -> Any:
        """
        Asserts a required value is not None or empty.
        Raises AgentInputError with a clear message if it is.

        Shorthand for the most common input validation pattern.

        Args:
            value:      The value to check.
            field_name: Name of the field (for error message).
            context:    Optional extra context for the error.

        Returns:
            The value unchanged (for chaining).

        Raises:
            AgentInputError: If value is None or empty string/list.

        Usage:
            resume_text = self._require(
                result.text,
                "resume text",
                "PDF extraction returned empty content",
            )
        """
        is_empty = (
            value is None
            or value == ""
            or value == []
            or value == {}
        )
        if is_empty:
            msg = f"Required field '{field_name}' is missing or empty."
            if context:
                msg += f" {context}"
            raise AgentInputError(msg)
        return value

    def _require_json_field(
        self,
        data: Dict[str, Any],
        field: str,
        default: Any = None,
    ) -> Any:
        """
        Safely extracts a field from Claude's JSON response.

        If the field is missing, returns the default value and
        logs a warning. This prevents KeyErrors from crashing
        the agent when Claude's JSON output is incomplete.

        Args:
            data:    The parsed JSON dict from Claude.
            field:   Key to extract.
            default: Value to return if key is missing.

        Returns:
            The field value or default.

        Usage:
            skills = self._require_json_field(
                data=parsed_json,
                field="all_skills_flat",
                default=[],
            )
        """
        value = data.get(field, default)
        if value is None and default is not None:
            self._warn(
                f"Claude's response was missing field '{field}'. "
                f"Using default: {default!r}"
            )
            return default
        return value

    def _truncate_text(
        self,
        text: str,
        max_chars: int = 12_000,
        label: str = "text",
    ) -> str:
        """
        Truncates text to stay within a character limit.

        Used to prevent context window overflows when passing
        long job descriptions or resume text to Claude.

        Args:
            text:      The text to possibly truncate.
            max_chars: Maximum number of characters to keep.
            label:     Description for the warning message.

        Returns:
            str: Original text or truncated version.
        """
        if len(text) <= max_chars:
            return text

        truncated = text[:max_chars]
        # Try to break at a sentence or word boundary
        last_period = truncated.rfind(".")
        if last_period > max_chars - 500:
            truncated = truncated[:last_period + 1]

        self._warn(
            f"{label} was truncated from {len(text):,} to "
            f"{len(truncated):,} characters to fit context window."
        )
        return truncated + "\n\n[Truncated]"

    # ----------------------------------------------------------
    # Internal Helpers
    # ----------------------------------------------------------

    async def _run_with_timeout_and_retry(
        self,
        input_data: Any,
        **kwargs: Any,
    ) -> Any:
        """
        Runs _execute() with timeout enforcement and retry logic.

        Retry policy:
            - Retries on AgentError (transient failures)
            - Does NOT retry on AgentInputError or AgentOutputError
              (these are deterministic — retrying won't help)
            - Max retries configured per agent via max_retries

        Timeout policy:
            - Raises AgentTimeoutError if _execute() takes
              longer than self.timeout_seconds

        Args:
            input_data: Passed through to _execute().
            **kwargs:   Passed through to _execute().

        Returns:
            Output from _execute().

        Raises:
            AgentTimeoutError: If timeout exceeded.
            AgentError:        If all retries exhausted.
            AgentInputError:   Never retried — raised immediately.
            AgentOutputError:  Never retried — raised immediately.
        """
        import asyncio

        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):

            if attempt > 1:
                # Exponential backoff between retries
                wait_seconds = 2 ** (attempt - 1)
                self.logger.warning(
                    f"Retry {attempt}/{self.max_retries} | "
                    f"waiting {wait_seconds}s | "
                    f"last_error={last_error}"
                )
                await asyncio.sleep(wait_seconds)

            try:
                # Enforce timeout using asyncio.wait_for
                output = await asyncio.wait_for(
                    self._execute(input_data, **kwargs),
                    timeout=self.timeout_seconds,
                )
                return output

            except asyncio.TimeoutError:
                raise AgentTimeoutError(
                    f"{self.agent_name} timed out after "
                    f"{self.timeout_seconds} seconds. "
                    f"Consider increasing timeout_seconds or "
                    f"reducing input size."
                )

            except (AgentInputError, AgentOutputError):
                # Deterministic errors — do not retry
                raise

            except AgentError as e:
                last_error = e
                if attempt == self.max_retries:
                    raise
                self.logger.warning(
                    f"AgentError on attempt {attempt}: {e} — will retry"
                )

        # Should not reach here — raise is in the loop
        raise AgentError(
            f"{self.agent_name} failed after "
            f"{self.max_retries} attempts. "
            f"Last error: {last_error}"
        )

    async def _build_failure_result(
        self,
        error: str,
        error_type: str,
        start_perf: float,
        started_at: datetime,
    ) -> AgentResult:
        """
        Builds a failed AgentResult and triggers on_failure() hook.

        Called by run() whenever any exception is caught.

        Args:
            error:      Human-readable error description.
            error_type: Exception class name.
            start_perf: perf_counter() value from run() start.
            started_at: UTC datetime of run() start.

        Returns:
            AgentResult with success=False.
        """
        duration_ms = (time.perf_counter() - start_perf) * 1000
        completed_at = datetime.now(timezone.utc)

        log_pipeline_step(
            step=self.step_number,
            total=self.total_steps,
            agent_name=self.agent_name,
            status="failed",
            detail=f"error_type={error_type} | {error[:100]}",
        )

        self.logger.error(
            f"Agent failed | "
            f"error_type={error_type} | "
            f"duration={round(duration_ms/1000, 2)}s | "
            f"error={error}"
        )

        result: AgentResult = AgentResult(
            success=False,
            output=None,
            agent_name=self.agent_name,
            agent_type=self.agent_type,
            duration_ms=round(duration_ms, 2),
            token_usage=self.tracker.summary(),
            error=error,
            error_type=error_type,
            warnings=list(self._warnings),
            metadata=dict(self._metadata),
            started_at=started_at,
            completed_at=completed_at,
        )

        # Call failure hook (subclasses may override)
        await self.on_failure(result)

        return result

    # ----------------------------------------------------------
    # Representation
    # ----------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"<{self.agent_name} "
            f"step={self.step_number}/{self.total_steps} "
            f"timeout={self.timeout_seconds}s>"
        )