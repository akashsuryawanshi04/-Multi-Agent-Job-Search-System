# ============================================================
# File: backend/services/llm_service.py
# Purpose: Centralized Claude AI service wrapper.
#          Single interface for all 8 agents to call Claude.
#
# Features:
#   - Async Anthropic SDK client (singleton)
#   - Automatic retry with exponential backoff (tenacity)
#   - Streaming support for long responses
#   - Structured JSON output parsing + validation
#   - Token usage tracking per agent call
#   - Detailed error classification and logging
#   - System prompt management per agent type
#   - Context window management
#
# Used by:
#   - backend/agents/base_agent.py              → all agents inherit
#   - backend/agents/resume_parser_agent.py     → Agent 1
#   - backend/agents/job_search_agent.py        → Agent 2
#   - backend/agents/job_matching_agent.py      → Agent 3
#   - backend/agents/skill_gap_agent.py         → Agent 4
#   - backend/agents/ats_optimizer_agent.py     → Agent 5
#   - backend/agents/cover_letter_agent.py      → Agent 6
#   - backend/agents/interview_prep_agent.py    → Agent 7
#   - backend/agents/tracker_agent.py           → Agent 8
# ============================================================

import json
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncGenerator, Dict, List, Optional, Type, Union

import anthropic
from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from backend.config import settings
from backend.utils.logger import get_service_logger


# ============================================================
# Module Logger
# ============================================================

logger = get_service_logger("LLMService")


# ============================================================
# Agent Type Enum
# ============================================================

class AgentType(str, Enum):
    """
    Identifies which agent is making a Claude API call.
    Used for logging, token tracking, and system prompt selection.
    """
    RESUME_PARSER    = "resume_parser"
    JOB_SEARCH       = "job_search"
    JOB_MATCHING     = "job_matching"
    SKILL_GAP        = "skill_gap"
    ATS_OPTIMIZER    = "ats_optimizer"
    COVER_LETTER     = "cover_letter"
    INTERVIEW_PREP   = "interview_prep"
    TRACKER          = "tracker"
    GENERAL          = "general"


# ============================================================
# Data Classes
# ============================================================

@dataclass
class LLMMessage:
    """
    A single message in the conversation with Claude.

    Attributes:
        role:    "user" or "assistant"
        content: The message text content
    """
    role: str       # "user" | "assistant"
    content: str

    def to_dict(self) -> Dict[str, str]:
        """Converts to the dict format required by Anthropic SDK."""
        return {"role": self.role, "content": self.content}


@dataclass
class LLMResponse:
    """
    Structured response from a Claude API call.

    Attributes:
        content:        The full text response from Claude
        input_tokens:   Number of tokens in the prompt
        output_tokens:  Number of tokens in the response
        model:          Model name that generated the response
        agent_type:     Which agent made this call
        duration_ms:    How long the API call took in milliseconds
        parsed_json:    Parsed JSON if the response was JSON-formatted
        raw_response:   The full Anthropic API response object
    """
    content: str
    input_tokens: int
    output_tokens: int
    model: str
    agent_type: AgentType
    duration_ms: float
    parsed_json: Optional[Dict[str, Any]] = None
    raw_response: Optional[Any] = None

    @property
    def total_tokens(self) -> int:
        """Total tokens used (input + output)."""
        return self.input_tokens + self.output_tokens

    @property
    def estimated_cost_usd(self) -> float:
        """
        Estimates cost in USD based on Claude Sonnet 4 pricing.
        Input:  $3.00 per million tokens
        Output: $15.00 per million tokens
        Update these rates if pricing changes.
        """
        input_cost  = (self.input_tokens  / 1_000_000) * 3.00
        output_cost = (self.output_tokens / 1_000_000) * 15.00
        return round(input_cost + output_cost, 6)

    def __repr__(self) -> str:
        return (
            f"<LLMResponse agent={self.agent_type.value} "
            f"tokens={self.total_tokens} "
            f"cost=${self.estimated_cost_usd} "
            f"duration={self.duration_ms}ms>"
        )


@dataclass
class TokenUsageTracker:
    """
    Tracks cumulative token usage across all Claude API calls
    in a pipeline run. Passed through the pipeline so we can
    report total cost and usage at the end.
    """
    total_input_tokens:  int   = 0
    total_output_tokens: int   = 0
    total_calls:         int   = 0
    total_duration_ms:   float = 0.0
    calls_by_agent: Dict[str, int] = field(default_factory=dict)

    def record(self, response: LLMResponse) -> None:
        """Records a completed API call into the tracker."""
        self.total_input_tokens  += response.input_tokens
        self.total_output_tokens += response.output_tokens
        self.total_calls         += 1
        self.total_duration_ms   += response.duration_ms

        agent = response.agent_type.value
        self.calls_by_agent[agent] = self.calls_by_agent.get(agent, 0) + 1

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def estimated_total_cost_usd(self) -> float:
        input_cost  = (self.total_input_tokens  / 1_000_000) * 3.00
        output_cost = (self.total_output_tokens / 1_000_000) * 15.00
        return round(input_cost + output_cost, 6)

    def summary(self) -> Dict[str, Any]:
        """Returns a summary dict for logging and API responses."""
        return {
            "total_calls":         self.total_calls,
            "total_tokens":        self.total_tokens,
            "total_input_tokens":  self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_duration_ms":   round(self.total_duration_ms, 2),
            "estimated_cost_usd":  self.estimated_total_cost_usd,
            "calls_by_agent":      self.calls_by_agent,
        }


# ============================================================
# Exception Classes
# ============================================================

class LLMError(Exception):
    """Base exception for all LLM service errors."""
    pass


class LLMRateLimitError(LLMError):
    """Raised when Claude API rate limit is hit."""
    pass


class LLMAuthError(LLMError):
    """Raised when the API key is invalid or expired."""
    pass


class LLMContextLengthError(LLMError):
    """Raised when the prompt exceeds Claude's context window."""
    pass


class LLMJSONParseError(LLMError):
    """Raised when Claude's response cannot be parsed as JSON."""
    pass


class LLMTimeoutError(LLMError):
    """Raised when the API call exceeds the configured timeout."""
    pass


# ============================================================
# System Prompts — One Per Agent Type
# ============================================================

SYSTEM_PROMPTS: Dict[AgentType, str] = {

    AgentType.RESUME_PARSER: """
You are an expert resume parser and information extraction specialist.
Your task is to analyze resumes and extract structured information with
high precision and completeness.

RULES:
- Extract ALL information present — never skip sections
- Infer seniority level from total experience and role titles
- Normalize dates to a consistent format (e.g. "Jan 2022" or "2022-01")
- Calculate total experience in years (decimal, e.g. 3.5)
- Group skills by category (Languages, Frameworks, Tools, etc.)
- Extract keywords that would matter for ATS systems
- Always respond with valid JSON only — no markdown, no explanation
- If a field is not present in the resume, use null
""".strip(),

    AgentType.JOB_MATCHING: """
You are an expert job matching analyst with deep knowledge of hiring
requirements across the technology industry.

Your task is to compare a candidate's resume against a job description
and produce an accurate, fair match analysis.

RULES:
- Score from 0.0 to 1.0 (0 = no match, 1 = perfect match)
- Be specific about which skills match and which are missing
- Consider transferable skills — not just exact keyword matches
- Weight required skills higher than preferred skills
- Account for experience level alignment
- Always respond with valid JSON only — no markdown, no explanation
""".strip(),

    AgentType.SKILL_GAP: """
You are a career development expert and technical skills advisor.

Your task is to analyze the gap between a candidate's current skills
and the skills required for their target role, then recommend a
practical learning roadmap.

RULES:
- Prioritize missing skills by importance to the role
- Suggest specific courses, platforms, and resources (Coursera, Udemy,
  official docs, YouTube channels, books)
- Estimate realistic time to learn each skill
- Distinguish between "must learn now" vs "learn eventually"
- Always respond with valid JSON only — no markdown, no explanation
""".strip(),

    AgentType.ATS_OPTIMIZER: """
You are an expert ATS (Applicant Tracking System) optimization specialist
and professional resume writer with 10+ years of experience.

Your task is to rewrite and optimize resumes to pass ATS screening
while remaining compelling to human readers.

RULES:
- Inject missing keywords naturally — never keyword stuffing
- Use strong action verbs to start each bullet point
- Quantify achievements wherever possible (%, $, time saved)
- Match the exact terminology used in the job description
- Maintain truthfulness — never fabricate experience or skills
- Use standard section headers ATS systems recognize
- Always respond with valid JSON only — no markdown, no explanation
""".strip(),

    AgentType.COVER_LETTER: """
You are an expert cover letter writer who crafts compelling,
personalized cover letters that get interviews.

Your task is to write a professional cover letter that connects the
candidate's specific experience to the company's needs.

RULES:
- Open with a strong hook — not "I am applying for..."
- Reference specific company details, products, or values
- Connect 2-3 specific resume achievements to the job requirements
- Show genuine enthusiasm for the specific role and company
- End with a clear, confident call to action
- Keep it to 3-4 paragraphs, under 400 words
- Professional but human tone — not robotic or generic
- Respond with the cover letter text directly — no JSON wrapper
""".strip(),

    AgentType.INTERVIEW_PREP: """
You are a senior technical interviewer and career coach with experience
at top technology companies (FAANG, startups, enterprise).

Your task is to generate a comprehensive interview preparation kit
tailored to the specific candidate and role.

RULES:
- Generate questions at the right difficulty for the experience level
- Mix technical, behavioral, and role-specific questions
- Provide model answers for behavioral questions using STAR format
- Include tips specific to the company's known interview style
- Flag topics where the candidate has skill gaps (from missing skills)
- Always respond with valid JSON only — no markdown, no explanation
""".strip(),

    AgentType.TRACKER: """
You are an intelligent job application tracking assistant.

Your task is to analyze application data and provide actionable
insights, follow-up recommendations, and status summaries.

RULES:
- Be concise and actionable in all recommendations
- Flag applications that need attention (no response, upcoming deadlines)
- Suggest optimal follow-up timing based on industry norms
- Always respond with valid JSON only — no markdown, no explanation
""".strip(),

    AgentType.JOB_SEARCH: """
You are a job search specialist who helps extract and structure
job listing information from raw scraped data.

RULES:
- Extract all available fields from the raw job data
- Normalize values to standard formats
- Infer missing fields when clearly implied by context
- Always respond with valid JSON only — no markdown, no explanation
""".strip(),

    AgentType.GENERAL: """
You are a helpful AI assistant for a job search platform.
Be concise, accurate, and professional in all responses.
""".strip(),
}


# ============================================================
# LLM Service Class
# ============================================================

class LLMService:
    """
    Centralized Claude AI service for the Job Search AI backend.

    Provides a clean async interface for all agents to interact
    with Claude. Handles client lifecycle, retries, error
    classification, JSON parsing, and token tracking.

    Usage:
        # Direct instantiation (for dependency injection):
        llm = LLMService()
        response = await llm.complete(
            agent_type=AgentType.RESUME_PARSER,
            user_message="Parse this resume: ...",
        )

        # Access parsed JSON:
        data = response.parsed_json

        # Check token usage:
        print(response.total_tokens)
        print(response.estimated_cost_usd)
    """

    def __init__(self) -> None:
        """
        Initializes the Anthropic async client.
        Client is created once and reused for all calls.
        """
        self._client = anthropic.AsyncAnthropic(
            api_key=settings.anthropic_api_key,
            timeout=settings.anthropic_timeout,
            max_retries=0,   # We handle retries ourselves via tenacity
        )
        self._model = settings.anthropic_model
        self._max_tokens = settings.anthropic_max_tokens
        self._max_retries = settings.anthropic_max_retries

        logger.info(
            f"LLMService initialized | "
            f"model={self._model} | "
            f"max_tokens={self._max_tokens} | "
            f"timeout={settings.anthropic_timeout}s"
        )

    # ----------------------------------------------------------
    # Primary Interface — complete()
    # ----------------------------------------------------------

    async def complete(
        self,
        agent_type: AgentType,
        user_message: str,
        conversation_history: Optional[List[LLMMessage]] = None,
        system_prompt_override: Optional[str] = None,
        expect_json: bool = False,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
        usage_tracker: Optional[TokenUsageTracker] = None,
    ) -> LLMResponse:
        """
        Sends a message to Claude and returns a structured response.

        This is the PRIMARY method all agents call. It handles:
        - System prompt selection by agent type
        - Conversation history management
        - Automatic retry with exponential backoff
        - JSON parsing when expect_json=True
        - Token usage tracking
        - Detailed error logging

        Args:
            agent_type:             Which agent is making this call.
            user_message:           The prompt/question to send Claude.
            conversation_history:   Previous messages for multi-turn
                                    conversations. Defaults to None.
            system_prompt_override: Custom system prompt — overrides
                                    the default for this agent type.
            expect_json:            If True, parse response as JSON
                                    and populate response.parsed_json.
            temperature:            Sampling temperature (0.0-1.0).
                                    Lower = more deterministic.
                                    Default 0.3 for structured tasks.
            max_tokens:             Override max tokens for this call.
            usage_tracker:          If provided, records token usage.

        Returns:
            LLMResponse: Structured response with content and metadata.

        Raises:
            LLMAuthError:          Invalid API key.
            LLMRateLimitError:     Rate limit exceeded after all retries.
            LLMContextLengthError: Prompt too long for context window.
            LLMTimeoutError:       Request timed out after all retries.
            LLMError:              Any other unrecoverable API error.
        """
        # Select system prompt
        system_prompt = (
            system_prompt_override
            or SYSTEM_PROMPTS.get(agent_type, SYSTEM_PROMPTS[AgentType.GENERAL])
        )

        # Build messages list
        messages = self._build_messages(
            user_message=user_message,
            history=conversation_history,
            expect_json=expect_json,
        )

        # Record start time for duration tracking
        start_time = time.perf_counter()

        # Execute with retry logic
        try:
            raw_response = await self._call_with_retry(
                system_prompt=system_prompt,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens or self._max_tokens,
            )
        except RetryError as e:
            # All retry attempts exhausted
            logger.error(
                f"All retry attempts exhausted | "
                f"agent={agent_type.value} | "
                f"error={str(e)}"
            )
            raise LLMError(
                f"Claude API failed after {self._max_retries} retries: {e}"
            )

        duration_ms = (time.perf_counter() - start_time) * 1000

        # Extract content text
        content = self._extract_content(raw_response)

        # Parse JSON if requested
        parsed_json: Optional[Dict[str, Any]] = None
        if expect_json:
            parsed_json = self._parse_json_response(
                content=content,
                agent_type=agent_type,
            )

        # Build structured response
        response = LLMResponse(
            content=content,
            input_tokens=raw_response.usage.input_tokens,
            output_tokens=raw_response.usage.output_tokens,
            model=raw_response.model,
            agent_type=agent_type,
            duration_ms=round(duration_ms, 2),
            parsed_json=parsed_json,
            raw_response=raw_response,
        )

        # Record to usage tracker if provided
        if usage_tracker:
            usage_tracker.record(response)

        logger.debug(
            f"LLM call completed | "
            f"agent={agent_type.value} | "
            f"tokens={response.total_tokens} | "
            f"cost=${response.estimated_cost_usd} | "
            f"duration={response.duration_ms}ms"
        )

        return response

    # ----------------------------------------------------------
    # Streaming Interface — stream()
    # ----------------------------------------------------------

    async def stream(
        self,
        agent_type: AgentType,
        user_message: str,
        system_prompt_override: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: Optional[int] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Streams Claude's response token by token.

        Used for long-running agent tasks (ATS optimization,
        cover letter generation) where we want to show the user
        live progress instead of waiting for the full response.

        Used by WebSocket endpoints to push tokens to the frontend
        as they arrive.

        Args:
            agent_type:             Agent making this call.
            user_message:           The prompt to send.
            system_prompt_override: Custom system prompt override.
            temperature:            Sampling temperature.
            max_tokens:             Max tokens override.

        Yields:
            str: Individual text chunks as they arrive from Claude.

        Usage:
            async for chunk in llm.stream(AgentType.COVER_LETTER, prompt):
                await websocket.send_text(chunk)
        """
        system_prompt = (
            system_prompt_override
            or SYSTEM_PROMPTS.get(agent_type, SYSTEM_PROMPTS[AgentType.GENERAL])
        )

        messages = self._build_messages(user_message=user_message)

        logger.debug(
            f"Starting stream | agent={agent_type.value}"
        )

        try:
            async with self._client.messages.stream(
                model=self._model,
                max_tokens=max_tokens or self._max_tokens,
                system=system_prompt,
                messages=messages,
                temperature=temperature,
            ) as stream:
                async for text_chunk in stream.text_stream:
                    yield text_chunk

        except anthropic.AuthenticationError as e:
            logger.error(f"Auth error during stream: {e}")
            raise LLMAuthError(f"Invalid Anthropic API key: {e}")

        except anthropic.RateLimitError as e:
            logger.warning(f"Rate limit during stream: {e}")
            raise LLMRateLimitError(f"Rate limit exceeded: {e}")

        except Exception as e:
            logger.error(f"Unexpected error during stream: {e}")
            raise LLMError(f"Stream failed: {e}")

    # ----------------------------------------------------------
    # Convenience Methods — Agents Call These
    # ----------------------------------------------------------

    async def complete_json(
        self,
        agent_type: AgentType,
        user_message: str,
        conversation_history: Optional[List[LLMMessage]] = None,
        system_prompt_override: Optional[str] = None,
        usage_tracker: Optional[TokenUsageTracker] = None,
        temperature: float = 0.1,    # Lower temp for JSON — more deterministic
    ) -> Dict[str, Any]:
        """
        Calls Claude and returns the parsed JSON response directly.

        Shortcut for agents that always expect JSON — avoids the
        boilerplate of setting expect_json=True and checking
        response.parsed_json every time.

        Args:
            agent_type:             Agent making this call.
            user_message:           The prompt to send.
            conversation_history:   Optional chat history.
            system_prompt_override: Custom system prompt.
            usage_tracker:          Token usage tracker.
            temperature:            Defaults to 0.1 for JSON tasks.

        Returns:
            Dict[str, Any]: The parsed JSON from Claude's response.

        Raises:
            LLMJSONParseError: If response cannot be parsed as JSON.
            LLMError:          Any API-level error.

        Usage in agents:
            data = await self.llm.complete_json(
                agent_type=AgentType.RESUME_PARSER,
                user_message=f"Parse this resume:\\n{resume_text}",
                usage_tracker=self.tracker,
            )
            skills = data.get("all_skills_flat", [])
        """
        response = await self.complete(
            agent_type=agent_type,
            user_message=user_message,
            conversation_history=conversation_history,
            system_prompt_override=system_prompt_override,
            expect_json=True,
            temperature=temperature,
            usage_tracker=usage_tracker,
        )

        if response.parsed_json is None:
            raise LLMJSONParseError(
                f"Agent '{agent_type.value}' received a response that "
                f"could not be parsed as JSON. "
                f"Raw response (first 500 chars): "
                f"{response.content[:500]}"
            )

        return response.parsed_json

    async def complete_text(
        self,
        agent_type: AgentType,
        user_message: str,
        system_prompt_override: Optional[str] = None,
        usage_tracker: Optional[TokenUsageTracker] = None,
        temperature: float = 0.7,    # Higher temp for creative text
    ) -> str:
        """
        Calls Claude and returns the plain text response.

        Used by Cover Letter Agent (free-form text output)
        and any agent that doesn't need structured JSON.

        Args:
            agent_type:             Agent making this call.
            user_message:           The prompt to send.
            system_prompt_override: Custom system prompt.
            usage_tracker:          Token usage tracker.
            temperature:            Defaults to 0.7 for creative tasks.

        Returns:
            str: Claude's text response, stripped of whitespace.

        Usage:
            letter = await self.llm.complete_text(
                agent_type=AgentType.COVER_LETTER,
                user_message=cover_letter_prompt,
                usage_tracker=self.tracker,
            )
        """
        response = await self.complete(
            agent_type=agent_type,
            user_message=user_message,
            system_prompt_override=system_prompt_override,
            expect_json=False,
            temperature=temperature,
            usage_tracker=usage_tracker,
        )

        return response.content.strip()

    # ----------------------------------------------------------
    # Internal Methods
    # ----------------------------------------------------------

    def _build_messages(
        self,
        user_message: str,
        history: Optional[List[LLMMessage]] = None,
        expect_json: bool = False,
    ) -> List[Dict[str, str]]:
        """
        Builds the messages list for the Anthropic API.

        Appends a JSON instruction suffix to the user message
        when expect_json=True, helping Claude stay on format.

        Args:
            user_message: The current user message.
            history:      Previous conversation messages.
            expect_json:  Whether to append JSON instruction.

        Returns:
            List[Dict]: Messages in Anthropic API format.
        """
        messages = []

        # Add conversation history if provided
        if history:
            for msg in history:
                messages.append(msg.to_dict())

        # Append JSON instruction to user message if needed
        final_message = user_message
        if expect_json:
            final_message = (
                f"{user_message}\n\n"
                f"IMPORTANT: Respond with ONLY valid JSON. "
                f"Do not include any text before or after the JSON. "
                f"Do not use markdown code blocks. "
                f"Start your response with {{ and end with }}"
            )

        messages.append({"role": "user", "content": final_message})

        return messages

    async def _call_with_retry(
        self,
        system_prompt: str,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> Any:
        """
        Calls the Anthropic API with automatic retry logic.

        Retry strategy:
        - Retries on: RateLimitError, APIConnectionError, InternalServerError
        - Does NOT retry on: AuthenticationError, PermissionError,
          InvalidRequestError (these won't succeed on retry)
        - Wait: exponential backoff starting at 2s, max 60s
        - Max attempts: from settings.anthropic_max_retries

        Args:
            system_prompt: The system prompt for this call.
            messages:      The formatted messages list.
            temperature:   Sampling temperature.
            max_tokens:    Maximum tokens to generate.

        Returns:
            anthropic.Message: The raw API response.
        """
        # Exceptions that are safe to retry
        retryable_errors = (
            anthropic.RateLimitError,
            anthropic.APIConnectionError,
            anthropic.InternalServerError,
        )

        async for attempt in AsyncRetrying(
            retry=retry_if_exception_type(retryable_errors),
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(
                multiplier=1,
                min=2,      # Start with 2 second wait
                max=60,     # Never wait more than 60 seconds
            ),
            reraise=True,
        ):
            with attempt:
                attempt_num = attempt.retry_state.attempt_number

                if attempt_num > 1:
                    logger.warning(
                        f"Retry attempt {attempt_num}/{self._max_retries} "
                        f"for Claude API call"
                    )

                try:
                    response = await self._client.messages.create(
                        model=self._model,
                        max_tokens=max_tokens,
                        system=system_prompt,
                        messages=messages,
                        temperature=temperature,
                    )
                    return response

                except anthropic.AuthenticationError as e:
                    # API key is wrong — no point retrying
                    logger.error(f"Authentication failed: {e}")
                    raise LLMAuthError(
                        f"Anthropic API key is invalid. "
                        f"Check ANTHROPIC_API_KEY in your .env file. "
                        f"Error: {e}"
                    )

                except anthropic.BadRequestError as e:
                    # Usually means the prompt is too long
                    error_str = str(e).lower()
                    if "context" in error_str or "too long" in error_str:
                        logger.error(f"Context length exceeded: {e}")
                        raise LLMContextLengthError(
                            f"Prompt exceeds Claude's context window. "
                            f"Reduce the input size. Error: {e}"
                        )
                    logger.error(f"Bad request to Claude API: {e}")
                    raise LLMError(f"Invalid request to Claude API: {e}")

                except anthropic.RateLimitError as e:
                    logger.warning(
                        f"Rate limit hit (attempt {attempt_num}): {e}"
                    )
                    raise   # Let tenacity handle the retry

                except anthropic.APIConnectionError as e:
                    logger.warning(
                        f"Connection error (attempt {attempt_num}): {e}"
                    )
                    raise   # Let tenacity handle the retry

                except anthropic.InternalServerError as e:
                    logger.warning(
                        f"Anthropic server error (attempt {attempt_num}): {e}"
                    )
                    raise   # Let tenacity handle the retry

                except TimeoutError as e:
                    logger.error(f"Request timed out: {e}")
                    raise LLMTimeoutError(
                        f"Claude API request timed out after "
                        f"{settings.anthropic_timeout}s. "
                        f"Try increasing ANTHROPIC_TIMEOUT in .env"
                    )

                except Exception as e:
                    logger.error(
                        f"Unexpected error calling Claude API: "
                        f"{type(e).__name__}: {e}"
                    )
                    raise LLMError(
                        f"Unexpected error: {type(e).__name__}: {e}"
                    )

    def _extract_content(self, response: Any) -> str:
        """
        Extracts the text content from an Anthropic API response.

        Args:
            response: Raw anthropic.Message response object.

        Returns:
            str: The text content from the first content block.

        Raises:
            LLMError: If no text content block is found.
        """
        if not response.content:
            raise LLMError("Claude returned an empty response with no content.")

        for block in response.content:
            if block.type == "text":
                return block.text

        raise LLMError(
            f"Claude response contained no text block. "
            f"Content types found: {[b.type for b in response.content]}"
        )

    def _parse_json_response(
        self,
        content: str,
        agent_type: AgentType,
    ) -> Optional[Dict[str, Any]]:
        """
        Parses Claude's text response as JSON with multiple fallbacks.

        Handles common cases where Claude wraps JSON in markdown
        code blocks or adds explanatory text before/after.

        Strategy:
            1. Try direct JSON parse (ideal case)
            2. Strip markdown code blocks and retry
            3. Extract first {...} or [...] block using regex
            4. Return None if all strategies fail

        Args:
            content:    Claude's text response.
            agent_type: Used for error logging context.

        Returns:
            Dict[str, Any]: Parsed JSON, or None if parsing failed.
        """
        stripped = content.strip()

        # Strategy 1: Direct parse
        try:
            result = json.loads(stripped)
            if isinstance(result, (dict, list)):
                return result if isinstance(result, dict) else {"items": result}
        except json.JSONDecodeError:
            pass

        # Strategy 2: Strip markdown code blocks
        # Handles ```json ... ``` and ``` ... ```
        clean = re.sub(r"```(?:json)?\s*", "", stripped)
        clean = re.sub(r"\s*```", "", clean).strip()
        try:
            result = json.loads(clean)
            if isinstance(result, (dict, list)):
                logger.debug(
                    f"JSON parsed after stripping markdown blocks | "
                    f"agent={agent_type.value}"
                )
                return result if isinstance(result, dict) else {"items": result}
        except json.JSONDecodeError:
            pass

        # Strategy 3: Extract first JSON object with regex
        json_pattern = re.search(r"\{[\s\S]*\}", stripped)
        if json_pattern:
            try:
                result = json.loads(json_pattern.group())
                logger.debug(
                    f"JSON extracted via regex fallback | "
                    f"agent={agent_type.value}"
                )
                return result
            except json.JSONDecodeError:
                pass

        # Strategy 4: Try to find a JSON array
        array_pattern = re.search(r"\[[\s\S]*\]", stripped)
        if array_pattern:
            try:
                result = json.loads(array_pattern.group())
                logger.debug(
                    f"JSON array extracted via regex fallback | "
                    f"agent={agent_type.value}"
                )
                return {"items": result}
            except json.JSONDecodeError:
                pass

        # All strategies failed
        logger.warning(
            f"Failed to parse JSON response | "
            f"agent={agent_type.value} | "
            f"content_preview={stripped[:200]!r}"
        )
        return None

    async def count_tokens(self, text: str) -> int:
        """
        Estimates token count for a given text string.

        Uses a simple approximation: ~4 characters per token.
        This avoids an extra API call for token counting.

        For exact counts, Claude's usage field in the response
        is always accurate — use that for billing.

        Args:
            text: The text to estimate tokens for.

        Returns:
            int: Estimated token count.
        """
        # ~4 characters per token is a reasonable approximation
        return len(text) // 4

    def truncate_to_token_limit(
        self,
        text: str,
        max_tokens: int = 50_000,
        chars_per_token: int = 4,
    ) -> str:
        """
        Truncates text to stay within a token budget.

        Used when resume text or job descriptions might exceed
        context window limits. Truncates from the end since
        the most important info is usually at the beginning.

        Args:
            text:             Input text to potentially truncate.
            max_tokens:       Maximum tokens to allow.
            chars_per_token:  Approximation factor.

        Returns:
            str: Truncated text with truncation notice if needed.
        """
        max_chars = max_tokens * chars_per_token

        if len(text) <= max_chars:
            return text

        truncated = text[:max_chars]
        logger.warning(
            f"Text truncated from {len(text)} to {max_chars} chars "
            f"(~{max_tokens} tokens) to fit context window"
        )

        return truncated + "\n\n[Content truncated to fit context window]"

    async def health_check(self) -> Dict[str, Any]:
        """
        Verifies the Claude API is reachable and responding.

        Called from main.py startup validation and GET /health.

        Returns:
            Dict with status, model, and latency information.

        Example:
            {
                "status": "healthy",
                "model": "claude-sonnet-4-5",
                "latency_ms": 842.3
            }
        """
        start = time.perf_counter()
        try:
            response = await self.complete(
                agent_type=AgentType.GENERAL,
                user_message="Reply with the single word: OK",
                max_tokens=10,
                temperature=0.0,
            )
            latency_ms = round((time.perf_counter() - start) * 1000, 2)

            return {
                "status": "healthy",
                "model": response.model,
                "latency_ms": latency_ms,
                "response": response.content.strip(),
            }

        except LLMAuthError as e:
            return {"status": "unhealthy", "error": "auth_failed", "detail": str(e)}
        except LLMError as e:
            return {"status": "unhealthy", "error": "api_error", "detail": str(e)}
        except Exception as e:
            return {"status": "unhealthy", "error": "unknown", "detail": str(e)}


# ============================================================
# Module-Level Singleton
# ============================================================
# Instantiated once at import time — reused across all agents.
# Avoids creating multiple Anthropic client instances.

llm_service: LLMService = LLMService()