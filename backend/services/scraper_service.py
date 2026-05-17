# ============================================================
# File: backend/services/scraper_service.py
# Purpose: Fetches live job listings from job board APIs.
#          Primary source: JSearch RapidAPI (covers LinkedIn,
#          Indeed, Glassdoor, ZipRecruiter, Naukri).
#
# Features:
#   - Async HTTP via httpx (non-blocking)
#   - Automatic rate limiting + delay between requests
#   - Pagination support (fetch multiple pages)
#   - Response normalization → JobCreate objects
#   - Deduplication by (source, external_job_id)
#   - Salary parsing (various formats → structured data)
#   - Skills extraction from job descriptions
#   - Graceful error handling (partial results on failure)
#
# Used by:
#   - backend/agents/job_search_agent.py → primary caller
#
# API Documentation:
#   https://rapidapi.com/letscrape-6bRBa3QguO5/api/jsearch
# ============================================================

import asyncio
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlencode

import httpx

from backend.config import settings
from backend.models.job import JobCreate
from backend.utils.logger import get_service_logger

logger = get_service_logger("ScraperService")


# ============================================================
# Data Classes
# ============================================================

@dataclass
class SearchParams:
    """
    Parameters for a job search request.

    Constructed by Agent 2 from the user's preferences
    and passed to ScraperService.search_jobs().

    Attributes:
        query:           Search query string e.g. "Python Backend Engineer"
        location:        Location string e.g. "Bangalore" or "Remote"
        experience_years:Candidate's years of experience (for filtering)
        work_mode:       "remote" | "onsite" | "hybrid" | None
        job_type:        "full_time" | "internship" | "contract" | None
        max_results:     Maximum jobs to return across all pages
        page:            Starting page number (1-based)
        date_posted:     Filter: "all" | "today" | "3days" | "week" | "month"
        country:         Country code for location filtering e.g. "in", "us"
    """
    query:            str
    location:         Optional[str]  = None
    experience_years: Optional[int]  = None
    work_mode:        Optional[str]  = None
    job_type:         Optional[str]  = None
    max_results:      int            = 20
    page:             int            = 1
    date_posted:      str            = "month"
    country:          str            = "in"   # India default

    def to_query_string(self) -> str:
        """
        Builds the optimized search query string for JSearch API.

        Combines the base query with location and work mode
        modifiers for better search relevance.

        Returns:
            str: Optimized query string e.g.
                 "Python Backend Engineer Bangalore remote"
        """
        parts = [self.query]

        if self.location and self.location.lower() != "remote":
            parts.append(self.location)

        if self.work_mode == "remote":
            parts.append("remote")

        return " ".join(parts)


@dataclass
class ScraperResult:
    """
    Result of a job search operation.

    Attributes:
        jobs:           List of normalized JobCreate objects.
        total_found:    Total jobs reported by the API (may exceed max_results).
        pages_fetched:  Number of API pages fetched.
        sources_used:   Which job board sources returned results.
        duration_ms:    Total time taken for all API calls.
        errors:         Non-fatal errors encountered during fetching.
        has_more:       True if more pages of results are available.
    """
    jobs:          List[JobCreate]
    total_found:   int
    pages_fetched: int
    sources_used:  List[str]
    duration_ms:   float
    errors:        List[str]        = field(default_factory=list)
    has_more:      bool             = False

    @property
    def jobs_count(self) -> int:
        """Number of jobs successfully fetched and normalized."""
        return len(self.jobs)

    @property
    def has_errors(self) -> bool:
        """True if any non-fatal errors occurred."""
        return len(self.errors) > 0

    def __repr__(self) -> str:
        return (
            f"<ScraperResult "
            f"jobs={self.jobs_count} "
            f"total_found={self.total_found} "
            f"pages={self.pages_fetched} "
            f"duration={round(self.duration_ms)}ms>"
        )


# ============================================================
# Scraper Service
# ============================================================

class ScraperService:
    """
    Async job listing fetcher using JSearch RapidAPI.

    JSearch aggregates jobs from:
        - LinkedIn
        - Indeed
        - Glassdoor
        - ZipRecruiter
        - Naukri (India)
        - Monster
        - CareerJet
        and many more sources.

    Usage:
        service = ScraperService()

        params = SearchParams(
            query="Python Backend Engineer",
            location="Bangalore",
            max_results=20,
        )

        result = await service.search_jobs(params)

        for job in result.jobs:
            print(job.title, job.company_name, job.location)
    """

    # JSearch API base URL
    _BASE_URL = "https://jsearch.p.rapidapi.com"

    # Page size per API call (JSearch max is 10 per page)
    _PAGE_SIZE = 10

    # Common tech skills to extract from job descriptions
    # Used when the API doesn't return structured skill data
    _TECH_SKILLS = {
        # Languages
        "python", "javascript", "typescript", "java", "golang", "go",
        "rust", "c++", "cpp", "c#", "ruby", "php", "swift", "kotlin",
        "scala", "r", "matlab", "bash", "shell",

        # Web Frameworks
        "fastapi", "django", "flask", "express", "nestjs", "spring",
        "rails", "laravel", "react", "angular", "vue", "nextjs",
        "nuxtjs", "svelte", "gatsby",

        # Databases
        "postgresql", "postgres", "mysql", "mongodb", "redis",
        "elasticsearch", "cassandra", "dynamodb", "sqlite",
        "oracle", "mssql", "neo4j", "influxdb",

        # Cloud & DevOps
        "aws", "gcp", "azure", "docker", "kubernetes", "k8s",
        "terraform", "ansible", "jenkins", "gitlab", "github actions",
        "ci/cd", "linux", "nginx", "apache",

        # Data & ML
        "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch",
        "keras", "spark", "hadoop", "airflow", "kafka", "tableau",
        "power bi", "looker", "dbt", "mlflow",

        # Tools & Practices
        "git", "jira", "agile", "scrum", "rest", "graphql", "grpc",
        "microservices", "serverless", "oauth", "jwt",

        # Mobile
        "react native", "flutter", "android", "ios", "xcode",
    }

    def __init__(self) -> None:
        """
        Initializes the scraper with HTTP client configuration.
        The httpx AsyncClient is created lazily on first use.
        """
        self._client: Optional[httpx.AsyncClient] = None
        self._api_key = settings.rapidapi_key
        self._api_host = settings.rapidapi_host
        self._timeout = settings.scraper_timeout
        self._delay = settings.scraper_delay

        # Track API call count for rate limiting
        self._call_count = 0
        self._last_call_time: float = 0.0

        if not self._api_key:
            logger.warning(
                "RAPIDAPI_KEY is not set. "
                "Job search will return mock/empty results. "
                "Set RAPIDAPI_KEY in .env to enable live job search."
            )

        logger.info(
            f"ScraperService initialized | "
            f"api_key={'set' if self._api_key else 'NOT SET'} | "
            f"timeout={self._timeout}s | "
            f"delay={self._delay}s"
        )

    # ----------------------------------------------------------
    # HTTP Client Management
    # ----------------------------------------------------------

    async def _get_client(self) -> httpx.AsyncClient:
        """
        Returns the shared async HTTP client.
        Creates it on first call (lazy initialization).

        Returns:
            httpx.AsyncClient: Configured HTTP client.
        """
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=10.0,
                    read=self._timeout,
                    write=10.0,
                    pool=5.0,
                ),
                headers={
                    "X-RapidAPI-Key":  self._api_key,
                    "X-RapidAPI-Host": self._api_host,
                    "Accept":          "application/json",
                    "Content-Type":    "application/json",
                },
                follow_redirects=True,
            )
        return self._client

    async def close(self) -> None:
        """
        Closes the HTTP client connection pool.
        Call this on application shutdown.
        """
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            logger.info("ScraperService HTTP client closed.")

    # ----------------------------------------------------------
    # Primary Interface
    # ----------------------------------------------------------

    async def search_jobs(
        self,
        params: SearchParams,
    ) -> ScraperResult:
        """
        Searches for job listings matching the given parameters.

        Fetches multiple pages if needed to reach max_results.
        Normalizes all raw API responses into JobCreate objects.
        Handles partial failures gracefully — returns whatever
        was successfully fetched even if some pages fail.

        Args:
            params: SearchParams object with search criteria.

        Returns:
            ScraperResult: Normalized jobs + search metadata.
        """
        if not self._api_key:
            logger.warning(
                "No RapidAPI key configured — returning empty results."
            )
            return ScraperResult(
                jobs=[],
                total_found=0,
                pages_fetched=0,
                sources_used=[],
                duration_ms=0.0,
                errors=["RAPIDAPI_KEY not configured."],
            )

        start_time = time.perf_counter()
        all_jobs: List[JobCreate] = []
        errors: List[str] = []
        sources_used: set = set()
        total_found = 0
        pages_fetched = 0

        # Calculate how many pages we need
        pages_needed = (
            (params.max_results + self._PAGE_SIZE - 1)
            // self._PAGE_SIZE
        )
        pages_needed = min(pages_needed, 5)  # Cap at 5 pages (50 jobs)

        logger.info(
            f"Job search started | "
            f"query='{params.to_query_string()}' | "
            f"max_results={params.max_results} | "
            f"pages_needed={pages_needed}"
        )

        # Fetch pages sequentially (respect rate limits)
        for page_num in range(1, pages_needed + 1):
            try:
                # Enforce delay between API calls
                await self._rate_limit_delay()

                raw_response = await self._fetch_page(
                    params=params,
                    page=page_num,
                )

                if raw_response is None:
                    errors.append(f"Page {page_num}: No response received.")
                    continue

                # Extract metadata from first page
                if page_num == 1:
                    total_found = raw_response.get(
                        "status_code_message", ""
                    ) or len(raw_response.get("data", []))
                    # JSearch doesn't reliably report total count
                    # so we use the data length as a proxy

                # Normalize raw jobs
                raw_jobs = raw_response.get("data", [])

                if not raw_jobs:
                    logger.debug(f"Page {page_num}: No jobs returned.")
                    break  # No more results

                normalized = self._normalize_jobs(raw_jobs)
                all_jobs.extend(normalized)
                pages_fetched += 1

                # Track sources
                for job in normalized:
                    if job.source:
                        sources_used.add(job.source)

                logger.debug(
                    f"Page {page_num}: fetched {len(normalized)} jobs | "
                    f"total_so_far={len(all_jobs)}"
                )

                # Stop if we have enough
                if len(all_jobs) >= params.max_results:
                    all_jobs = all_jobs[:params.max_results]
                    break

            except httpx.TimeoutException as e:
                error_msg = f"Page {page_num}: Request timed out: {e}"
                errors.append(error_msg)
                logger.warning(error_msg)
                break  # Timeout likely means API is slow — stop paging

            except httpx.HTTPStatusError as e:
                error_msg = (
                    f"Page {page_num}: HTTP {e.response.status_code} — "
                    f"{e.response.text[:200]}"
                )
                errors.append(error_msg)
                logger.error(error_msg)

                # 429 = rate limit — stop immediately
                if e.response.status_code == 429:
                    errors.append(
                        "Rate limit hit. "
                        "Increase SCRAPER_DELAY in .env or "
                        "upgrade your RapidAPI plan."
                    )
                    break

            except Exception as e:
                error_msg = (
                    f"Page {page_num}: Unexpected error: "
                    f"{type(e).__name__}: {e}"
                )
                errors.append(error_msg)
                logger.error(error_msg)

        # Deduplicate by external_job_id
        all_jobs = self._deduplicate(all_jobs)
        total_found = total_found or len(all_jobs)

        duration_ms = round(
            (time.perf_counter() - start_time) * 1000, 2
        )

        logger.info(
            f"Job search complete | "
            f"found={len(all_jobs)} | "
            f"pages={pages_fetched} | "
            f"sources={list(sources_used)} | "
            f"errors={len(errors)} | "
            f"duration={duration_ms}ms"
        )

        return ScraperResult(
            jobs=all_jobs,
            total_found=total_found,
            pages_fetched=pages_fetched,
            sources_used=list(sources_used),
            duration_ms=duration_ms,
            errors=errors,
            has_more=len(all_jobs) >= params.max_results,
        )

    # ----------------------------------------------------------
    # API Request
    # ----------------------------------------------------------

    async def _fetch_page(
        self,
        params: SearchParams,
        page: int,
    ) -> Optional[Dict[str, Any]]:
        """
        Fetches a single page of job results from JSearch API.

        Args:
            params: Search parameters.
            page:   Page number (1-based).

        Returns:
            Dict: Raw API response JSON, or None on failure.

        Raises:
            httpx.TimeoutException:  Request timed out.
            httpx.HTTPStatusError:   Non-2xx HTTP response.
        """
        client = await self._get_client()

        # Build query parameters
        query_params = {
            "query":        params.to_query_string(),
            "page":         str(page),
            "num_pages":    "1",
            "date_posted":  params.date_posted,
            "country":      params.country,
        }

        # Add work mode filter
        if params.work_mode:
            work_mode_map = {
                "remote":  "true",
                "onsite":  "false",
                "hybrid":  "false",
            }
            if params.work_mode == "remote":
                query_params["remote_jobs_only"] = "true"

        # Add job type filter
        if params.job_type:
            job_type_map = {
                "full_time":   "FULLTIME",
                "part_time":   "PARTTIME",
                "contract":    "CONTRACTOR",
                "internship":  "INTERN",
            }
            api_job_type = job_type_map.get(params.job_type)
            if api_job_type:
                query_params["employment_types"] = api_job_type

        self._call_count += 1
        self._last_call_time = time.perf_counter()

        logger.debug(
            f"API request #{self._call_count} | "
            f"page={page} | "
            f"query='{query_params['query']}'"
        )

        response = await client.get(
            url=f"{self._BASE_URL}/search",
            params=query_params,
        )
        response.raise_for_status()

        data = response.json()

        logger.debug(
            f"API response | "
            f"status={response.status_code} | "
            f"jobs_in_page={len(data.get('data', []))}"
        )

        return data

    async def fetch_job_details(
        self,
        job_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Fetches detailed information for a specific job listing.

        Used when the search results don't include the full
        job description — fetches complete details on demand.

        Args:
            job_id: JSearch job_id from search results.

        Returns:
            Dict: Detailed job data, or None if not found.
        """
        if not self._api_key:
            return None

        try:
            client = await self._get_client()
            await self._rate_limit_delay()

            response = await client.get(
                url=f"{self._BASE_URL}/job-details",
                params={"job_id": job_id, "extended_publisher_details": "false"},
            )
            response.raise_for_status()

            data = response.json()
            jobs = data.get("data", [])
            return jobs[0] if jobs else None

        except Exception as e:
            logger.warning(f"Failed to fetch job details for {job_id}: {e}")
            return None

    # ----------------------------------------------------------
    # Rate Limiting
    # ----------------------------------------------------------

    async def _rate_limit_delay(self) -> None:
        """
        Enforces minimum delay between API calls.

        Reads SCRAPER_DELAY from settings (default 2.0 seconds).
        Calculates remaining wait time based on when the last
        call was made — avoids unnecessary waiting if the previous
        call took longer than the delay.
        """
        if self._last_call_time > 0:
            elapsed = time.perf_counter() - self._last_call_time
            wait_time = self._delay - elapsed

            if wait_time > 0:
                logger.debug(
                    f"Rate limiting: waiting {round(wait_time, 2)}s "
                    f"before next API call"
                )
                await asyncio.sleep(wait_time)

    # ----------------------------------------------------------
    # Response Normalization
    # ----------------------------------------------------------

    def _normalize_jobs(
        self,
        raw_jobs: List[Dict[str, Any]],
    ) -> List[JobCreate]:
        """
        Converts raw JSearch API job objects into JobCreate objects.

        JSearch returns a specific JSON structure. This method
        maps every relevant field to our JobCreate schema,
        handling missing fields, type conversion, and validation.

        Args:
            raw_jobs: List of raw job dicts from JSearch API.

        Returns:
            List[JobCreate]: Normalized job objects ready for DB.
        """
        normalized: List[JobCreate] = []

        for raw in raw_jobs:
            try:
                job = self._normalize_single_job(raw)
                if job:
                    normalized.append(job)
            except Exception as e:
                logger.warning(
                    f"Failed to normalize job "
                    f"'{raw.get('job_title', 'unknown')}': {e}"
                )
                continue

        return normalized

    def _normalize_single_job(
        self,
        raw: Dict[str, Any],
    ) -> Optional[JobCreate]:
        """
        Normalizes a single raw JSearch job into JobCreate.

        JSearch field mapping:
            job_id              → external_job_id
            job_title           → title
            employer_name       → company_name
            employer_logo       → company_logo_url
            employer_website    → company_website
            job_city + job_state + job_country → location
            job_description     → job_description
            job_employment_type → job_type
            job_is_remote       → work_mode
            job_apply_link      → job_url
            job_posted_at_datetime_utc → posted_at
            job_required_skills → required_skills
            job_required_experience → experience_min_years
            job_salary_* fields → salary_*
            job_highlights      → responsibilities + requirements

        Args:
            raw: Single raw job dict from JSearch API.

        Returns:
            JobCreate: Normalized object, or None if title missing.
        """
        # Title and company are required — skip if missing
        title = (raw.get("job_title") or "").strip()
        company = (raw.get("employer_name") or "").strip()

        if not title or not company:
            return None

        # ── Location ───────────────────────────────────────────
        location = self._build_location(raw)

        # ── Work Mode ──────────────────────────────────────────
        work_mode = None
        if raw.get("job_is_remote"):
            work_mode = "remote"
        elif raw.get("job_work_from_home"):
            work_mode = "remote"

        # ── Job Type ───────────────────────────────────────────
        job_type = self._normalize_job_type(
            raw.get("job_employment_type")
        )

        # ── Source Platform ────────────────────────────────────
        source = self._extract_source(raw)

        # ── Job Description ────────────────────────────────────
        job_description = (raw.get("job_description") or "").strip()

        # ── Skills Extraction ──────────────────────────────────
        required_skills = self._extract_required_skills(raw)
        preferred_skills = self._extract_preferred_skills(raw)

        # If API didn't return structured skills, extract from JD
        if not required_skills and job_description:
            required_skills = self._extract_skills_from_text(
                job_description
            )

        # ── Keywords ───────────────────────────────────────────
        keywords = self._extract_keywords_from_highlights(raw)

        # ── Responsibilities & Requirements ────────────────────
        responsibilities, requirements = self._extract_highlights(raw)

        # ── Salary ─────────────────────────────────────────────
        (salary_min, salary_max,
         salary_currency, salary_period,
         salary_display) = self._parse_salary(raw)

        # ── Experience ─────────────────────────────────────────
        exp_min, exp_max, exp_level = self._parse_experience(raw)

        # ── Posted Date ────────────────────────────────────────
        posted_at = self._parse_posted_date(raw)

        return JobCreate(
            # Source
            source=source,
            external_job_id=raw.get("job_id"),
            job_url=raw.get("job_apply_link"),

            # Core
            title=title,
            company_name=company,
            company_logo_url=raw.get("employer_logo"),
            company_website=raw.get("employer_website"),
            industry=raw.get("job_category"),

            # Location
            location=location,
            country=raw.get("job_country"),
            work_mode=work_mode,

            # Description
            job_description=job_description or None,
            responsibilities=responsibilities or None,
            requirements=requirements or None,

            # Skills
            required_skills=required_skills or None,
            preferred_skills=preferred_skills or None,
            keywords=keywords or None,

            # Experience
            experience_min_years=exp_min,
            experience_max_years=exp_max,
            experience_level=exp_level,
            education_required=raw.get("job_required_education", {})
                               .get("required_credential")
                               if raw.get("job_required_education")
                               else None,

            # Salary
            salary_min=salary_min,
            salary_max=salary_max,
            salary_currency=salary_currency,
            salary_period=salary_period,
            salary_display=salary_display,

            # Metadata
            job_type=job_type,
            posted_at=posted_at,
        )

    # ----------------------------------------------------------
    # Field Parsers
    # ----------------------------------------------------------

    def _build_location(self, raw: Dict[str, Any]) -> Optional[str]:
        """
        Builds a human-readable location string from raw fields.

        JSearch provides city, state, and country separately.
        Combines them intelligently:
            "Bangalore, Karnataka, India" → "Bangalore, India"
            "New York, NY, US"           → "New York, US"
            ",,"                         → None (if all empty)

        Args:
            raw: Raw job dict.

        Returns:
            str: Location string or None.
        """
        parts = []

        city    = (raw.get("job_city")    or "").strip()
        state   = (raw.get("job_state")   or "").strip()
        country = (raw.get("job_country") or "").strip()

        if city:
            parts.append(city)

        # Include state only if it adds info (not same as city)
        if state and state.lower() != city.lower():
            # Skip verbose US state names — use city + country
            if len(state) <= 3 or country.upper() not in ("US", "USA"):
                parts.append(state)

        if country:
            parts.append(country)

        return ", ".join(parts) if parts else None

    def _normalize_job_type(
        self,
        raw_type: Optional[str],
    ) -> Optional[str]:
        """
        Maps JSearch employment_type strings to our standard values.

        Args:
            raw_type: Raw employment type string from API.

        Returns:
            str: Normalized job type or None.
        """
        if not raw_type:
            return None

        mapping = {
            "FULLTIME":   "full_time",
            "PARTTIME":   "part_time",
            "CONTRACTOR": "contract",
            "INTERN":     "internship",
            "TEMPORARY":  "contract",
            "PER_DIEM":   "contract",
            "VOLUNTEER":  "contract",
        }
        return mapping.get(raw_type.upper(), raw_type.lower())

    def _extract_source(self, raw: Dict[str, Any]) -> str:
        """
        Determines the original job board source.

        JSearch includes publisher info. We map known publishers
        to our standard source identifiers.

        Args:
            raw: Raw job dict.

        Returns:
            str: Source identifier e.g. "linkedin", "indeed".
        """
        # Try publisher name first
        publisher = (
            raw.get("job_publisher")
            or raw.get("source")
            or ""
        ).lower()

        source_map = {
            "linkedin":    "linkedin",
            "indeed":      "indeed",
            "glassdoor":   "glassdoor",
            "ziprecruiter":"ziprecruiter",
            "naukri":      "naukri",
            "monster":     "monster",
            "careerjet":   "careerjet",
            "dice":        "dice",
            "lever":       "lever",
            "greenhouse":  "greenhouse",
            "workday":     "workday",
        }

        for key, value in source_map.items():
            if key in publisher:
                return value

        # Fallback: use job apply link domain
        apply_link = raw.get("job_apply_link") or ""
        for key, value in source_map.items():
            if key in apply_link.lower():
                return value

        return "jsearch_api"

    def _extract_required_skills(
        self,
        raw: Dict[str, Any],
    ) -> List[str]:
        """
        Extracts required skills from structured API fields.

        JSearch sometimes provides job_required_skills as a list.
        This is more reliable than text extraction when available.

        Args:
            raw: Raw job dict.

        Returns:
            List[str]: Required skills or empty list.
        """
        skills = raw.get("job_required_skills")
        if not skills:
            return []

        if isinstance(skills, list):
            return [str(s).strip() for s in skills if s]

        if isinstance(skills, str):
            # Sometimes returned as comma-separated string
            return [s.strip() for s in skills.split(",") if s.strip()]

        return []

    def _extract_preferred_skills(
        self,
        raw: Dict[str, Any],
    ) -> List[str]:
        """
        Extracts preferred/nice-to-have skills.

        Looks in the "Nice to Have" section of job highlights.

        Args:
            raw: Raw job dict.

        Returns:
            List[str]: Preferred skills or empty list.
        """
        highlights = raw.get("job_highlights") or {}
        nice_to_have = highlights.get("Nice to have", [])

        if isinstance(nice_to_have, list):
            skills = []
            for item in nice_to_have:
                extracted = self._extract_skills_from_text(str(item))
                skills.extend(extracted)
            return list(set(skills))

        return []

    def _extract_skills_from_text(
        self,
        text: str,
    ) -> List[str]:
        """
        Extracts technology skills from free-form text.

        Scans the text for known technology keywords from
        the _TECH_SKILLS set. Case-insensitive matching.

        Args:
            text: Job description or requirement text.

        Returns:
            List[str]: Skills found in the text (title-cased).
        """
        if not text:
            return []

        text_lower = text.lower()
        found = []

        for skill in self._TECH_SKILLS:
            # Use word boundary matching to avoid false positives
            # e.g. "java" should not match "javascript"
            pattern = r"\b" + re.escape(skill) + r"\b"
            if re.search(pattern, text_lower):
                # Return with proper casing
                found.append(self._format_skill(skill))

        return found

    def _format_skill(self, skill: str) -> str:
        """
        Returns the properly cased version of a skill name.

        Args:
            skill: Lowercase skill string.

        Returns:
            str: Properly cased skill name.
        """
        proper_case = {
            "python": "Python", "javascript": "JavaScript",
            "typescript": "TypeScript", "java": "Java",
            "golang": "Go", "go": "Go", "rust": "Rust",
            "c++": "C++", "cpp": "C++", "c#": "C#",
            "ruby": "Ruby", "php": "PHP", "swift": "Swift",
            "kotlin": "Kotlin", "scala": "Scala",
            "fastapi": "FastAPI", "django": "Django",
            "flask": "Flask", "express": "Express",
            "nestjs": "NestJS", "spring": "Spring Boot",
            "react": "React", "angular": "Angular",
            "vue": "Vue.js", "nextjs": "Next.js",
            "postgresql": "PostgreSQL", "postgres": "PostgreSQL",
            "mysql": "MySQL", "mongodb": "MongoDB",
            "redis": "Redis", "elasticsearch": "Elasticsearch",
            "aws": "AWS", "gcp": "GCP", "azure": "Azure",
            "docker": "Docker", "kubernetes": "Kubernetes",
            "k8s": "Kubernetes", "terraform": "Terraform",
            "pandas": "Pandas", "numpy": "NumPy",
            "tensorflow": "TensorFlow", "pytorch": "PyTorch",
            "kafka": "Kafka", "airflow": "Apache Airflow",
            "git": "Git", "linux": "Linux",
            "graphql": "GraphQL", "grpc": "gRPC",
            "rest": "REST API", "ci/cd": "CI/CD",
            "microservices": "Microservices",
        }
        return proper_case.get(skill, skill.title())

    def _extract_highlights(
        self,
        raw: Dict[str, Any],
    ) -> Tuple[List[str], List[str]]:
        """
        Extracts responsibilities and requirements from
        job_highlights field.

        JSearch structures highlights as:
            {
                "Responsibilities": ["bullet 1", "bullet 2"],
                "Qualifications": ["bullet 1", "bullet 2"],
                "Benefits": ["bullet 1"]
            }

        Args:
            raw: Raw job dict.

        Returns:
            Tuple[List[str], List[str]]:
                (responsibilities, requirements)
        """
        highlights = raw.get("job_highlights") or {}

        responsibilities = []
        requirements = []

        # Responsibilities
        resp_keys = ["Responsibilities", "Job Description", "What You'll Do"]
        for key in resp_keys:
            items = highlights.get(key, [])
            if isinstance(items, list):
                responsibilities.extend(items)

        # Requirements / Qualifications
        req_keys = [
            "Qualifications", "Requirements",
            "What We're Looking For", "Basic Qualifications",
        ]
        for key in req_keys:
            items = highlights.get(key, [])
            if isinstance(items, list):
                requirements.extend(items)

        return (
            [str(r).strip() for r in responsibilities if r][:15],
            [str(r).strip() for r in requirements if r][:15],
        )

    def _extract_keywords_from_highlights(
        self,
        raw: Dict[str, Any],
    ) -> List[str]:
        """
        Extracts ATS keywords by combining skills from all text fields.

        Args:
            raw: Raw job dict.

        Returns:
            List[str]: Deduplicated keyword list.
        """
        all_text_parts = []

        # Add job description
        if raw.get("job_description"):
            all_text_parts.append(raw["job_description"])

        # Add highlight sections
        highlights = raw.get("job_highlights") or {}
        for section_items in highlights.values():
            if isinstance(section_items, list):
                all_text_parts.extend([str(i) for i in section_items])

        combined_text = " ".join(all_text_parts)
        return self._extract_skills_from_text(combined_text)

    def _parse_salary(
        self,
        raw: Dict[str, Any],
    ) -> Tuple[
        Optional[float], Optional[float],
        Optional[str], Optional[str], Optional[str]
    ]:
        """
        Parses salary information from JSearch salary fields.

        JSearch provides:
            job_min_salary, job_max_salary,
            job_salary_currency, job_salary_period

        Also tries to parse salary from job title or description
        as a fallback.

        Args:
            raw: Raw job dict.

        Returns:
            Tuple of:
                salary_min, salary_max, currency, period, display_string
        """
        salary_min = raw.get("job_min_salary")
        salary_max = raw.get("job_max_salary")
        currency   = raw.get("job_salary_currency") or "INR"
        period     = raw.get("job_salary_period")

        # Normalize period
        period_map = {
            "YEAR":  "annual",
            "MONTH": "monthly",
            "HOUR":  "hourly",
            "WEEK":  "weekly",
        }
        if period:
            period = period_map.get(period.upper(), period.lower())

        # Try to parse from salary string in description
        if salary_min is None and salary_max is None:
            salary_str = raw.get("job_salary") or ""
            salary_min, salary_max, currency, period = (
                self._parse_salary_string(salary_str)
            )

        # Build display string
        display = None
        if salary_min or salary_max:
            currency_symbols = {
                "INR": "₹", "USD": "$", "EUR": "€", "GBP": "£"
            }
            sym = currency_symbols.get(currency or "", "")

            def fmt(amount: Optional[float]) -> str:
                if not amount:
                    return ""
                if currency == "INR" and amount >= 100_000:
                    return f"{amount/100_000:.0f}L"
                if amount >= 1_000:
                    return f"{amount/1_000:.0f}k"
                return str(int(amount))

            per = f"/{period}" if period else ""
            if salary_min and salary_max:
                display = f"{sym}{fmt(salary_min)}-{sym}{fmt(salary_max)}{per}"
            elif salary_min:
                display = f"{sym}{fmt(salary_min)}+{per}"
            elif salary_max:
                display = f"Up to {sym}{fmt(salary_max)}{per}"

        return (
            float(salary_min) if salary_min else None,
            float(salary_max) if salary_max else None,
            currency,
            period,
            display,
        )

    def _parse_salary_string(
        self,
        salary_str: str,
    ) -> Tuple[
        Optional[float], Optional[float],
        Optional[str], Optional[str]
    ]:
        """
        Parses salary from a free-form string like "15-25 LPA" or "$80k/yr".

        Handles common Indian and US salary formats:
            "15-25 LPA"      → min=1500000, max=2500000, INR, annual
            "₹20 LPA"        → min=2000000, INR, annual
            "$80k-$120k/yr"  → min=80000, max=120000, USD, annual
            "5000-8000/month"→ min=5000, max=8000, None, monthly

        Args:
            salary_str: Free-form salary string.

        Returns:
            Tuple: (min, max, currency, period) — all may be None.
        """
        if not salary_str:
            return None, None, None, None

        s = salary_str.lower().strip()
        currency = None
        period = None
        salary_min = None
        salary_max = None

        # Detect currency
        if "₹" in s or "inr" in s or "lpa" in s or "lakh" in s:
            currency = "INR"
        elif "$" in s or "usd" in s:
            currency = "USD"
        elif "€" in s or "eur" in s:
            currency = "EUR"
        elif "£" in s or "gbp" in s:
            currency = "GBP"

        # Detect period
        if "per year" in s or "/year" in s or "/yr" in s or "lpa" in s or "annual" in s:
            period = "annual"
        elif "per month" in s or "/month" in s or "/mo" in s or "monthly" in s:
            period = "monthly"
        elif "per hour" in s or "/hour" in s or "/hr" in s or "hourly" in s:
            period = "hourly"

        # Extract numbers
        numbers = re.findall(r"(\d+(?:\.\d+)?)", s)

        if not numbers:
            return None, None, currency, period

        multiplier = 1.0

        # LPA = Lakh Per Annum = 100,000
        if "lpa" in s or "lakh" in s or "lac" in s:
            multiplier = 100_000.0
            currency = currency or "INR"
            period = period or "annual"
        # "k" suffix = thousands
        elif "k" in s:
            multiplier = 1_000.0

        values = [float(n) * multiplier for n in numbers[:2]]

        if len(values) >= 2:
            salary_min = min(values[0], values[1])
            salary_max = max(values[0], values[1])
        elif len(values) == 1:
            salary_min = values[0]

        return salary_min, salary_max, currency, period

    def _parse_experience(
        self,
        raw: Dict[str, Any],
    ) -> Tuple[Optional[int], Optional[int], Optional[str]]:
        """
        Parses experience requirements from API fields.

        Args:
            raw: Raw job dict.

        Returns:
            Tuple: (min_years, max_years, experience_level)
        """
        exp_data = raw.get("job_required_experience") or {}

        min_years = None
        max_years = None
        level = None

        if isinstance(exp_data, dict):
            min_val = exp_data.get("minimum_experience_in_months")
            if min_val:
                try:
                    min_years = round(int(min_val) / 12)
                except (ValueError, TypeError):
                    pass

            no_exp = exp_data.get("no_experience_required", False)
            if no_exp:
                min_years = 0
                level = "entry"

        # Infer level from title
        title = (raw.get("job_title") or "").lower()
        if not level:
            if any(w in title for w in ["senior", "sr.", "lead", "principal"]):
                level = "senior"
            elif any(w in title for w in ["junior", "jr.", "associate"]):
                level = "junior"
            elif "intern" in title:
                level = "internship"
            elif any(w in title for w in ["manager", "director", "head", "vp"]):
                level = "lead"

        return min_years, max_years, level

    def _parse_posted_date(
        self,
        raw: Dict[str, Any],
    ) -> Optional[datetime]:
        """
        Parses the job posting date from API response.

        JSearch provides UTC datetime string in ISO format.

        Args:
            raw: Raw job dict.

        Returns:
            datetime (UTC) or None.
        """
        date_str = raw.get("job_posted_at_datetime_utc")
        if not date_str:
            # Try Unix timestamp
            timestamp = raw.get("job_posted_at_timestamp")
            if timestamp:
                try:
                    return datetime.fromtimestamp(
                        int(timestamp), tz=timezone.utc
                    )
                except (ValueError, TypeError):
                    pass
            return None

        try:
            # Handle both "Z" suffix and "+00:00" suffix
            date_str = date_str.replace("Z", "+00:00")
            return datetime.fromisoformat(date_str)
        except (ValueError, AttributeError):
            return None

    # ----------------------------------------------------------
    # Deduplication
    # ----------------------------------------------------------

    def _deduplicate(
        self,
        jobs: List[JobCreate],
    ) -> List[JobCreate]:
        """
        Removes duplicate job listings from the results.

        Deduplication strategy:
            1. Primary: (source, external_job_id) — exact match
            2. Fallback: (company_name, title) fuzzy match
               for jobs without external IDs

        Args:
            jobs: List of potentially duplicate JobCreate objects.

        Returns:
            List[JobCreate]: Deduplicated list preserving order.
        """
        seen_ids: set = set()
        seen_titles: set = set()
        unique_jobs: List[JobCreate] = []

        for job in jobs:
            # Primary dedup key
            if job.external_job_id and job.source:
                dedup_key = f"{job.source}::{job.external_job_id}"
                if dedup_key in seen_ids:
                    continue
                seen_ids.add(dedup_key)

            # Fallback dedup key (title + company)
            title_key = (
                f"{job.company_name.lower().strip()}::"
                f"{job.title.lower().strip()}"
            )
            if title_key in seen_titles:
                continue
            seen_titles.add(title_key)

            unique_jobs.append(job)

        duplicates_removed = len(jobs) - len(unique_jobs)
        if duplicates_removed > 0:
            logger.debug(
                f"Deduplication removed {duplicates_removed} duplicates | "
                f"unique_jobs={len(unique_jobs)}"
            )

        return unique_jobs

    # ----------------------------------------------------------
    # Health Check
    # ----------------------------------------------------------

    async def health_check(self) -> Dict[str, Any]:
        """
        Verifies the scraper service is configured and reachable.

        Returns:
            Dict: Status info including API key presence and connectivity.
        """
        if not self._api_key:
            return {
                "status":  "degraded",
                "message": "RAPIDAPI_KEY not set — live job search disabled.",
            }

        try:
            client = await self._get_client()
            await self._rate_limit_delay()

            # Minimal test request
            response = await client.get(
                url=f"{self._BASE_URL}/search",
                params={
                    "query": "software engineer",
                    "page":  "1",
                    "num_pages": "1",
                },
            )

            if response.status_code == 200:
                return {
                    "status":  "healthy",
                    "message": "JSearch API is reachable.",
                    "api_host": self._api_host,
                }
            else:
                return {
                    "status":  "unhealthy",
                    "message": f"API returned HTTP {response.status_code}",
                }

        except Exception as e:
            return {
                "status":  "unhealthy",
                "error":   str(e),
            }


# ============================================================
# Module-Level Singleton
# ============================================================

scraper_service: ScraperService = ScraperService()