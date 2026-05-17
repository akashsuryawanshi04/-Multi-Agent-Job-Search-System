# ============================================================
# File: backend/services/embedding_service.py
# Purpose: Semantic similarity scoring using sentence-transformers.
#          Converts text to vector embeddings and computes
#          cosine similarity scores for job matching.
#
# Features:
#   - Lazy model loading (loads on first use, not import)
#   - In-memory embedding cache (avoids recomputing same text)
#   - Batch processing (score multiple jobs at once efficiently)
#   - Skill-level matching (individual skill comparison)
#   - TF-IDF keyword extraction for ATS scoring
#   - Fully synchronous internally, async wrapper for FastAPI
#
# Used by:
#   - backend/agents/job_matching_agent.py   → primary consumer
#   - backend/agents/skill_gap_agent.py      → skill comparison
#   - backend/agents/ats_optimizer_agent.py  → keyword scoring
# ============================================================

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from backend.config import settings
from backend.utils.logger import get_service_logger

logger = get_service_logger("EmbeddingService")


# ============================================================
# Data Classes
# ============================================================

@dataclass
class SimilarityResult:
    """
    Result of a semantic similarity comparison.

    Attributes:
        semantic_score:   Cosine similarity from sentence-transformers
                          embeddings. Range: 0.0 - 1.0.
                          Captures meaning, not just keywords.

        keyword_score:    TF-IDF based keyword overlap score.
                          Range: 0.0 - 1.0.
                          Captures exact term matches (ATS-style).

        skill_score:      Direct skill list overlap score.
                          Range: 0.0 - 1.0.
                          Captures skill-by-skill matching.

        overall_score:    Weighted combination of all three scores.
                          Range: 0.0 - 1.0.
                          Primary match score shown to the user.

        matched_skills:   Skills found in both resume and job.
        missing_skills:   Required skills absent from resume.
        bonus_skills:     Resume skills not required but relevant.
        duration_ms:      Time taken to compute this result.
    """
    semantic_score:  float
    keyword_score:   float
    skill_score:     float
    overall_score:   float
    matched_skills:  List[str]
    missing_skills:  List[str]
    bonus_skills:    List[str]
    duration_ms:     float

    # Score weights used to compute overall_score
    # Tunable without changing the interface
    SEMANTIC_WEIGHT: float  = 0.50
    KEYWORD_WEIGHT:  float  = 0.25
    SKILL_WEIGHT:    float  = 0.25

    @property
    def overall_score_percent(self) -> float:
        """Returns overall score as 0–100 percentage."""
        return round(self.overall_score * 100, 1)

    @property
    def match_label(self) -> str:
        """
        Human-readable match quality label.
            ≥ 80% → Excellent Match
            ≥ 60% → Good Match
            ≥ 40% → Fair Match
            <  40% → Low Match
        """
        pct = self.overall_score_percent
        if pct >= 80:
            return "Excellent Match"
        elif pct >= 60:
            return "Good Match"
        elif pct >= 40:
            return "Fair Match"
        return "Low Match"

    @property
    def skill_coverage_percent(self) -> float:
        """
        Percentage of required skills the candidate already has.
        Used in skill gap UI to show progress.
        """
        total = len(self.matched_skills) + len(self.missing_skills)
        if total == 0:
            return 100.0
        return round(len(self.matched_skills) / total * 100, 1)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes result for storing in JobMatch.skill_gap_analysis."""
        return {
            "semantic_score":        round(self.semantic_score, 4),
            "keyword_score":         round(self.keyword_score, 4),
            "skill_score":           round(self.skill_score, 4),
            "overall_score":         round(self.overall_score, 4),
            "overall_score_percent": self.overall_score_percent,
            "match_label":           self.match_label,
            "skill_coverage_percent":self.skill_coverage_percent,
            "matched_skills":        self.matched_skills,
            "missing_skills":        self.missing_skills,
            "bonus_skills":          self.bonus_skills,
            "duration_ms":           round(self.duration_ms, 2),
        }

    def __repr__(self) -> str:
        return (
            f"<SimilarityResult "
            f"overall={self.overall_score_percent}% "
            f"label='{self.match_label}' "
            f"matched={len(self.matched_skills)} "
            f"missing={len(self.missing_skills)}>"
        )


@dataclass
class BatchMatchResult:
    """
    Result of batch-matching one resume against multiple jobs.

    Attributes:
        results:      List of SimilarityResult, one per job.
        ranked_indices: Job indices sorted by score (best first).
        total_duration_ms: Total time for the entire batch.
    """
    results:           List[SimilarityResult]
    ranked_indices:    List[int]
    total_duration_ms: float

    @property
    def best_match(self) -> Optional[SimilarityResult]:
        """Returns the highest scoring result."""
        if not self.results:
            return None
        return self.results[self.ranked_indices[0]]

    @property
    def above_threshold(self) -> List[Tuple[int, SimilarityResult]]:
        """
        Returns (index, result) pairs where overall_score
        meets the configured minimum threshold.
        """
        threshold = settings.min_match_score
        return [
            (i, self.results[i])
            for i in self.ranked_indices
            if self.results[i].overall_score >= threshold
        ]


# ============================================================
# Embedding Cache
# ============================================================

class EmbeddingCache:
    """
    In-memory LRU cache for text embeddings.

    Avoids recomputing embeddings for the same text.
    This matters because:
    - The same resume is compared against 20-50 jobs per run
    - Computing embeddings is CPU-intensive (~50ms per text)
    - Caching the resume embedding saves 20-50 recomputations

    Cache key: MD5 hash of the text (avoids storing full text as key)
    Cache size: Configurable, defaults to 500 entries
    """

    def __init__(self, max_size: int = 500) -> None:
        self._cache: Dict[str, np.ndarray] = {}
        self._max_size = max_size
        self._hits = 0
        self._misses = 0

    def _make_key(self, text: str) -> str:
        """Creates a compact cache key from text content."""
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def get(self, text: str) -> Optional[np.ndarray]:
        """Returns cached embedding or None if not cached."""
        key = self._make_key(text)
        embedding = self._cache.get(key)
        if embedding is not None:
            self._hits += 1
            return embedding
        self._misses += 1
        return None

    def set(self, text: str, embedding: np.ndarray) -> None:
        """Stores an embedding. Evicts oldest entry if at capacity."""
        if len(self._cache) >= self._max_size:
            # Evict the first (oldest) entry — simple FIFO eviction
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        key = self._make_key(text)
        self._cache[key] = embedding

    def clear(self) -> None:
        """Clears all cached embeddings."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    @property
    def hit_rate(self) -> float:
        """Cache hit rate as a percentage."""
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return round(self._hits / total * 100, 1)

    @property
    def stats(self) -> Dict[str, Any]:
        """Cache performance statistics."""
        return {
            "size":     len(self._cache),
            "max_size": self._max_size,
            "hits":     self._hits,
            "misses":   self._misses,
            "hit_rate": f"{self.hit_rate}%",
        }


# ============================================================
# Embedding Service Class
# ============================================================

class EmbeddingService:
    """
    Semantic similarity service using sentence-transformers.

    Architecture:
        - sentence-transformers: Deep learning model that converts
          text to 384-dimensional vectors capturing semantic meaning.
          Model: all-MiniLM-L6-v2 (fast, accurate, small size ~90MB)

        - TF-IDF: Statistical method that finds important keywords
          in documents. Used for ATS-style keyword matching.

        - Skill matching: Direct string comparison with fuzzy
          normalization (lowercase, strip, common abbreviations).

    Why three scores?
        - Semantic alone misses exact keyword matches ATS care about
        - Keyword alone misses synonyms ("JavaScript" vs "JS")
        - Skill alone misses contextual fit
        - Combined gives the most accurate overall picture

    Usage:
        service = EmbeddingService()

        # Single comparison
        result = await service.compute_similarity(
            resume_text="5 years Python, FastAPI, PostgreSQL...",
            job_description="We need a Python backend engineer...",
            resume_skills=["Python", "FastAPI", "PostgreSQL"],
            required_skills=["Python", "Django", "AWS"],
        )
        print(result.overall_score_percent)  # e.g. 72.5

        # Batch comparison (more efficient)
        batch = await service.batch_match(
            resume_text="...",
            job_descriptions=["JD 1...", "JD 2...", ...],
            resume_skills=[...],
            jobs_required_skills=[[...], [...], ...],
        )
        top_jobs = batch.above_threshold
    """

    def __init__(self) -> None:
        """
        Initializes the service without loading the model.
        Model is loaded lazily on first use to avoid slowing
        down application startup.
        """
        self._model = None           # Loaded on first use
        self._model_name = settings.embedding_model
        self._cache = EmbeddingCache(max_size=500)
        self._tfidf = TfidfVectorizer(
            stop_words="english",
            max_features=5000,       # Vocabulary cap for memory efficiency
            ngram_range=(1, 2),      # Unigrams and bigrams
            lowercase=True,
            strip_accents="unicode",
        )
        self._is_initialized = False

        logger.info(
            f"EmbeddingService created | "
            f"model={self._model_name} | "
            f"(model loads on first use)"
        )

    # ----------------------------------------------------------
    # Model Initialization
    # ----------------------------------------------------------

    def _load_model(self) -> None:
        """
        Loads the sentence-transformers model into memory.

        Called automatically on first use. Takes 2-5 seconds
        on first call (model download ~90MB on very first run).
        Subsequent calls return instantly (model cached in memory).

        The model is loaded in a separate thread via
        asyncio.to_thread() to avoid blocking the event loop.
        """
        if self._model is not None:
            return

        logger.info(
            f"Loading sentence-transformers model: {self._model_name} "
            f"(first-time load may take a few seconds...)"
        )

        start = time.perf_counter()

        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
            self._is_initialized = True

            elapsed = round((time.perf_counter() - start) * 1000, 2)
            logger.info(
                f"Model loaded successfully | "
                f"model={self._model_name} | "
                f"load_time={elapsed}ms"
            )

        except Exception as e:
            logger.error(
                f"Failed to load sentence-transformers model "
                f"'{self._model_name}': {e}\n"
                f"Run: pip install sentence-transformers"
            )
            raise RuntimeError(
                f"EmbeddingService failed to load model '{self._model_name}'. "
                f"Ensure sentence-transformers is installed and the model "
                f"name is correct. Error: {e}"
            )

    async def _ensure_model_loaded(self) -> None:
        """
        Ensures the model is loaded before any computation.
        Runs model loading in a thread to not block async event loop.
        """
        if not self._is_initialized:
            await asyncio.to_thread(self._load_model)

    # ----------------------------------------------------------
    # Primary Interface
    # ----------------------------------------------------------

    async def compute_similarity(
        self,
        resume_text: str,
        job_description: str,
        resume_skills: Optional[List[str]] = None,
        required_skills: Optional[List[str]] = None,
        preferred_skills: Optional[List[str]] = None,
    ) -> SimilarityResult:
        """
        Computes semantic, keyword, and skill similarity between
        a resume and a single job description.

        This is the PRIMARY method called by Agent 3.

        Args:
            resume_text:      Full resume text (from pdf_service).
            job_description:  Full job description text.
            resume_skills:    Flat list of candidate's skills.
            required_skills:  Skills the job requires.
            preferred_skills: Nice-to-have skills for the job.

        Returns:
            SimilarityResult: All scores and skill breakdowns.

        Raises:
            RuntimeError: If model fails to load.
        """
        await self._ensure_model_loaded()

        start = time.perf_counter()

        # Run all three scoring methods concurrently in threads
        # (CPU-bound work moved off the async event loop)
        semantic_task = asyncio.to_thread(
            self._compute_semantic_score,
            resume_text,
            job_description,
        )
        keyword_task = asyncio.to_thread(
            self._compute_keyword_score,
            resume_text,
            job_description,
        )

        semantic_score, keyword_score = await asyncio.gather(
            semantic_task,
            keyword_task,
        )

        # Skill matching is fast — run synchronously
        skill_score, matched, missing, bonus = self._compute_skill_score(
            resume_skills=resume_skills or [],
            required_skills=required_skills or [],
            preferred_skills=preferred_skills or [],
        )

        # Weighted combination
        overall_score = (
            semantic_score  * SimilarityResult.SEMANTIC_WEIGHT
            + keyword_score * SimilarityResult.KEYWORD_WEIGHT
            + skill_score   * SimilarityResult.SKILL_WEIGHT
        )

        # Clamp to [0, 1]
        overall_score = max(0.0, min(1.0, overall_score))

        duration_ms = round((time.perf_counter() - start) * 1000, 2)

        result = SimilarityResult(
            semantic_score=round(semantic_score, 4),
            keyword_score=round(keyword_score, 4),
            skill_score=round(skill_score, 4),
            overall_score=round(overall_score, 4),
            matched_skills=matched,
            missing_skills=missing,
            bonus_skills=bonus,
            duration_ms=duration_ms,
        )

        logger.debug(
            f"Similarity computed | "
            f"overall={result.overall_score_percent}% | "
            f"semantic={round(semantic_score*100,1)}% | "
            f"keyword={round(keyword_score*100,1)}% | "
            f"skill={round(skill_score*100,1)}% | "
            f"matched={len(matched)} | "
            f"missing={len(missing)} | "
            f"duration={duration_ms}ms"
        )

        return result

    async def batch_match(
        self,
        resume_text: str,
        job_descriptions: List[str],
        resume_skills: Optional[List[str]] = None,
        jobs_required_skills: Optional[List[List[str]]] = None,
        jobs_preferred_skills: Optional[List[List[str]]] = None,
    ) -> BatchMatchResult:
        """
        Efficiently matches one resume against multiple job descriptions.

        More efficient than calling compute_similarity() in a loop
        because the resume embedding is computed once and reused
        for all job comparisons.

        Args:
            resume_text:           Full resume text.
            job_descriptions:      List of job description texts.
            resume_skills:         Candidate's skill list.
            jobs_required_skills:  List of required skill lists,
                                   one per job description.
            jobs_preferred_skills: List of preferred skill lists,
                                   one per job description.

        Returns:
            BatchMatchResult: All results ranked by overall score.

        Example:
            batch = await service.batch_match(
                resume_text=resume.raw_text,
                job_descriptions=[j.job_description for j in jobs],
                resume_skills=resume.skills,
                jobs_required_skills=[j.required_skills for j in jobs],
            )
            for rank, (idx, result) in enumerate(batch.above_threshold):
                print(f"#{rank+1}: {result.overall_score_percent}%")
        """
        await self._ensure_model_loaded()

        if not job_descriptions:
            return BatchMatchResult(
                results=[],
                ranked_indices=[],
                total_duration_ms=0.0,
            )

        start = time.perf_counter()
        n_jobs = len(job_descriptions)

        # Normalize optional lists to correct length
        req_skills_list  = jobs_required_skills  or [[] for _ in range(n_jobs)]
        pref_skills_list = jobs_preferred_skills or [[] for _ in range(n_jobs)]

        logger.info(
            f"Batch matching started | "
            f"jobs={n_jobs} | "
            f"resume_skills={len(resume_skills or [])}"
        )

        # ── Step 1: Compute resume embedding once ──────────────
        resume_embedding = await asyncio.to_thread(
            self._get_embedding, resume_text
        )

        # ── Step 2: Compute all job embeddings in one batch ────
        # SentenceTransformer.encode() is optimized for batches —
        # significantly faster than encoding one at a time
        job_embeddings = await asyncio.to_thread(
            self._get_batch_embeddings, job_descriptions
        )

        # ── Step 3: Compute all semantic scores at once ────────
        # cosine_similarity(1 x D, N x D) → (1 x N) array
        resume_emb_2d = resume_embedding.reshape(1, -1)
        semantic_scores = cosine_similarity(
            resume_emb_2d, job_embeddings
        )[0]   # Shape: (N,)

        # ── Step 4: Compute keyword + skill scores per job ─────
        results: List[SimilarityResult] = []

        for i in range(n_jobs):
            keyword_score = await asyncio.to_thread(
                self._compute_keyword_score,
                resume_text,
                job_descriptions[i],
            )

            skill_score, matched, missing, bonus = self._compute_skill_score(
                resume_skills=resume_skills or [],
                required_skills=req_skills_list[i],
                preferred_skills=pref_skills_list[i],
            )

            sem_score = float(semantic_scores[i])
            overall = (
                sem_score   * SimilarityResult.SEMANTIC_WEIGHT
                + keyword_score * SimilarityResult.KEYWORD_WEIGHT
                + skill_score   * SimilarityResult.SKILL_WEIGHT
            )
            overall = max(0.0, min(1.0, overall))

            results.append(SimilarityResult(
                semantic_score=round(sem_score, 4),
                keyword_score=round(keyword_score, 4),
                skill_score=round(skill_score, 4),
                overall_score=round(overall, 4),
                matched_skills=matched,
                missing_skills=missing,
                bonus_skills=bonus,
                duration_ms=0.0,  # Individual timing not tracked in batch
            ))

        # ── Step 5: Rank by overall score (highest first) ──────
        ranked_indices = sorted(
            range(n_jobs),
            key=lambda i: results[i].overall_score,
            reverse=True,
        )

        total_ms = round((time.perf_counter() - start) * 1000, 2)

        # Log cache performance after batch
        logger.info(
            f"Batch matching complete | "
            f"jobs={n_jobs} | "
            f"total_duration={total_ms}ms | "
            f"avg_duration={round(total_ms/n_jobs, 1)}ms/job | "
            f"cache_stats={self._cache.stats}"
        )

        return BatchMatchResult(
            results=results,
            ranked_indices=ranked_indices,
            total_duration_ms=total_ms,
        )

    # ----------------------------------------------------------
    # Scoring Methods (run in threads — CPU bound)
    # ----------------------------------------------------------

    def _compute_semantic_score(
        self,
        text_a: str,
        text_b: str,
    ) -> float:
        """
        Computes cosine similarity between two text embeddings.

        Uses sentence-transformers to convert both texts into
        384-dimensional dense vectors, then computes cosine
        similarity between them.

        Semantic similarity captures meaning and context:
        - "software engineer" ≈ "developer" (high similarity)
        - "Python expert" ≈ "experienced in Python" (high)
        - "Python" ≈ "Java" (medium — both programming languages)
        - "Python" ≈ "cooking" (low similarity)

        Args:
            text_a: First text (resume text).
            text_b: Second text (job description).

        Returns:
            float: Cosine similarity score in [0.0, 1.0].
        """
        # Truncate long texts — model has 256 token limit per input
        # ~1000 chars ≈ 256 tokens is safe
        max_chars = 4000
        text_a_trunc = text_a[:max_chars]
        text_b_trunc = text_b[:max_chars]

        emb_a = self._get_embedding(text_a_trunc)
        emb_b = self._get_embedding(text_b_trunc)

        # cosine_similarity expects 2D arrays
        score = cosine_similarity(
            emb_a.reshape(1, -1),
            emb_b.reshape(1, -1),
        )[0][0]

        # Scores from sentence-transformers can occasionally be
        # slightly outside [0,1] due to floating point — clamp
        return float(max(0.0, min(1.0, score)))

    def _compute_keyword_score(
        self,
        resume_text: str,
        job_description: str,
    ) -> float:
        """
        Computes TF-IDF based keyword overlap score.

        TF-IDF (Term Frequency-Inverse Document Frequency) finds
        the most important terms in the job description and checks
        how many appear in the resume. This mimics what ATS systems
        do when scoring resumes.

        Unlike semantic scoring, this is sensitive to exact matches:
        - Job says "React.js" → resume must say "React.js" (or "React")
        - Abbreviations and variations may not match

        Args:
            resume_text:      Candidate's resume text.
            job_description:  Job listing text.

        Returns:
            float: Keyword overlap score in [0.0, 1.0].
        """
        try:
            # Fit TF-IDF on both texts together, then compare
            tfidf_matrix = self._tfidf.fit_transform(
                [resume_text, job_description]
            )

            # Cosine similarity between the two TF-IDF vectors
            score = cosine_similarity(
                tfidf_matrix[0:1],   # Resume vector
                tfidf_matrix[1:2],   # Job description vector
            )[0][0]

            return float(max(0.0, min(1.0, score)))

        except Exception as e:
            logger.warning(f"TF-IDF scoring failed: {e} — returning 0.0")
            return 0.0

    def _compute_skill_score(
        self,
        resume_skills: List[str],
        required_skills: List[str],
        preferred_skills: List[str],
    ) -> Tuple[float, List[str], List[str], List[str]]:
        """
        Computes direct skill overlap between resume and job.

        Matching strategy:
            1. Normalize all skills (lowercase, strip whitespace)
            2. Build skill aliases map (js → javascript, etc.)
            3. Check each required skill against resume skills
            4. Check preferred skills for bonus coverage
            5. Find bonus resume skills not in requirements

        Args:
            resume_skills:    Skills from the candidate's resume.
            required_skills:  Skills the job requires.
            preferred_skills: Nice-to-have skills.

        Returns:
            Tuple of:
                float:       Skill match score [0.0, 1.0]
                List[str]:   Matched required skills
                List[str]:   Missing required skills
                List[str]:   Bonus resume skills (not required)
        """
        if not required_skills:
            # No required skills listed — give partial credit
            # for having any skills at all
            if resume_skills:
                return 0.5, [], [], list(resume_skills[:10])
            return 0.5, [], [], []

        # Normalize helper
        def normalize(skill: str) -> str:
            return skill.lower().strip()

        # Build normalized resume skill set with aliases
        resume_normalized = {normalize(s) for s in resume_skills}
        resume_with_aliases = self._expand_with_aliases(resume_normalized)

        # Match required skills
        matched: List[str] = []
        missing: List[str] = []

        for req_skill in required_skills:
            req_norm = normalize(req_skill)
            req_aliases = self._expand_with_aliases({req_norm})

            # Check if any alias of required skill appears
            # in any alias of resume skills
            if req_aliases & resume_with_aliases:
                matched.append(req_skill)
            else:
                missing.append(req_skill)

        # Find bonus skills: in resume but not required/preferred
        all_job_skills_norm = {
            normalize(s)
            for s in (required_skills + preferred_skills)
        }
        bonus: List[str] = [
            s for s in resume_skills
            if normalize(s) not in all_job_skills_norm
        ][:10]   # Cap bonus skills at 10 for readability

        # Compute score: matched / total required
        n_required = len(required_skills)
        score = len(matched) / n_required if n_required > 0 else 0.5

        # Small bonus for preferred skill coverage
        if preferred_skills:
            pref_matched = sum(
                1 for s in preferred_skills
                if self._expand_with_aliases({normalize(s)}) & resume_with_aliases
            )
            pref_bonus = (pref_matched / len(preferred_skills)) * 0.1
            score = min(1.0, score + pref_bonus)

        return (
            round(float(score), 4),
            matched,
            missing,
            bonus,
        )

    # ----------------------------------------------------------
    # Embedding Helpers
    # ----------------------------------------------------------

    def _get_embedding(self, text: str) -> np.ndarray:
        """
        Returns the sentence embedding for a text string.

        Checks cache first — computes and caches if not found.
        The cache eliminates recomputation of the same resume
        text across multiple job comparisons in a pipeline run.

        Args:
            text: Input text to embed.

        Returns:
            np.ndarray: 384-dimensional embedding vector.
        """
        # Check cache
        cached = self._cache.get(text)
        if cached is not None:
            return cached

        # Compute embedding
        embedding = self._model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2 normalization for cosine sim
            show_progress_bar=False,
        )

        # Store in cache
        self._cache.set(text, embedding)

        return embedding

    def _get_batch_embeddings(
        self,
        texts: List[str],
    ) -> np.ndarray:
        """
        Encodes multiple texts in a single batch.

        SentenceTransformer.encode() is significantly faster
        when processing a batch vs. encoding one text at a time
        because it can use GPU parallelism (if available) or
        optimized CPU batching.

        Args:
            texts: List of texts to encode.

        Returns:
            np.ndarray: Shape (N, 384) — one embedding per text.
        """
        # Check which texts are already cached
        embeddings = []
        uncached_indices = []
        uncached_texts = []

        for i, text in enumerate(texts):
            cached = self._cache.get(text)
            if cached is not None:
                embeddings.append((i, cached))
            else:
                uncached_indices.append(i)
                uncached_texts.append(text)

        # Batch encode uncached texts
        if uncached_texts:
            batch_embeddings = self._model.encode(
                uncached_texts,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
                batch_size=32,     # Process 32 texts at a time
            )

            # Cache and store the new embeddings
            for idx, (orig_idx, text) in enumerate(
                zip(uncached_indices, uncached_texts)
            ):
                emb = batch_embeddings[idx]
                self._cache.set(text, emb)
                embeddings.append((orig_idx, emb))

        # Sort by original index and stack into matrix
        embeddings.sort(key=lambda x: x[0])
        return np.vstack([emb for _, emb in embeddings])

    # ----------------------------------------------------------
    # Skill Alias Expansion
    # ----------------------------------------------------------

    def _expand_with_aliases(
        self,
        skills: set,
    ) -> set:
        """
        Expands a skill set with known technology aliases.

        This prevents mismatches caused by naming variations:
            "js" → also matches "javascript"
            "ml" → also matches "machine learning"
            "k8s" → also matches "kubernetes"

        The alias map covers the most common tech abbreviations.
        Matching is bidirectional: if resume has "js" and job
        requires "javascript", they match — and vice versa.

        Args:
            skills: Normalized (lowercase) skill strings.

        Returns:
            set: Original skills + all known aliases.
        """
        aliases: Dict[str, List[str]] = {
            # JavaScript ecosystem
            "js":               ["javascript"],
            "javascript":       ["js"],
            "ts":               ["typescript"],
            "typescript":       ["ts"],
            "nodejs":           ["node.js", "node"],
            "node.js":          ["nodejs", "node"],
            "reactjs":          ["react", "react.js"],
            "react":            ["reactjs", "react.js"],
            "vuejs":            ["vue", "vue.js"],
            "angularjs":        ["angular"],

            # Python
            "py":               ["python"],
            "python":           ["py"],
            "ml":               ["machine learning"],
            "machine learning": ["ml"],
            "dl":               ["deep learning"],
            "deep learning":    ["dl"],
            "nlp":              ["natural language processing"],

            # Cloud & Infrastructure
            "aws":              ["amazon web services"],
            "gcp":              ["google cloud", "google cloud platform"],
            "azure":            ["microsoft azure"],
            "k8s":              ["kubernetes"],
            "kubernetes":       ["k8s"],
            "tf":               ["terraform"],

            # Databases
            "pg":               ["postgresql", "postgres"],
            "postgresql":       ["pg", "postgres"],
            "postgres":         ["postgresql", "pg"],
            "mongo":            ["mongodb"],
            "mongodb":          ["mongo"],
            "es":               ["elasticsearch"],

            # Languages
            "c++":              ["cpp"],
            "cpp":              ["c++"],
            "golang":           ["go"],
            "go":               ["golang"],

            # Data
            "pandas":           ["dataframes"],
            "sklearn":          ["scikit-learn"],
            "scikit-learn":     ["sklearn"],
            "pytorch":          ["torch"],
            "torch":            ["pytorch"],

            # CI/CD
            "ci/cd":            ["cicd", "ci cd", "continuous integration"],
            "cicd":             ["ci/cd"],

            # Other
            "oop":              ["object oriented programming",
                                 "object-oriented programming"],
            "api":              ["rest api", "restful api", "rest"],
            "rest":             ["api", "rest api", "restful"],
        }

        expanded = set(skills)
        for skill in list(skills):
            for alias in aliases.get(skill, []):
                expanded.add(alias)

        return expanded

    # ----------------------------------------------------------
    # Utility Methods
    # ----------------------------------------------------------

    async def extract_keywords(
        self,
        text: str,
        top_n: int = 20,
    ) -> List[str]:
        """
        Extracts the most important keywords from a text
        using TF-IDF scoring.

        Used by Agent 5 (ATS Optimizer) to find which keywords
        from the job description are missing from the resume.

        Args:
            text:  Source text (job description or resume).
            top_n: Number of top keywords to return.

        Returns:
            List[str]: Top keywords sorted by TF-IDF importance.

        Usage in ATS Agent:
            jd_keywords = await embedding_service.extract_keywords(
                text=job.job_description,
                top_n=20,
            )
            # → ["machine learning", "python", "tensorflow", ...]
        """
        try:
            vectorizer = TfidfVectorizer(
                stop_words="english",
                max_features=top_n * 3,
                ngram_range=(1, 2),
                lowercase=True,
            )

            tfidf_matrix = await asyncio.to_thread(
                vectorizer.fit_transform, [text]
            )

            feature_names = vectorizer.get_feature_names_out()
            scores = tfidf_matrix.toarray()[0]

            # Sort by score descending
            ranked = sorted(
                zip(feature_names, scores),
                key=lambda x: x[1],
                reverse=True,
            )

            return [word for word, score in ranked[:top_n] if score > 0]

        except Exception as e:
            logger.warning(f"Keyword extraction failed: {e}")
            return []

    async def find_skill_gaps(
        self,
        resume_skills: List[str],
        required_skills: List[str],
        preferred_skills: Optional[List[str]] = None,
    ) -> Dict[str, List[str]]:
        """
        Returns detailed skill gap analysis.

        Convenience wrapper around _compute_skill_score()
        that returns a clean dict. Used directly by Agent 4.

        Args:
            resume_skills:    Skills the candidate has.
            required_skills:  Skills the job requires.
            preferred_skills: Nice-to-have skills.

        Returns:
            Dict with keys:
                matched:   Skills candidate has ✓
                missing:   Required skills candidate lacks ✗
                bonus:     Candidate skills not in requirements
                preferred_matched: Preferred skills candidate has
                preferred_missing: Preferred skills candidate lacks
        """
        _, matched, missing, bonus = self._compute_skill_score(
            resume_skills=resume_skills,
            required_skills=required_skills,
            preferred_skills=preferred_skills or [],
        )

        # Also analyze preferred skills separately
        pref_matched: List[str] = []
        pref_missing: List[str] = []

        if preferred_skills:
            def normalize(s: str) -> str:
                return s.lower().strip()

            resume_norm = self._expand_with_aliases(
                {normalize(s) for s in resume_skills}
            )

            for pref in preferred_skills:
                pref_aliases = self._expand_with_aliases(
                    {normalize(pref)}
                )
                if pref_aliases & resume_norm:
                    pref_matched.append(pref)
                else:
                    pref_missing.append(pref)

        return {
            "matched":            matched,
            "missing":            missing,
            "bonus":              bonus,
            "preferred_matched":  pref_matched,
            "preferred_missing":  pref_missing,
        }

    async def health_check(self) -> Dict[str, Any]:
        """
        Verifies the embedding service is functional.

        Loads the model if not already loaded and runs
        a quick similarity test to confirm end-to-end operation.

        Returns:
            Dict with status, model info, and cache stats.
        """
        try:
            await self._ensure_model_loaded()

            # Quick sanity check — identical texts should score ~1.0
            start = time.perf_counter()
            result = await self.compute_similarity(
                resume_text="Python developer with FastAPI experience",
                job_description="Python developer with FastAPI experience",
            )
            latency_ms = round((time.perf_counter() - start) * 1000, 2)

            return {
                "status":        "healthy",
                "model":         self._model_name,
                "test_score":    result.overall_score,
                "latency_ms":    latency_ms,
                "cache_stats":   self._cache.stats,
                "is_initialized": self._is_initialized,
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error":  str(e),
            }

    def clear_cache(self) -> None:
        """Clears the embedding cache. Useful for testing."""
        self._cache.clear()
        logger.info("Embedding cache cleared")

    @property
    def cache_stats(self) -> Dict[str, Any]:
        """Returns current cache statistics."""
        return self._cache.stats


# ============================================================
# Module-Level Singleton
# ============================================================

embedding_service: EmbeddingService = EmbeddingService()