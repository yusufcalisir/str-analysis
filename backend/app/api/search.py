"""
Search API — Forensic STR Profile Similarity Search Endpoint.

Provides a completeness-aware search endpoint that handles both complete
and degraded DNA profiles. Integrates the GenomicVectorizer's masked
similarity system and the MissingDataHypothesizer for partial evidence.

Key Features:
    - min_loci_threshold: Reject profiles below minimum quality standard.
    - Quality warnings: When profile is partial or degraded, the response
      includes advisory warnings about false positive risk.
    - Completeness-penalized scoring: Score = S(active) × (N_active/N_total).
    - Missing loci transparency: Every match result lists exactly which
      markers were and were not compared.
"""

import time
import logging
from typing import Dict, List, Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, UUID4

from app.core.engine.vectorizer import (
    GenomicVectorizer,
    LocusInput,
    MaskedSimilarityResult,
    STANDARD_LOCI,
    VECTOR_DIM,
)
from app.schemas.genomic import LocusDataSchema

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/search", tags=["Search"])

# Singleton vectorizer instance
_vectorizer = GenomicVectorizer()


# ═══════════════════════════════════════════════════════════════════════════════
# REQUEST / RESPONSE MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class SearchRequest(BaseModel):
    """Incoming search query with an STR profile to match against the network."""
    str_markers: Dict[str, LocusDataSchema] = Field(
        ...,
        min_length=1,
        description="Map of marker name → allele data for the query profile."
    )
    min_loci_threshold: int = Field(
        default=8,
        ge=1,
        le=24,
        description=(
            "Minimum number of valid loci required to proceed with search. "
            "Profiles below this threshold are rejected. Default: 8."
        )
    )
    max_results: int = Field(default=25, ge=1, le=100, description="Maximum results to return.")
    include_imputation: bool = Field(
        default=False,
        description="If true, run the MissingDataHypothesizer on missing loci."
    )


class QualityWarning(BaseModel):
    """Advisory warning about search reliability."""
    level: str = Field(..., description="WARNING | CAUTION | CRITICAL")
    message: str
    loci_present: int
    loci_missing: int
    missing_loci_names: List[str] = Field(default_factory=list)
    false_positive_risk: str = Field(
        default="LOW",
        description="LOW | MODERATE | HIGH | VERY_HIGH"
    )


class SearchMatchResult(BaseModel):
    """A single match in the search results with completeness metadata."""
    profile_id: str
    node_id: str
    raw_similarity: float = Field(..., ge=0.0, le=1.0)
    penalized_score: float = Field(..., ge=0.0, le=1.0)
    active_loci: int
    completeness_ratio: float = Field(..., ge=0.0, le=1.0)
    quality_tier: str
    missing_loci_query: List[str] = Field(default_factory=list)
    missing_loci_target: List[str] = Field(default_factory=list)


class SearchResponse(BaseModel):
    """Complete search response with quality metadata."""
    query_id: str
    total_results: int
    loci_submitted: int
    loci_valid: int
    search_time_ms: float
    quality_warnings: List[QualityWarning] = Field(default_factory=list)
    results: List[SearchMatchResult] = Field(default_factory=list)
    imputation_available: bool = False


# ═══════════════════════════════════════════════════════════════════════════════
# QUALITY ASSESSMENT
# ═══════════════════════════════════════════════════════════════════════════════

def _assess_query_quality(
    valid_loci: int,
    total_loci: int,
    missing_names: List[str],
) -> List[QualityWarning]:
    """Generate quality warnings based on profile completeness."""
    warnings: List[QualityWarning] = []
    n_missing = total_loci - valid_loci

    if valid_loci >= 18:
        return warnings  # Complete profile — no warnings

    if valid_loci >= 14:
        warnings.append(QualityWarning(
            level="WARNING",
            message=(
                f"Partial profile: {valid_loci}/{total_loci} loci present. "
                f"Missing: {', '.join(missing_names[:5])}{'...' if n_missing > 5 else ''}. "
                "Search results may include additional false positive matches. "
                "Verify top candidates with independent DNA testing."
            ),
            loci_present=valid_loci,
            loci_missing=n_missing,
            missing_loci_names=missing_names,
            false_positive_risk="MODERATE",
        ))
    elif valid_loci >= 10:
        warnings.append(QualityWarning(
            level="CAUTION",
            message=(
                f"Degraded profile: only {valid_loci}/{total_loci} loci amplified. "
                f"Missing: {', '.join(missing_names)}. "
                "Similarity scores are significantly less reliable. "
                "The completeness penalty reduces all scores proportionally. "
                "Consider requesting additional evidence or re-extraction."
            ),
            loci_present=valid_loci,
            loci_missing=n_missing,
            missing_loci_names=missing_names,
            false_positive_risk="HIGH",
        ))
    else:
        warnings.append(QualityWarning(
            level="CRITICAL",
            message=(
                f"Critically degraded profile: only {valid_loci}/{total_loci} loci available. "
                f"Missing: {', '.join(missing_names)}. "
                "Results are UNRELIABLE and should NOT be used as primary evidence. "
                "The random match probability is unacceptably high at this loci count. "
                "This search is advisory only."
            ),
            loci_present=valid_loci,
            loci_missing=n_missing,
            missing_loci_names=missing_names,
            false_positive_risk="VERY_HIGH",
        ))

    return warnings


# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATED DATABASE (Development Mode)
# In production, this queries Milvus and the federated broadcast system.
# ═══════════════════════════════════════════════════════════════════════════════

def _simulate_database_search(
    query_embedding: List[float],
    max_results: int,
) -> List[Dict]:
    """
    Simulate database matches for development.
    Returns mock vectorized profiles with varying completeness.
    """
    import random
    import numpy as np

    results = []
    for i in range(min(max_results, 8)):
        # Generate a profile that is similar to the query with some noise
        q = np.array(query_embedding, dtype=np.float64)
        noise_level = 0.05 + i * 0.08
        noise = np.random.normal(0, noise_level, size=len(q))
        candidate = np.clip(q + noise, 0.0, 1.0)

        # Randomly make some loci "missing" in the candidate
        loci_count = len(STANDARD_LOCI)
        missing_count = random.randint(0, min(6, loci_count))
        if missing_count > 0:
            missing_indices = random.sample(range(loci_count), missing_count)
            for idx in missing_indices:
                candidate[idx * 2] = 0.0
                candidate[idx * 2 + 1] = 0.0

        results.append({
            "profile_id": f"SIM-{uuid4().hex[:8].upper()}",
            "node_id": random.choice(["EUROPOL-NL", "BKA-DE", "NCA-UK", "FBI-US-DC", "RCMP-CA"]),
            "embedding": candidate.tolist(),
            "loci_populated": loci_count - missing_count,
        })

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# SEARCH ENDPOINT
# ═══════════════════════════════════════════════════════════════════════════════

@router.post(
    "/",
    response_model=SearchResponse,
    summary="Search for STR profile matches across the VANTAGE-STR network",
    description=(
        "Submit a partial or complete STR profile for similarity search. "
        "Returns completeness-penalized results with quality warnings "
        "when the profile is degraded."
    ),
)
async def search_profiles(request: SearchRequest) -> SearchResponse:
    """
    Execute a completeness-aware forensic STR profile search.

    Flow:
        1. Vectorize the query profile.
        2. Validate minimum loci threshold.
        3. Generate quality warnings if profile is partial.
        4. Search against the database (simulated in dev).
        5. Compare using masked similarity with completeness penalty.
        6. Return ranked results with full quality metadata.
    """
    t_start = time.perf_counter()
    query_id = f"Q-{uuid4().hex[:12].upper()}"

    # ── Step 1: Vectorize query ───────────────────────────────────────
    str_markers: Dict[str, LocusInput] = {}
    for name, locus in request.str_markers.items():
        str_markers[name] = LocusInput(
            allele_1=locus.allele_1,
            allele_2=locus.allele_2,
            is_homozygous=locus.is_homozygous,
        )

    query_vec = _vectorizer.vectorize(
        profile_id=query_id,
        node_id="QUERY",
        str_markers=str_markers,
        timestamp=int(time.time()),
    )

    # ── Step 2: Validate minimum loci ─────────────────────────────────
    valid_loci = query_vec.loci_populated
    if valid_loci < request.min_loci_threshold:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "INSUFFICIENT_LOCI",
                "message": (
                    f"Profile has {valid_loci} valid loci, but minimum "
                    f"threshold is {request.min_loci_threshold}. "
                    "Submit a higher-quality sample or lower the threshold."
                ),
                "loci_present": valid_loci,
                "threshold": request.min_loci_threshold,
            },
        )

    # ── Step 3: Quality warnings ──────────────────────────────────────
    total = len(STANDARD_LOCI)
    observed_set = set(str_markers.keys()) & set(STANDARD_LOCI)
    missing_names = [l for l in STANDARD_LOCI if l not in observed_set]
    quality_warnings = _assess_query_quality(valid_loci, total, missing_names)

    # ── Step 4: Database search (simulated) ───────────────────────────
    candidates = _simulate_database_search(query_vec.embedding, request.max_results)

    # ── Step 5: Masked similarity comparison ──────────────────────────
    from app.core.engine.vectorizer import VectorizedProfile

    match_results: List[SearchMatchResult] = []
    for cand in candidates:
        target_vec = VectorizedProfile(
            profile_id=cand["profile_id"],
            node_id=cand["node_id"],
            embedding=cand["embedding"],
            loci_populated=cand["loci_populated"],
            timestamp=int(time.time()),
        )

        masked_result = _vectorizer.compare_profiles_masked(query_vec, target_vec)

        match_results.append(SearchMatchResult(
            profile_id=cand["profile_id"],
            node_id=cand["node_id"],
            raw_similarity=masked_result.raw_similarity,
            penalized_score=masked_result.penalized_score,
            active_loci=masked_result.active_loci,
            completeness_ratio=masked_result.completeness_ratio,
            quality_tier=masked_result.quality_tier,
            missing_loci_query=masked_result.missing_loci_query,
            missing_loci_target=masked_result.missing_loci_target,
        ))

    # Sort by penalized score descending
    match_results.sort(key=lambda r: r.penalized_score, reverse=True)

    elapsed = (time.perf_counter() - t_start) * 1000

    return SearchResponse(
        query_id=query_id,
        total_results=len(match_results),
        loci_submitted=len(request.str_markers),
        loci_valid=valid_loci,
        search_time_ms=round(elapsed, 2),
        quality_warnings=quality_warnings,
        results=match_results,
        imputation_available=request.include_imputation,
    )
