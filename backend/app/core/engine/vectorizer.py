"""
Genomic Vectorizer — STR Profile to High-Dimensional Vector Transformation.

Translates discrete Short Tandem Repeat (STR) marker data into a 48-dimensional
float vector for sub-second similarity search via vector mathematics (cosine
similarity) instead of exhaustive string-based allele matching.

Architecture:
    Each of the 24 standard global STR loci is assigned a fixed pair of indices
    in the output vector [i*2, i*2+1], storing allele_1 and allele_2 respectively.
    This yields a deterministic 48-float embedding for any genomic profile.

Normalization:
    Raw repeat counts (typically ranging 3–50) are min-max scaled to [0, 1]
    using biologically plausible bounds. This prevents high-repeat loci (e.g.,
    SE33 with values up to 47) from dominating distance computations over
    low-repeat loci (e.g., TH01 with values around 6–10).

Determinism Guarantee:
    The same input profile MUST produce the exact same vector output regardless
    of platform, Python version, or execution order. This is critical for
    cross-node consistency in a decentralized forensic network where vectors
    computed on NODE-A must be directly comparable to vectors on NODE-B.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field, UUID4


# ═══════════════════════════════════════════════════════════════════════════════
# LOCUS INDEX MAP
# Deterministic mapping of 24 standard STR loci → vector index pairs.
# Sorted alphabetically to guarantee cross-platform consistency.
# Each locus occupies indices [i*2, i*2+1] in the 48-dim output vector.
# ═══════════════════════════════════════════════════════════════════════════════

STANDARD_LOCI: List[str] = sorted([
    # Core CODIS 13
    "CSF1PO", "D3S1358", "D5S818", "D7S820", "D8S1179",
    "D13S317", "D16S539", "D18S51", "D21S11", "FGA",
    "TH01", "TPOX", "VWA",
    # Extended CODIS 20 (2017+)
    "D1S1656", "D2S441", "D2S1338", "D10S1248", "D12S391",
    "D19S433", "D22S1045",
    # European Standard Set (ESS) additions
    "SE33",
    # Amelogenin — sex-determining marker
    "AMEL",
    # Penta D & Penta E — common in commercial kits
    "PENTA_D", "PENTA_E",
])

LOCUS_TO_INDEX: Dict[str, int] = {locus: i for i, locus in enumerate(STANDARD_LOCI)}

VECTOR_DIM: int = len(STANDARD_LOCI) * 2  # 48

# ═══════════════════════════════════════════════════════════════════════════════
# NORMALIZATION BOUNDS
# Biologically plausible allele repeat count range for min-max scaling.
# MIN_REPEAT: Lowest observed allele across all forensic loci (~3.0 for TH01).
# MAX_REPEAT: Highest observed allele across all forensic loci (~47.0 for SE33).
# ═══════════════════════════════════════════════════════════════════════════════

MIN_REPEAT: float = 2.0
MAX_REPEAT: float = 50.0
_REPEAT_RANGE: float = MAX_REPEAT - MIN_REPEAT


# ═══════════════════════════════════════════════════════════════════════════════
# DATA MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class LocusInput(BaseModel):
    """Allele data for a single STR locus, input to the vectorizer."""
    allele_1: float = Field(..., gt=0, le=100.0)
    allele_2: float = Field(..., gt=0, le=100.0)
    is_homozygous: bool = False


class VectorizedProfile(BaseModel):
    """Result of vectorization — a profile ID paired with its 48-dim embedding."""
    profile_id: str
    node_id: str
    embedding: List[float] = Field(..., min_length=VECTOR_DIM, max_length=VECTOR_DIM)
    loci_populated: int = Field(..., ge=0, le=len(STANDARD_LOCI))
    timestamp: int


class SimilarityResult(BaseModel):
    """Result of comparing two genomic vectors."""
    query_profile_id: str
    target_profile_id: str
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    shared_loci: int


class MaskedSimilarityResult(BaseModel):
    """
    Completeness-aware similarity result for partial DNA profiles.

    Applies the Masked Similarity formula:
        Score = S(u_active, v_active) × (N_active / N_total)

    Where:
        S(u_active, v_active) = cosine similarity on ONLY the indices
            where BOTH vectors have non-zero data.
        N_active = number of loci where both profiles have data.
        N_total = 24 (full CODIS+ESS+Penta panel).

    This ensures a 100% match on 5 loci scores lower than a 100% match
    on 20 loci, preventing false positives from degraded evidence.
    """
    query_profile_id: str
    target_profile_id: str
    raw_similarity: float = Field(..., ge=0.0, le=1.0, description="Cosine similarity on active loci only")
    penalized_score: float = Field(..., ge=0.0, le=1.0, description="Score × (active/total) completeness penalty")
    active_loci: int = Field(..., ge=0, description="Number of loci compared (non-zero in both)")
    total_loci: int = Field(default=len(STANDARD_LOCI), description="Total possible loci (24)")
    completeness_ratio: float = Field(..., ge=0.0, le=1.0, description="active_loci / total_loci")
    missing_loci_query: List[str] = Field(default_factory=list, description="Loci missing from the query profile")
    missing_loci_target: List[str] = Field(default_factory=list, description="Loci missing from the target profile")
    quality_tier: str = Field(default="complete", description="complete (≥18) | partial (10-17) | degraded (<10)")


# ═══════════════════════════════════════════════════════════════════════════════
# VECTORIZER
# ═══════════════════════════════════════════════════════════════════════════════

class GenomicVectorizer:
    """
    Transforms GenomicProfile STR marker data into a normalized 48-dimensional
    float vector for high-performance similarity search.

    Design principles:
        - Deterministic: identical input → identical output, always.
        - Normalized: allele repeat counts scaled to [0, 1] via min-max.
        - Sparse-safe: missing loci default to 0.0 (neutral in cosine space).

    Usage:
        vectorizer = GenomicVectorizer()
        result = vectorizer.vectorize(profile_id, node_id, markers, timestamp)
        score = vectorizer.cosine_similarity(vec_a, vec_b)
    """

    def __init__(
        self,
        neutral_value: float = 0.0,
        custom_means: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> None:
        """
        Initialize the vectorizer.

        Args:
            neutral_value: Default value for missing loci. 0.0 is neutral in
                cosine similarity (contributes nothing to dot product). Use
                population-specific means for forensic accuracy at the cost
                of introducing bias toward specific ethnic frequencies.
            custom_means: Optional dict mapping locus name → (mean_allele_1,
                mean_allele_2) for population-specific imputation of missing
                data. When provided, missing loci use these means instead of
                the neutral_value.
        """
        self._neutral_value = neutral_value
        self._custom_means = custom_means or {}

    def _normalize_allele(self, raw_value: float) -> float:
        """
        Min-max normalize a single allele repeat count to [0, 1].

        The normalization ensures that high-repeat loci like SE33 (up to 47
        repeats) do not disproportionately dominate the vector magnitude over
        low-repeat loci like TH01 (6–10 repeats). Without this, cosine
        similarity would be biased toward high-count markers.

        Args:
            raw_value: Raw allele repeat count (e.g., 14.0).

        Returns:
            Normalized value clamped to [0.0, 1.0].
        """
        normalized = (raw_value - MIN_REPEAT) / _REPEAT_RANGE
        return float(np.clip(normalized, 0.0, 1.0))

    def vectorize(
        self,
        profile_id: str,
        node_id: str,
        str_markers: Dict[str, LocusInput],
        timestamp: int,
    ) -> VectorizedProfile:
        """
        Transform a set of STR markers into a 48-dimensional normalized vector.

        The transformation is strictly deterministic:
        1. Initialize a zero-filled 48-dim array.
        2. For each known locus present in the input, write the normalized
           allele values to indices [locus_index*2, locus_index*2+1].
        3. For missing loci, use the neutral value (0.0) or population mean.
        4. Return the vector alongside metadata.

        Args:
            profile_id: UUID v4 string identifying this profile.
            node_id: Originating node/agency identifier.
            str_markers: Dict mapping marker name → LocusInput data.
            timestamp: Unix epoch timestamp of profile creation.

        Returns:
            VectorizedProfile with the 48-float embedding and metadata.

        Raises:
            ValueError: If profile_id or node_id is empty.
        """
        if not profile_id:
            raise ValueError("profile_id is required for vectorization")
        if not node_id:
            raise ValueError("node_id is required for vectorization")

        # Initialize with neutral values
        vector: NDArray[np.float64] = np.full(VECTOR_DIM, self._neutral_value, dtype=np.float64)
        loci_populated: int = 0

        # Apply population means for missing loci if configured
        for locus_name, (mean_a1, mean_a2) in self._custom_means.items():
            if locus_name in LOCUS_TO_INDEX:
                idx = LOCUS_TO_INDEX[locus_name]
                vector[idx * 2] = self._normalize_allele(mean_a1)
                vector[idx * 2 + 1] = self._normalize_allele(mean_a2)

        # Overwrite with actual observed allele data
        for marker_name, locus_data in str_markers.items():
            if marker_name not in LOCUS_TO_INDEX:
                # Unknown marker — skip silently.
                # In production, log to audit trail for compliance.
                continue

            idx = LOCUS_TO_INDEX[marker_name]
            vector[idx * 2] = self._normalize_allele(locus_data.allele_1)
            vector[idx * 2 + 1] = self._normalize_allele(locus_data.allele_2)
            loci_populated += 1

        return VectorizedProfile(
            profile_id=profile_id,
            node_id=node_id,
            embedding=vector.tolist(),
            loci_populated=loci_populated,
            timestamp=timestamp,
        )

    @staticmethod
    def cosine_similarity(
        u: List[float],
        v: List[float],
    ) -> float:
        """
        Compute cosine similarity between two genomic vectors.

        Formula: S(u, v) = (u · v) / (‖u‖ × ‖v‖)

        Returns a confidence score in [0, 1]:
            - 1.0 = identical profiles (perfect directional alignment).
            - 0.0 = orthogonal profiles (no shared allele signal).
            - Values near 1.0 with 20+ shared loci indicate a forensic match
              with random match probability < 1 in 10^18.

        Edge cases:
            - If either vector is all-zero (completely missing data), returns 0.0
              to avoid division by zero and prevent false matches.

        Args:
            u: First 48-dimensional genomic vector.
            v: Second 48-dimensional genomic vector.

        Returns:
            Cosine similarity score clamped to [0.0, 1.0].
        """
        u_arr: NDArray[np.float64] = np.asarray(u, dtype=np.float64)
        v_arr: NDArray[np.float64] = np.asarray(v, dtype=np.float64)

        dot_product: float = float(np.dot(u_arr, v_arr))
        norm_u: float = float(np.linalg.norm(u_arr))
        norm_v: float = float(np.linalg.norm(v_arr))

        # Guard against division by zero (empty/missing profiles)
        if norm_u == 0.0 or norm_v == 0.0:
            return 0.0

        similarity: float = dot_product / (norm_u * norm_v)

        # Clamp to [0, 1] — negative cosine is meaningless in this domain
        # since all normalized allele values are non-negative.
        return float(np.clip(similarity, 0.0, 1.0))

    @staticmethod
    def masked_cosine_similarity(
        u: List[float],
        v: List[float],
    ) -> Tuple[float, int, List[str], List[str]]:
        """
        Compute cosine similarity ONLY on loci where BOTH vectors are non-zero.

        This is the core of the partial-profile handling system. Instead of
        penalizing missing loci as zero-contributions to the dot product
        (which artificially deflates scores), we mask them out entirely and
        compute similarity only on the overlapping, informative dimensions.

        Args:
            u: First 48-dimensional genomic vector.
            v: Second 48-dimensional genomic vector.

        Returns:
            Tuple of (similarity_score, active_loci_count,
                      missing_from_u, missing_from_v).
        """
        u_arr = np.asarray(u, dtype=np.float64)
        v_arr = np.asarray(v, dtype=np.float64)

        active_loci: int = 0
        missing_u: List[str] = []
        missing_v: List[str] = []
        active_mask = np.zeros(VECTOR_DIM, dtype=bool)

        for i, locus_name in enumerate(STANDARD_LOCI):
            u_has = u_arr[i * 2] != 0.0 or u_arr[i * 2 + 1] != 0.0
            v_has = v_arr[i * 2] != 0.0 or v_arr[i * 2 + 1] != 0.0

            if u_has and v_has:
                active_loci += 1
                active_mask[i * 2] = True
                active_mask[i * 2 + 1] = True
            else:
                if not u_has:
                    missing_u.append(locus_name)
                if not v_has:
                    missing_v.append(locus_name)

        if active_loci == 0:
            return 0.0, 0, missing_u, missing_v

        # Extract only the active dimensions
        u_active = u_arr[active_mask]
        v_active = v_arr[active_mask]

        dot = float(np.dot(u_active, v_active))
        norm_u = float(np.linalg.norm(u_active))
        norm_v = float(np.linalg.norm(v_active))

        if norm_u == 0.0 or norm_v == 0.0:
            return 0.0, active_loci, missing_u, missing_v

        sim = float(np.clip(dot / (norm_u * norm_v), 0.0, 1.0))
        return sim, active_loci, missing_u, missing_v

    def compare_profiles(
        self,
        profile_a: VectorizedProfile,
        profile_b: VectorizedProfile,
    ) -> SimilarityResult:
        """
        Compare two vectorized profiles and return a structured similarity result.

        This is the high-level entry point for forensic comparison. It computes
        the cosine similarity and counts the number of shared populated loci
        to contextualize the confidence score.

        Args:
            profile_a: First vectorized genomic profile.
            profile_b: Second vectorized genomic profile.

        Returns:
            SimilarityResult with confidence score and shared loci count.
        """
        # Count shared loci: both must have non-zero values at the same index pair
        shared: int = 0
        for i in range(len(STANDARD_LOCI)):
            a_has = profile_a.embedding[i * 2] != 0.0 or profile_a.embedding[i * 2 + 1] != 0.0
            b_has = profile_b.embedding[i * 2] != 0.0 or profile_b.embedding[i * 2 + 1] != 0.0
            if a_has and b_has:
                shared += 1

        score = self.cosine_similarity(profile_a.embedding, profile_b.embedding)

        return SimilarityResult(
            query_profile_id=profile_a.profile_id,
            target_profile_id=profile_b.profile_id,
            confidence_score=round(score, 8),
            shared_loci=shared,
        )

    def compare_profiles_masked(
        self,
        profile_a: VectorizedProfile,
        profile_b: VectorizedProfile,
    ) -> MaskedSimilarityResult:
        """
        Completeness-penalized profile comparison for partial DNA evidence.

        Uses the Masked Similarity formula:
            PenalizedScore = S(active_dims) × (N_active / N_total)

        This method is preferred over compare_profiles when dealing with
        degraded or partial profiles, as it:
            1. Only compares dimensions where BOTH profiles have data.
            2. Applies a completeness penalty to prevent 5-locus "perfect"
               matches from outscoring 20-locus strong matches.

        Args:
            profile_a: Query profile (may be partial).
            profile_b: Target profile from the database.

        Returns:
            MaskedSimilarityResult with raw and penalized scores.
        """
        raw_sim, active, missing_a, missing_b = self.masked_cosine_similarity(
            profile_a.embedding, profile_b.embedding
        )

        total = len(STANDARD_LOCI)
        completeness = active / total if total > 0 else 0.0
        penalized = raw_sim * completeness

        # Classify quality tier
        if active >= 18:
            tier = "complete"
        elif active >= 10:
            tier = "partial"
        else:
            tier = "degraded"

        return MaskedSimilarityResult(
            query_profile_id=profile_a.profile_id,
            target_profile_id=profile_b.profile_id,
            raw_similarity=round(raw_sim, 8),
            penalized_score=round(penalized, 8),
            active_loci=active,
            total_loci=total,
            completeness_ratio=round(completeness, 4),
            missing_loci_query=missing_a,
            missing_loci_target=missing_b,
            quality_tier=tier,
        )

    @staticmethod
    def batch_cosine_similarity(
        query: List[float],
        matrix: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Compute cosine similarity between a query vector and a matrix of vectors.

        Optimized for batch operations using NumPy broadcasting. Used internally
        by search pipelines when Milvus is unavailable or for in-memory ranking.

        Args:
            query: 48-dimensional query vector.
            matrix: (N, 48) matrix of candidate vectors.

        Returns:
            (N,) array of similarity scores in [0, 1].
        """
        q: NDArray[np.float64] = np.asarray(query, dtype=np.float64)
        m: NDArray[np.float64] = np.asarray(matrix, dtype=np.float64)

        if m.ndim == 1:
            m = m.reshape(1, -1)

        # Dot products: (N,)
        dots: NDArray[np.float64] = m @ q

        # Norms
        q_norm: float = float(np.linalg.norm(q))
        m_norms: NDArray[np.float64] = np.linalg.norm(m, axis=1)

        # Guard against zero norms
        denominator: NDArray[np.float64] = q_norm * m_norms
        denominator = np.where(denominator == 0.0, 1.0, denominator)

        scores: NDArray[np.float64] = np.clip(dots / denominator, 0.0, 1.0)
        return scores
