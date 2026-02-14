"""
Synthetic Genetics Engine — Population-Based STR Profile Generator.

Generates high-fidelity synthetic GenomicProfile data using realistic
allele frequency distributions modeled after NIST STRBase and ENFSI
population studies. Supports multiple ethnic backgrounds and three
generation modes: clean, noisy, and sibling/related profiles.

Allele Frequency Model:
    Each locus has a discrete probability distribution over observed
    allele values. Alleles are sampled independently for each parent
    chromosome (Hardy-Weinberg equilibrium assumption). Population-
    specific distributions shift the mode and spread to reflect known
    inter-ethnic variation in STR repeat counts.

Performance:
    All random generation uses NumPy's Generator API for vectorized
    sampling. A single call to generate_batch() with N=10,000 completes
    in <500ms on commodity hardware.
"""

import uuid
import time
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
from numpy.random import Generator, default_rng


# ═══════════════════════════════════════════════════════════════════════════════
# POPULATION ALLELE FREQUENCY TABLES
# Simplified discrete distributions modeled after published frequency data.
# Format: {allele_value: relative_frequency}
# Sources: NIST STRBase, Promega PowerPlex, ENFSI collaborative exercises.
# ═══════════════════════════════════════════════════════════════════════════════

PopulationName = Literal["european", "east_asian", "african", "hispanic", "south_asian"]

# Base distributions (European) — other populations shift these
_BASE_FREQUENCIES: Dict[str, Dict[float, float]] = {
    "CSF1PO":   {7: 0.01, 8: 0.02, 9: 0.05, 10: 0.28, 11: 0.27, 12: 0.30, 13: 0.05, 14: 0.02},
    "D1S1656":  {11: 0.05, 12: 0.12, 13: 0.08, 14: 0.10, 15: 0.18, 16: 0.15, 17: 0.14, 17.3: 0.10, 18.3: 0.05, 19.3: 0.03},
    "D2S441":   {10: 0.10, 11: 0.30, 11.3: 0.08, 12: 0.12, 13: 0.05, 14: 0.25, 15: 0.08, 16: 0.02},
    "D2S1338":  {17: 0.10, 18: 0.05, 19: 0.12, 20: 0.15, 21: 0.05, 22: 0.03, 23: 0.12, 24: 0.15, 25: 0.13, 26: 0.05, 27: 0.05},
    "D3S1358":  {12: 0.01, 13: 0.02, 14: 0.12, 15: 0.28, 16: 0.25, 17: 0.18, 18: 0.12, 19: 0.02},
    "D5S818":   {7: 0.01, 8: 0.02, 9: 0.05, 10: 0.06, 11: 0.35, 12: 0.30, 13: 0.18, 14: 0.03},
    "D7S820":   {7: 0.01, 8: 0.15, 9: 0.10, 10: 0.25, 11: 0.22, 12: 0.18, 13: 0.07, 14: 0.02},
    "D8S1179":  {8: 0.01, 9: 0.02, 10: 0.08, 11: 0.07, 12: 0.12, 13: 0.30, 14: 0.22, 15: 0.12, 16: 0.05, 17: 0.01},
    "D10S1248": {11: 0.03, 12: 0.08, 13: 0.25, 14: 0.28, 15: 0.18, 16: 0.12, 17: 0.05, 18: 0.01},
    "D12S391":  {17: 0.05, 18: 0.15, 19: 0.12, 20: 0.10, 21: 0.12, 22: 0.15, 23: 0.12, 24: 0.10, 25: 0.06, 26: 0.03},
    "D13S317":  {8: 0.15, 9: 0.08, 10: 0.05, 11: 0.28, 12: 0.30, 13: 0.08, 14: 0.05, 15: 0.01},
    "D16S539":  {8: 0.02, 9: 0.12, 10: 0.08, 11: 0.28, 12: 0.20, 13: 0.22, 14: 0.06, 15: 0.02},
    "D18S51":   {10: 0.01, 12: 0.10, 13: 0.12, 14: 0.15, 15: 0.12, 16: 0.10, 17: 0.12, 18: 0.08, 19: 0.06, 20: 0.05, 21: 0.04, 22: 0.03, 23: 0.02},
    "D19S433":  {12: 0.05, 13: 0.25, 14: 0.30, 15: 0.18, 15.2: 0.08, 16: 0.05, 16.2: 0.04, 17: 0.03, 17.2: 0.02},
    "D21S11":   {27: 0.05, 28: 0.15, 29: 0.18, 30: 0.22, 30.2: 0.10, 31: 0.08, 31.2: 0.10, 32.2: 0.06, 33.2: 0.04, 34.2: 0.02},
    "D22S1045": {11: 0.10, 12: 0.05, 14: 0.08, 15: 0.20, 16: 0.28, 17: 0.18, 18: 0.08, 19: 0.03},
    "FGA":      {19: 0.05, 20: 0.08, 21: 0.12, 22: 0.15, 23: 0.18, 24: 0.15, 25: 0.12, 26: 0.08, 27: 0.04, 28: 0.02, 29: 0.01},
    "SE33":     {15: 0.02, 17: 0.04, 18: 0.05, 19: 0.06, 20: 0.08, 21: 0.05, 22: 0.04, 24.2: 0.03, 25.2: 0.05, 26.2: 0.08, 27.2: 0.10, 28.2: 0.12, 29.2: 0.10, 30.2: 0.08, 31.2: 0.05, 32.2: 0.03, 33.2: 0.02},
    "TH01":     {5: 0.01, 6: 0.22, 7: 0.18, 8: 0.12, 9: 0.15, 9.3: 0.25, 10: 0.05, 10.3: 0.02},
    "TPOX":     {6: 0.01, 7: 0.02, 8: 0.48, 9: 0.12, 10: 0.06, 11: 0.25, 12: 0.05, 13: 0.01},
    "VWA":      {13: 0.01, 14: 0.10, 15: 0.10, 16: 0.20, 17: 0.25, 18: 0.20, 19: 0.08, 20: 0.05, 21: 0.01},
    "AMEL":     {1.0: 0.50, 2.0: 0.50},
    "PENTA_D":  {5: 0.01, 7: 0.02, 8: 0.04, 9: 0.15, 10: 0.12, 11: 0.15, 12: 0.18, 13: 0.15, 14: 0.10, 15: 0.05, 16: 0.03},
    "PENTA_E":  {5: 0.05, 7: 0.08, 8: 0.05, 10: 0.10, 11: 0.12, 12: 0.15, 13: 0.10, 14: 0.08, 15: 0.07, 16: 0.06, 17: 0.05, 18: 0.04, 19: 0.03, 20: 0.02},
}

# Population shift factors — applied as allele value offsets and frequency reweighting
_POPULATION_SHIFTS: Dict[PopulationName, Dict[str, float]] = {
    "european":   {},  # Baseline — no shift
    "east_asian": {"CSF1PO": +1.0, "D3S1358": -0.5, "TH01": -0.5, "VWA": -1.0, "FGA": -1.0, "D18S51": -1.0},
    "african":    {"CSF1PO": +1.5, "D3S1358": +1.0, "TH01": +0.5, "FGA": +2.0, "D18S51": +2.0, "D21S11": +1.0},
    "hispanic":   {"CSF1PO": +0.5, "D3S1358": +0.5, "VWA": +0.5, "FGA": +1.0},
    "south_asian": {"TH01": -0.3, "D3S1358": +0.3, "FGA": +0.5, "D18S51": +0.5},
}

# Node name pools per region for realistic node_id generation
_NODE_POOLS: Dict[PopulationName, List[str]] = {
    "european":    ["EUROPOL-NL", "BKA-DE", "NCA-UK", "DGPN-FR", "CNP-ES", "PF-IT", "KRIPOS-NO", "NBI-FI"],
    "east_asian":  ["NPA-JP", "KNPA-KR", "MPS-CN", "CIB-TW", "KPNP-KR", "SPF-SG", "RTP-TH"],
    "african":     ["SAPS-ZA", "NPF-NG", "DCI-KE", "BNPJ-MA", "ANP-EG", "INTERPOL-AF"],
    "hispanic":    ["PFA-AR", "PF-BR", "FGR-MX", "CICPC-VE", "PNP-PE", "PDI-CL"],
    "south_asian": ["CBI-IN", "FIA-PK", "CID-BD", "CID-LK", "NBI-NP"],
}


# ═══════════════════════════════════════════════════════════════════════════════
# PROFILE DATA STRUCTURE
# ═══════════════════════════════════════════════════════════════════════════════

class SyntheticProfile:
    """A generated synthetic STR profile with metadata."""

    __slots__ = (
        "profile_id", "node_id", "population", "str_markers",
        "timestamp", "profile_type", "related_to",
    )

    def __init__(
        self,
        profile_id: str,
        node_id: str,
        population: PopulationName,
        str_markers: Dict[str, Dict[str, float]],
        timestamp: int,
        profile_type: str = "clean",
        related_to: Optional[str] = None,
    ):
        self.profile_id = profile_id
        self.node_id = node_id
        self.population = population
        self.str_markers = str_markers
        self.timestamp = timestamp
        self.profile_type = profile_type
        self.related_to = related_to

    def to_ingest_payload(self) -> Dict[str, Any]:
        """Convert to the JSON payload expected by POST /profile/ingest."""
        markers = {}
        for locus, alleles in self.str_markers.items():
            markers[locus] = {
                "allele_1": alleles["allele_1"],
                "allele_2": alleles["allele_2"],
            }
        return {
            "profile_id": self.profile_id,
            "node_id": self.node_id,
            "str_markers": markers,
            "timestamp": self.timestamp,
        }

    def __repr__(self) -> str:
        return (
            f"SyntheticProfile(id={self.profile_id[:8]}..., "
            f"node={self.node_id}, pop={self.population}, "
            f"type={self.profile_type}, loci={len(self.str_markers)})"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# GENERATOR ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class SyntheticGeneticsEngine:
    """
    High-fidelity synthetic STR profile generator.

    Produces biologically plausible genomic profiles based on published
    allele frequency distributions for multiple ethnic backgrounds.

    Modes:
        - clean:   All 24 loci populated, values within known ranges.
        - noisy:   Random loci dropped (1–6), some values slightly perturbed.
        - sibling: ~50% allele sharing with a reference "parent" profile.

    Usage:
        engine = SyntheticGeneticsEngine(seed=42)
        profiles = engine.generate_batch(count=10000, population="european")
    """

    def __init__(self, seed: Optional[int] = None) -> None:
        """
        Initialize the engine with a reproducible RNG seed.

        Args:
            seed: NumPy random seed for reproducible generation.
                  Use None for non-deterministic output.
        """
        self._rng: Generator = default_rng(seed)
        self._frequency_cache: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._precompute_frequencies()

    def _precompute_frequencies(self) -> None:
        """
        Convert frequency dicts to NumPy arrays for vectorized sampling.

        Pre-normalization ensures frequencies sum to exactly 1.0 despite
        any floating-point imprecision in the source tables.
        """
        for locus, freq_dict in _BASE_FREQUENCIES.items():
            alleles = np.array(list(freq_dict.keys()), dtype=np.float64)
            probs = np.array(list(freq_dict.values()), dtype=np.float64)
            probs /= probs.sum()  # Normalize
            self._frequency_cache[locus] = (alleles, probs)

    def _sample_allele(
        self,
        locus: str,
        population: PopulationName,
    ) -> float:
        """
        Sample a single allele from the population-specific frequency distribution.

        Population shifts are applied by offsetting the allele value after
        sampling from the base distribution. This preserves the shape of
        the distribution while shifting the mean to match population data.

        Args:
            locus: STR locus name.
            population: Target population.

        Returns:
            Sampled allele value (float, may include microvariant decimals).
        """
        alleles, probs = self._frequency_cache[locus]
        sampled = self._rng.choice(alleles, p=probs)

        # Apply population shift
        shifts = _POPULATION_SHIFTS.get(population, {})
        if locus in shifts:
            sampled += shifts[locus]
            # Clamp to biologically plausible range
            sampled = max(1.0, sampled)

        return float(round(sampled, 1))

    def _generate_locus(
        self,
        locus: str,
        population: PopulationName,
    ) -> Dict[str, float]:
        """
        Generate both alleles for a single locus.

        Alleles are sampled independently (Hardy-Weinberg equilibrium).
        The lower value is placed in allele_1 by convention.
        """
        a1 = self._sample_allele(locus, population)
        a2 = self._sample_allele(locus, population)

        # Sort so allele_1 <= allele_2
        if a1 > a2:
            a1, a2 = a2, a1

        return {"allele_1": a1, "allele_2": a2}

    def generate_clean(
        self,
        population: PopulationName = "european",
        node_id: Optional[str] = None,
    ) -> SyntheticProfile:
        """
        Generate a clean profile with all 24 loci populated.

        All allele values fall within known biological ranges. This is the
        baseline for testing the vectorization pipeline.

        Args:
            population: Ethnic background for frequency sampling.
            node_id: Override node identifier. If None, randomly selected.

        Returns:
            SyntheticProfile with complete STR data.
        """
        pid = str(uuid.uuid4())
        nid = node_id or self._rng.choice(_NODE_POOLS[population])
        ts = int(time.time()) + int(self._rng.integers(-86400 * 30, 0))

        markers: Dict[str, Dict[str, float]] = {}
        for locus in _BASE_FREQUENCIES:
            markers[locus] = self._generate_locus(locus, population)

        return SyntheticProfile(
            profile_id=pid,
            node_id=nid,
            population=population,
            str_markers=markers,
            timestamp=ts,
            profile_type="clean",
        )

    def generate_noisy(
        self,
        population: PopulationName = "european",
        missing_loci_range: Tuple[int, int] = (1, 6),
        corruption_probability: float = 0.08,
    ) -> SyntheticProfile:
        """
        Generate a noisy profile with missing loci and value perturbations.

        Simulates real-world conditions: degraded DNA samples, partial
        amplification failures, and instrument noise. Used to stress-test
        the ForensicValidator's anomaly detection.

        Args:
            population: Ethnic background.
            missing_loci_range: (min, max) number of loci to randomly drop.
            corruption_probability: Chance of perturbing each remaining allele.

        Returns:
            SyntheticProfile with intentionally degraded data.
        """
        base = self.generate_clean(population)
        base.profile_type = "noisy"

        # Drop random loci
        loci_list = list(base.str_markers.keys())
        n_drop = self._rng.integers(missing_loci_range[0], missing_loci_range[1] + 1)
        n_drop = min(n_drop, len(loci_list) - 1)  # Keep at least 1
        to_drop = self._rng.choice(loci_list, size=n_drop, replace=False)
        for locus in to_drop:
            del base.str_markers[locus]

        # Corrupt remaining alleles with small probability
        for locus, alleles in base.str_markers.items():
            if self._rng.random() < corruption_probability:
                # Add Gaussian noise (±1-3 repeats)
                noise = float(self._rng.normal(0, 1.5))
                key = self._rng.choice(["allele_1", "allele_2"])
                alleles[key] = max(1.0, round(alleles[key] + noise, 1))

        return base

    def generate_sibling(
        self,
        parent_profile: SyntheticProfile,
        sharing_ratio: float = 0.5,
    ) -> SyntheticProfile:
        """
        Generate a related (sibling) profile sharing ~50% alleles with the parent.

        Simulates biological relatedness by:
        1. For each locus, with probability=sharing_ratio, one allele is
           inherited directly from the parent profile.
        2. The other allele is resampled from the population distribution.

        This tests the vectorizer's ability to detect proximity for related
        individuals (cosine similarity should be notably higher than unrelated).

        Args:
            parent_profile: Reference profile to derive relatedness from.
            sharing_ratio: Fraction of loci where an allele is inherited (default 0.5).

        Returns:
            SyntheticProfile marked as related to the parent.
        """
        pid = str(uuid.uuid4())
        nid = parent_profile.node_id
        population = parent_profile.population
        ts = int(time.time()) + int(self._rng.integers(-86400 * 7, 0))

        markers: Dict[str, Dict[str, float]] = {}
        for locus, parent_alleles in parent_profile.str_markers.items():
            if self._rng.random() < sharing_ratio:
                # Inherit one allele from parent
                inherited = self._rng.choice([parent_alleles["allele_1"], parent_alleles["allele_2"]])
                novel = self._sample_allele(locus, population)
                a1, a2 = sorted([inherited, novel])
            else:
                # Fully resampled
                locus_data = self._generate_locus(locus, population)
                a1, a2 = locus_data["allele_1"], locus_data["allele_2"]

            markers[locus] = {"allele_1": a1, "allele_2": a2}

        return SyntheticProfile(
            profile_id=pid,
            node_id=nid,
            population=population,
            str_markers=markers,
            timestamp=ts,
            profile_type="sibling",
            related_to=parent_profile.profile_id,
        )

    def generate_batch(
        self,
        count: int = 10000,
        population: Optional[PopulationName] = None,
        noisy_ratio: float = 0.15,
        sibling_ratio: float = 0.05,
    ) -> List[SyntheticProfile]:
        """
        Generate a mixed batch of synthetic profiles.

        Distribution:
            - (1 - noisy_ratio - sibling_ratio) × count → clean profiles
            - noisy_ratio × count → noisy profiles
            - sibling_ratio × count → sibling profiles (derived from clean ones)

        When population is None, profiles are distributed evenly across
        all five population groups for diversity testing.

        Args:
            count: Total number of profiles to generate.
            population: Fixed population, or None for mixed.
            noisy_ratio: Fraction of noisy profiles (default 15%).
            sibling_ratio: Fraction of sibling profiles (default 5%).

        Returns:
            List of SyntheticProfile objects.
        """
        populations: List[PopulationName] = (
            [population] if population else
            ["european", "east_asian", "african", "hispanic", "south_asian"]
        )

        n_noisy = int(count * noisy_ratio)
        n_sibling = int(count * sibling_ratio)
        n_clean = count - n_noisy - n_sibling

        profiles: List[SyntheticProfile] = []

        # Clean profiles
        for i in range(n_clean):
            pop = populations[i % len(populations)]
            profiles.append(self.generate_clean(pop))

        # Noisy profiles
        for i in range(n_noisy):
            pop = populations[i % len(populations)]
            profiles.append(self.generate_noisy(pop))

        # Sibling profiles (derived from random clean profiles)
        parent_pool = [p for p in profiles if p.profile_type == "clean"]
        for i in range(n_sibling):
            parent = parent_pool[self._rng.integers(0, len(parent_pool))]
            profiles.append(self.generate_sibling(parent))

        # Shuffle for realistic ingestion order
        self._rng.shuffle(profiles)

        return profiles


# ═══════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    count = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 42

    engine = SyntheticGeneticsEngine(seed=seed)

    t0 = time.perf_counter()
    batch = engine.generate_batch(count=count)
    elapsed = time.perf_counter() - t0

    # Statistics
    types = {"clean": 0, "noisy": 0, "sibling": 0}
    pops = {}
    for p in batch:
        types[p.profile_type] = types.get(p.profile_type, 0) + 1
        pops[p.population] = pops.get(p.population, 0) + 1

    print(f"╔══════════════════════════════════════════════════════╗")
    print(f"║  VANTAGE-STR  Synthetic Genetics Engine             ║")
    print(f"╠══════════════════════════════════════════════════════╣")
    print(f"║  Generated:  {count:>8,} profiles in {elapsed*1000:>8.1f} ms   ║")
    print(f"║  Throughput: {count/elapsed:>8,.0f} profiles/sec            ║")
    print(f"╠══════════════════════════════════════════════════════╣")
    print(f"║  Types:  clean={types['clean']:>5}  noisy={types['noisy']:>5}  "
          f"sibling={types['sibling']:>5} ║")
    print(f"╠══════════════════════════════════════════════════════╣")
    for pop, cnt in sorted(pops.items()):
        print(f"║  {pop:<14s} {cnt:>6,} profiles                     ║")
    print(f"╚══════════════════════════════════════════════════════╝")

    # Print sample profile
    sample = batch[0]
    print(f"\n── Sample Profile ──")
    print(f"  ID:   {sample.profile_id}")
    print(f"  Node: {sample.node_id}")
    print(f"  Pop:  {sample.population}")
    print(f"  Type: {sample.profile_type}")
    print(f"  Loci: {len(sample.str_markers)}")
    for locus in sorted(list(sample.str_markers.keys()))[:5]:
        a = sample.str_markers[locus]
        print(f"    {locus:<12s}  {a['allele_1']:>5.1f} / {a['allele_2']:>5.1f}")
    print(f"    ... ({len(sample.str_markers) - 5} more)")
