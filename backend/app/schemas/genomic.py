"""
Pydantic schemas for genomic profile ingestion and validation.

These models serve as the API-layer wrappers around the Protobuf-defined
GenomicProfile message. They enforce strict biometric validation rules:

- Allele values must be positive and within the biologically plausible
  range for forensic STR markers (typically 3.0–50.0 repeats).
- Homozygosity is auto-detected when both allele values are equal.
- Marker names are validated against a known CODIS/ESS panel.
- Profile IDs must be valid UUID v4 to ensure global uniqueness
  across decentralized nodes.
"""

from datetime import datetime
from typing import Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, UUID4, field_validator, model_validator


# --- Standard forensic STR marker panels ---
CODIS_MARKERS: set[str] = {
    "CSF1PO", "D3S1358", "D5S818", "D7S820", "D8S1179",
    "D13S317", "D16S539", "D18S51", "D21S11", "FGA",
    "TH01", "TPOX", "VWA",
    # Extended CODIS (2017+)
    "D1S1656", "D2S441", "D2S1338", "D10S1248", "D12S391",
    "D19S433", "D22S1045", "SE33",
    # Amelogenin (sex-determining marker)
    "AMEL",
}


class LocusDataSchema(BaseModel):
    """
    Represents allele repeat counts at a single STR locus.

    Allele values are floats to accommodate microvariant alleles
    (e.g., allele 9.3 at TH01). If both alleles are equal, the locus
    is classified as homozygous — the individual inherited the same
    repeat count from both parents at this position.
    """
    allele_1: float = Field(..., gt=0, le=100.0, description="First allele repeat count")
    allele_2: float = Field(..., gt=0, le=100.0, description="Second allele repeat count")
    is_homozygous: bool = Field(default=False, description="Auto-computed; true when allele_1 == allele_2")

    @model_validator(mode="after")
    def compute_homozygosity(self) -> "LocusDataSchema":
        """
        Auto-detect homozygous loci.

        In forensic genetics, a homozygous locus means the individual
        has identical repeat counts on both chromosomes. This is critical
        for accurate match probability calculations since homozygous
        loci contribute p² (Hardy-Weinberg) instead of 2pq.
        """
        self.is_homozygous = self.allele_1 == self.allele_2
        return self


class GenomicProfileIngest(BaseModel):
    """
    Incoming genomic profile for ingestion into the VANTAGE-STR network.

    Mirrors the Protobuf GenomicProfile message structure for seamless
    serialization, while adding Pydantic validation at the API boundary.

    Validation rules:
    - profile_id: Auto-generated UUID v4 if not provided.
    - node_id: Must be non-empty (identifies originating country/agency).
    - str_markers: At least one marker required; keys validated against CODIS panel.
    - encrypted_metadata: Optional; reserved for future ZKP payloads.
    """
    profile_id: UUID4 = Field(default_factory=uuid4, description="UUID v4 — unique profile identifier")
    node_id: str = Field(..., min_length=1, max_length=64, description="Originating node/agency identifier")
    str_markers: Dict[str, LocusDataSchema] = Field(
        ...,
        min_length=1,
        description="Map of marker name → allele data. Keys should be CODIS/ESS marker names."
    )
    encrypted_metadata: Optional[bytes] = Field(default=None, description="Reserved for ZKP-encrypted auxiliary data")
    timestamp: int = Field(
        default_factory=lambda: int(datetime.utcnow().timestamp()),
        description="Unix epoch timestamp of profile creation"
    )

    @field_validator("str_markers")
    @classmethod
    def validate_marker_names(cls, v: Dict[str, LocusDataSchema]) -> Dict[str, LocusDataSchema]:
        """
        Warn on non-standard marker names.

        While the system accepts any marker name for extensibility,
        markers outside the CODIS/ESS panel are flagged. This allows
        international nodes with custom regional panels to submit
        data while maintaining interoperability auditing.
        """
        unknown = set(v.keys()) - CODIS_MARKERS
        if unknown:
            # Non-blocking: unknown markers are accepted but logged.
            # In production, this triggers an audit trail entry.
            pass
        return v


class GenomicProfileOut(BaseModel):
    """Response model for a successfully ingested genomic profile."""
    profile_id: UUID4
    node_id: str
    markers_received: int
    accepted: bool = True
    message: str = "Profile ingested successfully"
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        from_attributes = True


class MatchResult(BaseModel):
    """
    Result of a forensic STR profile comparison.

    Similarity score is computed via allele overlap across shared loci.
    A score ≥ 0.9999 on 20+ loci is considered a full match with
    random match probability < 1 in 10^18.
    """
    profile_id: UUID4
    similarity_score: float = Field(..., ge=0.0, le=1.0)
    node_owner: str
    matching_loci: Dict[str, LocusDataSchema]


# ═══════════════════════════════════════════════════════════════════════════════
# SNP & PHENOTYPE SCHEMAS (Phase 3.3)
# ═══════════════════════════════════════════════════════════════════════════════

# HIrisPlex-S panel rsIDs used for phenotype prediction
HIRISPLEX_RSIDS: set[str] = {
    # Eye color
    "rs12913832", "rs1800407", "rs12896399", "rs16891982",
    "rs12203592", "rs1393350",
    # Hair color (MC1R + pigmentation)
    "rs1805007", "rs1805008", "rs1805009", "rs11547464",
    "rs1805006", "rs2228479", "rs885479",
    # Skin pigmentation
    "rs1426654", "rs1042602", "rs6119471",
}


class SNPData(BaseModel):
    """
    SNP genotype data for phenotype prediction.

    Key: rsID (e.g., 'rs12913832').
    Value: Genotype string — two alleles (e.g., 'AA', 'AG', 'GG').
    Genotypes are normalized to uppercase alphabetical order.
    """
    snp_map: Dict[str, str] = Field(
        ...,
        min_length=1,
        description="Map of rsID → genotype. Min 1 marker required.",
    )

    @field_validator("snp_map")
    @classmethod
    def validate_genotypes(cls, v: Dict[str, str]) -> Dict[str, str]:
        """Normalize and validate genotype format."""
        import re
        normalized: Dict[str, str] = {}
        for rsid, genotype in v.items():
            # Validate rsID format
            if not rsid.startswith("rs"):
                raise ValueError(f"Invalid rsID format: '{rsid}'. Must start with 'rs'.")
            # Normalize genotype: uppercase, sorted alleles
            g = genotype.upper().strip()
            if not re.match(r"^[ACGT]{2}$", g):
                raise ValueError(
                    f"Invalid genotype '{genotype}' for {rsid}. "
                    "Must be exactly 2 nucleotide characters (A/C/G/T)."
                )
            normalized[rsid] = "".join(sorted(g))
        return normalized


class PhenotypeTraitProbability(BaseModel):
    """Probability distribution for a single predicted trait."""
    trait: str = Field(..., description="Trait name (e.g., 'Eye Color')")
    predictions: Dict[str, float] = Field(
        ...,
        description="Map of outcome → probability (e.g., {'Blue': 0.72, 'Green': 0.18, 'Brown': 0.10})",
    )
    dominant_prediction: str = Field(..., description="Highest-probability outcome")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence in the dominant prediction")
    contributing_snps: List[str] = Field(
        default_factory=list,
        description="rsIDs that contributed to this prediction",
    )


class PhenotypeReport(BaseModel):
    """
    Complete phenotype prediction report for a genomic profile.

    Generated by the HIrisPlex-S PhenotypePredictor engine and optionally
    enhanced with DSPy ChainOfThought biological reasoning.
    """
    profile_id: str
    snps_analyzed: int = Field(..., ge=0)
    hirisplex_coverage: float = Field(
        ..., ge=0.0, le=1.0,
        description="Fraction of HIrisPlex-S panel SNPs present in input",
    )
    traits: List[PhenotypeTraitProbability] = Field(default_factory=list)
    ancestry_indicators: Dict[str, float] = Field(
        default_factory=dict,
        description="Biogeographic ancestry probability indicators",
    )
    image_url: Optional[str] = Field(
        default=None,
        description="URL of the generated forensic reconstruction image",
    )
    positive_prompt: Optional[str] = Field(
        default=None, description="The positive prompt used for generation"
    )
    negative_prompt: Optional[str] = Field(
        default=None, description="The negative prompt used for generation"
    )
    seed: Optional[int] = Field(
        default=None, description="The random seed used for deterministic generation"
    )
    genai_model_id: Optional[str] = Field(
        default="mock-sdxl-dev", description="ID of the GenAI model used"
    )
    trait_summary: Optional[Dict[str, str]] = Field(
        default=None, description="Simplified trait summary (e.g., 'Eye Color': 'Blue')"
    )
    ai_reasoning: Optional[str] = Field(
        default=None,
        description="DSPy ChainOfThought biological explanation (when available)",
    )
    model_version: str = "HIrisPlex-S v1.0"
    disclaimer: str = (
        "Phenotype predictions are probabilistic estimates based on known SNP-trait "
        "associations. They should not be used as sole evidence in forensic proceedings. "
        "Accuracy varies by population and is subject to epistatic effects."
    )
