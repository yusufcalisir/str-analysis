"""
VANTAGE-STR — Core API Entry Point.

Vertical Anonymous Network for Tactical Analysis of Genomic Evidence.
This module initializes the FastAPI application and registers the
primary operational endpoints for health monitoring and profile ingestion.

Ingestion Pipeline:
    1. Pydantic validation (schema boundary)
    2. ForensicValidator — rule-based + DSPy ChainOfThought assessment
    3. If validity_score >= 0.85 → ACCEPTED → vectorize + store
    4. If validity_score < 0.85  → QUARANTINED → flagged for review
"""

import logging
import time
from datetime import datetime, timezone
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.core.config import settings
from app.schemas.genomic import GenomicProfileIngest, GenomicProfileOut, PhenotypeReport

logger = logging.getLogger(__name__)

# --- Application boot timestamp for uptime tracking ---
_BOOT_TIME: float = time.time()

app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Decentralized forensic STR matching network API",
    version="0.1.0",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
)

# --- CORS Configuration ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://vantage-str.vercel.app",
        "https://str-analysis.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════════════
# RESPONSE MODELS
# ═══════════════════════════════════════════════════════════════════════════════

class IngestResponse(BaseModel):
    """Extended ingestion response including validation details."""
    profile_id: str
    node_id: str
    markers_received: int
    accepted: bool
    decision: str  # ACCEPTED, QUARANTINED, REJECTED
    validity_score: float = Field(..., ge=0.0, le=1.0)
    anomaly_report: str = "No anomalies detected."
    is_poisoned: bool = False
    message: str


class QuarantineAlert(BaseModel):
    """Alert payload emitted when a profile is quarantined."""
    profile_id: str
    node_id: str
    validity_score: float
    anomaly_report: str
    is_poisoned: bool
    quarantined_at: str


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATOR INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

# Lazy-loaded validator to avoid import failures when DSPy is not configured
_validator = None


def _get_validator():
    """
    Lazy-initialize the ForensicValidator.

    Deferred to avoid hard failure at import time if DSPy or the LLM
    backend is not configured. Falls back to rule-only validation.
    """
    global _validator
    if _validator is None:
        try:
            from app.agents.forensic_validator import ForensicValidator
            _validator = ForensicValidator()
            logger.info("[MAIN] ForensicValidator initialized")
        except Exception as exc:
            logger.warning(f"[MAIN] ForensicValidator unavailable: {exc}")
    return _validator


# Lazy-loaded phenotype analyst (Phase 3.3)
_phenotype_analyst = None


def _get_phenotype_analyst():
    """
    Lazy-initialize the PhenotypeAnalyst.

    Deferred to avoid hard failure at import time if DSPy or the LLM
    backend is not configured. Falls back to engine-only predictions.
    """
    global _phenotype_analyst
    if _phenotype_analyst is None:
        try:
            from app.agents.phenotype_agent import PhenotypeAnalyst
            _phenotype_analyst = PhenotypeAnalyst()
            logger.info("[MAIN] PhenotypeAnalyst initialized")
        except Exception as exc:
            logger.warning(f"[MAIN] PhenotypeAnalyst unavailable: {exc}")
    return _phenotype_analyst


# ═══════════════════════════════════════════════════════════════════════════════
# MOCK SNP PROFILE STORE (Phase 3.3)
# In production, SNP data would be retrieved from PostgreSQL/Milvus.
# ═══════════════════════════════════════════════════════════════════════════════

_SNP_STORE: Dict[str, Dict[str, str]] = {
    # Pre-populated test profile: European phenotype — Blue eyes, Blond hair, Light skin
    "test-profile-eu": {
        "rs12913832": "GG",   # HERC2 — strong blue eye predictor
        "rs16891982": "GG",   # SLC45A2 — light pigmentation
        "rs1800407":  "GG",   # OCA2
        "rs12896399": "GT",   # SLC24A4 — blond association
        "rs12203592": "CC",   # IRF4
        "rs1393350":  "GA",   # TYR
        "rs1805007":  "CC",   # MC1R — no red variant
        "rs1805008":  "CC",   # MC1R — no red variant
        "rs1805009":  "GG",   # MC1R — no red variant
        "rs11547464": "GG",   # MC1R — no red variant
        "rs1805006":  "CC",   # MC1R — no red variant
        "rs1426654":  "AA",   # SLC24A5 — light skin (European)
        "rs1042602":  "CA",   # TYR — heterozygous
        "rs6119471":  "GG",   # ASIP
    },
    # Test profile: African phenotype — Brown eyes, Black hair, Dark skin
    "test-profile-af": {
        "rs12913832": "AA",   # HERC2 — brown eye predictor
        "rs16891982": "CC",   # SLC45A2 — dark pigmentation
        "rs1800407":  "GG",   # OCA2
        "rs12896399": "GG",   # SLC24A4
        "rs1805007":  "CC",   # MC1R — no red variant
        "rs1805008":  "CC",   # MC1R — no red variant
        "rs1426654":  "GG",   # SLC24A5 — dark skin (African)
        "rs1042602":  "CC",   # TYR — no light variant
        "rs6119471":  "GG",   # ASIP
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# STR PROFILE STORE (Phase 3.5) — For LR Computation
# In production, these come from the ingest pipeline + Milvus.
# ═══════════════════════════════════════════════════════════════════════════════

_STR_STORE: Dict[str, Dict[str, tuple]] = {
    # Test profile — European individual with realistic CODIS-20 genotypes
    "test-profile-eu": {
        "CSF1PO": (10, 12), "D1S1656": (15, 17.3), "D2S441": (11, 14),
        "D2S1338": (17, 23), "D3S1358": (15, 16), "D5S818": (11, 12),
        "D7S820": (10, 11), "D8S1179": (13, 14), "D10S1248": (13, 15),
        "D12S391": (18, 21), "D13S317": (11, 12), "D16S539": (11, 12),
        "D18S51": (14, 16), "D19S433": (13, 14), "D21S11": (29, 30),
        "D22S1045": (15, 16), "FGA": (21, 24), "SE33": (19, 27.2),
        "TH01": (6, 9.3), "TPOX": (8, 11), "VWA": (16, 17),
        "PENTA_D": (9, 13), "PENTA_E": (12, 17), "AMEL": (0, 1),
    },
    # Test profile — African individual
    "test-profile-af": {
        "CSF1PO": (10, 13), "D1S1656": (16, 18.3), "D2S441": (11, 11),
        "D2S1338": (19, 20), "D3S1358": (15, 18), "D5S818": (12, 13),
        "D7S820": (10, 12), "D8S1179": (12, 14), "D10S1248": (14, 15),
        "D12S391": (19, 22), "D13S317": (12, 14), "D16S539": (9, 12),
        "D18S51": (15, 19), "D19S433": (14, 15), "D21S11": (30, 32.2),
        "D22S1045": (16, 17), "FGA": (23, 25), "SE33": (20, 28.2),
        "TH01": (7, 9.3), "TPOX": (8, 11), "VWA": (16, 18),
        "PENTA_D": (10, 14), "PENTA_E": (11, 15), "AMEL": (0, 1),
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/")
def root() -> dict:
    """Root endpoint — confirms the API process is alive."""
    return {"message": "VANTAGE-STR API is operational", "status": "tactical_online"}


@app.get("/health")
def health_check() -> dict:
    """
    Health check endpoint for orchestration and monitoring.

    Returns the current node status, uptime in seconds, API version,
    and a UTC timestamp. Used by Docker health checks, load balancers,
    and the frontend Global Network Status indicator to determine
    whether this node is responsive and operational.
    """
    return {
        "status": "operational",
        "node_id": "PRIMARY",
        "uptime_seconds": round(time.time() - _BOOT_TIME, 2),
        "version": "0.1.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "validator_active": _validator is not None,
    }


@app.post(
    "/profile/ingest",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
)
def ingest_profile(profile: GenomicProfileIngest) -> IngestResponse:
    """
    Ingest a genomic STR profile into the VANTAGE-STR network.

    Full pipeline:
        1. Pydantic schema validation (enforced by FastAPI automatically).
        2. ForensicValidator assessment:
           - Rule-based pre-checks (allele bounds, CODIS completeness, rarity).
           - DSPy ChainOfThought (when LLM is available).
        3. Decision gate:
           - validity_score >= 0.85 → ACCEPTED: proceed to vectorization.
           - validity_score < 0.85  → QUARANTINED: flagged in PostgreSQL.
           - is_poisoned = True     → REJECTED: blocked entirely.
        4. If accepted, the profile is passed to GenomicVectorizer (Phase 1.2)
           and stored in Milvus (future integration).

    Raises:
        422: If no STR markers are provided.
        403: If profile is flagged as poisoned data.
    """
    if len(profile.str_markers) == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least one STR marker is required for profile ingestion.",
        )

    # ── Validation Gate ──
    validator = _get_validator()

    if validator is not None:
        try:
            result = validator.validate_profile(
                profile_id=str(profile.profile_id),
                str_markers=profile.str_markers,
                node_id=profile.node_id,
                skip_ai=True,  # AI skipped until DSPy LM is configured
            )

            # Rejected profiles are blocked immediately
            if result.decision == "REJECTED":
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail={
                        "error": "Profile rejected — suspected data poisoning",
                        "validity_score": result.validity_score,
                        "anomaly_report": result.anomaly_report,
                        "is_poisoned": result.is_poisoned,
                    },
                )

            # Quarantined profiles are accepted but flagged
            if result.decision == "QUARANTINED":
                alert = QuarantineAlert(
                    profile_id=str(profile.profile_id),
                    node_id=profile.node_id,
                    validity_score=result.validity_score,
                    anomaly_report=result.anomaly_report,
                    is_poisoned=result.is_poisoned,
                    quarantined_at=datetime.now(timezone.utc).isoformat(),
                )
                # TODO: Persist to PostgreSQL quarantine table
                # TODO: Emit WebSocket alert to frontend dashboard
                logger.warning(
                    f"[QUARANTINE] Profile {profile.profile_id} from {profile.node_id} "
                    f"quarantined with score {result.validity_score:.3f}"
                )

            # ── Data Persistence ──
            # Convert Pydantic models to internal tuple format for the analysis engine
            stored_markers = {}
            for marker, locus in profile.str_markers.items():
                stored_markers[marker] = (locus.allele_1, locus.allele_2)
            
            # Store in in-memory DB
            _STR_STORE[str(profile.profile_id)] = stored_markers
            logger.info(f"[INGEST] Stored {len(stored_markers)} markers for {profile.profile_id}")

            return IngestResponse(
                profile_id=str(profile.profile_id),
                node_id=profile.node_id,
                markers_received=len(profile.str_markers),
                accepted=result.decision == "ACCEPTED",
                decision=result.decision,
                validity_score=result.validity_score,
                anomaly_report=result.anomaly_report,
                is_poisoned=result.is_poisoned,
                message=(
                    f"Profile {result.decision.lower()}: {len(profile.str_markers)} markers "
                    f"validated from node '{profile.node_id}' "
                    f"(score: {result.validity_score:.3f})"
                ),
            )

        except HTTPException:
            raise
        except Exception as exc:
            logger.error(f"[MAIN] Validation error for {profile.profile_id}: {exc}")
            # Graceful degradation: accept but log the failure
            pass

    # Fallback if validator is unavailable
    # Persist data even without validation to allow testing
    stored_markers = {}
    for marker, locus in profile.str_markers.items():
        stored_markers[marker] = (locus.allele_1, locus.allele_2)
    _STR_STORE[str(profile.profile_id)] = stored_markers
    
    return IngestResponse(
        profile_id=str(profile.profile_id),
        node_id=profile.node_id,
        markers_received=len(profile.str_markers),
        accepted=True,
        decision="ACCEPTED",
        validity_score=1.0,
        message=(
            f"Profile accepted (no validator): {len(profile.str_markers)} markers "
            f"from node '{profile.node_id}'"
        ),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PHENOTYPE PREDICTION ENDPOINT (Phase 3.3)
# ═══════════════════════════════════════════════════════════════════════════════

# Simple in-memory cache for analysis results (Phase 3.4)
# Prevents re-running GenAI generation on every page load
_ANALYSIS_CACHE: Dict[str, Dict] = {}

@app.get(
    "/profile/phenotype/{profile_id}",
    response_model=PhenotypeReport,
)
async def get_phenotype(profile_id: str) -> PhenotypeReport:
    """
    Predict physical traits from SNP genotype data for a given profile.

    Uses the HIrisPlex-S model to predict:
        - Eye color (Blue / Green-Hazel / Brown)
        - Hair color (Red / Blond / Brown / Black)
        - Skin color (Very Light → Very Dark)
        - Biogeographic ancestry indicators
        - Forensic Facial Reconstruction (GenAI)

    Optionally enhanced with DSPy ChainOfThought biological reasoning
    when the LLM backend is available.

    Args:
        profile_id: UUID or test ID of the profile to analyze.

    Raises:
        404: If no SNP data is found for the given profile_id.
    """
    # Check cache first
    if profile_id in _ANALYSIS_CACHE:
        logger.info(f"[MAIN] Serving cached analysis for {profile_id} (Image URL present)")
        return PhenotypeReport(**_ANALYSIS_CACHE[profile_id])
    
    logger.info(f"[MAIN] No cache for {profile_id}, triggering fresh analysis...")

    # Look up SNP data from store
    snp_map = _SNP_STORE.get(profile_id)
    if snp_map is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No SNP data found for this profile. Phenotype prediction requires genomic input.",
        )

    # Run analysis
    analyst = _get_phenotype_analyst()

    if analyst is not None:
        try:
            result = await analyst.analyze(
                profile_id=profile_id,
                snp_map=snp_map,
                node_id="API-DIRECT",
                skip_ai=False,  # Enable AI/GenAI pipeline
            )
            # Cache the result
            _ANALYSIS_CACHE[profile_id] = result
            return PhenotypeReport(**result)
        except Exception as exc:
            logger.error(f"[MAIN] Phenotype analysis error for {profile_id}: {exc}")
            # Fall through to engine-only


    # Fallback: engine-only prediction
    from app.core.engine.phenotype_engine import PhenotypePredictor
    predictor = PhenotypePredictor()
    result = predictor.predict(profile_id, snp_map)
    
    # ── GEN-AI INTEGRATION (Fallback Mode) ──
    # Explicitly trigger visual reconstruction so the frontend isn't left empty
    try:
        from app.core.engine.prompt_architect import SuspectPromptGenerator
        from app.infrastructure.gen_ai_client import get_gen_ai_client
        
        # 1. Generate Prompt
        # Note: In a real system, we'd infer sex from AMEL marker in STR profile
        # For this fallback, we default to "male" or derive if possible
        sex_hint = "male" 
        if "female" in profile_id.lower(): sex_hint = "female"
        
        prompt_gen = SuspectPromptGenerator()
        generated_prompt = prompt_gen.generate(result, sex_hint=sex_hint)
        
        # 2. Generate Image (Mock or Real)
        # Using "mock" provider by default for speed/stability unless configured
        client = get_gen_ai_client(provider="mock") 
        gen_result = await client.generate_suspect_visual(
            prompt=generated_prompt.positive,
            negative_prompt=generated_prompt.negative,
            seed=generated_prompt.seed
        )
        
        # 3. Enrich Result
        result["image_url"] = gen_result.image_url
        result["positive_prompt"] = generated_prompt.positive
        result["negative_prompt"] = generated_prompt.negative
        result["seed"] = generated_prompt.seed
        result["trait_summary"] = generated_prompt.trait_summary
        result["genai_model_id"] = gen_result.model_id
        
        # Cache the enriched result
        _ANALYSIS_CACHE[profile_id] = result
        
    except Exception as e:
        logger.error(f"[MAIN] Visual reconstruction failed in fallback: {e}")
        # Failure here is non-blocking for the text report, but UI will show NO_DATA image
    
    return PhenotypeReport(**result)


# ═══════════════════════════════════════════════════════════════════════════════
# FACE RECONSTRUCTION ENDPOINT (Phase 3.4)
# ═══════════════════════════════════════════════════════════════════════════════

class ReconstructionResponse(BaseModel):
    """Response from the facial reconstruction pipeline."""
    profile_id: str
    image_url: str
    seed: int
    prompt_hash: str
    generation_time_ms: float
    model_id: str
    trait_summary: Dict[str, str]
    positive_prompt: str
    negative_prompt: str


@app.get(
    "/profile/reconstruct/{profile_id}",
    response_model=ReconstructionResponse,
)
async def reconstruct_face(profile_id: str, sex: str = "male") -> ReconstructionResponse:
    """
    Generate a forensic facial reconstruction from SNP phenotype data.

    Pipeline:
        1. Fetch SNP data for profile_id.
        2. Run PhenotypePredictor → trait probabilities.
        3. SuspectPromptGenerator → Stable Diffusion prompt.
        4. GenAI client → photorealistic image.

    Args:
        profile_id: UUID or test ID of the profile.
        sex: "male" or "female" hint (not derivable from SNPs alone).

    Raises:
        404: If no SNP data found for profile_id.
    """
    # Look up SNP data
    snp_map = _SNP_STORE.get(profile_id)
    if snp_map is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No SNP data found for profile '{profile_id}'. "
                   f"Available test profiles: {list(_SNP_STORE.keys())}",
        )

    # Stage 1: Phenotype prediction
    from app.core.engine.phenotype_engine import PhenotypePredictor
    predictor = PhenotypePredictor()
    phenotype_result = predictor.predict(profile_id, snp_map)

    # Stage 2: Prompt generation
    from app.core.engine.prompt_architect import SuspectPromptGenerator
    prompt_gen = SuspectPromptGenerator()
    generated_prompt = prompt_gen.generate(phenotype_result, sex_hint=sex)

    # Stage 3: Image generation
    from app.infrastructure.gen_ai_client import get_gen_ai_client
    client = get_gen_ai_client(provider="mock")  # Switch to "replicate" for production
    gen_result = await client.generate_suspect_visual(
        prompt=generated_prompt.positive,
        negative_prompt=generated_prompt.negative,
        seed=generated_prompt.seed,
    )

    logger.info(
        f"[RECONSTRUCT] Face generated for {profile_id}: "
        f"seed={gen_result.seed_used}, time={gen_result.generation_time_ms}ms"
    )

    return ReconstructionResponse(
        profile_id=profile_id,
        image_url=gen_result.image_url,
        seed=gen_result.seed_used,
        prompt_hash=gen_result.prompt_hash,
        generation_time_ms=gen_result.generation_time_ms,
        model_id=gen_result.model_id,
        trait_summary=generated_prompt.trait_summary,
        positive_prompt=generated_prompt.positive,
        negative_prompt=generated_prompt.negative,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# FORENSIC ANALYSIS ENDPOINT (Phase 3.5)
# Dynamic Likelihood Ratio Engine
# ═══════════════════════════════════════════════════════════════════════════════

class AnalysisRequest(BaseModel):
    """Request body for forensic analysis."""
    profile_id: str
    population: str = "European"


class PerLocusDetail(BaseModel):
    """Individual locus LR detail for the API response."""
    marker: str
    alleles: list
    is_homozygous: bool
    frequencies: list
    genotype_probability: float
    individual_lr: float
    log10_lr: float
    rarity_score: float


class AnalysisResponse(BaseModel):
    """Complete forensic analysis result."""
    profile_id: str
    population: str
    combined_lr: float
    log10_lr: float
    random_match_probability: float
    random_match_probability_str: str
    verbal_equivalence: str
    prosecution_probability: float
    defense_probability: float
    loci_analyzed: int
    per_locus_details: list
    high_frequency_warning: bool
    warning_message: str
    match_classification: str
    recommended_action: str
    forensic_hypothesis: str
    certainty_report: str
    thought_chain: list
    total_analysis_time_ms: float
    # Phase 3.6 — Kinship
    kinship_result: Optional[dict] = None
    familial_hit_detected: bool = False
    # Phase 3.7 — Bayesian Inference
    bayesian_posterior: float = 0.0
    prior_hp: float = 0.5
    bayesian_ci_lower: float = 0.0
    bayesian_ci_upper: float = 0.0
    degradation_index: float = 0.0
    dropout_warnings: list = []
    stutter_warnings: list = []
    iso17025_verbal: str = "INCONCLUSIVE"
    sensitivity_map: list = []


class KinshipRequest(BaseModel):
    """Request for kinship analysis between two profiles."""
    profile_a_id: str
    profile_b_id: str
    population: str = "European"


@app.post(
    "/profile/analyze",
    response_model=AnalysisResponse,
)
def analyze_profile(req: AnalysisRequest) -> AnalysisResponse:
    """
    Run the Dynamic Likelihood Ratio Engine on an ingested STR profile.

    Pipeline:
        1. Retrieve STR markers from profile store.
        2. Compute per-locus LR using real population allele frequencies.
        3. Compute CLR via product rule.
        4. Run ForensicAnalyst with dynamic reasoning (CLR-aware).
        5. Return full statistical breakdown + agent reasoning.

    Changing ANY allele value cascades through the entire analysis.
    """
    import hashlib

    profile_id = req.profile_id
    population = req.population

    # — Stage 1: Retrieve STR markers —
    str_markers = _STR_STORE.get(profile_id)

    if str_markers is None or len(str_markers) == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No genetic markers provided for analysis. Please ingest a valid profile first.",
        )

    # — Stage 2: Compute LR —
    from app.core.forensics.lr_calculator import compute_combined_lr
    lr_result = compute_combined_lr(str_markers, population)

    # — Stage 3: Run ForensicAnalyst with LR data —
    import time as _time
    t_start = _time.perf_counter()

    from app.agents.investigator_logic import ForensicAnalyst
    analyst = ForensicAnalyst()

    mock_matches = [{
        "node_id": "LOCAL-PRIMARY",
        "match_score": lr_result.prosecution_probability,
        "profile_id": profile_id,
        "local_reference_token": profile_id[:20],
    }]

    loci_detail = {"loci": {}}
    for detail in lr_result.per_locus_details:
        loci_detail["loci"][detail.marker] = {
            "query_a1": detail.allele_1,
            "query_a2": detail.allele_2,
            "match_a1": detail.allele_1,
            "match_a2": detail.allele_2,
            "freq_a1": detail.freq_1,
            "freq_a2": detail.freq_2,
        }

    investigation = analyst.investigate(
        query_id=f"ANALYSIS-{profile_id[:8]}",
        match_results=mock_matches,
        case_context=(
            f"Forensic STR analysis for profile {profile_id}. "
            f"Population: {population}. "
            f"CLR: {lr_result.combined_lr:.2e}. "
            f"Verbal: {lr_result.verbal_equivalence}. "
            f"{lr_result.warning_message}"
        ),
        loci_detail=loci_detail,
    )

    total_ms = (_time.perf_counter() - t_start) * 1000

    return AnalysisResponse(
        profile_id=profile_id,
        population=population,
        combined_lr=lr_result.combined_lr,
        log10_lr=lr_result.log10_lr,
        random_match_probability=lr_result.random_match_probability,
        random_match_probability_str=lr_result.random_match_probability_str,
        verbal_equivalence=lr_result.verbal_equivalence,
        prosecution_probability=lr_result.prosecution_probability,
        defense_probability=lr_result.defense_probability,
        loci_analyzed=lr_result.loci_analyzed,
        per_locus_details=[d.to_dict() for d in lr_result.per_locus_details],
        high_frequency_warning=lr_result.high_frequency_warning,
        warning_message=lr_result.warning_message,
        match_classification=investigation.match_classification.value,
        recommended_action=investigation.recommended_action.value,
        forensic_hypothesis=investigation.forensic_hypothesis,
        certainty_report=investigation.certainty_report,
        thought_chain=[
            {
                "step_number": t.step_number,
                "phase": t.phase,
                "content": t.content,
                "duration_ms": round(t.duration_ms, 2),
                "confidence": round(t.confidence, 4),
            }
            for t in investigation.thought_chain
        ],
        total_analysis_time_ms=round(total_ms, 2),
        kinship_result=investigation.kinship_result,
        familial_hit_detected=investigation.familial_hit_detected,
        # Phase 3.7 — Bayesian Inference
        bayesian_posterior=lr_result.posterior_hp,
        prior_hp=lr_result.prior_hp,
        bayesian_ci_lower=lr_result.bayesian_ci_lower,
        bayesian_ci_upper=lr_result.bayesian_ci_upper,
        degradation_index=lr_result.degradation_index,
        dropout_warnings=lr_result.dropout_warnings,
        stutter_warnings=lr_result.stutter_warnings,
        iso17025_verbal=lr_result.iso17025_verbal,
        sensitivity_map=lr_result.sensitivity_map,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# KINSHIP ANALYSIS ENDPOINT (PHASE 3.6)
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/profile/kinship")
def kinship_analysis(req: KinshipRequest):
    """
    Compute Kinship Index between two STR profiles.

    Pipeline:
        1. Retrieve both STR profiles from store.
        2. Compute per-locus KI for parent-child, full sibling, half sibling.
        3. Classify best-fit biological relationship.
        4. Return full kinship breakdown with IBD summary.
    """
    profile_a = _STR_STORE.get(req.profile_a_id)
    profile_b = _STR_STORE.get(req.profile_b_id)

    if not profile_a:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Profile '{req.profile_a_id}' not found in STR store.",
        )
    if not profile_b:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Profile '{req.profile_b_id}' not found in STR store.",
        )

    from app.core.forensics.kinship_engine import compute_kinship
    result = compute_kinship(profile_a, profile_b, req.population)

    logger.info(
        f"[KINSHIP] {req.profile_a_id} vs {req.profile_b_id}: "
        f"{result.relationship_type.value} (confidence: {result.confidence:.4f})"
    )

    return result.to_dict()


if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)
