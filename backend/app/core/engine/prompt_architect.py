"""
SuspectPromptGenerator — VANTAGE-STR Forensic 3D Composite System.

Phase 3.4: Generative AI Face Reconstruction.

/// ROLE & LEGAL DIRECTIVE ///
This module is the VANTAGE-STR Digital Forensic Modeler. It generates
3D Composite Profile prompts based on genetic data. It strictly avoids
generating realistic human photography to remain compliant with privacy
laws (KVKK/GDPR). Output must clearly look like a high-fidelity 3D
digital reconstruction, similar to medical or forensic modeling software.

/// VISUAL STYLE ///
- Medium: Professional 3D CGI rendering (Medical Simulation / Forensic Reconstruction)
- Skin: Clean, simplified 3D skin shader. Matte silicone or digital clay.
- Lighting: Flat, overhead studio lighting for craniofacial geometry.
- Background: Solid neutral gradient (light blue to white), clinical lab aesthetic.
- Banner: "COMPOSITE PROFILE" header (embedded in prompt)
- Watermark: DNA double-helix icon at base of neck

Reference: FISWG (Facial Identification Scientific Working Group) guidelines.
"""

import hashlib
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# PHENOTYPE → 3D COMPOSITE DESCRIPTOR MAPPING
# ═══════════════════════════════════════════════════════════════════════════════

# Eye color — 3D digital iris descriptors
EYE_COLOR_DESCRIPTORS: Dict[str, str] = {
    "Blue": (
        "clear steel blue 3D-rendered irises, digital non-organic iris texture, "
        "visible geometric iris fiber pattern, clean specular highlight"
    ),
    "Green/Hazel": (
        "hazel-green 3D-rendered irises with amber ring, "
        "digital non-organic iris texture, clean geometric pattern"
    ),
    "Brown": (
        "deep brown 3D-rendered irises, rich warm digital iris texture, "
        "smooth non-organic surface, clean specular highlight"
    ),
}

# Hair color — 3D modeled clumps, not individual strands
HAIR_COLOR_DESCRIPTORS: Dict[str, str] = {
    "Red": (
        "3D-modeled auburn-red hair, simplified fiber clumps, "
        "solid copper tone, clean sculpted lines, no stray hairs"
    ),
    "Blond": (
        "3D-modeled light blond hair, simplified golden fiber clumps, "
        "solid color, clean sculpted lines, no stray hairs"
    ),
    "Brown": (
        "3D-modeled dark brown hair, simplified fiber clumps, "
        "solid matte finish, clean sculpted lines, no stray hairs"
    ),
    "Black": (
        "3D-modeled jet-black hair, high-density simplified clumps, "
        "solid dark color, clean sculpted lines, no stray hairs"
    ),
}

# Skin tone — matte silicone / digital clay shader
SKIN_TONE_DESCRIPTORS: Dict[str, str] = {
    "Very Light": (
        "very fair 3D skin shader, clean matte porcelain surface, "
        "Fitzpatrick Type I-II digital tone, smooth silicone-like texture, "
        "no micro-pores, no blemishes"
    ),
    "Light": (
        "light 3D skin shader with subtle warm undertones, "
        "Fitzpatrick Type II-III digital tone, clean matte surface, "
        "smooth silicone-like texture, no micro-pores"
    ),
    "Intermediate": (
        "medium olive 3D skin shader, warm golden digital tone, "
        "Fitzpatrick Type III-IV, clean matte surface, "
        "smooth silicone-like texture, even color"
    ),
    "Dark": (
        "deep brown 3D skin shader, rich melanin digital tone, "
        "Fitzpatrick Type V, clean matte surface, "
        "smooth silicone-like texture, even warm color"
    ),
    "Very Dark": (
        "very dark ebony 3D skin shader, deep digital tone, "
        "Fitzpatrick Type VI, clean matte surface, "
        "smooth silicone-like texture, rich even color"
    ),
}

# Ancestry → craniofacial bone structure (geometric precision)
ANCESTRY_FACIAL_DESCRIPTORS: Dict[str, str] = {
    "European": (
        "Northern European craniofacial geometry, "
        "moderate brow ridge with sharp geometric clarity, "
        "narrow nasal bridge, defined angular cheekbones, "
        "medium lip volume, angular jawline"
    ),
    "African": (
        "West African craniofacial geometry, "
        "broad nasal bridge with rounded geometric form, "
        "full lips, prominent high cheekbones, "
        "wide facial structure, rounded jawline"
    ),
    "East Asian": (
        "East Asian craniofacial geometry, "
        "epicanthic fold modeled, flat nasal bridge, "
        "high prominent cheekbones, moderate lip volume, "
        "rounded jawline, smooth forehead plane"
    ),
    "South Asian": (
        "South Asian craniofacial geometry, "
        "moderate nasal bridge, defined brow ridge, "
        "high cheekbones, medium lip volume, "
        "oval face shape"
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# CORE 3D FORENSIC COMPOSITE SYSTEM PROMPT
# ═══════════════════════════════════════════════════════════════════════════════

FORENSIC_SYSTEM_PROMPT: str = (
    "Professional 3D CGI forensic facial reconstruction, "
    "medical simulation quality, digital forensic composite profile. "
    "High-fidelity 3D digital render, NOT a photograph. "
    "Clean simplified 3D skin shader, matte silicone or digital clay surface, "
    "no realistic micro-pores, no photographic imperfections, no real-world blemishes. "
    "Flat overhead studio lighting emphasizing craniofacial geometry, "
    "no dramatic shadows, no cinematic lighting. "
    "Solid neutral gradient background from light blue to white, "
    "clinical forensic laboratory aesthetic. "
    "Thick red banner at the top reading 'COMPOSITE PROFILE' in bold white sans-serif typography. "
    "Small DNA double-helix icon watermark at the base of the neck. "
    "Skeletal foundation focus with sharp geometric clarity on bone structure. "
    "3D-modeled simplified hair fiber clumps, solid color, clean lines. "
    "3D-rendered digital eye models with clear color but non-organic iris texture. "
    "Front-facing, neutral expression, no emotion. "
    "Single subject only"
)

# ── Negative Prompt — Blocks ALL Photorealism ──
NEGATIVE_PROMPT: str = (
    "real human photo, raw photography, 8k portrait, cinematic, "
    "skin pores, realistic sweat, bokeh, real-world background, "
    "jewelry, smiling, emotional expressions, messy hair, "
    "artistic shadows, film grain, DSLR style, "
    "Hasselblad, Canon, Nikon, camera lens, shallow depth of field, "
    "photorealistic, hyperrealistic, realistic skin texture, "
    "blemishes, wrinkles, acne, scars, blood vessels, "
    "vellus hair, peach fuzz, subsurface scattering, "
    "anime, cartoon, painting, watercolor, sketch, illustration, "
    "nudity, nsfw, suggestive, "
    "multiple people, extra limbs, deformed, disfigured, "
    "bad anatomy, bad proportions, mutation, extra fingers, "
    "watermark, text overlay, logo, signature, border, frame"
)


# ═══════════════════════════════════════════════════════════════════════════════
# PROMPT GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GeneratedPrompt:
    """Complete prompt payload for image generation."""
    positive: str
    negative: str
    seed: int
    profile_id: str
    trait_summary: Dict[str, str]


class SuspectPromptGenerator:
    """
    VANTAGE-STR Digital Forensic Modeler.

    Converts PhenotypeReport data into 3D CGI composite prompts that are
    KVKK/GDPR compliant. Output resembles medical/forensic modeling software,
    NOT photography.

    Architecture:
        1. Extract dominant phenotypic traits (eye, hair, skin, ancestry)
        2. Map to 3D composite descriptors (matte, geometric, clinical)
        3. Compose prompt: system → subject → morphology → features
        4. Apply anti-photorealism negative prompt
        5. Deterministic seed from profile_id hash
    """

    def __init__(self) -> None:
        logger.info("[PROMPT-ARCH] SuspectPromptGenerator initialized (3D Composite Mode — KVKK/GDPR Compliant)")

    @staticmethod
    def _hash_seed(profile_id: str) -> int:
        """
        Generate a deterministic seed from profile_id.
        SHA-256 → 32-bit integer. Same DNA → same reconstruction.
        """
        hash_bytes = hashlib.sha256(profile_id.encode("utf-8")).digest()
        seed = int.from_bytes(hash_bytes[:4], byteorder="big") & 0x7FFFFFFF
        return seed

    @staticmethod
    def _extract_dominant_trait(
        traits: List[Dict],
        trait_name: str,
    ) -> Tuple[str, float]:
        """Extract the dominant prediction for a given trait."""
        for trait in traits:
            if trait.get("trait") == trait_name:
                return trait.get("dominant_prediction", "Unknown"), trait.get("confidence", 0.0)
        return "Unknown", 0.0

    @staticmethod
    def _extract_dominant_ancestry(ancestry: Dict[str, float]) -> Tuple[str, float]:
        """Extract the most likely ancestry from indicators."""
        if not ancestry:
            return "Unknown", 0.0
        dominant = max(ancestry, key=ancestry.get)
        return dominant, ancestry[dominant]

    @staticmethod
    def _derive_age_range(profile_id: str) -> str:
        """Derive a plausible age range from profile hash."""
        hash_bytes = hashlib.sha256(profile_id.encode("utf-8")).digest()
        age_byte = hash_bytes[5]
        age_ranges = [
            "early 20s", "mid 20s", "late 20s",
            "early 30s", "mid 30s", "late 30s",
            "early 40s", "mid 40s",
        ]
        return age_ranges[age_byte % len(age_ranges)]

    @staticmethod
    def _derive_bone_features(profile_id: str, ancestry_pred: str) -> str:
        """
        Derive craniofacial bone structure features from profile hash.
        These are rendered with geometric precision in the 3D composite.
        """
        hash_bytes = hashlib.sha256(profile_id.encode("utf-8")).digest()
        features = []

        # Jawline geometry
        if hash_bytes[6] % 3 == 0:
            features.append("prominent angular jawline with geometric definition")
        elif hash_bytes[6] % 3 == 1:
            features.append("rounded jawline with smooth 3D contour")

        # Brow ridge
        if hash_bytes[7] % 4 == 0:
            features.append("heavy brow ridge rendered with sharp geometric clarity")

        # Nasal bridge detail
        if hash_bytes[8] % 3 == 0:
            features.append("narrow nasal bridge with defined geometric tip")
        elif hash_bytes[8] % 3 == 1:
            features.append("wide nasal bridge with rounded geometric form")

        # Ear lobes
        if hash_bytes[9] % 2 == 0:
            features.append("attached earlobes")
        else:
            features.append("detached earlobes")

        # Cheekbone prominence
        if hash_bytes[10] % 3 == 0:
            features.append("high prominent cheekbones with sharp geometric planes")

        return ", ".join(features) if features else ""

    def generate(
        self,
        phenotype_report: Dict,
        sex_hint: str = "male",
    ) -> GeneratedPrompt:
        """
        Generate a 3D forensic composite prompt from phenotype data.

        Privacy-compliant output: 3D CGI reconstruction, NOT photography.
        Includes COMPOSITE PROFILE banner and DNA watermark.

        Args:
            phenotype_report: Dict from PhenotypePredictor/PhenotypeAnalyst.
            sex_hint: "male" or "female".

        Returns:
            GeneratedPrompt with positive/negative prompts and seed.
        """
        traits = phenotype_report.get("traits", [])
        ancestry = phenotype_report.get("ancestry_indicators", {})
        profile_id = phenotype_report.get("profile_id", "unknown")

        # ── Extract dominant predictions ──
        eye_pred, eye_conf = self._extract_dominant_trait(traits, "Eye Color")
        hair_pred, hair_conf = self._extract_dominant_trait(traits, "Hair Color")
        skin_pred, skin_conf = self._extract_dominant_trait(traits, "Skin Color")
        ancestry_pred, ancestry_conf = self._extract_dominant_ancestry(ancestry)

        # ── Map to 3D composite descriptors ──
        eye_desc = EYE_COLOR_DESCRIPTORS.get(eye_pred, "neutral colored 3D-rendered digital irises")
        hair_desc = HAIR_COLOR_DESCRIPTORS.get(hair_pred, "3D-modeled natural-colored hair clumps")
        skin_desc = SKIN_TONE_DESCRIPTORS.get(skin_pred, "neutral 3D skin shader, clean matte surface")
        ancestry_desc = ANCESTRY_FACIAL_DESCRIPTORS.get(
            ancestry_pred,
            "mixed-ancestry craniofacial geometry, balanced proportions"
        )

        # ── Derive additional data ──
        age_range = self._derive_age_range(profile_id)
        bone_features = self._derive_bone_features(profile_id, ancestry_pred)

        # ── Build trait summary for UI overlay ──
        trait_summary: Dict[str, str] = {
            "Eye Color": f"{eye_pred} ({eye_conf:.0%})",
            "Hair Color": f"{hair_pred} ({hair_conf:.0%})",
            "Skin Tone": f"{skin_pred} ({skin_conf:.0%})",
            "Ancestry": f"{ancestry_pred} ({ancestry_conf:.0%})",
            "Sex": sex_hint.capitalize(),
            "Age Range": age_range,
        }

        # ── Compose 3D composite prompt ──
        sex_token = "adult male" if sex_hint.lower() == "male" else "adult female"

        prompt_parts: List[str] = [
            # Layer 1: System — 3D CGI enforcement + banner + watermark
            FORENSIC_SYSTEM_PROMPT,
            # Layer 2: Subject identity
            f"(({sex_token} 3D composite)), appears to be in their {age_range}",
            # Layer 3: Bone structure (craniofacial geometry)
            f"(({ancestry_desc}))",
            # Layer 4: Eyes (3D digital model)
            f"(({eye_desc}))",
            # Layer 5: Hair (3D sculpted clumps)
            f"({hair_desc})",
            # Layer 6: Skin (matte shader)
            f"({skin_desc})",
        ]

        # Layer 7: Bone features
        if bone_features:
            prompt_parts.append(f"{bone_features}")

        positive_prompt = ", ".join(prompt_parts)

        # ── Generate deterministic seed ──
        seed = self._hash_seed(profile_id)

        logger.info(
            f"[PROMPT-ARCH] ═══ 3D Composite Prompt Generated ═══\n"
            f"  Profile: {profile_id}\n"
            f"  Eye: {eye_pred} ({eye_conf:.0%}) | Hair: {hair_pred} ({hair_conf:.0%})\n"
            f"  Skin: {skin_pred} ({skin_conf:.0%}) | Ancestry: {ancestry_pred} ({ancestry_conf:.0%})\n"
            f"  Age: {age_range} | Seed: {seed}\n"
            f"  Bone Features: {bone_features or 'None'}\n"
            f"  Prompt Length: {len(positive_prompt)} chars\n"
            f"  Mode: KVKK/GDPR COMPLIANT — 3D CGI ONLY"
        )

        return GeneratedPrompt(
            positive=positive_prompt,
            negative=NEGATIVE_PROMPT,
            seed=seed,
            profile_id=profile_id,
            trait_summary=trait_summary,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# FEDERATED REFINEMENT STUB (Phase 3.4+)
# ═══════════════════════════════════════════════════════════════════════════════

class FederatedTraitRefiner:
    """
    Multi-node phenotype refinement stub.

    Merges independently derived phenotype predictions from multiple
    VANTAGE-STR nodes to produce a higher-confidence composite.
    """

    @staticmethod
    def merge_reports(reports: List[Dict], weights: Optional[List[float]] = None) -> Dict:
        """Merge trait predictions from multiple nodes."""
        if not reports:
            return {}

        if weights is None:
            weights = [1.0 / len(reports)] * len(reports)

        total_w = sum(weights)
        weights = [w / total_w for w in weights]

        merged_traits: Dict[str, Dict[str, float]] = {}
        for report, weight in zip(reports, weights):
            for trait in report.get("traits", []):
                trait_name = trait.get("trait", "")
                predictions = trait.get("predictions", {})
                if trait_name not in merged_traits:
                    merged_traits[trait_name] = {}
                for outcome, prob in predictions.items():
                    merged_traits[trait_name][outcome] = (
                        merged_traits[trait_name].get(outcome, 0.0) + prob * weight
                    )

        merged_ancestry: Dict[str, float] = {}
        for report, weight in zip(reports, weights):
            for ancestry, prob in report.get("ancestry_indicators", {}).items():
                merged_ancestry[ancestry] = merged_ancestry.get(ancestry, 0.0) + prob * weight

        merged_trait_list = []
        for trait_name, predictions in merged_traits.items():
            dominant = max(predictions, key=predictions.get)
            merged_trait_list.append({
                "trait": trait_name,
                "predictions": {k: round(v, 4) for k, v in predictions.items()},
                "dominant_prediction": dominant,
                "confidence": round(predictions[dominant], 4),
                "contributing_snps": [],
            })

        logger.info(
            f"[FED-REFINE] Merged {len(reports)} node reports "
            f"({len(merged_traits)} traits, {len(merged_ancestry)} ancestry markers)"
        )

        return {
            "profile_id": reports[0].get("profile_id", "merged"),
            "snps_analyzed": max(r.get("snps_analyzed", 0) for r in reports),
            "hirisplex_coverage": max(r.get("hirisplex_coverage", 0.0) for r in reports),
            "traits": merged_trait_list,
            "ancestry_indicators": {k: round(v, 4) for k, v in merged_ancestry.items()},
            "ai_reasoning": f"FEDERATED MERGE: Consensus from {len(reports)} nodes",
        }
