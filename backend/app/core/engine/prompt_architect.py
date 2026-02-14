"""
SuspectPromptGenerator — VANTAGE-STR Forensic Photorealistic System.

Phase 3.4: Generative AI Face Reconstruction.

/// ROLE & LEGAL DIRECTIVE ///
This module is the VANTAGE-STR Digital Forensic Modeler. It generates
high-fidelity forensic mugshot prompts based on genetic data.
Output must be indistinguishable from a real photograph captured in a
controlled forensic laboratory setting.

/// VISUAL STYLE ///
- Medium: Raw Forensic Photography (85mm lens, f/8 aperture).
- Skin: Unretouched, visible pores, uneven tone, natural oils.
- Lighting: Harsh clinical overhead lighting, neutral gray background.
- Context: Mugshot / Clinical Portrait.
- No Aesthetic Bias: Neutral, objective, anatomical.

Reference: FISWG (Facial Identification Scientific Working Group) guidelines.
"""

import hashlib
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# PHENOTYPE → FORENSIC DESCRIPTOR MAPPING
# ═══════════════════════════════════════════════════════════════════════════════

# Eye descriptors - Medical/Anatomical
EYE_COLOR_DESCRIPTORS: Dict[str, str] = {
    "Blue": (
        "piercing light blue eyes with realistic iris patterns, "
        "visible limbal ring, distinct crypts and furrows, "
        "clear sclera with natural vascularization"
    ),
    "Green/Hazel": (
        "complex hazel-green eyes with central heterochromia, "
        "amber collarette around the pupil, dark limbal ring, "
        "highly detailed iris stroma"
    ),
    "Brown": (
        "deep brown eyes with rich melanin density, "
        "smooth velvety iris texture, visible contraction furrows, "
        "natural light reflection"
    ),
}

# Hair descriptors - Morphology & Texture
HAIR_COLOR_DESCRIPTORS: Dict[str, str] = {
    "Red": (
        "natural auburn-red hair with photorealistic texture, "
        "mix of copper and strawberry blond individual strands, "
        "visible vellus hair at hairline"
    ),
    "Blond": (
        "natural light blond hair, fine texture, "
        "translucent tonal variation, visible individual strands, "
        "light refraction through hair shaft"
    ),
    "Brown": (
        "dark brown hair with natural sheen, "
        "thick individual strands, visible cuticle texture, "
        "realistic hairline irregularity"
    ),
    "Black": (
        "jet-black hair with high melanin content, "
        "coarse texture, light absorbing, "
        "high-contrast individual strands against scalp"
    ),
}

# Skin descriptors - Raw Texture & Reflectance
SKIN_TONE_DESCRIPTORS: Dict[str, str] = {
    "Very Light": (
        "Fitzpatrick Type I skin tone, pale with high UV sensitivity, "
        "translucent epidermis showing underlying vascularity, "
        "visible scattered ephelides (freckles), unretouched texture"
    ),
    "Light": (
        "Fitzpatrick Type II-III skin tone, light beige with subtle warm undertones, "
        "uneven pigmentation, visible pores and micro-comedones, "
        "natural skin oils, unretouched"
    ),
    "Intermediate": (
        "Fitzpatrick Type III-IV skin tone, olive complexion, "
        "even melanin distribution, distinct pore structure on nose and cheeks, "
        "natural localized hyperpigmentation, matte texture"
    ),
    "Dark": (
        "Fitzpatrick Type V skin tone, deep brown melanin rich, "
        "smooth texture with high specular highlights on cheekbones, "
        "visible pores, unretouched natural skin sheen"
    ),
    "Very Dark": (
        "Fitzpatrick Type VI skin tone, deep ebony complexion, "
        "light-absorbing epidermis with sharp specular highlights, "
        "smooth dense texture, unretouched"
    ),
}

# Ancestry descriptors - Craniofacial Anthropometry
ANCESTRY_FACIAL_DESCRIPTORS: Dict[str, str] = {
    "European": (
        "European ancestry facial morphology, Orthognathic profile, "
        "prominent nasal spine, narrow high nasal bridge, "
        "sharp zygomatic arches, thin lips, defined mandibular angle"
    ),
    "African": (
        "Sub-Saharan African ancestry facial morphology, Prognathic profile, "
        "wider interorbital distance, broad nasal bridge and aperture, "
        "full mucosal lip height, convex maxilla, strong jawline"
    ),
    "East Asian": (
        "East Asian ancestry facial morphology, "
        "Orthognathic profile with flatter midface, "
        "pronounced zygomatic projection, low nasal bridge, "
        "presence of epicanthic folds, shovel-shaped incisors indication"
    ),
    "South Asian": (
        "South Asian ancestry facial morphology, "
        "mesocephalic head shape, moderate nasal bridge height, "
        "distinct almond-shaped orbital structure, "
        "full lips, soft tissue thickness variance"
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# CORE PROMPT TEMPLATES (SENIOR FORENSIC GENETICIST MODE)
# ═══════════════════════════════════════════════════════════════════════════════

NEGATIVE_PROMPT: str = (
    "3d render, cgi, cartoon, anime, illustration, painting, drawing, sketch, "
    "smooth skin, plastic skin, doll, low resolution, blurry, distorted, "
    "makeup, jewelry, smiling, emotional, artistic lighting, cinematic shadows, "
    "filters, airbrushed, photoshop, watermark, text, logo, "
    "deformed, bad anatomy, disfigured, mutation, extra limbs"
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
    VANTAGE-STR Digital Forensic Modeler (Photorealistic Mode).
    
    Translates genetic data into standard forensic photography prompts.
    """

    def __init__(self) -> None:
        logger.info("[PROMPT-ARCH] SuspectPromptGenerator initialized (Mode: SENIOR FORENSIC GENETICIST)")

    @staticmethod
    def _hash_seed(profile_id: str) -> int:
        hash_bytes = hashlib.sha256(profile_id.encode("utf-8")).digest()
        seed = int.from_bytes(hash_bytes[:4], byteorder="big") & 0x7FFFFFFF
        return seed

    @staticmethod
    def _extract_dominant_trait(traits: List[Dict], trait_name: str) -> Tuple[str, float]:
        for trait in traits:
            if trait.get("trait") == trait_name:
                return trait.get("dominant_prediction", "Unknown"), trait.get("confidence", 0.0)
        return "Unknown", 0.0

    @staticmethod
    def _extract_dominant_ancestry(ancestry: Dict[str, float]) -> Tuple[str, float]:
        if not ancestry:
            return "Unknown", 0.0
        dominant = max(ancestry, key=ancestry.get)
        return dominant, ancestry[dominant]

    @staticmethod
    def _derive_age_range(profile_id: str) -> str:
        hash_bytes = hashlib.sha256(profile_id.encode("utf-8")).digest()
        age_byte = hash_bytes[5]
        # Forensic estimates are rarely exact, using ranges
        age_ranges = ["20-25", "25-30", "30-35", "35-40", "40-45", "45-50"]
        return age_ranges[age_byte % len(age_ranges)]

    @staticmethod
    def _derive_bone_features(profile_id: str) -> str:
        """Derive specific craniometric features from hash."""
        hash_bytes = hashlib.sha256(profile_id.encode("utf-8")).digest()
        features = []
        
        # Mandible
        if hash_bytes[6] % 3 == 0:
            features.append("strong square mandibular angle")
        elif hash_bytes[6] % 3 == 1:
            features.append("soft rounded chin")
            
        # Zygomatic
        if hash_bytes[7] % 2 == 0:
            features.append("high prominent zygomatic arches")
            
        # Nasal
        if hash_bytes[8] % 3 == 0:
            features.append("deviated nasal septum")
        
        return ", ".join(features)

    def generate(self, phenotype_report: Dict, sex_hint: str = "male") -> GeneratedPrompt:
        traits = phenotype_report.get("traits", [])
        ancestry = phenotype_report.get("ancestry_indicators", {})
        profile_id = phenotype_report.get("profile_id", "unknown")

        eye_pred, eye_conf = self._extract_dominant_trait(traits, "Eye Color")
        hair_pred, hair_conf = self._extract_dominant_trait(traits, "Hair Color")
        skin_pred, skin_conf = self._extract_dominant_trait(traits, "Skin Color")
        ancestry_pred, ancestry_conf = self._extract_dominant_ancestry(ancestry)

        eye_desc = EYE_COLOR_DESCRIPTORS.get(eye_pred, "photorealistic eyes")
        hair_desc = HAIR_COLOR_DESCRIPTORS.get(hair_pred, "natural hair texture")
        skin_desc = SKIN_TONE_DESCRIPTORS.get(skin_pred, "natural skin texture with visible pores")
        ancestry_desc = ANCESTRY_FACIAL_DESCRIPTORS.get(ancestry_pred, "mixed ancestry facial morphology")
        
        age_range = self._derive_age_range(profile_id)
        bone_features = self._derive_bone_features(profile_id)
        
        sex_term = "Male" if sex_hint.lower() == "male" else "Female"

        # ── Construct the "Senior Forensic Geneticist" Prompt ──
        # Template: "A clinical... mugshot of a [SEX], estimated age [AGE]. [ANCESTRY]... [EYES]... [HAIR]... [SKIN]... [CAM SPECS]"
        
        positive_prompt = (
            f"A clinical, high-resolution forensic mugshot of a {sex_term}, estimated age {age_range}. "
            f"{ancestry_desc}. "
            f"Facial structure features: {bone_features}. "
            f"Genetically predicted {eye_desc}. "
            f"{hair_desc}. "
            f"Skin tone is {skin_desc}. "
            f"Front-facing view, neutral expression. "
            f"Captured on 85mm lens, raw photo style, unretouched, 8k resolution, harsh forensic lighting, plain concrete background."
        )

        seed = self._hash_seed(profile_id)

        logger.info(
            f"[PROMPT-ARCH] Generated Forensic Prompt (Profile: {profile_id})\n"
            f"  Prompt: {positive_prompt[:100]}..."
        )

        trait_summary = {
            "Eye Color": f"{eye_pred} ({eye_conf:.0%})",
            "Hair Color": f"{hair_pred} ({hair_conf:.0%})",
            "Skin Tone": f"{skin_pred} ({skin_conf:.0%})",
            "Ancestry": f"{ancestry_pred} ({ancestry_conf:.0%})",
            "Sex": sex_term,
            "Age Range": age_range,
        }

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
