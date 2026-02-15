from typing import Dict, Any, List

class PhenotypeEngine:
    """
    Deterministic SNP-to-Phenotype mapping engine for forensic traits.
    Focuses on HERC2, MC1R, SLC24A5, SLC45A2.
    """
    
    # Major Forensic SNPs
    SNPS = {
        "rs12913832": {"gene": "HERC2", "trait": "Eye Color"},
        "rs1805007": {"gene": "MC1R", "trait": "Red Hair / Fair Skin"},
        "rs1426654": {"gene": "SLC24A5", "trait": "Skin Tone"},
        "rs16891982": {"gene": "SLC45A2", "trait": "Skin Tone / Hair Color"}
    }
    
    def predict_phenotype(self, snp_data: Dict[str, str], ancestry_probabilities: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Predicts physical traits based on SNP alleles and ancestry context.
        Applies Bayesian weighting if ancestry confidence is high (>70%).
        """
        traits = {
            "Ocular Pigmentation": "Insufficient Data",
            "Hair Morphology": "Insufficient Data",
            "Dermal Classification": "Insufficient Data",
            "Freckling Index": "Unknown"
        }
        
        # Determine dominant ancestry
        top_region = "Unknown"
        top_prob = 0.0
        if ancestry_probabilities:
            sorted_regions = sorted(
                ancestry_probabilities.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            if sorted_regions:
                top_region = sorted_regions[0][0]
                top_prob = sorted_regions[0][1]

        # 1. Eye Color (HERC2 rs12913832)
        herc2 = snp_data.get("rs12913832", "??")
        if "AA" in herc2:
            traits["Ocular Pigmentation"] = "Dark Brown / Black"
        elif "AG" in herc2 or "GA" in herc2:
            traits["Ocular Pigmentation"] = "Hazel / Green"
        elif "GG" in herc2:
             traits["Ocular Pigmentation"] = "Blue / Gray"
        else:
             # Bayesian Fallback
             if top_prob > 0.7:
                 if "Europe" in top_region:
                     traits["Ocular Pigmentation"] = "Likely Light/Intermediate (Bayesian Prior)"
                 elif "Asia" in top_region or "Africa" in top_region:
                     traits["Ocular Pigmentation"] = "Likely Dark Brown (Bayesian Prior)"
                 
        # 2. Skin Tone (SLC24A5 rs1426654)
        slc24 = snp_data.get("rs1426654", "??")
        if "AA" in slc24:
            traits["Dermal Classification"] = "Light / Pale (Type I-II)"
        elif "AG" in slc24 or "GA" in slc24:
            traits["Dermal Classification"] = "Intermediate / Tan (Type III-IV)"
        elif "GG" in slc24:
            traits["Dermal Classification"] = "Dark / Deep (Type V-VI)"
        else:
             if top_prob > 0.7:
                 if "Europe" in top_region:
                     traits["Dermal Classification"] = "Likely Type II-III (Bayesian Prior)"
                 elif "Africa" in top_region:
                     traits["Dermal Classification"] = "Likely Type V-VI (Bayesian Prior)"
                 elif "Asia" in top_region:
                     traits["Dermal Classification"] = "Likely Type III-IV (Bayesian Prior)"

        # 3. Hair Color/Type (Review MC1R & SLC45A2)
        mc1r = snp_data.get("rs1805007", "??") # R151C
        slc45 = snp_data.get("rs16891982", "??") # L374F
        
        hair_color = "Brown/Black"
        hair_texture = "Straight/Wavy"
        
        if "TT" in mc1r: # Red hair variant
             hair_color = "Red / Strawberry Blond"
             traits["Freckling Index"] = "High Probability"
        elif "CC" in slc45: # Darker
             hair_color = "Dark Brown / Black"
        elif "GG" in slc45: # Lighter
             hair_color = "Blond / Light Brown"
             
        # Ancestry Adjustments for Hair
        if "Africa" in top_region and top_prob > 0.6:
            hair_texture = "Coiled / Curly"
            if hair_color == "Brown/Black": hair_color = "Black"
        elif "Asia" in top_region and top_prob > 0.6:
            hair_texture = "Straight / Thick"
            if hair_color == "Brown/Black": hair_color = "Black"
             
        traits["Hair Morphology"] = f"{hair_texture}, {hair_color}"
        
        # Calculate Coherence Score
        coherence_results = self._calculate_coherence(traits, top_region, top_prob)
        
        return {
            "traits": traits,
            "reliability_score": self._calculate_reliability(snp_data),
            "coherence_score": coherence_results["score"],
            "coherence_status": coherence_results["status"],
            "snps_analyzed": [k for k in snp_data.keys() if k in self.SNPS]
        }

    def _calculate_coherence(self, traits: Dict[str, str], region: str, prob: float) -> Dict[str, Any]:
        """
        Calculates Scientific Coherence Score between Genotype and Phenotype.
        """
        if prob < 0.5 or region == "Unknown":
            return {"score": 0.5, "status": "Inconclusive (Low Parsimony)"}

        score = 1.0
        
        # Rule 1: Northern Euro + Dark Skin/Eyes = Penalty (Rare but possible)
        if "Europe" in region:
            if "Dark" in traits["Dermal Classification"] or "Dark" in traits["Ocular Pigmentation"]:
                score -= 0.3 # Possible, but lowers coherence relative to population mean
        
        # Rule 2: Africa + Light Eyes/Hair = Penalty
        if "Africa" in region:
             if "Blue" in traits["Ocular Pigmentation"] or "Blond" in traits["Hair Morphology"]:
                 score -= 0.4 # Highly unusual without admixture
        
        # Rule 3: East Asia + Light Eyes = Penalty
        if "Asia" in region:
             if "Blue" in traits["Ocular Pigmentation"]:
                 score -= 0.4
                 
        # Clamp score
        score = max(0.1, min(score, 1.0))
        
        status = "High Coherence"
        if score < 0.6: status = "Phenotypic Anomaly Detected"
        elif score < 0.8: status = "Moderate Coherence"
        
        return {"score": round(score, 2), "status": status}

    def _calculate_reliability(self, snp_data: Dict[str, str]) -> float:
        """
        Calculate reliability based on presence of key SNPs.
        """
        present = sum(1 for rs in self.SNPS if rs in snp_data)
        total = len(self.SNPS)
        return round(present / total, 2)
