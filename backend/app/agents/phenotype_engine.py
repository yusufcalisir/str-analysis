from typing import Dict, Any, List

class PhenotypeEngine:
    """
    Deterministic SNP-to-Phenotype mapping engine for forensic traits.
    Focuses on HERC2, MC1R, SLC24A5, SLC45A2.
    """
    
    # Major Forensic SNPs with Strict Mapping Definitions
    SNP_KNOWLEDGE_BASE = {
        "rs12913832": {
            "gene": "HERC2",
            "trait": "Eye Color",
            "mappings": {
                "AA": "Brown Eyes",
                "AG": "Green/Hazel Eyes",
                "GA": "Green/Hazel Eyes",
                "GG": "Blue Eyes"
            }
        },
        "rs1426654": {
            "gene": "SLC24A5",
            "trait": "Skin Tone",
            "mappings": {
                "AA": "Dark Skin",
                "AG": "Intermediate Skin",
                "GA": "Intermediate Skin",
                "GG": "Light Skin"
            }
        },
        "rs1805007": {
            "gene": "MC1R",
            "trait": "Hair Color",
            "mappings": {
                "TT": "Red Hair",
                "CC": "Non-Red Hair",
                "CT": "Carrier (Non-Red)",
                "TC": "Carrier (Non-Red)"
            }
        },
        "rs16891982": {
            "gene": "SLC45A2",
            "trait": "Hair Color",
            "mappings": {
                "GG": "Light Hair/Skin",
                "CC": "Dark Hair/Skin",
                "CG": "Intermediate",
                "GC": "Intermediate"
            }
        }
    }
    
    def predict_phenotype(self, snp_data: Dict[str, str], ancestry_context: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Predicts physical traits based ONLY on strict SNP mappings.
        Returns 'Insufficient Data' if key markers are missing.
        """
        traits = {
            "Ocular Pigmentation": {"value": "Insufficient Data", "source": []},
            "Dermal Classification": {"value": "Insufficient Data", "source": []},
            "Hair Morphology": {"value": "Insufficient Data", "source": []},
        }
        
        # 1. Eye Color (HERC2 rs12913832)
        # ==========================================
        snp = "rs12913832"
        if snp in snp_data:
            genotype = snp_data[snp]
            mapping = self.SNP_KNOWLEDGE_BASE[snp]["mappings"]
            val = mapping.get(genotype, "Unknown Genotype")
            traits["Ocular Pigmentation"] = {
                "value": val, 
                "source": [f"{snp} ({genotype})"]
            }
            
        # 2. Skin Tone (SLC24A5 rs1426654)
        # ==========================================
        snp = "rs1426654"
        if snp in snp_data:
            genotype = snp_data[snp]
            mapping = self.SNP_KNOWLEDGE_BASE[snp]["mappings"]
            val = mapping.get(genotype, "Unknown Genotype")
            traits["Dermal Classification"] = {
                "value": val, 
                "source": [f"{snp} ({genotype})"]
            }

        # 3. Hair Color (MC1R + SLC45A2)
        # ==========================================
        sources = []
        hair_color_guesses = []
        
        # Check MC1R for Red Hair
        if "rs1805007" in snp_data:
            g = snp_data["rs1805007"]
            m = self.SNP_KNOWLEDGE_BASE["rs1805007"]["mappings"].get(g, "Unknown")
            sources.append(f"rs1805007 ({g})")
            if "Red" in m and "Non-" not in m:
                hair_color_guesses.append("Red")
        
        # Check SLC45A2 for Light/Dark
        if "rs16891982" in snp_data:
            g = snp_data["rs16891982"]
            m = self.SNP_KNOWLEDGE_BASE["rs16891982"]["mappings"].get(g, "Unknown")
            sources.append(f"rs16891982 ({g})")
            if "Light" in m: hair_color_guesses.append("Blond/Light Brown")
            elif "Dark" in m: hair_color_guesses.append("Dark Brown/Black")
            
        if hair_color_guesses:
            # Simple priority: Red overrides others if present
            final_hair = "Red" if "Red" in hair_color_guesses else hair_color_guesses[0]
            traits["Hair Morphology"] = {
                "value": final_hair, 
                "source": sources
            }

        # Calculate Reliability
        reliability = self._calculate_reliability(snp_data)

        # Flatten for consistency but keep metadata
        final_report = {}
        trait_sources = {}
        
        for k, v in traits.items():
            final_report[k] = v["value"]
            trait_sources[k] = v["source"]

        return {
            "traits": final_report,
            "trait_sources": trait_sources, # New field for UI transparency
            "reliability_score": reliability,
            "coherence_score": 1.0 if reliability > 0.5 else 0.0, # Placeholder
            "coherence_status": "Verified via SNP" if reliability > 0.5 else "Low Data",
            "snps_analyzed": [k for k in snp_data.keys() if k in self.SNP_KNOWLEDGE_BASE]
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
        present = sum(1 for rs in self.SNP_KNOWLEDGE_BASE if rs in snp_data)
        total = len(self.SNP_KNOWLEDGE_BASE)
        return round(present / total, 2)
