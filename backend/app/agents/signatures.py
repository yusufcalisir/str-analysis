"""
DSPy Signature Definitions for VANTAGE-STR Forensic Validation.

Signatures define the input/output contract for DSPy modules. Each field
carries a description that guides the language model's reasoning when
evaluating genomic profiles for biological plausibility.

The BiometricValidationSignature is the core contract used by the
ForensicValidator module to intercept and assess incoming STR profiles
before they enter the vector database.
"""

import dspy


class BiometricValidationSignature(dspy.Signature):
    """
    Assess the biological plausibility and integrity of a forensic STR profile.

    You are an expert forensic geneticist reviewing Short Tandem Repeat (STR)
    data for a decentralized law enforcement network. Your task is to evaluate
    whether the submitted genomic profile is biologically valid, internally
    consistent, and free from signs of data manipulation or poisoning.

    Consider allele repeat count ranges, locus completeness against the CODIS
    and European Standard Set panels, and the statistical rarity of allele
    combinations within the declared population context.
    """

    # ── Inputs ──
    genomic_data: str = dspy.InputField(
        desc=(
            "Serialized STR marker data. Format: 'MARKER_NAME: allele_1/allele_2; ...' "
            "Example: 'D3S1358: 14.0/15.0; TH01: 9.3/6.0; FGA: 22.0/24.0'. "
            "Each marker represents a forensic STR locus with two allele repeat counts."
        )
    )
    population_context: str = dspy.InputField(
        desc=(
            "Metadata about the originating node and population context. "
            "Includes: node_id (agency/country), declared ethnic background if available, "
            "and regional allele frequency baseline. Used to assess whether allele "
            "combinations are statistically plausible for the declared population."
        )
    )

    # ── Outputs ──
    validity_score: float = dspy.OutputField(
        desc=(
            "A float between 0.0 and 1.0 representing overall biological plausibility. "
            "1.0 = unambiguously valid human STR profile. "
            "0.0 = clearly fabricated or corrupted data. "
            "Scores below 0.85 trigger quarantine protocol."
        )
    )
    anomaly_report: str = dspy.OutputField(
        desc=(
            "Detailed textual report of biological inconsistencies found. "
            "Must enumerate each anomaly with: (1) the affected locus, "
            "(2) the observed values, (3) the expected range, and "
            "(4) the forensic significance. Return 'No anomalies detected.' if clean."
        )
    )
    is_poisoned: bool = dspy.OutputField(
        desc=(
            "Boolean flag for suspected data poisoning or adversarial injection. "
            "True if the profile exhibits patterns consistent with: "
            "(a) synthetic allele generation, (b) duplication of known reference profiles, "
            "(c) statistically impossible allele combinations across multiple loci, or "
            "(d) values outside the human biological range. False otherwise."
        )
    )


class AnomalyClassificationSignature(dspy.Signature):
    """
    Classify a specific anomaly detected during STR profile validation.

    Given a single anomalous locus observation, determine the most likely
    cause: measurement error, rare biological variant, or deliberate tampering.
    """

    locus_name: str = dspy.InputField(desc="Name of the STR locus (e.g., 'D3S1358').")
    observed_alleles: str = dspy.InputField(desc="Observed allele values, e.g., '14.0/52.0'.")
    expected_range: str = dspy.InputField(desc="Expected allele range for this locus, e.g., '8-20'.")
    population_frequency: str = dspy.InputField(
        desc="Population frequency data for the observed allele combination, if available."
    )

    classification: str = dspy.OutputField(
        desc=(
            "One of: 'MEASUREMENT_ERROR', 'RARE_VARIANT', 'TAMPERING', 'UNKNOWN'. "
            "Must be accompanied by confidence level and reasoning."
        )
    )
    confidence: float = dspy.OutputField(desc="Confidence in the classification, 0.0 to 1.0.")
    recommendation: str = dspy.OutputField(
        desc="Recommended action: 'ACCEPT', 'MANUAL_REVIEW', 'QUARANTINE', or 'REJECT'."
    )


class PhenotypeAnalysisSignature(dspy.Signature):
    """
    Analyze SNP genotype data and produce a forensic phenotype prediction report.

    You are a forensic geneticist specializing in externally visible
    characteristics (EVCs) prediction using the HIrisPlex-S model. Given raw
    SNP genotypes and pre-computed trait probability scores from the
    PhenotypePredictor engine, provide a comprehensive analysis report
    explaining the biological reasoning behind each phenotype prediction.

    For each trait (eye color, hair color, skin pigmentation), explain:
    1. Which SNPs contributed to the prediction and their dosage effects.
    2. The molecular mechanism (e.g., OCA2 regulation by HERC2 for eye color).
    3. Any epistatic interactions between loci.
    4. Population-specific considerations affecting prediction accuracy.
    """

    # ── Inputs ──
    snp_data: str = dspy.InputField(
        desc=(
            "Serialized SNP genotype data. Format: 'rsID: genotype; rsID: genotype; ...' "
            "Example: 'rs12913832: GG; rs16891982: CG; rs1805007: CC'. "
            "Each entry is a Single Nucleotide Polymorphism with its diploid genotype."
        )
    )
    trait_probabilities: str = dspy.InputField(
        desc=(
            "Pre-computed trait probability scores from the HIrisPlex-S engine. "
            "Format: JSON string containing eye_color, hair_color, and skin_color "
            "probability distributions. These serve as the quantitative baseline "
            "for your biological reasoning."
        )
    )
    population_context: str = dspy.InputField(
        desc=(
            "Population and node metadata for ancestry-aware analysis. "
            "Includes originating node ID, estimated biogeographic ancestry, "
            "and any available population frequency data for the submitted SNPs."
        )
    )
    biological_facts: str = dspy.InputField(
        desc=(
            "Strictly verified phenotype traits derived from deterministic SNP mappings. "
            "Example: 'Ocular Pigmentation: Blue Eyes (Source: rs12913832 GG)'. "
            "You MUST align your reasoning with these facts. Do not halllucinate or deviate "
            "from these biologically confirmed traits."
        )
    )

    # ── Outputs ──
    phenotype_report: str = dspy.OutputField(
        desc=(
            "A comprehensive phenotype analysis report in JSON format containing: "
            "1) eye_color_analysis: {prediction, probability, biological_reasoning} "
            "2) hair_color_analysis: {prediction, probability, mc1r_status, biological_reasoning} "
            "3) skin_tone_analysis: {prediction, melanin_index, biological_reasoning} "
            "4) ancestry_assessment: {primary_ancestry, confidence, markers_used} "
            "5) overall_confidence: float (0-1) indicating reliability of predictions "
            "6) forensic_caveats: list of limitations relevant to this specific profile. "
            "Each biological_reasoning must reference specific gene names, molecular "
            "pathways, and cite the relevant allele effects observed in the data."
        )
    )

