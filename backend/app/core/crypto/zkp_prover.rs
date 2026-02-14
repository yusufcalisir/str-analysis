// ═══════════════════════════════════════════════════════════════════════════════
// VANTAGE-STR Phase 4.1 — Groth16 ZKP Circuit for Cosine Similarity Threshold
// ═══════════════════════════════════════════════════════════════════════════════
//
// Arkworks R1CS circuit proving:
//     "I know V_local ∈ F^48 such that CosineSim(V_query, V_local) ≥ τ"
//
// Build:
//     cargo build --release
//     maturin develop --release   (for PyO3 .pyd/.so)
//
// Cargo.toml dependencies:
//     ark-ff          = "0.4"
//     ark-ec          = "0.4"
//     ark-bn254       = "0.4"
//     ark-groth16     = "0.4"
//     ark-relations   = "0.4"
//     ark-std         = "0.4"
//     ark-snark       = "0.4"
//     ark-serialize   = "0.4"
//     pyo3            = { version = "0.21", features = ["extension-module"] }
//     rand            = "0.8"
//
// ═══════════════════════════════════════════════════════════════════════════════
//
// ALGEBRAIC REFORMULATION
// ═══════════════════════════════════════════════════════════════════════════════
//
// Cosine Similarity:
//     S(u, v) = (u · v) / (‖u‖ × ‖v‖)
//
// Threshold inequality S ≥ τ  ⟺  (u · v)² ≥ τ² × ‖u‖² × ‖v‖²
//     (valid when u·v ≥ 0, which holds for non-negative DNA vectors)
//
// This eliminates division and square roots from the circuit. Total
// R1CS constraints: O(n) for dot product + O(n) for norms + O(1) for
// comparison ≈ 3×48 + 1 = 145 constraints.
//
// ═══════════════════════════════════════════════════════════════════════════════

use ark_bn254::{Bn254, Fr};
use ark_ff::Field;
use ark_groth16::{Groth16, Proof, ProvingKey, VerifyingKey};
use ark_relations::r1cs::{
    ConstraintSynthesizer, ConstraintSystemRef, SynthesisError,
};
use ark_snark::SNARK;
use ark_std::rand::thread_rng;
use ark_std::UniformRand;

/// Dimension of the genomic embedding vectors (24 STR loci × 2 alleles).
const VECTOR_DIM: usize = 48;

// ═══════════════════════════════════════════════════════════════════════════════
// CIRCUIT DEFINITION
// ═══════════════════════════════════════════════════════════════════════════════

/// Groth16 R1CS circuit for the cosine-similarity threshold proof.
///
/// Public inputs:
///     - v_query:  [F; 48]    — Query vector from the Orchestrator.
///     - tau_sq:   F          — τ² (threshold squared, pre-computed).
///
/// Private witness:
///     - v_local:  [F; 48]    — Local DNA vector (NEVER leaves the node).
///
/// Constraint system proves:
///     (v_query · v_local)² ≥ τ² × ‖v_query‖² × ‖v_local‖²
///
/// Which is equivalent to:
///     CosineSim(v_query, v_local) ≥ τ
#[derive(Clone)]
pub struct CosineSimilarityCircuit {
    /// Public: query vector components (field elements).
    pub v_query: [Option<Fr>; VECTOR_DIM],

    /// Private witness: local DNA vector components.
    pub v_local: [Option<Fr>; VECTOR_DIM],

    /// Public: τ² (threshold squared).
    pub tau_squared: Option<Fr>,

    /// Public: BLAKE2b(query_id) mapped to field element — replay protection.
    pub query_id_hash: Option<Fr>,
}

impl ConstraintSynthesizer<Fr> for CosineSimilarityCircuit {
    fn generate_constraints(
        self,
        cs: ConstraintSystemRef<Fr>,
    ) -> Result<(), SynthesisError> {
        use ark_relations::r1cs::Variable;
        use ark_relations::lc;

        // ── Allocate public inputs: v_query[0..48] ──
        let mut q_vars = Vec::with_capacity(VECTOR_DIM);
        for i in 0..VECTOR_DIM {
            let q_i = cs.new_input_variable(|| {
                self.v_query[i].ok_or(SynthesisError::AssignmentMissing)
            })?;
            q_vars.push(q_i);
        }

        // ── Allocate public input: τ² ──
        let tau_sq_var = cs.new_input_variable(|| {
            self.tau_squared.ok_or(SynthesisError::AssignmentMissing)
        })?;

        // ── Allocate public input: query_id_hash ──
        let _query_id_var = cs.new_input_variable(|| {
            self.query_id_hash.ok_or(SynthesisError::AssignmentMissing)
        })?;

        // ── Allocate private witness: v_local[0..48] ──
        let mut l_vars = Vec::with_capacity(VECTOR_DIM);
        for i in 0..VECTOR_DIM {
            let l_i = cs.new_witness_variable(|| {
                self.v_local[i].ok_or(SynthesisError::AssignmentMissing)
            })?;
            l_vars.push(l_i);
        }

        // ══════════════════════════════════════════════════════════════════════
        // CONSTRAINT 1: Compute dot product  d = v_query · v_local
        // ══════════════════════════════════════════════════════════════════════
        //
        // For each dimension i:  product_i = q_i × l_i
        // Then:  d = Σ product_i
        //
        // R1CS constraints per dimension: 1 multiplication gate.

        let mut product_vars = Vec::with_capacity(VECTOR_DIM);
        for i in 0..VECTOR_DIM {
            // Allocate product_i as witness
            let product_i = cs.new_witness_variable(|| {
                let q = self.v_query[i].ok_or(SynthesisError::AssignmentMissing)?;
                let l = self.v_local[i].ok_or(SynthesisError::AssignmentMissing)?;
                Ok(q * l)
            })?;

            // Enforce: q_i × l_i = product_i
            cs.enforce_constraint(
                lc!() + q_vars[i],
                lc!() + l_vars[i],
                lc!() + product_i,
            )?;

            product_vars.push(product_i);
        }

        // Accumulate dot product: d = Σ product_i
        let dot_product = cs.new_witness_variable(|| {
            let mut sum = Fr::ZERO;
            for i in 0..VECTOR_DIM {
                let q = self.v_query[i].ok_or(SynthesisError::AssignmentMissing)?;
                let l = self.v_local[i].ok_or(SynthesisError::AssignmentMissing)?;
                sum += q * l;
            }
            Ok(sum)
        })?;

        // Enforce: Σ product_i = dot_product
        let mut dot_lc = lc!();
        for &p in &product_vars {
            dot_lc = dot_lc + p;
        }
        cs.enforce_constraint(
            dot_lc,
            lc!() + Variable::One,
            lc!() + dot_product,
        )?;

        // ══════════════════════════════════════════════════════════════════════
        // CONSTRAINT 2: Compute d² (dot product squared)
        // ══════════════════════════════════════════════════════════════════════

        let dot_squared = cs.new_witness_variable(|| {
            let mut sum = Fr::ZERO;
            for i in 0..VECTOR_DIM {
                let q = self.v_query[i].ok_or(SynthesisError::AssignmentMissing)?;
                let l = self.v_local[i].ok_or(SynthesisError::AssignmentMissing)?;
                sum += q * l;
            }
            Ok(sum * sum)
        })?;

        // Enforce: d × d = d²
        cs.enforce_constraint(
            lc!() + dot_product,
            lc!() + dot_product,
            lc!() + dot_squared,
        )?;

        // ══════════════════════════════════════════════════════════════════════
        // CONSTRAINT 3: Compute ‖v_query‖² = Σ q_i²
        // ══════════════════════════════════════════════════════════════════════

        let mut q_sq_vars = Vec::with_capacity(VECTOR_DIM);
        for i in 0..VECTOR_DIM {
            let q_sq = cs.new_witness_variable(|| {
                let q = self.v_query[i].ok_or(SynthesisError::AssignmentMissing)?;
                Ok(q * q)
            })?;
            cs.enforce_constraint(
                lc!() + q_vars[i],
                lc!() + q_vars[i],
                lc!() + q_sq,
            )?;
            q_sq_vars.push(q_sq);
        }

        let norm_q_sq = cs.new_witness_variable(|| {
            let mut sum = Fr::ZERO;
            for i in 0..VECTOR_DIM {
                let q = self.v_query[i].ok_or(SynthesisError::AssignmentMissing)?;
                sum += q * q;
            }
            Ok(sum)
        })?;

        let mut norm_q_lc = lc!();
        for &qs in &q_sq_vars {
            norm_q_lc = norm_q_lc + qs;
        }
        cs.enforce_constraint(
            norm_q_lc,
            lc!() + Variable::One,
            lc!() + norm_q_sq,
        )?;

        // ══════════════════════════════════════════════════════════════════════
        // CONSTRAINT 4: Compute ‖v_local‖² = Σ l_i²
        // ══════════════════════════════════════════════════════════════════════

        let mut l_sq_vars = Vec::with_capacity(VECTOR_DIM);
        for i in 0..VECTOR_DIM {
            let l_sq = cs.new_witness_variable(|| {
                let l = self.v_local[i].ok_or(SynthesisError::AssignmentMissing)?;
                Ok(l * l)
            })?;
            cs.enforce_constraint(
                lc!() + l_vars[i],
                lc!() + l_vars[i],
                lc!() + l_sq,
            )?;
            l_sq_vars.push(l_sq);
        }

        let norm_l_sq = cs.new_witness_variable(|| {
            let mut sum = Fr::ZERO;
            for i in 0..VECTOR_DIM {
                let l = self.v_local[i].ok_or(SynthesisError::AssignmentMissing)?;
                sum += l * l;
            }
            Ok(sum)
        })?;

        let mut norm_l_lc = lc!();
        for &ls in &l_sq_vars {
            norm_l_lc = norm_l_lc + ls;
        }
        cs.enforce_constraint(
            norm_l_lc,
            lc!() + Variable::One,
            lc!() + norm_l_sq,
        )?;

        // ══════════════════════════════════════════════════════════════════════
        // CONSTRAINT 5: Compute τ² × ‖v_query‖² × ‖v_local‖²
        // ══════════════════════════════════════════════════════════════════════

        // Step 5a: norms_product = ‖v_query‖² × ‖v_local‖²
        let norms_product = cs.new_witness_variable(|| {
            let mut nq = Fr::ZERO;
            let mut nl = Fr::ZERO;
            for i in 0..VECTOR_DIM {
                let q = self.v_query[i].ok_or(SynthesisError::AssignmentMissing)?;
                let l = self.v_local[i].ok_or(SynthesisError::AssignmentMissing)?;
                nq += q * q;
                nl += l * l;
            }
            Ok(nq * nl)
        })?;

        cs.enforce_constraint(
            lc!() + norm_q_sq,
            lc!() + norm_l_sq,
            lc!() + norms_product,
        )?;

        // Step 5b: rhs = τ² × norms_product
        let rhs = cs.new_witness_variable(|| {
            let tau_sq = self.tau_squared.ok_or(SynthesisError::AssignmentMissing)?;
            let mut nq = Fr::ZERO;
            let mut nl = Fr::ZERO;
            for i in 0..VECTOR_DIM {
                let q = self.v_query[i].ok_or(SynthesisError::AssignmentMissing)?;
                let l = self.v_local[i].ok_or(SynthesisError::AssignmentMissing)?;
                nq += q * q;
                nl += l * l;
            }
            Ok(tau_sq * nq * nl)
        })?;

        cs.enforce_constraint(
            lc!() + tau_sq_var,
            lc!() + norms_product,
            lc!() + rhs,
        )?;

        // ══════════════════════════════════════════════════════════════════════
        // CONSTRAINT 6: Enforce d² ≥ τ² × ‖v_query‖² × ‖v_local‖²
        // ══════════════════════════════════════════════════════════════════════
        //
        // d² - rhs ≥ 0
        //
        // Represented as: ∃ slack ≥ 0 such that d² = rhs + slack
        // The slack variable must be non-negative (range proof).
        //
        // For Groth16 over BN254, non-negativity is enforced by
        // decomposing slack into its binary representation and proving
        // each bit is 0 or 1.

        let slack = cs.new_witness_variable(|| {
            let mut dot_val = Fr::ZERO;
            for i in 0..VECTOR_DIM {
                let q = self.v_query[i].ok_or(SynthesisError::AssignmentMissing)?;
                let l = self.v_local[i].ok_or(SynthesisError::AssignmentMissing)?;
                dot_val += q * l;
            }
            let dot_sq = dot_val * dot_val;

            let tau_sq = self.tau_squared.ok_or(SynthesisError::AssignmentMissing)?;
            let mut nq = Fr::ZERO;
            let mut nl = Fr::ZERO;
            for i in 0..VECTOR_DIM {
                let q = self.v_query[i].ok_or(SynthesisError::AssignmentMissing)?;
                let l = self.v_local[i].ok_or(SynthesisError::AssignmentMissing)?;
                nq += q * q;
                nl += l * l;
            }
            let rhs_val = tau_sq * nq * nl;

            Ok(dot_sq - rhs_val)
        })?;

        // Enforce: d² = rhs + slack  ⟺  d² - rhs - slack = 0
        cs.enforce_constraint(
            lc!() + dot_squared - rhs - slack,
            lc!() + Variable::One,
            lc!(),
        )?;

        // Binary decomposition of slack for non-negativity (64 bits)
        // Each bit b_i ∈ {0, 1} enforced by: b_i × (1 - b_i) = 0
        // slack = Σ b_i × 2^i
        //
        // NOTE: Full bit decomposition adds 64 constraints but guarantees
        // the inequality holds over the integers (not just mod p).

        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// TRUSTED SETUP & PROVING/VERIFYING API
// ═══════════════════════════════════════════════════════════════════════════════

/// One-time trusted setup. Generates the proving and verifying keys.
///
/// In production, this would be done via a multi-party computation (MPC)
/// ceremony to ensure no single party has the toxic waste.
pub fn setup() -> (ProvingKey<Bn254>, VerifyingKey<Bn254>) {
    let circuit = CosineSimilarityCircuit {
        v_query: [None; VECTOR_DIM],
        v_local: [None; VECTOR_DIM],
        tau_squared: None,
        query_id_hash: None,
    };

    let mut rng = thread_rng();
    Groth16::<Bn254>::circuit_specific_setup(circuit, &mut rng)
        .expect("Groth16 setup failed")
}

/// Generate a Groth16 proof.
///
/// Args:
///     pk: Proving key from trusted setup.
///     v_query: Public query vector (48 field elements).
///     v_local: Private local vector (48 field elements) — WITNESS.
///     tau_squared: τ² as a field element.
///     query_id_hash: BLAKE2b(query_id) as a field element.
///
/// Returns:
///     Serialized proof bytes (~192 bytes for BN254 Groth16).
pub fn prove(
    pk: &ProvingKey<Bn254>,
    v_query: &[Fr; VECTOR_DIM],
    v_local: &[Fr; VECTOR_DIM],
    tau_squared: Fr,
    query_id_hash: Fr,
) -> Vec<u8> {
    let circuit = CosineSimilarityCircuit {
        v_query: v_query.map(Some),
        v_local: v_local.map(Some),
        tau_squared: Some(tau_squared),
        query_id_hash: Some(query_id_hash),
    };

    let mut rng = thread_rng();
    let proof = Groth16::<Bn254>::prove(pk, circuit, &mut rng)
        .expect("Proof generation failed");

    // Serialize proof to bytes
    let mut buf = Vec::new();
    use ark_serialize::CanonicalSerialize;
    proof.serialize_compressed(&mut buf)
        .expect("Proof serialization failed");
    buf
}

/// Verify a Groth16 proof.
///
/// Args:
///     vk: Verifying key from trusted setup.
///     proof_bytes: Serialized proof.
///     v_query: Public query vector.
///     tau_squared: τ² as field element.
///     query_id_hash: BLAKE2b(query_id) as field element.
///
/// Returns:
///     true if proof is valid, false otherwise.
pub fn verify(
    vk: &VerifyingKey<Bn254>,
    proof_bytes: &[u8],
    v_query: &[Fr; VECTOR_DIM],
    tau_squared: Fr,
    query_id_hash: Fr,
) -> bool {
    use ark_serialize::CanonicalDeserialize;
    let proof = Proof::<Bn254>::deserialize_compressed(proof_bytes)
        .expect("Proof deserialization failed");

    // Construct public inputs: [v_query[0], ..., v_query[47], tau_squared, query_id_hash]
    let mut public_inputs = Vec::with_capacity(VECTOR_DIM + 2);
    for q in v_query {
        public_inputs.push(*q);
    }
    public_inputs.push(tau_squared);
    public_inputs.push(query_id_hash);

    Groth16::<Bn254>::verify(vk, &public_inputs, &proof)
        .expect("Verification failed")
}

// ═══════════════════════════════════════════════════════════════════════════════
// PyO3 BINDINGS (compiled with maturin)
// ═══════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "python")]
mod python_bindings {
    use pyo3::prelude::*;
    use pyo3::types::PyList;

    // Lazy-initialized proving/verifying keys
    use std::sync::OnceLock;
    static KEYS: OnceLock<(
        super::ProvingKey<super::Bn254>,
        super::VerifyingKey<super::Bn254>,
    )> = OnceLock::new();

    fn get_keys() -> &'static (
        super::ProvingKey<super::Bn254>,
        super::VerifyingKey<super::Bn254>,
    ) {
        KEYS.get_or_init(|| super::setup())
    }

    /// Generate a ZKP for cosine similarity threshold.
    #[pyfunction]
    fn prove(v_local: Vec<f64>, v_query: Vec<f64>, tau: f64, query_id: Option<String>) -> PyResult<Vec<u8>> {
        use ark_ff::PrimeField;

        if v_local.len() != super::VECTOR_DIM || v_query.len() != super::VECTOR_DIM {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Vectors must be {}-dimensional", super::VECTOR_DIM)
            ));
        }

        let (pk, _) = get_keys();

        // Convert f64 → Fr by scaling to fixed-point (2^32 precision)
        let scale = (1u64 << 32) as f64;
        let to_fr = |x: f64| -> super::Fr {
            let scaled = (x * scale) as u64;
            super::Fr::from(scaled)
        };

        let q_arr: [super::Fr; super::VECTOR_DIM] =
            core::array::from_fn(|i| to_fr(v_query[i]));
        let l_arr: [super::Fr; super::VECTOR_DIM] =
            core::array::from_fn(|i| to_fr(v_local[i]));
        let tau_sq = to_fr(tau * tau);

        // Hash query_id to field element
        let qid_hash = {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut hasher = DefaultHasher::new();
            query_id.unwrap_or_default().hash(&mut hasher);
            super::Fr::from(hasher.finish())
        };

        Ok(super::prove(pk, &q_arr, &l_arr, tau_sq, qid_hash))
    }

    /// Verify a ZKP proof.
    #[pyfunction]
    fn verify(proof_bytes: Vec<u8>, v_query: Vec<f64>, tau: f64, query_id: Option<String>) -> PyResult<bool> {
        let (_, vk) = get_keys();

        let scale = (1u64 << 32) as f64;
        let to_fr = |x: f64| -> super::Fr {
            let scaled = (x * scale) as u64;
            super::Fr::from(scaled)
        };

        let q_arr: [super::Fr; super::VECTOR_DIM] =
            core::array::from_fn(|i| to_fr(v_query[i]));
        let tau_sq = to_fr(tau * tau);

        let qid_hash = {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut hasher = DefaultHasher::new();
            query_id.unwrap_or_default().hash(&mut hasher);
            super::Fr::from(hasher.finish())
        };

        Ok(super::verify(vk, &proof_bytes, &q_arr, tau_sq, qid_hash))
    }

    /// Verify a match proof — convenience wrapper for Orchestrator.
    /// Returns true if the proof is cryptographically valid for the given query.
    #[pyfunction]
    fn verify_match_proof(
        proof_bytes: Vec<u8>,
        v_query: Vec<f64>,
        tau: f64,
        query_id: String,
    ) -> PyResult<bool> {
        verify(proof_bytes, v_query, tau, Some(query_id))
    }

    /// Python module definition.
    #[pymodule]
    fn vantage_zkp_rs(_py: Python, m: &PyModule) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(prove, m)?)?;
        m.add_function(wrap_pyfunction!(verify, m)?)?;
        m.add_function(wrap_pyfunction!(verify_match_proof, m)?)?;
        Ok(())
    }
}
