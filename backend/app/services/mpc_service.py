import secrets
from typing import List, Tuple, Dict, Any
from functools import reduce
from web3 import Web3
from backend.app.infrastructure.blockchain.web3_service import Web3Service

# ═══════════════════════════════════════════════════════════════════════════════
# PRIME FIELD ARITHMETIC
# ═══════════════════════════════════════════════════════════════════════════════

# A large prime number for the finite field (Mersenne Prime 2^127 - 1 is a good candidate)
# For simplicity in this demo, we use a smaller but safe prime > 100 (max allele count)
# In production, use a cryptographic large prime (e.g., specific curve order).
# Here: 2^31 - 1 = 2147483647 (Mersenne 31)
PRIME = 2147483647


class PrimeField:
    """
    Finite Field arithmetic modulo a large prime.
    """

    @staticmethod
    def add(a: int, b: int) -> int:
        return (a + b) % PRIME

    @staticmethod
    def sub(a: int, b: int) -> int:
        return (a - b) % PRIME

    @staticmethod
    def mul(a: int, b: int) -> int:
        return (a * b) % PRIME

    @staticmethod
    def inv(n: int) -> int:
        """Modular inverse using extended Euclidean algorithm (via pow)."""
        return pow(n, PRIME - 2, PRIME)

    @staticmethod
    def div(a: int, b: int) -> int:
        return PrimeField.mul(a, PrimeField.inv(b))

    @staticmethod
    def eval_poly(poly: List[int], x: int) -> int:
        """Evaluates polynomial f(x) at x using Horner's method."""
        result = 0
        for coeff in reversed(poly):
            result = PrimeField.add(PrimeField.mul(result, x), coeff)
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# SHAMIR'S SECRET SHARING (SSS)
# ═══════════════════════════════════════════════════════════════════════════════

class ShamirSecretSharing:
    @staticmethod
    def generate_polynomial(secret: int, degree: int) -> List[int]:
        """
        Generates a random polynomial of given degree where f(0) = secret.
        Returns coefficients [a0, a1, ..., at-1].
        """
        coefficients = [secret] + [secrets.randbelow(PRIME) for _ in range(degree)]
        return coefficients

    @staticmethod
    def generate_shares(secret: int, n: int, t: int) -> List[Tuple[int, int]]:
        """
        Splits a secret into n shares with threshold t.
        Returns [(x1, y1), (x2, y2), ..., (xn, yn)].
        """
        if t > n:
            raise ValueError("Threshold t must be <= Total shares n")
            
        degree = t - 1
        poly = ShamirSecretSharing.generate_polynomial(secret, degree)
        
        shares = []
        for i in range(1, n + 1):
            x = i
            y = PrimeField.eval_poly(poly, x)
            shares.append((x, y))
            
        return shares

    @staticmethod
    def reconstruct_secret(shares: List[Tuple[int, int]]) -> int:
        """
        Reconstructs the secret f(0) using Lagrange interpolation.
        """
        # We need at least t shares, but we take whatever is given (assuming >= t)
        if not shares:
            return 0
            
        def lagrange_basis(j: int, x_val: int) -> int:
            numerator = 1
            denominator = 1
            xj, _ = shares[j]
            
            for m, (xm, _) in enumerate(shares):
                if m == j:
                    continue
                    
                # We want L_j(0), so x = 0
                # Basis term: (0 - xm) / (xj - xm)
                numerator = PrimeField.mul(numerator, PrimeField.sub(0, xm))
                denominator = PrimeField.mul(denominator, PrimeField.sub(xj, xm))
                
            return PrimeField.div(numerator, denominator)

        secret = 0
        for j, (_, yj) in enumerate(shares):
            basis = lagrange_basis(j, 0)
            term = PrimeField.mul(yj, basis)
            secret = PrimeField.add(secret, term)
            
        return secret


# ═══════════════════════════════════════════════════════════════════════════════
# MPC KINSHIP ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class MPCKinshipEngine:
    def __init__(self, web3_service: Web3Service):
        self.web3_service = web3_service
        # Mapping for allele vectorization (simplified)
        self.MAX_BUFFER = 1000  # Offset to ensure positive differences in Z_p for comparisons

    def vectorize_profile(self, str_markers: Dict[str, Any]) -> Dict[str, List[int]]:
        """
        Converts STR alleles to integer vectors mapped to Z_p.
        Floats are scaled to integers (x10).
        """
        vectorized = {}
        for marker, data in str_markers.items():
            # Example: allele 9.3 -> 93
            a1 = int(float(data['allele_1']) * 10)
            a2 = int(float(data['allele_2']) * 10)
            vectorized[marker] = [a1, a2]
        return vectorized

    async def secure_kinship_session(self, profile_a_id: str, vector_a: Dict[str, List[int]], 
                                   profile_b_id: str, vector_b: Dict[str, List[int]]) -> float:
        """
        Executes a secure MPC session to compute kinship coefficient.
        
        Protocol:
        1. Encrypt alleles (SSS).
        2. Compute IBD probabilities per locus using homomorphic subtraction.
        3. Aggregate IBD scores.
        4. Reconstruct final kinship coefficient.
        5. Log proof to blockchain.
        """
        
        # 1. Log Session Start (Blockchain Audit)
        session_id = secrets.token_hex(16)
        print(f"[MPC] Starting Session: {session_id}")
        
        shared_markers = set(vector_a.keys()) & set(vector_b.keys())
        total_ibd_score = 0
        total_loci = 0
        
        # 2. MPC Protocol Execution
        # N=5 (nodes), T=3 (threshold)
        N = 5
        T = 3
        
        for marker in shared_markers:
            alleles_a = vector_a[marker]
            alleles_b = vector_b[marker]
            
            # Secure IBD Calculation (Simulated)
            # Check for shared alleles (Identity By State - IBS)
            # In a real IBD calc, we'd also use population frequencies.
            # Here: IBS is a proxy for IBD for demo purposes.
            # Score: 2 shared = 1.0, 1 shared = 0.5, 0 shared = 0.0
            
            shared_count = 0
            for val_a in alleles_a:
                match_found = False
                for val_b in alleles_b:
                    # Alice generates shares for val_a
                    shares_a = ShamirSecretSharing.generate_shares(val_a, N, T)
                    # Bob generates shares for val_b
                    shares_b = ShamirSecretSharing.generate_shares(val_b, N, T)
                    
                    # --- NETWORK LAYER & HOMOMORPHIC SUBTRACTION ---
                    diff_shares = []
                    for i in range(N):
                        idx_a, y_a = shares_a[i]
                        _, y_b = shares_b[i]
                        y_diff = PrimeField.sub(y_a, y_b)
                        diff_shares.append((idx_a, y_diff))
                    
                    # --- RECONSTRUCTION ---
                    subset_shares = diff_shares[:T]
                    difference = ShamirSecretSharing.reconstruct_secret(subset_shares)
                    
                    if difference == 0:
                        match_found = True
                        break 
                
                if match_found:
                    shared_count += 1
            
            # IBS Score for this locus
            # Homozygote handling: if a=[10,10], b=[10,12], shared=1? 
            # Simplified: IBS = shared_count / 2
            ibd_prob = shared_count / 2.0
            total_ibd_score += ibd_prob
            total_loci += 1
                
        # 3. Compute Kinship Coefficient
        kinship_coefficient = 0.0
        if total_loci > 0:
            # Normalized Kinship: sum(IBD) / total_loci 
            # Note: This is a simplified "genetic similarity" score.
            # Real Kinship Coefficient (Phi) requires specific IBD0/IBD1/IBD2 weights.
            # For demo, we treat this as "Genetic Relatedness".
            # Parent-Child: ~0.5, Siblings: ~0.5, Half-Sib: ~0.25
            kinship_coefficient = total_ibd_score / total_loci / 2.0 # Divide by 2 to map 100% match to 0.5 (self)?? 
            # Actually, self-match is 1.0 similarity, but phi=0.5. 
            # Let's return raw similarity (0-1) for now, interpretable as "Shared DNA %".
            kinship_coefficient = total_ibd_score / total_loci

        # 4. Log Session to Blockchain
        try:
            # Relationship Inference
            relationship = "UNRELATED"
            if kinship_coefficient > 0.45: relationship = "PARENT_CHILD" # or SIBLING
            elif kinship_coefficient > 0.20: relationship = "HALF_SIBLING" # or UNCLE/AUNT
            elif kinship_coefficient > 0.10: relationship = "FIRST_COUSIN"
            
            # Hash the result for integrity
            result_hash = Web3.keccak(text=f"{session_id}:{kinship_coefficient}").hex()
            
            # Log to chain
            if self.web3_service and self.web3_service.is_connected():
                 tx_hash = self.web3_service.log_mpc_result(
                     session_id=session_id,
                     result_hash=result_hash,
                     relationship_type=relationship,
                     kinship_percent=kinship_coefficient
                 )
                 print(f"[MPC] Proof logged to blockchain: {tx_hash}")
        except Exception as e:
            print(f"[MPC] Blockchain logging failed: {e}")
        
        return kinship_coefficient
