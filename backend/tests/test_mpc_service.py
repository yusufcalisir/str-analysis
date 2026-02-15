
import pytest
from backend.app.services.mpc_service import PrimeField, ShamirSecretSharing, MPCKinshipEngine

# ═══════════════════════════════════════════════════════════════════════════════
# PRIME FIELD TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_prime_field_add():
    assert PrimeField.add(10, 20) == 30
    # Test modulo wrap-around (using small prime logic for test conceptualization)
    # PRIME is large, so standard addition holds for small inputs.

def test_prime_field_sub():
    assert PrimeField.sub(30, 10) == 20
    # Test negative result handling in modular arithmetic
    # (0 - 1) % P should be P - 1
    p = 2147483647
    assert PrimeField.sub(0, 1) == p - 1

def test_prime_field_mul():
    assert PrimeField.mul(10, 10) == 100

def test_prime_field_inv():
    # a * a^-1 = 1 (mod p)
    a = 12345
    inv_a = PrimeField.inv(a)
    assert PrimeField.mul(a, inv_a) == 1

# ═══════════════════════════════════════════════════════════════════════════════
# SHAMIR SECRETS SHARING TESTS
# ═══════════════════════════════════════════════════════════════════════════════

def test_sss_share_generation_and_reconstruction():
    secret = 42
    n = 5  # Total shares
    t = 3  # Threshold
    
    shares = ShamirSecretSharing.generate_shares(secret, n, t)
    assert len(shares) == n
    
    # Reconstruct with all shares
    reconstructed = ShamirSecretSharing.reconstruct_secret(shares)
    assert reconstructed == secret

def test_sss_threshold_property():
    secret = 999
    n = 5
    t = 3
    shares = ShamirSecretSharing.generate_shares(secret, n, t)
    
    # 1. Reconstruct with t shares (Should work)
    subset_t = shares[:t]
    assert ShamirSecretSharing.reconstruct_secret(subset_t) == secret
    
    # 2. Reconstruct with t+1 shares (Should work)
    subset_t_plus = shares[:t+1]
    assert ShamirSecretSharing.reconstruct_secret(subset_t_plus) == secret
    
    # 3. Reconstruct with t-1 shares (Should fail / produce garbage)
    # Note: It usually produces a valid integer in the field, just not the secret.
    subset_t_minus = shares[:t-1]
    # With only 2 points for a degree 2 polynomial (needs 3), linear interpolation
    # finds a line, but the constant term (secret) will be wrong unless poly was linear.
    # Generally, unconstrained interpolation yields a result, but it won't be 'secret'.
    # Because our polynomial is random, it's highly unlikely to match.
    assert ShamirSecretSharing.reconstruct_secret(subset_t_minus) != secret

pass

# ═══════════════════════════════════════════════════════════════════════════════
# MPC HOMOMORPHIC PROPERTIES
# ═══════════════════════════════════════════════════════════════════════════════

def test_sss_homomorphic_addition():
    """
    Test [A] + [B] = [A + B]
    """
    secret_a = 100
    secret_b = 200
    n = 5
    t = 3
    
    shares_a = ShamirSecretSharing.generate_shares(secret_a, n, t)
    shares_b = ShamirSecretSharing.generate_shares(secret_b, n, t)
    
    # Sum shares locally
    shares_sum = []
    for i in range(n):
        x_a, y_a = shares_a[i]
        x_b, y_b = shares_b[i]
        assert x_a == x_b # indices must match
        
        # y_sum = y_a + y_b (mod p)
        y_sum = PrimeField.add(y_a, y_b)
        shares_sum.append((x_a, y_sum))
        
    # Reconstruct sum
    reconstructed_sum = ShamirSecretSharing.reconstruct_secret(shares_sum[:t])
    assert reconstructed_sum == (secret_a + secret_b)

def test_sss_homomorphic_subtraction():
    """
    Test [A] - [B] = [A - B]  (Used for Equality Check: A - B = 0)
    """
    secret_a = 500
    secret_b = 500 # Equal secrets
    n = 5
    t = 3
    
    shares_a = ShamirSecretSharing.generate_shares(secret_a, n, t)
    shares_b = ShamirSecretSharing.generate_shares(secret_b, n, t)
    
    # Subtract shares locally
    shares_diff = []
    for i in range(n):
        x_a, y_a = shares_a[i]
        _, y_b = shares_b[i]
        
        # y_diff = y_a - y_b (mod p)
        y_diff = PrimeField.sub(y_a, y_b)
        shares_diff.append((x_a, y_diff))
        
    # Reconstruct difference
    reconstructed_diff = ShamirSecretSharing.reconstruct_secret(shares_diff[:t])
    assert reconstructed_diff == 0 # (500 - 500)

if __name__ == "__main__":
    test_prime_field_add()
    test_prime_field_sub()
    test_prime_field_mul()
    test_prime_field_inv()
    test_sss_share_generation_and_reconstruction()
    test_sss_threshold_property()
    test_sss_homomorphic_addition()
    test_sss_homomorphic_subtraction()
    print("All MPC tests passed!")
