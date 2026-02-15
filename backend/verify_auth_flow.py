import sys
import os

# Add backend to path
sys.path.append(os.getcwd())

from app.infrastructure.blockchain.auth_service import generate_session_token, validate_session_token
from app.main import AnalysisRequest
from pydantic import ValidationError

def test_auth_flow():
    print("--- Testing Auth Service ---")
    addr = "0x1234567890123456789012345678901234567890"
    
    # 1. Generate
    session = generate_session_token(addr)
    token = session["token"]
    print(f"Token generated: {token[:30]}...")
    
    # 2. Validate
    recovered_addr = validate_session_token(token)
    if recovered_addr == addr:
        print("✅ Token validation passed")
    else:
        print(f"❌ Token validation failed: got {recovered_addr}, expected {addr}")
        
    # 3. Schema Enforcement
    print("\n--- Testing Schema Enforcement ---")
    try:
        AnalysisRequest(profile_id="test-profile")
        print("❌ Schema failed to enforce blockchain_token")
    except ValidationError:
        print("✅ AnalysisRequest correctly rejected missing blockchain_token")
        
    try:
        req = AnalysisRequest(profile_id="test-profile", blockchain_token=token)
        print("✅ AnalysisRequest accepted valid payload")
    except ValidationError as e:
        print(f"❌ AnalysisRequest rejected valid payload: {e}")

if __name__ == "__main__":
    test_auth_flow()
