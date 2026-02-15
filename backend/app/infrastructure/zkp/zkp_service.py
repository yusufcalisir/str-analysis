import json
import logging
import subprocess
import os
from typing import Dict, Any, List

from app.schemas.zkp import ZKPayload
from app.core.config import settings

logger = logging.getLogger(__name__)

class ZKPService:
    def __init__(self, verification_key_path: str = "verification_key.json"):
        # Ensure path is absolute or relative to project root
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.vk_path = os.path.join(base_dir, verification_key_path)
        
        if not os.path.exists(self.vk_path):
            logger.warning(f"Verification Key not found at {self.vk_path}. ZKP verification will fail.")

    def verify_proof(self, payload: ZKPayload) -> bool:
        """
        Verifies a Groth16 proof using snarkjs via CLI.
        Payload contains the proof and public signals.
        """
        try:
            # Create temporary files for proof and public signals
            import tempfile
            
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as proof_file:
                json.dump(payload.proof.dict(), proof_file)
                proof_path = proof_file.name

            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as public_file:
                json.dump(payload.public_signals, public_file)
                public_path = public_file.name

            # Command: snarkjs groth16 verify verification_key.json public.json proof.json
            # Assumes npx and snarkjs are available in path
            cmd = ["npx", "snarkjs", "groth16", "verify", self.vk_path, public_path, proof_path]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Cleanup temp files
            os.unlink(proof_path)
            os.unlink(public_path)

            if result.returncode == 0 and "OK" in result.stdout:
                logger.info("ZKP Verification Successful")
                return True
            else:
                logger.error(f"ZKP Verification Failed: {result.stderr or result.stdout}")
                return False

        except Exception as e:
            logger.error(f"Error during ZKP verification: {str(e)}")
            return False

# Singleton instance
zkp_service = ZKPService()
