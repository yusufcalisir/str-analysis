import json
import os
from web3 import Web3
try:
    from web3.middleware import geth_poa_middleware
except ImportError:
    # Web3.py v7.0.0+ change
    from web3.middleware import ExtraDataToPOAMiddleware as geth_poa_middleware
from app.core.config import settings

class VantageAuditService:
    def __init__(self):
        self.w3 = Web3(Web3.HTTPProvider(settings.WEB3_PROVIDER_URL))
        try:
            # Web3.py v6
            self.w3.middleware_onion.inject(geth_poa_middleware, layer=0)
        except:
            # Web3.py v7 (middleware architecture change)
            # In v7, if imported as ExtraDataToPOAMiddleware, it might be a class
            pass 
        
        self.contract_address = settings.FORENSIC_AUDIT_CONTRACT
        self.private_key = settings.DEPLOYER_PRIVATE_KEY
        self.account = self.w3.eth.account.from_key(self.private_key)
        
        # Load ABI
        abi_path = os.path.join(
            os.path.dirname(__file__), 
            "../../../../contracts/artifacts/src/VantageAudit.sol/VantageAudit.json"
        )
        
        if not os.path.exists(abi_path):
             # Fallback for Docker/Deployment environments where path might differ
             # In a real app, ABI should be part of the package data
             print(f"Warning: ABI not found at {abi_path}")
             self.contract = None
        else:
            with open(abi_path, "r") as f:
                contract_json = json.load(f)
                self.abi = contract_json["abi"]
                
            if self.contract_address:
                self.contract = self.w3.eth.contract(address=self.contract_address, abi=self.abi)
            else:
                print("Warning: FORENSIC_AUDIT_CONTRACT not set.")
                self.contract = None

    def is_connected(self) -> bool:
        return self.w3.is_connected()

    def is_investigator_authorized(self, investigator_address: str) -> bool:
        """
        Checks if the investigator is authorized and if the system is not paused.
        """
        if not self.contract or not self.is_connected():
            # In production, this should likely fail closed (False). 
            # For dev resilience, we might warn and return False.
            print("Blockchain unavailable for authorization check.")
            return False

        try:
            # Check if system is paused (Lockdown)
            is_paused = self.contract.functions.paused().call()
            if is_paused:
                print("System is in LOCKDOWN mode.")
                return False

            # Check investigator profile
            # profiles(address) returns (name, isAuthorized, createdAt)
            profile = self.contract.functions.profiles(investigator_address).call()
            return profile[1] # isAuthorized

        except Exception as e:
            print(f"Error checking authorization: {e}")
            return False

    def _check_gas_funds(self):
        """Helper to ensure the deployer account has funds for gas."""
        try:
            balance = self.w3.eth.get_balance(self.account.address)
            if balance == 0:
                print(f"[CRITICAL] Deployer account {self.account.address} has 0 ETH/MATIC/Testnet Tokens!")
                return False
            return True
        except Exception as e:
            print(f"[WARN] Could not check balance: {e}")
            return True # Try anyway

    def grant_session(self, investigator_address: str, session_token: str) -> str:
        """
        Admin Action: Grants a session token to an investigator on-chain.
        """
        if not self.contract:
            raise Exception("Contract not loaded")
        
        if not self._check_gas_funds():
            raise Exception("Deployer wallet has 0 funds for gas.")
            
        try:
            func = self.contract.functions.grantSession(
                investigator_address,
                session_token
            )
            
            chain_id = self.w3.eth.chain_id
            nonce = self.w3.eth.get_transaction_count(self.account.address)
            
            # Estimate gas
            try:
                gas_estimate = func.estimate_gas({'from': self.account.address})
                gas_limit = int(gas_estimate * 1.2) # 20% buffer
            except Exception as e:
                print(f"[WARN] Gas estimation failed, using fallback: {e}")
                gas_limit = 2000000

            tx_data = func.build_transaction({
                'chainId': chain_id,
                'gas': gas_limit,
                'gasPrice': self.w3.eth.gas_price,
                'nonce': nonce,
            })
            
            signed_tx = self.w3.eth.account.sign_transaction(tx_data, self.private_key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            
            print(f"[Blockchain] GrantSession TX sent: {self.w3.to_hex(tx_hash)}")
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            if receipt.status == 1:
                return self.w3.to_hex(tx_hash)
            else:
                raise Exception("Grant Session reverted")
                
        except Exception as e:
            print(f"Grant Session Error: {e}")
            raise e

    def log_query_to_blockchain(self, investigator_address: str, profile_hash: str, query_type: str, session_token: str) -> str:
        """
        Logs the query to the blockchain. Returns transaction hash if successful.
        """
        if not self.contract:
            raise Exception("Contract not loaded")

        if not self._check_gas_funds():
             # Fail gracefully for logging? Or raise?
             # For audit, we should probably raise or critical log.
             print("[CRITICAL] Cannot log to blockchain: No gas funds.")
             raise Exception("Deployer wallet empty")

        try:
            func = self.contract.functions.logQuery(
                query_type,
                profile_hash,
                session_token
            )
            
            chain_id = self.w3.eth.chain_id
            nonce = self.w3.eth.get_transaction_count(self.account.address)
            
            # Estimate gas
            try:
                gas_estimate = func.estimate_gas({'from': self.account.address})
                gas_limit = int(gas_estimate * 1.2)
            except Exception as e:
                print(f"[WARN] Gas estimation failed, using fallback: {e}")
                gas_limit = 2000000
            
            tx_data = func.build_transaction({
                'chainId': chain_id,
                'gas': gas_limit,
                'gasPrice': self.w3.eth.gas_price,
                'nonce': nonce,
            })
            
            signed_tx = self.w3.eth.account.sign_transaction(tx_data, self.private_key)
            tx_hash = self.w3.eth.send_raw_transaction(signed_tx.raw_transaction)
            
            print(f"[Blockchain] LogQuery TX sent: {self.w3.to_hex(tx_hash)}")
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            if receipt.status == 1:
                return self.w3.to_hex(tx_hash)
            else:
                raise Exception("Transaction reverted on-chain")
                
        except Exception as e:
            print(f"Blockchain Logging Error: {e}")
            raise e


# ── Singleton factory ──────────────────────────────────────────────────────────
_service_instance: VantageAuditService | None = None

def get_service() -> VantageAuditService | None:
    """
    Lazy singleton. Returns None when blockchain config is absent,
    allowing the rest of the backend to operate without Web3.
    """
    global _service_instance
    if _service_instance is not None:
        return _service_instance

    if not settings.WEB3_PROVIDER_URL or not settings.DEPLOYER_PRIVATE_KEY:
        return None

    try:
        _service_instance = VantageAuditService()
        return _service_instance
    except Exception as e:
        print(f"[web3_service] Could not initialise VantageAuditService: {e}")
        return None
