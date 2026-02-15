from typing import Optional
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "VANTAGE-STR"
    API_V1_STR: str = "/api/v1"
    
    # Deployment
    PORT: int = 8000
    
    # Milvus Connection
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    MILVUS_URI: Optional[str] = None
    MILVUS_TOKEN: Optional[str] = None
    
    # Postgres Connection
    DATABASE_URL: Optional[str] = None
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_USER: str = "vantage"
    POSTGRES_PASSWORD: str = "vantage-pass"
    POSTGRES_DB: str = "vantage_db"
    
    # AI/DSPy Settings
    DSPY_MODEL: str = "gpt-4-turbo-preview"
    
    # Web3 / ForensicAudit Contract
    WEB3_PROVIDER_URL: str = "http://127.0.0.1:8545"
    FORENSIC_AUDIT_CONTRACT: str = ""
    DEPLOYER_PRIVATE_KEY: str = ""
    
    # Auth Settings
    SECRET_KEY: str = "vantage-str-dev-secret-change-in-production"
    SESSION_TOKEN_TTL_MINUTES: int = 15
    
    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings()
