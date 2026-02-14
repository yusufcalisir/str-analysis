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
    
    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings()
