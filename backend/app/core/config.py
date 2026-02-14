from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    PROJECT_NAME: str = "VANTAGE-STR"
    API_V1_STR: str = "/api/v1"
    
    # Milvus Connection
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    
    # Postgres Connection
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_USER: str = "vantage"
    POSTGRES_PASSWORD: str = "vantage-pass"
    POSTGRES_DB: str = "vantage_db"
    
    # AI/DSPy Settings
    DSPY_MODEL: str = "gpt-4-turbo-preview"
    
    class Config:
        case_sensitive = True

settings = Settings()
