from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    GOOGLE_API_KEY: SecretStr
    LLAMA_CLOUD_API_KEY: SecretStr
    TAVILY_API_KEY: SecretStr

    EMBEDDINGS_MODEL: str
    AUDIO_EXTRACTOR_LLM_MODEL_NAME: str
    MOCKUP_CRAFTER_LLM_MODEL_NAME: str
    WEB_SEARCHER_LLM_MODEL_NAME: str
    ENRICHER_LLM_MODEL_NAME: str
    SETUP_CRAFTER_MODEL_NAME: str
    CONVERSATION_LLM_MODEL_NAME: str

    SAMPLE_DIRECTORY: str
    RAW_DOCS_DIRECTORY: str
    PROCESSED_DOCS_DIRECTORY: str

    PEDAL_NAME: str

    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8')


settings = Settings()
