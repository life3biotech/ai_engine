import pydantic


class Settings(pydantic.BaseSettings):

    API_NAME: str = "life3_biotech_fastapi"
    API_V1_STR: str = "/api/v1"
    LOGGER_CONFIG_PATH: str = "../conf/base/logging.yml"

    PRED_MODEL_UUID: str
    PRED_MODEL_PATH: str


SETTINGS = Settings()
