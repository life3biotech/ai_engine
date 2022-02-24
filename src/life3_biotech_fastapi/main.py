import logging
import fastapi
from fastapi.middleware.cors import CORSMiddleware

import life3_biotech as life3
import life3_biotech_fastapi as life3_fapi


LOGGER = logging.getLogger(__name__)
LOGGER.info("Setting up logging configuration.")
life3.general_utils.setup_logging(
    logging_config_path=life3_fapi.config.SETTINGS.LOGGER_CONFIG_PATH)

API_V1_STR = life3_fapi.config.SETTINGS.API_V1_STR
APP = fastapi.FastAPI(
    title=life3_fapi.config.SETTINGS.API_NAME,
    openapi_url=f"{API_V1_STR}/openapi.json")
API_ROUTER = fastapi.APIRouter()
API_ROUTER.include_router(
    life3_fapi.v1.routers.model.ROUTER, prefix="/model", tags=["model"])
APP.include_router(
    API_ROUTER, prefix=life3_fapi.config.SETTINGS.API_V1_STR)

ORIGINS = ["*"]

APP.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])
