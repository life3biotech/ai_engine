import life3_biotech as life3
import life3_biotech_fastapi as life3_fapi


PRED_MODEL = life3.modeling.utils.load_model(
    life3_fapi.config.SETTINGS.PRED_MODEL_PATH)
