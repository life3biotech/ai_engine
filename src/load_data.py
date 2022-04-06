import os
import logging
import pathlib
import hydra
from pconst import const

# from . import life3_biotech as life3
import life3_biotech as life3


@hydra.main(config_path="../conf/base", config_name="pipelines.yml")
def main(args):
    """Main program to read in raw data files and process them."""

    logger = logging.getLogger(__name__)
    logger.info("Setting up logging configuration.")
    logger_config_path = os.path.join(
        hydra.utils.get_original_cwd(), "conf/base/logging.yml"
    )
    life3.general_utils.setup_logging(logger_config_path)

    pipeline_conf = life3.config.PipelineConfig(args, logger)

    if const.LOAD_DATA:
        run_data_pipeline(logger)

def run_data_pipeline(logger):
    preprocessor = life3.data_prep.preprocess.Preprocessor(logger)
    preprocessor.generate_image_tiles() # to be implemented
    df = preprocessor.preprocess_annotations()
    X_train, y_train, X_test, y_test, X_val, y_val = preprocessor.split_data(
        df, test_size=const.TEST_SIZE, val_size=const.VAL_SIZE)

    if 'efficientdet' in const.MODELS:
        ed_pipeline = life3.data_prep.preprocess_efficientdet.EfficientDetPipeline()
        # generate annotations for each dataset (train, val, test)
        ed_pipeline.generate_annotations(logger,"train", X_train)
        ed_pipeline.generate_annotations(logger,"val", X_val)
        ed_pipeline.generate_annotations(logger,"test", X_test)

    logger.info("Data preparation pipeline has completed.")

if __name__ == "__main__":
    main()
