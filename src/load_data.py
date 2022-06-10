import os
import logging
import pathlib
import hydra
from pconst import const

import life3_biotech as life3


@hydra.main(config_path="../conf/life3", config_name="pipelines.yml")
def main(args):
    """Main program to read in raw data files and process them."""

    logger = logging.getLogger(__name__)
    logger.info("Setting up logging configuration.")
    logger_config_path = os.path.join(
        hydra.utils.get_original_cwd(), "conf/life3/logging.yml"
    )
    life3.general_utils.setup_logging(logger_config_path)

    pipeline_conf = life3.config.PipelineConfig(args, logger)

    run_data_pipeline(logger)


def run_data_pipeline(logger):
    """This function calls the image slicing, annotation pre-processing and data splitting functions in order.

    Args:
        logger (logging.Logger): Logger object used for logging
    """    
    preprocessor = life3.data_prep.preprocess.Preprocessor(logger)
    if const.RUN_TILING:
        preprocessor.generate_image_tiles()
    df = preprocessor.preprocess_annotations()
    if const.TEST_SIZE == 0.0 and const.VAL_SIZE == 0.0:
        logger.info("Validation and test set sizes defined as zero; skipping data split process")
        X_train = df
    else:
        X_train, y_train, X_test, y_test, X_val, y_val = preprocessor.split_data(
            df, test_size=const.TEST_SIZE, val_size=const.VAL_SIZE
        )

    if "efficientdet" in const.MODELS:
        ed_pipeline = life3.data_prep.preprocess_efficientdet.EfficientDetPipeline()
        # generate annotations for each dataset (train, val, test)
        ed_pipeline.generate_annotations(logger, const.TRAIN_SET_FILENAME, X_train)
        if const.VAL_SIZE > 0.0:
            ed_pipeline.generate_annotations(logger, const.VAL_SET_FILENAME, X_val)
        if const.TEST_SIZE > 0.0:
            ed_pipeline.generate_annotations(logger, const.TEST_SET_FILENAME, X_test)

    logger.info("Data preparation pipeline has completed.")


if __name__ == "__main__":
    main()
