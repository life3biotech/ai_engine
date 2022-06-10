import os
import logging
import hydra

from datetime import datetime, timedelta, timezone
from pconst import const

import load_data
import life3_biotech as life3

@hydra.main(config_path="../conf/life3", config_name="pipelines.yml")
def main(args):
    """This main function does the following:
    - load logging config
    - loads training, validation and test data
    - initialises model layers and compile
    - trains, evaluates, and then exports the model
    """
    global current_datetime, model_args

    tzinfo = timezone(timedelta(hours=8))
    current_datetime = datetime.now(tzinfo).strftime("%Y%m%d_%H%M%S")

    os.chdir(
        hydra.utils.get_original_cwd()
    )  # Default behavior of hydra changes the working directory, this line changes the working directory back to the root

    logger = logging.getLogger(__name__)
    logger.info("Setting up logging configuration.")
    logger_config_path = os.path.join(
        hydra.utils.get_original_cwd(), "conf/base/logging.yml"
    )
    life3.general_utils.setup_logging(logger_config_path)

    logger.info(f"Current directory: {os.getcwd()}")
    logger.info(f"Current datetime: {current_datetime}")

    pipeline_conf = life3.config.PipelineConfig(args, logger)

    logger.info(f"Training {const.TRAIN_MODEL_NAME} model")

    if const.LOAD_DATA:
        logger.info("Running data pipeline prior to training")
        load_data.run_data_pipeline(logger)

    model_args = dict(args[const.TRAIN_MODEL_NAME])
    train_args = dict(args["train"])
    logger.info(f"{const.TRAIN_MODEL_NAME} model parameters: {model_args}")
    logger.info(f"Training parameters: {train_args}")

    if const.TRAIN_MODEL_NAME == "efficientdet":
        run_efficientdet(current_datetime, logger)

    else:
        logger.info("Model training has completed.")


def run_efficientdet(current_datetime: str, logger) -> None:
    """This function runs the EfficientDet model training and evaluation according to the parameters in the `pipelines.yml` config file. 

    Args:
        current_datetime (str): Current timestamp used as a suffix for model weights file
        logger (logging.Logger): Logger object used for logging
    """
    logger.info("Importing EfficientDet model")
    # import here to avoid dependency issues if EfficientDet model is removed from pipeline
    from life3_biotech.modeling.EfficientDet.train import main as train_efficientdet
    from life3_biotech.modeling.EfficientDet.eval.common import (
        main as eval_efficientdet,
    )

    logger.info("Parsing EfficientDet model parameters")
    args = life3.modeling.utils.transform_args(
        model_args
    )  # returns a list of arguments
    if args:
        model_path = None
        logger.info("Calling EfficientDet model training...")
        train_efficientdet(current_datetime, logger, args)
        logger.info("Calling EfficientDet model evaluation...")
        eval_metrics_dict = eval_efficientdet(
            current_datetime, logger, args, model_path
        )
        logger.info(f"Metrics: {eval_metrics_dict}")
    else:
        logger.warning(f"Unable to proceed with training {const.TRAIN_MODEL_NAME}")


if __name__ == "__main__":
    main()
