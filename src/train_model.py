import os
import logging
import hydra
import mlflow

from datetime import datetime, timedelta, timezone
from pconst import const

import load_data
import life3_biotech as life3
# from . import load_data
# from . import life3_biotech as life3


@hydra.main(config_path="../conf/base", config_name="pipelines.yml")
def main(args):
    """This main function does the following:
    - load logging config
    - initialise experiment tracking (MLflow)
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

    mlflow_init_status, mlflow_run = life3.general_utils.mlflow_init(
        args,
        setup_mlflow=args["train"]["setup_mlflow"],
        autolog=args["train"]["mlflow_autolog"],
    )
    life3.general_utils.mlflow_log(
        mlflow_init_status, "log_params", params=args["train"]
    )

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

    if mlflow_init_status:
        artifact_uri = mlflow.get_artifact_uri()
        logger.info("Artifact URI: {}".format(artifact_uri))
        life3.general_utils.mlflow_log(
            mlflow_init_status, "log_params", params={"artifact_uri": artifact_uri}
        )
        logger.info(
            "Model training with MLflow run ID {} has completed.".format(
                mlflow_run.info.run_id
            )
        )
        mlflow.end_run()
    else:
        logger.info("Model training has completed.")


def run_efficientdet(current_datetime: str, logger) -> None:
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
        if const.EVAL_ONLY:
            model_path = f"{const.SAVED_MODEL_PATH}{const.BEST_MODEL}"
        else:
            model_path = None
            logger.info("Calling EfficientDet model training...")
            train_efficientdet(current_datetime, logger, args)
        logger.info('Calling EfficientDet model evaluation...')
        eval_metrics_dict = eval_efficientdet(current_datetime, logger, args, model_path)
        logger.info(f'Metrics: {eval_metrics_dict}')
    else:
        logger.warning(f"Unable to proceed with training {const.TRAIN_MODEL_NAME}")


if __name__ == "__main__":
    main()
