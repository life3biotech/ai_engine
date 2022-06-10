import os
import time
import logging
import hydra
from pconst import const

import life3_biotech as life3
import inferencing as inference

from inferencing.inference_util import params_check

from pathlib import Path
import pandas as pd


@hydra.main(config_path="../conf/life3", config_name="pipelines.yml")
def main(args):
    """This main function does the following:
    - load logging config
    - gets list of files to be loaded for inferencing
    - loads trained model
    - conducts inferencing on data
    - outputs prediction results to a jsonline file
    """
    os.chdir(
        hydra.utils.get_original_cwd()
    )  # Default behavior of hydra changes the working directory, this line changes the working directory back to the root

    logger = logging.getLogger(__name__)
    logger.info("Setting up logging configuration.")
    logger_config_path = os.path.join(
        hydra.utils.get_original_cwd(), "conf/base/logging.yml"
    )
    life3.general_utils.setup_logging(logger_config_path)
    pipeline_conf = life3.config.PipelineConfig(args, logger)

    start = time.time()

    # Check if piplines parameter is different from calibrated parameter and warn user
    if const.USE_CALIBRATED_CELLSIZE:
        params_check_name = params_check(logger)

    batch_inference = inference.batch_inference.BatchInference(logger)
    batch_inference.inferencing()

    end = time.time()
    logger.info("The time of execution of above program is : {}".format(end - start))

    if const.USE_CALIBRATED_CELLSIZE:
        if len(params_check_name) > 0:
            logger.warning(
                "Config Parameters have changed, pipelines.yml differ from calibrated_params.csv. Affected parameters: "
                + ", ".join(str(x) for x in params_check_name)
            )
            logger.warning(
                "Config Parameters have changed, please rerun eval_model to recalibrate cellsize."
            )


if __name__ == "__main__":
    main()
