import os

import logging
import hydra

from pconst import const

import time

import life3_biotech as life3
import sahi

from pathlib import Path
import pandas as pd

from inferencing.eval import EvalCalibrate


@hydra.main(config_path="../conf/life3", config_name="pipelines.yml")
def main(args):
    """This main function does the following:
    - load logging config
    - gets list of files to be loaded for inferencing
    - loads trained model
    - conducts inferencing on data
    - outputs evaluation and calibration of ground truth vs prediction data distribution to png image file
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

    eval_calibrate = EvalCalibrate(logger)
    eval_calibrate.eval_process(
        calibrate_cell_size_bool=True  # Set calibrate_cell_size_bool to True to calibrate cellsize.
    )  # process a bunch of images

    # for testing of one full image data distribution
    # eval_calibrate.eval_process_oneimg(
    #     "50.jpg"
    # )  # process one full image only. (not working for slice image)

    end = time.time()
    logger.info("The time of execution of above program is : {}".format(end - start))


if __name__ == "__main__":
    main()
