# from ast import arg
import os

# import datetime
import logging
import hydra

# import jsonlines
from pconst import const

# import tensorflow as tf
import time

import life3_biotech as life3
import sahi


@hydra.main(config_path="../conf/base", config_name="pipelines.yml")
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

    detection_model = sahi.model.EfficientDetModel(
        device="cpu",  # or 'cuda:0'
    )

    # predicting without slicing the image
    # result = sahi.predict.get_prediction(
    #     const.IMAGE_INPUT_PATH,
    #     detection_model,
    # )

    result = sahi.predict.get_sliced_prediction(
        const.IMAGE_INPUT_PATH,
        detection_model,
        slice_height=const.SLICE_HEIGHT,
        slice_width=const.SLICE_WIDTH,
        overlap_height_ratio=const.OVERLAP_HEIGHT_RATIO,
        overlap_width_ratio=const.OVERLAP_WIDTH_RATIO,
        postprocess_type=const.POSTPROCESS_TYPE,
        postprocess_match_metric=const.POSTPROCESS_MATCH_METRIC,
        postprocess_match_threshold=const.POSTPROCESS_MATCH_THRESHOLD,
    )

    # Export predicted output annotations to csv
    df = result.to_coco_annotations(panda_df_bool=True)
    df.to_csv(const.CSV_OUTPUT)

    # Export predicted output image
    result.export_visuals(export_dir=const.IMAGE_OUTPUT_DIR)

    end = time.time()

    logger.info("Batch inferencing has completed.")
    logger.info("Output result location: {}".format(const.CSV_OUTPUT))
    logger.info("The time of execution of above program is : {}".format(end - start))


if __name__ == "__main__":
    main()
