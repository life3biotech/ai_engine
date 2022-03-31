import os
import logging
import pathlib
import hydra
from pconst import const

# from . import life3_biotech as life3
import life3_biotech as life3


@hydra.main(config_path="../conf/local", config_name="pipelines.yml")
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
        preprocessor = life3.data_prep.preprocess.Preprocessor(logger)
        preprocessor.generate_image_tiles()
        preprocessor.preprocess_annotations()
        preprocessor.split_data()  # to be implemented

        # if "efficientdet" in const.MODELS:
        #     ed_pipeline = life3.data_prep.preprocess_efficientdet.EfficientDetPipeline()
        #     # generate annotations for each dataset (train, val, test)
        #     ed_pipeline.generate_annotations(logger)

        logging.info("Data preparation pipeline has completed.")


if __name__ == "__main__":
    main()
