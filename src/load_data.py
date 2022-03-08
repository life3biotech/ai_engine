import os
import logging
import pathlib
import hydra

import life3_biotech as life3


@hydra.main(config_path="../conf/base", config_name="pipelines.yml")
def main(args):
    """Main program to read in raw data files and process them.
    """

    logger = logging.getLogger(__name__)
    logger.info("Setting up logging configuration.")
    logger_config_path = os.path.\
        join(hydra.utils.get_original_cwd(),
            "conf/base/logging.yml")
    life3.general_utils.setup_logging(logger_config_path)

    raw_data_dirs_list = args["data_prep"]["raw_dirs_paths"]
    processed_data_path = args["data_prep"]["processed_data_path"]

    for raw_data_dir in raw_data_dirs_list:

        raw_data_dir = os.path.join(
            hydra.utils.get_original_cwd(), raw_data_dir)
        processed_data_path = os.path.join(
            hydra.utils.get_original_cwd(), processed_data_path)
        # call preprocessing functions here

    logging.info("Data preparation pipeline has completed.")


if __name__ == "__main__":
    main()
