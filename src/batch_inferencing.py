import os
import datetime
import logging
import hydra
import jsonlines


import life3_biotech as life3


@hydra.main(config_path="../conf/base", config_name="pipelines.yml")
def main(args):
    """This main function does the following:
    - load logging config
    - gets list of files to be loaded for inferencing
    - loads trained model
    - conducts inferencing on data
    - outputs prediction results to a jsonline file
    """

    logger = logging.getLogger(__name__)
    logger.info("Setting up logging configuration.")
    logger_config_path = os.path.\
        join(hydra.utils.get_original_cwd(),
            "conf/base/logging.yml")
    life3.general_utils.setup_logging(logger_config_path)

    logger.info("Loading the model...")
    pred_model = life3.modeling.utils.load_model(
        args["inference"]["model_path"])

    # call inference functions here

    logger.info("Batch inferencing has completed.")
    logger.info("Output result location: {}/batch-infer-res.jsonl".
                format(os.getcwd()))

if __name__ == "__main__":
    main()
