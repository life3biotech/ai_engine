from ast import arg
import os
import datetime
import logging
import hydra
import jsonlines
from pconst import const
import tensorflow as tf

import life3_biotech as life3
from src.life3_biotech.config import PipelineConfig
from inference.inference_pipeline import PeekingDuckPipeline, get_image_list


@hydra.main(
    config_path="/Users/kwanchet/Documents/aisg_100e/life3_biotech/consultancy_project/development/custom_node_model/life3/conf/local",
    config_name="pipelines.yml",
)
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
    logger_config_path = os.path.join(
        hydra.utils.get_original_cwd(), "conf/base/logging.yml"
    )
    life3.general_utils.setup_logging(logger_config_path)
    pipeline_conf = PipelineConfig(args, logger)

    # logger.info("Loading the model...")
    # pred_model = life3.modeling.utils.load_model(args["inference"]["model_path"])

    # call inference functions here
    logger.info("Loading the input data")
    image_list = get_image_list(logger=logger)
    logger.info(f"Displaying image list:{image_list}")
    # dataset = tf.keras.utils.image_dataset_from_directory(
    #     args.inference.input_path, batch_size=1, shuffle=False
    # )

    logger.info("Instantiating model")
    pkd = PeekingDuckPipeline(
        logger=logger,
        model_node_type="life3_effdet",
        model_config=const.EFFICIENTDET_CONFIG,
        draw_node_type="draw_bbox",
        draw_config=const.DRAW_BBOX_CONFIG,
    )

    # pkd = PeekingDuckPipeline(
    #     logger=logger,
    #     model_node_type="effdet",
    #     model_config=args.inference.efficientdet_config,
    #     draw_node_type="draw_bbox",
    #     draw_config=args.inference.draw_bbox_config,
    # )

    logger.info("Generating model predictions")
    # pred_outputs = pkd.predict(
    #     dataset=dataset, save_output_option=True, output_path=args.inference.output_path
    # )

    pred_outputs = pkd.predict(
        image_list=image_list,
        save_output_option=True,
        output_path=const.INFERENCE_OUTPUT_PATH,
    )

    logger.info("Batch inferencing has completed.")
    logger.info("Output result location: {}".format(args.inference.output_path))


if __name__ == "__main__":
    main()
