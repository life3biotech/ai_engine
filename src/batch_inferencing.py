from ast import arg
import os
import datetime
import logging
import hydra
import jsonlines
import tensorflow as tf

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
    logger_config_path = os.path.join(
        hydra.utils.get_original_cwd(), "conf/base/logging.yml"
    )
    life3.general_utils.setup_logging(logger_config_path)

    # logger.info("Loading the model...")
    # pred_model = life3.modeling.utils.load_model(args["inference"]["model_path"])

    # call inference functions here
    logger.info("Loading the input data")
    dataset = tf.keras.utils.image_dataset_from_directory(
        args.inference.input_data_dir, batch_size=1, shuffle=False
    )

    logger.info("Instantiating model")
    pkd = life3.inference.inference_pipeline.PeekingDuckPipeline(
        logger=logger,
        model_node_type="effdet",
        model_config=args.inference.efficientdet_config,
        draw_node_type="draw_bbox",
        draw_config=args.inference.draw_bbox_config,
    )

    logger.info("Generating model predictions")
    pred_outputs = pkd.predict(
        dataset=dataset, save_output_option=True, output_path=args.inference.output_path
    )

    logger.info("Batch inferencing has completed.")
    logger.info("Output result location: {}".format(args.inference.output_path))


if __name__ == "__main__":
    main()
