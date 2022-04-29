import cv2
import numpy as np
import hydra
import os
import sys
import time
import logging
from pconst import const
from pathlib import Path
from PIL import Image
from typing import Any, Dict, List, Tuple
from io import BytesIO

sys.path.append("..")
from src.life3_biotech.config import PipelineConfig
from src.life3_biotech import general_utils
from src.life3_biotech.modeling.EfficientDet.model import efficientdet
from src.life3_biotech.modeling.EfficientDet.utils import (
    preprocess_image,
    postprocess_boxes,
)
from src.life3_biotech.modeling.EfficientDet.utils.draw_boxes import draw_boxes

# EFFICIENTDET_CONFIG = {
#     "model_type": 0,
#     "score_threshold": 0.5,
#     "detect_ids": [0],
# }


class Life3EfficientDetModel:
    """Life3 EfficientDet model"""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()

        self.logger = logging.getLogger(__name__)

    def load_model(self):
        """This function initialises the saved EfficientDet model weights to be used for inference.

        Returns:
            tensorflow.keras.models.Model: Keras model object
            OrderedDict: dictionary containing the mapping of class IDs to class names
        """
        weighted_bifpn = True
        global temp_model_location
        self.logger.info(f"Classes: {const.CLASS_MAP_REVERSE}")
        num_classes = len(const.CLASS_MAP)
        _, model = efficientdet(
            phi=const.ED_INFERENCE_BACKBONE,
            weighted_bifpn=weighted_bifpn,
            num_classes=num_classes,
            score_threshold=const.INFERENCE_CONFIDENCE_THRESH,
        )
        model.load_weights(const.INFERENCE_MODEL_PATH, by_name=True)
        self.logger.info(
            f"Inferencing on backbone B{const.ED_INFERENCE_BACKBONE} with saved model weights: {const.INFERENCE_MODEL_PATH}"
        )
        return model

    def get_image_list(self):
        """This function initialises the list of images for EfficientDet to perform inference on.
        In `single` or `rest_api` inference mode, the list will contain a single image file path/URL.

        Returns:
            list: List of image file paths or URLs
        """
        image_list = []
        for filename in os.listdir(const.INFERENCE_INPUT_PATH):
            img_file = Path(const.INFERENCE_INPUT_PATH, filename)
            if img_file.exists() and img_file.is_file():
                image_list.append(img_file)
            else:
                self.logger.error(f"Invalid image file path: {img_file}")
        return image_list

    def predict(self, model, image_file):
        """This function preprocesses the given image and performs inference on it using the given model weights.

        Args:
            model (tensorflow.keras.models.Model): Keras inference model object
            image_file (str): URL of image file

        Returns:
            str: Formatted output for response if inference mode is `rest_api`
            float: Floating point number representing time taken to preprocess given image before inference
            float: Floating point number representing time taken to predict on given image
        """
        colors = [(244, 223, 156), (164, 232, 241), (119, 118, 188)]
        image_size = const.ED_IMAGE_SIZES[const.ED_INFERENCE_BACKBONE]

        # output = None

        preprocess_time = time.time()
        filename = image_file.name
        self.logger.info(f"Loading image from {image_file}")
        src_image, image = load_image_from_path(str(image_file))
        h, w = image.shape[:2]
        image, scale = preprocess_image(image, image_size=image_size)

        preprocess_time = time.time() - preprocess_time
        # run network
        start = time.time()
        boxes, scores, labels = model.predict_on_batch([np.expand_dims(image, axis=0)])
        boxes, scores, labels = (
            np.squeeze(boxes),
            np.squeeze(scores),
            np.squeeze(labels),
        )
        inf_time = time.time() - start

        boxes = postprocess_boxes(boxes=boxes, scale=scale, height=h, width=w)

        # select indices which have a score above the threshold
        indices = np.where(scores[:] > const.INFERENCE_CONFIDENCE_THRESH)[0]

        # select those detections
        boxes = boxes[indices]
        labels = labels[indices]
        if len(boxes) == 0:
            self.logger.info(f"No inference results for image: {filename}")
        else:
            self.logger.info(f"Drawing {len(boxes)} boxes...")
            output_image = draw_boxes(
                src_image, boxes, scores, labels, colors, const.CLASS_MAP_REVERSE
            )
            if const.INFERENCE_SAVE_OUTPUT:
                if not os.path.exists(const.INFERENCE_OUTPUT_PATH):
                    os.makedirs(const.INFERENCE_OUTPUT_PATH)
                output_filepath = str(Path(const.INFERENCE_OUTPUT_PATH, filename))
                self.logger.info(f"Writing output to: {output_filepath}")
                saved = cv2.imwrite(output_filepath, output_image)
                if saved:
                    self.logger.info(
                        f"Saved inferenced image to {const.INFERENCE_OUTPUT_PATH}{filename}"
                    )

        pred_output = {
            "bboxes": boxes,
            "bbox_labels": labels,
            "img_path": output_filepath,
        }

        return pred_output, preprocess_time, inf_time

    @hydra.main(config_path="../../../../conf/local", config_name="pipelines.yml")
    def main(self, image, args):
        """This function iterates through the list of images, preprocesses each image, calls the prediction function on it and preprocesses the results.

        Returns:
            dict: Output dictionary to be formatted to JSON if inference mode is `rest_api`. Otherwise, None is returned.
        """
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

        os.chdir(hydra.utils.get_original_cwd())

        # logger = logging.getLogger(__name__)
        self.logger.info("Setting up logging configuration.")
        logger_config_path = os.path.join(
            hydra.utils.get_original_cwd(), "conf/base/logging.yml"
        )
        general_utils.setup_logging(logger_config_path)

        pipeline_conf = PipelineConfig(args, logger)

        # images = get_image_list(logger)
        self.logger.info(f"Number of images: {len(images)}")

        total_inf_time = 0
        total_preprocess_time = 0

        model = self.load_model()

        # for i in images:
        pred_output, preprocess_time, inf_time = self.predict(model, image)
        # total_preprocess_time += preprocess_time
        # total_inf_time += inf_time

        # if total_inf_time > 0:
        #     self.logger.info(f"Time taken: {round(total_inf_time,2)} seconds")
        #     self.logger.info(f"FPS: {round(len(images) / total_inf_time,2)}")
        #     self.logger.info(
        #         f"Avg preprocessing time: {round(total_preprocess_time/len(images),2)} seconds"
        #     )

        return pred_output


# if __name__ == "__main__":
#     print(EfficientDetModel(config=EFFICIENTDET_CONFIG))
