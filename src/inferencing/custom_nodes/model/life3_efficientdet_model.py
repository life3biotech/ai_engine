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

from life3_biotech.modeling.EfficientDet.model import efficientdet
from life3_biotech.modeling.EfficientDet.utils import (
    preprocess_image,
    postprocess_boxes,
)
from life3_biotech.modeling.EfficientDet.utils.draw_boxes import draw_boxes


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
            phi=const.INFERENCE_BACKBONE,
            weighted_bifpn=weighted_bifpn,
            num_classes=num_classes,
            score_threshold=const.INFERENCE_CONFIDENCE_THRESH,
        )
        model.load_weights(const.INFERENCE_MODEL_PATH, by_name=True)
        self.logger.info(
            f"Inferencing on backbone B{const.INFERENCE_BACKBONE} with saved model weights: {const.INFERENCE_MODEL_PATH}"
        )
        return model

    # def predict(self, model, image, filename):
    def predict(
        self, model, image
    ):  # remove filename - move output file handling to sahi
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
        image_size = const.ED_IMAGE_SIZES[const.INFERENCE_BACKBONE]

        preprocess_time = time.time()

        src_image = image.copy()
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
        scores = scores[indices]
        if len(boxes) == 0:
            self.logger.info(f"No inference results for image")
        else:
            self.logger.info(f"Drawing {len(boxes)} boxes...")
            output_image = draw_boxes(
                src_image, boxes, scores, labels, colors, const.CLASS_MAP_REVERSE
            )
            # if const.INFERENCE_SAVE_OUTPUT:   # Shifted save output to be handle by Sahi
            #     if not os.path.exists(const.INFERENCE_OUTPUT_PATH):
            #         os.makedirs(const.INFERENCE_OUTPUT_PATH)
            #     output_filepath = str(Path(const.INFERENCE_OUTPUT_PATH, filename))
            #     self.logger.info(f"Writing output to: {output_filepath}")
            #     saved = cv2.imwrite(output_filepath, output_image)
            #     if saved:
            #         self.logger.info(
            #             f"Saved inferenced image to {const.INFERENCE_OUTPUT_PATH}{filename}"
            #         )

            pred_output = {
                "bboxes": boxes,
                "bbox_labels": labels,
                "bbox_scores": scores,
                # "img_path": output_filepath,
            }

            return pred_output
