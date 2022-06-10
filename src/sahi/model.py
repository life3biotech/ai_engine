# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.
# Modified for Life3

import logging
import warnings
from typing import Dict, List, Optional

import numpy as np

from .prediction import ObjectPrediction
from .utils.compatibility import fix_full_shape_list, fix_shift_amount_list
from .utils.cv import get_bbox_from_bool_mask
from .utils.torch import cuda_is_available, empty_cuda_cache

import life3_biotech as life3
from inferencing.inference_pipeline import PeekingDuckPipeline

from pconst import const

logger = logging.getLogger(__name__)


class DetectionModel:
    def __init__(
        self,
        config_path: Optional[str] = None,
        device: Optional[str] = None,
        mask_threshold: float = 0.5,
        confidence_threshold: float = 0.3,
        category_mapping: Optional[Dict] = None,
        category_remapping: Optional[Dict] = None,
        load_at_init: bool = True,
        image_size: int = None,
    ):
        """
        Init object detection/instance segmentation model.
        Args:
            config_path: str
                Path for the mmdetection instance segmentation model config file
            device: str
                Torch device, "cpu" or "cuda"
            mask_threshold: float
                Value to threshold mask pixels, should be between 0 and 1
            confidence_threshold: float
                All predictions with score < confidence_threshold will be discarded
            category_mapping: dict: str to str
                Mapping from category id (str) to category name (str) e.g. {"1": "pedestrian"}
            category_remapping: dict: str to int
                Remap category ids based on category names, after performing inference e.g. {"car": 3}
            load_at_init: bool
                If True, automatically loads the model at initalization
            image_size: int
                Inference input size.
        """
        self.config_path = config_path
        self.model = None
        self.device = device
        self.mask_threshold = mask_threshold
        self.confidence_threshold = confidence_threshold
        self.category_mapping = category_mapping
        self.category_remapping = category_remapping
        self.image_size = image_size
        self._original_predictions = None
        self._object_prediction_list_per_image = None

        # automatically set device if its None
        if not (self.device):
            self.device = "cuda:0" if cuda_is_available() else "cpu"

        # automatically load model if load_at_init is True
        if load_at_init:
            self.load_model()

    def load_model(self):
        """
        This function should be implemented in a way that detection model
        should be initialized and set to self.model.
        (self.config_path, and self.device should be utilized)
        """
        raise NotImplementedError()

    def unload_model(self):
        """
        Unloads the model from CPU/GPU.
        """
        self.model = None
        empty_cuda_cache()

    def perform_inference(self, image: np.ndarray, image_size: int = None):
        """
        This function should be implemented in a way that prediction should be
        performed using self.model and the prediction result should be set to self._original_predictions.
        Args:
            image: np.ndarray
                A numpy array that contains the image to be predicted.
            image_size: int
                Inference input size.
        """
        raise NotImplementedError()

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: Optional[List[List[int]]] = [[0, 0]],
        full_shape_list: Optional[List[List[int]]] = None,
    ):
        """
        This function should be implemented in a way that self._original_predictions should
        be converted to a list of prediction.ObjectPrediction and set to
        self._object_prediction_list. self.mask_threshold can also be utilized.
        Args:
            shift_amount_list: list of list
                To shift the box and mask predictions from sliced image to full sized image, should
                be in the form of List[[shift_x, shift_y],[shift_x, shift_y],...]
            full_shape_list: list of list
                Size of the full image after shifting, should be in the form of
                List[[height, width],[height, width],...]
        """
        raise NotImplementedError()

    def _apply_category_remapping(self):
        """
        Applies category remapping based on mapping given in self.category_remapping
        """
        # confirm self.category_remapping is not None
        assert (
            self.category_remapping is not None
        ), "self.category_remapping cannot be None"
        # remap categories
        for object_prediction_list in self._object_prediction_list_per_image:
            for object_prediction in object_prediction_list:
                old_category_id_str = str(object_prediction.category.id)
                new_category_id_int = self.category_remapping[old_category_id_str]
                object_prediction.category.id = new_category_id_int

    def convert_original_predictions(
        self,
        slice_height: int = 512,
        slice_width: int = 512,
        shift_amount: Optional[List[int]] = [0, 0],
        full_shape: Optional[List[int]] = None,
    ):
        """
        Converts original predictions of the detection model to a list of
        prediction.ObjectPrediction object. Should be called after perform_inference().
        Args:
            shift_amount: list
                To shift the box and mask predictions from sliced image to full sized image, should be in the form of [shift_x, shift_y]
            full_shape: list
                Size of the full image after shifting, should be in the form of [height, width]
        """
        self._create_object_prediction_list_from_original_predictions(
            shift_amount_list=shift_amount,
            full_shape_list=full_shape,
            slice_height=slice_height,
            slice_width=slice_width,
        )
        if self.category_remapping:
            self._apply_category_remapping()

    @property
    def object_prediction_list(self):
        return self._object_prediction_list_per_image[0]

    @property
    def object_prediction_list_per_image(self):
        return self._object_prediction_list_per_image

    @property
    def original_predictions(self):
        return self._original_predictions


class EfficientDetModel(DetectionModel):
    def load_model(self):
        """
        Detection model is initialized and set to self.model.
        """

        logger.info("Instantiating model")

        model = PeekingDuckPipeline(
            logger=logger,
            model_node_type="life3_effdet",
            model_config=const.EFFICIENTDET_CONFIG,
            # draw_node_type="draw_bbox",
            # draw_config=const.DRAW_BBOX_CONFIG,
        )

        self.model = model

    def perform_inference(self, image: np.ndarray, image_size: int = None):
        """
        Prediction is performed using self.model and the prediction result is set to self._original_predictions.
        Args:
            image: np.ndarray
                A numpy array that contains the image to be predicted. 3 channel image should be in RGB order.
            image_size: int
                Inference input size.
        """
        prediction_result = self.model.predict(
            image,
            save_output_option=False,
        )

        self._original_predictions = prediction_result

    @property
    def num_categories(self):
        """
        Returns number of categories
        """
        return len(self.model.names)

    @property
    def has_mask(self):
        """
        Returns if model output contains segmentation mask
        """
        has_mask = self.model.with_mask
        return has_mask

    @property
    def category_names(self):
        return self.model.names

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: Optional[List[List[int]]] = [[0, 0]],
        full_shape_list: Optional[List[List[int]]] = None,
        slice_height: int = 512,
        slice_width: int = 512,
    ):
        """
        self._original_predictions is converted to a list of prediction.ObjectPrediction and set to
        self._object_prediction_list_per_image.
        Args:
            shift_amount_list: list of list
                To shift the box and mask predictions from sliced image to full sized image, should
                be in the form of List[[shift_x, shift_y],[shift_x, shift_y],...]
            full_shape_list: list of list
                Size of the full image after shifting, should be in the form of
                List[[height, width],[height, width],...]
        """
        original_predictions = self._original_predictions
        # compatilibty for sahi v0.8.15
        shift_amount_list = fix_shift_amount_list(shift_amount_list)
        full_shape_list = fix_full_shape_list(full_shape_list)

        # create object_prediction_list
        object_prediction_list_per_image = []
        object_prediction_list = []

        shift_amount = shift_amount_list[0]
        full_shape = None if full_shape_list is None else full_shape_list[0]

        try:
            for j in range(len(original_predictions["bboxes"])):

                # if output is normalized, peekingduck efficientDet bboxes are recorded normalized
                # x1 = int(original_predictions["bboxes"][j][0] * slice_width)
                # y1 = int(original_predictions["bboxes"][j][1] * slice_height)
                # x2 = int(original_predictions["bboxes"][j][2] * slice_width)
                # y2 = int(original_predictions["bboxes"][j][3] * slice_height)

                x1 = int(original_predictions["bboxes"][j][0])
                y1 = int(original_predictions["bboxes"][j][1])
                x2 = int(original_predictions["bboxes"][j][2])
                y2 = int(original_predictions["bboxes"][j][3])
                bbox = [x1, y1, x2, y2]

                score = original_predictions["bbox_scores"][j]
                # category_id = int(prediction[5])
                category_id = int(original_predictions["bbox_labels"][j])

                # fix negative box coords
                bbox[0] = max(0, bbox[0])
                bbox[1] = max(0, bbox[1])
                bbox[2] = max(0, bbox[2])
                bbox[3] = max(0, bbox[3])

                # fix out of image box coords
                if full_shape is not None:
                    bbox[0] = min(full_shape[1], bbox[0])
                    bbox[1] = min(full_shape[0], bbox[1])
                    bbox[2] = min(full_shape[1], bbox[2])
                    bbox[3] = min(full_shape[0], bbox[3])

                # ignore invalid predictions
                if not (bbox[0] < bbox[2]) or not (bbox[1] < bbox[3]):
                    logger.warning(f"ignoring invalid prediction with bbox: {bbox}")
                    continue

                class_dict = const.CLASS_MAP_REVERSE
                category_name = class_dict[category_id]

                object_prediction = ObjectPrediction(
                    bbox=bbox,
                    category_id=category_id,
                    score=score,
                    bool_mask=None,
                    category_name=category_name,
                    shift_amount=shift_amount,
                    full_shape=full_shape,
                )
                object_prediction_list.append(object_prediction)
        except:  # Skip image without prediction
            logger.warning(
                f"ignoring current slice image prediction with no bbox generated"
            )

        object_prediction_list_per_image.append(object_prediction_list)

        self._object_prediction_list_per_image = object_prediction_list_per_image
