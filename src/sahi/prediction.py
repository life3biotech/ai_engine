# OBSS SAHI Tool
# Code written by Fatih C Akyon, 2020.

import copy
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from PIL import Image

from .annotation import ObjectAnnotation
from .utils.coco import CocoAnnotation, CocoPrediction
from .utils.cv import read_image_as_pil, visualize_object_predictions
from .utils.file import Path


class PredictionScore:
    def __init__(self, value: float):
        """
        Arguments:
            score: prediction score between 0 and 1
        """
        # if score is a numpy object, convert it to python variable
        if type(value).__module__ == "numpy":
            value = copy.deepcopy(value).tolist()
        # set score
        self.value = value

    def is_greater_than_threshold(self, threshold):
        """
        Check if score is greater than threshold
        """
        return self.value > threshold

    def __repr__(self):
        return f"PredictionScore: <value: {self.value}>"


class ObjectPrediction(ObjectAnnotation):
    """
    Class for handling detection model predictions.
    """

    def __init__(
        self,
        bbox: Optional[List[int]] = None,
        category_id: Optional[int] = None,
        category_name: Optional[str] = None,
        bool_mask: Optional[np.ndarray] = None,
        score: Optional[float] = 0,
        shift_amount: Optional[List[int]] = [0, 0],
        full_shape: Optional[List[int]] = None,
    ):
        """
        Creates ObjectPrediction from bbox, score, category_id, category_name, bool_mask.

        Arguments:
            bbox: list
                [minx, miny, maxx, maxy]
            score: float
                Prediction score between 0 and 1
            category_id: int
                ID of the object category
            category_name: str
                Name of the object category
            bool_mask: np.ndarray
                2D boolean mask array. Should be None if model doesn't output segmentation mask.
            shift_amount: list
                To shift the box and mask predictions from sliced image
                to full sized image, should be in the form of [shift_x, shift_y]
            full_shape: list
                Size of the full image after shifting, should be in
                the form of [height, width]
        """
        self.score = PredictionScore(score)
        super().__init__(
            bbox=bbox,
            category_id=category_id,
            bool_mask=bool_mask,
            category_name=category_name,
            shift_amount=shift_amount,
            full_shape=full_shape,
        )

    def get_shifted_object_prediction(self):
        """
        Returns shifted version ObjectPrediction.
        Shifts bbox and mask coords.
        Used for mapping sliced predictions over full image.
        """
        if self.mask:
            return ObjectPrediction(
                bbox=self.bbox.get_shifted_box().to_voc_bbox(),
                category_id=self.category.id,
                score=self.score.value,
                bool_mask=self.mask.get_shifted_mask().bool_mask,
                category_name=self.category.name,
                shift_amount=[0, 0],
                full_shape=self.mask.get_shifted_mask().full_shape,
            )
        else:
            return ObjectPrediction(
                bbox=self.bbox.get_shifted_box().to_voc_bbox(),
                category_id=self.category.id,
                score=self.score.value,
                bool_mask=None,
                category_name=self.category.name,
                shift_amount=[0, 0],
                full_shape=None,
            )

    def to_coco_prediction(self, image_id=None):
        """
        Returns sahi.utils.coco.CocoPrediction representation of ObjectAnnotation.
        """
        if self.mask:
            coco_prediction = CocoPrediction.from_coco_segmentation(
                segmentation=self.mask.to_coco_segmentation(),
                category_id=self.category.id,
                category_name=self.category.name,
                score=self.score.value,
                image_id=image_id,
            )
        else:
            coco_prediction = CocoPrediction.from_coco_bbox(
                bbox=self.bbox.to_coco_bbox(),
                category_id=self.category.id,
                category_name=self.category.name,
                score=self.score.value,
                image_id=image_id,
            )
        return coco_prediction

    def to_fiftyone_detection(self, image_height: int, image_width: int):
        """
        Returns fiftyone.Detection representation of ObjectPrediction.
        """
        try:
            import fiftyone as fo
        except ImportError:
            raise ImportError(
                'Please run "pip install -U fiftyone" to install fiftyone first for fiftyone conversion.'
            )

        x1, y1, x2, y2 = self.bbox.to_voc_bbox()
        rel_box = [
            x1 / image_width,
            y1 / image_height,
            (x2 - x1) / image_width,
            (y2 - y1) / image_height,
        ]
        fiftyone_detection = fo.Detection(
            label=self.category.name, bounding_box=rel_box, confidence=self.score.value
        )
        return fiftyone_detection

    def __repr__(self):
        return f"""ObjectPrediction<
    bbox: {self.bbox},
    mask: {self.mask},
    score: {self.score},
    category: {self.category}>"""


class PredictionResult:
    def __init__(
        self,
        object_prediction_list: List[ObjectPrediction],
        image: Union[Image.Image, str, np.ndarray],
        durations_in_seconds: Optional[Dict] = None,
    ):
        self.image: Image.Image = read_image_as_pil(image)
        self.image_width, self.image_height = self.image.size
        self.object_prediction_list: List[ObjectPrediction] = object_prediction_list
        self.durations_in_seconds = durations_in_seconds

    def export_visuals(
        self,
        export_dir: str,
        file_name: str,
        text_size: float = None,
        rect_th: int = None,
        label_bool: bool = True,
        show_cellcount: bool = True,
        cellcount_info: pd.DataFrame = None,
    ):
        Path(export_dir).mkdir(parents=True, exist_ok=True)
        return visualize_object_predictions(
            image=np.ascontiguousarray(self.image),
            object_prediction_list=self.object_prediction_list,
            rect_th=rect_th,
            text_size=text_size,
            text_th=None,
            color=None,
            output_dir=export_dir,
            file_name=file_name,
            export_format="png",
            label_bool=label_bool,
            show_cellcount=show_cellcount,
            cellcount_info=cellcount_info,
        )

    def to_coco_annotations(self, panda_df_bool: bool = False):
        """

        Args:
            panda_df_bool (bool, optional): panda dataframe boolean. Defaults to True.

        Returns:
            _type_: dataframe if panda_df_bool=True else return coco list
        """
        coco_annotation_list = []
        for object_prediction in self.object_prediction_list:
            coco_annotation_list.append(object_prediction.to_coco_prediction().json)

        if not panda_df_bool:
            return coco_annotation_list
        else:
            return self._convert_sahi_cocoresult_csv(coco_annotation_list)

    def to_coco_predictions(self, image_id: Optional[int] = None):
        coco_prediction_list = []
        for object_prediction in self.object_prediction_list:
            coco_prediction_list.append(
                object_prediction.to_coco_prediction(image_id=image_id).json
            )
        return coco_prediction_list

    def to_imantics_annotations(self):
        imantics_annotation_list = []
        for object_prediction in self.object_prediction_list:
            imantics_annotation_list.append(object_prediction.to_imantics_annotation())
        return imantics_annotation_list

    def to_fiftyone_detections(self):
        try:
            import fiftyone as fo
        except ImportError:
            raise ImportError(
                'Please run "pip install -U fiftyone" to install fiftyone first for fiftyone conversion.'
            )

        fiftyone_detection_list: List[fo.Detection] = []
        for object_prediction in self.object_prediction_list:
            fiftyone_detection_list.append(
                object_prediction.to_fiftyone_detection(
                    image_height=self.image_height, image_width=self.image_width
                )
            )
        return fiftyone_detection_list

    def _convert_sahi_cocoresult_csv(self, coco_list):
        """
        Helper function to convert sahi generated coco list to panda dataframe

        Args:
            coco_list (list): Sahi coco list that contains dictionary elements

        Returns:
            panda dataframe: return as dataframe format
        """
        import pandas as pd

        bbox_list = [i["bbox"] for i in coco_list]
        score_list = [i["score"] for i in coco_list]
        category_id_list = [i["category_id"] for i in coco_list]
        category_name_list = [i["category_name"] for i in coco_list]
        area_list = [i["area"] for i in coco_list]

        dict_ = {
            "bbox": bbox_list,
            "score": score_list,
            "category_id": category_id_list,
            "category_name": category_name_list,
            "area": area_list,
        }
        df = pd.DataFrame(dict_)
        split_df = pd.DataFrame(
            df["bbox"].to_list(),
            columns=["bbox_x", "bbox_y", "bbox_width", "bbox_height"],
        )
        df = pd.concat([split_df, df], axis=1)
        df = df.drop("bbox", axis=1)
        return df
