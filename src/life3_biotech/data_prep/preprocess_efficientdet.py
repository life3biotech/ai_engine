import os
from pathlib import Path, PurePath
from typing import Dict, List, Tuple
import pandas as pd
from json import load
from collections import defaultdict
from csv import DictWriter, writer

DATA_FOLDER = Path("data")
RAW_FOLDER = PurePath(DATA_FOLDER, "raw")
RAW_IMAGE_FOLDER = PurePath(RAW_FOLDER, "images")
ANNOTATION_FOLDER = PurePath(RAW_FOLDER, "annotations")
COCO_ANNOTATION_PATH = PurePath(ANNOTATION_FOLDER, "coco_annotations.json")


class EfficientDetPipeline:
    def _load_coco_annotations(self, path_to_coco_annot: Path) -> Dict:
        """
        Load COCO annotations from the filepath provided

        Args:
            path_to_coco_annot (Path): File path to COCO annotations
        """
        with open(COCO_ANNOTATION_PATH, "r") as f:
            coco_annotation = load(f)

        return coco_annotation

    def convert_coco_to_csv(self) -> None:
        """
        Convert COCO annotation to a csv file to be ingested by models
        such as EfficientDet

        Args:
            None
        """
        coco_annotations = self._load_coco_annotations(COCO_ANNOTATION_PATH)

        image_info = coco_annotations["images"]
        annotations = coco_annotations["annotations"]
        categories_mapping = coco_annotations["categories"]

        class_mapping = self._create_class_mapping_csv(categories_mapping)
        self._create_csv_annotations(image_info, annotations, class_mapping)

    def _create_csv_annotations(
        self, image_info: List[Dict], annotations: List[Dict], class_mapping: Dict
    ) -> None:
        """
        Convert COCO annotations to CSV format which is required by models
        such as EfficientDet

        Where the CSV file will be in this format:

        path/to/image.jpg,x1,y1,x2,y2,class_name

        where x1 = xmin, y1 = ymin, x2 = x max, y2 = ymax
        Args:
            image_info (List[Dict]): Image files meta info such as filepath, id
            annotations (List[Dict]): Information about the annotations such as
                                    bounding box info and id
            class_mapping (Dict): Class mapping for dataset
        """
        data_value = []
        image_info_dict = {}

        # Convert image info to a dict for faster lookup
        for image in image_info:
            image_info_dict[image["id"]] = PurePath(
                RAW_IMAGE_FOLDER, image["file_name"]
            )

        for annotation in annotations:
            annotation_image = image_info_dict[annotation["image_id"]]
            class_id = annotation["category_id"]
            class_name = class_mapping[class_id].lower()
            bbox_info = annotation["bbox"]
            x, y, width, height = bbox_info
            x1, y1, x2, y2 = self._convert_coco_bbox_to_efficientdet_format(
                x,
                y,
                width,
                height,
            )
            data_value.append(
                [
                    str(annotation_image).replace("\\", "/"),
                    x1,
                    y1,
                    x2,
                    y2,
                    class_name,
                ]
            )

        with open(
            PurePath(ANNOTATION_FOLDER, "annotations.csv"), "w", newline=""
        ) as csv_file:
            csv_writer = writer(csv_file, delimiter=",")
            for data in data_value:
                csv_writer.writerow(data)

    def _convert_coco_bbox_to_efficientdet_format(
        self, x: float, y: float, width: float, height: float
    ) -> Tuple[int]:
        """
        Convert COCO bbox annotation style to EfficientDet format which is :
        x1,y1,x2,y2

        x1, y1 = x min, y min
        x2, y2 = x max, y max
        Args:
            x (float): x coordinate of top left box
            y (float): y coordinate of top left box
            width (float): Width of bounding box
            height (float): Height of bounding box
        """
        x1, y1 = int(x), int(y)
        x2, y2 = int(x + width), int(y + height)

        return x1, y1, x2, y2

    def _create_class_mapping_csv(self, categories_mapping: List[Dict]) -> Dict:
        """
        Create a class mapping csv file to be used for models such as EfficientDet.
        The CSV file will be in this format:

        class_name,id

        Note that indexing starts at 0 for the id column

        Args:
            categories_mapping (List[Dict]): List of all the classes in the dataset
                                            and their meta info.

        Returns:
            class_dict (Dict): Class mapping table for the dataset
        """
        data_value = []
        class_dict = {}

        for i, category in enumerate(categories_mapping):
            data_value.append([category["name"].lower(), i])
            class_dict[i + 1] = category[
                "name"
            ]  # Convert from 0 indexing to 1 indexing

        with open(
            PurePath(ANNOTATION_FOLDER, "class_mapping.csv"), "w", newline=""
        ) as csv_file:
            csv_writer = writer(csv_file, delimiter=",")
            for data in data_value:
                csv_writer.writerow(data)
        return class_dict


if __name__ == "__main__":
    pipeline = EfficientDetPipeline()
    pipeline.convert_coco_to_csv()
