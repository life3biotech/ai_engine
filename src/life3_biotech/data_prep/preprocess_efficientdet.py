import os
from pathlib import Path, PurePath
from typing import Dict, List, Tuple
import pandas as pd
from json import load
from collections import defaultdict
from csv import DictWriter, writer


class EfficientDetPipeline:

    def generate_annotations(self) -> None:
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
            PurePath(const.PROCESSED_DATA_PATH, "annotations.csv"), "w", newline=""
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


if __name__ == "__main__":
    pipeline = EfficientDetPipeline()
    pipeline.generate_annotations()
