import pandas as pd
import os
import shutil

from .preprocess_efficientdet import EfficientDetPipeline
from pathlib import Path, PurePath
from csv import DictWriter, writer
from typing import Dict, List, Tuple
from pconst import const
from json import load

class Preprocessor:
    def __init__(self, logger):
        self.logger = logger
        self.processed_annotations_df = None
    
    def preprocess_annotations(self):
        self._convert_raw_annotations()
        self._copy_raw_images()

    def _get_raw_annotation_file_paths(self):
        annot_files = []
        for data_subdir in const.DATA_SUBDIRS_PATH_LIST:
            annot_path = Path(data_subdir, const.ANNOTATIONS_SUBDIR, const.COCO_ANNOTATION_FILENAME)
            if annot_path.exists():
                annot_files.append(annot_path)
        return annot_files

    def _load_coco_annotations(self, path_to_coco_annot: PurePath) -> Dict:
        """
        Load COCO annotations from the filepath provided

        Args:
            path_to_coco_annot (PurePath): File path to COCO annotations
        """
        with open(path_to_coco_annot, "r") as f:
            coco_annotation = load(f)

        return coco_annotation

    def _convert_raw_annotations(self, save_csv=True) -> None:
        """
        Combine & convert COCO annotation files into a dataframe

        Args:
            save_csv (boolean): 
        """
        df_concat_list = []
        for annot_file_path in self._get_raw_annotation_file_paths():
            coco_annotations = self._load_coco_annotations(annot_file_path)

            image_info = coco_annotations["images"]
            df_images = pd.DataFrame(image_info)
            df_images.set_index('id', inplace=True)
            df_images.drop(labels=['license','flickr_url','coco_url','date_captured'], axis=1, inplace=True)

            categories_mapping = coco_annotations["categories"]
            df_cat = pd.DataFrame(categories_mapping)
            df_cat.rename(mapper={'name':'category_name'}, axis=1, inplace=True)
            df_cat.drop(labels=['supercategory'], axis=1, inplace=True)

            annotations = coco_annotations["annotations"]
            self.logger.debug(f'Number of annotations: {len(annotations)}')
            df_annot = pd.DataFrame(annotations)
            df_annot.set_index('id', inplace=True)
            df_annot.drop(labels=['segmentation','iscrowd','attributes'], axis=1, inplace=True)

            final_df = df_annot.merge(df_images, left_on='image_id', right_on='id')
            final_df = final_df.merge(df_cat, left_on='category_id', right_on='id')
            final_df.drop(labels=['id'], axis=1, inplace=True)
            df_concat_list.append(final_df)

        concatenated_df = pd.concat(df_concat_list, ignore_index=True)
        concatenated_df[['bbox_x','bbox_y', 'bbox_width', 'bbox_height']] = pd.DataFrame(concatenated_df.bbox.tolist(), index=concatenated_df.index)
        concatenated_df.drop('bbox', axis=1, inplace=True)
        if save_csv:
            annot_processed_path = PurePath(const.PROCESSED_DATA_PATH, "annotations_all.csv")
            concatenated_df.to_csv(annot_processed_path)
            self.logger.info(f'Annotations saved to {annot_processed_path}')
        self.processed_annotations_df = concatenated_df.copy()

        # class_mapping = self._generate_class_mapping(categories_mapping)

    def _create_class_mapping_csv(self, annotation_dir_path: PurePath) -> None:
        with open(
                PurePath(annotation_dir_path, "class_mapping.csv"), "w", newline=""
            ) as csv_file:
                csv_writer = writer(csv_file, delimiter=",")
                for data in data_value:
                    csv_writer.writerow(data)

    def _generate_class_mapping(self, categories_mapping: List[Dict], save_csv=False) -> Dict:
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

        if save_csv:
            self._create_class_mapping_csv()
        return class_dict

    def _copy_raw_images(self) -> None:
         for data_subdir in const.DATA_SUBDIRS_PATH_LIST:
            img_src_dir_path = PurePath(data_subdir, const.IMAGES_SUBDIR)
            self.logger.debug(f"Copying images from {img_src_dir_path}")
            for filename in os.listdir(img_src_dir_path):
                self.logger.debug(f"Destination file path: {Path(const.RAW_DATA_PATH, filename)}")
                if filename not in const.EXCLUDED_IMAGES and not Path(const.RAW_DATA_PATH, filename).exists():
                    try:
                        shutil.copy(PurePath(img_src_dir_path, filename), const.RAW_DATA_PATH)
                    except OSError as e:
                        self.logger.error(f"Error occurred while copying file: {e}")

    def generate_image_tiles(self):
        pass

    def split_data(self, save_csv=False):
        pass
    
    def preprocess_efficientdet(self):
        ed_pipeline = EfficientDetPipeline()
        ed_pipeline.generate_annotations()
