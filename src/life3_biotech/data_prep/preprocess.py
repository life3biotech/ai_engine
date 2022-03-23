import pandas as pd
import os
import shutil
import skimage

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
        self._copy_raw_images()
        df_concat_list = self._convert_raw_annotations()
        
        concatenated_df = pd.concat(df_concat_list, ignore_index=True)
        concatenated_df = self._engineer_features(concatenated_df)
        concatenated_df = self._clean_data(concatenated_df)

        annot_processed_path = PurePath(const.PROCESSED_DATA_PATH, const.COMBINED_ANNOTATIONS_FILENAME)
        concatenated_df.to_csv(annot_processed_path)
        self.logger.info(f'Annotations saved to {annot_processed_path}')
        self.processed_annotations_df = concatenated_df.copy()

    def _get_raw_annotation_file_paths(self):
        annot_files = []
        for data_subdir in const.DATA_SUBDIRS_PATH_LIST:
            annot_path = Path(data_subdir, const.ANNOTATIONS_SUBDIR, const.COCO_ANNOTATION_FILENAME)
            img_path = Path(data_subdir, const.IMAGES_SUBDIR)
            if annot_path.exists() and img_path.exists() and img_path.is_dir(): # check if both images & annotations subdirectories exist
                annot_files.append(annot_path)
            else:
                self.logger.error(f"Invalid directory structure in {data_subdir}. One of these subdirectories are missing: /images, /annotations")
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

    def _convert_raw_annotations(self) -> List:
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
        return df_concat_list

    def _clean_data(self, concatenated_df: pd.DataFrame) -> pd.DataFrame:
        df = concatenated_df.copy()

        images_list = [f for f in os.listdir(const.RAW_DATA_PATH) if f.endswith(tuple(const.ACCEPTED_IMAGE_FORMATS))]
        unique_images_list = df['file_name'].unique()

        df = self._clean_annotations(df, images_list, unique_images_list) 

        df = self._clean_images(df, unique_images_list)

        df = self._clean_class_labels(df)

        return df

    def _clean_annotations(self, concatenated_df: pd.DataFrame, images_list: List, unique_images_list: List) -> pd.DataFrame:
        df = concatenated_df.copy()

        self.logger.info(f"Removing annotations of predefined excluded images: {const.EXCLUDED_IMAGES}")
        df = df[~df['file_name'].isin(const.EXCLUDED_IMAGES)]

        annotations_no_images = list(set(unique_images_list) - set(images_list))
        self.logger.warning(f'Number of annotations with no corresponding images: {len(annotations_no_images)}')
        images_no_annotations = list(set(images_list) - set(unique_images_list))
        # log as a warning; image file will be ignored
        self.logger.warning(f'Number of images with no corresponding annotations: {len(images_no_annotations)}')

        if len(annotations_no_images) > 0:
            self.logger.info("Removing annotations with no corresponding images")
            df = df[~df['file_name'].isin(annotations_no_images)]

        invalid_annot_idx = df[
            (df['bbox_x_min'] > df['width']) | 
            (df['bbox_x_min'] < 0) | 
            (df['bbox_x_max'] > df['width']) | 
            (df['bbox_x_max'] < 0) | 
            (df['bbox_y_min'] > df['height']) | 
            (df['bbox_y_min'] < 0) | 
            (df['bbox_y_max'] > df['height']) | 
            (df['bbox_y_max'] < 0)
            ].index
        if len(invalid_annot_idx) > 0:
            self.logger.warning(f'Removing invalid annotations: {invalid_annot_idx}')
            df.drop(labels=invalid_annot_idx, inplace=True)
        else:
            self.logger.info("All annotations are valid")

        return df

    def _clean_images(self, concatenated_df: pd.DataFrame, unique_images_list: List) -> pd.DataFrame:
        df = concatenated_df.copy()
        self.logger.info("Checking & cleaning image files")
        invalid_image_filenames = []
        for image_filename in unique_images_list:
            image_filepath = Path(const.RAW_DATA_PATH, image_filename)
            height, width = self._validate_image(image_filepath)
            if width == 0 or height == 0:
                self.logger.warning(f"Corrupted image found: {image_filepath}")
                invalid_image_filenames.append(image_filename)
            else:
                row = df[df['file_name']==image_filename].iloc[0]
                if row['width'] != width or row['height'] != height:
                    self.logger.warning(f"Dimensions of actual image do not match metadata: {image_filepath}")
        df = df[~df['file_name'].isin(invalid_image_filenames)]
        return df

    def _validate_image(self, image_filepath):
        width = 0
        height = 0
        try:
            img = skimage.io.imread(image_filepath)
        except:
            return height, width
        height, width, _ = img.shape
        return height, width

    def _clean_class_labels(self, concatenated_df: pd.DataFrame):
        df = concatenated_df.copy()
        self.logger.info("Checking class labels")
        unique_labels = df['category_name'].unique()
        labels_to_remove = list(set(unique_labels) - set(const.CLASS_MAP.keys()))
        if len(labels_to_remove) > 0:
            self.logger.warning(f'Invalid class labels found: {labels_to_remove}')
            # Remove annotations with invalid class labels
            df = df[~df['category_name'].isin(labels_to_remove)]
        return df

    def _engineer_features(self, concatenated_df: pd.DataFrame) -> pd.DataFrame:
        df = concatenated_df.copy()
        df[['bbox_x_min','bbox_y_min', 'bbox_width', 'bbox_height']] = pd.DataFrame(df.bbox.tolist(), index=df.index)
        df['bbox_x_max'] = df['bbox_x_min'] + df['bbox_width']
        df['bbox_y_max'] = df['bbox_y_min'] + df['bbox_height']
        df.drop('bbox', axis=1, inplace=True)
        return df

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
