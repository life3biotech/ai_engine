import pandas as pd
import numpy as np
import os
import shutil
import skimage.io

from pathlib import Path, PurePath
from csv import DictWriter, writer
from typing import Dict, List, Tuple
from pconst import const
from json import load
from sklearn.model_selection import train_test_split

from . import coco_filter as coco_filter
from src.sahi.slicing import slice_coco
from src.sahi.utils.coco import Coco


class Preprocessor:
    """This class encapsulates the data preprocessing functionality that forms part of the data pipeline.

    Attributes:
        logger: Logger object used to log events to.
        processed_annotations_df: pandas DataFrame containing the final processed data.
        seed: Seed to use for data splitting function
    """

    def __init__(self, logger):
        self.logger = logger
        self.processed_annotations_df = None
        self.seed = 33

    def preprocess_annotations(self) -> None:
        """
        Calls the data preprocessing functions in order and saves the final preprocessed data in CSV format.
        """
        self._make_dir(const.PROCESSED_DATA_PATH)
        self._make_dir(const.RAW_DATA_PATH)
        self._make_dir(const.INTERIM_DATA_PATH)

        self._copy_raw_images(const.RUN_TILING)
        df_concat_list = self._convert_raw_annotations(const.RUN_TILING)

        concatenated_df = pd.concat(df_concat_list, ignore_index=True)
        concatenated_df = self._encode_classes(concatenated_df)
        concatenated_df = self._engineer_features(concatenated_df)
        concatenated_df = self._clean_data(concatenated_df)
        concatenated_df = self._add_metadata(concatenated_df)

        annot_processed_path = PurePath(
            const.INTERIM_DATA_PATH, const.COMBINED_ANNOTATIONS_FILENAME
        )
        concatenated_df.to_csv(annot_processed_path)
        self.logger.info(f"Annotations saved to {annot_processed_path}")
        self.processed_annotations_df = concatenated_df
        return concatenated_df

        """
 
        Returns:
            A list of COCO annotation file paths.
        """

    def _get_raw_annotation_file_paths(self, tile_ok: bool) -> List:
        """
        Traverses the list of data subdirectories specified in the config file and validates the directory structure of each one.

        Args:
            tile_ok (bool): True = Enable image tile function

        Returns:
            List: A list of COCO annotation file paths.
        """
        annot_files = []
        for data_subdir in const.DATA_SUBDIRS_PATH_LIST:

            if tile_ok:
                # Tile data path
                orig_folder = os.path.basename(os.path.normpath(data_subdir))
                annot_path = Path(
                    const.TILE_DATA_DIR_PATHS,
                    orig_folder,
                    const.ANNOTATIONS_SUBDIR,
                    const.COCO_ANNOTATION_FILENAME,
                )
                img_path = Path(
                    const.TILE_DATA_DIR_PATHS, orig_folder, const.IMAGES_SUBDIR
                )
            else:
                annot_path = Path(
                    data_subdir,
                    const.ANNOTATIONS_SUBDIR,
                    const.COCO_ANNOTATION_FILENAME,
                )
                img_path = Path(data_subdir, const.IMAGES_SUBDIR)

            if (
                annot_path.exists() and img_path.exists() and img_path.is_dir()
            ):  # check if both images & annotations subdirectories exist
                annot_files.append(annot_path)
            else:
                if tile_ok:
                    self.logger.error(
                        f"Invalid directory structure in {Path(const.TILE_DATA_DIR_PATHS,orig_folder)}. One of these subdirectories are missing: /images, /annotations"
                    )
                else:
                    self.logger.error(
                        f"Invalid directory structure in {data_subdir}. One of these subdirectories are missing: /images, /annotations"
                    )
        return annot_files

    def _load_coco_annotations(self, path_to_coco_annot: PurePath) -> Dict:
        """
        Loads COCO annotations from the file path provided.

        Args:
            path_to_coco_annot (PurePath): File path to COCO annotations file

        Returns:
            A dictionary representation of the COCO annotations file.
        """
        with open(path_to_coco_annot, "r") as f:
            coco_annotation = load(f)

        return coco_annotation

    def _convert_raw_annotations(self, tile_ok: bool) -> List:
        """
        Converts & combines COCO annotations into pandas DataFrames while removing unnecessary metadata.

        Returns:
            A list of DataFrames, with each dataframe containing data from one COCO annotation file.
        """
        df_concat_list = []
        for annot_file_path in self._get_raw_annotation_file_paths(tile_ok):
            coco_annotations = self._load_coco_annotations(annot_file_path)

            image_info = coco_annotations["images"]
            df_images = pd.DataFrame(image_info)
            df_images.set_index("id", inplace=True)
            if not const.RUN_TILING:
                df_images.drop(
                    labels=["license", "flickr_url", "coco_url", "date_captured"],
                    axis=1,
                    inplace=True,
                )

            categories_mapping = coco_annotations["categories"]
            df_cat = pd.DataFrame(categories_mapping)
            df_cat.rename(mapper={"name": "category_name"}, axis=1, inplace=True)
            df_cat.drop(labels=["supercategory"], axis=1, inplace=True)

            annotations = coco_annotations["annotations"]
            self.logger.debug(f"Number of annotations: {len(annotations)}")
            df_annot = pd.DataFrame(annotations)
            df_annot.set_index("id", inplace=True)
            if not const.RUN_TILING:
                df_annot.drop(
                    labels=["segmentation", "iscrowd", "attributes"],
                    axis=1,
                    inplace=True,
                )
            else:
                df_annot.drop(
                    labels=["segmentation", "iscrowd"],
                    axis=1,
                    inplace=True,
                )

            final_df = df_annot.merge(df_images, left_on="image_id", right_on="id")
            final_df = final_df.merge(df_cat, left_on="category_id", right_on="id")
            final_df.drop(labels=["id"], axis=1, inplace=True)
            df_concat_list.append(final_df)
        return df_concat_list

    def _encode_classes(self, concatenated_df: pd.DataFrame) -> pd.DataFrame:
        """
        Encodes category/labels into integer values for each annotation in the provided DataFrame.

        Args:
            concatenated_df (pd.DataFrame): DataFrame containing all annotations in the dataset

        Returns:
            A DataFrame containing the resulting annotations data.
        """
        df = concatenated_df.copy()
        if const.REMAP_CLASSES:
            self.logger.info(f"Remapping class labels: {const.CLASS_REMAPPING}")
            df["category_name"] = df["category_name"].map(const.CLASS_REMAPPING)

        self.logger.info("Encoding class labels to target column")
        df["encoded_target"] = df["category_name"].map(const.CLASS_MAP)
        df.drop(labels=["category_id"], axis=1, inplace=True)
        return df

    def _clean_data(self, concatenated_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calls the data cleaning functions in order.

        Args:
            concatenated_df (pd.DataFrame): DataFrame containing all annotations in the dataset

        Returns:
            A DataFrame containing the cleaned data.
        """
        df = concatenated_df.copy()

        images_list = [
            f
            for f in os.listdir(const.RAW_DATA_PATH)
            if f.endswith(tuple(const.ACCEPTED_IMAGE_FORMATS))
        ]
        unique_images_list = df["file_name"].unique()

        df = self._clean_annotations(df, images_list, unique_images_list)

        df = self._clean_images(df, unique_images_list)

        df = self._clean_class_labels(df)

        return df

    def _clean_annotations(
        self, concatenated_df: pd.DataFrame, images_list: List, unique_images_list: List
    ) -> pd.DataFrame:
        """
        Cleans annotations in the provided DataFrame.

        Args:
            concatenated_df (pd.DataFrame): DataFrame containing all annotations in the dataset
            images_list (List): List containing image file names found in the raw data directory
            unique_images_list (List): List containing unique image file names from the annotations data

        Returns:
            A DataFrame containing the cleaned annotations data.
        """
        df = concatenated_df.copy()
        self.logger.debug(f"**** Length of df before excluding images: {df.shape}")
        df = df[~df["file_name"].isin(const.EXCLUDED_IMAGES)]
        self.logger.info(
            f"Removing annotations of predefined excluded images: {const.EXCLUDED_IMAGES}"
        )
        self.logger.debug(f"**** Length of df after excluding images: {df.shape}")
        annotations_no_images = list(set(unique_images_list) - set(images_list))
        self.logger.warning(
            f"Number of annotations with no corresponding images: {len(annotations_no_images)}"
        )
        images_no_annotations = list(set(images_list) - set(unique_images_list))
        # log as a warning; image file will be ignored
        self.logger.warning(
            f"Number of images with no corresponding annotations: {len(images_no_annotations)}"
        )

        if len(annotations_no_images) > 0:
            df = df[~df["file_name"].isin(annotations_no_images)]
            self.logger.info("Removed annotations with no corresponding images")

        invalid_annot = df[
            (df["bbox_x_min"] > df["width"])
            | (df["bbox_x_min"] < 0)
            | (df["bbox_x_max"] > df["width"])
            | (df["bbox_x_max"] < 0)
            | (df["bbox_y_min"] > df["height"])
            | (df["bbox_y_min"] < 0)
            | (df["bbox_y_max"] > df["height"])
            | (df["bbox_y_max"] < 0)
        ]
        if len(invalid_annot) > 0:
            df.drop(labels=invalid_annot.index, inplace=True)
            self.logger.warning(f"Removed {len(invalid_annot)} invalid annotations")
        else:
            self.logger.info("All annotations are valid")

        return df

    def _clean_images(
        self, concatenated_df: pd.DataFrame, unique_images_list: List
    ) -> pd.DataFrame:
        """
        Checks validity of image files found in the provided DataFrame.

        Args:
            concatenated_df (pd.DataFrame): DataFrame containing all annotations in the dataset
            unique_images_list (List): List containing unique image file names from the annotations data

        Returns:
            A DataFrame containing cleaned annotations data after removal of invalid images.
        """
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
                row = df[df["file_name"] == image_filename].iloc[0]
                if row["width"] != width or row["height"] != height:
                    self.logger.warning(
                        f"Dimensions of actual image do not match metadata: {image_filepath}"
                    )
        df = df[~df["file_name"].isin(invalid_image_filenames)]
        return df

    def _validate_image(self, image_filepath: Path) -> Tuple:
        """
        Checks validity of image file at the given file path by attempting to read in image data using the skimage library.

        Args:
            image_filepath (Path): Full path where the image file is located

        Returns:
            A tuple containing the height & width of the image at the given file path if it is valid.
            If it is an invalid image, a tuple of zeroes (0,0) is returned.
        """
        width = 0
        height = 0
        try:
            img = skimage.io.imread(image_filepath)
        except:
            return height, width
        height, width, _ = img.shape
        return height, width

    def _clean_class_labels(self, concatenated_df: pd.DataFrame) -> pd.DataFrame:
        """
        Checks validity of class labels found in the provided DataFrame against the class map defined in the config file.

        Args:
            concatenated_df (pd.DataFrame): DataFrame containing all annotations in the dataset

        Returns:
            A DataFrame containing cleaned annotations data after removal of annotations labelled with invalid classes, if any.
        """
        df = concatenated_df.copy()
        self.logger.info("Checking class labels")
        unique_labels = df["category_name"].unique()
        labels_to_remove = list(set(unique_labels) - set(const.CLASS_MAP.keys()))
        if len(labels_to_remove) > 0:
            self.logger.warning(f"Invalid class labels found: {labels_to_remove}")
            # Remove annotations with invalid class labels
            df = df[~df["category_name"].isin(labels_to_remove)]
        return df

    def _engineer_features(self, concatenated_df: pd.DataFrame) -> pd.DataFrame:
        """
        Creates additional features based on existing features in the dataset.

        Args:
            concatenated_df (pd.DataFrame): DataFrame containing all annotations in the dataset

        Returns:
            A DataFrame containing annotations data with newly engineered features.
        """
        df = concatenated_df.copy()
        df[["bbox_x_min", "bbox_y_min", "bbox_width", "bbox_height"]] = pd.DataFrame(
            df.bbox.tolist(), index=df.index
        )
        df["bbox_x_max"] = round(df["bbox_x_min"] + df["bbox_width"], 2)
        df["bbox_y_max"] = round(df["bbox_y_min"] + df["bbox_height"], 2)
        df.drop("bbox", axis=1, inplace=True)
        return df

    def _copy_raw_images(self, tile_ok: bool) -> None:
        """
        Copies all image files from the data subdirectories into the raw data directory, except the excluded files specified in the config.

        Args:
            tile_ok (bool): True = Enable image tile function
        """
        for data_subdir in const.DATA_SUBDIRS_PATH_LIST:
            if tile_ok:
                # Tile data path
                orig_folder = os.path.basename(os.path.normpath(data_subdir))
                img_src_dir_path = Path(
                    const.TILE_DATA_DIR_PATHS, orig_folder, const.IMAGES_SUBDIR
                )
            else:
                img_src_dir_path = PurePath(data_subdir, const.IMAGES_SUBDIR)

            self.logger.info(f"Copying images from {img_src_dir_path}")
            for filename in os.listdir(img_src_dir_path):
                self.logger.debug(
                    f"Destination file path: {Path(const.RAW_DATA_PATH, filename)}"
                )
                if (
                    filename not in const.EXCLUDED_IMAGES
                    and not Path(const.RAW_DATA_PATH, filename).exists()
                    and Path(img_src_dir_path, filename).is_file()
                ):
                    try:
                        shutil.copy(
                            PurePath(img_src_dir_path, filename), const.RAW_DATA_PATH
                        )
                    except OSError as e:
                        self.logger.error(f"Error occurred while copying file: {e}")

    def _make_dir(self, dir_path: Path) -> None:
        """
        Create directory if it did not exist

        Args:
            dir_path (Path): Full path to create new directory
        """
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    def _add_metadata(self, concatenated_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add metadata 'incubation_day', 'dilution_factor' and 'optical_density' for stratification

        Args:
            concatenated_df (pd.DataFrame): DataFrame containing all annotations in the dataset

        Returns:
            pd.DataFrame: Appended metadata to annotations dataframe.
        """
        df = concatenated_df.copy()
        self.logger.info("Add metadata to treated dataframe")

        df_metadata = pd.read_excel(const.META_DATA_FILENAME, engine="openpyxl")
        df_metadata["Filename"] = df_metadata["Filename"].astype(str)
        df_metadata["file_type"] = df_metadata.Filename.str.rsplit(
            ".", n=1, expand=True
        )[1]
        # Remove extension from filename
        df_metadata["Filename"] = df_metadata.Filename.str.rsplit(
            ".", n=1, expand=True
        )[0]
        df_metadata.replace({"undiluted": 0, np.nan: 0}, inplace=True)
        df_metadata.rename(
            columns={
                "Incubation day": "incubation_day",
                "Dilution Factor (unit)": "dilution_factor",
                "Optical Density": "optical_density",
            },
            inplace=True,
        )
        selected_col = [
            "Filename",
            "incubation_day",
            "dilution_factor",
            "optical_density",
        ]

        if const.RUN_TILING:
            df["file_name_split"] = df.file_name.str.rsplit("_", n=4, expand=True)[0]
        else:
            df["file_name_split"] = df.file_name.str.rsplit(".", n=1, expand=True)[0]

        merged_df = pd.merge(
            left=df,
            right=df_metadata[selected_col],
            how="left",
            left_on="file_name_split",
            right_on="Filename",
        )

        # TODO Bin optical_density numerical value according to quartile value (KIV)
        # q1, q2, q3, q4 = (
        #     merged_df["optical_density"].quantile(0),
        #     merged_df["optical_density"].quantile(0.25),
        #     merged_df["optical_density"].quantile(0.5),
        #     merged_df["optical_density"].quantile(0.75),
        # )
        # merged_df["optical_density_bin"] = (
        #     pd.cut(
        #         merged_df.optical_density,
        #         bins=[q1, q2, q3, q4, np.inf],
        #         labels=False,
        #         right=False,
        #     )
        #     + 1
        # )

        # Create multi-column for stratification
        # incubation_day + dilution_factor
        merged_df["incub_day_dilu_fact"] = (
            merged_df["incubation_day"].astype(str)
            + "_"
            + merged_df["dilution_factor"].astype(str)
        )
        # # TODO incubation_day + dilution_factor + optical_density_bin
        # merged_df["incub_day_dilu_fact_optical_bin"] = (
        #     merged_df["incubation_day"].astype(str)
        #     + "_"
        #     + merged_df["dilution_factor"].astype(str)
        #     + "_"
        #     + merged_df["optical_density_bin"].astype(str)
        # )

        # Drop unneccessary column
        merged_df = merged_df.drop(columns=["file_name_split", "Filename"])

        return merged_df

    def generate_image_tiles(self) -> None:
        """
        Slice/Tile COCO annotation and image files and save to tile processed directory,
        keeping the same list of data subdirectories specified in the config file.
        """
        self.logger.info("Tile/Slice of images processing...")

        cocofilter = coco_filter.CocoFilter(self.logger)

        for data_subdir in const.DATA_SUBDIRS_PATH_LIST:
            # Original data path
            annot_path = Path(
                data_subdir,
                const.ANNOTATIONS_SUBDIR,
                const.COCO_ANNOTATION_FILENAME,
            )
            img_path = Path(data_subdir, const.IMAGES_SUBDIR)

            # Tile data path
            orig_folder = os.path.basename(os.path.normpath(data_subdir))
            before_tile_filter_annot_path = Path(
                const.TILE_DATA_DIR_PATHS,
                orig_folder,
                const.ANNOTATIONS_SUBDIR,
            )
            before_tile_filter_annot_filepath = Path(
                const.TILE_DATA_DIR_PATHS,
                orig_folder,
                const.ANNOTATIONS_SUBDIR,
                "before_tile_filter_annot.json",
            )
            tile_annot_path = Path(
                const.TILE_DATA_DIR_PATHS,
                orig_folder,
                const.ANNOTATIONS_SUBDIR,
                const.COCO_ANNOTATION_FILENAME,
            )
            tile_img_path = Path(
                const.TILE_DATA_DIR_PATHS, orig_folder, const.IMAGES_SUBDIR
            )

            # Filter, treatment, exclude images of coco json file
            self._make_dir(before_tile_filter_annot_path)
            cocofilter.filter_coco(
                annot_path,
                before_tile_filter_annot_filepath,
            )

            # Display coco stats before image tile
            coco = Coco.from_coco_dict_or_path(str(before_tile_filter_annot_filepath))
            self.logger.info(f"Coco path: {before_tile_filter_annot_filepath}")
            self.logger.info(f"Coco Stat: {coco.stats}")

            # Tile/slice image
            slice_coco(
                coco_annotation_file_path=before_tile_filter_annot_filepath,
                image_dir=img_path,
                output_coco_annotation_file_name=tile_annot_path,
                ignore_negative_samples=const.TILE_IGNORE_NEGATIVE_SAMPLES,
                output_dir=tile_img_path,
                slice_height=const.TILE_SLICE_HEIGHT,
                slice_width=const.TILE_SLICE_WIDTH,
                overlap_height_ratio=const.TILE_OVERLAP_HEIGHT_RATIO,
                overlap_width_ratio=const.TILE_OVERLAP_WIDTH_RATIO,
                verbose=False,
            )

        self.logger.info("Tile/Slice of images completed.")

    def split_data(self, concatenated_df: pd.DataFrame, test_size=0.2, val_size=0.1):
        """This function uses scikit-learn library to split the dataframe into train, test & validation sets
        and saves the split data to CSV files (dependent on configuration). If the value(s) of `test_size` and/or `val_size` are not provided,
        the function splits the data according to the default values.
        If column name is specify for stratify parameter in pipelines.yml, data will be split in a stratified fashion.
        Column name options are derived from metadata - 'incubation_day' or 'dilution_factor' or 'incub_day_dilu_fact'

        Args:
            concatenated_df (DataFrame): Pandas DataFrame containing cleaned and processed data
            test_size: Proportion of the dataset to include in the test split. Defaults to 0.2 (20%)
            val_size: Proportion of the train dataset tols - include in the validation split. Defaults to 0.1 (10%)
        Returns:
            X_train, y_train, X_test, y_test, X_val, y_val: Tuple containing split datasets
        """
        df = concatenated_df.copy()

        # Calculate the proportion of validation size as of the (1 - test size) because in the codes, validation split occurs after test split
        val_actual_size = val_size / (1 - test_size)

        split_var = "file_name"

        # Split without stratification
        if const.STRATIFY_COLUMN == "None":
            print("check const.STRATIFY_COLUMN done: ", const.STRATIFY_COLUMN)
            split_array = df["file_name"].unique()

            # Split images into train & test sets
            train, test = train_test_split(
                split_array,
                test_size=test_size,
                random_state=self.seed,
            )

            # Split train images further into train & validation
            train, val = train_test_split(
                train,
                test_size=val_actual_size,
                random_state=self.seed,
            )

            # Retrieve annotations belonging to images in each dataset
            X_train = df[df[split_var].isin(train)]
            images_train = X_train["file_name"].unique()
            y_train = X_train[const.TARGET_COL]
            train_defects_count = X_train[const.TARGET_COL].value_counts()

            X_val = df[df[split_var].isin(val)]
            images_val = X_val["file_name"].unique()
            y_val = X_val[const.TARGET_COL]
            val_defects_count = X_val[const.TARGET_COL].value_counts()

            X_test = df[df[split_var].isin(test)]
            images_test = X_test["file_name"].unique()
            y_test = X_test[const.TARGET_COL]
            test_defects_count = X_test[const.TARGET_COL].value_counts()

        else:  # Split with stratification
            print("check const.STRATIFY_COLUMN fail: ", const.STRATIFY_COLUMN)
            split_array = df[["file_name", const.STRATIFY_COLUMN]]
            split_array = split_array.drop_duplicates()

            # Split images into train & test sets
            train, test = train_test_split(
                split_array,
                test_size=test_size,
                random_state=self.seed,
                stratify=split_array[const.STRATIFY_COLUMN],
            )

            self.logger.info(
                f"1st Train stratify: {train[const.STRATIFY_COLUMN].value_counts(normalize=True) * 100} "
            )
            self.logger.info(
                f"1st Test stratify: {test[const.STRATIFY_COLUMN].value_counts(normalize=True) * 100} "
            )
            # Split train images further into train & validation
            train, val = train_test_split(
                train,
                test_size=val_actual_size,
                random_state=self.seed,
                stratify=train[const.STRATIFY_COLUMN],
            )

            self.logger.info(
                f" 2nd train: {train[const.STRATIFY_COLUMN].value_counts(normalize=True) * 100} "
            )
            self.logger.info(
                f"2nd val: {val[const.STRATIFY_COLUMN].value_counts(normalize=True) * 100} ",
            )

            # Retrieve annotations belonging to images in each dataset
            X_train = df[df[split_var].isin(train[split_var])]
            images_train = X_train["file_name"].unique()
            y_train = X_train[const.TARGET_COL]
            train_defects_count = X_train[const.TARGET_COL].value_counts()

            X_val = df[df[split_var].isin(val[split_var])]
            images_val = X_val["file_name"].unique()
            y_val = X_val[const.TARGET_COL]
            val_defects_count = X_val[const.TARGET_COL].value_counts()

            X_test = df[df[split_var].isin(test[split_var])]
            images_test = X_test["file_name"].unique()
            y_test = X_test[const.TARGET_COL]
            test_defects_count = X_test[const.TARGET_COL].value_counts()

        self.logger.info(f"Number of images in train: {len(images_train)}")
        self.logger.info(f"Number of images in validation: {len(images_val)}")
        self.logger.info(f"Number of images in test: {len(images_test)}")

        self.logger.info(f"Number of annotations in train set: {X_train.shape[0]}")
        self.logger.info(f"Number of annotations in validation set: {X_val.shape[0]}")
        self.logger.info(f"Number of annotations in test set: {X_test.shape[0]}")

        if const.SAVE_DATA_SPLITS:
            X_train.to_csv(
                PurePath(const.INTERIM_DATA_PATH, const.TRAIN_SET_FILENAME), index=False
            )
            X_val.to_csv(
                PurePath(const.INTERIM_DATA_PATH, const.VAL_SET_FILENAME), index=False
            )
            X_test.to_csv(
                PurePath(const.INTERIM_DATA_PATH, const.TEST_SET_FILENAME), index=False
            )
            self.logger.info("Saved train/val/test datasets to files")

        return X_train, y_train, X_test, y_test, X_val, y_val
