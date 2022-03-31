import hydra
from pconst import const
from collections import OrderedDict


class PipelineConfig:
    """
    This class initialises system-wide constant variables based on the key-value pairs provided in the pipelines.yml config file.
    These constants cannot be modified outside of this class.
    """

    def __init__(self, params, logger):
        """This method takes in parameters from the config file and initialises them as constants.

        Args:
            params (dict): Parameters read from the config file
            logger (logging.Logger): Logger object used for logging
        """
        logger.debug(f"Loading params into constants: {params}")

        # Initialise data pipeline constants
        if "data_prep" in params:
            data_prep_params = params["data_prep"]

            const.DATA_SUBDIRS_PATH_LIST = data_prep_params["data_subdirs_paths"]
            const.PROCESSED_DATA_PATH = data_prep_params["processed_data_path"]
            const.RAW_DATA_PATH = data_prep_params["raw_data_path"]
            const.LOAD_DATA = data_prep_params["load_data"]
            const.MODELS = data_prep_params["models"]
            const.ANNOTATIONS_SUBDIR = data_prep_params["annotations_subdir"]
            const.IMAGES_SUBDIR = data_prep_params["images_subdir"]
            const.COCO_ANNOTATION_FILENAME = data_prep_params[
                "coco_annotations_filename"
            ]
            const.CLASS_MAP = data_prep_params["class_map"]
            const.REMAP_CLASS = data_prep_params["remap_class"]
            const.EXCLUDED_IMAGES = data_prep_params["excluded_images"]
            const.COMBINED_ANNOTATIONS_FILENAME = data_prep_params[
                "combined_annotations_filename"
            ]
            const.ACCEPTED_IMAGE_FORMATS = data_prep_params["accepted_image_formats"]
            # Tile/slice processed images
            const.TILE_DATA_DIR_PATHS = data_prep_params["tile_data_dir_paths"]
            const.TILE_SLICE_HEIGHT = data_prep_params["tile_slice_height"]
            const.TILE_SLICE_WIDTH = data_prep_params["tile_slice_width"]
            const.TILE_OVERLAP_HEIGHT_RATIO = data_prep_params["tile_overlap_height_ratio"]
            const.TILE_OVERLAP_WIDTH_RATIO = data_prep_params["tile_overlap_width_ratio"]

        # Initialise constants for training all models
        if "train" in params:
            train_params = params["train"]

        # Initialise constants for augmentation pipeline within training
        if "train_augmentation" in params:
            train_augment_params = params["train_augmentation"]

        # Initialise constants for model inference
        if "inference" in params:
            train_params = params["inference"]

            const.INFERENCE_MODEL_NAME = train_params["model_name"]
            const.INFERENCE_INPUT_DIR = train_params["input_data_dir"]
            const.INFERENCE_SAVE_OUTPUT = train_params["save_output_image"]
            const.INFERENCE_OUTPUT_PATH = train_params["inference_output_path"]
            const.INFERENCE_MODE = train_params["inference_mode"]
