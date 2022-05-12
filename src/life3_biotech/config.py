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
            logger.info("Initialising data pipeline constants")
            data_prep_params = params["data_prep"]

            const.DATA_SUBDIRS_PATH_LIST = data_prep_params["data_subdirs_paths"]
            const.PROCESSED_DATA_PATH = data_prep_params["processed_data_path"]
            const.INTERIM_DATA_PATH = data_prep_params["interim_data_path"]
            const.RAW_DATA_PATH = data_prep_params["raw_data_path"]
            const.MODELS = data_prep_params["models"]
            const.ANNOTATIONS_SUBDIR = data_prep_params["annotations_subdir"]
            const.IMAGES_SUBDIR = data_prep_params["images_subdir"]
            const.COCO_ANNOTATION_FILENAME = data_prep_params[
                "coco_annotations_filename"
            ]
            const.CLASS_MAP = data_prep_params["class_map"]
            const.CLASS_MAP_REVERSE = {v: k for k, v in const.CLASS_MAP.items()}
            const.REMAP_CLASSES = data_prep_params["remap_classes"]
            const.CLASS_REMAPPING = data_prep_params["class_remapping"]
            const.EXCLUDED_IMAGES = data_prep_params["excluded_images"]
            const.COMBINED_ANNOTATIONS_FILENAME = data_prep_params[
                "combined_annotations_filename"
            ]
            const.ACCEPTED_IMAGE_FORMATS = data_prep_params["accepted_image_formats"]
            # Tile/slice processed images
            const.RUN_TILING = data_prep_params["run_tiling"]
            const.TILE_COCO_FILTER_CATEGORIES = data_prep_params[
                "tile_coco_filter_categories"
            ]
            const.TILE_DATA_DIR_PATHS = data_prep_params["tile_data_dir_paths"]
            const.TILE_SLICE_HEIGHT = data_prep_params["tile_slice_height"]
            const.TILE_SLICE_WIDTH = data_prep_params["tile_slice_width"]
            const.TILE_OVERLAP_HEIGHT_RATIO = data_prep_params[
                "tile_overlap_height_ratio"
            ]
            const.TILE_OVERLAP_WIDTH_RATIO = data_prep_params[
                "tile_overlap_width_ratio"
            ]
            const.TILE_IGNORE_NEGATIVE_SAMPLES = data_prep_params[
                "tile_ignore_negative_samples"
            ]

            const.TARGET_COL = data_prep_params["target_col"]
            const.SAVE_DATA_SPLITS = data_prep_params["save_data_splits"]
            const.VAL_SIZE = data_prep_params["val_size"]
            const.TEST_SIZE = data_prep_params["test_size"]
            const.TRAIN_SET_FILENAME = data_prep_params["train_base_filename"]
            const.VAL_SET_FILENAME = data_prep_params["validation_base_filename"]
            const.TEST_SET_FILENAME = data_prep_params["test_base_filename"]
            const.META_DATA_FILENAME = data_prep_params["meta_data_filename"]
            const.STRATIFY_COLUMN = data_prep_params["stratify_column"]

        # Initialise constants for training all models
        if "train" in params:
            train_params = params["train"]
            logger.info("Initialising training constants")
            const.LOAD_DATA = train_params["load_data"]
            const.TRAIN_MODEL_NAME = train_params["model_name"]
            const.TRAIN_EARLY_STOPPING = train_params["early_stopping"]
            const.TRAIN_EARLY_STOP_PATIENCE = train_params["patience"]
            const.TRAIN_SAVE_WEIGHTS_ONLY = train_params["save_weights_only"]
            const.EVAL_ONLY = train_params["eval_only"]
            const.EVAL_BATCH_SIZE = train_params["eval_batch_size"]
            const.EVAL_IOU_THRESHOLD = train_params["eval_iou_threshold"]
            const.EVAL_SCORE_THRESHOLD = train_params["eval_score_threshold"]
            const.LR_SCHEDULER = train_params["lr_scheduler"]
            const.INITIAL_LR = train_params["initial_lr"]
            # LR parameters for 'reduce_on_plateau'
            if "lr_reduce_factor" in train_params:
                const.LR_REDUCE_FACTOR = train_params["lr_reduce_factor"]
            if "lr_reduce_patience" in train_params:
                const.LR_REDUCE_PATIENCE = train_params["lr_reduce_patience"]
            if "lr_min_delta" in train_params:
                const.LR_MIN_DELTA = train_params["lr_min_delta"]

        # Initialise constants for augmentation pipeline within training
        if "train_augmentation" in params:
            train_augment_params = params["train_augmentation"]

        # Initialise constants for model inference
        if "inference" in params:
            inf_params = params["inference"]
            # Input/Output
            const.INFERENCE_MODEL_PATH = inf_params["model_path"]
            const.IMAGE_INPUT_PATH = inf_params["image_input_path"]
            const.CSV_OUTPUT = inf_params["csv_output"]
            const.SAVE_OUTPUT_IMAGE = inf_params["save_output_image"]
            const.IMAGE_OUTPUT_DIR = inf_params["image_output_dir"]
            # Model parameters
            const.INFERENCE_BACKBONE = inf_params["inference_backbone"]
            const.INFERENCE_CONFIDENCE_THRESH = inf_params["confidence_threshold"]
            const.INFERENCE_RUN_NMS = inf_params["run_nms"]
            const.INFERENCE_NMS_THRESH = inf_params["nms_threshold"]
            const.PKD_BASE_DIR = inf_params["pkd_base_dir"]
            # const.DRAW_BBOX_CONFIG = inf_params["draw_bbox_config"]
            const.EFFICIENTDET_CONFIG = inf_params["efficientdet_config"]
            # Postprocessing parameter
            const.SLICE_HEIGHT = inf_params["slice_height"]
            const.SLICE_WIDTH = inf_params["slice_width"]
            const.OVERLAP_HEIGHT_RATIO = inf_params["overlap_height_ratio"]
            const.OVERLAP_WIDTH_RATIO = inf_params["overlap_width_ratio"]
            const.POSTPROCESS_TYPE = inf_params["postprocess_type"]
            const.POSTPROCESS_MATCH_METRIC = inf_params["postprocess_match_metric"]
            const.POSTPROCESS_MATCH_THRESHOLD = inf_params[
                "postprocess_match_threshold"
            ]

        if "efficientdet" in params:
            ed_params = params["efficientdet"]
            logger.info("Initialising EfficientDet constants")
            if "train_annotations_path" in ed_params:
                const.TRAIN_ANNOTATIONS_PATH = ed_params["train_annotations_path"]
            if "val_annotations_path" in ed_params:
                const.VAL_ANNOTATIONS_PATH = ed_params["val_annotations_path"]
            if "test_annotations_path" in ed_params:
                const.TEST_ANNOTATIONS_PATH = ed_params["test_annotations_path"]
            if "snapshot-path" in ed_params:
                const.SAVED_MODEL_PATH = ed_params["snapshot-path"]
            if "saved_best_model" in ed_params:
                const.BEST_MODEL = ed_params["saved_best_model"]
            # Parameters for anchor boxes
            if "anchor_box_scales" in ed_params:
                const.ANCHOR_BOX_SCALES = ed_params["anchor_box_scales"]
            else:
                const.ANCHOR_BOX_SCALES = None
            if "anchor_box_ratios" in ed_params:
                const.ANCHOR_BOX_RATIOS = ed_params["anchor_box_ratios"]
            if "train_backbone" in ed_params:
                const.ED_TRAIN_BACKBONE = ed_params["train_backbone"]
            if "image_sizes" in ed_params:
                const.ED_IMAGE_SIZES = tuple(ed_params["image_sizes"])
