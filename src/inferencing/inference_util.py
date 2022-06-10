import pandas as pd
import numpy as np
from pathlib import Path, PurePath
from pconst import const
from typing import Tuple
from math import floor
import os
import sys


def get_image_list(logger, dir_path: Path) -> None:
    """This function initialises the list of images for EfficientDet to perform inference on.
    In `single` or `rest_api` inference mode, the list will contain a single image file path/URL.

    Attributes:
        logger: Logger object used to log events to.

    Returns:
        list: List of image file paths or URLs
    """
    image_list = []
    for filename in os.listdir(dir_path):
        img_file = Path(dir_path, filename).resolve()
        if img_file.exists() and img_file.is_file():
            if filename.lower().endswith(tuple(const.ACCEPTED_IMAGE_FORMATS)):
                image_list.append(str(filename))
        else:
            logger.error(f"Invalid image file path: {img_file}")
    return image_list


def make_dir(dir_path: Path) -> None:
    """
    Create directory if it did not exist

    Args:
        dir_path (Path): Full path to create new directory
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def get_col_name() -> list:
    """
    Store the column name of the generate_img_results() function.

    Returns:
        list: Return column name of the generate_img_results() function
    """
    return [
        "img_filename",
        "cell_tot",
        "cell_type_tot",
        "cell_accum_type_tot",
        "tot_small_cell",
        "tot_mid_cell",
        "tot_large_cell",
    ]


def generate_img_results(
    logger, df: pd.DataFrame, img_filename: str, eval_bool: bool = False
) -> pd.DataFrame:
    """
    Count one image cell based on user defined size and cell type. Return pandas dataframe

    Args:
        df (pd.DataFrame): Predicted bounding box results Pandas dataframe
        img_filename (str): Filename of the predicted image
        calibrate_cell_size_bool (bool, optional): Calibrate cell size

    Returns:
        pd.DataFrame: Process panda dataframe
    """
    from math import floor

    # Select longest length between width and height of bounding box
    df["longest_length"] = df[["bbox_width", "bbox_height"]].max(axis=1)

    # Count cell based on user defined values

    # convert micrometer to pixel value
    if eval_bool:
        (
            tot_small_cell,
            tot_medium_cell,
            tot_large_cell,
            _,
            _,
        ) = convert_cell_size(logger, df)
    else:
        (
            tot_small_cell,
            tot_medium_cell,
            tot_large_cell,
            _,
            _,
        ) = convert_cell_size(logger, df, const.USE_CALIBRATED_CELLSIZE)

    # Check total cells based on cell type
    df_category_name = df.category_name.value_counts()
    cell_type_tot = 0
    cell_accum_type_tot = 0
    # TODO check whether to hardcode the class map.
    for key, value in df_category_name.to_dict().items():
        if key == "cell":
            cell_type_tot = value
        elif key == "cell accumulation":
            cell_accum_type_tot = value

    process_df = pd.DataFrame(
        [
            [
                img_filename,
                len(df.index),
                cell_type_tot,
                cell_accum_type_tot,
                tot_small_cell,
                tot_medium_cell,
                tot_large_cell,
            ],
        ],
        columns=get_col_name(),
    )

    return process_df


def convert_cell_size(
    logger, df: pd.DataFrame, calibrate_cell_size_bool: bool = False
) -> Tuple[int, int, int, float, float]:
    """
    Count number of cells based on the conversion of um to pixel

    Args:
        logger: Logger object used to log events to.
        df (pd.DataFrame): cell longest_length info
        calibrate_cell_size_bool (bool, optional): Calibrate cell size
        by comparing ground truth vs predicted bbox proportion according to size. Defaults to False.

    Returns:
        Tuple[int, int, int, int, int]: return the number of cells based on size and their micrometer cutoff value
    """

    # convert micrometer to pixel value
    if calibrate_cell_size_bool:
        try:
            calibrated_df = pd.read_csv(
                PurePath("././conf", "calibrated_cellsize.csv"), index_col=0
            )

            small_mid_cell_pixel = calibrated_df.sm_cell_pixel_calibrated[0]
            mid_large_cell_pixel = calibrated_df.ml_cell_pixel_calibrated[0]

            small_mid_cell_um = small_mid_cell_pixel * const.UM_PIXEL_MAPPING
            mid_large_cell_um = mid_large_cell_pixel * const.UM_PIXEL_MAPPING

        except:
            logger.error(
                "Error getting calibrated data from conf/calibrated_cellsize.csv\nFalling back to uncalibrated cutoff values"
            )
            small_mid_cell_pixel = floor(
                const.SMALL_MID_CELL_CUTOFF / const.UM_PIXEL_MAPPING
            )
            mid_large_cell_pixel = floor(
                const.MID_LARGE_CELL_CUTOFF / const.UM_PIXEL_MAPPING
            )

            small_mid_cell_um = const.SMALL_MID_CELL_CUTOFF
            mid_large_cell_um = const.MID_LARGE_CELL_CUTOFF

    else:
        small_mid_cell_pixel = floor(
            const.SMALL_MID_CELL_CUTOFF / const.UM_PIXEL_MAPPING
        )
        mid_large_cell_pixel = floor(
            const.MID_LARGE_CELL_CUTOFF / const.UM_PIXEL_MAPPING
        )

        small_mid_cell_um = const.SMALL_MID_CELL_CUTOFF
        mid_large_cell_um = const.MID_LARGE_CELL_CUTOFF

    tot_small_cell = df[df.longest_length < small_mid_cell_pixel].shape[0]
    tot_medium_cell = df[
        (df.longest_length >= small_mid_cell_pixel)
        & (df.longest_length < mid_large_cell_pixel)
    ].shape[0]
    tot_large_cell = df[df.longest_length >= mid_large_cell_pixel].shape[0]

    return (
        tot_small_cell,
        tot_medium_cell,
        tot_large_cell,
        small_mid_cell_um,
        mid_large_cell_um,
    )


def save_calibrated_params(logger) -> None:
    """
    Saving the params that are used for the calibration.
    Used for inference check.
    If during inference, params differ from calibration params saved.
    Logger will warn user to rerun the eval_model function

    Args:
        logger: Logger object used to log events to.

    """
    params_df = create_params_df()
    params_df.to_csv(PurePath("././conf", "calibrated_params.csv"))


def create_params_df() -> pd.DataFrame:
    """Create dataframe for the parameters

    Returns:
        pd.DataFrame: Return the information as pandas dataframe
    """
    params_df = pd.DataFrame()

    params_df["params_name"] = [
        "inference_backbone",
        "confidence_threshold",
        "run_nms",
        "nms_threshold",
        "slice_height",
        "slice_width",
        "overlap_height_ratio",
        "overlap_width_ratio",
        "postprocess_type",
        "postprocess_bbox_sort",
        "postprocess_match_metric",
        "postprocess_match_threshold",
        "inference_slice",
        "max_detections",
        "class_specific_filter",
        "detect_quadrangle",
        "score_threshold",
        "select_top_k",
    ]

    params_df["values"] = [
        const.INFERENCE_BACKBONE,
        const.INFERENCE_CONFIDENCE_THRESH,
        const.INFERENCE_RUN_NMS,
        const.INFERENCE_NMS_THRESH,
        const.SLICE_HEIGHT,
        const.SLICE_WIDTH,
        const.OVERLAP_HEIGHT_RATIO,
        const.OVERLAP_WIDTH_RATIO,
        const.POSTPROCESS_TYPE,
        const.POSTPROCESS_BBOX_SORT,
        const.POSTPROCESS_MATCH_METRIC,
        const.POSTPROCESS_MATCH_THRESHOLD,
        const.INFERENCE_SLICE,
        const.MAX_DETECTIONS,
        const.CLASS_SPECIFIC_FILTER,
        const.DETECT_QUADRANGLE,
        const.SCORE_THRESHOLD,
        const.SELECT_TOP_K,
    ]

    return params_df


def params_check(logger) -> list:
    """Check if the parameters has been changed in pipelines.yml as compared to calibrated_params.csv
    Warn user to re-run eval_model function to re-calibrate the cellsize.

    Args:
        logger: Logger object used to log events to.

    Returns:
        list: name of parameters that have been changed
    """
    pipeline_params_df = create_params_df()

    try:
        calibrated_params_df = pd.read_csv(
            PurePath("././conf", "calibrated_params.csv"), index_col=0
        )
    except:
        logger.error(
            "!! No conf/calibrated_params.csv found, model not calibrated, run 'python -m src.eval_model' once to calibrate model !!"
        )
        sys.exit()

    # create a new column in pipeline_params_df to check if values matches
    pipeline_params_df["calibrated_values"] = calibrated_params_df["values"]
    pipeline_params_df["values"] = pipeline_params_df["values"].astype(str)
    pipeline_params_df["calibrated_values"] = pipeline_params_df[
        "calibrated_values"
    ].astype(str)

    pipeline_params_df.rename(columns={"values": "pipelines_values"}, inplace=True)

    pipeline_params_df["values_match"] = np.where(
        pipeline_params_df["pipelines_values"]
        == pipeline_params_df["calibrated_values"],
        "True",
        "False",
    )

    changes = pipeline_params_df[pipeline_params_df["values_match"] == "False"][
        "params_name"
    ].tolist()

    if len(changes) > 0:
        logger.warning(
            "Config Parameters have changed, pipelines.yml differ from calibrated_params.csv. Affected parameters: "
            + ", ".join(str(x) for x in changes)
        )
        logger.warning(
            "Config Parameters have changed, please rerun eval_model to recalibrate cellsize."
        )

    return changes
