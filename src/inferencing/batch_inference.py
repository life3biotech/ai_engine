import pandas as pd
from pathlib import Path, PurePath
from typing import Tuple
from pconst import const
from datetime import datetime, timedelta, timezone

import sahi
from .inference_util import (
    get_image_list,
    make_dir,
    generate_img_results,
    create_params_df,
)


class BatchInference:
    """This class encapsulates the data preprocessing functionality that forms part of the data pipeline.

    Attributes:
        logger: Logger object used to log events to.
        eval_bool: if running eval_model mode is true, will not save image results and csv results.
    """

    def __init__(self, logger, eval_bool=False):
        self.logger = logger
        self.eval_bool = eval_bool

        if self.eval_bool:  # for evaluation purpose
            self.image_input_dir = PurePath(
                const.PROCESSED_DATA_PATH, "eval_folder", "img"
            )
            self.csv_on = False
            # self.csv_output_dir = PurePath(const.RAW_DATA_PATH, "testimg_temp/csv")
            self.save_output_image = False  # do not save image output
            # self.image_output_dir = const.IMAGE_OUTPUT_DIR
        else:

            global current_datetime, model_args

            tzinfo = timezone(timedelta(hours=8))
            current_datetime = datetime.now(tzinfo).strftime("%Y%m%d_%H%M%S")

            self.image_input_dir = const.IMAGE_INPUT_DIR
            self.csv_on = True
            self.csv_output_dir = const.CSV_OUTPUT_DIR + "_" + current_datetime
            self.save_output_image = const.SAVE_OUTPUT_IMAGE
            self.image_output_dir = const.IMAGE_OUTPUT_DIR + "_" + current_datetime

    def inferencing(self) -> pd.DataFrame:
        """
        Inference multiple images in a folder

        Args:
            eval_mode (bool, optional): If set to True, inference is used for eval_model. Defaults to False.

        Returns:
            pd.DataFrame: return all images annotated boundingbox
        """

        # Init EfficientDet model as required by sahi
        detection_model = sahi.model.EfficientDetModel(
            device="cpu",  # or 'cuda:0'
        )

        image_list = get_image_list(self.logger, self.image_input_dir)
        self.logger.info("Total images to be inferred: {}".format(len(image_list)))

        # Batch inference
        cell_results_df = pd.DataFrame()
        tot_annotations_df = pd.DataFrame()
        for filename in image_list:
            img_file = str(Path(self.image_input_dir, filename).resolve())
            self.logger.info("Predicting image: {}".format(img_file))
            if not const.INFERENCE_SLICE:
                # predicting without slicing the image
                result = sahi.predict.get_prediction(
                    img_file,
                    detection_model,
                )
            else:
                result = sahi.predict.get_sliced_prediction(
                    img_file,
                    detection_model,
                    slice_height=const.SLICE_HEIGHT,
                    slice_width=const.SLICE_WIDTH,
                    overlap_height_ratio=const.OVERLAP_HEIGHT_RATIO,
                    overlap_width_ratio=const.OVERLAP_WIDTH_RATIO,
                    postprocess_type=const.POSTPROCESS_TYPE,
                    postprocess_match_metric=const.POSTPROCESS_MATCH_METRIC,
                    postprocess_match_threshold=const.POSTPROCESS_MATCH_THRESHOLD,
                )

            # Export predicted output annotations to csv
            df = result.to_coco_annotations(panda_df_bool=True)

            if self.csv_on:
                csv_output_path = Path(
                    self.csv_output_dir, filename.rsplit(".", 1)[0] + ".csv"
                ).resolve()
                make_dir(self.csv_output_dir)
                df.to_csv(csv_output_path)
            process_df = generate_img_results(self.logger, df, img_file, self.eval_bool)

            # Append predicted output annotations
            df["img_filename"] = img_file
            tot_annotations_df = tot_annotations_df.append(df)

            # Append one image cell info to a dataframe
            cell_results_df = cell_results_df.append(process_df, ignore_index=True)

            # Export predicted output image
            if self.save_output_image:
                make_dir(self.image_output_dir)
                result.export_visuals(
                    export_dir=self.image_output_dir,
                    file_name=filename.rsplit(".", 1)[0],
                    text_size=0,
                    rect_th=1,
                    label_bool=const.SAVE_OUTPUT_IMAGE_SHOWLABEL,
                    show_cellcount=const.SAVE_OUTPUT_IMAGE_SHOW_CELLCOUNT,
                    cellcount_info=process_df,
                )

        if self.csv_on:
            # Export all images predicted annotations to csv
            cell_annotated_path = Path(
                self.csv_output_dir, "predicted_annotation" + ".csv"
            ).resolve()
            tot_annotations_df.to_csv(cell_annotated_path)

            # Export predicted all images cell count to csv
            cell_results_path = Path(
                self.csv_output_dir, "predicted_results" + ".csv"
            ).resolve()
            cell_results_df.to_csv(cell_results_path)

        self.logger.info("Batch inferencing has completed.")

        if self.csv_on:
            self.logger.info(
                "Predicted per image CSV result location: {}".format(
                    self.csv_output_dir
                )
            )
            self.logger.info(
                "Predicted Total image Cell CSV result location: {}".format(
                    cell_results_path
                )
            )

        if self.save_output_image:
            self.logger.info(
                "Predicted Image result location: {}".format(self.image_output_dir)
            )

        return tot_annotations_df
