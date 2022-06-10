import pandas as pd
import numpy as np
from pconst import const
from pathlib import PurePath
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple
from scipy.stats import ks_2samp, ttest_ind
from .inference_util import make_dir, convert_cell_size, save_calibrated_params
from math import floor

import inferencing as inference


class EvalCalibrate:
    """
    Class for evaluating and calibrating ground truth vs predicted bounding box for cell.

    Attributes:
        logger: Logger object used to log events to.
    """

    def __init__(self, logger):
        self.logger = logger

    def copy_test_images(self, filenames: list) -> None:
        """
        Copies all test image files from the data subdirectories into the temp data directory.
        Args:
            filenames (list): List of filenames
        """
        testimg_tempfolder = PurePath(const.PROCESSED_DATA_PATH, "eval_folder", "img")
        make_dir(testimg_tempfolder)
        for filename in filenames:
            try:
                shutil.copy(PurePath(filename), testimg_tempfolder)
            except OSError as e:
                self.logger.error(f"Error occurred while copying file: {e}")

    def create_cell_histplots(
        self, testset_np: np.array, predicted_np: np.array, results_ks: str
    ) -> None:
        """To plot histplot kde for cells longest length in micro meter

        Args:
            testset_np (np.array): Test dataset annotations box info
            predicted_np (np.array): Predicted annotations box info
            results_ks (str): Kolmogorov-Smirnov test results
        """
        # Creating a displot

        fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
        sns.histplot(testset_np, kde=True, bins=30, label="Ground Truth", color="blue")
        sns.histplot(predicted_np, kde=True, bins=30, label="Predicted", color="orange")

        plt.title("Ground Truth vs Predicted distribution", fontsize=15)
        # ax.text(1, 1, str(results_ks), bbox=dict(facecolor="red", alpha=0.5))
        plt.annotate(
            results_ks,
            xy=(0.01, 0.95),
            xycoords="axes fraction",
            bbox=dict(facecolor="red", alpha=0.5),
        )
        plt.legend(prop={"size": 12})
        plt.xlabel("Cell length (micro-meter)")
        plt.ylabel("Density")

        distribution_output = PurePath(
            const.PROCESSED_DATA_PATH,
            "eval_folder",
            "distribution_output",
        )
        make_dir(distribution_output)

        distribution_output = PurePath(
            const.PROCESSED_DATA_PATH,
            "eval_folder",
            "distribution_output",
            "distribution.png",
        )
        plt.savefig(distribution_output)  # save as png

    def create_cell_barplots(
        self,
        testset_df: pd.DataFrame,
        predicted_df: pd.DataFrame,
        testset_len: int,
        predicted_len: int,
        calibrate_cell_size_bool: bool = False,
    ) -> None:
        """Plot # of cell barplot based on small, medium and large cell size

        Args:
            testset_df (pd.DataFrame):  Test dataset annotations box info
            predicted_df (pd.DataFrame): Predicted annotations box info
            testset_len (int): Total number of cell for test set
            predicted_len (int): Total number of cell for predicted set
            calibrate_cell_size_bool (bool, optional): Calibrate cell size
            by comparing ground truth vs predicted bbox proportion according to size. Defaults to False.
        """

        # Creating a barplot of small, medium, large cells
        (
            tot_small_cell,
            tot_medium_cell,
            tot_large_cell,
            small_mid_cell_um,
            mid_large_cell_um,
        ) = convert_cell_size(self.logger, testset_df)
        (
            tot_small_cell_pred,
            tot_medium_cell_pred,
            tot_large_cell_pred,
            small_mid_cell_um_pred,
            mid_large_cell_um_pred,
        ) = convert_cell_size(self.logger, predicted_df, calibrate_cell_size_bool)

        cell_dict = {
            "x": ["small", "small", "medium", "medium", "large", "large"],
            "y": [
                tot_small_cell,
                tot_small_cell_pred,
                tot_medium_cell,
                tot_medium_cell_pred,
                tot_large_cell,
                tot_large_cell_pred,
            ],
            "category": [
                "Ground Truth",
                "Predicted",
                "Ground Truth",
                "Predicted",
                "Ground Truth",
                "Predicted",
            ],
        }

        legend_title = (
            f"Ground truth # cell:{testset_len}  |  Predicted # cell:{predicted_len}\n"
        )

        legend_title2 = (
            f"Actual Cell Size um: sml-med={small_mid_cell_um:.2f}  |  med-lg={mid_large_cell_um:.2f}\n"
            f"Predicted Box Size um: sml-med={small_mid_cell_um_pred:.2f}  |  med-lg={mid_large_cell_um_pred:.2f}\n"
        )

        sm_change = (
            100 * (small_mid_cell_um_pred - small_mid_cell_um) / small_mid_cell_um
        )
        ml_change = (
            100 * (mid_large_cell_um_pred - mid_large_cell_um) / mid_large_cell_um
        )

        legend_title3 = f"sml-med cell size % allowance={sm_change:.2f} %  |  med-lg cell size % allowance={ml_change:.2f} %\n"

        fig = plt.figure(figsize=(7, 6), dpi=300)
        fig.subplots_adjust(top=0.8)
        ax = fig.add_subplot(211)
        plt.suptitle("Ground Truth vs Predicted Cell Size\n", fontsize=15)
        plt.title(legend_title + legend_title2 + legend_title3, fontsize=8, y=1)
        plt.ylabel("# of Cell")
        plots = sns.barplot(x="x", y="y", hue="category", data=cell_dict, ax=ax)

        for bar in plots.patches:

            # Using Matplotlib's annotate function and
            # passing the coordinates where the annotation shall be done
            # x-coordinate: bar.get_x() + bar.get_width() / 2
            # y-coordinate: bar.get_height()
            # free space to be left to make graph pleasing: (0, 8)
            # ha and va stand for the horizontal and vertical alignment
            plots.annotate(
                format(bar.get_height(), ""),
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center",
                va="center",
                size=10,
                xytext=(0, 8),
                textcoords="offset points",
            )

        ax2 = fig.add_subplot(212)

        # print classification reported as table
        list_cell_size = [
            tot_small_cell,
            tot_medium_cell,
            tot_large_cell,
            tot_small_cell_pred,
            tot_medium_cell_pred,
            tot_large_cell_pred,
        ]

        (
            classificationreport_df,
            macro_f1,
            weighted_f1,
        ) = self.generate_classification_report(list_cell_size)

        classificationreport_df = classificationreport_df.round(4)

        # font_size = 10
        bbox = [0, 0, 1, 1]
        ax2.axis("off")
        ax2.table(
            cellText=classificationreport_df.values,
            # rowLabels=classificationreport_df.cell_type,
            bbox=bbox,
            colLabels=classificationreport_df.columns,
        )
        # mpl_table.auto_set_font_size(False)
        # mpl_table.set_fontsize(font_size)

        f1_str = f"\nMacro f1: {macro_f1:.4f} | Weighted_f1: {weighted_f1:.4f}"
        plt.gcf().text(0.3, 0.07, f1_str, fontsize=10)
        # plt.subplots_adjust(bottom=0.3)

        if calibrate_cell_size_bool:  # Save as another filename if calibrated
            distribution_output = PurePath(
                const.PROCESSED_DATA_PATH,
                "eval_folder",
                "distribution_output",
                "calibrated_cellsize_barplot.png",
            )

        else:
            distribution_output = PurePath(
                const.PROCESSED_DATA_PATH,
                "eval_folder",
                "distribution_output",
                "cellsize_barplot.png",
            )
        plt.savefig(distribution_output)  # save as png

    def eval_process(
        self, calibrate_cell_size_bool: bool = False, all_dataset: bool = False
    ):
        """The main function to process predicted and ground truth cell longest length and plot histplot and barplot for visualization

        Args:
            calibrate_cell_size_bool (bool, optional): Execute calibration/mapping groundtruth cell size to predicted cell size. Defaults to False.
            all_dataset (bool, optional): to append test, train and val dataset for eval. Defaults to False.
        """
        # Retrieving original image name used for test dataset.
        col_name = ["image_file_path", "x1", "y1", "x2", "y2", "class_name"]

        testset_csv = PurePath(
            const.PROCESSED_DATA_PATH,
            f"{const.TEST_SET_FILENAME.split('.')[0]}_efficientdet_b{const.ED_TRAIN_BACKBONE}.csv",
        )
        testset_df = pd.read_csv(testset_csv, names=col_name)

        # Get data distribution comparison for all dataset
        if all_dataset:
            trainset_csv = PurePath(
                const.PROCESSED_DATA_PATH,
                f"{const.TRAIN_SET_FILENAME.split('.')[0]}_efficientdet_b{const.ED_TRAIN_BACKBONE}.csv",
            )
            trainset_df = pd.read_csv(trainset_csv, names=col_name)

            valset_csv = PurePath(
                const.PROCESSED_DATA_PATH,
                f"{const.VAL_SET_FILENAME.split('.')[0]}_efficientdet_b{const.ED_TRAIN_BACKBONE}.csv",
            )
            valset_df = pd.read_csv(valset_csv, names=col_name)

            testset_df = testset_df.append(trainset_df).append(valset_df)

        # Select longest length between width and height of bounding box
        testset_df["bbox_width"] = testset_df["x2"] - testset_df["x1"]
        testset_df["bbox_height"] = testset_df["y2"] - testset_df["y1"]
        testset_df["longest_length"] = testset_df[["bbox_width", "bbox_height"]].max(
            axis=1
        )

        # Getting unique filename of testset
        filename_df = testset_df.drop_duplicates(subset="image_file_path", keep="first")

        # Copy test image to temp folder for inference.
        self.copy_test_images(filename_df.iloc[:, 0].tolist())

        batch_inference = inference.batch_inference.BatchInference(
            self.logger, eval_bool=True
        )
        predicted_df = batch_inference.inferencing()

        testset_np = (testset_df["longest_length"] * const.UM_PIXEL_MAPPING).to_numpy()
        predicted_np = (
            predicted_df["longest_length"] * const.UM_PIXEL_MAPPING
        ).to_numpy()

        testset_len = len(testset_df["longest_length"])
        predicted_len = len(predicted_df["longest_length"])

        cell_count_total = f"Length of test set: {testset_len}  Length of predicted set: {predicted_len}"
        self.logger.info(cell_count_total)

        results_ks = ks_2samp(
            testset_np,
            predicted_np,
        )

        # results_ttest = ttest_ind(
        #     testset_np,
        #     predicted_np,
        # )
        # print(str(results_ks))
        # print(str(results_ttest))

        self.create_cell_histplots(testset_np, predicted_np, results_ks)
        self.create_cell_barplots(testset_df, predicted_df, testset_len, predicted_len)

        # save testset_np, predicted_np, cell info to conf folder
        testset_df[["longest_length"]].to_csv(
            PurePath("././conf", "testset_cellsize.csv")
        )
        predicted_df[["longest_length"]].to_csv(
            PurePath("././conf", "predictedset_cellsize.csv")
        )

        if calibrate_cell_size_bool:
            self.logger.info("Calibrating cell size.")
            # save calibrated cell info to conf folder
            (
                sm_cell_um,
                ml_cell_um,
                sm_cell_pixel,
                ml_cell_pixel,
            ) = self.calibrate_cell_size(testset_df, predicted_df)

            data = [
                [
                    const.SMALL_MID_CELL_CUTOFF,
                    const.MID_LARGE_CELL_CUTOFF,
                    floor(const.SMALL_MID_CELL_CUTOFF / const.UM_PIXEL_MAPPING),
                    floor(const.MID_LARGE_CELL_CUTOFF / const.UM_PIXEL_MAPPING),
                    sm_cell_um,
                    ml_cell_um,
                    sm_cell_pixel,
                    ml_cell_pixel,
                ]
            ]

            calibrated_df = pd.DataFrame(
                data,
                columns=[
                    "sm_cell_um",
                    "ml_cell_um",
                    "sm_cell_pixel",
                    "ml_cell_pixel",
                    "sm_cell_um_calibrated",
                    "ml_cell_um_calibrated",
                    "sm_cell_pixel_calibrated",
                    "ml_cell_pixel_calibrated",
                ],
            )
            calibrated_df.to_csv(PurePath("././conf", "calibrated_cellsize.csv"))

            self.logger.info(
                f"sm_cell_um_calibrated:{sm_cell_um}   ml_cell_um_calibrated:{ml_cell_um}"
            )
            self.logger.info(
                f"sm_cell_um_calibrated to pixel:{sm_cell_um/const.UM_PIXEL_MAPPING}   ml_cell_um_calibrated to pixel:{ml_cell_um/const.UM_PIXEL_MAPPING}"
            )

            # Save calibrated cell size barplot
            self.create_cell_barplots(
                testset_df,
                predicted_df,
                testset_len,
                predicted_len,
                calibrate_cell_size_bool,
            )

            # Save the param that will affect the calibration and require re-run of the eval_model function if the values changes
            save_calibrated_params(self.logger)

    def eval_process_oneimg(
        self,
        selected_filename: str,
        all_dataset: bool = True,
    ):
        """
        Process one predicted and ground truth cell longest length and plot histplot and barplot for visualization
        Note: only work for full image, not tile/slice image, used for testing

        Args:
            selected_filename (str): to filter dataframe according to one image filename (must be full image)
            calibrate_cell_size_bool (bool): Execute calibration/mapping groundtruth cell size to predicted cell size
            all_dataset (bool, optional): to append test, train and val dataset for eval. Defaults to True.
        """

        # Retrieving original image name used for test dataset.
        col_name = ["image_file_path", "x1", "y1", "x2", "y2", "class_name"]

        testset_csv = PurePath(
            const.PROCESSED_DATA_PATH,
            f"{const.TEST_SET_FILENAME.split('.')[0]}_efficientdet_b{const.ED_TRAIN_BACKBONE}.csv",
        )
        testset_df = pd.read_csv(testset_csv, names=col_name)

        # Get data distribution comparison for all dataset
        if all_dataset:
            trainset_csv = PurePath(
                const.PROCESSED_DATA_PATH,
                f"{const.TRAIN_SET_FILENAME.split('.')[0]}_efficientdet_b{const.ED_TRAIN_BACKBONE}.csv",
            )
            trainset_df = pd.read_csv(trainset_csv, names=col_name)

            valset_csv = PurePath(
                const.PROCESSED_DATA_PATH,
                f"{const.VAL_SET_FILENAME.split('.')[0]}_efficientdet_b{const.ED_TRAIN_BACKBONE}.csv",
            )
            valset_df = pd.read_csv(valset_csv, names=col_name)

            testset_df = testset_df.append(trainset_df).append(valset_df)

        # create orig_file_name to select one files for inference only
        testset_df["file_type"] = testset_df.image_file_path.str.rsplit(
            ".", n=1, expand=True
        )[1]
        testset_df["file_name_split"] = testset_df.image_file_path.str.rsplit(
            "/", n=1, expand=True
        )[1]
        testset_df["file_name_split"] = testset_df.file_name_split.str.rsplit(
            "_", n=4, expand=True
        )[0]
        testset_df["orig_file_name"] = testset_df["file_name_split"]
        testset_df = testset_df[testset_df.orig_file_name == selected_filename]

        # Select longest length between width and height of bounding box
        testset_df["bbox_width"] = testset_df["x2"] - testset_df["x1"]
        testset_df["bbox_height"] = testset_df["y2"] - testset_df["y1"]
        testset_df["longest_length"] = testset_df[["bbox_width", "bbox_height"]].max(
            axis=1
        )

        # Getting unique filename of testset
        filename_df = testset_df.drop_duplicates(subset="image_file_path", keep="first")

        # Copy test image to temp folder for inference.
        self.copy_test_images(filename_df.iloc[:, 0].tolist())

        batch_inference = inference.batch_inference.BatchInference(
            self.logger, eval_bool=True
        )
        predicted_df = batch_inference.inferencing()

        testset_np = (testset_df["longest_length"] * const.UM_PIXEL_MAPPING).to_numpy()
        predicted_np = (
            predicted_df["longest_length"] * const.UM_PIXEL_MAPPING
        ).to_numpy()

        testset_len = len(testset_df["longest_length"])
        predicted_len = len(predicted_df["longest_length"])

        cell_count_total = f"Length of test set: {testset_len}  Length of predicted set: {predicted_len}"
        self.logger.info(cell_count_total)

        results_ks = ks_2samp(
            testset_np,
            predicted_np,
        )

        # results_ttest = ttest_ind(
        #     testset_np,
        #     predicted_np,
        # )
        # print(str(results_ks))
        # print(str(results_ttest))

        self.create_cell_histplots(testset_np, predicted_np, results_ks)
        self.create_cell_barplots(testset_df, predicted_df, testset_len, predicted_len)

        # save testset_np, predicted_np, cell info to conf folder
        testset_df["longest_length"].to_csv(
            PurePath("././conf", "testset_cellsize.csv")
        )
        predicted_df["longest_length"].to_csv(
            PurePath("././conf", "predictedset_cellsize.csv")
        )

    def calibrate_cell_size(
        self, testset_df: pd.DataFrame, predicted_df: pd.DataFrame
    ) -> Tuple[np.float64, np.float64, np.float64, np.float64, np.float64]:
        """
        Calibrate cell size by comparing ground truth vs predicted bbox proportion according to size.

        Args:
            testset_df (pd.DataFrame):  Test dataset annotations box info
            predicted_df (pd.DataFrame): Predicted annotations box info

        Returns:
            Tuple[np.float64, np.float64, np.float64, np.float64, np.float64]: Return calibrated cell size info
                                                                               sm_cell_um, ml_cell_um, sm_cell_pixel, ml_cell_pixel
        """

        # Get test data small type proportion
        (
            tot_small_cell,
            tot_medium_cell,
            tot_large_cell,
            small_mid_cell_um,
            mid_large_cell_um,
        ) = convert_cell_size(self.logger, testset_df)
        total_cell = tot_small_cell + tot_medium_cell + tot_large_cell
        small_cell = tot_small_cell / total_cell
        medium_cell = tot_medium_cell / total_cell
        # large_cell = tot_large_cell / total_cell

        # print(
        #     f"Test Cell Size: {const.SMALL_MID_CELL_CUTOFF}   {const.MID_LARGE_CELL_CUTOFF}"
        # )
        # print(
        #     f"Test Cell Size pixel: {floor(const.SMALL_MID_CELL_CUTOFF / const.UM_PIXEL_MAPPING)}   {floor(const.MID_LARGE_CELL_CUTOFF / const.UM_PIXEL_MAPPING)}"
        # )
        # print(f"Test Proportion: {small_cell}    {medium_cell}     {large_cell}")

        sm_cell_pixel = predicted_df.longest_length.quantile(small_cell)
        ml_cell_pixel = predicted_df.longest_length.quantile(medium_cell + small_cell)
        # large_quantile = predicted_df.longest_length.quantile(
        #     large_cell + medium_cell + small_cell
        # )

        sm_cell_um = sm_cell_pixel * const.UM_PIXEL_MAPPING
        ml_cell_um = ml_cell_pixel * const.UM_PIXEL_MAPPING

        # print(f"Pred Cell Size um: small:{sm_cell_um}    med:{ml_cell_um}")

        # print(
        #     f"Pred Cell Size pixel: small-med:{sm_cell_pixel}    med-large:{ml_cell_pixel}"
        # )

        # print("Pred Cell Size info: ", predicted_df["longest_length"].describe())

        # print(
        #     "Pred small proportion:",
        #     len(predicted_df[predicted_df.longest_length < sm_cell_pixel])
        #     / len(predicted_df),
        # )
        # print(
        #     "Pred medium proportion:",
        #     len(
        #         predicted_df[
        #             (predicted_df["longest_length"] >= sm_cell_pixel)
        #             & (predicted_df["longest_length"] < ml_cell_pixel)
        #         ]
        #     )
        #     / len(predicted_df),
        # )
        # print(
        #     "Pred Large proportion:",
        #     len(predicted_df[predicted_df.longest_length >= ml_cell_pixel])
        #     / len(predicted_df),
        # )

        return sm_cell_um, ml_cell_um, sm_cell_pixel, ml_cell_pixel

    def generate_classification_report(
        self, list_cell: list
    ) -> Tuple[pd.DataFrame, float, float]:
        """
        Generate classification report using only the given list
        list_cell = [small, med, large ,small_pred, med_pred, large_pred ]
        If actual cell # differ from predict cell #,
        The difference is consider as wrong prediction and will classified under
        wrongly prediction column and split into 2.

        for example.
        actual small cell = 10
        predicted small cell = 15
                |        Predicted       |
                |------------------------|
                | Small | Medium | Large |
        Actual  | ------| ------ | ----- |
                | 10    |   3    |   2   |


        Confusion matrix arrangment
                        |          Predicted       |
                        |--------------------------|
                        | Small  | Medium  | Large |
                        | ------ |------- |------- |
                Small   | Cell 1 | Cell 2 | Cell 3 |
        Actual  Medium  | Cell 4 | Cell 5 | Cell 6 |
                Large   | Cell 7 | Cell 8 | Cell 9 |

        Args:
            list_cell (list): [small, med, large ,small_pred, med_pred, large_pred ]

        Returns:
            Tuple[pd.DataFrame, float, float]: precision recall and f1 score dataframe, macro_f1, weighted_f1
        """

        small, med, large, small_pred, med_pred, large_pred = list_cell

        # create confusion matrix for 3 classes
        cell1 = min(small, small_pred)
        cell2 = abs(floor((small - small_pred) / 2))
        cell3 = abs(small - cell1 - cell2)

        cell4 = abs(floor((med - med_pred) / 2))
        cell5 = min(med, med_pred)
        cell6 = abs(med - cell4 - cell5)

        cell7 = abs(floor((large - large_pred) / 2))
        cell9 = min(large, large_pred)
        cell8 = abs(large - cell7 - cell9)

        actual_small_tot = cell1 + cell2 + cell3
        actual_med_tot = cell4 + cell5 + cell6
        actual_large_tot = cell7 + cell8 + cell9
        actual_tot = actual_small_tot + actual_med_tot + actual_large_tot

        # precision, recall, f1 score
        small_precision = cell1 / (cell1 + cell4 + cell7)
        med_precision = cell5 / (cell2 + cell5 + cell8)
        large_precision = cell9 / (cell3 + cell6 + cell9)

        small_recall = cell1 / (cell1 + cell2 + cell3)
        med_recall = cell5 / (cell4 + cell5 + cell6)
        large_recall = cell9 / (cell7 + cell8 + cell9)

        small_f1 = self.calc_f1_score(small_precision, small_recall)
        med_f1 = self.calc_f1_score(med_precision, med_recall)
        large_f1 = self.calc_f1_score(large_precision, large_recall)

        macro_f1 = (small_f1 + med_f1 + large_f1) / 3
        weighted_f1 = (
            actual_small_tot * small_f1
            + actual_med_tot * med_f1
            + actual_large_tot * large_f1
        ) / actual_tot

        data = [
            [
                "small",
                small_precision,
                small_recall,
                small_f1,
            ],
            [
                "medium",
                med_precision,
                med_recall,
                med_f1,
            ],
            [
                "large",
                large_precision,
                large_recall,
                large_f1,
            ],
        ]

        classificationreport_df = pd.DataFrame(
            data,
            columns=[
                "cell_type",
                "precision",
                "recall",
                "f1",
            ],
        )

        return classificationreport_df, macro_f1, weighted_f1

    def calc_f1_score(self, precision: float, recall: float) -> float:
        """Calculate f1 score and check for zero.

        Args:
            precision (float): precision value
            recall (float): recall value

        Returns:
            float: f1 score
        """
        if (precision + recall) == 0:
            return 0.0
        else:
            f1 = 2 * ((precision * recall) / (precision + recall))

        return f1
