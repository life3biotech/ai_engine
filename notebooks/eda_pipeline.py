#!/usr/bin/env python

###########
# Generic #
###########
import glob
import os

########
# Libs #
########
import cv2
from matplotlib import colors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from operator import add
import pandas as pd
from PIL import Image
import seaborn as sns

##########
# Custom #
##########

class DataExplorer:
    def __init__(self):
        """Analyses the bounding boxes and images to
        gather some insights.

        Args:
            annot_df (pd.DataFrame): input dataframe
            img_dir (str): image directory

        """
        self.annot_df = None
        self.img_dir = None

    ####################
    # Helper functions #
    ####################

    def remap_col_cat(self, df, old_col_name, new_col_name, col_dict):
        """Remaps the column categories.

        Args:
            df (pd.Dataframe): input dataframe
            old_col_name (str): old column name
            new_col_name (str): new column name
            col_dict (dict): column-value dictionary

        Returns:
            df (pd.Dataframe): output dataframe
        """

        # reassign the column categories
        df[new_col_name] = df.loc[:, old_col_name]
        col_map = {new_col_name: col_dict}
        df = df.replace(col_map)
        return df

    def get_max_dim(self, df, new_col_name, col_list):
        """ "Gets the max dimension from either width or height."""

        df[new_col_name] = df[col_list].max(axis=1)
        return df

    def get_round_col(
        self,
        df,
        col_name,
        deci_pl=2,
    ):
        """Gets the rounded values in a column
        by a stated decimal place.

        Args:
            df (pd.DataFrame): annotation dataframe
            col_name (str): name of the column to be rounded
            deci_pl (int): decimal place of rounding
        Returns:
            df (pd.DataFrame): annotation dataframe
            new_col_name (str): name of the new column rounded
        """

        new_col_name = col_name + "_dp_" + str(deci_pl)
        df[new_col_name] = df[col_name].round(deci_pl)
        return df, new_col_name

    def get_square(self, df, col_names, deci_pl=2, square_col_name="square_cat"):
        """Gets square catgory based on
        rounded bb width and bb height.

        Args:
            df (pd.DataFrame): annotation dataframe
            col_names (list): list of column names
            deci_pl (int): decimal place of rounding
            square_col_name (str): name of square category column

        Returns:
            df (pd.DataFrame): annotation dataframe
        """

        # check the list of column names is at 2
        if len(col_names) > 2:
            raise ValueError(
                "Please use a pair of values; ideally the bb width and bb height"
            )

        # create the rounded up cols
        new_col_names = []
        for col_name in col_names:
            df, new_col_name = self.get_round_col(
                df=df, col_name=col_name, deci_pl=deci_pl
            )
            new_col_names.append(new_col_name)

        # categorise based on if-else condition
        first_new_col_name = new_col_names[0]
        second_new_col_name = new_col_names[1]
        df[square_col_name] = np.where(
            df[first_new_col_name] == df[second_new_col_name], "square", "non square"
        )
        return df

    def subset_col_value(self, df, col_name, col_value):
        """Subsets dataframe based on the column value
        Args:
            df (pd.DataFrame): annotation dataframe
            col_name (str): name of the column to subset
            col_value (str/int): the value to subset

        Returns:
            subset df
        """
        return df[df[col_name] == col_value]

    def get_keys_from_value(self, d, val):
        """Gets the keys from dictionary based on the values
        
        Args:
            d (dict): input dictionary
            val (values): input values
        Returns:
            k (keys): output keys
        """
        return [k for k, v in d.items() if v == val]

    ##################
    # Core functions #
    ##################

    def plot_label_distribution(
        self,
        df,
        class_col,
        plot_dim=(20, 10),
        font_size=20,
        title="Frequency Distribution of Classes",
    ):
        """
        This function plots a histogram showing the distribution of classes in the given dataset.

        Args:
            df (DataFrame): Pandas dataframe containing the dataset
        """
        print(df[class_col].value_counts())

        plt.figure(figsize=plot_dim)
        plt.title(title, fontsize=font_size)
        ax = sns.countplot(x=class_col, data=df)
        ax.set(xlabel="Class", ylabel="Count")
        ax.xaxis.get_label().set_fontsize(20)
        ax.yaxis.get_label().set_fontsize(20)
        plt.tick_params(axis='x', labelsize=20)
        plt.tick_params(axis='y', labelsize=20)
        plt.show()

    def plot_box_mid_pt(
        self,
        df,
        group_by_col,
        plot_dim=(40, 40),
        x_limit=(0, 2000),
        y_limit=(0, 1500),
        alpha=0.5,
    ):
        """
        This function plots the joint plot of the mid points of the bounding boxes
        Args:
            df (DataFrame): Pandas dataframe containing the bounding box data
            group_by_col (str): group by column name
            plot_dim (tuple): plot dimensions
            x_limit (tuple): x axis limit
            y_limit (tuple): y axis limit
        """
        df_copy = df.copy()
        df_copy["bbox_x_mid"] = df_copy["bbox_x"] + (df_copy["bbox_width"] / 2)
        df_copy["bbox_y_mid"] = df_copy["bbox_y"] + (df_copy["bbox_height"] / 2)

        # Joint Plot of b_w and b_h
        fig = plt.figure(figsize=plot_dim)
        g = sns.jointplot(
            data=df_copy, x="bbox_x_mid", y="bbox_y_mid", hue=group_by_col, alpha=alpha
        )
        g.set_axis_labels("Box Mid X", "Box Mid Y")
        g.ax_marg_x.set_xlim(x_limit)
        g.ax_marg_y.set_ylim(y_limit)

    def plot_box_wh(self, df, x_col, y_col, group_by_col, plot_dim=(20, 20), alpha=0.5):
        """
        This function plots the joint plot of the width and height of the bounding boxes
        Args:
            df (DataFrame): Pandas dataframe containing the bounding box data
            x_col (str): dimension of the bounding box
            y_col (str): dimension of the bounding box
            group_by_col (str): group by category
            plot_dim (tuple): dimension of the plot
        """

        # Joint Plot of b_w and b_h
        fig = plt.figure(figsize=plot_dim)
        g = sns.jointplot(data=df, x=x_col, y=y_col, hue=group_by_col, alpha=alpha)
        g.set_axis_labels("Box Width", "Box Height")
        # g.ax_joint.xaxis.set_major_locator(ticker.MultipleLocator(500))

    def plot_displot(self, df, x_col, group_by_col, element="step"):
        """
        This function plots a histogram showing the distribution of classes in the given dataset.

        Args:
            df (DataFrame): Pandas dataframe containing the dataset
            x_col (str) : feature of the dataframe
            group_by_col (str): grouping feature in the dataframe
        """

        print(df[[x_col]].describe())
        sns.displot(df, x=x_col, hue=group_by_col, element=element)

    def show_image(self, img_path):
        """_summary_

        Args:
            img_path (str): _description_

        Returns:
            _type_: _description_
        """
        print(f"Image: {img_path}")
        img_cv2 = cv2.imread(img_path)
        img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
        fig = plt.figure(figsize=(20, 10))
        plt.axis("off")
        plt.imshow(img_cv2)
        plt.show()
        return img_cv2


    def show_annotated_image(
        self,
        img_path,
        annot_df,
        box_colours_names=["red", "orange", "pink"],
        line_thickness=2,
        plot_dim=(20, 10),
    ):
        """Shows annotated images.

        Args:
            img_path (str): _description_
            annot_df (DataFrame): _description_
            box_colours (list, optional): _description_. Defaults to [(244, 223, 156), (164, 232, 241), (119, 118, 188)].
            line_thickness (int, optional): _description_. Defaults to 2.
            plot_dim (tuple, optional): _description_. Defaults to (20, 10).
        
        ######################
        # Implemenation note #
        ######################

        OpenCV and Matplotlib have different colour codes. For instance, OpenCV uses actual
        RGB values but Matplotlib uses its own values (within range 0 to 1). 
        User to key in the color names and use mapping dictionary to convert to OpenCV standards
        while use matplotlib's color to convert to its own values.

        """
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # get opencv colour codes
        opencv_rgb = {
            "red" : (255, 0, 0),
            "orange" : (255, 128, 0),
            "pink" : (255, 153, 255)
        }


        box_colours = [opencv_rgb[x] for x in box_colours_names]

        cat_names = []
        colours = []
        matplot_colours = []
        for index, row in annot_df.iterrows():
            cat_id = row["category_id"]
            colour = box_colours[cat_id - 1]

            # draw bboxes
            cv2.rectangle(
                img,
                (int(row["bbox_x"]), int(row["bbox_y"])),
                (
                    int(row["bbox_x"] + row["bbox_width"]),
                    int(row["bbox_y"] + row["bbox_height"]),
                ),
                colour,
                line_thickness,
            )
            cat_name = row["category_name"]
            cat_names.append(cat_name)
            colours.append(colour)
            
            # convert color to matplot format
            colour_key = self.get_keys_from_value(opencv_rgb, colour)
            matplot_colour = colors.to_rgb(colour_key[0])
            matplot_colours.append(matplot_colour)

        # get unique combinations of cat and color
        combinations = []
        for cat, colour in zip(cat_names, matplot_colours):
            if [cat, colour] not in combinations:
                combinations.append([cat, colour])

        # get cat and color with indexes separately
        cats = []
        colours = []
        idxes = []
        for idx, combination in enumerate(combinations): 
            cat = combination[0] 
            colour = combination[1]
            cats.append(cat)
            colours.append(colour)
            idxes.append(idx)

        # idx to cat; idx to colours mapping
        cats_mapping = dict(zip(idxes, cats))
        colours_mapping = dict(zip(idxes, colours))

        patches =[mpatches.Patch(color=colours_mapping[i],label=cats_mapping[i]) for i in colours_mapping]
        
        fig = plt.figure(figsize=plot_dim)
        plt.legend(handles=patches, loc=4, borderaxespad=0.)
        plt.axis("off")
        plt.imshow(img)


def load_image(data_dir, img_name, lib):
    """
    This function uses the PIL or CV to load image file

    Args:
        data_dir (string): image folder path
        img_name (string): image name

    Returns:
        img: image object for PIL; numpy array for CV

    """
    if lib == "pil":
        img = Image.open(data_dir + img_name)
    elif lib == "cv2":
        img_path = os.path.join(data_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def list_image_paths_by_ext(data_dir):
    """
    This function lists the image paths by extension

    Args:
        data_dir (string): image folder path

    Returns:
        img_path_list: image path list for each image in folder
        img_format_dict: image format composition of folder
    """
    img_format_list = ["JPG", "jpg", "png", "gif", "tga"]
    img_format_dict = {}
    img_path_list = []

    for img_format in img_format_list:
        img_paths = glob.glob(data_dir + "_*." + img_format)
        img_format_dict[img_format] = img_format_dict.get(img_format, 0) + len(
            img_paths
        )
        if len(img_paths) != 0:
            img_path_list.extend(img_paths)
    return img_path_list, img_format_dict


def get_red(red_val):
    """
    This function gets the signed hexadecimal value of red pixel

    Args:
        red_val (int): red pixel value

    Returns:
        red_hex: red hex string
    """
    return "#%02x%02x%02x" % (red_val, 0, 0)


def get_green(green_val):
    """
    This function gets the signed hexadecimal value of green pixel

    Args:
        green_val (int): green pixel value

    Returns:
        green_hex: green hex string
    """
    return "#%02x%02x%02x" % (0, green_val, 0)


def get_blue(blue_val):
    """
    This function gets the signed hexadecimal value of blue pixel

    Args:
        blue_val (int): blue pixel value

    Returns:
        blue_hex: blue hex string
    """
    return "#%02x%02x%02x" % (0, 0, blue_val)


def plot_cum_color_hist(data_dir):
    """
    This function plots the cumulative rgb distribution (x axis: pixel intensity; y axis: pixel count)

    Args:
        data_dir (string): image folder path
    """
    red_pixel_list, green_pixel_list, blue_pixel_list = [0] * 256, [0] * 256, [0] * 256

    img_path_list, _ = list_image_paths_by_ext(data_dir)

    img_names = [img_path.split("/")[-1] for img_path in img_path_list]

    for each in img_names:
        img = load_image(data_dir, each, "pil")
        r, g, b = img.split()
        red_pixel_counts = r.histogram()
        green_pixel_counts = g.histogram()
        blue_pixel_counts = b.histogram()
        red_pixel_list = list(map(add, red_pixel_list, red_pixel_counts))
        green_pixel_list = list(map(add, green_pixel_list, green_pixel_counts))
        blue_pixel_list = list(map(add, blue_pixel_list, blue_pixel_counts))

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(
        2, 2, sharex="col", sharey="row", figsize=(20, 20)
    )

    # R histogram
    for i in range(0, 256):
        ax1.bar(i, red_pixel_list[i], color=get_red(i), edgecolor=get_red(i), alpha=0.3)

    # G histogram
    for i in range(0, 256):
        ax2.bar(
            i,
            green_pixel_list[i],
            color=get_green(i),
            edgecolor=get_green(i),
            alpha=0.3,
        )

    # B histogram
    for i in range(0, 256):
        ax3.bar(
            i, blue_pixel_list[i], color=get_blue(i), edgecolor=get_blue(i), alpha=0.3
        )
