import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns
import glob
import cv2
from PIL import Image
from operator import add

def show_image(img_path):
    """_summary_

    Args:
        img_path (str): _description_

    Returns:
        _type_: _description_
    """    
    print(f'Image: {img_path}')
    img_cv2 = cv2.imread(img_path)
    img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    fig = plt.figure(figsize=(20, 10))
    plt.axis('off')
    plt.imshow(img_cv2)
    plt.show()
    return img_cv2

def show_annotated_image(img_path, annot_df, box_colours=[(244, 223, 156), (164, 232, 241), (119, 118, 188)], line_thickness=2, plot_dim=(20, 10)):
    """_summary_

    Args:
        img_path (str): _description_
        annot_df (DataFrame): _description_
        box_colours (list, optional): _description_. Defaults to [(244, 223, 156), (164, 232, 241), (119, 118, 188)].
        line_thickness (int, optional): _description_. Defaults to 2.
        plot_dim (tuple, optional): _description_. Defaults to (20, 10).
    """    
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for index, row in annot_df.iterrows():
        cat_id = row['category_id']
        colour = box_colours[cat_id-1]
        cv2.rectangle(img, (int(row['bbox_x']), int(row['bbox_y'])), (int(row['bbox_x']+row['bbox_width']), int(row['bbox_y']+row['bbox_height'])), colour, line_thickness)
    fig = plt.figure(figsize=plot_dim)
    plt.axis('off')
    plt.imshow(img);

def load_image(data_dir, img_name, lib):
    """ 
    This function uses the PIL or CV to load image file

    Args:
        data_dir (string): image folder path
        img_name (string): image name

    Returns: 
        img: image object for PIL; numpy array for CV

    """
    if lib == 'pil':
        img = Image.open(data_dir + img_name)
    elif lib == 'cv2':
        img_path = os.path.join(data_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def plot_label_distribution(df, class_col, plot_dim=(20,10), font_size=20):
    """
    This function plots a histogram showing the distribution of classes in the given dataset.

    Args:
        df (DataFrame): Pandas dataframe containing the dataset
    """
    print(df[class_col].value_counts())
    
    plt.figure(figsize=plot_dim)
    plt.title('Frequency Distribution of Classes', fontsize=font_size)
    ax = sns.countplot(x=class_col, data=df)
    ax.set(xlabel='Class', ylabel='Count')
    plt.show()

def plot_box_wh(df, x_col, y_col, group_by_col, plot_dim=(20,20)):
    """
    This function plots the joint plot of the width and height of the bounding boxes 
    Args:
        df (DataFrame): 
    """
    
    # Joint Plot of b_w and b_h
    fig = plt.figure(figsize=plot_dim)
    g = sns.jointplot(data=df, x=x_col, y=y_col, hue=group_by_col, alpha=0.5)
    g.set_axis_labels('Box Width', 'Box Height')
    g.ax_joint.xaxis.set_major_locator(ticker.MultipleLocator(500))
    return

def list_image_paths_by_ext(data_dir):
    """
    This function lists the image paths by extension

    Args:
        data_dir (string): image folder path

    Returns:
        img_path_list: image path list for each image in folder
        img_format_dict: image format composition of folder
    """
    img_format_list = ['JPG', 'jpg', 'png', 'gif', 'tga']
    img_format_dict = {}
    img_path_list = []
    
    for img_format in img_format_list:
        img_paths = glob.glob(data_dir + '_*.' + img_format)
        img_format_dict[img_format] = img_format_dict.get(img_format, 0) + len(img_paths)
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
    return '#%02x%02x%02x' % (red_val, 0, 0)


def get_green(green_val):
    """
    This function gets the signed hexadecimal value of green pixel

    Args:
        green_val (int): green pixel value

    Returns:
        green_hex: green hex string
    """
    return '#%02x%02x%02x' % (0, green_val, 0)


def get_blue(blue_val):
    """
    This function gets the signed hexadecimal value of blue pixel

    Args:
        blue_val (int): blue pixel value

    Returns:
        blue_hex: blue hex string
    """
    return '#%02x%02x%02x' % (0, 0, blue_val)

def plot_cum_color_hist(data_dir):
    """
    This function plots the cumulative rgb distribution (x axis: pixel intensity; y axis: pixel count)

    Args:
        data_dir (string): image folder path
    """
    red_pixel_list, green_pixel_list, blue_pixel_list = [0]*256, [0]*256, [0]*256

    img_path_list, _ = list_image_paths_by_ext(data_dir)

    img_names = [img_path.split('/')[-1] for img_path in img_path_list]

    for each in img_names:
        img = load_image(data_dir, each, 'pil')
        r, g, b = img.split()
        red_pixel_counts = r.histogram()
        green_pixel_counts = g.histogram()
        blue_pixel_counts = b.histogram()
        red_pixel_list = list(map(add, red_pixel_list, red_pixel_counts))
        green_pixel_list = list(map(add, green_pixel_list, green_pixel_counts))
        blue_pixel_list = list(map(add, blue_pixel_list, blue_pixel_counts))

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row', figsize = (20, 20))

    # R histogram
    for i in range(0, 256):
        ax1.bar(i, red_pixel_list[i], color=get_red(i), edgecolor=get_red(i), alpha=0.3)

    # G histogram
    for i in range(0, 256):
        ax2.bar(i, green_pixel_list[i], color=get_green(i), edgecolor=get_green(i), alpha=0.3)

    # B histogram
    for i in range(0, 256):
        ax3.bar(i, blue_pixel_list[i], color=get_blue(i), edgecolor=get_blue(i), alpha=0.3)

def plot_box_mid_pt(df, group_by_col, plot_dim=(40, 40)):
    """
    This function plots the joint plot of the mid points of the bounding boxes 
    Args:
        df (DataFrame): Pandas dataframe containing the bounding box data
    """
    df_copy = df.copy()
    df_copy['bbox_x_mid'] = df_copy['bbox_x'] + (df_copy['bbox_width']/2)
    df_copy['bbox_y_mid'] = df_copy['bbox_y'] + (df_copy['bbox_height']/2)
    
    # Joint Plot of b_w and b_h
    fig = plt.figure(figsize=plot_dim)
    g = sns.jointplot(data=df_copy, x='bbox_x_mid', y='bbox_y_mid', hue=group_by_col, alpha=0.5)
    g.set_axis_labels('Box Mid X', 'Box Mid Y')


def plot_displot(df, x_col, group_by_col):
    """
    This function plots a histogram showing the distribution of classes in the given dataset.

    Args:
        df (DataFrame): Pandas dataframe containing the dataset
        x_col : feature of the dataframe
    """

    print(df[x_col].value_counts())
    sns.displot(df, x=x_col, hue=group_by_col, element="step")
    