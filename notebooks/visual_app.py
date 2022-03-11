import streamlit as st
import os
import cv2
import json

def main(img_root_folder_path):
    st.title('Life3 Biotech Data')

    exclude_folders = [
    ]
    # generate list of dirs containing images & annotations subdirs
    folder_list = list(sorted(os.listdir(img_root_folder_path)))
    folder_list = [
        i for i in folder_list if \
            not i.startswith(".") and \
            i not in exclude_folders
    ]
    folder_name = st.sidebar.selectbox("1. Select directory", (folder_list))

    # generate list of subdirs containing images
    img_folder_name = [
        i for i in list(os.listdir(os.path.join(
            img_root_folder_path, folder_name
        ))) if not i.startswith(".")
    ]
    
    img_folder_name.remove("annotations")
    img_dir = os.path.join(
        img_root_folder_path, folder_name,
        img_folder_name[0]
    )
    json_fpath = os.path.join(
        img_root_folder_path, folder_name,
        "annotations/instances_default.json"
    )
    
    # generate list of images in subdir 
    img_list = list(sorted(os.listdir(img_dir)))
    img_name = st.sidebar.selectbox("2. Select image to plot", (img_list))
    img_fpath = os.path.join(img_dir, img_name)

    option_view = st.sidebar.radio(
        "Select View",
        ('Image with Bounding Boxes', 'Image Only'))   

    img = cv2.imread(img_fpath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_ori = img.copy()
    f = open(json_fpath)
    data = json.load(f)

    img_id = [i['id'] for i in data['images'] if i['file_name'] == img_name][0] # retrieve image ID by filename
    annots = [i['bbox'] for i in data['annotations'] if i['image_id'] == img_id] # retrieve bbox in annotations by image ID
    cats = [i['category_id'] for i in data['annotations'] if i['image_id'] == img_id] # retrieve category ID in annotations by image ID
    box_colours=[(254, 198, 1), (35, 100, 170), (234, 115, 23)] # list of bbox colours - 1 colour per label

    for annot, cat_id in zip(annots, cats):
        if len(annot) == 0:
            continue
        colour = box_colours[cat_id-1]
        img = cv2.rectangle(img, (int(annot[0]), int(annot[1])), (int(annot[0]+annot[2]), int(annot[1]+annot[3])), colour, 2)

    st.caption(f"Image: {img_fpath}")
    st.caption(f"Annotations: {json_fpath}")
    
    if option_view == 'Image with Bounding Boxes':
        st.image(img, use_column_width=True, clamp=True)
        st.text("Yellow = Cell\nOrange = Cell Accumulation (small)\nBlue = Cell Accumulation (large)")
    if option_view == 'Image Only':
        st.image(img_ori)

if __name__ == "__main__":
    main(
        img_root_folder_path="./data/uploaded"
    )
