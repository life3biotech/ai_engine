# Post-Processing Module 

Please ensure that you have completed the "Environment Setup" guide before proceeding to follow this guide and you have also followed the `4-inference-module.html`.

The guide provides you an overview on how you could set the parameter settings to detect cells.

## Overview

There are main components in the inference pipeline - model inference and post-processing tasks.
The model inference task is mainly trained or fine-tuned in the model training pipeline. An inference task could be further set according to your requirements of detecting large or small cells. This is done in post-processing task (annotation merging).  

This can be seen in the diagram below:

![Inference Module Process Flow](images/inference-module-flow.png)

## Configuration

The inference pipeline requires the main configuration file (dir: ai_engine\conf\base\pipelines.yml) used to select the parameter settings to produce the predictions. Assuming that you have followed the inference guide to make predictions, this guide focuses on other parameters that you could change. It is thus important to take note of the following parameters under inference section and its sub-sections:

- `# ===Defined cell size===`
This is set based on the default values provided by Life3. It is advisable to place values that commonly agreed amongst Life3's domain experts by adjusting `small_mid_cell_cutoff` and `mid_large_cell_cutoff` to set classification of large, medium and small cell sizes. `um_pixel_mapping` is a scaling factor consisting of image pixel:micrometer. 

- `efficientdet`
This is where you can obtain settings about the model. By default, there are NMS and other bounding box filter functions within the EfficientDet model. We have turned these off as we will carry out this task within a separate post-processing task. You should see the followings settings:

```
max_detections: 400
class_specific_filter: False 
detect_quadrangle: False 
select_top_k: False
```

By setting `max_detections` (also known as bounding box limit) to `400`, we allow a large number of bounding box detections to be flown to the post-processing module. The other EfficientDet parameters (such as `class_specific_filter`, `detect_quadrangle`, `select_top_k`) are set to `False`.

- `# ===Postprocessing parameter===`
This is where we will carry out post-processing tasks on the predictions made from EfficientDet model. One key feature of the inference pipeline is that it allows for tiling/slicing an image into multiple slices/tiles. This allows the model to detect the cell more clearly. The `slice_height` and `slice_width` determine the size of the slice. We have observed the followings after running multiple tests to find the best values:

`384 - Balance, good at detecting big and small object. 256, very good at small object but might miss big object. 512, very good at big object but might miss small object`

The `overlap_height_ratio` and `overlap_width_ratio` are to set the amount of fractional overlap between the slices. For instance, 
an overlap of 0.2 for a window of size 512 yields an overlap of 102 pixels.

The `postprocess_type` refers to the algorithm to be used for merging/eliminating bounding box detections. `NMS` seems to be the preferred option in setting. We have made a modification to the algorithm. Instead of relying on `IOU/IOS` to select the best bounding box detections, we have created a new function with a parameter named `postprocess_bbox_sort` to sort bounding box detections based on their areas. This way, the tightly bounded boxes will be retained. 

The `postprocess_match_metric` and its `postprocess_match_threshold` will be used as a mechanism to retain relevant bounding boxes.

If `inference_slice` is set to `True`, the slicing mechanism will be set. 

```
