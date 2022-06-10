# Inference Module 

Please ensure that you have completed the "Environment Setup" guide before proceeding to follow this guide.

## Overview

The diagram below shows the process flow of the inference module.
There are 2 parts to inference.
1. The first part is evaluation and calibration module.
2. The 2nd part is inference module.

The entrypoint of the evaluation and calibration module is the script `src/eval_model.py` by default.

The entrypoint of the inference module is the script `src/batch_inferencing.py` by default.

![Inference Module Process Flow](images/inference-module-flow.png)

## Configuration

The main configuration file used to customise the AI engine is `pipelines.yml`, located in `conf/life3` subfolder.

### General inference configuration

In `pipelines.yml`, the following parameters in the `inference` section are configurable.

### General Inference configuration

<table>
<tr>
<th>

<div>

Constant (`const.`)

</div></th>
<th>

<div>Parameter</div></th>
<th>

<div>Type</div></th>
<th>

<div>Description</div></th>
<th>

<div>Default Value</div></th>
</tr>
<tr>
<td>

### _Defined Cell Size_
</td>
<td>

</td>
<td>

</td>
<td>

</td>
<td>

</td>
</tr>
<tr>
<td>

<div>UM_PIXEL_MAPPING</div></td>
<td>

<div>um_pixel_mapping</div></td>
<td>

<div>float</div></td>
<td>Map one pixel of image to the corresponding micrometer for cell size. Unit: micrometer/pixel </td>
<td>

<div>
<div>0.369763541667</div></td>
</div></td>
</tr>
<tr>
<td>

<div>

<div>SMALL_MID_CELL_CUTOFF</div></td>
<td>

<div>small_mid_cell_cutoff</div></td>
<td>

<div>float</div></td>
<td>The upper bound of the small cell and lower bound of the mid-size cell in micrometer (μm) </td>
<td>

<div>
<div>4.0</div></td>
</div></td>
</tr>
<tr>
<td>

<div>

<div>MID_LARGE_CELL_CUTOFF</div></td>
<td>

<div>mid_large_cell_cutoff</div></td>
<td>

<div>float</div></td>
<td>The upper bound of the mid-size cell and lower bound of the large-size cell in micrometer (μm) </td>
<td>

<div>
<div>8.0</div></td>
</div></td>
</tr>
<tr>
<td>

<div>

### _Input/Output_
</td>
<td>

</td>
<td>

</td>
<td>

</td>
<td>

</td>
</tr>
<tr>
<td>

<div>INFERENCE_MODEL_PATH</div></td>
<td>

<div>model_path</div></td>
<td>

<div>str</div></td>
<td>Absolute or relative path pointing to the model weight to be used.</td>
<td>

<div>

</div></td>
</tr>
<tr>
<td>

<div>

<div>IMAGE_INPUT_DIR</div>
</div></td>
<td>

<div>

<div>image_input_dir</div>
</div></td>
<td>

<div>

<div>str</div>
</div></td>
<td>

<div>

<div>Absolute or relative path pointing to the input image directory for inference/prediction. File extensions = ".png" </div>
</div></td>
<td>

<div>

</div></td>
</tr>
<tr>
<td>

<div>

<div>CSV_OUTPUT_DIR</div>
</div></td>
<td>

<div>

<div>csv_output_dir</div>
</div></td>
<td>

<div>

<div>str</div>
</div></td>
<td>

<div>Absolute or relative path pointing to the output inferred/predicted annotated csv directory. File extensions = ".csv" Note: Consolidated cell count info for all images are saved as `predicted_results.csv` in the same folder</div></td>
<td>

<div>

</div></td>
</tr>
<tr>
<td>

<div>

<div>SAVE_OUTPUT_IMAGE</div>
</div></td>
<td>

<div>

<div>save_output_image</div>
</div></td>
<td>

<div>

<div>boolean</div>
</div></td>
<td>

<div>Determines whether to save inferred/predicted image.</div></td>
<td>

<div>True</div></td>
</tr>
<tr>
<td>

<div>

<div>SAVE_OUTPUT_IMAGE_SHOWLABEL</div>
</div></td>
<td>

<div>

<div>save_output_image_showlabel</div>
</div></td>
<td>

<div>

<div>boolean</div>
</div></td>
<td>

<div>Determines whether to save inferred/predicted image with prediction text label on each detected cell.</div></td>
<td>

<div>True</div></td>
</tr>
<tr>
<td>

<div>

<div>SAVE_OUTPUT_IMAGE_SHOW_CELLCOUNT</div>
</div></td>
<td>

<div>

<div>save_output_image_show_cellcount</div>
</div></td>
<td>

<div>

<div>boolean</div>
</div></td>
<td>

<div>Determines whether to save inferred/predicted image with prediction cell count text label on top left corner of the image.</div></td>
<td>

<div>True</div></td>
</tr>
<tr>
<td>

<div>

<div>IMAGE_OUTPUT_DIR</div>
</div></td>
<td>

<div>

<div>image_output_dir</div>
</div></td>
<td>

<div>

<div>str</div>
</div></td>
<td>

<div>Absolute or relative path pointing to the output inferred/predicted image with cell bounding box drawn.</div></td>
<td>

</td>
</tr>
<tr>
<td>

### _Model parameter_
</td>
<td>

</td>
<td>

</td>
<td>

</td>
<td>

</td>
</tr>
<tr>
<td>

<div>INFERENCE_BACKBONE</div></td>
<td>

<div>inference_backbone</div></td>
<td>

<div>int</div></td>
<td>

<div>Compound coefficient used to scale up EfficientNet, the backbone network. Possible values: 0, 1, 2, 3, 4, 5, 6.</div></td>
<td>

<div>0</div></td>
</tr>
<tr>
<td>

<div>INFERENCE_CONFIDENCE_THRESH</div></td>
<td>

<div>confidence_threshold</div></td>
<td>

<div>float</div></td>
<td>

<div>The confidence threshold is used to assess the probability of the object class appearing in the bounding box.</div></td>
<td>0.33</td>
</tr>
<tr>
<td>

<div>INFERENCE_RUN_NMS</div></td>
<td>

<div>run_nms</div></td>
<td>

<div>boolean</div></td>
<td>

<div>Determines whether the non-maximum Suppression is activated during inference.</div></td>
<td>

<div>

False

</div></td>
</tr>
<tr>
<td>

<div>INFERENCE_NMS_THRESH</div></td>
<td>

<div>nms_threshold</div></td>
<td>

<div>float</div></td>
<td>

<div>Non max suppression is a technique used mainly in object detection that aims at selecting the best bounding box out of a set of overlapping boxes.</div></td>
<td>

<div>0.2</div></td>
</tr>
<tr>
<td>

### _Postprocessing parameter_
</td>
<td>

</td>
<td>

</td>
<td>

</td>
<td>

</td>
</tr>
<tr>
<td>

<div>SLICE_HEIGHT</div></td>
<td>

<div>slice_height</div></td>
<td>

<div>int</div></td>
<td>

<div>The height of the image to be sliced (Suggestion: 256, 384, 512)
  384 - Balance, good at detecting big and small object. 256, very good at small object but might miss big object. 512, very good at big object but might miss small object</div></td>
<td>256</td>
</tr>
<tr>
<td>

<div>SLICE_WIDTH</div></td>
<td>

<div>slice_width</div></td>
<td>

<div>int</div></td>
<td>

<div>The width of the image to be sliced (Suggestion: 256, 384, 512)
  384 - Balance, good at detecting big and small object. 256, very good at small object but might miss big object. 512, very good at big object but might miss small object</div></td>
<td>256</td>
</tr>
<tr>
<td>OVERLAP_HEIGHT_RATIO</td>
<td>overlap_height_ratio</td>
<td>float</td>
<td>Fractional overlap in height of each window (e.g. an overlap of 0.2 for a window of size 512 yields an overlap of 102 pixels).</td>
<td>0.15</td>
</tr>
<tr>
<td>OVERLAP_WIDTH_RATIO</td>
<td>overlap_width_ratio</td>
<td>float</td>
<td>Fractional overlap in width of each window (e.g. an overlap of 0.2 for a window of size 512 yields an overlap of 102 pixels).</td>
<td>0.15</td>
</tr>
<tr>
<td>POSTPROCESS_TYPE</td>
<td>postprocess_type</td>
<td>str</td>
<td>Type of the postprocess to be used after sliced inference while merging/eliminating predictions. Options are 'NMM', 'GREEDYNMM' or 'NMS'. Default is 'GREEDYNMM'.</td>
<td>'NMS'</td>
</tr>
<tr>
<td>POSTPROCESS_BBOX_SORT</td>
<td>postprocess_bbox_sort</td>
<td>str</td>
<td>If True, sort bounding box according to area (Prioritise tighter bounding box). False sort bounding box acccording to score.</td>
<td>True</td>
</tr>
<tr>
<td>POSTPROCESS_MATCH_METRIC</td>
<td>postprocess_match_metric</td>
<td>str</td>
<td>

Metric to be used during object prediction matching after sliced prediction. <br>IOU = intersection over union.<br>IOS =

intersection over smaller area. Options are 'IOU' or 'IOS'
</td>
<td>"IOS"</td>
</tr>
<tr>
<td>POSTPROCESS_MATCH_THRESHOLD</td>
<td>postprocess_match_threshold</td>
<td>float</td>
<td>Sliced predictions having higher iou than postprocess_match_threshold will be postprocessed after sliced prediction.</td>
<td>0.01</td>
</tr>
</table>

---
---

### **Things to do before inference**
Before any inference or prediction on images can be performed, some parameters must be configured according to your environment settings. Open the file `pipelines.yml` and edit the following parameters to your current environment. 
Note: Consolidated cell count info for all images are saved as `predicted_results.csv` in the same folder as csv_output_dir

```plaintext
  model_path: "C:\\ai_engine\\models\\efficientdet_b0_20220510_201515.h5"
  image_input_dir: "C:\\ai_engine\\data\\inference\\input\\"
  csv_output_dir: "C:\\ai_engine\\data\\inference\\output\\"
  save_output_image: True
  image_output_dir: "C:\\ai_engine\\data\\inference\\output\\"
```

**_Note:_** Models trained on a single class (i.e. "cell") can only perform single class inference. If such a model is selected, another parameter in `pipelines.yml` must be amended. Under `data_prep`, comment out the line `'cell accumulation': 1` as shown below.

<div>

```plaintext
data_prep: 
 .
 . 
 .
 class_map: { 
        'cell': 0, 
        # 'cell accumulation': 1 
  }
```

</div>




### **Running the evaluation and calibration pipeline** 
(Only required to execute once until the next model change or parameters update)

**Evaluation and calibration** of the model is required every time a new model or new parameters are applied. You are required to execute this step once only until the next modification.

1. On the terminal, change to your working directory with the following command:

<div>

```plaintext
cd C:\ai_engine
```

</div>2. Activate the conda environment with the following command:

<div>

```plaintext
conda activate life3-biotech
```

</div>3. If there are known updates to the dependencies, update the conda environment by running:

<div>

```plaintext
conda env update --file life3-biotech-conda-env.yml
```

</div>4. Finally, run the following command to start the inference pipeline:

<div>

```plaintext
python3 -m src.eval_model
```
or,
```plaintext
python -m src.eval_model
```

</div>
  


### **Running the inference pipeline**

1. On the terminal, change to your working directory with the following command:

<div>

```plaintext
cd C:\ai_engine
```

</div>2. Activate the conda environment with the following command:

<div>

```plaintext
conda activate life3-biotech
```

</div>3. If there are known updates to the dependencies, update the conda environment by running:

<div>

```plaintext
conda env update --file life3-biotech-conda-env.yml
```

</div>4. Finally, run the following command to start the inference pipeline:

<div>

```plaintext
python3 -m src.batch_inferencing
```
or,
```plaintext
python -m src.batch_inferencing
```

</div>

### **Things to look out for**


1. At the end of the inference, if you see the following warning message in the terminal, it means some of the parameters in pipelines.yml have changed and might have affected the cell size calibration. In this case, it is recommeded to rerun `python -m src.eval_model` to recalibrate the cell size measurement.
```
[2022-06-09 16:27:47,064][__main__][WARNING] - Config Parameters have changed, pipelines.yml differ from calibrated_params.csv. Affected parameters: postprocess_bbox_sort
[2022-06-09 16:27:47,064][__main__][WARNING] - Config Parameters have changed, please rerun eval_model to recalibrate cellsize.
```

1. At the end of the inference, if you see the following warning message in the terminal, it means you have not perform model evaluation and calibration. Please run `python -m src.eval_model` to calibrate the optimal cell size. 
```

```

