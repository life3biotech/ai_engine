import cv2
import gradio as gr
import hydra
import logging
from pconst import const

# from batch_inferencing import main
import life3_biotech as life3
import inferencing as inference

# import sahi


# Get configuration
@hydra.main(config_path="../conf/life3", config_name="pipelines.yml")
def get_params(args):
    """Gets the params into the environment."""
    logger = logging.getLogger(__name__)
    pipeline_conf = life3.config.PipelineConfig(args, logger)


def main(input_image):
    """This main function supports inference on the app.
    It calls the AI model and run a single-image inferencec.

    Args:
        inp (str): input file path
    Returns:
        img_output (np.array): output image with predictions
    """

    # instantiate logging
    logger = logging.getLogger(__name__)
    # instantiate model
    batch_inference = inference.batch_inference.BatchInference(logger)
    # prediction
    export_result, process_df = batch_inference.single_inferencing(input_image)
    # get image with predictions
    img_output = export_result.get("image")

    percentage_small = process_df["tot_small_cell"][0] / process_df["cell_tot"][0] * 100
    percentage_medium = process_df["tot_mid_cell"][0] / process_df["cell_tot"][0] * 100
    percentage_large = process_df["tot_large_cell"][0] / process_df["cell_tot"][0] * 100

    results_text = f"Total Cell Count:  {process_df['cell_tot'][0]}  (100%)\nSmall:                       {process_df['tot_small_cell'][0]}  ({percentage_small:.2f}%)\nMedium:                 {process_df['tot_mid_cell'][0]}  ({percentage_medium:.2f}%)\nLarge:                      {process_df['tot_large_cell'][0]}  ({percentage_large:.2f}%)"
    return img_output, results_text


col_n = [
    "img_filename",
    "cell_tot",
    "cell_type_tot",
    "cell_accum_type_tot",
    "tot_small_cell",
    "tot_mid_cell",
    "tot_large_cell",
]

# get params into environmet
get_params()

# launch the app
demo = gr.Blocks()

title = "Life3 Biotech"
description = """
<center>
Using AI to analyze cell numbers and cell sizes of microalgae on microscope images. Drop an image to infer the number of cells.
<img src="file/src/images/life3.webp" width=200px>
</center>
"""

description_footer = """
<center>
<footer>
Created by AISG.
</footer>
</center>
"""

# define input and output
imagein_path = gr.Image(label="Image Input", type="filepath")
imageout = gr.Image(label="Inferred Output", type="pil", shape=None)
textout = gr.Textbox(label="Cell Info",)

gr.Interface(
    fn=main,
    inputs=imagein_path,
    outputs=[imageout, textout],
    allow_flagging="never",
    allow_screenshot=True,
    title=title,
    description=description,
    article=description_footer,
).launch(inbrowser=True, favicon_path="src/images/favicon.ico")
