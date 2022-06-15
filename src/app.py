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


def main(inp):
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
    export_result = batch_inference.single_inferencing(inp)
    # get image with predictions
    img_output = export_result.get("image")
    return img_output


# define input and output
imagein_path = gr.inputs.Image(label="Image Input", type="filepath")
imageout = gr.outputs.Image(label="Predicted Output", type="pil")

# get params into environmet
get_params()

# launch the app
demo = gr.Interface(
    fn=main, inputs=imagein_path, outputs=imageout, allow_flagging="never"
)
demo.launch()
