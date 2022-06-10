import cv2
import numpy as np
import hydra
import os
import time
import logging
from pconst import const
from pathlib import Path
from PIL import Image
from io import BytesIO
from life3_biotech.config import PipelineConfig
from life3_biotech import general_utils
from model import efficientdet
from utils import preprocess_image, postprocess_boxes
from utils.draw_boxes import draw_boxes


def process_image(image_file):
    """This function process the given PIL Image object.

    Args:
        image_url (str): URL to image file

    Returns:
        numpy.array: Numpy array representing the original image
        numpy.array: Numpy array representing the image after RGB conversion
    """
    pil_image = Image.open(BytesIO(image_file)).convert("RGB")
    image = np.asarray(pil_image)
    src_image = image.copy()
    image = image[:, :, ::-1].copy()
    return src_image, image


def load_image_from_path(image_filepath):
    """This function loads an image from the given file path using cv2.

    Args:
        image_filepath (str): Path to image file

    Returns:
        numpy.array: Numpy array representing the original image
        numpy.array: Numpy array representing the image after RGB conversion
    """
    image = cv2.imread(image_filepath)
    src_image = image.copy()
    image = image[:, :, ::-1]
    return src_image, image


def get_image_list(logger):
    """This function initialises the list of images for EfficientDet to perform inference on.
    In `single` or `rest_api` inference mode, the list will contain a single image file path/URL.

    Returns:
        list: List of image file paths or URLs
    """
    image_list = []
    for filename in os.listdir(const.INFERENCE_INPUT_PATH):
        img_file = Path(const.INFERENCE_INPUT_PATH, filename)
        if img_file.exists() and img_file.is_file():
            image_list.append(img_file)
        else:
            logger.error(f"Invalid image file path: {img_file}")
    return image_list


def load_model(logger):
    """This function initialises the saved EfficientDet model weights to be used for inference.

    Returns:
        tensorflow.keras.models.Model: Keras model object
        OrderedDict: dictionary containing the mapping of class IDs to class names
    """
    weighted_bifpn = True
    global temp_model_location
    logger.info(f"Classes: {const.CLASS_MAP_REVERSE}")
    num_classes = len(const.CLASS_MAP)
    _, model = efficientdet(
        phi=const.ED_INFERENCE_BACKBONE,
        weighted_bifpn=weighted_bifpn,
        num_classes=num_classes,
        score_threshold=const.INFERENCE_CONFIDENCE_THRESH,
    )
    model.load_weights(const.INFERENCE_MODEL_PATH, by_name=True)
    logger.info(
        f"Inferencing on backbone B{const.ED_INFERENCE_BACKBONE} with saved model weights: {const.INFERENCE_MODEL_PATH}"
    )
    return model


@hydra.main(config_path="../../../../conf/local", config_name="pipelines.yml")
def main(args):
    """This function iterates through the list of images, preprocesses each image, calls the prediction function on it and preprocesses the results.

    Returns:
        dict: Output dictionary to be formatted to JSON if inference mode is `rest_api`. Otherwise, None is returned.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    os.chdir(hydra.utils.get_original_cwd())

    logger = logging.getLogger(__name__)
    logger.info("Setting up logging configuration.")
    logger_config_path = os.path.join(
        hydra.utils.get_original_cwd(), "conf/base/logging.yml"
    )
    general_utils.setup_logging(logger_config_path)

    pipeline_conf = PipelineConfig(args, logger)

    images = get_image_list(logger)
    logger.info(f"Number of images: {len(images)}")

    total_inf_time = 0
    total_preprocess_time = 0

    model = load_model(logger)

    for i in images:
        output, preprocess_time, inf_time = predict(model, i, logger)
        total_preprocess_time += preprocess_time
        total_inf_time += inf_time

    if total_inf_time > 0:
        logger.info(f"Time taken: {round(total_inf_time,2)} seconds")
        logger.info(f"FPS: {round(len(images) / total_inf_time,2)}")
        logger.info(
            f"Avg preprocessing time: {round(total_preprocess_time/len(images),2)} seconds"
        )

    return output


def predict(model, image_file, logger):
    """This function preprocesses the given image and performs inference on it using the given model weights.

    Args:
        model (tensorflow.keras.models.Model): Keras inference model object
        image_file (str): URL of image file

    Returns:
        str: Formatted output for response if inference mode is `rest_api`
        float: Floating point number representing time taken to preprocess given image before inference
        float: Floating point number representing time taken to predict on given image
    """
    colors = [(244, 223, 156), (164, 232, 241), (119, 118, 188)]
    image_size = const.ED_IMAGE_SIZES[const.ED_INFERENCE_BACKBONE]

    output = None

    preprocess_time = time.time()
    filename = image_file.name
    logger.info(f"Loading image from {image_file}")
    src_image, image = load_image_from_path(str(image_file))
    h, w = image.shape[:2]
    image, scale = preprocess_image(image, image_size=image_size)

    preprocess_time = time.time() - preprocess_time
    # run network
    start = time.time()
    boxes, scores, labels = model.predict_on_batch([np.expand_dims(image, axis=0)])
    boxes, scores, labels = np.squeeze(boxes), np.squeeze(scores), np.squeeze(labels)
    inf_time = time.time() - start

    boxes = postprocess_boxes(boxes=boxes, scale=scale, height=h, width=w)

    # select indices which have a score above the threshold
    indices = np.where(scores[:] > const.INFERENCE_CONFIDENCE_THRESH)[0]

    # select those detections
    boxes = boxes[indices]
    labels = labels[indices]
    if len(boxes) == 0:
        logger.info(f"No inference results for image: {filename}")
    else:
        logger.info(f"Drawing {len(boxes)} boxes...")
        output_image = draw_boxes(
            src_image, boxes, scores, labels, colors, const.CLASS_MAP_REVERSE
        )
        if const.INFERENCE_SAVE_OUTPUT:
            if not os.path.exists(const.INFERENCE_OUTPUT_PATH):
                os.makedirs(const.INFERENCE_OUTPUT_PATH)
            output_filepath = str(Path(const.INFERENCE_OUTPUT_PATH, filename))
            logger.info(f"Writing output to: {output_filepath}")
            saved = cv2.imwrite(output_filepath, output_image)
            if saved:
                logger.info(
                    f"Saved inferenced image to {const.INFERENCE_OUTPUT_PATH}{filename}"
                )

    return output, preprocess_time, inf_time


if __name__ == "__main__":
    main()
