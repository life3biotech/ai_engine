"""Any miscellaneous utilities/functions to assist the model
training workflow are to be contained here."""

import os
import hydra
import tensorflow as tf

from pconst import const

def export_model(model):
    """Serialises and exports the trained model.

    Parameters
    ----------
    model : tf.keras.Model
        Trained model.
    """

    model_file_path = os.path.\
        join(hydra.utils.get_original_cwd(),
            "models/text-classification-model")

    model.save(model_file_path)


def load_model(path_to_model):
    """Function to load the predictive model.

    A sample utility function to be used for loading a Keras model
    saved in 'SavedModel' format. This function can be customised
    for more complex loading steps.

    Parameters
    ----------
    path_to_model : str
        Path to a directory containing a Keras model in
        'SavedModel' format.

    Returns
    -------
    Keras model instance
        An object with the compiled model.
    """

    loaded_model = tf.keras.models.load_model(path_to_model)
    return loaded_model

def transform_args(config):
    """This function transforms the configurations for the given model into a form that can be passed as hyperparameters to the training process of the model.

    Args:
        cfg (dict): Dictionary containing the configurations loaded from the main config file

    Returns:
        object: Arguments to be passed to the training process of the given model. Returns a list of strings if EfficientDet or RetinaNet is the chosen model.
    """

    if const.TRAIN_MODEL_NAME == 'efficientdet':
        args = ['--compute-val-loss']
        for items in config.items():
            key = str(items[0])
            value = str(items[1])
            if value.lower() not in ['true', 'false']:  # if value is not Boolean, append both key & value to list
                args.append('--' + key)
                args.append(value)
            elif value.lower() == 'true':  # if value is True, append only the key to list
                args.append('--' + key)
        return args
