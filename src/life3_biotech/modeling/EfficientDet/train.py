"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import argparse
from datetime import date
import os
import sys
import tensorflow as tf

# import keras
# import keras.preprocessing.image
# import keras.backend as K
# from keras.optimizers import Adam, SGD

from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from pconst import const

from augmentor.color import VisualEffect
from augmentor.misc import MiscEffect
from model import efficientdet
from losses import smooth_l1, focal, smooth_l1_quad
from efficientnet import BASE_WEIGHTS_PATH, WEIGHTS_HASHES


def makedirs(path):
    # Intended behavior: try to create the directory,
    # pass if the directory exists already, fails otherwise.
    # Meant for Python 2.7/3.n compatibility.
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def get_session():
    """
    Construct a modified tf session.
    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def create_callbacks(training_model, prediction_model, validation_generator, args, model_checkpoint_name, logger):
    """
    Creates the callbacks to use during training.

    Args
        training_model: The model that is used for training.
        prediction_model: The model that should be used for validation.
        validation_generator: The generator for creating validation data.
        args: parseargs args object.
        model_checkpoint_name (str): Filename to save model weights to
        logger (logging.Logger): Logger object used for logging

    Returns:
        A list of callbacks used for training.
    """
    callbacks = []

    tensorboard_callback = None

    if args.tensorboard_dir:
        if tf.version.VERSION > '2.0.0':
            file_writer = tf.summary.create_file_writer(args.tensorboard_dir)
            file_writer.set_as_default()
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir=args.tensorboard_dir,
            histogram_freq=0,
            batch_size=args.batch_size,
            write_graph=True,
            write_grads=False,
            write_images=False,
            embeddings_freq=0,
            embeddings_layer_names=None,
            embeddings_metadata=None
        )
        callbacks.append(tensorboard_callback)

    if args.evaluation and validation_generator:
        from eval.pascal import Evaluate
        evaluation = Evaluate(validation_generator, prediction_model, logger, tensorboard=tensorboard_callback)
        callbacks.append(evaluation)

    # save the model
    if args.snapshots:
        # ensure directory created first; otherwise h5py will error after epoch.
        makedirs(args.snapshot_path)
        checkpoint = keras.callbacks.ModelCheckpoint(
            os.path.join(
                args.snapshot_path,
                model_checkpoint_name
            ),
            verbose=1,
            save_weights_only=const.TRAIN_SAVE_WEIGHTS_ONLY,
            save_best_only=True,
            monitor="val_loss",
            mode='auto'
        )
        callbacks.append(checkpoint)

    if const.LR_SCHEDULER == 'reduce_on_plateau':
        callbacks.append(keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=const.LR_REDUCE_FACTOR,
            patience=const.LR_REDUCE_PATIENCE,
            verbose=1,
            mode='auto',
            min_delta=const.LR_MIN_DELTA,
            cooldown=0,
            min_lr=0
        ))

    if const.TRAIN_EARLY_STOPPING:
        es = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=const.TRAIN_EARLY_STOP_PATIENCE, verbose=1)
        callbacks.append(es)

    return callbacks


def create_generators(args):
    """
    Create generators for training and validation.

    Args
        args: parseargs object containing configuration for generators.
        preprocess_image: Function that preprocesses an image for the network.
    """
    common_args = {
        'batch_size': args.batch_size,
        'phi': const.ED_TRAIN_BACKBONE,
        'detect_text': args.detect_text,
        'detect_quadrangle': args.detect_quadrangle
    }

    # create random transform generator for augmenting training data
    if args.random_transform:
        misc_effect = MiscEffect()
        visual_effect = VisualEffect()
    else:
        misc_effect = None
        visual_effect = None

    from generators.csv_ import CSVGenerator
    train_generator = CSVGenerator(
        const.TRAIN_ANNOTATIONS_PATH,
        misc_effect=misc_effect,
        visual_effect=visual_effect,
        **common_args
    )

    if args.compute_val_loss and const.VAL_ANNOTATIONS_PATH:
        validation_generator = CSVGenerator(
            const.VAL_ANNOTATIONS_PATH,
            shuffle_groups=False,
            **common_args
        )
    else:
        validation_generator = None

    return train_generator, validation_generator


def check_args(parsed_args):
    """
    Function to check for inherent contradictions within parsed arguments.
    For example, batch_size < num_gpus
    Intended to raise errors prior to backend initialisation.

    Args
        parsed_args: parser.parse_args()

    Returns
        parsed_args
    """

    if parsed_args.gpu and parsed_args.batch_size < len(parsed_args.gpu.split(',')):
        raise ValueError(
            "Batch size ({}) must be equal to or higher than the number of GPUs ({})".format(parsed_args.batch_size,
                                                                                             len(parsed_args.gpu.split(
                                                                                                 ','))))

    return parsed_args


def parse_args(args):
    """
    Parse the arguments.
    """
    today = str(date.today())
    parser = argparse.ArgumentParser(description='Simple training script for training a EfficientDet network.')

    parser.add_argument('--train_annotations_path', help='Path to CSV file containing annotations for training.')
    parser.add_argument('--val-annotations-path',
                            help='Path to CSV file containing annotations for validation (optional).')
    parser.add_argument('--detect-quadrangle', help='If to detect quadrangle.', action='store_true', default=False)
    parser.add_argument('--detect-text', help='If is text detection task.', action='store_true', default=False)

    parser.add_argument('--snapshot', help='Resume training from a snapshot.')
    parser.add_argument('--freeze_backbone', help='Freeze training of backbone layers.', action='store_true')
    parser.add_argument('--freeze_bn', help='Freeze training of BatchNormalization layers.', action='store_true')
    parser.add_argument('--weighted_bifpn', help='Use weighted BiFPN', action='store_true', default=True)

    parser.add_argument('--batch_size', help='Size of the batches.', default=1, type=int)
    # parser.add_argument('--phi', help='Hyper parameter phi', default=0, type=int, choices=(0, 1, 2, 3, 4, 5, 6))
    parser.add_argument('--gpu', help='Id of the GPU to use (as reported by nvidia-smi).', default=0)
    parser.add_argument('--epochs', help='Number of epochs to train.', type=int, default=100)
    parser.add_argument('--steps', help='Number of steps per epoch.', type=int, default=10000)
    # parser.add_argument('--learning_rate', help='Learning rate to use for the optimizer function.', type=float, default=0.001)
    
    parser.add_argument('--snapshot-path',
                        help='Path to store snapshots of models during training',
                        default='checkpoints/{}'.format(today))
    parser.add_argument('--tensorboard-dir', help='Log directory for Tensorboard output',
                        default=None)
    parser.add_argument('--no-snapshots', help='Disable saving snapshots.', dest='snapshots', action='store_false')
    parser.add_argument('--no-evaluation', help='Disable per epoch evaluation.', dest='evaluation', action='store_false')
    parser.add_argument('--random-transform', help='Randomly transform image and annotations.', action='store_true', default=False)
    parser.add_argument('--compute_val_loss', help='Compute validation loss during training', dest='compute_val_loss',
                        action='store_true')

    # Fit generator arguments
    parser.add_argument('--multiprocessing', help='Use multiprocessing in fit_generator.', action='store_true')
    parser.add_argument('--workers', help='Number of generator workers.', type=int, default=1)
    parser.add_argument('--max-queue-size', help='Queue length for multiprocessing workers in fit_generator.', type=int,
                        default=10)
    return check_args(parser.parse_known_args(args)[0])

def get_model_path(current_datetime, args):
    return args.snapshot_path, f'efficientdet_b{const.ED_TRAIN_BACKBONE}_{current_datetime}.h5'

def main(current_datetime, logger, args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    # create the generators
    train_generator, validation_generator = create_generators(args)

    num_classes = train_generator.num_classes()
    num_anchors = train_generator.num_anchors

    # optionally choose specific GPU
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    # K.set_session(get_session())

    model, prediction_model = efficientdet(const.ED_TRAIN_BACKBONE,
                                           num_classes=num_classes,
                                           num_anchors=num_anchors,
                                           weighted_bifpn=args.weighted_bifpn,
                                           freeze_bn=args.freeze_bn,
                                           detect_quadrangle=args.detect_quadrangle
                                           )
    # load pretrained weights
    if args.snapshot:
        if args.snapshot == 'imagenet':
            model_name = 'efficientnet-b{}'.format(const.ED_TRAIN_BACKBONE)
            file_name = '{}_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5'.format(model_name)
            file_hash = WEIGHTS_HASHES[model_name][1]
            weights_path = keras.utils.get_file(file_name,
                                                BASE_WEIGHTS_PATH + file_name,
                                                cache_subdir='models',
                                                file_hash=file_hash)
            model.load_weights(weights_path, by_name=True)
        else:
            logger.info('Loading model, this may take a second...')
            model.load_weights(args.snapshot_path + args.snapshot, by_name=True)

    # freeze backbone layers
    if args.freeze_backbone:
        # 227, 329, 329, 374, 464, 566, 656
        for i in range(1, [227, 329, 329, 374, 464, 566, 656][const.ED_TRAIN_BACKBONE]):
            model.layers[i].trainable = False

    # if args.gpu and len(args.gpu.split(',')) > 1:
    #     model = keras.utils.multi_gpu_model(model, gpus=list(map(int, args.gpu.split(','))))

    # compile model
    model.compile(optimizer=Adam(learning_rate=const.INITIAL_LR), loss={
        'regression': smooth_l1_quad() if args.detect_quadrangle else smooth_l1(),
        'classification': focal()
    }, )

    # print(model.summary())

    _, model_checkpoint_name = get_model_path(current_datetime, args)
    # create the callbacks
    callbacks = create_callbacks(
        model,
        prediction_model,
        validation_generator,
        args,
        model_checkpoint_name,
        logger
    )

    if not args.compute_val_loss:
        validation_generator = None
    elif args.compute_val_loss and validation_generator is None:
        raise ValueError('When you have no validation data, you should not specify --compute-val-loss.')

    # start training
    return model.fit(
        x=train_generator,
        steps_per_epoch=args.steps,
        initial_epoch=0,
        epochs=args.epochs,
        verbose=2,
        callbacks=callbacks,
        workers=args.workers,
        use_multiprocessing=args.multiprocessing,
        max_queue_size=args.max_queue_size,
        validation_data=validation_generator
    )
