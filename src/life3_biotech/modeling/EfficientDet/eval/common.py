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
import numpy as np
import time
from generators.common import Generator
from utils.compute_overlap import compute_overlap
from tqdm import trange


def _compute_ap(recall, precision):
    """
    Compute the average precision, given the recall and precision curves.

    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    Args:
        recall: The recall curve (list).
        precision: The precision curve (list).

    Returns:
        The average precision as computed in py-faster-rcnn.

    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def _get_detections(generator: Generator, model, score_threshold=0.05, max_detections=100, visualize=False, test_set=False):
    """
    Get the detections from the model using the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_class_detections, 5]

    Args:
        generator: The generator used to run images through the model.
        model: The model to run on the images.
        score_threshold: The score confidence threshold to use.
        max_detections: The maximum number of detections to use per image.
        visualize: Boolean flag indicating whether outputs should be visualised. Currently not in use.
        test_set: Boolean flag indicating whether the data in the generator is from the test set.

    Returns:
        A list of lists containing the detections for each image in the generator.

    """
    # pylint: disable=no-member
    all_detections = [[None for i in range(generator.num_classes()) if generator.has_label(i)] for j in range(generator.size())]
    img_count = generator.size()
    total_time = 0

    for i in trange(generator.size(), mininterval=3.0, miniters=20, desc='Running network: '):
        image = generator.load_image(i)
        h, w = image.shape[:2]
        image, scale = generator.preprocess_image(image)

        start = time.time()
        # run network
        boxes, scores, *_, labels = model.predict_on_batch([np.expand_dims(image, axis=0)])
        boxes /= scale
        boxes[:, :, 0] = np.clip(boxes[:, :, 0], 0, w - 1)
        boxes[:, :, 1] = np.clip(boxes[:, :, 1], 0, h - 1)
        boxes[:, :, 2] = np.clip(boxes[:, :, 2], 0, w - 1)
        boxes[:, :, 3] = np.clip(boxes[:, :, 3], 0, h - 1)
        # select indices which have a score above the threshold
        indices = np.where(scores[0, :] > score_threshold)[0]

        # select those scores
        scores = scores[0][indices]
        # find the order with which to sort the scores
        scores_sort = np.argsort(-scores)[:max_detections]
        # select detections
        # (n, 4)
        image_boxes = boxes[0, indices[scores_sort], :]
        # (n, )
        image_scores = scores[scores_sort]
        # (n, )
        image_labels = labels[0, indices[scores_sort]]
        # (n, 6)
        detections = np.concatenate(
            [image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)
        seconds = time.time() - start
        total_time += seconds

        # copy detections to all_detections
        for class_id in range(generator.num_classes()):
            all_detections[i][class_id] = detections[detections[:, -1] == class_id, :-1]

    if test_set:  # calculate FPS only for test set
        fps = img_count / total_time
    else:
        fps = 0
    return all_detections, fps


def _get_annotations(generator):
    """
    Get the ground truth annotations from the generator.

    The result is a list of lists such that the size is:
        all_annotations[num_images][num_classes] = annotations[num_class_annotations, 5]

    Args:
        generator: The generator used to retrieve ground truth annotations.

    Returns:
        A list of lists containing the annotations for each image in the generator.

    """
    # pylint: disable=no-member
    all_annotations = [[None for i in range(generator.num_classes())] for j in range(generator.size())]

    for i in trange(generator.size(), miniters=int(generator.size() / 100), desc='Parsing annotations: '):
        # load the annotations
        annotations = generator.load_annotations(i)

        # copy detections to all_annotations
        for label in range(generator.num_classes()):
            if not generator.has_label(label):
                continue

            all_annotations[i][label] = annotations['bboxes'][annotations['labels'] == label, :].copy()

    return all_annotations


def evaluate(
    generator,
    model,
    logger,
    iou_threshold=[0.5],
    score_threshold=0.01,
    max_detections=100,
    visualize=False,
    test_set=False
):
    """
    Evaluate a given dataset using a given model.

    Args:
        generator: The generator that represents the dataset to evaluate.
        model: The model to evaluate.
        iou_threshold: The threshold used to consider when a detection is positive or negative.
        score_threshold: The score confidence threshold to use for detections.
        max_detections: The maximum number of detections to use per image.
        visualize: Show the visualized detections or not.

    Returns:
        A dict mapping class names to mAP scores.

    """
    metrics_dict = {}
    iou_threshold = list(iou_threshold)
    if len(iou_threshold) == 3:
        iou_threshold_range = np.linspace(start=iou_threshold[0], stop=iou_threshold[1], num=iou_threshold[2])
        iou_threshold_range = (np.round(iou_threshold_range, 2)).tolist()
    else:
        iou_threshold_range = iou_threshold
    logger.info(f'IoU threshold range: {iou_threshold_range}')

    # gather all detections and annotations
    all_detections, fps = _get_detections(generator, model, score_threshold=score_threshold, max_detections=max_detections, visualize=visualize, test_set=test_set)
    metrics_dict['FPS'] = round(fps, 4)
    all_annotations = _get_annotations(generator)
    average_precisions = {}
    num_tp = 0
    num_fp = 0
    total_annot = 0
    map5 = {}
    wap5 = {}

    # process detections and annotations
    for label in range(generator.num_classes()):
        average_precision_list = []  # for each class, create an empty list to store 1 AP value per IoU threshold
        for iou_threshold in iou_threshold_range:
            if not generator.has_label(label):
                continue

            false_positives = np.zeros((0,))
            true_positives = np.zeros((0,))
            scores = np.zeros((0,))
            num_annotations = 0.0

            for i in range(generator.size()):
                detections = all_detections[i][label]
                annotations = all_annotations[i][label]
                num_annotations += annotations.shape[0]
                detected_annotations = []

                for d in detections:
                    scores = np.append(scores, d[4])

                    if annotations.shape[0] == 0:
                        false_positives = np.append(false_positives, 1)
                        true_positives = np.append(true_positives, 0)
                        continue
                    overlaps = compute_overlap(np.expand_dims(d, axis=0), annotations)
                    assigned_annotation = np.argmax(overlaps, axis=1)
                    max_overlap = overlaps[0, assigned_annotation]

                    if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                        false_positives = np.append(false_positives, 0)
                        true_positives = np.append(true_positives, 1)
                        detected_annotations.append(assigned_annotation)
                    else:
                        false_positives = np.append(false_positives, 1)
                        true_positives = np.append(true_positives, 0)

            # no annotations -> AP for this class is 0 (is this correct?)
            if num_annotations == 0:
                average_precisions[label] = 0, 0
                continue

            # sort by score
            indices = np.argsort(-scores)
            false_positives = false_positives[indices]
            true_positives = true_positives[indices]

            # compute false positives and true positives
            false_positives = np.cumsum(false_positives)
            true_positives = np.cumsum(true_positives)

            if false_positives.shape[0] == 0:
                num_fp += 0
            else:
                num_fp += false_positives[-1]
            if true_positives.shape[0] == 0:
                num_tp += 0
            else:
                num_tp += true_positives[-1]

            # compute recall and precision
            recall = true_positives / num_annotations
            precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

            # compute average precision
            average_precision = _compute_ap(recall, precision)
            average_precision_list.append(average_precision)
            logger.debug(f'AP at IoU threshold {iou_threshold}: {round(average_precision,4)}')

        average_precision_class = np.mean(average_precision_list)
        if len(iou_threshold_range) > 1:
            ap5 = average_precision_list[iou_threshold_range.index(0.5)]
            if ap5 is not None:
                map5[label] = ap5
                logger.info(f'Class {label} mAP@0.5: {ap5}')
        logger.info(f'Class {label} mAP across thresholds: {average_precision_class}')
        average_precisions[label] = average_precision_class, num_annotations
        total_annot += num_annotations
    logger.info(f'Total # of annotations: {total_annot}')
    # compute per class average precision
    total_instances = []
    precisions = []
    weighted_ap = 0.0
    for label, (average_precision, num_annotations) in average_precisions.items():
        label_name = generator.label_to_name(label)
        logger.info(f'{round(num_annotations,0)} instances of class {label_name} with average precision: {round(average_precision, 4)}')
        total_instances.append(num_annotations)
        precisions.append(average_precision)
        proportion = num_annotations / total_annot
        weighted_ap += average_precision * proportion
        if len(map5) > 0:
            wap5[label] = map5[label] * proportion
        metrics_dict['AP_' + label_name] = round(average_precision, 4)
    # compute mean AP & weighted AP
    mean_ap = sum(precisions) / sum(x > 0 for x in total_instances)
    if test_set:
        metrics_dict['test_mAP'] = round(mean_ap, 4)
        metrics_dict['test_wAP'] = round(weighted_ap, 4)
        if len(map5) > 0 and len(wap5) > 0:
            metrics_dict['test_mAP0.5'] = round(np.mean(list(map5.values())), 4)
            metrics_dict['test_wAP0.5'] = round(np.mean(list(wap5.values())), 4)
    else:
        metrics_dict['mAP'] = round(mean_ap, 4)
        metrics_dict['wAP'] = round(weighted_ap, 4)
    return metrics_dict


def main(current_datetime, logger, args=None, model_path=None):
    from generators.csv_ import CSVGenerator
    from pconst import const
    from model import efficientdet
    from train import parse_args
    import os
    import sys

    # parse arguments
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)

    metrics_dict = {}
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    common_args = {
        'batch_size': const.EVAL_BATCH_SIZE,
        'phi': const.ED_TRAIN_BACKBONE,
    }
    test_generator = CSVGenerator(
        const.TEST_ANNOTATIONS_PATH,
        **common_args
    )
    if model_path is None:
        model_path = f'{const.SAVED_MODEL_PATH}efficientdet_b{const.ED_TRAIN_BACKBONE}_{current_datetime}.h5'
    num_classes = test_generator.num_classes()
    model, prediction_model = efficientdet(phi=const.ED_TRAIN_BACKBONE, num_classes=num_classes, weighted_bifpn=args.weighted_bifpn)
    logger.info(f'Loading weights from {model_path}')
    prediction_model.load_weights(model_path, by_name=True)
    logger.info('Starting evaluation...')
    metrics_dict = evaluate(
        test_generator,
        prediction_model,
        logger,
        iou_threshold=const.EVAL_IOU_THRESHOLD,
        score_threshold=const.EVAL_SCORE_THRESHOLD,
        visualize=False,
        test_set=True)
    return metrics_dict


if __name__ == '__main__':
    main()