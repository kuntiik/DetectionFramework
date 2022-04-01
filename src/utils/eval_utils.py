from icevision.all import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from typing import List, Any
from src.datamodules.DentalCaries import DentalCariesDataModule
import copy

# def show_boxes(img, predictions, ground_truth):
# TODO get inspired by this and revrite to make it more general
# https://github.com/airctic/icevision/blob/master/icevision/tfms/albumentations/albumentations_helpers.py
# https://github.com/airctic/icevision/blob/cc6d6a4a048f6ddda2782b6593dcd6b083a673e4/icevision/tfms/albumentations/albumentations_helpers.py#L140
def get_transform(tfms_list: List[Any], t: str) -> Any:
    """
    Extract transform `t` from `tfms_list`.
    Parameters
    ----------
    tfms_list: list of albumentations transforms.
    t: name (str) of the transform to look for and return from within `tfms_list`.
    Returns
    -------
    The `t` transform if found inside `tfms_list`, otherwise None.
    """
    for el in tfms_list:
        if t in str(type(el)):
            return el
    return None


def func_max_size(
    height: int, width: int, max_size: int, func: Callable[[int, int], int]
) -> Tuple[int, int]:
    scale = max_size / float(func(width, height))
    if scale != 1.0:
        height, width = tuple(dim * scale for dim in (height, width))
    return height, width


def get_size_without_padding(
    tfms: List[Any], before_height, before_width, after_height, after_width
):
    if get_transform(tfms, "Pad") is not None:
        t = get_transform(tfms, "SmallestMaxSize")
        if t is not None:
            presize = t.max_size
            after_height, after_width = func_max_size(before_height, before_width, presize, min)
        t = get_transform(tfms, "LongestMaxSize")
        if t is not None:
            size = t.max_size
            after_height, after_width = func_max_size(before_height, before_width, size, max)
    return after_height, after_width


def inverse_transform_bbox(
    bbox: BBox, tfms: List[Any], before_height, before_width, after_height, after_width
):
    after_height, after_width = get_size_without_padding(
        tfms, before_height, before_width, after_height, after_width
    )
    bbox = copy.deepcopy(bbox)
    pad = np.abs(after_height - after_width) / 2
    h_scale, w_scale = after_height / before_height, after_width / before_width
    if after_height < after_width:
        x1, x2, y1, y2 = bbox.xmin, bbox.xmax, bbox.ymin - pad, bbox.ymax - pad
    else:
        x1, x2, y1, y2 = bbox.xmin - pad, bbox.xmax - pad, bbox.ymin, bbox.ymax

    x1, x2, y1, y2 = (max(x1, 0), min(x2, after_width), max(y1, 0), min(y2, after_height))
    x1, x2, y1, y2 = (x1 / w_scale, x2 / w_scale, y1 / h_scale, y2 / h_scale)
    return BBox.from_xyxy(x1, y1, x2, y2)


# TODO Does not support rectangular target image size (resize)
def inverse_transform_preds(preds, dm: DentalCariesDataModule, stage="val"):

    inversed_preds = []
    hw_info_dataset = dm.valid_hw_info if stage == "val" else dm.train_hw_info
    resize_size = dm.hparams.image_size
    tfms = dm.default_transforms(dm.hparams.image_size)
    for pred, (orig_h, orig_w) in zip(preds, hw_info_dataset):
        inv_pred = copy.deepcopy(pred)
        orig_bboxes = inv_pred.detection.bboxes
        new_bboxes = []
        for bbox in orig_bboxes:
            new_bboxes.append(
                inverse_transform_bbox(bbox, tfms, orig_h, orig_w, resize_size, resize_size)
            )
        inv_pred.detection.bboxes = new_bboxes

        orig_bboxes = inv_pred.ground_truth.detection.bboxes
        new_bboxes = []
        for bbox in orig_bboxes:
            new_bboxes.append(
                inverse_transform_bbox(bbox, tfms, orig_h, orig_w, resize_size, resize_size)
            )
        inv_pred.ground_truth.detection.bboxes = new_bboxes
        inversed_preds.append(inv_pred)
    return inversed_preds


def print_boxes(img, preds: Composite, ground_truth: Composite):
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.imshow(img)
    fig.set_dpi(300)
    boxes = preds.bboxes
    scores = preds.scores
    for box, score in zip(boxes, scores):
        x, y, w, h = box.xywh
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor="r", facecolor="none")
        ax.add_patch(rect)

    gt_boxes = ground_truth.bboxes
    for box in gt_boxes:
        x, y, w, h = box.xywh
        rect = patches.Rectangle((x, y), w, h, linewidth=1, edgecolor="b", facecolor="none")
        ax.add_patch(rect)

    # for box in gt['detection']['bboxes']:
    #     x,y,w,h = box.xywh
    #     rect= patches.Rectangle((x,y), w, h, linewidth=1, edgecolor='b', facecolor = 'none')
    #     ax.add_patch(rect)
    plt.axis("off")
    fig.show()


def generate_detections_json(dm, preds, stage="val", preds2=None, stage2="train"):
    data = {}
    inv_preds = inverse_transform_preds(preds, dm, stage)
    for img_pred in inv_preds:
        boxes_list, scores, labels = [], [], []
        img_id = img_pred.common.record_id
        for bbox, score in zip(img_pred.detection.bboxes, img_pred.detection.scores):
            x1, y1, x2, y2 = bbox.xyxy
            boxes_list.append([x1, y1, x2, y2])
            scores.append(np.double(score))
            labels.append(0)
        data[img_id] = {
            "bboxes": boxes_list,
            "labels": labels,
            "scores": scores,
            "stage": stage,
        }
    if preds2 != None:
        inv_preds2 = inverse_transform_preds(preds2, dm, stage2)
        for img_pred in inv_preds2:
            boxes_list, scores, labels = [], [], []
            img_id = img_pred.common.record_id
            for bbox, score in zip(img_pred.detection.bboxes, img_pred.detection.scores):
                x1, y1, x2, y2 = bbox.xyxy
                boxes_list.append([x1, y1, x2, y2])
                scores.append(np.double(score))
                labels.append(0)
            data[img_id] = {
                "bboxes": boxes_list,
                "labels": labels,
                "scores": scores,
                "stage": stage2,
            }
    return data

