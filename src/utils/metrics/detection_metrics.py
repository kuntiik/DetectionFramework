import torch
from collections import Counter

# inspiration by https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/object_detection/metrics
def intersection_over_union(preds, labels):
    # to avoid expand dims we need to slice the torch array
    b1_x1 = preds[..., 0:1]
    b1_y1 = preds[..., 1:2]
    b1_x2 = preds[..., 2:3]
    b1_y2 = preds[..., 3:4]

    b2_x1 = labels[..., 0:1]
    b2_y1 = labels[..., 1:2]
    b2_x2 = labels[..., 2:3]
    b2_y2 = labels[..., 3:4]

    tl_corner_x = torch.max(b1_x1, b2_x1)
    tl_corner_y = torch.max(b1_y1, b2_y1)

    br_corner_x = torch.min(b1_x2, b2_x2)
    br_corner_y = torch.min(b1_y2, b2_y2)

    intersection = (br_corner_x - tl_corner_x).clamp(0) - (br_corner_y - tl_corner_y).clamp(0)

    pred_box_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    label_box_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)

    return intersection / (pred_box_area + label_box_area - intersection)


def non_max_suppression(bboxes, iou_threshold):
    # the format should be like [class, conf, *box]
    assert type(bboxes) == list

    bboxes = sorted(bboxes, key=lambda x: x[1], reversed=True)
    bboxes_suppressed = []

    while bboxes:
        chosen_box = bboxes.pop(0)
        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or intersection_over_union(torch.tensor(chosen_box[2:]), torch.tensor(box[2:]))
            < iou_threshold
        ]
        bboxes_suppressed.append(chosen_box)

    return bboxes_suppressed


def mean_average_precision(preds, labels, iou_threshold=0.5, num_classes=1):
    # preds = [[img_idx, class, conf, *box], ...]

    average_precisions = []
    epsilon = 1e-6

    for c in range(num_classes):
        detections = [pred for pred in preds if pred[1] == c]
        ground_truths = [label for label in labels if labels[1] == c]

        # how many boxes per image
        count_bboxes = Counter([gt[0] for gt in ground_truths])
        # to assign each bbox only once
        for key, val in count_bboxes.items():
            count_bboxes[key] = torch.zeros(val)

        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        for detection_idx, detection in enumerate(detections):
            ground_truth_img = [bbox for bbox in ground_truths if bbox[0] == detection[0]]
            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(torch.tensor(detection[3:]), torch.tensor(gt[2:]))
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold and count_bboxes[detection[0]][best_gt_idx] == 0:
                count_bboxes[detection[0]][best_gt_idx] = 1
                TP[detection_idx] = 1
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        precision = torch.cat((torch.tensor([1]), precisions))

        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)

