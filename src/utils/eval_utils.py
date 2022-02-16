from icevision.all import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# def show_boxes(img, predictions, ground_truth):

def inverse_transform_preds(pred : Composite, ground_truth : Composite, transformed_size = 1024, width = 1068, height = 847):
    new_pred = copy(pred)
    new_ground_truth = copy(ground_truth)
    new_pred.bboxes = inverse_transform_box(width, height, transformed_size, pred.bboxes)
    new_ground_truth.bboxes = inverse_transform_box(width, height, transformed_size, ground_truth.bboxes)
    return new_pred, new_ground_truth

def inverse_transform_box(width, height, t_size, bboxes : List[BBox]):
    scale = t_size / width
    inv_scale = 1 / scale
    pad_size = (t_size - height*scale) / 2
    boxes_out = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox.xyxy
        x1_t = x1 * inv_scale
        x2_t = x2 * inv_scale
        y1_t = y1 * inv_scale - pad_size
        y2_t = y2 * inv_scale - pad_size
        bbox_transformed = BBox.from_xyxy(x1_t, y1_t, x2_t, y2_t)
        boxes_out.append(bbox_transformed)
    return boxes_out

def print_boxes(img, preds : Composite, ground_truth : Composite):
    fig, ax = plt.subplots(figsize=(14,10))
    ax.imshow(img)
    fig.set_dpi(300)
    boxes = preds.bboxes
    scores = preds.scores
    for box, score in zip(boxes, scores):
        x,y,w,h = box.xywh
        rect= patches.Rectangle((x,y), w, h, linewidth=1, edgecolor='r', facecolor = 'none')
        ax.add_patch(rect)

    gt_boxes = ground_truth.bboxes
    for box in gt_boxes:
        x,y,w,h = box.xywh
        rect= patches.Rectangle((x,y), w, h, linewidth=1, edgecolor='b', facecolor = 'none')
        ax.add_patch(rect)

    # for box in gt['detection']['bboxes']:
    #     x,y,w,h = box.xywh
    #     rect= patches.Rectangle((x,y), w, h, linewidth=1, edgecolor='b', facecolor = 'none')
    #     ax.add_patch(rect)
    plt.axis('off')
    fig.show()

def generate_detections_json(preds, stage = 'valid', preds2=None, stage2='train'):
    data = {}
    for pred in preds:
        img_id = pred.common.record_id
        p, _ = inverse_transform_preds(pred.detection, pred.ground_truth.detection)
        # labels = p.labels
        scores = p.scores.tolist()
        labels = [0 for i in range(len(scores))]
        boxes_list = []
        for box in p.bboxes:
            x1, y1, x2, y2 = box.xyxy
            boxes_list.append([x1, y1, x2, y2])
        data[img_id] = {'bboxes' : boxes_list, 'labels' : labels, 'scores' : scores, 'stage' :stage}
    if preds2 != None:
        for pred in preds2:
            img_id = pred.common.record_id
            p, _ = inverse_transform_preds(pred.detection, pred.ground_truth.detection)
            # labels = p.labels
            scores = p.scores.tolist()
            labels = [0 for i in range(len(scores))]
            boxes_list = []
            for box in p.bboxes:
                x1, y1, x2, y2 = box.xyxy
                boxes_list.append([x1, y1, x2, y2])
            data[img_id] = {'bboxes' : boxes_list, 'labels' : labels, 'scores' : scores, 'stage' :stage2}
    return data

