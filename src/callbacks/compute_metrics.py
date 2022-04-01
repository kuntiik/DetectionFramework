from pytorch_lightning import Callback
from tqdm import tqdm
import torch
from icevision.models.ross.efficientdet import convert_raw_predictions
from icevision.metrics import COCOMetric, COCOMetricType
from zmq import device
from icevision.all import *
from icevision.models.ultralytics import yolov5
import torchvision.transforms as T


from pytorch_lightning import Trainer, Callback
from pytorch_lightning.loggers import WandbLogger, LoggerCollection

import wandb


def get_wandb_logger(trainer: Trainer) -> WandbLogger:
    """Safely get Weights&Biases logger from Trainer."""

    if trainer.fast_dev_run:
        raise Exception(
            "Cannot use wandb callbacks since pytorch lightning disables loggers in `fast_dev_run=true` mode."
        )

    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger

    if isinstance(trainer.logger, LoggerCollection):
        for logger in trainer.logger:
            if isinstance(logger, WandbLogger):
                return logger

    raise Exception(
        "You are using wandb related callback, but WandbLogger was not found for some reason..."
    )


class CalcualteTrainMAP(Callback):
    def __init__(self, interval=20):
        self.interval = interval
        self.index = 1
        self.metric = COCOMetric(metric_type=COCOMetricType.bbox)

    def on_train_epoch_end(self, trainer, pl_module):
        if self.index % self.interval == 0:
            # else:
            logger = trainer.logger
            experiment = logger.experiment
            # for index, batch in tqdm(enumerate(trainer.train_dataloaders[0])):
            # for index, batch in tqdm(enumerate(trainer.test_loader)):
            with torch.no_grad():
                # (xb, yb), records = batch
                # xb = xb.to(device=pl_module.device)
                pl_module.model.eval()
                train_inf_loader = models.ross.efficientdet.infer_dl(
                    trainer.datamodule.train_ds_valid_tfms, batch_size=1, shuffle=False
                )
                preds = models.ross.efficientdet.predict_from_dl(
                    pl_module.model, train_inf_loader, keep_images=False, detection_threshold=0.0
                )

                pl_module.model.train()
                # for key, value in yb.items():
                #     value = value.to(device=pl_module.device)
                # yb = yb.to(device=pl_module.device)
                # records = records.to(device=pl_module.device)
                # yb = {k : v.to(pl_module.device) for k,v in yb.items()}

                # raw_preds = pl_module(xb, yb)
                # preds = convert_raw_predictions((xb,yb), raw_preds['detections'], records, detection_threshold=0.0)
                self.metric.accumulate(preds=preds)
            metric_logs = self.metric.finalize()
            for k, v in metric_logs.items():
                self.log(f"train/{k}", v)

        self.index += 1


def get_batch_images_to_log(trainer, pl_module, batch):

    mean = (0.3669, 0.3669, 0.3669)
    std = (0.2768, 0.2768, 0.2768)
    div_m = [-mean[0] / std[0] for _ in range(3)]
    div_s = [1.0 / std[0] for _ in range(3)]
    # inv_tfms = T.Normalize(-mean / std, 1.0 / std)
    inv_tfms = T.Normalize(div_m, div_s)

    (xb, yb), records = batch
    xb = xb.to("cuda")
    with torch.no_grad():
        pl_module.model.eval()
        inference_out, training_out = pl_module.model(xb)
        preds = yolov5.convert_raw_predictions(
            batch=xb,
            raw_preds=inference_out,
            records=records,
            detection_threshold=0.001,
            nms_iou_threshold=0.6,
        )

    images_to_log = []
    # for index, (boxes, img) in enumerate(zip(box_preds, val_imgs)):
    for pred, img in zip(preds, xb):
        bbs = pred.detection.bboxes
        # labels = preds.detection.boxes["labels"]
        scores = pred.detection.scores
        labels = [0 for i in range(len(scores))]
        positions = []
        # for box, label in zip(bbs, labels):
        for i in range(len(labels)):
            box = bbs[i]
            label = labels[i]
            score = float(scores[i])
            x1, y1, x2, y2 = box.xyxy
            box_pos = {
                # "position": {"minX": x1, "minY": y1, "maxX": x2, "maxY": y2,},
                "position": {"minX": int(x1), "minY": int(y1), "maxX": int(x2), "maxY": int(y2)},
                "class_id": int(label),
                "scores": {"confidence": float(score)},
                "box_caption": str(int(label)) + ":" + str(round(float(score), 3)),
                "domain": "pixel",
            }
            positions.append(box_pos)

        bbs_gt = pred.ground_truth.detection.bboxes
        labels_gt = [0 for i in range(len(bbs_gt))]
        gt_positions = []
        for box, label in zip(bbs_gt, labels_gt):
            x1, y1, x2, y2 = box.xyxy
            box_pos = {
                "position": {"minX": int(x1), "minY": int(y1), "maxX": int(x2), "maxY": int(y2)},
                "class_id": int(label),
                "domain": "pixel",
            }
            gt_positions.append(box_pos)

        boxes_data = {
            "predictions": {"box_data": positions},
            "ground_truth": {"box_data": gt_positions},
        }
        pil_img = inv_tfms(img)
        tp = T.ToPILImage()
        pl_img = tp(inv_tfms(img))
        wandb_img = wandb.Image(pl_img, boxes=boxes_data)
        images_to_log.append(wandb_img)
    return images_to_log


class LogImagePredictionsDetection(Callback):
    def __init__(self):
        self.ready = True

    def on_sanity_check_start(self, trainer, pl_module) -> None:
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        self.ready = True

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.ready:

            logger = get_wandb_logger(trainer=trainer)
            experiment = logger.experiment
            # val_samples = next(iter(trainer.datamodule.val_dataloader()))
            val_samples = next(iter(trainer.val_dataloaders[0]))
            images_to_log = get_batch_images_to_log(trainer, pl_module, val_samples)
            experiment.log({"val/image_sample": images_to_log})


class LogImagePredictionsDetectionFull(Callback):
    def __init__(self, interval=10):
        self.ready = True
        self.interval = interval
        self.index = 0

    def on_sanity_check_start(self, trainer, pl_module) -> None:
        self.ready = False

    def on_sanity_check_end(self, trainer, pl_module):
        self.ready = True

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.ready:
            self.index += 1
            if self.index % self.interval == 0:

                logger = get_wandb_logger(trainer=trainer)
                experiment = logger.experiment
                # val_samples = next(iter(trainer.datamodule.val_dataloader()))
                for index, batch in enumerate(trainer.val_dataloaders[0]):
                    if index == 8:
                        break
                    images_to_log = get_batch_images_to_log(trainer, pl_module, batch)
                    experiment.log({"val/full_images" + str(index): images_to_log})

