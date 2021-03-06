from icevision.imports import *
from icevision.metrics import *
from icevision.models.ultralytics import yolov5
from yolov5.utils.loss import ComputeLoss
import pytorch_lightning as pl
from torchmetrics.detection.map import MeanAveragePrecision as MAP


def ice_preds_to_dict(preds_records):
    preds = []
    target = []
    for record in preds_records:
        preds.append(
            dict(
                boxes=torch.Tensor([[*box.xyxy] for box in record.pred.detection.bboxes]),
                scores=torch.Tensor(record.pred.detection.scores),
                labels=torch.Tensor(record.pred.detection.label_ids),
            )
        )

        target.append(
            dict(
                boxes=torch.Tensor([[*box.xyxy] for box in record.ground_truth.detection.bboxes]),
                labels=torch.Tensor(record.ground_truth.detection.label_ids),
            )
        )
    return preds, target


class YoloV5Module(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        learning_rate,
        optimizer="adam",
        scheduler_patience=10,
        scheduler_factor=0.2,
        weight_decay=1e-6,
    ):
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore=["model"])

        self.metrics = [COCOMetric(metric_type=COCOMetricType.bbox)]
        self.train_map = [COCOMetric(metric_type=COCOMetricType.bbox)]

        self.compute_loss = ComputeLoss(model)
        self.metrics_keys_to_log_to_prog_bar = [("AP (IoU=0.50) area=all", "val/Pascal_VOC")]

        self.MAP = MAP()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
        print("Model Created!")

    def configure_optimizers(self):
        if self.hparams.optimizer == "adam":
            optimizer = Adam(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
            )
        else:
            optimizer = SGD(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
            )
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                factor=self.hparams.scheduler_factor,
                patience=self.hparams.scheduler_patience,
            ),
            "monitor": "val/loss",
            "interval": "epoch",
            "name": "lr",
        }
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        (xb, yb), _ = batch
        preds = self(xb)
        loss = self.compute_loss(preds, yb)[0]
        self.log("train/loss", loss)

        return loss

    def predict_batch(self, batch):
        (xb, yb), records = batch
        with torch.no_grad():
            inference_out, training_out = self(xb)
            preds = yolov5.convert_raw_predictions(
                batch=xb,
                raw_preds=inference_out,
                records=records,
                detection_threshold=0.001,
                nms_iou_threshold=0.6,
            )
            return preds

    def validation_step(self, batch, batch_idx):
        (xb, yb), records = batch

        with torch.no_grad():
            inference_out, training_out = self(xb)
            preds = yolov5.convert_raw_predictions(
                batch=xb,
                raw_preds=inference_out,
                records=records,
                detection_threshold=0.001,
                nms_iou_threshold=0.6,
            )
            loss = self.compute_loss(training_out, yb)[0]

        self.accumulate_metrics(preds)
        self.log("val/loss", loss)

    def predict_step(self, batch, batch_idx):
        (xb, yb), records = batch
        with torch.no_grad():
            inference_out, training_out = self(xb)
            preds = yolov5.convert_raw_predictions(
                batch=xb,
                raw_preds=inference_out,
                records=records,
                detection_threshold=0.001,
                nms_iou_threshold=0.6,
            )
        return preds

    # def training_epoch_end(self, outs) -> None:

    def validation_epoch_end(self, outs):
        self.finalize_metrics()

    def accumulate_metrics(self, preds):
        for metric in self.metrics:
            metric.accumulate(preds=preds)

    def finalize_metrics(self) -> None:
        for metric in self.metrics:
            metric_logs = metric.finalize()
            for k, v in metric_logs.items():
                for entry in self.metrics_keys_to_log_to_prog_bar:
                    if entry[0] == k:
                        self.log(entry[1], v, prog_bar=True)
                        self.log(f"{metric.name}/{k}", round(v, 8))
                    else:
                        self.log(f"val/{metric.name}/{k}", round(v, 8))

    # def load_from_checkpoint(self, path):
    #     checkpoint = torch.load(path)
    #     self.model.load_state_dict(checkpoint["state_dict"])
