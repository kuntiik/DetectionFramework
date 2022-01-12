from icevision.all import *
import pytorch_lightning as pl
from icevision.models.ross import efficientdet


class EfficientDetModule(pl.LightningModule):
    def __init__(self, model : nn.Module, learning_rate):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.metrics = [COCOMetric(metric_type=COCOMetricType.bbox) ]
        # self.metrics_keys_to_log_to_prog_bar = [ ("AP (IoU=0.50:0.95) area=all", "COCOMetric") ]
        self.metrics_keys_to_log_to_prog_bar = [ ("AP (IoU=0.50) area=all", "COCOMetric") ]
    
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.learning_rate)
    
    def training_step(self, batch, batch_idx):
        (xb, yb), records = batch
        preds = self(xb, yb)
        loss = efficientdet.loss_fn(preds, yb)
        for k, v in preds.items():
            self.log(f"train/{k}", v)
        return loss
    
    def validation_step(self, batch, batch_idx) :
        (xb, yb), records = batch
        with torch.no_grad():
            raw_preds = self(xb, yb)
            preds = efficientdet.convert_raw_predictions((xb,yb), raw_preds['detections'], records, detection_threshold=0.0)
            loss = efficientdet.loss_fn(raw_preds, yb)
        
        for k, v in raw_preds.items():
            if "loss" in k:
                self.log(f"valid/{k}", v)
        self.accumulate_metrics(preds)
    
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
                        self.log(f"{metric.name}/{k}", v)
                        print(k, v, "\n")
                    else:
                        self.log(f"{metric.name}/{k}", v)