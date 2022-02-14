from pytorch_lightning import Callback
from tqdm import tqdm
import torch
from icevision.models.ross.efficientdet import convert_raw_predictions
from icevision.metrics import COCOMetric, COCOMetricType
from zmq import device
from icevision.all import *


class CalcualteTrainMAP(Callback):
    def __init__(self, interval = 20):
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
                train_inf_loader = models.ross.efficientdet.infer_dl(trainer.datamodule.train_ds_valid_tfms, batch_size = 1, shuffle=False)
                preds = models.ross.efficientdet.predict_from_dl(pl_module.model, train_inf_loader, keep_images=False, detection_threshold=0.0)

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


