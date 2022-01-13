import hydra
from icevision.models.ultralytics.yolov5 import backbones
from omegaconf import DictConfig
from pytorch_lightning import (Callback, LightningDataModule, LightningModule, Trainer, seed_everything)
from pytorch_lightning.loggers import LightningLoggerBase
from icevision.all import models
from src.modules.YoloV5Module import YoloV5Module

from src.utils import utils

log = utils.get_logger(__name__)

from src.modules.EfficientDetModule import EfficientDetModule
from src.datamodules.DentalCaries import DentalCariesDataModule

def train(config : DictConfig):
    if config.get("seed"):
        seed_everything(config.seed, workers=True)
    
    # log.info(f"Instantiating module <{config.model._target_}>")
    # model = hydra.utils.instantiate(config.model)
    # model = EfficientDetModule()

    #model_type = models.ross.efficientdet
    # backbone = model_type.backbones.tf_lite0(pretrained=0)

    model_type = models.ultralytics.yolov5
    backbone = model_type.backbones.extra_large_p6(pretrained=True)

    img_size = 1024 
    #backbone = model_type.backbones.tf_d4(pretrained=1)
    det_model = model_type.model(backbone=backbone, num_classes = 2, img_size = img_size)
    #model = EfficientDetModule(det_model, 1e-4)
    model = YoloV5Module(det_model, 1e-5)


    callbacks = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))


    logger = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))
    
    trainer = hydra.utils.instantiate(config.trainer, logger=logger, callbacks=callbacks)

    # dm = MNISTDataModule(num_workers=4, pin_memory=False)
    # dm = hydra.utils.instantiate(config.datamodule)
    # dm = DentalCariesDataModule('/home/kuntik/carries_dataset', 512, model_type, batch_size=2, num_workers=4)
    dm = DentalCariesDataModule('/home.stud/kuntluka/dataset/carries_dataset', img_size, model_type, batch_size = 2, num_workers = 4)

    trainer.tune(model=model, datamodule=dm)
    log.info("Starting training")
    trainer.fit(model=model, datamodule=dm)
