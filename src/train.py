import hydra
from omegaconf import DictConfig
from pytorch_lightning import (Callback, LightningDataModule, LightningModule, Trainer, seed_everything)
from pytorch_lightning.loggers import LightningLoggerBase
from icevision.all import models
import icevision

from src.utils import utils

log = utils.get_logger(__name__)


def train(config : DictConfig):

    if config.get("seed"):
        seed_everything(config.seed, workers=True)
    

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
    

    #use hydra to instantiate the model, datamodule and trainer
    model = hydra.utils.instantiate(config.module.model)
    dm = hydra.utils.instantiate(config.datamodule)
    trainer = hydra.utils.instantiate(config.trainer, logger=logger, callbacks=callbacks)

    trainer.tune(model=model, datamodule=dm)
    log.info("Starting training")
    trainer.fit(model=model, datamodule=dm)
