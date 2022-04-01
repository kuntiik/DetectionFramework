import hydra
import os
from matplotlib import transforms
import torch
from typing import List
from omegaconf import DictConfig
from pytorch_lightning import (
    Callback,
    LightningDataModule,
    LightningModule,
    Trainer,
    seed_everything,
)
from pytorch_lightning.loggers import LightningLoggerBase
from icevision.all import models
import icevision
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
import albumentations as A


from src.utils import utils

log = utils.get_logger(__name__)


def train(config: DictConfig):

    if config.get("seed"):
        seed_everything(config.seed, workers=True)

    # convert relative path to absolute (hydra requires absolute path)
    ckpt_path = config.module.get("ckpt_path")
    if ckpt_path and not os.path.isabs(ckpt_path):
        config.module.ckpt_path = os.path.join(hydra.utils.get_original_cwd(), ckpt_path)

    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                log.info(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    logger: List[LightningLoggerBase] = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log.info(f"Instantiating logger <{lg_conf._target_}>")
                logger.append(hydra.utils.instantiate(lg_conf))

    t = None
    if "transforms" in config:
        t = []
        for _, lg_conf in config.transforms.items():
            if lg_conf["apply"]:
                if "_target_" in lg_conf.t:
                    t.append(hydra.utils.instantiate(lg_conf.t))
                # log.info(f"Instantiate")

    # use hydra to instantiate the model, datamodule and trainer

    # YOLOv5 loggs model structure - suppress it
    with utils.DisableLogger():
        model: LightningModule = hydra.utils.instantiate(config.module.model)

    if config.module.get("pretrained"):
        checkpoint = torch.load(config.module.pretrained)
        model.load_state_dict(checkpoint["state_dict"])

    dm: LightningDataModule = hydra.utils.instantiate(config.datamodule, transforms=t)
    trainer: Trainer = hydra.utils.instantiate(
        config.trainer,
        logger=logger,
        callbacks=callbacks,
        # strategy="ddp"
        # plugins=[DDPPlugin(find_unused_parameters=False)],
    )

    utils.log_hyperparameters(
        config=config,
        model=model,
        datamodule=dm,
        trainer=trainer,
        callbacks=callbacks,
        logger=logger,
    )

    trainer.tune(model=model, datamodule=dm)
    log.info("Starting training")
    # trainer.fit(model=model, datamodule=dm)

    trainer.fit(model=model, datamodule=dm)

    for lg in logger:
        if isinstance(lg, pl.loggers.wandb.WandbLogger):
            import wandb

            wandb.finish()
    # optimized_metric = config.get("optimized_metric")
    # if optimized_metric and optimized_metric not in trainer.callback_metrics:
    #     raise Exception(
    #         "Metric for hyperparameter optimization not found! "
    #         "Make sure the `optimized_metric` in `hparams_search` config is correct!"
    #     )
    # score = trainer.callback_metrics.get(optimized_metric)
    # return score
