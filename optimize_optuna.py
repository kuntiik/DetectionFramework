from distutils.command.config import config
from turtle import st
import hydra
import optuna
from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback
from sqlalchemy import all_
import torch
import pytorch_lightning
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning import seed_everything

import albumentations as A
from albumentations.pytorch import ToTensorV2

import multiprocessing
from contextlib import contextmanager


class GpuQueue:
    def __init__(self) -> None:
        self.queue = multiprocessing.Manager().Queue()
        all_idxs = [1, 2, 3, 4]
        for idx in all_idxs:
            self.queue.put(idx)

    @contextmanager
    def one_gpu_per_process(self):
        current_idx = self.queue.get()
        yield current_idx
        self.queue.put(current_idx)


class Objective:
    def __init__(self, gpu_queue: GpuQueue, cfg) -> None:
        self.gpu_queue = gpu_queue
        self.cfg = cfg

    def __call__(self, trial: optuna.trial.Trial):
        cfg = self.cfg
        learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-2, log=True)
        optimizer = trial.suggest_categorical("optimizer", ["adam", "sgd"])
        weight_decay = trial.suggest_float("weight_decay", 1e-10, 1e-2, log=True)
        scheduler_factor = trial.suggest_float("scheduler_factor", 0.1, 0.6)
        scheduler_patience = trial.suggest_int("scheduler_patience", 2, 10)

        cfg.module.model.learning_rate = learning_rate
        cfg.module.model.optimizer = optimizer
        cfg.module.model.weight_decay = weight_decay
        cfg.module.model.scheduler_patience = scheduler_patience
        cfg.module.model.scheduler_factor = scheduler_factor

        if cfg.get("seed"):
            seed_everything(cfg.seed, workers=True)

        callbacks = []
        if "callbacks" in cfg:
            for _, cb_conf in cfg.callbacks.items():
                if "_target_" in cb_conf:
                    callbacks.append(hydra.utils.instantiate(cb_conf))

        logger = []
        if "logger" in cfg:
            for _, lg_conf in cfg.logger.items():
                if "_target_" in lg_conf:
                    logger.append(hydra.utils.instantiate(lg_conf))

        dm = hydra.utils.instantiate(cfg.datamodule)
        model = hydra.utils.instantiate(cfg.module.model)

        if cfg.module.pretrained != None:
            checkpoint = torch.load(cfg.module.pretrained)
            model.load_state_dict(checkpoint["state_dict"])

        callbacks.append(PyTorchLightningPruningCallback(trial, monitor="val/loss"))

        with self.gpu_queue.one_gpu_per_process() as gpu_i:
            trainer = hydra.utils.instantiate(
                cfg.trainer,
                logger=logger,
                callbacks=callbacks,
                num_sanity_val_steps=0,
                enable_checkpointing=False,
                gpus=[gpu_i],
            )
            trainer.fit(model=model, datamodule=dm)
        return trainer.callback_metrics["val/loss"].item()


def objective(trial: optuna.trial.Trial, cfg):

    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-2, log=True)
    optimizer = trial.suggest_categorical("optimizer", ["adam", "sgd"])
    weight_decay = trial.suggest_float("weight_decay", 1e-10, 1e-2, log=True)
    scheduler_factor = trial.suggest_float("scheduler_factor", 0.1, 0.6)
    scheduler_patience = trial.suggest_int("scheduler_patience", 2, 10)

    cfg.module.model.learning_rate = learning_rate
    cfg.module.model.optimizer = optimizer
    cfg.module.model.weight_decay = weight_decay
    cfg.module.model.scheduler_patience = scheduler_patience
    cfg.module.model.scheduler_factor = scheduler_factor

    if cfg.get("seed"):
        seed_everything(cfg.seed, workers=True)

    callbacks = []
    if "callbacks" in cfg:
        for _, cb_conf in cfg.callbacks.items():
            if "_target_" in cb_conf:
                callbacks.append(hydra.utils.instantiate(cb_conf))

    logger = []
    if "logger" in cfg:
        for _, lg_conf in cfg.logger.items():
            if "_target_" in lg_conf:
                logger.append(hydra.utils.instantiate(lg_conf))

    dm = hydra.utils.instantiate(cfg.datamodule)
    model = hydra.utils.instantiate(cfg.module.model)

    if cfg.module.pretrained != None:
        checkpoint = torch.load(cfg.module.pretrained)
        model.load_state_dict(checkpoint["state_dict"])

    callbacks.append(PyTorchLightningPruningCallback(trial, monitor="val/loss"))
    # trainer = hydra.utils.instantiate(config.trainer, logger=logger, callbacks=callbacks, plugins=[DDPPlugin(find_unused_parameters=False)],
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        logger=logger,
        callbacks=callbacks,
        num_sanity_val_steps=0,
        enable_checkpointing=False,
    )
    trainer.fit(model=model, datamodule=dm)
    return trainer.callback_metrics["val/loss"].item()


def main():
    with hydra.initialize(config_path="configs"):
        cfg = hydra.compose(config_name="config")
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(
        direction="minimize",
        pruner=pruner,
        # storage="sqlite:///optim_optuna.db",
        storage="mysql://root@34.116.169.28/example",
        study_name="effnet_optim",
    )
    # study.optimize(lambda trial: objective(trial, cfg), n_trials=50, n_jobs=4)
    study.optimize(Objective(GpuQueue(), cfg), n_trials=50, n_jobs=4)
    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial: ")
    trial = study.best_trial

    print(" Value: {}".format(trial.value))
    print(" Params: ")
    for key, value in trial.params.items():
        print("  {}: {}".format(key, value))


if __name__ == "__main__":
    main()
