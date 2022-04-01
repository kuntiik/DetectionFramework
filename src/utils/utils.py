import logging
from pytorch_lightning.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf
import warnings
from typing import List, Sequence
import rich.syntax
import rich.tree
import pytorch_lightning as pl

# Code from https://github.com/ashleve/lightning-hydra-template/blob/fb7c206889255faeeed671bf71f6160bae74e90f/src/utils/__init__.py#L36


def get_logger(name=__name__) -> logging.Logger:
    logger = logging.getLogger(name)

    for level in ("debug", "info", "warning", "error", "exception", "fatal", "critical"):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


log = get_logger(__name__)


def extras(config: DictConfig) -> None:
    log = get_logger(__name__)

    # disable python warnings if <config.ignore_warnings=True>
    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # verify experiment name is set when running in experiment mode
    if config.get("experiment_mode") and not config.get("name"):
        log.info(
            "Running in experiment mode without the experiment name specified! "
            "Use `python run.py mode=exp name=experiment_name`"
        )
        log.info("Exiting...")
        exit()

    if config.trainer.get("fast_dev_run"):
        log.info("Forcing debugger friendly configuration! <config.trainer.fast_dev_run=True>")
        if config.trainer.get("gpus"):
            config.trainer.gpus = 0
        if config.datamodule.get("pin_memory"):
            config.datamodule.pin_memory = False
        if config.datamodule.get("num_workers"):
            config.datamodule.num_workers = 0


@rank_zero_only
def print_config(
    config: DictConfig,
    print_order: Sequence[str] = ("datamodule", "model", "callbacks", "logger", "trainer",),
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.
    Args:
        config (DictConfig): Configuration composed by Hydra.
        print_order (Sequence[str], optional): Determines in what order config components are printed.
        resolve (bool, optional): Whether to resmodelolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    quee = []

    for field in print_order:
        quee.append(field) if field in config else log.info(f"Field '{field}' not found in config")

    for field in config:
        if field not in quee:
            quee.append(field)

    for field in quee:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = config[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    with open("config_tree.log", "w") as file:
        rich.print(tree, file=file)


@rank_zero_only
def log_hyperparameters(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.loggers.LightningLoggerBase],
) -> None:
    """Controls which config parts are saved by Lightning loggers.
    Additionaly saves:
    - number of model parameters
    """

    hparams = {}

    # choose which parts of hydra config are saved by loggers
    hparams["trainer"] = config["trainer"]
    hparams["module"] = config["module"]
    hparams["datamodule"] = config["datamodule"]

    if "seed" in config:
        hparams["seed"] = config["seed"]
    # if "callbacks" in config:
    #     hparams["callbacks"] = config["callbacks"]

    # save number of model parameters
    # hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    # hparams["model/params/trainable"] = sum(
    #     p.numel() for p in model.parameters() if p.requires_grad
    # )
    # hparams["model/params/non_trainable"] = sum(
    #     p.numel() for p in model.parameters() if not p.requires_grad
    # )

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)


class DisableLogger:
    def __enter__(self):
        logging.disable(logging.CRITICAL)

    def __exit__(self, exit_type, exit_value, exit_traceback):
        logging.disable(logging.NOTSET)
