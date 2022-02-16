<div align="center">

# Object Detection Framework
<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.7+-blue?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning 1.5+-792ee5?style=for-the-badge&logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: hydra" src="https://img.shields.io/badge/config-hydra 1.1-89b8cd?style=for-the-badge&labelColor=gray"></a>
<a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg?style=for-the-badge&labelColor=gray"></a>
</div>

## Project structure

```{bash}
.
├── configs                       <- Folder with all Hydra configuration files
│   ├── config.yaml               <- Main project configuration file. Other config files are merged by this one
│   ├── callbacks                 <- Callbacks config, choose which callbacks you want to include and their settings
│   ├── datamodule                <- Datamodule related config (path, batch-size,...)
│   ├── experiment                <- Experiment config
│   ├── hparams_search            <- Optuna config with parameter search-space.
│   ├── logger                    <- Logger configs
│   ├── module                    <- Configure models (learning rate, model related parameters, ...)
│   └── trainer                   <- Pytorch-Lightning trainer configuration
├── notebooks                     <- Jupyter notebooks (preprocessing and visualization)
├── optimize_optuna.py            <- Optuna multi-process optimization
├── optuna_single_process.py      <- Optuna single-process optimization
├── README.md   
├── run.py                        <- Run current configuration specified by Hydra configs
└── src                          
    ├── callbacks                 <- Pytorch lightning custom callbacks
    ├── datamodules               <- Pytorch lightning based datamodules and datasets
    ├── modules                   <- Pytorch lightning based modules - model and training loop implementation
    ├── train.py                  <- Here all source files are combined, based on the configuration
    └── utils                     
```

## To run:
- conda env create -f environment.yaml (heavy on dependencies)
- python run.py

# How to setup
- default setup in config.yaml
- swap the model by -module=efficient_det.yaml / -module=efficient_det.yaml
- set all parameters in the module config, except for batch_size

# Hyper-parameter search
This framework supports hyper-parameter search powered by Optuna. There are mutliple options how to approach this:
 ```{bash}
 pyhton run.py -m hprarams_search=<config from hparams_search folder>
 ```
 This is the fastest possible approach to setup, but Optuna will have limited capabilities. There will be no prunning available and search-space configuration is limited
 <br>
 ```{bash}
 python optimize_optuna.py
 ```
 Multi-process Optuna search. You need to modify content of this file (specify search-space and config overrides). This is discouraged approach, since there are situations, when single or more processes freeze. I am working on fixing this issue.
 
 <br>
 ```{bash}
 python optuna_single_process.py
 ```
 In this setting optuna launches single optimization process. You can run this file multiple times to get faster optimization results. The scaling should have near-linear impact on search time. If you run this on multiple nodes you need to provide database, that is accessible by all nodes
