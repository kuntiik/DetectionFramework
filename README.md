<div align="center">

# Object Detection Framework
<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.7+-blue?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning 1.5+-792ee5?style=for-the-badge&logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: hydra" src="https://img.shields.io/badge/config-hydra 1.1-89b8cd?style=for-the-badge&labelColor=gray"></a>
<a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg?style=for-the-badge&labelColor=gray"></a>
</div>
## Intorduction 
This is an object detection framework, which was made for dental caries detection from X-ray images. Despite it's purpose it can be modified to serve as general object detection framework.
## How to install
This project rellies on many external dependencies. The main dependencies are as follows: Pytorch-Lightning, Hydra, IceVision, Optuna. You can install this project by following commands (conda required).
```{bash}
- conda env create -f environment.yaml
- canda activate detection
```
## How to run
You can run the project with default setting by `python run.py`

## How to change setting
Most of the settings can be changed from command line. You can override any configuration by appending `<settings to change>=<reqiured settings>` to the `python run command`. For example `python run.py module=yolov5`. You can also add configuration, that is not specified in .yaml files by `+<setting to change>=<required settings>`, eg. `+trainer.fast_dev_run=True`. You have to look into documentation of the particular method to get list of all parameters or you can install Hydra's autocompletion by running the following command: `python run.py --hydra-help` and than following the manual, that will pop-up. 

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


# How to setup
- default setup in config.yaml
- swap the model by -module=efficient_det.yaml / -module=efficient_det.yaml
- set all parameters in the module config, except for batch_size

## Hyper-parameter search
This framework supports hyper-parameter search powered by Optuna. Optuna is a optimization toolbox, that uses methods shuch as Tree-structured Parzen Estimator to propose hyper-parameters to use in the next trial. The history of runs is kept in SQL database. You can optimization on multiple machines if you specify the path to you SQL storage. 
There are mutliple options how to use optuna by this framework.
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
 In this setting optuna launches single optimization process. You can run this file multiple times to get faster optimization results. The scaling should have near-linear impact on search time. If you run this on multiple nodes you need to provide database, that is accessible by all nodes.
