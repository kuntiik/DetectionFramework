import hydra
from omegaconf import DictConfig
from src.utils import utils
from omegaconf import OmegaConf

@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config : DictConfig):
    from src.train import train
    print(OmegaConf.to_yaml(config))

    # utils.extras(config)

    return train(config)


if __name__ == "__main__":
    main()