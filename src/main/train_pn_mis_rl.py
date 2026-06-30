import hydra
from omegaconf import DictConfig

from src.main.train_common import run_pipeline


@hydra.main(
    version_base=None,
    config_path="../../configs/hydra",
    config_name="train_pn_mis_rl",
)
def main(cfg: DictConfig) -> None:
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
