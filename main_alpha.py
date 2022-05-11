from pytorch_lightning.loggers import WandbLogger
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
import wandb

from project.image_prediction_logger import ImagePredictionLogger

@hydra.main(config_path="configs", config_name="defaults")
def main(config: DictConfig) -> None:
    pl.seed_everything(1234)
    # Instantiate all modules specified in the configs
    model = hydra.utils.instantiate(
        config.model,  # Object to instantiate
        # Overwrite arguments at runtime that depends on other modules
        input_dim=config.data.input_dim,
        output_dim=config.data.output_dim,
        # Don't instantiate optimizer submodules with hydra, let `configure_optimizers()` do it
        _recursive_=False,
    )

    data_module = hydra.utils.instantiate(config.data)
    data_module.setup()

    val_samples = next(iter(data_module.val_dataloader()))
    callbacks = [
        hydra.utils.instantiate(config.callbacks.model_checkpoint),
        hydra.utils.instantiate(config.callbacks.early_stopping),
        ImagePredictionLogger(val_samples)
    ]
    wandb_logger = WandbLogger(project="test-project")
    trainer = pl.Trainer(
        **OmegaConf.to_container(config.trainer),
        logger=wandb_logger,
        callbacks=callbacks,
    )

    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)
    wandb.finish()


if __name__ == '__main__':
    main()