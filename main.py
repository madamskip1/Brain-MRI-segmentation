from pytorch_lightning.loggers import WandbLogger
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
import wandb

from project.utils.image_prediction_logger import ImagePredictionLogger
from project.models.segmentation_module import SegmentationModule

@hydra.main(config_path="configs", config_name="defaults")
def main(config: DictConfig) -> None:

    if config.checkpoint_path:
        unet = hydra.utils.instantiate(config.segmentation_module_model)
        loss = hydra.utils.instantiate(config.loss)
        model = SegmentationModule.load_from_checkpoint(checkpoint_path=config.checkpoint_path, model=unet, loss_function=loss)
    else:
        model = hydra.utils.instantiate(config.model)

    data_module = hydra.utils.instantiate(config.data)
    data_module.prepare_data()
    data_module.setup()

    val_samples = next(iter(data_module.val_dataloader()))
    callbacks = [
        hydra.utils.instantiate(config.callbacks.model_checkpoint),
        hydra.utils.instantiate(config.callbacks.early_stopping),
        ImagePredictionLogger(val_samples)
    ]
    wandb_logger = WandbLogger(project="test-project-mri")
    
    trainer = pl.Trainer(
        **OmegaConf.to_container(config.trainer),
        logger=wandb_logger,
        callbacks=callbacks,
    )

    trainer.fit(model=model, datamodule=data_module)
    trainer.test(model=model, datamodule=data_module)
    wandb.finish()

if __name__ == "__main__":
    main()
    
