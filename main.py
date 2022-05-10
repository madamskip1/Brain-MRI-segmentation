
from MRIImagesDataModule import MRIImagesDataModule

from pytorch_lightning.loggers import WandbLogger
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
import wandb

from project.image_prediction_logger import ImagePredictionLogger

@hydra.main(config_path="configs", config_name="defaults")
def main(config: DictConfig) -> None:
    

    # data_module = hydra.utils.instantiate(config.data)
    # data_module.setup()
    
    # input = torch.rand((1, 1, 572, 572))
    # loss = DiceLoss()
    # print(loss(input, torch.rand((1, 1, 572, 572))))
    # print(input.size())
    # output = unet(input)
    # print(output.size())

    model = hydra.utils.instantiate(config.network)

    data_module = MRIImagesDataModule()
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
    
