import wandb
from pytorch_lightning.callbacks import Callback


class ImagePredictionLogger(Callback):
    def __init__(self, val_samples, num_samples=10):
        super().__init__()
        self.num_samples = num_samples
        self.val_imgs, self.val_labels = val_samples

    def on_validation_epoch_end(self, trainer, pl_module):
        # Bring the tensors to CPU
        val_imgs = self.val_imgs.to(device=pl_module.device)
        val_labels = self.val_labels.to(device=pl_module.device)
        # Get model prediction
        outputs = pl_module.predict(val_imgs)
        outputs = outputs.squeeze(1).to("cpu").numpy()

        val_labels = val_labels.squeeze(1).to("cpu").numpy()

        # Log the images as wandb Image
        trainer.logger.experiment.log({
            "examples": [wandb.Image(x, masks={"prediction": {"mask_data": pred}, "ground truth": {"mask_data": y}})
                         for x, pred, y in zip(val_imgs[:self.num_samples],
                                               outputs[:self.num_samples],
                                               val_labels[:self.num_samples])]
        })
