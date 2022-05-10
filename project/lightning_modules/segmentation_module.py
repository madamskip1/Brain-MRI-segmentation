from typing import Any, Optional, Union, List

import hydra
import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT

from DiceLoss import DiceLoss


class SegmentationModule(pl.LightningModule):
    def __init__(self, model, **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        #self.optimizer = optim
        self.loss_function = DiceLoss()

    def forward(self, x, *args, **kwargs) -> Any:
        return self.model(x)

    def training_step(self, batch, batch_idx, *args, **kwargs) -> STEP_OUTPUT:
        outputs, y = self.calc_outputs(batch)
        loss = self.loss_function(outputs, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        outputs, y = self.calc_outputs(batch)
        loss = self.loss_function(outputs, y)
        self.log("validation_loss", loss)
        accuracy = SegmentationModule.calc_acc(outputs, y)
        self.log("validation_accuracy", accuracy)
        return {"validation_loss": loss, "validation_accuracy": accuracy}

    def test_step(self, batch, batch_idx, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        outputs, y = self.calc_outputs(batch)
        loss = self.loss_function(outputs, y)
        self.log("test_loss", loss)
        accuracy = SegmentationModule.calc_acc(outputs, y)
        self.log("test_accuracy", accuracy)
        return {"test_loss": loss, "test_accuracy": accuracy}

    def validation_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        self.log("average_validation_loss", torch.stack([o["validation_loss"] for o in outputs]).mean())
        pass  # avg_loss = torch.stack([o["val_loss"] for o in outputs]).mean()

    def configure_optimizers(self):
        self.learning_rate = 1e-3
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.learning_rate)  # TODO Sprwadzić czy Adam do segmentacji czy inny
        return optimizer
    
    # def configure_optimizers(self):
    #     return hydra.utils.instantiate(self.hparams.optim, params=self.parameters())

    @staticmethod
    def calc_acc(outputs, targets, smooth=1.0):
        __outputs = outputs.view(-1)
        _targets = targets.view(-1)

        intersection = (__outputs * _targets).sum()
        total = (__outputs + _targets).sum()
        union = total - intersection
        IoU = (intersection + smooth) / (union + smooth)
        return IoU

    def calc_outputs(self, batch):
        x, y = batch
        outputs = self(x)
        return outputs, y
