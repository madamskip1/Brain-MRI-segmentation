from typing import Any, Optional, Union, List

import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT

from project.utils.dice_loss import DiceLoss
from project.utils.iou_accuracy import calculate_accuracy


class SegmentationModule(pl.LightningModule):
    def __init__(self, model, loss_function, learning_rate, mask_true_threshold, **kwargs):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.learning_rate = learning_rate
        self.loss_function = loss_function
        self.mask_true_threshold = mask_true_threshold

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
        accuracy = calculate_accuracy(outputs, y, self.mask_true_threshold)
        self.log("validation_accuracy", accuracy)
        return {"validation_loss": loss, "validation_accuracy": accuracy}

    def test_step(self, batch, batch_idx, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        outputs, y = self.calc_outputs(batch)
        loss = self.loss_function(outputs, y)
        self.log("test_loss", loss)
        accuracy = calculate_accuracy(outputs, y, self.mask_true_threshold)
        self.log("test_accuracy", accuracy)
        return {"test_loss": loss, "test_accuracy": accuracy}

    def validation_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        self.log("average_validation_loss", torch.stack([o["validation_loss"] for o in outputs]).mean())

    def configure_optimizers(self):
        print(self.parameters())
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.learning_rate)  # TODO Sprwadzić czy Adam do segmentacji czy inny
        return optimizer

    def predict(self, inputs):
        return self.model.predict(inputs)

    def calc_outputs(self, batch):
        x, y = batch
        outputs = self(x)
        return outputs, y
