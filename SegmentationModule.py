from typing import Any, Optional, Union, List

import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT


class SegmentationModule(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.learning_rate = 1e-3

    def forward(self, x, *args, **kwargs) -> Any:
        return self.model(x)

    def training_step(self, batch, batch_idx, *args, **kwargs) -> STEP_OUTPUT:
        outputs, y = self.calc_outputs(batch)
        loss = SegmentationModule.calc_loss(outputs, y)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        outputs, y = self.calc_outputs(batch)
        loss = SegmentationModule.calc_loss(outputs, y)
        acc = SegmentationModule.calc_acc(outputs, y)
        return {"val_loss": loss, "val_acc": acc}

    def test_step(self, batch, batch_idx, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        outputs, y = self.calc_outputs(batch)
        loss = SegmentationModule.calc_loss(outputs, y)
        acc = SegmentationModule.calc_acc(outputs, y)
        return {"test_loss": loss, "test_acc": acc}

    def validation_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        avg_loss = torch.stack([o["val_loss"] for o in outputs]).mean()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.learning_rate)  # TODO Sprwadzić czy Adam do segmentacji czy inny
        return optimizer

    @staticmethod
    def calc_loss(outputs, y):
        return 0  # TODO dla segmentacji + przerobić na osobną klasę

    @staticmethod
    def calc_acc(outputs, y):
        return 0  # TODO dla segmentacji + przerobić na osobną klasę

    def calc_outputs(self, batch):
        x, y = batch
        outputs = self(x)
        return outputs, y
