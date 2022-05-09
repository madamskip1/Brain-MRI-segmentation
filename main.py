import pytorch_lightning as pl
import torch

from DiceLoss import DiceLoss
from MRIImagesDataModule import MRIImagesDataModule
from SegmentationModule import SegmentationModule
from unet import UNet

if __name__ == "__main__":
    unet = UNet(in_channels=1, out_channels=1, first_layer_out_channels=32)
    input = torch.rand((1, 1, 572, 572))
    loss = DiceLoss()
    print(loss(input, torch.rand((1, 1, 572, 572))))
    print(input.size())
    output = unet(input)
    print(output.size())

    model = SegmentationModule(unet)

    data_module = MRIImagesDataModule()
    data_module.prepare_data()
    data_module.setup()
    train = data_module.train_dataloader()
    print(len(train.dataset))

    trainer = pl.Trainer(max_epochs=1, gpus=1)
    trainer.fit(model=model, datamodule=data_module)
    trainer.test(model=model, datamodule=data_module)
