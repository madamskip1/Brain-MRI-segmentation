import torch

from unet import UNet
from data_module import MRIImagesDataModule

if __name__ == "__main__":
    unet = UNet(in_channels=1, out_channels=3, first_layer_out_channels=32)
    input = torch.rand((1, 1, 572, 572))
    print(input.size())
    output = unet(input)
    print(output.size())
    assert output.size() == (1, 3, 388, 388)

    data_module = MRIImagesDataModule()
    data_module.prepare_data()
    #data_module.setup()
   # train = data_module.train_dataloader()
   # print(len(train.dataset))