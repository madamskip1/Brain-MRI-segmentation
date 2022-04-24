from unet import *
import torch

if __name__ == "__main__":
        unet = UNet()
        input = torch.rand((1,1, 572, 572))
        print(input.size())
        output = unet(input)
        print(output.size())
        output_size = output.size()
        assert output_size[1] == 1
        assert output_size[2] == 388
        assert output_size[3] == 388