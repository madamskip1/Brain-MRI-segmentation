from unet import *

if __name__ == "__main__":
    unet = UNet(in_channels=1, out_channels=3, first_layer_out_channels=32)
    input = torch.rand((1, 1, 572, 572))
    print(input.size())
    output = unet(input)
    print(output.size())
    assert output.size() == (1, 3, 388, 388)
