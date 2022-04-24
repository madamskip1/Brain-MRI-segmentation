from unet import *

if __name__ == "__main__":
    unet = UNet()
    input = torch.rand((1, 1, 572, 572))
    print(input.size())
    output = unet(input)
    print(output.size())
    assert output.size() == (1, 1, 388, 388)
