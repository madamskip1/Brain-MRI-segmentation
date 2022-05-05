import torch
import torch.nn as nn
import torch.nn.functional as f
from torchvision.transforms import CenterCrop


class UNet(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, first_layer_out_channels=64):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.first_layer_out_channels = first_layer_out_channels

        layer_out_channels = first_layer_out_channels

        self.conv_block_l_1 = UNet.__conv_block(self.in_channels, layer_out_channels, 3)
        self.conv_block_l_2 = UNet.__conv_block(layer_out_channels, layer_out_channels * 2, 3)
        self.conv_block_l_3 = UNet.__conv_block(layer_out_channels * 2, layer_out_channels * 4, 3)
        self.conv_block_l_4 = UNet.__conv_block(layer_out_channels * 4, layer_out_channels * 8, 3)

        contracting_block_out_channels = layer_out_channels * 8

        self.conv_block_mid = UNet.__conv_block(contracting_block_out_channels, contracting_block_out_channels * 2, 3)
        self.up_conv_mid = nn.ConvTranspose2d(in_channels=(contracting_block_out_channels * 2),
                                              out_channels=contracting_block_out_channels, kernel_size=2, stride=2)

        mid_block_out_channels = contracting_block_out_channels

        self.conv_block_r_4 = UNet.__up_conv_block(mid_block_out_channels * 2, mid_block_out_channels, 3)
        self.conv_block_r_3 = UNet.__up_conv_block(mid_block_out_channels, int(mid_block_out_channels / 2), 3)
        self.conv_block_r_2 = UNet.__up_conv_block(int(mid_block_out_channels / 2), int(mid_block_out_channels / 4), 3)
        self.conv_block_r_1 = UNet.__conv_block(int(mid_block_out_channels / 4), int(mid_block_out_channels / 8), 3)

        expensive_block_out_channels = int(mid_block_out_channels / 8)

        self.conv_out = nn.Conv2d(in_channels=expensive_block_out_channels, out_channels=self.out_channels,
                                  kernel_size=1)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        _, _, height, width = x.shape
        # Contracting path
        x = self.conv_block_l_1(x)
        encoded1 = torch.clone(x)
        x = self.maxpool(x)

        x = self.conv_block_l_2(x)
        encoded2 = torch.clone(x)
        x = self.maxpool(x)

        x = self.conv_block_l_3(x)
        encoded3 = torch.clone(x)
        x = self.maxpool(x)

        x = self.conv_block_l_4(x)
        encoded4 = torch.clone(x)
        x = self.maxpool(x)

        # Middle
        x = self.conv_block_mid(x)
        x = self.up_conv_mid(x)

        # Expansive path
        encoded4 = UNet.__crop_encoded(encoded4, x)
        x = torch.cat((x, encoded4), dim=1)
        x = self.conv_block_r_4(x)

        encoded3 = UNet.__crop_encoded(encoded3, x)
        x = torch.cat((x, encoded3), dim=1)
        x = self.conv_block_r_3(x)

        encoded2 = UNet.__crop_encoded(encoded2, x)
        x = torch.cat((x, encoded2), dim=1)
        x = self.conv_block_r_2(x)

        encoded1 = UNet.__crop_encoded(encoded1, x)
        x = torch.cat((x, encoded1), dim=1)
        x = self.conv_block_r_1(x)
        x = self.conv_out(x)

        x = torch.sigmoid(x)
        x = nn.functional.interpolate(x, (height, width))
        return x

    @staticmethod
    def __conv_block(in_channels, out_channels, kernel_size=3):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    @staticmethod
    def __up_conv_block(in_channels, out_channels, kernel_size=3):
        conv_block = UNet.__conv_block(in_channels, out_channels, kernel_size)
        conv_block.append(
            nn.ConvTranspose2d(in_channels=out_channels, out_channels=int(out_channels / 2), kernel_size=2, stride=2))

        return conv_block

    @staticmethod
    def __crop_encoded(encoded, decoded):
        _, _, h, w = decoded.size()
        crop_func = CenterCrop((h, w))
        return crop_func(encoded)
