import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import CenterCrop


class UNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv_block_l_1 = UNet.__conv_block(1, 64, 3)
        self.conv_block_l_2 = UNet.__conv_block(64, 128, 3)
        self.conv_block_l_3 = UNet.__conv_block(128, 256, 3)
        self.conv_block_l_4 = UNet.__conv_block(256, 512, 3)

        self.conv_block_mid = UNet.__conv_block(512, 1024, 3)
        self.up_conv_mid = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)

        self.conv_block_r_4 = UNet.__up_conv_block(1024, 512, 3)
        self.conv_block_r_3 = UNet.__up_conv_block(512, 256, 3)
        self.conv_block_r_2 = UNet.__up_conv_block(256, 128, 3)
        self.conv_block_r_1 = UNet.__conv_block(128, 64, 3)

        self.crop_4 = CenterCrop((56, 56))
        self.crop_3 = CenterCrop((104, 104))
        self.crop_2 = CenterCrop((200, 200))
        self.crop_1 = CenterCrop((392, 392))

        self.conv_out = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # Contracting path
        x = self.conv_block_l_1(x)
        assert x.size() == (1, 64, 568, 568)
        encoded1 = self.crop_1(x)
        assert encoded1.size() == (1, 64, 392, 392)
        x = self.maxpool(x)
        assert x.size() == (1, 64, 284, 284)

        x = self.conv_block_l_2(x)
        assert x.size() == (1, 128, 280, 280)
        encoded2 = self.crop_2(x)
        x = self.maxpool(x)
        assert x.size() == (1, 128, 140, 140)

        x = self.conv_block_l_3(x)
        assert x.size() == (1, 256, 136, 136)
        encoded3 = self.crop_3(x)
        x = self.maxpool(x)
        assert x.size() == (1, 256, 68, 68)

        x = self.conv_block_l_4(x)
        assert x.size() == (1, 512, 64, 64)
        encoded4 = self.crop_4(x)
        x = self.maxpool(x)
        assert x.size() == (1, 512, 32, 32)

        # Middle
        x = self.conv_block_mid(x)
        assert x.size() == (1, 1024, 28, 28)
        x = self.up_conv_mid(x)
        assert x.size() == (1, 512, 56, 56)

        # Expansive path
        x = torch.cat((x, encoded4), dim=1)
        x = self.conv_block_r_4(x)
        assert x.size() == (1, 256, 104, 104)

        x = torch.cat((x, encoded3), dim=1)
        x = self.conv_block_r_3(x)
        assert x.size() == (1, 128, 200, 200)

        x = torch.cat((x, encoded2), dim=1)
        x = self.conv_block_r_2(x)
        assert x.size() == (1, 64, 392, 392)

        x = torch.cat((x, encoded1), dim=1)
        x = self.conv_block_r_1(x)
        assert x.size() == (1, 64, 388, 388)
        x = self.conv_out(x)
        assert x.size() == (1, 1, 388, 388)

        return F.sigmoid(x)

    @staticmethod
    def __conv_block(in_channels, out_channels, kernel_size=3):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size),
            nn.ReLU()
        )

    @staticmethod
    def __up_conv_block(in_channels, out_channels, kernel_size=3):
        conv_block = UNet.__conv_block(in_channels, out_channels, kernel_size)
        conv_block.append(
            nn.ConvTranspose2d(in_channels=out_channels, out_channels=int(out_channels / 2), kernel_size=2, stride=2))

        return conv_block
