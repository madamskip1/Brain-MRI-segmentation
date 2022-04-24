import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import CenterCrop

class UNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv_l_1_1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3)
        self.conv_l_1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)

        self.conv_l_2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.conv_l_2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3)

        self.conv_l_3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
        self.conv_l_3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3)

        self.conv_l_4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3)
        self.conv_l_4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3)

        self.conv_mid_1 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3)
        self.conv_mid_2 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3)

        self.up_conv_4 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.crop_4 = CenterCrop((56, 56))
        self.conv_r_4_1 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3)
        self.conv_r_4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3)

        self.up_conv_3 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.crop_3 = CenterCrop((104, 104))
        self.conv_r_3_1 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3)
        self.conv_r_3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3)

        self.up_conv_2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.crop_2 = CenterCrop((200, 200))
        self.conv_r_2_1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3)
        self.conv_r_2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3)

        self.up_conv_1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.crop_1 = CenterCrop((392, 392))
        self.conv_r_1_1 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3)
        self.conv_r_1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3)

        self.conv_out = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride = 2)

    def forward(self, x):
        # Contracting path
        x = self.conv_l_1_1(x)
        x = F.relu(x)
        x = self.conv_l_1_2(x)
        x = F.relu(x)
        assert x.size() == (1, 64, 568, 568)
        encoded1 = torch.clone(x)
        encoded1 = self.crop_1(encoded1)
        x = self.maxpool(x)
        assert x.size() == (1, 64, 284, 284)

        x = self.conv_l_2_1(x)
        x = F.relu(x)
        x = self.conv_l_2_2(x)
        x = F.relu(x)
        assert x.size() == (1, 128, 280, 280)
        encoded2 = torch.clone(x)
        encoded2 = self.crop_2(encoded2)
        x = self.maxpool(x)
        assert x.size() == (1, 128, 140, 140)

        x = self.conv_l_3_1(x)
        x = F.relu(x)
        x = self.conv_l_3_2(x)
        x = F.relu(x)
        assert x.size() == (1, 256, 136, 136)
        encoded3 = torch.clone(x)
        encoded3 = self.crop_3(encoded3)
        x = self.maxpool(x)
        assert x.size() == (1, 256, 68, 68)

        x = self.conv_l_4_1(x)
        x = F.relu(x)
        x = self.conv_l_4_2(x)
        x = F.relu(x)
        assert x.size() == (1, 512, 64, 64)
        encoded4 = torch.clone(x)
        encoded4 = self.crop_4(encoded4)
        x = self.maxpool(x)
        assert x.size() == (1, 512, 32, 32)

        # Middle
        x = self.conv_mid_1(x)
        x = F.relu(x)
        x = self.conv_mid_2(x)
        x = F.relu(x)
        assert x.size() == (1, 1024, 28, 28)
        x = self.up_conv_4(x)
        assert x.size() == (1, 512, 56, 56)

        # Expansive path
        x = torch.cat((x, encoded4), dim = 1)
        x = self.conv_r_4_1(x)
        x = F.relu(x)
        x = self.conv_r_4_2(x)
        x = F.relu(x)
        assert x.size() == (1, 512, 52, 52)
        x = self.up_conv_3(x)
        assert x.size() == (1, 256, 104, 104)

        x = torch.cat((x, encoded3), dim=1)
        x = self.conv_r_3_1(x)
        x = F.relu(x)
        x = self.conv_r_3_2(x)
        x = F.relu(x)
        assert x.size() == (1, 256, 100, 100)
        x = self.up_conv_2(x)
        assert x.size() == (1, 128, 200, 200)

        x = torch.cat((x, encoded2), dim = 1)
        x = self.conv_r_2_1(x)
        x = F.relu(x)
        x = self.conv_r_2_2(x)
        x = F.relu(x)
        assert x.size() == (1, 128, 196, 196)
        x = self.up_conv_1(x)
        assert x.size() == (1, 64, 392, 392)

        x = torch.cat((x, encoded1), dim=1)
        x = self.conv_r_1_1(x)
        x = F.relu(x)
        x = self.conv_r_1_2(x)
        x = F.relu(x)
        assert x.size() == (1, 64, 388, 388)
        x = self.conv_out(x)
        assert x.size() == (1, 1, 388, 388)

        return F.sigmoid(x)