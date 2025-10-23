import torch
import torch.nn as nn
import torch.nn.functional as F
from network.ResBlock import ResBlock
from network.ConvBlock import ConvBlock


def weight_init(m):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.001)
        m.bias.data.zero_()
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)

    if isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight.data)


class WMEnc(nn.Module):
    def __init__(self, message_length=9, blocks=2, channels=64, attention=None):
        super().__init__()
        self.message_length = message_length
        self.linear1 = nn.Linear(message_length, message_length * message_length)
        self.Conv_message1 = ConvBlock(1, channels, blocks=blocks)
        self.att1 = ResBlock(1024 + channels, 1024, blocks=blocks, attention=attention)

        self.linear2 = nn.Linear(message_length, message_length * message_length)
        self.Conv_message2 = ConvBlock(1, channels, blocks=blocks)
        self.att2 = ResBlock(2048 + channels, 2048, blocks=blocks, attention=attention)

        self.linear3 = nn.Linear(message_length, message_length * message_length)
        self.Conv_message3 = ConvBlock(1, channels, blocks=blocks)
        self.att3 = ResBlock(1024 + channels, 1024, blocks=blocks, attention=attention)

        self.linear4 = nn.Linear(message_length, message_length * message_length)
        self.Conv_message4 = ConvBlock(1, channels, blocks=blocks)
        self.att4 = ResBlock(512 + channels, 512, blocks=blocks, attention=attention)

        self.linear5 = nn.Linear(message_length, message_length * message_length)
        self.Conv_message5 = ConvBlock(1, channels, blocks=blocks)
        self.att5 = ResBlock(256 + channels, 256, blocks=blocks, attention=attention)

        self.linear6 = nn.Linear(message_length, message_length * message_length)
        self.Conv_message6 = ConvBlock(1, channels, blocks=blocks)
        self.att6 = ResBlock(128 + channels, 128, blocks=blocks, attention=attention)

        self.linear7 = nn.Linear(message_length, message_length * message_length)
        self.Conv_message7 = ConvBlock(1, channels, blocks=blocks)
        self.att7 = ResBlock(64 + channels, 64, blocks=blocks, attention=attention)

        self.linear8 = nn.Linear(message_length, message_length * message_length)
        self.Conv_message8 = ConvBlock(1, channels, blocks=blocks)
        self.att8 = ResBlock(64 + channels, 64, blocks=blocks, attention=attention)
        # self.apply(weight_init)

    def forward(self, z_att, watermark):
        """
        z_att1: 1024x2x2, z_att2: 2048 x 4 x 4, z_att3: 1024 x 8 x 8, z_att4: 512 x 16 x 16
        z_att5: 256 x 32 x 32, z_att6: 128 x 64 x 64, z_att7: 64 x 128 x 128, z_att8: 64 x 256 x 256
        """
        # att0
        expanded_message = self.linear1(watermark)
        expanded_message = expanded_message.view(
            -1, 1, self.message_length, self.message_length
        )
        expanded_message = F.interpolate(
            expanded_message,
            size=(z_att[0].shape[2], z_att[0].shape[3]),
            mode="nearest",
        )  # bs x 1024 x 2 x 2
        expanded_message = self.Conv_message1(expanded_message)  # 2 x 2
        z_att0 = torch.cat((z_att[0], expanded_message), dim=1)
        z_att0 = self.att1(z_att0)
        # att1
        expanded_message = self.linear2(watermark)
        expanded_message = expanded_message.view(
            -1, 1, self.message_length, self.message_length
        )
        expanded_message = F.interpolate(
            expanded_message,
            size=(z_att[1].shape[2], z_att[1].shape[3]),
            mode="nearest",
        )
        expanded_message = self.Conv_message2(expanded_message)  #
        z_att1 = torch.cat((z_att[1], expanded_message), dim=1)
        z_att1 = self.att2(z_att1)
        # att2
        expanded_message = self.linear3(watermark)
        expanded_message = expanded_message.view(
            -1, 1, self.message_length, self.message_length
        )
        expanded_message = F.interpolate(
            expanded_message,
            size=(z_att[2].shape[2], z_att[2].shape[3]),
            mode="nearest",
        )
        expanded_message = self.Conv_message3(expanded_message)  #
        z_att2 = torch.cat((z_att[2], expanded_message), dim=1)
        z_att2 = self.att3(z_att2)
        # att3
        expanded_message = self.linear4(watermark)
        expanded_message = expanded_message.view(
            -1, 1, self.message_length, self.message_length
        )
        expanded_message = F.interpolate(
            expanded_message,
            size=(z_att[3].shape[2], z_att[3].shape[3]),
            mode="nearest",
        )
        expanded_message = self.Conv_message4(expanded_message)  #
        z_att3 = torch.cat((z_att[3], expanded_message), dim=1)
        z_att3 = self.att4(z_att3)
        # att4
        expanded_message = self.linear5(watermark)
        expanded_message = expanded_message.view(
            -1, 1, self.message_length, self.message_length
        )
        expanded_message = F.interpolate(
            expanded_message,
            size=(z_att[4].shape[2], z_att[4].shape[3]),
            mode="nearest",
        )
        expanded_message = self.Conv_message5(expanded_message)  #
        z_att4 = torch.cat((z_att[4], expanded_message), dim=1)
        z_att4 = self.att5(z_att4)
        # att5
        expanded_message = self.linear6(watermark)
        expanded_message = expanded_message.view(
            -1, 1, self.message_length, self.message_length
        )
        expanded_message = F.interpolate(
            expanded_message,
            size=(z_att[5].shape[2], z_att[5].shape[3]),
            mode="nearest",
        )
        expanded_message = self.Conv_message6(expanded_message)  #
        z_att5 = torch.cat((z_att[5], expanded_message), dim=1)
        z_att5 = self.att6(z_att5)
        # att6
        expanded_message = self.linear7(watermark)
        expanded_message = expanded_message.view(
            -1, 1, self.message_length, self.message_length
        )
        expanded_message = F.interpolate(
            expanded_message,
            size=(z_att[6].shape[2], z_att[6].shape[3]),
            mode="nearest",
        )
        expanded_message = self.Conv_message7(expanded_message)  #
        z_att6 = torch.cat((z_att[6], expanded_message), dim=1)
        z_att6 = self.att7(z_att6)
        # att7
        expanded_message = self.linear8(watermark)
        expanded_message = expanded_message.view(
            -1, 1, self.message_length, self.message_length
        )
        expanded_message = F.interpolate(
            expanded_message,
            size=(z_att[7].shape[2], z_att[7].shape[3]),
            mode="nearest",
        )
        expanded_message = self.Conv_message8(expanded_message)  #
        z_att7 = torch.cat((z_att[7], expanded_message), dim=1)
        z_att7 = self.att8(z_att7)

        return [z_att0, z_att1, z_att2, z_att3, z_att4, z_att5, z_att6, z_att7]
