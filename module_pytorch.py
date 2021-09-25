import torch
from torch import nn
from torchsummary import summary


# Get cpu or gpu device for training
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))
# Define model


class ResNetGenerator(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, dim=64, n_downsamplings=2,
                 n_blocks=9, image_size=512):
        super(ResNetGenerator, self).__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.dim = dim
        self.n_downsamplings = n_downsamplings
        self.n_blocks = n_blocks
        self.image_size = image_size

        self.main_1 = nn.Sequential(
            nn.Conv2d(self.input_channels, 64, (7, 7), padding=(3, 3), padding_mode='reflect', bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, (3, 3), stride=(2, 2), padding=(1, 1), padding_mode='zeros', bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, (3, 3), stride=(2, 2), padding=(1, 1), padding_mode='zeros', bias=False),
            nn.InstanceNorm2d(256),
            nn.ReLU()
        )

        self.residual_block = nn.Sequential(
            nn.InstanceNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect', bias=False),
            nn.InstanceNorm2d(256),
            nn.InstanceNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, (3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect', bias=False),
            nn.InstanceNorm2d(256)
        )
        self.main_2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, (3, 3), stride=(2, 2), padding=(1, 1), padding_mode='zeros', bias=False),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, (3, 3), stride=(2, 2), padding=(1, 1), padding_mode='zeros', bias=False),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, self.output_channels, (5, 5), (1, 1), padding=(3, 3), padding_mode='reflect'),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.main_1(x)
        for _ in range(self.n_blocks):
            h = x
            x = self.residual_block(x)
            x = torch.add(x, h)
        output = self.main_2(x)

        return output


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

# model = ResNetGenerator().to(device)
# print(model)
# summary(model, (3, 512, 512), 10)
