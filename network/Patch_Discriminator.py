# from . import *
import torch.nn as nn
class Patch_Discriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, blocks=3, channels=64):
        super(Patch_Discriminator, self).__init__()

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(3, channels, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, blocks):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(channels * nf_mult_prev, channels * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=False),
                nn.InstanceNorm2d(channels * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** blocks, 8)
        sequence += [
            nn.Conv2d(channels * nf_mult_prev, channels * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=False),
            nn.InstanceNorm2d(channels * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(channels * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)
    
if __name__ == '__main__':
    import torch
    # from torchinfo import summary
    model = Patch_Discriminator()
    image_size = 224
    batch_size = 2
    print(model)
    dummy = torch.rand(batch_size, 3, image_size, image_size)
    out = model(dummy)
    print(out.shape)
    # print(out[0][0].shape, out[1][0].shape, out[2][0].shape)
    print(torch.mean(torch.relu(1 - out[0])))


