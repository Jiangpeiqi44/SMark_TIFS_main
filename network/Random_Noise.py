import torch
import torch.nn as nn
import numpy as np
from noise_layers_sep import *
import torchvision.transforms as transforms

class Random_Noise(nn.Module):

    def __init__(self, layers):
        self.layers = layers
        super(Random_Noise, self).__init__()
        for i in range(len(layers)):
            layers[i] = eval(layers[i])
        self.noise = nn.Sequential(*layers)
        # print(self.noise)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def forward(self, image, cover_image):
        # image, cover_image, mask = image_cover_mask[0], image_cover_mask[1], image_cover_mask[2]
        forward_image = image.clone().detach()
        forward_cover_image = cover_image.clone().detach()
        # forward_mask = mask.clone().detach()
        noised_image_C = torch.zeros_like(forward_image)
        # noised_image_R = torch.zeros_like(forward_image)
        # noised_image_F = torch.zeros_like(forward_image)

        for index in range(forward_image.shape[0]):
            flag=True
            while flag:
                random_noise_layer_C = np.random.choice(self.noise, 1)[0]
                # print(random_noise_layer_C)
                noised_image_C[index] = random_noise_layer_C([forward_image[index].clone().unsqueeze(0), forward_cover_image[index].clone().unsqueeze(0)])
                if not torch.isnan(noised_image_C[index]).any():
                    flag=False
            '''single_image = ((noised_image_C[index].clamp(-1, 1).permute(1, 2, 0) + 1) / 2 * 255).add(0.5).clamp(0, 255).to('cpu', torch.uint8).numpy()
            im = Image.fromarray(single_image)
            read = np.array(im, dtype=np.uint8)
            noised_image_C[index] = self.transform(read).unsqueeze(0).to(image.device)

            single_image = ((noised_image_R[index].clamp(-1, 1).permute(1, 2, 0) + 1) / 2 * 255).add(0.5).clamp(0, 255).to('cpu', torch.uint8).numpy()
            im = Image.fromarray(single_image)
            read = np.array(im, dtype=np.uint8)
            noised_image_R[index] = self.transform(read).unsqueeze(0).to(image.device)

            single_image = ((noised_image_F[index].clamp(-1, 1).permute(1, 2, 0) + 1) / 2 * 255).add(0.5).clamp(0, 255).to('cpu', torch.uint8).numpy()
            im = Image.fromarray(single_image)
            read = np.array(im, dtype=np.uint8)
            noised_image_F[index] = self.transform(read).unsqueeze(0).to(image.device)

        noised_image_gap_C = noised_image_C - forward_image
        noised_image_gap_R = noised_image_R - forward_image
        noised_image_gap_F = noised_image_F - forward_image'''
        noised_image_gap_C = noised_image_C.clamp(-1, 1) - forward_image
        # noised_image_gap_R = noised_image_R.clamp(-1, 1) - forward_image
        # noised_image_gap_F = noised_image_F.clamp(-1, 1) - forward_image

        return image + noised_image_gap_C
    
    def test(self, image, cover_image, no_idx):
        # image, cover_image, mask = image_cover_mask[0], image_cover_mask[1], image_cover_mask[2]
        forward_image = image.clone().detach()
        forward_cover_image = cover_image.clone().detach()
        # forward_mask = mask.clone().detach()
        noised_image_C = torch.zeros_like(forward_image)
        # noised_image_R = torch.zeros_like(forward_image)
        # noised_image_F = torch.zeros_like(forward_image)

        for index in range(forward_image.shape[0]):
            random_noise_layer_C = self.noise[no_idx]
            # print(random_noise_layer_C)
            noised_image_C[index] = random_noise_layer_C([forward_image[index].clone().unsqueeze(0), forward_cover_image[index].clone().unsqueeze(0)])

            '''single_image = ((noised_image_C[index].clamp(-1, 1).permute(1, 2, 0) + 1) / 2 * 255).add(0.5).clamp(0, 255).to('cpu', torch.uint8).numpy()
            im = Image.fromarray(single_image)
            read = np.array(im, dtype=np.uint8)
            noised_image_C[index] = self.transform(read).unsqueeze(0).to(image.device)

            single_image = ((noised_image_R[index].clamp(-1, 1).permute(1, 2, 0) + 1) / 2 * 255).add(0.5).clamp(0, 255).to('cpu', torch.uint8).numpy()
            im = Image.fromarray(single_image)
            read = np.array(im, dtype=np.uint8)
            noised_image_R[index] = self.transform(read).unsqueeze(0).to(image.device)

            single_image = ((noised_image_F[index].clamp(-1, 1).permute(1, 2, 0) + 1) / 2 * 255).add(0.5).clamp(0, 255).to('cpu', torch.uint8).numpy()
            im = Image.fromarray(single_image)
            read = np.array(im, dtype=np.uint8)
            noised_image_F[index] = self.transform(read).unsqueeze(0).to(image.device)

        noised_image_gap_C = noised_image_C - forward_image
        noised_image_gap_R = noised_image_R - forward_image
        noised_image_gap_F = noised_image_F - forward_image'''
        noised_image_gap_C = noised_image_C.clamp(-1, 1) - forward_image
        # noised_image_gap_R = noised_image_R.clamp(-1, 1) - forward_image
        # noised_image_gap_F = noised_image_F.clamp(-1, 1) - forward_image

        return image + noised_image_gap_C
