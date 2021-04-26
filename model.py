import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision


class VGG_model(nn.Module):
    def __init__(self):
        super(VGG_model, self).__init__()
        self.model_layers = torchvision.models.vgg.vgg19(pretrained=True).features
        self.content_layers = ["0", "5", "10", "19", "28"]

        for parameter in self.model_layers.parameters():
            parameter.requires_grad = False

    def forward(self, image):
        batch_size = image.size(0)
        output = image
        output_layers = []
        for name, module in self.model_layers.named_children():
            output = module(output)
            if name in self.content_layers:
                output_layers.append(output)
        return output_layers

    def feature_perceptual_loss(self, recon_x, x):
        total_loss = 0
        for x1, x2 in zip(recon_x, x):
            total_loss += F.mse_loss(x1, x2)
        return total_loss

    def style_loss(self, generated_image, content_images, style_images, alpha=0.5, beta=0.5):

        Style_Loss = 0
        Content_Loss = 0

        for x, y, z in zip(generated_image, content_images, style_images):
            b, c, w, h = x.shape
            gm1 = x.view(c, w * h).mm(x.view(c, w * h).t())
            gm2 = z.view(c, w * h).mm(z.view(c, w * h).t())
            Content_Loss += torch.mean((x - y) ** 2)
            Style_Loss += torch.mean((gm1 - gm2) ** 2)

        return (alpha * Content_Loss) + (beta * Style_Loss)
