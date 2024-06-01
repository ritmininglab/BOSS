import torch
import torch.nn as nn
from torchvision.models import vision_transformer
import torch.nn.functional as F
import pdb

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class ViTWithFeaturesAndPredictions(nn.Module):
    def __init__(self, num_classes=1000, pretrained=False, **kwargs):
        super(ViTWithFeaturesAndPredictions, self).__init__()
        self.original_model = vision_transformer.vit_b_16(pretrained=pretrained, **kwargs)
        head_size = self.original_model.heads.head.in_features
        self.original_model.heads = Identity()
        self.fc = nn.Linear(head_size, num_classes)

    def forward(self, x):
        x = self.original_model(x)
        features = x.detach().cpu()
        x = self.fc(x)
        return x, features


def vit(num_classes=100, initial_train=None):
    return ViTWithFeaturesAndPredictions(num_classes=num_classes, pretrained=True)