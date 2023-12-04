import torch.nn as nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights


def build_model():
    weights = MobileNet_V3_Large_Weights.DEFAULT
    preprocess = weights.transforms()
    model = mobilenet_v3_large(weights=weights)
    model.classifier[-1] = nn.Linear(1280, 1)
    return model, preprocess


def freeze_weights(model):
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier[-1].parameters():
        param.requires_grad = True
    return model
