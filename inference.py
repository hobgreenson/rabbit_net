import argparse
import torch
import torch.nn.functional as F
from device import DEVICE
from model import build_model


class RabbitInference:

    def __init__(self, checkpoint_path, jit_model=False):
        self.model, self.preprocess = build_model()
        self.model.to(DEVICE)
        self.checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        self.model.eval()
        if jit_model:
            self.jit_model()

    def jit_model(self):
        x = torch.rand((1, 3, 224, 224), device=DEVICE)
        self.model = torch.jit.trace(self.model, (x, ))

    def predict(self, x):
        x = self.preprocess(x).unsqueeze(dim=0).to(DEVICE)
        y = F.sigmoid(self.model(x)).item()
        return y
