import torch
import torch.nn as nn
import pandas as pd
from src.config import NUM_CLASSES, INPUT_DIM

class AirQualityMLP(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, hidden_dims = [128, 64], num_classes = NUM_CLASSES, dropout = 0.3):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers += [
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.tensor) -> torch.Tensor:
        return self.network(x)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)