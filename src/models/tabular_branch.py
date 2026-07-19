import torch
import torch.nn as nn

class TabularContextEmbedder(nn.Module):
    def __init__(self, input_dim=7, out_features=128):
        super(TabularContextEmbedder, self).__init__()
        
        # Fix: Replaced BatchNorm1d with LayerNorm to completely avoid Batch Size == 1 runtime crashes
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, out_features),
            nn.LayerNorm(out_features),
            nn.ReLU()
        )

    def forward(self, x):
        # x shape can handle [Batch * TimeSteps, Input_Dim] seamlessly
        return self.mlp(x)