import torch
import torch.nn as nn

class GuidedCrossAttention(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4):
        super(GuidedCrossAttention, self).__init__()
        # batch_first=True natively handles [Batch, TimeSteps, EmbedDim] configurations
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embed_dim * 2, embed_dim)
        )

    def forward(self, vision_features, tabular_features):
        """
        Fix: Natively supports longitudinal temporal tracking dimensions without altering shape axes.
        Args:
            vision_features:  [Batch, TimeSteps, 128] spatial structural sequence.
            tabular_features: [Batch, TimeSteps, 128] clinical metrics timeline sequence.
        """
        # Multi-Head cross-attention calculated directly across matching time steps
        attn_output, _ = self.multihead_attn(query=tabular_features, key=vision_features, value=vision_features)
        
        # Residual tracking connections and structural feature normalizations
        x = self.layernorm1(tabular_features + attn_output)
        ffn_output = self.ffn(x)
        fused_tensor = self.layernorm2(x + ffn_output)
        
        return fused_tensor # Returns pristine trajectory matrix: [Batch, TimeSteps, 128]