import torch
import torch.nn as nn
from src.models.vision_branch import OptimizedDenseNet3D
from src.models.tabular_branch import TabularContextEmbedder
from src.models.attention import GuidedCrossAttention

class AlzheimerTrajectoryNet(nn.Module):
    def __init__(self, tabular_dim=7, embed_dim=128, num_classes=3):
        """
        Args:
            tabular_dim (int): Total baseline clinical tabular metrics (Default: 6).
            embed_dim (int): Standardized vector projection size (128).
            num_classes (int): Target diagnostic classes (0: Nondemented, 1: Converted, 2: Demented).
        """
        super(AlzheimerTrajectoryNet, self).__init__()
        
        self.vision_branch = OptimizedDenseNet3D(out_features=embed_dim)
        self.tabular_branch = TabularContextEmbedder(input_dim=tabular_dim, out_features=embed_dim)
        self.cross_attention = GuidedCrossAttention(embed_dim=embed_dim, num_heads=4)
        
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        # Mod A Fix: Deeper, regularized, and LayerNorm-stabilized classification structure
        self.classifier_head = nn.Sequential(
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, num_classes)
        )
        
        self.mmse_regression_head = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, images, tabular, masks):
        batch_size, max_timesteps, channels, d, h, w = images.shape
        
        flat_images = images.view(batch_size * max_timesteps, channels, d, h, w)
        flat_tabular = tabular.view(batch_size * max_timesteps, -1)
        
        flat_vision_feats = self.vision_branch(flat_images)    
        flat_tabular_feats = self.tabular_branch(flat_tabular) 
        
        vision_timeline = flat_vision_feats.view(batch_size, max_timesteps, -1)
        tabular_timeline = flat_tabular_feats.view(batch_size, max_timesteps, -1)
        
        fused_timeline = self.cross_attention(vision_timeline, tabular_timeline) 
        
        valid_lengths = masks.sum(dim=1).int().cpu()
        
        packed_fused = nn.utils.rnn.pack_padded_sequence(
            fused_timeline, valid_lengths, batch_first=True, enforce_sorted=False
        )
        
        packed_gru_out, _ = self.gru(packed_fused)
        gru_out, _ = nn.utils.rnn.pad_packed_sequence(packed_gru_out, batch_first=True, total_length=max_timesteps)
        
        expanded_masks = masks.unsqueeze(-1)
        masked_gru_out = gru_out * expanded_masks
        
        visit_counts = masks.sum(dim=1, keepdim=True)
        pooled_sequence = masked_gru_out.sum(dim=1) / (visit_counts + 1e-8) 
        
        class_logits = self.classifier_head(pooled_sequence)
        predicted_mmse = self.mmse_regression_head(pooled_sequence).squeeze(-1)
        
        return class_logits, predicted_mmse