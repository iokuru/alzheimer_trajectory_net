import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialSelfAttention3D(nn.Module):
    """
    Computes Multi-Head Scaled Dot-Product Attention across the 3D spatial dimensions.
    This forces the network to weigh the importance of disparate anatomical regions 
    (e.g., Hippocampus vs. Ventricles) globally before temporal extraction.
    """
    def __init__(self, in_channels, num_heads=4):
        super(SpatialSelfAttention3D, self).__init__()
        self.in_channels = in_channels
        
        # Native PyTorch Multi-Head Attention for optimized Q, K, V calculations
        self.attention = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(in_channels)
        
    def forward(self, x):
        # x shape: [Batch, Channels, Depth, Height, Width]
        B, C, D, H, W = x.size()
        
        # Flatten spatial volume into a sequence of tokens: [Batch, D*H*W, Channels]
        tokens = x.view(B, C, -1).permute(0, 2, 1)
        
        # Stabilize variance prior to attention calculations
        tokens_norm = self.norm(tokens)
        
        # Self-Attention: Query, Key, and Value are derived from the same spatial tokens
        attn_output, _ = self.attention(tokens_norm, tokens_norm, tokens_norm)
        
        # Residual connection to prevent gradient vanishing
        out_tokens = tokens + attn_output
        
        # Re-fold back into a 3D volumetric tensor: [Batch, Channels, Depth, Height, Width]
        out_volume = out_tokens.permute(0, 2, 1).view(B, C, D, H, W)
        
        return out_volume


class DenseLayer3D(nn.Module):
    def __init__(self, in_channels, growth_rate, drop_rate=0.2):
        super(DenseLayer3D, self).__init__()
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.conv1 = nn.Conv3d(in_channels, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm3d(4 * growth_rate)
        self.conv2 = nn.Conv3d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return torch.cat([x, out], 1)


class DenseBlock3D(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate, drop_rate=0.2):
        super(DenseBlock3D, self).__init__()
        layers = []
        for i in range(num_layers):
            layers.append(DenseLayer3D(in_channels + i * growth_rate, growth_rate, drop_rate))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class TransitionLayer3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer3D, self).__init__()
        self.bn = nn.BatchNorm3d(in_channels)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = self.pool(out)
        return out


class OptimizedDenseNet3D(nn.Module):
    def __init__(self, out_features=128, growth_rate=16, block_config=(4, 4, 4, 4), drop_rate=0.2):
        super(OptimizedDenseNet3D, self).__init__()
        
        num_init_features = 2 * growth_rate
        
        # Initial Convolution
        self.features = nn.Sequential(
            nn.Conv3d(1, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )
        
        # DenseBlocks tracking for dynamic hook registration in Grad-CAM
        self.blocks = nn.ModuleList()
        num_features = num_init_features
        
        for i, num_layers in enumerate(block_config):
            block = DenseBlock3D(num_layers, num_features, growth_rate, drop_rate)
            self.blocks.append(block)
            num_features = num_features + num_layers * growth_rate
            
            # --- INTEGRATED SELF-ATTENTION MODULE ---
            # Insert the attention mechanism exactly after DenseBlock3, mirroring the benchmark
            if i == 2:
                self.blocks.append(SpatialSelfAttention3D(in_channels=num_features, num_heads=4))
            
            if i != len(block_config) - 1:
                trans = TransitionLayer3D(num_features, num_features // 2)
                self.blocks.append(trans)
                num_features = num_features // 2
                
        self.final_bn = nn.BatchNorm3d(num_features)
        
        # Final Vector Projection
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Flatten(),
            nn.Linear(num_features, out_features)
        )

    def forward(self, x):
        features = self.features(x)
        for block in self.blocks:
            features = block(features)
            
        features = F.relu(self.final_bn(features))
        out = self.classifier(features)
        return out