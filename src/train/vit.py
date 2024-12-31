import math
import torch
from torch import nn
from torch.nn import functional as F


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, num_heads=12, embed_dim=768, depth=12):
        super(VisionTransformer, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Creating patch embedding
        self.patch_embedding = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.position_embedding = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads) for _ in range(depth)
        ])
        
        self.layernorm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (batch_size, in_channels, height, width)
        batch_size = x.shape[0]
        x = self.patch_embedding(x)  # (batch_size, embed_dim, num_patches_height, num_patches_width)
        x = x.flatten(2).transpose(1, 2)  # (batch_size, num_patches, embed_dim)
        
        # Add class token
        class_token = self.class_token.expand(batch_size, -1, -1)  # (batch_size, 1, embed_dim)
        x = torch.cat((class_token, x), dim=1)  # (batch_size, num_patches + 1, embed_dim)
        
        # Adding positional embedding
        x += self.position_embedding
        
        # Applying transformer layers
        for layer in self.transformer_layers:
            x = layer(x)

        # Final layer normalization
        x = self.layernorm(x)
        
        return x   # output (batchsize, num_patches + 1, embed_dim)

