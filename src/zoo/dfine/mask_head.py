import torch.nn as nn

# Add to your model definition
class MaskHead(nn.Module):
    def __init__(self, hidden_dim, num_heads=8):
        super().__init__()
        
        # Transformer decoder for mask features
        self.mask_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        
    def forward(self, x, bbox_mask):
        # x: features from backbone [B, C, H, W]
        # bbox_mask: binary masks for each predicted box [B, N, H, W]
        
        mask_logits = self.mask_head(x)
        return mask_logits