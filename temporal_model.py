import os
import torch
import torch.nn as nn
from torchvision import models

class TemporalCNN_LSTM(nn.Module):
    """
    Temporal Deepfake Detector combining:
    1. Spatial Feature Extractor: MobileNetV3-Large (per-frame feature extraction -> 960 dims)
    2. Temporal Aggregator: Bidirectional LSTM (256 hidden units -> 512 dims)
    3. Temporal Attention: Learns dynamic weights across the time sequence to focus on artifact frames
    4. Classification Head: Linear(512, 128) -> ReLU -> Dropout -> Linear(128, num_classes)
    """
    def __init__(self, num_classes=2, hidden_dim=256, num_layers=1, pretrained_spatial=True):
        super().__init__()
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained_spatial else None
        base = models.mobilenet_v3_large(weights=weights)
        self.spatial_extractor = base.features
        self.spatial_pool = base.avgpool
        
        self.lstm = nn.LSTM(
            input_size=960,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )
        
        # Temporal Attention Pooling (focuses on frames with unnatural artifacts/flickers)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x shape: (B, T, C, H, W) where T is the sequence length (e.g. 8)
        B, T, C, H, W = x.shape
        x_flat = x.view(B * T, C, H, W)
        
        feats = self.spatial_pool(self.spatial_extractor(x_flat))
        feats = torch.flatten(feats, 1) # (B*T, 960)
        feats_seq = feats.view(B, T, 960) # (B, T, 960)
        
        lstm_out, _ = self.lstm(feats_seq) # (B, T, hidden_dim*2)
        
        # Temporal Attention
        attn_scores = self.attention(lstm_out) # (B, T, 1)
        attn_weights = torch.softmax(attn_scores, dim=1) # (B, T, 1)
        context = torch.sum(lstm_out * attn_weights, dim=1) # (B, hidden_dim*2)
        
        return self.classifier(context)

def get_temporal_model(num_classes=2, pretrained_spatial=True):
    return TemporalCNN_LSTM(num_classes=num_classes, pretrained_spatial=pretrained_spatial)
