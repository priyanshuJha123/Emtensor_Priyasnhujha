import torch.nn as nn

class AnomalyDetectionModule(nn.Module):
    """Anomaly detection module using attention gates."""
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv3d(input_channels, output_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.attention(x) * x
