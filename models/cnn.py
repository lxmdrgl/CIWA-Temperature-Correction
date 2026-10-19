import torch
import torch.nn as nn

class CNNRegressor(nn.Module):
    def __init__(self, in_channels=1, hidden_size=256, num_outputs=1, dropout=0.2):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.num_outputs = num_outputs
        self.dropout = dropout
        
        self.features = nn.Sequential(
            # c -> 32
            nn.Conv2d(self.in_channels , 32, kernel_size=5, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(self.dropout),

            # 32 -> 64
            nn.Conv2d(32, 64, kernel_size=5, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(self.dropout),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            
            nn.LazyLinear(self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            
            nn.Linear(self.hidden_size, self.num_outputs)  # output
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x