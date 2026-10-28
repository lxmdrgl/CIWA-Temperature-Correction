import torch.nn as nn
from models.normalization import TemperatureNormalizedRegressor

# simple mlp for supervised regression
class MLPRegressor(TemperatureNormalizedRegressor):

    def __init__(self, input_size, hidden_size=256, num_outputs=1, dropout=0.2,
                 image="thermal", normalize="temp", in_channels=3):
        super().__init__(image=image, normalize=normalize, in_channels=in_channels)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_outputs = num_outputs
        self.dropout = dropout
        
        self.flatten = nn.Flatten()
        
        self.linear_relu_stack  = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            
            nn.Linear(self.hidden_size, self.num_outputs)  # output
        )

    def forward(self, x):
        x = self.normalize_inputs(x)
        x = self.flatten(x)
        output = self.linear_relu_stack(x)
        return output
