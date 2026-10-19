import torch
import torch.nn as nn

# simple mlp for supervised image classification
class MLPClassifier(nn.Module):

    def __init__(self, input_size, hidden_size=256, num_classes=10, dropout=0.2):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.dropout = dropout
        
        self.flatten = nn.Flatten()
        
        self.linear_relu_stack  = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            
            nn.Linear(self.hidden_size, self.num_classes)  # output
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits