

import torch
import torch.nn as nn
import torch.nn.functional as F

class DNNClassifier(nn.Module):
    def __init__(self, input_dim):
        super(DNNClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)   # Couche 1 : 64 neurones
        self.fc2 = nn.Linear(64, 32)          # Couche 2 : 32 neurones
        self.output = nn.Linear(32, 1)        # Couche de sortie : 1 neurone

    def forward(self, x):
        x = F.relu(self.fc1(x))               # Activation ReLU couche 1
        x = F.relu(self.fc2(x))               # Activation ReLU couche 2
        x = torch.sigmoid(self.output(x))     # Activation Sigmoid (classification binaire)
        return x

# Exemple d'utilisation plus tard :
# model = DNNClassifier(input_dim=30)
# output = model(torch.randn(16, 30))  # 16 échantillons fictifs, 30 features
