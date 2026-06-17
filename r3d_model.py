import torch
import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights

def get_r3d_model(num_classes):
    model = r3d_18(weights=R3D_18_Weights.DEFAULT)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model