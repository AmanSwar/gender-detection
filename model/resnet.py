import torch
import torch.nn as nn
from torchvision.models import resnet18 , ResNet18_Weights

class Custom_resnet(nn.Module):

    def __init__(self,  num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.base_model = resnet18(weights=ResNet18_Weights)
        self.base_model_in_ch = self.base_model.fc.in_features
        self.base_model.fc = nn.Linear(self.base_model_in_ch , num_classes)


    def forward(self , x):
        out = self.base_model(x)

        return out
    

if __name__ == "__main__":

    model = Custom_resnet(num_classes=2)

    print(model)
