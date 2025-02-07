import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0 , EfficientNet_B0_Weights

class Custom_efficientNet(nn.Module):

    def __init__(self , num_class):

        super().__init__()

        self.num_class = num_class
        self.base_model = efficientnet_b0(weights = EfficientNet_B0_Weights)
        self.base_model.classifier = nn.Sequential(
            nn.Linear(in_features=1280 , out_features=512 , bias=True),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.5 , inplace=True),
            nn.Linear(in_features=512 , out_features=128 , bias=True),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.5 , inplace=True),
            nn.Linear(in_features=128 , out_features=self.num_classes ,bias=True)
        )

    def forward(self , x):

        out = self.base_model(x)

        return out
        

if __name__ == "__main__":

    model = efficientnet_b0()

    print(model)