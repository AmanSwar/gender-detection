import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small , MobileNet_V3_Small_Weights


class Custom_mobileNet(nn.Module):

    def __init__(self , num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.base_model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights)
        self.base_model.classifier = nn.Sequential(
            nn.Linear(in_features=576 , out_features=1024 , bias=True),
            nn.Hardswish(),
            nn.Dropout(p=0.2 , inplace=True),
            nn.Linear(in_features=1024 , out_features=512 , bias=True),
            nn.Hardswish(),
            nn.Dropout(p=0.2 , inplace=True),
            nn.Linear(in_features=512 , out_features=128 , bias=True),
            nn.Hardswish(),
            nn.Dropout(p=0.5 , inplace=True),
            nn.Linear(in_features=128 , out_features=self.num_classes ,bias=True)

        )

    def forward(self , x):

        out = self.base_model(x)

        return out

if __name__ == "__main__":

    model = Custom_mobileNet(num_classes=2)

    print(model)
