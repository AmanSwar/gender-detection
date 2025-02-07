import torch.nn as nn
from torchvision.models import squeezenet1_1 , SqueezeNet1_1_Weights


class Custom_squeezenet(nn.Module):

    def __init__(self , num_class):
        super().__init__()
        self.num_class = num_class
        self.base_model = squeezenet1_1(weights=SqueezeNet1_1_Weights)


        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=0.5 , inplace=False),
            nn.Conv2d(512 , 2 , kernel_size=(1,1) , stride=(1,1)),
            nn.AdaptiveAvgPool2d(output_size=(1,1))
        )

    def forward(self , x):

        out = self.base_model(x)

        return out


if __name__ == "__main__":
    model = Custom_squeezenet(num_class=2)
    print(model)
