from torchvision import transforms
import torch

class GaussianNoise:

    def __init__(self , mean=0.0 , std=0.05):

        self.mean = mean
        self.std = std

    def __call__(self , tensor):

        n_tens = tensor + torch.randn(tensor.size()) * self.std + self.mean

        return torch.clamp(n_tens , 0.0 , 1.0)
    
    def __repr__(self):

        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std})"
    
       


class Aug:

    def __init__(self):

        self.physical_trans = transforms.Compose(
            [
                transforms.Resize(size=(80,80)),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5)
            ]
        )

        self.quality_trans = transforms.Compose(
            [
                transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
                transforms.ToTensor(),
                GaussianNoise(mean=0.0 , std=0.05),
                transforms.ToPILImage(),
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
                transforms.ToTensor()
            ]

        )

    def __call__(self , img):

        phys_img = self.physical_trans(img)

        qual_img = self.quality_trans(phys_img)

        return qual_img
    


class BaseAug:

    def __init__(self):
        
        self.trans = transforms.Compose(
            [
                transforms.Resize(size=(80,80)),
                transforms.ToTensor()
            ]
        )

    def __call__(self, img):
        img_ten = self.trans(img)

        return img_ten