from data_prep.data_agg import Dataset_1 , Dataset_2
from data_prep.data_aug import Aug , BaseAug
from torch.utils.data import DataLoader , Dataset

import random
from PIL import Image

class UniformTrianingDataset(Dataset):

    def __init__(self):
        super().__init__()
        self.augmentation = Aug()
        self.data1 = Dataset_1().get_training_data()
        self.data2 = Dataset_2().get_training_data()
        self.image_path = self._get_mixed_img()
        self.labels = self._get_mixed_labels()

        self._shuffle()

    def _get_mixed_img(self):
        agg_img = []
        agg_img.extend(self.data1[0])
        agg_img.extend(self.data2[0])

        return agg_img

    def _get_mixed_labels(self):
        agg_labels = []
        agg_labels.extend(self.data1[1])
        agg_labels.extend(self.data2[1])

        return agg_labels
    
    def _shuffle(self):

        img_path_label_pair = list(zip(self.image_path , self.labels))
        random.shuffle(img_path_label_pair)
        self.image_path , self.labels = map(list , zip(*img_path_label_pair))

    

    def __len__(self):

        return len(self.labels)

    def __getitem__(self, index):

        img_path = self.image_path[index]
        label = self.labels[index]

        img = Image.open(img_path)

        aug_img = self.augmentation(img=img)

        return aug_img , label
    

    



class UniformValidationDataset(Dataset):

    def __init__(self): 
        super().__init__()

        self.augmentation = BaseAug()


        self.data1 = Dataset_1().get_validation_data()
        self.data2 = Dataset_2().get_validation_data()
        self.image_path = self._get_mixed_img()
        self.labels = self._get_mixed_labels()

        self._shuffle()

    def _get_mixed_img(self):
        agg_img = []
        agg_img.extend(self.data1[0])
        agg_img.extend(self.data2[0])

        return agg_img

    def _get_mixed_labels(self):
        agg_labels = []
        agg_labels.extend(self.data1[1])
        agg_labels.extend(self.data2[1])

        return agg_labels
    
    def _shuffle(self):

        img_path_label_pair = list(zip(self.image_path , self.labels))
        random.shuffle(img_path_label_pair)
        self.image_path , self.labels = map(list , zip(*img_path_label_pair))

    def __len__(self):

        return len(self.labels)

    def __getitem__(self, index):

        img_path = self.image_path[index]
        label = self.labels[index]

        img = Image.open(img_path)

        aug_img = self.augmentation(img=img)

        

        return aug_img , label





