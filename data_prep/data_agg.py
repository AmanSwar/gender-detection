import torch

import os

class Dataset_1:

    def __init__(self , base_path="data/data1"):

        self.base_path = base_path
        self.training_data = self._get_data("train")
        self.validation_data = self._get_data("valid")

        self._encode_labels()

    def _get_data(self , subset):

        img_list = []
        labels = []
        if subset == "train":

            train_dir = os.path.join(self.base_path , "Training")

            for clss in os.listdir(train_dir):
                print(clss)
                clss_path = os.path.join(train_dir , clss)

                for imgs in os.listdir(clss_path):
                    img_path = os.path.join(clss_path , imgs)
                    img_list.append(img_path)
                    labels.append(clss)

        if subset == "valid":

            train_dir = os.path.join(self.base_path , "Validation")
            for clss in os.listdir(train_dir):
                clss_path = os.path.join(train_dir , clss)
                for imgs in os.listdir(clss_path):
                    img_path = os.path.join(clss_path , imgs)
                    img_list.append(img_path)
                    labels.append(clss)

        return img_list , labels
    
    def _encode_labels(self):

        label_dict = {'female' : 0 , 'male' : 1}

        for i in range(len(self.training_data[1])):
            self.training_data[1][i] = label_dict[self.training_data[1][i]]

        for i in range(len(self.validation_data[1])):
            self.validation_data[1][i] = label_dict[self.validation_data[1][i]]

    def get_training_data(self):

        return self.training_data
    
    def get_validation_data(self):

        return self.validation_data
    

class Dataset_2:

    def __init__(self , base_dir="data/data2"):
        self.base_dir = base_dir
        self.training_data = self._get_data("train")
        self.validation_data = self._get_data("valid")

        self._encode_labels()

    def _get_data(self, subset):

        img_list = []
        labels = []

        if subset == "train":
            train_dir = os.path.join(self.base_dir , "train")

            for clss in os.listdir(train_dir):

                clss_path = os.path.join(train_dir , clss)
                
                for imgs in os.listdir(clss_path):

                    img_path = os.path.join(clss_path , imgs)

                    img_list.append(img_path)
                    labels.append(clss)

        if subset == "valid":
            valid_dir = os.path.join(self.base_dir , "test")
            for clss in os.listdir(valid_dir):
                clss_path = os.path.join(valid_dir , clss)
                for imgs in os.listdir(clss_path):

                    img_path = os.path.join(clss_path , imgs)

                    img_list.append(img_path)
                    labels.append(clss)



        return img_list , labels
    
    def _encode_labels(self):

        label_dict = {'women' : 0 , 'men' : 1}

        for i in range(len(self.training_data[1])):
            self.training_data[1][i] = label_dict[self.training_data[1][i]]

        for i in range(len(self.validation_data[1])):
            self.validation_data[1][i] = label_dict[self.validation_data[1][i]]
    
    def get_training_data(self):

        return self.training_data
    
    def get_validation_data(self):

        return self.validation_data
    


if __name__ == "__main__":

    data2 = Dataset_2().get_training_data()

    print(data2[0][:5])
    print(data2[1][:5])
    
    from matplotlib import image as img
    from matplotlib import pyplot as plt

    plt.title("lake.jpg")
    image = img.imread(data2[0][0])
    plt.imshow(image)
    plt.show()







