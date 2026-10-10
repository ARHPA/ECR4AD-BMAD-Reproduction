import os
import PIL
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from dataloader.common import get_data_transforms
from tqdm import tqdm
import glob

class GleasonArvaniti(Dataset):
    def __init__(self, data_root, setname, image_size, mean_train = None, std_train = None):
        assert setname == "train" or setname == "test" or setname == "val"
        train_transform, test_transform  = get_data_transforms(image_size = image_size, mean_train = mean_train, std_train = std_train,)

        #The train test split is same as https://github.com/kaiko-ai/eva/blob/0392811714b554719959f6287d91c81b8f2b2a91/src/eva/vision/data/datasets/classification/gleason_arvaniti.py
        self.transform = test_transform
        if setname == "train":
            id_list = ["ZT111", "ZT199", "ZT204"]
            self.transform = train_transform
        elif setname == "val":
            id_list = ["ZT76"]

        self.data = []
        self.label = []
        self.setname = setname

        if setname == "train":
            image_path_list = []
            for id_name in id_list:
                image_path_list = image_path_list + glob.glob(os.path.join(data_root, "train_validation_patches_750", id_name+"_*", "*class_0.jpg"))
            self.label = [0] * len(image_path_list)
            self.data = image_path_list
            #print("length of train data", len(self.data))
        elif setname == "val":
            negative_path_list = []
            positive_path_list = []
            for id_name in id_list:
                negative_path_list = negative_path_list + glob.glob(os.path.join(data_root, "train_validation_patches_750", id_name+"_*", "*class_0.jpg"))
                positive_path_list = positive_path_list + glob.glob(os.path.join(data_root, "train_validation_patches_750", id_name+"_*", "*class_1.jpg")) 
                positive_path_list = positive_path_list + glob.glob(os.path.join(data_root, "train_validation_patches_750", id_name+"_*", "*class_2.jpg")) 
                positive_path_list = positive_path_list + glob.glob(os.path.join(data_root, "train_validation_patches_750", id_name+"_*", "*class_3.jpg")) 
            self.label = [0] * len(negative_path_list) + [1] * len(positive_path_list)
            self.data = negative_path_list + positive_path_list
            print("in val negative_path_list:", len(negative_path_list), "positive_path_list:", len(positive_path_list))
        elif setname == "test":
            negative_path_list = []
            positive_path_list = []
            negative_path_list = negative_path_list + glob.glob(os.path.join(data_root, "test_patches_750", "patho_1", "*", "*class_0.jpg"))
            positive_path_list = positive_path_list + glob.glob(os.path.join(data_root, "test_patches_750", "patho_1", "*", "*class_1.jpg")) 
            positive_path_list = positive_path_list + glob.glob(os.path.join(data_root, "test_patches_750", "patho_1", "*", "*class_2.jpg")) 
            positive_path_list = positive_path_list + glob.glob(os.path.join(data_root, "test_patches_750", "patho_1", "*", "*class_3.jpg")) 
            
            self.label = [0] * len(negative_path_list) + [1] * len(positive_path_list)

            print("negative_path_list:", len(negative_path_list), "positive_path_list:", len(positive_path_list))
            #exit()

            self.data = negative_path_list + positive_path_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        image_path = self.data[i]
        pil_image = Image.open(image_path).convert("RGB")
        image = self.transform(pil_image)
        pil_image.close()
        label = self.label[i]

        return image, label, image_path
