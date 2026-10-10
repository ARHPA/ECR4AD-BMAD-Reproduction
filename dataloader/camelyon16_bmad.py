import os
import PIL
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from dataloader.common import get_data_transforms
from tqdm import tqdm
import glob

THIS_PATH = os.path.dirname(__file__)

class Camelyon16BMAD(Dataset):

    def __init__(self, data_root, setname, image_size, mean_train = None, std_train = None):
        assert setname == "train" or setname == "test" or setname == "valid"
        train_transform, test_transform  = get_data_transforms(image_size = image_size, mean_train = mean_train, std_train = std_train,)

        if setname == "train":
            self.transform = train_transform
        else:
            self.transform = test_transform

        self.data = []
        self.label = []
        self.image_path = []
        self.setname = setname

        if setname == "train":
            image_path_list = glob.glob(os.path.join(data_root, setname, "good", "*.png"))
            self.label = self.label + [0] * len(image_path_list)
        else:
            bad_path_list = glob.glob(os.path.join(data_root, setname, "Ungood", "img" , "*.png"))
            good_path_list = glob.glob(os.path.join(data_root, setname, "good", "img",  "*.png")) 

            image_path_list = bad_path_list + good_path_list
            self.label = self.label + [1] * len(bad_path_list) + [0] * len(good_path_list)

        self.data = image_path_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        image_path = self.data[i]
        pil_image = Image.open(image_path).convert("RGB")
        image = self.transform(pil_image)
        pil_image.close()
        label = self.label[i]

        return image, label, image_path
