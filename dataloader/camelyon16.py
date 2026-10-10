import os
import PIL
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from dataloader.common import get_data_transforms
from tqdm import tqdm
import glob
import h5py
from .wsi_utils import read_region

PATCH_ROOT = ""
LABEL_ROOT = ""
TIF_ROOT = ""

class Camelyon16(Dataset):

    def __init__(self, setname, image_size, transform = None, mean_train = None, std_train = None, slide_name = None, negative_only = False):
        assert setname == "train" or setname == "test" or setname == "val"

        if transform is None:
            transform, gt_transform = get_data_transforms(image_size = image_size)
        self.transform = transform
        self.data = []
        self.label = []

        if setname == "train" or setname == "val":
            with open(f"dataloader/data_split/camelyon16/{setname}_list.txt") as f:
                slide_list = f.readlines()
                slide_list = [x.strip() for x in slide_list]

            coords_with_name = None
            for slide_name in tqdm(slide_list):
                patches_path = os.path.join(PATCH_ROOT, slide_name + "_patches.h5")
                with h5py.File(patches_path, "r") as f_patch:
                    coords = f_patch["coords"][:]
                    slide_col = np.array([[slide_name] * len(coords)]).T
                    if coords_with_name is None:
                        coords_with_name = np.concatenate([slide_col, coords], axis=1)
                    else:
                        coords_tmp = np.concatenate([slide_col, coords], axis=1)
                        coords_with_name = np.concatenate([coords_with_name, coords_tmp], axis = 0)

            if setname == "train":
                np.random.shuffle(coords_with_name)
            self.data = coords_with_name
            self.label = [0] * self.data.shape[0]
            print(f"length of {setname}:", len(self.label))
        else:

            is_negative = True if "negative" in slide_name else False

            slide_name = slide_name.replace("_negative", "")
            slide_name = slide_name.replace("_positive", "")

            #load test data by slides
            patches_path = os.path.join(PATCH_ROOT, slide_name + "_patches.h5")
            with h5py.File(patches_path, "r") as f_patch:
                coords = f_patch["coords"][:]
                slide_col = np.array([[slide_name] * len(coords)]).T
                coords_with_name = np.concatenate([slide_col, coords], axis=1)

            self.data = coords_with_name
            if is_negative:
                self.label = [0] * self.data.shape[0]
            else:
                label_h5 = slide_name + "-label.h5"
                label_path = os.path.join(LABEL_ROOT, label_h5)
                with h5py.File(label_path, "r") as f_label:
                    label_coords = f_label["coords"][:]
                    labels = f_label["labels"][:]

                    coord2label = {tuple(coord): label for coord, label in zip(label_coords, labels)}
                for _, x, y in coords_with_name:
                    label_value = coord2label.get((int(x), int(y)), None)
                    if label_value is not None and label_value == 1:
                        self.label.append(1)
                    else:
                        self.label.append(0)

            print("positive samples:", self.label.count(1), "negative samples:", self.label.count(0))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):

        slide_name, x, y = self.data[i]
        slide_name = str(slide_name)

        x = int(x)
        y = int(y)

        label = self.label[i]
        svs_path = os.path.join(TIF_ROOT, slide_name + ".tif")
        patch_size = 256
        level = 0
        pil_image = read_region(svs_path, location = (x, y), level = level, size = (patch_size, patch_size)).convert("RGB")
        image = self.transform(pil_image)
        pil_image.close()

        return image, label, slide_name

    @classmethod
    def load_test_slides(cls):
        with open("dataloader/data_split/camelyon16/test_list.txt") as f:
            test_slide_list = f.readlines()
            test_slide_list = [x.strip() for x in test_slide_list]

            return test_slide_list
