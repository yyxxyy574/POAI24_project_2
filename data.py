# load dataset

import numpy as np
import pandas as pd
import os
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

DATA_DIR = "./dataset"
MODE = {"train", "val", "test"}
IDX_TO_CLASS = {
    0: "building",
    1: "forest",
    2: "glacier",
    3: "mountain",
    4: "sea",
    5: "street"
}
MEAN = {
    "train": [0.43018678, 0.45764188, 0.4544048],
    "val": [0.4293201, 0.45429784, 0.44809443],
    "test": [0.43320544, 0.46033092, 0.45366083]
}
STD = {
    "train": [0.23773601, 0.2355641, 0.24400209],
    "val": [0.23823929, 0.2355989, 0.24303042],
    "test": [0.23736032, 0.234959, 0.24322382]
}

def split_folder():
    if os.path.exists(f"{DATA_DIR}/train"):
        return
    
    for mode in MODE:
        os.makedirs(f"{DATA_DIR}/{mode}")
        for class_name in IDX_TO_CLASS.values():
            os.makedirs(f"{DATA_DIR}/{mode}/{class_name}")

        data_info = pd.read_csv(f"{DATA_DIR}/{mode}_data.csv")
        img_names = data_info['image_name']
        img_idxes = data_info['label']
        for i in range(len(img_names)):
            img_name = img_names[i]
            img_idx = img_idxes[i]
            img = Image.open(f"{DATA_DIR}/imgs/{img_name}")
            img.save(f"{DATA_DIR}/{mode}/{IDX_TO_CLASS[img_idx]}/{img_name}")

def compute_mean_and_std(mode):
    assert mode in MODE, "mode must be one of train, val, and test"
    data_info = pd.read_csv(f"{DATA_DIR}/{mode}_data.csv")
    img_names = data_info['image_name']
    mean = np.zeros(3)
    std = np.zeros(3)

    for img_name in img_names:
        img = np.array(Image.open(f"{DATA_DIR}/imgs/{img_name}")) / 255.
        for d in range(3):
            mean[d] += img[:, :, d].mean()
            std[d] += img[:, :, d].std()
    
    mean /= len(img_names)
    std /= len(img_names)
    return mean, std


def load(mode, augment=False):
    if not augment:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN[mode], std=STD[mode])
        ])
    else:
        assert mode == "train", "only under train mode can augment"
        transform = transforms.Compose([
            transforms.Resize((200, 200)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(45),
            transforms.RandomCrop((180, 180)),
            transforms.RandomGrayscale(p=0.3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN[mode], std=STD[mode])
        ])
    img_loader = ImageFolder(f"{DATA_DIR}/{mode}/", transform)
    data_loader = DataLoader(img_loader, batch_size=128, shuffle=True)
    return data_loader