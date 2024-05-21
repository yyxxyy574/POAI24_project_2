# load dataset

import numpy as np
import pandas as pd
import os
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

# 数据集文件夹路径
DATA_DIR = "../dataset"
# 划分数据集：train、val、test
MODE = {"train", "val", "test"}
# IDX到类别名称的映射
IDX_TO_CLASS = {
    "dataset": {
        0: "building",
        1: "forest",
        2: "glacier",
        3: "mountain",
        4: "sea",
        5: "street"
    },
    "dataset_without_building": {
        0: "forest",
        1: "glacier",
        2: "mountain",
        3: "sea",
        4: "street"
    },
    "dataset_without_forest": {
        0: "building",
        1: "glacier",
        2: "mountain",
        3: "sea",
        4: "street"
    },
    "dataset_without_glacier": {
        0: "building",
        1: "forest",
        2: "mountain",
        3: "sea",
        4: "street"
    }
}
# 类别名称到IDX的映射
CLASS_TO_IDX = {
    "dataset": {
        "building": 0,
        "forest": 1,
        "glacier": 2,
        "mountain": 3,
        "sea": 4,
        "street": 5
    },
    "dataset_without_building": {
        "forest": 0,
        "glacier": 1,
        "mountain": 2,
        "sea": 3,
        "street": 4
    },
    "dataset_without_forest": {
        "building": 0,
        "glacier": 1,
        "mountain": 2,
        "sea": 3,
        "street": 4
    },
    "dataset_without_glacier": {
        "building": 0,
        "forest": 1,
        "mountain": 2,
        "sea": 3,
        "street": 4
    }
}
# 提前计算得到的各子数据集图像数据均值
MEAN = {
    "dataset": {
        "train": [0.43018678, 0.45764188, 0.4544048],
        "val": [0.4293201, 0.45429784, 0.44809443],
        "test": [0.43320544, 0.46033092, 0.45366083]
    },
    "dataset_without_building": {
        "train": [0.42723313, 0.45767236, 0.45401709],
        "val": [0.4261478, 0.45439733, 0.44749748],
        "test": [0.42879757, 0.45936094, 0.45267377]
    },
    "dataset_without_forest": {
        "train": [0.44953664, 0.47691758, 0.49546369],
        "val": [0.44857056, 0.47385164, 0.48927451],
        "test": [0.4544282, 0.48094731, 0.49682382]
    },
    "dataset_without_glacier": {
        "train": [0.42189549, 0.44239091, 0.42905676],
        "val": [0.42010844, 0.4388536, 0.42330006],
        "test": [0.423788, 0.4442588, 0.42714367]
    }
}
# 提前计算得到的各子数据集图像数据方差
STD = {
    "dataset": {
        "train": [0.23773601, 0.2355641, 0.24400209],
        "val": [0.23823929, 0.2355989, 0.24303042],
        "test": [0.23736032, 0.234959, 0.24322382]
    },
    "dataset_without_building": {
        "train": [0.23120482, 0.2301965, 0.23762788],
        "val": [0.2303304, 0.22870443, 0.23569961],
        "test": [0.2303304, 0.22870443, 0.23569961]
    },
    "dataset_without_forest": {
        "train": [0.23920256, 0.23744866 ,0.251838],
        "val": [0.24023413, 0.23805178, 0.25124702],
        "test": [0.23972289, 0.23757087, 0.25189509]
    },
    "dataset_without_glacier": {
        "train": [0.23627436, 0.2371281, 0.24711333],
        "val": [0.23630099, 0.23655915, 0.24559503],
        "test": [0.2356751, 0.23629267, 0.24620541]
    }
}

# 划分子数据集，并按类别存放图片数据
def split_folder():
    if os.path.exists(f"{DATA_DIR}/train"):
        return
    
    for mode in MODE:
        os.makedirs(f"{DATA_DIR}/{mode}")
        for class_name in IDX_TO_CLASS.values():
            os.makedirs(f"{DATA_DIR}/{mode}/{class_name}")

        # 读取csv中子数据集图片名称和类别的信息
        data_info = pd.read_csv(f"{DATA_DIR}/{mode}_data.csv")
        img_names = data_info['image_name']
        img_idxes = data_info['label']
        for i in range(len(img_names)):
            img_name = img_names[i]
            img_idx = img_idxes[i]
            img = Image.open(f"{DATA_DIR}/imgs/{img_name}")
            # 将子数据集中的图片划分到相应类别的文件夹下
            img.save(f"{DATA_DIR}/{mode}/{IDX_TO_CLASS[img_idx]}/{img_name}")

# 计计算图像数据的均值和方差，用于后续归一化处理
def compute_mean_and_std(mode):
    assert mode in MODE, "mode must be one of train, val, and test"
    data_info = pd.read_csv(f"{DATA_DIR}/{mode}_data.csv")
    img_names = data_info['image_name']
    mean = np.zeros(3)
    std = np.zeros(3)

    for img_name in img_names:
        img = np.array(Image.open(f"{DATA_DIR}/imgs/{img_name}")) / 255.
        # 累加每张图片各通道的均值和方差
        for d in range(3):
            mean[d] += img[:, :, d].mean()
            std[d] += img[:, :, d].std()
   
   # 得到整体均值和方差 
    mean /= len(img_names)
    std /= len(img_names)
    return mean, std

# 计算图像数据的均值和方差，用于后续归一化处理，对划分之后的文件夹计算
def compute_mean_and_std_dir(dataset, mode):
    assert mode in MODE, "mode must be one of train, val, and test"
    img_dir = f"../{dataset}/{mode}/"
    mean = np.zeros(3)
    std = np.zeros(3)
    num_images = 0

    for category in os.listdir(img_dir):
        for img_file in os.listdir(f"{img_dir}/{category}/"):
            img = np.array(Image.open(f"{img_dir}/{category}/{img_file}")) / 255.
            num_images += 1
            # 累加每张图片各通道的均值和方差
            for d in range(3):
                mean[d] += img[:, :, d].mean()
                std[d] += img[:, :, d].std()
   
   # 得到整体均值和方差 
    mean /= num_images
    std /= num_images
    return mean, std

# 加载数据集，进行缩放、归一化、随机增强等预处理
def load(dataset, mode, augment=False):
    if not augment:
        # 不需数据增强，则缩放到适当尺寸、转化为tensor并归一化，便于数据读取与训练
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN[dataset][mode], std=STD[dataset][mode])
        ])
    else:
        # 需要数据增强，只在train子数据集中进行
        assert mode == "train", "only under train mode can augment"
        # 加入随机翻转、旋转、裁剪的几何变换，和随机灰度化的色彩变换
        transform = transforms.Compose([
            transforms.Resize((200, 200)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(45),
            transforms.RandomCrop((180, 180)),
            transforms.RandomGrayscale(p=0.3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN[dataset][mode], std=STD[dataset][mode])
        ])
    # 构建可迭代的数据集类
    img_loader = ImageFolder(f"../{dataset}/{mode}/", transform)
    data_loader = DataLoader(img_loader, batch_size=128, shuffle=True)
    return data_loader

# 加载图片，进行缩放、归一化等预处理
def load_image(dataset, mode, img_name):
    # 缩放到适当尺寸、转化为tensor并归一化，便于数据读取与训练
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN[dataset][mode], std=STD[dataset][mode])
    ])
    # 将图像数据转化为Tensor
    img = Image.open(f"../{dataset}/{mode}/{img_name}")
    input_tensor = transform(img)
    # 将图像数据转化为numpy数组
    img = transforms.Resize((224, 224))(img)
    input_array = np.array(img).astype(np.float32) / 255.
    return input_tensor, input_array