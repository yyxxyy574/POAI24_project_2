# 定义主流程

import argparse
import wandb
import torch

import data
from model import basic, advance
from classifier import Classifier

if __name__ == "__main__":
    # 运行代码参数使用方法详见README.md
    parser = argparse.ArgumentParser("code")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--train", action="store_true")
    group.add_argument("--test", action="store_true")
    parser.add_argument("--save-name", metavar="file", action="store", default="model")
    parser.add_argument("--load-from", metavar="file", action="store", default=None)
    parser.add_argument("--model", metavar="model", action="store", default="basic")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--optimizer", metavar="optimizer", action="store", default="SGD")
    parser.add_argument("--learning-rate", metavar = "value", type = float, action = "store", default = 1e-3)
    parser.add_argument("--momentum", metavar = "value", type = float, action = "store", default = 1e-4)
    parser.add_argument("--weight-decay", metavar = "value", type = float, action = "store", default = 1e-5)
    options = parser.parse_args()
    
    # 首先划分数据集
    data.split_folder()
    # 指定运行设备
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")
    
    # 构建卷积神经网络模型
    assert options.model in {"basic", "advance"}, "not right model"
    if options.model == "basic":
        model = basic.Basic()
    elif options.model == "advance":
        model = advance.Advance()
    # 构建深度学习框架
    classfier = Classifier(options.save_name, options.load_from, model, device, augment=options.augment, optimizer=options.optimizer, lr=options.learning_rate, momentum=options.momentum, weight_decay=options.weight_decay)
    
    if options.train:
        wandb.init(project="POAI24_project_2", entity="yyxxyy574", name=options.save_name)
        classfier.train()
        
    if options.test:
        classfier.evluate()