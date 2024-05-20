# 基础模型

import torch.nn as nn

class Basic(nn.Module):
    def __init__(self):
        super(Basic, self).__init__()
        # 第一个卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # 第二个卷积层
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # 展平层
        self.flatten = nn.Flatten()
        # 分类层
        self.classifier = nn.Sequential(
            nn.Linear(32 * 56 * 56, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            # 最后输出维度为类别数6
            nn.Linear(128, 6)
        )

    def forward(self, x):
        # 首先进行卷积操作提取特征
        x = self.conv1(x)
        x = self.conv2(x)
        # 其次展平
        x = self.flatten(x)
        # 最后分类
        x = self.classifier(x)
        return x