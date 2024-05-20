# 改进后的模型

import torch.nn as nn

class Advance(nn.Module):
    def __init__(self):
        super(Advance, self).__init__()
        # 第一个卷积层
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        # 第二个卷积层
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        # 第三个卷积层
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        # 第四个卷积层
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        # 第五个卷积层
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2)
        )
        # 展平层
        self.flatten = nn.Flatten()
        # 分类层
        self.classifier = nn.Sequential(
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.ReLU(True),
            # 最后输出维度为类别数6
            nn.Linear(128, 6)
        )

    def forward(self, x):
        # 首先进行卷积操作提取特征
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        # 其次展平
        x = self.flatten(x)
        # 最后分类
        x = self.classifier(x)
        return x