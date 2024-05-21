# 进行训练和测试的分类器类

import numpy as np
import os
from tqdm import tqdm
import wandb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

import data
from model import basic, advance

# 保存模型及评价指标结果的文件夹路径
SAVE_DIR = "../results"

class Classifier:
    def __init__(self, dataset, save_name, load_from, model, device, augment, optimizer, lr=1e-3, momentum=1e-4, weight_decay=1e-5):
        # 使用的数据集
        self.dataset = dataset
        # 保存所用的文件名
        self.save_name = save_name
        # 使用的卷积神经网络模型
        self.model = model
        # 若指定了导入已有模型，则导入相应参数
        if load_from is not None:
            self.model.load_state_dict(torch.load(f"{SAVE_DIR}/{load_from}.pt"))
        self.device = device
        # 训练时是否进行数据增强
        self.augment = augment
        # 使用的优化器
        assert optimizer in {"SGD", "Adam"}, "optimizer type must be one of SGD and Adam"
        if optimizer == "SGD":
            self.optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        elif optimizer == "Adam":
            self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
    def train(self):
        # 采用cos学习率变化器
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 500, 1e-5, -1, verbose=False)
        # 采用交叉熵作为损失函数
        criterion = nn.CrossEntropyLoss()
        # 训练轮数
        num_epochs = 500
        # 验证集上最高的正确率，用于保存性能最好的模型
        highest_acc = 0
        # 加载训练集和验证集
        train_dataloader = data.load(self.dataset, "train", augment=self.augment)
        val_dataloader = data.load(self.dataset, "val")
        
        self.model.to(torch.device(self.device))
        
        for epoch in range(num_epochs):
            
            # train
            self.model.train()
            losses = []
            num_correct = 0
            train_data = tqdm(train_dataloader, position=0)
            for x, y in train_data:
                x = x.to(torch.float32).to(torch.device(self.device))
                y = y.to(torch.float32).to(torch.device(self.device))
                
                # 清空优化器梯度
                self.optimizer.zero_grad()
                
                # 运用当前模型进行预测
                y_pred_prob = self.model(x)
                # 根据预测结果和样本真实值计算loss并反向传播
                loss = criterion(y_pred_prob, y.long())
                loss.backward()
                losses.append(loss.detach().item())
                # 得到预测类别，并与样本真实值进行对比判断是否预测正确
                y_pred = y_pred_prob.max(1, keepdim=True)[1]
                num_correct += y_pred.eq(y.data.view_as(y_pred)).sum()
                # 优化器迭代
                self.optimizer.step()
                # 学习率变化器迭代
                scheduler.step()
                
                # 利用tqdm在终端输出训练进度和loss变化情况
                train_data.set_description(f"Epoch {epoch + 1}/{num_epochs}")
                train_data.set_postfix({"loss": loss.detach().item()})
            
            # 利用wandb记录该轮训练集上平均loss和正确率，同时在终端输出
            mean_loss = sum(losses) / len(losses)
            wandb.log({"train loss": mean_loss})
            acc = num_correct / len(train_dataloader.dataset)
            wandb.log({"train acc": acc})
            print(f"Epoch [{epoch + 1}/{num_epochs}]:")
            print(f"Train loss: {mean_loss:.4f}", "Train Accuracy: {}/{} ({:.0f}%)".format(num_correct, len(train_dataloader.dataset),100. * num_correct / len(train_dataloader.dataset)))
            
            # valid
            self.model.eval()
            losses = []
            num_correct = 0
            for x, y in val_dataloader:
                x = x.to(torch.float32).to(torch.device(self.device))
                y = y.to(torch.float32).to(torch.device(self.device))
                
                with torch.no_grad():
                    # 运用当前模型进行预测
                    y_pred_prob = self.model(x)
                    # 根据预测结果和样本真实值计算loss
                    loss = criterion(y_pred_prob, y.long())
                    # 得到预测类别，并与样本真实值进行对比判断是否预测正确
                    y_pred = y_pred_prob.max(1, keepdim=True)[1]
                    num_correct += y_pred.eq(y.data.view_as(y_pred)).sum()  
            losses.append(loss.item())

            # 利用wandb记录该轮验证集上平均loss和正确率，同时在终端输出
            mean_loss = sum(losses) / len(losses)
            wandb.log({"val loss": mean_loss})
            acc = num_correct / len(val_dataloader.dataset)
            wandb.log({"val acc": acc})
            print(f"Val loss: {mean_loss:.4f}", "Val Accuracy: {}/{} ({:.0f}%)".format(num_correct, len(val_dataloader.dataset),100. * num_correct / len(val_dataloader.dataset)))
            
            # save best model
            if not os.path.exists(SAVE_DIR):
                os.makedirs(SAVE_DIR)
            if acc > highest_acc:
                # 若该轮验证集上的正确率高于已记录的最高正确率，则更新最高正确率并保存当前模型的参数
                highest_acc = acc
                torch.save(self.model.state_dict(), f"{SAVE_DIR}/{self.save_name}.pt")
                print(f"Updating best model at epoch {epoch}")
                
    def evluate(self):
        test_y = []
        test_pred = []
        test_dataloader = data.load(self.dataset, "test")
        
        self.model.to(torch.device(self.device))
        self.model.eval()
        
        for x, y in tqdm(test_dataloader):
            x = x.to(torch.float32).to(torch.device(self.device))
            y = y.to(torch.float32).to(torch.device(self.device))
            
            with torch.no_grad():
                # 记录样本真实值
                test_y += np.array(y.cpu()).astype(int).tolist()
                # 运用当前模型进行预测
                y_pred_prob = self.model(x)
                # 得到预测类别
                y_pred = y_pred_prob.max(1, keepdim=True)[1]
                # 记录预测值
                test_pred += np.array(y_pred.reshape([-1]).cpu()).tolist()
        test_y = np.array(test_y)
        test_pred = np.array(test_pred)
        
        # 根据测试集上的样本真实值和预测值计算常见指标accuracy、precision、recall、f1和各类别详细数据report，并在终端输出
        accuracy = accuracy_score(test_y, test_pred)
        precision = precision_score(test_y, test_pred, average='macro')
        recall = recall_score(test_y, test_pred, average='macro')
        f1 = f1_score(test_y, test_pred, average='macro')
        report = classification_report(test_y, test_pred)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print(f"Detailed Report:")
        print(report)
        
        # 计算混淆矩阵，在终端输出，并进行可视化
        conf_matrix = confusion_matrix(test_y, test_pred)
        print("Confusion Matrix:")
        print(conf_matrix)
        plt.figure(figsize=(10, 8))
        plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(f'Confusion Matrix of {self.save_name}')
        plt.colorbar()
        tick_marks = np.arange(6)
        plt.xticks(tick_marks, tick_marks)
        plt.yticks(tick_marks, tick_marks)
        plt.ylabel('True number')
        plt.xlabel('Predicted number')
        thresh = conf_matrix.max() / 2.
        for i, j in np.ndindex(conf_matrix.shape):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                    horizontalalignment='center',
                    color='white' if conf_matrix[i, j] > thresh else 'black')
        plt.tight_layout()
        plt.savefig(f"{SAVE_DIR}/{self.save_name}.png")
        
    def explain(self, mode, img_name):
        target_layer = None
        if isinstance(self.model, basic.Basic):
            target_layer = [self.model.conv2[-1]]
        elif isinstance(self.model, advance.Advance):
            target_layer = [self.model.conv5[-1]]
        else:
            print("model is not correct")
            return
        input_tensor, input_array = data.load_image(self.dataset, mode, img_name)
        input_tensor = input_tensor.unsqueeze(dim=0)

        cam = GradCAM(model=self.model, target_layers=target_layer)
            
        img_idx = data.CLASS_TO_IDX[self.dataset][img_name.split("/")[0]]
        img_name = img_name.split("/")[1].split(".")[0]
        target = [ClassifierOutputTarget(img_idx)]
        
        grayscale_cam = cam(input_tensor=input_tensor, targets=target)
        
        grayscale_cam = grayscale_cam[0, :]
        visualization = Image.fromarray(show_cam_on_image(input_array, grayscale_cam, use_rgb=True))
        visualization.save(f"../results/{self.save_name}_{img_name}_gradcam.png")