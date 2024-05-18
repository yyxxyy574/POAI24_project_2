import numpy as np
import os
import math
from tqdm import tqdm
import wandb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

import data

SAVE_DIR = "./results"

class Classifier:
    def __init__(self, save_name, load_from, model, device, augment, optimizer, lr=1e-3, momentum=1e-4, weight_decay=1e-5):
        self.save_name = save_name
        self.model = model
        if load_from is not None:
            self.model.load_state_dict(torch.load(f"{SAVE_DIR}/{load_from}.pt"))
        self.device = device
        self.augment = augment
        assert optimizer in {"SGD", "Adam"}, "optimizer type must be one of SGD and Adam"
        if optimizer == "SGD":
            self.optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        elif optimizer == "Adam":
            self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
    def train(self):
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 500, 1e-5, -1, verbose=False)
        criterion = nn.CrossEntropyLoss()
        num_epochs = 500
        highest_acc = 0
        train_dataloader = data.load("train", augment=self.augment)
        val_dataloader = data.load("val")
        
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
                
                self.optimizer.zero_grad()
                
                y_pred_prob = self.model(x)
                loss = criterion(y_pred_prob, y.long())
                loss.backward()
                losses.append(loss.detach().item())
                y_pred = y_pred_prob.max(1, keepdim=True)[1]
                num_correct += y_pred.eq(y.data.view_as(y_pred)).sum()
                self.optimizer.step()
                scheduler.step()
                
                train_data.set_description(f"Epoch {epoch + 1}/{num_epochs}")
                train_data.set_postfix({"loss": loss.detach().item()})
                
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
                    y_pred_prob = self.model(x)
                    loss = criterion(y_pred_prob, y.long())
                    y_pred = y_pred_prob.max(1, keepdim=True)[1]
                    num_correct += y_pred.eq(y.data.view_as(y_pred)).sum()  
            losses.append(loss.item())

            mean_loss = sum(losses) / len(losses)
            wandb.log({"val loss": mean_loss})
            acc = num_correct / len(val_dataloader.dataset)
            wandb.log({"val acc": acc})
            print(f"Val loss: {mean_loss:.4f}", "Val Accuracy: {}/{} ({:.0f}%)".format(num_correct, len(val_dataloader.dataset),100. * num_correct / len(val_dataloader.dataset)))
            
            # save best model
            if not os.path.exists(SAVE_DIR):
                os.makedirs(SAVE_DIR)
            if acc > highest_acc:
                highest_acc = acc
                torch.save(self.model.state_dict(), f"{SAVE_DIR}/{self.save_name}.pt")
                print(f"Updating best model at epoch {epoch}")
                
    def evluate(self):
        test_y = []
        test_pred = []
        test_dataloader = data.load("test")
        
        self.model.to(torch.device(self.device))
        self.model.eval()
        
        for x, y in tqdm(test_dataloader):
            x = x.to(torch.float32).to(torch.device(self.device))
            y = y.to(torch.float32).to(torch.device(self.device))
            
            with torch.no_grad():
                test_y += np.array(y.cpu()).astype(int).tolist()
                y_pred_prob = self.model(x)
                y_pred = y_pred_prob.max(1, keepdim=True)[1]
                test_pred += np.array(y_pred.reshape([-1]).cpu()).tolist()
        test_y = np.array(test_y)
        test_pred = np.array(test_pred)
        
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