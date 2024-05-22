# POAI24_project_2
```bash
git clone https://github.com/yyxxyy574/POAI24_project_2.git
```
## 数据准备
```bash
cd POAI24_project_2
wget https://cloud.tsinghua.edu.cn/seafhttp/files/8d8af47a-450f-41a6-bee2-bd16799df9b5/scene_classification.zip
unzip scene_classification.zip
mv scene_classification dataset
```
### 改变数据集
```bash
cd code
python
```
```python
import data
data.split_folder()
exit()
```
此时dataset文件夹下，会出现名为train、val、test的文件夹，每个文件夹下会按照类别进行划分。若要进行对于类别之间相互作用/数据均衡的实验，可以根据需求进行相应修改，以删除building类别为例：
```bash
cd ..
mkdir dataset_without_building
cp -rf dataset/train dataset_without_building/
rm -rf dataset_without_building/train/building
cp -rf dataset/val dataset_without_building/
rm -rf dataset_without_building/val/building
cp -rf dataset/test dataset_without_building/
rm -rf dataset_without_building/test/building
```
## main.py运行参数
`--train`：指定训练模式，在训练集上训练模型并保存在验证集上性能最好的模型

``--test``: 指定测试模式，在测试集上测试模型的性能并打印和保存相关指标

``--explain``: 指定解释模式，在给定的模型和图片上绘制热力图并保存

``--dataset``: 指定所用数据集，默认为"dataset"

``--save-name``: 指定结果保存时使用的文件名，结果包括性能最好的模型、测试指标、解释性热力图

`--load-from`: 指定加载模型的文件名，训练模式下可不指定，若指定了在该模型下继续训练，而测试和解释模型下必须指定

`--explain-mode`: 指定被解释图片位于train/val/test哪个划分下

`--image-name`: 指定被解释图片的类别和文件名，如"building/47.jpg"

`--model`: 指定所用的卷积神经网络模型，从basic/advance中选择，默认为basic

`-augment`: 指定进行数据增强

`--optimizer`: 指定训练时所用优化器，从SGD/Adam中选择，默认为SGD

`--learning-rate`: 指定训练时初始学习率，默认为1e-3

`--momentum`: 指定训练时momentum，默认为1e-4

`--weight-decay`: 指定训练时weight_decay，默认为1e-5

运行前需进入code文件夹下：
```bash
cd code
```
## 训练示例
若使用basic模型在原始数据集dataset上训练，并不进行数据增强，使用SGD优化器，则：
```bash
python main.py --train --dataset=dataset --save-name=basic --model=basic
```
则`../results/basic.pt`为训练过程中在验证集上性能最好的模型

若使用advance模型在去掉building类别的数据集dataset_without_building上训练，并进行数据增强，使用Adam优化器，则：
```bash
python main.py --train --dataset=dataset_without_building --save-name=advance_augment_Adam_without_building --model=advance --augment --optimizer=Adam
```
则`../results/advance.pt`为训练过程中在验证集上性能最好的模型
## 测试示例
若要测试训练好的basic模型，则：
```bash
python main.py --test --dataset=dataset --save-name=basic --load-from=basic --model=basic
```
则终端会打印accuracy、precision、f1-score、recall和混淆矩阵等指标测试结果，`../results/basic.png`为混淆矩阵的可视化结果
## 解释示例
若要解释训练好的basic模型在`../dataset/train/building/47.jpg`上的结果，则：
```bash
python main.py --explain --dataset=dataset --save-name=basic --load-from=basic --explain-mode=train --image-name=building/47.jpg --model=basic
```
则`../results/basic_0_gradcam.png`为绘制的热力图