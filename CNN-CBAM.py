# 导入必要的库
import shutil

import numpy as np
import torch
import torchvision
import torchvision.models as models_ResN50
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
import torchvision.transforms.functional as F
from torchvision.datasets import folder
from torchvision.utils import draw_bounding_boxes
from torch.nn import BatchNorm2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os


import time
# 定义一个空列表
epoch_times = []

# 导入VGG模型
from torchvision.models import vgg16, VGG16_Weights
from prettytable import PrettyTable

def count_parameters (model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters ():
        if not parameter.requires_grad: continue
        params = parameter.numel ()
        table.add_row ( [name, params])
        total_params+=params
    print (table)
    print ("Total Trainable Params: {}".format(total_params))
    return total_params

# 定义超参
num_classes = 10  # 类别数
num_epochs = 100  # 训练轮数
batch_size = 16  # 批次大小
learning_rate = 0.01  # 学习率
device = torch.device("mps")
# 定义数据转换
transform = transforms.Compose(
    [torchvision.transforms.RandomCrop(224, padding=4),  # 随机裁剪
     transforms.RandomHorizontalFlip(p=0.5), #随机水平翻转
     transforms.RandomRotation(degrees=(0, 90), expand=False, center=None), #随机旋转角度0～90度
     transforms.ToTensor(), #将图片化为张量
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 加载训练集和测试集
trainset = torchvision.datasets.ImageFolder(root='.//dataSet//train', transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.ImageFolder(root='.//dataSet//test', transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=0)
train_data_size = len(trainset)
test_data_size = len(testset)

# 获取数据集中的所有类名
classes = testset.classes
flag = 0
# 定义通道注意力层的类
class ChannelAttention(torch.nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)  # 全局平均池化层，输出大小为1*1
        self.fc1 = torch.nn.Linear(in_channels, in_channels // 16)  # 全连接层，输入维度为通道数，输出维度为通道数除以16
        self.relu1 = torch.nn.ReLU()  # 激活函数
        self.fc2 = torch.nn.Linear(in_channels // 16, in_channels)  # 全连接层，输入维度为通道数除以16，输出维度为通道数

    def forward(self, x):
        out = self.avg_pool(x)  # 对输入数据进行全局平均池化，得到每个通道的平均值
        out = out.view(out.size(0), -1)  # 展平成一维向量，大小为批次大小*通道数
        out = self.fc1(out)  # 全连接层1 -> 激活函数
        out = self.relu1(out)
        out = self.fc2(out)  # 全连接层2 -> 输出
        out = out.view(out.size(0), out.size(1), 1, 1)  # 调整形状，大小为批次大小*通道数*1*1
        out = torch.sigmoid(out)  # 激活函数，得到每个通道的注意力权重，范围为[0,1]
        return out
# 定义空间注意力层的类
class SpatialAttention(torch.nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = torch.nn.Conv2d(2, 1, kernel_size=7, padding=3)  # 卷积层，输入通道为2，输出通道为1，卷积核大小为7

    def forward(self, x):
        avg_out = torch.mean(x, dim=1).unsqueeze(1)  # 对输入数据在通道维度上求平均值，得到平均特征图，大小为批次大小*1*高*宽
        max_out = torch.max(x, dim=1)[0].unsqueeze(1)  # 对输入数据在通道维度上求最大值，得到最大特征图，大小为批次大小*1*高*宽
        out = torch.cat([avg_out, max_out], dim=1)  # 将平均特征图和最大特征图在通道维度上拼接起来，大小为批次大小*2*高*宽
        out = self.conv1(out)  # 卷积层 -> 输出，大小为批次大小*1*高*宽
        out = torch.sigmoid(out)  # 激活函数，得到每个位置的注意力权重，范围为[0,1]
        return out
# 定义组合注意力模块的类
class CBAM(torch.nn.Module):
    def __init__(self, in_channels):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels)
        self.spatial_attention = SpatialAttention()
    #组合注意力
    def forward(self, x):
        x = self.channel_attention(x) * x  # 通道注意力乘以输入数据
        x = self.spatial_attention(x) * x  # 空间注意力乘以输入数据
        return x

# 定义卷积神经网络模型
class MYCNN(torch.nn.Module):
    def __init__(self):
        super(MYCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, 1, 1)  # 输入通道为3，输出通道为16，卷积核大小为3
        self.bn1 = torch.nn.BatchNorm2d(16)  # 批量归一化层
        self.conv1_2 = torch.nn.Conv2d(16, 16, 3, 1, 1)  # 输入通道为16，输出通道为16，卷积核大小为3
        self.bn1_2 = torch.nn.BatchNorm2d(16)  # 批量归一化层
        self.cbam_1 = CBAM(16) #第一个组合注意力模块，输入通道为16
        self.conv2 = torch.nn.Conv2d(16, 32, 3, 1, 1)  # 输入通道为16，输出通道为32，卷积核大小为3
        self.bn2 = torch.nn.BatchNorm2d(32)  # 批量归一化层
        self.conv2_2 = torch.nn.Conv2d(32,32,3,1,1)  # 输入通道为32，输出通道为32，卷积核大小为3
        self.bn2_2 = torch.nn.BatchNorm2d(32)  # 批量归一化层
        self.cbam_2 = CBAM(32) #第二个组合注意力模块，输入通道为32
        self.pool = torch.nn.MaxPool2d(2)  # 最大池化层，池化核大小为2，步长为2
        self.dropout = torch.nn.Dropout(p=0.5) #dropout 层随机关闭一些神经元
        self.fc1 = torch.nn.Linear(32 * 14 * 14, 128)  # 全连接层，输入维度为32*14*14，输出维度为128
        self.fc2 = torch.nn.Linear(128, num_classes)  # 全连接层，输入维度为128，输出维度为类别数

    def forward(self, x):
        # 卷积层1 -> 批量归一化 -> 激活函数 -> 池化层
        x = self.pool(torch.nn.functional.relu(self.bn1(self.conv1(x))))
        # 卷积层1_2 -> 批量归一化 -> 激活函数 -> 池化层
        x = self.pool(torch.nn.functional.relu(self.bn1_2(self.conv1_2(x))))
        # 组合注意力模块乘以输入数据
        x = self.cbam_1(x) * x
        # 卷积层2 -> 批量归一化 -> 激活函数 -> 池化层≥
        x = self.pool(torch.nn.functional.relu(self.bn2(self.conv2(x))))
        # 卷积层2_2 -> 批量归一化 -> 激活函数 -> 池化层
        x = self.pool(torch.nn.functional.relu(self.bn2_2(self.conv2_2(x))))
        x = self.cbam_2(x) * x  # 组合注意力模块乘以输入数据
        x = x.view(-1, 32 * 14 * 14)  # 展平成一维向量k,
        x = self.dropout(x)
        x = self.fc1(x)  # 全连接层1 -> 激活函数
        x = self.fc2(x)  # 全连接层2 -> 输出
        return x

# 添加tensorboard
writer = SummaryWriter("./logs_train_index")
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0

def copy_images(imagePaths, folder, preds):
    # 检查目标文件夹是否存在，如果不存在就创建它
    if not os.path.exists(folder):
        os.makedirs(folder)
    # 遍历图片路径和预测标签
    for path, pred in zip(imagePaths, preds):
        # 从路径中获取图片名称，并创建一个占位符对应于预测标签文件夹
        imageName = path.split(os.path.sep)[-1]
        labelFolder = os.path.join(folder, str(pred))
        # 检查预测标签文件夹是否存在，如果不存在就创建它
        if not os.path.exists(labelFolder):
            os.makedirs(labelFolder)
        # 复制图片到预测标签文件夹中
        shutil.copy(path, labelFolder)


# 创建模型实例
if __name__ == '__main__':
    model = MYCNN() #自建立模型
    #model = vgg16(weights=None) #vgg16模型
    #model = torchvision.models.mobilenet_v2(weights = None)
    model = model.to(device)
    #count_parameters(model)
    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失函数
    criterion = criterion.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  # 随机梯度下降优化器

    # 训练模型
    for i in range(num_epochs):  # 遍历每个训练轮数
        start_time = time.time() # 获取当前时间戳
        print("///第{}轮训练开始///".format(i + 1))
        model.train()
        fun_loss = 0 # 每个epoch初始化损失值
        for i, data in enumerate(trainloader, 0):  # 遍历每个批次的数据
            inputs, labels = data  # 获取输入和标签
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()  # 清零梯度
            outputs = model(inputs)  # 前向传播得到输出
            loss = criterion(outputs, labels)  # 计算损失
            loss.backward()  # 反向传播计算梯度
            optimizer.step()  # 更新参数
            total_train_step = total_train_step + 1
            fun_loss = fun_loss + loss.item()  # 累加损失

            if total_train_step % 100 == 0:
                print("训练次数:{},Loss:{}".format(total_train_step, loss.item()))
                writer.add_scalar("train_loss", loss.item(), total_train_step)
                writer.add_images('inputs', inputs[0], total_train_step, dataformats='CHW')
                out_img = outputs.view(-1, 1, num_classes)
                out_pic = torchvision.utils.make_grid(outputs)
                writer.add_images('outputs', out_pic, total_train_step, dataformats='CHW')
                fun_loss = 0.0
        # 指定学习率下降
        print("当前的学习率为:{}".format(learning_rate))
        if i <= 20:
            if i % 5 == 0:
                learning_rate = learning_rate*0.9
        if 20 < i <= 100:
            if i % 4 == 0:
                learning_rate = learning_rate * 0.9
        if 100 < i <= 200:
            if i % 2 == 0:
                learning_rate = learning_rate * 0.9
            '''
            # 获取当前时间戳
            end_time = time.time()
            # 计算运行时间
            epoch_time = end_time - start_time
            # 添加到列表中
            epoch_times.append(epoch_time)
            # 打印运行时间
            print(epoch_times)
            '''
        # 测试模型
        model.eval()
        total_test_loss = 0
        correct = 0  # 记录正确预测的数量
        with torch.no_grad():  # 不计算梯度
            total_accuracy = 0
            flag = flag+1
            for data in testloader:  # 遍历每个批次的数据
                imgs, targets = data
                imgs = imgs.to(device)
                targets = targets.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, targets)  # 定义loss变量
                total_test_loss = total_test_loss + loss
                accuracy = (outputs.argmax(1) == targets).sum()
                total_accuracy = total_accuracy + accuracy
                if flag == 4:
                    # 显示图片和标签
                    label_img = imgs.squeeze()
                    label_img = label_img[0]
                    label_img = F.to_pil_image(label_img.cpu())
                    plt.imshow(label_img, cmap="gray")
                    max_values, max_indices = torch.max(outputs, 1)  # 按行求最大值和索引
                    plt.title(classes[max_indices[0].item()])  # 取第一个样本的最大索引作为标签
                    plt.show()

        print("整体测试集上的Loss:{}".format(total_test_loss))
        print("整体测试上的正确率:{}".format(total_accuracy / test_data_size))
        writer.add_scalar("test_loss", loss.item(), total_test_step)
        writer.add_scalar("test_accuracy", total_accuracy / test_data_size, total_test_step)
        writer.add_graph(model, imgs)
        # 记录每个epoch运行时间
        #writer.add_scalar("epoch_times", epoch_time, total_test_step)
        # 记录每个epoch的学习率
        writer.add_scalar("learning_rate", learning_rate, total_test_step)
        total_test_step = total_test_step + 1
        torch.save(model.state_dict(), "model_{}.pth".format(i))
        # print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))  # 打印测试集准确率
    writer.close()