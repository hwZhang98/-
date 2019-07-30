from data_process import get_data
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
import torch
import torchvision
import torch.utils.data
from torch.utils.data import sampler

transform = T.Compose([
    T.Resize(224),
    T.ToTensor()
])
train_data = torchvision.datasets.MNIST(root=r'C:\Users\Administrator\Anaconda3\Lib\site-packages\torchvision\datasets',download=True,train=True, transform=transform)
# 地址前加r是因为\u对python来说是特殊字符，所以加一个r代表原生字符
test_data = torchvision.datasets.MNIST(root=r'C:\Users\Administrator\Anaconda3\Lib\site-packages\torchvision\datasets',download=True,train=False, transform=transform)
# 每一个数据都是元组，第一个位置为图片，第二个位置为标签
Loader_train = torch.utils.data.DataLoader(train_data, batch_size=100,
                                           sampler=sampler.SubsetRandomSampler(range(10000)))
Loader_val = torch.utils.data.DataLoader(train_data, batch_size=100,
                                         sampler=sampler.SubsetRandomSampler(range(10000, 20000)))
Loader_test = torch.utils.data.DataLoader(test_data, batch_size=100,
                                          sampler=sampler.SubsetRandomSampler(range(5000)))
data_test_image,data_test_label = train_data.data,train_data.targets

# def big_image(input_tensor=None, out_size=None):
#     '''
#     对图片进行放缩，输入张量默认为3为（N,H,W），输出张量为三维（N,OUT_SIZE,OUT_SIZE）
#     默认只处理正方形图片
#     :param input_tensor: 输入张量
#     :param out_size: 输出尺寸
#     :return:
#     '''
#     N = input_tensor.size(0)
#     tem_1 = np.zeros((N, out_size, out_size))
#     tem_2 = torch.zeros((N, out_size, out_size))
#     for i in range(N):
#         tem_1[i] = cv2.resize(input_tensor[i].numpy(), (224, 224))
#         tem_2[i] = torch.tensor(tem_1[i])
#     return tem_2
#
#
def add_axisANDchannl(t_1):
    '''
    默认在第二个位置增加一个维度，比如（N,H,W）变为（N,1,H,W）
    再复制通道，变为（N,3,H,W）
    :param t_1:
    :param t_2:
    :param t_3:
    :return:
    '''
    #t_1 = torch.unsqueeze(t_1, 1)
    print(t_1.size())
    t_1 = t_1.repeat(t_1.size(0), 3, t_1.size(2), t_1.size(3))
    return t_1


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Alexnet_model(nn.Module):
    def __init__(self):
        super(Alexnet_model, self).__init__()
        model = models.alexnet(pretrained=True)
        #model = models.squeezenet1_0()
        model.features[0] = nn.Conv2d(1, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))
        model.classifier[6] = nn.Linear(4096,10,bias=True)
        self.model = nn.Sequential(
            model,
            # nn.Conv2d(3, 32, kernel_size=3, stride=1,padding=1),
            # nn.BatchNorm2d(32),
            # nn.ReLU(inplace=True),
            # nn.AvgPool2d(kernel_size=2, stride=2),
            # nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(16),
            # nn.ReLU(inplace=True),
            # nn.AvgPool2d(kernel_size=2, stride=2),
            # nn.Dropout(p=0.5),
            # Flatten(),
            # nn.Linear(7 * 7 * 16, 10, bias=True)

            # nn.Linear(1000, 300, bias=True),
            # nn.ReLU(inplace=True),
            # nn.Dropout(p=0.5),
            # nn.Linear(300, 50, bias=True),
        )

    def predict(self, mode='train', loader=None):
        if mode == 'test':
            print('this is test accuracy:')
        elif mode == 'val':
            print('this is val accuracy')
        else:
            print('this is train accuracy')
        self.model.eval()
        with torch.no_grad():  # 的过程在不构建运算图中进行
            num_correct = 0
            num_samples = 0
            for x, y in loader:
                score = self.model(x)  # 得出得分函数
                _, y_pred = torch.max(score, 1)  # 得分最大的一列预测输出 ,返回的为最大值和下标
                num_correct += (y_pred.float() == y.float()).sum()
                num_samples += y_pred.size(0)
            acc = float(num_correct) / num_samples
            print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))


def fit(model=None, optimizer=None, learning_rate=1e-3, epochs=10, train=Loader_train,
        val=Loader_val, look=True):
    model.train()  # 进入训练模式
    for i in range(epochs):
        j = 0
        for x, y in train:
            j += 1
            # x = add_axisANDchannl(x)
            score = alexnet.model(x)
            loss = F.cross_entropy(score, y.to(torch.long))  # 计算损失  这里的LOSS可不是一个值
            optimizer.zero_grad()  # 梯度清零  不清零的话会增加内存占用
            loss.backward()  # 进行反向传播
            optimizer.step()  # 更新权重
            if j % 5 == 0:
                print('OK')
        if look:  # 每迭代1次 输出一次当前的损失
            print('Iteration %d, loss = %.4f' % (i, loss.item()))
            alexnet.predict(loader=train)  # 调用检查正确率的函数
            alexnet.predict(mode='val', loader=val)  # 调用检查正确率的函数,验证集
            print()


# vgg.fit(x_train=train_images,y_train=train_labels,x_val=val_images,y_val=val_labels,learning_rate=3e-3,epochs=5)
# vgg.predict(mode='test',x=test_images,y=test_labels)

alexnet = Alexnet_model()
optimizer = optim.Adam(alexnet.parameters(), lr=5e-4)
fit(alexnet, optimizer, train=Loader_train, val=Loader_val)
alexnet.predict(mode='test', loader=Loader_test)
