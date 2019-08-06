import torch.optim as optim
import torchvision.transforms as T
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
import torch
import torchvision
import torch.utils.data
from torch.utils.data import sampler


class HyperParameter:  # 超参类
    def __init__(self, lr, minibatch, epochs, mode, show_every):
        self.Learningrate = lr  # 学习率
        self.Minibatch = minibatch  # 每批个数
        self.Epochs = epochs  # 迭代次数
        self.Mode = mode  # 模式 train or test
        self.show, self.show_every = show_every

    def __str__(self):
        assert self.Mode not in ['train', 'val'], 'input the right mode in "train" or "test"'
        return 'Learningrate：', self.Learningrate, '  Minibatche', self.Minibatche, '/n', \
               '  Epochs', self.Epochs, '  Mode', self.Mode


class Dataloader(torch.utils.data.DataLoader):  # 为DataLoader 写入一个函数，方便观察传入的数据
    def get_one(self, idex):
        data = self.data[idex]
        label = self.label[idex]
        return data, label


class Flatten(nn.Module):  # 展平操作
    def forward(self, x):
        return x.view(x.size(0), -1)


class CNN_model(nn.Module):
    def __init__(self):
        super(CNN_model, self).__init__()
        # model = models.alexnet(pretrained=True)
        # model.features[0] = nn.Conv2d(1, 64, kernel_size=(11,11), stride=(4, 4), padding=(2, 2))
        # model.classifier[6] = nn.Linear(4096,10)
        # self.model = nn.Sequential(
        #     model,
        # )
        # self.model = nn.Sequential(
        #     nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(inplace=True),
        #     nn.AvgPool2d(kernel_size=2, stride=2),
        #     nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(inplace=True),
        #     nn.AvgPool2d(kernel_size=2, stride=2),
        #     nn.Dropout(p=0.5),
        #     Flatten(),
        #     nn.Linear(56 * 56 * 16, 10, bias=True)
        # )
        model = models.resnet18(pretrained=True)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = nn.Linear(in_features=512, out_features=10, bias=True)
        self.model = nn.Sequential(
            model
        )


class Fit:
    def __init__(self, model=None, HP=None, train_data=None, val_data=None):
        self.model = model
        self.learningrate = HP.Learningrate
        self.epochs = HP.Epochs
        self.mode = HP.Mode
        self.put = HP.show
        self.put_every = HP.show_every
        self.train_data = train_data
        self.val_data = val_data
        self.optimizer = optim.Adam(cnnmodel.model.parameters(), lr=self.learningrate)

    def train(self):
        self.model.train()  # 进入训练模式
        for i in range(self.epochs):
            for j, (x, y) in enumerate(self.train_data):
                score = self.model(x)
                loss = F.cross_entropy(score, y.to(torch.long), reduction='sum')  # 计算损失  这里的LOSS可不是一个值
                self.optimizer.zero_grad()  # 梯度清零  不清零的话会增加内存占用
                loss.backward()  # 进行反向传播
                self.optimizer.step()  # 更新权重
                if j % self.put_every == 0:
                    print('OK')
            if self.put:  # 每迭代1次 输出一次当前的损失
                print('Iteration %d, loss = %.4f' % (i, loss.item()))
                self.predict(loader=self.train_data)  # 调用检查正确率的函数
                self.predict(mode='val', loader=self.val_data)  # 调用检查正确率的函数,验证集
                print()

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


# 定义超参数
HP = HyperParameter(lr=0.0001, minibatch=10, epochs=10, mode='train', show_every=(True, 5))

# 下面对数据进行导入和预处理
transform = T.Compose([  # 对每张图片的操作
    T.Resize(224),
    T.ToTensor()
])
train_data = torchvision.datasets.FashionMNIST(
    root=r'C:\Users\Administrator\Anaconda3\Lib\site-packages\torchvision\datasets',
    download=True, train=True, transform=transform)
# 地址前加r是因为\u对python来说是特殊字符，所以加一个r代表原生字符
test_data = torchvision.datasets.FashionMNIST(
    root=r'C:\Users\Administrator\Anaconda3\Lib\site-packages\torchvision\datasets',
    download=True, train=False, transform=transform)
# 对数据进行打包，分批
Loader_train = Dataloader(dataset=train_data, batch_size=HP.Minibatch, sampler=sampler.SubsetRandomSampler(range(500)))
Loader_val = Dataloader(dataset=train_data, batch_size=HP.Minibatch,
                        sampler=sampler.SubsetRandomSampler(range(500, 700)))
Loader_test = Dataloader(dataset=test_data, batch_size=HP.Minibatch, sampler=sampler.SubsetRandomSampler(range(500)))

# 训练模型
cnnmodel = CNN_model()
fit = Fit(model=cnnmodel.model, HP=HP, train_data=Loader_train, val_data=Loader_val)
fit.train()
fit.predict(mode='test', loader=Loader_test)
