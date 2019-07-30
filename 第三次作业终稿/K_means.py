import numpy as np
import matplotlib.pyplot as plt
import operator


# 距离函数
def distance(image1, image2):
    num = 0
    for i in range(image1.shape[0]):
        num += (image1[i] - image2[i]) ** 2
    return num

class K_Mean():
    def __init__(self):  # 初始化聚类中心
        self.center_num = 20
        self.feat_num = 50
        self.center = np.zeros((self.center_num, self.feat_num))
        for i in range(self.center_num):
            self.center[i] = np.random.randint(-100, 100,self.feat_num)

    def train(self, X):
        self.center_old = self.center[:]
        while True:
            center_size = [0 for i in range(self.center_num)]  # 与聚类中心一个类的样本个数
            center_image = [[] for i in range(self.center_num)]  # 存储每个类的点
            for i in range(X.shape[0]):  # 遍历数据集，根据距离对样本进行分类
                min_way = []
                for j in range(self.center.shape[0]):  # 计算样本到簇中心的距离（平方）
                    way = distance(X[i].reshape(-1,1), self.center[j])  # 计算样本到簇中心的距离（平方）
                    min_way.append(way)
                class_num = min_way.index(min(min_way))  # 距离第几类最近
                center_size[class_num] += 1
                center_image[class_num].append(X[i])
                if i % 1000 == 0:
                    print('我是I', i)
            # 所有图像都已分类，更新聚类中心
            for i in range(self.center.shape[0]):
                a = 0
                # for j in center_image[i]:
                #     a += j
                self.center[i] = np.sum(np.array(center_image[i]),axis=0)/len(center_image[i])
                if len(center_image[i]) == 0:
                    self.center[i] = np.random.randint(-100, 100, 20)
                    break
                else:
                    self.center[i] = a / len(center_image[i])
                print('hahahah')
            if self.center.all() == self.center_old.all():
                print('xixixixi')
                break

    def predict(self, X):
        y_pred = np.zeros((X.shape[0], 1))
        for i in range(X.shape[0]):
            way = []
            for j in range(self.center.shape[0]):
                way.append(distance(X[i], self.center[j]))
            y_pred[i] = way.index(min(way))
        return y_pred


