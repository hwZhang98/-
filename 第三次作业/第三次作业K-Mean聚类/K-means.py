from data_process import load_image
import numpy as np
import matplotlib.pyplot as plt
import operator

# 导入训练集，并把它分为训练集与验证集
images = load_image(0)
labels = load_image(1).astype(np.int)  # 标签转化为整型
images_num = images.shape[0]
# 根据9比1的比例分为验证集和测试集
train_num = int(images_num * 0.9)
train_images = images[:train_num]
train_labels = labels[:train_num]
print(train_images.shape, train_labels.shape)
val_num = int(images_num * 0.1)
val_images = images[train_num:]
val_labels = labels[train_num:]
print(val_images.shape, val_labels.shape)
# plt.imshow(train_images[5000])
# plt.show()
# 导入测试集
test_images = load_image(2)
test_labels = load_image(3).astype(np.int)
test_num = test_images.shape[0]
print(test_images.shape, test_labels.shape)
# 数据集变形
train_images = train_images.reshape(train_num, -1)
val_images = val_images.reshape(val_num, -1)
test_images = test_images.reshape(test_num, -1)


# 距离函数
def distance(image1, image2):
    import math
    num = 0
    for i in range(image1.shape[0]):
        num += (image1[i] - image2[i]) ** 2
    return math.sqrt(num)


class K_Mean():
    def __init__(self):  # 初始化聚类中心
        self.center = np.zeros((10, 784))
        for i in range(10):
            self.center[i] = np.random.randint(0, 255, 784)

    def train(self, X):
        J = 0  # 清零代价函数
        self.center_old = self.center[:]
        while True:
            center_size = [0 for i in range(10)]  # 与聚类中心一个类的样本个数
            center_image = [[] for i in range(10)]  # 存储每个类的点
            for i in range(X.shape[0]):  # 遍历数据集，根据距离对样本进行分类
                min_way = []
                for j in range(self.center.shape[0]):  # 计算样本到簇0中心的距离（平方）
                    way = distance(X[i], self.center[j])  # 计算样本到簇2中心的距离（平方）
                    min_way.append(way)
                class_num = min_way.index(min(min_way))  # 距离第几类最近
                center_size[class_num] += 1
                center_image[class_num].append(X[i])
                if i % 1000 == 0:
                    print('我是I', i)
            # 所有图像都已分类，更新聚类中心
            for i in range(self.center.shape[0]):
                a = 0
                for j in center_image[i]:
                    a += j
                if len(center_image[i]) == 0:
                    self.center[i] = np.random.randint(0, 255, 784)
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


K = K_Mean()
K.train(val_images)  # 拿训练集练一练试一试
print("test 正确率%f" % np.mean(K.predict(test_images) == test_labels))
