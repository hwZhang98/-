from data_process import load_image
import numpy as np
import matplotlib.pyplot as plt
import cv2
from Naive_Bayes_classifier import NaiveBayesClassifier

# 导入训练集，并把它分为训练集与验证集
images = load_image(0)
labels = load_image(1).astype(np.int)  # 标签转化为整型
images_num = images.shape[0]
# 根据9比1的比例分为验证集和测试集
train_num = int(images_num * 0.9)
train_images = images[:train_num]
train_labels = labels[:train_num]
print(train_images.shape, train_labels.shape)
val_images = images[train_num:]
val_labels = labels[train_num:]
print(val_images.shape, val_labels.shape)
# plt.imshow(train_images[5000])
# plt.show()
# 导入测试集
test_images = load_image(2)
test_labels = load_image(3).astype(np.int)
test_num = test_images.shape[0]
plt.imshow(train_images[200])
plt.show()
# 数据集变形
train_images = train_images.reshape(train_images.shape[0], -1)
val_images = val_images.reshape(val_images.shape[0], -1)
test_images = test_images.reshape(test_images.shape[0], -1)
print(train_images[1].shape)
print(test_labels)
print(test_labels.shape,'shapeaaaaaaa')
# print(train_images.shape,'hahahahah')
# print(train_labels.shape,'sdadasdasda')
# 经过分析发现图像的特征跟像素点位置比较有关，而跟像素值关系不大，所以将数据转化为2值化
def binaryzation(data):
    xixi,next_data = cv2.threshold(data,1,1,cv2.THRESH_BINARY)
    return next_data
train_images = binaryzation(train_images)
val_images = binaryzation(val_images)
test_images = binaryzation(test_images)
#经过处理后，所以的数据的像素点变为0和1两种
bayes = NaiveBayesClassifier()
bayes.train(train_images,train_labels,num_class=10)
print('val 正确率为%f'%np.mean(bayes.predict(val_images)==val_labels))
print('test 正确率为%f'%np.mean(bayes.predict(test_images)==test_labels))