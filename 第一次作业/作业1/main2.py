from data_process import load_image
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
'''
首先对原始数据进行加工，把二进制转化为矩阵形式
接着调用sklearn里的各种分类器进行训练和测试
特征选取
1.图像像素
对应数据集为：
train_imgaes,train_labels,train_num   训练集
val_imgaes,val_labels,val_num   验证集
test_imgaes,test_labels,test_num   测试集
2.灰度直方图
对应数据集为：
b1,train_labels,train_num   训练集
b2,val_labels,val_num   验证集
b3,test_labels,test_num   测试集
将对应数据集代入下方各分类器即可观察输出
把对应位置的train改为b1即可，其他类似
'''
#导入训练集，并把它分为训练集与验证集
images = load_image(0)
labels = load_image(1).astype(np.int)  #标签转化为整型
images_num = images.shape[0]
#根据9比1的比例分为验证集和测试集
train_num = int(images_num*0.9)
train_images = images[:train_num]
train_labels = labels[:train_num]
print(train_images.shape,train_labels.shape)
val_num = int(images_num*0.1)
val_images = images[train_num:]
val_labels = labels[train_num:]
print(val_images.shape,val_labels.shape)
# plt.imshow(train_images[5000])
# plt.show()
#导入测试集
test_images = load_image(2)
test_labels = load_image(3).astype(np.int)
test_num = test_images.shape[0]
print(test_images.shape,test_labels.shape)

#数据集变形
train_images = train_images.reshape(train_num,-1)
# print(train_images.shape,'hahahahah')
# print(train_labels.shape,'sdadasdasda')
val_images = val_images.reshape(val_num,-1)
test_images = test_images.reshape(test_num,-1)

#上面为对数据进行处理的部分



#以下部分为使用灰度直方图作为特征进行学习========================

b1 = np.zeros((train_num, 256))   #对图片进行处理，统计每个像素值的个数，以个数为特征，输入样本有256个特征
for i in range(train_num):      #输入数据集为（N,256）     b1 此为训练集
    a = np.bincount(train_images[i].astype(int), minlength=256).reshape(1,-1)
    b1[i] = a
b2 = np.zeros((test_num, 256))  #b2测试集
for i in range(test_num):
    a = np.bincount(test_images[i].astype(int),minlength=256).reshape(1,-1)
    b2[i] = a
b3 = np.zeros((val_num, 256))   #b3验证集
for i in range(val_num):
    a = np.bincount(test_images[i].astype(int),minlength=256).reshape(1, -1)
    b3[i] = a


#决策树
tree = DecisionTreeClassifier()
tree.fit(b1,train_labels)                   #训练模型
value = tree.score(b2,test_labels)          #输出正确率
print(value)

#支持向量机
svm = SVC(gamma='scale',decision_function_shape='ovo')
svm.fit(b1,train_labels)
value = svm.score(b2,test_labels)
print(value)
#逻辑回归

logis = LogisticRegression(solver='saga',multi_class='multinomial')
logis.fit(b1,train_labels)
value = logis.score(b2,test_labels)
print(value)

