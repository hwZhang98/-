from data_process import load_image
import numpy as np
from linear_classifier import LinearSVM

'''
data_process包是对数据的处理，将其转换为矩阵
svm是线性分类器函数，有向量实现形式和另一种形式（忘了怎么称呼了）
linear_classifier是构造一个分类器类

首先对原始数据进行加工，把二进制转化为矩阵形式
接着调用svm分类器进行训练和测试
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
将对应数据集代入下方分类器即可观察输出
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
reg = 1e-4
w=np.random.randn(784, 10)*0.0001 #图片格式为（28，28），所以转化为28*28=784
#以下部分为使用灰度直方图作为特征进行学习=====================================

b1 = np.zeros((train_num, 256))
for i in range(train_num):
    a = np.bincount(train_images[i].astype(int), minlength=256).reshape(1,-1)
    b1[i] = a
b2 = np.zeros((test_num, 256))
for i in range(test_num):
    a = np.bincount(test_images[i].astype(int),minlength=256).reshape(1,-1)
    b2[i] = a
b3 = np.zeros((val_num, 256))
for i in range(val_num):
    a = np.bincount(test_images[i].astype(int),minlength=256).reshape(1, -1)
    b3[i] = a
#=================================================================================
# loss,grad=svm.svm_loss_naive(w,train_images,train_labels,reg)
# print(loss, grad)
svm = LinearSVM()    #创建分类器对象，此时W为空
loss_hist = svm.train(train_images, train_labels, learning_rate = 1e-7, reg = 2.5e4, num_iters = 1500, verbose = True)    #此时svm对象中有W

y_train_pred = svm.predict(train_images)
print('training accuracy: %f'%(np.mean(train_labels==y_train_pred)))
# y_val_pred = svm.predict(val_images)
# print('validation accuracy: %f'%(np.mean(val_labels==y_val_pred)))

# 超参数调优（交叉验证）
learning_rates = [1.4e-7, 1.5e-7, 1.6e-7]
# for循环的简化写法12个
regularization_strengths = [(1 + i * 0.1) * 1e4 for i in range(-3, 3)] + [(2 + i * 0.1) * 1e4 for i in range(-3, 3)]
results = {}  # 字典
best_val = -1
best_svm = None
for learning in learning_rates:  # 循环3次
    for regularization in regularization_strengths:  # 循环6次
        svm = LinearSVM()
        svm.train(train_images, train_labels, learning_rate=learning, reg=regularization, num_iters=2000)  # 训练
        y_train_pred = svm.predict(train_images)  # 预测（训练集）
        train_accuracy = np.mean(train_labels == y_train_pred)
        print('training accuracy: %f' % train_accuracy)
        y_val_pred = svm.predict(val_images)  # 预测（验证集）
        val_accuracy = np.mean(val_labels == y_val_pred)
        print('validation accuracy: %f' % val_accuracy)
        if val_accuracy > best_val:
            best_val = val_accuracy
            best_svm = svm
        results[(learning, regularization)] = (train_accuracy, val_accuracy)
for learn, reg in sorted(results):
    train_accuracy, val_accuracy = results[(learn, reg)]
    print('lr %e reg %e train accuracy: %f val accuracy: %f ' % (learn, reg, train_accuracy, val_accuracy))
print('best validation accuracy achieved during cross-validation: %f' % best_val)

#在测试集上验证
y_test_pred = best_svm.predict(test_images)
test_accuracy = np.mean(test_labels == y_test_pred)
print('svm on raw pixels final test set accuracy: %f' % (test_accuracy, ))

