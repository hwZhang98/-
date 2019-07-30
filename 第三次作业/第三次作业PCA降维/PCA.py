from data_process import load_image
import numpy as np
import matplotlib.pyplot as plt
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
#降维函数
def pca(X, feat_num):
    """
    data_mat：矩阵data_mat ，其中该矩阵中存储训练数据，每一行为一条训练数据
         保留前feat_num个特征
    return：降维后的数据集和原始数据被重构后的矩阵（即降维后反变换回矩阵）
    """
    num_data, dim = X.shape  # 获取数据条数和每条的维数
    mean_vals = X.mean(axis=0)  # (784,)  求出每列特征均值
    mean_removed = X - mean_vals  # 数据中心化，即指变量减去它的均值
    cov_mat = np.cov(mean_removed, rowvar=0)  #(784, 784) # 计算协方差矩阵
    eig_vals, eig_vects = np.linalg.eig(np.mat(cov_mat))  # 计算特征值和特征向量，shape分别为（784，）和(784, 784)
    eig_val_index = np.argsort(eig_vals)  # 对特征值进行从小到大排序，argsort返回的是索引
    eig_val_index = eig_val_index[:-(feat_num + 1): -1]  # 最大的前feat_num个特征的索引
    # 取前feat_num个特征后重构的特征向量矩阵reorganize eig vects,
    # shape为(784, feat_num)，feat_num最大为特征总数
    reg_eig_vects = eig_vects[:, eig_val_index]
    # 将数据转到新空间
    print('mean_remove',mean_removed.shape,type(mean_removed),'reg_eig_vect',reg_eig_vects.shape,type(reg_eig_vects))
    low_d_data_mat = mean_removed * reg_eig_vects  # shape:(X.shape[0], feat_num)
    recon_mat = (low_d_data_mat * reg_eig_vects.T) + mean_vals  # 根据前几个特征向量重构回去的矩阵，shape:(X.shape[0], 784)
    return low_d_data_mat, recon_mat

feat_X ,X = pca(train_images,100)

plt.subplot(2,1,1)
plt.ylabel("old")
plt.imshow(X[1].reshape(28,28))
plt.subplot(2,1,2)
plt.ylabel('PCA')
plt.imshow(feat_X[1].reshape(10,10))
plt.show()