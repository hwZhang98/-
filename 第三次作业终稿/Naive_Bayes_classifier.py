import numpy as np

# https://blog.csdn.net/fuqiuai/article/details/79458943
class NaiveBayesClassifier:
    def __init__(self):
        self.feature_prob_True = None  # 特征值取1的条件概率矩阵
        self.feature_prob_Flase = None  # 特征值取0的条件概率矩阵
        self.class_prob = None

    def train(self, X, y, num_class=10):
        '''
        :param X: (N,D)维输入图像，N表示图像数，每个图像都是D*1的列向量
        :param y: (N,)维标签，数组的每个数取值为0...k-1
        :param num_class:类别个数
        :param verbose: 为False时不显示迭代过程
        :return:
        '''
        num_train, num_feature = X.shape[0], X.shape[1]  # 样本数
        self.class_prob = np.bincount(y, minlength=num_class) / num_train
        self.feature_prob_True = np.zeros((num_feature, num_class))  # (D,C)特征与类别的条件概率
        self.num_class = num_class
        # 每一类的概率P(C)，一维数组(num_class，1)
        for i in range(num_train):              #
             self.feature_prob_True[:,y[i]] += X[i].T
        self.feature_prob_True= (self.feature_prob_True+ 1) / (num_train + num_class)  # 拉普拉斯平滑
            # 它的思想非常简单，就是对先验概率的分子（划分的计数）加1，分母加上类别数；对条件概率分子加1，
            # 分母加上对应特征的可能取值数量。这样在解决零概率问题的同时，也保证了概率和依然为1。
        if i % 100 == 0:
            print('running')
        self.feature_prob_Flase = 1 - self.feature_prob_True

    # 预测
    def predict(self, X):
        y_pred = np.zeros((X.shape[0], self.num_class))  # 初始值为（N,C）
        for i in range(X.shape[0]):  # 对于每一个样本的一组特征值来说
            a = np.tile(X[i].reshape(-1, 1), (1, self.num_class))  # 将X[i]从（1,D）复制为（D,C）
            result = a * self.feature_prob_True  # 第一个特征值取1矩阵与对应概率相乘
            result += (-a + 1) * self.feature_prob_Flase  # 第二个特征值取0矩阵与对应概率相乘
            result = np.prod(result, axis=0) * self.class_prob
            y_pred[i] = result
        y_pred = np.argmax(y_pred, axis=1)
        return y_pred
