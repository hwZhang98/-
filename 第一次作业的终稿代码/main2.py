import numpy as np
import matplotlib.pyplot as plt
def colicTest():            #对马这个数据进行处理
    '''
    :return: 列表形式
    '''
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    testSet = []
    testLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    for line in frTest.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        testSet.append(lineArr)
        testLabels.append(float(currLine[21]))

    return trainingSet,trainingLabels,testSet,testLabels
x_train,y_train,x_test,y_test = colicTest()
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
# 归一化  会降低精度
'''
a = np.mean(x_train,axis=0)
b = np.std(x_train,axis=0)
x_train = (x_train - a) / b

a = np.mean(x_test, axis=0)
b = np.std(x_test, axis=0)
x_test = (x_test - a) / b
'''
#下面是分类器用numpy实现
class  logistic_classifier():
    def __init__(self):
        pass

    def loss(self,W,X,y,reg,learning_rate):
        '''

        :param W:权重 为（D,1）矩阵
        :param X: 输入数据为（N,D）   N为样本数量，D为特征数目
        :param y: 输入数据标签为（N,1）
        :param reg: 正则系数
        :param learn_rate: 学习率
        :return: loss 与 dW（D,1）矩阵
        '''
        num = X.shape[0]
        Z = X.dot(W.T)                            #求出权重与输入矩阵乘积的加权和
        y_predect = 1/(1+np.exp(-Z))
        loss = 1/num*np.sum(-y*np.log(y_predect)-(1-y)*np.log(1-y_predect))
        v = np.zeros((1,W.shape[1]))
        p = 0.9
        v = v * p + (1-p)*(y-y_predect.T).dot(X)
        W = W + learning_rate * v
        loss += 0.5 * reg * np.sum(W * W)
        # W = W + learning_rate *(y-y_predect.T).dot(X)
        return loss,W

    def train(self, X, y, X_val, y_val, Learning_rate=1e-3, learning_rate_decay=0.95, reg=1e-4, num_iters=1000,
              batch_size=10, verbose=False):
        '''
        :param X:训练集数据
        :param y: 训练集标签
        :param X_val: 验证集数据
        :param y_val: 验证集标签
        :param learning_rate: 学习率
        :param learning_rate_decay: 学习率变化系数
        :param reg: 正则系数
        :param num_iters: 设置的总迭代次数
        :param batch_size: 随机梯度每次迭代选取一批个数
        :param verbose: 每次迭代是否显示当前数据
        :return:返回的列表中存储了损失，训练集平均正确率，验证集平均正确率
        每次先定义一次迭代准备选取多少样本进行更新参数，然后在理论遍历完一遍样本后
        记录当前各项数据
        '''
        num_train = X.shape[0]
        np.c_[np.ones((X.shape[0],1)),X]
        iterations_per_epoch = max(num_train / batch_size, 1)  # 多少次迭代就可以完成一次理论上遍历所有样本
        loss_history = []
        train_acc_history = []
        val_acc_history = []
        W = np.random.randn(1,X.shape[1])
        for it in range(num_iters):
            X_batch = None
            Y_batch = None
            idx = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X[idx]
            y_batch = y[idx]                    #用随机梯度的话LOSS会有锯齿状
            loss, W = self.loss(W, X=X_batch, y=y_batch, reg=reg, learning_rate=Learning_rate)
            loss_history.append(loss)
            # 参数更新
            if verbose and it % 100 == 0:  # 参数每更新100次，打印
                print('iteration %d / %d : loss %f ' % (it, num_iters, loss))

            if it % iterations_per_epoch == 0:  # 理论上选取完所有数据一轮结束
                train_acc = (self.predict(X_batch,W) == y_batch).mean()
                val_acc = (self.predict(X_val,W) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)
                Learning_rate *= learning_rate_decay  # 更新学习率
        return {
             'loss_history': loss_history,
             'train_acc_history': train_acc_history,
             'val_acc_history': val_acc_history
        },W

    def predict(self, X, W):
        '''
        预测正确率，根据权重与输入数据
        :param X:
        :param W:
        :return:
        '''
        y_pred = None
        Z = X.dot(W.T)  # 求出权重与输入矩阵乘积的加权和
        y_pred = 1 / (1 + np.exp(-Z))
        for i in range(y_pred.shape[0]):        #因为类别只有0，1，所以通过函数后要对他进行判断分类
            if y_pred[i] >= 0.5:
                y_pred[i] = 1
            else:
                y_pred[i] = 0
        return y_pred

best = logistic_classifier()
stats,W = best.train(x_train,y_train,x_test,y_test,num_iters=3000,batch_size=150,
                              Learning_rate=2.3e-4,learning_rate_decay=0.98,reg=0.5,verbose=True)
val_acc = (best.predict(x_test,W) == y_test).mean()
print(val_acc,'!!!!!!!!!!!!!!!!!!!!!!!')
plt.figure(figsize=(12,6))
plt.plot(stats['loss_history'])
plt.show()

#多参数调优，寻找最优参数
hidden_size = [30,50,100,150,200] #每批数量
#hidden_size = [250] #每批数量
results = {}
best_val_acc = 0
best_net = None
learning_rates = np.array([2.2e-4,2.3e-4,2.35e-4,2.25e-4,2.45e-4,2.4e-4,2.5e-4]) #学习率
regularization_strengths = [0.4,0.3,0.45,0.55,0.65,0.75,0.8]    #正则系数
print('running')
for hs in hidden_size:
    for lr in learning_rates:
        for reg in regularization_strengths:
            logis = logistic_classifier()#下面返回的stats里面存的是train的每一批迭代的正确率
            stats,W = logis.train(x_train,y_train,x_test,y_test,num_iters=3000,batch_size=hs,
                              Learning_rate=lr,learning_rate_decay=0.98,reg=reg,verbose=True)
            val_acc = (logis.predict(x_test,W) == y_test).mean()
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_net = logis
            results[(hs,lr,reg)] = val_acc
print('finshed')

for hs,lr,reg in sorted(results):
    val_acc = results[(hs,lr,reg)]
    if val_acc == best_val_acc:
        print('follow is best parameters!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!1')
    print('hs %d lr %e reg %e val accuracy: %f'%(hs,lr,reg,val_acc))

print("best validation accuracy achieved during cross_validationg:%f"% best_val_acc)

