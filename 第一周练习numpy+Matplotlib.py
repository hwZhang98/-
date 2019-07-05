#numpy 文献 https://docs.scipy.org/doc/numpy/reference/

import numpy as np
a = np.array([1, 2, 3])
print(type(a))
print(a.shape)                      # 输出a的形状
print(a[0], a[1], a[2])
a[0] = 5
print(a)

b = np.array([[1, 2, 3],
              [4, 5, 6]])
print(type(b))
print(b.shape)
print(b[0, 1], b[1, 1], b[1, 2])

a = np.zeros((2, 2))                #生成元素都为0的矩阵
print(a)

b = np.ones((1, 2))                 #生成元素都为1的矩阵
print(b)

c = np.full((3, 3), 6)              #生成常数矩阵
print(c)

d = np.eye(2)                       #生成单位矩阵
print(d)

e = np.random.random((2, 2))        #生成随机矩阵
print(e)

a = np.array([[1,2,3,4],
              [5,6,7,8],
              [9,10,11,12]])
b = a[:2, 1:3]
print(b)
# 数组的一个切片是指向相同数据的视图，因此如果修改它
# 也将修改原始数组
print(a[0, 1])   # 输出 "2"
b[0, 0] = 77    # b[0, 0] 和 a[0, 1] 位置的数据一致
print(a[0, 1])   # 输出 "77"

#整型数组访问：当我们使用切片语法访问数组时，得到的总是原数组的一个子集。
# 整型数组访问允许我们利用其它数组的数据构建一个新的数组：
a = np.array([[1,2], [3, 4], [5, 6]])
print(a[[0,1,2],[1,1,0]])

# 上面的整型数组索引示例与此等价:
print(np.array([a[0, 1], a[1, 1], a[2, 0]]))  # 输出 "[2 4 5]"

# 使用整型数组索引时，可以重用原数组中的相同元素:
print(a[[0, 0], [1, 1]])  # Prints "[2 2]"

# 等价于前面的整数数组索引示例
print(np.array([a[0, 1], a[0, 1]]))  # Prints "[2 2]"

# 创建一个新数组，从中选择元素
a = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])

print(a) # 输出   "array([[ 1,  2,  3],
         #                [ 4,  5,  6],
         #                [ 7,  8,  9],
         #                [10, 11, 12]])"

# 创建索引数组
b = np.array([0, 2, 0, 1])

# 使用 b 中的索引从 a 的每一行中选择一个元素
print(a[np.arange(4), b])  # 输出 "[ 1  6  7 11]"

# 使用 b 中的索引从 a 的每一行中修改一个元素
a[np.arange(4), b] += 10

print(a) # 输出    "array([[11,  2,  3],
         #                [ 4,  5, 16],
         #                [17,  8,  9],
         #                [10, 21, 12]])"

#布尔型数组访问：**布尔型数组访问可以让你选择数组中任意元素。通常，这种访问方式用于选取数组中满足某些条件的元素

a = np.array([[1,2], [3, 4], [5, 6]])
b = (a > 2)  # 找出 a 中大于 2 的元素;
                    # 这将返回布尔值表示的 Numpy 数组
                    # 形状和 a 相同, bool_idx 每个槽表示对应 a 的元素是否 > 2

print(b)     # 输出   "[[False False]
                    #          [ True  True]
                    #          [ True  True]]"

# 我们使用布尔数组索引来构造一个秩为 1 的数组
# 由对应于 bool_idx 的真值的 a 的元素组成
print(a[b])  # 输出   "[3 4 5 6]"

# 我们可以用一个简洁的声明做到同样的效果:
print(a[a > 2])     # 输出   "[3 4 5 6]"

#数据类型
x = np.array([1, 2])  # 让 Numpy 选择数据类型
print(x.dtype)        # 输出 "int64"

x = np.array([1.0, 2.0])   # 让 Numpy 选择数据类型
print(x.dtype)             # 输出 "float64"

x = np.array([1, 2], dtype=np.int64)   # 强制使用特定的数据类型
print(x.dtype)                         # 输出 "int64"

#数组计算

x = np.array([[1,2],[3,4]], dtype=np.float64)
y = np.array([[5,6],[7,8]], dtype=np.float64)

# 元素间求和；均会产生数组
# [[ 6.0  8.0]
#  [10.0 12.0]]
print(x + y)
print(np.add(x, y))

# 元素间求差；均会产生数组
# [[-4.0 -4.0]
#  [-4.0 -4.0]]
print(x - y)
print(np.subtract(x, y))

# 元素间乘法；均会产生数组
# 和 MATLAB 不同，* 是元素逐个相乘，而不是矩阵乘法。在 Numpy 中使用 dot 来进行矩阵乘法：
# [[ 5.0 12.0]
#  [21.0 32.0]]
print(x * y)
print(np.multiply(x, y))

# 元素间除法；均会产生数组
# [[ 0.2         0.33333333]
#  [ 0.42857143  0.5       ]]
print(x / y)
print(np.divide(x, y))

# 元素开根 ；产生数组
# [[ 1.          1.41421356]
#  [ 1.73205081  2.        ]]
print(np.sqrt(x))

# 矩阵相乘
x = np.array([[1,2],[3,4]])
y = np.array([[5,6],[7,8]])

v = np.array([9,10])
w = np.array([11, 12])

# 向量内积; 均会产生 219
print(v.dot(w))
print(np.dot(v, w))

# 矩阵 / 向量乘积; 均会产生秩为 1 的数组 [29 67]
print(x.dot(v))
print(np.dot(x, v))

# 矩阵 / 向量乘积; 均会产生秩为 2 的数组
# [[19 22]
#  [43 50]]
print(x.dot(y))
print(np.dot(x, y))

#sum函数     更多函数    https://docs.scipy.org/doc/numpy/reference/routines.math.html
x = np.array([[1,2],[3,4]])

print(np.sum(x))  # 计算所有元素的和; 输出 "10"
print(np.sum(x, axis=0))  # 计算每一列的和; 输出 "[4 6]"
print(np.sum(x, axis=1))  # 计算每一行的和; 输出 "[3 7]"

#转置矩阵    更多操作    https://docs.scipy.org/doc/numpy/reference/routines.array-manipulation.html
x = np.array([[1,2], [3,4]])
print(x)    # 输出  "[[1 2]
            #          [3 4]]"
print(x.T)  # 输出  "[[1 3]
            #          [2 4]]"

# 注意，对一个秩为 1 的数组进行转置没有任何作用:
v = np.array([1,2,3])
print(v)    # 输出   "[1 2 3]"
print(v.T)  # 输出   "[1 2 3]"

# Numpy ！！！广播机制！！！可以让我们不用创建 vv，就能直接运算，看看下面例子：             ！！！！！！！！！！！看不懂看链接
# 我们将向量 v 加到矩阵 x 的每一行
# 将结果存储在矩阵 y 中
x = np.array([[1,2,3], [4,5,6], [7,8,9], [10, 11, 12]])
v = np.array([1, 0, 1])
y = x + v  # 使用广播将 v 添加到 x 的每一行
print(y) # 输出 "[[ 2  2  4] ，        更多解释与例子  https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
         #          [ 5  5  7]，
         #          [ 8  8 10]，
         #          [11 11 13]]"
#更多Numpy知识看文献 https://docs.scipy.org/doc/numpy/reference/
#关于SciPy 基于 Numpy，提供了大量的计算和操作数组的函数，这些函数对于不同类型的科学和工程计算非常有用。
#熟悉 SciPy 的最好方法就是阅读文档。我们会强调对于本课程有用的部分。
#https://docs.scipy.org/doc/scipy/reference/index.html

# Matplotlib 快速入门
import matplotlib.pyplot as plt
x = np.arange(-3*np.pi, 3*np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)
print(x)
print(y)
plt.plot(x, y_sin)
plt.plot(x, y_cos)
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.title('Sine and Cosine')
plt.legend(['Sine', 'Cosine'])
plt.show()          #https://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot文档中阅读更多pyplot内容

#绘制多个图像
# 计算正弦和余弦曲线上点的 x 和 y 坐标
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

# 建立一个高度为 2，宽度为 1 的子图网格
# 并将第一个子图设为激活状态
plt.subplot(1, 2, 1)

# 绘制第一张图
plt.plot(x, y_sin)
plt.title('Sine')

# 将第二个子图设为激活状态，并绘制第二张图
plt.subplot(1, 2, 2)
plt.plot(x, y_cos)
plt.title('Cosine')

plt.show()
# 关于 subplot 的更多细节，可以阅读文档。https://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.subplot

#你可以使用 imshow 函数来显示图像，如下所示：
import imageio

img = imageio.imread('assetsshibe.jpg')
img_tinted = img * [1, 0.95, 0.9]

# 显示原图
plt.subplot(1, 2, 1)
plt.imshow(img)

# 显示着色图
plt.subplot(1, 2, 2)

# imshow 的一个小问题是，它可能会给出奇怪的结果
# 如果显示的数据不是 uint8 类型，我们在显示图像之前显式地将其转换为 uint8
plt.imshow(np.uint8(img_tinted))
plt.show()


