'''
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# 图片的读取：现在我们尝试读取一张示范图片：
IMG_URL = "lenna.jpg"   # 这个图片需要单独下载，或者你自己选用图片
img = Image.open(IMG_URL).convert('RGB')
#img.show()                     #此时应该弹窗显示图

plt.imshow(img)
plt.show()
#图像格式转换：图像的颜色转换可以使用 convert() 方法来实现。要读取一幅图像，
# 并将其转换成 灰度图像，只需要加上 convert('L'), 如下所示：
ary = np.array(img)[:, :, 0]
print(ary)
img_1 = img.convert('L')
#print('gray mode', img_1.mode)
#print(img_1.getpixel((0, 0)))
plt.imshow(img_1,cmap='gray')
plt.show()

#保存 PIL 图片：直接调用 Image 类的 save 方法。
#img.save('./assets/new_lenna.jpg')
# 在 assets 文件夹下应当出现新的名为 new_lenna.jpg 的图片

# 图片基础变换     https://pillow.readthedocs.io/en/latest/reference/Image.html
# PIL 图像可以进行多种基础操作，包括缩放大小、旋转角度等等，更多变换方式可以查阅文档。
plt.subplot(1,2,1)
plt.imshow(img.resize((200, 200)))  # 缩放图像大小
plt.subplot(1,2,2)
plt.imshow(img.rotate(90))          # 旋转图像角度
plt.show()
# 图像像素的矩阵表达
# 我们得到所使用 Lenna 图的分辨率，即长和宽的参数，并转换成对应的 Numpy 数组，
# 我们很容易发现 size 是原来长宽像素数量乘积的三倍。这是因为这里不仅考虑了长和宽，还考虑了 RGB 三个颜色通道。
# 通道(channel) 的概念在计算机视觉中经常遇到，我们在后续的课程中还会遇到它。

print(img.size)               # 图像的长和宽像素个数，即分辨率
img_np = np.asarray(img)      # 转化为 Numpy 数组
print(img_np.size)            # 两个 size 的含义不同
print(img_np.shape)           # 可以发现是 3 通道

plt.figure(figsize=(10,10))                  # 调整子图大小
plt.subplots_adjust(wspace =0.5, hspace =0)  # 调整子图间距

plt.subplot(1,3,1)
plt.imshow(img_np[:, :, 0])  # 红色 Red
plt.xlabel('red channel')

plt.subplot(1,3,2)
plt.imshow(img_np[:, :, 1])  # 绿色 Green
plt.xlabel('green channel')

plt.subplot(1,3,3)
plt.imshow(img_np[:, :, 2])  # 蓝色 Blue
plt.xlabel('blue channel')
plt.show()

img_l = Image.open(IMG_URL).convert('L')
plt.imshow(img_l)
print('PIL size: ' + str(img_l.size))
plt.show()
img_l_np = np.asarray(img_l)                           # 转换成 Numpy 数组
print('Numpy size: ' + str(img_l_np.size))             # 此时变成了单通道图片
print('Numpy shape: ' + str(img_l_np.shape))           # 下面这一行显示了颜色范围
print('Value range from ' + str(img_l_np.min()) + ' to ' + str(img_l_np.max()))

#我们先来看看一张直方图是如何使用 hist() 函数绘制的。

X = np.random.randn(10000)                         # 使用 random 类生成数目为 10000 的伪随机数
fig, ax = plt.subplots()                           # 创建一个绘制对象
ax.hist(X, bins=25, density=True, color = 'gray')  # 绘制灰度直方图
                                                   # 第一个参数是随机数序列，bins 指定直方的个数
                                                   # density 指定是否进行归一化，color指定直方图的颜色
x = np.linspace(-5, 5, 1000)
ax.plot(x, 1 / np.sqrt(2*np.pi) *                  # 随机数据服从正态分布
        np.exp(-(x**2)/2), linewidth=4)            # 因此画出正态分布曲线进行比对
plt.show()
#根据上面的矩阵进行绘制灰度直方图
fig, ax = plt.subplots()
pix = img_l_np.ravel()
ax.hist(pix, density=True, bins=256)   # ravel / flatten 将图像像素拉成一维数组
plt.show()                             # 横坐标是像素级范围 [0, 255]++
# 注意我们为了使用直方图进行表示，对二维图片中的像素进行了 flatten() 展平，拉升成了一维序列(或者说向量),
# 不难发现这一步中丢失了图像像素之间的空间信息。进行到这一步，你对直方图的使用情景有没有什么理解呢？关于将二维图片展平
# 的操作，我们以后还会遇到'''
#分割线————————————————————————————————————————————————

import cv2
import numpy as np
import matplotlib.pyplot as plt

IMG_URL = "lenna.jpg"
img = cv2.imread(IMG_URL)
print(img.size)
print(img.shape)             # OpenCV 的读图顺序是 BGR 顺序，需要转化为常见的 RGB 顺序：
img_rgb = img[:,:,::-1]      # 如果你理解切片操作，应该能理解这行代码
print(img_rgb.size)
print(img_rgb.shape)
plt.imshow(img_rgb)
plt.show()

# 三通道顺序是 Blue Green Red
# img_b = cv2.imread(IMG_URL, 0)
# img_g = cv2.imread(IMG_URL, 1)
# img_r = cv2.imread(IMG_URL, 2)

img_b, img_g, img_r = cv2.split(img)

plt.figure(figsize=(10,10))                  # 调整子图大小
plt.subplots_adjust(wspace =0.5, hspace =0)  # 调整子图间距

plt.subplot(1,4,1)
plt.imshow(img_b)

plt.subplot(1,4,2)
plt.imshow(img_g)

plt.subplot(1,4,3)
plt.imshow(img_r)

plt.subplot(1,4,4)
plt.imshow(cv2.merge([img_r, img_g, img_b]))  # 使用 merge() 还原
plt.show()
'''
hist = cv2.calcHist([img],  # 图片对象
                    [0],  # 使用的通道，这里使用灰度图计算直方图
                    None,  # 没有使用mask
                    [256],  # 表示这个直方图分成多少个 bins
                    [0.0, 255.0])  # 直方图柱总共表示的范围
print(hist.size)
print(hist.shape)
minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(hist)#maxVal用于归一化，找出矩阵中最大值和最小值即其对应的(x,y)的位置
print(maxVal)
print(hist[1])
'''
#由于OpenCV发展历史悠久，因此很多经典的处理方法都进行了封装，比如计算颜色直方图的calcHist()函数：
def calcAndDrawHist(image, color):
    hist = cv2.calcHist([image],  # 图片对象
                        [0],  # 使用的通道，这里使用灰度图计算直方图
                        None,  # 没有使用mask
                        [256],  # 表示这个直方图分成多少个 bins
                        [0.0, 255.0])  # 直方图柱总共表示的范围
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(hist)#maxVal用于归一化，找出矩阵中最大值和最小值即其对应的(x,y)的位置
    histImg = np.zeros([255, 255, 3], np.uint8)  # 绘制空的背景
    hpt = int(256*0.9)           # 下面的代码尝试自己理解/查文档  !这句不懂,是为了图像不顶格？好看？
    for h in range(256):
        intensity = int(hist[h] * hpt / maxVal)
        cv2.line(histImg, (h, 0), (h, intensity), color)    #画线函数，在histImg上画
    return histImg

img = cv2.imread(IMG_URL)
b, g, r = cv2.split(img)

histImgB = calcAndDrawHist(b, [255, 0, 0])  # 这里使用了驼峰命名法
histImgG = calcAndDrawHist(g, [0, 255, 0])
histImgR = calcAndDrawHist(r, [0, 0, 255])

plt.figure(figsize=(12,3))
plt.subplots_adjust(wspace =0.6, hspace =0)

plt.subplot(1,4,1)
plt.imshow(histImgB)

plt.subplot(1,4,2)
plt.imshow(histImgG)

plt.subplot(1,4,3)
plt.imshow(histImgR)

plt.subplot(1,4,4)
plt.imshow(img)
plt.show()

#参考 abid rahman 的做法，无需分离通道，用折线来描绘直方图的边界可在一副图中同时绘制三个通道的直方图。方法如下：
h = np.zeros((256, 256, 3))  # 创建用于绘制直方图的全 0 图像
bins = np.arange(256).reshape(256, 1)  # 直方图中各 bin 的顶点位置
color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # BGR三种颜色

for ch, col in enumerate(color):
    originHist = cv2.calcHist([img], [ch], None, [256], [0.0, 255.0])  # 计算对应通道的直方图
    cv2.normalize(originHist, originHist, 0, 255 * 0.9, cv2.NORM_MINMAX)  # 该函数将直方图的范围限定
    hist = np.int32(np.around(originHist))  # 四舍五入取整
    pts = np.column_stack((bins, hist))  # 叠加显示不同的子图
    cv2.polylines(h, [pts], False, col)  # 用线条的形式显示

h = np.flipud(h)  # 将矩阵里面的值倒序

plt.imshow(h)
plt.show()

# 颜色矩。图像中任何颜色分布均可以用它的矩来表示，矩也是图像处理中的常用概念。
print(r.shape)   # 在上一步中提取出的红色通道

# 一阶矩
ro_1a = r.mean()
print(ro_1a)

def mean(x=None):
    return np.sum(x) / np.size(x)

ro_1b = mean(r)
print(ro_1b)

# 二阶矩
ro_2a = r.std()
print(ro_2a)

def std(x=None):
    return np.sqrt(np.mean((x - x.mean()) ** 2))

ro_2b = std(r)
print(ro_2b)

# 三阶矩
def var(x=None):
    mid = np.mean(((x - x.mean()) ** 3))
    return np.sign(mid) * abs(mid) ** (1/3)

ro_3 = var(r)
print(ro_3)

# 简单的卷积(滤波) 运算

img = plt.imread(IMG_URL)   # 最后我要告诉你，matplotlib 本身就能读入图片，哈哈！

plt.subplot(1,2,1)
plt.imshow(img)

fil = np.array([[ -1, -1, -1],                        #这个是设置的滤波，也就是卷积核
                [ -1, 8, -1],
                [ -1, -1, -1]])

res = cv2.filter2D(img,-1,fil)                      #使用opencv的卷积函数

plt.subplot(1,2,2)
plt.imshow(res)                                     #显示卷积后的图片
plt.show()