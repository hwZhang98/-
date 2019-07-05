from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
plt.gray()                  #将imshow默认背景色调为gray
IMG_URL = "lenna.jpg"
img = Image.open(IMG_URL)
print('img mode'+str(img.mode))
img_np = np.array(img)
plt.subplot(1,2,1)
plt.imshow(img)
print('img size'+str(img.size))
print('img_np shape'+str(img_np.shape))
print('img_np size'+str(img_np.size))

img_1 = img.convert('L')
img_1_np = np.array(img_1)
print('img_1 mode'+str(img_1.mode))
plt.subplot(1,2,2)
plt.imshow(img_1)
print('img_1 size'+str(img_1.size))
print('img_1_np shape'+str(img_1_np.shape))
print('img_1_np size'+str(img_1_np.size))
print('Value range from ' + str(img_1_np.min()) + ' to ' + str(img_1_np.max()))
plt.show()
print(img_1_np)
#下面为分离通道测试------------------------------------------------------------------------分割线

img_r, img_g, img_b = img.convert('RGB').split()

plt.figure(figsize=(10,10))                    # 调整子图大小
plt.subplots_adjust(wspace =0.5, hspace =0.5)  # 调整子图间距

plt.subplot(2,3,1)
plt.imshow(img_r)  # 红色 Red
plt.xlabel('red channel')

plt.subplot(2,3,2)
plt.imshow(img_g)  # 绿色 Green
plt.xlabel('green channel')

plt.subplot(2,3,3)
plt.imshow(img_b)  # 蓝色 Blue
plt.xlabel('blue channel')


plt.subplot(2,3,4)
plt.imshow(img_np[:, :, 0])  # 红色 Red
plt.xlabel('red channel2')

plt.subplot(2,3,5)
plt.imshow(img_np[:, :, 1])  # 绿色 Green
plt.xlabel('green channel2')

plt.subplot(2,3,6)
plt.imshow(img_np[:, :, 2])  # 蓝色 Blue
plt.xlabel('blue channel2')

plt.show()