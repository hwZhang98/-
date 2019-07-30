import numpy as np
import struct
import matplotlib.pyplot as plt
#处理MNIST数据集参考博客   https://blog.csdn.net/panrenlong/article/details/81736754
#训练集文件
train_image_idx3_ubyte_filename = 'train-images.idx3-ubyte'
#训练集标签文件
train_labels_idx1_ubyte_filename = 'train-labels.idx1-ubyte'

#测试集文件
test_images_idx3_ubyte_filename = 't10k-images.idx3-ubyte'
#测试集标签文件
test_labels_idx1_ubyte_filename = 't10k-labels.idx1-ubyte'

filename = [train_image_idx3_ubyte_filename,train_labels_idx1_ubyte_filename,
            test_images_idx3_ubyte_filename,test_labels_idx1_ubyte_filename]

def load_image(file_num):
    '''
    :param file_num: 想要解析的文件名
    :return: 返回解析过的矩阵
    下面是文件二进制格式时的解释
    TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  60000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).

     TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  60000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9.
    '''
    if file_num == 0 or file_num == 2:
        return decode_idx3_ubyte(filename[file_num])
        print('this is {}'.format(filename[file_num]))
    elif file_num == 1 or file_num == 3:
        return decode_idx1_ubyte(filename[file_num])
        print('this is {}'.format(filename[file_num]))
    else:
        print('input num is error')
        return None

def decode_idx3_ubyte(idx3_ubyte_file):
    """
    解析idx3 数据文件的通用函数
    :param idx3_ubyte_file: idx3 数据文件路径
    :return: 数据集
    """
    bin_data = open(idx3_ubyte_file,'rb').read()        #读取二进制数据
    # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽
    offset = 0
    fmt_header = '>iiii'
    '''
       因为数据结构中前4行的数据类型都是32位整型，所以采用i格式，但我们需要读取前4行数据，所以需要4个i。
       我们后面会看到标签集中，只使用2个ii。
       struct.unpack_from(fmt=,buffer=,offset=) 关于  fmt='>iiii'的问题 查看下面博客
       https://blog.csdn.net/kingfoulin/article/details/81311416
    '''
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张, 图片大小: %d*%d' % (magic_number, num_images, num_rows, num_cols))
    #解析数据集
    image_size = num_rows*num_cols
    offset += struct.calcsize(fmt_header)
    #获得数据在缓存中的指针位置，从前面介绍的数据结构可以看出，
    # 读取了前4行之后，指针位置（即偏移位置offset）指向0016。
    fmt_image = '>' + str(image_size) + 'B'
    # 图像数据像素值的类型为unsigned char型，对应的format格式为B。这里还有加上图像大小784，
    # 是为了读取784个B格式数据，如果没有则只会读取一个值（即一副图像中的一个像素值）
    images = np.empty((num_images, num_rows, num_cols))
    for i in range(num_images):
        if (i + 1) % 10000 == 0:  #每一万张输出一次当前指针位置
            print('已解析 %d' % (i + 1) + '张')
            print(offset)
        images[i] = np.array(struct.unpack_from(fmt_image,bin_data,offset)).reshape((num_rows, num_cols))
        #每次把一张图片元素的元组转换为矩阵存入images
        offset += struct.calcsize(fmt_image) #当前的图片指针位置前移
    print('data shape is '+str(images.shape))
    #plt.imshow(images[5000])
    #plt.show()
    return images

def decode_idx1_ubyte(idx1_ubyte_file):
    """
    解析idx1 标签文件的通用函数
    :param idx1_ubyte_file: idx1 标签文件路径
    :return: 数据集
    """
    # 读取二进制数据
    bin_data = open(idx1_ubyte_file, 'rb').read()
    # 解析文件头信息，依次为魔数和标签数
    offset = 0
    fmt_header = '>ii'   #标签头 只有两个int
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张' % (magic_number, num_images))
    # 解析标签集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print ('已解析 %d' % (i + 1) + '张')
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    print('data shape is '+str(labels.shape))
    #print(labels[5000])
    return labels

def get_data(train_num,val_num,test_num,F=False,new_axis=False,add_channel=False,float=False):
    # 导入训练集，并把它分为训练集与验证集
    '''
    每个样本格式为（N,28,28）单通道
    :param train_num: 训练集的个数
    :param val_num: 验证集的个数，验证集加测试集不能超过60000
    :param test_num: 测试集的个数，不能超过20000
    :param F: 是否把数据直接展平，变为（N，-1）
    :param new_axis:是否增加一个维度，比如3维变4维
    :param add_channel:是否增加通道
    :param float: 是否变为浮点类型
    :return:
    '''
    if float:
        images = load_image(0).astype(np.float)
        labels = load_image(1).astype(np.float)
        test_images = load_image(2).astype(np.float)[:test_num]
        test_labels = load_image(3).astype(np.float)[:test_num]
    else:
        images = load_image(0).astype
        labels = load_image(1).astype(np.int)  # 标签转化为整型
        test_images = load_image(2).astype[:test_num]
        test_labels = load_image(3).astype(np.int)[:test_num]
    train_images = images[:train_num]
    train_labels = labels[:train_num]
    print('train_images num',train_images.shape,'train_labels', train_labels.shape)
    val_images = images[train_num:val_num+train_num]
    val_labels = labels[train_num:val_num+train_num]
    print('val_images num',val_images.shape,'val_labels', val_labels.shape)

    print('test_images num',test_images.shape,'test_labels',test_labels.shape)
    # 数据集变形,展平
    if new_axis:
        train_images = train_images[:,np.newaxis,:,:]   # 增加一个维度，默认为第二个位置的维度
        val_images = val_images[:, np.newaxis, :, :]
        test_images = test_images[:, np.newaxis, :, :]
    if add_channel:                                 # 增加通道，把1通道复制为3通道
        train_images = np.concatenate((train_images,train_images,train_images),axis=1)
        val_images = np.concatenate((val_images,val_images,val_images),axis=1)
        test_images = np.concatenate((test_images,test_images,test_images),axis=1)
    if F:           # 展平数据
        train_images = train_images.reshape(train_images.shape[0], -1)
        val_images = val_images.reshape(val_images.shape[0], -1)
        test_images = test_images.reshape(test_images.shape[0], -1)

    return train_images,train_labels,val_images,val_labels,test_images,test_labels
