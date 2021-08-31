import numpy as np
import tensorflow as tf
import tf2lib as tl
from PIL import Image


def make_dataset(img_paths, batch_size, load_size, crop_size, training, drop_remainder=True, shuffle=True, repeat=1):
    if training:
        @tf.function
        def _map_fn(img):  # preprocessing
            img = tf.image.random_flip_left_right(img)
            img = tf.image.resize(img, [load_size, load_size])
            img = tf.image.random_crop(img, [crop_size, crop_size, tf.shape(img)[-1]])
            img = tf.clip_by_value(img, 0, 255) / 255.0  # or img = tl.minmax_norm(img)
            img = img * 2 - 1
            return img
    else:
        @tf.function
        def _map_fn(img):  # preprocessing
            img = tf.image.resize(img, [crop_size, crop_size])
            # or img = tf.image.resize(img, [load_size, load_size]); img = tl.center_crop(img, crop_size)
            img = tf.clip_by_value(img, 0, 255) / 255.0  # or img = tl.minmax_norm(img)
            img = img * 2 - 1
            return img

    return tl.disk_image_batch_dataset(img_paths,
                                       batch_size,
                                       drop_remainder=drop_remainder,
                                       map_fn=_map_fn,
                                       shuffle=shuffle,
                                       repeat=repeat)


def make_zip_dataset(A_img_paths, B_img_paths, batch_size, load_size, crop_size,
                     training, shuffle=False, repeat=False):
    # zip two datasets aligned by the longer one
    if repeat:
        A_repeat = B_repeat = None  # cycle both
    else:
        if len(A_img_paths) >= len(B_img_paths):
            A_repeat = 1
            B_repeat = None  # cycle the shorter one
        else:
            A_repeat = None  # cycle the shorter one
            B_repeat = 1

    A_dataset = make_dataset(A_img_paths, batch_size, load_size, crop_size,
                             training, drop_remainder=True, shuffle=shuffle, repeat=A_repeat)
    B_dataset = make_dataset(B_img_paths, batch_size, load_size, crop_size,
                             training, drop_remainder=True, shuffle=shuffle, repeat=B_repeat)

    A_B_dataset = tf.data.Dataset.zip((A_dataset, B_dataset))
    len_dataset = max(len(A_img_paths), len(B_img_paths)) // batch_size

    return A_B_dataset, len_dataset


class ItemPool:

    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        self.items = []

    def __call__(self, in_items):
        # `in_items` should be a batch tensor

        if self.pool_size == 0:
            return in_items

        out_items = []
        for in_item in in_items:
            if len(self.items) < self.pool_size:
                self.items.append(in_item)
                out_items.append(in_item)
            else:
                if np.random.rand() > 0.5:
                    idx = np.random.randint(0, len(self.items))
                    out_item, self.items[idx] = self.items[idx], in_item
                    out_items.append(out_item)
                else:
                    out_items.append(in_item)
        return tf.stack(out_items, axis=0)


def get_data(path=r'I:\Image Processing\train.txt', training=True):
    """
    获取样本和标签对应的行：获取训练集和验证集的数量
    :return: lines： 样本和标签的对应行： [num_train, num_val] 训练集和验证集数量
    """

    # 读取训练样本和样本对应关系的文件 lines -> [1.jpg;1.jpg\n', '10.jpg;10.png\n', ......]
    # .jpg:样本  ：  .jpg：标签

    with open(path, 'r') as f:
        lines = f.readlines()

    print(lines)

    # 打乱行， 打乱数据有利于训练
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)

    if training:
        # 切分训练样本， 90% 训练： 10% 验证
        num_val = int(len(lines) * 0.1)
        num_train = len(lines) - num_val
        return lines, num_train, num_val

    if not training:
        num_test = len(lines)
        return lines, num_test


def get_dataset_label(lines, batch_size, A_img_paths=r'I:\Image Processing\Rebuild_Image_95/',
                      B_img_paths=r'I:\Image Processing\Mix_img\95\label/', training=True):
    """
        生成器， 读取图片， 并对图片进行处理， 生成（样本，标签）
        :param training:
        :param B_img_paths:
        :param A_img_paths:
        :param lines: 样本和标签的对应行
        :param batch_size: 一次处理的图片数
        :return:  返回（样本， 标签）
        """

    numbers = len(lines)
    read_line = 0
    while True:

        x_train = []
        y_train = []

        # 一次获取batch——size大小的数据

        for t in range(batch_size):
            np.random.shuffle(lines)

        # 1. 获取训练文件的名字
        train_x_name = lines[read_line].split(',')[0]

        # 根据图片名字读取图片
        img = Image.open(A_img_paths + train_x_name)
        img = img.resize((227, 227))
        img_array = np.array(img)

        img_array = img_array / 255.0  # 标准化
        img_array = img_array * 2 - 1
        x_train.append(img_array)

        # 2. 获取训练样本标签的名字
        train_y_name = lines[read_line].split(',')[1].replace('\n', '')

        # 根据图片名字读取图片
        img = Image.open(B_img_paths + train_y_name)
        # img.show()
        # print(train_y_name)
        img = img.resize((227, 227))  # 改变图片大小 -> (227, 227)
        img_array = np.array(img)
        # img_array, 三个通道数相同， 没法做交叉熵， 所以下面要进行”图像分层“

        # 生成标签， 标签的shape是（227， 227， class_numbers) = (227, 227, 2), 里面的值全是0
        labels = np.zeros((227, 227, 2), np.int)

        # 下面将(224,224,3) => (224,224,2),不仅是通道数的变化，还有，
        # 原本背景和裂缝在一个通道里面，现在将斑马线和背景放在不同的通道里面。
        # 如，labels,第0通道放背景，是背景的位置，显示为1，其余位置显示为0
        # labels, 第1通道放斑马线，图上斑马线的位置，显示1，其余位置显示为0
        # 相当于合并的图层分层！！！！
        labels[:, :, 0] = (img_array[:, :, 1] == 1).astype(int).reshape((227, 227))
        labels[:, :, 1] = (img_array[:, :, 1] != 1).astype(int).reshape((227, 227))
        y_train.append(labels)

        # 遍历所有数据，记录现在所处的行， 读取完所有数据后，read_line=0,打乱重新开始
        read_line = (read_line + 1) % numbers

        yield np.array(x_train), np.array(y_train)


def get_test_dataset_label(lines, A_img_paths=r'I:\Image Processing\Rebuild_Image_95/',
                           B_img_paths=r'I:\Image Processing\Mix_img\95\label/'):
    numbers = len(lines)
    read_line = 0

    x_train = []
    y_train = []

    for read_line in range(numbers):
        train_x_name = lines[read_line].split(',')[0]

        # 根据图片名字读取图片
        img = Image.open(A_img_paths + train_x_name)
        img = img.resize((227, 227))
        img_array = np.array(img)

        img_array = img_array / 255.0  # 标准化
        img_array = img_array * 2 - 1
        x_train.append(img_array)

        # 2. 获取训练样本标签的名字
        train_y_name = lines[read_line].split(',')[1].replace('\n', '')

        # 根据图片名字读取图片
        img = Image.open(B_img_paths + train_y_name)
        # img.show()
        # print(train_y_name)
        img = img.resize((227, 227))  # 改变图片大小 -> (227, 227)
        img_array = np.array(img)
        # img_array, 三个通道数相同， 没法做交叉熵， 所以下面要进行”图像分层“

        # 生成标签， 标签的shape是（227， 227， class_numbers) = (227, 227, 2), 里面的值全是0
        labels = np.zeros((227, 227, 2), np.int)

        # 下面将(224,224,3) => (224,224,2),不仅是通道数的变化，还有，
        # 原本背景和裂缝在一个通道里面，现在将斑马线和背景放在不同的通道里面。
        # 如，labels,第0通道放背景，是背景的位置，显示为1，其余位置显示为0
        # labels, 第1通道放斑马线，图上斑马线的位置，显示1，其余位置显示为0
        # 相当于合并的图层分层！！！！
        labels[:, :, 0] = (img_array[:, :, 1] == 0).astype(int).reshape((227, 227))
        labels[:, :, 1] = (img_array[:, :, 1] != 0).astype(int).reshape((227, 227))

        y_train.append(labels)

    return np.array(x_train), np.array(y_train)



