import tensorflow as tf
import argparse
import numpy as np
from PIL import Image
import module
from keras import backend as K
import seaborn as sns
import matplotlib.pyplot as plt
import datetime

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='95')
parser.add_argument('--datasets_dir', default='Mix_img')
parser.add_argument('--load_size', type=int, default=227)
parser.add_argument('--crop_size', type=int, default=227)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument("--mode", default='client')
parser.add_argument("--port", default=52162)
args = parser.parse_args()

A_img_paths = r'I:\Image Processing\Rebuild_Image_95/'
B_img_paths = r'I:\Image Processing\Mix_img\95\label/'


# a = os.listdir(A_img_paths)
# b = os.listdir(B_img_paths)
# with open('train.txt', 'w') as f:
#     for i, m in zip(a, b):
#         c = i + ',' + m + '\n'
#         f.writelines(c)

def get_data():
    """
    获取样本和标签对应的行：获取训练集和验证集的数量
    :return: lines： 样本和标签的对应行： [num_train, num_val] 训练集和验证集数量
    """

    # 读取训练样本和样本对应关系的文件 lines -> [1.jpg;1.jpg\n', '10.jpg;10.png\n', ......]
    # .jpg:样本  ：  .jpg：标签

    with open(r'I:\Image Processing\train.txt', 'r') as f:
        lines = f.readlines()

    print(lines)

    # 打乱行， 打乱数据有利于训练
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)

    # 切分训练样本， 90% 训练： 10% 验证
    num_val = int(len(lines) * 0.1)
    num_train = len(lines) - num_val

    return lines, num_train, num_val


def get_dataset_label(lines, batch_size):
    """
        生成器， 读取图片， 并对图片进行处理， 生成（样本，标签）
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
        img = img.resize((args.crop_size, args.crop_size))
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
        img = img.resize((args.crop_size, args.crop_size))  # 改变图片大小 -> (224, 224)
        img_array = np.array(img)
        # img_array, 三个通道数相同， 没法做交叉熵， 所以下面要进行”图像分层“

        # 生成标签， 标签的shape是（224， 224， class_numbers) = (224, 224, 2), 里面的值全是0
        labels = np.zeros((args.crop_size, args.crop_size, 2), np.int)

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


lines, num_train, num_val = get_data()
batch_size = 10
train_dataset = get_dataset_label(lines[:num_train], batch_size)
validation_dataset = get_dataset_label(lines[num_train:], batch_size)

model = module.ResnetGenerator()

optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.0005)


def Precision(y_true, y_pred):
    """精确率"""
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  # true positives
    pp = K.sum(K.round(K.clip(y_pred, 0, 1)))  # predicted positives
    precision = tp / (pp + K.epsilon())
    return precision


def Recall(y_true, y_pred):
    """召回率"""
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  # true positives
    pp = K.sum(K.round(K.clip(y_true, 0, 1)))  # possible positives
    recall = tp / (pp + K.epsilon())
    return recall


def F1(y_true, y_pred):
    """F1-score"""
    precision = Precision(y_true, y_pred)
    recall = Recall(y_true, y_pred)
    f1 = 2 * ((precision * recall) / (precision + recall + K.epsilon()))
    return f1


def IOU(y_true, y_pred):
    predict = K.round(K.clip(y_pred, 0, 1))
    Intersection = K.sum(y_true * predict)
    Union = K.sum(y_true + predict)
    iou = Intersection / (Union - Intersection)
    return iou


# CallBack
a = str(datetime.datetime.now())
b = list(a)
b[10] = '-'
b[13] = '-'
b[16] = '-'
c = ''.join(b)

tensorboard = tf.keras.callbacks.TensorBoard(log_dir='./output/{}/tensorboard/'.format(c))
checkpoint = tf.keras.callbacks.ModelCheckpoint('./output/{}/checkpoint/'.format(c) + 'ep{epoch:03d}-val_loss{'
                                                                                      'val_loss:.3f}-val_acc{'
                                                                                      'val_accuracy:.3f}.h5',
                                                monitor='val_accuracy', verbose=0,
                                                save_best_only=False, save_weights_only=False, mode='auto', period=1)


model.compile(optimizer=optimizer,
              loss='binary_crossentropy',
              metrics=['accuracy', Precision, Recall, F1, IOU])
model.fit(train_dataset,
          steps_per_epoch=max(1, num_train // batch_size),
          epochs=50,
          validation_data=validation_dataset,
          validation_steps=max(1, num_val // batch_size),
          initial_epoch=0,
          callbacks=[tensorboard, checkpoint])


def plot_heatmap(predict_array):
    if predict_array.ndim == 4:
        sns.heatmap(predict_array[:, :, :, 0].reshape(227, 227))
    else:
        sns.heatmap(predict_array.reshape(227, 227))
    plt.show()
