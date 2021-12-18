import os

import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def plot_heatmap(save_path, predict_array, size=(512, 512)):
    if predict_array.ndim == 4:
        sns.heatmap(predict_array[:, :, :, 0].reshape((predict_array.shape[1], predict_array.shape[2])),
                    xticklabels=False, yticklabels=False)
    else:
        sns.heatmap(predict_array[:, :, 0].reshape((predict_array.shape[0], predict_array.shape[1])),
                    xticklabels=False, yticklabels=False)
    plt.savefig(save_path)


# 此函数是为了裁剪师姐的数据而创建的，目前已经裁剪完成，但是由于裁剪中心的设置，有一些图片需要进一步调整
def crop_image(image_path, size=227):
    # 这里已经预设为 227
    # 前两个坐标点是左上角坐标
    # 后两个坐标点是右下角坐标
    # width在前， height在后
    image = Image.open(image_path)
    width, height = image.size
    box_box = [(0, 0, size, size), (0, height - size, size, height),
               (width - size, 0, width, size), (width - size, height - size, width, height)]
    i = 1
    for box in box_box:
        region = image.crop(box)
        region.save(r'C:\Users\liuye\Desktop\data\crop_data\{}_crop_{}.jpg'.format(image_path[-7:-4], i))
        i += 1


# 此函数是为了将原始的细长裂缝的裂缝图像227×227填充至512×512，然后调用分割效果极好的模型进行分割
def pad_img(img, pad_size=(512, 512), values=255):
    new_image = np.pad(img, ((512 - img.shape[0], 0), (512 - img.shape[1], 0), (0, 0)), 'constant',
                       constant_values=values)

    return new_image

# path = r'C:\Users\liuye\Desktop\Slender Cracks Positive/'
# for i in os.listdir(path):
#     file_path = path + i
#     img = cv2.imread(file_path)
#     new_img = pad_img(img)
#     plt.imsave(r'C:\Users\liuye\Desktop\Machine_Background\Pad Crack Image\{}.jpg'.format(i[:5]), new_img)
