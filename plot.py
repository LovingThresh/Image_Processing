import cv2
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image


def plot_heatmap(predict_array, size=(227, 227)):
    if predict_array.ndim == 4:
        sns.heatmap(predict_array[:, :, :, 0].reshape(size))
    else:
        sns.heatmap(predict_array.reshape(size))
    plt.show()


# 此函数是为了裁剪世界的数据而创建的，目前已经裁剪完成，但是由于裁剪中心的设置，有一些图片需要进一步调整
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
