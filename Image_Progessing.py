import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import random


def show(x):
    plt.imshow(x)
    plt.show()


# filepath = r'I:\Dataset\mask_data_POSITIVE/'
# filelist = os.listdir(filepath)
# for i in filelist:
#     path = filepath + i
#     src_images = cv2.imread(path)
#     image_black = src_images
#     image_white_mask = (image_black[:, :, 0] == 0).astype(np.uint8) * 255
#     image_white = image_black + np.expand_dims(image_white_mask, axis=2).repeat(3, axis=2)
#     cv2.imwrite('./src_images/{}.png'.format(i[:5]), image_white)


def Image_Processing(src_path, dst_path, num=80):
    src = cv2.imread(src_path)
    dst = cv2.imread(dst_path)
    a_mask = ((src[:, :, 0] != 255) | (src[:, :, 1] != 255) | (src[:, :, 2] != 255))\
        .astype(np.uint8)
    # src = ((src - src.min()) / (src.max() - src.min())).astype(np.uint8) * num
    mask = a_mask.astype(np.uint8) * 70
    src = src + np.expand_dims(mask, axis=2).repeat(3, axis=2)
    src_mask = np.zeros(src.shape, src.dtype)
    poly = np.array([[0, 0], [226, 0], [226, 226], [0, 226]])
    cv2.fillPoly(src_mask, [poly], (255, 255, 255))
    center = (113, 113)

    output1 = cv2.seamlessClone(src, dst, src_mask, center, cv2.MIXED_CLONE)
    if not os.path.exists(r'./Mix_img/{}'.format(num)):
        os.makedirs(r'./Mix_img/{}'.format(num))
        os.makedirs(r'./Mix_img/{}/image'.format(num))
        os.makedirs(r'./Mix_img/{}/label'.format(num))
    plt.imsave(r'./Mix_img/{}/image/{}_img.png'.format(num, src_path[-9:-4]), output1)
    plt.imsave(r'./Mix_img/{}/label/{}_img_mask.png'.format(num, src_path[-9:-4]), a_mask)


src_paths = r'src_images/'
dst_paths = r'I:\Dataset\road_data\Negative/'


def get_file_path(father_path):
    path_list = os.listdir(father_path)
    path_lists = []
    for i in path_list:
        path = father_path + i
        path_lists.append(path)

    return path_lists


# src_path_list = get_file_path(src_paths)
# dst_path_list = get_file_path(dst_paths)
#
# src_path_list_6000 = random.sample(src_path_list, 6000)
# dst_path_list_6000 = random.sample(dst_path_list, 6000)
#
# for s, d in zip(src_path_list_6000, dst_path_list_6000):
#     Image_Processing(s, d, 95)







