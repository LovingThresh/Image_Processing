import os
import cv2
import matplotlib.pyplot as plt
test_image_path = r'C:\Users\liuye\Desktop\data\val\img/'
test_label_path = r'C:\Users\liuye\Desktop\data\val\teacher_mask/'

test_image_list = os.listdir(test_image_path)
test_label_list = os.listdir(test_label_path)


with open('validation_HEYE_Teacher.txt', 'w') as f:
    for m, n in zip(test_image_list, test_label_list):
        text = m + ',' + n + '\n'
        f.write(text)


# 针对800*600的标签图片进行扩张
img = cv2.imread(r'L:\ALASegmentationNets\Data\Stage_2\train\mask\6192.png')
plt.imshow(img)
plt.show()
dilation = cv2.dilate(img, kernel=(5, 5), iterations=5)
plt.imshow(dilation)
plt.show()
