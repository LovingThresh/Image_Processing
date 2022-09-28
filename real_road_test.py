# -*- coding: utf-8 -*-
# @Time    : 2022/9/5 10:48
# @Author  : LiuYe
# @Email   : csu1704liuye@163.com | sy2113205@buaa.edu.cn
# @File    : real_road_test.py
# @Software: PyCharm
import numpy as np
import cv2

cap = cv2.VideoCapture(r'C:\Users\liuye\Desktop\phone.mp4')

i = 1
while cap.isOpened():
    ret, frame = cap.read()
    cv2.imwrite('test_images/{}.png'.format(i), frame)
    i = i + 1
    k = cv2.waitKey(20)
    if k & 0xff == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

