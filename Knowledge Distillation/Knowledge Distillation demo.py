import matplotlib.pyplot as plt
import numpy as np
import module
import math

# sequence = np.asarray([1, 2e-2, 3e-3, 4e-4, 5e-5])
# plt.figure()
# for T in range(1, 5):
#     sum_sequence = np.exp(sequence / T).sum()
#     softmax_sequence = [np.exp(sequence / T) / sum_sequence]
#     plt.scatter(sequence.reshape((1, 5)), softmax_sequence)
#     print(softmax_sequence)
#
# plt.show()

# 大概了解了softmax是如何通过参数T来控制软化的程度了
# 现在我需要构建就一个新的模型，前面的结构完全沿用AwA-SegNet,在最后一层添加一层
# 其作用是软化标签

# 先提取Teacher Label并保存起来
# TeacherNet = module.TeacherNet()
# for i in range(num_train):
#     train_data = train_dataset.__next__()[0]
#     train_data_name = train_lines[i].split(',')[0]
#     Teacher_Label = TeacherNet.predict(train_data, batch_size=1)
#     Teacher_Label = (Teacher_Label * 127.5 + 127.5)[:, :, :, 0].reshape((512, 512)).astype(np.uint8)
#     cv2.imwrite(r'C:\Users\liuye\Desktop\data\train\teacher_mask\{}.jpg'.format(train_data_name[:-4]),
#     Teacher_Label)
# for i in range(50):
#     train_data = test_dataset_label[0][i]
#     train_data_name = test_lines[i].split(',')[0]
#     Teacher_Label = TeacherNet.predict(train_data, batch_size=1)
#     Teacher_Label = (Teacher_Label * 127.5 + 127.5)[:, :, :, 0].reshape((512, 512)).astype(np.uint8)
#     cv2.imwrite(r'C:\Users\liuye\Desktop\data\val\teacher_mask\{}.jpg'.format(train_data_name[:-4]),
#     Teacher_Label)
# np.savez(r'C:\Users\liuye\Desktop\data\train\teacher_mask\{}.npz'.format(train_data_name),
#          Teacher_Label.reshape(512, 512, 2))
