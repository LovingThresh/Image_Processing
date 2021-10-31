import os

test_image_path = r'C:\Users\liuye\Desktop\data\val\img/'
test_label_path = r'C:\Users\liuye\Desktop\data\val\teacher_mask/'

test_image_list = os.listdir(test_image_path)
test_label_list = os.listdir(test_label_path)


with open('validation_HEYE_Teacher.txt', 'w') as f:
    for m, n in zip(test_image_list, test_label_list):
        text = m + ',' + n + '\n'
        f.write(text)

