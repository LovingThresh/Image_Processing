import os

test_image_path = r'./Test_Image/images/'
test_label_path = r'Test_Image/outputs/attachments/'

test_image_list = os.listdir(test_image_path)
test_label_list = os.listdir(test_label_path)


with open('text.txt', 'w') as f:
    for m, n in zip(test_image_list, test_label_list):
        text = m + ',' + n + '\n'
        f.write(text)

