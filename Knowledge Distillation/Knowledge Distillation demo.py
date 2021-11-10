# import matplotlib.pyplot as plt
# import numpy as np
# import module
# import math

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
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt
import keras.backend as K
import skimage
import cv2
from PIL import Image

K.set_learning_phase(0)


engine_name = 'model.plan'
onnx_path = 'model_1.onnx'
batch_size = 1

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER)


def build_engine(onnx_path, shape=(1, 512, 512, 3)):
    """
   This is the function to create the TensorRT engine
   Args:
      onnx_path : Path to onnx_file.
      shape : Shape of the input of the ONNX file.
  """
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1) \
            as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        config = builder.create_builder_config()
        config.max_workspace_size = (256 << 20)
        with open(onnx_path, 'rb') as model:
            parser.parse(model.read())
        network.get_input(0).shape = shape
        engine = builder.build_engine(network, config)
        return engine


engine = build_engine(onnx_path)


def save_engine(engine, file_name):
    buf = engine.serialize()
    with open(file_name, 'wb') as f:
        f.write(buf)


save_engine(engine, engine_name)


# def load_engine(trt_runtime, plan_path):
#     with open(engine_path, 'rb') as f:
#         engine_data = f.read()
#     engine = trt_runtime.deserialize_cuda_engine(engine_data)
#     return engine

def allocate_buffers(engine, batch_size, data_type):
    """
    This is the function to allocate buffers for input and output in the device
    Args:
       engine : The path to the TensorRT engine.
       batch_size : The batch size for execution time.
       data_type: The type of the data for input and output, for example trt.float32.

    Output:
       h_input_1: Input in the host.
       d_input_1: Input in the device.
       h_output_1: Output in the host.
       d_output_1: Output in the device.
       stream: CUDA stream.

    """

    # Determine dimensions and create page-locked memory buffers (which won't be swapped to disk) to hold host
    # inputs/outputs.
    h_input_1 = cuda.pagelocked_empty(batch_size * trt.volume(engine.get_binding_shape(0)), dtype=trt.nptype(data_type))
    h_output = cuda.pagelocked_empty(batch_size * trt.volume(engine.get_binding_shape(1)), dtype=trt.nptype(data_type))
    # Allocate device memory for inputs and outputs.
    d_input_1 = cuda.mem_alloc(h_input_1.nbytes)

    d_output = cuda.mem_alloc(h_output.nbytes)
    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()
    return h_input_1, d_input_1, h_output, d_output, stream


def load_images_to_buffer(pics, pagelocked_buffer):
    preprocessed = np.asarray(pics).ravel()
    np.copyto(pagelocked_buffer, preprocessed)


input_file_path = 'hy371.png'
HEIGHT = 512
WIDTH = 512
CLASSES = 2
image = cv2.imread(input_file_path)


def get_landmarks(img):
    img_in = (img - 127.5) / 127.5
    cv2.imwrite("tmp.jpg", img_in)
    img_in = np.transpose(img_in, (2, 0, 1)).astype(np.float32)
    img_in = np.expand_dims(img_in, axis=0)
    img_in /= 255.0
    # img_in = np.ascontiguousarray(img_in)
    print("Shape of the network input: ", img_in.shape)
    # print(img_in)

    # with get_engine("mobilefacenet-res2-6-10-2-dim512/onnx/face_reg_mnet.engine") as engine, engine.create_execution_context() as context:
    h_input = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(0)), dtype=np.float32)
    h_output = cuda.pagelocked_empty(trt.volume(context.get_binding_shape(1)), dtype=np.float32)

    # Allocate device memory for inputs and outputs.
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()

    # set the host input data
    # h_input = img_in
    np.copyto(h_input, img_in.ravel())
    # np.copyto(h_input, img_in.unsqueeze_(0))

    # print(h_input)
    # Transfer input data to the GPU.
    cuda.memcpy_htod_async(d_input, h_input, stream)
    # Run inference.
    context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    # Synchronize the stream
    stream.synchronize()
    # Return the host output.

    # print(h_output)
    return h_output
