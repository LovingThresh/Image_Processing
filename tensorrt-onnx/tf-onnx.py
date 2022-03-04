# 明确任务:实现tensorflow向onnx的转换


# python -m tf2onnx.convert --saved-model tensorrt-onnx/ep001-val_loss0.567-val_acc0.973 --output tensorrt-onnx/ep001 --opset 13

# 这个命令是成功的
#  python3 -m tf2onnx.convert --saved-model simple_rnn --output tensorrt-onnx/ep001.onnx --opset 13

# 接下来就是尝试把onnx变成trt


import cv2
import numpy as np
import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda

import os
import time
import common

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER)
onnx_path = r'I:\Image Processing\tensorrt-onnx\ep001.onnx'


def a_build_engine(onnx_path, shape=(1, 512, 512, 3)):
    """
   This is the function to create the TensorRT engine
   Args:
      onnx_path : Path to onnx_file.
      shape : Shape of the input of the ONNX file.
  """
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1) \
            as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        config = builder.create_builder_config()
        config.max_workspace_size = (1 << 30)
        with open(onnx_path, 'rb') as model:
            parser.parse(model.read())
        network.get_input(0).shape = shape
        engine = builder.build_serialized_network(network, config)
        # engine = builder.build_serialized_network()
        return engine


def load_engine(trt_runtime, engine_path):
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)
    return engine


# engine_file_path = '/home/JestonNano/ep599_2.engine'


def save_engine(engine, file_name):
    buf = engine.serialize()
    with open(file_name, 'wb') as f:
        f.write(buf)


engine_file_path = r'I:\Image Processing\tensorrt-onnx\ep599_2.engine'


def get_engine(onnx_file_path):
    """Attempts to load a serialized engine if available, otherwise builds a new TensorRT engine and saves it."""

    def build_engine():
        """Takes an ONNX file and creates a TensorRT engine to run inference with"""
        with trt.Builder(TRT_LOGGER) as builder, builder.create_network(
                1) as network, builder.create_builder_config() as config, trt.OnnxParser(network,
                                                                                         TRT_LOGGER) as parser, trt.Runtime(
                TRT_LOGGER) as runtime:
            config.max_workspace_size = 1 << 30  # 256MiB
            builder.max_batch_size = 1
            # Parse model file
            if not os.path.exists(onnx_file_path):
                print(
                    'ONNX file {} not found, please run yolov3_to_onnx.py first to generate it.'.format(onnx_file_path))
                exit(0)
            print('Loading ONNX file from path {}...'.format(onnx_file_path))
            with open(onnx_file_path, 'rb') as model:
                print('Beginning ONNX file parsing')
                if not parser.parse(model.read()):
                    print('ERROR: Failed to parse the ONNX file.')
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None
            # The actual yolov3.onnx is generated with batch size 64. Reshape input to batch size 1
            network.get_input(0).shape = [1, 512, 512, 3]
            print('Completed parsing of ONNX file')
            print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
            plan = builder.build_serialized_network(network, config)
            engine = runtime.deserialize_cuda_engine(plan)
            print("Completed creating Engine")
            with open(engine_file_path, "wb") as f:
                f.write(plan)
            return engine

    return build_engine()


# engine = get_engine(onnx_path)

# save_engine(engine, engine_path)

# engine = get_engine(onnx_path)

engine = load_engine(trt_runtime, r'I:\Image Processing\tensorrt-onnx\ep599_2.engine')


def load_normalized_data(data_path, pagelocked_buffer, target_size=(512, 512)):
    upsample_size = [int(target_size[1] / 8 * 4.0), int(target_size[0] / 8 * 4.0)]
    img = cv2.imread(data_path)
    img = img / 255.
    img = img * 2 - 1
    # 此时img.shape为H * W * C: 432, 848, 3
    print("图片shape", img.shape)
    # Flatten the image into a 1D array, normalize, and copy to pagelocked memory.
    # np.copyto(pagelocked_buffer, img.ravel())
    np.copyto(pagelocked_buffer, img.ravel())
    return img


input_image_path = r'I:\Image Processing\hy371.png'
with engine.create_execution_context() as context:
    inputs, outputs, bindings, stream = common.allocate_buffers(engine)
    # Do inference
    print('Running inference on image {}...'.format(input_image_path))
    upsample = load_normalized_data(input_image_path, pagelocked_buffer=inputs[0].host)
    # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
    t1 = time.time()
    trt_outputs = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
    print("推理时间", time.time() - t1)

a = trt_outputs[0]

y_true = cv2.imread('I:\Image Processing\hy371_mask.png')
y_true = np.around(y_true[:, :, 0] / 255)
y_pred = np.around(a)
# y_pred = np.round(a).reshape(1, 512, 512, 2)
# y_pred = y_pred[0,:,:,0]
# y_pred_0 = y_pred
y_pred_0 = np.zeros((512 * 512))

for i in range(512 * 512):
    y_pred_0[i] = y_pred[2 * i]
y_pred = y_pred_0.reshape(512, 512)
cv2.imwrite('y_pred_0.png', y_pred_0.reshape(512, 512, 1) * 255)

# y_true_0 = (y_true[:,:,0] / 255).ravel()
# y_true_1 = (1 - y_true[:,:,0] / 255).ravel()
# y_true = np.zeros((512*512*2))
# for i in range(512*512):
#     y_true[2 * i] = y_true_0[i]
#     y_true[2 * i + 1] = y_true_1[i]


# print(trt_outputs[0].shape)
# a = trt_outputs[0].reshape(1, 512, 512, 2)
# y_true = cv2.imread('hy371_mask.png')
# y_true = np.round(y_true[:, :, 0] / 255).reshape((1, 512, 512))
# y_pred = np.round(a[:, :, :, 0])
tp = np.sum(y_pred * y_true)
pp = np.sum(y_pred)

precision = tp / (pp + 1e-7)

pp = np.sum(y_true)
recall = tp / (pp + 1e-7)

f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
I = np.sum(y_true * y_pred)
U = np.sum(y_true + y_pred)
iou = I / (U - I)

print(precision)
print(recall)
print(f1)
print(iou)
print(1)
# # save_engine(engine, 'happy.engine')

# print(engine.get_binding_shape(0))
# print(engine.get_binding_shape(1))

# import pycuda.autoinit
# h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=trt.nptype(trt.float32))
# h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=trt.nptype(trt.float32))


# d_input = cuda.mem_alloc(h_input.nbytes)
# d_output = cuda.mem_alloc(h_output.nbytes)
# # Create a stream in which to copy inputs/outputs and run inference.
# stream = cuda.Stream()
# context = engine.create_execution_context()


# data_path = '/home/JestonNano/hy31.png'

# upsample = load_normalized_data(data_path, pagelocked_buffer=h_input)
# t1 = time.time()
# cuda.memcpy_htod_async(d_input, h_input, stream)
# # Run inference.
# context.execute_async(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
# # Transfer predictions back from the GPU.
# cuda.memcpy_dtoh_async(h_output, d_output, stream)
# # Synchronize the stream
# stream.synchronize()
# # Return the host output.
# print("推理时间", time.time() - t1)
# print(type(h_output))
# print(h_output.shape)
# out = h_output.reshape(512, 512, 2)
# print(out.shape)
# print(1)
