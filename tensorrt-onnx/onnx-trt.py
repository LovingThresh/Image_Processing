#  !wget https://s3.amazonaws.com/download.onnx/models/opset_8/resnet50.tar.gz
#  tar xzf resnet50.tar.gz

# trtexec --onnx=resnet50/model.onnx --saveEngine=resnet_engine.trt  --explicitBatch=1
# trtexec --onnx=C:\Users\liuye\Desktop\onnx\ep599.onnx --saveEngine=C:\Users\liuye\Desktop\onnx\resnet_engine.trt  --explicitBatch=1 --workspace=512


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


def load_engine(trt_runtime, engine_path):
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)
    return engine


def load_normalized_data(img, pagelocked_buffer, target_size=(512, 512)):
    img = img / 255.
    img = img * 2 - 1
    np.copyto(pagelocked_buffer, img.ravel())


videoCapture = cv2.VideoCapture(r'C:\Users\liuye\Desktop\video.mp4')

fps = videoCapture.get(cv2.CAP_PROP_FPS)
size = (int(512),
        int(512))
fNUMS = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)

engine = load_engine(trt_runtime, r'I:\Image Processing\ep529.engine')

#fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# videoWriter = cv2.VideoWriter("video_2.mp4", fourcc, 15, (640, 368), True)

success, frame = videoCapture.read()
with engine.create_execution_context() as context:
    inputs, outputs, bindings, stream = common.allocate_buffers(engine)
    # Do inference

    while success:  # Loop until there are no more frames.
        frame = cv2.resize(frame, (512, 512))
        load_normalized_data(frame, pagelocked_buffer=inputs[0].host)
        # Set host input to the image. The common.do_inference function will copy the input to the GPU before executing.
        trt_outputs = common.do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream)
        a = trt_outputs[0].reshape((512, 512, 2))
        y_pred = np.uint8(np.around(a[:, :, 0])).reshape(512, 512, 1)
        y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=2)
        frame = frame * (1 - y_pred) + y_pred * 255
        frame = cv2.resize(frame, (640, 368))
        cv2.imshow('windows', frame)
        cv2.waitKey(int(1000 / int(fps)))
        # videoWriter.write(frame)
        success, frame = videoCapture.read()

