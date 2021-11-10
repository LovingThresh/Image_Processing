# SPDX-License-Identifier: Apache-2.0

"""
This example builds a simple model without training.
It is converted into ONNX. Predictions are compared to
the predictions from tensorflow to check there is no
discrepencies. Inferencing time is also compared between
*onnxruntime*, *tensorflow* and *tensorflow.lite*.
"""
from onnxruntime import InferenceSession
import os
import subprocess
import timeit
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input
from Metrics import *

########################################
# Creates the model.
model = keras.models.load_model(r'output/2021-10-31-20-15-01.247650/checkpoint/ep529-val_loss0.213-val_acc0.946',
                                custom_objects={'Precision': Precision,
                                                'Recall': Recall,
                                                'F1': F1,
                                                'IOU': IOU,
                                                'H_KD_Loss': H_KD_Loss,
                                                'S_KD_Loss': S_KD_Loss
                                                })
print(model.summary())
input_names = [n.name for n in model.inputs]
output_names = [n.name for n in model.outputs]
print('inputs:', input_names)
print('outputs:', output_names)

########################################
# Training
# ....
# Skipped.

########################################
# Testing the model.
input = np.random.randn(2, 4, 4).astype(np.float32)
expected = model.predict(input)
print(expected)

########################################
# Saves the model.
if not os.path.exists("simple_rnn"):
    os.mkdir("simple_rnn")
tf.keras.models.save_model(model, "simple_rnn")

########################################
# Run the command line.
proc = subprocess.run('python -m tf2onnx.convert --saved-model simple_rnn '
                      '--output simple_rnn.onnx --opset 12'.split(),
                      capture_output=True)
print(proc.returncode)
print(proc.stdout.decode('ascii'))
print(proc.stderr.decode('ascii'))

########################################
# Runs onnxruntime.
session = InferenceSession(r"I:\Image Processing\tensorrt-onnx\ep001.onnx")
got = session.run(None, {'input_1': input})
print(got[0])

########################################
# Measures the differences.
print(np.abs(got[0] - expected).max())

########################################
# Measures processing time.
print('tf:', timeit.timeit('model.predict(input)',
                           number=100, globals=globals()))
print('ort:', timeit.timeit("session.run(None, {'input_1': input})",
                            number=100, globals=globals()))

import tensorrt as trt


def ONNX_build_engine(onnx_file_path, shape=(2, 4, 4)):
    """
    通过加载onnx文件，构建engine
    :param shape:
    :param onnx_file_path: onnx文件路径
    :return: engine
    """
    # 打印日志
    G_LOGGER = trt.Logger(trt.Logger.WARNING)

    with trt.Builder(G_LOGGER) as builder, builder.create_network(1) as network, trt.OnnxParser(network,
                                                                                                G_LOGGER) as parser:
        builder.max_batch_size = 100
        config = builder.create_builder_config()
        config.max_workspace_size = (2 << 30)

        print('Loading ONNX file from path {}...'.format(onnx_file_path))
        with open(onnx_file_path, 'rb') as model:
            print('Beginning ONNX file parsing')
            parser.parse(model.read())
        print('Completed parsing of ONNX file')

        print('Building an engine from file {}; this may take a while...'.format(onnx_file_path))
        network.get_input(0).shape = shape
        engine = builder.build_engine(network, config)
        print("Completed creating Engine")

        # 保存计划文件
        # with open(engine_file_path, "wb") as f:
        #     f.write(engine.serialize())
        return engine


import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER)


def build_engine(onnx_path, shape=[1, 512, 512, 3]):
    """
   This is the function to create the TensorRT engine
   Args:
      onnx_path : Path to onnx_file.
      shape : Shape of the input of the ONNX file.
  """
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(1) as network, trt.OnnxParser(network,
                                                                                                  TRT_LOGGER) as parser:
        config = builder.create_builder_config()
        config.max_workspace_size = (2 << 30)
        with open(onnx_path, 'rb') as model:
            parser.parse(model.read())
        network.get_input(0).shape = shape
        engine = builder.build_serialized_network(network, config)
        engine = builder.build_engine(network, config)
        return engine


engine = build_engine(r'I:\Image Processing\tensorrt-onnx\ep529.onnx')


def save_engine(engine, file_name):
    buf = engine.serialize()
    with open(file_name, 'wb') as f:
        f.write(buf)


def load_engine(trt_runtime, engine_path):
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)
    return engine
