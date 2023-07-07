# Ultralytics YOLO 🚀, AGPL-3.0 license
"""
Export a YOLOv8 PyTorch model to other formats. TensorFlow exports authored by https://github.com/zldrobit

Format                  | `format=argument`         | Model
---                     | ---                       | ---
PyTorch                 | -                         | yolov8n.pt
TorchScript             | `torchscript`             | yolov8n.torchscript
ONNX                    | `onnx`                    | yolov8n.onnx
OpenVINO                | `openvino`                | yolov8n_openvino_model/
TensorRT                | `engine`                  | yolov8n.engine
CoreML                  | `coreml`                  | yolov8n.mlmodel
TensorFlow SavedModel   | `saved_model`             | yolov8n_saved_model/
TensorFlow GraphDef     | `pb`                      | yolov8n.pb
TensorFlow Lite         | `tflite`                  | yolov8n.tflite
TensorFlow Edge TPU     | `edgetpu`                 | yolov8n_edgetpu.tflite
TensorFlow.js           | `tfjs`                    | yolov8n_web_model/
PaddlePaddle            | `paddle`                  | yolov8n_paddle_model/

Requirements:
    $ pip install ultralytics[export]

Python:
    from ultralytics import YOLO
    model = YOLO('yolov8n.pt')
    results = model.export(format='onnx')

CLI:
    $ yolo mode=export model=yolov8n.pt format=onnx

Inference:
    $ yolo predict model=yolov8n.pt                 # PyTorch
                         yolov8n.torchscript        # TorchScript
                         yolov8n.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                         yolov8n_openvino_model     # OpenVINO
                         yolov8n.engine             # TensorRT
                         yolov8n.mlmodel            # CoreML (macOS-only)
                         yolov8n_saved_model        # TensorFlow SavedModel
                         yolov8n.pb                 # TensorFlow GraphDef
                         yolov8n.tflite             # TensorFlow Lite
                         yolov8n_edgetpu.tflite     # TensorFlow Edge TPU
                         yolov8n_paddle_model       # PaddlePaddle

TensorFlow.js:
    $ cd .. && git clone https://github.com/zldrobit/tfjs-yolov5-example.git && cd tfjs-yolov5-example
    $ npm install
    $ ln -s ../../yolov5/yolov8n_web_model public/yolov8n_web_model
    $ npm start
"""
import platform

from ultralytics.yolo.utils import __version__

ARM64 = platform.machine() in ('arm64', 'aarch64')


def export_formats():
    """YOLOv8 export formats."""
    import pandas
    x = [
        ['PyTorch', '-', '.pt', True, True],
        ['TorchScript', 'torchscript', '.torchscript', True, True],
        ['ONNX', 'onnx', '.onnx', True, True],
        ['OpenVINO', 'openvino', '_openvino_model', True, False],
        ['TensorRT', 'engine', '.engine', False, True],
        ['CoreML', 'coreml', '.mlmodel', True, False],
        ['TensorFlow SavedModel', 'saved_model', '_saved_model', True, True],
        ['TensorFlow GraphDef', 'pb', '.pb', True, True],
        ['TensorFlow Lite', 'tflite', '.tflite', True, False],
        ['TensorFlow Edge TPU', 'edgetpu', '_edgetpu.tflite', True, False],
        ['TensorFlow.js', 'tfjs', '_web_model', True, False],
        ['PaddlePaddle', 'paddle', '_paddle_model', True, True], ]
    return pandas.DataFrame(x, columns=['Format', 'Argument', 'Suffix', 'CPU', 'GPU'])


def gd_outputs(gd):
    """TensorFlow GraphDef model output node names."""
    name_list, input_list = [], []
    for node in gd.node:  # tensorflow.core.framework.node_def_pb2.NodeDef
        name_list.append(node.name)
        input_list.extend(node.input)
    return sorted(f'{x}:0' for x in list(set(name_list) - set(input_list)) if not x.startswith('NoOp'))
