import argparse
import onnx
from onnx_caffe2.backend import Caffe2Backend


# argument parser
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Model Converting to proto buffer')
parser.add_argument('--onnx_proto_file', help='onnx proto file path')
parser.add_argument('--layers', default=20, help="number of layers of the trained model")

args = parser.parse_args()


ONNX_PROTO_FILE = args.onnx_proto_file
LAYERS = int(args.layers)


onnx_model = onnx.load(ONNX_PROTO_FILE)
# tensor([[-0.5216, -0.0826,  0.3003,  0.2617,  0.2749,  0.3003,  0.3669,  0.3221,
#          -0.3779, -0.1984]], grad_fn=<ThAddmmBackward>)
init_net, predict_net = Caffe2Backend.onnx_graph_to_caffe2_net(onnx_model.graph)
with open("onnx-init-20.pb", "wb") as f:
    f.write(init_net.SerializeToString())
with open("onnx-init-20.pbtxt".format(LAYERS), "w") as f:
    f.write(str(init_net))
with open("onnx-predict-20.pb".format(LAYERS), "wb") as f:
    f.write(predict_net.SerializeToString())
with open("onnx-predict-20.pbtxt".format(LAYERS), "w") as f:
    f.write(str(predict_net))

