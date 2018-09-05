import argparse
import resnet
import torch
# import onnx
# from onnx_caffe2.backend import Caffe2Backend


# argument parser
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Model Converting to proto buffer')
parser.add_argument('--pretrained_weights', help='pretrained weights path')
parser.add_argument('--layers', default=20, help="number of layers of the trained model")
args = parser.parse_args()


LAYERS = int(args.layers)
PRETRAINED_WEIGHTS = args.pretrained_weights


if LAYERS == 56:
    model = resnet.ResNet56()
else:
    model = resnet.ResNet20()


print("start loading pretrained weights")
model.load_state_dict(torch.load(PRETRAINED_WEIGHTS))
model.train(False)

x = torch.randn(1, 3, 32, 32)
outputs = model(x)
print("finish loading pretrained weights")

onnx_proto_file = "reset20.onnx"
torch_out = torch.onnx._export(model, x, onnx_proto_file, export_params=True)
print(torch_out)

# onnx_model = onnx.load(onnx_proto_file)
# tensor([[-0.5216, -0.0826,  0.3003,  0.2617,  0.2749,  0.3003,  0.3669,  0.3221,
#          -0.3779, -0.1984]], grad_fn=<ThAddmmBackward>)
# init_net, predict_net = Caffe2Backend.onnx_graph_to_caffe2_net(onnx_model.graph)
# with open("onnx-init-{}.pb".format(LAYERS), "wb") as f:
#     f.write(init_net.SerializeToString())
# with open("onnx-init-{}.pbtxt".format(LAYERS), "w") as f:
#     f.write(str(init_net))
# with open("onnx-predict-{}.pb".format(LAYERS), "wb") as f:
#     f.write(predict_net.SerializeToString())
# with open("onnx-predict-{}.pbtxt".format(LAYERS), "w") as f:
#     f.write(str(predict_net))

