This is the interview project for the job of AI engineer for **Secret Company**.

## Targets
- [x] reproduce the results of resnet on Cifar10 with pytorch.
- [x] convert pytorch model to caffe2 model
- [ ] predict image with c++ project organized by `cmake`


## Reproduce
This part is finished. It is easy, just implement the resnet and then train it with the suggested super parameters.

To evaluate the accuracy on test set for 20 layers resnet, just run:
```
git https://github.com/shellhue/Cifar10.git && cd Cifar10
python evaluate.py --layers=20 --weights='./weights/resnet20_164.pth'
```

The corresponding  error is:
```
8.33%  (target is 8.75%)
```

To evaluate the accuracy on test set for 56 layers resnet, just run:
```
git https://github.com/shellhue/Cifar10.git && cd Cifar10
python evaluate.py --layers=56 --weights='./weights/resnet56_164.pth'
```

The corresponding error is:
```
6.83%  (target is 6.97%)
```

## Convert to caffe2
To convert pytorch model to caffe2 model, two steps are needed.
First, convert pytorch model to onnx model:
```
python convert2onnx.py --layers=20 --pretrained_weights='./weights/resnet20_164.pth' // resnet20.onnx file will be created
```

Second, convert onnx model to caffe2 model:
```
python onnx2pb.py --layers=20 --onnx_proto_file='pathToOnnxProtoFile' // onnx-init-20.pb and onnx-predict-20.pb will be created
```

## Predict with c++
I haven’t finished this. But i know how to do it.
Just follow `caffe2_cpp_tutorial`[GitHub - leonardvandriel/caffe2_cpp_tutorial: C++ transcripts of the Caffe2 Python tutorials and other C++ example code](https://github.com/leonardvandriel/caffe2_cpp_tutorial), and then change the `pretrained.cc` in dir `/src/caffe2/binaries/` to use the `onnx-init-20.pb onnx-predict-20.pb`  files. But unfortunately, it is hard to make `caffe2_cpp_tutorial`run. I find that i have to install `caffe` by building the source which is very slow. 