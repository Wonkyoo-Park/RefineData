## Tensorflow2ONNX2Ncnn or TensorRT

Manual to convert the frozen model of tensorflow2.0 to oxxn to ncnn with an example of customized squeezenet.

### dependencies
This code was teseted with python(>=3.6) and need two python environments, under python envoronment 1:
```
tensorflow>=2.0
```
under python environment2:
```
tensorflow1 == 1.15.0
tf2onnx
ncnn (only onfficial version)
```

### 1) Freeze the tensorflow2.0-keras in tensorflow1.15.0
Freeze the frozen model of tensorflow2.0 to tensorflow1.15.0 again
```
python tensorflow_h5topb.py --src path -to-frozen-tensorflow2-keras.h5 --dst path-to-be-frozen-teensorflow1x.pb
```

### 2) tensorflow1.15.0 to onnx
Change the frozen model of tensorflow1.15.0 to onnx
```
python -m tf2onnx.convert --input path-to-frozen-tensorflow1.pb --inputs input-name --outputs output-name --output path-to-be-converted-model.onnx
example) python -m tf2onnx.convert --input frozen_model.pb --inputs input --outputs loss/Softmax:0 --output model.onnx
```

### 3) onnx to ncnn
```
onnx2ncnn.exe onnx-model-name.onnx ncnn-model-name.param ncnn-model-name.bin
example) onnx2ncnn.exe model.onnx ncnn.param ncnn.bin
```

### 4) onnx to trt
