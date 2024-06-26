#Quantization
/usr/bin/trtexec --onnx=/home/alumnos/mgonzalez/precision/casiabueno.onnx --saveEngine=/home/alumnos/mgonzalez/precision/casiabueno.trt
/usr/bin/trtexec --loadEngine=/home/alumnos/mgonzalez/precision/casiabueno.trt
#FP32
#/home/alumnos/mgonzalez/Downloads/TensorRT-6.0.1.5/bin/trtexec --onnx=/home/alumnos/mgonzalez/resnet50_onnx.onnx --saveEngine=/home/alumnos/mgonzalez/resnet50_batch1_fp32.trt
#/usr/src/tensorrt/bin/trtexec --onnx=/home/nano/resnet50_onnx.onnx --saveEngine=/home/nano/trt_models/resnet50_batch1_fp32.trt
#/usr/src/tensorrt/bin/trtexec --loadEngine=/home/nano/trt_models/resnet50_batch1_fp32.trt

#FP16
#/usr/src/tensorrt/bin/trtexec --onnx=/home/nano/resnet50_onnx.onnx --saveEngine=/home/nano/trt_models/resnet50_batch1_fp16.trt --fp16
#/usr/src/tensorrt/bin/trtexec --loadEngine=/home/nano/trt_models/resnet50_batch1_fp16.trt --fp16

#INT8
#/usr/src/tensorrt/bin/trtexec --onnx=/home/nano/resnet50_onnx.onnx --saveEngine=/home/nano/trt_models/resnet50_batch1_int8.trt --int8
#/usr/src/tensorrt/bin/trtexec --loadEngine=/home/nano/trt_models/resnet50_batch1_int8.trt --int8

#Best
#/usr/src/tensorrt/bin/trtexec --onnx=/home/nano/resnet50_onnx.onnx --saveEngine=/home/nano/trt_models/resnet50_batch1_best.trt --best
#/usr/src/tensorrt/bin/trtexec --loadEngine=/home/nano/trt_models/resnet50_batch1_best.trt --best

#Change batch size

#FP32
#/usr/src/tensorrt/bin/trtexec --onnx=/home/nano/resnet50_onnx_dynamic_batch.onnx --saveEngine=/home/nano/trt_models/resnet50_batch8_fp32.trt --shapes=\'input\':8x3x224x224
#/usr/src/tensorrt/bin/trtexec --loadEngine=/home/nano/trt_models/resnet50_batch8_fp32.trt
