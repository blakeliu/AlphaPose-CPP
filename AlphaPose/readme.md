# 功能
- 基于c++14实现top-down multi-person whole body estimation
- 目标检测yolox-s ncnn
- 姿态估计fastpose 基于libtorch或ncnn
- vs2019开发
- 

# package版本

- [onnxruntime-win-x64-1.11.1](https://github.com/microsoft/onnxruntime/releases/download/v1.11.1/onnxruntime-win-x64-1.11.1.zip)
- [libtorch-1.13.1 release](https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-1.13.1%2Bcpu.zip)
- [libtorch-1.31.1 debug](https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-debug-1.13.1%2Bcpu.zip)
- [ncnn not vulkan(HEAD detached at 03550ba5)]( https://github.com/nihui/ncnn.git)
- [opencv 4.6.0](https://sourceforge.net/projects/opencvlibrary/files/4.6.0/opencv-4.6.0-vc14_vc15.exe/download)
```

# onnx 转 ncnn
```shel
.\onnx2ncnn.exe models\halpe26_fast_res50_256x192.onnx models\halpe26_fast_res50_256x192.param models\halpe26_fast_res50_256x192.bin
.\ncnnoptimize.exe models\multi_domain_fast50_regression_256x192.param models\multi_domain_fast50_regression_256x192.bin models\multi_domain_fast50_regression_256x192.opt.param models\multi_domain_fast50_regression_256x192.opt.bin 65536
```

# run cli
```shell
.\AlphaPose.exe -dpm ..\..\AlphaPose\models\yolox_s.opt.param -dbm ..\..\AlphaPose\models\yolox_s.opt.bin -ppm ..\..\AlphaPose\models\multi_domain_fast50_regression_256x192.opt.param -pbm ..\..\AlphaPose\models\multi_domain_fast50_regression_256x192.opt.bin -pj 136 -i .\1.jpg -o 1_out.jpg  -dt 4 -pt 4 -wc 5
.\AlphaPose.exe -dpm ..\..\AlphaPose\models\yolox_s.opt.param -dbm ..\..\AlphaPose\models\yolox_s.opt.bin -ppm ..\..\AlphaPose\models\halpe26_fast_res50_256x192.opt.param -pbm ..\..\AlphaPose\models\halpe26_fast_res50_256x192.opt.bin -pj 26 -i .\1.jpg -o 1_out.jpg  -dt 4 -pt 4 -wc 5
```