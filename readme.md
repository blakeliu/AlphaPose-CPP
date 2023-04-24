# 功能
- 基于c++14实现top-down multi-person whole body estimation
- 目标检测yolox-s ncnn
- 姿态估计fastpose 基于libtorch或ncnn
- vs2019开发
- [模型文件仓库](http://192.168.2.200/tface/alpha_pose_models.git)

# package版本

- [onnxruntime-win-x64-1.11.1](https://github.com/microsoft/onnxruntime/releases/download/v1.11.1/onnxruntime-win-x64-1.11.1.zip)
- [libtorch-1.13.1 release](https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-1.13.1%2Bcpu.zip)
- [libtorch-1.31.1 debug](https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-debug-1.13.1%2Bcpu.zip)
- [ncnn build是否带vulkan(HEAD detached at 03550ba5)]( https://github.com/nihui/ncnn.git)
- [opencv 4.6.0](https://sourceforge.net/projects/opencvlibrary/files/4.6.0/opencv-4.6.0-vc14_vc15.exe/download)
- [VulkanSDK-1.3.236.0-Installer.exe]()
```

# onnx 转 ncnn
```shel
.\onnx2ncnn.exe models\halpe26_fast_res50_256x192.onnx models\halpe26_fast_res50_256x192.param models\halpe26_fast_res50_256x192.bin
.\ncnnoptimize.exe models\multi_domain_fast50_regression_256x192.param models\multi_domain_fast50_regression_256x192.bin models\multi_domain_fast50_regression_256x192.opt.param models\multi_domain_fast50_regression_256x192.opt.bin 65536
```

# run cli
- fp32
```shell
.\AlphaPose.exe -dpm ..\..\AlphaPose\models\yolox_s.opt.param -dbm ..\..\AlphaPose\models\yolox_s.opt.bin -ppm ..\..\AlphaPose\models\multi_domain_fast50_regression_256x192.opt.param -pbm ..\..\AlphaPose\models\multi_domain_fast50_regression_256x192.opt.bin -pj 136 -i .\1.jpg -o 1_out.jpg  -dt 4 -pt 4 -wc 5
.\AlphaPose.exe -dpm ..\..\AlphaPose\models\yolox_s.opt.param -dbm ..\..\AlphaPose\models\yolox_s.opt.bin -ppm ..\..\AlphaPose\models\halpe26_fast_res50_256x192.opt.param -pbm ..\..\AlphaPose\models\halpe26_fast_res50_256x192.opt.bin -pj 26 -i .\1.jpg -o 1_out.jpg  -dt 4 -pt 4 -wc 5
 -dpm  ..\alpha_pose_models\yolox_nano.opt.param -dbm ..\alpha_pose_models\yolox_nano.opt.bin -ppm ..\alpha_pose_models\halpe26_fast_res50_256x192.opt.param -pbm ..\alpha_pose_models\halpe26_fast_res50_256x192.opt.bin  -pj 26  -i pics\1.jpg -o pics\1-out.jpg -dt 4 -pt 4 -wc 5
```

- fp16
```shell
 -dpm  ..\alpha_pose_models\yolox_nano-opt.param -dbm ..\alpha_pose_models\yolox_nano-opt.bin -ppm ..\alpha_pose_models\halpe26_fast_res50_256x192-opt.param -pbm ..\alpha_pose_models\halpe26_fast_res50_256x192-opt.bin  -pj 26  -i pics\1.jpg -o pics\1-out.jpg -dt 4 -pt 4 -wc 5 -fp16
```


[yolov5lite-e fp16](https://pan.baidu.com/s/1kWtwx1C0OTTxbwqJyIyXWg)
```
-dpm  ..\alpha_pose_models\yolov5lites_fp16\yolov5lite-e.param -dbm ..\alpha_pose_models\yolov5lites_fp16\yolov5lite-e.bin -ppm ..\alpha_pose_models\halpe26_fast_res50_256x192-opt.param -pbm ..\alpha_pose_models\halpe26_fast_res50_256x192-opt.bin  -pj 26  -i pics\1.jpg -o pics\1-out.jpg -dt 4 -pt 4 -wc 5 -fp16
```

# time cost
```txt
Object Detector input height: 416, input width: 416
AlphaPose model load and init time: 1.07343
Yolox trans mat time: 0.0039237
Yolox infer mat time: 0.0382188
Yolox gen box time: 0.0001189
Yolox nms box time: 1.14e-05
Object detected time: 0.0454368
Preprocessed image tensor 0 time: 0.0009864
ncnn fastpose forward 0 time: 0.0600666
ncnn mat convert torch tensor0 time: 0.0003712
Generate landmarks 0 time: 0.0125228
Pose detected time: 0.0765909
AlphaPose model infer time: 0.122566
```

# models
- fp32
```txt
yolox_nano.opt
halpe26_fast_res50_256x192
halpe26_fast_res50_256x192-fp32

```
- fp16
```
yolox_nano-opt
halpe26_fast_res50_256x192-opt


```


# convert fp16 int8

- int8
```shell
/ncnn2table yolov5-lite-opt.param yolov5-lite-opt.bin imagelist.txt yolov5-lite.table mean=[104,117,123] norm=[0.229f, 0.224, 0.225] shape=[640,640,3] pixel=BGR thread=8 method=kl
```

# run
```
-dpm  ..\alpha_pose_models\yolov5lites_fp16\yolov5lite-e.param -dbm ..\alpha_pose_models\yolov5lites_fp16\yolov5lite-e.bin -pm ..\alpha_pose_models\rtmpose-s-halpe  -pj 136  -i pics\1.jpg -o pics\1-out.jpg -dt 4 -pt 4 -wc 5  -id 1
```