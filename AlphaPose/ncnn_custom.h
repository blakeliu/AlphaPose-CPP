//
// Created by DefTruth on 2021/10/31.
//

#ifndef NCNN_CUSTOM_H
#define NCNN_CUSTOM_H

#include "ncnn/net.h"
#include "ncnn/layer.h"

// YOLOX|YOLOP|YOLOR ... use the same focus in yolov5
class YoloV5Focus : public ncnn::Layer
{
public:
  YoloV5Focus()
  {
    one_blob_only = true;
  }

  virtual int forward(const ncnn::Mat &bottom_blob, ncnn::Mat &top_blob, const ncnn::Option &opt) const;
};

ncnn::Layer* YoloV5Focus_layer_creator(void * /*userdata*/);

#endif //NCNN_CUSTOM_H
