#ifndef _UTILS_H
#define _UTILS_H

//
// Created by tf on 22-1-25.
//

#include <cmath>
#include <vector>
#include <array>
#include <cassert>
#include <locale.h>
#include <string>
#include <algorithm>
#include <memory>
#include <fstream>
#include <unordered_map>
#include <unordered_set>
#include <limits>
#include <type_traits>
#include "opencv2/opencv.hpp"
#include "types.h"

namespace utils
{
	std::wstring to_wstring(const std::string& str);

	std::string to_string(const std::wstring& wstr);

	//box(x1,y1,x2,y2) to cnter and scale
	types::CenterScale<float> box_to_center_scale(types::Boxf& box, float aspect_ratio=1.0, float scale_mult=1.25);
	
	//cnter and scale to box(x1,y1,x2,y2)
	void center_scale_to_box(types::CenterScale<float>& cs, types::Boxf & box);

	//
	cv::Mat get_affine_transform(types::CenterScale<float>& center_scale, float output_w, float output_h, float rot=0, float shift_x=0, float shift_y=0, bool inverse=false);

	// draw functions
	void draw_landmarks_inplace(cv::Mat& mat, types::Landmarks& landmarks);

	void draw_boxes_inplace(cv::Mat& mat_inplace, const std::vector<types::Boxf>& boxes);


	void draw_boxes_with_landmarks_inplace(cv::Mat& mat_inplace, const std::vector<types::BoxfWithLandmarks>& boxes_kps, bool text);


	//softmax
	std::vector<float> softmax(const float* logits, unsigned int _size, unsigned int& max_id);


	void blending_nms(std::vector<types::Boxf>& input, std::vector<types::Boxf>& output,
		float iou_threshold, unsigned int topk);

	// reference: https://github.com/ultralytics/yolov5/blob/master/utils/general.py
	void offset_nms(std::vector<types::Boxf>& input, std::vector<types::Boxf>& output,
		float iou_threshold, unsigned int topk);


	// reference: https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/
	//            blob/master/ncnn/src/UltraFace.cpp
	void hard_nms(std::vector<types::Boxf>& input, std::vector<types::Boxf>& output,
		float iou_threshold, unsigned int topk);


	//sort 增序：由小到大
	template<typename T> std::vector<unsigned int> argsort(const std::vector<T>& arr);


	template<typename T> T clip(T v, T min_v, T max_v);
}

#endif // !_UTILS_H
