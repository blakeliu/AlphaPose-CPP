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
#include <chrono>
#include <type_traits>
#include "opencv2/opencv.hpp"
#include "types.h"

namespace utils
{
	class Timer {
	public:
		Timer() :
			start(std::chrono::system_clock::now())
		{};

		float count(void) {
			std::chrono::duration<double> duration = std::chrono::system_clock::now() - start;
			return duration.count();
		};
	private:
		std::chrono::time_point<std::chrono::system_clock> start;
	};
	std::wstring to_wstring(const std::string& str);

	std::string to_string(const std::wstring& wstr);

	void normalize_inplace(cv::Mat& mat_inplace, const float* mean, const float* scale);

	//box(x1,y1,x2,y2) to center and scale
	void box_to_center_scale(types::Boxf& box, std::vector<float>& center, std::vector<float>& scale, float aspect_ratio=1.0, float scale_mult=1.25);
	
	//cnter and scale to box(x1,y1,x2,y2)
	void center_scale_to_box(std::vector<float>&center, std::vector<float>& scale, types::Boxf & box);

	inline std::vector<float> get_dir(const float src_point_x, const float src_point_y, const float rot_rad) {
		float sn = sin(rot_rad);
		float cs = cos(rot_rad);
		std::vector<float> src_result{ 0.0,0.0 };
		src_result[0] = src_point_x * cs - src_point_y * sn;
		src_result[1] = src_point_x * sn + src_point_y * cs;
		return src_result;
	}
	inline cv::Point2f get_3rd_point(cv::Point2f &a, cv::Point2f &b) {
		float direction_0 = a.x - b.x;
		float direction_1 = a.y - b.y;
		return cv::Point2f{ b.x - direction_1, b.y + direction_0 };
	}

	void affine_tranform(const float x, const float y, cv::Mat& trans_mat, std::vector<float>& out_pts);

	//
	cv::Mat get_affine_transform(const std::vector<float>& center, const std::vector<float>& scale, const std::vector<float>& shift, const float output_h, const float output_w, const float rot=0, const bool inverse=false);

	// draw functions
	void draw_landmarks_inplace(cv::Mat& mat, types::Landmarks& landmarks);

	void draw_boxes_inplace(cv::Mat& mat_inplace, const std::vector<types::Boxf>& boxes);

	void draw_boxes_with_landmarks_inplace(cv::Mat& mat_inplace, const std::vector<types::BoxfWithLandmarks>& boxes_kps, bool text);

	//draw haple pose(136) or haple pose(26)
	void draw_pose_box_with_landmasks(cv::Mat& mat_inplace, const std::vector<types::BoxfWithLandmarks>& boxes_kps, int num_joints);

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
