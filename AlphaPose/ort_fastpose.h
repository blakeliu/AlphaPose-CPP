#ifndef ORT_FASTPOSE_H
#define ORT_FASTPOSE_H

#include "ort_handler.h"
#include "types.h"

namespace ort {
	class OrtFastPose :public BasicOrtHandler
	{
	public:
		explicit OrtFastPose(const std::string& _onnx_path, unsigned int _num_threads = 1) :
			BasicOrtHandler(_onnx_path, _num_threads)
		{};
		~OrtFastPose() override = default;

	private:
		typedef struct
		{
			float r;
			int dw;
			int dh;
			int new_unpad_w;
			int new_unpad_h;
			bool flag;
		} ScaleParams;

		

		const unsigned int heat_map_size[2] = { 64, 48 };
		const unsigned int input_size[2] = { 256, 192 };
		const float aspect_ratio = float(heat_map_size[1]) / float(heat_map_size[0]);
		const unsigned int feat_stride[2] = { input_size[0] / heat_map_size[0] , input_size[0] / heat_map_size[0] };
		static constexpr const float mean_vals[3] = { 225.f * 0.406f, 225.f * 0.457f, 225.f * 0.480 }; // bgr
		static constexpr const float scale_vals[3] = { 1 / (255.f), 1 / (255.f), 1 / (255.f) };

		static constexpr unsigned int num_joints = 134;
		std::array<int, num_joints>  eval_joints;

	private:
		Ort::Value transform(const cv::Mat& mat_rs) override; // without resize

		void resize_unscale(const cv::Mat& mat, cv::Mat& mat_rs,
			int target_height, int target_width,
			ScaleParams& scale_params);

		//crop box imgs
		void crop_boxes_imgs(const cv::Mat& mat, std::vector<types::Boxf>& detected_boxes, std::vector<cv::Mat>& person_imgs, std::vector<types::Boxf>& person_boxes);


	};
}


#endif // !ORT_FASTPOSE_H
