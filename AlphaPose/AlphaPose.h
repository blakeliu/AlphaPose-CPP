#ifndef ALPHA_POSE_H
#define ALPHA_POSE_H

//#include "torch_fastpose.h"
#include "ncnn_fastpose.h"
#include "ncnn_yolox.h"
#include "ncnn_fastestdet.h"

namespace alpha
{
	class AlphaPose
	{
	public:
		explicit AlphaPose(const std::string& _detector_param_path, const std::string& _detector_bin_path,
			const std::string& _pose_param_path, const std::string& _pose_bin_path,
			unsigned int _detector_num_threads = 1, unsigned int _pose_num_threads = 1,
			float _detector_score_threshold = 0.25f, float _detector_iou_threshold = 0.45f,
			int _detector_height = 640, int _detector_width = 640,
			int _pose_batch_size = 1, int _pose_num_joints = 136, bool _use_vulkan = false);
		~AlphaPose();
	protected:
		AlphaPose(const AlphaPose&) = delete; //
		AlphaPose(AlphaPose&&) = delete; //
		AlphaPose& operator=(const AlphaPose&) = delete; //
		AlphaPose& operator=(AlphaPose&&) = delete; //

	private:
		std::string detector_param_path;
		std::string detector_bin_path;
		std::string pose_param_path;
		std::string pose_bin_path;
		unsigned int detector_num_threads;
		unsigned int pose_num_threads;
		float detector_score_threshold;
		float detector_iou_threshold;
		int pose_batch_size;
		int pose_num_joints;
		//std::unique_ptr<NCNNYoloX> det_model = nullptr;
		std::unique_ptr<NCNNFastestDet> det_model = nullptr;
		//std::unique_ptr<TorchFastPose> pose_model = nullptr;
		std::unique_ptr<NCNNFastPose> pose_model = nullptr;

	public:
		void detect(cv::Mat& image, std::vector<types::BoxfWithLandmarks>& person_lds);

		void warm_up(int count);
	};
}
#endif // !ALPHA_POSE_H



