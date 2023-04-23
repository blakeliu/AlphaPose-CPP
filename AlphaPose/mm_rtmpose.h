#ifndef MM_RTMPOSE_H
#define MM_RTMPOSE_H
#include "headers.h"
#include "types.h"
#include "mmdeploy/pose_detector.hpp"

namespace alpha
{
	class MMRTMPose
	{
	public:
		explicit MMRTMPose(const std::string& _weight_path,
			unsigned int _num_threads = 1,
			int _batch_size = 1,
			int _num_joints = 136,
			int _input_height = 768,
			int _input_width = 384);
		~MMRTMPose();

	protected:
		MMRTMPose(const MMRTMPose&) = delete; //
		MMRTMPose(MMRTMPose&&) = delete; //
		MMRTMPose& operator=(const MMRTMPose&) = delete; //
		MMRTMPose& operator=(MMRTMPose&&) = delete; //

	private:
		const unsigned int num_threads;
		const int num_joints;
		const int input_height;
		const int input_width;
		const int batch_size;
		const std::string device_name = "cpu";
		//std::unique_ptr<mmdeploy::PoseDetector> _model = nullptr;
		mmdeploy::PoseDetector * net = nullptr;

	private:
		float check_scale(cv::Mat& input);
		float resize(cv::Mat& input, cv::Mat& output);

	public:
		void detect(const cv::Mat& image, std::vector<types::Boxf>& detected_boxes, std::vector<types::BoxfWithLandmarks>& person_lds);

		void warm_up(int count);
	};

};


#endif // !MM_RTMPOSE_H



