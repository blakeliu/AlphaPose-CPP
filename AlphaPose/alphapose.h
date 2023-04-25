#ifndef ALPHA_POSE_H
#define ALPHA_POSE_H

#include "headers.h"
#include "types.h"
#include "mmdeploy/pose_tracker.hpp"

namespace alpha
{
	class AlphaPose
	{
	public:
		explicit AlphaPose(const std::string& _det_weight_path,
			const std::string& _pose_weight_path,
			float _det_thr = 0.5f,
			float _det_nms_thr = 0.7f,
			float _pose_kpt_thr = 0.5f,
			float _pose_nms_thr = 0.5f,
			float _track_iou_thr = 0.4f);
		~AlphaPose();
	protected:
		AlphaPose(const AlphaPose&) = delete; //
		AlphaPose(AlphaPose&&) = delete; //
		AlphaPose& operator=(const AlphaPose&) = delete; //
		AlphaPose& operator=(AlphaPose&&) = delete; //

	private:
		std::string det_weight_path;
		std::string pose_weight_path;
		float det_score_threshold;
		float det_nms_threshold;
		float pose_score_threshold;
		float pose_nms_threshold;
		float track_iou_thr;
		const int pose_height = 256;
		const int pose_width = 192;

		std::unique_ptr<mmdeploy::PoseTracker> tracker = nullptr;
		//std::unique_ptr<mmdeploy::PoseTracker::State> state = nullptr;
		mmdeploy::PoseTracker::Params params;

	private:
		void init_tracker_params(mmdeploy::PoseTracker::Params& params);

	public:
		void detect(cv::Mat& image, std::vector<types::BoxfWithLandmarks>& person_lds);

		void warm_up(int count);


		void show(const std::vector<types::BoxfWithLandmarks>& boxes_kps, cv::Mat& image, int pose_num_joints=136, float kps_thr = 0.5, int resize = 1280);
	};
}
#endif // !ALPHA_POSE_H



