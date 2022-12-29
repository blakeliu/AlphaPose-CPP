#include "alphapose.h"

AlphaPose::AlphaPose(const std::string& _detector_param_path, const std::string& _detector_bin_path, 
	const std::string& _pose_weight_path, 
	unsigned int _detector_num_threads, unsigned int _pose_num_threads, 
	float _detector_score_threshold, float _detector_iou_threshold, 
	int _pose_batch_size, int _pose_num_joints): detector_param_path(_detector_param_path), detector_bin_path(_detector_bin_path), pose_weight_path(_pose_weight_path),
	detector_num_threads(_detector_num_threads), pose_num_threads(_pose_num_threads),
	detector_score_threshold(_detector_score_threshold), detector_iou_threshold(_detector_iou_threshold),
	pose_batch_size(_pose_batch_size), pose_num_joints(_pose_num_joints)
{
	yolo_model = std::make_unique<NCNNYoloX>(detector_param_path, detector_bin_path, detector_num_threads);
	pose_model = std::make_unique<FastPose>(pose_weight_path, pose_num_threads);
}

AlphaPose::~AlphaPose()
{
}

void AlphaPose::detect(cv::Mat& image, std::vector<types::BoxfWithLandmarks> person_lds)
{
	std::vector<types::Boxf> detected_boxes;
	yolo_model->detect(image, detected_boxes, detector_score_threshold, detector_iou_threshold);
	if (detected_boxes.size() > 0)
	{
		pose_model->detect(image, detected_boxes, person_lds);
	}
}
