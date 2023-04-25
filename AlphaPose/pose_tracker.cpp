#include <iostream>
#include <filesystem>
#include <chrono>
//#include "ort_yolox.h"
//#include "mmdeploy/pose_tracker.hpp"
#include "clipp.h"
#include "utils.h"
//#include "argparse.h"
//#include "mediaio.h"
//#include "visualize.h"
//#include "pose_tracker_params.h"
#include "alphapose.h"

int track(int argc, char* argv[]) {
	std::string detector_weight_path = "";
	std::string pose_weight_path = "";

	int warmup_count = 1;
	int cam_id = -1;
	float det_score_threshold = 0.5f;
	float det_nms_threshold = 0.7f;
	float pose_score_threshold = 0.5f;
	float pose_nms_threshold = 0.5f;
	float track_iou_threshold = 0.4f;
	int pose_num_joints = 136;
	bool help = false;
	auto cli = (
		clipp::required("-dm", "--detector_weight_path") & clipp::value("detector_weight_path prefix", detector_weight_path),
		clipp::required("-pm", "--pose_weight_path") & clipp::value("mmpose model dir", pose_weight_path),
		clipp::required("-det_score", "--det_score_thr") & clipp::value("det_score_threshold", det_score_threshold),
		clipp::required("-det_nms", "--det_nms_thr") & clipp::value("det_nms_threshold", det_nms_threshold),
		clipp::required("-pose_score", "--pose_score_thr") & clipp::value("pose_score_threshold", pose_score_threshold),
		clipp::required("-pose_nms", "--pose_nms_thr") & clipp::value("pose_nms_threshold", pose_nms_threshold),
		clipp::required("-track_iou", "--track_iou_thr") & clipp::value("track_iou_threshold", track_iou_threshold),

		clipp::option("-pj", "--pose_num_joints")& clipp::value("pose_num_joints", pose_num_joints),

		clipp::option("-id", "--cam_id") & clipp::value("camera id", cam_id),
		clipp::option("-h", "--help").set(help).doc("help")
		);
	if (!clipp::parse(argc, argv, cli)) {
		std::cout << clipp::make_man_page(cli, argv[0]);
	}

	if (help)
	{
		std::cout << clipp::make_man_page(cli, argv[0]);
		return 0;
	}

	std::unique_ptr<alpha::AlphaPose> pose_tracker = std::make_unique<alpha::AlphaPose>(
		detector_weight_path,
		pose_weight_path,
		det_score_threshold,
		det_nms_threshold,
		pose_score_threshold,
		pose_nms_threshold,
		track_iou_threshold
		);

	if (cam_id >= 0)
	{
		cv::VideoCapture cap;
		cap.open(cam_id);
		if (cap.isOpened())
		{
			std::cout << "Capture " << cam_id << " is opened" << std::endl;
			cv::Mat frame;
			for (;;)
			{
				cap >> frame;
				if (frame.empty())
					break;
				std::vector<types::BoxfWithLandmarks> person_kps;
				auto infer_t = utils::Timer();
				pose_tracker->detect(frame, person_kps, pose_num_joints);
				int fps = int(1.0f / infer_t.count());
				pose_tracker->show(person_kps, frame, pose_num_joints, det_nms_threshold);
				cv::putText(frame, std::to_string(fps), cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 1);
				cv::imshow("cap", frame);
				int c = cv::waitKey(10);
				if (c == 27 || c == 81)
				{
					break;
				}
			}
		}
	}

	return 0;
}

int main(int argc, char* argv[]) {
	std::cout << "Hello, RTMPose Tracker!" << std::endl;
	track(argc, argv);
	return 0;
}