#include <iostream>
#include <filesystem>
#include <chrono>
//#include "ort_yolox.h"
#include "mmdeploy/pose_tracker.hpp"
#include "clipp.h"
#include "utils.h"
#include "argparse.h"
#include "mediaio.h"
#include "visualize.h"
#include "pose_tracker_params.h"

int track(int argc, char* argv[]) {
	std::string detector_weight_path = "";
	std::string pose_weight_path = "";

	unsigned int detector_num_threads = 4;
	unsigned int pose_num_threads = 4;
	int warmup_count = 1;
	int cam_id = -1;
	float detector_score_threshold = 0.25f;
	float detector_iou_threshold = 0.45;
	//float detector_score_threshold = 0.60f;
	//float detector_iou_threshold = 0.60f;
	int pose_batch_size = 1;
	int pose_num_joints = 136;
	float pose_kpt_thr = 0.3f;
	std::string input_file = "";
	std::string output_file = "";
	bool help = false;
	bool use_vulkan = false;
	bool use_fp16 = false;
	auto cli = (
		clipp::required("-dm", "--detector_weight_path") & clipp::value("detector_weight_path prefix", detector_weight_path),
		clipp::required("-pm", "--pose_weight_path") & clipp::value("mmpose model dir", pose_weight_path),
		clipp::required("-i", "--input") & clipp::value("input image file path", input_file),
		clipp::required("-thresh", "--pose_kpt_thr") & clipp::value("pose_kpt_thr", pose_kpt_thr),
		clipp::option("-o", "--output") & clipp::value("save result image", output_file),
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

	// create pose tracker pipeline
	mmdeploy::PoseTracker tracker(mmdeploy::Model(detector_weight_path), mmdeploy::Model(pose_weight_path),
		mmdeploy::Device{ "cpu"});

	mmdeploy::PoseTracker::Params params;
	// initialize tracker params with program arguments
	InitTrackerParams(params);

	// create a tracker state for each video
	mmdeploy::PoseTracker::State state = tracker.CreateState(params);

	//utils::mediaio::Input input(std::to_string(cam_id), 0);
	//utils::mediaio::Output output(std::string(""), 1);

	//utils::Visualize v(0);
	//v.set_background("default");
	//v.set_skeleton(utils::Skeleton::get("coco-wholebody"));


	if (cam_id > 0)
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
				auto infer_t = utils::Timer();
				mmdeploy::PoseTracker::Result result = tracker.Apply(state, frame);
				int fps = int(1.0f / infer_t.count());

				std::vector<types::BoxfWithLandmarks> boxes_kps;
				for (const mmdeploy_pose_tracker_target_t& target : result) {
					types::Landmarks pose_lds;
					std::vector<float> socres;
					for (size_t i = 0; i < target.keypoint_count; ++i) {
						pose_lds.points.emplace_back(cv::Point2f(target.keypoints[i].x, target.keypoints[i].y));
						socres.push_back(target.scores[i]);
					}
					pose_lds.flag = true;
					types::BoxfWithLandmarks person_box_ld;
					types::Boxf box;
					box.x1 = target.bbox.left;
					box.y1 = target.bbox.bottom;
					box.x2 = target.bbox.right;
					box.y2 = target.bbox.bottom;
					box.score = 1;
					box.label = 0;
					box.label_text = "person";
					box.flag = true;

					person_box_ld.box = box;
					person_box_ld.landmarks = pose_lds;
					person_box_ld.flag = true;
					boxes_kps.push_back(person_box_ld);
				}

				cv::Mat show = frame.clone();
				utils::draw_pose_box_with_landmasks(show, boxes_kps, pose_num_joints);
				cv::putText(show, std::to_string(fps), cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 1);
				cv::imshow("cap", show);
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

//int main(int argc, char* argv[]) {
//	std::cout << "Hello, RTMPose Tracker!" << std::endl;
//	track(argc, argv);
//	return 0;
//}