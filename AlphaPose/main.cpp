#include <iostream>
#include <filesystem>
#include <chrono>
//#include "ort_yolox.h"
#include "clipp.h"
#include "utils.h"
#include "alphapose.h"
#include "mmdeploy/pose_detector.hpp"


int pose_det_demo() {
	std::string img_file = "pics/person.png";
	std::string model_path = "../alpha_pose_models/rtmpose-s-halpe";
	std::string device_name = "cpu";
	cv::Mat img = cv::imread(img_file);
	if (img.empty()) {
		fprintf(stderr, "failed to load image: %s\n", img_file.c_str());
		return -1;
	}

	using namespace mmdeploy;

	PoseDetector* detector = new PoseDetector{ Model(model_path), Device(device_name) };
	auto res = detector->Apply(img);

	for (int i = 0; i < res[0].length; i++) {
		cv::circle(img, { (int)res[0].point[i].x, (int)res[0].point[i].y }, 1, { 0, 255, 0 }, 2);
	}
	cv::imwrite("output_pose.png", img);
	if (detector) {
		delete detector;
	}
	detector = nullptr;
	return 0;
}



int cli(int argc, char* argv[]) {
	std::string detector_param_path = "";
	std::string detector_bin_path = "";
	//std::string pose_param_path = "";
	//std::string pose_bin_path = "";
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
	int pose_num_joints = 126;
	std::string input_file = "";
	std::string output_file = "";
	bool help = false;
	bool use_vulkan = false;
	bool use_fp16 = false;
	auto cli = (
		clipp::required("-dpm", "--detector_param_path") & clipp::value("detector_param_path prefix", detector_param_path),
		clipp::required("-dbm", "--detector_bin_path") & clipp::value("detector_bin_path prefix", detector_bin_path),
		clipp::required("-pm", "--pose_weight_path") & clipp::value("mmpose model dir", pose_weight_path),
		clipp::required("-i", "--input") & clipp::value("input image file path", input_file),
		clipp::option("-o", "--output") & clipp::value("save result image", output_file),
		clipp::option("-dt", "--detector_num_threads") & clipp::value("set number of detector_num_threads", detector_num_threads),
		clipp::option("-pt", "--pose_num_threads") & clipp::value("set number of pose_num_threads", pose_num_threads),
		clipp::option("-wc", "--warmup_count") & clipp::value("set number of warmup_count", warmup_count),
		clipp::option("-pj", "--pose_joints") & clipp::value("set number of pose_joints", pose_num_joints),
		clipp::option("-id", "--cam_id") & clipp::value("camera id", cam_id),
		clipp::option("-uv", "--use_vulkan").set(use_vulkan).doc("infer on vulkan"),
		clipp::option("-fp16", "--use_fp16").set(use_fp16).doc("infer speed by fp16"),
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

	int detector_height = 640;
	int detector_width = 640;
	if (detector_param_path.find("yolox_tiny") != std::string::npos || detector_param_path.find("yolox_nano") != std::string::npos ||
		detector_param_path.find("yolov5lite-s") != std::string::npos)
	{
		detector_height = 416;
		detector_width = 416;
	}
	else if (detector_param_path.find("FastestDet") != std::string::npos)
	{
		detector_height = 352;
		detector_width = 352;
	}
	else if (detector_param_path.find("yolov5lite-e") != std::string::npos)
	{
		detector_height = 320;
		detector_width = 320;
	}

	std::cout << "Object Detector input height: " << detector_height << ", input width: " << detector_width << std::endl;

	std::cout << "Cur dir: " << std::filesystem::current_path() << std::endl;

	auto init_t = utils::Timer();
	std::unique_ptr<alpha::AlphaPose> alpha_pose_model = std::make_unique<alpha::AlphaPose>(detector_param_path, detector_bin_path,
		pose_weight_path,
		detector_num_threads, pose_num_threads,
		detector_score_threshold, detector_iou_threshold,
		detector_height, detector_width,
		pose_batch_size, pose_num_joints, use_vulkan, use_fp16);

	alpha_pose_model->warm_up(warmup_count);
	std::cout << "AlphaPose model load and init time: " << init_t.count() << std::endl;

	if (cam_id > 0)
	{
		cv::VideoCapture cap;
		cap.open(cam_id);
		if (cap.isOpened())
		{
			std::cout << "Capture " << cam_id << " is opened" << std::endl;
			cv::Mat img;
			for (;;)
			{
				cap >> img;
				if (img.empty())
					break;
				cv::Mat show = img.clone();
				auto infer_t = utils::Timer();
				std::vector<types::BoxfWithLandmarks> person_lds;
				alpha_pose_model->detect(img, person_lds);
				int fps = int(1.0f / infer_t.count());

				utils::draw_pose_box_with_landmasks(show, person_lds, pose_num_joints);
				cv::putText(show, std::to_string(fps), cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 1);
				cv::imshow("cap", show);
				if (cv::waitKey(10) >= 0)
					break;
			}
		}
	}
	else
	{
		cv::Mat image = cv::imread(input_file, cv::IMREAD_COLOR);
		auto infer_t = utils::Timer();
		std::vector<types::BoxfWithLandmarks> person_lds;
		alpha_pose_model->detect(image, person_lds);
		std::cout << "AlphaPose model infer time: " << infer_t.count() << std::endl;

		cv::Mat show_img = image.clone();
		utils::draw_pose_box_with_landmasks(show_img, person_lds, pose_num_joints);
		cv::imwrite(output_file, show_img);
	}

	return 0;
}

int main(int argc, char* argv[]) {
	std::cout << "Hello, AlphaPose!" << std::endl;
	//test_alpha_pose_136();
	//test_alpha_pose_26();
	cli(argc, argv);
	//pose_det_demo();
	return 0;
}
