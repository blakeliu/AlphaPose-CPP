#include <iostream>
#include <filesystem>
#include <chrono>
//#include "ort_yolox.h"
#include "utils.h"
#include "alphapose.h"

#if 0
//namespace fs = std::filesystem;
static void test_onnxruntime()
{
	std::cout << "Current working directory: " << fs::current_path() << std::endl;
	fs::path onnx_path = "models/yolox_s.onnx";
	fs::path test_img_path = "models/1.jpg";
	std::string save_img_path = "1_out.jpg";

	if (!fs::exists(onnx_path)) {
		std::cout << "can not found: " << onnx_path.string() << std::endl;
	}

	unsigned int num_threads = 2;
	auto init_s = std::chrono::steady_clock::now();
	ort::OrtYoloX* yolox = new ort::OrtYoloX(onnx_path.string(), num_threads);
	auto init_e = std::chrono::steady_clock::now();
	std::chrono::duration<float> init_diff = init_e - init_s;
	std::cout << "Init time: " << init_diff.count() <<" s." << std::endl;

	std::vector<types::Boxf> detected_boxes;
	cv::Mat img_bgr = cv::imread(test_img_path.string());
	auto run_s = std::chrono::steady_clock::now();
	yolox->detect(img_bgr, detected_boxes);
	auto run_e = std::chrono::steady_clock::now();
	std::chrono::duration<float> run_diff = run_e - run_s;
	std::cout << "Run time: " << run_diff.count() << " s." << std::endl;
	std::cout << "onnx yolox detected boxes num: " << detected_boxes.size() << std::endl;
	
	utils::draw_boxes_inplace(img_bgr, detected_boxes);
	//cv::imwrite(save_img_path, img_bgr);
	cv::imshow("test", img_bgr);
	cv::waitKey(0);

	delete yolox;
}
#endif


void test_alpha_pose() {
	std::string detector_param_path = "models/yolox_s.opt.param";
	std::string detector_bin_path = "models/yolox_s.opt.bin";
	std::string pose_weight_path = "models/multi_domain_fast50_regression_256x192-jit.pt";
	unsigned int detector_num_threads = 1;
	unsigned int pose_num_threads = 1;
	float detector_score_threshold = 0.5;
	float detector_iou_threshold = 0.6;
	int pose_batch_size = 1;
	int pose_num_joints = 136;

	std::unique_ptr<AlphaPose> alpha_pose_model = std::make_unique<AlphaPose>(detector_param_path, detector_bin_path,
		pose_weight_path, detector_num_threads, pose_num_threads, 
		detector_score_threshold, detector_iou_threshold, pose_batch_size, pose_num_joints);

	std::string input_file = "pics/1.jpg";
	std::string output_file = "pics/1-out.jpg";

	std::vector<types::BoxfWithLandmarks> person_lds;
	cv::Mat image = cv::imread(input_file, cv::IMREAD_COLOR);
	alpha_pose_model->detect(image, person_lds);



}

int main() {
	std::cout << "Hello, AplhaPose!" << std::endl;
	test_alpha_pose();
	return 0;
}
