#include <iostream>
#include <filesystem>
#include <chrono>
#include "ort_yolox.h"
#include "utils.h"


namespace fs = std::filesystem;

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


int main() {
	std::cout << "Hello, AplhaPose!" << std::endl;

	test_onnxruntime();
	return 0;
}
