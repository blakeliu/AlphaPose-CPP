#include <iostream>
#include <filesystem>
#include <chrono>
//#include "ort_yolox.h"
#include "clipp.h"
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


void test_alpha_pose_136() {
	std::string detector_param_path = "models/yolox_s.opt.param";
	std::string detector_bin_path = "models/yolox_s.opt.bin";
	std::string pose_weight_path = "models/multi_domain_fast50_regression_256x192-jit.pt";
	unsigned int detector_num_threads = 4;
	unsigned int pose_num_threads = 4;
	int warmup_count = 1;
	float detector_score_threshold = 0.5;
	float detector_iou_threshold = 0.6;
	int pose_batch_size = 1;
	int pose_num_joints = 136;

	auto init_t = utils::Timer();
	std::unique_ptr<AlphaPose> alpha_pose_model = std::make_unique<AlphaPose>(detector_param_path, detector_bin_path,
		pose_weight_path, detector_num_threads, pose_num_threads, 
		detector_score_threshold, detector_iou_threshold, pose_batch_size, pose_num_joints);

	alpha_pose_model->warm_up(warmup_count);
	std::cout << "AlphaPose model load and init time: " << init_t.count() << std::endl;

	std::string input_file = "pics/1.jpg";
	std::string output_file = "pics/1-136-out.jpg";

	cv::Mat image = cv::imread(input_file, cv::IMREAD_COLOR);
	auto infer_t = utils::Timer();
	std::vector<types::BoxfWithLandmarks> person_lds;
	alpha_pose_model->detect(image, person_lds);
	std::cout << "AlphaPose model infer time: " << infer_t.count() << std::endl;

	cv::Mat show_img = image.clone();
	utils::draw_pose_box_with_landmasks(show_img, person_lds, pose_num_joints);
	cv::imwrite(output_file, show_img);
}

void test_alpha_pose_26() {
	std::string detector_param_path = "models/yolox_s.opt.param";
	std::string detector_bin_path = "models/yolox_s.opt.bin";
	std::string pose_weight_path = "models/halpe26_fast_res50_256x192-jit.pt";
	unsigned int detector_num_threads = 4;
	unsigned int pose_num_threads = 4;
	int warmup_count = 1;
	float detector_score_threshold = 0.5;
	float detector_iou_threshold = 0.6;
	int pose_batch_size = 1;
	int pose_num_joints = 26;

	auto init_t = utils::Timer();
	std::unique_ptr<AlphaPose> alpha_pose_model = std::make_unique<AlphaPose>(detector_param_path, detector_bin_path,
		pose_weight_path, detector_num_threads, pose_num_threads,
		detector_score_threshold, detector_iou_threshold, pose_batch_size, pose_num_joints);

	alpha_pose_model->warm_up(warmup_count);
	std::cout << "AlphaPose model load and init time: " << init_t.count() << std::endl;

	std::string input_file = "pics/1.jpg";
	std::string output_file = "pics/1-26-out.jpg";

	cv::Mat image = cv::imread(input_file, cv::IMREAD_COLOR);
	auto infer_t = utils::Timer();
	std::vector<types::BoxfWithLandmarks> person_lds;
	alpha_pose_model->detect(image, person_lds);
	std::cout << "AlphaPose model infer time: " << infer_t.count() << std::endl;

	cv::Mat show_img = image.clone();
	utils::draw_pose_box_with_landmasks(show_img, person_lds, pose_num_joints);
	cv::imwrite(output_file, show_img);
}


int cli(int argc, char* argv[]) {
	std::string detector_param_path = "";
	std::string detector_bin_path = "";
	std::string pose_weight_path = "";
	unsigned int detector_num_threads = 4;
	unsigned int pose_num_threads = 4;
	int warmup_count = 1;
	float detector_score_threshold = 0.5;
	float detector_iou_threshold = 0.6;
	int pose_batch_size = 1;
	int pose_num_joints = 126;
	std::string input_file = "";
	std::string output_file = "";
	bool help = false;
	auto cli = (
		clipp::required("-dpm", "--detector_param_path") & clipp::value("detector_param_path prefix", detector_param_path),
		clipp::required("-dbm", "--detector_bin_path") & clipp::value("detector_bin_path prefix", detector_bin_path),
		clipp::required("-pm", "--pose_weight_path") & clipp::value("pose_weight_path prefix", pose_weight_path),
		clipp::required("-i", "--input") & clipp::value("input image file path", input_file),
		clipp::option("-o", "--output") & clipp::value("save result image", output_file),
		clipp::option("-dt", "--detector_num_threads") & clipp::value("set number of detector_num_threads", detector_num_threads),
		clipp::option("-pt", "--pose_num_threads") & clipp::value("set number of pose_num_threads", pose_num_threads),
		clipp::option("-wc", "--warmup_count") & clipp::value("set number of warmup_count", warmup_count),
		clipp::option("-pj", "--pose_joints") & clipp::value("set number of pose_joints", pose_num_joints),
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

	
	auto init_t = utils::Timer();
	std::unique_ptr<AlphaPose> alpha_pose_model = std::make_unique<AlphaPose>(detector_param_path, detector_bin_path,
		pose_weight_path, detector_num_threads, pose_num_threads,
		detector_score_threshold, detector_iou_threshold, pose_batch_size, pose_num_joints);

	alpha_pose_model->warm_up(warmup_count);
	std::cout << "AlphaPose model load and init time: " << init_t.count() << std::endl;

	cv::Mat image = cv::imread(input_file, cv::IMREAD_COLOR);
	auto infer_t = utils::Timer();
	std::vector<types::BoxfWithLandmarks> person_lds;
	alpha_pose_model->detect(image, person_lds);
	std::cout << "AlphaPose model infer time: " << infer_t.count() << std::endl;

	cv::Mat show_img = image.clone();
	utils::draw_pose_box_with_landmasks(show_img, person_lds, pose_num_joints);
	cv::imwrite(output_file, show_img);
	return 0;
}

int main(int argc, char* argv[]) {
	std::cout << "Hello, AlphaPose!" << std::endl;
	//test_alpha_pose_136();
	//test_alpha_pose_26();
	cli(argc, argv);
	return 0;
}
