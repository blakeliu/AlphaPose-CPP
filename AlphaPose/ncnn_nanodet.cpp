#include "ncnn_nanodet.h"

alpha::NCNNNanoDet::NCNNNanoDet(const std::string& _param_path, 
	const std::string& _bin_path, 
	unsigned int _num_threads, 
	bool _use_vulkan, 
	int _input_height, 
	int _input_width) :BasicNCNNHandler(_param_path, _bin_path, _num_threads),
	input_height(_input_height), input_width(_input_width), use_vulkan(_use_vulkan)
{
	this->initialize_handler();
}

alpha::NCNNNanoDet::~NCNNNanoDet()
{
	if (net) delete net;
	net = nullptr;
}

void alpha::NCNNNanoDet::initialize_handler()
{
	net = new ncnn::Net();
	// init net, change this setting for better performance.
	net->opt.use_fp16_arithmetic = false;
	if (use_vulkan)
	{
		net->opt.use_vulkan_compute = true;
	}
	else
	{
		net->opt.use_vulkan_compute = false; // default
	}
	try
	{
		net->load_param(param_path);
		net->load_model(bin_path);
	}
	catch (const std::exception& e)
	{
		std::string msg = e.what();
		std::cerr << "NCNNNanoDet load failed: " << msg << std::endl;
		throw std::runtime_error(msg);
	}
#ifdef POSE_DEBUG
	this->print_debug_string();
#endif
}

void alpha::NCNNNanoDet::transform(const cv::Mat& mat_rs, ncnn::Mat& in)
{
}

void alpha::NCNNNanoDet::generate_bboxes(int img_height, int img_width, int input_height, int input_width, float score_threshold, std::vector<types::Boxf>& bbox_collection, ncnn::Mat& outputs)
{
}

void alpha::NCNNNanoDet::nms(std::vector<types::Boxf>& input, std::vector<types::Boxf>& output, float iou_threshold, unsigned int topk, unsigned int nms_type)
{
}

void alpha::NCNNNanoDet::detect(const cv::Mat& mat, std::vector<types::Boxf>& detected_boxes, float score_threshold, float iou_threshold, unsigned int topk, unsigned int nms_type)
{
}

void alpha::NCNNNanoDet::warm_up(int count)
{
}
