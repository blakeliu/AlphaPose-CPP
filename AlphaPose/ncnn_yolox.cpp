//
// Created by DefTruth on 2021/10/18.
//

#include "ncnn_yolox.h"
#include "ncnn_custom.h"
#include "utils.h"

using alpha::NCNNYoloX;

NCNNYoloX::NCNNYoloX(const std::string& _param_path,
	const std::string& _bin_path,
	unsigned int _num_threads,
	bool _use_vulkan,
	bool _use_fp16,
	int _input_height,
	int _input_width) :
	BasicNCNNHandler(_param_path, _bin_path, _num_threads),
	input_height(_input_height), input_width(_input_width),
	use_vulkan(_use_vulkan), use_fp16(_use_fp16), use_int8(false)
{
	this->initialize_handler();
}

NCNNYoloX::~NCNNYoloX()
{
	if (net) delete net;
	net = nullptr;
}

void NCNNYoloX::initialize_handler()
{
	net = new ncnn::Net();
	// init net, change this setting for better performance.
	//net->opt.use_fp16_arithmetic = false;
	net->opt.use_vulkan_compute = false; // default
	if (use_vulkan)
	{
		net->opt.use_vulkan_compute = true;
	}

	if (use_fp16)
	{
		net->opt.use_fp16_storage = true;
	}

	// setup Focus in yolov5
	net->register_custom_layer("YoloV5Focus", YoloV5Focus_layer_creator);
	try
	{
		net->load_param(param_path);
		net->load_model(bin_path);
	}
	catch (const std::exception& e)
	{
		std::string msg = e.what();
		std::cerr << "NCNNYoloX load failed: " << msg << std::endl;
		throw std::runtime_error(msg);
	}

#ifdef POSE_DEBUG
	this->print_debug_string();
#endif
}

void NCNNYoloX::transform(const cv::Mat& mat_rs, ncnn::Mat& in)
{
	// BGR NHWC -> RGB NCHW
	in = ncnn::Mat::from_pixels(mat_rs.data, ncnn::Mat::PIXEL_BGR2RGB, input_width, input_height);
	in.substract_mean_normalize(mean_vals, norm_vals);
}

void NCNNYoloX::resize_unscale(const cv::Mat& mat, cv::Mat& mat_rs,
	int target_height, int target_width,
	YoloXScaleParams& scale_params)
{
	if (mat.empty()) return;
	int img_height = static_cast<int>(mat.rows);
	int img_width = static_cast<int>(mat.cols);

	mat_rs = cv::Mat(target_height, target_width, CV_8UC3,
		cv::Scalar(114, 114, 114));
	// scale ratio (new / old) new_shape(h,w)
	float w_r = (float)target_width / (float)img_width;
	float h_r = (float)target_height / (float)img_height;
	float r = (std::min)(w_r, h_r);
	// compute padding
	int new_unpad_w = static_cast<int>((float)img_width * r); // floor
	int new_unpad_h = static_cast<int>((float)img_height * r); // floor
	int pad_w = target_width - new_unpad_w; // >=0
	int pad_h = target_height - new_unpad_h; // >=0

	int dw = pad_w / 2;
	int dh = pad_h / 2;

	// resize with unscaling
	cv::Mat new_unpad_mat;
	// cv::Mat new_unpad_mat = mat.clone(); // may not need clone.
	cv::resize(mat, new_unpad_mat, cv::Size(new_unpad_w, new_unpad_h));
	new_unpad_mat.copyTo(mat_rs(cv::Rect(dw, dh, new_unpad_w, new_unpad_h)));

	// record scale params.
	scale_params.r = r;
	scale_params.dw = dw;
	scale_params.dh = dh;
	scale_params.new_unpad_w = new_unpad_w;
	scale_params.new_unpad_h = new_unpad_h;
	scale_params.flag = true;
}

void NCNNYoloX::detect(const cv::Mat& mat, std::vector<types::Boxf>& detected_boxes,
	float score_threshold, float iou_threshold,
	unsigned int topk, unsigned int nms_type)
{
	if (mat.empty()) return;
	int img_height = static_cast<int>(mat.rows);
	int img_width = static_cast<int>(mat.cols);
	// resize & unscale
#ifdef POSE_TIMER
	utils::Timer trans_t;
#endif // POSE_TIMER
	cv::Mat mat_rs;
	YoloXScaleParams scale_params;
	this->resize_unscale(mat, mat_rs, input_height, input_width, scale_params);
	// 1. make input tensor
	ncnn::Mat input;
	this->transform(mat_rs, input);
#ifdef POSE_TIMER
	std::cout << "Yolox trans mat time: " << trans_t.count() << std::endl;
#endif // POSE_TIMER

#ifdef POSE_TIMER
	utils::Timer infer_t;
#endif // POSE_TIMER
	// 2. inference & extract
	auto extractor = net->create_extractor();
	extractor.set_light_mode(false);  // default
	extractor.set_num_threads(num_threads);
	extractor.input("inputs", input);
	ncnn::Mat outputs;
	extractor.extract("outputs", outputs); // (1,n=?,85=5+80=cxcy+cwch+obj_conf+cls_conf)
#ifdef POSE_TIMER
	std::cout << "Yolox infer mat time: " << infer_t.count() << std::endl;
#endif // POSE_TIMER

	// 3.rescale & exclude.
	std::vector<types::Boxf> bbox_collection;
#ifdef POSE_TIMER
	utils::Timer gen_t;
#endif // POSE_TIMER
	this->generate_bboxes(scale_params, bbox_collection, outputs, score_threshold, img_height, img_width);
#ifdef POSE_TIMER
	std::cout << "Yolox gen box time: " << gen_t.count() << std::endl;
#endif // POSE_TIMER

	// 4. hard|blend|offset nms with topk.
#ifdef POSE_TIMER
	utils::Timer nms_t;
#endif // POSE_TIMER
	this->nms(bbox_collection, detected_boxes, iou_threshold, topk, nms_type);
#ifdef POSE_TIMER
	std::cout << "Yolox nms box time: " << nms_t.count() << std::endl;
#endif // POSE_TIMER
}

void NCNNYoloX::warm_up(int count)
{
	BasicNCNNHandler::base_warm_up(input_height, input_width, 3, count);
}

void NCNNYoloX::generate_anchors(const int target_height,
	const int target_width,
	std::vector<int>& strides,
	std::vector<YoloXAnchor>& anchors)
{
	for (auto stride : strides)
	{
		int num_grid_w = target_width / stride;
		int num_grid_h = target_height / stride;
		for (int g1 = 0; g1 < num_grid_h; ++g1)
		{
			for (int g0 = 0; g0 < num_grid_w; ++g0)
			{
#ifdef _WIN32
				YoloXAnchor anchor;
				anchor.grid0 = g0;
				anchor.grid1 = g1;
				anchor.stride = stride;
				anchors.push_back(anchor);
#else
				anchors.push_back((YoloXAnchor) { g0, g1, stride });
#endif
			}
		}
	}
}

void NCNNYoloX::generate_bboxes(const YoloXScaleParams& scale_params,
	std::vector<types::Boxf>& bbox_collection,
	ncnn::Mat& outputs,
	float score_threshold, int img_height,
	int img_width)
{
	const unsigned int num_anchors = outputs.h;
	const unsigned int num_classes = outputs.w - 5;

	std::vector<YoloXAnchor> anchors;
	std::vector<int> strides = { 8, 16, 32 }; // might have stride=64
	this->generate_anchors(input_height, input_width, strides, anchors);

	float r_ = scale_params.r;
	int dw_ = scale_params.dw;
	int dh_ = scale_params.dh;

	bbox_collection.clear();
	unsigned int count = 0;

	for (unsigned int i = 0; i < num_anchors; ++i)
	{
		const float* offset_obj_cls_ptr =
			(float*)outputs.data + (i * (num_classes + 5)); // row ptr
		float obj_conf = offset_obj_cls_ptr[4];
		if (obj_conf < score_threshold) continue; // filter first.

		float cls_conf = offset_obj_cls_ptr[5];
		unsigned int label = 0;
		for (unsigned int j = 0; j < num_classes; ++j)
		{
			float tmp_conf = offset_obj_cls_ptr[j + 5];
			if (tmp_conf > cls_conf)
			{
				cls_conf = tmp_conf;
				label = j;
			}
		} // argmax

		float conf = obj_conf * cls_conf; // cls_conf (0.,1.)
		if (conf < score_threshold) continue; // filter

		const int grid0 = anchors.at(i).grid0;
		const int grid1 = anchors.at(i).grid1;
		const int stride = anchors.at(i).stride;

		float dx = offset_obj_cls_ptr[0];
		float dy = offset_obj_cls_ptr[1];
		float dw = offset_obj_cls_ptr[2];
		float dh = offset_obj_cls_ptr[3];

		float cx = (dx + (float)grid0) * (float)stride;
		float cy = (dy + (float)grid1) * (float)stride;
		float w = std::exp(dw) * (float)stride;
		float h = std::exp(dh) * (float)stride;
		float x1 = ((cx - w / 2.f) - (float)dw_) / r_;
		float y1 = ((cy - h / 2.f) - (float)dh_) / r_;
		float x2 = ((cx + w / 2.f) - (float)dw_) / r_;
		float y2 = ((cy + h / 2.f) - (float)dh_) / r_;

		types::Boxf box;
		box.x1 = (std::max)(0.f, x1);
		box.y1 = (std::max)(0.f, y1);
		box.x2 = (std::min)(x2, (float)img_width - 1.f);
		box.y2 = (std::min)(y2, (float)img_height - 1.f);
		box.score = conf;
		box.label = label;
		box.label_text = class_names[label];
		box.flag = true;
		bbox_collection.push_back(box);

		count += 1; // limit boxes for nms.
		if (count > max_nms)
			break;
	}
#ifdef POSE_DEBUG
	std::cout << "detected num_anchors: " << num_anchors << "\n";
	std::cout << "generate_bboxes num: " << bbox_collection.size() << "\n";
#endif
}

void NCNNYoloX::nms(std::vector<types::Boxf>& input, std::vector<types::Boxf>& output,
	float iou_threshold, unsigned int topk,
	unsigned int nms_type)
{
	if (nms_type == NMS::BLEND) utils::blending_nms(input, output, iou_threshold, topk);
	else if (nms_type == NMS::OFFSET) utils::offset_nms(input, output, iou_threshold, topk);
	else utils::hard_nms(input, output, iou_threshold, topk);
}












