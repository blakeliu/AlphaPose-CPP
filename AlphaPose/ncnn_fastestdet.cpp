#include "ncnn_fastestdet.h"
#include "utils.h"

alpha::NCNNFastestDet::NCNNFastestDet(const std::string& _param_path,
	const std::string& _bin_path,
	unsigned int _num_threads,
	bool _use_vulkan,
	int _input_height,
	int _input_width) :BasicNCNNHandler(_param_path, _bin_path, _num_threads),
	input_height(_input_height), input_width(_input_width), use_vulkan(_use_vulkan)
{
	this->initialize_handler();
}

alpha::NCNNFastestDet::~NCNNFastestDet()
{
	if (net) delete net;
	net = nullptr;
}

void alpha::NCNNFastestDet::initialize_handler()
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
		std::cerr << "NCNNFastestDet load failed: " << msg << std::endl;
		throw std::runtime_error(msg);
	}
#ifdef POSE_DEBUG
	this->print_debug_string();
#endif
}

void alpha::NCNNFastestDet::transform(const cv::Mat& mat_rs, ncnn::Mat& in)
{
	//in = ncnn::Mat::from_pixels(mat_rs.data, ncnn::Mat::PIXEL_BGR, input_width, input_height);
	in = ncnn::Mat::from_pixels_resize(mat_rs.data, ncnn::Mat::PIXEL_BGR, mat_rs.cols, mat_rs.rows, input_width, input_height);
	in.substract_mean_normalize(mean_vals, norm_vals);
}

void alpha::NCNNFastestDet::generate_bboxes(int img_height, int img_width, int input_height, int input_width, float score_threshold, std::vector<types::Boxf>& bbox_collection, ncnn::Mat& outputs)
{
	for (int h = 0; h < outputs.h; h++)
	{
		for (int w = 0; w < outputs.w; w++)
		{
			// 前景概率
			int obj_score_index = (0 * outputs.h * outputs.w) + (h * outputs.w) + w;
			float obj_score = outputs[obj_score_index];

			// 解析类别
			int category;
			float max_score = 0.0f;
			for (size_t i = 0; i < class_num; i++)
			{
				int obj_score_index = ((5 + i) * outputs.h * outputs.w) + (h * outputs.w) + w;
				float cls_score = outputs[obj_score_index];
				if (cls_score > max_score)
				{
					max_score = cls_score;
					category = i;
				}
			}
			float score = pow(max_score, 0.4) * pow(obj_score, 0.6);

			// 阈值筛选
			if (score > score_threshold)
			{
				// 解析坐标
				int x_offset_index = (1 * outputs.h * outputs.w) + (h * outputs.w) + w;
				int y_offset_index = (2 * outputs.h * outputs.w) + (h * outputs.w) + w;
				int box_width_index = (3 * outputs.h * outputs.w) + (h * outputs.w) + w;
				int box_height_index = (4 * outputs.h * outputs.w) + (h * outputs.w) + w;

				float x_offset = Tanh(outputs[x_offset_index]);
				float y_offset = Tanh(outputs[y_offset_index]);
				float box_width = Sigmoid(outputs[box_width_index]);
				float box_height = Sigmoid(outputs[box_height_index]);

				float cx = (w + x_offset) / outputs.w;
				float cy = (h + y_offset) / outputs.h;

				int x1 = (int)((cx - box_width * 0.5) * img_width);
				int y1 = (int)((cy - box_height * 0.5) * img_height);
				int x2 = (int)((cx + box_width * 0.5) * img_width);
				int y2 = (int)((cy + box_height * 0.5) * img_height);
				types::Boxf box;
				box.x1 = (std::max)((std::min)(x1, (int)(img_width - 1)), 0);
				box.y1 = (std::max)((std::min)(y1, (int)(img_height - 1)), 0);
				box.x2 = (std::max)((std::min)(x2, (int)(img_width - 1)), 0);
				box.y2 = (std::max)((std::min)(y2, (int)(img_height - 1)), 0);
				box.score = score;
				box.label = category;
				box.label_text = class_names[category];
				box.flag = true;
				bbox_collection.push_back(box);
			}
		}
	}
}

void alpha::NCNNFastestDet::nms(std::vector<types::Boxf>& input, std::vector<types::Boxf>& output, float iou_threshold, unsigned int topk, unsigned int nms_type)
{
	if (nms_type == NMS::BLEND) utils::blending_nms(input, output, iou_threshold, topk);
	else if (nms_type == NMS::OFFSET) utils::offset_nms(input, output, iou_threshold, topk);
	else utils::hard_nms(input, output, iou_threshold, topk);
}


void alpha::NCNNFastestDet::detect(const cv::Mat& mat, std::vector<types::Boxf>& detected_boxes, float score_threshold, float iou_threshold, unsigned int topk, unsigned int nms_type)
{
	if (mat.empty()) return;
	int img_height = static_cast<int>(mat.rows);
	int img_width = static_cast<int>(mat.cols);
	// 1. make input tensor
	// resize & unscale
	ncnn::Mat input;
	this->transform(mat, input);

#ifdef POSE_TIMER
	utils::Timer infer_t;
#endif // POSE_TIMER
	// 2. inference & extract
	auto extractor = net->create_extractor();
	extractor.set_light_mode(false);  // default
	extractor.set_num_threads(num_threads);
	extractor.input("input.1", input);
	ncnn::Mat outputs;
	extractor.extract("758", outputs); //
#ifdef POSE_TIMER
	std::cout << "NCNNFastestDet infer mat time: " << infer_t.count() << std::endl;
#endif // POSE_TIMER

	// 3.rescale & exclude.
	std::vector<types::Boxf> bbox_collection;
#ifdef POSE_TIMER
	utils::Timer gen_t;
#endif // POSE_TIMER
	this->generate_bboxes(img_height, img_width, input_height, input_width, score_threshold, bbox_collection, outputs);
#ifdef POSE_TIMER
	std::cout << "NCNNFastestDet gen box time: " << gen_t.count() << std::endl;
#endif // POSE_TIMER

	// 4. hard|blend|offset nms with topk.
#ifdef POSE_TIMER
	utils::Timer nms_t;
#endif // POSE_TIMER
	this->nms(bbox_collection, detected_boxes, iou_threshold, topk, nms_type);
#ifdef POSE_TIMER
	std::cout << "NCNNFastestDet nms box time: " << nms_t.count() << std::endl;
#endif // POSE_TIMER

}

void alpha::NCNNFastestDet::warm_up(int count)
{
	BasicNCNNHandler::base_warm_up(input_height, input_width, 3, count);
}
