#include "ncnn_custom.h"
#include "utils.h"
#include "ncnn_yolov5lite.h"

using alpha::NCNNYoloV5lite;

#define MAX_STRIDE 32

static inline float sigmoid(float x)
{
	return static_cast<float>(1.f / (1.f + exp(-x)));
}

// unsigmoid
static inline float unsigmoid(float y) {
	return static_cast<float>(-1.0 * (log((1.0 / y) - 1.0)));
}

alpha::NCNNYoloV5lite::NCNNYoloV5lite(const std::string& _param_path,
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

alpha::NCNNYoloV5lite::~NCNNYoloV5lite()
{
	if (net) delete net;
	net = nullptr;
}

void alpha::NCNNYoloV5lite::initialize_handler()
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
	try
	{
		net->load_param(param_path);
		net->load_model(bin_path);
	}
	catch (const std::exception& e)
	{
		std::string msg = e.what();
		std::cerr << "NCNNYoloV5lite load failed: " << msg << std::endl;
		throw std::runtime_error(msg);
	}
#ifdef POSE_DEBUG
	this->print_debug_string();
#endif
}

void alpha::NCNNYoloV5lite::transform(const cv::Mat& mat_rs, ncnn::Mat& in)
{
	// BGR NHWC -> RGB NCHW
	int h = mat_rs.rows;
	int w = mat_rs.cols;
	in = ncnn::Mat::from_pixels(mat_rs.data, ncnn::Mat::PIXEL_BGR2RGB, w, h);
	in.substract_mean_normalize(mean_vals, norm_vals);
}

void alpha::NCNNYoloV5lite::resize_unscale(const cv::Mat& mat, cv::Mat& mat_rs,
	int target_height, int target_width,
	YoloScaleParams& scale_params)
{
	if (mat.empty()) return;
	//int h = mat.rows;
	//int w = mat.cols;
	//float scale = (std::min)(static_cast<float>(input_height) / static_cast<float>(h),
	//	static_cast<float>(input_width) / static_cast<float>(w));

	//int new_unpad_w = int(w * scale + 0.5);
	//int new_unpad_h = int(h * scale + 0.5);

	//float dw = input_width - new_unpad_w;;
	//float dh = input_height - new_unpad_h;
	//dw /= 2;
	//dh /= 2;

	//// resize with unscaling
	cv::Mat unpad_mat;
	//cv::resize(mat, unpad_mat, cv::Size(new_unpad_w, new_unpad_h));
	//int top = int(round(dh - 0.1));
	//int bottom = int(round(dh + 0.1));
	//int left = int(round(dw - 0.1));
	//int right = int(round(dw + 0.1));
	int img_w = mat.cols;
	int img_h = mat.rows;

	// letterbox pad to multiple of MAX_STRIDE
	int w = img_w;
	int h = img_h;
	int target_size = (std::min)(input_height, input_width);
	float scale = 1.f;
	if (w > h)
	{
		scale = (float)target_size / w;
		w = target_size;
		h = h * scale;
	}
	else
	{
		scale = (float)target_size / h;
		h = target_size;
		w = w * scale;
	}
	cv::resize(mat, unpad_mat, cv::Size(w, h));

	int wpad = (w + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - w;
	int hpad = (h + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - h;

	cv::copyMakeBorder(unpad_mat, mat_rs, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
	//cv::copyMakeBorder(unpad_mat, mat_rs, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

	scale_params.r = scale;
	scale_params.dw = 0;
	scale_params.dh = 0;
	scale_params.new_unpad_w = wpad;
	scale_params.new_unpad_h = hpad;
	scale_params.flag = true;
}

void alpha::NCNNYoloV5lite::generate_bboxes(const ncnn::Mat& anchors, int stride,
	const ncnn::Mat& in_pad, const ncnn::Mat& feat_blob,
	float prob_threshold, std::vector<types::Boxf>& proposals_collection)
{
	const int num_grid = feat_blob.h;
	float unsig_pro = 0.f;
	if (prob_threshold > 0.6)
		unsig_pro = unsigmoid(prob_threshold);

	int num_grid_x;
	int num_grid_y;
	if (in_pad.w > in_pad.h) {
		num_grid_x = in_pad.w / stride;
		num_grid_y = num_grid / num_grid_x;
	}
	else {
		num_grid_y = in_pad.h / stride;
		num_grid_x = num_grid / num_grid_y;
	}

	const int num_class = feat_blob.w - 5;

	const int num_anchors = anchors.w / 2;

	for (int q = 0; q < num_anchors; q++) {
		const float anchor_w = anchors[q * 2];
		const float anchor_h = anchors[q * 2 + 1];

		const ncnn::Mat feat = feat_blob.channel(q);

		for (int i = 0; i < num_grid_y; i++) {
			for (int j = 0; j < num_grid_x; j++) {
				const float* featptr = feat.row(i * num_grid_x + j);

				// find class index with max class score

				float box_score = sigmoid(featptr[4]);
				// while prob_threshold > 0.6, unsigmoid better than sigmoid
				if (box_score >= prob_threshold) {
					int class_index = 0;
					float class_score = -FLT_MAX;
					for (int k = 0; k < num_class; k++) {
						float score = featptr[5 + k];
						if (score > class_score) {
							class_index = k;
							class_score = score;
						}
					}

					float confidence = box_score * sigmoid(class_score);
					if (confidence >= prob_threshold) {
						float dx = sigmoid(featptr[0]);
						float dy = sigmoid(featptr[1]);
						float dw = sigmoid(featptr[2]);
						float dh = sigmoid(featptr[3]);

						float pb_cx = (dx * 2.f - 0.5f + j) * stride;
						float pb_cy = (dy * 2.f - 0.5f + i) * stride;

						float pb_w = pow(dw * 2.f, 2) * anchor_w;
						float pb_h = pow(dh * 2.f, 2) * anchor_h;

						float x0 = pb_cx - pb_w * 0.5f;
						float y0 = pb_cy - pb_h * 0.5f;
						float x1 = pb_cx + pb_w * 0.5f;
						float y1 = pb_cy + pb_h * 0.5f;

						types::Boxf box;
						box.x1 = x0;
						box.y1 = y0;
						box.x2 = x1;
						box.y2 = y1;
						box.score = confidence;
						box.label = class_index;
						box.label_text = class_names[class_index];
						box.flag = true;
						proposals_collection.push_back(box);
					}
				}
			}
		}
	}
}

void alpha::NCNNYoloV5lite::generate_proposals(ncnn::Extractor& extractor,
	const ncnn::Mat& in_pad,
	const YoloScaleParams& scale_params,
	std::vector<types::Boxf>& proposals,
	float prob_threshold)
{
	// stride 8
	{
		ncnn::Mat out;
		extractor.extract("output", out);

		ncnn::Mat anchors(6);
		anchors[0] = 10.f;
		anchors[1] = 13.f;
		anchors[2] = 16.f;
		anchors[3] = 30.f;
		anchors[4] = 33.f;
		anchors[5] = 23.f;

		std::vector<types::Boxf> objects8;
		generate_bboxes(anchors, 8, in_pad, out, prob_threshold, objects8);

		proposals.insert(proposals.end(), objects8.begin(), objects8.end());
	}
	// stride 16
	{
		ncnn::Mat out;
		extractor.extract("1111", out);

		ncnn::Mat anchors(6);
		anchors[0] = 30.f;
		anchors[1] = 61.f;
		anchors[2] = 62.f;
		anchors[3] = 45.f;
		anchors[4] = 59.f;
		anchors[5] = 119.f;

		std::vector<types::Boxf> objects16;
		generate_bboxes(anchors, 16, in_pad, out, prob_threshold, objects16);

		proposals.insert(proposals.end(), objects16.begin(), objects16.end());
	}
	// stride 32
	{
		ncnn::Mat out;
		extractor.extract("2222", out);

		ncnn::Mat anchors(6);
		anchors[0] = 116.f;
		anchors[1] = 90.f;
		anchors[2] = 156.f;
		anchors[3] = 198.f;
		anchors[4] = 373.f;
		anchors[5] = 326.f;

		std::vector<types::Boxf> objects32;
		generate_bboxes(anchors, 32, in_pad, out, prob_threshold, objects32);

		proposals.insert(proposals.end(), objects32.begin(), objects32.end());
	}

}

void alpha::NCNNYoloV5lite::nms(std::vector<types::Boxf>& input, std::vector<types::Boxf>& output, float nms_threshold, unsigned int topk, unsigned int nms_type)
{
	if (nms_type == NMS::BLEND) utils::blending_nms(input, output, nms_threshold, topk);
	else if (nms_type == NMS::OFFSET) utils::offset_nms(input, output, nms_threshold, topk);
	else utils::hard_nms(input, output, nms_threshold, topk);
}

void alpha::NCNNYoloV5lite::detect(const cv::Mat& mat,
	std::vector<types::Boxf>& detected_boxes,
	float prob_threshold, float nms_threshold,
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
	YoloScaleParams scale_params;
	this->resize_unscale(mat, mat_rs, input_height, input_width, scale_params);
	// 1. make input tensor
	ncnn::Mat input;
	this->transform(mat_rs, input);
#ifdef POSE_TIMER
	std::cout << "YoloV5Lite trans mat time: " << trans_t.count() << std::endl;
#endif // POSE_TIMER

#ifdef POSE_TIMER
	utils::Timer infer_t;
#endif // POSE_TIMER
	// 2. inference & extract & generate proposals
	auto extractor = net->create_extractor();
	extractor.set_light_mode(false);  // default
	extractor.set_num_threads(num_threads);
	extractor.input("images", input);
	std::vector<types::Boxf> proposals_collection;
	this->generate_proposals(extractor, input, scale_params, proposals_collection, prob_threshold);
#ifdef POSE_TIMER
	std::cout << "YoloV5Lite infer mat time: " << infer_t.count() << std::endl;
#endif // POSE_TIMER

	// 3. hard|blend|offset nms with topk.
	std::vector<types::Boxf> picked_collection;
#ifdef POSE_TIMER
	utils::Timer nms_t;
#endif // POSE_TIMER
	this->nms(proposals_collection, picked_collection, nms_threshold, topk, nms_type);
#ifdef POSE_TIMER
	std::cout << "YoloV5Lite nms box time: " << nms_t.count() << std::endl;
#endif // POSE_TIMER

	// 4.fix coord offset
	int count = picked_collection.size();
	detected_boxes.resize(count);
	for (size_t i = 0; i < count; i++)
	{
		float x1 = (picked_collection[i].x1 - (scale_params.new_unpad_w / 2)) / scale_params.r;
		float y1 = (picked_collection[i].y1 - (scale_params.new_unpad_h / 2)) / scale_params.r;
		float x2 = (picked_collection[i].x2 - (scale_params.new_unpad_w / 2)) / scale_params.r;
		float y2 = (picked_collection[i].y2 - (scale_params.new_unpad_h / 2)) / scale_params.r;

		x1 = (std::max)((std::min)(x1, (float)(img_width - 1)), 0.f);
		y1 = (std::max)((std::min)(y1, (float)(img_height - 1)), 0.f);
		x2 = (std::max)((std::min)(x2, (float)(img_width - 1)), 0.f);
		y2 = (std::max)((std::min)(y2, (float)(img_height - 1)), 0.f);

		detected_boxes[i].x1 = x1;
		detected_boxes[i].y1 = y1;
		detected_boxes[i].x2 = x2;
		detected_boxes[i].y2 = y2;
		detected_boxes[i].score = picked_collection[i].score;
		detected_boxes[i].label = picked_collection[i].label;
		detected_boxes[i].label_text = picked_collection[i].label_text;
		detected_boxes[i].flag = true;
	}
}

void alpha::NCNNYoloV5lite::warm_up(int count)
{
	BasicNCNNHandler::base_warm_up(input_height, input_width, 3, count);
}
