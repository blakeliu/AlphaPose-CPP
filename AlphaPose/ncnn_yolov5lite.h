#ifndef NCNN_YOLOV5LITE_H
#define NCNN_YOLOV5LITE_H

#include "ncnn_handler.h"
#include "types.h"

namespace alpha {
	//https://github.com/ppogg/YOLOv5-Lite/blob/master/cpp_demo/ncnn/v5lite-s.cpp
	class NCNNYoloV5lite :
		public BasicNCNNHandler
	{
	public:
		explicit NCNNYoloV5lite(const std::string& _param_path,
			const std::string& _bin_path,
			unsigned int _num_threads = 1,
			bool _use_vulkan = false,
			bool _use_fp16 = false,
			int _input_height = 320,
			int _input_width = 320); //
		~NCNNYoloV5lite();

	private:
		// nested classes
		typedef struct GridAndStride
		{
			int grid0;
			int grid1;
			int stride;
		} YoloAnchor;

		typedef struct
		{
			float r;
			int dw;
			int dh;
			int new_unpad_w;
			int new_unpad_h;
			bool flag;
		} YoloScaleParams;

	private:
		// target image size after resize, might use 416 for small model(nano/tiny)
		const int input_height; // 320(v5lite-e)/416(v5lite-s)
		const int input_width; // 320(v5lite-e)/416(v5lite-s)

		const bool use_vulkan;
		const bool use_fp16;
		const bool use_int8;

		const float mean_vals[3] = { 0, 0, 0 };
		const float norm_vals[3] = { 1.f / (255.f), 1.f / (255.f), 1.f / (255.f) };

		const char* class_names[80] = {
			"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
			"fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
			"elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
			"skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
			"tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
			"sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
			"potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
			"cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
			"scissors", "teddy bear", "hair drier", "toothbrush"
		};
		enum NMS
		{
			HARD = 0, BLEND = 1, OFFSET = 2
		};
		static constexpr const unsigned int max_nms = 30000;

	protected:
		NCNNYoloV5lite(const NCNNYoloV5lite&) = delete; //
		NCNNYoloV5lite(NCNNYoloV5lite&&) = delete; //
		NCNNYoloV5lite& operator=(const NCNNYoloV5lite&) = delete; //
		NCNNYoloV5lite& operator=(NCNNYoloV5lite&&) = delete; //

	protected:
		virtual void initialize_handler();

	private:

		void transform(const cv::Mat& mat_rs, ncnn::Mat& in);

		void resize_unscale(const cv::Mat& mat,
			cv::Mat& mat_rs,
			int target_height,
			int target_width,
			YoloScaleParams& scale_params);

		void generate_bboxes(const ncnn::Mat& anchors, int stride, const ncnn::Mat& in_pad,
			const ncnn::Mat& feat_blob, float prob_threshold,
			std::vector<types::Boxf>& bbox_collection);

		void generate_proposals(
			ncnn::Extractor& extractor,
			const ncnn::Mat& in_pad,
			const YoloScaleParams& scale_params,
			std::vector<types::Boxf>& proposals,
			float prob_threshold); // 

		void nms(std::vector<types::Boxf>& input, std::vector<types::Boxf>& output,
			float nms_threshold, unsigned int topk, unsigned int nms_type);

	public:
		void detect(const cv::Mat& mat, std::vector<types::Boxf>& detected_boxes,
			float prob_threshold = 0.25f, float nms_threshold = 0.45f,
			unsigned int topk = 100, unsigned int nms_type = NMS::OFFSET);

		void warm_up(int count);
	};

}

#endif NCNN_YOLOV5LITE_H