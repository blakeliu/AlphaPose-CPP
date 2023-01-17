#ifndef NCNN_FASTEST_DET_H
#define NCNN_FASTEST_DET_H

#include "ncnn_handler.h"
#include "types.h"

namespace alpha {
	inline float Sigmoid(float x)
	{
		return 1.0f / (1.0f + exp(-x));
	}

	inline float Tanh(float x)
	{
		return 2.0f / (1.0f + exp(-2 * x)) - 1;
	}

	class NCNNFastestDet : public BasicNCNNHandler
	{
	public:
		//https://github.com/dog-qiuqiu/FastestDet
		explicit NCNNFastestDet(const std::string& _param_path,
			const std::string& _bin_path,
			unsigned int _num_threads = 1,
			bool _use_vulkan = false,
			int _input_height = 352,
			int _input_width = 352); //
		~NCNNFastestDet();
	protected:
		NCNNFastestDet(const NCNNFastestDet&) = delete; //
		NCNNFastestDet(NCNNFastestDet&&) = delete; //
		NCNNFastestDet& operator=(const NCNNFastestDet&) = delete; //
		NCNNFastestDet& operator=(NCNNFastestDet&&) = delete; //

	private:
		// target image size after resize, 
		const int input_height; // 352
		const int input_width; // 352

		const bool use_vulkan;

		const float mean_vals[3] = { 0.f, 0.f, 0.f };
		const float norm_vals[3] = { 1.f / 255.f, 1.f / 255.f, 1.f / 255.f };

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
		static constexpr const int class_num = 80;

	protected:
		virtual void initialize_handler();

	private:
		void transform(const cv::Mat& mat_rs, ncnn::Mat& in);

		void generate_bboxes(int img_height, int img_width,
			int input_height, int input_width, float score_threshold,
			std::vector<types::Boxf>& bbox_collection,
			ncnn::Mat& outputs); // rescale & exclude

		void nms(std::vector<types::Boxf>& input, std::vector<types::Boxf>& output,
			float iou_threshold, unsigned int topk, unsigned int nms_type);


	public:
		void detect(const cv::Mat& mat, std::vector<types::Boxf>& detected_boxes,
			float score_threshold = 0.65f, float iou_threshold = 0.45f,
			unsigned int topk = 100, unsigned int nms_type = NMS::HARD);

		void warm_up(int count);
	};
}

#endif // !NCNN_FASTEST_DET_H
