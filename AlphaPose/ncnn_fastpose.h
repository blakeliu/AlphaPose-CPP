#ifndef NCNN_FASTPOSE_H
#define NCNN_FASTPOSE_H
#define NOMINMAX
#undef min
#undef max
#include "torch/torch.h"
#include "torch/script.h"
#include "ncnn_handler.h"
#include "types.h"

namespace alpha
{

	class NCNNFastPose : public BasicNCNNHandler
	{
	public:
		explicit NCNNFastPose(const std::string& _param_path,
			const std::string& _bin_path,
			unsigned int _num_threads = 1,
			int _batch_size = 1,
			int _num_joints = 136,
			bool _use_vulkan = false,
			int _input_height = 256,
			int _input_width = 192,
			int _heatmap_channel = 3,
			int _heatmap_height = 64,
			int _heatmap_width = 48
		);
		~NCNNFastPose();

	protected:
		NCNNFastPose(const NCNNFastPose&) = delete; //
		NCNNFastPose(NCNNFastPose&&) = delete; //
		NCNNFastPose& operator=(const NCNNFastPose&) = delete; //
		NCNNFastPose& operator=(NCNNFastPose&&) = delete; //

	protected:
		virtual void initialize_handler();

	private:
		const int num_joints;
		const int input_height;
		const int input_width;
		const int batch_size;
		const int heatmap_channel;
		const int heatmap_height;
		const int heatmap_width;
		const bool use_vulkan;
		std::unique_ptr<torch::jit::script::Module> _model = nullptr;
	private:
		const float mean_vals[3] = { 255.f * 0.406f, 255.f * 0.457f, 255.f * 0.480f }; //bgr
		const float norm_vals[3] = { 1.f / 255.f, 1.f / 255.f , 1.f / 255.f };
		float aspect_ratio = 0.f;
		float feat_stride[2] = { 0.f, 0.f }; // h w

	private:

		void transform(const cv::Mat& mat_rs, ncnn::Mat& in);


		void integral_op(at::Tensor& hm_1d);

		void integral_tensor(at::Tensor& preds, at::Tensor& pred_joints, at::Tensor& pred_scores, const int num_joints, const int hm_height, const int hm_width);

		void crop_image(const cv::Mat& input_mat, cv::Mat& crop_mat, types::Boxf& detected_box, types::Boxf& cropped_box);

		void transfrom_preds(at::Tensor& input_coord, std::vector<float>& output_coord, const std::vector<float>& center, const std::vector<float>& scale, const int hm_width, const int hm_height);

		void get_max_pred(at::Tensor& heatmap, at::Tensor& preds, at::Tensor& maxvals);

		/*
		* pred_joints: [n, kp_num, 2]
		* pred_scores: [n, kp_num, 1]
		*/
		void fast_nms_pose(at::Tensor& pred_joints, at::Tensor& pred_scores, at::Tensor& final_joints, at::Tensor& final_scores);

		//halpe 136
		void heatmap_to_coord_simple_regress(at::Tensor& heatmap, types::Boxf& cropped_box, types::Landmarks& out_landmarks);

		//halpe 24
		void heatmap_to_coord_simple(at::Tensor& heatmap, types::Boxf& cropped_box, types::Landmarks& out_landmarks);

	public:
		void detect(const cv::Mat& image, std::vector<types::Boxf>& detected_boxes, std::vector<types::BoxfWithLandmarks>& person_lds);

		void warm_up(int count);

		static void print_pretty_tensor(const at::Tensor& m, std::vector<int>& channel_indexs);
	};

}
#endif // !NCNN_FASTPOSE_H