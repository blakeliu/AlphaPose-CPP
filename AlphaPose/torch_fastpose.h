#ifndef LIBTORCH_FAST_POSE_H
#define LIBTORCH_FAST_POSE_H
#define NOMINMAX
#undef min
#undef max
#include "torch/torch.h"
#include "torch/script.h"
#include "types.h"

class FastPose
{
public:
	explicit FastPose(const std::string& _weight_path,
		unsigned int _num_threads = 1,
		int _batch_size = 1,
		int _num_joints=136,
		int _input_height = 256,
		int _input_width = 192,
		int _heatmap_channel=3,
		int _heatmap_height=64,
		int _heatmap_width=48
		);
	~FastPose();

protected:
	FastPose(const FastPose&) = delete; //
	FastPose(FastPose&&) = delete; //
	FastPose& operator=(const FastPose&) = delete; //
	FastPose& operator=(FastPose&&) = delete; //

private:
	typedef struct
	{
		float r;
		int dw;
		int dh;
		int new_unpad_w;
		int new_unpad_h;
		bool flag;
	} ScaleParams;

private:
	const unsigned int num_threads;
	const int num_joints;
	const int input_height;
	const int input_width;
	const int batch_size;
	const int heatmap_channel;
	const int heatmap_height;
	const int heatmap_width;
	std::unique_ptr<torch::jit::script::Module> _model=nullptr;
private:
	const float mean_vals[3] = { 255.f * 0.406f, 255.f * 0.457f, 255.f * 0.480f }; //bgr
	const float norm_vals[3] = { 1.f / 255.f, 1.f / 255.f , 1.f / 255.f  };
	float aspect_ratio = 0.f;
	float feat_stride[2] = {0.f, 0.f}; // h w

private:
	void transform(cv::Mat& mat_rs, at::Tensor& tensor_out);

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

};

#endif // !LIBTORCH_FAST_POSE_H



