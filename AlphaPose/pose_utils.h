#ifndef TORCH_POSE_UTILS_H
#define TORCH_POSE_UTILS_H

#define NOMINMAX
#undef min
#undef max
#include "torch/torch.h"
#include "torch/script.h"
#include "types.h"

namespace alpha
{
	class TorchPoseUtils
	{
	protected:
		TorchPoseUtils(const TorchPoseUtils&) = delete; //
		TorchPoseUtils(TorchPoseUtils&&) = delete; //
		TorchPoseUtils& operator=(const TorchPoseUtils&) = delete; //
		TorchPoseUtils& operator=(TorchPoseUtils&&) = delete; //

	public:
		static void integral_op(at::Tensor& hm_1d);

		static void integral_tensor(at::Tensor& preds, at::Tensor& pred_joints, at::Tensor& pred_scores, const int num_joints, const int hm_height, const int hm_width);

		static void transfrom_preds(at::Tensor& input_coord, std::vector<float>& output_coord, const std::vector<float>& center, const std::vector<float>& scale, const int hm_width, const int hm_height);

		static void get_max_pred(at::Tensor& heatmap, at::Tensor& preds, at::Tensor& maxvals);

		/*
		* pred_joints: [n, kp_num, 2]
		* pred_scores: [n, kp_num, 1]
		*/
		static void fast_nms_pose(at::Tensor& pred_joints, at::Tensor& pred_scores, at::Tensor& final_joints, at::Tensor& final_scores);

		//halpe 136
		static void heatmap_to_coord_simple_regress(int num_threads, int num_joints, at::Tensor& heatmap, types::Boxf& cropped_box, types::Landmarks& out_landmarks);

		//halpe 24
		static void heatmap_to_coord_simple(int num_threads, at::Tensor& heatmap, types::Boxf& cropped_box, types::Landmarks& out_landmarks);

		static void print_pretty_tensor(const at::Tensor& m, std::vector<int>& channel_indexs);
	};
}


#endif // !TORCH_POSE_UTILS_H


