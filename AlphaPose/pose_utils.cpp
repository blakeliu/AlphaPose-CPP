#include "pose_utils.h"
#include "utils.h"


void alpha::TorchPoseUtils::integral_op(at::Tensor& hm_1d)
{
	int feat_dim = hm_1d.size(-1);
	at::Tensor range_tensor = at::arange(feat_dim, at::kFloat);
	hm_1d *= range_tensor;
}

void alpha::TorchPoseUtils::integral_tensor(at::Tensor& preds, at::Tensor& pred_joints, at::Tensor& pred_scores, const int num_joints, const int hm_height, const int hm_width)
{
	int bs = preds.size(0);
	int feat_dim = hm_height * hm_width;
	preds = preds.reshape({ bs, num_joints, feat_dim });
	preds = preds.sigmoid();

	//get hm confidence
	std::tuple<torch::Tensor, torch::Tensor> max_classes = torch::max(preds, 2, true);
	pred_scores = std::get<0>(max_classes);

	//norm to probabilty
	auto heatmaps = preds / preds.sum(2, true);
	heatmaps = heatmaps.reshape({ heatmaps.size(0), num_joints, 1, hm_height, hm_width });
	// edge probablity
	auto hm_x = heatmaps.sum({ 2, 3 });
	auto hm_y = heatmaps.sum({ 2, 4 });
	integral_op(hm_x);
	integral_op(hm_y);
	auto coord_x = hm_x.sum(2, true);
	auto coord_y = hm_y.sum(2, true);
	coord_x = coord_x / static_cast<float>(hm_width) - 0.5f;
	coord_y = coord_y / static_cast<float>(hm_height) - 0.5f;

	pred_joints = at::cat({ coord_x, coord_y }, 2);
	pred_joints = pred_joints.reshape({ pred_joints.size(0), num_joints, 2 });
}

void alpha::TorchPoseUtils::transfrom_preds(at::Tensor& input_coord, std::vector<float>& output_coord, const std::vector<float>& center, const std::vector<float>& scale, const int hm_width, const int hm_height)
{
	const std::vector<float> shift = { 0.f, 0.f };
	cv::Mat trans = utils::get_affine_transform(center, scale, shift, hm_height, hm_width, 0, true);
	const float x = input_coord[0].item<float>();
	const float y = input_coord[1].item<float>();
	utils::affine_tranform(x, y, trans, output_coord);
}

void alpha::TorchPoseUtils::get_max_pred(at::Tensor& heatmap, at::Tensor& preds, at::Tensor& maxvals)
{
	int hm_jonts_num = heatmap.size(0);
	int hm_height = heatmap.size(1);
	int hm_width = heatmap.size(2);

	at::Tensor reshap_hm = heatmap.reshape({ hm_jonts_num, -1 });
	std::tuple<torch::Tensor, torch::Tensor> max_classes = at::max(reshap_hm, 1);
	maxvals = std::get<0>(max_classes);
	at::Tensor idx = at::argmax(reshap_hm, 1);

	maxvals = maxvals.reshape({ hm_jonts_num, 1 });
	idx = idx.reshape({ hm_jonts_num, 1 });

	preds = at::tile(idx, { 1, 2 }).to(at::kFloat);
	preds.index({ "...", 0 }) = preds.index({ "...", 0 }) % hm_width;
	preds.index({ "...", 1 }) = at::floor(preds.index({ "...", 1 }) / hm_width);
	at::Tensor pred_mask = at::tile(at::greater(maxvals, 0.f), { 1, 2 }).to(at::kFloat);

	preds *= pred_mask;
}

void alpha::TorchPoseUtils::fast_nms_pose(at::Tensor& pred_joints, at::Tensor& pred_scores, at::Tensor& final_joints, at::Tensor& final_scores)
{
	at::Tensor normed_scores = pred_scores / at::sum(pred_scores, 0);
	final_joints = at::mul(pred_joints, normed_scores.repeat({ 1, 1, 2 })).sum(0);
	final_scores = at::mul(pred_scores, normed_scores).sum(0);
}

void alpha::TorchPoseUtils::heatmap_to_coord_simple_regress(int num_threads, int num_joints, at::Tensor& heatmap, types::Boxf& cropped_box, types::Landmarks& out_landmarks)
{
	const int bs = heatmap.size(0);
	const int hm_channel = heatmap.size(1);
	const int hm_height = heatmap.size(2);
	const int hm_width = heatmap.size(3);
	at::Tensor pred_joints;
	at::Tensor pred_scores;
	//post-processing
	integral_tensor(heatmap, pred_joints, pred_scores, num_joints, hm_height, hm_width);
#ifdef POSE_DEBUG
	std::cout << "pred_joints size: " << pred_joints.sizes() << std::endl;
	std::cout << "pred_scores size: " << pred_scores.sizes() << std::endl;
#endif // POSE_DEBUG

	pred_joints.index({ "...", 0 }) = (pred_joints.index({ "...", 0 }) + 0.5) * hm_width;
	pred_joints.index({ "...", 1 }) = (pred_joints.index({ "...", 1 }) + 0.5) * hm_height;

	//at::Tensor trans_joints = at::zeros_like(pred_joints);	
	float w = cropped_box.width();
	float h = cropped_box.height();
	const std::vector<float> center = { cropped_box.x1 + w * 0.5f, cropped_box.y1 + h * 0.5f };
	const std::vector<float> scale = { w, h };
	for (int i = 0; i < pred_joints.size(0); i++)
	{
#pragma omp parallel for num_threads(num_threads)
		for (int j = 0; j < pred_joints.size(1); j++)
		{
			std::vector<float> trans_pt;
			at::Tensor pt_tensor = pred_joints[i][j];
			transfrom_preds(pt_tensor, trans_pt, center, scale, hm_width, hm_height);
			pred_joints[i][j][0] = trans_pt[0];
			pred_joints[i][j][1] = trans_pt[1];
		}
	}

	at::Tensor final_joints;
	at::Tensor final_scores;
	fast_nms_pose(pred_joints, pred_scores, final_joints, final_scores);
#ifdef POSE_DEBUG
	std::cout << "final_joints size: " << final_joints.sizes() << std::endl;
	std::cout << "final_scores size: " << final_scores.sizes() << std::endl;
#endif // POSE_DEBUG
	std::vector<cv::Point2f> points;
	for (size_t i = 0; i < final_joints.size(0); i++)
	{
		float x = final_joints[i][0].item<float>();
		float y = final_joints[i][1].item<float>();
		points.emplace_back(cv::Point2f(x, y));
	}
	out_landmarks.points = points;
	out_landmarks.flag = true;
}

void alpha::TorchPoseUtils::heatmap_to_coord_simple(int num_threads, at::Tensor& heatmap, types::Boxf& cropped_box, types::Landmarks& out_landmarks)
{
	at::Tensor pred_joints, pred_scores;
	at::Tensor hms = heatmap[0]; //bs = 1

	get_max_pred(hms, pred_joints, pred_scores);
#ifdef POSE_DEBUG
	std::cout << "pred_joints size: " << pred_joints.sizes() << std::endl;
	std::cout << "pred_scores size: " << pred_scores.sizes() << std::endl;
#endif // POSE_DEBUG

	int hm_channel = hms.size(0);
	int hm_height = hms.size(1);
	int hm_width = hms.size(2);

	//post-processing
#pragma omp parallel for num_threads(num_threads)
	for (int i = 0; i < pred_joints.size(0); i++)
	{
		at::Tensor hm = hms[i];
		int px = static_cast<int>(std::round(pred_joints[i][0].item<float>()));
		int py = static_cast<int>(std::round(pred_joints[i][1].item<float>()));

		px = (std::max)((std::min)(px, (int)(hm_width - 1)), 0);
		py = (std::max)((std::min)(py, (int)(hm_height - 1)), 0);
		if (px > 1 && px < hm_width - 1 && py > 1 && px < hm_height - 1)
		{
			int px_add = px + 1;
			int py_add = py + 1;
			int px_sub = px - 1;
			int py_sub = py - 1;
			float diff_x = hm[py][px_add].item<float>() - hm[py][px_sub].item<float>();
			float diff_y = hm[py_add][px].item<float>() - hm[py_sub][px].item<float>();

			pred_joints[i][0] += diff_x == 0 ? 0 : (diff_x > 0) ? 0.25f : -0.25f;
			pred_joints[i][1] += diff_y == 0 ? 0 : (diff_y > 0) ? 0.25f : -0.25f;
		}
	}

	//transform box to scale
	float w = cropped_box.width();
	float h = cropped_box.height();
	const std::vector<float> center = { cropped_box.x1 + w * 0.5f, cropped_box.y1 + h * 0.5f };
	const std::vector<float> scale = { w, h };

#pragma omp parallel for num_threads(num_threads)
	for (int i = 0; i < pred_joints.size(0); i++)
	{
		std::vector<float> trans_pt;
		at::Tensor pt_tensor = pred_joints[i];
		transfrom_preds(pt_tensor, trans_pt, center, scale, hm_width, hm_height);
		pred_joints[i][0] = trans_pt[0];
		pred_joints[i][1] = trans_pt[1];
	}
	at::Tensor final_joints;
	at::Tensor final_scores;
	pred_joints = pred_joints.unsqueeze_(0);
	pred_scores = pred_scores.unsqueeze_(0);
#ifdef POSE_DEBUG
	std::cout << "pred_joints size: " << pred_joints.sizes() << std::endl;
	std::cout << "pred_scores size: " << pred_scores.sizes() << std::endl;
#endif // POSE_DEBUG
	fast_nms_pose(pred_joints, pred_scores, final_joints, final_scores);
#ifdef POSE_DEBUG
	std::cout << "final_joints size: " << final_joints.sizes() << std::endl;
	std::cout << "final_scores size: " << final_scores.sizes() << std::endl;
#endif // POSE_DEBUG
	std::vector<cv::Point2f> points;
	for (size_t i = 0; i < final_joints.size(0); i++)
	{
		float x = final_joints[i][0].item<float>();
		float y = final_joints[i][1].item<float>();
		points.emplace_back(cv::Point2f(x, y));
	}
	out_landmarks.points = points;
	out_landmarks.flag = true;
}

void alpha::TorchPoseUtils::print_pretty_tensor(const at::Tensor& m, std::vector<int>& channel_indexs)
{
	if (!channel_indexs.empty())
	{
		for (int c = 0; c < channel_indexs.size(); c++)
		{
			for (int i = 0; i < m.size(1); i++)
			{
				for (int j = 0; j < m.size(2); j++)
				{
					printf("%f ", m[channel_indexs[c]][i][j].item<float>());
				}
				printf("\n");
			}
			std::cout << "------------------------" << std::endl;
		}

	}
	else
	{
		for (int c = 0; c < m.size(0); c++) // c
		{
			for (int i = 0; i < m.size(1); i++)
			{
				std::cout << m[c][i] << std::endl;
			}
			std::cout << "------------------------" << std::endl;
		}
	}
}
