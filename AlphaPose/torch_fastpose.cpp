#include "torch_fastpose.h"
#include "utils.h"

FastPose::FastPose(const std::string& _weight_path, unsigned int _num_threads,
	int _batch_size, int _num_joints,
	int _input_height, int _input_width,
	int _heatmap_channel,
	int _heatmap_height, int _heatmap_width
) :
	num_threads(_num_threads),
	input_height(_input_height), input_width(_input_width),
	batch_size(_batch_size), heatmap_channel(_heatmap_channel),
	heatmap_height(_heatmap_height), heatmap_width(_heatmap_width),
	num_joints(_num_joints)
{
	try {
		_model = std::make_unique<torch::jit::script::Module>(torch::jit::load(_weight_path.data(), at::kCPU));
	}
	catch (const c10::Error& e) {
		std::string msg = std::string(e.what_without_backtrace());
		std::cerr << "error: " << msg << std::endl;
		throw std::runtime_error(msg);
	}
	catch (...) {
		std::string msg = "Internal ERROR!";
		throw std::runtime_error(msg);
	}
	_model->eval();

#ifdef POSE_DEBUG
	std::cout << "Init Inter-op num threads:" << at::get_num_interop_threads() << ", Intra-op num threads:" << at::get_num_threads() << std::endl;
#endif // POSE_DEBUG
	at::init_num_threads();
	at::set_num_threads(num_threads);
	at::set_num_interop_threads(num_threads);
#ifdef POSE_DEBUG
	std::cout << "Set Inter-op num threads:" << at::get_num_interop_threads() << ", Intra-op num threads:" << at::get_num_threads() << std::endl;
#endif // POSE_DEBUG

	aspect_ratio = static_cast<float>(input_width) / static_cast<float>(input_height);
	feat_stride[0] = static_cast<float>(input_height) / static_cast<float>(heatmap_height);
	feat_stride[1] = static_cast<float>(input_width) / static_cast<float>(heatmap_width);
}

FastPose::~FastPose()
{
}

void FastPose::transform(cv::Mat& mat_rs, at::Tensor& tensor_out)
{
	mat_rs.convertTo(mat_rs, CV_32FC3);
	utils::normalize_inplace(mat_rs, mean_vals, norm_vals);
	tensor_out = at::from_blob(mat_rs.data, { 1, input_height, input_width, 3 }, at::TensorOptions().dtype(at::kFloat));
	tensor_out = tensor_out.to(at::kCPU).permute({ 0, 3, 1, 2 });
}

void FastPose::integral_op(at::Tensor& hm_1d)
{
	int feat_dim = hm_1d.size(-1);
	at::Tensor range_tensor = at::arange(feat_dim, at::kFloat);
	hm_1d *= range_tensor;
}

void FastPose::integral_tensor(at::Tensor& preds, at::Tensor& pred_joints, at::Tensor& pred_scores, const int num_joints, const int hm_height, const int hm_width)
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


void FastPose::crop_image(const cv::Mat& input_mat, cv::Mat& crop_mat, types::Boxf& detected_box, types::Boxf& cropped_box)
{
	std::vector<float> center; // x y 
	std::vector<float> scale; // w h
	std::vector<float> shift = { 0.f, 0.f };
	utils::box_to_center_scale(detected_box, center, scale, aspect_ratio);

	cv::Mat trans = utils::get_affine_transform(center, scale, shift, input_height, input_width, 0);
	cv::warpAffine(input_mat, crop_mat, trans, cv::Size(input_width, input_height), cv::INTER_LINEAR);

	utils::center_scale_to_box(center, scale, cropped_box);
}

void FastPose::transfrom_preds(at::Tensor& input_coord, std::vector<float>& output_coord, const std::vector<float>& center, const std::vector<float>& scale, const int hm_width, const int hm_height)
{
	const std::vector<float> shift = { 0.f, 0.f };
	cv::Mat trans = utils::get_affine_transform(center, scale, shift, hm_height, hm_width, 0, true);
	const float x = input_coord[0].item<float>();
	const float y = input_coord[1].item<float>();
	utils::affine_tranform(x, y, trans, output_coord);
}

void FastPose::fast_nms_pose(at::Tensor& pred_joints, at::Tensor& pred_scores, at::Tensor& final_joints, at::Tensor& final_scores)
{
	at::Tensor normed_scores = pred_scores / at::sum(pred_scores, 0);
	final_joints = at::mul(pred_joints, normed_scores.repeat({ 1, 1, 2 })).sum(0);
	final_scores = at::mul(pred_scores, normed_scores).sum(0);
}

void FastPose::generate_landmarks(at::Tensor& heatmap, types::Boxf cropped_box, types::Landmarks& out_landmarks)
{
	const int bs = heatmap.size(0);
	const int hm_channel = heatmap.size(1);
	const int hm_height = heatmap.size(2);
	const int hm_width = heatmap.size(3);
	at::Tensor pred_joints;
	at::Tensor pred_scores;
	integral_tensor(heatmap, pred_joints, pred_scores, num_joints, hm_height, hm_width);
#ifdef POSE_DEBUG
	std::cout << "pred_joints size: " << pred_joints.sizes() << std::endl;
	std::cout << "pred_joints type: " << pred_joints.options()<< std::endl;
	std::cout << "pred_scores size: " << pred_scores.sizes() << std::endl;
	std::cout << "pred_scores type: " << pred_scores.options() << std::endl;
#endif // POSE_DEBUG

	pred_joints.index({ "...", 0 }) = (pred_joints.index({ "...", 0 }) + 0.5) * hm_width;
	pred_joints.index({ "...", 1 }) = (pred_joints.index({ "...", 1 }) + 0.5) * hm_height;

	at::Tensor trans_joints = at::zeros_like(pred_joints);
	float w = cropped_box.width();
	float h = cropped_box.height();
	const std::vector<float> center = { cropped_box.x1 + w * 0.5f, cropped_box.y1 + h * 0.5f };
	const std::vector<float> scale = { w, h };
	for (size_t i = 0; i < trans_joints.size(0); i++)
	{
		for (size_t j = 0; j < trans_joints.size(1); j++)
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
	std::cout << "final_joints type: " << final_joints.options() << std::endl;
	std::cout << "final_scores size: " << final_scores.sizes() << std::endl;
	std::cout << "final_scores type: " << final_scores.options() << std::endl;
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

void FastPose::detect(const cv::Mat& image, std::vector<types::Boxf>& detected_boxes, std::vector<types::BoxfWithLandmarks>& person_lds)
{
	for (size_t i = 0; i < detected_boxes.size(); i++)
	{
		types::Boxf box = detected_boxes[i];
		if (box.label == 0)
		{
			// 1. crop box mat
#ifdef POSE_DEBUG
			utils::Timer crop_t;
#endif // POSE_DEBUG
			cv::Mat mat_person;
			types::Boxf cropped_box;
			crop_image(image, mat_person, box, cropped_box);
#ifdef POSE_DEBUG
			std::cout << "Crop person " << i << " box mat time: " << crop_t.count() << std::endl;
			std::string p_name = std::string("person-") + std::to_string(person_lds.size()) + std::string(".jpg");
			cv::imwrite(p_name, mat_person);
#endif // POSE_DEBUG

			// 2. make input tensor
#ifdef POSE_DEBUG
			utils::Timer trans_t;
#endif // POSE_DEBUG
			at::Tensor mat_tensor;
			this->transform(mat_person, mat_tensor);
#ifdef POSE_DEBUG
			std::cout << "Transform tensor " << i << " time: " << trans_t.count() << std::endl;
#endif // POSE_DEBUG

			// 3. forward
#ifdef POSE_DEBUG
			utils::Timer forward_t;
#endif // POSE_DEBUG
			std::vector<torch::jit::IValue> inputs;
			inputs.emplace_back(mat_tensor);
			at::Tensor heatmap = _model->forward(inputs).toTensor();
#ifdef POSE_DEBUG
			std::cout << "Torch forward " << i << " time: " << forward_t.count() << std::endl;
			std::cout << "heatmap size: " << heatmap.sizes() << std::endl;
#endif // POSE_DEBUG

			// 4. generate landmarks
#ifdef POSE_DEBUG
			utils::Timer gen_t;
#endif // POSE_DEBUG
			types::Landmarks pose_lds;
			generate_landmarks(heatmap, cropped_box, pose_lds);
#ifdef POSE_DEBUG
			std::cout << "Generate landmarks " << i << " time: " << gen_t.count() << std::endl;
#endif // POSE_DEBUG
			types::BoxfWithLandmarks person_box_ld;
			person_box_ld.box = box;
			person_box_ld.landmarks = pose_lds;
			person_box_ld.flag = true;
			person_lds.push_back(person_box_ld);
		}
	}
}

void FastPose::warm_up(int count)
{
	at::Tensor mat_tensor = at::rand({1, 3, input_height, input_width}).to(at::kFloat).to(at::kCPU);
	std::vector<torch::jit::IValue> inputs;
	inputs.emplace_back(mat_tensor);
	for (size_t i = 0; i < count; i++)
	{
		at::Tensor heatmap = _model->forward(inputs).toTensor();
	}
}
