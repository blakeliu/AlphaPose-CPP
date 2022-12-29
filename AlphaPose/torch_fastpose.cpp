#include "torch_fastpose.h"
#include "utils.h"

FastPose::FastPose(const std::string& _weight_path, unsigned int _num_threads, 
	int _batch_size, int _num_joints,
	int _input_height, int _input_width,
	int _heatmap_channel,
	int _heatmap_height, int _heatmap_width
	):
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
	std::cout << "Init Inter-op num threads:" << at::get_num_interop_threads() << ", Intra-op num threads" << at::get_num_threads() << std::endl;
#endif // POSE_DEBUG
	at::init_num_threads();
	at::set_num_threads(num_threads);
	at::set_num_interop_threads(num_threads);
#ifdef POSE_DEBUG
	std::cout << "Seted Inter-op num threads:" << at::get_num_interop_threads() << ", Intra-op num threads" << at::get_num_threads() << std::endl;
#endif // POSE_DEBUG

	aspect_ratio = input_width / input_height;
	feat_stride[0] = input_height / heatmap_height;
	feat_stride[1] = input_width / heatmap_width;
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
	int feat_dim = hm_1d.size(1);
	at::Tensor range_tensor = at::arange(feat_dim, at::kFloat);
	hm_1d *= range_tensor;
}

void FastPose::integral_tensor(at::Tensor& preds, types::Boxf cropped_box)
{
}

void FastPose::generate_landmarks(at::Tensor& heatmap, types::Boxf cropped_box, types::Landmarks& out_landmarks)
{
}

void FastPose::detect(const cv::Mat& mat, std::vector<types::Boxf>& detected_boxes, std::vector<types::BoxfWithLandmarks>& person_lds)
{
	for (size_t i = 0; i < detected_boxes.size(); i++)
	{
		types::Boxf box = detected_boxes[i];
		if (box.label == 0)
		{
			std::vector<float> center;
			std::vector<float> scale;

			// 1. crop box

			utils::get_affine_transform();
#ifdef POSE_DEBUG
			utils::Timer crop_t;
#endif // POSE_DEBUG
			cv::Mat mat_rs;
			types::Boxf cropped_box;
#ifdef POSE_DEBUG
			std::cout << "Crop person " << i << " box mat time: " << crop_t.count() << std::endl;
#endif // POSE_DEBUG

			// 2. make input tensor
#ifdef POSE_DEBUG
			utils::Timer trans_t;
#endif // POSE_DEBUG
			at::Tensor mat_tensor;
			this->transform(mat_rs, mat_tensor);
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
			types::BoxfWithLandmarks person_box_lds;
			person_box_lds.box = box;
			person_box_lds.landmarks = pose_lds;
			person_box_lds.flag = true;
		}
	}
}
