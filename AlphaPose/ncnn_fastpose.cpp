#include "pose_utils.h"
#include <omp.h>
#include "ncnn_fastpose.h"
#include "utils.h"


using namespace alpha;

NCNNFastPose::NCNNFastPose(const std::string& _param_path, const std::string& _bin_path,
	unsigned int _num_threads, int _batch_size, int _num_joints, bool _use_vulkan, int _input_height, int _input_width,
	int _heatmap_channel, int _heatmap_height, int _heatmap_width) :
	BasicNCNNHandler(_param_path, _bin_path, _num_threads),
	batch_size(_batch_size), num_joints(_num_joints), input_height(_input_height), input_width(_input_width),
	heatmap_channel(_heatmap_channel), heatmap_height(_heatmap_height), heatmap_width(_heatmap_width),
	use_vulkan(_use_vulkan)
{
	this->initialize_handler();
	aspect_ratio = static_cast<float>(input_width) / static_cast<float>(input_height);
	feat_stride[0] = static_cast<float>(input_height) / static_cast<float>(heatmap_height);
	feat_stride[1] = static_cast<float>(input_width) / static_cast<float>(heatmap_width);

}

NCNNFastPose::~NCNNFastPose()
{
	if (net) delete net;
	net = nullptr;
}

void NCNNFastPose::initialize_handler()
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
		std::cerr << "NCNNFastPose load failed: " << msg << std::endl;
		throw std::runtime_error(msg);
	}
#ifdef POSE_DEBUG
	this->print_debug_string();
#endif
}

void NCNNFastPose::transform(const cv::Mat& mat_rs, ncnn::Mat& in)
{
	//mat_rs.convertTo(mat_rs, CV_32FC3);
	in = ncnn::Mat::from_pixels(mat_rs.data, ncnn::Mat::PIXEL_BGR, mat_rs.cols, mat_rs.rows);
	// (x-mean)*std
	in.substract_mean_normalize(mean_vals, norm_vals);
}

void NCNNFastPose::crop_image(const cv::Mat& input_mat, cv::Mat& crop_mat, types::Boxf& detected_box, types::Boxf& cropped_box)
{
	std::vector<float> center; // x y 
	std::vector<float> scale; // w h
	std::vector<float> shift = { 0.f, 0.f };
	utils::box_to_center_scale(detected_box, center, scale, aspect_ratio);

	cv::Mat trans = utils::get_affine_transform(center, scale, shift, input_height, input_width, 0);
	cv::warpAffine(input_mat, crop_mat, trans, cv::Size(input_width, input_height), cv::INTER_LINEAR);

	utils::center_scale_to_box(center, scale, cropped_box);
}

void NCNNFastPose::detect(const cv::Mat& image, std::vector<types::Boxf>& detected_boxes, std::vector<types::BoxfWithLandmarks>& person_lds)
{
	for (size_t i = 0; i < detected_boxes.size(); i++)
	{
		types::Boxf box = detected_boxes[i];
		if (box.label == 0)
		{
			// 1. crop box mat
#ifdef POSE_TIMER
			utils::Timer pre_t;
#endif // POSE_TIMER
			cv::Mat mat_person;
			types::Boxf cropped_box;
			crop_image(image, mat_person, box, cropped_box);
			// 2. make input tensor
			ncnn::Mat input;
			this->transform(mat_person, input);
#ifdef POSE_TIMER
			std::cout << "Preprocessed image tensor " << i << " time: " << pre_t.count() << std::endl;
#endif // POSE_TIMER

#ifdef POSE_DEBUG
			std::string p_name = std::string("person-") + std::to_string(person_lds.size()) + std::string(".jpg");
			cv::imwrite(p_name, mat_person);
			std::vector<int> mat_channel_indexs{ 0 };
			//this->print_pretty_mat(input, mat_channel_indexs);
#endif // POSE_DEBUG

			// 3. forward
#ifdef POSE_TIMER
			utils::Timer forward_t;
#endif // POSE_TIMER
			ncnn::Mat output;
			auto extractor = net->create_extractor();
			extractor.set_light_mode(false);  // default
			extractor.set_num_threads(num_threads);
			extractor.input("input", input);
			extractor.extract("output", output);

#ifdef POSE_DEBUG
			std::cout << "output c: " << output.c << ", d: " << output.d << ", w: " << output.w << ", h: " << output.h << std::endl;
			//std::vector<int> hm_channel_indexs{ 0, 135 };
			//this->print_pretty_mat(output, hm_channel_indexs);
#endif // POSE_DEBUG

			std::vector<cv::Mat> vec_mat_hms(output.c);
			cv::Mat pack_mat_hms;
			for (int i = 0; i < output.c; i++)
			{
				vec_mat_hms[i] = cv::Mat(output.h, output.w, CV_32FC1);
				memcpy((uchar*)vec_mat_hms[i].data, output.channel(i), static_cast<size_t>(output.w) * static_cast<size_t>(output.h) * sizeof(float));
			}
			cv::merge(vec_mat_hms, pack_mat_hms);
#ifdef POSE_DEBUG
			//std::cout << vec_mat_hms[int(output.c) - 1] << std::endl;
			std::cout << "pack_mat_hms c: " << pack_mat_hms.channels() << ", d: " << pack_mat_hms.dims << ", w: " << pack_mat_hms.cols << ", h: " << pack_mat_hms.rows << std::endl;
#endif // POSE_DEBUG
			at::Tensor heatmaps = at::from_blob(pack_mat_hms.data, { 1, pack_mat_hms.rows, pack_mat_hms.cols, pack_mat_hms.channels() }, at::TensorOptions().dtype(at::kFloat));
			heatmaps = heatmaps.permute({ 0, 3, 1, 2 });
#ifdef POSE_TIMER
			std::cout << "ncnn fastpose forward " << i << " time: " << forward_t.count() << std::endl;
#endif // POSE_TIMER
#ifdef POSE_DEBUG
			std::cout << "heatmap size: " << heatmaps.sizes() << std::endl;
			//std::vector<int> t_channel_indexs{ 0, 135 };
			//TorchPoseUtils::print_pretty_tensor(heatmaps[0], t_channel_indexs);
#endif // POSE_DEBUG

			// 4. generate landmarks
#ifdef POSE_TIMER
			utils::Timer gen_t;
#endif // POSE_TIMER
			types::Landmarks pose_lds;
			if (num_joints == 136)
			{
				TorchPoseUtils::heatmap_to_coord_simple_regress(num_threads, num_joints, heatmaps, cropped_box, pose_lds);
			}
			else if (num_joints == 26)
			{
				TorchPoseUtils::heatmap_to_coord_simple(num_threads, heatmaps, cropped_box, pose_lds);
			}
			else
			{
				std::stringstream ss;
				ss << "num joints must be 136 or 26, but got" << num_joints;
				throw std::runtime_error(ss.str());
			}
#ifdef POSE_TIMER
			std::cout << "Generate landmarks " << i << " time: " << gen_t.count() << std::endl;
#endif // POSE_TIMER
			types::BoxfWithLandmarks person_box_ld;
			person_box_ld.box = box;
			person_box_ld.landmarks = pose_lds;
			person_box_ld.flag = true;
			person_lds.push_back(person_box_ld);
		}
	}
}

void NCNNFastPose::warm_up(int count)
{
	BasicNCNNHandler::base_warm_up(input_height, input_width, 3, count);
}
