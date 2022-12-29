#include "ort_fastpose.h"
#include "ort_utils.h"
#include "utils.h"

Ort::Value ort::OrtFastPose::transform(const cv::Mat& mat_rs)
{
	cv::Mat canvas = mat_rs.clone();
	ortcv::utils::normalize_inplace(canvas, mean_vals, scale_vals);
	return ortcv::utils::create_tensor(
		canvas, input_node_dims, memory_info_handler,
		input_values_handler, ortcv::utils::CHW);
}

void ort::OrtFastPose::resize_unscale(const cv::Mat& mat, cv::Mat& mat_rs, int target_height, int target_width, ScaleParams& scale_params)
{
	if (mat.empty()) return;
	int img_height = static_cast<int>(mat.rows);
	int img_width = static_cast<int>(mat.cols);

	mat_rs = cv::Mat(target_height, target_width, CV_8UC3,
		cv::Scalar(114, 114, 114));
	// scale ratio (new / old) new_shape(h,w)
	float w_r = (float)target_width / (float)img_width;
	float h_r = (float)target_height / (float)img_height;
	float r = std::min(w_r, h_r);
	// compute padding
	int new_unpad_w = static_cast<int>((float)img_width * r); // floor
	int new_unpad_h = static_cast<int>((float)img_height * r); // floor
	int pad_w = target_width - new_unpad_w; // >=0
	int pad_h = target_height - new_unpad_h; // >=0

	int dw = pad_w / 2;
	int dh = pad_h / 2;

	// resize with unscaling
	cv::Mat new_unpad_mat;
	// cv::Mat new_unpad_mat = mat.clone(); // may not need clone.
	cv::resize(mat, new_unpad_mat, cv::Size(new_unpad_w, new_unpad_h));
	new_unpad_mat.copyTo(mat_rs(cv::Rect(dw, dh, new_unpad_w, new_unpad_h)));

	// record scale params.
	scale_params.r = r;
	scale_params.dw = dw;
	scale_params.dh = dh;
	scale_params.new_unpad_w = new_unpad_w;
	scale_params.new_unpad_h = new_unpad_h;
	scale_params.flag = true;
}

void ort::OrtFastPose::crop_boxes_imgs(const cv::Mat& mat, std::vector<types::Boxf>& detected_boxes, std::vector<cv::Mat>& imgs, std::vector<types::Boxf>& person_boxes)
{
	for (size_t i = 0; i < detected_boxes.size(); i++) {
		types::Boxf box = detected_boxes[i];
		//¹ýÂËperson
		if (box.label == 0) {
			types::CenterScale<float> cs = utils::box_to_center_scale(box, aspect_ratio);
			cv::Mat wrap_mat = utils::get_affine_transform(cs, static_cast<float>(input_size[0]), static_cast<float>(input_size[1]));

			cv::Mat warp_dst = cv::Mat::zeros(input_size[0], input_size[1], mat.type());
			cv::warpAffine(mat, warp_dst, wrap_mat, warp_dst.size());
			imgs.push_back(std::move(warp_dst));
			person_boxes.push_back(box);
		}
	}
}
