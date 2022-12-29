#include "utils.h"
#include <cmath>

std::wstring utils::to_wstring(const std::string& str)
{
	unsigned len = str.size() * 2;
	setlocale(LC_CTYPE, "");
	wchar_t* p = new wchar_t[len];
	mbstowcs(p, str.c_str(), len);
	std::wstring wstr(p);
	delete[] p;
	return wstr;
}

std::string utils::to_string(const std::wstring& wstr)
{
	unsigned len = wstr.size() * 4;
	setlocale(LC_CTYPE, "");
	char* p = new char[len];
	wcstombs(p, wstr.c_str(), len);
	std::string str(p);
	delete[] p;
	return str;
}

types::CenterScale<float> utils::box_to_center_scale(types::Boxf& box, float aspect_ratio, float scale_mult)
{
	float x = box.x1;
	float y = box.y1;
	float w = box.width();
	float h = box.height();
	float pixel_std = 1.0;
	float center_x = x + w * 0.5;
	float center_y = y + h * 0.5;
	if (w > aspect_ratio * h) {
		h = w / aspect_ratio;
	}
	else if (w < aspect_ratio * h)
	{
		w = h * aspect_ratio;
	}
	float scale_w = w * 1.0 / pixel_std;
	float scale_h = h * 1.0 / pixel_std;
	if (center_x != -1) {
		scale_w *= scale_mult;
		scale_h *= scale_mult;
	}
	types::CenterScale<float> cs;
	cs.center_x = center_x;
	cs.center_y = center_y;
	cs.scale_w = scale_w;
	cs.scale_h = scale_h;
	return cs;
}

void utils::center_scale_to_box(types::CenterScale<float>& cs, types::Boxf& box)
{
	float pixel_std = 1.0;
	float w = cs.scale_w * pixel_std;
	float h = cs.scale_h * pixel_std;
	float x1 = cs.center_x - w * 0.5;
	float y1 = cs.center_y - h * 0.5;
	float x2 = x1 + w;
	float y2 = y1 + h;
	box.x1 = x1;
	box.y1 = y1;
	box.x2 = x2;
	box.y2 = y2;
}

cv::Mat utils::get_affine_transform(types::CenterScale<float>& center_scale , float output_w, float output_h, float rot, float shift_x, float shift_y, bool inverse)
{
	// rotate the point by rot degree
	float rot_rad = rot * 3.14159 / 180;
	float sn = sin(rot_rad);
	float cs = cos(rot_rad);

	std::array<float, 2> src_point = {0, center_scale.scale_w * (-0.5)};
	std::array<float, 2> dst_point = { 0, output_w * (-0.5) };

	std::array<float, 2> src_rotate = { src_point[0] * cs - src_point[1] * sn, src_point[0] * sn - src_point[1] * cs };

	cv::Point2f srcTri[3];
	srcTri[0] = cv::Point2f(center_scale.center_x + center_scale.scale_w*shift_x, center_scale.center_y + center_scale.scale_h * shift_y);
	srcTri[1] = cv::Point2f(center_scale.center_x + src_rotate[0] +center_scale.scale_w * shift_x,
		center_scale.center_y + src_rotate[1] + center_scale.scale_h * shift_y);
	std::array<float, 2> src_direct = { srcTri[0].x - srcTri[1].x, srcTri[0].y - srcTri[1].y };
	srcTri[2] = cv::Point2f(srcTri[1].x + (-src_direct[1]), srcTri[1].y + (src_direct[0]));

	cv::Point2f dstTri[3];
	dstTri[0] = cv::Point2f(output_w*0.5, output_h*0.5);
	dstTri[1] = cv::Point2f(dstTri[0].x + dst_point[0], dstTri[0].y + dst_point[1]);
	std::array<float, 2> dst_direct = { dstTri[0].x - dstTri[1].x, dstTri[0].y - dstTri[1].y };
	dstTri[2] = cv::Point2f(dstTri[1].x + (-dst_direct[1]), dstTri[1].y + (dst_direct[0]));

	if (inverse)
	{
		cv::Mat warp_mat = getAffineTransform(dstTri, srcTri);
		return warp_mat;
	}
	else
	{
		cv::Mat warp_mat = getAffineTransform(srcTri, dstTri);
		return warp_mat;
	}
	
}

// draw functions
void utils::draw_landmarks_inplace(cv::Mat& mat, types::Landmarks& landmarks)
{
	if (landmarks.points.empty() || !landmarks.flag) return;
	for (const auto& point : landmarks.points)
		cv::circle(mat, point, 2, cv::Scalar(0, 255, 0), -1);
}

void utils::draw_boxes_inplace(cv::Mat& mat_inplace, const std::vector<types::Boxf>& boxes)
{
	if (boxes.empty()) return;
	for (const auto& box : boxes)
	{
		if (box.flag)
		{
			cv::rectangle(mat_inplace, box.rect(), cv::Scalar(255, 255, 0), 2);
			if (box.label_text)
			{
				std::string label_text(box.label_text);
				label_text = label_text + ":" + std::to_string(box.score).substr(0, 4);
				cv::putText(mat_inplace, label_text, box.tl(), cv::FONT_HERSHEY_SIMPLEX,
					0.6f, cv::Scalar(0, 255, 0), 2);

			}
		}
	}
}

void utils::draw_boxes_with_landmarks_inplace(cv::Mat& mat_inplace, const std::vector<types::BoxfWithLandmarks>& boxes_kps, bool text) {
	if (boxes_kps.empty()) return;
	for (const auto& box_kps : boxes_kps)
	{
		if (box_kps.flag)
		{
			// box
			if (box_kps.box.flag)
			{
				cv::rectangle(mat_inplace, box_kps.box.rect(), cv::Scalar(255, 255, 0), 2);
				if (box_kps.box.label_text && text)
				{
					std::string label_text(box_kps.box.label_text);
					label_text = label_text + ":" + std::to_string(box_kps.box.score).substr(0, 4);
					cv::putText(mat_inplace, label_text, box_kps.box.tl(), cv::FONT_HERSHEY_SIMPLEX,
						0.6f, cv::Scalar(0, 255, 0), 2);

				}
			}
			// landmarks
			if (box_kps.landmarks.flag && !box_kps.landmarks.points.empty())
			{
				for (const auto& point : box_kps.landmarks.points)
					cv::circle(mat_inplace, point, 2, cv::Scalar(0, 255, 0), -1);
			}
		}
	}
}


//softmax
std::vector<float> utils::softmax(const float* logits, unsigned int _size, unsigned int& max_id) {
	if (_size == 0 || logits == nullptr) return {};
	float max_prob = 0.f, total_exp = 0.f;
	std::vector<float> softmax_probs(_size);
	for (unsigned int i = 0; i < _size; ++i)
	{
		softmax_probs[i] = std::exp((float)logits[i]);
		total_exp += softmax_probs[i];
	}
	for (unsigned int i = 0; i < _size; ++i)
	{
		softmax_probs[i] = softmax_probs[i] / total_exp;
		if (softmax_probs[i] > max_prob)
		{
			max_id = i;
			max_prob = softmax_probs[i];
		}
	}
	return softmax_probs;
}

void utils::blending_nms(std::vector<types::Boxf>& input, std::vector<types::Boxf>& output,
	float iou_threshold, unsigned int topk)
{
	if (input.empty()) return;
	std::sort(input.begin(), input.end(),
		[](const types::Boxf& a, const types::Boxf& b)
		{ return a.score > b.score; });
	const unsigned int box_num = input.size();
	std::vector<int> merged(box_num, 0);

	unsigned int count = 0;
	for (unsigned int i = 0; i < box_num; ++i)
	{
		if (merged[i]) continue;
		std::vector<types::Boxf> buf;

		buf.push_back(input[i]);
		merged[i] = 1;

		for (unsigned int j = i + 1; j < box_num; ++j)
		{
			if (merged[j]) continue;

			float iou = static_cast<float>(input[i].iou_of(input[j]));
			if (iou > iou_threshold)
			{
				merged[j] = 1;
				buf.push_back(input[j]);
			}
		}

		float total = 0.f;
		for (unsigned int k = 0; k < buf.size(); ++k)
		{
			total += std::exp(buf[k].score);
		}
		types::Boxf rects;
		for (unsigned int l = 0; l < buf.size(); ++l)
		{
			float rate = std::exp(buf[l].score) / total;
			rects.x1 += buf[l].x1 * rate;
			rects.y1 += buf[l].y1 * rate;
			rects.x2 += buf[l].x2 * rate;
			rects.y2 += buf[l].y2 * rate;
			rects.score += buf[l].score * rate;
		}
		rects.flag = true;
		output.push_back(rects);

		// keep top k
		count += 1;
		if (count >= topk)
			break;
	}
}

// reference: https://github.com/ultralytics/yolov5/blob/master/utils/general.py
void utils::offset_nms(std::vector<types::Boxf>& input, std::vector<types::Boxf>& output,
	float iou_threshold, unsigned int topk)
{
	if (input.empty()) return;
	std::sort(input.begin(), input.end(),
		[](const types::Boxf& a, const types::Boxf& b)
		{ return a.score > b.score; });
	const unsigned int box_num = input.size();
	std::vector<int> merged(box_num, 0);

	const float offset = 4096.f;
	/** Add offset according to classes.
	 * That is, separate the boxes into categories, and each category performs its
	 * own NMS operation. The same offset will be used for those predicted to be of
	 * the same category. Therefore, the relative positions of boxes of the same
	 * category will remain unchanged. Box of different classes will be farther away
	 * after offset, because offsets are different. In this way, some overlapping but
	 * different categories of entities are not filtered out by the NMS. Very clever!
	 */
	for (unsigned int i = 0; i < box_num; ++i)
	{
		input[i].x1 += static_cast<float>(input[i].label) * offset;
		input[i].y1 += static_cast<float>(input[i].label) * offset;
		input[i].x2 += static_cast<float>(input[i].label) * offset;
		input[i].y2 += static_cast<float>(input[i].label) * offset;
	}

	unsigned int count = 0;
	for (unsigned int i = 0; i < box_num; ++i)
	{
		if (merged[i]) continue;
		std::vector<types::Boxf> buf;

		buf.push_back(input[i]);
		merged[i] = 1;

		for (unsigned int j = i + 1; j < box_num; ++j)
		{
			if (merged[j]) continue;

			float iou = static_cast<float>(input[i].iou_of(input[j]));

			if (iou > iou_threshold)
			{
				merged[j] = 1;
				buf.push_back(input[j]);
			}

		}
		output.push_back(buf[0]);

		// keep top k
		count += 1;
		if (count >= topk)
			break;
	}

	/** Substract offset.*/
	if (!output.empty())
	{
		for (unsigned int i = 0; i < output.size(); ++i)
		{
			output[i].x1 -= static_cast<float>(output[i].label) * offset;
			output[i].y1 -= static_cast<float>(output[i].label) * offset;
			output[i].x2 -= static_cast<float>(output[i].label) * offset;
			output[i].y2 -= static_cast<float>(output[i].label) * offset;
		}
	}

}

// reference: https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB/
//            blob/master/ncnn/src/UltraFace.cpp
void utils::hard_nms(std::vector<types::Boxf>& input, std::vector<types::Boxf>& output,
	float iou_threshold, unsigned int topk)
{
	if (input.empty()) return;
	std::sort(input.begin(), input.end(),
		[](const types::Boxf& a, const types::Boxf& b)
		{ return a.score > b.score; });
	const unsigned int box_num = input.size();
	std::vector<int> merged(box_num, 0);

	unsigned int count = 0;
	for (unsigned int i = 0; i < box_num; ++i)
	{
		if (merged[i]) continue;
		std::vector<types::Boxf> buf;

		buf.push_back(input[i]);
		merged[i] = 1;

		for (unsigned int j = i + 1; j < box_num; ++j)
		{
			if (merged[j]) continue;

			float iou = static_cast<float>(input[i].iou_of(input[j]));

			if (iou > iou_threshold)
			{
				merged[j] = 1;
				buf.push_back(input[j]);
			}

		}
		output.push_back(buf[0]);

		// keep top k
		count += 1;
		if (count >= topk)
			break;
	}
}

//sort 增序：由小到大
template<typename T> std::vector<unsigned int> utils::argsort(const std::vector<T>& arr)
{
	types::__assert_type<T>();
	if (arr.empty()) return {};
	const unsigned int _size = arr.size();
	std::vector<unsigned int > indices;
	for (unsigned int i = 0; i < _size; ++i) {
		indices.push_back(i);
	}
	std::sort(indices.begin(), indices.end(), [&arr](const unsigned int a, const unsigned int b) {
		return arr[a] < arr[b];
		});
	return indices;
}

template<typename T> T utils::clip(T v, T min_v, T max_v) {
	if (v < min_v) {
		return min_v;
	}
	else if (v > max_v) {
		return max_v;
	}
	else {
		return v;
	}
}