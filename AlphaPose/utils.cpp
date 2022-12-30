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

void utils::normalize_inplace(cv::Mat& mat_inplace, const float* mean, const float* scale)
{
	if (mat_inplace.type() != CV_32FC3) mat_inplace.convertTo(mat_inplace, CV_32FC3);
	for (unsigned int i = 0; i < mat_inplace.rows; ++i)
	{
		cv::Vec3f* p = mat_inplace.ptr<cv::Vec3f>(i);
		for (unsigned int j = 0; j < mat_inplace.cols; ++j)
		{
			p[j][0] = (p[j][0] - mean[0]) * scale[0];
			p[j][1] = (p[j][1] - mean[1]) * scale[1];
			p[j][2] = (p[j][2] - mean[2]) * scale[2];
		}
	}
}

void utils::box_to_center_scale(types::Boxf& box, std::vector<float>& center, std::vector<float>& scale, float aspect_ratio, float scale_mult)
{
	float x = box.x1;
	float y = box.y1;
	float w = box.width();
	float h = box.height();
	float pixel_std = 1.0;
	center.push_back(x + w * 0.5f);
	center.push_back(y + h * 0.5f);
	if (w > aspect_ratio * h) {
		h = w / aspect_ratio;
	}
	else if (w < aspect_ratio * h)
	{
		w = h * aspect_ratio;
	}
	scale.push_back(w * 1.0f / pixel_std);
	scale.push_back(h * 1.0f / pixel_std);
	if (center[0] != -1) {
		scale[0] *= scale_mult;
		scale[1] *= scale_mult;
	}
}

void utils::center_scale_to_box(std::vector<float>& center, std::vector<float>& scale, types::Boxf& box)
{
	float pixel_std = 1.0;
	float w = scale[0] * pixel_std;
	float h = scale[1] * pixel_std;
	float x1 = center[0] - w * 0.5;
	float y1 = center[1] - h * 0.5;
	float x2 = x1 + w;
	float y2 = y1 + h;
	box.x1 = x1;
	box.y1 = y1;
	box.x2 = x2;
	box.y2 = y2;
}

void utils::affine_tranform(const float x, const float y, cv::Mat& trans_mat, std::vector<float>& out_pts)
{
	double p[3] = { x, y, 1.0f };
	cv::Mat mat_pt(3, 1, trans_mat.type(), p);
	cv::Mat w = trans_mat * mat_pt;
	double t_x = w.at<double>(0, 0);
	double t_y = w.at<double>(1, 0);
	out_pts.push_back(static_cast<float>(t_x));
	out_pts.push_back(static_cast<float>(t_y));
}

cv::Mat utils::get_affine_transform(const std::vector<float>& center, const std::vector<float>& scale, const std::vector<float>& shift, const float output_h, const float output_w, const float rot, const bool inverse)
{
	// rotate the point by rot degree
	float rot_rad = rot * M_PI / 180;
	float src_w = scale[0];
	std::vector<float> src_dir = get_dir(0, -0.5 * src_w, rot_rad);
	std::vector<float> dst_dir{ 0.0, float(-0.5) * output_w };

	cv::Point2f srcTri[3];
	srcTri[0] = cv::Point2f(center[0] + scale[0] * shift[0], center[1] + scale[1] * shift[1]);
	srcTri[1] = cv::Point2f(center[0] + src_dir[0] + scale[0] * shift[0], center[1] + src_dir[1] + scale[1] * shift[1]);
	srcTri[2] = get_3rd_point(srcTri[0], srcTri[1]);

	cv::Point2f dstTri[3];
	dstTri[0] = cv::Point2f(output_w * 0.5, output_h * 0.5);
	dstTri[1] = cv::Point2f(output_w * 0.5 + dst_dir[0], output_h * 0.5 + dst_dir[1]);
	dstTri[2] = get_3rd_point(dstTri[0], dstTri[1]);
	cv::Mat warp_mat;
	if (inverse)
	{
		warp_mat = getAffineTransform(dstTri, srcTri);
	}
	else
	{
		warp_mat = getAffineTransform(srcTri, dstTri);
	}
	return warp_mat;

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

void utils::draw_pose_box_with_landmasks(cv::Mat& mat_inplace, const std::vector<types::BoxfWithLandmarks>& boxes_kps, int num_joints)
{
	if (boxes_kps.empty()) return;
	std::vector<std::array<int, 2>> l_pair;
	std::vector<std::array<int, 3>> p_color;
	std::vector<std::array<int, 3>> line_color;

	int len_pair = 0;
	int len_pcolor = 0;
	int len_lcolor = 0;

	if (num_joints == 136)
	{
		int _pair[][2] = {
			{0, 1}, {0, 2}, {1, 3}, {2, 4},  // Head
			{5, 18}, {6, 18}, {5, 7}, {7, 9}, {6, 8}, {8, 10},// Body
			{17, 18}, {18, 19}, {19, 11}, {19, 12},
			{11, 13}, {12, 14}, {13, 15}, {14, 16},
			{20, 24}, {21, 25}, {23, 25}, {22, 24}, {15, 24}, {16, 25},// Foot
			{26, 27},{27, 28},{28, 29},{29, 30},{30, 31},{31, 32},{32, 33},{33, 34},{34, 35},{35, 36},{36, 37},{37, 38},//Face
			{38, 39},{39, 40},{40, 41},{41, 42},{43, 44},{44, 45},{45, 46},{46, 47},{48, 49},{49, 50},{50, 51},{51, 52},//Face
			{53, 54},{54, 55},{55, 56},{57, 58},{58, 59},{59, 60},{60, 61},{62, 63},{63, 64},{64, 65},{65, 66},{66, 67},//Face
			{68, 69},{69, 70},{70, 71},{71, 72},{72, 73},{74, 75},{75, 76},{76, 77},{77, 78},{78, 79},{79, 80},{80, 81},//Face
			{81, 82},{82, 83},{83, 84},{84, 85},{85, 86},{86, 87},{87, 88},{88, 89},{89, 90},{90, 91},{91, 92},{92, 93},//Face
			{94,95},{95,96},{96,97},{97,98},{94,99},{99,100},{100,101},{101,102},{94,103},{103,104},{104,105},//LeftHand
			{105,106},{94,107},{107,108},{108,109},{109,110},{94,111},{111,112},{112,113},{113,114},//LeftHand
			{115,116},{116,117},{117,118},{118,119},{115,120},{120,121},{121,122},{122,123},{115,124},{124,125},//RightHand
			{125,126},{126,127},{115,128},{128,129},{129,130},{130,131},{115,132},{132,133},{133,134},{134,135}//RightHand
		};
		int _pcolor[][3] = { {0, 255, 255}, {0, 191, 255}, {0, 255, 102}, {0, 77, 255}, {0, 255, 0},  // Nose, LEye, REye, LEar, REar
				   {77, 255, 255}, {77, 255, 204}, {77, 204, 255}, {191, 255, 77}, {77, 191, 255}, {191, 255, 77},  // LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
				   {204, 77, 255}, {77, 255, 204}, {191, 77, 255}, {77, 255, 191}, {127, 77, 255}, {77, 255, 127},  // LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
				   {77, 255, 255}, {0, 255, 255}, {77, 204, 255},  // head, neck, shoulder
				   {0, 255, 255}, {0, 191, 255}, {0, 255, 102}, {0, 77, 255}, {0, 255, 0}, {77, 255, 255} }; // foot
		int _lcolor[][3] = { {0, 215, 255}, {0, 255, 204}, {0, 134, 255}, {0, 255, 50},
					  {0, 255, 102}, {77, 255, 222}, {77, 196, 255}, {77, 135, 255}, {191, 255, 77}, {77, 255, 77},
					  {77, 191, 255}, {204, 77, 255}, {77, 222, 255}, {255, 156, 127},
					  {0, 127, 255}, {255, 127, 77}, {0, 77, 255}, {255, 77, 36},
					  {0, 77, 255}, {0, 77, 255}, {0, 77, 255}, {0, 77, 255}, {255, 156, 127}, {255, 156, 127} };

		len_pair = sizeof(_pair) / sizeof(_pair[0]);
		len_pcolor = sizeof(_pcolor) / sizeof(_pcolor[0]);
		len_lcolor = sizeof(_lcolor) / sizeof(_lcolor[0]);
		l_pair.reserve(len_pair);
		p_color.reserve(len_pcolor);
		line_color.reserve(len_lcolor);

		for (auto& row : _pair) {
			std::array<int, 2> p = { row[0], row[1] };
			l_pair.push_back(p);
		}
		for (auto& row : _pcolor) {
			std::array<int, 3> p = { row[0], row[1], row[2] };
			p_color.push_back(p);
		}
		for (auto& row : _lcolor) {
			std::array<int, 3> p = { row[0], row[1], row[2] };
			line_color.push_back(p);
		}
	}
	else if (num_joints == 26)
	{
		int _pair[][2] = {
			{0, 1}, {0, 2}, {1, 3}, {2, 4},  // Head
			{5, 18}, {6, 18}, {5, 7}, {7, 9}, {6, 8}, {8, 10},// Body
			{17, 18}, {18, 19}, {19, 11}, {19, 12},
			{11, 13}, {12, 14}, {13, 15}, {14, 16},
			{20, 24}, {21, 25}, {23, 25}, {22, 24}, {15, 24}, {16, 25} }; // Foot
		int _pcolor[][3] = { {0, 255, 255}, {0, 191, 255}, {0, 255, 102}, {0, 77, 255}, {0, 255, 0},  // Nose, LEye, REye, LEar, REar
					{77, 255, 255}, {77, 255, 204}, {77, 204, 255}, {191, 255, 77}, {77, 191, 255}, {191, 255, 77},  // LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist
					{204, 77, 255}, {77, 255, 204}, {191, 77, 255}, {77, 255, 191}, {127, 77, 255}, {77, 255, 127},  // LHip, RHip, LKnee, Rknee, LAnkle, RAnkle, Neck
					{77, 255, 255}, {0, 255, 255}, {77, 204, 255},  // head, neck, shoulder
					{0, 255, 255}, {0, 191, 255}, {0, 255, 102}, {0, 77, 255}, {0, 255, 0}, {77, 255, 255} }; // foot

		int _lcolor[][3] = { {0, 215, 255}, {0, 255, 204}, {0, 134, 255}, {0, 255, 50},
						{0, 255, 102}, {77, 255, 222}, {77, 196, 255}, {77, 135, 255}, {191, 255, 77}, {77, 255, 77},
						{77, 191, 255}, {204, 77, 255}, {77, 222, 255}, {255, 156, 127},
						{0, 127, 255}, {255, 127, 77}, {0, 77, 255}, {255, 77, 36},
						{0, 77, 255}, {0, 77, 255}, {0, 77, 255}, {0, 77, 255}, {255, 156, 127}, {255, 156, 127} };
		len_pair = sizeof(_pair) / sizeof(_pair[0]);
		len_pcolor = sizeof(_pcolor) / sizeof(_pcolor[0]);
		len_lcolor = sizeof(_lcolor) / sizeof(_lcolor[0]);
		l_pair.reserve(len_pair);
		p_color.reserve(len_pcolor);
		line_color.reserve(len_lcolor);

		for (auto& row : _pair) {
			std::array<int, 2> p = { row[0], row[1] };
			l_pair.push_back(p);
		}
		for (auto& row : _pcolor) {
			std::array<int, 3> p = { row[0], row[1], row[2] };
			p_color.push_back(p);
		}
		for (auto& row : _lcolor) {
			std::array<int, 3> p = { row[0], row[1], row[2] };
			line_color.push_back(p);
		}
	}
	int h = mat_inplace.rows;
	int w = mat_inplace.cols;
	std::vector<std::array<int, 2>> part_line;
	for (const auto& box_kps : boxes_kps)
	{
		cv::Scalar color = cv::Scalar(255, 0, 0);
		cv::rectangle(mat_inplace, box_kps.box.rect(), color, 2); //draw box
		for (size_t i = 0; i < box_kps.landmarks.points.size(); i++) //draw points
		{
			cv::Point2f p = box_kps.landmarks.points[i];
			std::array<int, 2> _pl = { static_cast<int>(p.x), static_cast<int>(p.y) };
			part_line.push_back(_pl);
			if (i < len_pcolor)
			{
				cv::circle(mat_inplace, cv::Point(_pl[0], _pl[1]), 2, cv::Scalar(p_color[i][0], p_color[i][1], p_color[i][2]), -1);
			}
			else
			{
				cv::circle(mat_inplace, cv::Point(_pl[0], _pl[1]), 1, cv::Scalar(255, 255, 255), 2);
			}
			
		}
		int len_part_line = part_line.size();
		for (size_t i = 0; i < l_pair.size(); i++)
		{
			int start_p = l_pair[i][0];
			int end_p = l_pair[i][1];
			if (start_p < len_part_line && end_p < len_part_line)
			{
				auto start_xy = part_line[start_p];
				auto end_xy = part_line[end_p];
				float X[2] = { start_xy[0], end_xy[0] };
				float Y[2] = { start_xy[1], end_xy[1] };
				float mX = (X[0] + X[1]) * 0.5f;
				float mY = (Y[0] + Y[1]) * 0.5f;

				float length = pow((pow(Y[0]- Y[1], 2) + pow(X[0]- X[1], 2)), 0.5);
				float angle = atan2f(static_cast<double>(Y[0] - Y[1]), static_cast<double>(X[0] - X[1])) * 180.f / M_PI;

				std::vector<cv::Point> polygon;
				cv::ellipse2Poly(cv::Point(int(mX), int(mY)), cv::Size(int(length / 2), 1), int(angle), 0, 360, 1, polygon);
				if (i < len_lcolor)
				{
					cv::fillConvexPoly(mat_inplace, polygon, cv::Scalar(line_color[i][0], line_color[i][1], line_color[i][2]));
				}
				else
				{
					cv::line(mat_inplace, cv::Point(start_xy[0], start_xy[1]), cv::Point(end_xy[0], end_xy[1]), cv::Scalar(255, 255, 255), 1);
				}

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