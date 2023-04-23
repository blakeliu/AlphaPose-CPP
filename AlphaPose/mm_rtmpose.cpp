#include "mm_rtmpose.h"
#include "utils.h"

alpha::MMRTMPose::MMRTMPose(const std::string& _weight_path,
	unsigned int _num_threads,
	int _batch_size,
	int _num_joints,
	int _input_height,
	int _input_width) :
	num_threads(_num_threads),
	input_height(_input_height), input_width(_input_width),
	batch_size(_batch_size),
	num_joints(_num_joints)
{
	try {
		//_model = std::make_unique<mmdeploy::PoseDetector>(mmdeploy::Model(_weight_path), mmdeploy::Device(device_name));
		net = new mmdeploy::PoseDetector{ mmdeploy::Model(_weight_path), mmdeploy::Device(device_name) };
	}
	catch (...) {
		std::string msg = "Internal ERROR!";
		throw std::runtime_error(msg);
	}
}

alpha::MMRTMPose::~MMRTMPose()
{
	if (net) delete net;
	net = nullptr;
}



float alpha::MMRTMPose::check_scale(cv::Mat& input)
{
	float h = static_cast<float>(input.rows);
	float w = static_cast<float>(input.cols);
	float max_long_edge = std::max(input_height, input_width);
	float max_short_edge = std::min(input_height, input_width);

	float input_max_size = std::max(h, w);
	float scale_factor = 1.0f;
	if (input_max_size > max_long_edge) {
		scale_factor = std::min(max_long_edge / (std::max(h, w)), max_short_edge / (std::min(h, w)));
	}
	return scale_factor;
}

float alpha::MMRTMPose::resize(cv::Mat& input, cv::Mat& output)
{
	float h = static_cast<float>(input.rows);
	float w = static_cast<float>(input.cols);

	float scale_factor = check_scale(input);

	int new_w = int(w * scale_factor + 0.5);
	int new_h = int(h * scale_factor + 0.5);

	if (scale_factor != 1.0f)
	{
		cv::resize(input, output, cv::Size(new_w, new_h));
	}
	else
	{
		output = input;
	}
	return scale_factor;
}

void alpha::MMRTMPose::detect(const cv::Mat& image, std::vector<types::Boxf>& detected_boxes, std::vector<types::BoxfWithLandmarks>& person_lds)
{

	int height = image.rows;
	int width = image.cols;
	for (size_t i = 0; i < detected_boxes.size(); i++)
	{
		types::Boxf box = detected_boxes[i];
		if (box.label == 0)
		{
			//cv::Rect crop_rect = cv::Rect();
			cv::Mat crop_img = image(cv::Range(int(box.y1), int(box.y2)), cv::Range(int(box.x1), int(box.x2)));
			/*std::string img_file = "pics/person.png";
			cv::Mat img = cv::imread(img_file);
			if (img.empty()) {
				fprintf(stderr, "failed to load image: %s\n", img_file.c_str());
			}
			auto dets = net->Apply(img);

			for (int i = 0; i < dets[0].length; i++) {
				cv::circle(img, { (int)dets[0].point[i].x, (int)dets[0].point[i].y }, 1, { 0, 255, 0 }, 2);
			}
			cv::imwrite("person.png", img);*/

			cv::Mat resize_img;
			float scale = resize(crop_img, resize_img);
#ifdef POSE_TIMER
			utils::Timer infer_t;
#endif // POSE_TIMER
			auto res = net->Apply(resize_img);
#ifdef POSE_TIMER
			std::cout << "RTMPose infer time: " << infer_t.count() << std::endl;
#endif // POSE_TIMER

			/*for (int i = 0; i < res[0].length; i++) {
				cv::circle(resize_img, { (int)res[0].point[i].x, (int)res[0].point[i].y }, 1, { 0, 255, 0 }, 2);
			}*/
			//cv::imwrite("crop_img.png", resize_img);

			types::Landmarks pose_lds;
			for (size_t k = 0; k < res[0].length; k++)
			{

				int x = (int)(res[0].point[k].x / scale);
				int y = (int)(res[0].point[k].y / scale);

				x = (std::max)(0, x) + box.x1;
				x = (std::max)(0, (std::min)(x, width - 1));

				y = (std::max)(0, y) + box.y1;
				y = (std::max)(0, (std::min)(y, height - 1));
				//float x = int(res[0].point[k].x);
				//float y = int(res[0].point[k].y);
				pose_lds.points.emplace_back(cv::Point2f(x, y));
			}
			types::BoxfWithLandmarks person_box_ld;
			person_box_ld.box = box;
			person_box_ld.landmarks = pose_lds;
			person_box_ld.flag = true;
			person_lds.push_back(person_box_ld);
		}
	}
}

void alpha::MMRTMPose::warm_up(int count)
{
	cv::Mat img = cv::Mat::zeros(input_height, input_width, CV_8UC3);
	for (size_t i = 0; i < count; i++)
	{
		auto res = net->Apply(img);
	}
}
