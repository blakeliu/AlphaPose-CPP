#include "alphapose.h"

alpha::AlphaPose::AlphaPose(const std::string& _det_weight_path,
	const std::string& _pose_weight_path,
	float _det_thr,
	float _det_nms_thr,
	float _pose_kpt_thr,
	float _pose_nms_thr,
	float _track_iou_thr) :
	det_weight_path(_det_weight_path),
	pose_weight_path(_pose_weight_path),
	det_score_threshold(_det_thr),
	det_nms_threshold(_det_nms_thr),
	pose_score_threshold(_pose_kpt_thr),
	pose_nms_threshold(_pose_nms_thr),
	track_iou_thr(_track_iou_thr)
{
	try
	{
		tracker = std::make_unique<mmdeploy::PoseTracker>(mmdeploy::Model(det_weight_path), mmdeploy::Model(pose_weight_path),
			mmdeploy::Device{ "cpu" });
	}
	catch (const std::exception& e)
	{
		std::cout << "Tracker model init failed! det_weight_path: " << det_weight_path <<
			", pose_weight_path: " << pose_weight_path << ", err: " << e.what() << std::endl;
	}
	init_tracker_params(params);
}

alpha::AlphaPose::~AlphaPose()
{
}

void alpha::AlphaPose::init_tracker_params(mmdeploy::PoseTracker::Params& params)
{
	params->det_interval = 1;
	params->det_label = 0;
	params->det_thr = det_score_threshold;
	params->det_nms_thr = det_nms_threshold;
	params->det_min_bbox_size = -1;
	params->pose_max_num_bboxes = -1;
	params->pose_kpt_thr = pose_score_threshold;
	params->pose_nms_thr = pose_nms_threshold;
	params->pose_min_keypoints = -1;
	params->pose_bbox_scale = 1.25;
	params->pose_min_bbox_size = -1;
	params->track_iou_thr = track_iou_thr;
	params->track_max_missing = 10;
}

void alpha::AlphaPose::detect(cv::Mat& image, std::vector<types::BoxfWithLandmarks>& person_lds)
{
	if (image.empty())
	{
		return;
	}
	mmdeploy::PoseTracker::State state = tracker->CreateState(params);
	mmdeploy::PoseTracker::Result result = tracker->Apply(state, image);
	person_lds.clear();
	for (const mmdeploy_pose_tracker_target_t& target : result) {
		types::Landmarks pose_lds;
		pose_lds.scores.clear();
		for (size_t i = 0; i < target.keypoint_count; ++i) {
			pose_lds.points.emplace_back(cv::Point2f(target.keypoints[i].x, target.keypoints[i].y));
			pose_lds.scores.push_back(target.scores[i]);
		}
		pose_lds.flag = true;
		types::BoxfWithLandmarks person_box_ld;
		types::Boxf box;
		box.x1 = target.bbox.left;
		box.y1 = target.bbox.bottom;
		box.x2 = target.bbox.right;
		box.y2 = target.bbox.bottom;
		box.score = 1;
		box.label = 0;
		box.label_text = "person";
		box.flag = true;

		person_box_ld.box = box;
		person_box_ld.landmarks = pose_lds;
		person_box_ld.flag = true;
		person_lds.push_back(person_box_ld);
	}
}

void alpha::AlphaPose::warm_up(int count)
{
	mmdeploy::PoseTracker::State state = tracker->CreateState(params);
	cv::Mat img = cv::Mat::zeros(pose_height, pose_width, CV_8UC3);
	for (size_t i = 0; i < count; i++)
	{
		auto res = tracker->Apply(state, img);
	}
}

void alpha::AlphaPose::show(const std::vector<types::BoxfWithLandmarks>& boxes_kps, cv::Mat& image, int pose_num_joints, float kps_thr, int resize)
{
	if (boxes_kps.empty()) return;
	std::vector<std::array<int, 2>> l_pair;
	std::vector<std::array<int, 3>> p_color;
	std::vector<std::array<int, 3>> line_color;
	int len_pair = 0;
	int len_pcolor = 0;
	int len_lcolor = 0;
	std::vector<int> link_color_idxs;
	std::vector<int> point_color_idxs;

	int height = image.rows;
	int width = image.cols;
	float scale = float(resize) / std::max(height, width);
	cv::resize(image, image, cv::Size(0, 0), scale, scale);

	if (pose_num_joints == 136)
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

		link_color_idxs = { 0, 0, 0,0,
			1, 1, 1, 1, 1, 1,
			2, 2, 2, 2, 2, 2, 2, 2,
			3, 3, 3, 3, 3, 3, 3,
			4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,4, 4, 4, 4, 4, 4, 4, 4, 4, 4,4, 4, 4, 4, 4, 4, 4, 4, 4, 4,4, 4, 4, 4, 4, 4, 4, 4, 4, 4,4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
			5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
			6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
			7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7
		};

		point_color_idxs = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
			14, 15, 16,17, 18, 19, 20, 21, 22, 23, 24, 25,
			26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26,
			26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26, 26,
			26, 26, 26, 26, 26, 26, 26, 26,
			27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27,
			27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27
		};

		for (const types::BoxfWithLandmarks& person_kps : boxes_kps)
		{
			std::vector<int> show_point{ 0 };
			show_point.resize(pose_num_joints);
			for (size_t i = 0; i < len_pair; i++)
			{
				int s = _pair[i][0];
				int e = _pair[i][1];
				if (person_kps.landmarks.scores[s] > kps_thr && person_kps.landmarks.scores[e] > kps_thr)
				{
					show_point[s] = 1;
					show_point[e] = 1;
					cv::Point2f sp = person_kps.landmarks.points[s];
					cv::Point2f ep = person_kps.landmarks.points[e];
					int color_id = link_color_idxs[i];
					cv::line(image, cv::Point2f(sp.x * scale, sp.y * scale),
						cv::Point2f(ep.x * scale, ep.y * scale),
						cv::Scalar(_lcolor[color_id][0], _lcolor[color_id][1], _lcolor[color_id][2]), 1, cv::LINE_AA);
				}
			}
			for (size_t i = 0; i < person_kps.landmarks.points.size(); i++)
			{
				if (show_point[i] > 0)
				{
					int color_id = point_color_idxs[i];
					cv::circle(image, cv::Point2f(person_kps.landmarks.points[i].x * scale, person_kps.landmarks.points[i].y * scale), 1,
						cv::Scalar(_pcolor[color_id][0], _pcolor[color_id][1], _pcolor[color_id][2]), 2, cv::LINE_AA);
				}

			}
		}
	}
	else if (pose_num_joints == 26)
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

	}


}
