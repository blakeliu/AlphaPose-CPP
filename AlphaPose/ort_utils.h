#ifndef ORT_UTILS_H
#define ORT_UTILS_H
#include "headers.h"

namespace ortcv
{
	namespace utils
	{
		enum
		{
			CHW = 0, HWC = 1
		};
		Ort::Value create_tensor(const cv::Mat& mat, const std::vector<int64_t>& tensor_dims,
			const Ort::MemoryInfo& memory_info_handler,
			std::vector<float>& tensor_value_handler,
			unsigned int data_format = CHW) throw(std::runtime_error);

		cv::Mat normalize(const cv::Mat& mat, float mean, float scale);

		cv::Mat normalize(const cv::Mat& mat, const float mean[3], const float scale[3]);

		void normalize(const cv::Mat& inmat, cv::Mat& outmat, float mean, float scale);

		void normalize_inplace(cv::Mat& mat_inplace, float mean, float scale);

		void normalize_inplace(cv::Mat& mat_inplace, const float mean[3], const float scale[3]);
	}
}

#endif // !ORT_UTILS_H
