#ifndef _ORT_HANDLER_H
#define _ORT_HANDLER_H

#include "onnxruntime_cxx_api.h"
#include "headers.h"
namespace ort
{
	// single input & multi outputs£¬·Çdynamic shape
	class BasicOrtHandler
	{
	protected:
		Ort::Env ort_env;
		Ort::Session* ort_session = nullptr;
		const char* input_name = nullptr;
		std::vector<const char*> input_node_names;
		std::vector<int64_t> input_node_dims;  // 1 input only
		std::size_t input_tensor_size = 1;
		std::vector<float> input_values_handler;
		Ort::MemoryInfo memory_info_handler = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
		std::vector<const char*> output_node_names;
		std::vector<std::vector<int64_t>> output_node_dims; // >=1 outputs

		const ORT_CHAR* onnx_path = nullptr;
		const char* log_id = nullptr;
		int num_outputs = 1;

		const unsigned int num_threads;
	protected:
		explicit BasicOrtHandler(const std::string& _onnx_path, unsigned int _num_threads = 1);
		virtual ~BasicOrtHandler();

		//disable copyable
	protected:
		BasicOrtHandler(const BasicOrtHandler&) = delete;
		BasicOrtHandler(const BasicOrtHandler&&) = delete;
		BasicOrtHandler& operator=(const BasicOrtHandler&) = delete;
		BasicOrtHandler& operator=(BasicOrtHandler&) = delete;

	protected:
		virtual Ort::Value transform(const cv::Mat& mat) = 0;

	private:
		void initialize_handler();
		void print_debug_string();

	};

}


#endif // !_ORT_HANDLER_H



