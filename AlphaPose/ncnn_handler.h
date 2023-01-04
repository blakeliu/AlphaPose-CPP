//
// Created by DefTruth on 2021/10/7.
//

#ifndef NCNN_HANDLER_H
#define NCNN_HANDLER_H

#include "headers.h"
#include "ncnn/net.h"
#include "ncnn/layer.h"

class BasicNCNNHandler
{
protected:
	ncnn::Net* net = nullptr;
	const char* log_id = nullptr;
	const char* param_path = nullptr;
	const char* bin_path = nullptr;
	std::vector<const char*> input_names;
	std::vector<const char*> output_names;
	std::vector<int> input_indexes;
	std::vector<int> output_indexes;
	int num_outputs = 1;

protected:
	const unsigned int num_threads; // initialize at runtime.

protected:
	/*!
	 *ncnn模型初始化
	 * @param _param_path ncnn param path
	 * @param _bin_path  ncnn bin path
	 * @param _num_threads  线程数量, 默认=1
	 */
	explicit BasicNCNNHandler(const std::string& _param_path,
		const std::string& _bin_path,
		unsigned int _num_threads = 1);

	virtual ~BasicNCNNHandler();

	// un-copyable
protected:
	BasicNCNNHandler(const BasicNCNNHandler&) = delete; //
	BasicNCNNHandler(BasicNCNNHandler&&) = delete; //
	BasicNCNNHandler& operator=(const BasicNCNNHandler&) = delete; //
	BasicNCNNHandler& operator=(BasicNCNNHandler&&) = delete; //

private:
	virtual void transform(const cv::Mat& mat, ncnn::Mat& in) = 0;

protected:
	virtual void initialize_handler();

	virtual void print_debug_string();

public:
	static void print_shape(const ncnn::Mat& mat, const std::string name = "");

	/*!
	 * 模型预热功能
	 * @param _height
	 * @param _width
	 * @param _channel
	 */
	virtual void base_warm_up(int _height, int _width, int _channel = 3, int warmup_count = 1);


	static void print_pretty_mat(const ncnn::Mat& m, std::vector<int>& channel_indexs);
};

#endif //NCNN_HANDLER_H
