//
// Created by DefTruth on 2021/10/7.
//

#include "ncnn_handler.h"


BasicNCNNHandler::BasicNCNNHandler(
    const std::string &_param_path, const std::string &_bin_path, unsigned int _num_threads) :
    log_id(_param_path.data()), param_path(_param_path.data()),
    bin_path(_bin_path.data()), num_threads(_num_threads)
{
}

void BasicNCNNHandler::initialize_handler()
{
    // init net, change this setting for better performance.
    net = new ncnn::Net();
    net->opt.use_vulkan_compute = false; // default
    net->opt.use_fp16_arithmetic = false;
    net->load_param(param_path);
    net->load_model(bin_path);
    input_indexes = net->input_indexes();
    output_indexes = net->output_indexes();
 #ifdef NCNN_STRING
    input_names = net->input_names();
    output_names = net->output_names();
 #endif
    num_outputs = output_indexes.size();
#ifdef POSE_DEBUG
    this->print_debug_string();
#endif
}

BasicNCNNHandler::~BasicNCNNHandler()
{
      if (net) delete net;
      net = nullptr;
}

void BasicNCNNHandler::print_debug_string()
{
      std::cout << "LITENCNN_DEBUG LogId: " << log_id << "\n";
      std::cout << "=============== Input-Dims ==============\n";
      for (int i = 0; i < input_indexes.size(); ++i)
      {
        std::cout << "Input: ";
        auto tmp_in_blob = net->blobs().at(input_indexes.at(i));
    #ifdef NCNN_STRING
        std::cout << input_names.at(i) << ": ";
    #endif
        std::cout << "shape: c=" << tmp_in_blob.shape.c
        << " h=" << tmp_in_blob.shape.h << " w=" << tmp_in_blob.shape.w << "\n";
    }

      std::cout << "=============== Output-Dims ==============\n";
      for (int i = 0; i < output_indexes.size(); ++i)
      {
        auto tmp_out_blob = net->blobs().at(output_indexes.at(i));
        std::cout << "Output: ";
    #ifdef NCNN_STRING
        std::cout << output_names.at(i) << ": ";
    #endif
        std::cout << "shape: c=" << tmp_out_blob.shape.c
                  << " h=" << tmp_out_blob.shape.h << " w=" << tmp_out_blob.shape.w << "\n";
      }
      std::cout << "========================================\n";
}

// static method
void BasicNCNNHandler::print_shape(const ncnn::Mat &mat, const std::string name)
{
    std::cout << name <<  ": " << "c=" << mat.c << ",h=" << mat.h << ",w=" << mat.w << "\n";
}

void BasicNCNNHandler::base_warm_up(int _height, int _width, int _channel, int warmup_count)
{
    ncnn::Mat in = ncnn::Mat(_width, _height, _channel);
    in.fill(0.01f);
    ncnn::Mat out;
    if (net != nullptr){
        const std::vector<const char*>& net_input_names = net->input_names();
        const std::vector<const char*>& net_output_names = net->output_names();
        for (int i=0;i< warmup_count; i++){
            ncnn::Extractor ex = net->create_extractor();
            #ifdef NCNN_STRING
            ex.input(net_input_names[0], in);
            ex.extract(net_output_names[0], out);
            #else
            std::cout << "NCNN_STRING=0, can't available input_names and output_names!" << std::endl;
            #endif
        }
    } else{
        std::cout << "Can't warm up ncnn model!" << std::endl;
    }
}

void BasicNCNNHandler::print_pretty_mat(const ncnn::Mat& m, std::vector<int>& channel_indexs)
{
    if (!channel_indexs.empty())
    {
        for (size_t i = 0; i < channel_indexs.size(); i++)
        {
            const float* ptr = m.channel(channel_indexs[i]);
            for (int y = 0; y < m.h; y++)
            {
                for (int x = 0; x < m.w; x++)
                {
                    printf("%f ", ptr[x]);
                }
                ptr += m.w;
                printf("\n");
            }
            printf("------------------------\n");
        }

    }
    else
    {
        for (int q = 0; q < m.c; q++)
        {
            const float* ptr = m.channel(q);
            for (int y = 0; y < m.h; y++)
            {
                for (int x = 0; x < m.w; x++)
                {
                    printf("%f ", ptr[x]);
                }
                ptr += m.w;
                printf("\n");
            }
            printf("------------------------\n");
        }
    }
}
