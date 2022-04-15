// OnnxCMake.cpp : Defines the entry point for the application.
//

#include "OnnxCMake.h"
#include <onnxruntime_cxx_api.h>
#include <algorithm>
//#include <cuda_provider_factory.h>

using namespace std;

template <typename T>
static void softmax(T& input) {
    float rowmax = *std::max_element(input.begin(), input.end());
    std::vector<float> y(input.size());
    float sum = 0.0f;
    for (size_t i = 0; i != input.size(); ++i) {
        sum += y[i] = std::exp(input[i] - rowmax);
    }
    for (size_t i = 0; i != input.size(); ++i) {
        input[i] = y[i] / sum;
    }
}

int main()
{
    try {
        Ort::Env env(ORT_LOGGING_LEVEL_VERBOSE, "test");
        // Ort::SessionOptions options_;
        Ort::SessionOptions sessionOptions;
        // sessionOptions.
        // sessionOptions.SetIntraOpNumThreads(1);
        auto av_pr = Ort::GetAvailableProviders();
        bool enableCuda = false;
        if (enableCuda) {
            // Using CUDA backend
            std::cout << "Onnx test";
            // https://github.com/microsoft/onnxruntime/blob/v1.8.2/include/onnxruntime/core/session/onnxruntime_cxx_api.h#L329
            //OrtCUDAProviderOptions options;
            //options.device_id = 0;
            // OrtSessionOptionsAppendExecutionProvider_DML(sessionOptions, 0);

            // options.arena_extend_strategy = 0;
            //options.cuda_mem_limit = 2 * 1024 * 1024 * 1024;
            //options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearch::EXHAUSTIVE;
            //// options.do_copy_in_default_stream = 1;
            //options.has_user_compute_stream = 0;
            //Ort::ThrowOnError(
            //    OrtSessionOptionsAppendExecutionProvider_Tensorrt(sessionOptions, 0));
            Ort::ThrowOnError(
                OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0));
            //sessionOptions.AppendExecutionProvider_CUDA(options);
            //OrtSessionOptionsAppendExecutionProvider_DML(sessionOptions, 0);
            // OrtSessi
            // sessionOptions.AppendExecutionProvider_CUDA(options);
            std::cout << "onnx GPU session declared" << std::endl;
            // sessionOptions.DisableMemPattern();
            // sessionOptions.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        }
        std::cout << "onnx Creating session" << std::endl;
        Ort::Session session_(
            env, L"../../../../OnnxCMake/resnet18-v2-7.onnx",
            sessionOptions);
        std::cout << "onnx Created session" << std::endl;
        // auto name = session_.GetModelMetadata();
        // name.GetGraphDescription(allocator);
        auto allocator = Ort::AllocatorWithDefaultOptions();
        std::cout << "onnx Created allocator with default" << std::endl;

        // print name/shape of inputs

        static constexpr const int width_ = 224;
        static constexpr const int height_ = 224;
        Ort::Value input_tensor_{ nullptr };
        std::array<int64_t, 4> input_shape_{ 1, 3, width_, height_ };

        Ort::Value output_tensor_{ nullptr };
        std::array<int64_t, 2> output_shape_{ 1, 1000 };

        std::array<float, 3 * width_ * height_> input_image_{};
        std::array<float, 1000> results_{};
        int64_t result_{ 0 };

        const char* input_names[] = { "data" };
        const char* output_names[] = { "resnetv22_dense0_fwd" };

        // Create a single Ort tensor of random numbers
        std::vector<Ort::Value> input_tensors;
        std::cout << "onnx creating cpu memory info with default" << std::endl;

        auto memory_info = Ort::MemoryInfo::CreateCpu(
            OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
        std::cout << "onnx created cpu memory info with default" << std::endl;
        std::vector<float> input_tensor_values(3 * 224 * 224);
        std::generate(input_tensor_values.begin(), input_tensor_values.end(), [&] {
            return rand() % 255;
            });  // generate random numbers in the range [0, 255]
        input_tensor_ = Ort::Value::CreateTensor<float>(
            memory_info, input_tensor_values.data(), input_tensor_values.size(),
            input_shape_.data(), input_shape_.size());
        output_tensor_ = Ort::Value::CreateTensor<float>(
            memory_info, results_.data(), results_.size(), output_shape_.data(),
            output_shape_.size());

        std::cout << "onnx created input tensors" << std::endl;
        session_.Run(Ort::RunOptions{ nullptr }, input_names, &input_tensor_, 1,
            output_names, &output_tensor_, 1);
        auto tensor = output_tensor_.GetTensorMutableData<int>();

        // std::array<float, 1000> results{tensor};
        softmax(results_);
        result_ = std::distance(results_.begin(),
            std::max_element(results_.begin(), results_.end()));
        std::cout << "Result index in onnx: " << result_ << std::endl;
    }
    catch (const Ort::Exception& e) {
        std::cout << "Exception in onnx" << e.what() << " Error Code: " << e.GetOrtErrorCode()
            << std::endl;
    }
	return 0;
}
