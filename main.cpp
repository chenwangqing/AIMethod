
#include <stdint.h>
#include <string>
#include <stdio.h>
#include <onnxruntime_cxx_api.h>
#include <opencv4/opencv2/opencv.hpp>

#include <sys/time.h>

#include "TargetDetection.hpp"
#include "Ratiocinate.hpp"

#include <time.h>
#include <unistd.h>

#include "Tensor.hpp"

using namespace AIMethod;

std::vector<Tools::Letterbox> lets;
std::vector<cv::Mat>          imgs;

static uint64_t GetMillisecond(void)
{
    struct timeval ts = {0};
    gettimeofday(&ts, NULL);
    return ts.tv_sec * 1000 + (ts.tv_usec / 1000);
}

static void ExecCallback(IRatiocinate                            *infer,
                         const std::vector<std::string>          &input_names,
                         const std::vector<Tensor<float>>        &input_datas,
                         const std::vector<std::string>          &output_names,
                         const std::vector<IRatiocinate::Result> &output_datas,
                         void                                    *context,
                         const std::string                       &err)
{
    if (!err.empty())
        printf("Err: %s", err.c_str());
    else {
        TargetDetection det;
        auto            ret = Tensor<float>::MakeConst(output_datas[0].shape, output_datas[0].data);
        auto            rs  = det.Yolo(ret, lets);
        for (size_t i = 0; i < imgs.size(); i++) {
            auto &img  = imgs[i];
            auto &rets = rs[i];
            det.DrawBox(img, rets);
            std::string str = Tools::Format("./output/{0}.jpg", i);
            cv::imwrite(str.c_str(), img);
        }
    }
    return;
}

int main(int argc, char **argv)
{
    std::string              err;
    auto                     infer = Ratiocinate_Create();
    IRatiocinate::Parameters parameters;
    parameters.model = "./best-dynamic.onnx";
    // parameters.model = "./best.onnx";
#if CFG_INFER_ENGINE == INFER_ENGINE_ONNXRUNTIME
    parameters.threads = 2;
#endif
    err             = infer->LoadModel(parameters);
    infer->callback = ExecCallback;

    if (!err.empty()) {
        printf("ERR: %s\n", err.c_str());
        return 0;
    }

    imgs.push_back(cv::imread("./img/1.jpg"));
    // imgs.push_back(cv::imread("./img/2.jpg"));
    imgs.push_back(cv::imread("./img/4.png"));
    auto inputs = Tools::ImageBGRToNCHW(imgs, {416, 416}, lets, err);
    inputs *= 1.0f / 255.0f;

    err = infer->ExecAsync({"images"}, {"output0"}, {inputs});
    inputs.Clear();
    while (infer->IsRun())
        sleep(1);
    return 0;
}
