
#include <stdint.h>
#include <string>
#include <stdio.h>
#include "onnxruntime_cxx_api.h"
#include <opencv4/opencv2/opencv.hpp>

#include <sys/time.h>

#include "TargetDetection.hpp"
#include "Ratiocinate.hpp"

#include <time.h>
#include <unistd.h>

static uint64_t GetMillisecond(void)
{
    struct timeval ts = {0};
    gettimeofday(&ts, NULL);
    return ts.tv_sec * 1000 + (ts.tv_usec / 1000);
}

static void ExecCallback(IRatiocinate                                *infer,
                         std::map<std::string, IRatiocinate::Input>  &inputs,
                         std::map<std::string, IRatiocinate::Result> &results,
                         void                                        *context,
                         const std::string                           &err)
{
    if (!err.empty())
        printf("Err: %s", err.c_str());
    else {
        TargetDetection det;
        auto           &input_name  = infer->GetIOInfo(false)[0].name;
        auto           &output_name = infer->GetIOInfo(true)[0].name;
        auto            ret         = results[output_name];
        auto            in          = inputs[input_name];
        auto            rs          = det.Yolo(ret, in.lets);
        for (size_t i = 0; i < in.imgs.size(); i++) {
            auto &img  = in.imgs[i];
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
    parameters.model     = "./best.onnx";
    parameters.threads   = 2;
    parameters.is_normal = true;
    err                  = infer->LoadModel(parameters);
    infer->callback      = ExecCallback;

    auto                 name = infer->GetIOInfo(false)[0].name;
    std::vector<cv::Mat> imgs;
    imgs.push_back(cv::imread("./img/1.jpg"));
    imgs.push_back(cv::imread("./img/2.jpg"));
    err = infer->ExecAsync({{name, imgs}}, {416, 416});
    while (infer->IsRun())
        sleep(1);
    return 0;
}
