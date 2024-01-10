
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

static void Detection_Callback(TargetDetection *det,
                               std::vector<std::vector<TargetDetection::Result>> &results,
                               void *context,
                               const std::string &err)
{
    // std::vector<cv::Mat> &imgs = *static_cast<std::vector<cv::Mat> *>(context);
    // for (size_t i = 0; i < results.size(); i++)
    // {
    //     auto &img = imgs[i];
    //     // det->DrawBox(img, results[i], {255, 0, 0});
    //     char name[255];
    //     snprintf(name, 255, "./result-%ld.jpg", i);
    //     cv::imwrite(name, img);
    // }

    return;
}

static void ExecCallback(const std::map<std::string, IRatiocinate::Input> &inputs,
                         const std::map<std::string, IRatiocinate::Result> &result,
                         void *context,
                         const std::string &err)
{
    if (!err.empty())
        printf("Err: %s", err.c_str());
    return;
}

int main(int argc, char **argv)
{
    std::string err;
    auto inter = Ratiocinate_Create();
    IRatiocinate::Parameters parameters;
    parameters.model = "./best.onnx";
    parameters.threads = 2;
    parameters.is_normal = true;
    err = inter->LoadModel(parameters);
    inter->callback = ExecCallback;

    auto name = inter->GetIOInfo(false)[0].name;
    std::vector<cv::Mat> imgs;
    imgs.push_back(cv::imread("./img/1.jpg"));
    imgs.push_back(cv::imread("./img/2.jpg"));
    err = inter->ExecAsync({{name, imgs}}, {416, 416});
    while (inter->IsRun())
        sleep(1);
    return 0;
}
