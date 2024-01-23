
#include <stdint.h>
#include <string>
#include <stdio.h>
#include <onnxruntime_cxx_api.h>
#include <opencv4/opencv2/opencv.hpp>

#include <sys/time.h>

#include "TargetDetection.hpp"
#include "TargetSegmention.hpp"
#include "Ratiocinate.hpp"
#include "PoseEstimation.hpp"
#include "Algorithm.hpp"

#include <time.h>
#include <unistd.h>

#include "Tensor.hpp"
#include "Define.h"

using namespace AIMethod;

std::vector<Tools::Letterbox> lets;
std::vector<cv::Mat>          imgs;

#define IS_TARGETDETECTION 0

uint64_t GetMillisecond(void)
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
#if IS_TARGETDETECTION
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
#else
        if (output_datas.size() == 2) {
            cv::Scalar colors[8] = {
                CV_RGB(250, 0, 0),
                CV_RGB(0, 250, 0),
                CV_RGB(0, 0, 250),
                CV_RGB(0, 250, 250),
                CV_RGB(250, 250, 0),
                CV_RGB(250, 0, 250),
                CV_RGB(210, 110, 110),
                CV_RGB(50, 150, 250),
            };
            int              color_idx = 0;
            TargetDetection  det;
            TargetSegmention seg;
            auto             pred  = Tensor<float>::MakeConst(output_datas[0].shape, output_datas[0].data);
            auto             proto = Tensor<float>::MakeConst(output_datas[1].shape, output_datas[1].data);
            auto             rs    = seg.Yolo(det, pred, proto, lets);
            if (rs.size() == imgs.size()) {
                for (size_t i = 0; i < rs.size(); i++) {
                    for (auto &v : rs[i]) {
                        // 获取轮廓
                        std::vector<std::vector<cv::Point>> contours;
                        cv::findContours(v.mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_TC89_KCOS);
                        // 绘制
                        cv::polylines(imgs[i](v.box), contours, true, colors[(color_idx++) % 8], 5);
                        cv::imwrite(Tools::Format("./output/result-{0}.jpg", i).c_str(), imgs[i]);
                    }
                }
            }
        }
#endif
    }
    return;
}

void _TargetSegmention(IRatiocinate *infer)
{
    std::string              err;
    IRatiocinate::Parameters parameters;
    parameters.model = "./onnx/yolov5s-seg.onnx";
    // parameters.model = "./best.onnx";
#if CFG_INFER_ENGINE == INFER_ENGINE_ONNXRUNTIME
    parameters.threads = 2;
#endif
    err             = infer->LoadModel(parameters);
    infer->callback = ExecCallback;

    if (!err.empty()) {
        printf("ERR: %s\n", err.c_str());
        return;
    }

    imgs.push_back(cv::imread("./img/bus.jpg"));
    // imgs.push_back(cv::imread("./img/2.jpg"));
    // imgs.push_back(cv::imread("./img/4.png"));
#if IS_TARGETDETECTION
    cv::Size2i size(640, 640);
#else
    cv::Size2i size(640, 640);
#endif
    auto inputs = Tools::ImageBGRToNCHW(imgs, size, lets, err);
    inputs      = op.Mul($(inputs), 1 / 255.0f);

    err = infer->ExecAsync({"images"}, {"output0", "output1"}, {inputs});
    inputs.Clear();
    return;
}

static void ExecCallback_PoseEstimation(IRatiocinate                            *infer,
                                        const std::vector<std::string>          &input_names,
                                        const std::vector<Tensor<float>>        &input_datas,
                                        const std::vector<std::string>          &output_names,
                                        const std::vector<IRatiocinate::Result> &output_datas,
                                        void                                    *context,
                                        const std::string                       &err)
{
    if (!err.empty()) {
        printf("Err: %s", err.c_str());
        return;
    }
    auto           detections = Tensor<float>::MakeConst(output_datas[0].shape, output_datas[0].data);
    PoseEstimation pose;
    auto           ret = pose.Yolo(detections, lets[0]);
    pose.DrawBox(imgs[0], ret);
    cv::imwrite("./output/result.jpg", imgs[0]);
    return;
}

void _PoseEstimation(IRatiocinate *infer)
{
    std::string              err;
    IRatiocinate::Parameters parameters;
    parameters.model = "./onnx/yolov5s6_pose_640_ti_lite_54p9_82p2.onnx";
    // parameters.model = "./best.onnx";
#if CFG_INFER_ENGINE == INFER_ENGINE_ONNXRUNTIME
    parameters.threads = 2;
#endif
    err             = infer->LoadModel(parameters);
    infer->callback = ExecCallback_PoseEstimation;

    if (!err.empty()) {
        printf("ERR: %s\n", err.c_str());
        return;
    }

    imgs.push_back(cv::imread("./img/zidane.jpg"));
    cv::Size2i size(640, 640);
    auto       inputs = Tools::ImageBGRToNCHW(imgs, size, lets, err);
    inputs            = op.Mul($(inputs), 1 / 255.0f);

    err = infer->ExecAsync({"images"}, {"detections"}, {inputs});
    return;
}

class PoseEstimation_test {
private:
    strings          files;
    int              idx   = -1;
    IRatiocinate    *infer = nullptr;
    cv::Mat          img;
    Tools::Letterbox let;
    const string     dir = "/home/work/yolo5-onnx/img/test1";

    static void ExecCallback(IRatiocinate                            *infer,
                             const std::vector<std::string>          &input_names,
                             const std::vector<Tensor<float>>        &input_datas,
                             const std::vector<std::string>          &output_names,
                             const std::vector<IRatiocinate::Result> &output_datas,
                             void                                    *context,
                             const std::string                       &err)
    {
        PoseEstimation_test *ps = static_cast<PoseEstimation_test *>(context);
        if (!err.empty()) {
            printf("Err: %s", err.c_str());
            exit(1);
        }
        auto           detections = Tensor<float>::MakeConst(output_datas[0].shape, output_datas[0].data);
        PoseEstimation pose;
        auto           ret = pose.Yolo(detections, ps->let);
        pose.DrawBox(ps->img, ret);
        cv::imwrite(Tools::Format("./output/test.{0}.jpg", ps->idx).c_str(), ps->img);
        ps->Start();
        return;
    }

public:
    PoseEstimation_test(IRatiocinate *infer) :
        infer(infer)
    {
        files = Tools::GetFiles(this->dir, "jpg");

        std::string              err;
        IRatiocinate::Parameters parameters;
        parameters.model = "./onnx/yolov5s6_pose_640_ti_lite_54p9_82p2.onnx";
        // parameters.model = "./best.onnx";
#if CFG_INFER_ENGINE == INFER_ENGINE_ONNXRUNTIME
        parameters.threads = 2;
#endif
        err                     = infer->LoadModel(parameters);
        infer->callback         = this->ExecCallback;
        infer->callback_context = this;

        if (!err.empty()) {
            printf("ERR: %s\n", err.c_str());
            exit(1);
        }
        return;
    }

    void Start()
    {
        this->idx++;
        if (this->idx > (int)this->files.size())
            return;
        char name[1024];
        snprintf(name, sizeof(name), "%s/%s", this->dir.c_str(), this->files[this->idx].c_str());
        this->img = cv::imread(name);
        if (this->img.rows == 0)
            RUN_ERR("Err");
        cv::imshow("test", img);
        while (true) sleep(1);
        string                        err;
        std::vector<Tools::Letterbox> lets;
        cv::Size2i                    size(640, 640);

        auto inputs = Tools::ImageBGRToNCHW({img}, size, lets, err);
        inputs      = op.Mul($(inputs), 1 / 255.0f);
        this->let   = lets[0];
        err         = infer->ExecAsync({"images"}, {"detections"}, {inputs});
        if (err != "")
            RUN_ERR(err);
        return;
    }
};

void FaceRecognize_test()
{
    Tools::FaceRecognize face("/home/work/haar-cascade-files/haarcascade_frontalface_alt.xml",
                              "/home/work/haar-cascade-files/haarcascade_eye.xml",
                              "/home/work/haar-cascade-files/haarcascade_mcs_mouth.xml");
    auto                 img = cv::imread("./img/sample1.jpg");
    cv::Mat              gray;
    // 转灰度
   // cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    // 直方图均衡
   // cv::equalizeHist(gray, gray);
    // 识别
    auto ret = face.Recognize(img);
    face.DrawBox(img, ret);
    cv::imwrite("./output/face.jpg", img);
    return;
}


int main(int argc, char **argv)
{
#if 1
    FaceRecognize_test();
#elif 0
    auto                infer = Ratiocinate_Create();
    PoseEstimation_test tmp(infer);
    tmp.Start();
#endif
    while (true)
        sleep(1);
    return 0;
}
