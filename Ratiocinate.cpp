
#include "Ratiocinate.hpp"

/**
 * @brief    获取图像数据
 * @param    imgs           图片列表
 * @param    is_normal      是否归一化
 * @param    shape          形状
 * @param    err            错误
 * @return   std::vector<float>
 * @author   CXS (chenxiangshu@outlook.com)
 * @date     2024-01-09
 */
static std::vector<float> GetImageBGRValue(const std::vector<cv::Mat>    &imgs,
                                           bool                           is_normal,
                                           cv::Size2i                     shape,
                                           std::string                   &err,
                                           std::vector<Tools::Letterbox> &let_box)
{
    int                block_size = shape.height * shape.width * 3;
    std::vector<float> tmp(imgs.size() * block_size);
    cv::Mat            img_f32;
    for (size_t i = 0; i < imgs.size(); i++) {
        auto   img  = imgs[i];
        float *data = tmp.data() + i * block_size;
        if (img.size().empty()) {
            err = "The picture cannot be empty";
            return std::vector<float>();
        }
        if (img.type() != CV_32FC3) {
            img.convertTo(img_f32, CV_32FC3);   // 转float
            img = img_f32;
        }
        if (img.channels() != 3) {
            err = "The picture must be 3 channels";
            return std::vector<float>();
        }
        if (img.type() != CV_32FC3) {
            err = "Image data type conversion failed. Procedure";
            return std::vector<float>();
        }
        // 图像变换
        Tools::Letterbox let;
        if (img.size() != shape)
            img = Tools::Letterbox::Make(img, shape.height, shape.width, let);
        let_box.push_back(let);
        // BGR2RGB
        for (int i = 0; i < img.rows; i++) {
            for (int j = 0; j < img.cols; j++) {
                auto &tmp                                        = img_f32.at<cv::Vec3f>(i, j);
                data[i * img.cols + j + 0]                       = tmp[2];
                data[i * img.cols + j + 1 * img.cols * img.rows] = tmp[1];
                data[i * img.cols + j + 2 * img.cols * img.rows] = tmp[0];
            }
        }
    }
    if (is_normal) {
        // 归一化
        float *data = tmp.data();
        for (size_t i = 0; i < tmp.size(); i++, data++)
            *data /= 255.0f;
    }
    return tmp;
}

static const IRatiocinate::IOInfo *FindIOInfo(const std::vector<IRatiocinate::IOInfo> &infos, std::string name)
{
    for (auto &v : infos) {
        if (v.name == name)
            return &v;
    }
    return nullptr;
}

#if EN_ONNXRUNTIME
#include "onnxruntime_cxx_api.h"
class Ratiocinate : public IRatiocinate {
private:
    Ort::Session       *session = nullptr;
    Ort::Env            env;
    std::vector<IOInfo> input;
    std::vector<IOInfo> output;
    bool                is_normal;   // 输入归一化

    class InputData {
    public:
        std::vector<cv::Mat>          imgs;             // 图像
        std::vector<Tools::Letterbox> lets;             // 修正信息
        std::vector<float>            data;             // 数据
        const IOInfo                 *info = nullptr;   // IO信息

        InputData() {}
        InputData(InputData &&dat)
        {
            this->imgs = std::move(dat.imgs);
            this->lets = std::move(dat.lets);
            this->data = std::move(dat.data);
            this->info = dat.info;
        }

        void operator=(InputData &&dat)
        {
            this->imgs = std::move(dat.imgs);
            this->lets = std::move(dat.lets);
            this->data = std::move(dat.data);
            this->info = dat.info;
        }
    };

    struct
    {
        std::vector<InputData>    input_data;       // 输入数据
        std::vector<const char *> input_names;      // 输入名称
        std::vector<Ort::Value>   input_tensors;    // 输入张量
        std::vector<const char *> output_names;     // 输入名称
        std::vector<Ort::Value>   output_tensors;   // 输出张量
    } status;

    void ClearStatus()
    {
        this->status.input_data.clear();
        this->status.input_names.clear();
        this->status.input_tensors.clear();
        this->status.output_names.clear();
        this->status.output_tensors.clear();
        return;
    }

    static void RunAsyncCallbackFn(void        *user_data,
                                   OrtValue   **outputs,
                                   size_t       num_outputs,
                                   OrtStatusPtr status_ptr)
    {
        Ratiocinate *infer = static_cast<Ratiocinate *>(user_data);
        Ort::Status  status(status_ptr);
        if (infer->callback != nullptr) {
            std::map<std::string, Result> result;
            std::map<std::string, Input>  inputs;
            for (size_t i = 0; i < infer->status.input_names.size(); i++) {
                Input p;
                p.imgs = std::move(infer->status.input_data[i].imgs);
                p.lets = std::move(infer->status.input_data[i].lets);

                inputs[infer->status.input_names[i]] = p;
            }
            if (status.IsOK()) {
                for (size_t i = 0; i < infer->status.output_names.size(); i++) {
                    Result ret;
                    ret.data  = infer->status.output_tensors[i].GetTensorMutableData<float>();
                    ret.shape = infer->status.output_tensors[i].GetTensorTypeAndShapeInfo().GetShape();

                    result[infer->status.output_names[i]] = ret;
                }
                infer->callback(infer, inputs, result, infer->callback_context, std::string());
            } else {
                infer->callback(infer, inputs, result, infer->callback_context, status.GetErrorMessage());
            }
        }
        infer->ClearStatus();
        infer->is_runing.fetch_sub(1);
        return;
    }

public:
    virtual ~Ratiocinate()
    {
        if (this->session != nullptr)
            delete this->session;
        return;
    }

    virtual std::string LoadModel(const Parameters &params) override
    {
        this->is_normal = params.is_normal;
        Ort::SessionOptions options;
        // 设置线程数量
        options.SetIntraOpNumThreads(params.threads);
        // ORT_ENABLE_ALL: 启用所有可能的优化
        options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        try {
            this->session = new Ort::Session{this->env, params.model, options};
        }
        catch (std::exception &e) {
            return e.what();
        }
        if (this->session == nullptr)
            return "Failed to create a session";
        // 获取输入/输出信息
        Ort::AllocatorWithDefaultOptions allocator;
        for (size_t i = 0; i < this->session->GetInputCount(); i++) {
            IOInfo info;
            info.name  = this->session->GetInputNameAllocated(i, allocator).get();
            info.shape = this->session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
            this->input.push_back(info);
        }
        for (size_t i = 0; i < this->session->GetOutputCount(); i++) {
            IOInfo info;
            info.name  = this->session->GetOutputNameAllocated(i, allocator).get();
            info.shape = this->session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
            this->output.push_back(info);
        }
        return std::string();
    }

    virtual const std::vector<IOInfo> &GetIOInfo(bool isOutput) const override
    {
        return isOutput ? this->output : this->input;
    }

    std::string _ExecAsync(const std::map<std::string, std::vector<cv::Mat>> &inputs,
                           cv::Size2i                                         size)
    {
        if (inputs.size() != this->input.size())
            return "parameter error";
        // 检查形状
        for (auto &input : inputs) {
            auto info = FindIOInfo(this->input, input.first);
            if (info == nullptr)
                return "Parameter names do not match";
            if (input.second.size() == 0 || (input.second.size() != info->shape[0] && info->shape[0] != -1))
                return "The input shape does not match";
            if (info->shape.size() != 4)
                return "Image processing input must be [BCHW]";
            if (info->shape[1] != 3)
                return "The model must receive 3 channel images";
        }
        // 清除数据
        ClearStatus();
        std::string err;
        // 创建内存分配器
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        // 获取输入数据
        for (auto &input : inputs) {
            InputData dat;
            auto      info = FindIOInfo(this->input, input.first);
            if (info->shape[2] > 0)
                size.height = info->shape[2];
            if (info->shape[3] > 0)
                size.width = info->shape[3];
            if (size.width <= 0 || size.height <= 0)
                return "Picture size error";
            dat.imgs = input.second;
            dat.data = GetImageBGRValue(input.second, this->is_normal, size, err, dat.lets);
            if (dat.data.empty())
                return err;
            auto shape = info->shape;
            shape[0]   = input.second.size();
            shape[2]   = size.height;
            shape[3]   = size.width;
            // 创建张量
            auto tensor = Ort::Value::CreateTensor<float>(memory_info,
                                                          dat.data.data(),
                                                          dat.data.size(),
                                                          shape.data(),
                                                          4);
            this->status.input_data.push_back(std::move(dat));
            this->status.input_names.push_back(info->name.c_str());
            this->status.input_tensors.push_back(std::move(tensor));
        }
        // 设置输出
        for (auto &v : this->output) {
            this->status.output_names.push_back(v.name.c_str());
            this->status.output_tensors.push_back(std::move(Ort::Value{nullptr}));
        }
        // 执行
        try {
            this->is_runing.fetch_add(1);
            this->session->RunAsync(Ort::RunOptions{nullptr},
                                    this->status.input_names.data(),
                                    this->status.input_tensors.data(),
                                    this->status.input_data.size(),
                                    this->status.output_names.data(),
                                    this->status.output_tensors.data(),
                                    this->status.output_tensors.size(),
                                    RunAsyncCallbackFn,
                                    this);
        }
        catch (std::exception &e) {
            err = e.what();
            this->is_runing.fetch_sub(1);
        }
        return err;
    }

    virtual std::string ExecAsync(const std::map<std::string, std::vector<cv::Mat>> &inputs,
                                  cv::Size2i                                         size) override
    {
        int flag = 0;
        if (!this->is_runing.compare_exchange_strong(flag, 1))
            return "A task is running";
        auto ret = _ExecAsync(inputs, size);
        this->is_runing.fetch_sub(1);
        return ret;
    }

    virtual bool IsRun() override
    {
        return this->is_runing.load() != 0;
    }
};
#else
// TODO: 未完成
#include <opencv4/opencv2/dnn.hpp>
class Ratiocinate : public IRatiocinate {
private:
    cv::dnn::Net       *session = nullptr;
    bool                is_normal;   // 输入归一化
    std::vector<IOInfo> input;
    std::vector<IOInfo> output;

    class InputData {
    public:
        std::vector<cv::Mat>          imgs;   // 图像
        std::vector<Tools::Letterbox> lets;   // 修正信息
        cv::Mat                       data;
    };

public:
    virtual ~Ratiocinate()
    {
        if (this->session != nullptr)
            delete this->session;
        return;
    }

    virtual std::string LoadModel(const Parameters &params) override
    {
        this->is_normal = params.is_normal;
        try {
            this->session = new cv::dnn::Net(cv::dnn::readNetFromONNX(params.model));
        }
        catch (std::exception ex) {
            return ex.what();
        }
        if (this->session == nullptr || this->session->empty())
            return "Failed to create a session";
        return std::string();
    }

    virtual const std::vector<IOInfo> &GetIOInfo(bool isOutput) const override
    {
        return isOutput ? this->output : this->input;
    }

    std::string _ExecAsync(const std::map<std::string, std::vector<cv::Mat>> &inputs,
                           cv::Size2i                                         size)
    {
        std::string err;
        if (inputs.size() != 1 || inputs.begin()->second.size() != 1)
            return "parameter error";
        for (auto &in : inputs) {
            InputData dat;
            dat.imgs = in.second;
            auto d   = GetImageBGRValue(dat.imgs, this->is_normal, size, err, dat.lets);
            if (d.empty())
                return err;
        }
        return err;
    }

    virtual std::string ExecAsync(const std::map<std::string, std::vector<cv::Mat>> &inputs,
                                  cv::Size2i                                         size) override
    {
        int flag = 0;
        if (!this->is_runing.compare_exchange_strong(flag, 1))
            return "A task is running";
        auto ret = _ExecAsync(inputs, size);
        this->is_runing.fetch_sub(1);
        return ret;
    }

    virtual bool IsRun() override
    {
        return this->is_runing.load() != 0;
    }
};
#endif

IRatiocinate *Ratiocinate_Create()
{
    return new Ratiocinate();
}
