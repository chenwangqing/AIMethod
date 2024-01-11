
#include "Ratiocinate.hpp"

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
    Ort::Session             *session = nullptr;
    Ort::Env                  env;
    Ort::MemoryInfo           memory{nullptr};
    std::vector<IOInfo>       input;
    std::vector<IOInfo>       output;
    std::vector<const char *> input_names;    // 输入名称
    std::vector<const char *> output_names;   // 输除名称

    struct
    {
        std::map<std::string, Tensor<float>> inputs_data;      // 输入数据
        std::vector<Ort::Value>              input_tensors;    // 输入张量
        std::vector<Ort::Value>              output_tensors;   // 输出张量
    } status;

    void ClearStatus()
    {
        this->status.input_tensors.clear();
        this->status.output_tensors.clear();
        this->status.inputs_data.clear();
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
            std::map<std::string, Tensor<float>> result;
            if (status.IsOK()) {
                for (size_t i = 0; i < infer->output_names.size(); i++) {
                    auto             data  = infer->status.output_tensors[i].GetTensorMutableData<float>();
                    auto             shape = infer->status.output_tensors[i].GetTensorTypeAndShapeInfo().GetShape();
                    std::vector<int> _shape(shape.size());
                    for (size_t j = 0; j < shape.size(); j++)
                        _shape[j] = shape[j];
                    result[infer->output_names[i]] = Tensor<float>(_shape, data);
                }
                infer->callback(infer, infer->status.inputs_data, result, infer->callback_context, std::string());
            } else {
                infer->callback(infer, infer->status.inputs_data, result, infer->callback_context, status.GetErrorMessage());
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
            this->input_names.push_back(this->input[this->input.size() - 1].name.c_str());
        }
        for (size_t i = 0; i < this->session->GetOutputCount(); i++) {
            IOInfo info;
            info.name  = this->session->GetOutputNameAllocated(i, allocator).get();
            info.shape = this->session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
            this->output.push_back(info);
            this->output_names.push_back(this->output[this->output.size() - 1].name.c_str());
        }
        // 创建内存分配器
        this->memory = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        return std::string();
    }

    virtual const std::vector<IOInfo> &GetIOInfo(bool isOutput) const override
    {
        return isOutput ? this->output : this->input;
    }

    std::string _ExecAsync(const std::map<std::string, Tensor<float>> &inputs)
    {
        if (input.size() == 0 || inputs.size() != this->input.size())
            return "The input parameter cannot be empty";
        // 清除数据
        ClearStatus();
        std::string err;
        // 设置输入
        for (size_t i = 0; i < this->input.size(); i++) {
            auto it = inputs.find(this->input[i].name);
            if (it == inputs.end())
                return "Missing parameter:" + this->input[i].name;
            auto shape  = it->second.GetShape<int64_t>();
            auto tensor = Ort::Value::CreateTensor<float>(this->memory,
                                                          (float *)it->second.Value(),
                                                          it->second.Size(),
                                                          shape.data(),
                                                          shape.size());
            this->status.input_tensors.push_back(std::move(tensor));
            this->status.inputs_data[it->first] = it->second;
        }
        // 设置输出
        for (auto &v : this->output)
            this->status.output_tensors.push_back(std::move(Ort::Value{nullptr}));
        // 执行
        try {
            this->is_runing.fetch_add(1);
            this->session->RunAsync(Ort::RunOptions{nullptr},
                                    this->input_names.data(),
                                    this->status.input_tensors.data(),
                                    this->status.input_tensors.size(),
                                    this->output_names.data(),
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

    virtual std::string ExecAsync(const std::map<std::string, Tensor<float>> &inputs) override
    {
        int flag = 0;
        if (!this->is_runing.compare_exchange_strong(flag, 1))
            return "A task is running";
        auto ret = _ExecAsync(inputs);
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
