
#include "Ratiocinate.hpp"
#include <thread>

namespace AIMethod {

#if CFG_INFER_ENGINE == INFER_ENGINE_ONNXRUNTIME
#include "onnxruntime_cxx_api.h"
    class Ratiocinate : public IRatiocinate {
    private:
        Ort::Session   *session = nullptr;
        Ort::Env        env;
        Ort::MemoryInfo memory{nullptr};

        class Status : public IStatus {
        public:
            std::vector<Ort::Value>   input_values;
            std::vector<Ort::Value>   output_values;
            std::vector<const char *> _input_names;
            std::vector<const char *> _output_names;
        };

        static void RunAsyncCallbackFn(void        *user_data,
                                       OrtValue   **outputs,
                                       size_t       num_outputs,
                                       OrtStatusPtr status_ptr)
        {
            Status      *sta   = static_cast<Status *>(user_data);
            Ratiocinate *infer = dynamic_cast<Ratiocinate *>(sta->infer);
            Ort::Status  status(status_ptr);
            if (infer->callback != nullptr) {
                std::vector<Result> result;
                if (status.IsOK()) {
                    for (size_t i = 0; i < sta->output_names.size(); i++) {
                        auto             data  = sta->output_values[i].GetTensorMutableData<float>();
                        auto             shape = sta->output_values[i].GetTensorTypeAndShapeInfo().GetShape();
                        Result           rs;
                        std::vector<int> _shape(shape.size());
                        for (size_t j = 0; j < shape.size(); j++)
                            _shape[j] = shape[j];
                        rs.shape = std::move(_shape);
                        rs.data  = data;
                        result.push_back(std::move(rs));
                    }
                    infer->callback(infer,
                                    sta->input_names,
                                    sta->input_datas,
                                    sta->output_names,
                                    result,
                                    infer->callback_context,
                                    std::string());
                } else {
                    infer->callback(infer,
                                    sta->input_names,
                                    sta->input_datas,
                                    sta->output_names,
                                    result,
                                    infer->callback_context,
                                    status.GetErrorMessage());
                }
            }
            infer->is_runing.fetch_sub(1);
            delete sta;
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
            // 创建内存分配器
            this->memory = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
            return std::string();
        }

        std::string _ExecAsync(const std::vector<std::string>   &input_names,
                               const std::vector<std::string>   &output_names,
                               const std::vector<Tensor<float>> &input_datas)
        {
            if (input_names.size() == 0 || input_names.size() != input_datas.size() || output_names.size() == 0)
                return "The input parameter cannot be empty";
            Status     *status = new Status();
            std::string err;
            status->infer        = this;
            status->input_datas  = input_datas;
            status->input_names  = input_names;
            status->output_names = output_names;
            status->_input_names.resize(input_names.size());
            status->_output_names.resize(output_names.size());
            // 设置输入
            for (size_t i = 0; i < status->input_names.size(); i++) {
                auto shape  = input_datas[i].GetShape<int64_t>();
                auto tensor = Ort::Value::CreateTensor<float>(this->memory,
                                                              (float *)status->input_datas[i].Value(),
                                                              status->input_datas[i].Size(),
                                                              shape.data(),
                                                              shape.size());
                status->input_values.push_back(std::move(tensor));
                status->_input_names[i] = status->input_names[i].c_str();
            }
            // 设置输出
            for (size_t i = 0; i < status->output_names.size(); i++) {
                status->output_values.push_back(std::move(Ort::Value{nullptr}));
                status->_output_names[i] = status->output_names[i].c_str();
            }
            // 执行
            try {
                this->is_runing.fetch_add(1);
                this->session->RunAsync(Ort::RunOptions{nullptr},
                                        status->_input_names.data(),
                                        status->input_values.data(),
                                        status->input_values.size(),
                                        status->_output_names.data(),
                                        status->output_values.data(),
                                        status->output_values.size(),
                                        RunAsyncCallbackFn,
                                        status);
            }
            catch (std::exception &e) {
                err = e.what();
                this->is_runing.fetch_sub(1);
                delete status;
            }
            return err;
        }

        virtual std::string ExecAsync(const std::vector<std::string>   &input_name,
                                      const std::vector<std::string>   &output_name,
                                      const std::vector<Tensor<float>> &input_data) override
        {
            int flag = 0;
            if (!this->is_runing.compare_exchange_strong(flag, 1))
                return "A task is running";
            auto ret = _ExecAsync(input_name, output_name, input_data);
            if (!ret.empty() && this->callback != nullptr) {
                const std::vector<Result> tmp;
                this->callback(this, input_name, input_data, output_name, tmp, this->callback_context, ret);
            }
            this->is_runing.fetch_sub(1);
            return ret;
        }

        virtual bool IsRun() override
        {
            return this->is_runing.load() != 0;
        }
    };
#elif CFG_INFER_ENGINE == INFER_ENGINE_OPENCV
#include <opencv4/opencv2/dnn.hpp>
    class Ratiocinate : public IRatiocinate {
    private:
        cv::dnn::Net *session = nullptr;

        class Status : public IStatus {};

    public:
        virtual ~Ratiocinate()
        {
            if (this->session != nullptr)
                delete this->session;
            return;
        }

        virtual std::string LoadModel(const Parameters &params) override
        {
            try {
                this->session = new cv::dnn::Net(cv::dnn::readNetFromONNX(params.model));
                this->session->setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
                this->session->setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            }
            catch (std::exception ex) {
                return ex.what();
            }
            if (this->session == nullptr || this->session->empty())
                return "Failed to create a session";
            return std::string();
        }

        static void exec(Status *status)
        {
            if (status == nullptr) return;
            Ratiocinate               *infer        = dynamic_cast<Ratiocinate *>(status->infer);
            auto                      &input_names  = status->input_names;
            auto                      &input_datas  = status->input_datas;
            auto                      &output_names = status->output_names;
            std::string                err;
            std::vector<Tensor<float>> output_datas;
            try {
                for (size_t i = 0; i < input_names.size(); i++) {
                    auto    shape = input_datas[i].GetShape();
                    cv::Mat input(shape, CV_32F);
                    auto    s = input_datas[i].Size();
                    memcpy(input.data, input_datas[i].Value(), s * sizeof(float));
                    infer->session->setInput(input, input_names[i].c_str());
                }

                for (size_t i = 0; i < output_names.size(); i++) {
                    auto             data = infer->session->forward(output_names[i].c_str());
                    std::vector<int> shape(data.size.p, data.size.p + data.dims);
                    output_datas.push_back(Tensor<float>(shape, (float *)data.data));
                }
            }
            catch (std::exception &ex) {
                err = ex.what();
            }
            if (infer->callback != nullptr) {
                std::vector<Result> result;
                for (size_t i = 0; i < output_datas.size(); i++) {
                    Result rs;
                    rs.shape = output_datas[i].GetShape();
                    rs.data  = output_datas[i].Value();
                    result.push_back(rs);
                }
                infer->callback(infer,
                                input_names,
                                input_datas,
                                output_names,
                                result,
                                infer->callback_context,
                                err);
            }
            infer->is_runing.fetch_sub(1);
            delete status;
            return;
        }

        std::string _ExecAsync(const std::vector<std::string>   &input_names,
                               const std::vector<std::string>   &output_names,
                               const std::vector<Tensor<float>> &input_datas)
        {
            if (input_names.size() == 0 || input_names.size() != input_datas.size() || output_names.size() == 0)
                return "The input parameter cannot be empty";

            Status *status       = new Status();
            status->infer        = this;
            status->input_datas  = input_datas;
            status->input_names  = input_names;
            status->output_names = output_names;

            try {
                this->is_runing.fetch_add(1);
                std::thread thr(exec, status);
                thr.detach();   // 分离线程
            }
            catch (std::exception &ex) {
                this->is_runing.fetch_sub(0);
                delete status;
                return ex.what();
            }
            return std::string();
        }

        virtual std::string ExecAsync(const std::vector<std::string>   &input_name,
                                      const std::vector<std::string>   &output_name,
                                      const std::vector<Tensor<float>> &input_data) override
        {
            int flag = 0;
            if (!this->is_runing.compare_exchange_strong(flag, 1))
                return "A task is running";
            auto ret = _ExecAsync(input_name, output_name, input_data);
            if (!ret.empty() && this->callback != nullptr) {
                const std::vector<Result> tmp;
                this->callback(this, input_name, input_data, output_name, tmp, this->callback_context, ret);
            }
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
}   // namespace AIMethod
