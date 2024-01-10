
#include "TargetDetection.hpp"

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
static std::vector<float> GetImageValue(const std::vector<cv::Mat> &imgs,
                                        bool is_normal,
                                        cv::Size2i shape,
                                        std::string &err,
                                        std::vector<Tools::Letterbox> &let_box)
{
    int block_size = shape.height * shape.width * 3;
    std::vector<float> tmp(imgs.size() * block_size);
    cv::Mat img_f32;
    for (size_t i = 0; i < imgs.size(); i++)
    {
        auto img = imgs[i];
        float *data = tmp.data() + i * block_size;
        if (img.size().empty())
        {
            err = "The picture cannot be empty";
            return std::vector<float>();
        }
        if (img.type() != CV_32FC3)
        {
            img.convertTo(img_f32, CV_32FC3); // 转float
            img = img_f32;
        }
        if (img.channels() != 3)
        {
            err = "The picture must be 3 channels";
            return std::vector<float>();
        }
        if (img.type() != CV_32FC3)
        {
            err = "Image data type conversion failed. Procedure";
            return std::vector<float>();
        }
        // 图像变换
        Tools::Letterbox let;
        if (img.size() != shape)
            img = Tools::Letterbox::Make(img, shape.height, shape.width, let);
        let_box.push_back(let);
        // BGR2RGB
        for (int i = 0; i < img.rows; i++)
        {
            for (int j = 0; j < img.cols; j++)
            {
                auto &tmp = img_f32.at<cv::Vec3f>(i, j);
                data[i * img.cols + j + 0] = tmp[2];
                data[i * img.cols + j + 1 * img.cols * img.rows] = tmp[1];
                data[i * img.cols + j + 2 * img.cols * img.rows] = tmp[0];
            }
        }
    }
    if (is_normal)
    {
        // 归一化
        float *data = tmp.data();
        for (size_t i = 0; i < tmp.size(); i++, data++)
            *data /= 255.0f;
    }
    return tmp;
}

/**
 * @brief    获取结果
 * @param    dims
 * @param    data
 * @param    let_box
 * @param    confidence_threshold
 * @param    nms_threshold
 * @return   std::vector<std::vector<TargetDetection::Result>>
 * @author   CXS (chenxiangshu@outlook.com)
 * @date     2024-01-09
 */
static std::vector<std::vector<TargetDetection::Result>> GetResult(const std::vector<int64_t> &dims,
                                                                   const float *data,
                                                                   const std::vector<Tools::Letterbox> &let_box,
                                                                   float confidence_threshold,
                                                                   float nms_threshold)
{
    std::vector<std::vector<TargetDetection::Result>> result;
    if (dims.size() != 3 || dims[2] <= 5)
        return result;
    for (int64_t k = 0; k < dims[0]; k++, data += dims[1] * dims[2])
    {
        // 解析 x,y,w,h,目标框概率,类别0概率，类别1概率,...
        std::vector<cv::Rect> boxs;
        std::vector<int> classIds;
        std::vector<float> confidences;
        for (int64_t i = 0; i < dims[1]; i++)
        {
            auto detection = data + i * dims[2];
            // 获取每个类别置信度
            auto scores = detection + 5;
            // 获取概率最大的一类
            int classID = 0;
            int cnt = dims[2] - 5;
            for (int n = 1; n < cnt; n++)
            {
                if (scores[n] > scores[classID])
                    classID = n;
            }
            // 置信度为类别的概率和目标框概率值得乘积
            float confidence = scores[classID] * detection[4];
            // 概率太小不要
            if (confidence < confidence_threshold)
                continue;
            // 获取盒子信息
            int centerX = detection[0];
            int centerY = detection[1];
            int width = detection[2];
            int height = detection[3];
            cv::Rect r;
            r.x = centerX - (width >> 1);
            r.y = centerY - (height >> 1);
            r.width = width;
            r.height = height;
            boxs.push_back(r);
            classIds.push_back(classID);
            confidences.push_back(confidence);
        }
        // NMS处理
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxs, confidences, confidence_threshold, nms_threshold, indices);
        std::vector<TargetDetection::Result> rs;
        auto &let = let_box[k];
        for (auto idx : indices)
        {
            TargetDetection::Result t;
            t.box = let.Restore(boxs[idx]); // 还原坐标
            t.classId = classIds[idx];
            t.confidence = confidences[idx];
            rs.push_back(t);
        }
        result.push_back(std::move(rs));
    }
    return result;
}

#if EN_ONNXRUNTIME
#include "onnxruntime_cxx_api.h"
class TargetDetection_Inst : public TargetDetection
{
private:
    Ort::Session *session = nullptr;
    std::string input_name;
    std::string output_name;
    cv::Size2i input_shape;
    Ort::Env env;
    bool is_dynamic_input;

    const std::vector<const cv::Mat> imgs;
    std::vector<float> input_data;

    std::vector<std::vector<Result>> results;
    Ort::Value output_tensor{nullptr};
    Ort::Value input_tensor{nullptr};

    virtual std::string Load_Weight(const char *filename, size_t threads) override
    {
        Ort::SessionOptions options;
        options.SetIntraOpNumThreads(threads); // 设置线程数量
        this->session = new Ort::Session{this->env, filename, options};
        if (this->session == nullptr)
            return "Failed to create a session";
        if (this->session->GetInputCount() != 1)
            return "The number of model inputs is not 1";
        if (this->session->GetOutputCount() != 1)
            return "The number of model outputs is not 1";
        // 获取输入/输出信息
        Ort::AllocatorWithDefaultOptions allocator;
        this->input_name = this->session->GetInputNameAllocated(0, allocator).get();
        this->output_name = this->session->GetOutputNameAllocated(0, allocator).get();
        auto shape = this->session->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        if (shape.size() != 4 || shape[1] != 3)
            return "Input dimension error";
        this->is_dynamic_input = shape[0] == 0;
        this->input_shape.height = shape[2];
        this->input_shape.width = shape[3];
        shape = this->session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        if (shape.size() != 3 || shape[2] <= 5)
            return "Output dimension error";
        return std::string();
    }

    virtual cv::Size2i GetInputShape() const override
    {
        return this->input_shape;
    }

    void Finish()
    {
    }

    static void RunAsyncCallbackFn(void *user_data, OrtValue **outputs, size_t num_outputs, OrtStatusPtr status_ptr)
    {
        TargetDetection_Inst *det = static_cast<TargetDetection_Inst *>(user_data);
        Ort::Status status(status_ptr);
        if (!status.IsOK())
        {
        }

        // 获取结果
        // auto ret = GetResult(output_shape,
        //                      output_tensor.GetTensorMutableData<float>(),
        //                      let_box,
        //                      this->confidence_threshold,
        //                      this->nms_threshold);
        // if (ret.size() == 0)
        //     return "Parsing result failure";
        // results.push_back(std::move(ret[0]));
        // if (det->callback != nullptr)
        //     det->callback(det, det->results, det->user_context);
        return;
    }

public:
    TargetDetection_Inst()
    {
    }

    virtual ~TargetDetection_Inst()
    {
        if (this->session != nullptr)
            delete this->session;
        return;
    }

    virtual std::string DetectionAsync(const std::vector<cv::Mat> &imgs,
                                       bool is_normal) override
    {
        if (imgs.size() == 0)
            return "parameter error";
        std::vector<Tools::Letterbox> let_box;
        std::string err;
        this->isRun = true;
        if (this->is_dynamic_input)
        {
#if 0
            // 获取输入数据
            this->input_data = GetImageValue(imgs, is_normal, this->input_shape, err, let_box);
            if (this->input_data.size() == 0)
                return err;
            // 创建内存分配器
            auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
            // 创建张量
            int64_t input_shape[4] = {(int64_t)imgs.size(), 3, this->input_shape.height, this->input_shape.width};
            auto output_shape = this->session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
            output_shape[0] = (int64_t)imgs.size();
            int64_t output_size = output_shape[0] * output_shape[1] * output_shape[2];
            if (this->output_data.size() < output_size)
                this->output_data = std::vector<float>(output_size);
            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info,
                                                                      this->input_data.data(),
                                                                      this->input_data.size(),
                                                                      input_shape,
                                                                      4);
            Ort::Value output_tensor = Ort::Value::CreateTensor<float>(memory_info,
                                                                       this->output_data.data(),
                                                                       output_size,
                                                                       output_shape.data(),
                                                                       output_shape.size());
            const char *input_names[] = {this->input_name.c_str()};   // 输入节点名
            const char *output_names[] = {this->output_name.c_str()}; // 输出节点名
            // 执行
            try
            {
                this->session->RunAsync(Ort::RunOptions{nullptr},
                                        input_names,
                                        &input_tensor,
                                        1,
                                        output_names,
                                        &output_tensor,
                                        1, );
            }
            catch (std::exception &e)
            {
                err = e.what();
                return err;
            }
            auto ret = GetResult(output_shape,
                                 output_tensor.GetTensorMutableData<float>(),
                                 let_box,
                                 this->confidence_threshold,
                                 this->nms_threshold);
            this->input_data.clear();
            return err;
#endif
            return "";
        }
        else
        {
            std::vector<std::vector<Result>> results;
            std::vector<cv::Mat> tmp(1);
            for (auto &img : imgs)
            {
                // 获取输入数据
                tmp[0] = img;
                this->input_data = GetImageValue(tmp, is_normal, this->input_shape, err, let_box);
                if (this->input_data.size() == 0)
                    return err;
                // 创建内存分配器
                auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
                // 创建张量
                int64_t input_shape[4] = {1, 3, this->input_shape.height, this->input_shape.width};
                auto output_shape = this->session->GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
                this->input_tensor = Ort::Value::CreateTensor<float>(memory_info,
                                                                     this->input_data.data(),
                                                                     this->input_data.size(),
                                                                     input_shape,
                                                                     4);
                this->output_tensor.release();
                const char *input_names[] = {this->input_name.c_str()};   // 输入节点名
                const char *output_names[] = {this->output_name.c_str()}; // 输出节点名
                // 执行
                try
                {
                    this->session->RunAsync(Ort::RunOptions{nullptr},
                                            input_names,
                                            &input_tensor,
                                            1,
                                            output_names,
                                            &this->output_tensor,
                                            1,
                                            RunAsyncCallbackFn,
                                            this);
                }
                catch (std::exception &e)
                {
                    err = e.what();
                    return err;
                }
            }
            // this->input_data.clear();
        }
        this->isRun = false;
        return std::string();
    }

    virtual void CleanCache()
    {
        return;
    }
};
#endif

TargetDetection *TargetDetection::Make(const char *weight, size_t threads, std::string &err)
{
    if (weight == nullptr)
        return nullptr;
    TargetDetection *target = new TargetDetection_Inst();
    err = target->Load_Weight(weight, threads);
    if (err != "")
    {
        delete target;
        return nullptr;
    }
    return target;
}

void TargetDetection::DrawBox(cv::Mat &img,
                              const std::vector<TargetDetection::Result> &result,
                              const cv::Scalar &color)
{
    for (size_t i = 0; i < result.size(); i++)
    {
        auto &box = result[i].box;
        cv::rectangle(img, box, color, 2);
        char str[255];
        snprintf(str, 255, "%d(%.2f)", result[i].classId, result[i].confidence);
        cv::putText(img, str, {box.x, box.y - 10}, cv::FONT_HERSHEY_SIMPLEX, 0.5, color);
    }
    return;
}
