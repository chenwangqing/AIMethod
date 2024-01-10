
#include "TargetDetection.hpp"

void TargetDetection::DrawBox(cv::Mat                                    &img,
                              const std::vector<TargetDetection::Result> &result,
                              const cv::Scalar                           &color)
{
    for (size_t i = 0; i < result.size(); i++) {
        auto &box = result[i].box;
        cv::rectangle(img, box, color, 2);
        char str[255];
        snprintf(str, 255, "%d(%.2f)", result[i].classId, result[i].confidence);
        cv::putText(img, str, {box.x, box.y - 10}, cv::FONT_HERSHEY_SIMPLEX, 0.5, color);
    }
    return;
}

std::vector<std::vector<TargetDetection::Result>> TargetDetection::Yolo(const IRatiocinate::Result          &input,
                                                                        const std::vector<Tools::Letterbox> &lets)
{
    std::vector<std::vector<TargetDetection::Result>> result;

    auto &dims = input.shape;
    auto  data = input.data;
    if (dims.size() != 3 || dims[2] <= 5 || lets.size() != dims[0])
        return result;
    for (int64_t k = 0; k < dims[0]; k++, data += dims[1] * dims[2]) {
        // 解析 x,y,w,h,目标框概率,类别0概率，类别1概率,...
        std::vector<cv::Rect> boxs;
        std::vector<int>      classIds;
        std::vector<float>    confidences;
        for (int64_t i = 0; i < dims[1]; i++) {
            auto detection = data + i * dims[2];
            // 获取每个类别置信度
            auto scores = detection + 5;
            // 获取概率最大的一类
            int classID = 0;
            int cnt     = dims[2] - 5;
            for (int n = 1; n < cnt; n++) {
                if (scores[n] > scores[classID])
                    classID = n;
            }
            // 置信度为类别的概率和目标框概率值得乘积
            float confidence = scores[classID] * detection[4];
            // 概率太小不要
            if (confidence < confidence_threshold)
                continue;
            // 获取盒子信息
            int      centerX = detection[0];
            int      centerY = detection[1];
            int      width   = detection[2];
            int      height  = detection[3];
            cv::Rect r;
            r.x      = centerX - (width >> 1);
            r.y      = centerY - (height >> 1);
            r.width  = width;
            r.height = height;
            boxs.push_back(r);
            classIds.push_back(classID);
            confidences.push_back(confidence);
        }
        // NMS处理
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxs, confidences, confidence_threshold, nms_threshold, indices);
        std::vector<TargetDetection::Result> rs;
        auto                                &let = lets[k];
        for (auto idx : indices) {
            TargetDetection::Result t;
            t.box        = let.Restore(boxs[idx]);   // 还原坐标
            t.classId    = classIds[idx];
            t.confidence = confidences[idx];
            rs.push_back(t);
        }
        result.push_back(std::move(rs));
    }
    return result;
}
