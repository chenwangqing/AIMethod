
#include "PoseEstimation.hpp"

namespace AIMethod {

    const int skeleton[][2] = {
        {PoseEstimation::Result::KP_L_EYE, PoseEstimation::Result::KP_R_EYE},   // 眼睛

        {PoseEstimation::Result::KP_L_EYE, PoseEstimation::Result::KP_L_EAR},   // 眼睛-耳朵
        {PoseEstimation::Result::KP_R_EYE, PoseEstimation::Result::KP_R_EAR},   // 眼睛-耳朵

        {PoseEstimation::Result::KP_L_EAR, PoseEstimation::Result::KP_L_SHOULDER},   // 耳朵-肩膀
        {PoseEstimation::Result::KP_R_EAR, PoseEstimation::Result::KP_R_SHOULDER},   // 耳朵-肩膀

        {PoseEstimation::Result::KP_L_SHOULDER, PoseEstimation::Result::KP_R_SHOULDER},   // 肩膀

        {PoseEstimation::Result::KP_L_SHOULDER, PoseEstimation::Result::KP_L_ELBOW},   // 肩膀-手肘
        {PoseEstimation::Result::KP_R_SHOULDER, PoseEstimation::Result::KP_R_ELBOW},   // 肩膀-手肘

        {PoseEstimation::Result::KP_L_ELBOW, PoseEstimation::Result::KP_L_WRIST},   // 手肘-手腕
        {PoseEstimation::Result::KP_R_ELBOW, PoseEstimation::Result::KP_R_WRIST},   // 手肘-手腕

        {PoseEstimation::Result::KP_L_SHOULDER, PoseEstimation::Result::KP_L_CROTCH},   // 肩膀-跨步
        {PoseEstimation::Result::KP_R_SHOULDER, PoseEstimation::Result::KP_R_CROTCH},   // 肩膀-跨步

        {PoseEstimation::Result::KP_L_CROTCH, PoseEstimation::Result::KP_R_CROTCH},   // 跨步

        {PoseEstimation::Result::KP_L_CROTCH, PoseEstimation::Result::KP_L_KNEE},   // 跨步-膝盖
        {PoseEstimation::Result::KP_R_CROTCH, PoseEstimation::Result::KP_R_KNEE},   // 跨步-膝盖

        {PoseEstimation::Result::KP_L_KNEE, PoseEstimation::Result::KP_L_ANKLE},   // 膝盖-脚踝
        {PoseEstimation::Result::KP_R_KNEE, PoseEstimation::Result::KP_R_ANKLE},   // 膝盖-脚踝
    };

    std::vector<PoseEstimation::Result> PoseEstimation::Yolo(const Tensor<float>    &input,
                                                             const Tools::Letterbox &let) const
    {
        if (input.GetShape().size() != 2 || input.GetShape()[1] != 57)
            return std::vector<Result>();
        // 进行检测 xywh 置信度 类别
        std::vector<Result> result;
        cv::Mat             img = cv::imread("./img/bus.jpg");
        for (int i = 0; i < input.GetShape()[0]; i++) {
            Result rs;
            auto   data   = input.Value() + input.GetIdx(i, 0);
            rs.confidence = data[4];
            // 获取盒子信息
            int   x1         = data[0];
            int   y1         = data[1];
            int   x2         = data[2];
            int   y2         = data[3];
            float confidence = data[4];   // 置信度
            if (x1 < 0 || y1 < 0 || x2 < x1 || y2 < y1 || confidence < 0.5f)
                continue;
            cv::Rect r;
            r.x      = x1;
            r.y      = y1;
            r.width  = x2 - x1;
            r.height = y2 - y1;
            rs.box   = let.Restore(r);
            // 获取姿态信息
            data += 6;
            auto &pts         = rs.kpts;
            auto &confidences = rs.confidences;
            for (size_t k = 0; k < 17; k++) {
                pts[k].x       = data[k * 3 + 0];
                pts[k].y       = data[k * 3 + 1];
                pts[k]         = let.Restore(pts[k]);
                confidences[k] = data[k * 3 + 2];
            }
            result.push_back(rs);
        }
        cv::imwrite("./output/pk.jpg", img);
        return result;
    }

    void PoseEstimation::DrawBox(cv::Mat                   &img,
                                 const std::vector<Result> &result)
    {
        static const cv::Scalar colors[] = {
            CV_RGB(220, 255, 183),   // 鼻子
            CV_RGB(255, 104, 104),   // 眼睛
            CV_RGB(255, 104, 104),   // 眼睛
            CV_RGB(255, 187, 100),   // 耳朵
            CV_RGB(255, 187, 100),   // 耳朵
            CV_RGB(126, 137, 183),   // 肩膀
            CV_RGB(126, 137, 183),   // 肩膀
            CV_RGB(29, 243, 83),     // 手肘
            CV_RGB(29, 243, 83),     // 手肘
            CV_RGB(237, 90, 179),    // 手腕
            CV_RGB(237, 90, 179),    // 手腕
            CV_RGB(134, 74, 249),    // 胯部
            CV_RGB(134, 74, 249),    // 胯部
            CV_RGB(73, 66, 228),     // 膝盖
            CV_RGB(73, 66, 228),     // 膝盖
            CV_RGB(230, 185, 222),   // 脚踝
            CV_RGB(230, 185, 222),   // 脚踝
        };
        for (auto &rs : result) {
            // 绘制框
            cv::rectangle(img, rs.box, CV_RGB(255, 255, 0), 2);
            // 绘制关键点
            for (int k = 0; k < 17; k++) {
                if (rs.confidences[k] < 0.5f) continue;
                cv::circle(img, rs.kpts[k], 5, colors[k], 2);
            }
            // 绘制连线
            for (size_t k = 0; k < TAB_SIZE(skeleton); k++) {
                auto              &sk    = skeleton[k];
                const cv::Point2i &pos1  = rs.kpts[sk[0]];
                const cv::Point2i &pos2  = rs.kpts[sk[1]];
                float              conf1 = rs.confidences[sk[0]];
                float              conf2 = rs.confidences[sk[1]];
                if (conf1 < 0.5f || conf2 < 0.5f)
                    continue;
                auto color = colors[sk[0]] * 0.5f + colors[sk[1]] * 0.5f;
                cv::line(img, pos1, pos2, color, 2);
            }
        }
        return;
    }

}   // namespace AIMethod
