/**
 * @file     PoseEstimation.hpp
 * @brief    姿态估计
 * @author   CXS (chenxiangshu@outlook.com)
 * @version  1.0
 * @date     2024-01-18
 *
 * @copyright Copyright (c) 2024  Four-Faith
 *
 * @par 修改日志:
 * <table>
 * <tr><th>日期       <th>版本    <th>作者    <th>说明
 * <tr><td>2024-01-18 <td>1.0     <td>CXS     <td>创建
 * </table>
 */
#if !defined(__POSEESTIMATION_HP__)
#define __POSEESTIMATION_HP__
#include "Ratiocinate.hpp"
#include "TargetDetection.hpp"

namespace AIMethod {
    /**
     * @brief    姿态估计
     * @author   CXS (chenxiangshu@outlook.com)
     * @date     2024-01-18
     */
    class PoseEstimation {
    private:
    public:
        /**
         * @brief    姿态估计
         * @author   CXS (chenxiangshu@outlook.com)
         * @date     2024-01-18
         */
        class Result {
        public:
            enum
            {
                KP_NOSE       = 0,    // 鼻子
                KP_L_EYE      = 1,    // 左眼
                KP_R_EYE      = 2,    // 右眼
                KP_L_EAR      = 3,    // 左耳
                KP_R_EAR      = 4,    // 右耳
                KP_L_SHOULDER = 5,    // 左肩
                KP_R_SHOULDER = 6,    // 右肩
                KP_L_ELBOW    = 7,    // 左肘
                KP_R_ELBOW    = 8,    // 右肘
                KP_L_WRIST    = 9,    // 左腕
                KP_R_WRIST    = 10,   // 右腕
                KP_L_CROTCH   = 11,   // 左胯
                KP_R_CROTCH   = 12,   // 右胯
                KP_L_KNEE     = 13,   // 左膝
                KP_R_KNEE     = 14,   // 右膝
                KP_L_ANKLE    = 15,   // 左踝
                KP_R_ANKLE    = 16,   // 右踝
            };

            float       confidence;        // 置信度
            cv::Rect    box;               // 盒子信息
            cv::Point2i kpts[17];          // 关键点
            float       confidences[17];   // 关键点置信度
        };

        /**
         * @brief    YOLO 检测
         * @param    detections     推理结果
         * @param    let            图像形变
         * @author   CXS (chenxiangshu@outlook.com)
         * @date     2024-01-18
         */
        std::vector<Result> Yolo(const Tensor<float>    &detections,
                                 const Tools::Letterbox &let) const;

        /**
         * @brief    画盒子
         * @param    img            图像
         * @param    result         结果
         * @param    color          颜色
         * @author   CXS (chenxiangshu@outlook.com)
         * @date     2024-01-10
         */
        static void DrawBox(cv::Mat                   &img,
                            const std::vector<Result> &result);
    };
}   // namespace AIMethod
#endif   // __POSEESTIMATION_HP__
