/**
 * @file     TargetDetection.hpp
 * @brief    目标检测
 * @author   CXS (chenxiangshu@outlook.com)
 * @version  1.0
 * @date     2024-01-09
 *
 * @copyright Copyright (c) 2024  Four-Faith
 *
 * @par 修改日志:
 * <table>
 * <tr><th>日期       <th>版本    <th>作者    <th>说明
 * <tr><td>2024-01-09 <td>1.0     <td>CXS     <td>创建
 * </table>
 */
#if !defined(__TargetDetection_HPP__)
#define __TargetDetection_HPP__
#include "Ratiocinate.hpp"

namespace AIMethod {
    /**
     * @brief    目标检测
     * @author   CXS (chenxiangshu@outlook.com)
     * @date     2024-01-09
     */
    class TargetDetection {
    public:
        /**
         * @brief    结果
         * @author   CXS (chenxiangshu@outlook.com)
         * @date     2024-01-09
         */
        class Result {
        public:
            int      _index;   // 数据索引
            cv::Rect _box;     // 原始盒子

            int      classId;      // 类别
            float    confidence;   // 置信度
            cv::Rect box;          // 盒子信息
        };

        float confidence_threshold = 0.25f;   // 置信度阈值
        float nms_threshold        = 0.2f;    // NMS算法阈值

        /**
         * @brief    Yolo检测
         * @param    input          推理结果
         * @param    lets           图像形变
         * @return   std::vector<std::vector<TargetDetection::Result>>
         * @author   CXS (chenxiangshu@outlook.com)
         * @date     2024-01-10
         */
        std::vector<std::vector<TargetDetection::Result>> Yolo(const Tensor<float>                 &input,
                                                               const std::vector<Tools::Letterbox> &lets,
                                                               int                                  nm = 0) const;

        /**
         * @brief    画盒子
         * @param    img            图像
         * @param    result         结果
         * @param    color          颜色
         * @author   CXS (chenxiangshu@outlook.com)
         * @date     2024-01-10
         */
        static void DrawBox(cv::Mat                   &img,
                            const std::vector<Result> &result,
                            const cv::Scalar          &color = cv::Scalar(0, 0, 255));
    };
}   // namespace AIMethod
#endif   // __TargetDetection_HPP__
