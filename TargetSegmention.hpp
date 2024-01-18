/**
 * @file     TargetSegmention.hpp
 * @brief    目标分割
 * @author   CXS (chenxiangshu@outlook.com)
 * @version  1.0
 * @date     2024-01-15
 *
 * @copyright Copyright (c) 2024  Four-Faith
 *
 * @par 修改日志:
 * <table>
 * <tr><th>日期       <th>版本    <th>作者    <th>说明
 * <tr><td>2024-01-15 <td>1.0     <td>CXS     <td>创建
 * </table>
 */
#if !defined(__TargetSegmention_HPP__)
#define __TargetSegmention_HPP__
#include "Ratiocinate.hpp"
#include "TargetDetection.hpp"

namespace AIMethod {
    class TargetSegmention {
    private:
    public:
        /**
         * @brief    结果
         * @author   CXS (chenxiangshu@outlook.com)
         * @date     2024-01-09
         */
        class Result {
        public:
            int      classId;      // 类别
            float    confidence;   // 置信度
            cv::Rect box;          // 盒子信息
            cv::Mat  mask;         // [CV_8UC1]掩码（只有盒子范围）
        };

        /*
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
        */

        /**
         * @brief    Yolo分割
         * @param    det            检测对象
         * @param    pred           预测值
         * @param    proto          掩码原型
         * @param    lets           图像变形
         * @return   std::vector<std::vector<TargetSegmention::Result>>
         * @author   CXS (chenxiangshu@outlook.com)
         * @date     2024-01-17
         */
        std::vector<std::vector<TargetSegmention::Result>> Yolo(const TargetDetection               &det,
                                                                const Tensor<float>                 &pred,
                                                                const Tensor<float>                 &proto,
                                                                const std::vector<Tools::Letterbox> &lets) const;

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
#endif   // __TargetSegmention_HPP__
