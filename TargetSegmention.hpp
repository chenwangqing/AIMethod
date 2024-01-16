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
        class Result : public TargetDetection::Result {
        public:
        };

        /**
         * @brief    Yolo检测
         * @param    input          推理结果
         * @param    lets           图像形变
         * @return   std::vector<std::vector<TargetDetection::Result>>
         * @author   CXS (chenxiangshu@outlook.com)
         * @date     2024-01-10
         */
        std::vector<std::vector<TargetSegmention::Result>> Yolo(const TargetDetection               &det,
                                                                const Tensor<float>                 &pred,
                                                                const Tensor<float>                 &proto,
                                                                const std::vector<Tools::Letterbox> &lets) const;
    };
}   // namespace AIMethod
#endif   // __TargetSegmention_HPP__
