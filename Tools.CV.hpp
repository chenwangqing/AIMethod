/**
 * @file     Tools.CV.hpp
 * @brief    OpenCV工具
 * @author   CXS (chenxiangshu@outlook.com)
 * @version  1.1
 * @date     2024-01-08
 *
 * @copyright Copyright (c) 2024  Four-Faith
 *
 * @par 修改日志:
 * <table>
 * <tr><th>日期       <th>版本    <th>作者    <th>说明
 * <tr><td>2024-01-08 <td>1.0     <td>CXS     <td>创建
 * <tr><td>2024-01-17 <td>1.1     <td>CXS     <td>修正Letterbox::Restore越界错误
 * </table>
 */
#if !defined(__Tools_CV_hpp__)
#define __Tools_CV_hpp__
#include "Tools.hpp"
#include <opencv4/opencv2/opencv.hpp>

namespace Tools {
    /**
     * @brief    图像Letterbox处理
     * @author   CXS (chenxiangshu@outlook.com)
     * @date     2024-01-08
     */
    class Letterbox {
    private:
        int   _fill_width  = 0;
        int   _fill_height = 0;
        float _r           = 1.0f;

    public:
        int width;        // 原图像
        int height;       // 原图像
        int let_width;    // 变换后的图像
        int let_height;   // 变换后的图像

        /**
         * @brief    图像Letterbox处理
         * @param    src            源图像
         * @param    h              目标高度
         * @param    w              目标宽度
         * @param    let            变换信息
         * @return   cv::Mat
         * @author   CXS (chenxiangshu@outlook.com)
         * @date     2024-01-08
         */
        static cv::Mat Make(const cv::Mat &src, int h, int w, Letterbox &let);

        /**
         * @brief    图像Letterbox处理
         * @param    src            源图像
         * @param    h              目标高度
         * @param    w              目标宽度
         * @return   cv::Mat
         * @author   CXS (chenxiangshu@outlook.com)
         * @date     2024-01-08
         */
        static cv::Mat Make(const cv::Mat &src, int h, int w)
        {
            Letterbox let;
            return Make(src, h, w, let);
        }

        /**
         * @brief    还原坐标
         * @param    box            图像盒子
         * @return   cv::Rect
         * @author   CXS (chenxiangshu@outlook.com)
         * @date     2024-01-08
         */
        cv::Rect  Restore(const cv::Rect &box) const;
        cv::Point Restore(const cv::Point &pt) const;
    };

    /**
     * @brief    图片转张量
     * @param    imgs           图片列表
     * @param    size           转换大小
     * @param    lets           转换形变
     * @param    err            错误信息
     * @return   Tensor<float>
     * @author   CXS (chenxiangshu@outlook.com)
     * @date     2024-01-11
     */
    extern AIMethod::Tensor<float> ImageBGRToNCHW(const std::vector<cv::Mat>    &imgs,
                                                  const cv::Size2i              &size,
                                                  std::vector<Tools::Letterbox> &lets,
                                                  std::string                   &err);

}   // namespace Tools

#endif   // __Tools_CV_hpp__
