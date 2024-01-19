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
     * @brief    图像处理
     * @author   CXS (chenxiangshu@outlook.com)
     * @date     2024-01-19
     */
    class IMGProcess {
    private:
    public:
        /**
         * @brief    自适应直方图均衡
         * @param    src            原图    [8UC1/16UC1]
         * @param    dst            目标    [SRC]
         * @param    limit          设置对比度限制的阈值
         * @param    size           设置直方图均衡化的网格大小。输入图像将被分成大小相等的矩形块。
         * @note     RGB图像转换为YCbCr并仅对Y通道进行直方图均衡化来实现彩色图像的均衡化
         * @author   CXS (chenxiangshu@outlook.com)
         * @date     2024-01-19
         * @example
            auto                 img = cv::imread("./img/zidane.jpg");
            std::vector<cv::Mat> channels;
            cv::cvtColor(img, img, cv::COLOR_BGR2YCrCb);
            cv::split(img, channels);
            Tools::IMGProcess::AdaptiveHistogramEqualization(channels[0], channels[0]);
            cv::merge(channels, img);
            cv::cvtColor(img, img, cv::COLOR_YCrCb2BGR);
            cv::imwrite("./output/zidane.jpg", img);
         */
        static void AdaptiveHistogramEqualization(const cv::Mat    &src,
                                                  cv::Mat          &dst,
                                                  int               limit = 4,
                                                  const cv::Size2i &size  = {8, 8});

        /**
         * @brief    伽马变换
         * @param    src            原      [8UC1]
         * @param    dst            目标    [8UC1]
         * @param    gamma          小于1变亮 大于1变暗
         * @author   CXS (chenxiangshu@outlook.com)
         * @date     2024-01-19
         */
        static void AdjustGamma(const cv::Mat &src, cv::Mat &dst, float gamma);

        /**
         * @brief    自适应中值滤波器
         * @param    src            原  [8U]
         * @param    minSize        最小窗口
         * @param    maxSize        最大窗口
         * @return   cv::Mat
         * @author   CXS (chenxiangshu@outlook.com)
         * @date     2024-01-19
         */
        static cv::Mat AdaptiveMediaFilter(const cv::Mat &src, int minSize = 3, int maxSize = 7);
    };

    /**
     * @brief    图片转张量
     * @param    imgs           图片列表    [8UC3:BGR]
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
