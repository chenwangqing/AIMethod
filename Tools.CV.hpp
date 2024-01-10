/**
 * @file     Tools.CV.hpp
 * @brief    OpenCV工具
 * @author   CXS (chenxiangshu@outlook.com)
 * @version  1.0
 * @date     2024-01-08
 *
 * @copyright Copyright (c) 2024  Four-Faith
 *
 * @par 修改日志:
 * <table>
 * <tr><th>日期       <th>版本    <th>作者    <th>说明
 * <tr><td>2024-01-08 <td>1.0     <td>CXS     <td>创建
 * </table>
 */
#if !defined(__Tools_CV_hpp__)
#define __Tools_CV_hpp__
#include <stdint.h>
#include <opencv4/opencv2/opencv.hpp>

namespace Tools
{
    /**
     * @brief    图像Letterbox处理
     * @author   CXS (chenxiangshu@outlook.com)
     * @date     2024-01-08
     */
    class Letterbox
    {
    private:
        int _fill_width = 0;
        int _fill_height = 0;
        float _r = 1.0f;

    public:
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
        cv::Rect Restore(const cv::Rect &box) const;
    };
}

#endif // __Tools_CV_hpp__
