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
#include "Tools.CV.hpp"

#ifndef EN_ONNXRUNTIME
#define EN_ONNXRUNTIME 1
#endif

#ifndef EN_GPU
#define EN_GPU 0
#endif

/**
 * @brief    目标检测
 * @author   CXS (chenxiangshu@outlook.com)
 * @date     2024-01-09
 */
class TargetDetection
{
protected:
    bool isRun = false;

    /**
     * @brief    加载权重
     * @param    filename       权重文件
     * @return   true
     * @return   false
     * @author   CXS (chenxiangshu@outlook.com)
     * @date     2024-01-09
     */
    virtual std::string Load_Weight(const char *filename, size_t threads)
    {
        return std::string();
    }

public:
    float confidence_threshold = 0.25f; // 置信度阈值
    float nms_threshold = 0.2f;         // NMS算法阈值

    virtual ~TargetDetection() {}

    /**
     * @brief    获取输入形状
     * @return   Shape
     * @author   CXS (chenxiangshu@outlook.com)
     * @date     2024-01-09
     */
    virtual cv::Size2i GetInputShape() const
    {
        return cv::Size2i();
    }

    /**
     * @brief    结果
     * @author   CXS (chenxiangshu@outlook.com)
     * @date     2024-01-09
     */
    typedef struct
    {
        int classId;      // 类别
        float confidence; // 置信度
        cv::Rect box;     // 盒子信息
    } Result;

    /**
     * @brief    检测回调
     * @author   CXS (chenxiangshu@outlook.com)
     * @date     2024-01-09
     */
    typedef void (*Detection_Callback)(TargetDetection *det,
                                       std::vector<std::vector<Result>> &results,
                                       void *context,
                                       const std::string &err);

protected:
    Detection_Callback callback = nullptr;
    void *user_context = nullptr;

public:
    /**
     * @brief    设置回调
     * @param    callback
     * @author   CXS (chenxiangshu@outlook.com)
     * @date     2024-01-09
     */
    void SetDetectionCallback(Detection_Callback callback,
                              void *context)
    {
        this->callback = callback;
        this->user_context = context;
        return;
    }

    /**
     * @brief    检测
     * @param    imgs           输入图片 BGR 格式
     * @param    is_normal      是否归一化
     * @param    err            错误信息
     * @return   std::vector<std::vector<Result>>
     * @author   CXS (chenxiangshu@outlook.com)
     * @date     2024-01-09
     */
    virtual std::string DetectionAsync(const std::vector<cv::Mat> &imgs,
                                       bool is_normal)
    {
        return "Function not implemented";
    }

    /**
     * @brief    检测
     * @param    imgs           输入图片 BGR 格式
     * @param    is_normal      是否归一化
     * @return   std::vector<std::vector<Result>>
     * @author   CXS (chenxiangshu@outlook.com)
     * @date     2024-01-09
     */
    std::string DetectionAsync(const cv::Mat &img,
                               bool is_normal)
    {
        std::vector<cv::Mat> imgs{img};
        return DetectionAsync(imgs, is_normal);
    }

    /**
     * @brief    清理缓存
     * @author   CXS (chenxiangshu@outlook.com)
     * @date     2024-01-09
     */
    virtual void CleanCache() {}

    /**
     * @brief    正在运行
     * @return   true
     * @return   false
     * @author   CXS (chenxiangshu@outlook.com)
     * @date     2024-01-09
     */
    bool IsRun() const
    {
        return this->isRun;
    }

    /**
     * @brief    创建目标检测
     * @param    weight         权重文件 输入：[1,3,height,width] 输出：[批量,数量,结果]
     * @param    err            错误信息
     * @note     结果：[center x,center y,width,height,盒子概率,类别1概率,...,类别n概率]
     * @return   ITargetDetection*
     * @author   CXS (chenxiangshu@outlook.com)
     * @date     2024-01-09
     */
    static TargetDetection *Make(const char *weight, size_t threads, std::string &err);

    /**
     * @brief    绘制盒子
     * @param    img            图片
     * @param    result         结果
     * @param    color          颜色
     * @author   CXS (chenxiangshu@outlook.com)
     * @date     2024-01-09
     */
    static void DrawBox(cv::Mat &img, const std::vector<Result> &result, const cv::Scalar &color);
};

#endif // __TargetDetection_HPP__
