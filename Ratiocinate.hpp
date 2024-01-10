/**
 * @file     Ratiocinate.hpp
 * @brief    推理接口
 * @author   CXS (chenxiangshu@outlook.com)
 * @version  1.0
 * @date     2024-01-10
 *
 * @copyright Copyright (c) 2024  Four-Faith
 *
 * @par 修改日志:
 * <table>
 * <tr><th>日期       <th>版本    <th>作者    <th>说明
 * <tr><td>2024-01-10 <td>1.0     <td>CXS     <td>创建
 * </table>
 */
#if !defined(__Ratiocinate_HPP__)
#define __Ratiocinate_HPP__
#include "Tools.CV.hpp"

#ifndef EN_ONNXRUNTIME
#define EN_ONNXRUNTIME 1
#endif

#ifndef EN_GPU
#define EN_GPU 0
#endif

/**
 * @brief    推理接口
 * @author   CXS (chenxiangshu@outlook.com)
 * @date     2024-01-10
 */
class IRatiocinate {
protected:
    std::atomic<int> is_runing;   // 正在运行

public:
    IRatiocinate() :
        is_runing(0)
    {
    }

    virtual ~IRatiocinate() = default;

    virtual bool IsRun() = 0;

    /**
     * @brief    结果
     * @author   CXS (chenxiangshu@outlook.com)
     * @date     2024-01-10
     */
    class Result {
    public:
        std::vector<int64_t> shape;   // 形状
        float               *data;    // 结果数据
    };

    /**
     * @brief    输入
     * @author   CXS (chenxiangshu@outlook.com)
     * @date     2024-01-10
     */
    class Input {
    public:
        std::vector<cv::Mat>          imgs;   // 图像数据
        std::vector<Tools::Letterbox> lets;   // 图像修正
    };

    /**
     * @brief    执行回调
     * @author   CXS (chenxiangshu@outlook.com)
     * @date     2024-01-10
     */
    typedef void (*ExecCallback_t)(IRatiocinate                                *infer,
                                   std::map<std::string, IRatiocinate::Input>  &inputs,
                                   std::map<std::string, IRatiocinate::Result> &results,
                                   void                                        *context,
                                   const std::string                           &err);

    /**
     * @brief    参数
     * @author   CXS (chenxiangshu@outlook.com)
     * @date     2024-01-10
     */
    typedef struct
    {
        const char *model;       // 模型文件
        int         threads;     // 线程数量
        bool        is_normal;   // 输入归一化
    } Parameters;

    /**
     * @brief    IO信息
     * @author   CXS (chenxiangshu@outlook.com)
     * @date     2024-01-10
     */
    typedef struct
    {
        std::string          name;    // 名称
        std::vector<int64_t> shape;   // 形状
    } IOInfo;

    ExecCallback_t callback         = nullptr;   // 执行回调
    void          *callback_context = nullptr;   // 用户上下文

    /**
     * @brief    加载模型
     * @param    params        模型参数
     * @return   std::string
     * @author   CXS (chenxiangshu@outlook.com)
     * @date     2024-01-10
     */
    virtual std::string LoadModel(const Parameters &params) = 0;

    /**
     * @brief    获取IO信息
     * @return   const IOInfo&
     * @author   CXS (chenxiangshu@outlook.com)
     * @date     2024-01-10
     */
    virtual const std::vector<IOInfo> &GetIOInfo(bool isOutput) const = 0;

    /**
     * @brief    异步执行(图像处理)
     * @param    inputs         输入图片 BGR 格式
     * @param    size           推理时的大小（动态输入需要设置）
     * @return   std::string
     * @author   CXS (chenxiangshu@outlook.com)
     * @date     2024-01-10
     */
    virtual std::string ExecAsync(const std::map<std::string, std::vector<cv::Mat>> &inputs,
                                  cv::Size2i                                         size = cv::Size2i()) = 0;
};

/**
 * @brief    创建推理
 * @return   IRatiocinate*
 * @author   CXS (chenxiangshu@outlook.com)
 * @date     2024-01-10
 */
extern IRatiocinate *Ratiocinate_Create();

#endif   // __Ratiocinate_HPP__
