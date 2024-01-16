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
#include "Tensor.hpp"

// 设置推理引擎

#define INFER_ENGINE_OPENCV      0   //! 不支持动态输入
#define INFER_ENGINE_ONNXRUNTIME 1   // 支持动态输入

#ifndef CFG_INFER_ENGINE
#define CFG_INFER_ENGINE INFER_ENGINE_ONNXRUNTIME
#endif

// 启用GPU加速
#ifndef EN_GPU
#define EN_GPU 0
#endif

namespace AIMethod {

    /**
     * @brief    推理接口
     * @author   CXS (chenxiangshu@outlook.com)
     * @date     2024-01-10
     */
    class IRatiocinate {
    protected:
        std::atomic<int> is_runing;   // 正在运行

        class IStatus {
        public:
            IRatiocinate              *infer;
            std::vector<std::string>   input_names;
            std::vector<std::string>   output_names;
            std::vector<Tensor<float>> input_datas;
        };

    public:
        IRatiocinate() :
            is_runing(0)
        {
        }

        virtual ~IRatiocinate() = default;

        /**
         * @brief    正在运行
         * @return   true
         * @return   false
         * @author   CXS (chenxiangshu@outlook.com)
         * @date     2024-01-12
         */
        virtual bool IsRun() = 0;

        /**
         * @brief    推理结果
         * @author   CXS (chenxiangshu@outlook.com)
         * @date     2024-01-12
         */
        typedef struct
        {
            std::vector<int> shape;
            const float     *data;   // 仅在ExecCallback_t有效
        } Result;

        /**
         * @brief    执行回调
         * @param    infer         推理接口
         * @param    input_names   输入名称
         * @param    input_datas   输入数据
         * @param    output_names  输出名称
         * @param    output_names  输出名称
         * @param    output_datas  输出数据
         * @param    context       用户上下文
         * @param    err           错误信息
         * @author   CXS (chenxiangshu@outlook.com)
         * @date     2024-01-10
         */
        typedef void (*ExecCallback_t)(IRatiocinate                            *infer,
                                       const std::vector<std::string>          &input_names,
                                       const std::vector<Tensor<float>>        &input_datas,
                                       const std::vector<std::string>          &output_names,
                                       const std::vector<IRatiocinate::Result> &output_datas,
                                       void                                    *context,
                                       const std::string                       &err);

        /**
         * @brief    参数
         * @author   CXS (chenxiangshu@outlook.com)
         * @date     2024-01-10
         */
        typedef struct
        {
            const char *model;   // 模型文件
#if CFG_INFER_ENGINE == INFER_ENGINE_ONNXRUNTIME
            int threads;   // 线程数量
#endif
        } Parameters;

        ExecCallback_t callback         = nullptr;   // 执行回调(不关成功与否)
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
         * @brief    异步执行
         * @param    input_name     输入名称
         * @param    output_name    输出名称
         * @param    input_data     输入参数
         * @return   std::string
         * @author   CXS (chenxiangshu@outlook.com)
         * @date     2024-01-12
         */
        virtual std::string ExecAsync(const std::vector<std::string>   &input_names,
                                      const std::vector<std::string>   &output_names,
                                      const std::vector<Tensor<float>> &input_datas) = 0;
    };

    /**
     * @brief    创建推理
     * @return   IRatiocinate*
     * @author   CXS (chenxiangshu@outlook.com)
     * @date     2024-01-10
     */
    extern IRatiocinate *Ratiocinate_Create();
}   // namespace AIMethod
#endif   // __Ratiocinate_HPP__
