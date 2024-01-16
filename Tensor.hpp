/**
 * @file     Tensor.hpp
 * @brief    张量
 * @author   CXS (chenxiangshu@outlook.com)
 * @version  1.0
 * @date     2024-01-11
 *
 * @copyright Copyright (c) 2024  Four-Faith
 *
 * @par 修改日志:
 * <table>
 * <tr><th>日期       <th>版本    <th>作者    <th>说明
 * <tr><td>2024-01-11 <td>1.0     <td>CXS     <td>创建
 * </table>
 */
#if !defined(__TENSOR_HPP__)
#define __TENSOR_HPP__
#include <stdarg.h>
#include <stdint.h>
#include <atomic>
#include <map>
#include <string>
#include <vector>
#include <list>
#include <stdlib.h>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <mutex>
#include <string.h>
#include "Define.h"

namespace AIMethod {
    /**
     * @brief    张量
     * @tparam T
     * @author   CXS (chenxiangshu@outlook.com)
     * @date     2024-01-11
     */
    template<typename T>
    class Tensor {
    private:
        class Node {
        public:
            std::vector<T>   data;
            std::atomic<int> ref_count;

            Node() :
                ref_count(1) {}

            Node(const T *data, size_t size) :
                ref_count(1), data(data, data + size) {}
        };

        Node            *node = nullptr;
        std::vector<int> shape;   // 内置切片
        std::vector<int> shape_index;
        T               *data = nullptr;

        size_t TotalLength     = 0;
        size_t Dimensions_Size = 0;
        int   *Dimensions      = nullptr;

        /**
         * @brief    分离
         * @author   CXS (chenxiangshu@outlook.com)
         * @date     2024-01-11
         */
        void Separation()
        {
            if (this->node != nullptr && this->node->ref_count.fetch_sub(1) == 1)
                delete this->node;
            this->node = nullptr;
            this->data = nullptr;
            this->shape.clear();
            this->shape_index.clear();
            this->Dimensions_Size = 0;
            this->TotalLength     = 0;
            return;
        }

        /**
         * @brief    制作索引
         * @author   CXS (chenxiangshu@outlook.com)
         * @date     2024-01-12
         */
        void MakeIndex()
        {
            if (shape.size() == 0) {
                shape_index.clear();
                return;
            }
            shape_index.resize(shape.size());
            auto n             = shape.size();
            shape_index[n - 1] = 1;
            for (int i = n - 2; i >= 0; i--)
                shape_index[i] = shape_index[i + 1] * shape[i + 1];
            this->TotalLength     = shape[0] * shape_index[0];
            this->Dimensions_Size = shape.size();
            this->Dimensions      = this->shape.data();
            return;
        }

        Tensor(const std::vector<int> &shape, const T *data, Node *node) :
            shape(shape), data((T *)data), node(node)
        {
            MakeIndex();
        }

        void Copy(const Tensor &ps)
        {
            auto s      = ps.Size();
            this->node  = new Node(ps.data, s);
            this->shape = ps.shape;
            this->data  = this->node->data.data();
            this->MakeIndex();
            return;
        }

    public:
        Tensor()
        {
        }

        virtual ~Tensor()
        {
            Separation();
            return;
        }

        Tensor(const Tensor &ps)
        {
            if (ps.node == nullptr)
                return;
            if (ps.node == nullptr && ps.data != nullptr) {
                Copy(ps);
            } else {
                this->node = ps.node;
                if (this->node != nullptr)
                    this->node->ref_count.fetch_add(1);
                this->shape = ps.shape;
                this->data  = ps.data;
                this->MakeIndex();
            }
            return;
        }

        Tensor(Tensor &&ps)
        {
            this->node            = ps.node;
            this->data            = ps.data;
            this->shape           = std::move(ps.shape);
            this->shape_index     = std::move(ps.shape_index);
            this->TotalLength     = ps.TotalLength;
            this->Dimensions      = ps.Dimensions;
            this->Dimensions_Size = ps.Dimensions_Size;
            ps.node               = nullptr;
            ps.data               = nullptr;
            ps.TotalLength        = 0;
            ps.Dimensions         = nullptr;
            ps.Dimensions_Size    = 0;
            return;
        }

        Tensor(const std::vector<int> &shape)
        {
            if (shape.size() == 0)
                return;
            node        = new Node();
            this->shape = shape;
            size_t s    = this->shape[0];
            for (size_t i = 1; i < this->shape.size(); i++)
                s *= this->shape[i];
            node->data.resize(s);
            this->data = node->data.data();
            MakeIndex();
            return;
        }

        Tensor(const std::vector<int> &shape, const T *data)
        {
            if (shape.size() == 0)
                return;
            this->shape = shape;
            size_t s    = this->shape[0];
            for (size_t i = 1; i < this->shape.size(); i++)
                s *= this->shape[i];
            this->node = new Node(data, s);
            this->data = this->node->data.data();
            MakeIndex();
            return;
        }

        Tensor(const std::vector<int> &shape, std::vector<T> &&data)
        {
            if (shape.size() == 0)
                return;
            node        = new Node();
            this->shape = shape;
            node->data  = std::move(data);
            this->data  = node->data.data();
            MakeIndex();
            return;
        }

        Tensor(std::vector<int> &&shape, std::vector<T> &&data)
        {
            if (shape.size() == 0)
                return;
            node        = new Node();
            this->shape = std::move(shape);
            node->data  = std::move(data);
            this->data  = node->data.data();
            MakeIndex();
            return;
        }

        /**
         * @brief    创建常量（不会拷贝数据）
         * @param    shape          形状
         * @param    data           数据指针
         * @return   const Tensor
         * @author   CXS (chenxiangshu@outlook.com)
         * @date     2024-01-12
         */
        static const Tensor MakeConst(const std::vector<int> &shape, const T *data)
        {
            const Tensor tmp(shape, data, nullptr);
            return tmp;
        }

        /**
         * @brief    获取形状
         * @return   const std::vector<int>&
         * @author   CXS (chenxiangshu@outlook.com)
         * @date     2024-01-12
         */
        inline const std::vector<int> &GetShape() const
        {
            return this->shape;
        }

        /**
         * @brief    获取形状
         * @return   const std::vector<int>&
         * @author   CXS (chenxiangshu@outlook.com)
         * @date     2024-01-12
         */
        template<typename R>
        std::vector<R> GetShape() const
        {
            std::vector<R> shape(this->shape.size());
            for (size_t i = 0; i < shape.size(); i++)
                shape[i] = this->shape[i];
            return shape;
        }

        /**
         * @brief    获取形状索引
         * @return   const std::vector<int>&
         * @author   CXS (chenxiangshu@outlook.com)
         * @date     2024-01-12
         */
        inline const std::vector<int> &GetShapeIndex() const
        {
            return this->shape_index;
        }

        /**
         * @brief    元素数量
         * @return   size_t
         * @author   CXS (chenxiangshu@outlook.com)
         * @date     2024-01-16
         */
        inline size_t Size() const
        {
            return this->TotalLength;
        }

        /**
         * @brief    引用（没有进行实际拷贝）
         * @param    ps
         * @return   Tensor&
         * @author   CXS (chenxiangshu@outlook.com)
         * @date     2024-01-11
         */
        Tensor &operator=(const Tensor &ps)
        {
            Separation();
            if (ps.node == nullptr && ps.data != nullptr) {
                Copy(ps);
            } else {
                this->node  = ps.node;
                this->shape = ps.shape;
                this->data  = ps.data;
                if (this->node != nullptr)
                    this->node->ref_count.fetch_add(1);
                this->MakeIndex();
            }
            return *this;
        }

        Tensor &operator=(Tensor &&ps)
        {
            Separation();
            this->node            = ps.node;
            this->shape           = std::move(ps.shape);
            this->data            = ps.data;
            this->shape_index     = std::move(ps.shape_index);
            this->TotalLength     = ps.TotalLength;
            this->Dimensions      = ps.Dimensions;
            this->Dimensions_Size = ps.Dimensions_Size;
            ps.node               = nullptr;
            ps.data               = nullptr;
            ps.Dimensions         = nullptr;
            ps.TotalLength        = 0;
            return *this;
        }

        /**
         * @brief    算法内容一致
         * @param    ps
         * @return   true
         * @return   false
         * @author   CXS (chenxiangshu@outlook.com)
         * @date     2024-01-16
         */
        bool Equal(const Tensor &ps) const
        {
            return ps.Size() == this->Size() &&
                   ps.shape == this->shape &&
                   memcmp(this->data, ps.data, this->Size() * sizeof(T)) == 0;
        }

        /**
         * @brief    克隆
         * @return   Tensor
         * @author   CXS (chenxiangshu@outlook.com)
         * @date     2024-01-11
         */
        inline Tensor Clone() const
        {
            return this->node == nullptr ? Tensor() : Tensor(this->shape, this->data);
        }

        /**
         * @brief    克隆
         * @return   Tensor
         * @author   CXS (chenxiangshu@outlook.com)
         * @date     2024-01-11
         */
        template<typename R>
        Tensor<R> Clone() const
        {
            Tensor<R> result(this->shape);
            auto      s = this->Size();
            auto     *p = this->Value();
            auto     *r = result.Value();
            for (size_t i = 0; i < s; i++)
                r[i] = (R)p[i];
            return result;
        }

        inline T *Value()
        {
            return this->data;
        }

        inline const T *Value() const
        {
            return this->data;
        }

        /**
         * @brief    拷贝
         * @param    start          起始未知
         * @param    data           数据
         * @param    size           数据长度
         * @author   CXS (chenxiangshu@outlook.com)
         * @date     2024-01-16
         */
        void CopyTo(int start, const T *data, size_t size)
        {
            if (start < 0 || data == nullptr || size <= 0)
                return;
            if (size + start > this->Size())
                return;
            memcpy(this->data + start, data, size * sizeof(T));
            return;
        }

    private:
        int GetIdx(int low, va_list &vl) const
        {
            if (this->shape_index.size() == 0)
                return 0;
            low *= this->shape_index[0];
            for (size_t i = 1; i < this->shape_index.size(); i++) {
                int v = va_arg(vl, int);
                if (v <= 0)
                    continue;
                low += this->shape_index[i] * v;
            }
            return low;
        }

        int GetIdx(const int *dim) const
        {
            if (this->shape_index.size() == 0)
                return 0;
            int idx = 0;
            for (int i = 0; i < this->shape_index.size(); i++) {
                if (dim[i] <= 0)
                    continue;
                idx += this->shape_index[i] * dim[i];
            }
            return idx;
        }

    public:
        /**
         * @brief    获取索引
         * @param    idx            维度索引 （高到低）
         * @param    ...            长度必须和维度一致 可以填 -1
         * @return   int
         * @author   CXS (chenxiangshu@outlook.com)
         * @date     2023-06-13
         */
        inline int GetIdx(int idx, ...) const
        {
            va_list vl;
            va_start(vl, idx);
            idx = GetIdx(idx, vl);
            va_end(vl);
            return idx;
        }

        inline int GetIdx(int d1, int d2) const
        {
            return d1 * this->shape_index[0] + d2;
        }

        inline int GetIdx(int d1, int d2, int d3) const
        {
            return d1 * this->shape_index[0] + d2 * this->shape_index[1] + d3;
        }

        inline int GetIdx(int d1, int d2, int d3, int d4) const
        {
            return d1 * this->shape_index[0] + d2 * this->shape_index[1] + d3 * this->shape_index[2] + d4;
        }

        /**
         * @brief    获取数据
         * @param    idx            维度索引 （高到低）
         * @param    ...            长度必须和维度一致
         * @return   T&
         * @author   CXS (chenxiangshu@outlook.com)
         * @date     2023-06-13
         */
        inline T &At(int d1, int d2)
        {
            return this->data[d1 * this->shape_index[0] + d2];
        }

        /**
         * @brief    获取数据
         * @param    idx            维度索引 （高到低）
         * @param    ...            长度必须和维度一致
         * @return   T&
         * @author   CXS (chenxiangshu@outlook.com)
         * @date     2023-06-13
         */
        inline const T &At(int d1, int d2) const
        {
            return this->data[d1 * this->shape_index[0] + d2];
        }

        /**
         * @brief    获取数据
         * @param    idx            维度索引 （高到低）
         * @param    ...            长度必须和维度一致
         * @return   T&
         * @author   CXS (chenxiangshu@outlook.com)
         * @date     2023-06-13
         */
        inline T &At(int d1, int d2, int d3)
        {
            return this->data[d1 * this->shape_index[0] + d2 * this->shape_index[1] + d3];
        }

        /**
         * @brief    获取数据
         * @param    idx            维度索引 （高到低）
         * @param    ...            长度必须和维度一致
         * @return   T&
         * @author   CXS (chenxiangshu@outlook.com)
         * @date     2023-06-13
         */
        inline const T &At(int d1, int d2, int d3) const
        {
            return this->data[d1 * this->shape_index[0] + d2 * this->shape_index[1] + d3];
        }

        /**
         * @brief    获取数据
         * @param    idx            维度索引 （高到低）
         * @param    ...            长度必须和维度一致
         * @return   T&
         * @author   CXS (chenxiangshu@outlook.com)
         * @date     2023-06-13
         */
        inline T &At(int d1, int d2, int d3, int d4)
        {
            return this->data[d1 * this->shape_index[0] + d2 * this->shape_index[1] + d3 * this->shape_index[2] + d4];
        }

        /**
         * @brief    获取数据
         * @param    idx            维度索引 （高到低）
         * @param    ...            长度必须和维度一致
         * @return   T&
         * @author   CXS (chenxiangshu@outlook.com)
         * @date     2023-06-13
         */
        inline const T &At(int d1, int d2, int d3, int d4) const
        {
            return this->data[d1 * this->shape_index[0] + d2 * this->shape_index[1] + d3 * this->shape_index[2] + d4];
        }

        /**
         * @brief    获取数据
         * @param    idx            维度索引 （高到低）
         * @param    ...            长度必须和维度一致
         * @return   T&
         * @author   CXS (chenxiangshu@outlook.com)
         * @date     2023-06-13
         */
        inline T &At(int idx, ...)
        {
            va_list vl;
            va_start(vl, idx);
            idx = GetIdx(idx, vl);
            va_end(vl);
            return this->data[idx];
        }

        /**
         * @brief    获取数据
         * @param    idx            维度索引 （高到低）
         * @param    ...            长度必须和维度一致 可以填 -1
         * @return   T&
         * @author   CXS (chenxiangshu@outlook.com)
         * @date     2023-06-13
         */
        inline const T &At(int idx, ...) const
        {
            va_list vl;
            va_start(vl, idx);
            idx = GetIdx(idx, vl);
            va_end(vl);
            return this->data[idx];
        }

    private:
        Tensor _Slice(size_t idx, const std::vector<int> &_shape) const
        {
            auto s = this->Size();
            if (idx >= s || s == 0)
                return Tensor();
            size_t r     = 1;
            int    r_idx = -1;
            auto   shape = _shape;
            for (size_t i = 0; i < shape.size(); i++) {
                auto v = shape[i];
                if (v <= 0) {
                    if (r_idx >= 0) return Tensor();
                    r_idx = i;
                    continue;
                }
                r *= v;
            }
            if (r + idx > s) return Tensor();
            if (r_idx >= 0) {
                s -= idx;
                if (s % r) return Tensor();
                shape[r_idx] = s / r;
                r            = s;
            }
            Tensor ret;
            if (this->node == nullptr && this->data != nullptr) {
                ret.node = new Node(this->data + idx, r);
                ret.data = ret.node->data.data();
            } else {
                ret.node = this->node;
                ret.node->ref_count.fetch_add(1);
                ret.data = this->data + idx;
            }
            ret.shape = shape;
            ret.MakeIndex();
            return ret;
        }

    public:
        /**
         * @brief    切片【引用】
         * @param    idx            起始索引
         * @param    shape          形状
         * @return   Tensor
         * @author   CXS (chenxiangshu@outlook.com)
         * @date     2024-01-11
         */
        inline Tensor Slice(size_t idx, const std::vector<int> &shape)
        {
            return _Slice(idx, shape);
        }

        /**
         * @brief    切片【引用】
         * @param    idx            起始索引
         * @param    shape          形状
         * @return   Tensor
         * @author   CXS (chenxiangshu@outlook.com)
         * @date     2024-01-11
         */
        inline const Tensor Slice(size_t idx, const std::vector<int> &shape) const
        {
            return _Slice(idx, shape);
        }

    private:
        void _Broadcast(T         *dst,
                        const int *dst_dim,
                        const int *dst_dim_index,
                        size_t     dst_dims,
                        const T   *src,
                        const int *src_dim,
                        const int *src_dim_index,
                        size_t     src_dims) const
        {
            if (dst_dims > src_dims) {
                auto n  = dst_dims - src_dims;
                int  s  = dst_dim[0] * dst_dim_index[0];
                int  bs = dst_dim[n] * dst_dim_index[n];
                s       = s / bs;
                for (int i = 0; i < s; i++)
                    _Broadcast(dst + i * bs,
                               dst_dim + n,
                               dst_dim_index + n,
                               dst_dims - n,
                               src,
                               src_dim,
                               src_dim_index,
                               src_dims);
            } else {
                int n = dst_dim[0] / src_dim[0];
                if (src_dims > 1 && memcmp(dst_dim + 1, src_dim + 1, (dst_dims - 1) * sizeof(int)) == 0) {
                    int bs = src_dims == 1 ? 1 : src_dim_index[0];
                    for (int i = 0; i < n; i++)
                        memcpy(dst + i * bs, src, bs * sizeof(T));
                } else if (src_dims > 2) {
                    n = dst_dim[0];
                    for (int i = 0; i < n; i++)
                        _Broadcast(dst + i * dst_dim_index[0],
                                   dst_dim + 1,
                                   dst_dim_index + 1,
                                   dst_dims - 1,
                                   src + (i % src_dim[0]) * src_dim_index[0],
                                   src_dim + 1,
                                   src_dim_index + 1,
                                   src_dims - 1);
                } else {
                    // 列复制
                    for (int r = 0; r < dst_dim[0]; r++) {
                        for (int c = 0; c < dst_dim[1]; c++)
                            dst[r * dst_dim[1] + c] = src[r * src_dim[1]];
                    }
                }
            }
            return;
        }

    public:
        /**
         * @brief    广播
         * @param    shape          形状
         * @return   Tensor
         * @author   CXS (chenxiangshu@outlook.com)
         * @date     2024-01-15
         */
        Tensor Broadcast(const std::vector<int> &shape) const
        {
            if (shape.size() == 0 || shape.size() < this->GetShape().size())
                return Tensor();
            /* 如果两个数组的后缘维度（trailing dimension，即从末尾开始算起的维度）的轴长度相符，
                或其中的一方的长度为1，则认为它们是广播兼容的。广播会在缺失和（或）长度为1的维度上进行。
            */
            auto &old_shape = this->GetShape();
            for (size_t i = 0; i < shape.size() && i < this->GetShape().size(); i++) {
                if (old_shape[old_shape.size() - i - 1] != 1 &&
                    shape[shape.size() - i - 1] != old_shape[old_shape.size() - i - 1])
                    RUN_ERR("Broadcast dimension error");
            }
            Tensor ret;
            ret.shape = shape;
            ret.MakeIndex();
            ret.node = new Node();
            ret.node->data.resize(ret.Size());
            ret.data = ret.node->data.data();
            // 一种是两个数组的维数不相等，但是它们的后缘维度的轴长相符（其实就是从后数的连续若干个维度数都相同）
            // 一种是有一方的长度为1（其实就是如果从后数有维度不同，但是维度大小为1时，广播机制同样可以发挥作用）。
            _Broadcast(ret.data,
                       ret.shape.data(),
                       ret.shape_index.data(),
                       shape.size(),
                       this->data,
                       this->shape.data(),
                       this->shape_index.data(),
                       this->shape.size());
            return ret;
        }

        /**
         * @brief    清理
         * @author   CXS (chenxiangshu@outlook.com)
         * @date     2024-01-11
         */
        void Clear()
        {
            this->Separation();
            return;
        }

        static Tensor Zero(const std::vector<int> &shape)
        {
            Tensor ret(shape);
            memset(ret.Value(), 0, ret.Size() * sizeof(T));
            return ret;
        }

        static Tensor Ones(const std::vector<int> &shape)
        {
            Tensor ret(shape);
            auto   s = ret.Size();
            for (size_t i = 0; i < s; i++)
                ret.data[i] = 1;
            return ret;
        }

        /**
         * @brief    创建范围张量[0,1,2,3...]
         * @param    shape
         * @return   Tensor
         * @author   CXS (chenxiangshu@outlook.com)
         * @date     2024-01-16
         */
        static Tensor Arange(const std::vector<int> &shape, T start = 0)
        {
            Tensor ret(shape);
            auto   s = ret.Size();
            for (size_t i = 0; i < s; i++)
                ret.data[i] = (T)i + start;
            return ret;
        }
    };

    class Operation {
    private:
        Operation() {}

    public:
        static const Operation &__get()
        {
            static Operation op;
            return op;
        }

        Tensor<float> Mul(const Tensor<float> &a, const float b) const;
        Tensor<float> Mul(Tensor<float> &&a, const float b) const;
        // ax+b
        Tensor<float> Mul(const float a, const Tensor<float> &x, const float b) const;
        // ax+b
        Tensor<float> Mul(const float a, Tensor<float> &&x, const float b) const;
        Tensor<float> Mul(const Tensor<float> &a, const Tensor<float> &b) const;

        Tensor<float> Add(const Tensor<float> &a, const Tensor<float> &b) const;

        Tensor<float> Sigmoid(const Tensor<float> &a) const;
        Tensor<float> Sigmoid(Tensor<float> &&a) const;
    };

    extern Operation op;

}   // namespace AIMethod
#endif   // __TENSOR_HPP__
