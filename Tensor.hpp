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
    };

    Node            *node = nullptr;
    std::vector<int> shape;   // 内置切片
    std::vector<int> shape_index;
    T               *data = nullptr;

    /**
     * @brief    分离
     * @author   CXS (chenxiangshu@outlook.com)
     * @date     2024-01-11
     */
    void Separation()
    {
        if (this->node == nullptr)
            return;
        if (this->node->ref_count.fetch_sub(1) == 1)
            delete this->node;
        this->node = nullptr;
        this->data = nullptr;
        this->shape.clear();
        this->shape_index.clear();
        return;
    }

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
        this->node = ps.node;
        if (this->node != nullptr)
            this->node->ref_count.fetch_add(1);
        this->shape       = ps.shape;
        this->data        = ps.data;
        this->shape_index = ps.shape_index;
        return;
    }

    Tensor(Tensor &&ps)
    {
        this->node        = ps.node;
        this->data        = ps.data;
        this->shape       = std::move(ps.shape);
        this->shape_index = std::move(ps.shape_index);
        ps.node           = nullptr;
        ps.data           = nullptr;
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
        node        = new Node();
        this->shape = shape;
        size_t s    = this->shape[0];
        for (size_t i = 1; i < this->shape.size(); i++)
            s *= this->shape[i];
        node->data = std::vector<T>(data, data + s);
        this->data = node->data.data();
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

    inline const std::vector<int> &GetShape() const
    {
        return this->shape;
    }

    template<typename R>
    std::vector<R> GetShape() const
    {
        std::vector<R> shape(this->shape.size());
        for (size_t i = 0; i < shape.size(); i++)
            shape[i] = this->shape[i];
        return shape;
    }

    inline size_t Size() const
    {
        return this->node == nullptr ? 0 : this->shape[0] * this->shape_index[0];
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
        this->node        = ps.node;
        this->shape       = ps.shape;
        this->data        = ps.data;
        this->shape_index = ps.shape_index;
        if (this->node != nullptr)
            this->node->ref_count.fetch_add(1);
        return *this;
    }

    Tensor &operator=(Tensor &&ps)
    {
        Separation();
        this->node        = ps.node;
        this->shape       = std::move(ps.shape);
        this->data        = ps.data;
        this->shape_index = std::move(ps.shape_index);
        ps.node           = nullptr;
        ps.data           = nullptr;
        return *this;
    }

    bool operator==(const Tensor &ps) const
    {
        return ps.Size() == this->Size() &&
               ps.shape() == this->shape &&
               memcmp(this->data, ps.data, this->Size() * sizeof(T)) == 0;
    }

    inline bool operator!=(const Tensor &ps) const
    {
        return !operator==(ps);
    }

    /**
     * @brief    克隆
     * @return   Tensor
     * @author   CXS (chenxiangshu@outlook.com)
     * @date     2024-01-11
     */
    inline Tensor Clone() const
    {
        return this->node == nullptr ? Tensor() : Tensor(this->node->shape, this->node->data);
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
     * @brief    重新设置形状
     * @param    shape
     * @return   true
     * @return   false
     * @author   CXS (chenxiangshu@outlook.com)
     * @date     2024-01-11
     */
    bool ReShape(std::vector<int> &shape)
    {
        if (shape.size() == 0) {
            this->Separation();
            return true;
        }
        auto s   = this->Size();
        auto r   = 1;
        int  idx = -1;
        for (size_t i = 0; i < shape.size(); i++) {
            if (shape[i] == 0) {
                r = 0;
                break;
            }
            if (shape[i] < 0) {
                if (idx >= 0)
                    return false;
                idx = i;
            } else
                r *= shape[i];
        }
        if (r == 0) {
            this->Separation();
            return true;
        }
        if (r > s || (idx >= 0 && (s % r))) return false;
        this->shape = shape;
        if (idx >= 0) this->shape[idx] = s / r;
        return true;
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
    Tensor _Slice(size_t idx, const std::vector<int> &shape) const
    {
        auto s = this->Size();
        if (idx >= s || s == 0)
            return Tensor();
        size_t r = 1;
        for (auto &v : shape) {
            if (v <= 0)
                return Tensor();
            r *= v;
        }
        if (r + idx > s) return Tensor();
        Tensor ret;
        ret.node = this->node;
        ret.node->ref_count.fetch_add(1);
        ret.shape = shape;
        ret.data  = this->data + idx;
        ret.MakeIndex();
        return ret;
    }

public:
    /**
     * @brief    切片
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
     * @brief    切片
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
};

extern Tensor<float>  operator+(const Tensor<float> &a, float b);
extern Tensor<float>  operator+(float &a, const Tensor<float> b);
extern Tensor<float>  operator+(Tensor<float> &&a, float b);
extern Tensor<float>  operator+(float &a, Tensor<float> &&b);
extern Tensor<float> &operator+=(Tensor<float> &a, float b);
extern Tensor<float>  operator*(const Tensor<float> &a, float b);
extern Tensor<float>  operator*(Tensor<float> &&a, float b);
extern Tensor<float> &operator*=(Tensor<float> &a, float b);

#endif   // __TENSOR_HPP__
