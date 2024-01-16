
#include "Tensor.hpp"
#include "Define.h"

namespace AIMethod {

    // --------------------------------------------------------------------------------
    //                                   ADD
    // --------------------------------------------------------------------------------

    template<typename T>
    static Tensor<T> Add(const Tensor<T> &a, const T b)
    {
        Tensor<T> ret(a.GetShape());
        auto      v  = a.Value();
        auto      rv = ret.Value();
        auto      s  = a.Size();
        for (size_t i = 0; i < s; i++, v++, rv++)
            *rv = *v + b;
        return ret;
    }

    template<typename T>
    static void Add(Tensor<T> &a, const T b)
    {
        auto v = a.Value();
        auto s = a.Size();
        for (size_t i = 0; i < s; i++, v++)
            *v += b;
        return;
    }

    template<typename T>
    static Tensor<T> Add(const Tensor<T> &a, const Tensor<T> &b)
    {
        Tensor<T> ret;
        const T  *av = nullptr;
        const T  *bv = nullptr;
        T        *rv = nullptr;
        size_t    s  = 0;
        if (a.GetShape() == b.GetShape()) {
            av  = a.Value();
            bv  = b.Value();
            ret = Tensor<T>(a.GetShape());
            rv  = ret.Value();
            s   = ret.Size();
        } else
            RUN_ERR("Shape mismatch");
        for (size_t i = 0; i < s; i++)
            rv[i] = av[i] + bv[i];
        return ret;
    }

    Tensor<float> Operation::Add(const Tensor<float> &a, const Tensor<float> &b) const
    {
        return Add(a, b);
    }

    // --------------------------------------------------------------------------------
    //                                   MUL
    // --------------------------------------------------------------------------------

    template<typename T>
    static Tensor<T> Mul(const Tensor<T> &a, const T b)
    {
        Tensor<T> ret(a.GetShape());
        auto      v  = a.Value();
        auto      rv = ret.Value();
        auto      s  = a.Size();
        for (size_t i = 0; i < s; i++, v++, rv++)
            *rv = *v * b;
        return ret;
    }

    template<typename T>
    static void Mul(Tensor<T> &a, const T b)
    {
        auto v = a.Value();
        auto s = a.Size();
        for (size_t i = 0; i < s; i++, v++)
            *v *= b;
        return;
    }

    Tensor<float> Operation::Mul(const Tensor<float> &a, const float b) const
    {
        return Mul(a, b);
    }

    Tensor<float> Operation::Mul(Tensor<float> &&a, const float b) const
    {
        Mul(a, b);
        return $(a);
    }

    template<typename T>
    static Tensor<T> Mul(const float a, const Tensor<T> &x, const T b)
    {
        Tensor<T> ret(x.GetShape());
        auto      v  = x.Value();
        auto      rv = ret.Value();
        auto      s  = x.Size();
        for (size_t i = 0; i < s; i++, v++, rv++)
            *rv = *v * a + b;
        return ret;
    }

    template<typename T>
    static void Mul(const float a, Tensor<T> &x, const T b)
    {
        auto v = x.Value();
        auto s = x.Size();
        for (size_t i = 0; i < s; i++, v++)
            *v = *v * a + b;
        return;
    }

    Tensor<float> Operation::Mul(const float a, const Tensor<float> &x, const float b) const
    {
        return Mul(a, x, b);
    }

    Tensor<float> Operation::Mul(const float a, Tensor<float> &&x, const float b) const
    {
        Mul(a, x, b);
        return $(x);
    }

    /**
     * @brief    alpha*x*y+bias
     * @tparam T
     * @param    alpha
     * @param    a
     * @param    a_rows
     * @param    a_cols
     * @param    b
     * @param    b_rows
     * @param    b_cols
     * @param    result
     * @param    bias
     * @return   true
     * @return   false
     * @author   CXS (chenxiangshu@outlook.com)
     * @date     2023-08-14
     */
    template<typename T>
    static void Mul2D(const T &alpha,
                      const T *a,
                      int      a_rows,
                      int      a_cols,
                      const T *b,
                      int      b_rows,
                      int      b_cols,
                      T       *result,
                      const T &bias)
    {
        if (a_cols != b_rows)
            RUN_ERR("Matrix multiplication dimension error");
        int rows = a_rows;
        int cols = b_cols;
        for (int r = 0; r < rows; r++) {
            auto rev = result + r * cols;
            for (int c = 0; c < cols; c++) {
                T    val = 0;
                auto apv = a + r * a_cols;
                for (int i = 0; i < a_cols; i++)
                    val += apv[i] * b[i * b_cols + c];
                rev[c] = alpha * val + bias;
            }
        }
        return;
    }

    Tensor<float> Operation::Mul(const Tensor<float> &a, const Tensor<float> &b) const
    {
        auto &a_shape = a.GetShape();
        auto &b_shape = b.GetShape();
        if (a_shape.size() != b_shape.size() ||
            a_shape.size() < 2)
            RUN_ERR("Shape mismatch");
        if (a_shape.size() > 2 && memcmp(a_shape.data(), b_shape.data(), (a_shape.size() - 2) * sizeof(int)) != 0)
            RUN_ERR("Shape mismatch");
        size_t s = 1;
        size_t n = a_shape.size() - 2;
        for (size_t i = 0; i < n; i++)
            s *= a_shape[i];
        int  a_rows   = a_shape[n];
        int  a_cols   = a_shape[n + 1];
        int  b_rows   = b_shape[n];
        int  b_cols   = b_shape[n + 1];
        int  a_blocks = a_rows * a_cols;
        int  b_blocks = b_rows * b_cols;
        int  r_blocks = a_rows * b_cols;
        auto dim      = a_shape;
        dim[n]        = a_rows;
        dim[n + 1]    = b_cols;
        Tensor<float> ret(dim);
        for (size_t i = 0; i < s; i++) {
            auto *av = a.Value() + i * a_blocks;
            auto *bv = b.Value() + i * b_blocks;
            auto *rv = ret.Value() + i * r_blocks;
            Mul2D(1.0f, av, a_rows, a_cols, bv, b_rows, b_cols, rv, 0.0f);
        }
        return ret;
    }

    Operation op = Operation::__get();
}   // namespace AIMethod
