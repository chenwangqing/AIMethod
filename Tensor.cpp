
#include "Tensor.hpp"

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

Tensor<float> operator+(const Tensor<float> &a, float b)
{
    return Add(a, b);
}

Tensor<float> operator+(float &a, const Tensor<float> b)
{
    return Add(b, a);
}

Tensor<float> operator+(Tensor<float> &&a, float b)
{
    Add(a, b);
    return std::move(a);
}

Tensor<float> operator+(float &a, Tensor<float> &&b)
{
    Add(b, a);
    return std::move(b);
}

Tensor<float> &operator+=(Tensor<float> &a, float b)
{
    Add(a, b);
    return a;
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

Tensor<float> operator*(const Tensor<float> &a, float b)
{
    return Mul(a, b);
}

Tensor<float> operator*(Tensor<float> &&a, float b)
{
    Mul(a, b);
    return std::move(a);
}

Tensor<float> &operator*=(Tensor<float> &a, float b)
{
    Mul(a, b);
    return a;
}
