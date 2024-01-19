/**
 * @file     Algorithm.hpp
 * @brief    算法
 * @author   CXS (chenxiangshu@outlook.com)
 * @version  1.0
 * @date     2024-01-19
 *
 * @copyright Copyright (c) 2024  Four-Faith
 *
 * @par 修改日志:
 * <table>
 * <tr><th>日期       <th>版本    <th>作者    <th>说明
 * <tr><td>2024-01-19 <td>1.0     <td>CXS     <td>创建
 * </table>
 */
#if !defined(___ALGORITHM_HPP__)
#define ___ALGORITHM_HPP__
#include "Define.h"

namespace AIMethod {
    /**
     * @brief    算法
     * @tparam T
     * @author   CXS (chenxiangshu@outlook.com)
     * @date     2024-01-19
     */
    template<typename T>
    class AL {
    private:
    public:
        static inline void Swap(T &a, T &b)
        {
            T tmp = a;
            a     = b;
            b     = tmp;
        }

        static inline T Sum(const T *x, size_t s)
        {
            T sum = 0;
            for (size_t i = 0; i < s; i++)
                sum += x[i];
            return sum;
        }

        static inline T Mean(const T *x, size_t s)
        {
            return Sum(x, s) / s;
        }

        static inline T Max(const T *x, size_t s)
        {
            T max = x[0];
            for (size_t i = 1; i < s; i++)
                if (max < x[i]) max = x[i];
            return max;
        }

        static inline T Min(const T *x, size_t s)
        {
            T max = x[0];
            for (size_t i = 1; i < s; i++)
                if (max > x[i]) max = x[i];
            return max;
        }

        static inline int MaxIdx(const T *x, size_t s)
        {
            int idx = 0;
            for (size_t i = 1; i < s; i++)
                if (x[i] > x[idx]) idx = i;
            return idx;
        }

        static inline int MinIdx(const T *x, size_t s)
        {
            int idx = 0;
            for (size_t i = 1; i < s; i++)
                if (x[i] < x[idx]) idx = i;
            return idx;
        }

        /**
         * @brief    Sigmoid $\frac{1}{1+e^{-x}}$
         * @param    v              值
         * @param    result         结果
         * @param    size           长度
         * @author   CXS (chenxiangshu@outlook.com)
         * @date     2024-01-19
         */
        static inline void Sigmoid(const T *v, T *result, size_t size)
        {
            for (size_t i = 0; i < size; i++)
                result[i] = 1 / (1 + exp(-v[i]));
            return;
        }

        /**
         * @brief    快速中值
         * @param    data           数据
         * @param    knLength       数据长度
         * @return   T              返回中值
         * @author   CXS (chenxiangshu@outlook.com)
         * @date     2024-01-19
         */
        static T QuickMedian(const T *data, size_t knLength)
        {
            std::vector<T> _data(data, data + knLength);
            T             *pnData = _data.data();
            /*
            以下是一种快速提取中位数的算法，算法描述如下：
            1：取队首，队尾和中间的三个数，通过交换数据，使得队尾最大，中间的数据最小，队首的数据为哨兵。
            2：交换中间和第2个数据，通过变换数据，使存在一个位置A，在该位置前的数据都小于哨兵，在该位置后的数据都大于或等于哨兵。
            3：如果A的位置在队中之后，则更新队列为1~A，否则为A~end。并继续调用这个过程。
            */
            size_t nLow    = 0;
            size_t nHigh   = 0;
            size_t nMiddle = 0;
            size_t nMedian = 0;
            size_t nLTmp   = 0;
            size_t nHTmp   = 0;
            nMedian        = (knLength - 1) >> 1;
            nHigh          = knLength - 1;
            while (1) {
                if (nHigh == nLow)
                    return pnData[nHigh];
                if (nHigh == nLow + 1)
                    return pnData[nHigh] > pnData[nLow] ? pnData[nLow] : pnData[nHigh];
                nMiddle = (nHigh + nLow) >> 1;
                if (pnData[nLow] > pnData[nHigh])
                    Swap(pnData[nHigh], pnData[nLow]);
                if (pnData[nMiddle] > pnData[nHigh])
                    Swap(pnData[nMiddle], pnData[nHigh]);
                if (pnData[nMiddle] > pnData[nLow])
                    Swap(pnData[nMiddle], pnData[nLow]);
                Swap(pnData[nMiddle], pnData[nLow + 1]);
                nLTmp = nLow + 2;
                nHTmp = nHigh - 1;
                while (1) {
                    while (pnData[nLTmp] <= pnData[nLow])
                        nLTmp++;
                    while (pnData[nHTmp] >= pnData[nLow])
                        nHTmp--;
                    if (nLTmp > nHTmp) {
                        Swap(pnData[nHTmp], pnData[nLow]);
                        if (nHTmp > nMedian)
                            nHigh = nHTmp - 1;
                        else
                            nLow = nLTmp - 1;
                        break;
                    } else {
                        Swap(pnData[nLTmp], pnData[nHTmp]);
                        nLTmp++;
                        nHTmp--;
                    }
                }
            }
            // return
        }
    };

}   // namespace AIMethod
#endif   // ___ALGORITHM_HPP__
