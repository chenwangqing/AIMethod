/**
 * @file     Tools.hpp
 * @brief    工具
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
#if !defined(__TOOLS_HPP_)
#define __TOOLS_HPP_
#include "Tensor.hpp"

namespace Tools {
    namespace _Internal_Format {
        class ArgBase {
        public:
            ArgBase() {}
            virtual ~ArgBase() {}
            virtual void Format(std::ostringstream &ss, const std::string &fmt) = 0;
        };

        template<class T>
        class Arg : public ArgBase {
        public:
            Arg(T arg) :
                m_arg(arg) {}
            virtual ~Arg() {}
            virtual void Format(std::ostringstream &ss, const std::string &fmt)
            {
                ss << m_arg;
            }

        private:
            T m_arg;
        };

        class ArgArray : public std::vector<ArgBase *> {
        public:
            ArgArray() {}
            ~ArgArray()
            {
                std::for_each(begin(), end(), [](ArgBase *p) { delete p; });
            }
        };

        static void FormatItem(std::ostringstream &ss, const std::string &item, const ArgArray &args)
        {
            size_t      index     = 0;
            int         alignment = 0;
            std::string fmt;

            char *endptr = nullptr;
            index        = strtol(&item[0], &endptr, 10);
            if (index < 0 || index >= args.size()) {
                return;
            }

            if (*endptr == ',') {
                alignment = strtol(endptr + 1, &endptr, 10);
                if (alignment > 0) {
                    ss << std::right << std::setw(alignment);
                } else if (alignment < 0) {
                    ss << std::left << std::setw(-alignment);
                }
            }

            if (*endptr == ':') {
                fmt = endptr + 1;
            }

            args[index]->Format(ss, fmt);

            return;
        }

        template<class T>
        static void Transfer(ArgArray &argArray, T t)
        {
            argArray.push_back(new Arg<T>(t));
        }

        template<class T, typename... Args>
        static void Transfer(ArgArray &argArray, T t, Args &&...args)
        {
            Transfer(argArray, t);
            Transfer(argArray, args...);
        }
    }   // namespace _Internal_Format

    /**
     * @brief    字符串格式化
     * @tparam Args
     * @param    format         格式化说明
     * @param    args           参数
     * @note     Format("a = {0} b= {1}",12,23)
     * @note     Format("str = {0,10}","12334545")
     * @return   std::string
     * @author   CXS (chenxiangshu@outlook.com)
     * @date     2024-01-10
     */
    template<typename... Args>
    std::string Format(const std::string &format, Args &&...args)
    {
        if (sizeof...(args) == 0)
            return format;

        _Internal_Format::ArgArray argArray;
        _Internal_Format::Transfer(argArray, args...);
        size_t             start = 0;
        size_t             pos   = 0;
        std::ostringstream ss;
        while (true) {
            pos = format.find('{', start);
            if (pos == std::string::npos) {
                ss << format.substr(start);
                break;
            }

            ss << format.substr(start, pos - start);
            if (format[pos + 1] == '{') {
                ss << '{';
                start = pos + 2;
                continue;
            }

            start = pos + 1;
            pos   = format.find('}', start);
            if (pos == std::string::npos) {
                ss << format.substr(start - 1);
                break;
            }

            _Internal_Format::FormatItem(ss, format.substr(start, pos - start), argArray);
            start = pos + 1;
        }
        return ss.str();
    }
}   // namespace Tools
#endif   // __TOOLS_HPP_
