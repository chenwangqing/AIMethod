
#include "Tools.hpp"

namespace Tools {
    strings SplitString(const std::string &s, const std::string &c)
    {
        strings                v;
        std::string::size_type pos1, pos2;
        pos2 = s.find(c);
        pos1 = 0;
        while (std::string::npos != pos2) {
            v.push_back(s.substr(pos1, pos2 - pos1));
            pos1 = pos2 + c.size();
            pos2 = s.find(c, pos1);
        }
        if (pos1 != s.length())
            v.push_back(s.substr(pos1));
        return v;
    }

    std::string JoinString(const strings &s, const std::string &c)
    {
        if (s.size() == 0)
            return std::string();
        // 计算长度
        size_t mlen = 0;
        for (auto it = s.begin(); it != s.end(); it++) {
            if (it != s.begin())
                mlen += c.length();
            mlen += it->length();
        }
        std::string str(mlen, '0');
        char       *p = (char *)str.c_str();
        for (auto it = s.begin(); it != s.end(); it++) {
            if (!c.empty() && it != s.begin()) {
                memcpy(p, c.c_str(), c.length());
                p += c.length();
            }
            memcpy(p, it->c_str(), it->length());
            p += it->length();
        }
        return str;
    }

    /**
     * @brief    执行命令
     * @param    cmd
     * @return   string
     * @author   CXS (chenxiangshu@outlook.com)
     * @date     2024-01-19
     */
    static string RunCmd(const char *cmd)
    {
        auto fd = popen(cmd, "r");
        if (fd < 0) return "";
        char        tmp[1024];
        std::string str;
        while (!feof(fd)) {
            auto rs = fread(tmp, 1, sizeof(tmp) - 1, fd);
            if (rs == 0) break;
            tmp[rs] = '\0';
            str += tmp;
        }
        pclose(fd);
        return str;
    }

    strings GetFiles(const std::string &dir, const std::string &suffix)
    {
        return SplitString(RunCmd(("cd \"" + dir + "\" && ls *." + suffix).c_str()), "\n");
    }
}   // namespace Tools