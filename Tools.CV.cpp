
#include "Tools.CV.hpp"

namespace Tools
{
    // --------------------------------------------------------------------------------
    //                                Letterbox
    // --------------------------------------------------------------------------------

    cv::Mat Letterbox::Make(const cv::Mat &src, int h, int w, Letterbox &let)
    {
        int in_w = src.cols; // width
        int in_h = src.rows; // height
        int tar_w = w;
        int tar_h = h;
        float r = std::min(float(tar_h) / in_h, float(tar_w) / in_w);
        int inside_w = round(in_w * r);
        int inside_h = round(in_h * r);
        int pad_w = tar_w - inside_w;
        int pad_h = tar_h - inside_h;

        cv::Mat resize_img;

        cv::resize(src, resize_img, cv::Size(inside_w, inside_h));

        pad_w = pad_w / 2;
        pad_h = pad_h / 2;

        let._fill_height = pad_h;
        let._fill_width = pad_w;
        let._r = r;

        int top = int(round(pad_h - 0.1));
        int bottom = int(round(pad_h + 0.1));
        int left = int(round(pad_w - 0.1));
        int right = int(round(pad_w + 0.1));
        cv::copyMakeBorder(resize_img, resize_img, top, bottom, left, right, 0, cv::Scalar(114, 114, 114));

        return resize_img;
    }

    cv::Rect Letterbox::Restore(const cv::Rect &box) const
    {
        cv::Rect scaled_box;
        scaled_box.x = (box.x - this->_fill_width) / this->_r;
        scaled_box.y = (box.y - this->_fill_height) / this->_r;
        scaled_box.width = box.width / this->_r;
        scaled_box.height = box.height / this->_r;
        return scaled_box;
    }
}
