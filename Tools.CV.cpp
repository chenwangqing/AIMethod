
#include "Tools.CV.hpp"

namespace Tools {
    // --------------------------------------------------------------------------------
    //                                Letterbox
    // --------------------------------------------------------------------------------

    cv::Mat Letterbox::Make(const cv::Mat &src, int h, int w, Letterbox &let)
    {
        int   in_w     = src.cols;   // width
        int   in_h     = src.rows;   // height
        int   tar_w    = w;
        int   tar_h    = h;
        float r        = std::min(float(tar_h) / in_h, float(tar_w) / in_w);
        int   inside_w = round(in_w * r);
        int   inside_h = round(in_h * r);
        int   pad_w    = tar_w - inside_w;
        int   pad_h    = tar_h - inside_h;

        cv::Mat resize_img;

        cv::resize(src, resize_img, cv::Size(inside_w, inside_h));

        pad_w = pad_w / 2;
        pad_h = pad_h / 2;

        let._fill_height = pad_h;
        let._fill_width  = pad_w;
        let._r           = r;
        let.height       = in_h;
        let.width        = in_w;
        let.let_height   = h;
        let.let_width    = w;

        int top    = int(round(pad_h - 0.1));
        int bottom = int(round(pad_h + 0.1));
        int left   = int(round(pad_w - 0.1));
        int right  = int(round(pad_w + 0.1));
        cv::copyMakeBorder(resize_img, resize_img, top, bottom, left, right, 0, cv::Scalar(114, 114, 114));
        return resize_img;
    }

    cv::Rect Letterbox::Restore(const cv::Rect &box) const
    {
        cv::Rect scaled_box;
        scaled_box.x      = (box.x - this->_fill_width) / this->_r;
        scaled_box.y      = (box.y - this->_fill_height) / this->_r;
        scaled_box.width  = box.width / this->_r;
        scaled_box.height = box.height / this->_r;
        // 避免越界
        if (scaled_box.x < 0) scaled_box.x = 0;
        if (scaled_box.y < 0) scaled_box.y = 0;
        if (scaled_box.x > this->width) scaled_box.x = this->width;
        if (scaled_box.y > this->height) scaled_box.y = this->height;
        if (scaled_box.width + scaled_box.x > this->width) scaled_box.width = this->width - scaled_box.x;
        if (scaled_box.height + scaled_box.y > this->height) scaled_box.height = this->height - scaled_box.y;
        return scaled_box;
    }

    cv::Point Letterbox::Restore(const cv::Point &pt) const
    {
        cv::Point r;
        r.x = (pt.x - this->_fill_width) / this->_r;
        r.y = (pt.y - this->_fill_height) / this->_r;
        if (r.x < 0) r.x = 0;
        if (r.y < 0) r.y = 0;
        if (r.x > this->width) r.x = this->width;
        if (r.y > this->height) r.y = this->height;
        return r;
    }

    void IMGProcess::AdaptiveHistogramEqualization(const cv::Mat    &src,
                                                   cv::Mat          &out,
                                                   int               limit,
                                                   const cv::Size2i &size)
    {
        if (src.channels() != 1)
            RUN_ERR("Only single channel images are accepted");
        auto clahe = cv::createCLAHE();
        clahe->setClipLimit(limit);
        clahe->setTilesGridSize(size);
        clahe->apply(src, out);
        return;
    }

    void IMGProcess::AdjustGamma(const cv::Mat &src, cv::Mat &dst, float gamma)
    {
        if (src.type() != CV_8UC1)
            RUN_ERR("Type must be CV_8UC1");
        // 预处理
        uint8_t table[255];
        for (int i = 0; i < 255; i++)
            table[i] = powf(i / 255.0f, gamma) * 255;
        if (dst.size() != src.size())
            dst = cv::Mat(src.size(), CV_8UC1);
        // 应用
        size_t s = src.rows * src.cols;
        auto   p = (uint8_t *)src.data;
        auto   r = (uint8_t *)dst.data;
        for (size_t i = 0; i < s; i++)
            r[i] = table[p[i]];
        return;
    }

    // 自适应中值滤波窗口实现  // 图像 计算座标, 窗口尺寸和 最大尺寸
    static uchar adaptiveProcess(const cv::Mat &im, int row, int col, int kernelSize, int maxSize)
    {
        std::vector<uchar> pixels;
        for (int a = -kernelSize / 2; a <= kernelSize / 2; a++)
            for (int b = -kernelSize / 2; b <= kernelSize / 2; b++) {
                pixels.push_back(im.at<uchar>(row + a, col + b));
            }
        sort(pixels.begin(), pixels.end());
        auto min = pixels[0];
        auto max = pixels[kernelSize * kernelSize - 1];
        auto med = pixels[kernelSize * kernelSize / 2];
        auto zxy = im.at<uchar>(row, col);
        if (med > min && med < max) {
            // to B
            if (zxy > min && zxy < max)
                return zxy;
            else
                return med;
        } else {
            kernelSize += 2;
            if (kernelSize <= maxSize)
                return adaptiveProcess(im, row, col, kernelSize, maxSize);   // 增大窗口尺寸，继续A过程。
            else
                return med;
        }
    }

    static cv::Mat _adaptiveMediaFilter(const cv::Mat &src, int minSize, int maxSize)
    {
        cv::Mat dst;
        // 扩展图像的边界
        cv::copyMakeBorder(src, dst, maxSize / 2, maxSize / 2, maxSize / 2, maxSize / 2, cv::BorderTypes::BORDER_REFLECT);
        // 图像循环
        for (int j = maxSize / 2; j < dst.rows - maxSize / 2; j++) {
            for (int i = maxSize / 2; i < dst.cols * dst.channels() - maxSize / 2; i++) {
                dst.at<uchar>(j, i) = adaptiveProcess(dst, j, i, minSize, maxSize);
            }
        }
        cv::Rect r = cv::Rect(cv::Point(maxSize / 2, maxSize / 2), cv::Point(dst.cols - maxSize / 2, dst.rows - maxSize / 2));
        return dst(r);
    }

    cv::Mat IMGProcess::AdaptiveMediaFilter(const cv::Mat &src, int minSize, int maxSize)
    {
        if (src.channels() == 1)
            return _adaptiveMediaFilter(src, minSize, maxSize);
        std::vector<cv::Mat> channels;
        cv::split(src, channels);
        for (auto &ch : channels)
            ch = _adaptiveMediaFilter(ch, minSize, maxSize);
        cv::Mat dst;
        cv::merge(channels, dst);
        return dst;
    }

    AIMethod::Tensor<float> ImageBGRToNCHW(const std::vector<cv::Mat>    &imgs,
                                           const cv::Size2i              &size,
                                           std::vector<Tools::Letterbox> &lets,
                                           std::string                   &err)
    {
        int                block_size = size.height * size.width * 3;
        std::vector<float> tmp(imgs.size() * block_size);
        cv::Mat            img_f32;
        for (size_t i = 0; i < imgs.size(); i++) {
            auto   img  = imgs[i];
            float *data = tmp.data() + i * block_size;
            if (img.size().empty()) {
                err = "The picture cannot be empty";
                return AIMethod::Tensor<float>();
            }
            if (img.type() != CV_32FC3) {
                img.convertTo(img_f32, CV_32FC3);   // 转float
                img = img_f32;
            }
            if (img.channels() != 3) {
                err = "The picture must be 3 channels";
                return AIMethod::Tensor<float>();
            }
            if (img.type() != CV_32FC3) {
                err = "Image data type conversion failed. Procedure";
                return AIMethod::Tensor<float>();
            }
            // 图像变换
            Tools::Letterbox let;
            if (img.size() != size)
                img = Tools::Letterbox::Make(img, size.height, size.width, let);
            lets.push_back(let);
            // BGR2RGB
            for (int i = 0; i < img.rows; i++) {
                for (int j = 0; j < img.cols; j++) {
                    auto &tmp                                        = img.at<cv::Vec3f>(i, j);
                    data[i * img.cols + j + 0]                       = tmp[2];
                    data[i * img.cols + j + 1 * img.cols * img.rows] = tmp[1];
                    data[i * img.cols + j + 2 * img.cols * img.rows] = tmp[0];
                }
            }
        }
        return AIMethod::Tensor<float>({(int)imgs.size(), 3, size.height, size.width}, std::move(tmp));
    }

    // --------------------------------------------------------------------------------
    //                                FaceRecognize
    // --------------------------------------------------------------------------------

    Tools::FaceRecognize::FaceRecognize(const string &data_face, const string &data_eye, const string &data_mouth)
    {
        this->faceCascade.load(data_face);
        if (!data_eye.empty()) this->eyesCascade.load(data_eye);
        if (!data_mouth.empty()) this->mouthCascade.load(data_mouth);
        return;
    }

    std::vector<FaceRecognize::Result> Tools::FaceRecognize::Recognize(const cv::Mat &img)
    {
        std::vector<FaceRecognize::Result> results;
        if (img.empty())
            return results;
        std::vector<cv::Rect> faces;
        // 检测人脸
        this->faceCascade.detectMultiScale(img,
                                           faces,
                                           1.1,
                                           2,
                                           0 | cv::CASCADE_SCALE_IMAGE,
                                           cv::Size(10, 10));
        for (auto &face : faces) {
            Result   ret;
            cv::Rect eye[2];
            auto     faceROI = img(face);
            ret.faceBox      = face;
            // 检测眼睛
            if (!this->eyesCascade.empty()) {
                std::vector<cv::Rect> boxs;
                this->eyesCascade.detectMultiScale(faceROI,
                                                   boxs,
                                                   1.1,
                                                   2,
                                                   0 | cv::CASCADE_SCALE_IMAGE,
                                                   cv::Size(5, 5));
                for (size_t i = 0; i < 2 && i < boxs.size(); i++)
                    eye[i] = boxs[i];
            }
            ret.leftEyeBox  = eye[0];
            ret.rightEyeBox = eye[1];
            // 检测嘴部
            if (!this->eyesCascade.empty()) {
                std::vector<cv::Rect> boxs;
                this->mouthCascade.detectMultiScale(faceROI,
                                                    boxs,
                                                    1.1,
                                                    2,
                                                    0 | cv::CASCADE_SCALE_IMAGE,
                                                    cv::Size(5, 5));
                if (boxs.size() > 0) ret.mouthBox = boxs[0];
            }
            ret.isIntegrity = ret.mouthBox.area() > 0 &&
                              ret.leftEyeBox.area() > 0 &&
                              ret.rightEyeBox.area() > 0;
            // 修正位置
            if (ret.isIntegrity) {
                // 计算嘴巴位置 A
                int ax = ret.mouthBox.x + (ret.mouthBox.width >> 1);
                int ay = ret.mouthBox.y + (ret.mouthBox.height >> 1);
                // 计算眼睛位置
                int bx = ret.leftEyeBox.x + (ret.leftEyeBox.width >> 1);
                int cx = ret.rightEyeBox.x + (ret.rightEyeBox.width >> 1);
                int by = ret.leftEyeBox.y + (ret.leftEyeBox.height >> 1);
                int cy = ret.rightEyeBox.y + (ret.rightEyeBox.height >> 1);
                // 向量AB与AC的叉积的结果
                // AB=(bx-ax,by-ay)
                double ans = (bx - ax) * (cx - ax) - (by - ay) * (cy - ay);
                if (ans < 0) {
                    // 顺时针,调换眼睛
                    auto t          = ret.leftEyeBox;
                    ret.leftEyeBox  = ret.rightEyeBox;
                    ret.rightEyeBox = t;
                }
            }
            if (ret.leftEyeBox.area() > 0) {
                ret.leftEyeBox.x += ret.faceBox.x;
                ret.leftEyeBox.y += ret.faceBox.y;
            }
            if (ret.rightEyeBox.area() > 0) {
                ret.rightEyeBox.x += ret.faceBox.x;
                ret.rightEyeBox.y += ret.faceBox.y;
            }
            if (ret.mouthBox.area() > 0) {
                ret.mouthBox.x += ret.faceBox.x;
                ret.mouthBox.y += ret.faceBox.y;
            }
            results.push_back($(ret));
        }
        return results;
    }

    void FaceRecognize::DrawBox(cv::Mat                   &img,
                                const std::vector<Result> &results,
                                const cv::Scalar           color_face,
                                const cv::Scalar           color_eye,
                                const cv::Scalar           color_mouth)
    {
        for (auto &ret : results) {
            cv::rectangle(img, ret.faceBox, color_face, 2);
            if (ret.leftEyeBox.area() > 0) {
                int cx = ret.leftEyeBox.x + (ret.leftEyeBox.width >> 1) - 2;
                int cy = ret.leftEyeBox.y + (ret.leftEyeBox.height >> 1) - 2;
                cv::rectangle(img, ret.leftEyeBox, color_eye);
                cv::putText(img, "L", cv::Point2i(cx, cy), cv::FONT_HERSHEY_SIMPLEX, 0.5, color_eye);
            }
            if (ret.rightEyeBox.area() > 0) {
                int cx = ret.rightEyeBox.x + (ret.rightEyeBox.width >> 1) - 2;
                int cy = ret.rightEyeBox.y + (ret.rightEyeBox.height >> 1) - 2;
                cv::rectangle(img, ret.rightEyeBox, color_eye);
                cv::putText(img, "R", cv::Point2i(cx, cy), cv::FONT_HERSHEY_SIMPLEX, 0.5, color_eye);
            }
            if (ret.mouthBox.area() > 0)
                cv::rectangle(img, ret.mouthBox, color_mouth);
        }
        return;
    }
}   // namespace Tools
