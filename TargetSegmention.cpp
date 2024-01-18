
#include "TargetSegmention.hpp"

namespace AIMethod {
    std::vector<std::vector<TargetSegmention::Result>> TargetSegmention::Yolo(const TargetDetection               &det,
                                                                              const Tensor<float>                 &pred,
                                                                              const Tensor<float>                 &proto,
                                                                              const std::vector<Tools::Letterbox> &lets) const
    {
        std::vector<std::vector<TargetSegmention::Result>> list;
        if (pred.Size() == 0 ||
            pred.GetShape().size() != 3 ||
            proto.Size() == 0 ||
            proto.GetShape().size() != 4 ||
            pred.GetShape()[0] != proto.GetShape()[0])
            return list;
        int batch_size = pred.GetShape()[0];
        int nm         = proto.GetShape()[1];
        int nc         = pred.GetShape()[2] - 5 - nm;   // 类别
        int mh         = proto.GetShape()[2];
        int mw         = proto.GetShape()[3];
        // 进行检测
        auto dets = det.Yolo(pred, lets, nm);
        // 预测蒙版，将所有不在预测框内的内容归零。
        for (int i = 0; i < batch_size; i++) {
            std::vector<TargetSegmention::Result> rlist;
            auto                                  proto_in = proto.Slice(proto.GetIdx(i, 0, 0, 0), {nm, proto.GetShape()[2] * proto.GetShape()[3]});
            Tensor<float>                         marks({(int)dets[i].size(), nm});
            Tensor<float>                         boxs({(int)dets[i].size(), 4});
            auto                                 &let = lets[i];
            auto                                  mf  = ((float)mw / let.let_width);
            auto                                  hf  = ((float)mh / let.let_height);
            for (size_t n = 0; n < dets[i].size(); n++) {
                auto &v    = dets[i][n];
                auto  idx  = pred.GetIdx(i, v._index, 0);
                auto  mask = pred.Value() + idx + nc + 5;   // 最后的是mask
                marks.CopyTo(marks.GetIdx(n, 0), mask, nm);
                boxs.At(n, 0) = v._box.x * mf;
                boxs.At(n, 1) = v._box.y * hf;
                boxs.At(n, 2) = (v._box.x + v._box.width) * mf;
                boxs.At(n, 3) = (v._box.y + v._box.height) * hf;
            }

            marks = op.Sigmoid(op.Mul(marks, proto_in)).Slice(0, {-1, mh, mw});

            for (size_t k = 0; k < dets[i].size(); k++) {
                auto x1 = boxs.At(k, 0);
                auto y1 = boxs.At(k, 1);
                auto x2 = boxs.At(k, 2);
                auto y2 = boxs.At(k, 3);
                auto m  = marks.Slice(marks.GetIdx(k, 0, 0), {mh, mw});

                auto   &v = dets[i][k];
                int     w = x2 - x1;
                int     h = y2 - y1;
                cv::Mat mask_roi(cv::Size2i{w, h}, CV_32F);
                float  *data = (float *)mask_roi.data;
                for (int r = 0; r < h; r++) {
                    int  y  = r + y1;
                    auto rv = m.Value() + m.GetIdx(y, 0) + (int)x1;
                    auto p  = data + r * w;
                    for (int c = 0; c < w; c++)
                        p[c] = rv[c];
                }

                cv::Mat thr;
                cv::Mat out;
                // 向上采样(双线性插值)
                cv::resize(mask_roi, out, cv::Size2i{v.box.width, v.box.height}, 0, 0, cv::INTER_LINEAR);
                // 二值化
                cv::threshold(out, thr, 0.50f, 255, cv::THRESH_BINARY);
                thr.convertTo(mask_roi, CV_8UC1);

                Result rs;
                rs.box        = dets[i][k].box;
                rs.classId    = dets[i][k].classId;
                rs.confidence = dets[i][k].confidence;
                rs.mask       = $(mask_roi);
                rlist.push_back($(rs));
            }
            list.push_back($(rlist));
        }
        return list;
    }
}   // namespace AIMethod
