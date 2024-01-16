
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
            auto                                  mf  = ((float)mw / let.width);
            auto                                  hf  = ((float)mh / let.height);
            for (size_t n = 0; n < dets[i].size(); n++) {
                auto &v    = dets[i][n];
                auto  idx  = pred.GetIdx(i, v.index, 0);
                auto  mask = pred.Value() + idx + nm + 5;   // 最后的是mask
                marks.CopyTo(marks.GetIdx(n, 0), mask, nm);
                boxs.At(n, 0) = v.box.x * mf;
                boxs.At(n, 1) = v.box.y * hf;
                boxs.At(n, 2) = (v.box.x + v.box.width) * mf;
                boxs.At(n, 3) = (v.box.y + v.box.height) * hf;
            }

            marks = op.Sigmoid(op.Mul(marks, proto_in)).Slice(0, {-1, mh, mw});

            for (int k = 0; k < dets[i].size(); k++) {
                auto x1 = boxs.At(k, 0);
                auto y1 = boxs.At(k, 1);
                auto x2 = boxs.At(k, 2);
                auto y2 = boxs.At(k, 3);
                auto m  = marks.Slice(marks.GetIdx(k, 0, 0), {mh, mw});
                for (int r = 0; r < mh; r++) {
                    auto rv = m.Value() + m.GetIdx(r, 0);
                    if (!(r >= y1 && r < y2)) {
                        // 不在范围 整行设置0
                        memset(rv, 0, sizeof(float) * mw);
                        continue;
                    }
                    for (int c = 0; c < mw; c++) {
                        if (!(c >= x1 && c < x2))
                            rv[c] = 0;
                    }
                }
                // 向上采样(双线性插值)
                cv::Mat mask_img(cv::Size2i{mw, mh}, CV_32FC1);
                memcpy(mask_img.data, m.Value(), m.Size() * sizeof(float));
                cv::Mat out;
                cv::resize(mask_img, out, cv::Size2i{let.width, let.height}, 0, 0, cv::INTER_LINEAR);
                // 阈值
                cv::Mat to;
                cv::threshold(out, to, 0.5f, 255, cv::THRESH_BINARY);
                to.convertTo(out, CV_8U);
                Result rs;
                rs.box        = dets[i][k].box;
                rs.classId    = dets[i][k].classId;
                rs.confidence = dets[i][k].confidence;
                rs.index      = dets[i][k].index;
                rs.mask       = $(out);
                rlist.push_back($(rs));
            }
            list.push_back($(rlist));
        }
        return list;
    }
}   // namespace AIMethod
