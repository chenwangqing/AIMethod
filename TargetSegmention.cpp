
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
        int  batch_size = pred.GetShape()[0];
        int  nm         = proto.GetShape()[1];
        int  nc         = pred.GetShape()[2] - 5 - nm;   // 类别
        auto dets       = det.Yolo(pred, lets, nm);
        auto proto_in   = proto.Slice(0, {nm, -1});
        for (int i = 0; i < batch_size; i++) {
            Tensor<float> marks({(int)dets[i].size(), nm});
            for (auto &v : dets[i]) {
                auto idx  = pred.GetIdx(i, v.index, 0);
                auto mask = pred.Value() + idx + nm + 5;   // 最后的是mask
                idx       = marks.GetIdx(i, 0);
                memcpy(marks.Value() + idx, mask, nm);
            }
            
        }
        return list;
    }
}   // namespace AIMethod
