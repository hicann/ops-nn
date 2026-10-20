/** Copyright (c) 2026 Huawei Technologies Co., Ltd. See LICENSE for licensing details. */
#ifndef OPS_NN_ADDLAYERNORM_DYNAMICQUANT_FUSION_PASS_H
#define OPS_NN_ADDLAYERNORM_DYNAMICQUANT_FUSION_PASS_H
#include "ge/fusion/pass/pattern_fusion_pass.h"
namespace ops {
class __attribute__((visibility("default"))) AddLayerNormDynamicQuantFusionPass : public ge::fusion::PatternFusionPass {
protected:
    std::vector<ge::fusion::PatternUniqPtr> Patterns() override;
    bool MeetRequirements(const std::unique_ptr<ge::fusion::MatchResult>& match) override;
    ge::fusion::GraphUniqPtr Replacement(const std::unique_ptr<ge::fusion::MatchResult>& match) override;
};
} // namespace ops
#endif
