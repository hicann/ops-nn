/** Copyright (c) 2026 Huawei Technologies Co., Ltd. See LICENSE for licensing details. */
#include "addlayernorm_dynamicquant_fusion_pass.h"
#include "add_layer_norm_quant_fusion_common.h"

namespace ops {
using namespace aln_quant;
namespace {
const std::string kPass = "AddLayerNormDynamicQuantFusionPass";
PatternUniqPtr MakePattern(int branches, bool bias, bool mul, int smooth_mask, int swap_mask, int group_mask)
{
    EsGraphBuilder builder(kPass.c_str());
    int index = 0;
    auto input = [&]() { return builder.CreateInput(index++); };
    auto x1 = input();
    auto x2 = input();
    auto gamma = input();
    auto beta = input();
    auto b = bias ? input() : EsTensorHolder(nullptr);
    auto norm = ge::es::AddLayerNorm(x1, x2, gamma, beta, b);
    std::vector<EsTensorHolder> outputs, quant, scales;
    for (int i = 0; i < branches; ++i) {
        auto smooth = (mul || ((smooth_mask >> i) & 1)) ? input() : EsTensorHolder(nullptr);
        auto y = norm.y;
        if (mul) {
            y = ((swap_mask >> i) & 1) ? ge::es::Mul(smooth, y) : ge::es::Mul(y, smooth);
        }
        // Match each branch with or without group_index.
        auto group_index = ((group_mask >> i) & 1) ? input() : EsTensorHolder(nullptr);
        auto q = ge::es::DynamicQuant(y, mul ? EsTensorHolder(nullptr) : smooth, group_index);
        outputs.push_back(q.y);
        quant.push_back(q.y);
        scales.push_back(q.scale);
    }
    outputs.push_back(norm.x);
    outputs.insert(outputs.end(), scales.begin(), scales.end());
    return aln_quant::MakePattern(builder, outputs, norm.y, quant);
}
} // namespace
std::vector<PatternUniqPtr> AddLayerNormDynamicQuantFusionPass::Patterns()
{
    if (!IsSupportedPlatform()) {
        OPS_LOG_D(kPass.c_str(), "patterns_skipped : platform unsupported or unavailable.");
        return {};
    }
    std::vector<PatternUniqPtr> patterns;
    for (int branches : {2, 1}) {
        for (bool bias : {false, true}) {
            for (int mask = 0; mask < (1 << branches); ++mask) {
                for (int group_mask = 0; group_mask < (1 << branches); ++group_mask) {
                    patterns.push_back(MakePattern(branches, bias, false, mask, 0, group_mask));
                    patterns.push_back(MakePattern(branches, bias, true, 0, mask, group_mask));
                }
            }
        }
    }
    OPS_LOG_D(kPass.c_str(), "patterns_built : %zu.", patterns.size());
    return patterns;
}
bool AddLayerNormDynamicQuantFusionPass::MeetRequirements(const std::unique_ptr<MatchResult>& match)
{
    OPS_LOG_D(kPass.c_str(), "guard_begin.");
    Captures nodes;
    if (!match || !Capture(*match, nodes)) {
        OPS_LOG_D(kPass.c_str(), "guard_rejected : capture unavailable.");
        return false;
    }
    OPS_LOG_D(kPass.c_str(), "guard_capture : node=%s.", Name(nodes.norm).c_str());
    if (!CheckBoundary(nodes, kPass) || !CheckNorm(nodes.norm, kPass)) {
        return false;
    }
    const auto dtype = InputType(nodes.norm, 0);
    for (const auto& q : nodes.quant) {
        auto source = q.GetInDataNodesAndPortIndexs(0).first;
        if (source && Type(*source) == "Mul" && HasInput(q, 1)) {
            OPS_LOG_D(kPass.c_str(), "guard_rejected : quant after Mul has smooth input.");
            return false;
        }
        if (InputType(q, 0) != dtype) {
            OPS_LOG_D(kPass.c_str(), "guard_rejected : quant input dtype differs from normalization dtype.");
            return false;
        }
        if (HasInput(q, 1) && InputType(q, 1) != dtype) {
            OPS_LOG_D(kPass.c_str(), "guard_rejected : smooth dtype differs from normalization dtype.");
            return false;
        }
    }
    OPS_LOG_D(kPass.c_str(), "guard_passed.");
    return true;
}
GraphUniqPtr AddLayerNormDynamicQuantFusionPass::Replacement(const std::unique_ptr<MatchResult>& match)
{
    OPS_LOG_D(kPass.c_str(), "replacement_begin.");
    Captures nodes;
    if (!match || !Capture(*match, nodes)) {
        OPS_LOG_E(kPass.c_str(), "replacement_rejected : capture unavailable.");
        return nullptr;
    }
    EsGraphBuilder builder(kPass.c_str());
    ReplacementInputs in(builder, *match);
    if (!in.Valid()) {
        OPS_LOG_E(kPass.c_str(), "replacement_rejected : cannot prepare boundary inputs.");
        return nullptr;
    }
    std::array<EsTensorHolder, 2> smooth{nullptr, nullptr};
    for (size_t i = 0; i < nodes.quant.size(); ++i) {
        const auto& q = nodes.quant[i];
        auto source = q.GetInDataNodesAndPortIndexs(0).first;
        if (source && Type(*source) == "Mul") {
            auto first = source->GetInDataNodesAndPortIndexs(0).first;
            smooth[i] = in.Get(*source, first && Name(*first) == Name(nodes.norm) ? 1 : 0);
        } else {
            smooth[i] = in.Get(q, 1);
        }
    }
    float epsilon = kDefaultEpsilon;
    bool additional = false;
    nodes.norm.GetAttr("epsilon", epsilon);
    nodes.norm.GetAttr("additional_output", additional);
    auto fused = ge::es::AddLayerNormQuant(in.Get(nodes.norm, 0), in.Get(nodes.norm, 1), in.Get(nodes.norm, 2),
                                           in.Get(nodes.norm, 3), in.Get(nodes.norm, 4), smooth[0], smooth[1], nullptr,
                                           nullptr, "dynamic", epsilon, additional);
    if (!SetQuantInputDescs(fused)) {
        OPS_LOG_E(kPass.c_str(), "replacement_rejected : cannot set input descriptors.");
        return nullptr;
    }
    if (!SeedQuantOutputs(fused)) {
        OPS_LOG_E(kPass.c_str(), "replacement_rejected : cannot initialize unused output descriptors.");
        return nullptr;
    }
    std::vector<EsTensorHolder> outputs{fused.y1};
    if (nodes.quant.size() == 2) {
        outputs.push_back(fused.y2);
    }
    outputs.push_back(fused.x);
    outputs.push_back(fused.out_scales1);
    if (nodes.quant.size() == 2) {
        outputs.push_back(fused.out_scales2);
    }
    return Finish(builder, outputs, *match, kPass, in.Shapes());
}
REG_FUSION_PASS(AddLayerNormDynamicQuantFusionPass).Stage(ge::CustomPassStage::kAfterBuiltinFusionPass);
} // namespace ops
