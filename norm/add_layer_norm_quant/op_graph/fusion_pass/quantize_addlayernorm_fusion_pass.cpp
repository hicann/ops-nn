/** Copyright (c) 2026 Huawei Technologies Co., Ltd. See LICENSE for licensing details. */
#include "quantize_addlayernorm_fusion_pass.h"
#include "add_layer_norm_quant_fusion_common.h"

namespace ops {
using namespace aln_quant;
namespace {
const std::string kPass = "QuantizeAddLayerNormPass";
// b0=AscendQuantV2, b1=optional offset, b2=optional bias, b3=single branch.
// Generate dual-branch patterns before single-branch patterns.
constexpr size_t kQuantTypeBit = 0U;
constexpr size_t kOffsetBit = 1U;
constexpr size_t kBiasBit = 2U;
constexpr size_t kSingleBranchBit = 3U;
constexpr size_t kPatternBitNum = 4U;
constexpr size_t kPatternConfigNum = 1U << kPatternBitNum;

bool IsBitSet(size_t config, size_t bit) { return ((config >> bit) & 1U) != 0U; }

PatternUniqPtr MakePattern(size_t config)
{
    const std::string type = IsBitSet(config, kQuantTypeBit) ? "AscendQuantV2" : "Quantize";
    const bool has_offset = IsBitSet(config, kOffsetBit);
    const bool has_bias = IsBitSet(config, kBiasBit);
    const int branches = IsBitSet(config, kSingleBranchBit) ? 1 : 2;
    EsGraphBuilder builder(kPass.c_str());
    int index = 0;
    auto input = [&]() { return builder.CreateInput(index++); };
    auto x1 = input();
    auto x2 = input();
    auto gamma = input();
    auto beta = input();
    auto b = has_bias ? input() : EsTensorHolder(nullptr);
    auto norm = ge::es::AddLayerNorm(x1, x2, gamma, beta, b);
    std::vector<EsTensorHolder> quant;
    for (int i = 0; i < branches; ++i) {
        auto scale = input();
        auto zero_points = has_offset ? input() : EsTensorHolder(nullptr);
        quant.push_back(type == "Quantize" ? ge::es::Quantize(norm.y, scale, zero_points, "torch.qint8", -1) :
                                             ge::es::AscendQuantV2(norm.y, scale, zero_points));
    }
    auto outputs = quant;
    outputs.push_back(norm.x);
    return aln_quant::MakePattern(builder, outputs, norm.y, quant);
}
bool QuantAttributes(const GNode& q)
{
    if (!Int8Output(q)) {
        return false;
    }
    if (Type(q) == "Quantize") {
        ge::AscendString dtype;
        return q.GetAttr("dtype", dtype) == ge::GRAPH_SUCCESS && dtype == "torch.qint8";
    }
    int64_t dst = ge::DT_INT8;
    q.GetAttr("dst_type", dst);
    return dst == ge::DT_INT8;
}
} // namespace
std::vector<PatternUniqPtr> QuantizeAddLayerNormPass::Patterns()
{
    if (!IsSupportedPlatform()) {
        OPS_LOG_D(kPass.c_str(), "patterns_skipped : platform unsupported or unavailable.");
        return {};
    }
    std::vector<PatternUniqPtr> patterns;
    patterns.reserve(kPatternConfigNum);
    for (size_t config = 0U; config < kPatternConfigNum; ++config) {
        patterns.push_back(MakePattern(config));
    }
    OPS_LOG_D(kPass.c_str(), "patterns_built : %zu.", patterns.size());
    return patterns;
}
bool QuantizeAddLayerNormPass::MeetRequirements(const std::unique_ptr<MatchResult>& match)
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
        if (!QuantAttributes(q) || InputType(q, 0) != dtype) {
            OPS_LOG_D(kPass.c_str(), "guard_rejected : incompatible quant type/dtype.");
            return false;
        }
        const auto scale_dtype = InputType(q, 1);
        if (!IsFloat(scale_dtype) || (scale_dtype != ge::DT_FLOAT && scale_dtype != dtype)) {
            OPS_LOG_D(kPass.c_str(), "guard_rejected : unsupported scale dtype.");
            return false;
        }
    }
    if (nodes.quant.size() == 2) {
        const auto& q0 = nodes.quant[0];
        const auto& q1 = nodes.quant[1];
        if (HasInput(q0, 2) != HasInput(q1, 2) || InputType(q0, 1) != InputType(q1, 1) ||
            (HasInput(q0, 2) && InputType(q0, 2) != InputType(q1, 2))) {
            OPS_LOG_D(kPass.c_str(), "guard_rejected : dual quant optional inputs/dtypes differ.");
            return false;
        }
    }
    OPS_LOG_D(kPass.c_str(), "guard_passed.");
    return true;
}
GraphUniqPtr QuantizeAddLayerNormPass::Replacement(const std::unique_ptr<MatchResult>& match)
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
    std::array<EsTensorHolder, 2> scales{nullptr, nullptr}, zeros{nullptr, nullptr};
    for (size_t i = 0; i < nodes.quant.size(); ++i) {
        scales[i] = in.Get(nodes.quant[i], 1);
        zeros[i] = in.Get(nodes.quant[i], 2);
    }
    const auto& q = nodes.quant[0];
    const auto type = Type(q);
    float epsilon = kDefaultEpsilon;
    bool additional = false;
    nodes.norm.GetAttr("epsilon", epsilon);
    nodes.norm.GetAttr("additional_output", additional);
    auto x1 = in.Get(nodes.norm, 0), x2 = in.Get(nodes.norm, 1), gamma = in.Get(nodes.norm, 2);
    auto beta = in.Get(nodes.norm, 3), bias = in.Get(nodes.norm, 4);
    auto fused = ge::es::AddLayerNormQuant(x1, x2, gamma, beta, bias, scales[0], scales[1], zeros[0], zeros[1],
                                           "static", epsilon, additional, type == "Quantize");
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
    return Finish(builder, outputs, *match, kPass, in.Shapes());
}
REG_FUSION_PASS(QuantizeAddLayerNormPass).Stage(ge::CustomPassStage::kAfterBuiltinFusionPass);
} // namespace ops
