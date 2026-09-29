/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file layer_norm_onnx_fusion_pass.cpp
 * \brief ONNX 导入的 LayerNorm 展开子图 --> LayerNorm
 */
#include "layer_norm_onnx_fusion_pass.h"

#include <cmath>
#include <limits>
#include <set>
#include <string>
#include <vector>

#include "common/inc/error_util.h"
#include "cube_utils/cube_fp16_t.h"
#include "es_math_ops.h"
#include "es_nn_ops.h"
#include "platform/platform_info.h"

namespace ops {
namespace {
const std::string kPassName = "LayerNormONNXFusionPass";

// 只捕获所有形态都有的 7 个节点；可选节点（cast0 / clip / mul0 / add1）事后按边或按槽取。
constexpr size_t kCapReduceMean0 = 0U;
constexpr size_t kCapSub0 = 1U;
constexpr size_t kCapPow0 = 2U;
constexpr size_t kCapReduceMean1 = 3U;
constexpr size_t kCapAdd0 = 4U;
constexpr size_t kCapSqrt0 = 5U;
constexpr size_t kCapDiv0 = 6U;
constexpr size_t kCapNum = 7U;

// pattern 输入布局。固定前 4 个，其余按形态顺序追加：
//   0=x  1=axes0  2=axes1  3=eps  [pow_exp]  [clip_max, clip_min]  [gamma, beta]
constexpr size_t kInputX = 0U;
constexpr size_t kInputAxes0 = 1U;
constexpr size_t kInputAxes1 = 2U;
constexpr size_t kInputEps = 3U;
constexpr size_t kFixedInputNum = 4U;
// affine 恒定多贡献 gamma / beta 两个边界输入
constexpr size_t kAffineInputNum = 2U;

constexpr double kEpsilonUpperBound = 0.1;
constexpr float32_t kDefaultEpsilon = 1e-7f;
constexpr float kPowExpSquare = 2.0f;
// fp16 的 1.0 的位模式。
constexpr uint16_t kHalfOneBits = 15360U;

constexpr size_t kScalarDimNum = 1U;
constexpr int64_t kScalarDimValue = 1L;

const std::set<std::string> kSupportedNpuArch = {"3510", "5102"};

enum class ClipForm { kNone, kMaxThenMin, kMinThenMax };

struct PatternShape {
    bool has_cast;
    bool has_affine;
    ClipForm clip;
};

// clip 只与「无 cast + 无 affine」组合。
// **顺序有意义，带 affine 的必须排在对应的无 affine 之前**：无 affine 的 pattern 也能命中带 affine
const std::vector<PatternShape> kShapes = {
    {false, true, ClipForm::kNone},        // case1
    {false, false, ClipForm::kNone},       // case2
    {true, true, ClipForm::kNone},         // case3
    {true, false, ClipForm::kNone},        // case4
    {false, false, ClipForm::kMaxThenMin}, // case5
    {false, false, ClipForm::kMinThenMax}, // case6
};

//   bit0 Square/Pow  bit1 RealDiv/Div  bit2 Add
constexpr size_t kBitHasPow = 0U;
constexpr size_t kBitUseDiv = 1U;
constexpr size_t kBitEpsFirst = 2U;
// gamma / beta
constexpr size_t kBitGammaFirst = 3U;
constexpr size_t kBitBetaFirst = 4U;
constexpr size_t kCommConfigNum = 32U;
// pattern 总数 = 带 affine 的 2 个形态 × 32 配置字 + 无 affine 的 4 个形态 × 8 配置字
//              （无 affine 没有 gamma/beta 输入槽，bit3/bit4 无意义故跳过） = 64 + 32 = 96。
constexpr size_t kExpectedPatternNum = 96U;

bool IsBitSet(size_t config, size_t bit) { return ((config >> bit) & 1U) != 0U; }

bool IsSupportedPlatform()
{
    fe::PlatFormInfos platform_infos;
    fe::OptionalInfos optional_infos;
    if (fe::PlatformInfoManager::Instance().GetPlatformInfoWithOutSocVersion(platform_infos, optional_infos) !=
        SUCCESS) {
        OPS_LOG_D(kPassName.c_str(), "Get platform info failed, Skip.");
        return false;
    }
    std::string arch_str;
    if (!platform_infos.GetPlatformRes("version", "NpuArch", arch_str)) {
        OPS_LOG_D(kPassName.c_str(), "Platform NpuArch not found, Skip.");
        return false;
    }
    OPS_LOG_D(kPassName.c_str(), "Platform NpuArch: %s", arch_str.c_str());
    if (kSupportedNpuArch.count(arch_str) == 0U) {
        OPS_LOG_D(kPassName.c_str(), "Platform %s is not supported, Skip.", arch_str.c_str());
        return false;
    }
    return true;
}

std::string TypeOf(const GNode& node)
{
    AscendString type;
    if (node.GetType(type) != GRAPH_SUCCESS || type.GetString() == nullptr) {
        return std::string();
    }
    return std::string(type.GetString());
}

std::string NameOf(const GNode& node)
{
    AscendString name;
    if (node.GetName(name) != GRAPH_SUCCESS || name.GetString() == nullptr) {
        return std::string();
    }
    return std::string(name.GetString());
}

bool GetProducer(const GNode& node, int32_t index, GNode& producer)
{
    const auto peer = node.GetInDataNodesAndPortIndexs(index);
    if (peer.first == nullptr) {
        return false;
    }
    producer = *peer.first;
    return true;
}

bool CheckDynamic(const GNode& node, int32_t index)
{
    TensorDesc desc;
    if (node.GetInputDesc(index, desc) != GRAPH_SUCCESS) {
        return false;
    }
    for (const int64_t dim : desc.GetShape().GetDims()) {
        if (dim < 0L) {
            return true;
        }
    }
    return false;
}

// GetAllInputs() 是列表，本规则每槽只喂一个节点故取 at(0)。
bool SlotIo(const std::vector<SubgraphInput>& boundary, size_t slot, NodeIo& io)
{
    if (slot >= boundary.size()) {
        OPS_LOG_E(kPassName.c_str(), "guard_rejected : boundary slot %zu out of range (%zu).", slot, boundary.size());
        return false;
    }
    const auto node_inputs = boundary[slot].GetAllInputs();
    if (node_inputs.empty()) {
        OPS_LOG_E(kPassName.c_str(), "guard_rejected : boundary slot %zu has no consumer.", slot);
        return false;
    }
    io = node_inputs.at(0);
    return true;
}

// 按边界槽取整型向量（axes 常量）。
bool IntVecAt(const std::vector<SubgraphInput>& boundary, size_t slot, std::vector<int64_t>& values)
{
    NodeIo io;
    if (!SlotIo(boundary, slot, io)) {
        return false;
    }
    const int32_t index = static_cast<int32_t>(io.index);
    Tensor tensor;
    if (io.node.GetInputConstData(index, tensor) != GRAPH_SUCCESS) {
        return false;
    }
    TensorDesc desc;
    if (io.node.GetInputDesc(index, desc) != GRAPH_SUCCESS) {
        return false;
    }
    const uint8_t* data = tensor.GetData();
    if (data == nullptr) {
        return false;
    }
    const DataType dtype = desc.GetDataType();
    if (dtype == DT_INT32) {
        const size_t count = tensor.GetSize() / sizeof(int32_t);
        const int32_t* typed = reinterpret_cast<const int32_t*>(data);
        for (size_t i = 0U; i < count; ++i) {
            values.push_back(static_cast<int64_t>(typed[i]));
        }
        return true;
    }
    if (dtype == DT_INT64) {
        const size_t count = tensor.GetSize() / sizeof(int64_t);
        const int64_t* typed = reinterpret_cast<const int64_t*>(data);
        for (size_t i = 0U; i < count; ++i) {
            values.push_back(typed[i]);
        }
        return true;
    }
    OPS_LOG_D(kPassName.c_str(), "guard_rejected : axes const dtype is neither int32 nor int64.");
    return false;
}

struct InputLayout {
    size_t total = kFixedInputNum;
    size_t pow_exp = 0U;
    size_t clip_max = 0U;
    size_t clip_min = 0U;
    size_t gamma = 0U;
    size_t beta = 0U;
};

InputLayout MakeLayout(bool has_pow, bool has_clip, bool has_affine)
{
    InputLayout l;
    size_t next = kFixedInputNum;
    if (has_pow) {
        l.pow_exp = next++;
    }
    if (has_clip) {
        l.clip_max = next++;
        l.clip_min = next++;
    }
    if (has_affine) {
        l.gamma = next++;
        l.beta = next++;
    }
    l.total = next;
    return l;
}

// 按边界槽取标量常量的值。
bool ScalarAt(const std::vector<SubgraphInput>& boundary, size_t slot, float32_t& value)
{
    NodeIo io;
    if (!SlotIo(boundary, slot, io)) {
        return false;
    }
    const int32_t index = static_cast<int32_t>(io.index);
    Tensor tensor;
    if (io.node.GetInputConstData(index, tensor) != GRAPH_SUCCESS) {
        OPS_LOG_D(kPassName.c_str(), "guard_rejected : input %d of %s is not a const.", index, NameOf(io.node).c_str());
        return false;
    }
    TensorDesc desc;
    if (io.node.GetInputDesc(index, desc) != GRAPH_SUCCESS) {
        return false;
    }

    const std::vector<int64_t> tensor_dims = desc.GetShape().GetDims();
    if (!tensor_dims.empty()) {
        if (tensor_dims[0] < 0L) {
            OPS_LOG_D(kPassName.c_str(), "guard_rejected : cannot be applied for unknown shape.");
            return false;
        }
        if ((tensor_dims.size() != kScalarDimNum) || (tensor_dims[0] != kScalarDimValue)) {
            OPS_LOG_D(kPassName.c_str(), "guard_rejected : the const input of %s must be scalar.",
                      NameOf(io.node).c_str());
            return false;
        }
    }

    const DataType dtype = desc.GetDataType();
    const uint8_t* data = tensor.GetData();
    if (data == nullptr) {
        OPS_LOG_E(kPassName.c_str(), "guard_rejected : const data of %s is null.", NameOf(io.node).c_str());
        return false;
    }
    if (dtype == DT_FLOAT16) {
        value = static_cast<float32_t>(fp16_t(*(reinterpret_cast<const uint16_t*>(data))));
        return true;
    }
    if (dtype == DT_FLOAT) {
        value = *(reinterpret_cast<const float32_t*>(data));
        return true;
    }
    OPS_LOG_D(kPassName.c_str(), "guard_rejected : %s dtype is not float or fp16.", NameOf(io.node).c_str());
    return false;
}

// 校验常量 shape：rank 必须等于 axes 个数，且每维等于 x 在对应轴上的长度。
bool CheckShapeAt(const std::vector<SubgraphInput>& boundary, size_t slot, const std::vector<int64_t>& axes,
                  const std::vector<int64_t>& expect)
{
    NodeIo io;
    if (!SlotIo(boundary, slot, io)) {
        return false;
    }
    const int32_t index = static_cast<int32_t>(io.index);
    Tensor tensor;
    if (io.node.GetInputConstData(index, tensor) != GRAPH_SUCCESS) {
        OPS_LOG_D(kPassName.c_str(), "guard_rejected : input %d of %s is not a const.", index, NameOf(io.node).c_str());
        return false;
    }
    TensorDesc const_desc;
    if (io.node.GetInputDesc(index, const_desc) != GRAPH_SUCCESS) {
        return false;
    }
    const std::vector<int64_t> tensor_dims = const_desc.GetShape().GetDims();
    if (tensor_dims.size() != axes.size()) {
        OPS_LOG_D(kPassName.c_str(), "guard_rejected : the dim of const %s is not equal to the length of axes.",
                  NameOf(io.node).c_str());
        return false;
    }
    for (size_t i = 0U; i < axes.size(); ++i) {
        const size_t axis = static_cast<size_t>(axes[i]);
        if ((axis >= expect.size()) || (tensor_dims[i] != expect[axis])) {
            OPS_LOG_D(kPassName.c_str(), "guard_rejected : the value of %s is not equal to the expect shape.",
                      NameOf(io.node).c_str());
            return false;
        }
    }
    return true;
}

// 命中后从捕获节点还原出的全部信息。
struct Matched {
    GNode reduce_mean0;
    GNode sub0;
    GNode pow0;
    GNode reduce_mean1;
    GNode add0;
    GNode sqrt0;
    GNode div0;

    GNode mul0;
    GNode add1;

    bool has_pow = false;
    bool has_clip = false;
    bool with_affine = false;

    bool x_dynamic = false;
    bool gamma_dynamic = false;
    bool beta_dynamic = false;

    std::vector<int64_t> axes;
    std::vector<int64_t> input_dims;
    float32_t epsilon = kDefaultEpsilon;

    InputLayout layout;
    std::vector<SubgraphInput> boundary;
};

bool GetCapturedNode(const std::unique_ptr<MatchResult>& match_result, size_t index, GNode& node)
{
    NodeIo node_io;
    if (match_result->GetCapturedTensor(index, node_io) != SUCCESS) {
        OPS_LOG_D(kPassName.c_str(), "guard_rejected : get captured node %zu failed.", index);
        return false;
    }
    node = node_io.node;
    return true;
}

bool CollectMatched(const std::unique_ptr<MatchResult>& match_result, Matched& m)
{
    if (!GetCapturedNode(match_result, kCapReduceMean0, m.reduce_mean0) ||
        !GetCapturedNode(match_result, kCapSub0, m.sub0) || !GetCapturedNode(match_result, kCapPow0, m.pow0) ||
        !GetCapturedNode(match_result, kCapReduceMean1, m.reduce_mean1) ||
        !GetCapturedNode(match_result, kCapAdd0, m.add0) || !GetCapturedNode(match_result, kCapSqrt0, m.sqrt0) ||
        !GetCapturedNode(match_result, kCapDiv0, m.div0)) {
        return false;
    }

    // 可选 clip：sqrt0 的输入 0 若不是 add0，则是 Maximum / Minimum 链
    GNode sqrt_input;
    if (!GetProducer(m.sqrt0, 0, sqrt_input)) {
        return false;
    }
    if (NameOf(sqrt_input) != NameOf(m.add0)) {
        const std::string t = TypeOf(sqrt_input);
        if ((t != "Minimum") && (t != "Maximum")) {
            return false;
        }
        m.has_clip = true;
    }

    std::vector<SubgraphInput> boundary_inputs;
    const auto boundary = match_result->ToSubgraphBoundary();
    if ((boundary == nullptr) || (boundary->GetAllInputs(boundary_inputs) != SUCCESS)) {
        OPS_LOG_E(kPassName.c_str(), "guard_rejected : cannot get subgraph boundary inputs.");
        return false;
    }
    // base = 无 affine 时的边界输入个数：4 固定槽 + Pow 指数 + clip 的 max/min。affine 恒定多两个槽，
    m.has_pow = (TypeOf(m.pow0) == "Pow");
    const size_t base = kFixedInputNum + (m.has_pow ? 1U : 0U) + (m.has_clip ? 2U : 0U);
    if (boundary_inputs.size() == base) {
        m.with_affine = false;
    } else if (boundary_inputs.size() == (base + kAffineInputNum)) {
        m.with_affine = true;
    } else {
        OPS_LOG_E(kPassName.c_str(), "guard_rejected : boundary input num %zu is neither %zu nor %zu.",
                  boundary_inputs.size(), base, base + kAffineInputNum);
        return false;
    }

    m.layout = MakeLayout(m.has_pow, m.has_clip, m.with_affine);
    m.boundary = boundary_inputs;

    if (m.with_affine) {
        NodeIo gamma_io;
        NodeIo beta_io;
        if (!SlotIo(m.boundary, m.layout.gamma, gamma_io) || !SlotIo(m.boundary, m.layout.beta, beta_io)) {
            OPS_LOG_E(kPassName.c_str(), "guard_rejected : cannot locate mul0 / add1 of the affine tail.");
            return false;
        }
        m.mul0 = gamma_io.node;
        m.add1 = beta_io.node;
    }
    return true;
}

bool GetAxes(Matched& m)
{
    TensorDesc input_desc;
    if (m.reduce_mean0.GetInputDesc(0, input_desc) != GRAPH_SUCCESS) {
        return false;
    }
    m.input_dims = input_desc.GetShape().GetDims();
    const size_t dims_size = m.input_dims.size();
    if (dims_size < 1U) {
        OPS_LOG_D(kPassName.c_str(), "guard_rejected : input shape must be greater than one.");
        return false;
    }
    m.axes.clear();
    if (!IntVecAt(m.boundary, kInputAxes0, m.axes)) {
        OPS_LOG_D(kPassName.c_str(), "guard_rejected : fail to get axes const from reducemean0.");
        return false;
    }
    if (m.axes.empty()) {
        OPS_LOG_D(kPassName.c_str(), "guard_rejected : attr axes of reducemean can't be empty.");
        return false;
    }
    for (size_t i = 0U; i < m.axes.size(); ++i) {
        m.axes[i] = (m.axes[i] > 0) ? m.axes[i] : (m.axes[i] + static_cast<int64_t>(dims_size));
    }
    if (m.axes.back() != static_cast<int64_t>(dims_size - 1U)) {
        OPS_LOG_D(kPassName.c_str(), "guard_rejected : the axes format is not supported.");
        return false;
    }
    for (size_t i = 0U; (i + 1U) < m.axes.size(); ++i) {
        if ((m.axes[i + 1U] - m.axes[i]) != 1L) {
            OPS_LOG_D(kPassName.c_str(), "guard_rejected : the axes format is not supported.");
            return false;
        }
    }
    return true;
}

bool CheckClipValues(const Matched& m)
{
    float32_t max_val = 0.0f;
    float32_t min_val = 0.0f;
    if (!ScalarAt(m.boundary, m.layout.clip_max, max_val) || !ScalarAt(m.boundary, m.layout.clip_min, min_val)) {
        OPS_LOG_D(kPassName.c_str(), "guard_rejected : fail to get clip const value.");
        return false;
    }
    if ((min_val != INFINITY) || (max_val != 0.0f)) {
        OPS_LOG_D(kPassName.c_str(), "guard_rejected : clip val min is [%f], clip max is [%f].", min_val, max_val);
        return false;
    }
    return true;
}

bool CheckValue(Matched& m)
{
    bool keep_dims = false;
    if (m.reduce_mean0.GetAttr(AscendString("keep_dims"), keep_dims) != GRAPH_SUCCESS) {
        OPS_LOG_E(kPassName.c_str(), "guard_rejected : fail to get keep_dims from reducemean0.");
        return false;
    }
    if (!keep_dims) {
        OPS_LOG_D(kPassName.c_str(), "guard_rejected : the attr keep_dims in reducemean must be true.");
        return false;
    }

    if (m.has_pow) {
        float32_t exp = 0.0f;
        if (!ScalarAt(m.boundary, m.layout.pow_exp, exp)) {
            OPS_LOG_D(kPassName.c_str(), "guard_rejected : fail to get value from const node of pow0.");
            return false;
        }
        if (std::fabs(exp - kPowExpSquare) > std::numeric_limits<float>::epsilon()) {
            OPS_LOG_D(kPassName.c_str(), "guard_rejected : the exp of pow is %f, which should be equal to 2.", exp);
            return false;
        }
    }

    if (m.has_clip && !CheckClipValues(m)) {
        return false;
    }

    if (m.with_affine && !m.gamma_dynamic && !CheckShapeAt(m.boundary, m.layout.gamma, m.axes, m.input_dims)) {
        OPS_LOG_D(kPassName.c_str(), "guard_rejected : the shape of mul0 is not as expect.");
        return false;
    }
    if (m.with_affine && !m.beta_dynamic && !CheckShapeAt(m.boundary, m.layout.beta, m.axes, m.input_dims)) {
        OPS_LOG_D(kPassName.c_str(), "guard_rejected : the shape of add1 is not as expect.");
        return false;
    }

    if (!ScalarAt(m.boundary, kInputEps, m.epsilon)) {
        OPS_LOG_D(kPassName.c_str(), "guard_rejected : fail to get epsilon from const node of add0.");
        return false;
    }
    if (static_cast<double>(m.epsilon) > kEpsilonUpperBound) {
        OPS_LOG_D(kPassName.c_str(), "guard_rejected : the epsilon of add0 is %f, which should be close to 0.",
                  m.epsilon);
        return false;
    }
    return true;
}

// pattern 的输入个数与各可选输入的序号，由形态唯一确定。
PatternUniqPtr MakePattern(const PatternShape& shape, size_t config)
{
    const InputLayout layout = MakeLayout(IsBitSet(config, kBitHasPow), shape.clip != ClipForm::kNone,
                                          shape.has_affine);
    auto graph_builder = es::EsGraphBuilder("layer_norm_onnx_fusion_pattern");
    auto inputs = graph_builder.CreateInputs(layout.total);

    const auto& x = inputs[kInputX];
    auto rm0 = es::ReduceMean(x, inputs[kInputAxes0], true);
    auto sub0 = es::Sub(x, rm0);

    es::EsTensorHolder pow_input = sub0;
    if (shape.has_cast) {
        pow_input = es::Cast(sub0, static_cast<int64_t>(DT_FLOAT));
    }
    es::EsTensorHolder pow0 = IsBitSet(config, kBitHasPow) ? es::Pow(pow_input, inputs[layout.pow_exp]) :
                                                             es::Square(pow_input);

    auto rm1 = es::ReduceMean(pow0, inputs[kInputAxes1], true);
    auto add0 = IsBitSet(config, kBitEpsFirst) ? es::Add(inputs[kInputEps], rm1) : es::Add(rm1, inputs[kInputEps]);

    es::EsTensorHolder pre_sqrt = add0;
    if (shape.clip == ClipForm::kMaxThenMin) {
        auto max_node = es::Maximum(add0, inputs[layout.clip_max]);
        pre_sqrt = es::Minimum(max_node, inputs[layout.clip_min]);
    } else if (shape.clip == ClipForm::kMinThenMax) {
        auto min_node = es::Minimum(add0, inputs[layout.clip_min]);
        pre_sqrt = es::Maximum(min_node, inputs[layout.clip_max]);
    }
    auto sqrt0 = es::Sqrt(pre_sqrt);

    es::EsTensorHolder div0 = IsBitSet(config, kBitUseDiv) ? es::Div(sub0, sqrt0) : es::RealDiv(sub0, sqrt0);

    es::EsTensorHolder out = div0;
    if (shape.has_affine) {
        auto mul0 = IsBitSet(config, kBitGammaFirst) ? es::Mul(inputs[layout.gamma], div0) :
                                                       es::Mul(div0, inputs[layout.gamma]);
        out = IsBitSet(config, kBitBetaFirst) ? es::Add(inputs[layout.beta], mul0) : es::Add(mul0, inputs[layout.beta]);
    }

    auto graph = graph_builder.BuildAndReset({out});
    auto pattern = std::make_unique<Pattern>(std::move(*graph));
    pattern->CaptureTensor({*rm0.GetProducer(), 0})
        .CaptureTensor({*sub0.GetProducer(), 0})
        .CaptureTensor({*pow0.GetProducer(), 0})
        .CaptureTensor({*rm1.GetProducer(), 0})
        .CaptureTensor({*add0.GetProducer(), 0})
        .CaptureTensor({*sqrt0.GetProducer(), 0})
        .CaptureTensor({*div0.GetProducer(), 0});
    return pattern;
}

// without-affine 时在替换图里造 gamma=1 / beta=0 常量。
bool CreateAffineConst(es::EsGraphBuilder& builder, const Matched& m, es::EsTensorHolder& gamma,
                       es::EsTensorHolder& beta, TensorDesc& param_desc)
{
    TensorDesc div_out_desc;
    TensorDesc div_in_desc;
    if ((m.div0.GetOutputDesc(0, div_out_desc) != GRAPH_SUCCESS) ||
        (m.div0.GetInputDesc(0, div_in_desc) != GRAPH_SUCCESS)) {
        OPS_LOG_E(kPassName.c_str(), "replacement_rejected : cannot get desc of div0.");
        return false;
    }
    std::vector<int64_t> const_dims(m.axes.size(), 0L);
    for (size_t i = 0U; i < m.axes.size(); ++i) {
        const size_t axis = static_cast<size_t>(m.axes[i]);
        if (axis >= m.input_dims.size()) {
            OPS_LOG_E(kPassName.c_str(), "replacement_rejected : axes out of range when creating const.");
            return false;
        }
        const_dims[i] = m.input_dims[axis];
    }
    int64_t numel = 1L;
    for (const int64_t dim : const_dims) {
        numel *= dim;
    }
    if (numel <= 0L) {
        OPS_LOG_E(kPassName.c_str(), "replacement_rejected : const numel is not positive.");
        return false;
    }

    const DataType dtype = div_out_desc.GetDataType();
    const Format format = div_in_desc.GetFormat();
    param_desc = TensorDesc(Shape(const_dims), format, dtype);
    param_desc.SetOriginShape(Shape(const_dims));
    param_desc.SetOriginFormat(format);
    if (dtype == DT_FLOAT16) {
        gamma = builder.CreateConst<uint16_t>(std::vector<uint16_t>(static_cast<size_t>(numel), kHalfOneBits),
                                              const_dims, DT_FLOAT16, format);
        beta = builder.CreateConst<uint16_t>(std::vector<uint16_t>(static_cast<size_t>(numel), 0U), const_dims,
                                             DT_FLOAT16, format);
        return true;
    }
    if (dtype == DT_FLOAT) {
        gamma = builder.CreateConst<float32_t>(std::vector<float32_t>(static_cast<size_t>(numel), 1.0f), const_dims,
                                               DT_FLOAT, format);
        beta = builder.CreateConst<float32_t>(std::vector<float32_t>(static_cast<size_t>(numel), 0.0f), const_dims,
                                              DT_FLOAT, format);
        return true;
    }
    OPS_LOG_D(kPassName.c_str(), "guard_rejected : div0 dtype is not in (float16, float32).");
    return false;
}
} // namespace

std::vector<PatternUniqPtr> LayerNormONNXFusionPass::Patterns()
{
    OPS_LOG_D(kPassName.c_str(), "Enter Patterns for LayerNormONNXFusionPass.");
    if (!IsSupportedPlatform()) {
        return {};
    }
    std::vector<PatternUniqPtr> patterns;
    patterns.reserve(kExpectedPatternNum);
    for (const auto& shape : kShapes) {
        for (size_t config = 0U; config < kCommConfigNum; ++config) {
            if (!shape.has_affine && (IsBitSet(config, kBitGammaFirst) || IsBitSet(config, kBitBetaFirst))) {
                continue;
            }
            patterns.emplace_back(MakePattern(shape, config));
        }
    }
    return patterns;
}

bool LayerNormONNXFusionPass::MeetRequirements(const std::unique_ptr<MatchResult>& match_result)
{
    OPS_LOG_D(kPassName.c_str(), "guard_begin.");
    Matched m;
    if (!CollectMatched(match_result, m)) {
        return false;
    }

    m.x_dynamic = CheckDynamic(m.reduce_mean0, 0);
    if (m.with_affine) {
        m.gamma_dynamic = CheckDynamic(m.mul0, 0) || CheckDynamic(m.mul0, 1);
        m.beta_dynamic = CheckDynamic(m.add1, 0) || CheckDynamic(m.add1, 1);
    }

    if (!GetAxes(m)) {
        return false;
    }
    if (!CheckValue(m)) {
        return false;
    }
    // 动态 shape 且无 affine 时不融合：造不出正确的常量 shape。
    if (!m.with_affine && m.x_dynamic) {
        OPS_LOG_D(kPassName.c_str(), "guard_rejected : the scene is dynamic and without affine.");
        return false;
    }

    // 无 affine 时要自建 gamma=1 / beta=0 常量，只支持 fp16 / fp32。
    // **必须在这里校验、不能留到 Replacement**：Replacement 返回 nullptr 会触发框架 Assert → 整图编译失败。
    if (!m.with_affine) {
        TensorDesc div_out_desc;
        if (m.div0.GetOutputDesc(0, div_out_desc) != GRAPH_SUCCESS) {
            OPS_LOG_D(kPassName.c_str(), "guard_rejected : cannot get output desc of div0.");
            return false;
        }
        const DataType div_dtype = div_out_desc.GetDataType();
        if ((div_dtype != DT_FLOAT) && (div_dtype != DT_FLOAT16)) {
            OPS_LOG_D(kPassName.c_str(), "guard_rejected : div0 dtype is not in (float16, float32).");
            return false;
        }
    }

    OPS_LOG_D(kPassName.c_str(), "guard_passed.");
    return true;
}

std::unique_ptr<Graph> LayerNormONNXFusionPass::Replacement(const std::unique_ptr<MatchResult>& match_result)
{
    OPS_LOG_D(kPassName.c_str(), "replacement_begin.");
    Matched m;
    if (!CollectMatched(match_result, m) || !GetAxes(m)) {
        return nullptr;
    }
    float32_t epsilon = kDefaultEpsilon;
    if (!ScalarAt(m.boundary, kInputEps, epsilon)) {
        return nullptr;
    }

    const InputLayout& layout = m.layout;

    std::vector<SubgraphInput> subgraph_inputs;
    const auto boundary = match_result->ToSubgraphBoundary();
    if ((boundary == nullptr) || (boundary->GetAllInputs(subgraph_inputs) != SUCCESS)) {
        OPS_LOG_E(kPassName.c_str(), "replacement_rejected : cannot get subgraph boundary inputs.");
        return nullptr;
    }
    if (subgraph_inputs.size() != layout.total) {
        OPS_LOG_E(kPassName.c_str(), "replacement_rejected : subgraph input num %zu != expect %zu.",
                  subgraph_inputs.size(), layout.total);
        return nullptr;
    }

    auto graph_builder = es::EsGraphBuilder("replacement");
    std::vector<es::EsTensorHolder> placeholders;
    std::vector<TensorDesc> boundary_descs;
    placeholders.reserve(layout.total);
    boundary_descs.reserve(layout.total);
    for (size_t i = 0U; i < layout.total; ++i) {
        const auto node_inputs = subgraph_inputs[i].GetAllInputs();
        if (node_inputs.empty()) {
            OPS_LOG_E(kPassName.c_str(), "replacement_rejected : subgraph input %zu has no consumer.", i);
            return nullptr;
        }
        const auto& node_io = node_inputs.at(0);
        TensorDesc desc;
        if (node_io.node.GetInputDesc(static_cast<int32_t>(node_io.index), desc) != GRAPH_SUCCESS) {
            OPS_LOG_E(kPassName.c_str(), "replacement_rejected : cannot get boundary input desc.");
            return nullptr;
        }
        const std::string input_name = "input_" + std::to_string(i);
        placeholders.emplace_back(graph_builder.CreateInput(static_cast<int64_t>(i), input_name.c_str(),
                                                            desc.GetDataType(), desc.GetFormat(),
                                                            desc.GetShape().GetDims()));
        boundary_descs.emplace_back(desc);
    }

    es::EsTensorHolder gamma;
    es::EsTensorHolder beta;
    TensorDesc gamma_desc;
    TensorDesc beta_desc;
    if (m.with_affine) {
        gamma = placeholders[layout.gamma];
        beta = placeholders[layout.beta];
        gamma_desc = boundary_descs[layout.gamma];
        beta_desc = boundary_descs[layout.beta];
    } else {
        TensorDesc const_desc;
        if (!CreateAffineConst(graph_builder, m, gamma, beta, const_desc)) {
            return nullptr;
        }
        gamma_desc = const_desc;
        beta_desc = const_desc;
    }

    // begin_params_axis 取 beta 的 rank 取负；without-affine 时 beta rank 恰为 axes.size()。
    const int64_t beta_rank = static_cast<int64_t>(beta_desc.GetShape().GetDims().size());
    auto layer_norm = es::LayerNorm(placeholders[kInputX], gamma, beta, m.axes[0], 0L - beta_rank, epsilon);

    GNode* layer_norm_node = layer_norm.y.GetProducer();
    if (layer_norm_node == nullptr) {
        OPS_LOG_E(kPassName.c_str(), "replacement_rejected : cannot get producer of LayerNorm.");
        return nullptr;
    }
    // desc 全部从被消除的节点直接拷贝。
    if ((layer_norm_node->UpdateInputDesc(0, boundary_descs[kInputX]) != GRAPH_SUCCESS) ||
        (layer_norm_node->UpdateInputDesc(1, gamma_desc) != GRAPH_SUCCESS) ||
        (layer_norm_node->UpdateInputDesc(2, beta_desc) != GRAPH_SUCCESS)) {
        OPS_LOG_E(kPassName.c_str(), "replacement_rejected : cannot update input desc of LayerNorm.");
        return nullptr;
    }

    TensorDesc y_desc;
    TensorDesc mean_desc;
    TensorDesc variance_desc;
    const GNode& y_source = m.with_affine ? m.add1 : m.div0;
    if ((y_source.GetOutputDesc(0, y_desc) != GRAPH_SUCCESS) ||
        (m.reduce_mean0.GetOutputDesc(0, mean_desc) != GRAPH_SUCCESS) ||
        (m.reduce_mean1.GetOutputDesc(0, variance_desc) != GRAPH_SUCCESS)) {
        OPS_LOG_E(kPassName.c_str(), "replacement_rejected : cannot get output desc from matched nodes.");
        return nullptr;
    }
    if ((layer_norm_node->UpdateOutputDesc(0, y_desc) != GRAPH_SUCCESS) ||
        (layer_norm_node->UpdateOutputDesc(1, mean_desc) != GRAPH_SUCCESS) ||
        (layer_norm_node->UpdateOutputDesc(2, variance_desc) != GRAPH_SUCCESS)) {
        OPS_LOG_E(kPassName.c_str(), "replacement_rejected : cannot update output desc of LayerNorm.");
        return nullptr;
    }

    // axes / eps 折进属性后其 Const 节点变为悬空，交由 GE 后续死代码消除处理。
    OPS_LOG_D(kPassName.c_str(), "replacement_done.");
    return graph_builder.BuildAndReset({layer_norm.y});
}

REG_FUSION_PASS(LayerNormONNXFusionPass).Stage(CustomPassStage::kAfterInferShape);
} // namespace ops
