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
 * \file layer_norm_onnx_mul_fusion_pass.cpp
 * \brief ONNX 导入的 LayerNorm 展开子图（双 Sub 形态）--> LayerNormV3
 */
#include "layer_norm_onnx_mul_fusion_pass.h"

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
const std::string kPassName = "LayerNormONNXMULFusionPass";

// 只捕获所有形态都存在的 8 个节点；
constexpr size_t kCapMean0 = 0U;
constexpr size_t kCapSub0 = 1U;
constexpr size_t kCapSub1 = 2U;
constexpr size_t kCapSquare = 3U;
constexpr size_t kCapMean1 = 4U;
constexpr size_t kCapAdd0 = 5U;
constexpr size_t kCapRsqrt0 = 6U;
constexpr size_t kCapDiv0 = 7U;

// pattern 输入布局：0=x 1=axes0 2=axes1 3=eps [pow_exp] [rsqrt_exp] [gamma, beta]
constexpr size_t kInputX = 0U;
constexpr size_t kInputAxes0 = 1U;
constexpr size_t kInputAxes1 = 2U;
constexpr size_t kInputEps = 3U;
constexpr size_t kFixedInputNum = 4U;
// affine 恒定 gamma / beta 两个边界输入
constexpr size_t kAffineInputNum = 2U;

constexpr float kPowExpSquare = 2.0f;
constexpr float kPowSqrtExpValue = 0.5f;
// fp16 的 1.0 的位模式。
constexpr uint16_t kHalfOneBits = 15360U;

constexpr size_t kScalarDimNum = 1U;
constexpr int64_t kScalarDimValue = 1L;
constexpr int64_t kLastAxis = -1L;
constexpr int64_t kBeginParamsAxis = -1L;

// bit0 Mul/Pow;  bit1 Sqrt/Pow;  bit2 Add-eps;  bit3 Mul-gamma;  bit4 Add-beta;
// Mul / Add 可交换;    bit3/bit4 仅 with_affine 有意义（无 affine 没有 gamma/beta 槽），Patterns() 在那一档跳过。
constexpr size_t kBitWithPow = 0U;
constexpr size_t kBitRsqrtIsPow = 1U;
constexpr size_t kBitEpsFirst = 2U;
constexpr size_t kBitGammaFirst = 3U;
constexpr size_t kBitBetaFirst = 4U;
constexpr size_t kCommConfigNum = 32U;
// pattern 总数 = 32（有 affine，bit0~bit4 全用） + 8（无 affine，bit3/bit4 无意义跳过）。
constexpr size_t kExpectedPatternNum = 40U;

bool IsBitSet(size_t config, size_t bit) { return ((config >> bit) & 1U) != 0U; }

const std::set<std::string> kSupportedNpuArch = {"3510", "5102"};

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
        OPS_LOG_D(kPassName.c_str(), "guard_rejected : get tensor axes failed.");
        return false;
    }
    TensorDesc desc;
    if (io.node.GetInputDesc(index, desc) != GRAPH_SUCCESS) {
        return false;
    }
    const uint8_t* data = tensor.GetData();
    if (data == nullptr) {
        OPS_LOG_D(kPassName.c_str(), "guard_rejected : get const data failed.");
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
    OPS_LOG_D(kPassName.c_str(), "guard_rejected : reduce mean not support this axes type.");
    return false;
}

struct InputLayout {
    size_t total = kFixedInputNum;
    size_t pow_exp = 0U;
    size_t rsqrt_exp = 0U;
    size_t gamma = 0U;
    size_t beta = 0U;
};

InputLayout MakeLayout(bool with_pow, bool rsqrt_is_pow, bool with_affine)
{
    InputLayout l;
    size_t next = kFixedInputNum;
    if (with_pow) {
        l.pow_exp = next++;
    }
    if (rsqrt_is_pow) {
        l.rsqrt_exp = next++;
    }
    if (with_affine) {
        l.gamma = next++;
        l.beta = next++;
    }
    l.total = next;
    return l;
}

// 按边界槽取标量常量的值。口径：shape 必须是标量。
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

struct Matched {
    GNode mean0;
    GNode sub0;
    GNode sub1;
    GNode square_node; // mul0 或 pow0
    GNode mean1;
    GNode add0;
    GNode rsqrt0;
    GNode div0;
    GNode mul1;
    GNode add1;

    bool with_pow = false;
    bool rsqrt_is_pow = false;
    bool with_affine = false;

    std::vector<int64_t> axes0;
    std::vector<int64_t> axes1;
    std::vector<int64_t> input_dims;
    float32_t epsilon = 0.0f;

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
    if (!GetCapturedNode(match_result, kCapMean0, m.mean0) || !GetCapturedNode(match_result, kCapSub0, m.sub0) ||
        !GetCapturedNode(match_result, kCapSub1, m.sub1) || !GetCapturedNode(match_result, kCapSquare, m.square_node) ||
        !GetCapturedNode(match_result, kCapMean1, m.mean1) || !GetCapturedNode(match_result, kCapAdd0, m.add0) ||
        !GetCapturedNode(match_result, kCapRsqrt0, m.rsqrt0) || !GetCapturedNode(match_result, kCapDiv0, m.div0)) {
        return false;
    }
    m.with_pow = (TypeOf(m.square_node) == "Pow");
    m.rsqrt_is_pow = (TypeOf(m.rsqrt0) == "Pow");

    if (NameOf(m.sub0) == NameOf(m.sub1)) {
        OPS_LOG_D(kPassName.c_str(),
                  "guard_rejected : sub0 and sub1 are the same node, this is LayerNormONNXFusionPass's shape.");
        return false;
    }

    std::vector<SubgraphInput> boundary_inputs;
    const auto boundary = match_result->ToSubgraphBoundary();
    if ((boundary == nullptr) || (boundary->GetAllInputs(boundary_inputs) != SUCCESS)) {
        OPS_LOG_E(kPassName.c_str(), "guard_rejected : cannot get subgraph boundary inputs.");
        return false;
    }
    // base = 无 affine 时的边界输入个数：4 固定槽 + Pow 平方的指数 + Pow 开方的指数。affine 恒定多两个槽，
    const size_t base = kFixedInputNum + (m.with_pow ? 1U : 0U) + (m.rsqrt_is_pow ? 1U : 0U);
    if (boundary_inputs.size() == base) {
        m.with_affine = false;
    } else if (boundary_inputs.size() == (base + kAffineInputNum)) {
        m.with_affine = true;
    } else {
        OPS_LOG_E(kPassName.c_str(), "guard_rejected : boundary input num %zu is neither %zu nor %zu.",
                  boundary_inputs.size(), base, base + kAffineInputNum);
        return false;
    }

    m.layout = MakeLayout(m.with_pow, m.rsqrt_is_pow, m.with_affine);
    m.boundary = boundary_inputs;

    if (m.with_affine) {
        NodeIo gamma_io;
        NodeIo beta_io;
        if (!SlotIo(m.boundary, m.layout.gamma, gamma_io) || !SlotIo(m.boundary, m.layout.beta, beta_io)) {
            OPS_LOG_E(kPassName.c_str(), "guard_rejected : cannot locate mul1 / add1 of the affine tail.");
            return false;
        }
        m.mul1 = gamma_io.node;
        m.add1 = beta_io.node;
    }
    return true;
}

bool CheckAxesAndKeepDims(Matched& m)
{
    if (!IntVecAt(m.boundary, kInputAxes0, m.axes0) || !IntVecAt(m.boundary, kInputAxes1, m.axes1)) {
        return false;
    }
    bool keep_dims0 = false;
    bool keep_dims1 = false;
    if ((m.mean0.GetAttr(AscendString("keep_dims"), keep_dims0) != GRAPH_SUCCESS) ||
        (m.mean1.GetAttr(AscendString("keep_dims"), keep_dims1) != GRAPH_SUCCESS)) {
        OPS_LOG_E(kPassName.c_str(), "guard_rejected : get attr keep_dims failed.");
        return false;
    }
    if ((m.axes0.size() != 1U) || (m.axes1.size() != 1U) || (m.axes0[0] != m.axes1[0])) {
        OPS_LOG_D(kPassName.c_str(), "guard_rejected : the axes of mean are not same.");
        return false;
    }
    if (!keep_dims0 || !keep_dims1) {
        OPS_LOG_D(kPassName.c_str(), "guard_rejected : the attr keep_dims of mean is not true.");
        return false;
    }

    TensorDesc input_desc;
    if (m.mean0.GetInputDesc(0, input_desc) != GRAPH_SUCCESS) {
        return false;
    }
    m.input_dims = input_desc.GetShape().GetDims();
    const size_t dims_size = m.input_dims.size();
    if (dims_size < 1U) {
        OPS_LOG_D(kPassName.c_str(), "guard_rejected : input shape must be greater to one.");
        return false;
    }
    if ((m.axes0[0] != kLastAxis) && (m.axes0[0] != static_cast<int64_t>(dims_size - 1U))) {
        OPS_LOG_D(kPassName.c_str(), "guard_rejected : the axes of mean is not the last dim of input.");
        return false;
    }
    return true;
}

// 两个 Pow 指数常量：开方支路须为 0.5，平方支路须为 2。
bool CheckExponents(const Matched& m)
{
    if (m.rsqrt_is_pow) {
        float32_t rsqrt_exp = 0.0f;
        if (!ScalarAt(m.boundary, m.layout.rsqrt_exp, rsqrt_exp)) {
            OPS_LOG_D(kPassName.c_str(), "guard_rejected : fail to get value from const node of rsqrt0.");
            return false;
        }
        if (std::fabs(rsqrt_exp - kPowSqrtExpValue) > std::numeric_limits<float>::epsilon()) {
            OPS_LOG_D(kPassName.c_str(), "guard_rejected : the exp of pow is %f, which should be equal to 0.5.",
                      rsqrt_exp);
            return false;
        }
    }
    if (m.with_pow) {
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
    return true;
}

// gamma / beta 必须都是 1 维且长度相等。without-affine 场景新造的常量恒满足，故不必判。
bool CheckAffineShapes(const Matched& m)
{
    TensorDesc gamma_desc;
    TensorDesc beta_desc;
    NodeIo gamma_io;
    NodeIo beta_io;
    if (!SlotIo(m.boundary, m.layout.gamma, gamma_io) || !SlotIo(m.boundary, m.layout.beta, beta_io)) {
        return false;
    }
    if ((gamma_io.node.GetInputDesc(static_cast<int32_t>(gamma_io.index), gamma_desc) != GRAPH_SUCCESS) ||
        (beta_io.node.GetInputDesc(static_cast<int32_t>(beta_io.index), beta_desc) != GRAPH_SUCCESS)) {
        OPS_LOG_D(kPassName.c_str(), "guard_rejected : cannot get gamma/beta desc.");
        return false;
    }
    const std::vector<int64_t> gamma_dims = gamma_desc.GetShape().GetDims();
    const std::vector<int64_t> beta_dims = beta_desc.GetShape().GetDims();
    if (gamma_dims.size() != 1U) {
        OPS_LOG_D(kPassName.c_str(), "guard_rejected : gamma dims is [%zu], which not equal to 1.", gamma_dims.size());
        return false;
    }
    if (beta_dims.size() != 1U) {
        OPS_LOG_D(kPassName.c_str(), "guard_rejected : beta dims is [%zu], which not equal to 1.", beta_dims.size());
        return false;
    }
    if (gamma_dims[0] != beta_dims[0]) {
        OPS_LOG_D(kPassName.c_str(), "guard_rejected : gamma shape and beta shape are diff.");
        return false;
    }
    return true;
}

// 属性 / 常量守卫
bool CheckValue(Matched& m)
{
    if (!CheckAxesAndKeepDims(m) || !CheckExponents(m)) {
        return false;
    }
    if (!ScalarAt(m.boundary, kInputEps, m.epsilon)) {
        OPS_LOG_D(kPassName.c_str(), "guard_rejected : fail to get value from const node of add0.");
        return false;
    }
    return !m.with_affine || CheckAffineShapes(m);
}

PatternUniqPtr MakePattern(bool with_affine, size_t config)
{
    const InputLayout layout = MakeLayout(IsBitSet(config, kBitWithPow), IsBitSet(config, kBitRsqrtIsPow), with_affine);
    auto graph_builder = es::EsGraphBuilder("layer_norm_onnx_mul_fusion_pattern");
    auto inputs = graph_builder.CreateInputs(layout.total);

    const auto& x = inputs[kInputX];
    auto mean0 = es::ReduceMean(x, inputs[kInputAxes0], true);
    // 两个独立的 Sub 是本形态的特征：sub0 供分子，sub1 供方差
    auto sub0 = es::Sub(x, mean0);
    auto sub1 = es::Sub(x, mean0);

    es::EsTensorHolder square_node = IsBitSet(config, kBitWithPow) ? es::Pow(sub1, inputs[layout.pow_exp]) :
                                                                     es::Mul(sub1, sub1);

    auto mean1 = es::ReduceMean(square_node, inputs[kInputAxes1], true);
    auto add0 = IsBitSet(config, kBitEpsFirst) ? es::Add(inputs[kInputEps], mean1) : es::Add(mean1, inputs[kInputEps]);

    es::EsTensorHolder rsqrt0 = IsBitSet(config, kBitRsqrtIsPow) ? es::Pow(add0, inputs[layout.rsqrt_exp]) :
                                                                   es::Sqrt(add0);

    auto div0 = es::RealDiv(sub0, rsqrt0);

    es::EsTensorHolder out = div0;
    if (with_affine) {
        auto mul1 = IsBitSet(config, kBitGammaFirst) ? es::Mul(inputs[layout.gamma], div0) :
                                                       es::Mul(div0, inputs[layout.gamma]);
        out = IsBitSet(config, kBitBetaFirst) ? es::Add(inputs[layout.beta], mul1) : es::Add(mul1, inputs[layout.beta]);
    }

    auto graph = graph_builder.BuildAndReset({out});
    auto pattern = std::make_unique<Pattern>(std::move(*graph));
    pattern->CaptureTensor({*mean0.GetProducer(), 0})
        .CaptureTensor({*sub0.GetProducer(), 0})
        .CaptureTensor({*sub1.GetProducer(), 0})
        .CaptureTensor({*square_node.GetProducer(), 0})
        .CaptureTensor({*mean1.GetProducer(), 0})
        .CaptureTensor({*add0.GetProducer(), 0})
        .CaptureTensor({*rsqrt0.GetProducer(), 0})
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
    const int64_t last_dim = m.input_dims.back();
    if (last_dim <= 0L) {
        OPS_LOG_E(kPassName.c_str(), "replacement_rejected : last dim of input is not positive.");
        return false;
    }
    const std::vector<int64_t> const_dims = {last_dim};
    const size_t numel = static_cast<size_t>(last_dim);
    const DataType dtype = div_out_desc.GetDataType();
    const Format format = div_in_desc.GetFormat();
    param_desc = TensorDesc(Shape(const_dims), format, dtype);
    param_desc.SetOriginShape(Shape(const_dims));
    param_desc.SetOriginFormat(format);
    if (dtype == DT_FLOAT16) {
        gamma = builder.CreateConst<uint16_t>(std::vector<uint16_t>(numel, kHalfOneBits), const_dims, DT_FLOAT16,
                                              format);
        beta = builder.CreateConst<uint16_t>(std::vector<uint16_t>(numel, 0U), const_dims, DT_FLOAT16, format);
        return true;
    }
    if (dtype == DT_FLOAT) {
        gamma = builder.CreateConst<float32_t>(std::vector<float32_t>(numel, 1.0f), const_dims, DT_FLOAT, format);
        beta = builder.CreateConst<float32_t>(std::vector<float32_t>(numel, 0.0f), const_dims, DT_FLOAT, format);
        return true;
    }
    OPS_LOG_D(kPassName.c_str(), "guard_rejected : div0 dtype is not in (float16, float32).");
    return false;
}
} // namespace

std::vector<PatternUniqPtr> LayerNormONNXMULFusionPass::Patterns()
{
    OPS_LOG_D(kPassName.c_str(), "Enter Patterns for LayerNormONNXMULFusionPass.");
    if (!IsSupportedPlatform()) {
        return {};
    }
    std::vector<PatternUniqPtr> patterns;
    patterns.reserve(kExpectedPatternNum);
    for (const bool with_affine : {true, false}) {
        for (size_t config = 0U; config < kCommConfigNum; ++config) {
            if (!with_affine && (IsBitSet(config, kBitGammaFirst) || IsBitSet(config, kBitBetaFirst))) {
                continue;
            }
            patterns.emplace_back(MakePattern(with_affine, config));
        }
    }
    return patterns;
}

bool LayerNormONNXMULFusionPass::MeetRequirements(const std::unique_ptr<MatchResult>& match_result)
{
    OPS_LOG_D(kPassName.c_str(), "guard_begin.");
    Matched m;
    if (!CollectMatched(match_result, m)) {
        return false;
    }
    if (!CheckValue(m)) {
        return false;
    }

    // 无 affine 时要自建 gamma=1 / beta=0 常量，只支持 fp16 / fp32。
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

std::unique_ptr<Graph> LayerNormONNXMULFusionPass::Replacement(const std::unique_ptr<MatchResult>& match_result)
{
    OPS_LOG_D(kPassName.c_str(), "replacement_begin.");
    Matched m;
    if (!CollectMatched(match_result, m) || !CheckValue(m)) {
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

    // begin_norm_axis 取 axes0[0] 原值（可能是 -1，不做归一化），begin_params_axis 恒为 -1。
    auto layer_norm = es::LayerNormV3(placeholders[kInputX], gamma, beta, m.axes0[0], kBeginParamsAxis, m.epsilon);

    GNode* layer_norm_node = layer_norm.y.GetProducer();
    if (layer_norm_node == nullptr) {
        OPS_LOG_E(kPassName.c_str(), "replacement_rejected : cannot get producer of LayerNormV3.");
        return nullptr;
    }
    if ((layer_norm_node->UpdateInputDesc(0, boundary_descs[kInputX]) != GRAPH_SUCCESS) ||
        (layer_norm_node->UpdateInputDesc(1, gamma_desc) != GRAPH_SUCCESS) ||
        (layer_norm_node->UpdateInputDesc(2, beta_desc) != GRAPH_SUCCESS)) {
        OPS_LOG_E(kPassName.c_str(), "replacement_rejected : cannot update input desc of LayerNormV3.");
        return nullptr;
    }

    TensorDesc y_desc;
    TensorDesc mean_desc;
    TensorDesc rstd_desc;
    const GNode& y_source = m.with_affine ? m.add1 : m.div0;
    if ((y_source.GetOutputDesc(0, y_desc) != GRAPH_SUCCESS) ||
        (m.mean0.GetOutputDesc(0, mean_desc) != GRAPH_SUCCESS) ||
        (m.mean1.GetOutputDesc(0, rstd_desc) != GRAPH_SUCCESS)) {
        OPS_LOG_E(kPassName.c_str(), "replacement_rejected : cannot get output desc from matched nodes.");
        return nullptr;
    }
    if ((layer_norm_node->UpdateOutputDesc(0, y_desc) != GRAPH_SUCCESS) ||
        (layer_norm_node->UpdateOutputDesc(1, mean_desc) != GRAPH_SUCCESS) ||
        (layer_norm_node->UpdateOutputDesc(2, rstd_desc) != GRAPH_SUCCESS)) {
        OPS_LOG_E(kPassName.c_str(), "replacement_rejected : cannot update output desc of LayerNormV3.");
        return nullptr;
    }

    OPS_LOG_D(kPassName.c_str(), "replacement_done.");
    return graph_builder.BuildAndReset({layer_norm.y});
}

REG_FUSION_PASS(LayerNormONNXMULFusionPass).Stage(CustomPassStage::kAfterInferShape);
} // namespace ops
