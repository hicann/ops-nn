/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "ge/compliant_node_builder.h"
#include "ge/es_graph_builder.h"
#include "platform/platform_info.h"
#include "platform/platform_infos_def.h"
#include "register/register_custom_pass.h"

#include "../../../op_graph/fusion_pass/layer_norm_onnx_mul_fusion_pass.h"

using namespace std;
using namespace ge;
using namespace fe;
using namespace ops;

namespace {
struct GraphConfig {
    std::vector<int64_t> x_dims{2, 3, 8};
    DataType dtype = DT_FLOAT;
    bool with_pow = false;     // false: Mul(sub1,sub1)；true: Pow(sub1,2)
    bool rsqrt_is_pow = false; // false: Sqrt(add0)；true: Pow(add0,0.5)
    bool with_affine = true;
    float epsilon = 1e-5f;
    float pow_exp = 2.0f;
    float rsqrt_exp = 0.5f;
    bool keep_dims = true;
    int64_t axis = -1;
    // gamma 写成 2 维
    bool gamma_rank2 = false;
    // beta 写成 2 维
    bool beta_rank2 = false;
    // beta 长度写成 1，与 gamma 的 last_dim 不等
    bool beta_short = false;
    // 两个 ReduceMean 共用同一个 axes Const（GE 做过 CSE 后的常态）
    bool shared_axes_const = false;
    // 只造一个 Sub、同时喂平方支路和分子——这是 LayerNormONNXFusionPass 的形态
    bool single_sub = false;
    // eps 常量造成 fp16
    bool eps_fp16 = false;
    // eps 落 Add 的 x1（输入 0）而非 x2
    bool eps_first = false;
    // gamma / beta 挂在 Mul / Add 的输入 0（而非输入 1）
    bool gamma_first = false;
    bool beta_first = false;
    // 给指定节点的 0 号输出再挂一个片段外的消费者（Relu）；空串表示不加
    std::string extra_consumer_on;
};

// float32 -> fp16 位模式
uint16_t ToFp16Bits(float value)
{
    uint32_t bits = 0U;
    static_assert(sizeof(bits) == sizeof(value), "size mismatch");
    (void)memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16U) & 0x8000U;
    int32_t exp = static_cast<int32_t>((bits >> 23U) & 0xFFU) - 127 + 15;
    uint32_t mant = bits & 0x7FFFFFU;
    if (exp >= 0x1F) {
        return static_cast<uint16_t>(sign | 0x7C00U); // inf
    }
    if (exp <= 0) { // 非规格化：eps=1e-5 就落在这里
        if (exp < -10) {
            return static_cast<uint16_t>(sign);
        }
        mant |= 0x800000U;
        const uint32_t shift = static_cast<uint32_t>(14 - exp);
        const uint32_t sub = mant >> shift;
        const uint32_t rem = mant & ((1U << shift) - 1U);
        const uint32_t round = ((rem << 1U) > (1U << shift)) ? 1U : 0U;
        return static_cast<uint16_t>(sign | (sub + round));
    }
    const uint32_t half = mant >> 13U;
    const uint32_t round = ((mant & 0x1FFFU) > 0x1000U) ? 1U : 0U;
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10U) | (half + round));
}

// fp16 位模式 -> float32，用来算出"正确解码"应得的期望值
float Fp16BitsToFloat(uint16_t bits)
{
    const uint32_t sign = (static_cast<uint32_t>(bits) & 0x8000U) << 16U;
    uint32_t exp = (static_cast<uint32_t>(bits) >> 10U) & 0x1FU;
    uint32_t mant = static_cast<uint32_t>(bits) & 0x3FFU;
    uint32_t out = 0U;
    if (exp == 0U) {
        if (mant == 0U) {
            out = sign;
        } else { // 非规格化，规格化回 float
            exp = 1U;
            while ((mant & 0x400U) == 0U) {
                mant <<= 1U;
                --exp;
            }
            mant &= 0x3FFU;
            out = sign | ((exp + 127U - 15U) << 23U) | (mant << 13U);
        }
    } else if (exp == 0x1FU) {
        out = sign | 0x7F800000U | (mant << 13U);
    } else {
        out = sign | ((exp + 127U - 15U) << 23U) | (mant << 13U);
    }
    float value = 0.0f;
    (void)memcpy(&value, &out, sizeof(value));
    return value;
}

std::vector<int64_t> MeanDims(const std::vector<int64_t>& x_dims)
{
    std::vector<int64_t> dims = x_dims;
    dims.back() = 1;
    return dims;
}

GNode MakeNode(Graph* graph, const char* op_type, const char* name,
               const std::vector<es::CompliantNodeBuilder::IrInputDef>& inputs,
               const std::vector<es::CompliantNodeBuilder::IrOutputDef>& outputs)
{
    return es::CompliantNodeBuilder(graph).OpType(op_type).Name(name).IrDefInputs(inputs).IrDefOutputs(outputs).Build();
}

GNode MakeBinary(Graph* graph, const char* op_type, const char* name)
{
    return MakeNode(graph, op_type, name,
                    {{"x1", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                     {"x2", es::CompliantNodeBuilder::kEsIrInputRequired, ""}},
                    {{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}});
}

GNode MakeUnary(Graph* graph, const char* op_type, const char* name)
{
    return MakeNode(graph, op_type, name, {{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""}},
                    {{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}});
}

void SetDesc(GNode& node, int32_t index, const std::vector<int64_t>& dims, DataType dtype)
{
    TensorDesc desc(Shape(dims), FORMAT_ND, dtype);
    desc.SetOriginShape(Shape(dims));
    desc.SetOriginFormat(FORMAT_ND);
    node.UpdateInputDesc(index, desc);
}

void SetOutDesc(GNode& node, int32_t index, const std::vector<int64_t>& dims, DataType dtype)
{
    TensorDesc desc(Shape(dims), FORMAT_ND, dtype);
    desc.SetOriginShape(Shape(dims));
    desc.SetOriginFormat(FORMAT_ND);
    node.UpdateOutputDesc(index, desc);
}

GNode MakeMean(Graph* graph, es::EsGraphBuilder& graph_builder, const GraphConfig& cfg, const char* name,
               const GNode& input, const std::vector<int64_t>& in_dims, const std::vector<int64_t>& out_dims,
               GNode* shared_axes = nullptr)
{
    GNode mean = MakeNode(graph, "ReduceMean", name,
                          {{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                           {"axes", es::CompliantNodeBuilder::kEsIrInputRequired, ""}},
                          {{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}});
    bool keep_dims_attr = cfg.keep_dims;
    mean.SetAttr("keep_dims", keep_dims_attr);
    GNode src = input;
    es::AddEdgeAndUpdatePeerDesc(*graph, src, 0, mean, 0);
    // shared_axes 非空时复用调用方给的那个 Const 节点，否则各自新建一个
    if (shared_axes != nullptr) {
        es::AddEdgeAndUpdatePeerDesc(*graph, *shared_axes, 0, mean, 1);
    } else {
        auto axes_const = graph_builder.CreateConst<int64_t>({cfg.axis}, {1}, DT_INT64, FORMAT_ND);
        es::AddEdgeAndUpdatePeerDesc(*graph, *axes_const.GetProducer(), 0, mean, 1);
    }
    SetDesc(mean, 1, {1}, DT_INT64);
    SetDesc(mean, 0, in_dims, cfg.dtype);
    SetOutDesc(mean, 0, out_dims, cfg.dtype);
    return mean;
}

GraphPtr BuildOnnxMulLayerNormGraph(const std::string& graph_name, const GraphConfig& cfg)
{
    es::EsGraphBuilder graph_builder(graph_name.c_str());
    auto* graph = graph_builder.GetCGraphBuilder()->GetGraph();

    const std::vector<int64_t>& x_dims = cfg.x_dims;
    const std::vector<int64_t> mean_dims = MeanDims(x_dims);
    const int64_t last_dim = x_dims.back();

    auto x = graph_builder.CreateInput(0, "x", cfg.dtype, FORMAT_ND, x_dims);
    GNode x_node = *x.GetProducer();

    // shared_axes_const 时先造一个 Const，两个 ReduceMean 共用
    auto shared_axes_holder = graph_builder.CreateConst<int64_t>({cfg.axis}, {1}, DT_INT64, FORMAT_ND);
    GNode* shared_axes = cfg.shared_axes_const ? shared_axes_holder.GetProducer() : nullptr;
    GNode mean0 = MakeMean(graph, graph_builder, cfg, "mean0", x_node, x_dims, mean_dims, shared_axes);

    GNode sub0 = MakeBinary(graph, "Sub", "sub0");
    es::AddEdgeAndUpdatePeerDesc(*graph, x_node, 0, sub0, 0);
    es::AddEdgeAndUpdatePeerDesc(*graph, mean0, 0, sub0, 1);
    SetDesc(sub0, 0, x_dims, cfg.dtype);
    SetDesc(sub0, 1, mean_dims, cfg.dtype);
    SetOutDesc(sub0, 0, x_dims, cfg.dtype);

    // single_sub 时不另建 sub1，让平方支路直接复用 sub0——构造出规则一的形态
    GNode sub1 = sub0;
    if (!cfg.single_sub) {
        sub1 = MakeBinary(graph, "Sub", "sub1");
        es::AddEdgeAndUpdatePeerDesc(*graph, x_node, 0, sub1, 0);
        es::AddEdgeAndUpdatePeerDesc(*graph, mean0, 0, sub1, 1);
        SetDesc(sub1, 0, x_dims, cfg.dtype);
        SetDesc(sub1, 1, mean_dims, cfg.dtype);
        SetOutDesc(sub1, 0, x_dims, cfg.dtype);
    }

    // 平方支路：Mul(sub1, sub1) 或 Pow(sub1, 2)
    GNode square;
    if (cfg.with_pow) {
        square = MakeBinary(graph, "Pow", "pow0");
        auto exp_const = graph_builder.CreateConst<float>({cfg.pow_exp}, {1}, DT_FLOAT, FORMAT_ND);
        es::AddEdgeAndUpdatePeerDesc(*graph, sub1, 0, square, 0);
        es::AddEdgeAndUpdatePeerDesc(*graph, *exp_const.GetProducer(), 0, square, 1);
        SetDesc(square, 1, {1}, DT_FLOAT);
    } else {
        square = MakeBinary(graph, "Mul", "mul0");
        es::AddEdgeAndUpdatePeerDesc(*graph, sub1, 0, square, 0);
        es::AddEdgeAndUpdatePeerDesc(*graph, sub1, 0, square, 1);
        SetDesc(square, 1, x_dims, cfg.dtype);
    }
    SetDesc(square, 0, x_dims, cfg.dtype);
    SetOutDesc(square, 0, x_dims, cfg.dtype);

    GNode mean1 = MakeMean(graph, graph_builder, cfg, "mean1", square, x_dims, mean_dims, shared_axes);

    GNode add0 = MakeBinary(graph, "Add", "add0");
    const DataType eps_dtype = cfg.eps_fp16 ? DT_FLOAT16 : DT_FLOAT;
    es::EsTensorHolder eps_const = cfg.eps_fp16 ?
                                       graph_builder.CreateConst<uint16_t>({ToFp16Bits(cfg.epsilon)}, {1}, DT_FLOAT16,
                                                                           FORMAT_ND) :
                                       graph_builder.CreateConst<float>({cfg.epsilon}, {1}, DT_FLOAT, FORMAT_ND);
    // eps_first：常量落 x1(输入 0)、mean1 落 x2(输入 1)
    const int32_t eps_port = cfg.eps_first ? 0 : 1;
    const int32_t mean_port = cfg.eps_first ? 1 : 0;
    es::AddEdgeAndUpdatePeerDesc(*graph, mean1, 0, add0, mean_port);
    es::AddEdgeAndUpdatePeerDesc(*graph, *eps_const.GetProducer(), 0, add0, eps_port);
    // ONNX 的 Add 要求两输入同类型，故两侧 desc 一起跟随 eps_dtype
    SetDesc(add0, mean_port, mean_dims, cfg.eps_fp16 ? DT_FLOAT16 : cfg.dtype);
    SetDesc(add0, eps_port, {1}, eps_dtype);
    SetOutDesc(add0, 0, mean_dims, cfg.dtype);

    // rsqrt 位：Sqrt(add0) 或 Pow(add0, 0.5)
    GNode rsqrt0;
    if (cfg.rsqrt_is_pow) {
        rsqrt0 = MakeBinary(graph, "Pow", "rsqrt0");
        auto rsqrt_const = graph_builder.CreateConst<float>({cfg.rsqrt_exp}, {1}, DT_FLOAT, FORMAT_ND);
        es::AddEdgeAndUpdatePeerDesc(*graph, add0, 0, rsqrt0, 0);
        es::AddEdgeAndUpdatePeerDesc(*graph, *rsqrt_const.GetProducer(), 0, rsqrt0, 1);
        SetDesc(rsqrt0, 1, {1}, DT_FLOAT);
    } else {
        rsqrt0 = MakeUnary(graph, "Sqrt", "rsqrt0");
        es::AddEdgeAndUpdatePeerDesc(*graph, add0, 0, rsqrt0, 0);
    }
    SetDesc(rsqrt0, 0, mean_dims, cfg.dtype);
    SetOutDesc(rsqrt0, 0, mean_dims, cfg.dtype);

    GNode div0 = MakeBinary(graph, "RealDiv", "div0");
    es::AddEdgeAndUpdatePeerDesc(*graph, sub0, 0, div0, 0);
    es::AddEdgeAndUpdatePeerDesc(*graph, rsqrt0, 0, div0, 1);
    SetDesc(div0, 0, x_dims, cfg.dtype);
    SetDesc(div0, 1, mean_dims, cfg.dtype);
    SetOutDesc(div0, 0, x_dims, cfg.dtype);

    GNode out_node = div0;
    GNode mul1; // 提到 if 外，便于给它挂片段外的额外消费者（出度守卫用例）
    if (cfg.with_affine) {
        const std::vector<int64_t> gamma_dims = cfg.gamma_rank2 ? std::vector<int64_t>{1, last_dim} :
                                                                  std::vector<int64_t>{last_dim};
        mul1 = MakeBinary(graph, "Mul", "mul1");
        GNode add1 = MakeBinary(graph, "Add", "add1");
        auto gamma_const = graph_builder.CreateConst<float>(std::vector<float>(last_dim, 1.0f), gamma_dims, DT_FLOAT,
                                                            FORMAT_ND);
        const std::vector<int64_t> beta_dims = cfg.beta_rank2 ? std::vector<int64_t>{1, last_dim} :
                                               cfg.beta_short ? std::vector<int64_t>{1} :
                                                                std::vector<int64_t>{last_dim};
        const size_t beta_len = static_cast<size_t>(cfg.beta_short ? 1 : last_dim);
        auto beta_const = graph_builder.CreateConst<float>(std::vector<float>(beta_len, 0.0f), beta_dims, DT_FLOAT,
                                                           FORMAT_ND);
        const int32_t gamma_port = cfg.gamma_first ? 0 : 1;
        const int32_t beta_port = cfg.beta_first ? 0 : 1;
        es::AddEdgeAndUpdatePeerDesc(*graph, div0, 0, mul1, 1 - gamma_port);
        es::AddEdgeAndUpdatePeerDesc(*graph, *gamma_const.GetProducer(), 0, mul1, gamma_port);
        es::AddEdgeAndUpdatePeerDesc(*graph, mul1, 0, add1, 1 - beta_port);
        es::AddEdgeAndUpdatePeerDesc(*graph, *beta_const.GetProducer(), 0, add1, beta_port);
        SetDesc(mul1, 1 - gamma_port, x_dims, cfg.dtype);
        SetDesc(mul1, gamma_port, gamma_dims, DT_FLOAT);
        SetOutDesc(mul1, 0, x_dims, cfg.dtype);
        SetDesc(add1, 1 - beta_port, x_dims, cfg.dtype);
        SetDesc(add1, beta_port, beta_dims, DT_FLOAT);
        SetOutDesc(add1, 0, x_dims, cfg.dtype);
        out_node = add1;
    }

    // 额外消费者：挂在指定节点的 0 号输出上，并作为第二个图输出，避免被当成死代码。
    GNode extra_consumer;
    bool has_extra = false;
    if (!cfg.extra_consumer_on.empty()) {
        const std::map<std::string, std::pair<GNode*, std::vector<int64_t>>> taps = {
            {"mean0", {&mean0, mean_dims}},   {"sub0", {&sub0, x_dims}},      {"sub1", {&sub1, x_dims}},
            {"square", {&square, x_dims}},    {"mean1", {&mean1, mean_dims}}, {"add0", {&add0, mean_dims}},
            {"rsqrt0", {&rsqrt0, mean_dims}}, {"div0", {&div0, x_dims}},      {"mul1", {&mul1, x_dims}},
        };
        const auto it = taps.find(cfg.extra_consumer_on);
        if ((it != taps.end()) && (it->second.first != nullptr)) {
            extra_consumer = MakeUnary(graph, "Relu", "extra_consumer");
            es::AddEdgeAndUpdatePeerDesc(*graph, *(it->second.first), 0, extra_consumer, 0);
            SetDesc(extra_consumer, 0, it->second.second, cfg.dtype);
            SetOutDesc(extra_consumer, 0, it->second.second, cfg.dtype);
            has_extra = true;
        }
    }

    es::EsGraphBuilder::SetOutput(x, 0);
    std::unique_ptr<Graph> graph_unique = graph_builder.BuildAndReset();
    std::vector<std::pair<GNode, int32_t>> graph_outputs;
    graph_outputs.emplace_back(out_node, 0);
    if (has_extra) {
        graph_outputs.emplace_back(extra_consumer, 0);
    }
    graph_unique->SetOutputs(graph_outputs);
    return GraphPtr(std::move(graph_unique));
}

int CountType(const GraphPtr& graph, const std::string& op_type)
{
    int count = 0;
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        if (node.GetType(type) == GRAPH_SUCCESS && type.GetString() != nullptr &&
            std::string(type.GetString()) == op_type) {
            ++count;
        }
    }
    return count;
}

// 取融合后目标算子的 epsilon 属性。找不到该算子返回 false。
bool GetFusedEpsilon(const GraphPtr& graph, const std::string& op_type, float& epsilon)
{
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        if (node.GetType(type) != GRAPH_SUCCESS || type.GetString() == nullptr) {
            continue;
        }
        if (std::string(type.GetString()) != op_type) {
            continue;
        }
        return node.GetAttr("epsilon", epsilon) == GRAPH_SUCCESS;
    }
    return false;
}

Status RunPass(GraphPtr& graph)
{
    CustomPassContext pass_context;
    LayerNormONNXMULFusionPass pass;
    return pass.Run(graph, pass_context);
}
} // namespace

class LayerNormONNXMULFusionPassTest : public testing::Test {
protected:
    void SetUp() override { SetPlatform(3510); }

    static void SetPlatform(int32_t npu_arch)
    {
        const std::string soc = "Ascend950";
        fe::PlatformInfo platform_info;
        fe::OptionalInfo optional_info;
        platform_info.soc_info.ai_core_cnt = 64;
        platform_info.str_info.short_soc_version = soc;
        optional_info.soc_version = soc;
        fe::PlatformInfoManager::Instance().platform_info_map_[soc] = platform_info;
        fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optional_info);

        fe::PlatFormInfos platform_infos;
        (void)platform_infos.Init();
        std::map<std::string, std::string> version_res;
        version_res["NpuArch"] = std::to_string(npu_arch);
        version_res["Short_SoC_version"] = soc;
        platform_infos.SetPlatformRes("version", version_res);
        fe::OptionalInfos optional_infos;
        (void)optional_infos.Init();
        optional_infos.SetSocVersion(soc);
        fe::PlatformInfoManager::Instance().platform_infos_map_[soc] = platform_infos;
        fe::PlatformInfoManager::Instance().SetOptionalCompilationInfo(optional_infos);
    }
};

// 主形态：Mul 平方支路 + Sqrt + affine，完整融合
TEST_F(LayerNormONNXMULFusionPassTest, fusion_pattern1_mul_sqrt_with_affine)
{
    GraphConfig cfg;
    GraphPtr graph = BuildOnnxMulLayerNormGraph("onnx_mul_p1", cfg);
    EXPECT_EQ(RunPass(graph), SUCCESS);
    EXPECT_EQ(CountType(graph, "LayerNormV3"), 1);
    // affine 必须被**吸收进算子**（见规则一同名断言的说明）
    EXPECT_EQ(CountType(graph, "Mul"), 0);
    EXPECT_EQ(CountType(graph, "Add"), 0);
    EXPECT_EQ(CountType(graph, "RealDiv"), 0);
    EXPECT_EQ(CountType(graph, "Sqrt"), 0);
}

// 无 affine：规则自建 gamma=1 / beta=0 常量
TEST_F(LayerNormONNXMULFusionPassTest, fusion_pattern2_without_affine)
{
    GraphConfig cfg;
    cfg.with_affine = false;
    GraphPtr graph = BuildOnnxMulLayerNormGraph("onnx_mul_p2", cfg);
    EXPECT_EQ(RunPass(graph), SUCCESS);
    EXPECT_EQ(CountType(graph, "LayerNormV3"), 1);
}

// 平方支路走 Pow 而非 Mul
TEST_F(LayerNormONNXMULFusionPassTest, fusion_pattern3_pow_with_affine)
{
    GraphConfig cfg;
    cfg.with_pow = true;
    GraphPtr graph = BuildOnnxMulLayerNormGraph("onnx_mul_p3", cfg);
    EXPECT_EQ(RunPass(graph), SUCCESS);
    EXPECT_EQ(CountType(graph, "LayerNormV3"), 1);
    // affine 必须被**吸收进算子**（见规则一同名断言的说明）
    EXPECT_EQ(CountType(graph, "Mul"), 0);
    EXPECT_EQ(CountType(graph, "Add"), 0);
}

// 开方支路走 Pow(x, 0.5) 而非 Sqrt
TEST_F(LayerNormONNXMULFusionPassTest, fusion_rsqrt_as_pow_half)
{
    GraphConfig cfg;
    cfg.rsqrt_is_pow = true;
    GraphPtr graph = BuildOnnxMulLayerNormGraph("onnx_mul_rsqrt_pow", cfg);
    EXPECT_EQ(RunPass(graph), SUCCESS);
    EXPECT_EQ(CountType(graph, "LayerNormV3"), 1);
    // affine 必须被**吸收进算子**（见规则一同名断言的说明）
    EXPECT_EQ(CountType(graph, "Mul"), 0);
    EXPECT_EQ(CountType(graph, "Add"), 0);
}

// 两个 ReduceMean 共用一个 axes Const：边界按 pattern 输入槽组织、不按 tensor 去重
TEST_F(LayerNormONNXMULFusionPassTest, fusion_shared_axes_const)
{
    GraphConfig cfg;
    cfg.shared_axes_const = true;
    GraphPtr graph = BuildOnnxMulLayerNormGraph("onnx_mul_shared_axes", cfg);
    EXPECT_EQ(RunPass(graph), SUCCESS);
    EXPECT_EQ(CountType(graph, "LayerNormV3"), 1);
    // affine 必须被**吸收进算子**（见规则一同名断言的说明）
    EXPECT_EQ(CountType(graph, "Mul"), 0);
    EXPECT_EQ(CountType(graph, "Add"), 0);
}

// 单 Sub 形态是规则一的图：sub0 与 sub1 必须是不同节点，否则会把对方的图抢去融合
TEST_F(LayerNormONNXMULFusionPassTest, not_changed_on_single_sub_shape)
{
    GraphConfig cfg;
    cfg.single_sub = true;
    cfg.with_pow = true; // 规则一的平方支路是 Pow/Square，不是 Mul(sub,sub)
    GraphPtr graph = BuildOnnxMulLayerNormGraph("onnx_mul_single_sub", cfg);
    EXPECT_EQ(RunPass(graph), GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountType(graph, "LayerNormV3"), 0);
    // 原图必须分毫未动
    EXPECT_EQ(CountType(graph, "Sub"), 1);
    EXPECT_EQ(CountType(graph, "RealDiv"), 1);
}

// 平台 NpuArch 不在支持列表
TEST_F(LayerNormONNXMULFusionPassTest, not_changed_on_unsupported_platform)
{
    SetPlatform(3010);
    GraphConfig cfg;
    GraphPtr graph = BuildOnnxMulLayerNormGraph("onnx_mul_bad_arch", cfg);
    EXPECT_EQ(RunPass(graph), GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountType(graph, "LayerNormV3"), 0);
    EXPECT_EQ(CountType(graph, "RealDiv"), 1);
}

// ReduceMean 的 keep_dims 不为 true
TEST_F(LayerNormONNXMULFusionPassTest, not_changed_when_keep_dims_false)
{
    GraphConfig cfg;
    cfg.keep_dims = false;
    GraphPtr graph = BuildOnnxMulLayerNormGraph("onnx_mul_keepdims", cfg);
    EXPECT_EQ(RunPass(graph), GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountType(graph, "LayerNormV3"), 0);
}

// 开方支路的 Pow 指数不是 0.5
TEST_F(LayerNormONNXMULFusionPassTest, not_changed_when_rsqrt_pow_exp_not_half)
{
    GraphConfig cfg;
    cfg.rsqrt_is_pow = true;
    cfg.rsqrt_exp = 0.25f;
    GraphPtr graph = BuildOnnxMulLayerNormGraph("onnx_mul_rsqrt_bad", cfg);
    EXPECT_EQ(RunPass(graph), GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountType(graph, "LayerNormV3"), 0);
}

// 平方支路的 Pow 指数不是 2
TEST_F(LayerNormONNXMULFusionPassTest, not_changed_when_pow_exp_not_two)
{
    GraphConfig cfg;
    cfg.with_pow = true;
    cfg.pow_exp = 3.0f;
    GraphPtr graph = BuildOnnxMulLayerNormGraph("onnx_mul_pow_bad", cfg);
    EXPECT_EQ(RunPass(graph), GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountType(graph, "LayerNormV3"), 0);
}

// 片段内部节点多一个片段外消费者：matcher 自包含约束使其不融合
TEST_F(LayerNormONNXMULFusionPassTest, not_changed_when_sub1_has_extra_consumer)
{
    GraphConfig cfg;
    cfg.extra_consumer_on = "sub1";
    GraphPtr graph = BuildOnnxMulLayerNormGraph("onnx_mul_ln_extra_sub1", cfg);
    EXPECT_EQ(RunPass(graph), GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountType(graph, "LayerNormV3"), 0);
}

// gamma 挂在 Mul 的输入 0（可交换轴）
TEST_F(LayerNormONNXMULFusionPassTest, fusion_when_gamma_on_first_input)
{
    GraphConfig cfg;
    cfg.gamma_first = true;
    GraphPtr graph = BuildOnnxMulLayerNormGraph("onnx_mul_ln_gamma_first", cfg);
    EXPECT_EQ(RunPass(graph), SUCCESS);
    EXPECT_EQ(CountType(graph, "LayerNormV3"), 1);
    EXPECT_EQ(CountType(graph, "Mul"), 0);
    EXPECT_EQ(CountType(graph, "Add"), 0);
}

// beta 挂在 Add 的输入 0（可交换轴）
TEST_F(LayerNormONNXMULFusionPassTest, fusion_when_beta_on_first_input)
{
    GraphConfig cfg;
    cfg.beta_first = true;
    GraphPtr graph = BuildOnnxMulLayerNormGraph("onnx_mul_ln_beta_first", cfg);
    EXPECT_EQ(RunPass(graph), SUCCESS);
    EXPECT_EQ(CountType(graph, "LayerNormV3"), 1);
    EXPECT_EQ(CountType(graph, "Mul"), 0);
    EXPECT_EQ(CountType(graph, "Add"), 0);
}

// 无 affine 时 div0 是片段出口，出口允许有片段外消费者，照常融合
TEST_F(LayerNormONNXMULFusionPassTest, fusion_when_div0_has_extra_consumer_without_affine)
{
    GraphConfig cfg;
    cfg.with_affine = false;
    cfg.extra_consumer_on = "div0";
    GraphPtr graph = BuildOnnxMulLayerNormGraph("onnx_mul_ln_extra_div0_noaffine", cfg);
    EXPECT_EQ(RunPass(graph), SUCCESS);
    EXPECT_EQ(CountType(graph, "LayerNormV3"), 1);
}

// 归一化轴不是最后一维
TEST_F(LayerNormONNXMULFusionPassTest, not_changed_when_axis_is_not_last)
{
    GraphConfig cfg;
    cfg.axis = 1;
    GraphPtr graph = BuildOnnxMulLayerNormGraph("onnx_mul_axis1", cfg);
    EXPECT_EQ(RunPass(graph), GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountType(graph, "LayerNormV3"), 0);
}

// fp16 的 eps 常量按 fp16_t 解码，epsilon 属性值正确（源缺陷 A7 的修复点）
TEST_F(LayerNormONNXMULFusionPassTest, fusion_with_fp16_epsilon_value_is_correct)
{
    GraphConfig cfg;
    cfg.epsilon = 0.03125f;
    cfg.eps_fp16 = true;
    GraphPtr graph = BuildOnnxMulLayerNormGraph("onnx_mul_eps_fp16", cfg);
    EXPECT_EQ(RunPass(graph), SUCCESS);
    EXPECT_EQ(CountType(graph, "LayerNormV3"), 1);

    float epsilon = -1.0f;
    ASSERT_TRUE(GetFusedEpsilon(graph, "LayerNormV3", epsilon));
    // 期望值 = fp16 能表示的 1e-5（非规格化），而不是按 float* 错读出的位模式垃圾
    const float expected = Fp16BitsToFloat(ToFp16Bits(cfg.epsilon));
    EXPECT_FLOAT_EQ(expected, 0.03125f); // fp16 精确表示，编码/解码辅助函数自检
    EXPECT_FLOAT_EQ(epsilon, expected);
}

// eps 挂在 Add 的输入 0（eps 位轴）
TEST_F(LayerNormONNXMULFusionPassTest, fusion_when_eps_on_first_input)
{
    GraphConfig cfg;
    cfg.eps_first = true;
    cfg.epsilon = 0.03125f;
    GraphPtr graph = BuildOnnxMulLayerNormGraph("onnx_mul_eps_first", cfg);
    EXPECT_EQ(RunPass(graph), SUCCESS);
    EXPECT_EQ(CountType(graph, "LayerNormV3"), 1);
    EXPECT_EQ(CountType(graph, "Mul"), 0);
    EXPECT_EQ(CountType(graph, "Add"), 0);
    float epsilon = -1.0f;
    ASSERT_TRUE(GetFusedEpsilon(graph, "LayerNormV3", epsilon));
    EXPECT_FLOAT_EQ(epsilon, 0.03125f);
}

// 带 affine 的图必须完整融合：断言 Mul/Add 已被吸收，逐个变体压一遍
TEST_F(LayerNormONNXMULFusionPassTest, affine_absorbed_for_all_variants)
{
    for (const bool with_pow : {false, true}) {
        for (const bool rsqrt_is_pow : {false, true}) {
            GraphConfig cfg;
            cfg.with_pow = with_pow;
            cfg.rsqrt_is_pow = rsqrt_is_pow;
            const std::string name = std::string("onnx_mul_affine_") + (with_pow ? "pow" : "mul") + "_" +
                                     (rsqrt_is_pow ? "powsqrt" : "sqrt");
            GraphPtr graph = BuildOnnxMulLayerNormGraph(name.c_str(), cfg);
            EXPECT_EQ(RunPass(graph), SUCCESS) << name;
            EXPECT_EQ(CountType(graph, "LayerNormV3"), 1) << name;
            EXPECT_EQ(CountType(graph, "Mul"), 0) << name;
            EXPECT_EQ(CountType(graph, "Add"), 0) << name;
            EXPECT_EQ(CountType(graph, "Sub"), 0) << name;
            EXPECT_EQ(CountType(graph, "RealDiv"), 0) << name;
        }
    }
}

// 无 affine 且 dtype 非 fp16/fp32：必须是不融合，而不是整图编译失败
TEST_F(LayerNormONNXMULFusionPassTest, not_changed_when_no_affine_and_dtype_unsupported)
{
    GraphConfig cfg;
    cfg.with_affine = false;
    cfg.dtype = DT_BF16;
    GraphPtr graph = BuildOnnxMulLayerNormGraph("onnx_mul_noaffine_bf16", cfg);
    EXPECT_EQ(RunPass(graph), GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountType(graph, "LayerNormV3"), 0);
    EXPECT_EQ(CountType(graph, "RealDiv"), 1);
}

// 有 affine 时不走自建常量，dtype 门禁不应生效
TEST_F(LayerNormONNXMULFusionPassTest, fusion_with_affine_is_not_blocked_by_const_dtype_guard)
{
    GraphConfig cfg;
    cfg.with_affine = true;
    GraphPtr graph = BuildOnnxMulLayerNormGraph("onnx_mul_affine_dtype_guard", cfg);
    EXPECT_EQ(RunPass(graph), SUCCESS);
    EXPECT_EQ(CountType(graph, "LayerNormV3"), 1);
}
