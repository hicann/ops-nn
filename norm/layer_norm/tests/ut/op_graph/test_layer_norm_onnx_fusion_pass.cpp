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

#include "../../../op_graph/fusion_pass/layer_norm_onnx_fusion_pass.h"

using namespace std;
using namespace ge;
using namespace fe;
using namespace ops;

namespace {
// 测试图的形态开关
enum class ClipForm { kNone, kMaxThenMin, kMinThenMax };

struct GraphConfig {
    std::vector<int64_t> x_dims{2, 3, 8};
    DataType dtype = DT_FLOAT;
    std::string pow_type = "Pow";     // 或 Square
    std::string div_type = "RealDiv"; // 或 Div
    bool with_cast = false;
    bool with_affine = true;
    ClipForm clip = ClipForm::kNone;
    float epsilon = 1e-5f;
    float pow_exp = 2.0f;
    bool keep_dims = true;
    int64_t axis = -1;
    // gamma 常量 shape 故意写错
    bool broken_gamma_shape = false;
    // beta 常量 shape 故意写错
    bool broken_beta_shape = false;
    // 两个 ReduceMean 共用同一个 axes Const（GE 做过 CSE 后的常态）
    bool shared_axes_const = false;
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

// float32 -> fp16 位模式（eps 取值都在 fp16 可表示范围内）
uint16_t ToFp16Bits(float value)
{
    uint32_t bits = 0U;
    (void)memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16U) & 0x8000U;
    int32_t exp = static_cast<int32_t>((bits >> 23U) & 0xFFU) - 127 + 15;
    uint32_t mant = bits & 0x7FFFFFU;
    if (exp >= 0x1F) {
        return static_cast<uint16_t>(sign | 0x7C00U);
    }
    if (exp <= 0) {
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

float Fp16BitsToFloat(uint16_t bits)
{
    const uint32_t sign = (static_cast<uint32_t>(bits) & 0x8000U) << 16U;
    uint32_t exp = (static_cast<uint32_t>(bits) >> 10U) & 0x1FU;
    uint32_t mant = static_cast<uint32_t>(bits) & 0x3FFU;
    uint32_t out = 0U;
    if (exp == 0U) {
        if (mant == 0U) {
            out = sign;
        } else {
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

void SetDesc(GNode& node, int32_t in_index, const std::vector<int64_t>& dims, DataType dtype)
{
    TensorDesc desc(Shape(dims), FORMAT_ND, dtype);
    desc.SetOriginShape(Shape(dims));
    desc.SetOriginFormat(FORMAT_ND);
    node.UpdateInputDesc(in_index, desc);
}

void SetOutDesc(GNode& node, int32_t out_index, const std::vector<int64_t>& dims, DataType dtype)
{
    TensorDesc desc(Shape(dims), FORMAT_ND, dtype);
    desc.SetOriginShape(Shape(dims));
    desc.SetOriginFormat(FORMAT_ND);
    node.UpdateOutputDesc(out_index, desc);
}

// 按配置搭出一张待融合的 ONNX LayerNorm 展开图。
GraphPtr BuildOnnxLayerNormGraph(const std::string& graph_name, const GraphConfig& cfg)
{
    es::EsGraphBuilder graph_builder(graph_name.c_str());
    auto* graph = graph_builder.GetCGraphBuilder()->GetGraph();

    const std::vector<int64_t>& x_dims = cfg.x_dims;
    const std::vector<int64_t> mean_dims = MeanDims(x_dims);
    const int64_t last_dim = x_dims.back();
    const int64_t rank = static_cast<int64_t>(x_dims.size());
    const int64_t norm_axis = (cfg.axis < 0) ? (cfg.axis + rank) : cfg.axis;

    auto x = graph_builder.CreateInput(0, "x", cfg.dtype, FORMAT_ND, x_dims);
    auto* x_node = x.GetProducer();

    auto axes_const = graph_builder.CreateConst<int64_t>({cfg.axis}, {1}, DT_INT64, FORMAT_ND);
    auto axes_const1 = graph_builder.CreateConst<int64_t>({cfg.axis}, {1}, DT_INT64, FORMAT_ND);

    // reduce_mean0
    GNode rm0 = MakeNode(graph, "ReduceMean", "rm0",
                         {{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                          {"axes", es::CompliantNodeBuilder::kEsIrInputRequired, ""}},
                         {{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}});
    bool keep_dims_attr = cfg.keep_dims;
    rm0.SetAttr("keep_dims", keep_dims_attr);
    es::AddEdgeAndUpdatePeerDesc(*graph, *x_node, 0, rm0, 0);
    es::AddEdgeAndUpdatePeerDesc(*graph, *axes_const.GetProducer(), 0, rm0, 1);
    SetDesc(rm0, 1, {1}, DT_INT64);
    SetDesc(rm0, 0, x_dims, cfg.dtype);
    SetOutDesc(rm0, 0, mean_dims, cfg.dtype);

    // sub0 = Sub(x, mean0)
    GNode sub0 = MakeBinary(graph, "Sub", "sub0");
    es::AddEdgeAndUpdatePeerDesc(*graph, *x_node, 0, sub0, 0);
    es::AddEdgeAndUpdatePeerDesc(*graph, rm0, 0, sub0, 1);
    SetDesc(sub0, 0, x_dims, cfg.dtype);
    SetDesc(sub0, 1, mean_dims, cfg.dtype);
    SetOutDesc(sub0, 0, x_dims, cfg.dtype);

    GNode pow_input = sub0;
    GNode cast0;
    if (cfg.with_cast) {
        cast0 = MakeUnary(graph, "Cast", "cast0");
        es::AddEdgeAndUpdatePeerDesc(*graph, sub0, 0, cast0, 0);
        SetDesc(cast0, 0, x_dims, cfg.dtype);
        SetOutDesc(cast0, 0, x_dims, cfg.dtype);
        pow_input = cast0;
    }

    // pow0 = Pow(., 2) 或 Square(.)
    GNode pow0;
    if (cfg.pow_type == "Pow") {
        pow0 = MakeBinary(graph, "Pow", "pow0");
        auto exp_const = graph_builder.CreateConst<float>({cfg.pow_exp}, {1}, DT_FLOAT, FORMAT_ND);
        es::AddEdgeAndUpdatePeerDesc(*graph, pow_input, 0, pow0, 0);
        es::AddEdgeAndUpdatePeerDesc(*graph, *exp_const.GetProducer(), 0, pow0, 1);
        SetDesc(pow0, 1, {1}, DT_FLOAT);
    } else {
        pow0 = MakeUnary(graph, "Square", "pow0");
        es::AddEdgeAndUpdatePeerDesc(*graph, pow_input, 0, pow0, 0);
    }
    SetDesc(pow0, 0, x_dims, cfg.dtype);
    SetOutDesc(pow0, 0, x_dims, cfg.dtype);

    // reduce_mean1
    GNode rm1 = MakeNode(graph, "ReduceMean", "rm1",
                         {{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                          {"axes", es::CompliantNodeBuilder::kEsIrInputRequired, ""}},
                         {{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}});
    bool keep_dims_attr1 = cfg.keep_dims;
    rm1.SetAttr("keep_dims", keep_dims_attr1);
    es::AddEdgeAndUpdatePeerDesc(*graph, pow0, 0, rm1, 0);
    // shared_axes_const 时两个 ReduceMean 复用同一个 Const 节点
    es::AddEdgeAndUpdatePeerDesc(*graph, cfg.shared_axes_const ? *axes_const.GetProducer() : *axes_const1.GetProducer(),
                                 0, rm1, 1);
    SetDesc(rm1, 1, {1}, DT_INT64);
    SetDesc(rm1, 0, x_dims, cfg.dtype);
    SetOutDesc(rm1, 0, mean_dims, cfg.dtype);

    // add0 = Add(mean1, eps)，eps_first 时两个操作数互换
    GNode add0 = MakeBinary(graph, "Add", "add0");
    const DataType eps_dtype = cfg.eps_fp16 ? DT_FLOAT16 : DT_FLOAT;
    es::EsTensorHolder eps_const = cfg.eps_fp16 ?
                                       graph_builder.CreateConst<uint16_t>({ToFp16Bits(cfg.epsilon)}, {1}, DT_FLOAT16,
                                                                           FORMAT_ND) :
                                       graph_builder.CreateConst<float>({cfg.epsilon}, {1}, DT_FLOAT, FORMAT_ND);
    const int32_t eps_port = cfg.eps_first ? 0 : 1;
    const int32_t mean_port = cfg.eps_first ? 1 : 0;
    es::AddEdgeAndUpdatePeerDesc(*graph, rm1, 0, add0, mean_port);
    es::AddEdgeAndUpdatePeerDesc(*graph, *eps_const.GetProducer(), 0, add0, eps_port);
    // ONNX 的 Add 要求两输入同类型，故两侧 desc 一起跟随 eps_dtype
    SetDesc(add0, mean_port, mean_dims, cfg.eps_fp16 ? DT_FLOAT16 : cfg.dtype);
    SetDesc(add0, eps_port, {1}, eps_dtype);
    SetOutDesc(add0, 0, mean_dims, cfg.dtype);

    // 可选 clip：Maximum/Minimum，max 常量为 0、min 常量为 INFINITY
    GNode sqrt_input = add0;
    if (cfg.clip != ClipForm::kNone) {
        GNode max_node = MakeBinary(graph, "Maximum", "max0");
        GNode min_node = MakeBinary(graph, "Minimum", "min0");
        auto max_const = graph_builder.CreateConst<float>({0.0f}, {1}, DT_FLOAT, FORMAT_ND);
        auto min_const = graph_builder.CreateConst<float>({INFINITY}, {1}, DT_FLOAT, FORMAT_ND);
        if (cfg.clip == ClipForm::kMaxThenMin) {
            es::AddEdgeAndUpdatePeerDesc(*graph, add0, 0, max_node, 0);
            es::AddEdgeAndUpdatePeerDesc(*graph, *max_const.GetProducer(), 0, max_node, 1);
            es::AddEdgeAndUpdatePeerDesc(*graph, max_node, 0, min_node, 0);
            es::AddEdgeAndUpdatePeerDesc(*graph, *min_const.GetProducer(), 0, min_node, 1);
            sqrt_input = min_node;
        } else {
            es::AddEdgeAndUpdatePeerDesc(*graph, add0, 0, min_node, 0);
            es::AddEdgeAndUpdatePeerDesc(*graph, *min_const.GetProducer(), 0, min_node, 1);
            es::AddEdgeAndUpdatePeerDesc(*graph, min_node, 0, max_node, 0);
            es::AddEdgeAndUpdatePeerDesc(*graph, *max_const.GetProducer(), 0, max_node, 1);
            sqrt_input = max_node;
        }
        SetDesc(max_node, 0, mean_dims, cfg.dtype);
        SetDesc(max_node, 1, {1}, DT_FLOAT);
        SetOutDesc(max_node, 0, mean_dims, cfg.dtype);
        SetDesc(min_node, 0, mean_dims, cfg.dtype);
        SetDesc(min_node, 1, {1}, DT_FLOAT);
        SetOutDesc(min_node, 0, mean_dims, cfg.dtype);
    }

    // sqrt0
    GNode sqrt0 = MakeUnary(graph, "Sqrt", "sqrt0");
    es::AddEdgeAndUpdatePeerDesc(*graph, sqrt_input, 0, sqrt0, 0);
    SetDesc(sqrt0, 0, mean_dims, cfg.dtype);
    SetOutDesc(sqrt0, 0, mean_dims, cfg.dtype);

    // div0 = RealDiv|Div(sub0, sqrt0)
    GNode div0 = MakeBinary(graph, cfg.div_type.c_str(), "div0");
    es::AddEdgeAndUpdatePeerDesc(*graph, sub0, 0, div0, 0);
    es::AddEdgeAndUpdatePeerDesc(*graph, sqrt0, 0, div0, 1);
    SetDesc(div0, 0, x_dims, cfg.dtype);
    SetDesc(div0, 1, mean_dims, cfg.dtype);
    SetOutDesc(div0, 0, x_dims, cfg.dtype);

    GNode out_node = div0;
    GNode mul0; // 提到 if 外，便于给它挂片段外的额外消费者（出度守卫用例）
    if (cfg.with_affine) {
        const int64_t gamma_len = cfg.broken_gamma_shape ? (last_dim + 1) : last_dim;
        const int64_t beta_len = cfg.broken_beta_shape ? (last_dim + 1) : last_dim;
        mul0 = MakeBinary(graph, "Mul", "mul0");
        GNode add1 = MakeBinary(graph, "Add", "add1");
        auto gamma_const = graph_builder.CreateConst<float>(std::vector<float>(gamma_len, 1.0f), {gamma_len}, DT_FLOAT,
                                                            FORMAT_ND);
        auto beta_const = graph_builder.CreateConst<float>(std::vector<float>(beta_len, 0.0f), {beta_len}, DT_FLOAT,
                                                           FORMAT_ND);
        const int32_t gamma_port = cfg.gamma_first ? 0 : 1;
        const int32_t beta_port = cfg.beta_first ? 0 : 1;
        es::AddEdgeAndUpdatePeerDesc(*graph, div0, 0, mul0, 1 - gamma_port);
        es::AddEdgeAndUpdatePeerDesc(*graph, *gamma_const.GetProducer(), 0, mul0, gamma_port);
        es::AddEdgeAndUpdatePeerDesc(*graph, mul0, 0, add1, 1 - beta_port);
        es::AddEdgeAndUpdatePeerDesc(*graph, *beta_const.GetProducer(), 0, add1, beta_port);
        SetDesc(mul0, 1 - gamma_port, x_dims, cfg.dtype);
        SetDesc(mul0, gamma_port, {gamma_len}, DT_FLOAT);
        SetOutDesc(mul0, 0, x_dims, cfg.dtype);
        SetDesc(add1, 1 - beta_port, x_dims, cfg.dtype);
        SetDesc(add1, beta_port, {beta_len}, DT_FLOAT);
        SetOutDesc(add1, 0, x_dims, cfg.dtype);
        out_node = add1;
    }
    (void)norm_axis;

    // 额外消费者：挂在指定节点的 0 号输出上，并作为第二个图输出，避免被当成死代码
    GNode extra_consumer;
    bool has_extra = false;
    if (!cfg.extra_consumer_on.empty()) {
        const std::map<std::string, std::pair<GNode*, std::vector<int64_t>>> taps = {
            {"rm0", {&rm0, mean_dims}}, {"sub0", {&sub0, x_dims}},    {"pow0", {&pow0, x_dims}},
            {"rm1", {&rm1, mean_dims}}, {"add0", {&add0, mean_dims}}, {"sqrt0", {&sqrt0, mean_dims}},
            {"div0", {&div0, x_dims}},  {"mul0", {&mul0, x_dims}},
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
    LayerNormONNXFusionPass pass;
    return pass.Run(graph, pass_context);
}
} // namespace

class LayerNormONNXFusionPassTest : public testing::Test {
protected:
    void SetUp() override { SetPlatform(3510); }

    // 门禁读 ini 的 [version] NpuArch，故必须喂 PlatFormInfos 这条路径
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

// 主形态：Pow + RealDiv + affine，完整融合
TEST_F(LayerNormONNXFusionPassTest, fusion_case1_pow_realdiv_with_affine)
{
    GraphConfig cfg;
    GraphPtr graph = BuildOnnxLayerNormGraph("onnx_ln_case1", cfg);
    EXPECT_EQ(RunPass(graph), SUCCESS);
    EXPECT_EQ(CountType(graph, "LayerNorm"), 1);
    // affine 必须被**吸收进算子**，而不是只融了归一化段、把 Mul/Add 留在下游。
    // 这依赖 Patterns() 里 affine 变体排在 no-affine 之前——只断言算子个数抓不到这种退化。
    EXPECT_EQ(CountType(graph, "Mul"), 0);
    EXPECT_EQ(CountType(graph, "Add"), 0);
    EXPECT_EQ(CountType(graph, "Sqrt"), 0);
    EXPECT_EQ(CountType(graph, "RealDiv"), 0);
}

// 变体轴：Square 取代 Pow、Div 取代 RealDiv
TEST_F(LayerNormONNXFusionPassTest, fusion_case1_square_div_with_affine)
{
    GraphConfig cfg;
    cfg.pow_type = "Square";
    cfg.div_type = "Div";
    GraphPtr graph = BuildOnnxLayerNormGraph("onnx_ln_case1_sq", cfg);
    EXPECT_EQ(RunPass(graph), SUCCESS);
    EXPECT_EQ(CountType(graph, "LayerNorm"), 1);
}

// 无 affine：规则自建 gamma=1 / beta=0 常量
TEST_F(LayerNormONNXFusionPassTest, fusion_case2_without_affine)
{
    GraphConfig cfg;
    cfg.with_affine = false;
    GraphPtr graph = BuildOnnxLayerNormGraph("onnx_ln_case2", cfg);
    EXPECT_EQ(RunPass(graph), SUCCESS);
    EXPECT_EQ(CountType(graph, "LayerNorm"), 1);
}

// 含 Cast 的形态
TEST_F(LayerNormONNXFusionPassTest, fusion_case3_with_cast_and_affine)
{
    GraphConfig cfg;
    cfg.with_cast = true;
    GraphPtr graph = BuildOnnxLayerNormGraph("onnx_ln_case3", cfg);
    EXPECT_EQ(RunPass(graph), SUCCESS);
    EXPECT_EQ(CountType(graph, "LayerNorm"), 1);
    // affine 必须被**吸收进算子**，而不是只融了归一化段、把 Mul/Add 留在下游。
    // 这依赖 Patterns() 里 affine 变体排在 no-affine 之前——只断言算子个数抓不到这种退化。
    EXPECT_EQ(CountType(graph, "Mul"), 0);
    EXPECT_EQ(CountType(graph, "Add"), 0);
    EXPECT_EQ(CountType(graph, "Cast"), 0);
}

// clip 形态 Maximum -> Minimum
TEST_F(LayerNormONNXFusionPassTest, fusion_case5_clip_max_then_min)
{
    GraphConfig cfg;
    cfg.with_affine = false;
    cfg.clip = ClipForm::kMaxThenMin;
    GraphPtr graph = BuildOnnxLayerNormGraph("onnx_ln_case5", cfg);
    EXPECT_EQ(RunPass(graph), SUCCESS);
    EXPECT_EQ(CountType(graph, "LayerNorm"), 1);
    EXPECT_EQ(CountType(graph, "Maximum"), 0);
    EXPECT_EQ(CountType(graph, "Minimum"), 0);
}

// clip 形态 Minimum -> Maximum：槽号与链上出现顺序交叉
TEST_F(LayerNormONNXFusionPassTest, fusion_case6_clip_min_then_max)
{
    GraphConfig cfg;
    cfg.with_affine = false;
    cfg.clip = ClipForm::kMinThenMax;
    GraphPtr graph = BuildOnnxLayerNormGraph("onnx_ln_case6", cfg);
    EXPECT_EQ(RunPass(graph), SUCCESS);
    EXPECT_EQ(CountType(graph, "LayerNorm"), 1);
}

// 两个 ReduceMean 共用一个 axes Const：边界按 pattern 输入槽组织、不按 tensor 去重
TEST_F(LayerNormONNXFusionPassTest, fusion_shared_axes_const)
{
    GraphConfig cfg;
    cfg.shared_axes_const = true;
    GraphPtr graph = BuildOnnxLayerNormGraph("onnx_ln_shared_axes", cfg);
    EXPECT_EQ(RunPass(graph), SUCCESS);
    EXPECT_EQ(CountType(graph, "LayerNorm"), 1);
    // affine 必须被**吸收进算子**，而不是只融了归一化段、把 Mul/Add 留在下游。
    // 这依赖 Patterns() 里 affine 变体排在 no-affine 之前——只断言算子个数抓不到这种退化。
    EXPECT_EQ(CountType(graph, "Mul"), 0);
    EXPECT_EQ(CountType(graph, "Add"), 0);
}

// 平台 NpuArch 不在支持列表
TEST_F(LayerNormONNXFusionPassTest, not_changed_on_unsupported_platform)
{
    SetPlatform(3010);
    GraphConfig cfg;
    GraphPtr graph = BuildOnnxLayerNormGraph("onnx_ln_bad_arch", cfg);
    EXPECT_EQ(RunPass(graph), GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountType(graph, "LayerNorm"), 0);
    EXPECT_EQ(CountType(graph, "Sqrt"), 1);
}

// ReduceMean 的 keep_dims 不为 true
TEST_F(LayerNormONNXFusionPassTest, not_changed_when_keep_dims_false)
{
    GraphConfig cfg;
    cfg.keep_dims = false;
    GraphPtr graph = BuildOnnxLayerNormGraph("onnx_ln_keepdims", cfg);
    EXPECT_EQ(RunPass(graph), GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountType(graph, "LayerNorm"), 0);
}

// Pow 指数不是 2
TEST_F(LayerNormONNXFusionPassTest, not_changed_when_pow_exp_not_two)
{
    GraphConfig cfg;
    cfg.pow_exp = 3.0f;
    GraphPtr graph = BuildOnnxLayerNormGraph("onnx_ln_pow3", cfg);
    EXPECT_EQ(RunPass(graph), GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountType(graph, "LayerNorm"), 0);
}

// epsilon 超过 0.1（规则一特有的上限守卫）
TEST_F(LayerNormONNXFusionPassTest, not_changed_when_epsilon_too_large)
{
    GraphConfig cfg;
    cfg.epsilon = 0.5f;
    GraphPtr graph = BuildOnnxLayerNormGraph("onnx_ln_bigeps", cfg);
    EXPECT_EQ(RunPass(graph), GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountType(graph, "LayerNorm"), 0);
}

// 片段内部节点多一个片段外消费者：matcher 自包含约束使其不融合
TEST_F(LayerNormONNXFusionPassTest, not_changed_when_sub0_has_extra_consumer)
{
    GraphConfig cfg;
    cfg.extra_consumer_on = "sub0";
    GraphPtr graph = BuildOnnxLayerNormGraph("onnx_ln_extra_sub0", cfg);
    EXPECT_EQ(RunPass(graph), GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountType(graph, "LayerNorm"), 0);
}

// gamma 挂在 Mul 的输入 0（可交换轴）
TEST_F(LayerNormONNXFusionPassTest, fusion_when_gamma_on_first_input)
{
    GraphConfig cfg;
    cfg.gamma_first = true;
    GraphPtr graph = BuildOnnxLayerNormGraph("onnx_ln_gamma_first", cfg);
    EXPECT_EQ(RunPass(graph), SUCCESS);
    EXPECT_EQ(CountType(graph, "LayerNorm"), 1);
    EXPECT_EQ(CountType(graph, "Mul"), 0);
    EXPECT_EQ(CountType(graph, "Add"), 0);
}

// beta 挂在 Add 的输入 0（可交换轴）
TEST_F(LayerNormONNXFusionPassTest, fusion_when_beta_on_first_input)
{
    GraphConfig cfg;
    cfg.beta_first = true;
    GraphPtr graph = BuildOnnxLayerNormGraph("onnx_ln_beta_first", cfg);
    EXPECT_EQ(RunPass(graph), SUCCESS);
    EXPECT_EQ(CountType(graph, "LayerNorm"), 1);
    EXPECT_EQ(CountType(graph, "Mul"), 0);
    EXPECT_EQ(CountType(graph, "Add"), 0);
}

// 无 affine 时 div0 是片段出口，出口允许有片段外消费者，照常融合
TEST_F(LayerNormONNXFusionPassTest, fusion_when_div0_has_extra_consumer_without_affine)
{
    GraphConfig cfg;
    cfg.with_affine = false;
    cfg.extra_consumer_on = "div0";
    GraphPtr graph = BuildOnnxLayerNormGraph("onnx_ln_extra_div0_noaffine", cfg);
    EXPECT_EQ(RunPass(graph), SUCCESS);
    EXPECT_EQ(CountType(graph, "LayerNorm"), 1);
}

// 归一化轴不是最后一维
TEST_F(LayerNormONNXFusionPassTest, not_changed_when_axis_is_not_last)
{
    GraphConfig cfg;
    cfg.axis = 1;
    GraphPtr graph = BuildOnnxLayerNormGraph("onnx_ln_axis1", cfg);
    EXPECT_EQ(RunPass(graph), GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountType(graph, "LayerNorm"), 0);
}

// 动态 shape 且无 affine：造不出常量 shape，不融合
TEST_F(LayerNormONNXFusionPassTest, not_changed_when_dynamic_without_affine)
{
    GraphConfig cfg;
    cfg.x_dims = {-1, 3, 8};
    cfg.with_affine = false;
    GraphPtr graph = BuildOnnxLayerNormGraph("onnx_ln_dyn", cfg);
    EXPECT_EQ(RunPass(graph), GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountType(graph, "LayerNorm"), 0);
}

// fp16 的 eps 常量按 fp16_t 解码，epsilon 属性值正确
TEST_F(LayerNormONNXFusionPassTest, fusion_with_fp16_epsilon_value_is_correct)
{
    GraphConfig cfg;
    cfg.epsilon = 0.03125f; // 见规则二同名用例：避开 IR 默认值，且 fp16 可精确表示
    cfg.eps_fp16 = true;
    GraphPtr graph = BuildOnnxLayerNormGraph("onnx_eps_fp16", cfg);
    EXPECT_EQ(RunPass(graph), SUCCESS);
    EXPECT_EQ(CountType(graph, "LayerNorm"), 1);

    float epsilon = -1.0f;
    ASSERT_TRUE(GetFusedEpsilon(graph, "LayerNorm", epsilon));
    const float expected = Fp16BitsToFloat(ToFp16Bits(cfg.epsilon));
    EXPECT_FLOAT_EQ(expected, 0.03125f);
    EXPECT_FLOAT_EQ(epsilon, expected);
}

// eps 挂在 Add 的输入 0（eps 位轴）
TEST_F(LayerNormONNXFusionPassTest, fusion_when_eps_on_first_input)
{
    GraphConfig cfg;
    cfg.eps_first = true;
    cfg.epsilon = 0.03125f;
    GraphPtr graph = BuildOnnxLayerNormGraph("onnx_eps_first", cfg);
    EXPECT_EQ(RunPass(graph), SUCCESS);
    EXPECT_EQ(CountType(graph, "LayerNorm"), 1);
    EXPECT_EQ(CountType(graph, "Mul"), 0);
    EXPECT_EQ(CountType(graph, "Add"), 0);
    float epsilon = -1.0f;
    ASSERT_TRUE(GetFusedEpsilon(graph, "LayerNorm", epsilon));
    EXPECT_FLOAT_EQ(epsilon, 0.03125f);
}

// 带 affine 的图必须完整融合：断言 Mul/Add 已被吸收，逐个变体压一遍
TEST_F(LayerNormONNXFusionPassTest, affine_absorbed_for_all_variants)
{
    for (const char* pow_type : {"Pow", "Square"}) {
        for (const char* div_type : {"RealDiv", "Div"}) {
            for (const bool with_cast : {false, true}) {
                for (const bool eps_first : {false, true}) {
                    GraphConfig cfg;
                    cfg.pow_type = pow_type;
                    cfg.div_type = div_type;
                    cfg.with_cast = with_cast;
                    cfg.eps_first = eps_first;
                    const std::string name = std::string("onnx_affine_") + pow_type + "_" + div_type + "_" +
                                             (with_cast ? "cast" : "nocast") + "_" + (eps_first ? "epsx1" : "epsx2");
                    GraphPtr graph = BuildOnnxLayerNormGraph(name.c_str(), cfg);
                    EXPECT_EQ(RunPass(graph), SUCCESS) << name;
                    EXPECT_EQ(CountType(graph, "LayerNorm"), 1) << name;
                    EXPECT_EQ(CountType(graph, "Mul"), 0) << name;
                    EXPECT_EQ(CountType(graph, "Add"), 0) << name;
                    EXPECT_EQ(CountType(graph, "Sub"), 0) << name;
                }
            }
        }
    }
}

// 无 affine 且 dtype 非 fp16/fp32：必须是不融合，而不是整图编译失败
TEST_F(LayerNormONNXFusionPassTest, not_changed_when_no_affine_and_dtype_unsupported)
{
    GraphConfig cfg;
    cfg.with_affine = false;
    cfg.dtype = DT_BF16;
    GraphPtr graph = BuildOnnxLayerNormGraph("onnx_noaffine_bf16", cfg);
    // 关键：是 GRAPH_NOT_CHANGED（安全），不是失败
    EXPECT_EQ(RunPass(graph), GRAPH_NOT_CHANGED);
    EXPECT_EQ(CountType(graph, "LayerNorm"), 0);
    // 原图算子保持不动
    EXPECT_EQ(CountType(graph, "Sub"), 1);
    EXPECT_EQ(CountType(graph, "Sqrt"), 1);
}

// 有 affine 时不走自建常量，dtype 门禁不应生效
TEST_F(LayerNormONNXFusionPassTest, fusion_with_affine_is_not_blocked_by_const_dtype_guard)
{
    GraphConfig cfg;
    cfg.with_affine = true;
    GraphPtr graph = BuildOnnxLayerNormGraph("onnx_affine_dtype_guard", cfg);
    EXPECT_EQ(RunPass(graph), SUCCESS);
    EXPECT_EQ(CountType(graph, "LayerNorm"), 1);
}
