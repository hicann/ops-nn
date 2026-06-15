/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "ut_op_util.h"
#include "../../../op_graph/fusion_pass/multi_add_rms_norm_dynamic_quant_fusion_pass.h"
#include "es_nn_ops.h"
#include "es_math_ops.h"
#include "platform/platform_info.h"
#include "compliant_node_builder.h"

using namespace ge;
using namespace fe;
using namespace fusion;

static es::EsTensorHolder BuildConstNode(es::EsGraphBuilder& builder)
{
    static int counter = 0;
    std::string name = "Const_" + std::to_string(counter++);
    auto graph = builder.GetCGraphBuilder()->GetGraph();
    GNode node = es::CompliantNodeBuilder(graph)
        .OpType("Const")
        .Name(name.c_str())
        .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
        .Build();
    return es::EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 0));
}

static es::EsTensorHolder BuildReshapeNode(
    es::EsGraphBuilder& builder, const es::EsTensorHolder& x, const es::EsTensorHolder& shape)
{
    static int counter = 0;
    std::string name = "Reshape_" + std::to_string(counter++);
    auto graph = builder.GetCGraphBuilder()->GetGraph();
    GNode node = es::CompliantNodeBuilder(graph)
        .OpType("Reshape")
        .Name(name.c_str())
        .IrDefInputs({
            {"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
            {"shape", es::CompliantNodeBuilder::kEsIrInputRequired, ""}
        })
        .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
        .Build();
    es::AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(), node, 0);
    es::AddEdgeAndUpdatePeerDesc(*graph, *shape.GetProducer(), shape.GetProducerOutIndex(), node, 1);
    return es::EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 0));
}

static GNode BuildAddNode(
    es::EsGraphBuilder& builder, const es::EsTensorHolder& x1, const es::EsTensorHolder& x2)
{
    static int counter = 0;
    std::string name = "Add_" + std::to_string(counter++);
    auto graph = builder.GetCGraphBuilder()->GetGraph();
    GNode node = es::CompliantNodeBuilder(graph)
        .OpType("Add")
        .Name(name.c_str())
        .IrDefInputs({
            {"x1", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
            {"x2", es::CompliantNodeBuilder::kEsIrInputRequired, ""}
        })
        .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
        .Build();
    es::AddEdgeAndUpdatePeerDesc(*graph, *x1.GetProducer(), x1.GetProducerOutIndex(), node, 0);
    es::AddEdgeAndUpdatePeerDesc(*graph, *x2.GetProducer(), x2.GetProducerOutIndex(), node, 1);
    return node;
}

static es::EsTensorHolder BuildAddTensor(
    es::EsGraphBuilder& builder, const es::EsTensorHolder& x1, const es::EsTensorHolder& x2)
{
    GNode node = BuildAddNode(builder, x1, x2);
    return es::EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 0));
}

struct AddRmsNormResult {
    es::EsTensorHolder x;
    es::EsTensorHolder y;
    es::EsTensorHolder rstd;
};

static AddRmsNormResult BuildAddRmsNormNode(
    es::EsGraphBuilder& builder, const es::EsTensorHolder& x1,
    const es::EsTensorHolder& x2, const es::EsTensorHolder& gamma)
{
    static int counter = 0;
    std::string name = "AddRmsNorm_" + std::to_string(counter++);
    auto graph = builder.GetCGraphBuilder()->GetGraph();
    GNode node = es::CompliantNodeBuilder(graph)
        .OpType("AddRmsNorm")
        .Name(name.c_str())
        .IrDefInputs({
            {"x1", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
            {"x2", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
            {"gamma", es::CompliantNodeBuilder::kEsIrInputRequired, ""}
        })
        .IrDefOutputs({
            {"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
            {"rstd", es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
            {"x", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}
        })
        .IrDefAttrs({
            {"epsilon", es::CompliantNodeBuilder::kEsAttrOptional, "Float",
             es::CreateFrom(1e-6f)}
        })
        .Build();
    es::AddEdgeAndUpdatePeerDesc(*graph, *x1.GetProducer(), x1.GetProducerOutIndex(), node, 0);
    es::AddEdgeAndUpdatePeerDesc(*graph, *x2.GetProducer(), x2.GetProducerOutIndex(), node, 1);
    es::AddEdgeAndUpdatePeerDesc(*graph, *gamma.GetProducer(), gamma.GetProducerOutIndex(), node, 2);
    return {
        es::EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 2)),
        es::EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 0)),
        es::EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 1))
    };
}

struct DynamicQuantResult {
    es::EsTensorHolder y;
    es::EsTensorHolder scale;
};

static DynamicQuantResult BuildDynamicQuantNode(
    es::EsGraphBuilder& builder, const es::EsTensorHolder& x, const es::EsTensorHolder& smooth)
{
    static int counter = 0;
    std::string name = "DynamicQuant_" + std::to_string(counter++);
    auto graph = builder.GetCGraphBuilder()->GetGraph();
    GNode node = es::CompliantNodeBuilder(graph)
        .OpType("DynamicQuant")
        .Name(name.c_str())
        .IrDefInputs({
            {"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
            {"smooth_scales", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
            {"group_index", es::CompliantNodeBuilder::kEsIrInputOptional, ""}
        })
        .IrDefOutputs({
            {"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""},
            {"scale", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}
        })
        .Build();
    es::AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(), node, 0);
    es::AddEdgeAndUpdatePeerDesc(*graph, *smooth.GetProducer(), smooth.GetProducerOutIndex(), node, 1);
    return {
        es::EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 0)),
        es::EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 1))
    };
}

class multi_add_rms_norm_dynamic_quant_fusion_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "multi_add_rms_norm_dynamic_quant_fusion_test SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "multi_add_rms_norm_dynamic_quant_fusion_test TearDown" << std::endl;
    }

    void SetUpPlatform()
    {
        PlatformInfo platform_info;
        OptionalInfo opti_compilation_info;
        platform_info.soc_info.ai_core_cnt = 40;
        opti_compilation_info.soc_version = "Ascend910B1";
        platform_info.str_info.short_soc_version = "Ascend910B";
        PlatformInfoManager::Instance().platform_info_map_["Ascend910B1"] = platform_info;
        PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);
    }

    void ClearPlatform()
    {
        PlatformInfoManager::Instance().platform_info_map_.clear();
    }

    bool HasFusedOp(std::shared_ptr<Graph>& graph, const std::string& op_type)
    {
        for (auto node : graph->GetAllNodes()) {
            AscendString type;
            node.GetType(type);
            if (type == op_type.c_str()) {
                return true;
            }
        }
        return false;
    }
};

/**
 * Negative test: x1 inputs use DT_FLOAT (not FP16/BF16) -> fusion should NOT happen
 */
TEST_F(multi_add_rms_norm_dynamic_quant_fusion_test, test_wrong_dtype)
{
    SetUpPlatform();

    std::vector<int64_t> dims_x{1, 1024, 8192};
    std::vector<int64_t> dims_gamma{8192};

    auto graph_builder = es::EsGraphBuilder("test_wrong_dtype");
    auto x1_0 = graph_builder.CreateInput(0, "x1_0", DT_FLOAT, FORMAT_ND, dims_x);
    auto x1_1 = graph_builder.CreateInput(1, "x1_1", DT_FLOAT, FORMAT_ND, dims_x);
    auto x1_2 = graph_builder.CreateInput(2, "x1_2", DT_FLOAT, FORMAT_ND, dims_x);
    auto x2 = graph_builder.CreateInput(3, "x2", DT_FLOAT, FORMAT_ND, dims_x);
    auto gamma = graph_builder.CreateInput(4, "gamma", DT_FLOAT, FORMAT_ND, dims_gamma);
    auto smooth = graph_builder.CreateInput(5, "smooth", DT_FLOAT, FORMAT_ND, dims_gamma);

    auto add1 = BuildAddTensor(graph_builder, x1_0, x1_1);
    auto add2 = BuildAddTensor(graph_builder, add1, x1_2);
    auto arnResult = BuildAddRmsNormNode(graph_builder, add2, x2, gamma);
    auto dq1 = BuildDynamicQuantNode(graph_builder, arnResult.y, smooth);

    std::vector<es::EsTensorHolder> outputs;
    outputs.emplace_back(std::move(dq1.y));
    outputs.emplace_back(std::move(arnResult.x));
    outputs.emplace_back(std::move(arnResult.y));
    outputs.emplace_back(std::move(dq1.scale));
    std::shared_ptr<Graph> graph = graph_builder.BuildAndReset(outputs);
    CustomPassContext pass_context;

    ops::MultiAddRmsNormDynamicQuantFusionPass pass_dtype;
    Status ret = pass_dtype.Run(graph, pass_context);
    ClearPlatform();
    EXPECT_EQ(ret, GRAPH_NOT_CHANGED);
    EXPECT_FALSE(HasFusedOp(graph, "MultiAddRmsNormDynamicQuant"));
}

/**
 * Negative test: gamma uses 2D shape -> fusion should NOT happen
 */
TEST_F(multi_add_rms_norm_dynamic_quant_fusion_test, test_gamma_wrong_shape)
{
    SetUpPlatform();

    std::vector<int64_t> dims_x{1, 1024, 8192};
    std::vector<int64_t> dims_gamma_2d{1, 8192};

    auto graph_builder = es::EsGraphBuilder("test_gamma_wrong_shape");
    auto x1_0 = graph_builder.CreateInput(0, "x1_0", DT_FLOAT16, FORMAT_ND, dims_x);
    auto x1_1 = graph_builder.CreateInput(1, "x1_1", DT_FLOAT16, FORMAT_ND, dims_x);
    auto x1_2 = graph_builder.CreateInput(2, "x1_2", DT_FLOAT16, FORMAT_ND, dims_x);
    auto x2 = graph_builder.CreateInput(3, "x2", DT_FLOAT16, FORMAT_ND, dims_x);
    auto gamma = graph_builder.CreateInput(4, "gamma", DT_FLOAT16, FORMAT_ND, dims_gamma_2d);
    auto smooth = graph_builder.CreateInput(5, "smooth", DT_FLOAT16, FORMAT_ND, dims_x);

    auto add1 = BuildAddTensor(graph_builder, x1_0, x1_1);
    auto add2 = BuildAddTensor(graph_builder, add1, x1_2);
    auto arnResult = BuildAddRmsNormNode(graph_builder, add2, x2, gamma);
    auto dq1 = BuildDynamicQuantNode(graph_builder, arnResult.y, smooth);

    std::vector<es::EsTensorHolder> outputs;
    outputs.emplace_back(std::move(dq1.y));
    outputs.emplace_back(std::move(arnResult.x));
    outputs.emplace_back(std::move(arnResult.y));
    outputs.emplace_back(std::move(dq1.scale));
    std::shared_ptr<Graph> graph = graph_builder.BuildAndReset(outputs);
    CustomPassContext pass_context;

    ops::MultiAddRmsNormDynamicQuantFusionPass pass_gamma;
    Status ret2 = pass_gamma.Run(graph, pass_context);
    ClearPlatform();
    EXPECT_EQ(ret2, GRAPH_NOT_CHANGED);
    EXPECT_FALSE(HasFusedOp(graph, "MultiAddRmsNormDynamicQuant"));
}

/**
 * Negative test: non-910b/910_93 platform -> fusion should NOT happen
 */
TEST_F(multi_add_rms_norm_dynamic_quant_fusion_test, test_wrong_platform)
{
    PlatformInfo platform_info;
    OptionalInfo opti_compilation_info;
    platform_info.soc_info.ai_core_cnt = 40;
    opti_compilation_info.soc_version = "Ascend310P3";
    platform_info.str_info.short_soc_version = "Ascend310P";
    PlatformInfoManager::Instance().platform_info_map_["Ascend310P3"] = platform_info;
    PlatformInfoManager::Instance().SetOptionalCompilationInfo(opti_compilation_info);

    std::vector<int64_t> dims_x{1, 1024, 8192};
    std::vector<int64_t> dims_gamma{8192};

    auto graph_builder = es::EsGraphBuilder("test_wrong_platform");
    auto x1_0 = graph_builder.CreateInput(0, "x1_0", DT_FLOAT16, FORMAT_ND, dims_x);
    auto x1_1 = graph_builder.CreateInput(1, "x1_1", DT_FLOAT16, FORMAT_ND, dims_x);
    auto x1_2 = graph_builder.CreateInput(2, "x1_2", DT_FLOAT16, FORMAT_ND, dims_x);
    auto x2 = graph_builder.CreateInput(3, "x2", DT_FLOAT16, FORMAT_ND, dims_x);
    auto gamma = graph_builder.CreateInput(4, "gamma", DT_FLOAT16, FORMAT_ND, dims_gamma);
    auto smooth = graph_builder.CreateInput(5, "smooth", DT_FLOAT16, FORMAT_ND, dims_gamma);

    auto add1 = BuildAddTensor(graph_builder, x1_0, x1_1);
    auto add2 = BuildAddTensor(graph_builder, add1, x1_2);
    auto arnResult = BuildAddRmsNormNode(graph_builder, add2, x2, gamma);
    auto dq1 = BuildDynamicQuantNode(graph_builder, arnResult.y, smooth);

    std::vector<es::EsTensorHolder> outputs;
    outputs.emplace_back(std::move(dq1.y));
    outputs.emplace_back(std::move(arnResult.x));
    outputs.emplace_back(std::move(arnResult.y));
    outputs.emplace_back(std::move(dq1.scale));
    std::shared_ptr<Graph> graph = graph_builder.BuildAndReset(outputs);
    CustomPassContext pass_context;

    ops::MultiAddRmsNormDynamicQuantFusionPass pass_plat;
    Status ret3 = pass_plat.Run(graph, pass_context);
    ClearPlatform();
    EXPECT_EQ(ret3, GRAPH_NOT_CHANGED);
    EXPECT_FALSE(HasFusedOp(graph, "MultiAddRmsNormDynamicQuant"));
}
