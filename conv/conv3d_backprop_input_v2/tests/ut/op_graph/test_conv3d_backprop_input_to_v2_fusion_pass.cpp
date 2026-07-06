/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include <string>
#include <vector>

#include "platform/platform_info.h"
#include "register/register_custom_pass.h"
#include "ge/compliant_node_builder.h"
#include "ge/es_graph_builder.h"
#include "../../../op_graph/fusion_pass/conv3d_backprop_input_to_v2_fusion_pass.h"

using namespace ge;
using namespace fe;
using namespace fusion;
using namespace ops::ConvBackpropFusionUtils;

namespace {

void SetPlatform(const std::string& soc)
{
    PlatformInfo platformInfo;
    OptionalInfo optionalInfo;
    platformInfo.soc_info.ai_core_cnt = 64;
    platformInfo.str_info.short_soc_version = soc;
    optionalInfo.soc_version = soc;
    if (soc == "Ascend950" || soc == "MC62CM12A") {
        platformInfo.ai_core_intrinsic_dtype_map["Intrinsic_data_move_out2l1_dn2nz"] = {"float16", "float", "bfloat16"};
    }
    PlatformInfoManager::Instance().platform_info_map_[soc] = platformInfo;
    PlatformInfoManager::Instance().SetOptionalCompilationInfo(optionalInfo);
}

es::EsTensorHolder CreateConv3dBpInputNode(es::EsGraphBuilder& builder, const char* opType,
                                           const es::EsTensorHolder& inputSize, const es::EsTensorHolder& filter,
                                           const es::EsTensorHolder& dedy, std::vector<int64_t> strides,
                                           std::vector<int64_t> pads, std::vector<int64_t> dilations, int64_t groups,
                                           const std::string& dataFormat, DataType outDtype,
                                           const std::vector<int64_t>& outShape, Format outFormat)
{
    auto* graph = builder.GetCGraphBuilder()->GetGraph();
    auto node = es::CompliantNodeBuilder(graph)
                    .OpType(opType)
                    .Name(opType)
                    .IrDefInputs({{"input_size", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                  {"filter", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                  {"out_backprop", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                    .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                    .InstanceOutputDataType("y", outDtype)
                    .InstanceOutputShape("y", outShape)
                    .InstanceOutputFormat("y", outFormat)
                    .Build();

    es::AddEdgeAndUpdatePeerDesc(*graph, *inputSize.GetProducer(), inputSize.GetProducerOutIndex(), node, 0);
    es::AddEdgeAndUpdatePeerDesc(*graph, *filter.GetProducer(), filter.GetProducerOutIndex(), node, 1);
    es::AddEdgeAndUpdatePeerDesc(*graph, *dedy.GetProducer(), dedy.GetProducerOutIndex(), node, 2);

    TensorDesc inputSizeDesc, filterDesc, dedyDesc;
    inputSize.GetProducer()->GetOutputDesc(inputSize.GetProducerOutIndex(), inputSizeDesc);
    filter.GetProducer()->GetOutputDesc(filter.GetProducerOutIndex(), filterDesc);
    dedy.GetProducer()->GetOutputDesc(dedy.GetProducerOutIndex(), dedyDesc);
    node.UpdateInputDesc(0, inputSizeDesc);
    node.UpdateInputDesc(1, filterDesc);
    node.UpdateInputDesc(2, dedyDesc);

    node.SetAttr("strides", strides);
    node.SetAttr("pads", pads);
    node.SetAttr("dilations", dilations);
    node.SetAttr("groups", groups);
    AscendString fmt = dataFormat.c_str();
    node.SetAttr("data_format", fmt);
    int64_t implMode = 0x1;
    node.SetAttr("_op_impl_mode_enum", implMode);

    return es::EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 0));
}

bool CheckNodeExists(GraphPtr& graph, const std::string& type)
{
    for (auto node : graph->GetAllNodes()) {
        AscendString nodeType;
        node.GetType(nodeType);
        if (nodeType.GetString() == type)
            return true;
    }
    return false;
}

} // namespace

class Conv3dBpInputToV2FusionPassTest : public testing::Test {
protected:
    void SetUp() override { SetPlatform("Ascend950"); }
};

// Test 1: patternTest - FP16 basic fusion success (需要 Transpose)
TEST_F(Conv3dBpInputToV2FusionPassTest, patternTest)
{
    auto builder = es::EsGraphBuilder("patternTest");
    auto inputSize = builder.CreateInput(0, "input_size", DT_INT64, FORMAT_ND, {5});
    auto filter = builder.CreateInput(1, "filter", DT_FLOAT16, FORMAT_DHWCN, {1, 4, 4, 512, 1037});
    auto dedy = builder.CreateInput(2, "out_backprop", DT_FLOAT16, FORMAT_NDHWC, {256, 1, 4, 4, 1037});

    auto y = CreateConv3dBpInputNode(builder, "Conv3DBackpropInput", inputSize, filter, dedy, {1, 2, 2, 2, 1},
                                     {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NDHWC", DT_FLOAT16, {256, 1, 8, 8, 512},
                                     FORMAT_NDHWC);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::Conv3DBackpropInputToV2FusionPass pass({AscendString("Conv3DBackpropInput")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv3DBackpropInputV2"));
}

// Test 2: unsupportedPlatformFail - Ascend910_93 不支持
TEST_F(Conv3dBpInputToV2FusionPassTest, unsupportedPlatformFail)
{
    SetPlatform("Ascend910_93");
    auto builder = es::EsGraphBuilder("unsupportedPlatformFail");
    auto inputSize = builder.CreateInput(0, "input_size", DT_INT64, FORMAT_ND, {5});
    auto filter = builder.CreateInput(1, "filter", DT_FLOAT16, FORMAT_DHWCN, {1, 4, 4, 512, 1037});
    auto dedy = builder.CreateInput(2, "out_backprop", DT_FLOAT16, FORMAT_NDHWC, {256, 1, 4, 4, 1037});

    auto y = CreateConv3dBpInputNode(builder, "Conv3DBackpropInput", inputSize, filter, dedy, {1, 2, 2, 2, 1},
                                     {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NDHWC", DT_FLOAT16, {256, 1, 8, 8, 512},
                                     FORMAT_NDHWC);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::Conv3DBackpropInputToV2FusionPass pass({AscendString("Conv3DBackpropInput")});
    EXPECT_EQ(pass.Run(graph, ctx), GRAPH_NOT_CHANGED);
    EXPECT_FALSE(CheckNodeExists(graph, "Conv3DBackpropInputV2"));
}

// Test 4: bf16FusionSuccess - BF16 融合成功
TEST_F(Conv3dBpInputToV2FusionPassTest, bf16FusionSuccess)
{
    auto builder = es::EsGraphBuilder("bf16FusionSuccess");
    auto inputSize = builder.CreateInput(0, "input_size", DT_INT64, FORMAT_ND, {5});
    auto filter = builder.CreateInput(1, "filter", DT_BF16, FORMAT_DHWCN, {1, 4, 4, 512, 1037});
    auto dedy = builder.CreateInput(2, "out_backprop", DT_BF16, FORMAT_NDHWC, {256, 1, 4, 4, 1037});

    auto y = CreateConv3dBpInputNode(builder, "Conv3DBackpropInput", inputSize, filter, dedy, {1, 2, 2, 2, 1},
                                     {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NDHWC", DT_BF16, {256, 1, 8, 8, 512},
                                     FORMAT_NDHWC);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::Conv3DBackpropInputToV2FusionPass pass({AscendString("Conv3DBackpropInput")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv3DBackpropInputV2"));
}

// Test 5: fp32FusionSuccess - FP32 融合成功
TEST_F(Conv3dBpInputToV2FusionPassTest, fp32FusionSuccess)
{
    auto builder = es::EsGraphBuilder("fp32FusionSuccess");
    auto inputSize = builder.CreateInput(0, "input_size", DT_INT64, FORMAT_ND, {5});
    auto filter = builder.CreateInput(1, "filter", DT_FLOAT, FORMAT_DHWCN, {1, 4, 4, 512, 1037});
    auto dedy = builder.CreateInput(2, "out_backprop", DT_FLOAT, FORMAT_NDHWC, {256, 1, 4, 4, 1037});

    auto y = CreateConv3dBpInputNode(builder, "Conv3DBackpropInput", inputSize, filter, dedy, {1, 2, 2, 2, 1},
                                     {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NDHWC", DT_FLOAT, {256, 1, 8, 8, 512},
                                     FORMAT_NDHWC);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::Conv3DBackpropInputToV2FusionPass pass({AscendString("Conv3DBackpropInput")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv3DBackpropInputV2"));
}

// Test 6: mc62cm12APlatformFail - MC62CM12A 平台不支持
TEST_F(Conv3dBpInputToV2FusionPassTest, mc62cm12APlatformFail)
{
    SetPlatform("MC62CM12A");
    auto builder = es::EsGraphBuilder("mc62cm12APlatformFail");
    auto inputSize = builder.CreateInput(0, "input_size", DT_INT64, FORMAT_ND, {5});
    auto filter = builder.CreateInput(1, "filter", DT_FLOAT16, FORMAT_DHWCN, {1, 4, 4, 512, 1037});
    auto dedy = builder.CreateInput(2, "out_backprop", DT_FLOAT16, FORMAT_NDHWC, {256, 1, 4, 4, 1037});

    auto y = CreateConv3dBpInputNode(builder, "Conv3DBackpropInput", inputSize, filter, dedy, {1, 2, 2, 2, 1},
                                     {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NDHWC", DT_FLOAT16, {256, 1, 8, 8, 512},
                                     FORMAT_NDHWC);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::Conv3DBackpropInputToV2FusionPass pass({AscendString("Conv3DBackpropInput")});
    EXPECT_EQ(pass.Run(graph, ctx), GRAPH_NOT_CHANGED);
    EXPECT_FALSE(CheckNodeExists(graph, "Conv3DBackpropInputV2"));
}

// Test 7: differentShapeSmallBatch - 不同 shape（小 batch）
TEST_F(Conv3dBpInputToV2FusionPassTest, differentShapeSmallBatch)
{
    auto builder = es::EsGraphBuilder("differentShapeSmallBatch");
    auto inputSize = builder.CreateInput(0, "input_size", DT_INT64, FORMAT_ND, {5});
    auto filter = builder.CreateInput(1, "filter", DT_FLOAT16, FORMAT_DHWCN, {1, 5, 5, 128, 256});
    auto dedy = builder.CreateInput(2, "out_backprop", DT_FLOAT16, FORMAT_NDHWC, {64, 1, 16, 16, 256});

    auto y = CreateConv3dBpInputNode(builder, "Conv3DBackpropInput", inputSize, filter, dedy, {1, 2, 2, 2, 1},
                                     {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NDHWC", DT_FLOAT16, {64, 1, 32, 32, 128},
                                     FORMAT_NDHWC);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::Conv3DBackpropInputToV2FusionPass pass({AscendString("Conv3DBackpropInput")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv3DBackpropInputV2"));
}

// Test 8: noTransposeCase - filter 格式不是 DHWCN，不需要 Transpose 但仍融合成功
TEST_F(Conv3dBpInputToV2FusionPassTest, noTransposeCase)
{
    auto builder = es::EsGraphBuilder("noTransposeCase");
    auto inputSize = builder.CreateInput(0, "input_size", DT_INT64, FORMAT_ND, {5});
    auto filter = builder.CreateInput(1, "filter", DT_FLOAT16, FORMAT_NCDHW, {512, 1, 4, 4, 1037});
    auto dedy = builder.CreateInput(2, "out_backprop", DT_FLOAT16, FORMAT_NDHWC, {256, 1, 4, 4, 1037});

    auto y = CreateConv3dBpInputNode(builder, "Conv3DBackpropInput", inputSize, filter, dedy, {1, 2, 2, 2, 1},
                                     {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NDHWC", DT_FLOAT16, {256, 1, 8, 8, 512},
                                     FORMAT_NDHWC);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::Conv3DBackpropInputToV2FusionPass pass({AscendString("Conv3DBackpropInput")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv3DBackpropInputV2"));
}

// Test 9: strideNotMatchCase - stride 不满足条件，不需要 Transpose 但仍融合成功
TEST_F(Conv3dBpInputToV2FusionPassTest, strideNotMatchCase)
{
    auto builder = es::EsGraphBuilder("strideNotMatchCase");
    auto inputSize = builder.CreateInput(0, "input_size", DT_INT64, FORMAT_ND, {5});
    auto filter = builder.CreateInput(1, "filter", DT_FLOAT16, FORMAT_DHWCN, {1, 4, 4, 512, 1037});
    auto dedy = builder.CreateInput(2, "out_backprop", DT_FLOAT16, FORMAT_NDHWC, {256, 1, 4, 4, 1037});

    auto y = CreateConv3dBpInputNode(builder, "Conv3DBackpropInput", inputSize, filter, dedy, {1, 1, 1, 1, 1},
                                     {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NDHWC", DT_FLOAT16, {256, 1, 8, 8, 512},
                                     FORMAT_NDHWC);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::Conv3DBackpropInputToV2FusionPass pass({AscendString("Conv3DBackpropInput")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv3DBackpropInputV2"));
}

// Test 10: groupsNotMatchCase - groups 不是 1，不需要 Transpose 但仍融合成功
TEST_F(Conv3dBpInputToV2FusionPassTest, groupsNotMatchCase)
{
    auto builder = es::EsGraphBuilder("groupsNotMatchCase");
    auto inputSize = builder.CreateInput(0, "input_size", DT_INT64, FORMAT_ND, {5});
    auto filter = builder.CreateInput(1, "filter", DT_FLOAT16, FORMAT_DHWCN, {1, 4, 4, 512, 1037});
    auto dedy = builder.CreateInput(2, "out_backprop", DT_FLOAT16, FORMAT_NDHWC, {256, 1, 4, 4, 1037});

    auto y = CreateConv3dBpInputNode(builder, "Conv3DBackpropInput", inputSize, filter, dedy, {1, 2, 2, 2, 1},
                                     {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 2, "NDHWC", DT_FLOAT16, {256, 1, 8, 8, 512},
                                     FORMAT_NDHWC);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::Conv3DBackpropInputToV2FusionPass pass({AscendString("Conv3DBackpropInput")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv3DBackpropInputV2"));
}

// Test 11: shapeNotInWhitelistCase - shape 不在白名单，不需要 Transpose 但仍融合成功
TEST_F(Conv3dBpInputToV2FusionPassTest, shapeNotInWhitelistCase)
{
    auto builder = es::EsGraphBuilder("shapeNotInWhitelistCase");
    auto inputSize = builder.CreateInput(0, "input_size", DT_INT64, FORMAT_ND, {5});
    auto filter = builder.CreateInput(1, "filter", DT_FLOAT16, FORMAT_DHWCN, {1, 4, 4, 128, 256});
    auto dedy = builder.CreateInput(2, "out_backprop", DT_FLOAT16, FORMAT_NDHWC, {128, 1, 8, 8, 256});

    auto y = CreateConv3dBpInputNode(builder, "Conv3DBackpropInput", inputSize, filter, dedy, {1, 2, 2, 2, 1},
                                     {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NDHWC", DT_FLOAT16, {128, 1, 16, 16, 128},
                                     FORMAT_NDHWC);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::Conv3DBackpropInputToV2FusionPass pass({AscendString("Conv3DBackpropInput")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv3DBackpropInputV2"));
}