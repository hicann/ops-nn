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
#include "../../../op_graph/fusion_pass/conv3d_backprop_filter_to_v2_fusion_pass.h"

using namespace ge;
using namespace fe;
using namespace fusion;
using namespace ops::ConvBackpropFusionUtils;

namespace {

void SetPlatform(const std::string& soc)
{
    PlatformInfo platformInfo;
    OptionalInfo optionalInfo;
    platformInfo.soc_info.ai_core_cnt = 32;
    platformInfo.str_info.short_soc_version = soc;
    optionalInfo.soc_version = soc;
    if (soc == "Ascend950" || soc == "MC62CM12A") {
        platformInfo.ai_core_intrinsic_dtype_map["Intrinsic_data_move_out2l1_dn2nz"] = {"float16", "float", "bfloat16"};
    }
    PlatformInfoManager::Instance().platform_info_map_[soc] = platformInfo;
    PlatformInfoManager::Instance().SetOptionalCompilationInfo(optionalInfo);
}

es::EsTensorHolder CreateConv3dBpFilterNode(
    es::EsGraphBuilder& builder, const char* opType, const es::EsTensorHolder& x, const es::EsTensorHolder& filterSize,
    const es::EsTensorHolder& outBackprop, std::vector<int64_t> strides, std::vector<int64_t> pads,
    std::vector<int64_t> dilations, int64_t groups, const std::string& dataFormat, DataType outDtype,
    const std::vector<int64_t>& outShape, Format outFormat)
{
    auto* graph = builder.GetCGraphBuilder()->GetGraph();
    auto node = es::CompliantNodeBuilder(graph)
                    .OpType(opType)
                    .Name(opType)
                    .IrDefInputs(
                        {{"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                         {"filter_size", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                         {"out_backprop", es::CompliantNodeBuilder::kEsIrInputRequired, ""}})
                    .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                    .InstanceOutputDataType("y", outDtype)
                    .InstanceOutputShape("y", outShape)
                    .InstanceOutputFormat("y", outFormat)
                    .Build();

    es::AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(), node, 0);
    es::AddEdgeAndUpdatePeerDesc(*graph, *filterSize.GetProducer(), filterSize.GetProducerOutIndex(), node, 1);
    es::AddEdgeAndUpdatePeerDesc(*graph, *outBackprop.GetProducer(), outBackprop.GetProducerOutIndex(), node, 2);

    TensorDesc xDesc, filterSizeDesc, outBackpropDesc;
    x.GetProducer()->GetOutputDesc(x.GetProducerOutIndex(), xDesc);
    filterSize.GetProducer()->GetOutputDesc(filterSize.GetProducerOutIndex(), filterSizeDesc);
    outBackprop.GetProducer()->GetOutputDesc(outBackprop.GetProducerOutIndex(), outBackpropDesc);
    node.UpdateInputDesc(0, xDesc);
    node.UpdateInputDesc(1, filterSizeDesc);
    node.UpdateInputDesc(2, outBackpropDesc);

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

class Conv3dBpFilterToV2FusionPassTest : public testing::Test {
protected:
    void SetUp() override { SetPlatform("Ascend950"); }
};

// Test 1: patternTest - FP16 basic fusion success (需要 Transpose)
TEST_F(Conv3dBpFilterToV2FusionPassTest, patternTest)
{
    auto builder = es::EsGraphBuilder("patternTest");
    auto x = builder.CreateInput(0, "x", DT_FLOAT16, FORMAT_NCDHW, {2, 32, 16, 16, 16});
    auto filterSize = builder.CreateInput(1, "filter_size", DT_INT64, FORMAT_ND, {5});
    auto outBackprop = builder.CreateInput(2, "out_backprop", DT_FLOAT16, FORMAT_NDHWC, {2, 1, 8, 8, 64});

    auto y = CreateConv3dBpFilterNode(
        builder, "Conv3DBackpropFilter", x, filterSize, outBackprop, {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0},
        {1, 1, 1, 1, 1}, 1, "NCDHW", DT_FLOAT16, {2, 1, 4, 4, 32}, FORMAT_NDHWC);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::Conv3DBackpropFilterToV2FusionPass pass({AscendString("Conv3DBackpropFilter")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv3DBackpropFilterV2"));
}

// Test 2: unsupportedPlatformFail - Ascend910_93 不支持
TEST_F(Conv3dBpFilterToV2FusionPassTest, unsupportedPlatformFail)
{
    SetPlatform("Ascend910_93");
    auto builder = es::EsGraphBuilder("unsupportedPlatformFail");
    auto x = builder.CreateInput(0, "x", DT_FLOAT16, FORMAT_NCDHW, {2, 32, 16, 16, 16});
    auto filterSize = builder.CreateInput(1, "filter_size", DT_INT64, FORMAT_ND, {5});
    auto outBackprop = builder.CreateInput(2, "out_backprop", DT_FLOAT16, FORMAT_NDHWC, {2, 1, 8, 8, 64});

    auto y = CreateConv3dBpFilterNode(
        builder, "Conv3DBackpropFilter", x, filterSize, outBackprop, {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0},
        {1, 1, 1, 1, 1}, 1, "NCDHW", DT_FLOAT16, {2, 1, 4, 4, 32}, FORMAT_NDHWC);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::Conv3DBackpropFilterToV2FusionPass pass({AscendString("Conv3DBackpropFilter")});
    EXPECT_EQ(pass.Run(graph, ctx), GRAPH_NOT_CHANGED);
    EXPECT_FALSE(CheckNodeExists(graph, "Conv3DBackpropFilterV2"));
}

// Test 4: bf16FusionSuccess - BF16 融合成功
TEST_F(Conv3dBpFilterToV2FusionPassTest, bf16FusionSuccess)
{
    auto builder = es::EsGraphBuilder("bf16FusionSuccess");
    auto x = builder.CreateInput(0, "x", DT_BF16, FORMAT_NCDHW, {2, 32, 16, 16, 16});
    auto filterSize = builder.CreateInput(1, "filter_size", DT_INT64, FORMAT_ND, {5});
    auto outBackprop = builder.CreateInput(2, "out_backprop", DT_BF16, FORMAT_NDHWC, {2, 1, 8, 8, 64});

    auto y = CreateConv3dBpFilterNode(
        builder, "Conv3DBackpropFilter", x, filterSize, outBackprop, {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0},
        {1, 1, 1, 1, 1}, 1, "NCDHW", DT_BF16, {2, 1, 4, 4, 32}, FORMAT_NDHWC);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::Conv3DBackpropFilterToV2FusionPass pass({AscendString("Conv3DBackpropFilter")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv3DBackpropFilterV2"));
}

// Test 5: fp32FusionSuccess - FP32 融合成功
TEST_F(Conv3dBpFilterToV2FusionPassTest, fp32FusionSuccess)
{
    auto builder = es::EsGraphBuilder("fp32FusionSuccess");
    auto x = builder.CreateInput(0, "x", DT_FLOAT, FORMAT_NCDHW, {2, 32, 16, 16, 16});
    auto filterSize = builder.CreateInput(1, "filter_size", DT_INT64, FORMAT_ND, {5});
    auto outBackprop = builder.CreateInput(2, "out_backprop", DT_FLOAT, FORMAT_NDHWC, {2, 1, 8, 8, 64});

    auto y = CreateConv3dBpFilterNode(
        builder, "Conv3DBackpropFilter", x, filterSize, outBackprop, {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0},
        {1, 1, 1, 1, 1}, 1, "NCDHW", DT_FLOAT, {2, 1, 4, 4, 32}, FORMAT_NDHWC);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::Conv3DBackpropFilterToV2FusionPass pass({AscendString("Conv3DBackpropFilter")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv3DBackpropFilterV2"));
}

// Test 6: mc62cm12APlatformFail - MC62CM12A 平台不支持
TEST_F(Conv3dBpFilterToV2FusionPassTest, mc62cm12APlatformFail)
{
    SetPlatform("MC62CM12A");
    auto builder = es::EsGraphBuilder("mc62cm12APlatformFail");
    auto x = builder.CreateInput(0, "x", DT_FLOAT16, FORMAT_NCDHW, {2, 32, 16, 16, 16});
    auto filterSize = builder.CreateInput(1, "filter_size", DT_INT64, FORMAT_ND, {5});
    auto outBackprop = builder.CreateInput(2, "out_backprop", DT_FLOAT16, FORMAT_NDHWC, {2, 1, 8, 8, 64});

    auto y = CreateConv3dBpFilterNode(
        builder, "Conv3DBackpropFilter", x, filterSize, outBackprop, {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0},
        {1, 1, 1, 1, 1}, 1, "NCDHW", DT_FLOAT16, {2, 1, 4, 4, 32}, FORMAT_NDHWC);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::Conv3DBackpropFilterToV2FusionPass pass({AscendString("Conv3DBackpropFilter")});
    EXPECT_EQ(pass.Run(graph, ctx), GRAPH_NOT_CHANGED);
    EXPECT_FALSE(CheckNodeExists(graph, "Conv3DBackpropFilterV2"));
}

// Test 7: differentShapeSmallBatch - 不同 shape（小 batch）
TEST_F(Conv3dBpFilterToV2FusionPassTest, differentShapeSmallBatch)
{
    auto builder = es::EsGraphBuilder("differentShapeSmallBatch");
    auto x = builder.CreateInput(0, "x", DT_FLOAT16, FORMAT_NCDHW, {1, 16, 8, 8, 8});
    auto filterSize = builder.CreateInput(1, "filter_size", DT_INT64, FORMAT_ND, {5});
    auto outBackprop = builder.CreateInput(2, "out_backprop", DT_FLOAT16, FORMAT_NDHWC, {1, 1, 4, 4, 32});

    auto y = CreateConv3dBpFilterNode(
        builder, "Conv3DBackpropFilter", x, filterSize, outBackprop, {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0},
        {1, 1, 1, 1, 1}, 1, "NCDHW", DT_FLOAT16, {1, 1, 2, 2, 16}, FORMAT_NDHWC);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::Conv3DBackpropFilterToV2FusionPass pass({AscendString("Conv3DBackpropFilter")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv3DBackpropFilterV2"));
}

// Test 8: noTransposeCase - output 格式是 NCDHW，不需要 Transpose 但仍融合成功
TEST_F(Conv3dBpFilterToV2FusionPassTest, noTransposeCase)
{
    auto builder = es::EsGraphBuilder("noTransposeCase");
    auto x = builder.CreateInput(0, "x", DT_FLOAT16, FORMAT_NCDHW, {2, 32, 16, 16, 16});
    auto filterSize = builder.CreateInput(1, "filter_size", DT_INT64, FORMAT_ND, {5});
    auto outBackprop = builder.CreateInput(2, "out_backprop", DT_FLOAT16, FORMAT_NDHWC, {2, 1, 8, 8, 64});

    auto y = CreateConv3dBpFilterNode(
        builder, "Conv3DBackpropFilter", x, filterSize, outBackprop, {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0},
        {1, 1, 1, 1, 1}, 1, "NCDHW", DT_FLOAT16, {2, 32, 1, 4, 4}, FORMAT_NCDHW);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::Conv3DBackpropFilterToV2FusionPass pass({AscendString("Conv3DBackpropFilter")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv3DBackpropFilterV2"));
}

// Test 9: dhwcFormatCase - DHWCN 格式输出，需要 Transpose
TEST_F(Conv3dBpFilterToV2FusionPassTest, dhwcFormatCase)
{
    auto builder = es::EsGraphBuilder("dhwcFormatCase");
    auto x = builder.CreateInput(0, "x", DT_FLOAT16, FORMAT_NCDHW, {2, 32, 16, 16, 16});
    auto filterSize = builder.CreateInput(1, "filter_size", DT_INT64, FORMAT_ND, {5});
    auto outBackprop = builder.CreateInput(2, "out_backprop", DT_FLOAT16, FORMAT_NDHWC, {2, 1, 8, 8, 64});

    auto y = CreateConv3dBpFilterNode(
        builder, "Conv3DBackpropFilter", x, filterSize, outBackprop, {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0},
        {1, 1, 1, 1, 1}, 1, "NCDHW", DT_FLOAT16, {1, 4, 4, 32, 2}, FORMAT_DHWCN);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::Conv3DBackpropFilterToV2FusionPass pass({AscendString("Conv3DBackpropFilter")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv3DBackpropFilterV2"));
}

// Test 10: shapeLimitExceededNoTranspose - 3D 大 shape，totalCount 超过阈值，不需要 Transpose
// 32核: di=16, cin=256, cout=128, totalCount=524288 > shapeLimit=131072 (32*1024*4)
TEST_F(Conv3dBpFilterToV2FusionPassTest, shapeLimitExceededNoTranspose)
{
    auto builder = es::EsGraphBuilder("shapeLimitExceededNoTranspose");
    auto x = builder.CreateInput(0, "x", DT_FLOAT16, FORMAT_NCDHW, {2, 256, 16, 16, 16});
    auto filterSize = builder.CreateInput(1, "filter_size", DT_INT64, FORMAT_ND, {5});
    auto outBackprop = builder.CreateInput(2, "out_backprop", DT_FLOAT16, FORMAT_NDHWC, {128, 1, 4, 4, 256});

    auto y = CreateConv3dBpFilterNode(
        builder, "Conv3DBackpropFilter", x, filterSize, outBackprop, {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0},
        {1, 1, 1, 1, 1}, 1, "NCDHW", DT_FLOAT16, {128, 1, 4, 4, 256}, FORMAT_NDHWC);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::Conv3DBackpropFilterToV2FusionPass pass({AscendString("Conv3DBackpropFilter")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv3DBackpropFilterV2"));
    EXPECT_FALSE(CheckNodeExists(graph, "Transpose"));
}

// Test 11: diEqualsOneNeedTranspose - 2D 场景(di==1)，totalCount 未超阈值，需要 Transpose
// 32核: di=1, cin=32, cout=2, totalCount=64 <= shapeLimit=16384 (16*1024)
TEST_F(Conv3dBpFilterToV2FusionPassTest, diEqualsOneNeedTranspose)
{
    auto builder = es::EsGraphBuilder("diEqualsOneNeedTranspose");
    auto x = builder.CreateInput(0, "x", DT_FLOAT16, FORMAT_NCDHW, {2, 32, 1, 16, 16});
    auto filterSize = builder.CreateInput(1, "filter_size", DT_INT64, FORMAT_ND, {5});
    auto outBackprop = builder.CreateInput(2, "out_backprop", DT_FLOAT16, FORMAT_NDHWC, {2, 1, 4, 4, 32});

    auto y = CreateConv3dBpFilterNode(
        builder, "Conv3DBackpropFilter", x, filterSize, outBackprop, {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0},
        {1, 1, 1, 1, 1}, 1, "NCDHW", DT_FLOAT16, {2, 1, 4, 4, 32}, FORMAT_NDHWC);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::Conv3DBackpropFilterToV2FusionPass pass({AscendString("Conv3DBackpropFilter")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv3DBackpropFilterV2"));
    EXPECT_TRUE(CheckNodeExists(graph, "Transpose"));
}

// Test 12: diEqualsOneShapeLimitExceeded - 2D 场景(di==1)，totalCount 超过阈值，不需要 Transpose
// 32核: di=1, cin=256, cout=256, totalCount=65536 > shapeLimit=16384 (16*1024)
TEST_F(Conv3dBpFilterToV2FusionPassTest, diEqualsOneShapeLimitExceeded)
{
    auto builder = es::EsGraphBuilder("diEqualsOneShapeLimitExceeded");
    auto x = builder.CreateInput(0, "x", DT_FLOAT16, FORMAT_NCDHW, {256, 256, 1, 16, 16});
    auto filterSize = builder.CreateInput(1, "filter_size", DT_INT64, FORMAT_ND, {5});
    auto outBackprop = builder.CreateInput(2, "out_backprop", DT_FLOAT16, FORMAT_NDHWC, {256, 1, 4, 4, 256});

    auto y = CreateConv3dBpFilterNode(
        builder, "Conv3DBackpropFilter", x, filterSize, outBackprop, {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0},
        {1, 1, 1, 1, 1}, 1, "NCDHW", DT_FLOAT16, {256, 1, 4, 4, 256}, FORMAT_NDHWC);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::Conv3DBackpropFilterToV2FusionPass pass({AscendString("Conv3DBackpropFilter")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv3DBackpropFilterV2"));
    EXPECT_FALSE(CheckNodeExists(graph, "Transpose"));
}

// Test 13: xNdwhcFormatCase - x 输入为 NDHWC 格式，GetInputDepthDim 走 NDHWC 分支
// 32核: x: NDHWC, di=x[1]=16; totalCount=16*32*2=1024 <= shapeLimit=131072 (32*1024*4)
TEST_F(Conv3dBpFilterToV2FusionPassTest, xNdwhcFormatCase)
{
    auto builder = es::EsGraphBuilder("xNdwhcFormatCase");
    auto x = builder.CreateInput(0, "x", DT_FLOAT16, FORMAT_NDHWC, {2, 16, 16, 16, 32});
    auto filterSize = builder.CreateInput(1, "filter_size", DT_INT64, FORMAT_ND, {5});
    auto outBackprop = builder.CreateInput(2, "out_backprop", DT_FLOAT16, FORMAT_NDHWC, {2, 1, 8, 8, 64});

    auto y = CreateConv3dBpFilterNode(
        builder, "Conv3DBackpropFilter", x, filterSize, outBackprop, {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0},
        {1, 1, 1, 1, 1}, 1, "NCDHW", DT_FLOAT16, {2, 1, 4, 4, 32}, FORMAT_NDHWC);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::Conv3DBackpropFilterToV2FusionPass pass({AscendString("Conv3DBackpropFilter")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv3DBackpropFilterV2"));
    EXPECT_TRUE(CheckNodeExists(graph, "Transpose"));
}

// Test 14: xDhwhnFormatCase - x 输入为 DHWCN 格式，GetInputDepthDim 走 DHWCN 分支
// 32核: x: DHWCN, di=x[0]=16; totalCount=16*32*2=1024 <= shapeLimit=131072 (32*1024*4)
TEST_F(Conv3dBpFilterToV2FusionPassTest, xDhwhnFormatCase)
{
    auto builder = es::EsGraphBuilder("xDhwhnFormatCase");
    auto x = builder.CreateInput(0, "x", DT_FLOAT16, FORMAT_DHWCN, {16, 16, 16, 32, 2});
    auto filterSize = builder.CreateInput(1, "filter_size", DT_INT64, FORMAT_ND, {5});
    auto outBackprop = builder.CreateInput(2, "out_backprop", DT_FLOAT16, FORMAT_NDHWC, {2, 1, 8, 8, 64});

    auto y = CreateConv3dBpFilterNode(
        builder, "Conv3DBackpropFilter", x, filterSize, outBackprop, {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0},
        {1, 1, 1, 1, 1}, 1, "NCDHW", DT_FLOAT16, {2, 1, 4, 4, 32}, FORMAT_NDHWC);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::Conv3DBackpropFilterToV2FusionPass pass({AscendString("Conv3DBackpropFilter")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv3DBackpropFilterV2"));
    EXPECT_TRUE(CheckNodeExists(graph, "Transpose"));
}

// Test 15: boundary3DNeedTranspose - 3D 边界值，totalCount 恰好等于 shapeLimit，需要 Transpose
// 32核: di=32, cin=32, cout=128, totalCount=131072 == shapeLimit=131072 (32*1024*4)
TEST_F(Conv3dBpFilterToV2FusionPassTest, boundary3DNeedTranspose)
{
    auto builder = es::EsGraphBuilder("boundary3DNeedTranspose");
    auto x = builder.CreateInput(0, "x", DT_FLOAT16, FORMAT_NCDHW, {128, 32, 32, 16, 16});
    auto filterSize = builder.CreateInput(1, "filter_size", DT_INT64, FORMAT_ND, {5});
    auto outBackprop = builder.CreateInput(2, "out_backprop", DT_FLOAT16, FORMAT_NDHWC, {128, 1, 4, 4, 32});

    auto y = CreateConv3dBpFilterNode(
        builder, "Conv3DBackpropFilter", x, filterSize, outBackprop, {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0},
        {1, 1, 1, 1, 1}, 1, "NCDHW", DT_FLOAT16, {128, 1, 4, 4, 32}, FORMAT_NDHWC);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::Conv3DBackpropFilterToV2FusionPass pass({AscendString("Conv3DBackpropFilter")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv3DBackpropFilterV2"));
    EXPECT_TRUE(CheckNodeExists(graph, "Transpose"));
}

// Test 16: boundary2DNeedTranspose - 2D 边界值，totalCount 恰好等于 shapeLimit，需要 Transpose
// 32核: di=1, cin=128, cout=128, totalCount=16384 == shapeLimit=16384 (16*1024)
TEST_F(Conv3dBpFilterToV2FusionPassTest, boundary2DNeedTranspose)
{
    auto builder = es::EsGraphBuilder("boundary2DNeedTranspose");
    auto x = builder.CreateInput(0, "x", DT_FLOAT16, FORMAT_NCDHW, {128, 128, 1, 16, 16});
    auto filterSize = builder.CreateInput(1, "filter_size", DT_INT64, FORMAT_ND, {5});
    auto outBackprop = builder.CreateInput(2, "out_backprop", DT_FLOAT16, FORMAT_NDHWC, {128, 1, 4, 4, 128});

    auto y = CreateConv3dBpFilterNode(
        builder, "Conv3DBackpropFilter", x, filterSize, outBackprop, {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0},
        {1, 1, 1, 1, 1}, 1, "NCDHW", DT_FLOAT16, {128, 1, 4, 4, 128}, FORMAT_NDHWC);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::Conv3DBackpropFilterToV2FusionPass pass({AscendString("Conv3DBackpropFilter")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv3DBackpropFilterV2"));
    EXPECT_TRUE(CheckNodeExists(graph, "Transpose"));
}

// Test 17: coreCountZeroNoTranspose - coreCount 为 0，不需要 Transpose
TEST_F(Conv3dBpFilterToV2FusionPassTest, coreCountZeroNoTranspose)
{
    PlatformInfoManager::Instance().platform_info_map_["Ascend950"].soc_info.ai_core_cnt = 0;

    auto builder = es::EsGraphBuilder("coreCountZeroNoTranspose");
    auto x = builder.CreateInput(0, "x", DT_FLOAT16, FORMAT_NCDHW, {2, 32, 16, 16, 16});
    auto filterSize = builder.CreateInput(1, "filter_size", DT_INT64, FORMAT_ND, {5});
    auto outBackprop = builder.CreateInput(2, "out_backprop", DT_FLOAT16, FORMAT_NDHWC, {2, 1, 8, 8, 64});

    auto y = CreateConv3dBpFilterNode(
        builder, "Conv3DBackpropFilter", x, filterSize, outBackprop, {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0},
        {1, 1, 1, 1, 1}, 1, "NCDHW", DT_FLOAT16, {2, 1, 4, 4, 32}, FORMAT_NDHWC);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::Conv3DBackpropFilterToV2FusionPass pass({AscendString("Conv3DBackpropFilter")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv3DBackpropFilterV2"));
    EXPECT_FALSE(CheckNodeExists(graph, "Transpose"));
}
