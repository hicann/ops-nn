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
    platformInfo.soc_info.ai_core_cnt = 64;
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

// Test 10: totalCount 在阈值内 - di>1 时 totalCount <= coreCount*32*32*6，需要 Transpose
// coreCount=64; x NCDHW {2,256,32,32,32}: D=32; output NDHWC {2,1,8,8,256}: cin=256, cout=2;
// totalCount=di*cin*cout=32*256*2=16384; 阈值=coreCount*32*32*6=393216; 16384<=393216 → 需要Transpose
TEST_F(Conv3dBpFilterToV2FusionPassTest, largeDivideKCase)
{
    auto builder = es::EsGraphBuilder("largeDivideKCase");
    auto x = builder.CreateInput(0, "x", DT_FLOAT16, FORMAT_NCDHW, {2, 256, 32, 32, 32});
    auto filterSize = builder.CreateInput(1, "filter_size", DT_INT64, FORMAT_ND, {5});
    auto outBackprop = builder.CreateInput(2, "out_backprop", DT_FLOAT16, FORMAT_NDHWC, {2, 1, 16, 16, 512});

    auto y = CreateConv3dBpFilterNode(
        builder, "Conv3DBackpropFilter", x, filterSize, outBackprop, {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0},
        {1, 1, 1, 1, 1}, 1, "NCDHW", DT_FLOAT16, {2, 1, 8, 8, 256}, FORMAT_NDHWC);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::Conv3DBackpropFilterToV2FusionPass pass({AscendString("Conv3DBackpropFilter")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv3DBackpropFilterV2"));
    EXPECT_TRUE(CheckNodeExists(graph, "Transpose"));
}

// Test 11: x DHWCN format, output NDHWC - x 为 DHWCN 格式，验证 GetInputDepthDim DHWCN 分支
// coreCount=64; x DHWCN {4,8,8,16,2}: D=4; output NDHWC {2,1,4,4,16}: cin=16, cout=2;
// totalCount=di*cin*cout=4*16*2=128; 阈值=coreCount*32*32*6=393216; 128<=393216 → 需要Transpose
TEST_F(Conv3dBpFilterToV2FusionPassTest, xDhwcnFormatNeedTranspose)
{
    auto builder = es::EsGraphBuilder("xDhwcnFormatNeedTranspose");
    auto x = builder.CreateInput(0, "x", DT_FLOAT16, FORMAT_DHWCN, {4, 8, 8, 16, 2});
    auto filterSize = builder.CreateInput(1, "filter_size", DT_INT64, FORMAT_ND, {5});
    auto outBackprop = builder.CreateInput(2, "out_backprop", DT_FLOAT16, FORMAT_NDHWC, {2, 1, 4, 4, 16});

    auto y = CreateConv3dBpFilterNode(
        builder, "Conv3DBackpropFilter", x, filterSize, outBackprop, {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0},
        {1, 1, 1, 1, 1}, 1, "NCDHW", DT_FLOAT16, {2, 1, 4, 4, 16}, FORMAT_NDHWC);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::Conv3DBackpropFilterToV2FusionPass pass({AscendString("Conv3DBackpropFilter")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv3DBackpropFilterV2"));
    EXPECT_TRUE(CheckNodeExists(graph, "Transpose"));
}

// Test 12: x NDHWC format, output DHWCN - x 为 NDHWC 格式，验证 GetInputDepthDim NDHWC 分支
// coreCount=64; x NDHWC {2,16,8,8,32}: D=16; output DHWCN {8,8,16,32,2}: cin=32, cout=2;
// totalCount=di*cin*cout=16*32*2=1024; 阈值=coreCount*32*32*6=393216; 1024<=393216 → 需要Transpose
TEST_F(Conv3dBpFilterToV2FusionPassTest, xNdhwcFormatOutputDhwcn)
{
    auto builder = es::EsGraphBuilder("xNdhwcFormatOutputDhwcn");
    auto x = builder.CreateInput(0, "x", DT_FLOAT16, FORMAT_NDHWC, {2, 16, 8, 8, 32});
    auto filterSize = builder.CreateInput(1, "filter_size", DT_INT64, FORMAT_ND, {5});
    auto outBackprop = builder.CreateInput(2, "out_backprop", DT_FLOAT16, FORMAT_NDHWC, {2, 1, 4, 4, 64});

    auto y = CreateConv3dBpFilterNode(
        builder, "Conv3DBackpropFilter", x, filterSize, outBackprop, {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0},
        {1, 1, 1, 1, 1}, 1, "NCDHW", DT_FLOAT16, {8, 8, 16, 32, 2}, FORMAT_DHWCN);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::Conv3DBackpropFilterToV2FusionPass pass({AscendString("Conv3DBackpropFilter")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv3DBackpropFilterV2"));
    EXPECT_TRUE(CheckNodeExists(graph, "Transpose"));
}

// Test 13: totalCount 在阈值内 - di>1 时 totalCount <= coreCount*32*32*6，需要 Transpose
// coreCount=64; x NCDHW {2,512,16,16,16}: D=16; output NDHWC {2,1,8,8,512}: cin=512, cout=2;
// totalCount=di*cin*cout=16*512*2=16384; 阈值=coreCount*32*32*6=393216; 16384<=393216 → 需要Transpose
TEST_F(Conv3dBpFilterToV2FusionPassTest, largeCinNeedTranspose)
{
    auto builder = es::EsGraphBuilder("largeCinNeedTranspose");
    auto x = builder.CreateInput(0, "x", DT_FLOAT16, FORMAT_NCDHW, {2, 512, 16, 16, 16});
    auto filterSize = builder.CreateInput(1, "filter_size", DT_INT64, FORMAT_ND, {5});
    auto outBackprop = builder.CreateInput(2, "out_backprop", DT_FLOAT16, FORMAT_NDHWC, {2, 1, 8, 8, 1024});

    auto y = CreateConv3dBpFilterNode(
        builder, "Conv3DBackpropFilter", x, filterSize, outBackprop, {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0},
        {1, 1, 1, 1, 1}, 1, "NCDHW", DT_FLOAT16, {2, 1, 8, 8, 512}, FORMAT_NDHWC);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::Conv3DBackpropFilterToV2FusionPass pass({AscendString("Conv3DBackpropFilter")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv3DBackpropFilterV2"));
    EXPECT_TRUE(CheckNodeExists(graph, "Transpose"));
}

// Test 14: totalCount 在阈值内 - di>1 时 totalCount <= coreCount*32*32*6，需要 Transpose
// coreCount=64; x NDHWC {2,256,16,16,16}: D=256; output NDHWC {2,1,8,8,64}: cin=64, cout=2;
// totalCount=di*cin*cout=256*64*2=32768; 阈值=coreCount*32*32*6=393216; 32768<=393216 → 需要Transpose
TEST_F(Conv3dBpFilterToV2FusionPassTest, xNdhwcLargeDepthNeedTranspose)
{
    auto builder = es::EsGraphBuilder("xNdhwcLargeDepthNeedTranspose");
    auto x = builder.CreateInput(0, "x", DT_FLOAT16, FORMAT_NDHWC, {2, 256, 16, 16, 16});
    auto filterSize = builder.CreateInput(1, "filter_size", DT_INT64, FORMAT_ND, {5});
    auto outBackprop = builder.CreateInput(2, "out_backprop", DT_FLOAT16, FORMAT_NDHWC, {2, 1, 8, 8, 128});

    auto y = CreateConv3dBpFilterNode(
        builder, "Conv3DBackpropFilter", x, filterSize, outBackprop, {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0},
        {1, 1, 1, 1, 1}, 1, "NCDHW", DT_FLOAT16, {2, 1, 8, 8, 64}, FORMAT_NDHWC);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::Conv3DBackpropFilterToV2FusionPass pass({AscendString("Conv3DBackpropFilter")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv3DBackpropFilterV2"));
    EXPECT_TRUE(CheckNodeExists(graph, "Transpose"));
}

// Test 15: totalCount 在阈值内 - di>1 时 totalCount <= coreCount*32*32*6，需要 Transpose
// coreCount=64; x NCDHW {2,32,96,16,16}: D=96; output NDHWC {2,1,8,8,32}: cin=32, cout=2;
// totalCount=di*cin*cout=96*32*2=6144; 阈值=coreCount*32*32*6=393216; 6144<=393216 → 需要Transpose
TEST_F(Conv3dBpFilterToV2FusionPassTest, cond1BoundaryNeedTranspose)
{
    auto builder = es::EsGraphBuilder("cond1BoundaryNeedTranspose");
    auto x = builder.CreateInput(0, "x", DT_FLOAT16, FORMAT_NCDHW, {2, 32, 96, 16, 16});
    auto filterSize = builder.CreateInput(1, "filter_size", DT_INT64, FORMAT_ND, {5});
    auto outBackprop = builder.CreateInput(2, "out_backprop", DT_FLOAT16, FORMAT_NDHWC, {2, 1, 8, 8, 64});

    auto y = CreateConv3dBpFilterNode(
        builder, "Conv3DBackpropFilter", x, filterSize, outBackprop, {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0},
        {1, 1, 1, 1, 1}, 1, "NCDHW", DT_FLOAT16, {2, 1, 8, 8, 32}, FORMAT_NDHWC);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::Conv3DBackpropFilterToV2FusionPass pass({AscendString("Conv3DBackpropFilter")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv3DBackpropFilterV2"));
    EXPECT_TRUE(CheckNodeExists(graph, "Transpose"));
}

// Test 16: totalCount 在阈值内 - di>1 时 totalCount <= coreCount*32*32*6，需要 Transpose
// coreCount=64; x NCDHW {2,32,192,16,16}: D=192; output NDHWC {2,1,8,8,32}: cin=32, cout=2;
// totalCount=di*cin*cout=192*32*2=12288; 阈值=coreCount*32*32*6=393216; 12288<=393216 → 需要Transpose
TEST_F(Conv3dBpFilterToV2FusionPassTest, cond2BoundaryNeedTranspose)
{
    auto builder = es::EsGraphBuilder("cond2BoundaryNeedTranspose");
    auto x = builder.CreateInput(0, "x", DT_FLOAT16, FORMAT_NCDHW, {2, 32, 192, 16, 16});
    auto filterSize = builder.CreateInput(1, "filter_size", DT_INT64, FORMAT_ND, {5});
    auto outBackprop = builder.CreateInput(2, "out_backprop", DT_FLOAT16, FORMAT_NDHWC, {2, 1, 8, 8, 64});

    auto y = CreateConv3dBpFilterNode(
        builder, "Conv3DBackpropFilter", x, filterSize, outBackprop, {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0},
        {1, 1, 1, 1, 1}, 1, "NCDHW", DT_FLOAT16, {2, 1, 8, 8, 32}, FORMAT_NDHWC);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::Conv3DBackpropFilterToV2FusionPass pass({AscendString("Conv3DBackpropFilter")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv3DBackpropFilterV2"));
    EXPECT_TRUE(CheckNodeExists(graph, "Transpose"));
}

// Test 17: totalCount 在阈值内 - di>1 时 totalCount <= coreCount*32*32*6，需要 Transpose
// coreCount=64; x NCDHW {2,32,193,16,16}: D=193; output NDHWC {2,1,8,8,32}: cin=32, cout=2;
// totalCount=di*cin*cout=193*32*2=12352; 阈值=coreCount*32*32*6=393216; 12352<=393216 → 需要Transpose
TEST_F(Conv3dBpFilterToV2FusionPassTest, midDepthNeedTranspose)
{
    auto builder = es::EsGraphBuilder("midDepthNeedTranspose");
    auto x = builder.CreateInput(0, "x", DT_FLOAT16, FORMAT_NCDHW, {2, 32, 193, 16, 16});
    auto filterSize = builder.CreateInput(1, "filter_size", DT_INT64, FORMAT_ND, {5});
    auto outBackprop = builder.CreateInput(2, "out_backprop", DT_FLOAT16, FORMAT_NDHWC, {2, 1, 8, 8, 64});

    auto y = CreateConv3dBpFilterNode(
        builder, "Conv3DBackpropFilter", x, filterSize, outBackprop, {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0},
        {1, 1, 1, 1, 1}, 1, "NCDHW", DT_FLOAT16, {2, 1, 8, 8, 32}, FORMAT_NDHWC);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::Conv3DBackpropFilterToV2FusionPass pass({AscendString("Conv3DBackpropFilter")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv3DBackpropFilterV2"));
    EXPECT_TRUE(CheckNodeExists(graph, "Transpose"));
}

// Test 18: totalCount 在阈值内 - di>1 时 totalCount <= coreCount*32*32*6，需要 Transpose
// coreCount=64; x NCDHW {2,32,33,16,16}: D=33; output NDHWC {2,1,8,8,32}: cin=32, cout=2;
// totalCount=di*cin*cout=33*32*2=2112; 阈值=coreCount*32*32*6=393216; 2112<=393216 → 需要Transpose
TEST_F(Conv3dBpFilterToV2FusionPassTest, smallDepthNeedTranspose)
{
    auto builder = es::EsGraphBuilder("smallDepthNeedTranspose");
    auto x = builder.CreateInput(0, "x", DT_FLOAT16, FORMAT_NCDHW, {2, 32, 33, 16, 16});
    auto filterSize = builder.CreateInput(1, "filter_size", DT_INT64, FORMAT_ND, {5});
    auto outBackprop = builder.CreateInput(2, "out_backprop", DT_FLOAT16, FORMAT_NDHWC, {2, 1, 8, 8, 64});

    auto y = CreateConv3dBpFilterNode(
        builder, "Conv3DBackpropFilter", x, filterSize, outBackprop, {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0},
        {1, 1, 1, 1, 1}, 1, "NCDHW", DT_FLOAT16, {2, 1, 8, 8, 32}, FORMAT_NDHWC);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::Conv3DBackpropFilterToV2FusionPass pass({AscendString("Conv3DBackpropFilter")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv3DBackpropFilterV2"));
    EXPECT_TRUE(CheckNodeExists(graph, "Transpose"));
}

// Test 19: totalCount 在阈值内 - di==1 时 totalCount <= coreCount*32*32/2，需要 Transpose
// coreCount=64; x NCDHW {2,33,1,16,16}: D=1; output NDHWC {2,1,8,8,33}: cin=33, cout=2;
// totalCount=di*cin*cout=1*33*2=66; 阈值=coreCount*32*32/2=32768; 66<=32768 → 需要Transpose
TEST_F(Conv3dBpFilterToV2FusionPassTest, ceilRoundingNeedTranspose)
{
    auto builder = es::EsGraphBuilder("ceilRoundingNeedTranspose");
    auto x = builder.CreateInput(0, "x", DT_FLOAT16, FORMAT_NCDHW, {2, 33, 1, 16, 16});
    auto filterSize = builder.CreateInput(1, "filter_size", DT_INT64, FORMAT_ND, {5});
    auto outBackprop = builder.CreateInput(2, "out_backprop", DT_FLOAT16, FORMAT_NDHWC, {2, 1, 8, 8, 64});

    auto y = CreateConv3dBpFilterNode(
        builder, "Conv3DBackpropFilter", x, filterSize, outBackprop, {1, 1, 1, 1, 1}, {0, 0, 0, 0, 0, 0},
        {1, 1, 1, 1, 1}, 1, "NCDHW", DT_FLOAT16, {2, 1, 8, 8, 33}, FORMAT_NDHWC);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::Conv3DBackpropFilterToV2FusionPass pass({AscendString("Conv3DBackpropFilter")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv3DBackpropFilterV2"));
    EXPECT_TRUE(CheckNodeExists(graph, "Transpose"));
}
