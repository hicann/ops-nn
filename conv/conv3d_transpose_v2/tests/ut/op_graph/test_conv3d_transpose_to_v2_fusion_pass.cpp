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
#include "../../../op_graph/fusion_pass/conv3d_transpose_to_v2_fusion_pass.h"

using namespace ge;
using namespace fe;
using namespace fusion;
using namespace ops::ConvBackpropFusionUtils;

namespace {

constexpr int64_t AI_CORE_CNT = 64;
constexpr int64_t INPUT_SIZE_DIM = 5;
int64_t DEFAULT_IMPL_MODE = 0x1;

void SetPlatform(const std::string& soc)
{
    PlatformInfo platformInfo;
    OptionalInfo optionalInfo;
    platformInfo.soc_info.ai_core_cnt = AI_CORE_CNT;
    platformInfo.str_info.short_soc_version = soc;
    optionalInfo.soc_version = soc;
    PlatformInfoManager::Instance().platform_info_map_[soc] = platformInfo;
    PlatformInfoManager::Instance().SetOptionalCompilationInfo(optionalInfo);
}

es::EsTensorHolder CreateConv3DTransposeNode(es::EsGraphBuilder& builder, const char* opType,
                                             const es::EsTensorHolder& inputSize, const es::EsTensorHolder& x,
                                             const es::EsTensorHolder& filter, const es::EsTensorHolder& bias,
                                             std::vector<int64_t> strides, std::vector<int64_t> pads,
                                             std::vector<int64_t> dilations, int64_t groups,
                                             const std::string& dataFormat, DataType outDtype,
                                             const std::vector<int64_t>& outShape, Format outFormat)
{
    auto* graph = builder.GetCGraphBuilder()->GetGraph();
    auto node = es::CompliantNodeBuilder(graph)
                    .OpType(opType)
                    .Name(opType)
                    .IrDefInputs({{"input_size", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                  {"x", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                  {"filter", es::CompliantNodeBuilder::kEsIrInputRequired, ""},
                                  {"bias", es::CompliantNodeBuilder::kEsIrInputOptional, ""}})
                    .IrDefOutputs({{"y", es::CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                    .InstanceOutputDataType("y", outDtype)
                    .InstanceOutputShape("y", outShape)
                    .InstanceOutputFormat("y", outFormat)
                    .Build();

    es::AddEdgeAndUpdatePeerDesc(*graph, *inputSize.GetProducer(), inputSize.GetProducerOutIndex(), node, 0);
    es::AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(), node, 1);
    es::AddEdgeAndUpdatePeerDesc(*graph, *filter.GetProducer(), filter.GetProducerOutIndex(), node, 2);
    es::AddEdgeAndUpdatePeerDesc(*graph, *bias.GetProducer(), bias.GetProducerOutIndex(), node, 3);

    TensorDesc inputSizeDesc, xDesc, filterDesc, biasDesc;
    inputSize.GetProducer()->GetOutputDesc(inputSize.GetProducerOutIndex(), inputSizeDesc);
    x.GetProducer()->GetOutputDesc(x.GetProducerOutIndex(), xDesc);
    filter.GetProducer()->GetOutputDesc(filter.GetProducerOutIndex(), filterDesc);
    bias.GetProducer()->GetOutputDesc(bias.GetProducerOutIndex(), biasDesc);
    node.UpdateInputDesc(0, inputSizeDesc);
    node.UpdateInputDesc(1, xDesc);
    node.UpdateInputDesc(2, filterDesc);
    node.UpdateInputDesc(3, biasDesc);

    node.SetAttr("strides", strides);
    node.SetAttr("pads", pads);
    node.SetAttr("dilations", dilations);
    node.SetAttr("groups", groups);
    AscendString fmt = dataFormat.c_str();
    node.SetAttr("data_format", fmt);
    node.SetAttr("_op_impl_mode_enum", DEFAULT_IMPL_MODE);

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

class Conv3DTransposeToV2FusionPassTest : public testing::Test {
protected:
    void SetUp() override { SetPlatform("Ascend950"); }
};

// Test 1: patternTest - FP16 basic fusion success
TEST_F(Conv3DTransposeToV2FusionPassTest, patternTest)
{
    auto builder = es::EsGraphBuilder("patternTest");
    auto inputSize = builder.CreateInput(0, "input_size", DT_INT32, FORMAT_ND, {INPUT_SIZE_DIM});
    auto x = builder.CreateInput(1, "x", DT_FLOAT16, FORMAT_NCDHW, {2, 32, 16, 16, 16});
    auto filter = builder.CreateInput(2, "filter", DT_FLOAT16, FORMAT_NCDHW, {32, 64, 3, 3, 3});
    auto bias = builder.CreateInput(3, "bias", DT_FLOAT16, FORMAT_ND, {64});

    auto y = CreateConv3DTransposeNode(builder, "Conv3DTranspose", inputSize, x, filter, bias, {1, 1, 1, 1, 1},
                                       {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NCDHW", DT_FLOAT16, {2, 64, 18, 18, 18},
                                       FORMAT_NCDHW);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::Conv3DTransposeToV2FusionPass pass({AscendString("Conv3DTranspose")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv3DTransposeV2"));
}

// Test 2: unsupportedPlatformFail - Ascend910_93 not supported
TEST_F(Conv3DTransposeToV2FusionPassTest, unsupportedPlatformFail)
{
    SetPlatform("Ascend910_93");
    auto builder = es::EsGraphBuilder("unsupportedPlatformFail");
    auto inputSize = builder.CreateInput(0, "input_size", DT_INT32, FORMAT_ND, {INPUT_SIZE_DIM});
    auto x = builder.CreateInput(1, "x", DT_FLOAT16, FORMAT_NCDHW, {2, 32, 16, 16, 16});
    auto filter = builder.CreateInput(2, "filter", DT_FLOAT16, FORMAT_NCDHW, {32, 64, 3, 3, 3});
    auto bias = builder.CreateInput(3, "bias", DT_FLOAT16, FORMAT_ND, {64});

    auto y = CreateConv3DTransposeNode(builder, "Conv3DTranspose", inputSize, x, filter, bias, {1, 1, 1, 1, 1},
                                       {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NCDHW", DT_FLOAT16, {2, 64, 18, 18, 18},
                                       FORMAT_NCDHW);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::Conv3DTransposeToV2FusionPass pass({AscendString("Conv3DTranspose")});
    EXPECT_EQ(pass.Run(graph, ctx), GRAPH_NOT_CHANGED);
    EXPECT_FALSE(CheckNodeExists(graph, "Conv3DTransposeV2"));
}

// Test 3: bf16FusionSuccess
TEST_F(Conv3DTransposeToV2FusionPassTest, bf16FusionSuccess)
{
    auto builder = es::EsGraphBuilder("bf16FusionSuccess");
    auto inputSize = builder.CreateInput(0, "input_size", DT_INT32, FORMAT_ND, {INPUT_SIZE_DIM});
    auto x = builder.CreateInput(1, "x", DT_BF16, FORMAT_NCDHW, {2, 32, 16, 16, 16});
    auto filter = builder.CreateInput(2, "filter", DT_BF16, FORMAT_NCDHW, {32, 64, 3, 3, 3});
    auto bias = builder.CreateInput(3, "bias", DT_BF16, FORMAT_ND, {64});

    auto y = CreateConv3DTransposeNode(builder, "Conv3DTranspose", inputSize, x, filter, bias, {1, 1, 1, 1, 1},
                                       {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NCDHW", DT_BF16, {2, 64, 18, 18, 18},
                                       FORMAT_NCDHW);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::Conv3DTransposeToV2FusionPass pass({AscendString("Conv3DTranspose")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv3DTransposeV2"));
}

// Test 4: fp32FusionSuccess
TEST_F(Conv3DTransposeToV2FusionPassTest, fp32FusionSuccess)
{
    auto builder = es::EsGraphBuilder("fp32FusionSuccess");
    auto inputSize = builder.CreateInput(0, "input_size", DT_INT32, FORMAT_ND, {INPUT_SIZE_DIM});
    auto x = builder.CreateInput(1, "x", DT_FLOAT, FORMAT_NCDHW, {2, 32, 16, 16, 16});
    auto filter = builder.CreateInput(2, "filter", DT_FLOAT, FORMAT_NCDHW, {32, 64, 3, 3, 3});
    auto bias = builder.CreateInput(3, "bias", DT_FLOAT, FORMAT_ND, {64});

    auto y = CreateConv3DTransposeNode(builder, "Conv3DTranspose", inputSize, x, filter, bias, {1, 1, 1, 1, 1},
                                       {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NCDHW", DT_FLOAT, {2, 64, 18, 18, 18},
                                       FORMAT_NCDHW);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::Conv3DTransposeToV2FusionPass pass({AscendString("Conv3DTranspose")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv3DTransposeV2"));
}

// Test 5: transposeSuccess - filter NCDHW fp16, all conditions met, Transpose node created
TEST_F(Conv3DTransposeToV2FusionPassTest, transposeSuccess)
{
    auto builder = es::EsGraphBuilder("transposeSuccess");
    auto inputSize = builder.CreateInput(0, "input_size", DT_INT32, FORMAT_ND, {INPUT_SIZE_DIM});
    auto x = builder.CreateInput(1, "x", DT_FLOAT16, FORMAT_NCDHW, {2, 64, 16, 16, 16});
    auto filter = builder.CreateInput(2, "filter", DT_FLOAT16, FORMAT_NCDHW, {64, 64, 3, 3, 3});
    auto bias = builder.CreateInput(3, "bias", DT_FLOAT16, FORMAT_ND, {64});

    auto y = CreateConv3DTransposeNode(builder, "Conv3DTranspose", inputSize, x, filter, bias, {1, 1, 1, 1, 1},
                                       {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NCDHW", DT_FLOAT16, {2, 64, 18, 18, 18},
                                       FORMAT_NCDHW);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::Conv3DTransposeToV2FusionPass pass({AscendString("Conv3DTranspose")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv3DTransposeV2"));
    EXPECT_TRUE(CheckNodeExists(graph, "Transpose"));
}

// Test 6: noTranspose_filterNotNCDHW - filter format is NDHWC
TEST_F(Conv3DTransposeToV2FusionPassTest, noTranspose_filterNotNCDHW)
{
    auto builder = es::EsGraphBuilder("noTranspose_filterNotNCDHW");
    auto inputSize = builder.CreateInput(0, "input_size", DT_INT32, FORMAT_ND, {INPUT_SIZE_DIM});
    auto x = builder.CreateInput(1, "x", DT_FLOAT16, FORMAT_NDHWC, {2, 16, 16, 16, 32});
    auto filter = builder.CreateInput(2, "filter", DT_FLOAT16, FORMAT_NDHWC, {32, 3, 3, 3, 64});
    auto bias = builder.CreateInput(3, "bias", DT_FLOAT16, FORMAT_ND, {64});

    auto y = CreateConv3DTransposeNode(builder, "Conv3DTranspose", inputSize, x, filter, bias, {1, 1, 1, 1, 1},
                                       {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NDHWC", DT_FLOAT16, {2, 18, 18, 18, 64},
                                       FORMAT_NDHWC);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::Conv3DTransposeToV2FusionPass pass({AscendString("Conv3DTranspose")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv3DTransposeV2"));
    EXPECT_FALSE(CheckNodeExists(graph, "Transpose"));
}

// Test 7: noTranspose_groupsNotOne - groups > 1
TEST_F(Conv3DTransposeToV2FusionPassTest, noTranspose_groupsNotOne)
{
    auto builder = es::EsGraphBuilder("noTranspose_groupsNotOne");
    auto inputSize = builder.CreateInput(0, "input_size", DT_INT32, FORMAT_ND, {INPUT_SIZE_DIM});
    auto x = builder.CreateInput(1, "x", DT_FLOAT16, FORMAT_NCDHW, {2, 32, 16, 16, 16});
    auto filter = builder.CreateInput(2, "filter", DT_FLOAT16, FORMAT_NCDHW, {32, 64, 3, 3, 3});
    auto bias = builder.CreateInput(3, "bias", DT_FLOAT16, FORMAT_ND, {64});

    auto y = CreateConv3DTransposeNode(builder, "Conv3DTranspose", inputSize, x, filter, bias, {1, 1, 1, 1, 1},
                                       {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 2, "NCDHW", DT_FLOAT16, {2, 64, 18, 18, 18},
                                       FORMAT_NCDHW);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::Conv3DTransposeToV2FusionPass pass({AscendString("Conv3DTranspose")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv3DTransposeV2"));
    EXPECT_FALSE(CheckNodeExists(graph, "Transpose"));
}

// Test 8: noTranspose_filterDtypeNotSupported - filter dtype is bf16, only fp16/fp32 trigger transpose
TEST_F(Conv3DTransposeToV2FusionPassTest, noTranspose_filterDtypeNotSupported)
{
    auto builder = es::EsGraphBuilder("noTranspose_filterDtypeNotSupported");
    auto inputSize = builder.CreateInput(0, "input_size", DT_INT32, FORMAT_ND, {INPUT_SIZE_DIM});
    auto x = builder.CreateInput(1, "x", DT_BF16, FORMAT_NCDHW, {2, 32, 16, 16, 16});
    auto filter = builder.CreateInput(2, "filter", DT_BF16, FORMAT_NCDHW, {32, 32, 3, 3, 3});
    auto bias = builder.CreateInput(3, "bias", DT_FLOAT, FORMAT_ND, {32});

    auto y = CreateConv3DTransposeNode(builder, "Conv3DTranspose", inputSize, x, filter, bias, {1, 1, 1, 1, 1},
                                       {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NCDHW", DT_BF16, {2, 32, 18, 18, 18},
                                       FORMAT_NCDHW);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::Conv3DTransposeToV2FusionPass pass({AscendString("Conv3DTranspose")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv3DTransposeV2"));
    EXPECT_FALSE(CheckNodeExists(graph, "Transpose"));
}

// Test 9: noTranspose_kernelVolumeTooSmall - dk*hk*wk <= 1
TEST_F(Conv3DTransposeToV2FusionPassTest, noTranspose_kernelVolumeTooSmall)
{
    auto builder = es::EsGraphBuilder("noTranspose_kernelVolumeTooSmall");
    auto inputSize = builder.CreateInput(0, "input_size", DT_INT32, FORMAT_ND, {INPUT_SIZE_DIM});
    auto x = builder.CreateInput(1, "x", DT_FLOAT16, FORMAT_NCDHW, {2, 64, 16, 16, 16});
    auto filter = builder.CreateInput(2, "filter", DT_FLOAT16, FORMAT_NCDHW, {32, 64, 1, 1, 1});
    auto bias = builder.CreateInput(3, "bias", DT_FLOAT16, FORMAT_ND, {64});

    auto y = CreateConv3DTransposeNode(builder, "Conv3DTranspose", inputSize, x, filter, bias, {1, 1, 1, 1, 1},
                                       {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NCDHW", DT_FLOAT16, {2, 32, 16, 16, 16},
                                       FORMAT_NCDHW);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::Conv3DTransposeToV2FusionPass pass({AscendString("Conv3DTranspose")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv3DTransposeV2"));
    EXPECT_FALSE(CheckNodeExists(graph, "Transpose"));
}

// Test 10: noTranspose_cinTooSmall - cin=8 < 32 and cin != 16
TEST_F(Conv3DTransposeToV2FusionPassTest, noTranspose_cinTooSmall)
{
    auto builder = es::EsGraphBuilder("noTranspose_cinTooSmall");
    auto inputSize = builder.CreateInput(0, "input_size", DT_INT32, FORMAT_ND, {INPUT_SIZE_DIM});
    auto x = builder.CreateInput(1, "x", DT_FLOAT16, FORMAT_NCDHW, {2, 8, 16, 16, 16});
    auto filter = builder.CreateInput(2, "filter", DT_FLOAT16, FORMAT_NCDHW, {12, 8, 3, 3, 3});
    auto bias = builder.CreateInput(3, "bias", DT_FLOAT16, FORMAT_ND, {8});

    auto y = CreateConv3DTransposeNode(builder, "Conv3DTranspose", inputSize, x, filter, bias, {1, 1, 1, 1, 1},
                                       {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NCDHW", DT_FLOAT16, {2, 12, 18, 18, 18},
                                       FORMAT_NCDHW);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::Conv3DTransposeToV2FusionPass pass({AscendString("Conv3DTranspose")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv3DTransposeV2"));
    EXPECT_FALSE(CheckNodeExists(graph, "Transpose"));
}

// Test 11: noTranspose_cinLeqHkWk - cin=32 <= hk*wk=36
TEST_F(Conv3DTransposeToV2FusionPassTest, noTranspose_cinLeqHkWk)
{
    auto builder = es::EsGraphBuilder("noTranspose_cinLeqHkWk");
    auto inputSize = builder.CreateInput(0, "input_size", DT_INT32, FORMAT_ND, {INPUT_SIZE_DIM});
    auto x = builder.CreateInput(1, "x", DT_FLOAT16, FORMAT_NCDHW, {2, 32, 16, 16, 16});
    auto filter = builder.CreateInput(2, "filter", DT_FLOAT16, FORMAT_NCDHW, {64, 32, 3, 6, 6});
    auto bias = builder.CreateInput(3, "bias", DT_FLOAT16, FORMAT_ND, {32});

    auto y = CreateConv3DTransposeNode(builder, "Conv3DTranspose", inputSize, x, filter, bias, {1, 1, 1, 1, 1},
                                       {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NCDHW", DT_FLOAT16, {2, 64, 18, 21, 21},
                                       FORMAT_NCDHW);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::Conv3DTransposeToV2FusionPass pass({AscendString("Conv3DTranspose")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv3DTransposeV2"));
    EXPECT_FALSE(CheckNodeExists(graph, "Transpose"));
}

// Test 12: noTranspose_coutCinRatioOutOfRange - cout/cin ratio > 1.5
TEST_F(Conv3DTransposeToV2FusionPassTest, noTranspose_coutCinRatioOutOfRange)
{
    auto builder = es::EsGraphBuilder("noTranspose_coutCinRatioOutOfRange");
    auto inputSize = builder.CreateInput(0, "input_size", DT_INT32, FORMAT_ND, {INPUT_SIZE_DIM});
    auto x = builder.CreateInput(1, "x", DT_FLOAT16, FORMAT_NCDHW, {2, 32, 16, 16, 16});
    auto filter = builder.CreateInput(2, "filter", DT_FLOAT16, FORMAT_NCDHW, {64, 32, 3, 3, 3});
    auto bias = builder.CreateInput(3, "bias", DT_FLOAT16, FORMAT_ND, {32});

    auto y = CreateConv3DTransposeNode(builder, "Conv3DTranspose", inputSize, x, filter, bias, {1, 1, 1, 1, 1},
                                       {0, 0, 0, 0, 0, 0}, {1, 1, 1, 1, 1}, 1, "NCDHW", DT_FLOAT16, {2, 64, 18, 18, 18},
                                       FORMAT_NCDHW);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::Conv3DTransposeToV2FusionPass pass({AscendString("Conv3DTranspose")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv3DTransposeV2"));
    EXPECT_FALSE(CheckNodeExists(graph, "Transpose"));
}
