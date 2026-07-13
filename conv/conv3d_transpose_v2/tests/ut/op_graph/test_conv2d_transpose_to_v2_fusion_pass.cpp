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
#include "../../../op_graph/fusion_pass/conv2d_transpose_to_v2_fusion_pass.h"

using namespace ge;
using namespace ge::es;
using namespace fe;
using namespace fusion;

namespace {

constexpr int64_t AI_CORE_CNT = 64;
constexpr int64_t INPUT_SIZE_DIM = 4;
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

EsTensorHolder CreateConv2DTransposeNode(EsGraphBuilder& builder, const char* opType, const EsTensorHolder& inputSize,
                                         const EsTensorHolder& x, const EsTensorHolder& filter,
                                         const EsTensorHolder& bias, std::vector<int64_t> strides,
                                         std::vector<int64_t> pads, std::vector<int64_t> dilations, int64_t groups,
                                         const std::string& dataFormat, DataType outDtype,
                                         const std::vector<int64_t>& outShape, Format outFormat)
{
    auto* graph = builder.GetCGraphBuilder()->GetGraph();
    auto node = CompliantNodeBuilder(graph)
                    .OpType(opType)
                    .Name(opType)
                    .IrDefInputs({{"input_size", CompliantNodeBuilder::kEsIrInputRequired, ""},
                                  {"x", CompliantNodeBuilder::kEsIrInputRequired, ""},
                                  {"filter", CompliantNodeBuilder::kEsIrInputRequired, ""},
                                  {"bias", CompliantNodeBuilder::kEsIrInputOptional, ""}})
                    .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                    .InstanceOutputDataType("y", outDtype)
                    .InstanceOutputShape("y", outShape)
                    .InstanceOutputFormat("y", outFormat)
                    .Build();

    AddEdgeAndUpdatePeerDesc(*graph, *inputSize.GetProducer(), inputSize.GetProducerOutIndex(), node, 0);
    AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(), node, 1);
    AddEdgeAndUpdatePeerDesc(*graph, *filter.GetProducer(), filter.GetProducerOutIndex(), node, 2);
    AddEdgeAndUpdatePeerDesc(*graph, *bias.GetProducer(), bias.GetProducerOutIndex(), node, 3);

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
    std::vector<int64_t> defaultOutputPadding = {0, 0, 0, 0};
    node.SetAttr("output_padding", defaultOutputPadding);
    int64_t defaultOffsetX = 0;
    node.SetAttr("offset_x", defaultOffsetX);
    node.SetAttr("_op_impl_mode_enum", DEFAULT_IMPL_MODE);

    return EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 0));
}

bool CheckNodeExists(GraphPtr& graph, const std::string& type)
{
    for (auto node : graph->GetAllNodes()) {
        AscendString nodeType;
        node.GetType(nodeType);
        if (nodeType.GetString() == type) {
            return true;
        }
    }
    return false;
}

} // namespace

class Conv2DTransposeToV2FusionPassTest : public testing::Test {
protected:
    void SetUp() override { SetPlatform("Ascend950"); }
};

// Test 1: patternTest - FP16 NCHW basic fusion success
TEST_F(Conv2DTransposeToV2FusionPassTest, patternTest)
{
    auto builder = EsGraphBuilder("patternTest");
    auto inputSize = builder.CreateInput(0, "input_size", DT_INT32, FORMAT_ND, {INPUT_SIZE_DIM});
    auto x = builder.CreateInput(1, "x", DT_FLOAT16, FORMAT_NCHW, {2, 32, 16, 16});
    auto filter = builder.CreateInput(2, "filter", DT_FLOAT16, FORMAT_NCHW, {32, 64, 3, 3});
    auto bias = builder.CreateInput(3, "bias", DT_FLOAT16, FORMAT_ND, {64});

    auto y = CreateConv2DTransposeNode(builder, "Conv2DTranspose", inputSize, x, filter, bias, {1, 1, 2, 2},
                                       {0, 0, 1, 1}, {1, 1, 1, 1}, 1, "NCHW", DT_FLOAT16, {2, 64, 32, 32}, FORMAT_NCHW);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::Conv2DTransposeToV2FusionPass pass({AscendString("Conv2DTranspose")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv3DTranspose"));
    EXPECT_TRUE(CheckNodeExists(graph, "Unsqueeze"));
    EXPECT_TRUE(CheckNodeExists(graph, "Squeeze"));
}

// Test 2: unsupportedPlatformFail - platform not supported
TEST_F(Conv2DTransposeToV2FusionPassTest, unsupportedPlatformFail)
{
    // Ascend910_93 does not have Intrinsic_data_move_out2l1_dn2nz intrinsic
    SetPlatform("Ascend910_93");
    auto builder = EsGraphBuilder("unsupportedPlatformFail");
    auto inputSize = builder.CreateInput(0, "input_size", DT_INT32, FORMAT_ND, {INPUT_SIZE_DIM});
    auto x = builder.CreateInput(1, "x", DT_FLOAT16, FORMAT_NCHW, {2, 32, 16, 16});
    auto filter = builder.CreateInput(2, "filter", DT_FLOAT16, FORMAT_NCHW, {32, 64, 3, 3});
    auto bias = builder.CreateInput(3, "bias", DT_FLOAT16, FORMAT_ND, {64});

    auto y = CreateConv2DTransposeNode(builder, "Conv2DTranspose", inputSize, x, filter, bias, {1, 1, 2, 2},
                                       {0, 0, 1, 1}, {1, 1, 1, 1}, 1, "NCHW", DT_FLOAT16, {2, 64, 32, 32}, FORMAT_NCHW);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::Conv2DTransposeToV2FusionPass pass({AscendString("Conv2DTranspose")});
    EXPECT_EQ(pass.Run(graph, ctx), GRAPH_NOT_CHANGED);
    EXPECT_FALSE(CheckNodeExists(graph, "Conv3DTranspose"));
}

// Test 3: unsupportedDtypeFail - INT8 dtype not supported
TEST_F(Conv2DTransposeToV2FusionPassTest, unsupportedDtypeFail)
{
    auto builder = EsGraphBuilder("unsupportedDtypeFail");
    auto inputSize = builder.CreateInput(0, "input_size", DT_INT32, FORMAT_ND, {INPUT_SIZE_DIM});
    auto x = builder.CreateInput(1, "x", DT_INT8, FORMAT_NCHW, {2, 32, 16, 16});
    auto filter = builder.CreateInput(2, "filter", DT_INT8, FORMAT_NCHW, {32, 64, 3, 3});
    auto bias = builder.CreateInput(3, "bias", DT_INT32, FORMAT_ND, {64});

    auto y = CreateConv2DTransposeNode(builder, "Conv2DTranspose", inputSize, x, filter, bias, {1, 1, 2, 2},
                                       {0, 0, 1, 1}, {1, 1, 1, 1}, 1, "NCHW", DT_INT32, {2, 64, 32, 32}, FORMAT_NCHW);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::Conv2DTransposeToV2FusionPass pass({AscendString("Conv2DTranspose")});
    EXPECT_EQ(pass.Run(graph, ctx), GRAPH_NOT_CHANGED);
    EXPECT_FALSE(CheckNodeExists(graph, "Conv3DTranspose"));
}

// Test 4: nhwcFusionSuccess - NHWC format fusion
TEST_F(Conv2DTransposeToV2FusionPassTest, nhwcFusionSuccess)
{
    auto builder = EsGraphBuilder("nhwcFusionSuccess");
    auto inputSize = builder.CreateInput(0, "input_size", DT_INT32, FORMAT_ND, {INPUT_SIZE_DIM});
    auto x = builder.CreateInput(1, "x", DT_FLOAT16, FORMAT_NHWC, {2, 16, 16, 32});
    auto filter = builder.CreateInput(2, "filter", DT_FLOAT16, FORMAT_NHWC, {32, 3, 3, 64});
    auto bias = builder.CreateInput(3, "bias", DT_FLOAT16, FORMAT_ND, {64});

    auto y = CreateConv2DTransposeNode(builder, "Conv2DTranspose", inputSize, x, filter, bias, {1, 2, 2, 1},
                                       {0, 1, 1, 0}, {1, 1, 1, 1}, 1, "NHWC", DT_FLOAT16, {2, 32, 32, 64}, FORMAT_NHWC);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::Conv2DTransposeToV2FusionPass pass({AscendString("Conv2DTranspose")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv3DTranspose"));
}

// Test 5: noBiasFusionSuccess - Conv2DTranspose without bias input
TEST_F(Conv2DTransposeToV2FusionPassTest, noBiasFusionSuccess)
{
    auto builder = EsGraphBuilder("noBiasFusionSuccess");
    auto inputSize = builder.CreateInput(0, "input_size", DT_INT32, FORMAT_ND, {INPUT_SIZE_DIM});
    auto x = builder.CreateInput(1, "x", DT_FLOAT16, FORMAT_NCHW, {2, 32, 16, 16});
    auto filter = builder.CreateInput(2, "filter", DT_FLOAT16, FORMAT_NCHW, {32, 64, 3, 3});

    // Create Conv2DTranspose node without bias - only 3 required inputs
    auto* graph = builder.GetCGraphBuilder()->GetGraph();
    auto node = CompliantNodeBuilder(graph)
                    .OpType("Conv2DTranspose")
                    .Name("Conv2DTranspose_noBias")
                    .IrDefInputs({{"input_size", CompliantNodeBuilder::kEsIrInputRequired, ""},
                                  {"x", CompliantNodeBuilder::kEsIrInputRequired, ""},
                                  {"filter", CompliantNodeBuilder::kEsIrInputRequired, ""}})
                    .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                    .InstanceOutputDataType("y", DT_FLOAT16)
                    .InstanceOutputShape("y", std::vector<int64_t>{2, 64, 32, 32})
                    .InstanceOutputFormat("y", FORMAT_NCHW)
                    .Build();

    AddEdgeAndUpdatePeerDesc(*graph, *inputSize.GetProducer(), inputSize.GetProducerOutIndex(), node, 0);
    AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(), node, 1);
    AddEdgeAndUpdatePeerDesc(*graph, *filter.GetProducer(), filter.GetProducerOutIndex(), node, 2);

    TensorDesc inputSizeDesc, xDesc, filterDesc;
    inputSize.GetProducer()->GetOutputDesc(inputSize.GetProducerOutIndex(), inputSizeDesc);
    x.GetProducer()->GetOutputDesc(x.GetProducerOutIndex(), xDesc);
    filter.GetProducer()->GetOutputDesc(filter.GetProducerOutIndex(), filterDesc);
    node.UpdateInputDesc(0, inputSizeDesc);
    node.UpdateInputDesc(1, xDesc);
    node.UpdateInputDesc(2, filterDesc);

    std::vector<int64_t> stridesAttr = {1, 1, 2, 2};
    std::vector<int64_t> padsAttr = {0, 0, 1, 1};
    std::vector<int64_t> dilationsAttr = {1, 1, 1, 1};
    node.SetAttr("strides", stridesAttr);
    node.SetAttr("pads", padsAttr);
    node.SetAttr("dilations", dilationsAttr);
    int64_t groupsAttr = 1;
    node.SetAttr("groups", groupsAttr);
    AscendString fmtAttr("NCHW");
    node.SetAttr("data_format", fmtAttr);
    std::vector<int64_t> outputPaddingAttr = {0, 0, 0, 0};
    node.SetAttr("output_padding", outputPaddingAttr);
    int64_t offsetXAttr = 0;
    node.SetAttr("offset_x", offsetXAttr);
    node.SetAttr("_op_impl_mode_enum", DEFAULT_IMPL_MODE);

    auto y = EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 0));

    std::shared_ptr<Graph> graphPtr = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::Conv2DTransposeToV2FusionPass pass({AscendString("Conv2DTranspose")});
    EXPECT_EQ(pass.Run(graphPtr, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graphPtr, "Conv3DTranspose"));
}

// Test 6: bf16FusionSuccess - BF16 dtype fusion
TEST_F(Conv2DTransposeToV2FusionPassTest, bf16FusionSuccess)
{
    auto builder = EsGraphBuilder("bf16FusionSuccess");
    auto inputSize = builder.CreateInput(0, "input_size", DT_INT32, FORMAT_ND, {INPUT_SIZE_DIM});
    auto x = builder.CreateInput(1, "x", DT_BF16, FORMAT_NCHW, {2, 32, 16, 16});
    auto filter = builder.CreateInput(2, "filter", DT_BF16, FORMAT_NCHW, {32, 64, 3, 3});
    auto bias = builder.CreateInput(3, "bias", DT_BF16, FORMAT_ND, {64});

    auto y = CreateConv2DTransposeNode(builder, "Conv2DTranspose", inputSize, x, filter, bias, {1, 1, 2, 2},
                                       {0, 0, 1, 1}, {1, 1, 1, 1}, 1, "NCHW", DT_BF16, {2, 64, 32, 32}, FORMAT_NCHW);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::Conv2DTransposeToV2FusionPass pass({AscendString("Conv2DTranspose")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv3DTranspose"));
}

// Test 7: fp32FusionSuccess - FP32 dtype fusion
TEST_F(Conv2DTransposeToV2FusionPassTest, fp32FusionSuccess)
{
    auto builder = EsGraphBuilder("fp32FusionSuccess");
    auto inputSize = builder.CreateInput(0, "input_size", DT_INT32, FORMAT_ND, {INPUT_SIZE_DIM});
    auto x = builder.CreateInput(1, "x", DT_FLOAT, FORMAT_NCHW, {2, 32, 16, 16});
    auto filter = builder.CreateInput(2, "filter", DT_FLOAT, FORMAT_NCHW, {32, 64, 3, 3});
    auto bias = builder.CreateInput(3, "bias", DT_FLOAT, FORMAT_ND, {64});

    auto y = CreateConv2DTransposeNode(builder, "Conv2DTranspose", inputSize, x, filter, bias, {1, 1, 2, 2},
                                       {0, 0, 1, 1}, {1, 1, 1, 1}, 1, "NCHW", DT_FLOAT, {2, 64, 32, 32}, FORMAT_NCHW);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::Conv2DTransposeToV2FusionPass pass({AscendString("Conv2DTranspose")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv3DTranspose"));
    EXPECT_TRUE(CheckNodeExists(graph, "Unsqueeze"));
    EXPECT_TRUE(CheckNodeExists(graph, "Squeeze"));
}

// Test 8: hwcnWeightFusionSuccess - HWCN weight format, x/y still NCHW
TEST_F(Conv2DTransposeToV2FusionPassTest, hwcnWeightFusionSuccess)
{
    auto builder = EsGraphBuilder("hwcnWeightFusionSuccess");
    auto inputSize = builder.CreateInput(0, "input_size", DT_INT32, FORMAT_ND, {INPUT_SIZE_DIM});
    auto x = builder.CreateInput(1, "x", DT_FLOAT16, FORMAT_NCHW, {2, 32, 16, 16});
    auto filter = builder.CreateInput(2, "filter", DT_FLOAT16, FORMAT_HWCN, {3, 3, 64, 32});
    auto bias = builder.CreateInput(3, "bias", DT_FLOAT16, FORMAT_ND, {64});

    auto y = CreateConv2DTransposeNode(builder, "Conv2DTranspose", inputSize, x, filter, bias, {1, 1, 2, 2},
                                       {0, 0, 1, 1}, {1, 1, 1, 1}, 1, "NCHW", DT_FLOAT16, {2, 64, 32, 32}, FORMAT_NCHW);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::Conv2DTransposeToV2FusionPass pass({AscendString("Conv2DTranspose")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv3DTranspose"));
}

// Test 9: withOutputPaddingFusionSuccess - non-zero output_padding
TEST_F(Conv2DTransposeToV2FusionPassTest, withOutputPaddingFusionSuccess)
{
    auto builder = EsGraphBuilder("withOutputPaddingFusionSuccess");
    auto inputSize = builder.CreateInput(0, "input_size", DT_INT32, FORMAT_ND, {INPUT_SIZE_DIM});
    auto x = builder.CreateInput(1, "x", DT_FLOAT16, FORMAT_NCHW, {2, 32, 16, 16});
    auto filter = builder.CreateInput(2, "filter", DT_FLOAT16, FORMAT_NCHW, {32, 64, 3, 3});
    auto bias = builder.CreateInput(3, "bias", DT_FLOAT16, FORMAT_ND, {64});

    auto* graph = builder.GetCGraphBuilder()->GetGraph();
    auto node = CompliantNodeBuilder(graph)
                    .OpType("Conv2DTranspose")
                    .Name("Conv2DTranspose_outputPadding")
                    .IrDefInputs({{"input_size", CompliantNodeBuilder::kEsIrInputRequired, ""},
                                  {"x", CompliantNodeBuilder::kEsIrInputRequired, ""},
                                  {"filter", CompliantNodeBuilder::kEsIrInputRequired, ""},
                                  {"bias", CompliantNodeBuilder::kEsIrInputOptional, ""}})
                    .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                    .InstanceOutputDataType("y", DT_FLOAT16)
                    .InstanceOutputShape("y", std::vector<int64_t>{2, 64, 32, 32})
                    .InstanceOutputFormat("y", FORMAT_NCHW)
                    .Build();

    AddEdgeAndUpdatePeerDesc(*graph, *inputSize.GetProducer(), inputSize.GetProducerOutIndex(), node, 0);
    AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(), node, 1);
    AddEdgeAndUpdatePeerDesc(*graph, *filter.GetProducer(), filter.GetProducerOutIndex(), node, 2);
    AddEdgeAndUpdatePeerDesc(*graph, *bias.GetProducer(), bias.GetProducerOutIndex(), node, 3);

    TensorDesc inputSizeDesc, xDesc, filterDesc, biasDesc;
    inputSize.GetProducer()->GetOutputDesc(inputSize.GetProducerOutIndex(), inputSizeDesc);
    x.GetProducer()->GetOutputDesc(x.GetProducerOutIndex(), xDesc);
    filter.GetProducer()->GetOutputDesc(filter.GetProducerOutIndex(), filterDesc);
    bias.GetProducer()->GetOutputDesc(bias.GetProducerOutIndex(), biasDesc);
    node.UpdateInputDesc(0, inputSizeDesc);
    node.UpdateInputDesc(1, xDesc);
    node.UpdateInputDesc(2, filterDesc);
    node.UpdateInputDesc(3, biasDesc);

    std::vector<int64_t> stridesVal = {1, 1, 2, 2};
    std::vector<int64_t> padsVal = {0, 0, 1, 1};
    std::vector<int64_t> dilationsVal = {1, 1, 1, 1};
    node.SetAttr("strides", stridesVal);
    node.SetAttr("pads", padsVal);
    node.SetAttr("dilations", dilationsVal);
    int64_t groupsVal = 1;
    node.SetAttr("groups", groupsVal);
    AscendString fmt("NCHW");
    node.SetAttr("data_format", fmt);
    std::vector<int64_t> outputPaddingVal = {0, 0, 1, 1};
    node.SetAttr("output_padding", outputPaddingVal);
    int64_t offsetXVal = 0;
    node.SetAttr("offset_x", offsetXVal);
    node.SetAttr("_op_impl_mode_enum", DEFAULT_IMPL_MODE);

    auto y = EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 0));
    std::shared_ptr<Graph> graphPtr = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::Conv2DTransposeToV2FusionPass pass({AscendString("Conv2DTranspose")});
    EXPECT_EQ(pass.Run(graphPtr, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graphPtr, "Conv3DTranspose"));
}

// Test 10: groupsNotOneSuccess - groups > 1 fusion
TEST_F(Conv2DTransposeToV2FusionPassTest, groupsNotOneSuccess)
{
    auto builder = EsGraphBuilder("groupsNotOneSuccess");
    auto inputSize = builder.CreateInput(0, "input_size", DT_INT32, FORMAT_ND, {INPUT_SIZE_DIM});
    auto x = builder.CreateInput(1, "x", DT_FLOAT16, FORMAT_NCHW, {2, 64, 16, 16});
    auto filter = builder.CreateInput(2, "filter", DT_FLOAT16, FORMAT_NCHW, {64, 8, 3, 3});
    auto bias = builder.CreateInput(3, "bias", DT_FLOAT16, FORMAT_ND, {64});

    auto y = CreateConv2DTransposeNode(builder, "Conv2DTranspose", inputSize, x, filter, bias, {1, 1, 2, 2},
                                       {0, 0, 1, 1}, {1, 1, 1, 1}, 8, "NCHW", DT_FLOAT16, {2, 64, 32, 32}, FORMAT_NCHW);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::Conv2DTransposeToV2FusionPass pass({AscendString("Conv2DTranspose")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv3DTranspose"));
}

// Test 11: filterOnlyInt8Fail - x is fp16 but filter is INT8
TEST_F(Conv2DTransposeToV2FusionPassTest, filterOnlyInt8Fail)
{
    auto builder = EsGraphBuilder("filterOnlyInt8Fail");
    auto inputSize = builder.CreateInput(0, "input_size", DT_INT32, FORMAT_ND, {INPUT_SIZE_DIM});
    auto x = builder.CreateInput(1, "x", DT_FLOAT16, FORMAT_NCHW, {2, 32, 16, 16});
    auto filter = builder.CreateInput(2, "filter", DT_INT8, FORMAT_NCHW, {32, 64, 3, 3});
    auto bias = builder.CreateInput(3, "bias", DT_FLOAT, FORMAT_ND, {64});

    auto y = CreateConv2DTransposeNode(builder, "Conv2DTranspose", inputSize, x, filter, bias, {1, 1, 2, 2},
                                       {0, 0, 1, 1}, {1, 1, 1, 1}, 1, "NCHW", DT_FLOAT16, {2, 64, 32, 32}, FORMAT_NCHW);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::Conv2DTransposeToV2FusionPass pass({AscendString("Conv2DTranspose")});
    EXPECT_EQ(pass.Run(graph, ctx), GRAPH_NOT_CHANGED);
    EXPECT_FALSE(CheckNodeExists(graph, "Conv3DTranspose"));
}
