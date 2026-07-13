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
#include "../../../op_graph/fusion_pass/conv2d_backprop_input_to_v2_fusion_pass.h"

using namespace ge;
using namespace ge::es;
using namespace fe;
using namespace fusion;
using namespace ops::ConvBackpropFusionUtils;

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
    if (soc == "Ascend950") {
        platformInfo.ai_core_intrinsic_dtype_map["Intrinsic_data_move_out2l1_dn2nz"] = {"float16", "float", "bfloat16"};
    }
    PlatformInfoManager::Instance().platform_info_map_[soc] = platformInfo;
    PlatformInfoManager::Instance().SetOptionalCompilationInfo(optionalInfo);
}

EsTensorHolder CreateConv2DBpInputNode(EsGraphBuilder& builder, const char* opType, const EsTensorHolder& inputSize,
                                       const EsTensorHolder& filter, const EsTensorHolder& outBackprop,
                                       std::vector<int64_t> strides, std::vector<int64_t> pads,
                                       std::vector<int64_t> dilations, int64_t groups, const std::string& dataFormat,
                                       DataType outDtype, const std::vector<int64_t>& outShape, Format outFormat)
{
    auto* graph = builder.GetCGraphBuilder()->GetGraph();
    auto node = CompliantNodeBuilder(graph)
                    .OpType(opType)
                    .Name(opType)
                    .IrDefInputs({{"input_size", CompliantNodeBuilder::kEsIrInputRequired, ""},
                                  {"filter", CompliantNodeBuilder::kEsIrInputRequired, ""},
                                  {"out_backprop", CompliantNodeBuilder::kEsIrInputRequired, ""}})
                    .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                    .InstanceOutputDataType("y", outDtype)
                    .InstanceOutputShape("y", outShape)
                    .InstanceOutputFormat("y", outFormat)
                    .Build();

    AddEdgeAndUpdatePeerDesc(*graph, *inputSize.GetProducer(), inputSize.GetProducerOutIndex(), node, 0);
    AddEdgeAndUpdatePeerDesc(*graph, *filter.GetProducer(), filter.GetProducerOutIndex(), node, 1);
    AddEdgeAndUpdatePeerDesc(*graph, *outBackprop.GetProducer(), outBackprop.GetProducerOutIndex(), node, 2);

    TensorDesc inputSizeDesc, filterDesc, outBackpropDesc;
    inputSize.GetProducer()->GetOutputDesc(inputSize.GetProducerOutIndex(), inputSizeDesc);
    filter.GetProducer()->GetOutputDesc(filter.GetProducerOutIndex(), filterDesc);
    outBackprop.GetProducer()->GetOutputDesc(outBackprop.GetProducerOutIndex(), outBackpropDesc);
    node.UpdateInputDesc(0, inputSizeDesc);
    node.UpdateInputDesc(1, filterDesc);
    node.UpdateInputDesc(2, outBackpropDesc);

    node.SetAttr("strides", strides);
    node.SetAttr("pads", pads);
    node.SetAttr("dilations", dilations);
    node.SetAttr("groups", groups);
    AscendString fmt = dataFormat.c_str();
    node.SetAttr("data_format", fmt);
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

class Conv2DBpInputToV2FusionPassTest : public testing::Test {
protected:
    void SetUp() override { SetPlatform("Ascend950"); }
};

// Test 1: patternTest - FP16 NCHW basic fusion success
TEST_F(Conv2DBpInputToV2FusionPassTest, patternTest)
{
    auto builder = EsGraphBuilder("patternTest");
    auto inputSize = builder.CreateInput(0, "input_size", DT_INT32, FORMAT_ND, {INPUT_SIZE_DIM});
    auto filter = builder.CreateInput(1, "filter", DT_FLOAT16, FORMAT_NCHW, {32, 64, 3, 3});
    auto outBackprop = builder.CreateInput(2, "out_backprop", DT_FLOAT16, FORMAT_NCHW, {2, 64, 16, 16});

    auto y = CreateConv2DBpInputNode(builder, "Conv2DBackpropInput", inputSize, filter, outBackprop, {1, 1, 2, 2},
                                     {0, 0, 1, 1}, {1, 1, 1, 1}, 1, "NCHW", DT_FLOAT16, {2, 32, 32, 32}, FORMAT_NCHW);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::Conv2DBackpropInputToV2FusionPass pass({AscendString("Conv2DBackpropInput")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv3DBackpropInput"));
    EXPECT_TRUE(CheckNodeExists(graph, "Unsqueeze"));
    EXPECT_TRUE(CheckNodeExists(graph, "Squeeze"));
}

// Test 2: unsupportedPlatformFail - Ascend910_93 not supported
TEST_F(Conv2DBpInputToV2FusionPassTest, unsupportedPlatformFail)
{
    SetPlatform("Ascend910_93");
    auto builder = EsGraphBuilder("unsupportedPlatformFail");
    auto inputSize = builder.CreateInput(0, "input_size", DT_INT32, FORMAT_ND, {INPUT_SIZE_DIM});
    auto filter = builder.CreateInput(1, "filter", DT_FLOAT16, FORMAT_NCHW, {32, 64, 3, 3});
    auto outBackprop = builder.CreateInput(2, "out_backprop", DT_FLOAT16, FORMAT_NCHW, {2, 64, 16, 16});

    auto y = CreateConv2DBpInputNode(builder, "Conv2DBackpropInput", inputSize, filter, outBackprop, {1, 1, 2, 2},
                                     {0, 0, 1, 1}, {1, 1, 1, 1}, 1, "NCHW", DT_FLOAT16, {2, 32, 32, 32}, FORMAT_NCHW);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::Conv2DBackpropInputToV2FusionPass pass({AscendString("Conv2DBackpropInput")});
    EXPECT_EQ(pass.Run(graph, ctx), GRAPH_NOT_CHANGED);
    EXPECT_FALSE(CheckNodeExists(graph, "Conv3DBackpropInput"));
}

// Test 3: nhwcFusionSuccess - NHWC format fusion
TEST_F(Conv2DBpInputToV2FusionPassTest, nhwcFusionSuccess)
{
    auto builder = EsGraphBuilder("nhwcFusionSuccess");
    auto inputSize = builder.CreateInput(0, "input_size", DT_INT32, FORMAT_ND, {INPUT_SIZE_DIM});
    auto filter = builder.CreateInput(1, "filter", DT_FLOAT16, FORMAT_NHWC, {3, 3, 64, 32});
    auto outBackprop = builder.CreateInput(2, "out_backprop", DT_FLOAT16, FORMAT_NHWC, {2, 16, 16, 64});

    auto y = CreateConv2DBpInputNode(builder, "Conv2DBackpropInput", inputSize, filter, outBackprop, {1, 2, 2, 1},
                                     {0, 1, 1, 0}, {1, 1, 1, 1}, 1, "NHWC", DT_FLOAT16, {2, 32, 32, 32}, FORMAT_NHWC);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::Conv2DBackpropInputToV2FusionPass pass({AscendString("Conv2DBackpropInput")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv3DBackpropInput"));
    EXPECT_TRUE(CheckNodeExists(graph, "Unsqueeze"));
    EXPECT_TRUE(CheckNodeExists(graph, "Squeeze"));
}

// Test 4: bf16FusionSuccess - BF16 dtype fusion
TEST_F(Conv2DBpInputToV2FusionPassTest, bf16FusionSuccess)
{
    auto builder = EsGraphBuilder("bf16FusionSuccess");
    auto inputSize = builder.CreateInput(0, "input_size", DT_INT32, FORMAT_ND, {INPUT_SIZE_DIM});
    auto filter = builder.CreateInput(1, "filter", DT_BF16, FORMAT_NCHW, {32, 64, 3, 3});
    auto outBackprop = builder.CreateInput(2, "out_backprop", DT_BF16, FORMAT_NCHW, {2, 64, 16, 16});

    auto y = CreateConv2DBpInputNode(builder, "Conv2DBackpropInput", inputSize, filter, outBackprop, {1, 1, 2, 2},
                                     {0, 0, 1, 1}, {1, 1, 1, 1}, 1, "NCHW", DT_BF16, {2, 32, 32, 32}, FORMAT_NCHW);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::Conv2DBackpropInputToV2FusionPass pass({AscendString("Conv2DBackpropInput")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv3DBackpropInput"));
}

// Test 5: fp32FusionSuccess - FP32 dtype fusion
TEST_F(Conv2DBpInputToV2FusionPassTest, fp32FusionSuccess)
{
    auto builder = EsGraphBuilder("fp32FusionSuccess");
    auto inputSize = builder.CreateInput(0, "input_size", DT_INT32, FORMAT_ND, {INPUT_SIZE_DIM});
    auto filter = builder.CreateInput(1, "filter", DT_FLOAT, FORMAT_NCHW, {32, 64, 3, 3});
    auto outBackprop = builder.CreateInput(2, "out_backprop", DT_FLOAT, FORMAT_NCHW, {2, 64, 16, 16});

    auto y = CreateConv2DBpInputNode(builder, "Conv2DBackpropInput", inputSize, filter, outBackprop, {1, 1, 2, 2},
                                     {0, 0, 1, 1}, {1, 1, 1, 1}, 1, "NCHW", DT_FLOAT, {2, 32, 32, 32}, FORMAT_NCHW);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::Conv2DBackpropInputToV2FusionPass pass({AscendString("Conv2DBackpropInput")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv3DBackpropInput"));
}

// Test 6: differentShapeSuccess - different input shape
TEST_F(Conv2DBpInputToV2FusionPassTest, differentShapeSuccess)
{
    auto builder = EsGraphBuilder("differentShapeSuccess");
    auto inputSize = builder.CreateInput(0, "input_size", DT_INT32, FORMAT_ND, {INPUT_SIZE_DIM});
    auto filter = builder.CreateInput(1, "filter", DT_FLOAT16, FORMAT_NCHW, {16, 32, 5, 5});
    auto outBackprop = builder.CreateInput(2, "out_backprop", DT_FLOAT16, FORMAT_NCHW, {4, 32, 8, 8});

    auto y = CreateConv2DBpInputNode(builder, "Conv2DBackpropInput", inputSize, filter, outBackprop, {1, 1, 1, 1},
                                     {0, 0, 0, 0}, {1, 1, 1, 1}, 1, "NCHW", DT_FLOAT16, {4, 16, 8, 8}, FORMAT_NCHW);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::Conv2DBackpropInputToV2FusionPass pass({AscendString("Conv2DBackpropInput")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv3DBackpropInput"));
}
