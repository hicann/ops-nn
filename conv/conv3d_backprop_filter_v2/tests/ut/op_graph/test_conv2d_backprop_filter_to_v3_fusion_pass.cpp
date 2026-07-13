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
#include "../../../op_graph/fusion_pass/conv2d_backprop_filter_to_v3_fusion_pass.h"

using namespace ge;
using namespace ge::es;
using namespace fe;
using namespace fusion;
using namespace ops::ConvBackpropFusionUtils;

namespace {

constexpr int64_t kAiCoreCnt = 64;
int64_t kDefaultImplMode = 0x1;

void SetPlatform(const std::string& soc)
{
    PlatformInfo platformInfo;
    OptionalInfo optionalInfo;
    platformInfo.soc_info.ai_core_cnt = kAiCoreCnt;
    platformInfo.str_info.short_soc_version = soc;
    optionalInfo.soc_version = soc;
    if (soc == "Ascend950" || soc == "MC62CM12A") {
        platformInfo.ai_core_intrinsic_dtype_map["Intrinsic_data_move_out2l1_dn2nz"] = {"float16", "float", "bfloat16"};
    }
    PlatformInfoManager::Instance().platform_info_map_[soc] = platformInfo;
    PlatformInfoManager::Instance().SetOptionalCompilationInfo(optionalInfo);
}

EsTensorHolder CreateConv2DBpFilterNode(EsGraphBuilder& builder, const char* opType, const EsTensorHolder& x,
                                        const EsTensorHolder& filterSize, const EsTensorHolder& outBackprop,
                                        std::vector<int64_t> strides, std::vector<int64_t> pads,
                                        std::vector<int64_t> dilations, int64_t groups, const std::string& dataFormat,
                                        DataType outDtype, const std::vector<int64_t>& outShape, Format outFormat)
{
    auto* graph = builder.GetCGraphBuilder()->GetGraph();
    auto node = CompliantNodeBuilder(graph)
                    .OpType(opType)
                    .Name(opType)
                    .IrDefInputs({{"x", CompliantNodeBuilder::kEsIrInputRequired, ""},
                                  {"filter_size", CompliantNodeBuilder::kEsIrInputRequired, ""},
                                  {"out_backprop", CompliantNodeBuilder::kEsIrInputRequired, ""}})
                    .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                    .InstanceOutputDataType("y", outDtype)
                    .InstanceOutputShape("y", outShape)
                    .InstanceOutputFormat("y", outFormat)
                    .Build();

    AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(), node, 0);
    AddEdgeAndUpdatePeerDesc(*graph, *filterSize.GetProducer(), filterSize.GetProducerOutIndex(), node, 1);
    AddEdgeAndUpdatePeerDesc(*graph, *outBackprop.GetProducer(), outBackprop.GetProducerOutIndex(), node, 2);

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
    node.SetAttr("_op_impl_mode_enum", kDefaultImplMode);

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

class Conv2dBpFilterToV3FusionPassTest : public testing::Test {
protected:
    void SetUp() override { SetPlatform("Ascend950"); }
};

// Test 1: patternTest - FP16 NCHW basic fusion success
TEST_F(Conv2dBpFilterToV3FusionPassTest, patternTest)
{
    auto builder = EsGraphBuilder("patternTest");
    auto x = builder.CreateInput(0, "x", DT_FLOAT16, FORMAT_NCHW, {2, 32, 16, 16});
    auto filterSize = builder.CreateInput(1, "filter_size", DT_INT64, FORMAT_ND, {4});
    auto outBackprop = builder.CreateInput(2, "out_backprop", DT_FLOAT16, FORMAT_NCHW, {2, 64, 8, 8});

    auto y = CreateConv2DBpFilterNode(builder, "Conv2DBackpropFilter", x, filterSize, outBackprop, {1, 1, 1, 1},
                                      {0, 0, 0, 0}, {1, 1, 1, 1}, 1, "NCHW", DT_FLOAT16, {64, 32, 3, 3}, FORMAT_NCHW);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::Conv2DBackpropFilterToV3FusionPass pass({AscendString("Conv2DBackpropFilter")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv3DBackpropFilter"));
    EXPECT_TRUE(CheckNodeExists(graph, "Unsqueeze"));
    EXPECT_TRUE(CheckNodeExists(graph, "Squeeze"));
}

// Test 2: unsupportedPlatformFail - Ascend910_93 not supported
TEST_F(Conv2dBpFilterToV3FusionPassTest, unsupportedPlatformFail)
{
    SetPlatform("Ascend910_93");
    auto builder = EsGraphBuilder("unsupportedPlatformFail");
    auto x = builder.CreateInput(0, "x", DT_FLOAT16, FORMAT_NCHW, {2, 32, 16, 16});
    auto filterSize = builder.CreateInput(1, "filter_size", DT_INT64, FORMAT_ND, {4});
    auto outBackprop = builder.CreateInput(2, "out_backprop", DT_FLOAT16, FORMAT_NCHW, {2, 64, 8, 8});

    auto y = CreateConv2DBpFilterNode(builder, "Conv2DBackpropFilter", x, filterSize, outBackprop, {1, 1, 1, 1},
                                      {0, 0, 0, 0}, {1, 1, 1, 1}, 1, "NCHW", DT_FLOAT16, {64, 32, 3, 3}, FORMAT_NCHW);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::Conv2DBackpropFilterToV3FusionPass pass({AscendString("Conv2DBackpropFilter")});
    EXPECT_EQ(pass.Run(graph, ctx), GRAPH_NOT_CHANGED);
    EXPECT_FALSE(CheckNodeExists(graph, "Conv3DBackpropFilter"));
}

// Test 3: nhwcFusionSuccess - NHWC format fusion
TEST_F(Conv2dBpFilterToV3FusionPassTest, nhwcFusionSuccess)
{
    auto builder = EsGraphBuilder("nhwcFusionSuccess");
    auto x = builder.CreateInput(0, "x", DT_FLOAT16, FORMAT_NHWC, {2, 16, 16, 32});
    auto filterSize = builder.CreateInput(1, "filter_size", DT_INT64, FORMAT_ND, {4});
    auto outBackprop = builder.CreateInput(2, "out_backprop", DT_FLOAT16, FORMAT_NHWC, {2, 8, 8, 64});

    auto y = CreateConv2DBpFilterNode(builder, "Conv2DBackpropFilter", x, filterSize, outBackprop, {1, 1, 1, 1},
                                      {0, 0, 0, 0}, {1, 1, 1, 1}, 1, "NHWC", DT_FLOAT16, {3, 3, 32, 64}, FORMAT_NHWC);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::Conv2DBackpropFilterToV3FusionPass pass({AscendString("Conv2DBackpropFilter")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv3DBackpropFilter"));
    EXPECT_TRUE(CheckNodeExists(graph, "Unsqueeze"));
    EXPECT_TRUE(CheckNodeExists(graph, "Squeeze"));
}

// Test 4: bf16FusionSuccess - BF16 dtype fusion
TEST_F(Conv2dBpFilterToV3FusionPassTest, bf16FusionSuccess)
{
    auto builder = EsGraphBuilder("bf16FusionSuccess");
    auto x = builder.CreateInput(0, "x", DT_BF16, FORMAT_NCHW, {2, 32, 16, 16});
    auto filterSize = builder.CreateInput(1, "filter_size", DT_INT64, FORMAT_ND, {4});
    auto outBackprop = builder.CreateInput(2, "out_backprop", DT_BF16, FORMAT_NCHW, {2, 64, 8, 8});

    auto y = CreateConv2DBpFilterNode(builder, "Conv2DBackpropFilter", x, filterSize, outBackprop, {1, 1, 1, 1},
                                      {0, 0, 0, 0}, {1, 1, 1, 1}, 1, "NCHW", DT_BF16, {64, 32, 3, 3}, FORMAT_NCHW);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::Conv2DBackpropFilterToV3FusionPass pass({AscendString("Conv2DBackpropFilter")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv3DBackpropFilter"));
}

// Test 5: fp32FusionSuccess - FP32 dtype fusion
TEST_F(Conv2dBpFilterToV3FusionPassTest, fp32FusionSuccess)
{
    auto builder = EsGraphBuilder("fp32FusionSuccess");
    auto x = builder.CreateInput(0, "x", DT_FLOAT, FORMAT_NCHW, {2, 32, 16, 16});
    auto filterSize = builder.CreateInput(1, "filter_size", DT_INT64, FORMAT_ND, {4});
    auto outBackprop = builder.CreateInput(2, "out_backprop", DT_FLOAT, FORMAT_NCHW, {2, 64, 8, 8});

    auto y = CreateConv2DBpFilterNode(builder, "Conv2DBackpropFilter", x, filterSize, outBackprop, {1, 1, 1, 1},
                                      {0, 0, 0, 0}, {1, 1, 1, 1}, 1, "NCHW", DT_FLOAT, {64, 32, 3, 3}, FORMAT_NCHW);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::Conv2DBackpropFilterToV3FusionPass pass({AscendString("Conv2DBackpropFilter")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv3DBackpropFilter"));
}

// Test 6: differentShapeSuccess - different input shape
TEST_F(Conv2dBpFilterToV3FusionPassTest, differentShapeSuccess)
{
    auto builder = EsGraphBuilder("differentShapeSuccess");
    auto x = builder.CreateInput(0, "x", DT_FLOAT16, FORMAT_NCHW, {4, 16, 8, 8});
    auto filterSize = builder.CreateInput(1, "filter_size", DT_INT64, FORMAT_ND, {4});
    auto outBackprop = builder.CreateInput(2, "out_backprop", DT_FLOAT16, FORMAT_NCHW, {4, 32, 4, 4});

    auto y = CreateConv2DBpFilterNode(builder, "Conv2DBackpropFilter", x, filterSize, outBackprop, {1, 1, 1, 1},
                                      {0, 0, 0, 0}, {1, 1, 1, 1}, 1, "NCHW", DT_FLOAT16, {32, 16, 5, 5}, FORMAT_NCHW);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::Conv2DBackpropFilterToV3FusionPass pass({AscendString("Conv2DBackpropFilter")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv3DBackpropFilter"));
}

// Test 7: mc62cm12APlatformFail - MC62CM12A platform not supported
TEST_F(Conv2dBpFilterToV3FusionPassTest, mc62cm12APlatformFail)
{
    SetPlatform("MC62CM12A");
    auto builder = EsGraphBuilder("mc62cm12APlatformFail");
    auto x = builder.CreateInput(0, "x", DT_FLOAT16, FORMAT_NCHW, {2, 32, 16, 16});
    auto filterSize = builder.CreateInput(1, "filter_size", DT_INT64, FORMAT_ND, {4});
    auto outBackprop = builder.CreateInput(2, "out_backprop", DT_FLOAT16, FORMAT_NCHW, {2, 64, 8, 8});

    auto y = CreateConv2DBpFilterNode(builder, "Conv2DBackpropFilter", x, filterSize, outBackprop, {1, 1, 1, 1},
                                      {0, 0, 0, 0}, {1, 1, 1, 1}, 1, "NCHW", DT_FLOAT16, {64, 32, 3, 3}, FORMAT_NCHW);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::Conv2DBackpropFilterToV3FusionPass pass({AscendString("Conv2DBackpropFilter")});
    EXPECT_EQ(pass.Run(graph, ctx), GRAPH_NOT_CHANGED);
    EXPECT_FALSE(CheckNodeExists(graph, "Conv3DBackpropFilter"));
}
