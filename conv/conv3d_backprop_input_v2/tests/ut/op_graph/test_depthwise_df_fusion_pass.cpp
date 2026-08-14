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
#include "../../../op_graph/fusion_pass/depthwise_df_fusion_pass.h"

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

EsTensorHolder CreateDepthwiseConv2DBpInputDNode(EsGraphBuilder& builder, const char* opType,
                                                 const EsTensorHolder& filter, const EsTensorHolder& outBackprop,
                                                 std::vector<int64_t> strides, std::vector<int64_t> pads,
                                                 std::vector<int64_t> dilations, int64_t groups,
                                                 const std::string& dataFormat, DataType outDtype,
                                                 const std::vector<int64_t>& outShape, Format outFormat,
                                                 bool fromDepthwise = false)
{
    auto* graph = builder.GetCGraphBuilder()->GetGraph();
    auto node = CompliantNodeBuilder(graph)
                    .OpType(opType)
                    .Name(opType)
                    .IrDefInputs({{"filter", CompliantNodeBuilder::kEsIrInputRequired, ""},
                                  {"out_backprop", CompliantNodeBuilder::kEsIrInputRequired, ""}})
                    .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                    .InstanceOutputDataType("y", outDtype)
                    .InstanceOutputShape("y", outShape)
                    .InstanceOutputFormat("y", outFormat)
                    .Build();

    AddEdgeAndUpdatePeerDesc(*graph, *filter.GetProducer(), filter.GetProducerOutIndex(), node, 0);
    AddEdgeAndUpdatePeerDesc(*graph, *outBackprop.GetProducer(), outBackprop.GetProducerOutIndex(), node, 1);

    TensorDesc filterDesc, outBackpropDesc;
    filter.GetProducer()->GetOutputDesc(filter.GetProducerOutIndex(), filterDesc);
    outBackprop.GetProducer()->GetOutputDesc(outBackprop.GetProducerOutIndex(), outBackpropDesc);
    node.UpdateInputDesc(0, filterDesc);
    node.UpdateInputDesc(1, outBackpropDesc);

    node.SetAttr("strides", strides);
    node.SetAttr("pads", pads);
    node.SetAttr("dilations", dilations);
    node.SetAttr("groups", groups);
    AscendString fmt = dataFormat.c_str();
    node.SetAttr("data_format", fmt);
    node.SetAttr("_op_impl_mode_enum", DEFAULT_IMPL_MODE);
    node.SetAttr("from_depthwise", fromDepthwise);
    std::vector<int64_t> inputSizeAttr = outShape;
    node.SetAttr("input_size", inputSizeAttr);

    return EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 0));
}

EsTensorHolder CreateDepthwiseConv2DBpInputNode(EsGraphBuilder& builder, const char* opType,
                                                const EsTensorHolder& inputSize, const EsTensorHolder& filter,
                                                const EsTensorHolder& outBackprop, std::vector<int64_t> strides,
                                                std::vector<int64_t> pads, std::vector<int64_t> dilations,
                                                int64_t groups, const std::string& dataFormat, DataType outDtype,
                                                const std::vector<int64_t>& outShape, Format outFormat,
                                                bool fromDepthwise = false)
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
    node.SetAttr("from_depthwise", fromDepthwise);

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

bool GetNodeBoolAttr(GraphPtr& graph, const std::string& type, const std::string& attrName, bool& attrValue)
{
    for (auto node : graph->GetAllNodes()) {
        AscendString nodeType;
        node.GetType(nodeType);
        if (nodeType.GetString() == type) {
            return node.GetAttr(attrName.c_str(), attrValue) == GRAPH_SUCCESS;
        }
    }
    return false;
}

} // namespace

class DepthwiseDfFusionPassTest : public testing::Test {
protected:
    void SetUp() override { SetPlatform("Ascend950"); }
};

TEST_F(DepthwiseDfFusionPassTest, staticDNchwFp16Success)
{
    auto builder = EsGraphBuilder("staticDNchwFp16Success");
    auto filter = builder.CreateInput(0, "filter", DT_FLOAT16, FORMAT_NCHW, {32, 1, 3, 3});
    auto outBackprop = builder.CreateInput(1, "out_backprop", DT_FLOAT16, FORMAT_NCHW, {2, 32, 16, 16});

    auto y = CreateDepthwiseConv2DBpInputDNode(builder, "DepthwiseConv2DBackpropInputD", filter, outBackprop,
                                               {1, 1, 2, 2}, {0, 0, 1, 1}, {1, 1, 1, 1}, 32, "NCHW", DT_FLOAT16,
                                               {2, 32, 32, 32}, FORMAT_NCHW);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::DepthwiseDfFusionPass pass({AscendString("DepthwiseConv2DBackpropInputD")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv2DBackpropInputD"));
    EXPECT_TRUE(CheckNodeExists(graph, "Reshape"));
    EXPECT_TRUE(CheckNodeExists(graph, "Transpose"));
}

TEST_F(DepthwiseDfFusionPassTest, dynamicNchwFp16Success)
{
    auto builder = EsGraphBuilder("dynamicNchwFp16Success");
    auto inputSize = builder.CreateInput(0, "input_size", DT_INT32, FORMAT_ND, {INPUT_SIZE_DIM});
    auto filter = builder.CreateInput(1, "filter", DT_FLOAT16, FORMAT_NCHW, {32, 1, 3, 3});
    auto outBackprop = builder.CreateInput(2, "out_backprop", DT_FLOAT16, FORMAT_NCHW, {2, 32, 16, 16});

    auto y = CreateDepthwiseConv2DBpInputNode(builder, "DepthwiseConv2DBackpropInput", inputSize, filter, outBackprop,
                                              {1, 1, 2, 2}, {0, 0, 1, 1}, {1, 1, 1, 1}, 32, "NCHW", DT_FLOAT16,
                                              {2, 32, 32, 32}, FORMAT_NCHW);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::DepthwiseDfFusionPass pass({AscendString("DepthwiseConv2DBackpropInput")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv2DBackpropInput"));
    EXPECT_TRUE(CheckNodeExists(graph, "Reshape"));
    EXPECT_TRUE(CheckNodeExists(graph, "Transpose"));
}

TEST_F(DepthwiseDfFusionPassTest, staticDHwcnFp16Success)
{
    auto builder = EsGraphBuilder("staticDHwcnFp16Success");
    auto filter = builder.CreateInput(0, "filter", DT_FLOAT16, FORMAT_HWCN, {3, 3, 1, 32});
    auto outBackprop = builder.CreateInput(1, "out_backprop", DT_FLOAT16, FORMAT_NCHW, {2, 32, 16, 16});

    auto y = CreateDepthwiseConv2DBpInputDNode(builder, "DepthwiseConv2DBackpropInputD", filter, outBackprop,
                                               {1, 1, 2, 2}, {0, 0, 1, 1}, {1, 1, 1, 1}, 32, "NCHW", DT_FLOAT16,
                                               {2, 32, 32, 32}, FORMAT_NCHW);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::DepthwiseDfFusionPass pass({AscendString("DepthwiseConv2DBackpropInputD")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv2DBackpropInputD"));
    EXPECT_TRUE(CheckNodeExists(graph, "Reshape"));
    EXPECT_FALSE(CheckNodeExists(graph, "Transpose"));
}

TEST_F(DepthwiseDfFusionPassTest, dynamicHwcnFp16Success)
{
    auto builder = EsGraphBuilder("dynamicHwcnFp16Success");
    auto inputSize = builder.CreateInput(0, "input_size", DT_INT32, FORMAT_ND, {INPUT_SIZE_DIM});
    auto filter = builder.CreateInput(1, "filter", DT_FLOAT16, FORMAT_HWCN, {3, 3, 1, 32});
    auto outBackprop = builder.CreateInput(2, "out_backprop", DT_FLOAT16, FORMAT_NCHW, {2, 32, 16, 16});

    auto y = CreateDepthwiseConv2DBpInputNode(builder, "DepthwiseConv2DBackpropInput", inputSize, filter, outBackprop,
                                              {1, 1, 2, 2}, {0, 0, 1, 1}, {1, 1, 1, 1}, 32, "NCHW", DT_FLOAT16,
                                              {2, 32, 32, 32}, FORMAT_NCHW);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::DepthwiseDfFusionPass pass({AscendString("DepthwiseConv2DBackpropInput")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv2DBackpropInput"));
    EXPECT_TRUE(CheckNodeExists(graph, "Reshape"));
    EXPECT_FALSE(CheckNodeExists(graph, "Transpose"));
}

TEST_F(DepthwiseDfFusionPassTest, staticDNchwBf16Success)
{
    auto builder = EsGraphBuilder("staticDNchwBf16Success");
    auto filter = builder.CreateInput(0, "filter", DT_BF16, FORMAT_NCHW, {32, 1, 3, 3});
    auto outBackprop = builder.CreateInput(1, "out_backprop", DT_BF16, FORMAT_NCHW, {2, 32, 16, 16});

    auto y = CreateDepthwiseConv2DBpInputDNode(builder, "DepthwiseConv2DBackpropInputD", filter, outBackprop,
                                               {1, 1, 2, 2}, {0, 0, 1, 1}, {1, 1, 1, 1}, 32, "NCHW", DT_BF16,
                                               {2, 32, 32, 32}, FORMAT_NCHW);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::DepthwiseDfFusionPass pass({AscendString("DepthwiseConv2DBackpropInputD")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv2DBackpropInputD"));
}

TEST_F(DepthwiseDfFusionPassTest, staticDNchwFp32Success)
{
    auto builder = EsGraphBuilder("staticDNchwFp32Success");
    auto filter = builder.CreateInput(0, "filter", DT_FLOAT, FORMAT_NCHW, {32, 1, 3, 3});
    auto outBackprop = builder.CreateInput(1, "out_backprop", DT_FLOAT, FORMAT_NCHW, {2, 32, 16, 16});

    auto y = CreateDepthwiseConv2DBpInputDNode(builder, "DepthwiseConv2DBackpropInputD", filter, outBackprop,
                                               {1, 1, 2, 2}, {0, 0, 1, 1}, {1, 1, 1, 1}, 32, "NCHW", DT_FLOAT,
                                               {2, 32, 32, 32}, FORMAT_NCHW);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::DepthwiseDfFusionPass pass({AscendString("DepthwiseConv2DBackpropInputD")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv2DBackpropInputD"));
}

TEST_F(DepthwiseDfFusionPassTest, dynamicHwcnBf16Success)
{
    auto builder = EsGraphBuilder("dynamicHwcnBf16Success");
    auto inputSize = builder.CreateInput(0, "input_size", DT_INT32, FORMAT_ND, {INPUT_SIZE_DIM});
    auto filter = builder.CreateInput(1, "filter", DT_BF16, FORMAT_HWCN, {3, 3, 1, 32});
    auto outBackprop = builder.CreateInput(2, "out_backprop", DT_BF16, FORMAT_NCHW, {2, 32, 16, 16});

    auto y = CreateDepthwiseConv2DBpInputNode(builder, "DepthwiseConv2DBackpropInput", inputSize, filter, outBackprop,
                                              {1, 1, 2, 2}, {0, 0, 1, 1}, {1, 1, 1, 1}, 32, "NCHW", DT_BF16,
                                              {2, 32, 32, 32}, FORMAT_NCHW);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::DepthwiseDfFusionPass pass({AscendString("DepthwiseConv2DBackpropInput")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv2DBackpropInput"));
}

TEST_F(DepthwiseDfFusionPassTest, nonArch35NchwTransposeDSuccess)
{
    SetPlatform("Ascend910_93");
    auto builder = EsGraphBuilder("nonArch35NchwTransposeDSuccess");
    auto filter = builder.CreateInput(0, "filter", DT_FLOAT16, FORMAT_NCHW, {32, 1, 3, 3});
    auto outBackprop = builder.CreateInput(1, "out_backprop", DT_FLOAT16, FORMAT_NCHW, {2, 32, 16, 16});

    auto y = CreateDepthwiseConv2DBpInputDNode(builder, "DepthwiseConv2DBackpropInputD", filter, outBackprop,
                                               {1, 1, 2, 2}, {0, 0, 1, 1}, {1, 1, 1, 1}, 32, "NCHW", DT_FLOAT16,
                                               {2, 32, 32, 32}, FORMAT_NCHW);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::DepthwiseDfFusionPass pass({AscendString("DepthwiseConv2DBackpropInputD")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "DepthwiseConv2DBackpropInputD"));
    EXPECT_TRUE(CheckNodeExists(graph, "TransposeD"));
    EXPECT_FALSE(CheckNodeExists(graph, "Transpose"));
}

TEST_F(DepthwiseDfFusionPassTest, groupsZeroFail)
{
    auto builder = EsGraphBuilder("groupsZeroFail");
    auto filter = builder.CreateInput(0, "filter", DT_FLOAT16, FORMAT_NCHW, {32, 1, 3, 3});
    auto outBackprop = builder.CreateInput(1, "out_backprop", DT_FLOAT16, FORMAT_NCHW, {2, 32, 16, 16});

    auto y = CreateDepthwiseConv2DBpInputDNode(builder, "DepthwiseConv2DBackpropInputD", filter, outBackprop,
                                               {1, 1, 2, 2}, {0, 0, 1, 1}, {1, 1, 1, 1}, 0, "NCHW", DT_FLOAT16,
                                               {2, 32, 32, 32}, FORMAT_NCHW);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::DepthwiseDfFusionPass pass({AscendString("DepthwiseConv2DBackpropInputD")});
    EXPECT_EQ(pass.Run(graph, ctx), GRAPH_NOT_CHANGED);
    EXPECT_FALSE(CheckNodeExists(graph, "Conv2DBackpropInputD"));
}

TEST_F(DepthwiseDfFusionPassTest, groupsTooLargeFail)
{
    auto builder = EsGraphBuilder("groupsTooLargeFail");
    auto filter = builder.CreateInput(0, "filter", DT_FLOAT16, FORMAT_NCHW, {32, 1, 3, 3});
    auto outBackprop = builder.CreateInput(1, "out_backprop", DT_FLOAT16, FORMAT_NCHW, {2, 32, 16, 16});

    auto y = CreateDepthwiseConv2DBpInputDNode(builder, "DepthwiseConv2DBackpropInputD", filter, outBackprop,
                                               {1, 1, 2, 2}, {0, 0, 1, 1}, {1, 1, 1, 1}, 65536, "NCHW", DT_FLOAT16,
                                               {2, 32, 32, 32}, FORMAT_NCHW);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::DepthwiseDfFusionPass pass({AscendString("DepthwiseConv2DBackpropInputD")});
    EXPECT_EQ(pass.Run(graph, ctx), GRAPH_NOT_CHANGED);
    EXPECT_FALSE(CheckNodeExists(graph, "Conv2DBackpropInputD"));
}

TEST_F(DepthwiseDfFusionPassTest, groupsMaxBoundarySuccess)
{
    auto builder = EsGraphBuilder("groupsMaxBoundarySuccess");
    auto filter = builder.CreateInput(0, "filter", DT_FLOAT16, FORMAT_HWCN, {3, 3, 1, 32});
    auto outBackprop = builder.CreateInput(1, "out_backprop", DT_FLOAT16, FORMAT_NCHW, {2, 32, 16, 16});

    auto y = CreateDepthwiseConv2DBpInputDNode(builder, "DepthwiseConv2DBackpropInputD", filter, outBackprop,
                                               {1, 1, 2, 2}, {0, 0, 1, 1}, {1, 1, 1, 1}, 65535, "NCHW", DT_FLOAT16,
                                               {2, 32, 32, 32}, FORMAT_NCHW);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::DepthwiseDfFusionPass pass({AscendString("DepthwiseConv2DBackpropInputD")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv2DBackpropInputD"));
}

TEST_F(DepthwiseDfFusionPassTest, groupsMinBoundarySuccess)
{
    auto builder = EsGraphBuilder("groupsMinBoundarySuccess");
    auto filter = builder.CreateInput(0, "filter", DT_FLOAT16, FORMAT_HWCN, {3, 3, 1, 32});
    auto outBackprop = builder.CreateInput(1, "out_backprop", DT_FLOAT16, FORMAT_NCHW, {2, 32, 16, 16});

    auto y = CreateDepthwiseConv2DBpInputDNode(builder, "DepthwiseConv2DBackpropInputD", filter, outBackprop,
                                               {1, 1, 2, 2}, {0, 0, 1, 1}, {1, 1, 1, 1}, 1, "NCHW", DT_FLOAT16,
                                               {2, 32, 32, 32}, FORMAT_NCHW);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::DepthwiseDfFusionPass pass({AscendString("DepthwiseConv2DBackpropInputD")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv2DBackpropInputD"));
}

TEST_F(DepthwiseDfFusionPassTest, invalidFilterDimFail)
{
    auto builder = EsGraphBuilder("invalidFilterDimFail");
    auto filter = builder.CreateInput(0, "filter", DT_FLOAT16, FORMAT_NCHW, {32, 1, 3});
    auto outBackprop = builder.CreateInput(1, "out_backprop", DT_FLOAT16, FORMAT_NCHW, {2, 32, 16, 16});

    auto y = CreateDepthwiseConv2DBpInputDNode(builder, "DepthwiseConv2DBackpropInputD", filter, outBackprop,
                                               {1, 1, 2, 2}, {0, 0, 1, 1}, {1, 1, 1, 1}, 32, "NCHW", DT_FLOAT16,
                                               {2, 32, 32, 32}, FORMAT_NCHW);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::DepthwiseDfFusionPass pass({AscendString("DepthwiseConv2DBackpropInputD")});
    EXPECT_NE(pass.Run(graph, ctx), SUCCESS);
    EXPECT_FALSE(CheckNodeExists(graph, "Conv2DBackpropInputD"));
}

TEST_F(DepthwiseDfFusionPassTest, invalidFilterFormatFail)
{
    auto builder = EsGraphBuilder("invalidFilterFormatFail");
    auto filter = builder.CreateInput(0, "filter", DT_FLOAT16, FORMAT_ND, {32, 1, 3, 3});
    auto outBackprop = builder.CreateInput(1, "out_backprop", DT_FLOAT16, FORMAT_NCHW, {2, 32, 16, 16});

    auto y = CreateDepthwiseConv2DBpInputDNode(builder, "DepthwiseConv2DBackpropInputD", filter, outBackprop,
                                               {1, 1, 2, 2}, {0, 0, 1, 1}, {1, 1, 1, 1}, 32, "NCHW", DT_FLOAT16,
                                               {2, 32, 32, 32}, FORMAT_NCHW);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::DepthwiseDfFusionPass pass({AscendString("DepthwiseConv2DBackpropInputD")});
    EXPECT_EQ(pass.Run(graph, ctx), GRAPH_NOT_CHANGED);
    EXPECT_FALSE(CheckNodeExists(graph, "Conv2DBackpropInputD"));
}

TEST_F(DepthwiseDfFusionPassTest, formatMismatchFail)
{
    auto builder = EsGraphBuilder("formatMismatchFail");
    auto filter = builder.CreateInput(0, "filter", DT_FLOAT16, FORMAT_NCHW, {32, 1, 3, 3});
    auto outBackprop = builder.CreateInput(1, "out_backprop", DT_FLOAT16, FORMAT_NCHW, {2, 32, 16, 16});

    auto y = CreateDepthwiseConv2DBpInputDNode(builder, "DepthwiseConv2DBackpropInputD", filter, outBackprop,
                                               {1, 1, 2, 2}, {0, 0, 1, 1}, {1, 1, 1, 1}, 32, "NCHW", DT_FLOAT16,
                                               {2, 32, 32, 32}, FORMAT_NHWC);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::DepthwiseDfFusionPass pass({AscendString("DepthwiseConv2DBackpropInputD")});
    EXPECT_EQ(pass.Run(graph, ctx), GRAPH_NOT_CHANGED);
    EXPECT_FALSE(CheckNodeExists(graph, "Conv2DBackpropInputD"));
}

TEST_F(DepthwiseDfFusionPassTest, unknownRankDedySuccess)
{
    auto builder = EsGraphBuilder("unknownRankDedySuccess");
    auto inputSize = builder.CreateInput(0, "input_size", DT_INT32, FORMAT_ND, {INPUT_SIZE_DIM});
    auto filter = builder.CreateInput(1, "filter", DT_FLOAT16, FORMAT_HWCN, {3, 3, 1, 32});
    auto outBackprop = builder.CreateInput(2, "out_backprop", DT_FLOAT16, FORMAT_NCHW, {-2});

    auto y = CreateDepthwiseConv2DBpInputNode(builder, "DepthwiseConv2DBackpropInput", inputSize, filter, outBackprop,
                                              {1, 1, 2, 2}, {0, 0, 1, 1}, {1, 1, 1, 1}, 32, "NCHW", DT_FLOAT16, {-2},
                                              FORMAT_NCHW);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::DepthwiseDfFusionPass pass({AscendString("DepthwiseConv2DBackpropInput")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv2DBackpropInput"));
}

TEST_F(DepthwiseDfFusionPassTest, fromDepthwisePropagatedSuccess)
{
    auto builder = EsGraphBuilder("fromDepthwisePropagatedSuccess");
    auto filter = builder.CreateInput(0, "filter", DT_FLOAT16, FORMAT_HWCN, {3, 3, 1, 32});
    auto outBackprop = builder.CreateInput(1, "out_backprop", DT_FLOAT16, FORMAT_NCHW, {2, 32, 16, 16});

    auto y = CreateDepthwiseConv2DBpInputDNode(builder, "DepthwiseConv2DBackpropInputD", filter, outBackprop,
                                               {1, 1, 2, 2}, {0, 0, 1, 1}, {1, 1, 1, 1}, 32, "NCHW", DT_FLOAT16,
                                               {2, 32, 32, 32}, FORMAT_NCHW, true);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::DepthwiseDfFusionPass pass({AscendString("DepthwiseConv2DBackpropInputD")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv2DBackpropInputD"));
    bool fromDepthwise = false;
    EXPECT_TRUE(GetNodeBoolAttr(graph, "Conv2DBackpropInputD", "from_depthwise", fromDepthwise));
    EXPECT_TRUE(fromDepthwise);
}

TEST_F(DepthwiseDfFusionPassTest, fromDepthwiseDefaultFalse)
{
    auto builder = EsGraphBuilder("fromDepthwiseDefaultFalse");
    auto filter = builder.CreateInput(0, "filter", DT_FLOAT16, FORMAT_HWCN, {3, 3, 1, 32});
    auto outBackprop = builder.CreateInput(1, "out_backprop", DT_FLOAT16, FORMAT_NCHW, {2, 32, 16, 16});

    auto y = CreateDepthwiseConv2DBpInputDNode(builder, "DepthwiseConv2DBackpropInputD", filter, outBackprop,
                                               {1, 1, 2, 2}, {0, 0, 1, 1}, {1, 1, 1, 1}, 32, "NCHW", DT_FLOAT16,
                                               {2, 32, 32, 32}, FORMAT_NCHW);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::DepthwiseDfFusionPass pass({AscendString("DepthwiseConv2DBackpropInputD")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv2DBackpropInputD"));
    bool fromDepthwise = true;
    EXPECT_TRUE(GetNodeBoolAttr(graph, "Conv2DBackpropInputD", "from_depthwise", fromDepthwise));
    EXPECT_FALSE(fromDepthwise);
}

TEST_F(DepthwiseDfFusionPassTest, staticDInputSizeAttrPropagated)
{
    auto builder = EsGraphBuilder("staticDInputSizeAttrPropagated");
    auto filter = builder.CreateInput(0, "filter", DT_FLOAT16, FORMAT_HWCN, {3, 3, 1, 32});
    auto outBackprop = builder.CreateInput(1, "out_backprop", DT_FLOAT16, FORMAT_NCHW, {2, 32, 16, 16});

    auto y = CreateDepthwiseConv2DBpInputDNode(builder, "DepthwiseConv2DBackpropInputD", filter, outBackprop,
                                               {1, 1, 2, 2}, {0, 0, 1, 1}, {1, 1, 1, 1}, 32, "NCHW", DT_FLOAT16,
                                               {2, 32, 32, 32}, FORMAT_NCHW);
    auto* graph = builder.GetCGraphBuilder()->GetGraph();
    for (auto node : graph->GetAllNodes()) {
        AscendString nodeType;
        node.GetType(nodeType);
        if (std::string(nodeType.GetString()) == "DepthwiseConv2DBackpropInputD") {
            std::vector<int64_t> inputSizeAttr = {2, 32, 32, 32};
            node.SetAttr("input_size", inputSizeAttr);
            break;
        }
    }

    std::shared_ptr<Graph> graphBuilt = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::DepthwiseDfFusionPass pass({AscendString("DepthwiseConv2DBackpropInputD")});
    EXPECT_EQ(pass.Run(graphBuilt, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graphBuilt, "Conv2DBackpropInputD"));
}
