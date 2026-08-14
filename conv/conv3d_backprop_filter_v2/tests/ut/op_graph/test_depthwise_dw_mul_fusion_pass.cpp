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
#include "../../../op_graph/fusion_pass/depthwise_dw_mul_fusion_pass.h"

using namespace ge;
using namespace ge::es;
using namespace fe;
using namespace fusion;
using namespace ops::ConvBackpropFusionUtils;

namespace {

constexpr int64_t AI_CORE_CNT = 64;
constexpr int64_t FILTER_SIZE_DIM = 4;
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

EsTensorHolder CreateDepthwiseConv2DBpFilterDNode(EsGraphBuilder& builder, const char* opType, const EsTensorHolder& x,
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
                    .IrDefInputs({{"x", CompliantNodeBuilder::kEsIrInputRequired, ""},
                                  {"out_backprop", CompliantNodeBuilder::kEsIrInputRequired, ""}})
                    .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                    .InstanceOutputDataType("y", outDtype)
                    .InstanceOutputShape("y", outShape)
                    .InstanceOutputFormat("y", outFormat)
                    .Build();

    AddEdgeAndUpdatePeerDesc(*graph, *x.GetProducer(), x.GetProducerOutIndex(), node, 0);
    AddEdgeAndUpdatePeerDesc(*graph, *outBackprop.GetProducer(), outBackprop.GetProducerOutIndex(), node, 1);

    TensorDesc xDesc, outBackpropDesc;
    x.GetProducer()->GetOutputDesc(x.GetProducerOutIndex(), xDesc);
    outBackprop.GetProducer()->GetOutputDesc(outBackprop.GetProducerOutIndex(), outBackpropDesc);
    node.UpdateInputDesc(0, xDesc);
    node.UpdateInputDesc(1, outBackpropDesc);

    node.SetAttr("strides", strides);
    node.SetAttr("pads", pads);
    node.SetAttr("dilations", dilations);
    node.SetAttr("groups", groups);
    AscendString fmt = dataFormat.c_str();
    node.SetAttr("data_format", fmt);
    node.SetAttr("_op_impl_mode_enum", DEFAULT_IMPL_MODE);
    node.SetAttr("from_depthwise", fromDepthwise);
    std::vector<int64_t> filterSizeAttr = outShape;
    node.SetAttr("filter_size", filterSizeAttr);

    return EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(node, 0));
}

EsTensorHolder CreateDepthwiseConv2DBpFilterNode(EsGraphBuilder& builder, const char* opType, const EsTensorHolder& x,
                                                 const EsTensorHolder& filterSize, const EsTensorHolder& outBackprop,
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

class DepthwiseDwMulFusionPassTest : public testing::Test {
protected:
    void SetUp() override { SetPlatform("Ascend950"); }
};

TEST_F(DepthwiseDwMulFusionPassTest, staticDNchwFp16Success)
{
    auto builder = EsGraphBuilder("staticDNchwFp16Success");
    auto x = builder.CreateInput(0, "x", DT_FLOAT16, FORMAT_NCHW, {2, 32, 16, 16});
    auto outBackprop = builder.CreateInput(1, "out_backprop", DT_FLOAT16, FORMAT_NCHW, {2, 32, 16, 16});

    auto y = CreateDepthwiseConv2DBpFilterDNode(builder, "DepthwiseConv2DBackpropFilterD", x, outBackprop, {1, 1, 2, 2},
                                                {0, 0, 1, 1}, {1, 1, 1, 1}, 32, "NCHW", DT_FLOAT16, {32, 1, 3, 3},
                                                FORMAT_NCHW);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::DepthwiseDwMulFusionPass pass({AscendString("DepthwiseConv2DBackpropFilterD")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv2DBackpropFilterD"));
    EXPECT_TRUE(CheckNodeExists(graph, "Reshape"));
    EXPECT_TRUE(CheckNodeExists(graph, "Transpose"));
}

TEST_F(DepthwiseDwMulFusionPassTest, dynamicNchwFp16Success)
{
    auto builder = EsGraphBuilder("dynamicNchwFp16Success");
    auto x = builder.CreateInput(0, "x", DT_FLOAT16, FORMAT_NCHW, {2, 32, 16, 16});
    auto filterSize = builder.CreateInput(1, "filter_size", DT_INT32, FORMAT_ND, {FILTER_SIZE_DIM});
    auto outBackprop = builder.CreateInput(2, "out_backprop", DT_FLOAT16, FORMAT_NCHW, {2, 32, 16, 16});

    auto y = CreateDepthwiseConv2DBpFilterNode(builder, "DepthwiseConv2DBackpropFilter", x, filterSize, outBackprop,
                                               {1, 1, 2, 2}, {0, 0, 1, 1}, {1, 1, 1, 1}, 32, "NCHW", DT_FLOAT16,
                                               {32, 1, 3, 3}, FORMAT_NCHW);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::DepthwiseDwMulFusionPass pass({AscendString("DepthwiseConv2DBackpropFilter")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv2DBackpropFilter"));
    EXPECT_TRUE(CheckNodeExists(graph, "Reshape"));
    EXPECT_TRUE(CheckNodeExists(graph, "Transpose"));
}

TEST_F(DepthwiseDwMulFusionPassTest, staticDHwcnFp16Success)
{
    auto builder = EsGraphBuilder("staticDHwcnFp16Success");
    auto x = builder.CreateInput(0, "x", DT_FLOAT16, FORMAT_NCHW, {2, 32, 16, 16});
    auto outBackprop = builder.CreateInput(1, "out_backprop", DT_FLOAT16, FORMAT_NCHW, {2, 32, 16, 16});

    auto y = CreateDepthwiseConv2DBpFilterDNode(builder, "DepthwiseConv2DBackpropFilterD", x, outBackprop, {1, 1, 2, 2},
                                                {0, 0, 1, 1}, {1, 1, 1, 1}, 32, "NCHW", DT_FLOAT16, {3, 3, 1, 32},
                                                FORMAT_HWCN);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::DepthwiseDwMulFusionPass pass({AscendString("DepthwiseConv2DBackpropFilterD")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv2DBackpropFilterD"));
    EXPECT_TRUE(CheckNodeExists(graph, "Reshape"));
    EXPECT_FALSE(CheckNodeExists(graph, "Transpose"));
}

TEST_F(DepthwiseDwMulFusionPassTest, dynamicHwcnFp16Success)
{
    auto builder = EsGraphBuilder("dynamicHwcnFp16Success");
    auto x = builder.CreateInput(0, "x", DT_FLOAT16, FORMAT_NCHW, {2, 32, 16, 16});
    auto filterSize = builder.CreateInput(1, "filter_size", DT_INT32, FORMAT_ND, {FILTER_SIZE_DIM});
    auto outBackprop = builder.CreateInput(2, "out_backprop", DT_FLOAT16, FORMAT_NCHW, {2, 32, 16, 16});

    auto y = CreateDepthwiseConv2DBpFilterNode(builder, "DepthwiseConv2DBackpropFilter", x, filterSize, outBackprop,
                                               {1, 1, 2, 2}, {0, 0, 1, 1}, {1, 1, 1, 1}, 32, "NCHW", DT_FLOAT16,
                                               {3, 3, 1, 32}, FORMAT_HWCN);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::DepthwiseDwMulFusionPass pass({AscendString("DepthwiseConv2DBackpropFilter")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv2DBackpropFilter"));
    EXPECT_TRUE(CheckNodeExists(graph, "Reshape"));
    EXPECT_FALSE(CheckNodeExists(graph, "Transpose"));
}

TEST_F(DepthwiseDwMulFusionPassTest, staticDNchwBf16Success)
{
    auto builder = EsGraphBuilder("staticDNchwBf16Success");
    auto x = builder.CreateInput(0, "x", DT_BF16, FORMAT_NCHW, {2, 32, 16, 16});
    auto outBackprop = builder.CreateInput(1, "out_backprop", DT_BF16, FORMAT_NCHW, {2, 32, 16, 16});

    auto y = CreateDepthwiseConv2DBpFilterDNode(builder, "DepthwiseConv2DBackpropFilterD", x, outBackprop, {1, 1, 2, 2},
                                                {0, 0, 1, 1}, {1, 1, 1, 1}, 32, "NCHW", DT_BF16, {32, 1, 3, 3},
                                                FORMAT_NCHW);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::DepthwiseDwMulFusionPass pass({AscendString("DepthwiseConv2DBackpropFilterD")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv2DBackpropFilterD"));
}

TEST_F(DepthwiseDwMulFusionPassTest, staticDNchwFp32Success)
{
    auto builder = EsGraphBuilder("staticDNchwFp32Success");
    auto x = builder.CreateInput(0, "x", DT_FLOAT, FORMAT_NCHW, {2, 32, 16, 16});
    auto outBackprop = builder.CreateInput(1, "out_backprop", DT_FLOAT, FORMAT_NCHW, {2, 32, 16, 16});

    auto y = CreateDepthwiseConv2DBpFilterDNode(builder, "DepthwiseConv2DBackpropFilterD", x, outBackprop, {1, 1, 2, 2},
                                                {0, 0, 1, 1}, {1, 1, 1, 1}, 32, "NCHW", DT_FLOAT, {32, 1, 3, 3},
                                                FORMAT_NCHW);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::DepthwiseDwMulFusionPass pass({AscendString("DepthwiseConv2DBackpropFilterD")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv2DBackpropFilterD"));
}

TEST_F(DepthwiseDwMulFusionPassTest, dynamicHwcnBf16Success)
{
    auto builder = EsGraphBuilder("dynamicHwcnBf16Success");
    auto x = builder.CreateInput(0, "x", DT_BF16, FORMAT_NCHW, {2, 32, 16, 16});
    auto filterSize = builder.CreateInput(1, "filter_size", DT_INT32, FORMAT_ND, {FILTER_SIZE_DIM});
    auto outBackprop = builder.CreateInput(2, "out_backprop", DT_BF16, FORMAT_NCHW, {2, 32, 16, 16});

    auto y = CreateDepthwiseConv2DBpFilterNode(builder, "DepthwiseConv2DBackpropFilter", x, filterSize, outBackprop,
                                               {1, 1, 2, 2}, {0, 0, 1, 1}, {1, 1, 1, 1}, 32, "NCHW", DT_BF16,
                                               {3, 3, 1, 32}, FORMAT_HWCN);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::DepthwiseDwMulFusionPass pass({AscendString("DepthwiseConv2DBackpropFilter")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv2DBackpropFilter"));
}

TEST_F(DepthwiseDwMulFusionPassTest, nonArch35NchwTransposeDSuccess)
{
    SetPlatform("Ascend910_93");
    auto builder = EsGraphBuilder("nonArch35NchwTransposeDSuccess");
    auto x = builder.CreateInput(0, "x", DT_FLOAT16, FORMAT_NCHW, {2, 32, 16, 16});
    auto outBackprop = builder.CreateInput(1, "out_backprop", DT_FLOAT16, FORMAT_NCHW, {2, 32, 16, 16});

    auto y = CreateDepthwiseConv2DBpFilterDNode(builder, "DepthwiseConv2DBackpropFilterD", x, outBackprop, {1, 1, 2, 2},
                                                {0, 0, 1, 1}, {1, 1, 1, 1}, 32, "NCHW", DT_FLOAT16, {32, 1, 3, 3},
                                                FORMAT_NCHW);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::DepthwiseDwMulFusionPass pass({AscendString("DepthwiseConv2DBackpropFilterD")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv2DBackpropFilterD"));
    EXPECT_TRUE(CheckNodeExists(graph, "TransposeD"));
    EXPECT_FALSE(CheckNodeExists(graph, "Transpose"));
}

TEST_F(DepthwiseDwMulFusionPassTest, groupsZeroFail)
{
    auto builder = EsGraphBuilder("groupsZeroFail");
    auto x = builder.CreateInput(0, "x", DT_FLOAT16, FORMAT_NCHW, {2, 32, 16, 16});
    auto outBackprop = builder.CreateInput(1, "out_backprop", DT_FLOAT16, FORMAT_NCHW, {2, 32, 16, 16});

    auto y = CreateDepthwiseConv2DBpFilterDNode(builder, "DepthwiseConv2DBackpropFilterD", x, outBackprop, {1, 1, 2, 2},
                                                {0, 0, 1, 1}, {1, 1, 1, 1}, 0, "NCHW", DT_FLOAT16, {32, 1, 3, 3},
                                                FORMAT_NCHW);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::DepthwiseDwMulFusionPass pass({AscendString("DepthwiseConv2DBackpropFilterD")});
    EXPECT_EQ(pass.Run(graph, ctx), GRAPH_NOT_CHANGED);
    EXPECT_FALSE(CheckNodeExists(graph, "Conv2DBackpropFilterD"));
}

TEST_F(DepthwiseDwMulFusionPassTest, groupsTooLargeFail)
{
    auto builder = EsGraphBuilder("groupsTooLargeFail");
    auto x = builder.CreateInput(0, "x", DT_FLOAT16, FORMAT_NCHW, {2, 32, 16, 16});
    auto outBackprop = builder.CreateInput(1, "out_backprop", DT_FLOAT16, FORMAT_NCHW, {2, 32, 16, 16});

    auto y = CreateDepthwiseConv2DBpFilterDNode(builder, "DepthwiseConv2DBackpropFilterD", x, outBackprop, {1, 1, 2, 2},
                                                {0, 0, 1, 1}, {1, 1, 1, 1}, 65536, "NCHW", DT_FLOAT16, {32, 1, 3, 3},
                                                FORMAT_NCHW);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::DepthwiseDwMulFusionPass pass({AscendString("DepthwiseConv2DBackpropFilterD")});
    EXPECT_EQ(pass.Run(graph, ctx), GRAPH_NOT_CHANGED);
    EXPECT_FALSE(CheckNodeExists(graph, "Conv2DBackpropFilterD"));
}

TEST_F(DepthwiseDwMulFusionPassTest, groupsMaxBoundarySuccess)
{
    auto builder = EsGraphBuilder("groupsMaxBoundarySuccess");
    auto x = builder.CreateInput(0, "x", DT_FLOAT16, FORMAT_NCHW, {2, 32, 16, 16});
    auto outBackprop = builder.CreateInput(1, "out_backprop", DT_FLOAT16, FORMAT_NCHW, {2, 32, 16, 16});

    auto y = CreateDepthwiseConv2DBpFilterDNode(builder, "DepthwiseConv2DBackpropFilterD", x, outBackprop, {1, 1, 2, 2},
                                                {0, 0, 1, 1}, {1, 1, 1, 1}, 65535, "NCHW", DT_FLOAT16, {32, 1, 3, 3},
                                                FORMAT_HWCN);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::DepthwiseDwMulFusionPass pass({AscendString("DepthwiseConv2DBackpropFilterD")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv2DBackpropFilterD"));
}

TEST_F(DepthwiseDwMulFusionPassTest, groupsMinBoundarySuccess)
{
    auto builder = EsGraphBuilder("groupsMinBoundarySuccess");
    auto x = builder.CreateInput(0, "x", DT_FLOAT16, FORMAT_NCHW, {2, 32, 16, 16});
    auto outBackprop = builder.CreateInput(1, "out_backprop", DT_FLOAT16, FORMAT_NCHW, {2, 32, 16, 16});

    auto y = CreateDepthwiseConv2DBpFilterDNode(builder, "DepthwiseConv2DBackpropFilterD", x, outBackprop, {1, 1, 2, 2},
                                                {0, 0, 1, 1}, {1, 1, 1, 1}, 1, "NCHW", DT_FLOAT16, {32, 1, 3, 3},
                                                FORMAT_HWCN);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::DepthwiseDwMulFusionPass pass({AscendString("DepthwiseConv2DBackpropFilterD")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv2DBackpropFilterD"));
}

TEST_F(DepthwiseDwMulFusionPassTest, invalidFilterDimFail)
{
    auto builder = EsGraphBuilder("invalidFilterDimFail");
    auto x = builder.CreateInput(0, "x", DT_FLOAT16, FORMAT_NCHW, {2, 32, 16, 16});
    auto outBackprop = builder.CreateInput(1, "out_backprop", DT_FLOAT16, FORMAT_NCHW, {2, 32, 16, 16});

    auto y = CreateDepthwiseConv2DBpFilterDNode(builder, "DepthwiseConv2DBackpropFilterD", x, outBackprop, {1, 1, 2, 2},
                                                {0, 0, 1, 1}, {1, 1, 1, 1}, 32, "NCHW", DT_FLOAT16, {32, 1, 3},
                                                FORMAT_NCHW);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::DepthwiseDwMulFusionPass pass({AscendString("DepthwiseConv2DBackpropFilterD")});
    EXPECT_NE(pass.Run(graph, ctx), SUCCESS);
    EXPECT_FALSE(CheckNodeExists(graph, "Conv2DBackpropFilterD"));
}

TEST_F(DepthwiseDwMulFusionPassTest, invalidFilterFormatFail)
{
    auto builder = EsGraphBuilder("invalidFilterFormatFail");
    auto x = builder.CreateInput(0, "x", DT_FLOAT16, FORMAT_NCHW, {2, 32, 16, 16});
    auto outBackprop = builder.CreateInput(1, "out_backprop", DT_FLOAT16, FORMAT_NCHW, {2, 32, 16, 16});

    auto y = CreateDepthwiseConv2DBpFilterDNode(builder, "DepthwiseConv2DBackpropFilterD", x, outBackprop, {1, 1, 2, 2},
                                                {0, 0, 1, 1}, {1, 1, 1, 1}, 32, "NCHW", DT_FLOAT16, {32, 1, 3, 3},
                                                FORMAT_ND);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::DepthwiseDwMulFusionPass pass({AscendString("DepthwiseConv2DBackpropFilterD")});
    EXPECT_EQ(pass.Run(graph, ctx), GRAPH_NOT_CHANGED);
    EXPECT_FALSE(CheckNodeExists(graph, "Conv2DBackpropFilterD"));
}

TEST_F(DepthwiseDwMulFusionPassTest, fromDepthwisePropagatedSuccess)
{
    auto builder = EsGraphBuilder("fromDepthwisePropagatedSuccess");
    auto x = builder.CreateInput(0, "x", DT_FLOAT16, FORMAT_NCHW, {2, 32, 16, 16});
    auto outBackprop = builder.CreateInput(1, "out_backprop", DT_FLOAT16, FORMAT_NCHW, {2, 32, 16, 16});

    auto y = CreateDepthwiseConv2DBpFilterDNode(builder, "DepthwiseConv2DBackpropFilterD", x, outBackprop, {1, 1, 2, 2},
                                                {0, 0, 1, 1}, {1, 1, 1, 1}, 32, "NCHW", DT_FLOAT16, {3, 3, 1, 32},
                                                FORMAT_HWCN, true);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::DepthwiseDwMulFusionPass pass({AscendString("DepthwiseConv2DBackpropFilterD")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv2DBackpropFilterD"));
    bool fromDepthwise = false;
    EXPECT_TRUE(GetNodeBoolAttr(graph, "Conv2DBackpropFilterD", "from_depthwise", fromDepthwise));
    EXPECT_TRUE(fromDepthwise);
}

TEST_F(DepthwiseDwMulFusionPassTest, fromDepthwiseAlwaysTrue)
{
    auto builder = EsGraphBuilder("fromDepthwiseAlwaysTrue");
    auto x = builder.CreateInput(0, "x", DT_FLOAT16, FORMAT_NCHW, {2, 32, 16, 16});
    auto outBackprop = builder.CreateInput(1, "out_backprop", DT_FLOAT16, FORMAT_NCHW, {2, 32, 16, 16});

    auto y = CreateDepthwiseConv2DBpFilterDNode(builder, "DepthwiseConv2DBackpropFilterD", x, outBackprop, {1, 1, 2, 2},
                                                {0, 0, 1, 1}, {1, 1, 1, 1}, 32, "NCHW", DT_FLOAT16, {3, 3, 1, 32},
                                                FORMAT_HWCN);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::DepthwiseDwMulFusionPass pass({AscendString("DepthwiseConv2DBackpropFilterD")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv2DBackpropFilterD"));
    bool fromDepthwise = false;
    EXPECT_TRUE(GetNodeBoolAttr(graph, "Conv2DBackpropFilterD", "from_depthwise", fromDepthwise));
    EXPECT_TRUE(fromDepthwise);
}

TEST_F(DepthwiseDwMulFusionPassTest, staticDFilterSizeAttrSet)
{
    auto builder = EsGraphBuilder("staticDFilterSizeAttrSet");
    auto x = builder.CreateInput(0, "x", DT_FLOAT16, FORMAT_NCHW, {2, 32, 16, 16});
    auto outBackprop = builder.CreateInput(1, "out_backprop", DT_FLOAT16, FORMAT_NCHW, {2, 32, 16, 16});

    auto y = CreateDepthwiseConv2DBpFilterDNode(builder, "DepthwiseConv2DBackpropFilterD", x, outBackprop, {1, 1, 2, 2},
                                                {0, 0, 1, 1}, {1, 1, 1, 1}, 32, "NCHW", DT_FLOAT16, {3, 3, 1, 32},
                                                FORMAT_HWCN);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::DepthwiseDwMulFusionPass pass({AscendString("DepthwiseConv2DBackpropFilterD")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv2DBackpropFilterD"));
    bool found = false;
    for (auto node : graph->GetAllNodes()) {
        AscendString nodeType;
        node.GetType(nodeType);
        if (std::string(nodeType.GetString()) == "Conv2DBackpropFilterD") {
            std::vector<int64_t> filterSizeAttr;
            if (node.GetAttr("filter_size", filterSizeAttr) == GRAPH_SUCCESS && !filterSizeAttr.empty()) {
                found = true;
            }
        }
    }
    EXPECT_TRUE(found);
}

TEST_F(DepthwiseDwMulFusionPassTest, dynamicNchwFp32Success)
{
    auto builder = EsGraphBuilder("dynamicNchwFp32Success");
    auto x = builder.CreateInput(0, "x", DT_FLOAT, FORMAT_NCHW, {2, 32, 16, 16});
    auto filterSize = builder.CreateInput(1, "filter_size", DT_INT32, FORMAT_ND, {FILTER_SIZE_DIM});
    auto outBackprop = builder.CreateInput(2, "out_backprop", DT_FLOAT, FORMAT_NCHW, {2, 32, 16, 16});

    auto y = CreateDepthwiseConv2DBpFilterNode(builder, "DepthwiseConv2DBackpropFilter", x, filterSize, outBackprop,
                                               {1, 1, 2, 2}, {0, 0, 1, 1}, {1, 1, 1, 1}, 32, "NCHW", DT_FLOAT,
                                               {32, 1, 3, 3}, FORMAT_NCHW);

    std::shared_ptr<Graph> graph = builder.BuildAndReset({y});
    CustomPassContext ctx;
    ops::DepthwiseDwMulFusionPass pass({AscendString("DepthwiseConv2DBackpropFilter")});
    EXPECT_EQ(pass.Run(graph, ctx), SUCCESS);
    EXPECT_TRUE(CheckNodeExists(graph, "Conv2DBackpropFilter"));
    EXPECT_TRUE(CheckNodeExists(graph, "Reshape"));
    EXPECT_TRUE(CheckNodeExists(graph, "Transpose"));
}
