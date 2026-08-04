/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <vector>

#include "../../../../common/tests/ut/op_graph/test_conv_fusion_pass_framework.h"
#include "../../../op_graph/fusion_pass/split_conv2d_concat_fusion_pass.h"

#include "version/ge-compiler_version.h"
#if GE_COMPILER_VERSION_NUM >= 90000000U

using namespace ge;
using namespace es;
using namespace fe;
using namespace Ops;
using namespace NN;
using namespace Conv;
using namespace ConvFusionUtils;
using namespace SplitConv2dConcatFusion;
using namespace test_conv_fusion_framework;

#define CONV_DEBUG false

struct SplitConvGraphOptions {
    int32_t axis = 3;
    DataType dtype = DT_FLOAT16;
    DataType biasDtype = DT_UNDEFINED;
    Format fmt = FORMAT_NHWC;
    int32_t numSplit = 2;
    std::vector<int64_t> inputShape = {1, 7, 7, 32};
    std::vector<int64_t> filterShape = {3, 3, 16, 16};
    bool useSplitV = false;
    bool useConcatV2 = false;
    bool withBias = false;
    bool mismatchWeight = false;
    bool mismatchConcatFormat = false;
    bool mismatchWeightFormat = false;
    bool unknownWeightShape = false;
    bool mismatchInputCount = false;
    Format forceWeightFormat = FORMAT_RESERVED;
};

class SplitConv2dConcatFusionPassTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SplitConv2dConcatFusionPassTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "SplitConv2dConcatFusionPassTest TearDown" << std::endl; }

    GNode CreateFilterConst(Graph* graph, const std::string& name, DataType dtype, Format fmt,
                            const std::vector<int64_t>& shape)
    {
        auto node = CompliantNodeBuilder(graph)
                        .OpType("Const")
                        .Name(name.c_str())
                        .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                        .Build();
        size_t elemCount = 1;
        for (auto d : shape) {
            if (d > 0) {
                elemCount *= static_cast<size_t>(d);
            }
        }
        size_t bytesPerElem = 4;
        if (dtype == DT_FLOAT16) {
            bytesPerElem = 2;
        } else if (dtype == DT_INT8) {
            bytesPerElem = 1;
        }
        std::vector<uint8_t> data(elemCount * bytesPerElem, 0);
        Tensor filterTensor(TensorDesc(Shape(shape), fmt, dtype));
        filterTensor.SetData(data.data(), data.size());
        node.SetAttr(AscendString("value"), filterTensor);
        TensorDesc outDesc(Shape(shape), fmt, dtype);
        outDesc.SetOriginFormat(fmt);
        node.UpdateOutputDesc(0, outDesc);
        return node;
    }

    GNode CreateBiasConst(Graph* graph, const std::string& name, DataType dtype, int64_t biasLen)
    {
        return CreateFilterConst(graph, name, dtype, FORMAT_ND, {biasLen});
    }

    GNode CreateConv2DNode(Graph* graph, const std::string& name, DataType dtype, Format fmt,
                           const std::vector<int64_t>& inShape, const std::vector<int64_t>& filterShape, bool withBias)
    {
        auto node = CompliantNodeBuilder(graph)
                        .OpType("Conv2D")
                        .Name(name.c_str())
                        .IrDefInputs({{"x", CompliantNodeBuilder::kEsIrInputRequired, ""},
                                      {"filter", CompliantNodeBuilder::kEsIrInputRequired, ""},
                                      {"bias", CompliantNodeBuilder::kEsIrInputOptional, ""},
                                      {"offset_w", CompliantNodeBuilder::kEsIrInputOptional, ""}})
                        .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                        .Build();
        std::vector<int64_t> strides = {1, 1, 1, 1};
        std::vector<int64_t> pads = {0, 0, 0, 0};
        std::vector<int64_t> dilations = {1, 1, 1, 1};
        int64_t groups = 1;
        int64_t offsetX = 0;
        AscendString dataFormat(fmt == FORMAT_NHWC ? "NHWC" : "NCHW");
        node.SetAttr(AscendString("strides"), strides);
        node.SetAttr(AscendString("pads"), pads);
        node.SetAttr(AscendString("dilations"), dilations);
        node.SetAttr(AscendString("groups"), groups);
        node.SetAttr(AscendString("data_format"), dataFormat);
        node.SetAttr(AscendString("offset_x"), offsetX);

        Format filterFmt = (fmt == FORMAT_NHWC) ? FORMAT_HWCN : FORMAT_NCHW;
        TensorDesc fmapDesc(Shape(inShape), fmt, dtype);
        fmapDesc.SetOriginFormat(fmt);
        fmapDesc.SetOriginShape(Shape(inShape));
        node.UpdateInputDesc(0, fmapDesc);
        TensorDesc filterDesc(Shape(filterShape), filterFmt, dtype);
        filterDesc.SetOriginFormat(filterFmt);
        filterDesc.SetOriginShape(Shape(filterShape));
        node.UpdateInputDesc(1, filterDesc);
        if (withBias) {
            int64_t cout = (fmt == FORMAT_NHWC) ? filterShape[3] : filterShape[0];
            TensorDesc biasDesc(Shape({cout}), FORMAT_ND, dtype);
            biasDesc.SetOriginFormat(FORMAT_ND);
            biasDesc.SetOriginShape(Shape({cout}));
            node.UpdateInputDesc(2, biasDesc);
        }
        TensorDesc outDesc(Shape(inShape), fmt, dtype);
        outDesc.SetOriginFormat(fmt);
        outDesc.SetOriginShape(Shape(inShape));
        node.UpdateOutputDesc(0, outDesc);
        return node;
    }

    GNode BuildSplitNode(EsGraphBuilder& graphBuilder, const SplitConvGraphOptions& opt, const EsTensorHolder& input)
    {
        Graph* graph = graphBuilder.GetCGraphBuilder()->GetGraph();
        std::vector<int32_t> axisData = {opt.axis};
        auto splitDim = graphBuilder.CreateConst(axisData, {1}, DT_INT32);

        if (opt.useSplitV) {
            std::vector<int64_t> sizeSplits(opt.numSplit,
                                            opt.inputShape[(opt.fmt == FORMAT_NHWC) ? 3 : 1] / opt.numSplit);
            auto sizeConst = graphBuilder.CreateConst(sizeSplits, {opt.numSplit}, DT_INT64);
            auto splitNode = CompliantNodeBuilder(graph)
                                 .OpType("SplitV")
                                 .Name("splitv")
                                 .IrDefInputs({{"x", CompliantNodeBuilder::kEsIrInputRequired, ""},
                                               {"size_splits", CompliantNodeBuilder::kEsIrInputRequired, ""},
                                               {"split_dim", CompliantNodeBuilder::kEsIrInputRequired, ""}})
                                 .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputDynamic, ""}})
                                 .InstanceDynamicOutputNum("y", opt.numSplit)
                                 .Build();
            int64_t numSplitAttr = opt.numSplit;
            splitNode.SetAttr(AscendString("num_split"), numSplitAttr);
            graph->AddDataEdge(*input.GetProducer(), input.GetProducerOutIndex(), splitNode, 0);
            graph->AddDataEdge(*sizeConst.GetProducer(), sizeConst.GetProducerOutIndex(), splitNode, 1);
            graph->AddDataEdge(*splitDim.GetProducer(), splitDim.GetProducerOutIndex(), splitNode, 2);
            return splitNode;
        }

        auto splitNode = CompliantNodeBuilder(graph)
                             .OpType("Split")
                             .Name("split")
                             .IrDefInputs({{"split_dim", CompliantNodeBuilder::kEsIrInputRequired, ""},
                                           {"x", CompliantNodeBuilder::kEsIrInputRequired, ""}})
                             .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputDynamic, ""}})
                             .InstanceDynamicOutputNum("y", opt.numSplit)
                             .Build();
        int64_t numSplitAttr = opt.numSplit;
        splitNode.SetAttr(AscendString("num_split"), numSplitAttr);
        graph->AddDataEdge(*splitDim.GetProducer(), splitDim.GetProducerOutIndex(), splitNode, 0);
        graph->AddDataEdge(*input.GetProducer(), input.GetProducerOutIndex(), splitNode, 1);
        return splitNode;
    }

    GraphPtr BuildSplitConvConcatGraph(const SplitConvGraphOptions& opt)
    {
        EsGraphBuilder graphBuilder("test_split_conv_concat");
        auto input = graphBuilder.CreateInput(0, "input", opt.dtype, opt.fmt, opt.inputShape);
        Graph* graph = graphBuilder.GetCGraphBuilder()->GetGraph();

        auto splitNode = BuildSplitNode(graphBuilder, opt, input);
        int32_t dataIdx = opt.useSplitV ? 0 : 1;
        TensorDesc inputDesc(Shape(opt.inputShape), opt.fmt, opt.dtype);
        inputDesc.SetOriginFormat(opt.fmt);
        splitNode.UpdateInputDesc(dataIdx, inputDesc);

        std::vector<int64_t> splitOutShape = opt.inputShape;
        size_t cPos = (opt.fmt == FORMAT_NHWC) ? 3 : 1;
        splitOutShape[cPos] = opt.inputShape[cPos] / opt.numSplit;
        for (int32_t i = 0; i < opt.numSplit; ++i) {
            TensorDesc outDesc(Shape(splitOutShape), opt.fmt, opt.dtype);
            outDesc.SetOriginFormat(opt.fmt);
            splitNode.UpdateOutputDesc(i, outDesc);
        }

        Format defaultFilterFmt = (opt.fmt == FORMAT_NHWC) ? FORMAT_HWCN : FORMAT_NCHW;
        if (opt.forceWeightFormat != FORMAT_RESERVED) {
            defaultFilterFmt = opt.forceWeightFormat;
        }
        std::vector<GNode> convNodes;
        for (int32_t i = 0; i < opt.numSplit; ++i) {
            std::vector<int64_t> curFilterShape = opt.filterShape;
            if (opt.mismatchWeight && i == 1) {
                if (opt.fmt == FORMAT_NHWC) {
                    curFilterShape[3] = opt.filterShape[3] + 8;
                } else {
                    curFilterShape[0] = opt.filterShape[0] + 8;
                }
            }
            Format filterFmt = defaultFilterFmt;
            if (opt.mismatchWeightFormat && i == 1) {
                filterFmt = (defaultFilterFmt == FORMAT_HWCN) ? FORMAT_NCHW : FORMAT_HWCN;
            }
            bool branchBias = opt.withBias && !(opt.mismatchInputCount && i == 1);
            auto convNode = CreateConv2DNode(graph, "conv2d_" + std::to_string(i), opt.dtype, opt.fmt, splitOutShape,
                                             curFilterShape, branchBias);
            graph->AddDataEdge(splitNode, i, convNode, 0);
            std::vector<int64_t> filterDescShape = curFilterShape;
            if (opt.unknownWeightShape) {
                filterDescShape[0] = -1;
            }
            auto filter = CreateFilterConst(graph, "filter_" + std::to_string(i), opt.dtype, filterFmt, curFilterShape);
            if (opt.unknownWeightShape || opt.forceWeightFormat != FORMAT_RESERVED || opt.mismatchWeightFormat) {
                TensorDesc filterDesc(Shape(filterDescShape), filterFmt, opt.dtype);
                filterDesc.SetOriginFormat(filterFmt);
                filterDesc.SetOriginShape(Shape(filterDescShape));
                filter.UpdateOutputDesc(0, filterDesc);
                convNode.UpdateInputDesc(1, filterDesc);
            }
            graph->AddDataEdge(filter, 0, convNode, 1);
            if (branchBias) {
                int64_t biasLen = (opt.fmt == FORMAT_NHWC) ? curFilterShape[3] : curFilterShape[0];
                DataType biasDt = (opt.biasDtype == DT_UNDEFINED) ? opt.dtype : opt.biasDtype;
                auto bias = CreateBiasConst(graph, "bias_" + std::to_string(i), biasDt, biasLen);
                TensorDesc biasDesc(Shape({biasLen}), FORMAT_ND, biasDt);
                biasDesc.SetOriginFormat(FORMAT_ND);
                biasDesc.SetOriginShape(Shape({biasLen}));
                convNode.UpdateInputDesc(2, biasDesc);
                graph->AddDataEdge(bias, 0, convNode, 2);
            }
            convNodes.push_back(convNode);
        }

        std::vector<int32_t> axisData = {opt.axis};
        auto concatDim = graphBuilder.CreateConst(axisData, {1}, DT_INT32);
        std::vector<CompliantNodeBuilder::IrInputDef> concatInputs;
        if (opt.useConcatV2) {
            for (int32_t i = 0; i < opt.numSplit; ++i) {
                concatInputs.push_back({"x", CompliantNodeBuilder::kEsIrInputDynamic, ""});
            }
            concatInputs.push_back({"concat_dim", CompliantNodeBuilder::kEsIrInputRequired, ""});
        } else {
            concatInputs.push_back({"concat_dim", CompliantNodeBuilder::kEsIrInputRequired, ""});
            for (int32_t i = 0; i < opt.numSplit; ++i) {
                concatInputs.push_back({"x", CompliantNodeBuilder::kEsIrInputDynamic, ""});
            }
        }

        auto concatNode = CompliantNodeBuilder(graph)
                              .OpType(opt.useConcatV2 ? "ConcatV2" : "Concat")
                              .Name("concat")
                              .IrDefInputs(concatInputs)
                              .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                              .InstanceDynamicInputNum("x", opt.numSplit)
                              .Build();
        int64_t nAttr = opt.numSplit;
        concatNode.SetAttr(AscendString("N"), nAttr);

        if (opt.useConcatV2) {
            for (int32_t i = 0; i < opt.numSplit; ++i) {
                graph->AddDataEdge(convNodes[i], 0, concatNode, i);
                concatNode.UpdateInputDesc(i, TensorDesc(Shape(splitOutShape), opt.fmt, opt.dtype));
            }
            graph->AddDataEdge(*concatDim.GetProducer(), concatDim.GetProducerOutIndex(), concatNode, opt.numSplit);
        } else {
            graph->AddDataEdge(*concatDim.GetProducer(), concatDim.GetProducerOutIndex(), concatNode, 0);
            for (int32_t i = 0; i < opt.numSplit; ++i) {
                graph->AddDataEdge(convNodes[i], 0, concatNode, i + 1);
                concatNode.UpdateInputDesc(i + 1, TensorDesc(Shape(splitOutShape), opt.fmt, opt.dtype));
            }
        }

        Format concatOutFmt = opt.mismatchConcatFormat ? ((opt.fmt == FORMAT_NHWC) ? FORMAT_NCHW : FORMAT_NHWC) :
                                                         opt.fmt;
        TensorDesc concatOutDesc(Shape(opt.inputShape), concatOutFmt, opt.dtype);
        concatOutDesc.SetOriginFormat(concatOutFmt);
        concatNode.UpdateOutputDesc(0, concatOutDesc);

        auto concatHolder = EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(concatNode, 0));
        return graphBuilder.BuildAndReset({concatHolder});
    }

    GraphPtr BuildSplitConvConcatGraph(int32_t axis, DataType dtype, Format fmt, int32_t numSplit,
                                       const std::vector<int64_t>& inputShape, const std::vector<int64_t>& filterShape)
    {
        SplitConvGraphOptions opt;
        opt.axis = axis;
        opt.dtype = dtype;
        opt.fmt = fmt;
        opt.numSplit = numSplit;
        opt.inputShape = inputShape;
        opt.filterShape = filterShape;
        return BuildSplitConvConcatGraph(opt);
    }

    void TestTotalPass(const std::string& passName, GraphPtr& graph, Status expectRes)
    {
        if (CONV_DEBUG) {
            graph->DumpToFile(Graph::DumpFormat::kOnnx, AscendString((passName + "_before").c_str()));
        }
        CustomPassContext passContext;
        passContext.SetPassName(passName.c_str());
        ASplitConv2dConcatPass pass;
        auto res = pass.Run(graph, passContext);
        if (CONV_DEBUG) {
            graph->DumpToFile(Graph::DumpFormat::kOnnx, AscendString((passName + "_after").c_str()));
        }
        EXPECT_EQ(res, expectRes);
    }

    GraphPtr BuildSameGraphTwoStructures()
    {
        EsGraphBuilder graphBuilder("test_two_structures");
        SplitConvGraphOptions opt;
        opt.axis = 3;
        opt.fmt = FORMAT_NHWC;
        opt.numSplit = 2;
        opt.inputShape = {1, 7, 7, 32};
        opt.filterShape = {3, 3, 16, 16};

        auto input0 = graphBuilder.CreateInput(0, "input0", opt.dtype, opt.fmt, opt.inputShape);
        auto input1 = graphBuilder.CreateInput(1, "input1", opt.dtype, opt.fmt, opt.inputShape);
        Graph* graph = graphBuilder.GetCGraphBuilder()->GetGraph();

        auto buildOne = [&](const std::string& prefix, const EsTensorHolder& input) -> EsTensorHolder {
            std::vector<int32_t> axisData = {opt.axis};
            auto splitDim = graphBuilder.CreateConst(axisData, {1}, DT_INT32);
            auto splitNode = CompliantNodeBuilder(graph)
                                 .OpType("Split")
                                 .Name((prefix + "_split").c_str())
                                 .IrDefInputs({{"split_dim", CompliantNodeBuilder::kEsIrInputRequired, ""},
                                               {"x", CompliantNodeBuilder::kEsIrInputRequired, ""}})
                                 .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputDynamic, ""}})
                                 .InstanceDynamicOutputNum("y", opt.numSplit)
                                 .Build();
            int64_t numSplitAttr = opt.numSplit;
            splitNode.SetAttr(AscendString("num_split"), numSplitAttr);
            graph->AddDataEdge(*splitDim.GetProducer(), splitDim.GetProducerOutIndex(), splitNode, 0);
            graph->AddDataEdge(*input.GetProducer(), input.GetProducerOutIndex(), splitNode, 1);
            TensorDesc inputDesc(Shape(opt.inputShape), opt.fmt, opt.dtype);
            inputDesc.SetOriginFormat(opt.fmt);
            splitNode.UpdateInputDesc(1, inputDesc);
            std::vector<int64_t> splitOutShape = opt.inputShape;
            splitOutShape[3] = opt.inputShape[3] / opt.numSplit;
            for (int32_t i = 0; i < opt.numSplit; ++i) {
                TensorDesc outDesc(Shape(splitOutShape), opt.fmt, opt.dtype);
                outDesc.SetOriginFormat(opt.fmt);
                splitNode.UpdateOutputDesc(i, outDesc);
            }

            Format filterFmt = FORMAT_HWCN;
            std::vector<GNode> convNodes;
            for (int32_t i = 0; i < opt.numSplit; ++i) {
                auto convNode = CreateConv2DNode(graph, prefix + "_conv" + std::to_string(i), opt.dtype, opt.fmt,
                                                 splitOutShape, opt.filterShape, false);
                graph->AddDataEdge(splitNode, i, convNode, 0);
                auto filter = CreateFilterConst(graph, prefix + "_filter" + std::to_string(i), opt.dtype, filterFmt,
                                                opt.filterShape);
                graph->AddDataEdge(filter, 0, convNode, 1);
                convNodes.push_back(convNode);
            }

            auto concatDim = graphBuilder.CreateConst(axisData, {1}, DT_INT32);
            std::vector<CompliantNodeBuilder::IrInputDef> concatInputs = {
                {"concat_dim", CompliantNodeBuilder::kEsIrInputRequired, ""}};
            for (int32_t i = 0; i < opt.numSplit; ++i) {
                concatInputs.push_back({"x", CompliantNodeBuilder::kEsIrInputDynamic, ""});
            }
            auto concatNode = CompliantNodeBuilder(graph)
                                  .OpType("Concat")
                                  .Name((prefix + "_concat").c_str())
                                  .IrDefInputs(concatInputs)
                                  .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                                  .InstanceDynamicInputNum("x", opt.numSplit)
                                  .Build();
            int64_t nAttr = opt.numSplit;
            concatNode.SetAttr(AscendString("N"), nAttr);
            graph->AddDataEdge(*concatDim.GetProducer(), concatDim.GetProducerOutIndex(), concatNode, 0);
            for (int32_t i = 0; i < opt.numSplit; ++i) {
                graph->AddDataEdge(convNodes[i], 0, concatNode, i + 1);
            }
            TensorDesc concatOutDesc(Shape(opt.inputShape), opt.fmt, opt.dtype);
            concatOutDesc.SetOriginFormat(opt.fmt);
            concatNode.UpdateOutputDesc(0, concatOutDesc);
            return EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(concatNode, 0));
        };

        auto out0 = buildOne("s0", input0);
        auto out1 = buildOne("s1", input1);
        return graphBuilder.BuildAndReset({out0, out1});
    }
};

TEST_F(SplitConv2dConcatFusionPassTest, split_conv2d_concat_nhwc_axis3_success)
{
    auto graph = BuildSplitConvConcatGraph(3, DT_FLOAT16, FORMAT_NHWC, 2, {1, 7, 7, 32}, {3, 3, 16, 16});
    ASSERT_NE(graph, nullptr);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "Split"), 1);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "Conv2D"), 2);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "Concat"), 1);
    TestTotalPass("nhwc_axis3_success", graph, SUCCESS);
    EXPECT_FALSE(GraphChecker::HasNode(graph, "Split"));
    EXPECT_EQ(GraphChecker::CountNodes(graph, "Conv2D"), 1);
    EXPECT_TRUE(GraphChecker::HasNode(graph, "Concatv2HostCpuOp"));
    GNode groupConv;
    EXPECT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "Conv2D", groupConv));
    int64_t groupsAttr = 0;
    EXPECT_EQ(groupConv.GetAttr(AscendString("groups"), groupsAttr), GRAPH_SUCCESS);
    EXPECT_EQ(groupsAttr, 2);
    TensorDesc filterDesc;
    EXPECT_EQ(groupConv.GetInputDesc(1, filterDesc), GRAPH_SUCCESS);
    auto filterDims = filterDesc.GetOriginShape().GetDims();
    ASSERT_EQ(filterDims.size(), 4U);
    EXPECT_EQ(filterDims[3], 32);
}

TEST_F(SplitConv2dConcatFusionPassTest, split_conv2d_concat_nhwc_axis_neg1_success)
{
    auto graph = BuildSplitConvConcatGraph(-1, DT_FLOAT16, FORMAT_NHWC, 2, {1, 7, 7, 32}, {3, 3, 16, 16});
    ASSERT_NE(graph, nullptr);
    TestTotalPass("nhwc_axis_neg1_success", graph, SUCCESS);
    EXPECT_FALSE(GraphChecker::HasNode(graph, "Split"));
}

TEST_F(SplitConv2dConcatFusionPassTest, split_conv2d_concat_nchw_axis_neg1_reject)
{
    auto graph = BuildSplitConvConcatGraph(-1, DT_FLOAT16, FORMAT_NCHW, 2, {1, 32, 7, 7}, {16, 16, 3, 3});
    ASSERT_NE(graph, nullptr);
    TestTotalPass("nchw_axis_neg1_reject", graph, CONV_NOT_CHANGED);
    EXPECT_TRUE(GraphChecker::HasNode(graph, "Split"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "Concat"));
}

TEST_F(SplitConv2dConcatFusionPassTest, split_conv2d_concat_axis_mismatch_reject)
{
    auto graph = BuildSplitConvConcatGraph(2, DT_FLOAT16, FORMAT_NHWC, 2, {1, 7, 7, 32}, {3, 3, 16, 16});
    ASSERT_NE(graph, nullptr);
    TestTotalPass("axis_mismatch_reject", graph, CONV_NOT_CHANGED);
    EXPECT_TRUE(GraphChecker::HasNode(graph, "Split"));
}

TEST_F(SplitConv2dConcatFusionPassTest, split_conv2d_concat_3groups_success)
{
    auto graph = BuildSplitConvConcatGraph(3, DT_FLOAT16, FORMAT_NHWC, 3, {1, 7, 7, 48}, {3, 3, 16, 16});
    ASSERT_NE(graph, nullptr);
    TestTotalPass("3groups_success", graph, SUCCESS);
    EXPECT_FALSE(GraphChecker::HasNode(graph, "Split"));
    GNode groupConv;
    EXPECT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "Conv2D", groupConv));
    int64_t groupsAttr = 0;
    EXPECT_EQ(groupConv.GetAttr(AscendString("groups"), groupsAttr), GRAPH_SUCCESS);
    EXPECT_EQ(groupsAttr, 3);
}

TEST_F(SplitConv2dConcatFusionPassTest, split_conv2d_concat_nchw_axis1_success)
{
    auto graph = BuildSplitConvConcatGraph(1, DT_FLOAT16, FORMAT_NCHW, 2, {1, 32, 7, 7}, {16, 16, 3, 3});
    ASSERT_NE(graph, nullptr);
    TestTotalPass("nchw_axis1_success", graph, SUCCESS);
    EXPECT_FALSE(GraphChecker::HasNode(graph, "Split"));
    GNode groupConv;
    EXPECT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "Conv2D", groupConv));
    int64_t groupsAttr = 0;
    EXPECT_EQ(groupConv.GetAttr(AscendString("groups"), groupsAttr), GRAPH_SUCCESS);
    EXPECT_EQ(groupsAttr, 2);
    TensorDesc filterDesc;
    EXPECT_EQ(groupConv.GetInputDesc(1, filterDesc), GRAPH_SUCCESS);
    auto filterDims = filterDesc.GetOriginShape().GetDims();
    if (filterDims.empty()) {
        filterDims = filterDesc.GetShape().GetDims();
    }
    ASSERT_EQ(filterDims.size(), 4U);
    EXPECT_EQ(filterDims[0], 32);
}

TEST_F(SplitConv2dConcatFusionPassTest, split_conv2d_concat_unsupported_dtype_reject)
{
    auto graph = BuildSplitConvConcatGraph(3, DT_DOUBLE, FORMAT_NHWC, 2, {1, 7, 7, 32}, {3, 3, 16, 16});
    ASSERT_NE(graph, nullptr);
    TestTotalPass("unsupported_dtype_reject", graph, CONV_NOT_CHANGED);
    EXPECT_TRUE(GraphChecker::HasNode(graph, "Split"));
}

TEST_F(SplitConv2dConcatFusionPassTest, split_conv2d_concat_with_bias_success)
{
    SplitConvGraphOptions opt;
    opt.withBias = true;
    auto graph = BuildSplitConvConcatGraph(opt);
    ASSERT_NE(graph, nullptr);
    TestTotalPass("with_bias_success", graph, SUCCESS);
    EXPECT_FALSE(GraphChecker::HasNode(graph, "Split"));
    EXPECT_EQ(GraphChecker::CountNodes(graph, "Concatv2HostCpuOp"), 2);
    GNode groupConv;
    EXPECT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "Conv2D", groupConv));
    auto biasPair = groupConv.GetInDataNodesAndPortIndexs(2);
    EXPECT_NE(biasPair.first, nullptr);
}

TEST_F(SplitConv2dConcatFusionPassTest, splitv_conv2d_concatv2_success)
{
    SplitConvGraphOptions opt;
    opt.useSplitV = true;
    opt.useConcatV2 = true;
    auto graph = BuildSplitConvConcatGraph(opt);
    ASSERT_NE(graph, nullptr);
    TestTotalPass("splitv_concatv2_success", graph, SUCCESS);
    EXPECT_FALSE(GraphChecker::HasNode(graph, "SplitV"));
    EXPECT_FALSE(GraphChecker::HasNode(graph, "ConcatV2"));
    EXPECT_EQ(GraphChecker::CountNodes(graph, "Conv2D"), 1);
}

TEST_F(SplitConv2dConcatFusionPassTest, split_conv2d_concat_weight_mismatch_reject)
{
    SplitConvGraphOptions opt;
    opt.mismatchWeight = true;
    auto graph = BuildSplitConvConcatGraph(opt);
    ASSERT_NE(graph, nullptr);
    TestTotalPass("weight_mismatch_reject", graph, CONV_NOT_CHANGED);
    EXPECT_TRUE(GraphChecker::HasNode(graph, "Split"));
}

TEST_F(SplitConv2dConcatFusionPassTest, split_conv2d_concat_format_mismatch_reject)
{
    SplitConvGraphOptions opt;
    opt.mismatchConcatFormat = true;
    auto graph = BuildSplitConvConcatGraph(opt);
    ASSERT_NE(graph, nullptr);
    TestTotalPass("format_mismatch_reject", graph, CONV_NOT_CHANGED);
    EXPECT_TRUE(GraphChecker::HasNode(graph, "Split"));
}

TEST_F(SplitConv2dConcatFusionPassTest, split_conv2d_concat_reentrant_success)
{
    SplitConvGraphOptions opt;
    auto graph1 = BuildSplitConvConcatGraph(opt);
    auto graph2 = BuildSplitConvConcatGraph(opt);
    ASSERT_NE(graph1, nullptr);
    ASSERT_NE(graph2, nullptr);
    TestTotalPass("reentrant_1", graph1, SUCCESS);
    TestTotalPass("reentrant_2", graph2, SUCCESS);
    EXPECT_FALSE(GraphChecker::HasNode(graph1, "Split"));
    EXPECT_FALSE(GraphChecker::HasNode(graph2, "Split"));
}

TEST_F(SplitConv2dConcatFusionPassTest, dtype_float_success)
{
    SplitConvGraphOptions opt;
    opt.dtype = DT_FLOAT;
    auto graph = BuildSplitConvConcatGraph(opt);
    ASSERT_NE(graph, nullptr);
    TestTotalPass("dtype_float_success", graph, SUCCESS);
    EXPECT_FALSE(GraphChecker::HasNode(graph, "Split"));
    GNode groupConv;
    ASSERT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "Conv2D", groupConv));
    TensorDesc filterDesc;
    ASSERT_EQ(groupConv.GetInputDesc(1, filterDesc), GRAPH_SUCCESS);
    EXPECT_EQ(filterDesc.GetDataType(), DT_FLOAT);
}

TEST_F(SplitConv2dConcatFusionPassTest, dtype_int8_success)
{
    SplitConvGraphOptions opt;
    opt.dtype = DT_INT8;
    opt.withBias = true;
    opt.biasDtype = DT_INT32;
    auto graph = BuildSplitConvConcatGraph(opt);
    ASSERT_NE(graph, nullptr);
    TestTotalPass("dtype_int8_success", graph, SUCCESS);
    EXPECT_FALSE(GraphChecker::HasNode(graph, "Split"));
}

TEST_F(SplitConv2dConcatFusionPassTest, same_graph_two_structures)
{
    auto graph = BuildSameGraphTwoStructures();
    ASSERT_NE(graph, nullptr);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "Split"), 2);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "Conv2D"), 4);
    TestTotalPass("same_graph_two_structures", graph, SUCCESS);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "Split"), 0);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "Conv2D"), 2);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "Concatv2HostCpuOp"), 2);
}

TEST_F(SplitConv2dConcatFusionPassTest, split_concatv2_success)
{
    SplitConvGraphOptions opt;
    opt.useConcatV2 = true;
    auto graph = BuildSplitConvConcatGraph(opt);
    ASSERT_NE(graph, nullptr);
    TestTotalPass("split_concatv2_success", graph, SUCCESS);
    EXPECT_FALSE(GraphChecker::HasNode(graph, "Split"));
    EXPECT_FALSE(GraphChecker::HasNode(graph, "ConcatV2"));
    EXPECT_EQ(GraphChecker::CountNodes(graph, "Conv2D"), 1);
}

TEST_F(SplitConv2dConcatFusionPassTest, splitv_concat_success)
{
    SplitConvGraphOptions opt;
    opt.useSplitV = true;
    opt.useConcatV2 = false;
    auto graph = BuildSplitConvConcatGraph(opt);
    ASSERT_NE(graph, nullptr);
    TestTotalPass("splitv_concat_success", graph, SUCCESS);
    EXPECT_FALSE(GraphChecker::HasNode(graph, "SplitV"));
    EXPECT_FALSE(GraphChecker::HasNode(graph, "Concat"));
    EXPECT_EQ(GraphChecker::CountNodes(graph, "Conv2D"), 1);
}

TEST_F(SplitConv2dConcatFusionPassTest, host_concat_dim_desc)
{
    SplitConvGraphOptions opt;
    opt.withBias = true;
    auto graph = BuildSplitConvConcatGraph(opt);
    ASSERT_NE(graph, nullptr);
    TestTotalPass("host_concat_dim_desc", graph, SUCCESS);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "Concatv2HostCpuOp"), 2);
    for (auto node : graph->GetAllNodes()) {
        AscendString type;
        ASSERT_EQ(node.GetType(type), GRAPH_SUCCESS);
        if (std::string(type.GetString()) != "Concatv2HostCpuOp") {
            continue;
        }
        int32_t dimIdx = static_cast<int32_t>(node.GetInputsSize() - 1);
        TensorDesc dimDesc;
        ASSERT_EQ(node.GetInputDesc(dimIdx, dimDesc), GRAPH_SUCCESS);
        EXPECT_EQ(dimDesc.GetShape().GetDims(), (std::vector<int64_t>{1}));
        EXPECT_EQ(dimDesc.GetOriginShape().GetDims(), (std::vector<int64_t>{1}));
        EXPECT_EQ(dimDesc.GetFormat(), FORMAT_ND);
        EXPECT_EQ(dimDesc.GetOriginFormat(), FORMAT_ND);
        EXPECT_EQ(dimDesc.GetDataType(), DT_INT32);
    }
}

TEST_F(SplitConv2dConcatFusionPassTest, group_conv_attrs_copied)
{
    SplitConvGraphOptions opt;
    auto graph = BuildSplitConvConcatGraph(opt);
    ASSERT_NE(graph, nullptr);
    TestTotalPass("group_conv_attrs_copied", graph, SUCCESS);
    GNode groupConv;
    ASSERT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "Conv2D", groupConv));
    std::vector<int64_t> strides;
    std::vector<int64_t> pads;
    std::vector<int64_t> dilations;
    AscendString dataFormat;
    int64_t offsetX = -1;
    EXPECT_EQ(groupConv.GetAttr(AscendString("strides"), strides), GRAPH_SUCCESS);
    EXPECT_EQ(groupConv.GetAttr(AscendString("pads"), pads), GRAPH_SUCCESS);
    EXPECT_EQ(groupConv.GetAttr(AscendString("dilations"), dilations), GRAPH_SUCCESS);
    EXPECT_EQ(groupConv.GetAttr(AscendString("data_format"), dataFormat), GRAPH_SUCCESS);
    EXPECT_EQ(groupConv.GetAttr(AscendString("offset_x"), offsetX), GRAPH_SUCCESS);
    EXPECT_EQ(strides, (std::vector<int64_t>{1, 1, 1, 1}));
    EXPECT_EQ(pads, (std::vector<int64_t>{0, 0, 0, 0}));
    EXPECT_EQ(dilations, (std::vector<int64_t>{1, 1, 1, 1}));
    EXPECT_STREQ(dataFormat.GetString(), "NHWC");
    EXPECT_EQ(offsetX, 0);
}

TEST_F(SplitConv2dConcatFusionPassTest, filter_bias_n_expand)
{
    SplitConvGraphOptions opt;
    opt.withBias = true;
    opt.numSplit = 2;
    auto graph = BuildSplitConvConcatGraph(opt);
    ASSERT_NE(graph, nullptr);
    TestTotalPass("filter_bias_n_expand", graph, SUCCESS);
    GNode groupConv;
    ASSERT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "Conv2D", groupConv));
    TensorDesc filterDesc;
    TensorDesc biasDesc;
    ASSERT_EQ(groupConv.GetInputDesc(1, filterDesc), GRAPH_SUCCESS);
    ASSERT_EQ(groupConv.GetInputDesc(2, biasDesc), GRAPH_SUCCESS);
    auto filterDims = filterDesc.GetOriginShape().GetDims();
    auto biasDims = biasDesc.GetOriginShape().GetDims();
    ASSERT_EQ(filterDims.size(), 4U);
    ASSERT_EQ(biasDims.size(), 1U);
    EXPECT_EQ(filterDims[3], 32); // HWCN N: 16 * groups
    EXPECT_EQ(biasDims[0], 32);
}

TEST_F(SplitConv2dConcatFusionPassTest, weight_format_unsupported_reject)
{
    SplitConvGraphOptions opt;
    opt.forceWeightFormat = FORMAT_NHWC;
    auto graph = BuildSplitConvConcatGraph(opt);
    ASSERT_NE(graph, nullptr);
    TestTotalPass("weight_format_unsupported_reject", graph, CONV_NOT_CHANGED);
    EXPECT_TRUE(GraphChecker::HasNode(graph, "Split"));
}

TEST_F(SplitConv2dConcatFusionPassTest, weight_format_mismatch_reject)
{
    SplitConvGraphOptions opt;
    opt.mismatchWeightFormat = true;
    auto graph = BuildSplitConvConcatGraph(opt);
    ASSERT_NE(graph, nullptr);
    TestTotalPass("weight_format_mismatch_reject", graph, CONV_NOT_CHANGED);
    EXPECT_TRUE(GraphChecker::HasNode(graph, "Split"));
}

TEST_F(SplitConv2dConcatFusionPassTest, unknown_weight_shape_reject)
{
    SplitConvGraphOptions opt;
    opt.unknownWeightShape = true;
    auto graph = BuildSplitConvConcatGraph(opt);
    ASSERT_NE(graph, nullptr);
    TestTotalPass("unknown_weight_shape_reject", graph, CONV_NOT_CHANGED);
    EXPECT_TRUE(GraphChecker::HasNode(graph, "Split"));
}

TEST_F(SplitConv2dConcatFusionPassTest, conv_multi_consumer_reject)
{
    SplitConvGraphOptions opt;
    auto graph = BuildSplitConvConcatGraph(opt);
    ASSERT_NE(graph, nullptr);
    GNode conv0;
    ASSERT_TRUE(GraphChecker::FindNodeByNameSuffix(graph, "conv2d_0", conv0));
    auto relu = CompliantNodeBuilder(graph.get())
                    .OpType("Relu")
                    .Name("extra_relu")
                    .IrDefInputs({{"x", CompliantNodeBuilder::kEsIrInputRequired, ""}})
                    .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                    .Build();
    ASSERT_EQ(graph->AddDataEdge(conv0, 0, relu, 0), GRAPH_SUCCESS);
    TestTotalPass("conv_multi_consumer_reject", graph, CONV_NOT_CHANGED);
    EXPECT_TRUE(GraphChecker::HasNode(graph, "Split"));
}

TEST_F(SplitConv2dConcatFusionPassTest, conv_to_different_concat_reject)
{
    EsGraphBuilder graphBuilder("diff_concat");
    SplitConvGraphOptions opt;
    auto input = graphBuilder.CreateInput(0, "input", opt.dtype, opt.fmt, opt.inputShape);
    Graph* graph = graphBuilder.GetCGraphBuilder()->GetGraph();
    std::vector<int32_t> axisData = {opt.axis};
    auto splitDim = graphBuilder.CreateConst(axisData, {1}, DT_INT32);
    auto splitNode = CompliantNodeBuilder(graph)
                         .OpType("Split")
                         .Name("split")
                         .IrDefInputs({{"split_dim", CompliantNodeBuilder::kEsIrInputRequired, ""},
                                       {"x", CompliantNodeBuilder::kEsIrInputRequired, ""}})
                         .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputDynamic, ""}})
                         .InstanceDynamicOutputNum("y", 2)
                         .Build();
    int64_t numSplitAttr = 2;
    splitNode.SetAttr(AscendString("num_split"), numSplitAttr);
    graph->AddDataEdge(*splitDim.GetProducer(), splitDim.GetProducerOutIndex(), splitNode, 0);
    graph->AddDataEdge(*input.GetProducer(), input.GetProducerOutIndex(), splitNode, 1);
    std::vector<int64_t> splitOutShape = {1, 7, 7, 16};
    for (int32_t i = 0; i < 2; ++i) {
        TensorDesc outDesc(Shape(splitOutShape), FORMAT_NHWC, DT_FLOAT16);
        outDesc.SetOriginFormat(FORMAT_NHWC);
        splitNode.UpdateOutputDesc(i, outDesc);
    }
    std::vector<GNode> convNodes;
    std::vector<EsTensorHolder> outs;
    for (int32_t i = 0; i < 2; ++i) {
        auto convNode = CreateConv2DNode(graph, "conv2d_" + std::to_string(i), DT_FLOAT16, FORMAT_NHWC, splitOutShape,
                                         opt.filterShape, false);
        graph->AddDataEdge(splitNode, i, convNode, 0);
        auto filter = CreateFilterConst(graph, "filter_" + std::to_string(i), DT_FLOAT16, FORMAT_HWCN, opt.filterShape);
        graph->AddDataEdge(filter, 0, convNode, 1);
        auto concatDim = graphBuilder.CreateConst(axisData, {1}, DT_INT32);
        auto concatNode = CompliantNodeBuilder(graph)
                              .OpType("Concat")
                              .Name(("concat_" + std::to_string(i)).c_str())
                              .IrDefInputs({{"concat_dim", CompliantNodeBuilder::kEsIrInputRequired, ""},
                                            {"x", CompliantNodeBuilder::kEsIrInputDynamic, ""}})
                              .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                              .InstanceDynamicInputNum("x", 1)
                              .Build();
        int64_t nAttr = 1;
        concatNode.SetAttr(AscendString("N"), nAttr);
        graph->AddDataEdge(*concatDim.GetProducer(), concatDim.GetProducerOutIndex(), concatNode, 0);
        graph->AddDataEdge(convNode, 0, concatNode, 1);
        TensorDesc concatOutDesc(Shape(splitOutShape), FORMAT_NHWC, DT_FLOAT16);
        concatOutDesc.SetOriginFormat(FORMAT_NHWC);
        concatNode.UpdateOutputDesc(0, concatOutDesc);
        outs.push_back(EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(concatNode, 0)));
        convNodes.push_back(convNode);
    }
    GraphPtr g = graphBuilder.BuildAndReset(outs);
    ASSERT_NE(g, nullptr);
    TestTotalPass("conv_to_different_concat_reject", g, CONV_NOT_CHANGED);
    EXPECT_TRUE(GraphChecker::HasNode(g, "Split"));
}

TEST_F(SplitConv2dConcatFusionPassTest, non_const_weight_reject)
{
    SplitConvGraphOptions opt;
    auto graph = BuildSplitConvConcatGraph(opt);
    ASSERT_NE(graph, nullptr);
    GNode conv0;
    ASSERT_TRUE(GraphChecker::FindNodeByNameSuffix(graph, "conv2d_0", conv0));
    auto filterPair = conv0.GetInDataNodesAndPortIndexs(1);
    ASSERT_NE(filterPair.first, nullptr);
    GNode oldFilter = *filterPair.first;
    auto dataFilter = CompliantNodeBuilder(graph.get())
                          .OpType("Data")
                          .Name("data_filter")
                          .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                          .Build();
    TensorDesc filterDesc;
    ASSERT_EQ(conv0.GetInputDesc(1, filterDesc), GRAPH_SUCCESS);
    dataFilter.UpdateOutputDesc(0, filterDesc);
    ASSERT_EQ(graph->RemoveEdge(oldFilter, 0, conv0, 1), GRAPH_SUCCESS);
    ASSERT_EQ(graph->AddDataEdge(dataFilter, 0, conv0, 1), GRAPH_SUCCESS);
    TestTotalPass("non_const_weight_reject", graph, CONV_NOT_CHANGED);
    EXPECT_TRUE(GraphChecker::HasNode(graph, "Split"));
}

TEST_F(SplitConv2dConcatFusionPassTest, input_count_mismatch_reject)
{
    SplitConvGraphOptions opt;
    opt.withBias = true;
    opt.mismatchInputCount = true;
    auto graph = BuildSplitConvConcatGraph(opt);
    ASSERT_NE(graph, nullptr);
    TestTotalPass("input_count_mismatch_reject", graph, CONV_NOT_CHANGED);
    EXPECT_TRUE(GraphChecker::HasNode(graph, "Split"));
}

TEST_F(SplitConv2dConcatFusionPassTest, structure_first_out_not_conv_reject)
{
    SplitConvGraphOptions opt;
    auto graph = BuildSplitConvConcatGraph(opt);
    ASSERT_NE(graph, nullptr);
    GNode split;
    ASSERT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "Split", split));
    GNode conv0;
    ASSERT_TRUE(GraphChecker::FindNodeByNameSuffix(graph, "conv2d_0", conv0));
    ASSERT_EQ(graph->RemoveEdge(split, 0, conv0, 0), GRAPH_SUCCESS);
    auto relu = CompliantNodeBuilder(graph.get())
                    .OpType("Relu")
                    .Name("mid_relu")
                    .IrDefInputs({{"x", CompliantNodeBuilder::kEsIrInputRequired, ""}})
                    .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                    .Build();
    ASSERT_EQ(graph->AddDataEdge(split, 0, relu, 0), GRAPH_SUCCESS);
    ASSERT_EQ(graph->AddDataEdge(relu, 0, conv0, 0), GRAPH_SUCCESS);
    TestTotalPass("structure_first_out_not_conv_reject", graph, CONV_NOT_CHANGED);
    EXPECT_TRUE(GraphChecker::HasNode(graph, "Split"));
}

TEST_F(SplitConv2dConcatFusionPassTest, structure_conv_not_to_concat_reject)
{
    SplitConvGraphOptions opt;
    auto graph = BuildSplitConvConcatGraph(opt);
    ASSERT_NE(graph, nullptr);
    GNode conv0;
    GNode concat;
    ASSERT_TRUE(GraphChecker::FindNodeByNameSuffix(graph, "conv2d_0", conv0));
    ASSERT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "Concat", concat));
    ASSERT_EQ(graph->RemoveEdge(conv0, 0, concat, 1), GRAPH_SUCCESS);
    auto relu = CompliantNodeBuilder(graph.get())
                    .OpType("Relu")
                    .Name("tail_relu")
                    .IrDefInputs({{"x", CompliantNodeBuilder::kEsIrInputRequired, ""}})
                    .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                    .Build();
    ASSERT_EQ(graph->AddDataEdge(conv0, 0, relu, 0), GRAPH_SUCCESS);
    TestTotalPass("structure_conv_not_to_concat_reject", graph, CONV_NOT_CHANGED);
    EXPECT_TRUE(GraphChecker::HasNode(graph, "Split"));
}

#endif // GE_COMPILER_VERSION_NUM
