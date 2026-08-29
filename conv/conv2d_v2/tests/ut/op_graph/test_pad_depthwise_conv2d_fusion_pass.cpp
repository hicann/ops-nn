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
#include "../../../op_graph/fusion_pass/pad_depthwise_conv2d_fusion_pass.h"

#include "version/ge-compiler_version.h"
#if GE_COMPILER_VERSION_NUM >= 90100000U

using namespace ge;
using namespace es;
using namespace fe;
using namespace Ops;
using namespace NN;
using namespace Conv;
using namespace ConvFusionUtils;
using namespace PadDepthwiseConv2dFusion;
using namespace test_conv_fusion_framework;

#define CONV_DEBUG false

struct PadDwGraphOptions {
    DataType dtype = DT_FLOAT;
    Format format = FORMAT_NCHW;
    std::vector<int64_t> inputShape = {1, 8, 32, 32};
    std::vector<int64_t> filterShape = {3, 3, 8, 8};
    std::vector<int64_t> padValues = {0, 0, 0, 0, 1, 1, 1, 1};
    std::string paddingMode = "VALID";
    bool setPaddingAttr = true;
    bool addExtraConsumer = false;
    bool addMultiDwConsumer = false;
    bool addControlEdge = false;
    bool padFormatInvalid = false;
};

class PadDepthwiseConv2dFusionPassTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "PadDepthwiseConv2dFusionPassTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "PadDepthwiseConv2dFusionPassTest TearDown" << std::endl; }

    void SetSocMC62() { SocConfig::MC62().Apply(); }

    void SetSocAscend950() { SocConfig::Ascend950().Apply(); }

    void SetSocNonAscend950() { SocConfig("Ascend910B", "Ascend910B1").Apply(); }

    void TestTotalPass(const std::string& passName, GraphPtr& graph, Status expectRes)
    {
        if (CONV_DEBUG) {
            graph->DumpToFile(Graph::DumpFormat::kOnnx, AscendString((passName + "_before").c_str()));
        }
        CustomPassContext passContext;
        passContext.SetPassName(FUSION_NAME.c_str());
        PadDepthwiseConv2dFusionPass pass;
        auto res = pass.Run(graph, passContext);
        if (CONV_DEBUG) {
            graph->DumpToFile(Graph::DumpFormat::kOnnx, AscendString((passName + "_after").c_str()));
        }
        EXPECT_EQ(res, expectRes);
    }

    GNode CreateConstNode(Graph* graph, const std::string& name, DataType dtype, Format fmt,
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
        size_t elemSize = 4;
        if (dtype == DT_FLOAT16) {
            elemSize = 2;
        } else if (dtype == DT_INT8) {
            elemSize = 1;
        }
        std::vector<uint8_t> data(elemCount * elemSize, 0);
        Tensor tensor(TensorDesc(Shape(shape), fmt, dtype));
        tensor.SetData(data.data(), data.size());
        node.SetAttr(AscendString("value"), tensor);
        TensorDesc outDesc(Shape(shape), fmt, dtype);
        outDesc.SetOriginFormat(fmt);
        outDesc.SetOriginShape(Shape(shape));
        node.UpdateOutputDesc(0, outDesc);
        return node;
    }

    GNode CreatePaddingsConst(Graph* graph, const std::string& name, const std::vector<int64_t>& padValues)
    {
        auto node = CreateConstNode(graph, name, DT_INT32, FORMAT_ND, {4, 2});
        std::vector<uint8_t> bytes(padValues.size() * sizeof(int32_t), 0);
        for (size_t i = 0; i < padValues.size(); ++i) {
            int32_t v = static_cast<int32_t>(padValues[i]);
            bytes[i * 4] = static_cast<uint8_t>(v & 0xFF);
            bytes[i * 4 + 1] = static_cast<uint8_t>((v >> 8) & 0xFF);
            bytes[i * 4 + 2] = static_cast<uint8_t>((v >> 16) & 0xFF);
            bytes[i * 4 + 3] = static_cast<uint8_t>((v >> 24) & 0xFF);
        }
        Tensor tensor(TensorDesc(Shape({4, 2}), FORMAT_ND, DT_INT32));
        tensor.SetData(bytes.data(), bytes.size());
        node.SetAttr(AscendString("value"), tensor);
        return node;
    }

    GNode CreatePadNode(Graph* graph, Format format, const std::vector<int64_t>& inputShape,
                        const std::vector<int64_t>& outShape, bool padFormatInvalid)
    {
        Format padFormat = padFormatInvalid ? FORMAT_ND : format;
        auto pad = CompliantNodeBuilder(graph)
                       .OpType("Pad")
                       .Name("Pad")
                       .IrDefInputs({{"x", CompliantNodeBuilder::kEsIrInputRequired, ""},
                                     {"paddings", CompliantNodeBuilder::kEsIrInputRequired, ""}})
                       .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                       .Build();
        TensorDesc xDesc(Shape(inputShape), padFormat, DT_FLOAT);
        xDesc.SetOriginFormat(padFormat);
        xDesc.SetOriginShape(Shape(inputShape));
        pad.UpdateInputDesc(0, xDesc);
        TensorDesc pDesc(Shape({4, 2}), FORMAT_ND, DT_INT32);
        pDesc.SetOriginFormat(FORMAT_ND);
        pDesc.SetOriginShape(Shape({4, 2}));
        pad.UpdateInputDesc(1, pDesc);
        TensorDesc yDesc(Shape(outShape), format, DT_FLOAT);
        yDesc.SetOriginFormat(format);
        yDesc.SetOriginShape(Shape(outShape));
        pad.UpdateOutputDesc(0, yDesc);
        return pad;
    }

    GNode CreateDwNode(Graph* graph, const std::string& name, DataType dtype, Format format,
                       const std::vector<int64_t>& inShape, const std::vector<int64_t>& filterShape,
                       const std::vector<int64_t>& outShape, const std::string& paddingMode, bool setPaddingAttr)
    {
        Format filterFmt = (format == FORMAT_NHWC) ? FORMAT_HWCN : FORMAT_NCHW;
        auto dw = CompliantNodeBuilder(graph)
                      .OpType("DepthwiseConv2D")
                      .Name(name.c_str())
                      .IrDefInputs({{"x", CompliantNodeBuilder::kEsIrInputRequired, ""},
                                    {"filter", CompliantNodeBuilder::kEsIrInputRequired, ""}})
                      .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                      .Build();
        std::vector<int64_t> padsList = {0, 0, 0, 0};
        std::vector<int64_t> stridesList = {1, 1, 1, 1};
        std::vector<int64_t> dilationsList = {1, 1, 1, 1};
        int64_t groups = 1;
        int64_t offsetX = 0;
        AscendString dataFormat = AscendString((format == FORMAT_NHWC) ? "NHWC" : "NCHW");
        dw.SetAttr(PADS, padsList);
        dw.SetAttr(STRIDES, stridesList);
        dw.SetAttr(DILATIONS, dilationsList);
        dw.SetAttr(GROUPS, groups);
        dw.SetAttr(OFFSET_X, offsetX);
        dw.SetAttr(DATA_FORMAT, dataFormat);
        if (setPaddingAttr) {
            AscendString padding = AscendString(paddingMode.c_str());
            dw.SetAttr(PADDING, padding);
        }
        TensorDesc inDesc(Shape(inShape), format, dtype);
        inDesc.SetOriginFormat(format);
        inDesc.SetOriginShape(Shape(inShape));
        dw.UpdateInputDesc(0, inDesc);
        TensorDesc fDesc(Shape(filterShape), filterFmt, dtype);
        fDesc.SetOriginFormat(filterFmt);
        fDesc.SetOriginShape(Shape(filterShape));
        dw.UpdateInputDesc(1, fDesc);
        TensorDesc outDesc(Shape(outShape), format, dtype);
        outDesc.SetOriginFormat(format);
        outDesc.SetOriginShape(Shape(outShape));
        dw.UpdateOutputDesc(0, outDesc);
        return dw;
    }

    std::vector<int64_t> ComputeOutShape(Format format, const std::vector<int64_t>& inputShape,
                                         const std::vector<int64_t>& padValues)
    {
        std::vector<int64_t> outShape = inputShape;
        if (format == FORMAT_NCHW) {
            outShape[2] += padValues[4] + padValues[5];
            outShape[3] += padValues[6] + padValues[7];
        } else {
            outShape[1] += padValues[2] + padValues[3];
            outShape[2] += padValues[4] + padValues[5];
        }
        return outShape;
    }

    GNode AddExtraConsumer(Graph* graph, GNode& pad)
    {
        GNode relu = CompliantNodeBuilder(graph)
                         .OpType("Relu")
                         .Name("relu")
                         .IrDefInputs({{"x", CompliantNodeBuilder::kEsIrInputRequired, ""}})
                         .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                         .Build();
        graph->AddDataEdge(pad, 0, relu, 0);
        return relu;
    }

    GraphPtr BuildPadDwGraph(const std::string& tag, const PadDwGraphOptions& opt)
    {
        EsGraphBuilder graphBuilder(tag.c_str());
        auto x = graphBuilder.CreateInput(0, "x", opt.dtype, opt.format, opt.inputShape);
        Graph* graph = graphBuilder.GetCGraphBuilder()->GetGraph();

        auto paddingsConst = CreatePaddingsConst(graph, "paddings_const", opt.padValues);
        std::vector<int64_t> outShape = ComputeOutShape(opt.format, opt.inputShape, opt.padValues);
        GNode pad = CreatePadNode(graph, opt.format, opt.inputShape, outShape, opt.padFormatInvalid);

        GNode dw = CreateDwNode(graph, "dwc", opt.dtype, opt.format, outShape, opt.filterShape, outShape,
                                opt.paddingMode, opt.setPaddingAttr);
        auto filter = CreateConstNode(graph, "filter", opt.dtype,
                                      (opt.format == FORMAT_NHWC) ? FORMAT_HWCN : FORMAT_NCHW, opt.filterShape);

        graph->AddDataEdge(*x.GetProducer(), x.GetProducerOutIndex(), pad, 0);
        graph->AddDataEdge(paddingsConst, 0, pad, 1);
        graph->AddDataEdge(pad, 0, dw, 0);
        graph->AddDataEdge(filter, 0, dw, 1);

        if (opt.addExtraConsumer) {
            AddExtraConsumer(graph, pad);
        }
        if (opt.addMultiDwConsumer) {
            GNode dw2 = CreateDwNode(graph, "dwc2", opt.dtype, opt.format, outShape, opt.filterShape, outShape,
                                     opt.paddingMode, opt.setPaddingAttr);
            graph->AddDataEdge(pad, 0, dw2, 0);
            graph->AddDataEdge(filter, 0, dw2, 1);
        }

        auto dwHolder = EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(dw, 0));
        GraphPtr geGraph = graphBuilder.BuildAndReset({dwHolder});
        if (opt.addControlEdge) {
            GNode padNode;
            GNode dwNode;
            GraphChecker::FindFirstNodeByOpType(geGraph, "Pad", padNode);
            GraphChecker::FindFirstNodeByOpType(geGraph, "DepthwiseConv2D", dwNode);
            geGraph->AddControlEdge(padNode, dwNode);
        }
        return geGraph;
    }

    bool VerifyProducerType(GraphPtr& graph, const std::string& nodeType, int32_t inputIdx,
                            const std::string& expectProducerType)
    {
        GNode node;
        if (!GraphChecker::FindFirstNodeByOpType(graph, nodeType, node)) {
            return false;
        }
        std::string producerType;
        return GraphChecker::GetInputProducerType(node, inputIdx, producerType) && producerType == expectProducerType;
    }

    bool VerifyPadsAttr(GraphPtr& graph, const std::vector<int64_t>& expectPads)
    {
        GNode dw;
        if (!GraphChecker::FindFirstNodeByOpType(graph, "DepthwiseConv2D", dw)) {
            return false;
        }
        std::vector<int64_t> pads;
        return GraphChecker::GetListIntAttr(dw, "pads", pads) && pads == expectPads;
    }

    bool VerifyPaddingAttr(GraphPtr& graph, const std::string& expectPadding)
    {
        GNode dw;
        if (!GraphChecker::FindFirstNodeByOpType(graph, "DepthwiseConv2D", dw)) {
            return false;
        }
        std::string padding;
        return GraphChecker::GetNodeStringAttr(dw, "padding", padding) && padding == expectPadding;
    }
};

// ==========================================================================================
// Success: NCHW / NHWC
// ==========================================================================================
TEST_F(PadDepthwiseConv2dFusionPassTest, fusion_success_nchw)
{
    SetSocMC62();
    PadDwGraphOptions opt;
    opt.format = FORMAT_NCHW;
    opt.padValues = {0, 0, 0, 0, 1, 1, 1, 1};
    auto graph = BuildPadDwGraph("fusion_success_nchw", opt);
    TestTotalPass("fusion_success_nchw", graph, SUCCESS);

    EXPECT_FALSE(GraphChecker::HasNode(graph, "Pad"));
    EXPECT_TRUE(VerifyProducerType(graph, "DepthwiseConv2D", 0, "Data"));
    EXPECT_TRUE(VerifyPadsAttr(graph, {1, 1, 1, 1}));
    EXPECT_TRUE(VerifyPaddingAttr(graph, "SAME"));
}

TEST_F(PadDepthwiseConv2dFusionPassTest, fusion_success_nhwc)
{
    SetSocMC62();
    PadDwGraphOptions opt;
    opt.format = FORMAT_NHWC;
    opt.padValues = {0, 0, 1, 1, 1, 1, 0, 0};
    auto graph = BuildPadDwGraph("fusion_success_nhwc", opt);
    TestTotalPass("fusion_success_nhwc", graph, SUCCESS);

    EXPECT_FALSE(GraphChecker::HasNode(graph, "Pad"));
    EXPECT_TRUE(VerifyProducerType(graph, "DepthwiseConv2D", 0, "Data"));
    EXPECT_TRUE(VerifyPadsAttr(graph, {1, 1, 1, 1}));
    EXPECT_TRUE(VerifyPaddingAttr(graph, "SAME"));
}

// ==========================================================================================
// No fusion: various reject cases on non-Ascend950 soc
// ==========================================================================================
TEST_F(PadDepthwiseConv2dFusionPassTest, no_fusion)
{
    SetSocMC62();
    struct Point {
        const char* pointName;
        PadDwGraphOptions opt;
    };
    std::vector<Point> points;

    PadDwGraphOptions o1;
    o1.paddingMode = "SAME";
    points.push_back({"padding_not_valid", o1});

    PadDwGraphOptions o2;
    o2.setPaddingAttr = false;
    points.push_back({"missing_padding_attr", o2});

    PadDwGraphOptions o3;
    o3.padValues = {0, 0, 0, 0, 1, 1, 256, 1};
    points.push_back({"pad_range_out", o3});

    PadDwGraphOptions o4;
    o4.padFormatInvalid = true;
    points.push_back({"pad_format_invalid", o4});

    PadDwGraphOptions o5;
    o5.addExtraConsumer = true;
    points.push_back({"extra_consumer", o5});

    PadDwGraphOptions o6;
    o6.addMultiDwConsumer = true;
    points.push_back({"multi_dw_consumer", o6});

    PadDwGraphOptions o7;
    o7.addControlEdge = true;
    points.push_back({"control_edge", o7});

    for (const auto& p : points) {
        SCOPED_TRACE(p.pointName);
        auto graph = BuildPadDwGraph(std::string("no_fusion_") + p.pointName, p.opt);
        TestTotalPass(std::string("no_fusion_") + p.pointName, graph, CONV_NOT_CHANGED);
        EXPECT_TRUE(GraphChecker::HasNode(graph, "Pad"));
    }
}

// ==========================================================================================
// filter <= pad: rejected on non-Ascend950, fused on Ascend950
// ==========================================================================================
TEST_F(PadDepthwiseConv2dFusionPassTest, filter_le_pad_reject_non_ascend950)
{
    SetSocNonAscend950();
    PadDwGraphOptions opt;
    // filter(NCHW) H = dim[2] = 1，paddingsT=1：filter H 真实 <= pad，非 950 平台应拒绝
    opt.inputShape = {1, 8, 32, 32};
    opt.filterShape = {3, 3, 1, 8};
    opt.padValues = {0, 0, 0, 0, 1, 0, 1, 1};
    auto graph = BuildPadDwGraph("filter_le_pad_reject", opt);
    TestTotalPass("filter_le_pad_reject", graph, CONV_NOT_CHANGED);
    EXPECT_TRUE(GraphChecker::HasNode(graph, "Pad"));
}

TEST_F(PadDepthwiseConv2dFusionPassTest, filter_le_pad_fuse_on_ascend950)
{
    SetSocAscend950();
    PadDwGraphOptions opt;
    // 与 reject 用例同构图；Ascend950 跳过 CheckFilterVsPadding，因此 filter H <= pad 仍可融合
    opt.inputShape = {1, 8, 32, 32};
    opt.filterShape = {3, 3, 1, 8};
    opt.padValues = {0, 0, 0, 0, 1, 0, 1, 1};
    auto graph = BuildPadDwGraph("filter_le_pad_fuse_910", opt);
    TestTotalPass("filter_le_pad_fuse_910", graph, SUCCESS);
    EXPECT_FALSE(GraphChecker::HasNode(graph, "Pad"));
    EXPECT_TRUE(VerifyPadsAttr(graph, {1, 0, 1, 1}));
}

// =================================./install/autoconfig.shcd=========================================================
// Reentrant: two independent chains
// ==========================================================================================
TEST_F(PadDepthwiseConv2dFusionPassTest, reentrant_two_chains)
{
    SetSocMC62();
    EsGraphBuilder builder("reentrant_two_chains");
    auto x1 = builder.CreateInput(0, "x1", DT_FLOAT, FORMAT_NCHW, {1, 8, 32, 32});
    auto x2 = builder.CreateInput(1, "x2", DT_FLOAT, FORMAT_NCHW, {1, 8, 32, 32});
    Graph* graph = builder.GetCGraphBuilder()->GetGraph();

    auto p1 = CreatePaddingsConst(graph, "paddings1", {0, 0, 0, 0, 1, 1, 1, 1});
    auto p2 = CreatePaddingsConst(graph, "paddings2", {0, 0, 0, 0, 1, 1, 1, 1});
    GNode pad1 = CreatePadNode(graph, FORMAT_NCHW, {1, 8, 32, 32}, {1, 8, 34, 34}, false);
    GNode pad2 = CreatePadNode(graph, FORMAT_NCHW, {1, 8, 32, 32}, {1, 8, 34, 34}, false);
    GNode dw1 = CreateDwNode(graph, "dwc1", DT_FLOAT, FORMAT_NCHW, {1, 8, 34, 34}, {3, 3, 8, 8}, {1, 8, 34, 34},
                             "VALID", true);
    GNode dw2 = CreateDwNode(graph, "dwc2", DT_FLOAT, FORMAT_NCHW, {1, 8, 34, 34}, {3, 3, 8, 8}, {1, 8, 34, 34},
                             "VALID", true);
    auto f1 = CreateConstNode(graph, "filter1", DT_FLOAT, FORMAT_NCHW, {3, 3, 8, 8});
    auto f2 = CreateConstNode(graph, "filter2", DT_FLOAT, FORMAT_NCHW, {3, 3, 8, 8});

    graph->AddDataEdge(*x1.GetProducer(), x1.GetProducerOutIndex(), pad1, 0);
    graph->AddDataEdge(p1, 0, pad1, 1);
    graph->AddDataEdge(pad1, 0, dw1, 0);
    graph->AddDataEdge(f1, 0, dw1, 1);
    graph->AddDataEdge(*x2.GetProducer(), x2.GetProducerOutIndex(), pad2, 0);
    graph->AddDataEdge(p2, 0, pad2, 1);
    graph->AddDataEdge(pad2, 0, dw2, 0);
    graph->AddDataEdge(f2, 0, dw2, 1);

    auto holder1 = EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(dw1, 0));
    auto holder2 = EsTensorHolder(builder.GetCGraphBuilder()->GetTensorHolderFromNode(dw2, 0));
    GraphPtr geGraph = builder.BuildAndReset({holder1, holder2});

    TestTotalPass("reentrant_two_chains", geGraph, SUCCESS);
    EXPECT_EQ(GraphChecker::CountNodes(geGraph, "Pad"), 0);
    EXPECT_EQ(GraphChecker::CountNodes(geGraph, "DepthwiseConv2D"), 2);
    EXPECT_TRUE(VerifyPadsAttr(geGraph, {1, 1, 1, 1}));
}

#endif
