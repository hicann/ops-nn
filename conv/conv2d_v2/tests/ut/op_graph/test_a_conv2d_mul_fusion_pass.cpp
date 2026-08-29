/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <vector>

#include "../../../../common/tests/ut/op_graph/test_conv_fusion_pass_framework.h"
#include "../../../op_graph/fusion_pass/a_conv2d_mul_fusion_pass.h"

#include "version/ge-compiler_version.h"
#if GE_COMPILER_VERSION_NUM >= 90000000U

using namespace ge;
using namespace es;
using namespace fe;
using namespace Ops;
using namespace Ops::NN::Conv;
using namespace ConvFusionUtils;
using namespace AConv2dMulFusionConsts;
using namespace test_conv_fusion_framework;

#define CONV_DEBUG false

namespace {
constexpr int64_t CONV2D_C = 64;
constexpr int64_t CONV3D_C = 64;
constexpr size_t FLOAT16_BYTES = 2;
constexpr size_t FLOAT32_BYTES = 4;
const std::vector<int64_t> CONV2D_FMAP_NHWC = {1, 28, 28, CONV2D_C};
const std::vector<int64_t> CONV2D_FMAP_NCHW = {1, CONV2D_C, 28, 28};
const std::vector<int64_t> CONV2D_FILTER_HWCN = {1, 1, CONV2D_C, CONV2D_C};
const std::vector<int64_t> CONV2D_FILTER_NCHW = {CONV2D_C, CONV2D_C, 1, 1};
const std::vector<int64_t> CONV3D_FMAP_NDHWC = {1, 16, 28, 28, CONV3D_C};
const std::vector<int64_t> CONV3D_FILTER_DHWCN = {1, 1, 1, CONV3D_C, CONV3D_C};

struct MulFusionOptions {
    bool isConv3D = false;
    bool isNCHW = false;
    DataType dtype = DT_FLOAT;
    bool withBias = true;
    std::vector<int64_t> scaleShape = {1};
    bool scaleIsConst = true;
    bool filterIsConst = true;
    bool biasIsConst = true;
    bool filterNotConnected = false;
    bool biasNotConnected = false;
    bool overrideConvOutput = false;
    std::vector<int64_t> convOutputShape;
    Format convOutputFormat = FORMAT_ND;
    bool convMultiConsumer = false;
    bool mulMultiConsumer = false;
    bool mulSingleInput = false;
    bool reversedMulInputs = false;
    bool isDav3510 = false;
};
} // namespace

class AConv2dMulFusionTest : public testing::Test {
protected:
    static void SetUpTestCase() {}
    static void TearDownTestCase() {}

    GNode CreateConst(Graph* graph, const std::string& name, DataType dtype, Format fmt,
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
        size_t bytes = (dtype == DT_FLOAT16) ? FLOAT16_BYTES : FLOAT32_BYTES;
        std::vector<uint8_t> data(elemCount * bytes, 0);
        Tensor tensor(TensorDesc(Shape(shape), fmt, dtype));
        tensor.SetData(data.data(), data.size());
        node.SetAttr(AscendString("value"), tensor);
        TensorDesc outDesc(Shape(shape), fmt, dtype);
        outDesc.SetOriginFormat(fmt);
        outDesc.SetOriginShape(Shape(shape));
        node.UpdateOutputDesc(0, outDesc);
        return node;
    }

    GNode CreateConv2D(Graph* graph, const std::string& name, DataType dtype, Format fmt, bool withBias)
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
        const auto& fmapShape = (fmt == FORMAT_NHWC) ? CONV2D_FMAP_NHWC : CONV2D_FMAP_NCHW;
        const auto& filterShape = (fmt == FORMAT_NHWC) ? CONV2D_FILTER_HWCN : CONV2D_FILTER_NCHW;
        TensorDesc fmapDesc(Shape(fmapShape), fmt, dtype);
        fmapDesc.SetOriginFormat(fmt);
        fmapDesc.SetOriginShape(Shape(fmapShape));
        node.UpdateInputDesc(0, fmapDesc);
        TensorDesc filterDesc(Shape(filterShape), filterFmt, dtype);
        filterDesc.SetOriginFormat(filterFmt);
        filterDesc.SetOriginShape(Shape(filterShape));
        node.UpdateInputDesc(1, filterDesc);
        if (withBias) {
            TensorDesc biasDesc(Shape({CONV2D_C}), FORMAT_ND, dtype);
            biasDesc.SetOriginFormat(FORMAT_ND);
            biasDesc.SetOriginShape(Shape({CONV2D_C}));
            node.UpdateInputDesc(2, biasDesc);
        }
        TensorDesc outDesc(Shape(fmapShape), fmt, dtype);
        outDesc.SetOriginFormat(fmt);
        outDesc.SetOriginShape(Shape(fmapShape));
        node.UpdateOutputDesc(0, outDesc);
        return node;
    }

    GNode CreateConv3D(Graph* graph, const std::string& name, DataType dtype, bool withBias)
    {
        auto node = CompliantNodeBuilder(graph)
                        .OpType("Conv3D")
                        .Name(name.c_str())
                        .IrDefInputs({{"x", CompliantNodeBuilder::kEsIrInputRequired, ""},
                                      {"filter", CompliantNodeBuilder::kEsIrInputRequired, ""},
                                      {"bias", CompliantNodeBuilder::kEsIrInputOptional, ""},
                                      {"offset_w", CompliantNodeBuilder::kEsIrInputOptional, ""}})
                        .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                        .Build();
        std::vector<int64_t> strides = {1, 1, 1, 1, 1};
        std::vector<int64_t> pads = {0, 0, 0, 0, 0, 0};
        std::vector<int64_t> dilations = {1, 1, 1, 1, 1};
        int64_t groups = 1;
        int64_t offsetX = 0;
        AscendString dataFormat("NDHWC");
        node.SetAttr(AscendString("strides"), strides);
        node.SetAttr(AscendString("pads"), pads);
        node.SetAttr(AscendString("dilations"), dilations);
        node.SetAttr(AscendString("groups"), groups);
        node.SetAttr(AscendString("data_format"), dataFormat);
        node.SetAttr(AscendString("offset_x"), offsetX);

        Format filterFmt = FORMAT_DHWCN;
        TensorDesc fmapDesc(Shape(CONV3D_FMAP_NDHWC), FORMAT_NDHWC, dtype);
        fmapDesc.SetOriginFormat(FORMAT_NDHWC);
        fmapDesc.SetOriginShape(Shape(CONV3D_FMAP_NDHWC));
        node.UpdateInputDesc(0, fmapDesc);
        TensorDesc filterDesc(Shape(CONV3D_FILTER_DHWCN), filterFmt, dtype);
        filterDesc.SetOriginFormat(filterFmt);
        filterDesc.SetOriginShape(Shape(CONV3D_FILTER_DHWCN));
        node.UpdateInputDesc(1, filterDesc);
        if (withBias) {
            TensorDesc biasDesc(Shape({CONV3D_C}), FORMAT_ND, dtype);
            biasDesc.SetOriginFormat(FORMAT_ND);
            biasDesc.SetOriginShape(Shape({CONV3D_C}));
            node.UpdateInputDesc(2, biasDesc);
        }
        TensorDesc outDesc(Shape(CONV3D_FMAP_NDHWC), FORMAT_NDHWC, dtype);
        outDesc.SetOriginFormat(FORMAT_NDHWC);
        outDesc.SetOriginShape(Shape(CONV3D_FMAP_NDHWC));
        node.UpdateOutputDesc(0, outDesc);
        return node;
    }

    GNode CreateMul(Graph* graph, const std::string& name)
    {
        return CompliantNodeBuilder(graph)
            .OpType("Mul")
            .Name(name.c_str())
            .IrDefInputs({{"x1", CompliantNodeBuilder::kEsIrInputRequired, ""},
                          {"x2", CompliantNodeBuilder::kEsIrInputRequired, ""}})
            .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
            .Build();
    }

    GNode CreateRelu(Graph* graph, const std::string& name, DataType dtype, Format fmt,
                     const std::vector<int64_t>& shape)
    {
        auto node = CompliantNodeBuilder(graph)
                        .OpType("Relu")
                        .Name(name.c_str())
                        .IrDefInputs({{"x", CompliantNodeBuilder::kEsIrInputRequired, ""}})
                        .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                        .Build();
        TensorDesc inDesc(Shape(shape), fmt, dtype);
        inDesc.SetOriginFormat(fmt);
        inDesc.SetOriginShape(Shape(shape));
        node.UpdateInputDesc(0, inDesc);
        node.UpdateOutputDesc(0, inDesc);
        return node;
    }

    GraphPtr BuildMulFusionGraph(const std::string& tag, const MulFusionOptions& opt)
    {
        if (opt.isDav3510) {
            SocConfig::Ascend950().Apply();
        }
        EsGraphBuilder graphBuilder(tag.c_str());
        DataType dtype = opt.dtype;
        Format fmt = opt.isConv3D ? FORMAT_NDHWC : (opt.isNCHW ? FORMAT_NCHW : FORMAT_NHWC);
        const auto& fmapShape = opt.isConv3D ? CONV3D_FMAP_NDHWC : (opt.isNCHW ? CONV2D_FMAP_NCHW : CONV2D_FMAP_NHWC);

        int32_t inputIdx = 0;
        auto fmap = graphBuilder.CreateInput(inputIdx++, "fmap", dtype, fmt, fmapShape);
        Graph* graph = graphBuilder.GetCGraphBuilder()->GetGraph();

        GNode conv = opt.isConv3D ? CreateConv3D(graph, "conv", dtype, opt.withBias) :
                                    CreateConv2D(graph, "conv", dtype, fmt, opt.withBias);
        graph->AddDataEdge(*fmap.GetProducer(), fmap.GetProducerOutIndex(), conv, 0);
        if (opt.overrideConvOutput) {
            TensorDesc abnormalOut(Shape(opt.convOutputShape), opt.convOutputFormat, dtype);
            abnormalOut.SetOriginFormat(opt.convOutputFormat);
            abnormalOut.SetOriginShape(Shape(opt.convOutputShape));
            conv.UpdateOutputDesc(0, abnormalOut);
        }

        if (!opt.filterNotConnected) {
            if (opt.filterIsConst) {
                Format filterFmt = opt.isConv3D ? FORMAT_DHWCN : ((fmt == FORMAT_NHWC) ? FORMAT_HWCN : FORMAT_NCHW);
                const auto& filterShape = opt.isConv3D ?
                                              CONV3D_FILTER_DHWCN :
                                              ((fmt == FORMAT_NHWC) ? CONV2D_FILTER_HWCN : CONV2D_FILTER_NCHW);
                GNode filter = CreateConst(graph, "filter", dtype, filterFmt, filterShape);
                graph->AddDataEdge(filter, 0, conv, 1);
            } else {
                auto filterInput = graphBuilder.CreateInput(inputIdx++, "filter_data", dtype, FORMAT_ND,
                                                            CONV2D_FILTER_HWCN);
                graph->AddDataEdge(*filterInput.GetProducer(), filterInput.GetProducerOutIndex(), conv, 1);
            }
        }

        if (opt.withBias && !opt.biasNotConnected) {
            if (opt.biasIsConst) {
                GNode bias = CreateConst(graph, "bias", dtype, FORMAT_ND, {CONV2D_C});
                graph->AddDataEdge(bias, 0, conv, 2);
            } else {
                auto biasInput = graphBuilder.CreateInput(inputIdx++, "bias_data", dtype, FORMAT_ND, {CONV2D_C});
                graph->AddDataEdge(*biasInput.GetProducer(), biasInput.GetProducerOutIndex(), conv, 2);
            }
        }

        GNode mul = CreateMul(graph, "mul");
        int32_t mulConvPort = opt.reversedMulInputs ? 1 : 0;
        int32_t mulScalePort = opt.reversedMulInputs ? 0 : 1;
        graph->AddDataEdge(conv, 0, mul, mulConvPort);
        if (!opt.mulSingleInput) {
            if (opt.scaleIsConst) {
                GNode scale = CreateConst(graph, "scale", dtype, FORMAT_ND, opt.scaleShape);
                graph->AddDataEdge(scale, 0, mul, mulScalePort);
            } else {
                auto scaleInput = graphBuilder.CreateInput(inputIdx++, "scale_data", dtype, FORMAT_ND, opt.scaleShape);
                graph->AddDataEdge(*scaleInput.GetProducer(), scaleInput.GetProducerOutIndex(), mul, mulScalePort);
            }
        }

        GNode relu = CreateRelu(graph, "relu", dtype, fmt, fmapShape);
        graph->AddDataEdge(mul, 0, relu, 0);

        std::vector<EsTensorHolder> outputs;
        auto reluHolder = EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(relu, 0));
        outputs.push_back(reluHolder);

        if (opt.convMultiConsumer) {
            GNode extra = CreateRelu(graph, "extra_conv_consumer", dtype, fmt, fmapShape);
            graph->AddDataEdge(conv, 0, extra, 0);
            auto extraHolder = EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(extra, 0));
            outputs.push_back(extraHolder);
        }
        if (opt.mulMultiConsumer) {
            GNode extra = CreateRelu(graph, "extra_mul_consumer", dtype, fmt, fmapShape);
            graph->AddDataEdge(mul, 0, extra, 0);
            auto extraHolder = EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(extra, 0));
            outputs.push_back(extraHolder);
        }

        return graphBuilder.BuildAndReset(outputs);
    }

    void TestTotalPass(const std::string& passName, GraphPtr& graph, Status expectRes)
    {
        if (CONV_DEBUG) {
            graph->DumpToFile(Graph::DumpFormat::kOnnx, AscendString((passName + "_before").c_str()));
        }
        CustomPassContext passContext;
        passContext.SetPassName(FUSION_NAME.c_str());
        AConv2dMulFusion pass;
        auto res = pass.Run(graph, passContext);
        if (CONV_DEBUG) {
            graph->DumpToFile(Graph::DumpFormat::kOnnx, AscendString((passName + "_after").c_str()));
        }
        EXPECT_EQ(res, expectRes);
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
};

// ==========================================================================================
// Success: Conv2D with bias, scale[1] (legacy UT semantics)
// ==========================================================================================
TEST_F(AConv2dMulFusionTest, conv2d_bias_scale1_success)
{
    MulFusionOptions opt;
    opt.isConv3D = false;
    opt.withBias = true;
    opt.scaleShape = {1};
    auto graph = BuildMulFusionGraph("conv2d_bias_scale1", opt);
    TestTotalPass("conv2d_bias_scale1", graph, SUCCESS);

    EXPECT_EQ(GraphChecker::CountNodes(graph, "Conv2D"), 1);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "Mul"), 2);
    EXPECT_TRUE(VerifyProducerType(graph, "Conv2D", 1, "Mul"));
    EXPECT_TRUE(VerifyProducerType(graph, "Conv2D", 2, "Mul"));
    EXPECT_TRUE(VerifyProducerType(graph, "Relu", 0, "Conv2D"));
}

// ==========================================================================================
// Success: Conv2D without bias, scale[1] (scalar form per legacy UT: ge::Shape({1}))
// ==========================================================================================
TEST_F(AConv2dMulFusionTest, conv2d_nobias_scalar_success)
{
    MulFusionOptions opt;
    opt.withBias = false;
    opt.scaleShape = {1};
    auto graph = BuildMulFusionGraph("conv2d_nobias_scalar", opt);
    TestTotalPass("conv2d_nobias_scalar", graph, SUCCESS);

    EXPECT_EQ(GraphChecker::CountNodes(graph, "Conv2D"), 1);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "Mul"), 1);
    EXPECT_TRUE(VerifyProducerType(graph, "Conv2D", 1, "Mul"));
}

// ==========================================================================================
// Success: Conv2D without bias, scale 0D (empty shape, GetDims().size()==0)
// ==========================================================================================
TEST_F(AConv2dMulFusionTest, conv2d_nobias_scale_0d_success)
{
    MulFusionOptions opt;
    opt.withBias = false;
    opt.scaleShape = {};
    auto graph = BuildMulFusionGraph("conv2d_nobias_scale_0d", opt);
    TestTotalPass("conv2d_nobias_scale_0d", graph, SUCCESS);

    EXPECT_EQ(GraphChecker::CountNodes(graph, "Conv2D"), 1);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "Mul"), 1);
    EXPECT_TRUE(VerifyProducerType(graph, "Conv2D", 1, "Mul"));
}

// ==========================================================================================
// Success: Conv2D channel-wise scale[C]
// ==========================================================================================
TEST_F(AConv2dMulFusionTest, conv2d_channelwise_success)
{
    MulFusionOptions opt;
    opt.scaleShape = {CONV2D_C};
    auto graph = BuildMulFusionGraph("conv2d_channelwise", opt);
    TestTotalPass("conv2d_channelwise", graph, SUCCESS);

    EXPECT_EQ(GraphChecker::CountNodes(graph, "Mul"), 2);
    EXPECT_TRUE(VerifyProducerType(graph, "Conv2D", 1, "Mul"));
}

// ==========================================================================================
// Success: Conv3D with bias, scale[C]
// ==========================================================================================
TEST_F(AConv2dMulFusionTest, conv3d_bias_scaleC_success)
{
    MulFusionOptions opt;
    opt.isConv3D = true;
    opt.scaleShape = {CONV3D_C};
    auto graph = BuildMulFusionGraph("conv3d_bias_scaleC", opt);
    TestTotalPass("conv3d_bias_scaleC", graph, SUCCESS);

    EXPECT_EQ(GraphChecker::CountNodes(graph, "Conv3D"), 1);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "Mul"), 2);
    EXPECT_TRUE(VerifyProducerType(graph, "Conv3D", 1, "Mul"));
    EXPECT_TRUE(VerifyProducerType(graph, "Conv3D", 2, "Mul"));
}

// ==========================================================================================
// Success: Conv3D scale[1,1,1,1,C]
// ==========================================================================================
TEST_F(AConv2dMulFusionTest, conv3d_scale5d_success)
{
    MulFusionOptions opt;
    opt.isConv3D = true;
    opt.scaleShape = {1, 1, 1, 1, CONV3D_C};
    auto graph = BuildMulFusionGraph("conv3d_scale5d", opt);
    TestTotalPass("conv3d_scale5d", graph, SUCCESS);

    EXPECT_EQ(GraphChecker::CountNodes(graph, "Mul"), 2);
    EXPECT_TRUE(VerifyProducerType(graph, "Conv3D", 1, "Mul"));
}

// ==========================================================================================
// Success: Conv2D NCHW format with bias, scale[1]
// ==========================================================================================
TEST_F(AConv2dMulFusionTest, conv2d_nchw_bias_scale1_success)
{
    MulFusionOptions opt;
    opt.isNCHW = true;
    opt.withBias = true;
    opt.scaleShape = {1};
    auto graph = BuildMulFusionGraph("conv2d_nchw_bias_scale1", opt);
    TestTotalPass("conv2d_nchw_bias_scale1", graph, SUCCESS);

    EXPECT_EQ(GraphChecker::CountNodes(graph, "Conv2D"), 1);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "Mul"), 2);
    EXPECT_TRUE(VerifyProducerType(graph, "Conv2D", 1, "Mul"));
    EXPECT_TRUE(VerifyProducerType(graph, "Conv2D", 2, "Mul"));
    EXPECT_TRUE(VerifyProducerType(graph, "Relu", 0, "Conv2D"));
}

// ==========================================================================================
// Success: Conv2D NCHW format, channel-wise scale[C]
// ==========================================================================================
// TEST_F(AConv2dMulFusionTest, conv2d_nchw_channelwise_success)
// {
//     MulFusionOptions opt;
//     opt.isNCHW = true;
//     opt.scaleShape = {CONV2D_C};
//     auto graph = BuildMulFusionGraph("conv2d_nchw_channelwise", opt);
//     TestTotalPass("conv2d_nchw_channelwise", graph, SUCCESS);

//     EXPECT_EQ(GraphChecker::CountNodes(graph, "Conv2D"), 1);
//     EXPECT_EQ(GraphChecker::CountNodes(graph, "Mul"), 2);
//     EXPECT_TRUE(VerifyProducerType(graph, "Conv2D", 1, "Mul"));
// }

// ==========================================================================================
// No fusion: conv multi-consumer
// ==========================================================================================
TEST_F(AConv2dMulFusionTest, conv_multi_consumer_reject)
{
    MulFusionOptions opt;
    opt.convMultiConsumer = true;
    auto graph = BuildMulFusionGraph("conv_multi_consumer", opt);
    TestTotalPass("conv_multi_consumer", graph, CONV_NOT_CHANGED);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "Mul"), 1);
}

// ==========================================================================================
// No fusion: scale not Const
// ==========================================================================================
TEST_F(AConv2dMulFusionTest, scale_not_const_reject)
{
    MulFusionOptions opt;
    opt.scaleIsConst = false;
    opt.scaleShape = {1};
    auto graph = BuildMulFusionGraph("scale_not_const", opt);
    TestTotalPass("scale_not_const", graph, CONV_NOT_CHANGED);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "Mul"), 1);
}

// ==========================================================================================
// No fusion: filter not Const
// ==========================================================================================
TEST_F(AConv2dMulFusionTest, filter_not_const_reject)
{
    MulFusionOptions opt;
    opt.filterIsConst = false;
    auto graph = BuildMulFusionGraph("filter_not_const", opt);
    TestTotalPass("filter_not_const", graph, CONV_NOT_CHANGED);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "Mul"), 1);
}

// ==========================================================================================
// No fusion: scale shape mismatch
// ==========================================================================================
TEST_F(AConv2dMulFusionTest, scale_shape_mismatch_reject)
{
    MulFusionOptions opt;
    opt.scaleShape = {2};
    auto graph = BuildMulFusionGraph("scale_shape_mismatch", opt);
    TestTotalPass("scale_shape_mismatch", graph, CONV_NOT_CHANGED);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "Mul"), 1);
}

// ==========================================================================================
// No fusion: Conv3D mul multi-consumer
// ==========================================================================================
TEST_F(AConv2dMulFusionTest, conv3d_mul_multi_consumer_reject)
{
    MulFusionOptions opt;
    opt.isConv3D = true;
    opt.mulMultiConsumer = true;
    auto graph = BuildMulFusionGraph("conv3d_mul_multi_consumer", opt);
    TestTotalPass("conv3d_mul_multi_consumer", graph, CONV_NOT_CHANGED);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "Mul"), 1);
}

// ==========================================================================================
// No fusion: mul single connected input
// ==========================================================================================
TEST_F(AConv2dMulFusionTest, mul_single_input_reject)
{
    MulFusionOptions opt;
    opt.mulSingleInput = true;
    auto graph = BuildMulFusionGraph("mul_single_input", opt);
    TestTotalPass("mul_single_input", graph, CONV_NOT_CHANGED);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "Mul"), 1);
}

// ==========================================================================================
// Reentrant: two independent conv->mul chains, InitMember clears state
// ==========================================================================================
TEST_F(AConv2dMulFusionTest, reentrant_two_chains)
{
    EsGraphBuilder graphBuilder("reentrant_two_chains");
    DataType dtype = DT_FLOAT;
    Graph* geGraph = graphBuilder.GetCGraphBuilder()->GetGraph();

    std::vector<EsTensorHolder> outputs;
    int32_t inputIdx = 0;
    for (int32_t i = 0; i < 2; ++i) {
        std::string suffix = std::to_string(i);
        auto fmap = graphBuilder.CreateInput(inputIdx++, ("fmap" + suffix).c_str(), dtype, FORMAT_NHWC,
                                             CONV2D_FMAP_NHWC);
        GNode conv = CreateConv2D(geGraph, "conv" + suffix, dtype, FORMAT_NHWC, true);
        geGraph->AddDataEdge(*fmap.GetProducer(), fmap.GetProducerOutIndex(), conv, 0);
        GNode filter = CreateConst(geGraph, "filter" + suffix, dtype, FORMAT_HWCN, CONV2D_FILTER_HWCN);
        geGraph->AddDataEdge(filter, 0, conv, 1);
        GNode bias = CreateConst(geGraph, "bias" + suffix, dtype, FORMAT_ND, {CONV2D_C});
        geGraph->AddDataEdge(bias, 0, conv, 2);
        GNode mul = CreateMul(geGraph, "mul" + suffix);
        geGraph->AddDataEdge(conv, 0, mul, 0);
        GNode scale = CreateConst(geGraph, "scale" + suffix, dtype, FORMAT_ND, {1});
        geGraph->AddDataEdge(scale, 0, mul, 1);
        GNode relu = CreateRelu(geGraph, "relu" + suffix, dtype, FORMAT_NHWC, CONV2D_FMAP_NHWC);
        geGraph->AddDataEdge(mul, 0, relu, 0);
        auto holder = EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(relu, 0));
        outputs.push_back(holder);
    }

    GraphPtr graph(graphBuilder.BuildAndReset(outputs));
    TestTotalPass("reentrant_two_chains", graph, SUCCESS);

    EXPECT_EQ(GraphChecker::CountNodes(graph, "Conv2D"), 2);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "Mul"), 4);
}

// ==========================================================================================
// No fusion: Conv2D output goes to Relu directly (no Mul downstream).
// Verifies CheckMatchStructure handles "mul not found" without null deref.
// ==========================================================================================
TEST_F(AConv2dMulFusionTest, conv_no_mul_downstream_reject)
{
    EsGraphBuilder graphBuilder("conv_no_mul_downstream");
    DataType dtype = DT_FLOAT;
    Graph* geGraph = graphBuilder.GetCGraphBuilder()->GetGraph();

    auto fmap = graphBuilder.CreateInput(0, "fmap", dtype, FORMAT_NHWC, CONV2D_FMAP_NHWC);
    GNode conv = CreateConv2D(geGraph, "conv", dtype, FORMAT_NHWC, true);
    geGraph->AddDataEdge(*fmap.GetProducer(), fmap.GetProducerOutIndex(), conv, 0);
    GNode filter = CreateConst(geGraph, "filter", dtype, FORMAT_HWCN, CONV2D_FILTER_HWCN);
    geGraph->AddDataEdge(filter, 0, conv, 1);
    GNode bias = CreateConst(geGraph, "bias", dtype, FORMAT_ND, {CONV2D_C});
    geGraph->AddDataEdge(bias, 0, conv, 2);
    GNode relu = CreateRelu(geGraph, "relu", dtype, FORMAT_NHWC, CONV2D_FMAP_NHWC);
    geGraph->AddDataEdge(conv, 0, relu, 0);

    std::vector<EsTensorHolder> outputs;
    outputs.push_back(EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(relu, 0)));
    GraphPtr graph(graphBuilder.BuildAndReset(outputs));
    TestTotalPass("conv_no_mul_downstream", graph, CONV_NOT_CHANGED);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "Conv2D"), 1);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "Mul"), 0);
}

// ==========================================================================================
// No fusion: filter input not connected (L93)
// ==========================================================================================
TEST_F(AConv2dMulFusionTest, filter_input_null_reject)
{
    MulFusionOptions opt;
    opt.filterNotConnected = true;
    auto graph = BuildMulFusionGraph("filter_input_null", opt);
    TestTotalPass("filter_input_null", graph, CONV_NOT_CHANGED);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "Mul"), 1);
}

// ==========================================================================================
// No fusion: bias input not connected while hasBias (L104)
// ==========================================================================================
TEST_F(AConv2dMulFusionTest, bias_input_null_reject)
{
    MulFusionOptions opt;
    opt.withBias = true;
    opt.biasNotConnected = true;
    auto graph = BuildMulFusionGraph("bias_input_null", opt);
    TestTotalPass("bias_input_null", graph, CONV_NOT_CHANGED);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "Mul"), 1);
}

// ==========================================================================================
// No fusion: bias is not Const (L109)
// ==========================================================================================
TEST_F(AConv2dMulFusionTest, bias_not_const_reject)
{
    MulFusionOptions opt;
    opt.withBias = true;
    opt.biasIsConst = false;
    auto graph = BuildMulFusionGraph("bias_not_const", opt);
    TestTotalPass("bias_not_const", graph, CONV_NOT_CHANGED);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "Mul"), 1);
}

// ==========================================================================================
// No fusion: Conv2D output dim != 4 (L131)
// ==========================================================================================
TEST_F(AConv2dMulFusionTest, conv2d_output_dim_not4_reject)
{
    MulFusionOptions opt;
    opt.overrideConvOutput = true;
    opt.convOutputShape = {1, 16, 28, 28, CONV2D_C};
    opt.convOutputFormat = FORMAT_NHWC;
    auto graph = BuildMulFusionGraph("conv2d_output_dim_not4", opt);
    TestTotalPass("conv2d_output_dim_not4", graph, CONV_NOT_CHANGED);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "Mul"), 1);
}

// ==========================================================================================
// No fusion: Conv2D output format not NCHW/NHWC (L134)
// ==========================================================================================
TEST_F(AConv2dMulFusionTest, conv2d_output_format_invalid_reject)
{
    MulFusionOptions opt;
    opt.overrideConvOutput = true;
    opt.convOutputShape = CONV2D_FMAP_NHWC;
    opt.convOutputFormat = FORMAT_ND;
    auto graph = BuildMulFusionGraph("conv2d_output_format_invalid", opt);
    TestTotalPass("conv2d_output_format_invalid", graph, CONV_NOT_CHANGED);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "Mul"), 1);
}

// ==========================================================================================
// No fusion: Conv2D scale has unknown shape (L141)
// ==========================================================================================
TEST_F(AConv2dMulFusionTest, conv2d_scale_unknown_shape_reject)
{
    MulFusionOptions opt;
    opt.scaleShape = {-1};
    auto graph = BuildMulFusionGraph("conv2d_scale_unknown_shape", opt);
    TestTotalPass("conv2d_scale_unknown_shape", graph, CONV_NOT_CHANGED);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "Mul"), 1);
}

// ==========================================================================================
// No fusion: Conv2D scale dim > 1, not empty (L148)
// ==========================================================================================
TEST_F(AConv2dMulFusionTest, conv2d_scale_dim_not1_reject)
{
    MulFusionOptions opt;
    opt.scaleShape = {1, 1};
    auto graph = BuildMulFusionGraph("conv2d_scale_dim_not1", opt);
    TestTotalPass("conv2d_scale_dim_not1", graph, CONV_NOT_CHANGED);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "Mul"), 1);
}

// ==========================================================================================
// No fusion: Conv3D output dim != 5 (L160)
// ==========================================================================================
TEST_F(AConv2dMulFusionTest, conv3d_output_dim_not5_reject)
{
    MulFusionOptions opt;
    opt.isConv3D = true;
    opt.overrideConvOutput = true;
    opt.convOutputShape = {1, CONV3D_C, 28, 28};
    opt.convOutputFormat = FORMAT_NDHWC;
    auto graph = BuildMulFusionGraph("conv3d_output_dim_not5", opt);
    TestTotalPass("conv3d_output_dim_not5", graph, CONV_NOT_CHANGED);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "Mul"), 1);
}

// ==========================================================================================
// No fusion: Conv3D outputC < 0, unknown shape (L164)
// ==========================================================================================
TEST_F(AConv2dMulFusionTest, conv3d_outputC_unknown_reject)
{
    MulFusionOptions opt;
    opt.isConv3D = true;
    opt.overrideConvOutput = true;
    opt.convOutputShape = {1, 16, 28, 28, -1};
    opt.convOutputFormat = FORMAT_NDHWC;
    auto graph = BuildMulFusionGraph("conv3d_outputC_unknown", opt);
    TestTotalPass("conv3d_outputC_unknown", graph, CONV_NOT_CHANGED);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "Mul"), 1);
}

// ==========================================================================================
// No fusion: Conv3D scale has unknown shape (L170)
// ==========================================================================================
TEST_F(AConv2dMulFusionTest, conv3d_scale_unknown_shape_reject)
{
    MulFusionOptions opt;
    opt.isConv3D = true;
    opt.scaleShape = {-1};
    auto graph = BuildMulFusionGraph("conv3d_scale_unknown_shape", opt);
    TestTotalPass("conv3d_scale_unknown_shape", graph, CONV_NOT_CHANGED);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "Mul"), 1);
}

// ==========================================================================================
// No fusion: Conv3D scale 1D C mismatch (L175)
// ==========================================================================================
TEST_F(AConv2dMulFusionTest, conv3d_scale_1d_c_mismatch_reject)
{
    MulFusionOptions opt;
    opt.isConv3D = true;
    opt.scaleShape = {2};
    auto graph = BuildMulFusionGraph("conv3d_scale_1d_c_mismatch", opt);
    TestTotalPass("conv3d_scale_1d_c_mismatch", graph, CONV_NOT_CHANGED);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "Mul"), 1);
}

// ==========================================================================================
// No fusion: Conv3D scale 5D not channel-wise (L181)
// ==========================================================================================
TEST_F(AConv2dMulFusionTest, conv3d_scale_5d_not_channelwise_reject)
{
    MulFusionOptions opt;
    opt.isConv3D = true;
    opt.scaleShape = {1, 1, 1, 1, 2};
    auto graph = BuildMulFusionGraph("conv3d_scale_5d_not_channelwise", opt);
    TestTotalPass("conv3d_scale_5d_not_channelwise", graph, CONV_NOT_CHANGED);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "Mul"), 1);
}

// ==========================================================================================
// No fusion: Conv3D scale dim not 1 or 5 (L185)
// ==========================================================================================
TEST_F(AConv2dMulFusionTest, conv3d_scale_dim_not_1_or_5_reject)
{
    MulFusionOptions opt;
    opt.isConv3D = true;
    opt.scaleShape = {1, 1};
    auto graph = BuildMulFusionGraph("conv3d_scale_dim_not_1_or_5", opt);
    TestTotalPass("conv3d_scale_dim_not_1_or_5", graph, CONV_NOT_CHANGED);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "Mul"), 1);
}

// ==========================================================================================
// Success: Conv2D NCHW without bias, scale[1]
// ==========================================================================================
TEST_F(AConv2dMulFusionTest, conv2d_nchw_nobias_scale1_success)
{
    MulFusionOptions opt;
    opt.isNCHW = true;
    opt.withBias = false;
    opt.scaleShape = {1};
    auto graph = BuildMulFusionGraph("conv2d_nchw_nobias_scale1", opt);
    TestTotalPass("conv2d_nchw_nobias_scale1", graph, SUCCESS);

    EXPECT_EQ(GraphChecker::CountNodes(graph, "Conv2D"), 1);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "Mul"), 1);
    EXPECT_TRUE(VerifyProducerType(graph, "Conv2D", 1, "Mul"));
    EXPECT_TRUE(VerifyProducerType(graph, "Relu", 0, "Conv2D"));
}

// ==========================================================================================
// Success: Conv3D without bias, scale[C] (1D)
// ==========================================================================================
TEST_F(AConv2dMulFusionTest, conv3d_nobias_scaleC_success)
{
    MulFusionOptions opt;
    opt.isConv3D = true;
    opt.withBias = false;
    opt.scaleShape = {CONV3D_C};
    auto graph = BuildMulFusionGraph("conv3d_nobias_scaleC", opt);
    TestTotalPass("conv3d_nobias_scaleC", graph, SUCCESS);

    EXPECT_EQ(GraphChecker::CountNodes(graph, "Conv3D"), 1);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "Mul"), 1);
    EXPECT_TRUE(VerifyProducerType(graph, "Conv3D", 1, "Mul"));
    EXPECT_TRUE(VerifyProducerType(graph, "Relu", 0, "Conv3D"));
}

// ==========================================================================================
// Success: Conv3D without bias, scale[1,1,1,1,C] (5D)
// ==========================================================================================
TEST_F(AConv2dMulFusionTest, conv3d_nobias_scale5d_success)
{
    MulFusionOptions opt;
    opt.isConv3D = true;
    opt.withBias = false;
    opt.scaleShape = {1, 1, 1, 1, CONV3D_C};
    auto graph = BuildMulFusionGraph("conv3d_nobias_scale5d", opt);
    TestTotalPass("conv3d_nobias_scale5d", graph, SUCCESS);

    EXPECT_EQ(GraphChecker::CountNodes(graph, "Conv3D"), 1);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "Mul"), 1);
    EXPECT_TRUE(VerifyProducerType(graph, "Conv3D", 1, "Mul"));
    EXPECT_TRUE(VerifyProducerType(graph, "Relu", 0, "Conv3D"));
}

// ==========================================================================================
// Success: Conv2D with mul inputs reversed (scale at mul.0, conv at mul.1).
// Verifies DetermineMulInputIndices + index-aware edge relink.
// ==========================================================================================
TEST_F(AConv2dMulFusionTest, conv2d_mul_input_order_reversed_success)
{
    MulFusionOptions opt;
    opt.reversedMulInputs = true;
    opt.withBias = true;
    opt.scaleShape = {1};
    auto graph = BuildMulFusionGraph("conv2d_mul_input_order_reversed", opt);
    TestTotalPass("conv2d_mul_input_order_reversed", graph, SUCCESS);

    EXPECT_EQ(GraphChecker::CountNodes(graph, "Conv2D"), 1);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "Mul"), 2);
    EXPECT_TRUE(VerifyProducerType(graph, "Conv2D", 1, "Mul"));
    EXPECT_TRUE(VerifyProducerType(graph, "Conv2D", 2, "Mul"));
    EXPECT_TRUE(VerifyProducerType(graph, "Relu", 0, "Conv2D"));
}

// ==========================================================================================
// Success: Conv2D mul output with multiple downstream consumers.
// Verifies RelinkConvOutputToMulConsumers loop over >1 consumer.
// ==========================================================================================
TEST_F(AConv2dMulFusionTest, conv2d_mul_multi_consumer_success)
{
    MulFusionOptions opt;
    opt.mulMultiConsumer = true;
    opt.withBias = true;
    opt.scaleShape = {1};
    auto graph = BuildMulFusionGraph("conv2d_mul_multi_consumer", opt);
    TestTotalPass("conv2d_mul_multi_consumer", graph, SUCCESS);

    EXPECT_EQ(GraphChecker::CountNodes(graph, "Conv2D"), 1);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "Mul"), 2);
    EXPECT_TRUE(VerifyProducerType(graph, "Conv2D", 1, "Mul"));
    EXPECT_TRUE(VerifyProducerType(graph, "Conv2D", 2, "Mul"));
    EXPECT_EQ(GraphChecker::CountNodes(graph, "Relu"), 2);
}

// ==========================================================================================
// Success: Conv3D on dav3510, NDHWC output, scale[1,1,1,1,C]
// Verifies isDav3510 + NDHWC branch in CheckScaleShapeConv3d.
// ==========================================================================================
TEST_F(AConv2dMulFusionTest, conv3d_dav3510_ndhwc_scale5d_success)
{
    MulFusionOptions opt;
    opt.isConv3D = true;
    opt.isDav3510 = true;
    opt.scaleShape = {1, 1, 1, 1, CONV3D_C};
    auto graph = BuildMulFusionGraph("conv3d_dav3510_ndhwc_scale5d", opt);
    TestTotalPass("conv3d_dav3510_ndhwc_scale5d", graph, SUCCESS);

    EXPECT_EQ(GraphChecker::CountNodes(graph, "Mul"), 2);
    EXPECT_TRUE(VerifyProducerType(graph, "Conv3D", 1, "Mul"));
}

// ==========================================================================================
// No fusion: Conv3D on dav3510, NCDHW output intercepted (dav3510 only supports NDHWC)
// ==========================================================================================
TEST_F(AConv2dMulFusionTest, conv3d_dav3510_ncdhw_scale5d_reject)
{
    MulFusionOptions opt;
    opt.isConv3D = true;
    opt.isDav3510 = true;
    opt.overrideConvOutput = true;
    opt.convOutputShape = {1, CONV3D_C, 16, 28, 28};
    opt.convOutputFormat = FORMAT_NCDHW;
    opt.scaleShape = {1, CONV3D_C, 1, 1, 1};
    auto graph = BuildMulFusionGraph("conv3d_dav3510_ncdhw_scale5d", opt);
    TestTotalPass("conv3d_dav3510_ncdhw_scale5d", graph, CONV_NOT_CHANGED);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "Mul"), 1);
}

// ==========================================================================================
// No fusion: Conv3D on dav3510, output format is ND (not NDHWC).
// ==========================================================================================
TEST_F(AConv2dMulFusionTest, conv3d_dav3510_format_invalid_reject)
{
    MulFusionOptions opt;
    opt.isConv3D = true;
    opt.isDav3510 = true;
    opt.overrideConvOutput = true;
    opt.convOutputShape = {1, 16, 28, 28, CONV3D_C};
    opt.convOutputFormat = FORMAT_ND;
    opt.scaleShape = {1};
    auto graph = BuildMulFusionGraph("conv3d_dav3510_format_invalid", opt);
    TestTotalPass("conv3d_dav3510_format_invalid", graph, CONV_NOT_CHANGED);
}

// ==========================================================================================
// No fusion: Conv3D on dav3510, NCDHW output intercepted (dav3510 only supports NDHWC).
// ==========================================================================================
TEST_F(AConv2dMulFusionTest, conv3d_dav3510_ncdhw_scale5d_mismatch_reject)
{
    MulFusionOptions opt;
    opt.isConv3D = true;
    opt.isDav3510 = true;
    opt.overrideConvOutput = true;
    opt.convOutputShape = {1, CONV3D_C, 16, 28, 28};
    opt.convOutputFormat = FORMAT_NCDHW;
    opt.scaleShape = {1, 1, 1, 1, CONV3D_C};
    auto graph = BuildMulFusionGraph("conv3d_dav3510_ncdhw_scale5d_mismatch", opt);
    TestTotalPass("conv3d_dav3510_ncdhw_scale5d_mismatch", graph, CONV_NOT_CHANGED);
}

// ==========================================================================================
// No fusion: Conv2D on dav3510, NCHW + channel-wise scale 1D[C] intercepted
// ==========================================================================================
TEST_F(AConv2dMulFusionTest, conv2d_dav3510_nchw_channelwise_reject)
{
    MulFusionOptions opt;
    opt.isDav3510 = true;
    opt.isNCHW = true;
    opt.withBias = true;
    opt.scaleShape = {CONV2D_C};
    auto graph = BuildMulFusionGraph("conv2d_dav3510_nchw_channelwise", opt);
    TestTotalPass("conv2d_dav3510_nchw_channelwise", graph, CONV_NOT_CHANGED);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "Mul"), 1);
}

// ==========================================================================================
// No fusion: Conv3D on dav3510, NCDHW output intercepted (dav3510 only supports NDHWC)
// ==========================================================================================
TEST_F(AConv2dMulFusionTest, conv3d_dav3510_ncdhw_channelwise_reject)
{
    MulFusionOptions opt;
    opt.isConv3D = true;
    opt.isDav3510 = true;
    opt.overrideConvOutput = true;
    opt.convOutputShape = {1, CONV3D_C, 16, 28, 28};
    opt.convOutputFormat = FORMAT_NCDHW;
    opt.scaleShape = {CONV3D_C};
    auto graph = BuildMulFusionGraph("conv3d_dav3510_ncdhw_channelwise", opt);
    TestTotalPass("conv3d_dav3510_ncdhw_channelwise", graph, CONV_NOT_CHANGED);
    EXPECT_EQ(GraphChecker::CountNodes(graph, "Mul"), 1);
}

#endif
