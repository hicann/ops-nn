/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <functional>
#include <string>
#include <vector>

#include "../../../../common/tests/ut/op_graph/test_conv_fusion_pass_framework.h"
#include "../../../op_graph/fusion_pass/pad_conv2d_fusion_pass.h"

#include "version/ge-compiler_version.h"
#if GE_COMPILER_VERSION_NUM >= 90000000U

using namespace ge;
using namespace es;
using namespace fe;
using namespace Ops;
using namespace NN;
using namespace Conv;
using namespace ConvFusionUtils;
using namespace PadConv2dFusion;
using namespace test_conv_fusion_framework;

#define CONV_DEBUG false

namespace {
const std::string PAD_TYPE = "Pad";
const std::string PADV3_TYPE = "PadV3";
const std::string SOC_ASCEND950 = "Ascend950";
const std::string SOC_MC62 = "MC62";
const std::vector<int64_t> PADDINGS_CONST_DIMS = {4, 2};
const std::vector<int64_t> UNKNOWN_PAD_OUT_SHAPE = {-1, 230, 230, 3};

struct PadConvOptions {
    std::string graphName = "test_pad_conv2d";
    std::string padName = "pad";
    std::string padType = PADV3_TYPE;
    std::string padMode = "constant";
    bool setPadMode = true;
    DataType dtype = DT_FLOAT;
    Format fmt = FORMAT_NHWC;
    Format padInputFormat = FORMAT_NHWC;
    Format filterFmt = FORMAT_HWCN;
    std::vector<int64_t> dataShape = {1, 224, 224, 3};
    std::vector<int64_t> padOutShape = {1, 230, 230, 3};
    std::vector<int64_t> convOutShape = {1, 112, 112, 64};
    std::vector<int64_t> filterShape = {7, 7, 3, 64};
    // [batch_before, batch_after, h_before, h_after, w_before, w_after, c_before, c_after] for NHWC
    std::vector<int32_t> paddings = {0, 0, 3, 3, 3, 3, 0, 0};
    std::vector<int64_t> paddingsConstDims = PADDINGS_CONST_DIMS;
    std::vector<int64_t> convPads = {0, 0, 0, 0};
    bool paddingsContiguous = true;
    bool paddingsInt64 = false;
    bool paddingsAsFloat = false;
    bool withConstantValues = false;
    bool emptyConstantValues = false;
    DataType constantValuesDtype = DT_FLOAT;
    float constantValue = 0.0f;
    bool twoConv = false;
    bool twoDw = false;
    bool extraConsumer = false;
    bool withBackward = false;
    bool withDwOnly = false;
    bool withBnNoDx = false;
    bool dxMultiConsumer = false;
    std::string sliceOpType = "Slice";
    bool ctrlEdgeToConv = false;
    bool ctrlEdgeToOther = false;
    bool ctrlEdgeToDw = false;
    bool ctrlEdgeToDx = false;
    bool unknownPadShape = false;
    bool unknownPadInputShape = false;
    bool skipPadNode = false;
};

void ApplyNchwLayout(PadConvOptions& opt)
{
    opt.fmt = FORMAT_NCHW;
    opt.padInputFormat = FORMAT_NCHW;
    opt.filterFmt = FORMAT_NCHW;
    opt.dataShape = {1, 3, 224, 224};
    opt.padOutShape = {1, 3, 230, 230};
    opt.convOutShape = {1, 64, 112, 112};
    opt.filterShape = {64, 3, 7, 7};
}
} // namespace

class PadConv2dFusionPassTest : public testing::Test {
protected:
    static void SetUpTestCase() {}

    static void TearDownTestCase() {}

    void SetupSoc(const std::string& shortSoc, bool supportDn2Nz = false)
    {
        PlatformInfo platformInfo;
        OptionalInfo optiInfo;
        optiInfo.soc_version = shortSoc;
        platformInfo.str_info.short_soc_version = shortSoc;
        if (supportDn2Nz) {
            platformInfo.ai_core_intrinsic_dtype_map["Intrinsic_data_move_out2l1_dn2nz"] = {"f16,f32"};
        }
        PlatformInfoManager::Instance().platform_info_map_[shortSoc] = platformInfo;
        PlatformInfoManager::Instance().SetOptionalCompilationInfo(optiInfo);
    }

    GNode CreateConstNode(Graph* graph, const std::string& name, DataType dtype, Format fmt,
                          const std::vector<int64_t>& shape)
    {
        GNode node = CompliantNodeBuilder(graph)
                         .OpType("Const")
                         .Name(name.c_str())
                         .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                         .Build();
        node.UpdateOutputDesc(0, BuildTensorDesc(dtype, fmt, shape));
        return node;
    }

    GNode CreateSingleIoNode(Graph* graph, const std::string& opType, const std::string& name, const TensorDesc& desc)
    {
        GNode node = CompliantNodeBuilder(graph)
                         .OpType(opType.c_str())
                         .Name(name.c_str())
                         .IrDefInputs({{"x", CompliantNodeBuilder::kEsIrInputRequired, ""}})
                         .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                         .Build();
        node.UpdateInputDesc(0, desc);
        node.UpdateOutputDesc(0, desc);
        return node;
    }

    GNode CreateConv2DNode(Graph* graph, const std::string& name, const PadConvOptions& opt)
    {
        GNode node = CompliantNodeBuilder(graph)
                         .OpType("Conv2D")
                         .Name(name.c_str())
                         .IrDefInputs({{"x", CompliantNodeBuilder::kEsIrInputRequired, ""},
                                       {"filter", CompliantNodeBuilder::kEsIrInputRequired, ""},
                                       {"bias", CompliantNodeBuilder::kEsIrInputOptional, ""},
                                       {"offset_w", CompliantNodeBuilder::kEsIrInputOptional, ""}})
                         .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                         .Build();
        std::vector<int64_t> strides = {1, 2, 2, 1};
        std::vector<int64_t> dilations = {1, 1, 1, 1};
        std::vector<int64_t> pads = opt.convPads;
        int64_t groups = 1;
        AscendString dataFormat(opt.fmt == FORMAT_NHWC ? "NHWC" : "NCHW");
        node.SetAttr(AscendString("strides"), strides);
        node.SetAttr(AscendString("dilations"), dilations);
        node.SetAttr(AscendString("pads"), pads);
        node.SetAttr(AscendString("groups"), groups);
        node.SetAttr(AscendString("data_format"), dataFormat);
        node.UpdateInputDesc(0, BuildTensorDesc(opt.dtype, opt.fmt, opt.padOutShape));
        node.UpdateInputDesc(1, BuildTensorDesc(opt.dtype, opt.filterFmt, opt.filterShape));
        node.UpdateOutputDesc(0, BuildTensorDesc(opt.dtype, opt.fmt, opt.convOutShape));
        return node;
    }

    GNode CreateDwNode(Graph* graph, const std::string& name, const PadConvOptions& opt)
    {
        GNode node = CompliantNodeBuilder(graph)
                         .OpType("Conv2DBackpropFilterD")
                         .Name(name.c_str())
                         .IrDefInputs({{"x", CompliantNodeBuilder::kEsIrInputRequired, ""},
                                       {"out_backprop", CompliantNodeBuilder::kEsIrInputRequired, ""}})
                         .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                         .Build();
        std::vector<int64_t> dwPads = opt.convPads;
        node.SetAttr(AscendString("pads"), dwPads);
        node.UpdateInputDesc(0, BuildTensorDesc(opt.dtype, opt.fmt, opt.padOutShape));
        node.UpdateInputDesc(1, BuildTensorDesc(opt.dtype, opt.fmt, opt.convOutShape));
        node.UpdateOutputDesc(0, BuildTensorDesc(opt.dtype, opt.filterFmt, opt.filterShape));
        return node;
    }

    GNode CreatePadNode(EsGraphBuilder& graphBuilder, const PadConvOptions& opt, const EsTensorHolder& input)
    {
        Graph* graph = graphBuilder.GetCGraphBuilder()->GetGraph();
        std::vector<CompliantNodeBuilder::IrInputDef> padInputs = {
            {"x", CompliantNodeBuilder::kEsIrInputRequired, ""},
            {"paddings", CompliantNodeBuilder::kEsIrInputRequired, ""}};
        if (opt.padType == PADV3_TYPE) {
            padInputs.push_back({"constant_values", CompliantNodeBuilder::kEsIrInputOptional, ""});
        }
        GNode padNode = CompliantNodeBuilder(graph)
                            .OpType(opt.padType.c_str())
                            .Name(opt.padName.c_str())
                            .IrDefInputs(padInputs)
                            .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                            .Build();
        if (opt.padType == PADV3_TYPE) {
            bool paddingsContiguous = opt.paddingsContiguous;
            if (opt.setPadMode) {
                AscendString padMode(opt.padMode.c_str());
                padNode.SetAttr(AscendString("mode"), padMode);
            }
            padNode.SetAttr(AscendString("paddings_contiguous"), paddingsContiguous);
        }

        DataType paddingsDtype = DT_INT32;
        if (opt.paddingsAsFloat) {
            paddingsDtype = DT_FLOAT;
        } else if (opt.paddingsInt64) {
            paddingsDtype = DT_INT64;
        }
        std::vector<int64_t> padInShape = opt.dataShape;
        if (opt.unknownPadInputShape && !padInShape.empty()) {
            padInShape[0] = -1;
        }
        padNode.UpdateInputDesc(0, BuildTensorDesc(opt.dtype, opt.padInputFormat, padInShape));
        padNode.UpdateInputDesc(1, BuildTensorDesc(paddingsDtype, FORMAT_ND, opt.paddingsConstDims));
        padNode.UpdateOutputDesc(
            0, BuildTensorDesc(opt.dtype, opt.fmt, opt.unknownPadShape ? UNKNOWN_PAD_OUT_SHAPE : opt.padOutShape));
        graph->AddDataEdge(*input.GetProducer(), input.GetProducerOutIndex(), padNode, 0);

        if (opt.paddingsAsFloat) {
            std::vector<float> paddingsData(opt.paddings.begin(), opt.paddings.end());
            auto paddingsConst = graphBuilder.CreateConst(paddingsData, opt.paddingsConstDims);
            graph->AddDataEdge(*paddingsConst.GetProducer(), paddingsConst.GetProducerOutIndex(), padNode, 1);
        } else if (opt.paddingsInt64) {
            std::vector<int64_t> paddingsData(opt.paddings.begin(), opt.paddings.end());
            auto paddingsConst = graphBuilder.CreateConst(paddingsData, opt.paddingsConstDims);
            graph->AddDataEdge(*paddingsConst.GetProducer(), paddingsConst.GetProducerOutIndex(), padNode, 1);
        } else {
            auto paddingsConst = graphBuilder.CreateConst(opt.paddings, opt.paddingsConstDims);
            graph->AddDataEdge(*paddingsConst.GetProducer(), paddingsConst.GetProducerOutIndex(), padNode, 1);
        }

        if (opt.withConstantValues) {
            if (opt.emptyConstantValues) {
                auto constantConst = graphBuilder.CreateConst(std::vector<float>{0.0f}, std::vector<int64_t>{0});
                padNode.UpdateInputDesc(2, BuildTensorDesc(DT_FLOAT, FORMAT_ND, {0}));
                graph->AddDataEdge(*constantConst.GetProducer(), constantConst.GetProducerOutIndex(), padNode, 2);
            } else if (opt.constantValuesDtype == DT_INT32) {
                std::vector<int32_t> constantValues = {static_cast<int32_t>(opt.constantValue)};
                auto constantConst = graphBuilder.CreateConst(constantValues, {1});
                padNode.UpdateInputDesc(2, BuildTensorDesc(DT_INT32, FORMAT_ND, {1}));
                graph->AddDataEdge(*constantConst.GetProducer(), constantConst.GetProducerOutIndex(), padNode, 2);
            } else {
                std::vector<float> constantValues = {opt.constantValue};
                auto constantConst = graphBuilder.CreateConst(constantValues, {1});
                padNode.UpdateInputDesc(2, BuildTensorDesc(DT_FLOAT, FORMAT_ND, {1}));
                graph->AddDataEdge(*constantConst.GetProducer(), constantConst.GetProducerOutIndex(), padNode, 2);
            }
        }
        return padNode;
    }

    // pad -> dw, grads -> dbn -> dw/dx, dx -> slice -> relu
    void AddBackwardPath(EsGraphBuilder& graphBuilder, const PadConvOptions& opt, GNode& padNode, GNode& filterConst,
                         std::vector<EsTensorHolder>& outputs)
    {
        Graph* graph = graphBuilder.GetCGraphBuilder()->GetGraph();
        auto gradInput = graphBuilder.CreateInput(1, "grads", opt.dtype, opt.fmt, opt.convOutShape);
        TensorDesc gradDesc = BuildTensorDesc(opt.dtype, opt.fmt, opt.convOutShape);
        TensorDesc padOutDesc = BuildTensorDesc(opt.dtype, opt.fmt, opt.padOutShape);
        TensorDesc dataDesc = BuildTensorDesc(opt.dtype, opt.fmt, opt.dataShape);

        GNode dbnNode = CreateSingleIoNode(graph, "BNTrainingReduceGrad", "dbn", gradDesc);
        graph->AddDataEdge(*gradInput.GetProducer(), gradInput.GetProducerOutIndex(), dbnNode, 0);

        GNode dwNode = CreateDwNode(graph, "dw", opt);
        graph->AddDataEdge(padNode, 0, dwNode, 0);
        graph->AddDataEdge(dbnNode, 0, dwNode, 1);

        GNode dxNode = CompliantNodeBuilder(graph)
                           .OpType("Conv2DBackpropInputD")
                           .Name("dx")
                           .IrDefInputs({{"filter", CompliantNodeBuilder::kEsIrInputRequired, ""},
                                         {"out_backprop", CompliantNodeBuilder::kEsIrInputRequired, ""}})
                           .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                           .Build();
        std::vector<int64_t> dxPads = opt.convPads;
        dxNode.SetAttr(AscendString("pads"), dxPads);
        dxNode.UpdateInputDesc(0, BuildTensorDesc(opt.dtype, opt.filterFmt, opt.filterShape));
        dxNode.UpdateInputDesc(1, gradDesc);
        dxNode.UpdateOutputDesc(0, padOutDesc);
        graph->AddDataEdge(filterConst, 0, dxNode, 0);
        graph->AddDataEdge(dbnNode, 0, dxNode, 1);

        GNode sliceNode = CreateSingleIoNode(graph, opt.sliceOpType, "slice", padOutDesc);
        sliceNode.UpdateOutputDesc(0, dataDesc);
        graph->AddDataEdge(dxNode, 0, sliceNode, 0);

        GNode sliceConsumer = CreateSingleIoNode(graph, "Relu", "slice_relu", dataDesc);
        graph->AddDataEdge(sliceNode, 0, sliceConsumer, 0);

        if (opt.dxMultiConsumer) {
            GNode extraDxConsumer = CreateSingleIoNode(graph, "Relu", "dx_relu", padOutDesc);
            graph->AddDataEdge(dxNode, 0, extraDxConsumer, 0);
            outputs.emplace_back(
                EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(extraDxConsumer, 0)));
        }
        if (opt.ctrlEdgeToDw) {
            graph->AddControlEdge(padNode, dwNode);
        }
        if (opt.ctrlEdgeToDx) {
            graph->AddControlEdge(padNode, dxNode);
        }

        outputs.emplace_back(EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(dwNode, 0)));
        outputs.emplace_back(
            EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(sliceConsumer, 0)));
    }

    void AddDwOnlyPath(EsGraphBuilder& graphBuilder, const PadConvOptions& opt, GNode& padNode,
                       std::vector<EsTensorHolder>& outputs)
    {
        Graph* graph = graphBuilder.GetCGraphBuilder()->GetGraph();
        auto gradInput = graphBuilder.CreateInput(1, "grads", opt.dtype, opt.fmt, opt.convOutShape);
        GNode dwNode = CreateDwNode(graph, "dw", opt);
        graph->AddDataEdge(padNode, 0, dwNode, 0);
        graph->AddDataEdge(*gradInput.GetProducer(), gradInput.GetProducerOutIndex(), dwNode, 1);
        if (opt.ctrlEdgeToDw) {
            graph->AddControlEdge(padNode, dwNode);
        }
        outputs.emplace_back(EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(dwNode, 0)));
    }

    void AddBnNoDxPath(EsGraphBuilder& graphBuilder, const PadConvOptions& opt, GNode& padNode,
                       std::vector<EsTensorHolder>& outputs)
    {
        Graph* graph = graphBuilder.GetCGraphBuilder()->GetGraph();
        auto gradInput = graphBuilder.CreateInput(1, "grads", opt.dtype, opt.fmt, opt.convOutShape);
        TensorDesc gradDesc = BuildTensorDesc(opt.dtype, opt.fmt, opt.convOutShape);
        GNode dbnNode = CreateSingleIoNode(graph, "BNTrainingReduceGrad", "dbn", gradDesc);
        graph->AddDataEdge(*gradInput.GetProducer(), gradInput.GetProducerOutIndex(), dbnNode, 0);
        GNode dwNode = CreateDwNode(graph, "dw", opt);
        graph->AddDataEdge(padNode, 0, dwNode, 0);
        graph->AddDataEdge(dbnNode, 0, dwNode, 1);
        outputs.emplace_back(EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(dwNode, 0)));
    }

    GraphPtr BuildPadConvGraph(const PadConvOptions& opt)
    {
        EsGraphBuilder graphBuilder(opt.graphName.c_str());
        auto input = graphBuilder.CreateInput(0, "data", opt.dtype, opt.fmt, opt.dataShape);
        Graph* graph = graphBuilder.GetCGraphBuilder()->GetGraph();

        GNode filterConst = CreateConstNode(graph, "filter", opt.dtype, opt.filterFmt, opt.filterShape);
        GNode convNode = CreateConv2DNode(graph, "conv2d", opt);
        std::vector<EsTensorHolder> outputs;
        outputs.emplace_back(EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(convNode, 0)));

        if (opt.skipPadNode) {
            graph->AddDataEdge(*input.GetProducer(), input.GetProducerOutIndex(), convNode, 0);
            graph->AddDataEdge(filterConst, 0, convNode, 1);
            return graphBuilder.BuildAndReset(outputs);
        }

        GNode padNode = CreatePadNode(graphBuilder, opt, input);
        graph->AddDataEdge(padNode, 0, convNode, 0);
        graph->AddDataEdge(filterConst, 0, convNode, 1);

        if (opt.twoConv) {
            GNode secondConv = CreateConv2DNode(graph, "conv2d_1", opt);
            graph->AddDataEdge(padNode, 0, secondConv, 0);
            graph->AddDataEdge(filterConst, 0, secondConv, 1);
            outputs.emplace_back(
                EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(secondConv, 0)));
        }
        if (opt.twoDw) {
            auto gradInput = graphBuilder.CreateInput(1, "grads", opt.dtype, opt.fmt, opt.convOutShape);
            GNode dw0 = CreateDwNode(graph, "dw", opt);
            GNode dw1 = CreateDwNode(graph, "dw_1", opt);
            graph->AddDataEdge(padNode, 0, dw0, 0);
            graph->AddDataEdge(*gradInput.GetProducer(), gradInput.GetProducerOutIndex(), dw0, 1);
            graph->AddDataEdge(padNode, 0, dw1, 0);
            graph->AddDataEdge(*gradInput.GetProducer(), gradInput.GetProducerOutIndex(), dw1, 1);
            outputs.emplace_back(EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(dw0, 0)));
            outputs.emplace_back(EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(dw1, 0)));
        }
        if (opt.extraConsumer) {
            GNode padConsumer = CreateSingleIoNode(graph, "Relu", "pad_relu",
                                                   BuildTensorDesc(opt.dtype, opt.fmt, opt.padOutShape));
            graph->AddDataEdge(padNode, 0, padConsumer, 0);
            outputs.emplace_back(
                EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(padConsumer, 0)));
        }
        if (opt.withBackward) {
            AddBackwardPath(graphBuilder, opt, padNode, filterConst, outputs);
        }
        if (opt.withDwOnly) {
            AddDwOnlyPath(graphBuilder, opt, padNode, outputs);
        }
        if (opt.withBnNoDx) {
            AddBnNoDxPath(graphBuilder, opt, padNode, outputs);
        }
        if (opt.ctrlEdgeToConv) {
            graph->AddControlEdge(padNode, convNode);
        }
        if (opt.ctrlEdgeToOther) {
            GNode convConsumer = CreateSingleIoNode(graph, "Relu", "conv_relu",
                                                    BuildTensorDesc(opt.dtype, opt.fmt, opt.convOutShape));
            graph->AddDataEdge(convNode, 0, convConsumer, 0);
            graph->AddControlEdge(padNode, convConsumer);
            outputs.emplace_back(
                EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(convConsumer, 0)));
        }
        return graphBuilder.BuildAndReset(outputs);
    }

    GraphPtr BuildTwoPadConvGraph()
    {
        PadConvOptions opt;
        opt.graphName = "test_two_pad_conv2d";
        EsGraphBuilder graphBuilder(opt.graphName.c_str());
        Graph* graph = graphBuilder.GetCGraphBuilder()->GetGraph();
        const std::vector<std::vector<int32_t>> paddingsList = {{0, 0, 3, 3, 3, 3, 0, 0}, {0, 0, 1, 1, 2, 2, 0, 0}};

        std::vector<EsTensorHolder> outputs;
        for (int32_t idx = 0; idx < 2; ++idx) {
            std::string suffix = "_" + std::to_string(idx);
            auto input = graphBuilder.CreateInput(idx, ("data" + suffix).c_str(), opt.dtype, opt.fmt, opt.dataShape);
            std::vector<CompliantNodeBuilder::IrInputDef> padInputs = {
                {"x", CompliantNodeBuilder::kEsIrInputRequired, ""},
                {"paddings", CompliantNodeBuilder::kEsIrInputRequired, ""}};
            GNode padNode = CompliantNodeBuilder(graph)
                                .OpType(PAD_TYPE.c_str())
                                .Name(("pad" + suffix).c_str())
                                .IrDefInputs(padInputs)
                                .IrDefOutputs({{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}})
                                .Build();
            padNode.UpdateInputDesc(0, BuildTensorDesc(opt.dtype, opt.fmt, opt.dataShape));
            padNode.UpdateInputDesc(1, BuildTensorDesc(DT_INT32, FORMAT_ND, PADDINGS_CONST_DIMS));
            padNode.UpdateOutputDesc(0, BuildTensorDesc(opt.dtype, opt.fmt, opt.padOutShape));
            auto paddingsConst = graphBuilder.CreateConst(paddingsList[idx], PADDINGS_CONST_DIMS);
            graph->AddDataEdge(*input.GetProducer(), input.GetProducerOutIndex(), padNode, 0);
            graph->AddDataEdge(*paddingsConst.GetProducer(), paddingsConst.GetProducerOutIndex(), padNode, 1);

            GNode filterConst = CreateConstNode(graph, "filter" + suffix, opt.dtype, opt.filterFmt, opt.filterShape);
            GNode convNode = CreateConv2DNode(graph, "conv2d" + suffix, opt);
            graph->AddDataEdge(padNode, 0, convNode, 0);
            graph->AddDataEdge(filterConst, 0, convNode, 1);
            outputs.emplace_back(EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(convNode, 0)));
        }
        return graphBuilder.BuildAndReset(outputs);
    }

    GraphPtr BuildCascadePadConvGraph()
    {
        PadConvOptions opt;
        opt.graphName = "test_cascade_pad_conv2d";
        EsGraphBuilder graphBuilder(opt.graphName.c_str());
        Graph* graph = graphBuilder.GetCGraphBuilder()->GetGraph();
        auto input = graphBuilder.CreateInput(0, "data", opt.dtype, opt.fmt, opt.dataShape);

        opt.padName = "pad_0";
        opt.padType = PAD_TYPE;
        GNode pad0 = CreatePadNode(graphBuilder, opt, input);
        GNode filter0 = CreateConstNode(graph, "filter_0", opt.dtype, opt.filterFmt, opt.filterShape);
        GNode conv0 = CreateConv2DNode(graph, "conv2d_0", opt);
        graph->AddDataEdge(pad0, 0, conv0, 0);
        graph->AddDataEdge(filter0, 0, conv0, 1);

        PadConvOptions opt1 = opt;
        opt1.padName = "pad_1";
        opt1.dataShape = opt.convOutShape;
        opt1.padOutShape = {1, 114, 114, 64};
        opt1.convOutShape = {1, 54, 54, 64};
        opt1.filterShape = {7, 7, 64, 64};
        opt1.paddings = {0, 0, 1, 1, 1, 1, 0, 0};
        EsTensorHolder conv0Out(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(conv0, 0));
        GNode pad1 = CreatePadNode(graphBuilder, opt1, conv0Out);
        GNode filter1 = CreateConstNode(graph, "filter_1", opt1.dtype, opt1.filterFmt, opt1.filterShape);
        GNode conv1 = CreateConv2DNode(graph, "conv2d_1", opt1);
        graph->AddDataEdge(pad1, 0, conv1, 0);
        graph->AddDataEdge(filter1, 0, conv1, 1);

        std::vector<EsTensorHolder> outputs;
        outputs.emplace_back(EsTensorHolder(graphBuilder.GetCGraphBuilder()->GetTensorHolderFromNode(conv1, 0)));
        return graphBuilder.BuildAndReset(outputs);
    }

    void TestTotalPass(const std::string& passName, GraphPtr& graph, Status expectRes)
    {
        if (CONV_DEBUG) {
            graph->DumpToFile(Graph::DumpFormat::kOnnx, AscendString((passName + "_before").c_str()));
        }
        CustomPassContext passContext;
        passContext.SetPassName(passName.c_str());
        PadConv2dFusionPass pass;
        auto res = pass.Run(graph, passContext);
        if (CONV_DEBUG) {
            graph->DumpToFile(Graph::DumpFormat::kOnnx, AscendString((passName + "_after").c_str()));
        }
        EXPECT_EQ(res, expectRes);
    }

    void ExpectConvPads(GraphPtr& graph, const std::string& convName, const std::vector<int64_t>& expectPads)
    {
        GNode convNode;
        ASSERT_TRUE(GraphChecker::FindNodeByNameSuffix(graph, convName, convNode));
        std::vector<int64_t> pads;
        ASSERT_TRUE(GraphChecker::GetListIntAttr(convNode, "pads", pads));
        EXPECT_EQ(pads, expectPads);
    }

    void ExpectFusedCubeFmap(GraphPtr& graph, const std::string& nodeName, const std::vector<int64_t>& expectShape)
    {
        GNode cubeNode;
        ASSERT_TRUE(GraphChecker::FindNodeByNameSuffix(graph, nodeName, cubeNode));
        std::string producerType;
        EXPECT_TRUE(GraphChecker::GetInputProducerType(cubeNode, 0, producerType));
        EXPECT_EQ(producerType, "Data");
        std::vector<int64_t> fmapShape;
        EXPECT_TRUE(GraphChecker::GetOriginShape(cubeNode, 0, false, fmapShape));
        EXPECT_EQ(fmapShape, expectShape);
    }

    void ExpectNoInControl(GraphPtr& graph, const std::string& nodeName)
    {
        GNode node;
        ASSERT_TRUE(GraphChecker::FindNodeByNameSuffix(graph, nodeName, node));
        EXPECT_TRUE(node.GetInControlNodes().empty());
    }
};

// ==========================================================================================
// Fusion success: Pad / PadV3, NCHW / NHWC, contiguous / non-contiguous, int32 / int64
// ==========================================================================================
TEST_F(PadConv2dFusionPassTest, fusion_success)
{
    struct Point {
        const char* pointName;
        std::function<PadConvOptions()> makeOptions;
    } const points[] = {
        {"pad_nhwc",
         []() {
             PadConvOptions opt;
             opt.padType = PAD_TYPE;
             return opt;
         }},
        {"pad_nchw",
         []() {
             PadConvOptions opt;
             opt.padType = PAD_TYPE;
             ApplyNchwLayout(opt);
             opt.paddings = {0, 0, 0, 0, 3, 3, 3, 3};
             return opt;
         }},
        {"padv3_contiguous",
         []() {
             PadConvOptions opt;
             return opt;
         }},
        {"padv3_non_contiguous",
         []() {
             PadConvOptions opt;
             opt.paddingsContiguous = false;
             opt.paddings = {0, 3, 3, 0, 0, 3, 3, 0};
             return opt;
         }},
        {"padv3_int64_paddings",
         []() {
             PadConvOptions opt;
             opt.paddingsInt64 = true;
             return opt;
         }},
        {"padv3_constant_value_zero",
         []() {
             PadConvOptions opt;
             opt.withConstantValues = true;
             opt.constantValue = 0.0f;
             return opt;
         }},
        {"pad_with_control_edge_to_conv",
         []() {
             PadConvOptions opt;
             opt.padType = PAD_TYPE;
             opt.ctrlEdgeToConv = true;
             return opt;
         }},
        {"padv3_nchw",
         []() {
             PadConvOptions opt;
             ApplyNchwLayout(opt);
             opt.paddings = {0, 0, 0, 0, 3, 3, 3, 3};
             return opt;
         }},
        {"padv3_fp16",
         []() {
             PadConvOptions opt;
             opt.dtype = DT_FLOAT16;
             return opt;
         }},
    };

    for (const auto& point : points) {
        SCOPED_TRACE(point.pointName);
        SetupSoc(SOC_ASCEND950);
        PadConvOptions opt = point.makeOptions();
        opt.graphName = std::string("fusion_success_") + point.pointName;
        auto graph = BuildPadConvGraph(opt);
        TestTotalPass(opt.graphName, graph, SUCCESS);

        EXPECT_FALSE(GraphChecker::HasNode(graph, opt.padType));
        ExpectConvPads(graph, "conv2d", {3, 3, 3, 3});
        ExpectFusedCubeFmap(graph, "conv2d", opt.dataShape);
        ExpectNoInControl(graph, "conv2d");
    }
}

// ==========================================================================================
// Fusion success: conv pads are accumulated with pad paddings
// ==========================================================================================
TEST_F(PadConv2dFusionPassTest, conv_pads_accumulated)
{
    SetupSoc(SOC_ASCEND950);
    PadConvOptions opt;
    opt.graphName = "conv_pads_accumulated";
    opt.convPads = {1, 2, 3, 4};
    auto graph = BuildPadConvGraph(opt);
    TestTotalPass(opt.graphName, graph, SUCCESS);

    ExpectConvPads(graph, "conv2d", {4, 5, 6, 7});
    ExpectFusedCubeFmap(graph, "conv2d", opt.dataShape);
}

TEST_F(PadConv2dFusionPassTest, combined_pad_eq_255_success)
{
    SetupSoc(SOC_ASCEND950);
    PadConvOptions opt;
    opt.graphName = "combined_pad_eq_255";
    opt.paddings = {0, 0, 255, 0, 0, 0, 0, 0};
    auto graph = BuildPadConvGraph(opt);
    TestTotalPass(opt.graphName, graph, SUCCESS);

    EXPECT_FALSE(GraphChecker::HasNode(graph, opt.padType));
    ExpectConvPads(graph, "conv2d", {255, 0, 0, 0});
}

// ==========================================================================================
// Attr verification: padding / auto_pad depend on the dn2nz capability
// ==========================================================================================
TEST_F(PadConv2dFusionPassTest, padding_attr_without_dn2nz)
{
    SetupSoc(SOC_ASCEND950);
    PadConvOptions opt;
    opt.graphName = "padding_attr_without_dn2nz";
    auto graph = BuildPadConvGraph(opt);
    TestTotalPass(opt.graphName, graph, SUCCESS);

    GNode convNode;
    ASSERT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "Conv2D", convNode));
    std::string padding;
    EXPECT_TRUE(GraphChecker::GetNodeStringAttr(convNode, "padding", padding));
    EXPECT_EQ(padding, "SAME");
    std::string autoPad;
    EXPECT_FALSE(GraphChecker::GetNodeStringAttr(convNode, "auto_pad", autoPad));
}

TEST_F(PadConv2dFusionPassTest, padding_attr_with_dn2nz)
{
    SetupSoc(SOC_ASCEND950, true);
    PadConvOptions opt;
    opt.graphName = "padding_attr_with_dn2nz";
    auto graph = BuildPadConvGraph(opt);
    TestTotalPass(opt.graphName, graph, SUCCESS);

    GNode convNode;
    ASSERT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "Conv2D", convNode));
    std::string padding;
    EXPECT_TRUE(GraphChecker::GetNodeStringAttr(convNode, "padding", padding));
    EXPECT_EQ(padding, "EXPLICIT");
    std::string autoPad;
    EXPECT_TRUE(GraphChecker::GetNodeStringAttr(convNode, "auto_pad", autoPad));
    EXPECT_EQ(autoPad, "NOTSET");
}

// ==========================================================================================
// No fusion: topology, pad value, format and PadV3 attr restrictions
// ==========================================================================================
TEST_F(PadConv2dFusionPassTest, no_fusion)
{
    struct Point {
        const char* pointName;
        std::function<PadConvOptions()> makeOptions;
    } const points[] = {
        {"two_conv2d_consumers",
         []() {
             PadConvOptions opt;
             opt.twoConv = true;
             return opt;
         }},
        {"consumer_is_not_cube",
         []() {
             PadConvOptions opt;
             opt.extraConsumer = true;
             return opt;
         }},
        {"control_edge_to_other_node",
         []() {
             PadConvOptions opt;
             opt.ctrlEdgeToOther = true;
             return opt;
         }},
        {"batch_channel_pad_not_zero",
         []() {
             PadConvOptions opt;
             opt.paddings = {1, 1, 3, 3, 3, 3, 0, 0};
             return opt;
         }},
        {"pad_value_negative",
         []() {
             PadConvOptions opt;
             opt.paddings = {0, 0, -1, 3, 3, 3, 0, 0};
             return opt;
         }},
        {"combined_pad_out_of_range",
         []() {
             PadConvOptions opt;
             opt.paddings = {0, 0, 255, 3, 3, 3, 0, 0};
             opt.convPads = {1, 0, 0, 0};
             return opt;
         }},
        {"conv_pads_negative",
         []() {
             PadConvOptions opt;
             opt.convPads = {-1, 0, 0, 0};
             return opt;
         }},
        {"pad_input_format_unsupported",
         []() {
             PadConvOptions opt;
             opt.padInputFormat = FORMAT_ND;
             return opt;
         }},
        {"padv3_mode_not_constant",
         []() {
             PadConvOptions opt;
             opt.padMode = "reflect";
             return opt;
         }},
        {"padv3_constant_value_not_zero",
         []() {
             PadConvOptions opt;
             opt.withConstantValues = true;
             opt.constantValue = 1.0f;
             return opt;
         }},
        {"pad_unknown_shape",
         []() {
             PadConvOptions opt;
             opt.unknownPadShape = true;
             return opt;
         }},
        {"channel_pad_not_zero_nhwc",
         []() {
             PadConvOptions opt;
             opt.paddings = {0, 0, 0, 0, 3, 3, 1, 1};
             return opt;
         }},
        {"channel_pad_not_zero_nchw",
         []() {
             PadConvOptions opt;
             ApplyNchwLayout(opt);
             opt.paddings = {0, 0, 1, 1, 3, 3, 0, 0};
             return opt;
         }},
        {"two_dw_consumers",
         []() {
             PadConvOptions opt;
             opt.twoDw = true;
             return opt;
         }},
        {"paddings_size_less_than_4",
         []() {
             PadConvOptions opt;
             opt.paddings = {0, 0, 3, 3};
             opt.paddingsConstDims = {2, 2};
             return opt;
         }},
        {"conv_pads_size_not_4",
         []() {
             PadConvOptions opt;
             opt.convPads = {0, 0, 0};
             return opt;
         }},
        {"paddings_dtype_float",
         []() {
             PadConvOptions opt;
             opt.paddingsAsFloat = true;
             return opt;
         }},
        {"padv3_constant_values_not_float",
         []() {
             PadConvOptions opt;
             opt.withConstantValues = true;
             opt.constantValuesDtype = DT_INT32;
             return opt;
         }},
        {"padv3_constant_values_empty",
         []() {
             PadConvOptions opt;
             opt.withConstantValues = true;
             opt.emptyConstantValues = true;
             return opt;
         }},
        {"padv3_mode_attr_missing",
         []() {
             PadConvOptions opt;
             opt.setPadMode = false;
             return opt;
         }},
        {"pad_unknown_input_shape",
         []() {
             PadConvOptions opt;
             opt.unknownPadInputShape = true;
             return opt;
         }},
        {"conv_without_pad",
         []() {
             PadConvOptions opt;
             opt.skipPadNode = true;
             return opt;
         }},
    };

    for (const auto& point : points) {
        SCOPED_TRACE(point.pointName);
        SetupSoc(SOC_ASCEND950);
        PadConvOptions opt = point.makeOptions();
        opt.graphName = std::string("no_fusion_") + point.pointName;
        auto graph = BuildPadConvGraph(opt);
        TestTotalPass(opt.graphName, graph, CONV_NOT_CHANGED);

        if (opt.skipPadNode) {
            EXPECT_FALSE(GraphChecker::HasNode(graph, opt.padType));
            EXPECT_TRUE(GraphChecker::HasNode(graph, "Conv2D"));
        } else {
            EXPECT_TRUE(GraphChecker::HasNode(graph, opt.padType));
        }
    }
}

// ==========================================================================================
// Soc branch: filter H less than pad H is rejected on non 3510 soc only
// ==========================================================================================
TEST_F(PadConv2dFusionPassTest, filter_h_less_than_pad_reject_on_mc62)
{
    struct Point {
        const char* pointName;
        Format fmt;
        Format filterFmt;
        std::vector<int64_t> filterShape;
        std::vector<int32_t> paddings;
    } const points[] = {
        {"hwcn", FORMAT_NHWC, FORMAT_HWCN, {3, 3, 3, 64}, {0, 0, 3, 3, 3, 3, 0, 0}},
        {"nchw", FORMAT_NCHW, FORMAT_NCHW, {64, 3, 3, 3}, {0, 0, 0, 0, 3, 3, 3, 3}},
    };

    for (const auto& point : points) {
        SCOPED_TRACE(point.pointName);
        PadConvOptions opt;
        opt.graphName = std::string("filter_h_less_than_pad_") + point.pointName;
        opt.fmt = point.fmt;
        opt.padInputFormat = point.fmt;
        opt.filterFmt = point.filterFmt;
        opt.filterShape = point.filterShape;
        opt.paddings = point.paddings;
        if (point.fmt == FORMAT_NCHW) {
            opt.dataShape = {1, 3, 224, 224};
            opt.padOutShape = {1, 3, 230, 230};
            opt.convOutShape = {1, 64, 112, 112};
        }

        SetupSoc(SOC_MC62);
        auto rejectGraph = BuildPadConvGraph(opt);
        TestTotalPass(opt.graphName + "_mc62", rejectGraph, CONV_NOT_CHANGED);
        EXPECT_TRUE(GraphChecker::HasNode(rejectGraph, opt.padType));

        SetupSoc(SOC_ASCEND950);
        auto successGraph = BuildPadConvGraph(opt);
        TestTotalPass(opt.graphName + "_ascend950", successGraph, SUCCESS);
        EXPECT_FALSE(GraphChecker::HasNode(successGraph, opt.padType));
    }
}

TEST_F(PadConv2dFusionPassTest, filter_nhwc_skip_h_check_on_mc62)
{
    SetupSoc(SOC_MC62);
    PadConvOptions opt;
    opt.graphName = "filter_nhwc_skip_h_check";
    opt.filterFmt = FORMAT_NHWC;
    opt.filterShape = {3, 3, 3, 64};
    auto graph = BuildPadConvGraph(opt);
    TestTotalPass(opt.graphName, graph, SUCCESS);

    EXPECT_FALSE(GraphChecker::HasNode(graph, opt.padType));
    ExpectConvPads(graph, "conv2d", {3, 3, 3, 3});
}

TEST_F(PadConv2dFusionPassTest, filter_not_4d_reject_on_mc62)
{
    SetupSoc(SOC_MC62);
    PadConvOptions opt;
    opt.graphName = "filter_not_4d";
    opt.filterShape = {7, 7, 3};
    auto graph = BuildPadConvGraph(opt);
    TestTotalPass(opt.graphName, graph, CONV_NOT_CHANGED);

    EXPECT_TRUE(GraphChecker::HasNode(graph, opt.padType));
}

// ==========================================================================================
// Backward path: slice removed and dx attrs updated on non 3510 soc, rejected on 3510
// ==========================================================================================
TEST_F(PadConv2dFusionPassTest, backward_path_slice_removed_on_mc62)
{
    SetupSoc(SOC_MC62);
    PadConvOptions opt;
    opt.graphName = "backward_path_slice_removed";
    opt.withBackward = true;
    auto graph = BuildPadConvGraph(opt);
    TestTotalPass(opt.graphName, graph, SUCCESS);

    EXPECT_FALSE(GraphChecker::HasNode(graph, "PadV3"));
    EXPECT_FALSE(GraphChecker::HasNode(graph, "Slice"));
    ExpectConvPads(graph, "conv2d", {3, 3, 3, 3});
    ExpectConvPads(graph, "dw", {3, 3, 3, 3});
    ExpectConvPads(graph, "dx", {3, 3, 3, 3});

    GNode dxNode;
    ASSERT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "Conv2DBackpropInputD", dxNode));
    std::string dxPadding;
    EXPECT_TRUE(GraphChecker::GetNodeStringAttr(dxNode, "padding", dxPadding));
    EXPECT_EQ(dxPadding, "SAME");
    std::vector<int64_t> inputSize;
    EXPECT_TRUE(GraphChecker::GetListIntAttr(dxNode, "input_size", inputSize));
    EXPECT_EQ(inputSize, opt.dataShape);
    std::vector<int64_t> dxOutShape;
    EXPECT_TRUE(GraphChecker::GetOriginShape(dxNode, 0, true, dxOutShape));
    EXPECT_EQ(dxOutShape, opt.dataShape);

    GNode reluNode;
    ASSERT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "Relu", reluNode));
    std::string producerType;
    EXPECT_TRUE(GraphChecker::GetInputProducerType(reluNode, 0, producerType));
    EXPECT_EQ(producerType, "Conv2DBackpropInputD");

    GNode dwNode;
    ASSERT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "Conv2DBackpropFilterD", dwNode));
    EXPECT_TRUE(GraphChecker::GetInputProducerType(dwNode, 0, producerType));
    EXPECT_EQ(producerType, "Data");
    std::string dwPadding;
    EXPECT_TRUE(GraphChecker::GetNodeStringAttr(dwNode, "padding", dwPadding));
    EXPECT_EQ(dwPadding, "SAME");
    ExpectFusedCubeFmap(graph, "conv2d", opt.dataShape);
    ExpectFusedCubeFmap(graph, "dw", opt.dataShape);
}

TEST_F(PadConv2dFusionPassTest, backward_path_reject_on_ascend950)
{
    SetupSoc(SOC_ASCEND950);
    PadConvOptions opt;
    opt.graphName = "backward_path_reject";
    opt.withBackward = true;
    auto graph = BuildPadConvGraph(opt);
    TestTotalPass(opt.graphName, graph, CONV_NOT_CHANGED);

    EXPECT_TRUE(GraphChecker::HasNode(graph, "PadV3"));
    EXPECT_TRUE(GraphChecker::HasNode(graph, "Slice"));
}

TEST_F(PadConv2dFusionPassTest, backward_partial_and_ctrl)
{
    struct Point {
        const char* pointName;
        std::string soc;
        std::function<PadConvOptions()> makeOptions;
        Status expectRes;
        bool expectSliceRemain;
    } const points[] = {
        {"dw_only_mc62", SOC_MC62,
         []() {
             PadConvOptions opt;
             opt.withDwOnly = true;
             return opt;
         },
         SUCCESS, false},
        {"dw_only_ascend950", SOC_ASCEND950,
         []() {
             PadConvOptions opt;
             opt.withDwOnly = true;
             return opt;
         },
         CONV_NOT_CHANGED, false},
        {"bn_no_dx_mc62", SOC_MC62,
         []() {
             PadConvOptions opt;
             opt.withBnNoDx = true;
             return opt;
         },
         SUCCESS, false},
        {"dx_multi_consumer_mc62", SOC_MC62,
         []() {
             PadConvOptions opt;
             opt.withBackward = true;
             opt.dxMultiConsumer = true;
             return opt;
         },
         SUCCESS, true},
        {"sliced_mc62", SOC_MC62,
         []() {
             PadConvOptions opt;
             opt.withBackward = true;
             opt.sliceOpType = "SliceD";
             return opt;
         },
         SUCCESS, false},
        {"ctrl_edge_to_dw_mc62", SOC_MC62,
         []() {
             PadConvOptions opt;
             opt.withDwOnly = true;
             opt.ctrlEdgeToDw = true;
             return opt;
         },
         SUCCESS, false},
        {"ctrl_edge_to_dx_mc62", SOC_MC62,
         []() {
             PadConvOptions opt;
             opt.withBackward = true;
             opt.ctrlEdgeToDx = true;
             return opt;
         },
         SUCCESS, false},
    };

    for (const auto& point : points) {
        SCOPED_TRACE(point.pointName);
        SetupSoc(point.soc);
        PadConvOptions opt = point.makeOptions();
        opt.graphName = std::string("backward_partial_") + point.pointName;
        auto graph = BuildPadConvGraph(opt);
        TestTotalPass(opt.graphName, graph, point.expectRes);

        if (point.expectRes == SUCCESS) {
            EXPECT_FALSE(GraphChecker::HasNode(graph, opt.padType));
            ExpectConvPads(graph, "conv2d", {3, 3, 3, 3});
            ExpectConvPads(graph, "dw", {3, 3, 3, 3});
            ExpectFusedCubeFmap(graph, "conv2d", opt.dataShape);
            ExpectFusedCubeFmap(graph, "dw", opt.dataShape);
            if (opt.sliceOpType == "SliceD") {
                EXPECT_FALSE(GraphChecker::HasNode(graph, "SliceD"));
            } else if (point.expectSliceRemain) {
                EXPECT_TRUE(GraphChecker::HasNode(graph, "Slice"));
            } else if (opt.withBackward) {
                EXPECT_FALSE(GraphChecker::HasNode(graph, "Slice"));
            }
        } else {
            EXPECT_TRUE(GraphChecker::HasNode(graph, opt.padType));
        }
    }
}

// ==========================================================================================
// Reentrant: two independent Pad -> Conv2D structures in one graph
// ==========================================================================================
TEST_F(PadConv2dFusionPassTest, reentrant_two_structures)
{
    SetupSoc(SOC_ASCEND950);
    auto graph = BuildTwoPadConvGraph();
    TestTotalPass("reentrant_two_structures", graph, SUCCESS);

    EXPECT_FALSE(GraphChecker::HasNode(graph, "Pad"));
    EXPECT_EQ(GraphChecker::CountNodes(graph, "Conv2D"), 2);
    ExpectConvPads(graph, "conv2d_0", {3, 3, 3, 3});
    ExpectConvPads(graph, "conv2d_1", {1, 1, 2, 2});
}

TEST_F(PadConv2dFusionPassTest, reentrant_cascade)
{
    SetupSoc(SOC_ASCEND950);
    auto graph = BuildCascadePadConvGraph();
    TestTotalPass("reentrant_cascade", graph, SUCCESS);

    EXPECT_FALSE(GraphChecker::HasNode(graph, "Pad"));
    EXPECT_EQ(GraphChecker::CountNodes(graph, "Conv2D"), 2);
    ExpectConvPads(graph, "conv2d_0", {3, 3, 3, 3});
    ExpectConvPads(graph, "conv2d_1", {1, 1, 1, 1});
}

#endif
