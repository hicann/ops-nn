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
#include <vector>

#include "../../../../common/tests/ut/op_graph/test_conv_fusion_pass_framework.h"
#include "../../../op_graph/fusion_pass/a_deformable_conv2d_pass.h"

#include "version/ge-compiler_version.h"
#if GE_COMPILER_VERSION_NUM >= 90000000U

using namespace ge;
using namespace es;
using namespace fe;
using namespace Ops;
using namespace NN;
using namespace Conv;
using namespace ConvFusionUtils;
using namespace DfmConv2dFusion;
using namespace test_conv_fusion_framework;

#define CONV_DEBUG false

class ADeformableConv2dPassTest : public testing::Test {
public:
    // 旧 IR: x(0) filter(1) offsets(2) bias?(3) -> y(0)；attrs strides/pads/dilations/groups/data_format/
    //       deformable_groups/modulated
    NodeConfig MakeDfmCfg(const std::string &name, DataType dt, bool withBias, Format fmt = FORMAT_NCHW,
        const std::vector<int64_t> &xShape = {1, 16, 48, 48},
        const std::vector<int64_t> &fShape = {1, 16, 3, 3},
        const std::vector<int64_t> &yShape = {1, 1, 48, 48})
    {
        NodeConfig cfg(name);
        cfg.opType = DFM_CONV2D.GetString();
        cfg.inputDefs = {{"x", CompliantNodeBuilder::kEsIrInputRequired, ""},
            {"filter", CompliantNodeBuilder::kEsIrInputRequired, ""},
            {"offsets", CompliantNodeBuilder::kEsIrInputRequired, ""},
            {"bias", CompliantNodeBuilder::kEsIrInputOptional, ""}};
        cfg.outputDefs = {{"y", CompliantNodeBuilder::kEsIrOutputRequired, ""}};
        Format fFmt = (fmt == FORMAT_NHWC) ? FORMAT_HWCN : fmt;
        cfg.SetAttr("strides", std::vector<int64_t>{1, 1, 1, 1});
        cfg.SetAttr("pads", std::vector<int64_t>{0, 0, 0, 0});
        cfg.SetAttr("dilations", std::vector<int64_t>{1, 1, 1, 1});
        cfg.SetAttr("groups", int64_t(1));
        cfg.SetAttr("data_format", std::string(fmt == FORMAT_NHWC ? "NHWC" : "NCHW"));
        cfg.SetAttr(DFM_GROUPS.GetString(), int64_t(1));
        cfg.SetAttr("modulated", true);
        TensorInfo xInfo(dt, fmt, xShape, name + "_x");
        xInfo.SetDesc(BuildTensorDesc(dt, fmt, xShape, fmt, xShape));
        TensorInfo fInfo(dt, fFmt, fShape, name + "_filter");
        fInfo.SetDesc(BuildTensorDesc(dt, fFmt, fShape, fFmt, fShape));
        TensorInfo yInfo(dt, fmt, yShape, "y");
        yInfo.SetDesc(BuildTensorDesc(dt, fmt, yShape, fmt, yShape));
        cfg.AddInput(xInfo).AddInput(fInfo).AddInput(xInfo);
        if (withBias) {
            TensorInfo bInfo(dt, FORMAT_ND, {1}, name + "_bias");
            bInfo.SetDesc(BuildTensorDesc(dt, FORMAT_ND, {1}, FORMAT_ND, {1}));
            cfg.AddInput(bInfo);
        }
        cfg.AddOutput(yInfo);
        return cfg;
    }

    GraphPtr BuildSingleDfmGraph(const std::string &name, bool dav3510, DataType dt, bool withBias,
        Format fmt = FORMAT_NCHW)
    {
        TestGraph builder(name.c_str());
        if (dav3510) {
            builder.SetSocAscend950();
        } else {
            builder.SetSoc(SocConfig("Ascend910B", "Ascend910B1"));
        }
        auto cfg = MakeDfmCfg(name, dt, withBias, fmt);
        std::vector<int> autoInputs = {0, 1, 2};
        if (withBias) {
            autoInputs.push_back(3);
        }
        builder.AddNode(cfg, autoInputs).SetOutput(name).Build();
        return builder.Build();
    }

protected:
    static void SetUpTestCase() { std::cout << "ADeformableConv2dPassTest SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "ADeformableConv2dPassTest TearDown" << std::endl; }

    void TestTotalPass(const std::string &passName, GraphPtr &graph, Status expectRes)
    {
        CustomPassContext passContext;
        passContext.SetPassName(passName.c_str());
        ADeformableConv2dPass pass({DFM_CONV2D});

        if (CONV_DEBUG) {
            graph->DumpToFile(Graph::DumpFormat::kOnnx, AscendString((passName + "_before").c_str()));
        }
        auto res = pass.Run(graph, passContext);
        if (CONV_DEBUG) {
            graph->DumpToFile(Graph::DumpFormat::kOnnx, AscendString((passName + "_after").c_str()));
        }
        EXPECT_EQ(res, expectRes);
    }

    GNode GetFirstNodeByType(GraphPtr &graph, const std::string &opType)
    {
        GNode node;
        EXPECT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, opType, node));
        return node;
    }
};

// ==========================================================================================
// 成功主路径：DAV_3510 / 非 DAV_3510 × dtype × bias × format
// ==========================================================================================
TEST_F(ADeformableConv2dPassTest, dfm_conv2d_fusion_success)
{
    struct {
        const char *pointName;
        bool dav3510;
        DataType dt;
        bool withBias;
        Format fmt;
    } const points[] = {
        {"dav3510_fp16_nchw", true, DT_FLOAT16, false, FORMAT_NCHW},
        {"dav3510_fp16_bias_nchw", true, DT_FLOAT16, true, FORMAT_NCHW},
        {"dav3510_fp32_nhwc", true, DT_FLOAT, false, FORMAT_NHWC},
        {"dav3510_bf16_bias_nhwc", true, DT_BF16, true, FORMAT_NHWC},
        {"non_dav3510_fp16_nchw", false, DT_FLOAT16, false, FORMAT_NCHW},
        {"non_dav3510_fp32_bias", false, DT_FLOAT, true, FORMAT_NCHW},
    };

    for (const auto &p : points) {
        SCOPED_TRACE(p.pointName);
        std::string name = std::string("dfm_fusion_success_") + p.pointName;
        auto graph = BuildSingleDfmGraph(name, p.dav3510, p.dt, p.withBias, p.fmt);
        EXPECT_TRUE(GraphChecker::HasNode(graph, DFM_CONV2D.GetString()));
        TestTotalPass(name, graph, SUCCESS);
        EXPECT_TRUE(GraphChecker::HasNode(graph, "DeformableOffsets"));
        EXPECT_TRUE(GraphChecker::HasNode(graph, "Conv2D"));
        EXPECT_EQ(GraphChecker::CountNodes(graph, DFM_CONV2D.GetString()), 0);
    }
}

// ==========================================================================================
// 拒绝融合：入度非法 / filter format 非法 / shape 非 4D
// ==========================================================================================
TEST_F(ADeformableConv2dPassTest, dfm_conv2d_no_fusion)
{
    struct Point {
        const char *pointName;
        std::function<GraphPtr()> build;
    } const points[] = {
        {"bad_filter_format_nd", [this]() {
             NodeConfig cfg = MakeDfmCfg("dfm_no_fusion_bad_filter", DT_FLOAT16, false);
             TensorInfo badFilter(DT_FLOAT16, FORMAT_ND, {1, 16, 3, 3}, "dfm_no_fusion_bad_filter_filter");
             badFilter.SetDesc(BuildTensorDesc(DT_FLOAT16, FORMAT_ND, {1, 16, 3, 3}, FORMAT_ND, {1, 16, 3, 3}));
             cfg.inputs[1] = badFilter;
             TestGraph builder("dfm_no_fusion_bad_filter");
             builder.SetSocAscend950().AddNode(cfg, {0, 1, 2}).SetOutput("dfm_no_fusion_bad_filter").Build();
             return builder.Build();
         }},
        {"x_not_4d", [this]() {
             NodeConfig cfg = MakeDfmCfg("dfm_no_fusion_x_not_4d", DT_FLOAT16, false);
             TensorInfo badX(DT_FLOAT16, FORMAT_NCHW, {1, 16, 48}, "dfm_no_fusion_x_not_4d_x");
             badX.SetDesc(BuildTensorDesc(DT_FLOAT16, FORMAT_NCHW, {1, 16, 48}, FORMAT_NCHW, {1, 16, 48}));
             cfg.inputs[0] = badX;
             TestGraph builder("dfm_no_fusion_x_not_4d");
             builder.SetSocAscend950().AddNode(cfg, {0, 1, 2}).SetOutput("dfm_no_fusion_x_not_4d").Build();
             return builder.Build();
         }},
        {"output_not_4d", [this]() {
             NodeConfig cfg = MakeDfmCfg("dfm_no_fusion_output_not_4d", DT_FLOAT16, false);
             TensorInfo badOut(DT_FLOAT16, FORMAT_NCHW, {1, 1, 48}, "y");
             badOut.SetDesc(BuildTensorDesc(DT_FLOAT16, FORMAT_NCHW, {1, 1, 48}, FORMAT_NCHW, {1, 1, 48}));
             cfg.outputs[0] = badOut;
             TestGraph builder("dfm_no_fusion_output_not_4d");
             builder.SetSocAscend950().AddNode(cfg, {0, 1, 2}).SetOutput("dfm_no_fusion_output_not_4d").Build();
             return builder.Build();
         }},
        {"invalid_input_count", [this]() {
             NodeConfig cfg = MakeDfmCfg("dfm_no_fusion_invalid_input", DT_FLOAT16, false);
             cfg.inputDefs = {{"x", CompliantNodeBuilder::kEsIrInputRequired, ""},
                 {"filter", CompliantNodeBuilder::kEsIrInputRequired, ""},
                 {"bias", CompliantNodeBuilder::kEsIrInputOptional, ""}};
             cfg.inputs.resize(2);
             TestGraph builder("dfm_no_fusion_invalid_input");
             builder.SetSocAscend950().AddNode(cfg, {0, 1}).SetOutput("dfm_no_fusion_invalid_input").Build();
             return builder.Build();
         }},
        {"unknown_ksize", [this]() {
             NodeConfig cfg = MakeDfmCfg("dfm_no_fusion_unknown_ksize", DT_FLOAT16, false);
             TensorInfo unkFilter(DT_FLOAT16, FORMAT_NCHW, {-1, 16, -1, -1}, "dfm_no_fusion_unknown_ksize_filter");
             unkFilter.SetDesc(
                 BuildTensorDesc(DT_FLOAT16, FORMAT_NCHW, {-1, 16, -1, -1}, FORMAT_NCHW, {-1, 16, -1, -1}));
             cfg.inputs[1] = unkFilter;
             TestGraph builder("dfm_no_fusion_unknown_ksize");
             builder.SetSocAscend950().AddNode(cfg, {0, 1, 2}).SetOutput("dfm_no_fusion_unknown_ksize").Build();
             return builder.Build();
         }},
        {"x_format_nd", [this]() {
             NodeConfig cfg = MakeDfmCfg("dfm_no_fusion_x_format_nd", DT_FLOAT16, false);
             TensorInfo badX(DT_FLOAT16, FORMAT_ND, {1, 16, 48, 48}, "dfm_no_fusion_x_format_nd_x");
             badX.SetDesc(BuildTensorDesc(DT_FLOAT16, FORMAT_ND, {1, 16, 48, 48}, FORMAT_ND, {1, 16, 48, 48}));
             cfg.inputs[0] = badX;
             TestGraph builder("dfm_no_fusion_x_format_nd");
             builder.SetSocAscend950().AddNode(cfg, {0, 1, 2}).SetOutput("dfm_no_fusion_x_format_nd").Build();
             return builder.Build();
         }},
    };

    for (const auto &p : points) {
        SCOPED_TRACE(p.pointName);
        auto graph = p.build();
        EXPECT_TRUE(GraphChecker::HasNode(graph, DFM_CONV2D.GetString()));
        TestTotalPass(std::string("dfm_no_fusion_") + p.pointName, graph, GRAPH_NOT_CHANGED);
        EXPECT_TRUE(GraphChecker::HasNode(graph, DFM_CONV2D.GetString()));
        EXPECT_FALSE(GraphChecker::HasNode(graph, "Conv2D"));
    }
}

// ==========================================================================================
// attr/desc 迁移：单图多断言，避免重复建图
// ==========================================================================================
TEST_F(ADeformableConv2dPassTest, dfm_conv2d_attr_and_desc)
{
    auto graph = BuildSingleDfmGraph("dfm_attr_single_graph", true, DT_FLOAT16, true, FORMAT_NCHW);
    TestTotalPass("dfm_attr_single_graph", graph, SUCCESS);

    GNode conv2d = GetFirstNodeByType(graph, "Conv2D");
    GNode dfmOffset = GetFirstNodeByType(graph, "DeformableOffsets");

    std::vector<int64_t> strides;
    ASSERT_EQ(conv2d.GetAttr(AscendString("strides"), strides), GRAPH_SUCCESS);
    EXPECT_EQ(strides, (std::vector<int64_t>{1, 1, 3, 3}));

    std::vector<int64_t> pads;
    ASSERT_EQ(conv2d.GetAttr(AscendString("pads"), pads), GRAPH_SUCCESS);
    EXPECT_EQ(pads, (std::vector<int64_t>{0, 0, 0, 0}));

    std::vector<int64_t> dilations;
    ASSERT_EQ(conv2d.GetAttr(AscendString("dilations"), dilations), GRAPH_SUCCESS);
    EXPECT_EQ(dilations, (std::vector<int64_t>{1, 1, 1, 1}));

    int64_t offsetX = -1;
    ASSERT_EQ(conv2d.GetAttr(AscendString("offset_x"), offsetX), GRAPH_SUCCESS);
    EXPECT_EQ(offsetX, int64_t{0});

    int64_t groups = 0;
    ASSERT_EQ(conv2d.GetAttr(AscendString("groups"), groups), GRAPH_SUCCESS);
    EXPECT_EQ(groups, int64_t{1});

    AscendString dataFormat;
    ASSERT_EQ(conv2d.GetAttr(AscendString("data_format"), dataFormat), GRAPH_SUCCESS);
    EXPECT_STREQ(dataFormat.GetString(), "NCHW");

    TensorDesc filterDesc;
    ASSERT_EQ(conv2d.GetInputDesc(1, filterDesc), GRAPH_SUCCESS);
    EXPECT_EQ(filterDesc.GetDataType(), DT_FLOAT16);

    TensorDesc outDesc;
    ASSERT_EQ(conv2d.GetOutputDesc(0, outDesc), GRAPH_SUCCESS);
    EXPECT_EQ(outDesc.GetDataType(), DT_FLOAT16);
    EXPECT_EQ(conv2d.GetInputsSize(), size_t{3});

    std::vector<int64_t> ksize;
    ASSERT_TRUE(GraphChecker::GetListIntAttr(dfmOffset, "ksize", ksize));
    EXPECT_EQ(ksize, (std::vector<int64_t>{3, 3}));

    auto nonDavGraph = BuildSingleDfmGraph("dfm_attr_non_dav3510", false, DT_FLOAT16, false);
    TestTotalPass("dfm_attr_non_dav3510", nonDavGraph, SUCCESS);
    GNode nonDavConv2d = GetFirstNodeByType(nonDavGraph, "Conv2D");
    int64_t nonDavOffsetX = -1;
    EXPECT_NE(nonDavConv2d.GetAttr(AscendString("offset_x"), nonDavOffsetX), GRAPH_SUCCESS);
}

// ==========================================================================================
// DeformableOffsets input/output desc 与 attrs 透传
// ==========================================================================================
TEST_F(ADeformableConv2dPassTest, dfm_conv2d_deformable_offsets_desc_and_shape)
{
    struct Point {
        const char *pointName;
        std::function<GraphPtr()> build;
        std::function<void(GraphPtr &)> verify;
    } const points[] = {
        {"offsets_input_desc_passthrough_nchw", [this]() {
             NodeConfig cfg = MakeDfmCfg("dfm_offsets_desc_nchw", DT_FLOAT16, false, FORMAT_NCHW,
                 {1, 16, 12, 12}, {1, 16, 3, 3}, {1, 1, 4, 4});
             TestGraph builder("dfm_offsets_desc_nchw");
             builder.SetSocAscend950().AddNode(cfg, {0, 1, 2}).SetOutput("dfm_offsets_desc_nchw").Build();
             return builder.Build();
         }, [](GraphPtr &graph) {
             GNode dfmOffset;
             ASSERT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "DeformableOffsets", dfmOffset));
             GNode conv2d;
             ASSERT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "Conv2D", conv2d));
             TensorDesc xIn;
             TensorDesc offsetsIn;
             TensorDesc offsetOut;
             TensorDesc convFmapIn;
             ASSERT_EQ(dfmOffset.GetInputDesc(0, xIn), GRAPH_SUCCESS);
             ASSERT_EQ(dfmOffset.GetInputDesc(1, offsetsIn), GRAPH_SUCCESS);
             ASSERT_EQ(dfmOffset.GetOutputDesc(0, offsetOut), GRAPH_SUCCESS);
             ASSERT_EQ(conv2d.GetInputDesc(0, convFmapIn), GRAPH_SUCCESS);
             EXPECT_EQ(xIn.GetOriginShape().GetDims(), (std::vector<int64_t>{1, 16, 12, 12}));
             EXPECT_EQ(offsetsIn.GetOriginShape().GetDims(), (std::vector<int64_t>{1, 16, 12, 12}));
             EXPECT_EQ(offsetOut.GetOriginShape().GetDims(), (std::vector<int64_t>{1, 16, 12, 12}));
             EXPECT_EQ(convFmapIn.GetOriginShape().GetDims(), (std::vector<int64_t>{1, 16, 12, 12}));
             EXPECT_EQ(xIn.GetDataType(), DT_FLOAT16);
             EXPECT_EQ(offsetsIn.GetDataType(), DT_FLOAT16);
         }},
        {"offsets_output_deformed_shape_nhwc", [this]() {
             NodeConfig cfg = MakeDfmCfg("dfm_offsets_desc_nhwc", DT_FLOAT16, false, FORMAT_NHWC,
                 {1, 12, 12, 16}, {3, 3, 16, 1}, {1, 4, 4, 1});
             TestGraph builder("dfm_offsets_desc_nhwc");
             builder.SetSocAscend950().AddNode(cfg, {0, 1, 2}).SetOutput("dfm_offsets_desc_nhwc").Build();
             return builder.Build();
         }, [](GraphPtr &graph) {
             GNode dfmOffset;
             GNode conv2d;
             ASSERT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "DeformableOffsets", dfmOffset));
             ASSERT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "Conv2D", conv2d));
             TensorDesc offsetOut;
             ASSERT_EQ(dfmOffset.GetOutputDesc(0, offsetOut), GRAPH_SUCCESS);
             EXPECT_EQ(offsetOut.GetOriginShape().GetDims(), (std::vector<int64_t>{1, 12, 12, 16}));
             std::vector<int64_t> strides;
             ASSERT_EQ(conv2d.GetAttr(AscendString("strides"), strides), GRAPH_SUCCESS);
             EXPECT_EQ(strides, (std::vector<int64_t>{1, 3, 3, 1}));
         }},
        {"offsets_attrs_passthrough", [this]() {
             NodeConfig cfg = MakeDfmCfg("dfm_offsets_attrs", DT_FLOAT16, false);
             cfg.SetAttr("strides", std::vector<int64_t>{1, 2, 2, 1});
             cfg.SetAttr("pads", std::vector<int64_t>{1, 1, 1, 1});
             cfg.SetAttr("dilations", std::vector<int64_t>{1, 2, 2, 1});
             cfg.SetAttr(DFM_GROUPS.GetString(), int64_t(2));
             cfg.SetAttr("modulated", false);
             TestGraph builder("dfm_offsets_attrs");
             builder.SetSocAscend950().AddNode(cfg, {0, 1, 2}).SetOutput("dfm_offsets_attrs").Build();
             return builder.Build();
         }, [](GraphPtr &graph) {
             GNode dfmOffset;
             ASSERT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "DeformableOffsets", dfmOffset));
             std::vector<int64_t> ksize;
             std::vector<int64_t> strides;
             std::vector<int64_t> pads;
             std::vector<int64_t> dilations;
             int64_t dfmGroups = 0;
             bool modulated = true;
             ASSERT_TRUE(GraphChecker::GetListIntAttr(dfmOffset, "ksize", ksize));
             ASSERT_TRUE(GraphChecker::GetListIntAttr(dfmOffset, "strides", strides));
             ASSERT_TRUE(GraphChecker::GetListIntAttr(dfmOffset, "pads", pads));
             ASSERT_TRUE(GraphChecker::GetListIntAttr(dfmOffset, "dilations", dilations));
             ASSERT_EQ(dfmOffset.GetAttr(DFM_GROUPS, dfmGroups), GRAPH_SUCCESS);
             ASSERT_EQ(dfmOffset.GetAttr(AscendString("modulated"), modulated), GRAPH_SUCCESS);
             EXPECT_EQ(ksize, (std::vector<int64_t>{3, 3}));
             EXPECT_EQ(strides, (std::vector<int64_t>{1, 2, 2, 1}));
             EXPECT_EQ(pads, (std::vector<int64_t>{1, 1, 1, 1}));
             EXPECT_EQ(dilations, (std::vector<int64_t>{1, 2, 2, 1}));
             EXPECT_EQ(dfmGroups, int64_t{2});
             EXPECT_FALSE(modulated);
         }},
    };

    for (const auto &p : points) {
        SCOPED_TRACE(p.pointName);
        auto graph = p.build();
        TestTotalPass(std::string("dfm_offsets_") + p.pointName, graph, SUCCESS);
        p.verify(graph);
    }
}

TEST_F(ADeformableConv2dPassTest, dfm_conv2d_dynamic_out_hw_unknown_shape_range)
{
    NodeConfig cfg = MakeDfmCfg("dfm_dynamic_hw", DT_FLOAT16, false, FORMAT_NCHW,
        {-1, 16, -1, -1}, {1, 16, 3, 3}, {-1, 1, -1, -1});
    TensorDesc xDesc = BuildTensorDesc(DT_FLOAT16, FORMAT_NCHW, {-1, 16, -1, -1}, FORMAT_NCHW, {-1, 16, -1, -1});
    xDesc.SetShapeRange({{1, 8}, {16, 16}, {8, 64}, {8, 64}});
    cfg.inputs[0].SetDesc(xDesc);
    TensorDesc yDesc = BuildTensorDesc(DT_FLOAT16, FORMAT_NCHW, {-1, 1, -1, -1}, FORMAT_NCHW, {-1, 1, -1, -1});
    yDesc.SetShapeRange({{1, 8}, {1, 1}, {4, 32}, {4, 32}});
    cfg.outputs[0].SetDesc(yDesc);

    TestGraph builder("dfm_dynamic_hw");
    builder.SetSocAscend950().AddNode(cfg, {0, 1, 2}).SetOutput("dfm_dynamic_hw").Build();
    auto graph = builder.Build();
    TestTotalPass("dfm_dynamic_hw", graph, SUCCESS);

    GNode dfmOffset;
    ASSERT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "DeformableOffsets", dfmOffset));
    TensorDesc offsetOut;
    ASSERT_EQ(dfmOffset.GetOutputDesc(0, offsetOut), GRAPH_SUCCESS);
    auto outDims = offsetOut.GetOriginShape().GetDims();
    ASSERT_EQ(outDims.size(), size_t{4});
    EXPECT_EQ(outDims[2], int64_t{-1});
    EXPECT_EQ(outDims[3], int64_t{-1});
    std::vector<std::pair<int64_t, int64_t>> outRange;
    offsetOut.GetShapeRange(outRange);
    ASSERT_EQ(outRange.size(), size_t{4});
    EXPECT_GE(outRange[2].first, int64_t{12});
    EXPECT_LE(outRange[2].second, int64_t{96});
    EXPECT_GE(outRange[3].first, int64_t{12});
    EXPECT_LE(outRange[3].second, int64_t{96});
}

// ==========================================================================================
// impl_mode: DeformableConv2D 不支持 HF32，融合后 Conv2D 固定为 enable_float_32_execution(0x20)
// ==========================================================================================
TEST_F(ADeformableConv2dPassTest, dfm_conv2d_fused_op_impl_mode_enum_is_0x20)
{
    struct Point {
        const char *pointName;
        std::function<GraphPtr()> build;
    } const points[] = {
        {"parent_has_hf32_0x40", [this]() {
             NodeConfig cfg = MakeDfmCfg("dfm_op_impl_mode_hf32", DT_FLOAT16, false);
             cfg.SetAttr(OP_IMPL_MODE_ENUM.GetString(), int64_t{0x40});
             TestGraph builder("dfm_op_impl_mode_hf32");
             builder.SetSocAscend950().AddNode(cfg, {0, 1, 2}).SetOutput("dfm_op_impl_mode_hf32").Build();
             return builder.Build();
         }},
        {"parent_no_impl_mode_attr", [this]() {
             return BuildSingleDfmGraph("dfm_op_impl_mode_default", true, DT_FLOAT16, false);
         }},
    };

    for (const auto &p : points) {
        SCOPED_TRACE(p.pointName);
        auto graph = p.build();
        TestTotalPass(std::string("dfm_op_impl_mode_") + p.pointName, graph, SUCCESS);
        GNode conv2d = GetFirstNodeByType(graph, "Conv2D");
        int64_t implMode = static_cast<int64_t>(-1);
        ASSERT_EQ(conv2d.GetAttr(OP_IMPL_MODE_ENUM, implMode), GRAPH_SUCCESS);
        EXPECT_EQ(implMode, DfmConv2dFusion::ENABLE_FLOAT32_EXECUTION);
    }
}

// ==========================================================================================
// 拓扑：多节点 / 下游保持 / 空匹配 / 可重入
// ==========================================================================================
TEST_F(ADeformableConv2dPassTest, dfm_conv2d_graph_topology)
{
    struct {
        const char *pointName;
        std::function<void(ADeformableConv2dPassTest&)> run;
    } const points[] = {
        {"multi_dfm_same_graph", [](ADeformableConv2dPassTest &self) {
             auto cfg1 = self.MakeDfmCfg("dfm_1", DT_FLOAT16, false);
             auto cfg2 = self.MakeDfmCfg("dfm_2", DT_FLOAT16, false);
             auto graph = TestGraph("g").SetSocAscend950()
                 .AddNode(cfg1, {0, 1, 2}).AddNode(cfg2, {0, 1, 2})
                 .SetOutput("dfm_1").SetOutput("dfm_2").Build();
             self.TestTotalPass("dfm_topo_multi_same", graph, SUCCESS);
             EXPECT_EQ(GraphChecker::CountNodes(graph, "DeformableOffsets"), 2);
             EXPECT_EQ(GraphChecker::CountNodes(graph, "Conv2D"), 2);
         }},
        {"downstream_relu_preserved", [](ADeformableConv2dPassTest &self) {
             auto cfg = self.MakeDfmCfg("dfm", DT_FLOAT16, false);
             auto graph = TestGraph("g").SetSocAscend950()
                 .AddNode(cfg, {0, 1, 2})
                 .AddRelu(ReluConfig::Basic("relu", DT_FLOAT16))
                 .Connect("dfm", 0, "relu", 0)
                 .SetOutput("relu").Build();
             self.TestTotalPass("dfm_topo_downstream", graph, SUCCESS);
             EXPECT_TRUE(GraphChecker::HasNode(graph, "Relu"));
         }},
        {"graph_without_dfm", [](ADeformableConv2dPassTest &) {
             auto graph = TestGraph("g").SetSocAscend950()
                 .AddRelu(ReluConfig::Basic("relu", DT_FLOAT16), true)
                 .SetOutput("relu").Build();
             CustomPassContext passContext;
             passContext.SetPassName("dfm_topo_no_match");
             ADeformableConv2dPass pass({DFM_CONV2D});
             EXPECT_EQ(pass.Run(graph, passContext), GRAPH_NOT_CHANGED);
             EXPECT_FALSE(GraphChecker::HasNode(graph, "Conv2D"));
         }},
        {"reentrant_init_member", [](ADeformableConv2dPassTest &self) {
             auto graph1 = self.BuildSingleDfmGraph("dfm_reentrant_1", true, DT_FLOAT16, true);
             self.TestTotalPass("dfm_reentrant_1", graph1, SUCCESS);
             auto graph2 = self.BuildSingleDfmGraph("dfm_reentrant_2", true, DT_FLOAT16, false);
             self.TestTotalPass("dfm_reentrant_2", graph2, SUCCESS);
             EXPECT_EQ(GraphChecker::CountNodes(graph1, "Conv2D"), 1);
             EXPECT_EQ(GraphChecker::CountNodes(graph2, "Conv2D"), 1);
         }},
        {"fusion_graph_edge_topology", [](ADeformableConv2dPassTest &self) {
             auto cfg = self.MakeDfmCfg("dfm_edge", DT_FLOAT16, true);
             auto graph = TestGraph("g").SetSocAscend950()
                 .AddNode(cfg, {0, 1, 2, 3}).SetOutput("dfm_edge").Build();
             self.TestTotalPass("dfm_edge_topology", graph, SUCCESS);
             GNode dfmOffset;
             GNode conv2d;
             ASSERT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "DeformableOffsets", dfmOffset));
             ASSERT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "Conv2D", conv2d));
             std::string convFmapProducer;
             std::string convFilterProducer;
             std::string convBiasProducer;
             ASSERT_TRUE(GraphChecker::GetInputProducerType(conv2d, 0, convFmapProducer));
             ASSERT_TRUE(GraphChecker::GetInputProducerType(conv2d, 1, convFilterProducer));
             ASSERT_TRUE(GraphChecker::GetInputProducerType(conv2d, 2, convBiasProducer));
             EXPECT_EQ(convFmapProducer, "DeformableOffsets");
             EXPECT_NE(convFilterProducer, "DeformableOffsets");
             EXPECT_NE(convBiasProducer, "DeformableOffsets");
             EXPECT_TRUE(GraphChecker::HasOutEdgeTo(dfmOffset, 0, conv2d, 0));
         }},
        {"multi_dfm_mixed_eligibility", [](ADeformableConv2dPassTest &self) {
             auto okCfg = self.MakeDfmCfg("dfm_ok", DT_FLOAT16, false);
             NodeConfig badCfg = self.MakeDfmCfg("dfm_bad", DT_FLOAT16, false);
             TensorInfo badFilter(DT_FLOAT16, FORMAT_ND, {1, 16, 3, 3}, "dfm_bad_filter");
             badFilter.SetDesc(BuildTensorDesc(DT_FLOAT16, FORMAT_ND, {1, 16, 3, 3}, FORMAT_ND, {1, 16, 3, 3}));
             badCfg.inputs[1] = badFilter;
             auto graph = TestGraph("g").SetSocAscend950()
                 .AddNode(okCfg, {0, 1, 2}).AddNode(badCfg, {0, 1, 2})
                 .SetOutput("dfm_ok").SetOutput("dfm_bad").Build();
             CustomPassContext passContext;
             passContext.SetPassName("dfm_mixed_eligibility");
             ADeformableConv2dPass pass({DFM_CONV2D});
             EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);
             EXPECT_EQ(GraphChecker::CountNodes(graph, "DeformableOffsets"), 1);
             EXPECT_EQ(GraphChecker::CountNodes(graph, "Conv2D"), 1);
             EXPECT_EQ(GraphChecker::CountNodes(graph, DFM_CONV2D.GetString()), 1);
         }},
    };

    for (const auto &p : points) {
        SCOPED_TRACE(p.pointName);
        p.run(*this);
    }
}

#endif
