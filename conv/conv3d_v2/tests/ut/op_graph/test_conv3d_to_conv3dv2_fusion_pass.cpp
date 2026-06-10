/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#if GE_COMPILER_VERSION_NUM >= 90000000U
#include <functional>
#include <vector>

#include "../../../../common/tests/ut/op_graph/test_conv_fusion_pass_framework.h"
#include "../../../op_graph/fusion_pass/conv3d_to_conv3dv2_fusion_pass.h"

using namespace ge;
using namespace es;
using namespace fe;
using namespace Ops;
using namespace NN;
using namespace Conv;
using namespace ConvFusionUtils;
using namespace test_conv_fusion_framework;

#define CONV_DEBUG false

enum class SocProfile {
    Arch35,
    NonArch35,
    Reject310B1,
    NoCapability,
};

class Conv3dToConv3dV2FusionPassTest : public testing::Test {
public:
    void ApplySoc(SocProfile profile)
    {
        switch (profile) {
            case SocProfile::Arch35:
                SetSocRaw("Ascend950", {{"Intrinsic_data_move_out2l1_dn2nz", {"bfloat16"}}});
                break;
            case SocProfile::NonArch35:
                SetSocRaw("Ascend910B4", {{"Intrinsic_fix_pipe_l0c2out", {"bfloat16"}}});
                break;
            case SocProfile::Reject310B1:
                SetSocRaw("Ascend310B1", {
                    {"Intrinsic_fix_pipe_l0c2out", {"bfloat16"}},
                    {"Intrinsic_fix_pipe_l0c2ub", {"bfloat16"}}
                });
                break;
            case SocProfile::NoCapability:
                SetSocRaw("AscendNoCap", {});
                break;
        }
    }

    Conv3DConfig MakeConvCfg(DataType dtype, bool hasBias = false, DataType biasDtype = DT_FLOAT16)
    {
        Conv3DConfig convCfg = Conv3DConfig::Basic("Conv3D", dtype, FORMAT_NCDHW);
        if (hasBias) {
            convCfg.WithBias(biasDtype);
        }
        return convCfg;
    }

    GraphPtr BuildSingleConvGraph(const char *graphName, SocProfile soc, const Conv3DConfig &convCfg)
    {
        ApplySoc(soc);
        TestGraph builder(graphName);
        auto graph = builder.AddConv3D(convCfg).SetOutput(convCfg.name).Build();
        FixOutputDtype(builder, convCfg.name, convCfg.inputs[0].dtype);
        return graph;
    }

protected:
    static void SetUpTestCase()
    {
        std::cout << "Conv3dToConv3dV2FusionPassTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "Conv3dToConv3dV2FusionPassTest TearDown" << std::endl;
    }

    void SetSocRaw(const std::string &socName,
        const std::map<std::string, std::vector<std::string>> &intrinsicDtypeMap)
    {
        PlatformInfo platformInfo;
        platformInfo.str_info.short_soc_version = socName;
        OptionalInfo optionalInfo;
        optionalInfo.soc_version = socName;
        for (const auto &[k, v] : intrinsicDtypeMap) {
            platformInfo.ai_core_intrinsic_dtype_map[k] = v;
        }
        PlatformInfoManager::Instance().platform_info_map_[socName] = platformInfo;
        PlatformInfoManager::Instance().SetOptionalCompilationInfo(optionalInfo);
    }

    // Conv3DConfig::Basic hardcodes output to DT_INT32; override to match input dtype.
    void FixOutputDtype(TestGraph &testGraphBuilder, const std::string &nodeName, DataType dtype)
    {
        testGraphBuilder.UpdateNodeOutputDesc(nodeName, OUTPUT_INDEX, dtype, Format::FORMAT_NCDHW);
    }

    void TestTotalPass(const std::string &passName, GraphPtr &graph, Status expectRes)
    {
        CustomPassContext passContext;
        passContext.SetPassName(passName.c_str());
        Conv3dToConv3dV2FusionPass pass({AscendString("Conv3D")});

        if (CONV_DEBUG) {
            graph->DumpToFile(Graph::DumpFormat::kOnnx, AscendString((passName + "_before").c_str()));
        }
        auto res = pass.Run(graph, passContext);
        if (CONV_DEBUG) {
            graph->DumpToFile(Graph::DumpFormat::kOnnx, AscendString((passName + "_after").c_str()));
        }
        EXPECT_EQ(res, expectRes);
        if (expectRes == SUCCESS) {
            EXPECT_TRUE(GraphChecker::HasNode(graph, "Conv3DV2"));
            EXPECT_EQ(GraphChecker::CountNodes(graph, "Conv3D"), 0);
        } else if (expectRes == GRAPH_NOT_CHANGED) {
            EXPECT_TRUE(GraphChecker::HasNode(graph, "Conv3D"));
            EXPECT_FALSE(GraphChecker::HasNode(graph, "Conv3DV2"));
        }
    }

    GNode GetFirstConv3dV2Node(GraphPtr &graph)
    {
        GNode fused;
        EXPECT_TRUE(GraphChecker::FindFirstNodeByOpType(graph, "Conv3DV2", fused));
        return fused;
    }

    void SetConv3DDynamicShapes(TestGraph &testGraphBuilder, const std::string &nodeName, DataType dtype,
        Format format, bool dynamicFmap, bool dynamicFilter, bool dynamicBias = false)
    {
        const std::vector<int64_t> staticShape = {1, 16, 16, 244, 244};
        const std::vector<int64_t> dynamicShape = {-1, -1, -1, -1, -1};
        const auto &fmapShape = dynamicFmap ? dynamicShape : staticShape;
        const auto &filterShape = dynamicFilter ? dynamicShape : staticShape;
        testGraphBuilder.UpdateNodeInputDescEx(nodeName, INPUT_FMAP_INDEX,
            BuildTensorDesc(dtype, format, fmapShape, format, fmapShape));
        testGraphBuilder.UpdateNodeInputDescEx(nodeName, INPUT_FILTER_INDEX,
            BuildTensorDesc(dtype, format, filterShape, format, filterShape));
        if (dynamicBias) {
            testGraphBuilder.UpdateNodeInputDescEx(nodeName, INPUT_BIAS_INDEX,
                BuildTensorDesc(dtype, FORMAT_ND, {-1}, FORMAT_ND, {-1}));
        }
    }

    GraphPtr BuildFp16Conv3DWithIfmrAdd(TestGraph &testGraphBuilder)
    {
        const std::vector<int64_t> shape = {1, 16, 16, 244, 244};
        return testGraphBuilder
            .AddConv3D(Conv3DConfig::Basic("Conv3D", DT_FLOAT16, FORMAT_NCDHW))
            .AddIFMR(IFMRConfig::Basic("IFMR", DT_FLOAT16))
            .AddAdd(AddConfig::Basic("Add", DT_FLOAT16, FORMAT_NCDHW, shape))
            .Connect("Conv3D", 0, "Add", 0)
            .Connect("IFMR", 0, "Add", 1)
            .SetOutput("Add")
            .Build();
    }

    GraphPtr BuildConv3DWithTransData(TestGraph &testGraphBuilder, int64_t originC, SocProfile soc)
    {
        ApplySoc(soc);
        const std::vector<int64_t> fmapShape = {1, 32, 1, 240, 352, 16};
        const std::vector<int64_t> filterShape = {27, 1, 16, 16};
        const std::vector<int64_t> outShape = {1, 32, 1, 240, 352, 16};
        auto convConfig = Conv3DConfig::Basic("Conv3D", DT_FLOAT16, FORMAT_NDC1HWC0, fmapShape, filterShape, outShape);
        convConfig.SetAttr("data_format", std::string("NDHWC"));

        auto graph = testGraphBuilder
            .AddTransData(TransDataConfig::NdhwcToNdc1hwc0("TransData", DT_FLOAT16, originC))
            .AddConv3D(convConfig, false, true, false)
            .Connect("TransData", 0, "Conv3D", 0)
            .SetOutput("Conv3D")
            .Build();
        FixOutputDtype(testGraphBuilder, "Conv3D", DT_FLOAT16);
        return graph;
    }
};

// ==========================================================================================
// MeetRequirements dtype matrix success: arch35 / non-arch35 × dtype × bias
// ==========================================================================================
TEST_F(Conv3dToConv3dV2FusionPassTest, conv3d_to_conv3dv2_fusion_success)
{
    struct {
        const char *pointName;
        SocProfile soc;
        DataType dtype;
        bool hasBias;
        DataType biasDtype;
    } const points[] = {
        {"dav3510_fp16",          SocProfile::Arch35,     DT_FLOAT16,  false, DT_FLOAT16},
        {"dav3510_fp16_bias",     SocProfile::Arch35,     DT_FLOAT16,  true,  DT_FLOAT16},
        {"dav3510_fp32",          SocProfile::Arch35,     DT_FLOAT,    false, DT_FLOAT},
        {"dav3510_bf16",          SocProfile::Arch35,     DT_BF16,     false, DT_BF16},
        {"dav3510_hf8",           SocProfile::Arch35,     DT_HIFLOAT8, false, DT_FLOAT},
        {"dav3510_hf8_bias",      SocProfile::Arch35,     DT_HIFLOAT8, true,  DT_FLOAT},
        {"nonarch35_fp32",        SocProfile::NonArch35,  DT_FLOAT,    false, DT_FLOAT},
        {"nonarch35_fp16",        SocProfile::NonArch35,  DT_FLOAT16,  false, DT_FLOAT16},
        {"nonarch35_bf16",        SocProfile::NonArch35,  DT_BF16,     false, DT_BF16},
        {"nonarch35_fp16_bias",   SocProfile::NonArch35,  DT_FLOAT16,  true,  DT_FLOAT16},
        {"nonarch35_bf16_bias",   SocProfile::NonArch35,  DT_BF16,     true,  DT_FLOAT},
    };

    for (const auto &p : points) {
        SCOPED_TRACE(p.pointName);
        std::string name = std::string("conv3d_to_conv3dv2_fusion_success_") + p.pointName;
        auto graph = BuildSingleConvGraph(name.c_str(), p.soc, MakeConvCfg(p.dtype, p.hasBias, p.biasDtype));
        EXPECT_TRUE(GraphChecker::HasNode(graph, "Conv3D"));
        TestTotalPass(name, graph, SUCCESS);
    }
}

// ==========================================================================================
// MeetRequirements rejection: unsupported dtype / SOC / static shape / topology limitation
// ==========================================================================================
TEST_F(Conv3dToConv3dV2FusionPassTest, conv3d_to_conv3dv2_no_fusion)
{
    struct Point {
        const char *pointName;
        std::function<GraphPtr()> build;
    } const points[] = {
        {"unsupported_dtype", [this]() {
             return BuildSingleConvGraph("conv3d_to_conv3dv2_no_fusion_unsupported_dtype", SocProfile::Arch35,
                 Conv3DConfig::Basic("Conv3D", DT_INT8, FORMAT_NCDHW));
         }},
        {"bad_bias_dtype", [this]() {
             return BuildSingleConvGraph("conv3d_to_conv3dv2_no_fusion_bad_bias_dtype", SocProfile::Arch35,
                 Conv3DConfig::Basic("Conv3D", DT_FLOAT16, FORMAT_NCDHW).WithBias(DT_INT32));
         }},
        {"reject_soc_310b1", [this]() {
             return BuildSingleConvGraph("conv3d_to_conv3dv2_no_fusion_reject_soc_310b1", SocProfile::Reject310B1,
                 MakeConvCfg(DT_FLOAT16));
         }},
        {"soc_no_l0c2out", [this]() {
             return BuildSingleConvGraph("conv3d_to_conv3dv2_no_fusion_soc_no_l0c2out", SocProfile::NoCapability,
                 MakeConvCfg(DT_FLOAT16));
         }},
        {"nonarch35_bf16_bad_bias", [this]() {
             return BuildSingleConvGraph("conv3d_to_conv3dv2_no_fusion_nonarch35_bf16_bad_bias",
                 SocProfile::NonArch35, Conv3DConfig::Basic("Conv3D", DT_BF16).WithBias(DT_BF16));
         }},
        {"nonarch35_bf16_dynamic", [this]() {
             ApplySoc(SocProfile::NonArch35);
             TestGraph builder("conv3d_to_conv3dv2_no_fusion_nonarch35_bf16_dynamic");
             auto graph = builder.AddConv3D(Conv3DConfig::Basic("Conv3D", DT_BF16)).SetOutput("Conv3D").Build();
             FixOutputDtype(builder, "Conv3D", DT_BF16);
             SetConv3DDynamicShapes(builder, "Conv3D", DT_BF16, FORMAT_NCDHW, true, false);
             return graph;
         }},
        {"nonarch35_fp16_dynamic", [this]() {
             ApplySoc(SocProfile::NonArch35);
             TestGraph builder("conv3d_to_conv3dv2_no_fusion_nonarch35_fp16_dynamic");
             auto graph = builder.AddConv3D(Conv3DConfig::Basic("Conv3D", DT_FLOAT16)).SetOutput("Conv3D").Build();
             FixOutputDtype(builder, "Conv3D", DT_FLOAT16);
             SetConv3DDynamicShapes(builder, "Conv3D", DT_FLOAT16, FORMAT_NCDHW, true, false);
             return graph;
         }},
        {"nonarch35_fp32_dynamic", [this]() {
             ApplySoc(SocProfile::NonArch35);
             TestGraph builder("conv3d_to_conv3dv2_no_fusion_nonarch35_fp32_dynamic");
             auto graph = builder.AddConv3D(Conv3DConfig::Basic("Conv3D", DT_FLOAT)).SetOutput("Conv3D").Build();
             FixOutputDtype(builder, "Conv3D", DT_FLOAT);
             SetConv3DDynamicShapes(builder, "Conv3D", DT_FLOAT, FORMAT_NCDHW, true, true);
             return graph;
         }},
        {"nonarch35_fp16_fixpipe", [this]() {
             ApplySoc(SocProfile::NonArch35);
             TestGraph builder("conv3d_to_conv3dv2_no_fusion_nonarch35_fp16_fixpipe");
             const std::vector<int64_t> outShape = {1, 16, 16, 244, 244};
             auto graph = builder
                 .AddConv3D(Conv3DConfig::Basic("Conv3D", DT_FLOAT16))
                 .AddPostCube(PostCubeConfig::Basic("FixPipe", DT_FLOAT16, FORMAT_NCDHW, outShape), false)
                 .Connect("Conv3D", 0, "FixPipe", 0)
                 .SetOutput("FixPipe")
                 .Build();
             FixOutputDtype(builder, "Conv3D", DT_FLOAT16);
             return graph;
         }},
        {"nonarch35_bf16_fixpipe", [this]() {
             ApplySoc(SocProfile::NonArch35);
             TestGraph builder("conv3d_to_conv3dv2_no_fusion_nonarch35_bf16_fixpipe");
             const std::vector<int64_t> outShape = {1, 16, 16, 244, 244};
             auto graph = builder
                 .AddConv3D(Conv3DConfig::Basic("Conv3D", DT_BF16))
                 .AddPostCube(PostCubeConfig::Basic("FixPipe", DT_BF16, FORMAT_NCDHW, outShape), false)
                 .Connect("Conv3D", 0, "FixPipe", 0)
                 .SetOutput("FixPipe")
                 .Build();
             FixOutputDtype(builder, "Conv3D", DT_BF16);
             return graph;
         }},
        {"nonarch35_fp32_fixpipe", [this]() {
             ApplySoc(SocProfile::NonArch35);
             TestGraph builder("conv3d_to_conv3dv2_no_fusion_nonarch35_fp32_fixpipe");
             const std::vector<int64_t> outShape = {1, 16, 16, 244, 244};
             auto graph = builder
                 .AddConv3D(Conv3DConfig::Basic("Conv3D", DT_FLOAT))
                 .AddPostCube(PostCubeConfig::Basic("FixPipe", DT_FLOAT, FORMAT_NCDHW, outShape), false)
                 .Connect("Conv3D", 0, "FixPipe", 0)
                 .SetOutput("FixPipe")
                 .Build();
             FixOutputDtype(builder, "Conv3D", DT_FLOAT);
             return graph;
         }},
        {"nonarch35_fp16_transdata_c16", [this]() {
             TestGraph builder("conv3d_to_conv3dv2_no_fusion_nonarch35_fp16_transdata_c16");
             return BuildConv3DWithTransData(builder, 16, SocProfile::NonArch35);
         }},
        {"nonarch35_fp16_ifmr", [this]() {
             ApplySoc(SocProfile::NonArch35);
             TestGraph builder("conv3d_to_conv3dv2_no_fusion_nonarch35_fp16_ifmr");
             auto graph = BuildFp16Conv3DWithIfmrAdd(builder);
             FixOutputDtype(builder, "Conv3D", DT_FLOAT16);
             return graph;
         }},
        {"dtype_mismatch_filter", [this]() {
             ApplySoc(SocProfile::NonArch35);
             TestGraph builder("conv3d_to_conv3dv2_no_fusion_dtype_mismatch_filter");
             auto graph = builder.AddConv3D(Conv3DConfig::Basic("Conv3D", DT_FLOAT16)).SetOutput("Conv3D").Build();
             FixOutputDtype(builder, "Conv3D", DT_FLOAT16);
             builder.UpdateNodeInputDescEx("Conv3D", INPUT_FILTER_INDEX,
                 BuildTensorDesc(DT_FLOAT, FORMAT_NCDHW, {3, 16, 3, 3, 3}));
             return graph;
         }},
        {"output_dtype_mismatch", [this]() {
             ApplySoc(SocProfile::Arch35);
             TestGraph builder("conv3d_to_conv3dv2_no_fusion_output_dtype_mismatch");
             return builder.AddConv3D(Conv3DConfig::Basic("Conv3D", DT_FLOAT16, FORMAT_NCDHW))
                 .SetOutput("Conv3D")
                 .Build();
         }},
        {"hifloat8_bad_output", [this]() {
             ApplySoc(SocProfile::Arch35);
             TestGraph builder("conv3d_to_conv3dv2_no_fusion_hifloat8_bad_output");
             auto graph = builder.AddConv3D(Conv3DConfig::Basic("Conv3D", DT_HIFLOAT8, FORMAT_NCDHW))
                 .SetOutput("Conv3D")
                 .Build();
             FixOutputDtype(builder, "Conv3D", DT_FLOAT);
             return graph;
         }},
    };

    for (const auto &p : points) {
        SCOPED_TRACE(p.pointName);
        auto graph = p.build();
        EXPECT_TRUE(GraphChecker::HasNode(graph, "Conv3D"));
        TestTotalPass(std::string("conv3d_to_conv3dv2_no_fusion_") + p.pointName, graph, GRAPH_NOT_CHANGED);
    }
}

// ==========================================================================================
// Replacement attr/desc migration and GetConvBaseAttr derivation
// ==========================================================================================
TEST_F(Conv3dToConv3dV2FusionPassTest, conv3d_to_conv3dv2_replacement_attr_and_desc)
{
    struct Point {
        const char *pointName;
        std::function<GraphPtr()> build;
        std::function<void(GNode &)> verify;
    } const points[] = {
        {"fused_op_impl_mode_enum_is_default_0x1", [this]() {
             auto config = Conv3DConfig::Basic("Conv3D", DT_FLOAT, FORMAT_NCDHW);
             config.SetAttr("_op_impl_mode_enum", int64_t(0x40));
             return BuildSingleConvGraph("conv3d_to_conv3dv2_fused_op_impl_mode_enum_is_default_0x1",
                 SocProfile::Arch35, config);
         }, [](GNode &fused) {
             int64_t implMode = static_cast<int64_t>(-1);
             ASSERT_EQ(fused.GetAttr(AscendString("_op_impl_mode_enum"), implMode), GRAPH_SUCCESS);
             EXPECT_EQ(implMode, int64_t{0x1});
         }},
        {"pad_mode_from_padding_valid", [this]() {
             auto config = Conv3DConfig::Basic("Conv3D", DT_FLOAT, FORMAT_NCDHW);
             config.SetAttr("padding", std::string("VALID"));
             return BuildSingleConvGraph("conv3d_to_conv3dv2_pad_mode_from_padding_valid", SocProfile::Arch35, config);
         }, [](GNode &fused) {
             AscendString padMode;
             ASSERT_EQ(fused.GetAttr(AscendString("pad_mode"), padMode), GRAPH_SUCCESS);
             EXPECT_STREQ(padMode.GetString(), "VALID");
         }},
        {"pad_mode_from_padding_explicit", [this]() {
             auto config = Conv3DConfig::Basic("Conv3D", DT_FLOAT, FORMAT_NCDHW);
             config.SetAttr("padding", std::string("EXPLICIT"));
             return BuildSingleConvGraph("conv3d_to_conv3dv2_pad_mode_from_padding_explicit",
                 SocProfile::Arch35, config);
         }, [](GNode &fused) {
             AscendString padMode;
             if (fused.GetAttr(AscendString("pad_mode"), padMode) == GRAPH_SUCCESS) {
                 EXPECT_STREQ(padMode.GetString(), "SPECIFIC");
             }
         }},
        {"pad_mode_from_auto_pad", [this]() {
             auto config = Conv3DConfig::Basic("Conv3D", DT_FLOAT, FORMAT_NCDHW);
             config.SetAttr("auto_pad", std::string("SAME_UPPER"));
             return BuildSingleConvGraph("conv3d_to_conv3dv2_pad_mode_from_auto_pad", SocProfile::Arch35, config);
         }, [](GNode &fused) {
             AscendString padMode;
             ASSERT_EQ(fused.GetAttr(AscendString("pad_mode"), padMode), GRAPH_SUCCESS);
             EXPECT_STREQ(padMode.GetString(), "SAME_UPPER");
         }},
        {"enable_hf32_fp32_impl_0x40", [this]() {
             auto config = Conv3DConfig::Basic("Conv3D", DT_FLOAT, FORMAT_NCDHW);
             config.SetAttr("_op_impl_mode_enum", int64_t(0x40));
             return BuildSingleConvGraph("conv3d_to_conv3dv2_enable_hf32_fp32_impl_0x40", SocProfile::Arch35, config);
         }, [](GNode &fused) {
             bool enableHf32 = false;
             ASSERT_EQ(fused.GetAttr(AscendString("enable_hf32"), enableHf32), GRAPH_SUCCESS);
             EXPECT_TRUE(enableHf32);
         }},
        {"enable_hf32_false_nonarch35", [this]() {
             auto config = Conv3DConfig::Basic("Conv3D", DT_FLOAT);
             config.SetAttr("_op_impl_mode_enum", int64_t(0x40));
             return BuildSingleConvGraph("conv3d_to_conv3dv2_enable_hf32_false_nonarch35",
                 SocProfile::NonArch35, config);
         }, [](GNode &fused) {
             bool enableHf32 = true;
             if (GraphChecker::GetNodeBoolAttr(fused, "enable_hf32", enableHf32)) {
                 EXPECT_FALSE(enableHf32);
             }
         }},
        {"enable_hf32_false_fp16", [this]() {
             auto config = Conv3DConfig::Basic("Conv3D", DT_FLOAT16, FORMAT_NCDHW);
             config.SetAttr("_op_impl_mode_enum", int64_t(0x40));
             return BuildSingleConvGraph("conv3d_to_conv3dv2_enable_hf32_false_fp16", SocProfile::Arch35, config);
         }, nullptr},
        {"attr_strides_pads_dilations_groups", [this]() {
             auto config = Conv3DConfig::Basic("Conv3D", DT_FLOAT16, FORMAT_NCDHW);
             config.SetAttr("strides", std::vector<int64_t>{2, 2, 2, 2, 2});
             config.SetAttr("pads", std::vector<int64_t>{2, 2, 2, 2, 2, 2});
             config.SetAttr("dilations", std::vector<int64_t>{2, 2, 2, 2, 2});
             config.SetAttr("groups", int64_t(2));
             return BuildSingleConvGraph("conv3d_to_conv3dv2_attr_strides_pads_dilations_groups",
                 SocProfile::Arch35, config);
         }, [](GNode &fused) {
             std::vector<int64_t> strides, pads, dilations;
             int64_t groups = 0;
             ASSERT_EQ(fused.GetAttr(AscendString("strides"), strides), GRAPH_SUCCESS);
             ASSERT_EQ(fused.GetAttr(AscendString("pads"), pads), GRAPH_SUCCESS);
             ASSERT_EQ(fused.GetAttr(AscendString("dilations"), dilations), GRAPH_SUCCESS);
             ASSERT_EQ(fused.GetAttr(AscendString("groups"), groups), GRAPH_SUCCESS);
             EXPECT_EQ(strides, (std::vector<int64_t>{2, 2, 2, 2, 2}));
             EXPECT_EQ(pads, (std::vector<int64_t>{2, 2, 2, 2, 2, 2}));
             EXPECT_EQ(dilations, (std::vector<int64_t>{2, 2, 2, 2, 2}));
             EXPECT_EQ(groups, int64_t{2});
         }},
        {"attr_data_format_ndhwc", [this]() {
             auto config = Conv3DConfig::Basic("Conv3D", DT_FLOAT16, FORMAT_NDHWC);
             config.SetAttr("data_format", std::string("NDHWC"));
             return BuildSingleConvGraph("conv3d_to_conv3dv2_attr_data_format_ndhwc", SocProfile::Arch35, config);
         }, [](GNode &fused) {
             AscendString dataFormat;
             ASSERT_EQ(fused.GetAttr(AscendString("data_format"), dataFormat), GRAPH_SUCCESS);
             EXPECT_STREQ(dataFormat.GetString(), "NDHWC");
         }},
        {"desc_input_output_preserved", [this]() {
             return BuildSingleConvGraph("conv3d_to_conv3dv2_desc_input_output_preserved", SocProfile::Arch35,
                 Conv3DConfig::Basic("Conv3D", DT_FLOAT16, FORMAT_NCDHW,
                     {1, 16, 16, 240, 352}, {3, 16, 3, 3, 3}, {1, 3, 16, 240, 352}));
         }, [](GNode &fused) {
             TensorDesc fmapDesc, filterDesc, outDesc;
             ASSERT_EQ(fused.GetInputDesc(0, fmapDesc), GRAPH_SUCCESS);
             ASSERT_EQ(fused.GetInputDesc(1, filterDesc), GRAPH_SUCCESS);
             ASSERT_EQ(fused.GetOutputDesc(0, outDesc), GRAPH_SUCCESS);
             EXPECT_EQ(fmapDesc.GetDataType(), DT_FLOAT16);
             EXPECT_EQ(filterDesc.GetDataType(), DT_FLOAT16);
             EXPECT_EQ(outDesc.GetDataType(), DT_FLOAT16);
             EXPECT_EQ(fmapDesc.GetFormat(), FORMAT_NCDHW);
             EXPECT_EQ(outDesc.GetFormat(), FORMAT_NCDHW);
         }},
        {"bias_input_absent", [this]() {
             return BuildSingleConvGraph("conv3d_to_conv3dv2_bias_input_absent", SocProfile::Arch35,
                 MakeConvCfg(DT_FLOAT16));
         }, [](GNode &fused) { EXPECT_EQ(fused.GetInputsSize(), size_t{2}); }},
        {"bias_input_present", [this]() {
             return BuildSingleConvGraph("conv3d_to_conv3dv2_bias_input_present", SocProfile::Arch35,
                 MakeConvCfg(DT_FLOAT16, true, DT_FLOAT16));
         }, [](GNode &fused) {
             TensorDesc biasDesc;
             EXPECT_EQ(fused.GetInputsSize(), size_t{3});
             ASSERT_EQ(fused.GetInputDesc(2, biasDesc), GRAPH_SUCCESS);
             EXPECT_EQ(biasDesc.GetDataType(), DT_FLOAT16);
             EXPECT_EQ(biasDesc.GetFormat(), FORMAT_ND);
         }},
        {"offset_x_nonzero", [this]() {
             auto config = Conv3DConfig::Basic("Conv3D", DT_FLOAT16, FORMAT_NCDHW);
             config.SetAttr("offset_x", int64_t(1));
             return BuildSingleConvGraph("conv3d_to_conv3dv2_offset_x_nonzero", SocProfile::Arch35, config);
         }, [](GNode &fused) {
             int64_t offsetX = 0;
             ASSERT_EQ(fused.GetAttr(AscendString("offset_x"), offsetX), GRAPH_SUCCESS);
             EXPECT_EQ(offsetX, int64_t{1});
         }},
    };

    for (const auto &p : points) {
        SCOPED_TRACE(p.pointName);
        auto graph = p.build();
        TestTotalPass(std::string("conv3d_to_conv3dv2_replacement_attr_and_desc_") + p.pointName, graph, SUCCESS);
        if (p.verify) {
            GNode fused = GetFirstConv3dV2Node(graph);
            p.verify(fused);
        }
    }
}

// ==========================================================================================
// Graph topology: dynamic shape, multi-node, partial fusion, downstream consumer, empty match
// ==========================================================================================
TEST_F(Conv3dToConv3dV2FusionPassTest, conv3d_to_conv3dv2_graph_topology)
{
    struct {
        const char *pointName;
        std::function<void(Conv3dToConv3dV2FusionPassTest &)> run;
    } const points[] = {
        {"dynamic_shape_fp16", [](Conv3dToConv3dV2FusionPassTest &self) {
             self.ApplySoc(SocProfile::Arch35);
             TestGraph builder("conv3d_to_conv3dv2_graph_topology_dynamic_shape_fp16");
             auto graph = builder.AddConv3D(Conv3DConfig::Basic("Conv3D", DT_FLOAT16, FORMAT_NCDHW))
                 .SetOutput("Conv3D")
                 .Build();
             self.FixOutputDtype(builder, "Conv3D", DT_FLOAT16);
             self.SetConv3DDynamicShapes(builder, "Conv3D", DT_FLOAT16, FORMAT_NCDHW, true, true);
             self.TestTotalPass("conv3d_to_conv3dv2_graph_topology_dynamic_shape_fp16", graph, SUCCESS);
         }},
        {"dynamic_shape_fp32", [](Conv3dToConv3dV2FusionPassTest &self) {
             self.ApplySoc(SocProfile::Arch35);
             TestGraph builder("conv3d_to_conv3dv2_graph_topology_dynamic_shape_fp32");
             auto graph = builder.AddConv3D(Conv3DConfig::Basic("Conv3D", DT_FLOAT, FORMAT_NCDHW))
                 .SetOutput("Conv3D")
                 .Build();
             self.FixOutputDtype(builder, "Conv3D", DT_FLOAT);
             self.SetConv3DDynamicShapes(builder, "Conv3D", DT_FLOAT, FORMAT_NCDHW, true, true);
             self.TestTotalPass("conv3d_to_conv3dv2_graph_topology_dynamic_shape_fp32", graph, SUCCESS);
         }},
        {"fixpipe_arch35_fp16", [](Conv3dToConv3dV2FusionPassTest &self) {
             self.ApplySoc(SocProfile::Arch35);
             TestGraph builder("conv3d_to_conv3dv2_graph_topology_fixpipe_arch35_fp16");
             const std::vector<int64_t> outShape = {1, 16, 16, 244, 244};
             auto graph = builder
                 .AddConv3D(Conv3DConfig::Basic("Conv3D", DT_FLOAT16, FORMAT_NCDHW))
                 .AddPostCube(PostCubeConfig::Basic("FixPipe", DT_FLOAT16, FORMAT_NCDHW, outShape), false)
                 .Connect("Conv3D", 0, "FixPipe", 0)
                 .SetOutput("FixPipe")
                 .Build();
             self.FixOutputDtype(builder, "Conv3D", DT_FLOAT16);
             self.TestTotalPass("conv3d_to_conv3dv2_graph_topology_fixpipe_arch35_fp16", graph, SUCCESS);
         }},
        {"ifmr_arch35_fp16", [](Conv3dToConv3dV2FusionPassTest &self) {
             self.ApplySoc(SocProfile::Arch35);
             TestGraph builder("conv3d_to_conv3dv2_graph_topology_ifmr_arch35_fp16");
             auto graph = self.BuildFp16Conv3DWithIfmrAdd(builder);
             self.FixOutputDtype(builder, "Conv3D", DT_FLOAT16);
             self.TestTotalPass("conv3d_to_conv3dv2_graph_topology_ifmr_arch35_fp16", graph, SUCCESS);
         }},
        {"ifmr_nonarch35_bf16", [](Conv3dToConv3dV2FusionPassTest &self) {
             self.ApplySoc(SocProfile::NonArch35);
             TestGraph builder("conv3d_to_conv3dv2_graph_topology_ifmr_nonarch35_bf16");
             const std::vector<int64_t> shape = {1, 16, 16, 244, 244};
             auto graph = builder
                 .AddConv3D(Conv3DConfig::Basic("Conv3D", DT_BF16))
                 .AddIFMR(IFMRConfig::Basic("IFMR", DT_BF16))
                 .AddAdd(AddConfig::Basic("Add", DT_BF16, FORMAT_NCDHW, shape))
                 .Connect("Conv3D", 0, "Add", 0)
                 .Connect("IFMR", 0, "Add", 1)
                 .SetOutput("Add")
                 .Build();
             self.FixOutputDtype(builder, "Conv3D", DT_BF16);
             self.TestTotalPass("conv3d_to_conv3dv2_graph_topology_ifmr_nonarch35_bf16", graph, SUCCESS);
         }},
        {"transdata_c15_nonarch35", [](Conv3dToConv3dV2FusionPassTest &self) {
             TestGraph builder("conv3d_to_conv3dv2_graph_topology_transdata_c15_nonarch35");
             auto graph = self.BuildConv3DWithTransData(builder, 15, SocProfile::NonArch35);
             self.TestTotalPass("conv3d_to_conv3dv2_graph_topology_transdata_c15_nonarch35", graph, SUCCESS);
         }},
        {"multi_conv3d_same_graph", [](Conv3dToConv3dV2FusionPassTest &self) {
             self.ApplySoc(SocProfile::Arch35);
             TestGraph builder("conv3d_to_conv3dv2_graph_topology_multi_conv3d_same_graph");
             auto graph = builder
                 .AddConv3D(Conv3DConfig::Basic("Conv3D_1", DT_FLOAT16, FORMAT_NCDHW))
                 .AddConv3D(Conv3DConfig::Basic("Conv3D_2", DT_FLOAT16, FORMAT_NCDHW))
                 .SetOutput("Conv3D_1")
                 .SetOutput("Conv3D_2")
                 .Build();
             self.FixOutputDtype(builder, "Conv3D_1", DT_FLOAT16);
             self.FixOutputDtype(builder, "Conv3D_2", DT_FLOAT16);
             self.TestTotalPass("conv3d_to_conv3dv2_graph_topology_multi_conv3d_same_graph", graph, SUCCESS);
             EXPECT_EQ(GraphChecker::CountNodes(graph, "Conv3DV2"), 2);
         }},
        {"multi_conv3d_serial", [](Conv3dToConv3dV2FusionPassTest &self) {
             self.ApplySoc(SocProfile::Arch35);
             TestGraph builder("conv3d_to_conv3dv2_graph_topology_multi_conv3d_serial");
             auto graph = builder
                 .AddConv3D(Conv3DConfig::Basic("Conv3D_1", DT_FLOAT16, FORMAT_NCDHW))
                 .AddConv3D(Conv3DConfig::Basic("Conv3D_2", DT_FLOAT16, FORMAT_NCDHW,
                     {1, 3, 16, 244, 244}, {3, 3, 3, 3, 3}, {1, 3, 16, 244, 244}))
                 .Connect("Conv3D_1", 0, "Conv3D_2", 0)
                 .SetOutput("Conv3D_2")
                 .Build();
             self.FixOutputDtype(builder, "Conv3D_1", DT_FLOAT16);
             self.FixOutputDtype(builder, "Conv3D_2", DT_FLOAT16);
             self.TestTotalPass("conv3d_to_conv3dv2_graph_topology_multi_conv3d_serial", graph, SUCCESS);
             EXPECT_EQ(GraphChecker::CountNodes(graph, "Conv3DV2"), 2);
             EXPECT_EQ(GraphChecker::CountNodes(graph, "Conv3D"), 0);
         }},
        {"multi_conv3d_mixed_eligibility", [](Conv3dToConv3dV2FusionPassTest &self) {
             self.ApplySoc(SocProfile::Arch35);
             TestGraph builder("conv3d_to_conv3dv2_graph_topology_multi_conv3d_mixed_eligibility");
             auto graph = builder
                 .AddConv3D(Conv3DConfig::Basic("Conv3D_ok", DT_FLOAT16, FORMAT_NCDHW))
                 .AddConv3D(Conv3DConfig::Basic("Conv3D_bad", DT_INT8, FORMAT_NCDHW))
                 .SetOutput("Conv3D_ok")
                 .SetOutput("Conv3D_bad")
                 .Build();
             self.FixOutputDtype(builder, "Conv3D_ok", DT_FLOAT16);
             CustomPassContext passContext;
             passContext.SetPassName("conv3d_to_conv3dv2_graph_topology_multi_conv3d_mixed_eligibility");
             Conv3dToConv3dV2FusionPass pass({AscendString("Conv3D")});
             EXPECT_EQ(pass.Run(graph, passContext), SUCCESS);
             EXPECT_EQ(GraphChecker::CountNodes(graph, "Conv3DV2"), 1);
             EXPECT_EQ(GraphChecker::CountNodes(graph, "Conv3D"), 1);
         }},
        {"downstream_consumer_preserved", [](Conv3dToConv3dV2FusionPassTest &self) {
             self.ApplySoc(SocProfile::Arch35);
             const std::vector<int64_t> outShape = {1, 3, 16, 244, 244};
             TestGraph builder("conv3d_to_conv3dv2_graph_topology_downstream_consumer_preserved");
             auto graph = builder
                 .AddConv3D(Conv3DConfig::Basic("Conv3D", DT_FLOAT16, FORMAT_NCDHW))
                 .AddRelu(ReluConfig::Basic("Relu", DT_FLOAT16, FORMAT_NCDHW, outShape))
                 .Connect("Conv3D", 0, "Relu", 0)
                 .SetOutput("Relu")
                 .Build();
             self.FixOutputDtype(builder, "Conv3D", DT_FLOAT16);
             self.TestTotalPass("conv3d_to_conv3dv2_graph_topology_downstream_consumer_preserved", graph, SUCCESS);
             EXPECT_TRUE(GraphChecker::HasNode(graph, "Relu"));
         }},
        {"graph_without_conv3d", [](Conv3dToConv3dV2FusionPassTest &self) {
             self.ApplySoc(SocProfile::Arch35);
             const std::vector<int64_t> outShape = {1, 3, 16, 244, 244};
             auto graph = TestGraph("conv3d_to_conv3dv2_graph_topology_graph_without_conv3d")
                 .AddRelu(ReluConfig::Basic("Relu", DT_FLOAT16, FORMAT_NCDHW, outShape), true)
                 .SetOutput("Relu")
                 .Build();
             CustomPassContext passContext;
             passContext.SetPassName("conv3d_to_conv3dv2_graph_topology_graph_without_conv3d");
             Conv3dToConv3dV2FusionPass pass({AscendString("Conv3D")});
             EXPECT_EQ(pass.Run(graph, passContext), GRAPH_NOT_CHANGED);
             EXPECT_FALSE(GraphChecker::HasNode(graph, "Conv3DV2"));
         }},
        {"reentrant_two_graphs", [](Conv3dToConv3dV2FusionPassTest &self) {
             self.ApplySoc(SocProfile::Arch35);
             TestGraph builder1("conv3d_to_conv3dv2_graph_topology_reentrant_1");
             auto graph1 = builder1.AddConv3D(Conv3DConfig::Basic("Conv3D_1", DT_FLOAT16, FORMAT_NCDHW))
                 .SetOutput("Conv3D_1")
                 .Build();
             self.FixOutputDtype(builder1, "Conv3D_1", DT_FLOAT16);

             self.ApplySoc(SocProfile::Arch35);
             TestGraph builder2("conv3d_to_conv3dv2_graph_topology_reentrant_2");
             auto graph2 = builder2.AddConv3D(Conv3DConfig::Basic("Conv3D_2", DT_INT8, FORMAT_NCDHW))
                 .SetOutput("Conv3D_2")
                 .Build();

             CustomPassContext passContext1;
             passContext1.SetPassName("conv3d_to_conv3dv2_graph_topology_reentrant_1");
             Conv3dToConv3dV2FusionPass pass1({AscendString("Conv3D")});
             EXPECT_EQ(pass1.Run(graph1, passContext1), SUCCESS);
             EXPECT_TRUE(GraphChecker::HasNode(graph1, "Conv3DV2"));

             CustomPassContext passContext2;
             passContext2.SetPassName("conv3d_to_conv3dv2_graph_topology_reentrant_2");
             Conv3dToConv3dV2FusionPass pass2({AscendString("Conv3D")});
             EXPECT_EQ(pass2.Run(graph2, passContext2), GRAPH_NOT_CHANGED);
             EXPECT_FALSE(GraphChecker::HasNode(graph2, "Conv3DV2"));
             EXPECT_TRUE(GraphChecker::HasNode(graph2, "Conv3D"));
         }},
    };

    for (const auto &p : points) {
        SCOPED_TRACE(p.pointName);
        p.run(*this);
    }
}

#endif