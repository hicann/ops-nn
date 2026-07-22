/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_conv2d_v2_ascendc_pipeline_extra_tiling.cpp
 * \brief Additional UT for conv2d_v2 base tiling pipeline coverage
 */

#include <gtest/gtest.h>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <cstdio>
#include "log/log.h"
#include "array_ops.h"
#include "ut_op_util.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "../../../../common/op_host/op_tiling/arch35/conv_base_utils.h"
#include "../../../../common/op_host/op_tiling/arch35/conv_base.h"
#include "test_conv2d_v2_ascendc_utils_tiling.h"

using namespace ut_util;
using namespace conv_tiling_utils;

namespace {
struct PipelineTestParams {
    std::string caseName;
    std::vector<int64_t> fmShape;
    std::vector<int64_t> weightShape;
    std::vector<uint32_t> pads;
    std::vector<uint32_t> strides;
    std::vector<uint32_t> dilations;
    int64_t groups;
    std::string padMode;
    std::string format;
    ge::DataType dtype;
    bool hasBias;
    bool expectFail;
};

void RunPipelineTest(const PipelineTestParams& params)
{
    uint32_t padu = params.pads[0], padd = params.pads[1], padl = params.pads[2], padr = params.pads[3];
    uint32_t strideH = params.strides[0], strideW = params.strides[1];
    uint32_t dilationH = params.dilations[0], dilationW = params.dilations[1];

    int64_t cout = params.weightShape[0], ci_per_group = params.weightShape[1];
    int64_t kH = params.weightShape[2], kW = params.weightShape[3];
    int64_t batch = params.fmShape[0], cin = params.fmShape[1];
    int64_t hi = params.fmShape[2], wi = params.fmShape[3];

    ge::Format fmapFormat = ge::FORMAT_NCHW;
    ge::Format weightFormat = ge::FORMAT_NCHW;
    ge::Format outputFormat = ge::FORMAT_NCHW;

    gert::StorageShape featuremap;
    gert::StorageShape weight;
    gert::StorageShape bias = {{cout}, {cout}};
    gert::StorageShape output;

    ConvShape convShapeH = {static_cast<uint64_t>(hi), static_cast<uint64_t>(kH), padu, padd, dilationH, strideH};
    ConvShape convShapeW = {static_cast<uint64_t>(wi), static_cast<uint64_t>(kW), padl, padr, dilationW, strideW};
    int64_t ho = InferOut(convShapeH);
    int64_t wo = InferOut(convShapeW);
    if (params.padMode == "SAME" || params.padMode == "SAME_UPPER" || params.padMode == "SAME_LOWER") {
        ho = (hi + strideH - 1) / strideH;
        wo = (wi + strideW - 1) / strideW;
    }

    if (params.format == "NHWC") {
        fmapFormat = ge::FORMAT_NHWC;
        weightFormat = ge::FORMAT_HWCN;
        outputFormat = ge::FORMAT_NHWC;
        featuremap = {{batch, hi, wi, cin}, {batch, hi, wi, cin}};
        weight = {{kH, kW, ci_per_group, cout}, {kH, kW, ci_per_group, cout}};
        output = {{batch, ho, wo, cout}, {batch, ho, wo, cout}};
    } else {
        featuremap = {{batch, cin, hi, wi}, {batch, cin, hi, wi}};
        weight = {{cout, ci_per_group, kH, kW}, {cout, ci_per_group, kH, kW}};
        output = {{batch, cout, ho, wo}, {batch, cout, ho, wo}};
    }

    std::vector<void*> input_shape_ref = params.hasBias ? std::vector<void*>{&featuremap, &weight, &bias, nullptr} :
                                                          std::vector<void*>{&featuremap, &weight, nullptr, nullptr};
    std::vector<void*> output_shapes_ref = {&output};

    std::vector<int64_t> stridesVec = params.format == "NHWC" ?
                                          std::vector<int64_t>{1, static_cast<int64_t>(strideH),
                                                               static_cast<int64_t>(strideW), 1} :
                                          std::vector<int64_t>{1, 1, static_cast<int64_t>(strideH),
                                                               static_cast<int64_t>(strideW)};
    std::vector<int64_t> padsVec = {static_cast<int64_t>(padu), static_cast<int64_t>(padd), static_cast<int64_t>(padl),
                                    static_cast<int64_t>(padr)};
    std::vector<int64_t> dilationsVec = params.format == "NHWC" ?
                                            std::vector<int64_t>{1, static_cast<int64_t>(dilationH),
                                                                 static_cast<int64_t>(dilationW), 1} :
                                            std::vector<int64_t>{1, 1, static_cast<int64_t>(dilationH),
                                                                 static_cast<int64_t>(dilationW)};

    std::string op_type = "Conv2DV2";
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;

    std::string compile_info_string = R"({"hardware_info":
        {"BT_SIZE": 4096, "load3d_constraints": "1", "Intrinsic_fix_pipe_l0c2out": false,
        "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true,
        "Intrinsic_data_move_out2l1_nd2nz": false, "UB_SIZE": 253952,
        "L2_SIZE": 134217728, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536, "FB_SIZE": 4096,
        "BT_SIZE": 4096, "L0C_SIZE": 262144, "CORE_NUM": 32, "cube_core_cnt": 32, "vector_core_cnt": 64,
        "core_type_list": "CubeCore,VectorCore"}})";
    std::map<std::string, std::string> soc_infos, aicore_spec, intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);
    std::map<std::string, std::string> soc_version_infos = {{"NpuArch", "3510"}};
    aicore_spec.insert({"fb0_size", "4096"});

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    optiling::conv_ops_tiling::ConvTilingParseInfo compile_info;

    auto tilingDataPtr = gert::TilingData::CreateCap(MEM_SIZE_4K);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(MEM_SIZE_4K);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(tilingDataPtr, nullptr);

    ge::DataType bias_dtype = params.dtype;

    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(NUM_4, 1)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes(input_shape_ref)
                      .OutputShapes(output_shapes_ref)
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, params.dtype, fmapFormat, fmapFormat)
                      .NodeInputTd(1, params.dtype, weightFormat, weightFormat)
                      .NodeInputTd(DIM_2, bias_dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(DIM_3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, params.dtype, outputFormat, outputFormat)
                      .NodeAttrs({{"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(stridesVec)},
                                  {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(padsVec)},
                                  {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(dilationsVec)},
                                  {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(params.groups)},
                                  {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>(params.format)},
                                  {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                  {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>(params.padMode)},
                                  {"enable_hf32", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .TilingData(tilingDataPtr.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    ge::graphStatus ret = tiling_func(tiling_context);

    if (params.expectFail) {
        EXPECT_EQ(ret, ge::GRAPH_FAILED) << "Case [" << params.caseName << "] expected FAILED but got SUCCESS";
    } else {
        EXPECT_EQ(ret, ge::GRAPH_SUCCESS) << "Case [" << params.caseName << "] expected SUCCESS but got FAILED";
    }
}
} // namespace

class TestConv2dPipelineExtra : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// ============================================================================
// Group conv: groups=8 depthwise
// ============================================================================
TEST_F(TestConv2dPipelineExtra, group_conv_depthwise_success)
{
    RunPipelineTest({"group_dw",
                     {1, 8, 16, 16},
                     {8, 1, 3, 3},
                     {1, 1, 1, 1},
                     {1, 1},
                     {1, 1},
                     8,
                     "SPECIFIC",
                     "NCHW",
                     ge::DT_FLOAT16,
                     true,
                     false});
}

// ============================================================================
// Group conv: groups=2 with multiple channels per group
// ============================================================================
TEST_F(TestConv2dPipelineExtra, group_conv_2groups_success)
{
    RunPipelineTest({"group_2g",
                     {1, 32, 32, 32},
                     {64, 16, 3, 3},
                     {1, 1, 1, 1},
                     {1, 1},
                     {1, 1},
                     2,
                     "SPECIFIC",
                     "NCHW",
                     ge::DT_FLOAT16,
                     true,
                     false});
}

// ============================================================================
// Group conv: groups=4 with asymmetric pads + stride
// ============================================================================
TEST_F(TestConv2dPipelineExtra, group_conv_with_pad_stride_success)
{
    RunPipelineTest({"group_pad_stride",
                     {1, 16, 28, 28},
                     {32, 4, 3, 3},
                     {1, 1, 1, 1},
                     {2, 2},
                     {1, 1},
                     4,
                     "SPECIFIC",
                     "NCHW",
                     ge::DT_FLOAT16,
                     true,
                     false});
}

// ============================================================================
// NHWC format basic conv
// ============================================================================
TEST_F(TestConv2dPipelineExtra, nhwc_basic_success)
{
    RunPipelineTest({"nhwc_basic",
                     {1, 16, 32, 64},
                     {32, 16, 3, 3},
                     {1, 1, 1, 1},
                     {1, 1},
                     {1, 1},
                     1,
                     "SPECIFIC",
                     "NHWC",
                     ge::DT_FLOAT16,
                     true,
                     false});
}

// ============================================================================
// NHWC format with groups
// ============================================================================
TEST_F(TestConv2dPipelineExtra, nhwc_group_conv_success)
{
    RunPipelineTest({"nhwc_group",
                     {1, 16, 16, 32},
                     {16, 4, 3, 3},
                     {1, 1, 1, 1},
                     {2, 2},
                     {1, 1},
                     4,
                     "SPECIFIC",
                     "NHWC",
                     ge::DT_FLOAT16,
                     true,
                     false});
}

// ============================================================================
// FLOAT32 dtype
// ============================================================================
TEST_F(TestConv2dPipelineExtra, float32_basic_success)
{
    RunPipelineTest({"fp32_basic",
                     {1, 16, 16, 16},
                     {32, 16, 3, 3},
                     {1, 1, 1, 1},
                     {1, 1},
                     {1, 1},
                     1,
                     "SPECIFIC",
                     "NCHW",
                     ge::DT_FLOAT,
                     true,
                     false});
}

// ============================================================================
// SAME pad mode with large kernel
// ============================================================================
TEST_F(TestConv2dPipelineExtra, same_pad_large_kernel_success)
{
    RunPipelineTest({"same_largek",
                     {1, 16, 56, 56},
                     {32, 16, 7, 7},
                     {0, 0, 0, 0},
                     {2, 2},
                     {1, 1},
                     1,
                     "SAME",
                     "NCHW",
                     ge::DT_FLOAT16,
                     true,
                     false});
}

// ============================================================================
// VALID pad mode
// ============================================================================
TEST_F(TestConv2dPipelineExtra, valid_pad_success)
{
    RunPipelineTest({"valid_pad",
                     {1, 16, 32, 32},
                     {16, 16, 3, 3},
                     {0, 0, 0, 0},
                     {1, 1},
                     {1, 1},
                     1,
                     "VALID",
                     "NCHW",
                     ge::DT_FLOAT16,
                     true,
                     false});
}

// ============================================================================
// Dilated conv
// ============================================================================
TEST_F(TestConv2dPipelineExtra, dilated_conv_success)
{
    RunPipelineTest({"dilated",
                     {1, 16, 32, 32},
                     {32, 16, 3, 3},
                     {2, 2, 2, 2},
                     {1, 1},
                     {2, 2},
                     1,
                     "SPECIFIC",
                     "NCHW",
                     ge::DT_FLOAT16,
                     true,
                     false});
}

// ============================================================================
// Batch > 1
// ============================================================================
TEST_F(TestConv2dPipelineExtra, batch_2_success)
{
    RunPipelineTest({"batch2",
                     {2, 16, 28, 28},
                     {32, 16, 3, 3},
                     {1, 1, 1, 1},
                     {1, 1},
                     {1, 1},
                     1,
                     "SPECIFIC",
                     "NCHW",
                     ge::DT_FLOAT16,
                     true,
                     false});
}

// ============================================================================
// Batch > 1 with groups
// ============================================================================
TEST_F(TestConv2dPipelineExtra, batch_group_success)
{
    RunPipelineTest({"batch_group",
                     {4, 32, 14, 14},
                     {64, 8, 3, 3},
                     {1, 1, 1, 1},
                     {1, 1},
                     {1, 1},
                     4,
                     "SPECIFIC",
                     "NCHW",
                     ge::DT_FLOAT16,
                     true,
                     false});
}

// ============================================================================
// Large batch
// ============================================================================
TEST_F(TestConv2dPipelineExtra, batch_8_success)
{
    RunPipelineTest({"batch8",
                     {8, 16, 16, 16},
                     {16, 16, 1, 1},
                     {0, 0, 0, 0},
                     {1, 1},
                     {1, 1},
                     1,
                     "SPECIFIC",
                     "NCHW",
                     ge::DT_FLOAT16,
                     false,
                     false});
}

// ============================================================================
// Stride=2 with padding
// ============================================================================
TEST_F(TestConv2dPipelineExtra, stride2_with_pad_success)
{
    RunPipelineTest({"stride2_pad",
                     {1, 3, 224, 224},
                     {64, 3, 7, 7},
                     {3, 3, 3, 3},
                     {2, 2},
                     {1, 1},
                     1,
                     "SPECIFIC",
                     "NCHW",
                     ge::DT_FLOAT16,
                     true,
                     false});
}

// ============================================================================
// Pointwise conv (1x1 kernel)
// ============================================================================
TEST_F(TestConv2dPipelineExtra, pointwise_success)
{
    RunPipelineTest({"pointwise",
                     {1, 128, 28, 28},
                     {256, 128, 1, 1},
                     {0, 0, 0, 0},
                     {1, 1},
                     {1, 1},
                     1,
                     "SPECIFIC",
                     "NCHW",
                     ge::DT_FLOAT16,
                     false,
                     false});
}

// ============================================================================
// Small spatial input with large kernel
// ============================================================================
TEST_F(TestConv2dPipelineExtra, small_input_large_kernel_success)
{
    RunPipelineTest({"small_largek",
                     {1, 64, 7, 7},
                     {128, 64, 5, 5},
                     {0, 0, 0, 0},
                     {1, 1},
                     {1, 1},
                     1,
                     "SPECIFIC",
                     "NCHW",
                     ge::DT_FLOAT16,
                     true,
                     false});
}

// ============================================================================
// Asymmetric padding
// ============================================================================
TEST_F(TestConv2dPipelineExtra, asymmetric_pad_success)
{
    RunPipelineTest({"asym_pad",
                     {1, 16, 16, 16},
                     {16, 16, 3, 3},
                     {2, 1, 1, 2},
                     {1, 1},
                     {1, 1},
                     1,
                     "SPECIFIC",
                     "NCHW",
                     ge::DT_FLOAT16,
                     true,
                     false});
}

// ============================================================================
// NHWC with no bias
// ============================================================================
TEST_F(TestConv2dPipelineExtra, nhwc_no_bias_success)
{
    RunPipelineTest({"nhwc_nobias",
                     {1, 32, 14, 64},
                     {64, 32, 3, 3},
                     {1, 1, 1, 1},
                     {1, 1},
                     {1, 1},
                     1,
                     "SPECIFIC",
                     "NHWC",
                     ge::DT_FLOAT16,
                     false,
                     false});
}

// ============================================================================
// Depthwise conv (groups = cin = cout)
// ============================================================================
TEST_F(TestConv2dPipelineExtra, depthwise_separable_success)
{
    RunPipelineTest({"dw_sep",
                     {1, 32, 56, 56},
                     {32, 1, 3, 3},
                     {1, 1, 1, 1},
                     {1, 1},
                     {1, 1},
                     32,
                     "SPECIFIC",
                     "NCHW",
                     ge::DT_FLOAT16,
                     true,
                     false});
}
