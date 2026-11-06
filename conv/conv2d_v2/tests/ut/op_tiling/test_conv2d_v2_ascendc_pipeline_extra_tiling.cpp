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

// ============================================================================
// Tiling key computation coverage. The tiling key getters are only reachable
// through the full tiling pipeline, so drive them directly here.
// ============================================================================
#include "../../../op_host/op_tiling/arch35/conv2d_v2_base_tiling.h"

namespace {
using namespace optiling::conv_ops_tiling;

void InitTilingKeyState(optiling::conv_ops_tiling::Conv2dBaseTiling& sut,
                        optiling::conv_ops_tiling::ConvTilingParseInfo& opInfo)
{
    opInfo.aicoreNum = AIC_NUM;
    opInfo.l0cSize = MEM_SIZE_256K;
    opInfo.ubSize = MEM_SIZE_200K;
    opInfo.npuArch = NpuArch::DAV_3510;
    sut.opInfo_ = &opInfo;

    sut.shapeInfo_.batch = 1;
    sut.shapeInfo_.ci = 16;
    sut.shapeInfo_.hi = 16;
    sut.shapeInfo_.wi = 16;
    sut.shapeInfo_.co = 16;
    sut.shapeInfo_.kh = 3;
    sut.shapeInfo_.kw = 3;
    sut.shapeInfo_.ho = 14;
    sut.shapeInfo_.wo = 14;
    sut.attrInfo_.groups = 1;
    sut.attrInfo_.strideH = 1;
    sut.attrInfo_.strideW = 1;
    sut.attrInfo_.dilationH = 1;
    sut.attrInfo_.dilationW = 1;
    sut.descInfo_.fMapDtype = ge::DT_FLOAT16;
    sut.descInfo_.weightDtype = ge::DT_FLOAT16;
    sut.descInfo_.outDtype = ge::DT_FLOAT16;
    sut.descInfo_.fMapFormat = ge::FORMAT_NCHW;
    sut.descInfo_.weightFormat = ge::FORMAT_NCHW;
    sut.descInfo_.outFormat = ge::FORMAT_NCHW;
    sut.flagInfo_.hasBias = true;
    sut.flagInfo_.convGroupType = ConvGroupType::NORMAL_CONV;
    sut.paramInfo_.nodeType = "Conv2DV2";
    sut.convOpsConstParams_.m0 = CUBE_M0;
    sut.convOpsConstParams_.k0 = CUBE_K0_16;
    sut.convOpsConstParams_.n0 = CUBE_N0;
    sut.numBlocksRes.nDim = 1;
    sut.apiInputPlatformInfo.npuArch = NpuArch::DAV_3510;
    sut.apiInputPlatformInfo.l0BSize = MEM_SIZE_64K;

    sut.tilingData_.set_kAL1(144);
    sut.tilingData_.set_kBL1(144);
    sut.tilingData_.set_nBL1(16);
    sut.tilingData_.set_nL0(16);
    sut.tilingData_.set_hoL1(14);
    sut.tilingData_.set_woL1(14);
    sut.tilingData_.set_hoL0(16);
    sut.tilingData_.set_woL0(16);
    sut.tilingData_.set_singleCoreBatch(1);
    sut.tilingData_.set_singleCoreHo(14);
    sut.tilingData_.set_singleCoreWo(14);
    sut.tilingData_.set_pBufferFlag(27);
    sut.tilingData_.set_kernelH(3);
    sut.tilingData_.set_kernelW(3);
    sut.tilingData_.set_innerBatch(1);
    sut.tilingData_.set_iterateMNOrder(1);
}
} // namespace

class Conv2dTilingKeyExtra : public testing::Test {
protected:
    void SetUp() override
    {
        sut_ = std::make_shared<optiling::conv_ops_tiling::Conv2dBaseTiling>(nullptr);
        InitTilingKeyState(*sut_, opInfo_);
    }
    void TearDown() override {}

    std::shared_ptr<optiling::conv_ops_tiling::Conv2dBaseTiling> sut_;
    optiling::conv_ops_tiling::ConvTilingParseInfo opInfo_;
};

TEST_F(Conv2dTilingKeyExtra, getters_normal_conv)
{
    EXPECT_EQ(sut_->GetL0PingPongVal(), 27u & optiling::conv_ops_tiling::L0A_L0B_PB_FLAG_MASK);
    EXPECT_EQ(sut_->GetL1PingPongVal(),
              (27u & optiling::conv_ops_tiling::L1A_L1B_PB_FLAG_MASK) >> optiling::conv_ops_tiling::L1_PB_OFFSET);
    EXPECT_EQ(sut_->GetOutputOrderVal(), 0u);
    sut_->flagInfo_.mSplitModeFlag = true;
    EXPECT_EQ(sut_->GetOutputOrderVal(), 1u);
    EXPECT_EQ(sut_->GetWeightUbTrans(), optiling::conv_ops_tiling::WEIGHT_UB_TRANS_CLOSE);
    EXPECT_EQ(sut_->GetFmapCopyMode(), optiling::conv_ops_tiling::FMAP_LOAD3D_MODE);
    EXPECT_EQ(sut_->GetNoPad(), 1u); // CONV_NO_PAD
    sut_->attrInfo_.padTop = 1;
    EXPECT_EQ(sut_->GetNoPad(), 0u); // CONV_HAS_PAD

    sut_->tilingData_.set_bUbNStep(16);
    sut_->tilingData_.set_bUbKStep(16);
    EXPECT_EQ(sut_->GetWeightUbTrans(), optiling::conv_ops_tiling::WEIGHT_UB_TRANS_OPEN);
    sut_->tilingData_.set_khUb(1);
    sut_->tilingData_.set_kwUb(1);
    EXPECT_EQ(sut_->GetFmapCopyMode(), optiling::conv_ops_tiling::FMAP_DMA_MODE);
}

TEST_F(Conv2dTilingKeyExtra, getters_group_conv)
{
    sut_->flagInfo_.convGroupType = ConvGroupType::OPT_GROUP_CONV;
    sut_->tilingData_.set_pBufferFlag(optiling::conv_ops_tiling::L1_PB_ALL_OPEN
                                      << optiling::conv_ops_tiling::L1_PB_OFFSET);
    EXPECT_EQ(sut_->GetL1PingPongVal(), optiling::conv_ops_tiling::L1_PB_ALL_OPEN);
    sut_->tilingData_.set_pBufferFlag(optiling::conv_ops_tiling::L1_PB_ALL_CLOSE
                                      << optiling::conv_ops_tiling::L1_PB_OFFSET);
    EXPECT_EQ(sut_->GetL1PingPongVal(), optiling::conv_ops_tiling::L1_PB_ALL_CLOSE);
}

TEST_F(Conv2dTilingKeyExtra, fmp_tiling_hw_split_partial)
{
    sut_->tilingData_.set_singleCoreWo(24);
    sut_->tilingData_.set_woL0(16);
    sut_->tilingData_.set_woL1(32);
    sut_->tilingData_.set_hoL0(32);
    sut_->tilingData_.set_hoL1(32);
    sut_->tilingData_.set_singleCoreHo(32);
    EXPECT_EQ(sut_->GetFmpTilingValForHWSplit(false), optiling::conv_ops_tiling::FMP_OTHER);
}

TEST_F(Conv2dTilingKeyExtra, set_tiling_key_nchw)
{
    ASSERT_EQ(sut_->SetTilingKey(), ge::GRAPH_SUCCESS);
    EXPECT_NE(sut_->tilingKey_, 0u);
}

TEST_F(Conv2dTilingKeyExtra, set_tiling_key_nz_weight)
{
    sut_->descInfo_.weightFormat = ge::FORMAT_FRACTAL_Z;
    sut_->tilingData_.set_singleCoreBatch(1);
    ASSERT_EQ(sut_->SetTilingKey(), ge::GRAPH_SUCCESS);
    EXPECT_EQ(sut_->GetNoPad(), 1u); // CONV_NO_PAD
    EXPECT_NE(sut_->tilingKey_, 0u);
}

TEST_F(Conv2dTilingKeyExtra, reset_tiling_key_para_9201_group)
{
    sut_->apiInputPlatformInfo.npuArch = NpuArch::DAV_9201;
    sut_->flagInfo_.convGroupType = ConvGroupType::ORI_GROUP_CONV;
    sut_->tilingKeyPara_.weightTiling = 1;
    sut_->ReSetTilingKeyPara();
    EXPECT_EQ(sut_->tilingKeyPara_.weightTiling, 1u);

    sut_->apiInputPlatformInfo.npuArch = NpuArch::DAV_3510;
    sut_->flagInfo_.convGroupType = ConvGroupType::NORMAL_CONV;
    sut_->flagInfo_.mSplitModeFlag = true;
    sut_->tilingKeyPara_.weightTiling = 1;
    sut_->tilingKeyPara_.fmpTiling = 1;
    sut_->tilingKeyPara_.outputOrder = 1;
    sut_->ReSetTilingKeyPara();
    EXPECT_EQ(sut_->tilingKeyPara_.weightTiling, 1u);
    EXPECT_EQ(sut_->tilingKeyPara_.fmpTiling, 1u);
}

TEST_F(Conv2dTilingKeyExtra, reset_tiling_key_para_reset_flags)
{
    sut_->flagInfo_.mSplitModeFlag = false;
    sut_->tilingKeyPara_.weightTiling = 1;
    sut_->tilingKeyPara_.fmpTiling = 1;
    sut_->tilingKeyPara_.outputOrder = 0;
    sut_->ReSetTilingKeyPara();
    EXPECT_EQ(sut_->tilingKeyPara_.weightTiling, optiling::conv_ops_tiling::WEIGHT_OTHER);
    EXPECT_EQ(sut_->tilingKeyPara_.fmpTiling, optiling::conv_ops_tiling::FMP_OTHER);

    sut_->flagInfo_.convGroupType = ConvGroupType::OPT_GROUP_CONV;
    sut_->flagInfo_.mSplitModeFlag = true;
    sut_->tilingKeyPara_.weightTiling = 1;
    sut_->tilingKeyPara_.fmpTiling = 1;
    sut_->tilingKeyPara_.outputOrder = 1;
    sut_->tilingKeyPara_.fmapCppyMode = optiling::conv_ops_tiling::FMAP_LOAD3D_MODE;
    sut_->tilingKeyPara_.l1PingPong = optiling::conv_ops_tiling::L1_PB_ALL_OPEN;
    sut_->tilingData_.set_iterateMNOrder(0);
    sut_->tilingData_.set_hoL0(14);
    sut_->tilingKeyPara_.weightTiling = optiling::conv_ops_tiling::WEIGHT_OTHER;
    sut_->tilingKeyPara_.fmpTiling = optiling::conv_ops_tiling::FMP_OTHER;
    sut_->ReSetTilingKeyPara();
    EXPECT_EQ(sut_->tilingKeyPara_.weightTiling, optiling::conv_ops_tiling::FULLLOAD_BL1);
}

// ============================================================================
// ASWT / FullMN post-tiling branches that are not reachable on ascend960dt
// (DAV_5102 ASWT enable, FullMN buffer error path, non-9201 ASWT block dim).
// ============================================================================
namespace {
gert::KernelRunContextHolder BuildPostTilingContext(optiling::conv_ops_tiling::ConvTilingParseInfo& compileInfo,
                                                    fe::PlatFormInfos& platformInfo, const void* tilingData,
                                                    gert::ContinuousVector* ws)
{
    gert::StorageShape fmap = {{1, 16, 16, 16}, {1, 16, 16, 16}};
    gert::StorageShape weight = {{16, 16, 3, 3}, {16, 16, 3, 3}};
    gert::StorageShape bias = {{16}, {16}};
    gert::StorageShape output = {{1, 16, 16, 16}, {1, 16, 16, 16}};
    std::vector<void*> inputShapes = {&fmap, &weight, &bias, nullptr};
    std::vector<void*> outputShapes = {&output};
    return gert::TilingContextFaker()
        .SetOpType("Conv2DV2")
        .NodeIoNum(NUM_4, 1)
        .IrInstanceNum({1, 1, 1, 1})
        .InputShapes(inputShapes)
        .OutputShapes(outputShapes)
        .CompileInfo(&compileInfo)
        .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
        .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
        .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
        .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_NCHW, ge::FORMAT_NCHW)
        .TilingData(tilingData)
        .Workspace(ws)
        .Build();
}

void InitPostTilingSut(optiling::conv_ops_tiling::Conv2dBaseTiling& sut,
                       optiling::conv_ops_tiling::ConvTilingParseInfo& opInfo, gert::TilingContext* ctx)
{
    sut.opInfo_ = &opInfo;
    sut.paramInfo_.nodeType = "Conv2DV2";
    sut.paramInfo_.paramsIdxVec = {{0, 1, 2, 3, 4}, {0, 1, 2, 3, 4}, {0, 1, 2, 3, 4}};
    sut.shapeInfo_.batch = 1;
    sut.shapeInfo_.ci = 16;
    sut.shapeInfo_.hi = 16;
    sut.shapeInfo_.wi = 16;
    sut.shapeInfo_.co = 16;
    sut.shapeInfo_.kh = 3;
    sut.shapeInfo_.kw = 3;
    sut.shapeInfo_.ho = 16;
    sut.shapeInfo_.wo = 16;
    sut.attrInfo_.groups = 1;
    sut.descInfo_.fMapDtype = ge::DT_FLOAT16;
    sut.descInfo_.weightDtype = ge::DT_FLOAT16;
    sut.descInfo_.outDtype = ge::DT_FLOAT16;
    sut.descInfo_.fMapFormat = ge::FORMAT_NCHW;
    sut.descInfo_.weightFormat = ge::FORMAT_NCHW;
    sut.descInfo_.outFormat = ge::FORMAT_NCHW;
    sut.convOpsConstParams_.m0 = CUBE_M0;
    sut.convOpsConstParams_.k0 = CUBE_K0_16;
    sut.convOpsConstParams_.n0 = CUBE_N0;
    sut.numBlocksRes = {1, 1, 1, 1, 1, 0};
    (void)ctx;
}

void SetMNFullLoadKeyParams(optiling::conv_ops_tiling::Conv2dBaseTiling& sut)
{
    sut.descInfo_.weightFormat = ge::FORMAT_NHWC;
    sut.tilingKeyPara_.fmpTiling = optiling::conv_ops_tiling::ONLY_M_FULLLOAD_AL1_AL0;
    sut.tilingKeyPara_.weightTiling = optiling::conv_ops_tiling::ONLY_N_FULLLOAD_BL1_BL0;
    sut.tilingKeyPara_.l1PingPong = optiling::conv_ops_tiling::L1_PB_ALL_OPEN;
    sut.tilingKeyPara_.l0PingPong = optiling::conv_ops_tiling::L0_PB_ALL_OPEN;
    sut.tilingKeyPara_.groupType = static_cast<uint64_t>(optiling::conv_ops_tiling::ConvGroupType::NORMAL_CONV);
    sut.tilingKeyPara_.weightUbTrans = optiling::conv_ops_tiling::WEIGHT_UB_TRANS_CLOSE;
    sut.tilingKeyPara_.fmapCppyMode = optiling::conv_ops_tiling::FMAP_LOAD3D_MODE;
    sut.tilingKeyPara_.innerBatch = optiling::conv_ops_tiling::CONV_INNER_BATCH_SINGLE;
    sut.flagInfo_.mSplitModeFlag = true;
    sut.flagInfo_.enableC04Flag = false;
    sut.flagInfo_.disContinuousFlag = false;
    sut.flagInfo_.isASWT = false;
    sut.tilingData_.set_iterateMNOrder(static_cast<uint64_t>(optiling::conv_ops_tiling::IterateMNOrder::ITER_M_FST));
}
} // namespace

TEST_F(Conv2dTilingKeyExtra, post_tiling_fullmn_small_buffer)
{
    optiling::conv_ops_tiling::ConvTilingParseInfo opInfo;
    fe::PlatFormInfos platformInfo;
    auto tilingData = gert::TilingData::CreateCap(1);
    auto wsHolder = gert::ContinuousVector::Create<size_t>(MEM_SIZE_4K);
    auto ws = reinterpret_cast<gert::ContinuousVector*>(wsHolder.get());
    auto holder = BuildPostTilingContext(opInfo, platformInfo, tilingData.get(), ws);
    auto ctx = holder.GetContext<gert::TilingContext>();
    optiling::conv_ops_tiling::Conv2dBaseTiling sut(ctx);
    InitPostTilingSut(sut, opInfo, ctx);
    sut.opInfo_->npuArch = NpuArch::DAV_9201;
    SetMNFullLoadKeyParams(sut);
    EXPECT_TRUE(sut.IsMNFullLoadKPartia());
    EXPECT_EQ(sut.PostTiling(), ge::GRAPH_FAILED);
}

TEST_F(Conv2dTilingKeyExtra, post_tiling_fullmn_success)
{
    optiling::conv_ops_tiling::ConvTilingParseInfo opInfo;
    fe::PlatFormInfos platformInfo;
    auto tilingData = gert::TilingData::CreateCap(MEM_SIZE_4K);
    auto wsHolder = gert::ContinuousVector::Create<size_t>(MEM_SIZE_4K);
    auto ws = reinterpret_cast<gert::ContinuousVector*>(wsHolder.get());
    auto holder = BuildPostTilingContext(opInfo, platformInfo, tilingData.get(), ws);
    auto ctx = holder.GetContext<gert::TilingContext>();
    optiling::conv_ops_tiling::Conv2dBaseTiling sut(ctx);
    InitPostTilingSut(sut, opInfo, ctx);
    sut.opInfo_->npuArch = NpuArch::DAV_9201;
    SetMNFullLoadKeyParams(sut);
    EXPECT_EQ(sut.PostTiling(), ge::GRAPH_SUCCESS);
    EXPECT_EQ(ctx->GetRawTilingData()->GetDataSize(), sizeof(Ops::NN::Conv2dV2::Conv2DMNFullLoadKPartialTiling));
}

TEST_F(Conv2dTilingKeyExtra, post_tiling_aswt_5102_block_dim)
{
    optiling::conv_ops_tiling::ConvTilingParseInfo opInfo;
    fe::PlatFormInfos platformInfo;
    auto tilingData = gert::TilingData::CreateCap(MEM_SIZE_4K);
    auto wsHolder = gert::ContinuousVector::Create<size_t>(MEM_SIZE_4K);
    auto ws = reinterpret_cast<gert::ContinuousVector*>(wsHolder.get());
    auto holder = BuildPostTilingContext(opInfo, platformInfo, tilingData.get(), ws);
    auto ctx = holder.GetContext<gert::TilingContext>();
    optiling::conv_ops_tiling::Conv2dBaseTiling sut(ctx);
    InitPostTilingSut(sut, opInfo, ctx);
    sut.opInfo_->npuArch = NpuArch::DAV_5102;
    sut.opInfo_->aicoreNum = 4;
    sut.flagInfo_.isASWT = true;
    sut.tilingData_.set_innerBatch(1);
    sut.tilingData_.set_hoL1(16);
    sut.tilingData_.set_nBL1(16);
    EXPECT_EQ(sut.PostTiling(), ge::GRAPH_SUCCESS);
    EXPECT_EQ(ctx->GetBlockDim(), 4U);
}

TEST_F(Conv2dTilingKeyExtra, calc_aswt_strategy_5102_enable)
{
    optiling::conv_ops_tiling::ConvTilingParseInfo opInfo;
    fe::PlatFormInfos platformInfo;
    auto tilingData = gert::TilingData::CreateCap(MEM_SIZE_4K);
    auto wsHolder = gert::ContinuousVector::Create<size_t>(MEM_SIZE_4K);
    auto ws = reinterpret_cast<gert::ContinuousVector*>(wsHolder.get());
    auto holder = BuildPostTilingContext(opInfo, platformInfo, tilingData.get(), ws);
    auto ctx = holder.GetContext<gert::TilingContext>();
    optiling::conv_ops_tiling::Conv2dBaseTiling sut(ctx);
    InitPostTilingSut(sut, opInfo, ctx);
    sut.opInfo_->npuArch = NpuArch::DAV_5102;
    sut.opInfo_->aicoreNum = 32;
    sut.flagInfo_.mSplitModeFlag = true;
    sut.flagInfo_.convGroupType = optiling::conv_ops_tiling::ConvGroupType::NORMAL_CONV;
    sut.attrInfo_.groups = 1;
    sut.descInfo_.weightFormat = ge::FORMAT_FRACTAL_Z;
    sut.descInfo_.fMapFormat = ge::FORMAT_NCHW;
    sut.descInfo_.outFormat = ge::FORMAT_NCHW;
    sut.tilingData_.set_kAL1(288);
    sut.tilingData_.set_kBL1(288);
    sut.tilingData_.set_nBL1(16);
    sut.tilingData_.set_nL0(16);
    sut.tilingData_.set_hoL1(16);
    sut.tilingData_.set_hoL0(8);
    sut.tilingData_.set_singleCoreHo(16);
    sut.tilingData_.set_innerBatch(1);
    sut.tilingData_.set_pBufferFlag(0);
    sut.numBlocksRes = {1, 1, 1, 1, 1, 0};
    sut.CalcASWTStrategy();
    EXPECT_TRUE(sut.flagInfo_.isASWT);
    EXPECT_GT(sut.tilingData_.get_mWindows(), 0U);
}
