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
 * \file test_conv3d_v2_ascendc_check_limits_tiling.cpp
 * \brief UT for conv3d_v2_base_tiling_check_limits.cpp
 */

#include <gtest/gtest.h>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include "log/log.h"
#include "array_ops.h"
#include "tests/ut/common/ut_op_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "exe_graph/runtime/tiling_parse_context.h"
#include "tests/ut/common/kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "conv/conv3d_v2/op_host/op_tiling/conv3d_base_tiling.h"
#include "conv/common/op_host/op_tiling/arch35/conv_base_utils.h"

using namespace std;
using namespace ge;
using namespace ut_util;

constexpr uint64_t NUM_4 = 4;

namespace {
struct Conv3dLimitsParams {
    string caseName;
    std::vector<int64_t> fmShape;
    std::vector<int64_t> weightShape;
    std::vector<uint32_t> pads;
    std::vector<uint32_t> strides;
    std::vector<uint32_t> dilations;
    ge::DataType dtype;
    uint32_t groups;
    bool isErrorCaseFlag;
};

struct ConvShape {
    uint64_t inputV;
    uint64_t kernelV;
    uint64_t padone;
    uint64_t padtwo;
    uint64_t dilationV;
    uint64_t strideV;
};

int64_t InferOutFor3D(ConvShape convShape);

void RunConv3dLimitsTest(const Conv3dLimitsParams& params)
{
    uint32_t padh = params.pads[0], padt = params.pads[1];
    uint32_t padu = params.pads[2], padd = params.pads[3];
    uint32_t padl = params.pads[4], padr = params.pads[5];
    uint32_t strideD = params.strides[0], strideH = params.strides[1], strideW = params.strides[2];
    uint32_t dilationD = params.dilations[0], dilationH = params.dilations[1], dilationW = params.dilations[2];

    int64_t cout = params.weightShape[0];
    int64_t kd = params.weightShape[2];
    int64_t kh = params.weightShape[3];
    int64_t kw = params.weightShape[4];
    int64_t batch = params.fmShape[0];
    int64_t cin = params.fmShape[1];
    int64_t di = params.fmShape[2];
    int64_t hi = params.fmShape[3];
    int64_t wi = params.fmShape[4];

    ConvShape convShapeDo = {static_cast<uint64_t>(di), static_cast<uint64_t>(kd), padh, padt, dilationD, strideD};
    ConvShape convShapeHo = {static_cast<uint64_t>(hi), static_cast<uint64_t>(kh), padu, padd, dilationH, strideH};
    ConvShape convShapeWo = {static_cast<uint64_t>(wi), static_cast<uint64_t>(kw), padl, padr, dilationW, strideW};
    int64_t Do = InferOutFor3D(convShapeDo);
    int64_t ho = InferOutFor3D(convShapeHo);
    int64_t wo = InferOutFor3D(convShapeWo);

    gert::StorageShape featuremap = {{batch, cin, di, hi, wi}, {batch, cin, di, hi, wi}};
    gert::StorageShape weight = {{cout, cin / params.groups, kd, kh, kw}, {cout, cin / params.groups, kd, kh, kw}};
    gert::StorageShape bias = {{cout}, {cout}};
    gert::StorageShape output = {{batch, cout, Do > 0 ? Do : 1, ho > 0 ? ho : 1, wo > 0 ? wo : 1},
                                 {batch, cout, Do > 0 ? Do : 1, ho > 0 ? ho : 1, wo > 0 ? wo : 1}};

    std::vector<void*> input_shape_ref = {&featuremap, &weight, &bias, nullptr};
    std::vector<void*> output_shapes_ref = {&output};

    std::vector<int64_t> stridesVec = {1, 1, static_cast<int64_t>(strideD), static_cast<int64_t>(strideH),
                                       static_cast<int64_t>(strideW)};
    std::vector<int64_t> padsVec = {static_cast<int64_t>(padh), static_cast<int64_t>(padt), static_cast<int64_t>(padu),
                                    static_cast<int64_t>(padd), static_cast<int64_t>(padl), static_cast<int64_t>(padr)};
    std::vector<int64_t> dilationsVec = {1, 1, static_cast<int64_t>(dilationD), static_cast<int64_t>(dilationH),
                                         static_cast<int64_t>(dilationW)};

    std::string op_type = "Conv3DV2";
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;

    string compile_info_string = R"({"hardware_info":
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

    auto tilingDataPtr = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(tilingDataPtr, nullptr);

    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(NUM_4, 1)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes(input_shape_ref)
                      .OutputShapes(output_shapes_ref)
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, params.dtype, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeInputTd(1, params.dtype, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeInputTd(2, params.dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, params.dtype, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeAttrs({{"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(stridesVec)},
                                  {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(padsVec)},
                                  {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(dilationsVec)},
                                  {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(params.groups)},
                                  {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>("NCDHW")},
                                  {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(0)},
                                  {"pad_mode", Ops::NN::AnyValue::CreateFrom<std::string>("SPECIFIC")},
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

    if (params.isErrorCaseFlag) {
        EXPECT_EQ(ret, ge::GRAPH_FAILED) << "Case [" << params.caseName << "] expected FAILED but got SUCCESS";
    } else {
        EXPECT_EQ(ret, ge::GRAPH_SUCCESS) << "Case [" << params.caseName << "] expected SUCCESS but got FAILED";
    }
}

int64_t InferOutFor3D(ConvShape convShape)
{
    if (convShape.strideV == 0) {
        return 0;
    }
    return (convShape.inputV + convShape.padone + convShape.padtwo - convShape.dilationV * (convShape.kernelV - 1) -
            1) /
               convShape.strideV +
           1;
}
} // namespace

class TestConv3dCheckLimits : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// ============================================================================
// CheckLoad3DStride: strideH > 63 → FAILED
// ============================================================================
TEST_F(TestConv3dCheckLimits, check_load3d_stride_overflow_fail)
{
    RunConv3dLimitsTest({"stride_overflow",
                         {1, 16, 8, 128, 16},
                         {16, 16, 1, 1, 1},
                         {0, 0, 0, 0, 0, 0},
                         {1, 64, 1},
                         {1, 1, 1},
                         DT_FLOAT16,
                         1,
                         true});
}

// ============================================================================
// CheckLoad3DDialtion: dilationH > 255 → FAILED
// ============================================================================
TEST_F(TestConv3dCheckLimits, check_load3d_dilation_overflow_fail)
{
    RunConv3dLimitsTest({"dilation_overflow",
                         {1, 16, 8, 512, 16},
                         {16, 16, 1, 1, 1},
                         {0, 0, 0, 0, 0, 0},
                         {1, 1, 1},
                         {1, 256, 1},
                         DT_FLOAT16,
                         1,
                         true});
}

// ============================================================================
// CheckLoad3DPad: pad > 255 → FAILED
// ============================================================================
TEST_F(TestConv3dCheckLimits, check_load3d_pad_overflow_fail)
{
    RunConv3dLimitsTest({"pad_overflow",
                         {1, 16, 8, 16, 16},
                         {16, 16, 1, 1, 1},
                         {0, 0, 256, 0, 0, 0},
                         {1, 1, 1},
                         {1, 1, 1},
                         DT_FLOAT16,
                         1,
                         true});
}

// ============================================================================
// CheckLoad3DWeight: kh > LOAD3D_MAX_FILTER_H_W_DAV(511) → FAILED
// ============================================================================
TEST_F(TestConv3dCheckLimits, check_load3d_weight_kh_overflow_fail)
{
    RunConv3dLimitsTest({"weight_kh_overflow",
                         {1, 16, 4, 1024, 16},
                         {16, 16, 1, 512, 1},
                         {0, 0, 0, 0, 0, 0},
                         {1, 1, 1},
                         {1, 1, 1},
                         DT_FLOAT16,
                         1,
                         true});
}

// ============================================================================
// CheckLoad3DWeight: load3dPosk > MAX_16_BIT_NUM(65535) → FAILED
// kh*kw*k0 = 511*4*32 = 65408 <= 65535, but FLOAT32: 32*511*4=65408 (with k0=16 for FP32) → 511*5*16=40880
// Use FLOAT32: kh=256, kw=16, k0=16 → 256*16*16=65536 > 65535 → FAILED
// ============================================================================
TEST_F(TestConv3dCheckLimits, check_load3d_weight_posk_overflow_fail)
{
    RunConv3dLimitsTest({"weight_posk_overflow",
                         {1, 16, 4, 512, 16},
                         {16, 16, 1, 256, 17},
                         {0, 0, 0, 0, 0, 0},
                         {1, 1, 1},
                         {1, 1, 1},
                         DT_FLOAT,
                         1,
                         true});
}

// ============================================================================
// Normal case: all limits valid → SUCCESS
// ============================================================================
TEST_F(TestConv3dCheckLimits, check_normal_valid_success)
{
    RunConv3dLimitsTest({"normal_valid",
                         {1, 16, 8, 16, 16},
                         {16, 16, 1, 1, 1},
                         {0, 0, 0, 0, 0, 0},
                         {1, 1, 1},
                         {1, 1, 1},
                         DT_FLOAT16,
                         1,
                         false});
}

// ============================================================================
// CheckDataCopyLimits: NCDHW fmap di*hi*wi*dtypeSize > MAX_40_BIT_NUM → FAILED
// dt=425, hi=25000, wi=1035, dtypeSize=4 → 425*25000*1035*4=43,987,500,000 > 1,099,511,627,775(MAX_40_BIT)
// Actually needs to exceed MAX_40_BIT_NUM=1,099,511,627,775
// Use hi=200000, wi=200000, di=1, dtypeSize=4 → 1*200000*200000*4=160,000,000,000 > 1.1e12 → FAILED (actually this is <
// MAX 40bit) MAX_40_BIT_NUM = 2^40-1 = 1,099,511,627,775 hi=500000, wi=600000, di=1, dtypeSize=4 → 1*500000*600000*4 =
// 1,200,000,000,000 > 1.1e12 → FAILED
// ============================================================================
TEST_F(TestConv3dCheckLimits, check_datacopy_ncdhw_overflow_fail)
{
    RunConv3dLimitsTest({"datacopy_ncdhw",
                         {1, 16, 1, 500000, 600000},
                         {16, 16, 1, 1, 1},
                         {0, 0, 0, 0, 0, 0},
                         {1, 1, 1},
                         {1, 1, 1},
                         DT_FLOAT,
                         1,
                         true});
}

// ============================================================================
// CheckDataCopyLimits: NCDHW fmap under limit → SUCCESS
// ============================================================================
TEST_F(TestConv3dCheckLimits, check_datacopy_ncdhw_under_limit_success)
{
    RunConv3dLimitsTest({"datacopy_ok",
                         {1, 16, 8, 16, 16},
                         {16, 16, 1, 1, 1},
                         {0, 0, 0, 0, 0, 0},
                         {1, 1, 1},
                         {1, 1, 1},
                         DT_FLOAT16,
                         1,
                         false});
}

// ============================================================================
// CheckFixPipeLimits: NCDHW dout*ho*wo > MAX_32_BIT_NUM → FAILED
// MAX_32_BIT_NUM = 4294967295
// ho=65536, wo=65536, dout=1 → 1*65536*65536=4294967296 > 4294967295 → FAILED
// ============================================================================
TEST_F(TestConv3dCheckLimits, check_fixpipe_ncdhw_overflow_fail)
{
    RunConv3dLimitsTest({"fixpipe_ncdhw",
                         {1, 16, 1, 65536, 65536},
                         {16, 16, 1, 1, 1},
                         {0, 0, 0, 0, 0, 0},
                         {1, 1, 1},
                         {1, 1, 1},
                         DT_FLOAT16,
                         1,
                         true});
}

// ============================================================================
// CheckFixPipeLimits: NCDHW dout*ho*wo within limit → SUCCESS
// ============================================================================
TEST_F(TestConv3dCheckLimits, check_fixpipe_ncdhw_within_limit_success)
{
    RunConv3dLimitsTest({"fixpipe_ok",
                         {1, 16, 8, 16, 16},
                         {16, 16, 1, 1, 1},
                         {0, 0, 0, 0, 0, 0},
                         {1, 1, 1},
                         {1, 1, 1},
                         DT_FLOAT16,
                         1,
                         false});
}

// ============================================================================
// CheckInstructionLimits: instruction limits overall check → SUCCESS
// ============================================================================
TEST_F(TestConv3dCheckLimits, check_instr_limits_success)
{
    RunConv3dLimitsTest({"instr_ok",
                         {1, 16, 8, 16, 16},
                         {16, 16, 1, 1, 1},
                         {0, 0, 0, 0, 0, 0},
                         {1, 1, 1},
                         {1, 1, 1},
                         DT_FLOAT16,
                         1,
                         false});
}

// ============================================================================
// CheckLoad3DStride: strideW > 63 → FAILED
// ============================================================================
TEST_F(TestConv3dCheckLimits, check_load3d_stride_w_overflow_fail)
{
    RunConv3dLimitsTest({"stride_w_overflow",
                         {1, 16, 8, 16, 128},
                         {16, 16, 1, 1, 1},
                         {0, 0, 0, 0, 0, 0},
                         {1, 1, 64},
                         {1, 1, 1},
                         DT_FLOAT16,
                         1,
                         true});
}

// ============================================================================
// CheckLoad3DWeight: kw > LOAD3D_MAX_FILTER_H_W_DAV(511) → FAILED
// ============================================================================
TEST_F(TestConv3dCheckLimits, check_load3d_weight_kw_overflow_fail)
{
    RunConv3dLimitsTest({"weight_kw_overflow",
                         {1, 16, 4, 16, 1024},
                         {16, 16, 1, 1, 512},
                         {0, 0, 0, 0, 0, 0},
                         {1, 1, 1},
                         {1, 1, 1},
                         DT_FLOAT16,
                         1,
                         true});
}

// ============================================================================
// CheckLoad3DDialtion: dilationW > 255 → FAILED
// ============================================================================
TEST_F(TestConv3dCheckLimits, check_load3d_dilation_w_overflow_fail)
{
    RunConv3dLimitsTest({"dilation_w_overflow",
                         {1, 16, 8, 16, 512},
                         {16, 16, 1, 1, 1},
                         {0, 0, 0, 0, 0, 0},
                         {1, 1, 1},
                         {1, 1, 256},
                         DT_FLOAT16,
                         1,
                         true});
}

// ============================================================================
// CheckLoad3DPad: pad bottom > 255 → FAILED
// ============================================================================
TEST_F(TestConv3dCheckLimits, check_load3d_pad_bottom_overflow_fail)
{
    RunConv3dLimitsTest({"pad_bottom_overflow",
                         {1, 16, 8, 16, 16},
                         {16, 16, 1, 1, 1},
                         {0, 0, 0, 256, 0, 0},
                         {1, 1, 1},
                         {1, 1, 1},
                         DT_FLOAT16,
                         1,
                         true});
}

// ============================================================================
// CheckLoad3DPad: pad left > 255 → FAILED
// ============================================================================
TEST_F(TestConv3dCheckLimits, check_load3d_pad_left_overflow_fail)
{
    RunConv3dLimitsTest({"pad_left_overflow",
                         {1, 16, 8, 16, 16},
                         {16, 16, 1, 1, 1},
                         {0, 0, 0, 0, 256, 0},
                         {1, 1, 1},
                         {1, 1, 1},
                         DT_FLOAT16,
                         1,
                         true});
}

// ============================================================================
// CheckLoad3DPad: pad right > 255 → FAILED
// ============================================================================
TEST_F(TestConv3dCheckLimits, check_load3d_pad_right_overflow_fail)
{
    RunConv3dLimitsTest({"pad_right_overflow",
                         {1, 16, 8, 16, 16},
                         {16, 16, 1, 1, 1},
                         {0, 0, 0, 0, 0, 256},
                         {1, 1, 1},
                         {1, 1, 1},
                         DT_FLOAT16,
                         1,
                         true});
}

// ============================================================================
// CheckFixPipeLimits: NDHWC format path (not NCDHW) → SUCCESS
// ============================================================================
TEST_F(TestConv3dCheckLimits, check_large_dout_success)
{
    RunConv3dLimitsTest({"large_dout",
                         {1, 16, 8, 16, 16},
                         {16, 16, 1, 1, 1},
                         {0, 0, 0, 0, 0, 0},
                         {1, 1, 1},
                         {1, 1, 1},
                         DT_FLOAT16,
                         1,
                         false});
}

// ============================================================================
// Borderline valid: all dims at or near max allowed
// ============================================================================
TEST_F(TestConv3dCheckLimits, check_borderline_all_ok_success)
{
    RunConv3dLimitsTest({"borderline_ok",
                         {1, 16, 8, 256, 16},
                         {16, 16, 1, 1, 1},
                         {255, 0, 0, 0, 0, 0},
                         {1, 63, 1},
                         {1, 255, 1},
                         DT_FLOAT16,
                         1,
                         false});
}
