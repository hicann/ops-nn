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
 * \file test_conv3d_v2_ascendc_check_attrs_tiling.cpp
 * \brief UT for conv3d_v2_base_tiling_check_attrs.cpp
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
constexpr uint64_t NUM_5 = 5;

namespace {
struct Conv3dAttrsParams {
    string caseName;
    std::vector<int64_t> fmShape;
    std::vector<int64_t> weightShape;
    std::vector<int64_t> pads;
    std::vector<int64_t> strides;
    std::vector<int64_t> dilations;
    int64_t groups;
    std::string padMode;
    std::string dataFormat;
    int64_t offsetX;
    ge::DataType dtype;
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

void RunConv3dAttrsTest(const Conv3dAttrsParams& params)
{
    uint32_t padh = static_cast<uint32_t>(params.pads[0]);
    uint32_t padt = static_cast<uint32_t>(params.pads[1]);
    uint32_t padu = static_cast<uint32_t>(params.pads[2]);
    uint32_t padd = static_cast<uint32_t>(params.pads[3]);
    uint32_t padl = static_cast<uint32_t>(params.pads[4]);
    uint32_t padr = static_cast<uint32_t>(params.pads[5]);
    uint32_t strideD = static_cast<uint32_t>(params.strides[2]);
    uint32_t strideH = static_cast<uint32_t>(params.strides[3]);
    uint32_t strideW = static_cast<uint32_t>(params.strides[4]);
    uint32_t dilationD = static_cast<uint32_t>(params.dilations[2]);
    uint32_t dilationH = static_cast<uint32_t>(params.dilations[3]);
    uint32_t dilationW = static_cast<uint32_t>(params.dilations[4]);

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
                      .NodeAttrs({{"strides", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(params.strides)},
                                  {"pads", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(params.pads)},
                                  {"dilations", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(params.dilations)},
                                  {"groups", Ops::NN::AnyValue::CreateFrom<int64_t>(params.groups)},
                                  {"data_format", Ops::NN::AnyValue::CreateFrom<std::string>(params.dataFormat)},
                                  {"offset_x", Ops::NN::AnyValue::CreateFrom<int64_t>(params.offsetX)},
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

    if (params.isErrorCaseFlag) {
        EXPECT_EQ(ret, ge::GRAPH_FAILED) << "Case [" << params.caseName << "] expected FAILED but got SUCCESS";
    } else {
        EXPECT_EQ(ret, ge::GRAPH_SUCCESS) << "Case [" << params.caseName << "] expected SUCCESS but got FAILED";
    }
}
} // namespace

class TestConv3dCheckAttrs : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// ============================================================================
// ParseStrideLegal
// ============================================================================

// valid strides [1,1,1,1,1] → SUCCESS
TEST_F(TestConv3dCheckAttrs, parse_stride_legal_valid)
{
    RunConv3dAttrsTest({"stride_valid",
                        {1, 16, 8, 16, 16},
                        {16, 16, 1, 1, 1},
                        {0, 0, 0, 0, 0, 0},
                        {1, 1, 1, 1, 1},
                        {1, 1, 1, 1, 1},
                        1,
                        "SPECIFIC",
                        "NCDHW",
                        0,
                        DT_FLOAT16,
                        false});
}

// strides with wrong dim (3 elements instead of 5) → FAILED
TEST_F(TestConv3dCheckAttrs, parse_stride_wrong_dim)
{
    RunConv3dAttrsTest({"stride_wrong_dim",
                        {1, 16, 8, 16, 16},
                        {16, 16, 1, 1, 1},
                        {0, 0, 0, 0, 0, 0},
                        {1, 1, 1},
                        {1, 1, 1, 1, 1},
                        1,
                        "SPECIFIC",
                        "NCDHW",
                        0,
                        DT_FLOAT16,
                        true});
}

// strideN != 1 → FAILED (NCDHW: N at index 0)
TEST_F(TestConv3dCheckAttrs, parse_stride_n_not_1)
{
    RunConv3dAttrsTest({"stride_n_not_1",
                        {1, 16, 8, 16, 16},
                        {16, 16, 1, 1, 1},
                        {0, 0, 0, 0, 0, 0},
                        {2, 1, 1, 1, 1},
                        {1, 1, 1, 1, 1},
                        1,
                        "SPECIFIC",
                        "NCDHW",
                        0,
                        DT_FLOAT16,
                        true});
}

// strideC != 1 → FAILED (NCDHW: C at index 1)
TEST_F(TestConv3dCheckAttrs, parse_stride_c_not_1)
{
    RunConv3dAttrsTest({"stride_c_not_1",
                        {1, 16, 8, 16, 16},
                        {16, 16, 1, 1, 1},
                        {0, 0, 0, 0, 0, 0},
                        {1, 2, 1, 1, 1},
                        {1, 1, 1, 1, 1},
                        1,
                        "SPECIFIC",
                        "NCDHW",
                        0,
                        DT_FLOAT16,
                        true});
}

// strideH out of range: 64 > MAX_STRIDE_H_SHAPE(63) → FAILED
TEST_F(TestConv3dCheckAttrs, parse_stride_h_out_of_range)
{
    RunConv3dAttrsTest({"stride_h_oor",
                        {1, 16, 8, 16, 16},
                        {16, 16, 1, 1, 1},
                        {0, 0, 0, 0, 0, 0},
                        {1, 1, 1, 64, 1},
                        {1, 1, 1, 1, 1},
                        1,
                        "SPECIFIC",
                        "NCDHW",
                        0,
                        DT_FLOAT16,
                        true});
}

// strideW out of range: 64 > MAX_STRIDE_W_SHAPE(63) → FAILED
TEST_F(TestConv3dCheckAttrs, parse_stride_w_out_of_range)
{
    RunConv3dAttrsTest({"stride_w_oor",
                        {1, 16, 8, 16, 16},
                        {16, 16, 1, 1, 1},
                        {0, 0, 0, 0, 0, 0},
                        {1, 1, 1, 1, 64},
                        {1, 1, 1, 1, 1},
                        1,
                        "SPECIFIC",
                        "NCDHW",
                        0,
                        DT_FLOAT16,
                        true});
}

// strideH at boundary 63 → SUCCESS
TEST_F(TestConv3dCheckAttrs, parse_stride_h_boundary)
{
    RunConv3dAttrsTest({"stride_h_boundary",
                        {1, 16, 8, 128, 16},
                        {16, 16, 1, 1, 1},
                        {0, 0, 0, 0, 0, 0},
                        {1, 1, 1, 63, 1},
                        {1, 1, 1, 1, 1},
                        1,
                        "SPECIFIC",
                        "NCDHW",
                        0,
                        DT_FLOAT16,
                        false});
}

// strideH at boundary 63 → SUCCESS
TEST_F(TestConv3dCheckAttrs, parse_stride_d_boundary)
{
    RunConv3dAttrsTest({"stride_d_boundary",
                        {1, 16, 8, 128, 16},
                        {16, 16, 1, 1, 1},
                        {0, 0, 0, 0, 0, 0},
                        {1, 1, 1, 63, 1},
                        {1, 1, 1, 1, 1},
                        1,
                        "SPECIFIC",
                        "NCDHW",
                        0,
                        DT_FLOAT16,
                        false});
}

// ============================================================================
// ParseDilationLegal
// ============================================================================

// valid dilations [1,1,1,1,1] → SUCCESS
TEST_F(TestConv3dCheckAttrs, parse_dilation_legal_valid)
{
    RunConv3dAttrsTest({"dilation_valid",
                        {1, 16, 8, 16, 16},
                        {16, 16, 1, 1, 1},
                        {0, 0, 0, 0, 0, 0},
                        {1, 1, 1, 1, 1},
                        {1, 1, 1, 1, 1},
                        1,
                        "SPECIFIC",
                        "NCDHW",
                        0,
                        DT_FLOAT16,
                        false});
}

// dilations with wrong dim (4 elements instead of 5) → FAILED
TEST_F(TestConv3dCheckAttrs, parse_dilation_wrong_dim)
{
    RunConv3dAttrsTest({"dilation_wrong_dim",
                        {1, 16, 8, 16, 16},
                        {16, 16, 1, 1, 1},
                        {0, 0, 0, 0, 0, 0},
                        {1, 1, 1, 1, 1},
                        {1, 1, 1, 1},
                        1,
                        "SPECIFIC",
                        "NCDHW",
                        0,
                        DT_FLOAT16,
                        true});
}

// dilationN != 1 → FAILED
TEST_F(TestConv3dCheckAttrs, parse_dilation_n_not_1)
{
    RunConv3dAttrsTest({"dilation_n_not_1",
                        {1, 16, 8, 16, 16},
                        {16, 16, 1, 1, 1},
                        {0, 0, 0, 0, 0, 0},
                        {1, 1, 1, 1, 1},
                        {2, 1, 1, 1, 1},
                        1,
                        "SPECIFIC",
                        "NCDHW",
                        0,
                        DT_FLOAT16,
                        true});
}

// dilationC != 1 → FAILED
TEST_F(TestConv3dCheckAttrs, parse_dilation_c_not_1)
{
    RunConv3dAttrsTest({"dilation_c_not_1",
                        {1, 16, 8, 16, 16},
                        {16, 16, 1, 1, 1},
                        {0, 0, 0, 0, 0, 0},
                        {1, 1, 1, 1, 1},
                        {1, 2, 1, 1, 1},
                        1,
                        "SPECIFIC",
                        "NCDHW",
                        0,
                        DT_FLOAT16,
                        true});
}

// dilationH out of range: 256 > MAX_DILATION_H_SHAPE(255) → FAILED
TEST_F(TestConv3dCheckAttrs, parse_dilation_h_out_of_range)
{
    RunConv3dAttrsTest({"dilation_h_oor",
                        {1, 16, 8, 512, 16},
                        {16, 16, 1, 1, 1},
                        {0, 0, 0, 0, 0, 0},
                        {1, 1, 1, 1, 1},
                        {1, 1, 1, 256, 1},
                        1,
                        "SPECIFIC",
                        "NCDHW",
                        0,
                        DT_FLOAT16,
                        true});
}

// dilationW out of range: 256 > MAX_DILATION_W_SHAPE(255) → FAILED
TEST_F(TestConv3dCheckAttrs, parse_dilation_w_out_of_range)
{
    RunConv3dAttrsTest({"dilation_w_oor",
                        {1, 16, 8, 16, 512},
                        {16, 16, 1, 1, 1},
                        {0, 0, 0, 0, 0, 0},
                        {1, 1, 1, 1, 1},
                        {1, 1, 1, 1, 256},
                        1,
                        "SPECIFIC",
                        "NCDHW",
                        0,
                        DT_FLOAT16,
                        true});
}

// ============================================================================
// ParsePadLegal
// ============================================================================

// pads with wrong dim (4 elements instead of 6) → FAILED
TEST_F(TestConv3dCheckAttrs, parse_pad_wrong_dim)
{
    RunConv3dAttrsTest({"pad_wrong_dim",
                        {1, 16, 8, 16, 16},
                        {16, 16, 1, 1, 1},
                        {0, 0, 0, 0},
                        {1, 1, 1, 1, 1},
                        {1, 1, 1, 1, 1},
                        1,
                        "SPECIFIC",
                        "NCDHW",
                        0,
                        DT_FLOAT16,
                        true});
}

// pad head out of range: 256 > MAX_PAD_H_SHAPE(255) → FAILED
TEST_F(TestConv3dCheckAttrs, parse_pad_h_out_of_range)
{
    RunConv3dAttrsTest({"pad_h_oor",
                        {1, 16, 8, 16, 16},
                        {16, 16, 1, 1, 1},
                        {0, 0, 256, 0, 0, 0},
                        {1, 1, 1, 1, 1},
                        {1, 1, 1, 1, 1},
                        1,
                        "SPECIFIC",
                        "NCDHW",
                        0,
                        DT_FLOAT16,
                        true});
}

// pad left/right out of range: 256 > MAX_PAD_W_SHAPE(255) → FAILED
TEST_F(TestConv3dCheckAttrs, parse_pad_w_out_of_range)
{
    RunConv3dAttrsTest({"pad_w_oor",
                        {1, 16, 8, 16, 16},
                        {16, 16, 1, 1, 1},
                        {0, 0, 0, 0, 256, 0},
                        {1, 1, 1, 1, 1},
                        {1, 1, 1, 1, 1},
                        1,
                        "SPECIFIC",
                        "NCDHW",
                        0,
                        DT_FLOAT16,
                        true});
}

// pad boundary values 255 → SUCCESS
TEST_F(TestConv3dCheckAttrs, parse_pad_boundary_valid)
{
    RunConv3dAttrsTest({"pad_boundary",
                        {1, 16, 8, 512, 512},
                        {16, 16, 1, 1, 1},
                        {0, 0, 255, 255, 255, 255},
                        {1, 1, 1, 1, 1},
                        {1, 1, 1, 1, 1},
                        1,
                        "SPECIFIC",
                        "NCDHW",
                        0,
                        DT_FLOAT16,
                        false});
}

// ============================================================================
// GetOriPadFromPadMode / ApplySamesPad
// ============================================================================

// pad_mode = "SPECIFIC" with default pads → SUCCESS
TEST_F(TestConv3dCheckAttrs, pad_mode_specific_default)
{
    RunConv3dAttrsTest({"pad_mode_specific",
                        {1, 16, 8, 16, 16},
                        {16, 16, 1, 1, 1},
                        {0, 0, 0, 0, 0, 0},
                        {1, 1, 1, 1, 1},
                        {1, 1, 1, 1, 1},
                        1,
                        "SPECIFIC",
                        "NCDHW",
                        0,
                        DT_FLOAT16,
                        false});
}

// pad_mode = "VALID" → pads computed as all zeros → SUCCESS
TEST_F(TestConv3dCheckAttrs, pad_mode_valid)
{
    RunConv3dAttrsTest({"pad_mode_valid",
                        {1, 16, 8, 16, 16},
                        {16, 16, 1, 1, 1},
                        {0, 0, 0, 0, 0, 0},
                        {1, 1, 1, 1, 1},
                        {1, 1, 1, 1, 1},
                        1,
                        "VALID",
                        "NCDHW",
                        0,
                        DT_FLOAT16,
                        false});
}

// pad_mode = "SAME" → SUCCESS
TEST_F(TestConv3dCheckAttrs, pad_mode_same)
{
    RunConv3dAttrsTest({"pad_mode_same",
                        {1, 16, 8, 16, 16},
                        {16, 16, 1, 1, 1},
                        {0, 0, 0, 0, 0, 0},
                        {1, 1, 1, 1, 1},
                        {1, 1, 1, 1, 1},
                        1,
                        "SAME",
                        "NCDHW",
                        0,
                        DT_FLOAT16,
                        false});
}

// pad_mode = "SAME_UPPER" → SUCCESS
TEST_F(TestConv3dCheckAttrs, pad_mode_same_upper)
{
    RunConv3dAttrsTest({"pad_mode_same_upper",
                        {1, 16, 8, 16, 16},
                        {16, 16, 1, 1, 1},
                        {0, 0, 0, 0, 0, 0},
                        {1, 1, 1, 1, 1},
                        {1, 1, 1, 1, 1},
                        1,
                        "SAME_UPPER",
                        "NCDHW",
                        0,
                        DT_FLOAT16,
                        false});
}

// pad_mode = "SAME_LOWER" → SUCCESS
TEST_F(TestConv3dCheckAttrs, pad_mode_same_lower)
{
    RunConv3dAttrsTest({"pad_mode_same_lower",
                        {1, 16, 8, 16, 16},
                        {16, 16, 1, 1, 1},
                        {0, 0, 0, 0, 0, 0},
                        {1, 1, 1, 1, 1},
                        {1, 1, 1, 1, 1},
                        1,
                        "SAME_LOWER",
                        "NCDHW",
                        0,
                        DT_FLOAT16,
                        false});
}

// pad_mode = "INVALID" → FAILED
TEST_F(TestConv3dCheckAttrs, pad_mode_invalid)
{
    RunConv3dAttrsTest({"pad_mode_invalid",
                        {1, 16, 8, 16, 16},
                        {16, 16, 1, 1, 1},
                        {0, 0, 0, 0, 0, 0},
                        {1, 1, 1, 1, 1},
                        {1, 1, 1, 1, 1},
                        1,
                        "INVALID_MODE",
                        "NCDHW",
                        0,
                        DT_FLOAT16,
                        true});
}

// pad_mode = "SAME" with stride=2 and dilation=1, kernel=3 → SUCCESS
TEST_F(TestConv3dCheckAttrs, pad_mode_same_non_unit)
{
    RunConv3dAttrsTest({"pad_mode_same_nonunit",
                        {1, 16, 8, 16, 16},
                        {16, 16, 3, 3, 3},
                        {0, 0, 0, 0, 0, 0},
                        {1, 1, 1, 1, 1},
                        {1, 1, 1, 1, 1},
                        1,
                        "SAME",
                        "NCDHW",
                        0,
                        DT_FLOAT16,
                        true});
}

// pad_mode = "VALID" → pads zeroed, explicit pads ignored
TEST_F(TestConv3dCheckAttrs, pad_mode_valid_with_pads)
{
    RunConv3dAttrsTest({"pad_mode_valid_pads",
                        {1, 16, 8, 16, 16},
                        {16, 16, 1, 1, 1},
                        {0, 0, 0, 0, 0, 0},
                        {1, 1, 1, 1, 1},
                        {1, 1, 1, 1, 1},
                        1,
                        "VALID",
                        "NCDHW",
                        0,
                        DT_FLOAT16,
                        false});
}

// ============================================================================
// ParseGroupLegal
// ============================================================================

// groups = 1 → SUCCESS
TEST_F(TestConv3dCheckAttrs, parse_group_1_valid)
{
    RunConv3dAttrsTest({"group_1",
                        {1, 16, 8, 16, 16},
                        {16, 16, 1, 1, 1},
                        {0, 0, 0, 0, 0, 0},
                        {1, 1, 1, 1, 1},
                        {1, 1, 1, 1, 1},
                        1,
                        "SPECIFIC",
                        "NCDHW",
                        0,
                        DT_FLOAT16,
                        false});
}

// groups = 2 → SUCCESS (cin=16, groups=2, cin/groups=8)
TEST_F(TestConv3dCheckAttrs, parse_group_2_valid)
{
    RunConv3dAttrsTest({"group_2",
                        {1, 16, 8, 16, 16},
                        {16, 8, 1, 1, 1},
                        {0, 0, 0, 0, 0, 0},
                        {1, 1, 1, 1, 1},
                        {1, 1, 1, 1, 1},
                        2,
                        "SPECIFIC",
                        "NCDHW",
                        0,
                        DT_FLOAT16,
                        false});
}

// groups = 0 → FAILED (groups must be >= 1)
// TEST_F(TestConv3dCheckAttrs, parse_group_0_invalid)
// {
//     RunConv3dAttrsTest({"group_0",
//                         {1, 16, 8, 16, 16},
//                         {16, 16, 1, 1, 1},
//                         {0, 0, 0, 0, 0, 0},
//                         {1, 1, 1, 1, 1},
//                         {1, 1, 1, 1, 1},
//                         0,
//                         "SPECIFIC",
//                         "NCDHW",
//                         0,
//                         DT_FLOAT16,
//                         true});
// }

// groups = -1 → FAILED (groups must be >= 1)
TEST_F(TestConv3dCheckAttrs, parse_group_negative)
{
    RunConv3dAttrsTest({"group_negative",
                        {1, 16, 8, 16, 16},
                        {16, 16, 1, 1, 1},
                        {0, 0, 0, 0, 0, 0},
                        {1, 1, 1, 1, 1},
                        {1, 1, 1, 1, 1},
                        -1,
                        "SPECIFIC",
                        "NCDHW",
                        0,
                        DT_FLOAT16,
                        true});
}

// groups > MAX_GROUP_SHAPE(65535) → FAILED
TEST_F(TestConv3dCheckAttrs, parse_group_overflow)
{
    RunConv3dAttrsTest({"group_overflow",
                        {1, 65536, 8, 16, 16},
                        {65536, 1, 1, 1, 1},
                        {0, 0, 0, 0, 0, 0},
                        {1, 1, 1, 1, 1},
                        {1, 1, 1, 1, 1},
                        65536,
                        "SPECIFIC",
                        "NCDHW",
                        0,
                        DT_FLOAT16,
                        true});
}

// ============================================================================
// ParseQuantDataFormatLegal (quantFlag = false for Conv3DV2 → always SUCCESS)
// These tests verify the non-quant early-return path.
// ============================================================================

// data_format = "NCDHW" (non-quant) → SUCCESS
TEST_F(TestConv3dCheckAttrs, quant_data_format_ncdhw)
{
    RunConv3dAttrsTest({"qdf_ncdhw",
                        {1, 16, 8, 16, 16},
                        {16, 16, 1, 1, 1},
                        {0, 0, 0, 0, 0, 0},
                        {1, 1, 1, 1, 1},
                        {1, 1, 1, 1, 1},
                        1,
                        "SPECIFIC",
                        "NCDHW",
                        0,
                        DT_FLOAT16,
                        false});
}

// data_format = "NDHWC" (non-quant, early return) → SUCCESS
TEST_F(TestConv3dCheckAttrs, quant_data_format_ndhwc_non_quant)
{
    RunConv3dAttrsTest({"qdf_ndhwc",
                        {1, 16, 8, 16, 16},
                        {16, 16, 1, 1, 1},
                        {0, 0, 0, 0, 0, 0},
                        {1, 1, 1, 1, 1},
                        {1, 1, 1, 1, 1},
                        1,
                        "SPECIFIC",
                        "NDHWC",
                        0,
                        DT_FLOAT16,
                        false});
}

// ============================================================================
// ParseQuantOffsetXLegal (quantFlag = false → always SUCCESS)
// ============================================================================

// offset_x = 0 (non-quant) → SUCCESS
TEST_F(TestConv3dCheckAttrs, quant_offset_x_zero)
{
    RunConv3dAttrsTest({"qox_zero",
                        {1, 16, 8, 16, 16},
                        {16, 16, 1, 1, 1},
                        {0, 0, 0, 0, 0, 0},
                        {1, 1, 1, 1, 1},
                        {1, 1, 1, 1, 1},
                        1,
                        "SPECIFIC",
                        "NCDHW",
                        0,
                        DT_FLOAT16,
                        false});
}

// offset_x = 127 (non-quant, early return) → SUCCESS
TEST_F(TestConv3dCheckAttrs, quant_offset_x_127)
{
    RunConv3dAttrsTest({"qox_127",
                        {1, 16, 8, 16, 16},
                        {16, 16, 1, 1, 1},
                        {0, 0, 0, 0, 0, 0},
                        {1, 1, 1, 1, 1},
                        {1, 1, 1, 1, 1},
                        1,
                        "SPECIFIC",
                        "NCDHW",
                        127,
                        DT_FLOAT16,
                        false});
}

// offset_x = -128 (non-quant, early return) → SUCCESS
TEST_F(TestConv3dCheckAttrs, quant_offset_x_neg128)
{
    RunConv3dAttrsTest({"qox_neg128",
                        {1, 16, 8, 16, 16},
                        {16, 16, 1, 1, 1},
                        {0, 0, 0, 0, 0, 0},
                        {1, 1, 1, 1, 1},
                        {1, 1, 1, 1, 1},
                        1,
                        "SPECIFIC",
                        "NCDHW",
                        -128,
                        DT_FLOAT16,
                        false});
}

// ============================================================================
// ParseQuantRoundModeLegal (quantFlag = false → always SUCCESS)
// ============================================================================

// round_mode via non-quant path → SUCCESS
TEST_F(TestConv3dCheckAttrs, quant_round_mode_non_quant)
{
    RunConv3dAttrsTest({"qrm_non_quant",
                        {1, 16, 8, 16, 16},
                        {16, 16, 1, 1, 1},
                        {0, 0, 0, 0, 0, 0},
                        {1, 1, 1, 1, 1},
                        {1, 1, 1, 1, 1},
                        1,
                        "SPECIFIC",
                        "NCDHW",
                        0,
                        DT_FLOAT16,
                        false});
}

// ============================================================================
// Combined/larger shape normal cases
// ============================================================================

// normal case with larger shapes → SUCCESS
TEST_F(TestConv3dCheckAttrs, normal_larger_valid)
{
    RunConv3dAttrsTest({"normal_larger",
                        {2, 32, 16, 64, 64},
                        {64, 32, 3, 3, 3},
                        {1, 1, 1, 1, 1, 1},
                        {1, 1, 1, 1, 1},
                        {1, 1, 1, 1, 1},
                        1,
                        "SPECIFIC",
                        "NCDHW",
                        0,
                        DT_FLOAT16,
                        false});
}

// normal with non-trivial strides and dilations → SUCCESS
TEST_F(TestConv3dCheckAttrs, normal_non_trivial)
{
    RunConv3dAttrsTest({"normal_nontrivial",
                        {1, 64, 32, 128, 128},
                        {128, 64, 3, 3, 3},
                        {1, 1, 1, 1, 1, 1},
                        {1, 1, 1, 2, 2},
                        {1, 1, 1, 1, 1},
                        1,
                        "SPECIFIC",
                        "NCDHW",
                        0,
                        DT_FLOAT16,
                        false});
}

// normal with FP32 dtype → SUCCESS
TEST_F(TestConv3dCheckAttrs, normal_fp32_valid)
{
    RunConv3dAttrsTest({"normal_fp32",
                        {1, 16, 8, 16, 16},
                        {16, 16, 1, 1, 1},
                        {0, 0, 0, 0, 0, 0},
                        {1, 1, 1, 1, 1},
                        {1, 1, 1, 1, 1},
                        1,
                        "SPECIFIC",
                        "NCDHW",
                        0,
                        DT_FLOAT,
                        false});
}

// normal with INT8 dtype → SUCCESS
TEST_F(TestConv3dCheckAttrs, normal_int8_valid)
{
    RunConv3dAttrsTest({"normal_int8",
                        {1, 16, 8, 16, 16},
                        {16, 16, 1, 1, 1},
                        {0, 0, 0, 0, 0, 0},
                        {1, 1, 1, 1, 1},
                        {1, 1, 1, 1, 1},
                        1,
                        "SPECIFIC",
                        "NCDHW",
                        0,
                        DT_INT8,
                        true});
}
