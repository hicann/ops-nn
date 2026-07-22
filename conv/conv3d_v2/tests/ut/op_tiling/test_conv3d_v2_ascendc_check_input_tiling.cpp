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
 * \file test_conv3d_v2_ascendc_check_input_tiling.cpp
 * \brief UT for conv3d_v2_base_tiling_check_input.cpp
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
constexpr int64_t MAX_VAL = 1000000;

namespace {
struct ConvShape {
    uint64_t inputV;
    uint64_t kernelV;
    uint64_t padone;
    uint64_t padtwo;
    uint64_t dilationV;
    uint64_t strideV;
};

struct Conv3dInputParams {
    string caseName;
    std::vector<int64_t> fmShape;
    std::vector<int64_t> fmShapeOrigin;
    std::vector<int64_t> weightShape;
    std::vector<int64_t> weightStorageShape;
    std::vector<int64_t> biasShape;
    std::vector<int64_t> biasShapeOrigin;
    std::vector<int64_t> outputShape;
    std::vector<int64_t> outputShapeOrigin;
    std::vector<uint32_t> pads;
    std::vector<uint32_t> strides;
    std::vector<uint32_t> dilations;
    ge::DataType fmapDtype;
    ge::DataType weightDtype;
    ge::DataType biasDtype;
    ge::DataType scaleDtype;
    ge::DataType outputDtype;
    ge::Format weightOriginFormat;
    ge::Format weightStorageFormat;
    ge::Format biasFormat;
    ge::Format biasStorageFormat;
    ge::Format outputFormat;
    uint32_t groups;
    uint32_t hasBias;
    uint32_t hasScale;
    bool isErrorCaseFlag;
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

void RunConv3dInputTest(const Conv3dInputParams& params)
{
    uint32_t padh = params.pads[0], padt = params.pads[1];
    uint32_t padu = params.pads[2], padd = params.pads[3];
    uint32_t padl = params.pads[4], padr = params.pads[5];
    uint32_t strideD = params.strides[0], strideH = params.strides[1], strideW = params.strides[2];
    uint32_t dilationD = params.dilations[0], dilationH = params.dilations[1], dilationW = params.dilations[2];

    const auto& fs = params.fmShape;
    int64_t f0 = fs[0], f1 = fs.size() > 1 ? fs[1] : 1, f2 = fs.size() > 2 ? fs[2] : 1, f3 = fs.size() > 3 ? fs[3] : 1,
            f4 = fs.size() > 4 ? fs[4] : 1;
    gert::StorageShape featuremap = {{f0, f1, f2, f3, f4}, {f0, f1, f2, f3, f4}};

    const auto& ws = params.weightShape;
    const auto& wss = params.weightStorageShape.empty() ? params.weightShape : params.weightStorageShape;
    int64_t w0 = ws[0], w1 = ws.size() > 1 ? ws[1] : 1, w2 = ws.size() > 2 ? ws[2] : 1, w3 = ws.size() > 3 ? ws[3] : 1,
            w4 = ws.size() > 4 ? ws[4] : 1;
    int64_t wss0 = wss[0], wss1 = wss.size() > 1 ? wss[1] : 1, wss2 = wss.size() > 2 ? wss[2] : 1,
            wss3 = wss.size() > 3 ? wss[3] : 1, wss4 = wss.size() > 4 ? wss[4] : 1;
    gert::StorageShape weight = {{w0, w1, w2, w3, w4}, {wss0, wss1, wss2, wss3, wss4}};

    gert::StorageShape bias;
    bool hasBiasLocal = (params.hasBias == 1);
    if (hasBiasLocal) {
        const auto& bs = params.biasShape;
        int64_t b0 = bs[0], b1 = bs.size() > 1 ? bs[1] : 1, b2 = bs.size() > 2 ? bs[2] : 1,
                b3 = bs.size() > 3 ? bs[3] : 1, b4 = bs.size() > 4 ? bs[4] : 1;
        bias = {{b0, b1, b2, b3, b4}, {b0, b1, b2, b3, b4}};
    }

    gert::StorageShape scale;
    bool hasScaleLocal = (params.hasScale == 1);
    if (hasScaleLocal) {
        int64_t cout = params.weightShape[0];
        scale = {{cout}, {cout}};
    }

    std::vector<void*> input_shape_ref;
    input_shape_ref.push_back(&featuremap);
    input_shape_ref.push_back(&weight);
    input_shape_ref.push_back(hasBiasLocal ? &bias : nullptr);
    input_shape_ref.push_back(hasScaleLocal ? &scale : nullptr);

    const auto& os = params.outputShape;
    gert::StorageShape output = {{os[0], os[1], os[2], os[3], os[4]}, {os[0], os[1], os[2], os[3], os[4]}};
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
                      .NodeInputTd(0, params.fmapDtype, ge::FORMAT_NCDHW, ge::FORMAT_NCDHW)
                      .NodeInputTd(1, params.weightDtype, params.weightOriginFormat, params.weightStorageFormat)
                      .NodeInputTd(2, params.biasDtype, params.biasFormat, params.biasStorageFormat)
                      .NodeInputTd(3, params.scaleDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, params.outputDtype, params.outputFormat, params.outputFormat)
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
} // namespace

class TestConv3dCheckInput : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

#define CONV3D_INPUT_ERR(name, ...) \
    TEST_F(TestConv3dCheckInput, name) { RunConv3dInputTest(__VA_ARGS__); }
#define CONV3D_INPUT_OK(name, ...) \
    TEST_F(TestConv3dCheckInput, name) { RunConv3dInputTest(__VA_ARGS__); }

// Default params helper
#define DFL_FM_SHAPE {1, 16, 8, 16, 16}
#define DFL_WT_SHAPE {16, 16, 1, 1, 1}
#define DFL_BIAS {16}
#define DFL_OUT_SHAPE {1, 16, 8, 16, 16}
#define DFL_PADS {0, 0, 0, 0, 0, 0}
#define DFL_STRIDES {1, 1, 1}
#define DFL_DILAS {1, 1, 1}
#define DFL_DT_FP16 DT_FLOAT16, DT_FLOAT16, DT_FLOAT16, DT_FLOAT, DT_FLOAT16
#define DFL_FMTS ge::FORMAT_NCDHW, ge::FORMAT_NCDHW, ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_NCDHW

// 1. ParseFmapShape �?wrong dim (4D instead of 5D) �?FAILED
CONV3D_INPUT_ERR(parse_fmap_shape_wrong_dim_fail, {"fmap_wrong_dim",
                                                   {1, 16, 8, 16},
                                                   {},
                                                   DFL_WT_SHAPE,
                                                   {},
                                                   DFL_BIAS,
                                                   {},
                                                   DFL_OUT_SHAPE,
                                                   {},
                                                   DFL_PADS,
                                                   DFL_STRIDES,
                                                   DFL_DILAS,
                                                   DFL_DT_FP16,
                                                   DFL_FMTS,
                                                   1,
                                                   1,
                                                   0,
                                                   true})

// 2. ParseFmapShape �?correct dim (5D) �?SUCCESS
CONV3D_INPUT_OK(parse_fmap_shape_correct_dim_success, {"fmap_correct_dim",
                                                       DFL_FM_SHAPE,
                                                       {},
                                                       DFL_WT_SHAPE,
                                                       {},
                                                       DFL_BIAS,
                                                       {},
                                                       DFL_OUT_SHAPE,
                                                       {},
                                                       DFL_PADS,
                                                       DFL_STRIDES,
                                                       DFL_DILAS,
                                                       DFL_DT_FP16,
                                                       DFL_FMTS,
                                                       1,
                                                       1,
                                                       0,
                                                       true})

// 3. CheckFmapShape �?N=0 �?FAILED
CONV3D_INPUT_ERR(check_fmap_shape_n_zero_fail, {"fmap_N_zero",
                                                {0, 16, 8, 16, 16},
                                                {},
                                                DFL_WT_SHAPE,
                                                {},
                                                DFL_BIAS,
                                                {},
                                                DFL_OUT_SHAPE,
                                                {},
                                                DFL_PADS,
                                                DFL_STRIDES,
                                                DFL_DILAS,
                                                DFL_DT_FP16,
                                                DFL_FMTS,
                                                1,
                                                1,
                                                0,
                                                true})

// 4. CheckFmapShape �?C=0 �?FAILED
CONV3D_INPUT_ERR(check_fmap_shape_c_zero_fail, {"fmap_C_zero",
                                                {1, 0, 8, 16, 16},
                                                {},
                                                DFL_WT_SHAPE,
                                                {},
                                                DFL_BIAS,
                                                {},
                                                DFL_OUT_SHAPE,
                                                {},
                                                DFL_PADS,
                                                DFL_STRIDES,
                                                DFL_DILAS,
                                                DFL_DT_FP16,
                                                DFL_FMTS,
                                                1,
                                                1,
                                                0,
                                                true})

// 5. CheckFmapShape �?D=0 �?FAILED
CONV3D_INPUT_ERR(check_fmap_shape_d_zero_fail, {"fmap_D_zero",
                                                {1, 16, 0, 16, 16},
                                                {},
                                                DFL_WT_SHAPE,
                                                {},
                                                DFL_BIAS,
                                                {},
                                                DFL_OUT_SHAPE,
                                                {},
                                                DFL_PADS,
                                                DFL_STRIDES,
                                                DFL_DILAS,
                                                DFL_DT_FP16,
                                                DFL_FMTS,
                                                1,
                                                1,
                                                0,
                                                true})

// 6. CheckFmapShape �?H=0 �?FAILED
CONV3D_INPUT_ERR(check_fmap_shape_h_zero_fail, {"fmap_H_zero",
                                                {1, 16, 8, 0, 16},
                                                {},
                                                DFL_WT_SHAPE,
                                                {},
                                                DFL_BIAS,
                                                {},
                                                DFL_OUT_SHAPE,
                                                {},
                                                DFL_PADS,
                                                DFL_STRIDES,
                                                DFL_DILAS,
                                                DFL_DT_FP16,
                                                DFL_FMTS,
                                                1,
                                                1,
                                                0,
                                                true})

// 7. CheckFmapShape �?W=0 �?FAILED
CONV3D_INPUT_ERR(check_fmap_shape_w_zero_fail, {"fmap_W_zero",
                                                {1, 16, 8, 16, 0},
                                                {},
                                                DFL_WT_SHAPE,
                                                {},
                                                DFL_BIAS,
                                                {},
                                                DFL_OUT_SHAPE,
                                                {},
                                                DFL_PADS,
                                                DFL_STRIDES,
                                                DFL_DILAS,
                                                DFL_DT_FP16,
                                                DFL_FMTS,
                                                1,
                                                1,
                                                0,
                                                true})

// 8. CheckFmapShape �?D out of max range �?FAILED
CONV3D_INPUT_ERR(check_fmap_shape_d_overflow_fail, {"fmap_D_overflow",
                                                    {1, 16, MAX_VAL + 1, 16, 16},
                                                    {},
                                                    DFL_WT_SHAPE,
                                                    {},
                                                    DFL_BIAS,
                                                    {},
                                                    DFL_OUT_SHAPE,
                                                    {},
                                                    DFL_PADS,
                                                    DFL_STRIDES,
                                                    DFL_DILAS,
                                                    DFL_DT_FP16,
                                                    DFL_FMTS,
                                                    1,
                                                    1,
                                                    0,
                                                    true})

// 9. CheckFmapShape �?H out of max range �?FAILED
CONV3D_INPUT_ERR(check_fmap_shape_h_overflow_fail, {"fmap_H_overflow",
                                                    {1, 16, 8, MAX_VAL + 1, 16},
                                                    {},
                                                    DFL_WT_SHAPE,
                                                    {},
                                                    DFL_BIAS,
                                                    {},
                                                    DFL_OUT_SHAPE,
                                                    {},
                                                    DFL_PADS,
                                                    DFL_STRIDES,
                                                    DFL_DILAS,
                                                    DFL_DT_FP16,
                                                    DFL_FMTS,
                                                    1,
                                                    1,
                                                    0,
                                                    true})

// 10. CheckFmapShape �?W out of max range �?FAILED
CONV3D_INPUT_ERR(check_fmap_shape_w_overflow_fail, {"fmap_W_overflow",
                                                    {1, 16, 8, 16, MAX_VAL + 1},
                                                    {},
                                                    DFL_WT_SHAPE,
                                                    {},
                                                    DFL_BIAS,
                                                    {},
                                                    DFL_OUT_SHAPE,
                                                    {},
                                                    DFL_PADS,
                                                    DFL_STRIDES,
                                                    DFL_DILAS,
                                                    DFL_DT_FP16,
                                                    DFL_FMTS,
                                                    1,
                                                    1,
                                                    0,
                                                    true})

// 11. ParseWeightShape �?wrong dim (4D instead of 5D) �?FAILED
CONV3D_INPUT_ERR(parse_weight_shape_wrong_dim_fail, {"weight_wrong_dim",
                                                     DFL_FM_SHAPE,
                                                     {},
                                                     {16, 16, 1, 1},
                                                     {},
                                                     DFL_BIAS,
                                                     {},
                                                     DFL_OUT_SHAPE,
                                                     {},
                                                     DFL_PADS,
                                                     DFL_STRIDES,
                                                     DFL_DILAS,
                                                     DFL_DT_FP16,
                                                     DFL_FMTS,
                                                     1,
                                                     1,
                                                     0,
                                                     true})

// 12. CheckWeightShape �?cout (N) = 0 �?FAILED
CONV3D_INPUT_ERR(check_weight_shape_cout_zero_fail, {"weight_cout_zero",
                                                     DFL_FM_SHAPE,
                                                     {},
                                                     {0, 16, 1, 1, 1},
                                                     {},
                                                     {0},
                                                     {},
                                                     {1, 0, 8, 16, 16},
                                                     {},
                                                     DFL_PADS,
                                                     DFL_STRIDES,
                                                     DFL_DILAS,
                                                     DFL_DT_FP16,
                                                     DFL_FMTS,
                                                     1,
                                                     1,
                                                     0,
                                                     true})

// 13. CheckWeightShape �?kd = 0 �?FAILED
CONV3D_INPUT_ERR(check_weight_shape_kd_zero_fail, {"weight_kd_zero",
                                                   DFL_FM_SHAPE,
                                                   {},
                                                   {16, 16, 0, 1, 1},
                                                   {},
                                                   DFL_BIAS,
                                                   {},
                                                   DFL_OUT_SHAPE,
                                                   {},
                                                   DFL_PADS,
                                                   DFL_STRIDES,
                                                   DFL_DILAS,
                                                   DFL_DT_FP16,
                                                   DFL_FMTS,
                                                   1,
                                                   1,
                                                   0,
                                                   true})

// 14. CheckWeightShape �?kh = 0 �?FAILED
CONV3D_INPUT_ERR(check_weight_shape_kh_zero_fail, {"weight_kh_zero",
                                                   DFL_FM_SHAPE,
                                                   {},
                                                   {16, 16, 1, 0, 1},
                                                   {},
                                                   DFL_BIAS,
                                                   {},
                                                   DFL_OUT_SHAPE,
                                                   {},
                                                   DFL_PADS,
                                                   DFL_STRIDES,
                                                   DFL_DILAS,
                                                   DFL_DT_FP16,
                                                   DFL_FMTS,
                                                   1,
                                                   1,
                                                   0,
                                                   true})

// 15. CheckWeightShape �?kw = 0 �?FAILED
CONV3D_INPUT_ERR(check_weight_shape_kw_zero_fail, {"weight_kw_zero",
                                                   DFL_FM_SHAPE,
                                                   {},
                                                   {16, 16, 1, 1, 0},
                                                   {},
                                                   DFL_BIAS,
                                                   {},
                                                   DFL_OUT_SHAPE,
                                                   {},
                                                   DFL_PADS,
                                                   DFL_STRIDES,
                                                   DFL_DILAS,
                                                   DFL_DT_FP16,
                                                   DFL_FMTS,
                                                   1,
                                                   1,
                                                   0,
                                                   true})

// 16. CheckWeightShape �?kd overflow �?FAILED
CONV3D_INPUT_ERR(check_weight_shape_kd_overflow_fail, {"weight_kd_overflow",
                                                       DFL_FM_SHAPE,
                                                       {},
                                                       {16, 16, MAX_VAL + 1, 1, 1},
                                                       {},
                                                       DFL_BIAS,
                                                       {},
                                                       DFL_OUT_SHAPE,
                                                       {},
                                                       DFL_PADS,
                                                       DFL_STRIDES,
                                                       DFL_DILAS,
                                                       DFL_DT_FP16,
                                                       DFL_FMTS,
                                                       1,
                                                       1,
                                                       0,
                                                       true})

// 17. CheckWeightShape �?kh overflow �?FAILED
CONV3D_INPUT_ERR(check_weight_shape_kh_overflow_fail, {"weight_kh_overflow",
                                                       DFL_FM_SHAPE,
                                                       {},
                                                       {16, 16, 1, MAX_VAL + 1, 1},
                                                       {},
                                                       DFL_BIAS,
                                                       {},
                                                       DFL_OUT_SHAPE,
                                                       {},
                                                       DFL_PADS,
                                                       DFL_STRIDES,
                                                       DFL_DILAS,
                                                       DFL_DT_FP16,
                                                       DFL_FMTS,
                                                       1,
                                                       1,
                                                       0,
                                                       true})

// 18. CheckWeightShape �?kw overflow �?FAILED
CONV3D_INPUT_ERR(check_weight_shape_kw_overflow_fail, {"weight_kw_overflow",
                                                       DFL_FM_SHAPE,
                                                       {},
                                                       {16, 16, 1, 1, MAX_VAL + 1},
                                                       {},
                                                       DFL_BIAS,
                                                       {},
                                                       DFL_OUT_SHAPE,
                                                       {},
                                                       DFL_PADS,
                                                       DFL_STRIDES,
                                                       DFL_DILAS,
                                                       DFL_DT_FP16,
                                                       DFL_FMTS,
                                                       1,
                                                       1,
                                                       0,
                                                       true})

// 19. CheckWeightShape �?cout overflow �?FAILED
CONV3D_INPUT_ERR(check_weight_shape_cout_overflow_fail, {"weight_cout_overflow",
                                                         DFL_FM_SHAPE,
                                                         {},
                                                         {MAX_VAL + 1, 16, 1, 1, 1},
                                                         {},
                                                         DFL_BIAS,
                                                         {},
                                                         DFL_OUT_SHAPE,
                                                         {},
                                                         DFL_PADS,
                                                         DFL_STRIDES,
                                                         DFL_DILAS,
                                                         DFL_DT_FP16,
                                                         DFL_FMTS,
                                                         1,
                                                         1,
                                                         0,
                                                         true})

// 20. CheckWeightNZFormatShape �?FRACTAL_Z_3D storage shape wrong dim (3D) �?FAILED
CONV3D_INPUT_ERR(check_weight_nz_shape_wrong_dim_fail, {"weight_nz_wrong_dim",
                                                        DFL_FM_SHAPE,
                                                        {},
                                                        DFL_WT_SHAPE,
                                                        {1, 1, 16},
                                                        DFL_BIAS,
                                                        {},
                                                        DFL_OUT_SHAPE,
                                                        {},
                                                        DFL_PADS,
                                                        DFL_STRIDES,
                                                        DFL_DILAS,
                                                        DT_INT8,
                                                        DT_INT8,
                                                        DT_FLOAT,
                                                        DT_FLOAT,
                                                        DT_FLOAT16,
                                                        ge::FORMAT_NCDHW,
                                                        ge::FORMAT_FRACTAL_Z_3D,
                                                        ge::FORMAT_ND,
                                                        ge::FORMAT_ND,
                                                        ge::FORMAT_NDHWC,
                                                        1,
                                                        1,
                                                        1,
                                                        true})

// 21. ParseBiasShape �?bias 1D matches cout �?SUCCESS
CONV3D_INPUT_OK(parse_bias_shape_1d_success, {"bias_1d",
                                              DFL_FM_SHAPE,
                                              {},
                                              DFL_WT_SHAPE,
                                              {},
                                              DFL_BIAS,
                                              {},
                                              DFL_OUT_SHAPE,
                                              {},
                                              DFL_PADS,
                                              DFL_STRIDES,
                                              DFL_DILAS,
                                              DFL_DT_FP16,
                                              DFL_FMTS,
                                              1,
                                              1,
                                              0,
                                              true})

// 22. ParseBiasShape �?bias 5D [1,cout,1,1,1] �?SUCCESS (bias format = NCDHW)
CONV3D_INPUT_OK(parse_bias_shape_5d_success, {"bias_5d",
                                              DFL_FM_SHAPE,
                                              {},
                                              DFL_WT_SHAPE,
                                              {},
                                              {1, 16, 1, 1, 1},
                                              {},
                                              DFL_OUT_SHAPE,
                                              {},
                                              DFL_PADS,
                                              DFL_STRIDES,
                                              DFL_DILAS,
                                              DFL_DT_FP16,
                                              ge::FORMAT_NCDHW,
                                              ge::FORMAT_NCDHW,
                                              ge::FORMAT_NCDHW,
                                              ge::FORMAT_NCDHW,
                                              ge::FORMAT_NCDHW,
                                              1,
                                              1,
                                              0,
                                              false})

// 23. ParseBiasShape �?bias wrong dim (2D) �?FAILED
CONV3D_INPUT_ERR(parse_bias_shape_wrong_dim_fail, {"bias_2d",
                                                   DFL_FM_SHAPE,
                                                   {},
                                                   DFL_WT_SHAPE,
                                                   {},
                                                   {16, 1},
                                                   {},
                                                   DFL_OUT_SHAPE,
                                                   {},
                                                   DFL_PADS,
                                                   DFL_STRIDES,
                                                   DFL_DILAS,
                                                   DFL_DT_FP16,
                                                   DFL_FMTS,
                                                   1,
                                                   1,
                                                   0,
                                                   true})

// 24. ParseBiasShape �?bias C != weight cout �?FAILED
CONV3D_INPUT_ERR(parse_bias_shape_c_mismatch_fail, {"bias_c_mismatch",
                                                    DFL_FM_SHAPE,
                                                    {},
                                                    {32, 16, 1, 1, 1},
                                                    {},
                                                    {16},
                                                    {},
                                                    {1, 32, 8, 16, 16},
                                                    {},
                                                    DFL_PADS,
                                                    DFL_STRIDES,
                                                    DFL_DILAS,
                                                    DFL_DT_FP16,
                                                    DFL_FMTS,
                                                    1,
                                                    1,
                                                    0,
                                                    true})

// 25. CheckOutputShape �?output D out of max range �?FAILED
CONV3D_INPUT_ERR(check_output_shape_d_overflow_fail, {"out_D_overflow",
                                                      DFL_FM_SHAPE,
                                                      {},
                                                      DFL_WT_SHAPE,
                                                      {},
                                                      DFL_BIAS,
                                                      {},
                                                      {1, 16, MAX_VAL + 1, 16, 16},
                                                      {},
                                                      DFL_PADS,
                                                      DFL_STRIDES,
                                                      DFL_DILAS,
                                                      DFL_DT_FP16,
                                                      DFL_FMTS,
                                                      1,
                                                      1,
                                                      0,
                                                      true})

// 26. CheckOutputShape �?output H out of max range �?FAILED
CONV3D_INPUT_ERR(check_output_shape_h_overflow_fail, {"out_H_overflow",
                                                      DFL_FM_SHAPE,
                                                      {},
                                                      DFL_WT_SHAPE,
                                                      {},
                                                      DFL_BIAS,
                                                      {},
                                                      {1, 16, 8, MAX_VAL + 1, 16},
                                                      {},
                                                      DFL_PADS,
                                                      DFL_STRIDES,
                                                      DFL_DILAS,
                                                      DFL_DT_FP16,
                                                      DFL_FMTS,
                                                      1,
                                                      1,
                                                      0,
                                                      true})

// 27. CheckOutputShape �?output W out of max range �?FAILED
CONV3D_INPUT_ERR(check_output_shape_w_overflow_fail, {"out_W_overflow",
                                                      DFL_FM_SHAPE,
                                                      {},
                                                      DFL_WT_SHAPE,
                                                      {},
                                                      DFL_BIAS,
                                                      {},
                                                      {1, 16, 8, 16, MAX_VAL + 1},
                                                      {},
                                                      DFL_PADS,
                                                      DFL_STRIDES,
                                                      DFL_DILAS,
                                                      DFL_DT_FP16,
                                                      DFL_FMTS,
                                                      1,
                                                      1,
                                                      0,
                                                      true})

// 28. CheckOutputShape �?output D/H/W all valid �?SUCCESS
CONV3D_INPUT_OK(check_output_shape_valid_success, {"out_valid",
                                                   DFL_FM_SHAPE,
                                                   {},
                                                   DFL_WT_SHAPE,
                                                   {},
                                                   DFL_BIAS,
                                                   {},
                                                   DFL_OUT_SHAPE,
                                                   {},
                                                   DFL_PADS,
                                                   DFL_STRIDES,
                                                   DFL_DILAS,
                                                   DFL_DT_FP16,
                                                   DFL_FMTS,
                                                   1,
                                                   1,
                                                   0,
                                                   true})

// 29. CheckParamsDtype �?with bias, valid BF16 �?SUCCESS
CONV3D_INPUT_OK(check_params_dtype_with_bias_bf16_success, {"dtype_bias_bf16",
                                                            DFL_FM_SHAPE,
                                                            {},
                                                            DFL_WT_SHAPE,
                                                            {},
                                                            DFL_BIAS,
                                                            {},
                                                            DFL_OUT_SHAPE,
                                                            {},
                                                            DFL_PADS,
                                                            DFL_STRIDES,
                                                            DFL_DILAS,
                                                            DT_BF16,
                                                            DT_BF16,
                                                            DT_BF16,
                                                            DT_FLOAT,
                                                            DT_BF16,
                                                            DFL_FMTS,
                                                            1,
                                                            1,
                                                            0,
                                                            true})

// 30. CheckParamsDtype �?without bias, valid FP16 �?SUCCESS
CONV3D_INPUT_OK(check_params_dtype_without_bias_fp16_success, {"dtype_nobias_fp16",
                                                               DFL_FM_SHAPE,
                                                               {},
                                                               DFL_WT_SHAPE,
                                                               {},
                                                               {},
                                                               {},
                                                               DFL_OUT_SHAPE,
                                                               {},
                                                               DFL_PADS,
                                                               DFL_STRIDES,
                                                               DFL_DILAS,
                                                               DFL_DT_FP16,
                                                               DFL_FMTS,
                                                               1,
                                                               0,
                                                               0,
                                                               false})

// 31. CheckParamsDtype �?invalid dtype INT32 �?FAILED
CONV3D_INPUT_ERR(check_params_dtype_invalid_fail, {"dtype_invalid",
                                                   DFL_FM_SHAPE,
                                                   {},
                                                   DFL_WT_SHAPE,
                                                   {},
                                                   DFL_BIAS,
                                                   {},
                                                   DFL_OUT_SHAPE,
                                                   {},
                                                   DFL_PADS,
                                                   DFL_STRIDES,
                                                   DFL_DILAS,
                                                   DT_INT32,
                                                   DT_INT32,
                                                   DT_INT32,
                                                   DT_FLOAT,
                                                   DT_FLOAT16,
                                                   DFL_FMTS,
                                                   1,
                                                   1,
                                                   0,
                                                   true})

// 32. CheckParamsDtype �?with bias, valid FP32 �?SUCCESS
CONV3D_INPUT_OK(check_params_dtype_with_bias_fp32_success, {"dtype_bias_fp32",
                                                            DFL_FM_SHAPE,
                                                            {},
                                                            DFL_WT_SHAPE,
                                                            {},
                                                            DFL_BIAS,
                                                            {},
                                                            DFL_OUT_SHAPE,
                                                            {},
                                                            DFL_PADS,
                                                            DFL_STRIDES,
                                                            DFL_DILAS,
                                                            DT_FLOAT,
                                                            DT_FLOAT,
                                                            DT_FLOAT,
                                                            DT_FLOAT,
                                                            DT_FLOAT,
                                                            DFL_FMTS,
                                                            1,
                                                            1,
                                                            0,
                                                            true})

// 33. CheckInputDesc �?NCDHW/NCDHW/NCDHW format pass �?SUCCESS
CONV3D_INPUT_OK(check_input_desc_ncdhw_success, {"format_ncdhw",
                                                 DFL_FM_SHAPE,
                                                 {},
                                                 DFL_WT_SHAPE,
                                                 {},
                                                 DFL_BIAS,
                                                 {},
                                                 DFL_OUT_SHAPE,
                                                 {},
                                                 DFL_PADS,
                                                 DFL_STRIDES,
                                                 DFL_DILAS,
                                                 DFL_DT_FP16,
                                                 DFL_FMTS,
                                                 1,
                                                 1,
                                                 0,
                                                 true})
