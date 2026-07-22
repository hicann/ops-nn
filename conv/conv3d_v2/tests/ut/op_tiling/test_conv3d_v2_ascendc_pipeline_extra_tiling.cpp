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
 * \file test_conv3d_v2_ascendc_pipeline_extra_tiling.cpp
 * \brief Additional UT for conv3d_v2 base tiling pipeline coverage
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
struct Conv3dPipelineParams {
    string caseName;
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

struct ConvShape {
    uint64_t inputV;
    uint64_t kernelV;
    uint64_t padone;
    uint64_t padtwo;
    uint64_t dilationV;
    uint64_t strideV;
};

int64_t InferOut3D(ConvShape convShape)
{
    if (convShape.strideV == 0) {
        return 0;
    }
    return (convShape.inputV + convShape.padone + convShape.padtwo - convShape.dilationV * (convShape.kernelV - 1) -
            1) /
               convShape.strideV +
           1;
}

void RunConv3dPipelineTest(const Conv3dPipelineParams& params)
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

    bool isNdhwc = (params.format == "NDHWC");
    ge::Format fmapFormat = isNdhwc ? ge::FORMAT_NDHWC : ge::FORMAT_NCDHW;
    ge::Format weightFormat = isNdhwc ? ge::FORMAT_DHWCN : ge::FORMAT_NCDHW;
    ge::Format outputFormat = isNdhwc ? ge::FORMAT_NDHWC : ge::FORMAT_NCDHW;

    ConvShape convShapeDo = {static_cast<uint64_t>(di), static_cast<uint64_t>(kd), padh, padt, dilationD, strideD};
    ConvShape convShapeHo = {static_cast<uint64_t>(hi), static_cast<uint64_t>(kh), padu, padd, dilationH, strideH};
    ConvShape convShapeWo = {static_cast<uint64_t>(wi), static_cast<uint64_t>(kw), padl, padr, dilationW, strideW};
    int64_t Do = InferOut3D(convShapeDo);
    int64_t ho = InferOut3D(convShapeHo);
    int64_t wo = InferOut3D(convShapeWo);
    if (params.padMode == "SAME" || params.padMode == "SAME_UPPER" || params.padMode == "SAME_LOWER") {
        Do = (di + strideD - 1) / strideD;
        ho = (hi + strideH - 1) / strideH;
        wo = (wi + strideW - 1) / strideW;
    }

    gert::StorageShape featuremap;
    gert::StorageShape weight;
    gert::StorageShape bias = {{cout}, {cout}};
    gert::StorageShape output;

    if (isNdhwc) {
        featuremap = {{batch, di, hi, wi, cin}, {batch, di, hi, wi, cin}};
        weight = {{kd, kh, kw, cin / params.groups, cout}, {kd, kh, kw, cin / params.groups, cout}};
        output = {{batch, Do, ho, wo, cout}, {batch, Do, ho, wo, cout}};
    } else {
        featuremap = {{batch, cin, di, hi, wi}, {batch, cin, di, hi, wi}};
        weight = {{cout, cin / params.groups, kd, kh, kw}, {cout, cin / params.groups, kd, kh, kw}};
        output = {{batch, cout, Do, ho, wo}, {batch, cout, Do, ho, wo}};
    }

    std::vector<void*> input_shape_ref = params.hasBias ? std::vector<void*>{&featuremap, &weight, &bias, nullptr} :
                                                          std::vector<void*>{&featuremap, &weight, nullptr, nullptr};
    std::vector<void*> output_shapes_ref = {&output};

    std::vector<int64_t> stridesVec;
    std::vector<int64_t> dilationsVec;
    if (isNdhwc) {
        stridesVec = {1, static_cast<int64_t>(strideD), static_cast<int64_t>(strideH), static_cast<int64_t>(strideW),
                      1};
        dilationsVec = {1, static_cast<int64_t>(dilationD), static_cast<int64_t>(dilationH),
                        static_cast<int64_t>(dilationW), 1};
    } else {
        stridesVec = {1, 1, static_cast<int64_t>(strideD), static_cast<int64_t>(strideH),
                      static_cast<int64_t>(strideW)};
        dilationsVec = {1, 1, static_cast<int64_t>(dilationD), static_cast<int64_t>(dilationH),
                        static_cast<int64_t>(dilationW)};
    }
    std::vector<int64_t> padsVec = {static_cast<int64_t>(padh), static_cast<int64_t>(padt), static_cast<int64_t>(padu),
                                    static_cast<int64_t>(padd), static_cast<int64_t>(padl), static_cast<int64_t>(padr)};

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
                      .NodeInputTd(0, params.dtype, fmapFormat, fmapFormat)
                      .NodeInputTd(1, params.dtype, weightFormat, weightFormat)
                      .NodeInputTd(2, params.dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
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

class TestConv3dPipelineExtra : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// ============================================================================
// Basic 3D conv → SUCCESS
// ============================================================================
TEST_F(TestConv3dPipelineExtra, basic_3d_conv_success)
{
    RunConv3dPipelineTest({"basic_3d",
                           {1, 16, 8, 16, 16},
                           {16, 16, 3, 3, 3},
                           {1, 1, 1, 1, 1, 1},
                           {1, 1, 1},
                           {1, 1, 1},
                           1,
                           "SPECIFIC",
                           "NCDHW",
                           DT_FLOAT16,
                           true,
                           false});
}

// ============================================================================
// Group conv → SUCCESS
// ============================================================================
TEST_F(TestConv3dPipelineExtra, group_conv_success)
{
    RunConv3dPipelineTest({"group_conv",
                           {1, 32, 4, 16, 16},
                           {64, 8, 3, 3, 3},
                           {1, 1, 1, 1, 1, 1},
                           {1, 1, 1},
                           {1, 1, 1},
                           4,
                           "SPECIFIC",
                           "NCDHW",
                           DT_FLOAT16,
                           true,
                           false});
}

// ============================================================================
// NDHWC format → SUCCESS
// ============================================================================
TEST_F(TestConv3dPipelineExtra, ndhwc_format_success)
{
    RunConv3dPipelineTest({"ndhwc",
                           {1, 16, 8, 16, 32},
                           {32, 16, 3, 3, 3},
                           {1, 1, 1, 1, 1, 1},
                           {1, 1, 1},
                           {1, 1, 1},
                           1,
                           "SPECIFIC",
                           "NDHWC",
                           DT_FLOAT16,
                           true,
                           false});
}

// ============================================================================
// NDHWC with groups → SUCCESS
// ============================================================================
TEST_F(TestConv3dPipelineExtra, ndhwc_group_success)
{
    RunConv3dPipelineTest({"ndhwc_group",
                           {1, 16, 4, 16, 32},
                           {16, 4, 3, 3, 3},
                           {1, 1, 1, 1, 1, 1},
                           {2, 2, 2},
                           {1, 1, 1},
                           4,
                           "SPECIFIC",
                           "NDHWC",
                           DT_FLOAT16,
                           true,
                           false});
}

// ============================================================================
// FLOAT32 dtype → SUCCESS
// ============================================================================
TEST_F(TestConv3dPipelineExtra, float32_dtype_success)
{
    RunConv3dPipelineTest({"fp32",
                           {1, 16, 4, 16, 16},
                           {32, 16, 3, 3, 3},
                           {1, 1, 1, 1, 1, 1},
                           {1, 1, 1},
                           {1, 1, 1},
                           1,
                           "SPECIFIC",
                           "NCDHW",
                           DT_FLOAT,
                           true,
                           false});
}

// ============================================================================
// SAME pad mode → SUCCESS
// ============================================================================
TEST_F(TestConv3dPipelineExtra, same_pad_success)
{
    RunConv3dPipelineTest({"same_pad",
                           {1, 16, 8, 28, 28},
                           {32, 16, 3, 3, 3},
                           {0, 0, 0, 0, 0, 0},
                           {2, 2, 2},
                           {1, 1, 1},
                           1,
                           "SAME",
                           "NCDHW",
                           DT_FLOAT16,
                           true,
                           false});
}

// ============================================================================
// VALID pad mode → SUCCESS
// ============================================================================
TEST_F(TestConv3dPipelineExtra, valid_pad_success)
{
    RunConv3dPipelineTest({"valid_pad",
                           {1, 16, 8, 28, 28},
                           {32, 16, 3, 3, 3},
                           {0, 0, 0, 0, 0, 0},
                           {1, 1, 1},
                           {1, 1, 1},
                           1,
                           "VALID",
                           "NCDHW",
                           DT_FLOAT16,
                           true,
                           false});
}

// ============================================================================
// No bias → SUCCESS
// ============================================================================
TEST_F(TestConv3dPipelineExtra, no_bias_success)
{
    RunConv3dPipelineTest({"no_bias",
                           {1, 32, 4, 28, 28},
                           {64, 32, 3, 3, 3},
                           {1, 1, 1, 1, 1, 1},
                           {2, 2, 2},
                           {1, 1, 1},
                           1,
                           "SPECIFIC",
                           "NCDHW",
                           DT_FLOAT16,
                           false,
                           false});
}

// ============================================================================
// Depthwise-like conv (groups = cin) → SUCCESS
// ============================================================================
TEST_F(TestConv3dPipelineExtra, depthwise_success)
{
    RunConv3dPipelineTest({"dw",
                           {1, 32, 4, 14, 14},
                           {32, 1, 3, 3, 3},
                           {1, 1, 1, 1, 1, 1},
                           {1, 1, 1},
                           {1, 1, 1},
                           32,
                           "SPECIFIC",
                           "NCDHW",
                           DT_FLOAT16,
                           true,
                           false});
}

// ============================================================================
// Batch > 1 → SUCCESS
// ============================================================================
TEST_F(TestConv3dPipelineExtra, batch_2_success)
{
    RunConv3dPipelineTest({"batch2",
                           {2, 16, 8, 16, 16},
                           {16, 16, 3, 3, 3},
                           {1, 1, 1, 1, 1, 1},
                           {1, 1, 1},
                           {1, 1, 1},
                           1,
                           "SPECIFIC",
                           "NCDHW",
                           DT_FLOAT16,
                           true,
                           false});
}

// ============================================================================
// Pointwise 1x1x1 conv → SUCCESS
// ============================================================================
TEST_F(TestConv3dPipelineExtra, pointwise_3d_success)
{
    RunConv3dPipelineTest({"pointwise",
                           {1, 128, 4, 14, 14},
                           {256, 128, 1, 1, 1},
                           {0, 0, 0, 0, 0, 0},
                           {1, 1, 1},
                           {1, 1, 1},
                           1,
                           "SPECIFIC",
                           "NCDHW",
                           DT_FLOAT16,
                           false,
                           false});
}
