/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <string>
#include "log/log.h"
#include <gtest/gtest.h>
#include "register/op_impl_registry.h"
#include "platform/platform_infos_def.h"
#include "ut_op_common.h"
#include "ut_op_util.h"
#include "../../../../op_host/arch35/cla_gate_quant_tiling_arch35.h"
#include "tiling_context_faker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"

using namespace std;

class ClaGateQuantTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ClaGateQuantTilingTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "ClaGateQuantTilingTest TearDown" << std::endl; }
};

namespace {
constexpr int64_t SCALE_BLOCK_H = 64;
constexpr int64_t SCALE_LAST_DIM = 2;

// The operator takes four TND inputs and five attributes.  The attribute order
// (dst_type, round_mode, scale_alg, input_attn_layout, dual_axis_flag) must stay in
// sync with op_host/cla_gate_quant_def.cpp and the tiling implementation.
//
// Expected output shapes:
//   row_data   : [T, N*D]
//   row_scale  : [T, ceil(N*D/64), 2]
//   col_data   : [T, N*D]        when dual_axis_flag == true, otherwise [0]
//   col_scale  : [ceil(T/64), N*D, 2] when dual_axis_flag == true, otherwise [0]
static void ExecuteTestCase(ge::DataType xDtype, ge::DataType outDtype, int64_t t, int64_t n, int64_t d,
                            const string& round_mode, int64_t dst_type, int64_t scale_alg, bool dual_axis_flag,
                            const string& input_attn_layout, ge::graphStatus status = ge::GRAPH_SUCCESS,
                            int64_t colScaleDim0Delta = 0)
{
    string compile_info_string = R"({
         "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                           "Intrinsic_fix_pipe_l0c2out": false,
                           "Intrinsic_data_move_l12ub": true,
                           "Intrinsic_data_move_l0c2ub": true,
                           "Intrinsic_data_move_out2l1_nd2nz": false,
                           "UB_SIZE": 253952, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                           "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                           "CORE_NUM": 64}
                           })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> socversions = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();

    optiling::ClaGateQuantCompileInfo compile_info;
    compile_info.coreNum = 64;
    compile_info.ubSize = 253952;

    std::string op_type("ClaGateQuant");

    const int64_t k = n * d;
    const int64_t rowScaleNum = (k + SCALE_BLOCK_H - 1) / SCALE_BLOCK_H;
    const int64_t colScaleNum = (t + SCALE_BLOCK_H - 1) / SCALE_BLOCK_H;

    gert::StorageShape globalAttnShape({t, n, d}, {t, n, d});
    gert::StorageShape localAttnShape({t, n, d}, {t, n, d});
    gert::StorageShape globalGateShape({t, n}, {t, n});
    gert::StorageShape localGateShape({t, n}, {t, n});
    gert::StorageShape rowDataShape({t, k}, {t, k});
    gert::StorageShape rowScaleShape({t, rowScaleNum, SCALE_LAST_DIM}, {t, rowScaleNum, SCALE_LAST_DIM});
    // Single-axis mode (dual_axis_flag == false) emits empty col_data/col_scale ([0]).
    gert::StorageShape colDataShape = dual_axis_flag ? gert::StorageShape({t, k}, {t, k}) :
                                                       gert::StorageShape({0}, {0});
    gert::StorageShape colScaleShape = dual_axis_flag ?
                                           gert::StorageShape({colScaleNum + colScaleDim0Delta, k, SCALE_LAST_DIM},
                                                              {colScaleNum + colScaleDim0Delta, k, SCALE_LAST_DIM}) :
                                           gert::StorageShape({0}, {0});

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(4, 4)
                      .IrInstanceNum({1, 1, 1, 1}, {1, 1, 1, 1})
                      .InputShapes({&globalAttnShape, &localAttnShape, &globalGateShape, &localGateShape})
                      .OutputShapes({&rowDataShape, &rowScaleShape, &colDataShape, &colScaleShape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, outDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, outDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT8_E8M0, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"dst_type", Ops::NN::AnyValue::CreateFrom<int64_t>(dst_type)},
                                  {"round_mode", Ops::NN::AnyValue::CreateFrom<string>(round_mode)},
                                  {"scale_alg", Ops::NN::AnyValue::CreateFrom<int64_t>(scale_alg)},
                                  {"input_attn_layout", Ops::NN::AnyValue::CreateFrom<string>(input_attn_layout)},
                                  {"dual_axis_flag", Ops::NN::AnyValue::CreateFrom<bool>(dual_axis_flag)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context, nullptr);
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", socversions);

    ASSERT_EQ(optiling::TilingForClaGateQuant(tiling_context), status);
    if (status != ge::GRAPH_SUCCESS) {
        return;
    }

    auto rawTilingData = tiling_context->GetRawTilingData();
    ASSERT_NE(rawTilingData, nullptr);
    ASSERT_NE(rawTilingData->GetData(), nullptr);
    const auto* tilingData = reinterpret_cast<const ClaGateQuantTilingData*>(rawTilingData->GetData());
    EXPECT_EQ(tilingData->rowCount, t);
    EXPECT_EQ(tilingData->rowLength, k);
    EXPECT_EQ(tilingData->headCount, n);
    EXPECT_EQ(tilingData->headDim, d);
    EXPECT_GT(tilingData->usedCoreNum, 0);

    int64_t expectedTaskCount;
    if (dual_axis_flag) {
        int64_t expectedColTileNum = (k + 255) / 256;
        expectedTaskCount = ((t + SCALE_BLOCK_H - 1) / SCALE_BLOCK_H) * expectedColTileNum;
        EXPECT_EQ(tilingData->colTileNum, expectedColTileNum);
        EXPECT_EQ(tilingData->colTailSize, k % 256 == 0 ? 256 : k % 256);
    } else {
        expectedTaskCount = (t * k + 255) / 256;
        EXPECT_GT(tilingData->batchSegmentCapacity, 0);
        EXPECT_EQ(tilingData->streamTailSize, t * k % 256 == 0 ? 256 : t * k % 256);
    }
    EXPECT_EQ(tilingData->baseTaskCount * tilingData->usedCoreNum + tilingData->extraTaskCoreCount, expectedTaskCount);
}
} // namespace

TEST_F(ClaGateQuantTilingTest, test_tiling_fp16_to_fp8_e4m3fn)
{
    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT8_E4M3FN, 8, 32, 256, "rint", 36, 0, true, "TND", ge::GRAPH_SUCCESS);
}

TEST_F(ClaGateQuantTilingTest, test_tiling_bf16_to_fp8_e5m2)
{
    ExecuteTestCase(ge::DT_BF16, ge::DT_FLOAT8_E5M2, 4, 16, 128, "rint", 35, 0, true, "TND", ge::GRAPH_SUCCESS);
}

TEST_F(ClaGateQuantTilingTest, test_tiling_fp16_to_fp4_e2m1)
{
    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT4_E2M1, 4, 16, 256, "rint", 40, 0, true, "TND", ge::GRAPH_SUCCESS);
}

TEST_F(ClaGateQuantTilingTest, test_tiling_bf16_to_fp4_e1m2)
{
    ExecuteTestCase(ge::DT_BF16, ge::DT_FLOAT4_E1M2, 2, 8, 128, "floor", 41, 0, true, "TND", ge::GRAPH_SUCCESS);
}

TEST_F(ClaGateQuantTilingTest, test_tiling_fp8_scale_alg_1)
{
    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT8_E4M3FN, 4, 16, 128, "rint", 36, 1, true, "TND", ge::GRAPH_SUCCESS);
}

TEST_F(ClaGateQuantTilingTest, test_tiling_single_axis_fp8)
{
    ExecuteTestCase(ge::DT_BF16, ge::DT_FLOAT8_E4M3FN, 13, 11, 128, "rint", 36, 1, false, "TND", ge::GRAPH_SUCCESS);
}

TEST_F(ClaGateQuantTilingTest, test_tiling_single_axis_fp4)
{
    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT4_E2M1, 24, 11, 128, "floor", 40, 0, false, "TND", ge::GRAPH_SUCCESS);
}

TEST_F(ClaGateQuantTilingTest, test_tiling_invalid_round_mode_fp8)
{
    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT8_E4M3FN, 4, 16, 128, "floor", 36, 0, true, "TND", ge::GRAPH_FAILED);
}

TEST_F(ClaGateQuantTilingTest, test_tiling_invalid_dst_type)
{
    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT8_E4M3FN, 4, 16, 128, "rint", 99, 0, true, "TND", ge::GRAPH_FAILED);
}

TEST_F(ClaGateQuantTilingTest, test_tiling_invalid_scale_alg_fp4)
{
    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT4_E2M1, 4, 16, 128, "floor", 40, 1, true, "TND", ge::GRAPH_FAILED);
}

TEST_F(ClaGateQuantTilingTest, test_tiling_invalid_input_attn_layout)
{
    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT8_E4M3FN, 4, 16, 128, "rint", 36, 0, true, "BSND", ge::GRAPH_FAILED);
}

TEST_F(ClaGateQuantTilingTest, test_tiling_invalid_n)
{
    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT8_E4M3FN, 4, 256, 128, "rint", 36, 0, true, "TND", ge::GRAPH_FAILED);
}

// "round" is the third valid round_mode; it selects the TPL_ROUND tiling key.
TEST_F(ClaGateQuantTilingTest, test_tiling_fp4_round_mode)
{
    ExecuteTestCase(ge::DT_BF16, ge::DT_FLOAT4_E2M1, 8, 16, 128, "round", 40, 0, true, "TND", ge::GRAPH_SUCCESS);
}

TEST_F(ClaGateQuantTilingTest, test_tiling_single_axis_round_mode)
{
    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT4_E1M2, 4, 8, 256, "round", 41, 0, false, "TND", ge::GRAPH_SUCCESS);
}

// Tiny single-axis input: inputBytes/4096/2 rounds down to 0, so the core hint
// is clamped to 1 and only a single core is used.
TEST_F(ClaGateQuantTilingTest, test_tiling_single_axis_tiny_shape)
{
    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT8_E4M3FN, 1, 1, 128, "rint", 36, 1, false, "TND", ge::GRAPH_SUCCESS);
}

// Wide single-axis workload: many 1x256 tiles per core while the UB-derived
// batch capacity is smaller, exercising the sigmoid-tile-quantum rounding.
TEST_F(ClaGateQuantTilingTest, test_tiling_single_axis_batch_quantum)
{
    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT8_E5M2, 8192, 1, 256, "rint", 35, 0, false, "TND", ge::GRAPH_SUCCESS);
}

TEST_F(ClaGateQuantTilingTest, test_tiling_single_axis_batch_quantum_d128)
{
    ExecuteTestCase(ge::DT_BF16, ge::DT_FLOAT8_E4M3FN, 4096, 1, 128, "rint", 36, 1, false, "TND", ge::GRAPH_SUCCESS);
}

// Dual-axis with a col_scale whose first dim does not match ceil(T/64):
// CheckOutputShapes must reject it.
TEST_F(ClaGateQuantTilingTest, test_tiling_invalid_col_scale_shape)
{
    ExecuteTestCase(ge::DT_FLOAT16, ge::DT_FLOAT8_E4M3FN, 4, 16, 128, "rint", 36, 0, true, "TND", ge::GRAPH_FAILED, 1);
}
