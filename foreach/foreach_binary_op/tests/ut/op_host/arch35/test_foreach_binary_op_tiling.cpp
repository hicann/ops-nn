/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_foreach_binary_op_tiling.cpp
 * \brief arch35 (Ascend950) tiling UT for foreach_binary_op.
 *        Covers TilingKey = op_code * 4 + dtypeIdx over all 16 op_code x dtype combos,
 *        tensor_count boundary, empty (no-core) split, large/unaligned count, and
 *        out-of-range op_code error path.
 */

#include <iostream>
#include <vector>
#include <gtest/gtest.h>

#include "log/log.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "../../../../op_host/arch35/foreach_binary_op_tiling_arch35.h"

using namespace std;
using namespace ge;

class ForeachBinaryOpTilingArch35 : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ForeachBinaryOpTilingArch35 SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "ForeachBinaryOpTilingArch35 TearDown" << std::endl; }
};

// dtypeIdx: FP16=0, FP32=1, INT32=2, BF16=3  (must match GetDtypeIdx in tiling impl)
static int64_t DtypeIdx(ge::DataType dt)
{
    switch (dt) {
        case ge::DT_FLOAT16: return 0;
        case ge::DT_FLOAT:   return 1;
        case ge::DT_INT32:   return 2;
        case ge::DT_BF16:    return 3;
        default:             return 0;
    }
}

// Build a tiling context for ForeachBinaryOp and run the tiling function.
// tensorShapes: per-tensor element shape, applied identically to x1[i], x2[i], y[i].
static void RunTilingCase(int64_t opCode, ge::DataType dtype,
                          const std::vector<std::vector<int64_t>>& tensorShapes,
                          ge::graphStatus expectStatus, int64_t expectTilingKey)
{
    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    string compile_info_string = R"({
                                    "hardware_info": {
                                        "BT_SIZE": 0,
                                        "load3d_constraints": "1",
                                        "Intrinsic_fix_pipe_l0c2out": false,
                                        "Intrinsic_data_move_l12ub": true,
                                        "Intrinsic_data_move_l0c2ub": true,
                                        "Intrinsic_data_move_out2l1_nd2nz": false,
                                        "UB_SIZE": 262144,
                                        "L2_SIZE": 33554432,
                                        "L1_SIZE": 524288,
                                        "L0A_SIZE": 65536,
                                        "L0B_SIZE": 65536,
                                        "L0C_SIZE": 131072,
                                        "CORE_NUM": 64
                                    }
                                })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    ForeachBinaryOpCompileInfo compile_info;

    std::string op_type("ForeachBinaryOp");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    const size_t K = tensorShapes.size();
    std::vector<gert::StorageShape> x1Shapes(K);
    std::vector<gert::StorageShape> x2Shapes(K);
    std::vector<gert::StorageShape> yShapes(K);
    for (size_t i = 0; i < K; ++i) {
        for (auto d : tensorShapes[i]) {
            x1Shapes[i].MutableStorageShape().AppendDim(d);
            x1Shapes[i].MutableOriginShape().AppendDim(d);
            x2Shapes[i].MutableStorageShape().AppendDim(d);
            x2Shapes[i].MutableOriginShape().AppendDim(d);
            yShapes[i].MutableStorageShape().AppendDim(d);
            yShapes[i].MutableOriginShape().AppendDim(d);
        }
    }
    std::vector<gert::StorageShape*> inputShapes;
    for (size_t i = 0; i < K; ++i) inputShapes.push_back(&x1Shapes[i]);
    for (size_t i = 0; i < K; ++i) inputShapes.push_back(&x2Shapes[i]);
    std::vector<gert::StorageShape*> outputShapes;
    for (size_t i = 0; i < K; ++i) outputShapes.push_back(&yShapes[i]);

    auto param = gert::TilingData::CreateCap(8192);
    ASSERT_NE(param, nullptr);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(2, 1)
                      .IrInstanceNum({static_cast<uint32_t>(K), static_cast<uint32_t>(K)},
                                     {static_cast<uint32_t>(K)})
                      .InputShapes(inputShapes)
                      .OutputShapes(outputShapes)
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"op_code", Ops::NN::AnyValue::CreateFrom<int64_t>(opCode)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context, nullptr);

    EXPECT_EQ(tiling_func(tiling_context), expectStatus);
    if (expectStatus == ge::GRAPH_SUCCESS) {
        EXPECT_EQ(static_cast<int64_t>(tiling_context->GetTilingKey()), expectTilingKey);
    }
}

// ---- 16 combos: op_code (0..3) x dtype (FP16/FP32/INT32/BF16) ----
// single tensor per list, aligned size 32 elements.
TEST_F(ForeachBinaryOpTilingArch35, all_op_dtype_combos)
{
    const std::vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_FLOAT, ge::DT_INT32, ge::DT_BF16};
    std::vector<std::vector<int64_t>> shapes = {{32}};
    for (int64_t opCode = 0; opCode < 4; ++opCode) {
        for (auto dt : dtypes) {
            int64_t expectKey = opCode * 4 + DtypeIdx(dt);
            RunTilingCase(opCode, dt, shapes, ge::GRAPH_SUCCESS, expectKey);
        }
    }
}

// ---- white-box: per-core assignment when a core spans many tensors ----
// A foreach op is inherently multi-tensor, but the gert TilingContextFaker cannot build a tiling
// context with dynamic-input instance count > 1 together with CompileInfo/PlatformInfo (both come back
// null). So AssignCoresToTensors is exercised directly with real 16-tensor data, covering the
// cross-tensor accumulation in both the start and end scan loops plus the trailing-empty-core break.
TEST_F(ForeachBinaryOpTilingArch35, multi_tensor_assign_cores)
{
    constexpr int32_t kTensors = 16;
    constexpr int64_t kPerTensor = 64;
    constexpr int64_t kTotal = kTensors * kPerTensor;   // 1024

    ForeachBinaryOpTilingDataHost td{};
    for (int32_t t = 0; t < kTensors; ++t) td.tensorDataCountList[t] = kPerTensor;

    // 2 cores x 512 elements: each core spans 8 tensors and crosses tensor boundaries.
    EXPECT_EQ(optiling::AssignCoresToTensors(&td, kTensors, kTotal, 2, 512), 2);
    // core 0 -> elements [0, 511] -> tensors [0, 7]
    EXPECT_EQ(td.tensorStartList[0], 0);
    EXPECT_EQ(td.tensorStartOffsetList[0], 0);
    EXPECT_EQ(td.tensorEndList[0], 7);
    EXPECT_EQ(td.tensorEndOffsetList[0], 63);
    // core 1 -> elements [512, 1023] -> tensors [8, 15]
    EXPECT_EQ(td.tensorStartList[1], 8);
    EXPECT_EQ(td.tensorStartOffsetList[1], 0);
    EXPECT_EQ(td.tensorEndList[1], 15);
    EXPECT_EQ(td.tensorEndOffsetList[1], 63);

    // More cores than the data needs: the trailing cores start past the end and break out, so the
    // returned used-core count is clamped to what actually holds elements.
    ForeachBinaryOpTilingDataHost td2{};
    for (int32_t t = 0; t < kTensors; ++t) td2.tensorDataCountList[t] = kPerTensor;
    EXPECT_EQ(optiling::AssignCoresToTensors(&td2, kTensors, kTotal, 5, 512), 2);
}

// ---- empty input: total elements == 0 -> needCoreNum = 0, still SUCCESS ----
TEST_F(ForeachBinaryOpTilingArch35, empty_no_core_split)
{
    std::vector<std::vector<int64_t>> shapes = {{0}};
    RunTilingCase(0 /*add*/, ge::DT_FLOAT, shapes, ge::GRAPH_SUCCESS, 0 * 4 + 1);
}

// ---- large, unaligned count spanning many cores (per-core split + alignment) ----
// One instance per dynamic input, per the repo-wide tiling-UT convention (the gert faker does not
// support instance count > 1); the cross-tensor assignment is covered by multi_tensor_assign_cores
// above and list traversal by the infershape UT. A single large tensor (150004, not a multiple of 32)
// drives the full tiling func through the multi-core split and up-alignment path.
TEST_F(ForeachBinaryOpTilingArch35, large_unaligned_count)
{
    std::vector<std::vector<int64_t>> shapes = {{150004}};
    RunTilingCase(1 /*sub*/, ge::DT_FLOAT16, shapes, ge::GRAPH_SUCCESS, 1 * 4 + 0);
}

// ---- out-of-range op_code -> tiling failure ----
TEST_F(ForeachBinaryOpTilingArch35, invalid_op_code)
{
    std::vector<std::vector<int64_t>> shapes = {{32}};
    RunTilingCase(4 /*invalid*/, ge::DT_FLOAT, shapes, ge::GRAPH_FAILED, 0);
}
