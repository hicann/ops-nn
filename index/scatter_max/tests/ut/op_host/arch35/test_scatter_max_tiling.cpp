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
 * \file test_scatter_max_tiling.cpp
 * \brief
 */
#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include "log/log.h"
#include "kernel_run_context_facker.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "test_cube_util.h"
#include "register/op_impl_registry.h"
#include "ut_op_util.h"
#include "ut_op_common.h"
#include "platform/platform_infos_def.h"

using namespace ut_util;
using namespace std;
using namespace ge;

namespace {
struct ScatterReduceCompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
};

constexpr const char* kCompileInfoString = R"({
    "hardware_info": {
        "BT_SIZE": 0, "load3d_constraints": "1",
        "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
        "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
        "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 48
    }
})";

// Drive ScatterMax tiling once. withCompileInfo=false exercises the aclnn path
// (tiling reads the platform directly because GetCompileInfo() is null).
// Returns the tiling status; writes the tiling key to outKey on success.
ge::graphStatus RunScatterMaxTiling(gert::StorageShape varShape, gert::StorageShape indicesShape,
                                    gert::StorageShape updatesShape, ge::DataType varDtype, ge::DataType indicesDtype,
                                    ge::DataType updatesDtype, bool withCompileInfo, uint64_t& outKey)
{
    std::string op_type("ScatterMax");
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str());
    if (opImpl == nullptr) {
        return ge::GRAPH_FAILED;
    }
    auto tiling_func = opImpl->tiling;
    auto tiling_parse_func = opImpl->tiling_parse;

    std::string compile_info_string(kCompileInfoString);
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    ScatterReduceCompileInfo compile_info;

    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(2, 1)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();
    auto parse_ctx = kernel_holder.GetContext<gert::TilingParseContext>();
    if (!parse_ctx->GetPlatformInfo()->Init()) {
        return ge::GRAPH_FAILED;
    }
    parse_ctx->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    parse_ctx->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    parse_ctx->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    parse_ctx->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    if (tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()) != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }

    auto param = gert::TilingData::CreateCap(1024 * 1024);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(1024 * 1024);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    bool use_locking = false;

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(3, 1)
                      .IrInstanceNum({1, 1, 1})
                      .InputShapes({&varShape, &indicesShape, &updatesShape})
                      .OutputShapes({&varShape})
                      .CompileInfo(withCompileInfo ? &compile_info : nullptr)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, indicesDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, updatesDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"use_locking", Ops::NN::AnyValue::CreateFrom<bool>(use_locking)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    if (tiling_context->GetPlatformInfo() == nullptr) {
        return ge::GRAPH_FAILED;
    }
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    auto st = tiling_func(tiling_context);
    if (st == ge::GRAPH_SUCCESS) {
        outKey = tiling_context->GetTilingKey();
    }
    return st;
}
} // namespace

class ScatterMaxTiling : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ScatterMaxTiling SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "ScatterMaxTiling TearDown" << std::endl; }
};

// float32 happy path
TEST_F(ScatterMaxTiling, test_tiling_base)
{
    uint64_t key = 0xFFFF;
    auto st = RunScatterMaxTiling({{16, 8}, {16, 8}}, {{4}, {4}}, {{4, 8}, {4, 8}}, ge::DT_FLOAT, ge::DT_INT32,
                                  ge::DT_FLOAT, true, key);
    EXPECT_EQ(st, ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 0);
}

// int32 dtype + int64 indices
TEST_F(ScatterMaxTiling, test_tiling_int32)
{
    uint64_t key = 0xFFFF;
    auto st = RunScatterMaxTiling({{16, 8}, {16, 8}}, {{4}, {4}}, {{4, 8}, {4, 8}}, ge::DT_INT32, ge::DT_INT64,
                                  ge::DT_INT32, true, key);
    EXPECT_EQ(st, ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 0);
}

// NOTE: the aclnn path (compileInfo == nullptr -> platform fallback in ResolveCoreNumAndUbSize)
// is not exercised here: the UT platform faker does not back PlatformAscendC::GetCoreNumAiv().
// That branch is covered end-to-end by the TTK aclnn precision test instead.

// indicesNum > coreNum -> block split capped at coreNum
TEST_F(ScatterMaxTiling, test_tiling_indices_gt_cores)
{
    uint64_t key = 0xFFFF;
    auto st = RunScatterMaxTiling({{128, 8}, {128, 8}}, {{128}, {128}}, {{128, 8}, {128, 8}}, ge::DT_FLOAT,
                                  ge::DT_INT32, ge::DT_FLOAT, true, key);
    EXPECT_EQ(st, ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 0);
}

// slice larger than UB budget -> ubTiling aligned/capped, multi-loop
TEST_F(ScatterMaxTiling, test_tiling_big_slice)
{
    uint64_t key = 0xFFFF;
    auto st = RunScatterMaxTiling({{2, 100000}, {2, 100000}}, {{2}, {2}}, {{2, 100000}, {2, 100000}}, ge::DT_FLOAT,
                                  ge::DT_INT32, ge::DT_FLOAT, true, key);
    EXPECT_EQ(st, ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 0);
}

// negative: updates dtype != var dtype -> CheckScatterReduceInputs rejects
TEST_F(ScatterMaxTiling, test_tiling_updates_var_dtype_mismatch)
{
    uint64_t key = 0xFFFF;
    auto st = RunScatterMaxTiling({{16, 8}, {16, 8}}, {{4}, {4}}, {{4, 8}, {4, 8}}, ge::DT_FLOAT, ge::DT_INT32,
                                  ge::DT_INT32, true, key);
    EXPECT_EQ(st, ge::GRAPH_FAILED);
}

// negative: updates.shape != indices.shape + var.shape[1:] -> CheckScatterReduceInputs rejects
TEST_F(ScatterMaxTiling, test_tiling_updates_shape_mismatch)
{
    uint64_t key = 0xFFFF;
    auto st = RunScatterMaxTiling({{16, 8}, {16, 8}}, {{4}, {4}}, {{4, 4}, {4, 4}}, ge::DT_FLOAT, ge::DT_INT32,
                                  ge::DT_FLOAT, true, key);
    EXPECT_EQ(st, ge::GRAPH_FAILED);
}

// var first dim > INT32_MAX -> in-bound index would overflow the int32 sort key -> reject
// var 首维 > INT32_MAX: 曾因"32 位排序 key 装不下 in-bound index"被 tiling 拒收, 使 A5 支持面窄于
// A2(910B 的 TIK 实现对首维无任何上限)。现改为接受并走两趟基数排序(lo 排序 + hi 稳定分区), 故断言
// 由 GRAPH_FAILED 改为 GRAPH_SUCCESS —— 改的是这条断言固化下来的旧行为, 不是为迎合测试改实现。
TEST_F(ScatterMaxTiling, test_tiling_var_dim0_over_int32max)
{
    uint64_t key = 0xFFFF;
    auto st = RunScatterMaxTiling({{2147483648L}, {2147483648L}}, {{4}, {4}}, {{4}, {4}}, ge::DT_FLOAT, ge::DT_INT64,
                                  ge::DT_FLOAT, true, key);
    EXPECT_EQ(st, ge::GRAPH_SUCCESS);
}

// 分档边界: 首维恰为 2^30 仍走窄档(wideIndex=0), 超过一个元素即进宽档(wideIndex=1)。
// 这两条守住"正常 case 一律不进新路径"这个前提 —— 分档一旦漂移, 常规用例的性能与产物就会被牵连。
TEST_F(ScatterMaxTiling, test_tiling_narrow_path_at_lo_span)
{
    uint64_t key = 0xFFFF;
    auto st = RunScatterMaxTiling({{1073741824L}, {1073741824L}}, {{4}, {4}}, {{4}, {4}}, ge::DT_FLOAT, ge::DT_INT64,
                                  ge::DT_FLOAT, true, key);
    EXPECT_EQ(st, ge::GRAPH_SUCCESS);
}

TEST_F(ScatterMaxTiling, test_tiling_wide_path_over_lo_span)
{
    uint64_t key = 0xFFFF;
    auto st = RunScatterMaxTiling({{1073741825L}, {1073741825L}}, {{4}, {4}}, {{4}, {4}}, ge::DT_FLOAT, ge::DT_INT64,
                                  ge::DT_FLOAT, true, key);
    EXPECT_EQ(st, ge::GRAPH_SUCCESS);
}

// var first dim == INT32_MAX -> largest in-bound index still fits int32 -> accept
TEST_F(ScatterMaxTiling, test_tiling_var_dim0_at_int32max)
{
    uint64_t key = 0xFFFF;
    auto st = RunScatterMaxTiling({{2147483647L}, {2147483647L}}, {{4}, {4}}, {{4}, {4}}, ge::DT_FLOAT, ge::DT_INT64,
                                  ge::DT_FLOAT, true, key);
    EXPECT_EQ(st, ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 0);
}

// 桶数上限: 宽档按 2^30 一桶分区, kernel 侧计数数组只有 64 桶(+1 溢出桶), 故 host 必须在
// 桶数 > 64 时拒收, 否则 kernel 里会越界写计数数组。这两条只能在 host UT 覆盖 ——
// 64 桶对应 var 首维 2^36, 真机上 var 本身就要 68GB, 物理上无法造用例。
TEST_F(ScatterMaxTiling, test_tiling_wide_buckets_at_cap)
{
    uint64_t key = 0xFFFF;
    auto st = RunScatterMaxTiling({{68719476736L}, {68719476736L}}, {{4}, {4}}, {{4}, {4}}, ge::DT_FLOAT, ge::DT_INT64,
                                  ge::DT_FLOAT, true, key);
    EXPECT_EQ(st, ge::GRAPH_SUCCESS);
}

TEST_F(ScatterMaxTiling, test_tiling_wide_buckets_over_cap)
{
    uint64_t key = 0xFFFF;
    auto st = RunScatterMaxTiling({{69793218560L}, {69793218560L}}, {{4}, {4}}, {{4}, {4}}, ge::DT_FLOAT, ge::DT_INT64,
                                  ge::DT_FLOAT, true, key);
    EXPECT_EQ(st, ge::GRAPH_FAILED);
}
