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
                                    bool withCompileInfo, uint64_t& outKey)
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
                      .NodeInputTd(2, varDtype, ge::FORMAT_ND, ge::FORMAT_ND)
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
    auto st = RunScatterMaxTiling({{16, 8}, {16, 8}}, {{4}, {4}}, {{4, 8}, {4, 8}}, ge::DT_FLOAT, ge::DT_INT32, true,
                                  key);
    EXPECT_EQ(st, ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 0);
}

// int32 dtype + int64 indices
TEST_F(ScatterMaxTiling, test_tiling_int32)
{
    uint64_t key = 0xFFFF;
    auto st = RunScatterMaxTiling({{16, 8}, {16, 8}}, {{4}, {4}}, {{4, 8}, {4, 8}}, ge::DT_INT32, ge::DT_INT64, true,
                                  key);
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
                                  ge::DT_INT32, true, key);
    EXPECT_EQ(st, ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 0);
}

// slice larger than UB budget -> ubTiling aligned/capped, multi-loop
TEST_F(ScatterMaxTiling, test_tiling_big_slice)
{
    uint64_t key = 0xFFFF;
    auto st = RunScatterMaxTiling({{2, 100000}, {2, 100000}}, {{2}, {2}}, {{2, 100000}, {2, 100000}}, ge::DT_FLOAT,
                                  ge::DT_INT32, true, key);
    EXPECT_EQ(st, ge::GRAPH_SUCCESS);
    EXPECT_EQ(key, 0);
}
