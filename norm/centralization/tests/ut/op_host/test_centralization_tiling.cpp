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
 * \file test_centralization_tiling.cpp
 * \brief Tiling UT：大 shape（单维/总元素数超 INT32_MAX、2^32、2^34）不得在 Host 侧被截成 32 位。
 */

#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <vector>
#include <gtest/gtest.h>
#include "ut_op_util.h"
#include "test_cube_util.h"

using namespace ut_util;
#include "platform/platform_infos_def.h"
#include "kernel_run_context_facker.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"
#include "../../../op_host/arch35/centralization_tiling.h"
#include "../../../op_kernel/arch35/centralization_tiling_data.h"

using namespace ge;

namespace {
static const char* kCompileInfoStr = R"({
   "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                     "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                     "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                     "UB_SIZE": 253952, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                     "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                     "CORE_NUM": 64, "socVersion": "Ascend950"}
            })";

struct TilingResult {
    uint64_t key{UINT64_MAX};
    bool ok{false};
    CentralizationTilingData td{};
};

static TilingResult RunTiling(const std::vector<int64_t>& dims, ge::DataType dtype, std::vector<int64_t> axes)
{
    TilingResult result;
    std::map<std::string, std::string> soc_version_infos = {{"NpuArch", "3510"}};
    std::map<std::string, std::string> soc_infos;
    std::map<std::string, std::string> aicore_spec;
    std::map<std::string, std::string> intrinsics;
    GetPlatFormInfos(kCompileInfoStr, soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    optiling::CentralizationCompileInfo compile_info;

    const std::string op_type("Centralization");
    auto* impl = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str());
    if (impl == nullptr || impl->tiling == nullptr || impl->tiling_parse == nullptr) {
        return result;
    }
    auto tiling_func = impl->tiling;
    auto tiling_parse_func = impl->tiling_parse;

    std::string compile_info_string(kCompileInfoStr);
    auto kernel_holder = gert::KernelRunContextFaker()
                             .KernelIONum(3, 3)
                             .Inputs({const_cast<char*>(compile_info_string.c_str()),
                                      reinterpret_cast<void*>(&platform_info)})
                             .Outputs({&compile_info})
                             .Build();
    if (!kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init()) {
        return result;
    }
    auto parse_ctx = kernel_holder.GetContext<gert::TilingParseContext>();
    parse_ctx->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    parse_ctx->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    parse_ctx->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    parse_ctx->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    parse_ctx->GetPlatformInfo()->SetPlatformRes("version", soc_version_infos);
    if (tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()) != ge::GRAPH_SUCCESS) {
        return result;
    }

    gert::StorageShape x_shape;
    for (const int64_t dim : dims) {
        x_shape.MutableOriginShape().AppendDim(dim);
        x_shape.MutableStorageShape().AppendDim(dim);
    }
    gert::StorageShape y_shape = x_shape;

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_holder.get());
    if (param == nullptr) {
        return result;
    }
    auto holder = gert::TilingContextFaker()
                      .SetOpType(op_type)
                      .NodeIoNum(1, 1)
                      .IrInstanceNum({1})
                      .InputShapes({&x_shape})
                      .OutputShapes({&y_shape})
                      .NodeAttrs({{"axes", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(axes)}})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();
    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    if (tiling_context == nullptr || tiling_context->GetPlatformInfo() == nullptr) {
        return result;
    }
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    if (tiling_func(tiling_context) != ge::GRAPH_SUCCESS) {
        return result;
    }
    result.key = tiling_context->GetTilingKey();
    auto* raw = tiling_context->GetRawTilingData();
    if (raw != nullptr && raw->GetDataSize() >= sizeof(CentralizationTilingData)) {
        result.td = *reinterpret_cast<const CentralizationTilingData*>(raw->GetData());
    }
    result.ok = true;
    return result;
}
} // namespace

TEST(CentralizationTiling, NormalShapeKeepsCounts)
{
    auto r = RunTiling({2, 3, 4}, ge::DT_FLOAT, {1});
    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.td.totalCount, 24);
    EXPECT_EQ(r.td.reduceCount, 3);
    EXPECT_EQ(r.td.groupCount, 8);
}

TEST(CentralizationTiling, SingleDimOverInt32NotTruncated)
{
    constexpr int64_t kOverInt32 = 3000000000L; // > 2^31 - 1
    auto r = RunTiling({kOverInt32, 1, 1, 1}, ge::DT_FLOAT, {3});
    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.td.totalCount, static_cast<int64_t>(kOverInt32));
    EXPECT_EQ(r.td.groupCount, static_cast<int64_t>(kOverInt32));
    EXPECT_EQ(r.td.reduceCount, 1);
}

TEST(CentralizationTiling, Upboundary2Pow32Accepted)
{
    constexpr int64_t k2Pow32 = 4294967296L; // UINT32_MAX + 1
    auto r = RunTiling({k2Pow32, 1, 1, 1, 1, 1, 1}, ge::DT_FLOAT, {3});
    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.td.totalCount, k2Pow32);
    EXPECT_EQ(r.td.groupCount, k2Pow32);
    EXPECT_EQ(r.td.reduceCount, 1);
}

TEST(CentralizationTiling, UpboundaryReduce2Pow32Accepted)
{
    constexpr int64_t k2Pow32 = 4294967296L;
    auto r = RunTiling({1, k2Pow32, 1}, ge::DT_FLOAT, {});
    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.td.totalCount, k2Pow32);
    EXPECT_EQ(r.td.groupCount, k2Pow32);
    EXPECT_EQ(r.td.reduceCount, 1);
}

TEST(CentralizationTiling, FarOverUint32Accepted)
{
    constexpr int64_t k2Pow34 = 17179869184L; // 2^34
    auto r = RunTiling({1, 1, k2Pow34}, ge::DT_FLOAT, {2});
    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.td.totalCount, k2Pow34);
    EXPECT_EQ(r.td.reduceCount, k2Pow34);
}

TEST(CentralizationTiling, LargeReduceWith2Pow32GroupsAccepted)
{
    constexpr int64_t k2Pow32 = 4294967296L;
    constexpr int64_t kReduce = 16384L; // > kLargeReduceThreshold，走 LargeTrailing
    auto r = RunTiling({k2Pow32, kReduce}, ge::DT_FLOAT, {1});
    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.td.totalCount, k2Pow32 * kReduce);
    EXPECT_EQ(r.td.groupCount, k2Pow32);
    EXPECT_EQ(r.td.reduceCount, kReduce);
}
