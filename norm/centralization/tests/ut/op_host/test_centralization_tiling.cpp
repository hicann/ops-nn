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
    size_t workspace{0};
    CentralizationTilingData td{};
};

static TilingResult RunTilingWithUb(const std::vector<int64_t>& dims, ge::DataType dtype, std::vector<int64_t> axes,
                                    uint64_t ubSize, bool omitAxes = false)
{
    TilingResult result;
    std::map<std::string, std::string> soc_version_infos = {{"NpuArch", "3510"}};
    std::map<std::string, std::string> soc_infos;
    std::map<std::string, std::string> aicore_spec;
    std::map<std::string, std::string> intrinsics;
    std::string compile_info_string(kCompileInfoStr);
    const std::string ubMarker = "\"UB_SIZE\": 253952";
    const size_t ubPos = compile_info_string.find(ubMarker);
    if (ubPos != std::string::npos) {
        compile_info_string.replace(ubPos, ubMarker.size(), "\"UB_SIZE\": " + std::to_string(ubSize));
    }
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

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
    gert::TilingContextFaker faker;
    faker.SetOpType(op_type)
        .NodeIoNum(1, 1)
        .IrInstanceNum({1})
        .InputShapes({&x_shape})
        .OutputShapes({&y_shape})
        .CompileInfo(&compile_info)
        .PlatformInfo(reinterpret_cast<char*>(&platform_info))
        .NodeInputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeOutputTd(0, dtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .TilingData(param.get())
        .Workspace(ws_size);
    if (omitAxes) {
        faker.NodeAttrs({});
    } else {
        faker.NodeAttrs({{"axes", Ops::NN::AnyValue::CreateFrom<std::vector<int64_t>>(axes)}});
    }
    auto holder = faker.Build();
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
    if (auto* ws = tiling_context->GetWorkspaceSizes(1); ws != nullptr) {
        result.workspace = ws[0];
    }
    auto* raw = tiling_context->GetRawTilingData();
    if (raw != nullptr && raw->GetDataSize() >= sizeof(CentralizationTilingData)) {
        result.td = *reinterpret_cast<const CentralizationTilingData*>(raw->GetData());
    }
    result.ok = true;
    return result;
}

static TilingResult RunTiling(const std::vector<int64_t>& dims, ge::DataType dtype, std::vector<int64_t> axes)
{
    return RunTilingWithUb(dims, dtype, axes, 253952UL);
}

static size_t ElementBytes(ge::DataType dtype) { return dtype == ge::DT_FLOAT ? sizeof(float) : sizeof(uint16_t); }

static size_t LiveBufferBytes(const CentralizationTilingData& td, uint32_t key, ge::DataType dtype)
{
    size_t bytes = 0;
    if (key == 7030) {
        bytes = 2UL * td.largeTrailingInputBufferBytes + 2UL * td.largeTrailingOutputBufferBytes +
                td.largeTrailingCalcBufferBytes + td.largeTrailingMeanBufferBytes + 128UL;
    } else if (key == 7000) {
        const int64_t cols = td.ubFactor > 0 ? td.ubFactor : td.alignedReduce;
        const int64_t rows = td.rowsPerLoop > 1 ? td.rowsPerLoop : td.rowsPerBatch;
        bytes = 2UL * static_cast<size_t>(td.bufNum) * static_cast<size_t>(rows * cols) * ElementBytes(dtype);
    } else {
        const bool singleDepth = td.smallContiguous != 0 || td.largeContiguousSmallInner != 0;
        const size_t depth = singleDepth ? 1UL : 2UL;
        bytes = depth * (static_cast<size_t>(td.genericInputBufferBytes) +
                         static_cast<size_t>(td.genericOutputBufferBytes)) +
                static_cast<size_t>(td.genericSumBufferBytes);
    }
    return bytes;
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

TEST(CentralizationTiling, UbBudgetPathMatrix)
{
    struct Case {
        std::vector<int64_t> dims;
        std::vector<int64_t> axes;
        ge::DataType dtype;
        uint64_t expectedKey;
        int64_t expectedRedTile;
        int64_t expectedRows;
        int64_t expectedSlots;
        const char* name;
    };
    const std::vector<Case> cases = {
        {{2, 512, 1}, {1}, ge::DT_FLOAT, 0, 0, 1, 1, "small_mode1_fp32"},
        {{2, 64, 4}, {1}, ge::DT_FLOAT, 0, 0, 1, 16, "small_mode2_fp32"},
        {{2, 300, 8}, {1}, ge::DT_FLOAT, 0, 0, 1, 16, "contiguous_fp32"},
        {{64, 20000, 8}, {1}, ge::DT_FLOAT, 0, 320, 1, 1, "lcsi_fp32"},
        {{64, 20000, 8}, {1}, ge::DT_FLOAT16, 0, 640, 1, 1, "lcsi_fp16"},
        {{64, 20000, 1}, {1}, ge::DT_FLOAT, 7020, 0, 1, 16, "large_7020_row_parallel_fp32"},
        {{20000, 8}, {0}, ge::DT_FLOAT, 7020, 0, 1, 16, "large_7020_reduce_parallel_fp32"},
        {{4, 8, 16, 32}, {0, 2}, ge::DT_FLOAT, 0, 0, 1, 1, "irregular_fp32"},
        {{8, 20000}, {1}, ge::DT_FLOAT, 7030, 0, 1, 16, "large_trailing_fp32"},
        {{128, 20000}, {1}, ge::DT_FLOAT, 7030, 0, 1, 16, "large_trailing_row_parallel_fp32"},
        {{8, 4096}, {1}, ge::DT_FLOAT, 7000, 0, 3, 1, "fast_fp32"},
        {{8, 4096}, {1}, ge::DT_FLOAT16, 7000, 0, 7, 1, "fast_fp16"},
        {{2, 2000, 1}, {1}, ge::DT_FLOAT, 0, 0, 1, 16, "scalar_fp32"},
        {{2, 0, 4}, {1}, ge::DT_FLOAT, 8000, 0, 1, 1, "empty_fp32"},
    };
    for (const auto& c : cases) {
        auto r = RunTiling(c.dims, c.dtype, c.axes);
        ASSERT_TRUE(r.ok) << c.name;
        const size_t live = LiveBufferBytes(r.td, static_cast<uint32_t>(r.key), c.dtype);
        EXPECT_EQ(r.key, c.expectedKey) << c.name;
        EXPECT_LE(live, 253952UL) << c.name;
        if (r.key != 8000) {
            EXPECT_LE(live + 8192UL, 253952UL) << c.name;
        }
        EXPECT_EQ(r.td.largeContiguousReduceTile, c.expectedRedTile) << c.name;
        EXPECT_EQ(r.td.rowsPerLoop, c.expectedRows) << c.name;
        EXPECT_EQ(r.td.numAccSlots, c.expectedSlots) << c.name;
    }
}

TEST(CentralizationTiling, UbBudgetFallbackMatrix)
{
    struct Case {
        std::vector<int64_t> dims;
        std::vector<int64_t> axes;
        ge::DataType dtype;
        uint64_t ubSize;
        uint64_t expectedKey;
        int64_t expectedRedTile;
        const char* name;
    };
    const std::vector<Case> cases = {
        {{8, 4096}, {1}, ge::DT_FLOAT, 65536, 7030, 0, "fast_to_large_trailing_fp32"},
        {{8, 4096}, {1}, ge::DT_FLOAT, 32768, 7020, 0, "fast_to_generic_large_fp32"},
        {{8, 20000}, {1}, ge::DT_FLOAT, 32768, 7020, 0, "large_trailing_to_generic_fp32"},
        {{2, 512, 1}, {1}, ge::DT_FLOAT, 12288, 0, 0, "small_to_scalar_fp32"},
        {{2, 64, 4}, {1}, ge::DT_FLOAT, 24576, 0, 0, "small_to_contiguous_fp32"},
        {{64, 20000, 8}, {1}, ge::DT_FLOAT, 16384, 0, 10, "lcsi_tiny_ub_fp32"},
        {{4, 8, 16, 32}, {0, 2}, ge::DT_FLOAT, 16384, 0, 0, "irregular_tiny_ub_fp32"},
    };
    for (const auto& c : cases) {
        auto r = RunTilingWithUb(c.dims, c.dtype, c.axes, c.ubSize);
        ASSERT_TRUE(r.ok) << c.name;
        const size_t live = LiveBufferBytes(r.td, static_cast<uint32_t>(r.key), c.dtype);
        EXPECT_EQ(r.key, c.expectedKey) << c.name;
        EXPECT_LE(live, c.ubSize) << c.name;
        if (r.key != 8000) {
            EXPECT_LE(live + 8192UL, c.ubSize) << c.name;
        }
        EXPECT_EQ(r.td.largeContiguousReduceTile, c.expectedRedTile) << c.name;
    }
}

TEST(CentralizationTiling, UbBudgetMinimumFit)
{
    // contiguous 的最小 sum buffer 放不下 -> scalar（不返回 GRAPH_FAILED）
    auto a = RunTilingWithUb({2, 300, 8}, ge::DT_FLOAT, {1}, 12288);
    ASSERT_TRUE(a.ok);
    EXPECT_EQ(a.key, 0);
    EXPECT_EQ(a.td.contiguousGeneric, 0);
    EXPECT_EQ(a.td.smallContiguous, 0);
    EXPECT_EQ(a.td.largePath, 0);

    // small mode2 放不下，contiguous 也放不下 -> scalar
    auto b = RunTilingWithUb({2, 64, 4}, ge::DT_FLOAT, {1}, 12288);
    ASSERT_TRUE(b.ok);
    EXPECT_EQ(b.key, 0);
    EXPECT_EQ(b.td.contiguousGeneric, 0);
    EXPECT_EQ(b.td.smallContiguous, 0);

    // irregular 的最小 sum buffer 放不下 -> scalar
    auto c = RunTilingWithUb({4, 8, 16, 32}, ge::DT_FLOAT, {0, 2}, 10240);
    ASSERT_TRUE(c.ok);
    EXPECT_EQ(c.key, 0);
    EXPECT_EQ(c.td.irregularVectorizable, 0);

    // lcsi 最小 tile（1 个 reduce 行）仍可放下
    auto d = RunTilingWithUb({64, 20000, 8}, ge::DT_FLOAT, {1}, 12288);
    ASSERT_TRUE(d.ok);
    EXPECT_EQ(d.key, 0);
    EXPECT_EQ(d.td.largeContiguousSmallInner, 1);
    EXPECT_EQ(d.td.largeContiguousReduceTile, 5);
    EXPECT_LE(LiveBufferBytes(d.td, static_cast<uint32_t>(d.key), ge::DT_FLOAT) + 8192UL, 12288UL);

    // UB 连预留都不够：明确失败，不产生零步长/死循环
    auto e = RunTilingWithUb({2, 512, 1}, ge::DT_FLOAT, {1}, 8192);
    EXPECT_FALSE(e.ok);
}

TEST(CentralizationTiling, UbBudgetFallbackFlags)
{
    auto checkLargeFallback = [](const TilingResult& r, uint64_t expectedKey) {
        EXPECT_EQ(r.key, expectedKey);
        EXPECT_EQ(r.td.largePath, 1);
        EXPECT_EQ(r.td.contiguousGeneric, 0);
        EXPECT_EQ(r.td.smallContiguous, 0);
        EXPECT_EQ(r.td.largeContiguousSmallInner, 0);
        EXPECT_EQ(r.td.irregularVectorizable, 0);
        const size_t expectedWorkspace = 16UL * 1024UL * 1024UL + static_cast<size_t>(r.td.groupCount) *
                                                                      static_cast<size_t>(r.td.alignedCoresPerRow) *
                                                                      32UL;
        EXPECT_EQ(r.workspace, expectedWorkspace);
        EXPECT_GT(r.td.numAccSlots, 0);
        EXPECT_LE(r.td.numAccSlots, 16);
    };

    auto a = RunTilingWithUb({8, 4096}, ge::DT_FLOAT, {1}, 65536);
    ASSERT_TRUE(a.ok);
    checkLargeFallback(a, 7030);
    EXPECT_EQ(a.td.rowParallel, 1);
    auto b = RunTilingWithUb({8, 4096}, ge::DT_FLOAT, {1}, 32768);
    ASSERT_TRUE(b.ok);
    checkLargeFallback(b, 7020);
    EXPECT_EQ(b.td.rowParallel, 1);
    auto c = RunTilingWithUb({8, 20000}, ge::DT_FLOAT, {1}, 32768);
    ASSERT_TRUE(c.ok);
    checkLargeFallback(c, 7020);
    EXPECT_EQ(c.td.rowParallel, 0);
    EXPECT_EQ(c.td.coresPerRow, 8);

    auto d = RunTilingWithUb({2, 512, 1}, ge::DT_FLOAT, {1}, 12288);
    ASSERT_TRUE(d.ok);
    EXPECT_EQ(d.key, 0);
    EXPECT_EQ(d.td.largePath, 0);
    EXPECT_EQ(d.td.contiguousGeneric, 0);
    EXPECT_EQ(d.td.smallContiguous, 0);
    EXPECT_EQ(d.td.largeContiguousSmallInner, 0);
    EXPECT_EQ(d.td.irregularVectorizable, 0);
    EXPECT_EQ(d.workspace, 0UL);
}

TEST(CentralizationTiling, DuplicateAxesDeduplicated)
{
    auto r = RunTiling({2, 3, 4}, ge::DT_FLOAT, {1, 1});
    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.td.reduceRank, 1);
    EXPECT_EQ(r.td.reduceCount, 3);
    EXPECT_EQ(r.td.groupCount, 8);
}

TEST(CentralizationTiling, LongAxisListDeduplicated)
{
    auto r = RunTiling({2, 3, 4}, ge::DT_FLOAT, {0, 1, 2, 0});
    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.td.reduceRank, 3);
    EXPECT_EQ(r.td.reduceCount, 24);
    EXPECT_EQ(r.td.groupCount, 1);
}

TEST(CentralizationTiling, EmptyAxesNoReduce)
{
    auto r = RunTiling({2, 3, 4}, ge::DT_FLOAT, {});
    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.td.reduceRank, 0);
    EXPECT_EQ(r.td.reduceCount, 1);
    EXPECT_EQ(r.td.groupCount, 24);
    EXPECT_EQ(r.key, 7000U);
}

TEST(CentralizationTiling, Rank0Rejected)
{
    auto r = RunTiling({}, ge::DT_FLOAT, {-1});
    EXPECT_FALSE(r.ok);
}

TEST(CentralizationTiling, Rank9RejectedByToolchainLimit)
{
    auto r = RunTiling({2, 1, 1, 1, 1, 1, 1, 1, 4}, ge::DT_FLOAT, {8});
    EXPECT_FALSE(r.ok);
}

TEST(CentralizationTiling, AxesOmittedEqualsExplicitMinusOne)
{
    auto omitted = RunTilingWithUb({2, 3}, ge::DT_FLOAT, {}, 253952UL, true);
    auto explicitMinusOne = RunTilingWithUb({2, 3}, ge::DT_FLOAT, {-1}, 253952UL);
    auto empty = RunTilingWithUb({2, 3}, ge::DT_FLOAT, {}, 253952UL);

    ASSERT_TRUE(omitted.ok);
    ASSERT_TRUE(explicitMinusOne.ok);
    ASSERT_TRUE(empty.ok);

    EXPECT_EQ(omitted.td.reduceRank, 1);
    EXPECT_EQ(omitted.td.reduceCount, 3);
    EXPECT_EQ(omitted.td.groupCount, 2);
    EXPECT_EQ(omitted.key, explicitMinusOne.key);

    EXPECT_EQ(explicitMinusOne.td.reduceRank, 1);
    EXPECT_EQ(explicitMinusOne.td.reduceCount, 3);
    EXPECT_EQ(explicitMinusOne.td.groupCount, 2);

    EXPECT_EQ(empty.td.reduceRank, 0);
    EXPECT_EQ(empty.td.reduceCount, 1);
    EXPECT_EQ(empty.td.groupCount, 6);
    EXPECT_EQ(empty.key, 7000U);
}
