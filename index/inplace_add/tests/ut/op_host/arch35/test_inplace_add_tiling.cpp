/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <type_traits>
#include <vector>

#include "exe_graph/runtime/storage_shape.h"
#include "platform/platform_ascendc.h"
#include "platform/platform_infos_def.h"
#include "register/op_impl_registry.h"
#include "tiling_parse_context_faker.h"
#include "test_cube_util.h"
#include "tiling_context_faker.h"
#include "ut_op_common.h"
#include "../../../../op_host/arch35/inplace_add_tiling.h"
#include "../../../../op_kernel/arch35/inplace_add_tiling_data.h"

namespace {
using optiling::InplaceAddCompileInfo;

constexpr const char* COMPILE_INFO = R"({
    "hardware_info": {
        "BT_SIZE": 0, "load3d_constraints": "1",
        "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
        "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
        "UB_SIZE": 253952, "L2_SIZE": 33554432, "L1_SIZE": 524288,
        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 64
    }
})";
constexpr int64_t DEFAULT_CORE_NUM = 64;
constexpr int64_t DEFAULT_UB_SIZE = 253952;
constexpr uint32_t DCACHE_SIZE = 128U * 1024U;

static_assert(std::is_standard_layout<InplaceAddTilingData>::value, "TilingData must be standard layout");
static_assert(offsetof(InplaceAddTilingData, needCoreNum) == 0, "needCoreNum ABI offset changed");
static_assert(offsetof(InplaceAddTilingData, n) == 4, "n ABI offset changed");
static_assert(offsetof(InplaceAddTilingData, k) == 8, "k ABI offset changed");
static_assert(offsetof(InplaceAddTilingData, innerSize) == 16, "innerSize ABI offset changed");
static_assert(sizeof(InplaceAddTilingData) == 24, "Host and Kernel TilingData ABI changed");

struct PlatformResources {
    std::map<std::string, std::string> socInfos;
    std::map<std::string, std::string> aicoreSpec;
    std::map<std::string, std::string> intrinsics;
};

PlatformResources MakePlatformResources()
{
    PlatformResources resources;
    GetPlatFormInfos(COMPILE_INFO, resources.socInfos, resources.aicoreSpec, resources.intrinsics);
    return resources;
}

bool ConfigurePlatform(gert::TilingContext* context, PlatformResources& resources)
{
    if (context == nullptr || context->GetPlatformInfo() == nullptr) {
        return false;
    }
    context->GetPlatformInfo()->SetPlatformRes("SoCInfo", resources.socInfos);
    context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", resources.aicoreSpec);
    context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", resources.intrinsics);
    return true;
}

struct TilingResult {
    ge::graphStatus status = ge::GRAPH_FAILED;
    InplaceAddTilingData tilingData{};
    uint64_t tilingKey = std::numeric_limits<uint64_t>::max();
    uint32_t blockDim = 0;
    uint32_t localMemorySize = 0;
    size_t workspaceSize = 0;
    size_t expectedLibWorkspaceSize = 0;
    bool outputDescIsNull = false;
};

TilingResult RunInplaceAddTiling(const gert::StorageShape& xShape, const gert::StorageShape& indicesShape,
                                 const gert::StorageShape& vShape, const gert::StorageShape& yShape,
                                 ge::DataType xDtype = ge::DT_FLOAT, ge::DataType indicesDtype = ge::DT_INT32,
                                 ge::DataType vDtype = ge::DT_FLOAT, ge::DataType yDtype = ge::DT_FLOAT,
                                 InplaceAddCompileInfo* suppliedCompileInfo = nullptr, bool includeOutput = true)
{
    TilingResult result;
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("InplaceAdd");
    if (opImpl == nullptr || opImpl->tiling == nullptr) {
        return result;
    }

    PlatformResources resources = MakePlatformResources();
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    InplaceAddCompileInfo defaultCompileInfo;
    defaultCompileInfo.core_num = DEFAULT_CORE_NUM;
    defaultCompileInfo.ub_size = DEFAULT_UB_SIZE;
    InplaceAddCompileInfo* compileInfo = suppliedCompileInfo == nullptr ? &defaultCompileInfo : suppliedCompileInfo;

    auto rawTilingData = gert::TilingData::CreateCap(4096);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(4096);
    auto workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());
    if (rawTilingData == nullptr || workspace == nullptr) {
        return result;
    }

    gert::StorageShape mutableX = xShape;
    gert::StorageShape mutableIndices = indicesShape;
    gert::StorageShape mutableV = vShape;
    gert::StorageShape mutableY = yShape;
    gert::TilingContextFaker faker;
    faker.SetOpType("InplaceAdd")
        .NodeIoNum(3, 1)
        .IrInstanceNum({1, 1, 1}, {includeOutput ? 1U : 0U})
        .InputShapes({&mutableX, &mutableIndices, &mutableV})
        .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
        .NodeInputTd(0, xDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(1, indicesDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .NodeInputTd(2, vDtype, ge::FORMAT_ND, ge::FORMAT_ND)
        .TilingData(rawTilingData.get())
        .Workspace(workspace);
    if (includeOutput) {
        faker.OutputShapes({&mutableY}).NodeOutputTd(0, yDtype, ge::FORMAT_ND, ge::FORMAT_ND);
    }
    faker.CompileInfo(static_cast<const void*>(compileInfo));
    auto holder = faker.Build();

    auto context = holder.GetContext<gert::TilingContext>();
    if (context == nullptr) {
        return result;
    }
    result.outputDescIsNull = context->GetOutputDesc(0) == nullptr;
    if (!ConfigurePlatform(context, resources)) {
        return result;
    }
    result.status = opImpl->tiling(context);
    if (result.status != ge::GRAPH_SUCCESS) {
        return result;
    }

    auto raw = context->GetRawTilingData();
    if (raw != nullptr && raw->GetData() != nullptr) {
        result.tilingData = *reinterpret_cast<const InplaceAddTilingData*>(raw->GetData());
    }
    result.tilingKey = context->GetTilingKey();
    result.blockDim = context->GetBlockDim();
    result.localMemorySize = context->GetLocalMemorySize();
    auto workspaceSizes = context->GetWorkspaceSizes(1);
    if (workspaceSizes != nullptr) {
        result.workspaceSize = workspaceSizes[0];
    }
    platform_ascendc::PlatformAscendC ascendcPlatform(context->GetPlatformInfo());
    result.expectedLibWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    return result;
}

void ExpectSuccessfulSingleKeyTiling(const TilingResult& result)
{
    ASSERT_EQ(result.status, ge::GRAPH_SUCCESS);
    EXPECT_EQ(result.tilingKey, 0U);
    EXPECT_EQ(result.blockDim, static_cast<uint32_t>(result.tilingData.needCoreNum));
    EXPECT_GT(result.tilingData.needCoreNum, 0);
}

ge::graphStatus RunTilingParse(InplaceAddCompileInfo& compileInfo)
{
    auto opImpl = gert::OpImplRegistry::GetInstance().GetOpImpl("InplaceAdd");
    if (opImpl == nullptr || opImpl->tiling_parse == nullptr) {
        return ge::GRAPH_FAILED;
    }

    PlatformResources resources = MakePlatformResources();
    fe::PlatFormInfos platformInfo;
    platformInfo.Init();
    auto holder = gert::KernelRunContextFaker()
                      .KernelIONum(2, 1)
                      .Inputs({const_cast<char*>("{}"), reinterpret_cast<void*>(&platformInfo)})
                      .Outputs({&compileInfo})
                      .Build();
    auto parseContext = holder.GetContext<gert::TilingParseContext>();
    if (parseContext == nullptr || parseContext->GetPlatformInfo() == nullptr ||
        !parseContext->GetPlatformInfo()->Init()) {
        return ge::GRAPH_FAILED;
    }
    parseContext->GetPlatformInfo()->SetPlatformRes("SoCInfo", resources.socInfos);
    parseContext->GetPlatformInfo()->SetPlatformRes("AICoreSpec", resources.aicoreSpec);
    parseContext->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    parseContext->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", resources.intrinsics);
    return opImpl->tiling_parse(holder.GetContext<gert::KernelContext>());
}

gert::StorageShape MakeStorageShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape result;
    for (const int64_t dim : dims) {
        result.MutableOriginShape().AppendDim(dim);
        result.MutableStorageShape().AppendDim(dim);
    }
    return result;
}
} // namespace

class InplaceAddTilingTest : public testing::Test {};

TEST_F(InplaceAddTilingTest, tilingDataLayoutMatchesKernelAbi)
{
    EXPECT_EQ(sizeof(InplaceAddTilingData), 24U);
    EXPECT_EQ(offsetof(InplaceAddTilingData, needCoreNum), 0U);
    EXPECT_EQ(offsetof(InplaceAddTilingData, n), 4U);
    EXPECT_EQ(offsetof(InplaceAddTilingData, k), 8U);
    EXPECT_EQ(offsetof(InplaceAddTilingData, innerSize), 16U);
}

TEST_F(InplaceAddTilingTest, rankOneSuccess)
{
    auto result = RunInplaceAddTiling({{4}, {4}}, {{2}, {2}}, {{2}, {2}}, {{4}, {4}});
    ExpectSuccessfulSingleKeyTiling(result);
    EXPECT_EQ(result.tilingData.n, 4);
    EXPECT_EQ(result.tilingData.k, 2);
    EXPECT_EQ(result.tilingData.innerSize, 1);
}

TEST_F(InplaceAddTilingTest, rankEightSuccess)
{
    auto result = RunInplaceAddTiling({{2, 2, 1, 1, 1, 1, 1, 3}, {2, 2, 1, 1, 1, 1, 1, 3}}, {{1}, {1}},
                                      {{1, 2, 1, 1, 1, 1, 1, 3}, {1, 2, 1, 1, 1, 1, 1, 3}},
                                      {{2, 2, 1, 1, 1, 1, 1, 3}, {2, 2, 1, 1, 1, 1, 1, 3}});
    ExpectSuccessfulSingleKeyTiling(result);
    EXPECT_EQ(result.tilingData.innerSize, 6);
}

TEST_F(InplaceAddTilingTest, multiCoreSplitAndWorkspace)
{
    auto result = RunInplaceAddTiling({{64, 128}, {64, 128}}, {{16}, {16}}, {{16, 128}, {16, 128}},
                                      {{64, 128}, {64, 128}}, ge::DT_FLOAT16, ge::DT_INT32, ge::DT_FLOAT16,
                                      ge::DT_FLOAT16);
    ExpectSuccessfulSingleKeyTiling(result);
    EXPECT_EQ(result.tilingData.needCoreNum, 10);
    EXPECT_EQ(result.tilingData.n, 64);
    EXPECT_EQ(result.tilingData.k, 16);
    EXPECT_EQ(result.tilingData.innerSize, 128);
    EXPECT_EQ(result.localMemorySize, static_cast<uint32_t>(DEFAULT_UB_SIZE) - DCACHE_SIZE);
    EXPECT_EQ(result.workspaceSize, result.expectedLibWorkspaceSize);
    EXPECT_GT(result.workspaceSize, 0U);
}

TEST_F(InplaceAddTilingTest, allThirteenSupportedDtypesUseKeyZero)
{
    const std::vector<ge::DataType> dtypes = {
        ge::DT_FLOAT16, ge::DT_FLOAT,  ge::DT_BF16,   ge::DT_INT8,   ge::DT_INT16,     ge::DT_INT32,    ge::DT_INT64,
        ge::DT_UINT8,   ge::DT_UINT16, ge::DT_UINT32, ge::DT_UINT64, ge::DT_COMPLEX32, ge::DT_COMPLEX64};
    ASSERT_EQ(dtypes.size(), 13U);
    for (const ge::DataType dtype : dtypes) {
        SCOPED_TRACE(static_cast<int32_t>(dtype));
        auto result = RunInplaceAddTiling({{4, 6}, {4, 6}}, {{2}, {2}}, {{2, 6}, {2, 6}}, {{4, 6}, {4, 6}}, dtype,
                                          ge::DT_INT32, dtype, dtype);
        ExpectSuccessfulSingleKeyTiling(result);
    }
}

TEST_F(InplaceAddTilingTest, emptyIndicesSuccess)
{
    auto result = RunInplaceAddTiling({{4, 8}, {4, 8}}, {{0}, {0}}, {{0, 8}, {0, 8}}, {{4, 8}, {4, 8}});
    ExpectSuccessfulSingleKeyTiling(result);
    EXPECT_EQ(result.tilingData.k, 0);
    EXPECT_EQ(result.tilingData.innerSize, 8);
}

TEST_F(InplaceAddTilingTest, emptyFirstDimensionAndIndicesSuccess)
{
    auto result = RunInplaceAddTiling({{0, 8}, {0, 8}}, {{0}, {0}}, {{0, 8}, {0, 8}}, {{0, 8}, {0, 8}});
    ExpectSuccessfulSingleKeyTiling(result);
    EXPECT_EQ(result.tilingData.needCoreNum, 1);
    EXPECT_EQ(result.tilingData.n, 0);
    EXPECT_EQ(result.tilingData.k, 0);
    EXPECT_EQ(result.tilingData.innerSize, 8);
}

TEST_F(InplaceAddTilingTest, emptyTailSuccess)
{
    auto result = RunInplaceAddTiling({{4, 0, 3}, {4, 0, 3}}, {{2}, {2}}, {{2, 0, 3}, {2, 0, 3}},
                                      {{4, 0, 3}, {4, 0, 3}});
    ExpectSuccessfulSingleKeyTiling(result);
    EXPECT_EQ(result.tilingData.innerSize, 0);
    EXPECT_EQ(result.tilingData.needCoreNum, 1);
}

TEST_F(InplaceAddTilingTest, multipleEmptyTailAxesSuccess)
{
    auto result = RunInplaceAddTiling({{4, 0, 7, 0}, {4, 0, 7, 0}}, {{2}, {2}}, {{2, 0, 7, 0}, {2, 0, 7, 0}},
                                      {{4, 0, 7, 0}, {4, 0, 7, 0}});
    ExpectSuccessfulSingleKeyTiling(result);
    EXPECT_EQ(result.tilingData.innerSize, 0);
}

TEST_F(InplaceAddTilingTest, everyTailAxisCanBeEmpty)
{
    for (size_t zeroAxis = 1; zeroAxis <= 7; ++zeroAxis) {
        SCOPED_TRACE(zeroAxis);
        std::vector<int64_t> xDims(zeroAxis + 1, 1);
        xDims[0] = 4;
        xDims[zeroAxis] = 0;
        std::vector<int64_t> vDims = xDims;
        vDims[0] = 2;
        auto result = RunInplaceAddTiling(MakeStorageShape(xDims), {{2}, {2}}, MakeStorageShape(vDims),
                                          MakeStorageShape(xDims));
        ExpectSuccessfulSingleKeyTiling(result);
        EXPECT_EQ(result.tilingData.innerSize, 0);
        EXPECT_EQ(result.tilingData.needCoreNum, 1);
    }
}

TEST_F(InplaceAddTilingTest, finalEmptyTailAxisDominatesOverflowingPrefix)
{
    constexpr int64_t large = std::numeric_limits<int64_t>::max() / 2 + 1;
    const std::vector<int64_t> xDims = {4, large, 3, 1, 1, 1, 1, 0};
    const std::vector<int64_t> vDims = {2, large, 3, 1, 1, 1, 1, 0};
    auto result = RunInplaceAddTiling(MakeStorageShape(xDims), {{2}, {2}}, MakeStorageShape(vDims),
                                      MakeStorageShape(xDims));
    ExpectSuccessfulSingleKeyTiling(result);
    EXPECT_EQ(result.tilingData.innerSize, 0);
    EXPECT_EQ(result.tilingData.needCoreNum, 1);
}

TEST_F(InplaceAddTilingTest, int32MaxNUsesInt64CeilDivAndEndpoint)
{
    constexpr int64_t maxN = std::numeric_limits<int32_t>::max();
    auto result = RunInplaceAddTiling({{maxN}, {maxN}}, {{0}, {0}}, {{0}, {0}}, {{maxN}, {maxN}});
    ExpectSuccessfulSingleKeyTiling(result);
    EXPECT_EQ(result.tilingData.n, std::numeric_limits<int32_t>::max());
    EXPECT_EQ(result.tilingData.needCoreNum, 64);

    // The kernel derives its own flat, 512B-aligned slice from these three fields;
    // reproduce that arithmetic here so the endpoint stays covered on the host side.
    // The tensor is rank 1 float (rowSize == 1), so the copy phase spans n elements.
    constexpr int64_t alignElems = 512 / static_cast<int64_t>(sizeof(float));
    const int64_t coreNum = static_cast<int64_t>(result.tilingData.needCoreNum);
    const int64_t total = static_cast<int64_t>(result.tilingData.n) * result.tilingData.innerSize;
    int64_t perCore = (total + coreNum - 1) / coreNum;
    perCore = ((perCore + alignElems - 1) / alignElems) * alignElems;
    const int64_t unboundedEnd = (coreNum - 1) * perCore + perCore;
    EXPECT_GT(unboundedEnd, total);
    EXPECT_EQ(std::min(unboundedEnd, total), maxN);
    EXPECT_LT((coreNum - 1) * perCore, total);
}

TEST_F(InplaceAddTilingTest, platformDerivedCompileInfoSuccess)
{
    // The host UT platform faker does not expose PlatformAscendC::GetCoreNumAiv()
    // to a tiling context without compileInfo. Exercise the same platform data
    // through the registered tiling-parse callback before invoking tiling.
    InplaceAddCompileInfo compileInfo;
    ASSERT_EQ(RunTilingParse(compileInfo), ge::GRAPH_SUCCESS);
    auto result = RunInplaceAddTiling({{8, 4}, {8, 4}}, {{2}, {2}}, {{2, 4}, {2, 4}}, {{8, 4}, {8, 4}}, ge::DT_FLOAT,
                                      ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT, &compileInfo);
    ExpectSuccessfulSingleKeyTiling(result);
    EXPECT_EQ(result.localMemorySize, static_cast<uint32_t>(DEFAULT_UB_SIZE) - DCACHE_SIZE);
    EXPECT_EQ(result.workspaceSize, result.expectedLibWorkspaceSize);
}

TEST_F(InplaceAddTilingTest, tilingParseLoadsPlatformCoreAndUb)
{
    InplaceAddCompileInfo compileInfo;
    ASSERT_EQ(RunTilingParse(compileInfo), ge::GRAPH_SUCCESS);
    EXPECT_EQ(compileInfo.core_num, DEFAULT_CORE_NUM);
    EXPECT_EQ(compileInfo.ub_size, DEFAULT_UB_SIZE);
}

TEST_F(InplaceAddTilingTest, oneCoreBoundarySuccess)
{
    InplaceAddCompileInfo compileInfo;
    compileInfo.core_num = 1;
    compileInfo.ub_size = DEFAULT_UB_SIZE;
    auto result = RunInplaceAddTiling({{64, 128}, {64, 128}}, {{16}, {16}}, {{16, 128}, {16, 128}},
                                      {{64, 128}, {64, 128}}, ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT,
                                      &compileInfo);
    ExpectSuccessfulSingleKeyTiling(result);
    EXPECT_EQ(result.tilingData.needCoreNum, 1);
    EXPECT_EQ(result.tilingData.n, 64);
    EXPECT_EQ(result.tilingData.innerSize, 128);
}

TEST_F(InplaceAddTilingTest, rejectZeroCoreCount)
{
    InplaceAddCompileInfo compileInfo;
    compileInfo.core_num = 0;
    compileInfo.ub_size = DEFAULT_UB_SIZE;
    auto result = RunInplaceAddTiling({{4, 8}, {4, 8}}, {{1}, {1}}, {{1, 8}, {1, 8}}, {{4, 8}, {4, 8}}, ge::DT_FLOAT,
                                      ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT, &compileInfo);
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}

TEST_F(InplaceAddTilingTest, rejectCoreCountAboveInt32)
{
    InplaceAddCompileInfo compileInfo;
    compileInfo.core_num = static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1;
    compileInfo.ub_size = DEFAULT_UB_SIZE;
    auto result = RunInplaceAddTiling({{4, 8}, {4, 8}}, {{1}, {1}}, {{1, 8}, {1, 8}}, {{4, 8}, {4, 8}}, ge::DT_FLOAT,
                                      ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT, &compileInfo);
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}

TEST_F(InplaceAddTilingTest, rejectUbEqualToDcacheReserve)
{
    InplaceAddCompileInfo compileInfo;
    compileInfo.core_num = DEFAULT_CORE_NUM;
    compileInfo.ub_size = DCACHE_SIZE;
    auto result = RunInplaceAddTiling({{4, 8}, {4, 8}}, {{1}, {1}}, {{1, 8}, {1, 8}}, {{4, 8}, {4, 8}}, ge::DT_FLOAT,
                                      ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT, &compileInfo);
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}

TEST_F(InplaceAddTilingTest, ubOneByteAboveReserveSuccess)
{
    InplaceAddCompileInfo compileInfo;
    compileInfo.core_num = DEFAULT_CORE_NUM;
    compileInfo.ub_size = DCACHE_SIZE + 1;
    auto result = RunInplaceAddTiling({{4, 8}, {4, 8}}, {{1}, {1}}, {{1, 8}, {1, 8}}, {{4, 8}, {4, 8}}, ge::DT_FLOAT,
                                      ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT, &compileInfo);
    ExpectSuccessfulSingleKeyTiling(result);
    EXPECT_EQ(result.localMemorySize, 1U);
}

TEST_F(InplaceAddTilingTest, rejectLocalMemorySizeAboveUint32)
{
    InplaceAddCompileInfo compileInfo;
    compileInfo.core_num = DEFAULT_CORE_NUM;
    compileInfo.ub_size = static_cast<int64_t>(DCACHE_SIZE) +
                          static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) + 1;
    auto result = RunInplaceAddTiling({{4, 8}, {4, 8}}, {{1}, {1}}, {{1, 8}, {1, 8}}, {{4, 8}, {4, 8}}, ge::DT_FLOAT,
                                      ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT, &compileInfo);
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}

TEST_F(InplaceAddTilingTest, rejectRankZeroX)
{
    auto result = RunInplaceAddTiling({{}, {}}, {{0}, {0}}, {{}, {}}, {{}, {}});
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}

TEST_F(InplaceAddTilingTest, rejectRankNineX)
{
    auto result = RunInplaceAddTiling({{1, 1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1, 1, 1}}, {{1}, {1}},
                                      {{1, 1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1, 1, 1}},
                                      {{1, 1, 1, 1, 1, 1, 1, 1, 1}, {1, 1, 1, 1, 1, 1, 1, 1, 1}});
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}

TEST_F(InplaceAddTilingTest, rejectNonOneDimensionalIndices)
{
    auto result = RunInplaceAddTiling({{4, 8}, {4, 8}}, {{1, 1}, {1, 1}}, {{1, 8}, {1, 8}}, {{4, 8}, {4, 8}});
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}

TEST_F(InplaceAddTilingTest, rejectValueRankMismatch)
{
    auto result = RunInplaceAddTiling({{4, 8}, {4, 8}}, {{1}, {1}}, {{1}, {1}}, {{4, 8}, {4, 8}});
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}

TEST_F(InplaceAddTilingTest, rejectOutputRankMismatch)
{
    auto result = RunInplaceAddTiling({{4, 8}, {4, 8}}, {{1}, {1}}, {{1, 8}, {1, 8}}, {{32}, {32}});
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}

TEST_F(InplaceAddTilingTest, rejectValueFirstDimensionMismatch)
{
    auto result = RunInplaceAddTiling({{4, 8}, {4, 8}}, {{2}, {2}}, {{1, 8}, {1, 8}}, {{4, 8}, {4, 8}});
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}

TEST_F(InplaceAddTilingTest, rejectEmptyNWithNonEmptyK)
{
    auto result = RunInplaceAddTiling({{0, 8}, {0, 8}}, {{1}, {1}}, {{1, 8}, {1, 8}}, {{0, 8}, {0, 8}});
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}

TEST_F(InplaceAddTilingTest, rejectValueTailMismatch)
{
    auto result = RunInplaceAddTiling({{4, 8}, {4, 8}}, {{2}, {2}}, {{2, 7}, {2, 7}}, {{4, 8}, {4, 8}});
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}

TEST_F(InplaceAddTilingTest, rejectOutputDimensionMismatch)
{
    auto result = RunInplaceAddTiling({{4, 8}, {4, 8}}, {{2}, {2}}, {{2, 8}, {2, 8}}, {{4, 7}, {4, 7}});
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}

TEST_F(InplaceAddTilingTest, rejectNegativeN)
{
    auto result = RunInplaceAddTiling({{-1, 8}, {-1, 8}}, {{0}, {0}}, {{0, 8}, {0, 8}}, {{-1, 8}, {-1, 8}});
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}

TEST_F(InplaceAddTilingTest, rejectNegativeK)
{
    auto result = RunInplaceAddTiling({{4, 8}, {4, 8}}, {{-1}, {-1}}, {{-1, 8}, {-1, 8}}, {{4, 8}, {4, 8}});
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}

TEST_F(InplaceAddTilingTest, rejectNegativeTailDimension)
{
    auto result = RunInplaceAddTiling({{4, -1}, {4, -1}}, {{2}, {2}}, {{2, -1}, {2, -1}}, {{4, -1}, {4, -1}});
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}

TEST_F(InplaceAddTilingTest, rejectNAboveInt32)
{
    constexpr int64_t overflow = static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1;
    auto result = RunInplaceAddTiling({{overflow, 1}, {overflow, 1}}, {{0}, {0}}, {{0, 1}, {0, 1}},
                                      {{overflow, 1}, {overflow, 1}});
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}

TEST_F(InplaceAddTilingTest, rejectKAboveInt32)
{
    constexpr int64_t overflow = static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1;
    auto result = RunInplaceAddTiling({{1, 1}, {1, 1}}, {{overflow}, {overflow}}, {{overflow, 1}, {overflow, 1}},
                                      {{1, 1}, {1, 1}});
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}

TEST_F(InplaceAddTilingTest, rejectIndicesDtypeOtherThanInt32)
{
    auto result = RunInplaceAddTiling({{4, 8}, {4, 8}}, {{2}, {2}}, {{2, 8}, {2, 8}}, {{4, 8}, {4, 8}}, ge::DT_FLOAT,
                                      ge::DT_INT64, ge::DT_FLOAT, ge::DT_FLOAT);
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}

TEST_F(InplaceAddTilingTest, rejectMismatchedValueDtype)
{
    auto result = RunInplaceAddTiling({{4, 8}, {4, 8}}, {{2}, {2}}, {{2, 8}, {2, 8}}, {{4, 8}, {4, 8}}, ge::DT_FLOAT16,
                                      ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT16);
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}

TEST_F(InplaceAddTilingTest, rejectUnsupportedDataDtype)
{
    auto result = RunInplaceAddTiling({{4, 8}, {4, 8}}, {{2}, {2}}, {{2, 8}, {2, 8}}, {{4, 8}, {4, 8}}, ge::DT_DOUBLE,
                                      ge::DT_INT32, ge::DT_DOUBLE, ge::DT_DOUBLE);
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}

TEST_F(InplaceAddTilingTest, rejectMismatchedOutputDtype)
{
    auto result = RunInplaceAddTiling({{4, 8}, {4, 8}}, {{2}, {2}}, {{2, 8}, {2, 8}}, {{4, 8}, {4, 8}}, ge::DT_FLOAT,
                                      ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT16);
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}

TEST_F(InplaceAddTilingTest, rejectMissingOutputDesc)
{
    auto result = RunInplaceAddTiling({{4, 8}, {4, 8}}, {{2}, {2}}, {{2, 8}, {2, 8}}, {{4, 8}, {4, 8}}, ge::DT_FLOAT,
                                      ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT, nullptr, false);
    EXPECT_TRUE(result.outputDescIsNull);
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}

TEST_F(InplaceAddTilingTest, rejectInnerSizeOverflow)
{
    constexpr int64_t large = std::numeric_limits<int64_t>::max() / 2 + 1;
    auto result = RunInplaceAddTiling({{1, large, 3}, {1, large, 3}}, {{0}, {0}}, {{0, large, 3}, {0, large, 3}},
                                      {{1, large, 3}, {1, large, 3}});
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}

TEST_F(InplaceAddTilingTest, rejectComplexComponentCountOverflow)
{
    constexpr int64_t large = std::numeric_limits<int64_t>::max() / 2 + 1;
    auto result = RunInplaceAddTiling({{1, large}, {1, large}}, {{0}, {0}}, {{0, large}, {0, large}},
                                      {{1, large}, {1, large}}, ge::DT_COMPLEX32, ge::DT_INT32, ge::DT_COMPLEX32,
                                      ge::DT_COMPLEX32);
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}

TEST_F(InplaceAddTilingTest, rejectXElementCountOverflow)
{
    constexpr int64_t large = std::numeric_limits<int64_t>::max() / 2 + 1;
    auto result = RunInplaceAddTiling({{2, large}, {2, large}}, {{0}, {0}}, {{0, large}, {0, large}},
                                      {{2, large}, {2, large}});
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}

TEST_F(InplaceAddTilingTest, rejectVElementCountOverflow)
{
    constexpr int64_t large = std::numeric_limits<int64_t>::max() / 2 + 1;
    auto result = RunInplaceAddTiling({{1, large}, {1, large}}, {{2}, {2}}, {{2, large}, {2, large}},
                                      {{1, large}, {1, large}});
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}

TEST_F(InplaceAddTilingTest, rejectTotalWorkOverflow)
{
    constexpr int64_t large = std::numeric_limits<int64_t>::max() / 2 + 1;
    auto result = RunInplaceAddTiling({{1, large}, {1, large}}, {{1}, {1}}, {{1, large}, {1, large}},
                                      {{1, large}, {1, large}});
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}

TEST_F(InplaceAddTilingTest, rejectGmByteOffsetOverflow)
{
    constexpr int64_t large = std::numeric_limits<int64_t>::max() / 4 + 1;
    auto result = RunInplaceAddTiling({{1, large}, {1, large}}, {{0}, {0}}, {{0, large}, {0, large}},
                                      {{1, large}, {1, large}}, ge::DT_FLOAT, ge::DT_INT32, ge::DT_FLOAT, ge::DT_FLOAT);
    EXPECT_EQ(result.status, ge::GRAPH_FAILED);
}
