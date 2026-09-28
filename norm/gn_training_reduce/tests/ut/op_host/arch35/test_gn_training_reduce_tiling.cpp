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
 * \file test_gn_training_reduce_tiling.cpp
 * \brief GNTrainingReduce host tiling unit tests (ascend950 / arch35).
 */

#include <cstdint>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "kernel_run_context_facker.h"
#include "platform/platform_infos_def.h"
#include "test_cube_util.h"

#include "../../../../op_host/arch35/gn_training_reduce_tiling_arch35.h"
#include "../../../../op_kernel/arch35/gn_training_reduce_tiling_struct.h"

namespace GNTrainingReduceTilingUT {
using namespace ge;

namespace {

constexpr const char* kOpType = "GNTrainingReduce";

// Platform mock consumed by the tiling formulas: AIV core num (SoCInfo.ai_core_cnt)
// and UB bytes (AICoreSpec.ub_size).  blockSize / cacheLine are compile-time
// constants of Ops::Base::GetUbBlockSize() / GetCacheLineSize() (32 / 256).
constexpr int64_t kCoreNum = 24;
constexpr int64_t kUbSize = 248 * 1024; // 253952
constexpr uint64_t kTilingDataCap = 4096U;

const char* const kCompileInfo = R"({
 "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                   "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
                   "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                   "UB_SIZE": 253952, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                   "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                   "CORE_NUM": 24}
})";

struct TilingResult {
    bool ok = false;
    uint64_t tilingKey = 0U;
    uint64_t blockDim = 0U;
    std::vector<uint8_t> tilingData;
};

gert::StorageShape MakeStorageShape(const std::vector<int64_t>& dims)
{
    gert::StorageShape shape;
    for (const int64_t dim : dims) {
        shape.MutableOriginShape().AppendDim(dim);
        shape.MutableStorageShape().AppendDim(dim);
    }
    return shape;
}

// Output shape rule: NCHW -> (N, G, 1, 1, 1); NHWC -> (N, 1, 1, G, 1).
std::vector<int64_t> MakeOutShape(const std::vector<int64_t>& xShape, ge::Format format, int64_t numGroups)
{
    const int64_t n = xShape.empty() ? 1 : xShape[0];
    if (format == ge::FORMAT_NHWC) {
        return {n, 1, 1, numGroups, 1};
    }
    return {n, numGroups, 1, 1, 1};
}

TilingResult RunTiling(const std::vector<int64_t>& xShape, ge::DataType dtype, ge::Format format, int64_t numGroups)
{
    std::map<std::string, std::string> socInfos;
    std::map<std::string, std::string> aicoreSpec;
    std::map<std::string, std::string> intrinsics;
    GetPlatFormInfos(kCompileInfo, socInfos, aicoreSpec, intrinsics);
    socInfos["vector_core_cnt"] = std::to_string(kCoreNum);
    socInfos["cube_core_cnt"] = std::to_string(kCoreNum / 2);

    fe::PlatFormInfos platformInfo;
    platformInfo.Init();

    optiling::GNTrainingReduceCompileInfo compileInfo{kCoreNum, kUbSize};

    gert::StorageShape inShape = MakeStorageShape(xShape);
    const std::vector<int64_t> outDims = MakeOutShape(xShape, format, numGroups);
    gert::StorageShape sumShape = MakeStorageShape(outDims);
    gert::StorageShape squareSumShape = MakeStorageShape(outDims);

    auto tilingData = gert::TilingData::CreateCap(kTilingDataCap);
    auto workspaceHolder = gert::ContinuousVector::Create<size_t>(kTilingDataCap);
    auto workspace = reinterpret_cast<gert::ContinuousVector*>(workspaceHolder.get());

    auto holder = gert::TilingContextFaker()
                      .SetOpType(kOpType)
                      .NodeIoNum(1, 2)
                      .IrInstanceNum({1}, {1, 1})
                      .InputShapes({&inShape})
                      .OutputShapes({&sumShape, &squareSumShape})
                      .CompileInfo(&compileInfo)
                      .PlatformInfo(reinterpret_cast<char*>(&platformInfo))
                      .NodeInputTd(0, dtype, format, format)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs({{"num_groups", Ops::NN::AnyValue::CreateFrom<int64_t>(numGroups)}})
                      .TilingData(tilingData.get())
                      .Workspace(workspace)
                      .Build();

    TilingResult result;
    gert::TilingContext* context = holder.GetContext<gert::TilingContext>();
    if (context == nullptr) {
        return result;
    }
    context->GetPlatformInfo()->SetPlatformRes("SoCInfo", socInfos);
    context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicoreSpec);
    context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    result.ok = (optiling::TilingForGNTrainingReduce(context) == ge::GRAPH_SUCCESS);
    if (!result.ok) {
        return result;
    }
    result.tilingKey = context->GetTilingKey();
    result.blockDim = context->GetBlockDim();
    auto* rawTilingData = context->GetRawTilingData();
    if (rawTilingData == nullptr || rawTilingData->GetData() == nullptr) {
        result.ok = false;
        return result;
    }
    const auto* rawData = static_cast<const uint8_t*>(rawTilingData->GetData());
    result.tilingData.assign(rawData, rawData + rawTilingData->GetDataSize());
    return result;
}

const GNTrainingReduceTilingData* TilingDataOf(const TilingResult& result)
{
    return reinterpret_cast<const GNTrainingReduceTilingData*>(result.tilingData.data());
}

const GNTrainingReduceEmptyTilingData* EmptyTilingDataOf(const TilingResult& result)
{
    return reinterpret_cast<const GNTrainingReduceEmptyTilingData*>(result.tilingData.data());
}

// base 模板的公共结构不变量：pre/post buffer 非空且 32B 对齐、至少一个 R 迭代、不写 group 字段。
void ExpectBaseTiling(const TilingResult& r)
{
    ASSERT_EQ(r.tilingData.size(), sizeof(GNTrainingReduceTilingData));
    const auto* td = TilingDataOf(r);
    EXPECT_GE(td->axisNum, 1);
    EXPECT_LE(td->axisNum, GN_TRAINING_REDUCE_MAX_PATTERN_RANK);
    EXPECT_GE(td->aUbFactor, 1);
    EXPECT_GE(td->rUbFactor, 1);
    EXPECT_GE(td->rLoopCntTotal, 1);
    EXPECT_GT(td->preBufSize, 0);
    EXPECT_EQ(td->preBufSize % 32, 0);
    EXPECT_GT(td->postBufSize, 0);
    EXPECT_EQ(td->postBufSize % 32, 0);
    EXPECT_EQ(td->rGroupCnt, 0); // group 字段仅 group 模板写入
}

// group 模板的公共结构不变量。
void ExpectGroupTiling(const TilingResult& r)
{
    ASSERT_EQ(r.tilingData.size(), sizeof(GNTrainingReduceTilingData));
    const auto* td = TilingDataOf(r);
    EXPECT_GE(td->rGroupCnt, 1);
    EXPECT_GE(td->rLoopCntTotal, 1);
    EXPECT_GE(td->aUbFactor, 1);
    EXPECT_GT(td->preBufSize, 0);
    EXPECT_EQ(td->preBufSize % 32, 0);
    EXPECT_EQ(td->postBufSize % 32, 0);
}

} // namespace

class GNTrainingReduceTilingTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "GNTrainingReduceTilingTest SetUp." << std::endl; }

    static void TearDownTestCase() { std::cout << "GNTrainingReduceTilingTest TearDown." << std::endl; }
};

// tilingKey 0 (base) | fp32 NCHW, single A chunk, non-aligned tail-R.
TEST_F(GNTrainingReduceTilingTest, BaseFp32Nchw)
{
    const TilingResult r = RunTiling({2, 4, 3, 3}, ge::DT_FLOAT, ge::FORMAT_NCHW, 2);
    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.tilingKey, 0U); // base 模板
    EXPECT_GE(r.blockDim, 1U);
    ExpectBaseTiling(r);
}

// tilingKey 0 (base) | fp16 uses a block of 16 elements -> padded tail-R stride.
TEST_F(GNTrainingReduceTilingTest, BaseFp16Nchw)
{
    const TilingResult r = RunTiling({2, 4, 3, 3}, ge::DT_FLOAT16, ge::FORMAT_NCHW, 2);
    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.tilingKey, 0U); // base 模板
    EXPECT_GE(r.blockDim, 1U);
    ExpectBaseTiling(r);
}

// tilingKey 0 (base) | NHWC view fuses to ARAR (axisNum = 4, tail-R).
TEST_F(GNTrainingReduceTilingTest, BaseNhwcFp16)
{
    const TilingResult r = RunTiling({2, 3, 3, 4}, ge::DT_FLOAT16, ge::FORMAT_NHWC, 2);
    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.tilingKey, 0U); // base 模板
    EXPECT_GE(r.blockDim, 1U);
    ExpectBaseTiling(r);
}

// tilingKey 0 (base) | D*H*W == 1 -> PadRIfPureA -> tail-A (axisNum = 3, ARA).
TEST_F(GNTrainingReduceTilingTest, BaseM1Degenerate)
{
    const TilingResult r = RunTiling({1, 2, 1, 1}, ge::DT_FLOAT, ge::FORMAT_NCHW, 2);
    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.tilingKey, 0U); // base 模板
    ExpectBaseTiling(r);
}

// tilingKey 0 (base) | multi-loop R with all AIV cores used.
TEST_F(GNTrainingReduceTilingTest, BaseMultiLoopR)
{
    const TilingResult r = RunTiling({64, 2, 200, 200}, ge::DT_FLOAT, ge::FORMAT_NCHW, 1);
    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.tilingKey, 0U); // base 模板
    EXPECT_GT(r.blockDim, 1U);
    ExpectBaseTiling(r);
    EXPECT_GT(TilingDataOf(r)->rLoopCntTotal, 1); // 多段 R
}

// tilingKey 0 (base) | non-aligned tail-R: rUbFactor < rUbFactorAlign.
TEST_F(GNTrainingReduceTilingTest, BaseAlignTailR)
{
    const TilingResult r = RunTiling({3, 6, 13, 17}, ge::DT_FLOAT, ge::FORMAT_NCHW, 3);
    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.tilingKey, 0U); // base 模板
    EXPECT_GE(r.blockDim, 1U);
    ExpectBaseTiling(r);
    EXPECT_LE(TilingDataOf(r)->rUbFactor, TilingDataOf(r)->rUbFactorAlign);
}

// tilingKey 0 (base) | large A split into many chunks across all cores.
TEST_F(GNTrainingReduceTilingTest, BaseLargeAFp32)
{
    const TilingResult r = RunTiling({16, 320, 32, 32}, ge::DT_FLOAT, ge::FORMAT_NCHW, 32);
    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.tilingKey, 0U); // base 模板
    EXPECT_GT(r.blockDim, 1U);
    ExpectBaseTiling(r);
}

// tilingKey 1 (group) | small A, large R -> 走 group 模板（A x R 2D 网格）。
TEST_F(GNTrainingReduceTilingTest, GroupFp32)
{
    const TilingResult r = RunTiling({1, 2, 200, 200}, ge::DT_FLOAT, ge::FORMAT_NCHW, 2);
    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.tilingKey, 1U); // group 模板
    EXPECT_GE(r.blockDim, 1U);
    ExpectGroupTiling(r);
}

// tilingKey 1 (group) | NHWC tail-A view -> non-burst ComputeRUbFactor path.
TEST_F(GNTrainingReduceTilingTest, GroupNhwcFp32)
{
    const TilingResult r = RunTiling({1, 200, 200, 2}, ge::DT_FLOAT, ge::FORMAT_NHWC, 2);
    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.tilingKey, 1U); // group 模板
    EXPECT_GE(r.blockDim, 1U);
    ExpectGroupTiling(r);
}

// tilingKey 2 (empty) | EMPTY_A: N == 0 -> blockDim 1, kernel early-exits.
TEST_F(GNTrainingReduceTilingTest, EmptyAFp32)
{
    const TilingResult r = RunTiling({0, 2, 1, 1}, ge::DT_FLOAT, ge::FORMAT_NCHW, 2);
    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.tilingKey, 2U); // empty 模板
    EXPECT_EQ(r.blockDim, 1U);
    ASSERT_EQ(r.tilingData.size(), sizeof(GNTrainingReduceEmptyTilingData));
    const auto* td = EmptyTilingDataOf(r);
    EXPECT_EQ(td->usedCoreNum, 0); // EMPTY_A：所有核早退，禁用 SetBlockDim(0)
    EXPECT_EQ(td->aTotal, 0);
}

// tilingKey 2 (empty) | EMPTY_A via NHWC N == 0.
TEST_F(GNTrainingReduceTilingTest, EmptyANhwc)
{
    const TilingResult r = RunTiling({0, 3, 3, 4}, ge::DT_FLOAT, ge::FORMAT_NHWC, 2);
    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.tilingKey, 2U); // empty 模板
    EXPECT_EQ(r.blockDim, 1U);
    ASSERT_EQ(r.tilingData.size(), sizeof(GNTrainingReduceEmptyTilingData));
    const auto* td = EmptyTilingDataOf(r);
    EXPECT_EQ(td->usedCoreNum, 0);
    EXPECT_EQ(td->aTotal, 0);
}

// tilingKey 2 (empty) | EMPTY_R: H == 0 -> A-only split, single core.
TEST_F(GNTrainingReduceTilingTest, EmptyRH0)
{
    const TilingResult r = RunTiling({2, 4, 0, 7}, ge::DT_FLOAT, ge::FORMAT_NCHW, 2);
    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.tilingKey, 2U); // empty 模板
    EXPECT_EQ(r.blockDim, 1U);
    ASSERT_EQ(r.tilingData.size(), sizeof(GNTrainingReduceEmptyTilingData));
    const auto* td = EmptyTilingDataOf(r);
    EXPECT_GE(td->usedCoreNum, 1);
    EXPECT_GT(td->aTotal, 0);
    EXPECT_GE(td->aUbFactor, 1);
    EXPECT_EQ(td->postBufSize % 32, 0);
}

// tilingKey 2 (empty) | EMPTY_R with a large A total -> multi-core split.
TEST_F(GNTrainingReduceTilingTest, EmptyRMultiCore)
{
    const TilingResult r = RunTiling({100000, 0, 1, 1}, ge::DT_FLOAT, ge::FORMAT_NCHW, 1);
    ASSERT_TRUE(r.ok);
    EXPECT_EQ(r.tilingKey, 2U); // empty 模板
    EXPECT_GT(r.blockDim, 1U);
    ASSERT_EQ(r.tilingData.size(), sizeof(GNTrainingReduceEmptyTilingData));
    const auto* td = EmptyTilingDataOf(r);
    EXPECT_GT(td->usedCoreNum, 1);
    EXPECT_GT(td->aTotal, 0);
    EXPECT_EQ(td->postBufSize % 32, 0);
}

// Rejections: rank(x) != 4.
TEST_F(GNTrainingReduceTilingTest, RejectRankThree)
{
    EXPECT_FALSE(RunTiling({2, 4, 3}, ge::DT_FLOAT, ge::FORMAT_NCHW, 2).ok);
}

TEST_F(GNTrainingReduceTilingTest, RejectRankFive)
{
    EXPECT_FALSE(RunTiling({1, 2, 3, 4, 5}, ge::DT_FLOAT, ge::FORMAT_NCHW, 2).ok);
}

// Rejections: dtype not in {float16, float32}.
TEST_F(GNTrainingReduceTilingTest, RejectDtypeInt32)
{
    EXPECT_FALSE(RunTiling({2, 4, 3, 3}, ge::DT_INT32, ge::FORMAT_NCHW, 2).ok);
}

// Rejections: input format other than NCHW / NHWC.
TEST_F(GNTrainingReduceTilingTest, RejectFormatNcdhw)
{
    EXPECT_FALSE(RunTiling({2, 3, 4, 5}, ge::DT_FLOAT, ge::FORMAT_NCDHW, 2).ok);
}

// Rejections: input format ND is NOT accepted either (same contract as InferShape;
// design/Interface.md「数据 Format 支持」: x only NCHW / NHWC).
TEST_F(GNTrainingReduceTilingTest, RejectFormatNd)
{
    EXPECT_FALSE(RunTiling({2, 4, 3, 3}, ge::DT_FLOAT, ge::FORMAT_ND, 2).ok);
}

// Rejections: num_groups must divide C.
TEST_F(GNTrainingReduceTilingTest, RejectNumGroupsNotDivideC)
{
    EXPECT_FALSE(RunTiling({2, 4, 3, 3}, ge::DT_FLOAT, ge::FORMAT_NCHW, 3).ok);
}

// Rejections: N == 0 does NOT exempt the C % num_groups check; an empty tensor
// still requires num_groups | C (same contract as InferShape and the A2/A3 TBE
// reference _shape_check). NCHW C=3, NHWC C=3 with num_groups=2.
TEST_F(GNTrainingReduceTilingTest, RejectEmptyNonDivisibleCG)
{
    EXPECT_FALSE(RunTiling({0, 3, 2, 2}, ge::DT_FLOAT, ge::FORMAT_NCHW, 2).ok);
    EXPECT_FALSE(RunTiling({0, 2, 2, 3}, ge::DT_FLOAT, ge::FORMAT_NHWC, 2).ok);
}

// Rejections: num_groups must be >= 1.
TEST_F(GNTrainingReduceTilingTest, RejectNumGroupsZero)
{
    EXPECT_FALSE(RunTiling({2, 4, 3, 3}, ge::DT_FLOAT, ge::FORMAT_NCHW, 0).ok);
}

} // namespace GNTrainingReduceTilingUT
