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
 * \file test_sync_batch_norm_gather_stats_apt.cpp
 * \brief
 */

#include <array>
#include <cmath>
#include <vector>
#include <sstream>
#include <string>
#include "gtest/gtest.h"
#include "sync_batch_norm_gather_stats_tiling_def.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#include "register/op_def_registry.h"
#include "string.h"
#include <iostream>
#include <string>
#endif

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void sync_batch_norm_gather_stats(GM_ADDR total_sum, GM_ADDR total_square_sum,
                                                                   GM_ADDR sample_count, GM_ADDR running_mean,
                                                                   GM_ADDR running_var, GM_ADDR batch_mean,
                                                                   GM_ADDR batch_invstd, GM_ADDR running_mean_update,
                                                                   GM_ADDR running_var_update, GM_ADDR workspace,
                                                                   GM_ADDR tiling);

class sync_batch_norm_gather_stats_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "sync_batch_norm_gather_stats_test SetUp\n" << endl; }
    static void TearDownTestCase() { cout << "sync_batch_norm_gather_stats_test TearDown\n" << endl; }
};

std::string Shape2Str(const std::vector<int64_t>& shape)
{
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        oss << shape[i];
        if (i != shape.size() - 1) {
            oss << ",";
        }
    }
    oss << "]";
    return oss.str();
}

static inline int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

void ExcuteTestCase(const std::vector<int64_t>& sumShape, const std::vector<int64_t>& countShape,
                    const std::string& dtype, int64_t tilingKey, uint32_t blockNum, uint8_t* tiling)
{
    uint32_t typeSize = 4;
    uint32_t fp32TypeSize = 4;
    uint32_t int32TypeSize = 4;
    if (dtype != "float") {
        typeSize = 2;
    }

    const int64_t cLen = sumShape.back();
    size_t sumFileSize = GetShapeSize(sumShape) * typeSize;
    size_t countFileSize = GetShapeSize(countShape) * int32TypeSize;
    size_t meanFileSize = cLen * fp32TypeSize;

    size_t workspaceFileSize = 16 * 1024 * 1024;

    uint8_t* total_sum = (uint8_t*)AscendC::GmAlloc((sumFileSize + 31) / 32 * 32);
    uint8_t* total_square_sum = (uint8_t*)AscendC::GmAlloc((sumFileSize + 31) / 32 * 32);
    uint8_t* sample_count = (uint8_t*)AscendC::GmAlloc((countFileSize + 31) / 32 * 32);
    uint8_t* running_mean = (uint8_t*)AscendC::GmAlloc((meanFileSize + 31) / 32 * 32);
    uint8_t* running_var = (uint8_t*)AscendC::GmAlloc((meanFileSize + 31) / 32 * 32);
    uint8_t* batch_mean = (uint8_t*)AscendC::GmAlloc((meanFileSize + 31) / 32 * 32);
    uint8_t* batch_invstd = (uint8_t*)AscendC::GmAlloc((meanFileSize + 31) / 32 * 32);
    uint8_t* running_mean_update = (uint8_t*)AscendC::GmAlloc((meanFileSize + 31) / 32 * 32);
    uint8_t* running_var_update = (uint8_t*)AscendC::GmAlloc((meanFileSize + 31) / 32 * 32);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceFileSize);

    int32_t* count = reinterpret_cast<int32_t*>(sample_count);
    float* mean = reinterpret_cast<float*>(running_mean);
    float* var = reinterpret_cast<float*>(running_var);
    for (int64_t i = 0; i < GetShapeSize(countShape); ++i) {
        count[i] = 1;
    }
    for (int64_t i = 0; i < cLen; ++i) {
        mean[i] = 0.0f;
        var[i] = 1.0f;
    }

#if (__CCE_AICORE__ == 200)
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
#else
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
#endif
    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(sync_batch_norm_gather_stats, blockNum, total_sum, total_square_sum, sample_count, running_mean,
                running_var, batch_mean, batch_invstd, running_mean_update, running_var_update, workspace, tiling);

    AscendC::GmFree(total_sum);
    AscendC::GmFree(total_square_sum);
    AscendC::GmFree(sample_count);
    AscendC::GmFree(running_mean);
    AscendC::GmFree(running_var);
    AscendC::GmFree(batch_mean);
    AscendC::GmFree(batch_invstd);
    AscendC::GmFree(running_mean_update);
    AscendC::GmFree(running_var_update);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(sync_batch_norm_gather_stats_test, test_full_load_float32_value)
{
    constexpr int64_t N = 2;
    constexpr int64_t C = 16;
    size_t tilingSize = sizeof(SyncBatchNormGatherStatsTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    auto* td = reinterpret_cast<SyncBatchNormGatherStatsTilingData*>(tiling);
    td->blockDim = 1;
    td->blockFormer = 1;
    td->blockTail = 1;
    td->nLen = N;
    td->cLen = C;
    td->ubFormer = C;
    td->ubTail = C;
    const uint32_t blockNum = 2;
    td->momentum = 0.5f;
    td->eps = 0.001f;

    auto alloc = [](size_t bytes) { return (uint8_t*)AscendC::GmAlloc((bytes + 31) / 32 * 32); };
    uint8_t* total_sum = alloc(N * C * 4);
    uint8_t* total_square_sum = alloc(N * C * 4);
    uint8_t* sample_count = alloc(N * 4);
    uint8_t* running_mean = alloc(C * 4);
    uint8_t* running_var = alloc(C * 4);
    uint8_t* batch_mean = alloc(C * 4);
    uint8_t* batch_invstd = alloc(C * 4);
    uint8_t* running_mean_update = alloc(C * 4);
    uint8_t* running_var_update = alloc(C * 4);
    uint8_t* workspace = alloc(16 * 1024 * 1024);

    float* ts = (float*)total_sum;
    float* sq = (float*)total_square_sum;
    int32_t* cnt = (int32_t*)sample_count;
    float* rm = (float*)running_mean;
    float* rv = (float*)running_var;
    cnt[0] = 4;
    cnt[1] = 4;
    for (int64_t c = 0; c < C; ++c) {
        const float bmv = 3.0f * (c + 1) / 8.0f;
        ts[c] = c + 1.0f;
        ts[C + c] = 2.0f * (c + 1);
        sq[c] = 4.0f * (bmv * bmv + 1.0f);
        sq[C + c] = 4.0f * (bmv * bmv + 1.0f);
        rm[c] = 1.0f;
        rv[c] = 2.0f;
    }

#if (__CCE_AICORE__ == 200)
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
#else
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
#endif
    ICPU_SET_TILING_KEY(10001);
    ICPU_RUN_KF(sync_batch_norm_gather_stats, blockNum, total_sum, total_square_sum, sample_count, running_mean,
                running_var, batch_mean, batch_invstd, running_mean_update, running_var_update, workspace, tiling);

    float* bm = (float*)batch_mean;
    float* bi = (float*)batch_invstd;
    float* rmu = (float*)running_mean_update;
    float* rvu = (float*)running_var_update;
    for (int64_t c = 0; c < C; ++c) {
        const float expBm = 3.0f * (c + 1) / 8.0f;
        const float expBi = 1.0f / std::sqrt(1.0f + 0.001f);
        const float expRm = 0.5f + 0.5f * expBm;
        const float expRv = 1.0f + 0.5f * 8.0f / 7.0f;
        EXPECT_NEAR(bm[c], expBm, 1e-3 * (1 + std::fabs(expBm)));
        EXPECT_NEAR(bi[c], expBi, 1e-3);
        EXPECT_NEAR(rmu[c], expRm, 1e-3 * (1 + std::fabs(expRm)));
        EXPECT_NEAR(rvu[c], expRv, 1e-3 * (1 + std::fabs(expRv)));
    }

    AscendC::GmFree(total_sum);
    AscendC::GmFree(total_square_sum);
    AscendC::GmFree(sample_count);
    AscendC::GmFree(running_mean);
    AscendC::GmFree(running_var);
    AscendC::GmFree(batch_mean);
    AscendC::GmFree(batch_invstd);
    AscendC::GmFree(running_mean_update);
    AscendC::GmFree(running_var_update);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// countAll == 1：无偏方差的分母 (countAll - 1) 为 0。样本方差本身也恒为 0，
// 若不退化为有偏估计，running_var_update 会拿到 Inf * 0 = NaN 并污染后续推理。
TEST_F(sync_batch_norm_gather_stats_test, test_full_load_float32_single_sample_no_nan)
{
    constexpr int64_t N = 1;
    constexpr int64_t C = 16;
    size_t tilingSize = sizeof(SyncBatchNormGatherStatsTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    auto* td = reinterpret_cast<SyncBatchNormGatherStatsTilingData*>(tiling);
    td->blockDim = 1;
    td->blockFormer = 1;
    td->blockTail = 1;
    td->nLen = N;
    td->cLen = C;
    td->ubFormer = C;
    td->ubTail = C;
    const uint32_t blockNum = 1;
    td->momentum = 0.5f;
    td->eps = 0.001f;

    auto alloc = [](size_t bytes) { return (uint8_t*)AscendC::GmAlloc((bytes + 31) / 32 * 32); };
    uint8_t* total_sum = alloc(N * C * 4);
    uint8_t* total_square_sum = alloc(N * C * 4);
    uint8_t* sample_count = alloc(N * 4);
    uint8_t* running_mean = alloc(C * 4);
    uint8_t* running_var = alloc(C * 4);
    uint8_t* batch_mean = alloc(C * 4);
    uint8_t* batch_invstd = alloc(C * 4);
    uint8_t* running_mean_update = alloc(C * 4);
    uint8_t* running_var_update = alloc(C * 4);
    uint8_t* workspace = alloc(16 * 1024 * 1024);

    float* ts = (float*)total_sum;
    float* sq = (float*)total_square_sum;
    int32_t* cnt = (int32_t*)sample_count;
    float* rm = (float*)running_mean;
    float* rv = (float*)running_var;
    cnt[0] = 1;
    for (int64_t c = 0; c < C; ++c) {
        // 单样本：sum = x，squareSum = x^2，因此 batchVar = x^2 - x^2 = 0。
        const float x = c + 1.0f;
        ts[c] = x;
        sq[c] = x * x;
        rm[c] = 1.0f;
        rv[c] = 2.0f;
    }

#if (__CCE_AICORE__ == 200)
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
#else
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
#endif
    ICPU_SET_TILING_KEY(10001);
    ICPU_RUN_KF(sync_batch_norm_gather_stats, blockNum, total_sum, total_square_sum, sample_count, running_mean,
                running_var, batch_mean, batch_invstd, running_mean_update, running_var_update, workspace, tiling);

    float* bm = (float*)batch_mean;
    float* bi = (float*)batch_invstd;
    float* rmu = (float*)running_mean_update;
    float* rvu = (float*)running_var_update;
    for (int64_t c = 0; c < C; ++c) {
        const float expBm = c + 1.0f;
        const float expBi = 1.0f / std::sqrt(0.0f + 0.001f);
        const float expRm = 0.5f + 0.5f * expBm;
        // 退化为有偏估计后 momentum * batchVar = 0.5 * 0 = 0，只剩衰减掉的旧值。
        const float expRv = 2.0f * 0.5f;
        EXPECT_FALSE(std::isnan(rvu[c]));
        EXPECT_NEAR(bm[c], expBm, 1e-3 * (1 + std::fabs(expBm)));
        EXPECT_NEAR(bi[c], expBi, 1e-3 * (1 + std::fabs(expBi)));
        EXPECT_NEAR(rmu[c], expRm, 1e-3 * (1 + std::fabs(expRm)));
        EXPECT_NEAR(rvu[c], expRv, 1e-3 * (1 + std::fabs(expRv)));
    }

    AscendC::GmFree(total_sum);
    AscendC::GmFree(total_square_sum);
    AscendC::GmFree(sample_count);
    AscendC::GmFree(running_mean);
    AscendC::GmFree(running_var);
    AscendC::GmFree(batch_mean);
    AscendC::GmFree(batch_invstd);
    AscendC::GmFree(running_mean_update);
    AscendC::GmFree(running_var_update);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(sync_batch_norm_gather_stats_test, test_full_load_float32)
{
    std::vector<int64_t> sumShape = {2, 64};
    std::vector<int64_t> countShape = {2};
    std::string dtype = "float";
    uint64_t tilingKey = 10001;
    uint32_t blockNum = 2;
    size_t tilingSize = sizeof(SyncBatchNormGatherStatsTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    SyncBatchNormGatherStatsTilingData* tilingDatafromBin = reinterpret_cast<SyncBatchNormGatherStatsTilingData*>(
        tiling);

    tilingDatafromBin->blockDim = 2;
    tilingDatafromBin->blockFormer = 1;
    tilingDatafromBin->blockTail = 1;
    tilingDatafromBin->nLen = 2;
    tilingDatafromBin->cLen = 64;
    tilingDatafromBin->ubFormer = 32;
    tilingDatafromBin->ubTail = 32;
    tilingDatafromBin->momentum = 0.1;
    tilingDatafromBin->eps = 1e-5;
    ExcuteTestCase(sumShape, countShape, dtype, tilingKey, blockNum, (uint8_t*)tilingDatafromBin);
}

TEST_F(sync_batch_norm_gather_stats_test, test_not_full_load_float32)
{
    std::vector<int64_t> sumShape = {2, 128};
    std::vector<int64_t> countShape = {2};
    std::string dtype = "float";
    uint64_t tilingKey = 20001;
    uint32_t blockNum = 2;
    size_t tilingSize = sizeof(SyncBatchNormGatherStatsNNotFullLoadTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    SyncBatchNormGatherStatsNNotFullLoadTilingData*
        tilingDatafromBin = reinterpret_cast<SyncBatchNormGatherStatsNNotFullLoadTilingData*>(tiling);

    tilingDatafromBin->blockDim = 2;
    tilingDatafromBin->cLen = 128;
    tilingDatafromBin->cFactor = 64;
    tilingDatafromBin->cLoopMainBlock = 1;
    tilingDatafromBin->cTileMainBlock = 64;
    tilingDatafromBin->cLoopTailBlock = 1;
    tilingDatafromBin->cTailTailBlock = 64;
    tilingDatafromBin->nFactor = 2;
    tilingDatafromBin->nLoop = 1;
    tilingDatafromBin->nMainFoldCount = 0;
    tilingDatafromBin->nTail = 0;
    tilingDatafromBin->cacheBufferCount = 8;
    tilingDatafromBin->resultCacheId = 0;
    tilingDatafromBin->momentum = 0.1;
    tilingDatafromBin->eps = 1e-5;
    ExcuteTestCase(sumShape, countShape, dtype, tilingKey, blockNum, (uint8_t*)tilingDatafromBin);
}
