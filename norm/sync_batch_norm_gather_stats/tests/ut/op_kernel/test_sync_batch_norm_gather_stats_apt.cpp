/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_sync_batch_norm_gather_stats_apt.cpp
 * \brief
 */

#include <array>
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

extern "C" __global__ __aicore__ void sync_batch_norm_gather_stats(
    GM_ADDR total_sum, GM_ADDR total_square_sum, GM_ADDR sample_count, GM_ADDR running_mean, GM_ADDR running_var,
    GM_ADDR batch_mean, GM_ADDR batch_invstd, GM_ADDR running_mean_update, GM_ADDR running_var_update,
    GM_ADDR workspace, GM_ADDR tiling);

class sync_batch_norm_gather_stats_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "sync_batch_norm_gather_stats_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "sync_batch_norm_gather_stats_test TearDown\n" << endl;
    }
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

void ExcuteTestCase(
    const std::vector<int64_t>& sumShape, const std::vector<int64_t>& countShape, const std::string& dtype,
    int64_t tilingKey, uint32_t blockNum, uint8_t* tiling)
{
    uint32_t typeSize = 4;
    uint32_t fp32TypeSize = 4;
    uint32_t int64TypeSize = 8;
    if (dtype != "float") {
        typeSize = 2;
    }

    size_t sumFileSize = GetShapeSize(sumShape) * typeSize;
    size_t countFileSize = GetShapeSize(countShape) * int64TypeSize;
    size_t meanFileSize = GetShapeSize(sumShape) * fp32TypeSize;

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

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(
        sync_batch_norm_gather_stats, blockNum, total_sum, total_square_sum, sample_count, running_mean, running_var,
        batch_mean, batch_invstd, running_mean_update, running_var_update, workspace, tiling);

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
    SyncBatchNormGatherStatsTilingData* tilingDatafromBin =
        reinterpret_cast<SyncBatchNormGatherStatsTilingData*>(tiling);

    tilingDatafromBin->blockDim = 2;
    tilingDatafromBin->blockFormer = 32;
    tilingDatafromBin->blockTail = 32;
    tilingDatafromBin->nLen = 4;
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
    SyncBatchNormGatherStatsNNotFullLoadTilingData* tilingDatafromBin =
        reinterpret_cast<SyncBatchNormGatherStatsNNotFullLoadTilingData*>(tiling);

    tilingDatafromBin->blockDim = 2;
    tilingDatafromBin->cLen = 128;
    tilingDatafromBin->cFactor = 64;
    tilingDatafromBin->cLoopMainBlock = 1;
    tilingDatafromBin->cTileMainBlock = 64;
    tilingDatafromBin->cLoopTailBlock = 1;
    tilingDatafromBin->cTailTailBlock = 64;
    tilingDatafromBin->nFactor = 4;
    tilingDatafromBin->nLoop = 1;
    tilingDatafromBin->nMainFoldCount = 4;
    tilingDatafromBin->nTail = 0;
    tilingDatafromBin->cacheBufferCount = 8;
    tilingDatafromBin->resultCacheId = 0;
    tilingDatafromBin->momentum = 0.1;
    tilingDatafromBin->eps = 1e-5;
    ExcuteTestCase(sumShape, countShape, dtype, tilingKey, blockNum, (uint8_t*)tilingDatafromBin);
}