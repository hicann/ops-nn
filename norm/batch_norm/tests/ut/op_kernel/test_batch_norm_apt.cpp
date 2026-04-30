/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_batch_norm.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <sstream>
#include <string>
#include "gtest/gtest.h"
#include "batch_norm_tiling_def.h"

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

extern "C" __global__ __aicore__ void batch_norm(
    GM_ADDR x, GM_ADDR scale, GM_ADDR offset, GM_ADDR mean, GM_ADDR variance, GM_ADDR y, GM_ADDR batch_mean,
    GM_ADDR batch_variance, GM_ADDR reserve_space_1, GM_ADDR reserve_space_2, GM_ADDR reserve_space_3,
    GM_ADDR workspace, GM_ADDR tiling);

class batch_norm_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "batch_norm_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "batch_norm_test TearDown\n" << endl;
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
    const std::vector<int64_t>& xShape, const std::vector<int64_t>& wShape, const std::string& dtype, int64_t tilingKey,
    uint32_t blockNum, uint8_t* tiling, bool is_training = true)
{
    uint32_t typeSize = 4;
    uint32_t fp32TypeSize = 4;
    if (dtype != "float") {
        typeSize = 2;
    }
    uint32_t realBlockNum = (blockNum + 1) / 2;
    size_t xFileSize = GetShapeSize(xShape) * typeSize;
    size_t weightFileSize = GetShapeSize(wShape) * typeSize;
    size_t meanFileSize = GetShapeSize(wShape) * fp32TypeSize;

    size_t workspaceFileSize = 16 * 1024 * 1024;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc((xFileSize + 31) / 32 * 32);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc((weightFileSize + 31) / 32 * 32);
    uint8_t* offset = (uint8_t*)AscendC::GmAlloc((weightFileSize + 31) / 32 * 32);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc((meanFileSize + 31) / 32 * 32);
    uint8_t* variance = (uint8_t*)AscendC::GmAlloc((meanFileSize + 31) / 32 * 32);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc((xFileSize + 31) / 32 * 32);
    uint8_t* batch_mean = (uint8_t*)AscendC::GmAlloc((meanFileSize + 31) / 32 * 32);
    uint8_t* batch_variance = (uint8_t*)AscendC::GmAlloc((meanFileSize + 31) / 32 * 32);
    uint8_t* reserve_space_1 = (uint8_t*)AscendC::GmAlloc((meanFileSize + 31) / 32 * 32);
    uint8_t* reserve_space_2 = (uint8_t*)AscendC::GmAlloc((meanFileSize + 31) / 32 * 32);
    uint8_t* reserve_space_3 = (uint8_t*)AscendC::GmAlloc((xFileSize + 31) / 32 * 32);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceFileSize);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(
        batch_norm, realBlockNum, x, scale, offset, mean, variance, y, batch_mean, batch_variance, reserve_space_1,
        reserve_space_2, reserve_space_3, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(scale);
    AscendC::GmFree(offset);
    AscendC::GmFree(mean);
    AscendC::GmFree(variance);
    AscendC::GmFree(y);
    AscendC::GmFree(batch_mean);
    AscendC::GmFree(batch_variance);
    AscendC::GmFree(reserve_space_1);
    AscendC::GmFree(reserve_space_2);
    AscendC::GmFree(reserve_space_3);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(batch_norm_test, test_full_reduce_float32)
{
    std::vector<int64_t> xShape = {32, 2, 13, 16};
    std::vector<int64_t> wShape = {2};
    std::string dtype = "float";
    uint64_t tilingKey = 200000;
    uint32_t blockNum = 2;
    size_t tilingSize = sizeof(BatchNormFullReduceRegbaseTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    BatchNormFullReduceRegbaseTilingData* tilingDatafromBin =
        reinterpret_cast<BatchNormFullReduceRegbaseTilingData*>(tiling);

    tilingDatafromBin->r1 = 32;
    tilingDatafromBin->r0 = 208;
    tilingDatafromBin->a = 2;
    tilingDatafromBin->aFactor = 2;
    tilingDatafromBin->aBlockFactor = 1;
    tilingDatafromBin->blockNum = 2;
    tilingDatafromBin->r1r0LoopCount = 1;
    tilingDatafromBin->binaryAddQuotient = 8192;
    tilingDatafromBin->binaryAddK = 1;
    tilingDatafromBin->binaryAddLastNum = 64;
    tilingDatafromBin->powerOfTwoForR = 0;
    tilingDatafromBin->epsilon = 1e-5;
    tilingDatafromBin->momentum = 0.1;
    tilingDatafromBin->useRunningMeanVar = 1;
    ExcuteTestCase(xShape, wShape, dtype, tilingKey, blockNum, (uint8_t*)tilingDatafromBin);
}

TEST_F(batch_norm_test, test_welford_float32)
{
    std::vector<int64_t> xShape = {5, 2, 32, 128};
    std::vector<int64_t> wShape = {2};
    std::string dtype = "float";
    uint64_t tilingKey = 300000;
    uint32_t blockNum = 2;
    size_t tilingSize = sizeof(BatchNormWelfordRegbaseTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    BatchNormWelfordRegbaseTilingData* tilingDatafromBin = reinterpret_cast<BatchNormWelfordRegbaseTilingData*>(tiling);

    tilingDatafromBin->r1 = 5;
    tilingDatafromBin->r0 = 4096;
    tilingDatafromBin->a0 = 2;
    tilingDatafromBin->loopR1outer = 1;
    tilingDatafromBin->r1Factor = 5;
    tilingDatafromBin->loopR0outer = 1;
    tilingDatafromBin->r0Factor = 4096;
    tilingDatafromBin->realCoreNum = 2;
    tilingDatafromBin->numLastCore = 0;
    tilingDatafromBin->aBlockFactor = 1;
    tilingDatafromBin->aGatherLimit = 8;
    tilingDatafromBin->parallelN = 1;
    tilingDatafromBin->processSize = 40960;
    tilingDatafromBin->ubSize = 61440;
    tilingDatafromBin->elemNum = 4096;
    tilingDatafromBin->vlLenFp32 = 64;
    tilingDatafromBin->cutR1OrR0 = 0;
    tilingDatafromBin->binaryAddK = 1;
    tilingDatafromBin->binaryAddLastNum = 64;
    tilingDatafromBin->binaryAddQuotient = 8192;
    tilingDatafromBin->epsilon = 1e-5;
    tilingDatafromBin->momentum = 0.1;
    tilingDatafromBin->useRunningMeanVar = 1;
    ExcuteTestCase(xShape, wShape, dtype, tilingKey, blockNum, (uint8_t*)tilingDatafromBin);
}

TEST_F(batch_norm_test, test_infer_float32)
{
    std::vector<int64_t> xShape = {1, 2, 1, 1};
    std::vector<int64_t> wShape = {2};
    std::string dtype = "float";
    uint64_t tilingKey = 910000;
    uint32_t blockNum = 1;
    size_t tilingSize = sizeof(BatchNormInferTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    BatchNormInferTilingData* tilingDatafromBin = reinterpret_cast<BatchNormInferTilingData*>(tiling);

    tilingDatafromBin->totalTiles = 1;
    tilingDatafromBin->tilesPerCore = 1;
    tilingDatafromBin->usedCoreNums = 1;
    tilingDatafromBin->totalB0Len = 1;
    tilingDatafromBin->totalALen = 2;
    tilingDatafromBin->totalB1Len = 1;
    tilingDatafromBin->b0Outer = 1;
    tilingDatafromBin->aOuter = 1;
    tilingDatafromBin->b1Outer = 1;
    tilingDatafromBin->tileBlockB0Len = 1;
    tilingDatafromBin->tileBlockB0Tail = 1;
    tilingDatafromBin->tileBlockALen = 2;
    tilingDatafromBin->tileBlockATail = 2;
    tilingDatafromBin->tileBlockB1Len = 64;
    tilingDatafromBin->tileBlockB1Tail = 1;
    tilingDatafromBin->tileBlockAPaddingNum = 0;
    tilingDatafromBin->epsilon = 1e-5;
    ExcuteTestCase(xShape, wShape, dtype, tilingKey, blockNum, (uint8_t*)tilingDatafromBin, false);
}