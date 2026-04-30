/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_log_softmax_grad_apt.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <sstream>
#include <string>
#include "gtest/gtest.h"
#include "log_softmax_grad_tiling_def.h"

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

extern "C" __global__ __aicore__ void log_softmax_grad(
    GM_ADDR grad, GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

class log_softmax_grad_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "log_softmax_grad_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "log_softmax_grad_test TearDown\n" << endl;
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
    const std::vector<int64_t>& shape, const std::string& dtype, int64_t tilingKey, uint32_t blockNum, uint8_t* tiling)
{
    uint32_t typeSize = 4;
    if (dtype != "float") {
        typeSize = 2;
    }

    uint32_t realBlockNum = blockNum;
    size_t fileSize = GetShapeSize(shape) * typeSize;
    size_t workspaceFileSize = 16 * 1024 * 1024;

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc((fileSize + 31) / 32 * 32);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc((fileSize + 31) / 32 * 32);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc((fileSize + 31) / 32 * 32);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceFileSize);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(log_softmax_grad, realBlockNum, grad, x, y, workspace, tiling);

    AscendC::GmFree(grad);
    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(log_softmax_grad_test, test_ar_small_r_float32)
{
    std::vector<int64_t> shape = {32, 8};
    std::string dtype = "float";
    uint64_t tilingKey = 500;
    uint32_t blockNum = 1;
    size_t tilingSize = sizeof(SoftmaxGradARSmallRTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    SoftmaxGradARSmallRTilingData* tilingDatafromBin = reinterpret_cast<SoftmaxGradARSmallRTilingData*>(tiling);

    tilingDatafromBin->totalA0Len = 32;
    tilingDatafromBin->totalRLen = 8;
    tilingDatafromBin->tileA0Outer = 4;
    tilingDatafromBin->totalTiles = 1;
    tilingDatafromBin->tilesPerCore = 1;
    tilingDatafromBin->tileA0Len = 32;
    tilingDatafromBin->tileA0Tail = 32;
    tilingDatafromBin->rTileBase = 8;
    tilingDatafromBin->rAligned = 8;

    ExcuteTestCase(shape, dtype, tilingKey, blockNum, (uint8_t*)tilingDatafromBin);
}

TEST_F(log_softmax_grad_test, test_ar_float32)
{
    std::vector<int64_t> shape = {128, 1024};
    std::string dtype = "float";
    uint64_t tilingKey = 1000;
    uint32_t blockNum = 2;
    size_t tilingSize = sizeof(SoftmaxGradARTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    SoftmaxGradARTilingData* tilingDatafromBin = reinterpret_cast<SoftmaxGradARTilingData*>(tiling);

    tilingDatafromBin->a = 128;
    tilingDatafromBin->r = 1024;
    tilingDatafromBin->rAligned = 1024;
    tilingDatafromBin->ubFactor = 64;
    tilingDatafromBin->aBlockFactor = 64;
    tilingDatafromBin->rLoopCount = 1;

    ExcuteTestCase(shape, dtype, tilingKey, blockNum, (uint8_t*)tilingDatafromBin);
}

TEST_F(log_softmax_grad_test, test_ar_recompute_float32)
{
    std::vector<int64_t> shape = {64, 2048};
    std::string dtype = "float";
    uint64_t tilingKey = 2000;
    uint32_t blockNum = 2;
    size_t tilingSize = sizeof(SoftmaxGradARRecomputeTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    SoftmaxGradARRecomputeTilingData* tilingDatafromBin = reinterpret_cast<SoftmaxGradARRecomputeTilingData*>(tiling);

    tilingDatafromBin->a = 64;
    tilingDatafromBin->r = 2048;
    tilingDatafromBin->ubFactor = 1024;
    tilingDatafromBin->ubFactorTail = 0;
    tilingDatafromBin->aBlockFactor = 32;
    tilingDatafromBin->aLoopCountCeil = 2;
    tilingDatafromBin->basicBlockLoop = 32;
    tilingDatafromBin->mainFoldCount = 0;

    ExcuteTestCase(shape, dtype, tilingKey, blockNum, (uint8_t*)tilingDatafromBin);
}

TEST_F(log_softmax_grad_test, test_ara_float32)
{
    std::vector<int64_t> shape = {4, 8, 128, 256};
    std::string dtype = "float";
    uint64_t tilingKey = 10000;
    uint32_t blockNum = 2;
    size_t tilingSize = sizeof(SoftmaxGradARATilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    SoftmaxGradARATilingData* tilingDatafromBin = reinterpret_cast<SoftmaxGradARATilingData*>(tiling);

    tilingDatafromBin->totalRLen = 256;
    tilingDatafromBin->totalA0Len = 8;
    tilingDatafromBin->totalTiles = 2;
    tilingDatafromBin->tilesPerCore = 1;
    tilingDatafromBin->tileA0Outer = 4;
    tilingDatafromBin->tileA0Len = 4;
    tilingDatafromBin->tileA0Tail = 4;
    tilingDatafromBin->a1Outer = 4;
    tilingDatafromBin->a1Inner = 2;
    tilingDatafromBin->a1Tail = 2;

    ExcuteTestCase(shape, dtype, tilingKey, blockNum, (uint8_t*)tilingDatafromBin);
}

TEST_F(log_softmax_grad_test, test_ara_recompute_float32)
{
    std::vector<int64_t> shape = {4, 16, 1024};
    std::string dtype = "float";
    uint64_t tilingKey = 20000;
    uint32_t blockNum = 2;
    size_t tilingSize = sizeof(SoftmaxGradARARecomputeTilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    SoftmaxGradARARecomputeTilingData* tilingDatafromBin = reinterpret_cast<SoftmaxGradARARecomputeTilingData*>(tiling);

    tilingDatafromBin->totalRLen = 1024;
    tilingDatafromBin->totalA0Len = 16;
    tilingDatafromBin->totalTiles = 2;
    tilingDatafromBin->tilesPerCore = 1;
    tilingDatafromBin->usedCoreNums = 2;
    tilingDatafromBin->tileA0Outer = 8;
    tilingDatafromBin->tileA0Len = 8;
    tilingDatafromBin->tileA0Tail = 8;
    tilingDatafromBin->binAddRFactor = 1024;
    tilingDatafromBin->binAddRLoop = 1;
    tilingDatafromBin->binAddRTotalLoop = 1;
    tilingDatafromBin->binAddRTail = 0;
    tilingDatafromBin->binAddBasicBlockLoop = 32;
    tilingDatafromBin->binAddMainFoldCount = 0;
    tilingDatafromBin->binAddCacheBufferCount = 1;
    tilingDatafromBin->binAddResultCacheID = 0;

    ExcuteTestCase(shape, dtype, tilingKey, blockNum, (uint8_t*)tilingDatafromBin);
}