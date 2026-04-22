/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"

#include <cstdint>
#define CCE_AICORE 220
#include "rotate_quant.cpp"

using namespace std;

// Test dimensions
constexpr int32_t TEST_M = 4;
constexpr int32_t TEST_N = 64;
constexpr int32_t TEST_K = 64;
constexpr int32_t TEST_NUM_BLOCKS = 1;
constexpr int32_t TEST_AIC_CORE_NUM = 1;
constexpr int32_t TEST_AIV_CORE_NUM = 2;
constexpr int32_t KERNEL_RUN_CORE_NUM = 2;

// Tiling keys for different data types
constexpr uint64_t TILING_KEY_BF16 = 0UL;
constexpr uint64_t TILING_KEY_FP16 = 1UL;

// RotateQuant tiling defaults
constexpr int32_t TEST_INT_8 = 2;
constexpr int32_t DEFAULT_LOOP_M = 1;
constexpr int32_t DEFAULT_STEP_LOOP = 1;
constexpr int32_t DEFAULT_HEAD_CORE_NUM = 1;
constexpr uint32_t DEFAULT_UB_SIZE = 196608;

// Matmul tiling defaults
constexpr int32_t MATMUL_USED_CORE_NUM = 1;
constexpr int32_t MATMUL_BASE_M = 128;
constexpr int32_t MATMUL_BASE_N = 256;
constexpr int32_t MATMUL_BASE_K = 64;
constexpr int32_t MATMUL_DEPTH_A1 = 8;
constexpr int32_t MATMUL_DEPTH_B1 = 4;
constexpr int32_t MATMUL_STEP_KA = 4;
constexpr int32_t MATMUL_STEP_KB = 2;
constexpr int32_t MATMUL_STEP_M = 1;
constexpr int32_t MATMUL_STEP_N = 1;
constexpr int32_t MATMUL_ITERATE_ORDER = 1;
constexpr int32_t MATMUL_SHARE_MODE = 0;
constexpr int32_t MATMUL_IS_BIAS = 0;
constexpr int32_t MATMUL_TRANS_LENGTH = 0;
constexpr int32_t MATMUL_DB_L0A = 2;
constexpr int32_t MATMUL_DB_L0B = 2;
constexpr int32_t MATMUL_DB_L0C = 1;

class rotate_quant_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "rotate_quant_test SetUp" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "rotate_quant_test TearDown" << endl;
    }
};

static void InitRotateQuantTiling(RotateQuantOpt::RotateQuantTilingData* tilingData,
                                   int32_t M, int32_t N, int32_t K,
                                   int32_t numBlocks, int32_t aivCoreNum)
{
    tilingData->M = M;
    tilingData->N = N;
    tilingData->K = K;
    tilingData->loopM = DEFAULT_LOOP_M;
    tilingData->dstType = TEST_INT_8;
    tilingData->stepLoop = DEFAULT_STEP_LOOP;
    tilingData->numBlocks = numBlocks;
    tilingData->aicCoreNum = TEST_AIC_CORE_NUM;
    tilingData->aivCoreNum = aivCoreNum;
    tilingData->multiRowNumHeadCore = static_cast<uint32_t>(M);
    tilingData->headCoreNum = DEFAULT_HEAD_CORE_NUM;
    tilingData->rowPerHeadCore = static_cast<uint32_t>(M);
    tilingData->rowPerCubeHeadCore = static_cast<uint32_t>(M);
    tilingData->rowPerCubeTailCore = static_cast<uint32_t>(M);
    tilingData->rowPerVectorTailCore = static_cast<uint32_t>(M);
    tilingData->rowPerVectorLastCore = static_cast<uint32_t>(M);
    tilingData->lastUbRows = static_cast<uint32_t>(M);
    tilingData->ubSize = DEFAULT_UB_SIZE;

    int32_t rowsPerCore = tilingData->stepLoop * aivCoreNum * tilingData->multiRowNumHeadCore;
    int32_t matmulM = rowsPerCore * numBlocks;
    tilingData->matmulTiling.usedCoreNum = MATMUL_USED_CORE_NUM;
    tilingData->matmulTiling.M = matmulM;
    tilingData->matmulTiling.N = K;
    tilingData->matmulTiling.Ka = K;
    tilingData->matmulTiling.Kb = K;
    tilingData->matmulTiling.singleCoreM = matmulM;
    tilingData->matmulTiling.singleCoreN = K;
    tilingData->matmulTiling.singleCoreK = K;
    tilingData->matmulTiling.baseM = MATMUL_BASE_M;
    tilingData->matmulTiling.baseN = MATMUL_BASE_N;
    tilingData->matmulTiling.baseK = MATMUL_BASE_K;
    tilingData->matmulTiling.depthA1 = MATMUL_DEPTH_A1;
    tilingData->matmulTiling.depthB1 = MATMUL_DEPTH_B1;
    tilingData->matmulTiling.stepKa = MATMUL_STEP_KA;
    tilingData->matmulTiling.stepKb = MATMUL_STEP_KB;
    tilingData->matmulTiling.stepM = MATMUL_STEP_M;
    tilingData->matmulTiling.stepN = MATMUL_STEP_N;
    tilingData->matmulTiling.iterateOrder = MATMUL_ITERATE_ORDER;
    tilingData->matmulTiling.shareMode = MATMUL_SHARE_MODE;
    tilingData->matmulTiling.isBias = MATMUL_IS_BIAS;
    tilingData->matmulTiling.transLength = MATMUL_TRANS_LENGTH;
    tilingData->matmulTiling.dbL0A = MATMUL_DB_L0A;
    tilingData->matmulTiling.dbL0B = MATMUL_DB_L0B;
    tilingData->matmulTiling.dbL0C = MATMUL_DB_L0C;
}

TEST_F(rotate_quant_test, test_case_bf16_int8)
{
    int32_t M = TEST_M;
    int32_t N = TEST_N;
    int32_t K = TEST_K;
    int32_t numBlocks = TEST_NUM_BLOCKS;
    int32_t aivCoreNum = TEST_AIV_CORE_NUM;
    size_t xSize = static_cast<size_t>(M) * K * sizeof(uint16_t);
    size_t rotSize = static_cast<size_t>(K) * N * sizeof(uint16_t);
    size_t ySize = static_cast<size_t>(aivCoreNum) * M * N * sizeof(int8_t);
    size_t scaleSize = static_cast<size_t>(aivCoreNum) * M * sizeof(float);
    size_t workspaceSize = static_cast<size_t>(M) * N * sizeof(float);
    size_t tilingDataSize = sizeof(RotateQuantOpt::RotateQuantTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xSize);
    uint8_t* rot = (uint8_t*)AscendC::GmAlloc(rotSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(ySize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleSize);
    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    memset_s(x, xSize, 0, xSize);
    memset_s(rot, rotSize, 0, rotSize);
    memset_s(y, ySize, 0, ySize);
    memset_s(scale, scaleSize, 0, scaleSize);
    memset_s(workSpace, workspaceSize, 0, workspaceSize);

    auto* tilingData = reinterpret_cast<RotateQuantOpt::RotateQuantTilingData*>(tiling);
    memset_s(tilingData, tilingDataSize, 0, tilingDataSize);
    InitRotateQuantTiling(tilingData, M, N, K, numBlocks, aivCoreNum);

    ICPU_SET_TILING_KEY(TILING_KEY_BF16);
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    ICPU_RUN_KF(rotate_quant, KERNEL_RUN_CORE_NUM, x, rot, y, scale, workSpace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(rot);
    AscendC::GmFree(y);
    AscendC::GmFree(scale);
    AscendC::GmFree(workSpace);
    AscendC::GmFree(tiling);
}

TEST_F(rotate_quant_test, test_case_fp16_int8)
{
    int32_t M = TEST_M;
    int32_t N = TEST_N;
    int32_t K = TEST_K;
    int32_t numBlocks = TEST_NUM_BLOCKS;
    int32_t aivCoreNum = TEST_AIV_CORE_NUM;
    size_t xSize = static_cast<size_t>(M) * K * sizeof(uint16_t);
    size_t rotSize = static_cast<size_t>(K) * N * sizeof(uint16_t);
    size_t ySize = static_cast<size_t>(aivCoreNum) * M * N * sizeof(int8_t);
    size_t scaleSize = static_cast<size_t>(aivCoreNum) * M * sizeof(float);
    size_t workspaceSize = static_cast<size_t>(M) * N * sizeof(float);
    size_t tilingDataSize = sizeof(RotateQuantOpt::RotateQuantTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xSize);
    uint8_t* rot = (uint8_t*)AscendC::GmAlloc(rotSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(ySize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleSize);
    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    memset_s(x, xSize, 0, xSize);
    memset_s(rot, rotSize, 0, rotSize);
    memset_s(y, ySize, 0, ySize);
    memset_s(scale, scaleSize, 0, scaleSize);
    memset_s(workSpace, workspaceSize, 0, workspaceSize);

    auto* tilingData = reinterpret_cast<RotateQuantOpt::RotateQuantTilingData*>(tiling);
    memset_s(tilingData, tilingDataSize, 0, tilingDataSize);
    InitRotateQuantTiling(tilingData, M, N, K, numBlocks, aivCoreNum);

    ICPU_SET_TILING_KEY(TILING_KEY_FP16);
    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    ICPU_RUN_KF(rotate_quant, KERNEL_RUN_CORE_NUM, x, rot, y, scale, workSpace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(rot);
    AscendC::GmFree(y);
    AscendC::GmFree(scale);
    AscendC::GmFree(workSpace);
    AscendC::GmFree(tiling);
}
