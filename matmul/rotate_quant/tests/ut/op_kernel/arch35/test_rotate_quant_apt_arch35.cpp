/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
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
#include "../../../../op_kernel/arch35/rotate_quant_tiling_data.h"

using namespace std;
using RotateQuantAptOpt::RotateQuantAptTilingData;

extern "C" __global__ __aicore__ void rotate_quant(GM_ADDR x, GM_ADDR rot, GM_ADDR alpha, GM_ADDR y, GM_ADDR scale,
                                                   GM_ADDR workSpace, GM_ADDR tiling);

// rotate_quant arch35 kernel tiling defaults
constexpr int32_t APT_DEFAULT_CORE_NUM = 1;
constexpr int32_t APT_KERNEL_RUN_CORE_NUM = 1;
constexpr int32_t APT_DEFAULT_AXIS = -1;
constexpr int32_t APT_ROUND_MODE_RINT = 0;
constexpr int32_t APT_SCALE_ALG_DEFAULT = 0;
constexpr float APT_DST_TYPE_MAX_DEFAULT = 0.0f;

class rotate_quant_arch35_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "rotate_quant_arch35_test SetUp" << endl; }
    static void TearDownTestCase() { cout << "rotate_quant_arch35_test TearDown" << endl; }
};

static void InitRotateQuantAptTiling(RotateQuantAptTilingData* tilingData, uint32_t M, uint32_t N, uint32_t K,
                                     uint32_t nKRatio, uint32_t coreNum)
{
    tilingData->M = M;
    tilingData->N = N;
    tilingData->K = K;
    tilingData->B = (nKRatio > 1) ? nKRatio : 1;
    tilingData->nKRatio = nKRatio;
    tilingData->mL1 = M;
    tilingData->tailML1 = M;
    tilingData->nL1 = K;
    tilingData->kL1 = K;
    tilingData->baseM = M;
    tilingData->baseN = K;
    tilingData->baseK = K;
    tilingData->dstTypeMax = APT_DST_TYPE_MAX_DEFAULT;
    tilingData->invDstTypeMax = APT_DST_TYPE_MAX_DEFAULT;
    tilingData->axis = APT_DEFAULT_AXIS;
    tilingData->roundMode = APT_ROUND_MODE_RINT;
    tilingData->scaleAlg = APT_SCALE_ALG_DEFAULT;
    tilingData->trans = 0;
    tilingData->hasAlpha = 0;
    tilingData->aivNumRatio = 2; // UT按950语义(1:2)初始化
}

// hif4: bf16 x, K=64, y=HIFLOAT4 (4bit packed, 0.5 byte/elem), scale=HIFLOAT4_SCALE (4 bytes)
TEST_F(rotate_quant_arch35_test, arch35_hif4_bf16_k64)
{
    uint32_t M = 4;
    uint32_t N = 64;
    uint32_t K = 64;
    uint32_t nKRatio = N / K;
    size_t xSize = static_cast<size_t>(M) * N * sizeof(uint16_t);
    size_t rotSize = static_cast<size_t>(K) * K * sizeof(uint16_t);
    size_t ySize = static_cast<size_t>(M) * N / 2;                            // hif4: 4bit per elem
    size_t scaleSize = static_cast<size_t>(M) * (N / 64) * 1 * sizeof(float); // (M, CeilDiv(N,64), 1)
    size_t workspaceSize = static_cast<size_t>(M) * N * sizeof(float);
    size_t tilingDataSize = sizeof(RotateQuantAptTilingData);

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

    auto* tilingData = reinterpret_cast<RotateQuantAptTilingData*>(tiling);
    memset_s(tilingData, tilingDataSize, 0, tilingDataSize);
    InitRotateQuantAptTiling(tilingData, M, N, K, nKRatio, APT_DEFAULT_CORE_NUM);

    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    ICPU_RUN_KF(rotate_quant, APT_KERNEL_RUN_CORE_NUM, x, rot, nullptr, y, scale, workSpace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(rot);
    AscendC::GmFree(y);
    AscendC::GmFree(scale);
    AscendC::GmFree(workSpace);
    AscendC::GmFree(tiling);
}

// hif4: fp16 x, K=128, scale shape (M, 2, 1)
TEST_F(rotate_quant_arch35_test, arch35_hif4_fp16_k128)
{
    uint32_t M = 4;
    uint32_t N = 128;
    uint32_t K = 128;
    uint32_t nKRatio = N / K;
    size_t xSize = static_cast<size_t>(M) * N * sizeof(uint16_t);
    size_t rotSize = static_cast<size_t>(K) * K * sizeof(uint16_t);
    size_t ySize = static_cast<size_t>(M) * N / 2;
    size_t scaleSize = static_cast<size_t>(M) * (N / 64) * 1 * sizeof(float);
    size_t workspaceSize = static_cast<size_t>(M) * N * sizeof(float);
    size_t tilingDataSize = sizeof(RotateQuantAptTilingData);

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

    auto* tilingData = reinterpret_cast<RotateQuantAptTilingData*>(tiling);
    memset_s(tilingData, tilingDataSize, 0, tilingDataSize);
    InitRotateQuantAptTiling(tilingData, M, N, K, nKRatio, APT_DEFAULT_CORE_NUM);

    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    ICPU_RUN_KF(rotate_quant, APT_KERNEL_RUN_CORE_NUM, x, rot, nullptr, y, scale, workSpace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(rot);
    AscendC::GmFree(y);
    AscendC::GmFree(scale);
    AscendC::GmFree(workSpace);
    AscendC::GmFree(tiling);
}
