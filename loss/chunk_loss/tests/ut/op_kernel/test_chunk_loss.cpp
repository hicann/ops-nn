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
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>
#include "gtest/gtest.h"
#include "chunk_loss_tiling_def.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#endif

using namespace std;

extern "C" __global__ __aicore__ void chunk_loss(GM_ADDR hidden, GM_ADDR weight, GM_ADDR labels, GM_ADDR loss,
                                                 GM_ADDR maxOut, GM_ADDR sumOut, GM_ADDR dH, GM_ADDR workspace,
                                                 GM_ADDR tiling);

class chunk_loss_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "chunk_loss_test SetUp\n" << endl; }
    static void TearDownTestCase() { cout << "chunk_loss_test TearDown\n" << endl; }
};

TEST_F(chunk_loss_test, test_case_fp16_small)
{
    constexpr uint32_t m = 4;
    constexpr uint32_t h = 8;
    constexpr uint32_t v = 16;
    constexpr uint32_t aicNum = 1;

    size_t hiddenSize = static_cast<size_t>(m) * h * sizeof(half);
    size_t weightSize = static_cast<size_t>(v) * h * sizeof(half);
    size_t labelsSize = static_cast<size_t>(m) * sizeof(int64_t);
    size_t lossSize = sizeof(float);
    size_t maxSize = static_cast<size_t>(m) * sizeof(float);
    size_t sumSize = static_cast<size_t>(m) * sizeof(float);
    size_t dHSize = static_cast<size_t>(m) * h * sizeof(half);
    size_t tilingSize = sizeof(ChunkLossTilingData);
    size_t workspaceSize = 64 * 1024 * 1024;

    uint8_t* hidden = (uint8_t*)AscendC::GmAlloc(hiddenSize);
    uint8_t* weight = (uint8_t*)AscendC::GmAlloc(weightSize);
    uint8_t* labels = (uint8_t*)AscendC::GmAlloc(labelsSize);
    uint8_t* loss = (uint8_t*)AscendC::GmAlloc(lossSize);
    uint8_t* maxOut = (uint8_t*)AscendC::GmAlloc(maxSize);
    uint8_t* sumOut = (uint8_t*)AscendC::GmAlloc(sumSize);
    uint8_t* dH = (uint8_t*)AscendC::GmAlloc(dHSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    memset(hidden, 0, hiddenSize);
    memset(weight, 0, weightSize);
    memset(labels, 0, labelsSize);
    memset(loss, 0, lossSize);
    memset(maxOut, 0, maxSize);
    memset(sumOut, 0, sumSize);
    memset(dH, 0, dHSize);
    memset(workspace, 0, workspaceSize);
    memset(tiling, 0, tilingSize);

    ChunkLossTilingData* tilingData = reinterpret_cast<ChunkLossTilingData*>(tiling);
    tilingData->m = m;
    tilingData->h = h;
    tilingData->v = v;
    tilingData->aicNum = aicNum;
    tilingData->nSplit = 1;
    tilingData->deterministic = 0;
    tilingData->tileM = 128;
    tilingData->alpha = 4.0f;
    tilingData->ignoreIndex = -100;
    tilingData->syncBaseAddr = 0;

    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(chunk_loss, aicNum, hidden, weight, labels, loss, maxOut, sumOut, dH, workspace, tiling);

    AscendC::GmFree(hidden);
    AscendC::GmFree(weight);
    AscendC::GmFree(labels);
    AscendC::GmFree(loss);
    AscendC::GmFree(maxOut);
    AscendC::GmFree(sumOut);
    AscendC::GmFree(dH);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(chunk_loss_test, test_case_fp16_m1)
{
    constexpr uint32_t m = 1;
    constexpr uint32_t h = 8;
    constexpr uint32_t v = 16;
    constexpr uint32_t aicNum = 1;

    size_t hiddenSize = static_cast<size_t>(m) * h * sizeof(half);
    size_t weightSize = static_cast<size_t>(v) * h * sizeof(half);
    size_t labelsSize = static_cast<size_t>(m) * sizeof(int64_t);
    size_t lossSize = sizeof(float);
    size_t maxSize = static_cast<size_t>(m) * sizeof(float);
    size_t sumSize = static_cast<size_t>(m) * sizeof(float);
    size_t dHSize = static_cast<size_t>(m) * h * sizeof(half);
    size_t tilingSize = sizeof(ChunkLossTilingData);
    size_t workspaceSize = 64 * 1024 * 1024;

    uint8_t* hidden = (uint8_t*)AscendC::GmAlloc(hiddenSize);
    uint8_t* weight = (uint8_t*)AscendC::GmAlloc(weightSize);
    uint8_t* labels = (uint8_t*)AscendC::GmAlloc(labelsSize);
    uint8_t* loss = (uint8_t*)AscendC::GmAlloc(lossSize);
    uint8_t* maxOut = (uint8_t*)AscendC::GmAlloc(maxSize);
    uint8_t* sumOut = (uint8_t*)AscendC::GmAlloc(sumSize);
    uint8_t* dH = (uint8_t*)AscendC::GmAlloc(dHSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    memset(hidden, 0, hiddenSize);
    memset(weight, 0, weightSize);
    memset(labels, 0, labelsSize);
    memset(loss, 0, lossSize);
    memset(maxOut, 0, maxSize);
    memset(sumOut, 0, sumSize);
    memset(dH, 0, dHSize);
    memset(workspace, 0, workspaceSize);
    memset(tiling, 0, tilingSize);

    ChunkLossTilingData* tilingData = reinterpret_cast<ChunkLossTilingData*>(tiling);
    tilingData->m = m;
    tilingData->h = h;
    tilingData->v = v;
    tilingData->aicNum = aicNum;
    tilingData->nSplit = 1;
    tilingData->deterministic = 0;
    tilingData->tileM = 128;
    tilingData->alpha = 1.0f;
    tilingData->ignoreIndex = -100;
    tilingData->syncBaseAddr = 0;

    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(chunk_loss, aicNum, hidden, weight, labels, loss, maxOut, sumOut, dH, workspace, tiling);

    AscendC::GmFree(hidden);
    AscendC::GmFree(weight);
    AscendC::GmFree(labels);
    AscendC::GmFree(loss);
    AscendC::GmFree(maxOut);
    AscendC::GmFree(sumOut);
    AscendC::GmFree(dH);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
