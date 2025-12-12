/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/
#include <array>
#include <vector>
#include <gtest/gtest.h>

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#include "string.h"
#include <iostream>
#include <string>
#endif

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void embedding_bag(
    GM_ADDR weight, GM_ADDR indices, GM_ADDR offsets, GM_ADDR per_sample_weights, GM_ADDR y, GM_ADDR offset2bag,
    GM_ADDR bag_size, GM_ADDR max_indices, GM_ADDR workspace, GM_ADDR tiling);

class embedding_bag_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "embedding_bag_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "embedding_bag_test TearDown\n" << endl;
    }
};

TEST_F(embedding_bag_test, params_sum_false)
{
    size_t weight_ = 1024 * 4096 * sizeof(float);
    size_t indices_ = 1024 * sizeof(int32_t);
    size_t offsets_ = 1024 * sizeof(int32_t);
    size_t y_ = 1024 * 4096 * sizeof(float);
    size_t offset2bag_ = 1024 * sizeof(int32_t);
    size_t bag_size_ = 1024 * sizeof(int32_t);
    size_t max_indices_ = 1024 * sizeof(int32_t);
    size_t tiling_ = sizeof(EmbeddingBagTilingData);

    uint8_t* weight = (uint8_t*)AscendC::GmAlloc(weight_);
    uint8_t* indices = (uint8_t*)AscendC::GmAlloc(indices_);
    uint8_t* offsets = (uint8_t*)AscendC::GmAlloc(offsets_);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(y_);
    uint8_t* offset2bag = (uint8_t*)AscendC::GmAlloc(offset2bag_);
    uint8_t* bag_size = (uint8_t*)AscendC::GmAlloc(bag_size_);
    uint8_t* max_indices = (uint8_t*)AscendC::GmAlloc(max_indices_);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);

    memset(workspace, 0, 16 * 1024 * 1024);

    EmbeddingBagTilingData* tilingDatafromBin = reinterpret_cast<EmbeddingBagTilingData*>(tiling);

    tilingDatafromBin->formerOffsetNum = 1;
    tilingDatafromBin->tailOffsetNum = 0;
    tilingDatafromBin->numEmbeddings = 4096;
    tilingDatafromBin->computeRepTime = 1;
    tilingDatafromBin->indicesMaxMoveLength = 1024;
    tilingDatafromBin->usedCoreNum = 48;
    tilingDatafromBin->paddingIdx = -1;
    tilingDatafromBin->numIndices = 1024;
    tilingDatafromBin->mode = 1;
    tilingDatafromBin->tilingKey = 1;
    tilingDatafromBin->hasPerSampleWeights = 0;

    ICPU_SET_TILING_KEY(1);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(
        embedding_bag, 48, weight, indices, offsets, nullptr, y, offset2bag, bag_size, max_indices, workspace, tiling);

    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    AscendC::GmFree(weight);
    AscendC::GmFree(indices);
    AscendC::GmFree(offsets);
    AscendC::GmFree(y);
    AscendC::GmFree(offset2bag);
    AscendC::GmFree(bag_size);
    AscendC::GmFree(max_indices);
    AscendC::GmFree(tiling);

}

