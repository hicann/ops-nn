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
#include "fused_linear_online_max_sum_tiling_def.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#include "string.h"
#include <iostream>
#include <string>
#endif

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void fused_linear_online_max_sum(
    GM_ADDR input, GM_ADDR weight, GM_ADDR target, GM_ADDR logitsMaxLocal, GM_ADDR sumExpLogitsLocal,
    GM_ADDR predictedLogitsLocal, GM_ADDR targetMask, GM_ADDR maskedTarget, GM_ADDR vocabParallelLogitsOut,
    GM_ADDR workspace, GM_ADDR tiling);

class fused_linear_online_max_sum_test : public testing::Test
{
    protected:

    static void SetUpTestCase()
    {
        cout << "fused_linear_online_max_sum_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "fused_linear_online_max_sum_test TearDown\n" << endl;
    }
};

TEST_F(fused_linear_online_max_sum_test, test_case_bf16_int32) 
{
    size_t bt = 128;
    size_t h = 64;
    size_t v = 256;
    int64_t sysWorkspaceSize = static_cast<int64_t>(16 * 1024 * 1024);
    int64_t workspaceSize = sysWorkspaceSize + 1024;

    size_t tilingSize = sizeof(FusedLinearOnlineMaxSumTilingData);

    uint8_t *input = (uint8_t *)AscendC::GmAlloc(bt * h * sizeof(uint16_t));
    uint8_t *weight = (uint8_t *)AscendC::GmAlloc(v * h * sizeof(uint16_t));
    uint8_t *target = (uint8_t *)AscendC::GmAlloc(bt * sizeof(uint32_t));
    uint8_t *logits_max_local = (uint8_t *)AscendC::GmAlloc(bt * sizeof(uint32_t));
    uint8_t *sum_exp_logits_local = (uint8_t *)AscendC::GmAlloc(bt * sizeof(uint32_t));
    uint8_t *predicted_logits_local = (uint8_t *)AscendC::GmAlloc(bt * sizeof(uint32_t));
    uint8_t *target_mask = (uint8_t *)AscendC::GmAlloc((bt + 7) / 8 * sizeof(uint8_t));
    uint8_t *masked_target = (uint8_t *)AscendC::GmAlloc(bt * sizeof(int32_t));
    uint8_t *vocab_parallel_logtis_out = (uint8_t *)AscendC::GmAlloc(bt * v * sizeof(uint16_t));
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingSize);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceSize);

    memset_s(input, bt * h * sizeof(uint16_t), 0, bt * h * sizeof(uint16_t));
    memset_s(weight, v * h * sizeof(uint16_t), 0, v * h * sizeof(uint16_t));
    memset_s(target, 0, bt * sizeof(uint32_t), bt * sizeof(uint32_t));
    memset_s(workspace, workspaceSize, 0, workspaceSize);
    memset_s(logits_max_local, bt * sizeof(uint32_t), 0, bt * sizeof(uint32_t));
    memset_s(sum_exp_logits_local, bt * sizeof(uint32_t), 0, bt * sizeof(uint32_t));
    memset_s(predicted_logits_local, bt * sizeof(uint32_t), 0, bt * sizeof(uint32_t));
    memset_s(target_mask, (bt + 7) / 8 * sizeof(uint8_t), 0, (bt + 7) / 8 * sizeof(uint8_t));
    memset_s(masked_target, bt * sizeof(int32_t), 0, bt * sizeof(int32_t));
    memset_s(vocab_parallel_logtis_out, bt * v * sizeof(uint16_t), 0, bt * v * sizeof(uint16_t));

    FusedLinearOnlineMaxSumTilingData *tiling_data = reinterpret_cast<FusedLinearOnlineMaxSumTilingData*>(tiling);
    tiling_data->m = bt;
    tiling_data->k = h;
    tiling_data->n = v;
    tiling_data->bufSize = static_cast<uint64_t>(188160);
    tiling_data->cubeCoreNum = static_cast<uint64_t>(1);
    tiling_data->vecCoreNum = static_cast<uint64_t>(2);
    tiling_data->batchTaksPerVecCore = static_cast<uint64_t>(8);
    tiling_data->batchTaksTailVecCore = static_cast<uint64_t>(0);
    tiling_data->targetTasksPerLoop = static_cast<uint64_t>(1960);
    tiling_data->vocabStartIndex = static_cast<float>(0);
    tiling_data->vocabEndIndex = static_cast<float>(128);
    tiling_data->initWorkspaceLength = static_cast<uint64_t>(64);
    tiling_data->cubeCoreNumAligned = static_cast<uint64_t>(24);
    tiling_data->matmulInputEmptyFlag = static_cast<uint64_t>(0);
    tiling_data->mmTiling.M = 128;
    tiling_data->mmTiling.N = 256;
    tiling_data->mmTiling.Ka = 64;
    tiling_data->mmTiling.Kb = 64;
    tiling_data->mmTiling.singleCoreM = 128;
    tiling_data->mmTiling.singleCoreN = 256;
    tiling_data->mmTiling.singleCoreK = 64;
    tiling_data->mmTiling.baseM = 128;
    tiling_data->mmTiling.baseN = 256;
    tiling_data->mmTiling.baseK = 64;
    tiling_data->mmTiling.depthA1 = 1;
    tiling_data->mmTiling.depthB1 = 1;
    tiling_data->mmTiling.depthAL1CacheUB = 0;
    tiling_data->mmTiling.depthBL1CacheUB = 0;
    tiling_data->mmTiling.stepM = 1;
    tiling_data->mmTiling.stepN = 1;
    tiling_data->mmTiling.isBias = 0;
    tiling_data->mmTiling.transLength = 0;
    tiling_data->mmTiling.iterateOrder = 0;
    tiling_data->mmTiling.shareMode = 0;
    tiling_data->mmTiling.shareL1Size = 327680;
    tiling_data->mmTiling.shareL0CSize = 131072;
    tiling_data->mmTiling.shareUbSize = 0;
    tiling_data->mmTiling.batchM = 1;
    tiling_data->mmTiling.batchN = 1;
    tiling_data->mmTiling.singleBatchM = 1;
    tiling_data->mmTiling.singleBatchN = 1;
    tiling_data->mmTiling.stepKa = 1;
    tiling_data->mmTiling.stepKb = 1;

    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(fused_linear_online_max_sum, 1, input, weight, target, logits_max_local, sum_exp_logits_local,
                predicted_logits_local, target_mask, masked_target, vocab_parallel_logtis_out, workspace, tiling);

    AscendC::GmFree(input);
    AscendC::GmFree(weight);
    AscendC::GmFree(target);
    AscendC::GmFree(logits_max_local);
    AscendC::GmFree(sum_exp_logits_local);
    AscendC::GmFree(predicted_logits_local);
    AscendC::GmFree(target_mask);
    AscendC::GmFree(masked_target);
    AscendC::GmFree(vocab_parallel_logtis_out);
    AscendC::GmFree(tiling);
    AscendC::GmFree(workspace);
}
