/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include <numeric>
#include <functional>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"
#include "fused_linear_cross_entropy_loss_grad_tiling_data_def.h"

using namespace std;

extern "C" __global__ __aicore__ void fused_linear_cross_entropy_loss_grad(GM_ADDR grad, GM_ADDR input, GM_ADDR weight,
                                                                           GM_ADDR target_mask, GM_ADDR masked_target,
                                                                           GM_ADDR logits_max, GM_ADDR sum_exp_logits,
                                                                           GM_ADDR softmax, GM_ADDR input_grad_out,
                                                                           GM_ADDR weight_grad_out, GM_ADDR workspace,
                                                                           GM_ADDR tiling);

class fused_linear_cross_entropy_loss_grad_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "fused_linear_cross_entropy_loss_grad_test SetUp\n" << std::endl; }
    static void TearDownTestCase() { std::cout << "fused_linear_cross_entropy_loss_grad_test TearDown\n" << std::endl; }
};

TEST_F(fused_linear_cross_entropy_loss_grad_test, test_high_perf_case)
{
    uint32_t BT = 8;
    uint32_t V = 128;
    uint32_t H = 8;
    uint8_t* input = (uint8_t*)AscendC::GmAlloc(BT * H * 2);
    uint8_t* weight = (uint8_t*)AscendC::GmAlloc(V * H * 2);
    uint8_t* softmax = (uint8_t*)AscendC::GmAlloc(BT * V * 4);
    uint8_t* target_mask = (uint8_t*)AscendC::GmAlloc(BT * 1);
    uint8_t* masked_target = (uint8_t*)AscendC::GmAlloc(BT * 4);
    uint8_t* grad_output = (uint8_t*)AscendC::GmAlloc(BT * 4);
    uint8_t* grad_input = (uint8_t*)AscendC::GmAlloc(BT * H * 2);
    uint8_t* grad_weight = (uint8_t*)AscendC::GmAlloc(V * H * 2);
    uint8_t* logits_max = nullptr;
    uint8_t* sum_exp_logits = nullptr;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024); // >= userWorkspaceSize * 2B
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(FusedLinearCrossEntropyLossGradHighPerfTilingData));

    char* path_ = get_current_dir_name();
    string path(path_);

    FusedLinearCrossEntropyLossGradHighPerfTilingData*
        tilingData = reinterpret_cast<FusedLinearCrossEntropyLossGradHighPerfTilingData*>(tiling);

    tilingData->mm1Tiling.usedCoreNum = 20;
    tilingData->mm1Tiling.M = 128;
    tilingData->mm1Tiling.N = 8;
    tilingData->mm1Tiling.Ka = 128;
    tilingData->mm1Tiling.Kb = 128;
    tilingData->mm1Tiling.singleCoreM = 128;
    tilingData->mm1Tiling.singleCoreN = 256;
    tilingData->mm1Tiling.singleCoreK = 128;
    tilingData->mm1Tiling.baseM = 128;
    tilingData->mm1Tiling.baseN = 256;
    tilingData->mm1Tiling.baseK = 64;
    tilingData->mm1Tiling.depthA1 = 8;
    tilingData->mm1Tiling.depthB1 = 8;
    tilingData->mm1Tiling.stepM = 1;
    tilingData->mm1Tiling.stepN = 1;
    tilingData->mm1Tiling.isBias = 0;
    tilingData->mm1Tiling.transLength = 0;
    tilingData->mm1Tiling.iterateOrder = 0;
    tilingData->mm1Tiling.stepKa = 4;
    tilingData->mm1Tiling.stepKb = 4;
    tilingData->mm1Tiling.depthAL1CacheUB = 0;
    tilingData->mm1Tiling.depthBL1CacheUB = 0;
    tilingData->mm1Tiling.dbL0A = 2;
    tilingData->mm1Tiling.dbL0B = 2;
    tilingData->mm1Tiling.dbL0C = 1;

    tilingData->mm2Tiling.usedCoreNum = 20;
    tilingData->mm2Tiling.M = 128;
    tilingData->mm2Tiling.N = 8;
    tilingData->mm2Tiling.Ka = 16;
    tilingData->mm2Tiling.Kb = 8;
    tilingData->mm2Tiling.singleCoreM = 128;
    tilingData->mm2Tiling.singleCoreN = 256;
    tilingData->mm2Tiling.singleCoreK = 8;
    tilingData->mm2Tiling.baseM = 128;
    tilingData->mm2Tiling.baseN = 256;
    tilingData->mm2Tiling.baseK = 64;
    tilingData->mm2Tiling.depthA1 = 8;
    tilingData->mm2Tiling.depthB1 = 8;
    tilingData->mm2Tiling.stepM = 1;
    tilingData->mm2Tiling.stepN = 1;
    tilingData->mm2Tiling.isBias = 0;
    tilingData->mm2Tiling.transLength = 0;
    tilingData->mm2Tiling.iterateOrder = 0;
    tilingData->mm2Tiling.stepKa = 4;
    tilingData->mm2Tiling.stepKb = 4;
    tilingData->mm2Tiling.depthAL1CacheUB = 0;
    tilingData->mm2Tiling.depthBL1CacheUB = 0;
    tilingData->mm2Tiling.dbL0A = 2;
    tilingData->mm2Tiling.dbL0B = 2;
    tilingData->mm2Tiling.dbL0C = 1;

    tilingData->aicNum = 20;
    tilingData->aivNum = 40;
    tilingData->BT = 8;
    tilingData->V = 128;
    tilingData->H = 8;
    tilingData->BT_16ALIGNED = 16;
    tilingData->btPerCore = 1;
    tilingData->btLastCore = 0;
    tilingData->btLastCoreId = 8;
    tilingData->targetMaskSize = 8;
    tilingData->V64BAlignedSize = 128;
    tilingData->copyOutDstStride = 256;
    tilingData->VOut512BAlignedSize = 256;
    tilingData->VOut32BAlignedSize = 128;
    tilingData->wsNum = 1;
    tilingData->userWorkspaceSize = 16384; // max(VOut32B=128, BT16=16) * baseM(128) * wsNum(1)
    tilingData->btPerEpochSingle = 4;
    tilingData->btPerEpochTotal = 160;
    tilingData->quebindSoftmaxByteSize = 65536; // max(V64B*btPerEpochSingle*4, 256*64*2*2)
    tilingData->copyInDstStride = 0;
    tilingData->epochCnt = 1;
    tilingData->lastEpochCnt = 1;
    tilingData->using32BlockCnt = 0;
    tilingData->sizePerTask = 0;
    tilingData->taskCntPerV = 0;
    tilingData->sizeLastTask = 0;
    tilingData->taskCntTotal = 0;
    tilingData->mm1BaseMCnt = 1;
    tilingData->mm1BaseNCnt = 1;
    tilingData->mm1TailM = 8;
    tilingData->mm1TailN = 8;
    tilingData->mm2BaseMCnt = 1; // (V-1)/baseM+1 = 127/128+1，V=128 仅 1 个 baseM 块
    tilingData->mm2BaseNCnt = 1;
    tilingData->mm2BaseKCnt = 1;
    tilingData->mm2TailM = 128;
    tilingData->mm2TailN = 8;
    tilingData->mm2TailK = 8;
    tilingData->mm2BasePerVec = 1;
    tilingData->mm2BaseLastVec = 0;
    tilingData->mm2BaseLastVecId = 1;
    tilingData->mm2LastBaseK = 56;

    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    ICPU_SET_TILING_KEY(100UL);
    ICPU_RUN_KF(fused_linear_cross_entropy_loss_grad, 20, grad_output, input, weight, target_mask, masked_target,
                logits_max, sum_exp_logits, softmax, grad_input, grad_weight, workspace, (uint8_t*)(tilingData));

    AscendC::GmFree(input);
    AscendC::GmFree(weight);
    AscendC::GmFree(softmax);
    AscendC::GmFree(target_mask);
    AscendC::GmFree(masked_target);
    AscendC::GmFree(grad_output);
    AscendC::GmFree(grad_input);
    AscendC::GmFree(grad_weight);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(fused_linear_cross_entropy_loss_grad_test, test_mem_friendly_case)
{
    uint32_t BT = 8;
    uint32_t V = 128;
    uint32_t H = 8;
    uint8_t* grad_output = (uint8_t*)AscendC::GmAlloc(BT * 4);
    uint8_t* input = (uint8_t*)AscendC::GmAlloc(BT * H * 2);
    uint8_t* weight = (uint8_t*)AscendC::GmAlloc(V * H * 2);
    uint8_t* target_mask = (uint8_t*)AscendC::GmAlloc(BT * 1);
    uint8_t* masked_target = (uint8_t*)AscendC::GmAlloc(BT * 4);
    uint8_t* logits_max = (uint8_t*)AscendC::GmAlloc(BT * 4);
    uint8_t* sum_exp_logits = (uint8_t*)AscendC::GmAlloc(BT * 4);
    uint8_t* softmax = nullptr;
    uint8_t* grad_input = (uint8_t*)AscendC::GmAlloc(BT * H * 2);
    uint8_t* grad_weight = (uint8_t*)AscendC::GmAlloc(V * H * 2);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(22520320);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(FusedLinearCrossEntropyLossGradMemFriendlyTilingData));

    char* path_ = get_current_dir_name();
    string path(path_);

    FusedLinearCrossEntropyLossGradMemFriendlyTilingData*
        tilingData = reinterpret_cast<FusedLinearCrossEntropyLossGradMemFriendlyTilingData*>(tiling);

    tilingData->aicNum = 20;
    tilingData->aivNum = 40;
    tilingData->BT = 8;
    tilingData->V = 128;
    tilingData->H = 8;
    tilingData->targetMaskSize = 8;
    tilingData->baseBT = 128;
    tilingData->baseV = 256;
    tilingData->baseBTTail = 8;
    tilingData->baseVTail = 128;
    tilingData->mm0ResultSize = 32768;
    tilingData->mm0ResultSizePerEpoch = 655360;
    tilingData->V1MainSize = 16384;
    tilingData->V1InUBStride = 16;
    tilingData->V1OutUBStride = 8;
    tilingData->mm1ResultSize = 1024;
    tilingData->mm1ResultSizePerEpoch = 20480;
    tilingData->mm2ResultSize = 2048;
    tilingData->mm2ResultSizePerEpoch = 40960;
    tilingData->mmAccumRowOnetime = 1534;
    tilingData->mmAccumCyclePerRow = 0;
    tilingData->mmAccumCntPerCycle = 0;
    tilingData->mmAccumCntLastCycle = 0;
    tilingData->V2MainSize = 12272;
    tilingData->mmCastRowOnetime = 3068;
    tilingData->mmCastCyclePerRow = 0;
    tilingData->mmCastCntPerCycle = 0;
    tilingData->mmCastCntLastCycle = 0;
    tilingData->castMainSize = 24544;
    tilingData->poolByteSize = 196352;
    tilingData->mm1CastRowPerVec = 0;
    tilingData->mm1CastExtraRowCnt = 8;
    tilingData->mm2CastRowPerVec = 3;
    tilingData->mm2CastExtraRowCnt = 8;

    tilingData->mm0Tiling.usedCoreNum = 20;
    tilingData->mm0Tiling.M = 8;
    tilingData->mm0Tiling.N = 128;
    tilingData->mm0Tiling.Ka = 8;
    tilingData->mm0Tiling.Kb = 8;
    tilingData->mm0Tiling.singleCoreM = 8;
    tilingData->mm0Tiling.singleCoreN = 128;
    tilingData->mm0Tiling.singleCoreK = 8;
    tilingData->mm0Tiling.baseM = 16;
    tilingData->mm0Tiling.baseN = 128;
    tilingData->mm0Tiling.baseK = 16;
    tilingData->mm0Tiling.depthA1 = 1;
    tilingData->mm0Tiling.depthB1 = 1;
    tilingData->mm0Tiling.stepM = 1;
    tilingData->mm0Tiling.stepN = 1;
    tilingData->mm0Tiling.isBias = 0;
    tilingData->mm0Tiling.transLength = 0;
    tilingData->mm0Tiling.iterateOrder = 0;
    tilingData->mm0Tiling.shareMode = 0;
    tilingData->mm0Tiling.shareL1Size = 4608;
    tilingData->mm0Tiling.shareL0CSize = 8192;
    tilingData->mm0Tiling.shareUbSize = 0;
    tilingData->mm0Tiling.batchM = 1;
    tilingData->mm0Tiling.batchN = 1;
    tilingData->mm0Tiling.singleBatchM = 1;
    tilingData->mm0Tiling.singleBatchN = 1;
    tilingData->mm0Tiling.stepKa = 1;
    tilingData->mm0Tiling.stepKb = 1;
    tilingData->mm0Tiling.depthAL1CacheUB = 0;
    tilingData->mm0Tiling.depthBL1CacheUB = 0;
    tilingData->mm0Tiling.dbL0A = 2;
    tilingData->mm0Tiling.dbL0B = 2;
    tilingData->mm0Tiling.dbL0C = 1;
    tilingData->mm0Tiling.ALayoutInfoB = 0;
    tilingData->mm0Tiling.ALayoutInfoS = 0;
    tilingData->mm0Tiling.ALayoutInfoN = 0;
    tilingData->mm0Tiling.ALayoutInfoG = 0;
    tilingData->mm0Tiling.ALayoutInfoD = 0;
    tilingData->mm0Tiling.BLayoutInfoB = 0;
    tilingData->mm0Tiling.BLayoutInfoS = 0;
    tilingData->mm0Tiling.BLayoutInfoN = 0;
    tilingData->mm0Tiling.BLayoutInfoG = 0;
    tilingData->mm0Tiling.BLayoutInfoD = 0;
    tilingData->mm0Tiling.CLayoutInfoB = 0;
    tilingData->mm0Tiling.CLayoutInfoS1 = 0;
    tilingData->mm0Tiling.CLayoutInfoN = 0;
    tilingData->mm0Tiling.CLayoutInfoG = 0;
    tilingData->mm0Tiling.CLayoutInfoS2 = 0;
    tilingData->mm0Tiling.BatchNum = 0;
    tilingData->mm0Tiling.mxTypePara = 0;
    tilingData->mm1Tiling.usedCoreNum = 20;
    tilingData->mm1Tiling.M = 8;
    tilingData->mm1Tiling.N = 8;
    tilingData->mm1Tiling.Ka = 128;
    tilingData->mm1Tiling.Kb = 128;
    tilingData->mm1Tiling.singleCoreM = 8;
    tilingData->mm1Tiling.singleCoreN = 8;
    tilingData->mm1Tiling.singleCoreK = 128;
    tilingData->mm1Tiling.baseM = 16;
    tilingData->mm1Tiling.baseN = 16;
    tilingData->mm1Tiling.baseK = 128;
    tilingData->mm1Tiling.depthA1 = 1;
    tilingData->mm1Tiling.depthB1 = 1;
    tilingData->mm1Tiling.stepM = 1;
    tilingData->mm1Tiling.stepN = 1;
    tilingData->mm1Tiling.isBias = 0;
    tilingData->mm1Tiling.transLength = 0;
    tilingData->mm1Tiling.iterateOrder = 0;
    tilingData->mm1Tiling.shareMode = 0;
    tilingData->mm1Tiling.shareL1Size = 8192;
    tilingData->mm1Tiling.shareL0CSize = 1024;
    tilingData->mm1Tiling.shareUbSize = 0;
    tilingData->mm1Tiling.batchM = 1;
    tilingData->mm1Tiling.batchN = 1;
    tilingData->mm1Tiling.singleBatchM = 1;
    tilingData->mm1Tiling.singleBatchN = 1;
    tilingData->mm1Tiling.stepKa = 1;
    tilingData->mm1Tiling.stepKb = 1;
    tilingData->mm1Tiling.depthAL1CacheUB = 0;
    tilingData->mm1Tiling.depthBL1CacheUB = 0;
    tilingData->mm1Tiling.dbL0A = 2;
    tilingData->mm1Tiling.dbL0B = 2;
    tilingData->mm1Tiling.dbL0C = 1;
    tilingData->mm1Tiling.ALayoutInfoB = 0;
    tilingData->mm1Tiling.ALayoutInfoS = 0;
    tilingData->mm1Tiling.ALayoutInfoN = 0;
    tilingData->mm1Tiling.ALayoutInfoG = 0;
    tilingData->mm1Tiling.ALayoutInfoD = 0;
    tilingData->mm1Tiling.BLayoutInfoB = 0;
    tilingData->mm1Tiling.BLayoutInfoS = 0;
    tilingData->mm1Tiling.BLayoutInfoN = 0;
    tilingData->mm1Tiling.BLayoutInfoG = 0;
    tilingData->mm1Tiling.BLayoutInfoD = 0;
    tilingData->mm1Tiling.CLayoutInfoB = 0;
    tilingData->mm1Tiling.CLayoutInfoS1 = 0;
    tilingData->mm1Tiling.CLayoutInfoN = 0;
    tilingData->mm1Tiling.CLayoutInfoG = 0;
    tilingData->mm1Tiling.CLayoutInfoS2 = 0;
    tilingData->mm1Tiling.BatchNum = 0;
    tilingData->mm1Tiling.mxTypePara = 0;
    tilingData->mm2Tiling.usedCoreNum = 20;
    tilingData->mm2Tiling.M = 128;
    tilingData->mm2Tiling.N = 8;
    tilingData->mm2Tiling.Ka = 8;
    tilingData->mm2Tiling.Kb = 8;
    tilingData->mm2Tiling.singleCoreM = 128;
    tilingData->mm2Tiling.singleCoreN = 8;
    tilingData->mm2Tiling.singleCoreK = 8;
    tilingData->mm2Tiling.baseM = 128;
    tilingData->mm2Tiling.baseN = 16;
    tilingData->mm2Tiling.baseK = 16;
    tilingData->mm2Tiling.depthA1 = 1;
    tilingData->mm2Tiling.depthB1 = 1;
    tilingData->mm2Tiling.stepM = 1;
    tilingData->mm2Tiling.stepN = 1;
    tilingData->mm2Tiling.isBias = 0;
    tilingData->mm2Tiling.transLength = 0;
    tilingData->mm2Tiling.iterateOrder = 0;
    tilingData->mm2Tiling.shareMode = 0;
    tilingData->mm2Tiling.shareL1Size = 4608;
    tilingData->mm2Tiling.shareL0CSize = 8192;
    tilingData->mm2Tiling.shareUbSize = 0;
    tilingData->mm2Tiling.batchM = 1;
    tilingData->mm2Tiling.batchN = 1;
    tilingData->mm2Tiling.singleBatchM = 1;
    tilingData->mm2Tiling.singleBatchN = 1;
    tilingData->mm2Tiling.stepKa = 1;
    tilingData->mm2Tiling.stepKb = 1;
    tilingData->mm2Tiling.depthAL1CacheUB = 0;
    tilingData->mm2Tiling.depthBL1CacheUB = 0;
    tilingData->mm2Tiling.dbL0A = 2;
    tilingData->mm2Tiling.dbL0B = 2;
    tilingData->mm2Tiling.dbL0C = 1;
    tilingData->mm2Tiling.ALayoutInfoB = 0;
    tilingData->mm2Tiling.ALayoutInfoS = 0;
    tilingData->mm2Tiling.ALayoutInfoN = 0;
    tilingData->mm2Tiling.ALayoutInfoG = 0;
    tilingData->mm2Tiling.ALayoutInfoD = 0;
    tilingData->mm2Tiling.BLayoutInfoB = 0;
    tilingData->mm2Tiling.BLayoutInfoS = 0;
    tilingData->mm2Tiling.BLayoutInfoN = 0;
    tilingData->mm2Tiling.BLayoutInfoG = 0;
    tilingData->mm2Tiling.BLayoutInfoD = 0;
    tilingData->mm2Tiling.CLayoutInfoB = 0;
    tilingData->mm2Tiling.CLayoutInfoS1 = 0;
    tilingData->mm2Tiling.CLayoutInfoN = 0;
    tilingData->mm2Tiling.CLayoutInfoG = 0;
    tilingData->mm2Tiling.CLayoutInfoS2 = 0;
    tilingData->mm2Tiling.BatchNum = 0;
    tilingData->mm2Tiling.mxTypePara = 0;

    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    ICPU_SET_TILING_KEY(101UL);
    ICPU_RUN_KF(fused_linear_cross_entropy_loss_grad, 20, grad_output, input, weight, target_mask, masked_target,
                logits_max, sum_exp_logits, softmax, grad_input, grad_weight, workspace, (uint8_t*)(tilingData));

    AscendC::GmFree(grad_output);
    AscendC::GmFree(input);
    AscendC::GmFree(weight);
    AscendC::GmFree(target_mask);
    AscendC::GmFree(masked_target);
    AscendC::GmFree(logits_max);
    AscendC::GmFree(sum_exp_logits);
    AscendC::GmFree(grad_input);
    AscendC::GmFree(grad_weight);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

// ascend950 满核28块: BT=512 触发多 baseM 块与 wsNum workspace 轮转路径
// tiling 字段依据 HighPerfTilingFunc 公式推导:
//   BT_16ALIGNED=512, baseM/baseN=128/256, baseK=64(BT<23296)
//   V64BAlignedSize=256, VOut512BAlignedSize=256, VOut32BAlignedSize=256, stride均为0(V*4/V*2 恰好对齐)
//   mm1BaseMCnt=4(512/128), mm1BaseNCnt=1(H=128<256), mm2BaseMCnt=2(256/128), mm2BaseKCnt=8(512/64)
//   btPerEpochSingle=3, epochCnt=1, wsNum=min(56, max(4,2))=4
//   userWorkspaceSize = max(256,512)*128*4 = 262144 (half元素) -> 分配 1MB
TEST_F(fused_linear_cross_entropy_loss_grad_test, test_high_perf_case_aicore28_multiblock)
{
#if defined(UT_SOC_VERSION)
    // 本用例按 ascend950 满核28块拓扑构造 tiling(aicNum=28/aivNum=56),
    // 仅在 950 的 UT 目标中执行; 其它平台(如 910B1 上限20核)会因块数超限 abort
    if (std::string(UT_SOC_VERSION).find("950") == std::string::npos) {
        GTEST_SKIP() << "950 aicore28 topology UT, skip on " << UT_SOC_VERSION;
    }
#endif
    uint32_t BT = 512;
    uint32_t V = 256;
    uint32_t H = 128;
    uint8_t* input = (uint8_t*)AscendC::GmAlloc(BT * H * 2);
    uint8_t* weight = (uint8_t*)AscendC::GmAlloc(V * H * 2);
    uint8_t* softmax = (uint8_t*)AscendC::GmAlloc(BT * V * 4);
    uint8_t* target_mask = (uint8_t*)AscendC::GmAlloc(BT * 1);
    uint8_t* masked_target = (uint8_t*)AscendC::GmAlloc(BT * 4);
    uint8_t* grad_output = (uint8_t*)AscendC::GmAlloc(BT * 4);
    uint8_t* grad_input = (uint8_t*)AscendC::GmAlloc(BT * H * 2);
    uint8_t* grad_weight = (uint8_t*)AscendC::GmAlloc(V * H * 2);
    uint8_t* logits_max = nullptr;
    uint8_t* sum_exp_logits = nullptr;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(FusedLinearCrossEntropyLossGradHighPerfTilingData));

    FusedLinearCrossEntropyLossGradHighPerfTilingData*
        tilingData = reinterpret_cast<FusedLinearCrossEntropyLossGradHighPerfTilingData*>(tiling);

    tilingData->mm1Tiling.usedCoreNum = 28;
    tilingData->mm1Tiling.M = 128;
    tilingData->mm1Tiling.N = 128;
    tilingData->mm1Tiling.Ka = 256;
    tilingData->mm1Tiling.Kb = 256;
    tilingData->mm1Tiling.singleCoreM = 128;
    tilingData->mm1Tiling.singleCoreN = 128;
    tilingData->mm1Tiling.singleCoreK = 256;
    tilingData->mm1Tiling.baseM = 128;
    tilingData->mm1Tiling.baseN = 256;
    tilingData->mm1Tiling.baseK = 64;
    tilingData->mm1Tiling.depthA1 = 8;
    tilingData->mm1Tiling.depthB1 = 8;
    tilingData->mm1Tiling.stepM = 1;
    tilingData->mm1Tiling.stepN = 1;
    tilingData->mm1Tiling.isBias = 0;
    tilingData->mm1Tiling.transLength = 0;
    tilingData->mm1Tiling.iterateOrder = 0;
    tilingData->mm1Tiling.stepKa = 4;
    tilingData->mm1Tiling.stepKb = 4;
    tilingData->mm1Tiling.depthAL1CacheUB = 0;
    tilingData->mm1Tiling.depthBL1CacheUB = 0;
    tilingData->mm1Tiling.dbL0A = 2;
    tilingData->mm1Tiling.dbL0B = 2;
    tilingData->mm1Tiling.dbL0C = 1;

    tilingData->mm2Tiling.usedCoreNum = 28;
    tilingData->mm2Tiling.M = 128;
    tilingData->mm2Tiling.N = 128;
    tilingData->mm2Tiling.Ka = 512;
    tilingData->mm2Tiling.Kb = 512;
    tilingData->mm2Tiling.singleCoreM = 128;
    tilingData->mm2Tiling.singleCoreN = 128;
    tilingData->mm2Tiling.singleCoreK = 512;
    tilingData->mm2Tiling.baseM = 128;
    tilingData->mm2Tiling.baseN = 256;
    tilingData->mm2Tiling.baseK = 64;
    tilingData->mm2Tiling.depthA1 = 8;
    tilingData->mm2Tiling.depthB1 = 8;
    tilingData->mm2Tiling.stepM = 1;
    tilingData->mm2Tiling.stepN = 1;
    tilingData->mm2Tiling.isBias = 0;
    tilingData->mm2Tiling.transLength = 0;
    tilingData->mm2Tiling.iterateOrder = 0;
    tilingData->mm2Tiling.stepKa = 4;
    tilingData->mm2Tiling.stepKb = 4;
    tilingData->mm2Tiling.depthAL1CacheUB = 0;
    tilingData->mm2Tiling.depthBL1CacheUB = 0;
    tilingData->mm2Tiling.dbL0A = 2;
    tilingData->mm2Tiling.dbL0B = 2;
    tilingData->mm2Tiling.dbL0C = 1;

    tilingData->aicNum = 28;
    tilingData->aivNum = 56;
    tilingData->BT = 512;
    tilingData->V = 256;
    tilingData->H = 128;
    tilingData->BT_16ALIGNED = 512;
    tilingData->btPerCore = 10;
    tilingData->btLastCore = 2;
    tilingData->btLastCoreId = 51;
    tilingData->targetMaskSize = 512;
    tilingData->V64BAlignedSize = 256;
    tilingData->copyOutDstStride = 0;
    tilingData->VOut512BAlignedSize = 256;
    tilingData->VOut32BAlignedSize = 256;
    tilingData->wsNum = 4;
    tilingData->userWorkspaceSize = 262144; // max(VOut32B=256, BT16=512) * baseM(128) * wsNum(4)
    tilingData->btPerEpochSingle = 3;
    tilingData->btPerEpochTotal = 168;
    tilingData->quebindSoftmaxByteSize = 65536;
    tilingData->copyInDstStride = 0;
    tilingData->epochCnt = 1;
    tilingData->lastEpochCnt = 1;
    tilingData->using32BlockCnt = 0;
    tilingData->sizePerTask = 0;
    tilingData->taskCntPerV = 0;
    tilingData->sizeLastTask = 0;
    tilingData->taskCntTotal = 0;
    tilingData->mm1BaseMCnt = 4;
    tilingData->mm1BaseNCnt = 1;
    tilingData->mm1TailM = 128;
    tilingData->mm1TailN = 128;
    tilingData->mm2BaseMCnt = 2;
    tilingData->mm2BaseNCnt = 1;
    tilingData->mm2BaseKCnt = 8;
    tilingData->mm2TailM = 128;
    tilingData->mm2TailN = 128;
    tilingData->mm2TailK = 64;
    tilingData->mm2BasePerVec = 1;
    tilingData->mm2BaseLastVec = 0;
    tilingData->mm2BaseLastVecId = 8;
    tilingData->mm2LastBaseK = 0;

    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    ICPU_SET_TILING_KEY(100UL);
    ICPU_RUN_KF(fused_linear_cross_entropy_loss_grad, 28, grad_output, input, weight, target_mask, masked_target,
                logits_max, sum_exp_logits, softmax, grad_input, grad_weight, workspace, (uint8_t*)(tilingData));

    AscendC::GmFree(input);
    AscendC::GmFree(weight);
    AscendC::GmFree(softmax);
    AscendC::GmFree(target_mask);
    AscendC::GmFree(masked_target);
    AscendC::GmFree(grad_output);
    AscendC::GmFree(grad_input);
    AscendC::GmFree(grad_weight);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

// ascend950 满核28块: mem_friendly 分支(softmax为空, lmax/se 重算路径)
TEST_F(fused_linear_cross_entropy_loss_grad_test, test_mem_friendly_case_aicore28)
{
#if defined(UT_SOC_VERSION)
    // 本用例按 ascend950 满核28块拓扑构造 tiling, 仅在 950 的 UT 目标中执行
    if (std::string(UT_SOC_VERSION).find("950") == std::string::npos) {
        GTEST_SKIP() << "950 aicore28 topology UT, skip on " << UT_SOC_VERSION;
    }
#endif
    uint32_t BT = 8;
    uint32_t V = 128;
    uint32_t H = 8;
    uint8_t* grad_output = (uint8_t*)AscendC::GmAlloc(BT * 4);
    uint8_t* input = (uint8_t*)AscendC::GmAlloc(BT * H * 2);
    uint8_t* weight = (uint8_t*)AscendC::GmAlloc(V * H * 2);
    uint8_t* target_mask = (uint8_t*)AscendC::GmAlloc(BT * 1);
    uint8_t* masked_target = (uint8_t*)AscendC::GmAlloc(BT * 4);
    uint8_t* logits_max = (uint8_t*)AscendC::GmAlloc(BT * 4);
    uint8_t* sum_exp_logits = (uint8_t*)AscendC::GmAlloc(BT * 4);
    uint8_t* softmax = nullptr;
    uint8_t* grad_input = (uint8_t*)AscendC::GmAlloc(BT * H * 2);
    uint8_t* grad_weight = (uint8_t*)AscendC::GmAlloc(V * H * 2);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(22520320);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(FusedLinearCrossEntropyLossGradMemFriendlyTilingData));

    FusedLinearCrossEntropyLossGradMemFriendlyTilingData*
        tilingData = reinterpret_cast<FusedLinearCrossEntropyLossGradMemFriendlyTilingData*>(tiling);

    tilingData->aicNum = 28;
    tilingData->aivNum = 56;
    tilingData->BT = 8;
    tilingData->V = 128;
    tilingData->H = 8;
    tilingData->targetMaskSize = 8;
    tilingData->baseBT = 128;
    tilingData->baseV = 256;
    tilingData->baseBTTail = 8;
    tilingData->baseVTail = 128;
    tilingData->mm0ResultSize = 32768;
    tilingData->mm0ResultSizePerEpoch = 917504;
    tilingData->V1MainSize = 16384;
    tilingData->V1InUBStride = 16;
    tilingData->V1OutUBStride = 8;
    tilingData->mm1ResultSize = 1024;
    tilingData->mm1ResultSizePerEpoch = 28672;
    tilingData->mm2ResultSize = 2048;
    tilingData->mm2ResultSizePerEpoch = 57344;
    tilingData->mmAccumRowOnetime = 1534;
    tilingData->mmAccumCyclePerRow = 0;
    tilingData->mmAccumCntPerCycle = 0;
    tilingData->mmAccumCntLastCycle = 0;
    tilingData->V2MainSize = 12272;
    tilingData->mmCastRowOnetime = 3068;
    tilingData->mmCastCyclePerRow = 0;
    tilingData->mmCastCntPerCycle = 0;
    tilingData->mmCastCntLastCycle = 0;
    tilingData->castMainSize = 24544;
    tilingData->poolByteSize = 196352;
    tilingData->mm1CastRowPerVec = 0;
    tilingData->mm1CastExtraRowCnt = 8;
    tilingData->mm2CastRowPerVec = 3;
    tilingData->mm2CastExtraRowCnt = 8;

    // mm0/mm1/mm2 TCubeTiling: 与既有 mem 用例同形状, 仅核数改为 28
    tilingData->mm0Tiling.usedCoreNum = 28;
    tilingData->mm0Tiling.M = 8;
    tilingData->mm0Tiling.N = 128;
    tilingData->mm0Tiling.Ka = 8;
    tilingData->mm0Tiling.Kb = 8;
    tilingData->mm0Tiling.singleCoreM = 8;
    tilingData->mm0Tiling.singleCoreN = 128;
    tilingData->mm0Tiling.singleCoreK = 8;
    tilingData->mm0Tiling.baseM = 16;
    tilingData->mm0Tiling.baseN = 128;
    tilingData->mm0Tiling.baseK = 16;
    tilingData->mm0Tiling.depthA1 = 1;
    tilingData->mm0Tiling.depthB1 = 1;
    tilingData->mm0Tiling.stepM = 1;
    tilingData->mm0Tiling.stepN = 1;
    tilingData->mm0Tiling.isBias = 0;
    tilingData->mm0Tiling.transLength = 0;
    tilingData->mm0Tiling.iterateOrder = 0;
    tilingData->mm0Tiling.shareMode = 0;
    tilingData->mm0Tiling.shareL1Size = 4608;
    tilingData->mm0Tiling.shareL0CSize = 8192;
    tilingData->mm0Tiling.shareUbSize = 0;
    tilingData->mm0Tiling.batchM = 1;
    tilingData->mm0Tiling.batchN = 1;
    tilingData->mm0Tiling.singleBatchM = 1;
    tilingData->mm0Tiling.singleBatchN = 1;
    tilingData->mm0Tiling.stepKa = 1;
    tilingData->mm0Tiling.stepKb = 1;
    tilingData->mm0Tiling.depthAL1CacheUB = 0;
    tilingData->mm0Tiling.depthBL1CacheUB = 0;
    tilingData->mm0Tiling.dbL0A = 2;
    tilingData->mm0Tiling.dbL0B = 2;
    tilingData->mm0Tiling.dbL0C = 1;

    tilingData->mm1Tiling.usedCoreNum = 28;
    tilingData->mm1Tiling.M = 8;
    tilingData->mm1Tiling.N = 8;
    tilingData->mm1Tiling.Ka = 128;
    tilingData->mm1Tiling.Kb = 128;
    tilingData->mm1Tiling.singleCoreM = 8;
    tilingData->mm1Tiling.singleCoreN = 8;
    tilingData->mm1Tiling.singleCoreK = 128;
    tilingData->mm1Tiling.baseM = 16;
    tilingData->mm1Tiling.baseN = 16;
    tilingData->mm1Tiling.baseK = 128;
    tilingData->mm1Tiling.depthA1 = 1;
    tilingData->mm1Tiling.depthB1 = 1;
    tilingData->mm1Tiling.stepM = 1;
    tilingData->mm1Tiling.stepN = 1;
    tilingData->mm1Tiling.isBias = 0;
    tilingData->mm1Tiling.transLength = 0;
    tilingData->mm1Tiling.iterateOrder = 0;
    tilingData->mm1Tiling.shareMode = 0;
    tilingData->mm1Tiling.shareL1Size = 8192;
    tilingData->mm1Tiling.shareL0CSize = 1024;
    tilingData->mm1Tiling.shareUbSize = 0;
    tilingData->mm1Tiling.batchM = 1;
    tilingData->mm1Tiling.batchN = 1;
    tilingData->mm1Tiling.singleBatchM = 1;
    tilingData->mm1Tiling.singleBatchN = 1;
    tilingData->mm1Tiling.stepKa = 1;
    tilingData->mm1Tiling.stepKb = 1;
    tilingData->mm1Tiling.depthAL1CacheUB = 0;
    tilingData->mm1Tiling.depthBL1CacheUB = 0;
    tilingData->mm1Tiling.dbL0A = 2;
    tilingData->mm1Tiling.dbL0B = 2;
    tilingData->mm1Tiling.dbL0C = 1;

    tilingData->mm2Tiling.usedCoreNum = 28;
    tilingData->mm2Tiling.M = 128;
    tilingData->mm2Tiling.N = 8;
    tilingData->mm2Tiling.Ka = 8;
    tilingData->mm2Tiling.Kb = 8;
    tilingData->mm2Tiling.singleCoreM = 128;
    tilingData->mm2Tiling.singleCoreN = 8;
    tilingData->mm2Tiling.singleCoreK = 8;
    tilingData->mm2Tiling.baseM = 128;
    tilingData->mm2Tiling.baseN = 16;
    tilingData->mm2Tiling.baseK = 16;
    tilingData->mm2Tiling.depthA1 = 1;
    tilingData->mm2Tiling.depthB1 = 1;
    tilingData->mm2Tiling.stepM = 1;
    tilingData->mm2Tiling.stepN = 1;
    tilingData->mm2Tiling.isBias = 0;
    tilingData->mm2Tiling.transLength = 0;
    tilingData->mm2Tiling.iterateOrder = 0;
    tilingData->mm2Tiling.shareMode = 0;
    tilingData->mm2Tiling.shareL1Size = 4608;
    tilingData->mm2Tiling.shareL0CSize = 8192;
    tilingData->mm2Tiling.shareUbSize = 0;
    tilingData->mm2Tiling.batchM = 1;
    tilingData->mm2Tiling.batchN = 1;
    tilingData->mm2Tiling.singleBatchM = 1;
    tilingData->mm2Tiling.singleBatchN = 1;
    tilingData->mm2Tiling.stepKa = 1;
    tilingData->mm2Tiling.stepKb = 1;
    tilingData->mm2Tiling.depthAL1CacheUB = 0;
    tilingData->mm2Tiling.depthBL1CacheUB = 0;
    tilingData->mm2Tiling.dbL0A = 2;
    tilingData->mm2Tiling.dbL0B = 2;
    tilingData->mm2Tiling.dbL0C = 1;

    AscendC::SetKernelMode(KernelMode::MIX_MODE);
    ICPU_SET_TILING_KEY(101UL);
    ICPU_RUN_KF(fused_linear_cross_entropy_loss_grad, 28, grad_output, input, weight, target_mask, masked_target,
                logits_max, sum_exp_logits, softmax, grad_input, grad_weight, workspace, (uint8_t*)(tilingData));

    AscendC::GmFree(grad_output);
    AscendC::GmFree(input);
    AscendC::GmFree(weight);
    AscendC::GmFree(target_mask);
    AscendC::GmFree(masked_target);
    AscendC::GmFree(logits_max);
    AscendC::GmFree(sum_exp_logits);
    AscendC::GmFree(grad_input);
    AscendC::GmFree(grad_weight);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
