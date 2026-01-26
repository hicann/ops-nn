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
 * \file test_cross_entropy_loss.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "cross_entropy_loss_tiling_def.h"
#include "data_utils.h"

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void cross_entropy_loss(
    GM_ADDR input, GM_ADDR target, GM_ADDR weight, GM_ADDR loss, GM_ADDR log_prob, GM_ADDR zloss, GM_ADDR lse_for_zloss,
    GM_ADDR workspace, GM_ADDR tiling);

class cross_entropy_loss_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "cross_entropy_loss_test SetUp\n" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "cross_entropy_loss_test TearDown\n" << std::endl;
    }
};

TEST_F(cross_entropy_loss_test, test_case_bf16_mean)
{
    system(
        "cp -rf "
        "../../../../loss/cross_entropy_loss/tests/ut/op_kernel/cross_entropy_loss_data ./");
    system("chmod -R 755 ./cross_entropy_loss_data/");
    system("cd ./cross_entropy_loss_data/ && python3 gen_data.py 'bf16' '48' '4096' ");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t inputSize = 48 * 4096 * sizeof(half);
    size_t targetSize = 1 * 48 * sizeof(int64_t);
    size_t weightSize = 1 * 4096 * sizeof(float);
    size_t logProbOutSize = 48 * 4096 * sizeof(half);
    size_t lossOutSize = 1 * 1 * sizeof(half);
    size_t tilingSize = sizeof(CrossEntropyLossTilingData);

    uint8_t* input = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* target = (uint8_t*)AscendC::GmAlloc(targetSize);
    uint8_t* weight = (uint8_t*)AscendC::GmAlloc(weightSize);
    uint8_t* loss = (uint8_t*)AscendC::GmAlloc(lossOutSize);
    uint8_t* log_prob = (uint8_t*)AscendC::GmAlloc(logProbOutSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    uint32_t blockDim = 48;

    CrossEntropyLossTilingData* tilingData = reinterpret_cast<CrossEntropyLossTilingData*>(tiling);
    tilingData->targetNum = 4096;
    tilingData->frontCoreNum = 48;
    tilingData->frontBatchNum = 1;
    tilingData->tailCoreNum = 0;
    tilingData->tailBatchNum = 0;
    tilingData->inputUbSize = 4096;
    tilingData->castTmpBufByte = 16384;
    tilingData->lnTmpBufSize = 8;
    tilingData->weightTmpBufSize = 8;
    tilingData->totalTmpBufByte = 28800;
    tilingData->ubLoopNum = 0;
    tilingData->ubTailNum = 4096;
    tilingData->vecLoopNum = 0;
    tilingData->vecTailNum = 4096;
    tilingData->tailVecLoopNum = 0;
    tilingData->tailVecTailNum = 4096;
    tilingData->reduction = 1;
    tilingData->ignoreIndex = -100;
    tilingData->labelSmoothing = 0.0;
    tilingData->defaultWeight = 0;

    ReadFile("./cross_entropy_loss_data/input.bin", inputSize, input, inputSize);
    ReadFile("./cross_entropy_loss_data/target.bin", targetSize, target, targetSize);
    ReadFile("./cross_entropy_loss_data/weight.bin", weightSize, weight, weightSize);

    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(
        cross_entropy_loss, blockDim, input, target, weight, loss, log_prob, loss, loss, workspace,
        tiling); // zloss和lseForZloss暂不支持_f用l_zss占位

    AscendC::GmFree(input);
    AscendC::GmFree(target);
    AscendC::GmFree(weight);
    AscendC::GmFree(loss);
    AscendC::GmFree(log_prob);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(cross_entropy_loss_test, test_case_fp32_mean)
{
    system(
        "cp -rf "
        "../../../../loss/cross_entropy_loss/tests/ut/op_kernel/cross_entropy_loss_data ./");
    system("chmod -R 755 ./cross_entropy_loss_data/");
    system("cd ./cross_entropy_loss_data/ && python3 gen_data.py 'fp32' '48' '4096' ");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t inputSize = 48 * 4096 * sizeof(float);
    size_t targetSize = 1 * 48 * sizeof(int64_t);
    size_t weightSize = 1 * 4096 * sizeof(float);
    size_t logProbOutSize = 48 * 4096 * sizeof(float);
    size_t lossOutSize = 1 * 1 * sizeof(float);
    size_t tilingSize = sizeof(CrossEntropyLossTilingData);

    uint8_t* input = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* target = (uint8_t*)AscendC::GmAlloc(targetSize);
    uint8_t* weight = (uint8_t*)AscendC::GmAlloc(weightSize);
    uint8_t* loss = (uint8_t*)AscendC::GmAlloc(lossOutSize);
    uint8_t* log_prob = (uint8_t*)AscendC::GmAlloc(logProbOutSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    uint32_t blockDim = 48;

    CrossEntropyLossTilingData* tilingData = reinterpret_cast<CrossEntropyLossTilingData*>(tiling);
    tilingData->targetNum = 4096;
    tilingData->frontCoreNum = 48;
    tilingData->frontBatchNum = 1;
    tilingData->tailCoreNum = 0;
    tilingData->tailBatchNum = 0;
    tilingData->inputUbSize = 4096;
    tilingData->castTmpBufByte = 32;
    tilingData->lnTmpBufSize = 8;
    tilingData->weightTmpBufSize = 8;
    tilingData->totalTmpBufByte = 4224+32;
    tilingData->ubLoopNum = 0;
    tilingData->ubTailNum = 4096;
    tilingData->vecLoopNum = 0;
    tilingData->vecTailNum = 4096;
    tilingData->tailVecLoopNum = 0;
    tilingData->tailVecTailNum = 4096;
    tilingData->reduction = 1;
    tilingData->ignoreIndex = -100;
    tilingData->labelSmoothing = 0.0;
    tilingData->defaultWeight = 0;

    ReadFile("./cross_entropy_loss_data/input.bin", inputSize, input, inputSize);
    ReadFile("./cross_entropy_loss_data/target.bin", targetSize, target, targetSize);
    ReadFile("./cross_entropy_loss_data/weight.bin", weightSize, weight, weightSize);

    ICPU_SET_TILING_KEY(3);
    ICPU_RUN_KF(
        cross_entropy_loss, blockDim, input, target, weight, loss, log_prob, loss, loss, workspace,
        tiling); // zloss和lseForZloss暂不支持

    AscendC::GmFree(input);
    AscendC::GmFree(target);
    AscendC::GmFree(weight);
    AscendC::GmFree(loss);
    AscendC::GmFree(log_prob);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(cross_entropy_loss_test, test_case_fp16_mean)
{
    system(
        "cp -rf "
        "../../../../loss/cross_entropy_loss/tests/ut/op_kernel/cross_entropy_loss_data ./");
    system("chmod -R 755 ./cross_entropy_loss_data/");
    system("cd ./cross_entropy_loss_data/ && python3 gen_data.py 'fp16' '48' '4096' ");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t inputSize = 48 * 4096 * sizeof(half);
    size_t targetSize = 1 * 48 * sizeof(int64_t);
    size_t weightSize = 1 * 4096 * sizeof(float);
    size_t logProbOutSize = 48 * 4096 * sizeof(half);
    size_t lossOutSize = 1 * 1 * sizeof(half);
    size_t tilingSize = sizeof(CrossEntropyLossTilingData);

    uint8_t* input = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* target = (uint8_t*)AscendC::GmAlloc(targetSize);
    uint8_t* weight = (uint8_t*)AscendC::GmAlloc(weightSize);
    uint8_t* loss = (uint8_t*)AscendC::GmAlloc(lossOutSize);
    uint8_t* log_prob = (uint8_t*)AscendC::GmAlloc(logProbOutSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    uint32_t blockDim = 48;

    CrossEntropyLossTilingData* tilingData = reinterpret_cast<CrossEntropyLossTilingData*>(tiling);
    tilingData->targetNum = 4096;
    tilingData->frontCoreNum = 48;
    tilingData->frontBatchNum = 1;
    tilingData->tailCoreNum = 0;
    tilingData->tailBatchNum = 0;
    tilingData->inputUbSize = 4096;
    tilingData->castTmpBufByte = 16384;
    tilingData->lnTmpBufSize = 8;
    tilingData->weightTmpBufSize = 8;
    tilingData->totalTmpBufByte = 28800;
    tilingData->ubLoopNum = 0;
    tilingData->ubTailNum = 4096;
    tilingData->vecLoopNum = 0;
    tilingData->vecTailNum = 4096;
    tilingData->tailVecLoopNum = 0;
    tilingData->tailVecTailNum = 4096;
    tilingData->reduction = 1;
    tilingData->ignoreIndex = -100;
    tilingData->labelSmoothing = 0.0;
    tilingData->defaultWeight = 0;
    tilingData->defaultWeight = 0;

    ReadFile("./cross_entropy_loss_data/input.bin", inputSize, input, inputSize);
    ReadFile("./cross_entropy_loss_data/target.bin", targetSize, target, targetSize);
    ReadFile("./cross_entropy_loss_data/weight.bin", weightSize, weight, weightSize);

    ICPU_SET_TILING_KEY(2);
    ICPU_RUN_KF(
        cross_entropy_loss, blockDim, input, target, weight, loss, log_prob, loss, loss, workspace,
        tiling); // zloss和lseForZloss暂不支持_f用l_zss占位

    AscendC::GmFree(input);
    AscendC::GmFree(target);
    AscendC::GmFree(weight);
    AscendC::GmFree(loss);
    AscendC::GmFree(log_prob);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(cross_entropy_loss_test, test_case_fp16_mean_ignore_index)
{
    system(
        "cp -rf "
        "../../../../loss/cross_entropy_loss/tests/ut/op_kernel/cross_entropy_loss_data ./");
    system("chmod -R 755 ./cross_entropy_loss_data/");
    system("cd ./cross_entropy_loss_data/ && python3 gen_data.py 'fp16' '48' '4096' ");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t inputSize = 48 * 4096 * sizeof(half);
    size_t targetSize = 1 * 48 * sizeof(int64_t);
    size_t weightSize = 1 * 4096 * sizeof(float);
    size_t logProbOutSize = 48 * 4096 * sizeof(half);
    size_t lossOutSize = 1 * 1 * sizeof(half);
    size_t tilingSize = sizeof(CrossEntropyLossTilingData);

    uint8_t* input = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* target = (uint8_t*)AscendC::GmAlloc(targetSize);
    uint8_t* weight = (uint8_t*)AscendC::GmAlloc(weightSize);
    uint8_t* loss = (uint8_t*)AscendC::GmAlloc(lossOutSize);
    uint8_t* log_prob = (uint8_t*)AscendC::GmAlloc(logProbOutSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    uint32_t blockDim = 48;

    CrossEntropyLossTilingData* tilingData = reinterpret_cast<CrossEntropyLossTilingData*>(tiling);
    tilingData->targetNum = 4096;
    tilingData->frontCoreNum = 48;
    tilingData->frontBatchNum = 1;
    tilingData->tailCoreNum = 0;
    tilingData->tailBatchNum = 0;
    tilingData->inputUbSize = 4096;
    tilingData->castTmpBufByte = 16384;
    tilingData->lnTmpBufSize = 8;
    tilingData->weightTmpBufSize = 8;
    tilingData->totalTmpBufByte = 28800;
    tilingData->ubLoopNum = 0;
    tilingData->ubTailNum = 4096;
    tilingData->vecLoopNum = 0;
    tilingData->vecTailNum = 4096;
    tilingData->tailVecLoopNum = 0;
    tilingData->tailVecTailNum = 4096;
    tilingData->reduction = 1;
    tilingData->ignoreIndex = 1;
    tilingData->labelSmoothing = 0.0;
    tilingData->defaultWeight = 0;
    tilingData->defaultWeight = 0;

    ReadFile("./cross_entropy_loss_data/input.bin", inputSize, input, inputSize);
    ReadFile("./cross_entropy_loss_data/target.bin", targetSize, target, targetSize);
    ReadFile("./cross_entropy_loss_data/weight.bin", weightSize, weight, weightSize);

    ICPU_SET_TILING_KEY(2);
    ICPU_RUN_KF(
        cross_entropy_loss, blockDim, input, target, weight, loss, log_prob, loss, loss, workspace,
        tiling); // zloss和lseForZloss暂不支持_f用l_zss占位

    AscendC::GmFree(input);
    AscendC::GmFree(target);
    AscendC::GmFree(weight);
    AscendC::GmFree(loss);
    AscendC::GmFree(log_prob);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(cross_entropy_loss_test, test_case_fp32_mean_ignore_index)
{
    system(
        "cp -rf "
        "../../../../loss/cross_entropy_loss/tests/ut/op_kernel/cross_entropy_loss_data ./");
    system("chmod -R 755 ./cross_entropy_loss_data/");
    system("cd ./cross_entropy_loss_data/ && python3 gen_data.py 'fp32' '48' '4096' ");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t inputSize = 48 * 4096 * sizeof(float);
    size_t targetSize = 1 * 48 * sizeof(int64_t);
    size_t weightSize = 1 * 4096 * sizeof(float);
    size_t logProbOutSize = 48 * 4096 * sizeof(float);
    size_t lossOutSize = 1 * 1 * sizeof(float);
    size_t tilingSize = sizeof(CrossEntropyLossTilingData);

    uint8_t* input = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* target = (uint8_t*)AscendC::GmAlloc(targetSize);
    uint8_t* weight = (uint8_t*)AscendC::GmAlloc(weightSize);
    uint8_t* loss = (uint8_t*)AscendC::GmAlloc(lossOutSize);
    uint8_t* log_prob = (uint8_t*)AscendC::GmAlloc(logProbOutSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    uint32_t blockDim = 48;

    CrossEntropyLossTilingData* tilingData = reinterpret_cast<CrossEntropyLossTilingData*>(tiling);
    tilingData->targetNum = 4096;
    tilingData->frontCoreNum = 48;
    tilingData->frontBatchNum = 1;
    tilingData->tailCoreNum = 0;
    tilingData->tailBatchNum = 0;
    tilingData->inputUbSize = 4096;
    tilingData->castTmpBufByte = 32;
    tilingData->lnTmpBufSize = 8;
    tilingData->weightTmpBufSize = 8;
    tilingData->totalTmpBufByte = 4224+32;
    tilingData->ubLoopNum = 0;
    tilingData->ubTailNum = 4096;
    tilingData->vecLoopNum = 0;
    tilingData->vecTailNum = 4096;
    tilingData->tailVecLoopNum = 0;
    tilingData->tailVecTailNum = 4096;
    tilingData->reduction = 1;
    tilingData->ignoreIndex = 1;
    tilingData->labelSmoothing = 0.0;
    tilingData->defaultWeight = 0;

    ReadFile("./cross_entropy_loss_data/input.bin", inputSize, input, inputSize);
    ReadFile("./cross_entropy_loss_data/target.bin", targetSize, target, targetSize);
    ReadFile("./cross_entropy_loss_data/weight.bin", weightSize, weight, weightSize);

    ICPU_SET_TILING_KEY(3);
    ICPU_RUN_KF(
        cross_entropy_loss, blockDim, input, target, weight, loss, log_prob, loss, loss, workspace,
        tiling); // zloss和lseForZloss暂不支持

    AscendC::GmFree(input);
    AscendC::GmFree(target);
    AscendC::GmFree(weight);
    AscendC::GmFree(loss);
    AscendC::GmFree(log_prob);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(cross_entropy_loss_test, test_case_bf16_mean_no_weight)
{
    system(
        "cp -rf "
        "../../../../loss/cross_entropy_loss/tests/ut/op_kernel/cross_entropy_loss_data ./");
    system("chmod -R 755 ./cross_entropy_loss_data/");
    system("cd ./cross_entropy_loss_data/ && python3 gen_data.py 'bf16' '48' '4096' ");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t inputSize = 48 * 4096 * sizeof(half);
    size_t targetSize = 1 * 48 * sizeof(int64_t);
    size_t weightSize = 1 * 4096 * sizeof(float);
    size_t logProbOutSize = 48 * 4096 * sizeof(half);
    size_t lossOutSize = 1 * 1 * sizeof(half);
    size_t tilingSize = sizeof(CrossEntropyLossTilingData);

    uint8_t* input = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* target = (uint8_t*)AscendC::GmAlloc(targetSize);
    uint8_t* weight = (uint8_t*)AscendC::GmAlloc(weightSize);
    uint8_t* loss = (uint8_t*)AscendC::GmAlloc(lossOutSize);
    uint8_t* log_prob = (uint8_t*)AscendC::GmAlloc(logProbOutSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    uint32_t blockDim = 48;

    CrossEntropyLossTilingData* tilingData = reinterpret_cast<CrossEntropyLossTilingData*>(tiling);
    tilingData->targetNum = 4096;
    tilingData->frontCoreNum = 48;
    tilingData->frontBatchNum = 1;
    tilingData->tailCoreNum = 0;
    tilingData->tailBatchNum = 0;
    tilingData->inputUbSize = 4096;
    tilingData->castTmpBufByte = 16384;
    tilingData->lnTmpBufSize = 8;
    tilingData->weightTmpBufSize = 8;
    tilingData->totalTmpBufByte = 28800;
    tilingData->ubLoopNum = 0;
    tilingData->ubTailNum = 4096;
    tilingData->vecLoopNum = 0;
    tilingData->vecTailNum = 4096;
    tilingData->tailVecLoopNum = 0;
    tilingData->tailVecTailNum = 4096;
    tilingData->reduction = 1;
    tilingData->ignoreIndex = -100;
    tilingData->labelSmoothing = 0.0;
    tilingData->defaultWeight = 1;

    ReadFile("./cross_entropy_loss_data/input.bin", inputSize, input, inputSize);
    ReadFile("./cross_entropy_loss_data/target.bin", targetSize, target, targetSize);

    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(
        cross_entropy_loss, blockDim, input, target, weight, loss, log_prob, loss, loss, workspace,
        tiling); // zloss和lseForZloss暂不支持_f用l_zss占位

    AscendC::GmFree(input);
    AscendC::GmFree(target);
    AscendC::GmFree(weight);
    AscendC::GmFree(loss);
    AscendC::GmFree(log_prob);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(cross_entropy_loss_test, test_case_fp32_mean_no_weight)
{
    system(
        "cp -rf "
        "../../../../loss/cross_entropy_loss/tests/ut/op_kernel/cross_entropy_loss_data ./");
    system("chmod -R 755 ./cross_entropy_loss_data/");
    system("cd ./cross_entropy_loss_data/ && python3 gen_data.py 'fp32' '48' '4096' ");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t inputSize = 48 * 4096 * sizeof(float);
    size_t targetSize = 1 * 48 * sizeof(int64_t);
    size_t weightSize = 1 * 4096 * sizeof(float);
    size_t logProbOutSize = 48 * 4096 * sizeof(float);
    size_t lossOutSize = 1 * 1 * sizeof(float);
    size_t tilingSize = sizeof(CrossEntropyLossTilingData);

    uint8_t* input = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* target = (uint8_t*)AscendC::GmAlloc(targetSize);
    uint8_t* weight = (uint8_t*)AscendC::GmAlloc(weightSize);
    uint8_t* loss = (uint8_t*)AscendC::GmAlloc(lossOutSize);
    uint8_t* log_prob = (uint8_t*)AscendC::GmAlloc(logProbOutSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    uint32_t blockDim = 48;

    CrossEntropyLossTilingData* tilingData = reinterpret_cast<CrossEntropyLossTilingData*>(tiling);
    tilingData->targetNum = 4096;
    tilingData->frontCoreNum = 48;
    tilingData->frontBatchNum = 1;
    tilingData->tailCoreNum = 0;
    tilingData->tailBatchNum = 0;
    tilingData->inputUbSize = 4096;
    tilingData->castTmpBufByte = 32;
    tilingData->lnTmpBufSize = 8;
    tilingData->weightTmpBufSize = 8;
    tilingData->totalTmpBufByte = 4224+32;
    tilingData->ubLoopNum = 0;
    tilingData->ubTailNum = 4096;
    tilingData->vecLoopNum = 0;
    tilingData->vecTailNum = 4096;
    tilingData->tailVecLoopNum = 0;
    tilingData->tailVecTailNum = 4096;
    tilingData->reduction = 1;
    tilingData->ignoreIndex = -100;
    tilingData->labelSmoothing = 0.0;
    tilingData->defaultWeight = 1;

    ReadFile("./cross_entropy_loss_data/input.bin", inputSize, input, inputSize);
    ReadFile("./cross_entropy_loss_data/target.bin", targetSize, target, targetSize);

    ICPU_SET_TILING_KEY(3);
    ICPU_RUN_KF(
        cross_entropy_loss, blockDim, input, target, weight, loss, log_prob, loss, loss, workspace,
        tiling); // zloss和lseForZloss暂不支持

    AscendC::GmFree(input);
    AscendC::GmFree(target);
    AscendC::GmFree(weight);
    AscendC::GmFree(loss);
    AscendC::GmFree(log_prob);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(cross_entropy_loss_test, test_case_fp16_mean_no_weight_big_shape)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t inputSize = 2 * 40960 * sizeof(half);
    size_t targetSize =  2 * sizeof(int64_t);
    size_t weightSize = 0 * sizeof(float);
    size_t logProbOutSize = 2 * 40960 * sizeof(half);
    size_t lossOutSize = 1 * sizeof(half);
    size_t tilingSize = sizeof(CrossEntropyLossTilingData);

    uint8_t* input = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* target = (uint8_t*)AscendC::GmAlloc(targetSize);
    uint8_t* weight = (uint8_t*)AscendC::GmAlloc(weightSize);
    uint8_t* loss = (uint8_t*)AscendC::GmAlloc(lossOutSize);
    uint8_t* log_prob = (uint8_t*)AscendC::GmAlloc(logProbOutSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    uint32_t blockDim = 2;

    CrossEntropyLossTilingData* tilingData = reinterpret_cast<CrossEntropyLossTilingData*>(tiling);
    tilingData->targetNum = 40960;
    tilingData->frontCoreNum = 2;
    tilingData->tailCoreNum = 0;
    tilingData->frontBatchNum = 1;
    tilingData->tailBatchNum = 0;
    tilingData->inputUbSize = 23808;
    tilingData->castTmpBufByte = 95232;
    tilingData->lnTmpBufSize = 8;
    tilingData->weightTmpBufSize = 8;
    tilingData->totalTmpBufByte = 147104;
    tilingData->ubLoopNum = 1;
    tilingData->ubTailNum = 17152;
    tilingData->vecLoopNum = 2;
    tilingData->vecTailNum = 7424;
    tilingData->tailVecLoopNum = 2;
    tilingData->tailVecTailNum = 768;
    tilingData->reduction = 1;
    tilingData->ignoreIndex = 0;
    tilingData->labelSmoothing = 0.0;
    tilingData->defaultWeight = 1;

    ICPU_SET_TILING_KEY(2);
    ICPU_RUN_KF(
        cross_entropy_loss, blockDim, input, target, weight, loss, log_prob, loss, loss, workspace,
        tiling); // zloss和lseForZloss暂不支持

    AscendC::GmFree(input);
    AscendC::GmFree(target);
    AscendC::GmFree(weight);
    AscendC::GmFree(loss);
    AscendC::GmFree(log_prob);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(cross_entropy_loss_test, test_case_fp32_mean_no_weight_big_shape)
{
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t inputSize = 2 * 81920 * sizeof(float);
    size_t targetSize =  2 * sizeof(int64_t);
    size_t weightSize = 0 * sizeof(float);
    size_t logProbOutSize = 2 * 81920 * sizeof(float);
    size_t lossOutSize = 1 * sizeof(float);
    size_t tilingSize = sizeof(CrossEntropyLossTilingData);

    uint8_t* input = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* target = (uint8_t*)AscendC::GmAlloc(targetSize);
    uint8_t* weight = (uint8_t*)AscendC::GmAlloc(weightSize);
    uint8_t* loss = (uint8_t*)AscendC::GmAlloc(lossOutSize);
    uint8_t* log_prob = (uint8_t*)AscendC::GmAlloc(logProbOutSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    uint32_t blockDim = 2;

    CrossEntropyLossTilingData* tilingData = reinterpret_cast<CrossEntropyLossTilingData*>(tiling);
    tilingData->targetNum = 81920;
    tilingData->frontCoreNum = 2;
    tilingData->tailCoreNum = 0;
    tilingData->frontBatchNum = 1;
    tilingData->tailBatchNum = 0;
    tilingData->inputUbSize = 48000;
    tilingData->castTmpBufByte = 32;
    tilingData->lnTmpBufSize = 8;
    tilingData->weightTmpBufSize = 8;
    tilingData->totalTmpBufByte = 4256;
    tilingData->ubLoopNum = 1;
    tilingData->ubTailNum = 33920;
    tilingData->vecLoopNum = 5;
    tilingData->vecTailNum = 7040;
    tilingData->tailVecLoopNum = 4;
    tilingData->tailVecTailNum = 1152;
    tilingData->reduction = 1;
    tilingData->ignoreIndex = 0;
    tilingData->labelSmoothing = 0.0;
    tilingData->defaultWeight = 1;

    ICPU_SET_TILING_KEY(3);
    ICPU_RUN_KF(
        cross_entropy_loss, blockDim, input, target, weight, loss, log_prob, loss, loss, workspace,
        tiling); // zloss和lseForZloss暂不支持

    AscendC::GmFree(input);
    AscendC::GmFree(target);
    AscendC::GmFree(weight);
    AscendC::GmFree(loss);
    AscendC::GmFree(log_prob);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}