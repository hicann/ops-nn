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
 * \file test_apply_adam_w_v2.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "apply_adam_w_v2_tiling_def.h"
#include "data_utils.h"

using namespace std;

extern "C" __global__ __aicore__ void apply_adam_w_v2(
    GM_ADDR var, GM_ADDR expAvg, GM_ADDR expAvgSq, GM_ADDR grad, GM_ADDR step, GM_ADDR maxGradNorm, GM_ADDR workspace,
    GM_ADDR tiling);

class apply_adam_w_v2_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "apply_adam_w_v2_test SetUp\n" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "apply_adam_w_v2_test TearDown\n" << std::endl;
    }
};

int64_t GetShapeSize(const std::vector<int64_t>& shape)
{
    int64_t shapeSize = 1;
    for (auto i : shape) {
        shapeSize *= i;
    }
    return shapeSize;
}

TEST_F(apply_adam_w_v2_test, test_case_float32_amsgrad_true_maximize_true)
{
    system(
        "cp -rf "
        "../../../../optim/apply_adam_w_v2/tests/ut/op_kernel/apply_adam_w_v2_data ./");
    system("chmod -R 755 ./apply_adam_w_v2_data/");
    system("cd ./apply_adam_w_v2_data/ && python3 gen_data.py 'float32' 'true' 'true' ");
    system("ls -l apply_adam_w_v2_data");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024 + 48 * sizeof(int32_t);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(ApplyAdamWV2TilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    std::vector<int64_t> shape = {30, 4, 2};
    size_t bytes = 4;
    size_t inputSize = GetShapeSize(shape) * bytes;
    size_t outputSize = GetShapeSize(shape) * bytes;
    size_t workspaceBytesSize = 16 * 1024 * 1024 + 48 * sizeof(int32_t);
    uint32_t blockDim = 32;

    uint8_t* var = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* m = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* v = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* maxGradNorm = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* step = (uint8_t*)AscendC::GmAlloc(bytes);

    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(workspaceBytesSize);
    int64_t totalCoreNum = 32;
    int64_t numPerLoop = 208;
    int64_t numLastLoop = 32;
    float lr = 0.01;
    float beta1 = 0.99;
    float beta2 = 0.1;
    float weightDecay = 0.005;
    float eps = 0.000001;
    int64_t tilingKey105 = 105;
    ApplyAdamWV2TilingData* tilingData = reinterpret_cast<ApplyAdamWV2TilingData*>(tiling);

    tilingData->totalCoreNum = totalCoreNum;
    tilingData->handleExtraLoopCoreNum = 1;
    tilingData->usedCoreNum = 1;
    tilingData->numPerLoop = numPerLoop;
    tilingData->loopNumPerCore = 1;
    tilingData->numLastLoop = numLastLoop;
    tilingData->lr = lr;
    tilingData->beta1 = beta1;
    tilingData->beta2 = beta2;
    tilingData->weightDecay = weightDecay;
    tilingData->eps = eps;
    tilingData->amsgrad = true;
    tilingData->maximize = true;
    tilingData->isBfloat16 = false;

    std::string curPath = ".";
    ReadFile(curPath + "/apply_adam_w_v2_data/var.bin", inputSize, var, inputSize);
    ReadFile(curPath + "/apply_adam_w_v2_data/m.bin", inputSize, m, inputSize);
    ReadFile(curPath + "/apply_adam_w_v2_data/v.bin", inputSize, v, inputSize);
    ReadFile(curPath + "/apply_adam_w_v2_data/max_grad_norm.bin", inputSize, maxGradNorm, inputSize);
    ReadFile(curPath + "/apply_adam_w_v2_data/grad.bin", inputSize, grad, inputSize);
    ReadFile(curPath + "/apply_adam_w_v2_data/step.bin", bytes, step, bytes);

    ICPU_SET_TILING_KEY(tilingKey105);
    ICPU_RUN_KF(apply_adam_w_v2, blockDim, var, m, v, grad, step, maxGradNorm, workSpace, tiling);

    WriteFile(curPath + "/apply_adam_w_v2_data/out_var.bin", var, outputSize);
    WriteFile(curPath + "/apply_adam_w_v2_data/out_max_grad_norm.bin", maxGradNorm, outputSize);
    WriteFile(curPath + "/apply_adam_w_v2_data/out_m.bin", m, outputSize);
    WriteFile(curPath + "/apply_adam_w_v2_data/out_v.bin", v, outputSize);
    AscendC::GmFree((void*)var);
    AscendC::GmFree((void*)grad);
    AscendC::GmFree((void*)maxGradNorm);
    AscendC::GmFree((void*)m);
    AscendC::GmFree((void*)v);
    AscendC::GmFree((void*)step);
}

TEST_F(apply_adam_w_v2_test, test_case_float32_amsgrad_false_maximize_false)
{
    system(
        "cp -rf "
        "../../../../optim/apply_adam_w_v2/tests/ut/op_kernel/apply_adam_w_v2_data ./");
    system("chmod -R 755 ./apply_adam_w_v2_data/");
    system("cd ./apply_adam_w_v2_data/ && python3 gen_data.py 'float32' 'false' 'false' 'float32' 'int64' ");
    system("ls -l apply_adam_w_v2_data");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024 + 48 * sizeof(int32_t);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(ApplyAdamWV2TilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    std::vector<int64_t> shape = {30, 4, 2};
    size_t bytes = 4;
    size_t inputSize = GetShapeSize(shape) * bytes;
    size_t outputSize = GetShapeSize(shape) * bytes;
    size_t workspaceBytesSize = 16 * 1024 * 1024 + 48 * sizeof(int32_t);
    uint32_t blockDim = 32;
    size_t stepSize = 8;

    uint8_t* var = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* m = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* v = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* maxGradNorm = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* step = (uint8_t*)AscendC::GmAlloc(stepSize);

    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(workspaceBytesSize);
    int64_t totalCoreNum = 32;
    int64_t numPerLoop = 208;
    int64_t numLastLoop = 32;
    float lr = 0.01;
    float beta1 = 0.99;
    float beta2 = 0.1;
    float weightDecay = 0.005;
    float eps = 0.000001;
    int64_t tilingKey106 = 106;
    ApplyAdamWV2TilingData* tilingData = reinterpret_cast<ApplyAdamWV2TilingData*>(tiling);
    tilingData->totalCoreNum = totalCoreNum;
    tilingData->handleExtraLoopCoreNum = 1;
    tilingData->usedCoreNum = 1;
    tilingData->numPerLoop = numPerLoop;
    tilingData->loopNumPerCore = 1;
    tilingData->numLastLoop = numLastLoop;
    tilingData->lr = lr;
    tilingData->beta1 = beta1;
    tilingData->beta2 = beta2;
    tilingData->weightDecay = weightDecay;
    tilingData->eps = eps;
    tilingData->amsgrad = false;
    tilingData->maximize = true;
    tilingData->isBfloat16 = false;

    std::string curPath = ".";
    ReadFile(curPath + "/apply_adam_w_v2_data/var.bin", inputSize, var, inputSize);
    ReadFile(curPath + "/apply_adam_w_v2_data/m.bin", inputSize, m, inputSize);
    ReadFile(curPath + "/apply_adam_w_v2_data/v.bin", inputSize, v, inputSize);
    ReadFile(curPath + "/apply_adam_w_v2_data/max_grad_norm.bin", inputSize, maxGradNorm, inputSize);
    ReadFile(curPath + "/apply_adam_w_v2_data/grad.bin", inputSize, grad, inputSize);
    ReadFile(curPath + "/apply_adam_w_v2_data/step.bin", stepSize, step, stepSize);

    ICPU_SET_TILING_KEY(tilingKey106);
    ICPU_RUN_KF(apply_adam_w_v2, blockDim, var, m, v, grad, step, maxGradNorm, workSpace, tiling);

    WriteFile(curPath + "/apply_adam_w_v2_data/out_var.bin", var, outputSize);
    WriteFile(curPath + "/apply_adam_w_v2_data/out_max_grad_norm.bin", maxGradNorm, outputSize);
    WriteFile(curPath + "/apply_adam_w_v2_data/out_m.bin", m, outputSize);
    WriteFile(curPath + "/apply_adam_w_v2_data/out_v.bin", v, outputSize);
    AscendC::GmFree((void*)var);
    AscendC::GmFree((void*)grad);
    AscendC::GmFree((void*)maxGradNorm);
    AscendC::GmFree((void*)m);
    AscendC::GmFree((void*)v);
    AscendC::GmFree((void*)step);
}

TEST_F(apply_adam_w_v2_test, test_case_float16_amsgrad_true_maximize_true)
{
    system(
        "cp -rf "
        "../../../../optim/apply_adam_w_v2/tests/ut/op_kernel/apply_adam_w_v2_data ./");
    system("chmod -R 755 ./apply_adam_w_v2_data/");
    system("cd ./apply_adam_w_v2_data/ && python3 gen_data.py 'float16' 'true' 'true' ");
    system("ls -l apply_adam_w_v2_data");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024 + 48 * sizeof(int32_t);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(ApplyAdamWV2TilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    std::vector<int64_t> shape = {30, 4, 2};
    size_t bytes = 2;
    size_t inputSize = GetShapeSize(shape) * bytes;
    size_t outputSize = GetShapeSize(shape) * bytes;
    size_t stepSize = 4;
    size_t workspaceBytesSize = 16 * 1024 * 1024 + 48 * sizeof(int32_t);
    uint32_t blockDim = 32;

    uint8_t* var = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* m = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* v = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* maxGradNorm = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* step = (uint8_t*)AscendC::GmAlloc(stepSize);

    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(workspaceBytesSize);
    int64_t totalCoreNum = 32;
    int64_t numPerLoop = 208;
    int64_t numLastLoop = 32;
    float lr = 0.01;
    float beta1 = 0.99;
    float beta2 = 0.1;
    float weightDecay = 0.005;
    float eps = 0.000001;
    int64_t tilingKey103 = 103;
    ApplyAdamWV2TilingData* tilingData = reinterpret_cast<ApplyAdamWV2TilingData*>(tiling);
    tilingData->totalCoreNum = totalCoreNum;
    tilingData->handleExtraLoopCoreNum = 1;
    tilingData->usedCoreNum = 1;
    tilingData->numPerLoop = numPerLoop;
    tilingData->loopNumPerCore = 1;
    tilingData->numLastLoop = numLastLoop;
    tilingData->lr = lr;
    tilingData->beta1 = beta1;
    tilingData->beta2 = beta2;
    tilingData->weightDecay = weightDecay;
    tilingData->eps = eps;
    tilingData->amsgrad = true;
    tilingData->maximize = true;
    tilingData->isBfloat16 = false;

    std::string curPath = ".";
    ReadFile(curPath + "/apply_adam_w_v2_data/var.bin", inputSize, var, inputSize);
    ReadFile(curPath + "/apply_adam_w_v2_data/m.bin", inputSize, m, inputSize);
    ReadFile(curPath + "/apply_adam_w_v2_data/v.bin", inputSize, v, inputSize);
    ReadFile(curPath + "/apply_adam_w_v2_data/max_grad_norm.bin", inputSize, maxGradNorm, inputSize);
    ReadFile(curPath + "/apply_adam_w_v2_data/grad.bin", inputSize, grad, inputSize);
    ReadFile(curPath + "/apply_adam_w_v2_data/step.bin", stepSize, step, stepSize);

    ICPU_SET_TILING_KEY(tilingKey103);
    ICPU_RUN_KF(apply_adam_w_v2, blockDim, var, m, v, grad, step, maxGradNorm, workSpace, tiling);

    WriteFile(curPath + "/apply_adam_w_v2_data/out_var.bin", var, outputSize);
    WriteFile(curPath + "/apply_adam_w_v2_data/out_max_grad_norm.bin", maxGradNorm, outputSize);
    WriteFile(curPath + "/apply_adam_w_v2_data/out_m.bin", m, outputSize);
    WriteFile(curPath + "/apply_adam_w_v2_data/out_v.bin", v, outputSize);
    AscendC::GmFree((void*)var);
    AscendC::GmFree((void*)grad);
    AscendC::GmFree((void*)maxGradNorm);
    AscendC::GmFree((void*)m);
    AscendC::GmFree((void*)v);
    AscendC::GmFree((void*)step);
}

TEST_F(apply_adam_w_v2_test, test_case_float16_amsgrad_false_maximize_true)
{
    system(
        "cp -rf "
        "../../../../optim/apply_adam_w_v2/tests/ut/op_kernel/apply_adam_w_v2_data ./");
    system("chmod -R 755 ./apply_adam_w_v2_data/");
    system("cd ./apply_adam_w_v2_data/ && python3 gen_data.py 'float16' 'false' 'true' 'float16' 'int64' ");
    system("ls -l apply_adam_w_v2_data");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024 + 48 * sizeof(int32_t);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(ApplyAdamWV2TilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    std::vector<int64_t> shape = {30, 4, 2};
    size_t bytes = 2;
    size_t stepSize = 8;
    size_t inputSize = GetShapeSize(shape) * bytes;
    size_t outputSize = GetShapeSize(shape) * bytes;
    size_t workspaceBytesSize = 16 * 1024 * 1024 + 48 * sizeof(int32_t);
    uint32_t blockDim = 32;

    uint8_t* var = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* m = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* v = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* maxGradNorm = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* step = (uint8_t*)AscendC::GmAlloc(stepSize);

    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(workspaceBytesSize);
    int64_t totalCoreNum = 32;
    int64_t numPerLoop = 208;
    int64_t numLastLoop = 32;
    float lr = 0.01;
    float beta1 = 0.99;
    float beta2 = 0.1;
    float weightDecay = 0.005;
    float eps = 0.000001;
    int64_t tilingKey104 = 104;
    ApplyAdamWV2TilingData* tilingData = reinterpret_cast<ApplyAdamWV2TilingData*>(tiling);
    tilingData->totalCoreNum = totalCoreNum;
    tilingData->handleExtraLoopCoreNum = 1;
    tilingData->usedCoreNum = 1;
    tilingData->numPerLoop = numPerLoop;
    tilingData->loopNumPerCore = 1;
    tilingData->numLastLoop = numLastLoop;
    tilingData->lr = lr;
    tilingData->beta1 = beta1;
    tilingData->beta2 = beta2;
    tilingData->weightDecay = weightDecay;
    tilingData->eps = eps;
    tilingData->amsgrad = false;
    tilingData->maximize = true;
    tilingData->isBfloat16 = false;

    std::string curPath = ".";
    ReadFile(curPath + "/apply_adam_w_v2_data/var.bin", inputSize, var, inputSize);
    ReadFile(curPath + "/apply_adam_w_v2_data/m.bin", inputSize, m, inputSize);
    ReadFile(curPath + "/apply_adam_w_v2_data/v.bin", inputSize, v, inputSize);
    ReadFile(curPath + "/apply_adam_w_v2_data/max_grad_norm.bin", inputSize, maxGradNorm, inputSize);
    ReadFile(curPath + "/apply_adam_w_v2_data/grad.bin", inputSize, grad, inputSize);
    ReadFile(curPath + "/apply_adam_w_v2_data/step.bin", stepSize, step, stepSize);

    ICPU_SET_TILING_KEY(tilingKey104);
    ICPU_RUN_KF(apply_adam_w_v2, blockDim, var, m, v, grad, step, maxGradNorm, workSpace, tiling);

    WriteFile(curPath + "/apply_adam_w_v2_data/out_var.bin", var, outputSize);
    WriteFile(curPath + "/apply_adam_w_v2_data/out_max_grad_norm.bin", maxGradNorm, outputSize);
    WriteFile(curPath + "/apply_adam_w_v2_data/out_m.bin", m, outputSize);
    WriteFile(curPath + "/apply_adam_w_v2_data/out_v.bin", v, outputSize);
    AscendC::GmFree((void*)var);
    AscendC::GmFree((void*)grad);
    AscendC::GmFree((void*)maxGradNorm);
    AscendC::GmFree((void*)m);
    AscendC::GmFree((void*)v);
    AscendC::GmFree((void*)step);
}

TEST_F(apply_adam_w_v2_test, test_case_bfloat16_amsgrad_true_maximize_true)
{
    system(
        "cp -rf "
        "../../../../optim/apply_adam_w_v2/tests/ut/op_kernel/apply_adam_w_v2_data ./");
    system("chmod -R 755 ./apply_adam_w_v2_data/");
    system("cd ./apply_adam_w_v2_data/ && python3 gen_data.py 'bfloat16' 'true' 'true' ");
    system("ls -l apply_adam_w_v2_data");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024 + 48 * sizeof(int32_t);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(ApplyAdamWV2TilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    std::vector<int64_t> shape = {30, 4, 2};
    size_t bytes = 2;
    size_t stepSize = 4;
    size_t inputSize = GetShapeSize(shape) * bytes;
    size_t outputSize = GetShapeSize(shape) * bytes;
    size_t workspaceBytesSize = 16 * 1024 * 1024 + 48 * sizeof(int32_t);
    uint32_t blockDim = 32;

    uint8_t* var = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* m = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* v = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* maxGradNorm = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* step = (uint8_t*)AscendC::GmAlloc(stepSize);

    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(workspaceBytesSize);
    int64_t totalCoreNum = 32;
    int64_t numPerLoop = 208;
    int64_t numLastLoop = 32;
    float lr = 0.01;
    float beta1 = 0.99;
    float beta2 = 0.1;
    float weightDecay = 0.005;
    float eps = 0.000001;
    int64_t tilingKey101 = 101;
    ApplyAdamWV2TilingData* tilingData = reinterpret_cast<ApplyAdamWV2TilingData*>(tiling);
    tilingData->totalCoreNum = totalCoreNum;
    tilingData->handleExtraLoopCoreNum = 1;
    tilingData->usedCoreNum = 1;
    tilingData->numPerLoop = numPerLoop;
    tilingData->loopNumPerCore = 1;
    tilingData->numLastLoop = numLastLoop;
    tilingData->lr = lr;
    tilingData->beta1 = beta1;
    tilingData->beta2 = beta2;
    tilingData->weightDecay = weightDecay;
    tilingData->eps = eps;
    tilingData->amsgrad = true;
    tilingData->maximize = true;
    tilingData->isBfloat16 = true;

    std::string curPath = ".";
    ReadFile(curPath + "/apply_adam_w_v2_data/var.bin", inputSize, var, inputSize);
    ReadFile(curPath + "/apply_adam_w_v2_data/m.bin", inputSize, m, inputSize);
    ReadFile(curPath + "/apply_adam_w_v2_data/v.bin", inputSize, v, inputSize);
    ReadFile(curPath + "/apply_adam_w_v2_data/max_grad_norm.bin", inputSize, maxGradNorm, inputSize);
    ReadFile(curPath + "/apply_adam_w_v2_data/grad.bin", inputSize, grad, inputSize);
    ReadFile(curPath + "/apply_adam_w_v2_data/step.bin", stepSize, step, stepSize);

    ICPU_SET_TILING_KEY(tilingKey101);
    ICPU_RUN_KF(apply_adam_w_v2, blockDim, var, m, v, grad, step, maxGradNorm, workSpace, tiling);

    WriteFile(curPath + "/apply_adam_w_v2_data/out_var.bin", var, outputSize);
    WriteFile(curPath + "/apply_adam_w_v2_data/out_max_grad_norm.bin", maxGradNorm, outputSize);
    WriteFile(curPath + "/apply_adam_w_v2_data/out_m.bin", m, outputSize);
    WriteFile(curPath + "/apply_adam_w_v2_data/out_v.bin", v, outputSize);
    AscendC::GmFree((void*)var);
    AscendC::GmFree((void*)grad);
    AscendC::GmFree((void*)maxGradNorm);
    AscendC::GmFree((void*)m);
    AscendC::GmFree((void*)v);
    AscendC::GmFree((void*)step);
}

TEST_F(apply_adam_w_v2_test, test_case_bfloat16_amsgrad_false_maximize_false)
{
    system(
        "cp -rf "
        "../../../../optim/apply_adam_w_v2/tests/ut/op_kernel/apply_adam_w_v2_data ./");
    system("chmod -R 755 ./apply_adam_w_v2_data/");
    system("cd ./apply_adam_w_v2_data/ && python3 gen_data.py 'bfloat16' 'false' 'false' 'bfloat16' 'int64' ");
    system("ls -l apply_adam_w_v2_data");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024 + 48 * sizeof(int32_t);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(ApplyAdamWV2TilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    std::vector<int64_t> shape = {30, 4, 2};
    size_t bytes = 2;
    size_t stepSize = 8;
    size_t inputSize = GetShapeSize(shape) * bytes;
    size_t outputSize = GetShapeSize(shape) * bytes;
    size_t workspaceBytesSize = 16 * 1024 * 1024 + 48 * sizeof(int32_t);
    uint32_t blockDim = 32;

    uint8_t* var = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* m = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* v = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* maxGradNorm = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* step = (uint8_t*)AscendC::GmAlloc(stepSize);

    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(workspaceBytesSize);
    int64_t totalCoreNum = 32;
    int64_t numPerLoop = 208;
    int64_t numLastLoop = 32;
    float lr = 0.01;
    float beta1 = 0.99;
    float beta2 = 0.1;
    float weightDecay = 0.005;
    float eps = 0.000001;
    int64_t tilingKey102 = 102;
    ApplyAdamWV2TilingData* tilingData = reinterpret_cast<ApplyAdamWV2TilingData*>(tiling);
    tilingData->totalCoreNum = totalCoreNum;
    tilingData->handleExtraLoopCoreNum = 1;
    tilingData->usedCoreNum = 1;
    tilingData->numPerLoop = numPerLoop;
    tilingData->loopNumPerCore = 1;
    tilingData->numLastLoop = numLastLoop;
    tilingData->lr = lr;
    tilingData->beta1 = beta1;
    tilingData->beta2 = beta2;
    tilingData->weightDecay = weightDecay;
    tilingData->eps = eps;
    tilingData->amsgrad = false;
    tilingData->maximize = false;
    tilingData->isBfloat16 = true;

    std::string curPath = ".";
    ReadFile(curPath + "/apply_adam_w_v2_data/var.bin", inputSize, var, inputSize);
    ReadFile(curPath + "/apply_adam_w_v2_data/m.bin", inputSize, m, inputSize);
    ReadFile(curPath + "/apply_adam_w_v2_data/v.bin", inputSize, v, inputSize);
    ReadFile(curPath + "/apply_adam_w_v2_data/max_grad_norm.bin", inputSize, maxGradNorm, inputSize);
    ReadFile(curPath + "/apply_adam_w_v2_data/grad.bin", inputSize, grad, inputSize);
    ReadFile(curPath + "/apply_adam_w_v2_data/step.bin", stepSize, step, stepSize);

    ICPU_SET_TILING_KEY(tilingKey102);
    ICPU_RUN_KF(apply_adam_w_v2, blockDim, var, m, v, grad, step, maxGradNorm, workSpace, tiling);

    WriteFile(curPath + "/apply_adam_w_v2_data/out_var.bin", var, outputSize);
    WriteFile(curPath + "/apply_adam_w_v2_data/out_max_grad_norm.bin", maxGradNorm, outputSize);
    WriteFile(curPath + "/apply_adam_w_v2_data/out_m.bin", m, outputSize);
    WriteFile(curPath + "/apply_adam_w_v2_data/out_v.bin", v, outputSize);
    AscendC::GmFree((void*)var);
    AscendC::GmFree((void*)grad);
    AscendC::GmFree((void*)maxGradNorm);
    AscendC::GmFree((void*)m);
    AscendC::GmFree((void*)v);
    AscendC::GmFree((void*)step);
}

TEST_F(apply_adam_w_v2_test, test_case_float32_float16_amsgrad_true_maximize_true)
{
    system(
        "cp -rf "
        "../../../../optim/apply_adam_w_v2/tests/ut/op_kernel/apply_adam_w_v2_data ./");
    system("chmod -R 755 ./apply_adam_w_v2_data/");
    system("cd ./apply_adam_w_v2_data/ && python3 gen_data.py 'float32' 'true' 'true' 'float16' ");
    system("ls -l apply_adam_w_v2_data");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024 + 48 * sizeof(int32_t);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(ApplyAdamWV2TilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    std::vector<int64_t> shape = {30, 4, 2};
    size_t bytes = 4;
    size_t bytes2 = 2;
    size_t inputSize = GetShapeSize(shape) * bytes;
    size_t inputSize2 = GetShapeSize(shape) * bytes2;
    size_t outputSize = GetShapeSize(shape) * bytes;
    size_t outputSize2 = GetShapeSize(shape) * bytes2;
    size_t workspaceBytesSize = 16 * 1024 * 1024 + 48 * sizeof(int32_t);
    uint32_t blockDim = 32;

    uint8_t* var = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* m = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* v = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* maxGradNorm = (uint8_t*)AscendC::GmAlloc(inputSize2);
    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(inputSize2);
    uint8_t* step = (uint8_t*)AscendC::GmAlloc(bytes);
    int64_t totalCoreNum = 32;
    int64_t numPerLoop = 208;
    int64_t numLastLoop = 32;
    float lr = 0.01;
    float beta1 = 0.99;
    float beta2 = 0.1;
    float weightDecay = 0.005;
    float eps = 0.000001;
    int64_t tilingKey107 = 107;
    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(workspaceBytesSize);

    ApplyAdamWV2TilingData* tilingData = reinterpret_cast<ApplyAdamWV2TilingData*>(tiling);
    tilingData->totalCoreNum = totalCoreNum;
    tilingData->handleExtraLoopCoreNum = 1;
    tilingData->usedCoreNum = 1;
    tilingData->numPerLoop = numPerLoop;
    tilingData->loopNumPerCore = 1;
    tilingData->numLastLoop = numLastLoop;
    tilingData->lr = lr;
    tilingData->beta1 = beta1;
    tilingData->beta2 = beta2;
    tilingData->weightDecay = weightDecay;
    tilingData->eps = eps;
    tilingData->amsgrad = true;
    tilingData->maximize = true;
    tilingData->isBfloat16 = false;

    std::string curPath = ".";
    ReadFile(curPath + "/apply_adam_w_v2_data/var.bin", inputSize, var, inputSize);
    ReadFile(curPath + "/apply_adam_w_v2_data/m.bin", inputSize, m, inputSize);
    ReadFile(curPath + "/apply_adam_w_v2_data/v.bin", inputSize, v, inputSize);
    ReadFile(curPath + "/apply_adam_w_v2_data/max_grad_norm.bin", inputSize2, maxGradNorm, inputSize2);
    ReadFile(curPath + "/apply_adam_w_v2_data/grad.bin", inputSize2, grad, inputSize2);
    ReadFile(curPath + "/apply_adam_w_v2_data/step.bin", bytes, step, bytes);

    ICPU_SET_TILING_KEY(tilingKey107);
    ICPU_RUN_KF(apply_adam_w_v2, blockDim, var, m, v, grad, step, maxGradNorm, workSpace, tiling);

    WriteFile(curPath + "/apply_adam_w_v2_data/out_var.bin", var, outputSize);
    WriteFile(curPath + "/apply_adam_w_v2_data/out_max_grad_norm.bin", maxGradNorm, outputSize2);
    WriteFile(curPath + "/apply_adam_w_v2_data/out_m.bin", m, outputSize);
    WriteFile(curPath + "/apply_adam_w_v2_data/out_v.bin", v, outputSize);
    AscendC::GmFree((void*)var);
    AscendC::GmFree((void*)grad);
    AscendC::GmFree((void*)maxGradNorm);
    AscendC::GmFree((void*)m);
    AscendC::GmFree((void*)v);
    AscendC::GmFree((void*)step);
}

TEST_F(apply_adam_w_v2_test, test_case_float32_float16_amsgrad_true_maximize_true_step_int64)
{
    system(
        "cp -rf "
        "../../../../optim/apply_adam_w_v2/tests/ut/op_kernel/apply_adam_w_v2_data ./");
    system("chmod -R 755 ./apply_adam_w_v2_data/");
    system("cd ./apply_adam_w_v2_data/ && python3 gen_data.py 'float32' 'true' 'true' 'float16' 'int64' ");
    system("ls -l apply_adam_w_v2_data");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024 + 48 * sizeof(int32_t);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(ApplyAdamWV2TilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    std::vector<int64_t> shape = {30, 4, 2};
    size_t bytes = 4;
    size_t bytes2 = 2;
    size_t stepSize = 8;
    size_t inputSize = GetShapeSize(shape) * bytes;
    size_t inputSize2 = GetShapeSize(shape) * bytes2;
    size_t outputSize = GetShapeSize(shape) * bytes;
    size_t outputSize2 = GetShapeSize(shape) * bytes2;
    size_t workspaceBytesSize = 16 * 1024 * 1024 + 48 * sizeof(int32_t);
    uint32_t blockDim = 32;

    uint8_t* var = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* m = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* v = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* maxGradNorm = (uint8_t*)AscendC::GmAlloc(inputSize2);
    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(inputSize2);
    uint8_t* step = (uint8_t*)AscendC::GmAlloc(stepSize);

    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(workspaceBytesSize);
    int64_t totalCoreNum = 32;
    int64_t numPerLoop = 208;
    int64_t numLastLoop = 32;
    float lr = 0.01;
    float beta1 = 0.99;
    float beta2 = 0.1;
    float weightDecay = 0.005;
    float eps = 0.000001;
    int64_t tilingKey108 = 108;
    ApplyAdamWV2TilingData* tilingData = reinterpret_cast<ApplyAdamWV2TilingData*>(tiling);
    tilingData->totalCoreNum = totalCoreNum;
    tilingData->handleExtraLoopCoreNum = 1;
    tilingData->usedCoreNum = 1;
    tilingData->numPerLoop = numPerLoop;
    tilingData->loopNumPerCore = 1;
    tilingData->numLastLoop = numLastLoop;
    tilingData->lr = lr;
    tilingData->beta1 = beta1;
    tilingData->beta2 = beta2;
    tilingData->weightDecay = weightDecay;
    tilingData->eps = eps;
    tilingData->amsgrad = true;
    tilingData->maximize = true;
    tilingData->isBfloat16 = false;

    std::string curPath = ".";
    ReadFile(curPath + "/apply_adam_w_v2_data/var.bin", inputSize, var, inputSize);
    ReadFile(curPath + "/apply_adam_w_v2_data/m.bin", inputSize, m, inputSize);
    ReadFile(curPath + "/apply_adam_w_v2_data/v.bin", inputSize, v, inputSize);
    ReadFile(curPath + "/apply_adam_w_v2_data/max_grad_norm.bin", inputSize2, maxGradNorm, inputSize2);
    ReadFile(curPath + "/apply_adam_w_v2_data/grad.bin", inputSize2, grad, inputSize2);
    ReadFile(curPath + "/apply_adam_w_v2_data/step.bin", stepSize, step, stepSize);

    ICPU_SET_TILING_KEY(tilingKey108);
    ICPU_RUN_KF(apply_adam_w_v2, blockDim, var, m, v, grad, step, maxGradNorm, workSpace, tiling);

    WriteFile(curPath + "/apply_adam_w_v2_data/out_var.bin", var, outputSize);
    WriteFile(curPath + "/apply_adam_w_v2_data/out_max_grad_norm.bin", maxGradNorm, outputSize2);
    WriteFile(curPath + "/apply_adam_w_v2_data/out_m.bin", m, outputSize);
    WriteFile(curPath + "/apply_adam_w_v2_data/out_v.bin", v, outputSize);
    AscendC::GmFree((void*)var);
    AscendC::GmFree((void*)grad);
    AscendC::GmFree((void*)maxGradNorm);
    AscendC::GmFree((void*)m);
    AscendC::GmFree((void*)v);
    AscendC::GmFree((void*)step);
}

TEST_F(apply_adam_w_v2_test, test_case_float32_bfloat16_amsgrad_false_maximize_false)
{
    system(
        "cp -rf "
        "../../../../optim/apply_adam_w_v2/tests/ut/op_kernel/apply_adam_w_v2_data ./");
    system("chmod -R 755 ./apply_adam_w_v2_data/");
    system("cd ./apply_adam_w_v2_data/ && python3 gen_data.py 'float32' 'false' 'false' 'bfloat16' ");
    system("ls -l apply_adam_w_v2_data");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024 + 48 * sizeof(int32_t);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(ApplyAdamWV2TilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    std::vector<int64_t> shape = {30, 4, 2};
    size_t bytes = 4;
    size_t bytes2 = 2;
    size_t inputSize = GetShapeSize(shape) * bytes;
    size_t inputSize2 = GetShapeSize(shape) * bytes2;
    size_t outputSize = GetShapeSize(shape) * bytes;
    size_t outputSize2 = GetShapeSize(shape) * bytes2;
    size_t workspaceBytesSize = 16 * 1024 * 1024 + 48 * sizeof(int32_t);
    uint32_t blockDim = 32;

    uint8_t* var = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* m = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* v = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* maxGradNorm = (uint8_t*)AscendC::GmAlloc(inputSize2);
    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(inputSize2);
    uint8_t* step = (uint8_t*)AscendC::GmAlloc(bytes);

    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(workspaceBytesSize);
    int64_t totalCoreNum = 32;
    int64_t numPerLoop = 208;
    int64_t numLastLoop = 32;
    float lr = 0.01;
    float beta1 = 0.99;
    float beta2 = 0.1;
    float weightDecay = 0.005;
    float eps = 0.000001;
    int64_t tilingKey109 = 109;
    ApplyAdamWV2TilingData* tilingData = reinterpret_cast<ApplyAdamWV2TilingData*>(tiling);
    tilingData->totalCoreNum = totalCoreNum;
    tilingData->handleExtraLoopCoreNum = 1;
    tilingData->usedCoreNum = 1;
    tilingData->numPerLoop = numPerLoop;
    tilingData->loopNumPerCore = 1;
    tilingData->numLastLoop = numLastLoop;
    tilingData->lr = lr;
    tilingData->beta1 = beta1;
    tilingData->beta2 = beta2;
    tilingData->weightDecay = weightDecay;
    tilingData->eps = eps;
    tilingData->amsgrad = true;
    tilingData->maximize = true;
    tilingData->isBfloat16 = true;

    std::string curPath = ".";
    ReadFile(curPath + "/apply_adam_w_v2_data/var.bin", inputSize, var, inputSize);
    ReadFile(curPath + "/apply_adam_w_v2_data/m.bin", inputSize, m, inputSize);
    ReadFile(curPath + "/apply_adam_w_v2_data/v.bin", inputSize, v, inputSize);
    ReadFile(curPath + "/apply_adam_w_v2_data/max_grad_norm.bin", inputSize2, maxGradNorm, inputSize2);
    ReadFile(curPath + "/apply_adam_w_v2_data/grad.bin", inputSize2, grad, inputSize2);
    ReadFile(curPath + "/apply_adam_w_v2_data/step.bin", bytes, step, bytes);

    ICPU_SET_TILING_KEY(tilingKey109);
    ICPU_RUN_KF(apply_adam_w_v2, blockDim, var, m, v, grad, step, maxGradNorm, workSpace, tiling);

    WriteFile(curPath + "/apply_adam_w_v2_data/out_var.bin", var, outputSize);
    WriteFile(curPath + "/apply_adam_w_v2_data/out_max_grad_norm.bin", maxGradNorm, outputSize2);
    WriteFile(curPath + "/apply_adam_w_v2_data/out_m.bin", m, outputSize);
    WriteFile(curPath + "/apply_adam_w_v2_data/out_v.bin", v, outputSize);
    AscendC::GmFree((void*)var);
    AscendC::GmFree((void*)grad);
    AscendC::GmFree((void*)maxGradNorm);
    AscendC::GmFree((void*)m);
    AscendC::GmFree((void*)v);
    AscendC::GmFree((void*)step);
}

TEST_F(apply_adam_w_v2_test, test_case_float32_bfloat16_amsgrad_false_maximize_false_step_int64)
{
    system(
        "cp -rf "
        "../../../../optim/apply_adam_w_v2/tests/ut/op_kernel/apply_adam_w_v2_data ./");
    system("chmod -R 755 ./apply_adam_w_v2_data/");
    system("cd ./apply_adam_w_v2_data/ && python3 gen_data.py 'float32' 'false' 'false' 'bfloat16' 'int64' ");
    system("ls -l apply_adam_w_v2_data");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t sysWorkspaceSize = 16 * 1024 * 1024 + 48 * sizeof(int32_t);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(sysWorkspaceSize);
    size_t tilingSize = sizeof(ApplyAdamWV2TilingData);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    std::vector<int64_t> shape = {30, 4, 2};
    size_t bytes = 4;
    size_t bytes2 = 2;
    size_t stepSize = 8;
    size_t inputSize = GetShapeSize(shape) * bytes;
    size_t inputSize2 = GetShapeSize(shape) * bytes2;
    size_t outputSize = GetShapeSize(shape) * bytes;
    size_t outputSize2 = GetShapeSize(shape) * bytes2;
    size_t workspaceBytesSize = 16 * 1024 * 1024 + 48 * sizeof(int32_t);
    uint32_t blockDim = 32;

    uint8_t* var = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* m = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* v = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* maxGradNorm = (uint8_t*)AscendC::GmAlloc(inputSize2);
    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(inputSize2);
    uint8_t* step = (uint8_t*)AscendC::GmAlloc(stepSize);

    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(workspaceBytesSize);
    int64_t totalCoreNum = 32;
    int64_t numPerLoop = 208;
    int64_t numLastLoop = 32;
    float lr = 0.01;
    float beta1 = 0.99;
    float beta2 = 0.1;
    float weightDecay = 0.005;
    float eps = 0.000001;
    int64_t tilingKey110 = 110;
    ApplyAdamWV2TilingData* tilingData = reinterpret_cast<ApplyAdamWV2TilingData*>(tiling);
    tilingData->totalCoreNum = totalCoreNum;
    tilingData->handleExtraLoopCoreNum = 1;
    tilingData->usedCoreNum = 1;
    tilingData->numPerLoop = numPerLoop;
    tilingData->loopNumPerCore = 1;
    tilingData->numLastLoop = numLastLoop;
    tilingData->lr = lr;
    tilingData->beta1 = beta1;
    tilingData->beta2 = beta2;
    tilingData->weightDecay = weightDecay;
    tilingData->eps = eps;
    tilingData->amsgrad = true;
    tilingData->maximize = true;
    tilingData->isBfloat16 = true;

    std::string curPath = ".";
    ReadFile(curPath + "/apply_adam_w_v2_data/var.bin", inputSize, var, inputSize);
    ReadFile(curPath + "/apply_adam_w_v2_data/m.bin", inputSize, m, inputSize);
    ReadFile(curPath + "/apply_adam_w_v2_data/v.bin", inputSize, v, inputSize);
    ReadFile(curPath + "/apply_adam_w_v2_data/max_grad_norm.bin", inputSize2, maxGradNorm, inputSize2);
    ReadFile(curPath + "/apply_adam_w_v2_data/grad.bin", inputSize2, grad, inputSize2);
    ReadFile(curPath + "/apply_adam_w_v2_data/step.bin", stepSize, step, stepSize);

    ICPU_SET_TILING_KEY(tilingKey110);
    ICPU_RUN_KF(apply_adam_w_v2, blockDim, var, m, v, grad, step, maxGradNorm, workSpace, tiling);

    WriteFile(curPath + "/apply_adam_w_v2_data/out_var.bin", var, outputSize);
    WriteFile(curPath + "/apply_adam_w_v2_data/out_max_grad_norm.bin", maxGradNorm, outputSize2);
    WriteFile(curPath + "/apply_adam_w_v2_data/out_m.bin", m, outputSize);
    WriteFile(curPath + "/apply_adam_w_v2_data/out_v.bin", v, outputSize);
    AscendC::GmFree((void*)var);
    AscendC::GmFree((void*)grad);
    AscendC::GmFree((void*)maxGradNorm);
    AscendC::GmFree((void*)m);
    AscendC::GmFree((void*)v);
    AscendC::GmFree((void*)step);
}