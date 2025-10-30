/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_apply_fused_ema_adam_tiling_def.h"
#include "data_utils.h"

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void apply_fused_ema_adam(
    GM_ADDR grad, GM_ADDR var, GM_ADDR m, GM_ADDR v, GM_ADDR s, GM_ADDR step, GM_ADDR var_ref, GM_ADDR m_ref,
    GM_ADDR v_ref, GM_ADDR s_ref, GM_ADDR workspace, GM_ADDR tiling);

class apply_fused_ema_adam_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "apply_fused_ema_adam_test SetUp\n" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "apply_fused_ema_adam_test TearDown\n" << std::endl;
    }
};

TEST_F(apply_fused_ema_adam_test, test_case_float32_mode_0_bias_0)
{
    system(
        "cp -rf "
        "../../../../optim/apply_fused_ema_adam/tests/ut/op_kernel/apply_fused_ema_adam_data ./");
    system("chmod -R 755 ./apply_fused_ema_adam_data/");
    system("cd ./apply_fused_ema_adam_data/ && python3 gen_data.py 'f32' ");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t inputSize = 32 * 64280 * sizeof(float);
    size_t inputStepSize = 1 * 8 * sizeof(int64_t);
    size_t outputSize = 32 * 64280 * sizeof(float);
    size_t tilingSize = sizeof(ApplyFusedEmaAdamUtTilingData);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* var = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* m = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* v = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* s = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* step = (uint8_t*)AscendC::GmAlloc(inputStepSize);

    uint8_t* var_ref = (uint8_t*)AscendC::GmAlloc(outputSize);
    uint8_t* m_ref = (uint8_t*)AscendC::GmAlloc(outputSize);
    uint8_t* v_ref = (uint8_t*)AscendC::GmAlloc(outputSize);
    uint8_t* s_ref = (uint8_t*)AscendC::GmAlloc(outputSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    uint32_t blockDim = 48;

    ApplyFusedEmaAdamUtTilingData* tilingData = reinterpret_cast<ApplyFusedEmaAdamUtTilingData*>(tiling);
    tilingData->mode = 0;
    tilingData->biasCorrection = 0;
    tilingData->frontCoreNum = 16;
    tilingData->tailCoreNum = 32;
    tilingData->coreCalcNum = 42854;
    tilingData->loopNum = 8;
    tilingData->coreCalcMax = 5448;
    tilingData->frontCalcExtra = 4718;
    tilingData->tailCalcExtra = 4717;

    ReadFile("./apply_fused_ema_adam_data/grad.bin", inputSize, grad, inputSize);
    ReadFile("./apply_fused_ema_adam_data/var.bin", inputSize, var, inputSize);
    ReadFile("./apply_fused_ema_adam_data/m.bin", inputSize, m, inputSize);
    ReadFile("./apply_fused_ema_adam_data/v.bin", inputSize, v, inputSize);
    ReadFile("./apply_fused_ema_adam_data/s.bin", inputSize, s, inputSize);
    ReadFile("./apply_fused_ema_adam_data/step.bin", inputStepSize, step, inputStepSize);

    ICPU_SET_TILING_KEY(102);
    ICPU_RUN_KF(
        apply_fused_ema_adam, blockDim, grad, var, m, v, s, step, var_ref, m_ref, v_ref, s_ref, workspace, tiling);

    WriteFile("./apply_fused_ema_adam_data/out_var.bin", var_ref, outputSize);
    WriteFile("./apply_fused_ema_adam_data/out_m.bin", m_ref, outputSize);
    WriteFile("./apply_fused_ema_adam_data/out_v.bin", v_ref, outputSize);
    WriteFile("./apply_fused_ema_adam_data/out_s.bin", s_ref, outputSize);

    AscendC::GmFree(grad);
    AscendC::GmFree(var);
    AscendC::GmFree(m);
    AscendC::GmFree(v);
    AscendC::GmFree(s);
    AscendC::GmFree(step);
    AscendC::GmFree(var_ref);
    AscendC::GmFree(m_ref);
    AscendC::GmFree(v_ref);
    AscendC::GmFree(s_ref);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(apply_fused_ema_adam_test, test_case_float32_mode_1_bias_1)
{
    system(
        "cp -rf "
        "../../../../optim/apply_fused_ema_adam/tests/ut/op_kernel/apply_fused_ema_adam_data ./");
    system("chmod -R 755 ./apply_fused_ema_adam_data/");
    system("cd ./apply_fused_ema_adam_data/ && python3 gen_data.py 'f32' ");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t inputSize = 32 * 64280 * sizeof(float);
    size_t inputStepSize = 1 * 8 * sizeof(int64_t);
    size_t outputSize = 32 * 64280 * sizeof(float);
    size_t tilingSize = sizeof(ApplyFusedEmaAdamUtTilingData);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* var = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* m = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* v = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* s = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* step = (uint8_t*)AscendC::GmAlloc(inputStepSize);

    uint8_t* var_ref = (uint8_t*)AscendC::GmAlloc(outputSize);
    uint8_t* m_ref = (uint8_t*)AscendC::GmAlloc(outputSize);
    uint8_t* v_ref = (uint8_t*)AscendC::GmAlloc(outputSize);
    uint8_t* s_ref = (uint8_t*)AscendC::GmAlloc(outputSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    uint32_t blockDim = 48;

    ApplyFusedEmaAdamUtTilingData* tilingData = reinterpret_cast<ApplyFusedEmaAdamUtTilingData*>(tiling);
    tilingData->mode = 1;
    tilingData->biasCorrection = 1;
    tilingData->frontCoreNum = 16;
    tilingData->tailCoreNum = 32;
    tilingData->coreCalcNum = 42854;
    tilingData->loopNum = 7;
    tilingData->coreCalcMax = 6128;
    tilingData->frontCalcExtra = 6086;
    tilingData->tailCalcExtra = 6085;

    ReadFile("./apply_fused_ema_adam_data/grad.bin", inputSize, grad, inputSize);
    ReadFile("./apply_fused_ema_adam_data/var.bin", inputSize, var, inputSize);
    ReadFile("./apply_fused_ema_adam_data/m.bin", inputSize, m, inputSize);
    ReadFile("./apply_fused_ema_adam_data/v.bin", inputSize, v, inputSize);
    ReadFile("./apply_fused_ema_adam_data/s.bin", inputSize, s, inputSize);
    ReadFile("./apply_fused_ema_adam_data/step.bin", inputStepSize, step, inputStepSize);

    ICPU_SET_TILING_KEY(102);
    ICPU_RUN_KF(
        apply_fused_ema_adam, blockDim, grad, var, m, v, s, step, var_ref, m_ref, v_ref, s_ref, workspace, tiling);

    WriteFile("./apply_fused_ema_adam_data/out_var.bin", var_ref, outputSize);
    WriteFile("./apply_fused_ema_adam_data/out_m.bin", m_ref, outputSize);
    WriteFile("./apply_fused_ema_adam_data/out_v.bin", v_ref, outputSize);
    WriteFile("./apply_fused_ema_adam_data/out_s.bin", s_ref, outputSize);

    AscendC::GmFree(grad);
    AscendC::GmFree(var);
    AscendC::GmFree(m);
    AscendC::GmFree(v);
    AscendC::GmFree(s);
    AscendC::GmFree(step);
    AscendC::GmFree(var_ref);
    AscendC::GmFree(m_ref);
    AscendC::GmFree(v_ref);
    AscendC::GmFree(s_ref);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(apply_fused_ema_adam_test, test_case_float16_mode_0_bias_1)
{
    system(
        "cp -rf "
        "../../../../optim/apply_fused_ema_adam/tests/ut/op_kernel/apply_fused_ema_adam_data ./");
    system("chmod -R 755 ./apply_fused_ema_adam_data/");
    system("cd ./apply_fused_ema_adam_data/ && python3 gen_data.py 'f16' ");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t inputSize = 32 * 64280 * sizeof(half);
    size_t inputStepSize = 1 * 8 * sizeof(int64_t);
    size_t outputSize = 32 * 64280 * sizeof(half);
    size_t tilingSize = sizeof(ApplyFusedEmaAdamUtTilingData);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* var = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* m = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* v = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* s = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* step = (uint8_t*)AscendC::GmAlloc(inputStepSize);

    uint8_t* var_ref = (uint8_t*)AscendC::GmAlloc(outputSize);
    uint8_t* m_ref = (uint8_t*)AscendC::GmAlloc(outputSize);
    uint8_t* v_ref = (uint8_t*)AscendC::GmAlloc(outputSize);
    uint8_t* s_ref = (uint8_t*)AscendC::GmAlloc(outputSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    uint32_t blockDim = 48;

    ApplyFusedEmaAdamUtTilingData* tilingData = reinterpret_cast<ApplyFusedEmaAdamUtTilingData*>(tiling);
    tilingData->mode = 0;
    tilingData->biasCorrection = 1;
    tilingData->frontCoreNum = 16;
    tilingData->tailCoreNum = 32;
    tilingData->coreCalcNum = 42854;
    tilingData->loopNum = 12;
    tilingData->coreCalcMax = 3632;
    tilingData->frontCalcExtra = 2902;
    tilingData->tailCalcExtra = 2901;

    ReadFile("./apply_fused_ema_adam_data/grad.bin", inputSize, grad, inputSize);
    ReadFile("./apply_fused_ema_adam_data/var.bin", inputSize, var, inputSize);
    ReadFile("./apply_fused_ema_adam_data/m.bin", inputSize, m, inputSize);
    ReadFile("./apply_fused_ema_adam_data/v.bin", inputSize, v, inputSize);
    ReadFile("./apply_fused_ema_adam_data/s.bin", inputSize, s, inputSize);
    ReadFile("./apply_fused_ema_adam_data/step.bin", inputStepSize, step, inputStepSize);

    ICPU_SET_TILING_KEY(101);
    ICPU_RUN_KF(
        apply_fused_ema_adam, blockDim, grad, var, m, v, s, step, var_ref, m_ref, v_ref, s_ref, workspace, tiling);

    WriteFile("./apply_fused_ema_adam_data/out_var.bin", var_ref, outputSize);
    WriteFile("./apply_fused_ema_adam_data/out_m.bin", m_ref, outputSize);
    WriteFile("./apply_fused_ema_adam_data/out_v.bin", v_ref, outputSize);
    WriteFile("./apply_fused_ema_adam_data/out_s.bin", s_ref, outputSize);

    AscendC::GmFree(grad);
    AscendC::GmFree(var);
    AscendC::GmFree(m);
    AscendC::GmFree(v);
    AscendC::GmFree(s);
    AscendC::GmFree(step);
    AscendC::GmFree(var_ref);
    AscendC::GmFree(m_ref);
    AscendC::GmFree(v_ref);
    AscendC::GmFree(s_ref);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(apply_fused_ema_adam_test, test_case_float16_mode_1_bias_0)
{
    system(
        "cp -rf "
        "../../../../optim/apply_fused_ema_adam/tests/ut/op_kernel/apply_fused_ema_adam_data ./");
    system("chmod -R 755 ./apply_fused_ema_adam_data/");
    system("cd ./apply_fused_ema_adam_data/ && python3 gen_data.py 'f16' ");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t inputSize = 32 * 64280 * sizeof(half);
    size_t inputStepSize = 1 * 8 * sizeof(int64_t);
    size_t outputSize = 32 * 64280 * sizeof(half);
    size_t tilingSize = sizeof(ApplyFusedEmaAdamUtTilingData);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* var = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* m = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* v = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* s = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* step = (uint8_t*)AscendC::GmAlloc(inputStepSize);

    uint8_t* var_ref = (uint8_t*)AscendC::GmAlloc(outputSize);
    uint8_t* m_ref = (uint8_t*)AscendC::GmAlloc(outputSize);
    uint8_t* v_ref = (uint8_t*)AscendC::GmAlloc(outputSize);
    uint8_t* s_ref = (uint8_t*)AscendC::GmAlloc(outputSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    uint32_t blockDim = 48;

    ApplyFusedEmaAdamUtTilingData* tilingData = reinterpret_cast<ApplyFusedEmaAdamUtTilingData*>(tiling);
    tilingData->mode = 1;
    tilingData->biasCorrection = 0;
    tilingData->frontCoreNum = 16;
    tilingData->tailCoreNum = 32;
    tilingData->coreCalcNum = 42854;
    tilingData->loopNum = 11;
    tilingData->coreCalcMax = 4080;
    tilingData->frontCalcExtra = 2054;
    tilingData->tailCalcExtra = 2053;

    ReadFile("./apply_fused_ema_adam_data/grad.bin", inputSize, grad, inputSize);
    ReadFile("./apply_fused_ema_adam_data/var.bin", inputSize, var, inputSize);
    ReadFile("./apply_fused_ema_adam_data/m.bin", inputSize, m, inputSize);
    ReadFile("./apply_fused_ema_adam_data/v.bin", inputSize, v, inputSize);
    ReadFile("./apply_fused_ema_adam_data/s.bin", inputSize, s, inputSize);
    ReadFile("./apply_fused_ema_adam_data/step.bin", inputStepSize, step, inputStepSize);

    ICPU_SET_TILING_KEY(101);
    ICPU_RUN_KF(
        apply_fused_ema_adam, blockDim, grad, var, m, v, s, step, var_ref, m_ref, v_ref, s_ref, workspace, tiling);

    WriteFile("./apply_fused_ema_adam_data/out_var.bin", var_ref, outputSize);
    WriteFile("./apply_fused_ema_adam_data/out_m.bin", m_ref, outputSize);
    WriteFile("./apply_fused_ema_adam_data/out_v.bin", v_ref, outputSize);
    WriteFile("./apply_fused_ema_adam_data/out_s.bin", s_ref, outputSize);

    AscendC::GmFree(grad);
    AscendC::GmFree(var);
    AscendC::GmFree(m);
    AscendC::GmFree(v);
    AscendC::GmFree(s);
    AscendC::GmFree(step);
    AscendC::GmFree(var_ref);
    AscendC::GmFree(m_ref);
    AscendC::GmFree(v_ref);
    AscendC::GmFree(s_ref);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(apply_fused_ema_adam_test, test_case_bfloat16_mode_0_bias_0)
{
    system(
        "cp -rf "
        "../../../../optim/apply_fused_ema_adam/tests/ut/op_kernel/apply_fused_ema_adam_data ./");
    system("chmod -R 755 ./apply_fused_ema_adam_data/");
    system("cd ./apply_fused_ema_adam_data/ && python3 gen_data.py 'bf16' ");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t inputSize = 32 * 64280 * sizeof(DT_BF16);
    size_t inputStepSize = 1 * 8 * sizeof(int64_t);
    size_t outputSize = 32 * 64280 * sizeof(DT_BF16);
    size_t tilingSize = sizeof(ApplyFusedEmaAdamUtTilingData);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* var = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* m = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* v = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* s = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* step = (uint8_t*)AscendC::GmAlloc(inputStepSize);

    uint8_t* var_ref = (uint8_t*)AscendC::GmAlloc(outputSize);
    uint8_t* m_ref = (uint8_t*)AscendC::GmAlloc(outputSize);
    uint8_t* v_ref = (uint8_t*)AscendC::GmAlloc(outputSize);
    uint8_t* s_ref = (uint8_t*)AscendC::GmAlloc(outputSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    uint32_t blockDim = 48;

    ApplyFusedEmaAdamUtTilingData* tilingData = reinterpret_cast<ApplyFusedEmaAdamUtTilingData*>(tiling);
    tilingData->mode = 0;
    tilingData->biasCorrection = 0;
    tilingData->frontCoreNum = 16;
    tilingData->tailCoreNum = 32;
    tilingData->coreCalcNum = 42854;
    tilingData->loopNum = 12;
    tilingData->coreCalcMax = 3632;
    tilingData->frontCalcExtra = 2902;
    tilingData->tailCalcExtra = 2901;

    ReadFile("./apply_fused_ema_adam_data/grad.bin", inputSize, grad, inputSize);
    ReadFile("./apply_fused_ema_adam_data/var.bin", inputSize, var, inputSize);
    ReadFile("./apply_fused_ema_adam_data/m.bin", inputSize, m, inputSize);
    ReadFile("./apply_fused_ema_adam_data/v.bin", inputSize, v, inputSize);
    ReadFile("./apply_fused_ema_adam_data/s.bin", inputSize, s, inputSize);
    ReadFile("./apply_fused_ema_adam_data/step.bin", inputStepSize, step, inputStepSize);

    ICPU_SET_TILING_KEY(100);
    ICPU_RUN_KF(
        apply_fused_ema_adam, blockDim, grad, var, m, v, s, step, var_ref, m_ref, v_ref, s_ref, workspace, tiling);

    WriteFile("./apply_fused_ema_adam_data/out_var.bin", var_ref, outputSize);
    WriteFile("./apply_fused_ema_adam_data/out_m.bin", m_ref, outputSize);
    WriteFile("./apply_fused_ema_adam_data/out_v.bin", v_ref, outputSize);
    WriteFile("./apply_fused_ema_adam_data/out_s.bin", s_ref, outputSize);

    AscendC::GmFree(grad);
    AscendC::GmFree(var);
    AscendC::GmFree(m);
    AscendC::GmFree(v);
    AscendC::GmFree(s);
    AscendC::GmFree(step);
    AscendC::GmFree(var_ref);
    AscendC::GmFree(m_ref);
    AscendC::GmFree(v_ref);
    AscendC::GmFree(s_ref);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(apply_fused_ema_adam_test, test_case_bfloat16_mode_1_bias_1)
{
    system(
        "cp -rf "
        "../../../../optim/apply_fused_ema_adam/tests/ut/op_kernel/apply_fused_ema_adam_data ./");
    system("chmod -R 755 ./apply_fused_ema_adam_data/");
    system("cd ./apply_fused_ema_adam_data/ && python3 gen_data.py 'bf16' ");
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    size_t inputSize = 32 * 64280 * sizeof(DT_BF16);
    size_t inputStepSize = 1 * 8 * sizeof(int64_t);
    size_t outputSize = 32 * 64280 * sizeof(DT_BF16);
    size_t tilingSize = sizeof(ApplyFusedEmaAdamUtTilingData);

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* var = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* m = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* v = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* s = (uint8_t*)AscendC::GmAlloc(inputSize);
    uint8_t* step = (uint8_t*)AscendC::GmAlloc(inputStepSize);

    uint8_t* var_ref = (uint8_t*)AscendC::GmAlloc(outputSize);
    uint8_t* m_ref = (uint8_t*)AscendC::GmAlloc(outputSize);
    uint8_t* v_ref = (uint8_t*)AscendC::GmAlloc(outputSize);
    uint8_t* s_ref = (uint8_t*)AscendC::GmAlloc(outputSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);
    uint32_t blockDim = 48;

    ApplyFusedEmaAdamUtTilingData* tilingData = reinterpret_cast<ApplyFusedEmaAdamUtTilingData*>(tiling);
    tilingData->mode = 1;
    tilingData->biasCorrection = 1;
    tilingData->frontCoreNum = 16;
    tilingData->tailCoreNum = 32;
    tilingData->coreCalcNum = 42854;
    tilingData->loopNum = 11;
    tilingData->coreCalcMax = 4080;
    tilingData->frontCalcExtra = 2054;
    tilingData->tailCalcExtra = 2053;

    ReadFile("./apply_fused_ema_adam_data/grad.bin", inputSize, grad, inputSize);
    ReadFile("./apply_fused_ema_adam_data/var.bin", inputSize, var, inputSize);
    ReadFile("./apply_fused_ema_adam_data/m.bin", inputSize, m, inputSize);
    ReadFile("./apply_fused_ema_adam_data/v.bin", inputSize, v, inputSize);
    ReadFile("./apply_fused_ema_adam_data/s.bin", inputSize, s, inputSize);
    ReadFile("./apply_fused_ema_adam_data/step.bin", inputStepSize, step, inputStepSize);

    ICPU_SET_TILING_KEY(100);
    ICPU_RUN_KF(
        apply_fused_ema_adam, blockDim, grad, var, m, v, s, step, var_ref, m_ref, v_ref, s_ref, workspace, tiling);

    WriteFile("./apply_fused_ema_adam_data/out_var.bin", var_ref, outputSize);
    WriteFile("./apply_fused_ema_adam_data/out_m.bin", m_ref, outputSize);
    WriteFile("./apply_fused_ema_adam_data/out_v.bin", v_ref, outputSize);
    WriteFile("./apply_fused_ema_adam_data/out_s.bin", s_ref, outputSize);

    AscendC::GmFree(grad);
    AscendC::GmFree(var);
    AscendC::GmFree(m);
    AscendC::GmFree(v);
    AscendC::GmFree(s);
    AscendC::GmFree(step);
    AscendC::GmFree(var_ref);
    AscendC::GmFree(m_ref);
    AscendC::GmFree(v_ref);
    AscendC::GmFree(s_ref);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
