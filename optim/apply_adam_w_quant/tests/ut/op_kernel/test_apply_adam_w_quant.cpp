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
 * \file test_apply_adam_w_quant.cpp
 * \brief
 */
#include "data_utils.h"
#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_apply_adam_w_quant_tiling_def.h"

using namespace std;

extern "C" __global__ __aicore__ void apply_adam_w_quant(
    GM_ADDR var, GM_ADDR grad, GM_ADDR m, GM_ADDR v, GM_ADDR qmapM, GM_ADDR qmapV, GM_ADDR absmaxM, GM_ADDR absmaxV,
    GM_ADDR step, GM_ADDR varRef, GM_ADDR mRef, GM_ADDR vRef, GM_ADDR absmaxMRef, GM_ADDR absmaxVRef, GM_ADDR workspace,
    GM_ADDR tiling);

class apply_adam_w_quant_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "apply_adam_w_quant_test SetUp\n" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "apply_adam_w_quant_test TearDown\n" << std::endl;
    }
};

TEST_F(apply_adam_w_quant_test, test_case_float32_tilingkey100_true)
{
    uint32_t blockDim = 1;
    size_t int64_t_bytes = 8;
    size_t varSize = 96 * 256 * sizeof(float);
    size_t gradSize = 96 * 256 * sizeof(float);
    size_t mRefSize = 96 * 256 * sizeof(uint8_t);
    size_t vRefSize = 96 * 256 * sizeof(uint8_t);
    size_t qmapMSize = 256 * sizeof(float);
    size_t qmapVSize = 256 * sizeof(float);
    size_t absmaxMRefSize = 96 * sizeof(float);
    size_t absmaxVRefSize = 96 * sizeof(float);
    size_t stepSize = 1 * int64_t_bytes;
    size_t tilingSize = sizeof(ApplyAdamWQuantTilingDataTest);

    uint8_t* varRef = (uint8_t*)AscendC::GmAlloc(varSize);
    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(gradSize);
    uint8_t* mRef = (uint8_t*)AscendC::GmAlloc(mRefSize);
    uint8_t* vRef = (uint8_t*)AscendC::GmAlloc(vRefSize);
    uint8_t* qmapM = (uint8_t*)AscendC::GmAlloc(qmapMSize);
    uint8_t* qmapV = (uint8_t*)AscendC::GmAlloc(qmapVSize);
    uint8_t* absmaxMRef = (uint8_t*)AscendC::GmAlloc(absmaxMRefSize);
    uint8_t* absmaxVRef = (uint8_t*)AscendC::GmAlloc(absmaxVRefSize);
    uint8_t* step = (uint8_t*)AscendC::GmAlloc(stepSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    uint8_t* out_var_ref = (uint8_t*)AscendC::GmAlloc(varSize);
    uint8_t* out_m_ref = (uint8_t*)AscendC::GmAlloc(mRefSize);
    uint8_t* out_v_ref = (uint8_t*)AscendC::GmAlloc(vRefSize);
    uint8_t* out_absmax_m_ref = (uint8_t*)AscendC::GmAlloc(absmaxMRefSize);
    uint8_t* out_absmax_v_ref = (uint8_t*)AscendC::GmAlloc(absmaxVRefSize);

    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 1024);
    ApplyAdamWQuantTilingDataTest* tilingDatafromBin = reinterpret_cast<ApplyAdamWQuantTilingDataTest*>(tiling);
    tilingDatafromBin->use_num_core = 1;
    tilingDatafromBin->last_pre_core_row_work = 1;
    tilingDatafromBin->not_last_core_num = 0;
    tilingDatafromBin->not_last_pre_core_row_work = 2;
    tilingDatafromBin->last_core_last_block = 16;
    tilingDatafromBin->lr = 0.001;
    tilingDatafromBin->beta1 = 0.9;
    tilingDatafromBin->beta2 = 0.999;
    tilingDatafromBin->weight_decay = 1.0;
    tilingDatafromBin->eps = 1e-8;
    tilingDatafromBin->gnorm_scale = 1.0;
    tilingDatafromBin->block_size = 256;
    tilingDatafromBin->one_core_do_block_num_per_row = 16;
    tilingDatafromBin->tiling_key = 100;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(100);
    ICPU_RUN_KF(
        apply_adam_w_quant, blockDim, varRef, grad, mRef, vRef, qmapM, qmapV, absmaxMRef, absmaxVRef, step, out_var_ref,
        out_m_ref, out_v_ref, out_absmax_m_ref, out_absmax_v_ref, workSpace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree((void*)varRef);
    AscendC::GmFree((void*)grad);
    AscendC::GmFree((void*)mRef);
    AscendC::GmFree((void*)vRef);
    AscendC::GmFree((void*)qmapM);
    AscendC::GmFree((void*)qmapV);
    AscendC::GmFree((void*)absmaxMRef);
    AscendC::GmFree((void*)absmaxVRef);
    AscendC::GmFree((void*)step);
    AscendC::GmFree((void*)out_var_ref);
    AscendC::GmFree((void*)out_m_ref);
    AscendC::GmFree((void*)out_v_ref);
    AscendC::GmFree((void*)out_absmax_m_ref);
    AscendC::GmFree((void*)out_absmax_v_ref);
    AscendC::GmFree(tiling);
}

TEST_F(apply_adam_w_quant_test, test_case_float32_tilingkey200_true)
{
    uint32_t blockDim = 1;
    size_t int64_t_bytes = 8;
    size_t varSize = 96 * 256 * sizeof(int16_t);
    size_t gradSize = 96 * 256 * sizeof(int16_t);
    size_t mRefSize = 96 * 256 * sizeof(uint8_t);
    size_t vRefSize = 96 * 256 * sizeof(uint8_t);
    size_t qmapMSize = 256 * sizeof(float);
    size_t qmapVSize = 256 * sizeof(float);
    size_t absmaxMRefSize = 96 * sizeof(float);
    size_t absmaxVRefSize = 96 * sizeof(float);
    size_t stepSize = 1 * int64_t_bytes;
    size_t tilingSize = sizeof(ApplyAdamWQuantTilingDataTest);

    uint8_t* varRef = (uint8_t*)AscendC::GmAlloc(varSize);
    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(gradSize);
    uint8_t* mRef = (uint8_t*)AscendC::GmAlloc(mRefSize);
    uint8_t* vRef = (uint8_t*)AscendC::GmAlloc(vRefSize);
    uint8_t* qmapM = (uint8_t*)AscendC::GmAlloc(qmapMSize);
    uint8_t* qmapV = (uint8_t*)AscendC::GmAlloc(qmapVSize);
    uint8_t* absmaxMRef = (uint8_t*)AscendC::GmAlloc(absmaxMRefSize);
    uint8_t* absmaxVRef = (uint8_t*)AscendC::GmAlloc(absmaxVRefSize);
    uint8_t* step = (uint8_t*)AscendC::GmAlloc(stepSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    uint8_t* out_var_ref = (uint8_t*)AscendC::GmAlloc(varSize);
    uint8_t* out_m_ref = (uint8_t*)AscendC::GmAlloc(mRefSize);
    uint8_t* out_v_ref = (uint8_t*)AscendC::GmAlloc(vRefSize);
    uint8_t* out_absmax_m_ref = (uint8_t*)AscendC::GmAlloc(absmaxMRefSize);
    uint8_t* out_absmax_v_ref = (uint8_t*)AscendC::GmAlloc(absmaxVRefSize);

    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 1024);
    ApplyAdamWQuantTilingDataTest* tilingDatafromBin = reinterpret_cast<ApplyAdamWQuantTilingDataTest*>(tiling);
    tilingDatafromBin->use_num_core = 1;
    tilingDatafromBin->last_pre_core_row_work = 1;
    tilingDatafromBin->not_last_core_num = 0;
    tilingDatafromBin->not_last_pre_core_row_work = 2;
    tilingDatafromBin->last_core_last_block = 16;
    tilingDatafromBin->lr = 0.001;
    tilingDatafromBin->beta1 = 0.9;
    tilingDatafromBin->beta2 = 0.999;
    tilingDatafromBin->weight_decay = 1.0;
    tilingDatafromBin->eps = 1e-8;
    tilingDatafromBin->gnorm_scale = 1.0;
    tilingDatafromBin->block_size = 256;
    tilingDatafromBin->one_core_do_block_num_per_row = 16;
    tilingDatafromBin->tiling_key = 200;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(200);
    ICPU_RUN_KF(
        apply_adam_w_quant, blockDim, varRef, grad, mRef, vRef, qmapM, qmapV, absmaxMRef, absmaxVRef, step, out_var_ref,
        out_m_ref, out_v_ref, out_absmax_m_ref, out_absmax_v_ref, workSpace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree((void*)varRef);
    AscendC::GmFree((void*)grad);
    AscendC::GmFree((void*)mRef);
    AscendC::GmFree((void*)vRef);
    AscendC::GmFree((void*)qmapM);
    AscendC::GmFree((void*)qmapV);
    AscendC::GmFree((void*)absmaxMRef);
    AscendC::GmFree((void*)absmaxVRef);
    AscendC::GmFree((void*)step);
    AscendC::GmFree((void*)out_var_ref);
    AscendC::GmFree((void*)out_m_ref);
    AscendC::GmFree((void*)out_v_ref);
    AscendC::GmFree((void*)out_absmax_m_ref);
    AscendC::GmFree((void*)out_absmax_v_ref);
    AscendC::GmFree(tiling);
}


TEST_F(apply_adam_w_quant_test, test_case_float32_tilingkey300_true)
{
    uint32_t blockDim = 1;
    size_t int64_t_bytes = 8;
    size_t varSize = 96 * 256 * sizeof(int16_t);
    size_t gradSize = 96 * 256 * sizeof(int16_t);
    size_t mRefSize = 96 * 256 * sizeof(uint8_t);
    size_t vRefSize = 96 * 256 * sizeof(uint8_t);
    size_t qmapMSize = 256 * sizeof(float);
    size_t qmapVSize = 256 * sizeof(float);
    size_t absmaxMRefSize = 96 * sizeof(float);
    size_t absmaxVRefSize = 96 * sizeof(float);
    size_t stepSize = 1 * int64_t_bytes;
    size_t tilingSize = sizeof(ApplyAdamWQuantTilingDataTest);

    uint8_t* varRef = (uint8_t*)AscendC::GmAlloc(varSize);
    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(gradSize);
    uint8_t* mRef = (uint8_t*)AscendC::GmAlloc(mRefSize);
    uint8_t* vRef = (uint8_t*)AscendC::GmAlloc(vRefSize);
    uint8_t* qmapM = (uint8_t*)AscendC::GmAlloc(qmapMSize);
    uint8_t* qmapV = (uint8_t*)AscendC::GmAlloc(qmapVSize);
    uint8_t* absmaxMRef = (uint8_t*)AscendC::GmAlloc(absmaxMRefSize);
    uint8_t* absmaxVRef = (uint8_t*)AscendC::GmAlloc(absmaxVRefSize);
    uint8_t* step = (uint8_t*)AscendC::GmAlloc(stepSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    uint8_t* out_var_ref = (uint8_t*)AscendC::GmAlloc(varSize);
    uint8_t* out_m_ref = (uint8_t*)AscendC::GmAlloc(mRefSize);
    uint8_t* out_v_ref = (uint8_t*)AscendC::GmAlloc(vRefSize);
    uint8_t* out_absmax_m_ref = (uint8_t*)AscendC::GmAlloc(absmaxMRefSize);
    uint8_t* out_absmax_v_ref = (uint8_t*)AscendC::GmAlloc(absmaxVRefSize);

    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 1024);
    ApplyAdamWQuantTilingDataTest* tilingDatafromBin = reinterpret_cast<ApplyAdamWQuantTilingDataTest*>(tiling);
    tilingDatafromBin->use_num_core = 1;
    tilingDatafromBin->last_pre_core_row_work = 1;
    tilingDatafromBin->not_last_core_num = 0;
    tilingDatafromBin->not_last_pre_core_row_work = 2;
    tilingDatafromBin->last_core_last_block = 16;
    tilingDatafromBin->lr = 0.001;
    tilingDatafromBin->beta1 = 0.9;
    tilingDatafromBin->beta2 = 0.999;
    tilingDatafromBin->weight_decay = 1.0;
    tilingDatafromBin->eps = 1e-8;
    tilingDatafromBin->gnorm_scale = 1.0;
    tilingDatafromBin->block_size = 256;
    tilingDatafromBin->one_core_do_block_num_per_row = 16;
    tilingDatafromBin->tiling_key = 300;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(300);
    ICPU_RUN_KF(
        apply_adam_w_quant, blockDim, varRef, grad, mRef, vRef, qmapM, qmapV, absmaxMRef, absmaxVRef, step, out_var_ref,
        out_m_ref, out_v_ref, out_absmax_m_ref, out_absmax_v_ref, workSpace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree((void*)varRef);
    AscendC::GmFree((void*)grad);
    AscendC::GmFree((void*)mRef);
    AscendC::GmFree((void*)vRef);
    AscendC::GmFree((void*)qmapM);
    AscendC::GmFree((void*)qmapV);
    AscendC::GmFree((void*)absmaxMRef);
    AscendC::GmFree((void*)absmaxVRef);
    AscendC::GmFree((void*)step);
    AscendC::GmFree((void*)out_var_ref);
    AscendC::GmFree((void*)out_m_ref);
    AscendC::GmFree((void*)out_v_ref);
    AscendC::GmFree((void*)out_absmax_m_ref);
    AscendC::GmFree((void*)out_absmax_v_ref);
    AscendC::GmFree(tiling);
}
