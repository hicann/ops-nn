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
#include <cmath>
#include <cstring>
#include <limits>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "apply_adam_w_quant_tiling_def.h"

using namespace std;

extern "C" __global__ __aicore__ void apply_adam_w_quant(GM_ADDR var, GM_ADDR grad, GM_ADDR m, GM_ADDR v, GM_ADDR qmapM,
                                                         GM_ADDR qmapV, GM_ADDR absmaxM, GM_ADDR absmaxV, GM_ADDR step,
                                                         GM_ADDR varRef, GM_ADDR mRef, GM_ADDR vRef, GM_ADDR absmaxMRef,
                                                         GM_ADDR absmaxVRef, GM_ADDR workspace, GM_ADDR tiling);

class apply_adam_w_quant_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "apply_adam_w_quant_test SetUp\n" << std::endl; }
    static void TearDownTestCase() { std::cout << "apply_adam_w_quant_test TearDown\n" << std::endl; }
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
    tilingDatafromBin->last_block_size = 256;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(100);
    ICPU_RUN_KF(apply_adam_w_quant, blockDim, varRef, grad, mRef, vRef, qmapM, qmapV, absmaxMRef, absmaxVRef, step,
                out_var_ref, out_m_ref, out_v_ref, out_absmax_m_ref, out_absmax_v_ref, workSpace,
                (uint8_t*)(tilingDatafromBin));

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
    tilingDatafromBin->last_block_size = 256;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(200);
    ICPU_RUN_KF(apply_adam_w_quant, blockDim, varRef, grad, mRef, vRef, qmapM, qmapV, absmaxMRef, absmaxVRef, step,
                out_var_ref, out_m_ref, out_v_ref, out_absmax_m_ref, out_absmax_v_ref, workSpace,
                (uint8_t*)(tilingDatafromBin));

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
    tilingDatafromBin->last_block_size = 256;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(300);
    ICPU_RUN_KF(apply_adam_w_quant, blockDim, varRef, grad, mRef, vRef, qmapM, qmapV, absmaxMRef, absmaxVRef, step,
                out_var_ref, out_m_ref, out_v_ref, out_absmax_m_ref, out_absmax_v_ref, workSpace,
                (uint8_t*)(tilingDatafromBin));

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

#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 350)
TEST_F(apply_adam_w_quant_test, arch35_fp32_quant_and_unaligned_tail)
{
    constexpr uint32_t blockDim = 1;
    constexpr uint32_t blockSize = 256;
    constexpr uint32_t elementCount = blockSize + 1;
    constexpr uint32_t guardSize = 32;
    constexpr uint32_t absmaxCount = 2;

    auto* var = reinterpret_cast<float*>(AscendC::GmAlloc(elementCount * sizeof(float)));
    auto* grad = reinterpret_cast<float*>(AscendC::GmAlloc(elementCount * sizeof(float)));
    auto* stateM = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(elementCount * sizeof(uint8_t)));
    auto* stateV = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(elementCount * sizeof(uint8_t)));
    auto* qmapM = reinterpret_cast<float*>(AscendC::GmAlloc(blockSize * sizeof(float)));
    auto* qmapV = reinterpret_cast<float*>(AscendC::GmAlloc(blockSize * sizeof(float)));
    auto* absmaxM = reinterpret_cast<float*>(AscendC::GmAlloc(absmaxCount * sizeof(float)));
    auto* absmaxV = reinterpret_cast<float*>(AscendC::GmAlloc(absmaxCount * sizeof(float)));
    auto* step = reinterpret_cast<int64_t*>(AscendC::GmAlloc(sizeof(int64_t)));
    auto* varOut = reinterpret_cast<float*>(AscendC::GmAlloc(elementCount * sizeof(float)));
    auto* stateMOut = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(elementCount + guardSize));
    auto* stateVOut = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(elementCount + guardSize));
    auto* absmaxMOut = reinterpret_cast<float*>(AscendC::GmAlloc(absmaxCount * sizeof(float)));
    auto* absmaxVOut = reinterpret_cast<float*>(AscendC::GmAlloc(absmaxCount * sizeof(float)));
    auto* workspace = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(16 * 1024 * 1024));
    auto* tiling = reinterpret_cast<ApplyAdamWQuantTilingDataTest*>(
        AscendC::GmAlloc(sizeof(ApplyAdamWQuantTilingDataTest)));

    ASSERT_NE(var, nullptr);
    ASSERT_NE(grad, nullptr);
    ASSERT_NE(stateM, nullptr);
    ASSERT_NE(stateV, nullptr);
    ASSERT_NE(qmapM, nullptr);
    ASSERT_NE(qmapV, nullptr);
    ASSERT_NE(absmaxM, nullptr);
    ASSERT_NE(absmaxV, nullptr);
    ASSERT_NE(step, nullptr);
    ASSERT_NE(varOut, nullptr);
    ASSERT_NE(stateMOut, nullptr);
    ASSERT_NE(stateVOut, nullptr);
    ASSERT_NE(absmaxMOut, nullptr);
    ASSERT_NE(absmaxVOut, nullptr);
    ASSERT_NE(workspace, nullptr);
    ASSERT_NE(tiling, nullptr);

    for (uint32_t i = 0; i < elementCount; ++i) {
        var[i] = 1.0f;
        grad[i] = 0.001f;
        stateM[i] = 0;
        stateV[i] = 0;
        varOut[i] = 0.0f;
    }
    grad[0] = 1.0f;
    grad[elementCount - 1] = 0.5f;

    for (uint32_t i = 0; i < 128; ++i) {
        qmapM[i] = -1.0f + 0.999f * static_cast<float>(i) / 127.0f;
    }
    for (uint32_t i = 128; i < blockSize; ++i) {
        qmapM[i] = 0.1f + 0.9f * static_cast<float>(i - 128) / 127.0f;
    }
    for (uint32_t i = 0; i < blockSize; ++i) {
        qmapV[i] = static_cast<float>(i) / 255.0f;
    }

    float boundary = (qmapV[100] + qmapV[101]) * 0.5f;
    float boundaryGrad = std::sqrt(boundary);
    while (boundaryGrad * boundaryGrad >= boundary) {
        boundaryGrad = std::nextafter(boundaryGrad, 0.0f);
    }
    ASSERT_LT(boundaryGrad * boundaryGrad, boundary);
    ASSERT_GT(boundaryGrad * boundaryGrad + 1e-7f, boundary);
    grad[1] = boundaryGrad;

    absmaxM[0] = 1.0f;
    absmaxM[1] = 1.0f;
    absmaxV[0] = 1.0f;
    absmaxV[1] = 1.0f;
    step[0] = 0;
    std::memset(stateMOut, 0xA5, elementCount + guardSize);
    std::memset(stateVOut, 0xA5, elementCount + guardSize);

    tiling->use_num_core = 1;
    tiling->last_pre_core_row_work = 1;
    tiling->not_last_core_num = 0;
    tiling->not_last_pre_core_row_work = 2;
    tiling->last_core_last_block = 2;
    tiling->lr = 0.0f;
    tiling->beta1 = 0.0f;
    tiling->beta2 = 0.0f;
    tiling->weight_decay = 0.0f;
    tiling->eps = 1e-8f;
    tiling->gnorm_scale = 1.0f;
    tiling->block_size = blockSize;
    tiling->one_core_do_block_num_per_row = 2;
    tiling->tiling_key = 100;
    tiling->last_block_size = 1;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_SET_TILING_KEY(100);
    ICPU_RUN_KF(apply_adam_w_quant, blockDim, reinterpret_cast<uint8_t*>(var), reinterpret_cast<uint8_t*>(grad), stateM,
                stateV, reinterpret_cast<uint8_t*>(qmapM), reinterpret_cast<uint8_t*>(qmapV),
                reinterpret_cast<uint8_t*>(absmaxM), reinterpret_cast<uint8_t*>(absmaxV),
                reinterpret_cast<uint8_t*>(step), reinterpret_cast<uint8_t*>(varOut), stateMOut, stateVOut,
                reinterpret_cast<uint8_t*>(absmaxMOut), reinterpret_cast<uint8_t*>(absmaxVOut), workspace,
                reinterpret_cast<uint8_t*>(tiling));

    EXPECT_EQ(stateMOut[0], 255);
    EXPECT_EQ(stateMOut[elementCount - 1], 255);
    for (uint32_t i = 2; i < blockSize; ++i) {
        EXPECT_EQ(stateMOut[i], 128) << "index=" << i;
    }
    EXPECT_EQ(stateVOut[0], 255);
    EXPECT_EQ(stateVOut[1], 100);
    EXPECT_EQ(stateVOut[elementCount - 1], 255);
    EXPECT_NEAR(absmaxMOut[0], 1.0f, 1e-6f);
    EXPECT_NEAR(absmaxMOut[1], 0.5f, 1e-6f);
    EXPECT_NEAR(absmaxVOut[0], 1.0f, 1e-6f);
    EXPECT_NEAR(absmaxVOut[1], 0.25f, 1e-6f);
    for (uint32_t i = elementCount; i < elementCount + guardSize; ++i) {
        EXPECT_EQ(stateMOut[i], 0xA5) << "m guard index=" << i;
        EXPECT_EQ(stateVOut[i], 0xA5) << "v guard index=" << i;
    }

    AscendC::GmFree(var);
    AscendC::GmFree(grad);
    AscendC::GmFree(stateM);
    AscendC::GmFree(stateV);
    AscendC::GmFree(qmapM);
    AscendC::GmFree(qmapV);
    AscendC::GmFree(absmaxM);
    AscendC::GmFree(absmaxV);
    AscendC::GmFree(step);
    AscendC::GmFree(varOut);
    AscendC::GmFree(stateMOut);
    AscendC::GmFree(stateVOut);
    AscendC::GmFree(absmaxMOut);
    AscendC::GmFree(absmaxVOut);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
#endif
