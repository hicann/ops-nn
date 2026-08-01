/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <cstdint>
#include <vector>

#include "gtest/gtest.h"
#include "tikicpulib.h"

#include "arch35/deep_norm_grad.h"

namespace {

template <typename T>
void RunDeepNormGradArch35Kernel(GM_ADDR dy, GM_ADDR x, GM_ADDR gx, GM_ADDR gamma, GM_ADDR mean, GM_ADDR rstd,
                                 GM_ADDR dx, GM_ADDR dgx, GM_ADDR dbeta, GM_ADDR dgamma, GM_ADDR workspace,
                                 GM_ADDR tiling)
{
    (void)workspace;
    auto tilingData = reinterpret_cast<const DeepNormGradTilingDataArch35*>(tiling);
    DeepNormGradArch35::DeepNormGrad<T> op;
    op.Init(dy, x, gx, gamma, mean, rstd, dx, dgx, dbeta, dgamma, tilingData);
    op.Process();
}

} // namespace

TEST(DeepNormGradKernelArch35, Fp32BackwardAndGammaBeta)
{
    constexpr uint32_t rows = 3;
    constexpr uint32_t cols = 100;
    constexpr uint32_t total = rows * cols;
    constexpr float alpha = 0.3f;
    constexpr float eps = 1e-6f;
    constexpr uint32_t backwardBlockDim = 2;
    constexpr uint32_t gammaBetaBlockDim = 2;
    constexpr uint32_t blockDim = 2;

    auto dy = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(total * sizeof(float)));
    auto x = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(total * sizeof(float)));
    auto gx = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(total * sizeof(float)));
    auto gamma = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(cols * sizeof(float)));
    auto mean = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(rows * sizeof(float)));
    auto rstd = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(rows * sizeof(float)));
    auto dx = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(total * sizeof(float)));
    auto dgx = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(total * sizeof(float)));
    auto dbeta = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(cols * sizeof(float)));
    auto dgamma = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(cols * sizeof(float)));
    auto workspace = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(32));
    auto tiling = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sizeof(DeepNormGradTilingDataArch35)));

    auto dyData = reinterpret_cast<float*>(dy);
    auto xData = reinterpret_cast<float*>(x);
    auto gxData = reinterpret_cast<float*>(gx);
    auto gammaData = reinterpret_cast<float*>(gamma);
    auto meanData = reinterpret_cast<float*>(mean);
    auto rstdData = reinterpret_cast<float*>(rstd);
    for (uint32_t i = 0; i < total; ++i) {
        dyData[i] = static_cast<float>(static_cast<int32_t>(i % 11) - 5) * 0.02f;
        xData[i] = static_cast<float>(static_cast<int32_t>(i % 17) - 8) * 0.03125f;
        gxData[i] = static_cast<float>(static_cast<int32_t>(i % 13) - 6) * 0.015625f;
    }
    for (uint32_t c = 0; c < cols; ++c) {
        gammaData[c] = 0.75f + static_cast<float>(c % 11) * 0.01f;
    }

    std::vector<float> centered(total, 0.0f);
    for (uint32_t r = 0; r < rows; ++r) {
        float sum = 0.0f;
        for (uint32_t c = 0; c < cols; ++c) {
            uint32_t idx = r * cols + c;
            sum += alpha * xData[idx] + gxData[idx];
        }
        meanData[r] = sum / static_cast<float>(cols);
        float var = 0.0f;
        for (uint32_t c = 0; c < cols; ++c) {
            uint32_t idx = r * cols + c;
            centered[idx] = alpha * xData[idx] + gxData[idx] - meanData[r];
            var += centered[idx] * centered[idx];
        }
        rstdData[r] = 1.0f / std::sqrt(var / static_cast<float>(cols) + eps);
    }

    std::vector<float> expectDx(total, 0.0f);
    std::vector<float> expectDgx(total, 0.0f);
    std::vector<float> expectDbeta(cols, 0.0f);
    std::vector<float> expectDgamma(cols, 0.0f);
    const float invCols = 1.0f / static_cast<float>(cols);
    for (uint32_t r = 0; r < rows; ++r) {
        float sumTmp = 0.0f;
        float sumTmpNorm = 0.0f;
        for (uint32_t c = 0; c < cols; ++c) {
            uint32_t idx = r * cols + c;
            float dyGamma = dyData[idx] * gammaData[c];
            sumTmp += dyGamma * centered[idx] * rstdData[r] * rstdData[r] * rstdData[r];
            sumTmpNorm += dyGamma * rstdData[r];
        }
        float avgTmp = -invCols * sumTmp;
        float avgTmpNorm = -invCols * sumTmpNorm;
        for (uint32_t c = 0; c < cols; ++c) {
            uint32_t idx = r * cols + c;
            float dyGamma = dyData[idx] * gammaData[c];
            expectDgx[idx] = dyGamma * rstdData[r] + centered[idx] * avgTmp + avgTmpNorm;
            expectDx[idx] = expectDgx[idx] * alpha;
            expectDbeta[c] += dyData[idx];
            expectDgamma[c] += dyData[idx] * centered[idx] * rstdData[r];
        }
    }

    auto tilingData = reinterpret_cast<DeepNormGradTilingDataArch35*>(tiling);
    tilingData->numRows = rows;
    tilingData->numCols = cols;
    tilingData->rowsPerCore = 2;
    tilingData->colsPerCore = 64;
    tilingData->backwardBlockDim = backwardBlockDim;
    tilingData->gammaBetaBlockDim = gammaBetaBlockDim;
    tilingData->tileLength = 128;
    tilingData->tileLengthAlign = 128;
    tilingData->alpha = alpha;
    tilingData->invCols = invCols;

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(RunDeepNormGradArch35Kernel<float>, blockDim, dy, x, gx, gamma, mean, rstd, dx, dgx, dbeta, dgamma,
                workspace, tiling);

    auto dxData = reinterpret_cast<float*>(dx);
    auto dgxData = reinterpret_cast<float*>(dgx);
    auto dbetaData = reinterpret_cast<float*>(dbeta);
    auto dgammaData = reinterpret_cast<float*>(dgamma);
    for (uint32_t i = 0; i < total; ++i) {
        EXPECT_NEAR(dxData[i], expectDx[i], 4e-3f) << "dx index " << i;
        EXPECT_NEAR(dgxData[i], expectDgx[i], 1e-2f) << "dgx index " << i;
    }
    for (uint32_t c = 0; c < cols; ++c) {
        EXPECT_NEAR(dbetaData[c], expectDbeta[c], 2e-3f) << "dbeta index " << c;
        EXPECT_NEAR(dgammaData[c], expectDgamma[c], 2e-3f) << "dgamma index " << c;
    }

    AscendC::GmFree(dy);
    AscendC::GmFree(x);
    AscendC::GmFree(gx);
    AscendC::GmFree(gamma);
    AscendC::GmFree(mean);
    AscendC::GmFree(rstd);
    AscendC::GmFree(dx);
    AscendC::GmFree(dgx);
    AscendC::GmFree(dbeta);
    AscendC::GmFree(dgamma);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
