/*
 * Copyright (c) 2026 联通（广东）产业互联网有限公司.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "intern_vl_add_rms_norm_tiling_def.h"

extern "C" __global__ __aicore__ void intern_vl_add_rms_norm(
    GM_ADDR x, GM_ADDR residual, GM_ADDR gamma,
    GM_ADDR y, GM_ADDR residual_out,
    GM_ADDR workspace, GM_ADDR tiling);

class InternVlAddRmsNormKernelTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "InternVlAddRmsNormKernelTest SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "InternVlAddRmsNormKernelTest TearDown" << std::endl;
    }
};


template <typename T>
void RunBasicCase(uint64_t tilingKey, uint32_t dtypeSize, float tolerance,
    uint32_t batch = 2, uint32_t hiddenSize = 16, uint32_t tileSize = 16)
{
    uint32_t elemNum = batch * hiddenSize;
    uint32_t gammaNum = hiddenSize;
    std::vector<float> xRef(elemNum);
    for (uint32_t i = 0; i < elemNum; ++i) {
        xRef[i] = static_cast<float>(i);
    }
    std::vector<float> residualRef(elemNum, 1.0f);
    std::vector<float> gammaRef(gammaNum);
    for (uint32_t i = 0; i < gammaNum; ++i) {
        gammaRef[i] = 1.0f + 0.01f * static_cast<float>(i);
    }

    std::vector<T> xHost(elemNum);
    std::vector<T> residualHost(elemNum);
    std::vector<T> gammaHost(gammaNum);
    for (uint32_t i = 0; i < elemNum; ++i) {
        xHost[i] = static_cast<T>(xRef[i]);
        residualHost[i] = static_cast<T>(residualRef[i]);
    }
    for (uint32_t i = 0; i < gammaNum; ++i) {
        gammaHost[i] = static_cast<T>(gammaRef[i]);
    }

    auto* x = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(elemNum * sizeof(T)));
    auto* residual = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(elemNum * sizeof(T)));
    auto* gamma = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(gammaNum * sizeof(T)));
    auto* y = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(elemNum * sizeof(T)));
    auto* residualOut = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(elemNum * sizeof(T)));
    auto* workspace = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(1));
    auto* tiling = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sizeof(optiling::InternVlAddRmsNormTilingData)));

    std::memcpy(x, xHost.data(), elemNum * sizeof(T));
    std::memcpy(residual, residualHost.data(), elemNum * sizeof(T));
    std::memcpy(gamma, gammaHost.data(), gammaNum * sizeof(T));

    auto* tilingData = reinterpret_cast<optiling::InternVlAddRmsNormTilingData*>(tiling);
    tilingData->batch = batch;
    tilingData->hidden_size = hiddenSize;
    tilingData->tile_size = tileSize;
    tilingData->tile_num = (hiddenSize + tileSize - 1) / tileSize;
    tilingData->dtype_size = dtypeSize;
    tilingData->total_size = elemNum * sizeof(T);
    tilingData->eps = 1e-6f;
    tilingData->block_num = 1;
    tilingData->tokens_per_core = 2;

    if (tilingKey == 10) {
        ICPU_SET_TILING_KEY(10);
        ICPU_RUN_KF(intern_vl_add_rms_norm, 1, x, residual, gamma, y, residualOut, workspace, tiling);
    } else if (tilingKey == 20) {
        ICPU_SET_TILING_KEY(20);
        ICPU_RUN_KF(intern_vl_add_rms_norm, 1, x, residual, gamma, y, residualOut, workspace, tiling);
    } else {
        ICPU_SET_TILING_KEY(30);
        ICPU_RUN_KF(intern_vl_add_rms_norm, 1, x, residual, gamma, y, residualOut, workspace, tiling);
    }

    auto* yOut = reinterpret_cast<T*>(y);
    auto* residualOutHost = reinterpret_cast<T*>(residualOut);

    for (uint32_t row = 0; row < batch; ++row) {
        float sumSq = 0.0f;
        for (uint32_t col = 0; col < gammaNum; ++col) {
            uint32_t idx = row * gammaNum + col;
            float residualOutRef = xRef[idx] + residualRef[idx];
            sumSq += residualOutRef * residualOutRef;
        }

        float invRms = 1.0f / std::sqrt(sumSq / gammaNum + 1e-6f);
        for (uint32_t col = 0; col < gammaNum; ++col) {
            uint32_t idx = row * gammaNum + col;
            float residualOutRef = xRef[idx] + residualRef[idx];
            float yRef = residualOutRef * invRms * gammaRef[col];
            EXPECT_NEAR(static_cast<float>(residualOutHost[idx]), residualOutRef, tolerance);
            EXPECT_NEAR(static_cast<float>(yOut[idx]), yRef, tolerance);
        }
    }

    AscendC::GmFree(x);
    AscendC::GmFree(residual);
    AscendC::GmFree(gamma);
    AscendC::GmFree(y);
    AscendC::GmFree(residualOut);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(InternVlAddRmsNormKernelTest, float16_basic)
{
    RunBasicCase<half>(10, 2, 5e-3f);
}

TEST_F(InternVlAddRmsNormKernelTest, float32_basic)
{
    RunBasicCase<float>(20, 4, 1e-5f);
}

TEST_F(InternVlAddRmsNormKernelTest, bfloat16_basic)
{
    RunBasicCase<bfloat16_t>(30, 2, 5e-2f);
}

TEST_F(InternVlAddRmsNormKernelTest, float16_multi_tile)
{
    RunBasicCase<half>(10, 2, 5e-3f, 2, 32, 16);
}

TEST_F(InternVlAddRmsNormKernelTest, float32_multi_tile)
{
    RunBasicCase<float>(20, 4, 1e-5f, 2, 32, 16);
}

TEST_F(InternVlAddRmsNormKernelTest, bfloat16_multi_tile)
{
    RunBasicCase<bfloat16_t>(30, 2, 5e-2f, 2, 32, 16);
}
