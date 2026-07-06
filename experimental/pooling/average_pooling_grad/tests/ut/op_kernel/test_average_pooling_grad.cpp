/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Pei Haobo<@xiaopei-1>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <string>
#include <vector>
#include "gtest/gtest.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#include "kernel_ut_data_helper.h"
#include "kernel_ut_data_executor.h"
#endif
#include "../../../op_kernel/average_pooling_grad_tiling_data.h"

__global__ __aicore__ void average_pooling_grad(GM_ADDR orig_input_shape, GM_ADDR grad_output, GM_ADDR grad_input,
                                                GM_ADDR workspace, GM_ADDR tiling);

namespace {
constexpr uint32_t N = 2;
constexpr uint32_t C = 1;
constexpr uint32_t H = 4;
constexpr uint32_t W = 6;
constexpr uint32_t TOTAL_NUM = N * C * H * W;

void InitTilingData(AveragePoolingGradTilingData* tilingData)
{
    tilingData->n = N;
    tilingData->c = C;
    tilingData->hIn = H;
    tilingData->wIn = W;
    tilingData->hOut = H;
    tilingData->wOut = W;
    tilingData->kernelH = 1;
    tilingData->kernelW = 1;
    tilingData->strideH = 1;
    tilingData->strideW = 1;
    tilingData->padTop = 0;
    tilingData->padLeft = 0;
    tilingData->countIncludePad = 1;
    tilingData->divisorOverride = 0;
    tilingData->invDivisor = 1.0f;
    tilingData->totalElements = TOTAL_NUM;
    tilingData->normalCoreProcessNum = TOTAL_NUM;
    tilingData->tailCoreProcessNum = TOTAL_NUM;
    tilingData->usedCoreNum = 1;
    tilingData->tileH = H;
    tilingData->tileW = W;
    tilingData->tileTaskNum = N * C;
    tilingData->gradOutCacheDataNum = H * W;
}
} // namespace

class AveragePoolingGradTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        kernel_ut::SetupTestEnvironment(
            "experimental/pooling/average_pooling_grad/tests/ut/op_kernel/average_pooling_grad_data",
            "average_pooling_grad_data");
    }
    static void TearDownTestCase() {}
};

TEST_F(AveragePoolingGradTest, test_case_float32_1)
{
    kernel_ut::RunGenData("./average_pooling_grad_data", {"'(2, 1, 4, 6)'", "float32"});

    size_t dataByteSize = TOTAL_NUM * sizeof(float);
    const std::string dyFile = "./average_pooling_grad_data/float32_dy_t_average_pooling_grad.bin";
    uint8_t* dyGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(dataByteSize));
    ReadFile(dyFile, dataByteSize, dyGm, dataByteSize);

    uint8_t* dxGm = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(dataByteSize));
    uint8_t* ws = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(32));
    uint8_t* tl = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sizeof(AveragePoolingGradTilingData)));
    InitTilingData(reinterpret_cast<AveragePoolingGradTilingData*>(tl));

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto kf = [](GM_ADDR dy, GM_ADDR dx, GM_ADDR w, GM_ADDR tiling) {
        ::average_pooling_grad(nullptr, dy, dx, w, tiling);
    };
    ICPU_RUN_KF(kf, 1, dyGm, dxGm, ws, tl);

    const std::string outFile = "./average_pooling_grad_data/float32_output_dx_t_average_pooling_grad.bin";
    WriteFile(outFile, dxGm, dataByteSize);

    AscendC::GmFree(reinterpret_cast<void*>(dyGm));
    AscendC::GmFree(reinterpret_cast<void*>(dxGm));
    AscendC::GmFree(reinterpret_cast<void*>(ws));
    AscendC::GmFree(reinterpret_cast<void*>(tl));

    kernel_ut::RunCompareData("./average_pooling_grad_data", {"float32"});
}
