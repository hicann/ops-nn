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

#include <array>
#include <vector>
#include "gtest/gtest.h"
#include <cstdint>

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#include "string.h"
#include <string>
#include "kernel_ut_data_helper.h"
#include "kernel_ut_data_executor.h"
#endif
#include "../../../op_kernel/softmax_grad_tiling_data.h"

__global__ __aicore__ void softmax_grad(GM_ADDR y, GM_ADDR dy, GM_ADDR dx, GM_ADDR workspace, GM_ADDR tiling);

using namespace std;

class SoftmaxGradTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        kernel_ut::SetupTestEnvironment(
            "experimental/loss/softmax_grad/tests/ut/op_kernel/softmax_grad_data",
            "softmax_grad_data");
    }
    static void TearDownTestCase() {}
};

// ============================================================================
// float32 测试用例 (N=4, C=8, CPadded=8)
// ============================================================================

TEST_F(SoftmaxGradTest, test_case_float32_1)
{
    // 1. 数据生成
    kernel_ut::RunGenData("./softmax_grad_data", {"'(4, 8)'", "float32"});

    // 2. 申请内存并加载输入
    uint32_t N = 4;
    uint32_t C = 8;
    uint32_t totalNum = N * C; // 32
    size_t inputByteSize = totalNum * sizeof(float);

    std::string yFile = "./softmax_grad_data/float32_y_t_softmax_grad.bin";
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    ReadFile(yFile, inputByteSize, y, inputByteSize);

    std::string dyFile = "./softmax_grad_data/float32_dy_t_softmax_grad.bin";
    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    ReadFile(dyFile, inputByteSize, dy, inputByteSize);

    size_t outputByteSize = totalNum * sizeof(float);
    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(outputByteSize);

    uint8_t* ws = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tl = (uint8_t*)AscendC::GmAlloc(sizeof(SoftmaxGradTilingData));

    // 3. 构造 TilingData (float32: CPadded=8, usedCoreNum=1)
    SoftmaxGradTilingData* tilingData = reinterpret_cast<SoftmaxGradTilingData*>(tl);
    tilingData->N = N;
    tilingData->C = C;
    tilingData->CPadded = 8;
    tilingData->rowsPerCore = 4;
    tilingData->tailCoreRows = 0;
    tilingData->usedCoreNum = 1;

    // 4. 执行 kernel
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto kf = [](GM_ADDR y, GM_ADDR dy, GM_ADDR dx, GM_ADDR w, GM_ADDR tl) {
        ::softmax_grad(y, dy, dx, w, tl);
    };
    ICPU_RUN_KF(kf, 1, y, dy, dx, ws, tl);

    // 5. 写出结果并比对
    std::string outFile = "./softmax_grad_data/float32_output_dx_t_softmax_grad.bin";
    WriteFile(outFile, dx, outputByteSize);

    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)dy);
    AscendC::GmFree((void*)dx);
    AscendC::GmFree((void*)ws);
    AscendC::GmFree((void*)tl);

    kernel_ut::RunCompareData("./softmax_grad_data", {"float32"});
}
