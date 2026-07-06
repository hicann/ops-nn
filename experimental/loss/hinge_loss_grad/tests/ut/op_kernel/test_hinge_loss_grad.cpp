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
#include "../../../op_kernel/hinge_loss_grad_tiling_data.h"

extern "C" __global__ __aicore__ void hinge_loss_grad(GM_ADDR predict, GM_ADDR target, GM_ADDR gradOutput,
                                                       GM_ADDR gradInput, GM_ADDR workspace, GM_ADDR tiling);

using namespace std;

class HingeLossGradTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        kernel_ut::SetupTestEnvironment(
            "experimental/loss/hinge_loss_grad/tests/ut/op_kernel/hinge_loss_grad_data",
            "hinge_loss_grad_data");
    }
    static void TearDownTestCase() {}
};

// ============================================================================
// float32 测试用例
// ============================================================================

TEST_F(HingeLossGradTest, test_case_float32_1)
{
    // 1. 数据生成
    kernel_ut::RunGenData("./hinge_loss_grad_data", {"'(256, 32)'", "float32"});

    // 2. 申请内存并加载输入
    uint32_t totalNum = 256 * 32; // 8192
    size_t inputByteSize = totalNum * sizeof(float);

    std::string predFile = "./hinge_loss_grad_data/float32_predict_t_hinge_loss_grad.bin";
    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    ReadFile(predFile, inputByteSize, x1, inputByteSize);

    std::string targetFile = "./hinge_loss_grad_data/float32_target_t_hinge_loss_grad.bin";
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    ReadFile(targetFile, inputByteSize, x2, inputByteSize);

    std::string gradOutputFile = "./hinge_loss_grad_data/float32_grad_output_t_hinge_loss_grad.bin";
    uint8_t* x3 = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    ReadFile(gradOutputFile, inputByteSize, x3, inputByteSize);

    size_t outputByteSize = totalNum * sizeof(float);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);

    uint8_t* ws = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tl = (uint8_t*)AscendC::GmAlloc(sizeof(HingeLossGradTilingData));

    // 3. 手动构造 TilingData（float32: 4B/elem, UB_NUM_FLOAT=15, tileDataNum≈2048）
    HingeLossGradTilingData* tilingData = reinterpret_cast<HingeLossGradTilingData*>(tl);
    tilingData->smallCoreDataNum = 8192;
    tilingData->bigCoreDataNum = 0;
    tilingData->tileDataNum = 2048;
    tilingData->smallTailDataNum = 2048;
    tilingData->bigTailDataNum = 0;
    tilingData->finalSmallTileNum = 4;
    tilingData->finalBigTileNum = 0;
    tilingData->tailBlockNum = 0;

    // 4. 执行 kernel
    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto kf = [](GM_ADDR p, GM_ADDR t, GM_ADDR go, GM_ADDR gi, GM_ADDR ws, GM_ADDR tl) {
        ::hinge_loss_grad(p, t, go, gi, ws, tl);
    };
    ICPU_RUN_KF(kf, 1, x1, x2, x3, y, ws, tl);

    // 5. 写出结果并比对
    std::string outFile = "./hinge_loss_grad_data/float32_output_grad_input_t_hinge_loss_grad.bin";
    WriteFile(outFile, y, outputByteSize);

    AscendC::GmFree((void*)x1);
    AscendC::GmFree((void*)x2);
    AscendC::GmFree((void*)x3);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)ws);
    AscendC::GmFree((void*)tl);

    kernel_ut::RunCompareData("./hinge_loss_grad_data", {"float32"});
}
