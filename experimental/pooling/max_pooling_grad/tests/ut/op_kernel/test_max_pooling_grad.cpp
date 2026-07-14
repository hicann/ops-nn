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
#include "../../../op_kernel/max_pooling_grad_tiling_data.h"

// #1 改 DTYPE_DY 宏后 kernel 为普通函数（非 template）：前向声明 + 链接构建系统编译的
// tbe/ascendc/max_pooling_grad.cpp.o，不再 #include kernel.cpp（否则两者重复定义 max_pooling_grad）。
// 参照 average_pooling_grad test。
__global__ __aicore__ void max_pooling_grad(GM_ADDR dy, GM_ADDR x, GM_ADDR y, GM_ADDR dx, GM_ADDR workspace,
                                            GM_ADDR tiling);

using namespace std;

class MaxPoolingGradTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        kernel_ut::SetupTestEnvironment(
            "experimental/pooling/max_pooling_grad/tests/ut/op_kernel/max_pooling_grad_data", "max_pooling_grad_data");
    }
    static void TearDownTestCase() {}
};

TEST_F(MaxPoolingGradTest, test_case_float16_1)
{
    // 1. 数据生成（FP16 对齐后 128 元素）
    kernel_ut::RunGenData("./max_pooling_grad_data", {"'(2, 1, 4, 6)'", "float16"});

    // 2. 申请内存并加载输入（对齐后 128 元素）
    uint32_t alignedNum = 128; // smallCoreDataNum for [2,1,4,6] fp16
    size_t inputByteSize = alignedNum * sizeof(half);

    std::string dyFile = "./max_pooling_grad_data/float16_dy_t_max_pooling_grad.bin";
    uint8_t* dyGm = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    ReadFile(dyFile, inputByteSize, dyGm, inputByteSize);

    std::string xFile = "./max_pooling_grad_data/float16_x_t_max_pooling_grad.bin";
    uint8_t* xGm = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    ReadFile(xFile, inputByteSize, xGm, inputByteSize);

    std::string yFile = "./max_pooling_grad_data/float16_y_t_max_pooling_grad.bin";
    uint8_t* yGm = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    ReadFile(yFile, inputByteSize, yGm, inputByteSize);

    uint32_t dxNum = alignedNum;
    size_t outputByteSize = dxNum * sizeof(half);
    uint8_t* dxGm = (uint8_t*)AscendC::GmAlloc(outputByteSize);

    uint8_t* ws = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tl = (uint8_t*)AscendC::GmAlloc(sizeof(MaxPoolingGradTilingData));

    // 3. 构造 TilingData
    MaxPoolingGradTilingData* tilingData = reinterpret_cast<MaxPoolingGradTilingData*>(tl);
    tilingData->smallCoreDataNum = 128;
    tilingData->bigCoreDataNum = 256;
    tilingData->ubPartDataNum = 128;
    tilingData->smallCoreTailDataNum = 128;
    tilingData->bigCoreTailDataNum = 256;
    tilingData->smallCoreLoopNum = 1;
    tilingData->bigCoreLoopNum = 1;
    tilingData->tailBlockNum = 0;
    tilingData->lastCoreValidDataNum = 48;

    // 4. 执行 kernel
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    // #4: 钳位后单核只写有效区 [0:48)，padding 区 [48:128) 未写；
    // memset 使其=0=golden padding，保持整 128 元素比对通过（golden 在 padding 区亦为 0）。
    memset(dxGm, 0, outputByteSize);
    auto kf = [](GM_ADDR dy, GM_ADDR x, GM_ADDR y, GM_ADDR dx, GM_ADDR ws, GM_ADDR tl) {
        ::max_pooling_grad(dy, x, y, dx, ws, tl);
    };
    ICPU_RUN_KF(kf, 1, dyGm, xGm, yGm, dxGm, ws, tl);

    // 5. 写出结果并比对
    std::string outFile = "./max_pooling_grad_data/float16_output_dx_t_max_pooling_grad.bin";
    WriteFile(outFile, dxGm, outputByteSize);

    AscendC::GmFree((void*)dyGm);
    AscendC::GmFree((void*)xGm);
    AscendC::GmFree((void*)yGm);
    AscendC::GmFree((void*)dxGm);
    AscendC::GmFree((void*)ws);
    AscendC::GmFree((void*)tl);

    kernel_ut::RunCompareData("./max_pooling_grad_data", {"float16"});
}
