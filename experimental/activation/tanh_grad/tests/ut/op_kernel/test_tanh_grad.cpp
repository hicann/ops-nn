/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Zhou Jianhua <@LePenseur>
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
#include "../../../op_kernel/tanh_grad.cpp"
#include "../../../op_kernel/tanh_grad_tiling_data.h"

using namespace std;

class TanhGradTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        kernel_ut::SetupTestEnvironment("experimental/activation/tanh_grad/tests/ut/op_kernel/tanh_grad_data",
                                        "tanh_grad_data");
    }
    static void TearDownTestCase() {}
};

// ============================================================================
// float32 test
// ============================================================================

TEST_F(TanhGradTest, test_case_float32_1)
{
    kernel_ut::RunGenData("./tanh_grad_data", {"'(256, 32)'", "float32"});

    uint32_t totalNum = 256 * 32; // 8192
    size_t inputByteSize = totalNum * sizeof(float);

    std::string yFile = "./tanh_grad_data/float32_y_t_tanh_grad.bin";
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    ReadFile(yFile, inputByteSize, y, inputByteSize);

    std::string dyFile = "./tanh_grad_data/float32_dy_t_tanh_grad.bin";
    uint8_t* dy = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    ReadFile(dyFile, inputByteSize, dy, inputByteSize);

    size_t outputByteSize = totalNum * sizeof(float);
    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* ws = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tl = (uint8_t*)AscendC::GmAlloc(sizeof(TanhGradTilingData));

    TanhGradTilingData* tilingData = reinterpret_cast<TanhGradTilingData*>(tl);
    tilingData->smallCoreDataNum = 8192;
    tilingData->bigCoreDataNum = 0;
    tilingData->finalBigTileNum = 0;
    tilingData->finalSmallTileNum = 4;
    tilingData->tileDataNum = 2048;
    tilingData->smallTailDataNum = 2048;
    tilingData->bigTailDataNum = 0;
    tilingData->tailBlockNum = 0;

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto kf = [](GM_ADDR y, GM_ADDR dy, GM_ADDR dx, GM_ADDR w, GM_ADDR tl) { ::tanh_grad<0>(y, dy, dx, w, tl); };
    ICPU_RUN_KF(kf, 1, y, dy, dx, ws, tl);

    std::string outFile = "./tanh_grad_data/float32_output_t_tanh_grad.bin";
    WriteFile(outFile, dx, outputByteSize);

    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)dy);
    AscendC::GmFree((void*)dx);
    AscendC::GmFree((void*)ws);
    AscendC::GmFree((void*)tl);

    kernel_ut::RunCompareData("./tanh_grad_data", {"float32"});
}
