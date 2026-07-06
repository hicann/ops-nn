/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <iostream>
#include "gtest/gtest.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#endif
#include "../../../op_kernel/prelu.cpp"
#include "../../../op_kernel/prelu_tiling_data.h"
#include "../../../op_kernel/prelu_tiling_key.h"

using namespace std;

class prelu_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "prelu_test SetUp\n" << endl; }
    static void TearDownTestCase() { cout << "prelu_test TearDown\n" << endl; }
};

TEST_F(prelu_test, scalar_float32_success)
{
    constexpr uint32_t dataCount = 64;
    size_t inputByteSize = dataCount * sizeof(float);
    size_t weightByteSize = sizeof(float);
    size_t outputByteSize = dataCount * sizeof(float);
    size_t tilingDataSize = sizeof(PreluTilingData);
    uint32_t blockDim = 1;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* weight = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    PreluTilingData* tilingData = reinterpret_cast<PreluTilingData*>(tiling);
    tilingData->totalLength = dataCount;
    tilingData->usedCoreNum = 1;
    tilingData->formerNum = 0;
    tilingData->formerLength = dataCount;
    tilingData->tailLength = dataCount;
    tilingData->tileLength = dataCount;

    auto PreluKernel = [](GM_ADDR x, GM_ADDR weight, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
        ::prelu<PRELU_TPL_SCALAR_MODE>(x, weight, y, workspace, tiling);
    };

    ICPU_SET_TILING_KEY(GET_TPL_TILING_KEY(PRELU_TPL_SCALAR_MODE));
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(PreluKernel, blockDim, x, weight, y, workspace, (uint8_t*)(tilingData));

    AscendC::GmFree(x);
    AscendC::GmFree(weight);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
