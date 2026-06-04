/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#include "string.h"
#include <iostream>
#include <string>
#endif
#include "../../../op_kernel/elu_v2.h"
#include <cstdint>

extern "C" __global__ __aicore__ void elu_v2(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

using namespace std;

class elu_v2_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "elu_v2_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "elu_v2_test TearDown\n" << endl;
    }
};

TEST_F(elu_v2_test, test_case_0)
{
    size_t xByteSize = 32 * 4 * 4 * 4 * sizeof(float);
    size_t yByteSize = 32 * 4 * 4 * 4 * sizeof(float);
    size_t tilingDataSize = sizeof(EluV2TilingData);
    uint32_t blockDim = 1;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    auto* tilingDataFromBin = reinterpret_cast<EluV2TilingData*>(tiling);
    tilingDataFromBin->bigCoreDataNum = 2048;
    tilingDataFromBin->smallCoreDataNum = 2048;
    tilingDataFromBin->tileDataNum = 2048;
    tilingDataFromBin->bigCoreNum = 1;
    tilingDataFromBin->alpha = 0.5f;
    tilingDataFromBin->scale = 1.5f;
    tilingDataFromBin->inputScale = 2.0f;


    auto eluV2Kernel = elu_v2;

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(eluV2Kernel, blockDim, x, y, workspace, (uint8_t*)(tilingDataFromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
