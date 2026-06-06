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
#include "../../../op_kernel/kl_div_loss_grad.h"
#include <cstdint>

using namespace std;

class kl_div_loss_grad_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "kl_div_loss_grad_test SetUp\n" << endl; }
    static void TearDownTestCase() { cout << "kl_div_loss_grad_test TearDown\n" << endl; }
};

TEST_F(kl_div_loss_grad_test, test_case_0)
{
    size_t gradByteSize = 32 * 4 * 4 * 4 * sizeof(float);
    size_t inputByteSize = 32 * 4 * 4 * 4 * sizeof(float);
    size_t targetByteSize = 32 * 4 * 4 * 4 * sizeof(float);
    size_t yByteSize = 32 * 4 * 4 * 4 * sizeof(float);
    size_t tilingDataSize = sizeof(KlDivLossGradTilingData);
    uint32_t blockDim = 1;

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(gradByteSize);
    uint8_t* input = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* target = (uint8_t*)AscendC::GmAlloc(targetByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    auto* tilingDataFromBin = reinterpret_cast<KlDivLossGradTilingData*>(tiling);
    tilingDataFromBin->bigCoreDataNum = 2048;
    tilingDataFromBin->smallCoreDataNum = 2048;
    tilingDataFromBin->tileDataNum = 2048;
    tilingDataFromBin->bigCoreNum = 1;
    tilingDataFromBin->coff = 1.0f;

    auto klDivLossGradKernel = kl_div_loss_grad<false, false>;

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(klDivLossGradKernel, blockDim, grad, input, target, y, workspace, (uint8_t*)(tilingDataFromBin));

    AscendC::GmFree(grad);
    AscendC::GmFree(input);
    AscendC::GmFree(target);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
