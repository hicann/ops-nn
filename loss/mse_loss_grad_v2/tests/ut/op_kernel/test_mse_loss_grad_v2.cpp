/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_mse_loss_grad_v2_tiling_def.h"
#include "data_utils.h"

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void mse_loss_grad_v2(
    GM_ADDR predict, GM_ADDR label, GM_ADDR dout, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

class mse_loss_grad_v2_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "mse_loss_grad_v2 SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "mse_loss_grad_v2_test TearDown\n" << endl;
    }
};

TEST_F(mse_loss_grad_v2_test, test_case_mode_1_fp32_001)
{
    size_t inputPredictByteSize = 32 * 64 * sizeof(float);
    size_t inputLabelByteSize = 32 * 64 * sizeof(float);
    size_t inputDoutByteSize = 32 * 64 * sizeof(float);
    size_t outputYByteSize = 32 * 64 * sizeof(float);
    size_t tilingDataSize = sizeof(MseLossGradV2TilingData);

    uint8_t* predict = (uint8_t*)AscendC::GmAlloc(inputPredictByteSize);
    uint8_t* label = (uint8_t*)AscendC::GmAlloc(inputLabelByteSize);
    uint8_t* dout = (uint8_t*)AscendC::GmAlloc(inputDoutByteSize);

    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputYByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 1;

    char* path_ = get_current_dir_name();
    string path(path_);

    MseLossGradV2TilingData* tilingDatafromBin = reinterpret_cast<MseLossGradV2TilingData*>(tiling);

    tilingDatafromBin->cof = 2.0;
    tilingDatafromBin->totalLength = 2048;
    tilingDatafromBin->tileNum = 1;
    tilingDatafromBin->padLength = 0;
    tilingDatafromBin->blockLength = 2048;
    tilingDatafromBin->usedDb = 0;

    ICPU_SET_TILING_KEY(1);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(mse_loss_grad_v2, blockDim, predict, label, dout, y, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(predict);
    AscendC::GmFree(label);
    AscendC::GmFree(dout);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}