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
#include "../../../op_kernel/mse_loss_grad.cpp"
#include "../../../op_kernel/mse_loss_grad_tiling_data.h"
#include <cstdint>

using namespace std;

class mse_loss_grad_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "mse_loss_grad_test SetUp\n" << endl; }
    static void TearDownTestCase() { cout << "mse_loss_grad_test TearDown\n" << endl; }
};

TEST_F(mse_loss_grad_test, test_case_0)
{
    size_t inputPredictByteSize = 128 * sizeof(float);
    size_t inputLabelByteSize = 128 * sizeof(float);
    size_t inputDoutByteSize = 128 * sizeof(float);
    size_t outputYByteSize = 128 * sizeof(float);
    size_t tilingDataSize = sizeof(MseLossGradTilingData);

    uint8_t* predict = (uint8_t*)AscendC::GmAlloc(inputPredictByteSize);
    uint8_t* label = (uint8_t*)AscendC::GmAlloc(inputLabelByteSize);
    uint8_t* dout = (uint8_t*)AscendC::GmAlloc(inputDoutByteSize);

    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputYByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 1;

    char* path_ = get_current_dir_name();
    string path(path_);

    MseLossGradTilingData* tilingDatafromBin = reinterpret_cast<MseLossGradTilingData*>(tiling);
    tilingDatafromBin->smallCoreDataNum = 128;
    tilingDatafromBin->bigCoreDataNum = 256;
    tilingDatafromBin->finalBigTileNum = 2;
    tilingDatafromBin->finalSmallTileNum = 1;
    tilingDatafromBin->tileDataNum = 128;
    tilingDatafromBin->smallTailDataNum = 128;
    tilingDatafromBin->bigTailDataNum = 128;
    tilingDatafromBin->tailBlockNum = 0;
    tilingDatafromBin->cof = 2.0;
    tilingDatafromBin->usedDb = 0;

    auto KernelMseLossGrad = [](GM_ADDR predict, GM_ADDR label, GM_ADDR dout, GM_ADDR y, GM_ADDR workspace,
                                GM_ADDR tiling) { ::mse_loss_grad<0>(predict, label, dout, y, workspace, tiling); };

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(mse_loss_grad<0>, blockDim, predict, label, dout, y, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(predict);
    AscendC::GmFree(label);
    AscendC::GmFree(dout);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}