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
#include "../../../op_kernel/soft_margin_loss_grad.cpp"
#include "../../../op_kernel/soft_margin_loss_grad_tiling_data.h"
#include <cstdint>

using namespace std;

class soft_margin_loss_grad_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "soft_margin_loss_grad_test SetUp\n" << endl; }
    static void TearDownTestCase() { cout << "soft_margin_loss_grad_test TearDown\n" << endl; }
};

TEST_F(soft_margin_loss_grad_test, test_case_0)
{
    size_t xByteSize = 32 * 4 * 4 * 4 * sizeof(float);
    size_t yByteSize = 32 * 4 * 4 * 4 * sizeof(float);
    size_t grad_outByteSize = 32 * 4 * 4 * 4 * sizeof(float);
    size_t outByteSize = 32 * 4 * 4 * 4 * sizeof(float);
    size_t tiling_data_size = sizeof(SoftMarginLossGradTilingData);
    uint32_t blockDim = 1;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* grad_out = (uint8_t*)AscendC::GmAlloc(grad_outByteSize);
    uint8_t* out = (uint8_t*)AscendC::GmAlloc(outByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    SoftMarginLossGradTilingData* tilingDatafromBin = reinterpret_cast<SoftMarginLossGradTilingData*>(tiling);

    tilingDatafromBin->smallCoreDataNum = 2048;
    tilingDatafromBin->bigCoreDataNum = 2112;
    tilingDatafromBin->tileDataNum = 4032;
    tilingDatafromBin->smallTailDataNum = 2048;
    tilingDatafromBin->bigTailDataNum = 2112;
    tilingDatafromBin->finalSmallTileNum = 1;
    tilingDatafromBin->finalBigTileNum = 1;
    tilingDatafromBin->tailBlockNum = 0;
    tilingDatafromBin->cof = 1.0f;

    auto SoftMarginLossGradKernel = [](GM_ADDR x, GM_ADDR y, GM_ADDR grad_out, GM_ADDR out, GM_ADDR workspace,
                                       GM_ADDR tiling) {
        ::soft_margin_loss_grad<0>(x, y, grad_out, out, workspace, tiling);
    };

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(SoftMarginLossGradKernel, blockDim, x, y, grad_out, out, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(grad_out);
    AscendC::GmFree(out);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}