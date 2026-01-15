/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
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
#include "../../../op_kernel/mish_grad.cpp"
#include "../../../op_kernel/mish_grad_tiling_data.h"
#include <cstdint>

using namespace std;

class mish_grad_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "mish_grad_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "mish_grad_test TearDown\n" << endl;
    }
};

TEST_F(mish_grad_test, test_case_0)
{
    size_t gradByteSize = 32 * 4 * 4 * 4 * sizeof(float);
    size_t xByteSize = 32 * 4 * 4 * 4 * sizeof(float);
    size_t x_gradByteSize = 32 * 4 * 4 * 4 * sizeof(float);
    size_t tiling_data_size = sizeof(MishGradTilingData);
    uint32_t blockDim = 1;

    uint8_t* grad = (uint8_t*)AscendC::GmAlloc(gradByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* tanhx = nullptr;
    uint8_t* x_grad = (uint8_t*)AscendC::GmAlloc(x_gradByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    MishGradTilingData* tilingDatafromBin = reinterpret_cast<MishGradTilingData*>(tiling);

    tilingDatafromBin->smallCoreDataNum = 2048;
    tilingDatafromBin->bigCoreDataNum = 2112;
    tilingDatafromBin->tileDataNum = 4032;
    tilingDatafromBin->smallTailDataNum = 2048;
    tilingDatafromBin->bigTailDataNum = 2112;
    tilingDatafromBin->finalSmallTileNum = 1;
    tilingDatafromBin->finalBigTileNum = 1;
    tilingDatafromBin->tailBlockNum = 0;
    tilingDatafromBin->haveTanhx = 0;

    auto KernelMishGrad = [](GM_ADDR grad, GM_ADDR x, GM_ADDR tanhx, GM_ADDR x_grad, GM_ADDR workspace, GM_ADDR tiling) {
        ::mish_grad<0>(grad, x, tanhx, x_grad, workspace, tiling);
    };

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(KernelMishGrad, blockDim, grad, x, tanhx, x_grad, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(grad);
    AscendC::GmFree(x);
    AscendC::GmFree(x_grad);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}