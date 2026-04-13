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
#include "../../../op_kernel/elu_grad.cpp"
#include "../../../op_kernel/elu_grad_tiling_data.h"
#include <cstdint>

using namespace std;

class elu_grad_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "elu_grad_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "elu_grad_test TearDown\n" << endl;
    }
};

TEST_F(elu_grad_test, test_case_0)
{
    size_t gradsByteSize = 32 * 4 * 4 * 4 * sizeof(float);
    size_t activationsByteSize = 32 * 4 * 4 * 4 * sizeof(float);
    size_t yByteSize = 32 * 4 * 4 * 4 * sizeof(float);
    size_t tiling_data_size = sizeof(EluGradTilingData);
    uint32_t blockDim = 1;

    uint8_t* grads = (uint8_t*)AscendC::GmAlloc(gradsByteSize);
    uint8_t* activations = (uint8_t*)AscendC::GmAlloc(activationsByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    EluGradTilingData* tilingDatafromBin = reinterpret_cast<EluGradTilingData*>(tiling);

    tilingDatafromBin->smallCoreDataNum = 2048;
    tilingDatafromBin->bigCoreDataNum = 2112;
    tilingDatafromBin->tileDataNum = 4032;
    tilingDatafromBin->smallTailDataNum = 2048;
    tilingDatafromBin->bigTailDataNum = 2112;
    tilingDatafromBin->finalSmallTileNum = 1;
    tilingDatafromBin->finalBigTileNum = 1;
    tilingDatafromBin->tailBlockNum = 0;
    tilingDatafromBin->bufferOpen = 1;

    auto EluGradKernel = [](GM_ADDR grads, GM_ADDR activations, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
        ::elu_grad<0>(grads, activations, y, workspace, tiling);
    };

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(EluGradKernel, blockDim, grads, activations, y, workspace, (uint8_t *)(tilingDatafromBin));

    AscendC::GmFree(grads);
    AscendC::GmFree(activations);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}