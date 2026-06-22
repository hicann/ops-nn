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
#include "../../../op_kernel/apply_adagrad_d.cpp"
#include "../../../op_kernel/apply_adagrad_d_tiling_data.h"
#include <cstdint>

using namespace std;

class apply_adagrad_d_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "apply_adagrad_d_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "apply_adagrad_d_test TearDown\n" << endl;
    }
};

TEST_F(apply_adagrad_d_test, test_case_0)
{
    // inputs: var, accum, lr, grad; outputs: var_out, accum_out
    // shape: 32*4*4*4 float elements (2048 elements, 8192 bytes)
    size_t varByteSize   = 32 * 4 * 4 * 4 * sizeof(float);
    size_t accumByteSize = 32 * 4 * 4 * 4 * sizeof(float);
    size_t lrByteSize    = 32;  // lr is a scalar, 1 block
    size_t gradByteSize  = 32 * 4 * 4 * 4 * sizeof(float);
    size_t varOutByteSize   = varByteSize;
    size_t accumOutByteSize = accumByteSize;
    size_t tiling_data_size = sizeof(ApplyAdagradDTilingData);
    uint32_t blockDim = 1;

    uint8_t* var      = (uint8_t*)AscendC::GmAlloc(varByteSize);
    uint8_t* accum    = (uint8_t*)AscendC::GmAlloc(accumByteSize);
    uint8_t* lr       = (uint8_t*)AscendC::GmAlloc(lrByteSize);
    uint8_t* grad     = (uint8_t*)AscendC::GmAlloc(gradByteSize);
    uint8_t* var_out  = (uint8_t*)AscendC::GmAlloc(varOutByteSize);
    uint8_t* accum_out = (uint8_t*)AscendC::GmAlloc(accumOutByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling    = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    ApplyAdagradDTilingData* tilingDatafromBin = reinterpret_cast<ApplyAdagradDTilingData*>(tiling);

    // 2048 elements / 1 core, tileDataNum matches UB capacity
    tilingDatafromBin->smallCoreDataNum = 2048;
    tilingDatafromBin->bigCoreDataNum   = 2048;
    tilingDatafromBin->tileDataNum      = 2048;
    tilingDatafromBin->smallTailDataNum = 2048;
    tilingDatafromBin->bigTailDataNum   = 2048;
    tilingDatafromBin->finalSmallTileNum = 1;
    tilingDatafromBin->finalBigTileNum   = 1;
    tilingDatafromBin->tailBlockNum      = 0;
    tilingDatafromBin->totalDataNum      = 2048;
    tilingDatafromBin->updateSlots       = 1;  // true

    auto ApplyAdagradDKernel = [](GM_ADDR var, GM_ADDR accum, GM_ADDR lr, GM_ADDR grad,
                                   GM_ADDR var_out, GM_ADDR accum_out, GM_ADDR workspace, GM_ADDR tiling) {
        ::apply_adagrad_d<0>(var, accum, lr, grad, var_out, accum_out, workspace, tiling);
    };

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(ApplyAdagradDKernel, blockDim, var, accum, lr, grad, var_out, accum_out,
                workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(var);
    AscendC::GmFree(accum);
    AscendC::GmFree(lr);
    AscendC::GmFree(grad);
    AscendC::GmFree(var_out);
    AscendC::GmFree(accum_out);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}
