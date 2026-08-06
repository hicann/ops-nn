/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_sync_batch_norm_backward_elemt.cpp
 * \brief
 */

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#include "string.h"
#include <iostream>
#include <string>
#endif
#include <gtest/gtest.h> // 添加 GTest 头文件
#include "../../../op_kernel/sync_batch_norm_backward_elemt.cpp"
#include "../../../op_kernel/sync_batch_norm_backward_elemt_tilingdata.h"
#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void sync_batch_norm_backward_elemt(GM_ADDR grad_output, GM_ADDR save_input,
                                                                     GM_ADDR mean, GM_ADDR invstd, GM_ADDR weight,
                                                                     GM_ADDR mean_dy, GM_ADDR mean_dy_xmu,
                                                                     GM_ADDR grad_input, GM_ADDR workspace,
                                                                     GM_ADDR tiling);

class sync_batch_norm_backward_elemt_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "sync_batch_norm_backward_elemt_test SetUp\n" << endl; }
    static void TearDownTestCase() { cout << "sync_batch_norm_backward_elemt_test TearDown\n" << endl; }
};

TEST_F(sync_batch_norm_backward_elemt_test, test_case_001)
{
    size_t dataCount = 320; // 稍大于 bigCoreDataNum=256
    size_t inputGradOutputByteSize = dataCount * sizeof(float);
    size_t inputSaveInputByteSize = dataCount * sizeof(float);
    size_t inputMeanByteSize = dataCount * sizeof(float);
    size_t inputInvstdByteSize = dataCount * sizeof(float);
    size_t inputWeightByteSize = dataCount * sizeof(float);
    size_t inputMeanDyByteSize = dataCount * sizeof(float);
    size_t inputMeanDyXmuByteSize = dataCount * sizeof(float);

    size_t outputGradInputByteSize = dataCount * sizeof(float);
    size_t tilingDataSize = sizeof(SyncBatchNormBackwardElemtTilingData);

    uint8_t* grad_output = (uint8_t*)AscendC::GmAlloc(inputGradOutputByteSize);
    uint8_t* save_input = (uint8_t*)AscendC::GmAlloc(inputSaveInputByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(inputMeanByteSize);
    uint8_t* invstd = (uint8_t*)AscendC::GmAlloc(inputInvstdByteSize);
    uint8_t* weight = (uint8_t*)AscendC::GmAlloc(inputWeightByteSize);
    uint8_t* mean_dy = (uint8_t*)AscendC::GmAlloc(inputMeanDyByteSize);
    uint8_t* mean_dy_xmu = (uint8_t*)AscendC::GmAlloc(inputMeanDyXmuByteSize);

    uint8_t* grad_input = (uint8_t*)AscendC::GmAlloc(outputGradInputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16); // 修改变量名为 workspace
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 1;

    char* path_ = get_current_dir_name();
    string path(path_);

    SyncBatchNormBackwardElemtTilingData* tilingDatafromBin = reinterpret_cast<SyncBatchNormBackwardElemtTilingData*>(
        tiling);
    tilingDatafromBin->smallCoreDataNum = 128;
    tilingDatafromBin->bigCoreDataNum = 256;
    tilingDatafromBin->finalBigTileNum = 2;
    tilingDatafromBin->finalSmallTileNum = 1;
    tilingDatafromBin->tileDataNum = 128;
    tilingDatafromBin->smallTailDataNum = 128;
    tilingDatafromBin->bigTailDataNum = 128;
    tilingDatafromBin->tailBlockNum = 0;
    tilingDatafromBin->usedDb = 0;

    auto KernelSyncBatchNormBackwardElemt = [](GM_ADDR grad_output, GM_ADDR save_input, GM_ADDR mean, GM_ADDR invstd,
                                               GM_ADDR weight, GM_ADDR mean_dy, GM_ADDR mean_dy_xmu, GM_ADDR grad_input,
                                               GM_ADDR workspace, GM_ADDR tiling) {
        ::sync_batch_norm_backward_elemt<0>(grad_output, save_input, mean, invstd, weight, mean_dy, mean_dy_xmu,
                                            grad_input, workspace, tiling);
    };

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(sync_batch_norm_backward_elemt<0>, blockDim, grad_output, save_input, mean, invstd, weight, mean_dy,
                mean_dy_xmu, grad_input, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(grad_output);
    AscendC::GmFree(save_input);
    AscendC::GmFree(mean);
    AscendC::GmFree(invstd);
    AscendC::GmFree(weight);
    AscendC::GmFree(mean_dy);
    AscendC::GmFree(mean_dy_xmu);
    AscendC::GmFree(grad_input);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}
