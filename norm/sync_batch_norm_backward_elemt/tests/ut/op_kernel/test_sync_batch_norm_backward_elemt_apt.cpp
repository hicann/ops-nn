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
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"

using namespace std;

extern "C" __global__ __aicore__ void sync_batch_norm_backward_elemt(
    GM_ADDR grad_output, GM_ADDR save_input, GM_ADDR mean, GM_ADDR invstd, GM_ADDR weight, GM_ADDR mean_dy,
    GM_ADDR mean_dy_xmu, GM_ADDR grad_input, GM_ADDR workspace, GM_ADDR tiling);

struct EleBaseTilingData {
    int64_t dim0;
    int32_t coreNum;
    int32_t ubFormer;
    int64_t blockFormer;
    int64_t blockNum;
    int64_t ubLoopOfFormerBlock;
    int64_t ubLoopOfTailBlock;
    int64_t ubTailOfFormerBlock;
    int64_t ubTailOfTailBlock;
    int64_t elemNum;
    uint64_t scheMode;
};

struct SyncBatchNormBackwardElemtTilingData {
    EleBaseTilingData baseTiling;
};

class sync_batch_norm_backward_elemt_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "sync_batch_norm_backward_elemt_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "sync_batch_norm_backward_elemt_test TearDown\n" << endl;
    }
};
 	 
TEST_F(sync_batch_norm_backward_elemt_test, test_case_fp32_1)
{
    size_t dataLen = 100;
    size_t gradOutputByteSize = dataLen * sizeof(float);
    size_t saveInputByteSize = dataLen * sizeof(float);
    size_t meanByteSize = dataLen * sizeof(float);
    size_t invstdByteSize = dataLen * sizeof(float);
    size_t weightByteSize = dataLen * sizeof(float);
    size_t meanDyByteSize = dataLen * sizeof(float);
    size_t meanDyXmuByteSize = dataLen * sizeof(float);
    size_t gradInputByteSize = dataLen * sizeof(float);
    size_t tilingDataSize = sizeof(SyncBatchNormBackwardElemtTilingData);

    uint8_t* grad_output = (uint8_t*)AscendC::GmAlloc(gradOutputByteSize);
    uint8_t* save_input = (uint8_t*)AscendC::GmAlloc(saveInputByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(meanByteSize);
    uint8_t* invstd = (uint8_t*)AscendC::GmAlloc(invstdByteSize);
    uint8_t* weight = (uint8_t*)AscendC::GmAlloc(weightByteSize);
    uint8_t* mean_dy = (uint8_t*)AscendC::GmAlloc(meanDyByteSize);
    uint8_t* mean_dy_xmu = (uint8_t*)AscendC::GmAlloc(meanDyXmuByteSize);
    uint8_t* grad_input = (uint8_t*)AscendC::GmAlloc(gradInputByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16*1024*1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 1;

    SyncBatchNormBackwardElemtTilingData* tilingDatafromBin = 
        reinterpret_cast<SyncBatchNormBackwardElemtTilingData*>(tiling);

    tilingDatafromBin->baseTiling.dim0 = 100;
    tilingDatafromBin->baseTiling.coreNum = 1;
    tilingDatafromBin->baseTiling.ubFormer = 512;
    tilingDatafromBin->baseTiling.blockFormer = 100;
    tilingDatafromBin->baseTiling.blockNum = 1;
    tilingDatafromBin->baseTiling.ubLoopOfFormerBlock = 1;
    tilingDatafromBin->baseTiling.ubLoopOfTailBlock = 1;
    tilingDatafromBin->baseTiling.ubTailOfFormerBlock = 100;
    tilingDatafromBin->baseTiling.ubTailOfTailBlock = 100;
    tilingDatafromBin->baseTiling.elemNum = 100;
    tilingDatafromBin->baseTiling.scheMode = 1;

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(sync_batch_norm_backward_elemt, blockDim, grad_output, save_input, mean, invstd, weight, mean_dy,
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
}
