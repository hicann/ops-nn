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

extern "C" __global__ __aicore__ void sync_batch_norm_backward_reduce(
    GM_ADDR sumDy, GM_ADDR sumDyDxPad, GM_ADDR mean, GM_ADDR invertStd, GM_ADDR sumDyXmu, GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tiling);

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

struct SyncBatchNormBackwardReduceTilingData {
    EleBaseTilingData baseTiling;
};

class sync_batch_norm_backward_reduce_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "sync_batch_norm_backward_reduce_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "sync_batch_norm_backward_reduce_test TearDown\n" << endl;
    }
};
 
TEST_F(sync_batch_norm_backward_reduce_test, test_case_fp32_1)
{
    size_t dataLen = 100;
    size_t sumDyByteSize = dataLen * sizeof(float);
    size_t sumDyDxPadByteSize = dataLen * sizeof(float);
    size_t meanByteSize = dataLen * sizeof(float);
    size_t invertStdByteSize = dataLen * sizeof(float);
    size_t sumDyXmuByteSize = dataLen * sizeof(float);
    size_t yByteSize = dataLen * sizeof(float);
    size_t tilingDataSize = sizeof(SyncBatchNormBackwardReduceTilingData);

    uint8_t* sumDy = (uint8_t*)AscendC::GmAlloc(sumDyByteSize);
    uint8_t* sumDyDxPad = (uint8_t*)AscendC::GmAlloc(sumDyDxPadByteSize);
    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(meanByteSize);
    uint8_t* invertStd = (uint8_t*)AscendC::GmAlloc(invertStdByteSize);
    uint8_t* sumDyXmu = (uint8_t*)AscendC::GmAlloc(sumDyXmuByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16*1024*1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 1;

    SyncBatchNormBackwardReduceTilingData* tilingDatafromBin = 
        reinterpret_cast<SyncBatchNormBackwardReduceTilingData*>(tiling);

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
    ICPU_RUN_KF(sync_batch_norm_backward_reduce, blockDim, sumDy, sumDyDxPad, mean, invertStd, sumDyXmu, y,
                workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(sumDy);
    AscendC::GmFree(sumDyDxPad);
    AscendC::GmFree(mean);
    AscendC::GmFree(invertStd);
    AscendC::GmFree(sumDyXmu);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
