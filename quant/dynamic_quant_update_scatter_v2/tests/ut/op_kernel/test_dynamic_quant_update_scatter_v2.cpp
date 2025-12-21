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
#include "dynamic_quant_update_scatter_v2_tiling_def.h"
#include "data_utils.h"
#include <cstdint>

using namespace std;
using namespace AscendC;

extern "C" __global__ __aicore__ void dynamic_quant_update_scatter_v2(
    GM_ADDR x, GM_ADDR indices, GM_ADDR var, GM_ADDR varScale, GM_ADDR varOffset, GM_ADDR varOut, GM_ADDR varScaleOut,
    GM_ADDR varOffsetOut, GM_ADDR workSpace, GM_ADDR tiling);

class dynamic_quant_update_scatter_v2_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "dynamic_quant_update_scatter_v2_test SetUp\n" << endl;
        AscendC::SetKernelMode(KernelMode::AIV_MODE);
    }
    static void TearDownTestCase()
    {
        cout << "dynamic_quant_update_scatter_v2_test TearDown\n" << endl;
    }
};

// TEST_F(dynamic_quant_update_scatter_v2_test, test_case_0)
// {
//     size_t xByteSize = 192 * 1 * 14960 * sizeof(half);
//     // size_t xBf16ByteSize = 192 * 1 * 128 * sizeof(bfloat16_t);
//     size_t indicesByteSize = 192 * sizeof(uint32_t);
//     size_t varByteSize = 192 * 4 * 14960 * sizeof(int4b_t);
//     size_t varScaleByteSize = 192 * 4 * sizeof(float);
//     size_t varOffsetByteSize = 192 * 4 * sizeof(float);
//     size_t tiling_data_size = sizeof(DynamicQuantUpdateScatterV2TilingData);
//     uint32_t blockDim = 48;

//     uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
//     // uint8_t* xBf16 = (uint8_t*)AscendC::GmAlloc(xBf16ByteSize);
//     uint8_t* indices = (uint8_t*)AscendC::GmAlloc(indicesByteSize);
//     uint8_t* var = (uint8_t*)AscendC::GmAlloc(varByteSize);
//     uint8_t* varScale = (uint8_t*)AscendC::GmAlloc(varScaleByteSize);
//     uint8_t* varOffset = (uint8_t*)AscendC::GmAlloc(varOffsetByteSize);
//     uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
//     uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

//     char* path_ = get_current_dir_name();
//     string path(path_);

//     DynamicQuantUpdateScatterV2TilingData* tilingDatafromBin =
//         reinterpret_cast<DynamicQuantUpdateScatterV2TilingData*>(tiling);

//     tilingDatafromBin->coreNum = 48;
//     tilingDatafromBin->rowLen = 14960;
//     tilingDatafromBin->headCoreNum = 47;
//     tilingDatafromBin->rowPerHeadCore = 4;
//     tilingDatafromBin->rowPerTailCore = 4;
//     tilingDatafromBin->multiRowNumHeadCore = 1;
//     tilingDatafromBin->multiRowNumTailCore = 1;
//     tilingDatafromBin->batchSize = 192;
//     tilingDatafromBin->dstSeqLen = 4;

//     // ICPU_SET_TILING_KEY(0);
//     // ICPU_RUN_KF(dynamic_quant_update_scatter_v2, blockDim, xBf16, indices, var, varScale, varOffset, workSpace,
//     // (uint8_t*)(tilingDatafromBin));

//     ICPU_SET_TILING_KEY(1);
//     ICPU_RUN_KF(
//         dynamic_quant_update_scatter_v2, blockDim, x, indices, var, varScale, varOffset, var, varScale, varOffset,
//         workSpace, (uint8_t*)(tilingDatafromBin));

//     AscendC::GmFree(x);
//     AscendC::GmFree(indices);
//     AscendC::GmFree(var);
//     AscendC::GmFree(varScale);
//     AscendC::GmFree(varOffset);
//     AscendC::GmFree(workSpace);
//     AscendC::GmFree(tilingDatafromBin);
//     free(path_);
// }

TEST_F(dynamic_quant_update_scatter_v2_test, test_case_1)
{
    size_t xByteSize = 192 * 1 * 128 * sizeof(half);
    // size_t xBf16ByteSize = 192 * 1 * 128 * sizeof(bfloat16_t);
    size_t indicesByteSize = 192 * sizeof(uint32_t);
    size_t varByteSize = 192 * 1075 * 128 * sizeof(int4b_t);
    size_t varScaleByteSize = 192 * 1075 * sizeof(float);
    size_t varOffsetByteSize = 192 * 1075 * sizeof(float);
    size_t tiling_data_size = sizeof(DynamicQuantUpdateScatterV2TilingData);
    uint32_t blockDim = 48;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    // uint8_t* xBf16 = (uint8_t*)AscendC::GmAlloc(xBf16ByteSize);
    uint8_t* indices = (uint8_t*)AscendC::GmAlloc(indicesByteSize);
    uint8_t* var = (uint8_t*)AscendC::GmAlloc(varByteSize);
    uint8_t* varScale = (uint8_t*)AscendC::GmAlloc(varScaleByteSize);
    uint8_t* varOffset = (uint8_t*)AscendC::GmAlloc(varOffsetByteSize);
    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    DynamicQuantUpdateScatterV2TilingData* tilingDatafromBin =
        reinterpret_cast<DynamicQuantUpdateScatterV2TilingData*>(tiling);

    tilingDatafromBin->coreNum = 48;
    tilingDatafromBin->rowLen = 128;
    tilingDatafromBin->headCoreNum = 47;
    tilingDatafromBin->rowPerHeadCore = 4;
    tilingDatafromBin->rowPerTailCore = 4;
    tilingDatafromBin->multiRowNumHeadCore = 4;
    tilingDatafromBin->multiRowNumTailCore = 4;
    tilingDatafromBin->batchSize = 192;
    tilingDatafromBin->dstSeqLen = 1075;

    // ICPU_SET_TILING_KEY(2);
    // ICPU_RUN_KF(dynamic_quant_update_scatter_v2, blockDim, xBf16, indices, var, varScale, varOffset, workSpace,
    // (uint8_t*)(tilingDatafromBin));

    ICPU_SET_TILING_KEY(3);
    ICPU_RUN_KF(
        dynamic_quant_update_scatter_v2, blockDim, x, indices, var, varScale, varOffset, var, varScale, varOffset,
        workSpace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(indices);
    AscendC::GmFree(var);
    AscendC::GmFree(varScale);
    AscendC::GmFree(varOffset);
    AscendC::GmFree(workSpace);
    AscendC::GmFree(tilingDatafromBin);
    free(path_);
}