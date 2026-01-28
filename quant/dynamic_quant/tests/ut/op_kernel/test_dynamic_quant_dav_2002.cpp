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
#include "dynamic_quant_tiling_def.h"
#include "data_utils.h"

#include <cstdint>

#define __CCE_AICORE__ 200
#include "dynamic_quant.cpp"

using namespace std;

void dynamic_quant(
    GM_ADDR x, GM_ADDR smooth_scales, GM_ADDR group_index, GM_ADDR y, GM_ADDR scale, GM_ADDR workSpace, GM_ADDR tiling);

class dynamic_quant_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "dynamic_quant_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "dynamic_quant_test TearDown\n" << endl;
    }
};

TEST_F(dynamic_quant_test, test_case_310_01)
{
    size_t inputByteSize = 32 * 1088 * sizeof(uint16_t);
    size_t outputByteSize = 32 * 1088 * sizeof(int8_t);
    size_t scaleByteSize = 32 * sizeof(uint32_t);
    size_t tiling_data_size = sizeof(DynamicQuantTilingData);
    uint32_t blockDim = 8;

    uint8_t* input = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* output = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);

    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    DynamicQuantTilingData* tilingDatafromBin = reinterpret_cast<DynamicQuantTilingData*>(tiling);

    tilingDatafromBin->coreNum = blockDim;
    tilingDatafromBin->rowLen = 0;
    tilingDatafromBin->headCoreNum = 0;
    tilingDatafromBin->rowPerHeadCore = 0;
    tilingDatafromBin->rowPerTailCore = 0;
    tilingDatafromBin->multiRowNumHeadCore = 0;
    tilingDatafromBin->multiRowNumTailCore = 0;
    tilingDatafromBin->groupNum = 0;
    tilingDatafromBin->alignGroupNum = 0;
    tilingDatafromBin->hasSmooth = 0;

    tilingDatafromBin->sizeH = 1088;
    tilingDatafromBin->sizeX = 1088;
    tilingDatafromBin->sizeZOut = 1088;
    tilingDatafromBin->sizeCopyRow = 8;
    tilingDatafromBin->numCopyRow = 8;
    tilingDatafromBin->numHeadCore = 4;
    tilingDatafromBin->numTailCore = 4;
    tilingDatafromBin->numHeadTimes = 1;
    tilingDatafromBin->numTailTimes = 0;
    tilingDatafromBin->numLastTailRow = 0;
    tilingDatafromBin->innerLoopEle = 0;
    tilingDatafromBin->innerLoopTimes = 0;
    tilingDatafromBin->innerLoopTail = 0;
    tilingDatafromBin->alignType = 0;
    tilingDatafromBin->ubSize = 191*1024;

    // bf16
    ICPU_SET_TILING_KEY(10100);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(
        dynamic_quant, blockDim, input, nullptr, nullptr, output, scale, workSpace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(input);
    AscendC::GmFree(output);
    AscendC::GmFree(scale);
    AscendC::GmFree(workSpace);
    AscendC::GmFree(tiling);
    free(path_);
}


TEST_F(dynamic_quant_test, test_case_310_02)
{
    size_t inputByteSize = 32 * 1088 * sizeof(uint16_t);
    size_t outputByteSize = 32 * 1088 * sizeof(int8_t);
    size_t scaleByteSize = 32 * sizeof(uint32_t);
    size_t tiling_data_size = sizeof(DynamicQuantTilingData);
    uint32_t blockDim = 8;

    uint8_t* input = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* output = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);

    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    DynamicQuantTilingData* tilingDatafromBin = reinterpret_cast<DynamicQuantTilingData*>(tiling);

    tilingDatafromBin->coreNum = blockDim;
    tilingDatafromBin->rowLen = 0;
    tilingDatafromBin->headCoreNum = 0;
    tilingDatafromBin->rowPerHeadCore = 0;
    tilingDatafromBin->rowPerTailCore = 0;
    tilingDatafromBin->multiRowNumHeadCore = 0;
    tilingDatafromBin->multiRowNumTailCore = 0;
    tilingDatafromBin->groupNum = 0;
    tilingDatafromBin->alignGroupNum = 0;
    tilingDatafromBin->hasSmooth = 0;

    tilingDatafromBin->sizeH = 1088;
    tilingDatafromBin->sizeX = 1088;
    tilingDatafromBin->sizeZOut = 1088;
    tilingDatafromBin->sizeCopyRow = 8;
    tilingDatafromBin->numCopyRow = 8;
    tilingDatafromBin->numHeadCore = 1;
    tilingDatafromBin->numTailCore = 7;
    tilingDatafromBin->numHeadTimes = 1;
    tilingDatafromBin->numTailTimes = 0;
    tilingDatafromBin->numLastTailRow = 2;
    tilingDatafromBin->innerLoopEle = 0;
    tilingDatafromBin->innerLoopTimes = 0;
    tilingDatafromBin->innerLoopTail = 0;
    tilingDatafromBin->alignType = 1;
    tilingDatafromBin->ubSize = 191*1024;

    // bf16
    ICPU_SET_TILING_KEY(11000);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(
        dynamic_quant, blockDim, input, nullptr, nullptr, output, scale, workSpace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(input);
    AscendC::GmFree(output);
    AscendC::GmFree(scale);
    AscendC::GmFree(workSpace);
    AscendC::GmFree(tiling);
    free(path_);
}


TEST_F(dynamic_quant_test, test_case_310_03)
{
    size_t inputByteSize = 32 * 7168 * sizeof(uint16_t);
    size_t outputByteSize = 32 * 7168 * sizeof(int8_t);
    size_t scaleByteSize = 32 * sizeof(uint32_t);
    size_t tiling_data_size = sizeof(DynamicQuantTilingData);
    uint32_t blockDim = 8;

    uint8_t* input = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* output = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);

    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    DynamicQuantTilingData* tilingDatafromBin = reinterpret_cast<DynamicQuantTilingData*>(tiling);

    tilingDatafromBin->coreNum = blockDim;
    tilingDatafromBin->rowLen = 0;
    tilingDatafromBin->headCoreNum = 0;
    tilingDatafromBin->rowPerHeadCore = 0;
    tilingDatafromBin->rowPerTailCore = 0;
    tilingDatafromBin->multiRowNumHeadCore = 0;
    tilingDatafromBin->multiRowNumTailCore = 0;
    tilingDatafromBin->groupNum = 0;
    tilingDatafromBin->alignGroupNum = 0;
    tilingDatafromBin->hasSmooth = 0;

    tilingDatafromBin->sizeH = 7168;
    tilingDatafromBin->sizeX = 7168;
    tilingDatafromBin->sizeZOut = 7168;
    tilingDatafromBin->sizeCopyRow = 8;
    tilingDatafromBin->numCopyRow = 8;
    tilingDatafromBin->numHeadCore = 4;
    tilingDatafromBin->numTailCore = 4;
    tilingDatafromBin->numHeadTimes = 1;
    tilingDatafromBin->numTailTimes = 0;
    tilingDatafromBin->numLastTailRow = 0;
    tilingDatafromBin->innerLoopEle = 4096;
    tilingDatafromBin->innerLoopTimes = 1;
    tilingDatafromBin->innerLoopTail = 3072;
    tilingDatafromBin->alignType = 1;
    tilingDatafromBin->ubSize = 191*1024;

    // bf16
    ICPU_SET_TILING_KEY(11001);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(
        dynamic_quant, blockDim, input, nullptr, nullptr, output, scale, workSpace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(input);
    AscendC::GmFree(output);
    AscendC::GmFree(scale);
    AscendC::GmFree(workSpace);
    AscendC::GmFree(tiling);
    free(path_);
}
