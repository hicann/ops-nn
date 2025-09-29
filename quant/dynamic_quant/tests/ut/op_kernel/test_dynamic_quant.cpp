/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
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
#define __CCE_AICORE__ 310

using namespace std;

extern "C" void dynamic_quant(
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

TEST_F(dynamic_quant_test, test_case_fp16_bf16)
{
    size_t inputByteSize = 1 * 128 * 1024 * sizeof(uint16_t);
    size_t smoothScalesByteSzie = 1024 * sizeof(uint16_t);
    size_t smoothScalesMoeByteSzie = 8 * 1024 * sizeof(uint16_t);
    size_t groupIndexMoeByteSize = 8 * sizeof(int32_t);
    size_t outputByteSize = 1 * 128 * 1024 * sizeof(int8_t);
    size_t scaleByteSize = 1 * 128 * sizeof(uint32_t);
    size_t tiling_data_size = sizeof(DynamicQuantTilingData);
    uint32_t blockDim = 64;

    uint8_t* input = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* smooth_scales = (uint8_t*)AscendC::GmAlloc(smoothScalesByteSzie);
    uint8_t* group_index = nullptr;
    uint8_t* smooth_scales_moe = (uint8_t*)AscendC::GmAlloc(smoothScalesMoeByteSzie);
    uint8_t* group_index_moe = (uint8_t*)AscendC::GmAlloc(groupIndexMoeByteSize);
    uint8_t* output = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);

    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    DynamicQuantTilingData* tilingDatafromBin = reinterpret_cast<DynamicQuantTilingData*>(tiling);

    tilingDatafromBin->coreNum = 64;
    tilingDatafromBin->rowLen = 1024;
    tilingDatafromBin->headCoreNum = 0;
    tilingDatafromBin->rowPerHeadCore = 2;
    tilingDatafromBin->rowPerTailCore = 2;
    tilingDatafromBin->multiRowNumHeadCore = 2;
    tilingDatafromBin->multiRowNumTailCore = 2;
    tilingDatafromBin->innerLoopEle = 0;
    tilingDatafromBin->innerLoopTimes = 0;
    tilingDatafromBin->innerLoopTail = 0;
    tilingDatafromBin->groupNum = 0;
    tilingDatafromBin->alignGroupNum = 0;
    tilingDatafromBin->hasSmooth = 1;

    tilingDatafromBin->sizeH = 0;
    tilingDatafromBin->sizeX = 0;
    tilingDatafromBin->sizeZOut = 0;
    tilingDatafromBin->sizeCopyRow = 0;
    tilingDatafromBin->numCopyRow = 0;
    tilingDatafromBin->numHeadCore = 0;
    tilingDatafromBin->numTailCore = 0;
    tilingDatafromBin->numHeadTimes = 0;
    tilingDatafromBin->numTailTimes = 0;
    tilingDatafromBin->numLastTailRow = 0;
    tilingDatafromBin->alignType = 0;

    tilingDatafromBin->sizeH = 0;
    tilingDatafromBin->sizeX = 0;
    tilingDatafromBin->sizeZOut = 0;
    tilingDatafromBin->sizeCopyRow = 0;
    tilingDatafromBin->numCopyRow = 0;
    tilingDatafromBin->numHeadCore = 0;
    tilingDatafromBin->numTailCore = 0;
    tilingDatafromBin->numHeadTimes = 0;
    tilingDatafromBin->numTailTimes = 0;
    tilingDatafromBin->numLastTailRow = 0;
    tilingDatafromBin->alignType = 0;

    // bf16
    ICPU_SET_TILING_KEY(201);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(
        dynamic_quant, 64, input, smooth_scales, group_index, output, scale, workSpace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(input);
    AscendC::GmFree(smooth_scales);
    AscendC::GmFree(smooth_scales_moe);
    AscendC::GmFree(group_index_moe);
    AscendC::GmFree(output);
    AscendC::GmFree(scale);
    AscendC::GmFree(workSpace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(dynamic_quant_test, test_case_fp16_bf16_no_smooth)
{
    size_t inputByteSize = 1 * 128 * 1024 * sizeof(uint16_t);
    size_t outputByteSize = 1 * 128 * 1024 * sizeof(int8_t);
    size_t scaleByteSize = 1 * 128 * sizeof(uint32_t);
    size_t tiling_data_size = sizeof(DynamicQuantTilingData);
    uint32_t blockDim = 64;

    uint8_t* input = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* smooth_scales = nullptr;
    uint8_t* group_index = nullptr;
    uint8_t* output = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);

    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    DynamicQuantTilingData* tilingDatafromBin = reinterpret_cast<DynamicQuantTilingData*>(tiling);

    tilingDatafromBin->coreNum = 64;
    tilingDatafromBin->rowLen = 1024;
    tilingDatafromBin->headCoreNum = 0;
    tilingDatafromBin->rowPerHeadCore = 2;
    tilingDatafromBin->rowPerTailCore = 2;
    tilingDatafromBin->multiRowNumHeadCore = 2;
    tilingDatafromBin->multiRowNumTailCore = 2;
    tilingDatafromBin->innerLoopEle = 0;
    tilingDatafromBin->innerLoopTimes = 0;
    tilingDatafromBin->innerLoopTail = 0;
    tilingDatafromBin->groupNum = 0;
    tilingDatafromBin->alignGroupNum = 0;
    tilingDatafromBin->hasSmooth = 0;

    tilingDatafromBin->sizeH = 0;
    tilingDatafromBin->sizeX = 0;
    tilingDatafromBin->sizeZOut = 0;
    tilingDatafromBin->sizeCopyRow = 0;
    tilingDatafromBin->numCopyRow = 0;
    tilingDatafromBin->numHeadCore = 0;
    tilingDatafromBin->numTailCore = 0;
    tilingDatafromBin->numHeadTimes = 0;
    tilingDatafromBin->numTailTimes = 0;
    tilingDatafromBin->numLastTailRow = 0;
    tilingDatafromBin->alignType = 0;
    tilingDatafromBin->sizeH = 0;
    tilingDatafromBin->sizeX = 0;
    tilingDatafromBin->sizeZOut = 0;
    tilingDatafromBin->sizeCopyRow = 0;
    tilingDatafromBin->numCopyRow = 0;
    tilingDatafromBin->numHeadCore = 0;
    tilingDatafromBin->numTailCore = 0;
    tilingDatafromBin->numHeadTimes = 0;
    tilingDatafromBin->numTailTimes = 0;
    tilingDatafromBin->numLastTailRow = 0;
    tilingDatafromBin->alignType = 0;
    // bf16
    ICPU_SET_TILING_KEY(200);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(
        dynamic_quant, 1, input, smooth_scales, group_index, output, scale, workSpace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(input);
    AscendC::GmFree(output);
    AscendC::GmFree(scale);
    AscendC::GmFree(workSpace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(dynamic_quant_test, test_case_fp16_bf16_no_smooth_use_db)
{
    size_t inputByteSize = 1 * 128 * 15360 * sizeof(uint16_t);
    size_t outputByteSize = 1 * 128 * 15360 * sizeof(int8_t);
    size_t scaleByteSize = 1 * 128 * sizeof(uint32_t);
    size_t tiling_data_size = sizeof(DynamicQuantTilingData);
    uint32_t blockDim = 64;

    uint8_t* input = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* smooth_scales = nullptr;
    uint8_t* group_index = nullptr;
    uint8_t* output = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);

    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    DynamicQuantTilingData* tilingDatafromBin = reinterpret_cast<DynamicQuantTilingData*>(tiling);

    tilingDatafromBin->coreNum = 64;
    tilingDatafromBin->rowLen = 15360;
    tilingDatafromBin->headCoreNum = 0;
    tilingDatafromBin->rowPerHeadCore = 2;
    tilingDatafromBin->rowPerTailCore = 2;
    tilingDatafromBin->multiRowNumHeadCore = 1;
    tilingDatafromBin->multiRowNumTailCore = 1;
    tilingDatafromBin->innerLoopEle = 0;
    tilingDatafromBin->innerLoopTimes = 0;
    tilingDatafromBin->innerLoopTail = 0;
    tilingDatafromBin->groupNum = 0;
    tilingDatafromBin->alignGroupNum = 0;
    tilingDatafromBin->hasSmooth = 1;

    tilingDatafromBin->sizeH = 0;
    tilingDatafromBin->sizeX = 0;
    tilingDatafromBin->sizeZOut = 0;
    tilingDatafromBin->sizeCopyRow = 0;
    tilingDatafromBin->numCopyRow = 0;
    tilingDatafromBin->numHeadCore = 0;
    tilingDatafromBin->numTailCore = 0;
    tilingDatafromBin->numHeadTimes = 0;
    tilingDatafromBin->numTailTimes = 0;
    tilingDatafromBin->numLastTailRow = 0;
    tilingDatafromBin->alignType = 0;
    tilingDatafromBin->sizeH = 0;
    tilingDatafromBin->sizeX = 0;
    tilingDatafromBin->sizeZOut = 0;
    tilingDatafromBin->sizeCopyRow = 0;
    tilingDatafromBin->numCopyRow = 0;
    tilingDatafromBin->numHeadCore = 0;
    tilingDatafromBin->numTailCore = 0;
    tilingDatafromBin->numHeadTimes = 0;
    tilingDatafromBin->numTailTimes = 0;
    tilingDatafromBin->numLastTailRow = 0;
    tilingDatafromBin->alignType = 0;
    // bf16
    ICPU_SET_TILING_KEY(100);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(
        dynamic_quant, 1, input, smooth_scales, group_index, output, scale, workSpace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(input);
    AscendC::GmFree(output);
    AscendC::GmFree(scale);
    AscendC::GmFree(workSpace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(dynamic_quant_test, test_case_largeShape_fp16_int8_1)
{
    size_t inputByteSize = 8 * 9 * 27185 * sizeof(uint16_t);
    size_t smoothScalesByteSzie = 27185 * sizeof(uint16_t);
    size_t outputByteSize = 8 * 9 * 27185 * sizeof(int8_t);
    size_t scaleByteSize = 8 * 9 * sizeof(uint32_t);
    size_t tiling_data_size = sizeof(DynamicQuantTilingData);
    uint32_t blockDim = 64;

    uint8_t* input = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* smooth_scales = (uint8_t*)AscendC::GmAlloc(smoothScalesByteSzie);
    uint8_t* group_index = nullptr;
    uint8_t* output = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);

    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    DynamicQuantTilingData* tilingDatafromBin = reinterpret_cast<DynamicQuantTilingData*>(tiling);

    tilingDatafromBin->coreNum = 64;
    tilingDatafromBin->rowLen = 27185;
    tilingDatafromBin->headCoreNum = 8;
    tilingDatafromBin->rowPerHeadCore = 2;
    tilingDatafromBin->rowPerTailCore = 1;
    tilingDatafromBin->multiRowNumHeadCore = 1;
    tilingDatafromBin->multiRowNumTailCore = 1;
    tilingDatafromBin->innerLoopEle = 20224;
    tilingDatafromBin->innerLoopTimes = 1;
    tilingDatafromBin->innerLoopTail = 6961;
    tilingDatafromBin->groupNum = 0;
    tilingDatafromBin->alignGroupNum = 0;
    tilingDatafromBin->hasSmooth = 1;

    tilingDatafromBin->sizeH = 0;
    tilingDatafromBin->sizeX = 0;
    tilingDatafromBin->sizeZOut = 0;
    tilingDatafromBin->sizeCopyRow = 0;
    tilingDatafromBin->numCopyRow = 0;
    tilingDatafromBin->numHeadCore = 0;
    tilingDatafromBin->numTailCore = 0;
    tilingDatafromBin->numHeadTimes = 0;
    tilingDatafromBin->numTailTimes = 0;
    tilingDatafromBin->numLastTailRow = 0;
    tilingDatafromBin->alignType = 0;

    tilingDatafromBin->sizeH = 0;
    tilingDatafromBin->sizeX = 0;
    tilingDatafromBin->sizeZOut = 0;
    tilingDatafromBin->sizeCopyRow = 0;
    tilingDatafromBin->numCopyRow = 0;
    tilingDatafromBin->numHeadCore = 0;
    tilingDatafromBin->numTailCore = 0;
    tilingDatafromBin->numHeadTimes = 0;
    tilingDatafromBin->numTailTimes = 0;
    tilingDatafromBin->numLastTailRow = 0;
    tilingDatafromBin->alignType = 0;
    ICPU_SET_TILING_KEY(211);
    ICPU_RUN_KF(
        dynamic_quant, blockDim, input, smooth_scales, group_index, output, scale, workSpace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(input);
    AscendC::GmFree(smooth_scales);
    AscendC::GmFree(output);
    AscendC::GmFree(scale);
    AscendC::GmFree(workSpace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(dynamic_quant_test, test_case_largeShape_fp16_int8_2)
{
    size_t inputByteSize = 8 * 9 * 34961 * sizeof(uint16_t);
    size_t outputByteSize = 8 * 9 * 34961 * sizeof(int8_t);
    size_t scaleByteSize = 8 * 9 * sizeof(uint32_t);
    size_t tiling_data_size = sizeof(DynamicQuantTilingData);
    uint32_t blockDim = 64;

    uint8_t* input = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* smooth_scales = nullptr;
    uint8_t* group_index = nullptr;
    uint8_t* output = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);

    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    DynamicQuantTilingData* tilingDatafromBin = reinterpret_cast<DynamicQuantTilingData*>(tiling);

    tilingDatafromBin->coreNum = 64;
    tilingDatafromBin->rowLen = 34961;
    tilingDatafromBin->headCoreNum = 8;
    tilingDatafromBin->rowPerHeadCore = 2;
    tilingDatafromBin->rowPerTailCore = 1;
    tilingDatafromBin->multiRowNumHeadCore = 1;
    tilingDatafromBin->multiRowNumTailCore = 1;
    tilingDatafromBin->innerLoopEle = 30464;
    tilingDatafromBin->innerLoopTimes = 1;
    tilingDatafromBin->innerLoopTail = 4497;
    tilingDatafromBin->groupNum = 0;
    tilingDatafromBin->alignGroupNum = 0;
    tilingDatafromBin->hasSmooth = 0;

    tilingDatafromBin->sizeH = 0;
    tilingDatafromBin->sizeX = 0;
    tilingDatafromBin->sizeZOut = 0;
    tilingDatafromBin->sizeCopyRow = 0;
    tilingDatafromBin->numCopyRow = 0;
    tilingDatafromBin->numHeadCore = 0;
    tilingDatafromBin->numTailCore = 0;
    tilingDatafromBin->numHeadTimes = 0;
    tilingDatafromBin->numTailTimes = 0;
    tilingDatafromBin->numLastTailRow = 0;
    tilingDatafromBin->alignType = 0;
    tilingDatafromBin->sizeH = 0;
    tilingDatafromBin->sizeX = 0;
    tilingDatafromBin->sizeZOut = 0;
    tilingDatafromBin->sizeCopyRow = 0;
    tilingDatafromBin->numCopyRow = 0;
    tilingDatafromBin->numHeadCore = 0;
    tilingDatafromBin->numTailCore = 0;
    tilingDatafromBin->numHeadTimes = 0;
    tilingDatafromBin->numTailTimes = 0;
    tilingDatafromBin->numLastTailRow = 0;
    tilingDatafromBin->alignType = 0;
    ICPU_SET_TILING_KEY(210);
    ICPU_RUN_KF(
        dynamic_quant, blockDim, input, smooth_scales, group_index, output, scale, workSpace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(input);
    AscendC::GmFree(output);
    AscendC::GmFree(scale);
    AscendC::GmFree(workSpace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(dynamic_quant_test, test_case_310_01)
{
    size_t inputByteSize = 8 * 1024 * sizeof(uint16_t);
    size_t outputByteSize = 8 * 1024 * sizeof(int8_t);
    size_t scaleByteSize = 8 * sizeof(uint32_t);
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

    tilingDatafromBin->sizeH = 1024;
    tilingDatafromBin->sizeX = 1024;
    tilingDatafromBin->sizeZOut = 1024;
    tilingDatafromBin->sizeCopyRow = 8;
    tilingDatafromBin->numCopyRow = 8;
    tilingDatafromBin->numHeadCore = 1;
    tilingDatafromBin->numTailCore = 7;
    tilingDatafromBin->numHeadTimes = 1;
    tilingDatafromBin->numTailTimes = 0;
    tilingDatafromBin->numLastTailRow = 0;
    tilingDatafromBin->innerLoopEle = 0;
    tilingDatafromBin->innerLoopTimes = 0;
    tilingDatafromBin->innerLoopTail = 0;
    tilingDatafromBin->alignType = 0;

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
    size_t inputByteSize = 10 * 1088 * sizeof(uint16_t);
    size_t outputByteSize = 10 * 1088 * sizeof(int8_t);
    size_t scaleByteSize = 10 * sizeof(uint32_t);
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
    size_t inputByteSize = 8 * 7168 * sizeof(uint16_t);
    size_t outputByteSize = 8 * 7168 * sizeof(int8_t);
    size_t scaleByteSize = 8 * sizeof(uint32_t);
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
    tilingDatafromBin->numHeadCore = 1;
    tilingDatafromBin->numTailCore = 7;
    tilingDatafromBin->numHeadTimes = 1;
    tilingDatafromBin->numTailTimes = 0;
    tilingDatafromBin->numLastTailRow = 0;
    tilingDatafromBin->innerLoopEle = 4096;
    tilingDatafromBin->innerLoopTimes = 1;
    tilingDatafromBin->innerLoopTail = 3072;
    tilingDatafromBin->alignType = 1;

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