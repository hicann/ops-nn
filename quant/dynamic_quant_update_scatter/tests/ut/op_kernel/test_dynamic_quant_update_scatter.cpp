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
#include "dynamic_quant_update_scatter_tiling_def.h"
#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void dynamic_quant_update_scatter(
    GM_ADDR var, GM_ADDR varScale, GM_ADDR indices, GM_ADDR updates, GM_ADDR smoothScales, GM_ADDR varOut,
    GM_ADDR varScaleOut, GM_ADDR workSpace, GM_ADDR tiling);

class dynamic_quant_update_scatter_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "dynamic_quant_update_scatter_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "dynamic_quant_update_scatter_test TearDown\n" << endl;
    }
};

TEST_F(dynamic_quant_update_scatter_test, test_case_100)
{
    size_t inputVarByteSize = 24 * 1 * 128 * sizeof(int8_t);
    size_t inputVarScaleByteSize = 24 * 1 * 1 * sizeof(float);
    size_t inputIndicesByteSize = 24 * sizeof(uint32_t);
    size_t inputUpdatesByteSize = 24 * 1 * 128 * sizeof(half);
    size_t smoothScalesByteSzie = 128 * sizeof(half);
    size_t outputByteSize = 24 * 1 * 128 * sizeof(int8_t);
    size_t scaleByteSize = 24 * 1 * 1 * sizeof(float);
    size_t tiling_data_size = sizeof(DynamicQuantUpdateScatterTilingData);
    uint32_t blockDim = 24;

    uint8_t* var = (uint8_t*)AscendC::GmAlloc(inputVarByteSize);
    uint8_t* varScale = (uint8_t*)AscendC::GmAlloc(inputVarScaleByteSize);
    uint8_t* indices = (uint8_t*)AscendC::GmAlloc(inputIndicesByteSize);
    uint8_t* inputUpdates = (uint8_t*)AscendC::GmAlloc(inputUpdatesByteSize);
    uint8_t* smoothScales = (uint8_t*)AscendC::GmAlloc(smoothScalesByteSzie);
    uint8_t* output = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);

    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    DynamicQuantUpdateScatterTilingData* tilingDatafromBin =
        reinterpret_cast<DynamicQuantUpdateScatterTilingData*>(tiling);

    tilingDatafromBin->coreNum = 24;
    tilingDatafromBin->eachCoreBsNum = 1;
    tilingDatafromBin->lastCoreBsNum = 1;
    tilingDatafromBin->updateAxisShape = 1;
    tilingDatafromBin->srcBsStride = 128;
    tilingDatafromBin->dstBsStride = 128;
    tilingDatafromBin->indexElements = 24;
    tilingDatafromBin->numHead = 1;
    tilingDatafromBin->sizePerHead = 128;
    tilingDatafromBin->dataAxisShape = 1;
    tilingDatafromBin->numOneBlock = 32;
    tilingDatafromBin->innerLoopEle = 0;
    tilingDatafromBin->indicesShapeRank = 1;
    tilingDatafromBin->srcFirBsStride = 0;
    tilingDatafromBin->dstFirSecBsStride = 0;
    tilingDatafromBin->updateDim0 = 0;
    tilingDatafromBin->updateDim1 = 0;
    tilingDatafromBin->varElements = 3072;
    tilingDatafromBin->varScalesElements = 24;
    tilingDatafromBin->updatesElements = 3072;
    tilingDatafromBin->quantReptNum = 1;
    tilingDatafromBin->varOrigLastDimSize = 128;
    tilingDatafromBin->sizeSrcPerHead = 128;
    tilingDatafromBin->innerLoopFullRpt = 0;
    tilingDatafromBin->innerLoopTimes = 0;
    tilingDatafromBin->innerLoopTail = 0;
    tilingDatafromBin->innerLoopTimesLastCore = 0;
    tilingDatafromBin->innerLoopTailLastCore = 0;

    ICPU_SET_TILING_KEY(100);
    ICPU_RUN_KF(
        dynamic_quant_update_scatter, blockDim, var, varScale, indices, inputUpdates, smoothScales, output, scale,
        workSpace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(var);
    AscendC::GmFree(varScale);
    AscendC::GmFree(indices);
    AscendC::GmFree(inputUpdates);
    AscendC::GmFree(smoothScales);
    AscendC::GmFree(output);
    AscendC::GmFree(scale);
    AscendC::GmFree(workSpace);
    AscendC::GmFree(tilingDatafromBin);
    free(path_);
}

TEST_F(dynamic_quant_update_scatter_test, test_case_101)
{
    size_t inputVarByteSize = 2048 * 256 * 8 * 128 * sizeof(int8_t);
    size_t inputVarScaleByteSize = 2048 * 256 * 8 * sizeof(float);
    size_t inputIndicesByteSize = 2048 * 2 * sizeof(uint32_t);
    size_t inputUpdatesByteSize = 2048 * 1 * 8 * 128 * sizeof(half);
    size_t smoothScalesByteSzie = 128 * sizeof(half);
    size_t outputByteSize = 2048 * 256 * 8 * 128 * sizeof(int8_t);
    size_t scaleByteSize = 2048 * 256 * 8 * sizeof(float);
    size_t tiling_data_size = sizeof(DynamicQuantUpdateScatterTilingData);
    uint32_t blockDim = 48;

    uint8_t* var = (uint8_t*)AscendC::GmAlloc(inputVarByteSize);
    uint8_t* varScale = (uint8_t*)AscendC::GmAlloc(inputVarScaleByteSize);
    uint8_t* indices = (uint8_t*)AscendC::GmAlloc(inputIndicesByteSize);
    uint8_t* inputUpdates = (uint8_t*)AscendC::GmAlloc(inputUpdatesByteSize);
    uint8_t* smoothScales = (uint8_t*)AscendC::GmAlloc(smoothScalesByteSzie);
    uint8_t* output = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);

    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    DynamicQuantUpdateScatterTilingData* tilingDatafromBin =
        reinterpret_cast<DynamicQuantUpdateScatterTilingData*>(tiling);

    tilingDatafromBin->coreNum = 48;
    tilingDatafromBin->eachCoreBsNum = 43;
    tilingDatafromBin->lastCoreBsNum = 27;
    tilingDatafromBin->updateAxisShape = 1;
    tilingDatafromBin->srcBsStride = 1024;
    tilingDatafromBin->dstBsStride = 262144;
    tilingDatafromBin->indexElements = 4096;
    tilingDatafromBin->numHead = 1;
    tilingDatafromBin->sizePerHead = 1024;
    tilingDatafromBin->dataAxisShape = 256;
    tilingDatafromBin->numOneBlock = 32;
    tilingDatafromBin->innerLoopEle = 0;
    tilingDatafromBin->indicesShapeRank = 2;
    tilingDatafromBin->srcFirBsStride = 0;
    tilingDatafromBin->dstFirSecBsStride = 0;
    tilingDatafromBin->updateDim0 = 0;
    tilingDatafromBin->updateDim1 = 0;
    tilingDatafromBin->varElements = 536870912;
    tilingDatafromBin->varScalesElements = 4194304;
    tilingDatafromBin->updatesElements = 2097152;
    tilingDatafromBin->quantReptNum = 8;
    tilingDatafromBin->varOrigLastDimSize = 128;
    tilingDatafromBin->sizeSrcPerHead = 1024;
    tilingDatafromBin->innerLoopFullRpt = 0;
    tilingDatafromBin->innerLoopTimes = 0;
    tilingDatafromBin->innerLoopTail = 0;
    tilingDatafromBin->innerLoopTimesLastCore = 0;
    tilingDatafromBin->innerLoopTailLastCore = 0;

    ICPU_SET_TILING_KEY(101);
    ICPU_RUN_KF(
        dynamic_quant_update_scatter, blockDim, var, varScale, indices, inputUpdates, smoothScales, output, scale,
        workSpace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(var);
    AscendC::GmFree(varScale);
    AscendC::GmFree(indices);
    AscendC::GmFree(inputUpdates);
    AscendC::GmFree(smoothScales);
    AscendC::GmFree(output);
    AscendC::GmFree(scale);
    AscendC::GmFree(workSpace);
    AscendC::GmFree(tilingDatafromBin);
    free(path_);
}

TEST_F(dynamic_quant_update_scatter_test, test_case_102)
{
    size_t inputVarByteSize = 24 * 1 * 64 * 2048 * sizeof(int8_t);
    size_t inputVarScaleByteSize = 24 * 1 * 64 * 1 * sizeof(float);
    size_t inputIndicesByteSize = 24 * sizeof(uint32_t);
    size_t inputUpdatesByteSize = 24 * 1 * 64 * 2048 * sizeof(half);
    size_t smoothScalesByteSzie = 2048 * sizeof(half);
    size_t outputByteSize = 24 * 1 * 64 * 2048 * sizeof(int8_t);
    size_t scaleByteSize = 24 * 1 * 64 * 1 * sizeof(float);
    size_t tiling_data_size = sizeof(DynamicQuantUpdateScatterTilingData);
    uint32_t blockDim = 32;

    uint8_t* var = (uint8_t*)AscendC::GmAlloc(inputVarByteSize);
    uint8_t* varScale = (uint8_t*)AscendC::GmAlloc(inputVarScaleByteSize);
    uint8_t* indices = (uint8_t*)AscendC::GmAlloc(inputIndicesByteSize);
    uint8_t* inputUpdates = (uint8_t*)AscendC::GmAlloc(inputUpdatesByteSize);
    uint8_t* smoothScales = (uint8_t*)AscendC::GmAlloc(smoothScalesByteSzie);
    uint8_t* output = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);

    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    DynamicQuantUpdateScatterTilingData* tilingDatafromBin =
        reinterpret_cast<DynamicQuantUpdateScatterTilingData*>(tiling);

    tilingDatafromBin->coreNum = 32;
    tilingDatafromBin->eachCoreBsNum = 2;
    tilingDatafromBin->lastCoreBsNum = 2;
    tilingDatafromBin->updateAxisShape = 64;
    tilingDatafromBin->srcBsStride = 131072;
    tilingDatafromBin->dstBsStride = 131072;
    tilingDatafromBin->indexElements = 24;
    tilingDatafromBin->numHead = 1;
    tilingDatafromBin->sizePerHead = 2048;
    tilingDatafromBin->dataAxisShape = 64;
    tilingDatafromBin->numOneBlock = 32;
    tilingDatafromBin->innerLoopEle = 10240;
    tilingDatafromBin->indicesShapeRank = 1;
    tilingDatafromBin->srcFirBsStride = 131072;
    tilingDatafromBin->dstFirSecBsStride = 131072;
    tilingDatafromBin->updateDim0 = 24;
    tilingDatafromBin->updateDim1 = 1;
    tilingDatafromBin->varElements = 3145728;
    tilingDatafromBin->varScalesElements = 1536;
    tilingDatafromBin->updatesElements = 3145728;
    tilingDatafromBin->quantReptNum = 1;
    tilingDatafromBin->varOrigLastDimSize = 2048;
    tilingDatafromBin->sizeSrcPerHead = 2048;
    tilingDatafromBin->innerLoopFullRpt = 5;
    tilingDatafromBin->innerLoopTimes = 0;
    tilingDatafromBin->innerLoopTail = 2;
    tilingDatafromBin->innerLoopTimesLastCore = 2;
    tilingDatafromBin->innerLoopTailLastCore = 0;

    ICPU_SET_TILING_KEY(102);
    ICPU_RUN_KF(
        dynamic_quant_update_scatter, blockDim, var, varScale, indices, inputUpdates, smoothScales, output, scale,
        workSpace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(var);
    AscendC::GmFree(varScale);
    AscendC::GmFree(indices);
    AscendC::GmFree(inputUpdates);
    AscendC::GmFree(smoothScales);
    AscendC::GmFree(output);
    AscendC::GmFree(scale);
    AscendC::GmFree(workSpace);
    AscendC::GmFree(tilingDatafromBin);
    free(path_);
}

TEST_F(dynamic_quant_update_scatter_test, test_case_103)
{
    size_t inputVarByteSize = 1 * 24 * 12800 * sizeof(int8_t);
    size_t inputVarScaleByteSize = 1 * 24 * sizeof(float);
    size_t inputIndicesByteSize = 1 * sizeof(uint32_t) + 32;
    size_t inputUpdatesByteSize = 1 * 24 * 12800 * sizeof(half);
    size_t smoothScalesByteSzie = 12800 * sizeof(half);
    size_t outputByteSize = 1 * 24 * 12800 * sizeof(int8_t);
    size_t scaleByteSize = 24 * 1 * sizeof(float);
    size_t tiling_data_size = sizeof(DynamicQuantUpdateScatterTilingData);
    uint32_t blockDim = 24;

    uint8_t* var = (uint8_t*)AscendC::GmAlloc(inputVarByteSize);
    uint8_t* varScale = (uint8_t*)AscendC::GmAlloc(inputVarScaleByteSize);
    uint8_t* indices = (uint8_t*)AscendC::GmAlloc(inputIndicesByteSize);
    uint8_t* inputUpdates = (uint8_t*)AscendC::GmAlloc(inputUpdatesByteSize);
    uint8_t* smoothScales = (uint8_t*)AscendC::GmAlloc(smoothScalesByteSzie);
    uint8_t* output = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);

    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    DynamicQuantUpdateScatterTilingData* tilingDatafromBin =
        reinterpret_cast<DynamicQuantUpdateScatterTilingData*>(tiling);

    tilingDatafromBin->coreNum = 24;
    tilingDatafromBin->eachCoreBsNum = 1;
    tilingDatafromBin->lastCoreBsNum = 1;
    tilingDatafromBin->updateAxisShape = 24;
    tilingDatafromBin->srcBsStride = 307200;
    tilingDatafromBin->dstBsStride = 307200;
    tilingDatafromBin->indexElements = 1;
    tilingDatafromBin->numHead = 1;
    tilingDatafromBin->sizePerHead = 12800;
    tilingDatafromBin->dataAxisShape = 24;
    tilingDatafromBin->numOneBlock = 32;
    tilingDatafromBin->innerLoopEle = 9088;
    tilingDatafromBin->indicesShapeRank = 1;
    tilingDatafromBin->srcFirBsStride = 307200;
    tilingDatafromBin->dstFirSecBsStride = 307200;
    tilingDatafromBin->updateDim0 = 1;
    tilingDatafromBin->updateDim1 = 1;
    tilingDatafromBin->varElements = 307200;
    tilingDatafromBin->varScalesElements = 24;
    tilingDatafromBin->updatesElements = 307200;
    tilingDatafromBin->quantReptNum = 1;
    tilingDatafromBin->varOrigLastDimSize = 12800;
    tilingDatafromBin->sizeSrcPerHead = 12800;
    tilingDatafromBin->innerLoopFullRpt = 0;
    tilingDatafromBin->innerLoopTimes = 1;
    tilingDatafromBin->innerLoopTail = 3712;
    tilingDatafromBin->innerLoopTimesLastCore = 0;
    tilingDatafromBin->innerLoopTailLastCore = 0;

    ICPU_SET_TILING_KEY(103);
    ICPU_RUN_KF(
        dynamic_quant_update_scatter, blockDim, var, varScale, indices, inputUpdates, smoothScales, output, scale,
        workSpace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(var);
    AscendC::GmFree(varScale);
    AscendC::GmFree(indices);
    AscendC::GmFree(inputUpdates);
    AscendC::GmFree(smoothScales);
    AscendC::GmFree(output);
    AscendC::GmFree(scale);
    AscendC::GmFree(workSpace);
    AscendC::GmFree(tilingDatafromBin);
    free(path_);
}

TEST_F(dynamic_quant_update_scatter_test, test_case_105)
{
    size_t inputVarByteSize = 48 * 24 * 12800 * sizeof(int8_t);
    size_t inputVarScaleByteSize = 48 * 24 * sizeof(float);
    size_t inputIndicesByteSize = 48 * sizeof(uint32_t);
    size_t inputUpdatesByteSize = 48 * 24 * 12800 * sizeof(half);
    size_t smoothScalesByteSzie = 12800 * sizeof(half);
    size_t outputByteSize = 48 * 24 * 12800 * sizeof(int8_t);
    size_t scaleByteSize = 48 * 24 * sizeof(float);
    size_t tiling_data_size = sizeof(DynamicQuantUpdateScatterTilingData);
    uint32_t blockDim = 48;

    uint8_t* var = (uint8_t*)AscendC::GmAlloc(inputVarByteSize);
    uint8_t* varScale = (uint8_t*)AscendC::GmAlloc(inputVarScaleByteSize);
    uint8_t* indices = (uint8_t*)AscendC::GmAlloc(inputIndicesByteSize);
    uint8_t* inputUpdates = (uint8_t*)AscendC::GmAlloc(inputUpdatesByteSize);
    uint8_t* smoothScales = (uint8_t*)AscendC::GmAlloc(smoothScalesByteSzie);
    uint8_t* output = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);

    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    DynamicQuantUpdateScatterTilingData* tilingDatafromBin =
        reinterpret_cast<DynamicQuantUpdateScatterTilingData*>(tiling);

    tilingDatafromBin->coreNum = 48;
    tilingDatafromBin->eachCoreBsNum = 1;
    tilingDatafromBin->lastCoreBsNum = 1;
    tilingDatafromBin->updateAxisShape = 24;
    tilingDatafromBin->srcBsStride = 307200;
    tilingDatafromBin->dstBsStride = 307200;
    tilingDatafromBin->indexElements = 48;
    tilingDatafromBin->numHead = 1;
    tilingDatafromBin->sizePerHead = 12800;
    tilingDatafromBin->dataAxisShape = 24;
    tilingDatafromBin->numOneBlock = 32;
    tilingDatafromBin->innerLoopEle = 9088;
    tilingDatafromBin->indicesShapeRank = 1;
    tilingDatafromBin->srcFirBsStride = 307200;
    tilingDatafromBin->dstFirSecBsStride = 307200;
    tilingDatafromBin->updateDim0 = 48;
    tilingDatafromBin->updateDim1 = 1;
    tilingDatafromBin->varElements = 14745600;
    tilingDatafromBin->varScalesElements = 1152;
    tilingDatafromBin->updatesElements = 14745600;
    tilingDatafromBin->quantReptNum = 1;
    tilingDatafromBin->varOrigLastDimSize = 12800;
    tilingDatafromBin->sizeSrcPerHead = 12800;
    tilingDatafromBin->innerLoopFullRpt = 0;
    tilingDatafromBin->innerLoopTimes = 1;
    tilingDatafromBin->innerLoopTail = 3712;
    tilingDatafromBin->innerLoopTimesLastCore = 0;
    tilingDatafromBin->innerLoopTailLastCore = 0;

    ICPU_SET_TILING_KEY(105);
    ICPU_RUN_KF(
        dynamic_quant_update_scatter, blockDim, var, varScale, indices, inputUpdates, smoothScales, output, scale,
        workSpace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(var);
    AscendC::GmFree(varScale);
    AscendC::GmFree(indices);
    AscendC::GmFree(inputUpdates);
    AscendC::GmFree(smoothScales);
    AscendC::GmFree(output);
    AscendC::GmFree(scale);
    AscendC::GmFree(workSpace);
    AscendC::GmFree(tilingDatafromBin);
    free(path_);
}

TEST_F(dynamic_quant_update_scatter_test, test_case_106)
{
    size_t inputVarByteSize = 24 * 16384 * 128 * sizeof(int8_t);
    size_t inputVarScaleByteSize = 24 * 16384 * 1 * sizeof(float);
    size_t inputIndicesByteSize = 2 * sizeof(uint32_t) + 32;
    size_t inputUpdatesByteSize = 1 * 9216 * 128 * sizeof(half);
    size_t smoothScalesByteSzie = 128 * sizeof(half);
    size_t outputByteSize = 24 * 16384 * 128 * sizeof(int8_t);
    size_t scaleByteSize = 24 * 16384 * 1 * sizeof(float);
    size_t tiling_data_size = sizeof(DynamicQuantUpdateScatterTilingData);
    uint32_t blockDim = 48;

    uint8_t* var = (uint8_t*)AscendC::GmAlloc(inputVarByteSize);
    uint8_t* varScale = (uint8_t*)AscendC::GmAlloc(inputVarScaleByteSize);
    uint8_t* indices = (uint8_t*)AscendC::GmAlloc(inputIndicesByteSize);
    uint8_t* inputUpdates = (uint8_t*)AscendC::GmAlloc(inputUpdatesByteSize);
    uint8_t* smoothScales = (uint8_t*)AscendC::GmAlloc(smoothScalesByteSzie);
    uint8_t* output = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);

    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    DynamicQuantUpdateScatterTilingData* tilingDatafromBin =
        reinterpret_cast<DynamicQuantUpdateScatterTilingData*>(tiling);

    tilingDatafromBin->coreNum = 48;
    tilingDatafromBin->eachCoreBsNum = 192;
    tilingDatafromBin->lastCoreBsNum = 192;
    tilingDatafromBin->updateAxisShape = 9216;
    tilingDatafromBin->srcBsStride = 1179648;
    tilingDatafromBin->dstBsStride = 2097152;
    tilingDatafromBin->indexElements = 2;
    tilingDatafromBin->numHead = 1;
    tilingDatafromBin->sizePerHead = 128;
    tilingDatafromBin->dataAxisShape = 16384;
    tilingDatafromBin->numOneBlock = 32;
    tilingDatafromBin->innerLoopEle = 11264;
    tilingDatafromBin->indicesShapeRank = 2;
    tilingDatafromBin->srcFirBsStride = 1179648;
    tilingDatafromBin->dstFirSecBsStride = 2097152;
    tilingDatafromBin->updateDim0 = 1;
    tilingDatafromBin->updateDim1 = 1;
    tilingDatafromBin->varElements = 50331648;
    tilingDatafromBin->varScalesElements = 393216;
    tilingDatafromBin->updatesElements = 1179648;
    tilingDatafromBin->quantReptNum = 1;
    tilingDatafromBin->varOrigLastDimSize = 128;
    tilingDatafromBin->sizeSrcPerHead = 128;
    tilingDatafromBin->innerLoopFullRpt = 88;
    tilingDatafromBin->innerLoopTimes = 2;
    tilingDatafromBin->innerLoopTail = 16;
    tilingDatafromBin->innerLoopTimesLastCore = 2;
    tilingDatafromBin->innerLoopTailLastCore = 16;

    ICPU_SET_TILING_KEY(106);
    ICPU_RUN_KF(
        dynamic_quant_update_scatter, blockDim, var, varScale, indices, inputUpdates, smoothScales, output, scale,
        workSpace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(var);
    AscendC::GmFree(varScale);
    AscendC::GmFree(indices);
    AscendC::GmFree(inputUpdates);
    AscendC::GmFree(smoothScales);
    AscendC::GmFree(output);
    AscendC::GmFree(scale);
    AscendC::GmFree(workSpace);
    AscendC::GmFree(tilingDatafromBin);
    free(path_);
}