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
#include <limits>
#include <float.h>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "max_pool3d_with_argmax_v2_tiling_def.h"
#include "data_utils.h"

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void max_pool3d_with_argmax_v2(
    GM_ADDR x, GM_ADDR y, GM_ADDR indices, GM_ADDR workspace, GM_ADDR tiling);

class max_pool3d_with_argmax_v2_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "max_pool3d_with_argmax_v2_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "max_pool3d_with_argmax_v2_test TearDown\n" << endl;
    }
};

TEST_F(max_pool3d_with_argmax_v2_test, test_case_1)
{
    uint32_t blockDim = 1;
    size_t inputByteSize = 1 * 1 * 6 * 6 * 6 * sizeof(float);
    size_t outByteSize = 1 * 1 * 4 * 4 * 4 * sizeof(float);
    size_t indicesByteSize = 1 * 1 * 4 * 4 * 4 * sizeof(int32_t);
    size_t tilingDataSize = sizeof(MaxPool3DWithArgmaxV2NoSplitTilingData);

    uint8_t* input = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* out = (uint8_t*)AscendC::GmAlloc(outByteSize);
    uint8_t* indices = (uint8_t*)AscendC::GmAlloc(indicesByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    char* path_ = get_current_dir_name();
    string path(path_);

    MaxPool3DWithArgmaxV2NoSplitTilingData* tilingDatafromBin =
        reinterpret_cast<MaxPool3DWithArgmaxV2NoSplitTilingData*>(tiling);
    tilingDatafromBin->inputShapes[0] = 6;
    tilingDatafromBin->inputShapes[1] = 6;
    tilingDatafromBin->inputShapes[2] = 6;
    tilingDatafromBin->outShapes[0] = 4;
    tilingDatafromBin->outShapes[1] = 4;
    tilingDatafromBin->outShapes[2] = 4;
    tilingDatafromBin->kD = 2;
    tilingDatafromBin->kH = 2;
    tilingDatafromBin->kW = 2;
    tilingDatafromBin->sD = 2;
    tilingDatafromBin->sH = 2;
    tilingDatafromBin->sW = 2;
    tilingDatafromBin->pD = 1;
    tilingDatafromBin->pH = 1;
    tilingDatafromBin->pW = 1;
    tilingDatafromBin->dD = 1;
    tilingDatafromBin->dH = 1;
    tilingDatafromBin->dW = 1;
    tilingDatafromBin->batchesPerCore = 1;
    tilingDatafromBin->leftOverBatches = 0;
    tilingDatafromBin->partD = 8;
    tilingDatafromBin->partH = 8;
    tilingDatafromBin->partW = 8;
    tilingDatafromBin->minFloat = -std::numeric_limits<float>::infinity();
    tilingDatafromBin->ceilD = 0;
    tilingDatafromBin->ceilH = 0;
    tilingDatafromBin->ceilW = 0;
    tilingDatafromBin->sizeUb1 = 8 * 8 * 8 * 8;
    tilingDatafromBin->sizeUb2 = 8 * 8 * 8 * 8;
    tilingDatafromBin->sizeValues = 4 * 4 * 4 * 8;

    ICPU_SET_TILING_KEY(100000);

    ICPU_RUN_KF(max_pool3d_with_argmax_v2, blockDim, input, out, indices, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(input);
    AscendC::GmFree(out);
    AscendC::GmFree(indices);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(max_pool3d_with_argmax_v2_test, test_case_2)
{
    uint32_t blockDim = 1;
    size_t inputByteSize = 1 * 1 * 14 * 14 * 14 * sizeof(float);
    size_t outByteSize = 1 * 1 * 2 * 2 * 2 * sizeof(float);
    size_t indicesByteSize = 1 * 1 * 2 * 2 * 2 * sizeof(int32_t);
    size_t tilingDataSize = sizeof(MaxPool3DWithArgmaxV2SplitDTilingData);

    uint8_t* input = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* out = (uint8_t*)AscendC::GmAlloc(outByteSize);
    uint8_t* indices = (uint8_t*)AscendC::GmAlloc(indicesByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    char* path_ = get_current_dir_name();
    string path(path_);

    MaxPool3DWithArgmaxV2SplitDTilingData* tilingDatafromBin =
        reinterpret_cast<MaxPool3DWithArgmaxV2SplitDTilingData*>(tiling);
    tilingDatafromBin->inputShapes[0] = 14;
    tilingDatafromBin->inputShapes[1] = 14;
    tilingDatafromBin->inputShapes[2] = 14;
    tilingDatafromBin->outShapes[0] = 2;
    tilingDatafromBin->outShapes[1] = 2;
    tilingDatafromBin->outShapes[2] = 2;
    tilingDatafromBin->kD = 8;
    tilingDatafromBin->kH = 8;
    tilingDatafromBin->kW = 8;
    tilingDatafromBin->sD = 8;
    tilingDatafromBin->sH = 8;
    tilingDatafromBin->sW = 8;
    tilingDatafromBin->pD = 1;
    tilingDatafromBin->pH = 1;
    tilingDatafromBin->pW = 1;
    tilingDatafromBin->dD = 1;
    tilingDatafromBin->dH = 1;
    tilingDatafromBin->dW = 1;
    tilingDatafromBin->batchesPerCore = 1;
    tilingDatafromBin->leftOverBatches = 0;
    tilingDatafromBin->partsPerCore = 0;
    tilingDatafromBin->leftOverParts = 0;
    tilingDatafromBin->partD = 8;
    tilingDatafromBin->partH = 16;
    tilingDatafromBin->partW = 16;
    tilingDatafromBin->minFloat = -std::numeric_limits<float>::infinity();
    tilingDatafromBin->splitPointDIn[0] = 0;
    tilingDatafromBin->splitPointDOut[0] = 0;
    tilingDatafromBin->sizeIn[0] = 0;
    tilingDatafromBin->batchStart[0] = 0;
    tilingDatafromBin->ceilD = 0;
    tilingDatafromBin->ceilH = 0;
    tilingDatafromBin->ceilW = 0;
    tilingDatafromBin->sizeUb1 = 8 * 16 * 16 * 8;
    tilingDatafromBin->sizeUb2 = 8 * 16 * 16 * 8;
    tilingDatafromBin->sizeValues = 16 * 8;

    ICPU_SET_TILING_KEY(110000);

    ICPU_RUN_KF(max_pool3d_with_argmax_v2, blockDim, input, out, indices, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(input);
    AscendC::GmFree(out);
    AscendC::GmFree(indices);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(max_pool3d_with_argmax_v2_test, test_case_3)
{
    uint32_t blockDim = 1;
    size_t inputByteSize = 1 * 1 * 20 * 20 * 20 * sizeof(float);
    size_t outByteSize = 1 * 1 * 5 * 5 * 5 * sizeof(float);
    size_t indicesByteSize = 1 * 1 * 5 * 5 * 5 * sizeof(int32_t);
    size_t tilingDataSize = sizeof(MaxPool3DWithArgmaxV2SplitHTilingData);

    uint8_t* input = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* out = (uint8_t*)AscendC::GmAlloc(outByteSize);
    uint8_t* indices = (uint8_t*)AscendC::GmAlloc(indicesByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    char* path_ = get_current_dir_name();
    string path(path_);

    MaxPool3DWithArgmaxV2SplitHTilingData* tilingDatafromBin =
        reinterpret_cast<MaxPool3DWithArgmaxV2SplitHTilingData*>(tiling);
    tilingDatafromBin->inputShapes[0] = 20;
    tilingDatafromBin->inputShapes[1] = 20;
    tilingDatafromBin->inputShapes[2] = 20;
    tilingDatafromBin->outShapes[0] = 5;
    tilingDatafromBin->outShapes[1] = 5;
    tilingDatafromBin->outShapes[2] = 5;
    tilingDatafromBin->kD = 4;
    tilingDatafromBin->kH = 4;
    tilingDatafromBin->kW = 4;
    tilingDatafromBin->sD = 4;
    tilingDatafromBin->sH = 4;
    tilingDatafromBin->sW = 4;
    tilingDatafromBin->pD = 1;
    tilingDatafromBin->pH = 1;
    tilingDatafromBin->pW = 1;
    tilingDatafromBin->dD = 2;
    tilingDatafromBin->dH = 2;
    tilingDatafromBin->dW = 2;
    tilingDatafromBin->batchesPerCore = 1;
    tilingDatafromBin->leftOverBatches = 0;
    tilingDatafromBin->partsPerCore = 0;
    tilingDatafromBin->leftOverParts = 0;
    tilingDatafromBin->partD = 7;
    tilingDatafromBin->partH = 11;
    tilingDatafromBin->partW = 23;
    tilingDatafromBin->minFloat = -std::numeric_limits<float>::infinity();
    tilingDatafromBin->splitPointDIn[0] = 0;
    tilingDatafromBin->splitPointHIn[0] = 0;
    tilingDatafromBin->splitPointDOut[0] = 0;
    tilingDatafromBin->splitPointHOut[0] = 0;
    tilingDatafromBin->sizeIn[0] = 0;
    tilingDatafromBin->batchStart[0] = 0;
    tilingDatafromBin->ceilD = 0;
    tilingDatafromBin->ceilH = 0;
    tilingDatafromBin->ceilW = 1;
    tilingDatafromBin->sizeUb1 = 1776 * 8;
    tilingDatafromBin->sizeUb2 = 1776 * 8;
    tilingDatafromBin->sizeValues = 128 * 8;

    ICPU_SET_TILING_KEY(111000);

    ICPU_RUN_KF(max_pool3d_with_argmax_v2, blockDim, input, out, indices, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(input);
    AscendC::GmFree(out);
    AscendC::GmFree(indices);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(max_pool3d_with_argmax_v2_test, test_case_4)
{
    uint32_t blockDim = 1;
    size_t inputByteSize = 1 * 1 * 32 * 32 * 32 * sizeof(float);
    size_t outByteSize = 1 * 1 * 5 * 5 * 5 * sizeof(float);
    size_t indicesByteSize = 1 * 1 * 5 * 5 * 5 * sizeof(int32_t);
    size_t tilingDataSize = sizeof(MaxPool3DWithArgmaxV2SplitWTilingData);

    uint8_t* input = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* out = (uint8_t*)AscendC::GmAlloc(outByteSize);
    uint8_t* indices = (uint8_t*)AscendC::GmAlloc(indicesByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    char* path_ = get_current_dir_name();
    string path(path_);

    MaxPool3DWithArgmaxV2SplitWTilingData* tilingDatafromBin =
        reinterpret_cast<MaxPool3DWithArgmaxV2SplitWTilingData*>(tiling);
    tilingDatafromBin->inputShapes[0] = 32;
    tilingDatafromBin->inputShapes[1] = 32;
    tilingDatafromBin->inputShapes[2] = 32;
    tilingDatafromBin->outShapes[0] = 5;
    tilingDatafromBin->outShapes[1] = 5;
    tilingDatafromBin->outShapes[2] = 5;
    tilingDatafromBin->kD = 8;
    tilingDatafromBin->kH = 8;
    tilingDatafromBin->kW = 8;
    tilingDatafromBin->sD = 8;
    tilingDatafromBin->sH = 8;
    tilingDatafromBin->sW = 8;
    tilingDatafromBin->pD = 1;
    tilingDatafromBin->pH = 1;
    tilingDatafromBin->pW = 1;
    tilingDatafromBin->dD = 1;
    tilingDatafromBin->dH = 1;
    tilingDatafromBin->dW = 1;
    tilingDatafromBin->batchesPerCore = 1;
    tilingDatafromBin->leftOverBatches = 0;
    tilingDatafromBin->partsPerCore = 0;
    tilingDatafromBin->leftOverParts = 0;
    tilingDatafromBin->partD = 8;
    tilingDatafromBin->partH = 8;
    tilingDatafromBin->partW = 32;
    tilingDatafromBin->minFloat = -std::numeric_limits<float>::infinity();
    tilingDatafromBin->splitPointDIn[0] = 0;
    tilingDatafromBin->splitPointHIn[0] = 0;
    tilingDatafromBin->splitPointWIn[0] = 0;
    tilingDatafromBin->splitPointDOut[0] = 0;
    tilingDatafromBin->splitPointHOut[0] = 0;
    tilingDatafromBin->splitPointWOut[0] = 0;
    tilingDatafromBin->sizeIn[0] = 0;
    tilingDatafromBin->batchStart[0] = 0;
    tilingDatafromBin->ceilD = 0;
    tilingDatafromBin->ceilH = 0;
    tilingDatafromBin->ceilW = 0;
    tilingDatafromBin->sizeUb1 = 8 * 8 * 32 * 8;
    tilingDatafromBin->sizeUb2 = 8 * 8 * 32 * 8;
    tilingDatafromBin->sizeValues = 128 * 8;

    ICPU_SET_TILING_KEY(111100);

    ICPU_RUN_KF(max_pool3d_with_argmax_v2, blockDim, input, out, indices, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(input);
    AscendC::GmFree(out);
    AscendC::GmFree(indices);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(max_pool3d_with_argmax_v2_test, test_case_5)
{
    uint32_t blockDim = 1;
    size_t inputByteSize = 1 * 1 * 32 * 32 * 32 * sizeof(float);
    size_t outByteSize = 1 * 1 * 1 * 1 * 1 * sizeof(float);
    size_t indicesByteSize = 1 * 1 * 1 * 1 * 1 * sizeof(int32_t);
    size_t tilingDataSize = sizeof(MaxPool3DWithArgmaxV2HugeKernelTilingData);

    uint8_t* input = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* out = (uint8_t*)AscendC::GmAlloc(outByteSize);
    uint8_t* indices = (uint8_t*)AscendC::GmAlloc(indicesByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    char* path_ = get_current_dir_name();
    string path(path_);

    MaxPool3DWithArgmaxV2HugeKernelTilingData* tilingDatafromBin =
        reinterpret_cast<MaxPool3DWithArgmaxV2HugeKernelTilingData*>(tiling);
    tilingDatafromBin->inputShapes[0] = 32;
    tilingDatafromBin->inputShapes[1] = 32;
    tilingDatafromBin->inputShapes[2] = 32;
    tilingDatafromBin->outShapes[0] = 1;
    tilingDatafromBin->outShapes[1] = 1;
    tilingDatafromBin->outShapes[2] = 1;
    tilingDatafromBin->kD = 32;
    tilingDatafromBin->kH = 32;
    tilingDatafromBin->kW = 32;
    tilingDatafromBin->sD = 32;
    tilingDatafromBin->sH = 32;
    tilingDatafromBin->sW = 32;
    tilingDatafromBin->pD = 0;
    tilingDatafromBin->pH = 0;
    tilingDatafromBin->pW = 0;
    tilingDatafromBin->dD = 1;
    tilingDatafromBin->dH = 1;
    tilingDatafromBin->dW = 1;
    tilingDatafromBin->batchesPerCore = 1;
    tilingDatafromBin->leftOverBatches = 0;
    tilingDatafromBin->partD = 8;
    tilingDatafromBin->partH = 8;
    tilingDatafromBin->partW = 8;
    tilingDatafromBin->minFloat = -std::numeric_limits<float>::infinity();
    tilingDatafromBin->ceilD = 0;
    tilingDatafromBin->ceilH = 0;
    tilingDatafromBin->ceilW = 0;
    tilingDatafromBin->sizeUb1 = 8 * 8 * 8 * 8;
    tilingDatafromBin->sizeUb2 = 8 * 8 * 8 * 8;
    tilingDatafromBin->sizeValues = 16 * 8;

    ICPU_SET_TILING_KEY(111110);

    ICPU_RUN_KF(max_pool3d_with_argmax_v2, blockDim, input, out, indices, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(input);
    AscendC::GmFree(out);
    AscendC::GmFree(indices);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(max_pool3d_with_argmax_v2_test, test_case_no_expand_no_pad_float32)
{
    uint32_t blockDim = 24;
    size_t inputByteSize = 1 * 16 * 32 * 32 * 32 * sizeof(float);
    size_t outByteSize = 1 * 16 * 16 * 16 * 16 * sizeof(float);
    size_t indicesByteSize = 1 * 16 * 16 * 16 * 16 * sizeof(int32_t);
    size_t tilingDataSize = sizeof(MaxPool3DWithArgmaxV2NoExpandIndicesTilingData);

    uint8_t* input = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* out = (uint8_t*)AscendC::GmAlloc(outByteSize);
    uint8_t* indices = (uint8_t*)AscendC::GmAlloc(indicesByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    char* path_ = get_current_dir_name();
    string path(path_);

    MaxPool3DWithArgmaxV2NoExpandIndicesTilingData* tilingDatafromBin =
        reinterpret_cast<MaxPool3DWithArgmaxV2NoExpandIndicesTilingData*>(tiling);
    tilingDatafromBin->nc = 16;
    tilingDatafromBin->dx = 32;
    tilingDatafromBin->hx = 32;
    tilingDatafromBin->wx = 32;
    tilingDatafromBin->kd = 2;
    tilingDatafromBin->kh = 2;
    tilingDatafromBin->kw = 2;
    tilingDatafromBin->sd = 2;
    tilingDatafromBin->sh = 2;
    tilingDatafromBin->sw = 2;
    tilingDatafromBin->pf = 0;
    tilingDatafromBin->pb = 0;
    tilingDatafromBin->pt = 0;
    tilingDatafromBin->pd = 0;
    tilingDatafromBin->pl = 0;
    tilingDatafromBin->pr = 0;
    tilingDatafromBin->dy = 16;
    tilingDatafromBin->hy = 16;
    tilingDatafromBin->wy = 16;
    tilingDatafromBin->ncFactor = 16;
    tilingDatafromBin->ncTail = 16;
    tilingDatafromBin->ncOuter = 1;
    tilingDatafromBin->dyFactor = 1;
    tilingDatafromBin->dyTail = 1;
    tilingDatafromBin->dyOuter = 16;
    tilingDatafromBin->hyFactor = 6;
    tilingDatafromBin->hyTail = 4;
    tilingDatafromBin->hyOuter = 3;
    tilingDatafromBin->wyFactor = 16;
    tilingDatafromBin->wyTail = 16;
    tilingDatafromBin->wyOuter = 1;
    tilingDatafromBin->blockFactor = 2;
    tilingDatafromBin->blockTail = 2;
    tilingDatafromBin->totalIdx = 48;
    tilingDatafromBin->coreNums = 24;
    tilingDatafromBin->inputBufferSize = 49152;
    tilingDatafromBin->outputMaxBufferSize = 6144;
    tilingDatafromBin->outputIndiceBufferSize = 24576;
    tilingDatafromBin->indiceTempBufferSize = 4096;
    tilingDatafromBin->maskBufferSize = 512;

    ICPU_SET_TILING_KEY(300001UL);

    ICPU_RUN_KF(max_pool3d_with_argmax_v2, blockDim, input, out, indices, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(input);
    AscendC::GmFree(out);
    AscendC::GmFree(indices);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(max_pool3d_with_argmax_v2_test, test_case_no_expand_with_pad_float32)
{
    uint32_t blockDim = 22;
    size_t inputByteSize = 1 * 16 * 32 * 32 * 32 * sizeof(float);
    size_t outByteSize = 1 * 17 * 17 * 17 * 17 * sizeof(float);
    size_t indicesByteSize = 1 * 17 * 17 * 17 * 17 * sizeof(int32_t);
    size_t tilingDataSize = sizeof(MaxPool3DWithArgmaxV2NoExpandIndicesTilingData);

    uint8_t* input = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* out = (uint8_t*)AscendC::GmAlloc(outByteSize);
    uint8_t* indices = (uint8_t*)AscendC::GmAlloc(indicesByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    char* path_ = get_current_dir_name();
    string path(path_);

    MaxPool3DWithArgmaxV2NoExpandIndicesTilingData* tilingDatafromBin =
        reinterpret_cast<MaxPool3DWithArgmaxV2NoExpandIndicesTilingData*>(tiling);
    tilingDatafromBin->nc = 16;
    tilingDatafromBin->dx = 32;
    tilingDatafromBin->hx = 32;
    tilingDatafromBin->wx = 32;
    tilingDatafromBin->kd = 2;
    tilingDatafromBin->kh = 2;
    tilingDatafromBin->kw = 2;
    tilingDatafromBin->sd = 2;
    tilingDatafromBin->sh = 2;
    tilingDatafromBin->sw = 2;
    tilingDatafromBin->pf = 1;
    tilingDatafromBin->pb = 1;
    tilingDatafromBin->pt = 1;
    tilingDatafromBin->pd = 1;
    tilingDatafromBin->pl = 1;
    tilingDatafromBin->pr = 1;
    tilingDatafromBin->dy = 17;
    tilingDatafromBin->hy = 17;
    tilingDatafromBin->wy = 17;
    tilingDatafromBin->ncFactor = 16;
    tilingDatafromBin->ncTail = 16;
    tilingDatafromBin->ncOuter = 1;
    tilingDatafromBin->dyFactor = 1;
    tilingDatafromBin->dyTail = 1;
    tilingDatafromBin->dyOuter = 17;
    tilingDatafromBin->hyFactor = 4;
    tilingDatafromBin->hyTail = 1;
    tilingDatafromBin->hyOuter = 5;
    tilingDatafromBin->wyFactor = 17;
    tilingDatafromBin->wyTail = 17;
    tilingDatafromBin->wyOuter = 1;
    tilingDatafromBin->blockFactor = 4;
    tilingDatafromBin->blockTail = 1;
    tilingDatafromBin->totalIdx = 85;
    tilingDatafromBin->coreNums = 22;
    tilingDatafromBin->inputBufferSize = 40960;
    tilingDatafromBin->outputMaxBufferSize = 8192;
    tilingDatafromBin->outputIndiceBufferSize = 32768;
    tilingDatafromBin->indiceTempBufferSize = 8192;
    tilingDatafromBin->maskBufferSize = 1024;

    ICPU_SET_TILING_KEY(300002UL);

    ICPU_RUN_KF(max_pool3d_with_argmax_v2, blockDim, input, out, indices, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(input);
    AscendC::GmFree(out);
    AscendC::GmFree(indices);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(max_pool3d_with_argmax_v2_test, test_case_for_bigkernel_float32_with_dhwContinue_dhw_bigger_than_10240)
{
    uint32_t blockDim = 1;
    size_t inputByteSize = 1 * 1 * 12 * 32 * 32 * sizeof(float);
    size_t outByteSize = 1 * 1 * 1 * 1 * 1 * sizeof(float);
    size_t indicesByteSize = 1 * 1 * 1 * 1 * 1 * sizeof(int32_t);
    size_t tilingDataSize = sizeof(MaxPool3DWithArgmaxV2BigKernelTilingData);

    uint8_t* input = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* out = (uint8_t*)AscendC::GmAlloc(outByteSize);
    uint8_t* indices = (uint8_t*)AscendC::GmAlloc(indicesByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    char* path_ = get_current_dir_name();
    string path(path_);

    MaxPool3DWithArgmaxV2BigKernelTilingData* tilingDatafromBin =
        reinterpret_cast<MaxPool3DWithArgmaxV2BigKernelTilingData*>(tiling);
    tilingDatafromBin->inputShapes[0] = 12;
    tilingDatafromBin->inputShapes[1] = 32;
    tilingDatafromBin->inputShapes[2] = 32;
    tilingDatafromBin->outShapes[0] = 1;
    tilingDatafromBin->outShapes[1] = 1;
    tilingDatafromBin->outShapes[2] = 1;
    tilingDatafromBin->kD = 12;
    tilingDatafromBin->kH = 32;
    tilingDatafromBin->kW = 32;
    tilingDatafromBin->sD = 12;
    tilingDatafromBin->sH = 32;
    tilingDatafromBin->sW = 32;
    tilingDatafromBin->pD = 0;
    tilingDatafromBin->pH = 0;
    tilingDatafromBin->pW = 0;
    tilingDatafromBin->dD = 1;
    tilingDatafromBin->dH = 1;
    tilingDatafromBin->dW = 1;
    tilingDatafromBin->blockFactor = 0;
    tilingDatafromBin->blockTail = 1;
    tilingDatafromBin->totalIdx = 1;
    tilingDatafromBin->coreNums = 1;

    ICPU_SET_TILING_KEY(311110);

    ICPU_RUN_KF(max_pool3d_with_argmax_v2, blockDim, input, out, indices, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(input);
    AscendC::GmFree(out);
    AscendC::GmFree(indices);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(max_pool3d_with_argmax_v2_test, test_case_for_bigkernel_float32_with_dhwContinue_dhw_smaller_than_10240)
{
    uint32_t blockDim = 1;
    size_t inputByteSize = 1 * 1 * 8 * 32 * 32 * sizeof(float);
    size_t outByteSize = 1 * 1 * 1 * 1 * 1 * sizeof(float);
    size_t indicesByteSize = 1 * 1 * 1 * 1 * 1 * sizeof(int32_t);
    size_t tilingDataSize = sizeof(MaxPool3DWithArgmaxV2BigKernelTilingData);

    uint8_t* input = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* out = (uint8_t*)AscendC::GmAlloc(outByteSize);
    uint8_t* indices = (uint8_t*)AscendC::GmAlloc(indicesByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    char* path_ = get_current_dir_name();
    string path(path_);

    MaxPool3DWithArgmaxV2BigKernelTilingData* tilingDatafromBin =
        reinterpret_cast<MaxPool3DWithArgmaxV2BigKernelTilingData*>(tiling);
    tilingDatafromBin->inputShapes[0] = 8;
    tilingDatafromBin->inputShapes[1] = 32;
    tilingDatafromBin->inputShapes[2] = 32;
    tilingDatafromBin->outShapes[0] = 1;
    tilingDatafromBin->outShapes[1] = 1;
    tilingDatafromBin->outShapes[2] = 1;
    tilingDatafromBin->kD = 8;
    tilingDatafromBin->kH = 32;
    tilingDatafromBin->kW = 32;
    tilingDatafromBin->sD = 8;
    tilingDatafromBin->sH = 32;
    tilingDatafromBin->sW = 32;
    tilingDatafromBin->pD = 0;
    tilingDatafromBin->pH = 0;
    tilingDatafromBin->pW = 0;
    tilingDatafromBin->dD = 1;
    tilingDatafromBin->dH = 1;
    tilingDatafromBin->dW = 1;
    tilingDatafromBin->blockFactor = 0;
    tilingDatafromBin->blockTail = 1;
    tilingDatafromBin->totalIdx = 1;
    tilingDatafromBin->coreNums = 1;

    ICPU_SET_TILING_KEY(311110);

    ICPU_RUN_KF(max_pool3d_with_argmax_v2, blockDim, input, out, indices, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(input);
    AscendC::GmFree(out);
    AscendC::GmFree(indices);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(max_pool3d_with_argmax_v2_test, test_case_for_bigkernel_float32_with_hwContinue_dhw_smaller_than_10240)
{
    uint32_t blockDim = 8;
    size_t inputByteSize = 1 * 1 * 3 * 64 * 32 * sizeof(float);
    size_t outByteSize = 1 * 1 * 2 * 4 * 1 * sizeof(float);
    size_t indicesByteSize = 1 * 1 * 1 * 1 * 1 * sizeof(int32_t);
    size_t tilingDataSize = sizeof(MaxPool3DWithArgmaxV2BigKernelTilingData);

    uint8_t* input = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* out = (uint8_t*)AscendC::GmAlloc(outByteSize);
    uint8_t* indices = (uint8_t*)AscendC::GmAlloc(indicesByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    char* path_ = get_current_dir_name();
    string path(path_);

    MaxPool3DWithArgmaxV2BigKernelTilingData* tilingDatafromBin =
        reinterpret_cast<MaxPool3DWithArgmaxV2BigKernelTilingData*>(tiling);
    tilingDatafromBin->inputShapes[0] = 3;
    tilingDatafromBin->inputShapes[1] = 64;
    tilingDatafromBin->inputShapes[2] = 32;
    tilingDatafromBin->outShapes[0] = 2;
    tilingDatafromBin->outShapes[1] = 4;
    tilingDatafromBin->outShapes[2] = 1;
    tilingDatafromBin->kD = 1;
    tilingDatafromBin->kH = 16;
    tilingDatafromBin->kW = 32;
    tilingDatafromBin->sD = 2;
    tilingDatafromBin->sH = 16;
    tilingDatafromBin->sW = 32;
    tilingDatafromBin->pD = 0;
    tilingDatafromBin->pH = 0;
    tilingDatafromBin->pW = 0;
    tilingDatafromBin->dD = 1;
    tilingDatafromBin->dH = 1;
    tilingDatafromBin->dW = 1;
    tilingDatafromBin->blockFactor = 0;
    tilingDatafromBin->blockTail = 8;
    tilingDatafromBin->totalIdx = 8;
    tilingDatafromBin->coreNums = 8;

    ICPU_SET_TILING_KEY(311110);

    ICPU_RUN_KF(max_pool3d_with_argmax_v2, blockDim, input, out, indices, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(input);
    AscendC::GmFree(out);
    AscendC::GmFree(indices);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(max_pool3d_with_argmax_v2_test, test_case_for_bigkernel_float16_with_hwContinue_hw_bigger_than_10240)
{
    uint32_t blockDim = 4;
    size_t inputByteSize = 1 * 1 * 9 * 4409 * 3 * sizeof(half);
    size_t outByteSize = 1 * 1 * 4 * 1 * 1 * sizeof(half);
    size_t indicesByteSize = 1 * 1 * 1 * 1 * 1 * sizeof(int32_t);
    size_t tilingDataSize = sizeof(MaxPool3DWithArgmaxV2BigKernelTilingData);

    uint8_t* input = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* out = (uint8_t*)AscendC::GmAlloc(outByteSize);
    uint8_t* indices = (uint8_t*)AscendC::GmAlloc(indicesByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    char* path_ = get_current_dir_name();
    string path(path_);

    MaxPool3DWithArgmaxV2BigKernelTilingData* tilingDatafromBin =
        reinterpret_cast<MaxPool3DWithArgmaxV2BigKernelTilingData*>(tiling);
    tilingDatafromBin->inputShapes[0] = 9;
    tilingDatafromBin->inputShapes[1] = 4409;
    tilingDatafromBin->inputShapes[2] = 3;
    tilingDatafromBin->outShapes[0] = 4;
    tilingDatafromBin->outShapes[1] = 1;
    tilingDatafromBin->outShapes[2] = 1;
    tilingDatafromBin->kD = 2;
    tilingDatafromBin->kH = 4169;
    tilingDatafromBin->kW = 3;
    tilingDatafromBin->sD = 2;
    tilingDatafromBin->sH = 4169;
    tilingDatafromBin->sW = 3;
    tilingDatafromBin->pD = 0;
    tilingDatafromBin->pH = 0;
    tilingDatafromBin->pW = 0;
    tilingDatafromBin->dD = 1;
    tilingDatafromBin->dH = 1;
    tilingDatafromBin->dW = 1;
    tilingDatafromBin->blockFactor = 0;
    tilingDatafromBin->blockTail = 4;
    tilingDatafromBin->totalIdx = 4;
    tilingDatafromBin->coreNums = 4;

    ICPU_SET_TILING_KEY(311111);

    ICPU_RUN_KF(max_pool3d_with_argmax_v2, blockDim, input, out, indices, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(input);
    AscendC::GmFree(out);
    AscendC::GmFree(indices);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(max_pool3d_with_argmax_v2_test, test_case_for_bigkernel_float16_with_hwContinue_hw_smaller_than_10240_padding)
{
    uint32_t blockDim = 8;
    size_t inputByteSize = 1 * 1 * 40 * 69 * 35 * sizeof(half);
    size_t outByteSize = 1 * 1 * 2 * 4 * 1 * sizeof(half);
    size_t indicesByteSize = 1 * 1 * 1 * 1 * 1 * sizeof(int32_t);
    size_t tilingDataSize = sizeof(MaxPool3DWithArgmaxV2BigKernelTilingData);

    uint8_t* input = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* out = (uint8_t*)AscendC::GmAlloc(outByteSize);
    uint8_t* indices = (uint8_t*)AscendC::GmAlloc(indicesByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    char* path_ = get_current_dir_name();
    string path(path_);

    MaxPool3DWithArgmaxV2BigKernelTilingData* tilingDatafromBin =
        reinterpret_cast<MaxPool3DWithArgmaxV2BigKernelTilingData*>(tiling);
    tilingDatafromBin->inputShapes[0] = 40;
    tilingDatafromBin->inputShapes[1] = 69;
    tilingDatafromBin->inputShapes[2] = 35;
    tilingDatafromBin->outShapes[0] = 2;
    tilingDatafromBin->outShapes[1] = 4;
    tilingDatafromBin->outShapes[2] = 1;
    tilingDatafromBin->kD = 20;
    tilingDatafromBin->kH = 16;
    tilingDatafromBin->kW = 35;
    tilingDatafromBin->sD = 3;
    tilingDatafromBin->sH = 16;
    tilingDatafromBin->sW = 15;
    tilingDatafromBin->pD = 0;
    tilingDatafromBin->pH = 1;
    tilingDatafromBin->pW = 0;
    tilingDatafromBin->dD = 1;
    tilingDatafromBin->dH = 1;
    tilingDatafromBin->dW = 1;
    tilingDatafromBin->blockFactor = 8;
    tilingDatafromBin->blockTail = 0;
    tilingDatafromBin->totalIdx = 8;
    tilingDatafromBin->coreNums = 8;

    ICPU_SET_TILING_KEY(311111);

    ICPU_RUN_KF(max_pool3d_with_argmax_v2, blockDim, input, out, indices, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(input);
    AscendC::GmFree(out);
    AscendC::GmFree(indices);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(max_pool3d_with_argmax_v2_test, test_case_for_bigkernel_bfloat16_with_wContinue_dhw_smaller_than_10240_padding)
{
    uint32_t blockDim = 10;
    size_t inputByteSize = 1 * 1 * 8 * 78 * 11 * sizeof(bfloat16_t);
    size_t outByteSize = 1 * 1 * 5 * 1 * 2 * sizeof(bfloat16_t);
    size_t indicesByteSize = 1 * 1 * 1 * 1 * 1 * sizeof(int32_t);
    size_t tilingDataSize = sizeof(MaxPool3DWithArgmaxV2BigKernelTilingData);

    uint8_t* input = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* out = (uint8_t*)AscendC::GmAlloc(outByteSize);
    uint8_t* indices = (uint8_t*)AscendC::GmAlloc(indicesByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    char* path_ = get_current_dir_name();
    string path(path_);

    MaxPool3DWithArgmaxV2BigKernelTilingData* tilingDatafromBin =
        reinterpret_cast<MaxPool3DWithArgmaxV2BigKernelTilingData*>(tiling);
    tilingDatafromBin->inputShapes[0] = 8;
    tilingDatafromBin->inputShapes[1] = 78;
    tilingDatafromBin->inputShapes[2] = 11;
    tilingDatafromBin->outShapes[0] = 5;
    tilingDatafromBin->outShapes[1] = 1;
    tilingDatafromBin->outShapes[2] = 2;
    tilingDatafromBin->kD = 2;
    tilingDatafromBin->kH = 78;
    tilingDatafromBin->kW = 5;
    tilingDatafromBin->sD = 2;
    tilingDatafromBin->sH = 57;
    tilingDatafromBin->sW = 5;
    tilingDatafromBin->pD = 1;
    tilingDatafromBin->pH = 1;
    tilingDatafromBin->pW = 1;
    tilingDatafromBin->dD = 1;
    tilingDatafromBin->dH = 1;
    tilingDatafromBin->dW = 1;
    tilingDatafromBin->blockFactor = 0;
    tilingDatafromBin->blockTail = 10;
    tilingDatafromBin->totalIdx = 10;
    tilingDatafromBin->coreNums = 10;

    ICPU_SET_TILING_KEY(311112);

    ICPU_RUN_KF(max_pool3d_with_argmax_v2, blockDim, input, out, indices, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(input);
    AscendC::GmFree(out);
    AscendC::GmFree(indices);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(max_pool3d_with_argmax_v2_test, test_case_for_bigkernel_bfloat16_with_wContinue_w_bigger_than_10240)
{
    uint32_t blockDim = 9;
    size_t inputByteSize = 1 * 1 * 3 * 3 * 10245 * sizeof(bfloat16_t);
    size_t outByteSize = 1 * 1 * 3 * 3 * 1 * sizeof(bfloat16_t);
    size_t indicesByteSize = 1 * 1 * 1 * 1 * 1 * sizeof(int32_t);
    size_t tilingDataSize = sizeof(MaxPool3DWithArgmaxV2BigKernelTilingData);

    uint8_t* input = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* out = (uint8_t*)AscendC::GmAlloc(outByteSize);
    uint8_t* indices = (uint8_t*)AscendC::GmAlloc(indicesByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    char* path_ = get_current_dir_name();
    string path(path_);

    MaxPool3DWithArgmaxV2BigKernelTilingData* tilingDatafromBin =
        reinterpret_cast<MaxPool3DWithArgmaxV2BigKernelTilingData*>(tiling);
    tilingDatafromBin->inputShapes[0] = 3;
    tilingDatafromBin->inputShapes[1] = 3;
    tilingDatafromBin->inputShapes[2] = 10245;
    tilingDatafromBin->outShapes[0] = 3;
    tilingDatafromBin->outShapes[1] = 3;
    tilingDatafromBin->outShapes[2] = 1;
    tilingDatafromBin->kD = 1;
    tilingDatafromBin->kH = 1;
    tilingDatafromBin->kW = 10243;
    tilingDatafromBin->sD = 1;
    tilingDatafromBin->sH = 1;
    tilingDatafromBin->sW = 10243;
    tilingDatafromBin->pD = 0;
    tilingDatafromBin->pH = 0;
    tilingDatafromBin->pW = 0;
    tilingDatafromBin->dD = 1;
    tilingDatafromBin->dH = 1;
    tilingDatafromBin->dW = 1;
    tilingDatafromBin->blockFactor = 0;
    tilingDatafromBin->blockTail = 9;
    tilingDatafromBin->totalIdx = 9;
    tilingDatafromBin->coreNums = 9;

    ICPU_SET_TILING_KEY(311112);

    ICPU_RUN_KF(max_pool3d_with_argmax_v2, blockDim, input, out, indices, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(input);
    AscendC::GmFree(out);
    AscendC::GmFree(indices);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(max_pool3d_with_argmax_v2_test, test_case_for_bigkernel_bfloat16_with_wContinue_hw_bigger_than_10240)
{
    uint32_t blockDim = 6;
    size_t inputByteSize = 1 * 1 * 9 * 4409 * 9 * sizeof(bfloat16_t);
    size_t outByteSize = 1 * 1 * 2 * 1 * 3 * sizeof(bfloat16_t);
    size_t indicesByteSize = 1 * 1 * 1 * 1 * 1 * sizeof(int32_t);
    size_t tilingDataSize = sizeof(MaxPool3DWithArgmaxV2BigKernelTilingData);

    uint8_t* input = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* out = (uint8_t*)AscendC::GmAlloc(outByteSize);
    uint8_t* indices = (uint8_t*)AscendC::GmAlloc(indicesByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    char* path_ = get_current_dir_name();
    string path(path_);

    MaxPool3DWithArgmaxV2BigKernelTilingData* tilingDatafromBin =
        reinterpret_cast<MaxPool3DWithArgmaxV2BigKernelTilingData*>(tiling);
    tilingDatafromBin->inputShapes[0] = 9;
    tilingDatafromBin->inputShapes[1] = 4409;
    tilingDatafromBin->inputShapes[2] = 9;
    tilingDatafromBin->outShapes[0] = 2;
    tilingDatafromBin->outShapes[1] = 1;
    tilingDatafromBin->outShapes[2] = 3;
    tilingDatafromBin->kD = 4;
    tilingDatafromBin->kH = 4409;
    tilingDatafromBin->kW = 3;
    tilingDatafromBin->sD = 4;
    tilingDatafromBin->sH = 4409;
    tilingDatafromBin->sW = 3;
    tilingDatafromBin->pD = 0;
    tilingDatafromBin->pH = 0;
    tilingDatafromBin->pW = 0;
    tilingDatafromBin->dD = 1;
    tilingDatafromBin->dH = 1;
    tilingDatafromBin->dW = 1;
    tilingDatafromBin->blockFactor = 0;
    tilingDatafromBin->blockTail = 6;
    tilingDatafromBin->totalIdx = 6;
    tilingDatafromBin->coreNums = 6;

    ICPU_SET_TILING_KEY(311112);

    ICPU_RUN_KF(max_pool3d_with_argmax_v2, blockDim, input, out, indices, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(input);
    AscendC::GmFree(out);
    AscendC::GmFree(indices);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(max_pool3d_with_argmax_v2_test, test_case_for_bigkernel_bfloat16_with_wContinue_dhw_bigger_than_10240)
{
    uint32_t blockDim = 20;
    size_t inputByteSize = 1 * 1 * 300 * 500 * 500 * sizeof(bfloat16_t);
    size_t outByteSize = 1 * 1 * 7 * 3 * 3 * sizeof(bfloat16_t);
    size_t indicesByteSize = 1 * 1 * 1 * 1 * 1 * sizeof(int32_t);
    size_t tilingDataSize = sizeof(MaxPool3DWithArgmaxV2BigKernelTilingData);

    uint8_t* input = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* out = (uint8_t*)AscendC::GmAlloc(outByteSize);
    uint8_t* indices = (uint8_t*)AscendC::GmAlloc(indicesByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    char* path_ = get_current_dir_name();
    string path(path_);

    MaxPool3DWithArgmaxV2BigKernelTilingData* tilingDatafromBin =
        reinterpret_cast<MaxPool3DWithArgmaxV2BigKernelTilingData*>(tiling);
    tilingDatafromBin->inputShapes[0] = 99;
    tilingDatafromBin->inputShapes[1] = 60;
    tilingDatafromBin->inputShapes[2] = 74;
    tilingDatafromBin->outShapes[0] = 7;
    tilingDatafromBin->outShapes[1] = 3;
    tilingDatafromBin->outShapes[2] = 3;
    tilingDatafromBin->kD = 30;
    tilingDatafromBin->kH = 30;
    tilingDatafromBin->kW = 15;
    tilingDatafromBin->sD = 10;
    tilingDatafromBin->sH = 20;
    tilingDatafromBin->sW = 12;
    tilingDatafromBin->pD = 0;
    tilingDatafromBin->pH = 0;
    tilingDatafromBin->pW = 0;
    tilingDatafromBin->dD = 1;
    tilingDatafromBin->dH = 1;
    tilingDatafromBin->dW = 1;
    tilingDatafromBin->blockFactor = 3;
    tilingDatafromBin->blockTail = 3;
    tilingDatafromBin->totalIdx = 63;
    tilingDatafromBin->coreNums = 20;

    ICPU_SET_TILING_KEY(311112);

    ICPU_RUN_KF(max_pool3d_with_argmax_v2, blockDim, input, out, indices, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(input);
    AscendC::GmFree(out);
    AscendC::GmFree(indices);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}