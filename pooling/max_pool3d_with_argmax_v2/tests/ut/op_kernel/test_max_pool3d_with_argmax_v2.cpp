/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 /*!
 * \file test_max_pool3d_with_argmax_v2.cpp
 * \brief
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

TEST_F(max_pool3d_with_argmax_v2_test, test_case_no_expand_no_pad_float32)
{
    AscendC:: SetKernelMode(KernelMode::AIV_MODE);
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
    AscendC:: SetKernelMode(KernelMode::AIV_MODE);
    uint32_t blockDim = 2;
    size_t inputByteSize = 1 * 1 * 6 * 6 * 6 * sizeof(float);
    size_t outByteSize = 1 * 1 * 4 * 4 * 4 * sizeof(float);
    size_t indicesByteSize = 1 * 1 * 4 * 4 * 4 * sizeof(int32_t);
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
    tilingDatafromBin->nc = 1;
    tilingDatafromBin->dx = 6;
    tilingDatafromBin->hx = 6;
    tilingDatafromBin->wx = 6;
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
    tilingDatafromBin->dy = 4;
    tilingDatafromBin->hy = 4;
    tilingDatafromBin->wy = 4;
    tilingDatafromBin->ncFactor = 1;
    tilingDatafromBin->ncTail = 1;
    tilingDatafromBin->ncOuter = 1;
    tilingDatafromBin->dyFactor = 3;
    tilingDatafromBin->dyTail = 1;
    tilingDatafromBin->dyOuter = 2;
    tilingDatafromBin->hyFactor = 4;
    tilingDatafromBin->hyTail = 4;
    tilingDatafromBin->hyOuter = 1;
    tilingDatafromBin->wyFactor = 4;
    tilingDatafromBin->wyTail = 4;
    tilingDatafromBin->wyOuter = 1;
    tilingDatafromBin->blockFactor = 1;
    tilingDatafromBin->blockTail = 1;
    tilingDatafromBin->totalIdx = 2;
    tilingDatafromBin->coreNums = 2;
    tilingDatafromBin->inputBufferSize = 24576;
    tilingDatafromBin->outputMaxBufferSize = 12288;
    tilingDatafromBin->outputIndiceBufferSize = 49152;
    tilingDatafromBin->indiceTempBufferSize = 4096;
    tilingDatafromBin->maskBufferSize = 512;

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
    AscendC:: SetKernelMode(KernelMode::AIV_MODE);
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
    AscendC:: SetKernelMode(KernelMode::AIV_MODE);
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
    AscendC:: SetKernelMode(KernelMode::AIV_MODE);
    uint32_t blockDim = 6;
    size_t inputByteSize = 1 * 1 * 3 * 64 * 32 * sizeof(float);
    size_t outByteSize = 1 * 1 * 2 * 4 * 1 * sizeof(float);
    size_t indicesByteSize = 1 * 1 * 2 * 4 * 1 * sizeof(int32_t);
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
    tilingDatafromBin->kD = 2;
    tilingDatafromBin->kH = 32;
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
    tilingDatafromBin->blockTail = 6;
    tilingDatafromBin->totalIdx = 6;
    tilingDatafromBin->coreNums = 6;

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
    AscendC:: SetKernelMode(KernelMode::AIV_MODE);
    uint32_t blockDim = 4;
    size_t inputByteSize = 1 * 1 * 9 * 4409 * 3 * sizeof(half);
    size_t outByteSize = 1 * 1 * 4 * 1 * 1 * sizeof(half);
    size_t indicesByteSize = 1 * 1 * 4 * 1 * 1 * sizeof(int32_t);
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
    AscendC:: SetKernelMode(KernelMode::AIV_MODE);
    uint32_t blockDim = 8;
    size_t inputByteSize = 1 * 1 * 40 * 69 * 35 * sizeof(half);
    size_t outByteSize = 1 * 1 * 2 * 4 * 1 * sizeof(half);
    size_t indicesByteSize = 1 * 1 * 2 * 4 * 1 * sizeof(int32_t);
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
    tilingDatafromBin->kH = 17;
    tilingDatafromBin->kW = 35;
    tilingDatafromBin->sD = 20;
    tilingDatafromBin->sH = 17;
    tilingDatafromBin->sW = 35;
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
    AscendC:: SetKernelMode(KernelMode::AIV_MODE);
    uint32_t blockDim = 10;
    size_t inputByteSize = 1 * 1 * 8 * 78 * 11 * sizeof(bfloat16_t);
    size_t outByteSize = 1 * 1 * 5 * 1 * 2 * sizeof(bfloat16_t);
    size_t indicesByteSize = 1 * 1 * 5 * 1 * 2 * sizeof(int32_t);
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
    AscendC:: SetKernelMode(KernelMode::AIV_MODE);
    uint32_t blockDim = 9;
    size_t inputByteSize = 1 * 1 * 3 * 3 * 10245 * sizeof(bfloat16_t);
    size_t outByteSize = 1 * 1 * 3 * 3 * 1 * sizeof(bfloat16_t);
    size_t indicesByteSize = 1 * 1 * 3 * 3 * 1 * sizeof(int32_t);
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
    size_t indicesByteSize = 1 * 1 * 2 * 1 * 3 * sizeof(int32_t);
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
    size_t indicesByteSize = 1 * 1 * 7 * 3 * 3 * sizeof(int32_t);
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

TEST_F(max_pool3d_with_argmax_v2_test, test_case_for_simt_float32_int32_NCDHW_600001)
{
    AscendC:: SetKernelMode(KernelMode::AIV_MODE);
    uint32_t blockDim = 20;
    size_t inputByteSize = 4 * 30 * 14 * 8 * 14 * sizeof(float);
    size_t outByteSize = 4 * 30 * 7 * 2 * 3 * sizeof(float);
    size_t indicesByteSize = 4 * 30 * 7 * 2 * 3 * sizeof(int32_t);
    size_t tilingDataSize = sizeof(MaxPool3DWithArgmaxV2SimtTilingData);

    uint8_t* input = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* out = (uint8_t*)AscendC::GmAlloc(outByteSize);
    uint8_t* indices = (uint8_t*)AscendC::GmAlloc(indicesByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    char* path_ = get_current_dir_name();
    string path(path_);

    MaxPool3DWithArgmaxV2SimtTilingData* tilingDatafromBin =
        reinterpret_cast<MaxPool3DWithArgmaxV2SimtTilingData*>(tiling);
    tilingDatafromBin->threadNums = 256;
    tilingDatafromBin->blockNums = 20;
    tilingDatafromBin->nDim = 4;
    tilingDatafromBin->cDim = 30;
    tilingDatafromBin->dInDim = 14;
    tilingDatafromBin->hInDim = 8;
    tilingDatafromBin->wInDim = 14;
    tilingDatafromBin->dOutDim = 7;
    tilingDatafromBin->hOutDim = 2;
    tilingDatafromBin->dOutDim = 3;
    tilingDatafromBin->kSizeD = 2;
    tilingDatafromBin->kSizeH = 7;
    tilingDatafromBin->kSizeW = 6;
    tilingDatafromBin->stridesD = 2;
    tilingDatafromBin->stridesH = 7;
    tilingDatafromBin->stridesW = 6;
    tilingDatafromBin->padD = 0;
    tilingDatafromBin->padH = 0;
    tilingDatafromBin->padW = 0;
    tilingDatafromBin->dilationD = 1;
    tilingDatafromBin->dilationH = 1;
    tilingDatafromBin->dilationW = 1;

    ICPU_SET_TILING_KEY(600001);

    ICPU_RUN_KF(max_pool3d_with_argmax_v2, blockDim, input, out, indices, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(input);
    AscendC::GmFree(out);
    AscendC::GmFree(indices);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(max_pool3d_with_argmax_v2_test, test_case_for_simt_float16_NCDHW_600002)
{
    AscendC:: SetKernelMode(KernelMode::AIV_MODE);
    uint32_t blockDim = 19;
    size_t inputByteSize = 21 * 21 * 15 * 24 * 4 * sizeof(half);
    size_t outByteSize = 21 * 4 * 2 * 7 * 4 * sizeof(half);
    size_t indicesByteSize = 21 * 4 * 2 * 7 * 4 * sizeof(int64_t);
    size_t tilingDataSize = sizeof(MaxPool3DWithArgmaxV2SimtTilingData);

    uint8_t* input = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* out = (uint8_t*)AscendC::GmAlloc(outByteSize);
    uint8_t* indices = (uint8_t*)AscendC::GmAlloc(indicesByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    char* path_ = get_current_dir_name();
    string path(path_);

    MaxPool3DWithArgmaxV2SimtTilingData* tilingDatafromBin =
        reinterpret_cast<MaxPool3DWithArgmaxV2SimtTilingData*>(tiling);
    tilingDatafromBin->threadNums = 256;
    tilingDatafromBin->blockNums = 19;
    tilingDatafromBin->nDim = 21;
    tilingDatafromBin->cDim = 4;
    tilingDatafromBin->dInDim = 21;
    tilingDatafromBin->hInDim = 15;
    tilingDatafromBin->wInDim = 24;
    tilingDatafromBin->dOutDim = 4;
    tilingDatafromBin->hOutDim = 2;
    tilingDatafromBin->dOutDim = 7;
    tilingDatafromBin->kSizeD = 4;
    tilingDatafromBin->kSizeH = 4;
    tilingDatafromBin->kSizeW = 4;
    tilingDatafromBin->stridesD = 5;
    tilingDatafromBin->stridesH = 8;
    tilingDatafromBin->stridesW = 3;
    tilingDatafromBin->padD = 0;
    tilingDatafromBin->padH = 0;
    tilingDatafromBin->padW = 0;
    tilingDatafromBin->dilationD = 1;
    tilingDatafromBin->dilationH = 1;
    tilingDatafromBin->dilationW = 1;

    ICPU_SET_TILING_KEY(600002);

    ICPU_RUN_KF(max_pool3d_with_argmax_v2, blockDim, input, out, indices, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(input);
    AscendC::GmFree(out);
    AscendC::GmFree(indices);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(max_pool3d_with_argmax_v2_test, test_case_for_bigkernel_no_split)
{
    AscendC:: SetKernelMode(KernelMode::AIV_MODE);
    uint32_t blockDim = 20;
    size_t inputByteSize = 1 * 1 * 14 * 12 * 122 * sizeof(bfloat16_t);
    size_t outByteSize = 1 * 1 * 1 * 1 * 1 * sizeof(bfloat16_t);
    size_t indicesByteSize = 1 * 1 * 1 * 1 * 1 * sizeof(int32_t);
    size_t tilingDataSize = sizeof(MaxPool3DWithArgmaxV2BigKernelRegbaseTilingData);

    uint8_t* input = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* out = (uint8_t*)AscendC::GmAlloc(outByteSize);
    uint8_t* indices = (uint8_t*)AscendC::GmAlloc(indicesByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    char* path_ = get_current_dir_name();
    string path(path_);

    MaxPool3DWithArgmaxV2BigKernelRegbaseTilingData* tilingDatafromBin =
        reinterpret_cast<MaxPool3DWithArgmaxV2BigKernelRegbaseTilingData*>(tiling);
    tilingDatafromBin->dInDim = 14;
    tilingDatafromBin->hInDim = 12;
    tilingDatafromBin->wInDim = 122;
    tilingDatafromBin->dOutDim = 1;
    tilingDatafromBin->hOutDim = 1;
    tilingDatafromBin->wOutDim = 1;
    tilingDatafromBin->kD = 16;
    tilingDatafromBin->kH = 16;
    tilingDatafromBin->kW = 128;
    tilingDatafromBin->sD = 2;
    tilingDatafromBin->sH = 1;
    tilingDatafromBin->sW = 2;
    tilingDatafromBin->pD = 1;
    tilingDatafromBin->pH = 2;
    tilingDatafromBin->pW = 3;
    tilingDatafromBin->dD = 1;
    tilingDatafromBin->dH = 1;
    tilingDatafromBin->dW = 1;
    tilingDatafromBin->blockFactor = 1;
    tilingDatafromBin->blockTail = 0;
    tilingDatafromBin->totalIdx = 1;
    tilingDatafromBin->coreNums = 1;
    tilingDatafromBin->maxCount = 60416;
    tilingDatafromBin->isSigOut = 0;

    ICPU_SET_TILING_KEY(611110);

    ICPU_RUN_KF(max_pool3d_with_argmax_v2, blockDim, input, out, indices, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(input);
    AscendC::GmFree(out);
    AscendC::GmFree(indices);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(max_pool3d_with_argmax_v2_test, test_case_for_bigkernel_split_d)
{
    AscendC:: SetKernelMode(KernelMode::AIV_MODE);
    uint32_t blockDim = 20;
    size_t inputByteSize = 1 * 1 * 3 * 6 * 62000 * sizeof(bfloat16_t);
    size_t outByteSize = 1 * 1 * 4 * 3 * 336 * sizeof(bfloat16_t);
    size_t indicesByteSize = 1 * 1 * 4 * 3 * 336 * sizeof(int32_t);
    size_t tilingDataSize = sizeof(MaxPool3DWithArgmaxV2BigKernelRegbaseTilingData);

    uint8_t* input = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* out = (uint8_t*)AscendC::GmAlloc(outByteSize);
    uint8_t* indices = (uint8_t*)AscendC::GmAlloc(indicesByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    char* path_ = get_current_dir_name();
    string path(path_);

    MaxPool3DWithArgmaxV2BigKernelRegbaseTilingData* tilingDatafromBin =
        reinterpret_cast<MaxPool3DWithArgmaxV2BigKernelRegbaseTilingData*>(tiling);
    tilingDatafromBin->dInDim = 3;
    tilingDatafromBin->hInDim = 6;
    tilingDatafromBin->wInDim = 62000;
    tilingDatafromBin->dOutDim = 4;
    tilingDatafromBin->hOutDim = 3;
    tilingDatafromBin->wOutDim = 336;
    tilingDatafromBin->kD = 2;
    tilingDatafromBin->kH = 4;
    tilingDatafromBin->kW = 61000;
    tilingDatafromBin->sD = 1;
    tilingDatafromBin->sH = 2;
    tilingDatafromBin->sW = 3;
    tilingDatafromBin->pD = 1;
    tilingDatafromBin->pH = 1;
    tilingDatafromBin->pW = 3;
    tilingDatafromBin->dD = 1;
    tilingDatafromBin->dH = 1;
    tilingDatafromBin->dW = 1;
    tilingDatafromBin->blockFactor = 63;
    tilingDatafromBin->blockTail = 1;
    tilingDatafromBin->totalIdx = 4032;
    tilingDatafromBin->coreNums = 64;
    tilingDatafromBin->maxCount = 60416;
    tilingDatafromBin->isSigOut = 0;

    ICPU_SET_TILING_KEY(611110);

    ICPU_RUN_KF(max_pool3d_with_argmax_v2, blockDim, input, out, indices, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(input);
    AscendC::GmFree(out);
    AscendC::GmFree(indices);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(max_pool3d_with_argmax_v2_test, test_case_for_bigkernel_split_dh)
{
    AscendC:: SetKernelMode(KernelMode::AIV_MODE);
    uint32_t blockDim = 20;
    size_t inputByteSize = 1 * 2 * 10 * 512 * 141 * sizeof(bfloat16_t);
    size_t outByteSize = 1 * 2 * 7 * 8 * 5 * sizeof(bfloat16_t);
    size_t indicesByteSize = 1 * 2 * 7 * 8 * 5 * sizeof(int32_t);
    size_t tilingDataSize = sizeof(MaxPool3DWithArgmaxV2BigKernelRegbaseTilingData);

    uint8_t* input = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* out = (uint8_t*)AscendC::GmAlloc(outByteSize);
    uint8_t* indices = (uint8_t*)AscendC::GmAlloc(indicesByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    char* path_ = get_current_dir_name();
    string path(path_);

    MaxPool3DWithArgmaxV2BigKernelRegbaseTilingData* tilingDatafromBin =
        reinterpret_cast<MaxPool3DWithArgmaxV2BigKernelRegbaseTilingData*>(tiling);
    tilingDatafromBin->dInDim = 10;
    tilingDatafromBin->hInDim = 512;
    tilingDatafromBin->wInDim = 141;
    tilingDatafromBin->dOutDim = 7;
    tilingDatafromBin->hOutDim = 8;
    tilingDatafromBin->wOutDim = 5;
    tilingDatafromBin->kD = 8;
    tilingDatafromBin->kH = 513;
    tilingDatafromBin->kW = 131;
    tilingDatafromBin->sD = 1;
    tilingDatafromBin->sH = 2;
    tilingDatafromBin->sW = 3;
    tilingDatafromBin->pD = 2;
    tilingDatafromBin->pH = 3;
    tilingDatafromBin->pW = 2;
    tilingDatafromBin->dD = 1;
    tilingDatafromBin->dH = 1;
    tilingDatafromBin->dW = 1;
    tilingDatafromBin->blockFactor = 8;
    tilingDatafromBin->blockTail = 48;
    tilingDatafromBin->totalIdx = 560;
    tilingDatafromBin->coreNums = 64;
    tilingDatafromBin->maxCount = 60416;
    tilingDatafromBin->isSigOut = 0;

    ICPU_SET_TILING_KEY(611110);

    ICPU_RUN_KF(max_pool3d_with_argmax_v2, blockDim, input, out, indices, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(input);
    AscendC::GmFree(out);
    AscendC::GmFree(indices);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(max_pool3d_with_argmax_v2_test, test_case_for_bigkernel_split_dhw)
{
    AscendC:: SetKernelMode(KernelMode::AIV_MODE);
    uint32_t blockDim = 20;
    size_t inputByteSize = 1 * 8 * 32 * 32 * 256 * sizeof(bfloat16_t);
    size_t outByteSize = 1 * 8 * 9 * 9 * 65 * sizeof(bfloat16_t);
    size_t indicesByteSize = 1 * 8 * 9 * 9 * 65 * sizeof(int32_t);
    size_t tilingDataSize = sizeof(MaxPool3DWithArgmaxV2BigKernelRegbaseTilingData);

    uint8_t* input = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* out = (uint8_t*)AscendC::GmAlloc(outByteSize);
    uint8_t* indices = (uint8_t*)AscendC::GmAlloc(indicesByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    char* path_ = get_current_dir_name();
    string path(path_);

    MaxPool3DWithArgmaxV2BigKernelRegbaseTilingData* tilingDatafromBin =
        reinterpret_cast<MaxPool3DWithArgmaxV2BigKernelRegbaseTilingData*>(tiling);
    tilingDatafromBin->dInDim = 32;
    tilingDatafromBin->hInDim = 32;
    tilingDatafromBin->wInDim = 256;
    tilingDatafromBin->dOutDim = 9;
    tilingDatafromBin->hOutDim = 9;
    tilingDatafromBin->wOutDim = 65;
    tilingDatafromBin->kD = 16;
    tilingDatafromBin->kH = 16;
    tilingDatafromBin->kW = 128;
    tilingDatafromBin->sD = 2;
    tilingDatafromBin->sH = 2;
    tilingDatafromBin->sW = 2;
    tilingDatafromBin->pD = 0;
    tilingDatafromBin->pH = 0;
    tilingDatafromBin->pW = 0;
    tilingDatafromBin->dD = 1;
    tilingDatafromBin->dH = 1;
    tilingDatafromBin->dW = 1;
    tilingDatafromBin->blockFactor = 658;
    tilingDatafromBin->blockTail = 8;
    tilingDatafromBin->totalIdx = 42120;
    tilingDatafromBin->coreNums = 64;
    tilingDatafromBin->maxCount = 60416;
    tilingDatafromBin->isSigOut = 0;

    ICPU_SET_TILING_KEY(611110);

    ICPU_RUN_KF(max_pool3d_with_argmax_v2, blockDim, input, out, indices, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(input);
    AscendC::GmFree(out);
    AscendC::GmFree(indices);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}