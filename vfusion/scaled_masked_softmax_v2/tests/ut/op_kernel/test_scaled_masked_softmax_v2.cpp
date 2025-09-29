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
#include "scaled_masked_softmax_v2_tiling_def.h"
#include "data_utils.h"

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void scaled_masked_softmax_v2(
    GM_ADDR x, GM_ADDR mask, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

class scaled_masked_softmax_v2_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "scaled_masked_softmax_v2_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "scaled_masked_softmax_v2_test TearDown\n" << endl;
    }
};

TEST_F(scaled_masked_softmax_v2_test, test_case_half_align)
{
    uint32_t N = 1;
    uint32_t C = 32;
    uint32_t H = 32;
    uint32_t W = 128;

    uint32_t maskN = 1;
    uint32_t maskC = 32;

    size_t x_size = N * C * H * W * sizeof(half);
    size_t mask_size = maskN * maskC * H * W * sizeof(bool);
    size_t y_size = N * C * H * W * sizeof(half);

    size_t tiling_data_size = sizeof(ScaledMaskedSoftmaxV2TilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* mask = (uint8_t*)AscendC::GmAlloc(mask_size);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(y_size);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(4096 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 48;

    system("cp -rf ../../../../vfusion/scaled_masked_softmax_v2/tests/ut/op_kernel/scaled_masked_softmax_v2_data/ ./");
    system("chmod -R 755 ./scaled_masked_softmax_v2_data/");
    system("cd ./scaled_masked_softmax_v2_data/ && rm -rf ./*bin");
    system("cd ./scaled_masked_softmax_v2_data/ && python3 gen_data.py 1 32 32 128 1 32 half");

    char* path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/scaled_masked_softmax_v2_data/input_x.bin", x_size, x, x_size);
    ReadFile(path + "/scaled_masked_softmax_v2_data/input_mask.bin", mask_size, mask, mask_size);

    ScaledMaskedSoftmaxV2TilingData* tilingDatafromBin = reinterpret_cast<ScaledMaskedSoftmaxV2TilingData*>(tiling);
    tilingDatafromBin->coreNum = blockDim;
    tilingDatafromBin->batch = N;
    tilingDatafromBin->channel = C;
    tilingDatafromBin->height = H;
    tilingDatafromBin->width = W;
    tilingDatafromBin->maskBatch = maskN;
    tilingDatafromBin->maskChannel = maskC;
    tilingDatafromBin->maskHeight = H;
    tilingDatafromBin->maskWidth = W;
    tilingDatafromBin->scale = 1.0;
    tilingDatafromBin->maskMode = 0;
    tilingDatafromBin->paddingNum = 0;
    tilingDatafromBin->padLineNum = 128;
    tilingDatafromBin->alignedMaskPadding = 0;
    tilingDatafromBin->alignedMaskWidth = 128;
    tilingDatafromBin->nStep = 1;
    tilingDatafromBin->cStep = 1;
    tilingDatafromBin->headCoreNum = 16;
    tilingDatafromBin->lineHeadCore = 22;
    tilingDatafromBin->iterHeadCore = 1;
    tilingDatafromBin->lineHeadIter = 65;
    tilingDatafromBin->lineLastHeadIter = 22;
    tilingDatafromBin->lineTailCore = 21;
    tilingDatafromBin->iterTailCore = 1;
    tilingDatafromBin->lineTailIter = 65;
    tilingDatafromBin->lineLastTailIter = 21;

    tilingDatafromBin->softmaxTilingData.srcM = 65;
    tilingDatafromBin->softmaxTilingData.srcK = 128;
    tilingDatafromBin->softmaxTilingData.srcSize = 8320;
    tilingDatafromBin->softmaxTilingData.outMaxM = 65;
    tilingDatafromBin->softmaxTilingData.outMaxK = 16;
    tilingDatafromBin->softmaxTilingData.outMaxSize = 1040;
    tilingDatafromBin->softmaxTilingData.splitM = 65;
    tilingDatafromBin->softmaxTilingData.splitK = 128;
    tilingDatafromBin->softmaxTilingData.splitSize = 8320;
    tilingDatafromBin->softmaxTilingData.reduceM = 65;
    tilingDatafromBin->softmaxTilingData.reduceK = 16;
    tilingDatafromBin->softmaxTilingData.reduceSize = 1040;
    tilingDatafromBin->softmaxTilingData.rangeM = 1;
    tilingDatafromBin->softmaxTilingData.tailM = 0;
    tilingDatafromBin->softmaxTilingData.tailSplitSize = 0;
    tilingDatafromBin->softmaxTilingData.tailReduceSize = 0;

    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(scaled_masked_softmax_v2, blockDim, x, mask, y, workspace, (uint8_t*)(tilingDatafromBin));
    AscendC::GmFree(x);
    AscendC::GmFree(mask);
    AscendC::GmFree(y);

    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);

    free(path_);
}

TEST_F(scaled_masked_softmax_v2_test, test_case_half_align_broadcastC)
{
    uint32_t N = 1;
    uint32_t C = 32;
    uint32_t H = 32;
    uint32_t W = 128;

    uint32_t maskN = 1;
    uint32_t maskC = 1;

    size_t x_size = N * C * H * W * sizeof(half);
    size_t mask_size = maskN * maskC * H * W * sizeof(bool);
    size_t y_size = N * C * H * W * sizeof(half);

    size_t tiling_data_size = sizeof(ScaledMaskedSoftmaxV2TilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* mask = (uint8_t*)AscendC::GmAlloc(mask_size);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(y_size);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(4096 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 48;

    system("cp -rf ../../../../vfusion/scaled_masked_softmax_v2/tests/ut/op_kernel/scaled_masked_softmax_v2_data/ ./");
    system("chmod -R 755 ./scaled_masked_softmax_v2_data/");
    system("cd ./scaled_masked_softmax_v2_data/ && rm -rf ./*bin");
    system("cd ./scaled_masked_softmax_v2_data/ && python3 gen_data.py 1 32 32 128 1 1 half");

    char* path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/scaled_masked_softmax_v2_data/input_x.bin", x_size, x, x_size);
    ReadFile(path + "/scaled_masked_softmax_v2_data/input_mask.bin", mask_size, mask, mask_size);

    ScaledMaskedSoftmaxV2TilingData* tilingDatafromBin = reinterpret_cast<ScaledMaskedSoftmaxV2TilingData*>(tiling);
    tilingDatafromBin->coreNum = blockDim;
    tilingDatafromBin->batch = N;
    tilingDatafromBin->channel = C;
    tilingDatafromBin->height = H;
    tilingDatafromBin->width = W;
    tilingDatafromBin->maskBatch = maskN;
    tilingDatafromBin->maskChannel = maskC;
    tilingDatafromBin->maskHeight = H;
    tilingDatafromBin->maskWidth = W;
    tilingDatafromBin->scale = 1.0;
    tilingDatafromBin->maskMode = 0;
    tilingDatafromBin->paddingNum = 0;
    tilingDatafromBin->padLineNum = 128;
    tilingDatafromBin->alignedMaskPadding = 0;
    tilingDatafromBin->alignedMaskWidth = 128;
    tilingDatafromBin->nStep = 1;
    tilingDatafromBin->cStep = 0;
    tilingDatafromBin->headCoreNum = 16;
    tilingDatafromBin->lineHeadCore = 22;
    tilingDatafromBin->iterHeadCore = 1;
    tilingDatafromBin->lineHeadIter = 65;
    tilingDatafromBin->lineLastHeadIter = 22;
    tilingDatafromBin->lineTailCore = 21;
    tilingDatafromBin->iterTailCore = 1;
    tilingDatafromBin->lineTailIter = 65;
    tilingDatafromBin->lineLastTailIter = 21;

    tilingDatafromBin->softmaxTilingData.srcM = 65;
    tilingDatafromBin->softmaxTilingData.srcK = 128;
    tilingDatafromBin->softmaxTilingData.srcSize = 8320;
    tilingDatafromBin->softmaxTilingData.outMaxM = 65;
    tilingDatafromBin->softmaxTilingData.outMaxK = 16;
    tilingDatafromBin->softmaxTilingData.outMaxSize = 1040;
    tilingDatafromBin->softmaxTilingData.splitM = 65;
    tilingDatafromBin->softmaxTilingData.splitK = 128;
    tilingDatafromBin->softmaxTilingData.splitSize = 8320;
    tilingDatafromBin->softmaxTilingData.reduceM = 65;
    tilingDatafromBin->softmaxTilingData.reduceK = 16;
    tilingDatafromBin->softmaxTilingData.reduceSize = 1040;
    tilingDatafromBin->softmaxTilingData.rangeM = 1;
    tilingDatafromBin->softmaxTilingData.tailM = 0;
    tilingDatafromBin->softmaxTilingData.tailSplitSize = 0;
    tilingDatafromBin->softmaxTilingData.tailReduceSize = 0;

    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(scaled_masked_softmax_v2, blockDim, x, mask, y, workspace, (uint8_t*)(tilingDatafromBin));
    AscendC::GmFree(x);
    AscendC::GmFree(mask);
    AscendC::GmFree(y);

    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);

    free(path_);
}

TEST_F(scaled_masked_softmax_v2_test, test_case_half_align_broadcastN)
{
    uint32_t N = 32;
    uint32_t C = 1;
    uint32_t H = 32;
    uint32_t W = 128;

    uint32_t maskN = 1;
    uint32_t maskC = 1;

    size_t x_size = N * C * H * W * sizeof(half);
    size_t mask_size = maskN * maskC * H * W * sizeof(bool);
    size_t y_size = N * C * H * W * sizeof(half);

    size_t tiling_data_size = sizeof(ScaledMaskedSoftmaxV2TilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* mask = (uint8_t*)AscendC::GmAlloc(mask_size);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(y_size);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(4096 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 48;

    system("cp -rf ../../../../vfusion/scaled_masked_softmax_v2/tests/ut/op_kernel/scaled_masked_softmax_v2_data/ ./");
    system("chmod -R 755 ./scaled_masked_softmax_v2_data/");
    system("cd ./scaled_masked_softmax_v2_data/ && rm -rf ./*bin");
    system("cd ./scaled_masked_softmax_v2_data/ && python3 gen_data.py 32 1 32 128 1 1 half");

    char* path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/scaled_masked_softmax_v2_data/input_x.bin", x_size, x, x_size);
    ReadFile(path + "/scaled_masked_softmax_v2_data/input_mask.bin", mask_size, mask, mask_size);

    ScaledMaskedSoftmaxV2TilingData* tilingDatafromBin = reinterpret_cast<ScaledMaskedSoftmaxV2TilingData*>(tiling);
    tilingDatafromBin->coreNum = blockDim;
    tilingDatafromBin->batch = N;
    tilingDatafromBin->channel = C;
    tilingDatafromBin->height = H;
    tilingDatafromBin->width = W;
    tilingDatafromBin->maskBatch = maskN;
    tilingDatafromBin->maskChannel = maskC;
    tilingDatafromBin->maskHeight = H;
    tilingDatafromBin->maskWidth = W;
    tilingDatafromBin->scale = 1.0;
    tilingDatafromBin->maskMode = 0;
    tilingDatafromBin->paddingNum = 0;
    tilingDatafromBin->padLineNum = 128;
    tilingDatafromBin->alignedMaskPadding = 0;
    tilingDatafromBin->alignedMaskWidth = 128;
    tilingDatafromBin->nStep = 0;
    tilingDatafromBin->cStep = 1;
    tilingDatafromBin->headCoreNum = 16;
    tilingDatafromBin->lineHeadCore = 22;
    tilingDatafromBin->iterHeadCore = 1;
    tilingDatafromBin->lineHeadIter = 65;
    tilingDatafromBin->lineLastHeadIter = 22;
    tilingDatafromBin->lineTailCore = 21;
    tilingDatafromBin->iterTailCore = 1;
    tilingDatafromBin->lineTailIter = 65;
    tilingDatafromBin->lineLastTailIter = 21;

    tilingDatafromBin->softmaxTilingData.srcM = 65;
    tilingDatafromBin->softmaxTilingData.srcK = 128;
    tilingDatafromBin->softmaxTilingData.srcSize = 8320;
    tilingDatafromBin->softmaxTilingData.outMaxM = 65;
    tilingDatafromBin->softmaxTilingData.outMaxK = 16;
    tilingDatafromBin->softmaxTilingData.outMaxSize = 1040;
    tilingDatafromBin->softmaxTilingData.splitM = 65;
    tilingDatafromBin->softmaxTilingData.splitK = 128;
    tilingDatafromBin->softmaxTilingData.splitSize = 8320;
    tilingDatafromBin->softmaxTilingData.reduceM = 65;
    tilingDatafromBin->softmaxTilingData.reduceK = 16;
    tilingDatafromBin->softmaxTilingData.reduceSize = 1040;
    tilingDatafromBin->softmaxTilingData.rangeM = 1;
    tilingDatafromBin->softmaxTilingData.tailM = 0;
    tilingDatafromBin->softmaxTilingData.tailSplitSize = 0;
    tilingDatafromBin->softmaxTilingData.tailReduceSize = 0;

    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(scaled_masked_softmax_v2, blockDim, x, mask, y, workspace, (uint8_t*)(tilingDatafromBin));
    AscendC::GmFree(x);
    AscendC::GmFree(mask);
    AscendC::GmFree(y);

    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);

    free(path_);
}

TEST_F(scaled_masked_softmax_v2_test, test_case_half_align_broadcastNC)
{
    uint32_t N = 4;
    uint32_t C = 4;
    uint32_t H = 32;
    uint32_t W = 128;

    uint32_t maskN = 1;
    uint32_t maskC = 1;

    size_t x_size = N * C * H * W * sizeof(half);
    size_t mask_size = maskN * maskC * H * W * sizeof(bool);
    size_t y_size = N * C * H * W * sizeof(half);

    size_t tiling_data_size = sizeof(ScaledMaskedSoftmaxV2TilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* mask = (uint8_t*)AscendC::GmAlloc(mask_size);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(y_size);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(4096 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 48;

    system("cp -rf ../../../../vfusion/scaled_masked_softmax_v2/tests/ut/op_kernel/scaled_masked_softmax_v2_data/ ./");
    system("chmod -R 755 ./scaled_masked_softmax_v2_data/");
    system("cd ./scaled_masked_softmax_v2_data/ && rm -rf ./*bin");
    system("cd ./scaled_masked_softmax_v2_data/ && python3 gen_data.py 4 4 32 128 1 1 half");

    char* path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/scaled_masked_softmax_v2_data/input_x.bin", x_size, x, x_size);
    ReadFile(path + "/scaled_masked_softmax_v2_data/input_mask.bin", mask_size, mask, mask_size);

    ScaledMaskedSoftmaxV2TilingData* tilingDatafromBin = reinterpret_cast<ScaledMaskedSoftmaxV2TilingData*>(tiling);
    tilingDatafromBin->coreNum = blockDim;
    tilingDatafromBin->batch = N;
    tilingDatafromBin->channel = C;
    tilingDatafromBin->height = H;
    tilingDatafromBin->width = W;
    tilingDatafromBin->maskBatch = maskN;
    tilingDatafromBin->maskChannel = maskC;
    tilingDatafromBin->maskHeight = H;
    tilingDatafromBin->maskWidth = W;
    tilingDatafromBin->scale = 1.0;
    tilingDatafromBin->maskMode = 0;
    tilingDatafromBin->paddingNum = 0;
    tilingDatafromBin->padLineNum = 128;
    tilingDatafromBin->alignedMaskPadding = 0;
    tilingDatafromBin->alignedMaskWidth = 128;
    tilingDatafromBin->nStep = 0;
    tilingDatafromBin->cStep = 0;
    tilingDatafromBin->headCoreNum = 32;
    tilingDatafromBin->lineHeadCore = 11;
    tilingDatafromBin->iterHeadCore = 1;
    tilingDatafromBin->lineHeadIter = 65;
    tilingDatafromBin->lineLastHeadIter = 11;
    tilingDatafromBin->lineTailCore = 10;
    tilingDatafromBin->iterTailCore = 1;
    tilingDatafromBin->lineTailIter = 65;
    tilingDatafromBin->lineLastTailIter = 10;

    tilingDatafromBin->softmaxTilingData.srcM = 65;
    tilingDatafromBin->softmaxTilingData.srcK = 128;
    tilingDatafromBin->softmaxTilingData.srcSize = 8320;
    tilingDatafromBin->softmaxTilingData.outMaxM = 65;
    tilingDatafromBin->softmaxTilingData.outMaxK = 16;
    tilingDatafromBin->softmaxTilingData.outMaxSize = 1040;
    tilingDatafromBin->softmaxTilingData.splitM = 65;
    tilingDatafromBin->softmaxTilingData.splitK = 128;
    tilingDatafromBin->softmaxTilingData.splitSize = 8320;
    tilingDatafromBin->softmaxTilingData.reduceM = 65;
    tilingDatafromBin->softmaxTilingData.reduceK = 16;
    tilingDatafromBin->softmaxTilingData.reduceSize = 1040;
    tilingDatafromBin->softmaxTilingData.rangeM = 1;
    tilingDatafromBin->softmaxTilingData.tailM = 0;
    tilingDatafromBin->softmaxTilingData.tailSplitSize = 0;
    tilingDatafromBin->softmaxTilingData.tailReduceSize = 0;

    ICPU_SET_TILING_KEY(1);
    ICPU_RUN_KF(scaled_masked_softmax_v2, blockDim, x, mask, y, workspace, (uint8_t*)(tilingDatafromBin));
    AscendC::GmFree(x);
    AscendC::GmFree(mask);
    AscendC::GmFree(y);

    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);

    free(path_);
}

TEST_F(scaled_masked_softmax_v2_test, test_case_float_align)
{
    uint32_t N = 1;
    uint32_t C = 32;
    uint32_t H = 32;
    uint32_t W = 512;

    uint32_t maskN = 1;
    uint32_t maskC = 32;

    size_t x_size = N * C * H * W * sizeof(float);
    size_t mask_size = maskN * maskC * H * W * sizeof(bool);
    size_t y_size = N * C * H * W * sizeof(float);

    size_t tiling_data_size = sizeof(ScaledMaskedSoftmaxV2TilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(x_size);
    uint8_t* mask = (uint8_t*)AscendC::GmAlloc(mask_size);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(y_size);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(4096 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 48;

    system("cp -rf ../../../../vfusion/scaled_masked_softmax_v2/tests/ut/op_kernel/scaled_masked_softmax_v2_data/ ./");
    system("chmod -R 755 ./scaled_masked_softmax_v2_data/");
    system("cd ./scaled_masked_softmax_v2_data/ && rm -rf ./*bin");
    system("cd ./scaled_masked_softmax_v2_data/ && python3 gen_data.py 1 32 32 512 1 32 float32");

    char* path_ = get_current_dir_name();
    string path(path_);
    ReadFile(path + "/scaled_masked_softmax_v2_data/input_x.bin", x_size, x, x_size);
    ReadFile(path + "/scaled_masked_softmax_v2_data/input_mask.bin", mask_size, mask, mask_size);

    ScaledMaskedSoftmaxV2TilingData* tilingDatafromBin = reinterpret_cast<ScaledMaskedSoftmaxV2TilingData*>(tiling);
    tilingDatafromBin->coreNum = blockDim;
    tilingDatafromBin->batch = N;
    tilingDatafromBin->channel = C;
    tilingDatafromBin->height = H;
    tilingDatafromBin->width = W;
    tilingDatafromBin->maskBatch = maskN;
    tilingDatafromBin->maskChannel = maskC;
    tilingDatafromBin->maskHeight = H;
    tilingDatafromBin->maskWidth = W;
    tilingDatafromBin->scale = 1.0;
    tilingDatafromBin->maskMode = 0;
    tilingDatafromBin->paddingNum = 0;
    tilingDatafromBin->padLineNum = 128;
    tilingDatafromBin->alignedMaskPadding = 0;
    tilingDatafromBin->alignedMaskWidth = 128;
    tilingDatafromBin->nStep = 1;
    tilingDatafromBin->cStep = 1;
    tilingDatafromBin->headCoreNum = 16;
    tilingDatafromBin->lineHeadCore = 22;
    tilingDatafromBin->iterHeadCore = 1;
    tilingDatafromBin->lineHeadIter = 65;
    tilingDatafromBin->lineLastHeadIter = 22;
    tilingDatafromBin->lineTailCore = 21;
    tilingDatafromBin->iterTailCore = 1;
    tilingDatafromBin->lineTailIter = 65;
    tilingDatafromBin->lineLastTailIter = 21;

    tilingDatafromBin->softmaxTilingData.srcM = 65;
    tilingDatafromBin->softmaxTilingData.srcK = 128;
    tilingDatafromBin->softmaxTilingData.srcSize = 8320;
    tilingDatafromBin->softmaxTilingData.outMaxM = 65;
    tilingDatafromBin->softmaxTilingData.outMaxK = 16;
    tilingDatafromBin->softmaxTilingData.outMaxSize = 1040;
    tilingDatafromBin->softmaxTilingData.splitM = 65;
    tilingDatafromBin->softmaxTilingData.splitK = 128;
    tilingDatafromBin->softmaxTilingData.splitSize = 8320;
    tilingDatafromBin->softmaxTilingData.reduceM = 65;
    tilingDatafromBin->softmaxTilingData.reduceK = 16;
    tilingDatafromBin->softmaxTilingData.reduceSize = 1040;
    tilingDatafromBin->softmaxTilingData.rangeM = 1;
    tilingDatafromBin->softmaxTilingData.tailM = 0;
    tilingDatafromBin->softmaxTilingData.tailSplitSize = 0;
    tilingDatafromBin->softmaxTilingData.tailReduceSize = 0;

    ICPU_SET_TILING_KEY(0);
    ICPU_RUN_KF(scaled_masked_softmax_v2, blockDim, x, mask, y, workspace, (uint8_t*)(tilingDatafromBin));
    AscendC::GmFree(x);
    AscendC::GmFree(mask);
    AscendC::GmFree(y);

    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);

    free(path_);
}
