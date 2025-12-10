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
#include "swi_glu_quant_tiling_def.h"

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void swi_glu_quant(
    GM_ADDR x, GM_ADDR smooth_scales, GM_ADDR offsets, GM_ADDR group_index, GM_ADDR y, GM_ADDR scale, GM_ADDR workspace,
    GM_ADDR tiling);

class swi_glu_quant_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "swi_glu_quant_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "swi_glu_quant_test TearDown\n" << endl;
    }
};

TEST_F(swi_glu_quant_test, test_case_dynamic_bf16)
{
#undef DTYPE_Y
#define DTYPE_Y int8_t
    size_t inputByteSize = 256 * 256 * 2 * sizeof(int16_t);
    size_t smoothScalesByteSize = 16 * 256 * sizeof(int32_t);
    size_t groupIdxByteSize = 16 * sizeof(int32_t);

    size_t outputByteSize = 256 * 256 * sizeof(int8_t);
    size_t scaleByteSize = 256 * sizeof(int32_t);
    size_t tiling_data_size = sizeof(SwiGluQuantTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* smooth_scales = (uint8_t*)AscendC::GmAlloc(smoothScalesByteSize);
    uint8_t* group_index = (uint8_t*)AscendC::GmAlloc(groupIdxByteSize);

    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 4;
    char* path_ = get_current_dir_name();
    string path(path_);

    SwiGluQuantTilingData* tilingDatafromBin = reinterpret_cast<SwiGluQuantTilingData*>(tiling);

    tilingDatafromBin->groupLen = 16;
    tilingDatafromBin->rowLen = 256;
    tilingDatafromBin->colLen = 256;
    tilingDatafromBin->rowLenPerHeadCore = 7;
    tilingDatafromBin->rowLenPerTailCore = 6;
    tilingDatafromBin->basicRowLenHeadCore = 7;
    tilingDatafromBin->basicRowLenTailCore = 6;
    tilingDatafromBin->basicColLen = 256;
    tilingDatafromBin->headCoreNum = 16;
    tilingDatafromBin->realCoreNum = 40;
    tilingDatafromBin->activateLeft = 0;
    tilingDatafromBin->tilingKey = 206;
    ICPU_SET_TILING_KEY(206);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(
        swi_glu_quant, blockDim, x, smooth_scales, nullptr, group_index, y, scale, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(smooth_scales);
    AscendC::GmFree(group_index);
    AscendC::GmFree(y);
    AscendC::GmFree(scale);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(swi_glu_quant_test, test_case_bf16_static_per_channel)
{
#undef DTYPE_Y
#define DTYPE_Y int8_t
    size_t inputByteSize = 256 * 256 * 2 * sizeof(int16_t);
    size_t smoothScalesByteSize = 16 * 256 * sizeof(int32_t);
    size_t offsetsByteSize = 16 * 256 * sizeof(int32_t);
    size_t groupIdxByteSize = 16 * sizeof(int32_t);

    size_t outputByteSize = 256 * 256 * sizeof(int8_t);
    size_t scaleByteSize = 256 * sizeof(int32_t);
    size_t tiling_data_size = sizeof(SwiGluQuantTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* smooth_scales = (uint8_t*)AscendC::GmAlloc(smoothScalesByteSize);
    uint8_t* group_index = (uint8_t*)AscendC::GmAlloc(groupIdxByteSize);
    uint8_t* offsets = (uint8_t*)AscendC::GmAlloc(offsetsByteSize);

    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 4;
    SwiGluQuantTilingData* tilingDatafromBin = reinterpret_cast<SwiGluQuantTilingData*>(tiling);

    tilingDatafromBin->groupLen = 16;
    tilingDatafromBin->rowLen = 256;
    tilingDatafromBin->colLen = 256;
    tilingDatafromBin->rowLenPerHeadCore = 7;
    tilingDatafromBin->rowLenPerTailCore = 6;
    tilingDatafromBin->basicRowLenHeadCore = 7;
    tilingDatafromBin->basicRowLenTailCore = 6;
    tilingDatafromBin->basicColLen = 256;
    tilingDatafromBin->headCoreNum = 16;
    tilingDatafromBin->realCoreNum = 40;
    tilingDatafromBin->activateLeft = 0;
    tilingDatafromBin->tilingKey = 205;

    ICPU_SET_TILING_KEY(205);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(
        swi_glu_quant, blockDim, x, smooth_scales, nullptr, group_index, y, scale, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(smooth_scales);
    AscendC::GmFree(group_index);
    AscendC::GmFree(y);
    AscendC::GmFree(scale);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(swi_glu_quant_test, test_case_bf16_static_per_tensor)
{
#undef DTYPE_Y
#define DTYPE_Y int8_t
    size_t inputByteSize = 256 * 256 * 2 * sizeof(int16_t);
    size_t smoothScalesByteSize = 16 * sizeof(int32_t);
    size_t offsetsByteSize = 16 * sizeof(int32_t);
    size_t groupIdxByteSize = 16 * sizeof(int32_t);

    size_t outputByteSize = 256 * 256 * sizeof(int8_t);
    size_t scaleByteSize = 256 * sizeof(int32_t);
    size_t tiling_data_size = sizeof(SwiGluQuantTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* smooth_scales = (uint8_t*)AscendC::GmAlloc(smoothScalesByteSize);
    uint8_t* group_index = (uint8_t*)AscendC::GmAlloc(groupIdxByteSize);
    uint8_t* offsets = (uint8_t*)AscendC::GmAlloc(offsetsByteSize);

    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 4;
    SwiGluQuantTilingData* tilingDatafromBin = reinterpret_cast<SwiGluQuantTilingData*>(tiling);

    tilingDatafromBin->groupLen = 16;
    tilingDatafromBin->rowLen = 256;
    tilingDatafromBin->colLen = 256;
    tilingDatafromBin->rowLenPerHeadCore = 7;
    tilingDatafromBin->rowLenPerTailCore = 6;
    tilingDatafromBin->basicRowLenHeadCore = 7;
    tilingDatafromBin->basicRowLenTailCore = 6;
    tilingDatafromBin->basicColLen = 256;
    tilingDatafromBin->headCoreNum = 16;
    tilingDatafromBin->realCoreNum = 40;
    tilingDatafromBin->activateLeft = 0;
    tilingDatafromBin->tilingKey = 204;

    ICPU_SET_TILING_KEY(204);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(
        swi_glu_quant, blockDim, x, smooth_scales, nullptr, group_index, y, scale, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(smooth_scales);
    AscendC::GmFree(group_index);
    AscendC::GmFree(y);
    AscendC::GmFree(scale);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(swi_glu_quant_test, test_case_dynamic_fp16)
{
#undef DTYPE_Y
#define DTYPE_Y int8_t
    size_t inputByteSize = 256 * 256 * 2 * sizeof(int16_t);
    size_t smoothScalesByteSize = 16 * 256 * sizeof(int32_t);
    size_t groupIdxByteSize = 16 * sizeof(int32_t);

    size_t outputByteSize = 256 * 256 * sizeof(int8_t);
    size_t scaleByteSize = 256 * sizeof(int32_t);
    size_t tiling_data_size = sizeof(SwiGluQuantTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* smooth_scales = (uint8_t*)AscendC::GmAlloc(smoothScalesByteSize);
    uint8_t* group_index = (uint8_t*)AscendC::GmAlloc(groupIdxByteSize);

    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 4;
    char* path_ = get_current_dir_name();
    string path(path_);

    SwiGluQuantTilingData* tilingDatafromBin = reinterpret_cast<SwiGluQuantTilingData*>(tiling);

    tilingDatafromBin->groupLen = 16;
    tilingDatafromBin->rowLen = 256;
    tilingDatafromBin->colLen = 256;
    tilingDatafromBin->rowLenPerHeadCore = 7;
    tilingDatafromBin->rowLenPerTailCore = 6;
    tilingDatafromBin->basicRowLenHeadCore = 7;
    tilingDatafromBin->basicRowLenTailCore = 6;
    tilingDatafromBin->basicColLen = 256;
    tilingDatafromBin->headCoreNum = 16;
    tilingDatafromBin->realCoreNum = 40;
    tilingDatafromBin->activateLeft = 0;
    tilingDatafromBin->tilingKey = 106;
    ICPU_SET_TILING_KEY(106);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(
        swi_glu_quant, blockDim, x, smooth_scales, nullptr, group_index, y, scale, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(smooth_scales);
    AscendC::GmFree(group_index);
    AscendC::GmFree(y);
    AscendC::GmFree(scale);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}

TEST_F(swi_glu_quant_test, test_case_fp16_static_per_channel)
{
#undef DTYPE_Y
#define DTYPE_Y int8_t
    size_t inputByteSize = 256 * 256 * 2 * sizeof(int16_t);
    size_t smoothScalesByteSize = 16 * 256 * sizeof(int32_t);
    size_t offsetsByteSize = 16 * 256 * sizeof(int32_t);
    size_t groupIdxByteSize = 16 * sizeof(int32_t);

    size_t outputByteSize = 256 * 256 * sizeof(int8_t);
    size_t scaleByteSize = 256 * sizeof(int32_t);
    size_t tiling_data_size = sizeof(SwiGluQuantTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* smooth_scales = (uint8_t*)AscendC::GmAlloc(smoothScalesByteSize);
    uint8_t* group_index = (uint8_t*)AscendC::GmAlloc(groupIdxByteSize);
    uint8_t* offsets = (uint8_t*)AscendC::GmAlloc(offsetsByteSize);

    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 4;
    SwiGluQuantTilingData* tilingDatafromBin = reinterpret_cast<SwiGluQuantTilingData*>(tiling);

    tilingDatafromBin->groupLen = 16;
    tilingDatafromBin->rowLen = 256;
    tilingDatafromBin->colLen = 256;
    tilingDatafromBin->rowLenPerHeadCore = 7;
    tilingDatafromBin->rowLenPerTailCore = 6;
    tilingDatafromBin->basicRowLenHeadCore = 7;
    tilingDatafromBin->basicRowLenTailCore = 6;
    tilingDatafromBin->basicColLen = 256;
    tilingDatafromBin->headCoreNum = 16;
    tilingDatafromBin->realCoreNum = 40;
    tilingDatafromBin->activateLeft = 0;
    tilingDatafromBin->tilingKey = 105;

    ICPU_SET_TILING_KEY(105);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(
        swi_glu_quant, blockDim, x, smooth_scales, nullptr, group_index, y, scale, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(smooth_scales);
    AscendC::GmFree(group_index);
    AscendC::GmFree(y);
    AscendC::GmFree(scale);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(swi_glu_quant_test, test_case_fp16_static_per_tensor)
{
#undef DTYPE_Y
#define DTYPE_Y int8_t
    size_t inputByteSize = 256 * 256 * 2 * sizeof(int16_t);
    size_t smoothScalesByteSize = 16 * sizeof(int32_t);
    size_t offsetsByteSize = 16 * sizeof(int32_t);
    size_t groupIdxByteSize = 16 * sizeof(int32_t);

    size_t outputByteSize = 256 * 256 * sizeof(int8_t);
    size_t scaleByteSize = 256 * sizeof(int32_t);
    size_t tiling_data_size = sizeof(SwiGluQuantTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* smooth_scales = (uint8_t*)AscendC::GmAlloc(smoothScalesByteSize);
    uint8_t* group_index = (uint8_t*)AscendC::GmAlloc(groupIdxByteSize);
    uint8_t* offsets = (uint8_t*)AscendC::GmAlloc(offsetsByteSize);

    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 4;
    SwiGluQuantTilingData* tilingDatafromBin = reinterpret_cast<SwiGluQuantTilingData*>(tiling);

    tilingDatafromBin->groupLen = 16;
    tilingDatafromBin->rowLen = 256;
    tilingDatafromBin->colLen = 256;
    tilingDatafromBin->rowLenPerHeadCore = 7;
    tilingDatafromBin->rowLenPerTailCore = 6;
    tilingDatafromBin->basicRowLenHeadCore = 7;
    tilingDatafromBin->basicRowLenTailCore = 6;
    tilingDatafromBin->basicColLen = 256;
    tilingDatafromBin->headCoreNum = 16;
    tilingDatafromBin->realCoreNum = 40;
    tilingDatafromBin->activateLeft = 0;
    tilingDatafromBin->tilingKey = 104;

    ICPU_SET_TILING_KEY(104);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(
        swi_glu_quant, blockDim, x, smooth_scales, nullptr, group_index, y, scale, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(smooth_scales);
    AscendC::GmFree(group_index);
    AscendC::GmFree(y);
    AscendC::GmFree(scale);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(swi_glu_quant_test, test_case_dynamic_fp32)
{
#undef DTYPE_Y
#define DTYPE_Y int8_t
    size_t inputByteSize = 256 * 256 * 2 * sizeof(int32_t);
    size_t smoothScalesByteSize = 16 * 256 * sizeof(int32_t);
    size_t groupIdxByteSize = 16 * sizeof(int32_t);

    size_t outputByteSize = 256 * 256 * sizeof(int8_t);
    size_t scaleByteSize = 256 * sizeof(int32_t);
    size_t tiling_data_size = sizeof(SwiGluQuantTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* smooth_scales = (uint8_t*)AscendC::GmAlloc(smoothScalesByteSize);
    uint8_t* group_index = (uint8_t*)AscendC::GmAlloc(groupIdxByteSize);

    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 4;

    SwiGluQuantTilingData* tilingDatafromBin = reinterpret_cast<SwiGluQuantTilingData*>(tiling);

    tilingDatafromBin->groupLen = 16;
    tilingDatafromBin->rowLen = 256;
    tilingDatafromBin->colLen = 256;
    tilingDatafromBin->rowLenPerHeadCore = 7;
    tilingDatafromBin->rowLenPerTailCore = 6;
    tilingDatafromBin->basicRowLenHeadCore = 7;
    tilingDatafromBin->basicRowLenTailCore = 6;
    tilingDatafromBin->basicColLen = 256;
    tilingDatafromBin->headCoreNum = 16;
    tilingDatafromBin->realCoreNum = 40;
    tilingDatafromBin->activateLeft = 0;
    tilingDatafromBin->tilingKey = 306;
    ICPU_SET_TILING_KEY(306);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(
        swi_glu_quant, blockDim, x, smooth_scales, nullptr, group_index, y, scale, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(smooth_scales);
    AscendC::GmFree(group_index);
    AscendC::GmFree(y);
    AscendC::GmFree(scale);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(swi_glu_quant_test, test_case_fp32_static_per_channel)
{
#undef DTYPE_Y
#define DTYPE_Y int8_t
    size_t inputByteSize = 256 * 256 * 2 * sizeof(int32_t);
    size_t smoothScalesByteSize = 16 * 256 * sizeof(int32_t);
    size_t offsetsByteSize = 16 * 256 * sizeof(int32_t);
    size_t groupIdxByteSize = 16 * sizeof(int32_t);

    size_t outputByteSize = 256 * 256 * sizeof(int8_t);
    size_t scaleByteSize = 256 * sizeof(int32_t);
    size_t tiling_data_size = sizeof(SwiGluQuantTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* smooth_scales = (uint8_t*)AscendC::GmAlloc(smoothScalesByteSize);
    uint8_t* group_index = (uint8_t*)AscendC::GmAlloc(groupIdxByteSize);
    uint8_t* offsets = (uint8_t*)AscendC::GmAlloc(offsetsByteSize);

    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 4;
    SwiGluQuantTilingData* tilingDatafromBin = reinterpret_cast<SwiGluQuantTilingData*>(tiling);

    tilingDatafromBin->groupLen = 16;
    tilingDatafromBin->rowLen = 256;
    tilingDatafromBin->colLen = 256;
    tilingDatafromBin->rowLenPerHeadCore = 7;
    tilingDatafromBin->rowLenPerTailCore = 6;
    tilingDatafromBin->basicRowLenHeadCore = 7;
    tilingDatafromBin->basicRowLenTailCore = 6;
    tilingDatafromBin->basicColLen = 256;
    tilingDatafromBin->headCoreNum = 16;
    tilingDatafromBin->realCoreNum = 40;
    tilingDatafromBin->activateLeft = 0;
    tilingDatafromBin->tilingKey = 305;

    ICPU_SET_TILING_KEY(305);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(
        swi_glu_quant, blockDim, x, smooth_scales, nullptr, group_index, y, scale, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(smooth_scales);
    AscendC::GmFree(group_index);
    AscendC::GmFree(y);
    AscendC::GmFree(scale);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(swi_glu_quant_test, test_case_fp32_static_per_tensor)
{
#undef DTYPE_Y
#define DTYPE_Y int8_t
    size_t inputByteSize = 256 * 256 * 2 * sizeof(int32_t);
    size_t smoothScalesByteSize = 16 * sizeof(int32_t);
    size_t offsetsByteSize = 16 * sizeof(int32_t);
    size_t groupIdxByteSize = 16 * sizeof(int32_t);

    size_t outputByteSize = 256 * 256 * sizeof(int8_t);
    size_t scaleByteSize = 256 * sizeof(int32_t);
    size_t tiling_data_size = sizeof(SwiGluQuantTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* smooth_scales = (uint8_t*)AscendC::GmAlloc(smoothScalesByteSize);
    uint8_t* group_index = (uint8_t*)AscendC::GmAlloc(groupIdxByteSize);
    uint8_t* offsets = (uint8_t*)AscendC::GmAlloc(offsetsByteSize);

    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 2 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 4;
    SwiGluQuantTilingData* tilingDatafromBin = reinterpret_cast<SwiGluQuantTilingData*>(tiling);

    tilingDatafromBin->groupLen = 16;
    tilingDatafromBin->rowLen = 256;
    tilingDatafromBin->colLen = 256;
    tilingDatafromBin->rowLenPerHeadCore = 7;
    tilingDatafromBin->rowLenPerTailCore = 6;
    tilingDatafromBin->basicRowLenHeadCore = 7;
    tilingDatafromBin->basicRowLenTailCore = 6;
    tilingDatafromBin->basicColLen = 256;
    tilingDatafromBin->headCoreNum = 16;
    tilingDatafromBin->realCoreNum = 40;
    tilingDatafromBin->activateLeft = 0;
    tilingDatafromBin->tilingKey = 304;

    ICPU_SET_TILING_KEY(304);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(
        swi_glu_quant, blockDim, x, smooth_scales, nullptr, group_index, y, scale, workspace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(smooth_scales);
    AscendC::GmFree(group_index);
    AscendC::GmFree(y);
    AscendC::GmFree(scale);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}