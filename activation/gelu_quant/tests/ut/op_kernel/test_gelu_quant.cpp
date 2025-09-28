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
#include "test_gelu_quant.h"
#include "../data_utils.h"

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void gelu_quant(
    GM_ADDR x, GM_ADDR input_scale, GM_ADDR input_offset, GM_ADDR y, GM_ADDR out_scale, GM_ADDR workspace,
    GM_ADDR tiling_data);

class gelu_quant_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "gelu_quant_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "gelu_quant_test TearDown\n" << endl;
    }
};

TEST_F(gelu_quant_test, test_case_gelu_quant_0)
{
    size_t inputByteSize = 1 * 24 * 6912 * sizeof(uint32_t);
    size_t scaleInputByteSzie = 6912 * sizeof(uint32_t);
    size_t offsetInputByteSzie = 6912 * sizeof(uint32_t);
    size_t outputByteSize = 1 * 24 * 6912 * sizeof(int8_t);
    size_t scaleOutByteSize = 6912 * sizeof(uint32_t);
    size_t tiling_data_size = sizeof(GeluQuantTilingData);
    uint32_t blockDim = 24;

    uint8_t* x_input = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* scale_input = (uint8_t*)AscendC::GmAlloc(scaleInputByteSzie);
    uint8_t* offset_input = (uint8_t*)AscendC::GmAlloc(offsetInputByteSzie);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* scale_out = (uint8_t*)AscendC::GmAlloc(scaleOutByteSize);

    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    GeluQuantTilingData* tilingDatafromBin = reinterpret_cast<GeluQuantTilingData*>(tiling);

    tilingDatafromBin->usedCoreNum = 24;
    tilingDatafromBin->normalCoreProcessNum = 1;
    tilingDatafromBin->tailCoreProcessNum = 1;
    tilingDatafromBin->endAxisLen = 6912;
    tilingDatafromBin->endAxisLenAligned = 6912;
    tilingDatafromBin->coexistentNodeNum = 6;
    tilingDatafromBin->coexistentNodeElementNum = 7000;
    tilingDatafromBin->rowInner = 1;
    tilingDatafromBin->rowTail = 1;
    tilingDatafromBin->rowOuter = 1;
    tilingDatafromBin->colInner = 1;
    tilingDatafromBin->colTail = 1;
    tilingDatafromBin->colOuter = 1;
    tilingDatafromBin->quantMode = 1;
    tilingDatafromBin->approximate = 1;
    tilingDatafromBin->inputScaleType = 2;
    tilingDatafromBin->inputOffsetType = 0;
    tilingDatafromBin->tilingKey = 0;

    ICPU_SET_TILING_KEY(1001);
    ICPU_RUN_KF(
        gelu_quant, blockDim, x_input, scale_input, offset_input, y, scale_out, workSpace,
        (uint8_t*)(tilingDatafromBin));

    ICPU_SET_TILING_KEY(1002);
    ICPU_RUN_KF(
        gelu_quant, blockDim, x_input, scale_input, offset_input, y, scale_out, workSpace,
        (uint8_t*)(tilingDatafromBin));

    ICPU_SET_TILING_KEY(1003);
    ICPU_RUN_KF(
        gelu_quant, blockDim, x_input, scale_input, offset_input, y, scale_out, workSpace,
        (uint8_t*)(tilingDatafromBin));

    ICPU_SET_TILING_KEY(1004);
    ICPU_RUN_KF(
        gelu_quant, blockDim, x_input, scale_input, offset_input, y, scale_out, workSpace,
        (uint8_t*)(tilingDatafromBin));

    ICPU_SET_TILING_KEY(1005);
    ICPU_RUN_KF(
        gelu_quant, blockDim, x_input, scale_input, offset_input, y, scale_out, workSpace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x_input);
    AscendC::GmFree(scale_input);
    AscendC::GmFree(offset_input);
    AscendC::GmFree(y);
    AscendC::GmFree(scale_out);
    AscendC::GmFree(workSpace);
    AscendC::GmFree(tiling);
}
TEST_F(gelu_quant_test, test_case_gelu_quant_1)
{
    size_t inputByteSize = 1 * 24 * 6912 * sizeof(uint32_t);
    size_t scaleInputByteSzie = 6912 * sizeof(uint32_t);
    size_t offsetInputByteSzie = 6912 * sizeof(uint32_t);
    size_t outputByteSize = 1 * 24 * 6912 * sizeof(int8_t);
    size_t scaleOutByteSize = 6912 * sizeof(uint32_t);
    size_t tiling_data_size = sizeof(GeluQuantTilingData);
    uint32_t blockDim = 24;

    uint8_t* x_input = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* scale_input = (uint8_t*)AscendC::GmAlloc(scaleInputByteSzie);
    uint8_t* offset_input = (uint8_t*)AscendC::GmAlloc(offsetInputByteSzie);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* scale_out = (uint8_t*)AscendC::GmAlloc(scaleOutByteSize);

    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    GeluQuantTilingData* tilingDatafromBin = reinterpret_cast<GeluQuantTilingData*>(tiling);

    tilingDatafromBin->usedCoreNum = 24;
    tilingDatafromBin->normalCoreProcessNum = 1;
    tilingDatafromBin->tailCoreProcessNum = 1;
    tilingDatafromBin->endAxisLen = 6912;
    tilingDatafromBin->endAxisLenAligned = 6912;
    tilingDatafromBin->coexistentNodeNum = 6;
    tilingDatafromBin->coexistentNodeElementNum = 7000;
    tilingDatafromBin->rowInner = 1;
    tilingDatafromBin->rowTail = 1;
    tilingDatafromBin->rowOuter = 1;
    tilingDatafromBin->colInner = 1;
    tilingDatafromBin->colTail = 1;
    tilingDatafromBin->colOuter = 1;
    tilingDatafromBin->quantMode = 1;
    tilingDatafromBin->approximate = 1;
    tilingDatafromBin->inputScaleType = 2;
    tilingDatafromBin->inputOffsetType = 0;
    tilingDatafromBin->tilingKey = 0;

    ICPU_SET_TILING_KEY(1011);
    ICPU_RUN_KF(
        gelu_quant, blockDim, x_input, scale_input, offset_input, y, scale_out, workSpace,
        (uint8_t*)(tilingDatafromBin));

    ICPU_SET_TILING_KEY(1012);
    ICPU_RUN_KF(
        gelu_quant, blockDim, x_input, scale_input, offset_input, y, scale_out, workSpace,
        (uint8_t*)(tilingDatafromBin));

    ICPU_SET_TILING_KEY(1013);
    ICPU_RUN_KF(
        gelu_quant, blockDim, x_input, scale_input, offset_input, y, scale_out, workSpace,
        (uint8_t*)(tilingDatafromBin));

    ICPU_SET_TILING_KEY(1014);
    ICPU_RUN_KF(
        gelu_quant, blockDim, x_input, scale_input, offset_input, y, scale_out, workSpace,
        (uint8_t*)(tilingDatafromBin));

    ICPU_SET_TILING_KEY(1015);
    ICPU_RUN_KF(
        gelu_quant, blockDim, x_input, scale_input, offset_input, y, scale_out, workSpace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x_input);
    AscendC::GmFree(scale_input);
    AscendC::GmFree(offset_input);
    AscendC::GmFree(y);
    AscendC::GmFree(scale_out);
    AscendC::GmFree(workSpace);
    AscendC::GmFree(tiling);
}
TEST_F(gelu_quant_test, test_case_gelu_quant_2)
{
    size_t inputByteSize = 1 * 24 * 6912 * sizeof(uint32_t);
    size_t scaleInputByteSzie = 6912 * sizeof(uint32_t);
    size_t offsetInputByteSzie = 6912 * sizeof(uint32_t);
    size_t outputByteSize = 1 * 24 * 6912 * sizeof(int8_t);
    size_t scaleOutByteSize = 6912 * sizeof(uint32_t);
    size_t tiling_data_size = sizeof(GeluQuantTilingData);
    uint32_t blockDim = 24;

    uint8_t* x_input = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* scale_input = (uint8_t*)AscendC::GmAlloc(scaleInputByteSzie);
    uint8_t* offset_input = (uint8_t*)AscendC::GmAlloc(offsetInputByteSzie);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* scale_out = (uint8_t*)AscendC::GmAlloc(scaleOutByteSize);

    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    GeluQuantTilingData* tilingDatafromBin = reinterpret_cast<GeluQuantTilingData*>(tiling);

    tilingDatafromBin->usedCoreNum = 24;
    tilingDatafromBin->normalCoreProcessNum = 1;
    tilingDatafromBin->tailCoreProcessNum = 1;
    tilingDatafromBin->endAxisLen = 6912;
    tilingDatafromBin->endAxisLenAligned = 6912;
    tilingDatafromBin->coexistentNodeNum = 6;
    tilingDatafromBin->coexistentNodeElementNum = 7000;
    tilingDatafromBin->rowInner = 1;
    tilingDatafromBin->rowTail = 1;
    tilingDatafromBin->rowOuter = 1;
    tilingDatafromBin->colInner = 1;
    tilingDatafromBin->colTail = 1;
    tilingDatafromBin->colOuter = 1;
    tilingDatafromBin->quantMode = 1;
    tilingDatafromBin->approximate = 1;
    tilingDatafromBin->inputScaleType = 2;
    tilingDatafromBin->inputOffsetType = 0;
    tilingDatafromBin->tilingKey = 0;

    ICPU_SET_TILING_KEY(1021);
    ICPU_RUN_KF(
        gelu_quant, blockDim, x_input, scale_input, offset_input, y, scale_out, workSpace,
        (uint8_t*)(tilingDatafromBin));

    ICPU_SET_TILING_KEY(1022);
    ICPU_RUN_KF(
        gelu_quant, blockDim, x_input, scale_input, offset_input, y, scale_out, workSpace,
        (uint8_t*)(tilingDatafromBin));

    ICPU_SET_TILING_KEY(1023);
    ICPU_RUN_KF(
        gelu_quant, blockDim, x_input, scale_input, offset_input, y, scale_out, workSpace,
        (uint8_t*)(tilingDatafromBin));

    ICPU_SET_TILING_KEY(1024);
    ICPU_RUN_KF(
        gelu_quant, blockDim, x_input, scale_input, offset_input, y, scale_out, workSpace,
        (uint8_t*)(tilingDatafromBin));

    ICPU_SET_TILING_KEY(1025);
    ICPU_RUN_KF(
        gelu_quant, blockDim, x_input, scale_input, offset_input, y, scale_out, workSpace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x_input);
    AscendC::GmFree(scale_input);
    AscendC::GmFree(offset_input);
    AscendC::GmFree(y);
    AscendC::GmFree(scale_out);
    AscendC::GmFree(workSpace);
    AscendC::GmFree(tiling);
}
TEST_F(gelu_quant_test, test_case_gelu_quant_3)
{
    size_t inputByteSize = 1 * 24 * 6912 * sizeof(uint32_t);
    size_t scaleInputByteSzie = 6912 * sizeof(uint32_t);
    size_t offsetInputByteSzie = 6912 * sizeof(uint32_t);
    size_t outputByteSize = 1 * 24 * 6912 * sizeof(int8_t);
    size_t scaleOutByteSize = 6912 * sizeof(uint32_t);
    size_t tiling_data_size = sizeof(GeluQuantTilingData);
    uint32_t blockDim = 24;

    uint8_t* x_input = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* scale_input = (uint8_t*)AscendC::GmAlloc(scaleInputByteSzie);
    uint8_t* offset_input = (uint8_t*)AscendC::GmAlloc(offsetInputByteSzie);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* scale_out = (uint8_t*)AscendC::GmAlloc(scaleOutByteSize);

    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    GeluQuantTilingData* tilingDatafromBin = reinterpret_cast<GeluQuantTilingData*>(tiling);

    tilingDatafromBin->usedCoreNum = 24;
    tilingDatafromBin->normalCoreProcessNum = 1;
    tilingDatafromBin->tailCoreProcessNum = 1;
    tilingDatafromBin->endAxisLen = 6912;
    tilingDatafromBin->endAxisLenAligned = 6912;
    tilingDatafromBin->coexistentNodeNum = 6;
    tilingDatafromBin->coexistentNodeElementNum = 7000;
    tilingDatafromBin->rowInner = 1;
    tilingDatafromBin->rowTail = 1;
    tilingDatafromBin->rowOuter = 1;
    tilingDatafromBin->colInner = 1;
    tilingDatafromBin->colTail = 1;
    tilingDatafromBin->colOuter = 1;
    tilingDatafromBin->quantMode = 1;
    tilingDatafromBin->approximate = 1;
    tilingDatafromBin->inputScaleType = 2;
    tilingDatafromBin->inputOffsetType = 0;
    tilingDatafromBin->tilingKey = 0;

    ICPU_SET_TILING_KEY(1031);
    ICPU_RUN_KF(
        gelu_quant, blockDim, x_input, scale_input, offset_input, y, scale_out, workSpace,
        (uint8_t*)(tilingDatafromBin));

    ICPU_SET_TILING_KEY(1032);
    ICPU_RUN_KF(
        gelu_quant, blockDim, x_input, scale_input, offset_input, y, scale_out, workSpace,
        (uint8_t*)(tilingDatafromBin));

    ICPU_SET_TILING_KEY(1033);
    ICPU_RUN_KF(
        gelu_quant, blockDim, x_input, scale_input, offset_input, y, scale_out, workSpace,
        (uint8_t*)(tilingDatafromBin));

    ICPU_SET_TILING_KEY(1034);
    ICPU_RUN_KF(
        gelu_quant, blockDim, x_input, scale_input, offset_input, y, scale_out, workSpace,
        (uint8_t*)(tilingDatafromBin));

    ICPU_SET_TILING_KEY(1035);
    ICPU_RUN_KF(
        gelu_quant, blockDim, x_input, scale_input, offset_input, y, scale_out, workSpace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x_input);
    AscendC::GmFree(scale_input);
    AscendC::GmFree(offset_input);
    AscendC::GmFree(y);
    AscendC::GmFree(scale_out);
    AscendC::GmFree(workSpace);
    AscendC::GmFree(tiling);
}

TEST_F(gelu_quant_test, test_case_gelu_quant_4)
{
    size_t inputByteSize = 1 * 24 * 6912 * sizeof(uint32_t);
    size_t scaleInputByteSzie = 6912 * sizeof(uint32_t);
    size_t offsetInputByteSzie = 6912 * sizeof(uint32_t);
    size_t outputByteSize = 1 * 24 * 6912 * sizeof(int8_t);
    size_t scaleOutByteSize = 6912 * sizeof(uint32_t);
    size_t tiling_data_size = sizeof(GeluQuantTilingData);
    uint32_t blockDim = 24;

    uint8_t* x_input = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* scale_input = (uint8_t*)AscendC::GmAlloc(scaleInputByteSzie);
    uint8_t* offset_input = (uint8_t*)AscendC::GmAlloc(offsetInputByteSzie);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* scale_out = (uint8_t*)AscendC::GmAlloc(scaleOutByteSize);

    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    GeluQuantTilingData* tilingDatafromBin = reinterpret_cast<GeluQuantTilingData*>(tiling);

    tilingDatafromBin->usedCoreNum = 24;
    tilingDatafromBin->normalCoreProcessNum = 1;
    tilingDatafromBin->tailCoreProcessNum = 1;
    tilingDatafromBin->endAxisLen = 6912;
    tilingDatafromBin->endAxisLenAligned = 6912;
    tilingDatafromBin->coexistentNodeNum = 6;
    tilingDatafromBin->coexistentNodeElementNum = 7000;
    tilingDatafromBin->rowInner = 1;
    tilingDatafromBin->rowTail = 1;
    tilingDatafromBin->rowOuter = 1;
    tilingDatafromBin->colInner = 1;
    tilingDatafromBin->colTail = 1;
    tilingDatafromBin->colOuter = 1;
    tilingDatafromBin->quantMode = 1;
    tilingDatafromBin->approximate = 1;
    tilingDatafromBin->inputScaleType = 2;
    tilingDatafromBin->inputOffsetType = 0;
    tilingDatafromBin->tilingKey = 0;

    ICPU_SET_TILING_KEY(1041);
    ICPU_RUN_KF(
        gelu_quant, blockDim, x_input, scale_input, offset_input, y, scale_out, workSpace,
        (uint8_t*)(tilingDatafromBin));

    ICPU_SET_TILING_KEY(1042);
    ICPU_RUN_KF(
        gelu_quant, blockDim, x_input, scale_input, offset_input, y, scale_out, workSpace,
        (uint8_t*)(tilingDatafromBin));

    ICPU_SET_TILING_KEY(1043);
    ICPU_RUN_KF(
        gelu_quant, blockDim, x_input, scale_input, offset_input, y, scale_out, workSpace,
        (uint8_t*)(tilingDatafromBin));

    ICPU_SET_TILING_KEY(1044);
    ICPU_RUN_KF(
        gelu_quant, blockDim, x_input, scale_input, offset_input, y, scale_out, workSpace,
        (uint8_t*)(tilingDatafromBin));

    ICPU_SET_TILING_KEY(1045);
    ICPU_RUN_KF(
        gelu_quant, blockDim, x_input, scale_input, offset_input, y, scale_out, workSpace,
        (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x_input);
    AscendC::GmFree(scale_input);
    AscendC::GmFree(offset_input);
    AscendC::GmFree(y);
    AscendC::GmFree(scale_out);
    AscendC::GmFree(workSpace);
    AscendC::GmFree(tiling);
}