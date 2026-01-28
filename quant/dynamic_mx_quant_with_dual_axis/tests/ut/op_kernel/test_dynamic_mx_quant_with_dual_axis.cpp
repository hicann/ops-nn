/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_dynamic_mx_quant_with_dual_axis.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "../../../op_kernel/dynamic_mx_quant_with_dual_axis.cpp"
#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void dynamic_mx_quant_with_dual_axis(
    GM_ADDR x, GM_ADDR y1, GM_ADDR scale1, GM_ADDR y2, GM_ADDR scale2, GM_ADDR workSpace, GM_ADDR tiling);

class dynamic_mx_quant_with_dual_axis_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "dynamic_mx_quant_with_dual_axis_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "dynamic_mx_quant_with_dual_axis_test TearDown\n" << endl;
    }
};

TEST_F(dynamic_mx_quant_with_dual_axis_test, test_case_100)
{
    size_t inputXSize = 128 * 256 * sizeof(half);
    size_t outputY1 = 128 * 256 * sizeof(int8_t);
    size_t outputScale1 = 128 * 4 * 2 * sizeof(int8_t);
    size_t outputY2 = 128 * 256 * sizeof(int8_t);
    size_t outputScale2 = 2 * 256 * 2 * sizeof(int8_t);
    size_t tiling_data_size = sizeof(DynamicMxQuantWithDualAxisTilingData);
    uint32_t blockDim = 2;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXSize);
    uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(outputY1);
    uint8_t* scale1 = (uint8_t*)AscendC::GmAlloc(outputScale1);
    uint8_t* y2 = (uint8_t*)AscendC::GmAlloc(outputY2);
    uint8_t* scale2 = (uint8_t*)AscendC::GmAlloc(outputScale2);

    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    DynamicMxQuantWithDualAxisTilingData* tilingDatafromBin =
        reinterpret_cast<DynamicMxQuantWithDualAxisTilingData*>(tiling);

    tilingDatafromBin->totalCoreNum = 64;
    tilingDatafromBin->usedCoreNum = 2;
    tilingDatafromBin->roundMode = 4;
    tilingDatafromBin->dstType = 35;
    tilingDatafromBin->scaleAlg = 0;
    tilingDatafromBin->blockSize = 32;
    tilingDatafromBin->dim0 = 1;
    tilingDatafromBin->dimNeg2 = 128;
    tilingDatafromBin->dimNeg1 = 256;
    tilingDatafromBin->blockW = 256;
    tilingDatafromBin->splitBlockH = 64;
    tilingDatafromBin->tilingKey = 0;
    tilingDatafromBin->dimNeg2Tail = 64;
    tilingDatafromBin->dimNeg1Tail = 256;
    tilingDatafromBin->dimNeg2SplitBlockNum = 2;
    tilingDatafromBin->dimNeg1BlockNum = 1;
    tilingDatafromBin->blockPerHeadCore = 1;
    tilingDatafromBin->blockPerTailCore = 0;
    tilingDatafromBin->headCoreNum = 2;
    tilingDatafromBin->dimNeg2IsOdd = 0;
    tilingDatafromBin->dimNeg1IsOdd = 0;
    tilingDatafromBin->dimNeg1IsPad = 0;
    tilingDatafromBin->blockCountPerBatch = 2;
    tilingDatafromBin->scale1ColCountPerBatch = 8;
    tilingDatafromBin->scale2RowCountPerBatch = 4;

    ICPU_SET_TILING_KEY(0);

    auto dynamic_mx_quant_with_dual_axis_kernel = [](GM_ADDR x, GM_ADDR y1, GM_ADDR scale1, GM_ADDR y2, GM_ADDR scale2,
                                                     GM_ADDR workSpace, GM_ADDR tiling) {
        ::dynamic_mx_quant_with_dual_axis<0, 4, 0>(x, y1, scale1, y2, scale2, workSpace, tiling);
    };
    ICPU_RUN_KF(
        dynamic_mx_quant_with_dual_axis_kernel, blockDim, x, y1, scale1, y2, scale2, workSpace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(y1);
    AscendC::GmFree(scale1);
    AscendC::GmFree(y2);
    AscendC::GmFree(scale2);
    AscendC::GmFree(workSpace);
    AscendC::GmFree(tilingDatafromBin);
    free(path_);
}