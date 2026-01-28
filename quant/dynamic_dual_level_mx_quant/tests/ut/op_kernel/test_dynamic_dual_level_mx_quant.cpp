/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_dynamic_dual_level_mx_quant.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "../../../op_kernel/dynamic_dual_level_mx_quant.cpp"
#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void dynamic_dual_level_mx_quant(
    GM_ADDR x, GM_ADDR smooth_scale, GM_ADDR y, GM_ADDR level0_scale, GM_ADDR level1_scale, GM_ADDR workspace,
    GM_ADDR tiling);

class dynamic_dual_level_mx_quant_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "dynamic_dual_level_mx_quant_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "dynamic_dual_level_mx_quant_test TearDown\n" << endl;
    }
};

TEST_F(dynamic_dual_level_mx_quant_test, test_case_100)
{
    size_t inputXSize = 128 * 512 * sizeof(half);
    size_t smoothScaleSize = 512 * sizeof(half);
    size_t outputYSize = 128 * 256 * sizeof(int8_t);
    size_t level0ScaleSize = 128 * 1 * sizeof(float);
    size_t level1ScaleSize = 128 * 8 * 2 * sizeof(int8_t);
    size_t tiling_data_size = sizeof(DynamicDualLevelMxQuantTilingData);
    uint32_t blockDim = 2;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXSize);
    uint8_t* smoothScale = (uint8_t*)AscendC::GmAlloc(smoothScaleSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputYSize);
    uint8_t* level0Scale = (uint8_t*)AscendC::GmAlloc(level0ScaleSize);
    uint8_t* level1Scale = (uint8_t*)AscendC::GmAlloc(level1ScaleSize);

    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    std::string path_ = get_current_dir_name();
    string path(path_);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    DynamicDualLevelMxQuantTilingData* tilingDatafromBin = reinterpret_cast<DynamicDualLevelMxQuantTilingData*>(tiling);

    tilingDatafromBin->tilingKey = 4;
    tilingDatafromBin->totalCoreNum = 64;
    tilingDatafromBin->usedCoreNum = 2;
    tilingDatafromBin->roundMode = 4;
    tilingDatafromBin->level0BlockSize = 35;
    tilingDatafromBin->level1BlockSize = 0;
    tilingDatafromBin->rowSize = 32;
    tilingDatafromBin->colSize = 1;
    tilingDatafromBin->blockSizeRow = 128;
    tilingDatafromBin->blockSizeCol = 256;
    tilingDatafromBin->rowBlockNum = 256;
    tilingDatafromBin->colBlockNum = 64;
    tilingDatafromBin->rowTileNum = 0;
    tilingDatafromBin->colTileNum = 64;
    tilingDatafromBin->normalTileRowBlockNum = 256;
    tilingDatafromBin->normalTileColBlockNum = 2;
    tilingDatafromBin->tailTileRowBlockNum = 1;
    tilingDatafromBin->tailTileColBlockNum = 1;
    tilingDatafromBin->normalTileRowSize = 0;
    tilingDatafromBin->tailTileRowSize = 2;
    tilingDatafromBin->normalTileRowLoopNum = 0;
    tilingDatafromBin->normalTileColLoopNum = 0;
    tilingDatafromBin->tailTileRowLoopNum = 0;
    tilingDatafromBin->tailTileColLoopNum = 2;
    tilingDatafromBin->ubFactor = 8;
    tilingDatafromBin->copyMethod = 4;
    tilingDatafromBin->tailAlignNum = 4;

    ICPU_SET_TILING_KEY(0);

    auto dynamic_dual_level_mx_quant_kernel = [](GM_ADDR x, GM_ADDR smoothScale, GM_ADDR y, GM_ADDR level0Scale,
                                                 GM_ADDR level1Scale, GM_ADDR workSpace, GM_ADDR tiling) {
        ::dynamic_dual_level_mx_quant<4>(x, smoothScale, y, level0Scale, level1Scale, workSpace, tiling);
    };
    ICPU_RUN_KF(
        dynamic_dual_level_mx_quant_kernel, blockDim, x, smoothScale, y, level0Scale, level1Scale, workSpace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(smoothScale);
    AscendC::GmFree(y);
    AscendC::GmFree(level0Scale);
    AscendC::GmFree(level1Scale);
    AscendC::GmFree(workSpace);
    AscendC::GmFree(tilingDatafromBin);
    free(path_);
}