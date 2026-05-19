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
 * \file test_grouped_dynamic_mx_quant.cpp
 * \brief op_kernel UT for grouped_dynamic_mx_quant
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "../../../op_kernel/grouped_dynamic_mx_quant.cpp"
#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void grouped_dynamic_mx_quant(
    GM_ADDR x, GM_ADDR groupIndex, GM_ADDR y, GM_ADDR mxScale, GM_ADDR workspace, GM_ADDR tiling);

class grouped_dynamic_mx_quant_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "grouped_dynamic_mx_quant_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "grouped_dynamic_mx_quant_test TearDown\n" << endl;
    }
};

TEST_F(grouped_dynamic_mx_quant_test, test_case_bf16_fp8_e5m2_scale_alg_0)
{
    int64_t rowNum = 128;
    int64_t colNum = 256;
    int64_t groupNum = 2;
    int64_t blockSize = 32;

    size_t inputXSize = rowNum * colNum * sizeof(bfloat16_t);
    size_t outputY = rowNum * colNum * sizeof(uint8_t);
    size_t outputScale = 2 * (rowNum / 64 + groupNum) * colNum * sizeof(uint8_t);
    size_t groupIndexSize = groupNum * sizeof(int32_t);
    size_t tiling_data_size = sizeof(GroupedDynamicMxQuantTilingData);
    uint32_t blockDim = 2;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXSize);
    int32_t* groupIndex = (int32_t*)AscendC::GmAlloc(groupIndexSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputY);
    uint8_t* mxScale = (uint8_t*)AscendC::GmAlloc(outputScale);

    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    GroupedDynamicMxQuantTilingData* tilingDatafromBin =
        reinterpret_cast<GroupedDynamicMxQuantTilingData*>(tiling);

    groupIndex[0] = 64;
    groupIndex[1] = 128;

    tilingDatafromBin->totalCoreNum = 64;
    tilingDatafromBin->usedCoreNum = 2;
    tilingDatafromBin->blockFactor = 1;
    tilingDatafromBin->tailBlockFactor = 1;
    tilingDatafromBin->uo = 1;
    tilingDatafromBin->maxUbCol = 64;
    tilingDatafromBin->ubFactor = colNum;
    tilingDatafromBin->tailUbFactor = colNum;
    tilingDatafromBin->blockSize = blockSize;
    tilingDatafromBin->scaleAlg = 0;
    tilingDatafromBin->preAxisSize = rowNum;
    tilingDatafromBin->postAxisSize = colNum;
    tilingDatafromBin->dstTypeMax = 0.0f;

    ICPU_SET_TILING_KEY(1);

    auto grouped_dynamic_mx_quant_kernel = [](GM_ADDR x, GM_ADDR groupIndex, GM_ADDR y, GM_ADDR mxScale,
        GM_ADDR workSpace, GM_ADDR tiling) {
        ::grouped_dynamic_mx_quant<1>(x, (uint8_t*)groupIndex, y, mxScale, workSpace, tiling);
    };
    ICPU_RUN_KF(grouped_dynamic_mx_quant_kernel, blockDim, x, (uint8_t*)groupIndex, y, mxScale, workSpace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(groupIndex);
    AscendC::GmFree(y);
    AscendC::GmFree(mxScale);
    AscendC::GmFree(workSpace);
    AscendC::GmFree(tilingDatafromBin);
    free(path_);
}