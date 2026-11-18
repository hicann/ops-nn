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
 * \file test_grouped_dynamic_hi_f4_quant.cpp
 * \brief op_kernel UT for grouped_dynamic_hi_f4_quant
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "../../../op_kernel/grouped_dynamic_hi_f4_quant.cpp"
#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void grouped_dynamic_hi_f4_quant(GM_ADDR x, GM_ADDR group_index, GM_ADDR y,
                                                                  GM_ADDR scale, GM_ADDR workspace, GM_ADDR tiling);

class grouped_dynamic_hi_f4_quant_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "grouped_dynamic_hi_f4_quant_test SetUp\n" << endl; }
    static void TearDownTestCase() { cout << "grouped_dynamic_hi_f4_quant_test TearDown\n" << endl; }
};

TEST_F(grouped_dynamic_hi_f4_quant_test, test_case_bf16_fp8_e5m2_scale_alg_0)
{
    int64_t rowNum = 128;
    int64_t colNum = 256;
    int64_t groupNum = 1;

    size_t inputXSize = rowNum * colNum * sizeof(bfloat16_t);
    size_t outputY = rowNum * colNum / 2;
    size_t outputScale = (rowNum / 64 + groupNum) * colNum * sizeof(float);
    size_t groupIndexSize = groupNum * sizeof(int32_t);
    size_t tiling_data_size = sizeof(GroupedDynamicHiF4QuantTilingData);
    uint32_t blockDim = 64;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(inputXSize);
    int32_t* group_index = (int32_t*)AscendC::GmAlloc(groupIndexSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(outputY);
    uint8_t* mxScale = (uint8_t*)AscendC::GmAlloc(outputScale);

    uint8_t* workSpace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    GroupedDynamicHiF4QuantTilingData* tilingDatafromBin = reinterpret_cast<GroupedDynamicHiF4QuantTilingData*>(tiling);

    group_index[0] = rowNum;

    tilingDatafromBin->tilingKey = 1;
    tilingDatafromBin->totalCoreNum = 64;
    tilingDatafromBin->usedCoreNum = 64;
    tilingDatafromBin->groupNum = 1;
    tilingDatafromBin->rowSize = colNum;
    tilingDatafromBin->colSize = rowNum;
    tilingDatafromBin->blockRowSize = 128;
    tilingDatafromBin->tailBlockRowSize = 128;
    tilingDatafromBin->blockColSize = 256;
    tilingDatafromBin->blockRowNum = colNum / 128;
    tilingDatafromBin->scaleAlg = 0;

    ICPU_SET_TILING_KEY(1);

    auto grouped_dynamic_hi_f4_quant_kernel = [](GM_ADDR x, GM_ADDR group_index, GM_ADDR y, GM_ADDR scale,
                                                 GM_ADDR workSpace, GM_ADDR tiling) {
        ::grouped_dynamic_hi_f4_quant<1>(x, (uint8_t*)group_index, y, scale, workSpace, tiling);
    };
    ICPU_RUN_KF(grouped_dynamic_hi_f4_quant_kernel, blockDim, x, (uint8_t*)group_index, y, mxScale, workSpace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(group_index);
    AscendC::GmFree(y);
    AscendC::GmFree(mxScale);
    AscendC::GmFree(workSpace);
    AscendC::GmFree(tiling);
    free(path_);
}
