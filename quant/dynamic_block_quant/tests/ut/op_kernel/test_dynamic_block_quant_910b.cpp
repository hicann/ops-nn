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
 * \file test_dynamic_block_quant_910b.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"

extern "C" __global__ __aicore__ void dynamic_block_quant(
    GM_ADDR x, GM_ADDR y, GM_ADDR scale, GM_ADDR workspace, GM_ADDR tiling);
class dynamic_block_quant_910b : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        std::cout << "dynamic_block_quant_910b_test SetUp\n" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "dynamic_block_quant_910b_test TearDown\n" << std::endl;
    }
};

TEST_F(dynamic_block_quant_910b, test_case_float16_1)
{
    system(
        "cp -rf "
        "../../../../quant/dynamic_block_quant/tests/ut/op_kernel/dynamic_block_quant_data_910b ./");
    system("chmod -R 755 ./dynamic_block_quant_data_910b/");
    system("cd ./dynamic_block_quant_data_910b/ && python3 gen_data.py '(1, 128)' 'float16'");

    size_t xByteSize = 1 * 128 * sizeof(half);
    size_t yByteSize = 1 * 128 * sizeof(int8_t);
    size_t scaleByteSize = 1 * 1 * sizeof(float);
    size_t workspaceSize = 32;
    uint32_t blockDim = 1;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(DynamicBlockQuantTilingData));

    ReadFile("./dynamic_block_quant_data_910b/float16_input_x_dynamic_block_quant.bin", xByteSize, x, xByteSize);

    DynamicBlockQuantTilingData* tilingDatafromBin = reinterpret_cast<DynamicBlockQuantTilingData*>(tiling);

    tilingDatafromBin->tilingKey = 0;
    tilingDatafromBin->totalCoreNum = 40;
    tilingDatafromBin->ubSize = 196352;
    tilingDatafromBin->minScale = 0.0001f;
    tilingDatafromBin->roundMode = 1;
    tilingDatafromBin->dstType = 2;
    tilingDatafromBin->blockSizeRow = 1;
    tilingDatafromBin->blockSizeCol = 128;
    tilingDatafromBin->rowNum = 1;
    tilingDatafromBin->colNum = 128;
    tilingDatafromBin->usedCoreNum = 1;
    tilingDatafromBin->perCoreRowNum = 0;
    tilingDatafromBin->tailColBlockEndList[0] = 1;

    ICPU_SET_TILING_KEY(0);

    ICPU_RUN_KF(dynamic_block_quant, blockDim, x, y, scale, workspace, (uint8_t*)(tilingDatafromBin));

    WriteFile("./dynamic_block_quant_data_910b/float16_output_scale_dynamic_block_quant.bin", scale, scaleByteSize);

    AscendC::GmFree((void*)(x));
    AscendC::GmFree((void*)(y));
    AscendC::GmFree((void*)(scale));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./dynamic_block_quant_data_910b/ && python3 compare_data.py");
}