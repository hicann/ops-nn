/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_upsample_nearest.cpp
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
#include "flat_quant_tiling_def.h"

extern "C" __global__ __aicore__ void flat_quant(
    GM_ADDR x, GM_ADDR kroneckerP1, GM_ADDR kroneckerP2, GM_ADDR out, GM_ADDR qscale, GM_ADDR workspace,
    GM_ADDR tiling);

class flat_quant_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "flat_quant_test SetUp\n" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "flat_quant_test TearDown\n" << std::endl;
    }
};

TEST_F(flat_quant_test, test_case_float16_1)
{
    system(
        "cp -rf "
        "../../../../quant/flat_quant/tests/ut/op_kernel/flat_quant_data ./");
    system("chmod -R 755 ./flat_quant_data/");
    system("cd ./flat_quant_data/ && python3 gen_data.py '(16, 16, 16)' '(16, 16)' '(16, 16)' 'float16'");

    size_t xByteSize = 16 * 16 * 16 * sizeof(half);
    size_t p1ByteSize = 16 * 16 * sizeof(half);
    size_t p2ByteSize = 16 * 16 * sizeof(half);
    size_t outByteSize = 16 * 16 * 16 * sizeof(int8_t) / 2;
    size_t scaleByteSize = 16 * sizeof(float);
    size_t workspaceSize = (16 * 16 * 16 * 2 + 16 * 16 + 16 * 16) * sizeof(half) + 16 * 1024 * 1024;
    uint32_t blockDim = 24;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* kronecker_p1 = (uint8_t*)AscendC::GmAlloc(p1ByteSize);
    uint8_t* kronecker_p2 = (uint8_t*)AscendC::GmAlloc(p2ByteSize);
    uint8_t* out = (uint8_t*)AscendC::GmAlloc(outByteSize);
    uint8_t* quant_scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(FlatQuantTilingData));

    ReadFile("./flat_quant_data/float16_input_x_flat_quant.bin", xByteSize, x, xByteSize);
    ReadFile("./flat_quant_data/float16_input_kronecker_p1_flat_quant.bin", p1ByteSize, kronecker_p1, p1ByteSize);
    ReadFile("./flat_quant_data/float16_input_kronecker_p2_flat_quant.bin", p2ByteSize, kronecker_p2, p2ByteSize);

    FlatQuantTilingData* tilingDatafromBin = reinterpret_cast<FlatQuantTilingData*>(tiling);

    tilingDatafromBin->dataType = 0;
    tilingDatafromBin->K = 16;
    tilingDatafromBin->M = 16;
    tilingDatafromBin->N = 16;
    tilingDatafromBin->clipRatio = 1.0f;

    ICPU_SET_TILING_KEY(1);

    ICPU_RUN_KF(
        flat_quant, blockDim, x, kronecker_p1, kronecker_p2, out, quant_scale, workspace,
        (uint8_t*)(tilingDatafromBin));

    WriteFile("./flat_quant_data/float16_output_quant_scale_flat_quant.bin", quant_scale, scaleByteSize);

    AscendC::GmFree((void*)(x));
    AscendC::GmFree((void*)(kronecker_p1));
    AscendC::GmFree((void*)(kronecker_p2));
    AscendC::GmFree((void*)(out));
    AscendC::GmFree((void*)(quant_scale));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    // system("cd ./flat_quant_data/ && python3 compare_data.py");
}