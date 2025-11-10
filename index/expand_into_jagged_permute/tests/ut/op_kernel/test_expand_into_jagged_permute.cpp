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
 * \file test_expand_into_jagged_permute.cpp
 * \brief
 */

#include <array>
#include <vector>
#include "gtest/gtest.h"
#include "test_expand_into_jagged_permute_tiling_def.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#include "string.h"
#include <iostream>
#include <string>
#endif

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void expand_into_jagged_permute(
    GM_ADDR permute, GM_ADDR inputOffsets, GM_ADDR outputOffsets, GM_ADDR outputPermute, GM_ADDR workspace,
    GM_ADDR tiling);

class expand_into_jagged_permute_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "expand_into_jagged_permute_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "expand_into_jagged_permute_test TearDown\n" << endl;
    }
};

TEST_F(expand_into_jagged_permute_test, test_int32)
{
    size_t permuteByteSize = 3 * sizeof(int32_t);
    size_t inputOffsetsByteSize = 4 * sizeof(int32_t);
    size_t outputOffsetsByteSize = 6144 * 8 * sizeof(int32_t);
    // output
    size_t outputPermuteByteSize = 6 * sizeof(int32_t);

    size_t tilingDataSize = sizeof(ExpandIntoJaggedPermuteTilingDataDef);

    uint8_t* permute = (uint8_t*)AscendC::GmAlloc(permuteByteSize + 32);
    uint8_t* inputOffsets = (uint8_t*)AscendC::GmAlloc(inputOffsetsByteSize + 32);
    uint8_t* outputOffsets = (uint8_t*)AscendC::GmAlloc(outputOffsetsByteSize + 32);
    uint8_t* outputPermute = (uint8_t*)AscendC::GmAlloc(outputPermuteByteSize + 32);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 32);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize + 32);
    uint32_t blockDim = 48;

    char* path_ = get_current_dir_name();
    string path(path_);

    ExpandIntoJaggedPermuteTilingDataDef* tilingDatafromBin =
        reinterpret_cast<ExpandIntoJaggedPermuteTilingDataDef*>(tiling);
    tilingDatafromBin->realCoreNum = 48;
    tilingDatafromBin->frontCoreNum = 1;
    tilingDatafromBin->blockFactor = 1;
    tilingDatafromBin->tailCoreBlockFactor = 1;
    tilingDatafromBin->tailCoreBlockFactor = 1;
    tilingDatafromBin->lastTaskLen = 3;
    tilingDatafromBin->oneTaskLen = 128;
    tilingDatafromBin->oneTaskOffsetLen = 5;
    tilingDatafromBin->inputLen = 1;
    tilingDatafromBin->outputSize = 6;
    tilingDatafromBin->offsetLen = 4;

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(
        expand_into_jagged_permute, blockDim, permute, inputOffsets, outputOffsets, outputPermute, workspace, tiling);

    AscendC::GmFree(permute);
    AscendC::GmFree(inputOffsets);
    AscendC::GmFree(outputOffsets);
    AscendC::GmFree(outputPermute);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}
