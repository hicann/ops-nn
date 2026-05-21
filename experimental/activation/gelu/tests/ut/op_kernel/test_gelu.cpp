/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file in in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <array>
#include <vector>
#include "gtest/gtest.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#include "string.h"
#include <iostream>
#include <string>
#endif
#include "../../../op_kernel/gelu.cpp"
#include "../../../op_kernel/gelu_tiling_data.h"
#include <cstdint>

using namespace std;

class gelu_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "gelu_test SetUp\n" << endl; }
    static void TearDownTestCase() { cout << "gelu_test TearDown\n" << endl; }
};

TEST_F(gelu_test, test_case_fp32)
{
    size_t totalNum = 32 * 4 * 4 * 4;
    size_t xByteSize = totalNum * sizeof(float);
    size_t yByteSize = totalNum * sizeof(float);
    size_t tiling_data_size = sizeof(GeluTilingData);
    uint32_t numBlocks = 8;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    GeluTilingData* tilingDatafromBin = reinterpret_cast<GeluTilingData*>(tiling);

    tilingDatafromBin->smallCoreDataNum = 2048;
    tilingDatafromBin->bigCoreDataNum = 2048;
    tilingDatafromBin->finalBigTileNum = 6;
    tilingDatafromBin->finalSmallTileNum = 6;
    tilingDatafromBin->tileDataNum = 384;
    tilingDatafromBin->smallTailDataNum = 2048;
    tilingDatafromBin->bigTailDataNum = 2048;
    tilingDatafromBin->tailBlockNum = 0;

    auto GeluKernel = [](GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
        ::gelu<0>(x, y, workspace, tiling);
    };

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(GeluKernel, numBlocks, x, y, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}

TEST_F(gelu_test, test_case_fp16)
{
    size_t totalNum = 32 * 4 * 4 * 4;
    size_t xByteSize = totalNum * sizeof(float16);
    size_t yByteSize = totalNum * sizeof(float16);
    size_t tiling_data_size = sizeof(GeluTilingData);
    uint32_t numBlocks = 8;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    GeluTilingData* tilingDatafromBin = reinterpret_cast<GeluTilingData*>(tiling);

    tilingDatafromBin->smallCoreDataNum = 4096;
    tilingDatafromBin->bigCoreDataNum = 4096;
    tilingDatafromBin->finalBigTileNum = 6;
    tilingDatafromBin->finalSmallTileNum = 6;
    tilingDatafromBin->tileDataNum = 768;
    tilingDatafromBin->smallTailDataNum = 4096;
    tilingDatafromBin->bigTailDataNum = 4096;
    tilingDatafromBin->tailBlockNum = 0;

    auto GeluKernel = [](GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
        ::gelu<0>(x, y, workspace, tiling);
    };

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(GeluKernel, numBlocks, x, y, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
}
