/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
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
#include "../../../op_kernel/median.cpp"
#include "../../../op_kernel/median_tiling_data.h"
#include <cstdint>

using namespace std;

class median_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "median_test SetUp\n" << endl; }
    static void TearDownTestCase() { cout << "median_test TearDown\n" << endl; }
};

TEST_F(median_test, test_case_0)
{
    size_t xByteSize = 32 * 4 * 4 * 4 * sizeof(float);
    size_t yByteSize = 32 * 4 * 4 * 4 * sizeof(float);
    size_t zByteSize = 32 * 4 * 4 * 4 * sizeof(float);
    size_t tiling_data_size = sizeof(MedianTilingData);
    uint32_t numBlocks = 8;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* z = (uint8_t*)AscendC::GmAlloc(zByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    char* path_ = get_current_dir_name();
    string path(path_);

    MedianTilingData* tilingDatafromBin = reinterpret_cast<MedianTilingData*>(tiling);

    tilingDatafromBin->batch = 512;
    tilingDatafromBin->redLen = 4;
    tilingDatafromBin->mid = 1;
    tilingDatafromBin->nSeg = 0;
    tilingDatafromBin->pad = 0;

    // dtype 由编译期 DTYPE_INPUT=float 决定；redLen=4 且 fp → SMALL 路径；path 现由 tilingkey(schMode) 编译期区分，
    // 故直接实例化 schMode=MEDIAN_PATH_SMALL 的 kernel（host 侧本会为此 shape 选定该 tilingkey）。
    auto MedianKernel = [](GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
        ::median<MEDIAN_PATH_SMALL>(x, y, z, workspace, tiling);
    };

    ICPU_SET_TILING_KEY(MEDIAN_PATH_SMALL);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(MedianKernel, numBlocks, x, y, z, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(z);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}
