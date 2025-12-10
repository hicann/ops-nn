/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
*/
#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "group_quant_tiling_def.h"
#include "data_utils.h"

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void group_quant(
    GM_ADDR x, GM_ADDR scale, GM_ADDR groupIndex, GM_ADDR offset, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

class GroupQuantTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "GroupQuantTest SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "GroupQuantTest TearDown\n" << endl;
    }
};

TEST_F(GroupQuantTest, op_demo_case)
{
    size_t xByteSize = 6 * 4 * sizeof(float);
    size_t scaleByteSize = 4 * 4 * sizeof(float);
    size_t groupIndexByteSize = 4 * sizeof(int32_t);
    size_t offsetByteSize = 1 * sizeof(float);
    size_t yByteSize = 6 * 4 * sizeof(float);
    size_t tilingDataSize = sizeof(GroupQuantTilingData);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* scale = (uint8_t*)AscendC::GmAlloc(scaleByteSize);
    uint8_t* groupIndex = (uint8_t*)AscendC::GmAlloc(groupIndexByteSize);
    uint8_t* offset = (uint8_t*)AscendC::GmAlloc(offsetByteSize);

    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 6;

    char* path_ = get_current_dir_name();
    string path(path_);

    GroupQuantTilingData* tilingDatafromBin = reinterpret_cast<GroupQuantTilingData*>(tiling);

    tilingDatafromBin->dimS = 6;
    tilingDatafromBin->dimE = 4;
    tilingDatafromBin->dimH = 4;
    tilingDatafromBin->hasOffset = 1;
    tilingDatafromBin->needCoreNum = 6;
    tilingDatafromBin->preCoreNum = 6;
    tilingDatafromBin->xRowNumPreCore = 1;
    tilingDatafromBin->xRowNumPostCore = 1;
    reinterpret_cast<int32_t*>(groupIndex)[tilingDatafromBin->dimE - 1] = tilingDatafromBin->dimS;

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(group_quant, blockDim, x, scale, groupIndex, offset, y, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x);
    AscendC::GmFree(scale);
    AscendC::GmFree(offset);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}