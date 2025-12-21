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
#include "test_add_rms_norm_cast_regbase.h"
#include "data_utils.h"

#include <cstdint>

using namespace std;

extern "C" void add_rms_norm_cast(
    uint8_t* x1, uint8_t* x2, uint8_t* gamma, uint8_t* y1, uint8_t* y2, uint8_t* rstd, uint8_t* x, uint8_t* workspace,
    uint8_t* tiling);

class add_rms_norm_cast_regbase_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "add_rms_norm_cast_regbase_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "add_rms_norm_cast_regbase_test TearDown\n" << endl;
    }
};

TEST_F(add_rms_norm_cast_regbase_test, test_case_dynamic_dual_smooth)
{
    int M = 3;
    int N = 64;
    size_t xByteSize = M * N * sizeof(int16_t);
    size_t gammaByteSize = N * sizeof(int16_t);
    size_t rstdByteSize = M * sizeof(float);
    size_t y1ByteSize = M * N * sizeof(float);
    size_t y2ByteSize = M * N * sizeof(int16_t);
    size_t tilingDataSize = sizeof(AddRmsNormCastRegbaseTilingData);

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* gamma = (uint8_t*)AscendC::GmAlloc(gammaByteSize);

    uint8_t* y1 = (uint8_t*)AscendC::GmAlloc(y1ByteSize);
    uint8_t* y2 = (uint8_t*)AscendC::GmAlloc(y2ByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(rstdByteSize);
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024 + 1);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 3;

    char* path_ = get_current_dir_name();
    string path(path_);

    AddRmsNormCastRegbaseTilingData* tilingDatafromBin = reinterpret_cast<AddRmsNormCastRegbaseTilingData*>(tiling);

    tilingDatafromBin->numM = M;
    tilingDatafromBin->numN = N;
    tilingDatafromBin->baseM = M;
    tilingDatafromBin->baseN = N;
    tilingDatafromBin->baseNDtypeAlign = N;
    tilingDatafromBin->baseNReduceAlign = 128;
    tilingDatafromBin->powerSplit = 128;
    tilingDatafromBin->powerLoop = 1;
    tilingDatafromBin->mPerCore = 1;
    tilingDatafromBin->mLastCore = 1;
    tilingDatafromBin->epsilon = 1e-6;
    tilingDatafromBin->avgFactor = (1.0 / N);

    ICPU_SET_TILING_KEY(100);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(add_rms_norm_cast, blockDim, x1, x2, gamma, y1, y2, rstd, x, workspace, (uint8_t*)(tilingDatafromBin));

    AscendC::GmFree(x1);
    AscendC::GmFree(x2);
    AscendC::GmFree(gamma);
    AscendC::GmFree(y1);
    AscendC::GmFree(y2);
    AscendC::GmFree(rstd);
    AscendC::GmFree(x);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}
