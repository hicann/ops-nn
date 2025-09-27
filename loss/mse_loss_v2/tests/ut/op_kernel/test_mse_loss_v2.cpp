/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "test_mse_loss_v2.h"
#include "data_utils.h"

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void mse_loss_v2(
    GM_ADDR input, GM_ADDR target, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling);
class mse_loss_v2_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "mse_loss_v2_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "mse_loss_v2_test TearDown\n" << endl;
    }
};

TEST_F(mse_loss_v2_test, test_case_mean_fp16_01)
{
    system("cp -r ../../../../loss/mse_loss_v2/tests/ut/op_kernel/gen_data ./");
    system("chmod -R 755 ./gen_data/");
    system("cd ./gen_data/ && python3 gen_data.py '[30, 1024]' '[-100, 100]' 'float16' 'mean' ");

    size_t inputByteSize = 30 * 1024 * sizeof(half);
    size_t targetByteSize = 30 * 1024 * sizeof(half);
    // output
    size_t outputByteSize = sizeof(half);
    size_t tilingDataSize = sizeof(MSELossV2TilingDataTest);

    uint8_t* input = (uint8_t*)AscendC::GmAlloc(inputByteSize);
    uint8_t* target = (uint8_t*)AscendC::GmAlloc(targetByteSize);
    uint8_t* output = (uint8_t*)AscendC::GmAlloc(targetByteSize);

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16 * 1024 * 1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);
    uint32_t blockDim = 1;

    char* path_ = get_current_dir_name();
    string path(path_);

    auto read_file_0 = ReadFile(path + "/gen_data/input_input.bin", inputByteSize, input, inputByteSize);
    auto read_file_1 = ReadFile(path + "/gen_data/input_target.bin", targetByteSize, target, targetByteSize);
    cout << read_file_0 << " read_file_0 SetUp\n" << endl;
    cout << read_file_1 << " read_file_1 SetUp\n" << endl;
    MSELossV2TilingDataTest* tilingDatafromBin = reinterpret_cast<MSELossV2TilingDataTest*>(tiling);

    tilingDatafromBin->coreNum = 8;
    tilingDatafromBin->tailElems = 0;
    tilingDatafromBin->bufferNum = 1;
    tilingDatafromBin->scale = 3.255208333e-05;
    tilingDatafromBin->epochs = 0;
    tilingDatafromBin->epochsForLastCore = 0;
    tilingDatafromBin->coreLength = 3840;
    tilingDatafromBin->tileLength = 16336;
    tilingDatafromBin->tailTileLength = 3840;
    tilingDatafromBin->tailTileLengthForLastCore = 3840;

    ICPU_SET_TILING_KEY(32);
    ICPU_RUN_KF(mse_loss_v2, blockDim, input, target, output, workspace, (uint8_t*)(tilingDatafromBin));

    WriteFile(path + "/gen_data/out.bin", output, outputByteSize);

    AscendC::GmFree(input);
    AscendC::GmFree(target);
    AscendC::GmFree(output);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);

    // int res = system("cd ./gen_data/ && python3 mse_loss_v2_data_compare.py 'float16'");
    system("rm -rf gen_data");
    // ASSERT_EQ(res, 0);
}
