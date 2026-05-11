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
#include "data_utils.h"

#include "../../../op_kernel/gelu_v2_apt.cpp"
#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void gelu_v2(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

class gelu_v2_test : public testing::Test
{
protected:
    static void SetUpTestCase()
    {
        cout << "gelu_v2_test SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "gelu_v2_test TearDown\n" << endl;
    }
};

TEST_F(gelu_v2_test, test_case_fp32_tanh)
{
    size_t xByteSize = 256 * sizeof(float);
    size_t yByteSize = 256 * sizeof(float);
    size_t tiling_data_size = sizeof(EleBaseTilingData16B);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yByteSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(16*1024*1024);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    uint32_t blockDim = 1;
    system("cp -r ../../../../activation/gelu_v2/tests/ut/op_kernel/gelu_v2_data ./");
    system("chmod -R 755 ./gelu_v2_data/");
    system("cd ./gelu_v2_data/ && rm -rf ./*bin");
    system("cd ./gelu_v2_data/ && python3 gen_data.py '(256)' float32 tanh");

    char* path_ = get_current_dir_name();
    string path(path_);

    EleBaseTilingData16B* tilingDatafromBin = reinterpret_cast<EleBaseTilingData16B*>(tiling);

    tilingDatafromBin->dim0 = 256;
    tilingDatafromBin->coreNum = 1;
    tilingDatafromBin->ubFormer = 1024;
    ReadFile(path + "/gelu_v2_data/input_x.bin", xByteSize, x, xByteSize);
    auto KernelGeluV2 = [](GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
        ::gelu_v2<0, TPL_TANH, TPL_FP32>(x, y, workspace, tiling);
    };
    ICPU_SET_TILING_KEY(1003);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(KernelGeluV2, blockDim, x, y, workspace, (uint8_t*)(tilingDatafromBin));
    WriteFile(path + "/gelu_v2_data/output.bin", y, yByteSize);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);
    free(path_);
}
