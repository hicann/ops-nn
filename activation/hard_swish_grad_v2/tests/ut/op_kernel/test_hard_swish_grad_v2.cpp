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

#include <cstdint>

using namespace std;

extern "C" __global__ __aicore__ void hard_swish_grad_v2(GM_ADDR grad,
                                            GM_ADDR self,
                                            GM_ADDR out,
                                            GM_ADDR workspace,
                                            GM_ADDR tiling);

class hard_swish_grad_v2_test : public testing::Test {
 protected:
  static void SetUpTestCase() {
    cout << "hard_swish_grad_v2 SetUp\n" << endl;
  }
  static void TearDownTestCase() {
    cout << "hard_swish_grad_v2 TearDown\n" << endl;
  }
};

TEST_F(hard_swish_grad_v2_test, test_hard_swish_grad_v2_float_0)
{
    system(
        "cp -rf "
        "../../../../activation/hard_swish_grad_v2/tests/ut/op_kernel/hard_swish_grad_v2_data ./");
    system("chmod -R 755 ./hard_swish_grad_v2_data/");
    system("cd ./hard_swish_grad_v2_data/ && python3 gen_data.py '(2, 4)' 'float32'");
    size_t M = 2;
    size_t N = 4;
    size_t inputShapeSize = M * N * sizeof(float);

    uint8_t *gradOutput = (uint8_t *)AscendC::GmAlloc(inputShapeSize);
    uint8_t *self = (uint8_t *)AscendC::GmAlloc(inputShapeSize);
    uint8_t *out = (uint8_t *)AscendC::GmAlloc(inputShapeSize);

    uint64_t tilingKey = 0;
    uint32_t blockDim = 1;
    size_t workspaceFileSize = 16781184;
    size_t tilingDataSize = sizeof(HardSwishGradV2TilingData);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceFileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);

    std::string gradOutputFileName = "./hard_swish_grad_v2_data/float32_grad_output.bin";
    std::string selfFileName = "./hard_swish_grad_v2_data/float32_self.bin";

    ReadFile(gradOutputFileName, inputShapeSize, gradOutput, inputShapeSize);
    ReadFile(selfFileName, inputShapeSize, self, inputShapeSize);

    HardSwishGradV2TilingData* tilingDatafromBin = reinterpret_cast<HardSwishGradV2TilingData*>(tiling);
    tilingDatafromBin->elementNum = M * N ;
    tilingDatafromBin->needCoreNum = 1;
    tilingDatafromBin->ubSize = 192 * 1024;
    tilingDatafromBin->elementNumEachCore = 8 * 1024;

    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(hard_swish_grad_v2, blockDim, gradOutput, self, out, workspace, (uint8_t*)tilingDatafromBin);
    std::string outFileName = "./hard_swish_grad_v2_data/float32_npuout.bin";
    WriteFile(outFileName, out, inputShapeSize);

    AscendC::GmFree((void *)gradOutput);
    AscendC::GmFree((void *)self);
    AscendC::GmFree((void *)out);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    system("cd ./hard_swish_grad_v2_data/ && python3 compare_data.py 'float32'");
}

TEST_F(hard_swish_grad_v2_test, test_hard_swish_grad_v2_float16_1)
{
    system(
        "cp -rf "
        "../../../../activation/hard_swish_grad_v2/tests/ut/op_kernel/hard_swish_grad_v2_data ./");
    system("chmod -R 755 ./hard_swish_grad_v2_data/");
    system("cd ./hard_swish_grad_v2_data/ && python3 gen_data.py '(2, 4)' 'float16'");
    size_t M = 2;
    size_t N = 4;
    size_t inputShapeSize = M * N * sizeof(float);

    uint8_t *gradOutput = (uint8_t *)AscendC::GmAlloc(inputShapeSize);
    uint8_t *self = (uint8_t *)AscendC::GmAlloc(inputShapeSize);
    uint8_t *out = (uint8_t *)AscendC::GmAlloc(inputShapeSize);

    uint64_t tilingKey = 0;
    uint32_t blockDim = 1;
    size_t workspaceFileSize = 16781184;
    size_t tilingDataSize = sizeof(HardSwishGradV2TilingData);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceFileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);

    std::string gradOutputFileName = "./hard_swish_grad_v2_data/float16_grad_output.bin";
    std::string selfFileName = "./hard_swish_grad_v2_data/float16_self.bin";

    ReadFile(gradOutputFileName, inputShapeSize, gradOutput, inputShapeSize);
    ReadFile(selfFileName, inputShapeSize, self, inputShapeSize);

    HardSwishGradV2TilingData* tilingDatafromBin = reinterpret_cast<HardSwishGradV2TilingData*>(tiling);
    tilingDatafromBin->elementNum = M * N ;
    tilingDatafromBin->needCoreNum = 1;
    tilingDatafromBin->ubSize = 192 * 1024;
    tilingDatafromBin->elementNumEachCore = 8 * 1024;

    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(hard_swish_grad_v2, blockDim, gradOutput, self, out, workspace, (uint8_t*)tilingDatafromBin);
    std::string outFileName = "./hard_swish_grad_v2_data/float16_npuout.bin";
    WriteFile(outFileName, out, inputShapeSize);

    AscendC::GmFree((void *)gradOutput);
    AscendC::GmFree((void *)self);
    AscendC::GmFree((void *)out);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    system("cd ./hard_swish_grad_v2_data/ && python3 compare_data.py 'float16'");
}

TEST_F(hard_swish_grad_v2_test, test_hard_swish_grad_v2_bfloat16_t_2)
{
    system(
        "cp -rf "
        "../../../../activation/hard_swish_grad_v2/tests/ut/op_kernel/hard_swish_grad_v2_data ./");
    system("chmod -R 755 ./hard_swish_grad_v2_data/");
    system("cd ./hard_swish_grad_v2_data/ && python3 gen_data.py '(2, 4)' 'bfloat16_t'");
    size_t M = 2;
    size_t N = 4;
    size_t inputShapeSize = M * N * sizeof(float);

    uint8_t *gradOutput = (uint8_t *)AscendC::GmAlloc(inputShapeSize);
    uint8_t *self = (uint8_t *)AscendC::GmAlloc(inputShapeSize);
    uint8_t *out = (uint8_t *)AscendC::GmAlloc(inputShapeSize);

    uint64_t tilingKey = 0;
    uint32_t blockDim = 1;
    size_t workspaceFileSize = 16781184;
    size_t tilingDataSize = sizeof(HardSwishGradV2TilingData);
    uint8_t *workspace = (uint8_t *)AscendC::GmAlloc(workspaceFileSize);
    uint8_t *tiling = (uint8_t *)AscendC::GmAlloc(tilingDataSize);

    std::string gradOutputFileName = "./hard_swish_grad_v2_data/bfloat16_t_grad_output.bin";
    std::string selfFileName = "./hard_swish_grad_v2_data/bfloat16_t_self.bin";

    ReadFile(gradOutputFileName, inputShapeSize, gradOutput, inputShapeSize);
    ReadFile(selfFileName, inputShapeSize, self, inputShapeSize);

    HardSwishGradV2TilingData* tilingDatafromBin = reinterpret_cast<HardSwishGradV2TilingData*>(tiling);
    tilingDatafromBin->elementNum = M * N ;
    tilingDatafromBin->needCoreNum = 1;
    tilingDatafromBin->ubSize = 192 * 1024;
    tilingDatafromBin->elementNumEachCore = 8 * 1024;

    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(hard_swish_grad_v2, blockDim, gradOutput, self, out, workspace, (uint8_t*)tilingDatafromBin);
    std::string outFileName = "./hard_swish_grad_v2_data/bfloat16_t_npuout.bin";
    WriteFile(outFileName, out, inputShapeSize);

    AscendC::GmFree((void *)gradOutput);
    AscendC::GmFree((void *)self);
    AscendC::GmFree((void *)out);
    AscendC::GmFree((void *)workspace);
    AscendC::GmFree((void *)tiling);
    system("cd ./hard_swish_grad_v2_data/ && python3 compare_data.py 'bfloat16_t'");
}