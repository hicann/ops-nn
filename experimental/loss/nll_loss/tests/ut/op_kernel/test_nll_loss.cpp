/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_nll_loss.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#endif

#include "../../../op_kernel/nll_loss.cpp"

using namespace std;

extern "C" __global__ __aicore__ void nll_loss(GM_ADDR x, GM_ADDR target, GM_ADDR weight, GM_ADDR y,
                                               GM_ADDR total_weight, GM_ADDR workspace, GM_ADDR tiling);

class NllLossTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "nll_loss_test SetUp" << std::endl;
        const string cmd = "cp -rf " + dataPath + " ./";
        system(cmd.c_str());
        system("chmod -R 755 ./nll_loss_data/");
    }
    static void TearDownTestCase() { std::cout << "nll_loss_test TearDown" << std::endl; }

private:
    const static std::string rootPath;
    const static std::string dataPath;
};

const std::string NllLossTest::rootPath = "../../../../";
const std::string NllLossTest::dataPath = rootPath + "experimental/loss/nll_loss/tests/ut/op_kernel/nll_loss_data";

template <typename T1, typename T2>
inline T1 CeilAlign(T1 a, T2 b)
{
    return (a + b - 1) / b * b;
}

TEST_F(NllLossTest, test_case_float32_mean)
{
    uint32_t blockDim = 1;
    uint32_t rowNum = 4;
    uint32_t classNum = 8;
    system("cd ./nll_loss_data/ && python3 gen_data.py '(4, 8)' 'float32'");

    size_t xByteSize = rowNum * classNum * sizeof(float);
    size_t targetByteSize = rowNum * sizeof(int32_t);
    size_t weightByteSize = classNum * sizeof(float);
    size_t yByteSize = 1 * sizeof(float);
    size_t twByteSize = 1 * sizeof(float);

    std::string xFileName = "./nll_loss_data/float32_input_x_nll_loss.bin";
    std::string targetFileName = "./nll_loss_data/int32_input_target_nll_loss.bin";

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(CeilAlign(xByteSize, 32));
    uint8_t* target = (uint8_t*)AscendC::GmAlloc(CeilAlign(targetByteSize, 32));
    uint8_t* weight = (uint8_t*)AscendC::GmAlloc(CeilAlign(weightByteSize, 32));
    ReadFile(xFileName, xByteSize, x, xByteSize);
    ReadFile(targetFileName, targetByteSize, target, targetByteSize);

    uint8_t* y = (uint8_t*)AscendC::GmAlloc(CeilAlign(yByteSize, 32));
    uint8_t* totalWeight = (uint8_t*)AscendC::GmAlloc(CeilAlign(twByteSize, 32));

    size_t workspaceSize = 32 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(NllLossTilingData));

    NllLossTilingData* tilingData = reinterpret_cast<NllLossTilingData*>(tiling);
    tilingData->rowNum = rowNum;
    tilingData->classNum = classNum;
    tilingData->reduction = 1; // mean
    tilingData->ignoreIndex = -100;
    tilingData->hasWeight = 0;
    tilingData->targetIsInt64 = 0;
    tilingData->usedCoreNum = 1;
    tilingData->rowsPerCore = rowNum;
    tilingData->tileRows = rowNum;
    tilingData->useVector = 0;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto func = nll_loss<NLLLOSS_TPL_SCH_MODE_1>;
    ICPU_RUN_KF(func, blockDim, x, target, weight, y, totalWeight, workspace, (uint8_t*)(tilingData));

    std::string yFileName = "./nll_loss_data/float32_output_y_nll_loss.bin";
    WriteFile(yFileName, y, yByteSize);

    AscendC::GmFree((void*)(x));
    AscendC::GmFree((void*)(target));
    AscendC::GmFree((void*)(weight));
    AscendC::GmFree((void*)(y));
    AscendC::GmFree((void*)(totalWeight));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./nll_loss_data/ && python3 compare_data.py 'float32'");
}
