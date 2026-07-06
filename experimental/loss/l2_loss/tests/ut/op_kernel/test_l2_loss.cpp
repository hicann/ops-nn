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
 * \file test_l2_loss.cpp
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

#include "../../../op_kernel/l2_loss.cpp"

using namespace std;

extern "C" __global__ __aicore__ void l2_loss(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

class L2LossTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "l2_loss_test SetUp" << std::endl;
        const string cmd = "cp -rf " + dataPath + " ./";
        system(cmd.c_str());
        system("chmod -R 755 ./l2_loss_data/");
    }
    static void TearDownTestCase() { std::cout << "l2_loss_test TearDown" << std::endl; }

private:
    const static std::string rootPath;
    const static std::string dataPath;
};

const std::string L2LossTest::rootPath = "../../../../";
const std::string L2LossTest::dataPath = rootPath + "experimental/loss/l2_loss/tests/ut/op_kernel/l2_loss_data";

template <typename T1, typename T2>
inline T1 CeilAlign(T1 a, T2 b)
{
    return (a + b - 1) / b * b;
}

TEST_F(L2LossTest, test_case_float16_1)
{
    uint32_t blockDim = 1;
    uint32_t dataCount = 128 * 64;
    system("cd ./l2_loss_data/ && python3 gen_data.py '(128, 64)' 'float16'");

    size_t inputByteSize = dataCount * sizeof(half);
    size_t outputByteSize = 1 * sizeof(half);
    std::string inputFileName = "./l2_loss_data/float16_input_t_l2_loss.bin";

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));
    ReadFile(inputFileName, inputByteSize, x, inputByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(CeilAlign(outputByteSize, 32));

    size_t workspaceSize = 32 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(L2LossTilingData));

    L2LossTilingData* tilingData = reinterpret_cast<L2LossTilingData*>(tiling);
    tilingData->smallCoreDataNum = dataCount;
    tilingData->bigCoreDataNum = dataCount;
    tilingData->finalBigTileNum = 1;
    tilingData->finalSmallTileNum = 1;
    tilingData->tileDataNum = dataCount;
    tilingData->smallTailDataNum = dataCount;
    tilingData->bigTailDataNum = dataCount;
    tilingData->tailBlockNum = 0;
    tilingData->inputNum = dataCount;
    tilingData->blockNum = blockDim;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto func = l2_loss<ELEMENTWISE_TPL_SCH_MODE_0>;
    ICPU_RUN_KF(func, blockDim, x, y, workspace, (uint8_t*)(tilingData));

    std::string outputFileName = "./l2_loss_data/float16_output_t_l2_loss.bin";
    WriteFile(outputFileName, y, outputByteSize);

    AscendC::GmFree((void*)(x));
    AscendC::GmFree((void*)(y));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./l2_loss_data/ && python3 compare_data.py 'float16'");
}

TEST_F(L2LossTest, test_case_float32_1)
{
    uint32_t blockDim = 1;
    uint32_t dataCount = 256 * 33;
    system("cd ./l2_loss_data/ && python3 gen_data.py '(256, 33)' 'float32'");

    size_t inputByteSize = dataCount * sizeof(float);
    size_t outputByteSize = 1 * sizeof(float);
    std::string inputFileName = "./l2_loss_data/float32_input_t_l2_loss.bin";

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));
    ReadFile(inputFileName, inputByteSize, x, inputByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(CeilAlign(outputByteSize, 32));

    size_t workspaceSize = 32 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(L2LossTilingData));

    L2LossTilingData* tilingData = reinterpret_cast<L2LossTilingData*>(tiling);
    tilingData->smallCoreDataNum = dataCount;
    tilingData->bigCoreDataNum = dataCount;
    tilingData->finalBigTileNum = 1;
    tilingData->finalSmallTileNum = 1;
    tilingData->tileDataNum = dataCount;
    tilingData->smallTailDataNum = dataCount;
    tilingData->bigTailDataNum = dataCount;
    tilingData->tailBlockNum = 0;
    tilingData->inputNum = dataCount;
    tilingData->blockNum = blockDim;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto func = l2_loss<ELEMENTWISE_TPL_SCH_MODE_0>;
    ICPU_RUN_KF(func, blockDim, x, y, workspace, (uint8_t*)(tilingData));

    std::string outputFileName = "./l2_loss_data/float32_output_t_l2_loss.bin";
    WriteFile(outputFileName, y, outputByteSize);

    AscendC::GmFree((void*)(x));
    AscendC::GmFree((void*)(y));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./l2_loss_data/ && python3 compare_data.py 'float32'");
}

TEST_F(L2LossTest, test_case_bfloat16_1)
{
    uint32_t blockDim = 1;
    uint32_t dataCount = 128 * 64;
    system("cd ./l2_loss_data/ && python3 gen_data.py '(128, 64)' 'bfloat16'");

    size_t inputByteSize = dataCount * sizeof(uint16_t); // bfloat16 is 2 bytes
    size_t outputByteSize = 1 * sizeof(uint16_t);
    std::string inputFileName = "./l2_loss_data/bfloat16_input_t_l2_loss.bin";

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));
    ReadFile(inputFileName, inputByteSize, x, inputByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(CeilAlign(outputByteSize, 32));

    size_t workspaceSize = 32 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(L2LossTilingData));

    L2LossTilingData* tilingData = reinterpret_cast<L2LossTilingData*>(tiling);
    tilingData->smallCoreDataNum = dataCount;
    tilingData->bigCoreDataNum = dataCount;
    tilingData->finalBigTileNum = 1;
    tilingData->finalSmallTileNum = 1;
    tilingData->tileDataNum = dataCount;
    tilingData->smallTailDataNum = dataCount;
    tilingData->bigTailDataNum = dataCount;
    tilingData->tailBlockNum = 0;
    tilingData->inputNum = dataCount;
    tilingData->blockNum = blockDim;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto func = l2_loss<ELEMENTWISE_TPL_SCH_MODE_0>;
    ICPU_RUN_KF(func, blockDim, x, y, workspace, (uint8_t*)(tilingData));

    std::string outputFileName = "./l2_loss_data/bfloat16_output_t_l2_loss.bin";
    WriteFile(outputFileName, y, outputByteSize);

    AscendC::GmFree((void*)(x));
    AscendC::GmFree((void*)(y));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./l2_loss_data/ && python3 compare_data.py 'bfloat16'");
}
