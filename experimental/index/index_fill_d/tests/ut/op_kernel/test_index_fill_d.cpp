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
 * \file test_index_fill_d.cpp
 * \brief
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"

#include "../../../op_kernel/index_fill_d.cpp"

using namespace std;

class IndexFillDTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "index_fill_d_test SetUp" << std::endl;
        const string cmd = "cp -rf " + dataPath + " ./";
        system(cmd.c_str());
        system("chmod -R 755 ./index_fill_d_data/");
    }
    static void TearDownTestCase() { std::cout << "index_fill_d_test TearDown" << std::endl; }

private:
    const static std::string rootPath;
    const static std::string dataPath;
};

const std::string IndexFillDTest::rootPath = "../../../../experimental/";
const std::string IndexFillDTest::dataPath = rootPath + "index/index_fill_d/tests/ut/op_kernel/index_fill_d_data";

template <typename T1, typename T2>
inline T1 CeilAlign(T1 a, T2 b)
{
    return (a + b - 1) / b * b;
}

TEST_F(IndexFillDTest, test_case_float32_1)
{
    uint32_t blockDim = 1;
    system("cd ./index_fill_d_data/ && python3 gen_data.py '(4, 2)' 'float32'");
    uint32_t dataCount = 8;
    size_t inputByteSize = dataCount * sizeof(float);
    std::string x1_fileName = "./index_fill_d_data/float32_input1_t_index_fill_d.bin";
    std::string x2_fileName = "./index_fill_d_data/float32_input2_t_index_fill_d.bin";
    std::string x3_fileName = "./index_fill_d_data/float32_input3_t_index_fill_d.bin";

    uint8_t* x1 = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 256));
    ReadFile(x1_fileName, inputByteSize, x1, inputByteSize);
    uint8_t* x2 = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 256));
    ReadFile(x2_fileName, inputByteSize, x2, inputByteSize);
    uint8_t* x3 = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 256));
    ReadFile(x3_fileName, inputByteSize, x3, inputByteSize);
    size_t outputByteSize = dataCount * sizeof(float);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(CeilAlign(outputByteSize, 256));

    size_t workspaceSize = 16 * 1024 * 1024;
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(sizeof(IndexFillDTilingData));

    IndexFillDTilingData* tilingData = reinterpret_cast<IndexFillDTilingData*>(tiling);

    tilingData->smallCoreDataNum = 64;
    tilingData->bigCoreDataNum = 128;
    tilingData->tileDataNum = 4864;
    tilingData->smallTailDataNum = 64;
    tilingData->bigTailDataNum = 128;
    tilingData->finalSmallTileNum = 1;
    tilingData->finalBigTileNum = 1;
    tilingData->tailBlockNum = 0;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto func = index_fill_d<ELEMENTWISE_TPL_SCH_MODE_1>;
    ICPU_RUN_KF(func, blockDim, x1, x2, x3, y, workspace, (uint8_t*)(tilingData));

    std::string fileName = "./index_fill_d_data/float32_output_t_index_fill_d.bin";
    WriteFile(fileName, y, outputByteSize);

    AscendC::GmFree((void*)(x1));
    AscendC::GmFree((void*)(x2));
    AscendC::GmFree((void*)(x3));
    AscendC::GmFree((void*)(y));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./index_fill_d_data/ && python3 compare_data.py 'float32'");
}