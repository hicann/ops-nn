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
 * \file test_sigmoid.cpp
 * \brief
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
#include "../../../op_kernel/sigmoid.cpp"
#include "../../../op_kernel/sigmoid_tiling_data.h"
#include <cstdint>


using namespace std;


class SigmoidTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "sigmoid_test SetUp" << std::endl;
        const string cmd = "cp -rf " + dataPath + " ./";
        system(cmd.c_str());
        system("chmod -R 755 ./sigmoid_data/");
    }
    static void TearDownTestCase()
    {
        std::cout << "sigmoid_test TearDown" << std::endl;
    }

private:
    const static std::string rootPath;
    const static std::string dataPath;
};

const std::string SigmoidTest::rootPath = "../../../../";
const std::string SigmoidTest::dataPath = rootPath + "experimental/activation/sigmoid/tests/ut/op_kernel/sigmoid_data";

template <typename T1, typename T2>
inline T1 CeilAlign(T1 a, T2 b)
{
    return (a + b - 1) / b * b;
}

TEST_F(SigmoidTest, test_case_float16_1)
{

    size_t tiling_data_size = sizeof(SigmoidTilingData);
    system("cd ./sigmoid_data/ && python3 gen_data.py '(128, 64)' 'float16'");
    uint32_t dataCount = 128 * 64;
    size_t inputByteSize = dataCount * sizeof(half);
    std::string fileName = "./sigmoid_data/float16_input_t_sigmoid.bin";

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));
    ReadFile(fileName, inputByteSize, x, inputByteSize);
    size_t outputByteSize = dataCount * sizeof(half);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(CeilAlign(outputByteSize, 32));

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);
    
    
    SigmoidTilingData* tilingDatafromBin = reinterpret_cast<SigmoidTilingData*>(tiling);
    tilingDatafromBin->bigCoreDataNum = 8208;      // 每个大核处理的总数据量（通常包含对齐预留）
    tilingDatafromBin->smallCoreDataNum = 8192;    // 每个小核处理的总数据量
    tilingDatafromBin->finalBigTileNum = 1;       // 大核需要循环计算的次数
    tilingDatafromBin->finalSmallTileNum = 1;     // 小核需要循环计算的次数
    tilingDatafromBin->tileDataNum = 12288;       // 每一块（Tile）存入本地内存的大小（字节或个数）
    tilingDatafromBin->bigTailDataNum = 8208;      // 大核最后一块处理的数据量
    tilingDatafromBin->smallTailDataNum = 8192;    // 小核最后一块处理的数据量
    tilingDatafromBin->tailBlockNum = 0;          // 尾部剩余的核数（如果不整除）
    ICPU_SET_TILING_KEY(2);

    auto SigmoidleKernel = [](GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
        ::sigmoid<0>(x, y, workspace, tiling);
    };


    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(SigmoidleKernel, 1, x, y, workspace, (uint8_t *)(tilingDatafromBin)); // 根据实际设置核数，这里设置为8仅用于测试是否能测试kernel UT

    fileName = "./sigmoid_data/float16_output_t_sigmoid.bin";
    WriteFile(fileName, y, outputByteSize);

    AscendC::GmFree((void*)(x));
    AscendC::GmFree((void*)(y));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./sigmoid_data/ && python3 compare_data.py 'float16'");
}
