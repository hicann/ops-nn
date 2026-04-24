/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_mish.cpp
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
#include "tiling_case_executor.h"
#include "../op_host/mish_tiling.h"

using namespace std;

extern "C" __global__ __aicore__ void mish(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling);

class MishTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "mish_test SetUp" << std::endl;
        const string cmd = "cp -rf " + dataPath + " ./";
        system(cmd.c_str());
        system("chmod -R 755 ./mish_data/");
    }
    static void TearDownTestCase()
    {
        std::cout << "mish_test TearDown" << std::endl;
    }

private:
    const static std::string rootPath;
    const static std::string dataPath;
};

const std::string MishTest::rootPath = "../../../../";
const std::string MishTest::dataPath = rootPath + "math/mish/tests/ut/op_kernel/mish_data";

template <typename T1, typename T2>
inline T1 CeilAlign(T1 a, T2 b)
{
    return (a + b - 1) / b * b;
}

TEST_F(MishTest, test_case_float16_1)
{
    optiling::MishCompileInfo compileInfo = {64, 262144, false};
    gert::TilingContextPara tilingContextPara(
        "Mish",
        {
            {{{128, 64}, {128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        {
            {{{128, 64}, {128, 64}}, ge::DT_FLOAT16, ge::FORMAT_ND},
        },
        &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    system("cd ./mish_data/ && python3 gen_data.py '(128, 64)' 'float16'");
    uint32_t dataCount = 128 * 64;
    size_t inputByteSize = dataCount * sizeof(half);
    std::string fileName = "./mish_data/float16_input_t_mish.bin";
    ;
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));
    ReadFile(fileName, inputByteSize, x, inputByteSize);
    size_t outputByteSize = dataCount * sizeof(half);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(CeilAlign(outputByteSize, 32));

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(mish, tilingInfo.blockNum, x, y, workspace, tiling);

    fileName = "./mish_data/float16_output_t_mish.bin";
    WriteFile(fileName, y, outputByteSize);

    AscendC::GmFree((void*)(x));
    AscendC::GmFree((void*)(y));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./mish_data/ && python3 compare_data.py 'float16'");
}

TEST_F(MishTest, test_case_float32_1)
{
    optiling::MishCompileInfo compileInfo = {64, 262144, false};
    gert::TilingContextPara tilingContextPara(
        "Mish",
        {
            {{{256, 33}, {256, 33}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        {
            {{{256, 33}, {256, 33}}, ge::DT_FLOAT, ge::FORMAT_ND},
        },
        &compileInfo);
    TilingInfo tilingInfo;
    auto tilingRet = ExecuteTiling(tilingContextPara, tilingInfo);
    EXPECT_EQ(tilingRet, true);

    system("cd ./mish_data/ && python3 gen_data.py '(256, 33)' 'float32'");
    uint32_t dataCount = 256 * 33;
    size_t inputByteSize = dataCount * sizeof(float);
    std::string fileName = "./mish_data/float32_input_t_mish.bin";
    ;
    uint8_t* x = (uint8_t*)AscendC::GmAlloc(CeilAlign(inputByteSize, 32));
    ReadFile(fileName, inputByteSize, x, inputByteSize);
    size_t outputByteSize = dataCount * sizeof(float);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(CeilAlign(outputByteSize, 32));

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(tilingInfo.workspaceSizes[0]);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingInfo.tilingDataSize);
    std::memcpy(tiling, tilingInfo.tilingData.get(), tilingInfo.tilingDataSize);
    ICPU_SET_TILING_KEY(tilingInfo.tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(mish, tilingInfo.blockNum, x, y, workspace, tiling);

    fileName = "./mish_data/float32_output_t_mish.bin";
    WriteFile(fileName, y, outputByteSize);

    AscendC::GmFree((void*)(x));
    AscendC::GmFree((void*)(y));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./mish_data/ && python3 compare_data.py 'float32'");
}
