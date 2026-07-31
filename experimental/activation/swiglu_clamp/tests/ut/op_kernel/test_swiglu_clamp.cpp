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
 * \file test_swiglu_clamp.cpp
 * \brief SwigluClamp kernel UT (tikicpulib CPU simulation). Template follows sigmoid.
 *        x [128, 32] (2N=32, N=16) fp32 -> y [128, 16], limit=7.0, single core.
 *        Verifies the fp32 path: the if constexpr(std::is_same_v<DATA_T,float>) branch
 *        (no Cast float->float) and bufferCoefficient=44 tiling. bf16/fp16 (else branch,
 *        Cast fp32 -> compute -> Cast back) are covered end-to-end by ACLNN examples.
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
#include "../../../op_kernel/swiglu_clamp.cpp"
#include "../../../op_kernel/swiglu_clamp_tiling_data.h"
#include <cstdint>

using namespace std;

class SwigluClampKernelTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "swiglu_clamp kernel UT SetUp" << std::endl;
        const string cmd = "cp -rf " + dataPath + " ./";
        system(cmd.c_str());
        system("chmod -R 755 ./swiglu_clamp_data/");
    }
    static void TearDownTestCase() { std::cout << "swiglu_clamp kernel UT TearDown" << std::endl; }

private:
    const static std::string rootPath;
    const static std::string dataPath;
};

const std::string SwigluClampKernelTest::rootPath = "../../../../";
const std::string
    SwigluClampKernelTest::dataPath = rootPath +
                                      "experimental/activation/swiglu_clamp/tests/ut/op_kernel/swiglu_clamp_data";

template <typename T1, typename T2>
inline T1 CeilAlign(T1 a, T2 b)
{
    return (a + b - 1) / b * b;
}

// x [128, 32] (2N=32, N=16) fp32 -> y [128, 16]. Single core, tileM=128 (one tile).
TEST_F(SwigluClampKernelTest, test_case_float32_1)
{
    const int64_t M = 128; // rows
    const int64_t N = 16;  // half of last dim
    const float limit = 7.0f;
    size_t tiling_data_size = sizeof(SwigluClampTilingData);

    system("cd ./swiglu_clamp_data/ && python3 gen_data.py '(128, 32)' 'float32' '7.0'");
    uint32_t inCount = M * 2 * N; // x [128, 32]
    uint32_t outCount = M * N;    // y [128, 16]
    size_t inByteSize = inCount * sizeof(float);
    size_t outByteSize = outCount * sizeof(float);
    std::string fileName = "./swiglu_clamp_data/float32_input_t_swiglu_clamp.bin";

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(CeilAlign(inByteSize, 32));
    ReadFile(fileName, inByteSize, x, inByteSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(CeilAlign(outByteSize, 32));

    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(1024 * 1024 * 16);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tiling_data_size);

    // hand-fill tiling (single core: formerNum=0, tailNum=1, coreRows=tailLength=M, tileM=tileLength=M)
    SwigluClampTilingData* td = reinterpret_cast<SwigluClampTilingData*>(tiling);
    td->totalLength = M; // = row count
    td->N = N;
    td->formerNum = 0; // no former cores
    td->formerLength = 0;
    td->tailNum = 1; // single tail core takes all rows
    td->tailLength = M;
    td->tileLength = M; // tileM = M -> one tile processes all rows (UB fits at CPU sim)
    td->limit = limit;
    ICPU_SET_TILING_KEY(0);

    auto swigluClampKernel = [](GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
        ::swiglu_clamp<0>(x, y, workspace, tiling);
    };

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(swigluClampKernel, 1, x, y, workspace, (uint8_t*)(td));

    fileName = "./swiglu_clamp_data/float32_output_t_swiglu_clamp.bin";
    WriteFile(fileName, y, outByteSize);

    AscendC::GmFree((void*)(x));
    AscendC::GmFree((void*)(y));
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    system("cd ./swiglu_clamp_data/ && python3 compare_data.py 'float32'");
}
