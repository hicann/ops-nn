/*
 * Copyright (c) 2025 联通（广东）产业互联网有限公司.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include <type_traits>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "silu_mul_tiling_def.h"
#include "data_utils.h"

using namespace std;

#ifndef DTYPE_X
#define DTYPE_X float
#endif

extern "C" __global__ __aicore__ void silu_mul(GM_ADDR input, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling);

class silu_mul_test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        cout << "silu_mul SetUp\n" << endl;
    }
    static void TearDownTestCase()
    {
        cout << "silu_mul TearDown\n" << endl;
    }
};

// 【新增】简单的类型映射辅助结构体
template <typename T>
struct DataTypeName {
    static constexpr const char* val = "unknown";
};
template <>
struct DataTypeName<float> {
    static constexpr const char* val = "float32";
};
template <>
struct DataTypeName<half> {
    static constexpr const char* val = "float16";
};
#if !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
template <>
struct DataTypeName<bfloat16_t> {
    static constexpr const char* val = "bfloat16_t";
};
#endif

TEST_F(silu_mul_test, test_silu_mul_dynamic)
{
    const char* dtypeStr = DataTypeName<DTYPE_X>::val;
    std::cout << ">>> Current Test Type: " << dtypeStr << std::endl;

    system(
        "cp -rf "
        "../../../../activation/silu_mul/tests/ut/op_kernel/silu_mul_data ./");
    system("chmod -R 755 ./silu_mul_data/");

    std::string genCmd = std::string("cd ./silu_mul_data/ && python3 gen_data.py '(2, 4)' '(2, 2)' '") + dtypeStr + "'";
    system(genCmd.c_str());

    size_t M = 2;
    size_t N = 4;
    size_t D = N / 2;

    size_t xFileSize = M * N * sizeof(DTYPE_X);
    size_t yFileSize = M * D * sizeof(DTYPE_X);

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xFileSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yFileSize);

    uint64_t tilingKey = 0;
    uint32_t blockDim = 1;
    size_t workspaceFileSize = 16781184;
    size_t tilingDataSize = sizeof(SiluMulTilingData);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceFileSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    std::string fileName = std::string("./silu_mul_data/") + dtypeStr + "_input_silu_mul.bin";

    ReadFile(fileName, xFileSize, x, xFileSize);

    SiluMulTilingData* tilingDatafromBin = reinterpret_cast<SiluMulTilingData*>(tiling);
    tilingDatafromBin->lastDimSize = 4;
    tilingDatafromBin->batchSize = 2;
    tilingDatafromBin->PPMaxCalNum = 5888;
    tilingDatafromBin->needCoreNum = 1;

    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(silu_mul, blockDim, x, y, workspace, (uint8_t*)tilingDatafromBin);

    fileName = std::string("./silu_mul_data/") + dtypeStr + "_output_silu_mul.bin";
    WriteFile(fileName, y, yFileSize);

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    std::string compareCmd = std::string("cd ./silu_mul_data/ && python3 compare_data.py '") + dtypeStr + "'";
    system(compareCmd.c_str());
}