/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the 'License').
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file test_silu_mul_apt.cpp
 * \brief Kernel UT for SiluMul Arch35 (Ascend 950)
 *
 * CPU simulation (tikicpulib) of the ascend950 RegBase kernel entry.
 * Uses SiluMulArch35TilingData (the arch35 plain POD struct) and the apt
 * kernel symbol. dtype is selected by -DDTYPE_X via CMakeLists (one test
 * function compiled per dtype: fp32 / fp16).
 */

#include <array>
#include <vector>
#include <iostream>
#include <string>
#include <cstdint>
#include <type_traits>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"
#include "kernel_ut_data_helper.h"
#include "kernel_ut_data_executor.h"

// The arch35 kernel entry (op_kernel/arch35/silu_mul.cpp) is conditionally
// compiled: in UT mode (__CCE_UT_TEST__ / KERNELUT) it reads the tiling struct
// via memcpy instead of the device-only GET_TILING_DATA_WITH_STRUCT macro. We
// do NOT include arch35/silu_mul.cpp here: AddOpTestCase compiles it as a
// separate OBJECT library, and we link against the `silu_mul` symbol via the
// extern declaration below.
#include "../../../../op_kernel/arch35/silu_mul_tiling_struct.h"

using namespace std;

#ifndef DTYPE_X
#define DTYPE_X float
#endif

extern "C" __global__ __aicore__ void silu_mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling);

class silu_mul_apt_test : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "silu_mul_apt_test SetUp\n" << endl; }
    static void TearDownTestCase()
    {
        cout << "silu_mul_apt_test TearDown\n" << endl;
        kernel_ut::CleanGeneratedBinFiles("./silu_mul_data");
    }
};

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
template <>
struct DataTypeName<bfloat16_t> {
    static constexpr const char* val = "bfloat16_t";
};

// silu_mul: z = silu(x) * y = x * sigmoid(x) * y
TEST_F(silu_mul_apt_test, test_silu_mul_dynamic)
{
    const char* dtypeStr = DataTypeName<DTYPE_X>::val;
    std::cout << ">>> Current Test Type: " << dtypeStr << std::endl;

    kernel_ut::SetupTestEnvironment("activation/silu_mul/tests/ut/op_kernel/silu_mul_data", "silu_mul_data");
    kernel_ut::RunGenData("./silu_mul_data", {"'(2, 4)'", "'(2, 2)'", dtypeStr});
    std::string path = kernel_ut::GetTestWorkDir();

    size_t M = 2;
    size_t N = 4;
    size_t D = N;

    size_t xFileSize = M * N * sizeof(DTYPE_X);
    size_t yFileSize = M * D * sizeof(DTYPE_X);
    size_t zFileSize = xFileSize;

    uint8_t* x = (uint8_t*)AscendC::GmAlloc(xFileSize);
    uint8_t* y = (uint8_t*)AscendC::GmAlloc(yFileSize);
    uint8_t* z = (uint8_t*)AscendC::GmAlloc(zFileSize);

    uint64_t tilingKey = 0;
    uint32_t blockDim = 1;
    size_t workspaceFileSize = 16 * 1024 * 1024;
    size_t tilingDataSize = sizeof(SiluMulArch35TilingData);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceFileSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingDataSize);

    std::string xFileName = path + "/silu_mul_data/" + dtypeStr + "_input_x.bin";
    std::string yFileName = path + "/silu_mul_data/" + dtypeStr + "_input_y.bin";

    ReadFile(xFileName, xFileSize, x, xFileSize);
    ReadFile(yFileName, yFileSize, y, yFileSize);

    SiluMulArch35TilingData* tilingData = reinterpret_cast<SiluMulArch35TilingData*>(tiling);
    tilingData->lastDimSize = 4;
    tilingData->batchSize = 2;
    tilingData->PPMaxCalNum = 5888;
    tilingData->needCoreNum = 1;
    tilingData->maxUbSize = 184 * 1024;

    auto KernelSiluMul = [](GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
        ::silu_mul(x, y, z, workspace, tiling);
    };

    ICPU_SET_TILING_KEY(tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    ICPU_RUN_KF(KernelSiluMul, blockDim, x, y, z, workspace, (uint8_t*)tilingData);

    std::string zFileName = path + "/silu_mul_data/" + dtypeStr + "_output_silu_mul.bin";
    WriteFile(zFileName, z, zFileSize);

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)y);
    AscendC::GmFree((void*)z);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    kernel_ut::RunCompareData("./silu_mul_data", {dtypeStr});
}
