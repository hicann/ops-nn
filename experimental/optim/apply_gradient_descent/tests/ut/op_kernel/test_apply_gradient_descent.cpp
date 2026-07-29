/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
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
#include <unistd.h>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "apply_gradient_descent_tiling_def.h"
#include "data_utils.h"

using namespace std;

// Kernel entry: var_out = var - alpha * delta. Dispatch (tiling key): 1=fp16, 2=fp32, 3=bf16.
extern "C" __global__ __aicore__ void apply_gradient_descent(GM_ADDR var, GM_ADDR alpha, GM_ADDR delta, GM_ADDR var_out,
                                                             GM_ADDR workspace, GM_ADDR tiling);

class apply_gradient_descent_test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "apply_gradient_descent_test SetUp\n" << std::endl; }
    static void TearDownTestCase() { std::cout << "apply_gradient_descent_test TearDown\n" << std::endl; }
};

static constexpr int64_t TOTAL = 30 * 4 * 2; // 240 elements
static constexpr size_t ALPHA_BUF = 32;      // one 32B block, holds the single alpha element

// Locate the checked-in data/scripts dir by walking up from the runtime working directory
// (the UT executable's cwd lives somewhere under the repo build tree), so the copy does not
// depend on a fixed relative depth.
static std::string DataSrcDir()
{
    const std::string rel = "experimental/optim/apply_gradient_descent/tests/ut/op_kernel/apply_gradient_descent_data";
    std::string prefix;
    for (int i = 0; i < 12; i++) {
        std::string cand = prefix + rel;
        if (access(cand.c_str(), F_OK) == 0) {
            return cand;
        }
        prefix += "../";
    }
    return rel;
}

static void RunKernelCase(const char* dtypeStr, uint64_t tilingKey, size_t elemBytes)
{
    std::string cpCmd = std::string("cp -rf ") + DataSrcDir() + " ./";
    system(cpCmd.c_str());
    system("chmod -R 755 ./apply_gradient_descent_data/");
    std::string genCmd = std::string("cd ./apply_gradient_descent_data/ && python3 gen_data.py ") + dtypeStr;
    system(genCmd.c_str());

    AscendC::SetKernelMode(KernelMode::AIV_MODE);

    size_t ioSize = static_cast<size_t>(TOTAL) * elemBytes;
    size_t workspaceSize = 16 * 1024 * 1024 + 512;
    size_t tilingSize = sizeof(ApplyGradientDescentTilingData);
    uint32_t blockDim = 1;

    uint8_t* var = (uint8_t*)AscendC::GmAlloc(ioSize);
    uint8_t* alpha = (uint8_t*)AscendC::GmAlloc(ALPHA_BUF);
    uint8_t* delta = (uint8_t*)AscendC::GmAlloc(ioSize);
    uint8_t* varOut = (uint8_t*)AscendC::GmAlloc(ioSize);
    uint8_t* workspace = (uint8_t*)AscendC::GmAlloc(workspaceSize);
    uint8_t* tiling = (uint8_t*)AscendC::GmAlloc(tilingSize);

    std::string dataDir = "./apply_gradient_descent_data/";
    ReadFile(dataDir + "var.bin", ioSize, var, ioSize);
    ReadFile(dataDir + "alpha.bin", elemBytes, alpha, elemBytes);
    ReadFile(dataDir + "delta.bin", ioSize, delta, ioSize);

    ApplyGradientDescentTilingData* td = reinterpret_cast<ApplyGradientDescentTilingData*>(tiling);
    td->totalDataCount = static_cast<uint64_t>(TOTAL);
    td->tileDataCount = 128;
    td->blockElems = 32;
    td->blocksPerCore = (static_cast<uint64_t>(TOTAL) + td->blockElems - 1) / td->blockElems; // 8
    td->needCoreNum = 1;
    td->remCoreNum = 0;
    td->reserved = 0;

    ICPU_SET_TILING_KEY(tilingKey);
    ICPU_RUN_KF(apply_gradient_descent, blockDim, var, alpha, delta, varOut, workspace, tiling);

    WriteFile(dataDir + "out_var.bin", varOut, ioSize);

    AscendC::GmFree((void*)var);
    AscendC::GmFree((void*)alpha);
    AscendC::GmFree((void*)delta);
    AscendC::GmFree((void*)varOut);
    AscendC::GmFree((void*)workspace);
    AscendC::GmFree((void*)tiling);

    std::string cmpCmd = std::string("cd ./apply_gradient_descent_data/ && python3 compare_data.py ") + dtypeStr;
    int cmpRet = system(cmpCmd.c_str());
    EXPECT_EQ(cmpRet, 0) << "precision compare failed for dtype " << dtypeStr;
}

TEST_F(apply_gradient_descent_test, test_case_float32) { RunKernelCase("float32", 2, sizeof(float)); }

TEST_F(apply_gradient_descent_test, test_case_float16) { RunKernelCase("float16", 1, sizeof(uint16_t)); }

TEST_F(apply_gradient_descent_test, test_case_bfloat16) { RunKernelCase("bfloat16", 3, sizeof(uint16_t)); }
