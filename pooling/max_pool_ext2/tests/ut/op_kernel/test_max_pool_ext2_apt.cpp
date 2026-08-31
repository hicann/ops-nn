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
 * \file test_max_pool_ext2_apt.cpp
 * \brief Kernel UT for max_pool_ext2 operator
 */

#include "../../../op_kernel/max_pool_ext2_apt.cpp"
#include "max_pool_ext2_tiling.h"
#include <cstdio>
#include <string>
#include "gtest/gtest.h"
#include "tikicpulib.h"
#include "data_utils.h"

namespace {
const char* DATA_DIR = "./max_pool_ext2_data";

bool RunCmd(const std::string& cmd) { return std::system(cmd.c_str()) == 0; }

void RunKernelCase(const std::string& genArgs, uint32_t tilingKey, int64_t hIn, int64_t wIn, int64_t hOut, int64_t wOut,
                   int64_t kH, int64_t kW, int64_t sH, int64_t sW, int64_t tPad, int64_t lPad, int64_t nDim,
                   int64_t cDim, uint32_t numBlocks, int64_t blockIdxOffset)
{
    ASSERT_TRUE(RunCmd("cd " + std::string(DATA_DIR) + " && python3 gen_data.py " + genArgs))
        << "gen_data.py failed: " << genArgs;

    const int64_t total = nDim * cDim * hOut * wOut;
    const size_t xBytes = static_cast<size_t>(nDim * cDim * hIn * wIn) * sizeof(float);
    const size_t yBytes = static_cast<size_t>(total) * sizeof(float);

    uint8_t* x = static_cast<uint8_t*>(AscendC::GmAlloc(xBytes));
    uint8_t* y = static_cast<uint8_t*>(AscendC::GmAlloc(yBytes));
    uint8_t* workspace = static_cast<uint8_t*>(AscendC::GmAlloc(32));
    uint8_t* tiling = static_cast<uint8_t*>(AscendC::GmAlloc(sizeof(MaxPoolExt2TilingData)));
    ASSERT_NE(x, nullptr);
    ASSERT_NE(y, nullptr);
    ASSERT_NE(tiling, nullptr);

    auto* td = reinterpret_cast<MaxPoolExt2TilingData*>(tiling);
    td->hInDim = hIn;
    td->wInDim = wIn;
    td->hOutDim = hOut;
    td->wOutDim = wOut;
    td->kH = kH;
    td->kW = kW;
    td->sH = sH;
    td->sW = sW;
    td->tPad = tPad;
    td->lPad = lPad;
    td->nDim = nDim;
    td->cDim = cDim;
    td->totalElements = total;
    td->needCoreNum = static_cast<int64_t>(numBlocks);
    td->blockFactor = (numBlocks > 0) ? (total + numBlocks - 1) / numBlocks : 0;
    td->blockTail = (numBlocks > 0) ? (total - td->blockFactor * (numBlocks - 1)) : 0;

    const std::string inFile = std::string(DATA_DIR) + "/float32_input_max_pool_ext2.bin";
    size_t bytesRead = 0;
    ASSERT_TRUE(ReadFile(inFile, bytesRead, x, xBytes)) << "read input failed";

    ICPU_SET_TILING_KEY(tilingKey);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    if (tilingKey == 0) {
        ICPU_RUN_KF(max_pool_ext2<0>, numBlocks, x, y, workspace, tiling);
    } else {
        ICPU_RUN_KF(max_pool_ext2<1>, numBlocks, x, y, workspace, tiling);
    }

    const std::string outFile = std::string(DATA_DIR) + "/float32_output_max_pool_ext2.bin";
    ASSERT_TRUE(WriteFile(outFile, y, yBytes)) << "write output failed";

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(workspace);
    AscendC::GmFree(tiling);

    ASSERT_TRUE(RunCmd("cd " + std::string(DATA_DIR) + " && python3 compare_data.py 'float32'"))
        << "golden comparison FAILED";
}
} // namespace

class MaxPoolExt2Test : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        printf("MaxPoolExt2Test SetUp\n");
        const std::string cmd = "cp -rf " + dataPath + " ./";
        ASSERT_EQ(std::system(cmd.c_str()), 0) << "copy data dir failed: " << dataPath;
        std::system("chmod -R 755 ./max_pool_ext2_data/");
    }
    static void TearDownTestCase() { printf("MaxPoolExt2Test TearDown\n"); }

private:
    static const std::string dataPath;
};

const std::string MaxPoolExt2Test::dataPath = "../../../../pooling/max_pool_ext2/tests/ut/op_kernel/max_pool_ext2_data";

// NCHW float32 VALID: (1,4,8,8) k[1,1,2,2] s[1,1,2,2] -> (1,4,4,4)
TEST_F(MaxPoolExt2Test, test_case_nchw_float32_valid)
{
    RunKernelCase("'(1, 4, 8, 8)' 'float32' '[1,1,2,2]' '[1,1,2,2]' 'VALID' 'NCHW'", 0, 8, 8, 4, 4, 2, 2, 2, 2, 0, 0, 1,
                  4, 1, 0);
}

// NHWC float32 SAME: (1,8,8,4) k[1,2,2,1] s[1,2,2,1] -> (1,4,4,4)
TEST_F(MaxPoolExt2Test, test_case_nhwc_float32_same)
{
    RunKernelCase("'(1, 8, 8, 4)' 'float32' '[1,2,2,1]' '[1,2,2,1]' 'SAME' 'NHWC'", 1, 8, 8, 4, 4, 2, 2, 2, 2, 0, 0, 1,
                  4, 1, 0);
}

// NHWC float32 SAME 非对称padding: (1,8,9,3) k[1,3,3,1] s[1,2,2,1]
// outH=4, outW=5, padH=(4-1)*2+3-8=1, padW=(5-1)*2+3-9=2
TEST_F(MaxPoolExt2Test, test_case_nhwc_float32_same_padding)
{
    RunKernelCase("'(1, 8, 9, 3)' 'float32' '[1,3,3,1]' '[1,2,2,1]' 'SAME' 'NHWC'", 1, 8, 9, 4, 5, 3, 3, 2, 2, 0, 1, 1,
                  3, 1, 0);
}

// NCHW float32 VALID 空输出: (1,2,2,2) k[1,1,5,5] -> H/W 输出为0
TEST_F(MaxPoolExt2Test, test_case_nchw_float32_valid_empty_output)
{
    RunKernelCase("'(1, 2, 2, 2)' 'float32' '[1,1,5,5]' '[1,1,1,1]' 'VALID' 'NCHW'", 0, 2, 2, 0, 0, 5, 5, 1, 1, 0, 0, 1,
                  2, 1, 0);
}

// NCHW float32 VALID 多batch多channel: (2,3,6,6) k[1,1,3,3] s[1,1,2,2] -> (2,3,2,2)
TEST_F(MaxPoolExt2Test, test_case_nchw_float32_multi_batch)
{
    RunKernelCase("'(2, 3, 6, 6)' 'float32' '[1,1,3,3]' '[1,1,2,2]' 'VALID' 'NCHW'", 0, 6, 6, 2, 2, 3, 3, 2, 2, 0, 0, 2,
                  3, 1, 0);
}
