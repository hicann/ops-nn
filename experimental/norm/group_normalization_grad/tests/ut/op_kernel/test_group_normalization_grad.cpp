/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Zhou Jianhua <@LePenseur>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <array>
#include <vector>
#include "gtest/gtest.h"
#include <cstdint>

#ifdef __CCE_KT_TEST__
#include "tikicpulib.h"
#include "data_utils.h"
#include "string.h"
#include <string>
#include "kernel_ut_data_helper.h"
#include "kernel_ut_data_executor.h"
#endif
#include "../../../op_kernel/group_normalization_grad.cpp"
#include "../../../op_kernel/group_normalization_grad_tiling_data.h"

using namespace std;

class GroupNormalizationGradTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        kernel_ut::SetupTestEnvironment(
            "experimental/norm/group_normalization_grad/tests/ut/op_kernel/group_normalization_grad_data",
            "group_normalization_grad_data");
    }
    static void TearDownTestCase() {}
};

// ============================================================================
// The kernel entry is `template <uint32_t schMode>`. Under __CCE_KT_TEST__ the
// if-constexpr ladder maps schMode 0/1/2 -> float/half/bfloat16_t, so a single
// UT binary exercises all three dtypes. The tiling key is always 0 (schMode is a
// scheduling selector decoupled from dtype); only the template argument picks
// the dtype here. blockDim > 1 validates the group-based core distribution.
// `shapeStr`/`groupCount` parameterize the shape so an uneven (tailBlockNum>0)
// distribution can also be covered.
// ============================================================================
template <uint32_t schMode>
static void RunGroupNormGradCase(const string& dType, size_t elemByteSize, uint32_t blockDim, const string& shapeStr,
                                 uint32_t groupCount)
{
    kernel_ut::RunGenData("./group_normalization_grad_data", {shapeStr, dType});

    // layout [N, G, M]; groupCount = N*G, groupElemNum = M = 64 (kept fixed here)
    const uint32_t groupElemNum = 64;
    const uint32_t totalNum = groupCount * groupElemNum;
    size_t inputByteSize = totalNum * elemByteSize;    // non-const: ReadFile takes size_t&
    size_t scalarByteSize = groupCount * elemByteSize; // mean/rstd: one scalar per group

    auto readInput = [&](const string& kind) -> uint8_t* {
        uint8_t* p = (uint8_t*)AscendC::GmAlloc(inputByteSize);
        ReadFile("./group_normalization_grad_data/" + dType + "_" + kind + "_t_group_normalization_grad.bin",
                 inputByteSize, p, inputByteSize);
        return p;
    };
    uint8_t* x = readInput("x");
    uint8_t* dy = readInput("dy");
    uint8_t* gamma = readInput("gamma");

    uint8_t* mean = (uint8_t*)AscendC::GmAlloc(scalarByteSize);
    ReadFile("./group_normalization_grad_data/" + dType + "_mean_t_group_normalization_grad.bin", scalarByteSize, mean,
             scalarByteSize);
    uint8_t* rstd = (uint8_t*)AscendC::GmAlloc(scalarByteSize);
    ReadFile("./group_normalization_grad_data/" + dType + "_rstd_t_group_normalization_grad.bin", scalarByteSize, rstd,
             scalarByteSize);

    size_t outputByteSize = totalNum * elemByteSize;
    uint8_t* dx = (uint8_t*)AscendC::GmAlloc(outputByteSize);
    uint8_t* ws = (uint8_t*)AscendC::GmAlloc(32);
    uint8_t* tl = (uint8_t*)AscendC::GmAlloc(sizeof(GroupNormalizationGradTilingData));

    // group-based core distribution matching tiling.cpp (coreNum == blockDim here)
    const uint32_t smallCoreGroupNum = groupCount / blockDim;
    const uint32_t tailBlockNum = groupCount % blockDim;
    const uint32_t bigCoreGroupNum = smallCoreGroupNum + 1;

    GroupNormalizationGradTilingData* tilingData = reinterpret_cast<GroupNormalizationGradTilingData*>(tl);
    tilingData->groupElemNum = groupElemNum;
    tilingData->groupCount = groupCount;
    tilingData->smallCoreGroupNum = smallCoreGroupNum;
    tilingData->bigCoreGroupNum = bigCoreGroupNum;
    tilingData->finalGroupTileNum = 1;
    tilingData->tileDataNum = groupElemNum;
    tilingData->alignedTileDataNum = groupElemNum;
    tilingData->tailDataNum = groupElemNum;
    tilingData->tailBlockNum = tailBlockNum;
    tilingData->groupElemNumFloat = static_cast<float>(groupElemNum);
    tilingData->groupElemNumReciprocal = 1.0f / static_cast<float>(groupElemNum);

    ICPU_SET_TILING_KEY(0);
    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto kf = [](GM_ADDR x, GM_ADDR dy, GM_ADDR gamma, GM_ADDR mean, GM_ADDR rstd, GM_ADDR dx, GM_ADDR w, GM_ADDR tl) {
        ::group_normalization_grad<schMode>(x, dy, gamma, mean, rstd, dx, w, tl);
    };
    ICPU_RUN_KF(kf, blockDim, x, dy, gamma, mean, rstd, dx, ws, tl);

    WriteFile("./group_normalization_grad_data/" + dType + "_output_t_group_normalization_grad.bin", dx,
              outputByteSize);

    AscendC::GmFree((void*)x);
    AscendC::GmFree((void*)dy);
    AscendC::GmFree((void*)gamma);
    AscendC::GmFree((void*)mean);
    AscendC::GmFree((void*)rstd);
    AscendC::GmFree((void*)dx);
    AscendC::GmFree((void*)ws);
    AscendC::GmFree((void*)tl);

    kernel_ut::RunCompareData("./group_normalization_grad_data", {dType});
}

// float32, single core
TEST_F(GroupNormalizationGradTest, test_case_float32_1)
{
    RunGroupNormGradCase<0>("float32", sizeof(float), 1, "'(2, 2, 64)'", 4);
}

// float16, single core (schMode 1 -> half)
TEST_F(GroupNormalizationGradTest, test_case_float16_1)
{
    RunGroupNormGradCase<1>("float16", sizeof(half), 1, "'(2, 2, 64)'", 4);
}

// bfloat16, single core (schMode 2 -> bfloat16_t); also regresses the
// DataCopyPad + PipeBarrier bf16 scalar-read path in ReadScalar.
TEST_F(GroupNormalizationGradTest, test_case_bfloat16_1)
{
    RunGroupNormGradCase<2>("bfloat16", sizeof(bfloat16_t), 1, "'(2, 2, 64)'", 4);
}

// float32, 2 cores, even distribution: groupCount=4, blockDim=2 -> tailBlockNum=0,
// both cores take the small-core branch (smallCoreGroupNum=2).
TEST_F(GroupNormalizationGradTest, test_case_float32_multicore)
{
    RunGroupNormGradCase<0>("float32", sizeof(float), 2, "'(2, 2, 64)'", 4);
}

// float32, 2 cores, uneven distribution: groupCount=5, blockDim=2 -> tailBlockNum=1,
// bigCoreGroupNum=3. block0 takes the big-core branch (groups {0,1,2}) and block1 the
// small-core branch with the (blockIdx - tailBlockNum) offset compensation (groups {3,4}).
TEST_F(GroupNormalizationGradTest, test_case_float32_multicore_uneven)
{
    RunGroupNormGradCase<0>("float32", sizeof(float), 2, "'(1, 5, 64)'", 5);
}
