/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <cstdint>
#include <iostream>
#include "gtest/gtest.h"
#include "tikicpulib.h"

// The kernel entry is templated on schMode; the data type comes from DTYPE_X (half in this UT).
template <uint64_t schMode>
__global__ __aicore__ void situ_glu_group(GM_ADDR x, GM_ADDR weight, GM_ADDR groupIndex, GM_ADDR y, GM_ADDR workspace,
                                          GM_ADDR tiling);

namespace {
class SituGluGroupKernelTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SituGluGroupKernelTest SetUp" << std::endl; }

    static void TearDownTestCase() { std::cout << "SituGluGroupKernelTest TearDown" << std::endl; }
};

void FillSituGluGroupTiling(SituGluGroupTilingData* tilingData, int64_t bs, int64_t d, uint32_t coreNum, float beta,
                            float linearBeta, int64_t activateLeft)
{
    int64_t splitD = d / 2;
    (void)coreNum; // row tiling is recomputed from GetBlockNum() inside the kernel
    tilingData->bs = bs;
    tilingData->d = d;
    tilingData->splitD = splitD;
    tilingData->rowFactor = 1;
    tilingData->dLoop = 1;
    tilingData->dFactor = splitD;
    tilingData->tailDFactor = splitD;
    tilingData->beta = beta;
    tilingData->linearBeta = linearBeta;
    tilingData->activateLeft = activateLeft;
    // Match the host tiling: the kernel pool gets total UB (253952) minus the 8K reserve that
    // absorbs the full-VL tail over-read of the VF loader / any framework stack usage.
    tilingData->ubSize = 253952 - 8192;
    tilingData->groupIndexLoop = 0;
    tilingData->groupIndexFactor = 0;
    tilingData->tailGroupIndexFactor = 0;
    tilingData->hasWeight = 0;
    tilingData->hasGroupIndex = 0;
}

// Run situ_glu_group (arch35) with optional weight / group_index, one row per core.
void RunKernel(int64_t bs, int64_t d, bool withWeight, int64_t gLen, float beta, float linearBeta, int64_t activateLeft)
{
    int64_t splitD = d / 2;
    uint32_t blockDim = static_cast<uint32_t>(bs);

    uint8_t* x = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(bs * d * sizeof(half)));
    uint8_t* y = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(bs * splitD * sizeof(half)));
    uint8_t* weight = withWeight ? reinterpret_cast<uint8_t*>(AscendC::GmAlloc(bs * sizeof(float))) : nullptr;
    uint8_t* groupIndex = (gLen > 0) ? reinterpret_cast<uint8_t*>(AscendC::GmAlloc(gLen * sizeof(int64_t))) : nullptr;
    uint8_t* workspace = nullptr;
    uint8_t* tiling = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sizeof(SituGluGroupTilingData)));

    if (groupIndex != nullptr) {
        // count mode: fill group_index so its sum equals bs (all rows are produced).
        auto* gi = reinterpret_cast<int64_t*>(groupIndex);
        for (int64_t i = 0; i < gLen; i++) {
            gi[i] = 0;
        }
        gi[gLen - 1] = bs;
    }

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto* tilingData = reinterpret_cast<SituGluGroupTilingData*>(tiling);
    FillSituGluGroupTiling(tilingData, bs, d, blockDim, beta, linearBeta, activateLeft);
    tilingData->hasWeight = withWeight ? 1 : 0;
    if (groupIndex != nullptr) {
        tilingData->groupIndexLoop = 1;
        tilingData->groupIndexFactor = gLen;
        tilingData->tailGroupIndexFactor = gLen;
        tilingData->hasGroupIndex = 1;
    }

    ICPU_SET_TILING_KEY(0); // schMode == SITU_GLU_GROUP_TPL_SCH_MODE_0
    ICPU_RUN_KF((situ_glu_group<0>), blockDim, x, weight, groupIndex, y, workspace, tiling);

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    if (weight != nullptr) {
        AscendC::GmFree(weight);
    }
    if (groupIndex != nullptr) {
        AscendC::GmFree(groupIndex);
    }
    AscendC::GmFree(tiling);
}

float RefSituGroup(float gate, float up, float beta, float linearBeta, float rowWeight)
{
    const float sig = 1.0f / (1.0f + std::exp(-gate));
    float y = beta * std::tanh(gate / beta) * sig;
    y *= (linearBeta > 0.0f) ? (linearBeta * std::tanh(up / linearBeta)) : up;
    return y * rowWeight;
}

// Numeric check of the VF activation against a CPU golden (fp16 inputs are quantized first).
void RunKernelNumeric(int64_t bs, int64_t d, bool withWeight, bool hasLinearBeta, int64_t activateLeft)
{
    const int64_t splitD = d / 2;
    const uint32_t blockDim = static_cast<uint32_t>(bs);
    const float beta = 2.0f;
    const float linearBeta = hasLinearBeta ? 3.0f : 0.0f;

    uint8_t* x = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(bs * d * sizeof(half)));
    uint8_t* y = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(bs * splitD * sizeof(half)));
    uint8_t* weight = withWeight ? reinterpret_cast<uint8_t*>(AscendC::GmAlloc(bs * sizeof(float))) : nullptr;
    uint8_t* workspace = nullptr;
    uint8_t* tiling = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sizeof(SituGluGroupTilingData)));

    auto* xH = reinterpret_cast<half*>(x);
    for (int64_t i = 0; i < bs * d; i++) {
        xH[i] = half(static_cast<float>((i % 37) - 18) * 0.125f);
    }
    auto* w = reinterpret_cast<float*>(weight);
    if (withWeight) {
        for (int64_t r = 0; r < bs; r++) {
            w[r] = 0.5f + 0.25f * static_cast<float>(r);
        }
    }

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto* tilingData = reinterpret_cast<SituGluGroupTilingData*>(tiling);
    FillSituGluGroupTiling(tilingData, bs, d, blockDim, beta, linearBeta, activateLeft);
    tilingData->hasWeight = withWeight ? 1 : 0;

    ICPU_SET_TILING_KEY(0); // schMode == SITU_GLU_GROUP_TPL_SCH_MODE_0
    ICPU_RUN_KF((situ_glu_group<0>), blockDim, x, weight, nullptr, y, workspace, tiling);

    auto* yH = reinterpret_cast<half*>(y);
    for (int64_t r = 0; r < bs; r++) {
        for (int64_t c = 0; c < splitD; c++) {
            const int64_t gateIdx = (activateLeft == 1) ? r * d + c : r * d + splitD + c;
            const int64_t upIdx = (activateLeft == 1) ? r * d + splitD + c : r * d + c;
            const float gate = static_cast<float>(xH[gateIdx]);
            const float up = static_cast<float>(xH[upIdx]);
            const float expected = RefSituGroup(gate, up, beta, linearBeta, withWeight ? w[r] : 1.0f);
            EXPECT_NEAR(static_cast<float>(yH[r * splitD + c]), expected, 1e-2) << "row " << r << " col " << c;
        }
    }

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    if (weight != nullptr) {
        AscendC::GmFree(weight);
    }
    AscendC::GmFree(tiling);
}

// group_index truncation: only the first realBs rows are computed/written. The tail rows are
// intentionally left undefined, so only rows [0, realBs) are compared against the golden.
void RunKernelPartialGroup(int64_t bs, int64_t d, bool withWeight, bool hasLinearBeta, int64_t activateLeft,
                           int64_t realBs)
{
    const int64_t splitD = d / 2;
    const uint32_t blockDim = static_cast<uint32_t>(bs);
    const float beta = 2.0f;
    const float linearBeta = hasLinearBeta ? 3.0f : 0.0f;
    const int64_t gLen = 2;

    uint8_t* x = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(bs * d * sizeof(half)));
    uint8_t* y = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(bs * splitD * sizeof(half)));
    uint8_t* weight = withWeight ? reinterpret_cast<uint8_t*>(AscendC::GmAlloc(bs * sizeof(float))) : nullptr;
    uint8_t* groupIndex = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(gLen * sizeof(int64_t)));
    uint8_t* workspace = nullptr;
    uint8_t* tiling = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sizeof(SituGluGroupTilingData)));

    auto* xH = reinterpret_cast<half*>(x);
    for (int64_t i = 0; i < bs * d; i++) {
        xH[i] = half(static_cast<float>((i % 37) - 18) * 0.125f);
    }
    auto* w = reinterpret_cast<float*>(weight);
    if (withWeight) {
        for (int64_t r = 0; r < bs; r++) {
            w[r] = 0.5f + 0.25f * static_cast<float>(r);
        }
    }
    auto* gi = reinterpret_cast<int64_t*>(groupIndex);
    gi[0] = realBs;
    gi[1] = 0;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto* tilingData = reinterpret_cast<SituGluGroupTilingData*>(tiling);
    FillSituGluGroupTiling(tilingData, bs, d, blockDim, beta, linearBeta, activateLeft);
    tilingData->hasWeight = withWeight ? 1 : 0;
    tilingData->groupIndexLoop = 1;
    tilingData->groupIndexFactor = gLen;
    tilingData->tailGroupIndexFactor = gLen;
    tilingData->hasGroupIndex = 1;

    ICPU_SET_TILING_KEY(0); // schMode == SITU_GLU_GROUP_TPL_SCH_MODE_0
    ICPU_RUN_KF((situ_glu_group<0>), blockDim, x, weight, groupIndex, y, workspace, tiling);

    auto* yH = reinterpret_cast<half*>(y);
    for (int64_t r = 0; r < realBs; r++) {
        for (int64_t c = 0; c < splitD; c++) {
            const int64_t gateIdx = (activateLeft == 1) ? r * d + c : r * d + splitD + c;
            const int64_t upIdx = (activateLeft == 1) ? r * d + splitD + c : r * d + c;
            const float gate = static_cast<float>(xH[gateIdx]);
            const float up = static_cast<float>(xH[upIdx]);
            const float expected = RefSituGroup(gate, up, beta, linearBeta, withWeight ? w[r] : 1.0f);
            EXPECT_NEAR(static_cast<float>(yH[r * splitD + c]), expected, 1e-2) << "row " << r << " col " << c;
        }
    }

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    if (weight != nullptr) {
        AscendC::GmFree(weight);
    }
    AscendC::GmFree(groupIndex);
    AscendC::GmFree(tiling);
}

// group_index length not a multiple of the int64 VL (32): the sum loop's tail mask must still
// yield exactly realBs. Pre-fill y with a sentinel so we can assert that exactly rows
// [0, realBs) were written and rows [realBs, bs) were left untouched.
void RunKernelGroupTail(int64_t bs, int64_t d, int64_t gLen, int64_t realBs)
{
    const int64_t splitD = d / 2;
    const uint32_t blockDim = static_cast<uint32_t>(bs);
    const float beta = 2.0f;
    const float linearBeta = 3.0f;
    const float kSentinel = 12345.0f;
    const half kSentinelH = half(kSentinel);

    uint8_t* x = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(bs * d * sizeof(half)));
    uint8_t* y = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(bs * splitD * sizeof(half)));
    uint8_t* groupIndex = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(gLen * sizeof(int64_t)));
    uint8_t* workspace = nullptr;
    uint8_t* tiling = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sizeof(SituGluGroupTilingData)));

    auto* xH = reinterpret_cast<half*>(x);
    for (int64_t i = 0; i < bs * d; i++) {
        xH[i] = half(static_cast<float>((i % 37) - 18) * 0.125f);
    }
    auto* yH = reinterpret_cast<half*>(y);
    for (int64_t i = 0; i < bs * splitD; i++) {
        yH[i] = kSentinelH;
    }

    auto* gi = reinterpret_cast<int64_t*>(groupIndex);
    for (int64_t i = 0; i < gLen; i++) {
        gi[i] = 0;
    }
    gi[gLen - 1] = realBs;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto* tilingData = reinterpret_cast<SituGluGroupTilingData*>(tiling);
    FillSituGluGroupTiling(tilingData, bs, d, blockDim, beta, linearBeta, 1);
    tilingData->hasWeight = 0;
    tilingData->groupIndexLoop = 1;
    tilingData->groupIndexFactor = gLen;
    tilingData->tailGroupIndexFactor = gLen;
    tilingData->hasGroupIndex = 1;

    ICPU_SET_TILING_KEY(0); // schMode == SITU_GLU_GROUP_TPL_SCH_MODE_0
    ICPU_RUN_KF((situ_glu_group<0>), blockDim, x, nullptr, groupIndex, y, workspace, tiling);

    for (int64_t r = 0; r < realBs; r++) {
        for (int64_t c = 0; c < splitD; c++) {
            const float gate = static_cast<float>(xH[r * d + c]);
            const float up = static_cast<float>(xH[r * d + splitD + c]);
            const float expected = RefSituGroup(gate, up, beta, linearBeta, 1.0f);
            EXPECT_NEAR(static_cast<float>(yH[r * splitD + c]), expected, 1e-2)
                << "gLen=" << gLen << " row " << r << " col " << c;
        }
    }
    for (int64_t r = realBs; r < bs; r++) {
        for (int64_t c = 0; c < splitD; c++) {
            EXPECT_FLOAT_EQ(static_cast<float>(yH[r * splitD + c]), static_cast<float>(kSentinelH))
                << "gLen=" << gLen << " wrote row " << r << " beyond realBs=" << realBs;
        }
    }

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(groupIndex);
    AscendC::GmFree(tiling);
}

// Multi-chunk group_index (groupIndexLoop > 1) with a non-VL tail. Chunk0/chunk1 are full 64-element
// chunks filled with 1s; chunk2 is a 33-element tail (last value 2) whose stale lanes 33..63
// hold 1s from a recycled double buffer. Correct masking sums 64 + 64 + 2 = 130; a leaked
// tail would push realBs past 130 and write rows that must stay sentinel.
void RunKernelMultiChunkTail()
{
    const int64_t bs = 140;
    const int64_t d = 256;
    const int64_t splitD = d / 2;
    const int64_t groupIndexFactor = 64;
    const int64_t tailGroupIndexFactor = 33;
    const int64_t gLen = 2 * groupIndexFactor + tailGroupIndexFactor;
    const int64_t realBs = 2 * groupIndexFactor + 2;
    const uint32_t blockDim = static_cast<uint32_t>(bs);
    const float beta = 2.0f;
    const float linearBeta = 3.0f;
    const float kSentinel = 12345.0f;
    const half kSentinelH = half(kSentinel);

    uint8_t* x = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(bs * d * sizeof(half)));
    uint8_t* y = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(bs * splitD * sizeof(half)));
    uint8_t* groupIndex = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(gLen * sizeof(int64_t)));
    uint8_t* workspace = nullptr;
    uint8_t* tiling = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sizeof(SituGluGroupTilingData)));

    auto* xH = reinterpret_cast<half*>(x);
    for (int64_t i = 0; i < bs * d; i++) {
        xH[i] = half(static_cast<float>((i % 37) - 18) * 0.125f);
    }
    auto* yH = reinterpret_cast<half*>(y);
    for (int64_t i = 0; i < bs * splitD; i++) {
        yH[i] = kSentinelH;
    }

    auto* gi = reinterpret_cast<int64_t*>(groupIndex);
    for (int64_t i = 0; i < gLen; i++) {
        gi[i] = (i < 2 * groupIndexFactor) ? 1 : 0;
    }
    gi[gLen - 1] = realBs - 2 * groupIndexFactor;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto* tilingData = reinterpret_cast<SituGluGroupTilingData*>(tiling);
    FillSituGluGroupTiling(tilingData, bs, d, blockDim, beta, linearBeta, 1);
    tilingData->hasWeight = 0;
    tilingData->groupIndexLoop = 3;
    tilingData->groupIndexFactor = groupIndexFactor;
    tilingData->tailGroupIndexFactor = tailGroupIndexFactor;
    tilingData->hasGroupIndex = 1;

    ICPU_SET_TILING_KEY(0); // schMode == SITU_GLU_GROUP_TPL_SCH_MODE_0
    ICPU_RUN_KF((situ_glu_group<0>), blockDim, x, nullptr, groupIndex, y, workspace, tiling);

    for (int64_t r = 0; r < realBs; r++) {
        for (int64_t c = 0; c < splitD; c++) {
            const float gate = static_cast<float>(xH[r * d + c]);
            const float up = static_cast<float>(xH[r * d + splitD + c]);
            const float expected = RefSituGroup(gate, up, beta, linearBeta, 1.0f);
            EXPECT_NEAR(static_cast<float>(yH[r * splitD + c]), expected, 1e-2) << "row " << r << " col " << c;
        }
    }
    for (int64_t r = realBs; r < bs; r++) {
        for (int64_t c = 0; c < splitD; c++) {
            EXPECT_FLOAT_EQ(static_cast<float>(yH[r * splitD + c]), static_cast<float>(kSentinelH))
                << "wrote row " << r << " beyond realBs=" << realBs;
        }
    }

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(groupIndex);
    AscendC::GmFree(tiling);
}

// The runtime may launch fewer blocks than the host tiling assumed (core limiting / fused
// scheduling). The group path must split realBs over the blocks actually launched (GetBlockNum());
// splitting over the host core count would assign rows to non-existent blocks and drop them.
// Only 2 blocks are launched here, while the tiling helper is passed a different core count: with
// the buggy split only rows 0..1 are written and rows 2..7 stay at the sentinel.
void RunKernelLimitedBlocks()
{
    const int64_t bs = 8;
    const int64_t d = 256;
    const int64_t splitD = d / 2;
    const int64_t gLen = 4;
    const int64_t realBs = bs;
    const uint32_t blockDim = 2;
    const uint32_t tilingCoreNum = 8;
    const float beta = 2.0f;
    const float linearBeta = 3.0f;
    const float kSentinel = 12345.0f;
    const half kSentinelH = half(kSentinel);

    uint8_t* x = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(bs * d * sizeof(half)));
    uint8_t* y = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(bs * splitD * sizeof(half)));
    uint8_t* groupIndex = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(gLen * sizeof(int64_t)));
    uint8_t* workspace = nullptr;
    uint8_t* tiling = reinterpret_cast<uint8_t*>(AscendC::GmAlloc(sizeof(SituGluGroupTilingData)));

    auto* xH = reinterpret_cast<half*>(x);
    for (int64_t i = 0; i < bs * d; i++) {
        xH[i] = half(static_cast<float>((i % 37) - 18) * 0.125f);
    }
    auto* yH = reinterpret_cast<half*>(y);
    for (int64_t i = 0; i < bs * splitD; i++) {
        yH[i] = kSentinelH;
    }

    auto* gi = reinterpret_cast<int64_t*>(groupIndex);
    for (int64_t i = 0; i < gLen; i++) {
        gi[i] = 0;
    }
    gi[gLen - 1] = realBs;

    AscendC::SetKernelMode(KernelMode::AIV_MODE);
    auto* tilingData = reinterpret_cast<SituGluGroupTilingData*>(tiling);
    FillSituGluGroupTiling(tilingData, bs, d, tilingCoreNum, beta, linearBeta, 1);
    tilingData->hasWeight = 0;
    tilingData->groupIndexLoop = 1;
    tilingData->groupIndexFactor = gLen;
    tilingData->tailGroupIndexFactor = gLen;
    tilingData->hasGroupIndex = 1;

    ICPU_SET_TILING_KEY(0); // schMode == SITU_GLU_GROUP_TPL_SCH_MODE_0
    ICPU_RUN_KF((situ_glu_group<0>), blockDim, x, nullptr, groupIndex, y, workspace, tiling);

    for (int64_t r = 0; r < bs; r++) {
        for (int64_t c = 0; c < splitD; c++) {
            const float gate = static_cast<float>(xH[r * d + c]);
            const float up = static_cast<float>(xH[r * d + splitD + c]);
            const float expected = RefSituGroup(gate, up, beta, linearBeta, 1.0f);
            EXPECT_NEAR(static_cast<float>(yH[r * splitD + c]), expected, 1e-2)
                << "row " << r << " col " << c << " was not produced with blockDim=" << blockDim;
        }
    }

    AscendC::GmFree(x);
    AscendC::GmFree(y);
    AscendC::GmFree(groupIndex);
    AscendC::GmFree(tiling);
}

TEST_F(SituGluGroupKernelTest, basic_fp16) { RunKernel(2, 256, false, 0, 1.0f, 0.0f, 1); }

TEST_F(SituGluGroupKernelTest, weight_group_fp16) { RunKernel(4, 256, true, 8, 4.0f, 25.0f, 1); }

TEST_F(SituGluGroupKernelTest, activate_right_fp16) { RunKernel(2, 256, true, 0, 1.0f, 0.0f, 0); }

// group_index element count >= 97 exercises VFProcessGroupIndexLargeVf (four-unfold int64 reduce).
TEST_F(SituGluGroupKernelTest, group_index_large) { RunKernel(4, 256, false, 97, 1.0f, 0.0f, 1); }

TEST_F(SituGluGroupKernelTest, numeric_activate_left) { RunKernelNumeric(2, 256, false, false, 1); }

TEST_F(SituGluGroupKernelTest, numeric_weight_linear_beta) { RunKernelNumeric(2, 256, true, true, 1); }

// d = 250 -> splitD = 125 (not 32B aligned), exercises the masked VF tail path.
TEST_F(SituGluGroupKernelTest, numeric_activate_right_unaligned) { RunKernelNumeric(2, 250, true, true, 0); }

// group_index sum = 3 < rows = 4: only rows 0..2 are written and compared.
TEST_F(SituGluGroupKernelTest, numeric_group_truncation) { RunKernelPartialGroup(4, 256, true, true, 1, 3); }

TEST_F(SituGluGroupKernelTest, numeric_group_truncation_no_weight)
{
    RunKernelPartialGroup(2, 256, false, false, 0, 1);
}

// group_index lengths that are not a multiple of the int64 VL (32). int64 VL = 256B / 8B = 32.
// realBs is pinned to 5 while bs = 8: an under-count leaves rows [0,5) as sentinel, an
// over-count writes rows [5,8), so both directions are detected.
TEST_F(SituGluGroupKernelTest, group_index_tail_33) { RunKernelGroupTail(8, 256, 33, 5); }
TEST_F(SituGluGroupKernelTest, group_index_tail_40) { RunKernelGroupTail(8, 256, 40, 5); }
TEST_F(SituGluGroupKernelTest, group_index_tail_65) { RunKernelGroupTail(8, 256, 65, 5); }
TEST_F(SituGluGroupKernelTest, group_index_tail_97) { RunKernelGroupTail(8, 256, 97, 5); }
// 257 -> loopCount 9, fourLoopCount 2, tailLoopNum 1: exercises the four-unfold main loop
// together with a 1-element tail.
TEST_F(SituGluGroupKernelTest, group_index_tail_257) { RunKernelGroupTail(8, 256, 257, 5); }

// groupIndexLoop = 3 with a 33-element tail (double-buffer recycle path, withUbReduce=true).
TEST_F(SituGluGroupKernelTest, group_index_multichunk_tail) { RunKernelMultiChunkTail(); }

// blockDim (2) < host tiling core count (8): the row split must follow the launched block count.
TEST_F(SituGluGroupKernelTest, group_index_limited_blocks) { RunKernelLimitedBlocks(); }
} // namespace
