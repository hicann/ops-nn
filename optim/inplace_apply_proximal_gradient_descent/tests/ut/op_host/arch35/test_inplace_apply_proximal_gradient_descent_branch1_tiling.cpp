/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * =============================================================================
 * Branch-1（历史分支名，FP16 SB，runtime key 0）TilingUT。
 * DESIGN-BRANCH-1 §9.1 的 B1-E0/E1/R0/A0/T0/U0/N0/R16 共 8 例，
 * 期望值由本文件的 §2 手算 oracle 独立生成，不调用被测 TilingFunc。
 * dtype 由 def 外层二进制承载；runtime key 只编码 BUFFER_MODE。切分 oracle
 * 不读取 datatype，也不调用被测 TilingFunc。
 * =============================================================================
 */

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include "../../../../op_host/arch35/inplace_apply_proximal_gradient_descent_tiling_host_arch35.h"

namespace {

// §0/§2/§4 常量，手抄自 DESIGN-BRANCH-1。
constexpr int64_t kSmallDataMax = 1024;
constexpr int64_t kMinElemsPerCore = 2048;
constexpr int64_t kBlockAlignElems = 512;
constexpr uint64_t kUbSize = 253952ULL;
constexpr int64_t kUbReserveBytes = 8192;
constexpr int64_t kWorstBytesPerElem = 26;
constexpr int64_t kInstanceFixedBytes = 192;
constexpr int64_t kUbAlignElems = 128;
constexpr int64_t kUbFactorExpected = 9344;
constexpr int64_t kFp16Bytes = 2;
constexpr int64_t kFp16DataCopyAlignElems = 16;
constexpr uint32_t kAvailableCoreNum = 80;

inline int64_t CeilDiv(int64_t value, int64_t divisor)
{
    return value / divisor + static_cast<int64_t>(value % divisor != 0);
}

inline int64_t AlignUp(int64_t value, int64_t align) { return CeilDiv(value, align) * align; }

inline int64_t AlignDown(int64_t value, int64_t align) { return (value / align) * align; }

struct OracleBranch1Tiling {
    int64_t dim0;
    int32_t usedCoreNum;
    int32_t reserved;
    int64_t blockFactor;
    int64_t blockTail;
    int64_t ubFactor;
    int64_t ubLoopOfFormerBlock;
    int64_t ubTailOfFormerBlock;
    int64_t ubLoopOfTailBlock;
    int64_t ubTailOfTailBlock;
};

// §2.1/§2.2 独立切分 oracle。只使用本文件常量和基础整数算术。
OracleBranch1Tiling OracleComputeBranch1(int64_t dim0, uint32_t availableCoreNum, uint64_t ubSize)
{
    OracleBranch1Tiling t{};
    t.dim0 = dim0;
    if (dim0 == 0) {
        t.usedCoreNum = 1;
        return t;
    }

    const int64_t candidateCoreNum = std::min<int64_t>(static_cast<int64_t>(availableCoreNum),
                                                       std::max<int64_t>(1, CeilDiv(dim0, kMinElemsPerCore)));
    t.blockFactor = AlignUp(CeilDiv(dim0, candidateCoreNum), kBlockAlignElems);
    t.usedCoreNum = static_cast<int32_t>(CeilDiv(dim0, t.blockFactor));
    t.blockTail = dim0 - static_cast<int64_t>(t.usedCoreNum - 1) * t.blockFactor;
    t.ubFactor = AlignDown((static_cast<int64_t>(ubSize) - kUbReserveBytes) / kWorstBytesPerElem, kUbAlignElems);
    t.ubLoopOfFormerBlock = CeilDiv(t.blockFactor, t.ubFactor);
    t.ubTailOfFormerBlock = t.blockFactor - (t.ubLoopOfFormerBlock - 1) * t.ubFactor;
    t.ubLoopOfTailBlock = CeilDiv(t.blockTail, t.ubFactor);
    t.ubTailOfTailBlock = t.blockTail - (t.ubLoopOfTailBlock - 1) * t.ubFactor;
    return t;
}

struct Branch1Case {
    const char* name;
    gert::Shape shape;
    int64_t dim0;
    int64_t expectedBlockFactor;
    int64_t expectedRightPadding;
};

const Branch1Case kCases[] = {
    {"B1E0", gert::Shape({0}), 0, 0, 0},
    {"B1E1", gert::Shape({2, 0, 3}), 0, 0, 0},
    {"B1R0", gert::Shape{}, 1, 512, 15},
    {"B1A0", gert::Shape({512}), 512, 512, 0},
    {"B1T0", gert::Shape({513}), 513, 1024, 15},
    {"B1U0", gert::Shape({1024}), 1024, 1024, 0},
    {"B1N0", gert::Shape({33}), 33, 512, 15},
    {"B1R16", gert::Shape({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2}), 2, 512, 14},
};
constexpr size_t kCaseCount = sizeof(kCases) / sizeof(kCases[0]);

void ExpectBranch1Result(const Branch1Case& p, const InplaceApplyProximalGradientDescentTilingData& actual, bool ok)
{
    // Shape/rank 覆盖使用公共展平入口，但期望 dim0 是表中独立手算常量。
    int64_t flattened = -1;
    ASSERT_TRUE(optiling::CalcDim0(p.shape, flattened));
    EXPECT_EQ(flattened, p.dim0);

    // §0/§2：FP16 由外层二进制承载，mode 0 生成 runtime key 0。
    const uint64_t bufferMode = (p.dim0 <= kSmallDataMax) ? 0U : 1U;
    EXPECT_EQ(bufferMode, 0U);
    if (p.dim0 == kSmallDataMax) {
        EXPECT_EQ((kSmallDataMax + 1 <= kSmallDataMax) ? 0U : 1U, 1U);
    }
    if (!ok) {
        return;
    }

    const OracleBranch1Tiling expected = OracleComputeBranch1(p.dim0, kAvailableCoreNum, kUbSize);
    EXPECT_EQ(actual.dim0, expected.dim0);
    EXPECT_EQ(actual.usedCoreNum, expected.usedCoreNum);
    EXPECT_EQ(actual.reserved, expected.reserved);
    EXPECT_EQ(actual.blockFactor, expected.blockFactor);
    EXPECT_EQ(actual.blockTail, expected.blockTail);
    EXPECT_EQ(actual.ubFactor, expected.ubFactor);
    EXPECT_EQ(actual.ubLoopOfFormerBlock, expected.ubLoopOfFormerBlock);
    EXPECT_EQ(actual.ubTailOfFormerBlock, expected.ubTailOfFormerBlock);
    EXPECT_EQ(actual.ubLoopOfTailBlock, expected.ubLoopOfTailBlock);
    EXPECT_EQ(actual.ubTailOfTailBlock, expected.ubTailOfTailBlock);
    EXPECT_EQ(expected.blockFactor, p.expectedBlockFactor);

    if (p.dim0 == 0) {
        EXPECT_EQ(expected.usedCoreNum, 1);
        EXPECT_EQ(expected.blockFactor, 0);
        EXPECT_EQ(expected.blockTail, 0);
        EXPECT_EQ(expected.ubFactor, 0);
        EXPECT_EQ(expected.ubLoopOfFormerBlock, 0);
        EXPECT_EQ(expected.ubTailOfFormerBlock, 0);
        EXPECT_EQ(expected.ubLoopOfTailBlock, 0);
        EXPECT_EQ(expected.ubTailOfTailBlock, 0);
        return;
    }

    EXPECT_EQ(expected.usedCoreNum, 1);
    EXPECT_EQ(expected.blockTail, p.dim0);
    EXPECT_EQ(expected.ubFactor, kUbFactorExpected);
    EXPECT_EQ(expected.ubLoopOfTailBlock, 1);
    EXPECT_EQ(expected.ubTailOfTailBlock, p.dim0);

    // 核区间和 tile 的 GM 有效区间连续、无重叠且并集严格为 [0, dim0)。
    int64_t gmCursor = 0;
    for (int32_t core = 0; core < expected.usedCoreNum; ++core) {
        const int64_t coreStart = static_cast<int64_t>(core) * expected.blockFactor;
        const int64_t coreLength = (core == expected.usedCoreNum - 1) ? expected.blockTail : expected.blockFactor;
        const int64_t loopCount = (core == expected.usedCoreNum - 1) ? expected.ubLoopOfTailBlock :
                                                                       expected.ubLoopOfFormerBlock;
        const int64_t tailCount = (core == expected.usedCoreNum - 1) ? expected.ubTailOfTailBlock :
                                                                       expected.ubTailOfFormerBlock;
        EXPECT_EQ(gmCursor, coreStart);
        for (int64_t loop = 0; loop < loopCount; ++loop) {
            const int64_t tileStart = coreStart + loop * expected.ubFactor;
            const int64_t count = (loop == loopCount - 1) ? tailCount : expected.ubFactor;
            EXPECT_EQ(gmCursor, tileStart);
            EXPECT_GT(count, 0);
            EXPECT_LE(tileStart + count, coreStart + coreLength);
            gmCursor += count;
        }
        EXPECT_EQ(gmCursor, coreStart + coreLength);
    }
    EXPECT_EQ(gmCursor, expected.dim0);

    // GM 只读写 count*2B；32B 对齐补齐仅增加 UB 侧元素，不扩大 GM 区间。
    const int64_t count = expected.ubTailOfTailBlock;
    const int64_t rightPadding = (kFp16DataCopyAlignElems - count % kFp16DataCopyAlignElems) % kFp16DataCopyAlignElems;
    EXPECT_EQ(rightPadding, p.expectedRightPadding);
    EXPECT_EQ((count + rightPadding) % kFp16DataCopyAlignElems, 0);
    EXPECT_EQ(count * kFp16Bytes, p.dim0 * kFp16Bytes);
    EXPECT_EQ(gmCursor * kFp16Bytes, p.dim0 * kFp16Bytes);

    // §4 Host 统一预算上界和本 FP16 SB 实例实际物理占用。
    const int64_t hostUpperBound = kWorstBytesPerElem * expected.ubFactor + kUbReserveBytes;
    const int64_t instancePhysicalBytes = kWorstBytesPerElem * expected.ubFactor + kInstanceFixedBytes;
    EXPECT_EQ(hostUpperBound, 251136);
    EXPECT_LE(hostUpperBound, static_cast<int64_t>(kUbSize));
    EXPECT_EQ(instancePhysicalBytes, 243136);
    EXPECT_LE(instancePhysicalBytes, static_cast<int64_t>(kUbSize));
}

} // namespace

class Branch1TilingTest : public testing::TestWithParam<Branch1Case> {};

TEST_P(Branch1TilingTest, FormulaAndInvariants)
{
    const Branch1Case& p = GetParam();
    const optiling::Branch1TilingInputs in{p.dim0, kAvailableCoreNum, kUbSize};
    InplaceApplyProximalGradientDescentTilingData actual{};
    std::memset(&actual, 0, sizeof(actual));
    const bool ok = optiling::ComputeBranch1Tiling(in, actual);
    EXPECT_TRUE(ok);
    ExpectBranch1Result(p, actual, ok);
}

INSTANTIATE_TEST_SUITE_P(DesignBranch1, Branch1TilingTest, testing::ValuesIn(kCases, kCases + kCaseCount),
                         [](const testing::TestParamInfo<Branch1Case>& info) { return info.param.name; });
