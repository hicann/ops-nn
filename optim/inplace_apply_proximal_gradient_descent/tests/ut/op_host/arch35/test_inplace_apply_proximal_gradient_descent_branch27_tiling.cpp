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
 * Branch-27（历史分支名，BF16 SB，runtime key 0）测试先行 TilingUT。
 * DESIGN-BRANCH-27 §9.1 的 B27-E0/E1/R0/A0/T0/U0/N0/R16 共 8 例；
 * 期望值由本文件按 §2 独立手算，不调用被测 Tiling 函数。
 */

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include "graph/types.h"
#include "../../../../op_host/arch35/inplace_apply_proximal_gradient_descent_tiling_host_arch35.h"

namespace {

// DESIGN-BRANCH-27 §0/§2/§4 常量。
constexpr int64_t kSmallDataMax = 1024;
constexpr int64_t kMinElemsPerCore = 2048;
constexpr int64_t kBlockAlignElems = 512;
constexpr uint64_t kUbSize = 253952ULL;
constexpr int64_t kUbReserveBytes = 8192;
constexpr int64_t kWorstBytesPerElem = 26;
constexpr int64_t kInstanceFixedBytes = 192;
constexpr int64_t kUbAlignElems = 128;
constexpr int64_t kUbFactorExpected = 9344;
constexpr int64_t kBf16Bytes = 2;
constexpr int64_t kDataCopyAlignElems = 16;
constexpr uint32_t kAvailableCoreNum = 80;

inline int64_t CeilDiv(int64_t value, int64_t divisor)
{
    return value / divisor + static_cast<int64_t>(value % divisor != 0);
}

inline int64_t AlignUp(int64_t value, int64_t align) { return CeilDiv(value, align) * align; }

inline int64_t AlignDown(int64_t value, int64_t align) { return (value / align) * align; }

struct OracleRoute {
    int64_t bufferMode;
    int32_t tilingKey;
};

// §0：BF16 由 def 外层二进制承载，runtime key 只等于 BUFFER_MODE。
OracleRoute OracleRouteBranch27(ge::DataType input0Type, int64_t dim0)
{
    const int64_t bufferMode = (dim0 <= kSmallDataMax) ? 0 : 1;
    if (input0Type != ge::DT_BF16) {
        return {bufferMode, -1};
    }
    return {bufferMode, static_cast<int32_t>(bufferMode)};
}

struct OracleBranch27Tiling {
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

// §2.1/§2.2 独立切分 oracle：仅使用设计常量和基础整数算术。
OracleBranch27Tiling OracleComputeBranch27(int64_t dim0, uint32_t availableCoreNum, uint64_t ubSize)
{
    OracleBranch27Tiling t{};
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

struct Branch27Case {
    const char* name;
    gert::Shape shape;
    int64_t dim0;
    int64_t expectedBlockFactor;
    int64_t expectedRightPadding;
};

const Branch27Case kCases[] = {
    {"B27E0", gert::Shape({0}), 0, 0, 0},
    {"B27E1", gert::Shape({2, 0, 3}), 0, 0, 0},
    {"B27R0", gert::Shape{}, 1, 512, 15},
    {"B27A0", gert::Shape({512}), 512, 512, 0},
    {"B27T0", gert::Shape({513}), 513, 1024, 15},
    {"B27U0", gert::Shape({1024}), 1024, 1024, 0},
    {"B27N0", gert::Shape({33}), 33, 512, 15},
    {"B27R16", gert::Shape({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2}), 2, 512, 14},
};
constexpr size_t kCaseCount = sizeof(kCases) / sizeof(kCases[0]);

void ExpectBranch27Result(const Branch27Case& p, const InplaceApplyProximalGradientDescentTilingData& actual, bool ok)
{
    int64_t flattened = -1;
    ASSERT_TRUE(optiling::CalcDim0(p.shape, flattened));
    EXPECT_EQ(flattened, p.dim0);

    const OracleRoute route = OracleRouteBranch27(ge::DT_BF16, p.dim0);
    EXPECT_EQ(route.bufferMode, 0);
    EXPECT_EQ(route.tilingKey, 0);
    if (p.dim0 == kSmallDataMax) {
        EXPECT_EQ(OracleRouteBranch27(ge::DT_BF16, kSmallDataMax + 1).tilingKey, 1);
    }
    if (!ok) {
        return;
    }

    const OracleBranch27Tiling expected = OracleComputeBranch27(p.dim0, kAvailableCoreNum, kUbSize);
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
        EXPECT_LE(kWorstBytesPerElem * expected.ubFactor + kUbReserveBytes, static_cast<int64_t>(kUbSize));
        EXPECT_LE(kWorstBytesPerElem * expected.ubFactor + kInstanceFixedBytes, static_cast<int64_t>(kUbSize));
        return;
    }

    EXPECT_EQ(expected.usedCoreNum, 1);
    EXPECT_EQ(expected.blockTail, p.dim0);
    EXPECT_EQ(expected.ubFactor, kUbFactorExpected);
    EXPECT_EQ(expected.ubLoopOfTailBlock, 1);
    EXPECT_EQ(expected.ubTailOfTailBlock, p.dim0);

    // 核与 tile 的 GM 半开区间连续、无重叠，且并集严格为 [0, dim0)。
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

    // GM 仅访问 count*2B；32B 对齐的 rightPadding 只扩展 UB 侧。
    const int64_t count = expected.ubTailOfTailBlock;
    const int64_t rightPadding = (kDataCopyAlignElems - count % kDataCopyAlignElems) % kDataCopyAlignElems;
    EXPECT_EQ(rightPadding, p.expectedRightPadding);
    EXPECT_EQ(gmCursor * kBf16Bytes, p.dim0 * kBf16Bytes);
    EXPECT_EQ((count + rightPadding) * kBf16Bytes % 32, 0);
    EXPECT_LE(count + rightPadding, expected.ubFactor);

    // §4 的 26 B/elem Host 上界与 BF16 SB 实例物理占用上界。
    const int64_t hostUpperBound = kWorstBytesPerElem * expected.ubFactor + kUbReserveBytes;
    const int64_t instancePhysicalBytes = kWorstBytesPerElem * expected.ubFactor + kInstanceFixedBytes;
    EXPECT_EQ(hostUpperBound, 251136);
    EXPECT_LE(hostUpperBound, static_cast<int64_t>(kUbSize));
    EXPECT_EQ(instancePhysicalBytes, 243136);
    EXPECT_LE(instancePhysicalBytes, static_cast<int64_t>(kUbSize));
}

} // namespace

class Branch27TilingTest : public testing::TestWithParam<Branch27Case> {};

TEST_P(Branch27TilingTest, FormulaAndInvariants)
{
    const Branch27Case& p = GetParam();
    const optiling::Branch27TilingInputs in{p.dim0, kAvailableCoreNum, kUbSize};
    InplaceApplyProximalGradientDescentTilingData actual{};
    std::memset(&actual, 0, sizeof(actual));
    const bool ok = optiling::ComputeBranch27Tiling(in, actual);
    EXPECT_TRUE(ok);
    ExpectBranch27Result(p, actual, ok);
}

INSTANTIATE_TEST_SUITE_P(DesignBranch27, Branch27TilingTest, testing::ValuesIn(kCases, kCases + kCaseCount),
                         [](const testing::TestParamInfo<Branch27Case>& info) { return info.param.name; });
