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
 * Branch-257（历史分支名，FP16 DB，runtime key 1）测试先行 TilingUT。
 * DESIGN-BRANCH-257 §9.1 的 B257-L0/X0/C0/A0/P08/U0/R16 共 7 例；
 * 期望值由本文件按 §2 独立手算，不调用被测 Tiling 函数。
 */

#include <algorithm>
#include <cstdint>
#include <gtest/gtest.h>
#include "graph/types.h"
#include "../../../../op_host/arch35/inplace_apply_proximal_gradient_descent_tiling_host_arch35.h"

namespace {

// DESIGN-BRANCH-257 §0/§2/§4 常量。
constexpr int64_t kSmallDataMax = 1024;
constexpr int64_t kMinElemsPerCore = 2048;
constexpr int64_t kBlockAlignElems = 512;
constexpr uint64_t kUbSize = 253952ULL;
constexpr int64_t kUbReserveBytes = 8192;
constexpr int64_t kWorstBytesPerElem = 36;
constexpr int64_t kFp16DbBytesPerElem = 32;
constexpr int64_t kInstanceFixedBytes = 192;
constexpr int64_t kUbAlignElems = 128;
constexpr int64_t kUbFactorExpected = 6784;
constexpr int64_t kFp16DataCopyAlignElems = 16;
constexpr int64_t kMinRecommendedDmaBytes = 16384;

inline int64_t CeilDiv(int64_t value, int64_t divisor)
{
    return value / divisor + static_cast<int64_t>(value % divisor != 0);
}

inline int64_t AlignUp(int64_t value, int64_t align) { return CeilDiv(value, align) * align; }

inline int64_t AlignDown(int64_t value, int64_t align) { return (value / align) * align; }

int64_t OracleFlatten(const gert::Shape& shape)
{
    int64_t dim0 = 1;
    for (size_t i = 0; i < shape.GetDimNum(); ++i) {
        dim0 *= shape.GetDim(i);
    }
    return dim0;
}

struct OracleRoute {
    int64_t bufferMode;
    int32_t tilingKey;
};

// §0：FP16 由 def 外层二进制承载，runtime key 只等于 BUFFER_MODE。
OracleRoute OracleRouteBranch257(ge::DataType input0Type, int64_t dim0)
{
    const int64_t bufferMode = (dim0 <= kSmallDataMax) ? 0 : 1;
    if (input0Type != ge::DT_FLOAT16) {
        return {bufferMode, -1};
    }
    return {bufferMode, static_cast<int32_t>(bufferMode)};
}

struct OracleBranch257Tiling {
    int64_t dim0;
    int64_t candidateCoreNum;
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
OracleBranch257Tiling OracleComputeBranch257(int64_t dim0, uint32_t availableCoreNum, uint64_t ubSize)
{
    OracleBranch257Tiling t{};
    t.dim0 = dim0;
    t.candidateCoreNum = std::min<int64_t>(static_cast<int64_t>(availableCoreNum),
                                           std::max<int64_t>(1, CeilDiv(dim0, kMinElemsPerCore)));
    t.blockFactor = AlignUp(CeilDiv(dim0, t.candidateCoreNum), kBlockAlignElems);
    t.usedCoreNum = static_cast<int32_t>(CeilDiv(dim0, t.blockFactor));
    t.blockTail = dim0 - static_cast<int64_t>(t.usedCoreNum - 1) * t.blockFactor;
    t.ubFactor = AlignDown((static_cast<int64_t>(ubSize) - kUbReserveBytes) / kWorstBytesPerElem, kUbAlignElems);
    t.ubLoopOfFormerBlock = CeilDiv(t.blockFactor, t.ubFactor);
    t.ubTailOfFormerBlock = t.blockFactor - (t.ubLoopOfFormerBlock - 1) * t.ubFactor;
    t.ubLoopOfTailBlock = CeilDiv(t.blockTail, t.ubFactor);
    t.ubTailOfTailBlock = t.blockTail - (t.ubLoopOfTailBlock - 1) * t.ubFactor;
    return t;
}

struct Branch257Case {
    const char* name;
    gert::Shape shape;
    int64_t dim0;
    uint32_t availableCoreNum;
    bool belongsToBranch;
    int64_t expectedCandidateCoreNum;
    int32_t expectedUsedCoreNum;
    int64_t expectedBlockFactor;
    int64_t expectedBlockTail;
    int64_t expectedFormerLoop;
    int64_t expectedFormerTail;
    int64_t expectedTailLoop;
    int64_t expectedTailTail;
    int64_t expectedRightPadding;
};

const Branch257Case kCases[] = {
    {"B257L0", gert::Shape({1025}), 1025, 80, true, 1, 1, 1536, 1025, 1, 1536, 1, 1025, 15},
    {"B257X0", gert::Shape({1024}), 1024, 80, false, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {"B257C0", gert::Shape({2049}), 2049, 80, true, 2, 2, 1536, 513, 1, 1536, 1, 513, 15},
    {"B257A0", gert::Shape({4096}), 4096, 80, true, 2, 2, 2048, 2048, 1, 2048, 1, 2048, 0},
    {"B257P08", gert::Shape({257, 129}), 33153, 80, true, 17, 17, 2048, 385, 1, 2048, 1, 385, 15},
    {"B257U0", gert::Shape({600000}), 600000, 80, true, 80, 79, 7680, 960, 2, 896, 1, 960, 0},
    {"B257R16", gert::Shape({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1025}), 1025, 80, true, 1, 1, 1536, 1025, 1,
     1536, 1, 1025, 15},
};
constexpr size_t kCaseCount = sizeof(kCases) / sizeof(kCases[0]);

InplaceApplyProximalGradientDescentTilingData MakeSentinelTilingData()
{
    InplaceApplyProximalGradientDescentTilingData td{};
    td.dim0 = -201;
    td.usedCoreNum = -202;
    td.reserved = -203;
    td.blockFactor = -204;
    td.blockTail = -205;
    td.ubFactor = -206;
    td.ubLoopOfFormerBlock = -207;
    td.ubTailOfFormerBlock = -208;
    td.ubLoopOfTailBlock = -209;
    td.ubTailOfTailBlock = -210;
    return td;
}

void ExpectSameTilingData(const InplaceApplyProximalGradientDescentTilingData& actual,
                          const InplaceApplyProximalGradientDescentTilingData& expected)
{
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
}

template <typename T>
void ExpectCompletePartition(const T& t)
{
    ASSERT_GT(t.usedCoreNum, 0);
    ASSERT_GT(t.blockFactor, 0);
    ASSERT_GT(t.blockTail, 0);
    ASSERT_GT(t.ubFactor, 0);

    int64_t gmCursor = 0;
    for (int32_t core = 0; core < t.usedCoreNum; ++core) {
        const int64_t coreStart = static_cast<int64_t>(core) * t.blockFactor;
        const bool isTailCore = core == t.usedCoreNum - 1;
        const int64_t coreLength = isTailCore ? t.blockTail : t.blockFactor;
        const int64_t loopCount = isTailCore ? t.ubLoopOfTailBlock : t.ubLoopOfFormerBlock;
        const int64_t tailCount = isTailCore ? t.ubTailOfTailBlock : t.ubTailOfFormerBlock;
        EXPECT_EQ(gmCursor, coreStart);
        EXPECT_GT(coreLength, 0);
        EXPECT_LE(coreLength, t.blockFactor);
        EXPECT_GT(loopCount, 0);
        for (int64_t loop = 0; loop < loopCount; ++loop) {
            const int64_t tileStart = coreStart + loop * t.ubFactor;
            const int64_t count = (loop == loopCount - 1) ? tailCount : t.ubFactor;
            const int64_t rightPadding = (kFp16DataCopyAlignElems - count % kFp16DataCopyAlignElems) %
                                         kFp16DataCopyAlignElems;
            EXPECT_EQ(gmCursor, tileStart);
            EXPECT_GT(count, 0);
            EXPECT_LE(count, t.ubFactor);
            EXPECT_LE(tileStart + count, coreStart + coreLength);
            EXPECT_EQ((count + rightPadding) % kFp16DataCopyAlignElems, 0);
            EXPECT_LE(count + rightPadding, t.ubFactor);
            gmCursor += count;
        }
        EXPECT_EQ(gmCursor, coreStart + coreLength);
    }
    EXPECT_EQ(gmCursor, t.dim0);
}

void ExpectBranch257Result(const Branch257Case& p, const InplaceApplyProximalGradientDescentTilingData& actual,
                           const InplaceApplyProximalGradientDescentTilingData& before, bool ok)
{
    const int64_t flattened = OracleFlatten(p.shape);
    EXPECT_EQ(flattened, p.dim0);
    const OracleRoute route = OracleRouteBranch257(ge::DT_FLOAT16, flattened);
    EXPECT_EQ(route.bufferMode, p.belongsToBranch ? 1 : 0);
    EXPECT_EQ(route.tilingKey, p.belongsToBranch ? 1 : 0);

    if (!p.belongsToBranch) {
        EXPECT_EQ(p.dim0, kSmallDataMax);
        EXPECT_FALSE(ok);
        ExpectSameTilingData(actual, before);
        return;
    }

    EXPECT_GT(p.dim0, kSmallDataMax);
    EXPECT_TRUE(ok);
    const OracleBranch257Tiling expected = OracleComputeBranch257(p.dim0, p.availableCoreNum, kUbSize);

    EXPECT_EQ(expected.candidateCoreNum, p.expectedCandidateCoreNum);
    EXPECT_EQ(expected.usedCoreNum, p.expectedUsedCoreNum);
    EXPECT_EQ(expected.blockFactor, p.expectedBlockFactor);
    EXPECT_EQ(expected.blockTail, p.expectedBlockTail);
    EXPECT_EQ(expected.ubLoopOfFormerBlock, p.expectedFormerLoop);
    EXPECT_EQ(expected.ubTailOfFormerBlock, p.expectedFormerTail);
    EXPECT_EQ(expected.ubLoopOfTailBlock, p.expectedTailLoop);
    EXPECT_EQ(expected.ubTailOfTailBlock, p.expectedTailTail);
    EXPECT_EQ(expected.ubFactor, kUbFactorExpected);
    EXPECT_LE(expected.usedCoreNum, expected.candidateCoreNum);
    EXPECT_LE(expected.candidateCoreNum, static_cast<int64_t>(p.availableCoreNum));
    EXPECT_EQ(expected.blockFactor % kBlockAlignElems, 0);
    EXPECT_GT(expected.blockTail, 0);
    EXPECT_LE(expected.blockTail, expected.blockFactor);

    // §4 Host DB 上界与本 FP16 DB 实例物理占用均须落在目标 UB 内。
    const int64_t hostUpperBound = kWorstBytesPerElem * expected.ubFactor + kUbReserveBytes;
    const int64_t instancePhysicalBytes = kFp16DbBytesPerElem * expected.ubFactor + kInstanceFixedBytes;
    EXPECT_EQ(hostUpperBound, 252416);
    EXPECT_LE(hostUpperBound, static_cast<int64_t>(kUbSize));
    EXPECT_EQ(instancePhysicalBytes, 217280);
    EXPECT_LE(instancePhysicalBytes, static_cast<int64_t>(kUbSize));
    const int64_t oneDirectionBytes = expected.ubFactor * 2;
    EXPECT_EQ(oneDirectionBytes, 13568);
    EXPECT_EQ(3 * oneDirectionBytes, 40704);
    EXPECT_LT(oneDirectionBytes, kMinRecommendedDmaBytes);
    EXPECT_GT(8192 * kFp16DbBytesPerElem + kInstanceFixedBytes, static_cast<int64_t>(kUbSize));

    ExpectCompletePartition(expected);
    if (p.dim0 == 1025 || p.dim0 == 2049 || p.dim0 == 33153) {
        EXPECT_EQ(expected.ubLoopOfFormerBlock, 1);
        EXPECT_EQ(expected.ubLoopOfTailBlock, 1);
    }
    if (p.dim0 == 600000) {
        EXPECT_GT(expected.ubLoopOfFormerBlock, 1);
        EXPECT_EQ(expected.ubLoopOfTailBlock, 1);
    }
    EXPECT_EQ(
        (kFp16DataCopyAlignElems - expected.ubTailOfTailBlock % kFp16DataCopyAlignElems) % kFp16DataCopyAlignElems,
        p.expectedRightPadding);
    if (p.expectedCandidateCoreNum == 80) {
        EXPECT_EQ(expected.usedCoreNum, 79);
        EXPECT_LT(expected.usedCoreNum, expected.candidateCoreNum);
    }
    if (p.shape.GetDimNum() == 16) {
        EXPECT_STREQ(p.name, "B257R16");
        EXPECT_EQ(expected.dim0, 1025);
    }

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
    ExpectCompletePartition(actual);
}

} // namespace

class Branch257TilingTest : public testing::TestWithParam<Branch257Case> {};

TEST_P(Branch257TilingTest, FormulaAndInvariants)
{
    const Branch257Case& p = GetParam();
    const optiling::Branch257TilingInputs in{p.dim0, p.availableCoreNum, kUbSize};
    InplaceApplyProximalGradientDescentTilingData actual = MakeSentinelTilingData();
    const InplaceApplyProximalGradientDescentTilingData before = actual;
    const bool ok = optiling::ComputeBranch257Tiling(in, actual);
    ExpectBranch257Result(p, actual, before, ok);
}

INSTANTIATE_TEST_SUITE_P(DesignBranch257, Branch257TilingTest, testing::ValuesIn(kCases, kCases + kCaseCount),
                         [](const testing::TestParamInfo<Branch257Case>& info) { return info.param.name; });
