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
 * Branch-256（历史分支名，FP32 DB，runtime key 1）测试先行 TilingUT。
 * DESIGN-BRANCH-256 §9.1 的 B256-L0/X0/C0/A0/P08/U0/R16 共 7 例；
 * 期望值由本文件按 §2 独立手算，不调用被测 Tiling 函数。
 */

#include <algorithm>
#include <cstdint>
#include <gtest/gtest.h>
#include "graph/types.h"
#include "../../../../op_host/arch35/inplace_apply_proximal_gradient_descent_tiling_host_arch35.h"

namespace {

// DESIGN-BRANCH-256 §0/§2/§4 常量。
constexpr int64_t kSmallDataMax = 1024;
constexpr int64_t kMinElemsPerCore = 2048;
constexpr int64_t kBlockAlignElems = 512;
constexpr uint64_t kUbSize = 253952ULL;
constexpr int64_t kUbReserveBytes = 8192;
constexpr int64_t kWorstBytesPerElem = 36;
constexpr int64_t kInstanceFixedBytes = 192;
constexpr int64_t kUbAlignElems = 128;
constexpr int64_t kUbFactorExpected = 6784;
constexpr int64_t kFp32DataCopyAlignElems = 8;

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

// §0：FP32 由 def 外层二进制承载，runtime key 只等于 BUFFER_MODE。
OracleRoute OracleRouteBranch256(ge::DataType input0Type, int64_t dim0)
{
    const int64_t bufferMode = (dim0 <= kSmallDataMax) ? 0 : 1;
    if (input0Type != ge::DT_FLOAT) {
        return {bufferMode, -1};
    }
    return {bufferMode, static_cast<int32_t>(bufferMode)};
}

struct OracleBranch256Tiling {
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
OracleBranch256Tiling OracleComputeBranch256(int64_t dim0, uint32_t availableCoreNum, uint64_t ubSize)
{
    OracleBranch256Tiling t{};
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

struct Branch256Case {
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

const Branch256Case kCases[] = {
    {"B256L0", gert::Shape({1025}), 1025, 80, true, 1, 1, 1536, 1025, 1, 1536, 1, 1025, 7},
    {"B256X0", gert::Shape({1024}), 1024, 80, false, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    {"B256C0", gert::Shape({2049}), 2049, 80, true, 2, 2, 1536, 513, 1, 1536, 1, 513, 7},
    {"B256A0", gert::Shape({4096}), 4096, 80, true, 2, 2, 2048, 2048, 1, 2048, 1, 2048, 0},
    {"B256P08", gert::Shape({257, 129}), 33153, 80, true, 17, 17, 2048, 385, 1, 2048, 1, 385, 7},
    {"B256U0", gert::Shape({600000}), 600000, 80, true, 80, 79, 7680, 960, 2, 896, 1, 960, 0},
    {"B256R16", gert::Shape({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1025}), 1025, 80, true, 1, 1, 1536, 1025, 1,
     1536, 1, 1025, 7},
};
constexpr size_t kCaseCount = sizeof(kCases) / sizeof(kCases[0]);

InplaceApplyProximalGradientDescentTilingData MakeSentinelTilingData()
{
    InplaceApplyProximalGradientDescentTilingData td{};
    td.dim0 = -101;
    td.usedCoreNum = -102;
    td.reserved = -103;
    td.blockFactor = -104;
    td.blockTail = -105;
    td.ubFactor = -106;
    td.ubLoopOfFormerBlock = -107;
    td.ubTailOfFormerBlock = -108;
    td.ubLoopOfTailBlock = -109;
    td.ubTailOfTailBlock = -110;
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
            const int64_t rightPadding = (kFp32DataCopyAlignElems - count % kFp32DataCopyAlignElems) %
                                         kFp32DataCopyAlignElems;
            EXPECT_EQ(gmCursor, tileStart);
            EXPECT_GT(count, 0);
            EXPECT_LE(count, t.ubFactor);
            EXPECT_LE(tileStart + count, coreStart + coreLength);
            EXPECT_EQ((count + rightPadding) % kFp32DataCopyAlignElems, 0);
            EXPECT_LE(count + rightPadding, t.ubFactor);
            gmCursor += count;
        }
        EXPECT_EQ(gmCursor, coreStart + coreLength);
    }
    EXPECT_EQ(gmCursor, t.dim0);
}

void ExpectBranch256Result(const Branch256Case& p, const InplaceApplyProximalGradientDescentTilingData& actual,
                           const InplaceApplyProximalGradientDescentTilingData& before, bool ok)
{
    const int64_t flattened = OracleFlatten(p.shape);
    EXPECT_EQ(flattened, p.dim0);
    const OracleRoute route = OracleRouteBranch256(ge::DT_FLOAT, flattened);
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
    const OracleBranch256Tiling expected = OracleComputeBranch256(p.dim0, p.availableCoreNum, kUbSize);

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

    // §4 Host DB 上界与本 FP32 DB 实例物理占用均须落在目标 UB 内。
    const int64_t hostUpperBound = kWorstBytesPerElem * expected.ubFactor + kUbReserveBytes;
    const int64_t instancePhysicalBytes = kWorstBytesPerElem * expected.ubFactor + kInstanceFixedBytes;
    EXPECT_EQ(hostUpperBound, 252416);
    EXPECT_LE(hostUpperBound, static_cast<int64_t>(kUbSize));
    EXPECT_EQ(instancePhysicalBytes, 244416);
    EXPECT_LE(instancePhysicalBytes, static_cast<int64_t>(kUbSize));

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
        (kFp32DataCopyAlignElems - expected.ubTailOfTailBlock % kFp32DataCopyAlignElems) % kFp32DataCopyAlignElems,
        p.expectedRightPadding);
    if (p.expectedCandidateCoreNum == 80) {
        EXPECT_EQ(expected.usedCoreNum, 79);
        EXPECT_LT(expected.usedCoreNum, expected.candidateCoreNum);
    }
    if (p.shape.GetDimNum() == 16) {
        EXPECT_EQ(p.name, std::string("B256R16"));
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

class Branch256TilingTest : public testing::TestWithParam<Branch256Case> {};

TEST_P(Branch256TilingTest, FormulaAndInvariants)
{
    const Branch256Case& p = GetParam();
    const optiling::Branch256TilingInputs in{p.dim0, p.availableCoreNum, kUbSize};
    InplaceApplyProximalGradientDescentTilingData actual = MakeSentinelTilingData();
    const InplaceApplyProximalGradientDescentTilingData before = actual;
    const bool ok = optiling::ComputeBranch256Tiling(in, actual);
    ExpectBranch256Result(p, actual, before, ok);
}

INSTANTIATE_TEST_SUITE_P(DesignBranch256, Branch256TilingTest, testing::ValuesIn(kCases, kCases + kCaseCount),
                         [](const testing::TestParamInfo<Branch256Case>& info) { return info.param.name; });
