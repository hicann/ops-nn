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
 * inplace_apply_proximal_gradient_descent_package/tests/ut/test_inplace_apply_proximal_gradient_descent_branch0_tiling.cpp
 * =============================================================================
 * Role: DESIGN 13 里程碑 3 / STATE Task 26 —— Branch-0（FP32 SB，runtime key 0,
 *       fp32-small-data-sb）TilingUT（测试先行）。8 例（gtest 参数化），对应
 *       DESIGN-BRANCH-0.md §9.1 表 B0-E0/E1/R0/A0/T0/U0/N0/R16（empty、高 rank
 *       empty、rank 0 单元素、512 对齐、513 非对齐尾、1024 上边界、33 非对齐、
 *       rank 16）。覆盖 runtime key `0` 的 FP32 `dim0<=1024` 单核路由、
 *       `blockFactor∈{512,1024}`、`ubLoopOfTailBlock=1`、`ubTailOfTailBlock=dim0`、
 *       `ubSize=253952` 下 `ubFactor=9344`、GM 半开区间并集恰为 `[0,dim0)`、
 *       padding 只发生在 UB、`26*ubFactor+8192<=ubSize`。另覆盖 Task 31 全量运行
 *       中 4 个 PROFILE_CRASH case 的具体 shape 与混合标量载体，防止实现侧回归。
 *
 * 独立性：期望值全部由独立 oracle（§2.1 empty / §2.2 非空小数据公式手算）给出，
 * 不调用被测 TilingFunc（ComputeBranch0Tiling 尚未实现，stub 阶段期望通过基线
 * = 0，仅要求可编译）。same-source 编译 tiling_host_arch35.cpp，本 UT 测的就是 Task 27
 * 将实现的真实公式。
 * =============================================================================
 */

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <gtest/gtest.h>
#include "../../../../op_host/arch35/inplace_apply_proximal_gradient_descent_tiling_host_arch35.h"

namespace {

// ===== §0/§2/§4 分支常量（手抄自 DESIGN-BRANCH-0，与 oracle 同源）=====
constexpr int64_t kMinSplitThreshold = 1024; // §2 BUFFER_MODE 阈值：dim0<=1024 → mode 0
constexpr int64_t kMinElemsPerCore = 2048;   // §2.2 候选核数阈值
constexpr int64_t kElemAlignFactor = 512;    // §2.2 blockFactor 对齐粒度（元素数）
constexpr uint64_t kUbSize = 253952ULL;      // §4 Ascend 950 每核 UB 字节数
constexpr int64_t kReserveBytes = 8192;      // §4 scalar + 保护余量（字节）
constexpr int64_t kBytesPerElemWorst = 26;   // §4 全部 SB dtype 统一最坏值（B/elem）
constexpr int64_t kAlignElems = 128;         // §2.2 ubFactor 对齐粒度（元素数）
constexpr int64_t kUbFactorExpected = 9344;  // §4 (253952-8192)/26 → AlignDown(,128)
constexpr uint32_t kAvailCoreNum = 80;       // 目标平台可用 AIV 核数

inline int64_t CeilDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }
inline int64_t AlignUp(int64_t v, int64_t f) { return CeilDiv(v, f) * f; }
inline int64_t AlignDown(int64_t v, int64_t f) { return (v / f) * f; }

// ===== 独立路由 oracle（§0/§2 手算）=====
// dtype 由 def 外层二进制承载，runtime key 只编码 BUFFER_MODE；§9.1 表 8 例
// 均为 dim0<=1024，故全部必须命中 key 0 单核路由，无独立 empty key。
struct OracleRoute {
    int64_t bufferMode; // §2 BUFFER_MODE = (dim0<=1024) ? 0 : 1
    int32_t tilingKey;  // §0 TPL(BUFFER_MODE)：mode 0/1 → key 0/1
};

OracleRoute OracleRouteBranch0(int64_t dim0)
{
    const int64_t mode = (dim0 <= kMinSplitThreshold) ? 0 : 1;
    return {mode, static_cast<int32_t>(mode)};
}

// ===== 独立切分 oracle（§2.1 empty / §2.2 非空小数据，手写公式，绝不调被测函数）=====
struct OracleBranch0Tiling {
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

OracleBranch0Tiling OracleComputeBranch0(int64_t dim0, uint32_t availCore, uint64_t ubSize)
{
    OracleBranch0Tiling t{};
    t.dim0 = dim0;
    t.reserved = 0;
    // §2.1 empty：核数固定 1，不执行 UB 容量除法与 loop/tail 计算，其余字段全 0
    if (dim0 == 0) {
        t.usedCoreNum = 1;
        return t;
    }
    // §2.2 非空小数据（1<=dim0<=1024）：
    //   candidateCoreNum=min(avail,max(1,CeilDiv(dim0,2048)))=1
    //   blockFactor=AlignUp(CeilDiv(dim0,candidate),512)∈{512,1024}
    //   usedCoreNum=CeilDiv(dim0,blockFactor)=1；blockTail=dim0
    const int64_t candidate = std::min<int64_t>(static_cast<int64_t>(availCore),
                                                std::max<int64_t>(1, CeilDiv(dim0, kMinElemsPerCore)));
    t.blockFactor = AlignUp(CeilDiv(dim0, candidate), kElemAlignFactor);
    t.usedCoreNum = static_cast<int32_t>(CeilDiv(dim0, t.blockFactor));
    t.blockTail = dim0 - static_cast<int64_t>(t.usedCoreNum - 1) * t.blockFactor;
    // §2.2 UB 预算：ubFactor=AlignDown((ubSize-8192)/26,128)
    t.ubFactor = AlignDown((static_cast<int64_t>(ubSize) - kReserveBytes) / kBytesPerElemWorst, kAlignElems);
    // 非尾核与尾核的 UB loop/tail（blockFactor/blockTail < 9344 → 各 1 次）
    t.ubLoopOfFormerBlock = CeilDiv(t.blockFactor, t.ubFactor);
    t.ubTailOfFormerBlock = t.blockFactor - (t.ubLoopOfFormerBlock - 1) * t.ubFactor;
    t.ubLoopOfTailBlock = CeilDiv(t.blockTail, t.ubFactor);
    t.ubTailOfTailBlock = t.blockTail - (t.ubLoopOfTailBlock - 1) * t.ubFactor;
    return t;
}

// ===== §9.1 表用例（name 同时用于 INSTANTIATE 的用例名）=====
struct Branch0Case {
    const char* name;            // §9.1 表 Case 名
    const char* shape;           // FP32 shape（文档用途）
    int64_t dim0;                // 展平元素总数
    int64_t expectedBlockFactor; // §9.1 表期望（0=empty）
    int64_t expectedRightPad;    // 尾 tile UB 右补（32B=8 个 FP32 对齐，手算）
};

const Branch0Case kCases[] = {
    {"B0E0", "[0]", 0, 0, 0},        {"B0E1", "[2,0,3]", 0, 0, 0},
    {"B0R0", "[]", 1, 512, 7},       {"B0A0", "[512]", 512, 512, 0},
    {"B0T0", "[513]", 513, 1024, 7}, {"B0U0", "[1024]", 1024, 1024, 0},
    {"B0N0", "[33]", 33, 512, 7},    {"B0R16", "[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2]", 2, 512, 6},
};
constexpr size_t kCaseCount = sizeof(kCases) / sizeof(kCases[0]);

// ===== 路由 + 切分 oracle 对照与 §9.1 不变量（仅调 Oracle*，不出现被测函数）=====
void ExpectBranch0Result(const Branch0Case& p, const InplaceApplyProximalGradientDescentTilingData& act, bool ok)
{
    // —— 路由（§0/§2）：8 例全部必须为 key 0 + BUFFER_MODE=0——
    const OracleRoute route = OracleRouteBranch0(p.dim0);
    EXPECT_EQ(route.bufferMode, int64_t{0});
    EXPECT_EQ(route.tilingKey, 0);
    // §9.1 B0-U0 上边界：dim0=1024 为 key 0，1025 起切换 key 1。
    if (p.dim0 == kMinSplitThreshold) {
        EXPECT_EQ(OracleRouteBranch0(kMinSplitThreshold + 1).tilingKey, 1);
    }
    if (!ok || route.tilingKey != 0) {
        return; // stub 阶段 ComputeBranch0Tiling 未实现：跳过逐字段（act 全 0）
    }

    const OracleBranch0Tiling exp = OracleComputeBranch0(p.dim0, kAvailCoreNum, kUbSize);

    // —— 逐字段对照 oracle（§2.1/§2.2 手算期望）——
    EXPECT_EQ(act.dim0, exp.dim0);
    EXPECT_EQ(act.usedCoreNum, exp.usedCoreNum);
    EXPECT_EQ(act.reserved, exp.reserved);
    EXPECT_EQ(act.blockFactor, exp.blockFactor);
    EXPECT_EQ(act.blockTail, exp.blockTail);
    EXPECT_EQ(act.ubFactor, exp.ubFactor);
    EXPECT_EQ(act.ubLoopOfFormerBlock, exp.ubLoopOfFormerBlock);
    EXPECT_EQ(act.ubTailOfFormerBlock, exp.ubTailOfFormerBlock);
    EXPECT_EQ(act.ubLoopOfTailBlock, exp.ubLoopOfTailBlock);
    EXPECT_EQ(act.ubTailOfTailBlock, exp.ubTailOfTailBlock);

    // §9.1 表期望 cross-check
    EXPECT_EQ(exp.blockFactor, p.expectedBlockFactor);

    // §2.1 empty：整个空 Tensor 类只有核数与全零切分
    if (p.dim0 == 0) {
        EXPECT_EQ(exp.usedCoreNum, 1);
        EXPECT_EQ(exp.blockFactor, int64_t{0});
        EXPECT_EQ(exp.blockTail, int64_t{0});
        EXPECT_EQ(exp.ubFactor, int64_t{0});
        EXPECT_EQ(exp.ubLoopOfFormerBlock, int64_t{0});
        EXPECT_EQ(exp.ubTailOfFormerBlock, int64_t{0});
        EXPECT_EQ(exp.ubLoopOfTailBlock, int64_t{0});
        EXPECT_EQ(exp.ubTailOfTailBlock, int64_t{0});
        return;
    }

    // —— 非空不变量（§9.1 末段 + §2.2）——
    // key 0 单核路由：usedCoreNum=1，唯一核即尾核
    EXPECT_EQ(exp.usedCoreNum, 1);
    EXPECT_EQ(exp.blockTail, p.dim0);
    // ubLoopOfTailBlock=1、ubTailOfTailBlock=dim0
    EXPECT_EQ(exp.ubLoopOfTailBlock, int64_t{1});
    EXPECT_EQ(exp.ubTailOfTailBlock, p.dim0);
    // blockFactor∈{512,1024}；ubSize=253952 下 ubFactor=9344
    EXPECT_TRUE(exp.blockFactor == 512 || exp.blockFactor == 1024);
    EXPECT_EQ(exp.ubFactor, kUbFactorExpected);

    // GM 半开区间并集恰为 [0,dim0)：usedCoreNum 个区间
    // [i*blockFactor, i*blockFactor+len) 连续（前核 factor、末核 tail）、无重叠无遗漏
    int64_t cursor = 0;
    for (int32_t core = 0; core < exp.usedCoreNum; ++core) {
        EXPECT_EQ(cursor, static_cast<int64_t>(core) * exp.blockFactor);
        const int64_t len = (core == exp.usedCoreNum - 1) ? exp.blockTail : exp.blockFactor;
        EXPECT_GT(len, 0);
        cursor += len;
    }
    EXPECT_EQ(cursor, exp.dim0);

    // padding 只发生在 UB：GM 有效访问恒为唯一 tile 的 count（尾搬运
    // blockLen=count*4，CopyOut 仅写 count*4 字节）；32B（8 个 FP32）右补仅
    // 落在该 count 之后的 UB 侧，GM 区间并集已验等于 [0,dim0) 故不扩张
    const int64_t count = exp.blockTail;
    EXPECT_EQ(count * static_cast<int64_t>(sizeof(float)), p.dim0 * 4);
    EXPECT_EQ((8 - count % 8) % 8, p.expectedRightPad);
    EXPECT_GE(AlignUp(count, 8), count);

    // UB 预算上界：26*ubFactor + 8192 <= ubSize
    EXPECT_LE(kBytesPerElemWorst * exp.ubFactor + kReserveBytes, static_cast<int64_t>(kUbSize));
}

} // namespace

// =============================================================================
// §9.1 表 8 例：被测 ComputeBranch0Tiling（stub） vs 独立 oracle（§2 手算）
// =============================================================================

class Branch0TilingTest : public testing::TestWithParam<Branch0Case> {};

TEST_P(Branch0TilingTest, FormulaAndInvariants)
{
    const Branch0Case& p = GetParam();
    optiling::Branch0TilingInputs in{p.dim0, kAvailCoreNum, kUbSize};
    InplaceApplyProximalGradientDescentTilingData act{};
    std::memset(&act, 0, sizeof(act)); // stub 阶段函数不填 → 全 0
    const bool ok = optiling::ComputeBranch0Tiling(in, act);
    EXPECT_TRUE(ok); // 本分支 Tiling 未实现：stub 恒 false → 8 例全红（0 passed）
    ExpectBranch0Result(p, act, ok);
}

INSTANTIATE_TEST_SUITE_P(DesignBranch0, Branch0TilingTest, testing::ValuesIn(kCases, kCases + kCaseCount),
                         [](const testing::TestParamInfo<Branch0Case>& info) { return info.param.name; });

TEST(Branch0TilingTest, Task31ProfileCrashShapesAndScalarCarriers)
{
    struct ProfileCase {
        gert::Shape var;
        gert::Shape alpha;
        gert::Shape l1;
        gert::Shape l2;
        int64_t dim0;
    };
    const ProfileCase cases[] = {
        {gert::Shape({4, 17, 7}), gert::Shape({1}), gert::Shape({1}), gert::Shape({1}), 476},
        {gert::Shape({2, 4, 7}), gert::Shape({1}), gert::Shape({1}), gert::Shape({1}), 56},
        {gert::Shape({1}), gert::Shape({1}), gert::Shape{}, gert::Shape({1}), 1},
        {gert::Shape({5}), gert::Shape({1}), gert::Shape{}, gert::Shape({1}), 5},
    };

    for (const auto& p : cases) {
        int64_t dim0 = -1;
        ASSERT_TRUE(optiling::CalcDim0(p.var, dim0));
        EXPECT_EQ(dim0, p.dim0);
        EXPECT_TRUE(optiling::ExactShapeEqual(p.var, p.var));
        EXPECT_TRUE(optiling::IsSharedScalarShape(p.alpha));
        EXPECT_TRUE(optiling::IsSharedScalarShape(p.l1));
        EXPECT_TRUE(optiling::IsSharedScalarShape(p.l2));

        InplaceApplyProximalGradientDescentTilingData td{};
        ASSERT_TRUE(optiling::ComputeBranch0Tiling({dim0, kAvailCoreNum, kUbSize}, td));
        EXPECT_EQ(td.dim0, p.dim0);
        EXPECT_EQ(td.usedCoreNum, 1);
        EXPECT_EQ(td.blockFactor, 512);
        EXPECT_EQ(td.blockTail, p.dim0);
        EXPECT_EQ(td.ubFactor, kUbFactorExpected);
        EXPECT_EQ(td.ubLoopOfTailBlock, 1);
        EXPECT_EQ(td.ubTailOfTailBlock, p.dim0);
    }
}
