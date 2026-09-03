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
 * inplace_apply_proximal_gradient_descent_package/tests/ut/test_inplace_apply_proximal_gradient_descent_tiling.cpp
 * =============================================================================
 * Role: DESIGN 13 里程碑 2 / STATE Task 24 —— 公共 TilingUT（测试先行）。
 *       12 个 gtest 用例，覆盖公共 Host Tiling 纯公式（DESIGN §9.3/§9.4/§9.5/§9.7）：
 *         - §9.3 展平/逐维同形/标量载体 shape 校验 4 例（含 0-D、[1]、[1,1] 非法、
 *           零维、INT64 溢出）；
 *         - §9.4/§9.5 平台与 UB 预算及多核与溢出边界 4 例（empty、单核、未饱和、
 *           核数饱和、INT64_MAX、对齐后负 tail 反例、对齐不可表示）；
 *         - §9.7 typed TilingData/workspace/SetBlockDim 提交及失败副作用 4 例。
 *
 * 独立性：期望值全部为按 DESIGN §9 公式手算的常量（§9.5 边界证明表 / §9.9
 * 失败点表手写），不调用被测 TilingFunc（TilingFunc 尚未实现，stub 阶段
 * 期望通过基线 = 0，仅要求可编译）。same-source 编译 tiling_host_arch35.cpp，本 UT
 * 测的就是 Task 25 将实现的真实公式。
 * =============================================================================
 */

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>
#include <gtest/gtest.h>
#include "../../../../op_host/arch35/inplace_apply_proximal_gradient_descent_tiling_host_arch35.h"

namespace {

constexpr int64_t kInt64Max = std::numeric_limits<int64_t>::max();

// §9.4/§9.5 常量（手抄自 DESIGN §9.4/§9.5，与 oracle 同源）
constexpr uint64_t kUbSize = 253952ULL;    // Ascend 950 实测 UB 字节数
constexpr int64_t kUbFactorSb = 9344;      // (253952-8192)/26 → 对齐 128
constexpr int64_t kUbFactorDb = 6784;      // (253952-8192)/36 → 对齐 128
constexpr int64_t kMinElemsPerCore = 2048; // §9.5 候选核阈值
constexpr int64_t kElemAlignFactor = 512;  // §9.5 blockFactor 对齐粒度

// ===== 多核 oracle（§9.5 公式手算，逐字段常量）=====
struct MultiCoreExpect {
    bool ok;
    int32_t usedCoreNum;
    int64_t blockFactor;
    int64_t blockTail;
};

void ExpectMultiCore(int64_t dim0, uint32_t avail, const MultiCoreExpect& exp)
{
    int32_t used = -1;
    int64_t factor = -1;
    int64_t tail = -1;
    const bool ok = optiling::CalcMultiCore(dim0, avail, used, factor, tail);
    if (exp.ok) {
        EXPECT_TRUE(ok);
        if (!ok) {
            return; // stub 阶段返回哨兵，跳过后续逐值断言（避免哨兵值参与算术）
        }
        EXPECT_EQ(used, exp.usedCoreNum);
        EXPECT_EQ(factor, exp.blockFactor);
        EXPECT_EQ(tail, exp.blockTail);
        EXPECT_GT(tail, 0);
        EXPECT_LE(tail, factor);
        EXPECT_LE(used, static_cast<int32_t>(avail));
        // 前 used-1 核各 factor 元素 + 尾核 tail = dim0（半开区间并集恰为 [0, dim0)）；
        // 用 (used-1)*factor+tail 形式避免 used*factor 在 INT64_MAX 用例上溢出
        EXPECT_EQ(static_cast<int64_t>(used - 1) * factor + tail, dim0);
    } else {
        EXPECT_FALSE(ok);
        // 失败路径：三个 out 参数保持调用前取值（-1）
        EXPECT_EQ(used, int32_t{-1});
        EXPECT_EQ(factor, int64_t{-1});
        EXPECT_EQ(tail, int64_t{-1});
    }
}

// ===== §9.7 提交 harness（回调计数 + 可注入 SetBlockDim 失败）=====
bool RunCommit(InplaceApplyProximalGradientDescentTilingData* td, size_t* workspace, std::vector<uint32_t>* calls,
               bool blockDimSucceeds)
{
    optiling::SetBlockDimFn fn = [calls, blockDimSucceeds](uint32_t usedCoreNum) {
        calls->push_back(usedCoreNum);
        return blockDimSucceeds;
    };
    return optiling::CommitTilingData(td, workspace, fn,
                                      /*dim0*/ 4096, /*usedCoreNum*/ 2, /*blockFactor*/ 2048,
                                      /*blockTail*/ 2048, /*ubFactor*/ 9344,
                                      /*ubLoopOfFormerBlock*/ 1, /*ubTailOfFormerBlock*/ 2048,
                                      /*ubLoopOfTailBlock*/ 1, /*ubTailOfTailBlock*/ 2048);
}

// dim0=4096/SB 的 10 个字段期望值（§9.1 oracle，手算）
void ExpectAllFields(const InplaceApplyProximalGradientDescentTilingData& td)
{
    EXPECT_EQ(td.dim0, 4096);
    EXPECT_EQ(td.usedCoreNum, 2);
    EXPECT_EQ(td.reserved, 0);
    EXPECT_EQ(td.blockFactor, 2048);
    EXPECT_EQ(td.blockTail, 2048);
    EXPECT_EQ(td.ubFactor, 9344);
    EXPECT_EQ(td.ubLoopOfFormerBlock, 1);
    EXPECT_EQ(td.ubTailOfFormerBlock, 2048);
    EXPECT_EQ(td.ubLoopOfTailBlock, 1);
    EXPECT_EQ(td.ubTailOfTailBlock, 2048);
}

} // namespace

// =============================================================================
// §9.3 展平/同形/标量载体 shape 校验（4 例）
// =============================================================================

// 展平：0-D 空乘积=1；[1] =1；一般多维连乘；零维（含高 rank）短路为 0。
TEST(APGDCommonTiling, CalcDim0Flatten) // 例 1
{
    const gert::Shape rank0{};
    int64_t dim0 = -1;
    EXPECT_TRUE(optiling::CalcDim0(rank0, dim0));
    EXPECT_EQ(dim0, int64_t{1});

    dim0 = -1;
    EXPECT_TRUE(optiling::CalcDim0(gert::Shape({1}), dim0));
    EXPECT_EQ(dim0, int64_t{1});

    dim0 = -1;
    EXPECT_TRUE(optiling::CalcDim0(gert::Shape({2, 3}), dim0));
    EXPECT_EQ(dim0, int64_t{6});

    dim0 = -1;
    EXPECT_TRUE(optiling::CalcDim0(gert::Shape({257, 129}), dim0));
    EXPECT_EQ(dim0, int64_t{33153}); // §9.3 逐维相乘

    dim0 = -1;
    EXPECT_TRUE(optiling::CalcDim0(gert::Shape({2, 0, 3}), dim0));
    EXPECT_EQ(dim0, int64_t{0}); // 任一维为 0 → dim0=0（仍合法）

    dim0 = -1;
    EXPECT_TRUE(optiling::CalcDim0(gert::Shape({7}), dim0));
    EXPECT_EQ(dim0, int64_t{7});
}

// 展平失败：INT64 乘法溢出与未知/非法负维；断言失败后 dim0 的 §9.3 副作用语义。
TEST(APGDCommonTiling, CalcDim0OverflowAndUnknownDim) // 例 2
{
    int64_t dim0 = 0;
    EXPECT_TRUE(optiling::CalcDim0(gert::Shape({1024}), dim0));
    EXPECT_EQ(dim0, int64_t{1024});

    dim0 = 0;
    EXPECT_TRUE(optiling::CalcDim0(gert::Shape({kInt64Max}), dim0));
    EXPECT_EQ(dim0, kInt64Max);

    // 溢出：乘法前检查 dim0 > INT64_MAX/dim 失败；dim0 停在已完成的部分积
    dim0 = 0;
    EXPECT_FALSE(optiling::CalcDim0(gert::Shape({kInt64Max, 2}), dim0));
    EXPECT_EQ(dim0, kInt64Max);

    // 负维（Tiling 时仍未知或非法）：首轮扫描即 return false，dim0 保持调用前取值
    dim0 = 0;
    EXPECT_FALSE(optiling::CalcDim0(gert::Shape({3, 5, -1}), dim0));
    EXPECT_EQ(dim0, int64_t{0});
}

// 逐维完全同形：仅 numel 相等不算同形；rank 相同且逐维相等才 true。
TEST(APGDCommonTiling, ExactShapeEqualCase) // 例 3
{
    EXPECT_TRUE(optiling::ExactShapeEqual(gert::Shape({2, 3}), gert::Shape({2, 3})));
    EXPECT_FALSE(optiling::ExactShapeEqual(gert::Shape({2, 3}), gert::Shape({6}))); // numel 相同但 rank 不同
    EXPECT_FALSE(optiling::ExactShapeEqual(gert::Shape({2, 3}), gert::Shape({2, 4})));
    EXPECT_FALSE(optiling::ExactShapeEqual(gert::Shape({}), gert::Shape({1}))); // 0-D vs [1]，numel 均为 1
    EXPECT_TRUE(
        optiling::ExactShapeEqual(gert::Shape({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2}),
                                  gert::Shape({1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2}))); // rank 16 上界
}

// 标量载体：仅 0-D [] 或一维 [1] 合法；[1,1]/[0]/[2]/三维全 1 均非法。
TEST(APGDCommonTiling, IsSharedScalarShapeCase) // 例 4
{
    EXPECT_TRUE(optiling::IsSharedScalarShape(gert::Shape{}));
    EXPECT_TRUE(optiling::IsSharedScalarShape(gert::Shape({1})));
    EXPECT_FALSE(optiling::IsSharedScalarShape(gert::Shape({1, 1}))); // §9.3 明确拒绝
    EXPECT_FALSE(optiling::IsSharedScalarShape(gert::Shape({0})));    // 零元素载体非法
    EXPECT_FALSE(optiling::IsSharedScalarShape(gert::Shape({2})));
    EXPECT_FALSE(optiling::IsSharedScalarShape(gert::Shape({1, 1, 1})));
}

// =============================================================================
// §9.4/§9.5 平台与 UB 预算及多核与溢出边界（4 例）
// =============================================================================

// UB 预算（SB=26 与 DB=36 B/elem 统一最坏情况，128 对齐）与 empty 短路。
TEST(APGDCommonTiling, UbBudgetAndEmptyShortCircuit) // 例 5
{
    int64_t ubFactor = -1;
    EXPECT_TRUE(optiling::CalcUbFactor(kUbSize, 0, ubFactor));
    EXPECT_EQ(ubFactor, kUbFactorSb); // (253952-8192)/26=9452 → AlignDown(,128)=9344
    ubFactor = -1;
    EXPECT_TRUE(optiling::CalcUbFactor(kUbSize, 1, ubFactor));
    EXPECT_EQ(ubFactor, kUbFactorDb); // (253952-8192)/36=6826 → AlignDown(,128)=6784

    // 失败路径：ubSize<=8192（保留量不足）、对齐后 ubFactor==0、mode 越界、ubSize 超 int64
    ubFactor = -1;
    EXPECT_FALSE(optiling::CalcUbFactor(8192, 0, ubFactor));
    EXPECT_EQ(ubFactor, int64_t{-1}); // 未写入
    ubFactor = 1234;
    EXPECT_FALSE(optiling::CalcUbFactor(8193, 0, ubFactor));
    EXPECT_EQ(ubFactor, int64_t{0}); // §9.4 行序：先赋值 0 再返回 false
    ubFactor = -1;
    EXPECT_FALSE(optiling::CalcUbFactor(kUbSize, 2, ubFactor));
    EXPECT_EQ(ubFactor, int64_t{-1});
    ubFactor = -1;
    EXPECT_FALSE(optiling::CalcUbFactor(1ULL << 63, 0, ubFactor));
    EXPECT_EQ(ubFactor, int64_t{-1});

    // empty（§9.5：dim0=0 不调多核/loop 公式，直接短路；helper 侧为守卫返回 false）
    ExpectMultiCore(0, 80, {false, 0, 0, 0});
    int64_t loop = -1;
    int64_t tail = -1;
    EXPECT_FALSE(optiling::CalcLoopTail(0, kUbFactorSb, loop, tail));
    EXPECT_EQ(loop, int64_t{-1});
    EXPECT_EQ(tail, int64_t{-1});
    EXPECT_TRUE(optiling::CalcLoopTail(2048, kUbFactorSb, loop, tail));
    EXPECT_EQ(loop, int64_t{1});
    EXPECT_EQ(tail, int64_t{2048});
}

// 单核与未饱和多核：候选核→512 对齐 blockFactor→实际核数重算。
TEST(APGDCommonTiling, SingleCoreAndUnsaturatedCores) // 例 6
{
    // dim0=1：候选=1，blockFactor=AlignUp(1,512)=512，used=1，tail=1（§9.5 单元素行）
    ExpectMultiCore(1, 80, {true, 1, 512, 1});
    // dim0=4096：候选=2，factor=2048，used=2，tail=2048（§9.5 未饱和行）
    ExpectMultiCore(4096, 80, {true, 2, 2048, 2048});
    // dim0=2049：512 元素公共对齐后的完整区间为 [0,1536) 与 [1536,2049)。
    ExpectMultiCore(2049, 80, {true, 2, 1536, 513});
    // dim0=4097：候选=3，raw=1366 → factor=1536，used=3，tail=4097-2*1536=1025
    ExpectMultiCore(4097, 80, {true, 3, 1536, 1025});

    // 正整数 helper 直接校验（§9.5 无溢出 CeilDiv/AlignUp）
    int64_t r = -1;
    EXPECT_TRUE(optiling::CeilDivPositive(4097, 2048, r));
    EXPECT_EQ(r, int64_t{3});
    r = -1;
    EXPECT_TRUE(optiling::AlignUpPositive(1366, kElemAlignFactor, r));
    EXPECT_EQ(r, int64_t{1536});
    r = -1;
    EXPECT_FALSE(optiling::CeilDivPositive(0, 1, r));
    EXPECT_EQ(r, int64_t{-1});

    // UB loop/tail（factor=9344）
    int64_t loop = -1;
    int64_t tail = -1;
    EXPECT_TRUE(optiling::CalcLoopTail(20480, kUbFactorSb, loop, tail));
    EXPECT_EQ(loop, int64_t{3});
    EXPECT_EQ(tail, int64_t{20480 - 2 * kUbFactorSb}); // 1792，0<tail<=factor
}

// 核数饱和与“对齐后减少实际核数”反例：163841 不得沿用候选 80 产生负 tail。
TEST(APGDCommonTiling, CoreSaturationAndAlignedTailFix) // 例 7
{
    // 核数饱和（§9.5 表）：dim0=163840 得候选=80、factor=2048、used=80、tail=2048
    ExpectMultiCore(163840, 80, {true, 80, 2048, 2048});
    // 对齐后减少实际核数（缺陷反例修复）：候选=80、raw=2049 → factor=2560、
    // used=CeilDiv(163841,2560)=65、tail=163841-64*2560=1，不再产生负 tail
    ExpectMultiCore(163841, 80, {true, 65, 2560, 1});

    // availableCoreNum==0 守卫（§9.5 CalcMultiCore 前置检查）
    ExpectMultiCore(163841, 0, {false, 0, 0, 0});

    // 尾核/非尾核 UB loop-tail（factor=9344；尾核长度 1 或 2048）
    int64_t loop = -1;
    int64_t tail = -1;
    EXPECT_TRUE(optiling::CalcLoopTail(2048, kUbFactorSb, loop, tail));
    EXPECT_EQ(loop, int64_t{1});
    EXPECT_EQ(tail, int64_t{2048});
    loop = -1;
    tail = -1;
    EXPECT_TRUE(optiling::CalcLoopTail(1, kUbFactorSb, loop, tail));
    EXPECT_EQ(loop, int64_t{1});
    EXPECT_EQ(tail, int64_t{1});
}

// INT64_MAX 边界与“对齐不可表示”反例（§9.5 最后两行）。
TEST(APGDCommonTiling, Int64MaxAndUnrepresentableAlign) // 例 8
{
    // INT64_MAX、available=80：raw=CeilDiv(INT64_MAX,80)=115292150460684698 →
    // factor=AlignUp(,512)=115292150460684800；used=CeilDiv(INT64_MAX,factor)=80；
    // 前缀 79*factor=9108079886394099200 < INT64_MAX；tail=115292150460676607
    ExpectMultiCore(kInt64Max, 80, {true, 80, 115292150460684800, 115292150460676607});

    int64_t r = -1;
    EXPECT_TRUE(optiling::CeilDivPositive(kInt64Max, kMinElemsPerCore, r));
    EXPECT_EQ(r, int64_t{4503599627370496});
    r = -1;
    EXPECT_TRUE(optiling::AlignUpPositive(115292150460684698, kElemAlignFactor, r));
    EXPECT_EQ(r, int64_t{115292150460684800});

    // 对齐不可表示：available=1、dim0=INT64_MAX → 下一个 512 倍数超 int64，
    // AlignUpPositive 加法前返回 false；CalcMultiCore 随之 false（§9.5 表末行）
    r = -1;
    EXPECT_FALSE(optiling::AlignUpPositive(kInt64Max, kElemAlignFactor, r));
    EXPECT_EQ(r, int64_t{-1});
    ExpectMultiCore(kInt64Max, 1, {false, 0, 0, 0});
}

// =============================================================================
// §9.7 typed TilingData/workspace/SetBlockDim 提交及失败副作用（4 例）
// =============================================================================

// 成功提交：写全 10 个字段（含 reserved=0）、workspace[0]=0、按 usedCoreNum 调
// SetBlockDim 恰好一次并返回 true。
TEST(APGDCommonTiling, CommitTilingDataSuccessWritesAllFields) // 例 9
{
    InplaceApplyProximalGradientDescentTilingData td{};
    size_t workspace = 12345;
    std::vector<uint32_t> calls;
    const bool ok = RunCommit(&td, &workspace, &calls, /*blockDimSucceeds*/ true);
    EXPECT_TRUE(ok);
    ExpectAllFields(td);
    EXPECT_EQ(workspace, size_t{0}); // §9.6 workspace[0]=0
    ASSERT_EQ(calls.size(), size_t{1});
    EXPECT_EQ(calls[0], uint32_t{2}); // SetBlockDim(usedCoreNum)
}

// td 为空：GRAPH_FAILED 语义，不写 workspace、不调 SetBlockDim。
TEST(APGDCommonTiling, CommitTilingDataNullTdStopsAllSideEffects) // 例 10
{
    size_t workspace = 0;
    std::memset(&workspace, 0x11, sizeof(workspace)); // 调用前哨兵模式
    std::vector<uint32_t> calls;
    const bool ok = RunCommit(nullptr, &workspace, &calls, /*blockDimSucceeds*/ true);
    EXPECT_FALSE(ok);           // §9.9 失败点表：typed TilingData 为空 → GRAPH_FAILED
    EXPECT_TRUE(calls.empty()); // 无 SetBlockDim 副作用
    const unsigned char expect = 0x11;
    for (size_t i = 0; i < sizeof(workspace); ++i) {
        EXPECT_EQ(reinterpret_cast<unsigned char*>(&workspace)[i], expect);
    }
}

// workspace 为空：GRAPH_FAILED 语义，不写 td、不调 SetBlockDim。
TEST(APGDCommonTiling, CommitTilingDataNullWorkspaceStopsAllSideEffects) // 例 11
{
    InplaceApplyProximalGradientDescentTilingData td{};
    std::memset(&td, 0x11, sizeof(td)); // 调用前哨兵模式
    std::vector<uint32_t> calls;
    const bool ok = RunCommit(&td, nullptr, &calls, /*blockDimSucceeds*/ true);
    EXPECT_FALSE(ok); // §9.9 失败点表：workspace slot 为空 → GRAPH_FAILED
    EXPECT_TRUE(calls.empty());
    const unsigned char expect = 0x11;
    const auto* bytes = reinterpret_cast<const unsigned char*>(&td);
    for (size_t i = 0; i < sizeof(td); ++i) {
        EXPECT_EQ(bytes[i], expect);
    }
}

// SetBlockDim 失败：返回 false（glue 不得继续 selector 提交）；按 §9.9 行序字段
// 已写、workspace[0]=0、回调恰好调用一次。
TEST(APGDCommonTiling, CommitTilingDataBlockDimFailureBlocksSubmit) // 例 12
{
    InplaceApplyProximalGradientDescentTilingData td{};
    size_t workspace = 12345;
    std::vector<uint32_t> calls;
    const bool ok = RunCommit(&td, &workspace, &calls, /*blockDimSucceeds*/ false);
    EXPECT_FALSE(ok);    // §9.9：SetBlockDim != GRAPH_SUCCESS → GRAPH_FAILED，不调 selector
    ExpectAllFields(td); // 行序副作用：字段先写、blockDim 后校验
    EXPECT_EQ(workspace, size_t{0});
    ASSERT_EQ(calls.size(), size_t{1});
    EXPECT_EQ(calls[0], uint32_t{2});
}
