/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// =============================================================================
// LpNormReduce_package/op_kernel/arch35/lp_norm_reduce_base.h
// =============================================================================
//
// ROLE: LpNormReduce Base 模板（tilingKey=0）kernel 类实现。
//
//   据 docs/LpNormReduce/design/Kernel.md（§9 跨分支共享 VF 计算链）与
//   docs/LpNormReduce/design/branches/DESIGN-BRANCH-0.md §3–§5（流水线 / 内存
//   规划 / Kernel 计算链）落码，范式来源 reduction 二分归约参考实现
//   euclidean_norm_base.h（占位替换：Square→Abs+p 分支、Sum→Sum/Max/Min 按
//   pOrder 分发、pad_value 按 pOrder 分发、PostElewise 去 Sqrt）。
//
//   数学（spec.yaml）：y = Σ|x|^p 沿 axes 归约（不开方、无 epsilon）；
//   p=+inf 哨兵(2147483647)→max|x|、p=−inf 哨兵(−2147483648)→min|x|、
//   p=0→非零计数、p=1→Σ|x|、p≥2→|x|^p 二进制快速幂（平方-乘，≤63 轮 Mul）。
//   fp16 输入 PreElewise 升 fp32 累加（dtype_policy.accumulator_dtype: float32），
//   PostElewise 缩位回 fp16（SatMode::NO_SAT + RoundMode::CAST_RINT）。
//
//   架构（reduction 二分归约范式）：
//   - 统一 TBuf + SetFlag/WaitFlag（不使用 TQue），5 路 UB 物理节点：
//     preInBuf(DT) + preReduceResult(fp32) + preReduceResultTail(fp32) +
//     cacheBuf(16KB 恒定) + outBuf(DT)，深度均 1（不开 DoubleBuffer）
//   - Reduce 直写 cacheBuf[cacheID × levelStride]，DoCaching 就地吸收低层
//   - 二分缓存树 Phase A/B：主段 [0, bisectionPos) + 尾段配对合并，
//     sum 族固定合并顺序保证 bitwise 可复现；max/min 幂等（树退化为根）
//   - 无跨核同步（base 不调用 SyncAll、不设置 ScheduleMode）
//
// =============================================================================

#ifndef LP_NORM_REDUCE_BASE_H_
#define LP_NORM_REDUCE_BASE_H_

#include "kernel_operator.h"            // Ascend C core framework
#include "op_kernel/platform_util.h"    // Ops::Base::GetVRegSize / GetUbBlockSize
#include "op_kernel/math_util.h"        // Ops::Base::CeilDiv / CeilAlign / FloorAlign
#include "adv_api/reduce/reduce.h"      // ReduceSum / ReduceMax / ReduceMin + Pattern
#include "lp_norm_reduce_tiling_data.h" // LpNormReduceTilingData（base/group 共用）

namespace NsLpNormReduce {

using namespace AscendC;

// ─── 共享常量（Kernel.md §9；Ops::Base 取平台实测值） ───
constexpr uint32_t VL_BYTES = Ops::Base::GetVRegSize(); // 向量寄存器字节数（256B）
constexpr uint32_t REP_F32 = VL_BYTES / sizeof(float);  // 单 repeat fp32 lane 数（64）
constexpr uint16_t REP_F32_U16 = static_cast<uint16_t>(REP_F32);
constexpr uint32_t UB_BLOCK_BYTES = Ops::Base::GetUbBlockSize();  // 32B
constexpr uint32_t UB_BLOCK_F32 = UB_BLOCK_BYTES / sizeof(float); // = 8
constexpr uint64_t UINT64_BITS = 64;
constexpr uint64_t UINT64_TOP_BIT_IDX = UINT64_BITS - 1;

// A/R 轴模式化后偶位 A、奇位 R，相邻同类型轴（同为 A 或同为 R）的固定间距。
constexpr int32_t AXIS_INTERVAL = 2;
// b16（fp16/bf16）单元素字节数。
constexpr size_t BYTES_PER_B16_ELEM = 2;
// ReduceXxx srcShape 的 2D 维度（AR/RA 两 pattern 均为二维）。
constexpr size_t REDUCE_SHAPE_DIM = 2;
// FindNearestPower2 数学边界：v ≤ 2 时最近二次幂为 1。
constexpr uint64_t NEAREST_POW2_SMALL_BOUND = 2;

// DataCopyPad 轴层级索引：[0]最内块长 / [1]blockCount / [2]loop1 / [3]loop2 / [4..]外层软循环。
constexpr int32_t BLOCK_COUNT_AXIS_IDX = 1;
constexpr int32_t LOOP1_AXIS_IDX = 2;
constexpr int32_t LOOP2_AXIS_IDX = 3;
constexpr int32_t OUTER_LOOP_AXIS_BASE = 4;

// attr p 的 ±inf 整型哨兵（spec.yaml attributes.p；-2147483648 写作
// -2147483647LL-1 避免字面量溢出）
constexpr int64_t P_INF_SENTINEL = 2147483647;
constexpr int64_t N_INF_SENTINEL = -2147483647LL - 1;

// ─── Cast trait（Kernel.md §9.1 / §9.6，cast-rules.md 口径） ───
// 扩位（b16/int32 → fp32）：CAST_NONE 直转
constexpr AscendC::Reg::CastTrait CAST_TRAIT_TO_FP32{AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
                                                     AscendC::Reg::MaskMergeMode::ZEROING,
                                                     AscendC::RoundMode::CAST_NONE};
// 缩位（fp32 → fp16）：NO_SAT 不饱和截断（超出 fp16 范围 → Inf，IEEE-754 溢出，
// spec extreme_inputs 口径）+ CAST_RINT 就近舍入
constexpr AscendC::Reg::CastTrait CAST_TRAIT_FROM_FP32_FP16{
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};
// 缩位（fp32 → int32）：结构保留（本算子当前 dtype 集下编译期丢弃）
constexpr AscendC::Reg::CastTrait CAST_TRAIT_FROM_FP32_INT32{
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_TRUNC};

// ─── pad_value 按 pOrder 分发（Kernel.md §9.3 取值表） ───
// sum 族 0.0f / +inf 哨兵（max）−Inf / −inf 哨兵（min）+Inf；
// ∓Inf 对 valid lane 恒为中性元（幂等），保证 spec extreme_inputs
// 「IEEE-754 不截断」（有限 pad 会把全 +Inf 输入的 min|x| 钳到有限值）。
constexpr float PAD_CLEAR_SUM = 0.0f;
constexpr float PAD_CLEAR_MAX = -__builtin_huge_valf(); // −Inf（max 分支 pad_value）
constexpr float PAD_CLEAR_MIN = __builtin_huge_valf();  // +Inf（min 分支 pad_value）

__aicore__ inline float PadValueOf(int64_t pOrder)
{
    if (pOrder == P_INF_SENTINEL) {
        return PAD_CLEAR_MAX;
    }
    if (pOrder == N_INF_SENTINEL) {
        return PAD_CLEAR_MIN;
    }
    return PAD_CLEAR_SUM;
}

// =============================================================================
// §9.1 PreElewise —— pre-elewise Cast + Abs + p 分支预处理（VF 融合链）
//
// 公式对应：|x| 及其 p 分支（p=0 的 1[x≠0]、p≥2 的 |x|^p 二进制快速幂——平方-乘，≤63 轮）；
// p=±inf 哨兵 / p=1 直通 |x|（Reducer 在 ReduceChunk 按 pOrder 分发）。
// p 分支互斥、pOrder 运行时标量选择（寄存器链不因分支断开）；NaN 按 IEEE-754
// 传播（|NaN|=NaN、NaN≠0 为真 → p=0 计 1，与 golden (x != 0) 一致）。
// padded 整 tile 覆盖：R 方向 pad 由清零 VF 在 Reduce 前清为 pad_value，
// A 方向 garbage 由 lane 隔离（Kernel.md §9.4）。
// =============================================================================
template <typename DType>
__simd_vf__ inline void PreElewiseVfImpl(__ubuf__ DType* src, __ubuf__ float* dst, uint32_t totalElems,
                                         uint16_t repeatTime, int64_t pOrder)
{
    constexpr bool IsFp32 = std::is_same_v<DType, float>;
    constexpr bool IsB16 = (sizeof(DType) == BYTES_PER_B16_ELEM);
    AscendC::Reg::RegTensor<float> f32Reg;
    AscendC::Reg::MaskReg mask;
    uint32_t remaining = totalElems;

    for (uint16_t i = 0; i < repeatTime; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(REP_F32);
        mask = AscendC::Reg::UpdateMask<float>(remaining);

        if constexpr (IsFp32) {
            AscendC::Reg::LoadAlign(f32Reg, src + off);
        } else if constexpr (IsB16) {
            AscendC::Reg::RegTensor<DType> b16Reg;
            AscendC::Reg::LoadAlign<DType, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(b16Reg, src + off);
            AscendC::Reg::Cast<float, DType, CAST_TRAIT_TO_FP32>(f32Reg, b16Reg, mask);
        } else {
            AscendC::Reg::RegTensor<int32_t> iReg;
            AscendC::Reg::LoadAlign(iReg, src + off);
            AscendC::Reg::Cast<float, int32_t, CAST_TRAIT_TO_FP32>(f32Reg, iReg, mask);
        }

        AscendC::Reg::Abs(f32Reg, f32Reg, mask); // |x|（fp32 计算域）

        if (pOrder == 0) { // p=0：非零计数 → 0/1
            AscendC::Reg::RegTensor<float> zeroReg;
            AscendC::Reg::RegTensor<float> oneReg;
            AscendC::Reg::MaskReg neMask;
            AscendC::Reg::Duplicate(zeroReg, 0.0f);
            AscendC::Reg::Duplicate(oneReg, 1.0f);
            AscendC::Reg::Compare<float, AscendC::CMPMODE::NE>(neMask, f32Reg, zeroReg, mask);
            AscendC::Reg::Select<float>(f32Reg, oneReg, zeroReg, neMask);
        } else if (pOrder >= 2 && pOrder != P_INF_SENTINEL) {
            // p≥2（非哨兵）：二进制快速幂 |x|^p（平方-乘，≤63 轮向量 Mul——对齐
            // changwei lp_norm_reduce_dag.h IntegerPowerSimtCompute 的算法；exponent
            // 为运行时标量，轮数由其二进制位数决定，与 tile 元素数无关）。
            // ⛔ +inf 哨兵 2147483647 仍须排除——它按 §9.1 取值表走「直通 |x| +
            // ReduceMax」，落入本分支会污染 max 语义。
            // 顺序乘法 → 二进制幂的浮点结合次序变化在 fp32 计算域容差内，且与
            // changwei 参考实现求值次序一致（codex 审查缺陷 #1：原 p−1 次线性
            // 乘法在合法大 p 下 ~21 亿轮/ tile，表现为 kernel 超时）。
            AscendC::Reg::RegTensor<float> absReg;
            AscendC::Reg::Abs(absReg, f32Reg, mask); // base = |x|（Abs 幂等）
            AscendC::Reg::Duplicate(f32Reg, 1.0f);   // result = 1
            uint64_t exp = static_cast<uint64_t>(pOrder);
            while (exp != 0UL) {
                if ((exp & 1UL) != 0UL) {
                    AscendC::Reg::Mul(f32Reg, f32Reg, absReg, mask); // result *= base
                }
                exp >>= 1UL;
                if (exp != 0UL) {
                    AscendC::Reg::Mul(absReg, absReg, absReg, mask); // base *= base
                }
            }
        }
        // p=±inf 哨兵 / p=1：直通 |x|

        AscendC::Reg::StoreAlign(dst + off, f32Reg, mask);
    }
}

// =============================================================================
// §9.3 Partial chunk 与行 pad 清零 —— pad_value 按 pOrder 分发
//
// 搬运不指定 paddingValue（isPad=false，BurstPad / ExtensionPad 均为脏数据），
// R 方向 pad 在 Reduce 前必须清为 pad_value（A 方向 garbage 天然隔离不清，
// Kernel.md §9.4 ⛔ 禁止额外 A 方向清零）。清零域 = preReduceResult(fp32)，
// 时机 = PreElewise 之后、Reduce 之前。
// =============================================================================

// tail-R 分支：ExtensionPad 按 A entry 逐行清（extStart 起跳过 BurstPad 区间）
__simd_vf__ inline void ClearChunkExtTailRVfImpl(__ubuf__ float* base, uint32_t extStart, uint32_t aStride,
                                                 uint32_t extLanes, uint16_t aU16, uint16_t repPerA, float padVal)
{
    AscendC::Reg::RegTensor<float> idReg;
    AscendC::Reg::Duplicate(idReg, padVal);

    for (uint16_t aIdx = 0; aIdx < aU16; ++aIdx) {
        int32_t aOff = static_cast<int32_t>(aIdx) * static_cast<int32_t>(aStride);
        uint32_t remaining = extLanes;
        for (uint16_t r = 0; r < repPerA; ++r) {
            int32_t off = aOff + static_cast<int32_t>(extStart) +
                          static_cast<int32_t>(r) * static_cast<int32_t>(REP_F32);
            auto mask = AscendC::Reg::UpdateMask<float>(remaining); // 引用语义：remaining 自动递减
            AscendC::Reg::StoreAlign(base + off, idReg, mask);
        }
    }
}

// tail-A 分支：ExtensionPad 连续整段清（R 切分轴在 UB 最外层，stale 区连续）
__simd_vf__ inline void ClearChunkExtTailAVfImpl(__ubuf__ float* base, uint32_t startElem, uint32_t totalClear,
                                                 uint16_t repCount, float padVal)
{
    AscendC::Reg::RegTensor<float> idReg;
    AscendC::Reg::Duplicate(idReg, padVal);
    AscendC::Reg::MaskReg mask;
    uint32_t remaining = totalClear;
    for (uint16_t i = 0; i < repCount; ++i) {
        int32_t off = static_cast<int32_t>(startElem) + static_cast<int32_t>(i) * static_cast<int32_t>(REP_F32);
        mask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::StoreAlign(base + off, idReg, mask);
    }
}

// tail-R 路径 BurstPad 清零：StoreAlign 起点 FloorAlign（block 对齐）+ mask 三段
// —— notStart 掏空 valid 前缀、maskEnd 截住 block 尾，只写 [validR, padEndInRow)
__simd_vf__ inline void ClearInnerBurstTailPadVfImpl(__ubuf__ float* base, uint16_t rowCntU16, int32_t rowStrideI,
                                                     int32_t windowOff, uint32_t padEnd, uint32_t partialStartInBlock,
                                                     float padVal)
{
    AscendC::Reg::RegTensor<float> idReg;
    AscendC::Reg::Duplicate(idReg, padVal);

    uint32_t cntEnd = padEnd;                // ⛔ UpdateMask 引用语义：cntEnd / cntStart 必须是
    uint32_t cntStart = partialStartInBlock; // 独立非 const 局部变量（被消耗递减）
    auto maskEnd = AscendC::Reg::UpdateMask<float>(cntEnd);
    auto maskStart = AscendC::Reg::UpdateMask<float>(cntStart);
    auto allMask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg notStart;
    AscendC::Reg::MaskReg padMask;
    AscendC::Reg::Not(notStart, maskStart, allMask);
    AscendC::Reg::And(padMask, maskEnd, notStart, allMask);

    for (uint16_t row = 0; row < rowCntU16; ++row) {
        int32_t rowOff = static_cast<int32_t>(row) * rowStrideI;
        AscendC::Reg::StoreAlign(base + rowOff + windowOff, idReg, padMask);
    }
}

// Phase B 主/尾块配对合并 —— 合并算子按 pOrder 分发：
// sum 族 Add（sum(拼接 R) = sum(主)+sum(尾)）/ +inf Max / −inf Min（max/min 可分解）
__simd_vf__ inline void MergeTmpBufVfImpl(__ubuf__ float* mainBuf, __ubuf__ float* tailBuf, uint32_t totalElems,
                                          uint16_t repeatTime, int64_t pOrder)
{
    AscendC::Reg::RegTensor<float> aReg;
    AscendC::Reg::RegTensor<float> bReg;
    AscendC::Reg::MaskReg mask;
    uint32_t remaining = totalElems;
    for (uint16_t i = 0; i < repeatTime; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(REP_F32);
        mask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::LoadAlign(aReg, mainBuf + off);
        AscendC::Reg::LoadAlign(bReg, tailBuf + off);
        if (pOrder == P_INF_SENTINEL) {
            AscendC::Reg::Max(aReg, aReg, bReg, mask); // max(主块, 尾块)
        } else if (pOrder == N_INF_SENTINEL) {
            AscendC::Reg::Min(aReg, aReg, bReg, mask); // min(主块, 尾块)
        } else {
            AscendC::Reg::Add(aReg, aReg, bReg, mask); // sum 族
        }
        AscendC::Reg::StoreAlign(mainBuf + off, aReg, mask);
    }
}

// §9.2 DoCaching：当前层级结果就地吸收全部低层（合并算子按 pOrder 分发：
// sum 族 Add / +inf Max / −inf Min——max/min 幂等，树退化为根节点线性累加）
__simd_vf__ inline void DoCachingVfImpl(__ubuf__ float* cacheBuf, uint32_t laneN, uint32_t levelStride,
                                        int32_t levelOff, uint16_t repeatTime, uint16_t cacheLevelCnt, int64_t pOrder)
{
    AscendC::Reg::RegTensor<float> aReg;
    AscendC::Reg::RegTensor<float> bReg;
    AscendC::Reg::MaskReg mask;
    uint32_t remaining = laneN;
    for (uint16_t i = 0; i < repeatTime; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(REP_F32);
        mask = AscendC::Reg::UpdateMask<float>(remaining);

        AscendC::Reg::LoadAlign(aReg, cacheBuf + levelOff + off);

        for (uint16_t j = 0; j < cacheLevelCnt; ++j) {
            int32_t lowerLevelOff = static_cast<int32_t>(j) * static_cast<int32_t>(levelStride) + off;
            AscendC::Reg::LoadAlign(bReg, cacheBuf + lowerLevelOff);
            if (pOrder == P_INF_SENTINEL) {
                AscendC::Reg::Max(aReg, aReg, bReg, mask);
            } else if (pOrder == N_INF_SENTINEL) {
                AscendC::Reg::Min(aReg, aReg, bReg, mask);
            } else {
                AscendC::Reg::Add(aReg, aReg, bReg, mask);
            }
        }
        AscendC::Reg::StoreAlign(cacheBuf + levelOff + off, aReg, mask);
    }
}

// =============================================================================
// §9.6 PostElewise —— 树根 fp32 → [缩位 Cast] → outBuf（无算子专属后处理）
//
// 本算子不开方（1/p 开方与 epsilon 分母保护归配套更新段算子 LpNormUpdate，
// REQUIREMENTS §2.1.2）、无均值系数、无 post_reduce_input——VF 链仅剩
// 树根读 → [Cast 缩位] → StoreAlign（替换参考实现的 Sqrt 占位为直通）。
// 按 padded 整行处理（A 方向 garbage Cast 无害），CopyOut 只拷 valid。
// =============================================================================
template <typename DType>
__simd_vf__ inline void PostElewiseVfImpl(__ubuf__ float* rootPtr, __ubuf__ DType* outPtr, uint32_t laneN,
                                          uint16_t repeatTime)
{
    constexpr bool IsFp32 = std::is_same_v<DType, float>;
    constexpr bool IsB16 = (sizeof(DType) == BYTES_PER_B16_ELEM);
    AscendC::Reg::RegTensor<float> f32Reg;
    AscendC::Reg::MaskReg mask;
    uint32_t remaining = laneN;
    for (uint16_t i = 0; i < repeatTime; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(REP_F32);
        mask = AscendC::Reg::UpdateMask<float>(remaining);

        AscendC::Reg::LoadAlign(f32Reg, rootPtr + off);
        // 算子专属 PostElewise：无（不开方、无系数、无 epsilon——直通）

        if constexpr (IsFp32) {
            AscendC::Reg::StoreAlign(outPtr + off, f32Reg, mask);
        } else if constexpr (IsB16) {
            AscendC::Reg::RegTensor<DType> b16Reg;
            AscendC::Reg::Cast<DType, float, CAST_TRAIT_FROM_FP32_FP16>(b16Reg, f32Reg, mask);
            AscendC::Reg::StoreAlign<DType, AscendC::Reg::StoreDist::DIST_PACK_B32>(outPtr + off, b16Reg, mask);
        } else {
            AscendC::Reg::RegTensor<int32_t> iReg;
            AscendC::Reg::Cast<int32_t, float, CAST_TRAIT_FROM_FP32_INT32>(iReg, f32Reg, mask);
            AscendC::Reg::StoreAlign(outPtr + off, iReg, mask);
        }
    }
}

// ---------------------------------------------------------------------------
// UBAxisDesc：GM→UB 轴映射（actual/padded 双值 + GM 步长），供 DoCopyInTile
// 组装 DataCopyPad 参数（DESIGN-BRANCH-0.md §5.2）。
// ---------------------------------------------------------------------------
struct UBAxisDesc {
    int32_t gmIdx;     // 原 pattern 轴序号（0..axisNum-1），GM 偏移还原用
    int64_t actualNum; // actual 元素数 → blockLen / blockCount / loopSize
    int64_t paddedNum; // UB 行步距元素数 → dstStride / ubStride
    int64_t gmStride;  // GM stride（元素，来自 TilingData axisStride）
};

// ===========================================================================
// LpNormReduceBaseKernel<DType> —— Base 模板 kernel 类（tilingKey=0）
//
// 数据流（DESIGN-BRANCH-0.md §3）：CopyIn(MTE2) → Compute(V 流水：
// PreElewise → R 方向 pad 清零 → Phase A 合并 → Reduce → DoCaching) →
// PostElewise → CopyOut(MTE3)；两级循环 = aLoop 外层 × rIdx 内层（二分缓存树），
// 每 aLoopIdx 一棵独立树，R 循环收尾后一次性 PostElewise + CopyOut。
// UB 持有 = 5 节点（P_pre=2 + P_pre_ext=1 + cacheBuf=1 + P_post=1，§4 划分表）。
// ===========================================================================
template <typename DType>
class LpNormReduceBaseKernel {
public:
    using DT = DType;

    __aicore__ inline LpNormReduceBaseKernel() {}

    // Base 初始化：缓存 TilingData 指针、现算二分树/输出步长派生量、绑定 GM、
    // 分配 5 个 UB buffer（preIn/preRes/preResTail/cache/out）、取 4 类 eventID。
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const LpNormReduceTilingData* td, TPipe* pipe);
    // Base 主流程：blockIdx → [aLoopStart, aLoopEnd) 大小核映射；每 aLoop 依次
    // R 主段二分树归约 → PostElewise → CopyOut；跨流水同步按 §5.5 持有法则
    // （MTE2_V / V_MTE3 正向 RAW + V_MTE2 / MTE3_V 跨迭代反向 WAR）。
    __aicore__ inline void Process();

protected:
    __aicore__ inline void UnravelBlockLoop(int64_t& aLoopStart, int64_t& aLoopEnd);
    __aicore__ inline void UnravelALoop(int64_t aLoopIdx, int64_t aIdx[], int64_t& aSplitChunkIdx);
    __aicore__ inline int64_t UnravelRLoop(int64_t rIdx, int64_t rOuterIdx[], int64_t& rChunkIdx, int64_t& rLen);

    // 单 R chunk 处理（主/尾块共用）：CopyIn(MTE2) + PreElewise(V) + R 方向 pad
    // 清零；同步对内嵌（V_MTE2 WAR 由 firstChunkOfCore 控制首块跳过）。
    __aicore__ inline void ProcessOneRChunk(int64_t outerGmOff, int64_t aLen, int64_t rIdx, __ubuf__ float* dst,
                                            bool firstChunkOfCore);
    // 后处理：cacheBuf 树根 → [缩位 Cast] → outBuf（VF 融合链，无同步——
    // V_MTE3 对由 Process 按 §5.5 统一插入）。
    __aicore__ inline void PostElewise();
    // 输出搬运：outBuf 有效段（aLen×innerAProd）DataCopyPad 直写 GM y
    // （tail-R dense / tail-A+LastA 单 burst / tail-A+!LastA 多 burst 三路径）。
    __aicore__ inline void CopyOut(int64_t outerOutOff, int64_t aLen);

    // 构造 GM→UB 的轴映射表（actual/padded 双值 + GM 步长），供 DoCopyInTile
    // 组装 DataCopyPad 参数。UB 排布由 tail 类型决定：tail-R → [A_bundle 在外,
    // R_bundle 在内]（AR 视图）；tail-A → [R_bundle 在外, A_bundle 在内]（RA 视图）。
    __aicore__ inline int32_t BuildUBAxes(int64_t aLen, int64_t rLen, UBAxisDesc out[]);
    // 输入搬运：按轴映射表组装 extParams/loopParams（K≥3 开 Loop 模式，
    // SetLoopModePara / ResetLoopModePara 成对），外层 for 覆盖 >4 维。
    __aicore__ inline void DoCopyInTile(int64_t baseGmOff, int64_t aLen, int64_t rLen, const LocalTensor<DT>& preIn);

    // §9.1 PreElewise wrapper：一条 asc_vf_call 融合链（Cast+Abs+p 分支）
    __aicore__ inline void PreElewiseVf(__ubuf__ DT* src, __ubuf__ float* dst);
    // §9.3 pad 清零 wrapper（padVal 由 PadValueOf(pOrder) 分发）
    __aicore__ inline void ClearChunkExtensionVf(__ubuf__ float* base, int64_t rLen);
    __aicore__ inline void ClearInnerBurstTailPadVf(__ubuf__ float* base, int64_t rLen);
    // §9.3 Phase A 主/尾块合并 wrapper
    __aicore__ inline void MergeTmpBufVf(__ubuf__ float* mainBuf, __ubuf__ float* tailBuf);
    // §9.2 DoCaching wrapper：本层就地吸收全部低层
    __aicore__ inline void DoCachingVf(uint16_t cacheID);
    // §9.5 Reduce 高阶 API：ReduceSum/Max/Min 按 pOrder 分发，dst 直写
    // cacheBuf[cacheID×levelStride]，sharedTmpBuffer=preResTail，isReuseSource=true
    __aicore__ inline void ReduceChunk(uint16_t cacheID);

    __aicore__ inline int32_t LastAAxis() const;
    __aicore__ inline int32_t LastRAxis() const;
    __aicore__ inline uint16_t GetCacheID(int64_t idx) const;
    __aicore__ inline uint64_t FindNearestPower2(uint64_t v) const;
    __aicore__ inline uint64_t CalLog2(uint64_t v) const;
    __aicore__ inline int64_t RLenOfChunk(int64_t rChunkIdx) const;

    const LpNormReduceTilingData* td_ = nullptr;
    bool isTailR_ = false; // tail 类型：Init 由 axisNum 奇偶现算（偶→tail-R、奇→tail-A）
    int64_t rSplitChunkCnt_ = 0;
    int64_t bisectionPos_ = 0;                  // 二分主段长度（严格小于 M 的最大 2 幂）
    int64_t bisectionTail_ = 0;                 // 二分尾段长度（Phase A 配对数）
    int64_t cacheCount_ = 0;                    // cacheBuf 层数 = CalLog2(bisectionPos)+1
    int64_t outStride_[MAX_PATTERN_RANK] = {0}; // 输出 GM 各 A 轴步长（kernel 现算）

    GlobalTensor<DT> xGm_;
    GlobalTensor<DT> yGm_;
    TPipe* pipe_ = nullptr;
    TBuf<TPosition::VECCALC> preInBuf_;
    TBuf<TPosition::VECCALC> preReduceResult_;
    TBuf<TPosition::VECCALC> preReduceResultTail_;
    TBuf<TPosition::VECCALC> cacheBuf_;
    TBuf<TPosition::VECCALC> outBuf_;
    int32_t evMTE2toV_ = 0; // Event ID（§5.5 持有法则 trace 结论：4 类事件）
    int32_t evVtoMTE2_ = 0;
    int32_t evVtoMTE3_ = 0;
    int32_t evMte3toV_ = 0;
};

// ---------------------------------------------------------------------------
// 基础工具：轴定位 / 二分树数学（Kernel.md §9.2，无状态、仅依赖入参）
// ---------------------------------------------------------------------------

template <typename DType>
__aicore__ inline int32_t LpNormReduceBaseKernel<DType>::LastAAxis() const
{
    for (int32_t i = td_->axisNum - 1; i >= 0; --i) {
        if (i % AXIS_INTERVAL == 0) {
            return i;
        }
    }
    return 0;
}

template <typename DType>
__aicore__ inline int32_t LpNormReduceBaseKernel<DType>::LastRAxis() const
{
    for (int32_t i = td_->axisNum - 1; i >= 0; --i) {
        if (i % AXIS_INTERVAL == 1) {
            return i;
        }
    }
    return 1;
}

// FindNearestPower2(v)：严格小于 v 的最大 2 幂（v = 0 → 0，v ≤ 2 → 1）
template <typename DType>
__aicore__ inline uint64_t LpNormReduceBaseKernel<DType>::FindNearestPower2(uint64_t v) const
{
    if (v == 0) {
        return 0;
    }
    if (v <= NEAREST_POW2_SMALL_BOUND) {
        return 1;
    }
    const uint64_t num = v - 1;
    const uint64_t pow = UINT64_TOP_BIT_IDX - AscendC::ScalarCountLeadingZero(num);
    return static_cast<uint64_t>(1) << pow;
}

// CalLog2(v)：⌊log2(v)⌋（v ≥ 1）
template <typename DType>
__aicore__ inline uint64_t LpNormReduceBaseKernel<DType>::CalLog2(uint64_t v) const
{
    uint64_t res = 0;
    while (v > 1) {
        v >>= 1;
        ++res;
    }
    return res;
}

// GetCacheID(idx)：idx 尾随 1 个数 = count_ones(idx ^ (idx+1)) − 1 → chunk 结果
// 写入的树层级（序列自洽：低层先写后读、被吸收后才可能被覆写，cacheBuf 不需预清零）
template <typename DType>
__aicore__ inline uint16_t LpNormReduceBaseKernel<DType>::GetCacheID(int64_t idx) const
{
    const uint64_t v = static_cast<uint64_t>(idx);
    return static_cast<uint16_t>(AscendC::ScalarGetCountOfValue<1>(v ^ (v + 1)) - 1);
}

template <typename DType>
__aicore__ inline int64_t LpNormReduceBaseKernel<DType>::RLenOfChunk(int64_t rChunkIdx) const
{
    const int64_t rAxisSize = td_->axisShape[td_->rSplitIdx];
    const int64_t start = rChunkIdx * td_->rUbFactor;
    return (start + td_->rUbFactor > rAxisSize) ? (rAxisSize - start) : td_->rUbFactor;
}

// ---------------------------------------------------------------------------
// Init（DESIGN-BRANCH-0.md §5.1）
// ---------------------------------------------------------------------------

template <typename DType>
__aicore__ inline void LpNormReduceBaseKernel<DType>::Init(GM_ADDR x, GM_ADDR y, const LpNormReduceTilingData* td,
                                                           TPipe* pipe)
{
    td_ = td;
    isTailR_ = (td_->axisNum % AXIS_INTERVAL == 0); // 偶数轴→tail-R，奇数轴→tail-A

    rSplitChunkCnt_ = Ops::Base::CeilDiv(td->axisShape[td->rSplitIdx], td->rUbFactor);
    bisectionPos_ = static_cast<int64_t>(FindNearestPower2(static_cast<uint64_t>(td->rLoopCntTotal)));
    bisectionTail_ = td->rLoopCntTotal - bisectionPos_;
    cacheCount_ = static_cast<int64_t>(CalLog2(static_cast<uint64_t>(bisectionPos_))) + 1;

    // 输出 GM 步长：输出 = 各 A 轴 size 顺序拼接、连续紧凑无 R 维
    // （output_strides[k_A] = ∏(更内 A 轴 size)，从最内 A 轴向外累积）
    {
        int64_t outStrideAcc = 1;
        for (int32_t i = td->axisNum - 1; i >= 0; --i) {
            if (i % AXIS_INTERVAL == 0) {
                outStride_[i] = outStrideAcc;
                outStrideAcc *= td->axisShape[i];
            }
        }
    }

    // GM 绑定（base 不绑定 workspace：无用户 workspace，DESIGN-BRANCH-0.md §8）
    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT*>(x));
    yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT*>(y));

    // TBuf 分配（§4 划分表：3 路 pre 同尺寸 + cacheBuf 恒 16KB + outBuf；
    // 全部 VECCALC、深度 1，范式约束：统一 TBuf、不使用 TQue）
    pipe_ = pipe;
    pipe_->InitBuffer(preInBuf_, td->preBufSize);
    pipe_->InitBuffer(preReduceResult_, td->preBufSize);
    pipe_->InitBuffer(preReduceResultTail_, td->preBufSize);
    pipe_->InitBuffer(cacheBuf_, td->cacheBufUbSize);
    pipe_->InitBuffer(outBuf_, td->postBufSize);

    // Event ID（§5.5 持有法则 trace 结论：4 类事件；不需要的类型不取）
    evMTE2toV_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    evVtoMTE2_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    evVtoMTE3_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    evMte3toV_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
}

// ---------------------------------------------------------------------------
// 多核映射 / 迭代解码（DESIGN-BRANCH-0.md §2 / §5.3）
// ---------------------------------------------------------------------------

// blockIdx → [aLoopStart, aLoopEnd)：前 aBigCoreCnt 核处理 aBigCoreLoopCnt 个
// aLoop、其余核 aSmallCoreLoopCnt 个（最大负载差 ≤ 1，迭代-核映射固定 → bitwise 可复现）
template <typename DType>
__aicore__ inline void LpNormReduceBaseKernel<DType>::UnravelBlockLoop(int64_t& aLoopStart, int64_t& aLoopEnd)
{
    const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    if (blockIdx < static_cast<int64_t>(td_->aBigCoreCnt)) {
        aLoopStart = blockIdx * td_->aBigCoreLoopCnt;
        aLoopEnd = aLoopStart + td_->aBigCoreLoopCnt;
    } else {
        aLoopStart = static_cast<int64_t>(td_->aBigCoreCnt) * td_->aBigCoreLoopCnt +
                     (blockIdx - static_cast<int64_t>(td_->aBigCoreCnt)) * td_->aSmallCoreLoopCnt;
        aLoopEnd = aLoopStart + td_->aSmallCoreLoopCnt;
    }
}

// aLoopIdx → (外层 A 轴索引 aIdx[], aSplitChunkIdx)：row-major、chunk 在最内
template <typename DType>
__aicore__ inline void LpNormReduceBaseKernel<DType>::UnravelALoop(int64_t aLoopIdx, int64_t aIdx[],
                                                                   int64_t& aSplitChunkIdx)
{
    int64_t aLoopRem = aLoopIdx;
    aSplitChunkIdx = aLoopRem % td_->aSplitChunkCnt;
    aLoopRem /= td_->aSplitChunkCnt;
    for (int32_t k = td_->aSplitIdx - AXIS_INTERVAL; k >= 0; k -= AXIS_INTERVAL) {
        aIdx[k] = aLoopRem % td_->axisShape[k];
        aLoopRem /= td_->axisShape[k];
    }
}

// rIdx → (外层 R 轴索引, chunk 序号, valid 长度)，返回 R 侧 GM 偏移（元素）：
//   Σ_{k_R < rSplitIdx} rOuterIdx[k_R]×axisStride[k_R] + rChunkIdx×rUbFactor×axisStride[rSplitIdx]
template <typename DType>
__aicore__ inline int64_t LpNormReduceBaseKernel<DType>::UnravelRLoop(int64_t rIdx, int64_t rOuterIdx[],
                                                                      int64_t& rChunkIdx, int64_t& rLen)
{
    rChunkIdx = rIdx % rSplitChunkCnt_;
    int64_t rLoopRem = rIdx / rSplitChunkCnt_;
    int64_t gmOff = 0;
    for (int32_t k = td_->rSplitIdx - AXIS_INTERVAL; k >= 1; k -= AXIS_INTERVAL) {
        rOuterIdx[k] = rLoopRem % td_->axisShape[k];
        rLoopRem /= td_->axisShape[k];
        gmOff += rOuterIdx[k] * td_->axisStride[k];
    }
    rLen = RLenOfChunk(rChunkIdx);
    return gmOff + rChunkIdx * td_->rUbFactor * td_->axisStride[td_->rSplitIdx];
}

// ---------------------------------------------------------------------------
// Process（DESIGN-BRANCH-0.md §5.3）—— aLoop 外层 × rIdx 内层（二分缓存树）
// ---------------------------------------------------------------------------
template <typename DType>
__aicore__ inline void LpNormReduceBaseKernel<DType>::Process()
{
    const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    if (blockIdx >= static_cast<int64_t>(td_->usedCoreNum)) {
        return; // 多余核早退（aSmallCoreLoopCnt==0 时仅前 aBigCoreCnt 核工作）
    }

    int64_t aLoopStart = 0;
    int64_t aLoopEnd = 0;
    UnravelBlockLoop(aLoopStart, aLoopEnd);

    bool firstChunkOfCore = true; // 首个 CopyIn 前跳过 WaitFlag<V_MTE2>（§5.5 首轮规则）

    for (int64_t aLoopIdx = aLoopStart; aLoopIdx < aLoopEnd; ++aLoopIdx) {
        int64_t aIdx[MAX_PATTERN_RANK] = {0};
        int64_t aSplitChunkIdx = 0;
        UnravelALoop(aLoopIdx, aIdx, aSplitChunkIdx);

        // A 侧 GM / 输出偏移：外层 A 轴 + A 切分轴 chunk 起点
        int64_t chunkGmOff = 0;
        int64_t chunkOutOff = 0;
        for (int32_t k = td_->aSplitIdx - AXIS_INTERVAL; k >= 0; k -= AXIS_INTERVAL) {
            chunkGmOff += aIdx[k] * td_->axisStride[k];
            chunkOutOff += aIdx[k] * outStride_[k];
        }
        const int64_t aChunkStart = aSplitChunkIdx * td_->aUbFactor;
        const int64_t aEnd = aChunkStart + td_->aUbFactor;
        const int64_t aLen = (aEnd > td_->axisShape[td_->aSplitIdx]) ? (td_->axisShape[td_->aSplitIdx] - aChunkStart) :
                                                                       td_->aUbFactor;
        chunkGmOff += aChunkStart * td_->axisStride[td_->aSplitIdx];
        chunkOutOff += aChunkStart * outStride_[td_->aSplitIdx];

        __ubuf__ float* preRes = reinterpret_cast<__ubuf__ float*>(preReduceResult_.Get<float>().GetPhyAddr());
        __ubuf__ float* preResTail = reinterpret_cast<__ubuf__ float*>(preReduceResultTail_.Get<float>().GetPhyAddr());

        // ── R 主段循环 [0, bisectionPos_)：每个 aLoopIdx 一棵独立二分缓存树 ──
        for (int64_t rIdx = 0; rIdx < bisectionPos_; ++rIdx) {
            // 主块：CopyIn(MTE2) + PreElewise(V) → preReduceResult（含 pad 清零）
            ProcessOneRChunk(chunkGmOff, aLen, rIdx, preRes, firstChunkOfCore);
            firstChunkOfCore = false;

            // Phase A（rIdx < bisectionTail_）：配对尾块 rIdx + bisectionPos_ →
            //    preReduceResultTail，MergeTmpBufVf 逐元素合并进主块
            if (rIdx < bisectionTail_) {
                ProcessOneRChunk(chunkGmOff, aLen, rIdx + bisectionPos_, preResTail, firstChunkOfCore);
                MergeTmpBufVf(preRes, preResTail);
            }

            // Reduce（ReduceSum/Max/Min 按 pOrder 分发）：dst = cacheBuf[cacheID×levelStride]
            const uint16_t cacheID = GetCacheID(rIdx);
            ReduceChunk(cacheID);
            // DoCaching：本层就地吸收全部低层（合并算子按 pOrder 分发）
            DoCachingVf(cacheID);
        }

        // PostElewise：树根 fp32 → [缩位 Cast] → outBuf（V，不含同步——对在下方）
        if (aLoopIdx != aLoopStart) {
            WaitFlag<HardEvent::MTE3_V>(evMte3toV_); // WAR：上轮 CopyOut 已读完 outBuf
        }
        PostElewise();
        SetFlag<HardEvent::V_MTE3>(evVtoMTE3_); // V→MTE3：outBuf 可读
        WaitFlag<HardEvent::V_MTE3>(evVtoMTE3_);

        // CopyOut（MTE3，三路径）：outBuf → GM_y（valid = aLen × innerAProd）
        CopyOut(chunkOutOff, aLen);
        SetFlag<HardEvent::MTE3_V>(evMte3toV_); // 跨 aLoop WAR 保护（末轮无害保留）
    }
}

// ---------------------------------------------------------------------------
// ProcessOneRChunk（DESIGN-BRANCH-0.md §5.2）—— CopyIn + PreElewise + pad 清零
// ---------------------------------------------------------------------------
template <typename DType>
__aicore__ inline void LpNormReduceBaseKernel<DType>::ProcessOneRChunk(int64_t outerGmOff, int64_t aLen, int64_t rIdx,
                                                                       __ubuf__ float* dst, bool firstChunkOfCore)
{
    int64_t rOuterIdx[MAX_PATTERN_RANK] = {0};
    int64_t rChunkIdx = 0;
    int64_t rLen = 0;
    const int64_t rOff = UnravelRLoop(rIdx, rOuterIdx, rChunkIdx, rLen);

    if (!firstChunkOfCore) {
        WaitFlag<HardEvent::V_MTE2>(evVtoMTE2_); // WAR：上一 V 已读完 preInBuf（§5.5）
    }
    DoCopyInTile(outerGmOff + rOff, aLen, rLen, preInBuf_.Get<DT>()); // MTE2 写 preInBuf
    SetFlag<HardEvent::MTE2_V>(evMTE2toV_);                           // MTE2→V
    WaitFlag<HardEvent::MTE2_V>(evMTE2toV_);                          // 等 CopyIn 完成

    PreElewiseVf(reinterpret_cast<__ubuf__ DT*>(preInBuf_.Get<DT>().GetPhyAddr()), dst);
    SetFlag<HardEvent::V_MTE2>(evVtoMTE2_); // V 读完 preInBuf，允许覆写（pad 清零不触 preInBuf）

    // R 方向 pad 清零（Kernel.md §9.3 决策表：partial → ClearChunkExtensionVf 恒清；
    // tail-R 且 burst 尾轴非对齐 → ClearInnerBurstTailPadVf；tail-A full → 不清；
    // pad_value 按 pOrder 分发）
    if (rLen < td_->rUbFactor) {
        ClearChunkExtensionVf(dst, rLen);
    }
    if (isTailR_) {
        ClearInnerBurstTailPadVf(dst, rLen);
    }
}

// ---------------------------------------------------------------------------
// CopyIn（DESIGN-BRANCH-0.md §5.2）—— BuildUBAxes + DoCopyInTile
// ---------------------------------------------------------------------------

template <typename DType>
__aicore__ inline int32_t LpNormReduceBaseKernel<DType>::BuildUBAxes(int64_t aLen, int64_t rLen, UBAxisDesc out[])
{
    int32_t k = 0;
    const int32_t lastA = LastAAxis();
    const int32_t lastR = LastRAxis();
    const int64_t bsElem = static_cast<int64_t>(UB_BLOCK_BYTES) / static_cast<int64_t>(sizeof(DT));

    if (isTailR_) {
        // tail-R：UB = [A_bundle 在外, R_bundle 在内]（AR 视图，最内 R 是 burst 尾轴）
        // —— 先收 R 轴（内层），再收 A 轴（外层），保证 R_bundle 连续在内
        for (int32_t i = td_->axisNum - 1; i >= td_->rSplitIdx; --i) {
            if (i % AXIS_INTERVAL != 1) {
                continue;
            }
            int64_t actual = 0;
            int64_t padded = 0;
            if (i == td_->rSplitIdx) {
                actual = rLen;
                padded = td_->rUbFactorAlign;
            } else if (i == lastR) {
                actual = td_->axisShape[i];
                padded = Ops::Base::CeilAlign(actual, bsElem); // 非切分 burst 尾轴行宽对齐
            } else {
                actual = td_->axisShape[i];
                padded = actual;
            }
            out[k].gmIdx = i;
            out[k].actualNum = actual;
            out[k].paddedNum = padded;
            out[k].gmStride = td_->axisStride[i];
            ++k;
        }
        for (int32_t i = td_->axisNum - 1; i >= td_->aSplitIdx; --i) {
            if (i % AXIS_INTERVAL != 0) {
                continue;
            }
            int64_t actual = 0;
            int64_t padded = 0;
            if (i == td_->aSplitIdx) {
                actual = aLen;
                padded = td_->aUbFactor;
            } else {
                actual = td_->axisShape[i];
                padded = actual;
            }
            out[k].gmIdx = i;
            out[k].actualNum = actual;
            out[k].paddedNum = padded;
            out[k].gmStride = td_->axisStride[i];
            ++k;
        }
    } else {
        // tail-A：UB = [R_bundle 在外, A_bundle 在内]（RA 视图，最内 A 是 burst 尾轴）
        for (int32_t i = td_->axisNum - 1; i >= td_->aSplitIdx; --i) {
            if (i % AXIS_INTERVAL != 0) {
                continue;
            }
            int64_t actual = 0;
            int64_t padded = 0;
            if (i == td_->aSplitIdx) {
                actual = aLen;
                padded = td_->aUbFactor;
            } else if (i == lastA) {
                actual = td_->axisShape[i];
                padded = Ops::Base::CeilAlign(actual, bsElem); // 非切分 burst 尾轴行宽对齐
            } else {
                actual = td_->axisShape[i];
                padded = actual;
            }
            out[k].gmIdx = i;
            out[k].actualNum = actual;
            out[k].paddedNum = padded;
            out[k].gmStride = td_->axisStride[i];
            ++k;
        }
        for (int32_t i = td_->axisNum - 1; i >= td_->rSplitIdx; --i) {
            if (i % AXIS_INTERVAL != 1) {
                continue;
            }
            int64_t actual = 0;
            int64_t padded = 0;
            if (i == td_->rSplitIdx) {
                actual = rLen;
                padded = td_->rUbFactorAlign;
            } else {
                actual = td_->axisShape[i];
                padded = actual;
            }
            out[k].gmIdx = i;
            out[k].actualNum = actual;
            out[k].paddedNum = padded;
            out[k].gmStride = td_->axisStride[i];
            ++k;
        }
    }
    return k;
}

template <typename DType>
__aicore__ inline void LpNormReduceBaseKernel<DType>::DoCopyInTile(int64_t baseGmOff, int64_t aLen, int64_t rLen,
                                                                   const LocalTensor<DT>& preIn)
{
    UBAxisDesc ubAxes[MAX_PATTERN_RANK];
    const int32_t axisCnt = BuildUBAxes(aLen, rLen, ubAxes);

    DataCopyExtParams extParams;
    LoopModeParams loopParams;
    loopParams.loop1Size = 0;
    loopParams.loop1SrcStride = 0;
    loopParams.loop1DstStride = 0;
    loopParams.loop2Size = 0;
    loopParams.loop2SrcStride = 0;
    loopParams.loop2DstStride = 0;

    const int64_t dtBytes = static_cast<int64_t>(sizeof(DT));
    // ── 第 1 根（burst 尾轴）→ blockLen；行内 padding 由 paddedNum 决定 dstStride ──
    extParams.blockLen = static_cast<uint32_t>(ubAxes[0].actualNum * dtBytes); // 有效字节

    // isPad=false：BurstPad 为脏数据，由 Kernel.md §9.3 在 Reduce 前清为 pad_value
    DataCopyPadExtParams<DT> padParams{false, 0, 0, 0};

    const int64_t copyPadBytes = Ops::Base::CeilAlign(static_cast<int64_t>(extParams.blockLen),
                                                      static_cast<int64_t>(UB_BLOCK_BYTES));
    const int64_t target0Bytes = ubAxes[0].paddedNum * dtBytes;           // 固定 UB 行步距
    extParams.dstStride = (target0Bytes - copyPadBytes) / UB_BLOCK_BYTES; // UB 侧 gap（datablock 单位）

    // ── 第 2 根 → blockCount；srcStride = GM 侧 gap(尾→头) = 减一次 blockLen ──
    if (axisCnt > BLOCK_COUNT_AXIS_IDX) {
        extParams.blockCount = static_cast<uint16_t>(ubAxes[BLOCK_COUNT_AXIS_IDX].actualNum);
        extParams.srcStride = ubAxes[BLOCK_COUNT_AXIS_IDX].gmStride * dtBytes -
                              static_cast<int64_t>(extParams.blockLen);
    } else {
        extParams.blockCount = 1;
        extParams.srcStride = 0;
    }
    extParams.rsv = 0; // 必须显式填 0（datacopypad-rules）

    // ── UB 每层字节步长：ubStride[0]=dt；ubStride[k]=ubStride[k-1]×paddedNum[k-1] ──
    int64_t ubStride[MAX_PATTERN_RANK] = {0};
    ubStride[0] = dtBytes;
    for (int32_t i = 1; i < axisCnt; ++i) {
        ubStride[i] = ubStride[i - 1] * ubAxes[i - 1].paddedNum;
    }

    // ── 第 3/4 根 → loop1/loop2；loop*Stride = GM 侧 advance(头→头) = axisStride×dt ──
    if (axisCnt > LOOP1_AXIS_IDX) {
        loopParams.loop1Size = static_cast<uint32_t>(ubAxes[LOOP1_AXIS_IDX].actualNum);
        loopParams.loop1SrcStride = static_cast<uint64_t>(ubAxes[LOOP1_AXIS_IDX].gmStride) *
                                    static_cast<uint64_t>(dtBytes);
        loopParams.loop1DstStride = static_cast<uint64_t>(ubStride[LOOP1_AXIS_IDX]);
        loopParams.loop2Size = 1; // ★ 哪怕外层不用也必须 ≥ 1，否则整批搬运 0 次
    }
    if (axisCnt > LOOP2_AXIS_IDX) {
        loopParams.loop2Size = static_cast<uint32_t>(ubAxes[LOOP2_AXIS_IDX].actualNum);
        loopParams.loop2SrcStride = static_cast<uint64_t>(ubAxes[LOOP2_AXIS_IDX].gmStride) *
                                    static_cast<uint64_t>(dtBytes);
        loopParams.loop2DstStride = static_cast<uint64_t>(ubStride[LOOP2_AXIS_IDX]);
    }

    const bool useLoopMode = (axisCnt > LOOP1_AXIS_IDX);
    if (useLoopMode) {
        SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB); // ★ 调用前 set
    }

    // ── 第 5 根及更外 → for 软循环拆；K≤4 时 outerProd=1，循环一次 ──
    int64_t outerProd = 1;
    for (int32_t k = OUTER_LOOP_AXIS_BASE; k < axisCnt; ++k) {
        outerProd *= ubAxes[k].actualNum;
    }

    for (int64_t outerFlat = 0; outerFlat < outerProd; ++outerFlat) {
        int64_t addGmOffElem = 0;
        int64_t addUbOffBytes = 0;
        int64_t outerRem = outerFlat;
        for (int32_t k = OUTER_LOOP_AXIS_BASE; k < axisCnt; ++k) {
            const int64_t axisSize = ubAxes[k].actualNum;
            const int64_t axisIdx = outerRem % axisSize;
            outerRem /= axisSize;
            addGmOffElem += axisIdx * ubAxes[k].gmStride;
            addUbOffBytes += axisIdx * ubStride[k];
        }
        const int64_t ubOffElems = addUbOffBytes / dtBytes;
        DataCopyPad(preIn[ubOffElems], xGm_[baseGmOff + addGmOffElem], extParams, padParams);
    }

    if (useLoopMode) {
        ResetLoopModePara(DataCopyMVType::OUT_TO_UB); // ★ 调用后 reset（成对出现）
    }
}

// ---------------------------------------------------------------------------
// VF wrappers（各分支 kernel 类私有成员；参数从 TilingData 现算，
// asc_vf_call 显式传参——VF 内禁止访问对象级成员变量）
// ---------------------------------------------------------------------------

// §9.1 PreElewise wrapper：padded 整 tile 覆盖
template <typename DType>
__aicore__ inline void LpNormReduceBaseKernel<DType>::PreElewiseVf(__ubuf__ DT* src, __ubuf__ float* dst)
{
    const uint32_t totalElems = static_cast<uint32_t>(td_->aUbFactor * td_->innerAProdAlign * td_->rUbFactorAlign *
                                                      td_->innerRProdAlign);
    const uint16_t repeatTime = static_cast<uint16_t>(
        Ops::Base::CeilDiv(totalElems, static_cast<uint32_t>(REP_F32_U16)));
    asc_vf_call<PreElewiseVfImpl<DT>>(src, dst, totalElems, repeatTime, td_->pOrder);
}

// §9.3 ClearChunkExtensionVf wrapper：调用前调用方已保证 rLen < rUbFactor
template <typename DType>
__aicore__ inline void LpNormReduceBaseKernel<DType>::ClearChunkExtensionVf(__ubuf__ float* base, int64_t rLen)
{
    if (rLen >= td_->rUbFactor) {
        return;
    }
    const float padVal = PadValueOf(td_->pOrder);

    if (isTailR_) {
        // tail-R：ExtensionPad 按 A entry 逐行清（extStart 起跳过 BurstPad 区间）
        const uint32_t aBundleEntries = static_cast<uint32_t>(td_->aUbFactor * td_->innerAProdAlign);
        const uint32_t innerRPA = static_cast<uint32_t>(td_->innerRProdAlign);
        const uint32_t rLenInner = static_cast<uint32_t>(rLen) * innerRPA;
        const uint32_t extStart = Ops::Base::CeilAlign(rLenInner, UB_BLOCK_F32); // 32B 对齐起点
        const uint32_t aStride = static_cast<uint32_t>(td_->rUbFactorAlign) * innerRPA;
        if (extStart >= aStride) {
            return;
        }
        const uint32_t extLanes = aStride - extStart;
        const uint32_t repPerA = Ops::Base::CeilDiv(extLanes, REP_F32);
        const uint16_t aU16 = static_cast<uint16_t>(aBundleEntries);

        asc_vf_call<ClearChunkExtTailRVfImpl>(base, extStart, aStride, extLanes, aU16, static_cast<uint16_t>(repPerA),
                                              padVal);
    } else {
        // tail-A：ExtensionPad 连续整段清（R 切分轴在 UB 最外层，stale 区连续）
        const uint32_t cellElems = static_cast<uint32_t>(td_->aUbFactor * td_->innerAProdAlign * td_->innerRProdAlign);
        const uint32_t startElem = static_cast<uint32_t>(rLen) * cellElems;
        const uint32_t totalClear = (static_cast<uint32_t>(td_->rUbFactor) - static_cast<uint32_t>(rLen)) * cellElems;
        const uint32_t repCount = Ops::Base::CeilDiv(totalClear, REP_F32);

        asc_vf_call<ClearChunkExtTailAVfImpl>(base, startElem, totalClear, static_cast<uint16_t>(repCount), padVal);
    }
}

// §9.3 ClearInnerBurstTailPadVf wrapper（tail-R 路径；burst 尾轴对齐则无 BurstPad 直接返回）
template <typename DType>
__aicore__ inline void LpNormReduceBaseKernel<DType>::ClearInnerBurstTailPadVf(__ubuf__ float* base, int64_t rLen)
{
    const uint32_t bsInput = UB_BLOCK_BYTES / static_cast<uint32_t>(sizeof(DT));
    const int32_t lastR = LastRAxis();
    const uint32_t validR = (td_->rSplitIdx == lastR) ? static_cast<uint32_t>(rLen) :
                                                        static_cast<uint32_t>(td_->axisShape[lastR]);
    if (validR % bsInput == 0) {
        return; // burst 尾轴对齐，无 BurstPad
    }
    const uint32_t rowStride = (td_->rSplitIdx == lastR) ?
                                   static_cast<uint32_t>(td_->rUbFactorAlign * td_->innerRProdAlign) :
                                   Ops::Base::CeilAlign(static_cast<uint32_t>(td_->axisShape[lastR]), bsInput);
    // rSplit!=LastR：按全量行清（partial 需要的行在 UB 中不连续——A entry 在最外层，
    // 行号含 aStride 跳跃），多清的 stale 行其 BurstPad 位置同为脏数据，多清无害
    const uint32_t rowCnt = (td_->rSplitIdx == lastR) ?
                                static_cast<uint32_t>(td_->aUbFactor * td_->innerAProdAlign) :
                                static_cast<uint32_t>(td_->aUbFactor * td_->innerAProdAlign * td_->rUbFactorAlign *
                                                      td_->innerRProdAlign) /
                                    rowStride;

    const uint32_t padEndInRow = Ops::Base::CeilAlign(validR, bsInput);
    const uint32_t partialBlockIdx = validR / UB_BLOCK_F32; // StoreAlign 起点 FloorAlign（block 对齐）
    const uint32_t partialStartInBlock = validR % UB_BLOCK_F32;
    const uint32_t padEnd = padEndInRow - partialBlockIdx * UB_BLOCK_F32;

    asc_vf_call<ClearInnerBurstTailPadVfImpl>(base, static_cast<uint16_t>(rowCnt), static_cast<int32_t>(rowStride),
                                              static_cast<int32_t>(partialBlockIdx * UB_BLOCK_F32), padEnd,
                                              partialStartInBlock, PadValueOf(td_->pOrder));
}

// §9.3 MergeTmpBufVf wrapper
template <typename DType>
__aicore__ inline void LpNormReduceBaseKernel<DType>::MergeTmpBufVf(__ubuf__ float* mainBuf, __ubuf__ float* tailBuf)
{
    const uint32_t totalElems = static_cast<uint32_t>(td_->aUbFactor * td_->innerAProdAlign * td_->rUbFactorAlign *
                                                      td_->innerRProdAlign);
    const uint16_t repeatTime = static_cast<uint16_t>(
        Ops::Base::CeilDiv(totalElems, static_cast<uint32_t>(REP_F32_U16)));
    asc_vf_call<MergeTmpBufVfImpl>(mainBuf, tailBuf, totalElems, repeatTime, td_->pOrder);
}

// §9.2 DoCachingVf wrapper：laneN / levelStride / levelOff 从 TilingData 现算
template <typename DType>
__aicore__ inline void LpNormReduceBaseKernel<DType>::DoCachingVf(uint16_t cacheID)
{
    const uint32_t laneN = static_cast<uint32_t>(td_->aUbFactor * td_->innerAProdAlign);
    const uint32_t levelStride = Ops::Base::CeilAlign(laneN, UB_BLOCK_F32);
    const int32_t levelOff = static_cast<int32_t>(cacheID) * static_cast<int32_t>(levelStride);
    const uint16_t repeatTime = static_cast<uint16_t>(Ops::Base::CeilDiv(laneN, static_cast<uint32_t>(REP_F32_U16)));

    __ubuf__ float* cachePtr = reinterpret_cast<__ubuf__ float*>(cacheBuf_.Get<float>().GetPhyAddr());
    asc_vf_call<DoCachingVfImpl>(cachePtr, laneN, levelStride, levelOff, repeatTime, cacheID, td_->pOrder);
}

// §9.5 ReduceChunk —— ReduceSum/Max/Min 按 pOrder 分发（同族同签名，仅 API 名不同）。
// Pattern 按 tail 类型：tail-R（axisNum 偶）→ AR {aBundle, rBundle} 沿内层 R reduce；
// tail-A（axisNum 奇）→ RA {rBundle, aBundle} 沿外层 R reduce。dst = cacheBuf[cacheID×levelStride]
// （随后 DoCachingVf(cacheID) 就地吸收低层），sharedTmpBuffer = preReduceResultTail
// （Phase A 主尾配对完成后即空闲，复用不增加 UB 开销），isReuseSource 恒 true
// （Reduce 后不再读 preReduceResult，下一 rIdx 由 CopyIn + PreElewise 整块覆写）。
template <typename DType>
__aicore__ inline void LpNormReduceBaseKernel<DType>::ReduceChunk(uint16_t cacheID)
{
    const uint32_t laneA = static_cast<uint32_t>(td_->aUbFactor * td_->innerAProdAlign);
    const uint32_t levelStride = Ops::Base::CeilAlign(laneA, UB_BLOCK_F32);
    const int32_t levelOff = static_cast<int32_t>(cacheID) * static_cast<int32_t>(levelStride);
    constexpr bool reuseSrc = true; // isReuseSource：Reduce 后不再读 src（§9.5 结论）

    if (isTailR_) {
        uint32_t srcShape[REDUCE_SHAPE_DIM] = {laneA,
                                               static_cast<uint32_t>(td_->rUbFactorAlign * td_->innerRProdAlign)};
        if (td_->pOrder == P_INF_SENTINEL) {
            AscendC::ReduceMax<float, AscendC::Pattern::Reduce::AR, reuseSrc>(
                cacheBuf_.Get<float>()[levelOff], preReduceResult_.Get<float>(), preReduceResultTail_.Get<uint8_t>(),
                srcShape, /*srcInnerPad=*/true);
        } else if (td_->pOrder == N_INF_SENTINEL) {
            AscendC::ReduceMin<float, AscendC::Pattern::Reduce::AR, reuseSrc>(
                cacheBuf_.Get<float>()[levelOff], preReduceResult_.Get<float>(), preReduceResultTail_.Get<uint8_t>(),
                srcShape, /*srcInnerPad=*/true);
        } else {
            AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, reuseSrc>(
                cacheBuf_.Get<float>()[levelOff], preReduceResult_.Get<float>(), preReduceResultTail_.Get<uint8_t>(),
                srcShape, /*srcInnerPad=*/true);
        }
    } else {
        uint32_t srcShape[REDUCE_SHAPE_DIM] = {static_cast<uint32_t>(td_->rUbFactorAlign * td_->innerRProdAlign),
                                               laneA};
        if (td_->pOrder == P_INF_SENTINEL) {
            AscendC::ReduceMax<float, AscendC::Pattern::Reduce::RA, reuseSrc>(
                cacheBuf_.Get<float>()[levelOff], preReduceResult_.Get<float>(), preReduceResultTail_.Get<uint8_t>(),
                srcShape, /*srcInnerPad=*/true);
        } else if (td_->pOrder == N_INF_SENTINEL) {
            AscendC::ReduceMin<float, AscendC::Pattern::Reduce::RA, reuseSrc>(
                cacheBuf_.Get<float>()[levelOff], preReduceResult_.Get<float>(), preReduceResultTail_.Get<uint8_t>(),
                srcShape, /*srcInnerPad=*/true);
        } else {
            AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, reuseSrc>(
                cacheBuf_.Get<float>()[levelOff], preReduceResult_.Get<float>(), preReduceResultTail_.Get<uint8_t>(),
                srcShape, /*srcInnerPad=*/true);
        }
    }
}

// ---------------------------------------------------------------------------
// §9.6 PostElewise wrapper（不含同步——V_MTE3 对由 Process 按 §5.5 插入）
// ---------------------------------------------------------------------------
template <typename DType>
__aicore__ inline void LpNormReduceBaseKernel<DType>::PostElewise()
{
    const uint32_t laneN = static_cast<uint32_t>(td_->aUbFactor * td_->innerAProdAlign);
    const uint32_t levelStride = Ops::Base::CeilAlign(laneN, UB_BLOCK_F32);
    const int32_t rootOff = static_cast<int32_t>(cacheCount_ - 1) * static_cast<int32_t>(levelStride);

    __ubuf__ float* rootPtr = reinterpret_cast<__ubuf__ float*>(cacheBuf_.Get<float>().GetPhyAddr()) + rootOff;
    __ubuf__ DT* outPtr = reinterpret_cast<__ubuf__ DT*>(outBuf_.Get<DT>().GetPhyAddr());

    const uint16_t repeatTime = static_cast<uint16_t>(Ops::Base::CeilDiv(laneN, static_cast<uint32_t>(REP_F32_U16)));

    asc_vf_call<PostElewiseVfImpl<DT>>(rootPtr, outPtr, laneN, repeatTime);
}

// ---------------------------------------------------------------------------
// CopyOut（DESIGN-BRANCH-0.md §5.4）—— 三路径决策 + 单次 DataCopyPad（UB→GM）
// ---------------------------------------------------------------------------

template <typename DType>
__aicore__ inline void LpNormReduceBaseKernel<DType>::CopyOut(int64_t outerOutOff, int64_t aLen)
{
    auto outLocal = outBuf_.Get<DT>();

    DataCopyExtParams outParams;
    if (isTailR_) {
        // 路径 1：tail-R，A_bundle dense 单 burst（innerAProd = aSplitIdx 右侧 A 轴真实乘积）
        int64_t innerAProd = 1;
        for (int32_t k = td_->aSplitIdx + AXIS_INTERVAL; k <= LastAAxis(); k += AXIS_INTERVAL) {
            innerAProd *= td_->axisShape[k];
        }
        outParams.blockLen = static_cast<uint32_t>(aLen * innerAProd * static_cast<int64_t>(sizeof(DT)));
        outParams.blockCount = 1;
    } else {
        const int32_t lastA = LastAAxis();
        const int64_t lastASize = td_->axisShape[lastA];
        if (td_->aSplitIdx == lastA) {
            // 路径 2：tail-A 且切最内 A 轴，单 burst
            outParams.blockLen = static_cast<uint32_t>(aLen * static_cast<int64_t>(sizeof(DT)));
            outParams.blockCount = 1;
        } else {
            // 路径 3：tail-A 且非切最内 A 轴，多 burst（HW 按 32B 跨 A 方向 pad 读取）
            int64_t innerAProd = 1;
            for (int32_t k = td_->aSplitIdx + AXIS_INTERVAL; k <= lastA; k += AXIS_INTERVAL) {
                innerAProd *= td_->axisShape[k];
            }
            outParams.blockLen = static_cast<uint32_t>(lastASize * static_cast<int64_t>(sizeof(DT)));
            outParams.blockCount = static_cast<uint16_t>(aLen * innerAProd / lastASize);
        }
    }
    outParams.srcStride = 0; // UB 侧 gap=0，HW 自动按 CeilAlign(blockLen, 32B) 读取下一个块
    outParams.dstStride = 0; // GM 侧 byte 对齐，dense 写出
    outParams.rsv = 0;       // 必须显式填 0（datacopypad-rules）
    DataCopyPad(yGm_[outerOutOff], outLocal, outParams);
}

} // namespace NsLpNormReduce

#endif // LP_NORM_REDUCE_BASE_H_
