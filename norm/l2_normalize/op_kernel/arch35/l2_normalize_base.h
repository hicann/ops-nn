/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// =============================================================================
// norm/l2_normalize/op_kernel/arch35/l2_normalize_base.h
// =============================================================================
//
// ROLE: Base 模板 kernel 类（TPL_SEL_0，isGroup=0 / isEmptyTensor=0）——非空输入
//   的兜底分支：A 方向分核、核间零依赖（无 SyncAll）、无跨核归约。
// 计算链：x →square→ x² →Σ_axis→ s →max(·,eps)→ s_clamped →√→
//   denom →x/denom→ y：
//     reduce 遍：CopyIn(S1) → CastSquareVf(S2+S3, fp16 先扩位) → pad 清零(S3a,
//     pad_value=0) → [Phase A 尾块配对 MergeTmpBufVf(S3b)] → ReduceSum(S4, fp32,
//     dst 直写二分缓存树) → DoCachingVf(S5)；
//     S6+S7：PostElewise = Maxs(eps)+Sqrt 一条 VF → denom(B3)（eps 钳在平方和上，
//     先 max 后 sqrt）；
//     除法遍（S8–S12）：R 全载单遍（rLoopCntTotal==1，denom 驻 B3、x 驻 B0 免二次
//     读）或 R 切分两遍（denom 经 workspace GM 中转 S7a/S9）→ denom 广播物化(S9)
//     → DivCastVf(S10+S11, Div+缩位 Cast) → CopyOut y(S12)。
//
// Buffer（全部 TBuf 深度 1，VECCALC，禁 TQue）：
//   B0 preInBuf(D_T) / B1 preReduceResult(fp32, 除法遍复用为 y) /
//   B2 preReduceResultTail(fp32, denom_bcast) / B3 postReduceResult(denom) /
//   cacheBuf(16KB 二分缓存树)。
//
// group 模板（TPL_SEL_2）继承本类复用全部 VF/子步骤。
// =============================================================================

#ifndef OPS_NORM_L2_NORMALIZE_BASE_H_
#define OPS_NORM_L2_NORMALIZE_BASE_H_

#include "kernel_operator.h"            // Ascend C kernel framework
#include "adv_api/reduce/reduce.h"      // AscendC::ReduceSum (AR / RA)
#include "l2_normalize_tiling_struct.h" // L2NormalizeTilingData / MAX_PATTERN_RANK

namespace NsL2Normalize {

using namespace AscendC;

// ─── 常量（元素粒度口径禁止混用）───
constexpr uint32_t VL_BYTES = 256;                     // VL = 256B（dav 3510）
constexpr uint32_t REP_F32 = VL_BYTES / sizeof(float); // fp32 每 rep 64 lane
constexpr uint16_t REP_F32_U16 = static_cast<uint16_t>(REP_F32);
constexpr uint32_t UB_BLOCK_BYTES = 32;                           // UB datablock 32B
constexpr uint32_t UB_BLOCK_F32 = UB_BLOCK_BYTES / sizeof(float); // = 8（fp32 域取整专用）
constexpr int32_t AXIS_INTERVAL = 2;                              // 偶位 A / 奇位 R 的轴间距
constexpr int32_t REDUCE_SHAPE_DIM = 2;                           // ReduceSum srcShape 维数
constexpr uint64_t NEAREST_POW2_SMALL_BOUND = 2;                  // FindNearestPower2 小值界
constexpr int32_t UINT64_TOP_BIT_IDX = 63;
constexpr float PAD_CLEAR_VALUE = 0.0f;                               // sum reducer 单位元
constexpr int64_t BRC_BLOCKCNT_LIMIT = 4095;                          // DataCopyPad blockCount 上限
constexpr uint64_t LOOP_FIELD_LIMIT = static_cast<uint64_t>(1) << 21; // Loop size/UB stride 上界（开区间）
constexpr uint64_t GM_STRIDE_LIMIT = static_cast<uint64_t>(1) << 40;  // GM byte stride 上界（开区间）

// 扩位 b16 → fp32：sat=UNKNOWN、round=CAST_NONE（RoundMode 是顶层
// AscendC::RoundMode，不在 AscendC::Reg:: 下）
constexpr AscendC::Reg::CastTrait CAST_TRAIT_TO_FP32{AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
                                                     AscendC::Reg::MaskMergeMode::ZEROING,
                                                     AscendC::RoundMode::CAST_NONE};

// 缩位 fp32 → fp16：sat=NO_SAT、round=CAST_RINT 就近舍入向偶数（对齐 TBE cast_to 口径）
constexpr AscendC::Reg::CastTrait CAST_TRAIT_FROM_FP32_FP16{
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};

// 高精度模式（设计文档 §7「精度处理」认可的备选：ST 实测超差时切 0-ULP，不换数据流）：
//   Sqrt 恒 0-ULP FTZ_FALSE（fp16/fp32 同）：默认 INTRINSIC Sqrt 对 subnormal 输入 FTZ
//   刷 0，eps<=0 且 x² 落 fp32 subnormal 窗口（|x|<1.08e-19）时 denom 被刷成 0 →
//   y=±inf，与 golden（numpy IEEE 保留 subnormal）NaN/Inf mismatch（FR-P1-001 L1_094）。
//   Div 仅 fp16 路径切 0-ULP：默认 Div 对 subnormal 输出 FTZ + 最大 1 ULP，fp16
//   subnormal 输出区间（|y| < 6.1e-5）1 ULP 位差即被 stat_rel_err mare 放大成超差；
//   fp32 subnormal 输出（|y| < 1.18e-38）恒小于用例 absolute_precision（1e-8）被
//   绝对容差吸收，默认 Div 足够。
constexpr AscendC::Reg::DivSpecificMode DIV_MODE_0ULP{AscendC::Reg::MaskMergeMode::ZEROING, true,
                                                      AscendC::DivAlgo::PRECISION_0ULP_FTZ_FALSE};
constexpr AscendC::Reg::SqrtSpecificMode SQRT_MODE_0ULP{AscendC::Reg::MaskMergeMode::ZEROING, true,
                                                        AscendC::SqrtAlgo::PRECISION_0ULP_FTZ_FALSE};

// ─── 整数对齐辅助（kernel 侧无 Ops::Base，逐条内联展开）───
__aicore__ inline uint32_t CeilDivU32(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

__aicore__ inline uint32_t CeilAlignU32(uint32_t a, uint32_t b) { return CeilDivU32(a, b) * b; }

__aicore__ inline int64_t CeilDivI64(int64_t a, int64_t b) { return a / b + ((a % b) != 0 ? 1 : 0); }

// ════════════════════════════════════════════════════════════════════════════
// __simd_vf__ 函数（asc_vf_call 调用目标；VF 7 条硬约束：for 从 0 起、≤4 层、
// uint16_t 循环变量、禁数组、禁对象成员、禁运行时 if/else、asc_vf_call 启动）
// ════════════════════════════════════════════════════════════════════════════

// S2+S3 PreElewise：Cast(b16→fp32) + Square 同一 VF（fp32 实例链长 1）
template <typename DType>
__simd_vf__ inline void CastSquareVfImpl(__ubuf__ DType* src, __ubuf__ float* dst, uint32_t totalElems,
                                         uint16_t repeatTime)
{
    constexpr bool IsFp32 = AscendC::IsSameType<DType, float>::value;
    AscendC::Reg::RegTensor<float> f32Reg;
    AscendC::Reg::MaskReg mask;
    uint32_t remaining = totalElems;

    for (uint16_t i = 0; i < repeatTime; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(REP_F32);
        mask = AscendC::Reg::UpdateMask<float>(remaining);

        if constexpr (IsFp32) {
            AscendC::Reg::LoadAlign(f32Reg, src + off); // fp32 原生装载，无 Cast
        } else {
            AscendC::Reg::RegTensor<DType> b16Reg;
            AscendC::Reg::LoadAlign<DType, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(b16Reg, src + off);
            AscendC::Reg::Cast<float, DType, CAST_TRAIT_TO_FP32>(f32Reg, b16Reg, mask);
        }

        AscendC::Reg::Mul(f32Reg, f32Reg, f32Reg, mask); // square：x*x（NaN/Inf IEEE 传播）
        AscendC::Reg::StoreAlign(dst + off, f32Reg, mask);
    }
}

// S3a tail-R ExtensionPad 清零：partial chunk 的 stale 区域按 A entry 逐行清（§9.3）
__simd_vf__ inline void ClearChunkExtTailRVfImpl(__ubuf__ float* base, uint32_t extStart, uint32_t aStride,
                                                 uint32_t extLanes, uint16_t aU16, uint16_t repPerA)
{
    AscendC::Reg::RegTensor<float> idReg;
    AscendC::Reg::Duplicate(idReg, PAD_CLEAR_VALUE);

    for (uint16_t aIdx = 0; aIdx < aU16; ++aIdx) {
        int32_t aOff = static_cast<int32_t>(aIdx) * static_cast<int32_t>(aStride);
        uint32_t remaining = extLanes;
        for (uint16_t r = 0; r < repPerA; ++r) {
            int32_t off = aOff + static_cast<int32_t>(extStart) +
                          static_cast<int32_t>(r) * static_cast<int32_t>(REP_F32);
            auto mask = AscendC::Reg::UpdateMask<float>(remaining);
            AscendC::Reg::StoreAlign(base + off, idReg, mask);
        }
    }
}

// S3a tail-A ExtensionPad 清零：rSplit 在 UB 最外层，单段连续清零（§9.3）
__simd_vf__ inline void ClearChunkExtTailAVfImpl(__ubuf__ float* base, uint32_t startElem, uint32_t totalClear,
                                                 uint16_t repCount)
{
    AscendC::Reg::RegTensor<float> idReg;
    AscendC::Reg::Duplicate(idReg, PAD_CLEAR_VALUE);
    AscendC::Reg::MaskReg mask;
    uint32_t remaining = totalClear;
    for (uint16_t i = 0; i < repCount; ++i) {
        int32_t off = static_cast<int32_t>(startElem) + static_cast<int32_t>(i) * static_cast<int32_t>(REP_F32);
        mask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::StoreAlign(base + off, idReg, mask);
    }
}

// S3a BurstPad 清零（tail-R burst 尾轴非对齐，§9.3）：StoreAlign 起点 FloorAlign +
//   mask 三段（maskEnd ∧ ¬maskStart 精确覆盖 [validR, padEndInRow)）
__simd_vf__ inline void ClearInnerBurstTailPadVfImpl(__ubuf__ float* base, uint16_t rowCntU16, int32_t rowStrideI,
                                                     int32_t windowOff, uint32_t padEnd, uint32_t partialStartInBlock)
{
    AscendC::Reg::RegTensor<float> idReg;
    AscendC::Reg::Duplicate(idReg, PAD_CLEAR_VALUE);

    // cntEnd/cntStart 必须是独立的非 const 局部变量：UpdateMask 引用语义会消耗它们
    uint32_t cntEnd = padEnd;
    uint32_t cntStart = partialStartInBlock;
    auto maskEnd = AscendC::Reg::UpdateMask<float>(cntEnd);
    auto maskStart = AscendC::Reg::UpdateMask<float>(cntStart);
    auto allMask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::MaskReg notStart, padMask;
    AscendC::Reg::Not(notStart, maskStart, allMask);
    AscendC::Reg::And(padMask, maskEnd, notStart, allMask); // [partialStartInBlock, padEnd) 置 1

    for (uint16_t row = 0; row < rowCntU16; ++row) {
        int32_t rowOff = static_cast<int32_t>(row) * rowStrideI;
        AscendC::Reg::StoreAlign(base + rowOff + windowOff, idReg, padMask);
    }
}

// S3b Phase A 主尾配对逐元素相加：main ⊕ tail → main（§9.2）
__simd_vf__ inline void MergeTmpBufVfImpl(__ubuf__ float* mainBuf, __ubuf__ float* tailBuf, uint32_t totalElems,
                                          uint16_t repeatTime)
{
    AscendC::Reg::RegTensor<float> aReg, bReg;
    AscendC::Reg::MaskReg mask;
    uint32_t remaining = totalElems;
    for (uint16_t i = 0; i < repeatTime; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(REP_F32);
        mask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::LoadAlign(aReg, mainBuf + off);
        AscendC::Reg::LoadAlign(bReg, tailBuf + off);
        AscendC::Reg::Add(aReg, aReg, bReg, mask);
        AscendC::Reg::StoreAlign(mainBuf + off, aReg, mask);
    }
}

// S5 二分缓存树正序吸收 + 覆盖写（§9.2；ReduceSum 已把结果写入本层 cacheBuf[levelOff]）
__simd_vf__ inline void DoCachingVfImpl(__ubuf__ float* cacheBuf, uint32_t laneN, uint32_t levelStride,
                                        int32_t levelOff, uint16_t repeatTime, uint16_t cacheLevelCnt)
{
    AscendC::Reg::RegTensor<float> aReg, bReg;
    AscendC::Reg::MaskReg mask;
    uint32_t remaining = laneN;
    for (uint16_t i = 0; i < repeatTime; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(REP_F32);
        mask = AscendC::Reg::UpdateMask<float>(remaining);

        AscendC::Reg::LoadAlign(aReg, cacheBuf + levelOff + off);

        for (uint16_t j = 0; j < cacheLevelCnt; ++j) { // 正序吸收所有低层（顺序不可换）
            int32_t lowerLevelOff = static_cast<int32_t>(j) * static_cast<int32_t>(levelStride) + off;
            AscendC::Reg::LoadAlign(bReg, cacheBuf + lowerLevelOff);
            AscendC::Reg::Add(aReg, aReg, bReg, mask);
        }
        AscendC::Reg::StoreAlign(cacheBuf + levelOff + off, aReg, mask);
    }
}

// ═══ fp16 精确求和 VF 链（补偿求和，误差 O(N·ε²)，等效 golden 的 fp64 累加单次舍回；
//     详见 DoOneAChunk fp16 分支注释）═══

// 连续 fp32 区域清零（(s,c) 累加器初始化 / E 数组初始化；mask 按余量精确截断，不越界）
// ⚠ UpdateMask(uint32_t&) 引用语义会消耗入参（POST_UPDATE 自减 64），必须传独立左值
__simd_vf__ inline void ZeroRegionVfImpl(__ubuf__ float* base, uint32_t totalElems, uint16_t repCnt)
{
    AscendC::Reg::RegTensor<float> idReg;
    AscendC::Reg::Duplicate(idReg, PAD_CLEAR_VALUE);
    AscendC::Reg::MaskReg mask;
    for (uint16_t i = 0; i < repCnt; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(REP_F32);
        uint32_t remaining = totalElems - static_cast<uint32_t>(off);
        mask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::StoreAlign(base + off, idReg, mask);
    }
}

// tail-A 行式 Kahan：tile [rowCnt][laneStride]（R 外 / A 内，dense），逐行把 x² 行切片
// 补偿并入 (s, c)。按 64-lane rep 分组，(s, c) 常驻寄存器跨行累加，rep 首尾经 UB 续算
// （跨 rLoop 持久）。pad 行/巷已被 Clear*Vf 清零（0 为补偿求和精确单位元）。
// 补偿步为 Neumaier（顺序无关）：x² 恒非负 ⇒ p > s 即 |p|>|s|，逐 lane 选正序 Fast2Sum
// 残差——经典 Kahan 的 Fast2Sum(s, y) 在 |y|>|s| 时 (t-s) 非 Sterbenz 区间、残差丢失，
// 真机实测致 s 偏 1 ULP → fp16 次正规输出翻转（FR-P1-002 残差形态）。
__simd_vf__ inline void KahanRowsVfImpl(__ubuf__ float* tile, __ubuf__ float* sPtr, __ubuf__ float* cPtr,
                                        uint16_t rowCnt, uint32_t laneStride, uint32_t laneN, uint16_t repCnt)
{
    AscendC::Reg::RegTensor<float> sReg, cReg, vReg, tReg, ebReg, esReg, errReg;
    AscendC::Reg::MaskReg mask, pgt;
    for (uint16_t rep = 0; rep < repCnt; ++rep) {
        int32_t repOff = static_cast<int32_t>(rep) * static_cast<int32_t>(REP_F32);
        uint32_t remaining = laneN - static_cast<uint32_t>(repOff);
        mask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::LoadAlign(sReg, sPtr + repOff);
        AscendC::Reg::LoadAlign(cReg, cPtr + repOff);
        for (uint16_t r = 0; r < rowCnt; ++r) {
            int32_t off = static_cast<int32_t>(r) * static_cast<int32_t>(laneStride) + repOff;
            AscendC::Reg::LoadAlign(vReg, tile + off);
            AscendC::Reg::Add(tReg, sReg, vReg, mask);   // t = s + p
            AscendC::Reg::Sub(ebReg, vReg, tReg, mask);  // (p - t)
            AscendC::Reg::Add(ebReg, ebReg, sReg, mask); // eBig = (p - t) + s（|p|≥|s| 精确）
            AscendC::Reg::Sub(esReg, sReg, tReg, mask);  // (s - t)
            AscendC::Reg::Add(esReg, esReg, vReg, mask); // eSml = (s - t) + p（|s|>|p| 精确）
            AscendC::Reg::Compare<float, AscendC::CMPMODE::GT>(pgt, vReg, sReg, mask); // 非负域：p>s ⟺ |p|>|s|
            AscendC::Reg::Select<float>(errReg, ebReg, esReg, pgt);
            AscendC::Reg::Add(cReg, cReg, errReg, mask); // c += err
            AscendC::Reg::Move(sReg, tReg);              // s = t
        }
        AscendC::Reg::StoreAlign(sPtr + repOff, sReg, mask);
        AscendC::Reg::StoreAlign(cPtr + repOff, cReg, mask);
    }
}

// tail-R 逐巷补偿求和（gather 馈送）：tile [laneA][rPadded]（A 外 / R 内）。R 内层使
// A 向量化的连续 Load 不可达（行切片按 rPadded 跨步、且尾块 lane 基址 4B 粒度非
// 32B 对齐，真机 aivec errcode 340），改用 Reg::Gather（vgather2，u32 逐 lane 索引、
// 4B 粒度，CANN cumsum 转置同款原语）按行距取同一 R 巷的 64 个 A 值：
//   p[i] = T[(a0 + i) × rPadded + rLane]
// 再做与 KahanRowsVfImpl 完全同构的 Neumaier 补偿步并入 (s, c)（A 向量化、跨 rLoop 持久；
// 顺序无关补偿，见 KahanRowsVfImpl 注释）。pad 行/巷已被 Clear*Vf 清零（0 为精确单位元）；
// 对齐纪律：LoadAlign/StoreAlign 基址均为 64-lane 倍（sPtr/cPtr + repOff），gather 基址
// tPtr 对齐、偏移全部在索引内。
__simd_vf__ inline void KahanMergeRowsGatherVfImpl(__ubuf__ float* tPtr, __ubuf__ float* sPtr, __ubuf__ float* cPtr,
                                                   uint32_t rPadded, uint32_t laneN, uint16_t repCnt)
{
    AscendC::Reg::RegTensor<float> sReg, cReg, pReg, tReg, ebReg, esReg, errReg;
    AscendC::Reg::RegTensor<uint32_t> idxMul, idx;
    AscendC::Reg::MaskReg mask, pgt;
    AscendC::Reg::MaskReg allMask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::Arange(reinterpret_cast<AscendC::Reg::RegTensor<int32_t>&>(idxMul), static_cast<int32_t>(0));
    AscendC::Reg::Muls(idxMul, idxMul, rPadded, allMask); // idxMul[i] = i × rPadded（行距）
    for (uint16_t rep = 0; rep < repCnt; ++rep) {
        int32_t repOff = static_cast<int32_t>(rep) * static_cast<int32_t>(REP_F32);
        uint32_t remaining = laneN - static_cast<uint32_t>(repOff);
        mask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::LoadAlign(sReg, sPtr + repOff);
        AscendC::Reg::LoadAlign(cReg, cPtr + repOff);
        for (uint32_t rLane = 0; rLane < rPadded; ++rLane) {
            // idx[i] = (repOff + i) × rPadded + rLane；Addend = repOff×rPadded + rLane < 2^31
            AscendC::Reg::Adds(idx, idxMul, static_cast<uint32_t>(repOff) * rPadded + rLane, mask);
            AscendC::Reg::Gather(pReg, tPtr, idx, mask); // p[i] = T[行(repOff+i) 的 rLane 巷]
            AscendC::Reg::Add(tReg, sReg, pReg, mask);   // t = s + p
            AscendC::Reg::Sub(ebReg, pReg, tReg, mask);  // (p - t)
            AscendC::Reg::Add(ebReg, ebReg, sReg, mask); // eBig = (p - t) + s（|p|≥|s| 精确）
            AscendC::Reg::Sub(esReg, sReg, tReg, mask);  // (s - t)
            AscendC::Reg::Add(esReg, esReg, pReg, mask); // eSml = (s - t) + p（|s|>|p| 精确）
            AscendC::Reg::Compare<float, AscendC::CMPMODE::GT>(pgt, pReg, sReg, mask); // 非负域：p>s ⟺ |p|>|s|
            AscendC::Reg::Select<float>(errReg, ebReg, esReg, pgt);
            AscendC::Reg::Add(cReg, cReg, errReg, mask); // c += err
            AscendC::Reg::Move(sReg, tReg);              // s = t
        }
        AscendC::Reg::StoreAlign(sPtr + repOff, sReg, mask);
        AscendC::Reg::StoreAlign(cPtr + repOff, cReg, mask);
    }
}

// 收尾：s_final = s + c（单次舍入 ⇒ 总和正确舍入值），写回 sPtr 供 PostElewise 消费
__simd_vf__ inline void FinalizeSumVfImpl(__ubuf__ float* sPtr, __ubuf__ float* cPtr, uint32_t laneN, uint16_t repCnt)
{
    AscendC::Reg::RegTensor<float> sReg, cReg;
    AscendC::Reg::MaskReg mask;
    for (uint16_t i = 0; i < repCnt; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(REP_F32);
        uint32_t remaining = laneN - static_cast<uint32_t>(off);
        mask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::LoadAlign(sReg, sPtr + off);
        AscendC::Reg::LoadAlign(cReg, cPtr + off);
        AscendC::Reg::Add(sReg, sReg, cReg, mask);
        AscendC::Reg::StoreAlign(sPtr + off, sReg, mask);
    }
}

// S6+S7 PostElewise：Maxs(eps) + Sqrt 同一 VF（全 fp32 域；eps 钳在平方和上，§9.6）
// Sqrt 恒 0-ULP FTZ_FALSE（IEEE 正确舍入，subnormal 输入不刷零，见 SQRT_MODE_0ULP 注释）。
__simd_vf__ inline void PostElewiseVfImpl(__ubuf__ float* rootPtr, __ubuf__ float* denomPtr, uint32_t laneN, float eps,
                                          uint16_t repeatTime)
{
    AscendC::Reg::RegTensor<float> f32Reg;
    AscendC::Reg::MaskReg mask;
    uint32_t remaining = laneN;
    for (uint16_t i = 0; i < repeatTime; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(REP_F32);
        mask = AscendC::Reg::UpdateMask<float>(remaining);

        AscendC::Reg::LoadAlign(f32Reg, rootPtr + off); // 树根 s（fp32，padded 整行）
        AscendC::Reg::Maxs(f32Reg, f32Reg, eps, mask);  // s_clamped = max(s, eps)
        // eps may be zero or negative per the public contract. Since s is a square sum, max(s, eps) remains
        // nonnegative (apart from IEEE NaN propagation); an all-zero slice with eps <= 0 intentionally yields
        // a zero denominator, matching canndev semantics.
        AscendC::Reg::Sqrt<float, &SQRT_MODE_0ULP>(f32Reg, f32Reg, mask); // denom = sqrt(s_clamped)
        AscendC::Reg::StoreAlign(denomPtr + off, f32Reg, mask);
    }
}

// S9 tail-A 广播物化：denom 行装载一次、逐 R 行复写（[rBundle, laneA]，§9.7）
__simd_vf__ inline void BroadcastDenomTailAVfImpl(__ubuf__ float* denomPtr, __ubuf__ float* bcastPtr, uint32_t laneA,
                                                  uint32_t rBundle, uint16_t repTime)
{
    AscendC::Reg::RegTensor<float> dReg;
    AscendC::Reg::MaskReg mask;
    uint32_t remaining = laneA;
    for (uint16_t i = 0; i < repTime; ++i) {
        mask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::LoadAlign(dReg, denomPtr + static_cast<int32_t>(i) * static_cast<int32_t>(REP_F32));
        for (uint16_t r = 0; r < static_cast<uint16_t>(rBundle); ++r) {
            AscendC::Reg::StoreAlign(bcastPtr + static_cast<int32_t>(r) * static_cast<int32_t>(laneA) +
                                         static_cast<int32_t>(i) * static_cast<int32_t>(REP_F32),
                                     dReg, mask);
        }
    }
}

// S9 tail-R 广播物化：DIST_BRC_B32 逐 A 行标量广播（[laneA, rBundle]，§9.7）
__simd_vf__ inline void BroadcastDenomTailRVfImpl(__ubuf__ float* denomPtr, __ubuf__ float* bcastPtr, uint32_t laneA,
                                                  uint32_t rBundle, uint16_t repPerRow)
{
    AscendC::Reg::RegTensor<float> dReg;
    AscendC::Reg::MaskReg mask;
    for (uint16_t a = 0; a < static_cast<uint16_t>(laneA); ++a) {
        AscendC::Reg::LoadAlign<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(dReg, denomPtr + a);
        uint32_t remaining = rBundle;
        for (uint16_t j = 0; j < repPerRow; ++j) {
            mask = AscendC::Reg::UpdateMask<float>(remaining);
            AscendC::Reg::StoreAlign(bcastPtr + static_cast<int32_t>(a) * static_cast<int32_t>(rBundle) +
                                         static_cast<int32_t>(j) * static_cast<int32_t>(REP_F32),
                                     dReg, mask);
        }
    }
}

// S10+S11 除法遍：y = x / denom_bcast（+ fp16 扩位/缩位 Cast，§9.7）
// HiPrec：fp16 路径 Div 切 0-ULP（IEEE 正确舍入，见 DIV_MODE_0ULP 注释）。
template <typename DType, bool HiPrecDiv>
__simd_vf__ inline void DivCastVfImpl(__ubuf__ DType* xPost, __ubuf__ float* denomBcast, __ubuf__ DType* y,
                                      uint32_t totalElems, uint16_t repeatTime)
{
    constexpr bool IsFp32 = AscendC::IsSameType<DType, float>::value;
    AscendC::Reg::RegTensor<float> xReg, dReg;
    AscendC::Reg::MaskReg mask;
    uint32_t remaining = totalElems;
    for (uint16_t i = 0; i < repeatTime; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(REP_F32);
        mask = AscendC::Reg::UpdateMask<float>(remaining);

        AscendC::Reg::LoadAlign(dReg, denomBcast + off); // denom_bcast（fp32，已按 x_post 布局物化）
        if constexpr (IsFp32) {
            AscendC::Reg::LoadAlign(xReg, xPost + off); // fp32 原生装载
            AscendC::Reg::Div(xReg, xReg, dReg, mask);  // y = x / denom
            AscendC::Reg::StoreAlign(y + off, xReg, mask);
        } else {
            AscendC::Reg::RegTensor<DType> xB16, yB16;
            AscendC::Reg::LoadAlign<DType, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(xB16, xPost + off);
            AscendC::Reg::Cast<float, DType, CAST_TRAIT_TO_FP32>(xReg, xB16, mask); // x 扩位（同 S2）
            if constexpr (HiPrecDiv) {
                AscendC::Reg::Div<float, &DIV_MODE_0ULP>(xReg, xReg, dReg, mask); // y = x / denom（0-ULP）
            } else {
                AscendC::Reg::Div(xReg, xReg, dReg, mask); // y = x / denom
            }
            AscendC::Reg::Cast<DType, float, CAST_TRAIT_FROM_FP32_FP16>(yB16, xReg, mask); // y 缩位
            AscendC::Reg::StoreAlign<DType, AscendC::Reg::StoreDist::DIST_PACK_B32>(y + off, yB16, mask);
        }
    }
}

// UB 内一根轴的描述（innermost-first 排列）
struct UBAxisDesc {
    int32_t gmIdx;      // 合轴后 GM 轴下标（td_->axisStride[gmIdx] 为 GM stride，element 计）
    int64_t ubSize;     // 本次搬入实际元素数（split 轴 = aLen/rLen，含尾块 valid）
    int64_t paddedSize; // UB 内该轴步长（元素计；供 dstStride / ubStride 计算）
    int64_t gmStride;   // GM stride（element 计）
};

// Loop Mode 字段宽度不足时退回软件循环；UB stride 还须满足 32B 对齐。
// GM 侧起始地址 32B 对齐：官方 SetLoopModePara.md 约束说明"源操作数和目的操作数的
// 起始地址需要保证32字节对齐"。Loop Mode 使能期间发出的每条 DataCopyPad 的 GM
// 起始字节 = gmBase + (baseGmOff + Σ_{kk≥4} ix×gmStride[kk] + rowStart×gmStride[1])
// ×dtBytes（软件展开轴 kk≥4、axis1 分批 rowStart 才产生额外偏移）：
//   ① 首条指令：gmBase + baseGmOff×dtBytes 须对齐（gmBase 含 tensor 自身基址，
//      覆盖带 storage offset 的连续视图）；
//   ② 软件展开轴 kk≥4 仅在 ubSize>1 时产生非零偏移，该轴 gmStride×dtBytes 须对齐
//      （各块起始逐块变化，任一非对齐即整体回退）；
//   ③ axis1 分批（rows 超出单条 blockCount 上限时 rowStart 非零，步长
//      canBatchAxis1 ? 4095 : 1）时 gmStride[1]×dtBytes 须对齐（4095 与 32 互素，
//      4095×s≡0(mod32) ⟺ s≡0(mod32)，两种分批步长合并为同一判定）。
// UB 侧起始/迭代地址由 ubStride[k≥1] 恒 32B 对齐构造性保证（下方既有校验）。
__aicore__ inline bool CanUseCopyLoopMode(const UBAxisDesc ubAxes[], const int64_t ubStride[], int32_t axisCount,
                                          int64_t dtBytes, uint64_t gmBase, int64_t baseGmOff, bool canBatchAxis1)
{
    if (axisCount < 3) {
        return false;
    }
    const int32_t loopAxisCount = (axisCount < 4) ? axisCount : 4;
    for (int32_t i = 2; i < loopAxisCount; ++i) {
        if (ubAxes[i].ubSize <= 0 || static_cast<uint64_t>(ubAxes[i].ubSize) >= LOOP_FIELD_LIMIT ||
            ubAxes[i].gmStride < 0 ||
            static_cast<uint64_t>(ubAxes[i].gmStride) >= GM_STRIDE_LIMIT / static_cast<uint64_t>(dtBytes) ||
            ubStride[i] < 0 || static_cast<uint64_t>(ubStride[i]) >= LOOP_FIELD_LIMIT ||
            ubStride[i] % static_cast<int64_t>(UB_BLOCK_BYTES) != 0) {
            return false;
        }
    }
    // GM 起始地址 32B 对齐（①②③，见函数头注释；不满足则回退 softwareAxisBegin=2
    // 软件循环路径，与既有回退机制一致）
    if ((gmBase + static_cast<uint64_t>(baseGmOff) * static_cast<uint64_t>(dtBytes)) % UB_BLOCK_BYTES != 0) {
        return false;
    }
    for (int32_t i = 4; i < axisCount; ++i) {
        if (ubAxes[i].ubSize > 1 && ubAxes[i].gmStride * dtBytes % static_cast<int64_t>(UB_BLOCK_BYTES) != 0) {
            return false;
        }
    }
    const int64_t maxRowsPerCopy = canBatchAxis1 ? static_cast<int64_t>(BRC_BLOCKCNT_LIMIT) : 1;
    if (ubAxes[1].ubSize > maxRowsPerCopy && ubAxes[1].gmStride * dtBytes % static_cast<int64_t>(UB_BLOCK_BYTES) != 0) {
        return false;
    }
    return true;
}

// ════════════════════════════════════════════════════════════════════════════
// L2NormalizeBaseKernel — Base 模板 kernel 类（TPL_SEL_0）
// ════════════════════════════════════════════════════════════════════════════
template <typename DType>
class L2NormalizeBaseKernel {
public:
    __aicore__ inline L2NormalizeBaseKernel() {}

    // Init：GM 绑定 + TBuf 分配 + pattern/二分树派生量 + eventID（§5.1）
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const L2NormalizeTilingData* td,
                                AscendC::TPipe* pipe);
    // Process：blockIdx → [aLoopStart, aLoopEnd) 大小核映射 → 逐 aLoop 两路径（§5.3）
    __aicore__ inline void Process();

protected:
    // ─── 索引解码（§2 F1/F4 编码的逆运算）───
    __aicore__ inline void UnravelALoop(int64_t aLoopIdx, int64_t aIdx[], int64_t& aSplitChunkIdx) const;
    __aicore__ inline int64_t UnravelRLoop(int64_t rIdx, int64_t rOuterIdx[], int64_t& rChunkIdx, int64_t& rLen) const;

    // ─── 主流程 ───
    __aicore__ inline void DoOneAChunk(int64_t outerGmOff, int64_t aLen);          // reduce 遍 S1–S5
    __aicore__ inline void DoOneAChunkFp16Exact(int64_t outerGmOff, int64_t aLen); // reduce 遍（fp16 补偿求和）
    __aicore__ inline void DividePass(int64_t chunkGmOff, int64_t aLen, bool needCopyInX,
                                      int64_t wsSlotOff); // 除法遍 S8–S12
    __aicore__ inline void PostElewise();                 // S6+S7

    // ─── CopyIn / CopyOut（§5.2 / §5.4）───
    __aicore__ inline int32_t BuildUBAxes(int64_t aLen, int64_t rLen, UBAxisDesc out[]) const;
    __aicore__ inline void EmitTileCopyIn(int64_t baseGmOff, const UBAxisDesc ubAxes[], int32_t K,
                                          AscendC::LocalTensor<DType>& preInLocal); // 发射体（group P3 继承）
    __aicore__ inline void EmitTileCopyOut(int64_t baseGmOff, const UBAxisDesc ubAxes[], int32_t K,
                                           AscendC::LocalTensor<DType>& yLocal); // 发射体（group P3 继承）
    __aicore__ inline void DoCopyInTile(int64_t baseGmOff, int64_t aLen, int64_t rLen,
                                        AscendC::LocalTensor<DType>& preInLocal);
    __aicore__ inline void CopyInDenom(int64_t wsSlotOff); // S9 denom 搬入
    __aicore__ inline void DoCopyOutTile(int64_t baseGmOff, int64_t aLen, int64_t rLen,
                                         AscendC::LocalTensor<DType>& yLocal);
    __aicore__ inline void CopyOutDenomToWorkspace(int64_t aLoopIdx); // S7a denom 中转写

    // ─── VF 调用侧（成员变量提取为参数，VF 硬约束 5）───
    __aicore__ inline void CastSquareVf(__ubuf__ DType* src, __ubuf__ float* dst);
    __aicore__ inline void ClearChunkExtensionVf(__ubuf__ float* base, int64_t rLen);
    __aicore__ inline void ClearInnerBurstTailPadVf(__ubuf__ float* base, int64_t rLen);
    __aicore__ inline void MergeTmpBufVf(__ubuf__ float* mainBuf, __ubuf__ float* tailBuf);
    __aicore__ inline void DoCachingVf(uint16_t cacheID);
    __aicore__ inline void BroadcastDenomTailAVf();
    __aicore__ inline void BroadcastDenomTailRVf(__ubuf__ float* denomPtr);
    __aicore__ inline void DivCastVf(__ubuf__ DType* xPost, __ubuf__ DType* y);

    // ─── 辅助 ───
    __aicore__ inline int32_t LastAAxis() const;
    __aicore__ inline int32_t LastRAxis() const;
    __aicore__ inline uint16_t GetCacheID(int64_t idx) const;
    __aicore__ inline uint64_t FindNearestPower2(uint64_t v) const;
    __aicore__ inline uint64_t CalLog2(uint64_t v) const;

    // ─── tilingdata ───
    const L2NormalizeTilingData* td_ = nullptr;
    AscendC::TPipe* pipe_ = nullptr;

    // ─── pattern / 二分树派生量（Init 一次现算，§5.1）───
    bool isTailR_ = false;
    int64_t bisectionPos_ = 0;
    int64_t bisectionTail_ = 0;
    int64_t cacheCount_ = 0;
    int64_t outStride_[MAX_PATTERN_RANK] = {0}; // A 轴 dense stride（group P1 经继承消费）

    // ─── GM 张量 ───
    AscendC::GlobalTensor<DType> xGm_;
    AscendC::GlobalTensor<DType> yGm_;
    AscendC::GlobalTensor<float> wsGm_; // denom GM 中转区（仅 R 切分两遍消费）

    // ─── TBuf（深度 1，VECCALC，禁 TQue，§4 Buffer 划分表）───
    AscendC::TBuf<AscendC::TPosition::VECCALC> preInBuf_;            // B0: x tile（D_T 原始位）
    AscendC::TBuf<AscendC::TPosition::VECCALC> preReduceResult_;     // B1: x_square / y
    AscendC::TBuf<AscendC::TPosition::VECCALC> preReduceResultTail_; // B2: 尾块/denom_bcast
    AscendC::TBuf<AscendC::TPosition::VECCALC> postReduceResult_;    // B3: denom（A-shaped fp32）
    AscendC::TBuf<AscendC::TPosition::VECCALC> cacheBuf_;            // 二分缓存树（恒 16KB）

    // ─── eventID（FetchEventID 获取，禁硬编码；Set/Wait 严格一一配对，§5.5）───
    AscendC::TEventID evMTE2toV_ = 0;    // CopyIn(MTE2) 写 UB → V 读
    AscendC::TEventID evVtoMTE2_ = 0;    // V 读 B0/B2/B3 → MTE2 覆写（WAR）
    AscendC::TEventID evVtoMTE3_ = 0;    // V 写 B1/B3 → CopyOut(MTE3) 读
    AscendC::TEventID evMTE3toV_ = 0;    // MTE3 读 B1(y) → V 覆写（WAR）
    AscendC::TEventID evMTE3toMTE2_ = 0; // MTE3 完成 → MTE2 发起（R 切分两遍专用）
};

// ════════════════════════════════════════════════════════════════════════════
// 辅助函数实现
// ════════════════════════════════════════════════════════════════════════════
template <typename DType>
__aicore__ inline int32_t L2NormalizeBaseKernel<DType>::LastAAxis() const
{
    for (int32_t i = td_->axisNum - 1; i >= 0; --i) {
        if (i % AXIS_INTERVAL == 0) {
            return i;
        }
    }
    return 0;
}

template <typename DType>
__aicore__ inline int32_t L2NormalizeBaseKernel<DType>::LastRAxis() const
{
    for (int32_t i = td_->axisNum - 1; i >= 0; --i) {
        if (i % AXIS_INTERVAL == 1) {
            return i;
        }
    }
    return 1;
}

// FindNearestPower2：严格小于 v 的最大 2 的幂（例 0→0, 1→1, 2→1, 3→2, 4→2, 8→4, 20→16）
template <typename DType>
__aicore__ inline uint64_t L2NormalizeBaseKernel<DType>::FindNearestPower2(uint64_t v) const
{
    if (v == 0) {
        return 0;
    }
    if (v <= NEAREST_POW2_SMALL_BOUND) {
        return 1;
    }
    const uint64_t num = v - 1;
    const uint64_t pow = static_cast<uint64_t>(UINT64_TOP_BIT_IDX) -
                         static_cast<uint64_t>(AscendC::CountLeadingZero(num));
    return static_cast<uint64_t>(1) << pow;
}

// CalLog2：⌊log2(v)⌋（v ≥ 1）
template <typename DType>
__aicore__ inline uint64_t L2NormalizeBaseKernel<DType>::CalLog2(uint64_t v) const
{
    uint64_t res = 0;
    while (v > 1) {
        v >>= 1;
        ++res;
    }
    return res;
}

// GetCacheID：写入层级 = i 二进制末尾连续 1 的个数（例 0→0, 1→1, 2→0, 3→2, 7→3）
template <typename DType>
__aicore__ inline uint16_t L2NormalizeBaseKernel<DType>::GetCacheID(int64_t idx) const
{
    const uint64_t v = static_cast<uint64_t>(idx);
    return static_cast<uint16_t>(AscendC::GetBitCount<1>(v ^ (v + 1)) - 1);
}

// ════════════════════════════════════════════════════════════════════════════
// Init（§5.1）
// ════════════════════════════════════════════════════════════════════════════
template <typename DType>
__aicore__ inline void L2NormalizeBaseKernel<DType>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace,
                                                          const L2NormalizeTilingData* td, AscendC::TPipe* pipe)
{
    td_ = td;
    pipe_ = pipe;

    // ── GM 绑定 ──
    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DType*>(x));
    yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DType*>(y));
    wsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace)); // fp32 denom 中转区（§8）

    // ── TBuf 分配（B0–B3 + cacheBuf 共 5 份物理槽，深度 1，VECCALC；禁 TQue）──
    pipe_->InitBuffer(preInBuf_, static_cast<uint32_t>(td->preBufSize));
    pipe_->InitBuffer(preReduceResult_, static_cast<uint32_t>(td->preBufSize));
    pipe_->InitBuffer(preReduceResultTail_, static_cast<uint32_t>(td->preBufSize));
    pipe_->InitBuffer(postReduceResult_, static_cast<uint32_t>(td->postBufSize));
    pipe_->InitBuffer(cacheBuf_, static_cast<uint32_t>(td->cacheBufUbSize));

    // ── pattern / 二分树派生量（R 全载 rLoopCntTotal=1 → P=1、T=0、cacheCount=1，
    //    树根即 cacheBuf[0]；GetCacheID 序列 + 覆盖写保证 cacheBuf 不需预清零）──
    isTailR_ = (td->axisNum % AXIS_INTERVAL == 0); // 偶 → tail-R / 奇 → tail-A
    bisectionPos_ = static_cast<int64_t>(FindNearestPower2(static_cast<uint64_t>(td->rLoopCntTotal)));
    bisectionTail_ = td->rLoopCntTotal - bisectionPos_; // Phase A 主尾配对数
    cacheCount_ = static_cast<int64_t>(CalLog2(static_cast<uint64_t>(bisectionPos_))) + 1;

    // 输出端 A 轴 dense stride：base 自身不消费（y 写回走 x 原布局 axisStride，§5.4）；
    // group P1 partial 写 workspace 列偏移经继承消费。
    int64_t acc = 1;
    for (int32_t i = td->axisNum - 1; i >= 0; --i) {
        if (i % AXIS_INTERVAL == 0) { // 偶下标 = A 轴
            outStride_[i] = acc;
            acc *= td->axisShape[i];
        }
    }

    // ── eventID 获取（范式硬约束：FetchEventID 获取、禁硬编码）──
    evMTE2toV_ = pipe_->FetchEventID<HardEvent::MTE2_V>();
    evVtoMTE2_ = pipe_->FetchEventID<HardEvent::V_MTE2>();
    evVtoMTE3_ = pipe_->FetchEventID<HardEvent::V_MTE3>();
    evMTE3toV_ = pipe_->FetchEventID<HardEvent::MTE3_V>();
    evMTE3toMTE2_ = pipe_->FetchEventID<HardEvent::MTE3_MTE2>();
}

// ════════════════════════════════════════════════════════════════════════════
// 索引解码（§5.3；§2 F1/F4 编码的逆运算，chunk 在最内 row-major）
// ════════════════════════════════════════════════════════════════════════════

// UnravelALoop：aLoopIdx → (aIdx[] 外层 A 下标, aSplitChunkIdx)
template <typename DType>
__aicore__ inline void L2NormalizeBaseKernel<DType>::UnravelALoop(int64_t aLoopIdx, int64_t aIdx[],
                                                                  int64_t& aSplitChunkIdx) const
{
    int64_t rem = aLoopIdx;
    aSplitChunkIdx = rem % td_->aSplitChunkCnt;
    rem /= td_->aSplitChunkCnt;
    for (int32_t k = td_->aSplitIdx - AXIS_INTERVAL; k >= 0; k -= AXIS_INTERVAL) { // aSplit 左邻 A 向外
        aIdx[k] = rem % td_->axisShape[k];
        rem /= td_->axisShape[k];
    }
}

// UnravelRLoop：rIdx → (rOuterIdx[] 外层 R 下标, rSplit chunk, rLen)；返回该 rChunk 的
// GM 偏移（element 计）。外层 R 轴 = 奇下标 < rSplitIdx（合轴 pattern 偶 A / 奇 R）。
template <typename DType>
__aicore__ inline int64_t L2NormalizeBaseKernel<DType>::UnravelRLoop(int64_t rIdx, int64_t rOuterIdx[],
                                                                     int64_t& rChunkIdx, int64_t& rLen) const
{
    const int64_t rChunksOnSplit = CeilDivI64(td_->axisShape[td_->rSplitIdx], td_->rUbFactor);
    rChunkIdx = rIdx % rChunksOnSplit;
    int64_t cur = rIdx / rChunksOnSplit;
    int64_t gmOff = 0;
    for (int32_t i = td_->rSplitIdx - AXIS_INTERVAL; i >= 1; i -= AXIS_INTERVAL) { // 外层 R 轴（奇下标）
        rOuterIdx[i] = cur % td_->axisShape[i];
        cur /= td_->axisShape[i];
        gmOff += rOuterIdx[i] * td_->axisStride[i];
    }
    const int64_t start = rChunkIdx * td_->rUbFactor;
    rLen = (start + td_->rUbFactor > td_->axisShape[td_->rSplitIdx]) ? (td_->axisShape[td_->rSplitIdx] - start) :
                                                                       td_->rUbFactor; // R 尾 chunk valid
    return gmOff + start * td_->axisStride[td_->rSplitIdx];
}

// ════════════════════════════════════════════════════════════════════════════
// Process（§5.3）：blockIdx → [aLoopStart, aLoopEnd) → 逐 aLoop
//   DoOneAChunk（reduce 遍）→ PostElewise（S6+S7）→ DividePass（除法遍）
// ════════════════════════════════════════════════════════════════════════════
template <typename DType>
__aicore__ inline void L2NormalizeBaseKernel<DType>::Process()
{
    const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    if (blockIdx >= static_cast<int64_t>(td_->usedCoreNum)) {
        return; // idle 核早退（aLoopCntTotal < coreNum，§0）
    }
    int64_t aLoopStart = 0;
    int64_t aLoopEnd = 0; // §2 F2 大小核映射
    if (blockIdx < static_cast<int64_t>(td_->aBigCoreCnt)) {
        aLoopStart = blockIdx * td_->aBigCoreLoopCnt;
        aLoopEnd = aLoopStart + td_->aBigCoreLoopCnt;
    } else {
        aLoopStart = static_cast<int64_t>(td_->aBigCoreCnt) * td_->aBigCoreLoopCnt +
                     (blockIdx - static_cast<int64_t>(td_->aBigCoreCnt)) * td_->aSmallCoreLoopCnt;
        aLoopEnd = aLoopStart + td_->aSmallCoreLoopCnt;
    }

    const int64_t aUnit = td_->aUbFactor * td_->innerAProdAlign;
    const int64_t aSplitAxisSize = td_->axisShape[td_->aSplitIdx];
    for (int64_t aLoopIdx = aLoopStart; aLoopIdx < aLoopEnd; ++aLoopIdx) {
        if (aLoopIdx != aLoopStart) {
            WaitFlag<HardEvent::MTE3_V>(evMTE3toV_); // 上一 aLoop 末轮 S12 读 B1(y) → 本 aLoop
                                                     // CastSquareVf 覆写 B1（WAR，§5.5 持有法则）
            WaitFlag<HardEvent::V_MTE2>(evVtoMTE2_); // 上一 aLoop 除法遍末轮 B0/B2 WAR（§5.5）
        }
        int64_t aIdx[MAX_PATTERN_RANK] = {0};
        int64_t aSplitChunkIdx = 0;
        UnravelALoop(aLoopIdx, aIdx, aSplitChunkIdx);

        const int64_t aChunkStart = aSplitChunkIdx * td_->aUbFactor;
        const int64_t aEndVal = aChunkStart + td_->aUbFactor;
        const int64_t aLen = (aEndVal > aSplitAxisSize) ? (aSplitAxisSize - aChunkStart) // A 尾块 valid
                                                          :
                                                          td_->aUbFactor;
        if (aLen <= 0) {
            continue; // 防御性兜底（aLoopCntTotal 严格上界，正常不触发）
        }

        int64_t chunkGmOff = 0;
        for (int32_t k = td_->aSplitIdx - AXIS_INTERVAL; k >= 0; k -= AXIS_INTERVAL) {
            chunkGmOff += aIdx[k] * td_->axisStride[k];
        }
        chunkGmOff += aChunkStart * td_->axisStride[td_->aSplitIdx];

        DoOneAChunk(chunkGmOff, aLen); // ── reduce 遍 S1–S5：s 就绪于 cacheBuf 树根 ──
        PostElewise();                 // ── S6+S7：denom = sqrt(max(s, eps)) → B3 ──

        if (td_->rLoopCntTotal == 1) {
            // 路径 A · R 全载单遍：denom 驻 B3，x_tile 驻 B0（S8 免二次读），单轮除法
            DividePass(chunkGmOff, aLen, /*needCopyInX=*/false, /*wsSlotOff=*/0);
        } else {
            // 路径 B · R 切分两遍：denom 中转写 workspace（S7a）后除法遍重扫 GM
            SetFlag<HardEvent::V_MTE2>(evVtoMTE2_); // reduce 遍全部 V 完成 → 除法遍首轮 S8 配对
            SetFlag<HardEvent::V_MTE3>(evVtoMTE3_);
            WaitFlag<HardEvent::V_MTE3>(evVtoMTE3_);      // B3: V 写 → MTE3 读
            CopyOutDenomToWorkspace(aLoopIdx);            // S7a（MTE3 写 ws；tail-R 读 B3）
            SetFlag<HardEvent::MTE3_MTE2>(evMTE3toMTE2_); // ws 写→读序 + tail-R B3 覆写（§5.5）
            DividePass(chunkGmOff, aLen, /*needCopyInX=*/true, /*wsSlotOff=*/aLoopIdx * aUnit);
        }
        if (aLoopIdx != aLoopEnd - 1) {
            SetFlag<HardEvent::MTE3_V>(evMTE3toV_); // 末轮 S12 读 B1(y) 完成 → 下一 aLoop
                                                    // CastSquareVf 覆写 B1（WAR 配对）
            SetFlag<HardEvent::V_MTE2>(evVtoMTE2_); // 除法遍末轮 B0/B2 WAR → 下一 aLoop S1 配对
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// reduce 遍（DoOneAChunk，S1–S5，Phase A 主尾配对 + Phase B 单块，§5.3）
//   fp32：硬件 ReduceSum + 二分缓存树（本实现，精度 O(log·ε)，fp32 容差内充裕）。
//   fp16：转 DoOneAChunkFp16Exact（补偿求和，见其头注释）。
// ════════════════════════════════════════════════════════════════════════════
template <typename DType>
__aicore__ inline void L2NormalizeBaseKernel<DType>::DoOneAChunk(int64_t outerGmOff, int64_t aLen)
{
    if constexpr (!AscendC::IsSameType<DType, float>::value) {
        DoOneAChunkFp16Exact(outerGmOff, aLen);
        return;
    }
    auto xLocal = preInBuf_.Get<DType>();
    auto preResLocal = preReduceResult_.Get<float>();
    auto preResTailLocal = preReduceResultTail_.Get<float>();
    __ubuf__ DType* xPtr = reinterpret_cast<__ubuf__ DType*>(xLocal.GetPhyAddr());
    __ubuf__ float* preResPtr = reinterpret_cast<__ubuf__ float*>(preResLocal.GetPhyAddr());
    __ubuf__ float* preResTailPtr = reinterpret_cast<__ubuf__ float*>(preResTailLocal.GetPhyAddr());

    for (int64_t rIdx = 0; rIdx < bisectionPos_; ++rIdx) {
        if (rIdx != 0) {
            WaitFlag<HardEvent::V_MTE2>(evVtoMTE2_); // B0 WAR：上一轮 CastSquareVf 已读 B0
        }
        int64_t rOuterIdx[MAX_PATTERN_RANK] = {0};
        int64_t rChunkMain = 0;
        int64_t rLenMain = 0;
        const int64_t rOffMain = UnravelRLoop(rIdx, rOuterIdx, rChunkMain, rLenMain);
        DoCopyInTile(outerGmOff + rOffMain, aLen, rLenMain, xLocal); // S1 主块 → B0
        SetFlag<HardEvent::MTE2_V>(evMTE2toV_);
        WaitFlag<HardEvent::MTE2_V>(evMTE2toV_); // B0 就绪

        CastSquareVf(xPtr, preResPtr); // S2+S3: x²（fp16: Cast↑+Mul 一条 VF）→ B1
        if (rLenMain < td_->rUbFactor) {
            ClearChunkExtensionVf(preResPtr, rLenMain); // S3a: ExtensionPad 清零（partial chunk）
        }
        if (isTailR_) {
            ClearInnerBurstTailPadVf(preResPtr, rLenMain); // S3a: BurstPad 清零（tail-R 非对齐）
        }
        if (rIdx < bisectionTail_) { // Phase A：尾块配对（M = P + T，§9.2）
            int64_t rOuterIdxT[MAX_PATTERN_RANK] = {0};
            int64_t rChunkTail = 0;
            int64_t rLenTail = 0;
            const int64_t rOffTail = UnravelRLoop(rIdx + bisectionPos_, rOuterIdxT, rChunkTail, rLenTail);
            SetFlag<HardEvent::V_MTE2>(evVtoMTE2_);
            WaitFlag<HardEvent::V_MTE2>(evVtoMTE2_);                     // B0 WAR：本轮 CastSquareVf 已读 B0
            DoCopyInTile(outerGmOff + rOffTail, aLen, rLenTail, xLocal); // S1 尾块（复用 B0）
            SetFlag<HardEvent::MTE2_V>(evMTE2toV_);
            WaitFlag<HardEvent::MTE2_V>(evMTE2toV_);
            CastSquareVf(xPtr, preResTailPtr); // S2+S3 → B2
            if (rLenTail < td_->rUbFactor) {
                ClearChunkExtensionVf(preResTailPtr, rLenTail);
            }
            if (isTailR_) {
                ClearInnerBurstTailPadVf(preResTailPtr, rLenTail);
            }
            MergeTmpBufVf(preResPtr, preResTailPtr); // S3b: main ⊕ tail → main（B1）
        }
        // S4: ReduceSum（AR/RA 按 tail 运行时二选一；src=B1、sharedTmpBuffer=B2、
        //     dst=cacheBuf[levelOff]；srcShape 按 padded 值、srcInnerPad=true、
        //     isReuseSource=true）
        const uint16_t cacheID = GetCacheID(rIdx);
        const uint32_t laneA = static_cast<uint32_t>(td_->aUbFactor * td_->innerAProdAlign);
        const uint32_t levelStride = CeilAlignU32(laneA, UB_BLOCK_F32);
        const int32_t levelOff = static_cast<int32_t>(cacheID) * static_cast<int32_t>(levelStride);
        if (isTailR_) {
            uint32_t srcShape[REDUCE_SHAPE_DIM] = {laneA,
                                                   static_cast<uint32_t>(td_->rUbFactorAlign * td_->innerRProdAlign)};
            AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, /*isReuseSource=*/true>(
                cacheBuf_.Get<float>()[levelOff], preResLocal, preReduceResultTail_.Get<uint8_t>(), srcShape,
                /*srcInnerPad=*/true);
        } else {
            uint32_t srcShape[REDUCE_SHAPE_DIM] = {static_cast<uint32_t>(td_->rUbFactorAlign * td_->innerRProdAlign),
                                                   laneA};
            AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, /*isReuseSource=*/true>(
                cacheBuf_.Get<float>()[levelOff], preResLocal, preReduceResultTail_.Get<uint8_t>(), srcShape,
                /*srcInnerPad=*/true);
        }
        DoCachingVf(cacheID); // S5: 二分缓存树正序吸收 + 覆盖写
        if (rIdx != bisectionPos_ - 1) {
            SetFlag<HardEvent::V_MTE2>(evVtoMTE2_); // 为下一轮 WaitFlag 配对
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// reduce 遍 fp16 精确求和（DoOneAChunkFp16Exact）
//   动机：golden（fp64 累加单次舍回）与 fp32 两级树求和存在 O(ε) 位差；fp16 输出落在
//   subnormal 区间（|y| < 6.1e-5）时，1 个 fp16 subnormal ULP 的相对误差可达 0.5，
//   stat_rel_err mare 判 FAIL（容差内无法吸收）。须使 fp32 平方和逐位等于
//   round_fp32(精确和)，下游 0-ULP sqrt/div + CAST_RINT 与 golden 逐位一致。
//   方案：Kahan 补偿求和（误差 O(N·ε²)）——
//     tail-A（tile [R][A]，R 外层）：KahanRowsVf 逐行并入 (s, c)（A 向量化）；
//     tail-R（tile [A][R]，R 内层，连续 Load 跨步不可达且尾块 lane 基址 4B 粒度非
//     32B 对齐）：KahanMergeRowsGatherVf 以 vgather2 逐 R 巷取 64 个 A 值，做与
//     tail-A 完全同构的 A 向量化 Kahan 并入 (s, c)。
//   缓冲：s → cacheBuf[0, laneA)，c → B3（postReduceResult)；cacheBuf 仅用 [0, laneA)。
//   B0 不参与（x tile 驻留，R 全载单遍除法免二次读）；B2 仅供除法遍 denom_bcast。
// ════════════════════════════════════════════════════════════════════════════
template <typename DType>
__aicore__ inline void L2NormalizeBaseKernel<DType>::DoOneAChunkFp16Exact(int64_t outerGmOff, int64_t aLen)
{
    auto xLocal = preInBuf_.Get<DType>();
    auto preResLocal = preReduceResult_.Get<float>();
    auto postLocal = postReduceResult_.Get<float>();
    auto cacheLocal = cacheBuf_.Get<float>();
    __ubuf__ DType* xPtr = reinterpret_cast<__ubuf__ DType*>(xLocal.GetPhyAddr());
    __ubuf__ float* preResPtr = reinterpret_cast<__ubuf__ float*>(preResLocal.GetPhyAddr());
    __ubuf__ float* cPtr = reinterpret_cast<__ubuf__ float*>(postLocal.GetPhyAddr());
    __ubuf__ float* sPtr = reinterpret_cast<__ubuf__ float*>(cacheLocal.GetPhyAddr());

    const uint32_t laneA = static_cast<uint32_t>(td_->aUbFactor * td_->innerAProdAlign);
    const uint32_t rPadded = static_cast<uint32_t>(td_->rUbFactorAlign * td_->innerRProdAlign);
    const uint16_t laneRep = static_cast<uint16_t>(CeilDivU32(laneA, static_cast<uint32_t>(REP_F32_U16)));

    // (s, c) 累加器清零（跨 rLoop 持久；每 aLoop 重置）
    asc_vf_call<ZeroRegionVfImpl>(sPtr, laneA, laneRep);
    asc_vf_call<ZeroRegionVfImpl>(cPtr, laneA, laneRep);

    for (int64_t rIdx = 0; rIdx < td_->rLoopCntTotal; ++rIdx) {
        if (rIdx != 0) {
            WaitFlag<HardEvent::V_MTE2>(evVtoMTE2_); // B0 WAR：上一轮 CastSquareVf 已读 B0
        }
        int64_t rOuterIdx[MAX_PATTERN_RANK] = {0};
        int64_t rChunkMain = 0;
        int64_t rLenMain = 0;
        const int64_t rOffMain = UnravelRLoop(rIdx, rOuterIdx, rChunkMain, rLenMain);
        DoCopyInTile(outerGmOff + rOffMain, aLen, rLenMain, xLocal); // S1 → B0
        SetFlag<HardEvent::MTE2_V>(evMTE2toV_);
        WaitFlag<HardEvent::MTE2_V>(evMTE2toV_); // B0 就绪

        CastSquareVf(xPtr, preResPtr); // S2+S3: x²（Cast↑+Mul）→ B1
        if (rLenMain < td_->rUbFactor) {
            ClearChunkExtensionVf(preResPtr, rLenMain); // S3a: ExtensionPad 清零（partial chunk）
        }
        if (isTailR_) {
            ClearInnerBurstTailPadVf(preResPtr, rLenMain); // S3a: BurstPad 清零（tail-R 非对齐）
        }

        if (isTailR_) {
            // tail-R：逐 R 巷 gather 取 x²（跨步访问 4B 粒度，vgather2）→ A 向量化 Kahan
            // 并入 (s, c)（与 tail-A 的 KahanRowsVfImpl 同构；B2 不参与）
            asc_vf_call<KahanMergeRowsGatherVfImpl>(preResPtr, sPtr, cPtr, rPadded, laneA, laneRep);
        } else {
            // tail-A：行式 Kahan 逐行并入 (s, c)（tile [rPadded][laneA] dense，pad 行已清零）
            asc_vf_call<KahanRowsVfImpl>(preResPtr, sPtr, cPtr, static_cast<uint16_t>(rPadded), laneA, laneA, laneRep);
        }
        if (rIdx != td_->rLoopCntTotal - 1) {
            SetFlag<HardEvent::V_MTE2>(evVtoMTE2_); // 为下一轮 S1 配对（B0 WAR）
        }
    }

    // 收尾：s_final = s + c（单次舍入）→ cacheBuf[0, laneA)（PostElewise 的 fp16 树根）
    asc_vf_call<FinalizeSumVfImpl>(sPtr, cPtr, laneA, laneRep);
}

// ════════════════════════════════════════════════════════════════════════════
// 除法遍（DividePass，S8–S12，§5.3）
//   R 全载单遍 needCopyInX=false（x_tile 驻 B0，仅 j=0 一轮，S9 为 B3→B2 纯 VF 广播）；
//   R 切分两遍逐 rChunk 循环：S8 x 二次读 + S9 denom 搬入/广播 → S10+S11 DivCastVf
//   → S12 CopyOut y。
// ════════════════════════════════════════════════════════════════════════════
template <typename DType>
__aicore__ inline void L2NormalizeBaseKernel<DType>::DividePass(int64_t chunkGmOff, int64_t aLen, bool needCopyInX,
                                                                int64_t wsSlotOff)
{
    auto xLocal = preInBuf_.Get<DType>();
    auto yLocal = preReduceResult_.Get<DType>(); // B1 复用为 y（§4 复用关系）
    __ubuf__ DType* xPtr = reinterpret_cast<__ubuf__ DType*>(xLocal.GetPhyAddr());
    __ubuf__ DType* yPtr = reinterpret_cast<__ubuf__ DType*>(yLocal.GetPhyAddr());
    __ubuf__ float* denomPtr = reinterpret_cast<__ubuf__ float*>(postReduceResult_.Get<float>().GetPhyAddr());

    for (int64_t j = 0; j < td_->rLoopCntTotal; ++j) {
        int64_t rOuterIdx[MAX_PATTERN_RANK] = {0};
        int64_t rChunkIdx = 0;
        int64_t rLen = 0;
        const int64_t rOff = UnravelRLoop(j, rOuterIdx, rChunkIdx, rLen);

        if (j == 0) {
            if (needCopyInX) {
                WaitFlag<HardEvent::MTE3_MTE2>(evMTE3toMTE2_); // ws 写→读序（+ tail-R B3 覆写）
                WaitFlag<HardEvent::V_MTE2>(evVtoMTE2_);       // reduce 遍末轮 B0/B2 WAR（Process 内 Set）
            }
        } else {
            WaitFlag<HardEvent::V_MTE2>(evVtoMTE2_); // 上一轮 DivCast/Broadcast 读 B0/B2/B3 WAR
        }
        if (needCopyInX) {
            DoCopyInTile(chunkGmOff + rOff, aLen, rLen, xLocal); // S8: x 二次读 → B0
            CopyInDenom(wsSlotOff);                              // S9: tail-A 广播直达 B2 / tail-R dense → B3
            SetFlag<HardEvent::MTE2_V>(evMTE2toV_);
            WaitFlag<HardEvent::MTE2_V>(evMTE2toV_); // B0/B2(或 B3) 就绪
            if (isTailR_) {
                BroadcastDenomTailRVf(denomPtr); // S9 tail-R: B3→B2 行常量（DIST_BRC_B32）
            }
        } else {
            if (isTailR_) {
                BroadcastDenomTailRVf(denomPtr); // S9 R 全载 tail-R: B3 驻留 → B2
            } else {
                BroadcastDenomTailAVf(); // S9 R 全载 tail-A: B3 驻留 → B2 行复制
            }
        }
        if (j != 0) {
            WaitFlag<HardEvent::MTE3_V>(evMTE3toV_); // B1(y) WAR：上一轮 CopyOut 已读
        }
        DivCastVf(xPtr, yPtr); // S10+S11: y = x / denom_bcast（+ fp16 Cast↓）→ B1
        SetFlag<HardEvent::V_MTE3>(evVtoMTE3_);
        WaitFlag<HardEvent::V_MTE3>(evVtoMTE3_);              // B1: V 写 → MTE3 读
        DoCopyOutTile(chunkGmOff + rOff, aLen, rLen, yLocal); // S12: y → GM（§5.4）
        if (j != td_->rLoopCntTotal - 1) {
            SetFlag<HardEvent::MTE3_V>(evMTE3toV_); // 为下一轮 DivCast 覆写 B1 配对
            SetFlag<HardEvent::V_MTE2>(evVtoMTE2_); // 为下一轮 S8/S9 搬入覆写 B0/B2(或 B3) 配对
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// PostElewise（S6+S7，§9.6）：cacheBuf 树根 → Maxs(eps)+Sqrt 一条 VF → denom(B3)
//   fp32：二分树根 [（cacheCount_-1)×levelStride)；fp16：补偿求和结果 cacheBuf[0, laneA)。
// ════════════════════════════════════════════════════════════════════════════
template <typename DType>
__aicore__ inline void L2NormalizeBaseKernel<DType>::PostElewise()
{
    constexpr bool kFp16Prec = !AscendC::IsSameType<DType, float>::value;
    const uint32_t laneN = static_cast<uint32_t>(td_->aUbFactor * td_->innerAProdAlign);
    const uint32_t levelStride = CeilAlignU32(laneN, UB_BLOCK_F32);
    const int32_t rootOff = kFp16Prec ? 0 : static_cast<int32_t>(cacheCount_ - 1) * static_cast<int32_t>(levelStride);

    __ubuf__ float* rootPtr = reinterpret_cast<__ubuf__ float*>(cacheBuf_.Get<float>().GetPhyAddr()) + rootOff;
    __ubuf__ float* denomPtr = reinterpret_cast<__ubuf__ float*>(postReduceResult_.Get<float>().GetPhyAddr());

    const uint16_t repeatTime = static_cast<uint16_t>(CeilDivU32(laneN, static_cast<uint32_t>(REP_F32_U16)));

    asc_vf_call<PostElewiseVfImpl>(rootPtr, denomPtr, laneN, td_->eps, repeatTime);
}

// ════════════════════════════════════════════════════════════════════════════
// BuildUBAxes（§5.2）：tail-R → 内 bundle = R、外 bundle = A；tail-A → 内 A、外 R。
//   out[0] 为 burst 尾轴（最内层，GM stride 1）；split 轴 padded 取
//   aUbFactor / rUbFactorAlign；非 split 的 burst 尾轴整根 CeilAlign（按 D_T 计）；
//   其余轴 ubSize == paddedSize（= axisShape 原值）。
// ════════════════════════════════════════════════════════════════════════════
template <typename DType>
__aicore__ inline int32_t L2NormalizeBaseKernel<DType>::BuildUBAxes(int64_t aLen, int64_t rLen, UBAxisDesc out[]) const
{
    int32_t k = 0;
    const int32_t lastA = LastAAxis();
    const int32_t lastR = LastRAxis();
    const int64_t bsElem = static_cast<int64_t>(UB_BLOCK_BYTES) / static_cast<int64_t>(sizeof(DType));

    if (isTailR_) {
        for (int32_t i = td_->axisNum - 1; i >= td_->rSplitIdx; --i) { // 内 bundle = R
            if (i % AXIS_INTERVAL != 1) {
                continue;
            }
            int64_t actual = 0;
            int64_t padded = 0;
            if (i == td_->rSplitIdx) {
                actual = rLen;
                padded = td_->rUbFactorAlign; // burst 尾轴全载非对齐时 > rUbFactor，其余相等
            } else if (i == lastR) {
                actual = td_->axisShape[i];
                padded = CeilAlignU32(static_cast<uint32_t>(actual), static_cast<uint32_t>(bsElem));
            } else {
                actual = padded = td_->axisShape[i];
            }
            out[k] = {i, actual, padded, td_->axisStride[i]};
            ++k;
        }
        for (int32_t i = td_->axisNum - 1; i >= td_->aSplitIdx; --i) { // 外 bundle = A
            if (i % AXIS_INTERVAL != 0) {
                continue;
            }
            const int64_t actual = (i == td_->aSplitIdx) ? aLen : td_->axisShape[i];
            const int64_t padded = (i == td_->aSplitIdx) ? td_->aUbFactor : td_->axisShape[i];
            out[k] = {i, actual, padded, td_->axisStride[i]}; // tail-R 下 A 无对齐要求
            ++k;
        }
    } else {
        for (int32_t i = td_->axisNum - 1; i >= td_->aSplitIdx; --i) { // 内 bundle = A
            if (i % AXIS_INTERVAL != 0) {
                continue;
            }
            int64_t actual = 0;
            int64_t padded = 0;
            if (i == td_->aSplitIdx) {
                actual = aLen;
                padded = td_->aUbFactor; // tail-A + aSplit==LastA 时 aUnit 对齐由 Host 爬坡保证
            } else if (i == lastA) {
                actual = td_->axisShape[i];
                padded = CeilAlignU32(static_cast<uint32_t>(actual), static_cast<uint32_t>(bsElem));
            } else {
                actual = padded = td_->axisShape[i];
            }
            out[k] = {i, actual, padded, td_->axisStride[i]};
            ++k;
        }
        for (int32_t i = td_->axisNum - 1; i >= td_->rSplitIdx; --i) { // 外 bundle = R
            if (i % AXIS_INTERVAL != 1) {
                continue;
            }
            const int64_t actual = (i == td_->rSplitIdx) ? rLen : td_->axisShape[i];
            const int64_t padded = (i == td_->rSplitIdx) ? td_->rUbFactorAlign : td_->axisShape[i];
            out[k] = {i, actual, padded, td_->axisStride[i]}; // tail-A 下 rUbFactorAlign == rUbFactor
            ++k;
        }
    }
    return k; // UB 内轴数 K（2 ≤ K ≤ axisNum）
}

// ════════════════════════════════════════════════════════════════════════════
// EmitTileCopyIn（S1 / S8 发射体，§5.2）：给定 UBAxisDesc 轴描述发射 DataCopyPad。
//   K 平台分级（字段范围内优先使用 950 Loop，超范围时由软件循环完整回退）；
//   isPad=false（pad 为脏数据，reduce 前由 Clear*Vf 清零）；MTE2 stride 单位
//   src=GM byte / dst=UB datablock(32B)；rsv 显式填 0。group Phase 3 的
//   DoCopyInTileP3 经继承以 P3 段轴描述复用。
// ════════════════════════════════════════════════════════════════════════════
template <typename DType>
__aicore__ inline void L2NormalizeBaseKernel<DType>::EmitTileCopyIn(int64_t baseGmOff, const UBAxisDesc ubAxes[],
                                                                    int32_t K, AscendC::LocalTensor<DType>& preInLocal)
{
    DataCopyExtParams extParams;
    LoopModeParams loopParams; // 构造函数已全 0 初始化
    const int64_t dtBytes = static_cast<int64_t>(sizeof(DType));
    extParams.blockLen = static_cast<uint32_t>(ubAxes[0].ubSize * dtBytes); // burst 尾轴 valid 字节

    DataCopyPadExtParams<DType> padParams{false, 0, 0, 0}; // UB 侧 HW 自动补 dummy（脏数据，S3a 清零）
    const int64_t copyPadBytes = CeilAlignU32(static_cast<uint32_t>(extParams.blockLen), UB_BLOCK_BYTES);
    const int64_t target0Bytes = ubAxes[0].paddedSize * dtBytes;
    // UB 侧块间 gap（datablock 单位）：跳过 [CeilAlign(blockLen,32B), paddedSize×s) 的 stale 段
    extParams.dstStride = static_cast<uint32_t>((target0Bytes - copyPadBytes) / static_cast<int64_t>(UB_BLOCK_BYTES));
    const int64_t gmGapBytes = (K >= 2) ? ubAxes[1].gmStride * dtBytes - static_cast<int64_t>(extParams.blockLen) : 0;
    const bool canBatchAxis1 = (gmGapBytes >= 0) && (static_cast<uint64_t>(gmGapBytes) < GM_STRIDE_LIMIT);
    extParams.srcStride = canBatchAxis1 ? gmGapBytes : 0;
    extParams.rsv = 0;

    // 每层 UB 字节步长 = ∏ paddedSize[0..i-1] × sizeof(D_T)（ubAxes[0].paddedSize 恒为
    // bsElem 倍数 ⇒ ubStride[k≥1] 恒 32B 对齐，Loop UB 侧对齐要求构造性满足）
    int64_t ubStride[MAX_PATTERN_RANK];
    ubStride[0] = dtBytes;
    for (int32_t i = 1; i < K; ++i) {
        ubStride[i] = ubStride[i - 1] * ubAxes[i - 1].paddedSize;
    }
    // GM 起始地址对齐校验含 xGm_ 基址（CopyIn 源侧；官方 SetLoopModePara.md 32B 约束）
    const bool useLoopMode = CanUseCopyLoopMode(
        ubAxes, ubStride, K, dtBytes, reinterpret_cast<uintptr_t>(xGm_.GetPhyAddr()), baseGmOff, canBatchAxis1);
    if (useLoopMode) {
        loopParams.loop1Size = static_cast<uint32_t>(ubAxes[2].ubSize);
        loopParams.loop1SrcStride = static_cast<uint64_t>(ubAxes[2].gmStride) * static_cast<uint64_t>(dtBytes);
        loopParams.loop1DstStride = static_cast<uint64_t>(ubStride[2]);
        loopParams.loop2Size = 1;
        if (K >= 4) {
            loopParams.loop2Size = static_cast<uint32_t>(ubAxes[3].ubSize);
            loopParams.loop2SrcStride = static_cast<uint64_t>(ubAxes[3].gmStride) * static_cast<uint64_t>(dtBytes);
            loopParams.loop2DstStride = static_cast<uint64_t>(ubStride[3]);
        }
        SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
    }

    // Loop Mode 不可编码时，从 axis 2 开始全部软件展开；可编码时只展开 axis 4 及以上。
    const int32_t softwareAxisBegin = useLoopMode ? 4 : 2;
    int64_t outerProd = 1;
    for (int32_t kk = softwareAxisBegin; kk < K; ++kk) {
        outerProd *= ubAxes[kk].ubSize;
    }
    for (int64_t outerFlat = 0; outerFlat < outerProd; ++outerFlat) {
        int64_t addGmOffElem = 0;
        int64_t addUbOffBytes = 0;
        int64_t cur = outerFlat;
        for (int32_t kk = softwareAxisBegin; kk < K; ++kk) {
            const int64_t ix = cur % ubAxes[kk].ubSize;
            cur /= ubAxes[kk].ubSize;
            addGmOffElem += ix * ubAxes[kk].gmStride;
            addUbOffBytes += ix * ubStride[kk];
        }
        const int64_t axis1Rows = (K >= 2) ? ubAxes[1].ubSize : 1;
        const int64_t maxRowsPerCopy = canBatchAxis1 ? BRC_BLOCKCNT_LIMIT : 1;
        for (int64_t rowStart = 0; rowStart < axis1Rows; rowStart += maxRowsPerCopy) {
            const int64_t rowsLeft = axis1Rows - rowStart;
            const int64_t rows = (rowsLeft < maxRowsPerCopy) ? rowsLeft : maxRowsPerCopy;
            extParams.blockCount = static_cast<uint16_t>(rows);
            const int64_t rowGmOff = (K >= 2) ? rowStart * ubAxes[1].gmStride : 0;
            const int64_t rowUbOffBytes = (K >= 2) ? rowStart * ubStride[1] : 0;
            DataCopyPad(preInLocal[(addUbOffBytes + rowUbOffBytes) / dtBytes],
                        xGm_[baseGmOff + addGmOffElem + rowGmOff], extParams, padParams);
        }
    }
    if (useLoopMode) {
        ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
    }
}

// ════════════════════════════════════════════════════════════════════════════
// DoCopyInTile（S1 / S8，§5.2）：GM → B0。轴描述由 BuildUBAxes 产出后经
//   EmitTileCopyIn 发射（发射体与 group P3 共享）。
// ════════════════════════════════════════════════════════════════════════════
template <typename DType>
__aicore__ inline void L2NormalizeBaseKernel<DType>::DoCopyInTile(int64_t baseGmOff, int64_t aLen, int64_t rLen,
                                                                  AscendC::LocalTensor<DType>& preInLocal)
{
    UBAxisDesc ubAxes[MAX_PATTERN_RANK];
    const int32_t K = BuildUBAxes(aLen, rLen, ubAxes);
    EmitTileCopyIn(baseGmOff, ubAxes, K, preInLocal);
}

// ════════════════════════════════════════════════════════════════════════════
// CopyInDenom（S9，§5.2）：ws denom 区 → B2（tail-A，srcStride=0 广播）/ B3（tail-R，
//   dense）。tail-A：blockCount = rBundle（>4095 分段循环，段内 srcStride=0 语义不变），
//   dstStride=0 = UB 行宽 aUnit×4B 紧排（tail-A 下 aUnit 必对齐）；GM 读取恒在 denom
//   区内（padded 槽位保证不越界）。
// ════════════════════════════════════════════════════════════════════════════
template <typename DType>
__aicore__ inline void L2NormalizeBaseKernel<DType>::CopyInDenom(int64_t wsSlotOff)
{
    const int64_t aUnit = td_->aUbFactor * td_->innerAProdAlign;
    // GM→UB 4 参形式：isPad=false（整 padded 行直读，UB 侧 32B dummy 段不被 valid 消费）
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
    if (isTailR_) {
        DataCopyExtParams ext{1, static_cast<uint32_t>(aUnit * static_cast<int64_t>(sizeof(float))), 0, 0, 0};
        DataCopyPad(postReduceResult_.Get<float>(), wsGm_[wsSlotOff], ext, padParams);
    } else {
        // tail-A 广播搬入：srcStride 为 gap 语义（块尾→下一块块头的额外空隙），重读同一
        // 源行须取 srcStride = −blockLen（datacopypad-rules：stride=0 是 dense 紧排而非重读；
        // GM 侧 byte 单位支持负值）；blockCount > 4095 分段循环，段内语义不变
        const int64_t rBundle = td_->rUbFactorAlign * td_->innerRProdAlign;
        const int64_t segCnt = CeilDivI64(rBundle, BRC_BLOCKCNT_LIMIT);
        const int64_t segRows = CeilDivI64(rBundle, segCnt);
        const int64_t blockLen = aUnit * static_cast<int64_t>(sizeof(float));
        auto bcastLocal = preReduceResultTail_.Get<float>();
        for (int64_t seg = 0; seg < segCnt; ++seg) {
            const int64_t rows = (segRows < rBundle - seg * segRows) ? segRows : (rBundle - seg * segRows);
            DataCopyExtParams ext{static_cast<uint16_t>(rows), static_cast<uint32_t>(blockLen), -blockLen, 0, 0};
            DataCopyPad(bcastLocal[seg * segRows * aUnit], wsGm_[wsSlotOff], ext, padParams);
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// EmitTileCopyOut（S12 发射体，§5.4）：给定 UBAxisDesc 轴描述发射 DataCopyPad。
//   写回几何为 EmitTileCopyIn 的镜像（同一轴描述、K 分级同款）；MTE3 方向
//   stride 单位 src=UB datablock(32B) / dst=GM byte；按 valid 段拷出
//   （pad/garbage 不出有效段，§9.7 pad 语义）。group Phase 3 的 DoCopyOutTileP3
//   经继承以 P3 段轴描述复用。
// ════════════════════════════════════════════════════════════════════════════
template <typename DType>
__aicore__ inline void L2NormalizeBaseKernel<DType>::EmitTileCopyOut(int64_t baseGmOff, const UBAxisDesc ubAxes[],
                                                                     int32_t K, AscendC::LocalTensor<DType>& yLocal)
{
    DataCopyExtParams extParams;
    LoopModeParams loopParams;
    const int64_t dtBytes = static_cast<int64_t>(sizeof(DType));
    extParams.blockLen = static_cast<uint32_t>(ubAxes[0].ubSize * dtBytes); // burst 尾轴 valid 字节
    const int64_t copyPadBytes = CeilAlignU32(static_cast<uint32_t>(extParams.blockLen), UB_BLOCK_BYTES);
    const int64_t src0Bytes = ubAxes[0].paddedSize * dtBytes;
    // UB 侧块间 gap（datablock 单位）：跳过 [CeilAlign(blockLen,32B), paddedSize×s) 的 pad/garbage 段
    extParams.srcStride = static_cast<uint32_t>((src0Bytes - copyPadBytes) / static_cast<int64_t>(UB_BLOCK_BYTES));
    const int64_t gmGapBytes = (K >= 2) ? ubAxes[1].gmStride * dtBytes - static_cast<int64_t>(extParams.blockLen) : 0;
    const bool canBatchAxis1 = (gmGapBytes >= 0) && (static_cast<uint64_t>(gmGapBytes) < GM_STRIDE_LIMIT);
    extParams.dstStride = canBatchAxis1 ? gmGapBytes : 0;
    extParams.rsv = 0;

    int64_t ubStride[MAX_PATTERN_RANK];
    ubStride[0] = dtBytes;
    for (int32_t i = 1; i < K; ++i) {
        ubStride[i] = ubStride[i - 1] * ubAxes[i - 1].paddedSize; // ubStride[k≥1] 恒 32B 对齐
    }
    // GM 起始地址对齐校验含 yGm_ 基址（CopyOut 目的侧；官方 SetLoopModePara.md 32B 约束）
    const bool useLoopMode = CanUseCopyLoopMode(
        ubAxes, ubStride, K, dtBytes, reinterpret_cast<uintptr_t>(yGm_.GetPhyAddr()), baseGmOff, canBatchAxis1);
    if (useLoopMode) {
        loopParams.loop1Size = static_cast<uint32_t>(ubAxes[2].ubSize);
        loopParams.loop1SrcStride = static_cast<uint64_t>(ubStride[2]); // UB 侧（byte，advance 语义）
        loopParams.loop1DstStride = static_cast<uint64_t>(ubAxes[2].gmStride) * static_cast<uint64_t>(dtBytes);
        loopParams.loop2Size = 1;
        if (K >= 4) {
            loopParams.loop2Size = static_cast<uint32_t>(ubAxes[3].ubSize);
            loopParams.loop2SrcStride = static_cast<uint64_t>(ubStride[3]);
            loopParams.loop2DstStride = static_cast<uint64_t>(ubAxes[3].gmStride) * static_cast<uint64_t>(dtBytes);
        }
        SetLoopModePara(loopParams, DataCopyMVType::UB_TO_OUT);
    }

    const int32_t softwareAxisBegin = useLoopMode ? 4 : 2;
    int64_t outerProd = 1;
    for (int32_t kk = softwareAxisBegin; kk < K; ++kk) {
        outerProd *= ubAxes[kk].ubSize;
    }
    for (int64_t outerFlat = 0; outerFlat < outerProd; ++outerFlat) {
        int64_t addGmOffElem = 0;
        int64_t addUbOffBytes = 0;
        int64_t cur = outerFlat;
        for (int32_t kk = softwareAxisBegin; kk < K; ++kk) {
            const int64_t ix = cur % ubAxes[kk].ubSize;
            cur /= ubAxes[kk].ubSize;
            addGmOffElem += ix * ubAxes[kk].gmStride;
            addUbOffBytes += ix * ubStride[kk];
        }
        const int64_t axis1Rows = (K >= 2) ? ubAxes[1].ubSize : 1;
        const int64_t maxRowsPerCopy = canBatchAxis1 ? BRC_BLOCKCNT_LIMIT : 1;
        for (int64_t rowStart = 0; rowStart < axis1Rows; rowStart += maxRowsPerCopy) {
            const int64_t rowsLeft = axis1Rows - rowStart;
            const int64_t rows = (rowsLeft < maxRowsPerCopy) ? rowsLeft : maxRowsPerCopy;
            extParams.blockCount = static_cast<uint16_t>(rows);
            const int64_t rowGmOff = (K >= 2) ? rowStart * ubAxes[1].gmStride : 0;
            const int64_t rowUbOffBytes = (K >= 2) ? rowStart * ubStride[1] : 0;
            DataCopyPad(yGm_[baseGmOff + addGmOffElem + rowGmOff], yLocal[(addUbOffBytes + rowUbOffBytes) / dtBytes],
                        extParams);
        }
    }
    if (useLoopMode) {
        ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
    }
}

// ════════════════════════════════════════════════════════════════════════════
// DoCopyOutTile（S12，§5.4）：B1(y) → GM。轴描述由 BuildUBAxes 产出后经
//   EmitTileCopyOut 发射（发射体与 group P3 共享）。
// ════════════════════════════════════════════════════════════════════════════
template <typename DType>
__aicore__ inline void L2NormalizeBaseKernel<DType>::DoCopyOutTile(int64_t baseGmOff, int64_t aLen, int64_t rLen,
                                                                   AscendC::LocalTensor<DType>& yLocal)
{
    UBAxisDesc ubAxes[MAX_PATTERN_RANK];
    const int32_t K = BuildUBAxes(aLen, rLen, ubAxes);
    EmitTileCopyOut(baseGmOff, ubAxes, K, yLocal);
}

// ════════════════════════════════════════════════════════════════════════════
// CopyOutDenomToWorkspace（S7a，§5.4）：B3 → ws denom 区本 aLoop 槽位（仅 R 切分
//   两遍路径）。逐 aLoop padded 槽位（槽宽 aUnit，slot = aLoopIdx × aUnit，§8）：
//   整 padded 行直写（含 inner-A pad garbage，不被 valid 消费）；UB 侧不足 32B 时
//   HW 自动补 dummy、GM 侧丢弃。
// ════════════════════════════════════════════════════════════════════════════
template <typename DType>
__aicore__ inline void L2NormalizeBaseKernel<DType>::CopyOutDenomToWorkspace(int64_t aLoopIdx)
{
    const int64_t aUnit = td_->aUbFactor * td_->innerAProdAlign;
    DataCopyExtParams ext{1, static_cast<uint32_t>(aUnit * static_cast<int64_t>(sizeof(float))), 0, 0, 0};
    DataCopyPad(wsGm_[aLoopIdx * aUnit], postReduceResult_.Get<float>(), ext);
}

// ════════════════════════════════════════════════════════════════════════════
// VF 调用侧
// ════════════════════════════════════════════════════════════════════════════
template <typename DType>
__aicore__ inline void L2NormalizeBaseKernel<DType>::CastSquareVf(__ubuf__ DType* src, __ubuf__ float* dst)
{
    const uint32_t totalElems = static_cast<uint32_t>(td_->aUbFactor * td_->innerAProdAlign * td_->rUbFactorAlign *
                                                      td_->innerRProdAlign);
    const uint16_t repeatTime = static_cast<uint16_t>(CeilDivU32(totalElems, static_cast<uint32_t>(REP_F32_U16)));
    asc_vf_call<CastSquareVfImpl<DType>>(src, dst, totalElems, repeatTime);
}

// ExtensionPad 清零（S3a，§9.3 决策表）：tail-R 按 A entry 逐行清 / tail-A 单段连续清
template <typename DType>
__aicore__ inline void L2NormalizeBaseKernel<DType>::ClearChunkExtensionVf(__ubuf__ float* base, int64_t rLen)
{
    if (rLen >= td_->rUbFactor) {
        return; // full chunk：无 ExtensionPad（BurstPad 另走 ClearInnerBurstTailPadVf）
    }

    if (isTailR_) {
        // extStart 跳过 BurstPad 区间：从最后一个含 valid 数据的 fp32 block 起点（32B 对齐）开始
        const uint32_t aBundleEntries = static_cast<uint32_t>(td_->aUbFactor * td_->innerAProdAlign);
        const uint32_t innerRPA = static_cast<uint32_t>(td_->innerRProdAlign);
        const uint32_t rLenInner = static_cast<uint32_t>(rLen) * innerRPA;
        const uint32_t extStart = CeilAlignU32(rLenInner, UB_BLOCK_F32);
        const uint32_t aStride = static_cast<uint32_t>(td_->rUbFactorAlign) * innerRPA; // 每 A entry 行宽
        if (extStart >= aStride) {
            return;
        }
        const uint32_t extLanes = aStride - extStart;
        const uint32_t repPerA = CeilDivU32(extLanes, REP_F32);
        const uint16_t aU16 = static_cast<uint16_t>(aBundleEntries);

        asc_vf_call<ClearChunkExtTailRVfImpl>(base, extStart, aStride, extLanes, aU16, static_cast<uint16_t>(repPerA));
    } else {
        const uint32_t cellElems = static_cast<uint32_t>(td_->aUbFactor * td_->innerAProdAlign * td_->innerRProdAlign);
        const uint32_t startElem = static_cast<uint32_t>(rLen) * cellElems;
        const uint32_t totalClear = (static_cast<uint32_t>(td_->rUbFactor) - static_cast<uint32_t>(rLen)) * cellElems;
        const uint32_t repCount = CeilDivU32(totalClear, REP_F32);

        asc_vf_call<ClearChunkExtTailAVfImpl>(base, startElem, totalClear, static_cast<uint16_t>(repCount));
    }
}

// BurstPad 清零（S3a，§9.3 触发条件表）：tail-R 且 burst 尾轴非对齐恒触发；
//   mask 精确覆盖 [validR, padEndInRow)，起点 FloorAlign 32B 对齐。
template <typename DType>
__aicore__ inline void L2NormalizeBaseKernel<DType>::ClearInnerBurstTailPadVf(__ubuf__ float* base, int64_t rLen)
{
    const uint32_t bsInput = UB_BLOCK_BYTES / static_cast<uint32_t>(sizeof(DType)); // BurstPad 按 D_T 计
    const int32_t lastR = LastRAxis();
    const uint32_t validR = (td_->rSplitIdx == lastR) ? static_cast<uint32_t>(rLen) :
                                                        static_cast<uint32_t>(td_->axisShape[lastR]);
    if (validR % bsInput == 0) {
        return; // burst 尾轴对齐：无 BurstPad
    }
    const uint32_t rowStride = (td_->rSplitIdx == lastR) ?
                                   static_cast<uint32_t>(td_->rUbFactorAlign * td_->innerRProdAlign) :
                                   CeilAlignU32(static_cast<uint32_t>(td_->axisShape[lastR]), bsInput);
    // rSplit!=LastR：按全量行清（要清的行在内存里不连续，多清的 stale 行本就是脏数据）
    const uint32_t rowCnt = (td_->rSplitIdx == lastR) ?
                                static_cast<uint32_t>(td_->aUbFactor * td_->innerAProdAlign) :
                                static_cast<uint32_t>(td_->aUbFactor * td_->innerAProdAlign * td_->rUbFactorAlign *
                                                      td_->innerRProdAlign) /
                                    rowStride;

    const uint32_t padEndInRow = CeilAlignU32(validR, bsInput);
    const uint32_t partialBlockIdx = validR / UB_BLOCK_F32; // StoreAlign 起点 block 对齐（FloorAlign）
    const uint32_t partialStartInBlock = validR % UB_BLOCK_F32;
    const uint32_t padEnd = padEndInRow - partialBlockIdx * UB_BLOCK_F32;

    asc_vf_call<ClearInnerBurstTailPadVfImpl>(base, static_cast<uint16_t>(rowCnt), static_cast<int32_t>(rowStride),
                                              static_cast<int32_t>(partialBlockIdx * UB_BLOCK_F32), padEnd,
                                              partialStartInBlock);
}

template <typename DType>
__aicore__ inline void L2NormalizeBaseKernel<DType>::MergeTmpBufVf(__ubuf__ float* mainBuf, __ubuf__ float* tailBuf)
{
    const uint32_t totalElems = static_cast<uint32_t>(td_->aUbFactor * td_->innerAProdAlign * td_->rUbFactorAlign *
                                                      td_->innerRProdAlign);
    const uint16_t repeatTime = static_cast<uint16_t>(CeilDivU32(totalElems, static_cast<uint32_t>(REP_F32_U16)));
    asc_vf_call<MergeTmpBufVfImpl>(mainBuf, tailBuf, totalElems, repeatTime);
}

template <typename DType>
__aicore__ inline void L2NormalizeBaseKernel<DType>::DoCachingVf(uint16_t cacheID)
{
    const uint32_t laneN = static_cast<uint32_t>(td_->aUbFactor * td_->innerAProdAlign);
    const uint32_t levelStride = CeilAlignU32(laneN, UB_BLOCK_F32);
    const int32_t levelOff = static_cast<int32_t>(cacheID) * static_cast<int32_t>(levelStride);
    const uint16_t repeatTime = static_cast<uint16_t>(CeilDivU32(laneN, static_cast<uint32_t>(REP_F32_U16)));

    __ubuf__ float* cachePtr = reinterpret_cast<__ubuf__ float*>(cacheBuf_.Get<float>().GetPhyAddr());
    asc_vf_call<DoCachingVfImpl>(cachePtr, laneN, levelStride, levelOff, repeatTime, cacheID);
}

// S9 R 全载 tail-A：denom 驻 B3 → B2 行复制
template <typename DType>
__aicore__ inline void L2NormalizeBaseKernel<DType>::BroadcastDenomTailAVf()
{
    const uint32_t laneA = static_cast<uint32_t>(td_->aUbFactor * td_->innerAProdAlign);
    const uint32_t rBundle = static_cast<uint32_t>(td_->rUbFactorAlign * td_->innerRProdAlign);
    const uint16_t repTime = static_cast<uint16_t>(CeilDivU32(laneA, static_cast<uint32_t>(REP_F32_U16)));
    __ubuf__ float* denomPtr = reinterpret_cast<__ubuf__ float*>(postReduceResult_.Get<float>().GetPhyAddr());
    __ubuf__ float* bcastPtr = reinterpret_cast<__ubuf__ float*>(preReduceResultTail_.Get<float>().GetPhyAddr());
    asc_vf_call<BroadcastDenomTailAVfImpl>(denomPtr, bcastPtr, laneA, rBundle, repTime);
}

// S9 tail-R：B3 → B2 行常量（R 全载 B3 驻留 / R 切分 dense 搬入 B3 后同款）
template <typename DType>
__aicore__ inline void L2NormalizeBaseKernel<DType>::BroadcastDenomTailRVf(__ubuf__ float* denomPtr)
{
    const uint32_t laneA = static_cast<uint32_t>(td_->aUbFactor * td_->innerAProdAlign);
    const uint32_t rBundle = static_cast<uint32_t>(td_->rUbFactorAlign * td_->innerRProdAlign);
    const uint16_t repPerRow = static_cast<uint16_t>(CeilDivU32(rBundle, static_cast<uint32_t>(REP_F32_U16)));
    __ubuf__ float* bcastPtr = reinterpret_cast<__ubuf__ float*>(preReduceResultTail_.Get<float>().GetPhyAddr());
    asc_vf_call<BroadcastDenomTailRVfImpl>(denomPtr, bcastPtr, laneA, rBundle, repPerRow);
}

template <typename DType>
__aicore__ inline void L2NormalizeBaseKernel<DType>::DivCastVf(__ubuf__ DType* xPost, __ubuf__ DType* y)
{
    constexpr bool kFp16Prec = !AscendC::IsSameType<DType, float>::value;
    const uint32_t totalElems = static_cast<uint32_t>(td_->aUbFactor * td_->innerAProdAlign * td_->rUbFactorAlign *
                                                      td_->innerRProdAlign);
    const uint16_t repeatTime = static_cast<uint16_t>(CeilDivU32(totalElems, static_cast<uint32_t>(REP_F32_U16)));
    __ubuf__ float* bcastPtr = reinterpret_cast<__ubuf__ float*>(preReduceResultTail_.Get<float>().GetPhyAddr());
    asc_vf_call<DivCastVfImpl<DType, kFp16Prec>>(xPost, bcastPtr, y, totalElems, repeatTime);
}

} // namespace NsL2Normalize

#endif // OPS_NORM_L2_NORMALIZE_BASE_H_
