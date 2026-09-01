/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file bn3d_training_reduce_grad_kernel.h
 * \brief BN3DTrainingReduceGrad 的 Ascend C kernel 计算实现
 */

#pragma once
#include "kernel_operator.h"                       // Ascend C core framework
#include "bn3d_training_reduce_grad_tiling_data.h" // BN3DTrainingReduceGradTilingData<RANK> / 公共常量
#include "bn3_d_training_reduce_grad_struct.h" // ASCENDC_TPL_ARGS_DECL（必须 include，否则构建系统走 #define 包装器路径）

// ---------------------------------------------------------------------------
// 输入 slot 编号
// ---------------------------------------------------------------------------
constexpr int IN_GRADS = 0, IN_X = 1, IN_DIFF_SCALE = 2, IN_DIFF_OFFSET = 3, IN_SCALE = 4, IN_BATCH_MEAN = 5,
              IN_BATCH_VARIANCE = 6;
// 派生常量: 7 输入 dtype 异构 → 拆两组（前 2 个 dtype=T, 后 5 个恒 f32）
constexpr int64_t NUM_DATA_INPUTS = 2;  // grads / x（dtype = DTYPE_GRADS）
constexpr int64_t NUM_PARAM_INPUTS = 5; // diff_scale / diff_offset / scale / batch_mean / batch_variance（f32）

// 升位 b16 → f32: 扩位唯一组合（sat=UNKNOWN, round=CAST_NONE）
static constexpr AscendC::Reg::CastTrait kCastTraitB16ToF32 = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_NONE};
// 缩位 f32 → b16（f16 / bf16 统一）: sat=NO_SAT（不可 UNKNOWN）, round=CAST_RINT
// （不可 CAST_NONE）——CAST_RINT = round-half-even：f16 cast 即 IEEE 默认 RN、
// bf16 round 对齐旧仓动态参考实现；NO_SAT 使回落溢出按 IEEE 自然得 ±Inf
static constexpr AscendC::Reg::CastTrait kCastTraitF32ToB16 = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
    AscendC::RoundMode::CAST_RINT};

// ============================================================
// Kernel 侧泛型辅助函数（int64_t* 版本, 无 std::vector; 抄自 adam 参考源码）
// ============================================================
// 核间 tile 区间: 前 coresTail 个核各处理 tilesMain+1 个 tile, 其余核各 tilesMain 个;
// 区间并集 = [0, totalTiles) 无重叠无遗漏, 与 多核切分（MultiCoreSplit）字段契约一致
__aicore__ inline void GetCoreRange(int64_t coreId, int64_t tilesMain, int64_t coresTail, int64_t& start, int64_t& end)
{
    if (coreId < coresTail) {
        start = coreId * (tilesMain + 1);
        end = start + tilesMain + 1;
    } else {
        start = coresTail * (tilesMain + 1) + (coreId - coresTail) * tilesMain;
        end = start + tilesMain;
    }
}

// split 轴本 tile 段长: 末段 aITail, 其余 aI（覆盖主块+尾块）
__aicore__ inline int64_t GetUBSplitRange(int64_t aOOff, int64_t aO, int64_t aI, int64_t aITail)
{
    return (aOOff == aO - 1) ? aITail : aI;
}

// flat tile 索引 → effective 坐标: split 轴坐标 = aOOff × aI; 轴 0..splitAxis-1 由 outer 解出;
// 轴 splitAxis+1..rank-1 为 UB 内轴（全量进 UB）, 坐标恒 0
__aicore__ inline bool FlatToEffectiveCoord(int64_t flat, const int64_t* maxBroShape, int64_t rank, int64_t splitAxis,
                                            int64_t aI, int64_t aO, int64_t* effCoord)
{
    for (int64_t d = 0; d < rank; d++)
        effCoord[d] = 0;
    int64_t aOOff = flat % aO;
    int64_t outer = flat / aO;
    for (int64_t d = splitAxis - 1; d >= 0; d--) {
        effCoord[d] = outer % maxBroShape[d];
        outer /= maxBroShape[d];
    }
    effCoord[splitAxis] = aOOff * aI;
    return true;
}

// 坐标 → 输入 GM 偏移（元素个数; broadcast 轴 stride=0 实现随路广播;
// 必须返回元素数而非字节——gmIn_[] 的 operator[] 取元素索引）
__aicore__ inline int64_t CalcInputOffset(const int64_t* effCoord, const int64_t* strides, int64_t rank)
{
    int64_t offset = 0;
    for (int64_t d = 0; d < rank; d++)
        offset += effCoord[d] * strides[d];
    return offset; // 元素个数, gmIn_[]/gmParam_[] 的 index
}

// 坐标 → 输出 GM 偏移（元素个数）
__aicore__ inline int64_t CalcOutputOffset(const int64_t* effCoord, const int64_t* strides, int64_t rank)
{
    int64_t offset = 0;
    for (int64_t d = 0; d < rank; d++)
        offset += effCoord[d] * strides[d];
    return offset; // 元素个数, gmOut_[] 的 index
}

// 本 tile 输出元素数（CopyOut 的 DataCopyPad blockLen 依据）
__aicore__ inline int64_t CalcOutputTransferCount(const int64_t* normalShape, int64_t rank, int64_t splitAxis,
                                                  int64_t aISeg)
{
    int64_t splitElems = (normalShape[splitAxis] == 1) ? 1 : aISeg;
    int64_t innerElems = 1;
    for (int64_t d = splitAxis + 1; d < rank; d++)
        innerElems *= normalShape[d];
    return splitElems * innerElems;
}

// ===========================================================================
// 5 条计算 VF 链（全以 <float> 实例化; 模板形参 T 仅为匹配 asc_vf_call<ChainXxxVF<float>> 语法）
// ===========================================================================
// VL = 256 / sizeof(T) = 64（f32, 每寄存器 256 字节）;
// count ≤ perBufElems = 16384 → uint32 无溢出;
// LoadAlign/StoreAlign 要求 32B 对齐: TBuf 槽位 perBufBytes=(UB/P)&~31 对齐,
// off = i×64×4 恒 32B 倍数。
// 链长核对: ChainS=2 / ChainA1=3 / ChainA2=2 / ChainB=2 / ChainC=2, 全部 ≤ 7。

// ChainSVF: s = sqrt(bv + eps)  [Adds + Sqrt, 链长 2; dst==src 原地, unary 合法, epsilon_guard]
template <typename T>
__simd_vf__ inline void ChainSVF(__ubuf__ T* dst, __ubuf__ T* src, T eps, int64_t count, uint16_t repeat)
{
    AscendC::Reg::RegTensor<T> rDst;
    AscendC::Reg::MaskReg mask;
    constexpr uint32_t VL = 256 / sizeof(T); // = 64
    uint32_t remaining = static_cast<uint32_t>(count);
    for (uint16_t i = 0; i < repeat; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(VL);
        mask = AscendC::Reg::UpdateMask<T>(remaining); // remaining 自动递减, 不可再手动递减
        AscendC::Reg::LoadAlign(rDst, src + off);
        AscendC::Reg::Adds(rDst, rDst, eps, mask); // bv + eps（ε 经参数传入, 禁止硬编码）
        AscendC::Reg::Sqrt(rDst, rDst, mask);      // sqrt(...)
        AscendC::Reg::StoreAlign(dst + off, rDst, mask);
    }
}

// ChainA1VF: t_a = (x - bm) * ds * inv_num  [Sub + Mul + Muls, 链长 3]
template <typename T>
__simd_vf__ inline void ChainA1VF(__ubuf__ T* dst, __ubuf__ T* x, __ubuf__ T* bm, __ubuf__ T* ds, T invNum,
                                  int64_t count, uint16_t repeat)
{
    AscendC::Reg::RegTensor<T> rX, rBm, rDs, rDst;
    AscendC::Reg::MaskReg mask;
    constexpr uint32_t VL = 256 / sizeof(T);
    uint32_t remaining = static_cast<uint32_t>(count);
    for (uint16_t i = 0; i < repeat; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(VL);
        mask = AscendC::Reg::UpdateMask<T>(remaining);
        AscendC::Reg::LoadAlign(rX, x + off);
        AscendC::Reg::LoadAlign(rBm, bm + off);
        AscendC::Reg::LoadAlign(rDs, ds + off);
        AscendC::Reg::Sub(rDst, rX, rBm, mask);       // x - bm
        AscendC::Reg::Mul(rDst, rDst, rDs, mask);     // * ds
        AscendC::Reg::Muls(rDst, rDst, invNum, mask); // * inv_num
        AscendC::Reg::StoreAlign(dst + off, rDst, mask);
    }
}

// ChainA2VF: t1 = grads - t_a / s  [Div + Sub, 链长 2; ÷s 在括号内、先于 grads−, 顺序 2]
template <typename T>
__simd_vf__ inline void ChainA2VF(__ubuf__ T* dst, __ubuf__ T* grads, __ubuf__ T* tA, __ubuf__ T* s, int64_t count,
                                  uint16_t repeat)
{
    AscendC::Reg::RegTensor<T> rGrads, rTa, rS, rTmp, rDst;
    AscendC::Reg::MaskReg mask;
    constexpr uint32_t VL = 256 / sizeof(T);
    uint32_t remaining = static_cast<uint32_t>(count);
    for (uint16_t i = 0; i < repeat; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(VL);
        mask = AscendC::Reg::UpdateMask<T>(remaining);
        AscendC::Reg::LoadAlign(rGrads, grads + off);
        AscendC::Reg::LoadAlign(rTa, tA + off);
        AscendC::Reg::LoadAlign(rS, s + off);
        AscendC::Reg::Div(rTmp, rTa, rS, mask);      // t_a / s
        AscendC::Reg::Sub(rDst, rGrads, rTmp, mask); // grads - t_a/s
        AscendC::Reg::StoreAlign(dst + off, rDst, mask);
    }
}

// ChainBVF: t2 = t1 - do * inv_num  [Muls + Sub, 链长 2]
template <typename T>
__simd_vf__ inline void ChainBVF(__ubuf__ T* dst, __ubuf__ T* t1, __ubuf__ T* do_, T invNum, int64_t count,
                                 uint16_t repeat)
{
    AscendC::Reg::RegTensor<T> rT1, rDo, rTmp, rDst;
    AscendC::Reg::MaskReg mask;
    constexpr uint32_t VL = 256 / sizeof(T);
    uint32_t remaining = static_cast<uint32_t>(count);
    for (uint16_t i = 0; i < repeat; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(VL);
        mask = AscendC::Reg::UpdateMask<T>(remaining);
        AscendC::Reg::LoadAlign(rT1, t1 + off);
        AscendC::Reg::LoadAlign(rDo, do_ + off);
        AscendC::Reg::Muls(rTmp, rDo, invNum, mask); // do * inv_num
        AscendC::Reg::Sub(rDst, rT1, rTmp, mask);    // t1 - do*inv_num
        AscendC::Reg::StoreAlign(dst + off, rDst, mask);
    }
}

// ChainCVF: y = t2 * sc / s  [Mul + Div, 链长 2]
template <typename T>
__simd_vf__ inline void ChainCVF(__ubuf__ T* dst, __ubuf__ T* t2, __ubuf__ T* sc, __ubuf__ T* s, int64_t count,
                                 uint16_t repeat)
{
    AscendC::Reg::RegTensor<T> rT2, rSc, rS, rTmp, rDst;
    AscendC::Reg::MaskReg mask;
    constexpr uint32_t VL = 256 / sizeof(T);
    uint32_t remaining = static_cast<uint32_t>(count);
    for (uint16_t i = 0; i < repeat; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(VL);
        mask = AscendC::Reg::UpdateMask<T>(remaining);
        AscendC::Reg::LoadAlign(rT2, t2 + off);
        AscendC::Reg::LoadAlign(rSc, sc + off);
        AscendC::Reg::LoadAlign(rS, s + off);
        AscendC::Reg::Mul(rTmp, rT2, rSc, mask); // t2 * sc
        AscendC::Reg::Div(rDst, rTmp, rS, mask); // (t2*sc)/s
        AscendC::Reg::StoreAlign(dst + off, rDst, mask);
    }
}

// ===========================================================================
// NaN/Inf 传播对齐 flags 与分类修复链（元素级，全 <float>/<int32_t> 实例化）
//
// 背景：全输入 (-inf, inf) 极端用例
//   下参数幅值可达 ±3.4e38，t = ds·(x−bm) 的真值（f64 ~1e71）超出 f32 表示
//   范围，kernel 中间量溢出 ±Inf，与 f64 golden 的「有限巨大值」分类不一致，
//   经 IEEE 传播后产生 golden 无的 NaN / 符号不符的 ±Inf。数学上任意运算
//   结合序都无法在 f32 域表示 |y|~1e70 的量级，故按 DESIGN 「特殊值
//   逐点对齐」语义对 f32 中间量**仅做分类修复**（不 clamp、不改有限值）：
//
//   condA: s=+Inf ∧ t 有限 ∧ g/do/sc 有限 → golden y=±0（kernel 得 NaN）
//   condB: s 有限 ∧ g=±Inf ∧ t 有限 ∧ sign(g)==sign(t) ∧ ¬(do=±Inf ∧
//          sign(do)==sign(g)) → golden y=sign(g)·sign(sc)·Inf（kernel 得 NaN）
//   condC: s 有限 ∧ g 有限 ∧ t 有限 ∧ do=±Inf ∧ sign(t1)==sign(do)
//          → golden y=−sign(do)·sign(sc)·Inf（kernel 得 NaN）
//
// 其余情形 kernel 的 f32 链与 golden 分类一致（±Inf 符号由溢出保号/真 Inf
//   自然传播），修复条件对常规数据恒不成立（幂等）。flags 为 int32 位掩码
//   （元素级，B1 槽常驻，逐链累积）：
//   bit0 tFin  bit1 gtSame(sign(g)==sign(t_a))  bit2 gFin  bit3 gInf
//   bit4 gSign bit5 sFin  bit6 sInf  bit7 doFin bit8 doInf
//   bit9 doSign bit10 t1SameDo  bit11 scFin  bit12 scSign
// ===========================================================================
constexpr int32_t FLAG_TFIN = 1 << 0;   // t = ds·(x−bm) 有限（xbmHalf/ds 均有限）
constexpr int32_t FLAG_GTSAME = 1 << 1; // sign(g) == sign(t_a)
constexpr int32_t FLAG_GFIN = 1 << 2;
constexpr int32_t FLAG_GINF = 1 << 3;
constexpr int32_t FLAG_GSIGN = 1 << 4;
constexpr int32_t FLAG_SFIN = 1 << 5;
constexpr int32_t FLAG_SINF = 1 << 6;
constexpr int32_t FLAG_DOFIN = 1 << 7;
constexpr int32_t FLAG_DOINF = 1 << 8;
constexpr int32_t FLAG_DOSIGN = 1 << 9;
constexpr int32_t FLAG_T1SAMEDO = 1 << 10; // sign(t1) == sign(do)
constexpr int32_t FLAG_SCFIN = 1 << 11;
constexpr int32_t FLAG_SCSIGN = 1 << 12;
constexpr int32_t FLAG_BITS_CONDA = FLAG_SINF | FLAG_TFIN | FLAG_GFIN | FLAG_DOFIN | FLAG_SCFIN; // 2245
constexpr int32_t FLAG_BITS_CONDB = FLAG_SFIN | FLAG_GINF | FLAG_TFIN;                           // 41
constexpr int32_t FLAG_BITS_CONDC = FLAG_SFIN | FLAG_GFIN | FLAG_TFIN | FLAG_DOINF;              // 293
constexpr int32_t FLAG_BITS_GSIGN_DOSIGN = FLAG_GSIGN | FLAG_DOSIGN;                             // 528
constexpr int32_t FLAG_BITS_GSIGN_SCSIGN = FLAG_GSIGN | FLAG_SCSIGN;                             // 4112
constexpr int32_t FLAG_BITS_DOSIGN_SCSIGN = FLAG_DOSIGN | FLAG_SCSIGN;                           // 4608

// suspect 判定阈值（Stage-1 条件化修复链）: 修复链触发条件（condA/B/C）均要求
// g/do/s 存在字面 ±Inf；tile 级 max(|g|,|do|,s) > 3e38（f32 正常最大 ~3.4e38，
// 仅 ±Inf 与 3e38 量级极端值可超过）→ 走原完整路径，否则快速路径。
// NaN 使比较为假 → !(mx<=thr) 为真 → 保守路由完整路径（行为与原实现一致）。
static constexpr float kSuspectThreshold = 3.0e38f;

// FlagVF1: flags[bit0 tFin] = isfinite(x·0.5−bm·0.5) && isfinite(ds)

template <typename T>
__simd_vf__ inline void FlagVF1(__ubuf__ int32_t* dstFlags, __ubuf__ T* x, __ubuf__ T* bm, __ubuf__ T* ds,
                                int64_t count, uint16_t repeat)
{
    AscendC::Reg::RegTensor<T> rX, rBm, rDs, rTmp;
    AscendC::Reg::RegTensor<int32_t> rFlags, rC;
    AscendC::Reg::MaskReg mask, mFin, mTmp;
    constexpr uint32_t VL = 256 / sizeof(T);
    uint32_t remaining = static_cast<uint32_t>(count);
    for (uint16_t i = 0; i < repeat; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(VL);
        mask = AscendC::Reg::UpdateMask<T>(remaining);
        AscendC::Reg::LoadAlign(rX, x + off);
        AscendC::Reg::LoadAlign(rBm, bm + off);
        AscendC::Reg::LoadAlign(rDs, ds + off);
        AscendC::Reg::Muls(rX, rX, T(0.5), mask);   // x·0.5（2 的幂缩放精确）
        AscendC::Reg::Muls(rBm, rBm, T(0.5), mask); // bm·0.5
        AscendC::Reg::Sub(rTmp, rX, rBm, mask);     // xbmHalf
        AscendC::Reg::Sub(rX, rTmp, rTmp, mask);    // xbmHalf−xbmHalf（有限 ⟺ 0）
        AscendC::Reg::Compares<T, AscendC::CMPMODE::EQ>(mFin, rX, T(0), mask);
        AscendC::Reg::Sub(rBm, rDs, rDs, mask); // ds−ds
        AscendC::Reg::Compares<T, AscendC::CMPMODE::EQ>(mTmp, rBm, T(0), mask);
        AscendC::Reg::And(mFin, mFin, mTmp, mask);
        AscendC::Reg::Duplicate<int32_t>(rFlags, 0);
        AscendC::Reg::Duplicate<int32_t>(rC, FLAG_TFIN);
        AscendC::Reg::Select(rFlags, rC, rFlags, mFin); // mFin ? 1 : 0
        AscendC::Reg::StoreAlign(dstFlags + off, rFlags, mask);
    }
}

// FlagVF2: flags |= gFin<<2 | gInf<<3 | gSign<<4 | sFin<<5 | sInf<<6 | gtSame<<1
//   gtSame = (g·t_a > 0)（t_a 为 ChainA1 产物，含溢出 ±Inf，符号保真）
template <typename T>
__simd_vf__ inline void FlagVF2(__ubuf__ int32_t* dstFlags, __ubuf__ int32_t* srcFlags, __ubuf__ T* g, __ubuf__ T* s,
                                __ubuf__ T* tA, int64_t count, uint16_t repeat)
{
    AscendC::Reg::RegTensor<T> rG, rS, rTA, rTmp;
    AscendC::Reg::RegTensor<int32_t> rFlags, rC, rAcc, rZero;
    AscendC::Reg::MaskReg mask, m1, m2;
    constexpr uint32_t VL = 256 / sizeof(T);
    uint32_t remaining = static_cast<uint32_t>(count);
    for (uint16_t i = 0; i < repeat; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(VL);
        mask = AscendC::Reg::UpdateMask<T>(remaining);
        AscendC::Reg::LoadAlign(rFlags, srcFlags + off);
        AscendC::Reg::LoadAlign(rG, g + off);
        AscendC::Reg::LoadAlign(rS, s + off);
        AscendC::Reg::LoadAlign(rTA, tA + off);
        AscendC::Reg::Duplicate<int32_t>(rZero, 0);
        // bit2 gFin
        AscendC::Reg::Sub(rTmp, rG, rG, mask);
        AscendC::Reg::Compares<T, AscendC::CMPMODE::EQ>(m1, rTmp, T(0), mask);
        AscendC::Reg::Duplicate<int32_t>(rC, FLAG_GFIN);
        AscendC::Reg::Select(rAcc, rC, rZero, m1);
        AscendC::Reg::Add(rFlags, rFlags, rAcc, mask);
        // bit3 gInf
        AscendC::Reg::Compares<T, AscendC::CMPMODE::EQ>(m1, rG, __builtin_inff(), mask);
        AscendC::Reg::Compares<T, AscendC::CMPMODE::EQ>(m2, rG, -__builtin_inff(), mask);
        AscendC::Reg::Or(m1, m1, m2, mask);
        AscendC::Reg::Duplicate<int32_t>(rC, FLAG_GINF);
        AscendC::Reg::Select(rAcc, rC, rZero, m1);
        AscendC::Reg::Add(rFlags, rFlags, rAcc, mask);
        // bit4 gSign
        AscendC::Reg::Compares<T, AscendC::CMPMODE::GT>(m1, rG, T(0), mask);
        AscendC::Reg::Duplicate<int32_t>(rC, FLAG_GSIGN);
        AscendC::Reg::Select(rAcc, rC, rZero, m1);
        AscendC::Reg::Add(rFlags, rFlags, rAcc, mask);
        // bit5 sFin
        AscendC::Reg::Sub(rTmp, rS, rS, mask);
        AscendC::Reg::Compares<T, AscendC::CMPMODE::EQ>(m1, rTmp, T(0), mask);
        AscendC::Reg::Duplicate<int32_t>(rC, FLAG_SFIN);
        AscendC::Reg::Select(rAcc, rC, rZero, m1);
        AscendC::Reg::Add(rFlags, rFlags, rAcc, mask);
        // bit6 sInf
        AscendC::Reg::Compares<T, AscendC::CMPMODE::EQ>(m1, rS, __builtin_inff(), mask);
        AscendC::Reg::Duplicate<int32_t>(rC, FLAG_SINF);
        AscendC::Reg::Select(rAcc, rC, rZero, m1);
        AscendC::Reg::Add(rFlags, rFlags, rAcc, mask);
        // bit1 gtSame = (g·t_a > 0)
        AscendC::Reg::Mul(rTmp, rG, rTA, mask);
        AscendC::Reg::Compares<T, AscendC::CMPMODE::GT>(m1, rTmp, T(0), mask);
        AscendC::Reg::Duplicate<int32_t>(rC, FLAG_GTSAME);
        AscendC::Reg::Select(rAcc, rC, rZero, m1);
        AscendC::Reg::Add(rFlags, rFlags, rAcc, mask);
        AscendC::Reg::StoreAlign(dstFlags + off, rFlags, mask);
    }
}

// FlagVF3: flags |= doFin<<7 | doInf<<8 | doSign<<9 | t1SameDo<<10
//   t1SameDo = (t1·do > 0)（t1 为 ChainA2 产物，含溢出 ±Inf，符号保真）
template <typename T>
__simd_vf__ inline void FlagVF3(__ubuf__ int32_t* dstFlags, __ubuf__ int32_t* srcFlags, __ubuf__ T* do_, __ubuf__ T* t1,
                                int64_t count, uint16_t repeat)
{
    AscendC::Reg::RegTensor<T> rDo, rT1, rTmp;
    AscendC::Reg::RegTensor<int32_t> rFlags, rC, rAcc, rZero;
    AscendC::Reg::MaskReg mask, m1, m2;
    constexpr uint32_t VL = 256 / sizeof(T);
    uint32_t remaining = static_cast<uint32_t>(count);
    for (uint16_t i = 0; i < repeat; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(VL);
        mask = AscendC::Reg::UpdateMask<T>(remaining);
        AscendC::Reg::LoadAlign(rFlags, srcFlags + off);
        AscendC::Reg::LoadAlign(rDo, do_ + off);
        AscendC::Reg::LoadAlign(rT1, t1 + off);
        AscendC::Reg::Duplicate<int32_t>(rZero, 0);
        // bit7 doFin
        AscendC::Reg::Sub(rTmp, rDo, rDo, mask);
        AscendC::Reg::Compares<T, AscendC::CMPMODE::EQ>(m1, rTmp, T(0), mask);
        AscendC::Reg::Duplicate<int32_t>(rC, FLAG_DOFIN);
        AscendC::Reg::Select(rAcc, rC, rZero, m1);
        AscendC::Reg::Add(rFlags, rFlags, rAcc, mask);
        // bit8 doInf
        AscendC::Reg::Compares<T, AscendC::CMPMODE::EQ>(m1, rDo, __builtin_inff(), mask);
        AscendC::Reg::Compares<T, AscendC::CMPMODE::EQ>(m2, rDo, -__builtin_inff(), mask);
        AscendC::Reg::Or(m1, m1, m2, mask);
        AscendC::Reg::Duplicate<int32_t>(rC, FLAG_DOINF);
        AscendC::Reg::Select(rAcc, rC, rZero, m1);
        AscendC::Reg::Add(rFlags, rFlags, rAcc, mask);
        // bit9 doSign
        AscendC::Reg::Compares<T, AscendC::CMPMODE::GT>(m1, rDo, T(0), mask);
        AscendC::Reg::Duplicate<int32_t>(rC, FLAG_DOSIGN);
        AscendC::Reg::Select(rAcc, rC, rZero, m1);
        AscendC::Reg::Add(rFlags, rFlags, rAcc, mask);
        // bit10 t1SameDo = (t1·do > 0)
        AscendC::Reg::Mul(rTmp, rT1, rDo, mask);
        AscendC::Reg::Compares<T, AscendC::CMPMODE::GT>(m1, rTmp, T(0), mask);
        AscendC::Reg::Duplicate<int32_t>(rC, FLAG_T1SAMEDO);
        AscendC::Reg::Select(rAcc, rC, rZero, m1);
        AscendC::Reg::Add(rFlags, rFlags, rAcc, mask);
        AscendC::Reg::StoreAlign(dstFlags + off, rFlags, mask);
    }
}

// FlagVF4: flags |= scFin<<11 | scSign<<12
template <typename T>
__simd_vf__ inline void FlagVF4(__ubuf__ int32_t* dstFlags, __ubuf__ int32_t* srcFlags, __ubuf__ T* sc, int64_t count,
                                uint16_t repeat)
{
    AscendC::Reg::RegTensor<T> rSc, rTmp;
    AscendC::Reg::RegTensor<int32_t> rFlags, rC, rAcc, rZero;
    AscendC::Reg::MaskReg mask, m1;
    constexpr uint32_t VL = 256 / sizeof(T);
    uint32_t remaining = static_cast<uint32_t>(count);
    for (uint16_t i = 0; i < repeat; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(VL);
        mask = AscendC::Reg::UpdateMask<T>(remaining);
        AscendC::Reg::LoadAlign(rFlags, srcFlags + off);
        AscendC::Reg::LoadAlign(rSc, sc + off);
        AscendC::Reg::Duplicate<int32_t>(rZero, 0);
        // bit11 scFin
        AscendC::Reg::Sub(rTmp, rSc, rSc, mask);
        AscendC::Reg::Compares<T, AscendC::CMPMODE::EQ>(m1, rTmp, T(0), mask);
        AscendC::Reg::Duplicate<int32_t>(rC, FLAG_SCFIN);
        AscendC::Reg::Select(rAcc, rC, rZero, m1);
        AscendC::Reg::Add(rFlags, rFlags, rAcc, mask);
        // bit12 scSign
        AscendC::Reg::Compares<T, AscendC::CMPMODE::GT>(m1, rSc, T(0), mask);
        AscendC::Reg::Duplicate<int32_t>(rC, FLAG_SCSIGN);
        AscendC::Reg::Select(rAcc, rC, rZero, m1);
        AscendC::Reg::Add(rFlags, rFlags, rAcc, mask);
        AscendC::Reg::StoreAlign(dstFlags + off, rFlags, mask);
    }
}

// FixVF: 依据 flags 对 y 做分类修复（Select 语义: mask 置位取 src0）
//   condA: (flags & 2245)==2245            → y = ±0（golden y=±0；0 的符号不影响比对）
//   condB: (flags & 41)==41 ∧ bit1 ∧ ¬(bit8 ∧ eq(bit4,bit9)) → y = eq(bit4,bit12) ? +Inf : −Inf
//   condC: (flags & 293)==293 ∧ bit10      → y = eq(bit9,bit12) ? −Inf : +Inf
//   其中 eq(a,b) = (flags & (a|b)) ∈ {0, a|b}
template <typename T>
__simd_vf__ inline void FixVF(__ubuf__ T* dstY, __ubuf__ T* srcY, __ubuf__ int32_t* srcFlags, int64_t count,
                              uint16_t repeat)
{
    AscendC::Reg::RegTensor<T> rY, rPinf, rNinf, rZeroF;
    AscendC::Reg::RegTensor<int32_t> rF, rTmp, rC;
    AscendC::Reg::MaskReg mask, m1, m2, m3, m4;
    constexpr uint32_t VL = 256 / sizeof(T);
    uint32_t remaining = static_cast<uint32_t>(count);
    for (uint16_t i = 0; i < repeat; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(VL);
        mask = AscendC::Reg::UpdateMask<T>(remaining);
        AscendC::Reg::LoadAlign(rY, srcY + off);
        AscendC::Reg::LoadAlign(rF, srcFlags + off);
        AscendC::Reg::Duplicate<T>(rPinf, __builtin_inff());
        AscendC::Reg::Duplicate<T>(rNinf, -__builtin_inff());
        AscendC::Reg::Duplicate<T>(rZeroF, T(0));
        // ---- condA: (flags & 2245)==2245 → y = +0 ----
        AscendC::Reg::Duplicate<int32_t>(rC, FLAG_BITS_CONDA);
        AscendC::Reg::And(rTmp, rF, rC, mask);
        AscendC::Reg::Compares<int32_t, AscendC::CMPMODE::EQ>(m1, rTmp, FLAG_BITS_CONDA, mask);
        AscendC::Reg::Select(rY, rZeroF, rY, m1);
        // ---- condB: baseB(41) ∧ gtSame(2) ∧ ¬(doInf(256) ∧ eq(gSign,doSign)(528)) ----
        AscendC::Reg::Duplicate<int32_t>(rC, FLAG_BITS_GSIGN_DOSIGN);
        AscendC::Reg::And(rTmp, rF, rC, mask);
        AscendC::Reg::Compares<int32_t, AscendC::CMPMODE::EQ>(m4, rTmp, FLAG_BITS_GSIGN_DOSIGN, mask);
        AscendC::Reg::Compares<int32_t, AscendC::CMPMODE::EQ>(m1, rTmp, 0, mask);
        AscendC::Reg::Or(m4, m4, m1, mask); // eq(gSign, doSign)
        AscendC::Reg::Duplicate<int32_t>(rC, FLAG_DOINF);
        AscendC::Reg::And(rTmp, rF, rC, mask);
        AscendC::Reg::Compares<int32_t, AscendC::CMPMODE::NE>(m3, rTmp, 0, mask); // doInf
        AscendC::Reg::And(m3, m3, m4, mask);                                      // doInf ∧ eq(gSign, doSign)
        AscendC::Reg::Not(m4, m3, mask);                                          // m4 = ¬(doInf ∧ eq)
        AscendC::Reg::Duplicate<int32_t>(rC, FLAG_BITS_CONDB);
        AscendC::Reg::And(rTmp, rF, rC, mask);
        AscendC::Reg::Compares<int32_t, AscendC::CMPMODE::EQ>(m1, rTmp, FLAG_BITS_CONDB, mask);
        AscendC::Reg::Duplicate<int32_t>(rC, FLAG_GTSAME);
        AscendC::Reg::And(rTmp, rF, rC, mask);
        AscendC::Reg::Compares<int32_t, AscendC::CMPMODE::NE>(m2, rTmp, 0, mask);
        AscendC::Reg::And(m1, m1, m2, mask); // baseB ∧ gtSame
        AscendC::Reg::And(m1, m1, m4, mask); // condB
        // ---- condC: baseC(293) ∧ t1SameDo(1024) ----
        AscendC::Reg::Duplicate<int32_t>(rC, FLAG_BITS_CONDC);
        AscendC::Reg::And(rTmp, rF, rC, mask);
        AscendC::Reg::Compares<int32_t, AscendC::CMPMODE::EQ>(m2, rTmp, FLAG_BITS_CONDC, mask);
        AscendC::Reg::Duplicate<int32_t>(rC, FLAG_T1SAMEDO);
        AscendC::Reg::And(rTmp, rF, rC, mask);
        AscendC::Reg::Compares<int32_t, AscendC::CMPMODE::NE>(m3, rTmp, 0, mask);
        AscendC::Reg::And(m2, m2, m3, mask); // condC
        //   yC = eq(doSign, scSign)(4608) ? −Inf : +Inf
        AscendC::Reg::Duplicate<int32_t>(rC, FLAG_BITS_DOSIGN_SCSIGN);
        AscendC::Reg::And(rTmp, rF, rC, mask);
        AscendC::Reg::Compares<int32_t, AscendC::CMPMODE::EQ>(m3, rTmp, FLAG_BITS_DOSIGN_SCSIGN, mask);
        AscendC::Reg::Compares<int32_t, AscendC::CMPMODE::EQ>(m4, rTmp, 0, mask);
        AscendC::Reg::Or(m3, m3, m4, mask);             // eq(doSign, scSign)
        AscendC::Reg::Select(rZeroF, rNinf, rPinf, m3); // yC（rZeroF 复用为值寄存器）
        AscendC::Reg::Select(rY, rZeroF, rY, m2);       // condC → yC
        //   yB = eq(gSign, scSign)(4112) ? +Inf : −Inf
        AscendC::Reg::Duplicate<int32_t>(rC, FLAG_BITS_GSIGN_SCSIGN);
        AscendC::Reg::And(rTmp, rF, rC, mask);
        AscendC::Reg::Compares<int32_t, AscendC::CMPMODE::EQ>(m2, rTmp, FLAG_BITS_GSIGN_SCSIGN, mask);
        AscendC::Reg::Compares<int32_t, AscendC::CMPMODE::EQ>(m3, rTmp, 0, mask);
        AscendC::Reg::Or(m2, m2, m3, mask);             // eq(gSign, scSign)
        AscendC::Reg::Select(rZeroF, rPinf, rNinf, m2); // yB
        AscendC::Reg::Select(rY, rZeroF, rY, m1);       // condB → yB
        AscendC::Reg::StoreAlign(dstY + off, rY, mask);
    }
}

// ===========================================================================
// 2 个 Cast 断点 VF
// ===========================================================================
// CastUpVF: b16 → f32 升位（kCastTraitB16ToF32 = ZERO/UNKNOWN/ZEROING/CAST_NONE）
//   LoadAlign<T, DIST_UNPACK_B16>：每次加载 VL/2 字节（64 个 T 元素）解包到
//   f32 lane 低 16 位（漏 DIST_UNPACK_B16 报 lane 不匹配）
template <typename T>
__simd_vf__ inline void CastUpVF(__ubuf__ float* dst, __ubuf__ T* src, int64_t count, uint16_t repeat)
{
    AscendC::Reg::RegTensor<float> f32Reg;
    AscendC::Reg::RegTensor<T> b16Reg;
    AscendC::Reg::MaskReg mask;
    constexpr uint32_t VL_F32 = 256 / sizeof(float); // = 64
    uint32_t remaining = static_cast<uint32_t>(count);
    for (uint16_t i = 0; i < repeat; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(VL_F32);
        mask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(b16Reg, src + off);
        AscendC::Reg::Cast<float, T, kCastTraitB16ToF32>(f32Reg, b16Reg, mask);
        AscendC::Reg::StoreAlign(dst + off, f32Reg, mask);
    }
}

// CastDownVF: f32 → b16 回落（kCastTraitF32ToB16 = ZERO/NO_SAT/ZEROING/CAST_RINT）
//   StoreAlign<T, DIST_PACK_B32>：从 f32 lane 低 16 位提取 b16 打包成 dense
//   （漏 DIST_PACK_B32 典型症状 output[0] 对、其余全错）
template <typename T>
__simd_vf__ inline void CastDownVF(__ubuf__ T* dst, __ubuf__ float* src, int64_t count, uint16_t repeat)
{
    AscendC::Reg::RegTensor<float> f32Reg;
    AscendC::Reg::RegTensor<T> b16Reg;
    AscendC::Reg::MaskReg mask;
    constexpr uint32_t VL_F32 = 256 / sizeof(float);
    uint32_t remaining = static_cast<uint32_t>(count);
    for (uint16_t i = 0; i < repeat; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(VL_F32);
        mask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::LoadAlign(f32Reg, src + off);
        AscendC::Reg::Cast<T, float, kCastTraitF32ToB16>(b16Reg, f32Reg, mask);
        AscendC::Reg::StoreAlign<T, AscendC::Reg::StoreDist::DIST_PACK_B32>(dst + off, b16Reg, mask);
    }
}

// ===========================================================================
// 性能优化新增 VF（Stage-1：条件化 Fix 链 + 批量 CopyIn 扁平流水线）
// ===========================================================================
// ChainA1FusedVF: t_a = (cast(x) - bm) * ds * inv_num  [Cast+Sub+Mul+Muls, 链长 4]
//   与 ChainA1VF 数值完全一致（cast 确定性），仅将 b16→f32 升位融合进链内，
//   省去 x_f32 的 UB 物化（1 写 1 读，8B/元素）。x 仍以 T 原生 dtype 驻留 UB。
template <typename T>
__simd_vf__ inline void ChainA1FusedVF(__ubuf__ float* dst, __ubuf__ T* x, __ubuf__ float* bm, __ubuf__ float* ds,
                                       float invNum, int64_t count, uint16_t repeat)
{
    AscendC::Reg::RegTensor<float> rX, rBm, rDs, rDst;
    AscendC::Reg::RegTensor<T> rXB16;
    AscendC::Reg::MaskReg mask;
    constexpr uint32_t VL_F32 = 256 / sizeof(float);
    uint32_t remaining = static_cast<uint32_t>(count);
    for (uint16_t i = 0; i < repeat; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(VL_F32);
        mask = AscendC::Reg::UpdateMask<float>(remaining);
        AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(rXB16, x + off);
        AscendC::Reg::Cast<float, T, kCastTraitB16ToF32>(rX, rXB16, mask); // b16 → f32
        AscendC::Reg::LoadAlign(rBm, bm + off);
        AscendC::Reg::LoadAlign(rDs, ds + off);
        AscendC::Reg::Sub(rDst, rX, rBm, mask);       // x - bm
        AscendC::Reg::Mul(rDst, rDst, rDs, mask);     // * ds
        AscendC::Reg::Muls(rDst, rDst, invNum, mask); // * inv_num
        AscendC::Reg::StoreAlign(dst + off, rDst, mask);
    }
}

// SuspectCheckVF: scratch[0] = max(|g|, |do|, s)（tile 级 reduce-max）
//   NaN/Inf 分类修复链（FlagVF1-4+FixVF）的触发条件（condA/B/C）均要求
//   g/do/s 中存在字面 ±Inf（s=sqrt(bv+eps) ≥ 0，仅 +Inf 可能）；全有限元素时
//   修复条件恒不成立（幂等），可整体跳过。本链产出 tile 幅值上界：
//     mx > 3e38（含 +Inf，f32 最大 ~3.4e38）→ suspect，走原完整路径（含修复链）
//     mx ≤ 3e38 或 NaN（!(mx<=thr) 亦判 suspect，保守路由）→ 有限数据快速路径
//   快速路径与原实现仅差"修复链不触发"的分支，数值逐位一致。
template <typename T>
__simd_vf__ inline void SuspectCheckVF(__ubuf__ T* g, __ubuf__ T* s, __ubuf__ T* do_, __ubuf__ float* scratch,
                                       int64_t count, uint16_t repeat)
{
    AscendC::Reg::RegTensor<T> rG, rS, rDo, rT, rRed, rAcc;
    AscendC::Reg::MaskReg mask;
    constexpr uint32_t VL = 256 / sizeof(T);
    uint32_t remaining = static_cast<uint32_t>(count);
    AscendC::Reg::Duplicate(rAcc, T(0));
    for (uint16_t i = 0; i < repeat; ++i) {
        int32_t off = static_cast<int32_t>(i) * static_cast<int32_t>(VL);
        mask = AscendC::Reg::UpdateMask<T>(remaining);
        AscendC::Reg::LoadAlign(rG, g + off);
        AscendC::Reg::LoadAlign(rS, s + off);
        AscendC::Reg::LoadAlign(rDo, do_ + off);
        AscendC::Reg::Abs(rG, rG, mask);   // |g|（≥0）
        AscendC::Reg::Abs(rDo, rDo, mask); // |do|（≥0）
        AscendC::Reg::Max(rT, rG, rDo, mask);
        AscendC::Reg::Max(rT, rT, rS, mask);     // s ≥ 0 无需 Abs
        AscendC::Reg::Max(rAcc, rAcc, rT, mask); // 跨块逐 lane 累积（普通寄存器间 Max）
    }
    AscendC::Reg::MaskReg mAll = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
    // 单次 reduce 作用在普通累积寄存器上（Reduce→FIRST_ELEMENT 存储同 nll_loss 范式）
    AscendC::Reg::Reduce<AscendC::Reg::ReduceType::MAX, float>(rRed, rAcc, mAll);
    AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(
        scratch, rRed, mAll); // 仅写 lane0（4B）到 scratch
}

// ===========================================================================
// Pass1VF: NDHWC 寄存器驻留参数快速路径计算核（Stage-2）
//
// 背景: Stage-1 快速路径仍以 NDDMA 广播把 5 个 f32 参数展开成 tile 形状向量
//   （MTE2 20B/元素 + V 链参数读 24B/元素），占 UB 流量 ~2/3，是 NDHWC f16/bf16
//   的主要瓶颈。本路径利用参数仅随通道 c 变化的性质：
//   - 5 个参数以 C 长度原始向量一次搬入 UB（5·C·4B，无广播）；
//   - 块枚举按 (w, cb): 块 (w, cb) 覆盖 tile 内局部 [(w·C + cb·64), +min(64, C−cb·64))
//     元素（mask 截尾；C 为最内维保证块不跨行周期）；
//   - 每个 cb 的 5 个参数值装载进寄存器后跨全部 w 复用 → 参数零逐元素流量；
//   - s = sqrt(bv+eps) 每 cb 仅算一次（与逐元素计算逐位一致）。
// 数值: 与 Stage-1 快速路径同运算序（Sub/Mul/Muls/Div/Sub/Muls/Sub/Mul/Div），
//   每元素输入相同 → 结果逐位一致；|g|/|params|/s 幅值累积进 scratch（跨 tile
//   由调用方维护），kernel 末尾统一判定 suspect → 回退 ProcessTiles() 全量重算。
//
// 寄存器预算: 参数 5 + 工作约 8 + 累积 2 ≈ 15 个向量寄存器，远低于 VF 上限。
// 链长: 每 (w, cb) 块 13 op（含 b16 cast），单块内顺序依赖与既有链一致。
// ===========================================================================
template <typename T>
__simd_vf__ inline void Pass1VF(__ubuf__ T* g, __ubuf__ T* x, __ubuf__ T* y, __ubuf__ float* params, int64_t wCnt,
                                int64_t C, int64_t CB, float invNum, float eps, __ubuf__ float* scratch, bool firstTile)
{
    constexpr bool kIsB16 = !std::is_same_v<T, float>;
    (void)params;
    AscendC::Reg::RegTensor<float> rBm, rDs, rDo, rSc, rBv, rS;
    AscendC::Reg::RegTensor<float> rGf, rXf, rTa, rT1, rT2, rTmp, rYf;
    AscendC::Reg::RegTensor<float> rP1, rP2; // 参数幅值检查专用（不动原始参数寄存器）
    AscendC::Reg::RegTensor<T> rGB16, rXB16, rYB16;
    AscendC::Reg::RegTensor<float> rAcc, rRed;
    AscendC::Reg::MaskReg mask;
    // 跨 tile 幅值累积: lane0 持有当前最大值（首 tile 置 0，其余从 scratch 续）
    if (firstTile) {
        AscendC::Reg::Duplicate(rAcc, 0.0f);
    } else {
        AscendC::Reg::LoadAlign(rAcc, scratch);
    }
    for (int64_t cb = 0; cb < CB; cb++) {
        uint32_t lanes = static_cast<uint32_t>((C - cb * 64 >= 64) ? 64 : (C - cb * 64));
        mask = AscendC::Reg::UpdateMask<float>(lanes);
        AscendC::Reg::LoadAlign(rBm, params + cb * 64);         // bm
        AscendC::Reg::LoadAlign(rDs, params + C + cb * 64);     // ds
        AscendC::Reg::LoadAlign(rDo, params + 2 * C + cb * 64); // do
        AscendC::Reg::LoadAlign(rSc, params + 3 * C + cb * 64); // sc
        AscendC::Reg::LoadAlign(rBv, params + 4 * C + cb * 64); // bv
        AscendC::Reg::Adds(rBv, rBv, eps, mask);                // bv + eps
        AscendC::Reg::Sqrt(rS, rBv, mask);                      // s（每 cb 一次）
        // 参数幅值保守累积（|bm|/|ds|/|do|/|sc|/s → 独立寄存器, 原参数保持原值）
        AscendC::Reg::Abs(rP1, rBm, mask);
        AscendC::Reg::Abs(rP2, rDs, mask);
        AscendC::Reg::Max(rP1, rP1, rP2, mask);
        AscendC::Reg::Abs(rP2, rDo, mask);
        AscendC::Reg::Max(rP1, rP1, rP2, mask);
        AscendC::Reg::Abs(rP2, rSc, mask);
        AscendC::Reg::Max(rP1, rP1, rP2, mask);
        AscendC::Reg::Max(rP1, rP1, rS, mask); // s ≥ 0 无需 Abs
        AscendC::Reg::Max(rAcc, rAcc, rP1, mask);
        int32_t cBase = static_cast<int32_t>(cb * 64);
        for (int32_t w = 0; w < static_cast<int32_t>(wCnt); w++) {
            int32_t off = w * static_cast<int32_t>(C) + cBase; // tile 内局部元素偏移
            if constexpr (kIsB16) {
                AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(rGB16, g + off);
                AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(rXB16, x + off);
                AscendC::Reg::Cast<float, T, kCastTraitB16ToF32>(rGf, rGB16, mask);
                AscendC::Reg::Cast<float, T, kCastTraitB16ToF32>(rXf, rXB16, mask);
            } else {
                AscendC::Reg::LoadAlign(rGf, g + off);
                AscendC::Reg::LoadAlign(rXf, x + off);
            }
            AscendC::Reg::Sub(rTa, rXf, rBm, mask);      // x - bm
            AscendC::Reg::Mul(rTa, rTa, rDs, mask);      // * ds
            AscendC::Reg::Muls(rTa, rTa, invNum, mask);  // * inv_num
            AscendC::Reg::Div(rTmp, rTa, rS, mask);      // t_a / s
            AscendC::Reg::Sub(rT1, rGf, rTmp, mask);     // t1 = g - t_a/s
            AscendC::Reg::Muls(rTmp, rDo, invNum, mask); // do * inv_num
            AscendC::Reg::Sub(rT2, rT1, rTmp, mask);     // t2 = t1 - do*inv
            AscendC::Reg::Mul(rTmp, rT2, rSc, mask);     // t2 * sc
            AscendC::Reg::Div(rYf, rTmp, rS, mask);      // y_f32 = (t2*sc)/s
            AscendC::Reg::Abs(rTmp, rGf, mask);          // |g| 累积
            AscendC::Reg::Max(rAcc, rAcc, rTmp, mask);
            if constexpr (kIsB16) {
                AscendC::Reg::Cast<T, float, kCastTraitF32ToB16>(rYB16, rYf, mask);
                AscendC::Reg::StoreAlign<T, AscendC::Reg::StoreDist::DIST_PACK_B32>(y + off, rYB16, mask);
            } else {
                AscendC::Reg::StoreAlign(y + off, rYf, mask);
            }
        }
    }
    AscendC::Reg::MaskReg mAll = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
    AscendC::Reg::Reduce<AscendC::Reg::ReduceType::MAX, float>(rRed, rAcc, mAll);
    AscendC::Reg::StoreAlign<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(scratch, rRed, mAll);
}

// ===========================================================================
// class BN3DTrainingReduceGradKernel<T, RANK>
//
// 单一 kernel class（RANK=4 / RANK=8 双实例化 = tilingKey=0 / tilingKey=1 两
// 分支共用模板；计算链与同步结构同构，差异仅在 NDDMA 维数 ND=min(RANK,5)）。
//
// 模板参数:
//   T    — DTYPE_GRADS（编译期注入: float / half / bfloat16_t）
//   RANK — 4 / 8（TilingKey）
//
// 类成员溯源:
//   MAX_RANK = 8             — 坐标缓冲大小（本算子有效 rank ≤ 5）
//   MAX_NDDMA_DIMS = 5       — NDDMA 硬件 5 维上限
//   ND = min(RANK, 5)        — RANK=4 → 4; RANK=8 → 5
//   VL_F32 = GetVecLen()/4   — VF 链全 float, 每寄存器 256B = 64 个 f32
//   PHYS_NODES = 8           — TBuf 槽位 S0..S7（批量 CopyIn 扁平流水线布局）+ 32B scratch
//   gmIn_[2] / gmParam_[5]   — 7 输入 dtype 异构 → 拆两组（前 2 个 T, 后 5 个 f32）
// ===========================================================================
template <typename T, int64_t RANK>
class BN3DTrainingReduceGradKernel {
    static constexpr int64_t MAX_RANK = 8;       // 坐标缓冲大小（本算子有效 rank ≤ 5）
    static constexpr int64_t MAX_NDDMA_DIMS = 5; // NDDMA 硬件 5 维上限
    static constexpr int64_t ND = (RANK <= MAX_NDDMA_DIMS) ? RANK : MAX_NDDMA_DIMS;
    static constexpr uint32_t VL_F32 = AscendC::GetVecLen() / sizeof(float); // = 64

    AscendC::TPipe pipe_;
    const BN3DTrainingReduceGradTilingData<RANK>* td_;
    AscendC::GlobalTensor<T> gmIn_[NUM_DATA_INPUTS];             // grads / x（dtype T）
    AscendC::GlobalTensor<float> gmParam_[NUM_PARAM_INPUTS];     // 5 参数（恒 f32）
    AscendC::GlobalTensor<T> gmOut_[MAX_OUTPUT_SLOTS];           // y
    AscendC::TBuf<AscendC::TPosition::VECCALC> buf_[PHYS_NODES]; // P=8 个槽位 S0..S7
    AscendC::TBuf<AscendC::TPosition::VECCALC> scratchBuf_; // suspect 标量（4B，独占避免与 VF 链竞争）
    AscendC::MultiCopyParams<T, ND> nddmaParams_[NUM_DATA_INPUTS];
    AscendC::MultiCopyParams<float, ND> nddmaParamParams_[NUM_PARAM_INPUTS];
    int64_t nddmaOuterIters_[NUM_DATA_INPUTS]; // RANK>5 逐段搬运计数（本算子恒 1）
    int64_t nddmaParamOuterIters_[NUM_PARAM_INPUTS];
    int64_t nddmaDims_;
    float invNum_; // 1/num
    float eps_;    // attr epsilon（TilingData 传入, 禁止硬编码）

public:
    __aicore__ inline void Init(GM_ADDR inputs[MAX_INPUT_SLOTS], GM_ADDR outputs[MAX_OUTPUT_SLOTS],
                                const BN3DTrainingReduceGradTilingData<RANK>* td);
    __aicore__ inline void Process();

private:
    __aicore__ inline void Pass1();                     // Stage-2: NDHWC 寄存器参数快速路径
    __aicore__ inline void ProcessTiles(bool pass1Ran); // Stage-1 tile 流水线（快速/完整二选一）
    template <typename DT>
    __aicore__ inline void Pass1CopyIn(const AscendC::GlobalTensor<DT>& gm, int64_t elemOff, int slot, int64_t count,
                                       int64_t dstElemOff = 0);
    __aicore__ inline void CopyInData(const int64_t* coord, int dataIdx, int slot, int64_t aISeg);
    __aicore__ inline void CopyInParam(const int64_t* coord, int paramIdx, int slot, int64_t aISeg);
    __aicore__ inline void CopyOutOne(const int64_t* coord, int slot, int64_t aISeg);
    template <typename DT>
    __aicore__ inline void InitNddmaGroup(AscendC::MultiCopyParams<DT, ND>* nddmaArr, int64_t* outerIters,
                                          int64_t slotBase, int64_t numSlots);
    template <typename DT>
    __aicore__ inline void CopyInBrcImpl(const int64_t* coord, const AscendC::GlobalTensor<DT>* gmArr,
                                         const AscendC::MultiCopyParams<DT, ND>* nddmaArr, const int64_t* outerIters,
                                         int64_t groupIdx, int64_t gmSlot, int slot, int64_t aISeg);
    // ---- Stage-2 pass1 状态（Init 自推导, 见 Init 尾部）----
    bool p1Enabled_ = false; // NDHWC 且 C ∈ [12, perBufBytes/20] 且 C 不参与切分
    bool p1Suspect_ = false; // pass1/passN 末尾幅值判定 → 回退 ProcessTiles
    int64_t p1C_ = 0;        // 通道数（最内维）
    int64_t p1CB_ = 0;       // ceil(C / 64) 个 c-block
};

// ===========================================================================
// Init: GM 绑定 + TBuf 分配 + NDDMA 参数预计算 + 标量预计算
// ===========================================================================
template <typename T, int64_t RANK>
__aicore__ inline void BN3DTrainingReduceGradKernel<T, RANK>::Init(GM_ADDR inputs[MAX_INPUT_SLOTS],
                                                                   GM_ADDR outputs[MAX_OUTPUT_SLOTS],
                                                                   const BN3DTrainingReduceGradTilingData<RANK>* td)
{
    td_ = td;
    // GM 绑定: 前 2 个输入（grads/x）按 T 绑定, 后 5 个参数按 f32 绑定
    // （__gm__ → __gm__ C 风格转换允许）
    for (int i = 0; i < NUM_DATA_INPUTS; i++)
        gmIn_[i].SetGlobalBuffer((__gm__ T*)inputs[IN_GRADS + i]);
    for (int i = 0; i < NUM_PARAM_INPUTS; i++)
        gmParam_[i].SetGlobalBuffer((__gm__ float*)inputs[IN_DIFF_SCALE + i]);
    for (int i = 0; i < MAX_OUTPUT_SLOTS; i++)
        gmOut_[i].SetGlobalBuffer((__gm__ T*)outputs[i]);
    // TBuf 分配: PHYS_NODES=8 个槽位（批量 CopyIn 扁平布局）+ 1 个 256B scratch
    // （suspect 幅值标量专用; 256B=64 f32 使 LoadAlign 续读不越界）；perBufBytes
    //   由 host 按 (UB − 1024)/8 预留 scratch 与 masked 块过读余量后下发
    for (int i = 0; i < PHYS_NODES; i++)
        pipe_.InitBuffer(buf_[i], td_->perBufBytes);
    pipe_.InitBuffer(scratchBuf_, 256);
    // ---- Stage-2 pass1 使能自推导（无 TilingData 变更, 全 kernel 侧判定）----
    //   NDHWC 判据: 参数张量（diff_scale 等 shape 为 1,...,C,...,1）仅在 C 所在
    //   维 stride ≠ 0 → 该维即通道轴; 通道轴 == RANK−1（最内）时 C 为最内维,
    //   pass1 的 (w, cb) 块枚举才成立。C ∈ [12, perBufBytes/20)（过小 lane 利用
    //   率不足、过大参数驻留超槽位）; split.axis < RANK−1 保证 tile 均为完整
    //   (w, cb) 块并集（C 不参与 UB 切分）。
    {
        int64_t ch = -1;
        for (int64_t d = 0; d < RANK; d++) {
            if (td_->inputStrides[IN_DIFF_SCALE][d] != 0) {
                if (ch >= 0) {
                    ch = -1;
                    break;
                } // 两个以上非零 stride → 非纯内维通道
                ch = d;
            }
        }
        if (ch == RANK - 1 && td_->split.axis < RANK - 1) {
            int64_t C = td_->maxBroShape[RANK - 1];
            // LoadAlign 32B 对齐约束: 块地址 w·C·sizeof(T) 恒对齐 ⟺ C·sizeof(T) ≡ 0 (mod 32)
            // （f32 → C % 8 == 0; f16/bf16 → C % 16 == 0）。
            // 注: odd-C 的 padded 行方案（MTE2/MTE3 以 stride 补齐）因 MTE stride 亦需
            // 32B 对齐而不可行（dstStride=(pad−C)·szT 无法对齐）, odd-C 走 Stage-1 路径。
            constexpr int64_t kAlignElems = 32 / static_cast<int64_t>(sizeof(T));
            if (C >= 12 && C % kAlignElems == 0 && 5 * C * 4 <= td_->perBufBytes) {
                p1Enabled_ = true;
                p1C_ = C;
                constexpr int64_t kCBlockElems = 64; // Pass1 c-block 大小：64 通道元素/块
                p1CB_ = (C + (kCBlockElems - 1)) / kCBlockElems;
            }
        }
    }
    // NDDMA 参数预计算: 两组（T 数据组 + f32 参数组）, 五字段全初始化
    nddmaDims_ = (RANK - td_->split.axis <= ND) ? (RANK - td_->split.axis) : ND;
    InitNddmaGroup<T>(nddmaParams_, nddmaOuterIters_, IN_GRADS, NUM_DATA_INPUTS);
    InitNddmaGroup<float>(nddmaParamParams_, nddmaParamOuterIters_, IN_DIFF_SCALE, NUM_PARAM_INPUTS);
    // 标量预计算: ·inv_num 替代 ÷num; eps 来自 TilingData（禁止硬编码）
    invNum_ = 1.0f / static_cast<float>(td_->num);
    eps_ = td_->epsilon;
    // 多核场景刷新 NDDMA Cache
    AscendC::NdDmaDci();
}

// ===========================================================================
// InitNddmaGroup: NDDMA 参数预计算（adam Init 行 118–144 同构, 按 DT 分组）
// 三层不重叠: Flat(d<k) / Outer(k≤d<RANK-nddmaDims) / NDDMA(d≥RANK-nddmaDims);
// 填充方向翻转: NDDMA dim[0] = 最内维, 循环从 d=RANK-1 起 nd=0 递增
// ===========================================================================
template <typename T, int64_t RANK>
template <typename DT>
__aicore__ inline void BN3DTrainingReduceGradKernel<T, RANK>::InitNddmaGroup(AscendC::MultiCopyParams<DT, ND>* nddmaArr,
                                                                             int64_t* outerIters, int64_t slotBase,
                                                                             int64_t numSlots)
{
    int64_t k = td_->split.axis;
    for (int64_t inp = 0; inp < numSlots; inp++) {
        int64_t inner = 1;
        int64_t nd = 0;
        for (int64_t d = RANK - 1; d >= k && nd < ND; d--) {
            nddmaArr[inp].loopInfo.loopSize[nd] = (d == k) ? 0 : td_->maxBroShape[d];
            nddmaArr[inp].loopInfo.loopSrcStride[nd] = td_->inputStrides[slotBase + inp][d];
            nddmaArr[inp].loopInfo.loopDstStride[nd] = inner;
            nddmaArr[inp].loopInfo.loopLpSize[nd] = 0; // 必须显式填 0, 否则随机值进硬件
            nddmaArr[inp].loopInfo.loopRpSize[nd] = 0;
            inner *= (d == k) ? td_->split.aI : td_->maxBroShape[d];
            nd++;
        }
        for (; nd < ND; nd++) {
            nddmaArr[inp].loopInfo.loopSize[nd] = 1;
            nddmaArr[inp].loopInfo.loopSrcStride[nd] = 0;
            nddmaArr[inp].loopInfo.loopDstStride[nd] = inner;
            nddmaArr[inp].loopInfo.loopLpSize[nd] = 0;
            nddmaArr[inp].loopInfo.loopRpSize[nd] = 0;
        }
        // outer loop 只覆盖 flat 层与 NDDMA 之间的 gap（本算子有效 rank ≤ 5 → 恒为 1）
        outerIters[inp] = 1;
        for (int64_t d = k; d < RANK - nddmaDims_; d++)
            outerIters[inp] *= (d == k) ? td_->split.aI : td_->maxBroShape[d];
    }
}

// ===========================================================================
// Process: 批量 CopyIn 扁平流水线（性能优化版）
//          每 tile: MTE2 一次性搬入全部 7 输入（8 槽位 S0..S7 布局）
//          → 1 次 MTE2_V 同步 → V 计算段 → 1 次 V_MTE3 同步 → CopyOut。
//          tile 内无 MTE2/V 交错 → 无需 V_MTE2 事件（原实现 6~8 次交替同步）。
//
// slot 布局（两 dtype 路径统一装载，计算期按死参数原地覆写）:
//   S0: x(T)   — f16 suspect 路径中段让位 flags；两路径末段为 y(T)（CastDown 落点）
//   S1: g(T/f32) — f16: CastUp 至 S7 后让位（suspect 路径承接 x_f32）
//   S2: bm → t_a      S3: ds      S4: bv → s      S5: do → t2      S6: sc → y_f32
//   S7: f16: g_f32 → t1；f32: flags（仅 suspect 路径）
//
// 条件化 NaN/Inf 分类修复（Stage-1 核心优化）:
//   修复链触发条件（condA/B/C, 见 FixVF 注释）均要求 g/do/s 存在字面 ±Inf。
//   SuspectCheckVF 求 tile 级 max(|g|, |do|, s) 写入 scratch（32B 独立 TBuf），
//   标量侧 LocalMemBar<VEC_STORE, SCALAR_LOAD> + GetValue 后分支:
//     !(mx <= 3e38)（含 ±Inf 与 NaN, 保守路由）→ suspect: 原完整路径
//       （FlagVF1-4 + 5 链 + FixVF, 与原实现逐位一致, 覆盖极端值精度用例）
//     否则 → 快速路径: 仅 5 条数学链（运算序与 golden 逐步对齐, 数值与原实现
//       逐位一致——修复链在全有限元素 tile 上恒为幂等空操作）
//
// 同步事件 3 类: MTE2_V（批量搬入后）/ V_MTE3（计算段后）/ MTE3_MTE2（跨轮 WAR）。
// ===========================================================================
// Process: 总分派（Stage-2）
//   p1Enabled_（NDHWC 且 C 满足条件）→ Pass1() 寄存器参数快速路径；末尾幅值
//   判定 suspect（|g|/|params|/s > 3e38 或 NaN）→ 回退 ProcessTiles(true) 全量
//   重算（分区与 Pass1 完全一致 → 每核独立回退即正确）。
//   否则（NCDHW / 小 C / C 参与切分）直接 ProcessTiles(false)。
//   退出前统一 PipeBarrier<PIPE_ALL>。
// ===========================================================================
template <typename T, int64_t RANK>
__aicore__ inline void BN3DTrainingReduceGradKernel<T, RANK>::Process()
{
    if (p1Enabled_) {
        Pass1();
        if (p1Suspect_)
            ProcessTiles(true);
    } else {
        ProcessTiles(false);
    }
    AscendC::PipeBarrier<PIPE_ALL>();
}

// ===========================================================================
// Pass1: NDHWC 寄存器参数快速路径驱动（每 tile: MTE2 数据 2 拷 → V(Pass1VF)
//        → MTE3 y）。tile 枚举/多核分区与 ProcessTiles 完全一致（同一 GetCoreRange
//        与 flat→coord 映射），仅搬运方式不同: 数据为 1D 连续拷（无 NDDMA 广播），
//        参数整存于 S3（5 段 C 长度, 每 tile 复用）。
// slot 布局: S0=g(T) / S1=x(T) / S2=y(T) / S3=params(5×C×4B)
// ===========================================================================
template <typename T, int64_t RANK>
__aicore__ inline void BN3DTrainingReduceGradKernel<T, RANK>::Pass1()
{
    int32_t evMTE2toV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
    int32_t evVtoMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
    int32_t evMTE3toMTE2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
    int32_t evVtoS = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S));

    int64_t start, end;
    GetCoreRange(AscendC::GetBlockIdx(), td_->multicore.tilesMain, td_->multicore.coresTail, start, end);
    p1Suspect_ = false;
    if (start >= end)
        return; // 本核无 tile（分区与 ProcessTiles 一致, 无需回退）

    constexpr int S0 = 0, S1 = 1, S2 = 2, S3 = 3;

    int64_t innerCount = 1;
    for (int64_t d = td_->split.axis + 1; d < RANK; d++)
        innerCount *= td_->maxBroShape[d];

    int64_t coord[MAX_RANK] = {};
    for (int64_t flat = start; flat < end; flat++) {
        int64_t aISeg = GetUBSplitRange(flat % td_->split.aO, td_->split.aO, td_->split.aI, td_->split.aITail);
        int64_t count = aISeg * innerCount; // 本 tile 元素数（C 的整倍数）
        FlatToEffectiveCoord(flat, td_->maxBroShape, RANK, td_->split.axis, td_->split.aI, td_->split.aO, coord);

        // 跨轮 WAR: 上轮 CopyOut(MTE3) 读毕 → 本轮 CopyIn(MTE2) 可覆写（首轮跳过）
        if (flat != start)
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evMTE3toMTE2);

        // ===== MTE2: 数据 2 拷（1D 连续, 无广播）+ 参数 5 段（仅首 tile, 复用至后续 tile）=====
        int64_t gOff = CalcInputOffset(coord, td_->inputStrides[IN_GRADS], RANK);
        int64_t xOff = CalcInputOffset(coord, td_->inputStrides[IN_X], RANK);
        int64_t wCnt = count / p1C_;
        Pass1CopyIn(gmIn_[0], gOff, S0, count); // grads
        Pass1CopyIn(gmIn_[1], xOff, S1, count); // x
        if (flat == start) {
            // 5 参数 C 长度原始向量 → S3（bm|ds|do|sc|bv 顺序与 Pass1VF 布局一致）
            constexpr int P_BM = 3, P_DS = 0, P_DO = 1, P_SC = 2, P_BV = 4; // gmParam_ 组内序
            Pass1CopyIn(gmParam_[P_BM], 0, S3, p1C_, 0 * p1C_);
            Pass1CopyIn(gmParam_[P_DS], 0, S3, p1C_, 1 * p1C_);
            Pass1CopyIn(gmParam_[P_DO], 0, S3, p1C_, 2 * p1C_);
            Pass1CopyIn(gmParam_[P_SC], 0, S3, p1C_, 3 * p1C_);
            Pass1CopyIn(gmParam_[P_BV], 0, S3, p1C_, 4 * p1C_);
        }
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);

        // ===== V: Pass1VF —— (w, cb) 块枚举 + 寄存器参数, 幅值累积进 scratch =====
        asc_vf_call<Pass1VF<T>>(
            (__ubuf__ T*)buf_[S0].template Get<T>().GetPhyAddr(), (__ubuf__ T*)buf_[S1].template Get<T>().GetPhyAddr(),
            (__ubuf__ T*)buf_[S2].template Get<T>().GetPhyAddr(),
            (__ubuf__ float*)buf_[S3].template Get<float>().GetPhyAddr(), wCnt, p1C_, p1CB_, invNum_, eps_,
            (__ubuf__ float*)scratchBuf_.template Get<float>().GetPhyAddr(), flat == start);

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evVtoMTE3);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evVtoMTE3);

        // ===== MTE3: y 搬出（cnt 计算与 ProcessTiles 同源）=====
        CopyOutOne(coord, S2, aISeg);

        // 跨轮 WAR 置位（末轮亦置: 供回退 ProcessTiles 首 tile 等待）
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evMTE3toMTE2);
    }

    // ===== 末尾幅值判定: scratch（跨 tile 累积的 max(|g|,|params|,s)）→ 标量 =====
    AscendC::SetFlag<AscendC::HardEvent::V_S>(evVtoS);
    AscendC::WaitFlag<AscendC::HardEvent::V_S>(evVtoS);
    p1Suspect_ = !(scratchBuf_.template Get<float>().GetValue(0) <= kSuspectThreshold);
}

// ===========================================================================
// Pass1CopyIn: 1D 连续 GM→UB 拷贝（pass1 专用, 无广播; dstElemOff 支持 S3 内
//              多段参数布局）
// ===========================================================================
template <typename T, int64_t RANK>
template <typename DT>
__aicore__ inline void BN3DTrainingReduceGradKernel<T, RANK>::Pass1CopyIn(const AscendC::GlobalTensor<DT>& gm,
                                                                          int64_t elemOff, int slot, int64_t count,
                                                                          int64_t dstElemOff)
{
    // 连续块拷贝走 DataCopyPad 原生路径（1D NDDMA 参数路径在大块下有病理性开销）
    AscendC::DataCopyExtParams extParams;
    extParams.blockCount = 1;
    extParams.blockLen = count * sizeof(DT); // 有效字节（元素数 × sizeof）
    extParams.srcStride = 0;
    extParams.dstStride = 0;
    AscendC::DataCopyPadExtParams<DT> padParams; // 默认 isPad=false
    AscendC::DataCopyPad(buf_[slot].template Get<DT>()[dstElemOff], gm[elemOff], extParams, padParams);
}

// ===========================================================================
// ProcessTiles: Stage-1 tile 流水线（批量 CopyIn + suspect 二选一计算段）
//   pass1Ran: Pass1 已执行且写过 y → 首 tile 需等待 Pass1 末尾的 MTE3_MTE2 置位
// ===========================================================================
template <typename T, int64_t RANK>
__aicore__ inline void BN3DTrainingReduceGradKernel<T, RANK>::ProcessTiles(bool pass1Ran)
{
    int32_t evMTE2toV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
    int32_t evVtoMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
    int32_t evMTE3toMTE2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
    int32_t evVtoS = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S));

    int64_t start, end;
    GetCoreRange(AscendC::GetBlockIdx(), td_->multicore.tilesMain, td_->multicore.coresTail, start, end);

    constexpr int S0 = 0, S1 = 1, S2 = 2, S3 = 3, S4 = 4, S5 = 5, S6 = 6, S7 = 7;

    int64_t innerCount = 1;
    for (int64_t d = td_->split.axis + 1; d < RANK; d++)
        innerCount *= td_->maxBroShape[d];

    int64_t coord[MAX_RANK] = {};
    for (int64_t flat = start; flat < end; flat++) {
        int64_t aISeg = GetUBSplitRange(flat % td_->split.aO, td_->split.aO, td_->split.aI, td_->split.aITail);
        int64_t count = aISeg * innerCount;                     // 本 tile 元素数（≤ perBufElems）
        uint16_t repeat = AscendC::CeilDivision(count, VL_F32); // VF 循环次数（f32 口径）
        FlatToEffectiveCoord(flat, td_->maxBroShape, RANK, td_->split.axis, td_->split.aI, td_->split.aO, coord);

        // 跨轮 WAR: 上轮 CopyOut(MTE3) 读毕 → 本轮 CopyIn(MTE2) 可覆写
        // （首轮: 非 pass1 回退时跳过; pass1 回退时等 Pass1 末尾置位）
        if (flat != start || pass1Ran)
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evMTE3toMTE2);

        // ===== MTE2 批量搬入: 7 输入一次到位（数据 2 + 参数 5, NDDMA 随路广播）=====
        CopyInData(coord, IN_X, S0, aISeg);
        CopyInData(coord, IN_GRADS, S1, aISeg);
        CopyInParam(coord, IN_BATCH_MEAN, S2, aISeg);
        CopyInParam(coord, IN_DIFF_SCALE, S3, aISeg);
        CopyInParam(coord, IN_BATCH_VARIANCE, S4, aISeg);
        CopyInParam(coord, IN_DIFF_OFFSET, S5, aISeg);
        CopyInParam(coord, IN_SCALE, S6, aISeg);
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);

        // 编译期 dtype 分派
        if constexpr (std::is_same_v<T, float>) {
            // ===== fp32 路径: ChainS 原地 S4（bv→s）→ suspect 判定 → 二选一计算段 =====
            asc_vf_call<ChainSVF<float>>((__ubuf__ float*)buf_[S4].template Get<float>().GetPhyAddr(),
                                         (__ubuf__ float*)buf_[S4].template Get<float>().GetPhyAddr(), eps_, count,
                                         repeat);
            asc_vf_call<SuspectCheckVF<float>>((__ubuf__ float*)buf_[S1].template Get<float>().GetPhyAddr(),
                                               (__ubuf__ float*)buf_[S4].template Get<float>().GetPhyAddr(),
                                               (__ubuf__ float*)buf_[S5].template Get<float>().GetPhyAddr(),
                                               (__ubuf__ float*)scratchBuf_.template Get<float>().GetPhyAddr(), count,
                                               repeat);
            // V→S 硬件事件: 保证 SuspectCheckVF 的 scratch 存储完成后标量再读
            // （LocalMemBar 为 __simd_callee__ 仅限 VF 函数内使用, 标量上下文须走事件）
            AscendC::SetFlag<AscendC::HardEvent::V_S>(evVtoS);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(evVtoS);
            const bool suspect = !(scratchBuf_.template Get<float>().GetValue(0) <= kSuspectThreshold);
            if (!suspect) {
                // -- 快速路径: 5 条数学链, 全原地覆写死参数槽 --
                // t_a = (x - bm) * ds * inv_num → S2（覆写 bm）
                asc_vf_call<ChainA1VF<float>>((__ubuf__ float*)buf_[S2].template Get<float>().GetPhyAddr(),
                                              (__ubuf__ float*)buf_[S0].template Get<float>().GetPhyAddr(),
                                              (__ubuf__ float*)buf_[S2].template Get<float>().GetPhyAddr(),
                                              (__ubuf__ float*)buf_[S3].template Get<float>().GetPhyAddr(), invNum_,
                                              count, repeat);
                // t1 = grads - t_a/s → S1（覆写 g）
                asc_vf_call<ChainA2VF<float>>((__ubuf__ float*)buf_[S1].template Get<float>().GetPhyAddr(),
                                              (__ubuf__ float*)buf_[S1].template Get<float>().GetPhyAddr(),
                                              (__ubuf__ float*)buf_[S2].template Get<float>().GetPhyAddr(),
                                              (__ubuf__ float*)buf_[S4].template Get<float>().GetPhyAddr(), count,
                                              repeat);
                // t2 = t1 - do * inv_num → S5（覆写 do）
                asc_vf_call<ChainBVF<float>>((__ubuf__ float*)buf_[S5].template Get<float>().GetPhyAddr(),
                                             (__ubuf__ float*)buf_[S1].template Get<float>().GetPhyAddr(),
                                             (__ubuf__ float*)buf_[S5].template Get<float>().GetPhyAddr(), invNum_,
                                             count, repeat);
                // y = (t2 * sc) / s → S6（覆写 sc）
                asc_vf_call<ChainCVF<float>>((__ubuf__ float*)buf_[S6].template Get<float>().GetPhyAddr(),
                                             (__ubuf__ float*)buf_[S5].template Get<float>().GetPhyAddr(),
                                             (__ubuf__ float*)buf_[S6].template Get<float>().GetPhyAddr(),
                                             (__ubuf__ float*)buf_[S4].template Get<float>().GetPhyAddr(), count,
                                             repeat);
            } else {
                // -- 完整路径（原实现同构: Flag 链 + 5 链 + Fix, flags 驻 S7）--
                asc_vf_call<FlagVF1<float>>((__ubuf__ int32_t*)buf_[S7].template Get<int32_t>().GetPhyAddr(),
                                            (__ubuf__ float*)buf_[S0].template Get<float>().GetPhyAddr(),
                                            (__ubuf__ float*)buf_[S2].template Get<float>().GetPhyAddr(),
                                            (__ubuf__ float*)buf_[S3].template Get<float>().GetPhyAddr(), count,
                                            repeat);
                asc_vf_call<ChainA1VF<float>>((__ubuf__ float*)buf_[S2].template Get<float>().GetPhyAddr(),
                                              (__ubuf__ float*)buf_[S0].template Get<float>().GetPhyAddr(),
                                              (__ubuf__ float*)buf_[S2].template Get<float>().GetPhyAddr(),
                                              (__ubuf__ float*)buf_[S3].template Get<float>().GetPhyAddr(), invNum_,
                                              count, repeat);
                asc_vf_call<FlagVF2<float>>((__ubuf__ int32_t*)buf_[S7].template Get<int32_t>().GetPhyAddr(),
                                            (__ubuf__ int32_t*)buf_[S7].template Get<int32_t>().GetPhyAddr(),
                                            (__ubuf__ float*)buf_[S1].template Get<float>().GetPhyAddr(),
                                            (__ubuf__ float*)buf_[S4].template Get<float>().GetPhyAddr(),
                                            (__ubuf__ float*)buf_[S2].template Get<float>().GetPhyAddr(), count,
                                            repeat);
                asc_vf_call<ChainA2VF<float>>((__ubuf__ float*)buf_[S1].template Get<float>().GetPhyAddr(),
                                              (__ubuf__ float*)buf_[S1].template Get<float>().GetPhyAddr(),
                                              (__ubuf__ float*)buf_[S2].template Get<float>().GetPhyAddr(),
                                              (__ubuf__ float*)buf_[S4].template Get<float>().GetPhyAddr(), count,
                                              repeat);
                asc_vf_call<FlagVF3<float>>((__ubuf__ int32_t*)buf_[S7].template Get<int32_t>().GetPhyAddr(),
                                            (__ubuf__ int32_t*)buf_[S7].template Get<int32_t>().GetPhyAddr(),
                                            (__ubuf__ float*)buf_[S5].template Get<float>().GetPhyAddr(),
                                            (__ubuf__ float*)buf_[S1].template Get<float>().GetPhyAddr(), count,
                                            repeat);
                asc_vf_call<ChainBVF<float>>((__ubuf__ float*)buf_[S5].template Get<float>().GetPhyAddr(),
                                             (__ubuf__ float*)buf_[S1].template Get<float>().GetPhyAddr(),
                                             (__ubuf__ float*)buf_[S5].template Get<float>().GetPhyAddr(), invNum_,
                                             count, repeat);
                asc_vf_call<FlagVF4<float>>((__ubuf__ int32_t*)buf_[S7].template Get<int32_t>().GetPhyAddr(),
                                            (__ubuf__ int32_t*)buf_[S7].template Get<int32_t>().GetPhyAddr(),
                                            (__ubuf__ float*)buf_[S6].template Get<float>().GetPhyAddr(), count,
                                            repeat);
                asc_vf_call<ChainCVF<float>>((__ubuf__ float*)buf_[S6].template Get<float>().GetPhyAddr(),
                                             (__ubuf__ float*)buf_[S5].template Get<float>().GetPhyAddr(),
                                             (__ubuf__ float*)buf_[S6].template Get<float>().GetPhyAddr(),
                                             (__ubuf__ float*)buf_[S4].template Get<float>().GetPhyAddr(), count,
                                             repeat);
                asc_vf_call<FixVF<float>>((__ubuf__ float*)buf_[S6].template Get<float>().GetPhyAddr(),
                                          (__ubuf__ float*)buf_[S6].template Get<float>().GetPhyAddr(),
                                          (__ubuf__ int32_t*)buf_[S7].template Get<int32_t>().GetPhyAddr(), count,
                                          repeat);
            }
            // RAW: V 写毕 → MTE3 可读
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evVtoMTE3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evVtoMTE3);
            // CopyOut y（f32 路径 y_f32 驻 S6）
            CopyOutOne(coord, S6, aISeg);
        } else {
            // ===== fp16/bf16 路径: CastUp g → S7; ChainS 原地 S4; suspect 判定; 二选一 =====
            // （T=half 与 T=bfloat16_t 共用同一分支）
            asc_vf_call<CastUpVF<T>>((__ubuf__ float*)buf_[S7].template Get<float>().GetPhyAddr(),
                                     (__ubuf__ T*)buf_[S1].template Get<T>().GetPhyAddr(), count,
                                     repeat); // g_T@S1 → g_f32@S7（S1 随即让位）
            asc_vf_call<ChainSVF<float>>((__ubuf__ float*)buf_[S4].template Get<float>().GetPhyAddr(),
                                         (__ubuf__ float*)buf_[S4].template Get<float>().GetPhyAddr(), eps_, count,
                                         repeat);
            asc_vf_call<SuspectCheckVF<float>>((__ubuf__ float*)buf_[S7].template Get<float>().GetPhyAddr(),
                                               (__ubuf__ float*)buf_[S4].template Get<float>().GetPhyAddr(),
                                               (__ubuf__ float*)buf_[S5].template Get<float>().GetPhyAddr(),
                                               (__ubuf__ float*)scratchBuf_.template Get<float>().GetPhyAddr(), count,
                                               repeat);
            // V→S 硬件事件: 保证 SuspectCheckVF 的 scratch 存储完成后标量再读
            // （LocalMemBar 为 __simd_callee__ 仅限 VF 函数内使用, 标量上下文须走事件）
            AscendC::SetFlag<AscendC::HardEvent::V_S>(evVtoS);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(evVtoS);
            const bool suspect = !(scratchBuf_.template Get<float>().GetValue(0) <= kSuspectThreshold);
            if (!suspect) {
                // -- 快速路径: x 升位融合进 A1（免 x_f32 物化）; flags/修复链整体跳过 --
                // t_a = (cast(x) - bm) * ds * inv_num → S2（覆写 bm）
                asc_vf_call<ChainA1FusedVF<T>>((__ubuf__ float*)buf_[S2].template Get<float>().GetPhyAddr(),
                                               (__ubuf__ T*)buf_[S0].template Get<T>().GetPhyAddr(),
                                               (__ubuf__ float*)buf_[S2].template Get<float>().GetPhyAddr(),
                                               (__ubuf__ float*)buf_[S3].template Get<float>().GetPhyAddr(), invNum_,
                                               count, repeat);
                // t1 = g_f32 - t_a/s → S7（覆写 g_f32）
                asc_vf_call<ChainA2VF<float>>((__ubuf__ float*)buf_[S7].template Get<float>().GetPhyAddr(),
                                              (__ubuf__ float*)buf_[S7].template Get<float>().GetPhyAddr(),
                                              (__ubuf__ float*)buf_[S2].template Get<float>().GetPhyAddr(),
                                              (__ubuf__ float*)buf_[S4].template Get<float>().GetPhyAddr(), count,
                                              repeat);
                // t2 = t1 - do * inv_num → S5（覆写 do）
                asc_vf_call<ChainBVF<float>>((__ubuf__ float*)buf_[S5].template Get<float>().GetPhyAddr(),
                                             (__ubuf__ float*)buf_[S7].template Get<float>().GetPhyAddr(),
                                             (__ubuf__ float*)buf_[S5].template Get<float>().GetPhyAddr(), invNum_,
                                             count, repeat);
                // y_f32 = (t2 * sc) / s → S6（覆写 sc）
                asc_vf_call<ChainCVF<float>>((__ubuf__ float*)buf_[S6].template Get<float>().GetPhyAddr(),
                                             (__ubuf__ float*)buf_[S5].template Get<float>().GetPhyAddr(),
                                             (__ubuf__ float*)buf_[S6].template Get<float>().GetPhyAddr(),
                                             (__ubuf__ float*)buf_[S4].template Get<float>().GetPhyAddr(), count,
                                             repeat);
            } else {
                // -- 完整路径: x 升位物化至 S1（S0 让位 flags）; Flag 链 + 5 链 + Fix --
                asc_vf_call<CastUpVF<T>>((__ubuf__ float*)buf_[S1].template Get<float>().GetPhyAddr(),
                                         (__ubuf__ T*)buf_[S0].template Get<T>().GetPhyAddr(), count,
                                         repeat); // x_T@S0 → x_f32@S1（S0 让位 flags）
                asc_vf_call<FlagVF1<float>>((__ubuf__ int32_t*)buf_[S0].template Get<int32_t>().GetPhyAddr(),
                                            (__ubuf__ float*)buf_[S1].template Get<float>().GetPhyAddr(),
                                            (__ubuf__ float*)buf_[S2].template Get<float>().GetPhyAddr(),
                                            (__ubuf__ float*)buf_[S3].template Get<float>().GetPhyAddr(), count,
                                            repeat);
                asc_vf_call<ChainA1VF<float>>((__ubuf__ float*)buf_[S2].template Get<float>().GetPhyAddr(),
                                              (__ubuf__ float*)buf_[S1].template Get<float>().GetPhyAddr(),
                                              (__ubuf__ float*)buf_[S2].template Get<float>().GetPhyAddr(),
                                              (__ubuf__ float*)buf_[S3].template Get<float>().GetPhyAddr(), invNum_,
                                              count, repeat);
                asc_vf_call<FlagVF2<float>>((__ubuf__ int32_t*)buf_[S0].template Get<int32_t>().GetPhyAddr(),
                                            (__ubuf__ int32_t*)buf_[S0].template Get<int32_t>().GetPhyAddr(),
                                            (__ubuf__ float*)buf_[S7].template Get<float>().GetPhyAddr(),
                                            (__ubuf__ float*)buf_[S4].template Get<float>().GetPhyAddr(),
                                            (__ubuf__ float*)buf_[S2].template Get<float>().GetPhyAddr(), count,
                                            repeat);
                asc_vf_call<ChainA2VF<float>>((__ubuf__ float*)buf_[S7].template Get<float>().GetPhyAddr(),
                                              (__ubuf__ float*)buf_[S7].template Get<float>().GetPhyAddr(),
                                              (__ubuf__ float*)buf_[S2].template Get<float>().GetPhyAddr(),
                                              (__ubuf__ float*)buf_[S4].template Get<float>().GetPhyAddr(), count,
                                              repeat);
                asc_vf_call<FlagVF3<float>>((__ubuf__ int32_t*)buf_[S0].template Get<int32_t>().GetPhyAddr(),
                                            (__ubuf__ int32_t*)buf_[S0].template Get<int32_t>().GetPhyAddr(),
                                            (__ubuf__ float*)buf_[S5].template Get<float>().GetPhyAddr(),
                                            (__ubuf__ float*)buf_[S7].template Get<float>().GetPhyAddr(), count,
                                            repeat);
                asc_vf_call<ChainBVF<float>>((__ubuf__ float*)buf_[S5].template Get<float>().GetPhyAddr(),
                                             (__ubuf__ float*)buf_[S7].template Get<float>().GetPhyAddr(),
                                             (__ubuf__ float*)buf_[S5].template Get<float>().GetPhyAddr(), invNum_,
                                             count, repeat);
                asc_vf_call<FlagVF4<float>>((__ubuf__ int32_t*)buf_[S0].template Get<int32_t>().GetPhyAddr(),
                                            (__ubuf__ int32_t*)buf_[S0].template Get<int32_t>().GetPhyAddr(),
                                            (__ubuf__ float*)buf_[S6].template Get<float>().GetPhyAddr(), count,
                                            repeat);
                asc_vf_call<ChainCVF<float>>((__ubuf__ float*)buf_[S6].template Get<float>().GetPhyAddr(),
                                             (__ubuf__ float*)buf_[S5].template Get<float>().GetPhyAddr(),
                                             (__ubuf__ float*)buf_[S6].template Get<float>().GetPhyAddr(),
                                             (__ubuf__ float*)buf_[S4].template Get<float>().GetPhyAddr(), count,
                                             repeat);
                asc_vf_call<FixVF<float>>((__ubuf__ float*)buf_[S6].template Get<float>().GetPhyAddr(),
                                          (__ubuf__ float*)buf_[S6].template Get<float>().GetPhyAddr(),
                                          (__ubuf__ int32_t*)buf_[S0].template Get<int32_t>().GetPhyAddr(), count,
                                          repeat);
            }
            // y(T) 回落: y_f32@S6 → S0（f16 fast: x_T 已耗; suspect: flags 已耗）
            asc_vf_call<CastDownVF<T>>((__ubuf__ T*)buf_[S0].template Get<T>().GetPhyAddr(),
                                       (__ubuf__ float*)buf_[S6].template Get<float>().GetPhyAddr(), count, repeat);
            // RAW: V 写毕 → MTE3 可读
            AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evVtoMTE3);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evVtoMTE3);
            // CopyOut y（b16 路径 y_T 驻 S0）
            CopyOutOne(coord, S0, aISeg);
        }

        // 跨轮 WAR: 本轮 CopyOut(MTE3) 读毕 → 下轮 CopyIn(MTE2) 可覆写（末轮跳过）
        if (flat != end - 1)
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evMTE3toMTE2);
    }
    // 排空由 Process() 统一 PipeBarrier<PIPE_ALL> 完成
}

// ===========================================================================
// CopyInData / CopyInParam: 公共包装（dtype 分派到 T / f32 两组 GM + NDDMA 参数）
// 调用点传入全局 input slot 编号（IN_GRADS..IN_BATCH_VARIANCE），
// 组内局部索引 = 全局编号 − 组基址（IN_GRADS=0 / IN_DIFF_SCALE=2）
// ===========================================================================
template <typename T, int64_t RANK>
__aicore__ inline void BN3DTrainingReduceGradKernel<T, RANK>::CopyInData(const int64_t* coord, int dataIdx, int slot,
                                                                         int64_t aISeg)
{
    CopyInBrcImpl<T>(coord, gmIn_, nddmaParams_, nddmaOuterIters_, dataIdx - IN_GRADS, dataIdx, slot, aISeg);
}

template <typename T, int64_t RANK>
__aicore__ inline void BN3DTrainingReduceGradKernel<T, RANK>::CopyInParam(const int64_t* coord, int paramIdx, int slot,
                                                                          int64_t aISeg)
{
    CopyInBrcImpl<float>(coord, gmParam_, nddmaParamParams_, nddmaParamOuterIters_, paramIdx - IN_DIFF_SCALE, paramIdx,
                         slot, aISeg);
}

// ===========================================================================
// CopyInBrcImpl: NDDMA 搬入核心（adam CopyInBrc 行 359–395 同构, 类型名适配）
// 随路 broadcast 在搬运阶段完成——广播轴 loopSrcStride=0（host 通道轴重塑时
//   PrecomputeInputStrides 预计算），硬件只读 1 份展开到目标；loopSize = 目标
//   shape（广播后），非源 shape。
// NDDMA 5 维上限: if constexpr (RANK <= MAX_NDDMA_DIMS) 单次 DataCopy 覆盖所有维,
//   else 分支经 nddmaOuterIters 逐段搬运超出部分（本算子有效 rank ≤ 5, 恒 1 段）
// ===========================================================================
template <typename T, int64_t RANK>
template <typename DT>
__aicore__ inline void BN3DTrainingReduceGradKernel<T, RANK>::CopyInBrcImpl(
    const int64_t* coord, const AscendC::GlobalTensor<DT>* gmArr, const AscendC::MultiCopyParams<DT, ND>* nddmaArr,
    const int64_t* outerIters, int64_t groupIdx, int64_t gmSlot, int slot, int64_t aISeg)
{
    int64_t k = td_->split.axis;
    int64_t off = CalcInputOffset(coord, td_->inputStrides[gmSlot], RANK); // 元素索引
    const int64_t* dstShape = td_->maxBroShape;

    auto params = nddmaArr[groupIdx];
    int64_t kNd = RANK - 1 - k;
    int64_t inner = 1;
    for (int64_t nd = 0; nd < ND; nd++) {
        if (nd == kNd)
            params.loopInfo.loopSize[nd] = aISeg; // split 轴段长（每次按 aISeg 覆写）
        params.loopInfo.loopDstStride[nd] = inner;
        inner *= params.loopInfo.loopSize[nd];
    }

    static constexpr AscendC::NdDmaConfig cfg = {false, AscendC::NdDmaConfig::unsetPad, AscendC::NdDmaConfig::unsetPad,
                                                 false};

    if constexpr (RANK <= MAX_NDDMA_DIMS) {
        // RANK ≤ 5: 单次 DataCopy 覆盖所有维（loopSrcStride 广播轴=0 随路展开）
        AscendC::DataCopy<DT, ND, cfg>(buf_[slot].template Get<DT>(), gmArr[groupIdx][off], params);
    } else {
        // RANK > 5: nddmaOuterIters 逐段搬运超出部分（本算子恒 1 段）
        AscendC::LocalTensor<DT> buf = buf_[slot].template Get<DT>();
        int64_t elemBase = off;
        for (int64_t oi = 0; oi < outerIters[groupIdx]; oi++) {
            int64_t elemAdj = 0, tmp = oi;
            for (int64_t d = RANK - nddmaDims_ - 1; d >= k; d--) {
                int64_t sz = (d == k) ? aISeg : dstShape[d];
                elemAdj += (tmp % sz) * td_->inputStrides[gmSlot][d];
                tmp /= sz;
            }
            AscendC::DataCopy<DT, ND, cfg>(buf[oi * inner], gmArr[groupIdx][elemBase + elemAdj], params);
        }
    }
}

// ===========================================================================
// CopyOutOne: DataCopyPad 搬出（adam CopyOutOne 行 399–410 同构; 单输出 y）
// blockLen = cnt × sizeof(T) 字节、不要求 32B 对齐、只写有效字节（尾块 / 小
//   tensor 覆盖）；单位契约: off 为元素数（GlobalTensor::operator[] 元素索引），
//   blockLen 为字节（cnt × sizeof(T)）
// ===========================================================================
template <typename T, int64_t RANK>
__aicore__ inline void BN3DTrainingReduceGradKernel<T, RANK>::CopyOutOne(const int64_t* coord, int slot, int64_t aISeg)
{
    int64_t off = CalcOutputOffset(coord, td_->outputStrides[0], RANK); // 元素索引
    int64_t cnt = CalcOutputTransferCount(td_->outputShapes[0], RANK, td_->split.axis, aISeg);
    AscendC::DataCopyExtParams extParams;
    extParams.blockCount = 1;
    extParams.blockLen = cnt * sizeof(T); // 有效字节, 不要求 32B 对齐, 只写 blockLen 字节
    extParams.srcStride = 0;
    extParams.dstStride = 0;
    AscendC::DataCopyPad(gmOut_[0][off], buf_[slot].template Get<T>(), extParams);
}
