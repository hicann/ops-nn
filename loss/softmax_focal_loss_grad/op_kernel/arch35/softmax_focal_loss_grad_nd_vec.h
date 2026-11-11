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
 * \file softmax_focal_loss_grad_nd.h
 * \brief softmax_focal_loss_grad regbase kernel
 *
 * 计算语义(对齐 A2 tbe.dsl softmax_focal_loss_grad_compute):
 *   wf = alpha * exp(gamma       * log(1-p)) * t     WF = sum_j wf
 *   wb = alpha * exp((gamma-1)   * log(1-p)) * t
 *   wd = wf - wb = -alpha * exp((gamma-1) * log(1-p)) * t * p    WD = sum_j wd
 *   ce = -log(p) * t * w                             CE = sum_j ce
 *   wt = w * t                                       W  = sum_j wt
 *   d_ce = p * W - t * w
 *   d_wf = -gamma * (WD + wb) * p
 *   grad = (d_ce * WF + d_wf * CE) * dout * coef     coef: mean 为 1/numel, 否则 1.0
 *
 * 调度: 行分核, 核内按行块推进, 行内按列块推进(列块数为 1 时即全载路径)。
 * 两趟: 第一趟求四个行标量, 第二趟逐元素组合出 grad(wb 在第二趟重算, 不占额外缓冲)。
 */

#ifndef SOFTMAX_FOCAL_LOSS_GRAD_ND_VEC_H
#define SOFTMAX_FOCAL_LOSS_GRAD_ND_VEC_H

#include "softmax_focal_loss_grad_nd.h"

namespace SoftmaxFocalLossGrad {

// High/low FP32 pairs retain row-sum residuals through the FP16 output stage.
__aicore__ inline void AddFloatPair(RegTensor<float>& hi, RegTensor<float>& lo, RegTensor<float>& ah,
                                    RegTensor<float>& al, RegTensor<float>& bh, RegTensor<float>& bl, MaskReg preg)
{
    RegTensor<float> sum, back, error, temp, zero;
    AscendC::Reg::Add(sum, ah, bh, preg);
    AscendC::Reg::Sub(back, sum, ah, preg);
    AscendC::Reg::Sub(temp, sum, back, preg);
    AscendC::Reg::Sub(error, ah, temp, preg);
    AscendC::Reg::Sub(temp, bh, back, preg);
    AscendC::Reg::Add(error, error, temp, preg);
    AscendC::Reg::Add(error, error, al, preg);
    AscendC::Reg::Add(error, error, bl, preg);
    MaskReg valid;
    AscendC::Reg::Duplicate(zero, 0.0f);
    AscendC::Reg::Compare<float, AscendC::CMPMODE::EQ>(valid, error, error, preg);
    AscendC::Reg::Select(error, error, zero, valid);
    AscendC::Reg::Add(hi, sum, error, preg);
    AscendC::Reg::Sub(temp, hi, sum, preg);
    AscendC::Reg::Sub(lo, error, temp, preg);
}

__aicore__ inline void AccumulateRowPair(RegTensor<float>& hi, RegTensor<float>& lo, RegTensor<float>& value,
                                         RegTensor<float>& zeroReg, MaskReg mask)
{
    RegTensor<float> sum, back, temp, err;
    MaskReg pregFinite;
    AscendC::Reg::Add(sum, hi, value, mask);
    AscendC::Reg::Sub(back, sum, hi, mask);
    AscendC::Reg::Sub(temp, sum, back, mask);
    AscendC::Reg::Sub(temp, hi, temp, mask);
    AscendC::Reg::Sub(err, value, back, mask);
    AscendC::Reg::Add(err, temp, err, mask);
    AscendC::Reg::Add(err, err, lo, mask);
    AscendC::Reg::Compare<float, AscendC::CMPMODE::EQ>(pregFinite, err, err, mask);
    AscendC::Reg::Select(err, err, zeroReg, pregFinite);
    AscendC::Reg::Add(hi, sum, err, mask);
    AscendC::Reg::Sub(back, hi, sum, mask);
    AscendC::Reg::Sub(temp, hi, back, mask);
    AscendC::Reg::Sub(temp, sum, temp, mask);
    AscendC::Reg::Sub(lo, err, back, mask);
    AscendC::Reg::Add(lo, temp, lo, mask);
    AscendC::Reg::Compare<float, AscendC::CMPMODE::EQ>(pregFinite, lo, lo, mask);
    AscendC::Reg::Select(lo, lo, zeroReg, pregFinite);
}

template <bool rightLowIsZero = false>
__aicore__ inline void MultiplyFloatPair(RegTensor<float>& hi, RegTensor<float>& lo, RegTensor<float>& ah,
                                         RegTensor<float>& al, RegTensor<float>& bh, RegTensor<float>& bl, MaskReg preg)
{
    RegTensor<float> product, error, temp, cross, zero;
    AscendC::Reg::Mul(product, ah, bh, preg);
    AscendC::Reg::Muls(temp, product, -1.0f, preg);
    AscendC::Reg::Muls(error, ah, 1.0f, preg);
    AscendC::Reg::MulDstAdd(error, bh, temp, preg);
    if constexpr (!rightLowIsZero) {
        AscendC::Reg::Mul(cross, ah, bl, preg);
        AscendC::Reg::Add(error, error, cross, preg);
    }
    AscendC::Reg::Mul(cross, al, bh, preg);
    AscendC::Reg::Add(error, error, cross, preg);
    MaskReg valid;
    AscendC::Reg::Duplicate(zero, 0.0f);
    AscendC::Reg::Compare<float, AscendC::CMPMODE::EQ>(valid, error, error, preg);
    AscendC::Reg::Select(error, error, zero, valid);
    AscendC::Reg::Add(hi, product, error, preg);
    AscendC::Reg::Sub(temp, hi, product, preg);
    AscendC::Reg::Sub(lo, error, temp, preg);
}

// CastTrait 必须定义在命名空间/文件作用域(Cast 的模板参数是 const 引用)
constexpr static AscendC::Reg::CastTrait CAST_TRAIT_S32_TO_FP32_LOCAL = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
    RoundMode::CAST_NONE};
constexpr static AscendC::Reg::CastTrait CAST_TRAIT_FP32_TO_S32_FLOOR_LOCAL = {
    AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
    RoundMode::CAST_FLOOR};

template <typename T, typename TW, uint64_t hasWeight, uint64_t gammaMode>
__aicore__ inline void SoftmaxFocalLossGradND<T, TW, hasWeight, gammaMode>::LoadPTW(
    RegTensor<float>& p32, RegTensor<float>& t32, RegTensor<float>& w32, __ubuf__ T* predAddr,
    __ubuf__ int32_t* targetAddr, __ubuf__ TW* weightAddr, AscendC::Reg::AddrReg offT, AscendC::Reg::AddrReg offW,
    AscendC::Reg::AddrReg offF, MaskReg preg)
{
    RegTensor<T> predReg;
    RegTensor<int32_t> targetRegI32;
    RegTensor<TW> weightReg;

    if constexpr (sizeof(T) == sizeof(half)) {
        AscendC::Reg::LoadAlign<T, LoadDist::DIST_UNPACK_B16>(predReg, predAddr, offT);
        AscendC::Reg::Cast<float, T, castB16ToB32>(p32, predReg, preg);
    } else {
        AscendC::Reg::LoadAlign(p32, predAddr, offT);
    }
    AscendC::Reg::LoadAlign(targetRegI32, targetAddr, offF);
    AscendC::Reg::Cast<float, int32_t, castI32ToF32>(t32, targetRegI32, preg);
    if constexpr (hasWeight == 1) {
        if constexpr (sizeof(TW) == sizeof(half)) {
            AscendC::Reg::LoadAlign<TW, LoadDist::DIST_UNPACK_B16>(weightReg, weightAddr, offW);
            AscendC::Reg::Cast<float, TW, castB16ToB32>(w32, weightReg, preg);
        } else {
            AscendC::Reg::LoadAlign(w32, weightAddr, offW);
        }
    } else {
        AscendC::Reg::Duplicate(w32, 1.0f);
    }
}

template <typename T, typename TW, uint64_t hasWeight, uint64_t gammaMode>
__aicore__ inline void SoftmaxFocalLossGradND<T, TW, hasWeight, gammaMode>::SumsOfSeg(
    __ubuf__ float* wfAddr, __ubuf__ float* wdAddr, __ubuf__ float* ceAddr, __ubuf__ float* wtAddr,
    RegTensor<float>& p32, RegTensor<float>& t32, RegTensor<float>& w32, AscendC::Reg::AddrReg offF, MaskReg preg,
    MaskReg storePreg)
{
    RegTensor<float> oneReg;
    RegTensor<float> l1p;
    RegTensor<float> tmpReg;
    RegTensor<float> outReg;
    RegTensor<float> zeroReg;
    AscendC::Reg::Duplicate(zeroReg, 0.0f);
    float gamma = gamma_;
    float gammaSub1 = gamma_ - 1.0f;
    float alpha = alpha_;

    // l1p = log(1 - p)
    AscendC::Reg::Duplicate(oneReg, 1.0f);
    AscendC::Reg::Sub(tmpReg, oneReg, p32, preg);
    if constexpr (gammaMode == 0) {
        AscendC::Reg::Log(l1p, tmpReg, preg);
    }

    // wf = alpha * exp(gamma * l1p) * t
    if constexpr (gammaMode == 1) {
        AscendC::Reg::Muls(outReg, tmpReg, 1.0f, preg);
    } else if constexpr (gammaMode == 2) {
        AscendC::Reg::Mul(outReg, tmpReg, tmpReg, preg);
    } else {
        AscendC::Reg::Muls(tmpReg, l1p, gamma, preg);
        AscendC::Reg::Exp(outReg, tmpReg, preg);
    }
    AscendC::Reg::Muls(outReg, outReg, alpha, preg);
    AscendC::Reg::Mul(outReg, outReg, t32, preg);
    AscendC::Reg::Select(outReg, outReg, zeroReg, preg);
    AscendC::Reg::StoreAlign(wfAddr, outReg, offF, storePreg);

    // wd = wf - wb = -alpha * exp((gamma - 1) * l1p) * t * p
    // 直接产差值而非分别产 wf/wb 再相减: gamma 较大时 (1-p)^gamma 与 (1-p)^(gamma-1) 量级接近,
    // 先求和再相减会抵消掉有效位()。
    if constexpr (gammaMode == 1) {
        AscendC::Reg::Duplicate(outReg, 1.0f);
    } else if constexpr (gammaMode == 2) {
        AscendC::Reg::Sub(outReg, oneReg, p32, preg);
    } else {
        AscendC::Reg::Muls(tmpReg, l1p, gammaSub1, preg);
        AscendC::Reg::Exp(outReg, tmpReg, preg);
    }
    AscendC::Reg::Muls(outReg, outReg, -alpha, preg);
    AscendC::Reg::Mul(outReg, outReg, t32, preg);
    AscendC::Reg::Mul(outReg, outReg, p32, preg);
    AscendC::Reg::Select(outReg, outReg, zeroReg, preg);
    AscendC::Reg::StoreAlign(wdAddr, outReg, offF, storePreg);

    // ce = -log(p) * t * w
    AscendC::Reg::Log(tmpReg, p32, preg);
    AscendC::Reg::Muls(outReg, tmpReg, -1.0f, preg);
    AscendC::Reg::Mul(outReg, outReg, t32, preg);
    AscendC::Reg::Mul(outReg, outReg, w32, preg);
    AscendC::Reg::Select(outReg, outReg, zeroReg, preg);
    AscendC::Reg::StoreAlign(ceAddr, outReg, offF, storePreg);

    // wt = w * t
    AscendC::Reg::Mul(outReg, w32, t32, preg);
    AscendC::Reg::Select(outReg, outReg, zeroReg, preg);
    AscendC::Reg::StoreAlign(wtAddr, outReg, offF, storePreg);
}

template <typename T, typename TW, uint64_t hasWeight, uint64_t gammaMode>
__aicore__ inline void SoftmaxFocalLossGradND<T, TW, hasWeight, gammaMode>::ExactRowSum4(
    __ubuf__ float* src0, __ubuf__ float* src1, __ubuf__ float* src2, __ubuf__ float* src3, __ubuf__ float* accBase,
    int64_t rows, int64_t len)
{
    LocalTensor<float> dfLocal = dfBuf_.template Get<float>();
    // 行槽布局: 4 个累加量的 hi 槽连续排列, 之后是 4 个 lo 槽, 每个累加量占 VL_FP32 * ROW_SLOT_ELEM 个 float
    constexpr int64_t slotSpan = static_cast<int64_t>(VL_FP32) * static_cast<int64_t>(ROW_SLOT_ELEM);
    __ubuf__ float* rowHi = (__ubuf__ float*)dfLocal.GetPhyAddr();
    __ubuf__ float* rowLo = rowHi + 4 * slotSpan;

    int64_t strideF = StrideOfF32(len);
    uint32_t vfLen = VL_FP32;
    uint16_t repeatTimes = static_cast<uint16_t>(len / vfLen);
    uint32_t tailNum = static_cast<uint32_t>(len % vfLen);
    uint16_t tailLoop = tailNum != 0 ? 1 : 0;

    // 四链交错只用于精确路径: 其余档位按行寻址的行距可能小于一个 32B 块(紧致布局 strideF == len < 8),
    // 逐行加载地址非 32B 对齐 -> 非法, 必须走各自的小 D 分支。
    if (!ExactPath()) {
        ExactRowSum(src0, accBase, accBase + 4 * accStride_, rows, len);
        ExactRowSum(src1, accBase + accStride_, accBase + 5 * accStride_, rows, len);
        ExactRowSum(src2, accBase + 2 * accStride_, accBase + 6 * accStride_, rows, len);
        ExactRowSum(src3, accBase + 3 * accStride_, accBase + 7 * accStride_, rows, len);
        return;
    }

    for (int64_t g = 0; g < rows; g += static_cast<int64_t>(VL_FP32)) {
        int64_t left = rows - g;
        uint16_t cnt = static_cast<uint16_t>(left < static_cast<int64_t>(VL_FP32) ? left :
                                                                                    static_cast<int64_t>(VL_FP32));
        uint32_t cntMask = static_cast<uint32_t>(cnt);
        uint32_t oneCnt = 1U;
        __ubuf__ float* grpSrc0 = src0 + g * strideF;
        __ubuf__ float* grpSrc1 = src1 + g * strideF;
        __ubuf__ float* grpSrc2 = src2 + g * strideF;
        __ubuf__ float* grpSrc3 = src3 + g * strideF;

        __VEC_SCOPE__
        {
            RegTensor<float> hi0;
            RegTensor<float> lo0;
            RegTensor<float> rR0;
            RegTensor<float> vR0;
            RegTensor<float> hi1;
            RegTensor<float> lo1;
            RegTensor<float> rR1;
            RegTensor<float> vR1;
            RegTensor<float> hi2;
            RegTensor<float> lo2;
            RegTensor<float> rR2;
            RegTensor<float> vR2;
            RegTensor<float> hi3;
            RegTensor<float> lo3;
            RegTensor<float> rR3;
            RegTensor<float> vR3;
            RegTensor<float> zeroReg;
            MaskReg pregOne = AscendC::Reg::UpdateMask<float>(oneCnt);
            MaskReg pregTail = AscendC::Reg::UpdateMask<float>(tailNum);
            MaskReg pregMain = AscendC::Reg::CreateMask<float, MaskPattern::ALL>();

            AscendC::Reg::Duplicate(zeroReg, 0.0f);
            for (uint16_t i = 0; i < cnt; ++i) {
                AscendC::Reg::Duplicate(hi0, 0.0f);
                AscendC::Reg::Duplicate(lo0, 0.0f);
                AscendC::Reg::Duplicate(hi1, 0.0f);
                AscendC::Reg::Duplicate(lo1, 0.0f);
                AscendC::Reg::Duplicate(hi2, 0.0f);
                AscendC::Reg::Duplicate(lo2, 0.0f);
                AscendC::Reg::Duplicate(hi3, 0.0f);
                AscendC::Reg::Duplicate(lo3, 0.0f);
                for (uint16_t j = 0; j < repeatTimes + tailLoop; ++j) {
                    MaskReg preg = (j < repeatTimes) ? pregMain : pregTail;
                    AscendC::Reg::AddrReg offF = AscendC::Reg::CreateAddrReg<float>(i, static_cast<uint32_t>(strideF),
                                                                                    j, vfLen);
                    AscendC::Reg::LoadAlign(vR0, grpSrc0, offF);
                    AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM>(rR0, vR0, preg);
                    AscendC::Reg::LoadAlign(vR1, grpSrc1, offF);
                    AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM>(rR1, vR1, preg);
                    AscendC::Reg::LoadAlign(vR2, grpSrc2, offF);
                    AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM>(rR2, vR2, preg);
                    AscendC::Reg::LoadAlign(vR3, grpSrc3, offF);
                    AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM>(rR3, vR3, preg);
                    AccumulateRowPair(hi0, lo0, rR0, zeroReg, pregMain);
                    AccumulateRowPair(hi1, lo1, rR1, zeroReg, pregMain);
                    AccumulateRowPair(hi2, lo2, rR2, zeroReg, pregMain);
                    AccumulateRowPair(hi3, lo3, rR3, zeroReg, pregMain);
                }
                AscendC::Reg::AddrReg offRow = AscendC::Reg::CreateAddrReg<float>(i, ROW_SLOT_ELEM);
                AscendC::Reg::StoreAlign(rowHi + 0 * slotSpan, hi0, offRow, pregOne);
                AscendC::Reg::StoreAlign(rowLo + 0 * slotSpan, lo0, offRow, pregOne);
                AscendC::Reg::StoreAlign(rowHi + 1 * slotSpan, hi1, offRow, pregOne);
                AscendC::Reg::StoreAlign(rowLo + 1 * slotSpan, lo1, offRow, pregOne);
                AscendC::Reg::StoreAlign(rowHi + 2 * slotSpan, hi2, offRow, pregOne);
                AscendC::Reg::StoreAlign(rowLo + 2 * slotSpan, lo2, offRow, pregOne);
                AscendC::Reg::StoreAlign(rowHi + 3 * slotSpan, hi3, offRow, pregOne);
                AscendC::Reg::StoreAlign(rowLo + 3 * slotSpan, lo3, offRow, pregOne);
            }
        }
        AscendC::PipeBarrier<PIPE_V>();
        MergeRowSlots(accBase, rowHi, rowLo, slotSpan, g, cntMask);
    }
}

template <typename T, typename TW, uint64_t hasWeight, uint64_t gammaMode>
__aicore__ inline void SoftmaxFocalLossGradND<T, TW, hasWeight, gammaMode>::MergeRowSlots(__ubuf__ float* accBase,
                                                                                          __ubuf__ float* rowHi,
                                                                                          __ubuf__ float* rowLo,
                                                                                          int64_t slotSpan, int64_t g,
                                                                                          uint32_t cntMask)
{
    __VEC_SCOPE__
    {
        RegTensor<float> rhReg;
        RegTensor<float> rlReg;
        RegTensor<float> ahReg;
        RegTensor<float> alReg;
        RegTensor<float> sR0;
        RegTensor<float> eR0;
        RegTensor<float> bR0;
        RegTensor<float> tR0;
        RegTensor<float> zeroReg;
        MaskReg pregFinite;
        MaskReg pregCnt = AscendC::Reg::UpdateMask<float>(cntMask);

        AscendC::Reg::Duplicate(zeroReg, 0.0f);
        AscendC::Reg::RegTensor<int32_t> idxReg;
        AscendC::Reg::Arange(idxReg, 0);
        AscendC::Reg::Muls(idxReg, idxReg, static_cast<int32_t>(ROW_SLOT_ELEM), pregCnt);
        for (uint16_t slot = 0; slot < 4; ++slot) {
            AscendC::Reg::Gather(rhReg, rowHi + slot * slotSpan, (AscendC::Reg::RegTensor<uint32_t>&)idxReg, pregCnt);
            AscendC::Reg::Gather(rlReg, rowLo + slot * slotSpan, (AscendC::Reg::RegTensor<uint32_t>&)idxReg, pregCnt);
            AscendC::Reg::LoadAlign(ahReg, accBase + slot * accStride_ + g);
            AscendC::Reg::LoadAlign(alReg, accBase + (4 + slot) * accStride_ + g);
            AscendC::Reg::Add(sR0, ahReg, rhReg, pregCnt);
            AscendC::Reg::Sub(bR0, sR0, ahReg, pregCnt);
            AscendC::Reg::Sub(tR0, sR0, bR0, pregCnt);
            AscendC::Reg::Sub(tR0, ahReg, tR0, pregCnt);
            AscendC::Reg::Sub(eR0, rhReg, bR0, pregCnt);
            AscendC::Reg::Add(eR0, tR0, eR0, pregCnt);
            AscendC::Reg::Add(eR0, eR0, alReg, pregCnt);
            AscendC::Reg::Compare<float, AscendC::CMPMODE::EQ>(pregFinite, eR0, eR0, pregCnt);
            AscendC::Reg::Select(eR0, eR0, zeroReg, pregFinite);
            AscendC::Reg::Add(ahReg, sR0, eR0, pregCnt);
            AscendC::Reg::Sub(bR0, ahReg, sR0, pregCnt);
            AscendC::Reg::Sub(tR0, ahReg, bR0, pregCnt);
            AscendC::Reg::Sub(tR0, sR0, tR0, pregCnt);
            AscendC::Reg::Sub(alReg, eR0, bR0, pregCnt);
            AscendC::Reg::Add(alReg, tR0, alReg, pregCnt);
            AscendC::Reg::Compare<float, AscendC::CMPMODE::EQ>(pregFinite, alReg, alReg, pregCnt);
            AscendC::Reg::Select(alReg, alReg, zeroReg, pregFinite);
            AscendC::Reg::Add(sR0, ahReg, rlReg, pregCnt);
            AscendC::Reg::Sub(bR0, sR0, ahReg, pregCnt);
            AscendC::Reg::Sub(tR0, sR0, bR0, pregCnt);
            AscendC::Reg::Sub(tR0, ahReg, tR0, pregCnt);
            AscendC::Reg::Sub(eR0, rlReg, bR0, pregCnt);
            AscendC::Reg::Add(eR0, tR0, eR0, pregCnt);
            AscendC::Reg::Add(eR0, eR0, alReg, pregCnt);
            AscendC::Reg::Compare<float, AscendC::CMPMODE::EQ>(pregFinite, eR0, eR0, pregCnt);
            AscendC::Reg::Select(eR0, eR0, zeroReg, pregFinite);
            AscendC::Reg::Add(ahReg, sR0, eR0, pregCnt);
            AscendC::Reg::Sub(bR0, ahReg, sR0, pregCnt);
            AscendC::Reg::Sub(tR0, ahReg, bR0, pregCnt);
            AscendC::Reg::Sub(tR0, sR0, tR0, pregCnt);
            AscendC::Reg::Sub(alReg, eR0, bR0, pregCnt);
            AscendC::Reg::Add(alReg, tR0, alReg, pregCnt);
            AscendC::Reg::Compare<float, AscendC::CMPMODE::EQ>(pregFinite, alReg, alReg, pregCnt);
            AscendC::Reg::Select(alReg, alReg, zeroReg, pregFinite);
            AscendC::Reg::StoreAlign(accBase + slot * accStride_ + g, ahReg, pregCnt);
            AscendC::Reg::StoreAlign(accBase + (4 + slot) * accStride_ + g, alReg, pregCnt);
        }
    }
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename T, typename TW, uint64_t hasWeight, uint64_t gammaMode>
__aicore__ inline void SoftmaxFocalLossGradND<T, TW, hasWeight, gammaMode>::ExactRowSum(__ubuf__ float* src,
                                                                                        __ubuf__ float* accHi,
                                                                                        __ubuf__ float* accLo,
                                                                                        int64_t rows, int64_t len)
{
    LocalTensor<float> dfLocal = dfBuf_.template Get<float>();
    // 行槽按 ROW_SLOT_ELEM 对齐摆放: StoreAlign 要求 32B 对齐, 单元素偏移是非法地址
    __ubuf__ float* rowHi = (__ubuf__ float*)dfLocal.GetPhyAddr();
    __ubuf__ float* rowLo = rowHi + VL_FP32 * ROW_SLOT_ELEM;

    int64_t strideF = StrideOfF32(len);
    uint32_t vfLen = VL_FP32;
    uint16_t repeatTimes = static_cast<uint16_t>(len / vfLen);
    uint32_t tailNum = static_cast<uint32_t>(len % vfLen);
    uint16_t tailLoop = tailNum != 0 ? 1 : 0;

    // 紧致布局(行距 == len, len 为 2 的幂且 < 一个 fp32 块): 一次 64 车道覆盖 64/len 行,
    // log2(len) 级 PairReduceElem 把相邻元素两两并和 —— 2 元素求和无次序自由度, 4/8 元素是平衡树,
    // 与块内硬件树形同级, 不增加舍入层数。行和打包在前 64/len 个车道, 组首偏移 64/len 个 float(32B 对齐)。
    if (strideF == len && len < static_cast<int64_t>(ROW_SLOT_ELEM) && (len & (len - 1)) == 0) {
        int64_t rowsPerVec = static_cast<int64_t>(VL_FP32) / len;
        int64_t levels = 0;
        for (int64_t q = len; q > 1; q >>= 1) {
            ++levels;
        }
        for (int64_t g = 0; g < rows; g += rowsPerVec) {
            int64_t left = rows - g;
            int64_t rowCnt = left < rowsPerVec ? left : rowsPerVec;
            uint32_t dataCnt = static_cast<uint32_t>(rowCnt * len);
            uint32_t rowMask = static_cast<uint32_t>(rowCnt);
            uint16_t lvl = static_cast<uint16_t>(levels);
            __ubuf__ float* grpSrc = src + g * len;
            __ubuf__ float* grpAccHi = accHi + g;
            __ubuf__ float* grpAccLo = accLo + g;
            __VEC_SCOPE__
            {
                RegTensor<float> vReg;
                RegTensor<float> rReg;
                RegTensor<float> ahReg;
                RegTensor<float> alReg;
                RegTensor<float> sReg;
                RegTensor<float> eReg;
                RegTensor<float> bReg;
                RegTensor<float> tReg;
                RegTensor<float> zeroReg;
                MaskReg pregFinite;
                MaskReg pregData = AscendC::Reg::UpdateMask<float>(dataCnt);
                MaskReg pregRow = AscendC::Reg::UpdateMask<float>(rowMask);
                MaskReg pregAll = AscendC::Reg::CreateMask<float, MaskPattern::ALL>();

                AscendC::Reg::Duplicate(zeroReg, 0.0f);
                AscendC::Reg::LoadAlign(vReg, grpSrc);
                // 尾组: 先把越界车道置零再用满掩码运算(掩码进长链会丢尾块贡献)
                AscendC::Reg::Select(vReg, vReg, zeroReg, pregData);
                AscendC::Reg::Duplicate(rReg, 0.0f);
                AscendC::Reg::Add(rReg, rReg, vReg, pregAll);
                for (uint16_t q = 0; q < lvl; ++q) {
                    AscendC::Reg::PairReduceElem<AscendC::Reg::PairReduce::SUM>(rReg, rReg, pregAll);
                }
                AscendC::Reg::LoadAlign(ahReg, grpAccHi);
                AscendC::Reg::LoadAlign(alReg, grpAccLo);
                AscendC::Reg::Add(sReg, ahReg, rReg, pregRow);
                AscendC::Reg::Sub(bReg, sReg, ahReg, pregRow);
                AscendC::Reg::Sub(tReg, sReg, bReg, pregRow);
                AscendC::Reg::Sub(tReg, ahReg, tReg, pregRow);
                AscendC::Reg::Sub(eReg, rReg, bReg, pregRow);
                AscendC::Reg::Add(eReg, tReg, eReg, pregRow);
                AscendC::Reg::Add(eReg, eReg, alReg, pregRow);
                AscendC::Reg::Compare<float, AscendC::CMPMODE::EQ>(pregFinite, eReg, eReg, pregRow);
                AscendC::Reg::Select(eReg, eReg, zeroReg, pregFinite);
                AscendC::Reg::Add(ahReg, sReg, eReg, pregRow);
                AscendC::Reg::Sub(bReg, ahReg, sReg, pregRow);
                AscendC::Reg::Sub(tReg, ahReg, bReg, pregRow);
                AscendC::Reg::Sub(tReg, sReg, tReg, pregRow);
                AscendC::Reg::Sub(alReg, eReg, bReg, pregRow);
                AscendC::Reg::Add(alReg, tReg, alReg, pregRow);
                AscendC::Reg::Compare<float, AscendC::CMPMODE::EQ>(pregFinite, alReg, alReg, pregRow);
                AscendC::Reg::Select(alReg, alReg, zeroReg, pregFinite);
                AscendC::Reg::StoreAlign(grpAccHi, ahReg, pregRow);
                AscendC::Reg::StoreAlign(grpAccLo, alReg, pregRow);
            }
        }
        AscendC::PipeBarrier<PIPE_V>();
        return;
    }
    // 一行占整数个 32B datablock 时, ReduceDataBlock 一条指令就给出 8 个块和。
    //  - 一行一块(strideF == 8): 连续加载 64 车道 = 8 行, 车道 k 即第 k 行的和;
    //  - 一行两块(strideF == 16): 按块跨步加载分别取齐每行的前/后半块, 逐元素相加后仍是
    //    "车道 k = 第 k 行的和"; 两个 8 元素块树和再相加等价 16 元素平衡树, 不增加舍入层数。
    // 两种情形都是 8 行一组, 组首在 accHi/accLo 上的偏移恒为 8 float(32B 对齐), StoreAlign 合法。
    // 两条路径各自成环: 分支不进循环体, 以免打断矢量流水。
    // 一行最多两块走本路径: 每多一块多一次跨步加载与一次加法, 再多则不如组路径
    // (组路径每行只发一条硬件 Reduce)。
    constexpr int64_t maxBlkPerRowSlot = 2;
    int64_t blkPerRowSlot = strideF / static_cast<int64_t>(ROW_SLOT_ELEM);
    if (strideF % static_cast<int64_t>(ROW_SLOT_ELEM) == 0 && blkPerRowSlot >= 1 && blkPerRowSlot <= maxBlkPerRowSlot &&
        rows >= static_cast<int64_t>(VL_FP32) / strideF) {
        constexpr int64_t rowsPerGrp = static_cast<int64_t>(ROW_SLOT_ELEM);
        uint32_t blkStride = static_cast<uint32_t>(blkPerRowSlot);
        if (blkPerRowSlot == 1) {
            for (int64_t g = 0; g < rows; g += rowsPerGrp) {
                int64_t left = rows - g;
                uint32_t cnt = static_cast<uint32_t>(left < rowsPerGrp ? left : rowsPerGrp);
                __ubuf__ float* grpSrc = src + g * strideF;
                __ubuf__ float* grpAccHi = accHi + g;
                __ubuf__ float* grpAccLo = accLo + g;
                __VEC_SCOPE__
                {
                    RegTensor<float> vReg;
                    RegTensor<float> rReg;
                    RegTensor<float> ahReg;
                    RegTensor<float> alReg;
                    RegTensor<float> sReg;
                    RegTensor<float> eReg;
                    RegTensor<float> bReg;
                    RegTensor<float> tReg;
                    RegTensor<float> zeroReg;
                    MaskReg pregFinite;
                    MaskReg pregRow = AscendC::Reg::UpdateMask<float>(cnt);
                    MaskReg pregAll = AscendC::Reg::CreateMask<float, MaskPattern::ALL>();

                    AscendC::Reg::Duplicate(zeroReg, 0.0f);
                    AscendC::Reg::LoadAlign(vReg, grpSrc);
                    AscendC::Reg::ReduceDataBlock<AscendC::Reg::ReduceType::SUM>(rReg, vReg, pregAll);
                    AscendC::Reg::LoadAlign(ahReg, grpAccHi);
                    AscendC::Reg::LoadAlign(alReg, grpAccLo);
                    AscendC::Reg::Add(sReg, ahReg, rReg, pregRow);
                    AscendC::Reg::Sub(bReg, sReg, ahReg, pregRow);
                    AscendC::Reg::Sub(tReg, sReg, bReg, pregRow);
                    AscendC::Reg::Sub(tReg, ahReg, tReg, pregRow);
                    AscendC::Reg::Sub(eReg, rReg, bReg, pregRow);
                    AscendC::Reg::Add(eReg, tReg, eReg, pregRow);
                    AscendC::Reg::Add(eReg, eReg, alReg, pregRow);
                    AscendC::Reg::Compare<float, AscendC::CMPMODE::EQ>(pregFinite, eReg, eReg, pregRow);
                    AscendC::Reg::Select(eReg, eReg, zeroReg, pregFinite);
                    AscendC::Reg::Add(ahReg, sReg, eReg, pregRow);
                    AscendC::Reg::Sub(bReg, ahReg, sReg, pregRow);
                    AscendC::Reg::Sub(tReg, ahReg, bReg, pregRow);
                    AscendC::Reg::Sub(tReg, sReg, tReg, pregRow);
                    AscendC::Reg::Sub(alReg, eReg, bReg, pregRow);
                    AscendC::Reg::Add(alReg, tReg, alReg, pregRow);
                    AscendC::Reg::Compare<float, AscendC::CMPMODE::EQ>(pregFinite, alReg, alReg, pregRow);
                    AscendC::Reg::Select(alReg, alReg, zeroReg, pregFinite);
                    AscendC::Reg::StoreAlign(grpAccHi, ahReg, pregRow);
                    AscendC::Reg::StoreAlign(grpAccLo, alReg, pregRow);
                }
            }
        } else {
            for (int64_t g = 0; g < rows; g += rowsPerGrp) {
                int64_t left = rows - g;
                uint32_t cnt = static_cast<uint32_t>(left < rowsPerGrp ? left : rowsPerGrp);
                __ubuf__ float* grpSrc = src + g * strideF;
                __ubuf__ float* grpAccHi = accHi + g;
                __ubuf__ float* grpAccLo = accLo + g;
                __VEC_SCOPE__
                {
                    RegTensor<float> vReg;
                    RegTensor<float> v1Reg;
                    RegTensor<float> rReg;
                    RegTensor<float> ahReg;
                    RegTensor<float> alReg;
                    RegTensor<float> sReg;
                    RegTensor<float> eReg;
                    RegTensor<float> bReg;
                    RegTensor<float> tReg;
                    RegTensor<float> zeroReg;
                    MaskReg pregFinite;
                    MaskReg pregRow = AscendC::Reg::UpdateMask<float>(cnt);
                    MaskReg pregAll = AscendC::Reg::CreateMask<float, MaskPattern::ALL>();

                    AscendC::Reg::Duplicate(zeroReg, 0.0f);
                    AscendC::Reg::LoadAlign<float, AscendC::Reg::DataCopyMode::DATA_BLOCK_COPY>(vReg, grpSrc, blkStride,
                                                                                                pregAll);
                    AscendC::Reg::LoadAlign<float, AscendC::Reg::DataCopyMode::DATA_BLOCK_COPY>(
                        v1Reg, grpSrc + static_cast<int64_t>(ROW_SLOT_ELEM), blkStride, pregAll);
                    AscendC::Reg::Add(vReg, vReg, v1Reg, pregAll);
                    AscendC::Reg::ReduceDataBlock<AscendC::Reg::ReduceType::SUM>(rReg, vReg, pregAll);
                    AscendC::Reg::LoadAlign(ahReg, grpAccHi);
                    AscendC::Reg::LoadAlign(alReg, grpAccLo);
                    AscendC::Reg::Add(sReg, ahReg, rReg, pregRow);
                    AscendC::Reg::Sub(bReg, sReg, ahReg, pregRow);
                    AscendC::Reg::Sub(tReg, sReg, bReg, pregRow);
                    AscendC::Reg::Sub(tReg, ahReg, tReg, pregRow);
                    AscendC::Reg::Sub(eReg, rReg, bReg, pregRow);
                    AscendC::Reg::Add(eReg, tReg, eReg, pregRow);
                    AscendC::Reg::Add(eReg, eReg, alReg, pregRow);
                    AscendC::Reg::Compare<float, AscendC::CMPMODE::EQ>(pregFinite, eReg, eReg, pregRow);
                    AscendC::Reg::Select(eReg, eReg, zeroReg, pregFinite);
                    AscendC::Reg::Add(ahReg, sReg, eReg, pregRow);
                    AscendC::Reg::Sub(bReg, ahReg, sReg, pregRow);
                    AscendC::Reg::Sub(tReg, ahReg, bReg, pregRow);
                    AscendC::Reg::Sub(tReg, sReg, tReg, pregRow);
                    AscendC::Reg::Sub(alReg, eReg, bReg, pregRow);
                    AscendC::Reg::Add(alReg, tReg, alReg, pregRow);
                    AscendC::Reg::Compare<float, AscendC::CMPMODE::EQ>(pregFinite, alReg, alReg, pregRow);
                    AscendC::Reg::Select(alReg, alReg, zeroReg, pregFinite);
                    AscendC::Reg::StoreAlign(grpAccHi, ahReg, pregRow);
                    AscendC::Reg::StoreAlign(grpAccLo, alReg, pregRow);
                }
            }
        }
        AscendC::PipeBarrier<PIPE_V>();
        return;
    }
    // 按 VL 行分组: 组内每行的行和写进 rowHi/rowLo 的第 i 槽(单车道写), 组尾再用一次
    // 矢量运算把整组行一并并回 accHi/accLo。跨 chunk 合并不再逐行走矢量->标量传输。
    for (int64_t g = 0; g < rows; g += static_cast<int64_t>(VL_FP32)) {
        int64_t left = rows - g;
        uint16_t cnt = static_cast<uint16_t>(left < static_cast<int64_t>(VL_FP32) ? left :
                                                                                    static_cast<int64_t>(VL_FP32));
        uint32_t cntMask = static_cast<uint32_t>(cnt);
        uint32_t oneCnt = 1U; // UpdateMask 取左值
        __ubuf__ float* grpSrc = src + g * strideF;
        __ubuf__ float* grpAccHi = accHi + g;
        __ubuf__ float* grpAccLo = accLo + g;

        __VEC_SCOPE__
        {
            RegTensor<float> hiReg;
            RegTensor<float> loReg;
            RegTensor<float> vReg;
            RegTensor<float> rReg;
            RegTensor<float> sReg;
            RegTensor<float> eReg;
            RegTensor<float> bReg;
            RegTensor<float> tReg;
            RegTensor<float> zeroReg;
            MaskReg pregFinite;

            // UpdateMask 排在 CreateMask(ALL) 之前取(否则 pregMain 会被改成尾块掩码)
            MaskReg pregOne = UpdateMask<float>(oneCnt);
            MaskReg pregTail = UpdateMask<float>(tailNum);
            MaskReg pregMain = AscendC::Reg::CreateMask<float, MaskPattern::ALL>();

            for (uint16_t i = 0; i < cnt; ++i) {
                AscendC::Reg::Duplicate(hiReg, 0.0f);
                AscendC::Reg::Duplicate(loReg, 0.0f);
                AscendC::Reg::Duplicate(zeroReg, 0.0f);
                for (uint16_t j = 0; j < repeatTimes + tailLoop; ++j) {
                    MaskReg preg = (j < repeatTimes) ? pregMain : pregTail;
                    AscendC::Reg::AddrReg offF = AscendC::Reg::CreateAddrReg<float>(i, static_cast<uint32_t>(strideF),
                                                                                    j, vfLen);
                    AscendC::Reg::LoadAlign(vReg, grpSrc, offF);
                    // 块内走硬件树形规约(本身即最优序), 块间才补偿
                    AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM>(rReg, vReg, preg);
                    AscendC::Reg::Add(sReg, hiReg, rReg, pregMain);
                    AscendC::Reg::Sub(bReg, sReg, hiReg, pregMain);
                    AscendC::Reg::Sub(tReg, sReg, bReg, pregMain);
                    AscendC::Reg::Sub(tReg, hiReg, tReg, pregMain);
                    AscendC::Reg::Sub(eReg, rReg, bReg, pregMain);
                    AscendC::Reg::Add(eReg, tReg, eReg, pregMain);
                    AscendC::Reg::Add(eReg, eReg, loReg, pregMain);
                    AscendC::Reg::Compare<float, AscendC::CMPMODE::EQ>(pregFinite, eReg, eReg, pregMain);
                    AscendC::Reg::Select(eReg, eReg, zeroReg, pregFinite);
                    AscendC::Reg::Add(hiReg, sReg, eReg, pregMain);
                    AscendC::Reg::Sub(bReg, hiReg, sReg, pregMain);
                    AscendC::Reg::Sub(tReg, hiReg, bReg, pregMain);
                    AscendC::Reg::Sub(tReg, sReg, tReg, pregMain);
                    AscendC::Reg::Sub(loReg, eReg, bReg, pregMain);
                    AscendC::Reg::Add(loReg, tReg, loReg, pregMain);
                    AscendC::Reg::Compare<float, AscendC::CMPMODE::EQ>(pregFinite, loReg, loReg, pregMain);
                    AscendC::Reg::Select(loReg, loReg, zeroReg, pregFinite);
                }
                // 行和在 lane0: 单车道写回本组第 i 槽
                AscendC::Reg::AddrReg offRow = AscendC::Reg::CreateAddrReg<float>(i, ROW_SLOT_ELEM);
                AscendC::Reg::StoreAlign(rowHi, hiReg, offRow, pregOne);
                AscendC::Reg::StoreAlign(rowLo, loReg, offRow, pregOne);
            }
        }

        // 行槽在上一个作用域里写入, 这里要读回: 同一矢量流水上的写后读必须显式设屏障,
        // 否则是潜在竞态(偶发红)。
        AscendC::PipeBarrier<PIPE_V>();

        __VEC_SCOPE__
        {
            RegTensor<float> ahReg;
            RegTensor<float> alReg;
            RegTensor<float> rhReg;
            RegTensor<float> rlReg;
            RegTensor<float> vReg;
            RegTensor<float> sReg;
            RegTensor<float> eReg;
            RegTensor<float> bReg;
            RegTensor<float> tReg;
            RegTensor<float> zeroReg;
            MaskReg pregFinite;
            MaskReg pregCnt = UpdateMask<float>(cntMask);

            AscendC::Reg::Duplicate(zeroReg, 0.0f);
            // 行槽是 8 float 跨步的, 用一条 Gather 把整组行的 lane0 收进一个寄存器
            AscendC::Reg::RegTensor<int32_t> idxReg;
            AscendC::Reg::Arange(idxReg, 0);
            AscendC::Reg::Muls(idxReg, idxReg, static_cast<int32_t>(ROW_SLOT_ELEM), pregCnt);
            AscendC::Reg::Gather(rhReg, rowHi, (AscendC::Reg::RegTensor<uint32_t>&)idxReg, pregCnt);
            AscendC::Reg::Gather(rlReg, rowLo, (AscendC::Reg::RegTensor<uint32_t>&)idxReg, pregCnt);
            AscendC::Reg::LoadAlign(ahReg, grpAccHi);
            AscendC::Reg::LoadAlign(alReg, grpAccLo);
            // Kahan 的低位是待补量, 本 chunk 的行和 = rh - rl
            // 整对精确并入: 先 hi+lo 折成单值会把残差舍掉()
            AscendC::Reg::Add(sReg, ahReg, rhReg, pregCnt);
            AscendC::Reg::Sub(bReg, sReg, ahReg, pregCnt);
            AscendC::Reg::Sub(tReg, sReg, bReg, pregCnt);
            AscendC::Reg::Sub(tReg, ahReg, tReg, pregCnt);
            AscendC::Reg::Sub(eReg, rhReg, bReg, pregCnt);
            AscendC::Reg::Add(eReg, tReg, eReg, pregCnt);
            AscendC::Reg::Add(eReg, eReg, alReg, pregCnt);
            AscendC::Reg::Compare<float, AscendC::CMPMODE::EQ>(pregFinite, eReg, eReg, pregCnt);
            AscendC::Reg::Select(eReg, eReg, zeroReg, pregFinite);
            AscendC::Reg::Add(ahReg, sReg, eReg, pregCnt);
            AscendC::Reg::Sub(bReg, ahReg, sReg, pregCnt);
            AscendC::Reg::Sub(tReg, ahReg, bReg, pregCnt);
            AscendC::Reg::Sub(tReg, sReg, tReg, pregCnt);
            AscendC::Reg::Sub(alReg, eReg, bReg, pregCnt);
            AscendC::Reg::Add(alReg, tReg, alReg, pregCnt);
            AscendC::Reg::Compare<float, AscendC::CMPMODE::EQ>(pregFinite, alReg, alReg, pregCnt);
            AscendC::Reg::Select(alReg, alReg, zeroReg, pregFinite);
            AscendC::Reg::Add(sReg, ahReg, rlReg, pregCnt);
            AscendC::Reg::Sub(bReg, sReg, ahReg, pregCnt);
            AscendC::Reg::Sub(tReg, sReg, bReg, pregCnt);
            AscendC::Reg::Sub(tReg, ahReg, tReg, pregCnt);
            AscendC::Reg::Sub(eReg, rlReg, bReg, pregCnt);
            AscendC::Reg::Add(eReg, tReg, eReg, pregCnt);
            AscendC::Reg::Add(eReg, eReg, alReg, pregCnt);
            AscendC::Reg::Compare<float, AscendC::CMPMODE::EQ>(pregFinite, eReg, eReg, pregCnt);
            AscendC::Reg::Select(eReg, eReg, zeroReg, pregFinite);
            AscendC::Reg::Add(ahReg, sReg, eReg, pregCnt);
            AscendC::Reg::Sub(bReg, ahReg, sReg, pregCnt);
            AscendC::Reg::Sub(tReg, ahReg, bReg, pregCnt);
            AscendC::Reg::Sub(tReg, sReg, tReg, pregCnt);
            AscendC::Reg::Sub(alReg, eReg, bReg, pregCnt);
            AscendC::Reg::Add(alReg, tReg, alReg, pregCnt);
            AscendC::Reg::Compare<float, AscendC::CMPMODE::EQ>(pregFinite, alReg, alReg, pregCnt);
            AscendC::Reg::Select(alReg, alReg, zeroReg, pregFinite);
            AscendC::Reg::StoreAlign(grpAccHi, ahReg, pregCnt);
            AscendC::Reg::StoreAlign(grpAccLo, alReg, pregCnt);
        }
    }
    // 行标量随后由 GradVec 读取, 收尾设一次屏障
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename T, typename TW, uint64_t hasWeight, uint64_t gammaMode>
__aicore__ inline void SoftmaxFocalLossGradND<T, TW, hasWeight, gammaMode>::SumsVec(
    __ubuf__ T* predAddr, __ubuf__ int32_t* targetAddr, __ubuf__ TW* weightAddr, __ubuf__ float* wfAddr,
    __ubuf__ float* wdAddr, __ubuf__ float* ceAddr, __ubuf__ float* wtAddr, int64_t rows, int64_t len)
{
    int64_t strideT = StrideOfT(len);
    int64_t strideW = StrideOfW(len);
    int64_t strideF = StrideOfF32(len);

    uint16_t aTimes = static_cast<uint16_t>(rows);
    uint32_t vfLen = VL_FP32;
    uint16_t repeatTimes = static_cast<uint16_t>(len / vfLen);
    uint32_t tailNum = static_cast<uint32_t>(len % vfLen);
    uint16_t tailLoop = tailNum != 0 ? 1 : 0;
    uint32_t tailAlign = static_cast<uint32_t>(strideF - repeatTimes * vfLen);

    // 一行装不满一个向量长度时, 按"一行一块"只有 S/VL 的车道利用率(D=2 时 8/64)。
    // 各缓冲行距一致时, 一块可以装 R = VL/S 行 —— 逐元素计算不关心行边界, 只换寻址。
    int64_t strideUni = strideF;
    bool packRows = (len < static_cast<int64_t>(vfLen)) && (len == strideUni) && (strideT == strideUni) &&
                    ((hasWeight == 0) || (strideW == strideUni)) && (rows > 1);
    if (packRows) {
        uint16_t rowsPerBlk = static_cast<uint16_t>(static_cast<int64_t>(vfLen) / strideUni);
        uint16_t groups = static_cast<uint16_t>((rows + rowsPerBlk - 1) / rowsPerBlk);
        uint32_t fullLanes = static_cast<uint32_t>(rowsPerBlk * strideUni);
        uint32_t lastLanes = static_cast<uint32_t>((rows - static_cast<int64_t>(groups - 1) * rowsPerBlk) * strideUni);
        uint32_t blkStep = fullLanes;
        __VEC_SCOPE__
        {
            RegTensor<float> p32;
            RegTensor<float> t32;
            RegTensor<float> w32;

            MaskReg pregLast = UpdateMask<float>(lastLanes);
            MaskReg pregFull = UpdateMask<float>(fullLanes);

            for (uint16_t g = 0; g < groups; ++g) {
                MaskReg preg = (g + 1 < groups) ? pregFull : pregLast;
                AscendC::Reg::AddrReg offT = AscendC::Reg::CreateAddrReg<T>(g, blkStep);
                AscendC::Reg::AddrReg offW = AscendC::Reg::CreateAddrReg<TW>(g, blkStep);
                AscendC::Reg::AddrReg offF = AscendC::Reg::CreateAddrReg<float>(g, blkStep);

                LoadPTW(p32, t32, w32, predAddr, targetAddr, weightAddr, offT, offW, offF, preg);
                SumsOfSeg(wfAddr, wdAddr, ceAddr, wtAddr, p32, t32, w32, offF, preg, preg);
            }
        }
        return;
    }

    __VEC_SCOPE__
    {
        RegTensor<float> p32;
        RegTensor<float> t32;
        RegTensor<float> w32;

        MaskReg pregMain = AscendC::Reg::CreateMask<float, MaskPattern::ALL>();
        MaskReg pregTail = UpdateMask<float>(tailNum);
        MaskReg storeTail = UpdateMask<float>(tailAlign);

        for (uint16_t i = 0; i < aTimes; ++i) {
            for (uint16_t j = 0; j < repeatTimes + tailLoop; ++j) {
                MaskReg preg = (j < repeatTimes) ? pregMain : pregTail;
                AscendC::Reg::AddrReg offT = AscendC::Reg::CreateAddrReg<T>(i, static_cast<uint32_t>(strideT), j,
                                                                            vfLen);
                AscendC::Reg::AddrReg offW = AscendC::Reg::CreateAddrReg<TW>(i, static_cast<uint32_t>(strideW), j,
                                                                             vfLen);
                AscendC::Reg::AddrReg offF = AscendC::Reg::CreateAddrReg<float>(i, static_cast<uint32_t>(strideF), j,
                                                                                vfLen);

                LoadPTW(p32, t32, w32, predAddr, targetAddr, weightAddr, offT, offW, offF, preg);
                SumsOfSeg(wfAddr, wdAddr, ceAddr, wtAddr, p32, t32, w32, offF, preg,
                          (j < repeatTimes) ? pregMain : storeTail);
            }
        }
    }
}

template <typename T, typename TW, uint64_t hasWeight, uint64_t gammaMode>
__aicore__ inline void SoftmaxFocalLossGradND<T, TW, hasWeight, gammaMode>::ComputeSums(int64_t rows, int64_t len)
{
    LocalTensor<T> predBuf = predQue_.template DeQue<T>();
    LocalTensor<int32_t> targetBuf = targetQue_.template DeQue<int32_t>();
    LocalTensor<T> doutBuf = doutQue_.template DeQue<T>();
    LocalTensor<float> wfBuf = wfBuf_.template Get<float>();
    LocalTensor<float> wdBuf = wdBuf_.template Get<float>();
    LocalTensor<float> ceBuf = ceBuf_.template Get<float>();
    LocalTensor<float> wtBuf = wtBuf_.template Get<float>();

    LocalTensor<TW> weightBuf;
    __ubuf__ TW* weightAddr = nullptr;
    if constexpr (hasWeight == 1) {
        weightBuf = weightQue_.template DeQue<TW>();
        weightAddr = (__ubuf__ TW*)weightBuf.GetPhyAddr();
    }

    SumsVec((__ubuf__ T*)predBuf.GetPhyAddr(), (__ubuf__ int32_t*)targetBuf.GetPhyAddr(), weightAddr,
            (__ubuf__ float*)wfBuf.GetPhyAddr(), (__ubuf__ float*)wdBuf.GetPhyAddr(),
            (__ubuf__ float*)ceBuf.GetPhyAddr(), (__ubuf__ float*)wtBuf.GetPhyAddr(), rows, len);

    predQue_.FreeTensor(predBuf);
    targetQue_.FreeTensor(targetBuf);
    doutQue_.FreeTensor(doutBuf);
    if constexpr (hasWeight == 1) {
        weightQue_.FreeTensor(weightBuf);
    }
}

template <typename T, typename TW, uint64_t hasWeight, uint64_t gammaMode>
__aicore__ inline void SoftmaxFocalLossGradND<T, TW, hasWeight, gammaMode>::MakeSegOffsets(
    AscendC::Reg::AddrReg& offT, AscendC::Reg::AddrReg& offW, AscendC::Reg::AddrReg& offF, AscendC::Reg::AddrReg& offO,
    uint16_t i, uint16_t j, int64_t strideT, int64_t strideW, int64_t strideF, uint32_t vfLen)
{
    // offO 走 T 的行距(grad 落 fp32 缓冲但按输出行布局), 与 offF 的 fp32 行距区分
    offT = AscendC::Reg::CreateAddrReg<T>(i, static_cast<uint32_t>(strideT), j, vfLen);
    offW = AscendC::Reg::CreateAddrReg<TW>(i, static_cast<uint32_t>(strideW), j, vfLen);
    offF = AscendC::Reg::CreateAddrReg<float>(i, static_cast<uint32_t>(strideF), j, vfLen);
    offO = AscendC::Reg::CreateAddrReg<float>(i, static_cast<uint32_t>(strideT), j, vfLen);
}

template <typename T, typename TW, uint64_t hasWeight, uint64_t gammaMode>
__aicore__ inline void SoftmaxFocalLossGradND<T, TW, hasWeight, gammaMode>::LoadRowScalars(
    RegTensor<float>& wfB, RegTensor<float>& wdB, RegTensor<float>& ceB, RegTensor<float>& wB, __ubuf__ float* accAddr,
    int64_t accStride, uint16_t rowIdx)
{
    // 行标量广播: WF / WB / CE / W
    AscendC::Reg::LoadAlign<float, LoadDist::DIST_BRC_B32>(wfB, accAddr + rowIdx);
    AscendC::Reg::LoadAlign<float, LoadDist::DIST_BRC_B32>(wdB, accAddr + accStride + rowIdx);
    AscendC::Reg::LoadAlign<float, LoadDist::DIST_BRC_B32>(ceB, accAddr + 2 * accStride + rowIdx);
    AscendC::Reg::LoadAlign<float, LoadDist::DIST_BRC_B32>(wB, accAddr + 3 * accStride + rowIdx);
}

template <typename T, typename TW, uint64_t hasWeight, uint64_t gammaMode>
__aicore__ inline void SoftmaxFocalLossGradND<T, TW, hasWeight, gammaMode>::GradOfSeg(
    __ubuf__ float* gradAddr, RegTensor<float>& p32, RegTensor<float>& t32, RegTensor<float>& w32,
    RegTensor<float>& d32, RegTensor<float>& wfB, RegTensor<float>& wdB, RegTensor<float>& ceB, RegTensor<float>& wB,
    AscendC::Reg::AddrReg offO, MaskReg preg, __ubuf__ float* accAddr, int64_t accStride, uint16_t rowIdx)
{
    RegTensor<float> oneReg;
    RegTensor<float> tmpReg;
    RegTensor<float> dCe;
    RegTensor<float> dWf;
    float gammaNeg = -gamma_;
    float gammaSub1 = gamma_ - 1.0f;
    float alpha = alpha_;
    float coef = coef_;

    // d_ce = p * W - t * w
    AscendC::Reg::Mul(dCe, p32, wB, preg);
    AscendC::Reg::Mul(tmpReg, t32, w32, preg);
    AscendC::Reg::Sub(dCe, dCe, tmpReg, preg);

    // wb(逐元素) = alpha * exp((gamma - 1) * log(1 - p)) * t
    AscendC::Reg::Duplicate(oneReg, 1.0f);
    AscendC::Reg::Sub(tmpReg, oneReg, p32, preg);
    if constexpr (gammaMode == 1) {
        AscendC::Reg::Duplicate(tmpReg, 1.0f);
    } else if constexpr (gammaMode == 0) {
        AscendC::Reg::Log(tmpReg, tmpReg, preg);
        AscendC::Reg::Muls(tmpReg, tmpReg, gammaSub1, preg);
        AscendC::Reg::Exp(tmpReg, tmpReg, preg);
    }
    AscendC::Reg::Muls(tmpReg, tmpReg, alpha, preg);
    AscendC::Reg::Mul(tmpReg, tmpReg, t32, preg);

    if constexpr (sizeof(T) == sizeof(half)) {
        if (r_ > static_cast<int64_t>(VL_FP32)) {
            RegTensor<float> wfLow, wdLow, ceLow, wLow;
            RegTensor<float> zero, scalar, negativeT, wbLow;
            RegTensor<float> aHi, aLo, bHi, bLo, cHi, cLo, outHi, outLo;
            AscendC::Reg::Duplicate(zero, 0.0f);
            AscendC::Reg::LoadAlign<float, LoadDist::DIST_BRC_B32>(wfLow, accAddr + 4 * accStride + rowIdx);
            AscendC::Reg::LoadAlign<float, LoadDist::DIST_BRC_B32>(wdLow, accAddr + 5 * accStride + rowIdx);
            AscendC::Reg::LoadAlign<float, LoadDist::DIST_BRC_B32>(ceLow, accAddr + 6 * accStride + rowIdx);
            AscendC::Reg::LoadAlign<float, LoadDist::DIST_BRC_B32>(wLow, accAddr + 7 * accStride + rowIdx);

            // d_ce = p * W - t * w.
            MultiplyFloatPair(aHi, aLo, p32, zero, wB, wLow, preg);
            AscendC::Reg::Muls(negativeT, t32, -1.0f, preg);
            MultiplyFloatPair<true>(bHi, bLo, negativeT, zero, w32, zero, preg);
            AddFloatPair(cHi, cLo, aHi, aLo, bHi, bLo, preg);
            MultiplyFloatPair(aHi, aLo, cHi, cLo, wfB, wfLow, preg);

            // The polynomial modes avoid transcendental rounding in wb.
            AscendC::Reg::Duplicate(wbLow, 0.0f);
            if constexpr (gammaMode == 1 || gammaMode == 2) {
                AscendC::Reg::Duplicate(scalar, 1.0f);
                if constexpr (gammaMode == 2) {
                    AscendC::Reg::Muls(negativeT, p32, -1.0f, preg);
                    AddFloatPair(tmpReg, wbLow, scalar, zero, negativeT, zero, preg);
                } else {
                    AscendC::Reg::Muls(tmpReg, scalar, 1.0f, preg);
                }
                AscendC::Reg::Duplicate(scalar, alpha);
                MultiplyFloatPair<true>(bHi, bLo, tmpReg, wbLow, scalar, zero, preg);
                MultiplyFloatPair<true>(tmpReg, wbLow, bHi, bLo, t32, zero, preg);
            }
            AddFloatPair(bHi, bLo, wdB, wdLow, tmpReg, wbLow, preg);
            MultiplyFloatPair<true>(cHi, cLo, bHi, bLo, p32, zero, preg);
            AscendC::Reg::Duplicate(scalar, gammaNeg);
            MultiplyFloatPair<true>(bHi, bLo, cHi, cLo, scalar, zero, preg);
            MultiplyFloatPair(cHi, cLo, bHi, bLo, ceB, ceLow, preg);
            AddFloatPair(outHi, outLo, aHi, aLo, cHi, cLo, preg);
            // Preserve the low part of dout * coef when normalization is moved earlier.
            RegTensor<float> absDout, scaledHi, scaledLo, finalScale, unit;
            MaskReg normalizeFirst;
            AscendC::Reg::Duplicate(scalar, coef);
            AscendC::Reg::Abs(absDout, d32, preg);
            AscendC::Reg::CompareScalar<float, AscendC::CMPMODE::GT>(normalizeFirst, absDout, 1.0f, preg);
            MultiplyFloatPair<true>(scaledHi, scaledLo, d32, zero, scalar, zero, preg);
            AscendC::Reg::Select(scaledHi, scaledHi, d32, normalizeFirst);
            AscendC::Reg::Select(scaledLo, scaledLo, zero, normalizeFirst);
            AscendC::Reg::Duplicate(finalScale, coef);
            AscendC::Reg::Duplicate(unit, 1.0f);
            AscendC::Reg::Select(finalScale, unit, finalScale, normalizeFirst);
            MultiplyFloatPair(aHi, aLo, outHi, outLo, scaledHi, scaledLo, preg);
            MultiplyFloatPair<true>(outHi, outLo, aHi, aLo, finalScale, zero, preg);

            // Resolve the FP16 overflow midpoint using the complete low component.
            MaskReg midpoint, inward, repair;
            AscendC::Reg::CompareScalar<float, AscendC::CMPMODE::EQ>(midpoint, outHi, 65520.0f, preg);
            AscendC::Reg::CompareScalar<float, AscendC::CMPMODE::LT>(inward, outLo, 0.0f, preg);
            AscendC::Reg::MaskAnd(repair, midpoint, inward, preg);
            AscendC::Reg::Duplicate(scalar, 65519.99609375f);
            AscendC::Reg::Select(outHi, scalar, outHi, repair);
            AscendC::Reg::CompareScalar<float, AscendC::CMPMODE::EQ>(midpoint, outHi, -65520.0f, preg);
            AscendC::Reg::CompareScalar<float, AscendC::CMPMODE::GT>(inward, outLo, 0.0f, preg);
            AscendC::Reg::MaskAnd(repair, midpoint, inward, preg);
            AscendC::Reg::Duplicate(scalar, -65519.99609375f);
            AscendC::Reg::Select(outHi, scalar, outHi, repair);
            AscendC::Reg::StoreAlign(gradAddr, outHi, offO, preg);
            return;
        }
    }

    // d_wf = -gamma * (WD + wb) * p, 其中 WD = sum_j (wf - wb)
    AscendC::Reg::Add(dWf, wdB, tmpReg, preg);
    AscendC::Reg::Mul(dWf, dWf, p32, preg);
    AscendC::Reg::Muls(dWf, dWf, gammaNeg, preg);

    // grad = (d_ce * WF + d_wf * CE) * dout * coef
    AscendC::Reg::Mul(dWf, dWf, ceB, preg);
    AscendC::Reg::MulDstAdd(dCe, wfB, dWf, preg);
    // Move normalization ahead of a large dout to avoid intermediate overflow.
    // Small dout retains the original order to avoid premature underflow.
    RegTensor<float> absDout, scaledDout, finalScale;
    MaskReg normalizeFirst;
    AscendC::Reg::Abs(absDout, d32, preg);
    AscendC::Reg::CompareScalar<float, AscendC::CMPMODE::GT>(normalizeFirst, absDout, 1.0f, preg);
    AscendC::Reg::Muls(scaledDout, d32, coef, preg);
    AscendC::Reg::Select(scaledDout, scaledDout, d32, normalizeFirst);
    AscendC::Reg::Duplicate(finalScale, coef);
    AscendC::Reg::Duplicate(oneReg, 1.0f);
    AscendC::Reg::Select(finalScale, oneReg, finalScale, normalizeFirst);
    AscendC::Reg::Mul(dCe, dCe, scaledDout, preg);
    AscendC::Reg::Mul(dCe, dCe, finalScale, preg);
    AscendC::Reg::StoreAlign(gradAddr, dCe, offO, preg);
}

template <typename T, typename TW, uint64_t hasWeight, uint64_t gammaMode>
__aicore__ inline void SoftmaxFocalLossGradND<T, TW, hasWeight, gammaMode>::GradVec(
    __ubuf__ T* predAddr, __ubuf__ int32_t* targetAddr, __ubuf__ T* doutAddr, __ubuf__ TW* weightAddr,
    __ubuf__ float* accAddr, __ubuf__ float* gradAddr, int64_t rows, int64_t len)
{
    int64_t strideT = StrideOfT(len);
    int64_t strideW = StrideOfW(len);
    int64_t strideF = StrideOfF32(len);
    int64_t accStride = accStride_;

    uint16_t aTimes = static_cast<uint16_t>(rows);
    uint32_t vfLen = VL_FP32;
    uint16_t repeatTimes = static_cast<uint16_t>(len / vfLen);
    uint32_t tailNum = static_cast<uint32_t>(len % vfLen);
    uint16_t tailLoop = tailNum != 0 ? 1 : 0;
    uint32_t tailAlign = static_cast<uint32_t>(strideT - repeatTimes * vfLen);

    // 同 SumsVec: 一块装 R 行。行标量不再按行广播, 改为按车道取(车道 l 属于行 l/S)。
    int64_t strideUni = strideF;
    bool packRows = (sizeof(T) != sizeof(half) || r_ <= static_cast<int64_t>(VL_FP32)) &&
                    (len < static_cast<int64_t>(vfLen)) && (strideT == strideUni) &&
                    ((hasWeight == 0) || (strideW == strideUni)) && (rows > 1);
    if (packRows) {
        uint16_t rowsPerBlk = static_cast<uint16_t>(static_cast<int64_t>(vfLen) / strideUni);
        uint16_t groups = static_cast<uint16_t>((rows + rowsPerBlk - 1) / rowsPerBlk);
        uint32_t fullLanes = static_cast<uint32_t>(rowsPerBlk * strideUni);
        uint32_t lastLanes = static_cast<uint32_t>((rows - static_cast<int64_t>(groups - 1) * rowsPerBlk) * strideUni);
        uint32_t blkStep = fullLanes;
        float strideFp = static_cast<float>(strideUni);
        __VEC_SCOPE__
        {
            RegTensor<T> doutReg;
            RegTensor<float> p32;
            RegTensor<float> t32;
            RegTensor<float> w32;
            RegTensor<float> d32;
            RegTensor<float> wfB;
            RegTensor<float> wdB;
            RegTensor<float> ceB;
            RegTensor<float> wB;
            RegTensor<int32_t> laneId;
            RegTensor<int32_t> idxCur;
            RegTensor<float> laneF;
            RegTensor<float> strideReg;

            MaskReg pregLast = UpdateMask<float>(lastLanes);
            MaskReg pregFull = UpdateMask<float>(fullLanes);

            // 车道 -> 行号: 行号 = 车道号 / 行距。用真除(Div)而不是乘倒数,
            // 1/24 这类不精确的倒数会让 l=24 落到 0.99999994 而 FLOOR 掉一档。
            AscendC::Reg::Arange(laneId, 0);
            AscendC::Reg::Cast<float, int32_t, CAST_TRAIT_S32_TO_FP32_LOCAL>(laneF, laneId, pregFull);
            AscendC::Reg::Duplicate(strideReg, strideFp);
            AscendC::Reg::Div(laneF, laneF, strideReg, pregFull);
            AscendC::Reg::Cast<int32_t, float, CAST_TRAIT_FP32_TO_S32_FLOOR_LOCAL>(laneId, laneF, pregFull);

            for (uint16_t g = 0; g < groups; ++g) {
                MaskReg preg = (g + 1 < groups) ? pregFull : pregLast;
                AscendC::Reg::Adds(idxCur, laneId, static_cast<int32_t>(g) * static_cast<int32_t>(rowsPerBlk), preg);
                AscendC::Reg::Gather(wfB, accAddr, (AscendC::Reg::RegTensor<uint32_t>&)idxCur, preg);
                AscendC::Reg::Gather(wdB, accAddr + accStride, (AscendC::Reg::RegTensor<uint32_t>&)idxCur, preg);
                AscendC::Reg::Gather(ceB, accAddr + 2 * accStride, (AscendC::Reg::RegTensor<uint32_t>&)idxCur, preg);
                AscendC::Reg::Gather(wB, accAddr + 3 * accStride, (AscendC::Reg::RegTensor<uint32_t>&)idxCur, preg);

                AscendC::Reg::AddrReg offT = AscendC::Reg::CreateAddrReg<T>(g, blkStep);
                AscendC::Reg::AddrReg offW = AscendC::Reg::CreateAddrReg<TW>(g, blkStep);
                AscendC::Reg::AddrReg offF = AscendC::Reg::CreateAddrReg<float>(g, blkStep);
                AscendC::Reg::AddrReg offO = AscendC::Reg::CreateAddrReg<float>(g, blkStep);

                LoadPTW(p32, t32, w32, predAddr, targetAddr, weightAddr, offT, offW, offF, preg);
                if constexpr (sizeof(T) == sizeof(half)) {
                    AscendC::Reg::LoadAlign<T, LoadDist::DIST_UNPACK_B16>(doutReg, doutAddr, offT);
                    AscendC::Reg::Cast<float, T, castB16ToB32>(d32, doutReg, preg);
                } else {
                    AscendC::Reg::LoadAlign(d32, doutAddr, offT);
                }
                GradOfSeg(gradAddr, p32, t32, w32, d32, wfB, wdB, ceB, wB, offO, preg, accAddr, accStride, 0);
            }
        }
        return;
    }

    __VEC_SCOPE__
    {
        RegTensor<T> doutReg;
        RegTensor<float> p32;
        RegTensor<float> t32;
        RegTensor<float> w32;
        RegTensor<float> d32;
        RegTensor<float> wfB;
        RegTensor<float> wdB;
        RegTensor<float> ceB;
        RegTensor<float> wB;

        MaskReg pregMain = AscendC::Reg::CreateMask<float, MaskPattern::ALL>();
        MaskReg pregTail = UpdateMask<float>(tailAlign);

        for (uint16_t i = 0; i < aTimes; ++i) {
            LoadRowScalars(wfB, wdB, ceB, wB, accAddr, accStride, i);

            for (uint16_t j = 0; j < repeatTimes + tailLoop; ++j) {
                MaskReg preg = (j < repeatTimes) ? pregMain : pregTail;
                AscendC::Reg::AddrReg offT;
                AscendC::Reg::AddrReg offW;
                AscendC::Reg::AddrReg offF;
                AscendC::Reg::AddrReg offO;
                MakeSegOffsets(offT, offW, offF, offO, i, j, strideT, strideW, strideF, vfLen);

                LoadPTW(p32, t32, w32, predAddr, targetAddr, weightAddr, offT, offW, offF, preg);
                if constexpr (sizeof(T) == sizeof(half)) {
                    AscendC::Reg::LoadAlign<T, LoadDist::DIST_UNPACK_B16>(doutReg, doutAddr, offT);
                    AscendC::Reg::Cast<float, T, castB16ToB32>(d32, doutReg, preg);
                } else {
                    AscendC::Reg::LoadAlign(d32, doutAddr, offT);
                }

                GradOfSeg(gradAddr, p32, t32, w32, d32, wfB, wdB, ceB, wB, offO, preg, accAddr, accStride, i);
            }
        }
    }
}

template <typename T, typename TW, uint64_t hasWeight, uint64_t gammaMode>
__aicore__ inline void SoftmaxFocalLossGradND<T, TW, hasWeight, gammaMode>::ComputeGrad(int64_t rows, int64_t len)
{
    LocalTensor<T> predBuf = predQue_.template DeQue<T>();
    LocalTensor<int32_t> targetBuf = targetQue_.template DeQue<int32_t>();
    LocalTensor<T> doutBuf = doutQue_.template DeQue<T>();
    LocalTensor<float> accBuf = accBuf_.template Get<float>();
    LocalTensor<float> gradF32 = wfBuf_.template Get<float>(); // 第一趟的 wf 缓冲已无用, 复用
    LocalTensor<T> gradBuf = gradQue_.template AllocTensor<T>();

    LocalTensor<TW> weightBuf;
    __ubuf__ TW* weightAddr = nullptr;
    if constexpr (hasWeight == 1) {
        weightBuf = weightQue_.template DeQue<TW>();
        weightAddr = (__ubuf__ TW*)weightBuf.GetPhyAddr();
    }

    GradVec((__ubuf__ T*)predBuf.GetPhyAddr(), (__ubuf__ int32_t*)targetBuf.GetPhyAddr(),
            (__ubuf__ T*)doutBuf.GetPhyAddr(), weightAddr, (__ubuf__ float*)accBuf.GetPhyAddr(),
            (__ubuf__ float*)gradF32.GetPhyAddr(), rows, len);

    int32_t total = static_cast<int32_t>(rows * StrideOfT(len));
    if constexpr (sizeof(T) == sizeof(half)) {
        AscendC::Cast(gradBuf, gradF32, AscendC::RoundMode::CAST_RINT, total);
    } else {
        AscendC::Copy(gradBuf, gradF32, total);
    }
    gradQue_.template EnQue<T>(gradBuf);

    predQue_.FreeTensor(predBuf);
    targetQue_.FreeTensor(targetBuf);
    doutQue_.FreeTensor(doutBuf);
    if constexpr (hasWeight == 1) {
        weightQue_.FreeTensor(weightBuf);
    }
}

} // namespace SoftmaxFocalLossGrad

#endif // SOFTMAX_FOCAL_LOSS_GRAD_ND_VEC_H
