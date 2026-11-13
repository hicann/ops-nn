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
 * \file softmax_focal_loss_nd.h
 * \brief softmax_focal_loss regbase kernel
 *
 * 计算语义(对齐 A2 tbe.dsl softmax_focal_loss_compute):
 *   ce[b][j] = -log(pred) * target * weight
 *   CE[b]    = sum_j ce[b][j]
 *   fw[b][j] = alpha * exp(gamma * log(1 - pred)) * target
 *   FW[b]    = sum_j fw[b][j]
 *   y[b][j]  = CE[b] * FW[b]          // 整行同值
 *
 * 调度: 行分核, 核内按行块推进, 行内按列块推进(列块数为 1 时即全载路径)。
 */

#ifndef SOFTMAX_FOCAL_LOSS_ND_VEC_H
#define SOFTMAX_FOCAL_LOSS_ND_VEC_H

#include "softmax_focal_loss_nd.h"

namespace SoftmaxFocalLoss {

template <typename T, typename TW, uint64_t hasWeight>
template <bool byAddrReg>
__aicore__ inline void SoftmaxFocalLossND<T, TW, hasWeight>::LoadPredTargetWeight(
    RegTensor<float>& p32, RegTensor<float>& t32, RegTensor<float>& w32, __ubuf__ T* predPtr,
    __ubuf__ int32_t* targetPtr, __ubuf__ TW* weightPtr, AscendC::Reg::AddrReg offT, AscendC::Reg::AddrReg offW,
    AscendC::Reg::AddrReg offF, MaskReg preg)
{
    RegTensor<T> predReg;
    RegTensor<int32_t> targetRegI32;
    RegTensor<TW> weightReg;

    if constexpr (sizeof(T) == sizeof(half)) {
        if constexpr (byAddrReg) {
            AscendC::Reg::LoadAlign<T, LoadDist::DIST_UNPACK_B16>(predReg, predPtr, offT);
        } else {
            AscendC::Reg::LoadAlign<T, LoadDist::DIST_UNPACK_B16>(predReg, predPtr);
        }
        AscendC::Reg::Cast<float, T, castB16ToB32>(p32, predReg, preg);
    } else {
        if constexpr (byAddrReg) {
            AscendC::Reg::LoadAlign(p32, predPtr, offT);
        } else {
            AscendC::Reg::LoadAlign(p32, predPtr);
        }
    }

    if constexpr (byAddrReg) {
        AscendC::Reg::LoadAlign(targetRegI32, targetPtr, offF);
    } else {
        AscendC::Reg::LoadAlign(targetRegI32, targetPtr);
    }
    AscendC::Reg::Cast<float, int32_t, castI32ToF32>(t32, targetRegI32, preg);

    if constexpr (hasWeight == 1) {
        if constexpr (sizeof(TW) == sizeof(half)) {
            if constexpr (byAddrReg) {
                AscendC::Reg::LoadAlign<TW, LoadDist::DIST_UNPACK_B16>(weightReg, weightPtr, offW);
            } else {
                AscendC::Reg::LoadAlign<TW, LoadDist::DIST_UNPACK_B16>(weightReg, weightPtr);
            }
            AscendC::Reg::Cast<float, TW, castB16ToB32>(w32, weightReg, preg);
        } else {
            if constexpr (byAddrReg) {
                AscendC::Reg::LoadAlign(w32, weightPtr, offW);
            } else {
                AscendC::Reg::LoadAlign(w32, weightPtr);
            }
        }
    } else {
        AscendC::Reg::Duplicate(w32, 1.0f);
    }
}

template <typename T, typename TW, uint64_t hasWeight>
__aicore__ inline void SoftmaxFocalLossND<T, TW, hasWeight>::CeFwOfSeg(RegTensor<float>& ceReg, RegTensor<float>& fwReg,
                                                                       RegTensor<float>& p32, RegTensor<float>& t32,
                                                                       RegTensor<float>& w32, float gamma, float alpha,
                                                                       MaskReg preg)
{
    RegTensor<float> oneReg;
    RegTensor<float> tmpReg;

    // ce = -log(p) * t * w
    AscendC::Reg::Log(tmpReg, p32, preg);
    AscendC::Reg::Muls(ceReg, tmpReg, -1.0f, preg);
    AscendC::Reg::Mul(ceReg, ceReg, t32, preg);
    AscendC::Reg::Mul(ceReg, ceReg, w32, preg);

    // fw = alpha * exp(gamma * log(1 - p)) * t
    AscendC::Reg::Duplicate(oneReg, 1.0f);
    AscendC::Reg::Sub(tmpReg, oneReg, p32, preg);
    AscendC::Reg::Log(tmpReg, tmpReg, preg);
    AscendC::Reg::Muls(tmpReg, tmpReg, gamma, preg);
    AscendC::Reg::Exp(fwReg, tmpReg, preg);
    AscendC::Reg::Muls(fwReg, fwReg, alpha, preg);
    AscendC::Reg::Mul(fwReg, fwReg, t32, preg);
}

template <typename T, typename TW, uint64_t hasWeight>
__aicore__ inline void SoftmaxFocalLossND<T, TW, hasWeight>::CeFwTailSeg(
    __ubuf__ T* predAddr, __ubuf__ int32_t* targetAddr, __ubuf__ TW* weightAddr, __ubuf__ float* ceAddr,
    __ubuf__ float* fwAddr, int64_t rowIdx, int64_t strideT, int64_t strideW, int64_t strideF, uint32_t doneCols,
    MaskReg preg, MaskReg storeMask)
{
    RegTensor<float> p32;
    RegTensor<float> t32;
    RegTensor<float> w32;
    RegTensor<float> ceReg;
    RegTensor<float> fwReg;

    __ubuf__ T* predTail = predAddr + rowIdx * strideT + doneCols;
    __ubuf__ int32_t* targetTail = targetAddr + rowIdx * strideF + doneCols;
    __ubuf__ float* ceTail = ceAddr + rowIdx * strideF + doneCols;
    __ubuf__ float* fwTail = fwAddr + rowIdx * strideF + doneCols;
    __ubuf__ TW* weightTail = hasWeight == 1 ? weightAddr + rowIdx * strideW + doneCols : weightAddr;
    AscendC::Reg::AddrReg dummy = AscendC::Reg::CreateAddrReg<float>(0, 0, 0, VL_FP32);

    LoadPredTargetWeight<false>(p32, t32, w32, predTail, targetTail, weightTail, dummy, dummy, dummy, preg);
    CeFwOfSeg(ceReg, fwReg, p32, t32, w32, gamma_, alpha_, preg);

    AscendC::Reg::StoreAlign(ceTail, ceReg, storeMask);
    AscendC::Reg::StoreAlign(fwTail, fwReg, storeMask);
}

template <typename T, typename TW, uint64_t hasWeight>
__aicore__ inline void SoftmaxFocalLossND<T, TW, hasWeight>::CeFwVec(__ubuf__ T* predAddr, __ubuf__ int32_t* targetAddr,
                                                                     __ubuf__ TW* weightAddr, __ubuf__ float* ceAddr,
                                                                     __ubuf__ float* fwAddr, int64_t rows, int64_t len)
{
    int64_t strideT = StrideOfT(len);
    int64_t strideW = StrideOfW(len);
    int64_t strideF = StrideOfF32(len);

    uint16_t aTimes = static_cast<uint16_t>(rows);
    uint32_t vfLen = VL_FP32;
    uint16_t repeatTimes = static_cast<uint16_t>(len / vfLen);
    uint32_t tailNum = static_cast<uint32_t>(len % vfLen);
    uint16_t tailLoop = tailNum != 0 ? 1 : 0;
    // Calculate valid tail lanes only; ZEROING supplies zeros for the aligned reduction padding.
    uint32_t tailAlign = static_cast<uint32_t>(strideF - repeatTimes * vfLen);
    float gamma = gamma_;
    float alpha = alpha_;

    __VEC_SCOPE__
    {
        RegTensor<float> p32;
        RegTensor<float> t32;
        RegTensor<float> w32;
        RegTensor<float> ceReg;
        RegTensor<float> fwReg;

        MaskReg pregMain = AscendC::Reg::CreateMask<float, MaskPattern::ALL>();
        MaskReg pregTail = UpdateMask<float>(tailNum);
        MaskReg pregTailStore = UpdateMask<float>(tailAlign);

        for (uint16_t i = 0; i < aTimes; ++i) {
            for (uint16_t j = 0; j < repeatTimes; ++j) {
                AscendC::Reg::AddrReg offT = AscendC::Reg::CreateAddrReg<T>(i, static_cast<uint32_t>(strideT), j,
                                                                            vfLen);
                AscendC::Reg::AddrReg offW = AscendC::Reg::CreateAddrReg<TW>(i, static_cast<uint32_t>(strideW), j,
                                                                             vfLen);
                AscendC::Reg::AddrReg offF = AscendC::Reg::CreateAddrReg<float>(i, static_cast<uint32_t>(strideF), j,
                                                                                vfLen);

                LoadPredTargetWeight<true>(p32, t32, w32, predAddr, targetAddr, weightAddr, offT, offW, offF, pregMain);
                CeFwOfSeg(ceReg, fwReg, p32, t32, w32, gamma, alpha, pregMain);

                AscendC::Reg::StoreAlign(ceAddr, ceReg, offF, pregMain);
                AscendC::Reg::StoreAlign(fwAddr, fwReg, offF, pregMain);
            }

            for (uint16_t k = 0; k < tailLoop; ++k) {
                CeFwTailSeg(predAddr, targetAddr, weightAddr, ceAddr, fwAddr, i, strideT, strideW, strideF,
                            repeatTimes * vfLen, pregTail, pregTailStore);
            }
        }
    }
}

template <typename T, typename TW, uint64_t hasWeight>
__aicore__ inline void SoftmaxFocalLossND<T, TW, hasWeight>::ComputeCeFw(int64_t rows, int64_t len)
{
    LocalTensor<T> predBuf = predQue_.template DeQue<T>();
    LocalTensor<int32_t> targetBuf = targetQue_.template DeQue<int32_t>();
    LocalTensor<float> ceBuf = ceBuf_.template Get<float>();
    LocalTensor<float> fwBuf = fwBuf_.template Get<float>();

    LocalTensor<TW> weightBuf;
    __ubuf__ TW* weightAddr = nullptr;
    if constexpr (hasWeight == 1) {
        weightBuf = weightQue_.template DeQue<TW>();
        weightAddr = (__ubuf__ TW*)weightBuf.GetPhyAddr();
    }

    CeFwVec((__ubuf__ T*)predBuf.GetPhyAddr(), (__ubuf__ int32_t*)targetBuf.GetPhyAddr(), weightAddr,
            (__ubuf__ float*)ceBuf.GetPhyAddr(), (__ubuf__ float*)fwBuf.GetPhyAddr(), rows, len);

    predQue_.FreeTensor(predBuf);
    targetQue_.FreeTensor(targetBuf);
    if constexpr (hasWeight == 1) {
        weightQue_.FreeTensor(weightBuf);
    }
}

template <typename T, typename TW, uint64_t hasWeight>
__aicore__ inline void SoftmaxFocalLossND<T, TW, hasWeight>::MergeRowSlots2(__ubuf__ float* ceHi, __ubuf__ float* ceLo,
                                                                            __ubuf__ float* fwHi, __ubuf__ float* fwLo,
                                                                            __ubuf__ float* rowHi,
                                                                            __ubuf__ float* rowLo, int64_t slotSpan,
                                                                            int64_t g, uint32_t cntMask)
{
    __VEC_SCOPE__
    {
        RegTensor<float> rhReg;
        RegTensor<float> rlReg;
        RegTensor<float> ahReg;
        RegTensor<float> alReg;
        RegTensor<float> sR;
        RegTensor<float> eR;
        RegTensor<float> bR;
        RegTensor<float> tR;
        RegTensor<float> zeroReg;
        MaskReg pregFinite;
        MaskReg pregCnt = AscendC::Reg::UpdateMask<float>(cntMask);

        AscendC::Reg::Duplicate(zeroReg, 0.0f);
        AscendC::Reg::RegTensor<int32_t> idxReg;
        AscendC::Reg::Arange(idxReg, 0);
        AscendC::Reg::Muls(idxReg, idxReg, static_cast<int32_t>(ROW_SLOT_ELEM), pregCnt);
        AscendC::Reg::Gather(rhReg, rowHi + 0 * slotSpan, (AscendC::Reg::RegTensor<uint32_t>&)idxReg, pregCnt);
        AscendC::Reg::Gather(rlReg, rowLo + 0 * slotSpan, (AscendC::Reg::RegTensor<uint32_t>&)idxReg, pregCnt);
        AscendC::Reg::LoadAlign(ahReg, ceHi + g);
        AscendC::Reg::LoadAlign(alReg, ceLo + g);
        AscendC::Reg::Add(sR, ahReg, rhReg, pregCnt);
        AscendC::Reg::Sub(bR, sR, ahReg, pregCnt);
        AscendC::Reg::Sub(tR, sR, bR, pregCnt);
        AscendC::Reg::Sub(tR, ahReg, tR, pregCnt);
        AscendC::Reg::Sub(eR, rhReg, bR, pregCnt);
        AscendC::Reg::Add(eR, tR, eR, pregCnt);
        AscendC::Reg::Add(eR, eR, alReg, pregCnt);
        AscendC::Reg::Compare<float, AscendC::CMPMODE::EQ>(pregFinite, eR, eR, pregCnt);
        AscendC::Reg::Select(eR, eR, zeroReg, pregFinite);
        AscendC::Reg::Add(ahReg, sR, eR, pregCnt);
        AscendC::Reg::Sub(bR, ahReg, sR, pregCnt);
        AscendC::Reg::Sub(tR, ahReg, bR, pregCnt);
        AscendC::Reg::Sub(tR, sR, tR, pregCnt);
        AscendC::Reg::Sub(alReg, eR, bR, pregCnt);
        AscendC::Reg::Add(alReg, tR, alReg, pregCnt);
        AscendC::Reg::Compare<float, AscendC::CMPMODE::EQ>(pregFinite, alReg, alReg, pregCnt);
        AscendC::Reg::Select(alReg, alReg, zeroReg, pregFinite);
        AscendC::Reg::Add(sR, ahReg, rlReg, pregCnt);
        AscendC::Reg::Sub(bR, sR, ahReg, pregCnt);
        AscendC::Reg::Sub(tR, sR, bR, pregCnt);
        AscendC::Reg::Sub(tR, ahReg, tR, pregCnt);
        AscendC::Reg::Sub(eR, rlReg, bR, pregCnt);
        AscendC::Reg::Add(eR, tR, eR, pregCnt);
        AscendC::Reg::Add(eR, eR, alReg, pregCnt);
        AscendC::Reg::Compare<float, AscendC::CMPMODE::EQ>(pregFinite, eR, eR, pregCnt);
        AscendC::Reg::Select(eR, eR, zeroReg, pregFinite);
        AscendC::Reg::Add(ahReg, sR, eR, pregCnt);
        AscendC::Reg::Sub(bR, ahReg, sR, pregCnt);
        AscendC::Reg::Sub(tR, ahReg, bR, pregCnt);
        AscendC::Reg::Sub(tR, sR, tR, pregCnt);
        AscendC::Reg::Sub(alReg, eR, bR, pregCnt);
        AscendC::Reg::Add(alReg, tR, alReg, pregCnt);
        AscendC::Reg::Compare<float, AscendC::CMPMODE::EQ>(pregFinite, alReg, alReg, pregCnt);
        AscendC::Reg::Select(alReg, alReg, zeroReg, pregFinite);
        AscendC::Reg::StoreAlign(ceHi + g, ahReg, pregCnt);
        AscendC::Reg::StoreAlign(ceLo + g, alReg, pregCnt);
        AscendC::Reg::Gather(rhReg, rowHi + 1 * slotSpan, (AscendC::Reg::RegTensor<uint32_t>&)idxReg, pregCnt);
        AscendC::Reg::Gather(rlReg, rowLo + 1 * slotSpan, (AscendC::Reg::RegTensor<uint32_t>&)idxReg, pregCnt);
        AscendC::Reg::LoadAlign(ahReg, fwHi + g);
        AscendC::Reg::LoadAlign(alReg, fwLo + g);
        AscendC::Reg::Add(sR, ahReg, rhReg, pregCnt);
        AscendC::Reg::Sub(bR, sR, ahReg, pregCnt);
        AscendC::Reg::Sub(tR, sR, bR, pregCnt);
        AscendC::Reg::Sub(tR, ahReg, tR, pregCnt);
        AscendC::Reg::Sub(eR, rhReg, bR, pregCnt);
        AscendC::Reg::Add(eR, tR, eR, pregCnt);
        AscendC::Reg::Add(eR, eR, alReg, pregCnt);
        AscendC::Reg::Compare<float, AscendC::CMPMODE::EQ>(pregFinite, eR, eR, pregCnt);
        AscendC::Reg::Select(eR, eR, zeroReg, pregFinite);
        AscendC::Reg::Add(ahReg, sR, eR, pregCnt);
        AscendC::Reg::Sub(bR, ahReg, sR, pregCnt);
        AscendC::Reg::Sub(tR, ahReg, bR, pregCnt);
        AscendC::Reg::Sub(tR, sR, tR, pregCnt);
        AscendC::Reg::Sub(alReg, eR, bR, pregCnt);
        AscendC::Reg::Add(alReg, tR, alReg, pregCnt);
        AscendC::Reg::Compare<float, AscendC::CMPMODE::EQ>(pregFinite, alReg, alReg, pregCnt);
        AscendC::Reg::Select(alReg, alReg, zeroReg, pregFinite);
        AscendC::Reg::Add(sR, ahReg, rlReg, pregCnt);
        AscendC::Reg::Sub(bR, sR, ahReg, pregCnt);
        AscendC::Reg::Sub(tR, sR, bR, pregCnt);
        AscendC::Reg::Sub(tR, ahReg, tR, pregCnt);
        AscendC::Reg::Sub(eR, rlReg, bR, pregCnt);
        AscendC::Reg::Add(eR, tR, eR, pregCnt);
        AscendC::Reg::Add(eR, eR, alReg, pregCnt);
        AscendC::Reg::Compare<float, AscendC::CMPMODE::EQ>(pregFinite, eR, eR, pregCnt);
        AscendC::Reg::Select(eR, eR, zeroReg, pregFinite);
        AscendC::Reg::Add(ahReg, sR, eR, pregCnt);
        AscendC::Reg::Sub(bR, ahReg, sR, pregCnt);
        AscendC::Reg::Sub(tR, ahReg, bR, pregCnt);
        AscendC::Reg::Sub(tR, sR, tR, pregCnt);
        AscendC::Reg::Sub(alReg, eR, bR, pregCnt);
        AscendC::Reg::Add(alReg, tR, alReg, pregCnt);
        AscendC::Reg::Compare<float, AscendC::CMPMODE::EQ>(pregFinite, alReg, alReg, pregCnt);
        AscendC::Reg::Select(alReg, alReg, zeroReg, pregFinite);
        AscendC::Reg::StoreAlign(fwHi + g, ahReg, pregCnt);
        AscendC::Reg::StoreAlign(fwLo + g, alReg, pregCnt);
    }
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename T, typename TW, uint64_t hasWeight>
__aicore__ inline void SoftmaxFocalLossND<T, TW, hasWeight>::ReduceFwWithResidual(RegTensor<float>& hi,
                                                                                  RegTensor<float>& lo,
                                                                                  RegTensor<float>& value,
                                                                                  RegTensor<float>& zero, MaskReg mask)
{
    RegTensor<uint32_t> bits;
    RegTensor<float> high, low;
    MaskReg number;
    // Leave 12 significant bits in the high part for the vector sum.
    AscendC::Reg::Duplicate(bits, 0xfffff000U);
    AscendC::Reg::And((RegTensor<uint32_t>&)high, (RegTensor<uint32_t>&)value, bits, mask);
    AscendC::Reg::Compare<float, AscendC::CMPMODE::EQ>(number, value, value, mask);
    AscendC::Reg::Select(high, high, value, number);
    AscendC::Reg::Sub(low, value, high, mask);
    AscendC::Reg::Compare<float, AscendC::CMPMODE::EQ>(number, low, low, mask);
    AscendC::Reg::Select(low, low, zero, number);
    AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM>(hi, high, mask);
    AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM>(lo, low, mask);
}

template <typename T, typename TW, uint64_t hasWeight>
__aicore__ inline void SoftmaxFocalLossND<T, TW, hasWeight>::ComputeCompensatedProduct()
{
    // Streamed column tiles always process one row at a time.
    LocalTensor<float> accLocal = accBuf_.template Get<float>();
    auto acc = (__ubuf__ float*)accLocal.GetPhyAddr();
    uint32_t one = 1;
    __VEC_SCOPE__
    {
        RegTensor<float> ce, fw, ceLo, fwLo, product, error, magnitude, zero, value;
        MaskReg finite, nonzero;
        MaskReg mask = UpdateMask<float>(one);
        AscendC::Reg::LoadAlign<float, LoadDist::DIST_BRC_B32>(ce, acc);
        AscendC::Reg::LoadAlign<float, LoadDist::DIST_BRC_B32>(fw, acc + accStride_);
        AscendC::Reg::LoadAlign<float, LoadDist::DIST_BRC_B32>(ceLo, acc + 4 * accStride_);
        AscendC::Reg::LoadAlign<float, LoadDist::DIST_BRC_B32>(fwLo, acc + 5 * accStride_);
        AscendC::Reg::Mul(product, ce, fw, mask);
        AscendC::Reg::Muls(error, product, -1.0f, mask);
        AscendC::Reg::MulAddDst(error, ce, fw, mask);
        AscendC::Reg::MulAddDst(error, ce, fwLo, mask);
        AscendC::Reg::MulAddDst(error, ceLo, fw, mask);
        AscendC::Reg::MulAddDst(error, ceLo, fwLo, mask);
        // A non-finite or zero correction leaves the original product intact.
        AscendC::Reg::Duplicate(zero, 0.0f);
        AscendC::Reg::Abs(magnitude, error, mask);
        AscendC::Reg::Compares<float, AscendC::CMPMODE::LE>(finite, magnitude, 0x1.fffffep127f, mask);
        AscendC::Reg::Select(error, error, zero, finite);
        AscendC::Reg::Compare<float, AscendC::CMPMODE::NE>(nonzero, error, zero, mask);
        AscendC::Reg::Add(value, product, error, mask);
        AscendC::Reg::Select(value, value, product, nonzero);
        AscendC::Reg::StoreAlign(acc + 3 * accStride_, value, mask);
    }
}

template <typename T, typename TW, uint64_t hasWeight>
template <bool compensateFw, bool mergeResult>
__aicore__ inline void SoftmaxFocalLossND<T, TW, hasWeight>::ExactRowSum2(__ubuf__ float* srcCe, __ubuf__ float* srcFw,
                                                                          __ubuf__ float* ceHi, __ubuf__ float* ceLo,
                                                                          __ubuf__ float* fwHi, __ubuf__ float* fwLo,
                                                                          int64_t rows, int64_t len)
{
    LocalTensor<float> dfLocal = dfBuf_.template Get<float>();
    constexpr int64_t slotSpan = static_cast<int64_t>(VL_FP32) * static_cast<int64_t>(ROW_SLOT_ELEM);
    __ubuf__ float* rowHi = (__ubuf__ float*)dfLocal.GetPhyAddr();
    __ubuf__ float* rowLo = rowHi + 2 * slotSpan;

    int64_t strideF = StrideOfF32(len);
    uint32_t vfLen = VL_FP32;
    // 行内按对齐长度规约: 补齐位上 ce/fw 由 CeFwVec 写成 0(求和单位元), 但不能越过行尾读到下一行
    uint16_t repeatTimes = static_cast<uint16_t>(strideF / vfLen);
    uint32_t tailNum = static_cast<uint32_t>(strideF % vfLen);
    uint16_t tailLoop = tailNum != 0 ? 1 : 0;

    for (int64_t g = 0; g < rows; g += static_cast<int64_t>(VL_FP32)) {
        int64_t left = rows - g;
        uint16_t cnt = static_cast<uint16_t>(left < static_cast<int64_t>(VL_FP32) ? left :
                                                                                    static_cast<int64_t>(VL_FP32));
        uint32_t cntMask = static_cast<uint32_t>(cnt);
        uint32_t oneCnt = 1U;
        __ubuf__ float* grpSrcCe = srcCe + g * strideF;
        __ubuf__ float* grpSrcFw = srcFw + g * strideF;

        __VEC_SCOPE__
        {
            RegTensor<float> hi0;
            RegTensor<float> lo0;
            RegTensor<float> sR0;
            RegTensor<float> eR0;
            RegTensor<float> bR0;
            RegTensor<float> tR0;
            RegTensor<float> rR0;
            RegTensor<float> vR0;
            RegTensor<float> hi1;
            RegTensor<float> lo1;
            RegTensor<float> sR1;
            RegTensor<float> eR1;
            RegTensor<float> bR1;
            RegTensor<float> tR1;
            RegTensor<float> rR1;
            RegTensor<float> vR1;
            RegTensor<float> fwBlockLo;
            RegTensor<float> zeroReg;
            MaskReg pregFinite;
            MaskReg pregOne = AscendC::Reg::UpdateMask<float>(oneCnt);
            MaskReg pregTail = AscendC::Reg::UpdateMask<float>(tailNum);
            MaskReg pregMain = AscendC::Reg::CreateMask<float, MaskPattern::ALL>();

            AscendC::Reg::Duplicate(zeroReg, 0.0f);
            for (uint16_t i = 0; i < cnt; ++i) {
                AscendC::Reg::Duplicate(hi0, 0.0f);
                AscendC::Reg::Duplicate(lo0, 0.0f);
                AscendC::Reg::Duplicate(hi1, 0.0f);
                AscendC::Reg::Duplicate(lo1, 0.0f);
                for (uint16_t j = 0; j < repeatTimes + tailLoop; ++j) {
                    MaskReg preg = (j < repeatTimes) ? pregMain : pregTail;
                    AscendC::Reg::AddrReg offF = AscendC::Reg::CreateAddrReg<float>(i, static_cast<uint32_t>(strideF),
                                                                                    j, vfLen);
                    AscendC::Reg::LoadAlign(vR0, grpSrcCe, offF);
                    AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM>(rR0, vR0, preg);
                    AscendC::Reg::LoadAlign(vR1, grpSrcFw, offF);
                    if constexpr (compensateFw) {
                        ReduceFwWithResidual(rR1, fwBlockLo, vR1, zeroReg, preg);
                    } else {
                        AscendC::Reg::Reduce<AscendC::Reg::ReduceType::SUM>(rR1, vR1, preg);
                    }
                    AscendC::Reg::Add(sR0, hi0, rR0, pregMain);
                    AscendC::Reg::Sub(bR0, sR0, hi0, pregMain);
                    AscendC::Reg::Sub(tR0, sR0, bR0, pregMain);
                    AscendC::Reg::Sub(tR0, hi0, tR0, pregMain);
                    AscendC::Reg::Sub(eR0, rR0, bR0, pregMain);
                    AscendC::Reg::Add(eR0, tR0, eR0, pregMain);
                    AscendC::Reg::Add(eR0, eR0, lo0, pregMain);
                    AscendC::Reg::Compare<float, AscendC::CMPMODE::EQ>(pregFinite, eR0, eR0, pregMain);
                    AscendC::Reg::Select(eR0, eR0, zeroReg, pregFinite);
                    AscendC::Reg::Add(hi0, sR0, eR0, pregMain);
                    AscendC::Reg::Sub(bR0, hi0, sR0, pregMain);
                    AscendC::Reg::Sub(tR0, hi0, bR0, pregMain);
                    AscendC::Reg::Sub(tR0, sR0, tR0, pregMain);
                    AscendC::Reg::Sub(lo0, eR0, bR0, pregMain);
                    AscendC::Reg::Add(lo0, tR0, lo0, pregMain);
                    AscendC::Reg::Compare<float, AscendC::CMPMODE::EQ>(pregFinite, lo0, lo0, pregMain);
                    AscendC::Reg::Select(lo0, lo0, zeroReg, pregFinite);
                    AscendC::Reg::Add(sR1, hi1, rR1, pregMain);
                    AscendC::Reg::Sub(bR1, sR1, hi1, pregMain);
                    AscendC::Reg::Sub(tR1, sR1, bR1, pregMain);
                    AscendC::Reg::Sub(tR1, hi1, tR1, pregMain);
                    AscendC::Reg::Sub(eR1, rR1, bR1, pregMain);
                    AscendC::Reg::Add(eR1, tR1, eR1, pregMain);
                    AscendC::Reg::Add(eR1, eR1, lo1, pregMain);
                    if constexpr (compensateFw) {
                        AscendC::Reg::Add(eR1, eR1, fwBlockLo, pregMain);
                    }
                    AscendC::Reg::Compare<float, AscendC::CMPMODE::EQ>(pregFinite, eR1, eR1, pregMain);
                    AscendC::Reg::Select(eR1, eR1, zeroReg, pregFinite);
                    AscendC::Reg::Add(hi1, sR1, eR1, pregMain);
                    AscendC::Reg::Sub(bR1, hi1, sR1, pregMain);
                    AscendC::Reg::Sub(tR1, hi1, bR1, pregMain);
                    AscendC::Reg::Sub(tR1, sR1, tR1, pregMain);
                    AscendC::Reg::Sub(lo1, eR1, bR1, pregMain);
                    AscendC::Reg::Add(lo1, tR1, lo1, pregMain);
                    AscendC::Reg::Compare<float, AscendC::CMPMODE::EQ>(pregFinite, lo1, lo1, pregMain);
                    AscendC::Reg::Select(lo1, lo1, zeroReg, pregFinite);
                }
                AscendC::Reg::AddrReg offRow = AscendC::Reg::CreateAddrReg<float>(i, ROW_SLOT_ELEM);
                AscendC::Reg::StoreAlign(rowHi + 0 * slotSpan, hi0, offRow, pregOne);
                AscendC::Reg::StoreAlign(rowLo + 0 * slotSpan, lo0, offRow, pregOne);
                AscendC::Reg::StoreAlign(rowHi + 1 * slotSpan, hi1, offRow, pregOne);
                AscendC::Reg::StoreAlign(rowLo + 1 * slotSpan, lo1, offRow, pregOne);
            }
        }
        AscendC::PipeBarrier<PIPE_V>();
        if constexpr (mergeResult) {
            MergeRowSlots2(ceHi, ceLo, fwHi, fwLo, rowHi, rowLo, slotSpan, g, cntMask);
        }
    }
}

template <typename T, typename TW, uint64_t hasWeight>
__aicore__ inline void SoftmaxFocalLossND<T, TW, hasWeight>::DenseRowsVec(__ubuf__ T* pred, __ubuf__ int32_t* target,
                                                                          __ubuf__ TW* weight, __ubuf__ float* output,
                                                                          uint32_t count)
{
    uint16_t repeats = static_cast<uint16_t>((count + VL_FP32 - 1) / VL_FP32);
    float gamma = gamma_;
    float alpha = alpha_;
    __VEC_SCOPE__
    {
        RegTensor<float> p32;
        RegTensor<float> t32;
        RegTensor<float> w32;
        RegTensor<float> ce;
        RegTensor<float> fw;
        RegTensor<float> result;
        for (uint16_t i = 0; i < repeats; ++i) {
            MaskReg mask = UpdateMask<float>(count);
            auto offT = AscendC::Reg::CreateAddrReg<T>(i, VL_FP32);
            auto offW = AscendC::Reg::CreateAddrReg<TW>(i, VL_FP32);
            auto offF = AscendC::Reg::CreateAddrReg<float>(i, VL_FP32);
            LoadPredTargetWeight<true>(p32, t32, w32, pred, target, weight, offT, offW, offF, mask);
            CeFwOfSeg(ce, fw, p32, t32, w32, gamma, alpha, mask);
            // A one-element reduction followed by accumulation into +0 normalizes signed zero.
            AscendC::Reg::Adds(ce, ce, 0.0f, mask);
            AscendC::Reg::Adds(fw, fw, 0.0f, mask);
            AscendC::Reg::Mul(result, ce, fw, mask);
            AscendC::Reg::StoreAlign(output, result, offF, mask);
        }
    }
}

template <typename T, typename TW, uint64_t hasWeight>
__aicore__ inline void SoftmaxFocalLossND<T, TW, hasWeight>::ProcessDenseRows()
{
    int64_t block = static_cast<int64_t>(GetBlockIdx());
    int64_t start = block * blockFactor_;
    int64_t count = block == realCoreNum_ - 1 ? tailBlockFactor_ : blockFactor_;
    for (int64_t off = 0; off < count; off += rowsPerTile_) {
        int64_t len = count - off;
        len = len > rowsPerTile_ ? rowsPerTile_ : len;
        LocalTensor<T> pred = predQue_.template AllocTensor<T>();
        LocalTensor<int32_t> target = targetQue_.template AllocTensor<int32_t>();
        DataCopyExtParams predCopy{1, static_cast<uint32_t>(len * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> predPad{false, 0, 0, 0};
        DataCopyPad(pred, predGm_[start + off], predCopy, predPad);
        DataCopyExtParams targetCopy{1, static_cast<uint32_t>(len * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> targetPad{false, 0, 0, 0};
        DataCopyPad(target, targetGm_[start + off], targetCopy, targetPad);
        predQue_.template EnQue<T>(pred);
        targetQue_.template EnQue<int32_t>(target);

        LocalTensor<TW> weight;
        __ubuf__ TW* weightPtr = nullptr;
        if constexpr (hasWeight == 1) {
            weight = weightQue_.template AllocTensor<TW>();
            DataCopyExtParams weightCopy{1, static_cast<uint32_t>(len * sizeof(TW)), 0, 0, 0};
            DataCopyPadExtParams<TW> weightPad{false, 0, 0, 0};
            DataCopyPad(weight, weightGm_[start + off], weightCopy, weightPad);
            weightQue_.template EnQue<TW>(weight);
            weight = weightQue_.template DeQue<TW>();
            weightPtr = (__ubuf__ TW*)weight.GetPhyAddr();
        }
        pred = predQue_.template DeQue<T>();
        target = targetQue_.template DeQue<int32_t>();
        LocalTensor<float> yF32 = yF32Buf_.template Get<float>();
        DenseRowsVec((__ubuf__ T*)pred.GetPhyAddr(), (__ubuf__ int32_t*)target.GetPhyAddr(), weightPtr,
                     (__ubuf__ float*)yF32.GetPhyAddr(), static_cast<uint32_t>(len));
        predQue_.FreeTensor(pred);
        targetQue_.FreeTensor(target);
        if constexpr (hasWeight == 1) {
            weightQue_.FreeTensor(weight);
        }

        LocalTensor<T> output = yQue_.template AllocTensor<T>();
        if constexpr (sizeof(T) == sizeof(half)) {
            Cast(output, yF32, RoundMode::CAST_RINT, static_cast<int32_t>(len));
        } else {
            Copy(output, yF32, static_cast<int32_t>(len));
        }
        yQue_.template EnQue<T>(output);
        output = yQue_.template DeQue<T>();
        DataCopyExtParams outputCopy{1, static_cast<uint32_t>(len * sizeof(T)), 0, 0, 0};
        DataCopyPad(yGm_[start + off], output, outputCopy);
        yQue_.FreeTensor(output);
    }
}

} // namespace SoftmaxFocalLoss

#endif // SOFTMAX_FOCAL_LOSS_ND_VEC_H
