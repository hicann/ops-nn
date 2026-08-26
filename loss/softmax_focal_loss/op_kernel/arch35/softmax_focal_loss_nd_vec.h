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
    MaskReg preg)
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

    AscendC::Reg::StoreAlign(ceTail, ceReg, preg);
    AscendC::Reg::StoreAlign(fwTail, fwReg, preg);
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
    // 末段按对齐长度参与计算与落盘: padding 位上 ce/fw 恒为 0, 正好是求和单位元
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
        MaskReg pregTail = UpdateMask<float>(tailAlign);

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
                            repeatTimes * vfLen, pregTail);
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

} // namespace SoftmaxFocalLoss

#endif // SOFTMAX_FOCAL_LOSS_ND_VEC_H
