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
 *   wb = alpha * exp((gamma-1)   * log(1-p)) * t     WB = sum_j wb
 *   ce = -log(p) * t * w                             CE = sum_j ce
 *   wt = w * t                                       W  = sum_j wt
 *   d_ce = p * W - t * w
 *   d_wf = -gamma * ((WF - WB) + wb) * p
 *   grad = (d_ce * WF + d_wf * CE) * dout * coef     coef: mean 为 1/numel, 否则 1.0
 *
 * 调度: 行分核, 核内按行块推进, 行内按列块推进(列块数为 1 时即全载路径)。
 * 两趟: 第一趟求四个行标量, 第二趟逐元素组合出 grad(wb 在第二趟重算, 不占额外缓冲)。
 */

#ifndef SOFTMAX_FOCAL_LOSS_GRAD_ND_VEC_H
#define SOFTMAX_FOCAL_LOSS_GRAD_ND_VEC_H

#include "softmax_focal_loss_grad_nd.h"

namespace SoftmaxFocalLossGrad {

template <typename T, typename TW, uint64_t hasWeight>
__aicore__ inline void SoftmaxFocalLossGradND<T, TW, hasWeight>::LoadPTW(
    RegTensor<float>& p32, RegTensor<float>& t32, RegTensor<float>& w32, __ubuf__ T* predAddr,
    __ubuf__ int32_t* targetAddr, __ubuf__ TW* weightAddr, AscendC::MicroAPI::AddrReg offT,
    AscendC::MicroAPI::AddrReg offW, AscendC::MicroAPI::AddrReg offF, MaskReg preg)
{
    RegTensor<T> predReg;
    RegTensor<int32_t> targetRegI32;
    RegTensor<TW> weightReg;

    if constexpr (sizeof(T) == sizeof(half)) {
        AscendC::MicroAPI::LoadAlign<T, LoadDist::DIST_UNPACK_B16>(predReg, predAddr, offT);
        AscendC::MicroAPI::Cast<float, T, castB16ToB32>(p32, predReg, preg);
    } else {
        AscendC::MicroAPI::LoadAlign(p32, predAddr, offT);
    }
    AscendC::MicroAPI::LoadAlign(targetRegI32, targetAddr, offF);
    AscendC::MicroAPI::Cast<float, int32_t, castI32ToF32>(t32, targetRegI32, preg);
    if constexpr (hasWeight == 1) {
        if constexpr (sizeof(TW) == sizeof(half)) {
            AscendC::MicroAPI::LoadAlign<TW, LoadDist::DIST_UNPACK_B16>(weightReg, weightAddr, offW);
            AscendC::MicroAPI::Cast<float, TW, castB16ToB32>(w32, weightReg, preg);
        } else {
            AscendC::MicroAPI::LoadAlign(w32, weightAddr, offW);
        }
    } else {
        AscendC::MicroAPI::Duplicate(w32, 1.0f);
    }
}

template <typename T, typename TW, uint64_t hasWeight>
__aicore__ inline void SoftmaxFocalLossGradND<T, TW, hasWeight>::SumsOfSeg(
    __ubuf__ float* wfAddr, __ubuf__ float* wbAddr, __ubuf__ float* ceAddr, __ubuf__ float* wtAddr,
    RegTensor<float>& p32, RegTensor<float>& t32, RegTensor<float>& w32, AscendC::MicroAPI::AddrReg offF, MaskReg preg)
{
    RegTensor<float> oneReg;
    RegTensor<float> l1p;
    RegTensor<float> tmpReg;
    RegTensor<float> outReg;
    float gamma = gamma_;
    float gammaSub1 = gamma_ - 1.0f;
    float alpha = alpha_;

    // l1p = log(1 - p)
    AscendC::MicroAPI::Duplicate(oneReg, 1.0f);
    AscendC::MicroAPI::Sub(tmpReg, oneReg, p32, preg);
    AscendC::MicroAPI::Log(l1p, tmpReg, preg);

    // wf = alpha * exp(gamma * l1p) * t
    AscendC::MicroAPI::Muls(tmpReg, l1p, gamma, preg);
    AscendC::MicroAPI::Exp(outReg, tmpReg, preg);
    AscendC::MicroAPI::Muls(outReg, outReg, alpha, preg);
    AscendC::MicroAPI::Mul(outReg, outReg, t32, preg);
    AscendC::MicroAPI::StoreAlign(wfAddr, outReg, offF, preg);

    // wb = alpha * exp((gamma - 1) * l1p) * t
    AscendC::MicroAPI::Muls(tmpReg, l1p, gammaSub1, preg);
    AscendC::MicroAPI::Exp(outReg, tmpReg, preg);
    AscendC::MicroAPI::Muls(outReg, outReg, alpha, preg);
    AscendC::MicroAPI::Mul(outReg, outReg, t32, preg);
    AscendC::MicroAPI::StoreAlign(wbAddr, outReg, offF, preg);

    // ce = -log(p) * t * w
    AscendC::MicroAPI::Log(tmpReg, p32, preg);
    AscendC::MicroAPI::Muls(outReg, tmpReg, -1.0f, preg);
    AscendC::MicroAPI::Mul(outReg, outReg, t32, preg);
    AscendC::MicroAPI::Mul(outReg, outReg, w32, preg);
    AscendC::MicroAPI::StoreAlign(ceAddr, outReg, offF, preg);

    // wt = w * t
    AscendC::MicroAPI::Mul(outReg, w32, t32, preg);
    AscendC::MicroAPI::StoreAlign(wtAddr, outReg, offF, preg);
}

template <typename T, typename TW, uint64_t hasWeight>
__aicore__ inline void SoftmaxFocalLossGradND<T, TW, hasWeight>::SumsVec(
    __ubuf__ T* predAddr, __ubuf__ int32_t* targetAddr, __ubuf__ TW* weightAddr, __ubuf__ float* wfAddr,
    __ubuf__ float* wbAddr, __ubuf__ float* ceAddr, __ubuf__ float* wtAddr, int64_t rows, int64_t len)
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

    __VEC_SCOPE__
    {
        RegTensor<float> p32;
        RegTensor<float> t32;
        RegTensor<float> w32;

        MaskReg pregMain = AscendC::MicroAPI::CreateMask<float, MaskPattern::ALL>();
        MaskReg pregTail = UpdateMask<float>(tailAlign);

        for (uint16_t i = 0; i < aTimes; ++i) {
            for (uint16_t j = 0; j < repeatTimes + tailLoop; ++j) {
                MaskReg preg = (j < repeatTimes) ? pregMain : pregTail;
                AscendC::MicroAPI::AddrReg offT = AscendC::MicroAPI::CreateAddrReg<T>(i, static_cast<uint32_t>(strideT),
                                                                                      j, vfLen);
                AscendC::MicroAPI::AddrReg offW = AscendC::MicroAPI::CreateAddrReg<TW>(
                    i, static_cast<uint32_t>(strideW), j, vfLen);
                AscendC::MicroAPI::AddrReg offF = AscendC::MicroAPI::CreateAddrReg<float>(
                    i, static_cast<uint32_t>(strideF), j, vfLen);

                LoadPTW(p32, t32, w32, predAddr, targetAddr, weightAddr, offT, offW, offF, preg);
                SumsOfSeg(wfAddr, wbAddr, ceAddr, wtAddr, p32, t32, w32, offF, preg);
            }
        }
    }
}

template <typename T, typename TW, uint64_t hasWeight>
__aicore__ inline void SoftmaxFocalLossGradND<T, TW, hasWeight>::ComputeSums(int64_t rows, int64_t len)
{
    LocalTensor<T> predBuf = predQue_.template DeQue<T>();
    LocalTensor<int32_t> targetBuf = targetQue_.template DeQue<int32_t>();
    LocalTensor<T> doutBuf = doutQue_.template DeQue<T>();
    LocalTensor<float> wfBuf = wfBuf_.template Get<float>();
    LocalTensor<float> wbBuf = wbBuf_.template Get<float>();
    LocalTensor<float> ceBuf = ceBuf_.template Get<float>();
    LocalTensor<float> wtBuf = wtBuf_.template Get<float>();

    LocalTensor<TW> weightBuf;
    __ubuf__ TW* weightAddr = nullptr;
    if constexpr (hasWeight == 1) {
        weightBuf = weightQue_.template DeQue<TW>();
        weightAddr = (__ubuf__ TW*)weightBuf.GetPhyAddr();
    }

    SumsVec((__ubuf__ T*)predBuf.GetPhyAddr(), (__ubuf__ int32_t*)targetBuf.GetPhyAddr(), weightAddr,
            (__ubuf__ float*)wfBuf.GetPhyAddr(), (__ubuf__ float*)wbBuf.GetPhyAddr(),
            (__ubuf__ float*)ceBuf.GetPhyAddr(), (__ubuf__ float*)wtBuf.GetPhyAddr(), rows, len);

    predQue_.FreeTensor(predBuf);
    targetQue_.FreeTensor(targetBuf);
    doutQue_.FreeTensor(doutBuf);
    if constexpr (hasWeight == 1) {
        weightQue_.FreeTensor(weightBuf);
    }
}

template <typename T, typename TW, uint64_t hasWeight>
__aicore__ inline void SoftmaxFocalLossGradND<T, TW, hasWeight>::MakeSegOffsets(
    AscendC::MicroAPI::AddrReg& offT, AscendC::MicroAPI::AddrReg& offW, AscendC::MicroAPI::AddrReg& offF,
    AscendC::MicroAPI::AddrReg& offO, uint16_t i, uint16_t j, int64_t strideT, int64_t strideW, int64_t strideF,
    uint32_t vfLen)
{
    // offO 走 T 的行距(grad 落 fp32 缓冲但按输出行布局), 与 offF 的 fp32 行距区分
    offT = AscendC::MicroAPI::CreateAddrReg<T>(i, static_cast<uint32_t>(strideT), j, vfLen);
    offW = AscendC::MicroAPI::CreateAddrReg<TW>(i, static_cast<uint32_t>(strideW), j, vfLen);
    offF = AscendC::MicroAPI::CreateAddrReg<float>(i, static_cast<uint32_t>(strideF), j, vfLen);
    offO = AscendC::MicroAPI::CreateAddrReg<float>(i, static_cast<uint32_t>(strideT), j, vfLen);
}

template <typename T, typename TW, uint64_t hasWeight>
__aicore__ inline void SoftmaxFocalLossGradND<T, TW, hasWeight>::LoadRowScalars(
    RegTensor<float>& wfB, RegTensor<float>& wbB, RegTensor<float>& ceB, RegTensor<float>& wB, __ubuf__ float* accAddr,
    int64_t accStride, uint16_t rowIdx)
{
    // 行标量广播: WF / WB / CE / W
    AscendC::MicroAPI::LoadAlign<float, LoadDist::DIST_BRC_B32>(wfB, accAddr + rowIdx);
    AscendC::MicroAPI::LoadAlign<float, LoadDist::DIST_BRC_B32>(wbB, accAddr + accStride + rowIdx);
    AscendC::MicroAPI::LoadAlign<float, LoadDist::DIST_BRC_B32>(ceB, accAddr + 2 * accStride + rowIdx);
    AscendC::MicroAPI::LoadAlign<float, LoadDist::DIST_BRC_B32>(wB, accAddr + 3 * accStride + rowIdx);
}

template <typename T, typename TW, uint64_t hasWeight>
__aicore__ inline void SoftmaxFocalLossGradND<T, TW, hasWeight>::GradOfSeg(
    __ubuf__ float* gradAddr, RegTensor<float>& p32, RegTensor<float>& t32, RegTensor<float>& w32,
    RegTensor<float>& d32, RegTensor<float>& wfB, RegTensor<float>& wbB, RegTensor<float>& ceB, RegTensor<float>& wB,
    AscendC::MicroAPI::AddrReg offO, MaskReg preg)
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
    AscendC::MicroAPI::Mul(dCe, p32, wB, preg);
    AscendC::MicroAPI::Mul(tmpReg, t32, w32, preg);
    AscendC::MicroAPI::Sub(dCe, dCe, tmpReg, preg);

    // wb(逐元素) = alpha * exp((gamma - 1) * log(1 - p)) * t
    AscendC::MicroAPI::Duplicate(oneReg, 1.0f);
    AscendC::MicroAPI::Sub(tmpReg, oneReg, p32, preg);
    AscendC::MicroAPI::Log(tmpReg, tmpReg, preg);
    AscendC::MicroAPI::Muls(tmpReg, tmpReg, gammaSub1, preg);
    AscendC::MicroAPI::Exp(tmpReg, tmpReg, preg);
    AscendC::MicroAPI::Muls(tmpReg, tmpReg, alpha, preg);
    AscendC::MicroAPI::Mul(tmpReg, tmpReg, t32, preg);

    // d_wf = -gamma * ((WF - WB) + wb) * p
    AscendC::MicroAPI::Sub(dWf, wfB, wbB, preg);
    AscendC::MicroAPI::Add(dWf, dWf, tmpReg, preg);
    AscendC::MicroAPI::Mul(dWf, dWf, p32, preg);
    AscendC::MicroAPI::Muls(dWf, dWf, gammaNeg, preg);

    // grad = (d_ce * WF + d_wf * CE) * dout * coef
    AscendC::MicroAPI::Mul(dCe, dCe, wfB, preg);
    AscendC::MicroAPI::Mul(dWf, dWf, ceB, preg);
    AscendC::MicroAPI::Add(dCe, dCe, dWf, preg);
    AscendC::MicroAPI::Mul(dCe, dCe, d32, preg);
    AscendC::MicroAPI::Muls(dCe, dCe, coef, preg);
    AscendC::MicroAPI::StoreAlign(gradAddr, dCe, offO, preg);
}

template <typename T, typename TW, uint64_t hasWeight>
__aicore__ inline void SoftmaxFocalLossGradND<T, TW, hasWeight>::GradVec(
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

    __VEC_SCOPE__
    {
        RegTensor<T> doutReg;
        RegTensor<float> p32;
        RegTensor<float> t32;
        RegTensor<float> w32;
        RegTensor<float> d32;
        RegTensor<float> wfB;
        RegTensor<float> wbB;
        RegTensor<float> ceB;
        RegTensor<float> wB;

        MaskReg pregMain = AscendC::MicroAPI::CreateMask<float, MaskPattern::ALL>();
        MaskReg pregTail = UpdateMask<float>(tailAlign);

        for (uint16_t i = 0; i < aTimes; ++i) {
            LoadRowScalars(wfB, wbB, ceB, wB, accAddr, accStride, i);

            for (uint16_t j = 0; j < repeatTimes + tailLoop; ++j) {
                MaskReg preg = (j < repeatTimes) ? pregMain : pregTail;
                AscendC::MicroAPI::AddrReg offT;
                AscendC::MicroAPI::AddrReg offW;
                AscendC::MicroAPI::AddrReg offF;
                AscendC::MicroAPI::AddrReg offO;
                MakeSegOffsets(offT, offW, offF, offO, i, j, strideT, strideW, strideF, vfLen);

                LoadPTW(p32, t32, w32, predAddr, targetAddr, weightAddr, offT, offW, offF, preg);
                if constexpr (sizeof(T) == sizeof(half)) {
                    AscendC::MicroAPI::LoadAlign<T, LoadDist::DIST_UNPACK_B16>(doutReg, doutAddr, offT);
                    AscendC::MicroAPI::Cast<float, T, castB16ToB32>(d32, doutReg, preg);
                } else {
                    AscendC::MicroAPI::LoadAlign(d32, doutAddr, offT);
                }

                GradOfSeg(gradAddr, p32, t32, w32, d32, wfB, wbB, ceB, wB, offO, preg);
            }
        }
    }
}

template <typename T, typename TW, uint64_t hasWeight>
__aicore__ inline void SoftmaxFocalLossGradND<T, TW, hasWeight>::ComputeGrad(int64_t rows, int64_t len)
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
