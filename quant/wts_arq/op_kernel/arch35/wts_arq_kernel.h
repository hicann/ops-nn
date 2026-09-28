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
 * \file wts_arq_kernel.h
 * \brief WtsARQ RegBase kernel for arch35 (DAV_3510 / Ascend 950).
 *
 * Compute chain (DESIGN.md §3.5/§3.6, all in fp32 on registers):
 *   1. range fix:      w_min = min(w_min, 0), w_max = max(w_max, 0)
 *   2. scale:          offset_flag=true  -> w_max/255 - w_min/255 (div before sub)
 *                      offset_flag=false -> max(|w_min|/128, w_max/127)
 *   3. eps fallback:   scale < eps -> 1.0 (Compares<LT> + Select)
 *   4. offset (true):  offset = -rint(w_min/scale) - 128 (fused in QuantOffsetVF)
 *   5. quant:          q = clamp(rint(w/scale) [+ offset], -128, 127) [- offset]
 *   6. dequant:        y = q * scale
 *
 * CopyIn for w_min/w_max is dispatched at runtime by brcMode:
 *   0 = plain DataCopyPad (no broadcast axis), 1 = NDDMA hardware broadcast
 *   (stride=0 axes, WithoutLoop/WithLoop by schMode), 2 = compact copy +
 *   UB Broadcast dynamic API. OneDim (shapeLen == 1) scalar broadcast is
 *   expanded with Reg::Duplicate.
 */
#ifndef WTS_ARQ_KERNEL_H_
#define WTS_ARQ_KERNEL_H_

#include "kernel_operator.h"
#include "adv_api/pad/broadcast.h"
#include "wts_arq_tiling_data.h"
#include "wts_arq_tiling_key.h"

#include <type_traits>

namespace WtsArq {

constexpr uint32_t WTS_ARQ_NDDMA_DIMS = 5; // NDDMA max dims

// High-precision fp32 division (error-compensated, 0 ulp for normal values).
// The intrinsic vdiv rounds differently from the CPU oracle: for w/scale values
// that fall just below a half integer (e.g. 63.49999778) it returns the half
// value itself (63.5), so CAST_RINT rounds up instead of down and the quantized
// result drifts by one step. Scale and quant divisions must match the delivered
// golden's fp32 semantics, so both use the precision mode.
constexpr AscendC::Reg::DivSpecificMode WTS_ARQ_DIV_0ULP = {AscendC::Reg::MaskMergeMode::ZEROING, true,
                                                            AscendC::DivAlgo::PRECISION_0ULP_FTZ_TRUE};

__simd_vf__ inline void ClipRangeVF(__ubuf__ float* minAddr, __ubuf__ float* maxAddr, uint32_t count,
                                    uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<float> minReg;
        AscendC::Reg::RegTensor<float> maxReg;
        AscendC::Reg::RegTensor<float> outReg;
        AscendC::Reg::MaskReg mask;
        AscendC::Reg::AddrReg aReg;
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            aReg = AscendC::Reg::CreateAddrReg<float>(i, oneRepeatSize);
            mask = AscendC::Reg::UpdateMask<float>(count);
            AscendC::Reg::LoadAlign(minReg, minAddr, aReg);
            AscendC::Reg::Mins<float>(outReg, minReg, 0.0f, mask);
            AscendC::Reg::StoreAlign(minAddr, outReg, aReg, mask);
            AscendC::Reg::LoadAlign(maxReg, maxAddr, aReg);
            AscendC::Reg::Maxs<float>(outReg, maxReg, 0.0f, mask);
            AscendC::Reg::StoreAlign(maxAddr, outReg, aReg, mask);
        }
    }
}

// offset_flag = false: scale = max(|w_min| / 128, w_max / 127)
__simd_vf__ inline void ScaleSymVF(__ubuf__ float* scaleAddr, __ubuf__ float* minAddr, __ubuf__ float* maxAddr,
                                   uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<float> minReg;
        AscendC::Reg::RegTensor<float> maxReg;
        AscendC::Reg::RegTensor<float> absMinReg;
        AscendC::Reg::RegTensor<float> scaleLowReg;
        AscendC::Reg::RegTensor<float> scaleHighReg;
        AscendC::Reg::RegTensor<float> scaleReg;
        AscendC::Reg::RegTensor<float> div127Reg;
        AscendC::Reg::MaskReg mask;
        AscendC::Reg::AddrReg aReg;
        AscendC::Reg::Duplicate(div127Reg, 127.0f);
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            aReg = AscendC::Reg::CreateAddrReg<float>(i, oneRepeatSize);
            mask = AscendC::Reg::UpdateMask<float>(count);
            AscendC::Reg::LoadAlign(minReg, minAddr, aReg);
            AscendC::Reg::LoadAlign(maxReg, maxAddr, aReg);
            AscendC::Reg::Abs<float, AscendC::Reg::MaskMergeMode::ZEROING>(absMinReg, minReg, mask);
            // /128 is exact under Muls; /127 must use 0 ulp division (see WTS_ARQ_DIV_0ULP).
            AscendC::Reg::Muls<float>(scaleLowReg, absMinReg, 1.0f / 128.0f, mask);
            AscendC::Reg::Div<float, &WTS_ARQ_DIV_0ULP>(scaleHighReg, maxReg, div127Reg, mask);
            AscendC::Reg::Max<float>(scaleReg, scaleLowReg, scaleHighReg, mask);
            AscendC::Reg::StoreAlign(scaleAddr, scaleReg, aReg, mask);
        }
    }
}

// offset_flag = true: scale = w_max/255 - w_min/255 (divide before subtract, spec
// numerical_stability.div_before_subtract_overflow_guard; never compute w_max - w_min first)
__simd_vf__ inline void ScaleOffsetVF(__ubuf__ float* scaleAddr, __ubuf__ float* minAddr, __ubuf__ float* maxAddr,
                                      uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<float> minReg;
        AscendC::Reg::RegTensor<float> maxReg;
        AscendC::Reg::RegTensor<float> scaleLowReg;
        AscendC::Reg::RegTensor<float> scaleHighReg;
        AscendC::Reg::RegTensor<float> scaleReg;
        AscendC::Reg::RegTensor<float> div255Reg;
        AscendC::Reg::MaskReg mask;
        AscendC::Reg::AddrReg aReg;
        AscendC::Reg::Duplicate(div255Reg, 255.0f);
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            aReg = AscendC::Reg::CreateAddrReg<float>(i, oneRepeatSize);
            mask = AscendC::Reg::UpdateMask<float>(count);
            AscendC::Reg::LoadAlign(minReg, minAddr, aReg);
            AscendC::Reg::LoadAlign(maxReg, maxAddr, aReg);
            // 0 ulp division keeps w_max/255 - w_min/255 bit-consistent with the golden.
            AscendC::Reg::Div<float, &WTS_ARQ_DIV_0ULP>(scaleHighReg, maxReg, div255Reg, mask);
            AscendC::Reg::Div<float, &WTS_ARQ_DIV_0ULP>(scaleLowReg, minReg, div255Reg, mask);
            AscendC::Reg::Sub<float>(scaleReg, scaleHighReg, scaleLowReg, mask);
            AscendC::Reg::StoreAlign(scaleAddr, scaleReg, aReg, mask);
        }
    }
}

// scale < eps -> scale = 1.0 (spec numerical_stability.scale_eps_fallback)
__simd_vf__ inline void ScaleEpsSelectVF(__ubuf__ float* scaleAddr, float eps, uint32_t count, uint32_t oneRepeatSize,
                                         uint16_t repeatTimes)
{
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<float> scaleReg;
        AscendC::Reg::RegTensor<float> oneReg;
        AscendC::Reg::RegTensor<float> outReg;
        AscendC::Reg::MaskReg mask;
        AscendC::Reg::MaskReg ltMask;
        AscendC::Reg::AddrReg aReg;
        AscendC::Reg::Duplicate(oneReg, 1.0f);
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            aReg = AscendC::Reg::CreateAddrReg<float>(i, oneRepeatSize);
            mask = AscendC::Reg::UpdateMask<float>(count);
            AscendC::Reg::LoadAlign(scaleReg, scaleAddr, aReg);
            AscendC::Reg::Compares<float, AscendC::CMPMODE::LT>(ltMask, scaleReg, eps, mask);
            AscendC::Reg::Select<float>(outReg, oneReg, scaleReg, ltMask);
            AscendC::Reg::StoreAlign(scaleAddr, outReg, aReg, mask);
        }
    }
}

// offset_flag = false: q = clamp(rint(w / scale), -128, 127)
__simd_vf__ inline void QuantSymVF(__ubuf__ float* dstAddr, __ubuf__ float* wAddr, __ubuf__ float* scaleAddr,
                                   uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<float> wReg;
        AscendC::Reg::RegTensor<float> scaleReg;
        AscendC::Reg::RegTensor<float> rawReg;
        AscendC::Reg::RegTensor<float> roundReg;
        AscendC::Reg::RegTensor<float> clampReg;
        AscendC::Reg::MaskReg mask;
        AscendC::Reg::AddrReg aReg;
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            aReg = AscendC::Reg::CreateAddrReg<float>(i, oneRepeatSize);
            mask = AscendC::Reg::UpdateMask<float>(count);
            AscendC::Reg::LoadAlign(wReg, wAddr, aReg);
            AscendC::Reg::LoadAlign(scaleReg, scaleAddr, aReg);
            AscendC::Reg::Div<float, &WTS_ARQ_DIV_0ULP>(rawReg, wReg, scaleReg, mask);
            AscendC::Reg::Truncate<float, AscendC::RoundMode::CAST_RINT>(roundReg, rawReg, mask);
            AscendC::Reg::Maxs<float>(clampReg, roundReg, -128.0f, mask);
            AscendC::Reg::Mins<float>(clampReg, clampReg, 127.0f, mask);
            AscendC::Reg::StoreAlign(dstAddr, clampReg, aReg, mask);
        }
    }
}

// offset_flag = true: offset = -rint(w_min/scale) - 128 (computed inline per element);
// q = clamp(rint(w/scale) + offset, -128, 127) - offset
__simd_vf__ inline void QuantOffsetVF(__ubuf__ float* dstAddr, __ubuf__ float* wAddr, __ubuf__ float* minAddr,
                                      __ubuf__ float* scaleAddr, uint32_t count, uint32_t oneRepeatSize,
                                      uint16_t repeatTimes)
{
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<float> wReg;
        AscendC::Reg::RegTensor<float> minReg;
        AscendC::Reg::RegTensor<float> scaleReg;
        AscendC::Reg::RegTensor<float> offsetRawReg;
        AscendC::Reg::RegTensor<float> offsetReg;
        AscendC::Reg::RegTensor<float> rawReg;
        AscendC::Reg::RegTensor<float> yReg;
        AscendC::Reg::RegTensor<float> clampReg;
        AscendC::Reg::MaskReg mask;
        AscendC::Reg::AddrReg aReg;
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            aReg = AscendC::Reg::CreateAddrReg<float>(i, oneRepeatSize);
            mask = AscendC::Reg::UpdateMask<float>(count);
            AscendC::Reg::LoadAlign(wReg, wAddr, aReg);
            AscendC::Reg::LoadAlign(minReg, minAddr, aReg);
            AscendC::Reg::LoadAlign(scaleReg, scaleAddr, aReg);

            AscendC::Reg::Div<float, &WTS_ARQ_DIV_0ULP>(offsetRawReg, minReg, scaleReg, mask);
            AscendC::Reg::Truncate<float, AscendC::RoundMode::CAST_RINT>(offsetReg, offsetRawReg, mask);
            AscendC::Reg::Muls<float>(offsetReg, offsetReg, -1.0f, mask);
            AscendC::Reg::Adds<float>(offsetReg, offsetReg, -128.0f, mask);

            AscendC::Reg::Div<float, &WTS_ARQ_DIV_0ULP>(rawReg, wReg, scaleReg, mask);
            AscendC::Reg::Truncate<float, AscendC::RoundMode::CAST_RINT>(yReg, rawReg, mask);
            AscendC::Reg::Add<float>(yReg, yReg, offsetReg, mask);
            AscendC::Reg::Maxs<float>(clampReg, yReg, -128.0f, mask);
            AscendC::Reg::Mins<float>(clampReg, clampReg, 127.0f, mask);
            AscendC::Reg::Sub<float>(yReg, clampReg, offsetReg, mask);
            AscendC::Reg::StoreAlign(dstAddr, yReg, aReg, mask);
        }
    }
}

// y = q * scale
__simd_vf__ inline void DequantVF(__ubuf__ float* dstAddr, __ubuf__ float* quantAddr, __ubuf__ float* scaleAddr,
                                  uint32_t count, uint32_t oneRepeatSize, uint16_t repeatTimes)
{
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<float> quantReg;
        AscendC::Reg::RegTensor<float> scaleReg;
        AscendC::Reg::RegTensor<float> outReg;
        AscendC::Reg::MaskReg mask;
        AscendC::Reg::AddrReg aReg;
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            aReg = AscendC::Reg::CreateAddrReg<float>(i, oneRepeatSize);
            mask = AscendC::Reg::UpdateMask<float>(count);
            AscendC::Reg::LoadAlign(quantReg, quantAddr, aReg);
            AscendC::Reg::LoadAlign(scaleReg, scaleAddr, aReg);
            AscendC::Reg::Mul<float>(outReg, quantReg, scaleReg, mask);
            AscendC::Reg::StoreAlign(dstAddr, outReg, aReg, mask);
        }
    }
}

// OneDim scalar broadcast: expand a scalar to fill the whole tile buffer
__simd_vf__ inline void DuplicateFillVF(__ubuf__ float* dstAddr, float value, uint32_t count, uint32_t oneRepeatSize,
                                        uint16_t repeatTimes)
{
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<float> valReg;
        AscendC::Reg::MaskReg mask;
        AscendC::Reg::AddrReg aReg;
        AscendC::Reg::Duplicate(valReg, value);
        for (uint16_t i = 0; i < repeatTimes; ++i) {
            aReg = AscendC::Reg::CreateAddrReg<float>(i, oneRepeatSize);
            mask = AscendC::Reg::UpdateMask<float>(count);
            AscendC::Reg::StoreAlign(dstAddr, valReg, aReg, mask);
        }
    }
}

template <typename T, uint32_t RANK>
class WtsArqKernel {
    static constexpr uint32_t VL = AscendC::GetVecLen() / sizeof(float);

public:
    __aicore__ inline void Init(GM_ADDR w, GM_ADDR wMin, GM_ADDR wMax, GM_ADDR y, const WtsArqTilingData<RANK>* td)
    {
        td_ = td;
        gmW_.SetGlobalBuffer((__gm__ T*)w);
        gmMin_.SetGlobalBuffer((__gm__ T*)wMin);
        gmMax_.SetGlobalBuffer((__gm__ T*)wMax);
        gmY_.SetGlobalBuffer((__gm__ T*)y);

        const int64_t perBufElems = td_->perBufElems;
        pipe_.InitBuffer(bufW_, perBufElems * sizeof(float));
        pipe_.InitBuffer(bufMin_, perBufElems * sizeof(float));
        pipe_.InitBuffer(bufMax_, perBufElems * sizeof(float));
        pipe_.InitBuffer(bufScale_, perBufElems * sizeof(float));
        pipe_.InitBuffer(bufCast_, perBufElems * sizeof(T));

        // w / y are contiguous ND: derive strides from collapsed dims
        wStrides_[RANK - 1] = 1;
        for (int64_t d = static_cast<int64_t>(RANK) - 2; d >= 0; d--) {
            wStrides_[d] = wStrides_[d + 1] * td_->dims[d + 1];
        }
    }

    __aicore__ inline void Process()
    {
        if (td_->fusedProduct <= 0) {
            return; // empty tensor: host does not launch real work
        }
        const int64_t blockIdx = AscendC::GetBlockIdx();
        if (blockIdx >= td_->blockNum) {
            return;
        }
        const int64_t myTiles = (blockIdx == td_->blockNum - 1) ? td_->blockTail : td_->blockFormer;

        const int32_t evMTE2toV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
        const int32_t evVtoMTE2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE2));
        const int32_t evVtoMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
        const int32_t evMTE3toMTE2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
        const int32_t evMTE2toS = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_S));
        const int32_t evVtoS = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_S));

        const int64_t axis = td_->ubSplitAxis;
        int64_t inner = 1;
        for (int64_t d = axis + 1; d < static_cast<int64_t>(RANK); d++) {
            inner *= td_->dims[d];
        }

        int64_t coord[RANK];
        for (int64_t t = 0; t < myTiles; t++) {
            const int64_t flat = blockIdx * td_->blockFormer + t;
            const int64_t aOIdx = flat % td_->ubOuter;
            int64_t outerIdx = flat / td_->ubOuter;
            for (int64_t d = 0; d < static_cast<int64_t>(RANK); d++) {
                coord[d] = 0;
            }
            for (int64_t d = axis - 1; d >= 0; d--) {
                coord[d] = outerIdx % td_->dims[d];
                outerIdx /= td_->dims[d];
            }
            coord[axis] = aOIdx * td_->ubFormer;

            const int64_t aISeg = (aOIdx == td_->ubOuter - 1) ? td_->ubTail : td_->ubFormer;
            const int64_t count = aISeg * inner;
            const int64_t wOff = CalcOffset(coord, wStrides_);
            const int64_t minOff = CalcOffset(coord, td_->minStrides);
            const int64_t maxOff = CalcOffset(coord, td_->maxStrides);
            const uint32_t uCount = static_cast<uint32_t>(count);
            const uint16_t repeatTimes = static_cast<uint16_t>((count + static_cast<int64_t>(VL) - 1) /
                                                               static_cast<int64_t>(VL));

            if (t > 0) {
                AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(evMTE3toMTE2);
            }

            LoadRangeInput(gmMin_, minOff, td_->minStrides, aISeg, count, bufMin_, evMTE2toV, evVtoMTE2, evMTE2toS,
                           evVtoS);
            LoadRangeInput(gmMax_, maxOff, td_->maxStrides, aISeg, count, bufMax_, evMTE2toV, evVtoMTE2, evMTE2toS,
                           evVtoS);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);

            asc_vf_call<ClipRangeVF>((__ubuf__ float*)bufMin_.template Get<float>().GetPhyAddr(),
                                     (__ubuf__ float*)bufMax_.template Get<float>().GetPhyAddr(), uCount, VL,
                                     repeatTimes);
            if (td_->offsetFlag != 0) {
                asc_vf_call<ScaleOffsetVF>((__ubuf__ float*)bufScale_.template Get<float>().GetPhyAddr(),
                                           (__ubuf__ float*)bufMin_.template Get<float>().GetPhyAddr(),
                                           (__ubuf__ float*)bufMax_.template Get<float>().GetPhyAddr(), uCount, VL,
                                           repeatTimes);
            } else {
                asc_vf_call<ScaleSymVF>((__ubuf__ float*)bufScale_.template Get<float>().GetPhyAddr(),
                                        (__ubuf__ float*)bufMin_.template Get<float>().GetPhyAddr(),
                                        (__ubuf__ float*)bufMax_.template Get<float>().GetPhyAddr(), uCount, VL,
                                        repeatTimes);
            }
            asc_vf_call<ScaleEpsSelectVF>((__ubuf__ float*)bufScale_.template Get<float>().GetPhyAddr(), td_->eps,
                                          uCount, VL, repeatTimes);

            LoadW(wOff, count, evMTE2toV, evVtoMTE2);

            if (td_->offsetFlag != 0) {
                asc_vf_call<QuantOffsetVF>((__ubuf__ float*)bufW_.template Get<float>().GetPhyAddr(),
                                           (__ubuf__ float*)bufW_.template Get<float>().GetPhyAddr(),
                                           (__ubuf__ float*)bufMin_.template Get<float>().GetPhyAddr(),
                                           (__ubuf__ float*)bufScale_.template Get<float>().GetPhyAddr(), uCount, VL,
                                           repeatTimes);
            } else {
                asc_vf_call<QuantSymVF>((__ubuf__ float*)bufW_.template Get<float>().GetPhyAddr(),
                                        (__ubuf__ float*)bufW_.template Get<float>().GetPhyAddr(),
                                        (__ubuf__ float*)bufScale_.template Get<float>().GetPhyAddr(), uCount, VL,
                                        repeatTimes);
            }
            asc_vf_call<DequantVF>((__ubuf__ float*)bufW_.template Get<float>().GetPhyAddr(),
                                   (__ubuf__ float*)bufW_.template Get<float>().GetPhyAddr(),
                                   (__ubuf__ float*)bufScale_.template Get<float>().GetPhyAddr(), uCount, VL,
                                   repeatTimes);

            if constexpr (std::is_same_v<T, half>) {
                AscendC::Cast<half, float>(bufCast_.template Get<half>(), bufW_.template Get<float>(),
                                           AscendC::RoundMode::CAST_RINT, uCount);
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evVtoMTE3);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evVtoMTE3);
                CopyOut(wOff, count, bufCast_);
            } else {
                AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evVtoMTE3);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evVtoMTE3);
                CopyOut(wOff, count, bufW_);
            }

            if (t != myTiles - 1) {
                AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(evMTE3toMTE2);
            }
        }
    }

private:
    __aicore__ inline int64_t CalcOffset(const int64_t* coord, const int64_t* strides)
    {
        int64_t offset = 0;
        for (int64_t d = 0; d < static_cast<int64_t>(RANK); d++) {
            offset += coord[d] * strides[d];
        }
        return offset;
    }

    // Linear contiguous DataCopyPad GM -> UB
    __aicore__ inline void CopyInLinear(const AscendC::GlobalTensor<T>& gm, int64_t off, AscendC::LocalTensor<T>& dst,
                                        int64_t count)
    {
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(count * sizeof(T));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        AscendC::DataCopyPadExtParams<T> padParams;
        padParams.isPad = false;
        padParams.leftPadding = 0;
        padParams.rightPadding = 0;
        padParams.paddingValue = 0;
        AscendC::DataCopyPad(dst, gm[off], copyParams, padParams);
    }

    // NDDMA copy of one tile. compact=false: broadcast axes keep full loopSize with
    // srcStride=0 (hardware broadcast into full tile layout); compact=true: broadcast
    // axes collapse to loopSize=1 (compact source layout for a later UB Broadcast).
    // Returns the number of elements written to dst.
    __aicore__ inline int64_t NddmaCopy(AscendC::LocalTensor<T>& dst, const AscendC::GlobalTensor<T>& gm, int64_t gmOff,
                                        const int64_t* strides, int64_t aISeg, bool compact)
    {
        const int64_t axis = td_->ubSplitAxis;
        const int64_t rankI = static_cast<int64_t>(RANK);
        const int64_t innerDims = (rankI - axis) < static_cast<int64_t>(WTS_ARQ_NDDMA_DIMS) ? (rankI - axis) :
                                                                                              WTS_ARQ_NDDMA_DIMS;

        AscendC::MultiCopyParams<T, WTS_ARQ_NDDMA_DIMS> params{};
        int64_t ubCum = 1;
        for (int64_t nd = 0; nd < static_cast<int64_t>(WTS_ARQ_NDDMA_DIMS); nd++) {
            const int64_t d = rankI - 1 - nd;
            int64_t sz = 1;
            int64_t srcStride = 0;
            if (nd < innerDims) {
                sz = (d == axis) ? aISeg : td_->dims[d];
                srcStride = strides[d];
                if (compact && strides[d] == 0) {
                    sz = 1;
                }
            }
            params.loopInfo.loopSize[nd] = static_cast<uint32_t>(sz);
            params.loopInfo.loopSrcStride[nd] = static_cast<uint64_t>(srcStride);
            params.loopInfo.loopDstStride[nd] = static_cast<uint32_t>(ubCum);
            params.loopInfo.loopLpSize[nd] = 0;
            params.loopInfo.loopRpSize[nd] = 0;
            ubCum *= sz;
        }
        const int64_t innerElems = ubCum;

        static constexpr AscendC::NdDmaConfig cfg = {false, AscendC::NdDmaConfig::unsetPad,
                                                     AscendC::NdDmaConfig::unsetPad, false};

        if (rankI - axis <= static_cast<int64_t>(WTS_ARQ_NDDMA_DIMS)) {
            AscendC::DataCopy<T, WTS_ARQ_NDDMA_DIMS, cfg>(dst, gm[gmOff], params);
            return innerElems;
        }

        // WithLoop: inner 5 dims go to NDDMA, outer axes are walked by this loop
        int64_t outerIters = 1;
        for (int64_t d = axis; d < rankI - static_cast<int64_t>(WTS_ARQ_NDDMA_DIMS); d++) {
            int64_t sz = (d == axis) ? aISeg : td_->dims[d];
            if (compact && strides[d] == 0) {
                sz = 1;
            }
            outerIters *= sz;
        }
        for (int64_t oi = 0; oi < outerIters; oi++) {
            int64_t elemAdj = 0;
            int64_t tmp = oi;
            for (int64_t d = rankI - static_cast<int64_t>(WTS_ARQ_NDDMA_DIMS) - 1; d >= axis; d--) {
                int64_t sz = (d == axis) ? aISeg : td_->dims[d];
                if (compact && strides[d] == 0) {
                    sz = 1;
                }
                elemAdj += (tmp % sz) * strides[d];
                tmp /= sz;
            }
            AscendC::DataCopy<T, WTS_ARQ_NDDMA_DIMS, cfg>(dst[oi * innerElems], gm[gmOff + elemAdj], params);
        }
        return outerIters * innerElems;
    }

    // UB Broadcast (brcMode=2): expand compact staging into a full tile
    __aicore__ inline void UbBroadcast(AscendC::LocalTensor<T> dst, AscendC::LocalTensor<T> src, int64_t aISeg,
                                       const int64_t* strides)
    {
        const int64_t axis = td_->ubSplitAxis;
        const uint32_t rankB = static_cast<uint32_t>(static_cast<int64_t>(RANK) - axis);
        uint32_t dstShape[kWtsArqMaxRank];
        uint32_t srcShape[kWtsArqMaxRank];
        for (uint32_t i = 0; i < rankB; i++) {
            const int64_t d = axis + i;
            const uint32_t full = static_cast<uint32_t>((d == axis) ? aISeg : td_->dims[d]);
            dstShape[i] = full;
            srcShape[i] = (strides[d] == 0) ? 1U : full;
        }
        AscendC::BroadcastTiling brcTiling;
        AscendC::GetBroadcastTilingInfo<T>(rankB, dstShape, srcShape, false, brcTiling);
        AscendC::Broadcast<T>(dst, src, dstShape, srcShape, &brcTiling);
    }

    // Load w_min / w_max of one tile into a fp32 buffer, applying brcMode dispatch.
    __aicore__ inline void LoadRangeInput(const AscendC::GlobalTensor<T>& gm, int64_t off, const int64_t* strides,
                                          int64_t aISeg, int64_t count, AscendC::TBuf<AscendC::TPosition::VECCALC>& dst,
                                          int32_t evMTE2toV, int32_t evVtoMTE2, int32_t evMTE2toS, int32_t evVtoS)
    {
        // OneDim scalar broadcast: the real (collapsed, unpadded) stride lives at
        // index RANK-1; leading slots are host padding zeros (DESIGN.md §3.5: only
        // a stride-0 last axis means scalar broadcast, otherwise plain DataCopyPad).
        const bool isScalar = (td_->shapeLen == 1) && (strides[RANK - 1] == 0);
        if (isScalar) {
            LoadScalarExpand(gm, off, count, dst, evMTE2toV, evMTE2toS, evVtoS);
            return;
        }

        if constexpr (std::is_same_v<T, half>) {
            // fp16: copy/broadcast in half into a staging area, then Cast up to fp32.
            // B_SCALE is idle until scale computation, used as half scratch for the
            // broadcast/NDDMA destination; B_CAST holds the compact source in UB BRC mode.
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(evVtoMTE2);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evVtoMTE2);
            if (td_->brcMode == WTS_ARQ_BRC_UB) {
                AscendC::LocalTensor<T> staging = bufCast_.template Get<T>();
                AscendC::LocalTensor<T> scratch = bufScale_.template Get<T>();
                NddmaCopy(staging, gm, off, strides, aISeg, true);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
                UbBroadcast(scratch, staging, aISeg, strides);
                AscendC::Cast<float, half>(dst.template Get<float>(), bufScale_.template Get<half>(),
                                           AscendC::RoundMode::CAST_NONE, static_cast<uint32_t>(count));
            } else if (td_->brcMode == WTS_ARQ_BRC_NDDMA) {
                AscendC::LocalTensor<T> scratch = bufScale_.template Get<T>();
                NddmaCopy(scratch, gm, off, strides, aISeg, false);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
                AscendC::Cast<float, half>(dst.template Get<float>(), bufScale_.template Get<half>(),
                                           AscendC::RoundMode::CAST_NONE, static_cast<uint32_t>(count));
            } else {
                AscendC::LocalTensor<T> staging = bufCast_.template Get<T>();
                CopyInLinear(gm, off, staging, count);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
                AscendC::Cast<float, half>(dst.template Get<float>(), bufCast_.template Get<half>(),
                                           AscendC::RoundMode::CAST_NONE, static_cast<uint32_t>(count));
            }
        } else {
            if (td_->brcMode == WTS_ARQ_BRC_UB) {
                AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(evVtoMTE2);
                AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evVtoMTE2);
                AscendC::LocalTensor<T> staging = bufCast_.template Get<T>();
                NddmaCopy(staging, gm, off, strides, aISeg, true);
                AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
                AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
                UbBroadcast(dst.template Get<T>(), staging, aISeg, strides);
            } else if (td_->brcMode == WTS_ARQ_BRC_NDDMA) {
                AscendC::LocalTensor<T> dstT = dst.template Get<T>();
                NddmaCopy(dstT, gm, off, strides, aISeg, false);
            } else {
                AscendC::LocalTensor<T> dstT = dst.template Get<T>();
                CopyInLinear(gm, off, dstT, count);
            }
        }
    }

    // OneDim scalar broadcast: load 1 element, read it as scalar, Duplicate-expand
    __aicore__ inline void LoadScalarExpand(const AscendC::GlobalTensor<T>& gm, int64_t off, int64_t count,
                                            AscendC::TBuf<AscendC::TPosition::VECCALC>& dst, int32_t evMTE2toV,
                                            int32_t evMTE2toS, int32_t evVtoS)
    {
        float scalarVal = 0.0f;
        if constexpr (std::is_same_v<T, half>) {
            AscendC::LocalTensor<T> staging = bufCast_.template Get<T>();
            CopyInLinear(gm, off, staging, 1);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
            AscendC::Cast<float, half>(dst.template Get<float>(), bufCast_.template Get<half>(),
                                       AscendC::RoundMode::CAST_NONE, 1);
            AscendC::SetFlag<AscendC::HardEvent::V_S>(evVtoS);
            AscendC::WaitFlag<AscendC::HardEvent::V_S>(evVtoS);
            scalarVal = dst.template Get<float>().GetValue(0);
        } else {
            AscendC::LocalTensor<T> dstT = dst.template Get<T>();
            CopyInLinear(gm, off, dstT, 1);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_S>(evMTE2toS);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_S>(evMTE2toS);
            scalarVal = dst.template Get<float>().GetValue(0);
        }
        const uint32_t uCount = static_cast<uint32_t>(count);
        const uint16_t repeatTimes = static_cast<uint16_t>((count + static_cast<int64_t>(VL) - 1) /
                                                           static_cast<int64_t>(VL));
        asc_vf_call<DuplicateFillVF>((__ubuf__ float*)dst.template Get<float>().GetPhyAddr(), scalarVal, uCount, VL,
                                     repeatTimes);
    }

    // Load w of one tile (always contiguous, never broadcast)
    __aicore__ inline void LoadW(int64_t off, int64_t count, int32_t evMTE2toV, int32_t evVtoMTE2)
    {
        if constexpr (std::is_same_v<T, half>) {
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(evVtoMTE2);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(evVtoMTE2);
            AscendC::LocalTensor<T> staging = bufCast_.template Get<T>();
            CopyInLinear(gmW_, off, staging, count);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
            AscendC::Cast<float, half>(bufW_.template Get<float>(), bufCast_.template Get<half>(),
                                       AscendC::RoundMode::CAST_NONE, static_cast<uint32_t>(count));
        } else {
            AscendC::LocalTensor<T> dstT = bufW_.template Get<T>();
            CopyInLinear(gmW_, off, dstT, count);
            AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
            AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evMTE2toV);
        }
    }

    __aicore__ inline void CopyOut(int64_t off, int64_t count, AscendC::TBuf<AscendC::TPosition::VECCALC>& src)
    {
        AscendC::DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = static_cast<uint32_t>(count * sizeof(T));
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        AscendC::DataCopyPad(gmY_[off], src.template Get<T>(), copyParams);
    }

private:
    AscendC::TPipe pipe_;
    const WtsArqTilingData<RANK>* td_{nullptr};
    AscendC::GlobalTensor<T> gmW_;
    AscendC::GlobalTensor<T> gmMin_;
    AscendC::GlobalTensor<T> gmMax_;
    AscendC::GlobalTensor<T> gmY_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufW_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufMin_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufMax_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufScale_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufCast_;
    int64_t wStrides_[RANK];
};

} // namespace WtsArq

#endif // WTS_ARQ_KERNEL_H_
