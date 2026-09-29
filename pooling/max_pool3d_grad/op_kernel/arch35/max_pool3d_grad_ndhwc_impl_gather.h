/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MAX_POOL3D_GRAD_NDHWC_IMPL_GATHER_H
#define MAX_POOL3D_GRAD_NDHWC_IMPL_GATHER_H

#include "max_pool3d_grad_ndhwc_small_kernel.h"
#include "pool_utils/arch35/index/max_pool_with_argmax_index.h"
#include "pool_utils/arch35/compute/max_pool_negative_value.h"

namespace MaxPool3DGradNDHWCNameSpace {

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPool3DGradNDHWCSmallKernel<T, INDEX_T, IS_CHECK_RANGE>::Init(
    GM_ADDR origX, GM_ADDR origY, GM_ADDR grad, GM_ADDR y, const Pool3DGradNDHWCTilingData& tilingData)
{
    (void)origY;
    Base::ParseTilingData(tilingData);
    vToMte2Event_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    mte3ToMte2Event_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));

    Base::blockIdx_ = GetBlockIdx();
    if (Base::blockIdx_ >= tilingData_.base.usedCoreNum) {
        return;
    }
    Base::curCoreProcessNum_ = (Base::blockIdx_ + 1 == tilingData_.base.usedCoreNum) ?
                                   tilingData_.base.tailCoreProcessNum :
                                   tilingData_.base.normalCoreProcessNum;

    xGm_.SetGlobalBuffer((__gm__ T*)origX);
    Base::gradGm_.SetGlobalBuffer((__gm__ T*)grad);
    Base::yGm_.SetGlobalBuffer((__gm__ T*)y);

    isBigC_ = (cDim_ * static_cast<int64_t>(sizeof(T)) > V_REG_SIZE / 2);

    pipe_.InitBuffer(Base::argmaxBuff_, tilingData_.base.argmaxBufferSize);
    pipe_.InitBuffer(Base::helpBuf_, HELP_BUFFER);

    int64_t calcSize = Base::isPad_ ? tilingData_.base.inputBufferSize : 0;
    int64_t forwardSize = tilingData_.base.inputBufferSize * BUFFER_NUM + calcSize;
    int64_t backwardSize = tilingData_.base.gradBufferSize * BUFFER_NUM +
                           tilingData_.base.outputBufferSize * BUFFER_NUM;
    totalStageBufferSize_ = (forwardSize > backwardSize) ? forwardSize : backwardSize;

    pipe_.InitBufPool(forwardBufPool_, static_cast<uint32_t>(totalStageBufferSize_));
    pipe_.InitBufPool(backwardBufPool_, static_cast<uint32_t>(totalStageBufferSize_), forwardBufPool_);
    forwardBufPool_.InitBuffer(inputQue_, BUFFER_NUM, static_cast<uint32_t>(tilingData_.base.inputBufferSize));
    if (Base::isPad_) {
        forwardBufPool_.InitBuffer(inputCalcBuff_, static_cast<uint32_t>(tilingData_.base.inputBufferSize));
    }
    backwardBufPool_.InitBuffer(Base::gradQue_, BUFFER_NUM, static_cast<uint32_t>(tilingData_.base.gradBufferSize));
    backwardBufPool_.InitBuffer(Base::outputQue_, BUFFER_NUM, static_cast<uint32_t>(tilingData_.base.outputBufferSize));
}

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPool3DGradNDHWCSmallKernel<T, INDEX_T, IS_CHECK_RANGE>::Process()
{
    if (Base::blockIdx_ >= tilingData_.base.usedCoreNum) {
        return;
    }

    for (int64_t loopNum = 0; loopNum < Base::curCoreProcessNum_; ++loopNum) {
        Base::ScalarCompute(loopNum);

        if (dArgmaxActual_ == 0 || hArgmaxActual_ == 0 || wArgmaxActual_ == 0) {
            Base::ProcessNoArgmaxBlock();
            Base::Mte3Drain();
            continue;
        }

        ForwardScalarCompute();
        ForwardCopyIn();
        Forward();

        Base::VDrainToMte2();

        Base::CopyInGrad();
        Base::Backward();
        Base::CopyOut();
        Base::Mte3Drain();
    }
}

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPool3DGradNDHWCSmallKernel<T, INDEX_T, IS_CHECK_RANGE>::ForwardComputeTile()
{
    __ubuf__ T* xAddr = xForwardAddr_;
    __ubuf__ INDEX_T* argmaxAddr = (__ubuf__ INDEX_T*)argmaxBuff_.template Get<INDEX_T>().GetPhyAddr();

    int64_t vlT2 = V_REG_SIZE / sizeof(INDEX_T);
    int64_t C = cOutputActual_;
    int64_t wc = wArgmaxActual_ * C;
    int64_t hwc = hArgmaxActual_ * wc;
    int64_t dhwc = dArgmaxActual_ * hwc;
    int64_t ndhwc = nOutputActual_ * dhwc;

    if (isBigC_) {
        ChunkCGather(xAddr, argmaxAddr);
    } else if (dhwc <= vlT2) {
        MultiNcGather(xAddr, argmaxAddr);
    } else if (hwc <= vlT2) {
        MultiDepGather(xAddr, argmaxAddr);
    } else if (wc <= vlT2) {
        MultiRowGather(xAddr, argmaxAddr);
    } else {
        SingleRowGather(xAddr, argmaxAddr);
    }
}

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__simd_callee__ inline void MaxPool3DGradNDHWCSmallKernel<T, INDEX_T, IS_CHECK_RANGE>::ProcessWNDHWCKernel(
    __ubuf__ T* xAddr, int32_t hOffset, uint32_t colStride, uint32_t rowStride, uint32_t depStride,
    Reg::RegTensor<int32_t>& indexReg, uint16_t dKernel, uint16_t hKernel, uint16_t wKernel, uint16_t repeatElem,
    Reg::RegTensor<INDEX_T>& argmaxDStart, Reg::RegTensor<INDEX_T>& argmaxHStart, Reg::RegTensor<INDEX_T>& argmaxWStart,
    Reg::RegTensor<INDEX_T>& argmaxDRes, Reg::RegTensor<INDEX_T>& argmaxHRes, Reg::RegTensor<INDEX_T>& argmaxWRes,
    uint32_t dDilation, uint32_t hDilation, uint32_t wDilation)
{
    Reg::RegTensor<int32_t> indexWithOffset;
    Reg::RegTensor<T> calcReg;
    Reg::RegTensor<T> maxReg;
    Reg::RegTensor<INDEX_T> candReg;

    Reg::MaskReg allMaskU32 = Reg::CreateMask<int32_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg allMaskIdx = Reg::CreateMask<INDEX_T, Reg::MaskPattern::ALL>();
    uint32_t gatherCount = static_cast<uint32_t>(repeatElem);
    Reg::MaskReg gatherMask = Reg::UpdateMask<T>(gatherCount);
    Reg::MaskReg gtMask, neMask;

    PoolUtils::Compute::DuplicateNegInfRegVF<T>(maxReg);

    for (uint16_t d = 0; d < dKernel; d++) {
        for (uint16_t h = 0; h < hKernel; h++) {
            for (uint16_t w = 0; w < wKernel; w++) {
                int32_t relIndex = d * depStride * dDilation + h * rowStride * hDilation + w * colStride * wDilation;
                int32_t offset = hOffset + relIndex;

                Reg::Adds(indexWithOffset, indexReg, offset, allMaskU32);

                if constexpr (std::is_same<T, float>::value) {
                    Reg::DataCopyGather(calcReg, xAddr, (Reg::RegTensor<uint32_t>&)indexWithOffset, gatherMask);
                } else {
                    Reg::RegTensor<uint16_t> indexConvert;
                    Reg::Pack(indexConvert, indexWithOffset);
                    Reg::DataCopyGather(calcReg, xAddr, indexConvert, gatherMask);
                }

                Reg::Compare<T, CMPMODE::GT>(gtMask, calcReg, maxReg, gatherMask);
                Reg::Compare<T, CMPMODE::NE>(neMask, calcReg, calcReg, gatherMask);
                Reg::MaskOr(gtMask, gtMask, neMask, gatherMask);

                if constexpr (sizeof(INDEX_T) / sizeof(T) == 1) {
                    Reg::Adds(candReg, argmaxDStart, static_cast<INDEX_T>(d * dDilation), allMaskIdx);
                    Reg::Select(argmaxDRes, candReg, argmaxDRes, gtMask);
                    Reg::Adds(candReg, argmaxHStart, static_cast<INDEX_T>(h * hDilation), allMaskIdx);
                    Reg::Select(argmaxHRes, candReg, argmaxHRes, gtMask);
                    Reg::Adds(candReg, argmaxWStart, static_cast<INDEX_T>(w * wDilation), allMaskIdx);
                    Reg::Select(argmaxWRes, candReg, argmaxWRes, gtMask);
                } else {
                    Reg::MaskReg gtMaskUnpack;
                    Reg::UnPack(gtMaskUnpack, gtMask);
                    Reg::Adds(candReg, argmaxDStart, static_cast<INDEX_T>(d * dDilation), allMaskIdx);
                    Reg::Select(argmaxDRes, candReg, argmaxDRes, gtMaskUnpack);
                    Reg::Adds(candReg, argmaxHStart, static_cast<INDEX_T>(h * hDilation), allMaskIdx);
                    Reg::Select(argmaxHRes, candReg, argmaxHRes, gtMaskUnpack);
                    Reg::Adds(candReg, argmaxWStart, static_cast<INDEX_T>(w * wDilation), allMaskIdx);
                    Reg::Select(argmaxWRes, candReg, argmaxWRes, gtMaskUnpack);
                }

                Reg::Max(maxReg, maxReg, calcReg, gatherMask);
            }
        }
    }
}

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__simd_callee__ inline void MaxPool3DGradNDHWCSmallKernel<T, INDEX_T, IS_CHECK_RANGE>::ComposeAndStoreArgmax(
    Reg::RegTensor<INDEX_T>& argmaxDRes, Reg::RegTensor<INDEX_T>& argmaxHRes, Reg::RegTensor<INDEX_T>& argmaxWRes,
    __ubuf__ INDEX_T* argmaxAddr, uint16_t repeatElem, int32_t hInput, int32_t wInput, uint32_t isPad)
{
    Reg::MaskReg allMaskIdx = Reg::CreateMask<INDEX_T, Reg::MaskPattern::ALL>();

    if (isPad) {
        Reg::RegTensor<INDEX_T> zeroReg;
        Reg::MaskReg ltMask;
        Reg::Duplicate(zeroReg, static_cast<INDEX_T>(0));
        Reg::Compare<INDEX_T, CMPMODE::LT>(ltMask, argmaxDRes, zeroReg, allMaskIdx);
        Reg::Select(argmaxDRes, zeroReg, argmaxDRes, ltMask);
        Reg::Compare<INDEX_T, CMPMODE::LT>(ltMask, argmaxHRes, zeroReg, allMaskIdx);
        Reg::Select(argmaxHRes, zeroReg, argmaxHRes, ltMask);
        Reg::Compare<INDEX_T, CMPMODE::LT>(ltMask, argmaxWRes, zeroReg, allMaskIdx);
        Reg::Select(argmaxWRes, zeroReg, argmaxWRes, ltMask);
    }

    Reg::RegTensor<INDEX_T> argmaxRes;
    Reg::RegTensor<INDEX_T> tmp;
    Reg::Muls(argmaxRes, argmaxDRes, static_cast<INDEX_T>(hInput * wInput), allMaskIdx);
    Reg::Muls(tmp, argmaxHRes, static_cast<INDEX_T>(wInput), allMaskIdx);
    Reg::Add(argmaxRes, argmaxRes, tmp, allMaskIdx);
    Reg::Add(argmaxRes, argmaxRes, argmaxWRes, allMaskIdx);

    Reg::UnalignReg u1;
    Reg::DataCopyUnAlign(argmaxAddr, argmaxRes, u1, repeatElem);
    Reg::DataCopyUnAlignPost(argmaxAddr, u1, 0);
}

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__simd_vf__ inline void MaxPool3DGradNDHWCSmallKernel<T, INDEX_T, IS_CHECK_RANGE>::ProcessWNDHWC(
    __ubuf__ T* xAddr, __ubuf__ INDEX_T* argmaxAddr, uint16_t repeatElem, uint16_t cChunk, uint16_t dKernel,
    uint16_t hKernel, uint16_t wKernel, uint32_t colStride, uint32_t rowStride, uint32_t depStride, uint32_t wBaseStep,
    int32_t dOrigin, int32_t hOrigin, int32_t wOrigin, int32_t sW, int32_t hInput, int32_t wInput, uint32_t dDilation,
    uint32_t hDilation, uint32_t wDilation, uint32_t isPad)
{
    Reg::RegTensor<int32_t> indexReg;
    Reg::RegTensor<INDEX_T> dStart, hStart, wStart, dRes, hRes, wRes;
    Reg::MaskReg allMaskIdx = Reg::CreateMask<INDEX_T, Reg::MaskPattern::ALL>();

    PoolUtils::Index::GenGatterIndex2DVF<int32_t>(indexReg, static_cast<int32_t>(wBaseStep),
                                                  static_cast<int32_t>(cChunk), 1);
    PoolUtils::Index::GenGatterIndex2DVF<INDEX_T>(wStart, static_cast<INDEX_T>(sW), static_cast<INDEX_T>(cChunk), 0);
    Reg::Adds(wStart, wStart, static_cast<INDEX_T>(wOrigin), allMaskIdx);
    Reg::Duplicate(hStart, static_cast<INDEX_T>(hOrigin));
    Reg::Duplicate(dStart, static_cast<INDEX_T>(dOrigin));
    Reg::Copy(dRes, dStart, allMaskIdx);
    Reg::Copy(hRes, hStart, allMaskIdx);
    Reg::Copy(wRes, wStart, allMaskIdx);

    ProcessWNDHWCKernel(xAddr, 0, colStride, rowStride, depStride, indexReg, dKernel, hKernel, wKernel, repeatElem,
                        dStart, hStart, wStart, dRes, hRes, wRes, dDilation, hDilation, wDilation);

    ComposeAndStoreArgmax(dRes, hRes, wRes, argmaxAddr, repeatElem, hInput, wInput, isPad);
}

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__simd_vf__ inline void MaxPool3DGradNDHWCSmallKernel<T, INDEX_T, IS_CHECK_RANGE>::ProcessWNDHWC2D(
    __ubuf__ T* xAddr, __ubuf__ INDEX_T* argmaxAddr, uint16_t repeatElem, uint16_t cChunk, uint16_t wOut,
    uint16_t dKernel, uint16_t hKernel, uint16_t wKernel, uint32_t colStride, uint32_t rowStride, uint32_t depStride,
    uint32_t wBaseStep, uint32_t hBaseStep, int32_t dOrigin, int32_t hOrigin, int32_t wOrigin, int32_t sW, int32_t sH,
    int32_t hInput, int32_t wInput, uint32_t dDilation, uint32_t hDilation, uint32_t wDilation, uint32_t isPad)
{
    Reg::RegTensor<int32_t> indexReg;
    Reg::RegTensor<INDEX_T> dStart, hStart, wStart, dRes, hRes, wRes;
    Reg::MaskReg allMaskIdx = Reg::CreateMask<INDEX_T, Reg::MaskPattern::ALL>();
    int32_t wcChunk = static_cast<int32_t>(wOut) * static_cast<int32_t>(cChunk);

    PoolUtils::Index::GenGatterIndex3DVF<int32_t>(indexReg, static_cast<int32_t>(hBaseStep), wcChunk,
                                                  static_cast<int32_t>(wBaseStep), static_cast<int32_t>(cChunk), 1);
    PoolUtils::Index::GenGatterIndex3DVF<INDEX_T>(hStart, static_cast<INDEX_T>(sH), wcChunk, 0,
                                                  static_cast<INDEX_T>(cChunk), 0);
    PoolUtils::Index::GenGatterIndex3DVF<INDEX_T>(wStart, 0, wcChunk, static_cast<INDEX_T>(sW),
                                                  static_cast<INDEX_T>(cChunk), 0);
    Reg::Adds(hStart, hStart, static_cast<INDEX_T>(hOrigin), allMaskIdx);
    Reg::Adds(wStart, wStart, static_cast<INDEX_T>(wOrigin), allMaskIdx);
    Reg::Duplicate(dStart, static_cast<INDEX_T>(dOrigin));
    Reg::Copy(dRes, dStart, allMaskIdx);
    Reg::Copy(hRes, hStart, allMaskIdx);
    Reg::Copy(wRes, wStart, allMaskIdx);

    ProcessWNDHWCKernel(xAddr, 0, colStride, rowStride, depStride, indexReg, dKernel, hKernel, wKernel, repeatElem,
                        dStart, hStart, wStart, dRes, hRes, wRes, dDilation, hDilation, wDilation);

    ComposeAndStoreArgmax(dRes, hRes, wRes, argmaxAddr, repeatElem, hInput, wInput, isPad);
}

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__simd_vf__ inline void MaxPool3DGradNDHWCSmallKernel<T, INDEX_T, IS_CHECK_RANGE>::ProcessWNDHWC3D(
    __ubuf__ T* xAddr, __ubuf__ INDEX_T* argmaxAddr, uint16_t repeatElem, uint16_t cChunk, uint16_t wOut, uint16_t hOut,
    uint16_t dKernel, uint16_t hKernel, uint16_t wKernel, uint32_t colStride, uint32_t rowStride, uint32_t depStride,
    uint32_t wBaseStep, uint32_t hBaseStep, uint32_t dBaseStep, int32_t dOrigin, int32_t hOrigin, int32_t wOrigin,
    int32_t sD, int32_t sW, int32_t sH, int32_t hInput, int32_t wInput, uint32_t dDilation, uint32_t hDilation,
    uint32_t wDilation, uint32_t isPad)
{
    Reg::RegTensor<int32_t> indexReg;
    Reg::RegTensor<INDEX_T> dStart, hStart, wStart, dRes, hRes, wRes;
    Reg::MaskReg allMaskIdx = Reg::CreateMask<INDEX_T, Reg::MaskPattern::ALL>();
    int32_t wcChunk = static_cast<int32_t>(wOut) * static_cast<int32_t>(cChunk);
    int32_t hwcChunk = static_cast<int32_t>(hOut) * wcChunk;

    PoolUtils::Index::GenGatterIndex4DVF<int32_t>(indexReg, static_cast<int32_t>(dBaseStep), hwcChunk,
                                                  static_cast<int32_t>(hBaseStep), wcChunk,
                                                  static_cast<int32_t>(wBaseStep), static_cast<int32_t>(cChunk), 1);
    PoolUtils::Index::GenGatterIndex4DVF<INDEX_T>(dStart, static_cast<INDEX_T>(sD), hwcChunk, 0, wcChunk, 0,
                                                  static_cast<INDEX_T>(cChunk), 0);
    PoolUtils::Index::GenGatterIndex4DVF<INDEX_T>(hStart, 0, hwcChunk, static_cast<INDEX_T>(sH), wcChunk, 0,
                                                  static_cast<INDEX_T>(cChunk), 0);
    PoolUtils::Index::GenGatterIndex4DVF<INDEX_T>(wStart, 0, hwcChunk, 0, wcChunk, static_cast<INDEX_T>(sW),
                                                  static_cast<INDEX_T>(cChunk), 0);
    Reg::Adds(dStart, dStart, static_cast<INDEX_T>(dOrigin), allMaskIdx);
    Reg::Adds(hStart, hStart, static_cast<INDEX_T>(hOrigin), allMaskIdx);
    Reg::Adds(wStart, wStart, static_cast<INDEX_T>(wOrigin), allMaskIdx);
    Reg::Copy(dRes, dStart, allMaskIdx);
    Reg::Copy(hRes, hStart, allMaskIdx);
    Reg::Copy(wRes, wStart, allMaskIdx);

    ProcessWNDHWCKernel(xAddr, 0, colStride, rowStride, depStride, indexReg, dKernel, hKernel, wKernel, repeatElem,
                        dStart, hStart, wStart, dRes, hRes, wRes, dDilation, hDilation, wDilation);

    ComposeAndStoreArgmax(dRes, hRes, wRes, argmaxAddr, repeatElem, hInput, wInput, isPad);
}

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__simd_vf__ inline void MaxPool3DGradNDHWCSmallKernel<T, INDEX_T, IS_CHECK_RANGE>::ProcessWNDHWC4D(
    __ubuf__ T* xAddr, __ubuf__ INDEX_T* argmaxAddr, uint16_t repeatElem, uint16_t cChunk, uint16_t wOut, uint16_t hOut,
    uint16_t dOut, uint16_t dKernel, uint16_t hKernel, uint16_t wKernel, uint32_t colStride, uint32_t rowStride,
    uint32_t depStride, uint32_t wBaseStep, uint32_t hBaseStep, uint32_t dBaseStep, uint32_t ncBaseStep,
    int32_t dOrigin, int32_t hOrigin, int32_t wOrigin, int32_t sD, int32_t sW, int32_t sH, int32_t hInput,
    int32_t wInput, uint32_t dDilation, uint32_t hDilation, uint32_t wDilation, uint32_t isPad)
{
    Reg::RegTensor<int32_t> indexReg;
    Reg::RegTensor<INDEX_T> dStart, hStart, wStart, dRes, hRes, wRes;
    Reg::MaskReg allMaskIdx = Reg::CreateMask<INDEX_T, Reg::MaskPattern::ALL>();
    int32_t wcChunk = static_cast<int32_t>(wOut) * static_cast<int32_t>(cChunk);
    int32_t hwcChunk = static_cast<int32_t>(hOut) * wcChunk;
    int32_t dhwcChunk = static_cast<int32_t>(dOut) * hwcChunk;

    PoolUtils::Index::GenGatterIndex5DVF<int32_t>(
        indexReg, static_cast<int32_t>(ncBaseStep), dhwcChunk, static_cast<int32_t>(dBaseStep), hwcChunk,
        static_cast<int32_t>(hBaseStep), wcChunk, static_cast<int32_t>(wBaseStep), static_cast<int32_t>(cChunk), 1);
    PoolUtils::Index::GenGatterIndex5DVF<INDEX_T>(dStart, 0, dhwcChunk, static_cast<INDEX_T>(sD), hwcChunk, 0, wcChunk,
                                                  0, static_cast<INDEX_T>(cChunk), 0);
    PoolUtils::Index::GenGatterIndex5DVF<INDEX_T>(hStart, 0, dhwcChunk, 0, hwcChunk, static_cast<INDEX_T>(sH), wcChunk,
                                                  0, static_cast<INDEX_T>(cChunk), 0);
    PoolUtils::Index::GenGatterIndex5DVF<INDEX_T>(wStart, 0, dhwcChunk, 0, hwcChunk, 0, wcChunk,
                                                  static_cast<INDEX_T>(sW), static_cast<INDEX_T>(cChunk), 0);
    Reg::Adds(dStart, dStart, static_cast<INDEX_T>(dOrigin), allMaskIdx);
    Reg::Adds(hStart, hStart, static_cast<INDEX_T>(hOrigin), allMaskIdx);
    Reg::Adds(wStart, wStart, static_cast<INDEX_T>(wOrigin), allMaskIdx);
    Reg::Copy(dRes, dStart, allMaskIdx);
    Reg::Copy(hRes, hStart, allMaskIdx);
    Reg::Copy(wRes, wStart, allMaskIdx);

    ProcessWNDHWCKernel(xAddr, 0, colStride, rowStride, depStride, indexReg, dKernel, hKernel, wKernel, repeatElem,
                        dStart, hStart, wStart, dRes, hRes, wRes, dDilation, hDilation, wDilation);

    ComposeAndStoreArgmax(dRes, hRes, wRes, argmaxAddr, repeatElem, hInput, wInput, isPad);
}

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPool3DGradNDHWCSmallKernel<T, INDEX_T, IS_CHECK_RANGE>::SingleRowGather(
    __ubuf__ T* xAddr, __ubuf__ INDEX_T* argmaxAddr)
{
    constexpr uint32_t DOUBLE_HEAD = 2;
    (void)DOUBLE_HEAD;

    uint16_t C = static_cast<uint16_t>(cOutputActual_);
    uint16_t dKernel = tilingData_.base.dKernel;
    uint16_t hKernel = tilingData_.base.hKernel;
    uint16_t wKernel = tilingData_.base.wKernel;
    uint32_t colStride = static_cast<uint32_t>(cAligned_);
    uint32_t rowStride = static_cast<uint32_t>(wInputActualPad_ * cAligned_);
    uint32_t depStride = static_cast<uint32_t>(hInputActualPad_ * rowStride);
    uint32_t wBaseStep = static_cast<uint32_t>(tilingData_.base.wStride * cAligned_);
    int32_t dOrigin = static_cast<int32_t>(dArgmaxActualStart_ * tilingData_.base.dStride - tilingData_.base.padD);
    int32_t hOrigin = static_cast<int32_t>(hArgmaxActualStart_ * tilingData_.base.hStride - tilingData_.base.padH);
    int32_t wOrigin = static_cast<int32_t>(wArgmaxActualStart_ * tilingData_.base.wStride - tilingData_.base.padW);
    int32_t sD = static_cast<int32_t>(tilingData_.base.dStride);
    int32_t sH = static_cast<int32_t>(tilingData_.base.hStride);
    int32_t sW = static_cast<int32_t>(tilingData_.base.wStride);
    int32_t hInput = static_cast<int32_t>(tilingData_.base.hOutput);
    int32_t wInput = static_cast<int32_t>(tilingData_.base.wOutput);
    uint32_t dDilation = tilingData_.base.dilationD;
    uint32_t hDilation = tilingData_.base.dilationH;
    uint32_t wDilation = tilingData_.base.dilationW;
    uint32_t isPad = isPad_ ? 1 : 0;

    uint16_t vlT2 = static_cast<uint16_t>(V_REG_SIZE / sizeof(INDEX_T));
    uint16_t wFactor = vlT2 / C;
    uint16_t loopW = static_cast<uint16_t>(wArgmaxActual_ / wFactor);
    uint16_t tailW = static_cast<uint16_t>(wArgmaxActual_ - loopW * wFactor);
    if (tailW == 0) {
        loopW = loopW - 1;
        tailW = wFactor;
    }
    uint32_t wChunkElems = static_cast<uint32_t>(wFactor) * C;
    uint32_t tailElems = static_cast<uint32_t>(tailW) * C;

    uint32_t argH = static_cast<uint32_t>(wArgmaxActual_) * C;
    uint32_t argD = static_cast<uint32_t>(hArgmaxActual_) * argH;
    uint32_t argN = static_cast<uint32_t>(dArgmaxActual_) * argD;

    for (int64_t n = 0; n < nOutputActual_; n++) {
        for (int64_t d = 0; d < dArgmaxActual_; d++) {
            for (int64_t h = 0; h < hArgmaxActual_; h++) {
                int64_t xOff = n * ncBaseStep_ + d * dBaseStep_ + h * hBaseStep_;
                int64_t argOff = static_cast<int64_t>(n) * argN + d * argD + h * argH;
                int32_t dOriginEff = dOrigin + static_cast<int32_t>(d) * sD;
                int32_t hOriginEff = hOrigin + static_cast<int32_t>(h) * sH;
                for (uint16_t wLoop = 0; wLoop < loopW; wLoop++) {
                    int32_t wOriginEff = wOrigin + static_cast<int32_t>(wLoop * wFactor) * sW;
                    ProcessWNDHWC(xAddr + xOff + static_cast<int64_t>(wLoop) * wFactor * wBaseStep,
                                  argmaxAddr + argOff + static_cast<int64_t>(wLoop) * wChunkElems,
                                  static_cast<uint16_t>(wChunkElems), C, dKernel, hKernel, wKernel, colStride,
                                  rowStride, depStride, wBaseStep, dOriginEff, hOriginEff, wOriginEff, sW, hInput,
                                  wInput, dDilation, hDilation, wDilation, isPad);
                }
                int32_t wOriginTail = wOrigin + static_cast<int32_t>(loopW * wFactor) * sW;
                ProcessWNDHWC(xAddr + xOff + static_cast<int64_t>(loopW) * wFactor * wBaseStep,
                              argmaxAddr + argOff + static_cast<int64_t>(loopW) * wChunkElems,
                              static_cast<uint16_t>(tailElems), C, dKernel, hKernel, wKernel, colStride, rowStride,
                              depStride, wBaseStep, dOriginEff, hOriginEff, wOriginTail, sW, hInput, wInput, dDilation,
                              hDilation, wDilation, isPad);
            }
        }
    }
}

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPool3DGradNDHWCSmallKernel<T, INDEX_T, IS_CHECK_RANGE>::MultiRowGather(
    __ubuf__ T* xAddr, __ubuf__ INDEX_T* argmaxAddr)
{
    uint16_t C = static_cast<uint16_t>(cOutputActual_);
    uint16_t dKernel = tilingData_.base.dKernel;
    uint16_t hKernel = tilingData_.base.hKernel;
    uint16_t wKernel = tilingData_.base.wKernel;
    uint32_t colStride = static_cast<uint32_t>(cAligned_);
    uint32_t rowStride = static_cast<uint32_t>(wInputActualPad_ * cAligned_);
    uint32_t depStride = static_cast<uint32_t>(hInputActualPad_ * rowStride);
    uint32_t wBaseStep = static_cast<uint32_t>(tilingData_.base.wStride * cAligned_);
    uint32_t hBaseStep = static_cast<uint32_t>(tilingData_.base.hStride * rowStride);
    int32_t dOrigin = static_cast<int32_t>(dArgmaxActualStart_ * tilingData_.base.dStride - tilingData_.base.padD);
    int32_t hOrigin = static_cast<int32_t>(hArgmaxActualStart_ * tilingData_.base.hStride - tilingData_.base.padH);
    int32_t wOrigin = static_cast<int32_t>(wArgmaxActualStart_ * tilingData_.base.wStride - tilingData_.base.padW);
    int32_t sD = static_cast<int32_t>(tilingData_.base.dStride);
    int32_t sH = static_cast<int32_t>(tilingData_.base.hStride);
    int32_t sW = static_cast<int32_t>(tilingData_.base.wStride);
    int32_t hInput = static_cast<int32_t>(tilingData_.base.hOutput);
    int32_t wInput = static_cast<int32_t>(tilingData_.base.wOutput);
    uint32_t dDilation = tilingData_.base.dilationD;
    uint32_t hDilation = tilingData_.base.dilationH;
    uint32_t wDilation = tilingData_.base.dilationW;
    uint32_t isPad = isPad_ ? 1 : 0;

    uint16_t vlT2 = static_cast<uint16_t>(V_REG_SIZE / sizeof(INDEX_T));
    uint16_t wOutA = static_cast<uint16_t>(wArgmaxActual_);
    uint16_t hBatch = vlT2 / static_cast<uint16_t>(wOutA * C);
    uint16_t hLoopTimes = static_cast<uint16_t>(hArgmaxActual_ / hBatch);
    uint16_t hTail = static_cast<uint16_t>(hArgmaxActual_ - hLoopTimes * hBatch);
    if (hTail == 0) {
        hLoopTimes = hLoopTimes - 1;
        hTail = hBatch;
    }
    uint32_t chunkElems = static_cast<uint32_t>(hBatch) * wOutA * C;
    uint32_t tailElems = static_cast<uint32_t>(hTail) * wOutA * C;

    uint32_t argH = static_cast<uint32_t>(wArgmaxActual_) * C;
    uint32_t argD = static_cast<uint32_t>(hArgmaxActual_) * argH;
    uint32_t argN = static_cast<uint32_t>(dArgmaxActual_) * argD;

    for (int64_t n = 0; n < nOutputActual_; n++) {
        for (int64_t d = 0; d < dArgmaxActual_; d++) {
            int64_t xBase = n * ncBaseStep_ + d * dBaseStep_;
            int64_t argBase = static_cast<int64_t>(n) * argN + d * argD;
            int32_t dOriginEff = dOrigin + static_cast<int32_t>(d) * sD;
            for (uint16_t hLoop = 0; hLoop < hLoopTimes; hLoop++) {
                int32_t hOriginEff = hOrigin + static_cast<int32_t>(hLoop * hBatch) * sH;
                ProcessWNDHWC2D(xAddr + xBase + static_cast<int64_t>(hLoop) * hBatch * hBaseStep_,
                                argmaxAddr + argBase + static_cast<int64_t>(hLoop) * chunkElems,
                                static_cast<uint16_t>(chunkElems), C, wOutA, dKernel, hKernel, wKernel, colStride,
                                rowStride, depStride, wBaseStep, hBaseStep, dOriginEff, hOriginEff, wOrigin, sW, sH,
                                hInput, wInput, dDilation, hDilation, wDilation, isPad);
            }
            int32_t hOriginTail = hOrigin + static_cast<int32_t>(hLoopTimes * hBatch) * sH;
            ProcessWNDHWC2D(xAddr + xBase + static_cast<int64_t>(hLoopTimes) * hBatch * hBaseStep_,
                            argmaxAddr + argBase + static_cast<int64_t>(hLoopTimes) * chunkElems,
                            static_cast<uint16_t>(tailElems), C, wOutA, dKernel, hKernel, wKernel, colStride, rowStride,
                            depStride, wBaseStep, hBaseStep, dOriginEff, hOriginTail, wOrigin, sW, sH, hInput, wInput,
                            dDilation, hDilation, wDilation, isPad);
        }
    }
}

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPool3DGradNDHWCSmallKernel<T, INDEX_T, IS_CHECK_RANGE>::MultiDepGather(
    __ubuf__ T* xAddr, __ubuf__ INDEX_T* argmaxAddr)
{
    uint16_t C = static_cast<uint16_t>(cOutputActual_);
    uint16_t dKernel = tilingData_.base.dKernel;
    uint16_t hKernel = tilingData_.base.hKernel;
    uint16_t wKernel = tilingData_.base.wKernel;
    uint32_t colStride = static_cast<uint32_t>(cAligned_);
    uint32_t rowStride = static_cast<uint32_t>(wInputActualPad_ * cAligned_);
    uint32_t depStride = static_cast<uint32_t>(hInputActualPad_ * rowStride);
    uint32_t wBaseStep = static_cast<uint32_t>(tilingData_.base.wStride * cAligned_);
    uint32_t hBaseStep = static_cast<uint32_t>(tilingData_.base.hStride * rowStride);
    uint32_t dBaseStep = static_cast<uint32_t>(tilingData_.base.dStride * depStride);
    int32_t dOrigin = static_cast<int32_t>(dArgmaxActualStart_ * tilingData_.base.dStride - tilingData_.base.padD);
    int32_t hOrigin = static_cast<int32_t>(hArgmaxActualStart_ * tilingData_.base.hStride - tilingData_.base.padH);
    int32_t wOrigin = static_cast<int32_t>(wArgmaxActualStart_ * tilingData_.base.wStride - tilingData_.base.padW);
    int32_t sD = static_cast<int32_t>(tilingData_.base.dStride);
    int32_t sH = static_cast<int32_t>(tilingData_.base.hStride);
    int32_t sW = static_cast<int32_t>(tilingData_.base.wStride);
    int32_t hInput = static_cast<int32_t>(tilingData_.base.hOutput);
    int32_t wInput = static_cast<int32_t>(tilingData_.base.wOutput);
    uint32_t dDilation = tilingData_.base.dilationD;
    uint32_t hDilation = tilingData_.base.dilationH;
    uint32_t wDilation = tilingData_.base.dilationW;
    uint32_t isPad = isPad_ ? 1 : 0;

    uint16_t vlT2 = static_cast<uint16_t>(V_REG_SIZE / sizeof(INDEX_T));
    uint16_t wOutA = static_cast<uint16_t>(wArgmaxActual_);
    uint16_t hOutA = static_cast<uint16_t>(hArgmaxActual_);
    uint16_t hwc = hOutA * wOutA * C;
    uint16_t dBatch = vlT2 / hwc;
    uint16_t dLoopTimes = static_cast<uint16_t>(dArgmaxActual_ / dBatch);
    uint16_t dTail = static_cast<uint16_t>(dArgmaxActual_ - dLoopTimes * dBatch);
    if (dTail == 0) {
        dLoopTimes = dLoopTimes - 1;
        dTail = dBatch;
    }
    uint32_t chunkElems = static_cast<uint32_t>(dBatch) * hwc;
    uint32_t tailElems = static_cast<uint32_t>(dTail) * hwc;

    uint32_t argH = static_cast<uint32_t>(wArgmaxActual_) * C;
    uint32_t argD = static_cast<uint32_t>(hArgmaxActual_) * argH;
    uint32_t argN = static_cast<uint32_t>(dArgmaxActual_) * argD;

    for (int64_t n = 0; n < nOutputActual_; n++) {
        int64_t xBase = n * ncBaseStep_;
        int64_t argBase = static_cast<int64_t>(n) * argN;
        for (uint16_t dLoop = 0; dLoop < dLoopTimes; dLoop++) {
            int32_t dOriginEff = dOrigin + static_cast<int32_t>(dLoop * dBatch) * sD;
            ProcessWNDHWC3D(xAddr + xBase + static_cast<int64_t>(dLoop) * dBatch * dBaseStep,
                            argmaxAddr + argBase + static_cast<int64_t>(dLoop) * chunkElems,
                            static_cast<uint16_t>(chunkElems), C, wOutA, hOutA, dKernel, hKernel, wKernel, colStride,
                            rowStride, depStride, wBaseStep, hBaseStep, dBaseStep, dOriginEff, hOrigin, wOrigin, sD, sW,
                            sH, hInput, wInput, dDilation, hDilation, wDilation, isPad);
        }
        int32_t dOriginTail = dOrigin + static_cast<int32_t>(dLoopTimes * dBatch) * sD;
        ProcessWNDHWC3D(xAddr + xBase + static_cast<int64_t>(dLoopTimes) * dBatch * dBaseStep,
                        argmaxAddr + argBase + static_cast<int64_t>(dLoopTimes) * chunkElems,
                        static_cast<uint16_t>(tailElems), C, wOutA, hOutA, dKernel, hKernel, wKernel, colStride,
                        rowStride, depStride, wBaseStep, hBaseStep, dBaseStep, dOriginTail, hOrigin, wOrigin, sD, sW,
                        sH, hInput, wInput, dDilation, hDilation, wDilation, isPad);
    }
}

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPool3DGradNDHWCSmallKernel<T, INDEX_T, IS_CHECK_RANGE>::MultiNcGather(
    __ubuf__ T* xAddr, __ubuf__ INDEX_T* argmaxAddr)
{
    uint16_t C = static_cast<uint16_t>(cOutputActual_);
    uint16_t dKernel = tilingData_.base.dKernel;
    uint16_t hKernel = tilingData_.base.hKernel;
    uint16_t wKernel = tilingData_.base.wKernel;
    uint32_t colStride = static_cast<uint32_t>(cAligned_);
    uint32_t rowStride = static_cast<uint32_t>(wInputActualPad_ * cAligned_);
    uint32_t depStride = static_cast<uint32_t>(hInputActualPad_ * rowStride);
    uint32_t wBaseStep = static_cast<uint32_t>(tilingData_.base.wStride * cAligned_);
    uint32_t hBaseStep = static_cast<uint32_t>(tilingData_.base.hStride * rowStride);
    uint32_t dBaseStep = static_cast<uint32_t>(tilingData_.base.dStride * depStride);
    uint32_t ncBaseStep = static_cast<uint32_t>(dInputActualPad_ * depStride);
    int32_t dOrigin = static_cast<int32_t>(dArgmaxActualStart_ * tilingData_.base.dStride - tilingData_.base.padD);
    int32_t hOrigin = static_cast<int32_t>(hArgmaxActualStart_ * tilingData_.base.hStride - tilingData_.base.padH);
    int32_t wOrigin = static_cast<int32_t>(wArgmaxActualStart_ * tilingData_.base.wStride - tilingData_.base.padW);
    int32_t sD = static_cast<int32_t>(tilingData_.base.dStride);
    int32_t sH = static_cast<int32_t>(tilingData_.base.hStride);
    int32_t sW = static_cast<int32_t>(tilingData_.base.wStride);
    int32_t hInput = static_cast<int32_t>(tilingData_.base.hOutput);
    int32_t wInput = static_cast<int32_t>(tilingData_.base.wOutput);
    uint32_t dDilation = tilingData_.base.dilationD;
    uint32_t hDilation = tilingData_.base.dilationH;
    uint32_t wDilation = tilingData_.base.dilationW;
    uint32_t isPad = isPad_ ? 1 : 0;

    uint16_t vlT2 = static_cast<uint16_t>(V_REG_SIZE / sizeof(INDEX_T));
    uint16_t wOutA = static_cast<uint16_t>(wArgmaxActual_);
    uint16_t hOutA = static_cast<uint16_t>(hArgmaxActual_);
    uint16_t dOutA = static_cast<uint16_t>(dArgmaxActual_);
    uint16_t dhwc = dOutA * hOutA * wOutA * C;
    uint16_t ncBatch = vlT2 / dhwc;
    uint16_t ncLoopTimes = static_cast<uint16_t>(nOutputActual_ / ncBatch);
    uint16_t ncTail = static_cast<uint16_t>(nOutputActual_ - ncLoopTimes * ncBatch);
    if (ncTail == 0) {
        ncLoopTimes = ncLoopTimes - 1;
        ncTail = ncBatch;
    }
    uint32_t chunkElems = static_cast<uint32_t>(ncBatch) * dhwc;
    uint32_t tailElems = static_cast<uint32_t>(ncTail) * dhwc;

    uint32_t argH = static_cast<uint32_t>(wArgmaxActual_) * C;
    uint32_t argD = static_cast<uint32_t>(hArgmaxActual_) * argH;
    uint32_t argN = static_cast<uint32_t>(dArgmaxActual_) * argD;

    for (uint16_t ncLoop = 0; ncLoop < ncLoopTimes; ncLoop++) {
        ProcessWNDHWC4D(xAddr + static_cast<int64_t>(ncLoop) * ncBatch * ncBaseStep,
                        argmaxAddr + static_cast<int64_t>(ncLoop) * chunkElems, static_cast<uint16_t>(chunkElems), C,
                        wOutA, hOutA, dOutA, dKernel, hKernel, wKernel, colStride, rowStride, depStride, wBaseStep,
                        hBaseStep, dBaseStep, ncBaseStep, dOrigin, hOrigin, wOrigin, sD, sW, sH, hInput, wInput,
                        dDilation, hDilation, wDilation, isPad);
    }
    ProcessWNDHWC4D(xAddr + static_cast<int64_t>(ncLoopTimes) * ncBatch * ncBaseStep,
                    argmaxAddr + static_cast<int64_t>(ncLoopTimes) * chunkElems, static_cast<uint16_t>(tailElems), C,
                    wOutA, hOutA, dOutA, dKernel, hKernel, wKernel, colStride, rowStride, depStride, wBaseStep,
                    hBaseStep, dBaseStep, ncBaseStep, dOrigin, hOrigin, wOrigin, sD, sW, sH, hInput, wInput, dDilation,
                    hDilation, wDilation, isPad);
}

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPool3DGradNDHWCSmallKernel<T, INDEX_T, IS_CHECK_RANGE>::ForwardScalarCompute()
{
    inputStrideW_ = cDim_;
    inputStrideH_ = tilingData_.base.wOutput * cDim_;
    inputStrideD_ = tilingData_.base.hOutput * inputStrideH_;

    dInputActualPad_ = (dArgmaxActual_ - 1) * tilingData_.base.dStride +
                       (tilingData_.base.dKernel - 1) * tilingData_.base.dilationD + 1;
    hInputActualPad_ = (hArgmaxActual_ - 1) * tilingData_.base.hStride +
                       (tilingData_.base.hKernel - 1) * tilingData_.base.dilationH + 1;
    wInputActualPad_ = (wArgmaxActual_ - 1) * tilingData_.base.wStride +
                       (tilingData_.base.wKernel - 1) * tilingData_.base.dilationW + 1;

    ncBaseStep_ = static_cast<uint32_t>(dInputActualPad_ * hInputActualPad_ * wInputActualPad_ * cAligned_);
    dBaseStep_ = static_cast<uint32_t>(tilingData_.base.dStride * hInputActualPad_ * wInputActualPad_ * cAligned_);
    hBaseStep_ = static_cast<uint32_t>(tilingData_.base.hStride * wInputActualPad_ * cAligned_);

    if (isPad_) {
        int64_t lRelBound = wArgmaxActualStart_ * tilingData_.base.wStride - tilingData_.base.padW;
        int64_t rRelBound = lRelBound + wInputActualPad_ - tilingData_.base.wOutput;
        leftOffsetToInputLeft_ = lRelBound >= 0 ? 0 : -lRelBound;
        rightOffsetToInputRight_ = rRelBound > 0 ? rRelBound : 0;

        int64_t tRelBound = hArgmaxActualStart_ * tilingData_.base.hStride - tilingData_.base.padH;
        int64_t downRelBound = tRelBound + hInputActualPad_ - tilingData_.base.hOutput;
        topOffsetToInputTop_ = tRelBound >= 0 ? 0 : -tRelBound;
        downOffsetToInputDown_ = downRelBound > 0 ? downRelBound : 0;

        int64_t fRelBound = dArgmaxActualStart_ * tilingData_.base.dStride - tilingData_.base.padD;
        int64_t backRelBound = fRelBound + dInputActualPad_ - tilingData_.base.dOutput;
        frontOffsetToInputFront_ = fRelBound >= 0 ? 0 : -fRelBound;
        backOffsetToInputBack_ = backRelBound > 0 ? backRelBound : 0;

        dInputActualNoPad_ = dInputActualPad_ - frontOffsetToInputFront_ - backOffsetToInputBack_;
        hInputActualNoPad_ = hInputActualPad_ - topOffsetToInputTop_ - downOffsetToInputDown_;
        wInputActualNoPad_ = wInputActualPad_ - leftOffsetToInputLeft_ - rightOffsetToInputRight_;
    } else {
        leftOffsetToInputLeft_ = 0;
        rightOffsetToInputRight_ = 0;
        topOffsetToInputTop_ = 0;
        downOffsetToInputDown_ = 0;
        frontOffsetToInputFront_ = 0;
        backOffsetToInputBack_ = 0;
        dInputActualNoPad_ = dInputActualPad_;
        hInputActualNoPad_ = hInputActualPad_;
        wInputActualNoPad_ = wInputActualPad_;
    }

    int64_t dValidStart = dArgmaxActualStart_ * tilingData_.base.dStride - tilingData_.base.padD;
    dValidStart = dValidStart > 0 ? dValidStart : 0;
    int64_t hValidStart = hArgmaxActualStart_ * tilingData_.base.hStride - tilingData_.base.padH;
    hValidStart = hValidStart > 0 ? hValidStart : 0;
    int64_t wValidStart = wArgmaxActualStart_ * tilingData_.base.wStride - tilingData_.base.padW;
    wValidStart = wValidStart > 0 ? wValidStart : 0;

    forwardInputOffset_ = nAxisIndex_ * tilingData_.base.highAxisInner * tilingData_.base.dOutput *
                              tilingData_.base.hOutput * tilingData_.base.wOutput * cDim_ +
                          dValidStart * inputStrideD_ + hValidStart * inputStrideH_ + wValidStart * inputStrideW_ +
                          cAxisIndex_ * tilingData_.cOutputInner;
}

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPool3DGradNDHWCSmallKernel<T, INDEX_T, IS_CHECK_RANGE>::ForwardCopyIn()
{
    LocalTensor<T> xLocal = inputQue_.AllocTensor<T>();
    int64_t C = cOutputActual_;
    int64_t wLoad = isPad_ ? wInputActualNoPad_ : wInputActualPad_;
    int64_t hLoad = isPad_ ? hInputActualNoPad_ : hInputActualPad_;
    int64_t dLoad = isPad_ ? dInputActualNoPad_ : dInputActualPad_;
    int64_t rowElt = wLoad * cAligned_;

    constexpr uint32_t DIM_C = 0;
    constexpr uint32_t DIM_W = 1;
    constexpr uint32_t DIM_H = 2;
    constexpr uint32_t DIM_D = 3;
    constexpr uint32_t DIM_N = 4;
    MultiCopyLoopInfo<5> loopInfo;
    loopInfo.loopSize[DIM_C] = static_cast<uint16_t>(C);
    loopInfo.loopSize[DIM_W] = static_cast<uint16_t>(wLoad);
    loopInfo.loopSize[DIM_H] = static_cast<uint16_t>(hLoad);
    loopInfo.loopSize[DIM_D] = static_cast<uint16_t>(dLoad);
    loopInfo.loopSize[DIM_N] = static_cast<uint16_t>(nOutputActual_);
    loopInfo.loopSrcStride[DIM_C] = 1;
    loopInfo.loopSrcStride[DIM_W] = cDim_;
    loopInfo.loopSrcStride[DIM_H] = inputStrideH_;
    loopInfo.loopSrcStride[DIM_D] = inputStrideD_;
    loopInfo.loopSrcStride[DIM_N] = tilingData_.base.dOutput * inputStrideD_;
    loopInfo.loopDstStride[DIM_C] = 1;
    loopInfo.loopDstStride[DIM_W] = cAligned_;
    loopInfo.loopDstStride[DIM_H] = rowElt;
    loopInfo.loopDstStride[DIM_D] = hLoad * rowElt;
    loopInfo.loopDstStride[DIM_N] = dLoad * hLoad * rowElt;
    static constexpr MultiCopyConfig config = {false};
    MultiCopyParams<T, 5> paramsMain = {loopInfo};
    DataCopy<T, 5, config>(xLocal, xGm_[forwardInputOffset_], paramsMain);
    inputQue_.EnQue(xLocal);
}

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPool3DGradNDHWCSmallKernel<T, INDEX_T, IS_CHECK_RANGE>::Forward()
{
    LocalTensor<T> xLocal = inputQue_.DeQue<T>();
    xForwardAddr_ = (__ubuf__ T*)xLocal.GetPhyAddr();

    if (isPad_) {
        LocalTensor<T> calcLocal = inputCalcBuff_.Get<T>();
        __ubuf__ T* calcAddr = (__ubuf__ T*)calcLocal.GetPhyAddr();
        int64_t calcTotalElems = nOutputActual_ * dInputActualPad_ * hInputActualPad_ * wInputActualPad_ * cAligned_;
        uint32_t vl = static_cast<uint32_t>(V_REG_SIZE / sizeof(T));
        DupCalcNegInfVf(calcAddr, calcTotalElems, vl);
        CopyValidToCalcVf(
            calcAddr, xForwardAddr_, static_cast<uint32_t>(nOutputActual_), static_cast<uint32_t>(dInputActualNoPad_),
            static_cast<uint32_t>(hInputActualNoPad_), static_cast<uint32_t>(wInputActualNoPad_ * cAligned_),
            static_cast<uint32_t>(dInputActualPad_), static_cast<uint32_t>(hInputActualPad_),
            static_cast<uint32_t>(wInputActualPad_ * cAligned_),
            static_cast<uint32_t>(leftOffsetToInputLeft_ * cAligned_), static_cast<uint32_t>(frontOffsetToInputFront_),
            static_cast<uint32_t>(topOffsetToInputTop_), vl);
        xForwardAddr_ = calcAddr;
    }

    LocalTensor<INDEX_T> argmaxLocal = argmaxBuff_.template Get<INDEX_T>();
    Duplicate(argmaxLocal, INDEX_T(-1), argmaxBufferSize_ / sizeof(INDEX_T));
    SetFlag<HardEvent::V_MTE2>(vToMte2Event_);
    WaitFlag<HardEvent::V_MTE2>(vToMte2Event_);

    ForwardComputeTile();

    inputQue_.FreeTensor(xLocal);
}

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__simd_vf__ inline void MaxPool3DGradNDHWCSmallKernel<T, INDEX_T, IS_CHECK_RANGE>::DupCalcNegInfVf(__ubuf__ T* calcAddr,
                                                                                                   uint64_t totalElems,
                                                                                                   uint32_t vl)
{
    Reg::RegTensor<T> v0;
    PoolUtils::Compute::DuplicateNegInfRegVF<T>(v0);
    uint32_t loop = static_cast<uint32_t>(totalElems / vl);
    uint32_t tail = static_cast<uint32_t>(totalElems - loop * vl);
    Reg::MaskReg allMask = Reg::CreateMask<T, Reg::MaskPattern::ALL>();
    for (uint32_t i = 0; i < loop; i++) {
        Reg::StoreAlign(calcAddr, v0, allMask);
        calcAddr += vl;
    }
    if (tail > 0) {
        Reg::MaskReg tailMask = Reg::UpdateMask<T>(tail);
        Reg::StoreAlign(calcAddr, v0, tailMask);
    }
}

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__simd_vf__ inline void MaxPool3DGradNDHWCSmallKernel<T, INDEX_T, IS_CHECK_RANGE>::CopyValidToCalcVf(
    __ubuf__ T* calcAddr, __ubuf__ T* srcAddr, uint32_t nActual, uint32_t dNoPad, uint32_t hNoPad,
    uint32_t wNoPadRowElems, uint32_t dPad, uint32_t hPad, uint32_t wPadRowElems, uint32_t leftElems, uint32_t front,
    uint32_t top, uint32_t vl)
{
    Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_STORE>();

    uint32_t srcN = dNoPad * hNoPad * wNoPadRowElems;
    uint32_t srcD = hNoPad * wNoPadRowElems;
    uint32_t srcH = wNoPadRowElems;
    uint32_t dstN = dPad * hPad * wPadRowElems;
    uint32_t dstD = hPad * wPadRowElems;
    uint32_t dstH = wPadRowElems;

    uint16_t loopCols = static_cast<uint16_t>(wNoPadRowElems / vl);
    uint16_t tailCols = static_cast<uint16_t>(wNoPadRowElems - loopCols * vl);

    Reg::RegTensor<T> v0;
    Reg::UnalignReg u0;

    for (uint32_t n = 0; n < nActual; n++) {
        for (uint32_t d = 0; d < dNoPad; d++) {
            for (uint32_t h = 0; h < hNoPad; h++) {
                __ubuf__ T* srcCur = srcAddr + n * srcN + d * srcD + h * srcH;
                __ubuf__ T* dstCur = calcAddr + n * dstN + (d + front) * dstD + (h + top) * dstH + leftElems;
                for (uint16_t k = 0; k < loopCols; k++) {
                    Reg::DataCopy<T, Reg::PostLiteral::POST_MODE_UPDATE>(v0, srcCur, vl);
                    Reg::DataCopyUnAlign(dstCur, v0, u0, vl);
                }
                Reg::DataCopy<T, Reg::PostLiteral::POST_MODE_UPDATE>(v0, srcCur, vl);
                Reg::DataCopyUnAlign(dstCur, v0, u0, tailCols);
                Reg::DataCopyUnAlignPost(dstCur, u0, 0);
            }
        }
    }
}

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__simd_callee__ inline void MaxPool3DGradNDHWCSmallKernel<T, INDEX_T, IS_CHECK_RANGE>::MaxPoolSingleChannelWithArgmax(
    __ubuf__ INDEX_T* argmaxAddr, __ubuf__ T* srcAddr, uint16_t kD, uint16_t kH, uint16_t kW, uint32_t depStride,
    uint32_t rowStride, uint32_t colStride, uint16_t repeatElms, int32_t curInD, int32_t curInH, int32_t curInW,
    int32_t hInput, int32_t wInput, uint32_t dDilation, uint32_t hDilation, uint32_t wDilation)
{
    Reg::RegTensor<T> res;
    Reg::RegTensor<T> vd0;
    Reg::RegTensor<INDEX_T> argmaxRes;
    Reg::RegTensor<INDEX_T> candReg;

    Reg::MaskReg allMaskIdx = Reg::CreateMask<INDEX_T, Reg::MaskPattern::ALL>();
    uint32_t num = repeatElms;
    Reg::MaskReg p0 = Reg::UpdateMask<T>(num);
    Reg::MaskReg gtMask, neMask;

    PoolUtils::Compute::DuplicateNegInfRegVF<T>(res);
    const int32_t firstValidD = curInD < 0 ? 0 : curInD;
    const int32_t firstValidH = curInH < 0 ? 0 : curInH;
    const int32_t firstValidW = curInW < 0 ? 0 : curInW;
    const int32_t initIndex = firstValidD * hInput * wInput + firstValidH * wInput + firstValidW;
    Reg::Duplicate(argmaxRes, static_cast<INDEX_T>(initIndex));

    for (uint16_t dIdx = 0; dIdx < kD; dIdx++) {
        for (uint16_t hIdx = 0; hIdx < kH; hIdx++) {
            for (uint16_t wIdx = 0; wIdx < kW; wIdx++) {
                Reg::AddrReg aReg = Reg::CreateAddrReg<T>(dIdx, depStride, hIdx, rowStride, wIdx, colStride);
                Reg::LoadAlign(vd0, srcAddr, aReg);

                Reg::Compare<T, CMPMODE::GT>(gtMask, vd0, res, p0);
                Reg::Compare<T, CMPMODE::NE>(neMask, vd0, vd0, p0);
                Reg::MaskOr(gtMask, gtMask, neMask, p0);

                int32_t kernelIndex = (curInD + static_cast<int32_t>(dIdx * dDilation)) * hInput * wInput +
                                      (curInH + static_cast<int32_t>(hIdx * hDilation)) * wInput +
                                      (curInW + static_cast<int32_t>(wIdx * wDilation));
                Reg::Duplicate(candReg, static_cast<INDEX_T>(kernelIndex));

                if constexpr (sizeof(INDEX_T) / sizeof(T) == 1) {
                    Reg::Select(argmaxRes, candReg, argmaxRes, gtMask);
                } else {
                    Reg::MaskReg gtMaskUnpack;
                    Reg::UnPack(gtMaskUnpack, gtMask);
                    Reg::Select(argmaxRes, candReg, argmaxRes, gtMaskUnpack);
                }

                Reg::Max(res, vd0, res, p0);
            }
        }
    }

    Reg::UnalignReg u1;
    Reg::DataCopyUnAlign(argmaxAddr, argmaxRes, u1, repeatElms);
    Reg::DataCopyUnAlignPost(argmaxAddr, u1, 0);
}

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__simd_vf__ inline void MaxPool3DGradNDHWCSmallKernel<T, INDEX_T, IS_CHECK_RANGE>::ProcessBigCPosition(
    __ubuf__ INDEX_T* argmaxAddr, __ubuf__ T* srcAddr, uint16_t cLoop, uint16_t tailNum, uint16_t vl, uint16_t dKernel,
    uint16_t hKernel, uint16_t wKernel, uint32_t depStride, uint32_t rowStride, uint32_t colStride, int32_t curInD,
    int32_t curInH, int32_t curInW, int32_t hInput, int32_t wInput, uint32_t dDilation, uint32_t hDilation,
    uint32_t wDilation)
{
    for (uint16_t m = 0; m < cLoop; m++) {
        MaxPoolSingleChannelWithArgmax(argmaxAddr + m * vl, srcAddr + m * vl, dKernel, hKernel, wKernel, depStride,
                                       rowStride, colStride, vl, curInD, curInH, curInW, hInput, wInput, dDilation,
                                       hDilation, wDilation);
    }
    MaxPoolSingleChannelWithArgmax(argmaxAddr + cLoop * vl, srcAddr + cLoop * vl, dKernel, hKernel, wKernel, depStride,
                                   rowStride, colStride, tailNum, curInD, curInH, curInW, hInput, wInput, dDilation,
                                   hDilation, wDilation);
}

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPool3DGradNDHWCSmallKernel<T, INDEX_T, IS_CHECK_RANGE>::ChunkCGather(
    __ubuf__ T* xAddr, __ubuf__ INDEX_T* argmaxAddr)
{
    uint16_t C = static_cast<uint16_t>(cOutputActual_);
    uint16_t vl = static_cast<uint16_t>(V_REG_SIZE / sizeof(INDEX_T));
    uint16_t cLoop = C / vl;
    uint16_t tailNum = static_cast<uint16_t>(C - cLoop * vl);

    uint32_t colStride = static_cast<uint32_t>(cAligned_ * tilingData_.base.dilationW);
    uint32_t rowStride = static_cast<uint32_t>(wInputActualPad_ * cAligned_ * tilingData_.base.dilationH);
    uint32_t depStride = static_cast<uint32_t>(hInputActualPad_ * wInputActualPad_ * cAligned_ *
                                               tilingData_.base.dilationD);
    uint16_t dKernel = static_cast<uint16_t>(tilingData_.base.dKernel);
    uint16_t hKernel = static_cast<uint16_t>(tilingData_.base.hKernel);
    uint16_t wKernel = static_cast<uint16_t>(tilingData_.base.wKernel);

    int32_t hInput = static_cast<int32_t>(tilingData_.base.hOutput);
    int32_t wInput = static_cast<int32_t>(tilingData_.base.wOutput);
    uint32_t dDilation = tilingData_.base.dilationD;
    uint32_t hDilation = tilingData_.base.dilationH;
    uint32_t wDilation = tilingData_.base.dilationW;

    uint32_t argH = static_cast<uint32_t>(wArgmaxActual_) * C;
    uint32_t argD = static_cast<uint32_t>(hArgmaxActual_) * argH;
    uint32_t argN = static_cast<uint32_t>(dArgmaxActual_) * argD;
    int64_t planeIn = static_cast<int64_t>(hInputActualPad_) * wInputActualPad_ * cAligned_;
    int64_t batchIn = dInputActualPad_ * planeIn;

    for (int64_t n = 0; n < nOutputActual_; n++) {
        for (int64_t d = 0; d < dArgmaxActual_; d++) {
            for (int64_t h = 0; h < hArgmaxActual_; h++) {
                for (int64_t w = 0; w < wArgmaxActual_; w++) {
                    int32_t curInD = static_cast<int32_t>((dArgmaxActualStart_ + d) * tilingData_.base.dStride -
                                                          tilingData_.base.padD);
                    int32_t curInH = static_cast<int32_t>((hArgmaxActualStart_ + h) * tilingData_.base.hStride -
                                                          tilingData_.base.padH);
                    int32_t curInW = static_cast<int32_t>((wArgmaxActualStart_ + w) * tilingData_.base.wStride -
                                                          tilingData_.base.padW);

                    int64_t srcOff = n * batchIn + d * tilingData_.base.dStride * planeIn +
                                     h * tilingData_.base.hStride *
                                         (static_cast<int64_t>(wInputActualPad_) * cAligned_) +
                                     w * tilingData_.base.wStride * cAligned_;
                    int64_t argOff = static_cast<int64_t>(n) * argN + d * argD + h * argH + static_cast<int64_t>(w) * C;

                    ProcessBigCPosition(argmaxAddr + argOff, xAddr + srcOff, cLoop, tailNum, vl, dKernel, hKernel,
                                        wKernel, depStride, rowStride, colStride, curInD, curInH, curInW, hInput,
                                        wInput, dDilation, hDilation, wDilation);
                }
            }
        }
    }
}

} // namespace MaxPool3DGradNDHWCNameSpace
#endif
