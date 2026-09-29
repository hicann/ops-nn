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
 * @file max_pool3d_grad_ndhwc_big_kernel.h
 * @brief NDHWC MaxPool3DGrad 大 kernel 模板（kv>=128）
 *
 * 组合: MaxPool3DGradNDHWCBigKernel : public MaxPool3DGradNDHWCBackwardBase
 * （反向基类见 max_pool3d_grad_ndhwc_impl_scatter.h, 大小 kernel 共用）
 */

#ifndef MAX_POOL3D_GRAD_NDHWC_BIG_KERNEL_H_
#define MAX_POOL3D_GRAD_NDHWC_BIG_KERNEL_H_

#include "max_pool3d_grad_ndhwc_impl_scatter.h"
#include "pool_utils/arch35/index/max_pool_with_argmax_index.h"
#include "pool_utils/arch35/compute/max_pool_negative_value.h"

namespace MaxPool3DGradNDHWCNameSpace {

// ==================== 大 kernel 模板类 ====================

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
class MaxPool3DGradNDHWCBigKernel : public MaxPool3DGradNDHWCBackwardBase<T, INDEX_T, IS_CHECK_RANGE> {
    using Base = MaxPool3DGradNDHWCBackwardBase<T, INDEX_T, IS_CHECK_RANGE>;

public:
    __aicore__ inline MaxPool3DGradNDHWCBigKernel() {}

    __aicore__ inline void Init(GM_ADDR origX, GM_ADDR origY, GM_ADDR grad, GM_ADDR y,
                                const Pool3DGradNDHWCTilingData& tilingData);
    __aicore__ inline void Process();

    using Base::argmaxBuff_;
    using Base::argmaxBufferSize_;
    using Base::blockIdx_;
    using Base::cAligned_;
    using Base::cAxisGradOffset_;
    using Base::cDim_;
    using Base::cOutputActual_;
    using Base::curCoreProcessNum_;
    using Base::dArgmaxActual_;
    using Base::dArgmaxActualStart_;
    using Base::hArgmaxActual_;
    using Base::hArgmaxActualStart_;
    using Base::helpBuf_;
    using Base::nAxisIndex_;
    using Base::nOutputActual_;
    using Base::tilingData_;
    using Base::wArgmaxActual_;
    using Base::wArgmaxActualStart_;

private:
    __aicore__ inline void ForwardBigKernel();
    __aicore__ inline void CalcRowKernelSize(int64_t nIdx, int64_t dIdx, int64_t hIdx, int64_t& curKD, int64_t& curKH,
                                             int64_t& rowOriginFlat, int64_t& rowInOffset);
    __aicore__ inline void CalcKernelSize(int64_t nIdx, int64_t dIdx, int64_t hIdx, int64_t wIdx, int64_t& curKD,
                                          int64_t& curKH, int64_t& curKW, int64_t& curInOffset,
                                          int64_t& curOriginIndex);
    __aicore__ inline void NoSplitKernelProcess(int64_t curKD, int64_t curKH, int64_t curKW, int64_t curInOffset,
                                                int64_t curOriginIndex, int64_t argmaxOffset);
    __aicore__ inline void SplitKernelProcess(int64_t curKD, int64_t curKH, int64_t curKW, int64_t curInOffset,
                                              int64_t curOriginIndex, int64_t argmaxOffset);
    __aicore__ inline void InitMergeBuffer(int64_t argmaxOffset, int64_t curOriginIndex);
    template <bool MERGE = false>
    __aicore__ inline void ComputeSingleArgmax(__local_mem__ T* xSrcBase, int64_t rowStride, int64_t planeStride,
                                               int64_t curKW, int64_t curKH, int64_t curKD, int64_t curOriginIndex,
                                               int64_t argmaxOffset);
    __aicore__ inline void CopyInMultiRows_3D(int64_t offset, int64_t blockLen, int64_t blockCount, int64_t loopD);

    TPipe pipe_;
    GlobalTensor<T> xGm_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQue_;
    TBuf<TPosition::VECCALC> maxValBuf_;
    int64_t maxCount_ = 1;

    TBufPool<TPosition::VECCALC> forwardBufPool_;
    TBufPool<TPosition::VECCALC> backwardBufPool_;
    int64_t totalStageBufferSize_ = 0;
};

// ==================== Init / Process ====================

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPool3DGradNDHWCBigKernel<T, INDEX_T, IS_CHECK_RANGE>::Init(
    GM_ADDR origX, GM_ADDR origY, GM_ADDR grad, GM_ADDR y, const Pool3DGradNDHWCTilingData& tilingData)
{
    Base::ParseTilingData(tilingData);
    Base::vToMte2Event_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    Base::mte3ToMte2Event_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));

    Base::blockIdx_ = GetBlockIdx();
    Base::curCoreProcessNum_ = (Base::blockIdx_ == tilingData_.base.usedCoreNum - 1) ?
                                   tilingData_.base.tailCoreProcessNum :
                                   tilingData_.base.normalCoreProcessNum;

    xGm_.SetGlobalBuffer((__gm__ T*)origX);
    Base::gradGm_.SetGlobalBuffer((__gm__ T*)grad);
    Base::yGm_.SetGlobalBuffer((__gm__ T*)y);

    pipe_.InitBuffer(Base::argmaxBuff_, tilingData_.base.argmaxBufferSize);
    pipe_.InitBuffer(Base::helpBuf_, HELP_BUFFER);

    maxCount_ = tilingData_.base.inputBufferSize / static_cast<int64_t>(sizeof(T));
    int64_t mergeBufSize = cAligned_ * static_cast<int64_t>(sizeof(T));
    int64_t forwardSize = tilingData_.base.inputBufferSize * BUFFER_NUM + mergeBufSize;
    int64_t backwardSize = tilingData_.base.gradBufferSize + tilingData_.base.outputBufferSize;
    totalStageBufferSize_ = (forwardSize > backwardSize) ? forwardSize : backwardSize;

    pipe_.InitBufPool(forwardBufPool_, static_cast<uint32_t>(totalStageBufferSize_));
    pipe_.InitBufPool(backwardBufPool_, static_cast<uint32_t>(totalStageBufferSize_), forwardBufPool_);

    forwardBufPool_.InitBuffer(inputQue_, BUFFER_NUM, static_cast<uint32_t>(tilingData_.base.inputBufferSize));
    forwardBufPool_.InitBuffer(maxValBuf_, static_cast<uint32_t>(cAligned_ * sizeof(T)));
    backwardBufPool_.InitBuffer(Base::outputQue_, 1, static_cast<uint32_t>(tilingData_.base.outputBufferSize));
    backwardBufPool_.InitBuffer(Base::gradQue_, 1, static_cast<uint32_t>(tilingData_.base.gradBufferSize));
}

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPool3DGradNDHWCBigKernel<T, INDEX_T, IS_CHECK_RANGE>::Process()
{
    if (blockIdx_ >= tilingData_.base.usedCoreNum) {
        return;
    }

    for (int64_t loopNum = 0; loopNum < curCoreProcessNum_; ++loopNum) {
        Base::ScalarCompute(loopNum);

        if (dArgmaxActual_ == 0 || hArgmaxActual_ == 0 || wArgmaxActual_ == 0) {
            Base::ProcessNoArgmaxBlock();
            Base::Mte3Drain();
            continue;
        }

        ForwardBigKernel();

        Base::VDrainToMte2();

        Base::CopyInGrad();
        Base::Backward();
        Base::CopyOut();
        Base::Mte3Drain();
    }
}

// ==================== ForwardBigKernel ====================

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPool3DGradNDHWCBigKernel<T, INDEX_T, IS_CHECK_RANGE>::ForwardBigKernel()
{
    LocalTensor<INDEX_T> argmaxLocal = argmaxBuff_.template Get<INDEX_T>();
    Duplicate<INDEX_T>(argmaxLocal, static_cast<INDEX_T>(-1),
                       static_cast<uint32_t>(argmaxBufferSize_ / sizeof(INDEX_T)));
    PipeBarrier<PIPE_V>();

    for (int64_t nIdx = 0; nIdx < nOutputActual_; nIdx++) {
        for (int64_t dIdx = 0; dIdx < dArgmaxActual_; dIdx++) {
            for (int64_t hIdx = 0; hIdx < hArgmaxActual_; hIdx++) {
                for (int64_t wIdx = 0; wIdx < wArgmaxActual_; wIdx++) {
                    int64_t curKD = 0;
                    int64_t curKH = 0;
                    int64_t curKW = 0;
                    int64_t curInOffset = 0;
                    int64_t curOriginIndex = 0;
                    CalcKernelSize(nIdx, dIdx, hIdx, wIdx, curKD, curKH, curKW, curInOffset, curOriginIndex);
                    if (curKD <= 0 || curKH <= 0 || curKW <= 0) {
                        continue;
                    }
                    const int64_t argmaxOffset = ((nIdx * dArgmaxActual_ + dIdx) * hArgmaxActual_ + hIdx) *
                                                     wArgmaxActual_ * cOutputActual_ +
                                                 wIdx * cOutputActual_;
                    if (curKD * curKH * curKW * cAligned_ <= maxCount_) {
                        NoSplitKernelProcess(curKD, curKH, curKW, curInOffset, curOriginIndex, argmaxOffset);
                    } else {
                        SplitKernelProcess(curKD, curKH, curKW, curInOffset, curOriginIndex, argmaxOffset);
                    }
                }
            }
        }
    }
}

// ==================== CalcKernelSize ====================

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPool3DGradNDHWCBigKernel<T, INDEX_T, IS_CHECK_RANGE>::CalcRowKernelSize(
    int64_t nIdx, int64_t dIdx, int64_t hIdx, int64_t& curKD, int64_t& curKH, int64_t& rowOriginFlat,
    int64_t& rowInOffset)
{
    const int64_t dOut = dArgmaxActualStart_ + dIdx;
    const int64_t hOut = hArgmaxActualStart_ + hIdx;

    int64_t curOriginD = dOut * tilingData_.base.dStride - tilingData_.base.padD;
    int64_t curOriginH = hOut * tilingData_.base.hStride - tilingData_.base.padH;
    curKD = tilingData_.base.dKernel;
    curKH = tilingData_.base.hKernel;

    if (curOriginD < 0) {
        curKD += curOriginD;
        curOriginD = 0;
    }
    if (curOriginD + curKD > tilingData_.base.dOutput) {
        curKD = tilingData_.base.dOutput - curOriginD;
    }
    if (curOriginH < 0) {
        curKH += curOriginH;
        curOriginH = 0;
    }
    if (curOriginH + curKH > tilingData_.base.hOutput) {
        curKH = tilingData_.base.hOutput - curOriginH;
    }

    rowOriginFlat = curOriginD * tilingData_.base.hOutput * tilingData_.base.wOutput +
                    curOriginH * tilingData_.base.wOutput;
    const int64_t nOffset = (nAxisIndex_ * tilingData_.base.highAxisInner + nIdx) * tilingData_.base.dOutput *
                            tilingData_.base.hOutput * tilingData_.base.wOutput;
    rowInOffset = (nOffset + rowOriginFlat) * cDim_ + cAxisGradOffset_;
}

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPool3DGradNDHWCBigKernel<T, INDEX_T, IS_CHECK_RANGE>::CalcKernelSize(
    int64_t nIdx, int64_t dIdx, int64_t hIdx, int64_t wIdx, int64_t& curKD, int64_t& curKH, int64_t& curKW,
    int64_t& curInOffset, int64_t& curOriginIndex)
{
    CalcRowKernelSize(nIdx, dIdx, hIdx, curKD, curKH, curOriginIndex, curInOffset);

    const int64_t wOut = wArgmaxActualStart_ + wIdx;
    int64_t curOriginW = wOut * tilingData_.base.wStride - tilingData_.base.padW;
    curKW = tilingData_.base.wKernel;
    if (curOriginW < 0) {
        curKW += curOriginW;
        curOriginW = 0;
    }
    if (curOriginW + curKW > tilingData_.base.wOutput) {
        curKW = tilingData_.base.wOutput - curOriginW;
    }

    curOriginIndex += curOriginW;
    curInOffset += curOriginW * cDim_;
}

// ==================== CopyInMultiRows_3D ====================

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPool3DGradNDHWCBigKernel<T, INDEX_T, IS_CHECK_RANGE>::CopyInMultiRows_3D(int64_t offset,
                                                                                                   int64_t blockLen,
                                                                                                   int64_t blockCount,
                                                                                                   int64_t loopD)
{
    LocalTensor<T> xLocal = inputQue_.template AllocTensor<T>();
    DataCopyPadExtParams<T> padExtParams = {false, 0, 0, 0};
    DataCopyExtParams extParams;
    extParams.blockCount = static_cast<uint32_t>(blockLen);
    extParams.blockLen = static_cast<uint32_t>(cOutputActual_ * sizeof(T));
    extParams.srcStride = static_cast<uint32_t>((cDim_ - cOutputActual_) * sizeof(T));
    extParams.dstStride = static_cast<uint32_t>((cAligned_ - cOutputActual_) * sizeof(T)) / platform::GetUbBlockSize();
    extParams.rsv = 0;
    LoopModeParams loopParams;
    loopParams.loop2Size = static_cast<uint32_t>(loopD);
    loopParams.loop1Size = static_cast<uint32_t>(blockCount);
    loopParams.loop1SrcStride = static_cast<uint32_t>(tilingData_.base.wOutput * cDim_ * sizeof(T));
    loopParams.loop1DstStride = static_cast<uint32_t>(blockLen * cAligned_ * sizeof(T));
    loopParams.loop2SrcStride = static_cast<uint32_t>(tilingData_.base.hOutput * tilingData_.base.wOutput * cDim_ *
                                                      sizeof(T));
    loopParams.loop2DstStride = static_cast<uint32_t>(blockCount * blockLen * cAligned_ * sizeof(T));
    SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
    DataCopyPad<T>(xLocal, xGm_[offset], extParams, padExtParams);
    ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
    inputQue_.template EnQue<T>(xLocal);
}

// ==================== NoSplitKernelProcess ====================

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPool3DGradNDHWCBigKernel<T, INDEX_T, IS_CHECK_RANGE>::NoSplitKernelProcess(
    int64_t curKD, int64_t curKH, int64_t curKW, int64_t curInOffset, int64_t curOriginIndex, int64_t argmaxOffset)
{
    CopyInMultiRows_3D(curInOffset, curKW, curKH, curKD);
    LocalTensor<T> xLocal = inputQue_.template DeQue<T>();
    __local_mem__ T* xLocalAddr = (__local_mem__ T*)xLocal.GetPhyAddr();
    const int64_t rowStride = curKW * cAligned_;
    ComputeSingleArgmax<false>(xLocalAddr, rowStride, curKH * rowStride, curKW, curKH, curKD, curOriginIndex,
                               argmaxOffset);
    inputQue_.template FreeTensor<T>(xLocal);
}

// ==================== SplitKernelProcess / InitMergeBuffer ====================

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPool3DGradNDHWCBigKernel<T, INDEX_T, IS_CHECK_RANGE>::SplitKernelProcess(
    int64_t curKD, int64_t curKH, int64_t curKW, int64_t curInOffset, int64_t curOriginIndex, int64_t argmaxOffset)
{
    InitMergeBuffer(argmaxOffset, curOriginIndex);
    if (curKW <= 0 || curKH <= 0 || curKD <= 0 || maxCount_ <= 0) {
        return;
    }

    if (curKW * cAligned_ <= maxCount_) {
        const int64_t hFactor = maxCount_ / (curKW * cAligned_);
        for (int64_t d = 0; d < curKD; d++) {
            for (int64_t hStart = 0; hStart < curKH; hStart += hFactor) {
                const int64_t curh = (curKH - hStart < hFactor) ? curKH - hStart : hFactor;
                const int64_t rowAdvance = (d * tilingData_.base.hOutput + hStart) * tilingData_.base.wOutput;
                CopyInMultiRows_3D(curInOffset + rowAdvance * cDim_, curKW, curh, 1);
                LocalTensor<T> xLocal = inputQue_.template DeQue<T>();
                __local_mem__ T* xLocalAddr = (__local_mem__ T*)xLocal.GetPhyAddr();
                const int64_t rowStride = curKW * cAligned_;
                ComputeSingleArgmax<true>(xLocalAddr, rowStride, curh * rowStride, curKW, curh, 1,
                                          curOriginIndex + rowAdvance, argmaxOffset);
                inputQue_.template FreeTensor<T>(xLocal);
            }
        }
    } else {
        const int64_t wFactor = maxCount_ / cAligned_;
        for (int64_t d = 0; d < curKD; d++) {
            for (int64_t h = 0; h < curKH; h++) {
                const int64_t rowAdvance = (d * tilingData_.base.hOutput + h) * tilingData_.base.wOutput;
                for (int64_t wStart = 0; wStart < curKW; wStart += wFactor) {
                    const int64_t curw = (curKW - wStart < wFactor) ? curKW - wStart : wFactor;
                    CopyInMultiRows_3D(curInOffset + (rowAdvance + wStart) * cDim_, curw, 1, 1);
                    LocalTensor<T> xLocal = inputQue_.template DeQue<T>();
                    __local_mem__ T* xLocalAddr = (__local_mem__ T*)xLocal.GetPhyAddr();
                    const int64_t rowStride = curw * cAligned_;
                    ComputeSingleArgmax<true>(xLocalAddr, rowStride, rowStride, curw, 1, 1,
                                              curOriginIndex + rowAdvance + wStart, argmaxOffset);
                    inputQue_.template FreeTensor<T>(xLocal);
                }
            }
        }
    }
}

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
__aicore__ inline void MaxPool3DGradNDHWCBigKernel<T, INDEX_T, IS_CHECK_RANGE>::InitMergeBuffer(int64_t argmaxOffset,
                                                                                                int64_t curOriginIndex)
{
    LocalTensor<INDEX_T> argmaxLocal = argmaxBuff_.template Get<INDEX_T>();
    __ubuf__ INDEX_T* argmaxAddr = (__ubuf__ INDEX_T*)argmaxLocal.GetPhyAddr();
    LocalTensor<T> maxValLocal = maxValBuf_.template Get<T>();
    __ubuf__ T* maxValAddr = (__ubuf__ T*)maxValLocal.GetPhyAddr();
    constexpr int64_t vlT = platform::GetVRegSize() / sizeof(T);
    constexpr int64_t vlIdx = platform::GetVRegSize() / sizeof(int32_t);
    constexpr int64_t vlStep = vlT < vlIdx ? vlT : vlIdx;
    for (int64_t c0 = 0; c0 < cOutputActual_; c0 += vlStep) {
        const int64_t vc = (cOutputActual_ - c0 < vlStep) ? cOutputActual_ - c0 : vlStep;
        __VEC_SCOPE__
        {
            Reg::MaskReg maskAll = Reg::CreateMask<T, Reg::MaskPattern::ALL>();
            Reg::RegTensor<T> negInfReg;
            Reg::RegTensor<int32_t> initIdxReg;
            Reg::UnalignRegForStore u0;
            PoolUtils::Compute::DuplicateNegInfReg<T>(negInfReg);
            Reg::Duplicate(initIdxReg, static_cast<int32_t>(curOriginIndex), maskAll);
            __ubuf__ T* maxDst = maxValAddr + c0;
            Reg::StoreUnAlign(maxDst, negInfReg, u0, static_cast<uint32_t>(vc));
            Reg::StoreUnAlignPost(maxDst, u0, 0);
            __ubuf__ uint32_t* idxDst = (__ubuf__ uint32_t*)(argmaxAddr + argmaxOffset + c0);
            Reg::RegTensor<uint32_t>& initStoreU32 = (Reg::RegTensor<uint32_t>&)initIdxReg;
            Reg::StoreUnAlign<uint32_t>(idxDst, initStoreU32, u0, static_cast<uint32_t>(vc));
            Reg::StoreUnAlignPost<uint32_t>(idxDst, u0, 0);
        }
    }
}

// ==================== ComputeSingleArgmax ====================

template <typename T, typename INDEX_T, const uint32_t IS_CHECK_RANGE>
template <bool MERGE>
__aicore__ inline void MaxPool3DGradNDHWCBigKernel<T, INDEX_T, IS_CHECK_RANGE>::ComputeSingleArgmax(
    __local_mem__ T* xSrcBase, int64_t rowStride, int64_t planeStride, int64_t curKW, int64_t curKH, int64_t curKD,
    int64_t curOriginIndex, int64_t argmaxOffset)
{
    LocalTensor<INDEX_T> argmaxLocal = argmaxBuff_.template Get<INDEX_T>();
    __local_mem__ INDEX_T* argmaxAddr = (__local_mem__ INDEX_T*)argmaxLocal.GetPhyAddr();
    LocalTensor<T> maxValLocal = maxValBuf_.template Get<T>();
    __local_mem__ T* maxValAddr = (__local_mem__ T*)maxValLocal.GetPhyAddr();
    const int64_t ubBlockElems = platform::GetUbBlockSize() / sizeof(T);
    const int64_t cAlignedC = (cOutputActual_ + ubBlockElems - 1) / ubBlockElems * ubBlockElems;
    constexpr int64_t vlT = platform::GetVRegSize() / sizeof(T);
    constexpr int64_t vlIdx = platform::GetVRegSize() / sizeof(int32_t);
    constexpr int64_t vlStep = vlT < vlIdx ? vlT : vlIdx;
    for (int64_t c0 = 0; c0 < cOutputActual_; c0 += vlStep) {
        int64_t vc = cOutputActual_ - c0 < vlStep ? cOutputActual_ - c0 : vlStep;
        __VEC_SCOPE__
        {
            Reg::MaskReg maskAll = Reg::CreateMask<T, Reg::MaskPattern::ALL>();
            Reg::RegTensor<T> xReg, maxReg;
            Reg::RegTensor<int32_t> idxReg, candIdx;
            Reg::MaskReg gtMask, nanMask, updMask;
            if constexpr (MERGE) {
                Reg::LocalMemBar<Reg::MemType::VEC_STORE, Reg::MemType::VEC_LOAD>();
                Reg::UnalignRegForLoad ul0;
                __ubuf__ T* maxSrc = (__ubuf__ T*)maxValAddr + c0;
                Reg::LoadUnAlignPre(ul0, maxSrc);
                Reg::LoadUnAlign(maxReg, ul0, maxSrc, static_cast<uint32_t>(vc));
                __ubuf__ int32_t* idxSrc = (__ubuf__ int32_t*)argmaxAddr + argmaxOffset + c0;
                Reg::LoadUnAlignPre(ul0, idxSrc);
                Reg::LoadUnAlign(idxReg, ul0, idxSrc, static_cast<uint32_t>(vc));
            } else {
                PoolUtils::Compute::DuplicateNegInfReg<T>(maxReg);
                Reg::Duplicate(idxReg, static_cast<int32_t>(curOriginIndex), maskAll);
            }
            const int32_t inW = static_cast<int32_t>(tilingData_.base.wOutput);
            const int32_t inHw = static_cast<int32_t>(tilingData_.base.hOutput * tilingData_.base.wOutput);
            Reg::RegTensor<int32_t> hFlatReg;
            for (uint16_t d = 0; d < static_cast<uint16_t>(curKD); d++) {
                const int32_t dFlat = static_cast<int32_t>(curOriginIndex) + d * inHw;
                for (uint16_t h = 0; h < static_cast<uint16_t>(curKH); h++) {
                    const int32_t hFlat = dFlat + h * inW;
                    Reg::Duplicate(hFlatReg, hFlat, maskAll);
                    for (uint16_t w = 0; w < static_cast<uint16_t>(curKW); w++) {
                        auto srcAddr = xSrcBase + d * planeStride + h * rowStride + w * cAlignedC + c0;
                        Reg::DataCopy(xReg, srcAddr);
                        Reg::Compare<T, CMPMODE::NE>(nanMask, xReg, xReg, maskAll);
                        Reg::Compare<T, CMPMODE::GT>(gtMask, xReg, maxReg, maskAll);
                        Reg::MaskXor(updMask, gtMask, nanMask, maskAll);
                        Reg::Max(maxReg, xReg, maxReg, maskAll);
                        Reg::Adds(candIdx, hFlatReg, static_cast<int32_t>(w), maskAll);
                        if constexpr (sizeof(int32_t) / sizeof(T) == 2) {
                            Reg::MaskReg updMaskB32;
                            Reg::UnPack(updMaskB32, updMask);
                            Reg::Select(idxReg, candIdx, idxReg, updMaskB32);
                        } else {
                            Reg::Select(idxReg, candIdx, idxReg, updMask);
                        }
                    }
                }
            }
            Reg::UnalignRegForStore u0;
            Reg::RegTensor<uint32_t>& idxStoreU32 = (Reg::RegTensor<uint32_t>&)idxReg;
            __ubuf__ uint32_t* idxDst = (__ubuf__ uint32_t*)(argmaxAddr + argmaxOffset + c0);
            if constexpr (MERGE) {
                Reg::LocalMemBar<Reg::MemType::VEC_LOAD, Reg::MemType::VEC_STORE>();
                __ubuf__ T* maxDst = maxValAddr + c0;
                Reg::StoreUnAlign(maxDst, maxReg, u0, static_cast<uint32_t>(vc));
                Reg::StoreUnAlignPost(maxDst, u0, 0);
            }
            Reg::StoreUnAlign<uint32_t>(idxDst, idxStoreU32, u0, static_cast<uint32_t>(vc));
            Reg::StoreUnAlignPost<uint32_t>(idxDst, u0, 0);
        }
    }
}

} // namespace MaxPool3DGradNDHWCNameSpace

#endif // MAX_POOL3D_GRAD_NDHWC_BIG_KERNEL_H_
