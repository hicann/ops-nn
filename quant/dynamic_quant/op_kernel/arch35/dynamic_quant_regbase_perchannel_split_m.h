/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the 'License').
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dynamic_quant_regbase_perchannel_split_m.h
 * \brief
 */
#ifndef DYNAMIC_QUANT_REGBASE_PERCHANNEL_SPLIT_M_H
#define DYNAMIC_QUANT_REGBASE_PERCHANNEL_SPLIT_M_H

#include "dynamic_quant_regbase_base.h"
#include "dynamic_quant_regbase_perchannel_base.h"
#include "../inc/kernel_utils.h"

namespace DynamicQuantPerChannel {
template <typename xDtype, typename yDtype, bool hasSmooth, bool isSymmetrical = true>
class DynamicQuantRegbasePerChannnelSplitM : public DynamicQuantNDOpt::DynamicQuantBase {
private:
    // 如果输出的数据类型是INT4，用INT8处理，其余的输出类型不变
    using yCopyDtype = std::conditional_t<IsSameType<yDtype, int4b_t>::value, uint8_t, yDtype>;

public:
    __aicore__ inline DynamicQuantRegbasePerChannnelSplitM(TPipe *pipe)
    {
        pPipe = pipe;
    }

    // 没有group_index输入
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR smooth_scales, GM_ADDR y, GM_ADDR scale, GM_ADDR offset,
        GM_ADDR workSpace, const DynamicQuantTilingData *__restrict tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ReInit();
    template <bool copyInMax>
    __aicore__ inline void CopyIn(uint32_t bIdx, uint32_t mIdx, uint32_t nIdx);
    __aicore__ inline void ParseTilingData(const DynamicQuantTilingData *tilingData);

    // __aicore__ inline void ComputeMinMax(uint32_t bIdx, uint32_t mIdx, uint32_t nIdx);
    __aicore__ inline void ComputeMinMax(__ubuf__ xDtype *inAddr, __ubuf__ xDtype *smoothAddr,
        __ubuf__ float *colMaxLocalAddr, __ubuf__ float *colMinLocalAddr, uint32_t nSize, uint16_t nLoopNum,
        uint16_t mLoopNum);

    __aicore__ inline void CopyOutMinMax(uint32_t bIdx, uint32_t nIdx, uint32_t count);
    __aicore__ inline void Compute(uint32_t bIdx, uint32_t mIdx, uint32_t nIdx);
    __aicore__ inline void PreProcessSingleBlock(uint32_t bIdx, uint32_t mIdx, uint32_t nIdx);
    __aicore__ inline void PostProcessSingleBlock(uint32_t bIdx, uint32_t mIdx, uint32_t nIdx);
    __aicore__ inline void CopyOut(uint32_t bIdx, uint32_t mIdx, uint32_t nIdx);

private:
    /* ascendc variable */
    TQue<QuePosition::VECIN, USE_BUFFER_NUM> inQueue_;
    TQue<QuePosition::VECIN, USE_BUFFER_NUM> smoothQueue_;
    TQue<QuePosition::VECIN, USE_BUFFER_NUM> colMaxQueueIn_;
    TQue<QuePosition::VECIN, USE_BUFFER_NUM> colMinQueueIn_;

    TQue<QuePosition::VECOUT, USE_BUFFER_NUM> outQueue_;
    TQue<QuePosition::VECOUT, USE_BUFFER_NUM> scaleQueue_;
    TQue<QuePosition::VECOUT, USE_BUFFER_NUM> offsetQueue_;
    TQue<QuePosition::VECOUT, USE_BUFFER_NUM> colMaxQueue_;
    TQue<QuePosition::VECOUT, USE_BUFFER_NUM> colMinQueue_;

    /* global memory address */
    GlobalTensor<xDtype> inGm, smoothGm;
    GlobalTensor<float> colMaxGm;
    GlobalTensor<float> colMinGm;
    GlobalTensor<float> scaleGm;
    GlobalTensor<float> offsetGm;
    GlobalTensor<yCopyDtype> outGm;

    uint32_t coreNum_ = 0;
    uint32_t headCoreNum_ = 0;
    uint32_t totalBatchLen_ = 0;
    uint32_t mLen_ = 0;
    uint32_t mBlockSize_ = 0;
    uint32_t mTailBlockSize_ = 0;
    uint32_t mBlockNum_ = 0;
    uint32_t nLen_ = 0;
    uint32_t nBlockSize_ = 0;
    uint32_t nTailBlockSize_ = 0;
    uint32_t nTailBlockSizeAligned_ = 0;
    uint32_t nTailBlockSizeAlignedB8_ = 0;
    uint32_t nTailBlockSizePadding_ = 0;
    uint32_t colMaxPadding_ = 0;
    uint32_t nBlockNum_ = 0;
    uint32_t nBaseLoopNum_ = 0;
    uint32_t blockPerHead_ = 0;
    uint32_t blockPerTail_ = 0;
    uint32_t totalBlockNum_ = 0;

    uint32_t curCoreProcessNum_ = 0;
    uint32_t blockStart_ = 0;
    uint32_t blockEnd_ = 0;
    uint32_t batchStart_ = 0;
    uint32_t batchEnd_ = 0;
    uint32_t outBufferSize_ = 0;
    float maxValue_ = 0.0f;
    float offsetValue_ = 0.0f;
    float offsetDivValue_ = 0.0f;
    bool isAsymmetrical = false;
};

template <typename xDtype, typename yDtype, bool hasSmooth, bool isSymmetrical>
__aicore__ inline void DynamicQuantRegbasePerChannnelSplitM<xDtype, yDtype, hasSmooth, isSymmetrical>::Init(GM_ADDR x,
    GM_ADDR smooth_scales, GM_ADDR y, GM_ADDR scale, GM_ADDR offset, GM_ADDR workSpace,
    const DynamicQuantTilingData *__restrict tilingData)
{
    DynamicQuantNDOpt::SetFloatOverflowModeForRegbase<yDtype>();
    ParseTilingData(tilingData);
    blockIdx = GetBlockIdx();
    if (blockIdx >= coreNum_) {
        return;
    }
    SetMaxValue<yDtype>(maxValue_, offsetValue_, offsetDivValue_);

    // calc params
    curCoreProcessNum_ = blockIdx < headCoreNum_ ? blockPerHead_ : blockPerTail_;
    blockStart_ = blockIdx < headCoreNum_ ? blockIdx * blockPerHead_
                                          : headCoreNum_ * blockPerHead_ + (blockIdx - headCoreNum_) * blockPerTail_;
    blockEnd_ = blockStart_ + curCoreProcessNum_;

    batchStart_ = blockStart_ / (mBlockNum_ * nBlockNum_);
    batchEnd_ = blockEnd_ / (mBlockNum_ * nBlockNum_);

    // init buffers
    inGm.SetGlobalBuffer((__gm__ xDtype *)x);
    colMaxGm.SetGlobalBuffer((__gm__ float *)workSpace + (batchStart_ * nLen_));
    outGm.SetGlobalBuffer((__gm__ yCopyDtype *)y);
    scaleGm.SetGlobalBuffer((__gm__ float *)scale);
    InitGlobalMemory<float>(colMaxGm, (batchEnd_ - batchStart_) * nLen_, NEG_INFINITY);

    if constexpr (isSymmetrical == false) {
        colMinGm.SetGlobalBuffer((__gm__ float *)workSpace + (totalBatchLen_ * nLen_) + (batchStart_ * nLen_));
        PipeBarrier<PIPE_ALL>();
        InitGlobalMemory<float>(colMinGm, (batchEnd_ - batchStart_) * nLen_, POS_INFINITY);
    }

    TEventID eventIdMTE3ToMTE2 = GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2);
    SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2);
    WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIdMTE3ToMTE2);

    SyncAll<true>();
    if constexpr (hasSmooth) {
        smoothGm.SetGlobalBuffer((__gm__ xDtype *)smooth_scales);
        // smooth_scale is col-wise
        pPipe->InitBuffer(smoothQueue_, USE_BUFFER_NUM, mBlockSize_ * sizeof(xDtype));
    }

    pPipe->InitBuffer(inQueue_, USE_BUFFER_NUM, mBlockSize_ * nBlockSize_ * sizeof(xDtype));
    pPipe->InitBuffer(colMaxQueue_, USE_BUFFER_NUM, nBlockSize_ * sizeof(float));
    if constexpr (isSymmetrical == false) {
        offsetGm.SetGlobalBuffer((__gm__ float *)offset);
        pPipe->InitBuffer(colMinQueue_, USE_BUFFER_NUM, nBlockSize_ * sizeof(float));
    }
}

template <typename xDtype, typename yDtype, bool hasSmooth, bool isSymmetrical>
__aicore__ inline void DynamicQuantRegbasePerChannnelSplitM<xDtype, yDtype, hasSmooth, isSymmetrical>::ReInit()
{
    pPipe->Reset();
    if constexpr (hasSmooth) {
        // smooth_scale is col-wise
        pPipe->InitBuffer(smoothQueue_, USE_BUFFER_NUM, mBlockSize_ * sizeof(xDtype));
    }

    pPipe->InitBuffer(inQueue_, USE_BUFFER_NUM, mBlockSize_ * nBlockSize_ * sizeof(xDtype));
    pPipe->InitBuffer(colMaxQueueIn_, USE_BUFFER_NUM, nBlockSize_ * sizeof(float));

    pPipe->InitBuffer(outQueue_, USE_BUFFER_NUM, outBufferSize_ * sizeof(yCopyDtype));
    pPipe->InitBuffer(scaleQueue_, USE_BUFFER_NUM, nBlockSize_ * sizeof(float));

    if constexpr (isSymmetrical == false) {
        pPipe->InitBuffer(offsetQueue_, USE_BUFFER_NUM, nBlockSize_ * sizeof(float));
        pPipe->InitBuffer(colMinQueueIn_, USE_BUFFER_NUM, nBlockSize_ * sizeof(float));
    }
}

template <typename xDtype, typename yDtype, bool hasSmooth, bool isSymmetrical>
__aicore__ inline void DynamicQuantRegbasePerChannnelSplitM<xDtype, yDtype, hasSmooth, isSymmetrical>::ParseTilingData(
    const DynamicQuantTilingData *tilingData)
{
    coreNum_ = tilingData->coreNum;
    headCoreNum_ = tilingData->headCoreNum;
    totalBatchLen_ = tilingData->totalBatchLen;
    mLen_ = tilingData->mLen;
    mBlockSize_ = tilingData->mBlockSize;
    mTailBlockSize_ = tilingData->mTailBlockSize;
    mBlockNum_ = tilingData->mBlockNum;
    nLen_ = tilingData->nLen;
    nBlockSize_ = tilingData->nBlockSize;
    nTailBlockSize_ = tilingData->nTailBlockSize;
    nTailBlockSizeAligned_ = ops::CeilAlign(nTailBlockSize_, DynamicQuantNDOpt::SIXTEEN);
    if constexpr(IsSameType<yDtype, int4b_t>::value) {
        nTailBlockSizeAlignedB8_ = ops::CeilAlign(nTailBlockSize_, DynamicQuantNDOpt::SIXTY_FOUR);
    } else {
        nTailBlockSizeAlignedB8_ = ops::CeilAlign(nTailBlockSize_, DynamicQuantNDOpt::THIRTY_TWO);
    }

    nTailBlockSizePadding_ = nTailBlockSizeAligned_ - nTailBlockSize_;

    colMaxPadding_ = ops::CeilAlign(nTailBlockSize_, DynamicQuantNDOpt::EIGHT) - nTailBlockSize_;
    nBlockNum_ = tilingData->nBlockNum;
    nBaseLoopNum_ = tilingData->nBaseLoopNum;
    blockPerHead_ = tilingData->blockPerHead;
    blockPerTail_ = tilingData->blockPerTail;
    totalBlockNum_ = tilingData->totalBlockNum;
    if constexpr (IsSameType<yDtype, int4b_t>::value) {
        outBufferSize_ = mBlockSize_ * nBlockSize_ / 2;
    } else {
        outBufferSize_ = mBlockSize_ * nBlockSize_;
    }
}

template <typename xDtype, typename yDtype, bool hasSmooth, bool isSymmetrical>
__aicore__ inline void DynamicQuantRegbasePerChannnelSplitM<xDtype, yDtype, hasSmooth, isSymmetrical>::Process()
{
    if (blockIdx >= coreNum_) {
        return;
    }
    for (uint32_t idx = blockStart_; idx < blockEnd_; idx++) {
        uint32_t bIdx = idx / nBlockNum_ / mBlockNum_;
        uint32_t nIdx = (idx - (bIdx * nBlockNum_ * mBlockNum_)) / mBlockNum_;
        uint32_t mIdx = (idx - (bIdx * nBlockNum_ * mBlockNum_) - nIdx * mBlockNum_);
        PreProcessSingleBlock(bIdx, mIdx, nIdx);
    }

    PipeBarrier<PIPE_ALL>();
    SyncAll<true>();
    ReInit();

    for (uint32_t idx = blockStart_; idx < blockEnd_; idx++) {
        uint32_t bIdx = idx / nBlockNum_ / mBlockNum_;
        uint32_t nIdx = (idx - (bIdx * nBlockNum_ * mBlockNum_)) / mBlockNum_;
        uint32_t mIdx = (idx - (bIdx * nBlockNum_ * mBlockNum_) - nIdx * mBlockNum_);
        PostProcessSingleBlock(bIdx, mIdx, nIdx);
    }
}

template <typename xDtype, typename yDtype, bool hasSmooth, bool isSymmetrical>
__aicore__ inline void
    DynamicQuantRegbasePerChannnelSplitM<xDtype, yDtype, hasSmooth, isSymmetrical>::PreProcessSingleBlock(
        uint32_t bIdx, uint32_t mIdx, uint32_t nIdx)
{
    uint32_t count = nIdx + 1 == nBlockNum_ ? nTailBlockSize_ : nBlockSize_;
    uint32_t nSize = nIdx + 1 == nBlockNum_ ? nTailBlockSizeAligned_ : nBlockSize_;
    uint16_t nLoopNum = ops::CeilDiv(nSize, DynamicQuantNDOpt::SIXTY_FOUR);
    uint16_t mLoopNum = mIdx + 1 == mBlockNum_ ? mTailBlockSize_ : mBlockSize_;

    CopyIn<false>(bIdx, mIdx, nIdx);

    LocalTensor<xDtype> inLocal = inQueue_.template DeQue<xDtype>();
    LocalTensor<float> colMaxLocal = colMaxQueue_.template AllocTensor<float>();
    LocalTensor<float> colMinLocal;
    LocalTensor<xDtype> smoothLocal;

    __ubuf__ xDtype *inAddr = (__ubuf__ xDtype *)inLocal.GetPhyAddr();
    __ubuf__ xDtype *smoothAddr = nullptr;

    __ubuf__ float *colMaxLocalAddr = (__ubuf__ float *)colMaxLocal.GetPhyAddr();
    __ubuf__ float *colMinLocalAddr = nullptr;

    if constexpr (hasSmooth) {
        smoothLocal = smoothQueue_.template DeQue<xDtype>();
        smoothAddr = (__ubuf__ xDtype *)smoothLocal.GetPhyAddr();
    }
    if constexpr (!isSymmetrical) {
        colMinLocal = colMinQueue_.template AllocTensor<float>();
        colMinLocalAddr = (__ubuf__ float *)colMinLocal.GetPhyAddr();
    }

    ComputeMinMax(inAddr, smoothAddr, colMaxLocalAddr, colMinLocalAddr, nSize, nLoopNum, mLoopNum);

    inQueue_.template FreeTensor(inLocal);
    colMaxQueue_.template EnQue(colMaxLocal);
    if constexpr (hasSmooth) {
        smoothQueue_.template FreeTensor(smoothLocal);
    }
    if constexpr (!isSymmetrical) {
        colMinQueue_.template EnQue(colMinLocal);
    }
    CopyOutMinMax(bIdx, nIdx, count);
}

template <typename xDtype, typename yDtype, bool hasSmooth, bool isSymmetrical>
__aicore__ inline void
    DynamicQuantRegbasePerChannnelSplitM<xDtype, yDtype, hasSmooth, isSymmetrical>::PostProcessSingleBlock(
        uint32_t bIdx, uint32_t mIdx, uint32_t nIdx)
{
    CopyIn<true>(bIdx, mIdx, nIdx);
    Compute(bIdx, mIdx, nIdx);
    CopyOut(bIdx, mIdx, nIdx);
}

template <typename xDtype, typename yDtype, bool hasSmooth, bool isSymmetrical>
template <bool copyInMax>
__aicore__ inline void DynamicQuantRegbasePerChannnelSplitM<xDtype, yDtype, hasSmooth, isSymmetrical>::CopyIn(
    uint32_t bIdx, uint32_t mIdx, uint32_t nIdx)
{
    uint64_t xOffset = bIdx * nLen_ * mLen_ + mIdx * mBlockSize_ * nLen_ + nIdx * nBlockSize_;
    uint64_t smoothOffset = mIdx * mBlockSize_;
    uint32_t nCopyInElePerRow = (nIdx + 1) == nBlockNum_ ? nTailBlockSize_ : nBlockSize_;
    uint8_t nCopyInPadNum = (nIdx + 1) == nBlockNum_ ? nTailBlockSizePadding_ : 0;
    uint32_t mCopyEle = (mIdx + 1) == mBlockNum_ ? mTailBlockSize_ : mBlockSize_;

    LocalTensor<xDtype> inLocal = inQueue_.template AllocTensor<xDtype>();
    DataCopyExtParams copyParams = {static_cast<uint16_t>(mCopyEle),
        static_cast<uint32_t>(nCopyInElePerRow * sizeof(xDtype)),
        static_cast<uint32_t>((nLen_ - nCopyInElePerRow) * sizeof(xDtype)),
        0,
        0};
    DataCopyPadExtParams<xDtype> padParams{true, 0, nCopyInPadNum, 0};
    DataCopyPad(inLocal, inGm[xOffset], copyParams, padParams);
    inQueue_.template EnQue(inLocal);
    if constexpr (hasSmooth) {
        LocalTensor<xDtype> smoothLocal = smoothQueue_.template AllocTensor<xDtype>();
        DataCopyExtParams smoothCopyParams = {1, static_cast<uint32_t>(mCopyEle * sizeof(xDtype)), 0, 0, 0};
        DataCopyPadExtParams<xDtype> smoothPadParams{true, 0, 0, 0};
        DataCopyPad(smoothLocal, smoothGm[smoothOffset], smoothCopyParams, smoothPadParams);
        smoothQueue_.template EnQue(smoothLocal);
    }
    if constexpr (copyInMax) {
        uint64_t colMaxOffset = (bIdx - batchStart_) * nLen_ + nIdx * nBlockSize_;
        LocalTensor<float> colMaxLocal = colMaxQueueIn_.template AllocTensor<float>();
        DataCopyExtParams colMaxCopyParams = {1, static_cast<uint32_t>(nCopyInElePerRow * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> colMaxPadParams{false, 0, 0, 0};
        DataCopyPad(colMaxLocal, colMaxGm[colMaxOffset], colMaxCopyParams, colMaxPadParams);
        colMaxQueueIn_.template EnQue(colMaxLocal);
        if constexpr (!isSymmetrical) {
            LocalTensor<float> colMinLocal = colMinQueueIn_.template AllocTensor<float>();
            DataCopyPad(colMinLocal, colMinGm[colMaxOffset], colMaxCopyParams, colMaxPadParams);
            colMinQueueIn_.template EnQue(colMinLocal);
        }
    }
}

template <typename xDtype, typename yDtype, bool hasSmooth, bool isSymmetrical>
__aicore__ inline void DynamicQuantRegbasePerChannnelSplitM<xDtype, yDtype, hasSmooth, isSymmetrical>::ComputeMinMax(
    __ubuf__ xDtype *inAddr, __ubuf__ xDtype *smoothAddr, __ubuf__ float *colMaxLocalAddr,
    __ubuf__ float *colMinLocalAddr, uint32_t nSize, uint16_t nLoopNum, uint16_t mLoopNum)
{
    uint32_t vl = DynamicQuantNDOpt::SIXTY_FOUR;
    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<xDtype> vregIn;
        MicroAPI::RegTensor<float> vregInFp32;
        MicroAPI::RegTensor<xDtype> vregSmooth;
        MicroAPI::RegTensor<float> vregSmoothFp32;
        MicroAPI::RegTensor<float> vregAbs;
        MicroAPI::RegTensor<float> vregCurrentMax;
        MicroAPI::RegTensor<float> vregColMax;
        MicroAPI::RegTensor<float> vregColMin;
        MicroAPI::MaskReg preg0;

        uint32_t sregN = nSize;
        if constexpr (isSymmetrical) {
            for (uint16_t i = 0; i < nLoopNum; i++) {
                preg0 = MicroAPI::UpdateMask<float>(sregN);
                MicroAPI::Duplicate<float>(vregColMax, 0.0f, preg0);
                for (uint16_t j = 0; j < mLoopNum; j++) {
                    MicroAPI::DataCopy<xDtype, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                        vregIn, (__ubuf__ xDtype *)(inAddr + i * vl + j * nSize));
                    MicroAPI::Cast<float, xDtype, castTraitB16ToB32>(vregInFp32, vregIn, preg0);
                    if constexpr (hasSmooth) {
                        MicroAPI::DataCopy<xDtype, MicroAPI::LoadDist::DIST_BRC_B16>(
                            vregSmooth, (__ubuf__ xDtype *)(smoothAddr + j));
                        MicroAPI::Cast<float, xDtype, castTraitB16ToB32>(vregSmoothFp32, vregSmooth, preg0);
                        MicroAPI::Mul<float>(vregInFp32, vregInFp32, vregSmoothFp32, preg0);
                    }
                    MicroAPI::Abs<float>(vregAbs, vregInFp32, preg0);
                    MicroAPI::Max<float>(vregColMax, vregAbs, vregColMax, preg0);
                }
                MicroAPI::DataCopy<float>((__ubuf__ float *)colMaxLocalAddr + i * vl, vregColMax, preg0);
            }
        } else {
            for (uint16_t i = 0; i < nLoopNum; i++) {
                preg0 = MicroAPI::UpdateMask<float>(sregN);
                MicroAPI::Duplicate<float>(vregColMax, NEG_INFINITY, preg0);
                MicroAPI::Duplicate<float>(vregColMin, POS_INFINITY, preg0);
                for (uint16_t j = 0; j < mLoopNum; j++) {
                    MicroAPI::DataCopy<xDtype, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                        vregIn, (__ubuf__ xDtype *)(inAddr + i * vl + j * nSize));
                    MicroAPI::Cast<float, xDtype, castTraitB16ToB32>(vregInFp32, vregIn, preg0);
                    if constexpr (hasSmooth) {
                        MicroAPI::DataCopy<xDtype, MicroAPI::LoadDist::DIST_BRC_B16>(
                            vregSmooth, (__ubuf__ xDtype *)smoothAddr + j);
                        MicroAPI::Cast<float, xDtype, castTraitB16ToB32>(vregSmoothFp32, vregSmooth, preg0);
                        MicroAPI::Mul<float>(vregInFp32, vregInFp32, vregSmoothFp32, preg0);
                    }
                    MicroAPI::Max<float>(vregColMax, vregInFp32, vregColMax, preg0);
                    MicroAPI::Min<float>(vregColMin, vregInFp32, vregColMin, preg0);
                }
                MicroAPI::DataCopy<float>((__ubuf__ float *)colMaxLocalAddr + i * vl, vregColMax, preg0);
                MicroAPI::DataCopy<float>((__ubuf__ float *)colMinLocalAddr + i * vl, vregColMin, preg0);
            }
        }
    }
}

template <typename xDtype, typename yDtype, bool hasSmooth, bool isSymmetrical>
__aicore__ inline void DynamicQuantRegbasePerChannnelSplitM<xDtype, yDtype, hasSmooth, isSymmetrical>::CopyOutMinMax(
    uint32_t bIdx, uint32_t nIdx, uint32_t count)
{
    LocalTensor<float> colMaxLocal = colMaxQueue_.template DeQue<float>();
    LocalTensor<float> colMinLocal;
    if constexpr (!isSymmetrical) {
        colMinLocal = colMinQueue_.template DeQue<float>();
    }
    uint64_t colMaxOffset = (bIdx - batchStart_) * nLen_ + nIdx * nBlockSize_;
    SetAtomicMax<float>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0};
    DataCopyPad(colMaxGm[colMaxOffset], colMaxLocal, copyParams);
    if constexpr (!isSymmetrical) {
        SetAtomicMin<float>();
        DataCopyPad(colMinGm[colMaxOffset], colMinLocal, copyParams);
        colMinQueue_.template FreeTensor(colMinLocal);
    }
    SetAtomicNone();
    colMaxQueue_.template FreeTensor(colMaxLocal);
}

template <typename xDtype, typename yDtype, bool hasSmooth, bool isSymmetrical>
__aicore__ inline void DynamicQuantRegbasePerChannnelSplitM<xDtype, yDtype, hasSmooth, isSymmetrical>::Compute(
    uint32_t bIdx, uint32_t mIdx, uint32_t nIdx)
{
    LocalTensor<xDtype> inLocal = inQueue_.template DeQue<xDtype>();
    LocalTensor<yCopyDtype> yLocal = outQueue_.template AllocTensor<yCopyDtype>();
    LocalTensor<float> scaleLocal = scaleQueue_.template AllocTensor<float>();
    LocalTensor<float> colMaxLocal = colMaxQueueIn_.template DeQue<float>();
    LocalTensor<float> colMinLocal;
    LocalTensor<float> offsetLocal;
    LocalTensor<xDtype> smoothLocal;

    __ubuf__ xDtype *inAddr = (__ubuf__ xDtype *)inLocal.GetPhyAddr();
    __ubuf__ xDtype *smoothAddr;

    __ubuf__ float *colMaxLocalAddr = (__ubuf__ float *)colMaxLocal.GetPhyAddr();

    __ubuf__ float *colMinLocalAddr;
    __ubuf__ float *offsetLocalAddr;

    __ubuf__ yCopyDtype *yAddr = (__ubuf__ yCopyDtype *)yLocal.GetPhyAddr();
    __ubuf__ float *scaleLocalAddr = (__ubuf__ float *)scaleLocal.GetPhyAddr();

    if constexpr (hasSmooth) {
        smoothLocal = smoothQueue_.template DeQue<xDtype>();
        smoothAddr = (__ubuf__ xDtype *)smoothLocal.GetPhyAddr();
    }
    if constexpr (!isSymmetrical) {
        colMinLocal = colMinQueueIn_.template DeQue<float>();
        colMinLocalAddr = (__ubuf__ float *)colMinLocal.GetPhyAddr();
        offsetLocal = offsetQueue_.template AllocTensor<float>();
        offsetLocalAddr = (__ubuf__ float *)offsetLocal.GetPhyAddr();
    }

    uint32_t vl = DynamicQuantNDOpt::SIXTY_FOUR;
    // nLoopNum is calculated by float aligned num, there's 64 float elements in one register at most
    uint32_t nSize = nIdx + 1 == nBlockNum_ ? nTailBlockSizeAligned_ : nBlockSize_;
    uint32_t nSizeOut = nIdx + 1 == nBlockNum_ ? nTailBlockSizeAlignedB8_ : nBlockSize_;

    uint16_t nLoopNum = ops::CeilDiv(nSize, vl);
    uint16_t mLoopNum = mIdx + 1 == mBlockNum_ ? mTailBlockSize_ : mBlockSize_;

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<xDtype> vregIn;
        MicroAPI::RegTensor<float> vregInFp32;
        MicroAPI::RegTensor<xDtype> vregSmooth;
        MicroAPI::RegTensor<float> vregSmoothFp32;
        MicroAPI::RegTensor<float> vregAbs;
        MicroAPI::RegTensor<float> vregCurrentMax;
        MicroAPI::RegTensor<float> vregColMax;
        MicroAPI::RegTensor<float> vregColMin;

        MicroAPI::RegTensor<float> vregMaxFactor;
        MicroAPI::RegTensor<float> vregOffsetFactor;
        MicroAPI::RegTensor<float> vregScale;

        MicroAPI::RegTensor<float> vregSub;
        MicroAPI::RegTensor<float> vregDiv;
        MicroAPI::RegTensor<float> vregMaxDivScale;
        MicroAPI::RegTensor<float> vregOffsetVal;
        MicroAPI::RegTensor<float> vregOffsetDivVal;
        MicroAPI::RegTensor<float> vregOffset;

        MicroAPI::RegTensor<yCopyDtype> vregOut;
        MicroAPI::RegTensor<float> vregOutFp32;
        MicroAPI::MaskReg preg0;
        MicroAPI::MaskReg pregAll = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg pregH = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::H>();
        MicroAPI::Duplicate<float>(vregMaxFactor, maxValue_, pregAll);

        uint32_t sregN = nSize;
        if constexpr (isSymmetrical) {
            for (uint16_t i = 0; i < nLoopNum; i++) {
                preg0 = MicroAPI::UpdateMask<float>(sregN);
                MicroAPI::DataCopy<float>(vregColMax, (__ubuf__ float *)(colMaxLocalAddr + i * vl));
                MicroAPI::Mul<float>(vregScale, vregColMax, vregMaxFactor, preg0);
                for (uint16_t j = 0; j < mLoopNum; j++) {
                    auto outAddr = yAddr + i * vl + j * nSizeOut;
                    MicroAPI::DataCopy<xDtype, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                        vregIn, (__ubuf__ xDtype *)(inAddr + i * vl + j * nSize));
                    MicroAPI::Cast<float, xDtype, castTraitB16ToB32>(vregInFp32, vregIn, preg0);
                    if constexpr (hasSmooth) {
                        MicroAPI::DataCopy<xDtype, MicroAPI::LoadDist::DIST_BRC_B16>(
                            vregSmooth, (__ubuf__ xDtype *)(smoothAddr + j));
                        MicroAPI::Cast<float, xDtype, castTraitB16ToB32>(vregSmoothFp32, vregSmooth, preg0);
                        MicroAPI::Mul<float>(vregInFp32, vregInFp32, vregSmoothFp32, preg0);
                    }
                    MicroAPI::Div<float>(vregOutFp32, vregInFp32, vregScale, preg0);
                    // cast implement
                    CastToDstType<yDtype, yCopyDtype>(vregOutFp32, vregOut, preg0);
                    if constexpr (IsSameType<yDtype, int4b_t>::value) {
                        outAddr = yAddr + (i * vl + j * nSizeOut) / 2;
                        MicroAPI::DataCopy<yCopyDtype, MicroAPI::StoreDist::DIST_PACK4_B32>(outAddr, vregOut, pregH);
                    } else {
                        MicroAPI::DataCopy<yCopyDtype, MicroAPI::StoreDist::DIST_PACK4_B32>(outAddr, vregOut, preg0);
                    }
                }
                MicroAPI::DataCopy<float>((__ubuf__ float *)scaleLocalAddr + i * vl, vregScale, preg0);
            }
        } else {
            MicroAPI::Duplicate<float>(vregOffsetVal, offsetValue_, pregAll);
            MicroAPI::Duplicate<float>(vregOffsetDivVal, offsetDivValue_, pregAll);
            for (uint16_t i = 0; i < nLoopNum; i++) {
                preg0 = MicroAPI::UpdateMask<float>(sregN);
                MicroAPI::DataCopy<float>(vregColMax, (__ubuf__ float *)(colMaxLocalAddr + i * vl));
                MicroAPI::DataCopy<float>(vregColMin, (__ubuf__ float *)(colMinLocalAddr + i * vl));
                MicroAPI::Sub<float>(vregSub, vregColMax, vregColMin, preg0);
                MicroAPI::Mul<float>(vregScale, vregSub, vregOffsetDivVal, preg0);                
                MicroAPI::Div<float, &divHighPrecisionMode>(vregMaxDivScale, vregColMax, vregScale, preg0);
                MicroAPI::Sub<float>(vregOffset, vregOffsetVal, vregMaxDivScale, preg0);
                for (uint16_t j = 0; j < mLoopNum; j++) {
                    auto outAddr = yAddr + i * vl + j * nSizeOut;
                    MicroAPI::DataCopy<xDtype, MicroAPI::LoadDist::DIST_UNPACK_B16>(
                        vregIn, (__ubuf__ xDtype *)inAddr + i * vl + j * nSize);
                    MicroAPI::Cast<float, xDtype, castTraitB16ToB32>(vregInFp32, vregIn, preg0);
                    if constexpr (hasSmooth) {
                        MicroAPI::DataCopy<xDtype, MicroAPI::LoadDist::DIST_BRC_B16>(
                            vregSmooth, (__ubuf__ xDtype *)(smoothAddr + j));
                        MicroAPI::Cast<float, xDtype, castTraitB16ToB32>(vregSmoothFp32, vregSmooth, preg0);
                        MicroAPI::Mul<float>(vregInFp32, vregInFp32, vregSmoothFp32, preg0);
                    }
                    MicroAPI::Div<float>(vregDiv, vregInFp32, vregScale, preg0);
                    MicroAPI::Add<float>(vregOutFp32, vregDiv, vregOffset, preg0);
                    // cast implement
                    CastToDstType<yDtype, yCopyDtype>(vregOutFp32, vregOut, preg0);
                    if constexpr (IsSameType<yDtype, int4b_t>::value) {
                        outAddr = yAddr + (i * vl + j * nSizeOut) / 2;
                        MicroAPI::DataCopy<yCopyDtype, MicroAPI::StoreDist::DIST_PACK4_B32>(outAddr, vregOut, pregH);
                    } else {
                        MicroAPI::DataCopy<yCopyDtype, MicroAPI::StoreDist::DIST_PACK4_B32>(outAddr, vregOut, preg0);
                    }
                }
                MicroAPI::DataCopy<float>((__ubuf__ float *)scaleLocalAddr + i * vl, vregScale, preg0);
                MicroAPI::DataCopy<float>((__ubuf__ float *)offsetLocalAddr + i * vl, vregOffset, preg0);
            }
        }
    }
    inQueue_.template FreeTensor(inLocal);
    colMaxQueueIn_.template FreeTensor(colMaxLocal);
    if constexpr (hasSmooth) {
        smoothQueue_.template FreeTensor(smoothLocal);
    }
    outQueue_.template EnQue(yLocal);
    scaleQueue_.template EnQue(scaleLocal);
    if constexpr (!isSymmetrical) {
        colMinQueueIn_.template FreeTensor(colMinLocal);
        offsetQueue_.template EnQue(offsetLocal);
    }
}

template <typename xDtype, typename yDtype, bool hasSmooth, bool isSymmetrical>
__aicore__ inline void DynamicQuantRegbasePerChannnelSplitM<xDtype, yDtype, hasSmooth, isSymmetrical>::CopyOut(
    uint32_t bIdx, uint32_t mIdx, uint32_t nIdx)
{
    uint64_t yOffset = bIdx * nLen_ * mLen_ + mIdx * mBlockSize_ * nLen_ + nIdx * nBlockSize_;
    uint64_t scaleOffset = bIdx * nLen_ + nIdx * nBlockSize_;
    uint32_t nCopyInElePerRow = (nIdx + 1) == nBlockNum_ ? nTailBlockSize_ : nBlockSize_;
    uint32_t mCopyEle = (mIdx + 1) == mBlockNum_ ? mTailBlockSize_ : mBlockSize_;

    LocalTensor<yCopyDtype> yLocal = outQueue_.template DeQue<yCopyDtype>();

    LocalTensor<float> scaleLocal = scaleQueue_.template DeQue<float>();
    LocalTensor<float> offsetLocal;

    DataCopyExtParams scaleCopyParams{1, static_cast<uint32_t>(nCopyInElePerRow * sizeof(float)), 0, 0, 0};
    DataCopyPad(scaleGm[scaleOffset], scaleLocal, scaleCopyParams);
    if constexpr (!isSymmetrical) {
        offsetLocal = offsetQueue_.template DeQue<float>();
        DataCopyPad(offsetGm[scaleOffset], offsetLocal, scaleCopyParams);
        offsetQueue_.template FreeTensor(offsetLocal);
    }

    DataCopyExtParams copyParams{static_cast<uint16_t>(mCopyEle),
        static_cast<uint32_t>(nCopyInElePerRow * sizeof(yCopyDtype)),
        0,
        static_cast<uint32_t>((nLen_ - nCopyInElePerRow) * sizeof(yCopyDtype)) ,
        0};
    if constexpr (IsSameType<yDtype, int4b_t>::value) {
        copyParams.blockLen = copyParams.blockLen >> 1;
        copyParams.dstStride = copyParams.dstStride >> 1;
        DataCopyPad(outGm[yOffset / 2], yLocal, copyParams);
    } else {
        DataCopyPad(outGm[yOffset], yLocal, copyParams);
    }
    outQueue_.template FreeTensor(yLocal);
    scaleQueue_.template FreeTensor(scaleLocal);
}

}  // namespace DynamicQuantPerChannel
#endif  // DYNAMIC_QUANT_REGBASE_PERCHANNEL_SPLIT_M_H