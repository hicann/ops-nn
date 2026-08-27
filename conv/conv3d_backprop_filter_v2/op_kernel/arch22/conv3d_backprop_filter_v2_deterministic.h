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
 * \file conv3d_backprop_filter_v2_deterministic.h
 * \brief
 */

#ifndef CONV3D_BACKPROP_FILTER_V2_DETERMINISTIC_H
#define CONV3D_BACKPROP_FILTER_V2_DETERMINISTIC_H

#if defined(DETERMINISTIC_MODE) && DETERMINISTIC_MODE == 1
__aicore__ inline void InitMixCoreBuffer()
{
    dw_.ctx.pipe_.InitBuffer(tmpBuf_, UB_SIZE); // ub space for reduce calculation
}

__aicore__ inline void CalMaxIterate()
{
    uint32_t maxSingleShapeGroup = group_ < singleCoreGroup_ ? group_ : singleCoreGroup_;
    uint32_t maxSingleShapeDk = dk_ < singleCoreDk_ ? dk_ : singleCoreDk_;
    uint64_t maxSingleShapeBatch = static_cast<uint64_t>(batch_) * dout_ < singleCoreBatch_ ?
                                       static_cast<uint64_t>(batch_) * dout_ :
                                       singleCoreBatch_;
    uint64_t maxSingleShapeM = m_ < singleCoreCout_ ? m_ : singleCoreCout_;
    uint64_t maxSingleShapeN = n_ < singleCoreCin_ * hk_ * wk_ ? n_ : singleCoreCin_ * hk_ * wk_;
    uint64_t maxMIter = Ceil(maxSingleShapeM, dw_.ctx.tiling_->baseM);
    uint64_t maxNIter = Ceil(
        Ceil(maxSingleShapeN / (hk_ * wk_), dw_.ctx.tiling_->channelSize) * dw_.ctx.tiling_->channelSize * (hk_ * wk_),
        dw_.ctx.tiling_->baseN);
    maxIterate_ = maxSingleShapeGroup * maxSingleShapeDk * maxSingleShapeBatch * maxMIter * maxNIter;
}

__aicore__ inline void ReachMaxIterate()
{
    uint16_t remainClearTimes = DOUBLE_BUFFER;
    while (syncTimes_ < maxIterate_) {
        if ASCEND_IS_AIC {
            if (remainClearTimes > 0) {
                ClearL0C();
            }
            if (syncTimes_ > 1) {
                WaitVector(gmPingPongEventId_);
            }
            if (remainClearTimes > 0) {
                LoadL0CToWorkspace(workspaceGm_[offsetWorkspaceC_ + gmPingPongEventId_ * singleSize_]);
                remainClearTimes--;
            }
            NotifyVector(gmPingPongEventId_);
            syncTimes_++;
        }
        if ASCEND_IS_AIV {
            WaitCube(gmPingPongEventId_);
            NotifyCube(gmPingPongEventId_);
            syncTimes_++;
        }
        gmPingPongEventId_ &= 1;
        gmPingPongEventId_ ^= 1;
    }
}

__aicore__ inline void ReduceKInUb()
{
    LocalTensor<yType> ubSrc1 = tmpBuf_.template Get<yType>();
    LocalTensor<yType> ubSrc2 = ubSrc1[dataSize_];
    LocalTensor<yType> ubDst = ubSrc1;
    uint64_t alignCoutG = static_cast<uint64_t>(dw_.ctx.tiling_->cout1G) * dw_.ctx.tiling_->channelSize;
    uint64_t dstOffset = static_cast<uint64_t>(dw_.ctx.curNL0Idx_) * dw_.ctx.tiling_->baseN * alignCoutG +
                         static_cast<uint64_t>(dw_.ctx.curML0Idx_) * dw_.ctx.tiling_->baseM *
                             dw_.ctx.tiling_->channelSize;
    uint64_t totalN1 = DivCeil(static_cast<uint64_t>(dw_.ctx.baseUseN_), dw_.ctx.tiling_->channelSize);
    uint64_t dataBlockSize = static_cast<uint64_t>(dw_.ctx.tiling_->baseM) * dw_.ctx.tiling_->channelSize;
    uint64_t n1InOneCal = dataBlockSize == 0 ? 0 : dataSize_ / dataBlockSize;
    uint64_t repeat = n1InOneCal == 0 ? 0 : DivCeil(totalN1, n1InOneCal);
    uint64_t tailN1 = totalN1 % n1InOneCal == 0 ? n1InOneCal : totalN1 % n1InOneCal;
    uint64_t useDataSize = n1InOneCal * dataBlockSize;
    for (uint64_t i = 0; i < repeat; i++) {
        // offset of reading cache worskpace
        uint64_t baseWorkspaceOffset = gmPingPongEventId_ * singleSize_ + i * useDataSize;
        // offset of final writing address
        uint64_t totalOffset = offsetC_ + dstOffset + i * n1InOneCal * dw_.ctx.tiling_->channelSize * alignCoutG;
        // handling tail data
        if (i == repeat - 1) {
            useDataSize = tailN1 * dataBlockSize;
            n1InOneCal = tailN1;
        }
        SetFlag<HardEvent::MTE3_V>(0);
        WaitFlag<HardEvent::MTE3_V>(0);
        uint32_t groupCoreFactor = dkDim_ * batchDim_ * kDim_ * nDim_ * mDim_;
        uint32_t dkCoreFactor = batchDim_ * kDim_ * nDim_ * mDim_;
        uint32_t batchCoreFactor = mDim_ * nDim_ * kDim_;
        uint32_t mCoreFactor = nDim_ * kDim_;
        for (uint32_t curBatchIndx = 0; curBatchIndx < batchDim_; curBatchIndx++) {
            for (uint32_t curKIndx = 0; curKIndx < kDim_; curKIndx++) {
                uint32_t curBlkIndx = groupCoreIndx_ * groupCoreFactor + dkCoreIndx_ * dkCoreFactor +
                                      curBatchIndx * batchCoreFactor + mCoreIndx_ * mCoreFactor + nCoreIndx_ * kDim_ +
                                      curKIndx;
                uint64_t curWorkspaceOffset = baseWorkspaceOffset + DOUBLE_BUFFER * curBlkIndx * singleSize_;
                SetFlag<HardEvent::V_MTE2>(0);
                WaitFlag<HardEvent::V_MTE2>(0);
                if (curBatchIndx == 0 && curKIndx == 0) {
                    DataCopy(ubSrc1, workspaceGm_[curWorkspaceOffset], useDataSize);
                    continue;
                }
                DataCopy(ubSrc2, workspaceGm_[curWorkspaceOffset], useDataSize);
                SetFlag<HardEvent::MTE2_V>(0);
                WaitFlag<HardEvent::MTE2_V>(0);
                PipeBarrier<PIPE_V>();
                Add<yType>(ubDst, ubSrc1, ubSrc2, useDataSize);
            }
        }
        DataCopyParams loadUbToGmParams(n1InOneCal, DivCeil(dataBlockSize * sizeof(float), ONE_BLK_SIZE), 0, 0);
        uint64_t yGmOffsetInterval = alignCoutG * dw_.ctx.tiling_->channelSize;
        uint64_t dstStride = DivCeil((yGmOffsetInterval - dataBlockSize) * sizeof(float), ONE_BLK_SIZE);
        LoadUBToGm(yGm_[totalOffset], ubDst, loadUbToGmParams, dstStride, dataBlockSize, yGmOffsetInterval);
    }
}

__aicore__ inline void DeterministicIterateAll()
{
    bool isCompute = (singleShapeNInCurrentHo_ != 0 && singleShapeMInCurrentHo_ != 0);
    for (uint64_t k = 0; k < dw_.ctx.mIter_ * dw_.ctx.nIter_; k++) {
        if ASCEND_IS_AIC {
            if (syncTimes_ > 1) {
                WaitVector(gmPingPongEventId_);
            }
            ClearL0C();
            LoadL0CToWorkspace(workspaceGm_[offsetWorkspaceC_ + gmPingPongEventId_ * singleSize_]);
            if (isCompute) {
                dw_.Iterate();
                // 0: disable atomic add; true: enable sequential write
                dw_.GetTensorC(workspaceGm_[offsetWorkspaceC_ + gmPingPongEventId_ * singleSize_], 0, true);
            }
            NotifyVector(gmPingPongEventId_);
            syncTimes_++;
        }
        if ASCEND_IS_AIV {
            dw_.Iterate();
            WaitCube(gmPingPongEventId_);
            // Only vector cores with kCoreIndx_==0 and batchCoreIndx_==0 are used
            if (kCoreIndx_ == 0 && batchCoreIndx_ == 0) {
                ReduceKInUb();
            }
            NotifyCube(gmPingPongEventId_);
            syncTimes_++;
        }
        gmPingPongEventId_ &= 1;
        gmPingPongEventId_ ^= 1;
    }
    dw_.ctx.isFirstIter_ = true;
}

__aicore__ inline void ClearL0C()
{
    if ASCEND_IS_AIC {
        LocalTensor<xType> l0a;
        LocalTensor<xType> l0b;
        LocalTensor<float> l0c;
        l0a = dw_.ctx.l0aBuf_.template Get<xType>();
        l0b = dw_.ctx.l0bBuf_.template Get<xType>();
        l0c = dw_.ctx.l0cPing_.template AllocTensor<float>();
        PipeBarrier<PIPE_MTE1>();
        InitConstValue(l0a, {1, static_cast<uint16_t>(DivCeil(TOTAL_L0A_SIZE, 512)), 0,
                             static_cast<xType>(0)}); // 512: datablock size on L0A
        InitConstValue(l0b, {1, static_cast<uint16_t>(DivCeil(TOTAL_L0B_SIZE, 512)), 0,
                             static_cast<xType>(0)}); // 512: datablock size on L0B
        MmadParams mmadParams;
        mmadParams.m = dw_.ctx.tiling_->baseM;
        mmadParams.n = dw_.ctx.tiling_->baseN;
        mmadParams.k = dw_.ctx.tiling_->baseK;
        mmadParams.cmatrixInitVal = true;
        TEventID eventId = GetTPipePtr()->FetchEventID<HardEvent::MTE1_M>();
        SetFlag<HardEvent::MTE1_M>(eventId);
        WaitFlag<HardEvent::MTE1_M>(eventId);
        MmadImpl(l0c, l0a, l0b, mmadParams);
        // MMAD计算量baseM*baseN小于一定阈值时需要添加PIPE_M同步,当前平台阈值为10*256
        if (mmadParams.m * mmadParams.n < 2560) {
            PipeBarrier<PIPE_M>();
        }
        eventId = GetTPipePtr()->FetchEventID<HardEvent::M_MTE1>();
        SetFlag<HardEvent::M_MTE1>(eventId);
        WaitFlag<HardEvent::M_MTE1>(eventId);
        dw_.ctx.l0cPing_.EnQue(l0c);
    }
}

__aicore__ inline void LoadL0CToWorkspace(const GlobalTensor<float>& output)
{
    if ASCEND_IS_AIC {
        LocalTensor<float> l0c;
        l0c = dw_.ctx.l0cPing_.template DeQue<float>();
        uint64_t dstStrideIn = dw_.ctx.tiling_->baseM * dw_.ctx.tiling_->channelSize * sizeof(float) / ONE_BLK_SIZE;
        FixpipeParamsV220 fixpipeParams(
            static_cast<uint16_t>(dw_.ctx.tiling_->baseN), static_cast<uint16_t>(dw_.ctx.tiling_->baseM),
            ShiftCeilM0(dw_.ctx.tiling_->baseM, dw_.ctx.tiling_->m0) * dw_.ctx.tiling_->m0, dstStrideIn, 0);
        if constexpr (IsSameType<xType, float>::value) {
            fixpipeParams.isChannelSplit = true;
        }
        Fixpipe<float, float, CFG_NZ>(output, l0c, fixpipeParams);
        dw_.ctx.l0cPing_.FreeTensor(l0c);
    }
}

__aicore__ inline void LoadUBToGm(const GlobalTensor<float>& output, const LocalTensor<float>& src,
                                  DataCopyParams& loadUbToGmParams, const uint64_t dstStride,
                                  const uint64_t dataBlockSize, const uint64_t yGmOffsetInterval)
{
    SetAtomicAdd<float>();
    SetFlag<HardEvent::V_MTE3>(0);
    WaitFlag<HardEvent::V_MTE3>(0);
    SetFlag<HardEvent::MTE2_MTE3>(0);
    WaitFlag<HardEvent::MTE2_MTE3>(0);
    if (dstStride <= ConvolutionBackpropFunc::MAX_16BITS_STRIDE) {
        loadUbToGmParams.dstStride = dstStride;
        DataCopy(output, src, loadUbToGmParams);
    } else {
        uint16_t blockCount = loadUbToGmParams.blockCount;
        loadUbToGmParams.blockCount = 1;
        uint64_t yGmOffset = 0;
        uint64_t ubDstOffset = 0;
        for (uint16_t blockIndex = 0; blockIndex < blockCount; blockIndex++) {
            DataCopy(output[yGmOffset], src[ubDstOffset], loadUbToGmParams);
            yGmOffset += yGmOffsetInterval;
            ubDstOffset += dataBlockSize;
        }
    }
    SetAtomicNone();
}

__aicore__ inline void NotifyCube(bool gmPingPongEventId)
{
    CrossCoreSetFlag<SYNC_MODE0, PIPE_MTE2>(SYNC_AIV_ONLY_ALL_FLAG + gmPingPongEventId);
    CrossCoreWaitFlag(SYNC_AIV_ONLY_ALL_FLAG + gmPingPongEventId);
    CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE3>(SYNC_AIV_AIC_FLAG + gmPingPongEventId);
}

__aicore__ inline void WaitCube(bool gmPingPongEventId) { CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG + gmPingPongEventId); }

__aicore__ inline void NotifyVector(bool gmPingPongEventId)
{
    CrossCoreSetFlag<SYNC_MODE0, PIPE_FIX>(SYNC_AIC_ONLY_ALL_FLAG + gmPingPongEventId);
    CrossCoreWaitFlag(SYNC_AIC_ONLY_ALL_FLAG + gmPingPongEventId);
    CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_AIC_AIV_FLAG + gmPingPongEventId);
}

__aicore__ inline void WaitVector(bool gmPingPongEventId) { CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG + gmPingPongEventId); }
#endif // DETERMINISTIC_MODE

#endif
