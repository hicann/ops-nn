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
 * \file conv3d_backprop_input_v2_aiv.h
 * \brief
 */

#ifndef CONV3D_BACKPROP_INPUT_V2_AIV_H
#define CONV3D_BACKPROP_INPUT_V2_AIV_H

#if __CCE_AICORE__ == 220
__aicore__ inline void InitMixCoreBuffer(GM_ADDR workSpace)
{
    l0cOutGm_.SetGlobalBuffer((__gm__ yType*)workSpace);
    if ASCEND_IS_AIV {
        uint32_t halfUbSize = TOTAL_UB_SIZE / HALF_FACTOR;
        dedx_.ctx.pipe_.InitBuffer(vecInQueue_, 1, halfUbSize);
        dedx_.ctx.pipe_.InitBuffer(vecOutQueue_, 1, halfUbSize);
    }
}

__aicore__ inline void CopyInToUB()
{
    if ASCEND_IS_AIC {
        return;
    }

    LocalTensor<yType> vecInBuf_ = vecInQueue_.template AllocTensor<yType>();
    int64_t srcOffset = block_idx * dedx_.ctx.tiling_->baseM * dedx_.ctx.tiling_->baseN;
    DataCopyParams loadGm2UbParams;
    loadGm2UbParams.srcStride = 0;
    loadGm2UbParams.dstStride = (dedx_.ctx.tiling_->baseM - dedx_.ctx.baseUseM_) * dedx_.ctx.tiling_->c0 *
                                sizeof(yType) / ONE_BLK_SIZE;
    loadGm2UbParams.blockLen = static_cast<uint16_t>(dedx_.ctx.baseUseM_ * dedx_.ctx.tiling_->c0 * sizeof(yType) /
                                                     ONE_BLK_SIZE);
    loadGm2UbParams.blockCount = static_cast<uint16_t>(Ceil(dedx_.ctx.baseUseN_, BLOCK_CUBE));
    DataCopy(vecInBuf_, l0cOutGm_[srcOffset], loadGm2UbParams);
    vecInQueue_.EnQue(vecInBuf_);
}

__aicore__ inline void TransFormat()
{
    if ASCEND_IS_AIC {
        return;
    }

    LocalTensor<yType> vecInBuf_ = vecInQueue_.template DeQue<yType>();
    LocalTensor<yType> vecOutBuf_ = vecOutQueue_.template AllocTensor<yType>();
    TransDataTo5HDParams transDataParams;
    transDataParams.dstHighHalf = false;
    transDataParams.srcHighHalf = false;
    transDataParams.repeatTimes = static_cast<uint16_t>(Ceil(dedx_.ctx.baseUseM_, NCHW_CONV_ADDR_LIST_SIZE));
    transDataParams.dstRepStride = 1;
    transDataParams.srcRepStride = NCHW_CONV_ADDR_LIST_SIZE;
    // 参考AscendC API的使用说明，当repeatTimes为1时，repStride需要设置为0
    if (transDataParams.repeatTimes == 1) {
        transDataParams.dstRepStride = 0;
        transDataParams.srcRepStride = 0;
    }
    uint64_t dstLocalList[NCHW_CONV_ADDR_LIST_SIZE];
    uint64_t srcLocalList[NCHW_CONV_ADDR_LIST_SIZE];
    int64_t baseCount = 0;
    int64_t dstCount = 0;
    int64_t srcCount = 0;
    int loopCount = (dedx_.ctx.baseUseN_ >> dedx_.ctx.tiling_->c0Bits);
    uint64_t baseCountIncrement = (dedx_.ctx.tiling_->baseM << dedx_.ctx.tiling_->c0Bits);
    for (int j = 0; j < loopCount; j++) {
        dstCount = baseCount;
        for (int i = 0; i < NCHW_CONV_ADDR_LIST_SIZE; i++) {
            dstLocalList[i] = reinterpret_cast<uint64_t>(vecOutBuf_[dstCount].GetPhyAddr());
            dstCount += dedx_.ctx.tiling_->baseM;
        }
        srcCount = baseCount;
        for (int i = 0; i < NCHW_CONV_ADDR_LIST_SIZE; i++) {
            srcLocalList[i] = reinterpret_cast<uint64_t>(vecInBuf_[srcCount].GetPhyAddr());
            srcCount += dedx_.ctx.tiling_->c0;
        }
        TransDataTo5HD<half>(dstLocalList, srcLocalList, transDataParams);
        baseCount += baseCountIncrement;
    }
    vecOutQueue_.EnQue(vecOutBuf_);
    vecInQueue_.FreeTensor(vecInBuf_);
}

__aicore__ inline void CopyOutToGm(const GlobalTensor<yType>& output)
{
    if ASCEND_IS_AIC {
        return;
    }

    LocalTensor<yType> vecOutBuf_ = vecOutQueue_.template DeQue<yType>();
    uint64_t diHiWi = static_cast<uint64_t>(dedx_.ctx.tiling_->di) * dedx_.ctx.tiling_->hi * dedx_.ctx.tiling_->wi;
    uint64_t dstStride = (diHiWi - dedx_.ctx.baseUseM_) * sizeof(yType);
    uint64_t dstOffset = static_cast<uint64_t>(dedx_.ctx.curNL0Idx_) * dedx_.ctx.tiling_->baseN * diHiWi +
                         static_cast<uint64_t>(dedx_.ctx.curML0Idx_) * dedx_.ctx.tiling_->baseM +
                         static_cast<uint64_t>(dedx_.ctx.curDinIdx_) * dedx_.ctx.tiling_->hi * dedx_.ctx.tiling_->wi;
    uint32_t curCinSize = dedx_.ctx.baseUseN_ <
                                  (dedx_.ctx.singleShapeCin_ - dedx_.ctx.curNL0Idx_ * dedx_.ctx.tiling_->baseN) ?
                              dedx_.ctx.baseUseN_ :
                              (dedx_.ctx.singleShapeCin_ - dedx_.ctx.curNL0Idx_ * dedx_.ctx.tiling_->baseN);
    DataCopyExtParams storeUb2GmParams;
    // 用&f 代替对16取余
    if (((dedx_.ctx.baseUseM_ & 0xf) == 0) && dstStride <= UINT32_MAX) {
        storeUb2GmParams.srcStride = (dedx_.ctx.tiling_->baseM - dedx_.ctx.baseUseM_) * sizeof(yType) / ONE_BLK_SIZE;
        storeUb2GmParams.dstStride = dstStride;
        storeUb2GmParams.blockLen = dedx_.ctx.baseUseM_ * sizeof(yType);
        storeUb2GmParams.blockCount = curCinSize;
        DataCopyPad(output[dstOffset], vecOutBuf_, storeUb2GmParams);
    } else {
        uint32_t ubOffset = 0;
        storeUb2GmParams.srcStride = 0;
        storeUb2GmParams.dstStride = 0;
        storeUb2GmParams.blockLen = dedx_.ctx.baseUseM_ * sizeof(yType);
        storeUb2GmParams.blockCount = 1;
        for (uint32_t n = 0; n < curCinSize; n++) {
            DataCopyPad(output[dstOffset], vecOutBuf_[ubOffset], storeUb2GmParams);
            dstOffset += diHiWi;
            ubOffset += dedx_.ctx.tiling_->baseM;
        }
    }
    vecOutQueue_.FreeTensor(vecOutBuf_);
}

__aicore__ inline bool JudgeComputeNecessary()
{
    for (uint64_t curKdIdx = 0; curKdIdx < dedx_.ctx.tiling_->dk; curKdIdx++) {
        int64_t dTmp = 0;
        if (unlikely(dedx_.ctx.tiling_->strideD > dedx_.ctx.tiling_->dk)) {
            dTmp = dedx_.ctx.curDinIdx_ + dedx_.ctx.tiling_->padFront;
            if (CalcRemainder(dTmp, dedx_.ctx.tiling_->strideD) >= dedx_.ctx.tiling_->dk ||
                dTmp / dedx_.ctx.tiling_->strideD >= dedx_.ctx.tiling_->dout) {
                continue;
            }
        } else {
            dTmp = dedx_.ctx.curDinIdx_ + dedx_.ctx.tiling_->padFront - curKdIdx * dedx_.ctx.tiling_->dilationD;
            if (dTmp < 0 || CalcRemainder(dTmp, dedx_.ctx.tiling_->strideD) > 0 ||
                dTmp >= dedx_.ctx.tiling_->dout * dedx_.ctx.tiling_->strideD) {
                continue;
            }
        }
        return true;
    }
    return false;
}

__aicore__ inline void VecPostProcess(const GlobalTensor<yType>& output, uint8_t enAtomic = 0,
                                      bool enSequentialWrite = false)
{
    if ASCEND_IS_AIC {
        return;
    }

    if (!dedx_.ctx.needComputeFlag_ || !JudgeComputeNecessary()) {
        CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE2>(SYNC_AIV_AIC_FLAG);
        return;
    }

    if (GetSubBlockIdx() > 0) {
        CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE2>(SYNC_AIV_AIC_FLAG);
        return;
    }
    if (unlikely(enAtomic == 1)) {
        SetAtomicAdd<yType>();
    }
    if constexpr (yFormat == FORMAT_NCDHW) {
        if (!enSequentialWrite) {
            CopyInToUB();
            CrossCoreSetFlag<SYNC_MODE2, PIPE_MTE2>(SYNC_AIV_AIC_FLAG);
            TransFormat();
            CopyOutToGm(output);
        }
    }
    if (unlikely(enAtomic == 1)) {
        SetAtomicNone();
    }
}

__aicore__ inline void MergeOutputTransDataIterateAll(const GlobalTensor<yType>& output, uint8_t enAtomic = 0)
{
    while (dedx_.Iterate()) {
        if ASCEND_IS_AIC {
            if (likely(!isFirstIter_)) {
                CrossCoreWaitFlag(SYNC_AIV_AIC_FLAG);
            }
            dedx_.GetTensorC(l0cOutGm_[0]);
            CrossCoreSetFlag<SYNC_MODE2, PIPE_FIX>(SYNC_AIC_AIV_FLAG);
        }

        if ASCEND_IS_AIV {
            CrossCoreWaitFlag(SYNC_AIC_AIV_FLAG);
            VecPostProcess(output, enAtomic);
        }
        isFirstIter_ = false;
    }
    dedx_.ctx.isFirstIter_ = true;
}
#endif

#endif
