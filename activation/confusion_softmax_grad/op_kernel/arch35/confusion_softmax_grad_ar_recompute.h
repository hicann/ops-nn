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
 * \file confusion_softmax_grad_ar_recompute.h
 * \brief
 */

#ifndef NORM_CONFUSION_SOFTMAX_GRAD_AR_RECOMPUTE_H
#define NORM_CONFUSION_SOFTMAX_GRAD_AR_RECOMPUTE_H

#include "confusion_softmax_grad_base.h"

namespace ConfusionSoftmaxGradOps
{
using namespace AscendC;

constexpr int64_t AR_RECOMPUTE_SUM_BUFFER_BTYES = 32;
constexpr int64_t AR_RECOMPUTE_BINARY_CACHE_BTYES = 2048;
constexpr int64_t AR_RECOMPUTE_SUM_LEN = AR_RECOMPUTE_SUM_BUFFER_BTYES / sizeof(float);
constexpr static float CONST_FP32_MIN = -(__builtin_inff());
constexpr int64_t A_IN_IN = 1;

template <typename T>
class ConfusionSoftmaxGradArRecompute : public ConfusionSoftmaxGradOpsBase
{
public:
    __aicore__ inline ConfusionSoftmaxGradArRecompute(TPipe* pipe)
    {
        pipe_ = pipe;
    };
    __aicore__ inline void Init(GM_ADDR x0, GM_ADDR x1, GM_ADDR y, const SoftmaxGradARRecomputeTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CalcReduceSum(int64_t xDimOffset);
    __aicore__ inline void CalcOutVF(uint32_t ubFactor);

    __aicore__ inline void MainBlockVF(__local_mem__ float* dst, uint32_t ubFactor);
    __aicore__ inline void FoldBlockVF(__local_mem__ float* dst, uint32_t ubFactor);
    __aicore__ inline void LoadTensorForDtypeT(__local_mem__ T* src, AscendC::MicroAPI::RegTensor<float>& dst, AscendC::MicroAPI::MaskReg& pregMask,
                                               uint32_t offset);

    __aicore__ inline void StoreTensorForDtypeTOut(__local_mem__ T* dst, AscendC::MicroAPI::RegTensor<float>& src,
                                                   AscendC::MicroAPI::MaskReg& preg, uint32_t offset);
    __aicore__ inline void CopyInX0(int64_t xGmOffset, uint32_t ubFactor);
    __aicore__ inline void CopyInX0X1(int64_t xGmOffset, uint32_t ubFactor);
    __aicore__ inline void CopyOutY(int64_t yGmOffset, int64_t ubFactor);

protected:
    GlobalTensor<T> x0Gm_;
    GlobalTensor<T> x1Gm_;
    GlobalTensor<T> yGm_;

    const SoftmaxGradARRecomputeTilingData* tl_ = nullptr;
    TPipe* pipe_ = nullptr;
    TQue<QuePosition::VECIN, 1> x0Queue_;
    TQue<QuePosition::VECIN, 1> x1Queue_;
    TQue<QuePosition::VECOUT, 1> yQueue_;

    TBuf<> xSumBuffer_;
    TBuf<> cachebuffer_;

    uint32_t blockIdx_ = GetBlockIdx();
    uint64_t currentRowBlock_ = 0;
    uint32_t resultCacheID_ = 0;

    LocalTensor<float> xSumTensor_;
};

template <typename T>
__aicore__ inline void ConfusionSoftmaxGradArRecompute<T>::Init(GM_ADDR x0, GM_ADDR x1, GM_ADDR y,
                                                       const SoftmaxGradARRecomputeTilingData* tilingData)
{
    tl_ = tilingData;

    int64_t rowBlockCount = ops::FloorDiv(tl_->a, tl_->aBlockFactor);
    int64_t tailBlockFactor = tl_->a - rowBlockCount * tl_->aBlockFactor;

    if (blockIdx_ < rowBlockCount) {
        currentRowBlock_ = tl_->aBlockFactor;
    } else {
        currentRowBlock_ = tailBlockFactor;
    }

    if (tl_->basicBlockLoop == 0) {
        resultCacheID_ = 0;
    } else {
        resultCacheID_ = GetCacheID(tl_->basicBlockLoop - 1);
    }

    x0Gm_.SetGlobalBuffer((__gm__ T*)x0);
    x1Gm_.SetGlobalBuffer((__gm__ T*)x1);
    yGm_.SetGlobalBuffer((__gm__ T*)y);

    pipe_->InitBuffer(x0Queue_, DOUBLE_BUFFER, tl_->ubFactor * sizeof(T));
    pipe_->InitBuffer(x1Queue_, DOUBLE_BUFFER, tl_->ubFactor * sizeof(T));
    pipe_->InitBuffer(yQueue_, DOUBLE_BUFFER, tl_->ubFactor * sizeof(float));

    pipe_->InitBuffer(xSumBuffer_, AR_RECOMPUTE_SUM_BUFFER_BTYES);
    pipe_->InitBuffer(cachebuffer_, AR_RECOMPUTE_BINARY_CACHE_BTYES);
}

template <typename T>
__aicore__ inline void ConfusionSoftmaxGradArRecompute<T>::Process()
{
    int64_t xDimOffsetPerCore = tl_->aBlockFactor * blockIdx_;  // 每个核按行的偏移

    for (uint64_t rowIdx = 0; rowIdx < currentRowBlock_; rowIdx++) {
        int64_t xDimOffset = (xDimOffsetPerCore + rowIdx) * tl_->r;  // 每行的偏移量

        CalcReduceSum(xDimOffset);

        for (uint64_t ubIdx = 0; ubIdx < tl_->aLoopCountCeil; ubIdx++) {
            int64_t xUbOffset = xDimOffset + tl_->ubFactor * ubIdx;
            int64_t ubFactor = tl_->ubFactor;
            if (ubIdx == tl_->aLoopCountCeil - 1 && tl_->ubFactorTail > 0) {
                ubFactor = tl_->ubFactorTail;
            }

            CopyInX0(xUbOffset, ubFactor);
            CalcOutVF(ubFactor);
            CopyOutY(xUbOffset, ubFactor);
        }
    }
}

template <typename T>
__aicore__ inline void ConfusionSoftmaxGradArRecompute<T>::CalcReduceSum(int64_t xDimOffset)
{
    LocalTensor<float> cacheLocal = cachebuffer_.Get<float>();
    LocalTensor<float> xSum = xSumBuffer_.Get<float>();

    LocalTensor<float> xTmp = yQueue_.AllocTensor<float>();
    __local_mem__ float* xTmpLocal = (__local_mem__ float*)xTmp.GetPhyAddr();

    // ub间累加fold折叠到main
    for (uint64_t basicBlockIdx = 0; basicBlockIdx < tl_->basicBlockLoop; basicBlockIdx++) {
        int64_t xMainOffset = xDimOffset + tl_->ubFactor * basicBlockIdx;
        int64_t xFoldOffset = xDimOffset + tl_->ubFactor * (tl_->basicBlockLoop + basicBlockIdx);

        CopyInX0X1(xMainOffset, tl_->ubFactor);
        MainBlockVF(xTmpLocal, tl_->ubFactor);

        if (basicBlockIdx < tl_->mainFoldCount) {
            CopyInX0X1(xFoldOffset, tl_->ubFactor);
            FoldBlockVF(xTmpLocal, tl_->ubFactor);
        } else if ((basicBlockIdx == tl_->mainFoldCount) && (tl_->ubFactorTail > 0)) {
            CopyInX0X1(xFoldOffset, tl_->ubFactorTail);
            FoldBlockVF(xTmpLocal, tl_->ubFactorTail);
        }

        // 计算UB内二分累加
        int64_t cacheId = GetCacheID(basicBlockIdx);
        uint32_t srcShape[2] = {uint32_t(A_IN_IN), uint32_t(tl_->ubFactor)};
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(xSum, xTmp, srcShape, false);
        UpdateCache(cacheLocal, xSum, cacheId, A_IN_IN * AR_RECOMPUTE_SUM_LEN, A_IN_IN);
    }

    // R很小，不需要做UB间二分累加
    if (tl_->basicBlockLoop == 0) {
        CopyInX0X1(xDimOffset, tl_->ubFactor);
        MainBlockVF(xTmpLocal, tl_->ubFactor);
        uint32_t srcShape[2] = {uint32_t(A_IN_IN), uint32_t(tl_->ubFactor)};
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(xSum, xTmp, srcShape, false);
    }

    yQueue_.FreeTensor(xTmp);

    xSumTensor_ = tl_->basicBlockLoop > 0 ? cacheLocal[resultCacheID_ * AR_RECOMPUTE_SUM_LEN] : xSum;
}

// cast + mul
template <typename T>
__aicore__ inline void ConfusionSoftmaxGradArRecompute<T>::MainBlockVF(__local_mem__ float* dst, uint32_t ubFactor)
{
    LocalTensor<T> x0 = x0Queue_.DeQue<T>();
    LocalTensor<T> x1 = x1Queue_.DeQue<T>();

    __local_mem__ T* x0Local = (__local_mem__ T*)x0.GetPhyAddr();
    __local_mem__ T* x1Local = (__local_mem__ T*)x1.GetPhyAddr();

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<float> reg0, reg1;
        AscendC::MicroAPI::MaskReg pregMask;

        uint32_t sreg = ubFactor;
        uint16_t loopNum = CeilDivision(ubFactor, VL_FP32);
        for (uint16_t j = 0; j < loopNum; j++) {
            pregMask = AscendC::MicroAPI::UpdateMask<float>(sreg);
            uint32_t offset = j * VL_FP32;
            LoadTensorForDtypeT(x0Local, reg0, pregMask, offset);
            LoadTensorForDtypeT(x1Local, reg1, pregMask, offset);

            Mul(reg0, reg0, reg1, pregMask);

            AscendC::MicroAPI::DataCopy(dst + offset, reg0, pregMask);
        }
    }

    x0Queue_.FreeTensor(x0);
    x1Queue_.FreeTensor(x1);
}

template <typename T>
__aicore__ inline void ConfusionSoftmaxGradArRecompute<T>::FoldBlockVF(__local_mem__ float* dst, uint32_t ubFactor)
{
    LocalTensor<T> x0 = x0Queue_.DeQue<T>();
    LocalTensor<T> x1 = x1Queue_.DeQue<T>();

    __local_mem__ T* x0Local = (__local_mem__ T*)x0.GetPhyAddr();
    __local_mem__ T* x1Local = (__local_mem__ T*)x1.GetPhyAddr();

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<float> reg0, reg1;
        AscendC::MicroAPI::MaskReg pregMask;
        AscendC::MicroAPI::MaskReg maskFull =
            AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();

        uint16_t loopTimes = CeilDivision(ubFactor, VL_FP32);
        uint32_t sreg = ubFactor;
        for (uint16_t j = 0; j < loopTimes; j++) {
            pregMask = AscendC::MicroAPI::UpdateMask<float>(sreg);
            uint32_t offset = j * VL_FP32;

            LoadTensorForDtypeT(x0Local, reg0, pregMask, offset);
            LoadTensorForDtypeT(x1Local, reg1, pregMask, offset);

            Mul(reg1, reg0, reg1, pregMask);

            AscendC::MicroAPI::DataCopy(reg0, dst + offset);

            AscendC::MicroAPI::Add(reg1, reg0, reg1, pregMask);
            AscendC::MicroAPI::Copy<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(reg0, reg1, pregMask);

            AscendC::MicroAPI::DataCopy(dst + offset, reg0, maskFull);
        }
    }

    x0Queue_.FreeTensor(x0);
    x1Queue_.FreeTensor(x1);
}

template <typename T>
__aicore__ inline void ConfusionSoftmaxGradArRecompute<T>::CalcOutVF(uint32_t ubFactor)
{
    LocalTensor<T> x0 = x0Queue_.DeQue<T>();
    LocalTensor<T> y = yQueue_.AllocTensor<T>();

    __local_mem__ float* xSumLocal = (__local_mem__ float*)xSumTensor_.GetPhyAddr();
    __local_mem__ T* x0Local = (__local_mem__ T*)x0.GetPhyAddr();
    __local_mem__ T* yLocal = (__local_mem__ T*)y.GetPhyAddr();

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<float> sumReg, x0Reg;
        AscendC::MicroAPI::MaskReg pregMask;

        uint32_t sreg = ubFactor;
        uint16_t loopTimes = CeilDivision(ubFactor, VL_FP32);

        AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(sumReg, xSumLocal);

        for (uint16_t j = 0; j < loopTimes; j++) {
            pregMask = AscendC::MicroAPI::UpdateMask<float>(sreg);
            uint32_t offset = j * VL_FP32;
            LoadTensorForDtypeT(x0Local, x0Reg, pregMask, offset);
            Sub(x0Reg, x0Reg, sumReg, pregMask);
            StoreTensorForDtypeTOut(yLocal, x0Reg, pregMask, offset);
        }
    }

    yQueue_.EnQue(y);
    x0Queue_.FreeTensor(x0);
}

template <typename T>
__aicore__ inline void ConfusionSoftmaxGradArRecompute<T>::LoadTensorForDtypeT(__local_mem__ T* src, AscendC::MicroAPI::RegTensor<float>& dst,
                                                                      AscendC::MicroAPI::MaskReg& pregMask, uint32_t offset)
{
    if constexpr (IsSameType<T, float>::value) {
        DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(dst, (__local_mem__ float*)src + offset);
    } else {  // fp16、bf16
        AscendC::MicroAPI::RegTensor<T> xFp16;
        DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(xFp16, ((__local_mem__ T*)src + offset));
        Cast<float, T, castTraitFp16ToFp32>(dst, xFp16, pregMask);
    }
}

template <typename T>
__aicore__ inline void ConfusionSoftmaxGradArRecompute<T>::StoreTensorForDtypeTOut(__local_mem__ T* dst,
                                                                          AscendC::MicroAPI::RegTensor<float>& src,
                                                                          AscendC::MicroAPI::MaskReg& preg,
                                                                          uint32_t offset)
{
    if constexpr (IsSameType<T, float>::value) {
        DataCopy<T, AscendC::MicroAPI::StoreDist::DIST_NORM>(dst + offset, src, preg);
    } else {
        AscendC::MicroAPI::RegTensor<T> xFp16;
        Cast<T, float, castTraitFp32ToFp16>(xFp16, src, preg);
        DataCopy<T, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(dst + offset, xFp16, preg);
    }
}

template <typename T>
__aicore__ inline void ConfusionSoftmaxGradArRecompute<T>::CopyOutY(int64_t yGmOffset, int64_t ubFactor)
{
    LocalTensor<T> y = yQueue_.DeQue<T>();
    DataCopyExtParams copyOutParams;
    copyOutParams.blockCount = 1;
    copyOutParams.blockLen = ubFactor * sizeof(T);
    copyOutParams.srcStride = 0;
    copyOutParams.dstStride = 0;
    DataCopyPad<T, PaddingMode::Normal>(yGm_[yGmOffset], y, copyOutParams);
    yQueue_.FreeTensor(y);
}

template <typename T>
__aicore__ inline void ConfusionSoftmaxGradArRecompute<T>::CopyInX0X1(int64_t xGmOffset, uint32_t ubFactor)
{
    DataCopyExtParams params;
    params.blockCount = 1;
    params.blockLen = ubFactor * sizeof(T);
    params.srcStride = 0;
    params.dstStride = 0;
    DataCopyPadExtParams<T> padParams;
    padParams.isPad = false;

    LocalTensor<T> x0 = x0Queue_.AllocTensor<T>();
    DataCopyPad(x0, x0Gm_[xGmOffset], params, padParams);
    x0Queue_.EnQue(x0);

    LocalTensor<T> x1 = x1Queue_.AllocTensor<T>();
    DataCopyPad(x1, x1Gm_[xGmOffset], params, padParams);
    x1Queue_.EnQue(x1);
}

template <typename T>
__aicore__ inline void ConfusionSoftmaxGradArRecompute<T>::CopyInX0(int64_t xGmOffset, uint32_t ubFactor)
{
    DataCopyExtParams params;
    params.blockCount = 1;
    params.blockLen = ubFactor * sizeof(T);
    params.srcStride = 0;
    params.dstStride = 0;
    DataCopyPadExtParams<T> padParams;
    padParams.isPad = false;

    LocalTensor<T> x0 = x0Queue_.AllocTensor<T>();
    DataCopyPad(x0, x0Gm_[xGmOffset], params, padParams);
    x0Queue_.EnQue(x0);
}

}  // namespace ConfusionSoftmaxGradOps

#endif  // NORM_CONFUSION_SOFTMAX_GRAD_AR_RECOMPUTE_H