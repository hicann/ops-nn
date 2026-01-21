/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file softmax_grad_ar_recompute.h
 * \brief
 */

#ifndef SOFTMAX_GRAD_AR_RECOMPUTE_H
#define SOFTMAX_GRAD_AR_RECOMPUTE_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "../inc/platform.h"
#include "op_kernel/math_util.h"
#include "softmax_grad_base.h"

namespace SoftmaxGradOps
{
using namespace AscendC;

static constexpr int64_t AR_RECOMPUTE_SUM_BUFFER_BTYES = 32;
static constexpr int64_t AR_RECOMPUTE_BINARY_TMP_BTYES = 512;
static constexpr int64_t AR_RECOMPUTE_BINARY_CACHE_BTYES = 2048;
static constexpr int64_t AR_RECOMPUTE_SUM_LEN = AR_RECOMPUTE_SUM_BUFFER_BTYES / sizeof(float);
static constexpr float CONST_FP32_ZERO = 0.0;
static constexpr int64_t A_IN_IN = 1;

template <typename T>
class SoftmaxGradArRecompute : public SoftmaxGradOpsBase
{
public:
    __aicore__ inline SoftmaxGradArRecompute(TPipe* pipe)
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

    __aicore__ inline void LastReduceSum(const LocalTensor<float>& dstTensor, const LocalTensor<float>& srcTensor,
                                         const LocalTensor<float>& reduceSumTempTensor, const int64_t aSize,
                                         const int64_t rSize, const int64_t stride);

    __aicore__ inline void LastReduceSumSmallR(const LocalTensor<float>& dstTensor, const LocalTensor<float>& srcTensor,
                                               const int64_t aSize, const int64_t rSize, const int64_t stride);

    __aicore__ inline void LoadTensorForDtypeT(__local_mem__ T* src, RegTensor<float>& dst, MaskReg& pregMask,
                                               uint32_t offset);
    __aicore__ inline void StoreTensorForDtypeTOut(__local_mem__ T* dst, AscendC::MicroAPI::RegTensor<float>& src,
                                                   AscendC::MicroAPI::MaskReg& preg, uint32_t offset);

    __aicore__ inline void CopyInX(int64_t xGmOffset, uint32_t ubFactor);
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
    TBuf<> reduceSumTempTensor_;
    TBuf<> cachebuffer_;

    uint32_t blockIdx_ = GetBlockIdx();
    uint64_t currentRowBlock_ = 0;
    uint32_t resultCacheID_ = 0;

    LocalTensor<float> xSumTensor_;
};

template <typename T>
__aicore__ inline void SoftmaxGradArRecompute<T>::Init(GM_ADDR x0, GM_ADDR x1, GM_ADDR y,
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
    pipe_->InitBuffer(reduceSumTempTensor_, AR_RECOMPUTE_BINARY_TMP_BTYES);
    pipe_->InitBuffer(cachebuffer_, AR_RECOMPUTE_BINARY_CACHE_BTYES);
}

template <typename T>
__aicore__ inline void SoftmaxGradArRecompute<T>::Process()
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

            CopyInX(xUbOffset, ubFactor);
            CalcOutVF(ubFactor);
            CopyOutY(xUbOffset, ubFactor);
        }
    }
}

template <typename T>
__aicore__ inline void SoftmaxGradArRecompute<T>::CalcReduceSum(int64_t xDimOffset)
{
    LocalTensor<float> reduceSumTempLocal = reduceSumTempTensor_.Get<float>();
    LocalTensor<float> cacheLocal = cachebuffer_.Get<float>();
    LocalTensor<float> xSum = xSumBuffer_.Get<float>();

    LocalTensor<float> xTmp = yQueue_.AllocTensor<float>();  // 复用y做二分累加
    __local_mem__ float* xTmpLocal = (__local_mem__ float*)xTmp.GetPhyAddr();

    // ub间累加fold折叠到main
    for (uint64_t basicBlockIdx = 0; basicBlockIdx < tl_->basicBlockLoop; basicBlockIdx++) {
        int64_t xMainOffset = xDimOffset + tl_->ubFactor * basicBlockIdx;
        int64_t xFoldOffset = xDimOffset + tl_->ubFactor * (tl_->basicBlockLoop + basicBlockIdx);

        CopyInX(xMainOffset, tl_->ubFactor);
        MainBlockVF(xTmpLocal, tl_->ubFactor);

        if (basicBlockIdx < tl_->mainFoldCount) {
            CopyInX(xFoldOffset, tl_->ubFactor);
            FoldBlockVF(xTmpLocal, tl_->ubFactor);
        } else if ((basicBlockIdx == tl_->mainFoldCount) && (tl_->ubFactorTail > 0)) {
            CopyInX(xFoldOffset, tl_->ubFactorTail);
            FoldBlockVF(xTmpLocal, tl_->ubFactorTail);
        }

        AscendC::Duplicate(xSum, CONST_FP32_ZERO, AR_RECOMPUTE_SUM_BUFFER_BTYES / sizeof(float));
        // 计算UB内二分累加
        int64_t cacheId = GetCacheID(basicBlockIdx);
        LastReduceSum(xSum, xTmp, reduceSumTempLocal, A_IN_IN, tl_->ubFactor, tl_->r);
        UpdateCache(cacheLocal, xSum, cacheId, A_IN_IN * AR_RECOMPUTE_SUM_LEN, A_IN_IN);
    }

    // R很小，不需要做UB间二分累加
    if (tl_->basicBlockLoop == 0) {
        CopyInX(xDimOffset, tl_->ubFactor);
        MainBlockVF(xTmpLocal, tl_->ubFactor);
        AscendC::Duplicate(xSum, CONST_FP32_ZERO, AR_RECOMPUTE_SUM_BUFFER_BTYES / sizeof(float));
        LastReduceSum(xSum, xTmp, reduceSumTempLocal, A_IN_IN, tl_->ubFactor, tl_->r);
    }

    yQueue_.FreeTensor(xTmp);

    xSumTensor_ = tl_->basicBlockLoop > 0 ? cacheLocal[resultCacheID_ * AR_RECOMPUTE_SUM_LEN] : xSum;
}

// cast + mul
template <typename T>
__aicore__ inline void SoftmaxGradArRecompute<T>::MainBlockVF(__local_mem__ float* dst, uint32_t ubFactor)
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
__aicore__ inline void SoftmaxGradArRecompute<T>::FoldBlockVF(__local_mem__ float* dst, uint32_t ubFactor)
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
__aicore__ inline void SoftmaxGradArRecompute<T>::CalcOutVF(uint32_t ubFactor)
{
    LocalTensor<T> x0 = x0Queue_.DeQue<T>();
    LocalTensor<T> x1 = x1Queue_.DeQue<T>();
    LocalTensor<T> y = yQueue_.AllocTensor<T>();

    __local_mem__ float* xSumLocal = (__local_mem__ float*)xSumTensor_.GetPhyAddr();
    __local_mem__ T* x0Local = (__local_mem__ T*)x0.GetPhyAddr();
    __local_mem__ T* x1Local = (__local_mem__ T*)x1.GetPhyAddr();
    __local_mem__ T* yLocal = (__local_mem__ T*)y.GetPhyAddr();

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<float> sumReg, x0Reg, x1Reg;
        AscendC::MicroAPI::MaskReg pregMask;

        uint32_t sreg = ubFactor;
        uint16_t loopTimes = CeilDivision(ubFactor, VL_FP32);

        AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(sumReg, xSumLocal);

        for (uint16_t j = 0; j < loopTimes; j++) {
            pregMask = AscendC::MicroAPI::UpdateMask<float>(sreg);
            uint32_t offset = j * VL_FP32;
            LoadTensorForDtypeT(x0Local, x0Reg, pregMask, offset);
            LoadTensorForDtypeT(x1Local, x1Reg, pregMask, offset);

            Mul(x1Reg, x0Reg, x1Reg, pregMask);
            Mul(x0Reg, x0Reg, sumReg, pregMask);
            Sub(x0Reg, x1Reg, x0Reg, pregMask);
            StoreTensorForDtypeTOut(yLocal, x0Reg, pregMask, offset);
        }
    }

    yQueue_.EnQue(y);

    x0Queue_.FreeTensor(x0);
    x1Queue_.FreeTensor(x1);
}

template <typename T>
__aicore__ inline void SoftmaxGradArRecompute<T>::LoadTensorForDtypeT(__local_mem__ T* src, RegTensor<float>& dst,
                                                                      MaskReg& pregMask, uint32_t offset)
{
    if constexpr (IsSameType<T, float>::value) {
        DataCopy<float, LoadDist::DIST_NORM>(dst, (__local_mem__ float*)src + offset);
    } else {  // fp16、bf16
        RegTensor<T> xFp16;
        DataCopy<T, LoadDist::DIST_UNPACK_B16>(xFp16, ((__local_mem__ T*)src + offset));
        Cast<float, T, castTraitFp16ToFp32>(dst, xFp16, pregMask);
    }
}

template <typename T>
__aicore__ inline void SoftmaxGradArRecompute<T>::StoreTensorForDtypeTOut(__local_mem__ T* dst,
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
__aicore__ inline void SoftmaxGradArRecompute<T>::CopyOutY(int64_t yGmOffset, int64_t ubFactor)
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
__aicore__ inline void SoftmaxGradArRecompute<T>::CopyInX(int64_t xGmOffset, uint32_t ubFactor)
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
__aicore__ inline void SoftmaxGradArRecompute<T>::LastReduceSum(const LocalTensor<float>& dstTensor,
                                                                const LocalTensor<float>& srcTensor,
                                                                const LocalTensor<float>& reduceSumTempTensor,
                                                                const int64_t aSize, const int64_t rSize,
                                                                const int64_t stride)
{
    if (aSize <= 0 || rSize <= 0) {
        return;
    }

    if (rSize <= CONST_TWO * VL_FP32) {
        LastReduceSumSmallR(dstTensor, srcTensor, aSize, rSize, stride);
        return;
    }

    int64_t ceilVLCount =
        Ops::Base::CeilDiv(static_cast<int64_t>(rSize * sizeof(float)), static_cast<int64_t>(platform::GetVRegSize()));
    int64_t floorVLCount =
        ops::FloorDiv(static_cast<int64_t>(rSize * sizeof(float)), static_cast<int64_t>(platform::GetVRegSize()));
    int64_t foldPoint = FindNearestPower2(ceilVLCount);

    uint16_t outerLoopTimes = aSize;
    uint16_t tailFoldLoopTimes = static_cast<uint16_t>(ceilVLCount - floorVLCount);
    uint32_t tailFoldElemCount = static_cast<uint32_t>(rSize - floorVLCount * VL_FP32);
    uint16_t mainFoldLoopTimes = static_cast<uint16_t>(floorVLCount - foldPoint);
    uint16_t unFoldLoopTimes = static_cast<uint16_t>(foldPoint + foldPoint - ceilVLCount);
    uint32_t outerLoopStride = stride;
    uint32_t innerLoopStride = VL_FP32;
    uint32_t outerLoopDstStride =
        ops::Aligned(static_cast<int64_t>(foldPoint), static_cast<int64_t>(platform::GetUbBlockSize() / sizeof(float)));

    int64_t foldSrcBOffset = foldPoint * VL_FP32;
    int64_t tailSrcAOffset = mainFoldLoopTimes * VL_FP32;
    int64_t tailSrcBOffset = floorVLCount * VL_FP32;
    int64_t unFoldSrcOffset = (mainFoldLoopTimes + tailFoldLoopTimes) * VL_FP32;

    __VEC_SCOPE__
    {
        __local_mem__ float* dst = (__local_mem__ float*)reduceSumTempTensor.GetPhyAddr();
        __local_mem__ float* foldSrcA = (__local_mem__ float*)srcTensor.GetPhyAddr();
        __local_mem__ float* foldSrcB = (__local_mem__ float*)srcTensor.GetPhyAddr() + foldSrcBOffset;
        __local_mem__ float* tailSrcA = (__local_mem__ float*)srcTensor.GetPhyAddr() + tailSrcAOffset;
        __local_mem__ float* tailSrcB = (__local_mem__ float*)srcTensor.GetPhyAddr() + tailSrcBOffset;
        __local_mem__ float* unFoldSrc = (__local_mem__ float*)srcTensor.GetPhyAddr() + unFoldSrcOffset;
        AscendC::MicroAPI::MaskReg pFull = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::UnalignReg UReg;

        for (uint16_t i = 0; i < outerLoopTimes; ++i) {
            dst = (__local_mem__ float*)reduceSumTempTensor.GetPhyAddr() + i * outerLoopDstStride;
            for (uint16_t j = 0; j < mainFoldLoopTimes; ++j) {
                AscendC::MicroAPI::RegTensor<float> aReg, bReg, cReg, dReg;
                DataCopy(aReg, (__local_mem__ float*)foldSrcA + i * outerLoopStride + j * innerLoopStride);
                DataCopy(bReg, (__local_mem__ float*)foldSrcB + i * outerLoopStride + j * innerLoopStride);
                Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(cReg, aReg, bReg, pFull);
                ReduceSum(dReg, cReg, pFull);
                AscendC::MicroAPI::DataCopyUnAlign((__local_mem__ float*&)dst, dReg, UReg, 1);
            }
            for (uint16_t j = 0; j < tailFoldLoopTimes; ++j) {
                uint32_t count = static_cast<uint32_t>(tailFoldElemCount);
                AscendC::MicroAPI::RegTensor<float> aReg, bReg, cReg;
                AscendC::MicroAPI::MaskReg pMask = AscendC::MicroAPI::UpdateMask<float>(count);
                DataCopy(aReg, (__local_mem__ float*)tailSrcA + i * outerLoopStride + j * innerLoopStride);
                DataCopy(bReg, (__local_mem__ float*)tailSrcB + i * outerLoopStride + j * innerLoopStride);
                Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(cReg, aReg, bReg, pMask);
                Copy<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(aReg, cReg, pMask);
                ReduceSum(bReg, aReg, pFull);
                AscendC::MicroAPI::DataCopyUnAlign((__local_mem__ float*&)dst, bReg, UReg, 1);
            }
            for (uint16_t j = 0; j < unFoldLoopTimes; ++j) {
                AscendC::MicroAPI::RegTensor<float> aReg, bReg;
                DataCopy(aReg, (__local_mem__ float*)unFoldSrc + i * outerLoopStride + j * innerLoopStride);
                ReduceSum(bReg, aReg, pFull);
                AscendC::MicroAPI::DataCopyUnAlign((__local_mem__ float*&)dst, bReg, UReg, 1);
            }
            AscendC::MicroAPI::DataCopyUnAlignPost((__local_mem__ float*&)dst, UReg, 0);
        }
    }
    LastReduceSumSmallR(dstTensor, reduceSumTempTensor, aSize, foldPoint, outerLoopDstStride);
}

template <typename T>
__aicore__ inline void SoftmaxGradArRecompute<T>::LastReduceSumSmallR(const LocalTensor<float>& dstTensor,
                                                                      const LocalTensor<float>& srcTensor,
                                                                      const int64_t aSize, const int64_t rSize,
                                                                      const int64_t stride)
{
    if (aSize <= 0) {
        return;
    }
    if (rSize <= 0) {
        return;
    }
    if (rSize > CONST_TWO * VL_FP32) {
        return;
    }

    uint16_t loopTimes = aSize;
    if (rSize <= VL_FP32) {
        __local_mem__ float* dst = (__local_mem__ float*)dstTensor.GetPhyAddr();
        __local_mem__ float* src = (__local_mem__ float*)srcTensor.GetPhyAddr();

        __VEC_SCOPE__
        {
            uint32_t count = static_cast<uint32_t>(rSize);
            uint32_t constOne = 1;
            AscendC::MicroAPI::RegTensor<float> aReg, bReg, sumReg;

            AscendC::MicroAPI::MaskReg pMask = AscendC::MicroAPI::UpdateMask<float>(count);
            AscendC::MicroAPI::MaskReg maskOne = AscendC::MicroAPI::UpdateMask<float>(constOne);
            AscendC::MicroAPI::UnalignReg UReg;
            for (uint16_t i = 0; i < loopTimes; ++i) {
                AscendC::MicroAPI::DataCopy(aReg, (__local_mem__ float*)src + i * static_cast<uint32_t>(stride));
                AscendC::MicroAPI::ReduceSum(bReg, aReg, pMask);
                AscendC::MicroAPI::DataCopy(sumReg, dst);
                AscendC::MicroAPI::Add(bReg, bReg, sumReg, maskOne);
                AscendC::MicroAPI::DataCopyUnAlign((__local_mem__ float*&)dst, bReg, UReg, 1);
            }
            AscendC::MicroAPI::DataCopyUnAlignPost((__local_mem__ float*&)dst, UReg, 0);
        }
    } else {
        __local_mem__ float* dst = (__local_mem__ float*)dstTensor.GetPhyAddr();
        __local_mem__ float* src0 = (__local_mem__ float*)srcTensor.GetPhyAddr();
        __local_mem__ float* src1 = (__local_mem__ float*)srcTensor.GetPhyAddr() + VL_FP32;

        __VEC_SCOPE__
        {
            uint32_t count = static_cast<uint32_t>(rSize - VL_FP32);
            uint32_t constOne = 1;
            AscendC::MicroAPI::RegTensor<float> aReg, bReg, cReg, sumReg;

            AscendC::MicroAPI::UnalignReg UReg;
            AscendC::MicroAPI::MaskReg pMask = AscendC::MicroAPI::UpdateMask<float>(count);
            AscendC::MicroAPI::MaskReg maskOne = AscendC::MicroAPI::UpdateMask<float>(constOne);
            AscendC::MicroAPI::MaskReg pFull =
                AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            for (uint16_t i = 0; i < loopTimes; ++i) {
                AscendC::MicroAPI::DataCopy(aReg, (__local_mem__ float*)src0 + i * static_cast<uint32_t>(stride));
                AscendC::MicroAPI::DataCopy(bReg, (__local_mem__ float*)src1 + i * static_cast<uint32_t>(stride));
                AscendC::MicroAPI::Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(cReg, aReg, bReg, pMask);
                AscendC::MicroAPI::Copy<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(aReg, cReg, pMask);
                AscendC::MicroAPI::ReduceSum(bReg, aReg, pFull);
                AscendC::MicroAPI::DataCopy(sumReg, dst);
                AscendC::MicroAPI::Add(bReg, bReg, sumReg, maskOne);
                AscendC::MicroAPI::DataCopyUnAlign((__local_mem__ float*&)dst, bReg, UReg, 1);
            }
            AscendC::MicroAPI::DataCopyUnAlignPost((__local_mem__ float*&)dst, UReg, 0);
        }
    }
}

}  // namespace SoftmaxGradOps
#endif  // SOFTMAX_GRAD_AR_RECOMPUTE_H