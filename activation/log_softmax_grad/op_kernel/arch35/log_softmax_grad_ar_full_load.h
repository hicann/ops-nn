/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file log_softmax_grad_ar_full_load.h
 * \brief
 */

#ifndef LOG_SOFTMAX_GRAD_AR_FULL_LOAD_H
#define LOG_SOFTMAX_GRAD_AR_FULL_LOAD_H

#include "kernel_tiling/kernel_tiling.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#include "adv_api/reduce/reduce.h"
#else
#include "kernel_operator.h"
#endif
#include "../inc/platform.h"
#include "log_softmax_grad_base.h"

namespace LogSoftmaxGradOps
{
using namespace AscendC;
using namespace AscendC::MicroAPI;

using AscendC::MicroAPI::LoadDist;
using AscendC::MicroAPI::MaskMergeMode;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::StoreDist;

constexpr static uint32_t DOUBLE_BUFFER = 2;
constexpr static uint32_t BLOCK_SIZE = 32;  // 32B
constexpr static uint32_t AR_FULL_LOAD_BINARY_TMP_BYTES = 512;

template <typename T>
class LogSoftmaxGradAR : public LogSoftmaxGradOpsBase
{
public:
    __aicore__ inline LogSoftmaxGradAR(TPipe* pipe)
    {
        pipe_ = pipe;
    };

    __aicore__ inline void Init(GM_ADDR grad, GM_ADDR x, GM_ADDR y, const SoftmaxGradARTilingData* tilingData);

    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessUB(int64_t ubA, int64_t aOffset);

    __aicore__ inline void NormCompute(const LocalTensor<T>& dstTensor, const LocalTensor<T>& gradTensor,
                                       const LocalTensor<T>& xTensor, const LocalTensor<float>& reduceSumTempTensor,
                                       const int64_t aSize);

    __aicore__ inline void NormComputePost(const LocalTensor<T>& dstTensor, const LocalTensor<T>& gradTensor,
                                           const LocalTensor<T>& xTensor, const LocalTensor<float>& binAddTmpTensor,
                                           const int64_t aSize, const int64_t rSize, const int64_t stride);

    __aicore__ inline void NormComputePostWithMul(const LocalTensor<T>& dstTensor, const LocalTensor<T>& gradTensor,
                                                  const LocalTensor<T>& xTensor, const int64_t aSize);

    __aicore__ inline void CopyInX(const LocalTensor<T>& xInUb, const GlobalTensor<T>& xInGm, int64_t ubA,
                                   int64_t offset);

    __aicore__ inline void CopyOutY(const LocalTensor<T>& yOutUb, int64_t ubA, int64_t offset);

    __aicore__ inline void StoreTensorForDtypeTOut(__local_mem__ T* dst, RegTensor<float>& src, MaskReg& preg,
                                                   uint32_t offset);

    __aicore__ inline void LoadTensorForDtypeTIn(__local_mem__ T* src, RegTensor<float>& dst, MaskReg& preg,
                                                 uint32_t offset);

private:
    /* global memory address */
    GlobalTensor<T> gradGm_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;

    /* ascendc variable */
    TPipe* pipe_ = nullptr;
    TQue<QuePosition::VECIN, 1> gradQueue_;
    TQue<QuePosition::VECIN, 1> xQueue_;
    TQue<QuePosition::VECOUT, 1> yQueue_;
    TBuf<> binaryTmpLocalBuffer_;

    int64_t blockA_ = 0;  // 获取分块操作中的单个块的大小
    const SoftmaxGradARTilingData* tl_ = nullptr;
};

template <typename T>
__aicore__ inline void LogSoftmaxGradAR<T>::Init(GM_ADDR grad, GM_ADDR x, GM_ADDR y,
                                              const SoftmaxGradARTilingData* tilingData)
{
    this->tl_ = tilingData;

    // 获取分块操作中的单个块的大小。判断是否是最后一块，是最后一块，则等于剩余元素的数量，否则等于固定的单核处理的行数
    this->blockA_ = (AscendC::GetBlockIdx() == AscendC::GetBlockNum() - 1)
                        ? (tl_->a - tl_->aBlockFactor * (AscendC::GetBlockNum() - 1))
                        : tl_->aBlockFactor;

    // 初始化GM Tensor
    int64_t aGmOffset = tl_->aBlockFactor * AscendC::GetBlockIdx() * tl_->r;
    gradGm_.SetGlobalBuffer((__gm__ T*)grad + aGmOffset);
    xGm_.SetGlobalBuffer((__gm__ T*)x + aGmOffset);
    yGm_.SetGlobalBuffer((__gm__ T*)y + aGmOffset);

    // 初始化Pipe
    int64_t ubBufferSize = tl_->ubFactor * tl_->rAligned;
    pipe_->InitBuffer(this->gradQueue_, DOUBLE_BUFFER, ubBufferSize * sizeof(T));
    pipe_->InitBuffer(this->xQueue_, DOUBLE_BUFFER, ubBufferSize * sizeof(T));
    pipe_->InitBuffer(this->yQueue_, DOUBLE_BUFFER, ubBufferSize * sizeof(T));
    pipe_->InitBuffer(this->binaryTmpLocalBuffer_, AR_FULL_LOAD_BINARY_TMP_BYTES);
}

template <typename T>
__aicore__ inline void LogSoftmaxGradAR<T>::Process()
{
    // ubLoop: 表示需要多少个子块来覆盖singleA大小的数据
    int64_t ubLoop = ops::CeilDiv(this->blockA_, tl_->ubFactor);
    int64_t lastUbFactor = this->blockA_ - tl_->ubFactor * (ubLoop - 1);
    // 循环处理每个子块
    for (int64_t ubLoopIdx = 0; ubLoopIdx < ubLoop; ubLoopIdx++) {
        // aOffset：计算当前子块的偏移量
        int64_t aOffset = ubLoopIdx * tl_->ubFactor * tl_->r;
        int64_t ubA = (ubLoopIdx == (ubLoop - 1)) ? lastUbFactor : tl_->ubFactor;
        ProcessUB(ubA, aOffset);
    }
}

template <typename T>
__aicore__ inline void LogSoftmaxGradAR<T>::ProcessUB(int64_t ubA, int64_t aOffset)
{
    LocalTensor<T> gradInUb = gradQueue_.AllocTensor<T>();
    CopyInX(gradInUb, gradGm_, ubA, aOffset);
    gradQueue_.EnQue(gradInUb);

    LocalTensor<T> xInUb = xQueue_.AllocTensor<T>();
    CopyInX(xInUb, xGm_, ubA, aOffset);
    xQueue_.EnQue(xInUb);

    gradInUb = gradQueue_.DeQue<T>();
    xInUb = xQueue_.DeQue<T>();

    LocalTensor<float> binaryTmpLocalTensor = binaryTmpLocalBuffer_.AllocTensor<float>();

    LocalTensor<T> yInUb = yQueue_.AllocTensor<T>();
    NormCompute(yInUb, gradInUb, xInUb, binaryTmpLocalTensor, ubA);

    yQueue_.EnQue(yInUb);
    gradQueue_.FreeTensor<T>(gradInUb);
    xQueue_.FreeTensor<T>(xInUb);

    LocalTensor<T> yOutUb = yQueue_.DeQue<T>();
    CopyOutY(yOutUb, ubA, aOffset);
    yQueue_.FreeTensor(yOutUb);
}

template <typename T>
__aicore__ inline void LogSoftmaxGradAR<T>::NormCompute(const LocalTensor<T>& dstTensor, const LocalTensor<T>& gradTensor,
                                                     const LocalTensor<T>& xTensor,
                                                     const LocalTensor<float>& reduceSumTempTensor, const int64_t aSize)
{
    if (aSize <= 0) {
        return;
    }
    if (tl_->r <= 0) {
        return;
    }
    if (tl_->r <= CONST_TWO * VL_FP32) {
        NormComputePostWithMul(dstTensor, gradTensor, xTensor, aSize);
        return;
    }

    int64_t ceilVLCount =
        ops::CeilDiv(static_cast<int64_t>(tl_->r * sizeof(float)), static_cast<int64_t>(platform::GetVRegSize()));
    int64_t floorVLCount =
        ops::FloorDiv(static_cast<int64_t>(tl_->r * sizeof(float)), static_cast<int64_t>(platform::GetVRegSize()));
    int64_t foldPoint = FindNearestPower2(ceilVLCount);

    uint16_t outerLoopTimes = aSize;
    uint16_t tailFoldLoopTimes = ceilVLCount - floorVLCount;
    uint32_t tailFoldElemCount = static_cast<uint32_t>(tl_->r - floorVLCount * VL_FP32);
    uint16_t mainFoldLoopTimes = floorVLCount - foldPoint;
    uint16_t unFoldLoopTimes = foldPoint + foldPoint - ceilVLCount;
    uint32_t outerLoopStride = tl_->rAligned;
    uint32_t innerLoopStride = VL_FP32;
    uint32_t outerLoopDstStride =
        ops::Aligned(static_cast<int64_t>(foldPoint), static_cast<int64_t>(platform::GetUbBlockSize() / sizeof(float)));

    int64_t foldSrcBOffset = foldPoint * VL_FP32;
    int64_t tailSrcAOffset = mainFoldLoopTimes * VL_FP32;
    int64_t tailSrcBOffset = floorVLCount * VL_FP32;
    int64_t unFoldSrcOffset = (mainFoldLoopTimes + tailFoldLoopTimes) * VL_FP32;

    __local_mem__ float* dst = (__local_mem__ float*)reduceSumTempTensor.GetPhyAddr();
    __local_mem__ T* foldGradA = (__local_mem__ T*)gradTensor.GetPhyAddr();
    __local_mem__ T* foldGradB = (__local_mem__ T*)gradTensor.GetPhyAddr() + foldSrcBOffset;
    __local_mem__ T* tailGradA = (__local_mem__ T*)gradTensor.GetPhyAddr() + tailSrcAOffset;
    __local_mem__ T* tailGradB = (__local_mem__ T*)gradTensor.GetPhyAddr() + tailSrcBOffset;
    __local_mem__ T* unFoldGrad = (__local_mem__ T*)gradTensor.GetPhyAddr() + unFoldSrcOffset;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::MaskReg pFull = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::UnalignReg UReg;

        for (uint16_t i = 0; i < outerLoopTimes; ++i) {
            dst = (__local_mem__ float*)reduceSumTempTensor.GetPhyAddr() + i * outerLoopDstStride;
            for (uint16_t j = 0; j < mainFoldLoopTimes; ++j) {
                AscendC::MicroAPI::RegTensor<float> reg0, reg1, reg2;
                LoadTensorForDtypeTIn(foldGradA, reg0, pFull, i * outerLoopStride + j * innerLoopStride);
                LoadTensorForDtypeTIn(foldGradB, reg1, pFull, i * outerLoopStride + j * innerLoopStride);

                Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(reg0, reg0, reg1, pFull);
                ReduceSum(reg2, reg0, pFull);
                AscendC::MicroAPI::DataCopyUnAlign((__local_mem__ float*&)dst, reg2, UReg, 1);
            }
            for (uint16_t j = 0; j < tailFoldLoopTimes; ++j) {
                uint32_t count = static_cast<uint32_t>(tailFoldElemCount);
                AscendC::MicroAPI::RegTensor<float> reg0, reg1, reg2;
                AscendC::MicroAPI::MaskReg pMask = AscendC::MicroAPI::UpdateMask<float>(count);

                LoadTensorForDtypeTIn(tailGradA, reg0, pFull, i * outerLoopStride + j * innerLoopStride);
                LoadTensorForDtypeTIn(tailGradB, reg1, pMask, i * outerLoopStride + j * innerLoopStride);

                Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(reg1, reg0, reg1, pMask);
                Copy<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(reg0, reg1, pMask);
                ReduceSum(reg2, reg0, pFull);
                AscendC::MicroAPI::DataCopyUnAlign((__local_mem__ float*&)dst, reg2, UReg, 1);
            }
            for (uint16_t j = 0; j < unFoldLoopTimes; ++j) {
                AscendC::MicroAPI::RegTensor<float> reg0, reg1;
                LoadTensorForDtypeTIn(unFoldGrad, reg0, pFull, i * outerLoopStride + j * innerLoopStride);

                ReduceSum(reg1, reg0, pFull);
                AscendC::MicroAPI::DataCopyUnAlign((__local_mem__ float*&)dst, reg1, UReg, 1);
            }
            AscendC::MicroAPI::DataCopyUnAlignPost((__local_mem__ float*&)dst, UReg, 0);
        }
    }
    NormComputePost(dstTensor, gradTensor, xTensor, reduceSumTempTensor, aSize, foldPoint, outerLoopDstStride);
}

template <typename T>
__aicore__ inline void LogSoftmaxGradAR<T>::NormComputePostWithMul(const LocalTensor<T>& dstTensor,
                                                                const LocalTensor<T>& gradTensor,
                                                                const LocalTensor<T>& xTensor, const int64_t aSize)
{
    uint32_t rSize = tl_->r;
    uint32_t rAligned = tl_->rAligned;

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
        __local_mem__ T* dst = (__local_mem__ T*)dstTensor.GetPhyAddr();
        __local_mem__ T* grad = (__local_mem__ T*)gradTensor.GetPhyAddr();
        __local_mem__ T* x = (__local_mem__ T*)xTensor.GetPhyAddr();

        __VEC_SCOPE__
        {
            uint32_t count = static_cast<uint32_t>(rSize);
            AscendC::MicroAPI::RegTensor<float> reg0, reg1, reg2;
            AscendC::MicroAPI::MaskReg pMask = AscendC::MicroAPI::UpdateMask<float>(count);
            AscendC::MicroAPI::MaskReg pFull =
                AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            AscendC::MicroAPI::MaskReg maskOri;
            for (uint16_t i = 0; i < loopTimes; ++i) {
                LoadTensorForDtypeTIn(grad, reg0, pMask, i * rAligned);
                LoadTensorForDtypeTIn(x, reg1, pMask, i * rAligned);

                ReduceSum(reg2, reg0, pMask);
                Duplicate(reg2, reg2, pFull);

                Exp(reg1, reg1, pMask);
                Mul(reg1, reg1, reg2, pMask);
                Sub(reg1, reg0, reg1, pMask);

                StoreTensorForDtypeTOut(dst, reg1, pMask, i * rAligned);
            }
        }
    } else {
        __local_mem__ T* dst = (__local_mem__ T*)dstTensor.GetPhyAddr();
        __local_mem__ T* grad = (__local_mem__ T*)gradTensor.GetPhyAddr();
        __local_mem__ T* x = (__local_mem__ T*)xTensor.GetPhyAddr();
        __local_mem__ T* grad_1 = (__local_mem__ T*)gradTensor.GetPhyAddr() + VL_FP32;
        __local_mem__ T* x_1 = (__local_mem__ T*)xTensor.GetPhyAddr() + VL_FP32;

        __VEC_SCOPE__
        {
            uint32_t count = static_cast<uint32_t>(rSize - VL_FP32);
            AscendC::MicroAPI::RegTensor<float> reg0, reg1, regExp, reg0_1, reg1_1, reg2, reg2_1, reg2_2;
            AscendC::MicroAPI::MaskReg pMask = AscendC::MicroAPI::UpdateMask<float>(count);
            AscendC::MicroAPI::MaskReg pFull =
                AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            AscendC::MicroAPI::MaskReg maskOri;
            for (uint16_t i = 0; i < loopTimes; ++i) {
                LoadTensorForDtypeTIn(grad, reg0, pFull, i * rAligned);
                LoadTensorForDtypeTIn(x, reg1, pFull, i * rAligned);
                LoadTensorForDtypeTIn(grad_1, reg0_1, pMask, i * rAligned);
                LoadTensorForDtypeTIn(x_1, reg1_1, pMask, i * rAligned);

                Copy<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(reg2_1, reg0, pFull);
                Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(reg2_2, reg0, reg0_1, pMask);
                Copy<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(reg0, reg2_2, pMask);
                ReduceSum(reg2, reg0, pFull);
                Duplicate(reg2, reg2, pFull);

                Exp(regExp, reg1, pFull);
                Mul(reg1, regExp, reg2, pFull);
                Sub(reg1, reg2_1, reg1, pFull);
                StoreTensorForDtypeTOut(dst, reg1, pFull, i * rAligned);

                Exp(regExp, reg1_1, pMask);
                Mul(reg1_1, regExp, reg2, pMask);
                Sub(reg1_1, reg0_1, reg1_1, pMask);
                StoreTensorForDtypeTOut(dst, reg1_1, pMask, i * rAligned + VL_FP32);
            }
        }
    }
}

template <typename T>
__aicore__ inline void LogSoftmaxGradAR<T>::NormComputePost(const LocalTensor<T>& dstTensor,
                                                         const LocalTensor<T>& gradTensor, const LocalTensor<T>& xTensor,
                                                         const LocalTensor<float>& binAddTmpTensor, const int64_t aSize,
                                                         const int64_t rSize, const int64_t stride)
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
    uint16_t rLoopCount = tl_->rLoopCount;
    uint16_t oriR = tl_->r;
    uint16_t oriRAligned = tl_->rAligned;

    if (rSize <= VL_FP32) {
        __local_mem__ T* dst = (__local_mem__ T*)dstTensor.GetPhyAddr();
        __local_mem__ T* grad = (__local_mem__ T*)gradTensor.GetPhyAddr();
        __local_mem__ T* x = (__local_mem__ T*)xTensor.GetPhyAddr();
        __local_mem__ float* sumTmp = (__local_mem__ float*)binAddTmpTensor.GetPhyAddr();

        __VEC_SCOPE__
        {
            uint32_t count = static_cast<uint32_t>(rSize);
            AscendC::MicroAPI::RegTensor<float> reg0, reg1, reg2, regExp;
            AscendC::MicroAPI::MaskReg pMask = AscendC::MicroAPI::UpdateMask<float>(count);
            AscendC::MicroAPI::MaskReg pFull =
                AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            AscendC::MicroAPI::MaskReg maskOri;
            for (uint16_t i = 0; i < loopTimes; ++i) {
                DataCopy(reg0, (__local_mem__ float*)sumTmp + i * static_cast<uint32_t>(stride));
                ReduceSum(reg1, reg0, pMask);
                Duplicate(reg2, reg1, pFull);

                uint32_t sreg0 = static_cast<uint32_t>(oriR);
                for (uint16_t j = 0; j < rLoopCount; ++j) {
                    maskOri = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                    uint32_t offset = j * VL_FP32 + i * oriRAligned;
                    LoadTensorForDtypeTIn(grad, reg0, maskOri, offset);
                    LoadTensorForDtypeTIn(x, reg1, maskOri, offset);
                    Exp(regExp, reg1, maskOri);
                    Mul(reg1, regExp, reg2, maskOri);
                    Sub(reg1, reg0, reg1, maskOri);
                    StoreTensorForDtypeTOut(dst, reg1, maskOri, offset);
                }
            }
        }
    } else {
        __local_mem__ T* dst = (__local_mem__ T*)dstTensor.GetPhyAddr();
        __local_mem__ float* sumTmpA = (__local_mem__ float*)binAddTmpTensor.GetPhyAddr();
        __local_mem__ float* sumTmpB = (__local_mem__ float*)binAddTmpTensor.GetPhyAddr() + VL_FP32;

        __local_mem__ T* grad = (__local_mem__ T*)gradTensor.GetPhyAddr();
        __local_mem__ T* x = (__local_mem__ T*)xTensor.GetPhyAddr();

        __VEC_SCOPE__
        {
            uint32_t count = static_cast<uint32_t>(rSize - VL_FP32);
            AscendC::MicroAPI::RegTensor<float> reg0, reg1, reg2, regExp;
            AscendC::MicroAPI::MaskReg pMask = AscendC::MicroAPI::UpdateMask<float>(count);
            AscendC::MicroAPI::MaskReg pFull =
                AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            AscendC::MicroAPI::MaskReg maskOri;
            for (uint16_t i = 0; i < loopTimes; ++i) {
                DataCopy(reg0, (__local_mem__ float*)sumTmpA + i * stride);
                DataCopy(reg1, (__local_mem__ float*)sumTmpB + i * stride);
                Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(reg1, reg0, reg1, pMask);
                Copy<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(reg0, reg1, pMask);
                ReduceSum(reg2, reg0, pFull);
                Duplicate(reg2, reg2, pFull);
                uint32_t sreg0 = static_cast<uint32_t>(oriR);
                for (uint16_t j = 0; j < rLoopCount; ++j) {
                    maskOri = AscendC::MicroAPI::UpdateMask<float>(sreg0);
                    uint32_t offset = j * VL_FP32 + i * oriRAligned;

                    LoadTensorForDtypeTIn(grad, reg0, maskOri, offset);
                    LoadTensorForDtypeTIn(x, reg1, maskOri, offset);

                    Exp(regExp, reg1, maskOri);
                    Mul(reg1, regExp, reg2, maskOri);
                    Sub(reg1, reg0, reg1, maskOri);
                    StoreTensorForDtypeTOut(dst, reg1, maskOri, offset);
                }
            }
        }
    }
}

template <typename T>
__aicore__ inline void LogSoftmaxGradAR<T>::LoadTensorForDtypeTIn(__local_mem__ T* src, RegTensor<float>& dst,
                                                               MaskReg& preg, uint32_t offset)
{
    if constexpr (IsSameType<T, float>::value) {
        DataCopy<float, LoadDist::DIST_NORM>(dst, (__local_mem__ float*)src + offset);
    } else {  // fp16、bf16
        RegTensor<T> xFp16;
        DataCopy<T, LoadDist::DIST_UNPACK_B16>(xFp16, ((__local_mem__ T*)src + offset));
        Cast<float, T, castTraitFp16ToFp32>(dst, xFp16, preg);
    }
}

template <typename T>
__aicore__ inline void LogSoftmaxGradAR<T>::CopyInX(const LocalTensor<T>& xInUb, const GlobalTensor<T>& xInGm, int64_t ubA,
                                                 int64_t offset)
{
    DataCopyPadParams padParams{false, 0, 0, 0};
    DataCopyParams copyInParams;
    copyInParams.blockCount = ubA;
    copyInParams.blockLen = tl_->r * sizeof(T);
    copyInParams.srcStride = 0;
    copyInParams.dstStride = (tl_->rAligned - tl_->r) * sizeof(T) / BLOCK_SIZE;
    DataCopyPad(xInUb, xInGm[offset], copyInParams, padParams);
}

template <typename T>
__aicore__ inline void LogSoftmaxGradAR<T>::StoreTensorForDtypeTOut(__local_mem__ T* dst,
                                                                 AscendC::MicroAPI::RegTensor<float>& src,
                                                                 AscendC::MicroAPI::MaskReg& preg, uint32_t offset)
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
__aicore__ inline void LogSoftmaxGradAR<T>::CopyOutY(const LocalTensor<T>& yOutUb, int64_t ubA, int64_t offset)
{
    DataCopyParams copyOutParams;
    copyOutParams.blockCount = ubA;
    copyOutParams.blockLen = tl_->r * sizeof(T);
    copyOutParams.srcStride = (tl_->rAligned - tl_->r) * sizeof(T) / BLOCK_SIZE;
    copyOutParams.dstStride = 0;
    DataCopyPad(yGm_[offset], yOutUb, copyOutParams);
}

}  // namespace LogSoftmaxGradOps
#endif // LOG_SOFTMAX_GRAD_AR_FULL_LOAD_H
