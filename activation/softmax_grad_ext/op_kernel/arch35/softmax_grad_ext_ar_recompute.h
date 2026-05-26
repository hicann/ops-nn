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
 * \file softmax_grad_ext_ar_recompute.h
 * \brief
 */

#ifndef SOFTMAX_GRAD_EXT_AR_RECOMPUTE_H
#define SOFTMAX_GRAD_EXT_AR_RECOMPUTE_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"
#include "softmax_grad_ext_base.h"

namespace SoftmaxGradExt {
using namespace AscendC;

static constexpr int64_t AR_RECOMPUTE_SUM_BUFFER_BTYES = 32;
static constexpr int64_t AR_RECOMPUTE_BINARY_TMP_BTYES = 512;
static constexpr int64_t AR_RECOMPUTE_BINARY_CACHE_BTYES = 2048;
static constexpr int64_t AR_RECOMPUTE_SUM_LEN = AR_RECOMPUTE_SUM_BUFFER_BTYES / sizeof(float);
static constexpr float CONST_FP32_ZERO = 0.0;
static constexpr int64_t A_IN_IN = 1;
static constexpr int32_t AR_BUFFER_BUM = 2;
static constexpr int32_t DOUBLE_BUFFER = 2;

template <typename T>
class SoftmaxGradExtARRecompute : public SoftmaxGradExtBase {
public:
    __aicore__ inline SoftmaxGradExtARRecompute(TPipe* pipe)
    {
        pipe_ = pipe;
    };

    __aicore__ inline void Init(
        GM_ADDR grad, GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const SoftmaxGradExtARRecomputeTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CalcReduceSum(int64_t xDimOffset);
    __aicore__ inline void CalcOutVF(uint32_t ubFactor);
    __aicore__ inline void CalcOutVFSameShape(uint32_t ubFactor);

    __aicore__ inline void MainBlockVF(__local_mem__ float* dst, uint32_t ubFactor);
    __aicore__ inline void FoldBlockVF(__local_mem__ float* dst, uint32_t ubFactor);

    __aicore__ inline void LastReduceSum(
        const LocalTensor<float>& dstTensor, const LocalTensor<float>& srcTensor,
        const LocalTensor<float>& reduceSumTempTensor, const int64_t aSize, const int64_t rSize, const int64_t stride);

    __aicore__ inline void LastReduceSumSmallR(
        const LocalTensor<float>& dstTensor, const LocalTensor<float>& srcTensor, const int64_t aSize,
        const int64_t rSize, const int64_t stride);

    __aicore__ inline void LoadTensorForDtypeT(
        __local_mem__ T* src, AscendC::MicroAPI::RegTensor<float>& dst, AscendC::MicroAPI::MaskReg& pregMask,
        uint32_t offset);
    __aicore__ inline void StoreTensorForDtypeTOut(
        __local_mem__ T* dst, AscendC::MicroAPI::RegTensor<float>& src, AscendC::MicroAPI::MaskReg& preg,
        uint32_t offset);

    __aicore__ inline void CopyInX(int64_t xGmOffset, uint32_t ubFactor);
    __aicore__ inline void CopyInXOnePoint(int64_t xGmOffset, uint32_t ubFactor);
    __aicore__ inline void CopyInXSameShape(int64_t xGmOffset, uint32_t ubFactor);
    __aicore__ inline void CopyOutY(int64_t yGmOffset, int64_t ubFactor);

protected:
    GlobalTensor<T> gradGm_;
    GlobalTensor<T> x0Gm_;
    GlobalTensor<T> x1Gm_;
    GlobalTensor<T> yGm_;

    const SoftmaxGradExtARRecomputeTilingData* tl_ = nullptr;
    TPipe* pipe_ = nullptr;
    TQue<QuePosition::VECIN, 1> gradQueue_;
    TQue<QuePosition::VECIN, 1> x0Queue_;
    TQue<QuePosition::VECIN, 1> x1Queue_;
    TQue<QuePosition::VECOUT, 1> yQueue_;

    TBuf<> xSumBuffer_;
    TBuf<> reduceSumTempTensor_;
    TBuf<> cachebuffer_;

    uint32_t blockIdx_ = GetBlockIdx();
    uint64_t currentRowBlock_ = 0;
    uint32_t resultCacheID_ = 0;
    uint64_t isX2Scalar_ = 0;

    LocalTensor<float> xSumTensor_;
};

template <typename T>
__aicore__ inline void SoftmaxGradExtARRecompute<T>::Init(
    GM_ADDR grad, GM_ADDR x0, GM_ADDR x1, GM_ADDR y, const SoftmaxGradExtARRecomputeTilingData* tilingData)
{
    tl_ = tilingData;

    // 普通核数量，比如coreNum为48，a为95，那么普通核处理2个a，其中一个尾核处理1个a
    int64_t rowBlockCount = ops::FloorDiv(tl_->a, tl_->aBlockFactor);     // 普通核数量
    int64_t tailBlockFactor = tl_->a - rowBlockCount * tl_->aBlockFactor; // 尾核处理a的数量

    // 设置currentRowBlock_，当前核计算的a数量
    if (blockIdx_ < rowBlockCount) {
        currentRowBlock_ = tl_->aBlockFactor;
    } else {
        currentRowBlock_ = tailBlockFactor;
    }

    isX2Scalar_ = tl_->x2IsScalar;

    if (tl_->basicBlockLoop == 0) {
        resultCacheID_ = 0;
    } else {
        resultCacheID_ = GetCacheID(tl_->basicBlockLoop - 1);
    }

    gradGm_.SetGlobalBuffer((__gm__ T*)grad);
    x0Gm_.SetGlobalBuffer((__gm__ T*)x0);
    x1Gm_.SetGlobalBuffer((__gm__ T*)x1);
    yGm_.SetGlobalBuffer((__gm__ T*)y);

    // 每个UB能放下ubFactor个数的成套数据
    pipe_->InitBuffer(gradQueue_, DOUBLE_BUFFER, tl_->ubFactor * sizeof(T));
    pipe_->InitBuffer(x0Queue_, DOUBLE_BUFFER, tl_->ubFactor * sizeof(T));
    pipe_->InitBuffer(x1Queue_, DOUBLE_BUFFER, tl_->ubFactor * sizeof(T));
    pipe_->InitBuffer(yQueue_, DOUBLE_BUFFER, tl_->ubFactor * sizeof(float));
    pipe_->InitBuffer(xSumBuffer_, AR_RECOMPUTE_SUM_BUFFER_BTYES);          // 32B
    pipe_->InitBuffer(reduceSumTempTensor_, AR_RECOMPUTE_BINARY_TMP_BTYES); // 512B
    pipe_->InitBuffer(cachebuffer_, AR_RECOMPUTE_BINARY_CACHE_BTYES);       // 2048B
}

template <typename T>
__aicore__ inline void SoftmaxGradExtARRecompute<T>::Process()
{
    int64_t xDimOffsetPerCore = tl_->aBlockFactor * blockIdx_; // 每个核按行的维度偏移，48核，阿伟95，每个核偏移2
    if (isX2Scalar_ == 0) {
        // 每个核计算a的个数的for循环
        for (uint64_t rowIdx = 0; rowIdx < currentRowBlock_; rowIdx++) {
            int64_t xDimOffset = (xDimOffsetPerCore + rowIdx) * tl_->r; // 每行的偏移地址，偏移量为r
            // 最终ReduceSum放在 xSumTensor_
            CalcReduceSum(xDimOffset);

            for (uint64_t ubIdx = 0; ubIdx < tl_->aLoopCountCeil; ubIdx++) {
                int64_t xUbOffset = xDimOffset + tl_->ubFactor * ubIdx;
                int64_t ubFactor = tl_->ubFactor;
                if (ubIdx == tl_->aLoopCountCeil - 1 && tl_->ubFactorTail > 0) {
                    ubFactor = tl_->ubFactorTail;
                }
                // 同shape输出
                CopyInXSameShape(xUbOffset, ubFactor);
                CalcOutVFSameShape(ubFactor);
                CopyOutY(xUbOffset, ubFactor);
            }
        }
    } else {
        // 每个核计算a的个数的for循环
        for (uint64_t rowIdx = 0; rowIdx < currentRowBlock_; rowIdx++) {
            int64_t xDimOffset = (xDimOffsetPerCore + rowIdx) * tl_->r; // 每行的偏移地址，偏移量为r
            // 最终ReduceSum放在 xSumTensor_
            CalcReduceSum(xDimOffset);

            for (uint64_t ubIdx = 0; ubIdx < tl_->aLoopCountCeil; ubIdx++) {
                int64_t xUbOffset = xDimOffset + tl_->ubFactor * ubIdx;
                int64_t ubFactor = tl_->ubFactor;
                if (ubIdx == tl_->aLoopCountCeil - 1 && tl_->ubFactorTail > 0) {
                    ubFactor = tl_->ubFactorTail;
                }
                // 单点输出
                // 从GM地址[xUbOffset]开始搬ubFactor个数元素到UB，同时搬grad和x0
                CopyInXOnePoint(xUbOffset, ubFactor);
                CalcOutVF(ubFactor);
                CopyOutY(xUbOffset, ubFactor);
            }
        }
    }
}

// 计算r轴上的reduceSum
template <typename T>
__aicore__ inline void SoftmaxGradExtARRecompute<T>::CalcReduceSum(int64_t xDimOffset)
{
    LocalTensor<float> reduceSumTempLocal = reduceSumTempTensor_.Get<float>(); // 大小为512B
    LocalTensor<float> cacheLocal = cachebuffer_.Get<float>();                 // 大小为2048B
    LocalTensor<float> xSum = xSumBuffer_.Get<float>();                        // 大小为32B

    LocalTensor<float> xTmp = yQueue_.AllocTensor<float>(); // 复用y做二分累加
    __local_mem__ float* xTmpLocal = (__local_mem__ float*)xTmp.GetPhyAddr();

    // ub间累加fold折叠到main
    for (uint64_t basicBlockIdx = 0; basicBlockIdx < tl_->basicBlockLoop; basicBlockIdx++) {
        int64_t xMainOffset = xDimOffset + tl_->ubFactor * basicBlockIdx;
        int64_t xFoldOffset = xDimOffset + tl_->ubFactor * (tl_->basicBlockLoop + basicBlockIdx);
        // CopyInX实际搬入grad和x0
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
    // ReduceSum最终结果放在xSumTensor_
    xSumTensor_ = tl_->basicBlockLoop > 0 ? cacheLocal[resultCacheID_ * AR_RECOMPUTE_SUM_LEN] : xSum;
}

// cast + mul，cast成FP32之后求gradi * x0i
template <typename T>
__aicore__ inline void SoftmaxGradExtARRecompute<T>::MainBlockVF(__local_mem__ float* dst, uint32_t ubFactor)
{
    LocalTensor<T> x0 = x0Queue_.DeQue<T>();
    LocalTensor<T> grad = gradQueue_.DeQue<T>();

    __local_mem__ T* x0Local = (__local_mem__ T*)x0.GetPhyAddr();
    __local_mem__ T* gradLocal = (__local_mem__ T*)grad.GetPhyAddr();

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
            LoadTensorForDtypeT(gradLocal, reg1, pregMask, offset);

            Mul(reg0, reg0, reg1, pregMask);

            AscendC::MicroAPI::DataCopy(dst + offset, reg0, pregMask);
        }
    }
    x0Queue_.FreeTensor(x0);
    gradQueue_.FreeTensor(grad);
}

// fold块的cast + mul
template <typename T>
__aicore__ inline void SoftmaxGradExtARRecompute<T>::FoldBlockVF(__local_mem__ float* dst, uint32_t ubFactor)
{
    LocalTensor<T> x0 = x0Queue_.DeQue<T>();
    LocalTensor<T> grad = gradQueue_.DeQue<T>();
    __local_mem__ T* x0Local = (__local_mem__ T*)x0.GetPhyAddr();
    __local_mem__ T* gradLocal = (__local_mem__ T*)grad.GetPhyAddr();

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
            LoadTensorForDtypeT(gradLocal, reg1, pregMask, offset);

            Mul(reg0, reg1, reg0, pregMask);

            AscendC::MicroAPI::DataCopy(reg1, dst + offset);

            AscendC::MicroAPI::Add(reg0, reg1, reg0, pregMask);
            AscendC::MicroAPI::Copy<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(reg1, reg0, pregMask);

            AscendC::MicroAPI::DataCopy(dst + offset, reg1, maskFull);
        }
    }

    x0Queue_.FreeTensor(x0);
    gradQueue_.FreeTensor(grad);
}

// 单点计算
template <typename T>
__aicore__ inline void SoftmaxGradExtARRecompute<T>::CalcOutVF(uint32_t ubFactor)
{
    LocalTensor<T> x0 = x0Queue_.DeQue<T>();
    LocalTensor<T> x1 = x1Queue_.DeQue<T>();
    LocalTensor<T> grad = gradQueue_.DeQue<T>();
    LocalTensor<T> y = yQueue_.AllocTensor<T>();

    Duplicate<T>(x1, x1, ubFactor);

    __local_mem__ float* xSumLocal = (__local_mem__ float*)xSumTensor_.GetPhyAddr();
    __local_mem__ T* x0Local = (__local_mem__ T*)x0.GetPhyAddr();
    __local_mem__ T* x1Local = (__local_mem__ T*)x1.GetPhyAddr();
    __local_mem__ T* gradLocal = (__local_mem__ T*)grad.GetPhyAddr();
    __local_mem__ T* yLocal = (__local_mem__ T*)y.GetPhyAddr();

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<float> sumReg, x0Reg, gradReg, x1Reg;
        AscendC::MicroAPI::MaskReg pregMask;

        uint32_t sreg = ubFactor;
        uint16_t loopTimes = CeilDivision(ubFactor, VL_FP32);

        AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(sumReg, xSumLocal);
        for (uint16_t j = 0; j < loopTimes; j++) {
            pregMask = AscendC::MicroAPI::UpdateMask<float>(sreg);
            uint32_t offset = j * VL_FP32;
            LoadTensorForDtypeT(gradLocal, gradReg, pregMask, offset);
            LoadTensorForDtypeT(x0Local, x0Reg, pregMask, offset);
            LoadTensorForDtypeT(x1Local, x1Reg, pregMask, offset);

            // 计算公式：yi = (x0i * gradi - x0i * (reduceSum)) * x1i
            // 计算公式：x0i * gradi
            Mul(gradReg, gradReg, x0Reg, pregMask);
            Mul(x0Reg, x0Reg, sumReg, pregMask);
            // 计算公式：(x0i * gradi - x0i * (reduceSum))
            Sub(gradReg, gradReg, x0Reg, pregMask);
            // 计算公式：yi = (x0i * gradi - x0i * (reduceSum)) * x1i
            Mul(gradReg, gradReg, x1Reg, pregMask);
            StoreTensorForDtypeTOut(yLocal, gradReg, pregMask, offset);
        }
    }
    yQueue_.EnQue(y);
    x0Queue_.FreeTensor(x0);
    x1Queue_.FreeTensor(x1);
    gradQueue_.FreeTensor(grad);
}

// 同shape计算
template <typename T>
__aicore__ inline void SoftmaxGradExtARRecompute<T>::CalcOutVFSameShape(uint32_t ubFactor)
{
    LocalTensor<T> x0 = x0Queue_.DeQue<T>();
    LocalTensor<T> x1 = x1Queue_.DeQue<T>();
    LocalTensor<T> grad = gradQueue_.DeQue<T>();
    LocalTensor<T> y = yQueue_.AllocTensor<T>();

    __local_mem__ float* xSumLocal = (__local_mem__ float*)xSumTensor_.GetPhyAddr();
    __local_mem__ T* x0Local = (__local_mem__ T*)x0.GetPhyAddr();
    __local_mem__ T* x1Local = (__local_mem__ T*)x1.GetPhyAddr();
    __local_mem__ T* gradLocal = (__local_mem__ T*)grad.GetPhyAddr();
    __local_mem__ T* yLocal = (__local_mem__ T*)y.GetPhyAddr();

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<float> sumReg, x0Reg, gradReg, x1Reg;
        AscendC::MicroAPI::MaskReg pregMask;

        uint32_t sreg = ubFactor;
        uint16_t loopTimes = CeilDivision(ubFactor, VL_FP32);

        AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(sumReg, xSumLocal);
        for (uint16_t j = 0; j < loopTimes; j++) {
            pregMask = AscendC::MicroAPI::UpdateMask<float>(sreg);
            uint32_t offset = j * VL_FP32;
            LoadTensorForDtypeT(gradLocal, gradReg, pregMask, offset);
            LoadTensorForDtypeT(x0Local, x0Reg, pregMask, offset);
            LoadTensorForDtypeT(x1Local, x1Reg, pregMask, offset);

            // 计算公式：yi = (x0i * gradi - x0i * (reduceSum)) * x1i
            // 计算公式：x0i * gradi
            Mul(gradReg, gradReg, x0Reg, pregMask);
            Mul(x0Reg, x0Reg, sumReg, pregMask);
            // 计算公式：(x0i * gradi - x0i * (reduceSum))
            Sub(gradReg, gradReg, x0Reg, pregMask);
            // 计算公式：yi = (x0i * gradi - x0i * (reduceSum)) * x1i
            Mul(gradReg, gradReg, x1Reg, pregMask);
            StoreTensorForDtypeTOut(yLocal, gradReg, pregMask, offset);
        }
    }
    yQueue_.EnQue(y);
    x0Queue_.FreeTensor(x0);
    x1Queue_.FreeTensor(x1);
    gradQueue_.FreeTensor(grad);
}

// 从UB搬到寄存器的同时转换数据类型
template <typename T>
__aicore__ inline void SoftmaxGradExtARRecompute<T>::LoadTensorForDtypeT(
    __local_mem__ T* src, AscendC::MicroAPI::RegTensor<float>& dst, AscendC::MicroAPI::MaskReg& pregMask,
    uint32_t offset)
{
    if constexpr (IsSameType<T, float>::value) {
        DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(dst, (__local_mem__ float*)src + offset);
    } else { // fp16、bf16
        AscendC::MicroAPI::RegTensor<T> xFp16;
        DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(xFp16, ((__local_mem__ T*)src + offset));
        Cast<float, T, castTraitFp16ToFp32>(dst, xFp16, pregMask);
    }
}

// 从寄存器搬到UB的同时转换为原来数据类型
template <typename T>
__aicore__ inline void SoftmaxGradExtARRecompute<T>::StoreTensorForDtypeTOut(
    __local_mem__ T* dst, AscendC::MicroAPI::RegTensor<float>& src, AscendC::MicroAPI::MaskReg& preg, uint32_t offset)
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
__aicore__ inline void SoftmaxGradExtARRecompute<T>::CopyOutY(int64_t yGmOffset, int64_t ubFactor)
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

// 从地址[地址偏移]开始搬出ubFactor个数元素，同时搬grad和x0
template <typename T>
__aicore__ inline void SoftmaxGradExtARRecompute<T>::CopyInX(int64_t xGmOffset, uint32_t ubFactor)
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

    LocalTensor<T> grad = gradQueue_.AllocTensor<T>();
    DataCopyPad(grad, gradGm_[xGmOffset], params, padParams);
    gradQueue_.EnQue(grad);
}

// 从地址[地址偏移]开始搬出ubFactor个数元素，同时搬grad、x0和x1
template <typename T>
__aicore__ inline void SoftmaxGradExtARRecompute<T>::CopyInXSameShape(int64_t xGmOffset, uint32_t ubFactor)
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

    LocalTensor<T> grad = gradQueue_.AllocTensor<T>();
    DataCopyPad(grad, gradGm_[xGmOffset], params, padParams);
    gradQueue_.EnQue(grad);
}

// 从地址[地址偏移]开始搬出ubFactor个数元素，同时搬grad、x0和x1
template <typename T>
__aicore__ inline void SoftmaxGradExtARRecompute<T>::CopyInXOnePoint(int64_t xGmOffset, uint32_t ubFactor)
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

    LocalTensor<T> grad = gradQueue_.AllocTensor<T>();
    DataCopyPad(grad, gradGm_[xGmOffset], params, padParams);
    gradQueue_.EnQue(grad);

    DataCopyExtParams x2CopyParams{
        static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(T)), static_cast<uint32_t>(0), static_cast<uint32_t>(0),
        static_cast<uint32_t>(0)};
    DataCopyPadExtParams<T> x2PadParams{true, static_cast<uint8_t>(0), static_cast<uint8_t>(0), static_cast<T>(0.0)};

    LocalTensor<T> x1 = x1Queue_.AllocTensor<T>();
    DataCopyPad(x1, x1Gm_[0], x2CopyParams, x2PadParams);
    x1Queue_.EnQue(x1);
}

// 最终的ReduceSum
template <typename T>
__aicore__ inline void SoftmaxGradExtARRecompute<T>::LastReduceSum(
    const LocalTensor<float>& dstTensor, const LocalTensor<float>& srcTensor,
    const LocalTensor<float>& reduceSumTempTensor, const int64_t aSize, const int64_t rSize, const int64_t stride)
{
    if (aSize <= 0 || rSize <= 0) {
        return;
    }

    if (rSize <= CONST_TWO * VL_FP32) {
        LastReduceSumSmallR(dstTensor, srcTensor, aSize, rSize, stride);
        return;
    }

    int64_t ceilVLCount =
        ops::CeilDiv(static_cast<int64_t>(rSize * sizeof(float)), static_cast<int64_t>(platform::GetVRegSize()));
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

// R很小情况下的最终的ReduceSum
template <typename T>
__aicore__ inline void SoftmaxGradExtARRecompute<T>::LastReduceSumSmallR(
    const LocalTensor<float>& dstTensor, const LocalTensor<float>& srcTensor, const int64_t aSize, const int64_t rSize,
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
                AscendC::MicroAPI::DataCopy(aReg, (__local_mem__ float*)src + i * stride);
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
                AscendC::MicroAPI::DataCopy(aReg, (__local_mem__ float*)src0 + i * stride);
                AscendC::MicroAPI::DataCopy(bReg, (__local_mem__ float*)src1 + i * stride);
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

} // namespace SoftmaxGradExt
#endif // SOFTMAX_GRAD_EXT_AR_RECOMPUTE_H