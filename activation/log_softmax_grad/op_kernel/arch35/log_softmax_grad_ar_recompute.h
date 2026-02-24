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
 * \file log_softmax_grad_ar_recompute.h
 * \brief
 */

#ifndef LOG_SOFTMAX_GRAD_AR_RECOMPUTE_H
#define LOG_SOFTMAX_GRAD_AR_RECOMPUTE_H

#include "kernel_tiling/kernel_tiling.h"
#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#include "adv_api/reduce/reduce.h"
#else
#include "kernel_operator.h"
#endif
#include "../inc/platform.h"
#include "op_kernel/math_util.h"
#include "log_softmax_grad_base.h"

namespace LogSoftmaxGradOps
{
using namespace AscendC;
constexpr int64_t AR_RECOMPUTE_SUM_BUFFER_BTYES = 32;
constexpr int64_t AR_RECOMPUTE_BINARY_CACHE_BTYES = 2048;
constexpr int64_t AR_RECOMPUTE_SUM_LEN = AR_RECOMPUTE_SUM_BUFFER_BTYES / sizeof(float);
constexpr int64_t A_IN_IN = 1;

template <typename T>
class LogSoftmaxGradArRecompute : public LogSoftmaxGradOpsBase
{
public:
    __aicore__ inline LogSoftmaxGradArRecompute(TPipe* pipe)
    {
        pipe_ = pipe;
    };

    __aicore__ inline void Init(GM_ADDR grad, GM_ADDR x, GM_ADDR y, const SoftmaxGradARRecomputeTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CalculateOutVF(const LocalTensor<T>& yLocal, const LocalTensor<T>& xLocal,
                                          const LocalTensor<T>& gradLocal, __local_mem__ float*& gradSumPtr,
                                          uint32_t a, uint32_t ubFactor);
    __aicore__ inline void CastVF(const LocalTensor<float>& gradFp32Local, const LocalTensor<T>& gradLocal, uint32_t a,
                                  uint32_t ubFactor);
    __aicore__ inline void FoldBlockVF(const LocalTensor<float>& grad1Fp32Local, const LocalTensor<T>& grad2Local,
                                       uint32_t a, uint32_t ubFactor);
    __aicore__ inline void UpdateCache(const LocalTensor<float>& dstTensor, const LocalTensor<float>& srcTensor,
                                       const int64_t cacheId, const int64_t stride, const int64_t count);

protected:
    GlobalTensor<T> gradGm_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;

    const SoftmaxGradARRecomputeTilingData* tl_ = nullptr;
    TPipe* pipe_ = nullptr;
    TQue<QuePosition::VECIN, 1> gradQueue_;
    TQue<QuePosition::VECIN, 1> xQueue_;
    TQue<QuePosition::VECOUT, 1> yQueue_;

    TBuf<> gradSumBuffer;
    TBuf<> cachebuffer;

    uint32_t blockIdx_ = GetBlockIdx();
    uint64_t currentRowBlock_ = 0;
    uint32_t resultCacheID_ = 0;

    LocalTensor<float> totalSumLocal_;

    static constexpr bool xToFp32_ = !IsSameType<T, float>::value;
    static constexpr bool yToFp32_ = IsSameType<T, float>::value;
};

template <typename T>
__aicore__ inline void LogSoftmaxGradArRecompute<T>::Init(GM_ADDR grad, GM_ADDR x, GM_ADDR y,
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

    gradGm_.SetGlobalBuffer((__gm__ T*)grad);
    xGm_.SetGlobalBuffer((__gm__ T*)x);
    yGm_.SetGlobalBuffer((__gm__ T*)y);

    pipe_->InitBuffer(gradQueue_, DOUBLE_BUFFER, tl_->ubFactor * sizeof(T));
    pipe_->InitBuffer(xQueue_, DOUBLE_BUFFER, tl_->ubFactor * sizeof(T));
    pipe_->InitBuffer(yQueue_, DOUBLE_BUFFER, tl_->ubFactor * sizeof(float));

    pipe_->InitBuffer(gradSumBuffer, AR_RECOMPUTE_SUM_BUFFER_BTYES);
    pipe_->InitBuffer(cachebuffer, AR_RECOMPUTE_BINARY_CACHE_BTYES);
}

template <typename T>
__aicore__ inline void LogSoftmaxGradArRecompute<T>::Process()
{
    DataCopyPadExtParams<T> padExtParams{false, 0, 0, 0};

    int64_t xDimOffsetPerCore = tl_->aBlockFactor * blockIdx_;  // 每个核按行的偏移
    LocalTensor<float> gradSumLocal = gradSumBuffer.Get<float>();

    DataCopyExtParams x1DataCopyExtParams;
    x1DataCopyExtParams.blockCount = 1;
    x1DataCopyExtParams.srcStride = 0;
    x1DataCopyExtParams.dstStride = 0;

    DataCopyExtParams x2DataCopyExtParams;
    x2DataCopyExtParams.blockCount = 1;
    x2DataCopyExtParams.srcStride = 0;
    x2DataCopyExtParams.dstStride = 0;

    // 对每行循环
    for (uint64_t rowIdx = 0; rowIdx < currentRowBlock_; rowIdx++) {
        int64_t xDimOffset = (xDimOffsetPerCore + rowIdx) * tl_->r;  // 每行的偏移量

        LocalTensor<float> cacheLocal = cachebuffer.Get<float>();

        // step 1. UB间二分累加：计算每行的Σgrad
        LocalTensor<float> tmpLocal = yQueue_.AllocTensor<float>();
        LocalTensor<float> grad1Fp32Local = tmpLocal[0];

        x1DataCopyExtParams.blockLen = tl_->ubFactor * sizeof(T);
        x2DataCopyExtParams.blockLen = tl_->ubFactor * sizeof(T);

        for (uint64_t basicBlockIdx = 0; basicBlockIdx < tl_->basicBlockLoop; basicBlockIdx++) {
            int64_t gradUbOffset1 = xDimOffset + tl_->ubFactor * basicBlockIdx;                          // 主块
            int64_t gradUbOffset2 = xDimOffset + tl_->ubFactor * (tl_->basicBlockLoop + basicBlockIdx);  // 被折叠块
            LocalTensor<T> grad1Local = gradQueue_.AllocTensor<T>();

            DataCopyPad(grad1Local[0], gradGm_[gradUbOffset1], x1DataCopyExtParams, padExtParams);
            gradQueue_.EnQue<T>(grad1Local);
            grad1Local = gradQueue_.DeQue<T>();

            CastVF(grad1Fp32Local, grad1Local, A_IN_IN, tl_->ubFactor);
            gradQueue_.FreeTensor(grad1Local);

            // 折叠部分：grad2折叠到grad1上
            if (basicBlockIdx < tl_->mainFoldCount) {
                LocalTensor<T> grad2Local = gradQueue_.AllocTensor<T>();
                DataCopyPad(grad2Local[0], gradGm_[gradUbOffset2], x2DataCopyExtParams, padExtParams);
                gradQueue_.EnQue<T>(grad2Local);
                grad2Local = gradQueue_.DeQue<T>();

                FoldBlockVF(grad1Fp32Local, grad2Local, A_IN_IN, tl_->ubFactor);
                gradQueue_.FreeTensor(grad2Local);
            } else if ((basicBlockIdx == tl_->mainFoldCount) && (tl_->ubFactorTail > 0)) {
                LocalTensor<T> grad2Local = gradQueue_.AllocTensor<T>();
                x2DataCopyExtParams.blockLen = tl_->ubFactorTail * sizeof(T);  // 这里的grad2为尾块
                DataCopyPad(grad2Local[0], gradGm_[gradUbOffset2], x2DataCopyExtParams, padExtParams);
                gradQueue_.EnQue<T>(grad2Local);
                grad2Local = gradQueue_.DeQue<T>();

                FoldBlockVF(grad1Fp32Local, grad2Local, A_IN_IN, tl_->ubFactorTail);
                gradQueue_.FreeTensor(grad2Local);
            }

            // 计算UB内二分累加，并用UpdateCache计算UB间的和
            int64_t cacheId = GetCacheID(basicBlockIdx);
            uint32_t srcShape[2] = {uint32_t(A_IN_IN), uint32_t(tl_->ubFactor)};
            AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(gradSumLocal, tmpLocal, srcShape, false);
            UpdateCache(cacheLocal, gradSumLocal, cacheId, A_IN_IN * AR_RECOMPUTE_SUM_LEN, A_IN_IN);
        }

        // R很小，不需要做UB间二分累加
        if (tl_->basicBlockLoop == 0) {
            LocalTensor<T> grad1Local = gradQueue_.AllocTensor<T>();
            DataCopyPad(grad1Local[0], gradGm_[xDimOffset], x1DataCopyExtParams, padExtParams);
            gradQueue_.EnQue<T>(grad1Local);
            grad1Local = gradQueue_.DeQue<T>();

            CastVF(grad1Fp32Local, grad1Local, A_IN_IN, tl_->ubFactor);
            gradQueue_.FreeTensor(grad1Local);
            uint32_t srcShape[2] = {uint32_t(A_IN_IN), uint32_t(tl_->ubFactor)};
            AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(gradSumLocal, tmpLocal, srcShape, false);
        }
        totalSumLocal_ = tl_->basicBlockLoop > 0 ? cacheLocal[resultCacheID_ * AR_RECOMPUTE_SUM_LEN] : gradSumLocal;

        yQueue_.FreeTensor(tmpLocal);

        __local_mem__ float* gradSumPtr = (__local_mem__ float*)totalSumLocal_.GetPhyAddr();
        // step 3. 遍历UB块，计算除法
        for (uint64_t ubIdx = 0; ubIdx < tl_->aLoopCountCeil; ubIdx++) {
            int64_t xUbOffset = xDimOffset + tl_->ubFactor * ubIdx;
            int64_t ubFactor = tl_->ubFactor;
            if (ubIdx == tl_->aLoopCountCeil - 1 && tl_->ubFactorTail > 0) {
                ubFactor = tl_->ubFactorTail;
            }

            LocalTensor<T> gradLocal = gradQueue_.AllocTensor<T>();
            LocalTensor<T> xLocal = xQueue_.AllocTensor<T>();
            LocalTensor<T> yLocal = yQueue_.AllocTensor<T>();

            x1DataCopyExtParams.blockLen = ubFactor * sizeof(T);
            DataCopyPad(xLocal, xGm_[xUbOffset], x1DataCopyExtParams, padExtParams);
            xQueue_.EnQue<T>(xLocal);
            xLocal = xQueue_.DeQue<T>();
            DataCopyPad(gradLocal, gradGm_[xUbOffset], x1DataCopyExtParams, padExtParams);
            gradQueue_.EnQue<T>(gradLocal);
            gradLocal = gradQueue_.DeQue<T>();

            CalculateOutVF(yLocal, xLocal, gradLocal, gradSumPtr, A_IN_IN, ubFactor);
            xQueue_.FreeTensor(xLocal);
            gradQueue_.FreeTensor(gradLocal);
            yQueue_.EnQue<T>(yLocal);
            yLocal = yQueue_.DeQue<T>();

            DataCopyPad(yGm_[xUbOffset], yLocal[0], x1DataCopyExtParams);
            yQueue_.FreeTensor(yLocal);
        }
    }
}


template <typename T>
__aicore__ inline void LogSoftmaxGradArRecompute<T>::CalculateOutVF(const LocalTensor<T>& yLocal,
                                                                    const LocalTensor<T>& xLocal,
                                                                    const LocalTensor<T>& gradLocal,
                                                                    __local_mem__ float*& gradSumPtr,
                                                                    uint32_t a,
                                                                    uint32_t ubFactor)
{
    __local_mem__ T* yPtr = (__local_mem__ T*)yLocal.GetPhyAddr();
    __local_mem__ T* xPtr = (__local_mem__ T*)xLocal.GetPhyAddr();
    __local_mem__ T* gradPtr = (__local_mem__ T*)gradLocal.GetPhyAddr();

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> xRegFp16, gradRegFp16;
        MicroAPI::RegTensor<float> sumReg, xRegFp32, gradRegFp32, expReg, vreg0, vreg1;
        MicroAPI::RegTensor<T> vreg2;
        MicroAPI::MaskReg mask;

        uint32_t width = ubFactor;
        uint16_t repeatTimes = CeilDivision(ubFactor, VL_FP32);

        MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_BRC_B32>(sumReg, gradSumPtr);

        for (uint16_t j = 0; j < repeatTimes; j++) {
            mask = MicroAPI::UpdateMask<float>(width);
            auto gradAddr = gradPtr + j * VL_FP32;
            auto xAddr = xPtr + j * VL_FP32;
            auto yAddr = yPtr + j * VL_FP32;

            if constexpr (xToFp32_) {
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(xRegFp16, xAddr);
                MicroAPI::Cast<float, T, castTraitFp16ToFp32>(xRegFp32, xRegFp16, mask);
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(gradRegFp16, gradAddr);
                MicroAPI::Cast<float, T, castTraitFp16ToFp32>(gradRegFp32, gradRegFp16, mask);
            } else {
                MicroAPI::DataCopy(xRegFp32, xAddr);
                MicroAPI::DataCopy(gradRegFp32, gradAddr);
            }

            MicroAPI::Exp(expReg, xRegFp32, mask);
            MicroAPI::Mul(vreg0, expReg, sumReg, mask);
            MicroAPI::Sub(vreg1, gradRegFp32, vreg0, mask);

            if constexpr (yToFp32_) {
                MicroAPI::DataCopy(yAddr, vreg1, mask);
            } else {
                MicroAPI::Cast<T, float, castTraitFp32ToFp16>(vreg2, vreg1, mask);
                MicroAPI::DataCopy<T, MicroAPI::StoreDist::DIST_PACK_B32>(yAddr, vreg2, mask);
            }
        }
    }
}

template <typename T>
__aicore__ inline void LogSoftmaxGradArRecompute<T>::CastVF(const LocalTensor<float>& gradFp32Local,
                                                            const LocalTensor<T>& gradLocal,
                                                            uint32_t a,
                                                            uint32_t ubFactor)
{
    __local_mem__ float* gradFp32Ptr = (__local_mem__ float*)gradFp32Local.GetPhyAddr();
    __local_mem__ T* gradPtr = (__local_mem__ T*)gradLocal.GetPhyAddr();

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> vreg0;
        MicroAPI::RegTensor<float> vreg1, vreg2, vreg3;
        MicroAPI::MaskReg mask;

        uint32_t width = ubFactor;
        uint16_t repeatTimes = CeilDivision(ubFactor, VL_FP32);

        for (uint16_t j = 0; j < repeatTimes; j++) {
            mask = MicroAPI::UpdateMask<float>(width);
            auto gradAddr = gradPtr + j * VL_FP32;
            auto gradFp32Addr = gradFp32Ptr + j * VL_FP32;

            if constexpr (xToFp32_) {
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(vreg0, gradAddr);
                MicroAPI::Cast<float, T, castTraitFp16ToFp32>(vreg1, vreg0, mask);
            } else {
                MicroAPI::DataCopy(vreg1, gradAddr);
            }

            MicroAPI::DataCopy(gradFp32Addr, vreg1, mask);
        }
    }
}

template <typename T>
__aicore__ inline void LogSoftmaxGradArRecompute<T>::FoldBlockVF(const LocalTensor<float>& grad1Fp32Local,
                                                                 const LocalTensor<T>& grad2Local,
                                                                 uint32_t a,
                                                                 uint32_t ubFactor)
{
    __local_mem__ float* grad1Fp32Ptr = (__local_mem__ float*)grad1Fp32Local.GetPhyAddr();
    __local_mem__ T* grad2Ptr = (__local_mem__ T*)grad2Local.GetPhyAddr();

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<T> vreg0;
        MicroAPI::RegTensor<float> vreg1, vreg2, vreg3;
        MicroAPI::MaskReg mask;
        MicroAPI::MaskReg maskFull = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();

        uint16_t foldTimes = CeilDivision(ubFactor, VL_FP32);

        uint32_t width = ubFactor;
        for (uint16_t j = 0; j < foldTimes; j++) {
            mask = MicroAPI::UpdateMask<float>(width);
            auto grad1Addr = grad1Fp32Ptr + j * VL_FP32;
            auto grad2Addr = grad2Ptr + j * VL_FP32;

            if constexpr (xToFp32_) {
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(vreg0, grad2Addr);
                MicroAPI::Cast<float, T, castTraitFp16ToFp32>(vreg2, vreg0, mask);
            } else {
                MicroAPI::DataCopy(vreg2, grad2Addr);
            }

            MicroAPI::DataCopy(vreg1, grad1Addr);

            MicroAPI::Add(vreg3, vreg1, vreg2, mask);
            MicroAPI::Copy<float, MicroAPI::MaskMergeMode::MERGING>(vreg1, vreg3, mask);

            MicroAPI::DataCopy(grad1Addr, vreg1, maskFull);
        }
    }
}

template <typename T>
__aicore__ inline void LogSoftmaxGradArRecompute<T>::UpdateCache(const LocalTensor<float>& dstTensor,
                                                                 const LocalTensor<float>& srcTensor,
                                                                 const int64_t cacheId, const int64_t stride,
                                                                 const int64_t count)
{
    uint16_t outerLoopTimes =
        Ops::Base::CeilDiv(static_cast<int64_t>(count * sizeof(float)), static_cast<int64_t>(platform::GetVRegSize()));
    uint16_t innerLoopTimes = cacheId;
    uint32_t outerLoopStride = VL_FP32;
    uint32_t innerLoopStride = stride;

    __local_mem__ float* dst = (__local_mem__ float*)dstTensor.GetPhyAddr();
    __local_mem__ float* cache = (__local_mem__ float*)dstTensor.GetPhyAddr() + cacheId * stride;
    __local_mem__ float* src = (__local_mem__ float*)srcTensor.GetPhyAddr();

    __VEC_SCOPE__
    {
        uint32_t sreg = static_cast<uint32_t>(count);
        MicroAPI::RegTensor<float> aReg, bReg;
        MicroAPI::MaskReg pMask;
        for (uint16_t i = 0; i < outerLoopTimes; ++i) {
            pMask = MicroAPI::UpdateMask<float>(sreg);
            MicroAPI::DataCopy(aReg, (__local_mem__ float*)src + i * outerLoopStride);
            for (uint16_t j = 0; j < innerLoopTimes; ++j) {
                MicroAPI::DataCopy(bReg, (__local_mem__ float*)dst + i * outerLoopStride + j * innerLoopStride);
                MicroAPI::Add<float, MicroAPI::MaskMergeMode::ZEROING>(aReg, aReg, bReg, pMask);
            }
            MicroAPI::DataCopy((__local_mem__ float*)cache + i * outerLoopStride, aReg, pMask);
        }
    }
}
}  // namespace LogSoftmaxGradOps
#endif // LOG_SOFTMAX_GRAD_AR_RECOMPUTE_H
