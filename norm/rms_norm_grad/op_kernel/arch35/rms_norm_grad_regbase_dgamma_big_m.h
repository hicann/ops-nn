/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file rms_norm_grad_regbase_dgamma_big_m.h
 * \brief
 */

#ifndef RMS_NORM_GRAD_DGAMMA_BIG_M_
#define RMS_NORM_GRAD_DGAMMA_BIG_M_

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "../inc/kernel_utils.h"
#include "rms_norm_grad_dgamma_regbase_helper.h"
#include "rms_norm_grad_regbase_common.h"

namespace RmsNormGrad {

template <typename T>
class RmsNormGradDgammaBigM
{
    static constexpr uint32_t VECTOR_LENGTH = platform::GetVRegSize();
    static constexpr uint32_t VL_FP32 = VECTOR_LENGTH / sizeof(float);
    static constexpr int64_t BLOCK_SIZE = platform::GetUbBlockSize();
    static constexpr int64_t TRIPLE_BUFFER = 3;
    static constexpr int64_t DOUBLE_BUFFER = 2;

public:
    __aicore__ inline RmsNormGradDgammaBigM(){};
    __aicore__ inline void Init(
        GM_ADDR dy, GM_ADDR x, GM_ADDR rstd, GM_ADDR pdGamma, GM_ADDR workspace,
        const RmsNormGradBigMTilingData* tilingData, TPipe* pipeIn)
    {
        td_ = tilingData;
        blockIdx_ = GetBlockIdx();
        if (blockIdx_ >= td_->dgammaUsedCoreNum) {
            return;
        }

        const int64_t startM =
            (blockIdx_ * td_->dgammaMPerBlock) + (blockIdx_ < td_->dgammaMReminder ? blockIdx_ : td_->dgammaMReminder);

        if (blockIdx_ < td_->dgammaMReminder) {
            M = td_->dgammaMToProcessMainBlock;
            Mloop = td_->dgammaMLoopMainBlock;
            MTotalLoop = td_->dgammaMTotalLoopMainBlock;
            Mtail = td_->dgammaMTailMainBlock;
            BasicBlockLoop = td_->dgammaBasicBlockLoopMainBlock;
            MainFoldCount = td_->dgammaMainFoldCountMainBlock;
            CacheBufferCount = td_->dgammaCacheBufferCountMainBlock;
            ResultCacheID = td_->dgammaResultCacheIDMainBlock;
        } else {
            M = td_->dgammaMToProcessTailBlock;
            Mloop = td_->dgammaMLoopTailBlock;
            MTotalLoop = td_->dgammaMTotalLoopTailBlock;
            Mtail = td_->dgammaMTailTailBlock;
            BasicBlockLoop = td_->dgammaBasicBlockLoopTailBlock;
            MainFoldCount = td_->dgammaMainFoldCountTailBlock;
            CacheBufferCount = td_->dgammaCacheBufferCountTailBlock;
            ResultCacheID = td_->dgammaResultCacheIDTailBlock;
        }

        // Init GM
        int64_t dyOffset = startM * td_->dxTilingData.cols;
        dyInGm_.SetGlobalBuffer((__gm__ T*)dy + dyOffset);
        xInGm_.SetGlobalBuffer((__gm__ T*)x + dyOffset);
        rstdInGm_.SetGlobalBuffer((__gm__ float*)rstd + startM);

        int64_t colOffset = blockIdx_ * td_->dxTilingData.cols;
        dgammaTmpGm_.SetGlobalBuffer((__gm__ float*)workspace + colOffset);
        dgammaGm_.SetGlobalBuffer((__gm__ float*)pdGamma);

        // Init Pipe
        pipe_ = pipeIn;

        int64_t dyBufLen = td_->dgammaMfactorBlockAligned * td_->dgammaNfactorBlockAligned;
        pipe_->InitBuffer(inQueueDy_, TRIPLE_BUFFER, dyBufLen * sizeof(T));
        pipe_->InitBuffer(inQueueX_, TRIPLE_BUFFER, dyBufLen * sizeof(T));
        pipe_->InitBuffer(dgammaCalcBuf_, dyBufLen * sizeof(float));

        pipe_->InitBuffer(inQueueRstd_, TRIPLE_BUFFER, td_->dgammaMfactorBlockAligned * sizeof(float));
        int64_t nFactorAlignedBufSize = td_->dgammaNfactorBlockAligned * sizeof(float);
        pipe_->InitBuffer(dgammaTmpOutQueue_, DOUBLE_BUFFER, nFactorAlignedBufSize);
        pipe_->InitBuffer(reduceOutTmpBuffer_, nFactorAlignedBufSize);

        int64_t cacheBufSize = CacheBufferCount * nFactorAlignedBufSize;
        pipe_->InitBuffer(reduceCacheBuffer_, cacheBufSize);
    }

    __aicore__ inline void Process()
    {
        // 核内计算
        if (blockIdx_ < td_->dgammaUsedCoreNum) {
            ProcessStg0();
        }

        SyncAll();

        // 0核做核间累加
        if (blockIdx_ != 0) {
            return;
        }

        InitBufferStg1();
        ProcessStg1();
    }

private:
    // 核内计算
    __aicore__ inline void ProcessStg0()
    {
        dgammaCalcTensor_ = dgammaCalcBuf_.Get<float>();

        reduceOutTmpTensor_ = reduceOutTmpBuffer_.Get<float>();
        reduceCacheTensor_ = reduceCacheBuffer_.Get<float>();

        int64_t totalRounds = td_->dgammaNloop + (td_->dgammaNtail > 0 ? 1 : 0);

        int64_t mfactor = BasicBlockLoop ?
                              td_->dgammaMfactorBlockAligned :
                              (M == td_->dgammaMfactorBlockAligned ? td_->dgammaMfactorBlockAligned : Mtail);
        int64_t loopCnt = BasicBlockLoop ? BasicBlockLoop : 1;

        for (int64_t round = 0; round < totalRounds; ++round) {
            int64_t ni = (round < td_->dgammaNloop) ? round : td_->dgammaNloop;
            int64_t nfactor = (round < td_->dgammaNloop) ? td_->dgammaNfactorBlockAligned : td_->dgammaNtail;

            for (int64_t i = 0; i < loopCnt; ++i) {
                ProcessMainBlock(ni, i, mfactor, nfactor);
                if (BasicBlockLoop != 0 && ((i < MainFoldCount) || (i == MainFoldCount && Mtail > 0))) {
                    ProcessFoldBlock(ni, i, (i < MainFoldCount) ? td_->dgammaMfactorBlockAligned : Mtail, nfactor);
                }
                ProcessSummation(ni, i, mfactor, td_->dgammaNfactorBlockAligned);
            }

            LocalTensor<float> dgammaTmpOutTensor = dgammaTmpOutQueue_.template AllocTensor<float>();

            CopyUB2UB(dgammaTmpOutTensor, reduceCacheTensor_[ResultCacheID * td_->dgammaNfactorBlockAligned], nfactor);

            dgammaTmpOutQueue_.EnQue(dgammaTmpOutTensor);

            int64_t offset = ni * td_->dgammaNfactorBlockAligned;
            CopyOutDgammaTmpStg0(offset, nfactor);
        }
    }

    __aicore__ inline void ProcessMainBlock(
        const int64_t ni, const int64_t basicBlockIdx, const int64_t mfactor, const int64_t nfactor)
    {
        int64_t dyOffset = ni * td_->dgammaNfactorBlockAligned +
                           basicBlockIdx * td_->dgammaMfactorBlockAligned * td_->dxTilingData.cols;

        CopyInDyAndX(dyOffset, mfactor, nfactor, td_->dgammaNfactorBlockAligned, td_->dxTilingData.cols);

        int64_t rstdOffset = basicBlockIdx * td_->dgammaMfactorBlockAligned;

        CopyInRstd(rstdOffset, mfactor);

        LocalTensor<T> dyTensor = inQueueDy_.template DeQue<T>();
        LocalTensor<T> xTensor = inQueueX_.template DeQue<T>();
        LocalTensor<float> rstdTensor = inQueueRstd_.template DeQue<float>();
        ComputeDgammaTmpMain(dgammaCalcTensor_, dyTensor, xTensor, rstdTensor, mfactor, td_->dgammaNfactorBlockAligned);
        inQueueDy_.FreeTensor(dyTensor);
        inQueueX_.FreeTensor(xTensor);
        inQueueRstd_.FreeTensor(rstdTensor);
    }

    __aicore__ inline void ProcessFoldBlock(
        const int64_t ni, const int64_t basicBlockIdx, const int64_t mfactor, const int64_t nfactor)
    {
        int64_t dyOffset = ni * td_->dgammaNfactorBlockAligned +
                           (basicBlockIdx + BasicBlockLoop) * td_->dgammaMfactorBlockAligned * td_->dxTilingData.cols;

        CopyInDyAndX(dyOffset, mfactor, nfactor, td_->dgammaNfactorBlockAligned, td_->dxTilingData.cols);
        int64_t rstdOffset = (basicBlockIdx + BasicBlockLoop) * td_->dgammaMfactorBlockAligned;
        CopyInRstd(rstdOffset, mfactor);

        LocalTensor<T> dyTensor = inQueueDy_.template DeQue<T>();
        LocalTensor<T> xTensor = inQueueX_.template DeQue<T>();
        LocalTensor<float> rstdTensor = inQueueRstd_.template DeQue<float>();
        ComputeDgammaTmpFold(dgammaCalcTensor_, dyTensor, xTensor, rstdTensor, mfactor, td_->dgammaNfactorBlockAligned);
        inQueueDy_.FreeTensor(dyTensor);
        inQueueX_.FreeTensor(xTensor);
        inQueueRstd_.FreeTensor(rstdTensor);
    }

    __aicore__ inline void ProcessSummation(
        const int64_t ni, const int64_t basicBlockIdx, const int64_t mfactor, const int64_t nfactor)
    {
        int64_t cacheID = GetCacheID(basicBlockIdx);
        uint32_t srcShape[2] = {static_cast<uint32_t>(mfactor), static_cast<uint32_t>(nfactor)};

        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(
            reduceOutTmpTensor_, dgammaCalcTensor_, srcShape, false);

        UpdateCache(reduceCacheTensor_, reduceOutTmpTensor_, cacheID, td_->dgammaNfactorBlockAligned, nfactor);
    }

    __aicore__ inline void ComputeDgammaTmpMain(
        const LocalTensor<float>& dstTensor, const LocalTensor<T>& dyTensor, const LocalTensor<T>& xTensor,
        const LocalTensor<float>& rstdTensor, const int64_t rowSize, const int64_t colSize)
    {
        uint16_t outerLoopTimes = static_cast<uint16_t>(rowSize);
        uint16_t innerLoopTimes = ops::CeilDiv(colSize, static_cast<int64_t>(VL_FP32));
        uint32_t outerStride = td_->dgammaNfactorBlockAligned;
        uint32_t innerStride = VL_FP32;

        __local_mem__ float* dst = (__local_mem__ float*)dstTensor.GetPhyAddr();
        __local_mem__ T* x = (__local_mem__ T*)xTensor.GetPhyAddr();
        __local_mem__ T* dy = (__local_mem__ T*)dyTensor.GetPhyAddr();
        __local_mem__ float* rstd = (__local_mem__ float*)rstdTensor.GetPhyAddr();

        __VEC_SCOPE__
        {
            for (uint16_t i = 0; i < outerLoopTimes; ++i) {
                uint32_t count = static_cast<uint32_t>(colSize);
                AscendC::MicroAPI::RegTensor<float> rstdReg;
                DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                    rstdReg, (__local_mem__ float*)rstd + static_cast<uint32_t>(i));

                AscendC::MicroAPI::RegTensor<float> xReg;
                AscendC::MicroAPI::RegTensor<float> dyReg;
                AscendC::MicroAPI::MaskReg pMask;
                for (uint16_t j = 0; j < innerLoopTimes; ++j) {
                    pMask = AscendC::MicroAPI::UpdateMask<float>(count);
                    uint32_t offset = i * outerStride + j * innerStride;
                    LoadOneTensor<T>(xReg, x, pMask, offset);
                    Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(xReg, xReg, rstdReg, pMask);
                    LoadOneTensor<T>(dyReg, dy, pMask, offset);
                    Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(xReg, xReg, dyReg, pMask);
                    DataCopy((__local_mem__ float*)dst + offset, xReg, pMask);
                }
            }
        }
    }

    __aicore__ inline void ComputeDgammaTmpFold(
        const LocalTensor<float>& dstTensor, const LocalTensor<T>& dyTensor, const LocalTensor<T>& xTensor,
        const LocalTensor<float>& rstdTensor, const int64_t rowSize, const int64_t colSize)
    {
        uint16_t outerLoopTimes = static_cast<uint16_t>(rowSize);
        uint16_t innerLoopTimes = ops::CeilDiv(colSize, static_cast<int64_t>(VL_FP32));
        uint32_t outerStride = td_->dgammaNfactorBlockAligned;
        uint32_t innerStride = VL_FP32;

        __local_mem__ float* dst = (__local_mem__ float*)dstTensor.GetPhyAddr();
        __local_mem__ T* x = (__local_mem__ T*)xTensor.GetPhyAddr();
        __local_mem__ T* dy = (__local_mem__ T*)dyTensor.GetPhyAddr();
        __local_mem__ float* rstd = (__local_mem__ float*)rstdTensor.GetPhyAddr();

        __VEC_SCOPE__
        {
            for (uint16_t i = 0; i < outerLoopTimes; ++i) {
                uint32_t count = static_cast<uint32_t>(colSize);
                AscendC::MicroAPI::RegTensor<float> rstdReg;
                DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                    rstdReg, (__local_mem__ float*)rstd + static_cast<uint32_t>(i));

                AscendC::MicroAPI::RegTensor<float> xReg;
                AscendC::MicroAPI::RegTensor<float> dyReg;
                AscendC::MicroAPI::MaskReg pMask;
                for (uint16_t j = 0; j < innerLoopTimes; ++j) {
                    pMask = AscendC::MicroAPI::UpdateMask<float>(count);
                    uint32_t offset = i * outerStride + j * innerStride;
                    LoadOneTensor<T>(xReg, x, pMask, offset);
                    Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(xReg, xReg, rstdReg, pMask);
                    LoadOneTensor<T>(dyReg, dy, pMask, offset);
                    Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(xReg, xReg, dyReg, pMask);
                    LoadOneTensor<float>(dyReg, dst, pMask, offset);
                    Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(xReg, xReg, dyReg, pMask);
                    DataCopy((__local_mem__ float*)dst + offset, xReg, pMask);
                }
            }
        }
    }

    template <typename T1>
    __aicore__ inline void LoadOneTensor(
        MicroAPI::RegTensor<float>& dst, const __local_mem__ void* input, MicroAPI::MaskReg& preg, uint32_t offset)
    {
        if constexpr (!IsSameType<T1, float>::value) {
            MicroAPI::RegTensor<T1> xFp16;
            DataCopy<T1, MicroAPI::LoadDist::DIST_UNPACK_B16>(xFp16, (__local_mem__ T1*)(input) + offset);
            Cast<float, T1, castTraitB162B32>(dst, xFp16, preg);
        } else {
            DataCopy(dst, (__local_mem__ float*)(input) + offset);
        }
    }

    __aicore__ inline void CopyOutDgammaTmpStg0(int64_t offset, const int64_t curAInnerLen)
    {
        DataCopyExtParams copyOutParams;
        copyOutParams.blockCount = 1;
        copyOutParams.blockLen = curAInnerLen * sizeof(float);
        copyOutParams.srcStride = 0;
        copyOutParams.dstStride = 0;

        LocalTensor<float> dgammaTmp = dgammaTmpOutQueue_.template DeQue<float>();
        DataCopyPad<float, PaddingMode::Normal>(dgammaTmpGm_[offset], dgammaTmp, copyOutParams);
        dgammaTmpOutQueue_.FreeTensor(dgammaTmp);
    }

    __aicore__ inline void CopyInDyAndX(
        const int64_t gmOffset, const int64_t nburst, const int64_t burstLen, const int64_t dstStride,
        const int64_t srcStride)
    {
        DataCopyExtParams params;
        params.blockCount = nburst;
        params.blockLen = burstLen * sizeof(T);
        params.srcStride = srcStride * sizeof(T) - params.blockLen;
        params.dstStride = (dstStride - burstLen) * sizeof(T) / BLOCK_SIZE;

        DataCopyPadExtParams<T> padParams;
        padParams.isPad = false;

        LocalTensor<T> dyTensor = inQueueDy_.AllocTensor<T>();
        DataCopyPad<T, PaddingMode::Normal>(dyTensor, dyInGm_[gmOffset], params, padParams);
        inQueueDy_.EnQue(dyTensor);

        LocalTensor<T> xTensor = inQueueX_.AllocTensor<T>();
        DataCopyPad<T, PaddingMode::Normal>(xTensor, xInGm_[gmOffset], params, padParams);
        inQueueX_.EnQue(xTensor);
    }

    __aicore__ inline void CopyInRstd(const int64_t gmOffset, const int64_t burstLen)
    {
        DataCopyExtParams params;
        params.blockCount = 1;
        params.blockLen = burstLen * sizeof(float);

        DataCopyPadExtParams<float> padParams;
        padParams.isPad = false;

        LocalTensor<float> rstdTensor = inQueueRstd_.template AllocTensor<float>();
        DataCopyPad<float, PaddingMode::Normal>(rstdTensor, rstdInGm_[gmOffset], params, padParams);
        inQueueRstd_.EnQue(rstdTensor);
    }

    __aicore__ inline void UpdateCache(
        const LocalTensor<float>& dstTensor, const LocalTensor<float>& srcTensor, const int64_t cacheID,
        const int64_t stride, const int64_t count)
    {
        uint16_t outerLoopTimes = ops::CeilDiv(static_cast<uint32_t>(count), VL_FP32);
        uint16_t innerLoopTimes = cacheID;
        uint32_t outerLoopStride = VL_FP32;
        uint32_t innerLoopStride = stride;
        __VEC_SCOPE__
        {
            __local_mem__ float* dst = (__local_mem__ float*)dstTensor.GetPhyAddr();
            __local_mem__ float* cah = (__local_mem__ float*)dstTensor.GetPhyAddr() + cacheID * stride;
            __local_mem__ float* src = (__local_mem__ float*)srcTensor.GetPhyAddr();
            uint32_t sreg = static_cast<uint32_t>(count);
            AscendC::MicroAPI::RegTensor<float> aReg, bReg;
            AscendC::MicroAPI::MaskReg pMask;
            for (uint16_t i = 0; i < outerLoopTimes; ++i) {
                pMask = AscendC::MicroAPI::UpdateMask<float>(sreg);
                DataCopy(aReg, (__local_mem__ float*)src + static_cast<uint32_t>(i * outerLoopStride));
                for (uint16_t j = 0; j < innerLoopTimes; ++j) {
                    DataCopy(
                        bReg,
                        (__local_mem__ float*)dst + static_cast<uint32_t>(i * outerLoopStride + j * innerLoopStride));
                    Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(aReg, aReg, bReg, pMask);
                }
                DataCopy((__local_mem__ float*)cah + static_cast<uint32_t>(i * outerLoopStride), aReg, pMask);
            }
        }
    }

    __aicore__ inline void CopyUB2UB(
        const LocalTensor<float>& dstTensor, const LocalTensor<float>& srcTensor, const int64_t count)
    {
        DataCopy(
            dstTensor, srcTensor,
            ops::Aligned(static_cast<int64_t>(count), static_cast<int64_t>(BLOCK_SIZE / sizeof(float))));
    }

    // 核间累加
    __aicore__ inline void InitBufferStg1()
    {
        pipe_->Reset();
        int64_t aSizeIn = td_->dgammaAInnerAlignedStg1 * td_->dgammaUsedCoreNum * sizeof(float);
        pipe_->InitBuffer(dgammaTmpInQue_, DOUBLE_BUFFER, aSizeIn);

        pipe_->InitBuffer(dgammaOutQue_, DOUBLE_BUFFER, td_->dgammaAInnerAlignedStg1 * sizeof(float));
    }

    __aicore__ inline void ProcessStg1()
    {
        for (int64_t i = 0; i < td_->dgammaAOuterStg1; i++) {
            uint16_t curAInnerLen =
                i != (td_->dgammaAOuterStg1 - 1) ? td_->dgammaAInnerAlignedStg1 : td_->dgammaATailStg1;

            int64_t offset = i * td_->dgammaAInnerAlignedStg1;
            CopyInDgammaTmp(offset, curAInnerLen);

            LocalTensor<float> dgammaTmpIn = dgammaTmpInQue_.template DeQue<float>();
            LocalTensor<float> dgammaOut = dgammaOutQue_.AllocTensor<float>();

            uint32_t srcShape[2] = {
                static_cast<uint32_t>(td_->dgammaUsedCoreNum), static_cast<uint32_t>(td_->dgammaAInnerAlignedStg1)};

            ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(dgammaOut, dgammaTmpIn, srcShape, false);
            dgammaTmpInQue_.FreeTensor(dgammaTmpIn);
            dgammaOutQue_.EnQue(dgammaOut);
            CopyOutDgamma(offset, curAInnerLen);
        }
    }

    __aicore__ inline void CopyInDgammaTmp(int64_t offset, int64_t curALen)
    {
        DataCopyPadExtParams<float> dataCopyPadExtParams;
        dataCopyPadExtParams.isPad = false;
        dataCopyPadExtParams.leftPadding = 0;
        dataCopyPadExtParams.rightPadding = 0;
        dataCopyPadExtParams.paddingValue = 0;

        DataCopyExtParams copyInParams;
        copyInParams.blockCount = td_->dgammaUsedCoreNum;
        copyInParams.blockLen = curALen * sizeof(float);
        copyInParams.srcStride = (td_->dxTilingData.cols - curALen) * sizeof(float);
        copyInParams.dstStride = (td_->dgammaAInnerAlignedStg1 - curALen) * sizeof(float) / BLOCK_SIZE;

        LocalTensor<float> dgammaTmpTensor = dgammaTmpInQue_.AllocTensor<float>();
        DataCopyPad<float, PaddingMode::Normal>(
            dgammaTmpTensor, dgammaTmpGm_[offset], copyInParams, dataCopyPadExtParams);
        dgammaTmpInQue_.EnQue(dgammaTmpTensor);
    }

    __aicore__ inline void CopyOutDgamma(int64_t offset, uint32_t curAInnerLen)
    {
        DataCopyExtParams copyOutParams;
        copyOutParams.blockCount = 1;
        copyOutParams.blockLen = curAInnerLen * sizeof(float);
        copyOutParams.srcStride = 0;
        copyOutParams.dstStride = 0;

        LocalTensor<float> dgamma = dgammaOutQue_.DeQue<float>();
        DataCopyPad<float, PaddingMode::Normal>(dgammaGm_[offset], dgamma, copyOutParams);
        dgammaOutQue_.FreeTensor(dgamma);
    }

private:
    const RmsNormGradBigMTilingData* td_;
    TPipe* pipe_;

    int64_t blockIdx_ = 0;

    int64_t M = 0;

    int64_t Mloop = 0;
    int64_t Mtail = 0;
    int64_t MTotalLoop = 0;

    int64_t BasicBlockLoop = 0;
    int64_t MainFoldCount = 0;
    int64_t CacheBufferCount = 0;
    int64_t ResultCacheID = 0;

    // Global Tensor
    GlobalTensor<T> dyInGm_;
    GlobalTensor<T> xInGm_;
    GlobalTensor<float> rstdInGm_;
    GlobalTensor<float> dgammaTmpGm_;
    GlobalTensor<float> dgammaGm_;

    // Local Tensor
    LocalTensor<float> dgammaCalcTensor_;
    LocalTensor<float> reduceOutTmpTensor_;
    LocalTensor<float> reduceCacheTensor_;

    LocalTensor<float> dyMain_;
    LocalTensor<float> xMain_;
    LocalTensor<float> rstd_;

    // TQue
    TQue<QuePosition::VECIN, 1> inQueueDy_;
    TQue<QuePosition::VECIN, 1> inQueueX_;
    TQue<QuePosition::VECIN, 1> inQueueRstd_;
    TQue<QuePosition::VECOUT, 1> dgammaTmpOutQueue_;
    TQue<QuePosition::VECIN, 1> dgammaTmpInQue_;
    TQue<QuePosition::VECOUT, 1> dgammaOutQue_;

    TBuf<TPosition::VECCALC> dgammaCalcBuf_;
    TBuf<TPosition::VECCALC> reduceCacheBuffer_;
    TBuf<TPosition::VECCALC> reduceOutTmpBuffer_;
}; // RmsNormGradDgammaBigM

} // namespace RmsNormGrad
#endif // RMS_NORM_GRAD_DGAMMA_BIG_M_