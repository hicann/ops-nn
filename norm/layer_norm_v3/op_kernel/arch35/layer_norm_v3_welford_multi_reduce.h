/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file layer_norm_v3_welford_multi_reduce.h
 * \brief
 */

#ifndef LAYER_NORM_V3_WELFORD_MULTI_REDUCE_H
#define LAYER_NORM_V3_WELFORD_MULTI_REDUCE_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "layer_norm_v3_common.h"
#include "../../inc/platform.h"
#include "../../norm_common/reduce_common_regbase.h"

namespace LayerNormV3 {
using namespace AscendC;
using AscendC::MicroAPI::CreateMask;
using AscendC::MicroAPI::LoadDist;
using AscendC::MicroAPI::MaskPattern;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::StoreDist;
using AscendC::MicroAPI::UpdateMask;

template <typename T, typename U, typename M, bool IsOutRstd>
class LayerNormV3WelfordMultiReduce {
public:
    __aicore__ inline LayerNormV3WelfordMultiReduce(const LayerNormV3TilingDataWelfordMultiReduce* tilingData,
                                                    TPipe* pipeIn)
    {
        pipe_ = pipeIn;
        td_ = tilingData;
    }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean, GM_ADDR lastout)
    {
        if (GetBlockIdx() >= td_->numBlocks) {
            return;
        }

        currentBlockFactor_ = (GetBlockIdx() == td_->mainBlockCount) ? td_->tailBlockFactor : td_->mainBlockFactor;
        int64_t fmOffset = GetBlockIdx() * td_->mainBlockFactor * td_->n;
        int64_t paramOffset = GetBlockIdx() * td_->mainBlockFactor;

        xGm_.SetGlobalBuffer((__gm__ T*)x + fmOffset, currentBlockFactor_ * td_->n);
        yGm_.SetGlobalBuffer((__gm__ T*)y + fmOffset, currentBlockFactor_ * td_->n);
        meanGm_.SetGlobalBuffer((__gm__ M*)mean + paramOffset, currentBlockFactor_);
        lastoutGm_.SetGlobalBuffer((__gm__ M*)lastout + paramOffset, currentBlockFactor_);
        if (td_->nullptrGamma == 0) {
            gammaGm_.SetGlobalBuffer((__gm__ U*)gamma, td_->r0);
        }
        if (td_->nullptrBeta == 0) {
            betaGm_.SetGlobalBuffer((__gm__ U*)beta, td_->r0);
        }

        pipe_->InitBuffer(inQueueX_, BUFFER_NUM, td_->tileLength * sizeof(T));
        pipe_->InitBuffer(outQueueY_, BUFFER_NUM, td_->tileLength * sizeof(T));
        pipe_->InitBuffer(outQueueMean_, BUFFER_NUM, AGGREGATION_COUNT * sizeof(float));
        pipe_->InitBuffer(outQueueLastout_, BUFFER_NUM, AGGREGATION_COUNT * sizeof(float));
        pipe_->InitBuffer(welfordTempBuffer_, td_->welfordTempSize);
        pipe_->InitBuffer(apiTempBuffer_, td_->apiTempBufferSize);

        if (td_->cutR1OrR0 == 1) {
            // cutR1: gamma/beta TBuf resident, or packed when gammaPackedSize > 0
            if (td_->gammaPackedSize > 0) {
                if (td_->nullptrGamma == 0) {
                    pipe_->InitBuffer(gammaPackedTBuf_, td_->gammaPackedSize);
                }
                if (td_->nullptrBeta == 0) {
                    pipe_->InitBuffer(betaPackedTBuf_, td_->gammaPackedSize);
                }
            } else {
                if (td_->nullptrGamma == 0) {
                    pipe_->InitBuffer(gammaTBuf_, td_->r0 * sizeof(U));
                }
                if (td_->nullptrBeta == 0) {
                    pipe_->InitBuffer(betaTBuf_, td_->r0 * sizeof(U));
                }
            }
        } else {
            // cutR0: gamma/beta double buffer queue
            if (td_->nullptrGamma == 0) {
                pipe_->InitBuffer(inQueueGamma_, BUFFER_NUM, td_->tileLength * sizeof(U));
            }
            if (td_->nullptrBeta == 0) {
                pipe_->InitBuffer(inQueueBeta_, BUFFER_NUM, td_->tileLength * sizeof(U));
            }
        }
    }

    __aicore__ inline void Process()
    {
        if (GetBlockIdx() >= td_->numBlocks) {
            return;
        }

        ResetCache();
        paramAddr_ = 0;
        mean_ = welfordTempBuffer_.Get<float>();
        variance_ = mean_[td_->tileLength];
        if constexpr (IsOutRstd) {
            varianceTensor_ = variance_[td_->tileLength];
        } else {
            rstdTensor_ = variance_[td_->tileLength];
        }
        shared_ = apiTempBuffer_.Get<uint8_t>();

        // Preload gamma/beta for cutR1 mode
        if (td_->cutR1OrR0 == 1) {
            if (td_->gammaPackedSize == 0) {
                if (td_->nullptrGamma == 0) {
                    gammaResident_ = gammaTBuf_.Get<U>();
                    DataCopyExtParams cpParams;
                    cpParams.blockCount = 1;
                    cpParams.blockLen = td_->r0 * sizeof(U);
                    DataCopyPadExtParams<U> padParams;
                    padParams.isPad = false;
                    DataCopyPad(gammaResident_, gammaGm_, cpParams, padParams);
                }
                if (td_->nullptrBeta == 0) {
                    betaResident_ = betaTBuf_.Get<U>();
                    DataCopyExtParams cpParams2;
                    cpParams2.blockCount = 1;
                    cpParams2.blockLen = td_->r0 * sizeof(U);
                    DataCopyPadExtParams<U> padParams2;
                    padParams2.isPad = false;
                    DataCopyPad(betaResident_, betaGm_, cpParams2, padParams2);
                }
            }

            // Build packed gamma/beta: replicate r0 elements r1ComputeFactor times
            if (td_->r1ComputeFactor > 1 && td_->gammaPackedSize > 0) {
                int64_t r0Align = td_->r0Align;
                int64_t r1CF = td_->r1ComputeFactor;
                if (td_->nullptrGamma == 0) {
                    gammaPackedLocal_ = gammaPackedTBuf_.Get<U>();
                    for (int64_t i = 0; i < r1CF; i++) {
                        DataCopyExtParams cpOnce;
                        cpOnce.blockCount = 1;
                        cpOnce.blockLen = static_cast<uint16_t>(td_->r0 * sizeof(U));
                        cpOnce.srcStride = 0;
                        cpOnce.dstStride = 0;
                        DataCopyPadExtParams<U> padOnce;
                        padOnce.isPad = false;
                        padOnce.leftPadding = 0;
                        padOnce.rightPadding = 0;
                        padOnce.paddingValue = 0;
                        DataCopyPad(gammaPackedLocal_[i * r0Align], gammaGm_, cpOnce, padOnce);
                    }
                }
                if (td_->nullptrBeta == 0) {
                    betaPackedLocal_ = betaPackedTBuf_.Get<U>();
                    for (int64_t i = 0; i < r1CF; i++) {
                        DataCopyExtParams cpOnce2;
                        cpOnce2.blockCount = 1;
                        cpOnce2.blockLen = static_cast<uint16_t>(td_->r0 * sizeof(U));
                        cpOnce2.srcStride = 0;
                        cpOnce2.dstStride = 0;
                        DataCopyPadExtParams<U> padOnce2;
                        padOnce2.isPad = false;
                        padOnce2.leftPadding = 0;
                        padOnce2.rightPadding = 0;
                        padOnce2.paddingValue = 0;
                        DataCopyPad(betaPackedLocal_[i * r0Align], betaGm_, cpOnce2, padOnce2);
                    }
                }
            }
        }

        for (int64_t i = 0; i < currentBlockFactor_; ++i) {
            // Phase 1: Welford
            welfordCount_ = 0;
            WelfordInitialize(mean_, variance_, td_->tileLength);
            for (int64_t k = 0; k < td_->welfordUpdateTimes; k++) {
                int64_t offset = i * td_->n + k * td_->tileLength;
                ProcessWelfordUpdate(inQueueX_, xGm_, offset, td_->tileLength, td_->tileLength, welfordCount_, mean_,
                                     variance_, shared_);
            }
            if (td_->welfordUpdateTail > 0) {
                int64_t offset = i * td_->n + td_->welfordUpdateTimes * td_->tileLength;
                ProcessWelfordUpdate(inQueueX_, xGm_, offset, td_->welfordUpdateTail, td_->tileLength, welfordCount_,
                                     mean_, variance_, shared_);
            }

            WelfordFinalizePara para;
            para.rnLength = welfordCount_;
            para.abLength = td_->tileLength;
            para.headCount = welfordCount_;
            para.headCountLength = td_->welfordUpdateTail;
            para.tailCount = welfordCount_ - (td_->welfordUpdateTail > 0);
            para.tailCountLength = td_->tileLength - td_->welfordUpdateTail;
            para.abRec = 1.0f / static_cast<float>(td_->tileLength);
            para.rRec = 1.0f / static_cast<float>(td_->n);
            if constexpr (IsOutRstd) {
                WelfordFinalize<true>(meanTensor_[cacheCount_], varianceTensor_[cacheCount_], mean_, variance_, shared_,
                                      para);
            } else {
                WelfordFinalize<true>(meanTensor_[cacheCount_], lastoutTensor_[cacheCount_], mean_, variance_, shared_,
                                      para);
            }

            // Phase 2: Normalize
            if (td_->cutR1OrR0 == 1) {
                NormalizeCutR1(i);
            } else {
                NormalizeCutR0(i);
            }

            cacheCount_++;
            if (cacheCount_ >= AGGREGATION_COUNT) {
                RefreshCache<M>(cacheCount_, paramAddr_, meanTensor_, lastoutTensor_, outQueueMean_, outQueueLastout_,
                                meanGm_, lastoutGm_);
                ResetCache();
            }
        }

        if (cacheCount_ > 0) {
            RefreshCache<M>(cacheCount_, paramAddr_, meanTensor_, lastoutTensor_, outQueueMean_, outQueueLastout_,
                            meanGm_, lastoutGm_);
        }
    }

private:
    __aicore__ inline void ResetCache()
    {
        cacheCount_ = 0;
        meanTensor_ = outQueueMean_.template AllocTensor<float>();
        lastoutTensor_ = outQueueLastout_.template AllocTensor<float>();
    }

    __aicore__ inline void NormalizeCutR1(const int64_t rowIdx)
    {
        int64_t r0Aligned = td_->r0Align;

        __local_mem__ float* meanAddr = (__local_mem__ float*)meanTensor_.GetPhyAddr() + cacheCount_;
        __local_mem__ float* rstdAddr;

        if constexpr (IsOutRstd) {
            // variance is in varianceTensor_, compute rstd and store to lastoutTensor_ (for GM output)
            __local_mem__ float* varAddr = (__local_mem__ float*)varianceTensor_.GetPhyAddr() + cacheCount_;
            rstdAddr = (__local_mem__ float*)lastoutTensor_.GetPhyAddr() + cacheCount_;
            {
                __VEC_SCOPE__
                {
                    RegTensor<float> varReg;
                    RegTensor<float> rstdRegTmp;
                    MaskReg pregAll = CreateMask<float, MaskPattern::ALL>();
                    MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
                    DataCopy<float, LoadDist::DIST_BRC_B32>(varReg, varAddr);
                    NormCommon::ComputeRstdNewtonRaphsonReg(varReg, rstdRegTmp, pregAll, td_->epsilon);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(rstdAddr, rstdRegTmp, pregOne);
                }
            }
        } else {
            __local_mem__ float* varAddr = (__local_mem__ float*)lastoutTensor_.GetPhyAddr() + cacheCount_;
            rstdAddr = (__local_mem__ float*)rstdTensor_.GetPhyAddr() + cacheCount_;
            {
                __VEC_SCOPE__
                {
                    RegTensor<float> varReg;
                    RegTensor<float> rstdRegTmp;
                    MaskReg pregAll = CreateMask<float, MaskPattern::ALL>();
                    MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
                    DataCopy<float, LoadDist::DIST_BRC_B32>(varReg, varAddr);
                    NormCommon::ComputeRstdNewtonRaphsonReg(varReg, rstdRegTmp, pregAll, td_->epsilon);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(rstdAddr, rstdRegTmp, pregOne);
                }
            }
        }

        __local_mem__ U* gammaAddr;
        __local_mem__ U* betaAddr;
        if (td_->r1ComputeFactor > 1) {
            gammaAddr = (__local_mem__ U*)gammaPackedLocal_.GetPhyAddr();
            betaAddr = (__local_mem__ U*)betaPackedLocal_.GetPhyAddr();
        } else {
            gammaAddr = (__local_mem__ U*)gammaResident_.GetPhyAddr();
            betaAddr = (__local_mem__ U*)betaResident_.GetPhyAddr();
        }

        for (int64_t r1Loop = 0; r1Loop < td_->loopR1outer; r1Loop++) {
            int64_t r1Start = r1Loop * td_->r1Factor;
            int64_t curR1 = td_->r1Factor;
            if (r1Start + curR1 > td_->r1) {
                curR1 = td_->r1 - r1Start;
            }
            int64_t fmOffset = rowIdx * td_->n + r1Start * td_->r0;

            LocalTensor<T> xTensor = inQueueX_.template AllocTensor<T>();
            DataCopyPadExtParams<T> padInParams;
            padInParams.isPad = (td_->r0 != td_->r0Align);
            padInParams.leftPadding = 0;
            padInParams.rightPadding = 0;
            padInParams.paddingValue = 0;
            DataCopyExtParams copyInParams;
            copyInParams.blockCount = curR1;
            copyInParams.blockLen = td_->r0 * sizeof(T);
            copyInParams.srcStride = 0;
            copyInParams.dstStride = 0;
            DataCopyPad(xTensor, xGm_[fmOffset], copyInParams, padInParams);
            inQueueX_.EnQue(xTensor);
            xTensor = inQueueX_.template DeQue<T>();

            LocalTensor<T> yTensor = outQueueY_.template AllocTensor<T>();

            __local_mem__ T* xUbAddr = (__local_mem__ T*)xTensor.GetPhyAddr();
            __local_mem__ T* yUbAddr = (__local_mem__ T*)yTensor.GetPhyAddr();
            NormalizeCutR1VF(xUbAddr, yUbAddr, gammaAddr, betaAddr, meanAddr, rstdAddr, r0Aligned, curR1);

            inQueueX_.FreeTensor(xTensor);

            outQueueY_.EnQue(yTensor);
            yTensor = outQueueY_.template DeQue<T>();
            DataCopyExtParams copyOutParams;
            copyOutParams.blockCount = curR1;
            copyOutParams.blockLen = td_->r0 * sizeof(T);
            copyOutParams.srcStride = 0;
            copyOutParams.dstStride = 0;
            DataCopyPad(yGm_[fmOffset], yTensor, copyOutParams);
            outQueueY_.FreeTensor(yTensor);
        }
    }

    __aicore__ inline void NormalizeCutR1VF(__local_mem__ T* xAddr, __local_mem__ T* yOutAddr,
                                            __local_mem__ U* gammaAddr, __local_mem__ U* betaAddr,
                                            __local_mem__ float* meanAddr, __local_mem__ float* rstdAddr,
                                            int64_t r0Aligned, int64_t curR1)
    {
        int64_t r1ComputeFactor = td_->r1ComputeFactor;

        if (r1ComputeFactor <= 1) {
            NormalizeCutR1VFSingle(xAddr, yOutAddr, gammaAddr, betaAddr, meanAddr, rstdAddr, r0Aligned, curR1);
            return;
        }

        __local_mem__ U* gammaPackedAddr = (__local_mem__ U*)gammaPackedLocal_.GetPhyAddr();
        __local_mem__ U* betaPackedAddr = (__local_mem__ U*)betaPackedLocal_.GetPhyAddr();

        uint32_t packedLen = static_cast<uint32_t>(r1ComputeFactor * r0Aligned);
        int64_t mainLoops = curR1 / r1ComputeFactor;
        int64_t tailRows = curR1 - mainLoops * r1ComputeFactor;
        int64_t tailStart = mainLoops * r1ComputeFactor;
        uint32_t tailLen = static_cast<uint32_t>(tailRows * r0Aligned);

        // Main loop: full packed iterations
        if (mainLoops > 0) {
            __VEC_SCOPE__
            {
                RegTensor<float> meanReg;
                RegTensor<float> rstdReg;
                RegTensor<float> gamma;
                RegTensor<float> beta;
                RegTensor<float> xReg;
                RegTensor<float> yReg;
                MaskReg pregFull;

                DataCopy<float, LoadDist::DIST_BRC_B32>(meanReg, meanAddr);
                DataCopy<float, LoadDist::DIST_BRC_B32>(rstdReg, rstdAddr);

                uint32_t fullSreg = packedLen;
                pregFull = UpdateMask<float>(fullSreg);
                LoadTensorForDtype<U>(gamma, gammaPackedAddr, pregFull, 0);
                LoadTensorForDtype<U>(beta, betaPackedAddr, pregFull, 0);

                for (uint16_t outerIdx = 0; outerIdx < static_cast<uint16_t>(mainLoops); outerIdx++) {
                    LoadTensorForDtype<T>(xReg, xAddr, pregFull, outerIdx * packedLen);
                    Sub(yReg, xReg, meanReg, pregFull);
                    Mul(yReg, yReg, rstdReg, pregFull);
                    Mul(yReg, yReg, gamma, pregFull);
                    Add(yReg, yReg, beta, pregFull);
                    StoreTensorForDtype<T>(yOutAddr, yReg, pregFull, outerIdx * packedLen);
                }
            }
        }

        // Tail: remaining rows processed one-by-one
        if (tailRows > 0) {
            __VEC_SCOPE__
            {
                RegTensor<float> meanReg;
                RegTensor<float> rstdReg;
                RegTensor<float> gamma;
                RegTensor<float> beta;
                RegTensor<float> xReg;
                RegTensor<float> yReg;
                MaskReg pregTail;

                DataCopy<float, LoadDist::DIST_BRC_B32>(meanReg, meanAddr);
                DataCopy<float, LoadDist::DIST_BRC_B32>(rstdReg, rstdAddr);

                uint32_t fullSreg = tailLen;
                pregTail = UpdateMask<float>(fullSreg);
                LoadTensorForDtype<U>(gamma, gammaPackedAddr, pregTail, 0);
                LoadTensorForDtype<U>(beta, betaPackedAddr, pregTail, 0);

                LoadTensorForDtype<T>(xReg, xAddr, pregTail, tailStart * r0Aligned);
                Sub(yReg, xReg, meanReg, pregTail);
                Mul(yReg, yReg, rstdReg, pregTail);
                Mul(yReg, yReg, gamma, pregTail);
                Add(yReg, yReg, beta, pregTail);
                StoreTensorForDtype<T>(yOutAddr, yReg, pregTail, tailStart * r0Aligned);
            }
        }
    }

    __aicore__ inline void NormalizeCutR1VFSingle(__local_mem__ T* xAddr, __local_mem__ T* yOutAddr,
                                                  __local_mem__ U* gammaAddr, __local_mem__ U* betaAddr,
                                                  __local_mem__ float* meanAddr, __local_mem__ float* rstdAddr,
                                                  int64_t r0Aligned, int64_t curR1)
    {
        uint32_t r0Num = static_cast<uint32_t>(td_->r0);
        uint16_t loopCount = static_cast<uint16_t>((r0Num + VL_B32 - 1) / VL_B32);

        __VEC_SCOPE__
        {
            RegTensor<float> meanReg;
            RegTensor<float> rstdReg;
            RegTensor<float> gamma;
            RegTensor<float> beta;
            RegTensor<float> xReg;
            RegTensor<float> yReg;
            MaskReg pregLoop;

            DataCopy<float, LoadDist::DIST_BRC_B32>(meanReg, meanAddr);
            DataCopy<float, LoadDist::DIST_BRC_B32>(rstdReg, rstdAddr);

            uint32_t sreg = r0Num;
            for (uint16_t r = 0; r < loopCount; r++) {
                pregLoop = UpdateMask<float>(sreg);

                LoadTensorForDtype<U>(gamma, gammaAddr, pregLoop, r * VL_B32);
                LoadTensorForDtype<U>(beta, betaAddr, pregLoop, r * VL_B32);

                for (uint16_t r1Inner = 0; r1Inner < static_cast<uint16_t>(curR1); r1Inner++) {
                    // Load x
                    LoadTensorForDtype<T>(xReg, xAddr, pregLoop, r1Inner * r0Aligned + r * VL_B32);

                    // y = (x - mean) * rstd * gamma + beta
                    Sub(yReg, xReg, meanReg, pregLoop);
                    Mul(yReg, yReg, rstdReg, pregLoop);
                    Mul(yReg, yReg, gamma, pregLoop);
                    Add(yReg, yReg, beta, pregLoop);

                    // Store y
                    StoreTensorForDtype<T>(yOutAddr, yReg, pregLoop, r1Inner * r0Aligned + r * VL_B32);
                }
            }
        }
    }

    template <typename DType>
    __aicore__ inline void LoadTensorForDtype(RegTensor<float>& dst, __local_mem__ DType* src, MaskReg& preg,
                                              uint32_t offset)
    {
        if constexpr (IsSameType<DType, float>::value) {
            DataCopy<float, LoadDist::DIST_NORM>(dst, src + offset);
        } else {
            RegTensor<DType> tmp;
            DataCopy<DType, LoadDist::DIST_UNPACK_B16>(tmp, src + offset);
            Cast<float, DType, castTraitB162B32>(dst, tmp, preg);
        }
    }

    template <typename DType>
    __aicore__ inline void StoreTensorForDtype(__local_mem__ DType* dst, RegTensor<float>& src, MaskReg& preg,
                                               uint32_t offset)
    {
        if constexpr (IsSameType<DType, float>::value) {
            DataCopy<DType, StoreDist::DIST_NORM>(dst + offset, src, preg);
        } else {
            RegTensor<DType> tmp;
            Cast<DType, float, castTraitB322B16>(tmp, src, preg);
            DataCopy<DType, StoreDist::DIST_PACK_B32>(dst + offset, tmp, preg);
        }
    }

    __aicore__ inline void NormalizeCutR0(const int64_t rowIdx)
    {
        // gamma/beta loaded per segment from GM
        for (int64_t r1Idx = 0; r1Idx < td_->r1; r1Idx++) {
            for (int64_t r0Loop = 0; r0Loop < td_->loopR0outer; r0Loop++) {
                int64_t r0Offset = r0Loop * td_->r0Factor;
                int64_t r0Len = td_->r0Factor;
                if (r0Offset + r0Len > td_->r0) {
                    r0Len = td_->r0 - r0Offset;
                }
                int64_t fmOffset = rowIdx * td_->n + r1Idx * td_->r0 + r0Offset;
                ProcessNormalizeTile(fmOffset, r0Offset, r0Len);
            }
        }
    }

    __aicore__ inline void ProcessNormalizeTile(const int64_t fmOffset, const int64_t gammaOffset,
                                                const int64_t elemCnt)
    {
        LocalTensor<T> xTensor = inQueueX_.template AllocTensor<T>();
        CopyIn(xTensor, xGm_[fmOffset], elemCnt);
        inQueueX_.EnQue(xTensor);
        xTensor = inQueueX_.template DeQue<T>();

        LocalTensor<U> gammaTensor;
        LocalTensor<U> betaTensor;

        if (td_->nullptrGamma == 0) {
            gammaTensor = inQueueGamma_.template AllocTensor<U>();
            CopyIn(gammaTensor, gammaGm_[gammaOffset], elemCnt);
            inQueueGamma_.EnQue(gammaTensor);
            gammaTensor = inQueueGamma_.template DeQue<U>();
        }
        if (td_->nullptrBeta == 0) {
            betaTensor = inQueueBeta_.template AllocTensor<U>();
            CopyIn(betaTensor, betaGm_[gammaOffset], elemCnt);
            inQueueBeta_.EnQue(betaTensor);
            betaTensor = inQueueBeta_.template DeQue<U>();
        }

        LocalTensor<T> yTensor = outQueueY_.template AllocTensor<T>();

        NormalizePara normPara;
        normPara.aLength = 1;
        normPara.rLength = elemCnt;
        normPara.rLengthWithPadding = elemCnt;
        if constexpr (IsOutRstd) {
            Normalize<U, T, false>(yTensor, lastoutTensor_[cacheCount_], meanTensor_[cacheCount_],
                                   varianceTensor_[cacheCount_], xTensor, gammaTensor, betaTensor, shared_,
                                   td_->epsilon, normPara);
        } else {
            Normalize<U, T, false>(yTensor, rstdTensor_[cacheCount_], meanTensor_[cacheCount_],
                                   lastoutTensor_[cacheCount_], xTensor, gammaTensor, betaTensor, shared_, td_->epsilon,
                                   normPara);
        }

        inQueueX_.FreeTensor(xTensor);
        if (td_->nullptrGamma == 0) {
            inQueueGamma_.FreeTensor(gammaTensor);
        }
        if (td_->nullptrBeta == 0) {
            inQueueBeta_.FreeTensor(betaTensor);
        }

        outQueueY_.EnQue(yTensor);
        yTensor = outQueueY_.template DeQue<T>();
        CopyOut(yGm_[fmOffset], yTensor, elemCnt);
        outQueueY_.FreeTensor(yTensor);
    }

private:
    TPipe* pipe_;
    const LayerNormV3TilingDataWelfordMultiReduce* __restrict td_;

    // Constants
    constexpr static int64_t BUFFER_NUM = 2;
    constexpr static int64_t AGGREGATION_COUNT = 256;

    // TQue
    TQue<QuePosition::VECIN, 1> inQueueX_;
    TQue<QuePosition::VECIN, 1> inQueueGamma_;
    TQue<QuePosition::VECIN, 1> inQueueBeta_;

    TQue<QuePosition::VECOUT, 1> outQueueY_;
    TQue<QuePosition::VECOUT, 1> outQueueMean_;
    TQue<QuePosition::VECOUT, 1> outQueueLastout_;

    // TBuf
    TBuf<TPosition::VECCALC> welfordTempBuffer_;
    TBuf<TPosition::VECCALC> apiTempBuffer_;
    TBuf<TPosition::VECCALC> gammaTBuf_;
    TBuf<TPosition::VECCALC> betaTBuf_;
    TBuf<TPosition::VECCALC> gammaPackedTBuf_;
    TBuf<TPosition::VECCALC> betaPackedTBuf_;

    // Global Tensor
    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    GlobalTensor<M> meanGm_;
    GlobalTensor<M> lastoutGm_;
    GlobalTensor<U> gammaGm_;
    GlobalTensor<U> betaGm_;

    // Local Tensor
    LocalTensor<float> meanTensor_;
    LocalTensor<float> lastoutTensor_;
    LocalTensor<float> varianceTensor_;
    LocalTensor<float> rstdTensor_;
    LocalTensor<float> mean_;
    LocalTensor<float> variance_;
    LocalTensor<uint8_t> shared_;
    LocalTensor<U> gammaResident_;
    LocalTensor<U> betaResident_;
    LocalTensor<U> gammaPackedLocal_;
    LocalTensor<U> betaPackedLocal_;

    // Params
    int64_t currentBlockFactor_ = 0;
    int64_t cacheCount_ = 0;
    int64_t paramAddr_ = 0;
    int64_t welfordCount_ = 0;
};

} // namespace LayerNormV3

#endif // LAYER_NORM_V3_WELFORD_MULTI_REDUCE_H