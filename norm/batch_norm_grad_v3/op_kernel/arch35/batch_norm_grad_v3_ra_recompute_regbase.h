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
 * \file batch_norm_grad_v3_ra_recompute_regbase.h
 * \brief
 */

#ifndef __BATCH_NORM_GRAD_V3_RA_RECOMPUTE_REGBASE_H__
#define __BATCH_NORM_GRAD_V3_RA_RECOMPUTE_REGBASE_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "batch_norm_grad_v3_common.h"

namespace BatchNormGradV3 {
using namespace AscendC;
using AscendC::MicroAPI::LocalMemBar;
using AscendC::MicroAPI::MemType;

template <typename DY_TYPE, typename WEIGHT_TYPE, int BUFFER_NUM = 1>
class BatchNormGradV3RARecompute {
public:
    __aicore__ inline BatchNormGradV3RARecompute(TPipe* pipe)
    {
        pipe_ = pipe;
    }

    __aicore__ inline void Init(
        __gm__ uint8_t* dy, __gm__ uint8_t* x, __gm__ uint8_t* mean, __gm__ uint8_t* rstd, __gm__ uint8_t* gamma,
        __gm__ uint8_t* dx, __gm__ uint8_t* dgamma, __gm__ uint8_t* dbeta, __gm__ uint8_t* workspace,
        const BatchNormGradV3RARecomputeTilingData* tilingData)
    {
        const BatchNormGradV3BaseTilingData& baseTilingData = tilingData->baseTilingData;
        rDim_ = baseTilingData.r1Dim;
        aDim_ = baseTilingData.aDim;
        blockDim = tilingData->blockDim;
        mainBlockCount = tilingData->mainBlockCount;
        mainBlockFactor = tilingData->mainBlockFactor;
        tailBlockFactor = tilingData->tailBlockFactor;
        blockFactor = GetBlockIdx() >= mainBlockCount ? tailBlockFactor : mainBlockFactor;

        aLoopFactor = tilingData->aLoopFactor;
        aLoopFactorAligned = tilingData->aLoopFactorAligned;
        aLoopTimes = blockFactor / aLoopFactor;
        aLoopTail = blockFactor - aLoopTimes * aLoopFactor;
        rLoopFactor = tilingData->rLoopFactor;
        rLoopTimes = tilingData->rLoopTimes;
        rLoopTail = tilingData->rLoopTail;

        binaryFoldPoint = tilingData->binaryFoldPoint;
        binaryBlockCount = tilingData->binaryBlockCount;
        binaryTailBlock = tilingData->binaryTailBlock;
        cacheBufferCount = tilingData->cacheBufferCount;
        reciprocal = tilingData->reciprocal;
        resultCacheId = GetCacheID(binaryFoldPoint - 1);

        // init gm tensor
        int64_t offset = tilingData->mainBlockFactor * GetBlockIdx();
        dyGm_.SetGlobalBuffer((__gm__ DY_TYPE*)dy + offset);
        xGm_.SetGlobalBuffer((__gm__ DY_TYPE*)x + offset);
        meanGm_.SetGlobalBuffer((__gm__ float*)mean + offset);
        rstdGm_.SetGlobalBuffer((__gm__ float*)rstd + offset);
        gammaGm_.SetGlobalBuffer((__gm__ WEIGHT_TYPE*)gamma + offset);
        dxGm_.SetGlobalBuffer((__gm__ DY_TYPE*)dx + offset);
        dgammaGm_.SetGlobalBuffer((__gm__ WEIGHT_TYPE*)dgamma + offset);
        dbetaGm_.SetGlobalBuffer((__gm__ WEIGHT_TYPE*)dbeta + offset);

        // init queue
        int64_t feat_bytes = rLoopFactor * aLoopFactorAligned * sizeof(float);
        int64_t para_bytes = aLoopFactorAligned * sizeof(float);
        feat_bytes = RoundUpOneBlock(feat_bytes);
        para_bytes = RoundUpOneBlock(para_bytes);
        pipe_->InitBuffer(dyInQue_, BUFFER_NUM, feat_bytes);
        pipe_->InitBuffer(xInQue_, BUFFER_NUM, feat_bytes);
        pipe_->InitBuffer(meanInQue_, BUFFER_NUM, para_bytes);
        pipe_->InitBuffer(rstdInQue_, BUFFER_NUM, para_bytes);
        pipe_->InitBuffer(gammaInQue_, BUFFER_NUM, para_bytes);
        pipe_->InitBuffer(dxOutQue_, BUFFER_NUM, feat_bytes);
        pipe_->InitBuffer(dbetaOutQue_, BUFFER_NUM, para_bytes * cacheBufferCount);
        pipe_->InitBuffer(dgammaOutQue_, BUFFER_NUM, para_bytes * cacheBufferCount);
    }

    __aicore__ inline void Process()
    {
        if (GetBlockIdx() >= blockDim) {
            return;
        }

        for (int64_t i = 0; i < aLoopTimes; ++i) {
            ProcessOneRAFactor(i * aLoopFactor, aLoopFactor);
        }
        if (aLoopTail > 0) {
            ProcessOneRAFactor(aLoopTimes * aLoopFactor, aLoopTail);
        }
    }

public:
    __aicore__ inline void ProcessMainBlock(int64_t baseOffset, int64_t binaryLoopIdx, int64_t rLength, int64_t aLength)
    {
        int64_t offset = baseOffset + binaryLoopIdx * rLoopFactor * aDim_;
        dyMain = dyInQue_.template AllocTensor<float>();
        xMain = xInQue_.template AllocTensor<float>();

        dyFold = dyInQue_.template AllocTensor<float>();
        LocalTensor<DY_TYPE> dyFoldType = dyFold.template ReinterpretCast<DY_TYPE>();
        CopyIn(dyFoldType, dyGm_[offset], rLength, aLength, aLoopFactorAligned, aDim_);
        dyInQue_.EnQue(dyFold);
        dyFold = dyInQue_.template DeQue<float>();

        xFold = xInQue_.template AllocTensor<float>();
        LocalTensor<DY_TYPE> xFoldType = xFold.template ReinterpretCast<DY_TYPE>();
        CopyIn(xFoldType, xGm_[offset], rLength, aLength, aLoopFactorAligned, aDim_);
        xInQue_.EnQue(xFold);
        xFold = xInQue_.template DeQue<float>();

        meanTensor = meanInQue_.template AllocTensor<float>();
        CopyIn(meanTensor, meanGm_[baseOffset], aLength);
        meanInQue_.EnQue(meanTensor);
        meanTensor = meanInQue_.template DeQue<float>();

        rstdTensor = rstdInQue_.template AllocTensor<float>();
        CopyIn(rstdTensor, rstdGm_[baseOffset], aLength);
        rstdInQue_.EnQue(rstdTensor);
        rstdTensor = rstdInQue_.template DeQue<float>();

        uint16_t outerLoopTimes = CeilDiv(aLength, VL_FP32);
        uint16_t innerLoopTimes = rLength;
        uint16_t outerLoopStride = VL_FP32;
        uint16_t innerLoopStride = aLoopFactorAligned;
        uint32_t factor = aLength;
        __VEC_SCOPE__
        {
            __local_mem__ float* xMainAddr = (__local_mem__ float*)xMain.GetPhyAddr();
            __local_mem__ float* dyMainAddr = (__local_mem__ float*)dyMain.GetPhyAddr();
            __local_mem__ float* xFoldAddr = (__local_mem__ float*)xFold.GetPhyAddr();
            __local_mem__ float* dyFoldAddr = (__local_mem__ float*)dyFold.GetPhyAddr();
            __local_mem__ float* meanAddr = (__local_mem__ float*)meanTensor.GetPhyAddr();
            __local_mem__ float* rstdAddr = (__local_mem__ float*)rstdTensor.GetPhyAddr();
            AscendC::MicroAPI::MaskReg pMask;
            uint32_t count = factor;
            AscendC::MicroAPI::RegTensor<float> xMainReg, dyMainReg, meanReg, rstdReg;
            for (uint16_t i = 0; i < outerLoopTimes; ++i) {
                pMask = AscendC::MicroAPI::UpdateMask<float>(count);
                LoadOneTensor<float>(meanAddr, meanReg, pMask, i * outerLoopStride);
                LoadOneTensor<float>(rstdAddr, rstdReg, pMask, i * outerLoopStride);
                for (uint16_t j = 0; j < innerLoopTimes; ++j) {
                    uint32_t offset = i * outerLoopStride + j * innerLoopStride;
                    LoadOneTensor<DY_TYPE>(xFoldAddr, xMainReg, pMask, offset);
                    LoadOneTensor<DY_TYPE>(dyFoldAddr, dyMainReg, pMask, offset);
                    AscendC::MicroAPI::Sub<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
                        xMainReg, xMainReg, meanReg, pMask);
                    AscendC::MicroAPI::Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
                        xMainReg, xMainReg, dyMainReg, pMask);
                    AscendC::MicroAPI::Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
                        xMainReg, xMainReg, rstdReg, pMask);
                    StoreOneTensor<float>(xMainAddr, xMainReg, pMask, offset);
                    StoreOneTensor<float>(dyMainAddr, dyMainReg, pMask, offset);
                }
            }
        }
        dyInQue_.FreeTensor(dyFold);
        xInQue_.FreeTensor(xFold);
    }

    __aicore__ inline void ProcessFoldBlock(int64_t baseOffset, int64_t binaryLoopIdx, int64_t rLength, int64_t aLength)
    {
        int64_t offset = baseOffset + (binaryLoopIdx + binaryFoldPoint) * rLoopFactor * aDim_;
        dyFold = dyInQue_.template AllocTensor<float>();
        LocalTensor<DY_TYPE> dyFoldType = dyFold.template ReinterpretCast<DY_TYPE>();
        CopyIn(dyFoldType, dyGm_[offset], rLength, aLength, aLoopFactorAligned, aDim_);
        dyInQue_.EnQue(dyFold);
        dyFold = dyInQue_.template DeQue<float>();

        xFold = xInQue_.template AllocTensor<float>();
        LocalTensor<DY_TYPE> xFoldType = xFold.template ReinterpretCast<DY_TYPE>();
        CopyIn(xFoldType, xGm_[offset], rLength, aLength, aLoopFactorAligned, aDim_);
        xInQue_.EnQue(xFold);
        xFold = xInQue_.template DeQue<float>();

        uint16_t outerLoopTimes = CeilDiv(aLength, VL_FP32);
        uint16_t innerLoopTimes = rLength;
        uint16_t outerLoopStride = VL_FP32;
        uint16_t innerLoopStride = aLoopFactorAligned;
        uint32_t factor = aLength;
        __VEC_SCOPE__
        {
            __local_mem__ float* dyMainAddr = (__local_mem__ float*)dyMain.GetPhyAddr();
            __local_mem__ float* dyFoldAddr = (__local_mem__ float*)dyFold.GetPhyAddr();
            __local_mem__ float* xMainAddr = (__local_mem__ float*)xMain.GetPhyAddr();
            __local_mem__ float* xFoldAddr = (__local_mem__ float*)xFold.GetPhyAddr();
            __local_mem__ float* meanAddr = (__local_mem__ float*)meanTensor.GetPhyAddr();
            __local_mem__ float* rstdAddr = (__local_mem__ float*)rstdTensor.GetPhyAddr();
            AscendC::MicroAPI::MaskReg pMask;
            uint32_t count = factor;
            AscendC::MicroAPI::RegTensor<float> dyMainReg, dyFoldReg, xMainReg, xFoldReg, meanReg, rstdReg;
            for (uint16_t i = 0; i < outerLoopTimes; ++i) {
                pMask = AscendC::MicroAPI::UpdateMask<float>(count);
                LoadOneTensor<float>(meanAddr, meanReg, pMask, i * outerLoopStride);
                LoadOneTensor<float>(rstdAddr, rstdReg, pMask, i * outerLoopStride);
                for (uint16_t j = 0; j < innerLoopTimes; ++j) {
                    uint32_t offset = i * outerLoopStride + j * innerLoopStride;
                    LoadOneTensor<float>(dyMainAddr, dyMainReg, pMask, offset);
                    LoadOneTensor<DY_TYPE>(dyFoldAddr, dyFoldReg, pMask, offset);
                    AscendC::MicroAPI::Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
                        dyMainReg, dyMainReg, dyFoldReg, pMask);
                    StoreOneTensor<float>(dyMainAddr, dyMainReg, pMask, offset);
                    LoadOneTensor<float>(xMainAddr, xMainReg, pMask, offset);
                    LoadOneTensor<DY_TYPE>(xFoldAddr, xFoldReg, pMask, offset);
                    AscendC::MicroAPI::Sub<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
                        xFoldReg, xFoldReg, meanReg, pMask);
                    AscendC::MicroAPI::Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
                        xFoldReg, xFoldReg, dyFoldReg, pMask);
                    AscendC::MicroAPI::Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
                        xFoldReg, xFoldReg, rstdReg, pMask);
                    AscendC::MicroAPI::Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
                        xMainReg, xMainReg, xFoldReg, pMask);
                    StoreOneTensor<float>(xMainAddr, xMainReg, pMask, offset);
                }
            }
        }
        dyInQue_.FreeTensor(dyFold);
        xInQue_.FreeTensor(xFold);
    }

    __aicore__ inline void ProcessSummation(int64_t blkIdx, int64_t rLength, int64_t aLength)
    {
        int64_t cacheId = GetCacheID(blkIdx);
        meanInQue_.FreeTensor(meanTensor);
        rstdInQue_.FreeTensor(rstdTensor);
        uint16_t outerLoopTimes = CeilDiv(aLength, VL_FP32);
        uint16_t cacheIdLoop = cacheId;
        uint16_t outerLoopStride = VL_FP32;
        uint16_t innerLoopStride = aLoopFactorAligned;
        uint32_t factor = aLength;
        __VEC_SCOPE__
        {
            __local_mem__ float* dbetaAddr = (__local_mem__ float*)dbetaTensor.GetPhyAddr();
            __local_mem__ float* dgammaAddr = (__local_mem__ float*)dgammaTensor.GetPhyAddr();
            __local_mem__ float* dyMainAddr = (__local_mem__ float*)dyMain.GetPhyAddr();
            __local_mem__ float* xMainAddr = (__local_mem__ float*)xMain.GetPhyAddr();
            AscendC::MicroAPI::MaskReg pMask;
            uint32_t count = factor;
            AscendC::MicroAPI::RegTensor<float> dbetaReg, dgammaReg;
            AscendC::MicroAPI::RegTensor<float> dbetaCacheReg, dgammaCacheReg;
            AscendC::MicroAPI::RegTensor<float> dyMainReg, xMainReg;
            AscendC::MicroAPI::RegTensor<float> aReg, bReg, cReg, dReg;
            for (uint16_t i = 0; i < outerLoopTimes; ++i) {
                pMask = AscendC::MicroAPI::UpdateMask<float>(count);
                uint16_t recursionTimes = 5;
                uint16_t loopTimes = 64;
                for (uint16_t k = 0; k < recursionTimes; ++k) {
                    loopTimes = loopTimes >> 1;
                    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
                    for (uint16_t j = 0; j < loopTimes; ++j) {
                        LoadTwoTensorSum(
                            dyMainAddr, dyMainReg, pMask, i * outerLoopStride + j * innerLoopStride,
                            i * outerLoopStride + (j + loopTimes) * innerLoopStride);
                        StoreOneTensor<float>(dyMainAddr, dyMainReg, pMask, i * outerLoopStride + j * innerLoopStride);
                        LoadTwoTensorSum(
                            xMainAddr, xMainReg, pMask, i * outerLoopStride + j * innerLoopStride,
                            i * outerLoopStride + (j + loopTimes) * innerLoopStride);
                        StoreOneTensor<float>(xMainAddr, xMainReg, pMask, i * outerLoopStride + j * innerLoopStride);
                    }
                }
                {
                    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
                    LoadTwoTensorSum(
                        dyMainAddr, dbetaReg, pMask, i * outerLoopStride, i * outerLoopStride + innerLoopStride);
                    LoadTwoTensorSum(
                        xMainAddr, dgammaReg, pMask, i * outerLoopStride, i * outerLoopStride + innerLoopStride);
                }

                for (uint16_t j = 0; j < cacheIdLoop; ++j) { // update cache
                    uint32_t offset = i * outerLoopStride + j * innerLoopStride;
                    LoadOneTensor<float>(dbetaAddr, dbetaCacheReg, pMask, offset);
                    LoadOneTensor<float>(dgammaAddr, dgammaCacheReg, pMask, offset);
                    AscendC::MicroAPI::Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
                        dbetaReg, dbetaReg, dbetaCacheReg, pMask);
                    AscendC::MicroAPI::Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
                        dgammaReg, dgammaReg, dgammaCacheReg, pMask);
                }
                StoreOneTensor<float>(dbetaAddr, dbetaReg, pMask, i * outerLoopStride + cacheIdLoop * innerLoopStride);
                StoreOneTensor<float>(
                    dgammaAddr, dgammaReg, pMask, i * outerLoopStride + cacheIdLoop * innerLoopStride);
            }
        }
        dyInQue_.FreeTensor(dyMain);
        xInQue_.FreeTensor(xMain);
    }

    __aicore__ inline void ProcessDx(int64_t baseOffset, int64_t blkIdx, int64_t rLength, int64_t aLength)
    {
        // ProcessDx
        int64_t offset = baseOffset + blkIdx * rLoopFactor * aDim_;
        dyTensor = dyInQue_.template AllocTensor<DY_TYPE>();
        CopyIn(dyTensor, dyGm_[offset], rLength, aLength, aLoopFactorAligned, aDim_);
        dyInQue_.EnQue(dyTensor);
        dyTensor = dyInQue_.template DeQue<DY_TYPE>();
        xTensor = xInQue_.template AllocTensor<DY_TYPE>();
        CopyIn(xTensor, xGm_[offset], rLength, aLength, aLoopFactorAligned, aDim_);
        xInQue_.EnQue(xTensor);
        xTensor = xInQue_.template DeQue<DY_TYPE>();

        dxTensor = dxOutQue_.template AllocTensor<DY_TYPE>();
        uint16_t outerLoopTimes = CeilDiv(aLength, VL_FP32);
        uint16_t innerLoopTimes = rLength;
        uint16_t outerLoopStride = VL_FP32;
        uint16_t innerLoopStride = aLoopFactorAligned;
        uint32_t factor = aLength;
        uint32_t gammaBetaOffset = resultCacheId * aLoopFactorAligned;
        __VEC_SCOPE__
        {
            __local_mem__ float* dyAddr = (__local_mem__ float*)dyTensor.GetPhyAddr();
            __local_mem__ float* xAddr = (__local_mem__ float*)xTensor.GetPhyAddr();
            __local_mem__ float* meanAddr = (__local_mem__ float*)meanTensor.GetPhyAddr();
            __local_mem__ float* rstdAddr = (__local_mem__ float*)rstdTensor.GetPhyAddr();
            __local_mem__ float* gammaAddr = (__local_mem__ float*)gammaTensor.GetPhyAddr();
            __local_mem__ float* dxAddr = (__local_mem__ float*)dxTensor.GetPhyAddr();
            __local_mem__ float* dbetaAddr = (__local_mem__ float*)dbetaTensor.GetPhyAddr() + gammaBetaOffset;
            __local_mem__ float* dgammaAddr = (__local_mem__ float*)dgammaTensor.GetPhyAddr() + gammaBetaOffset;
            AscendC::MicroAPI::MaskReg pMask;
            uint32_t count = factor;
            AscendC::MicroAPI::RegTensor<float> dyReg, xReg, meanReg, rstdReg, gammaReg, dxReg, dbetaReg, dgammaReg;
            for (uint16_t i = 0; i < outerLoopTimes; ++i) {
                pMask = AscendC::MicroAPI::UpdateMask<float>(count);
                LoadOneTensor<float>(meanAddr, meanReg, pMask, i * outerLoopStride);
                LoadOneTensor<float>(rstdAddr, rstdReg, pMask, i * outerLoopStride);
                LoadOneTensor<WEIGHT_TYPE>(gammaAddr, gammaReg, pMask, i * outerLoopStride);
                LoadOneTensor<float>(dbetaAddr, dbetaReg, pMask, i * outerLoopStride);
                LoadOneTensor<float>(dgammaAddr, dgammaReg, pMask, i * outerLoopStride);
                for (uint16_t j = 0; j < innerLoopTimes; ++j) {
                    uint32_t offset = i * outerLoopStride + j * innerLoopStride;
                    LoadOneTensor<DY_TYPE>(dyAddr, dyReg, pMask, offset);
                    LoadOneTensor<DY_TYPE>(xAddr, xReg, pMask, offset);
                    AscendC::MicroAPI::Sub<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
                        xReg, xReg, meanReg, pMask);
                    AscendC::MicroAPI::Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
                        xReg, xReg, rstdReg, pMask);
                    AscendC::MicroAPI::Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
                        xReg, xReg, dgammaReg, pMask);
                    AscendC::MicroAPI::Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
                        xReg, xReg, dbetaReg, pMask);
                    AscendC::MicroAPI::Muls<float, float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
                        xReg, xReg, reciprocal, pMask);
                    AscendC::MicroAPI::Sub<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(dyReg, dyReg, xReg, pMask);
                    AscendC::MicroAPI::Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
                        dxReg, rstdReg, gammaReg, pMask);
                    AscendC::MicroAPI::Mul<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(
                        dxReg, dxReg, dyReg, pMask);
                    StoreOneTensor<DY_TYPE>(dxAddr, dxReg, pMask, offset);
                }
            }
        }
        dyInQue_.FreeTensor(dyTensor);
        xInQue_.FreeTensor(xTensor);
        // Copy Out dxTensor
        dxOutQue_.EnQue(dxTensor);
        dxTensor = dxOutQue_.template DeQue<DY_TYPE>();
        CopyOut(dxGm_[offset], dxTensor, rLength, aLength, aDim_, aLoopFactorAligned);
        dxOutQue_.FreeTensor(dxTensor);
    }

    __aicore__ inline void ProcessOneRAFactor(int64_t gmOffset, int64_t aLength)
    {
        // ProcessOneRAFactor
        dbetaTensor = dbetaOutQue_.template AllocTensor<float>();
        dgammaTensor = dgammaOutQue_.template AllocTensor<float>();

        int64_t hasBinaryTailBlock = binaryTailBlock > 0 ? 1 : 0;
        int64_t loop1 = binaryBlockCount - binaryFoldPoint - hasBinaryTailBlock;
        for (int64_t binaryIdx = 0; binaryIdx < loop1; ++binaryIdx) {
            ProcessMainBlock(gmOffset, binaryIdx, rLoopFactor, aLength);
            ProcessFoldBlock(gmOffset, binaryIdx, rLoopFactor, aLength);
            ProcessSummation(binaryIdx, rLoopFactor, aLength);
        }
        int64_t loop2 = hasBinaryTailBlock;
        for (int64_t binaryIdx = 0; binaryIdx < loop2; ++binaryIdx) {
            ProcessMainBlock(gmOffset, loop1, rLoopFactor, aLength);
            ProcessFoldBlock(gmOffset, loop1, binaryTailBlock, aLength);
            ProcessSummation(loop1, rLoopFactor, aLength);
        }
        for (int64_t binaryIdx = loop1 + loop2; binaryIdx < binaryFoldPoint; ++binaryIdx) {
            ProcessMainBlock(gmOffset, binaryIdx, rLoopFactor, aLength);
            ProcessSummation(binaryIdx, rLoopFactor, aLength);
        }

        // Copy In gamma
        gammaTensor = gammaInQue_.template AllocTensor<WEIGHT_TYPE>();
        CopyIn(gammaTensor, gammaGm_[gmOffset], aLength);
        gammaInQue_.EnQue(gammaTensor);
        gammaTensor = gammaInQue_.template DeQue<WEIGHT_TYPE>();

        // Copy In mean
        meanTensor = meanInQue_.template AllocTensor<float>();
        CopyIn(meanTensor, meanGm_[gmOffset], aLength);
        meanInQue_.EnQue(meanTensor);
        meanTensor = meanInQue_.template DeQue<float>();

        // Copy In rstd
        rstdTensor = rstdInQue_.template AllocTensor<float>();
        CopyIn(rstdTensor, rstdGm_[gmOffset], aLength);
        rstdInQue_.EnQue(rstdTensor);
        rstdTensor = rstdInQue_.template DeQue<float>();

        for (int64_t i = 0; i < rLoopTimes; i++) {
            ProcessDx(gmOffset, i, rLoopFactor, aLength);
        }
        if (rLoopTail > 0) {
            ProcessDx(gmOffset, rLoopTimes, rLoopTail, aLength);
        }

        int64_t resultOffset = resultCacheId * aLoopFactorAligned;
        __VEC_SCOPE__
        {
            __local_mem__ float* dbetaAddr = (__local_mem__ float*)dbetaTensor.GetPhyAddr() + resultOffset;
            __local_mem__ float* dgammaAddr = (__local_mem__ float*)dgammaTensor.GetPhyAddr() + resultOffset;
            __local_mem__ WEIGHT_TYPE* dbetaTypeAddr = (__local_mem__ WEIGHT_TYPE*)dbetaTensor.GetPhyAddr();
            __local_mem__ WEIGHT_TYPE* dgammaTypeAddr = (__local_mem__ WEIGHT_TYPE*)dgammaTensor.GetPhyAddr();
            AscendC::MicroAPI::MaskReg pFull =
                AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            AscendC::MicroAPI::RegTensor<float> dbetaReg, dgammaReg;
            LoadOneTensor<float>(dbetaAddr, dbetaReg, pFull, 0);
            LoadOneTensor<float>(dgammaAddr, dgammaReg, pFull, 0);
            StoreOneTensor<WEIGHT_TYPE>(dbetaTypeAddr, dbetaReg, pFull, 0);
            StoreOneTensor<WEIGHT_TYPE>(dgammaTypeAddr, dgammaReg, pFull, 0);
        }

        dbetaOutQue_.EnQue(dbetaTensor);
        dbetaTensor = dbetaOutQue_.template DeQue<float>();
        LocalTensor<WEIGHT_TYPE> dbetaType = dbetaTensor.template ReinterpretCast<WEIGHT_TYPE>();
        CopyOut(dbetaGm_[gmOffset], dbetaType, aLength);
        dgammaOutQue_.EnQue(dgammaTensor);
        dgammaTensor = dgammaOutQue_.template DeQue<float>();
        LocalTensor<WEIGHT_TYPE> dgammaType = dgammaTensor.template ReinterpretCast<WEIGHT_TYPE>();
        CopyOut(dgammaGm_[gmOffset], dgammaType, aLength);

        gammaInQue_.FreeTensor(gammaTensor);
        meanInQue_.FreeTensor(meanTensor);
        rstdInQue_.FreeTensor(rstdTensor);
        dbetaOutQue_.FreeTensor(dbetaTensor);
        dgammaOutQue_.FreeTensor(dgammaTensor);
    }

private:
    TPipe* pipe_ = nullptr;
    int64_t rDim_;
    int64_t aDim_;
    int64_t blockDim;
    int64_t mainBlockCount;
    int64_t mainBlockFactor;
    int64_t tailBlockFactor;
    int64_t blockFactor;

    int64_t aLoopFactor;
    int64_t aLoopFactorAligned;
    int64_t aLoopTimes;
    int64_t aLoopTail;
    int64_t rLoopFactor;
    int64_t rLoopTimes;
    int64_t rLoopTail;

    int64_t binaryFoldPoint;
    int64_t binaryBlockCount;
    int64_t binaryTailBlock;
    int64_t cacheBufferCount;
    int64_t resultCacheId;
    float reciprocal;

    // global tensor
    GlobalTensor<DY_TYPE> dyGm_, xGm_, dxGm_;
    GlobalTensor<WEIGHT_TYPE> gammaGm_, dgammaGm_, dbetaGm_;
    GlobalTensor<float> meanGm_, rstdGm_, workspaceGm_;

    // local tensor
    LocalTensor<float> dyMain, dyFold, xMain, xFold;
    LocalTensor<float> meanTensor, rstdTensor;
    LocalTensor<float> dbetaTensor, dgammaTensor;
    LocalTensor<WEIGHT_TYPE> gammaTensor;
    LocalTensor<DY_TYPE> dxTensor, dyTensor, xTensor;

    // queue
    TQue<QuePosition::VECIN, BUFFER_NUM> dyInQue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> xInQue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> meanInQue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> rstdInQue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> gammaInQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> dxOutQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> dbetaOutQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> dgammaOutQue_;
};

} // namespace BatchNormGradV3
#endif // __BATCH_NORM_GRAD_V3_RA_RECOMPUTE_REGBASE_H__