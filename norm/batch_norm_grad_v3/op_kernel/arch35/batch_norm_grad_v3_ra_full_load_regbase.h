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
 * \file batch_norm_grad_v3_ra_full_load_regbase.h
 * \brief
 */

#ifndef __BATCH_NORM_GRAD_V3_RA_FULL_LOAD_REGBASE_H__
#define __BATCH_NORM_GRAD_V3_RA_FULL_LOAD_REGBASE_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "batch_norm_grad_v3_common.h"

namespace BatchNormGradV3 {
using namespace AscendC;
using AscendC::Reg::LocalMemBar;
using AscendC::Reg::MaskMergeMode;
using AscendC::Reg::MemType;

template <typename DY_TYPE, typename WEIGHT_TYPE, int BUFFER_NUM = 1, bool PACKED = false>
class BatchNormGradV3RAFullLoad {
public:
    int64_t outputMask_ = OUTPUT_MASK_ALL;

    __aicore__ inline void ComputeStoreDx(__ubuf__ DY_TYPE* xAddr, __ubuf__ DY_TYPE* dyAddr, __ubuf__ float* dxAddr,
                                          RegTensor<float>& meanReg, RegTensor<float>& rstdReg,
                                          RegTensor<float>& gammaReg, RegTensor<float>& dgammaReg,
                                          RegTensor<float>& dbetaReg, float reciprocal, MaskReg& pMask, uint32_t offset)
    {
        // dx = (dy − ((x̂·dgamma + dbeta)·(1/N))) · grad_scale
        RegTensor<float> xReg, dyReg, gsReg;
        LoadOneTensor<DY_TYPE>(xAddr, xReg, pMask, offset);
        LoadOneTensor<DY_TYPE>(dyAddr, dyReg, pMask, offset);
        Sub<float, MaskMergeMode::ZEROING>(xReg, xReg, meanReg, pMask);
        Mul<float, MaskMergeMode::ZEROING>(xReg, xReg, rstdReg, pMask);
        Mul<float, MaskMergeMode::ZEROING>(xReg, xReg, dgammaReg, pMask);
        Add<float, MaskMergeMode::ZEROING>(xReg, xReg, dbetaReg, pMask);
        Muls<float, float, MaskMergeMode::ZEROING>(xReg, xReg, reciprocal, pMask);
        Sub<float, MaskMergeMode::ZEROING>(dyReg, dyReg, xReg, pMask);
        Mul<float, MaskMergeMode::ZEROING>(gsReg, rstdReg, gammaReg, pMask);
        Mul<float, MaskMergeMode::ZEROING>(dyReg, dyReg, gsReg, pMask);
        StoreOneTensor<DY_TYPE>(dxAddr, dyReg, pMask, offset);
    }
    __aicore__ inline BatchNormGradV3RAFullLoad(TPipe* pipe) { pipe_ = pipe; }

    __aicore__ inline void Init(__gm__ uint8_t* dy, __gm__ uint8_t* x, __gm__ uint8_t* mean, __gm__ uint8_t* rstd,
                                __gm__ uint8_t* gamma, __gm__ uint8_t* dx, __gm__ uint8_t* dgamma,
                                __gm__ uint8_t* dbeta, __gm__ uint8_t* workspace,
                                const BatchNormGradV3RAFullLoadTilingData* tilingData)
    {
        outputMask_ = tilingData->outputMask;
        if (outputMask_ == OUTPUT_MASK_NONE) {
            return;
        }
        const BatchNormGradV3BaseTilingData& baseTilingData = tilingData->baseTilingData;
        rDim_ = baseTilingData.r1Dim;
        aDim_ = baseTilingData.aDim;
        originalR_ = rDim_;
        if constexpr (PACKED) {
            while (packedRowsPerGroup_ * TWO * aDim_ <= VL_FP32) {
                packedRowsPerGroup_ *= TWO;
            }
            rDim_ = CeilDiv(originalR_, packedRowsPerGroup_);
        }
        numBlocks = tilingData->numBlocks;
        mainBlockCount = tilingData->mainBlockCount;
        mainBlockFactor = tilingData->mainBlockFactor;
        tailBlockFactor = tilingData->tailBlockFactor;

        reduceLoopTimes = tilingData->reduceLoopTimes;
        reduceRecursionLoop = tilingData->reduceRecursionLoop;
        foldLoopStep1 = tilingData->foldLoopStep1;
        foldLoopStep2 = tilingData->foldLoopStep2;
        foldLoopStep3 = tilingData->foldLoopStep3;
        foldLoopOffset1 = tilingData->foldLoopOffset1;
        foldLoopOffset2 = tilingData->foldLoopOffset2;
        foldLoopOffset3 = tilingData->foldLoopOffset3;

        // calc ub tiling
        int64_t blockFactor = GetBlockIdx() < mainBlockCount ? mainBlockFactor : tailBlockFactor;
        int64_t mainALoopFactorAligned = tilingData->mainALoopFactorAligned;
        int64_t tailALoopFactorAligned = tilingData->tailALoopFactorAligned;
        aLoopFactor = GetBlockIdx() < mainBlockCount ? tilingData->mainALoopFactor : tilingData->tailALoopFactor;
        aLoopFactorAligned = GetBlockIdx() < mainBlockCount ? mainALoopFactorAligned : tailALoopFactorAligned;
        aLoopCount = blockFactor / aLoopFactor;
        aLoopTail = blockFactor - aLoopCount * aLoopFactor;

        int64_t offset = tilingData->mainBlockFactor * GetBlockIdx();
        dyGm_.SetGlobalBuffer((__gm__ DY_TYPE*)dy + offset);
        xGm_.SetGlobalBuffer((__gm__ DY_TYPE*)x + offset);
        meanGm_.SetGlobalBuffer((__gm__ float*)mean + offset);
        rstdGm_.SetGlobalBuffer((__gm__ float*)rstd + offset);
        gammaGm_.SetGlobalBuffer((__gm__ WEIGHT_TYPE*)gamma + offset);
        dxGm_.SetGlobalBuffer((__gm__ DY_TYPE*)dx + offset);
        dgammaGm_.SetGlobalBuffer((__gm__ WEIGHT_TYPE*)dgamma + offset);
        dbetaGm_.SetGlobalBuffer((__gm__ WEIGHT_TYPE*)dbeta + offset);

        int64_t feat_bytes = rDim_ * aLoopFactorAligned * sizeof(DY_TYPE);
        int64_t para_bytes = aLoopFactorAligned * sizeof(float);
        int64_t gamma_bytes = aLoopFactorAligned * sizeof(WEIGHT_TYPE);
        pipe_->InitBuffer(dyInQue_, BUFFER_NUM, feat_bytes);
        pipe_->InitBuffer(xInQue_, BUFFER_NUM, feat_bytes);
        pipe_->InitBuffer(meanInQue_, BUFFER_NUM, para_bytes);
        pipe_->InitBuffer(rstdInQue_, BUFFER_NUM, para_bytes);
        pipe_->InitBuffer(gammaInQue_, BUFFER_NUM, gamma_bytes);
        pipe_->InitBuffer(dxOutQue_, BUFFER_NUM, feat_bytes);
        pipe_->InitBuffer(dbetaOutQue_, BUFFER_NUM, gamma_bytes);
        pipe_->InitBuffer(dgammaOutQue_, BUFFER_NUM, gamma_bytes);

        if (rDim_ > FOUR) {
            int64_t ws_bytes = (foldLoopStep1 + foldLoopStep2 + foldLoopStep3) * aLoopFactorAligned * sizeof(float);
            pipe_->InitBuffer(dbetaBinaryAddBuf_, ws_bytes);
            pipe_->InitBuffer(dgammaBinaryAddBuf_, ws_bytes);
        }
    }

    __aicore__ inline void Process()
    {
        if (outputMask_ == OUTPUT_MASK_NONE) {
            return;
        }
        if (aLoopFactor <= 0) {
            return;
        }
        if (GetBlockIdx() >= numBlocks) {
            return;
        }
        for (int64_t idx = 0; idx < aLoopCount; ++idx) {
            ProcessBlock(idx * aLoopFactor, aLoopFactor, aLoopFactorAligned);
        }
        if (aLoopTail > 0) {
            ProcessBlock(aLoopCount * aLoopFactor, aLoopTail, aLoopFactorAligned);
        }
    }

public:
    __aicore__ inline void ProcessBlock(const int64_t gmOffset, const int64_t channelFactor, const int64_t stride)
    {
        const int64_t factor = PACKED ? channelFactor * packedRowsPerGroup_ : channelFactor;
        const uint32_t channels = channelFactor;
        const uint16_t rowsPerGroup = packedRowsPerGroup_;
        /* Process one Block */
        // load dy
        LocalTensor<DY_TYPE> dy = dyInQue_.template AllocTensor<DY_TYPE>();
        LoadInput(dy, dyGm_[gmOffset], channelFactor, stride);
        dyInQue_.EnQue(dy);
        dy = dyInQue_.template DeQue<DY_TYPE>();

        // load x
        LocalTensor<DY_TYPE> x = xInQue_.template AllocTensor<DY_TYPE>();
        LoadInput(x, xGm_[gmOffset], channelFactor, stride);
        xInQue_.EnQue(x);
        x = xInQue_.template DeQue<DY_TYPE>();

        // load mean
        LocalTensor<float> mean = meanInQue_.template AllocTensor<float>();
        LoadChannel(mean, meanGm_[gmOffset], channelFactor);
        meanInQue_.EnQue(mean);
        mean = meanInQue_.template DeQue<float>();

        // load rstd
        LocalTensor<float> rstd = rstdInQue_.template AllocTensor<float>();
        LoadChannel(rstd, rstdGm_[gmOffset], channelFactor);
        rstdInQue_.EnQue(rstd);
        rstd = rstdInQue_.template DeQue<float>();

        // load gamma
        LocalTensor<WEIGHT_TYPE> gamma = gammaInQue_.template AllocTensor<WEIGHT_TYPE>();
        LoadChannel(gamma, gammaGm_[gmOffset], channelFactor);
        gammaInQue_.EnQue(gamma);
        gamma = gammaInQue_.template DeQue<WEIGHT_TYPE>();

        // calc
        LocalTensor<WEIGHT_TYPE> dbeta = dbetaOutQue_.template AllocTensor<WEIGHT_TYPE>();
        LocalTensor<WEIGHT_TYPE> dgamma = dgammaOutQue_.template AllocTensor<WEIGHT_TYPE>();
        LocalTensor<DY_TYPE> dx = dxOutQue_.template AllocTensor<DY_TYPE>();
        uint16_t outerLoop = (factor + 63) / 64;
        uint16_t innerLoop = rDim_;
        uint32_t outerStride = 64;
        uint32_t innerStride = stride;
        float reciprocal = 1.0f / static_cast<float>(originalR_);

        // dichotomy add
        uint16_t vfFoldLoopStep1 = foldLoopStep1;
        uint16_t vfFoldLoopStep2 = foldLoopStep2;
        uint16_t vfFoldLoopStep3 = foldLoopStep3;
        uint32_t vfFoldLoopOffset1 = foldLoopOffset1;
        uint32_t vfFoldLoopOffset2 = foldLoopOffset2;
        uint32_t vfFoldLoopOffset3 = foldLoopOffset3;
        uint32_t vfReduceLoopTimes = reduceLoopTimes;
        uint32_t vfReduceRecursionLoop = reduceRecursionLoop;
        uint16_t fullFoldLoopStep1 = vfFoldLoopStep1;
        uint16_t packedTailLoops = 0;
        uint32_t lastGroupElements = factor;
        if constexpr (PACKED) {
            uint32_t tailRows = originalR_ % packedRowsPerGroup_;
            if (tailRows != 0) {
                packedTailLoops = 1;
                lastGroupElements = tailRows * channels;
                if (innerLoop > FOUR && innerLoop % TWO == 0) {
                    --fullFoldLoopStep1;
                }
            }
        }

        if (rDim_ == ONE) {
            __VEC_SCOPE__
            {
                __ubuf__ DY_TYPE* dyAddr = (__ubuf__ DY_TYPE*)dy.GetPhyAddr();
                __ubuf__ DY_TYPE* xAddr = (__ubuf__ DY_TYPE*)x.GetPhyAddr();
                __ubuf__ float* meanAddr = (__ubuf__ float*)mean.GetPhyAddr();
                __ubuf__ float* rstdAddr = (__ubuf__ float*)rstd.GetPhyAddr();
                __ubuf__ WEIGHT_TYPE* gammaAddr = (__ubuf__ WEIGHT_TYPE*)gamma.GetPhyAddr();
                __ubuf__ float* dxAddr = (__ubuf__ float*)dx.GetPhyAddr();
                __ubuf__ WEIGHT_TYPE* dbetaAddr = (__ubuf__ WEIGHT_TYPE*)dbeta.GetPhyAddr();
                __ubuf__ WEIGHT_TYPE* dgammaAddr = (__ubuf__ WEIGHT_TYPE*)dgamma.GetPhyAddr();
                AscendC::Reg::MaskReg pMask;
                uint32_t count = factor;

                AscendC::Reg::RegTensor<float> dbetaReg, dgammaReg;
                AscendC::Reg::RegTensor<float> xReg, dyReg;
                AscendC::Reg::RegTensor<float> meanReg, rstdReg, gammaReg;
                for (uint16_t i = 0; i < outerLoop; ++i) {
                    pMask = AscendC::Reg::UpdateMask<float>(count);
                    Reg::MaskReg lastGroupMask = GetLastGroupMask(pMask, lastGroupElements);
                    LoadOneTensor<float>(meanAddr, meanReg, pMask, i * outerStride);
                    LoadOneTensor<float>(rstdAddr, rstdReg, pMask, i * outerStride);
                    LoadOneTensor<WEIGHT_TYPE>(gammaAddr, gammaReg, pMask, i * outerStride);
                    LoadOneTensor<DY_TYPE>(xAddr, xReg, pMask, i * outerStride);
                    LoadOneTensor<DY_TYPE>(dyAddr, dyReg, pMask, i * outerStride);
                    MaskPackedTail(dyReg, lastGroupMask, packedTailLoops);
                    Move<float, AscendC::Reg::MaskMergeMode::MERGING>(dbetaReg, dyReg, pMask);
                    StoreGradient(dbetaAddr, dbetaReg, pMask, i * outerStride, channels, rowsPerGroup);
                    Move<float, AscendC::Reg::MaskMergeMode::MERGING>(dgammaReg, xReg, pMask);
                    Sub<float, AscendC::Reg::MaskMergeMode::ZEROING>(dgammaReg, dgammaReg, meanReg, pMask);
                    MaskPackedTail(dgammaReg, lastGroupMask, packedTailLoops);
                    Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(dgammaReg, dgammaReg, dyReg, pMask);
                    Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(dgammaReg, dgammaReg, rstdReg, pMask);
                    StoreGradient(dgammaAddr, dgammaReg, pMask, i * outerStride, channels, rowsPerGroup);

                    // dx = (dy − ((x̂·dgamma + dbeta)·(1/N))) · grad_scale
                    AscendC::Reg::RegTensor<float> gsReg1;
                    Sub<float, AscendC::Reg::MaskMergeMode::ZEROING>(xReg, xReg, meanReg, pMask);
                    Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(xReg, xReg, rstdReg, pMask);
                    Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(xReg, xReg, dgammaReg, pMask);
                    Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(xReg, xReg, dbetaReg, pMask);
                    Muls<float, float, AscendC::Reg::MaskMergeMode::ZEROING>(xReg, xReg, reciprocal, pMask);
                    Sub<float, AscendC::Reg::MaskMergeMode::ZEROING>(dyReg, dyReg, xReg, pMask);
                    Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(gsReg1, rstdReg, gammaReg, pMask);
                    Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(dyReg, dyReg, gsReg1, pMask);
                    StoreOneTensor<DY_TYPE>(dxAddr, dyReg, pMask, i * outerStride);
                }
            }
        } else if (rDim_ == TWO) {
            __VEC_SCOPE__
            {
                __ubuf__ DY_TYPE* dyAddr = (__ubuf__ DY_TYPE*)dy.GetPhyAddr();
                __ubuf__ DY_TYPE* xAddr = (__ubuf__ DY_TYPE*)x.GetPhyAddr();
                __ubuf__ float* meanAddr = (__ubuf__ float*)mean.GetPhyAddr();
                __ubuf__ float* rstdAddr = (__ubuf__ float*)rstd.GetPhyAddr();
                __ubuf__ WEIGHT_TYPE* gammaAddr = (__ubuf__ WEIGHT_TYPE*)gamma.GetPhyAddr();
                __ubuf__ float* dxAddr = (__ubuf__ float*)dx.GetPhyAddr();
                __ubuf__ WEIGHT_TYPE* dbetaAddr = (__ubuf__ WEIGHT_TYPE*)dbeta.GetPhyAddr();
                __ubuf__ WEIGHT_TYPE* dgammaAddr = (__ubuf__ WEIGHT_TYPE*)dgamma.GetPhyAddr();
                AscendC::Reg::MaskReg pMask;
                uint32_t count = factor;

                AscendC::Reg::RegTensor<float> dbetaReg, dgammaReg;
                AscendC::Reg::RegTensor<float> xReg, dyReg;
                AscendC::Reg::RegTensor<float> xReg2, dyReg2;
                AscendC::Reg::RegTensor<float> meanReg, rstdReg, gammaReg;
                for (uint16_t i = 0; i < outerLoop; ++i) {
                    pMask = AscendC::Reg::UpdateMask<float>(count);
                    Reg::MaskReg lastGroupMask = GetLastGroupMask(pMask, lastGroupElements);
                    LoadOneTensor<float>(meanAddr, meanReg, pMask, i * outerStride);
                    LoadOneTensor<float>(rstdAddr, rstdReg, pMask, i * outerStride);
                    LoadOneTensor<WEIGHT_TYPE>(gammaAddr, gammaReg, pMask, i * outerStride);
                    LoadOneTensor<DY_TYPE>(xAddr, xReg, pMask, i * outerStride);
                    LoadOneTensor<DY_TYPE>(dyAddr, dyReg, pMask, i * outerStride);
                    LoadOneTensor<DY_TYPE>(xAddr, xReg2, pMask, i * outerStride + innerStride);
                    LoadOneTensor<DY_TYPE>(dyAddr, dyReg2, pMask, i * outerStride + innerStride);
                    MaskPackedTail(dyReg2, lastGroupMask, packedTailLoops);
                    // dbetaReg = dyReg + dyReg2
                    Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(dbetaReg, dyReg, dyReg2, pMask);
                    StoreGradient(dbetaAddr, dbetaReg, pMask, i * outerStride, channels, rowsPerGroup);
                    // dgammaReg = (xReg - meanReg) * dyReg + (xReg2 - meanReg) * dyReg2
                    Sub<float, AscendC::Reg::MaskMergeMode::ZEROING>(xReg, xReg, meanReg, pMask);
                    Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(dgammaReg, xReg, dyReg, pMask);
                    Sub<float, AscendC::Reg::MaskMergeMode::ZEROING>(xReg2, xReg2, meanReg, pMask);
                    MaskPackedTail(xReg2, lastGroupMask, packedTailLoops);
                    // MulAddDst 单次舍入
                    MulAddDst<float, AscendC::Reg::MaskMergeMode::ZEROING>(dgammaReg, xReg2, dyReg2, pMask);
                    // dgamma 先归约后乘 rstd（归约完成后统一乘一次，pMask 覆盖全部有效 lane）
                    Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(dgammaReg, dgammaReg, rstdReg, pMask);
                    StoreGradient(dgammaAddr, dgammaReg, pMask, i * outerStride, channels, rowsPerGroup);

                    // dx = (dy − ((x̂·dgamma + dbeta)·(1/N))) · grad_scale
                    ComputeStoreDx(xAddr, dyAddr, dxAddr, meanReg, rstdReg, gammaReg, dgammaReg, dbetaReg, reciprocal,
                                   pMask, i * outerStride);

                    ComputeStoreDx(xAddr, dyAddr, dxAddr, meanReg, rstdReg, gammaReg, dgammaReg, dbetaReg, reciprocal,
                                   pMask, i * outerStride + innerStride);
                }
            }
        } else if (rDim_ == THREE) {
            __VEC_SCOPE__
            {
                __ubuf__ DY_TYPE* dyAddr = (__ubuf__ DY_TYPE*)dy.GetPhyAddr();
                __ubuf__ DY_TYPE* xAddr = (__ubuf__ DY_TYPE*)x.GetPhyAddr();
                __ubuf__ float* meanAddr = (__ubuf__ float*)mean.GetPhyAddr();
                __ubuf__ float* rstdAddr = (__ubuf__ float*)rstd.GetPhyAddr();
                __ubuf__ WEIGHT_TYPE* gammaAddr = (__ubuf__ WEIGHT_TYPE*)gamma.GetPhyAddr();
                __ubuf__ float* dxAddr = (__ubuf__ float*)dx.GetPhyAddr();
                __ubuf__ WEIGHT_TYPE* dbetaAddr = (__ubuf__ WEIGHT_TYPE*)dbeta.GetPhyAddr();
                __ubuf__ WEIGHT_TYPE* dgammaAddr = (__ubuf__ WEIGHT_TYPE*)dgamma.GetPhyAddr();
                AscendC::Reg::MaskReg pMask;
                uint32_t count = factor;

                AscendC::Reg::RegTensor<float> dbetaReg, dgammaReg;
                AscendC::Reg::RegTensor<float> xReg, dyReg;
                AscendC::Reg::RegTensor<float> xReg2, dyReg2;
                AscendC::Reg::RegTensor<float> xReg3, dyReg3;
                AscendC::Reg::RegTensor<float> meanReg, rstdReg, gammaReg;
                for (uint16_t i = 0; i < outerLoop; ++i) {
                    pMask = AscendC::Reg::UpdateMask<float>(count);
                    Reg::MaskReg lastGroupMask = GetLastGroupMask(pMask, lastGroupElements);
                    LoadOneTensor<float>(meanAddr, meanReg, pMask, i * outerStride);
                    LoadOneTensor<float>(rstdAddr, rstdReg, pMask, i * outerStride);
                    LoadOneTensor<WEIGHT_TYPE>(gammaAddr, gammaReg, pMask, i * outerStride);
                    LoadOneTensor<DY_TYPE>(xAddr, xReg, pMask, i * outerStride);
                    LoadOneTensor<DY_TYPE>(dyAddr, dyReg, pMask, i * outerStride);
                    LoadOneTensor<DY_TYPE>(xAddr, xReg2, pMask, i * outerStride + innerStride);
                    LoadOneTensor<DY_TYPE>(dyAddr, dyReg2, pMask, i * outerStride + innerStride);
                    LoadOneTensor<DY_TYPE>(xAddr, xReg3, pMask, i * outerStride + innerStride * 2);
                    LoadOneTensor<DY_TYPE>(dyAddr, dyReg3, pMask, i * outerStride + innerStride * 2);
                    MaskPackedTail(dyReg3, lastGroupMask, packedTailLoops);
                    // dbetaReg = dyReg + dyReg2 + dyReg3
                    Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(dbetaReg, dyReg, dyReg2, pMask);
                    Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(dbetaReg, dbetaReg, dyReg3, pMask);
                    StoreGradient(dbetaAddr, dbetaReg, pMask, i * outerStride, channels, rowsPerGroup);
                    // dgammaReg = (xReg - meanReg) * dyReg
                    //           + (xReg2 - meanReg) * dyReg2
                    //           + (xReg3 - meanReg) * dyReg3
                    Sub<float, AscendC::Reg::MaskMergeMode::ZEROING>(xReg, xReg, meanReg, pMask);
                    Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(dgammaReg, xReg, dyReg, pMask);
                    Sub<float, AscendC::Reg::MaskMergeMode::ZEROING>(xReg2, xReg2, meanReg, pMask);
                    // 串行累加，后续元素 MulAddDst 单次舍入
                    MulAddDst<float, AscendC::Reg::MaskMergeMode::ZEROING>(dgammaReg, xReg2, dyReg2, pMask);
                    Sub<float, AscendC::Reg::MaskMergeMode::ZEROING>(xReg3, xReg3, meanReg, pMask);
                    MaskPackedTail(xReg3, lastGroupMask, packedTailLoops);
                    MulAddDst<float, AscendC::Reg::MaskMergeMode::ZEROING>(dgammaReg, xReg3, dyReg3, pMask);
                    // dgamma 先归约后乘 rstd（归约完成后统一乘一次，pMask 覆盖全部有效 lane）
                    Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(dgammaReg, dgammaReg, rstdReg, pMask);
                    StoreGradient(dgammaAddr, dgammaReg, pMask, i * outerStride, channels, rowsPerGroup);

                    // dx = (dy − ((x̂·dgamma + dbeta)·(1/N))) · grad_scale
                    ComputeStoreDx(xAddr, dyAddr, dxAddr, meanReg, rstdReg, gammaReg, dgammaReg, dbetaReg, reciprocal,
                                   pMask, i * outerStride);

                    ComputeStoreDx(xAddr, dyAddr, dxAddr, meanReg, rstdReg, gammaReg, dgammaReg, dbetaReg, reciprocal,
                                   pMask, i * outerStride + innerStride);

                    ComputeStoreDx(xAddr, dyAddr, dxAddr, meanReg, rstdReg, gammaReg, dgammaReg, dbetaReg, reciprocal,
                                   pMask, i * outerStride + innerStride * 2);
                }
            }
        } else if (rDim_ == FOUR) {
            __VEC_SCOPE__
            {
                __ubuf__ DY_TYPE* dyAddr = (__ubuf__ DY_TYPE*)dy.GetPhyAddr();
                __ubuf__ DY_TYPE* xAddr = (__ubuf__ DY_TYPE*)x.GetPhyAddr();
                __ubuf__ float* meanAddr = (__ubuf__ float*)mean.GetPhyAddr();
                __ubuf__ float* rstdAddr = (__ubuf__ float*)rstd.GetPhyAddr();
                __ubuf__ WEIGHT_TYPE* gammaAddr = (__ubuf__ WEIGHT_TYPE*)gamma.GetPhyAddr();
                __ubuf__ float* dxAddr = (__ubuf__ float*)dx.GetPhyAddr();
                __ubuf__ WEIGHT_TYPE* dbetaAddr = (__ubuf__ WEIGHT_TYPE*)dbeta.GetPhyAddr();
                __ubuf__ WEIGHT_TYPE* dgammaAddr = (__ubuf__ WEIGHT_TYPE*)dgamma.GetPhyAddr();
                AscendC::Reg::MaskReg pMask;
                uint32_t count = factor;

                AscendC::Reg::RegTensor<float> dbetaReg, dgammaReg;
                AscendC::Reg::RegTensor<float> xReg, dyReg;
                AscendC::Reg::RegTensor<float> xReg2, dyReg2;
                AscendC::Reg::RegTensor<float> xReg3, dyReg3;
                AscendC::Reg::RegTensor<float> xReg4, dyReg4;
                AscendC::Reg::RegTensor<float> meanReg, rstdReg, gammaReg;
                for (uint16_t i = 0; i < outerLoop; ++i) {
                    pMask = AscendC::Reg::UpdateMask<float>(count);
                    Reg::MaskReg lastGroupMask = GetLastGroupMask(pMask, lastGroupElements);
                    LoadOneTensor<float>(meanAddr, meanReg, pMask, i * outerStride);
                    LoadOneTensor<float>(rstdAddr, rstdReg, pMask, i * outerStride);
                    LoadOneTensor<WEIGHT_TYPE>(gammaAddr, gammaReg, pMask, i * outerStride);

                    // calc dbeta
                    LoadOneTensor<DY_TYPE>(dyAddr, dyReg, pMask, i * outerStride);
                    LoadOneTensor<DY_TYPE>(dyAddr, dyReg3, pMask, i * outerStride + innerStride * 2);
                    Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(dyReg, dyReg, dyReg3, pMask);
                    LoadOneTensor<DY_TYPE>(dyAddr, dyReg2, pMask, i * outerStride + innerStride);
                    LoadOneTensor<DY_TYPE>(dyAddr, dyReg4, pMask, i * outerStride + innerStride * 3);
                    MaskPackedTail(dyReg4, lastGroupMask, packedTailLoops);
                    Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(dyReg2, dyReg2, dyReg4, pMask);
                    Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(dbetaReg, dyReg, dyReg2, pMask);
                    StoreGradient(dbetaAddr, dbetaReg, pMask, i * outerStride, channels, rowsPerGroup);

                    // calc dgamma
                    // 串行累加（元素 0,1,2,3 序），首元素独立舍入，后续元素 MulAddDst 单次舍入
                    LoadOneTensor<DY_TYPE>(xAddr, xReg, pMask, i * outerStride);
                    LoadOneTensor<DY_TYPE>(dyAddr, dyReg, pMask, i * outerStride);
                    Sub<float, AscendC::Reg::MaskMergeMode::ZEROING>(xReg, xReg, meanReg, pMask);
                    Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(dgammaReg, xReg, dyReg, pMask);
                    LoadOneTensor<DY_TYPE>(xAddr, xReg2, pMask, i * outerStride + innerStride);
                    LoadOneTensor<DY_TYPE>(dyAddr, dyReg2, pMask, i * outerStride + innerStride);
                    Sub<float, AscendC::Reg::MaskMergeMode::ZEROING>(xReg2, xReg2, meanReg, pMask);
                    MulAddDst<float, AscendC::Reg::MaskMergeMode::ZEROING>(dgammaReg, xReg2, dyReg2, pMask);
                    LoadOneTensor<DY_TYPE>(xAddr, xReg3, pMask, i * outerStride + innerStride * 2);
                    LoadOneTensor<DY_TYPE>(dyAddr, dyReg3, pMask, i * outerStride + innerStride * 2);
                    Sub<float, AscendC::Reg::MaskMergeMode::ZEROING>(xReg3, xReg3, meanReg, pMask);
                    MulAddDst<float, AscendC::Reg::MaskMergeMode::ZEROING>(dgammaReg, xReg3, dyReg3, pMask);
                    LoadOneTensor<DY_TYPE>(xAddr, xReg4, pMask, i * outerStride + innerStride * 3);
                    LoadOneTensor<DY_TYPE>(dyAddr, dyReg4, pMask, i * outerStride + innerStride * 3);
                    MaskPackedTail(dyReg4, lastGroupMask, packedTailLoops);
                    Sub<float, AscendC::Reg::MaskMergeMode::ZEROING>(xReg4, xReg4, meanReg, pMask);
                    MaskPackedTail(xReg4, lastGroupMask, packedTailLoops);
                    // MulAddDst 单次舍入
                    MulAddDst<float, AscendC::Reg::MaskMergeMode::ZEROING>(dgammaReg, xReg4, dyReg4, pMask);
                    // dgamma 先归约后乘 rstd（归约完成后统一乘一次，pMask 覆盖全部有效 lane）
                    Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(dgammaReg, dgammaReg, rstdReg, pMask);
                    StoreGradient(dgammaAddr, dgammaReg, pMask, i * outerStride, channels, rowsPerGroup);

                    // dx = (dy − ((x̂·dgamma + dbeta)·(1/N))) · grad_scale
                    ComputeStoreDx(xAddr, dyAddr, dxAddr, meanReg, rstdReg, gammaReg, dgammaReg, dbetaReg, reciprocal,
                                   pMask, i * outerStride);

                    ComputeStoreDx(xAddr, dyAddr, dxAddr, meanReg, rstdReg, gammaReg, dgammaReg, dbetaReg, reciprocal,
                                   pMask, i * outerStride + innerStride);

                    ComputeStoreDx(xAddr, dyAddr, dxAddr, meanReg, rstdReg, gammaReg, dgammaReg, dbetaReg, reciprocal,
                                   pMask, i * outerStride + innerStride * 2);

                    ComputeStoreDx(xAddr, dyAddr, dxAddr, meanReg, rstdReg, gammaReg, dgammaReg, dbetaReg, reciprocal,
                                   pMask, i * outerStride + innerStride * 3);
                }
            }
        } else {
            LocalTensor<float> dbetaws = dbetaBinaryAddBuf_.Get<float>();
            LocalTensor<float> dgammaws = dgammaBinaryAddBuf_.Get<float>();

            __VEC_SCOPE__
            {
                __ubuf__ DY_TYPE* dyAddr = (__ubuf__ DY_TYPE*)dy.GetPhyAddr();
                __ubuf__ DY_TYPE* xAddr = (__ubuf__ DY_TYPE*)x.GetPhyAddr();
                __ubuf__ float* meanAddr = (__ubuf__ float*)mean.GetPhyAddr();
                __ubuf__ float* rstdAddr = (__ubuf__ float*)rstd.GetPhyAddr();
                __ubuf__ WEIGHT_TYPE* gammaAddr = (__ubuf__ WEIGHT_TYPE*)gamma.GetPhyAddr();
                __ubuf__ float* dxAddr = (__ubuf__ float*)dx.GetPhyAddr();
                __ubuf__ WEIGHT_TYPE* dbetaAddr = (__ubuf__ WEIGHT_TYPE*)dbeta.GetPhyAddr();
                __ubuf__ WEIGHT_TYPE* dgammaAddr = (__ubuf__ WEIGHT_TYPE*)dgamma.GetPhyAddr();
                __ubuf__ float* dbetawsAddr = (__ubuf__ float*)dbetaws.GetPhyAddr();
                __ubuf__ float* dgammawsAddr = (__ubuf__ float*)dgammaws.GetPhyAddr();
                AscendC::Reg::MaskReg pMask;
                uint32_t count = factor;

                AscendC::Reg::RegTensor<float> dbetaReg, dgammaReg;
                AscendC::Reg::RegTensor<float> xReg, yReg;
                AscendC::Reg::RegTensor<float> meanReg, rstdReg, gammaReg;
                for (uint16_t i = 0; i < outerLoop; ++i) {
                    pMask = AscendC::Reg::UpdateMask<float>(count);
                    Reg::MaskReg lastGroupMask = GetLastGroupMask(pMask, lastGroupElements);
                    LoadOneTensor<float>(meanAddr, meanReg, pMask, i * outerStride);
                    LoadOneTensor<float>(rstdAddr, rstdReg, pMask, i * outerStride);
                    // Keep the partial group out of the complete-group loop.
                    for (uint16_t j = 0; j < fullFoldLoopStep1; ++j) {
                        FoldFourRows(dyAddr, xAddr, dbetawsAddr, dgammawsAddr, meanReg, rstdReg, pMask, pMask,
                                     i * outerStride, innerStride, vfFoldLoopOffset1, j, 0);
                    }
                    for (uint16_t j = fullFoldLoopStep1; j < vfFoldLoopStep1; ++j) {
                        FoldFourRows(dyAddr, xAddr, dbetawsAddr, dgammawsAddr, meanReg, rstdReg, pMask, lastGroupMask,
                                     i * outerStride, innerStride, vfFoldLoopOffset1, j, packedTailLoops);
                    }
                    for (uint16_t j = 0; j < vfFoldLoopStep2; ++j) {
                        AscendC::Reg::RegTensor<float> dy1Reg, dy2Reg, x1Reg, x2Reg;
                        AscendC::Reg::RegTensor<float> dyFold1Reg, xFold1Reg;
                        LoadOneTensor<DY_TYPE>(xAddr, x1Reg, pMask, i * outerStride + vfFoldLoopOffset2 * innerStride);
                        LoadOneTensor<DY_TYPE>(dyAddr, dy1Reg, pMask,
                                               i * outerStride + vfFoldLoopOffset2 * innerStride);
                        Sub<float, AscendC::Reg::MaskMergeMode::ZEROING>(x1Reg, x1Reg, meanReg, pMask);
                        Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(x1Reg, x1Reg, dy1Reg, pMask);
                        LoadOneTensor<DY_TYPE>(
                            dyAddr, dyFold1Reg, pMask,
                            i * outerStride + vfFoldLoopOffset2 * innerStride + vfFoldLoopOffset1 * innerStride);
                        MaskPackedTail(dyFold1Reg, lastGroupMask, packedTailLoops);
                        Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(dy1Reg, dy1Reg, dyFold1Reg, pMask);
                        LoadOneTensor<DY_TYPE>(
                            xAddr, xFold1Reg, pMask,
                            i * outerStride + vfFoldLoopOffset2 * innerStride + vfFoldLoopOffset1 * innerStride);
                        Sub<float, AscendC::Reg::MaskMergeMode::ZEROING>(xFold1Reg, xFold1Reg, meanReg, pMask);
                        MaskPackedTail(xFold1Reg, lastGroupMask, packedTailLoops);
                        // MulAddDst 单次舍入
                        MulAddDst<float, AscendC::Reg::MaskMergeMode::ZEROING>(x1Reg, xFold1Reg, dyFold1Reg, pMask);
                        LoadOneTensor<DY_TYPE>(dyAddr, dy2Reg, pMask,
                                               i * outerStride + (vfFoldLoopOffset2 + 1) * innerStride);
                        Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(dy1Reg, dy1Reg, dy2Reg, pMask);
                        StoreOneTensor<float>(dbetawsAddr, dy1Reg, pMask,
                                              i * outerStride + vfFoldLoopStep1 * innerStride);
                        LoadOneTensor<DY_TYPE>(xAddr, x2Reg, pMask,
                                               i * outerStride + (vfFoldLoopOffset2 + 1) * innerStride);
                        Sub<float, AscendC::Reg::MaskMergeMode::ZEROING>(x2Reg, x2Reg, meanReg, pMask);
                        // MulAddDst 单次舍入
                        MulAddDst<float, AscendC::Reg::MaskMergeMode::ZEROING>(x1Reg, x2Reg, dy2Reg, pMask);
                        StoreOneTensor<float>(dgammawsAddr, x1Reg, pMask,
                                              i * outerStride + vfFoldLoopStep1 * innerStride);
                    }
                    for (uint16_t j = 0; j < vfFoldLoopStep3; ++j) {
                        AscendC::Reg::RegTensor<float> dy1Reg, dy2Reg, x1Reg, x2Reg;
                        LoadOneTensor<DY_TYPE>(xAddr, x1Reg, pMask,
                                               i * outerStride + (vfFoldLoopOffset3 + 2 * j + 0) * innerStride);
                        LoadOneTensor<DY_TYPE>(dyAddr, dy1Reg, pMask,
                                               i * outerStride + (vfFoldLoopOffset3 + 2 * j + 0) * innerStride);
                        Sub<float, AscendC::Reg::MaskMergeMode::ZEROING>(x1Reg, x1Reg, meanReg, pMask);
                        Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(x1Reg, x1Reg, dy1Reg, pMask);
                        LoadOneTensor<DY_TYPE>(dyAddr, dy2Reg, pMask,
                                               i * outerStride + (vfFoldLoopOffset3 + 2 * j + 1) * innerStride);
                        Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(dy1Reg, dy1Reg, dy2Reg, pMask);
                        StoreOneTensor<float>(dbetawsAddr, dy1Reg, pMask,
                                              i * outerStride + (vfFoldLoopStep1 + vfFoldLoopStep2 + j) * innerStride);
                        LoadOneTensor<DY_TYPE>(xAddr, x2Reg, pMask,
                                               i * outerStride + (vfFoldLoopOffset3 + 2 * j + 1) * innerStride);
                        Sub<float, AscendC::Reg::MaskMergeMode::ZEROING>(x2Reg, x2Reg, meanReg, pMask);
                        // MulAddDst 单次舍入
                        MulAddDst<float, AscendC::Reg::MaskMergeMode::ZEROING>(x1Reg, x2Reg, dy2Reg, pMask);
                        StoreOneTensor<float>(dgammawsAddr, x1Reg, pMask,
                                              i * outerStride + (vfFoldLoopStep1 + vfFoldLoopStep2 + j) * innerStride);
                    }
                    uint16_t loopTimes = vfReduceLoopTimes;
                    for (uint16_t k = 0; k < static_cast<uint16_t>(vfReduceRecursionLoop); ++k) {
                        loopTimes = loopTimes >> 1;
                        LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
                        for (uint16_t j = 0; j < loopTimes; ++j) {
                            LoadTwoTensorSum(dbetawsAddr, xReg, pMask, i * outerStride + (2 * j + 0) * innerStride,
                                             i * outerStride + (2 * j + 1) * innerStride);
                            StoreOneTensor<float>(dbetawsAddr, xReg, pMask, i * outerStride + j * innerStride);
                            LoadTwoTensorSum(dgammawsAddr, xReg, pMask, i * outerStride + (2 * j + 0) * innerStride,
                                             i * outerStride + (2 * j + 1) * innerStride);
                            StoreOneTensor<float>(dgammawsAddr, xReg, pMask, i * outerStride + j * innerStride);
                        }
                    }
                    LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
                    LoadTwoTensorSum(dbetawsAddr, dbetaReg, pMask, i * outerStride + 0 * innerStride,
                                     i * outerStride + 1 * innerStride);
                    StoreGradient(dbetaAddr, dbetaReg, pMask, i * outerStride, channels, rowsPerGroup);
                    LoadTwoTensorSum(dgammawsAddr, dgammaReg, pMask, i * outerStride + 0 * innerStride,
                                     i * outerStride + 1 * innerStride);
                    // dgamma 先归约后乘 rstd（归约完成后统一乘一次，pMask 覆盖全部有效 lane）
                    Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(dgammaReg, dgammaReg, rstdReg, pMask);
                    StoreGradient(dgammaAddr, dgammaReg, pMask, i * outerStride, channels, rowsPerGroup);

                    // dx = (dy − ((x̂·dgamma + dbeta)·(1/N))) · grad_scale
                    LoadOneTensor<WEIGHT_TYPE>(gammaAddr, gammaReg, pMask, i * outerStride);
                    for (uint16_t j = 0; j < innerLoop; ++j) {
                        LoadOneTensor<DY_TYPE>(xAddr, xReg, pMask, i * outerStride + j * innerStride);
                        Sub<float, AscendC::Reg::MaskMergeMode::ZEROING>(xReg, xReg, meanReg, pMask);
                        Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(xReg, xReg, rstdReg, pMask);
                        Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(xReg, xReg, dgammaReg, pMask);
                        Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(xReg, xReg, dbetaReg, pMask);
                        Muls<float, float, AscendC::Reg::MaskMergeMode::ZEROING>(xReg, xReg, reciprocal, pMask);
                        LoadOneTensor<DY_TYPE>(dyAddr, yReg, pMask, i * outerStride + j * innerStride);
                        Sub<float, AscendC::Reg::MaskMergeMode::ZEROING>(yReg, yReg, xReg, pMask);
                        Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(yReg, yReg, gammaReg, pMask);
                        Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(yReg, yReg, rstdReg, pMask);
                        StoreOneTensor<DY_TYPE>(dxAddr, yReg, pMask, i * outerStride + j * innerStride);
                    }
                }
            }
        }
        dyInQue_.FreeTensor(dy);
        xInQue_.FreeTensor(x);
        meanInQue_.FreeTensor(mean);
        rstdInQue_.FreeTensor(rstd);
        gammaInQue_.FreeTensor(gamma);

        // store dbeta
        dbetaOutQue_.EnQue(dbeta);
        dbeta = dbetaOutQue_.template DeQue<WEIGHT_TYPE>();
        if ((outputMask_ & OUTPUT_MASK_DBETA) != OUTPUT_MASK_NONE) {
            CopyOut(dbetaGm_[gmOffset], dbeta, channelFactor);
        }
        dbetaOutQue_.FreeTensor(dbeta);

        // store dgamma
        dgammaOutQue_.EnQue(dgamma);
        dgamma = dgammaOutQue_.template DeQue<WEIGHT_TYPE>();
        if ((outputMask_ & OUTPUT_MASK_DGAMMA) != OUTPUT_MASK_NONE) {
            CopyOut(dgammaGm_[gmOffset], dgamma, channelFactor);
        }
        dgammaOutQue_.FreeTensor(dgamma);

        // store dx
        dxOutQue_.EnQue(dx);
        dx = dxOutQue_.template DeQue<DY_TYPE>();
        if ((outputMask_ & OUTPUT_MASK_DX) != OUTPUT_MASK_NONE) {
            if constexpr (PACKED) {
                PackedCopyOutput(dxGm_[gmOffset], dx, originalR_, channels, rowsPerGroup);
            } else {
                CopyOut(dxGm_[gmOffset], dx, rDim_, factor, aDim_, stride);
            }
        }
        dxOutQue_.FreeTensor(dx);
    }

private:
    __aicore__ inline void LoadInput(const LocalTensor<DY_TYPE>& dst, const GlobalTensor<DY_TYPE>& src,
                                     int64_t channels, int64_t stride)
    {
        if constexpr (PACKED) {
            PackedCopyInput(dst, src, originalR_, channels, packedRowsPerGroup_);
        } else {
            CopyIn(dst, src, rDim_, channels, stride, aDim_);
        }
    }

    template <typename T>
    __aicore__ inline void LoadChannel(const LocalTensor<T>& dst, const GlobalTensor<T>& src, int64_t channels)
    {
        if constexpr (PACKED) {
            PackedCopyChannel(dst, src, channels, packedRowsPerGroup_);
        } else {
            CopyIn(dst, src, 1, channels, 0, aDim_);
        }
    }

    __aicore__ inline void MaskPackedTail(Reg::RegTensor<float>& value, Reg::MaskReg& valid, uint16_t tailMaskLoops)
    {
        if constexpr (PACKED) {
            // Hard-zero the tail group's padding lanes (UB residue may be NaN/Inf);
            // masked-out lanes are written to zero, never read. Applies to both dy
            // and x registers of the tail row so garbage cannot enter the reductions.
            for (uint16_t t = 0; t < tailMaskLoops; ++t) {
                Reg::Muls<float, float, Reg::MaskMergeMode::ZEROING>(value, value, 1.0f, valid);
            }
        }
    }

    __aicore__ inline Reg::MaskReg GetLastGroupMask(const Reg::MaskReg& mask, uint32_t elements)
    {
        if constexpr (PACKED) {
            return Reg::UpdateMask<float>(elements);
        } else {
            return mask;
        }
    }

    __aicore__ inline void FoldFourRows(__ubuf__ DY_TYPE* dyAddr, __ubuf__ DY_TYPE* xAddr, __ubuf__ float* dbetawsAddr,
                                        __ubuf__ float* dgammawsAddr, Reg::RegTensor<float>& meanReg,
                                        Reg::RegTensor<float>& rstdReg, Reg::MaskReg& pMask,
                                        Reg::MaskReg& lastGroupMask, uint32_t outerOffset, uint32_t innerStride,
                                        uint32_t foldOffset, uint16_t j, uint16_t tailMaskLoops)
    {
        AscendC::Reg::RegTensor<float> dy1Reg, dy2Reg, x1Reg, x2Reg;
        AscendC::Reg::RegTensor<float> dyFold1Reg, dyFold2Reg, xFold1Reg, xFold2Reg;

        LoadOneTensor<DY_TYPE>(xAddr, x1Reg, pMask, outerOffset + (2 * j + 0) * innerStride);
        LoadOneTensor<DY_TYPE>(dyAddr, dy1Reg, pMask, outerOffset + (2 * j + 0) * innerStride);
        // dgamma 先归约后乘 rstd：段内不乘（raw 部分和入 ws），最终 StoreGradient 前统一乘一次
        Sub<float, AscendC::Reg::MaskMergeMode::ZEROING>(x1Reg, x1Reg, meanReg, pMask);
        Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(x1Reg, x1Reg, dy1Reg, pMask);
        LoadOneTensor<DY_TYPE>(dyAddr, dyFold1Reg, pMask, outerOffset + (2 * j + 0 + foldOffset) * innerStride);
        Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(dy1Reg, dy1Reg, dyFold1Reg, pMask);
        LoadOneTensor<DY_TYPE>(xAddr, xFold1Reg, pMask, outerOffset + (2 * j + 0 + foldOffset) * innerStride);
        Sub<float, AscendC::Reg::MaskMergeMode::ZEROING>(xFold1Reg, xFold1Reg, meanReg, pMask);
        // MulAddDst 单次舍入
        MulAddDst<float, AscendC::Reg::MaskMergeMode::ZEROING>(x1Reg, xFold1Reg, dyFold1Reg, pMask);
        LoadOneTensor<DY_TYPE>(xAddr, x2Reg, pMask, outerOffset + (2 * j + 1) * innerStride);
        LoadOneTensor<DY_TYPE>(dyAddr, dy2Reg, pMask, outerOffset + (2 * j + 1) * innerStride);
        Sub<float, AscendC::Reg::MaskMergeMode::ZEROING>(x2Reg, x2Reg, meanReg, pMask);
        Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(x2Reg, x2Reg, dy2Reg, pMask);
        LoadOneTensor<DY_TYPE>(dyAddr, dyFold2Reg, pMask, outerOffset + (2 * j + 1 + foldOffset) * innerStride);
        MaskPackedTail(dyFold2Reg, lastGroupMask, tailMaskLoops);
        Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(dy2Reg, dy2Reg, dyFold2Reg, pMask);
        Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(dy1Reg, dy1Reg, dy2Reg, pMask);
        StoreOneTensor<float>(dbetawsAddr, dy1Reg, pMask, outerOffset + j * innerStride);
        LoadOneTensor<DY_TYPE>(xAddr, xFold2Reg, pMask, outerOffset + (2 * j + 1 + foldOffset) * innerStride);
        Sub<float, AscendC::Reg::MaskMergeMode::ZEROING>(xFold2Reg, xFold2Reg, meanReg, pMask);
        MaskPackedTail(xFold2Reg, lastGroupMask, tailMaskLoops);
        // MulAddDst 单次舍入
        MulAddDst<float, AscendC::Reg::MaskMergeMode::ZEROING>(x2Reg, xFold2Reg, dyFold2Reg, pMask);
        Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(x1Reg, x1Reg, x2Reg, pMask);
        StoreOneTensor<float>(dgammawsAddr, x1Reg, pMask, outerOffset + j * innerStride);
    }

    __aicore__ inline void StoreGradient(__ubuf__ WEIGHT_TYPE* dst, Reg::RegTensor<float>& value, Reg::MaskReg& mask,
                                         uint32_t offset, uint32_t channels, uint16_t rowsPerGroup)
    {
        if constexpr (PACKED) {
            PackedFoldChannels(value, channels, rowsPerGroup);
            uint32_t count = channels;
            Reg::MaskReg channelMask = Reg::UpdateMask<float>(count);
            StoreOneTensor<WEIGHT_TYPE>(dst, value, channelMask, offset);
            PackedBroadcastChannels(value, channels, rowsPerGroup);
        } else {
            StoreOneTensor<WEIGHT_TYPE>(dst, value, mask, offset);
        }
    }

    int64_t originalR_{0};
    uint16_t packedRowsPerGroup_{1};
    TPipe* pipe_ = nullptr;
    int64_t rDim_;
    int64_t aDim_;
    int64_t numBlocks;
    int64_t mainBlockCount;
    int64_t mainBlockFactor;
    int64_t tailBlockFactor;

    int64_t aLoopFactor;
    int64_t aLoopFactorAligned;
    int64_t aLoopCount;
    int64_t aLoopTail;

    int64_t reduceLoopTimes;
    int64_t reduceRecursionLoop;
    int64_t foldLoopStep1;
    int64_t foldLoopStep2;
    int64_t foldLoopStep3;
    int64_t foldLoopOffset1;
    int64_t foldLoopOffset2;
    int64_t foldLoopOffset3;

    GlobalTensor<DY_TYPE> dyGm_, xGm_, dxGm_;
    GlobalTensor<WEIGHT_TYPE> gammaGm_, dgammaGm_, dbetaGm_;
    GlobalTensor<float> meanGm_, rstdGm_, workspaceGm_;

    TQue<QuePosition::VECIN, BUFFER_NUM> dyInQue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> xInQue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> meanInQue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> rstdInQue_;
    TQue<QuePosition::VECIN, BUFFER_NUM> gammaInQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> dxOutQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> dbetaOutQue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> dgammaOutQue_;
    TBuf<TPosition::VECCALC> dbetaBinaryAddBuf_;
    TBuf<TPosition::VECCALC> dgammaBinaryAddBuf_;
};
} // namespace BatchNormGradV3
#endif // __BATCH_NORM_GRAD_V3_RA_FULL_LOAD_REGBASE_H__
