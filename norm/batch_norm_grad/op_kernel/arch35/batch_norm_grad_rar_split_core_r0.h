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
 * \file batch_norm_grad_rar_split_core_r0.h
 * \brief
 */

#ifndef __BATCH_NORM_GRAD_SPLIT_CORE_R0_H__
#define __BATCH_NORM_GRAD_SPLIT_CORE_R0_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "batch_norm_grad_common.h"
#include "batch_norm_grad_rar_split_core_r1.h"

namespace BatchNormGrad
{
using namespace AscendC;
using namespace AscendC::MicroAPI;

template <typename DY_DTYPE, typename WEIGHT_DTYPE>
class BatchNormGradRARSplitCoreR0
{
public:
    __aicore__ inline BatchNormGradRARSplitCoreR0(TPipe* pipeIn,
                                                    const BatchNormGradRARSplitCoreR0TilingData* tilingDataIn)
    {
        pipe_ = pipeIn;
        tilingData_ = tilingDataIn;
    }

    __aicore__ inline void Init(GM_ADDR dy, GM_ADDR x, GM_ADDR mean, GM_ADDR rstd, GM_ADDR gamma, GM_ADDR dx,
                                GM_ADDR dgamma, GM_ADDR dbeta, GM_ADDR workspace)
    {
        blkIdx_ = GetBlockIdx();
        isLastCore_ = blkIdx_ == (tilingData_->usedCoreNums - 1) ? true : false;
        hRecipVal_ = 1.0f / (float)(tilingData_->r1Dim * tilingData_->r0Dim);
        r0BlockOffset_ = blkIdx_ * tilingData_->r0Inner;

        meanGm_.SetGlobalBuffer((__gm__ float*)(mean));
        rstdGm_.SetGlobalBuffer((__gm__ float*)(rstd));
        gammaGm_.SetGlobalBuffer((__gm__ WEIGHT_DTYPE*)(gamma));
        dgammaGm_.SetGlobalBuffer((__gm__ WEIGHT_DTYPE*)(dgamma));
        dbetaGm_.SetGlobalBuffer((__gm__ WEIGHT_DTYPE*)(dbeta));

        dbetaTmpGm_.SetGlobalBuffer((__gm__ float*)(workspace));
        dgammaTmpGm_.SetGlobalBuffer((__gm__ float*)(workspace) +
                                     tilingData_->usedCoreNums * tilingData_->aDimAligned);

        dyGm_.SetGlobalBuffer((__gm__ DY_DTYPE*)(dy));
        xGm_.SetGlobalBuffer((__gm__ DY_DTYPE*)(x));
        dxGm_.SetGlobalBuffer((__gm__ DY_DTYPE*)(dx));
    }

    __aicore__ inline void Stage0InitBuffer()
    {
        // 二分相关buff初始化
        // r0InnerInnerStg0, r0InnerTailStg0 is aligned
        int64_t dyLen = tilingData_->r1InnerStg0 * tilingData_->r0InnerInnerStg0 * tilingData_->aInnerStg0;

        pipe_->InitBuffer(dyInQue_, BUFFER_NUM, dyLen * sizeof(DY_DTYPE));
        pipe_->InitBuffer(xInQue_, BUFFER_NUM, dyLen * sizeof(DY_DTYPE));

        // fp16 增加dy cast结果的空间
        if constexpr (IsSameType<DY_DTYPE, float>::value) {
            pipe_->InitBuffer(reduceInTmpBuffer_, dyLen * sizeof(float));
        } else {
            pipe_->InitBuffer(reduceInTmpBuffer_, dyLen * sizeof(float) * CONST_TWO);
            reduceTmpBufOffset_ = dyLen;
        }
        int64_t meanSize = tilingData_->aInnerAlignedStg0 * sizeof(float);
        pipe_->InitBuffer(meanInQue_, BUFFER_NUM, meanSize);
        pipe_->InitBuffer(rstdInQue_, BUFFER_NUM, meanSize);

        pipe_->InitBuffer(dbetaTmpOutQue_, BUFFER_NUM, meanSize);
        pipe_->InitBuffer(dgammaTmpOutQue_, BUFFER_NUM, meanSize);

        // 核内reducesum结果block对齐存储
        pipe_->InitBuffer(dbetaCacheBuffer_, meanSize * tilingData_->binAddCacheBufferCount);
        pipe_->InitBuffer(dgammaCacheBuffer_, meanSize * tilingData_->binAddCacheBufferCount);
    }

    __aicore__ inline void Stage1InitBuffer()
    {
        pipe_->Reset();

        int64_t aSizeIn = tilingData_->aInnerStg1 * sizeof(float) * tilingData_->usedCoreNums;
        pipe_->InitBuffer(dbetaTmpInQue_, BUFFER_NUM, aSizeIn);
        pipe_->InitBuffer(dgammaTmpInQue_, BUFFER_NUM, aSizeIn);

        pipe_->InitBuffer(dbetaTmpOutQue_, BUFFER_NUM, tilingData_->aInnerStg1 * sizeof(float));
        pipe_->InitBuffer(dgammaTmpOutQue_, BUFFER_NUM, tilingData_->aInnerStg1 * sizeof(float));
        pipe_->InitBuffer(dbetaOutQue_, BUFFER_NUM, tilingData_->aInnerStg1 * sizeof(WEIGHT_DTYPE));
        pipe_->InitBuffer(dgammaOutQue_, BUFFER_NUM, tilingData_->aInnerStg1 * sizeof(WEIGHT_DTYPE));
    }

    __aicore__ inline void Stage2InitBuffer()
    {
        pipe_->Reset();

        int64_t dyLen = tilingData_->r1InnerStg2 * tilingData_->r0InnerInnerStg2 * tilingData_->aInnerStg2;
        pipe_->InitBuffer(dyInQue_, BUFFER_NUM, dyLen * sizeof(DY_DTYPE));
        pipe_->InitBuffer(xInQue_, BUFFER_NUM, dyLen * sizeof(DY_DTYPE));
        pipe_->InitBuffer(dxOutQue_, BUFFER_NUM, dyLen * sizeof(DY_DTYPE));

        int64_t meanSize = tilingData_->aInnerAlignedStg2 * sizeof(float);
        pipe_->InitBuffer(meanInQue_, BUFFER_NUM, meanSize);
        pipe_->InitBuffer(rstdInQue_, BUFFER_NUM, meanSize);
        pipe_->InitBuffer(gammaInQue_, BUFFER_NUM, meanSize);
        pipe_->InitBuffer(dbetaTmpInQue_, BUFFER_NUM, meanSize);
        pipe_->InitBuffer(dgammaTmpInQue_, BUFFER_NUM, meanSize);
    }

    __aicore__ inline void Process()
    {
        // stage 0： 核内二分累加dbeta/dgamma
        Stage0InitBuffer();
        CalcDbetaDgammaInCore();

        // stage 1: 计算核间dbeta/dgamma
        SyncAll();
        if (blkIdx_ == 0) {
            Stage1InitBuffer();
            CalcDbetaDgamma();
        }

        // stage 2: 计算输出
        SyncAll();
        Stage2InitBuffer();
        CalcOutput();
    }

    // stage 0
    __aicore__ inline void CalcDbetaDgammaInCore()
    {
        for (int64_t i = 0; i < tilingData_->aOuterStg0; i++) {
            uint32_t curAInnerLen =
                i != (tilingData_->aOuterStg0 - 1) ? tilingData_->aInnerStg0 : tilingData_->aTailStg0;
            int64_t aOffset = i * tilingData_->aInnerStg0;

            CopyInMeanAndRstd(aOffset, curAInnerLen);
            LocalTensor<float> meanTensor = meanInQue_.template DeQue<float>();
            LocalTensor<float> rstdTensor = rstdInQue_.template DeQue<float>();
            __local_mem__ float* meanLocal = (__local_mem__ float*)meanTensor.GetPhyAddr();
            __local_mem__ float* rstdLocal = (__local_mem__ float*)rstdTensor.GetPhyAddr();

            CalcReduceSum(curAInnerLen, aOffset, meanLocal, rstdLocal);

            meanInQue_.FreeTensor(meanTensor);
            rstdInQue_.FreeTensor(rstdTensor);

            CopyOutDbetaDgammaTmp(curAInnerLen, aOffset);
        }
    }

    __aicore__ inline void CalcReduceSum(uint32_t curAInnerLen, int64_t aOffset, __local_mem__ float* meanLocal,
                                         __local_mem__ float* rstdLocal)
    {
        LocalTensor<float> dbetaCacheTensor = dbetaCacheBuffer_.Get<float>();
        LocalTensor<float> dgammaCacheTensor = dgammaCacheBuffer_.Get<float>();
        reduceInTmpTensor_ = reduceInTmpBuffer_.Get<float>();

        dbetaMainTmpTensor_ = dbetaTmpOutQue_.template AllocTensor<float>();
        dgammaMainTmpTensor_ = dgammaTmpOutQue_.template AllocTensor<float>();

        int64_t r0OuterInCore = tilingData_->r0InnerOuterStg0;
        int64_t r0InerIncore = tilingData_->r0InnerInnerStg0;
        int64_t r0TailIncore = tilingData_->r0InnerTailStg0;
        int64_t basicLoop = tilingData_->binAddBasicBlockLoop;
        int64_t mainLoop = tilingData_->binAddMainFoldCount;

        if (isLastCore_) {
            r0OuterInCore = tilingData_->r0TailOuterStg0;
            r0TailIncore = tilingData_->r0TailTailStg0;
            basicLoop = tilingData_->lastCoreBinAddBasicBlockLoop;
            mainLoop = tilingData_->lastCoreBinAddMainFoldCount;
        }

        for (int64_t basicBlockIdx = 0; basicBlockIdx < basicLoop; basicBlockIdx++) {
            int64_t r1OuterIdx = basicBlockIdx / r0OuterInCore;
            int64_t r0OuterIdx = basicBlockIdx % r0OuterInCore;
            int64_t dyOffset = r1OuterIdx * tilingData_->r1InnerStg0 * tilingData_->aDim * tilingData_->r0Dim +
                               aOffset * tilingData_->r0Dim + r0OuterIdx * tilingData_->r0InnerInnerStg0 + r0BlockOffset_;
            uint32_t curR1Len = 
                r1OuterIdx != (tilingData_->r1OuterStg0 - 1) ? tilingData_->r1InnerStg0 : tilingData_->r1TailStg0;
            uint32_t curR0Len = r0OuterIdx != (r0OuterInCore - 1) ? r0InerIncore : r0TailIncore;
            uint32_t curR0AlignedLen = curR0Len;
            if (r0OuterIdx == (r0OuterInCore - 1) && isLastCore_) {
                curR0AlignedLen = tilingData_->r0TailTailAlignedStg0;
            }

            ProcessMainBlock(meanLocal, rstdLocal, curAInnerLen, curR1Len, curR0Len, curR0AlignedLen, dyOffset);

            if (basicBlockIdx < mainLoop) {
                int64_t r1OuterIdxFold = (basicBlockIdx + basicLoop) / r0OuterInCore;
                int64_t r0OuterIdxFold = (basicBlockIdx + basicLoop) % r0OuterInCore;
                int64_t dyOffsetFold =
                    r1OuterIdxFold * tilingData_->r1InnerStg0 * tilingData_->aDim * tilingData_->r0Dim +
                    aOffset * tilingData_->r0Dim + r0OuterIdxFold * tilingData_->r0InnerInnerStg0 + r0BlockOffset_;
                uint32_t curR1LenFold = r1OuterIdxFold != (tilingData_->r1OuterStg0  - 1) ?
                    tilingData_->r1InnerStg0 : tilingData_->r1TailStg0;
                uint32_t curR0LenFold = r0OuterIdxFold != (r0OuterInCore - 1) ? r0InerIncore : r0TailIncore;
                uint32_t curR0AlignedLenFold = curR0LenFold;
                if (r0OuterIdxFold == (r0OuterInCore - 1) && isLastCore_) {
                    curR0AlignedLen = tilingData_->r0TailTailAlignedStg0;
                }

                ProcessFoldBlock(meanLocal, rstdLocal, curAInnerLen, curR1LenFold, curR0LenFold, curR0AlignedLenFold,
                                 dyOffsetFold);
            }
            int64_t cacheID = GetCacheID(basicBlockIdx);
            UpdateCache(dbetaCacheTensor, dbetaMainTmpTensor_, cacheID, tilingData_->aInnerAlignedStg0,
                        tilingData_->aInnerAlignedStg0);
            UpdateCache(dgammaCacheTensor, dgammaMainTmpTensor_, cacheID, tilingData_->aInnerAlignedStg0,
                        tilingData_->aInnerAlignedStg0);
        }

        if (basicLoop == 0) {  // 直接输出到dbetaMainTmpTensor_，dgammaMainTmpTensor_
            int64_t dyOffset = aOffset * tilingData_->r0Dim + r0BlockOffset_;
            uint32_t r0AlignedLen = isLastCore_ ? tilingData_->r0TailTailAlignedStg0 : r0TailIncore;
            ProcessMainBlock(meanLocal, rstdLocal, curAInnerLen, tilingData_->r1TailStg0, r0TailIncore,
                             r0AlignedLen, dyOffset);
        } else {
            // ub to ub copy
            uint32_t resCacheId =
                isLastCore_ ? tilingData_->lastCoreBinAddResultCacheID : tilingData_->binAddResultCacheID;
            DataCopy(dbetaMainTmpTensor_, dbetaCacheTensor[resCacheId * tilingData_->aInnerAlignedStg0],
                     tilingData_->aInnerAlignedStg0);  // curAInnerLen < tilingData_->aInnerAlignedStg0
            DataCopy(dgammaMainTmpTensor_, dgammaCacheTensor[resCacheId * tilingData_->aInnerAlignedStg0],
                     tilingData_->aInnerAlignedStg0);
        }

        dbetaTmpOutQue_.EnQue(dbetaMainTmpTensor_);
        dgammaTmpOutQue_.EnQue(dgammaMainTmpTensor_);
    }

    __aicore__ inline void ProcessMainBlock(__local_mem__ float* meanLocal, __local_mem__ float* rstdLocal,
                                            const uint32_t curAInnerLen, const uint32_t curR1Len,
                                            const uint32_t curR0Len, const uint32_t curR0AlignedLen,
                                            const int64_t dyOffset)
    {
        CopyInDyAndX(curAInnerLen, curR1Len, curR0Len, curR0AlignedLen, dyOffset);

        LocalTensor<DY_DTYPE> dyTensor = dyInQue_.DeQue<DY_DTYPE>();
        LocalTensor<DY_DTYPE> xTensor = xInQue_.DeQue<DY_DTYPE>();

        CalcDyAndX(dyTensor, xTensor, reduceInTmpTensor_, meanLocal, rstdLocal, curR0AlignedLen, curAInnerLen,
                   curR1Len);
        xInQue_.FreeTensor(xTensor);
        if constexpr (!IsSameType<DY_DTYPE, float>::value) {
            dyInQue_.FreeTensor(dyTensor);
        }

        uint32_t fusedRLen = curR0AlignedLen * curR1Len;
        uint32_t srcShape[2] = {curAInnerLen, fusedRLen};

        // ReduceSum高阶api只输出A轴大小，按block补齐部分仍是0
        Duplicate(dbetaMainTmpTensor_, float(0.0), tilingData_->aInnerAlignedStg0);
        Duplicate(dgammaMainTmpTensor_, float(0.0), tilingData_->aInnerAlignedStg0);

        if constexpr (IsSameType<DY_DTYPE, float>::value) {
            AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(dbetaMainTmpTensor_, dyTensor, srcShape,
                                                                          false);
            dyInQue_.FreeTensor(dyTensor);
        } else {
            AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(
                dbetaMainTmpTensor_, reduceInTmpTensor_[reduceTmpBufOffset_], srcShape, false);
        }

        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(dgammaMainTmpTensor_, reduceInTmpTensor_[0],
                                                                      srcShape, false);
    }

    // fold cast fp32 加到main
    __aicore__ inline void ProcessFoldBlock(__local_mem__ float* meanLocal, __local_mem__ float* rstdLocal,
                                            const uint32_t curAInnerLen, const uint32_t curR1Len,
                                            const uint32_t curR0Len, const uint32_t curR0AlignedLen,
                                            const int64_t dyOffset)
    {
        CopyInDyAndX(curAInnerLen, curR1Len, curR0Len, curR0AlignedLen, dyOffset);

        LocalTensor<DY_DTYPE> dyTensor = dyInQue_.DeQue<DY_DTYPE>();
        LocalTensor<DY_DTYPE> xTensor = xInQue_.DeQue<DY_DTYPE>();

        CalcDyAndX(dyTensor, xTensor, reduceInTmpTensor_, meanLocal, rstdLocal, curR0AlignedLen, curAInnerLen,
                   curR1Len);

        xInQue_.FreeTensor(xTensor);
        if constexpr (!IsSameType<DY_DTYPE, float>::value) {
            dyInQue_.FreeTensor(dyTensor);
        }

        uint32_t fusedRLen = curR0AlignedLen * curR1Len;
        uint32_t srcShape[2] = {curAInnerLen, fusedRLen};

        LocalTensor<float> dbetaFoldTmpTensor = dbetaTmpOutQue_.template AllocTensor<float>();
        LocalTensor<float> dgammaFoldTmpTensor = dgammaTmpOutQue_.template AllocTensor<float>();

        // ReduceSum高阶api只输出A轴大小，按block补齐部分仍是0
        Duplicate(dbetaFoldTmpTensor, float(0.0), tilingData_->aInnerAlignedStg0);
        Duplicate(dgammaFoldTmpTensor, float(0.0), tilingData_->aInnerAlignedStg0);

        if constexpr (IsSameType<DY_DTYPE, float>::value) {
            AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(dbetaFoldTmpTensor, dyTensor, srcShape,
                                                                          false);
            dyInQue_.FreeTensor(dyTensor);
        } else {
            AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(
                dbetaFoldTmpTensor, reduceInTmpTensor_[reduceTmpBufOffset_], srcShape, false);
        }

        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(dgammaFoldTmpTensor, reduceInTmpTensor_[0],
                                                                      srcShape, false);

        Add(dbetaMainTmpTensor_, dbetaMainTmpTensor_, dbetaFoldTmpTensor, tilingData_->aInnerAlignedStg0);
        Add(dgammaMainTmpTensor_, dgammaMainTmpTensor_, dgammaFoldTmpTensor, tilingData_->aInnerAlignedStg0);

        dbetaTmpOutQue_.FreeTensor(dbetaFoldTmpTensor);
        dgammaTmpOutQue_.FreeTensor(dgammaFoldTmpTensor);
    }

    __aicore__ inline void CalcDyAndX(LocalTensor<DY_DTYPE>& dyTensor, LocalTensor<DY_DTYPE>& xTensor,
                                      LocalTensor<float>& reduceTmpBufferTensor, __local_mem__ float* meanLocal,
                                      __local_mem__ float* rstdLocal, uint32_t curR0AlignedLen, uint16_t curAInnerLen,
                                      uint16_t curR1Len)
    {
        __local_mem__ DY_DTYPE* dyLocal = (__local_mem__ DY_DTYPE*)dyTensor.GetPhyAddr();
        __local_mem__ DY_DTYPE* xLocal = (__local_mem__ DY_DTYPE*)xTensor.GetPhyAddr();
        __local_mem__ float* dgammaReduceTmpLocal = (__local_mem__ float*)reduceTmpBufferTensor.GetPhyAddr();

        __local_mem__ float* dbetaReduceTmpLocal =
            (__local_mem__ float*)reduceTmpBufferTensor.GetPhyAddr() + reduceTmpBufOffset_;

        uint16_t loopNum = ops::CeilDiv(curR0AlignedLen, VL_FP32);

        __VEC_SCOPE__
        {
            MaskReg pregMask;
            MaskReg pregMaskAll = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            RegTensor<float> regMean, regRstd, regDy, regX;

            for (uint16_t i = 0; i < curAInnerLen; i++) {
                uint32_t sreg = curR0AlignedLen;
                LoadsTensorForDtypeT<float>(meanLocal, regMean, pregMaskAll, i);
                LoadsTensorForDtypeT<float>(rstdLocal, regRstd, pregMaskAll, i);
                for (uint16_t j = 0; j < loopNum; j++) {
                    pregMask = UpdateMask<float>(sreg);
                    for (uint16_t k = 0; k < curR1Len; k++) {
                        uint32_t offset = i * curR1Len * curR0AlignedLen + k * curR0AlignedLen + j * VL_FP32;
                        LoadOneTensor<DY_DTYPE>(dyLocal, regDy, pregMask, offset);
                        LoadOneTensor<DY_DTYPE>(xLocal, regX, pregMask, offset);

                        Sub(regX, regX, regMean, pregMask);
                        Mul(regX, regX, regDy, pregMask);
                        Mul(regX, regX, regRstd, pregMask);
                        DataCopy((__local_mem__ float*)dgammaReduceTmpLocal + offset, regX, pregMask);

                        if constexpr (!IsSameType<DY_DTYPE, float>::value) {
                            DataCopy((__local_mem__ float*)dbetaReduceTmpLocal + offset, regDy, pregMask);
                        }
                    }
                }
            }
        }
    }

    __aicore__ inline void CopyOutDbetaDgammaTmp(const uint32_t curAInnerLen, const int64_t aOffset)
    {
        int64_t gmOffset = blkIdx_ * tilingData_->aDimAligned + aOffset;
        DataCopyExtParams copyOutParams;
        copyOutParams.blockCount = 1;
        copyOutParams.blockLen = curAInnerLen * sizeof(float);
        copyOutParams.srcStride = 0;
        copyOutParams.dstStride = 0;

        LocalTensor<float> dbetaOutTmpTensor = dbetaTmpOutQue_.DeQue<float>();
        DataCopyPad<float, PaddingMode::Normal>(dbetaTmpGm_[gmOffset], dbetaOutTmpTensor, copyOutParams);
        dbetaTmpOutQue_.FreeTensor(dbetaOutTmpTensor);

        LocalTensor<float> dgammaOutTmpTensor = dgammaTmpOutQue_.DeQue<float>();
        DataCopyPad<float, PaddingMode::Normal>(dgammaTmpGm_[gmOffset], dgammaOutTmpTensor, copyOutParams);
        dgammaTmpOutQue_.FreeTensor(dgammaOutTmpTensor);
    }

    __aicore__ inline int64_t GetCacheID(const int64_t idx)
    {
        return bcnt1(idx ^ (idx + CONST_ONE)) - CONST_ONE;
    }

    __aicore__ inline void UpdateCache(const LocalTensor<float>& dstTensor, const LocalTensor<float>& srcTensor,
                                       const int64_t cacheID, const uint32_t stride, const uint32_t count)
    {
        uint16_t outerLoopTimes = ops::CeilDiv(count, VL_FP32);
        uint16_t innerLoopTimes = cacheID;
        uint32_t outerLoopStride = VL_FP32;
        uint32_t innerLoopStride = stride;

        __local_mem__ float* dst = (__local_mem__ float*)dstTensor.GetPhyAddr();
        __local_mem__ float* cah = (__local_mem__ float*)dstTensor.GetPhyAddr() + cacheID * stride;
        __local_mem__ float* src = (__local_mem__ float*)srcTensor.GetPhyAddr();

        __VEC_SCOPE__
        {
            uint32_t sreg = static_cast<uint32_t>(count);
            AscendC::MicroAPI::RegTensor<float> aReg, bReg;
            AscendC::MicroAPI::MaskReg pMask;
            for (uint16_t i = 0; i < outerLoopTimes; ++i) {
                pMask = AscendC::MicroAPI::UpdateMask<float>(sreg);
                DataCopy(aReg, (__local_mem__ float*)src + i * outerLoopStride);
                for (uint16_t j = 0; j < innerLoopTimes; ++j) {
                    DataCopy(bReg, (__local_mem__ float*)dst + i * outerLoopStride + j * innerLoopStride);
                    Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(aReg, aReg, bReg, pMask);
                }
                DataCopy((__local_mem__ float*)cah + i * outerLoopStride, aReg, pMask);
            }
        }
    }

    // stage 1
    __aicore__ inline void CalcDbetaDgamma()
    {
        for (int64_t i = 0; i < tilingData_->aOuterStg1; i++) {
            uint16_t curAInnerLen =
                i != (tilingData_->aOuterStg1 - 1) ? tilingData_->aInnerStg1 : tilingData_->aTailStg1;

            int64_t offset = i * tilingData_->aInnerStg1;
            CopyInDbetaDgammaTmp(offset, curAInnerLen);

            LocalTensor<float> dbetaTmpIn = dbetaTmpInQue_.template DeQue<float>();
            LocalTensor<float> dgammaTmpIn = dgammaTmpInQue_.template DeQue<float>();
            LocalTensor<float> dbetaTmpOutTensor = dbetaTmpOutQue_.AllocTensor<float>();
            LocalTensor<float> dgammaTmpOutTensor = dgammaTmpOutQue_.AllocTensor<float>();

            uint32_t srcShape[2] = {static_cast<uint32_t>(tilingData_->usedCoreNums),
                                    static_cast<uint32_t>(tilingData_->aInnerStg1)};

            ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(dbetaTmpOutTensor, dbetaTmpIn, srcShape, false);
            ReduceSum<float, AscendC::Pattern::Reduce::RA, true>(dgammaTmpOutTensor, dgammaTmpIn, srcShape, false);

            dbetaTmpInQue_.FreeTensor(dbetaTmpIn);
            dgammaTmpInQue_.FreeTensor(dgammaTmpIn);

            // 小shape输出处理
            LocalTensor<WEIGHT_DTYPE> dbetaOutTensor = dbetaOutQue_.AllocTensor<WEIGHT_DTYPE>();
            LocalTensor<WEIGHT_DTYPE> dgammaOutTensor = dgammaOutQue_.AllocTensor<WEIGHT_DTYPE>();

            __local_mem__ float* dbetaTmpOutLocal = (__local_mem__ float*)dbetaTmpOutTensor.GetPhyAddr();
            __local_mem__ float* dgammaTmpOutLocal = (__local_mem__ float*)dgammaTmpOutTensor.GetPhyAddr();
            __local_mem__ WEIGHT_DTYPE* dbetaOutLocal = (__local_mem__ WEIGHT_DTYPE*)dbetaOutTensor.GetPhyAddr();
            __local_mem__ WEIGHT_DTYPE* dgammaOutLocal = (__local_mem__ WEIGHT_DTYPE*)dgammaOutTensor.GetPhyAddr();

            uint16_t loopNum = ops::CeilDiv(static_cast<uint32_t>(curAInnerLen), VL_FP32);

            __VEC_SCOPE__
            {
                MaskReg pregMask;
                uint32_t sreg = curAInnerLen;
                RegTensor<float> regDbeta, regDgamma;

                for (uint16_t i = 0; i < loopNum; i++) {
                    pregMask = UpdateMask<float>(sreg);
                    uint32_t offset = i * VL_FP32;
                    LoadOneTensor<float>(dbetaTmpOutLocal, regDbeta, pregMask, offset);
                    LoadOneTensor<float>(dgammaTmpOutLocal, regDgamma, pregMask, offset);
                    StoreOneTensor<WEIGHT_DTYPE>(dbetaOutLocal, regDbeta, pregMask, offset);
                    StoreOneTensor<WEIGHT_DTYPE>(dgammaOutLocal, regDgamma, pregMask, offset);
                }
            }

            dbetaTmpOutQue_.EnQue(dbetaTmpOutTensor);
            dgammaTmpOutQue_.EnQue(dgammaTmpOutTensor);

            dbetaOutQue_.EnQue(dbetaOutTensor);
            dgammaOutQue_.EnQue(dgammaOutTensor);

            CopyOutDbetaDgammaAndTmp(offset, curAInnerLen);
        }
    }

    __aicore__ inline void CopyInDbetaDgammaTmp(int64_t offset, int64_t curALen)
    {
        DataCopyPadExtParams<float> dataCopyPadExtParams;
        dataCopyPadExtParams.isPad = false;
        dataCopyPadExtParams.leftPadding = 0;
        dataCopyPadExtParams.rightPadding = 0;
        dataCopyPadExtParams.paddingValue = 0;

        DataCopyExtParams copyInParams;
        copyInParams.blockCount = tilingData_->usedCoreNums;
        copyInParams.blockLen = curALen * sizeof(float);
        copyInParams.srcStride = (tilingData_->aDimAligned - curALen) * sizeof(float);
        copyInParams.dstStride = (tilingData_->aInnerStg1 - curALen) * sizeof(float) / BLOCK_SIZE;

        LocalTensor<float> dbetaTmpTensor = dbetaTmpInQue_.AllocTensor<float>();
        DataCopyPad<float, PaddingMode::Normal>(dbetaTmpTensor, dbetaTmpGm_[offset], copyInParams,
                                                dataCopyPadExtParams);
        dbetaTmpInQue_.EnQue(dbetaTmpTensor);

        LocalTensor<float> dgammaTmpTensor = dgammaTmpInQue_.AllocTensor<float>();
        DataCopyPad<float, PaddingMode::Normal>(dgammaTmpTensor, dgammaTmpGm_[offset], copyInParams,
                                                dataCopyPadExtParams);
        dgammaTmpInQue_.EnQue(dgammaTmpTensor);
    }

    __aicore__ inline void CopyOutDbetaDgammaAndTmp(int64_t offset, uint32_t curAInnerLen)
    {
        // 小shape输出
        DataCopyExtParams copyOutParams;
        copyOutParams.blockCount = 1;
        copyOutParams.blockLen = curAInnerLen * sizeof(WEIGHT_DTYPE);
        copyOutParams.srcStride = 0;
        copyOutParams.dstStride = 0;

        LocalTensor<WEIGHT_DTYPE> dbeta = dbetaOutQue_.DeQue<WEIGHT_DTYPE>();
        DataCopyPad<WEIGHT_DTYPE, PaddingMode::Normal>(dbetaGm_[offset], dbeta, copyOutParams);
        dbetaOutQue_.FreeTensor(dbeta);

        LocalTensor<WEIGHT_DTYPE> dgamma = dgammaOutQue_.DeQue<WEIGHT_DTYPE>();
        DataCopyPad<WEIGHT_DTYPE, PaddingMode::Normal>(dgammaGm_[offset], dgamma, copyOutParams);
        dgammaOutQue_.FreeTensor(dgamma);

        // 拷贝到workspace 临时空间
        DataCopyExtParams copyOutParams2;
        copyOutParams2.blockCount = 1;
        copyOutParams2.blockLen = curAInnerLen * sizeof(float);
        copyOutParams2.srcStride = 0;
        copyOutParams2.dstStride = 0;

        LocalTensor<float> dbetaTmp = dbetaTmpOutQue_.DeQue<float>();
        DataCopyPad<float, PaddingMode::Normal>(dbetaTmpGm_[offset], dbetaTmp, copyOutParams2);
        dbetaTmpOutQue_.FreeTensor(dbetaTmp);

        LocalTensor<float> dgammaTmp = dgammaTmpOutQue_.DeQue<float>();
        DataCopyPad<float, PaddingMode::Normal>(dgammaTmpGm_[offset], dgammaTmp, copyOutParams2);
        dgammaTmpOutQue_.FreeTensor(dgammaTmp);
    }

    // stage 2
    __aicore__ inline void CalcOutput()
    {
        for (int64_t i = 0; i < tilingData_->aOuterStg2; i++) {
            uint16_t curAInnerLen =
                i != (tilingData_->aOuterStg2 - 1) ? tilingData_->aInnerStg2 : tilingData_->aTailStg2;
            int64_t aOffset = i * tilingData_->aInnerStg2;
            CopyInSmallShapes(aOffset, curAInnerLen);

            LocalTensor<float> mean = meanInQue_.template DeQue<float>();
            LocalTensor<float> rstd = rstdInQue_.template DeQue<float>();
            LocalTensor<WEIGHT_DTYPE> gamma = gammaInQue_.template DeQue<WEIGHT_DTYPE>();
            LocalTensor<float> dbetaTmp = dbetaTmpInQue_.template DeQue<float>();
            LocalTensor<float> dgammaTmp = dgammaTmpInQue_.template DeQue<float>();

            __local_mem__ float* meanAddr = (__local_mem__ float*)mean.GetPhyAddr();
            __local_mem__ float* rstdAddr = (__local_mem__ float*)rstd.GetPhyAddr();
            __local_mem__ WEIGHT_DTYPE* gammaAddr = (__local_mem__ WEIGHT_DTYPE*)gamma.GetPhyAddr();
            __local_mem__ float* dbetaTmpAddr = (__local_mem__ float*)dbetaTmp.GetPhyAddr();
            __local_mem__ float* dgammaTmpAddr = (__local_mem__ float*)dgammaTmp.GetPhyAddr();

            int64_t r0OuterInCore = tilingData_->r0InnerOuterStg2;
            int64_t r0InerIncore = tilingData_->r0InnerInnerStg2;
            int64_t r0TailIncore = tilingData_->r0InnerTailStg2;
            if (isLastCore_) {
                r0OuterInCore = tilingData_->r0TailOuterStg2;
                r0TailIncore = tilingData_->r0TailTailStg2;
            }

            for (int64_t j = 0; j < tilingData_->r1OuterStg2; j++) {
                uint16_t curR1Len = 
                    j != (tilingData_->r1OuterStg2 - 1) ? tilingData_->r1InnerStg2 : tilingData_->r1TailStg2;
                for (int64_t k = 0; k < r0OuterInCore; k++) {
                    int64_t dyOffset = j * tilingData_->aDim * tilingData_->r0Dim * tilingData_->r1InnerStg2 +
                                       i * tilingData_->aInnerStg2 * tilingData_->r0Dim +
                                       k * tilingData_->r0InnerInnerStg2 + r0BlockOffset_;
                    uint32_t curR0Len = k != (r0OuterInCore - 1) ? r0InerIncore : r0TailIncore;
                    uint32_t curR0AlignedLen = curR0Len;
                    if (k == (r0OuterInCore - 1) && isLastCore_) {
                        curR0AlignedLen = tilingData_->r0TailTailAlignedStg2;
                    }

                    CopyInDyAndX(curAInnerLen, curR1Len, curR0Len, curR0AlignedLen, dyOffset);

                    LocalTensor<DY_DTYPE> dyTensor = dyInQue_.DeQue<DY_DTYPE>();
                    __local_mem__ DY_DTYPE* dyLocal = (__local_mem__ DY_DTYPE*)dyTensor.GetPhyAddr();

                    LocalTensor<DY_DTYPE> xTensor = xInQue_.DeQue<DY_DTYPE>();
                    __local_mem__ DY_DTYPE* xLocal = (__local_mem__ DY_DTYPE*)xTensor.GetPhyAddr();

                    LocalTensor<DY_DTYPE> dxTensor = dxOutQue_.AllocTensor<DY_DTYPE>();
                    __local_mem__ DY_DTYPE* dxLocal = (__local_mem__ DY_DTYPE*)dxTensor.GetPhyAddr();

                    uint16_t loopNum = ops::CeilDiv(curR0Len, VL_FP32);

                    __VEC_SCOPE__
                    {
                        MaskReg pregMaskAll =
                            AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();

                        MaskReg pregMask;

                        RegTensor<float> regMean, regRstd, regGamma, regDbetaTmp, regDgammaTmp, regDy, regX;

                        for (uint16_t i = 0; i < curAInnerLen; i++) {
                            LoadsTensorForDtypeT<float>(meanAddr, regMean, pregMaskAll, i);
                            LoadsTensorForDtypeT<float>(rstdAddr, regRstd, pregMaskAll, i);
                            LoadsTensorForDtypeT<float>(dbetaTmpAddr, regDbetaTmp, pregMaskAll, i);
                            LoadsTensorForDtypeT<float>(dgammaTmpAddr, regDgammaTmp, pregMaskAll, i);
                            LoadsTensorForDtypeT<WEIGHT_DTYPE>(gammaAddr, regGamma, pregMaskAll, i);

                            uint32_t sreg = curR0Len;
                            for (uint16_t j = 0; j < loopNum; j++) {  // r0
                                pregMask = UpdateMask<float>(sreg);
                                for (uint16_t k = 0; k < curR1Len; k++) {
                                    uint32_t offset =
                                        i * curR1Len * curR0AlignedLen + k * curR0AlignedLen + j * VL_FP32;
                                    LoadOneTensor<DY_DTYPE>(dyLocal, regDy, pregMask, offset);
                                    LoadOneTensor<DY_DTYPE>(xLocal, regX, pregMask, offset);

                                    Sub(regX, regX, regMean, pregMask);
                                    Mul(regX, regX, regRstd, pregMask);
                                    Mul(regX, regX, regDgammaTmp, pregMask);
                                    Add(regX, regX, regDbetaTmp, pregMask);
                                    Muls(regX, regX, hRecipVal_, pregMask);
                                    Sub(regX, regDy, regX, pregMask);
                                    Mul(regX, regX, regRstd, pregMask);
                                    Mul(regX, regX, regGamma, pregMask);

                                    StoreOneTensor<DY_DTYPE>(dxLocal, regX, pregMask, offset);
                                }
                            }
                        }
                    }
                    dyInQue_.FreeTensor(dyTensor);
                    xInQue_.FreeTensor(xTensor);

                    dxOutQue_.EnQue(dxTensor);
                    CopyOutDx(curAInnerLen, curR1Len, curR0Len, curR0AlignedLen, dyOffset);
                }
            }

            dbetaTmpInQue_.FreeTensor(dbetaTmp);
            dgammaTmpInQue_.FreeTensor(dgammaTmp);
            meanInQue_.FreeTensor(mean);
            rstdInQue_.FreeTensor(rstd);
            gammaInQue_.FreeTensor(gamma);
        }
    }

    __aicore__ inline void CopyInDyAndX(uint32_t curAInnerLen, uint32_t curR1Len, uint32_t curR0Len,
                                        uint32_t curR0AlignedLen, int64_t gmOffset)
    {
        // RAR -> AR
        LoopModeParams loopParams;
        loopParams.loop2Size = 1;
        loopParams.loop1Size = curAInnerLen;
        loopParams.loop2SrcStride = 0;
        loopParams.loop2DstStride = 0;
        loopParams.loop1SrcStride = tilingData_->r0Dim * sizeof(DY_DTYPE);
        loopParams.loop1DstStride = curR0AlignedLen * curR1Len * sizeof(DY_DTYPE);

        DataCopyPadExtParams<DY_DTYPE> dataCopyPadExtParams;
        dataCopyPadExtParams.isPad = true;
        dataCopyPadExtParams.leftPadding = 0;
        dataCopyPadExtParams.rightPadding = 0;  // 默认补齐到block对齐，实际大小（curR0AlignedLen - curR0Len）;
        dataCopyPadExtParams.paddingValue = 0;

        DataCopyExtParams copyInParams;
        copyInParams.blockCount = curR1Len;
        copyInParams.blockLen = curR0Len * sizeof(DY_DTYPE);
        copyInParams.srcStride = (tilingData_->r0Dim * tilingData_->aDim - curR0Len) * sizeof(DY_DTYPE);
        copyInParams.dstStride = (curR0AlignedLen - curR0Len) * sizeof(DY_DTYPE) / BLOCK_SIZE;

        LocalTensor<DY_DTYPE> x = xInQue_.template AllocTensor<DY_DTYPE>();
        SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
        DataCopyPad<DY_DTYPE, PaddingMode::Normal>(x, xGm_[gmOffset], copyInParams, dataCopyPadExtParams);
        ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
        xInQue_.EnQue(x);

        LocalTensor<DY_DTYPE> dy = dyInQue_.template AllocTensor<DY_DTYPE>();
        SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);
        DataCopyPad<DY_DTYPE, PaddingMode::Normal>(dy, dyGm_[gmOffset], copyInParams, dataCopyPadExtParams);
        ResetLoopModePara(DataCopyMVType::OUT_TO_UB);
        dyInQue_.EnQue(dy);
    }

    __aicore__ inline void CopyOutDx(uint32_t curAInnerLen, uint32_t curR1Len, uint32_t curR0Len,
                                     uint32_t curR0AlignedLen, int64_t offset)
    {
        // AR-> RAR
        LoopModeParams loopParams;
        loopParams.loop2Size = 1;
        loopParams.loop1Size = curAInnerLen;
        loopParams.loop2SrcStride = 0;
        loopParams.loop2DstStride = 0;
        loopParams.loop1SrcStride = curR0AlignedLen * curR1Len * sizeof(DY_DTYPE);
        loopParams.loop1DstStride = tilingData_->r0Dim * sizeof(DY_DTYPE);

        DataCopyExtParams copyOutParams;
        copyOutParams.blockCount = curR1Len;
        copyOutParams.blockLen = curR0Len * sizeof(DY_DTYPE);
        copyOutParams.srcStride = (curR0AlignedLen - curR0Len) * sizeof(DY_DTYPE) / BLOCK_SIZE;
        copyOutParams.dstStride = (tilingData_->r0Dim * tilingData_->aDim - curR0Len) * sizeof(DY_DTYPE);

        SetLoopModePara(loopParams, DataCopyMVType::UB_TO_OUT);

        LocalTensor<DY_DTYPE> dxTensor = dxOutQue_.DeQue<DY_DTYPE>();
        DataCopyPad<DY_DTYPE, PaddingMode::Normal>(dxGm_[offset], dxTensor, copyOutParams);
        dxOutQue_.FreeTensor(dxTensor);

        ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
    }

    __aicore__ inline void CopyInMeanAndRstd(int64_t offset, uint32_t dataLen)
    {
        DataCopyPadExtParams<float> dataCopyPadExtParams;
        dataCopyPadExtParams.isPad = false;
        dataCopyPadExtParams.leftPadding = 0;
        dataCopyPadExtParams.rightPadding = 0;
        dataCopyPadExtParams.paddingValue = 0;

        DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.blockLen = dataLen * sizeof(float);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;

        LocalTensor<float> mean = meanInQue_.template AllocTensor<float>();
        DataCopyPad(mean, meanGm_[offset], copyInParams, dataCopyPadExtParams);
        meanInQue_.EnQue(mean);

        LocalTensor<float> rstd = rstdInQue_.template AllocTensor<float>();
        DataCopyPad(rstd, rstdGm_[offset], copyInParams, dataCopyPadExtParams);
        rstdInQue_.EnQue(rstd);
    }

    __aicore__ inline void CopyInSmallShapes(int64_t offset, uint32_t dataLen)
    {
        DataCopyPadExtParams<float> dataCopyPadExtParams;
        dataCopyPadExtParams.isPad = false;
        dataCopyPadExtParams.leftPadding = 0;
        dataCopyPadExtParams.rightPadding = 0;
        dataCopyPadExtParams.paddingValue = 0;

        DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.blockLen = dataLen * sizeof(float);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;

        LocalTensor<float> mean = meanInQue_.template AllocTensor<float>();
        DataCopyPad(mean, meanGm_[offset], copyInParams, dataCopyPadExtParams);
        meanInQue_.EnQue(mean);

        LocalTensor<float> rstd = rstdInQue_.template AllocTensor<float>();
        DataCopyPad(rstd, rstdGm_[offset], copyInParams, dataCopyPadExtParams);
        rstdInQue_.EnQue(rstd);

        LocalTensor<float> dbetaTmp = dbetaTmpInQue_.template AllocTensor<float>();
        DataCopyPad(dbetaTmp, dbetaTmpGm_[offset], copyInParams, dataCopyPadExtParams);
        dbetaTmpInQue_.EnQue(dbetaTmp);

        LocalTensor<float> dgammaTmp = dgammaTmpInQue_.template AllocTensor<float>();
        DataCopyPad(dgammaTmp, dgammaTmpGm_[offset], copyInParams, dataCopyPadExtParams);
        dgammaTmpInQue_.EnQue(dgammaTmp);

        DataCopyPadExtParams<WEIGHT_DTYPE> dataCopyPadExtParams1;
        dataCopyPadExtParams1.isPad = false;
        dataCopyPadExtParams1.leftPadding = 0;
        dataCopyPadExtParams1.rightPadding = 0;
        dataCopyPadExtParams1.paddingValue = 0;

        DataCopyExtParams copyInParams1;
        copyInParams1.blockCount = 1;
        copyInParams1.blockLen = dataLen * sizeof(WEIGHT_DTYPE);
        copyInParams1.srcStride = 0;
        copyInParams1.dstStride = 0;

        LocalTensor<WEIGHT_DTYPE> gamma = gammaInQue_.template AllocTensor<WEIGHT_DTYPE>();
        DataCopyPad(gamma, gammaGm_[offset], copyInParams1, dataCopyPadExtParams1);
        gammaInQue_.EnQue(gamma);
    }

private:
    const BatchNormGradRARSplitCoreR0TilingData* tilingData_{nullptr};
    TPipe* pipe_{nullptr};

    GlobalTensor<DY_DTYPE> dyGm_;
    GlobalTensor<DY_DTYPE> xGm_;
    GlobalTensor<float> meanGm_;
    GlobalTensor<float> rstdGm_;
    GlobalTensor<float> dbetaTmpGm_;
    GlobalTensor<float> dgammaTmpGm_;

    TQue<QuePosition::VECIN, BUFFER_DEPTH> dyInQue_;
    TQue<QuePosition::VECIN, BUFFER_DEPTH> xInQue_;
    TQue<QuePosition::VECIN, BUFFER_DEPTH> meanInQue_;
    TQue<QuePosition::VECIN, BUFFER_DEPTH> rstdInQue_;
    TBuf<TPosition::VECCALC> dbetaCacheBuffer_;
    TBuf<TPosition::VECCALC> dgammaCacheBuffer_;
    TBuf<TPosition::VECCALC> reduceInTmpBuffer_;
    TQue<QuePosition::VECOUT, BUFFER_DEPTH> dbetaTmpOutQue_;
    TQue<QuePosition::VECOUT, BUFFER_DEPTH> dgammaTmpOutQue_;

    LocalTensor<float> reduceInTmpTensor_;
    LocalTensor<float> dbetaMainTmpTensor_;
    LocalTensor<float> dgammaMainTmpTensor_;

    TQue<QuePosition::VECIN, BUFFER_DEPTH> dbetaTmpInQue_;
    TQue<QuePosition::VECIN, BUFFER_DEPTH> dgammaTmpInQue_;
    TQue<QuePosition::VECOUT, BUFFER_DEPTH> dbetaOutQue_;
    TQue<QuePosition::VECOUT, BUFFER_DEPTH> dgammaOutQue_;

    GlobalTensor<WEIGHT_DTYPE> gammaGm_;
    GlobalTensor<WEIGHT_DTYPE> dgammaGm_;
    GlobalTensor<WEIGHT_DTYPE> dbetaGm_;
    GlobalTensor<DY_DTYPE> dxGm_;
    TQue<QuePosition::VECIN, BUFFER_DEPTH> gammaInQue_;
    TQue<QuePosition::VECOUT, BUFFER_DEPTH> dxOutQue_;

    int64_t blkIdx_{0};
    int64_t reduceTmpBufOffset_{0};
    float hRecipVal_{0};
    bool isLastCore_{false};
    int64_t r0BlockOffset_{0};
};
}  // namespace BatchNormGrad
#endif  // __BATCH_NORM_GRAD_SPLIT_CORE_R0_H__
