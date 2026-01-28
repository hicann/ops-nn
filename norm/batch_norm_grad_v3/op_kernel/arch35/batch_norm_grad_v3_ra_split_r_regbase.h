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
 * \file batch_norm_grad_v3_ra_split_r_regbase.h
 * \brief
 */

#ifndef __BATCH_NORM_GRAD_V3_RA_SPLIT_R_REGBASE_H__
#define __BATCH_NORM_GRAD_V3_RA_SPLIT_R_REGBASE_H__

#include "kernel_operator.h"
#include "../inc/platform.h"
#include "batch_norm_grad_v3_common.h"

namespace BatchNormGradV3 {
using namespace AscendC;

template <typename DY_TYPE, typename WEIGHT_TYPE>
class BatchNormGradV3RASplitR {
public:
    __aicore__ inline BatchNormGradV3RASplitR(TPipe* pipe)
    {
        pipe_ = pipe;
    }

    __aicore__ inline void Init(
        GM_ADDR dy, GM_ADDR x, GM_ADDR mean, GM_ADDR rstd, GM_ADDR gamma, GM_ADDR dx, GM_ADDR dgamma, GM_ADDR dbeta,
        GM_ADDR workspace, const BatchNormGradV3RASplitRTilingData* tilingData)
    {
        blockIdx_ = GetBlockIdx();
        tilingData_ = tilingData;
        int64_t offset = blockIdx_ * tilingData_->blockFactor * tilingData_->aDim;
        aFactorAlign_ = tilingData_->aFactorAlign;
        if (blockIdx_ == (tilingData_->usedCoreNum - 1)) {
            currBlockFactor_ = tilingData_->tailBlockFactor;
            binaryBlockCnt_ = tilingData_->lastCoreBlockCnt;
            binaryFoldPoint_ = tilingData_->lastCoreFoldPoint;
            binaryBlockTail_ = tilingData_->lastCoreLoopTail;
            dxLoopFactor_ = tilingData_->dxLastCoreFactor;
            dxLoopTail_ = tilingData_->dxLastCoreTail;
            dxLoopTimes_ = tilingData_->dxLastCoreTimes;
        } else {
            currBlockFactor_ = tilingData_->blockFactor;
            binaryBlockCnt_ = tilingData_->binaryBlockCnt;
            binaryFoldPoint_ = tilingData_->binaryFoldPoint;
            binaryBlockTail_ = tilingData_->binaryBlockTail;
            dxLoopFactor_ = tilingData_->dxLoopFactor;
            dxLoopTail_ = tilingData_->dxLoopTail;
            dxLoopTimes_ = tilingData_->dxLoopTimes;
        }
        currLoopFactor_ = (tilingData_->rLoopFactor > currBlockFactor_) ? currBlockFactor_ : tilingData_->rLoopFactor;

        dyGm_.SetGlobalBuffer((__gm__ DY_TYPE*)(dy) + offset);
        xGm_.SetGlobalBuffer((__gm__ DY_TYPE*)(x) + offset);
        dxGm_.SetGlobalBuffer((__gm__ DY_TYPE*)(dx) + offset);
        meanGm_.SetGlobalBuffer((__gm__ float*)(mean));
        rstdGm_.SetGlobalBuffer((__gm__ float*)(rstd));
        gammaGm_.SetGlobalBuffer((__gm__ WEIGHT_TYPE*)(gamma));
        dgammaGm_.SetGlobalBuffer((__gm__ WEIGHT_TYPE*)(dgamma));
        dbetaGm_.SetGlobalBuffer((__gm__ WEIGHT_TYPE*)(dbeta));
        dbetaWorkSpace_.SetGlobalBuffer((__gm__ float*)(workspace));
        dgammaWorkSpace_.SetGlobalBuffer((__gm__ float*)(workspace) + tilingData_->usedCoreNum * tilingData_->aDim);
        reciprocal_ = 1.0f / (float)(tilingData_->rDim);
        resultCacheId_ = GetCacheID(binaryFoldPoint_ - 1);
    }

    __aicore__ inline void Process()
    {
        // stage 0： 核内二分累加dbeta/dgamma
        Stage0InitBuffer();
        CalcDbetaDgammaInCore();
        // stage 1: 计算核间dbeta/dgamma
        SyncAll();
        if (blockIdx_ == 0) {
            Stage1InitBuffer();
            ProcessDbetaDgamma();
        }
        // stage 2: 计算输出
        SyncAll();
        Stage2InitBuffer();
        ProcessDX();
    }

private:
    __aicore__ inline void Stage0InitBuffer()
    {
        int64_t rDimSize = currLoopFactor_ * aFactorAlign_;
        int64_t aDimSize = aFactorAlign_ * sizeof(float);
        pipe_->InitBuffer(dyInQue_, BUFFER_NUM, rDimSize * sizeof(DY_TYPE));
        pipe_->InitBuffer(xInQue_, BUFFER_NUM, rDimSize * sizeof(DY_TYPE));
        pipe_->InitBuffer(dyTmpQue_, rDimSize * sizeof(float));
        pipe_->InitBuffer(xTmpQue_, rDimSize * sizeof(float));
        pipe_->InitBuffer(meanInQue_, BUFFER_NUM, aDimSize);
        pipe_->InitBuffer(rstdInQue_, BUFFER_NUM, aDimSize);
        pipe_->InitBuffer(dbetaWsOutQue_, BUFFER_NUM, aDimSize);
        pipe_->InitBuffer(dgammaWsOutQue_, BUFFER_NUM, aDimSize);
        pipe_->InitBuffer(dbetaCacheBuffer_, aDimSize * tilingData_->cacheBuffCnt);
        pipe_->InitBuffer(dgammaCacheBuffer_, aDimSize * tilingData_->cacheBuffCnt);
    }

    __aicore__ inline void CalcDbetaDgammaInCore()
    {
        for (uint32_t idx = 0; idx < tilingData_->aLoopTimes; idx++) {
            int64_t aLength = (idx == tilingData_->aLoopTimes - 1) ? tilingData_->aFactorTail : tilingData_->aFactor;
            dbetaWsTensor_ = dbetaWsOutQue_.AllocTensor<float>();
            dgammaWsTensor_ = dgammaWsOutQue_.AllocTensor<float>();
            dbetaCacheTensor_ = dbetaCacheBuffer_.Get<float>();
            dgammaCacheTensor_ = dgammaCacheBuffer_.Get<float>();
            int64_t baseOffset = idx * tilingData_->aFactor;
            LoadMeanRstdToUb(baseOffset, aLength);
            meanTensor_ = meanInQue_.template DeQue<float>();
            rstdTensor_ = rstdInQue_.template DeQue<float>();
            ProcessOneAFactor(baseOffset, aLength);
            meanInQue_.FreeTensor(meanTensor_);
            rstdInQue_.FreeTensor(rstdTensor_);
            int64_t cacheOffset = resultCacheId_ * aFactorAlign_;
            DataCopy(dbetaWsTensor_, dbetaCacheTensor_[cacheOffset], aFactorAlign_);
            DataCopy(dgammaWsTensor_, dgammaCacheTensor_[cacheOffset], aFactorAlign_);
            dbetaWsOutQue_.EnQue(dbetaWsTensor_);
            dgammaWsOutQue_.EnQue(dgammaWsTensor_);
            int64_t wsOffset = baseOffset + blockIdx_ * tilingData_->aDim;
            StoreDbetaDgammaToWs(wsOffset, aLength);
        }
    }

    __aicore__ inline void ProcessOneAFactor(int64_t baseOffset, int64_t aLength)
    {
        int64_t foldCnt = binaryBlockCnt_ - binaryFoldPoint_;
        for (int64_t loopIdx = 0; loopIdx < binaryFoldPoint_; ++loopIdx) {
            int64_t mainOffset = baseOffset + loopIdx * currLoopFactor_ * tilingData_->aDim;
            // 只有main部分的场景
            LoadDyXToUb(mainOffset, currLoopFactor_, aLength, aFactorAlign_, tilingData_->aDim);
            LocalTensor<DY_TYPE> dyTensor = dyInQue_.DeQue<DY_TYPE>();
            LocalTensor<DY_TYPE> xTensor = xInQue_.DeQue<DY_TYPE>();
            dyMainTensor_ = dyTmpQue_.Get<float>();
            xMainTensor_ = xTmpQue_.Get<float>();
            ProcessMainBlock(currLoopFactor_, aLength, dyTensor, xTensor);
            dyInQue_.FreeTensor(dyTensor);
            xInQue_.FreeTensor(xTensor);
            if (loopIdx < foldCnt) {
                int64_t foldOffset = baseOffset + (loopIdx + binaryFoldPoint_) * currLoopFactor_ * tilingData_->aDim;
                int64_t rLength = (loopIdx == foldCnt - 1) ? binaryBlockTail_ : currLoopFactor_;
                LoadDyXToUb(foldOffset, rLength, aLength, aFactorAlign_, tilingData_->aDim);
                dyFoldTensor_ = dyInQue_.DeQue<DY_TYPE>();
                xFoldTensor_ = xInQue_.DeQue<DY_TYPE>();
                ProcessFoldBlock(rLength, aLength);
                dyInQue_.FreeTensor(dyFoldTensor_);
                xInQue_.FreeTensor(xFoldTensor_);
            }
            ProcessSummation(loopIdx, currLoopFactor_, aLength);
        }
    }

    __aicore__ inline void ProcessMainBlock(
        const int64_t rLength, const int64_t aLength, const LocalTensor<DY_TYPE> dyTensor,
        const LocalTensor<DY_TYPE> xTensor)
    {
        uint16_t outerLoopTimes = CeilDiv(aLength, VL_FP32);
        uint16_t innerLoopTimes = rLength;
        uint16_t outerLoopStride = VL_FP32;
        uint16_t innerLoopStride = aFactorAlign_;

        __VEC_SCOPE__
        {
            __local_mem__ DY_TYPE* xAddr = (__local_mem__ DY_TYPE*)xTensor.GetPhyAddr();
            __local_mem__ DY_TYPE* dyAddr = (__local_mem__ DY_TYPE*)dyTensor.GetPhyAddr();
            __local_mem__ float* meanAddr = (__local_mem__ float*)meanTensor_.GetPhyAddr();
            __local_mem__ float* rstdAddr = (__local_mem__ float*)rstdTensor_.GetPhyAddr();
            __local_mem__ float* xMainAddr = (__local_mem__ float*)xMainTensor_.GetPhyAddr();
            __local_mem__ float* dyMainAddr = (__local_mem__ float*)dyMainTensor_.GetPhyAddr();
            MicroAPI::MaskReg pMask;
            uint32_t count = aLength;
            MicroAPI::RegTensor<float> xMainReg, dyMainReg, meanReg, rstdReg;
            for (uint16_t i = 0; i < outerLoopTimes; ++i) {
                pMask = MicroAPI::UpdateMask<float>(count);
                LoadOneTensor<float>(meanAddr, meanReg, pMask, i * outerLoopStride);
                LoadOneTensor<float>(rstdAddr, rstdReg, pMask, i * outerLoopStride);
                for (uint16_t j = 0; j < innerLoopTimes; ++j) {
                    int32_t offset = i * outerLoopStride + j * innerLoopStride;
                    LoadOneTensor<DY_TYPE>(xAddr, xMainReg, pMask, offset);
                    LoadOneTensor<DY_TYPE>(dyAddr, dyMainReg, pMask, offset);
                    MicroAPI::Sub<float, MicroAPI::MaskMergeMode::ZEROING>(xMainReg, xMainReg, meanReg, pMask);
                    MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(xMainReg, xMainReg, dyMainReg, pMask);
                    MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(xMainReg, xMainReg, rstdReg, pMask);
                    StoreOneTensor<float>(xMainAddr, xMainReg, pMask, offset);
                    StoreOneTensor<float>(dyMainAddr, dyMainReg, pMask, offset);
                }
            }
        }
    }

    __aicore__ inline void ProcessFoldBlock(int64_t rLength, int64_t aLength)
    {
        uint16_t outerLoopTimes = CeilDiv(aLength, VL_FP32);
        uint16_t innerLoopTimes = rLength;
        uint16_t outerLoopStride = VL_FP32;
        uint16_t innerLoopStride = aFactorAlign_;

        __VEC_SCOPE__
        {
            __local_mem__ float* dyMainAddr = (__local_mem__ float*)dyMainTensor_.GetPhyAddr();
            __local_mem__ float* xMainAddr = (__local_mem__ float*)xMainTensor_.GetPhyAddr();
            __local_mem__ DY_TYPE* dyFoldAddr = (__local_mem__ DY_TYPE*)dyFoldTensor_.GetPhyAddr();
            __local_mem__ DY_TYPE* xFoldAddr = (__local_mem__ DY_TYPE*)xFoldTensor_.GetPhyAddr();
            __local_mem__ float* meanAddr = (__local_mem__ float*)meanTensor_.GetPhyAddr();
            __local_mem__ float* rstdAddr = (__local_mem__ float*)rstdTensor_.GetPhyAddr();
            MicroAPI::MaskReg pMask;
            uint32_t count = aLength;
            MicroAPI::RegTensor<float> dyMainReg, dyFoldReg, xMainReg, xFoldReg, meanReg, rstdReg;
            for (uint16_t i = 0; i < outerLoopTimes; ++i) {
                pMask = MicroAPI::UpdateMask<float>(count);
                LoadOneTensor<float>(meanAddr, meanReg, pMask, i * outerLoopStride);
                LoadOneTensor<float>(rstdAddr, rstdReg, pMask, i * outerLoopStride);
                for (uint16_t j = 0; j < innerLoopTimes; ++j) {
                    int32_t offset = i * outerLoopStride + j * innerLoopStride;
                    LoadOneTensor<float>(dyMainAddr, dyMainReg, pMask, offset);
                    LoadOneTensor<DY_TYPE>(dyFoldAddr, dyFoldReg, pMask, offset);
                    MicroAPI::Add<float, MicroAPI::MaskMergeMode::ZEROING>(dyMainReg, dyMainReg, dyFoldReg, pMask);
                    StoreOneTensor<float>(dyMainAddr, dyMainReg, pMask, offset);
                    LoadOneTensor<float>(xMainAddr, xMainReg, pMask, offset);
                    LoadOneTensor<DY_TYPE>(xFoldAddr, xFoldReg, pMask, offset);
                    MicroAPI::Sub<float, MicroAPI::MaskMergeMode::ZEROING>(xFoldReg, xFoldReg, meanReg, pMask);
                    MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(xFoldReg, xFoldReg, dyFoldReg, pMask);
                    MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(xFoldReg, xFoldReg, rstdReg, pMask);
                    MicroAPI::Add<float, MicroAPI::MaskMergeMode::ZEROING>(xMainReg, xMainReg, xFoldReg, pMask);
                    StoreOneTensor<float>(xMainAddr, xMainReg, pMask, offset);
                }
            }
        }
    }

    __aicore__ inline void ProcessSummation(int64_t loopIdx, uint32_t rLength, uint32_t aLength)
    {
        int64_t cacheId = GetCacheID(loopIdx);
        uint32_t srcShape[2] = {rLength, static_cast<uint32_t>(aFactorAlign_)};
        ReduceSum<float, Pattern::Reduce::RA, true>(dbetaWsTensor_, dyMainTensor_, srcShape, false);
        ReduceSum<float, Pattern::Reduce::RA, true>(dgammaWsTensor_, xMainTensor_, srcShape, false);
        // 当前的dbeta, dgamma保存到中，读取Cache中的Tensor， 更新到cacheTensor中
        UpdateCache(dbetaCacheTensor_, dbetaWsTensor_, cacheId, aLength);
        UpdateCache(dgammaCacheTensor_, dgammaWsTensor_, cacheId, aLength);
    }

    __aicore__ inline int64_t GetCacheID(const int64_t idx)
    {
        return ScalarGetCountOfValue<1>(idx ^ (idx + 1)) - 1;
    }

    __aicore__ inline void UpdateCache(
        const LocalTensor<float>& dstTensor, const LocalTensor<float>& srcTensor, const int64_t cacheIdLoop,
        const uint32_t aLength)
    {
        uint16_t outerLoopTimes = ops::CeilDiv(aLength, VL_FP32);
        uint16_t innerLoopTimes = cacheIdLoop;
        uint32_t outerLoopStride = VL_FP32;
        uint32_t innerLoopStride = aFactorAlign_;
        __local_mem__ float* dst = (__local_mem__ float*)dstTensor.GetPhyAddr();
        __local_mem__ float* src = (__local_mem__ float*)srcTensor.GetPhyAddr();

        __VEC_SCOPE__
        {
            uint32_t sreg = static_cast<uint32_t>(aLength);
            MicroAPI::RegTensor<float> srcReg, cacheReg;
            MicroAPI::MaskReg pMask;
            for (uint16_t i = 0; i < outerLoopTimes; ++i) {
                pMask = MicroAPI::UpdateMask<float>(sreg);
                DataCopy(srcReg, (__local_mem__ float*)src + i * outerLoopStride);
                for (uint16_t j = 0; j < innerLoopTimes; ++j) {
                    int32_t offset = i * outerLoopStride + j * innerLoopStride;
                    DataCopy(cacheReg, (__local_mem__ float*)dst + offset);
                    MicroAPI::Add<float, MicroAPI::MaskMergeMode::ZEROING>(srcReg, srcReg, cacheReg, pMask);
                }
                DataCopy(
                    (__local_mem__ float*)dst + i * outerLoopStride + cacheIdLoop * innerLoopStride, srcReg, pMask);
            }
        }
    }

    // stage1
    __aicore__ inline void Stage1InitBuffer()
    {
        pipe_->Reset();

        int64_t aDimSize = aFactorAlign_ * tilingData_->usedCoreNum * sizeof(float);
        pipe_->InitBuffer(dbetaWsInQue_, BUFFER_NUM, aDimSize);
        pipe_->InitBuffer(dgammaWsInQue_, BUFFER_NUM, aDimSize);
        pipe_->InitBuffer(dbetaWsOutQue_, BUFFER_NUM, aFactorAlign_ * sizeof(float));
        pipe_->InitBuffer(dgammaWsOutQue_, BUFFER_NUM, aFactorAlign_ * sizeof(float));
        pipe_->InitBuffer(dbetaOutQue_, BUFFER_NUM, aFactorAlign_ * sizeof(WEIGHT_TYPE));
        pipe_->InitBuffer(dgammaOutQue_, BUFFER_NUM, aFactorAlign_ * sizeof(WEIGHT_TYPE));
    }

    __aicore__ inline void LoadDbetaDgammaFromWs(
        const int64_t offset, const int64_t rowSize, const uint32_t colSize, const int64_t dstStride,
        const int64_t srcStride)
    {
        DataCopyExtParams copyParams;
        copyParams.blockCount = rowSize;
        copyParams.blockLen = colSize * sizeof(float);
        copyParams.srcStride = (srcStride - colSize) * sizeof(float);
        copyParams.dstStride = (dstStride - colSize) * sizeof(float) / BLOCK_SIZE;
        DataCopyPadExtParams<float> PadParam{false, 0, 0, 0};

        LocalTensor<float> dbetaWsTensor = dbetaWsInQue_.AllocTensor<float>();
        DataCopyPad<float, PaddingMode::Normal>(dbetaWsTensor, dbetaWorkSpace_[offset], copyParams, PadParam);
        dbetaWsInQue_.EnQue(dbetaWsTensor);

        LocalTensor<float> dgammaWsTensor = dgammaWsInQue_.AllocTensor<float>();
        DataCopyPad<float, PaddingMode::Normal>(dgammaWsTensor, dgammaWorkSpace_[offset], copyParams, PadParam);
        dgammaWsInQue_.EnQue(dgammaWsTensor);
    }

    __aicore__ inline void StoreDbetaDgammaToWs(const int64_t offset, const int64_t aLength)
    {
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = aLength * sizeof(float);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;

        LocalTensor<float> dbetaWsTensor = dbetaWsOutQue_.DeQue<float>();
        DataCopyPad<float, PaddingMode::Normal>(dbetaWorkSpace_[offset], dbetaWsTensor, copyParams);
        dbetaWsOutQue_.FreeTensor(dbetaWsTensor);

        LocalTensor<float> dgammaWsTensor = dgammaWsOutQue_.DeQue<float>();
        DataCopyPad<float, PaddingMode::Normal>(dgammaWorkSpace_[offset], dgammaWsTensor, copyParams);
        dgammaWsOutQue_.FreeTensor(dgammaWsTensor);
    }

    __aicore__ inline void ProcessDbetaDgamma()
    {
        for (int64_t i = 0; i < tilingData_->aLoopTimes; i++) {
            int64_t baseOffset = i * tilingData_->aFactor;
            int64_t aLength = (i == tilingData_->aLoopTimes - 1) ? tilingData_->aFactorTail : tilingData_->aFactor;
            LoadDbetaDgammaFromWs(baseOffset, tilingData_->usedCoreNum, aLength, aFactorAlign_, tilingData_->aDim);
            LocalTensor<float> dbetaWsTensor = dbetaWsInQue_.template DeQue<float>();
            LocalTensor<float> dgammaWsTensor = dgammaWsInQue_.template DeQue<float>();
            LocalTensor<float> dbetaTmpTensor = dbetaWsOutQue_.AllocTensor<float>();
            LocalTensor<float> dgammaTmpTensor = dgammaWsOutQue_.AllocTensor<float>();
            // workspace空间上的dbeta, dgamma reduce成一行
            uint32_t srcShape[2] = {
                static_cast<uint32_t>(tilingData_->usedCoreNum), static_cast<uint32_t>(aFactorAlign_)};
            ReduceSum<float, Pattern::Reduce::RA, true>(dbetaTmpTensor, dbetaWsTensor, srcShape, false);
            ReduceSum<float, Pattern::Reduce::RA, true>(dgammaTmpTensor, dgammaWsTensor, srcShape, false);
            dbetaWsInQue_.FreeTensor(dbetaWsTensor);
            dgammaWsInQue_.FreeTensor(dgammaWsTensor);

            LocalTensor<WEIGHT_TYPE> dbetaOutTensor = dbetaOutQue_.AllocTensor<WEIGHT_TYPE>();
            LocalTensor<WEIGHT_TYPE> dgammaOutTensor = dgammaOutQue_.AllocTensor<WEIGHT_TYPE>();
            if constexpr (IsSameType<WEIGHT_TYPE, float>::value) {
                DataCopy(dbetaOutTensor, dbetaTmpTensor, aFactorAlign_);
                DataCopy(dgammaOutTensor, dgammaTmpTensor, aFactorAlign_);
            } else {
                DbetaDgammaTypeConvers(aLength, dbetaTmpTensor, dgammaTmpTensor, dbetaOutTensor, dgammaOutTensor);
            }
            dbetaWsOutQue_.EnQue(dbetaTmpTensor);
            dgammaWsOutQue_.EnQue(dgammaTmpTensor);
            // 保存float32类型到dbeta, dgamma到workspace空间，计算dx时使用
            StoreDbetaDgammaToWs(baseOffset, aLength);
            dbetaOutQue_.EnQue(dbetaOutTensor);
            dgammaOutQue_.EnQue(dgammaOutTensor);
            StoreDbetaDgammaToGM(baseOffset, aLength);
        }
    }

    __aicore__ inline void DbetaDgammaTypeConvers(
        const uint32_t aLength, const LocalTensor<float> dbetaTmpTensor, const LocalTensor<float> dgammaTmpTensor,
        const LocalTensor<WEIGHT_TYPE> dbetaOutTensor, const LocalTensor<WEIGHT_TYPE> dgammaOutTensor)
    {
        __local_mem__ float* dbetaTmpLocal = (__local_mem__ float*)dbetaTmpTensor.GetPhyAddr();
        __local_mem__ float* dgammaTmpLocal = (__local_mem__ float*)dgammaTmpTensor.GetPhyAddr();
        __local_mem__ WEIGHT_TYPE* dbetaOutLocal = (__local_mem__ WEIGHT_TYPE*)dbetaOutTensor.GetPhyAddr();
        __local_mem__ WEIGHT_TYPE* dgammaOutLocal = (__local_mem__ WEIGHT_TYPE*)dgammaOutTensor.GetPhyAddr();

        uint16_t loopNum = ops::CeilDiv(static_cast<uint32_t>(aLength), VL_FP32);

        __VEC_SCOPE__
        {
            MaskReg pregMask;
            uint32_t sreg = aLength;
            RegTensor<float> regDbeta, regDgamma;
            for (uint16_t i = 0; i < loopNum; i++) {
                pregMask = MicroAPI::UpdateMask<float>(sreg);
                int32_t offset = i * VL_FP32;
                LoadOneTensor<float>(dbetaTmpLocal, regDbeta, pregMask, offset);
                LoadOneTensor<float>(dgammaTmpLocal, regDgamma, pregMask, offset);
                StoreOneTensor<WEIGHT_TYPE>(dbetaOutLocal, regDbeta, pregMask, offset);
                StoreOneTensor<WEIGHT_TYPE>(dgammaOutLocal, regDgamma, pregMask, offset);
            }
        }
    }

    __aicore__ inline void StoreDbetaDgammaToGM(int64_t offset, uint32_t aLength)
    {
        DataCopyExtParams copyOutParams;
        copyOutParams.blockCount = 1;
        copyOutParams.blockLen = aLength * sizeof(WEIGHT_TYPE);
        copyOutParams.srcStride = 0;
        copyOutParams.dstStride = 0;

        LocalTensor<WEIGHT_TYPE> dbetaOutTensor = dbetaOutQue_.DeQue<WEIGHT_TYPE>();
        DataCopyPad<WEIGHT_TYPE, PaddingMode::Normal>(dbetaGm_[offset], dbetaOutTensor, copyOutParams);
        dbetaOutQue_.FreeTensor(dbetaOutTensor);

        LocalTensor<WEIGHT_TYPE> dgammaOutTensor = dgammaOutQue_.DeQue<WEIGHT_TYPE>();
        DataCopyPad<WEIGHT_TYPE, PaddingMode::Normal>(dgammaGm_[offset], dgammaOutTensor, copyOutParams);
        dgammaOutQue_.FreeTensor(dgammaOutTensor);
    }

    // stage 2
    __aicore__ inline void Stage2InitBuffer()
    {
        pipe_->Reset();

        int64_t rDimSize = tilingData_->dxLoopFactor * aFactorAlign_;
        int64_t aDimSize = aFactorAlign_ * sizeof(float);
        pipe_->InitBuffer(dyInQue_, BUFFER_NUM, rDimSize * sizeof(DY_TYPE));
        pipe_->InitBuffer(xInQue_, BUFFER_NUM, rDimSize * sizeof(DY_TYPE));
        pipe_->InitBuffer(dxOutQue_, BUFFER_NUM, rDimSize * sizeof(DY_TYPE));
        pipe_->InitBuffer(meanInQue_, BUFFER_NUM, aDimSize);
        pipe_->InitBuffer(rstdInQue_, BUFFER_NUM, aDimSize);
        pipe_->InitBuffer(gammaInQue_, BUFFER_NUM, aDimSize);
        pipe_->InitBuffer(dbetaWsInQue_, BUFFER_NUM, aDimSize);
        pipe_->InitBuffer(dgammaWsInQue_, BUFFER_NUM, aDimSize);
    }

    __aicore__ inline void ProcessDX()
    {
        for (int64_t i = 0; i < tilingData_->aLoopTimes; i++) {
            int64_t baseOffset = i * tilingData_->aFactor;
            int64_t aLength = (i == tilingData_->aLoopTimes - 1) ? tilingData_->aFactorTail : tilingData_->aFactor;
            LoadMeanRstdToUb(baseOffset, aLength);
            meanTensor_ = meanInQue_.template DeQue<float>();
            rstdTensor_ = rstdInQue_.template DeQue<float>();
            LoadDbetaDgammaFromWs(baseOffset, 1, aLength, aFactorAlign_, aLength);
            LocalTensor<float> dbeta = dbetaWsInQue_.template DeQue<float>();
            LocalTensor<float> dgamma = dgammaWsInQue_.template DeQue<float>();
            LoadGamma(baseOffset, aLength);
            LocalTensor<WEIGHT_TYPE> gamma = gammaInQue_.template DeQue<WEIGHT_TYPE>();

            for (int64_t j = 0; j < dxLoopTimes_; j++) {
                int64_t offset = baseOffset + j * dxLoopFactor_ * tilingData_->aDim;
                int64_t rLength = (j == dxLoopTimes_ - 1) ? dxLoopTail_ : dxLoopFactor_;
                CalDxVF(offset, rLength, aLength, dbeta, dgamma, gamma);
                StoreDxToGM(offset, rLength, aLength, tilingData_->aDim, aFactorAlign_);
            }
            dbetaWsInQue_.FreeTensor(dbeta);
            dgammaWsInQue_.FreeTensor(dgamma);
            meanInQue_.FreeTensor(meanTensor_);
            rstdInQue_.FreeTensor(rstdTensor_);
            gammaInQue_.FreeTensor(gamma);
        }
    }

    __aicore__ inline void CalDxVF(
        const int64_t gmOffset, const uint16_t rLength, const uint16_t aLength, const LocalTensor<float>& dbetaTensor,
        const LocalTensor<float>& dgammaTensor, const LocalTensor<WEIGHT_TYPE>& gammaTensor)
    {
        LoadDyXToUb(gmOffset, rLength, aLength, aFactorAlign_, tilingData_->aDim);
        LocalTensor<DY_TYPE> dyTensor = dyInQue_.DeQue<DY_TYPE>();
        LocalTensor<DY_TYPE> xTensor = xInQue_.DeQue<DY_TYPE>();
        LocalTensor<DY_TYPE> dxTensor = dxOutQue_.template AllocTensor<DY_TYPE>();
        uint16_t outerLoopTimes = CeilDiv(aLength, VL_FP32);
        uint16_t innerLoopTimes = rLength;
        uint16_t outerLoopStride = VL_FP32;
        uint16_t innerLoopStride = aFactorAlign_;
        float reciprocal = reciprocal_;

        __VEC_SCOPE__
        {
            __local_mem__ DY_TYPE* dyAddr = (__local_mem__ DY_TYPE*)dyTensor.GetPhyAddr();
            __local_mem__ DY_TYPE* xAddr = (__local_mem__ DY_TYPE*)xTensor.GetPhyAddr();
            __local_mem__ float* meanAddr = (__local_mem__ float*)meanTensor_.GetPhyAddr();
            __local_mem__ float* rstdAddr = (__local_mem__ float*)rstdTensor_.GetPhyAddr();
            __local_mem__ WEIGHT_TYPE* gammaAddr = (__local_mem__ WEIGHT_TYPE*)gammaTensor.GetPhyAddr();
            __local_mem__ DY_TYPE* dxAddr = (__local_mem__ DY_TYPE*)dxTensor.GetPhyAddr();
            __local_mem__ float* dbetaAddr = (__local_mem__ float*)dbetaTensor.GetPhyAddr();
            __local_mem__ float* dgammaAddr = (__local_mem__ float*)dgammaTensor.GetPhyAddr();
            MicroAPI::MaskReg pMask;
            uint32_t count = aLength;
            MicroAPI::RegTensor<float> dyReg, xReg, meanReg, rstdReg, gammaReg, dxReg, dbetaReg, dgammaReg;
            for (uint16_t i = 0; i < outerLoopTimes; ++i) {
                pMask = MicroAPI::UpdateMask<float>(count);
                LoadOneTensor<float>(meanAddr, meanReg, pMask, i * outerLoopStride);
                LoadOneTensor<float>(rstdAddr, rstdReg, pMask, i * outerLoopStride);
                LoadOneTensor<WEIGHT_TYPE>(gammaAddr, gammaReg, pMask, i * outerLoopStride);
                LoadOneTensor<float>(dbetaAddr, dbetaReg, pMask, i * outerLoopStride);
                LoadOneTensor<float>(dgammaAddr, dgammaReg, pMask, i * outerLoopStride);
                for (uint16_t j = 0; j < innerLoopTimes; ++j) {
                    uint32_t offset = i * outerLoopStride + j * innerLoopStride;
                    LoadOneTensor<DY_TYPE>(dyAddr, dyReg, pMask, offset);
                    LoadOneTensor<DY_TYPE>(xAddr, xReg, pMask, offset);
                    MicroAPI::Sub<float, MicroAPI::MaskMergeMode::ZEROING>(xReg, xReg, meanReg, pMask);
                    MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(xReg, xReg, rstdReg, pMask);
                    MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(xReg, xReg, dgammaReg, pMask);
                    MicroAPI::Add<float, MicroAPI::MaskMergeMode::ZEROING>(xReg, xReg, dbetaReg, pMask);
                    MicroAPI::Muls<float, float, MicroAPI::MaskMergeMode::ZEROING>(xReg, xReg, reciprocal, pMask);
                    MicroAPI::Sub<float, MicroAPI::MaskMergeMode::ZEROING>(dyReg, dyReg, xReg, pMask);
                    MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(dxReg, rstdReg, gammaReg, pMask);
                    MicroAPI::Mul<float, MicroAPI::MaskMergeMode::ZEROING>(dxReg, dxReg, dyReg, pMask);
                    StoreOneTensor<DY_TYPE>(dxAddr, dxReg, pMask, offset);
                }
            }
        }
        dyInQue_.FreeTensor(dyTensor);
        xInQue_.FreeTensor(xTensor);
        dxOutQue_.EnQue(dxTensor);
    }

    __aicore__ inline void StoreDxToGM(
        const int64_t offset, const int64_t rowSize, const int64_t colSize, const int64_t dstStride,
        const int64_t srcStride)
    {
        DataCopyExtParams copyParams;
        copyParams.blockCount = rowSize;
        copyParams.blockLen = colSize * sizeof(DY_TYPE);
        copyParams.srcStride = (srcStride - colSize) * sizeof(DY_TYPE) / BLOCK_SIZE;
        copyParams.dstStride = (dstStride - colSize) * sizeof(DY_TYPE);

        LocalTensor<DY_TYPE> dxTensor = dxOutQue_.template DeQue<DY_TYPE>();
        DataCopyPad<DY_TYPE, PaddingMode::Normal>(dxGm_[offset], dxTensor, copyParams);
        dbetaOutQue_.FreeTensor(dxTensor);
    }

    __aicore__ inline void LoadDyXToUb(
        const int64_t offset, const uint32_t rowSize, const uint32_t colSize, const int64_t dstStride,
        const int64_t srcStride)
    {
        DataCopyExtParams params;
        params.blockCount = rowSize;
        params.blockLen = colSize * sizeof(DY_TYPE);
        params.srcStride = (srcStride - colSize) * sizeof(DY_TYPE);
        params.dstStride = (dstStride - colSize) * sizeof(DY_TYPE) / BLOCK_SIZE;
        DataCopyPadExtParams<DY_TYPE> padParam{false, 0, 0, 0};

        LocalTensor<DY_TYPE> dyTensor = dyInQue_.template AllocTensor<DY_TYPE>();
        DataCopyPad(dyTensor, dyGm_[offset], params, padParam);
        dyInQue_.EnQue(dyTensor);

        LocalTensor<DY_TYPE> xTensor = xInQue_.template AllocTensor<DY_TYPE>();
        DataCopyPad(xTensor, xGm_[offset], params, padParam);
        xInQue_.EnQue(xTensor);
    }

    __aicore__ inline void LoadMeanRstdToUb(const int64_t offset, const uint32_t aLength)
    {
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = aLength * sizeof(float);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPadExtParams<float> padParam{false, 0, 0, 0};

        meanTensor_ = meanInQue_.template AllocTensor<float>();
        DataCopyPad(meanTensor_, meanGm_[offset], copyParams, padParam);
        meanInQue_.EnQue(meanTensor_);

        rstdTensor_ = rstdInQue_.template AllocTensor<float>();
        DataCopyPad(rstdTensor_, rstdGm_[offset], copyParams, padParam);
        rstdInQue_.EnQue(rstdTensor_);
    }

    __aicore__ inline void LoadGamma(int64_t offset, uint32_t aLength)
    {
        DataCopyExtParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = aLength * sizeof(WEIGHT_TYPE);
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopyPadExtParams<WEIGHT_TYPE> PadParam{false, 0, 0, 0};
        LocalTensor<WEIGHT_TYPE> gammaTensor = gammaInQue_.template AllocTensor<WEIGHT_TYPE>();
        DataCopyPad(gammaTensor, gammaGm_[offset], copyParams, PadParam);
        gammaInQue_.EnQue(gammaTensor);
    }

private:
    const BatchNormGradV3RASplitRTilingData* tilingData_{nullptr};
    TPipe* pipe_{nullptr};

    GlobalTensor<DY_TYPE> dyGm_, xGm_;
    GlobalTensor<float> meanGm_, rstdGm_;
    GlobalTensor<float> dbetaWorkSpace_, dgammaWorkSpace_; // workspace
    GlobalTensor<WEIGHT_TYPE> gammaGm_, dgammaGm_, dbetaGm_;
    GlobalTensor<DY_TYPE> dxGm_;
    LocalTensor<DY_TYPE> dyFoldTensor_, xFoldTensor_;
    LocalTensor<float> xMainTensor_, dyMainTensor_;
    LocalTensor<float> meanTensor_, rstdTensor_;
    LocalTensor<float> dbetaWsTensor_, dgammaWsTensor_;
    LocalTensor<float> dbetaCacheTensor_, dgammaCacheTensor_;

    TQue<QuePosition::VECIN, 1> dyInQue_, xInQue_, meanInQue_, rstdInQue_;
    TBuf<TPosition::VECCALC> dyTmpQue_, xTmpQue_;
    TBuf<TPosition::VECCALC> dbetaCacheBuffer_, dgammaCacheBuffer_;
    TQue<QuePosition::VECOUT, 1> dbetaWsOutQue_, dgammaWsOutQue_;
    TQue<QuePosition::VECIN, 1> dbetaWsInQue_, dgammaWsInQue_;
    TQue<QuePosition::VECOUT, 1> dbetaOutQue_, dgammaOutQue_;
    TQue<QuePosition::VECIN, 1> gammaInQue_;
    TQue<QuePosition::VECOUT, 1> dxOutQue_;

    static constexpr int64_t ULONG_BIT_LEN = 64;
    static constexpr int32_t BUFFER_NUM = 2;
    static constexpr uint32_t BLOCK_SIZE = platform::GetUbBlockSize();

    uint32_t resultCacheId_{0};
    int64_t blockIdx_{0};
    int64_t currBlockFactor_{0};
    int64_t currLoopFactor_{0};
    int64_t binaryBlockCnt_{0};
    int64_t binaryFoldPoint_{0};
    int64_t binaryBlockTail_{0};
    int64_t dxLoopFactor_{0};
    int64_t dxLoopTail_{0};
    int64_t dxLoopTimes_{0};
    int64_t aFactorAlign_{0};
    float reciprocal_{0.0};
};
} // namespace BatchNormGradV3
#endif // __BATCH_NORM_GRAD_V3_RA_SPLIT_R_REGBASE_H__