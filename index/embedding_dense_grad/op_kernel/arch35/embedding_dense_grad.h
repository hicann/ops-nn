/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file embedding_dense_grad.h
 * \brief embedding_dense_grad
 */

#ifndef EMBEDDING_DENSE_GRAD_H
#define EMBEDDING_DENSE_GRAD_H

#include <cstdint>
#include "kernel_operator.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"

namespace EmbeddingDenseGrad {
using namespace AscendC;
using namespace ops;
using namespace EmbeddingDenseGradBase;

constexpr int32_t NDDMA_DIM = 5;

template <typename T, typename U, typename VGatherIndexDType>
class KernelEmbeddingDenseGrad : public EmbeddingDenseGradProcessIndices<U> {
public:
    __aicore__ inline KernelEmbeddingDenseGrad(TPipe& pipe, const EmbeddingDenseGradSimdTilingData& tilingData) : pipe_(pipe), tiling_(tilingData){};

    __aicore__ inline void InitGmBuffer(GM_ADDR grad, GM_ADDR indices, GM_ADDR gradWeight, GM_ADDR workspace)
    {
        uint64_t initPerCore = (tiling_.embeddingDim * tiling_.numWeights + tiling_.clearBlock - 1) / tiling_.clearBlock;
        uint64_t initLastCore = tiling_.embeddingDim * tiling_.numWeights - (tiling_.clearBlock - 1) * initPerCore;
        uint64_t initCoreReal = blockId_ == (tiling_.clearBlock - 1) ? initLastCore : initPerCore;

        if (blockId_ >= tiling_.clearBlock) {
            return;
        }

        AscendC::GlobalTensor<T> gradWeightGmInit;
        AscendC::GlobalTensor<U> countsGmInit;
        AscendC::GlobalTensor<float> compute32GmInit;
        gradWeightGmInit.SetGlobalBuffer((__gm__ T *)gradWeight + blockId_ * initPerCore);
        countsGmInit.SetGlobalBuffer((__gm__ U *)workspace, tiling_.numWeights);
        int64_t offset = tiling_.numWeights * sizeof(U) + blockId_ * initPerCore * sizeof(float);
        compute32GmInit.SetGlobalBuffer((__gm__ float *)((__gm__ uint8_t *)workspace + offset), initCoreReal);

        if (0 == blockId_) {
            if (scaleGradByFreq_) {
                InitGlobalMemory(countsGmInit, tiling_.numWeights, (U)0);
            }
        }
        InitGlobalMemory(gradWeightGmInit, initCoreReal, (T)0);
        if constexpr (!is_same<float, T>::value) {
            InitGlobalMemory(compute32GmInit, initCoreReal, (float)0);
        }
        if (scaleGradByFreq_) {
            if constexpr (is_same<float, T>::value) {
                InitGlobalMemory(compute32GmInit, initCoreReal, (float)0);
            }
        }
    }

    __aicore__ inline void PipeInitBuffer() {
        gradFactorPerRowAlign_ = Aligned(gradFactorPerRow_, b32AlignNum_);
        pipe_.InitBuffer(inQueueGrad, 1, tiling_.gradFactor * sizeof(T));
        pipe_.InitBuffer(outQueueRes_, 1, gradFactorPerRowAlign_ * tiling_.indicesFactor * sizeof(float));
    }

    __aicore__ inline void Init(GM_ADDR grad, GM_ADDR indices, GM_ADDR gradWeight, GM_ADDR workspace)
    {
        gradGm_.SetGlobalBuffer((__gm__ T *)(grad));
        gradWeightGm_.SetGlobalBuffer((__gm__ T *)(gradWeight));

        loopPerCoreGrad_ = tiling_.loopPerCoreGrad;
        gradFactorPerRow_ = tiling_.gradFactorPerRow;
        gradFactorPerRowTail_ = tiling_.gradFactorPerRowTail;
        scaleGradByFreq_ = static_cast<int32_t>(tiling_.scaleGradByFreq);
        if (scaleGradByFreq_) {
            loopPerCoreGradFreq_ = tiling_.loopPerCoreGradFreq;
            gradFactorPerRowFreq_ = tiling_.gradFactorPerRowFreq;
            gradFactorPerRowTailFreq_ = tiling_.gradFactorPerRowTailFreq;
        } else {
            if constexpr (!is_same<T, float>::value) {
                baseACast_ = tiling_.baseACast;
                baseWCast_ = tiling_.baseWCast;
                cntACast_ = tiling_.cntACast;
                cntWCast_ = tiling_.cntWCast;
                tailACast_ = tiling_.tailACast;
                tailWCast_ = tiling_.tailWCast;
            }
        }

        blockId_ = GetBlockIdx();
        blockNums_ = GetBlockNum();
        if (blockId_ == (tiling_.processBlock - 1)) {
            gradFactorPerRow_ = tiling_.embeddingDimLastCore < gradFactorPerRow_ ? tiling_.embeddingDimLastCore : gradFactorPerRow_;
            loopPerCoreGrad_ = (tiling_.embeddingDimLastCore + gradFactorPerRow_ - 1) / gradFactorPerRow_;
            gradFactorPerRowTail_ = tiling_.embeddingDimLastCore - (loopPerCoreGrad_ - 1) * gradFactorPerRow_;
            if (scaleGradByFreq_) {
                gradFactorPerRowFreq_ = \
                    tiling_.embeddingDimLastCore < gradFactorPerRowFreq_ ? tiling_.embeddingDimLastCore : gradFactorPerRowFreq_;
                loopPerCoreGradFreq_ = (tiling_.embeddingDimLastCore + gradFactorPerRowFreq_ - 1) / gradFactorPerRowFreq_;
                gradFactorPerRowTailFreq_ = tiling_.embeddingDimLastCore - (loopPerCoreGradFreq_ - 1) * gradFactorPerRowFreq_;
            } else {
                if constexpr (!is_same<T, float>::value) {
                    baseACast_ = tiling_.embeddingDimLastCore < baseACast_ ? tiling_.embeddingDimLastCore : baseACast_;
                    cntACast_ = (tiling_.embeddingDimLastCore + baseACast_ - 1) / baseACast_;
                    tailACast_ = tiling_.embeddingDimLastCore - (cntACast_ - 1) * baseACast_;
                }
            }
        }

        countsGm_.SetGlobalBuffer((__gm__ U *)workspace, tiling_.numWeights);
        compute32Gm_.SetGlobalBuffer((__gm__ float *)((__gm__ uint8_t *)workspace + tiling_.numWeights * sizeof(U)), tiling_.embeddingDim * tiling_.numWeights);

        InitGmBuffer(grad, indices, gradWeight, workspace);
        PipeBarrier<PIPE_ALL>();
        SyncAll();

        this->IndicesInitBuffer(pipe_, tiling_.indicesFactor, tiling_.sortSharedBufSize, indices);
        PipeInitBuffer();
    }

    __aicore__ inline void InitBufferFreq()
    {
        pipe_.Reset();
        uint32_t gradFactorPerRowFreqAlign = Aligned(gradFactorPerRowFreq_, blockAlignNum_);
        uint32_t resBufFactorFreq = gradFactorPerRowFreqAlign * tiling_.indicesFactorFreq;

        pipe_.InitBuffer(outQueueGradWeight, 1, resBufFactorFreq * sizeof(float));
        pipe_.InitBuffer(gradLocalFreBuf_, resBufFactorFreq * sizeof(float));
        pipe_.InitBuffer(indiceFreBuf_, Aligned(static_cast<uint32_t>(tiling_.indicesFactorFreq * sizeof(U)), platform::GetUbBlockSize()));
    }

    __aicore__ inline void InitBufferCast()
    {
        pipe_.Reset();
        baseACastAlign_ = Aligned(baseACast_, blockAlignNum_);
        uint32_t castBufFactor = baseWCast_ * baseACastAlign_;

        pipe_.InitBuffer(inQueueGrad, 1, castBufFactor * sizeof(float));
        pipe_.InitBuffer(outQueueGradWeight, 1, castBufFactor * sizeof(T));
    }

    __aicore__ inline void DupSegmentProcess(uint32_t aFactorIdx, uint32_t gradFactorPerRowReal, int64_t arNum,
                                      const LocalTensor<int32_t> &noDupResTensor)
    {
        LocalTensor<int32_t> noDupResProcessLoop = this->noDupProcessLoopBuf_.template Get<int32_t>();
        LocalTensor<uint32_t> sortedIndiceIndex = this->sortedIndiceIndexBuf_.template Get<uint32_t>();
        LocalTensor<int32_t> noDupRes = noDupResTensor;
        LocalTensor<float> resLocal = outQueueRes_.AllocTensor<float>();
        LocalTensor<T> gradLocal = inQueueGrad.DeQue<T>();
        __local_mem__ T * gradLocalAddr = (__ubuf__ T *)gradLocal.GetPhyAddr();
        __local_mem__ float *resBufAddr = (__ubuf__ float *)resLocal.GetPhyAddr();
        __local_mem__ float *resBufBaseAddr = resBufAddr;

        int32_t sclar0 = 0;

        uint32_t loopPerGradRow = (gradFactorPerRowReal + vfLengthFp32_ - 1) / vfLengthFp32_;
        uint32_t colCount = gradFactorPerRowReal;
        uint32_t ubAddrUpdate = 0;

        GroupAddParams gradParams;
        gradParams.gradPerRowNum = gradFactorPerRowReal;
        gradParams.gradWeightGmindex = 0;
        gradParams.sortedIndiceIndexAddr = (__ubuf__ uint32_t *)sortedIndiceIndex.GetPhyAddr();
        gradParams.ubOutOffset = 0;

        __VEC_SCOPE__
        {
            for (uint16_t i = 0; i < (uint16_t)arNum; ++i) {
                MicroAPI::RegTensor<int32_t> serReg, serRegBase;

                MicroAPI::Arange(serRegBase, sclar0);

                gradParams.segCount = static_cast<uint32_t>(noDupRes(i));
                gradParams.loopSegCount = static_cast<uint32_t>(noDupResProcessLoop(i));
                gradParams.loopSegCoutTail = gradParams.segCount - gradParams.loopSegCount  * PROCESS_GROUP;
                gradParams.ubOutOffset = 0;
                colCount = gradFactorPerRowReal;
                resBufAddr = resBufBaseAddr + i * gradFactorPerRowAlign_;

                for (uint16_t j = 0; j < (uint16_t)loopPerGradRow; ++j) {
                    MicroAPI::MaskReg maskRegUpdate = MicroAPI::UpdateMask<uint32_t>(colCount);
                    MicroAPI::Adds(serReg, serRegBase, perVfIndicesIdx_ * j, maskRegUpdate);
                    this->template ProcessPerGradGroup<T, float, VGatherIndexDType>(gradLocalAddr,
                                                                    resBufAddr, maskRegUpdate, serReg,  gradParams);                   
                    gradParams.ubOutOffset = gradParams.ubOutOffset + vfLengthFp32_;
                }
                gradParams.gradWeightGmindex = gradParams.segCount + gradParams.gradWeightGmindex;
            }
        }
        inQueueGrad.FreeTensor<T>(gradLocal);
        outQueueRes_.EnQue(resLocal);
    }

    __aicore__ inline void CopyOutProcess(uint32_t aFactorIdx, uint32_t gradFactorPerRowReal, int64_t &arNum,
                                   const LocalTensor<U> &sortedIndice, const LocalTensor<int32_t> &noDupRes)
    {
        LocalTensor<float> resLocal = outQueueRes_.DeQue<float>();

        int32_t gradWeightGmOutIndex = shiftOffset_;
        int64_t gradOffsetInterval = aFactorIdx * gradFactorPerRow_ + blockId_ * tiling_.embeddingDimPerCore;
        DataCopyExtParams dataCopyParams {
            static_cast<uint16_t>(1),
            static_cast<uint32_t>(gradFactorPerRowReal * sizeof(float)),
            static_cast<uint32_t>(0),
            static_cast<uint32_t>(0),
            static_cast<uint32_t>(0)
        };
        SetAtomicAdd<float>();
        for (uint32_t i = 0; i < static_cast<uint32_t>(arNum); ++i) {
            int64_t gradWeightGmIndice = static_cast<U>(sortedIndice(gradWeightGmOutIndex));
            int64_t gradWeightGmOffset = gradWeightGmIndice * tiling_.embeddingDim + gradOffsetInterval;
            gradWeightGmOutIndex = gradWeightGmOutIndex + noDupRes(i);

            if (gradWeightGmIndice != tiling_.paddingIdx && gradWeightGmIndice >= 0 && gradWeightGmIndice < tiling_.numWeights) {
                uint32_t ubOutAddrUpdate = i * gradFactorPerRowAlign_;
                if constexpr (is_same<float, T>::value) {
                    if (scaleGradByFreq_) {
                        DataCopyPad(compute32Gm_[gradWeightGmOffset], resLocal[ubOutAddrUpdate], dataCopyParams);
                    } else {
                        DataCopyPad(gradWeightGm_[gradWeightGmOffset], resLocal[ubOutAddrUpdate], dataCopyParams);
                    }
                } else {
                    DataCopyPad(compute32Gm_[gradWeightGmOffset], resLocal[ubOutAddrUpdate], dataCopyParams);
                }
            }
        }
        SetAtomicNone();

        outQueueRes_.FreeTensor<float>(resLocal);
    }

    __aicore__ inline void PrepareFreqBeforeCopyToWs(const LocalTensor<U> &resLocal, const LocalTensor<int32_t> &noDupRes, int64_t arNum)
    {
        auto resultAddr = (__ubuf__ U *)resLocal.GetPhyAddr();
        auto noDupResAddr = (__ubuf__ int32_t *)noDupRes.GetPhyAddr();

        uint32_t uniqueIndexNum = static_cast<uint32_t>(arNum);
        uint16_t loopNum = ops::Ceil(uniqueIndexNum, perVfIndicesIdx_);
        int32_t interval = b32AlignNum_;
        uint32_t resultOffset = static_cast<uint32_t>(interval) * perVfIndicesIdx_;

        __VEC_SCOPE__
        {
            MicroAPI::RegTensor<int32_t> freqReg, indexDstReg;
            MicroAPI::MaskReg maskFull = MicroAPI::CreateMask<int32_t, AscendC::MicroAPI::MaskPattern::ALL>();
            MicroAPI::Arange(indexDstReg, (int32_t)0);
            MicroAPI::Muls(indexDstReg, indexDstReg, interval, maskFull);

            for (uint16_t i = 0; i < loopNum; ++i) {
                MicroAPI::MaskReg mask = MicroAPI::UpdateMask<uint32_t>(uniqueIndexNum);
                MicroAPI::DataCopy<int32_t, MicroAPI::PostLiteral::POST_MODE_UPDATE>(freqReg, noDupResAddr, perVfIndicesIdx_);
                MicroAPI::DataCopyScatter(resultAddr, freqReg, (MicroAPI::RegTensor<uint32_t> &)indexDstReg, mask);

                resultAddr = resultAddr + resultOffset;
            }
        }

        outQueueRes_.EnQue<U>(resLocal);
    }

    __aicore__ inline void CopyOutIndicesForFreq(const LocalTensor<U> &sortedIndice, const LocalTensor<int32_t> &noDupRes,
                                          int32_t indicesFactorReal, int64_t &arNum)
    {
        LocalTensor<U> tmpBufInt32 = outQueueRes_.AllocTensor<U>();
        if constexpr (is_same<int32_t, U>::value) {
            PrepareFreqBeforeCopyToWs(tmpBufInt32, noDupRes, arNum);
            SetAtomicAdd<int32_t>();
            tmpBufInt32 = outQueueRes_.DeQue<U>();
        }

        int32_t gradWeightGmOutIndex = shiftOffset_;
        DataCopyParams dataCopyParams{(uint16_t)1, (uint16_t)(1 * sizeof(U)), 0, 0};
        for (uint16_t i = 0; i < arNum; ++i) {
            U gradWeightGmIndice = sortedIndice(gradWeightGmOutIndex);
            if (gradWeightGmIndice != tiling_.paddingIdx && gradWeightGmIndice >= 0 && gradWeightGmIndice < tiling_.numWeights) {
                if constexpr (is_same<int32_t, U>::value) {
                    DataCopyPad(countsGm_[gradWeightGmIndice], tmpBufInt32[i * b32AlignNum_], dataCopyParams);
                } else {
                    __gm__ U *freqGmAddr = countsGm_.GetPhyAddr(gradWeightGmIndice);
                    AtomicAdd(freqGmAddr, static_cast<U>(noDupRes(i)));
                    DataCacheCleanAndInvalid<U, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(countsGm_);
                }
            }
            gradWeightGmOutIndex = gradWeightGmOutIndex + noDupRes(i);
        }

        if constexpr (is_same<int32_t, U>::value) {
            SetAtomicNone();
        }
        outQueueRes_.FreeTensor<U>(tmpBufInt32);
    }

    __aicore__ inline void CopyGradFromWs(uint64_t gmOffset, uint32_t indicesNum, uint32_t gradPerRowNum)
    {
        LocalTensor<float> gradFp32Local = inQueueGrad.AllocTensor<float>();

        DataCopyExtParams copyExtParamsGrad;
        copyExtParamsGrad.blockCount = static_cast<uint16_t>(indicesNum);
        copyExtParamsGrad.blockLen = static_cast<uint32_t>(gradPerRowNum * sizeof(float));
        copyExtParamsGrad.srcStride = \
            static_cast<uint32_t>((tiling_.embeddingDim - gradPerRowNum) * sizeof(float));
        copyExtParamsGrad.dstStride = 0;
        DataCopyPadExtParams<float> padParamsGrad{false, 0, 0, 0};

        DataCopyPad(gradFp32Local, compute32Gm_[gmOffset], copyExtParamsGrad, padParamsGrad);
        inQueueGrad.EnQue(gradFp32Local);
    }

    __aicore__ inline void CopyGradToGm(uint64_t gmOffset, uint32_t indicesNum, uint32_t gradPerRowNum)
    {
        LocalTensor<T> gradOutLocal = outQueueGradWeight.DeQue<T>();

        DataCopyExtParams extParams {
            static_cast<uint16_t>(indicesNum),                                   // blockCount
            static_cast<uint32_t>(gradPerRowNum * sizeof(T)),                      // blockLen
            0,                                                                  // srcStride
            static_cast<uint32_t>((tiling_.embeddingDim - gradPerRowNum) * sizeof(T)),    // dstStride
            0                                                                   // rsv
        };
        DataCopyPad(gradWeightGm_[gmOffset], gradOutLocal, extParams);
        outQueueGradWeight.FreeTensor(gradOutLocal);
    }

    __aicore__ inline void ExecGradConvert(uint32_t indicesNum, uint32_t gradPerRowNum)
    {
        LocalTensor<float> gradFp32Local = inQueueGrad.DeQue<float>();
        LocalTensor<T> gradOutLocal = outQueueGradWeight.AllocTensor<T>();
        __local_mem__ float * gradFp32BaseAddr = (__ubuf__ float *)gradFp32Local.GetPhyAddr();
        __local_mem__ T *resBufBaseAddr = (__ubuf__ T *)gradOutLocal.GetPhyAddr();

        uint16_t loopPerRow = ops::Ceil(gradPerRowNum, vfLengthFp32_);
        uint32_t gradPerRowAlign = ops::CeilAlign(gradPerRowNum, b32AlignNum_);
        uint32_t resultPerRowAlign = baseACastAlign_;

        __VEC_SCOPE__ {
            MicroAPI::RegTensor<float> srcReg;
            MicroAPI::MaskReg maskUpdate;
            for (uint16_t i = 0; i < static_cast<uint16_t>(indicesNum); i++) {
                uint32_t colCount = gradPerRowNum;
                __local_mem__ float * gradFp32Addr = gradFp32BaseAddr + i * gradPerRowAlign;
                __local_mem__ T *resBufAddr = resBufBaseAddr + i * resultPerRowAlign;
                uint32_t offset = 0;
                for (uint16_t j = 0; j < loopPerRow; j++) {
                    maskUpdate = MicroAPI::UpdateMask<int32_t>(colCount);
                    ops::LoadOneTensorForDtypeT(gradFp32Addr, srcReg, maskUpdate, offset);
                    ops::StoreOneTensorForDtypeT(resBufAddr, srcReg, maskUpdate, offset);
                    offset = offset + vfLengthFp32_;
                }
            }
        }

        outQueueGradWeight.EnQue<T>(gradOutLocal);
        inQueueGrad.FreeTensor(gradFp32Local);
    }

    __aicore__ inline void ComputeCastMulti()
    {
        uint32_t gradProcessNumPerLoop = baseACastAlign_ * baseWCast_;
        uint64_t gradWeightGmOffset = blockId_ * tiling_.embeddingDimPerCore;
        uint64_t offsetPerLoop = baseWCast_ * tiling_.embeddingDim;
        for (uint64_t weightIdx = 0; weightIdx < cntWCast_ - 1; weightIdx++) {
            CopyGradFromWs(gradWeightGmOffset, baseWCast_, baseACast_);
            ExecGradConvert(baseWCast_, baseACast_);
            CopyGradToGm(gradWeightGmOffset, baseWCast_, baseACast_);
            gradWeightGmOffset = gradWeightGmOffset + offsetPerLoop;
        }

        CopyGradFromWs(gradWeightGmOffset, tailWCast_, baseACast_);
        ExecGradConvert(tailWCast_, baseACast_);
        CopyGradToGm(gradWeightGmOffset, tailWCast_, baseACast_);
    }

    __aicore__ inline void ComputeCastSpilt()
    {
        uint32_t indiceNumPerLoop = 1;
        for (uint64_t weightIdx = 0; weightIdx < tiling_.numWeights; weightIdx++) {
            uint64_t gradWeightGmOffset = weightIdx * tiling_.embeddingDim + blockId_ * tiling_.embeddingDimPerCore;

            for (uint64_t aDimIdx = 0; aDimIdx < cntACast_ - 1; aDimIdx++) {                
                CopyGradFromWs(gradWeightGmOffset, indiceNumPerLoop, baseACast_);

                LocalTensor<float> gradFp32Local = inQueueGrad.DeQue<float>();
                LocalTensor<T> gradOutLocal = outQueueGradWeight.AllocTensor<T>();
                Cast(gradOutLocal, gradFp32Local, RoundMode::CAST_RINT, baseACast_);
                outQueueGradWeight.EnQue<T>(gradOutLocal);
                inQueueGrad.FreeTensor(gradFp32Local);
            
                CopyGradToGm(gradWeightGmOffset, indiceNumPerLoop, baseACast_);
                gradWeightGmOffset = gradWeightGmOffset + baseACast_;
            }

            CopyGradFromWs(gradWeightGmOffset, indiceNumPerLoop, tailACast_);

            LocalTensor<float> gradFp32Local = inQueueGrad.DeQue<float>();
            LocalTensor<T> gradOutLocal = outQueueGradWeight.AllocTensor<T>();
            Cast(gradOutLocal, gradFp32Local, RoundMode::CAST_RINT, tailACast_);
            outQueueGradWeight.EnQue<T>(gradOutLocal);
            inQueueGrad.FreeTensor(gradFp32Local);
        
            CopyGradToGm(gradWeightGmOffset, indiceNumPerLoop, tailACast_);
        }
    }

    __aicore__ inline void ComputeFinalValueWithFreqProcess(LocalTensor<U> &indiceFreqBuf, LocalTensor<float> &gradLocalFreBuf,
                                                     uint64_t gradOffset, uint32_t weightPerRowFreqReal,
                                                     uint32_t weightPerRowFreqDtypeAlign, uint32_t indicesFactorFreqReal)
    {
        event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));

        uint32_t gradWeightGmOutFreindex = 0;
        uint32_t gradWeightGmFreIndice = 0;
        uint32_t gradWeightGmFreOffset = 0;
        uint32_t ubOffset = 0;
        uint32_t ubAddrUpdate = 0;
        gradLocalFreq = outQueueGradWeight.AllocTensor<float>();
        uint32_t indicesNumber = indicesFactorFreqReal * weightPerRowFreqDtypeAlign;
        Duplicate(gradLocalFreq, 0.0f, indicesNumber);
        event_t eventMTE2_S= static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(eventMTE2_S);
        WaitFlag<HardEvent::MTE2_S>(eventMTE2_S);

        for (uint32_t weightIdx = 0; weightIdx < indicesFactorFreqReal; weightIdx++) {
            auto indiceFreqNum = indiceFreqBuf(weightIdx);
            if (indiceFreqNum != 0) {
                float freqNum = 1.0f / indiceFreqNum;
                uint32_t weightOffset = weightIdx * weightPerRowFreqDtypeAlign;
                Muls(gradLocalFreq[weightOffset], gradLocalFreBuf[weightOffset], freqNum,
                     weightPerRowFreqReal);
            }
        }

        if constexpr (!is_same<T, float>::value) {
            LocalTensor<T> gradWeightLocalB16 = gradLocalFreq.ReinterpretCast<T>();
            Cast(gradWeightLocalB16, gradLocalFreq, RoundMode::CAST_RINT, indicesNumber);
        }
        outQueueGradWeight.EnQue<float>(gradLocalFreq);
        LocalTensor<float> gradWeightLocalOut = outQueueGradWeight.DeQue<float>();
        LocalTensor<T> gradWeightLocalOutB16 = gradWeightLocalOut.template ReinterpretCast<T>();

        DataCopyExtParams copyExtParamsFre;
        copyExtParamsFre.blockCount = static_cast<uint16_t>(indicesFactorFreqReal);
        copyExtParamsFre.blockLen = static_cast<uint32_t>(weightPerRowFreqReal * sizeof(T));
        copyExtParamsFre.srcStride = 0;
        copyExtParamsFre.dstStride = static_cast<uint32_t>((tiling_.embeddingDim - weightPerRowFreqReal) * sizeof(T));
        DataCopyPad<T>(gradWeightGm_[gradOffset], gradWeightLocalOutB16, copyExtParamsFre);
        outQueueGradWeight.FreeTensor<float>(gradWeightLocalOut);
    }

    __aicore__ inline void ComputeFinalValueWithFreq()
    {
        int64_t arNum = 0;
        LocalTensor<U> indiceFreqBuf = indiceFreBuf_.Get<U>();
        LocalTensor<float> gradLocalFreBuf = gradLocalFreBuf_.Get<float>();

        DataCopyExtParams copyExtParamsIndicesMain;
        copyExtParamsIndicesMain.blockCount = static_cast<uint16_t>(1);
        copyExtParamsIndicesMain.blockLen = static_cast<uint32_t>(tiling_.indicesFactorFreq * sizeof(U));
        copyExtParamsIndicesMain.srcStride = 0;
        copyExtParamsIndicesMain.dstStride = 0;
        DataCopyPadExtParams<U> padParamsIndices{false, 0, 0, 0};
        DataCopyPadExtParams<float> padParamsWeight{false, 0, 0, 0};
        event_t eventMTE2_V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));

        // Main
        for (uint32_t i = 0; i < tiling_.loopPerCoreIndiceFreq - 1; i++) {
            int64_t indicesOffset = i * tiling_.indicesFactorFreq;
            DataCopyPad<U>(indiceFreqBuf, countsGm_[indicesOffset], copyExtParamsIndicesMain, padParamsIndices);
            for (uint32_t j = 0; j < loopPerCoreGradFreq_; j++) {
                uint32_t weightPerRowFreqReal = \
                    j != (loopPerCoreGradFreq_ - 1) ? gradFactorPerRowFreq_ : gradFactorPerRowTailFreq_;
                uint32_t weightPerRowFreqDtypeAlign = Aligned(weightPerRowFreqReal, blockAlignNum_);
                uint32_t weightPerRowFreqB32Align = Aligned(weightPerRowFreqReal, BLOCK_ALIGN_B32);
                uint64_t gradOffset = indicesOffset * tiling_.embeddingDim +
                              blockId_ * tiling_.embeddingDimPerCore + j * gradFactorPerRowFreq_;
                DataCopyExtParams copyExtParamsWeight;
                copyExtParamsWeight.blockCount = static_cast<uint16_t>(tiling_.indicesFactorFreq);
                copyExtParamsWeight.blockLen = static_cast<uint32_t>(weightPerRowFreqReal * sizeof(float));
                copyExtParamsWeight.srcStride = \
                    static_cast<uint32_t>((tiling_.embeddingDim - weightPerRowFreqReal) * sizeof(float));
                copyExtParamsWeight.dstStride = (weightPerRowFreqDtypeAlign - weightPerRowFreqB32Align) / BLOCK_ALIGN_B32;
                DataCopyPad<float>(gradLocalFreBuf, compute32Gm_[gradOffset], copyExtParamsWeight, padParamsWeight);
                SetFlag<HardEvent::MTE2_V>(eventMTE2_V);
                WaitFlag<HardEvent::MTE2_V>(eventMTE2_V);
                ComputeFinalValueWithFreqProcess(indiceFreqBuf, gradLocalFreBuf, gradOffset,
                                                 weightPerRowFreqReal, weightPerRowFreqDtypeAlign, tiling_.indicesFactorFreq);
            }
        }
        // Tail
        int64_t indicesOffset = (tiling_.loopPerCoreIndiceFreq - 1) * tiling_.indicesFactorFreq;
        DataCopyExtParams copyExtParamsIndices;
        copyExtParamsIndices.blockCount = static_cast<uint16_t>(1);
        copyExtParamsIndices.blockLen = static_cast<uint32_t>(tiling_.indicesFactorFreqTail * sizeof(U));
        copyExtParamsIndices.srcStride = 0;
        copyExtParamsIndices.dstStride = 0;
        DataCopyPad<U>(indiceFreqBuf, countsGm_[indicesOffset], copyExtParamsIndices, padParamsIndices);
        for (uint32_t j = 0; j < loopPerCoreGradFreq_; j++) {
            uint32_t weightPerRowFreqReal = \
                j != (loopPerCoreGradFreq_ - 1) ? gradFactorPerRowFreq_ : gradFactorPerRowTailFreq_;
            uint32_t weightPerRowFreqDtypeAlign = Aligned(weightPerRowFreqReal, blockAlignNum_);
            uint32_t weightPerRowFreqB32Align = Aligned(weightPerRowFreqReal, BLOCK_ALIGN_B32);
            uint64_t gradOffset = indicesOffset * tiling_.embeddingDim +
                          blockId_ * tiling_.embeddingDimPerCore + j * gradFactorPerRowFreq_;
            DataCopyExtParams copyExtParamsWeight;
            copyExtParamsWeight.blockCount = static_cast<uint16_t>(tiling_.indicesFactorFreqTail);
            copyExtParamsWeight.blockLen = static_cast<uint32_t>(weightPerRowFreqReal * sizeof(float));
            copyExtParamsWeight.srcStride = \
                static_cast<uint32_t>((tiling_.embeddingDim - weightPerRowFreqReal) * sizeof(float));
            copyExtParamsWeight.dstStride = (weightPerRowFreqDtypeAlign - weightPerRowFreqB32Align) / BLOCK_ALIGN_B32;
            DataCopyPad<float>(gradLocalFreBuf, compute32Gm_[gradOffset], copyExtParamsWeight, padParamsWeight);
            SetFlag<HardEvent::MTE2_V>(eventMTE2_V);
            WaitFlag<HardEvent::MTE2_V>(eventMTE2_V);
            ComputeFinalValueWithFreqProcess(indiceFreqBuf, gradLocalFreBuf, gradOffset,
                                             weightPerRowFreqReal, weightPerRowFreqDtypeAlign, tiling_.indicesFactorFreqTail);
        }
    }

    __aicore__ inline void CopyProcess()
    {
        if (blockId_ >= tiling_.processBlock) {
            return;
        }
        int64_t arNum = 0;
        LocalTensor<U> sortedIndice = this->sortedIndiceBuf_.template Get<U>();
        LocalTensor<int32_t> noDupRes = this->noDupBuf_.template Get<int32_t>();

        // Main
        for (uint32_t i = 0; i < tiling_.loopPerCoreIndice - 1; i++) {
            int64_t indicesOffset = i * tiling_.indicesFactor;
            this->CopyIndicesFromGm(indicesOffset, tiling_.indicesFactor);
            this->ProcessIndices(sortedIndice, tiling_.indicesFactor, arNum);
            for (uint32_t j = 0; j < loopPerCoreGrad_; j++) {
                uint32_t gradFactorPerRowReal = \
                    j != (loopPerCoreGrad_ - 1) ? gradFactorPerRow_ : gradFactorPerRowTail_;
                uint64_t gradOffset = i * tiling_.indicesFactor * tiling_.embeddingDim +
                              blockId_ * tiling_.embeddingDimPerCore +
                              j * gradFactorPerRow_;
                LocalTensor<T> gradLocal = inQueueGrad.AllocTensor<T>();
                DataCopyExtParams copyExtParamsGrad;
                copyExtParamsGrad.blockCount = static_cast<uint16_t>(tiling_.indicesFactor);
                copyExtParamsGrad.blockLen = static_cast<uint32_t>(gradFactorPerRowReal * sizeof(T));
                copyExtParamsGrad.srcStride = \
                    static_cast<uint32_t>((tiling_.embeddingDim - gradFactorPerRowReal) * sizeof(T));
                copyExtParamsGrad.dstStride = 0;
                DataCopyPadExtParams<T> padParamsGrad{false, 0, 0, 0};
                DataCopyPad<T, PaddingMode::Compact>(gradLocal, gradGm_[gradOffset], copyExtParamsGrad, padParamsGrad);
                inQueueGrad.EnQue(gradLocal);
                DupSegmentProcess(j, gradFactorPerRowReal, arNum, noDupRes);
                CopyOutProcess(j, gradFactorPerRowReal, arNum, sortedIndice, noDupRes);
            }
            if (scaleGradByFreq_ && (0 == blockId_)) {
                CopyOutIndicesForFreq(sortedIndice, noDupRes, tiling_.indicesFactor, arNum);
            }
        }
        int64_t indicesOffset = (tiling_.loopPerCoreIndice - 1) * tiling_.indicesFactor;
        this->CopyIndicesFromGm(indicesOffset, tiling_.indicesFactorTail);
        this->ProcessIndices(sortedIndice, tiling_.indicesFactorTail, arNum);
        for (uint32_t j = 0; j < loopPerCoreGrad_; j++) {
            uint32_t gradFactorPerRowReal = \
                j != (loopPerCoreGrad_ - 1) ? gradFactorPerRow_ : gradFactorPerRowTail_;
            uint64_t gradOffset = (tiling_.loopPerCoreIndice - 1) * tiling_.indicesFactor * tiling_.embeddingDim +
                          blockId_ * tiling_.embeddingDimPerCore +
                          j * gradFactorPerRow_;
            LocalTensor<T> gradLocal = inQueueGrad.AllocTensor<T>();
            DataCopyExtParams copyExtParamsGrad;
            copyExtParamsGrad.blockCount = static_cast<uint16_t>(tiling_.indicesFactorTail);
            copyExtParamsGrad.blockLen = static_cast<uint32_t>(gradFactorPerRowReal * sizeof(T));
            copyExtParamsGrad.srcStride = \
                static_cast<uint32_t>((tiling_.embeddingDim - gradFactorPerRowReal) * sizeof(T));
            copyExtParamsGrad.dstStride = 0;
            DataCopyPadExtParams<T> padParamsGrad{false, 0, 0, 0};
            DataCopyPad<T, PaddingMode::Compact>(gradLocal, gradGm_[gradOffset], copyExtParamsGrad, padParamsGrad);
            inQueueGrad.EnQue(gradLocal);
            DupSegmentProcess(j, gradFactorPerRowReal, arNum, noDupRes);
            CopyOutProcess(j, gradFactorPerRowReal, arNum, sortedIndice, noDupRes);
        }
        if (scaleGradByFreq_ && (0 == blockId_)) {
            CopyOutIndicesForFreq(sortedIndice, noDupRes, tiling_.indicesFactorTail, arNum);
        }
    }

    __aicore__ inline void Process()
    {
        CopyProcess();
        if (scaleGradByFreq_) {
            PipeBarrier<PIPE_ALL>();
            SyncAll();
            if (blockId_ >= tiling_.processBlock) {
                return;
            }
            InitBufferFreq();
            ComputeFinalValueWithFreq();
        } else {
            if constexpr (!is_same<T, float>::value) {
                PipeBarrier<PIPE_ALL>();
                SyncAll();
                if (blockId_ >= tiling_.processBlock) {
                    return;
                }
                InitBufferCast();
                if (cntACast_ == 1) {
                    ComputeCastMulti();
                } else {
                    ComputeCastSpilt();
                }
            }
        }
    }

private:
    TPipe& pipe_;
    TQue<QuePosition::VECIN, 1> inQueueGrad;
    TQue<QuePosition::VECOUT, 1> outQueueRes_;
    TQue<QuePosition::VECOUT, 1> outQueueGradWeight;
    TBuf<QuePosition::VECCALC> resBuf_;
    TBuf<QuePosition::VECCALC> castBuf_;
    TBuf<QuePosition::VECCALC> indiceFreBuf_;
    TBuf<QuePosition::VECCALC> gradLocalFreBuf_;
    AscendC::GlobalTensor<T> gradGm_;
    AscendC::GlobalTensor<T> gradWeightGm_;
    AscendC::GlobalTensor<U> countsGm_;
    AscendC::GlobalTensor<float> compute32Gm_;

    LocalTensor<float> gradLocalFreq;

    int32_t scaleGradByFreq_{0};
    int64_t blockId_{0};
    int64_t blockNums_{0};

    uint64_t loopPerCoreGrad_{0};
    uint32_t loopPerCoreGradFreq_{0};
    uint32_t gradFactorPerRow_{0};
    uint32_t gradFactorPerRowTail_{0};
    uint32_t gradFactorPerRowFreq_{0};
    uint32_t gradFactorPerRowTailFreq_{0};
    uint32_t gradFactorPerRowAlign_{0};

    uint32_t embeddingDimPerCoreReal_{0};
    // tilingData
    const EmbeddingDenseGradSimdTilingData& tiling_;
    // Cast
    uint32_t baseACast_{0};
    uint32_t baseACastAlign_{0};
    uint32_t baseWCast_{0};
    uint64_t cntACast_{0};
    uint64_t cntWCast_{0};
    uint32_t tailACast_{0};
    uint32_t tailWCast_{0};

    static constexpr uint32_t blockAlignNum_ = platform::GetUbBlockSize() / sizeof(T);
    static constexpr uint32_t b32AlignNum_ = platform::GetUbBlockSize() / sizeof(float);
    static constexpr uint32_t vfLengthFp32_ = platform::GetVRegSize() / sizeof(float);
    static constexpr uint32_t shiftOffset_ = platform::GetUbBlockSize() / sizeof(U);
    static constexpr uint32_t perVfIndicesIdx_ = platform::GetVRegSize() / sizeof(int32_t);
};
}
#endif
