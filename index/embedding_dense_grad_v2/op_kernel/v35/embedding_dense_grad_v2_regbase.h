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
 * \file embedding_dense_grad_v2_regbase.h
 * \brief embedding_dense_grad_v2
 */

#ifndef EMBEDDING_DENSE_GRAD_V2_REGBASE_H
#define EMBEDDING_DENSE_GRAD_V2_REGBASE_H

#include "kernel_operator.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"
#include "../inc/load_store_utils.h"
namespace EmbeddingDenseGradV2 {
using namespace AscendC;

static constexpr uint32_t ONE_BLK_SIZE = platform::GetUbBlockSize();
static const uint32_t CACHE_LINE = 128;
static const uint32_t HELP_FRE = 2;
static const uint32_t PADDING_SIZE = 64;
static const uint32_t DOUBLE = 2;
static const uint32_t FOUR = 4;
static const uint32_t ATOMIC_ELEMENT = 1;
static const uint32_t SECTOR_LINE = 256;
static const uint32_t PER_COLUMN_ALIGN = SECTOR_LINE / sizeof(float);
static const uint16_t PROCESS_GROUP = 10;
static const int64_t MAX_ELEWISE_AXIS_LIMIT = 8192;

template<typename GRAD_T, typename CAST_T, typename IDX_T, bool isScale = false, bool SplitEltwiseAxis = false, int32_t bufferNum = 1>
class EmbeddingDenseGradV2Kernel
{
public:
    __aicore__ inline EmbeddingDenseGradV2Kernel(TPipe& pipe, const EmbeddingDenseGradV2TilingData4RegBase& tilingData) : pipe_(pipe), tiling_(tilingData){};
    __aicore__ inline void Init(GM_ADDR grad, GM_ADDR sortIndices, GM_ADDR posIdx, GM_ADDR gradWeight, GM_ADDR workSpace);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessNotSplitEltwiseAxis();
    template <bool isColTail = false>
    __aicore__ inline void ProcessSplitEltwiseAxis(int64_t colBaseOffset);
    __aicore__ inline void CopyIndices(LocalTensor<IDX_T>& indicesTensor, LocalTensor<int32_t>& posIdxTensor, IDX_T indicesNumber, int64_t offset);
    __aicore__ inline void CalcDupNum(LocalTensor<int32_t>& noDupCount, int64_t& arNum, const LocalTensor<IDX_T>& indicesTensor, IDX_T indicesNumber);
    __aicore__ inline void UniqueGetElm(const LocalTensor<IDX_T> &indicesTensor, LocalTensor<int32_t> &noDupCount, uint32_t indicesFactorReal, int64_t &arNum);
    __aicore__ inline void UniqueStat(const LocalTensor<IDX_T> &indicesTensor, LocalTensor<int32_t> &noDupCount, uint32_t indicesFactorReal, int64_t &arNum);
    __aicore__ inline void CopyGrad(LocalTensor<int32_t>& posIdxTensor, IDX_T indicesNumber, uint32_t colLen, int64_t colOffset);
    template <bool useRes>
    __aicore__ inline void AddGradPerIndice(LocalTensor<GRAD_T>& outTensor, LocalTensor<CAST_T>& lastResTensor, const LocalTensor<GRAD_T>& gradTensor, uint32_t colLen, int32_t curGradCount, float scaleFreq);
    template <bool isColTail>
    __aicore__ inline void AddFirstGradByGroup(const LocalTensor<GRAD_T>& gradTensor, uint32_t colLen, uint32_t colOffset, LocalTensor<int32_t>& noDupCount, int64_t arNum, uint32_t resColOffset);
    __aicore__ inline void AddGradByGroup(const LocalTensor<GRAD_T>& gradTensor, uint32_t colLen, uint32_t colOffset, LocalTensor<int32_t>& noDupCount, int64_t arNum, uint32_t resColOffset);
    template <bool SaveEnd, bool isColTail, bool isUpdateFreq = false>
    __aicore__ inline void CopyOutGradWeight(LocalTensor<IDX_T>& indicesTensor, LocalTensor<int32_t>& noDupCountTensor, int64_t arNum, uint32_t colLen, int64_t colOffset, uint32_t resColOffset);
    __aicore__ inline void CopyIndicesToWs();
    __aicore__ inline void CopyIndicesFromWs(IDX_T indicesNumber, uint32_t endOffset);
    __aicore__ inline bool CheckHasSameIndices(uint32_t indicesNumber, uint32_t endOffset);
    __aicore__ inline void GatherIndices(LocalTensor<IDX_T>& indicesBeginTensor, LocalTensor<IDX_T>& indicesEndTensor, uint32_t indicesNumber);
    __aicore__ inline void InterleaveIndices(uint32_t totalIndicesNum, uint32_t endOffset);
    __aicore__ inline void CopyGradFromWs(IDX_T indicesNumber, uint32_t elewiseNumber, uint32_t stride, int64_t gmOffset);
    __aicore__ inline void CopyScaleFreqFromWs(uint32_t totalIndicesNum);
    __aicore__ inline void GatherScaleFreq(uint32_t totalIndicesNum);
    __aicore__ inline void CalcFreqBetweenBlock(LocalTensor<int32_t> &noDupCount, int64_t arNum, uint32_t totalIndicesNum);
    __aicore__ inline void AddGradPerIndiceBetweenBlock(LocalTensor<GRAD_T>& outTensor, const LocalTensor<CAST_T>& gradTensor, uint32_t colLen, uint32_t colLenAlign, int32_t curGradCount, float freqInBlock);
    __aicore__ inline void AddGradBetweenBlock(uint32_t colLen, uint32_t gradColAlign, LocalTensor<int32_t>& noDupCount, int64_t &arNum, uint32_t totalIndicesNum);
    __aicore__ inline void ReInitBuffer(uint32_t totalIndicesNumber);
    __aicore__ inline void ProcessGradBetweenBlock();

protected:
    TPipe& pipe_;
    TQue<QuePosition::VECIN, bufferNum> inQueGrad_;
    TQue<QuePosition::VECOUT, bufferNum> outQueGradWeight_;

    TBuf<QuePosition::VECIN> indicesBuf_;
    TBuf<QuePosition::VECIN> posIdxBuf_;
    TBuf<QuePosition::VECCALC> noDupIdxCountBuf_;  // 保存ub内每个词索引的重复次数的数组
    TBuf<QuePosition::VECCALC> lastResBuf_;

    // In/Out Gm
    GlobalTensor<GRAD_T> gradGm_;
    GlobalTensor<IDX_T> sortIndicesGm_;
    GlobalTensor<int32_t> posIdxGm_;
    GlobalTensor<GRAD_T> gradWeightGm_;

    // workspace
    GlobalTensor<IDX_T> idxStartGm_;
    GlobalTensor<CAST_T> gradStartGm_;
    GlobalTensor<IDX_T> idxBeginWsGm_;
    GlobalTensor<IDX_T> idxBeginFreqWsGm_;
    GlobalTensor<CAST_T> gradBeginWsGm_;
    GlobalTensor<IDX_T> idxEndWsGm_;
    GlobalTensor<IDX_T> idxEndFreqWsGm_;
    GlobalTensor<CAST_T> gradEndWsGm_;
    GlobalTensor<IDX_T> blockSyncGm_;

    // var
    IDX_T firstResIdx_ {-2};
    IDX_T firstResCount_ {0};
    IDX_T lastResIdx_ {-2};
    IDX_T lastResCount_ {0};
    int64_t gradLenInWs_ {0};
    int64_t scatterLoopNums_ {0};
    int64_t elewiseLoopNums_ {0};
    int64_t elewiseOuterTail_ {0};
    uint32_t scatterTail_ {0};
    uint32_t elewiseTail_ {0};
    uint32_t elewiseNormal_ {0};
    uint64_t indicesStart_ {0};
    int32_t blockIdx_ {0};
    uint32_t elewiseAligned_ {0};
    uint32_t isBeginRecord_ {0};
    uint32_t isBeginPadding_ {0};
    constexpr static uint16_t vfFp32Block_ = platform::GetVRegSize() / sizeof(float);
    constexpr static uint32_t perVfIndices_ = platform::GetVRegSize() / sizeof(IDX_T);
    constexpr static uint32_t indexCountInBlock_ = ONE_BLK_SIZE / sizeof(IDX_T);
    constexpr static uint32_t copyGradAlign_ = ONE_BLK_SIZE/ sizeof(GRAD_T);

    // tilingData
    const EmbeddingDenseGradV2TilingData4RegBase& tiling_;

    using RegTensorType = typename std::conditional<IsSameType<IDX_T, int64_t>::value, typename MicroAPI::RegTensor<IDX_T, MicroAPI::RegTraitNumTwo>,
                                            typename MicroAPI::RegTensor<IDX_T>>::type;
    using RegCastType = typename std::conditional<IsSameType<IDX_T, int64_t>::value, typename MicroAPI::RegTensor<uint64_t, MicroAPI::RegTraitNumTwo>,
                                            typename MicroAPI::RegTensor<uint32_t>>::type;
};

template <typename GRAD_T, typename CAST_T, typename IDX_T, bool isScale, bool SplitEltwiseAxis, int32_t bufferNum>
__aicore__ inline void EmbeddingDenseGradV2Kernel<GRAD_T, CAST_T, IDX_T, isScale, SplitEltwiseAxis, bufferNum>::Init(GM_ADDR grad, GM_ADDR sortIndices, GM_ADDR posIdx, GM_ADDR gradWeight, GM_ADDR workSpace)
{
    blockIdx_ = GetBlockIdx();

    indicesStart_ = blockIdx_ * tiling_.scatterFactor * tiling_.normalBlockLoopNum;
    if (blockIdx_ < tiling_.usedCoreNum - 1) {
        scatterLoopNums_ = tiling_.normalBlockLoopNum;
        scatterTail_ = tiling_.scatterFactor;
    } else {
        scatterLoopNums_ = tiling_.tailBlockLoopNum;
        scatterTail_ = tiling_.scatterDimSize - (blockIdx_ * tiling_.normalBlockLoopNum + tiling_.tailBlockLoopNum - 1) * tiling_.scatterFactor;
    }

    int64_t lastResBufSize = ops::CeilAlign(tiling_.elewiseDimSize, static_cast<int64_t>(ONE_BLK_SIZE / sizeof(CAST_T))) * sizeof(CAST_T);
    if constexpr (SplitEltwiseAxis) {
        if (tiling_.elewiseDimOuterLoopNum < DOUBLE) {
            elewiseOuterTail_ = tiling_.elewiseDimSize;
            elewiseLoopNums_ = ops::CeilDiv(tiling_.elewiseDimSize, static_cast<int64_t>(tiling_.elewiseFactor));
            elewiseTail_ = tiling_.elewiseDimSize - (elewiseLoopNums_ - 1) * tiling_.elewiseFactor;
        } else {
            constexpr int64_t normalElewiseNum = MAX_ELEWISE_AXIS_LIMIT / sizeof(CAST_T);
            elewiseLoopNums_ = ops::CeilDiv(normalElewiseNum, static_cast<int64_t>(tiling_.elewiseFactor));
            elewiseTail_ = normalElewiseNum - (elewiseLoopNums_ - 1) * tiling_.elewiseFactor;
            elewiseOuterTail_ = tiling_.elewiseDimSize - (tiling_.elewiseDimOuterLoopNum - 1) * normalElewiseNum;
            lastResBufSize = MAX_ELEWISE_AXIS_LIMIT;
        }
    } else {
        elewiseTail_ = tiling_.elewiseDimSize;
        elewiseLoopNums_ = 1;
    }

    elewiseAligned_ = ops::CeilAlign(tiling_.elewiseFactor, static_cast<uint32_t>(ONE_BLK_SIZE / sizeof(GRAD_T)));

    // init params
    gradGm_.SetGlobalBuffer((__gm__ GRAD_T *)grad);
    sortIndicesGm_.SetGlobalBuffer((__gm__ IDX_T *)sortIndices);
    posIdxGm_.SetGlobalBuffer((__gm__ int32_t *)posIdx);
    gradWeightGm_.SetGlobalBuffer((__gm__ GRAD_T *)gradWeight);

    int64_t blockSyncWs = CACHE_LINE;
    gradLenInWs_ = ops::CeilAlign(tiling_.elewiseDimSize * sizeof(CAST_T), static_cast<uint64_t>(CACHE_LINE));
    int64_t idxBeginOffset = blockSyncWs + blockIdx_ * (CACHE_LINE * DOUBLE + gradLenInWs_) * DOUBLE;
    int64_t idxBeginFreqOffset = idxBeginOffset + CACHE_LINE;
    int64_t gradBeginOffset = idxBeginFreqOffset + CACHE_LINE;
    int64_t idxEndOffset = gradBeginOffset + gradLenInWs_;
    int64_t idxEndFreqOffset = idxEndOffset + CACHE_LINE;
    int64_t gradEndOffset = idxEndFreqOffset + CACHE_LINE;

    int64_t end0offset = blockSyncWs + CACHE_LINE * DOUBLE + gradLenInWs_;
    idxStartGm_.SetGlobalBuffer((__gm__ IDX_T*)((__gm__ uint8_t*)workSpace + end0offset));
    gradStartGm_.SetGlobalBuffer((__gm__ CAST_T*)((__gm__ uint8_t*)workSpace + end0offset + DOUBLE * CACHE_LINE));

    idxBeginWsGm_.SetGlobalBuffer((__gm__ IDX_T*)((__gm__ uint8_t*)workSpace + idxBeginOffset));
    idxBeginFreqWsGm_.SetGlobalBuffer((__gm__ IDX_T*)((__gm__ uint8_t*)workSpace + idxBeginFreqOffset));
    gradBeginWsGm_.SetGlobalBuffer((__gm__ CAST_T*)((__gm__ uint8_t*)workSpace + gradBeginOffset));
    idxEndWsGm_.SetGlobalBuffer((__gm__ IDX_T*)((__gm__ uint8_t*)workSpace + idxEndOffset));
    idxEndFreqWsGm_.SetGlobalBuffer((__gm__ IDX_T*)((__gm__ uint8_t*)workSpace + idxEndFreqOffset));
    gradEndWsGm_.SetGlobalBuffer((__gm__ CAST_T*)((__gm__ uint8_t*)workSpace + gradEndOffset));

    blockSyncGm_.SetGlobalBuffer((__gm__ IDX_T*)((__gm__ uint8_t*)workSpace));

    // scatterFactor 已经在tiling侧按照32Byte对齐
    pipe_.InitBuffer(inQueGrad_, bufferNum, tiling_.scatterFactor * elewiseAligned_ * sizeof(GRAD_T));
    pipe_.InitBuffer(indicesBuf_, tiling_.scatterFactor * sizeof(IDX_T) + PADDING_SIZE);
    pipe_.InitBuffer(posIdxBuf_, tiling_.scatterFactor * sizeof(int32_t));
    pipe_.InitBuffer(noDupIdxCountBuf_, tiling_.scatterFactor * sizeof(IDX_T) + PADDING_SIZE);
    pipe_.InitBuffer(outQueGradWeight_, bufferNum, tiling_.scatterFactor * elewiseAligned_ * sizeof(GRAD_T));
    pipe_.InitBuffer(lastResBuf_, lastResBufSize);
}

template <typename GRAD_T, typename CAST_T, typename IDX_T, bool isScale, bool SplitEltwiseAxis, int32_t bufferNum>
__aicore__ inline void EmbeddingDenseGradV2Kernel<GRAD_T, CAST_T, IDX_T, isScale, SplitEltwiseAxis, bufferNum>::CopyIndices(LocalTensor<IDX_T>& indicesTensor,
                                                                            LocalTensor<int32_t>& posIdxTensor, IDX_T indicesNumber, int64_t offset)
{
    uint64_t copyPosition = indicesStart_ + offset;

    IDX_T indicesNumAligned = ops::CeilAlign(indicesNumber, static_cast<IDX_T>(ONE_BLK_SIZE / sizeof(IDX_T)));
    Duplicate(indicesTensor, (IDX_T)-1, indicesNumAligned + PADDING_SIZE / sizeof(IDX_T));
    event_t eventV_MTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(eventV_MTE2);
    WaitFlag<HardEvent::V_MTE2>(eventV_MTE2);

    uint8_t rightPadNum = static_cast<uint8_t>(indicesNumAligned - indicesNumber);
    DataCopyPadExtParams<IDX_T> padParams{true, static_cast<uint8_t>(0), rightPadNum, static_cast<uint8_t>(-1)};
    DataCopyExtParams copyParam {
        static_cast<uint16_t>(1),
        static_cast<uint32_t>(indicesNumber * sizeof(IDX_T)),
        static_cast<uint32_t>(0),
        static_cast<uint32_t>(0),
        static_cast<uint32_t>(0)
    };

    DataCopyPad(indicesTensor[indexCountInBlock_], sortIndicesGm_[copyPosition], copyParam, padParams);

    int32_t posIdxNumAligned = static_cast<int32_t>(indicesNumAligned);
    uint8_t rightPadPosIdxNum = static_cast<uint8_t>(posIdxNumAligned - indicesNumber);
    DataCopyPadExtParams<int32_t> padPosIdxParams{true, static_cast<uint8_t>(0), rightPadPosIdxNum, static_cast<uint8_t>(-1)};
    DataCopyExtParams copyPosIdxParam {
        static_cast<uint16_t>(1),
        static_cast<uint32_t>(indicesNumber * sizeof(int32_t)),
        static_cast<uint32_t>(0),
        static_cast<uint32_t>(0),
        static_cast<uint32_t>(0)
    };

    DataCopyPad(posIdxTensor, posIdxGm_[copyPosition], copyPosIdxParam, padPosIdxParams);
}

template <typename GRAD_T, typename CAST_T, typename IDX_T, bool isScale, bool SplitEltwiseAxis, int32_t bufferNum>
__aicore__ inline void EmbeddingDenseGradV2Kernel<GRAD_T, CAST_T, IDX_T, isScale, SplitEltwiseAxis, bufferNum>::CopyGrad(LocalTensor<int32_t>& posIdxTensor,
                                                                                                    IDX_T indicesNumber, uint32_t colLen, int64_t colOffset)
{
    LocalTensor<GRAD_T> gradTensor = inQueGrad_.template AllocTensor<GRAD_T>();

    uint8_t rightPadNum = static_cast<uint8_t>(ops::CeilAlign(colLen, copyGradAlign_) - colLen);
    DataCopyPadExtParams<GRAD_T> padParams{true, static_cast<uint8_t>(0), rightPadNum, static_cast<uint8_t>(0)};
    DataCopyExtParams copyParam {
        static_cast<uint16_t>(1),
        static_cast<uint32_t>(colLen * sizeof(GRAD_T)),
        static_cast<uint32_t>(0),
        static_cast<uint32_t>(0),
        static_cast<uint32_t>(0)
    };

    // 不做padding过滤，直接拷贝。否则，scalar过多导致性能下降。
    for (uint32_t i = 0; i < indicesNumber; i++) {
        uint64_t gmOffset = posIdxTensor(i) * tiling_.elewiseDimSize + colOffset;
        uint32_t gradOffset = elewiseAligned_ * i;
        DataCopyPad(gradTensor[gradOffset], gradGm_[gmOffset], copyParam, padParams);
    }

    inQueGrad_.template EnQue<GRAD_T>(gradTensor);
}

template <typename GRAD_T, typename CAST_T, typename IDX_T, bool isScale, bool SplitEltwiseAxis, int32_t bufferNum>
__aicore__ inline void EmbeddingDenseGradV2Kernel<GRAD_T, CAST_T, IDX_T, isScale, SplitEltwiseAxis, bufferNum>::CalcDupNum(LocalTensor<int32_t>& noDupCount, int64_t& arNum,
                                                                                                    const LocalTensor<IDX_T>& indicesTensor, IDX_T indicesNumber)
{
    event_t eventMTE2_V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventMTE2_V);
    WaitFlag<HardEvent::MTE2_V>(eventMTE2_V);

    Duplicate(noDupCount, 0, indicesNumber);
    UniqueGetElm(indicesTensor, noDupCount, indicesNumber, arNum);
    arNum = ((AscendC::MicroAPI::GetSpr<AscendC::SpecialPurposeReg::AR>()) / sizeof(int32_t)) - 1; // noDupRes的数组大小
    UniqueStat(indicesTensor, noDupCount, indicesNumber, arNum);
}

template <typename GRAD_T, typename CAST_T, typename IDX_T, bool isScale, bool SplitEltwiseAxis, int32_t bufferNum>
__aicore__ inline void EmbeddingDenseGradV2Kernel<GRAD_T, CAST_T, IDX_T, isScale, SplitEltwiseAxis, bufferNum>::UniqueGetElm(const LocalTensor<IDX_T> &indicesTensor, LocalTensor<int32_t> &noDupCount,
                                uint32_t indicesFactorReal, int64_t &arNum)
{
    __local_mem__ IDX_T *sortedIndicesAddr = (__ubuf__ IDX_T *)indicesTensor.GetPhyAddr();
    __local_mem__ int32_t *noDupResAddr = (__ubuf__ int32_t *)noDupCount.GetPhyAddr();

    uint16_t loopCnt = (uint16_t)((indicesFactorReal + perVfIndices_) / perVfIndices_);
    int32_t scalar = 0;
    constexpr uint32_t shiftOffset = ONE_BLK_SIZE / sizeof(IDX_T);
    uint32_t counter = indicesFactorReal + 1;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<int32_t> orderReg;
        AscendC::MicroAPI::RegTensor<int32_t> selReg;
        AscendC::MicroAPI::RegTensor<IDX_T> indicesReg;
        AscendC::MicroAPI::RegTensor<IDX_T> indicesShiftOneReg;

        AscendC::MicroAPI::MaskReg maskReg =
            AscendC::MicroAPI::CreateMask<IDX_T, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg cmpMask;
        AscendC::MicroAPI::MaskReg maskRegUpdate;
        AscendC::MicroAPI::UnalignReg u0;
        MicroAPI::UnalignReg ureg;
        AscendC::MicroAPI::ClearSpr<AscendC::SpecialPurposeReg::AR>();

        for (uint16_t i = 0; i < loopCnt; ++i) {
            scalar = i * perVfIndices_;
            auto sortedIndicesAddrUpdate = sortedIndicesAddr + shiftOffset + i * perVfIndices_;
            AscendC::MicroAPI::Arange(orderReg, scalar);
            maskRegUpdate = AscendC::MicroAPI::UpdateMask<IDX_T>(counter);
            DataCopy(indicesReg, sortedIndicesAddrUpdate);
            AscendC::MicroAPI::DataCopyUnAlignPre(u0, sortedIndicesAddrUpdate - 1);
            AscendC::MicroAPI::DataCopyUnAlign<IDX_T>(indicesShiftOneReg, u0, sortedIndicesAddrUpdate - 1);
            AscendC::MicroAPI::Compare<IDX_T, CMPMODE::NE>(cmpMask, indicesReg, indicesShiftOneReg, maskRegUpdate);
            if constexpr (IsSameType<IDX_T, int64_t>::value) {
                AscendC::MicroAPI::MaskReg maskHalf;
                AscendC::MicroAPI::MaskPack<AscendC::MicroAPI::HighLowPart::LOWEST>(maskHalf, cmpMask);
                // vSQZ
                AscendC::MicroAPI::GatherMask<int32_t, AscendC::MicroAPI::GatherMaskMode::STORE_REG>
                    (selReg, orderReg, maskHalf);
            } else {
                // vSQZ
                AscendC::MicroAPI::GatherMask<int32_t, AscendC::MicroAPI::GatherMaskMode::STORE_REG>
                    (selReg, orderReg, cmpMask);
            }

            AscendC::MicroAPI::DataCopyUnAlign<int32_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>
                (noDupResAddr, selReg, ureg);
            AscendC::MicroAPI::DataCopyUnAlignPost(noDupResAddr, ureg);
        }
    }
}

template <typename GRAD_T, typename CAST_T, typename IDX_T, bool isScale, bool SplitEltwiseAxis, int32_t bufferNum>
__aicore__ inline void EmbeddingDenseGradV2Kernel<GRAD_T, CAST_T, IDX_T, isScale, SplitEltwiseAxis, bufferNum>::UniqueStat(const LocalTensor<IDX_T> &indicesTensor,
                                LocalTensor<int32_t> &noDupCount, uint32_t indicesFactorReal, int64_t &arNum)
{
    __local_mem__ int32_t *noDupResAddr = (__ubuf__ int32_t *)noDupCount.GetPhyAddr();

    constexpr uint32_t vfLen = platform::GetVRegSize() / sizeof(int32_t);
    uint16_t loopCntStatFre = (arNum + vfLen - 1) / vfLen;
    uint32_t counterStatFre = arNum;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<int32_t> beginReg;
        AscendC::MicroAPI::RegTensor<int32_t> endReg;
        AscendC::MicroAPI::RegTensor<int32_t> subReg;
        AscendC::MicroAPI::MaskReg maskRegUpdateFre;
        AscendC::MicroAPI::UnalignReg u0Fre;
        for (uint16_t i = 0; i < loopCntStatFre; i++) {
            auto noDupResAddrUpdate = noDupResAddr + i * vfLen + 1;
            maskRegUpdateFre = AscendC::MicroAPI::UpdateMask<int32_t>(counterStatFre);
            DataCopy(beginReg, noDupResAddr + i * vfLen);
            AscendC::MicroAPI::DataCopyUnAlignPre(u0Fre, noDupResAddrUpdate);
            AscendC::MicroAPI::DataCopyUnAlign<int32_t>(endReg, u0Fre, noDupResAddrUpdate);
            AscendC::MicroAPI::Sub(subReg, endReg, beginReg, maskRegUpdateFre);
            DataCopy(noDupResAddr + i * vfLen, subReg, maskRegUpdateFre);
        }
    }

    event_t eventV_S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventV_S);
    WaitFlag<HardEvent::V_S>(eventV_S);
}

template <typename GRAD_T, typename CAST_T, typename IDX_T, bool isScale, bool SplitEltwiseAxis, int32_t bufferNum>
template <bool useRes>
__aicore__ inline void EmbeddingDenseGradV2Kernel<GRAD_T, CAST_T, IDX_T, isScale, SplitEltwiseAxis, bufferNum>::AddGradPerIndice(LocalTensor<GRAD_T>& outTensor, LocalTensor<CAST_T>& lastResTensor,
                                                                                                const LocalTensor<GRAD_T>& gradTensor, uint32_t colLen, int32_t curGradCount, float scaleFreq)
{
    auto ubGradAddr = (__ubuf__ GRAD_T *)gradTensor.GetPhyAddr();
    auto ubResultAddr = (__ubuf__ GRAD_T *)outTensor.GetPhyAddr();
    auto ubLastBufAddr = (__ubuf__ CAST_T *)lastResTensor.GetPhyAddr();

    uint32_t processColLen = colLen;
    uint32_t perColAlign = elewiseAligned_;
    uint16_t perColLoopNum = ops::CeilDiv(colLen, static_cast<uint32_t>(vfFp32Block_));
    uint16_t loopSegCount = curGradCount / PROCESS_GROUP;
    uint16_t normalSegCount = loopSegCount * PROCESS_GROUP;
    uint16_t loopSegCountTail = curGradCount - normalSegCount;

    __VEC_SCOPE__ {
        MicroAPI::RegTensor<float> srcGrad;
        MicroAPI::RegTensor<float> groupAdd;
        MicroAPI::RegTensor<float> gradResult;
        MicroAPI::MaskReg preg;

        for (uint16_t col = 0; col < perColLoopNum; col++) { // grad 一行
            preg = MicroAPI::UpdateMask<float>(processColLen);
            uint32_t colOffset = col * vfFp32Block_;  // 列方向偏移
            if constexpr (useRes) { // 相同UB间相同索引，需要累加
                ops::LoadOneTensorForDtypeT<CAST_T>(ubLastBufAddr, gradResult, preg, colOffset);
            } else {
                Duplicate(gradResult, (float)0, preg);
            }

            for (uint16_t mainSeg = 0; mainSeg < loopSegCount; mainSeg++) {   // Main group
                Duplicate(groupAdd, (float)0, preg);  // 分组累加，每组累加前清空reg

                uint32_t groupRowStart = mainSeg * PROCESS_GROUP;
                for (uint16_t group = 0; group < PROCESS_GROUP; group++) {   // per group
                    uint32_t gradOffset = (groupRowStart + group) * perColAlign + colOffset;
                    ops::LoadOneTensorForDtypeT<GRAD_T>(ubGradAddr, srcGrad, preg, gradOffset);
                    Add(groupAdd, groupAdd, srcGrad, preg);
                }
                Add(gradResult, gradResult, groupAdd, preg);
            }

            Duplicate(groupAdd, (float)0, preg);
            for (uint16_t tailRow = 0; tailRow < loopSegCountTail; tailRow++) {  // Tail group
                uint32_t gradOffset = (normalSegCount + tailRow) * perColAlign + colOffset;
                ops::LoadOneTensorForDtypeT<GRAD_T>(ubGradAddr, srcGrad, preg, gradOffset);
                Add(groupAdd, groupAdd, srcGrad, preg);
            }
            Add(gradResult, gradResult, groupAdd, preg);
            // lastRes不进行scale，直接存储
            ops::StoreOneTensorForDtypeT<CAST_T>(ubLastBufAddr, gradResult, preg, colOffset);

            if constexpr (isScale) {
                Muls(gradResult, gradResult, scaleFreq, preg);
            }

            ops::StoreOneTensorForDtypeT<GRAD_T>(ubResultAddr, gradResult, preg, colOffset);
        }
    }
}

template <typename GRAD_T, typename CAST_T, typename IDX_T, bool isScale, bool SplitEltwiseAxis, int32_t bufferNum>
template <bool isColTail>
__aicore__ inline void EmbeddingDenseGradV2Kernel<GRAD_T, CAST_T, IDX_T, isScale, SplitEltwiseAxis, bufferNum>::AddFirstGradByGroup(const LocalTensor<GRAD_T>& gradTensor, uint32_t colLen,
                                                                                                    uint32_t colOffset, LocalTensor<int32_t>& noDupCount, int64_t arNum, uint32_t resColOffset)
{
    LocalTensor<GRAD_T> outTensor = outQueGradWeight_.template AllocTensor<GRAD_T>();
    LocalTensor<CAST_T> lastRes = lastResBuf_.Get<CAST_T>();
    LocalTensor<CAST_T> lastResTensor = lastRes[resColOffset];
    LocalTensor<IDX_T> indicesTensor = indicesBuf_.Get<IDX_T>();

    IDX_T curIndex = indicesTensor[indexCountInBlock_](0);
    IDX_T curCount = noDupCount(0);
    IDX_T firstIdxCount = curCount;
    if (curIndex != tiling_.paddingIdx) {
        if (lastResIdx_ != curIndex) {
            this->template AddGradPerIndice<false>(outTensor, lastResTensor, gradTensor, colLen, curCount, 1.0f / static_cast<float>(firstIdxCount));
        } else {
            firstIdxCount = lastResCount_ + curCount;
            if (firstResIdx_ == curIndex) {
                isBeginRecord_ = false; // UB间index相同，需要重新存begin grad
            }
            this->template AddGradPerIndice<true>(outTensor, lastResTensor, gradTensor, colLen, curCount, 1.0f / static_cast<float>(firstIdxCount));
        }
    } else {
        isBeginPadding_ = 1;
    }

    /*
    *  1、UB间和UB内idx全相同
    *  2、UB间idx前N-1次全相同，第N次arNum > 1
    *  3、UB间idx前N-1次全相同，第N次arNum = 1，但是idx不同
    *  4、第一次UB内arNum > 1
    */
    if (!isBeginRecord_ && !isBeginPadding_) {
        if constexpr (isColTail) {
            isBeginRecord_ = true;
            firstResIdx_ = curIndex;
            firstResCount_ = firstIdxCount;
        }

        DataCopyExtParams copyParam {
            static_cast<uint16_t>(1),
            static_cast<uint32_t>(colLen * sizeof(CAST_T)),
            static_cast<uint32_t>(0),
            static_cast<uint32_t>(0),
            static_cast<uint32_t>(0)
        };

        event_t eventV_MTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventV_MTE3);
        WaitFlag<HardEvent::V_MTE3>(eventV_MTE3);

        DataCopyPad(gradBeginWsGm_[colOffset], lastResTensor, copyParam);

        event_t eventMTE3_V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventMTE3_V);
        WaitFlag<HardEvent::MTE3_V>(eventMTE3_V);
    }

    outQueGradWeight_.template EnQue<GRAD_T>(outTensor);
}

template <typename GRAD_T, typename CAST_T, typename IDX_T, bool isScale, bool SplitEltwiseAxis, int32_t bufferNum>
__aicore__ inline void EmbeddingDenseGradV2Kernel<GRAD_T, CAST_T, IDX_T, isScale, SplitEltwiseAxis, bufferNum>::AddGradByGroup(const LocalTensor<GRAD_T>& gradTensor, uint32_t colLen,
                                                                                                        uint32_t colOffset, LocalTensor<int32_t>& noDupCount, int64_t arNum, uint32_t resColOffset)
{
    LocalTensor<CAST_T> lastRes = lastResBuf_.Get<CAST_T>();
    LocalTensor<CAST_T> lastResTensor = lastRes[resColOffset];
    LocalTensor<GRAD_T> outTensor = outQueGradWeight_.template DeQue<GRAD_T>();
    LocalTensor<IDX_T> indicesTensor = indicesBuf_.Get<IDX_T>()[indexCountInBlock_];

    uint32_t gradRowStart = noDupCount(0);
    uint32_t idxOffset = gradRowStart;

    for (int64_t i = 1; i < arNum; i++) {
        uint32_t gradCount = noDupCount(i);
        IDX_T curIdx = indicesTensor(idxOffset);
        idxOffset = idxOffset + gradCount;
        if (curIdx != tiling_.paddingIdx) {
            LocalTensor<GRAD_T> outLocal = outTensor[i * elewiseAligned_];
            LocalTensor<GRAD_T> gradLocal = gradTensor[gradRowStart * elewiseAligned_];
            if constexpr (isScale) {
                this->template AddGradPerIndice<false>(outLocal, lastResTensor, gradLocal, colLen, gradCount, 1.0f / static_cast<float>(gradCount));
            } else {
                this->template AddGradPerIndice<false>(outLocal, lastResTensor, gradLocal, colLen, gradCount, 1.0f);
            }
        }
        gradRowStart = gradRowStart + gradCount;
    }

    outQueGradWeight_.template EnQue<GRAD_T>(outTensor);
}

template <typename GRAD_T, typename CAST_T, typename IDX_T, bool isScale, bool SplitEltwiseAxis, int32_t bufferNum>
template <bool SaveEnd, bool isColTail, bool isUpdateFreq>
__aicore__ inline void EmbeddingDenseGradV2Kernel<GRAD_T, CAST_T, IDX_T, isScale, SplitEltwiseAxis, bufferNum>::CopyOutGradWeight(LocalTensor<IDX_T>& indicesTensor,
                                                                LocalTensor<int32_t>& noDupCountTensor, int64_t arNum, uint32_t colLen, int64_t colOffset, uint32_t resColOffset)
{
    LocalTensor<GRAD_T> outTensor = outQueGradWeight_.template DeQue<GRAD_T>();
    LocalTensor<IDX_T> indices = indicesTensor[indexCountInBlock_];

    DataCopyExtParams copyParam {
        static_cast<uint16_t>(1),
        static_cast<uint32_t>(colLen * sizeof(GRAD_T)),
        static_cast<uint32_t>(0),
        static_cast<uint32_t>(0),
        static_cast<uint32_t>(0)
    };

    uint32_t idxOffset = 0;
    IDX_T curLastIdx = -2; // 确保本核最后放到workspace的indices是升序的
    IDX_T curLastCount = 0;
    for (int64_t i = 0; i < arNum; i++) {
        IDX_T idx = indices(idxOffset);
        IDX_T interval = noDupCountTensor(i);
        idxOffset = idxOffset + interval;
        if (idx == tiling_.paddingIdx || idx < 0 || idx >= tiling_.numWeights) {
            continue;
        }

        uint64_t gmOffset = idx * tiling_.elewiseDimSize + colOffset;
        DataCopyPad(gradWeightGm_[gmOffset], outTensor[i * elewiseAligned_], copyParam);
        curLastCount = interval;
        curLastIdx = idx;
    }

    if constexpr (isColTail) {
        // indices升序排列，curLastIdx有效时，lastResIdx 一定小于等于curLastIdx
        if (lastResIdx_ < curLastIdx) {
            lastResIdx_ = curLastIdx;
            lastResCount_ = curLastCount;
        } else if (lastResIdx_ == curLastIdx) {
            if constexpr (isUpdateFreq) {
                lastResCount_ = lastResCount_ + curLastCount;
            }
        }
    }

    if constexpr (SaveEnd) {
        LocalTensor<CAST_T> lastResTensor = lastResBuf_.Get<CAST_T>();
        copyParam.blockLen = colLen * sizeof(CAST_T);
        DataCopyPad(gradEndWsGm_[colOffset], lastResTensor[resColOffset], copyParam);
    }

    outQueGradWeight_.FreeTensor(outTensor);
}

template <typename GRAD_T, typename CAST_T, typename IDX_T, bool isScale, bool SplitEltwiseAxis, int32_t bufferNum>
__aicore__ inline void EmbeddingDenseGradV2Kernel<GRAD_T, CAST_T, IDX_T, isScale, SplitEltwiseAxis, bufferNum>::AddGradPerIndiceBetweenBlock(LocalTensor<GRAD_T>& outTensor,
                                            const LocalTensor<CAST_T>& gradTensor, uint32_t colLen, uint32_t colLenAlign, int32_t curGradCount, float freqInBlock)
{
    auto ubGradAddr = (__ubuf__ CAST_T *)gradTensor.GetPhyAddr();
    auto ubResultAddr = (__ubuf__ GRAD_T *)outTensor.GetPhyAddr();

    uint32_t processColLen = colLen;
    uint32_t perColAlign = colLenAlign;
    uint16_t perColLoopNum = ops::CeilDiv(processColLen, static_cast<uint32_t>(vfFp32Block_));
    uint16_t loopSegCount = curGradCount / PROCESS_GROUP;
    uint16_t normalSegCount = loopSegCount * PROCESS_GROUP;
    uint16_t loopSegCountTail = curGradCount - normalSegCount;
    float scaleFreq = freqInBlock;

    __VEC_SCOPE__ {
        MicroAPI::RegTensor<float> srcGrad;
        MicroAPI::RegTensor<float> groupAdd;
        MicroAPI::RegTensor<float> gradResult;

        MicroAPI::MaskReg preg = MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        for (uint16_t col = 0; col < perColLoopNum; col++) { // grad 一行
            uint32_t colOffset = col * vfFp32Block_;  // 列方向的偏移
            Duplicate(gradResult, (float)0, preg);

            for (uint16_t mainSeg = 0; mainSeg < loopSegCount; mainSeg++) {
                Duplicate(groupAdd, (float)0, preg);

                uint32_t groupRowStart = mainSeg * PROCESS_GROUP;
                for (uint16_t group = 0; group < PROCESS_GROUP; group++) {
                    uint32_t gradOffset = (groupRowStart + group) * perColAlign + colOffset;
                    ops::LoadOneTensorForDtypeT<CAST_T>(ubGradAddr, srcGrad, preg, gradOffset);
                    Add(groupAdd, groupAdd, srcGrad, preg);
                }
                Add(gradResult, gradResult, groupAdd, preg);
            }

            Duplicate(groupAdd, (float)0, preg);
            for (uint16_t tailRow = 0; tailRow < loopSegCountTail; tailRow++) {
                uint32_t gradOffset = (normalSegCount + tailRow) * perColAlign + colOffset;
                ops::LoadOneTensorForDtypeT<CAST_T>(ubGradAddr, srcGrad, preg, gradOffset);
                Add(groupAdd, groupAdd, srcGrad, preg);
            }
            Add(gradResult, gradResult, groupAdd, preg);
            if constexpr (isScale) {
                Muls(gradResult, gradResult, scaleFreq, preg);
            }

            ops::StoreOneTensorForDtypeT<GRAD_T>(ubResultAddr, gradResult, preg, colOffset);
            preg = MicroAPI::UpdateMask<float>(processColLen);
        }
    }
}

template <typename GRAD_T, typename CAST_T, typename IDX_T, bool isScale, bool SplitEltwiseAxis, int32_t bufferNum>
__aicore__ inline void EmbeddingDenseGradV2Kernel<GRAD_T, CAST_T, IDX_T, isScale, SplitEltwiseAxis, bufferNum>::AddGradBetweenBlock(uint32_t colLen, uint32_t gradColAlign,
                                                                                        LocalTensor<int32_t>& noDupCount, int64_t &arNum, uint32_t totalIndicesNum)
{
    LocalTensor<CAST_T> gradTensor = inQueGrad_.template DeQue<CAST_T>();
    LocalTensor<GRAD_T> outTensor = outQueGradWeight_.template AllocTensor<GRAD_T>();
    LocalTensor<IDX_T> freqTensor = lastResBuf_.Get<IDX_T>();
    LocalTensor<IDX_T> indicesTensor = indicesBuf_.Get<IDX_T>()[indexCountInBlock_];

    event_t eventV_S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventV_S);
    WaitFlag<HardEvent::V_S>(eventV_S);

    uint32_t gradRowStart = 0;
    uint32_t idxOffset = 0;
    for (int64_t i = 0; i < arNum; i++) {
        IDX_T curGradCount = noDupCount(i);
        IDX_T curIdx = indicesTensor(idxOffset);
        idxOffset = idxOffset + curGradCount;
        if (curIdx != tiling_.paddingIdx && curIdx >= 0) {
            uint32_t resultRow = i * elewiseAligned_;
            LocalTensor<GRAD_T> outLocal = outTensor[resultRow];
            LocalTensor<CAST_T> gradLocal = gradTensor[gradRowStart * gradColAlign];
            if constexpr (isScale) {
                IDX_T freq = freqTensor(i);
                AddGradPerIndiceBetweenBlock(outLocal, gradLocal, colLen, gradColAlign, curGradCount, 1.0f / static_cast<float>(freq));
            } else {
                AddGradPerIndiceBetweenBlock(outLocal, gradLocal, colLen, gradColAlign, curGradCount, 1.0f);
            }
        }
        gradRowStart = gradRowStart + curGradCount;
    }

    outQueGradWeight_.template EnQue<GRAD_T>(outTensor);
    inQueGrad_.FreeTensor(gradTensor);
}

template <typename GRAD_T, typename CAST_T, typename IDX_T, bool isScale, bool SplitEltwiseAxis, int32_t bufferNum>
__aicore__ inline void EmbeddingDenseGradV2Kernel<GRAD_T, CAST_T, IDX_T, isScale, SplitEltwiseAxis, bufferNum>::CopyIndicesToWs()
{
    LocalTensor<IDX_T> indicesTensor = indicesBuf_.Get<IDX_T>();
    LocalTensor<IDX_T> noDupCountTensor = noDupIdxCountBuf_.Get<IDX_T>();

    event_t eventS_V= static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventS_V);
    WaitFlag<HardEvent::S_V>(eventS_V);

    Duplicate(indicesTensor, (IDX_T)lastResIdx_, indexCountInBlock_);    // End indices
    Duplicate(noDupCountTensor, (IDX_T)lastResCount_, indexCountInBlock_);  // End Freq

    DataCopyExtParams copyIdxParam {
        static_cast<uint16_t>(1),
        static_cast<uint32_t>(sizeof(IDX_T)),
        static_cast<uint32_t>(0),
        static_cast<uint32_t>(0),
        static_cast<uint32_t>(0)
    };

    if (!isBeginRecord_ || firstResIdx_ == lastResIdx_) {
        /*
        * 1、从头到尾只有一个index
        * 2、从头到尾有两个index，其中一个index == padding_index
        */
        Duplicate(indicesTensor[indexCountInBlock_], (IDX_T)lastResIdx_, indexCountInBlock_);    // Begin indices
        Duplicate(noDupCountTensor[indexCountInBlock_], (IDX_T)0, indexCountInBlock_);  // Begin Freq

        InitGlobalMemory(gradBeginWsGm_, tiling_.elewiseDimSize, (CAST_T)0);
    } else {
        Duplicate(indicesTensor[indexCountInBlock_], (IDX_T)firstResIdx_, indexCountInBlock_);    // Begin indices
        Duplicate(noDupCountTensor[indexCountInBlock_], (IDX_T)firstResCount_, indexCountInBlock_);  // Begin Freq

        event_t eventV_MTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventV_MTE3);
        WaitFlag<HardEvent::V_MTE3>(eventV_MTE3);
    }

    DataCopyPad(idxEndWsGm_, indicesTensor, copyIdxParam);
    DataCopyPad(idxEndFreqWsGm_, noDupCountTensor, copyIdxParam);
    DataCopyPad(idxBeginWsGm_, indicesTensor[indexCountInBlock_], copyIdxParam);
    DataCopyPad(idxBeginFreqWsGm_, noDupCountTensor[indexCountInBlock_], copyIdxParam);
}

template <typename GRAD_T, typename CAST_T, typename IDX_T, bool isScale, bool SplitEltwiseAxis, int32_t bufferNum>
__aicore__ inline void EmbeddingDenseGradV2Kernel<GRAD_T, CAST_T, IDX_T, isScale, SplitEltwiseAxis, bufferNum>::ProcessNotSplitEltwiseAxis()
{
    int64_t offset = 0;
    LocalTensor<int32_t> posIdxTensor = posIdxBuf_.Get<int32_t>();
    LocalTensor<IDX_T> indicesTensor = indicesBuf_.Get<IDX_T>();
    LocalTensor<int32_t> noDupCountTensor = noDupIdxCountBuf_.Get<int32_t>();
    LocalTensor<CAST_T> lastResTensor = lastResBuf_.Get<CAST_T>();
    Duplicate(lastResTensor, (CAST_T)0, tiling_.elewiseDimSize);
    int64_t arNum = 0;
    // Normal block
    for (int64_t i = 0; i < scatterLoopNums_ - 1; i++) {
        offset = i * tiling_.scatterFactor;
        CopyIndices(indicesTensor, posIdxTensor, tiling_.scatterFactor, offset);
        CalcDupNum(noDupCountTensor, arNum, indicesTensor, tiling_.scatterFactor);
        CopyGrad(posIdxTensor, tiling_.scatterFactor, tiling_.elewiseFactor, 0);
        LocalTensor<GRAD_T> gradTensor = inQueGrad_.template DeQue<GRAD_T>();
        this->template AddFirstGradByGroup<true>(gradTensor, tiling_.elewiseFactor, 0, noDupCountTensor, arNum, 0);
        AddGradByGroup(gradTensor, tiling_.elewiseFactor, 0, noDupCountTensor, arNum, 0);
        inQueGrad_.FreeTensor(gradTensor);
        this->template CopyOutGradWeight<false, true, true>(indicesTensor, noDupCountTensor, arNum, tiling_.elewiseFactor, 0, 0);
    }

    // Tail block
    offset = (scatterLoopNums_ - 1) * tiling_.scatterFactor;
    CopyIndices(indicesTensor, posIdxTensor, scatterTail_, offset);
    CalcDupNum(noDupCountTensor, arNum, indicesTensor, scatterTail_);

    CopyGrad(posIdxTensor, scatterTail_, tiling_.elewiseFactor, 0);
    LocalTensor<GRAD_T> gradTensor = inQueGrad_.template DeQue<GRAD_T>();
    this->template AddFirstGradByGroup<true>(gradTensor, tiling_.elewiseFactor, 0, noDupCountTensor, arNum, 0);
    AddGradByGroup(gradTensor, tiling_.elewiseFactor, 0, noDupCountTensor, arNum, 0);
    inQueGrad_.FreeTensor(gradTensor);
    this->template CopyOutGradWeight<true, true, true>(indicesTensor, noDupCountTensor, arNum, tiling_.elewiseFactor, 0, 0);
    CopyIndicesToWs();

    SyncAll();
    if (tiling_.usedCoreNum > 1 && blockIdx_ == 0) {
        event_t eventMTE3_MTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventMTE3_MTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventMTE3_MTE2);
        ProcessGradBetweenBlock();
    }
}

template <typename GRAD_T, typename CAST_T, typename IDX_T, bool isScale, bool SplitEltwiseAxis, int32_t bufferNum>
template <bool isColTail>
__aicore__ inline void EmbeddingDenseGradV2Kernel<GRAD_T, CAST_T, IDX_T, isScale, SplitEltwiseAxis, bufferNum>::ProcessSplitEltwiseAxis(int64_t colBaseOffset)
{
    int64_t offset = 0;
    LocalTensor<int32_t> posIdxTensor = posIdxBuf_.Get<int32_t>();
    LocalTensor<IDX_T> indicesTensor = indicesBuf_.Get<IDX_T>();
    LocalTensor<int32_t> noDupCountTensor = noDupIdxCountBuf_.Get<int32_t>();
    LocalTensor<CAST_T> lastResTensor = lastResBuf_.Get<CAST_T>();
    constexpr int32_t lastResTensorNum = static_cast<int32_t>(MAX_ELEWISE_AXIS_LIMIT / sizeof(CAST_T));
    Duplicate(lastResTensor, (CAST_T)0, lastResTensorNum);

    int64_t arNum = 0;
    uint32_t colTailOffset = (elewiseLoopNums_ - 1) * tiling_.elewiseFactor + colBaseOffset;
    // Normal block
    for (int64_t i = 0; i < scatterLoopNums_ - 1; i++) {
        offset = i * tiling_.scatterFactor;
        CopyIndices(indicesTensor, posIdxTensor, tiling_.scatterFactor, offset);
        CalcDupNum(noDupCountTensor, arNum, indicesTensor, tiling_.scatterFactor);
        uint32_t colOffset = colBaseOffset;
        uint32_t resColOffset = 0;
        for (int64_t colLoop = 0; colLoop < elewiseLoopNums_ -1; colLoop++) {
            CopyGrad(posIdxTensor, tiling_.scatterFactor, tiling_.elewiseFactor, colOffset);
            LocalTensor<GRAD_T> gradTensor = inQueGrad_.template DeQue<GRAD_T>();
            this->template AddFirstGradByGroup<false>(gradTensor, tiling_.elewiseFactor, colOffset, noDupCountTensor, arNum, resColOffset);
            AddGradByGroup(gradTensor, tiling_.elewiseFactor, colOffset, noDupCountTensor, arNum, resColOffset);
            inQueGrad_.FreeTensor(gradTensor);
            this->template CopyOutGradWeight<false, false, false>(indicesTensor, noDupCountTensor, arNum, tiling_.elewiseFactor, colOffset, resColOffset);
            colOffset = colOffset + tiling_.elewiseFactor;
            resColOffset = resColOffset + tiling_.elewiseFactor;
        }

        CopyGrad(posIdxTensor, tiling_.scatterFactor, elewiseTail_, colOffset);
        LocalTensor<GRAD_T> gradTensor = inQueGrad_.template DeQue<GRAD_T>();
        this->template AddFirstGradByGroup<true>(gradTensor, elewiseTail_, colOffset, noDupCountTensor, arNum, resColOffset);
        AddGradByGroup(gradTensor, elewiseTail_, colOffset, noDupCountTensor, arNum, resColOffset);
        inQueGrad_.FreeTensor(gradTensor);
        this->template CopyOutGradWeight<false, true, true>(indicesTensor, noDupCountTensor, arNum, elewiseTail_, colOffset, resColOffset);
    }

    // Tail block
    offset = (scatterLoopNums_ - 1) * tiling_.scatterFactor;
    CopyIndices(indicesTensor, posIdxTensor, scatterTail_, offset);
    CalcDupNum(noDupCountTensor, arNum, indicesTensor, scatterTail_);
    uint32_t colOffset = colBaseOffset;
    uint32_t resColOffset = 0;
    for (int64_t colLoop = 0; colLoop < elewiseLoopNums_ -1; colLoop++) {
        CopyGrad(posIdxTensor, scatterTail_, tiling_.elewiseFactor, colOffset);
        LocalTensor<GRAD_T> gradTensor = inQueGrad_.template DeQue<GRAD_T>();
        this->template AddFirstGradByGroup<false>(gradTensor, tiling_.elewiseFactor, colOffset, noDupCountTensor, arNum, resColOffset);
        AddGradByGroup(gradTensor, tiling_.elewiseFactor, colOffset, noDupCountTensor, arNum, resColOffset);
        inQueGrad_.FreeTensor(gradTensor);
        this->template CopyOutGradWeight<true, false, false>(indicesTensor, noDupCountTensor, arNum, tiling_.elewiseFactor, colOffset, resColOffset);
        colOffset = colOffset + tiling_.elewiseFactor;
        resColOffset = resColOffset + tiling_.elewiseFactor;
    }

    CopyGrad(posIdxTensor, scatterTail_, elewiseTail_, colOffset);
    LocalTensor<GRAD_T> gradTensor = inQueGrad_.template DeQue<GRAD_T>();
    this->template AddFirstGradByGroup<true>(gradTensor, elewiseTail_, colOffset, noDupCountTensor, arNum, resColOffset);
    AddGradByGroup(gradTensor, elewiseTail_, colOffset, noDupCountTensor, arNum, resColOffset);
    inQueGrad_.FreeTensor(gradTensor);
    this->template CopyOutGradWeight<true, isColTail, true>(indicesTensor, noDupCountTensor, arNum, elewiseTail_, colOffset, resColOffset);
}

template <typename GRAD_T, typename CAST_T, typename IDX_T, bool isScale, bool SplitEltwiseAxis, int32_t bufferNum>
__aicore__ inline void EmbeddingDenseGradV2Kernel<GRAD_T, CAST_T, IDX_T, isScale, SplitEltwiseAxis, bufferNum>::Process()
{
    if constexpr (!SplitEltwiseAxis) {
        ProcessNotSplitEltwiseAxis();
    } else {
        int64_t colBaseOffset = 0;
        int64_t normalElewiseNum = MAX_ELEWISE_AXIS_LIMIT / sizeof(CAST_T);
        for (int64_t i = 0; i < tiling_.elewiseDimOuterLoopNum - 1; i++) {
            isBeginRecord_ = false;
            isBeginPadding_ = false;
            firstResIdx_ = -2;
            firstResCount_ = 0;
            lastResIdx_ = -2;
            lastResCount_ = 0;
            PipeBarrier<PIPE_ALL>();
            ProcessSplitEltwiseAxis<false>(colBaseOffset);
            colBaseOffset = colBaseOffset + normalElewiseNum;
        }

        isBeginRecord_ = false;
        isBeginPadding_ = false;
        firstResIdx_ = -2;
        firstResCount_ = 0;
        lastResIdx_ = -2;
        lastResCount_ = 0;
        elewiseLoopNums_ = ops::CeilDiv(elewiseOuterTail_, static_cast<int64_t>(tiling_.elewiseFactor));
        elewiseTail_ = elewiseOuterTail_ - (elewiseLoopNums_ - 1) * tiling_.elewiseFactor;
        PipeBarrier<PIPE_ALL>();
        ProcessSplitEltwiseAxis<true>(colBaseOffset);
        CopyIndicesToWs();

        SyncAll();
        if (tiling_.usedCoreNum > 1 && blockIdx_ < tiling_.blockNumForLastProcess) {
            event_t eventMTE3_MTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
            SetFlag<HardEvent::MTE3_MTE2>(eventMTE3_MTE2);
            WaitFlag<HardEvent::MTE3_MTE2>(eventMTE3_MTE2);
            ProcessGradBetweenBlock();
        }
    }
}

template <typename GRAD_T, typename CAST_T, typename IDX_T, bool isScale, bool SplitEltwiseAxis, int32_t bufferNum>
__aicore__ inline void EmbeddingDenseGradV2Kernel<GRAD_T, CAST_T, IDX_T, isScale, SplitEltwiseAxis, bufferNum>::CopyIndicesFromWs(IDX_T indicesNumber, uint32_t endOffset)
{
    // copy in
    LocalTensor<IDX_T> indicesBeginTensor = inQueGrad_.template AllocTensor<IDX_T>();
    LocalTensor<IDX_T> indicesEndTensor = indicesBeginTensor[endOffset];

    uint8_t rightPadNum = indexCountInBlock_ - 1;
    DataCopyPadExtParams<IDX_T> padParams{true, static_cast<uint8_t>(0), static_cast<uint8_t>(rightPadNum), static_cast<uint8_t>(-1)};

    uint32_t stride = (CACHE_LINE * DOUBLE + gradLenInWs_) * DOUBLE - sizeof(IDX_T);

    DataCopyExtParams copyIdxParam {
        static_cast<uint16_t>(indicesNumber),
        static_cast<uint32_t>(sizeof(IDX_T)),
        static_cast<uint32_t>(stride),
        static_cast<uint32_t>(0),
        static_cast<uint32_t>(0)
    };

    int64_t beginOffset = (CACHE_LINE + CACHE_LINE) / sizeof(IDX_T) +  gradLenInWs_ / sizeof(IDX_T);

    DataCopyPad(indicesBeginTensor, idxStartGm_[beginOffset], copyIdxParam, padParams); // start1

    copyIdxParam.blockCount = static_cast<uint16_t>(indicesNumber + 1);
    DataCopyPad(indicesEndTensor, idxStartGm_, copyIdxParam, padParams); // End0

    inQueGrad_.template EnQue<IDX_T>(indicesBeginTensor);
}

template <typename GRAD_T, typename CAST_T, typename IDX_T, bool isScale, bool SplitEltwiseAxis, int32_t bufferNum>
__aicore__ inline bool EmbeddingDenseGradV2Kernel<GRAD_T, CAST_T, IDX_T, isScale, SplitEltwiseAxis, bufferNum>::CheckHasSameIndices(uint32_t indicesNumber, uint32_t endOffset)
{
    LocalTensor<IDX_T> indicesBeginTensor = inQueGrad_.template DeQue<IDX_T>();
    LocalTensor<IDX_T> indicesEndTensor = indicesBeginTensor[endOffset];
    GatherIndices(indicesBeginTensor, indicesEndTensor, indicesNumber);

    LocalTensor<uint8_t> cmpResult = indicesBuf_.Get<uint8_t>();
    Compare(cmpResult, indicesEndTensor, indicesBeginTensor, CMPMODE::EQ, indicesNumber);

    inQueGrad_.template EnQue<IDX_T>(indicesBeginTensor);

    LocalTensor<uint64_t> checkResult = cmpResult.template ReinterpretCast<uint64_t>();
    int64_t sameIdxCount = ScalarGetCountOfValue<1>(checkResult(0));  // 统计 bit=1的数量，bit 1 代表核间的头和尾相等，需要累加
    return sameIdxCount != 0;
}

template <typename GRAD_T, typename CAST_T, typename IDX_T, bool isScale, bool SplitEltwiseAxis, int32_t bufferNum>
__aicore__ inline void EmbeddingDenseGradV2Kernel<GRAD_T, CAST_T, IDX_T, isScale, SplitEltwiseAxis, bufferNum>::CopyScaleFreqFromWs(uint32_t totalIndicesNum)
{
    LocalTensor<IDX_T> freqTensor = lastResBuf_.Get<IDX_T>();
    uint8_t rightPadNum = indexCountInBlock_ - 1;
    DataCopyPadExtParams<IDX_T> padParams{true, static_cast<uint8_t>(0), static_cast<uint8_t>(rightPadNum), static_cast<uint8_t>(0)};

    uint32_t stride = CACHE_LINE * DOUBLE + gradLenInWs_ - sizeof(IDX_T);

    DataCopyExtParams copyIdxParam {
        static_cast<uint16_t>(totalIndicesNum),
        static_cast<uint32_t>(sizeof(IDX_T)),
        static_cast<uint32_t>(stride),
        static_cast<uint32_t>(0),
        static_cast<uint32_t>(0)
    };

    constexpr uint64_t scaleFreqStart = CACHE_LINE / sizeof(IDX_T);

    DataCopyPad(freqTensor, idxStartGm_[scaleFreqStart], copyIdxParam, padParams);  // End0 freq
    event_t eventMTE2_V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventMTE2_V);
    WaitFlag<HardEvent::MTE2_V>(eventMTE2_V);
}

template <typename GRAD_T, typename CAST_T, typename IDX_T, bool isScale, bool SplitEltwiseAxis, int32_t bufferNum>
__aicore__ inline void EmbeddingDenseGradV2Kernel<GRAD_T, CAST_T, IDX_T, isScale, SplitEltwiseAxis, bufferNum>::GatherScaleFreq(uint32_t totalIndicesNum)
{
    LocalTensor<IDX_T> freqTensor = lastResBuf_.Get<IDX_T>();
    LocalTensor<IDX_T> dstFreqTensor = posIdxBuf_.Get<IDX_T>();

    auto ubFreqAddr = (__ubuf__ IDX_T*)freqTensor.GetPhyAddr();
    auto ubDstAddr = (__ubuf__ IDX_T*)dstFreqTensor.GetPhyAddr();

    IDX_T interval = indexCountInBlock_;
    constexpr static uint32_t perVfProcessNum = platform::GetVRegSize() / sizeof(int32_t);
    IDX_T loopCnt = totalIndicesNum / perVfProcessNum;
    uint32_t tail = totalIndicesNum - loopCnt * perVfProcessNum;
    uint32_t srcStride = perVfProcessNum * interval;
    uint32_t dstStride = perVfProcessNum;

    __VEC_SCOPE__ {
        RegTensorType index;
        RegTensorType indexOri;
        RegCastType indexCast;
        RegTensorType vregFreq;

        MicroAPI::MaskReg pMaskAll;
        if constexpr (IsSameType<IDX_T, int64_t>::value) {
            pMaskAll = MicroAPI::CreateMask<IDX_T, MicroAPI::MaskPattern::ALL, MicroAPI::RegTraitNumTwo>();
        } else {
            pMaskAll = MicroAPI::CreateMask<IDX_T, MicroAPI::MaskPattern::ALL>();
        }

        MicroAPI::Arange(index, 0);
        Muls(index, index, (IDX_T)interval, pMaskAll);
        indexCast = (RegCastType&)index;

        for (uint16_t i = 0; i < loopCnt; ++i) {
            MicroAPI::DataCopyGather(vregFreq, (__local_mem__ IDX_T *)(ubFreqAddr), indexCast, pMaskAll);
            MicroAPI::DataCopy((__local_mem__ IDX_T *)ubDstAddr, vregFreq, pMaskAll);
            ubFreqAddr = ubFreqAddr + srcStride;
            ubDstAddr = ubDstAddr + dstStride;
        }

        MicroAPI::MaskReg pTail;
        if constexpr (IsSameType<IDX_T, int64_t>::value) {
            pTail = MicroAPI::UpdateMask<IDX_T, MicroAPI::RegTraitNumTwo>(tail);
        } else {
            pTail = MicroAPI::UpdateMask<IDX_T>(tail);
        }

        MicroAPI::DataCopyGather(vregFreq, (__local_mem__ IDX_T *)(ubFreqAddr), indexCast, pTail);
        MicroAPI::DataCopy((__local_mem__ IDX_T *)ubDstAddr, vregFreq, pTail);
    }
}

template <typename GRAD_T, typename CAST_T, typename IDX_T, bool isScale, bool SplitEltwiseAxis, int32_t bufferNum>
__aicore__ inline void EmbeddingDenseGradV2Kernel<GRAD_T, CAST_T, IDX_T, isScale, SplitEltwiseAxis, bufferNum>::CalcFreqBetweenBlock(LocalTensor<int32_t> &noDupCount, int64_t arNum, uint32_t totalIndicesNum)
{
    LocalTensor<IDX_T> freqTensor = posIdxBuf_.Get<IDX_T>();
    LocalTensor<IDX_T> freqResTensor = lastResBuf_.Get<IDX_T>();
    __local_mem__ IDX_T* ubFreqBaseSrcAddr = (__local_mem__ IDX_T*)freqTensor.GetPhyAddr();
    __local_mem__ IDX_T* ubFreqBaseDstAddr = (__local_mem__ IDX_T*)freqResTensor.GetPhyAddr();

    LocalTensor<int32_t> gradCount = noDupCount;
    uint16_t noDupCountNum = arNum;
    constexpr static uint32_t perVfProcessNum = platform::GetVRegSize() / sizeof(int32_t);
    const uint32_t vfIndicesLen = perVfProcessNum;

    __local_mem__ IDX_T* ubFreqSrcAddr = ubFreqBaseSrcAddr;
    __local_mem__ IDX_T* ubFreqDstAddr = ubFreqBaseDstAddr;
    for (uint16_t i = 0; i < noDupCountNum; i++) {
        uint32_t countNum = static_cast<uint32_t>(gradCount(i));
        uint32_t sumLoop = countNum / perVfProcessNum;
        uint32_t tail = countNum - sumLoop * perVfProcessNum;
        __VEC_SCOPE__ {
            RegTensorType freqReg;
            RegTensorType sumReg;
            RegTensorType resultReg;
            MicroAPI::UnalignReg udst;
            MicroAPI::UnalignReg uSrc;
            MicroAPI::MaskReg pMaskAll;
            if constexpr (IsSameType<IDX_T, int64_t>::value) {
                pMaskAll = MicroAPI::CreateMask<IDX_T, MicroAPI::MaskPattern::ALL, MicroAPI::RegTraitNumTwo>();
            } else {
                pMaskAll = MicroAPI::CreateMask<IDX_T, MicroAPI::MaskPattern::ALL>();
            }

            MicroAPI::Duplicate(resultReg, 0);
            MicroAPI::DataCopyUnAlignPre(uSrc, ubFreqSrcAddr);
            for (uint16_t j = 0; j < (uint16_t)sumLoop; j++) {
                MicroAPI::DataCopyUnAlign<IDX_T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(freqReg, uSrc, ubFreqSrcAddr, vfIndicesLen);
                MicroAPI::ReduceSum(sumReg, freqReg, pMaskAll);
                MicroAPI::Add(resultReg, resultReg, sumReg, pMaskAll);
            }

            MicroAPI::DataCopyUnAlignPre(uSrc, ubFreqSrcAddr);
            MicroAPI::DataCopyUnAlign<IDX_T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(freqReg, uSrc, ubFreqSrcAddr, tail);
            MicroAPI::MaskReg preg;
            if constexpr (IsSameType<IDX_T, int64_t>::value) {
                preg = MicroAPI::UpdateMask<IDX_T, MicroAPI::RegTraitNumTwo>(tail);
            } else {
                preg = MicroAPI::UpdateMask<IDX_T>(tail);
            }
            MicroAPI::ReduceSum(sumReg, freqReg, preg);
            MicroAPI::Add(resultReg, resultReg, sumReg, preg);

            MicroAPI::DataCopyUnAlign<IDX_T, MicroAPI::PostLiteral::POST_MODE_UPDATE>(ubFreqDstAddr, resultReg, udst, 1);
            MicroAPI::DataCopyUnAlignPost(ubFreqDstAddr, udst, 0);
        }
        ubFreqDstAddr = ubFreqDstAddr + 1;
        ubFreqSrcAddr = ubFreqSrcAddr + countNum;
    }
}

template <typename GRAD_T, typename CAST_T, typename IDX_T, bool isScale, bool SplitEltwiseAxis, int32_t bufferNum>
__aicore__ inline void EmbeddingDenseGradV2Kernel<GRAD_T, CAST_T, IDX_T, isScale, SplitEltwiseAxis, bufferNum>::GatherIndices(LocalTensor<IDX_T>& indicesBeginTensor, LocalTensor<IDX_T>& indicesEndTensor, uint32_t indicesNumber)
{
    auto ubBeginAddr = (__ubuf__ IDX_T*)indicesBeginTensor.GetPhyAddr();
    auto ubEndAddr = (__ubuf__ IDX_T*)indicesEndTensor.GetPhyAddr();

    IDX_T interval = indexCountInBlock_;
    uint32_t beginNumber = indicesNumber;
    uint32_t endNumber = indicesNumber + 1;

    __VEC_SCOPE__ {
        RegTensorType index;
        RegCastType indexCast;
        RegTensorType vregBegin;
        RegTensorType vregEnd;

        MicroAPI::MaskReg beginMask;
        MicroAPI::MaskReg endMask;
        if constexpr (IsSameType<IDX_T, int64_t>::value) {
            beginMask = MicroAPI::UpdateMask<IDX_T, MicroAPI::RegTraitNumTwo>(beginNumber);
            endMask = MicroAPI::UpdateMask<IDX_T, MicroAPI::RegTraitNumTwo>(endNumber);
        } else {
            beginMask = MicroAPI::UpdateMask<IDX_T>(beginNumber);
            endMask = MicroAPI::UpdateMask<IDX_T>(endNumber);
        }

        MicroAPI::Arange(index, 0);

        Muls(index, index, (IDX_T)interval, endMask);
        indexCast = (RegCastType&)index;

        MicroAPI::DataCopyGather(vregBegin, (__local_mem__ IDX_T *)(ubBeginAddr), indexCast, beginMask);
        MicroAPI::DataCopyGather(vregEnd, (__local_mem__ IDX_T *)(ubEndAddr), indexCast, endMask);

        MicroAPI::DataCopy((__local_mem__ IDX_T *)ubBeginAddr, vregBegin, beginMask);
        MicroAPI::DataCopy((__local_mem__ IDX_T *)ubEndAddr, vregEnd, endMask);
    }
}

template <typename GRAD_T, typename CAST_T, typename IDX_T, bool isScale, bool SplitEltwiseAxis, int32_t bufferNum>
__aicore__ inline void EmbeddingDenseGradV2Kernel<GRAD_T, CAST_T, IDX_T, isScale, SplitEltwiseAxis, bufferNum>::InterleaveIndices(uint32_t totalIndicesNum, uint32_t endOffset)
{
    LocalTensor<IDX_T> indicesBeginTensor = inQueGrad_.template DeQue<IDX_T>();
    LocalTensor<IDX_T> indicesEndTensor = indicesBeginTensor[endOffset];
    LocalTensor<IDX_T> rearrangeIndices = indicesBuf_.Get<IDX_T>();

    Duplicate(rearrangeIndices, (IDX_T)-1, totalIndicesNum + PADDING_SIZE / sizeof(IDX_T));

    auto ubIdxBeginAddr = (__ubuf__ IDX_T*)indicesBeginTensor.GetPhyAddr();
    auto ubIdxEndAddr = (__ubuf__ IDX_T*)indicesEndTensor.GetPhyAddr();
    auto ubDstAddr = (__ubuf__ IDX_T*)rearrangeIndices.GetPhyAddr(indexCountInBlock_);

    uint32_t tailNum = totalIndicesNum > perVfIndices_? (totalIndicesNum - perVfIndices_) : 0;
    uint32_t stride = perVfIndices_;
    if constexpr (IsSameType<IDX_T, int64_t>::value) {
        stride = perVfIndices_ * DOUBLE;
    }

    __VEC_SCOPE__
    {
        RegTensorType vSrcReg0;
        RegTensorType vSrcReg1;
        RegTensorType vDstReg0;
        RegTensorType vDstReg1;

        AscendC::MicroAPI::DataCopy(vSrcReg0, ubIdxEndAddr);
        AscendC::MicroAPI::DataCopy(vSrcReg1, ubIdxBeginAddr);
        AscendC::MicroAPI::Interleave(vDstReg0, vDstReg1, vSrcReg0, vSrcReg1);

        MicroAPI::MaskReg lowerMask;
        MicroAPI::MaskReg higherMask;
        if constexpr (IsSameType<IDX_T, int64_t>::value) {
            lowerMask = AscendC::MicroAPI::UpdateMask<IDX_T, MicroAPI::RegTraitNumTwo>(totalIndicesNum);
            higherMask = AscendC::MicroAPI::UpdateMask<IDX_T, MicroAPI::RegTraitNumTwo>(tailNum);
        } else {
            lowerMask = AscendC::MicroAPI::UpdateMask<IDX_T>(totalIndicesNum);
            higherMask = AscendC::MicroAPI::UpdateMask<IDX_T>(tailNum);
        }

        AscendC::MicroAPI::DataCopy(ubDstAddr, vDstReg0, lowerMask);
        AscendC::MicroAPI::DataCopy(ubDstAddr + stride, vDstReg1, higherMask);
    }

    inQueGrad_.FreeTensor(indicesBeginTensor);
}

template <typename GRAD_T, typename CAST_T, typename IDX_T, bool isScale, bool SplitEltwiseAxis, int32_t bufferNum>
__aicore__ inline void EmbeddingDenseGradV2Kernel<GRAD_T, CAST_T, IDX_T, isScale, SplitEltwiseAxis, bufferNum>::CopyGradFromWs(IDX_T indicesNumber, uint32_t elewiseNumber, uint32_t stride, int64_t gmOffset)
{
    LocalTensor<CAST_T> gradTensor = inQueGrad_.template AllocTensor<CAST_T>();
    uint8_t gradPadNum = static_cast<uint8_t>(ops::CeilAlign(elewiseNumber, static_cast<uint32_t>(ONE_BLK_SIZE / sizeof(CAST_T))) - elewiseNumber);
    DataCopyPadExtParams<CAST_T> padParams{true, static_cast<uint8_t>(0), static_cast<uint8_t>(gradPadNum), static_cast<uint8_t>(0)};
    DataCopyExtParams copyParam {
        static_cast<uint16_t>(indicesNumber),
        static_cast<uint32_t>(elewiseNumber * sizeof(CAST_T)),
        static_cast<uint32_t>(stride),
        static_cast<uint32_t>(0),
        static_cast<uint32_t>(0)
    };
    DataCopyPad(gradTensor, gradStartGm_[gmOffset], copyParam, padParams);
    inQueGrad_.template EnQue<CAST_T>(gradTensor);
}

template <typename GRAD_T, typename CAST_T, typename IDX_T, bool isScale, bool SplitEltwiseAxis, int32_t bufferNum>
__aicore__ inline void EmbeddingDenseGradV2Kernel<GRAD_T, CAST_T, IDX_T, isScale, SplitEltwiseAxis, bufferNum>::ReInitBuffer(uint32_t totalIndicesNumber)
{
    pipe_.Reset();

    uint32_t indicesBufSize = ops::CeilAlign(totalIndicesNumber + 1, indexCountInBlock_) * sizeof(IDX_T);
    uint32_t freqBufferSize = ops::CeilAlign(totalIndicesNumber + 1, indexCountInBlock_) * ONE_BLK_SIZE; // 包含Begin和End
    elewiseNormal_ = tiling_.elewiseDimLastPerUbProcessNum;
    if (tiling_.elewiseDimSize < tiling_.elewiseDimLastPerUbProcessNum) {
        elewiseNormal_ = tiling_.elewiseDimSize;
    }

    if (blockIdx_ == (tiling_.blockNumForLastProcess - 1)) {
        elewiseLoopNums_ = tiling_.elewiseDimLastTailLoopNum;
        elewiseTail_ = tiling_.elewiseDimLastProcessTailNum;
    } else {
        elewiseLoopNums_ = tiling_.elewiseDimLastNormalLoopNum;
        elewiseTail_ = tiling_.elewiseDimLastPerUbProcessNum;
    }

    uint32_t gradTotalNumPerUb = totalIndicesNumber * tiling_.elewiseDimLastPerUbProcessNum;
    pipe_.InitBuffer(inQueGrad_, 1, gradTotalNumPerUb * sizeof(float));
    pipe_.InitBuffer(outQueGradWeight_, 1, gradTotalNumPerUb * sizeof(GRAD_T));
    pipe_.InitBuffer(indicesBuf_, indicesBufSize + PADDING_SIZE);
    pipe_.InitBuffer(noDupIdxCountBuf_, indicesBufSize + PADDING_SIZE);
    pipe_.InitBuffer(posIdxBuf_, indicesBufSize * DOUBLE);
    pipe_.InitBuffer(lastResBuf_, freqBufferSize);
}

template <typename GRAD_T, typename CAST_T, typename IDX_T, bool isScale, bool SplitEltwiseAxis, int32_t bufferNum>
__aicore__ inline void EmbeddingDenseGradV2Kernel<GRAD_T, CAST_T, IDX_T, isScale, SplitEltwiseAxis, bufferNum>::ProcessGradBetweenBlock()
{
    IDX_T indicesNumber = tiling_.usedCoreNum - 1;
    uint32_t endOffset = indicesNumber * indexCountInBlock_;
    uint32_t totalIndicesNumber = tiling_.usedCoreNum * DOUBLE - 1; // End0, Begin1, End1 ... End(usedCore - 1)
    ReInitBuffer(totalIndicesNumber);
    CopyIndicesFromWs(indicesNumber, endOffset);

    if (CheckHasSameIndices(indicesNumber, endOffset)) {  // 核间存在相同的索引
        int64_t arNum = 0;
        InterleaveIndices(totalIndicesNumber, endOffset);
        LocalTensor<IDX_T> indicesTensor = indicesBuf_.Get<IDX_T>();
        LocalTensor<int32_t> noDupCountTensor = noDupIdxCountBuf_.Get<int32_t>();
        CalcDupNum(noDupCountTensor, arNum, indicesTensor, totalIndicesNumber);

        if constexpr (isScale) {
            CopyScaleFreqFromWs(totalIndicesNumber);
            GatherScaleFreq(totalIndicesNumber);
            CalcFreqBetweenBlock(noDupCountTensor, arNum, totalIndicesNumber);
        }

        elewiseAligned_ = ops::CeilAlign(elewiseNormal_, copyGradAlign_);
        int64_t colOffset = blockIdx_ * tiling_.elewiseDimLastNormalLoopNum * tiling_.elewiseDimLastPerUbProcessNum;
        int64_t mainStride = DOUBLE * CACHE_LINE + gradLenInWs_ - elewiseNormal_ * sizeof(CAST_T);
        uint32_t gradColAlign = ops::CeilAlign(elewiseNormal_, static_cast<uint32_t>(ONE_BLK_SIZE / sizeof(CAST_T)));

        for (uint32_t i = 0; i < elewiseLoopNums_ - 1; i++) {
            CopyGradFromWs(totalIndicesNumber, elewiseNormal_, mainStride, colOffset);
            AddGradBetweenBlock(elewiseNormal_, gradColAlign, noDupCountTensor, arNum, totalIndicesNumber);
            this->template CopyOutGradWeight<false, false>(indicesTensor, noDupCountTensor, arNum, elewiseNormal_, colOffset, 0);
            colOffset = colOffset + elewiseNormal_;
        }

        elewiseAligned_ = ops::CeilAlign(elewiseTail_, copyGradAlign_);
        gradColAlign = ops::CeilAlign(elewiseTail_, static_cast<uint32_t>(ONE_BLK_SIZE / sizeof(CAST_T)));
        int64_t tailStride = DOUBLE * CACHE_LINE + gradLenInWs_ - elewiseTail_ * sizeof(CAST_T);
        CopyGradFromWs(totalIndicesNumber, elewiseTail_, tailStride, colOffset);
        AddGradBetweenBlock(elewiseTail_, gradColAlign, noDupCountTensor, arNum, totalIndicesNumber);
        this->template CopyOutGradWeight<false, false>(indicesTensor, noDupCountTensor, arNum, elewiseTail_, colOffset, 0);
    }
}

}
#endif