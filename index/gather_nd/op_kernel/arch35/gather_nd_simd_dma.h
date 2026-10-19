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
 * \file gather_nd_simd_dma.h
 * \brief
 */
#ifndef ASCENDC_GATHER_ND_SIMD_DMA_H_
#define ASCENDC_GATHER_ND_SIMD_DMA_H_

#include "kernel_operator.h"
#include "../inc/platform.h"
#include "op_kernel/math_util.h"

namespace GatherNd {
using namespace AscendC;
constexpr int32_t HELP_BUFFER_RANK_SIZE = 64;
constexpr int64_t ONE_BLOCK_SIZE = platform::GetUbBlockSize();
constexpr Reg::CastTrait castTraitB322B64 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING,
                                             RoundMode::UNKNOWN};

template <typename T, bool CAST>
struct ColOffsetTraits {
    using type = typename std::conditional<CAST, int64_t, T>::type;
};

template <typename INDICES_T, bool isCastIndices, bool NIS>
class GatherNdSimdDma {
public:
    __aicore__ inline GatherNdSimdDma(){};

    using OFFSET_T = typename ColOffsetTraits<INDICES_T, isCastIndices>::type;
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR indices, GM_ADDR y, const GatherNdGatherDmaTilingData* tilingData,
                                TPipe* pipe);
    __aicore__ inline void Process();
    __aicore__ inline void CopyInIndices(LocalTensor<INDICES_T>& indicesLocal, int32_t burstLen, int32_t coreOffset);
    __aicore__ inline void IndicesColBuf();
    __aicore__ inline void CopyOut(int64_t offset, uint32_t nBurst, uint32_t copyLen);
    __aicore__ inline void GatherDmaProcess();
    __aicore__ inline void GatherDMA(LocalTensor<int8_t> dstLocal, LocalTensor<OFFSET_T> offsetLocal,
                                     uint16_t offsetNum, uint32_t burstLen, uint32_t dstStride);
    static __simd_vf__ inline void DealIndicesBoundVf(__ubuf__ INDICES_T* indicesLocalPtr,
                                                      __ubuf__ OFFSET_T* outOfstLocalPtr, __ubuf__ OFFSET_T* xShapeAddr,
                                                      __ubuf__ OFFSET_T* strideShapeAddr, uint16_t loopCnt,
                                                      uint16_t rankSizeLoops, OFFSET_T rankSize, uint32_t vfLen,
                                                      uint32_t dataLen, OFFSET_T aSize);
    using IndexRegType = typename std::conditional<
        IsSameType<INDICES_T, int64_t>::value, typename AscendC::Reg::RegTensor<uint64_t, AscendC::Reg::RegTraitNumTwo>,
        typename AscendC::Reg::RegTensor<uint32_t>>::type;
    using InnerRegType = typename std::conditional<
        IsSameType<OFFSET_T, int64_t>::value, typename AscendC::Reg::RegTensor<int64_t, AscendC::Reg::RegTraitNumTwo>,
        typename AscendC::Reg::RegTensor<int32_t>>::type;

private:
    GlobalTensor<int8_t> xGm_;
    GlobalTensor<INDICES_T> indicesGm_;
    GlobalTensor<int8_t> yGm_;
    TPipe* pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM_SIMD> indexQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM_SIMD> offsetQueue_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM_SIMD> inQueue_;
    TBuf<QuePosition::VECCALC> helpTBuf_;
    const GatherNdGatherDmaTilingData* tilingData_;

    int32_t needCoreNum_;
    int32_t rank_;
    int64_t aSizeAligned;
    int64_t yOffsetBase_ = 0;

    int64_t currentCoreRows;
    int64_t ubFactor;
    int32_t blockIdx_;
};

template <typename INDICES_T, bool isCastIndices, bool NIS>
__aicore__ inline void GatherNdSimdDma<INDICES_T, isCastIndices, NIS>::Init(
    GM_ADDR x, GM_ADDR indices, GM_ADDR y, const GatherNdGatherDmaTilingData* tilingData, TPipe* pipe)
{
    pipe_ = pipe;
    tilingData_ = tilingData;
    blockIdx_ = static_cast<int32_t>(GetBlockIdx());
    needCoreNum_ = static_cast<int32_t>(tilingData_->needCoreNum);
    aSizeAligned = tilingData_->aSizeAligned;
    rank_ = tilingData_->rank;

    xGm_.SetGlobalBuffer((__gm__ int8_t*)x);

    if (GetBlockIdx() < tilingData_->preBlockFactor) {
        yOffsetBase_ = (tilingData_->blockFactor + 1) * GetBlockIdx();
        currentCoreRows = tilingData_->blockFactor + 1;
    } else {
        yOffsetBase_ = tilingData_->blockFactor * GetBlockIdx() + tilingData_->preBlockFactor;
        currentCoreRows = tilingData_->blockFactor;
    }
    ubFactor = tilingData_->ubFactor;

    indicesGm_.SetGlobalBuffer((__gm__ INDICES_T*)indices);
    yGm_.SetGlobalBuffer((__gm__ int8_t*)y);
    pipe_->InitBuffer(helpTBuf_, HELP_BUFFER_SIZE_VG);
    pipe_->InitBuffer(inQueue_, BUFFER_NUM_SIMD, ubFactor * aSizeAligned);
    pipe_->InitBuffer(indexQueue_, BUFFER_NUM_SIMD,
                      ops::CeilAlign(static_cast<int64_t>(ubFactor * rank_ * sizeof(INDICES_T)), ONE_BLOCK_SIZE));
    pipe_->InitBuffer(offsetQueue_, BUFFER_NUM_SIMD,
                      ops::CeilAlign(static_cast<int64_t>(ubFactor * sizeof(OFFSET_T)), ONE_BLOCK_SIZE));
}

template <typename INDICES_T, bool isCastIndices, bool NIS>
__aicore__ inline void GatherNdSimdDma<INDICES_T, isCastIndices, NIS>::CopyOut(int64_t offset, uint32_t nBurst,
                                                                               uint32_t copyLen)
{
    DataCopyExtParams dataCopyExtParams;
    dataCopyExtParams.blockCount = nBurst;
    dataCopyExtParams.blockLen = copyLen;
    dataCopyExtParams.srcStride = 0;
    dataCopyExtParams.dstStride = 0;
    LocalTensor<int8_t> yLocal = inQueue_.DeQue<int8_t>();
    DataCopyPad(yGm_[offset], yLocal, dataCopyExtParams);
    inQueue_.FreeTensor(yLocal);
}

template <typename INDICES_T, bool isCastIndices, bool NIS>
__aicore__ inline void GatherNdSimdDma<INDICES_T, isCastIndices, NIS>::CopyInIndices(
    LocalTensor<INDICES_T>& indicesLocal, int32_t burstLen, int32_t coreOffset)
{
    DataCopyPadExtParams<INDICES_T> dataCopyPadExtParams;
    dataCopyPadExtParams.isPad = false;
    dataCopyPadExtParams.leftPadding = 0;
    dataCopyPadExtParams.rightPadding = 0;
    dataCopyPadExtParams.paddingValue = 0;

    DataCopyExtParams dataCopyExtParams;
    dataCopyExtParams.blockCount = 1;
    dataCopyExtParams.blockLen = static_cast<uint32_t>(burstLen * rank_ * sizeof(INDICES_T));
    dataCopyExtParams.srcStride = 0;
    dataCopyExtParams.dstStride = 0;
    DataCopyPad(indicesLocal, indicesGm_[coreOffset], dataCopyExtParams, dataCopyPadExtParams);
}

template <typename INDICES_T, bool isCastIndices, bool NIS>
__simd_vf__ inline void GatherNdSimdDma<INDICES_T, isCastIndices, NIS>::DealIndicesBoundVf(
    __ubuf__ INDICES_T* indicesLocalPtr, __ubuf__ OFFSET_T* outOfstLocalPtr, __ubuf__ OFFSET_T* xShapeAddr,
    __ubuf__ OFFSET_T* strideShapeAddr, uint16_t loopCnt, uint16_t rankSizeLoops, OFFSET_T rankSize, uint32_t vfLen,
    uint32_t dataLen, OFFSET_T aSize)
{
    InnerRegType inReg;
    InnerRegType outReg;
    InnerRegType orderReg;
    InnerRegType zeroConstReg;
    InnerRegType upLimitConstReg;
    InnerRegType tmpReg;
    IndexRegType indexReg;
    AscendC::Reg::MaskReg tmpMask;
    AscendC::Reg::MaskReg pregLoop;
    AscendC::Reg::MaskReg overstepMask;
    AscendC::Reg::Duplicate(zeroConstReg, OFFSET_T(0));

    for (uint16_t i = 0; i < loopCnt; i++) {
        if constexpr (IsSameType<OFFSET_T, int64_t>::value) {
            pregLoop = AscendC::Reg::UpdateMask<OFFSET_T, AscendC::Reg::RegTraitNumTwo>(dataLen);
            overstepMask = AscendC::Reg::CreateMask<OFFSET_T, AscendC::Reg::MaskPattern::ALLF,
                                                    AscendC::Reg::RegTraitNumTwo>();
        } else {
            pregLoop = AscendC::Reg::UpdateMask<OFFSET_T>(dataLen);
            overstepMask = AscendC::Reg::CreateMask<OFFSET_T, AscendC::Reg::MaskPattern::ALLF>();
        }
        AscendC::Reg::Duplicate(outReg, 0, pregLoop);
        AscendC::Reg::Arange(orderReg, i * vfLen);
        AscendC::Reg::Muls(orderReg, orderReg, rankSize, pregLoop);
        for (uint16_t dim = 0; dim < rankSizeLoops; dim++) {
            indexReg = (IndexRegType&)orderReg;
            if constexpr (IsSameType<INDICES_T, int32_t>::value && IsSameType<OFFSET_T, int64_t>::value) {
                AscendC::Reg::RegTensor<int32_t> castReg;
                AscendC::Reg::Gather(castReg, indicesLocalPtr, indexReg, pregLoop);
                Reg::Cast<int64_t, int32_t, castTraitB322B64>(inReg, castReg, pregLoop);
            } else {
                AscendC::Reg::Gather(inReg, indicesLocalPtr, indexReg, pregLoop);
            }

            OFFSET_T xShapeDimkSize = xShapeAddr[dim];
            if constexpr (NIS) {
                AscendC::Reg::Compare<OFFSET_T, CMPMODE::LT>(tmpMask, inReg, zeroConstReg, pregLoop);
                AscendC::Reg::Adds(tmpReg, inReg, xShapeDimkSize, tmpMask); // 补偿负索引场景
                Move<OFFSET_T, AscendC::Reg::MaskMergeMode::MERGING>(inReg, tmpReg, tmpMask);
            }

            AscendC::Reg::Compare<OFFSET_T, CMPMODE::LT>(tmpMask, inReg, zeroConstReg, pregLoop); // 负越界
            AscendC::Reg::Or(overstepMask, overstepMask, tmpMask, pregLoop);

            AscendC::Reg::Duplicate(upLimitConstReg, OFFSET_T(xShapeDimkSize));
            AscendC::Reg::Compare<OFFSET_T, CMPMODE::GE>(tmpMask, inReg, upLimitConstReg, pregLoop); // 正越界
            AscendC::Reg::Or(overstepMask, overstepMask, tmpMask, pregLoop);

            OFFSET_T strideValue = strideShapeAddr[dim];
            AscendC::Reg::Muls(inReg, inReg, strideValue, pregLoop);
            AscendC::Reg::Add(outReg, inReg, outReg, pregLoop);
            AscendC::Reg::Adds(orderReg, orderReg, (OFFSET_T)(1), pregLoop);
        }

        AscendC::Reg::Muls(outReg, outReg, aSize, pregLoop);
        AscendC::Reg::Duplicate(tmpReg, OFFSET_T(-1), overstepMask); // 补偿越界
        Move<OFFSET_T, AscendC::Reg::MaskMergeMode::MERGING>(outReg, tmpReg, overstepMask);
        auto outOfstAddr = outOfstLocalPtr + i * vfLen;
        AscendC::Reg::StoreAlign(outOfstAddr, outReg, pregLoop);
    }
}

template <typename INDICES_T, bool isCastIndices, bool NIS>
__aicore__ inline void GatherNdSimdDma<INDICES_T, isCastIndices, NIS>::GatherDMA(LocalTensor<int8_t> dstLocal,
                                                                                 LocalTensor<OFFSET_T> offsetLocal,
                                                                                 uint16_t offsetNum, uint32_t burstLen,
                                                                                 uint32_t dstStride)
{
    GatherParams gatherDataCopyParams;
    gatherDataCopyParams.offsetNum = offsetNum;
    gatherDataCopyParams.burstLen = burstLen;
    gatherDataCopyParams.dstStride = dstStride;

    GatherPadParams<int8_t> gatherPadParams{
        GatherPaddingMode::CONSTANT_PADDING_MODE,
        static_cast<int8_t>(0),
    };

    GatherDataCopy(dstLocal, xGm_, offsetLocal, gatherDataCopyParams, gatherPadParams);
    inQueue_.EnQue<int8_t>(dstLocal);
}

template <typename INDICES_T, bool isCastIndices, bool NIS>
__aicore__ inline void GatherNdSimdDma<INDICES_T, isCastIndices, NIS>::GatherDmaProcess()
{
    int64_t loopNum = (currentCoreRows + ubFactor - 1) / ubFactor;
    uint32_t vfLen = platform::GetVRegSize() / sizeof(int32_t);

    // 获取stride和shape
    AscendC::LocalTensor<OFFSET_T> helpTensor = helpTBuf_.Get<OFFSET_T>();
    auto xShapeTensor = helpTensor[0];
    for (uint32_t i = 0; i < MAX_SUPPORT_DIM; i++) {
        xShapeTensor.SetValue(i, (OFFSET_T)(tilingData_->xShape[i]));
    }
    auto strideShapeTensor = helpTensor[HELP_BUFFER_RANK_SIZE / sizeof(OFFSET_T)];
    for (uint32_t i = 0; i < MAX_SUPPORT_DIM; i++) {
        strideShapeTensor.SetValue(i, (OFFSET_T)(tilingData_->strideShape[i]));
    }
    __ubuf__ OFFSET_T* xShapeAddr = (__ubuf__ OFFSET_T*)xShapeTensor.GetPhyAddr();
    __ubuf__ OFFSET_T* strideShapeAddr = (__ubuf__ OFFSET_T*)strideShapeTensor.GetPhyAddr();

    event_t eventIdStoV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdStoV);
    WaitFlag<HardEvent::S_V>(eventIdStoV);

    for (int64_t i = 0; i < loopNum; i++) {
        int64_t rows = (i == loopNum - 1) ? (currentCoreRows - i * ubFactor) : ubFactor;

        int32_t indicesNumOffset = (yOffsetBase_ + i * ubFactor) * rank_;
        LocalTensor<INDICES_T> indicesLocal = indexQueue_.AllocTensor<INDICES_T>();
        LocalTensor<OFFSET_T> offsetLocal = offsetQueue_.AllocTensor<OFFSET_T>();
        CopyInIndices(indicesLocal, rows, indicesNumOffset);
        indexQueue_.EnQue(indicesLocal);

        indicesLocal = indexQueue_.DeQue<INDICES_T>();
        __ubuf__ INDICES_T* indicesLocalPtr = ((__ubuf__ INDICES_T*)indicesLocal.GetPhyAddr());
        __ubuf__ OFFSET_T* outOfstLocalPtr = (__ubuf__ OFFSET_T*)offsetLocal.GetPhyAddr();

        uint32_t dataLen = static_cast<uint32_t>(rows);
        uint32_t indicesLenTimes = Ops::Base::CeilDiv(dataLen, vfLen);
        uint16_t loopCnt = static_cast<uint16_t>(indicesLenTimes);
        DealIndicesBoundVf(indicesLocalPtr, outOfstLocalPtr, xShapeAddr, strideShapeAddr, loopCnt,
                           static_cast<uint16_t>(rank_), static_cast<OFFSET_T>(rank_), vfLen, dataLen,
                           static_cast<OFFSET_T>(tilingData_->aSize));
        indexQueue_.FreeTensor(indicesLocal);

        LocalTensor<int8_t> xLocal = inQueue_.AllocTensor<int8_t>();
        Duplicate<int8_t>(xLocal, int8_t(0), rows * aSizeAligned);
        event_t eventIdVtoMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventIdVtoMTE2);
        WaitFlag<HardEvent::V_MTE2>(eventIdVtoMTE2);

        GatherDMA(xLocal, offsetLocal, static_cast<uint16_t>(rows), tilingData_->aSize, aSizeAligned);
        offsetQueue_.FreeTensor(offsetLocal);

        int64_t yOffset = (yOffsetBase_ + i * ubFactor) * tilingData_->aSize;
        CopyOut(yOffset, rows, tilingData_->aSize);
    }
}

template <typename INDICES_T, bool isCastIndices, bool NIS>
__aicore__ inline void GatherNdSimdDma<INDICES_T, isCastIndices, NIS>::Process()
{
    if (blockIdx_ >= tilingData_->needCoreNum) {
        return;
    }
    GatherDmaProcess();
}
} // namespace GatherNd
#endif // ASCENDC_GATHER_ND_SIMD_DMA_H_
