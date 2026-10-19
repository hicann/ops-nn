/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file gather_v2_simd_two_dim_gdma.h
 * \brief
 */
#ifndef GATHER_V2_SIMD_TWO_DIM_GDMA_H
#define GATHER_V2_SIMD_TWO_DIM_GDMA_H

#ifndef K_MAX_SHAPE_DIM
#define K_MAX_SHAPE_DIM 0
#endif

#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_vec_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "op_kernel/platform_util.h"

namespace gatherv2 {
using namespace AscendC;

constexpr int32_t BUFFER_NUM_SIMD_TWO_DIM_DMA = 2;
constexpr int32_t UB_AGLIN_VALUE = 32;

constexpr static AscendC::Reg::CastTrait castTraitB32B64 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::UNKNOWN,
    AscendC::Reg::MaskMergeMode::ZEROING,
};

template <typename INDICES_T, bool isCastIndices, bool NIS>
class SimdTwoDimGatherDMA {
public:
    __aicore__ inline SimdTwoDimGatherDMA(TPipe* pipe) : pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR indices, GM_ADDR y,
                                const GatherV2TilingDataSimdTwoDimDma* tilingData);
    __aicore__ inline void Process();
    template <typename T>
    __aicore__ inline void CopyIn(LocalTensor<T> xLocal, GlobalTensor<T> xGm, int64_t offset, uint32_t nBurst,
                                  uint32_t copyLen);
    __aicore__ inline void GenIndices(int64_t idx, int64_t endIdx, int64_t rows);
    __aicore__ inline void CopyOut(int64_t offset, uint32_t nBurst, uint32_t copyLen, uint32_t dstStride);
    __aicore__ inline void ProcessGatherDMA();
    template <typename OFFSET_T>
    __aicore__ inline void GatherDMA(LocalTensor<OFFSET_T> offsetLocal, LocalTensor<int8_t> dstLocal,
                                     uint16_t offsetNum, uint32_t burstLen, uint32_t dstStride);

private:
    GlobalTensor<int8_t> xGm_;
    GlobalTensor<INDICES_T> indicesGm_;
    GlobalTensor<int8_t> yGm_;
    TPipe* pipe_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM_SIMD_TWO_DIM_DMA> inQueue_;
    TBuf<QuePosition::VECCALC> indexBuf_;
    TBuf<QuePosition::VECCALC> indexUbBuf_;
    TBuf<QuePosition::VECCALC> indicesCastBuf_;
    const GatherV2TilingDataSimdTwoDimDma* tilingData_;

    int64_t indicesOffsetBase_ = 0;
    int64_t curIndexSize_ = 0;

    int32_t indicesCoreIdx_ = 0;
    int32_t innerCoreIdx_ = 0;
    int64_t curInnerSize_ = 0;
};

template <typename INDICES_T, bool isCastIndices, bool NIS>
__aicore__ inline void SimdTwoDimGatherDMA<INDICES_T, isCastIndices, NIS>::Init(
    GM_ADDR x, GM_ADDR indices, GM_ADDR y, const GatherV2TilingDataSimdTwoDimDma* tilingData)
{
    tilingData_ = tilingData;

    indicesCoreIdx_ = GetBlockIdx() / tilingData_->colCoreNum;
    innerCoreIdx_ = GetBlockIdx() % tilingData_->colCoreNum;
    curInnerSize_ = (innerCoreIdx_ == tilingData_->colCoreNum - 1) ? tilingData_->tailColCoreInners :
                                                                     tilingData_->normalColCoreInners;

    xGm_.SetGlobalBuffer((__gm__ int8_t*)x + innerCoreIdx_ * tilingData_->normalColCoreInners);

    indicesGm_.SetGlobalBuffer((__gm__ INDICES_T*)indices);
    yGm_.SetGlobalBuffer((__gm__ int8_t*)y + innerCoreIdx_ * tilingData_->normalColCoreInners);

    pipe_->InitBuffer(inQueue_, BUFFER_NUM_SIMD_TWO_DIM_DMA, tilingData_->colUbFactor * tilingData_->indicesUbFactor);
    pipe_->InitBuffer(indexBuf_, tilingData_->indiceFactor * sizeof(INDICES_T));
    pipe_->InitBuffer(indexUbBuf_, (tilingData_->indicesUbFactor * sizeof(INDICES_T) + UB_AGLIN_VALUE - 1) /
                                       UB_AGLIN_VALUE * UB_AGLIN_VALUE);
    if constexpr (isCastIndices) {
        pipe_->InitBuffer(indicesCastBuf_, (tilingData_->indicesUbFactor * sizeof(int64_t) + UB_AGLIN_VALUE - 1) /
                                               UB_AGLIN_VALUE * UB_AGLIN_VALUE);
    }
}

template <typename INDICES_T, bool isCastIndices, bool NIS>
template <typename T>
__aicore__ inline void SimdTwoDimGatherDMA<INDICES_T, isCastIndices, NIS>::CopyIn(LocalTensor<T> xLocal,
                                                                                  GlobalTensor<T> xGm, int64_t offset,
                                                                                  uint32_t nBurst, uint32_t copyLen)
{
    DataCopyPadExtParams<T> dataCopyPadExtParams;
    dataCopyPadExtParams.isPad = false;
    dataCopyPadExtParams.leftPadding = 0;
    dataCopyPadExtParams.rightPadding = 0;
    dataCopyPadExtParams.paddingValue = 0;

    DataCopyExtParams dataCoptExtParams;
    dataCoptExtParams.blockCount = nBurst;
    dataCoptExtParams.blockLen = copyLen * sizeof(T);
    dataCoptExtParams.srcStride = 0;
    dataCoptExtParams.dstStride = 0;
    DataCopyPad(xLocal, xGm[offset], dataCoptExtParams, dataCopyPadExtParams);
}

template <typename INDICES_T, bool isCastIndices, bool NIS>
__simd_vf__ inline void GenIndicesVF(__ubuf__ INDICES_T* indicesAddr, __ubuf__ int64_t* indicesCastAddr, uint32_t vfLen,
                                     uint16_t loopCnt, uint32_t size, uint32_t sizeTwo, int64_t gatherDims,
                                     int64_t totalColByte, INDICES_T zero)
{
    Reg::RegTensor<INDICES_T> indicesReg;
    Reg::RegTensor<INDICES_T> invalidIndicesReg;
    Reg::RegTensor<INDICES_T> tmpReg;
    Reg::RegTensor<int64_t, Reg::RegTraitNumTwo> indicesCastReg;
    Reg::MaskReg maskReg;
    Reg::MaskReg maskRegTwo;

    Reg::MaskReg negMask;
    Reg::MaskReg cmpMask;

    Reg::Duplicate(invalidIndicesReg, static_cast<INDICES_T>(-1));
    for (uint16_t i = 0; i < loopCnt; i++) {
        maskReg = Reg::UpdateMask<INDICES_T>(size);
        Reg::LoadAlign<INDICES_T, Reg::LoadDist::DIST_NORM>(indicesReg, indicesAddr + i * vfLen);
        if constexpr (NIS) {
            Reg::Compares<INDICES_T, CMPMODE::LT>(negMask, indicesReg, zero, maskReg);
            Reg::Adds(tmpReg, indicesReg, static_cast<INDICES_T>(gatherDims), negMask); // 补偿负索引场景
            Reg::Move<INDICES_T, AscendC::Reg::MaskMergeMode::MERGING>(indicesReg, tmpReg, negMask);
        }
        Reg::Compares<INDICES_T, CMPMODE::LT>(cmpMask, indicesReg, gatherDims, maskReg);
        Reg::Select(indicesReg, indicesReg, invalidIndicesReg, cmpMask);
        if constexpr (isCastIndices && IsSameType<INDICES_T, int32_t>::value) {
            maskRegTwo = Reg::UpdateMask<int64_t, Reg::RegTraitNumTwo>(sizeTwo);
            Reg::Cast<int64_t, int32_t, castTraitB32B64>(indicesCastReg, indicesReg, maskRegTwo);
            Reg::Muls(indicesCastReg, indicesCastReg, totalColByte, maskRegTwo);
            Reg::StoreAlign(indicesCastAddr + i * vfLen, indicesCastReg, maskRegTwo);
        } else {
            Reg::Muls(indicesReg, indicesReg, static_cast<INDICES_T>(totalColByte), maskReg);
            Reg::StoreAlign(indicesAddr + i * vfLen, indicesReg, maskReg);
        }
    }
}

template <typename INDICES_T, bool isCastIndices, bool NIS>
__aicore__ inline void SimdTwoDimGatherDMA<INDICES_T, isCastIndices, NIS>::GenIndices(int64_t idx, int64_t endIdx,
                                                                                      int64_t rows)
{
    if (idx + rows - 1 >= indicesOffsetBase_ + curIndexSize_) { // idx + rows > indicesOffsetBase_ + curIndexSize_

        int64_t copyLen = (endIdx - idx) > tilingData_->indiceFactor ? tilingData_->indiceFactor : endIdx - idx;
        indicesOffsetBase_ = idx;
        curIndexSize_ = copyLen;
        LocalTensor<INDICES_T> tmpLocal = indexBuf_.Get<INDICES_T>();
        CopyIn(tmpLocal, indicesGm_, indicesOffsetBase_, 1, copyLen);

        event_t eventIdMTE2toV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMTE2toV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMTE2toV);
    }
    LocalTensor<INDICES_T> indexLocal = indexBuf_.Get<INDICES_T>();
    LocalTensor<INDICES_T> indexOffsetLocal = indexUbBuf_.Get<INDICES_T>();
    Copy(indexOffsetLocal, indexLocal[idx - indicesOffsetBase_], rows);

    __ubuf__ INDICES_T* indicesAddr = (__ubuf__ INDICES_T*)indexOffsetLocal.GetPhyAddr();
    __ubuf__ int64_t* indicesCastAddr = nullptr;
    if constexpr (isCastIndices) {
        LocalTensor<int64_t> indicesCastLocal = indicesCastBuf_.Get<int64_t>();
        indicesCastAddr = (__ubuf__ int64_t*)indicesCastLocal.GetPhyAddr();
    }
    uint32_t vfLen = AscendC::VECTOR_REG_WIDTH / sizeof(INDICES_T);
    uint16_t loopCnt = (rows + vfLen - 1) / vfLen;
    uint32_t size = rows;
    uint32_t sizeTwo = rows;
    int64_t gatherDims = tilingData_->gatherDimSize;
    int64_t totalColByte = tilingData_->innerByte;
    INDICES_T zero = 0;

    GenIndicesVF<INDICES_T, isCastIndices, NIS>(indicesAddr, indicesCastAddr, vfLen, loopCnt, size, sizeTwo, gatherDims,
                                                totalColByte, zero);
}

template <typename INDICES_T, bool isCastIndices, bool NIS>
template <typename OFFSET_T>
__aicore__ inline void SimdTwoDimGatherDMA<INDICES_T, isCastIndices, NIS>::GatherDMA(LocalTensor<OFFSET_T> offsetLocal,
                                                                                     LocalTensor<int8_t> dstLocal,
                                                                                     uint16_t offsetNum,
                                                                                     uint32_t burstLen,
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

    event_t eventIdMTE2_V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMTE2_V);
    WaitFlag<HardEvent::MTE2_V>(eventIdMTE2_V);
}

template <typename INDICES_T, bool isCastIndices, bool NIS>
__aicore__ inline void SimdTwoDimGatherDMA<INDICES_T, isCastIndices, NIS>::CopyOut(int64_t offset, uint32_t nBurst,
                                                                                   uint32_t copyLen, uint32_t dstStride)
{
    DataCopyExtParams dataCoptExtParams;
    dataCoptExtParams.blockCount = nBurst;
    dataCoptExtParams.blockLen = copyLen;
    dataCoptExtParams.srcStride = 0;
    dataCoptExtParams.dstStride = dstStride;
    LocalTensor<int8_t> yLocal = inQueue_.DeQue<int8_t>();

    DataCopyPad(yGm_[offset], yLocal, dataCoptExtParams);
    inQueue_.FreeTensor(yLocal);
}

template <typename INDICES_T, bool isCastIndices, bool NIS>
__aicore__ inline void SimdTwoDimGatherDMA<INDICES_T, isCastIndices, NIS>::ProcessGatherDMA()
{
    // indice -> g
    // output -> g, i
    int64_t yStart = 0;
    int64_t currentCoreElements;
    if (indicesCoreIdx_ < tilingData_->tailBlockFactor) {
        yStart = (tilingData_->blockFactor + 1) * indicesCoreIdx_;
        currentCoreElements = tilingData_->blockFactor + 1;
    } else {
        yStart = tilingData_->blockFactor * indicesCoreIdx_ + tilingData_->tailBlockFactor;
        currentCoreElements = tilingData_->blockFactor;
    }
    int64_t yLoopNum = (currentCoreElements + tilingData_->indicesUbFactor - 1) / tilingData_->indicesUbFactor;
    int64_t indiceEndIdx = yStart + currentCoreElements;

    for (int64_t i = 0; i < yLoopNum; i++) {
        int64_t rows = (i == yLoopNum - 1) ? (currentCoreElements - i * tilingData_->indicesUbFactor) :
                                             tilingData_->indicesUbFactor;
        int64_t globalRowOffset = yStart + i * tilingData_->indicesUbFactor;
        GenIndices(globalRowOffset, indiceEndIdx, rows);
        uint32_t cols = curInnerSize_;
        uint32_t colsAlign = (cols + UB_AGLIN_VALUE - 1) / UB_AGLIN_VALUE * UB_AGLIN_VALUE;
        int64_t xLocalSize = colsAlign * rows;
        LocalTensor<int8_t> xLocal = inQueue_.AllocTensor<int8_t>();
        Duplicate<int8_t>(xLocal, 0, xLocalSize);

        event_t eventIdV_MTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventIdV_MTE2);
        WaitFlag<HardEvent::V_MTE2>(eventIdV_MTE2);

        if constexpr (isCastIndices) {
            LocalTensor<int64_t> indicesCastLocal = indicesCastBuf_.Get<int64_t>();
            GatherDMA<int64_t>(indicesCastLocal, xLocal, static_cast<uint16_t>(rows), cols, colsAlign);
        } else {
            LocalTensor<INDICES_T> indexOffsetLocal = indexUbBuf_.Get<INDICES_T>();
            GatherDMA<INDICES_T>(indexOffsetLocal, xLocal, static_cast<uint16_t>(rows), cols, colsAlign);
        }
        int64_t yGmOffset = globalRowOffset * tilingData_->innerByte;

        event_t eventIdMTE2_MTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
        SetFlag<HardEvent::MTE2_MTE3>(eventIdMTE2_MTE3);
        WaitFlag<HardEvent::MTE2_MTE3>(eventIdMTE2_MTE3);

        CopyOut(yGmOffset, rows, cols, tilingData_->innerByte - cols);
    }
}

template <typename INDICES_T, bool isCastIndices, bool NIS>
__aicore__ inline void SimdTwoDimGatherDMA<INDICES_T, isCastIndices, NIS>::Process()
{
    if (static_cast<int32_t>(GetBlockIdx()) >= tilingData_->needCoreNum) {
        return;
    }
    ProcessGatherDMA();
}

} // namespace gatherv2
#endif // GATHER_V2_SIMD_TWO_DIM_GDMA_H
