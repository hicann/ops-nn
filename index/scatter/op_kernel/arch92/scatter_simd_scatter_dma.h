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
 * \file scatter_simd_scatter_dma.h
 * \brief
 */

#ifndef ASCENDC_SCATTER_SIMD_SCATTER_DMA_H_
#define ASCENDC_SCATTER_SIMD_SCATTER_DMA_H_

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"

namespace SCATTER {
using namespace AscendC;

using OffsetRegType = Reg::RegTensor<int64_t, Reg::RegTraitNumTwo>;

constexpr Reg::CastTrait castTraitB322B64 = {Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING,
                                             RoundMode::UNKNOWN};

template <typename IDX_T>
__simd_callee__ inline void GatherIndexToI64(OffsetRegType& dstReg, Reg::RegTensor<int32_t>& castReg32,
                                             __ubuf__ IDX_T* srcPtr, OffsetRegType& idxReg, Reg::MaskReg& pregLoop)
{
    Reg::RegTensor<uint32_t> indexReg = (AscendC::Reg::RegTensor<uint32_t>&)idxReg.reg[0];
    if constexpr (IsSameType<IDX_T, int32_t>::value) {
        Reg::Gather(castReg32, srcPtr, indexReg, pregLoop);
        Reg::Cast<int64_t, int32_t, castTraitB322B64>(dstReg, castReg32, pregLoop);
    } else {
        Reg::Gather(dstReg, srcPtr, indexReg, pregLoop);
    }
}

__simd_callee__ inline void ComputeFinalOffset(OffsetRegType& offsetReg, OffsetRegType& tmpReg,
                                               OffsetRegType& batchDstReg, OffsetRegType& outerIReg,
                                               OffsetRegType& scatterDstReg, __ubuf__ int64_t* offsetAddr,
                                               int64_t inputDim1, int64_t inputDim2, int64_t stride,
                                               Reg::MaskReg& pregLoop)
{
    // offset = ((batchDst * inputDim1 + outerI) * inputDim2 + scatterDst) * stride
    Reg::Muls(tmpReg, batchDstReg, inputDim1, pregLoop);
    Reg::Add(tmpReg, tmpReg, outerIReg, pregLoop);
    Reg::Muls(tmpReg, tmpReg, inputDim2, pregLoop);
    Reg::Add(tmpReg, tmpReg, scatterDstReg, pregLoop);
    Reg::Muls(offsetReg, tmpReg, stride, pregLoop);
    Reg::StoreAlign(offsetAddr, offsetReg, pregLoop);
}

template <typename IDX_T, const uint16_t IDX_DIM>
static __simd_vf__ inline void ComputeOffsetForMode0VecImpl(__ubuf__ IDX_T* idxLocalPtr,
                                                            __ubuf__ int64_t* offsetLocalPtr, int64_t rowsStart,
                                                            int64_t batchStart, uint32_t rows, uint32_t vfLen,
                                                            uint16_t loopCnt, int64_t updatesDim1, int64_t inputDim1,
                                                            int64_t inputDim2, int64_t stride)
{
    OffsetRegType srcIdxReg, batchIReg, localBatchReg, outerIReg, batchDstReg, scatterDstReg;
    OffsetRegType tmpReg1, tmpReg2, offsetReg, gatheredIdxReg;
    Reg::RegTensor<int32_t> castReg32;
    Reg::MaskReg pregLoop;

    for (uint16_t i = 0; i < loopCnt; i++) {
        uint32_t dataLen = static_cast<uint32_t>(rows - i * vfLen);
        pregLoop = Reg::UpdateMask<int64_t, Reg::RegTraitNumTwo>(dataLen);

        Reg::Arange(srcIdxReg, static_cast<int64_t>(rowsStart + i * vfLen));
        Reg::Duplicate(tmpReg1, updatesDim1, pregLoop);
        Reg::Div(batchIReg, srcIdxReg, tmpReg1, pregLoop);
        Reg::Muls(tmpReg1, batchIReg, updatesDim1, pregLoop);
        Reg::Sub(outerIReg, srcIdxReg, tmpReg1, pregLoop);

        Reg::Adds(localBatchReg, batchIReg, static_cast<int64_t>(-batchStart), pregLoop);

        if constexpr (IDX_DIM == DIM_2) {
            Reg::Muls(tmpReg1, localBatchReg, 2, pregLoop);
            GatherIndexToI64<IDX_T>(batchDstReg, castReg32, (__ubuf__ IDX_T*)idxLocalPtr, tmpReg1, pregLoop);
            Reg::Adds(tmpReg2, tmpReg1, static_cast<int64_t>(1), pregLoop);
            GatherIndexToI64<IDX_T>(gatheredIdxReg, castReg32, (__ubuf__ IDX_T*)idxLocalPtr, tmpReg2, pregLoop);
        } else {
            Reg::Adds(batchDstReg, batchIReg, static_cast<int64_t>(0), pregLoop);
            GatherIndexToI64<IDX_T>(gatheredIdxReg, castReg32, (__ubuf__ IDX_T*)idxLocalPtr, localBatchReg, pregLoop);
        }
        Reg::Adds(scatterDstReg, gatheredIdxReg, static_cast<int64_t>(0), pregLoop);

        ComputeFinalOffset(offsetReg, tmpReg1, batchDstReg, outerIReg, scatterDstReg,
                           (__ubuf__ int64_t*)(offsetLocalPtr + i * vfLen), inputDim1, inputDim2, stride, pregLoop);
    }
}

template <typename IDX_T, const uint16_t IDX_DIM>
static __simd_vf__ inline void ComputeOffsetForMode1VecImpl(__ubuf__ IDX_T* idxLocalPtr,
                                                            __ubuf__ int64_t* offsetLocalPtr, int64_t rowsStart,
                                                            int64_t batchStart, uint32_t rows, uint32_t vfLen,
                                                            uint16_t loopCnt, int64_t dim12, int64_t updatesDim2,
                                                            int64_t inputDim1, int64_t inputDim2, int64_t stride)
{
    OffsetRegType srcIdxReg, batchIReg, localBatchReg, outerIReg, scatterDstReg;
    OffsetRegType tmpReg1, tmpReg2, offsetReg, gatheredIdxReg, batchDstReg;
    Reg::RegTensor<int32_t> castReg32;
    Reg::MaskReg pregLoop;

    for (uint16_t i = 0; i < loopCnt; i++) {
        uint32_t dataLen = static_cast<uint32_t>(rows - i * vfLen);
        pregLoop = Reg::UpdateMask<int64_t, Reg::RegTraitNumTwo>(dataLen);

        Reg::Arange(srcIdxReg, static_cast<int64_t>(rowsStart + i * vfLen));
        Reg::Duplicate(tmpReg1, dim12, pregLoop);
        Reg::Div(batchIReg, srcIdxReg, tmpReg1, pregLoop);
        Reg::Muls(tmpReg1, batchIReg, dim12, pregLoop);
        Reg::Sub(tmpReg2, srcIdxReg, tmpReg1, pregLoop);
        Reg::Duplicate(tmpReg1, updatesDim2, pregLoop);
        Reg::Div(outerIReg, tmpReg2, tmpReg1, pregLoop);
        Reg::Muls(tmpReg1, outerIReg, updatesDim2, pregLoop);
        Reg::Sub(scatterDstReg, tmpReg2, tmpReg1, pregLoop);

        Reg::Adds(localBatchReg, batchIReg, static_cast<int64_t>(-batchStart), pregLoop);

        if constexpr (IDX_DIM == DIM_2) {
            Reg::Muls(tmpReg1, localBatchReg, 2, pregLoop);
            GatherIndexToI64<IDX_T>(batchDstReg, castReg32, (__ubuf__ IDX_T*)idxLocalPtr, tmpReg1, pregLoop);
            Reg::Adds(tmpReg2, tmpReg1, static_cast<int64_t>(1), pregLoop);
            GatherIndexToI64<IDX_T>(gatheredIdxReg, castReg32, (__ubuf__ IDX_T*)idxLocalPtr, tmpReg2, pregLoop);
        } else {
            Reg::Adds(batchDstReg, batchIReg, static_cast<int64_t>(0), pregLoop);
            GatherIndexToI64<IDX_T>(gatheredIdxReg, castReg32, (__ubuf__ IDX_T*)idxLocalPtr, localBatchReg, pregLoop);
        }
        Reg::Add(scatterDstReg, scatterDstReg, gatheredIdxReg, pregLoop);

        ComputeFinalOffset(offsetReg, tmpReg1, batchDstReg, outerIReg, scatterDstReg,
                           (__ubuf__ int64_t*)(offsetLocalPtr + i * vfLen), inputDim1, inputDim2, stride, pregLoop);
    }
}

template <typename IDX_T, const uint16_t SIMD_MODE>
class ScatterSimdScatterDMA {
public:
    __aicore__ inline ScatterSimdScatterDMA(const ScatterTilingData* tiling, TPipe* pipe)
        : tilingData_(tiling), pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR indices, GM_ADDR updates, GM_ADDR y);
    __aicore__ inline void Process();
    template <typename T>
    __aicore__ inline void CopyIn(LocalTensor<T> xLocal, GlobalTensor<T> xGm, int64_t offset, uint32_t nBurst,
                                  uint32_t copyLen);
    template <const uint16_t IDX_DIM>
    __aicore__ inline void NoSplitColProcess(int64_t colsAlign);
    template <const uint16_t IDX_DIM>
    __aicore__ inline void NoSplitColProcessForMode0(int64_t colsAlign);
    template <const uint16_t IDX_DIM>
    __aicore__ inline void NoSplitColProcessForMode1(int64_t colsAlign);

private:
    GlobalTensor<IDX_T> indicesGm_;
    GlobalTensor<int8_t> updatesGm_;
    GlobalTensor<int8_t> yGm_;
    TQue<QuePosition::VECIN, DB_BUFFER> indexQue_;
    TQue<QuePosition::VECCALC, DB_BUFFER> offsetQue_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, DB_BUFFER> inQueue_;

    TPipe* pipe_ = nullptr;
    const ScatterTilingData* tilingData_;

    int32_t ubBlockSize_{0};
    int64_t blockIdx_{0};
    int64_t curCoreData_{0};
};

template <typename IDX_T, const uint16_t SIMD_MODE>
__aicore__ inline void ScatterSimdScatterDMA<IDX_T, SIMD_MODE>::Init(GM_ADDR x, GM_ADDR indices, GM_ADDR updates,
                                                                     GM_ADDR y)
{
    yGm_.SetGlobalBuffer((__gm__ int8_t*)(y));
    updatesGm_.SetGlobalBuffer((__gm__ int8_t*)(updates));
    indicesGm_.SetGlobalBuffer((__gm__ IDX_T*)(indices));
    ubBlockSize_ = static_cast<int64_t>(Ops::Base::GetUbBlockSize());
    pipe_->InitBuffer(inQueue_, DB_BUFFER, tilingData_->updatesUbSize);
    pipe_->InitBuffer(offsetQue_, DB_BUFFER, tilingData_->offsetUbSize);
    pipe_->InitBuffer(indexQue_, DB_BUFFER, tilingData_->indicesUbSize);

    blockIdx_ = GetBlockIdx();
    curCoreData_ = blockIdx_ != (tilingData_->aivCoreNum - 1) ? tilingData_->blockFactor : tilingData_->tailBlockData;
}

template <typename IDX_T, const uint16_t SIMD_MODE>
template <typename T>
__aicore__ inline void ScatterSimdScatterDMA<IDX_T, SIMD_MODE>::CopyIn(LocalTensor<T> xLocal, GlobalTensor<T> xGm,
                                                                       int64_t offset, uint32_t nBurst,
                                                                       uint32_t copyLen)
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

template <typename IDX_T, const uint16_t SIMD_MODE>
template <const uint16_t IDX_DIM>
__aicore__ inline void ScatterSimdScatterDMA<IDX_T, SIMD_MODE>::NoSplitColProcess(int64_t colsAlign)
{
    if constexpr (SIMD_MODE == 0) {
        NoSplitColProcessForMode0<IDX_DIM>(colsAlign);
    } else {
        NoSplitColProcessForMode1<IDX_DIM>(colsAlign);
    }
}

template <typename IDX_T, const uint16_t SIMD_MODE>
template <const uint16_t IDX_DIM>
__aicore__ inline void ScatterSimdScatterDMA<IDX_T, SIMD_MODE>::NoSplitColProcessForMode0(int64_t colsAlign)
{
    int64_t maxRowsOnce = tilingData_->maxRowsOnce;
    int64_t loopNum = Ops::Base::CeilDiv(curCoreData_, maxRowsOnce);
    int64_t srcStart = blockIdx_ * tilingData_->blockFactor;
    int64_t cols = tilingData_->updatesDim2 * tilingData_->updatesDim3;
    constexpr uint32_t vfLen = Ops::Base::GetVRegSize() / sizeof(int64_t);
    for (int64_t i = 0; i < loopNum; i++) {
        int64_t rows = (i == loopNum - 1) ? (curCoreData_ - i * maxRowsOnce) : maxRowsOnce;
        int64_t rowsStart = srcStart + i * maxRowsOnce;
        int64_t batchStart = rowsStart / tilingData_->updatesDim1;
        int64_t batchEnd = (rowsStart + rows - 1) / tilingData_->updatesDim1;
        int32_t idxNum = batchEnd - batchStart + 1;
        uint16_t loopCnt = static_cast<uint16_t>(Ops::Base::CeilDiv(rows, static_cast<int64_t>(vfLen)));
        // Load indices → DeQue → SIMD compute offsets → FreeTensor
        LocalTensor<IDX_T> idxLocal = indexQue_.AllocTensor<IDX_T>();
        CopyIn(idxLocal, indicesGm_, batchStart * IDX_DIM, 1, idxNum * IDX_DIM);
        indexQue_.EnQue<IDX_T>(idxLocal);

        idxLocal = indexQue_.DeQue<IDX_T>();
        LocalTensor<int64_t> offsetLocal = offsetQue_.AllocTensor<int64_t>();
        __ubuf__ IDX_T* idxLocalPtr = (__ubuf__ IDX_T*)idxLocal.GetPhyAddr();
        __ubuf__ int64_t* offsetLocalPtr = (__ubuf__ int64_t*)offsetLocal.GetPhyAddr();
        ComputeOffsetForMode0VecImpl<IDX_T, IDX_DIM>(
            idxLocalPtr, offsetLocalPtr, rowsStart, batchStart, static_cast<uint32_t>(rows), vfLen, loopCnt,
            tilingData_->updatesDim1, tilingData_->inputDim1, tilingData_->inputDim2,
            tilingData_->inputDim3 * tilingData_->dtypeSize);
        indexQue_.FreeTensor<IDX_T>(idxLocal);

        event_t eventV2MT3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventV2MT3);
        WaitFlag<HardEvent::V_MTE3>(eventV2MT3);

        // Load updates
        LocalTensor<int8_t> xLocal = inQueue_.AllocTensor<int8_t>();
        int64_t srcOffset = rowsStart * cols * tilingData_->dtypeSize;
        CopyIn(xLocal, updatesGm_, srcOffset, rows, cols * tilingData_->dtypeSize);
        inQueue_.EnQue<int8_t>(xLocal);

        LocalTensor<int8_t> yLocal = inQueue_.DeQue<int8_t>();
        ScatterParams scatterParams(rows, cols * tilingData_->dtypeSize, colsAlign);
        ScatterDataCopy(yGm_, yLocal, offsetLocal, scatterParams);
        inQueue_.FreeTensor(yLocal);
        offsetQue_.FreeTensor(offsetLocal);
        event_t eventMTE32V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventMTE32V);
        WaitFlag<HardEvent::MTE3_V>(eventMTE32V);
    }
}

template <typename IDX_T, const uint16_t SIMD_MODE>
template <const uint16_t IDX_DIM>
__aicore__ inline void ScatterSimdScatterDMA<IDX_T, SIMD_MODE>::NoSplitColProcessForMode1(int64_t colsAlign)
{
    int64_t maxRowsOnce = tilingData_->loopLength * tilingData_->dtypeSize / colsAlign;
    int64_t loopNum = Ops::Base::CeilDiv(curCoreData_, maxRowsOnce);
    int64_t srcStart = blockIdx_ * tilingData_->blockFactor;
    int64_t cols = tilingData_->updatesDim3;
    int64_t dim12 = tilingData_->updatesDim1 * tilingData_->updatesDim2;
    constexpr uint32_t vfLen = Ops::Base::GetVRegSize() / sizeof(int64_t);
    for (int64_t i = 0; i < loopNum; i++) {
        int64_t rows = (i == loopNum - 1) ? (curCoreData_ - i * maxRowsOnce) : maxRowsOnce;
        int64_t rowsStart = srcStart + i * maxRowsOnce;
        int64_t batchStart = rowsStart / dim12;
        int64_t batchEnd = (rowsStart + rows - 1) / dim12;
        int32_t idxNum = batchEnd - batchStart + 1;
        uint16_t loopCnt = static_cast<uint16_t>(Ops::Base::CeilDiv(rows, static_cast<int64_t>(vfLen)));

        // Load indices → DeQue → SIMD compute offsets → FreeTensor
        LocalTensor<IDX_T> idxLocal = indexQue_.AllocTensor<IDX_T>();
        CopyIn(idxLocal, indicesGm_, batchStart * IDX_DIM, 1, idxNum * IDX_DIM);
        indexQue_.EnQue<IDX_T>(idxLocal);

        idxLocal = indexQue_.DeQue<IDX_T>();
        LocalTensor<int64_t> offsetLocal = offsetQue_.AllocTensor<int64_t>();
        __ubuf__ IDX_T* idxLocalPtr = (__ubuf__ IDX_T*)idxLocal.GetPhyAddr();
        __ubuf__ int64_t* offsetLocalPtr = (__ubuf__ int64_t*)offsetLocal.GetPhyAddr();
        ComputeOffsetForMode1VecImpl<IDX_T, IDX_DIM>(
            idxLocalPtr, offsetLocalPtr, rowsStart, batchStart, static_cast<uint32_t>(rows), vfLen, loopCnt, dim12,
            tilingData_->updatesDim2, tilingData_->inputDim1, tilingData_->inputDim2,
            tilingData_->inputDim3 * tilingData_->dtypeSize);
        indexQue_.FreeTensor<IDX_T>(idxLocal);

        event_t eventV2MT3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventV2MT3);
        WaitFlag<HardEvent::V_MTE3>(eventV2MT3);

        // Load updates
        LocalTensor<int8_t> xLocal = inQueue_.AllocTensor<int8_t>();
        int64_t srcOffset = (srcStart + i * maxRowsOnce) * tilingData_->updatesDim3 * tilingData_->dtypeSize;
        CopyIn(xLocal, updatesGm_, srcOffset, rows, cols * tilingData_->dtypeSize);
        inQueue_.EnQue<int8_t>(xLocal);

        LocalTensor<int8_t> yLocal = inQueue_.DeQue<int8_t>();
        ScatterParams scatterParams(rows, cols * tilingData_->dtypeSize, colsAlign);
        ScatterDataCopy(yGm_, yLocal, offsetLocal, scatterParams);
        inQueue_.FreeTensor(yLocal);
        offsetQue_.FreeTensor(offsetLocal);
        event_t eventMTE32V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventMTE32V);
        WaitFlag<HardEvent::MTE3_V>(eventMTE32V);
    }
}

template <typename IDX_T, const uint16_t SIMD_MODE>
__aicore__ inline void ScatterSimdScatterDMA<IDX_T, SIMD_MODE>::Process()
{
    if (blockIdx_ >= tilingData_->aivCoreNum) {
        return;
    }
    int64_t colsAlign = 0;
    if constexpr (SIMD_MODE == 0) {
        // split b*o
        colsAlign = (tilingData_->updatesDim2 * tilingData_->updatesDim3 * tilingData_->dtypeSize + ubBlockSize_ - 1) /
                    ubBlockSize_ * ubBlockSize_;
    } else {
        // split b*o*s
        colsAlign = (tilingData_->updatesDim3 * tilingData_->dtypeSize + ubBlockSize_ - 1) / ubBlockSize_ *
                    ubBlockSize_;
    }
    if (tilingData_->indicesDim == DIM_2) {
        NoSplitColProcess<DIM_2>(colsAlign);
    } else {
        NoSplitColProcess<1>(colsAlign);
    }
}
} // namespace SCATTER

#endif // ASCENDC_SCATTER_SIMD_SCATTER_DMA_H_
