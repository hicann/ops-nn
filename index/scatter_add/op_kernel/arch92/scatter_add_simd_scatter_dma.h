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
 * \file scatter_add_simd_scatter_dma.h
 * \brief simd_scatter_dma kernel of scatter_add
 */

#ifndef SCATTER_ADD_SIMD_SCATTER_DMA_H
#define SCATTER_ADD_SIMD_SCATTER_DMA_H

#include "../arch35/scatter_add_common.h"

namespace ScatterAdd {
using namespace AscendC;
using namespace ScatterAddCommon;

constexpr uint64_t BUFFER_NUM_DMA = 2;

constexpr static AscendC::Reg::CastTrait castTraitB32B64 = {
    AscendC::Reg::RegLayout::ZERO,
    AscendC::Reg::SatMode::UNKNOWN,
    AscendC::Reg::MaskMergeMode::ZEROING,
};

template <typename T, typename U>
__aicore__ inline void ScatterAddDMA(GlobalTensor<T> dstGm, LocalTensor<T> srcLocal, LocalTensor<U> offsetLocal,
                                     uint16_t offsetNum, uint32_t burstLen, uint32_t srcStride)
{
    SetAtomicAdd<T>();
    ScatterParams scatterDataCopyParams;
    scatterDataCopyParams.offsetNum = offsetNum;
    scatterDataCopyParams.burstLen = burstLen * sizeof(T);
    scatterDataCopyParams.srcStride = srcStride * sizeof(T);

    event_t eventIdV_MTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdV_MTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIdV_MTE3);

    ScatterDataCopy(dstGm, srcLocal, offsetLocal, scatterDataCopyParams);
    SetAtomicNone();

    event_t eventIdMTE3_V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eventIdMTE3_V);
    WaitFlag<HardEvent::MTE3_V>(eventIdMTE3_V);
}

template <typename T, typename U, bool updatesIsScalar, bool isCastIndices, uint32_t scatterOp>
class ScatterAddScatterDMA {
public:
    __aicore__ inline ScatterAddScatterDMA(const ScatterAddTilingData& tilingData, TPipe& pipe)
        : tilingData_(tilingData), pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR varRef, GM_ADDR workspace);
    __aicore__ inline void ProcessAtomicAdd();
    __aicore__ inline void Process();
    __aicore__ inline void ComputeOffsetForScatterDMA(LocalTensor<U> indicesLocal, uint32_t rows, uint64_t colIdx);

private:
    AscendC::GlobalTensor<T> varGm_;
    AscendC::GlobalTensor<U> indicesGm_;
    AscendC::GlobalTensor<T> updatesGm_;
    AscendC::GlobalTensor<T> varRefGm_;

    TQueBind<updatesIsScalar ? QuePosition::VECOUT : QuePosition::VECIN,
             updatesIsScalar ? QuePosition::GM : QuePosition::VECOUT, BUFFER_NUM_DMA>
        updatesQueue_;
    TQue<QuePosition::VECIN, BUFFER_NUM_DMA> indicesQueue_;
    TBuf<QuePosition::VECCALC> indicesCastBuf_;
    TPipe& pipe_;
    const ScatterAddTilingData& tilingData_;
};

template <typename T, typename U, bool updatesIsScalar, bool isCastIndices, uint32_t scatterOp>
__aicore__ inline void ScatterAddScatterDMA<T, U, updatesIsScalar, isCastIndices, scatterOp>::Init(
    GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR varRef, GM_ADDR workspace)
{
    if (GetBlockIdx() >= GetBlockNum()) {
        return;
    }

    pipe_.InitBuffer(updatesQueue_, BUFFER_NUM_DMA, tilingData_.ubFactorRow * tilingData_.ubFactorCol * sizeof(T));
    pipe_.InitBuffer(indicesQueue_, BUFFER_NUM_DMA,
                     ops::CeilAlign(tilingData_.ubFactorRow * sizeof(U), UB_AGLIN_VALUE));
    if constexpr (isCastIndices) {
        pipe_.InitBuffer(indicesCastBuf_, ops::CeilAlign(tilingData_.ubFactorRow * sizeof(int64_t), UB_AGLIN_VALUE));
    }
    varRefGm_.SetGlobalBuffer((__gm__ T*)(varRef));

    if (GetBlockIdx() < tilingData_.atomicAddCoreNum) {
        indicesGm_.SetGlobalBuffer((__gm__ U*)(indices));
        updatesGm_.SetGlobalBuffer((__gm__ T*)(updates));
    }
}

template <typename U, bool isCastIndices>
__simd_vf__ inline void ComputeOffsetVF(__ubuf__ U* indicesAddr, __ubuf__ int64_t* indicesCastAddr, uint32_t vfLen,
                                        uint16_t loopCnt, uint32_t size, uint32_t sizeTwo, int64_t blockColOffsetByte,
                                        int64_t totalColByte, int64_t maxIndicesValue)
{
    Reg::RegTensor<U> indicesReg;
    Reg::RegTensor<U> invalidIndicesReg;
    Reg::RegTensor<int64_t, Reg::RegTraitNumTwo> indicesCastReg;
    Reg::MaskReg maskReg;
    Reg::MaskReg maskRegTwo;
    Reg::MaskReg cmpMask;

    Reg::Duplicate(invalidIndicesReg, static_cast<U>(-1));
    for (uint16_t i = 0; i < loopCnt; i++) {
        maskReg = Reg::UpdateMask<U>(size);
        Reg::LoadAlign<U, Reg::LoadDist::DIST_NORM>(indicesReg, indicesAddr + i * vfLen);
        Reg::Compares<U, CMPMODE::LT>(cmpMask, indicesReg, maxIndicesValue, maskReg);
        Reg::Select(indicesReg, indicesReg, invalidIndicesReg, cmpMask);
        if constexpr (isCastIndices && IsSameType<U, int32_t>::value) {
            maskRegTwo = Reg::UpdateMask<int64_t, Reg::RegTraitNumTwo>(sizeTwo);
            Reg::Cast<int64_t, int32_t, castTraitB32B64>(indicesCastReg, indicesReg, maskRegTwo);
            Reg::Muls(indicesCastReg, indicesCastReg, totalColByte, maskRegTwo);
            Reg::Adds(indicesCastReg, indicesCastReg, blockColOffsetByte, maskRegTwo);
            Reg::StoreAlign(indicesCastAddr + i * vfLen, indicesCastReg, maskRegTwo);
        } else {
            Reg::Muls(indicesReg, indicesReg, static_cast<U>(totalColByte), maskReg);
            Reg::Adds(indicesReg, indicesReg, static_cast<U>(blockColOffsetByte), maskReg);
            Reg::StoreAlign(indicesAddr + i * vfLen, indicesReg, maskReg);
        }
    }
}

template <typename T, typename U, bool updatesIsScalar, bool isCastIndices, uint32_t scatterOp>
__aicore__ inline void ScatterAddScatterDMA<T, U, updatesIsScalar, isCastIndices,
                                            scatterOp>::ComputeOffsetForScatterDMA(LocalTensor<U> indicesLocal,
                                                                                   uint32_t rows, uint64_t colIdx)
{
    __ubuf__ U* indicesAddr = (__ubuf__ U*)indicesLocal.GetPhyAddr();
    __ubuf__ int64_t* indicesCastAddr = nullptr;
    if constexpr (isCastIndices) {
        LocalTensor<int64_t> indicesCastLocal = indicesCastBuf_.Get<int64_t>();
        indicesCastAddr = (__ubuf__ int64_t*)indicesCastLocal.GetPhyAddr();
    }

    uint32_t vfLen = platform::GetVRegSize() / sizeof(U);
    uint16_t loopCnt = ops::CeilDiv(static_cast<uint32_t>(rows), vfLen);
    uint32_t size = rows;
    uint32_t sizeTwo = rows;
    int64_t blockColOffsetByte = static_cast<int64_t>(colIdx * tilingData_.normBlockCol * sizeof(T));
    int64_t totalColByte = tilingData_.varShape[1] * sizeof(T);
    int64_t maxIndicesValue = tilingData_.varShape[0];

    ComputeOffsetVF<U, isCastIndices>(indicesAddr, indicesCastAddr, vfLen, loopCnt, size, sizeTwo, blockColOffsetByte,
                                      totalColByte, maxIndicesValue);
}

template <typename T, typename U, bool updatesIsScalar, bool isCastIndices, uint32_t scatterOp>
__aicore__ inline void ScatterAddScatterDMA<T, U, updatesIsScalar, isCastIndices, scatterOp>::ProcessAtomicAdd()
{
    if constexpr (updatesIsScalar) {
        LocalTensor<T> updatesLocal = updatesQueue_.template AllocTensor<T>();
        int32_t broadcastCount = static_cast<int32_t>(tilingData_.ubFactorRow *
                                                      tilingData_.ubFactorCol); // ScatterDMA 需要多行
        BroadcastUpdatesScalar<T, scatterOp>(updatesLocal, updatesGm_, broadcastCount);
        updatesQueue_.EnQue(updatesLocal);
    }

    uint64_t rowIdx = GetBlockIdx() / tilingData_.colTileNum; // 当前block在第几个分块行
    uint64_t colIdx = GetBlockIdx() % tilingData_.colTileNum; // 当前block在第几个分块列
    uint64_t curCoreRows = rowIdx != (tilingData_.rowTileNum - 1) ? tilingData_.normBlockRow :
                                                                    tilingData_.tailBlockRow; // 当前分块行数
    uint64_t curCoreCols = colIdx != (tilingData_.colTileNum - 1) ? tilingData_.normBlockCol :
                                                                    tilingData_.tailBlockCol; // 当前分块列数

    uint64_t blockOffsetUpdate = rowIdx * tilingData_.normBlockRow * tilingData_.varShape[1] +
                                 colIdx * tilingData_.normBlockCol;  // 当前updates块的偏移位置
    uint64_t blockOffsetindices = rowIdx * tilingData_.normBlockRow; // 当前indices的偏移位置

    uint64_t rowUbLoopNum = ops::CeilDiv(curCoreRows, tilingData_.ubFactorRow); // 当前分块在行方向需要UB循环多少次

    for (uint64_t rowLoop = 0; rowLoop < rowUbLoopNum; rowLoop++) {
        uint64_t rows = (rowLoop == rowUbLoopNum - 1) ? (curCoreRows - rowLoop * tilingData_.ubFactorRow) :
                                                        tilingData_.ubFactorRow;

        LocalTensor<U> indicesLocal = indicesQueue_.AllocTensor<U>();
        CopyIn(indicesLocal, indicesGm_, blockOffsetindices + rowLoop * tilingData_.ubFactorRow, 1, rows);
        indicesQueue_.EnQue<U>(indicesLocal);

        indicesLocal = indicesQueue_.DeQue<U>();
        ComputeOffsetForScatterDMA(indicesLocal, rows, colIdx);

        uint64_t cols = curCoreCols;
        uint64_t colsAlign = ops::Aligned(static_cast<uint64_t>(cols * sizeof(T)),
                                          static_cast<uint64_t>(ONE_BLOCK_SIZE)) /
                             sizeof(T);

        if constexpr (!updatesIsScalar) {
            LocalTensor<T> updatesLocal = updatesQueue_.template AllocTensor<T>();
            uint64_t offset = blockOffsetUpdate + rowLoop * tilingData_.ubFactorRow * tilingData_.varShape[1];

            CopyIn(updatesLocal, updatesGm_, offset, rows, cols, tilingData_.varShape[1] - cols);

            if constexpr (scatterOp == SUB) {
                auto VWaitMTE2EventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                SetFlag<HardEvent::MTE2_V>(VWaitMTE2EventID);
                WaitFlag<HardEvent::MTE2_V>(VWaitMTE2EventID);
                NegateUpdate<T>(updatesLocal, static_cast<uint32_t>(rows * colsAlign));
            }
            updatesQueue_.EnQue(updatesLocal);
        }
        LocalTensor<T> updatesLocal = updatesQueue_.template DeQue<T>();
        if constexpr (isCastIndices) {
            LocalTensor<int64_t> indicesCastLocal = indicesCastBuf_.Get<int64_t>();
            ScatterAddDMA<T, int64_t>(varRefGm_, updatesLocal, indicesCastLocal, static_cast<uint16_t>(rows), cols,
                                      colsAlign);
        } else {
            ScatterAddDMA<T, U>(varRefGm_, updatesLocal, indicesLocal, static_cast<uint16_t>(rows), cols, colsAlign);
        }

        if constexpr (updatesIsScalar) {
            updatesQueue_.EnQue(updatesLocal);
        } else {
            updatesQueue_.FreeTensor(updatesLocal);
        }

        indicesQueue_.FreeTensor(indicesLocal);
    }

    if constexpr (updatesIsScalar) {
        LocalTensor<T> updatesLocal = updatesQueue_.template DeQue<T>();
        updatesQueue_.FreeTensor(updatesLocal);
    }
}

template <typename T, typename U, bool updatesIsScalar, bool isCastIndices, uint32_t scatterOp>
__aicore__ inline void ScatterAddScatterDMA<T, U, updatesIsScalar, isCastIndices, scatterOp>::Process()
{
    if (GetBlockIdx() >= GetBlockNum() || GetBlockIdx() >= tilingData_.atomicAddCoreNum) {
        return;
    }

    ProcessAtomicAdd();
}

} // namespace ScatterAdd
#endif // SCATTER_ADD_SIMD_SCATTER_DMA_H
