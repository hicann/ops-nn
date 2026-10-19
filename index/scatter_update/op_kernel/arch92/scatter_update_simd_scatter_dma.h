/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file scatter_update_simd_scatter_dma.h
 * \brief A6(ascend960) ScatterDMA kernel of scatter_update
 */
#ifndef SCATTER_UPDATE_SIMD_SCATTER_DMA_H
#define SCATTER_UPDATE_SIMD_SCATTER_DMA_H

#include "../arch35/scatter_update_common.h"
#include "../arch35/scatter_update_struct.h"
#include "../inc/platform.h"

namespace ScatterUpdate {
using namespace AscendC;

constexpr AscendC::Reg::CastTrait castTraitB32B64 = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
                                                     AscendC::Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

template <typename T, typename U>
__aicore__ inline void ScatterDMA(GlobalTensor<T> dstGm, LocalTensor<T> srcLocal, LocalTensor<U> offsetLocal,
                                  uint16_t offsetNum, uint32_t burstLen, uint32_t srcStride)
{
    ScatterParams scatterDataCopyParams;
    scatterDataCopyParams.offsetNum = offsetNum;
    scatterDataCopyParams.burstLen = burstLen * sizeof(T);
    scatterDataCopyParams.srcStride = srcStride * sizeof(T);

    event_t eventIdV_MTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdV_MTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIdV_MTE3);

    ScatterDataCopy(dstGm, srcLocal, offsetLocal, scatterDataCopyParams);

    event_t eventIdMTE3_V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eventIdMTE3_V);
    WaitFlag<HardEvent::MTE3_V>(eventIdMTE3_V);
}

template <typename T>
__aicore__ inline void BroadcastUpdatesScalar(LocalTensor<T> updatesLocal, GlobalTensor<T> updatesGm, int32_t count)
{
    T updatesValue = updatesGm.GetValue(0);
    auto vWaitSEventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(vWaitSEventID);
    WaitFlag<HardEvent::S_V>(vWaitSEventID);
    Duplicate(updatesLocal, updatesValue, count);
}

template <typename T>
__aicore__ inline void CopyIn(LocalTensor<T> dstLocal, GlobalTensor<T> srcGm, uint64_t offset, uint32_t nBurst,
                              uint32_t copyLen, uint32_t srcStride = 0, uint32_t dstStride = 0)
{
    DataCopyPadExtParams<T> dataCopyPadExtParams;
    dataCopyPadExtParams.isPad = false;
    dataCopyPadExtParams.leftPadding = 0;
    dataCopyPadExtParams.rightPadding = 0;
    dataCopyPadExtParams.paddingValue = 0;

    DataCopyExtParams dataCoptExtParams;
    dataCoptExtParams.blockCount = nBurst;               // 连续传输块个数
    dataCoptExtParams.blockLen = copyLen * sizeof(T);    // 每块大小
    dataCoptExtParams.srcStride = srcStride * sizeof(T); // 源地址相邻块间隔
    dataCoptExtParams.dstStride = dstStride * sizeof(T); // 目的地址相邻块间隔
    DataCopyPad(dstLocal, srcGm[offset], dataCoptExtParams, dataCopyPadExtParams);
}

template <typename T, typename U, bool updatesIsScalar, bool isCastIndices>
class ScatterUpdateScatterDMA {
public:
    __aicore__ inline ScatterUpdateScatterDMA(const ScatterUpdateTilingData& tilingData, TPipe& pipe)
        : tilingData_(tilingData), pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR workspace);
    __aicore__ inline void Process();
    __aicore__ inline void Compute();
    __aicore__ inline void updatesScalarPreprocess();
    __aicore__ inline void ComputeOffsetForScatterDMA(LocalTensor<U> indicesLocal, uint32_t indicesCount,
                                                      int64_t maxIndicesValue, int64_t totalCol);
    __simd_vf__ inline void ComputeOffsetvf(__ubuf__ U* indicesAddr, uint16_t loopCnt, uint32_t vfLen,
                                            int64_t maxIndicesValue, int64_t blockColOffsetByte, uint32_t size,
                                            uint32_t sizeTwo, int64_t totalColByte, __ubuf__ int64_t* indicesCastAddr);

private:
    AscendC::GlobalTensor<T> varGm_;
    AscendC::GlobalTensor<U> indicesGm_;
    AscendC::GlobalTensor<T> updatesGm_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> indicesQueue_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, DOUBLE_BUFFER> updatesQueue_;
    TBuf<QuePosition::VECCALC> indicesCastBuf_;

    TPipe& pipe_;
    const ScatterUpdateTilingData& tilingData_;

    uint32_t blockIdx_{0};
    uint64_t rowBlockIdx_{0};
    uint64_t colBlockIdx_{0};
    uint64_t currBlockRowNum_{0};
    uint64_t currBlockColNum_{0};
    uint64_t blockRowOffset_{0};
    uint64_t blockColOffset_{0};
    uint64_t blockUpdatesOffset_{0};
};

template <typename T, typename U, bool updatesIsScalar, bool isCastIndices>
__aicore__ inline void ScatterUpdateScatterDMA<T, U, updatesIsScalar, isCastIndices>::Init(GM_ADDR var, GM_ADDR indices,
                                                                                           GM_ADDR updates,
                                                                                           GM_ADDR workspace)
{
    varGm_.SetGlobalBuffer((__gm__ T*)(var));
    indicesGm_.SetGlobalBuffer((__gm__ U*)(indices));
    updatesGm_.SetGlobalBuffer((__gm__ T*)(updates));
    pipe_.InitBuffer(indicesQueue_, DOUBLE_BUFFER,
                     ops::CeilAlign(tilingData_.indicesUbFactor * sizeof(U), UB_AGLIN_VALUE));
    pipe_.InitBuffer(updatesQueue_, DOUBLE_BUFFER,
                     tilingData_.indicesUbFactor * tilingData_.updateColUbFactor * sizeof(T));
    if constexpr (isCastIndices) {
        pipe_.InitBuffer(indicesCastBuf_,
                         ops::CeilAlign(tilingData_.indicesUbFactor * sizeof(int64_t), UB_AGLIN_VALUE));
    }

    blockIdx_ = GetBlockIdx();
    rowBlockIdx_ = blockIdx_ / tilingData_.colTileNum;
    colBlockIdx_ = blockIdx_ % tilingData_.colTileNum;
    currBlockRowNum_ = rowBlockIdx_ == tilingData_.rowTileNum - 1 ? tilingData_.tailBlockRowNum :
                                                                    tilingData_.normBlockRowNum;
    currBlockColNum_ = colBlockIdx_ == tilingData_.colTileNum - 1 ? tilingData_.tailBlockColNum :
                                                                    tilingData_.normBlockColNum;

    blockRowOffset_ = rowBlockIdx_ * tilingData_.normBlockRowNum; // 当前核行偏移
    blockColOffset_ = colBlockIdx_ * tilingData_.normBlockColNum; // 当前核列偏移
    blockUpdatesOffset_ = rowBlockIdx_ * tilingData_.normBlockRowNum * tilingData_.varShape[1] +
                          blockColOffset_; // 当前核updates块偏移位置
}

template <typename T, typename U, bool updatesIsScalar, bool isCastIndices>
__aicore__ inline void ScatterUpdateScatterDMA<T, U, updatesIsScalar, isCastIndices>::updatesScalarPreprocess()
{
    if constexpr (updatesIsScalar) {
        LocalTensor<T> updatesLocal = updatesQueue_.AllocTensor<T>();
        int32_t broadcastCount = static_cast<int32_t>(tilingData_.indicesUbFactor * tilingData_.updateColUbFactor);
        BroadcastUpdatesScalar<T>(updatesLocal, updatesGm_, broadcastCount);
        updatesQueue_.EnQue<T>(updatesLocal);
    }
}

template <typename T, typename U, bool updatesIsScalar, bool isCastIndices>
__simd_vf__ inline void ScatterUpdateScatterDMA<T, U, updatesIsScalar, isCastIndices>::ComputeOffsetvf(
    __ubuf__ U* indicesAddr, uint16_t loopCnt, uint32_t vfLen, int64_t maxIndicesValue, int64_t blockColOffsetByte,
    uint32_t size, uint32_t sizeTwo, int64_t totalColByte, __ubuf__ int64_t* indicesCastAddr)
{
    AscendC::Reg::RegTensor<U> indicesReg;
    AscendC::Reg::RegTensor<U> invalidIndicesReg;
    AscendC::Reg::RegTensor<int64_t, AscendC::Reg::RegTraitNumTwo> indicesCastReg;
    AscendC::Reg::MaskReg maskReg;
    AscendC::Reg::MaskReg maskRegTwo;
    AscendC::Reg::MaskReg cmpMask;

    AscendC::Reg::Duplicate(invalidIndicesReg, static_cast<U>(-1));
    for (uint16_t i = 0; i < loopCnt; i++) {
        maskReg = AscendC::Reg::UpdateMask<U>(size);
        AscendC::Reg::LoadAlign<U, AscendC::Reg::LoadDist::DIST_NORM>(indicesReg, indicesAddr + i * vfLen);
        AscendC::Reg::Compares<U, CMPMODE::GE>(cmpMask, indicesReg, 0, maskReg); // 负越界
        AscendC::Reg::Select(indicesReg, indicesReg, invalidIndicesReg, cmpMask);
        AscendC::Reg::Compares<U, CMPMODE::LT>(cmpMask, indicesReg, maxIndicesValue, maskReg); // 正越界
        AscendC::Reg::Select(indicesReg, indicesReg, invalidIndicesReg, cmpMask);
        if constexpr (isCastIndices && IsSameType<U, int32_t>::value) {
            maskRegTwo = AscendC::Reg::UpdateMask<int64_t, AscendC::Reg::RegTraitNumTwo>(sizeTwo);
            AscendC::Reg::Cast<int64_t, int32_t, castTraitB32B64>(indicesCastReg, indicesReg, maskRegTwo);
            AscendC::Reg::Muls(indicesCastReg, indicesCastReg, totalColByte, maskRegTwo);
            AscendC::Reg::Adds(indicesCastReg, indicesCastReg, blockColOffsetByte, maskRegTwo);
            AscendC::Reg::StoreAlign(indicesCastAddr + i * vfLen, indicesCastReg, maskRegTwo);
        } else {
            AscendC::Reg::Muls(indicesReg, indicesReg, static_cast<U>(totalColByte), maskReg);
            AscendC::Reg::Adds(indicesReg, indicesReg, static_cast<U>(blockColOffsetByte), maskReg);
            AscendC::Reg::StoreAlign(indicesAddr + i * vfLen, indicesReg, maskReg);
        }
    }
}

template <typename T, typename U, bool updatesIsScalar, bool isCastIndices>
__aicore__ inline void ScatterUpdateScatterDMA<T, U, updatesIsScalar, isCastIndices>::ComputeOffsetForScatterDMA(
    LocalTensor<U> indicesLocal, uint32_t indicesCount, int64_t maxIndicesValue, int64_t totalCol)
{
    __ubuf__ U* indicesAddr = (__ubuf__ U*)indicesLocal.GetPhyAddr();
    __ubuf__ int64_t* indicesCastAddr = nullptr;
    if constexpr (isCastIndices) {
        LocalTensor<int64_t> indicesCastLocal = indicesCastBuf_.Get<int64_t>();
        indicesCastAddr = (__ubuf__ int64_t*)indicesCastLocal.GetPhyAddr();
    }
    uint32_t vfLen = platform::GetVRegSize() / sizeof(U);
    uint16_t loopCnt = ops::CeilDiv(indicesCount, vfLen);
    uint32_t size = indicesCount;
    uint32_t sizeTwo = indicesCount;
    int64_t blockColOffsetByte = static_cast<int64_t>(blockColOffset_ * sizeof(T));
    int64_t totalColByte = totalCol * sizeof(T);

    ComputeOffsetvf(indicesAddr, loopCnt, vfLen, maxIndicesValue, blockColOffsetByte, size, sizeTwo, totalColByte,
                    indicesCastAddr);
}

template <typename T, typename U, bool updatesIsScalar, bool isCastIndices>
__aicore__ inline void ScatterUpdateScatterDMA<T, U, updatesIsScalar, isCastIndices>::Compute()
{
    updatesScalarPreprocess();

    uint64_t rowUbLoopNum = ops::CeilDiv(currBlockRowNum_, tilingData_.indicesUbFactor); // 当前分块在行方向UB循环次数
    uint64_t colUbLoopNum = ops::CeilDiv(currBlockColNum_, tilingData_.updateColUbFactor); // 当前分块在列方向UB循环次数
    int64_t totalCol = tilingData_.varShape[1];
    for (uint64_t rowLoop = 0; rowLoop < rowUbLoopNum; rowLoop++) {
        uint32_t rowTailLoopNum = currBlockRowNum_ - rowLoop * tilingData_.indicesUbFactor;
        uint32_t rows = (rowLoop == rowUbLoopNum - 1) ? rowTailLoopNum : tilingData_.indicesUbFactor;
        LocalTensor<U> indicesLocal = indicesQueue_.AllocTensor<U>();
        CopyIn<U>(indicesLocal, indicesGm_, blockRowOffset_ + rowLoop * tilingData_.indicesUbFactor, 1, rows);
        indicesQueue_.EnQue<U>(indicesLocal);

        indicesLocal = indicesQueue_.DeQue<U>();
        ComputeOffsetForScatterDMA(indicesLocal, rows, static_cast<int64_t>(tilingData_.varShape[0]), totalCol);
        for (uint64_t colLoop = 0; colLoop < colUbLoopNum; colLoop++) {
            uint32_t colTailLoopNum = currBlockColNum_ - colLoop * tilingData_.updateColUbFactor;
            uint32_t cols = (colLoop == colUbLoopNum - 1) ?
                                colTailLoopNum :
                                tilingData_.updateColUbFactor; // 当前UB拿到的updates小块的列数
            uint32_t colsAlign = ops::Aligned(static_cast<uint32_t>(cols * sizeof(T)),
                                              static_cast<uint32_t>(UB_AGLIN_VALUE)) /
                                 sizeof(T);
            if constexpr (!updatesIsScalar) {
                LocalTensor<T> updatesLocal = updatesQueue_.AllocTensor<T>();
                uint64_t updatesoffset = blockUpdatesOffset_ + rowLoop * tilingData_.indicesUbFactor * totalCol +
                                         colLoop * tilingData_.updateColUbFactor;
                CopyIn<T>(updatesLocal, updatesGm_, updatesoffset, rows, cols, totalCol - cols);
                updatesQueue_.EnQue<T>(updatesLocal);
            }

            LocalTensor<T> updatesLocal = updatesQueue_.DeQue<T>();
            if constexpr (isCastIndices) {
                LocalTensor<int64_t> indicesCastLocal = indicesCastBuf_.Get<int64_t>();
                ScatterDMA<T, int64_t>(varGm_, updatesLocal, indicesCastLocal, static_cast<uint16_t>(rows), cols,
                                       colsAlign);
                Adds(indicesCastLocal, indicesCastLocal,
                     static_cast<int64_t>(tilingData_.updateColUbFactor * sizeof(T)), static_cast<int32_t>(rows));
            } else {
                ScatterDMA<T, U>(varGm_, updatesLocal, indicesLocal, static_cast<uint16_t>(rows), cols, colsAlign);
                Adds(indicesLocal, indicesLocal, static_cast<U>(tilingData_.updateColUbFactor * sizeof(T)),
                     static_cast<int32_t>(rows));
            }

            if constexpr (updatesIsScalar) {
                updatesQueue_.EnQue<T>(updatesLocal);
            } else {
                updatesQueue_.FreeTensor<T>(updatesLocal);
            }
        }
        indicesQueue_.FreeTensor(indicesLocal);
    }
    if constexpr (updatesIsScalar) {
        LocalTensor<T> updatesLocal = updatesQueue_.DeQue<T>();
        updatesQueue_.FreeTensor<T>(updatesLocal);
    }
}

template <typename T, typename U, bool updatesIsScalar, bool isCastIndices>
__aicore__ inline void ScatterUpdateScatterDMA<T, U, updatesIsScalar, isCastIndices>::Process()
{
    if (GetBlockIdx() >= this->tilingData_.usedCoreNum) {
        return;
    }

    Compute();
}

} // namespace ScatterUpdate

#endif // SCATTER_UPDATE_SIMD_SCATTER_DMA_H
