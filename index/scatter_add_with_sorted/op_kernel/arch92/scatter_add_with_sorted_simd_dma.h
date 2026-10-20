/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SCATTER_ADD_WITH_SORTED_SIMD_DMA_H
#define SCATTER_ADD_WITH_SORTED_SIMD_DMA_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../inc/platform.h"
#include "../arch35/scatter_add_with_sorted_struct.h"
#include "../arch35/scatter_add_with_sorted_simd_common.h"

namespace ScatterAddWithSorted {
using namespace AscendC;

constexpr uint32_t sortBufferNum = 1;
constexpr uint32_t SORT_PADDING = 64;
constexpr uint64_t UB_AGLIN_VALUE = 32;
constexpr int64_t VFLEN_INT64 = platform::GetVRegSize() / sizeof(int64_t);
constexpr int64_t VFLEN_INT32 = platform::GetVRegSize() / sizeof(int32_t);

constexpr static AscendC::MicroAPI::CastTrait castTraitB32B64 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
};

template <typename T, typename U, bool withPos, bool isCastIndices>
class ScatterAddWithSortedSimdDMA {
public:
    __aicore__ inline ScatterAddWithSortedSimdDMA(const ScatterAddWithSortedSimdDMATilingData& tilingData, TPipe& pipe)
        : tilingData_(tilingData), pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR var, GM_ADDR updates, GM_ADDR indices, GM_ADDR pos, GM_ADDR varRef,
                                GM_ADDR workspace);
    __aicore__ inline void Process();
    __aicore__ inline void Compute();
    __aicore__ inline void ProcessPos(uint64_t indicesGmOffset, uint32_t cols, uint32_t colsAlign, uint32_t rows);
    template <bool isGatherDma>
    __aicore__ inline void ComputeOffsetForDMA(LocalTensor<U> posLocal, uint32_t rows, int64_t maxIndicesValue,
                                               bool unAlign = false);
    template <typename OFFSET_T>
    __aicore__ inline void GatherDMA(LocalTensor<OFFSET_T> offsetLocal, LocalTensor<T> dstLocal, uint16_t offsetNum,
                                     uint32_t burstLen, uint32_t dstStride);
    __aicore__ inline uint32_t ProcessIndices(uint64_t indicesGmOffset, uint64_t rowLoop, uint32_t rows,
                                              uint64_t rowUbLoopNum);
    __aicore__ inline uint32_t ComputeUniqueIdNum(LocalTensor<U> indicesLocal, LocalTensor<int32_t> uniqueIdCountLocal,
                                                  int64_t dataLen);
    __aicore__ inline void ComputeUniqueIdTimes(LocalTensor<int32_t>& noDupRes, uint32_t& arNum);
    template <typename XT>
    __aicore__ inline void CopyIn(LocalTensor<XT> xLocal, GlobalTensor<XT> xGm, uint64_t offset, uint32_t nBurst,
                                  uint32_t copyLen, uint32_t srcStride = 0, uint32_t dstStride = 0);
    __aicore__ inline void ComputeUpdatesSum(uint64_t cols, uint64_t colsAlign, uint32_t uniqueIdNum);
    template <typename OFFSET_T>
    __aicore__ inline void ScatterAddDMA(LocalTensor<T> srcLocal, LocalTensor<OFFSET_T> offsetLocal, uint16_t offsetNum,
                                         uint32_t burstLen, uint32_t srcStride);
    __aicore__ inline void CopyOutToWorkspace(LocalTensor<T> dataLocal, int32_t burstLen, int32_t writePosition);
    __aicore__ inline void CopyOutIndexToWorkspace(LocalTensor<U>& tmpLocal);

private:
    GlobalTensor<T> varGm_;
    GlobalTensor<U> indicesGm_;
    GlobalTensor<U> posGm_;
    GlobalTensor<T> updatesGm_;
    GlobalTensor<T> varRefGm_;
    GlobalTensor<T> sumWorkspace_;
    GlobalTensor<U> indexWorkspace_;

    TQue<QuePosition::VECIN, sortBufferNum> updatesQueue_;
    TQue<QuePosition::VECIN, sortBufferNum> indicesQueue_;
    TQue<QuePosition::VECIN, sortBufferNum> posQueue_;
    TQue<QuePosition::VECIN, 1> lastCoreLastIndexQueue_;
    TQue<QuePosition::VECIN, 1> nextCoreFirstIndexQueue_;
    TQue<QuePosition::VECOUT, sortBufferNum> outQueueRes_;
    TBuf<QuePosition::VECCALC> uniqueIdCountQue_;
    TBuf<QuePosition::VECCALC> updateSumIdxQue_;

    TBuf<QuePosition::VECCALC> posCastBuf_;
    TBuf<QuePosition::VECCALC> indicesCastBuf_;
    TBuf<QuePosition::VECCALC> tmpBuf_;
    TBuf<QuePosition::VECCALC> tmpBufOneRow_;

    TPipe& pipe_;
    const ScatterAddWithSortedSimdDMATilingData& tilingData_;

    uint64_t rowCoreIdx_ = 0;
    uint64_t colCoreIdx_ = 0;
    bool isStartRowCore_ = false;
    bool isEndRowCore_ = false;
    U prevCoreLastIndex_ = static_cast<U>(-2);
    U nextCoreFirstIndex_ = static_cast<U>(-2);
    U headIndex_ = static_cast<U>(-1);
    U tailIndex_ = static_cast<U>(-1);
    U curLoopLastId_ = static_cast<U>(-1);
    U nextLoopFirstId_ = static_cast<U>(-2);

    bool headToWorkspace_ = false;
    bool tailToWorkspace_ = false;
    bool saveLastId_ = false; // 保存当前ub循环的最后一个id的累加和

    static constexpr uint32_t vfLenUpdate_ = platform::GetVRegSize() / sizeof(T);
};

template <typename T, typename U, bool withPos, bool isCastIndices>
__aicore__ inline void ScatterAddWithSortedSimdDMA<T, U, withPos, isCastIndices>::Init(GM_ADDR var, GM_ADDR updates,
                                                                                       GM_ADDR indices, GM_ADDR pos,
                                                                                       GM_ADDR varRef,
                                                                                       GM_ADDR workspace)
{
    rowCoreIdx_ = GetBlockIdx() / tilingData_.colTileNum; // 当前block在第几个分块行
    colCoreIdx_ = GetBlockIdx() % tilingData_.colTileNum; // 当前block在第几个分块列
    isStartRowCore_ = (rowCoreIdx_ == 0);
    isEndRowCore_ = (rowCoreIdx_ == tilingData_.rowTileNum - 1);

    varRefGm_.SetGlobalBuffer((__gm__ T*)(varRef));
    indicesGm_.SetGlobalBuffer((__gm__ U*)(indices));
    updatesGm_.SetGlobalBuffer((__gm__ T*)(updates));
    if constexpr (withPos) {
        posGm_.SetGlobalBuffer((__gm__ U*)(pos));
    }

    int64_t sumAreaBytes = static_cast<int64_t>(tilingData_.rowTileNum) * kDouble * tilingData_.vecAlignSize;
    int64_t indexAreaOffsetBytes = sumAreaBytes + static_cast<int64_t>(rowCoreIdx_) * kCacheLineSize;
    int64_t sumOffsetT = static_cast<int64_t>(rowCoreIdx_) * kDouble * tilingData_.vecAlignSize / sizeof(T);
    int64_t indexOffsetU = (indexAreaOffsetBytes + sizeof(U) - 1) / sizeof(U);
    sumWorkspace_.SetGlobalBuffer((__gm__ T*)workspace + sumOffsetT);
    indexWorkspace_.SetGlobalBuffer((__gm__ U*)workspace + indexOffsetU);

    pipe_.InitBuffer(updatesQueue_, sortBufferNum,
                     ops::CeilAlign(tilingData_.ubFactorRow * tilingData_.ubFactorCol * sizeof(T), UB_AGLIN_VALUE));
    pipe_.InitBuffer(indicesQueue_, sortBufferNum,
                     ops::CeilAlign(tilingData_.ubFactorRow * sizeof(U), UB_AGLIN_VALUE) + SORT_PADDING);
    if constexpr (withPos) {
        pipe_.InitBuffer(posQueue_, sortBufferNum, ops::CeilAlign(tilingData_.ubFactorRow * sizeof(U), UB_AGLIN_VALUE));
    }
    pipe_.InitBuffer(lastCoreLastIndexQueue_, 1, platform::GetUbBlockSize());
    pipe_.InitBuffer(nextCoreFirstIndexQueue_, 1, platform::GetUbBlockSize());
    pipe_.InitBuffer(outQueueRes_, sortBufferNum, tilingData_.ubFactorRow * tilingData_.ubFactorCol * sizeof(T));
    pipe_.InitBuffer(uniqueIdCountQue_,
                     ops::CeilAlign((tilingData_.ubFactorRow + 1) * sizeof(int32_t), UB_AGLIN_VALUE));
    pipe_.InitBuffer(updateSumIdxQue_, ops::CeilAlign((tilingData_.ubFactorRow + 1) * sizeof(U), UB_AGLIN_VALUE));
    pipe_.InitBuffer(tmpBuf_, platform::GetUbBlockSize());
    pipe_.InitBuffer(tmpBufOneRow_, tilingData_.ubFactorCol * sizeof(T));
    if constexpr (isCastIndices) {
        pipe_.InitBuffer(posCastBuf_, ops::CeilAlign(tilingData_.ubFactorRow * sizeof(int64_t), UB_AGLIN_VALUE));
        pipe_.InitBuffer(indicesCastBuf_, ops::CeilAlign(tilingData_.ubFactorRow * sizeof(int64_t), UB_AGLIN_VALUE));
    }
}

template <typename T, typename U, bool withPos, bool isCastIndices>
template <typename XT>
__aicore__ inline void ScatterAddWithSortedSimdDMA<T, U, withPos, isCastIndices>::CopyIn(
    LocalTensor<XT> xLocal, GlobalTensor<XT> xGm, uint64_t offset, uint32_t nBurst, uint32_t copyLen,
    uint32_t srcStride, uint32_t dstStride)
{
    DataCopyPadExtParams<XT> dataCopyPadExtParams;
    dataCopyPadExtParams.isPad = false;
    dataCopyPadExtParams.leftPadding = 0;
    dataCopyPadExtParams.rightPadding = 0;
    dataCopyPadExtParams.paddingValue = 0;

    DataCopyExtParams dataCoptExtParams;
    dataCoptExtParams.blockCount = nBurst;                // 连续传输块个数
    dataCoptExtParams.blockLen = copyLen * sizeof(XT);    // 每块大小
    dataCoptExtParams.srcStride = srcStride * sizeof(XT); // 源地址相邻块间隔
    dataCoptExtParams.dstStride = dstStride * sizeof(XT); // 目的地址相邻块间隔
    DataCopyPad(xLocal, xGm[offset], dataCoptExtParams, dataCopyPadExtParams);
}

template <typename IDX_T>
__aicore__ inline void ComputeUniqueIdNumInt64(__local_mem__ IDX_T* indicesAddr,
                                               __local_mem__ int32_t* uniqueIdCountsAddr, uint16_t loopCnt,
                                               int64_t dataLen)
{
    uint32_t counter = dataLen + 1;
    AscendC::MicroAPI::RegTensor<int32_t> orderReg, selReg;
    AscendC::MicroAPI::RegTensor<IDX_T> sortedIdxReg, sortedIdxShiftOneReg;
    AscendC::MicroAPI::MaskReg cmpMask, maskReg, maskHalf;
    AscendC::MicroAPI::UnalignReg u0, uOut;
    for (uint16_t i = 0; i < loopCnt; ++i) {
        AscendC::MicroAPI::Arange(orderReg, i * VFLEN_INT64);
        maskReg = AscendC::MicroAPI::UpdateMask<IDX_T>(counter);
        auto startAddr = indicesAddr + i * VFLEN_INT64;
        DataCopy(sortedIdxReg, startAddr);
        AscendC::MicroAPI::DataCopyUnAlignPre(u0, startAddr - 1);
        AscendC::MicroAPI::DataCopyUnAlign<IDX_T>(sortedIdxShiftOneReg, u0, startAddr - 1);
        AscendC::MicroAPI::Compare<IDX_T, CMPMODE::NE>(cmpMask, sortedIdxReg, sortedIdxShiftOneReg, maskReg);
        AscendC::MicroAPI::MaskPack<AscendC::MicroAPI::HighLowPart::LOWEST>(maskHalf, cmpMask);
        AscendC::MicroAPI::GatherMask<int32_t, AscendC::MicroAPI::GatherMaskMode::STORE_REG>(selReg, orderReg,
                                                                                             maskHalf);
        AscendC::MicroAPI::DataCopyUnAlign<int32_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            uniqueIdCountsAddr, selReg, uOut);
    }
    AscendC::MicroAPI::DataCopyUnAlignPost(uniqueIdCountsAddr, uOut);
}

template <typename IDX_T>
__aicore__ inline void ComputeUniqueIdNumInt32(__local_mem__ IDX_T* indicesAddr,
                                               __local_mem__ int32_t* uniqueIdCountsAddr, uint16_t loopCnt,
                                               int64_t dataLen)
{
    uint32_t counter = dataLen + 1;
    AscendC::MicroAPI::RegTensor<int32_t> orderReg, selReg;
    AscendC::MicroAPI::RegTensor<IDX_T> sortedIdxReg, sortedIdxShiftOneReg;
    AscendC::MicroAPI::MaskReg cmpMask, maskReg;
    AscendC::MicroAPI::UnalignReg u0, uOut;
    for (uint16_t i = 0; i < loopCnt; ++i) {
        AscendC::MicroAPI::Arange(orderReg, i * VFLEN_INT32);
        maskReg = AscendC::MicroAPI::UpdateMask<IDX_T>(counter);
        auto startAddr = indicesAddr + i * VFLEN_INT32;
        DataCopy(sortedIdxReg, startAddr);
        AscendC::MicroAPI::DataCopyUnAlignPre(u0, startAddr - 1);
        AscendC::MicroAPI::DataCopyUnAlign<IDX_T>(sortedIdxShiftOneReg, u0, startAddr - 1);
        AscendC::MicroAPI::Compare<IDX_T, CMPMODE::NE>(cmpMask, sortedIdxReg, sortedIdxShiftOneReg, maskReg);
        AscendC::MicroAPI::GatherMask<int32_t, AscendC::MicroAPI::GatherMaskMode::STORE_REG>(selReg, orderReg, cmpMask);
        AscendC::MicroAPI::DataCopyUnAlign<int32_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            uniqueIdCountsAddr, selReg, uOut);
    }
    AscendC::MicroAPI::DataCopyUnAlignPost(uniqueIdCountsAddr, uOut);
}

template <typename T, typename U, bool withPos, bool isCastIndices>
__aicore__ inline uint32_t ScatterAddWithSortedSimdDMA<T, U, withPos, isCastIndices>::ComputeUniqueIdNum(
    LocalTensor<U> indicesLocal, LocalTensor<int32_t> uniqueIdCountLocal, int64_t dataLen)
{
    __local_mem__ U* indicesAddr = (__local_mem__ U*)indicesLocal[(UB_AGLIN_VALUE / sizeof(U))].GetPhyAddr();
    __local_mem__ int32_t* uniqueIdCountsAddr = (__local_mem__ int32_t*)uniqueIdCountLocal.GetPhyAddr();

    constexpr int64_t vfLen = platform::GetVRegSize() / sizeof(U);
    uint16_t loopCnt = ops::CeilDiv(dataLen + 1, vfLen);
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::ClearSpr<AscendC::SpecialPurposeReg::AR>();

        if constexpr (std::is_same<int64_t, U>::value) {
            ComputeUniqueIdNumInt64<U>(indicesAddr, uniqueIdCountsAddr, loopCnt, dataLen);
        } else {
            ComputeUniqueIdNumInt32<U>(indicesAddr, uniqueIdCountsAddr, loopCnt, dataLen);
        }
    }
    uint32_t arCount = AscendC::MicroAPI::GetSpr<AscendC::SpecialPurposeReg::AR>() / sizeof(int32_t);
    uint32_t uniqueIdNum;

    // 防止索引全为-1的场景下溢
    if (arCount > 0) {
        uniqueIdNum = arCount - 1;
    } else {
        uniqueIdNum = 0;
    }
    return uniqueIdNum;
}

template <typename T, typename U, bool withPos, bool isCastIndices>
__aicore__ inline void ScatterAddWithSortedSimdDMA<T, U, withPos, isCastIndices>::ComputeUniqueIdTimes(
    LocalTensor<int32_t>& noDupRes, uint32_t& arNum)
{
    __local_mem__ int32_t* noDupResAddr = (__ubuf__ int32_t*)noDupRes.GetPhyAddr();
    uint16_t loopCntStatFre = (arNum + VFLEN_INT32 - 1) / VFLEN_INT32;
    uint32_t counterStatFre = arNum;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<int32_t> beginReg;
        AscendC::MicroAPI::RegTensor<int32_t> endReg;
        AscendC::MicroAPI::RegTensor<int32_t> subReg;
        AscendC::MicroAPI::MaskReg maskRegUpdate;
        AscendC::MicroAPI::UnalignReg u0;

        for (uint16_t i = 0; i < loopCntStatFre; i++) {
            auto noDupResAddrUpdate = noDupResAddr + i * VFLEN_INT32 + 1;
            maskRegUpdate = AscendC::MicroAPI::UpdateMask<int32_t>(counterStatFre);
            AscendC::MicroAPI::DataCopy(beginReg, noDupResAddr + i * VFLEN_INT32);
            AscendC::MicroAPI::DataCopyUnAlignPre(u0, noDupResAddrUpdate);
            AscendC::MicroAPI::DataCopyUnAlign<int32_t>(endReg, u0, noDupResAddrUpdate);
            AscendC::MicroAPI::Sub(subReg, endReg, beginReg, maskRegUpdate);
            AscendC::MicroAPI::DataCopy(noDupResAddr + i * VFLEN_INT32, subReg, maskRegUpdate);
        }
    }
}

template <typename U, bool isCastIndices>
__simd_vf__ inline void ComputeOffsetVF(__ubuf__ U* posAddr, __ubuf__ int64_t* posCastAddr, uint32_t vfLen,
                                        uint16_t loopCnt, uint32_t size, uint32_t sizeTwo, int64_t blockColOffsetByte,
                                        int64_t totalColByte, int64_t maxIndicesValue)
{
    MicroAPI::RegTensor<U> indicesReg;
    MicroAPI::RegTensor<U> invalidIndicesReg;
    MicroAPI::RegTensor<int64_t, MicroAPI::RegTraitNumTwo> indicesCastReg;
    MicroAPI::MaskReg maskReg;
    MicroAPI::MaskReg maskRegTwo;
    MicroAPI::MaskReg cmpMask;

    MicroAPI::Duplicate(invalidIndicesReg, static_cast<U>(-1));
    for (uint16_t i = 0; i < loopCnt; i++) {
        maskReg = MicroAPI::UpdateMask<U>(size);
        MicroAPI::DataCopy<U, MicroAPI::LoadDist::DIST_NORM>(indicesReg, posAddr + i * vfLen);
        MicroAPI::Compares<U, CMPMODE::LT>(cmpMask, indicesReg, maxIndicesValue, maskReg);
        MicroAPI::Select(indicesReg, indicesReg, invalidIndicesReg, cmpMask);
        if constexpr (isCastIndices && IsSameType<U, int32_t>::value) {
            maskRegTwo = MicroAPI::UpdateMask<int64_t, MicroAPI::RegTraitNumTwo>(sizeTwo);
            MicroAPI::Cast<int64_t, int32_t, castTraitB32B64>(indicesCastReg, indicesReg, maskRegTwo);
            MicroAPI::Muls(indicesCastReg, indicesCastReg, totalColByte, maskRegTwo);
            MicroAPI::Adds(indicesCastReg, indicesCastReg, blockColOffsetByte, maskRegTwo);
            MicroAPI::DataCopy(posCastAddr + i * vfLen, indicesCastReg, maskRegTwo);
        } else {
            MicroAPI::Muls(indicesReg, indicesReg, static_cast<U>(totalColByte), maskReg);
            MicroAPI::Adds(indicesReg, indicesReg, static_cast<U>(blockColOffsetByte), maskReg);
            MicroAPI::DataCopy(posAddr + i * vfLen, indicesReg, maskReg);
        }
    }
}

template <typename U, bool isCastIndices>
__simd_vf__ inline void ComputeOffsetVFUnAlign(__ubuf__ U* posAddr, __ubuf__ int64_t* posCastAddr, uint32_t vfLen,
                                               uint16_t loopCnt, uint32_t size, uint32_t sizeTwo,
                                               int64_t blockColOffsetByte, int64_t totalColByte,
                                               int64_t maxIndicesValue)
{
    MicroAPI::RegTensor<U> indicesReg;
    MicroAPI::RegTensor<U> invalidIndicesReg;
    MicroAPI::RegTensor<int64_t, MicroAPI::RegTraitNumTwo> indicesCastReg;
    MicroAPI::MaskReg maskReg;
    MicroAPI::MaskReg maskRegTwo;
    MicroAPI::MaskReg cmpMask;

    AscendC::MicroAPI::UnalignReg u0;

    MicroAPI::Duplicate(invalidIndicesReg, static_cast<U>(-1));
    for (uint16_t i = 0; i < loopCnt; i++) {
        maskReg = MicroAPI::UpdateMask<U>(size);
        AscendC::MicroAPI::DataCopyUnAlignPre(u0, posAddr + i * vfLen + 1);
        AscendC::MicroAPI::DataCopyUnAlign<U>(indicesReg, u0, posAddr + i * vfLen + 1);
        MicroAPI::Compares<U, CMPMODE::LT>(cmpMask, indicesReg, maxIndicesValue, maskReg);
        MicroAPI::Select(indicesReg, indicesReg, invalidIndicesReg, cmpMask);
        if constexpr (isCastIndices && IsSameType<U, int32_t>::value) {
            maskRegTwo = MicroAPI::UpdateMask<int64_t, MicroAPI::RegTraitNumTwo>(sizeTwo);
            MicroAPI::Cast<int64_t, int32_t, castTraitB32B64>(indicesCastReg, indicesReg, maskRegTwo);
            MicroAPI::Muls(indicesCastReg, indicesCastReg, totalColByte, maskRegTwo);
            MicroAPI::Adds(indicesCastReg, indicesCastReg, blockColOffsetByte, maskRegTwo);
            MicroAPI::DataCopy(posCastAddr + i * vfLen, indicesCastReg, maskRegTwo);
        } else {
            MicroAPI::Muls(indicesReg, indicesReg, static_cast<U>(totalColByte), maskReg);
            MicroAPI::Adds(indicesReg, indicesReg, static_cast<U>(blockColOffsetByte), maskReg);
            MicroAPI::DataCopy(posAddr + i * vfLen, indicesReg, maskReg);
        }
    }
}

template <typename T, typename U, bool withPos, bool isCastIndices>
template <bool isGatherDma>
__aicore__ inline void ScatterAddWithSortedSimdDMA<T, U, withPos, isCastIndices>::ComputeOffsetForDMA(
    LocalTensor<U> posLocal, uint32_t rows, int64_t maxIndicesValue, bool unAlign)
{
    __local_mem__ U* posAddr = (__local_mem__ U*)posLocal.GetPhyAddr();
    __local_mem__ int64_t* posCastAddr = nullptr;
    if constexpr (isCastIndices) {
        if constexpr (isGatherDma) {
            LocalTensor<int64_t> posCastLocal = posCastBuf_.Get<int64_t>();
            posCastAddr = (__local_mem__ int64_t*)posCastLocal.GetPhyAddr();
        } else {
            LocalTensor<int64_t> posCastLocal = indicesCastBuf_.Get<int64_t>();
            posCastAddr = (__local_mem__ int64_t*)posCastLocal.GetPhyAddr();
        }
    }

    uint32_t vfLen = platform::GetVRegSize() / sizeof(U);
    uint16_t loopCnt = ops::CeilDiv(static_cast<uint32_t>(rows), vfLen);
    uint32_t size = rows;
    uint32_t sizeTwo = rows;
    int64_t blockColOffsetByte = static_cast<int64_t>(colCoreIdx_ * tilingData_.normBlockCol * sizeof(T));
    int64_t totalColByte = tilingData_.varShape[1] * sizeof(T);

    if (unAlign) {
        ComputeOffsetVFUnAlign<U, isCastIndices>(posAddr, posCastAddr, vfLen, loopCnt, size, sizeTwo,
                                                 blockColOffsetByte, totalColByte, maxIndicesValue);
    } else {
        ComputeOffsetVF<U, isCastIndices>(posAddr, posCastAddr, vfLen, loopCnt, size, sizeTwo, blockColOffsetByte,
                                          totalColByte, maxIndicesValue);
    }
}

template <typename T, typename U, bool withPos, bool isCastIndices>
template <typename OFFSET_T>
__aicore__ inline void ScatterAddWithSortedSimdDMA<T, U, withPos, isCastIndices>::GatherDMA(
    LocalTensor<OFFSET_T> offsetLocal, LocalTensor<T> dstLocal, uint16_t offsetNum, uint32_t burstLen,
    uint32_t dstStride)
{
    event_t eventIdV_MTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(eventIdV_MTE2);
    WaitFlag<HardEvent::V_MTE2>(eventIdV_MTE2);

    GatherParams gatherDataCopyParams;
    gatherDataCopyParams.offsetNum = offsetNum;
    gatherDataCopyParams.burstLen = burstLen * sizeof(T);
    gatherDataCopyParams.dstStride = dstStride * sizeof(T);

    GatherPadParams<T> gatherPadParams{
        GatherPaddingMode::CONSTANT_PADDING_MODE,
        static_cast<T>(0),
    };

    GatherDataCopy(dstLocal, updatesGm_, offsetLocal, gatherDataCopyParams, gatherPadParams);

    updatesQueue_.EnQue<T>(dstLocal);

    //
    event_t eventIdMTE2_V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMTE2_V);
    WaitFlag<HardEvent::MTE2_V>(eventIdMTE2_V);
}

template <typename T, typename U, bool withPos, bool isCastIndices>
template <typename OFFSET_T>
__aicore__ inline void ScatterAddWithSortedSimdDMA<T, U, withPos, isCastIndices>::ScatterAddDMA(
    LocalTensor<T> srcLocal, LocalTensor<OFFSET_T> offsetLocal, uint16_t offsetNum, uint32_t burstLen,
    uint32_t srcStride)
{
    SetAtomicAdd<T>();
    ScatterParams scatterDataCopyParams;
    scatterDataCopyParams.offsetNum = offsetNum;
    scatterDataCopyParams.burstLen = burstLen * sizeof(T);
    scatterDataCopyParams.srcStride = srcStride * sizeof(T);

    event_t eventIdV_MTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdV_MTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIdV_MTE3);

    ScatterDataCopy(varRefGm_, srcLocal, offsetLocal, scatterDataCopyParams);
    SetAtomicNone();

    //
    event_t eventIdMTE3_V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eventIdMTE3_V);
    WaitFlag<HardEvent::MTE3_V>(eventIdMTE3_V);
}

template <typename T, typename U, bool withPos, bool isCastIndices>
__aicore__ inline void ScatterAddWithSortedSimdDMA<T, U, withPos, isCastIndices>::ProcessPos(uint64_t indicesGmOffset,
                                                                                             uint32_t cols,
                                                                                             uint32_t colsAlign,
                                                                                             uint32_t rows)
{
    if constexpr (withPos) {
        LocalTensor<U> posLocal = posQueue_.AllocTensor<U>();
        CopyIn(posLocal, posGm_, indicesGmOffset, 1, rows);
        posQueue_.EnQue<U>(posLocal);
        posLocal = posQueue_.DeQue<U>();
        ComputeOffsetForDMA<true>(posLocal, rows, tilingData_.indicesNum);

        // GatherDMA
        LocalTensor<T> updatesLocal = updatesQueue_.AllocTensor<T>();
        if constexpr (isCastIndices) {
            LocalTensor<int64_t> posCastLocal = posCastBuf_.Get<int64_t>();
            GatherDMA<int64_t>(posCastLocal, updatesLocal, static_cast<uint16_t>(rows), cols, colsAlign);
        } else {
            GatherDMA<U>(posLocal, updatesLocal, static_cast<uint16_t>(rows), cols, colsAlign);
        }
        posQueue_.FreeTensor(posLocal);
    }
}

template <typename T, typename U, bool withPos, bool isCastIndices>
__aicore__ inline uint32_t ScatterAddWithSortedSimdDMA<T, U, withPos, isCastIndices>::ProcessIndices(
    uint64_t indicesGmOffset, uint64_t rowLoop, uint32_t rows, uint64_t rowUbLoopNum)
{
    int64_t shiftOffset = UB_AGLIN_VALUE / sizeof(U); // 计算索引重复度需要偏移一个block
    LocalTensor<U> indicesLocal = indicesQueue_.AllocTensor<U>();
    // 末轮不读下一轮首id，多拷1个会在末核最后一轮越界读indices末尾，故收敛为rows
    uint32_t copyLen = (rowLoop == rowUbLoopNum - 1) ? rows : rows + 1;
    CopyIn(indicesLocal[shiftOffset], indicesGm_, indicesGmOffset, 1, copyLen);
    indicesQueue_.EnQue<U>(indicesLocal);

    indicesLocal = indicesQueue_.DeQue<U>();
    LocalTensor<int32_t> uniqueIdCountLocal = uniqueIdCountQue_.Get<int32_t>();

    Duplicate(indicesLocal, (U)-1, shiftOffset);
    event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIDVToS);
    WaitFlag<HardEvent::V_S>(eventIDVToS);

    if (rowLoop != rowUbLoopNum - 1 && rowLoop > 0) {
        curLoopLastId_ = indicesLocal.GetValue(shiftOffset + rows - 1);
        nextLoopFirstId_ = indicesLocal.GetValue(shiftOffset + rows);
    }
    if (rowLoop == 0) {
        headIndex_ = indicesLocal.GetValue(shiftOffset);
        if (!isStartRowCore_ && headIndex_ == prevCoreLastIndex_) {
            headToWorkspace_ = true;
        }
    }

    if (rowLoop == rowUbLoopNum - 1) {
        tailIndex_ = indicesLocal.GetValue(shiftOffset + rows - 1);
        if (!isEndRowCore_ && tailIndex_ == nextCoreFirstIndex_) {
            tailToWorkspace_ = true;
        }
    }
    indicesLocal(shiftOffset + rows) = -1;
    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);

    LocalTensor<U> updateSumIdxLocal = updateSumIdxQue_.Get<U>();
    uint32_t uniqueIdNum = ComputeUniqueIdNum(
        indicesLocal, uniqueIdCountLocal,
        rows); // 此时 uniqueIdCountLocal 存放排好序的 非重复的 indices的索引 (0, 3, 5)

    __local_mem__ U* indicesLocalAddr = (__local_mem__ U*)indicesLocal.GetPhyAddr() + shiftOffset;
    __local_mem__ uint32_t* uniqueIdCountLocalAddr = (__local_mem__ uint32_t*)uniqueIdCountLocal.GetPhyAddr();
    __local_mem__ U* updateSumIdxLocalAddr = (__local_mem__ U*)updateSumIdxLocal.GetPhyAddr();

    uint16_t loopCnt = (uniqueIdNum + VFLEN_INT32 - 1) / VFLEN_INT32;
    uint32_t sizeTwo = uniqueIdNum;
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg maskRegTwo;
        MicroAPI::RegTensor<int64_t, MicroAPI::RegTraitNumTwo> gatherRegB64;
        AscendC::MicroAPI::RegTensor<uint32_t> idxReg;
        AscendC::MicroAPI::RegTensor<U> gatherReg;
        for (uint16_t i = 0; i < loopCnt; ++i) {
            AscendC::MicroAPI::MaskReg maskReg = AscendC::MicroAPI::UpdateMask<int32_t>(uniqueIdNum);
            auto idxAddr = uniqueIdCountLocalAddr + i * VFLEN_INT32;
            auto dstAddr = updateSumIdxLocalAddr + i * VFLEN_INT32;
            AscendC::MicroAPI::DataCopy(idxReg, idxAddr);
            if constexpr (IsSameType<U, int64_t>::value) {
                maskRegTwo = MicroAPI::UpdateMask<int64_t, MicroAPI::RegTraitNumTwo>(sizeTwo);
                MicroAPI::DataCopyGather(gatherRegB64, indicesLocalAddr, idxReg, maskRegTwo);
                AscendC::MicroAPI::DataCopy(dstAddr, gatherRegB64, maskRegTwo);
            } else {
                MicroAPI::DataCopyGather(gatherReg, indicesLocalAddr, idxReg, maskReg);
                AscendC::MicroAPI::DataCopy(dstAddr, gatherReg, maskReg);
            }
        }
    }

    ComputeUniqueIdTimes(uniqueIdCountLocal, uniqueIdNum); // 计算每个indices的重复个数存放于 uniqueIdCountLocal (3,2)
    indicesQueue_.FreeTensor(indicesLocal);

    return uniqueIdNum;
}

template <typename UPDATE_T>
__aicore__ inline void ProcessUpdateTensor(__local_mem__ UPDATE_T* updatesLocalBaseAddr,
                                           __local_mem__ UPDATE_T* resLocalAddr, MicroAPI::MaskReg& maskRegUpdate,
                                           MicroAPI::AddrReg& addrReg, uint16_t idRepeatTimes, uint32_t colsAlign)
{
    MicroAPI::RegTensor<UPDATE_T> addReg;
    MicroAPI::RegTensor<UPDATE_T> outReg;
    MicroAPI::MaskReg maskReg = MicroAPI::CreateMask<UPDATE_T, MicroAPI::MaskPattern::ALL>();
    MicroAPI::Duplicate(outReg, (UPDATE_T)0, maskReg);

    for (uint16_t pIdx = 0; pIdx < idRepeatTimes; ++pIdx) {
        auto startAddr = updatesLocalBaseAddr + pIdx * colsAlign;
        MicroAPI::DataCopy(addReg, startAddr, addrReg);
        MicroAPI::Add(outReg, outReg, addReg, maskRegUpdate);
    }
    MicroAPI::DataCopy(resLocalAddr, outReg, addrReg, maskRegUpdate);
}

template <typename T, typename U, bool withPos, bool isCastIndices>
__aicore__ inline void ScatterAddWithSortedSimdDMA<T, U, withPos, isCastIndices>::ComputeUpdatesSum(
    uint64_t cols, uint64_t colsAlign, uint32_t uniqueIdNum)
{
    LocalTensor<int32_t> uniqueIdCountLocal = uniqueIdCountQue_.Get<int32_t>();
    LocalTensor<T> updatesLocal = updatesQueue_.DeQue<T>();
    LocalTensor<T> resLocal = outQueueRes_.AllocTensor<T>();

    __local_mem__ int32_t* uniqueIdCountAddr = (__local_mem__ int32_t*)uniqueIdCountLocal.GetPhyAddr();
    __local_mem__ T* updatesLocalAddr = (__local_mem__ T*)updatesLocal.GetPhyAddr();
    __local_mem__ T* resLocalAddr = (__local_mem__ T*)resLocal.GetPhyAddr();
    __local_mem__ T* resLocalBaseAddr = resLocalAddr;
    __local_mem__ T* updatesLocalBaseAddr = updatesLocalAddr;

    uint32_t loopPerRow = (cols + vfLenUpdate_ - 1) / vfLenUpdate_;

    uint32_t updateRowIdx = 0;

    __VEC_SCOPE__
    {
        for (uint16_t i = 0; i < (uint16_t)uniqueIdNum; ++i) {
            uint16_t idRepeatTimes = static_cast<uint16_t>(uniqueIdCountAddr[i]); //
            uint32_t colCount = cols;
            resLocalAddr = resLocalBaseAddr + i * colsAlign;
            updatesLocalBaseAddr = updatesLocalAddr + updateRowIdx * colsAlign;
            for (uint16_t j = 0; j < (uint16_t)loopPerRow; ++j) {
                MicroAPI::MaskReg maskRegUpdate = MicroAPI::UpdateMask<T>(colCount);
                auto addrReg = MicroAPI::CreateAddrReg<T>(j, static_cast<uint16_t>(vfLenUpdate_));

                ProcessUpdateTensor<T>(updatesLocalBaseAddr, resLocalAddr, maskRegUpdate, addrReg, idRepeatTimes,
                                       colsAlign);
            }
            updateRowIdx = updateRowIdx + idRepeatTimes;
        }
    }
    outQueueRes_.EnQue<T>(resLocal);

    updatesQueue_.FreeTensor<T>(updatesLocal);
}

template <typename T, typename U, bool withPos, bool isCastIndices>
__aicore__ inline void ScatterAddWithSortedSimdDMA<T, U, withPos, isCastIndices>::CopyOutToWorkspace(
    LocalTensor<T> dataLocal, int32_t burstLen, int32_t writePosition)
{
    DataCopyExtParams ext;
    ext.blockCount = 1;
    ext.blockLen = static_cast<uint32_t>(burstLen) * sizeof(T);
    ext.srcStride = 0;
    ext.dstStride = 0;

    uint64_t workspaceOffset = static_cast<uint64_t>(writePosition) * (tilingData_.vecAlignSize / sizeof(T));
    DataCopyPad(sumWorkspace_[workspaceOffset], dataLocal, ext);
}

template <typename T, typename U, bool withPos, bool isCastIndices>
__aicore__ inline void ScatterAddWithSortedSimdDMA<T, U, withPos, isCastIndices>::CopyOutIndexToWorkspace(
    LocalTensor<U>& tmpLocal)
{
    DataCopyExtParams ext;
    ext.blockCount = 1;
    ext.blockLen = kDouble * sizeof(U);
    ext.srcStride = 0;
    ext.dstStride = 0;
    DataCopyPad(indexWorkspace_, tmpLocal, ext);
}

template <typename T, typename U, bool withPos, bool isCastIndices>
__aicore__ inline void ScatterAddWithSortedSimdDMA<T, U, withPos, isCastIndices>::Compute()
{
    if (!isStartRowCore_) {
        LocalTensor<U> tmp = lastCoreLastIndexQueue_.AllocTensor<U>();
        CopyIn(tmp, indicesGm_, static_cast<uint64_t>(rowCoreIdx_) * tilingData_.normBlockRow - 1, 1, 1);

        event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(ev);
        WaitFlag<HardEvent::MTE2_S>(ev);

        prevCoreLastIndex_ = tmp.GetValue(0);
        lastCoreLastIndexQueue_.FreeTensor(tmp);
    } else {
        prevCoreLastIndex_ = static_cast<U>(-1);
    }

    if (!isEndRowCore_) {
        LocalTensor<U> tmp = nextCoreFirstIndexQueue_.AllocTensor<U>();
        CopyIn(tmp, indicesGm_, static_cast<uint64_t>(rowCoreIdx_ + 1) * tilingData_.normBlockRow, 1, 1);

        event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(ev);
        WaitFlag<HardEvent::MTE2_S>(ev);

        nextCoreFirstIndex_ = tmp.GetValue(0);
        nextCoreFirstIndexQueue_.FreeTensor(tmp);
    } else {
        nextCoreFirstIndex_ = static_cast<U>(-1);
    }

    uint64_t curCoreRows = rowCoreIdx_ != (tilingData_.rowTileNum - 1) ? tilingData_.normBlockRow :
                                                                         tilingData_.tailBlockRow; // 当前分块行数
    uint64_t curCoreCols = colCoreIdx_ != (tilingData_.colTileNum - 1) ? tilingData_.normBlockCol :
                                                                         tilingData_.tailBlockCol; // 当前分块列数

    uint64_t blockOffsetindices = rowCoreIdx_ * tilingData_.normBlockRow; // 当前indices的偏移位置
    uint64_t blockOffsetUpdate = rowCoreIdx_ * tilingData_.normBlockRow * tilingData_.varShape[1] +
                                 colCoreIdx_ * tilingData_.normBlockCol; // 当前updates块的偏移位置

    uint64_t rowUbLoopNum = ops::CeilDiv(
        curCoreRows, static_cast<uint64_t>(tilingData_.ubFactorRow)); // 当前分块在行方向需要UB循环多少次

    uint64_t cols = curCoreCols;
    uint64_t colsAlign = ops::Aligned(static_cast<uint64_t>(cols * sizeof(T)), static_cast<uint64_t>(ONE_BLOCK_SIZE)) /
                         sizeof(T);

    for (uint64_t rowLoop = 0; rowLoop < rowUbLoopNum; rowLoop++) {
        bool unalign = false;
        uint64_t rows = (rowLoop == rowUbLoopNum - 1) ? (curCoreRows - rowLoop * tilingData_.ubFactorRow) :
                                                        tilingData_.ubFactorRow;

        uint64_t indicesGmOffset = blockOffsetindices + rowLoop * tilingData_.ubFactorRow;
        ProcessPos(indicesGmOffset, cols, colsAlign, rows);
        uint32_t uniqueIdNum = ProcessIndices(indicesGmOffset, rowLoop, rows, rowUbLoopNum);

        if constexpr (!withPos) {
            // moveAlign
            LocalTensor<T> updatesLocal = updatesQueue_.AllocTensor<T>();
            uint64_t offset = blockOffsetUpdate + rowLoop * tilingData_.ubFactorRow * tilingData_.varShape[1];
            CopyIn(updatesLocal, updatesGm_, offset, rows, cols, tilingData_.varShape[1] - cols);
            updatesQueue_.EnQue<T>(updatesLocal);
        }

        ComputeUpdatesSum(cols, colsAlign, uniqueIdNum);
        LocalTensor<T> baseResLocal = outQueueRes_.DeQue<T>();
        if (saveLastId_) { // 上次循环最后一个id加到当前第一行
            LocalTensor<T> tmpOneRowLocal = tmpBufOneRow_.Get<T>();
            Add(baseResLocal, baseResLocal, tmpOneRowLocal, cols);
        }
        LocalTensor<T> resLocal = baseResLocal;

        saveLastId_ = false;
        if (curLoopLastId_ == nextLoopFirstId_) {
            saveLastId_ = true;
        }

        if (rowUbLoopNum == 1 && (headToWorkspace_ || tailToWorkspace_)) {
            if (headIndex_ == tailIndex_) {                 // 只有一个唯一索引
                if (headToWorkspace_ && tailToWorkspace_) { // 首尾都需要ws，放到 pos1
                    KernelUtil::WaitVToMte3();
                    CopyOutToWorkspace(resLocal, cols, 1);
                    outQueueRes_.FreeTensor(baseResLocal);
                    break;
                } else if (headToWorkspace_) {
                    KernelUtil::WaitVToMte3();
                    CopyOutToWorkspace(baseResLocal, cols, 0); // 搬第一行到 pos0
                    outQueueRes_.FreeTensor(baseResLocal);
                    break;
                } else if (tailToWorkspace_) {
                    KernelUtil::WaitVToMte3();
                    CopyOutToWorkspace(baseResLocal[(uniqueIdNum - 1) * colsAlign], cols, 1); // 搬最后一行到 pos1
                    outQueueRes_.FreeTensor(baseResLocal);
                    break;
                }
            } else {                                        // 有多个唯一索引
                if (headToWorkspace_ && tailToWorkspace_) { // 核首尾id都需要ws
                    KernelUtil::WaitVToMte3();
                    CopyOutToWorkspace(baseResLocal, cols, 0);                                // 搬第一行到 pos0
                    CopyOutToWorkspace(baseResLocal[(uniqueIdNum - 1) * colsAlign], cols, 1); // 搬最后一行到 pos1
                    uniqueIdNum = uniqueIdNum - 2;
                    if (uniqueIdNum == 0) {
                        outQueueRes_.FreeTensor(baseResLocal);
                        break;
                    }
                    resLocal = baseResLocal[colsAlign]; // 更新 resLocal 和 updateSumIdxLocal
                    unalign = true;
                } else if (headToWorkspace_) { // 核首id需要ws
                    KernelUtil::WaitVToMte3();
                    CopyOutToWorkspace(baseResLocal, cols, 0); // 搬第一行到 pos0
                    resLocal = baseResLocal[colsAlign];
                    unalign = true;
                    uniqueIdNum = uniqueIdNum - 1;
                } else if (tailToWorkspace_) { // 核尾id需要ws
                    KernelUtil::WaitVToMte3();
                    CopyOutToWorkspace(baseResLocal[(uniqueIdNum - 1) * colsAlign], cols, 1); // 搬最后一行到 pos1
                    uniqueIdNum = uniqueIdNum - 1;
                }
            }
        } else if (rowLoop == 0) {  // 第一次ub循环
            if (headToWorkspace_) { // 核首行需要ws
                KernelUtil::WaitVToMte3();
                CopyOutToWorkspace(baseResLocal, cols, 0); // 搬第一行到 pos0
                uniqueIdNum = uniqueIdNum - 1;
                if (uniqueIdNum == 0) {
                    outQueueRes_.FreeTensor(baseResLocal);
                    continue;
                }
                resLocal = baseResLocal[colsAlign];
                unalign = true;
            }

        } else if (rowLoop == rowUbLoopNum - 1) { // 最后一次ub循环且核尾行需要ws
            if (tailToWorkspace_) {
                KernelUtil::WaitVToMte3();
                CopyOutToWorkspace(baseResLocal[(uniqueIdNum - 1) * colsAlign], cols, 1); // 搬最后一行到 pos1
                uniqueIdNum = uniqueIdNum - 1;
                if (uniqueIdNum == 0) {
                    outQueueRes_.FreeTensor(baseResLocal);
                    break;
                }
            }
        } else { // 0 < rowLoop < rowUbLoopNum - 1
            if (saveLastId_) {
                LocalTensor<T> tmpOneRowLocal = tmpBufOneRow_.Get<T>();
                Copy(tmpOneRowLocal, baseResLocal[(uniqueIdNum - 1) * colsAlign], cols);
                uniqueIdNum = uniqueIdNum - 1;
                if (uniqueIdNum == 0) {
                    outQueueRes_.FreeTensor(baseResLocal);
                    continue;
                }
            }
        }

        LocalTensor<U> updateSumIdxLocal = updateSumIdxQue_.Get<U>();
        ComputeOffsetForDMA<false>(updateSumIdxLocal, uniqueIdNum, tilingData_.varShape[0], unalign);

        if constexpr (isCastIndices) {
            LocalTensor<int64_t> indicesCastLocal = indicesCastBuf_.Get<int64_t>();
            ScatterAddDMA<int64_t>(resLocal, indicesCastLocal, static_cast<uint16_t>(uniqueIdNum), cols, colsAlign);
        } else {
            ScatterAddDMA<U>(resLocal, updateSumIdxLocal, static_cast<uint16_t>(uniqueIdNum), cols, colsAlign);
        }
        outQueueRes_.FreeTensor(baseResLocal);
    }
    LocalTensor<U> tmpIndicesLocal = tmpBuf_.Get<U>();
    U headFinalIndex = headToWorkspace_ ? headIndex_ : static_cast<U>(-1);
    U tailFinalIndex = tailToWorkspace_ ? tailIndex_ : static_cast<U>(-1);

    if (headToWorkspace_ && tailToWorkspace_ && headIndex_ == tailIndex_ && rowUbLoopNum == 1) {
        headFinalIndex = static_cast<U>(-1);
    }

    tmpIndicesLocal.SetValue(0, headFinalIndex);
    tmpIndicesLocal.SetValue(1, tailFinalIndex);

    event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(ev);
    WaitFlag<HardEvent::S_MTE3>(ev);

    CopyOutIndexToWorkspace(tmpIndicesLocal);
}

template <typename T, typename U, bool withPos, bool isCastIndices>
__aicore__ inline void ScatterAddWithSortedSimdDMA<T, U, withPos, isCastIndices>::Process()
{
    if (GetBlockIdx() >= GetBlockNum()) {
        return;
    }

    Compute();
}

} // namespace ScatterAddWithSorted
#endif // SCATTER_ADD_WITH_SORTED_SIMD_DMA_H
