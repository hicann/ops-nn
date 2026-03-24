/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
/*!
 * \file scatter_add_simd_sort_support_atomicadd.h
 * \brief simd kernel of scatter_add
 */

#ifndef SCATTER_ADD_SIMD_SORT_SUPPORT_ATOMICADD_H
#define SCATTER_ADD_SIMD_SORT_SUPPORT_ATOMICADD_H

#include "scatter_add_common.h"

namespace ScatterAdd {
using namespace AscendC;
using namespace ScatterAddCommon;

constexpr uint32_t SORT_PADDING = 64;
constexpr AscendC::MicroAPI::CastTrait castTraitU32U16 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::NO_SAT,
    AscendC::MicroAPI::MaskMergeMode::ZEROING
};

template<typename T, typename U, typename CAST_T, bool updatesIsScalar, uint32_t castType, uint32_t scatterOp>
class ScatterAddSIMDSortSupportAtomicAdd {
public:
    __aicore__ inline ScatterAddSIMDSortSupportAtomicAdd(const ScatterAddTilingData& tilingData, TPipe& pipe) :
        tilingData_(tilingData), pipe_(pipe) {};
    __aicore__ inline void Init(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR varRef, GM_ADDR workspace);
    __aicore__ inline void Compute();
    __aicore__ inline void Process();
    __aicore__ inline uint32_t ProcessIndices(uint64_t blockOffsetindices, uint64_t rowLoop, uint32_t rows);
    __aicore__ inline void ComputeUpdatesSum(uint64_t cols, uint64_t colsAlign, uint32_t uniqueIdNum);
    __aicore__ inline void CopyResToGm(uint32_t cols, uint32_t colsAlign, uint64_t ubOffset, uint32_t& uniqueIdNum);
    __aicore__ inline void ProcessPerUpdateScalar(
        __local_mem__ T* resLocalAddr, MicroAPI::MaskReg& maskRegUpdate, MicroAPI::AddrReg& addrReg,
        updateAddParams& params, T updateScalarValue);

    template <typename VGatherIndexDType>
    __aicore__ inline void ComputeUpdatesSumRegbase(uint64_t cols, uint64_t colsAlign, uint32_t uniqueIdNum);
    template <typename VGatherIndexDType, typename VGatherIndexDTypeInt>
    __aicore__ inline void ProcessPerUpdateGroup(
        __local_mem__ T* updatesLocalAddr, __local_mem__ T* resLocalAddr, MicroAPI::MaskReg& maskRegUpdate,
        MicroAPI::RegTensor<VGatherIndexDTypeInt>& serReg, MicroAPI::AddrReg& addrReg, updateAddParams& params);

private:
    AscendC::GlobalTensor<T> varGm_;
    AscendC::GlobalTensor<U> indicesGm_;
    AscendC::GlobalTensor<T> updatesGm_;
    AscendC::GlobalTensor<T> varRefGm_;

    TQue<QuePosition::VECIN, 1> updatesQueue_;
    TQue<QuePosition::VECIN, 1> indicesQueue_;
    TQue<QuePosition::VECOUT, 1> outQueueRes_;
    TBuf<QuePosition::VECCALC> castIndicesQue_;
    TBuf<QuePosition::VECCALC> sortIndicesQue_;
    TBuf<QuePosition::VECCALC> updatesOriginIdexQue_;
    TBuf<QuePosition::VECCALC> uniqueIdCountQue_;
    
    TPipe& pipe_;
    const ScatterAddTilingData& tilingData_;

    static constexpr uint32_t shiftOffset_ = platform::GetUbBlockSize() / sizeof(CAST_T);
    static constexpr uint32_t vfLenUpdate_ = VECTOR_LENGTH / sizeof(T);
};

template<typename T, typename U, typename CAST_T, bool updatesIsScalar, uint32_t castType, uint32_t scatterOp>
__aicore__ inline void ScatterAddSIMDSortSupportAtomicAdd<T, U, CAST_T, updatesIsScalar, castType, scatterOp>::Init(
                           GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR varRef, GM_ADDR workspace)
{
    varRefGm_.SetGlobalBuffer((__gm__ T*)(varRef));
    indicesGm_.SetGlobalBuffer((__gm__ U*)(indices));
    updatesGm_.SetGlobalBuffer((__gm__ T*)(updates));

    pipe_.InitBuffer(indicesQueue_, 1, ops::CeilAlign(tilingData_.ubFactorRow * sizeof(U), UB_AGLIN_VALUE));
    pipe_.InitBuffer(updatesQueue_, 1, ops::CeilAlign(tilingData_.ubFactorRow * tilingData_.ubFactorCol * sizeof(T), UB_AGLIN_VALUE));
    pipe_.InitBuffer(outQueueRes_, 1, tilingData_.ubFactorRow * tilingData_.ubFactorCol * sizeof(T));
    pipe_.InitBuffer(updatesOriginIdexQue_, ops::CeilAlign(tilingData_.ubFactorRow * sizeof(uint32_t), UB_AGLIN_VALUE));
    pipe_.InitBuffer(uniqueIdCountQue_, ops::CeilAlign(tilingData_.ubFactorRow * sizeof(int32_t), UB_AGLIN_VALUE) + SORT_PADDING);
    if constexpr (castType == CAST_0) {
        pipe_.InitBuffer(sortIndicesQue_, ops::CeilAlign(tilingData_.ubFactorRow * sizeof(U), UB_AGLIN_VALUE) + SORT_PADDING);
    } else {
        pipe_.InitBuffer(sortIndicesQue_, ops::CeilAlign(tilingData_.ubFactorRow * sizeof(CAST_T), UB_AGLIN_VALUE) + SORT_PADDING);
        pipe_.InitBuffer(castIndicesQue_, ops::CeilAlign(tilingData_.ubFactorRow * sizeof(CAST_T), UB_AGLIN_VALUE));
    }
}

template<typename T, typename U, typename CAST_T, bool updatesIsScalar, uint32_t castType, uint32_t scatterOp>
__aicore__ inline uint32_t ScatterAddSIMDSortSupportAtomicAdd<T, U, CAST_T, updatesIsScalar, castType, scatterOp>::ProcessIndices(
                                                           uint64_t blockOffsetindices, uint64_t rowLoop, uint32_t rows)
{
    LocalTensor<U> indicesLocal = indicesQueue_.AllocTensor<U>();
    CopyIn(indicesLocal, indicesGm_, blockOffsetindices + rowLoop * tilingData_.ubFactorRow, 1, rows);
    indicesQueue_.EnQue<U>(indicesLocal);

    indicesLocal = indicesQueue_.DeQue<U>();
    LocalTensor<uint32_t> updatesOriginIdxLocal = updatesOriginIdexQue_.Get<uint32_t>();
    LocalTensor<int32_t> uniqueIdCountLocal = uniqueIdCountQue_.Get<int32_t>();
    LocalTensor<CAST_T> indicesSortedLocal = sortIndicesQue_.Get<CAST_T>();

    uint32_t uniqueIdNum = 0;
    if constexpr (castType == CAST_0) {
        uniqueIdNum = SortAndComputeUniqueIdx<U>(rows, indicesLocal, indicesSortedLocal, uniqueIdCountLocal, updatesOriginIdxLocal);
    } else {
        LocalTensor<CAST_T> indicesCastLocal = castIndicesQue_.Get<CAST_T>();
        IndicesSortCast<U, CAST_T, castType>(indicesLocal, indicesCastLocal, uniqueIdCountLocal, rows);
        uniqueIdNum = SortAndComputeUniqueIdx<CAST_T>(
            rows, indicesCastLocal, indicesSortedLocal, uniqueIdCountLocal, updatesOriginIdxLocal);
    }
    
    ComputeUniqueIdTimes(uniqueIdCountLocal, uniqueIdNum); // 计算每个indices的重复度存放于uniqueIdCountLocal
    indicesQueue_.FreeTensor(indicesLocal);

    return uniqueIdNum;
}

template<typename T, typename U, typename CAST_T, bool updatesIsScalar, uint32_t castType, uint32_t scatterOp>
template <typename VGatherIndexDType, typename VGatherIndexDTypeInt>
__aicore__ inline void ScatterAddSIMDSortSupportAtomicAdd<T, U, CAST_T, updatesIsScalar, castType, scatterOp>::ProcessPerUpdateGroup(
    __local_mem__ T* updatesLocalAddr, __local_mem__ T* resLocalAddr, MicroAPI::MaskReg& maskRegUpdate,
    MicroAPI::RegTensor<VGatherIndexDTypeInt>& serReg, MicroAPI::AddrReg& addrReg, updateAddParams& params)
{
    if constexpr (IsSameType<VGatherIndexDType, uint16_t>::value) {
        MicroAPI::RegTensor<VGatherIndexDType> initIdsReg, idsReg;
        MicroAPI::RegTensor<VGatherIndexDType> initIdsReg1;
        MicroAPI::RegTensor<uint32_t> tmReg;
        MicroAPI::RegTensor<T> gatherOut;
        MicroAPI::RegTensor<int16_t> gatherOutInt16;  // 用于int8转int16
        MicroAPI::RegTensor<T> outReg;
        MicroAPI::MaskReg maskReg = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
        MicroAPI::MaskReg maskRegU32 = MicroAPI::CreateMask<uint32_t, MicroAPI::MaskPattern::ALL>();

        MicroAPI::Duplicate(outReg, (T)0, maskReg);
        for (uint16_t pIdx = 0; pIdx < params.segCount; ++pIdx) {
            MicroAPI::DataCopy<uint32_t, MicroAPI::LoadDist::DIST_BRC_B32>(tmReg, params.sortedIdxAddr + (params.outGmIndex + pIdx));
            MicroAPI::Cast<uint16_t, uint32_t, castTraitU32U16>((MicroAPI::RegTensor<uint16_t>&)tmReg, tmReg, maskRegU32);
            MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::HIGHEST>(
                (MicroAPI::RegTensor<uint16_t>&)initIdsReg, (MicroAPI::RegTensor<uint32_t>&)tmReg);
            MicroAPI::Pack<uint16_t, uint32_t, MicroAPI::HighLowPart::LOWEST>(
                (MicroAPI::RegTensor<uint16_t>&)initIdsReg1, (MicroAPI::RegTensor<uint32_t>&)tmReg);
            MicroAPI::Add(initIdsReg, initIdsReg, initIdsReg1, maskReg);
            MicroAPI::Muls(idsReg, initIdsReg, params.xPerRowNum, maskRegUpdate);
            MicroAPI::Add(idsReg, (MicroAPI::RegTensor<VGatherIndexDType>&)serReg, idsReg, maskRegUpdate);
            MicroAPI::DataCopyGather(gatherOut, updatesLocalAddr, idsReg, maskRegUpdate);
            MicroAPI::Add(outReg, outReg, gatherOut, maskRegUpdate);
        }

        MicroAPI::DataCopy(resLocalAddr, outReg, addrReg, maskRegUpdate);
    } else {
        MicroAPI::RegTensor<VGatherIndexDType> initIdsReg, idsReg;
        MicroAPI::RegTensor<T> gatherOut;
        MicroAPI::RegTensor<T> outReg;
        MicroAPI::MaskReg maskReg = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();

        MicroAPI::Duplicate(outReg, (T)0, maskReg);
        for (uint16_t pIdx = 0; pIdx < params.segCount; ++pIdx) {
            MicroAPI::DataCopy<uint32_t, MicroAPI::LoadDist::DIST_BRC_B32>(
                (MicroAPI::RegTensor<uint32_t>&)initIdsReg, params.sortedIdxAddr + (params.outGmIndex + pIdx));
            MicroAPI::Muls(idsReg, initIdsReg, params.xPerRowNum, maskRegUpdate);
            MicroAPI::Add(idsReg, (MicroAPI::RegTensor<VGatherIndexDType>&)serReg, idsReg, maskRegUpdate);
            MicroAPI::DataCopyGather(gatherOut, updatesLocalAddr, idsReg, maskRegUpdate);
            MicroAPI::Add(outReg, outReg, gatherOut, maskRegUpdate);
        }

        MicroAPI::DataCopy(resLocalAddr, outReg, addrReg, maskRegUpdate);
    }
}

template<typename T, typename U, typename CAST_T, bool updatesIsScalar, uint32_t castType, uint32_t scatterOp>
__aicore__ inline void ScatterAddSIMDSortSupportAtomicAdd<T, U, CAST_T, updatesIsScalar, castType, scatterOp>::ProcessPerUpdateScalar(
    __local_mem__ T* resLocalAddr, MicroAPI::MaskReg& maskRegUpdate, MicroAPI::AddrReg& addrReg,
    updateAddParams& params, T updateScalarValue)
{
    MicroAPI::RegTensor<T> addOut;
    MicroAPI::RegTensor<T> outReg;
    MicroAPI::MaskReg maskReg = MicroAPI::CreateMask<T, MicroAPI::MaskPattern::ALL>();
    MicroAPI::Duplicate(outReg, (T)0, maskReg);
    MicroAPI::Duplicate(addOut, updateScalarValue, maskReg);
    for (uint16_t pIdx = 0; pIdx < params.segCount; ++pIdx) {
        MicroAPI::Add(outReg, outReg, addOut, maskRegUpdate);
    }
    MicroAPI::DataCopy(resLocalAddr, outReg, addrReg, maskRegUpdate);
}

template<typename T, typename U, typename CAST_T, bool updatesIsScalar, uint32_t castType, uint32_t scatterOp>
template <typename VGatherIndexDType>
__aicore__ inline void ScatterAddSIMDSortSupportAtomicAdd<T, U, CAST_T, updatesIsScalar, castType, scatterOp>::ComputeUpdatesSumRegbase(
                                                               uint64_t cols, uint64_t colsAlign, uint32_t uniqueIdNum)
{
    LocalTensor<uint32_t> updatesOriginIdxLocal = updatesOriginIdexQue_.Get<uint32_t>();
    LocalTensor<int32_t> uniqueIdCountLocal = uniqueIdCountQue_.Get<int32_t>();
    LocalTensor<T> updatesLocal = updatesQueue_.DeQue<T>();
    LocalTensor<T> resLocal = outQueueRes_.AllocTensor<T>();

    __local_mem__ T* updatesLocalAddr = (__local_mem__ T*)updatesLocal.GetPhyAddr();
    __local_mem__ T* resLocalAddr = (__local_mem__ T*)resLocal.GetPhyAddr();
    __local_mem__ T* resLocalBaseAddr = resLocalAddr;

    int32_t sclar0 = 0;
    T updateScalarValue = updatesLocal(0);
    uint32_t loopPerRow = (cols + vfLenUpdate_ - 1) / vfLenUpdate_;
    updateAddParams params;
    params.xPerRowNum = colsAlign;
    params.outGmIndex = 0;
    params.sortedIdxAddr = (__local_mem__ uint32_t*)updatesOriginIdxLocal.GetPhyAddr();
    using VGatherIndexDTypeInt = std::conditional_t<std::is_same_v<VGatherIndexDType, uint16_t>, int16_t, int32_t>;

    __VEC_SCOPE__
    {
        for (uint16_t i = 0; i < (uint16_t)uniqueIdNum; ++i) {
            MicroAPI::RegTensor<VGatherIndexDTypeInt> serReg;
            MicroAPI::RegTensor<VGatherIndexDTypeInt> serRegBase;

            MicroAPI::Arange(serRegBase, (VGatherIndexDTypeInt)sclar0);

            params.segCount = static_cast<uint16_t>(uniqueIdCountLocal(i));
            uint32_t colCount = cols;
            resLocalAddr = resLocalBaseAddr + i * colsAlign;
            for (uint16_t j = 0; j < (uint16_t)loopPerRow; ++j) {
                MicroAPI::MaskReg maskRegUpdate = MicroAPI::UpdateMask<VGatherIndexDType>(colCount);
                auto addrReg = MicroAPI::CreateAddrReg<T>(j, static_cast<uint16_t>(vfLenUpdate_));
                if constexpr (updatesIsScalar) {
                    ProcessPerUpdateScalar(resLocalAddr, maskRegUpdate, addrReg, params, updateScalarValue);
                } else {
                    MicroAPI::Adds(serReg, serRegBase, (VGatherIndexDTypeInt)(vfLenUpdate_ * j), maskRegUpdate);
                    ProcessPerUpdateGroup<VGatherIndexDType, VGatherIndexDTypeInt>(
                        updatesLocalAddr, resLocalAddr, maskRegUpdate, serReg, addrReg, params);
                }
            }
            params.outGmIndex = params.outGmIndex + params.segCount;
        }
    }

    outQueueRes_.EnQue<T>(resLocal);
    if constexpr (updatesIsScalar) {
        updatesQueue_.EnQue<T>(updatesLocal);
    } else {
        updatesQueue_.FreeTensor<T>(updatesLocal);
    }
}

template<typename T, typename U, typename CAST_T, bool updatesIsScalar, uint32_t castType, uint32_t scatterOp>
__aicore__ inline void ScatterAddSIMDSortSupportAtomicAdd<T, U, CAST_T, updatesIsScalar, castType, scatterOp>::ComputeUpdatesSum(
                                                               uint64_t cols, uint64_t colsAlign, uint32_t uniqueIdNum)
{
    LocalTensor<uint32_t> updatesOriginIdxLocal = updatesOriginIdexQue_.Get<uint32_t>();
    LocalTensor<int32_t> uniqueIdCountLocal = uniqueIdCountQue_.Get<int32_t>();
    LocalTensor<T> updatesLocal = updatesQueue_.DeQue<T>();
    LocalTensor<T> resLocal = outQueueRes_.AllocTensor<T>();
    Duplicate(resLocal, static_cast<T>(0), static_cast<int32_t>(uniqueIdNum * colsAlign));
    outQueueRes_.EnQue<T>(resLocal);

    resLocal = outQueueRes_.DeQue<T>();
    uint32_t indicesOffset = 0;
    for (uint32_t i = 0; i < uniqueIdNum; i++) {
        uint32_t uniqueTimes = static_cast<uint32_t>(uniqueIdCountLocal(i));   // 重复次数
        for (uint32_t j = 0; j < uniqueTimes; j++) {
            if constexpr (updatesIsScalar) {
                Add(resLocal[i * colsAlign], resLocal[i * colsAlign], updatesLocal, cols);
            } else {
                uint32_t srcIdx = updatesOriginIdxLocal.GetValue(indicesOffset + j);
                Add(resLocal[i * colsAlign], resLocal[i * colsAlign], updatesLocal[srcIdx * colsAlign], cols);
            }
        }
        indicesOffset += uniqueTimes;
    }

    outQueueRes_.EnQue<T>(resLocal);
    if constexpr (updatesIsScalar) {
        updatesQueue_.EnQue<T>(updatesLocal);
    } else {
        updatesQueue_.FreeTensor<T>(updatesLocal);
    }
}

template<typename T, typename U, typename CAST_T, bool updatesIsScalar, uint32_t castType, uint32_t scatterOp>
__aicore__ inline void ScatterAddSIMDSortSupportAtomicAdd<T, U, CAST_T, updatesIsScalar, castType, scatterOp>::CopyResToGm(
                                     uint32_t cols, uint32_t colsAlign, uint64_t ubOffset, uint32_t& uniqueIdNum)
{
    LocalTensor<T> resLocal = outQueueRes_.DeQue<T>();
    LocalTensor<CAST_T> indicesSortedLocal = sortIndicesQue_.Get<CAST_T>();
    LocalTensor<int32_t> uniqueIdCountLocal = uniqueIdCountQue_.Get<int32_t>();
    if constexpr (scatterOp == SUB && !updatesIsScalar) {
        NegateUpdate<T>(resLocal, static_cast<uint32_t>(uniqueIdNum * colsAlign));
        auto MTE3WaitVEventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(MTE3WaitVEventID);
        WaitFlag<HardEvent::V_MTE3>(MTE3WaitVEventID);
    }

    int32_t tmpIndex = shiftOffset_;

    SetAtomicAdd<T>();
    for (uint32_t i = 0; i < uniqueIdNum; i++) {
        uint64_t dstIndices = indicesSortedLocal(tmpIndex);       // 获取每个可能重复的indices的第一个的值
        uint64_t offset = dstIndices * tilingData_.varShape[1] + ubOffset;  // 通过indices值获取对应var的偏移
        tmpIndex = tmpIndex + uniqueIdCountLocal(i);

        if (dstIndices < 0 || dstIndices >= tilingData_.varShape[0]) {
            continue;
        }

        CopyOut(varRefGm_, resLocal[i * colsAlign], offset, 1, cols);
    }
    SetAtomicNone();
    outQueueRes_.FreeTensor(resLocal);
}

template<typename T, typename U, typename CAST_T, bool updatesIsScalar, uint32_t castType, uint32_t scatterOp>
__aicore__ inline void ScatterAddSIMDSortSupportAtomicAdd<T, U, CAST_T, updatesIsScalar, castType, scatterOp>::Compute()
{
    if (GetBlockIdx() >= tilingData_.atomicAddCoreNum) {
        return;
    }

    if constexpr (updatesIsScalar) {
        LocalTensor<T> updatesLocal = updatesQueue_.AllocTensor<T>();
        BroadcastUpdatesScalar<T, scatterOp>(updatesLocal, updatesGm_, static_cast<int32_t>(tilingData_.ubFactorCol));
        updatesQueue_.EnQue<T>(updatesLocal);
    }
    uint64_t rowIdx = GetBlockIdx() / tilingData_.colTileNum;   // 当前block在第几个分块行
    uint64_t colIdx = GetBlockIdx() % tilingData_.colTileNum;   // 当前block在第几个分块列
    uint64_t curCoreRows = rowIdx != (tilingData_.rowTileNum - 1) ? tilingData_.normBlockRow : tilingData_.tailBlockRow;  // 当前分块行数
    uint64_t curCoreCols = colIdx != (tilingData_.colTileNum - 1) ? tilingData_.normBlockCol : tilingData_.tailBlockCol;  // 当前分块列数

    uint64_t blockOffsetindices = rowIdx * tilingData_.normBlockRow;   // 当前indices的偏移位置
    uint64_t blockOffsetUpdate = rowIdx * tilingData_.normBlockRow * tilingData_.varShape[1] + colIdx * tilingData_.normBlockCol;  // 当前updates块的偏移位置
    uint64_t colUbLoopNum = ops::CeilDiv(curCoreCols, tilingData_.ubFactorCol);   // 当前分块在列方向需要UB循环多少次
    uint64_t rowUbLoopNum = ops::CeilDiv(curCoreRows, tilingData_.ubFactorRow);   // 当前分块在行方向需要UB循环多少次

    for (uint64_t rowLoop = 0; rowLoop < rowUbLoopNum; rowLoop++) {
        uint64_t rows = (rowLoop == rowUbLoopNum - 1) ? (curCoreRows - rowLoop * tilingData_.ubFactorRow) : tilingData_.ubFactorRow;
        uint32_t uniqueIdNum = ProcessIndices(blockOffsetindices, rowLoop, rows);

        for (uint64_t colLoop = 0; colLoop < colUbLoopNum; colLoop++) {
            uint64_t cols = (colLoop == colUbLoopNum - 1) ? (curCoreCols - colLoop * tilingData_.ubFactorCol) : tilingData_.ubFactorCol;  // 当前UB拿到的updates小块的列数
            uint64_t colsAlign = ops::Aligned(static_cast<uint64_t>(cols * sizeof(T)), static_cast<uint64_t>(ONE_BLOCK_SIZE)) / sizeof(T);

            if constexpr (!updatesIsScalar) {
                LocalTensor<T> updatesLocal = updatesQueue_.AllocTensor<T>();
                uint64_t offset = blockOffsetUpdate + rowLoop * tilingData_.ubFactorRow * tilingData_.varShape[1] + colLoop * tilingData_.ubFactorCol;
                CopyIn(updatesLocal, updatesGm_, offset, rows, cols, tilingData_.varShape[1] - cols);
                updatesQueue_.EnQue<T>(updatesLocal);
            }
            if constexpr (IsSameType<T, int8_t>::value) {
                ComputeUpdatesSum(cols, colsAlign, uniqueIdNum);
            } else if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
                ComputeUpdatesSumRegbase<uint16_t>(cols, colsAlign, uniqueIdNum);
            } else {
                ComputeUpdatesSumRegbase<uint32_t>(cols, colsAlign, uniqueIdNum);
            }
            
            uint64_t ubOffset = colIdx * tilingData_.normBlockCol + colLoop * tilingData_.ubFactorCol;
            CopyResToGm(cols, colsAlign, ubOffset, uniqueIdNum);
        }
    }

    if constexpr (updatesIsScalar) {
        LocalTensor<T> updatesLocal = updatesQueue_.DeQue<T>();
        updatesQueue_.FreeTensor<T>(updatesLocal);
    }
}

template<typename T, typename U, typename CAST_T, bool updatesIsScalar, uint32_t castType, uint32_t scatterOp>
__aicore__ inline void ScatterAddSIMDSortSupportAtomicAdd<T, U, CAST_T, updatesIsScalar, castType, scatterOp>::Process()
{
    if (GetBlockIdx() >= GetBlockNum()) {
        return;
    }

    Compute();
}
}
#endif  // SCATTER_ADD_SIMD_SORT_SUPPORT_ATOMICADD_H