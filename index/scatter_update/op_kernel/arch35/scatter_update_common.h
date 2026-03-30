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
 * \file scatter_update_common.h
 * \brief scatter_update
 */
#ifndef ASCENDC_SCATTER_UPDATE_COMMON_H_
#define ASCENDC_SCATTER_UPDATE_COMMON_H_

#include "kernel_operator.h"
#include "../inc/platform.h"
#include "op_kernel/platform_util.h"
#include "indices_sort_utils.h"
#include "op_kernel/math_util.h"
#include "../inc/load_store_utils.h"
#include "scatter_update_struct.h"

namespace ScatterUpdate {
using namespace AscendC;

#ifdef __DAV_FPGA__
constexpr uint32_t THREAD_NUM_DETERMINISTIC = 256;
#else
constexpr uint32_t THREAD_NUM_DETERMINISTIC = 1024;
#endif

constexpr uint64_t UB_AGLIN_VALUE = 32;
constexpr uint64_t SORT_PAD_NUM = 2;
constexpr uint64_t HASH_BUCKER_BUFFER_SIZE = 128 * sizeof(float);
constexpr int64_t DOUBLE_BUFFER = 2;
constexpr float SORT_HIST_THRESHOLD = 0.01f;
constexpr uint32_t TWO = 2;
constexpr uint32_t THREE = 3;
constexpr uint32_t FOUR = 4;
constexpr uint32_t CAST_NOT_CAST = 0;
constexpr uint32_t CAST_INT32_TO_INT16 = 1;
constexpr uint32_t CAST_INT64_TO_INT32 = 2;
constexpr uint32_t CAST_INT64_TO_INT16 = 3;
constexpr uint32_t CAST_INT32_TO_UINT8 = 4;
constexpr uint32_t CAST_INT64_TO_UINT8 = 5;
constexpr int64_t VFLEN_INT64 = platform::GetVRegSize() / sizeof(int64_t);
constexpr int64_t VFLEN_INT32 = platform::GetVRegSize() / sizeof(int32_t);
constexpr int64_t VFLEN_INT16 = platform::GetVRegSize() / sizeof(int16_t);
constexpr int64_t VFLEN_INT16HALF = platform::GetVRegSize() / sizeof(int16_t) / TWO;
constexpr int64_t VFLEN_UINT8 = platform::GetVRegSize() / sizeof(uint8_t);
constexpr int64_t VFLEN_UINT8HALFHALF = platform::GetVRegSize() / sizeof(uint8_t) / FOUR;

constexpr SortConfig sortConfig{SortType::RADIX_SORT, false};

// todo：处理计数不同indices的核心逻辑
template<typename IDX_T>
__aicore__ inline void ComputeUniqueIdNumInt64(__local_mem__ IDX_T* indicesAddr, __local_mem__ int32_t* uniqueIdCountsAddr, uint16_t loopCnt, int64_t dataLen)
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
        AscendC::MicroAPI::GatherMask<int32_t, AscendC::MicroAPI::GatherMaskMode::STORE_REG>(selReg, orderReg, maskHalf);
        AscendC::MicroAPI::DataCopyUnAlign<int32_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
            uniqueIdCountsAddr, selReg, uOut);
    }
    AscendC::MicroAPI::DataCopyUnAlignPost(uniqueIdCountsAddr, uOut);
}

template<typename IDX_T>
__aicore__ inline void ComputeUniqueIdNumInt32(__local_mem__ IDX_T* indicesAddr, __local_mem__ int32_t* uniqueIdCountsAddr, uint16_t loopCnt, int64_t dataLen)
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

template<typename IDX_T>
__aicore__ inline void ComputeUniqueIdNumInt16(__local_mem__ IDX_T* indicesAddr, __local_mem__ int32_t* uniqueIdCountsAddr, uint16_t loopCnt, int64_t dataLen)
{
    uint32_t counter = dataLen + 1;
    AscendC::MicroAPI::RegTensor<int32_t> orderReg, orderReg2, selReg, selReg2;
    AscendC::MicroAPI::RegTensor<IDX_T> sortedIdxReg, sortedIdxShiftOneReg;
    AscendC::MicroAPI::MaskReg cmpMask, maskReg, maskDouble1, maskDouble2;
    AscendC::MicroAPI::UnalignReg u0, uOut;
    for (uint16_t i = 0; i < loopCnt; ++i) {
        AscendC::MicroAPI::Arange(orderReg, i * VFLEN_INT16);
        AscendC::MicroAPI::Arange(orderReg2, i * VFLEN_INT16 + VFLEN_INT16HALF);
        maskReg = AscendC::MicroAPI::UpdateMask<IDX_T>(counter);
        auto startAddr = indicesAddr + i * VFLEN_INT16;
        DataCopy(sortedIdxReg, startAddr);
        AscendC::MicroAPI::DataCopyUnAlignPre(u0, startAddr - 1);
        AscendC::MicroAPI::DataCopyUnAlign<IDX_T>(sortedIdxShiftOneReg, u0, startAddr - 1);
        AscendC::MicroAPI::Compare<IDX_T, CMPMODE::NE>(cmpMask, sortedIdxReg, sortedIdxShiftOneReg, maskReg);
        AscendC::MicroAPI::MaskUnPack<AscendC::MicroAPI::HighLowPart::LOWEST>(maskDouble1, cmpMask);
        AscendC::MicroAPI::MaskUnPack<AscendC::MicroAPI::HighLowPart::HIGHEST>(maskDouble2, cmpMask);
        AscendC::MicroAPI::GatherMask<int32_t, AscendC::MicroAPI::GatherMaskMode::STORE_REG>(selReg, orderReg, maskDouble1);
        AscendC::MicroAPI::GatherMask<int32_t, AscendC::MicroAPI::GatherMaskMode::STORE_REG>(selReg2, orderReg2, maskDouble2);
        AscendC::MicroAPI::DataCopyUnAlign<int32_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(uniqueIdCountsAddr, selReg, uOut);
        AscendC::MicroAPI::DataCopyUnAlign<int32_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(uniqueIdCountsAddr, selReg2, uOut);
    }
    AscendC::MicroAPI::DataCopyUnAlignPost(uniqueIdCountsAddr, uOut);
}

template<typename IDX_T>
__aicore__ inline void ComputeUniqueIdNumUint8(__local_mem__ IDX_T* indicesAddr, __local_mem__ int32_t* uniqueIdCountsAddr, uint16_t loopCnt, int64_t dataLen)
{
    uint32_t counter = dataLen + 1;
    AscendC::MicroAPI::RegTensor<int32_t> orderReg, orderReg2, orderReg3, orderReg4;
    AscendC::MicroAPI::RegTensor<int32_t> selReg, selReg2, selReg3, selReg4;
    AscendC::MicroAPI::RegTensor<IDX_T> sortedIdxReg, sortedIdxShiftOneReg;
    AscendC::MicroAPI::MaskReg cmpMask, maskReg, maskFour1, maskFour2, maskFour3, maskFour4;
    AscendC::MicroAPI::UnalignReg u0, uOut;
    for (uint16_t i = 0; i < loopCnt; ++i) {
        AscendC::MicroAPI::Arange(orderReg, i * VFLEN_UINT8);
        AscendC::MicroAPI::Arange(orderReg2, i * VFLEN_UINT8 + VFLEN_UINT8HALFHALF);
        AscendC::MicroAPI::Arange(orderReg3, i * VFLEN_UINT8 + VFLEN_UINT8HALFHALF * TWO);
        AscendC::MicroAPI::Arange(orderReg4, i * VFLEN_UINT8 + VFLEN_UINT8HALFHALF * THREE);
        maskReg = AscendC::MicroAPI::UpdateMask<IDX_T>(counter);
        auto startAddr = indicesAddr + i * VFLEN_UINT8;
        DataCopy(sortedIdxReg, startAddr);
        AscendC::MicroAPI::DataCopyUnAlignPre(u0, startAddr - 1);
        AscendC::MicroAPI::DataCopyUnAlign<IDX_T>(sortedIdxShiftOneReg, u0, startAddr - 1);
        AscendC::MicroAPI::Compare<IDX_T, CMPMODE::NE>(cmpMask, sortedIdxReg, sortedIdxShiftOneReg, maskReg);
        AscendC::MicroAPI::MaskUnPack<AscendC::MicroAPI::HighLowPart::LOWEST>(maskFour3, cmpMask);
        AscendC::MicroAPI::MaskUnPack<AscendC::MicroAPI::HighLowPart::HIGHEST>(maskFour4, cmpMask);
        AscendC::MicroAPI::MaskUnPack<AscendC::MicroAPI::HighLowPart::LOWEST>(maskFour1, maskFour3);
        AscendC::MicroAPI::MaskUnPack<AscendC::MicroAPI::HighLowPart::HIGHEST>(maskFour2, maskFour3);
        AscendC::MicroAPI::MaskUnPack<AscendC::MicroAPI::HighLowPart::LOWEST>(maskFour3, maskFour4);
        AscendC::MicroAPI::MaskUnPack<AscendC::MicroAPI::HighLowPart::HIGHEST>(maskFour4, maskFour4);
        AscendC::MicroAPI::GatherMask<int32_t, AscendC::MicroAPI::GatherMaskMode::STORE_REG>(selReg, orderReg, maskFour1);
        AscendC::MicroAPI::GatherMask<int32_t, AscendC::MicroAPI::GatherMaskMode::STORE_REG>(selReg2, orderReg2, maskFour2);
        AscendC::MicroAPI::GatherMask<int32_t, AscendC::MicroAPI::GatherMaskMode::STORE_REG>(selReg3, orderReg3, maskFour3);
        AscendC::MicroAPI::GatherMask<int32_t, AscendC::MicroAPI::GatherMaskMode::STORE_REG>(selReg4, orderReg4, maskFour4);
        AscendC::MicroAPI::DataCopyUnAlign<int32_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(uniqueIdCountsAddr, selReg, uOut);
        AscendC::MicroAPI::DataCopyUnAlign<int32_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(uniqueIdCountsAddr, selReg2, uOut);
        AscendC::MicroAPI::DataCopyUnAlign<int32_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(uniqueIdCountsAddr, selReg3, uOut);
        AscendC::MicroAPI::DataCopyUnAlign<int32_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(uniqueIdCountsAddr, selReg4, uOut);
    }
    AscendC::MicroAPI::DataCopyUnAlignPost(uniqueIdCountsAddr, uOut);
}

template<typename INX_T>
__aicore__ inline uint32_t ComputeUniqueIdNum(
    LocalTensor<INX_T> indicesLocal, LocalTensor<int32_t> uniqueIdCountLocal, int64_t dataLen)
{
    __local_mem__ INX_T* indicesAddr = (__local_mem__ INX_T*)indicesLocal[(UB_AGLIN_VALUE / sizeof(INX_T))].GetPhyAddr();
    __local_mem__ int32_t* uniqueIdCountsAddr = (__local_mem__ int32_t*)uniqueIdCountLocal.GetPhyAddr();

    int64_t vfLen = platform::GetVRegSize() / sizeof(INX_T);
    uint16_t loopCnt = ops::CeilDiv(dataLen + 1, vfLen);
    uint32_t counter = dataLen + 1;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::ClearSpr<AscendC::SpecialPurposeReg::AR>();

        if constexpr (std::is_same<int64_t, INX_T>::value) {
 	        ComputeUniqueIdNumInt64<INX_T>(indicesAddr, uniqueIdCountsAddr, loopCnt, dataLen);
 	    } else if constexpr (std::is_same<int32_t, INX_T>::value) {
 	        ComputeUniqueIdNumInt32<INX_T>(indicesAddr, uniqueIdCountsAddr, loopCnt, dataLen);
 	    } else if constexpr (std::is_same<int16_t, INX_T>::value) {
 	        ComputeUniqueIdNumInt16<INX_T>(indicesAddr, uniqueIdCountsAddr, loopCnt, dataLen);
 	    } else {  // uint8
 	        ComputeUniqueIdNumUint8<INX_T>(indicesAddr, uniqueIdCountsAddr, loopCnt, dataLen);
        }
    }
    uint32_t uniqueIdNum = ((AscendC::MicroAPI::GetSpr<AscendC::SpecialPurposeReg::AR>()) / sizeof(int32_t)) - 1;
    return uniqueIdNum;
}

template<typename INX_T>
__aicore__ inline uint32_t SortAndComputeUniqueIdx(int64_t rowLen, LocalTensor<INX_T> indicesSrcLocal, LocalTensor<INX_T> sortIndicesLocal, 
    LocalTensor<int32_t> uniqueIdCountLocal, LocalTensor<uint32_t> updatesOriginIdexLocal)
{
    event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIDVToS);
    WaitFlag<HardEvent::V_S>(eventIDVToS);
    int64_t shiftOffset = UB_AGLIN_VALUE / sizeof(INX_T);
    LocalTensor<INX_T> shiftSortLocal = sortIndicesLocal[shiftOffset];
    AscendC::Sort<INX_T, true, sortConfig>(
        shiftSortLocal, updatesOriginIdexLocal, indicesSrcLocal, static_cast<uint32_t>(rowLen));
    Duplicate(sortIndicesLocal, (INX_T)-1, shiftOffset);
    SetFlag<HardEvent::V_S>(eventIDVToS);
    WaitFlag<HardEvent::V_S>(eventIDVToS);
    shiftSortLocal(rowLen) = -1;
    
    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
    return ComputeUniqueIdNum(sortIndicesLocal, uniqueIdCountLocal, rowLen);
}

// todo：将数据cast后放在 indicesCastLocal中
template <typename IDX_T, typename CAST_T, uint32_t castType>
__aicore__ inline void IndicesSortCast(LocalTensor<IDX_T> indicesLocal, LocalTensor<CAST_T> indicesCastLocal,
                                                LocalTensor<int32_t> indicesCastTmpLocal, uint32_t indicesCount)
{
    if constexpr (castType == CAST_INT32_TO_UINT8) {  // int32 Cast uint8
        CompareScalar(indicesCastLocal, indicesLocal, static_cast<IDX_T>(0), CMPMODE::GE, indicesCount);
        Select(indicesLocal, indicesCastLocal, indicesLocal, static_cast<IDX_T>(255), SELMODE::VSEL_TENSOR_SCALAR_MODE, indicesCount);
        Cast<CAST_T, IDX_T>(indicesCastLocal, indicesLocal, RoundMode::CAST_NONE, indicesCount);
    } else if constexpr (castType == CAST_INT64_TO_INT16) {  // int64 Cast int16
        Cast<int32_t, IDX_T>(indicesCastTmpLocal, indicesLocal, RoundMode::CAST_NONE, indicesCount);
        Cast<CAST_T, int32_t>(indicesCastLocal, indicesCastTmpLocal, RoundMode::CAST_NONE, indicesCount);
    } else if constexpr (castType == CAST_INT64_TO_UINT8) {  // int64 Cast uint8
        CompareScalar(indicesCastLocal, indicesLocal, static_cast<IDX_T>(0), CMPMODE::GE, indicesCount);
        Select(indicesLocal, indicesCastLocal, indicesLocal, static_cast<IDX_T>(255), SELMODE::VSEL_TENSOR_SCALAR_MODE, indicesCount);
        Cast<int32_t, IDX_T>(indicesCastTmpLocal, indicesLocal, RoundMode::CAST_NONE, indicesCount);
        Cast<CAST_T, int32_t>(indicesCastLocal, indicesCastTmpLocal, RoundMode::CAST_NONE, indicesCount);
    } else {    // CAST_INT32_TO_INT16 + CAST_INT64_TO_INT32, int32 Cast int16 + int64 Cast int32
        Cast<CAST_T, IDX_T>(indicesCastLocal, indicesLocal, RoundMode::CAST_NONE, indicesCount);
    }
}

template<typename T, typename U, typename MASK_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM_DETERMINISTIC) inline void ScatterUpdateSimtCalcMaskUnSort(
    uint32_t indicesCount, uint64_t varFirstDimSize, uint64_t indicesStartGmOffset,
    __gm__ MASK_T* workspaceMaskAddr, __local_mem__ U* indicesLocalAddr)
{
    for (uint32_t i = Simt::GetThreadIdx(); i < indicesCount; i += Simt::GetThreadNum()) {
        U indicesValue = indicesLocalAddr[i];
        if (!(indicesValue >= 0 && indicesValue < varFirstDimSize)) {
            continue;
        }
        Simt::AtomicMax(workspaceMaskAddr + indicesValue, static_cast<MASK_T>(indicesStartGmOffset + i));
    }
}

template<typename T, typename U, typename MASK_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM_DETERMINISTIC) inline void ScatterUpdateSimtCalcMaskSort(
    uint32_t uniqueIdNum, uint64_t varFirstDimSize, uint64_t indicesStartGmOffset, __gm__ MASK_T* workspaceMaskAddr,
    __local_mem__ U* indicesSortedPtr, __local_mem__ uint32_t* updatesOriginIdxAddr, __local_mem__ int32_t* uniqueIdCountAddr)
{
    for (uint32_t i = Simt::GetThreadIdx(); i < uniqueIdNum; i += Simt::GetThreadNum()) {
        int32_t repeatTimes = uniqueIdCountAddr[i + 1] - uniqueIdCountAddr[i];
        int32_t lastIndicesIdx = uniqueIdCountAddr[i] + repeatTimes - 1;
        U indicesValue = indicesSortedPtr[lastIndicesIdx];
        if (!(indicesValue >= 0 && indicesValue < varFirstDimSize)) {
            continue;
        }

        uint32_t indicesLocalOffset = updatesOriginIdxAddr[lastIndicesIdx];
        Simt::AtomicMax(workspaceMaskAddr + indicesValue,
                        static_cast<MASK_T>(indicesStartGmOffset + indicesLocalOffset));
    }
}

template<typename T, typename U, typename MASK_T, bool splitCol, typename CAST_T, uint32_t castType>
class ScatterUpdateDeterministicCommon {
public:
    __aicore__ inline ScatterUpdateDeterministicCommon(const ScatterUpdateTilingData& tilingData, TPipe& pipe) : 
        tilingData_(tilingData), pipe_(pipe) {};
    __aicore__ inline void InitBase(GM_ADDR var, GM_ADDR indices, GM_ADDR updates);
    __aicore__ inline void InitSetBuffer(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR workspace);
    __aicore__ inline void CalcMask();
    __aicore__ inline void CopyInIndices(uint64_t indicesGmOffset, uint32_t indicesCount);

protected:
    AscendC::GlobalTensor<T> varGm_;
    AscendC::GlobalTensor<U> indicesGm_;
    AscendC::GlobalTensor<T> updatesGm_;
    AscendC::GlobalTensor<MASK_T> workspaceMask_;
    AscendC::GlobalTensor<MASK_T> workspaceMaskBlock_;
    TQue<QuePosition::VECIN, 1> indicesQue_;
    TBuf<QuePosition::VECCALC> castIndicesQue_;
    TBuf<QuePosition::VECCALC> sortIndicesQue_;
    TBuf<QuePosition::VECCALC> updatesOriginIdexQue_;
    TBuf<QuePosition::VECCALC> uniqueIdCountQue_;
    TBuf<QuePosition::VECCALC> hashBuffer_;
    
    TPipe& pipe_;
    const ScatterUpdateTilingData& tilingData_;

    uint32_t blockIdx_ = GetBlockIdx();
    uint64_t indicesBlockLoop_{0};
    uint64_t indicesTailLoopSize_{0};
    static constexpr uint32_t shiftOffset_ = platform::GetUbBlockSize() / sizeof(CAST_T);
    static constexpr MASK_T maskDefault_ = -1;
};

template<typename T, typename U, typename MASK_T, bool splitCol, typename CAST_T, uint32_t castType>
__aicore__ inline void ScatterUpdateDeterministicCommon<T, U, MASK_T, splitCol, CAST_T, castType>::InitBase(
                                     GM_ADDR var, GM_ADDR indices, GM_ADDR updates)
{
    pipe_.InitBuffer(indicesQue_, 1, ops::CeilAlign(tilingData_.indicesUbFactor * sizeof(U), UB_AGLIN_VALUE));
    varGm_.SetGlobalBuffer((__gm__ T *)(var));
    indicesGm_.SetGlobalBuffer((__gm__ U *)(indices));
    updatesGm_.SetGlobalBuffer((__gm__ T *)(updates));
}

template<typename T, typename U, typename MASK_T, bool splitCol, typename CAST_T, uint32_t castType>
__aicore__ inline void ScatterUpdateDeterministicCommon<T, U, MASK_T, splitCol, CAST_T, castType>::InitSetBuffer(
                                GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR workspace)
{
    indicesBlockLoop_ = tilingData_.normBlockLoop;
    indicesTailLoopSize_ = tilingData_.normBlockTail;
    if (blockIdx_ == tilingData_.usedCoreNum - 1) {
        indicesBlockLoop_ = tilingData_.tailBlockLoop;
        indicesTailLoopSize_ = tilingData_.tailBlockTail;
    }
    uint64_t maskBlockOffset = blockIdx_ * tilingData_.maskNormBlockLen;
    uint64_t maskBlockLen = blockIdx_ == tilingData_.usedCoreNum - 1 ? tilingData_.maskTailBlockLen : 
                                                                                   tilingData_.maskNormBlockLen;
    workspaceMask_.SetGlobalBuffer((__gm__ MASK_T *)(workspace));
    workspaceMaskBlock_.SetGlobalBuffer((__gm__ MASK_T *)(workspace) + maskBlockOffset);
    if constexpr (castType == CAST_NOT_CAST) {
        pipe_.InitBuffer(sortIndicesQue_, 
            ops::CeilAlign(tilingData_.indicesUbFactor * sizeof(U), UB_AGLIN_VALUE) + SORT_PAD_NUM * UB_AGLIN_VALUE);
    } else {
        pipe_.InitBuffer(sortIndicesQue_, 
            ops::CeilAlign(tilingData_.indicesUbFactor * sizeof(CAST_T), UB_AGLIN_VALUE) + SORT_PAD_NUM * UB_AGLIN_VALUE);
        pipe_.InitBuffer(castIndicesQue_, ops::CeilAlign(tilingData_.indicesUbFactor * sizeof(CAST_T), UB_AGLIN_VALUE));
    }
    pipe_.InitBuffer(updatesOriginIdexQue_, 
                           ops::CeilAlign(tilingData_.indicesUbFactor * sizeof(uint32_t), UB_AGLIN_VALUE));
    pipe_.InitBuffer(uniqueIdCountQue_, 
        ops::CeilAlign(tilingData_.indicesUbFactor * sizeof(int32_t), UB_AGLIN_VALUE) + SORT_PAD_NUM * UB_AGLIN_VALUE);
    pipe_.InitBuffer(hashBuffer_, HASH_BUCKER_BUFFER_SIZE);

    InitGlobalMemory(workspaceMaskBlock_, maskBlockLen, maskDefault_);
    SyncAll();
}

template<typename T, typename U, typename MASK_T, bool splitCol, typename CAST_T, uint32_t castType>
__aicore__ inline void ScatterUpdateDeterministicCommon<T, U, MASK_T, splitCol, CAST_T, castType>::CopyInIndices(
                                        uint64_t indicesGmOffset, uint32_t indicesCount)
{
    LocalTensor<U> indicesLocal = indicesQue_.AllocTensor<U>();

    DataCopyExtParams indicesCopyParams { 1, static_cast<uint32_t>(indicesCount * sizeof(U)), 0, 0, 0 };
    DataCopyPadExtParams<U> indicesPadParams { false, 0, 0, 0 };
    DataCopyPad(indicesLocal, indicesGm_[indicesGmOffset], indicesCopyParams, indicesPadParams);
    indicesQue_.EnQue<U>(indicesLocal);
}

template<typename T, typename U, typename MASK_T, bool splitCol, typename CAST_T, uint32_t castType>
__aicore__ inline void ScatterUpdateDeterministicCommon<T, U, MASK_T, splitCol, CAST_T, castType>::CalcMask()
{
    uint32_t indicesCount = tilingData_.indicesUbFactor;
    for (uint64_t idx = 0; idx < indicesBlockLoop_; idx++) {
        if (idx == indicesBlockLoop_ - 1) {
            indicesCount = indicesTailLoopSize_;
        }
        uint64_t indicesStartGmOffset = blockIdx_ * tilingData_.normBlockIndices + idx * tilingData_.indicesUbFactor;
        CopyInIndices(indicesStartGmOffset, indicesCount);

        LocalTensor<U> indicesLocal = indicesQue_.DeQue<U>();
        LocalTensor<float> hashLocal = hashBuffer_.Get<float>();
        float maxScore = 0.0f;
        if constexpr (IsSameType<U, int32_t>::value) {
            IndexStatisticInt32(indicesLocal, hashLocal, maxScore, indicesCount, tilingData_.varShape[1]);
        } else {
            IndexStatisticInt64(indicesLocal, hashLocal, maxScore, indicesCount, tilingData_.varShape[1]);
        }

        __gm__ MASK_T* workspaceMaskAddr = (__gm__ MASK_T*)(workspaceMask_.GetPhyAddr());
        uint64_t varFirstDimSize = tilingData_.varShape[0];

        if (maxScore > SORT_HIST_THRESHOLD) {
            LocalTensor<CAST_T> indicesSortedLocal = sortIndicesQue_.Get<CAST_T>();
            LocalTensor<uint32_t> updatesOriginIdxLocal = updatesOriginIdexQue_.Get<uint32_t>();
            LocalTensor<int32_t> uniqueIdCountLocal = uniqueIdCountQue_.Get<int32_t>();
            __local_mem__ CAST_T* indicesSortedPtr = (__local_mem__ CAST_T*)(indicesSortedLocal.GetPhyAddr()) + shiftOffset_;
            __local_mem__ uint32_t* updatesOriginIdxAddr = (__local_mem__ uint32_t*)(updatesOriginIdxLocal.GetPhyAddr());
            __local_mem__ int32_t* uniqueIdCountAddr = (__local_mem__ int32_t*)(uniqueIdCountLocal.GetPhyAddr());
            uint32_t uniqueIdNum = 0;
            if constexpr (castType == CAST_NOT_CAST) {
                uniqueIdNum = SortAndComputeUniqueIdx<U>(
                                indicesCount, indicesLocal, indicesSortedLocal, uniqueIdCountLocal, updatesOriginIdxLocal);
            } else {
                LocalTensor<CAST_T> indicesCastLocal = castIndicesQue_.Get<CAST_T>();
 	            IndicesSortCast<U, CAST_T, castType>(indicesLocal, indicesCastLocal, uniqueIdCountLocal, indicesCount);
 	            uniqueIdNum = SortAndComputeUniqueIdx<CAST_T>(
 	                            indicesCount, indicesCastLocal, indicesSortedLocal, uniqueIdCountLocal, updatesOriginIdxLocal);
            }

            Simt::VF_CALL<ScatterUpdateSimtCalcMaskSort<T, CAST_T, MASK_T>>(Simt::Dim3(THREAD_NUM_DETERMINISTIC), 
                    uniqueIdNum, varFirstDimSize, indicesStartGmOffset, workspaceMaskAddr, indicesSortedPtr,
                    updatesOriginIdxAddr, uniqueIdCountAddr);
        } else {
            __local_mem__ U* indicesLocalAddr = (__local_mem__ U*)(indicesLocal.GetPhyAddr());
            Simt::VF_CALL<ScatterUpdateSimtCalcMaskUnSort<T, U, MASK_T>>(Simt::Dim3(THREAD_NUM_DETERMINISTIC), 
                    indicesCount, varFirstDimSize, indicesStartGmOffset, workspaceMaskAddr, indicesLocalAddr);
        }

        indicesQue_.FreeTensor(indicesLocal);
    }
}

}  // namespace ScatterUpdate
#endif