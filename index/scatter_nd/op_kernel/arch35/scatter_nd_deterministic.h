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
 * \file scatter_nd_deterministic.h
 * \brief deterministic kernel of scatter_nd
 */

#ifndef SCATTER_ND_DETERMINISTIC_IMPL_H
#define SCATTER_ND_DETERMINISTIC_IMPL_H

#include "kernel_operator.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"
#include "../inc/load_store_utils.h"

namespace ScatterNdDeterministic {
using namespace AscendC;

constexpr uint64_t DOUBLE_BUF = 2;
constexpr uint64_t BLOCK_SIZE = 32;
constexpr uint32_t DOUBLE = 2;
constexpr uint64_t THREE = 3;
constexpr uint32_t SORT_STAT_PADDING = 64;
constexpr uint64_t UB_AGLIN_VALUE = 32;
constexpr uint16_t MAX_RANK_COUNT = 7;

static constexpr MicroAPI::CastTrait castTraitFP322INT32 = {MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::SAT,
                                                        MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

static constexpr MicroAPI::CastTrait castTraitINT322FP32 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT,
                                                        MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

static constexpr MicroAPI::CastTrait castTraitFP322T = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT,
                                                            MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

constexpr AscendC::MicroAPI::CastTrait castTraitB162B32 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

static constexpr SortConfig sortConfig{SortType::RADIX_SORT, false};

template<typename T, typename U>
class ScatterNdDeterministicImpl {
public:
    __aicore__ inline ScatterNdDeterministicImpl(const ScatterNdTilingData& tilingData, TPipe& pipe) :
        tilingData_(tilingData), pipe_(pipe) {};
    __aicore__ inline void Init(GM_ADDR indices, GM_ADDR updates, GM_ADDR y, GM_ADDR workspace);
    __aicore__ inline void InitWspZero();
    __aicore__ inline void CopyInUpdates(int64_t offset, int64_t dataLen);
    __aicore__ inline void CopyOutUpdates(int64_t offset, int64_t dataLen);
    __aicore__ inline void ComputOutOfset(const LocalTensor<U> indicesLocal, int32_t indicesLen, int32_t rankSize);    
    __aicore__ inline void ExcuteAtomicAdd(int64_t varRefOffset, int64_t dataLen);
    __aicore__ inline void ProcessSingleLoopIndices(int64_t indicesOffset, int64_t indicesLen);
    __aicore__ inline void ProcessAtomicAdd();
    __aicore__ inline void CopyInIndicesAndUpdates(uint64_t loopIdNum, uint64_t rowNums);
    __aicore__ inline void ComputeUniqueIdNum(LocalTensor<U>& sortedIndicesLocal, LocalTensor<U>& uniqueIdCountLocalU, uint32_t dataLen);
    __aicore__ inline void ComputeUinqueIdTimes(LocalTensor<U>& uniqueIdCountLocalU, uint32_t uniqueIdNum);
    __aicore__ inline void ComputeSum(LocalTensor<U>& uniqueIdCountLocalU, LocalTensor<uint32_t>& updatesOriginIdexLocal,
        uint32_t uniqueIdNum, uint32_t dataLen);
    __aicore__ inline void ComputeSumWithCast(LocalTensor<U>& uniqueIdCountLocalU, LocalTensor<uint32_t>& updatesOriginIdexLocal,
        uint32_t uniqueIdNum, uint32_t dataLen);
    __aicore__ inline void ProcessSortAndSum(uint64_t loopIdx, uint32_t dataLen);
    __aicore__ inline void CopySumOutToWs(uint64_t loopIdx, uint64_t dataLen);
    __aicore__ inline void CopySumAndIdxIn(uint64_t loopIdx, uint64_t dataCount);
    __aicore__ inline void ComputeRValueAndQuantize(uint64_t loopIdx, uint64_t dataCount);
    __aicore__ inline void QuantizeForSum(uint64_t curRowIdx, uint32_t RCountsValue, __local_mem__ float* updateSumAddr);
    __aicore__ inline void CopyQuantizedSumOutToIntWs(uint64_t RValueOffset);
    __aicore__ inline void CopyIntSumIn(uint64_t loopIdx, uint64_t dataCount);
    __aicore__ inline void CopyCoreLogicCoreSumIdIn(uint64_t loopIdx, uint64_t dataCount);
    __aicore__ inline void ComputeRValueAndDeQuantize(uint64_t loopIdx, uint64_t dataLen);
    __aicore__ inline void ComputeRValueAndDeQuantize_idxSplit(uint64_t loopIdx, uint64_t dataLen);
    __aicore__ inline void DeQuantizeForSum(uint64_t curRowIdx, uint32_t RCountsValue, __local_mem__ int32_t* updateSumIntAddr);
    __aicore__ inline void CopyOutDeQuantizedSum(uint64_t varOffset);
    __aicore__ inline void Process();

private:
    AscendC::GlobalTensor<T> varGm_;
    AscendC::GlobalTensor<U> indicesGm_;
    AscendC::GlobalTensor<T> updatesGm_;
    AscendC::GlobalTensor<T> yGm_;
    AscendC::GlobalTensor<T> yGmInit_;
    AscendC::GlobalTensor<float> workspaceMaxValue_;
    AscendC::GlobalTensor<float> workspaceLogicCoreSumValue_;
    AscendC::GlobalTensor<U> workspaceCoreLogicCoreSumId_;
    AscendC::GlobalTensor<uint32_t> workspaceMaxValueCount_;
    AscendC::GlobalTensor<int32_t> workspaceInt32Res_;
    AscendC::GlobalTensor<int32_t> workspaceInt32ResInit_;
    AscendC::GlobalTensor<float> workspaceMaxValueInit_;

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, DOUBLE_BUF> dataQueue_;
    TQue<QuePosition::VECIN, DOUBLE_BUF> updatesQueue_;
    TBuf<QuePosition::VECCALC> indicesBuf_;
    TBuf<QuePosition::VECCALC> outOfstBuf_;
    TBuf<QuePosition::VECCALC> calcBuf_;

    TQue<QuePosition::VECIN, DOUBLE_BUF> indicesQue_;
    TBuf<QuePosition::VECCALC> sortedIndicesQue_;
    TBuf<QuePosition::VECCALC> updatesOriginIdexQue_;
    TQue<QuePosition::VECIN, DOUBLE_BUF> updateSumInQue_;
    TQue<QuePosition::VECOUT, DOUBLE_BUF> updateSumQue_;
    TQue<QuePosition::VECOUT, DOUBLE_BUF> updateSumIdxQueue_;
    TQue<QuePosition::VECIN, DOUBLE_BUF> RValueQue_;
    TQue<QuePosition::VECIN, DOUBLE_BUF> RCountsQue_;
    TPipe& pipe_;
    const ScatterNdTilingData& tilingData_;

    bool isDeterministic{false};
    uint64_t shiftOffset_{0};
    uint32_t uniqueIdNum_{0};
    uint64_t indicesUbLoop_{0};
    uint64_t tailIndicesUbLength_{0};
    uint64_t updatesSumUbFactor_{0};
    uint64_t indicesUbFactor_{0};
    uint64_t postVarAlignSize_{0};
    uint64_t postVarAlignSizeFp32_{0};
    uint64_t strideList[MAX_RANK_COUNT];
    uint64_t initPerCore{0};
    uint64_t initCoreReal{0};

    using IndexRegType = typename std::conditional<IsSameType<U, int64_t>::value,
                         typename AscendC::MicroAPI::RegTensor<uint64_t, AscendC::MicroAPI::RegTraitNumTwo>,
                         typename AscendC::MicroAPI::RegTensor<uint32_t>>::type;  
    using InnerRegType = typename std::conditional<IsSameType<U, int64_t>::value,
                         typename AscendC::MicroAPI::RegTensor<int64_t, AscendC::MicroAPI::RegTraitNumTwo>,
                         typename AscendC::MicroAPI::RegTensor<int32_t>>::type;
};

template<typename T, typename U>
__aicore__ inline void ScatterNdDeterministicImpl<T, U>::Init(GM_ADDR indices, GM_ADDR updates,
    GM_ADDR y, GM_ADDR workspace)
{
    postVarAlignSize_ = ops::CeilAlign(tilingData_.afterAxis * sizeof(T), UB_AGLIN_VALUE) / sizeof(T);
    postVarAlignSizeFp32_ = ops::CeilAlign(tilingData_.afterAxis * sizeof(float), UB_AGLIN_VALUE) / sizeof(float);
    uint32_t coreNum = GetBlockNum();
    initPerCore = tilingData_.outputSize / coreNum;
    uint32_t initTailCore = tilingData_.outputSize - (coreNum - 1) * initPerCore;
    initCoreReal = GetBlockIdx() == (coreNum - 1) ? initTailCore : initPerCore;
    uint64_t yGmOffset = GetBlockIdx() * initPerCore;
    yGm_.SetGlobalBuffer((__gm__ T *)(y));
    yGmInit_.SetGlobalBuffer((__gm__ T *)(y) + yGmOffset);
    // 非量化分支
    if (tilingData_.isDeterministic == 0) {
        indicesGm_.SetGlobalBuffer((__gm__ U *)(indices));
        updatesGm_.SetGlobalBuffer((__gm__ T *)(updates) + GetBlockIdx() * tilingData_.perCoreHandleCol);

        InitGlobalMemory(yGmInit_, initCoreReal, (T)0);

        auto mteWaitMTE3EventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(mteWaitMTE3EventID);
        WaitFlag<HardEvent::MTE3_MTE2>(mteWaitMTE3EventID);

        pipe_.InitBuffer(dataQueue_, DOUBLE_BUF, tilingData_.updatesUbFactor * sizeof(T));
        pipe_.InitBuffer(indicesBuf_, tilingData_.indicesUbFactor * tilingData_.rankSize * sizeof(U));
        pipe_.InitBuffer(outOfstBuf_, tilingData_.indicesUbFactor * sizeof(U));
        pipe_.InitBuffer(calcBuf_, MAX_RANK_COUNT * sizeof(U));

        SyncAll();
        return;
    }
    // 量化分支
    shiftOffset_ = UB_AGLIN_VALUE / sizeof(U);
    indicesUbLoop_ = GetBlockIdx() == tilingData_.logicCoreNum - 1 ? tilingData_.tailCoreIndicesLoopSize : tilingData_.indicesLoopSize;
    pipe_.InitBuffer(indicesQue_, DOUBLE_BUF, ops::CeilAlign(tilingData_.indicesUbFactor * tilingData_.rankSize * sizeof(U), UB_AGLIN_VALUE));
    pipe_.InitBuffer(outOfstBuf_, tilingData_.indicesUbFactor * sizeof(U));
    pipe_.InitBuffer(calcBuf_, MAX_RANK_COUNT * sizeof(U));  
    pipe_.InitBuffer(sortedIndicesQue_, ops::CeilAlign(tilingData_.indicesUbFactor * sizeof(U), UB_AGLIN_VALUE) + SORT_STAT_PADDING);
    pipe_.InitBuffer(updatesOriginIdexQue_, ops::CeilAlign(tilingData_.indicesUbFactor * sizeof(uint32_t), UB_AGLIN_VALUE));
    pipe_.InitBuffer(updateSumIdxQueue_, DOUBLE_BUF, ops::CeilAlign(tilingData_.indicesUbFactor * sizeof(U), UB_AGLIN_VALUE));
    pipe_.InitBuffer(updatesQueue_, DOUBLE_BUF, tilingData_.indicesUbFactor * postVarAlignSize_ * sizeof(T));
    pipe_.InitBuffer(updateSumQue_, DOUBLE_BUF, tilingData_.indicesUbFactor * postVarAlignSizeFp32_ * sizeof(float));
    
    indicesGm_.SetGlobalBuffer((__gm__ U *)(indices) + GetBlockIdx() * tilingData_.perCoreHandleIndices * tilingData_.rankSize);
    updatesGm_.SetGlobalBuffer((__gm__ T *)(updates) + GetBlockIdx() * tilingData_.perCoreHandleIndices * tilingData_.afterAxis);

    workspaceMaxValueCount_.SetGlobalBuffer((__gm__ uint32_t *)(workspace));
    workspaceMaxValue_.SetGlobalBuffer((__gm__ float *)workspace + tilingData_.rankFusedAxis);
    workspaceMaxValueInit_.SetGlobalBuffer((__gm__ float *)workspace + tilingData_.rankFusedAxis + GetBlockIdx() * initPerCore);
    workspaceInt32Res_.SetGlobalBuffer((__gm__ int32_t *)workspace + tilingData_.rankFusedAxis +
        tilingData_.rankFusedAxis * tilingData_.afterAxis);
    workspaceInt32ResInit_.SetGlobalBuffer((__gm__ int32_t *)workspace + tilingData_.rankFusedAxis +
        tilingData_.rankFusedAxis * tilingData_.afterAxis + GetBlockIdx() * initPerCore);
    workspaceLogicCoreSumValue_.SetGlobalBuffer((__gm__ float*)workspace + tilingData_.rankFusedAxis + 
    tilingData_.rankFusedAxis * tilingData_.afterAxis * DOUBLE + GetBlockIdx() * tilingData_.perCoreHandleIndices * postVarAlignSizeFp32_);
    
    __gm__ float* workspaceCoreLogicCoreSumIdStart = (__gm__ float*)workspace + tilingData_.rankFusedAxis + 
                                                        tilingData_.rankFusedAxis * tilingData_.afterAxis * DOUBLE + 
                                                        tilingData_.logicCoreNum * tilingData_.perCoreHandleIndices * postVarAlignSizeFp32_;
    workspaceCoreLogicCoreSumId_.SetGlobalBuffer((__gm__ U*)workspaceCoreLogicCoreSumIdStart + GetBlockIdx() * tilingData_.perCoreHandleIndices);
}

template<typename T, typename U>
__aicore__ inline void ScatterNdDeterministicImpl<T,  U>::InitWspZero()
{
    InitGlobalMemory(workspaceCoreLogicCoreSumId_, tilingData_.perCoreHandleIndices, (U)(-1));
    InitGlobalMemory(workspaceLogicCoreSumValue_, tilingData_.perCoreHandleIndices * postVarAlignSizeFp32_, (float)(0));
    InitGlobalMemory(workspaceMaxValueCount_, tilingData_.rankFusedAxis, (uint32_t)(0));
    InitGlobalMemory(workspaceMaxValueInit_, initCoreReal, (float)(0));
    InitGlobalMemory(workspaceInt32ResInit_, initCoreReal, (int)(0));
}

template<typename T, typename U>
__aicore__ inline void ScatterNdDeterministicImpl<T,  U>::CopyInUpdates(int64_t offset, int64_t dataLen)
{
    DataCopyExtParams inParams = { static_cast<uint16_t>(1), static_cast<uint32_t>(dataLen * sizeof(T)),
                                   static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0) };
    DataCopyPadExtParams<T> padParams = { false, static_cast<uint8_t>(0), static_cast<uint8_t>(0), static_cast<T>(0) };
    LocalTensor<T> updatesLocal = dataQueue_.AllocTensor<T>();
    DataCopyPad(updatesLocal, updatesGm_[offset], inParams, padParams);
    dataQueue_.EnQue(updatesLocal);
}

template<typename T, typename U>
__aicore__ inline void ScatterNdDeterministicImpl<T,  U>::CopyOutUpdates(int64_t offset, int64_t dataLen)
{
    DataCopyExtParams outParams = { static_cast<uint16_t>(1), static_cast<uint32_t>(dataLen * sizeof(T)),
                                    static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0) };
    LocalTensor<T> updatesLocal = dataQueue_.DeQue<T>();
    SetAtomicAdd<T>();
    DataCopyPad(yGm_[offset], updatesLocal, outParams);
    SetAtomicNone();
    dataQueue_.FreeTensor(updatesLocal);
}

template <typename T, typename U>
__aicore__ inline void ScatterNdDeterministicImpl<T, U>::ComputOutOfset(
    const LocalTensor<U> indicesLocal, int32_t indicesLen, int32_t rankSize)
{
    LocalTensor<U> outOfstLocal = outOfstBuf_.Get<U>();
    LocalTensor<U> calcLocal = calcBuf_.Get<U>();

    __local_mem__ U* indicesLocalPtr = ((__local_mem__ U*)indicesLocal.GetPhyAddr());
    __local_mem__ U* outOfstLocalPtr = ((__local_mem__ U*)outOfstLocal.GetPhyAddr());

    uint32_t dataLen = indicesLen;
    constexpr uint32_t vfLen = platform::GetVRegSize() / sizeof(int32_t);
    uint32_t indicesLenTimes = ops::CeilDiv(dataLen, vfLen);
    uint16_t loopCnt = static_cast<uint16_t>(indicesLenTimes);
    uint16_t rankSizeLoops = static_cast<uint16_t>(rankSize);

    __VEC_SCOPE__
    {
        InnerRegType inReg;
        InnerRegType outReg;
        InnerRegType orderReg;
        IndexRegType indexReg;
        AscendC::MicroAPI::MaskReg pregLoop;

        for (uint16_t i = 0; i < loopCnt; i++) {
            if constexpr (IsSameType<U, int64_t>::value) {
                pregLoop = AscendC::MicroAPI::UpdateMask<U, AscendC::MicroAPI::RegTraitNumTwo>(dataLen);
            } else {
                pregLoop = AscendC::MicroAPI::UpdateMask<U>(dataLen);
            }
            AscendC::MicroAPI::Duplicate(outReg, 0, pregLoop);
            AscendC::MicroAPI::Arange(orderReg, i * vfLen);
            AscendC::MicroAPI::Muls(orderReg, orderReg, rankSize, pregLoop);
            for (uint16_t dim = 0; dim < rankSizeLoops; dim++) {
                U strideValue = calcLocal(dim);
                indexReg = (IndexRegType&)orderReg;
                AscendC::MicroAPI::DataCopyGather(inReg, indicesLocalPtr, indexReg, pregLoop);
                AscendC::MicroAPI::Muls(inReg, inReg, strideValue, pregLoop);
                AscendC::MicroAPI::Add(outReg, inReg, outReg, pregLoop);
                AscendC::MicroAPI::Adds(orderReg, orderReg, (U)(1), pregLoop);
            }
            auto outOfstAddr = outOfstLocalPtr + i * vfLen;
            AscendC::MicroAPI::DataCopy(outOfstAddr, outReg, pregLoop);
        }
    }
    return;
}

template<typename T, typename U>
__aicore__ inline void ScatterNdDeterministicImpl<T,  U>::ProcessSingleLoopIndices(int64_t indicesOffset,
    int64_t indicesLen)
{
    DataCopyExtParams inParams = { static_cast<uint16_t>(1), static_cast<uint32_t>(indicesLen * tilingData_.rankSize * sizeof(U)),
                                   static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0) };
    DataCopyPadExtParams<U> padParams = { false, static_cast<uint8_t>(0), static_cast<uint8_t>(0), static_cast<U>(0) };
    LocalTensor<U> indicesLocal = indicesBuf_.Get<U>();
    DataCopyPad(indicesLocal, indicesGm_[indicesOffset * tilingData_.rankSize], inParams, padParams);
    auto vectorWaitMTE2EventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(vectorWaitMTE2EventID);
    WaitFlag<HardEvent::MTE2_V>(vectorWaitMTE2EventID);
    int32_t rankSize = tilingData_.rankSize;
    ComputOutOfset(indicesLocal, indicesLen, rankSize);
    LocalTensor<U> outOfstLocal = outOfstBuf_.Get<U>();
    auto scalarWaitVectorEventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(scalarWaitVectorEventID);
    WaitFlag<HardEvent::V_S>(scalarWaitVectorEventID);

    for (int64_t i = 0; i < indicesLen; ++i) {
        int64_t varRefOffset = outOfstLocal(i) * tilingData_.afterAxis;
        int64_t updatesOffset = (indicesOffset + i) * tilingData_.afterAxis;
        int64_t updatesTailUbFactor = tilingData_.updatesTailUbFactor;
        int64_t updatesLoopSize = tilingData_.updatesLoopSize;
        if (GetBlockIdx() == tilingData_.logicCoreNum - 1) {
            updatesLoopSize = tilingData_.tailCoreColsLoopSize;
            updatesTailUbFactor = tilingData_.tailCoreColsTailUbFactor;
        }
        for (int64_t updatesLoopIdx = 0; updatesLoopIdx < updatesLoopSize - 1; ++updatesLoopIdx) {
            CopyInUpdates(updatesOffset, tilingData_.updatesUbFactor);
            updatesOffset += tilingData_.updatesUbFactor;
            CopyOutUpdates(varRefOffset, tilingData_.updatesUbFactor);
            varRefOffset += tilingData_.updatesUbFactor;
        }
        varRefOffset += GetBlockIdx() * tilingData_.perCoreHandleCol;
        CopyInUpdates(updatesOffset, updatesTailUbFactor);
        CopyOutUpdates(varRefOffset, updatesTailUbFactor);
    }
}

template<typename T, typename U>
__aicore__ inline void ScatterNdDeterministicImpl<T,  U>::ProcessAtomicAdd()
{
    if (GetBlockIdx() >= tilingData_.logicCoreNum) {
        return;
    }

    int64_t indicesTailUbFactor = tilingData_.indicesTailUbFactor;
    int64_t indicesLoopSize = tilingData_.indicesLoopSize;
    int64_t indicesOffset = 0;

    LocalTensor<U> calcLocal = calcBuf_.Get<U>();
    for (int32_t i = 0; i < MAX_RANK_COUNT; i++) {
        calcLocal(i) = tilingData_.strideList[i];
    }
    for (int64_t indicesLoopIdx = 0; indicesLoopIdx < indicesLoopSize - 1; ++indicesLoopIdx) {
        indicesOffset = indicesLoopIdx * tilingData_.indicesUbFactor;
        ProcessSingleLoopIndices(indicesOffset, tilingData_.indicesUbFactor);
    }
    indicesOffset = (indicesLoopSize - 1) * tilingData_.indicesUbFactor;
    ProcessSingleLoopIndices(indicesOffset, indicesTailUbFactor);
}

template<typename T, typename U>
__aicore__ inline void ScatterNdDeterministicImpl<T,  U>:: CopyInIndicesAndUpdates(uint64_t loopIdNum, uint64_t rowNums)
{
    uint64_t rowNumsTmp1 = std::is_same<half, T>::value ? rowNums : 1;
    uint64_t rowNumsTmp2 = std::is_same<half, T>::value ? 1 : rowNums;
    // copy indices
    uint64_t IdOffset = loopIdNum * tilingData_.indicesUbFactor * tilingData_.rankSize;
    LocalTensor<U> indicesLocal = indicesQue_.AllocTensor<U>();
    DataCopyExtParams idCopyParam { static_cast<uint16_t>(1), static_cast<uint32_t>(rowNums * tilingData_.rankSize * sizeof(U)),
                                    static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0) };
    DataCopyPadExtParams<U> idPadParam { false, static_cast<uint8_t>(0), static_cast<uint8_t>(0), static_cast<U>(0) };
    DataCopyPad(indicesLocal, indicesGm_[IdOffset], idCopyParam, idPadParam);
    indicesQue_.EnQue(indicesLocal);
    // copy updates
    uint64_t updatesOffset = loopIdNum * tilingData_.indicesUbFactor * tilingData_.afterAxis;
    LocalTensor<T> updatesLocal = updatesQueue_.AllocTensor<T>();
    DataCopyExtParams dataCopyParam { static_cast<uint16_t>(rowNumsTmp1), static_cast<uint32_t>(rowNumsTmp2 * tilingData_.afterAxis * sizeof(T)),
                                      static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0) };
    DataCopyPadExtParams<T> dataPadParam { false, static_cast<uint8_t>(0), static_cast<uint8_t>(0), static_cast<T>(0) };
    DataCopyPad(updatesLocal, updatesGm_[updatesOffset], dataCopyParam, dataPadParam);
    updatesQueue_.EnQue(updatesLocal);
}

template<typename T, typename U>
__aicore__ inline void ScatterNdDeterministicImpl<T,  U>::ComputeUniqueIdNum(LocalTensor<U>& sortedIndicesLocal, 
    LocalTensor<U>& uniqueIdCountLocalU, uint32_t dataLen)
{
    LocalTensor<U> updateSumIdxLocal =  updateSumIdxQueue_.AllocTensor<U>();
    LocalTensor<int32_t> uniqueIdCountLocal = uniqueIdCountLocalU.template ReinterpretCast<int32_t>();

    __local_mem__ U* sortedIndicesAddr = (__local_mem__ U*)sortedIndicesLocal[shiftOffset_].GetPhyAddr();
    __local_mem__ int32_t* uniqueIdCountsAddr = (__local_mem__ int32_t*)uniqueIdCountLocal.GetPhyAddr();
    __local_mem__ U* updateSumIdxAddr = (__local_mem__ U*)updateSumIdxLocal.GetPhyAddr();

    constexpr uint32_t vfLen = platform::GetVRegSize() / sizeof(U);
    uint16_t loopCnt = ops::CeilDiv(dataLen + 1, vfLen);
    uint32_t counter = dataLen + 1;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<int32_t> orderReg;
        AscendC::MicroAPI::RegTensor<U> sortedIdxReg;
        AscendC::MicroAPI::RegTensor<U> sortedIdxShiftOneReg;
        AscendC::MicroAPI::RegTensor<int32_t> selReg0;
        AscendC::MicroAPI::MaskReg cmpMask;
        AscendC::MicroAPI::MaskReg maskRegUpdate;
        AscendC::MicroAPI::UnalignReg u0;
        AscendC::MicroAPI::UnalignReg u1;
        AscendC::MicroAPI::UnalignReg ureg0;
        AscendC::MicroAPI::ClearSpr<AscendC::SpecialPurposeReg::AR>();

        for (uint16_t i = 0; i < loopCnt; ++i) {
            AscendC::MicroAPI::Arange(orderReg, i * vfLen);
            maskRegUpdate = AscendC::MicroAPI::UpdateMask<U>(counter);
            auto startAddr = sortedIndicesAddr + i * vfLen;
            DataCopy(sortedIdxReg, startAddr);
            AscendC::MicroAPI::DataCopyUnAlignPre(u1, startAddr - 1);
            AscendC::MicroAPI::DataCopyUnAlign<U>(sortedIdxShiftOneReg, u1, startAddr - 1);
        
            AscendC::MicroAPI::Compare<U, CMPMODE::NE>(cmpMask, sortedIdxReg, sortedIdxShiftOneReg, maskRegUpdate);
            if constexpr (std::is_same<int64_t, U>::value) {
                AscendC::MicroAPI::MaskReg maskHalf;
                AscendC::MicroAPI::MaskPack<AscendC::MicroAPI::HighLowPart::LOWEST>(maskHalf, cmpMask);
                AscendC::MicroAPI::GatherMask<int32_t, AscendC::MicroAPI::GatherMaskMode::STORE_REG>(selReg0, orderReg,
                    maskHalf);
            } else {
                AscendC::MicroAPI::GatherMask<int32_t, AscendC::MicroAPI::GatherMaskMode::STORE_REG>(selReg0, orderReg,
                    cmpMask);
            }
                                                                                            
            AscendC::MicroAPI::DataCopyUnAlign<int32_t, AscendC::MicroAPI::PostLiteral::POST_MODE_UPDATE>(
                                                uniqueIdCountsAddr, selReg0, ureg0);
        }
        AscendC::MicroAPI::DataCopyUnAlignPost(uniqueIdCountsAddr, ureg0);
    }
    uniqueIdNum_ = ((AscendC::MicroAPI::GetSpr<AscendC::SpecialPurposeReg::AR>()) / sizeof(int32_t)) - 1;
    event_t eventS2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventS2V);
    WaitFlag<HardEvent::V_S>(eventS2V);
    for (uint32_t ubIdx = 0; ubIdx < uniqueIdNum_; ++ubIdx) {
        updateSumIdxLocal(ubIdx) = sortedIndicesLocal(shiftOffset_ + uniqueIdCountLocal(ubIdx));
    }
    auto mte3WaitSEventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(mte3WaitSEventID);
    WaitFlag<HardEvent::S_MTE3>(mte3WaitSEventID);
    updateSumIdxQueue_.EnQue(updateSumIdxLocal);
}

template<typename T, typename U>
__aicore__ inline void ScatterNdDeterministicImpl<T,  U>::ComputeUinqueIdTimes(LocalTensor<U>& uniqueIdCountLocalU, uint32_t uniqueIdNum)
{
    LocalTensor<int32_t> uniqueIdCountLocal = uniqueIdCountLocalU.template ReinterpretCast<int32_t>();
    __local_mem__ int32_t* uniqueIdCountsAddr = (__local_mem__ int32_t*)uniqueIdCountLocal.GetPhyAddr();

    // compute repeated num of each id
    constexpr uint32_t vfLen = platform::GetVRegSize() / sizeof(uint32_t);
    uint16_t loopSize = ops::CeilDiv(uniqueIdNum, vfLen);
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<int32_t> preReg;
        AscendC::MicroAPI::RegTensor<int32_t> postReg;
        AscendC::MicroAPI::RegTensor<int32_t> subReg;
        AscendC::MicroAPI::MaskReg maskReg;
        AscendC::MicroAPI::UnalignReg uIn;
        AscendC::MicroAPI::UnalignReg uOut;
        AscendC::MicroAPI::UnalignReg uInShift;
        for (uint16_t i = 0; i < loopSize; ++i) {
            maskReg = AscendC::MicroAPI::UpdateMask<uint32_t>(uniqueIdNum);
            auto startAddr = uniqueIdCountsAddr + i * vfLen;
            auto startAddrOfstOne = startAddr + 1;
            DataCopy(preReg, startAddr);
            AscendC::MicroAPI::DataCopyUnAlignPre(uInShift, startAddrOfstOne);
            AscendC::MicroAPI::DataCopyUnAlign<int32_t>(postReg, uInShift, startAddrOfstOne, vfLen);
            AscendC::MicroAPI::Sub(subReg, postReg, preReg, maskReg);
            DataCopy(startAddr, subReg, maskReg);
        }
    }
}

template<typename T, typename U>
__aicore__ inline void ScatterNdDeterministicImpl<T,  U>::ComputeSumWithCast(LocalTensor<U>& uniqueIdCountLocalU, LocalTensor<uint32_t>& updatesOriginIdexLocal,
    uint32_t uniqueIdNum, uint32_t dataLen)
{
    LocalTensor<uint32_t> uniqueIdCountLocal = uniqueIdCountLocalU.template ReinterpretCast<uint32_t>();
    LocalTensor<T> updatesLocal = updatesQueue_.DeQue<T>();
    LocalTensor<float> updateSumLocal = updateSumQue_.AllocTensor<float>();
    Duplicate(updateSumLocal, float(0), dataLen * postVarAlignSizeFp32_);

    __local_mem__ T* updatesAddr = (__local_mem__ T*)updatesLocal.GetPhyAddr();
    __local_mem__ float* updateSumAddr = (__local_mem__ float*)updateSumLocal.GetPhyAddr();
    constexpr uint32_t vfLen = platform::GetVRegSize() / sizeof(float);
    uint16_t loopSize = ops::CeilDiv(static_cast<uint32_t>(tilingData_.afterAxis), vfLen);
    int32_t idLocation = 0;
    uint32_t vlSize = platform::GetVRegSize();
    int32_t maskLen = tilingData_.afterAxis;
    int64_t colLenAlignSize = ops::CeilAlign(tilingData_.afterAxis * sizeof(T), UB_AGLIN_VALUE) / sizeof(T);
    int64_t colLenAlignFp32 = ops::CeilAlign(tilingData_.afterAxis * sizeof(float), UB_AGLIN_VALUE) / sizeof(float);

    __VEC_SCOPE__
    {
        for (uint16_t i = 0; i < static_cast<uint16_t>(uniqueIdNum); i++) {
            AscendC::MicroAPI::RegTensor<float> sumReg;
            AscendC::MicroAPI::RegTensor<float> castReg;
            AscendC::MicroAPI::MaskReg maskReg;
            AscendC::MicroAPI::MaskReg zeroMask = AscendC::MicroAPI::CreateMask<float>();
            uint32_t maskLenth = static_cast<uint32_t>(maskLen);
            uint16_t idRepeatTimes = static_cast<uint16_t>(uniqueIdCountLocal(i));
            for (uint16_t j = 0; j < loopSize; ++j) {
                maskReg = AscendC::MicroAPI::UpdateMask<float>(maskLenth);
                AscendC::MicroAPI::Duplicate(sumReg, (float)0, zeroMask);
                for (uint16_t k = 0; k < idRepeatTimes; k++) {
                    auto updatesOffet = updatesOriginIdexLocal(idLocation + k) * colLenAlignSize + j * vfLen;
                    ops::LoadOneTensorForDtypeT<T>(updatesAddr, castReg, maskReg, updatesOffet);
                    AscendC::MicroAPI::Add(sumReg, sumReg, castReg, maskReg);
                }
                auto updateSumAddrOfstAddr = updateSumAddr + i * colLenAlignFp32 + j * vfLen;
                AscendC::MicroAPI::DataCopy(updateSumAddrOfstAddr, sumReg, maskReg);
            }
            idLocation += idRepeatTimes;
        }
    }
    updateSumQue_.EnQue(updateSumLocal);
    updatesQueue_.FreeTensor(updatesLocal);
}

template<typename T, typename U>
__aicore__ inline void ScatterNdDeterministicImpl<T,  U>::ComputeSum(LocalTensor<U>& uniqueIdCountLocalU, LocalTensor<uint32_t>& updatesOriginIdexLocal,
    uint32_t uniqueIdNum, uint32_t dataLen)
{
    LocalTensor<uint32_t> uniqueIdCountLocal = uniqueIdCountLocalU.template ReinterpretCast<uint32_t>();
    LocalTensor<T> updatesLocal = updatesQueue_.template DeQue<T>();
    LocalTensor<float> updateSumLocal = updateSumQue_.AllocTensor<float>();
    Duplicate(updateSumLocal, float(0), dataLen * postVarAlignSizeFp32_);

    __local_mem__ T* updatesAddr = (__local_mem__ T*)updatesLocal.GetPhyAddr();
    __local_mem__ float* updateSumAddr = (__local_mem__ float*)updateSumLocal.GetPhyAddr();

    constexpr uint32_t vfLen = platform::GetVRegSize() / sizeof(T);
    uint16_t loopSize = (tilingData_.afterAxis + vfLen - 1) / vfLen;
    int32_t idLocation = 0;
    uint32_t vlSize = platform::GetVRegSize();
    uint32_t maskLen = static_cast<uint32_t>((postVarAlignSizeFp32_ * sizeof(float) + vlSize - 1) / vlSize * vlSize / sizeof(float) * dataLen);
    __VEC_SCOPE__
    {
        for (uint16_t i = 0; i < static_cast<uint16_t>(uniqueIdNum); i++) {
            AscendC::MicroAPI::RegTensor<float> sumReg;
            AscendC::MicroAPI::RegTensor<T> tmpReg;
            AscendC::MicroAPI::MaskReg maskReg =
                AscendC::MicroAPI::CreateMask<int32_t, AscendC::MicroAPI::MaskPattern::ALL>();
            MicroAPI::UnalignReg u0;
            AscendC::MicroAPI::MaskReg maskReg1;
            uint16_t idRepeatTimes = static_cast<uint16_t>(uniqueIdCountLocal(i));
            for (uint16_t j = 0; j < loopSize; ++j) {
                maskReg1 = AscendC::MicroAPI::UpdateMask<int32_t>(maskLen);
                AscendC::MicroAPI::Duplicate(sumReg, (float)0, maskReg);
                for (uint16_t k = 0; k < idRepeatTimes; k++) {
                    auto updatesOffet = updatesOriginIdexLocal(idLocation + k) * tilingData_.afterAxis + j * vfLen;
                    auto startAddr = updatesAddr + updatesOffet;
                    AscendC::MicroAPI::DataCopyUnAlignPre(u0, startAddr);
                    AscendC::MicroAPI::DataCopyUnAlign<T>(tmpReg, u0, startAddr, vfLen);
                    AscendC::MicroAPI::Add(sumReg, sumReg, tmpReg, maskReg1);
                }
                auto sumOffset = i * postVarAlignSizeFp32_ + j * vfLen;
                auto curUpdateSumAddr = updateSumAddr + sumOffset;
                DataCopy(curUpdateSumAddr, sumReg, maskReg1);
            }
            idLocation += idRepeatTimes;
        }
    }
    updateSumQue_.EnQue(updateSumLocal);
    updatesQueue_.FreeTensor(updatesLocal);
}

template<typename T, typename U>
__aicore__ inline void ScatterNdDeterministicImpl<T,  U>::ProcessSortAndSum(uint64_t loopIdx, uint32_t dataLen)
{
    LocalTensor<U> indicesLocal = indicesQue_.DeQue<U>();
    LocalTensor<U> sortedIndicesLocal = sortedIndicesQue_.Get<U>();   
    
    ComputOutOfset(indicesLocal, dataLen, tilingData_.rankSize);
    LocalTensor<U> indicesOfset = outOfstBuf_.Get<U>();

    LocalTensor<uint32_t> updatesOriginIdexLocal = updatesOriginIdexQue_.Get<uint32_t>();   
    Duplicate(sortedIndicesLocal, static_cast<U>(-1), shiftOffset_ * DOUBLE + tilingData_.indicesUbFactor);
    LocalTensor<U> shiftSortLocal = sortedIndicesLocal[shiftOffset_];

    AscendC::Sort<U, false, sortConfig>(shiftSortLocal, updatesOriginIdexLocal, indicesOfset, dataLen);
    ComputeUniqueIdNum(sortedIndicesLocal, indicesOfset, dataLen);
    event_t eventV2S =static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventV2S);
    WaitFlag<HardEvent::S_V>(eventV2S);
    ComputeUinqueIdTimes(indicesOfset, uniqueIdNum_);
    event_t eventS2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventS2V);
    WaitFlag<HardEvent::V_S>(eventS2V);
    if constexpr (std::is_same<half, T>::value) {
        ComputeSumWithCast(indicesOfset, updatesOriginIdexLocal, uniqueIdNum_, dataLen);
    } else {
        ComputeSum(indicesOfset, updatesOriginIdexLocal, uniqueIdNum_, dataLen);
    }
    indicesQue_.FreeTensor(indicesLocal);
}

template<typename T, typename U>
__aicore__ inline void ScatterNdDeterministicImpl<T,  U>::CopySumOutToWs(uint64_t loopIdx, uint64_t dataLen)
{
    // updatesSum和updatesSumId整块搬出去
    uint32_t updatesWspOffset = loopIdx * tilingData_.indicesUbFactor * postVarAlignSizeFp32_;
    LocalTensor<float> updateSumLocal = updateSumQue_.DeQue<float>();
    LocalTensor<U> updateSumIdxLocal = updateSumIdxQueue_.DeQue<U>();
    DataCopyExtParams outParams = { static_cast<uint16_t>(1), static_cast<uint32_t>(uniqueIdNum_ * postVarAlignSizeFp32_ * sizeof(float)),
                                    static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0) };
    DataCopyPad(workspaceLogicCoreSumValue_[updatesWspOffset], updateSumLocal, outParams);
    outParams = { static_cast<uint16_t>(1), static_cast<uint32_t>(uniqueIdNum_ * sizeof(U)), 0, 0, 0 };
    DataCopyPad(workspaceCoreLogicCoreSumId_[loopIdx * tilingData_.indicesUbFactor], updateSumIdxLocal, outParams);
    auto vWaitMTEEventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(vWaitMTEEventID);
    WaitFlag<HardEvent::MTE3_V>(vWaitMTEEventID);
    PipeBarrier<PIPE_ALL>();

    // max逐行搬出, max计数加1
    Abs(updateSumLocal, updateSumLocal, uniqueIdNum_ * postVarAlignSizeFp32_);
    for (uint32_t i = 0; i < uniqueIdNum_; i++) {
        uint64_t maxWspOffset = static_cast<uint64_t>(updateSumIdxLocal(i) * tilingData_.afterAxis);
        if (updateSumIdxLocal(i) < 0 || updateSumIdxLocal(i) >= static_cast<U>(tilingData_.rankFusedAxis)) {
            continue;
        }
        outParams = {static_cast<uint16_t>(1), static_cast<uint32_t>(tilingData_.afterAxis * sizeof(float)), 0, 0, 0 };
        SetAtomicMax<float>();
        DataCopyPad(workspaceMaxValue_[maxWspOffset], updateSumLocal[i * postVarAlignSizeFp32_], outParams);
        SetAtomicNone();
        uint32_t maxCountsOffset = updateSumIdxLocal(i);
        AscendC::AtomicAdd((__gm__ uint32_t *)(workspaceMaxValueCount_[maxCountsOffset].GetPhyAddr()), uint32_t(1));
        DataCacheCleanAndInvalid<uint32_t, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_ALL>(workspaceMaxValueCount_[maxCountsOffset]);
    }
    updateSumQue_.FreeTensor(updateSumLocal);
    updateSumIdxQueue_.FreeTensor(updateSumIdxLocal);
}

template<typename T, typename U>
__aicore__ inline void ScatterNdDeterministicImpl<T,  U>::CopySumAndIdxIn(uint64_t loopIdx, uint64_t dataCount)
{
    LocalTensor<float> updateSumLocal = updateSumInQue_.AllocTensor<float>();
    LocalTensor<U> updateSumIdxLocal = updateSumIdxQueue_.AllocTensor<U>();
    

    uint64_t indicesUbOffset = loopIdx * tilingData_.indicesUbFactor;
    DataCopyExtParams dataCopyParam { static_cast<uint16_t>(1), static_cast<uint32_t>(dataCount * sizeof(U)),
                                      static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0) };
    DataCopyPadExtParams<U> dataPadParam { false, static_cast<uint8_t>(0), static_cast<uint8_t>(0), static_cast<U>(0) };
    DataCopyPad(updateSumIdxLocal, workspaceCoreLogicCoreSumId_[indicesUbOffset], dataCopyParam, dataPadParam);

    uint64_t updatesSumUboffset = loopIdx * tilingData_.indicesUbFactor * postVarAlignSizeFp32_;
    DataCopyExtParams dataCopy { static_cast<uint16_t>(1), static_cast<uint32_t>(dataCount * postVarAlignSizeFp32_ * sizeof(float)),
                                static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0) };
    DataCopyPadExtParams<float> dataPad { false, static_cast<uint8_t>(0), static_cast<uint8_t>(0), static_cast<float>(0) };
    DataCopyPad(updateSumLocal, workspaceLogicCoreSumValue_[updatesSumUboffset], dataCopy, dataPad);

    updateSumInQue_.EnQue(updateSumLocal);
    updateSumIdxQueue_.EnQue(updateSumIdxLocal);
}

template<typename T, typename U>
__aicore__ inline void ScatterNdDeterministicImpl<T,  U>::ComputeRValueAndQuantize(uint64_t loopIdx, uint64_t dataCount) 
{
    LocalTensor<float> updateSumLocal = updateSumInQue_.DeQue<float>();
    LocalTensor<U> updateSumIdxLocal = updateSumIdxQueue_.DeQue<U>();

    __local_mem__ float* updateSumAddr = (__local_mem__ float*)updateSumLocal.GetPhyAddr();
    auto sWaitMTEEventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(sWaitMTEEventID);
    WaitFlag<HardEvent::MTE2_S>(sWaitMTEEventID);

    for (int64_t i = 0; i < dataCount; i++) {
        if (updateSumIdxLocal(i) < 0 || updateSumIdxLocal(i) >= static_cast<U>(tilingData_.rankFusedAxis)) {
            continue;
        }
        uint64_t RValueOffset = updateSumIdxLocal(i) * tilingData_.afterAxis;
        DataCopyExtParams dataCopyParam { static_cast<uint16_t>(1), static_cast<uint32_t>(tilingData_.afterAxis * sizeof(float)),
                                         static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0) };
        DataCopyPadExtParams<float> dataPadParam { false, static_cast<uint8_t>(0), static_cast<uint8_t>(0), static_cast<float>(0) };
        event_t eventS2Mte = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
        SetFlag<HardEvent::S_MTE2>(eventS2Mte);
        WaitFlag<HardEvent::S_MTE2>(eventS2Mte);
        LocalTensor<float> updateRValueLocal = RValueQue_.AllocTensor<float>();
        DataCopyPad(updateRValueLocal, workspaceMaxValue_[RValueOffset], dataCopyParam, dataPadParam);
        RValueQue_.EnQue(updateRValueLocal);
        uint32_t RCountsValue = workspaceMaxValueCount_(updateSumIdxLocal(i));
        auto vWaitSEventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(vWaitSEventID);
        WaitFlag<HardEvent::S_V>(vWaitSEventID);
        QuantizeForSum(i, RCountsValue, updateSumAddr);
        CopyQuantizedSumOutToIntWs(RValueOffset);
    }
    updateSumInQue_.FreeTensor(updateSumLocal);
    updateSumIdxQueue_.FreeTensor(updateSumIdxLocal);
}

template<typename T, typename U>
__aicore__ inline void ScatterNdDeterministicImpl<T,  U>::QuantizeForSum(uint64_t curRowIdx, uint32_t RCountsValue,
    __local_mem__ float* updateSumAddr)
{
    LocalTensor<float> updateRValueLocal = RValueQue_.DeQue<float>();
    LocalTensor<int> updateSumIntLocal = updateSumQue_.AllocTensor<int>();
    
    __local_mem__ float* updatesRValueAddr = (__local_mem__ float*)updateRValueLocal.GetPhyAddr();
    __local_mem__ int* updateSumIntAddr = (__local_mem__ int*)updateSumIntLocal.GetPhyAddr();

    uint32_t maskLen = static_cast<uint32_t>(tilingData_.afterAxis);
    constexpr uint32_t vfLen = platform::GetVRegSize() / sizeof(float);
    uint16_t loopCnt = ops::CeilDiv(maskLen, vfLen);
    float scaling = static_cast<float>(1 << 30);
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<float> dataReg;
        AscendC::MicroAPI::RegTensor<float> rReg;
        AscendC::MicroAPI::RegTensor<float> resReg;
        AscendC::MicroAPI::RegTensor<int32_t> scaleReg;
        AscendC::MicroAPI::RegTensor<float> oneReg;
        AscendC::MicroAPI::MaskReg cmpReg;
        AscendC::MicroAPI::Duplicate(oneReg, (float)1);
        AscendC::MicroAPI::MaskReg pregLoop;
        for (uint16_t i = 0; i < loopCnt; i++) {
            pregLoop = AscendC::MicroAPI::UpdateMask<float>(maskLen);
            auto sumOffset = updateSumAddr + curRowIdx * postVarAlignSizeFp32_ + i * vfLen;
            auto rValueOffset = updatesRValueAddr + i * vfLen;
            auto sumIntOffset = updateSumIntAddr + i * vfLen;
            DataCopy(dataReg, sumOffset);
            DataCopy(rReg, rValueOffset);
            Muls(rReg, rReg, (float)RCountsValue, pregLoop);
            CompareScalar<float, CMPMODE::EQ>(cmpReg, rReg, (float)0, pregLoop);
            Select(rReg, oneReg, rReg, cmpReg);
            Div(resReg, dataReg, rReg, pregLoop);
            Muls(resReg, resReg, scaling, pregLoop);
            Cast<int32_t, float, castTraitFP322INT32>(scaleReg, resReg, pregLoop);
            DataCopy(sumIntOffset, scaleReg, pregLoop);
        }
    }
    RValueQue_.FreeTensor(updateRValueLocal);
    updateSumQue_.EnQue(updateSumIntLocal);
}

template<typename T, typename U>
__aicore__ inline void ScatterNdDeterministicImpl<T,  U>::CopyQuantizedSumOutToIntWs(uint64_t RValueOffset)
{
    LocalTensor<int> updateSumIntLocal = updateSumQue_.DeQue<int>();
    DataCopyExtParams outParams = { static_cast<uint16_t>(1), static_cast<uint32_t>(tilingData_.afterAxis * sizeof(int)),
                                    static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0) };
    SetAtomicAdd<int>();
    DataCopyPad(workspaceInt32Res_[RValueOffset], updateSumIntLocal, outParams);
    SetAtomicNone();
    updateSumQue_.FreeTensor(updateSumIntLocal);
}

template<typename T, typename U>
__aicore__ inline void ScatterNdDeterministicImpl<T,  U>::CopyIntSumIn(uint64_t loopIdx, uint64_t dataCount)
{
    LocalTensor<int> updateIntSumLocal = updateSumInQue_.AllocTensor<int>();

    uint64_t updatesSumUboffset = (GetBlockIdx() * tilingData_.perCoreHandleOutputRows + loopIdx * tilingData_.rowsInUb) * tilingData_.afterAxis;
    DataCopyExtParams dataCopyParam { static_cast<uint16_t>(dataCount), static_cast<uint32_t>(tilingData_.afterAxis * sizeof(int)),
                                      static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0) };
    DataCopyPadExtParams<int> dataPadParam { false, static_cast<uint8_t>(0), static_cast<uint8_t>(0), static_cast<int>(0) };
    DataCopyPad(updateIntSumLocal, workspaceInt32Res_[updatesSumUboffset], dataCopyParam, dataPadParam);

    updateSumInQue_.EnQue(updateIntSumLocal);
}

template<typename T, typename U>
__aicore__ inline void ScatterNdDeterministicImpl<T,  U>::CopyCoreLogicCoreSumIdIn(uint64_t loopIdx, uint64_t dataCount)
{
    LocalTensor<U> updateSumIdxLocal = updateSumIdxQueue_.AllocTensor<U>();
    uint64_t idWsOffset = loopIdx * tilingData_.rowsInUb;
    DataCopyExtParams dataCopyParam { static_cast<uint16_t>(1), static_cast<uint32_t>(dataCount * sizeof(U)),
                                      static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0) };
    DataCopyPadExtParams<U> dataPadParam { false, static_cast<uint8_t>(0), static_cast<uint8_t>(0), static_cast<U>(0) };
    DataCopyPad(updateSumIdxLocal, workspaceCoreLogicCoreSumId_[idWsOffset], dataCopyParam, dataPadParam);
    updateSumIdxQueue_.EnQue(updateSumIdxLocal);
}

template<typename T, typename U>
__aicore__ inline void ScatterNdDeterministicImpl<T,  U>::ComputeRValueAndDeQuantize(uint64_t loopIdx, uint64_t dataLen)
{
    LocalTensor<int> updateSumIntLocal = updateSumInQue_.DeQue<int>();
    __local_mem__ int32_t * updateSumIntAddr = (__local_mem__ int32_t*)updateSumIntLocal.GetPhyAddr();

    for (int64_t i = 0; i < dataLen; i++) {
        uint64_t wspRValueOffset = (GetBlockIdx() * tilingData_.perCoreHandleOutputRows + loopIdx * tilingData_.rowsInUb + i) * tilingData_.afterAxis;
        uint64_t RCountsOffset = GetBlockIdx() * tilingData_.perCoreHandleOutputRows + loopIdx * tilingData_.rowsInUb + i;
        DataCopyExtParams dataCopyParam1 { static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(float) * tilingData_.afterAxis),
                                           static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0) };
        DataCopyPadExtParams<float> dataPadParam1 { false, static_cast<uint8_t>(0), static_cast<uint8_t>(0), static_cast<float>(0) };
        LocalTensor<float> updateRValueLocal = RValueQue_.AllocTensor<float>();
        DataCopyPad(updateRValueLocal, workspaceMaxValue_[wspRValueOffset], dataCopyParam1, dataPadParam1);
        RValueQue_.EnQue(updateRValueLocal);
        PipeBarrier<PIPE_ALL>();
        uint32_t RCountsValue = workspaceMaxValueCount_(RCountsOffset);
        auto vWaitSEventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(vWaitSEventID);
        WaitFlag<HardEvent::S_V>(vWaitSEventID);
        DeQuantizeForSum(i, RCountsValue, updateSumIntAddr);
        CopyOutDeQuantizedSum(wspRValueOffset);
    }

    updateSumInQue_.FreeTensor(updateSumIntLocal);
}

template<typename T, typename U>
__aicore__ inline void ScatterNdDeterministicImpl<T,  U>::ComputeRValueAndDeQuantize_idxSplit(uint64_t loopIdx, uint64_t dataLen)
{
    DataCopyExtParams dataCopyParam { static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(float) * tilingData_.afterAxis),
                                      static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0) };
    DataCopyPadExtParams<float> dataPadParam { false, static_cast<uint8_t>(0), static_cast<uint8_t>(0), static_cast<float>(0) };
    
    DataCopyExtParams dataCopyParam2 { static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(int) * tilingData_.afterAxis),
                                       static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0) };
    DataCopyPadExtParams<int> dataPadParam2 { false, static_cast<uint8_t>(0), static_cast<uint8_t>(0), static_cast<int>(0) };
    
    auto sWaitMte2EventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(sWaitMte2EventID);
    WaitFlag<HardEvent::MTE2_S>(sWaitMte2EventID);

    LocalTensor<U> updateSumIdxLocal = updateSumIdxQueue_.DeQue<U>();

    for (int64_t i = 0; i < dataLen; i++) {
        int64_t curIdx = updateSumIdxLocal(i);
        if(curIdx < 0 || curIdx >= static_cast<U>(tilingData_.rankFusedAxis)){
            break;
        }
        uint64_t wspRValueOffset = curIdx * tilingData_.afterAxis;
        uint64_t RCountsOffset = curIdx;
        uint64_t updatesSumIntoffset = curIdx * tilingData_.afterAxis;
       
        LocalTensor<float> updateRValueLocal = RValueQue_.AllocTensor<float>();
        DataCopyPad(updateRValueLocal, workspaceMaxValue_[wspRValueOffset], dataCopyParam, dataPadParam);
        RValueQue_.EnQue(updateRValueLocal);
        
        LocalTensor<int> updateSumInLocal = updateSumInQue_.AllocTensor<int>();
        DataCopyPad(updateSumInLocal, workspaceInt32Res_[updatesSumIntoffset], dataCopyParam2, dataPadParam2);
        updateSumInQue_.EnQue(updateSumInLocal);
        
        uint32_t RCountsValue = workspaceMaxValueCount_(RCountsOffset);

        auto vWaitSEventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(vWaitSEventID);
        WaitFlag<HardEvent::S_V>(vWaitSEventID);
        
        updateSumInLocal = updateSumInQue_.DeQue<int>();
        __local_mem__ int32_t * updateSumIntAddr = (__local_mem__ int32_t*)updateSumInLocal.GetPhyAddr();
        DeQuantizeForSum(0, RCountsValue, updateSumIntAddr);
        updateSumInQue_.FreeTensor(updateSumInLocal);

        CopyOutDeQuantizedSum(wspRValueOffset);
    }
    updateSumIdxQueue_.FreeTensor(updateSumIdxLocal);
}

template<typename T, typename U>
__aicore__ inline void ScatterNdDeterministicImpl<T,  U>::DeQuantizeForSum(uint64_t curRowIdx, uint32_t RCountsValue, __local_mem__ int32_t* updateSumIntAddr)
{
    LocalTensor<float> updateRValueLocal = RValueQue_.DeQue<float>();
    LocalTensor<T> updateSumLocal = updateSumQue_.AllocTensor<T>();
    
    __local_mem__ T* updateSumAddr = (__local_mem__ T*)updateSumLocal.GetPhyAddr();
    __local_mem__ float* updatesRValueAddr = (__local_mem__ float*)updateRValueLocal.GetPhyAddr();

    uint32_t maskLen = static_cast<uint32_t>(tilingData_.afterAxis);
    constexpr uint32_t vfLen = platform::GetVRegSize() / sizeof(float);
    uint16_t loopCnt = (tilingData_.afterAxis + vfLen - 1) / vfLen;
    float scaling = static_cast<float>(1 << 30);
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<float> dataReg;
        AscendC::MicroAPI::RegTensor<float> rReg;
        AscendC::MicroAPI::RegTensor<float> resRegFp32;
        AscendC::MicroAPI::RegTensor<T> resReg;
        AscendC::MicroAPI::RegTensor<int32_t> scaleReg;
        AscendC::MicroAPI::MaskReg pregLoop;
        AscendC::MicroAPI::MaskReg maskReg =
        AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::UnalignReg uIn;
        AscendC::MicroAPI::UnalignReg uOut;
        AscendC::MicroAPI::RegTensor<float> oneReg;
        AscendC::MicroAPI::MaskReg cmpReg;
        AscendC::MicroAPI::Duplicate(oneReg, (float)1);
        AscendC::MicroAPI::Duplicate(dataReg, scaling, maskReg);
        auto sumIntOffset = updateSumIntAddr + curRowIdx * postVarAlignSizeFp32_;
        auto rValueOffset = updatesRValueAddr;
        auto sumResOffset = updateSumAddr;
        for (uint16_t i = 0; i < loopCnt; ++i) {
            pregLoop = AscendC::MicroAPI::UpdateMask<float>(maskLen);
            DataCopy(scaleReg, sumIntOffset + i * vfLen);
            DataCopy(rReg, rValueOffset + i * vfLen);
            Muls(rReg, rReg, (float)RCountsValue, pregLoop);
            CompareScalar<float, CMPMODE::EQ>(cmpReg, rReg, (float)0, pregLoop);
            Select(rReg, oneReg, rReg, cmpReg);
            Cast<float, int, castTraitINT322FP32>(resRegFp32, scaleReg, pregLoop);
            Mul(resRegFp32, resRegFp32, rReg, pregLoop);
            Div(resRegFp32, resRegFp32, dataReg, pregLoop);
            if constexpr (!std::is_same<float, T>::value) {
                Cast<T, float, castTraitFP322T>(resReg, resRegFp32, pregLoop);
                DataCopy<T, MicroAPI::StoreDist::DIST_PACK_B32>(sumResOffset + i * vfLen, resReg, pregLoop);
            } else {
                DataCopy(sumResOffset + i * vfLen, resRegFp32, pregLoop);
            }        
        }
    }
    updateSumQue_.EnQue(updateSumLocal);
    RValueQue_.FreeTensor(updateRValueLocal);
}

template<typename T, typename U>
__aicore__ inline void ScatterNdDeterministicImpl<T,  U>::CopyOutDeQuantizedSum(uint64_t varOffset)
{
    LocalTensor<T> invQuantDataLoacl = updateSumQue_.DeQue<T>();
    DataCopyExtParams dataTCopyParam { static_cast<uint16_t>(1), static_cast<uint32_t>(tilingData_.afterAxis * sizeof(T)),
                                       static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0) };
    DataCopyPad(yGm_[varOffset], invQuantDataLoacl, dataTCopyParam);
    updateSumQue_.FreeTensor(invQuantDataLoacl);
}

template<typename T, typename U>
__aicore__ inline void ScatterNdDeterministicImpl<T,  U>::Process()
{
    if (tilingData_.isDeterministic == 0) {
        if (GetBlockIdx() < tilingData_.logicCoreNum) {
            ProcessAtomicAdd();
        }
        return;
    }

    InitWspZero();
    if (tilingData_.isIdxSplit == 1) {
        InitGlobalMemory(yGmInit_, initCoreReal, (T)0);
    }
    SyncAll();

    LocalTensor<U> calcLocal = calcBuf_.Get<U>();
    for (int32_t i = 0; i < MAX_RANK_COUNT; i++) {
        calcLocal(i) = tilingData_.strideList[i];
    }

    if (GetBlockIdx() < tilingData_.logicCoreNum) {

        indicesUbLoop_ = GetBlockIdx() == tilingData_.logicCoreNum - 1 ? tilingData_.tailCoreIndicesLoopSize : tilingData_.indicesLoopSize;
        for (uint64_t loopIdx = 0; loopIdx < indicesUbLoop_ - 1; loopIdx++) {
            CopyInIndicesAndUpdates(loopIdx, tilingData_.indicesUbFactor);
            ProcessSortAndSum(loopIdx, tilingData_.indicesUbFactor);
            CopySumOutToWs(loopIdx, tilingData_.indicesUbFactor);
        }
        tailIndicesUbLength_ = GetBlockIdx() == tilingData_.logicCoreNum - 1 ? tilingData_.tailCoreIndicesTailUbFactor : tilingData_.indicesTailUbFactor;
        CopyInIndicesAndUpdates(indicesUbLoop_ - 1, tailIndicesUbLength_);
        ProcessSortAndSum(indicesUbLoop_ - 1, tailIndicesUbLength_);
        CopySumOutToWs(indicesUbLoop_ - 1, tailIndicesUbLength_);
    }

    SyncAll();
    if (GetBlockIdx() < tilingData_.logicCoreNum) {
        pipe_.Reset();
        pipe_.InitBuffer(RValueQue_, DOUBLE_BUF, tilingData_.indicesUbFactor * postVarAlignSizeFp32_ * sizeof(float));
        pipe_.InitBuffer(updateSumInQue_, DOUBLE_BUF, tilingData_.indicesUbFactor * postVarAlignSizeFp32_ * sizeof(float));
        pipe_.InitBuffer(updateSumQue_, DOUBLE_BUF, postVarAlignSizeFp32_ * sizeof(float));
        pipe_.InitBuffer(updateSumIdxQueue_, DOUBLE_BUF, ops::CeilAlign(tilingData_.indicesUbFactor * sizeof(U), UB_AGLIN_VALUE));

        for (uint64_t loopIdx = 0; loopIdx < indicesUbLoop_ - 1; loopIdx++) {
            CopySumAndIdxIn(loopIdx, tilingData_.indicesUbFactor);
            ComputeRValueAndQuantize(loopIdx, tilingData_.indicesUbFactor);
        }
        CopySumAndIdxIn(indicesUbLoop_ - 1, tailIndicesUbLength_);
        ComputeRValueAndQuantize(indicesUbLoop_ - 1, tailIndicesUbLength_);
    }
    SyncAll();

    if (GetBlockIdx() >= tilingData_.deQuantizeCoreNum) {
        return;
    }

    pipe_.Reset();
    pipe_.InitBuffer(RValueQue_, DOUBLE_BUF, tilingData_.rowsInUb * postVarAlignSizeFp32_ * sizeof(float));
    pipe_.InitBuffer(updateSumInQue_, DOUBLE_BUF, tilingData_.rowsInUb * postVarAlignSizeFp32_ * sizeof(float));
    pipe_.InitBuffer(updateSumQue_, DOUBLE_BUF, postVarAlignSize_ * sizeof(T));

    if (tilingData_.isIdxSplit == 1) {
        pipe_.InitBuffer(updateSumIdxQueue_, DOUBLE_BUF, ops::CeilAlign(tilingData_.rowsInUb * sizeof(U), UB_AGLIN_VALUE));
        
        uint64_t dequantUbLoopSize = GetBlockIdx() != tilingData_.deQuantizeCoreNum - 1 ? tilingData_.dequantLoopSize : tilingData_.tailCoreDequantLoopSize;
        uint64_t tailUbLen = GetBlockIdx() != tilingData_.deQuantizeCoreNum - 1 ? tilingData_.dequantTailUbFactor : tilingData_.tailCoreDequantTailUbFactor;

        for (uint64_t loopIdx = 0; loopIdx < dequantUbLoopSize - 1; loopIdx++) {
            CopyCoreLogicCoreSumIdIn(loopIdx, tilingData_.rowsInUb);
            ComputeRValueAndDeQuantize_idxSplit(loopIdx, tilingData_.rowsInUb);
        }
        CopyCoreLogicCoreSumIdIn(dequantUbLoopSize - 1, tailUbLen);
        ComputeRValueAndDeQuantize_idxSplit(dequantUbLoopSize - 1, tailUbLen);
    } else {
        uint64_t curCoreHandleVarRows = GetBlockIdx() == tilingData_.deQuantizeCoreNum - 1 ? tilingData_.tailCoreHandleOutputRows : tilingData_.perCoreHandleOutputRows;
        uint64_t rowsInLoop = ops::CeilDiv(curCoreHandleVarRows, tilingData_.rowsInUb);
        uint64_t tailUbLen = curCoreHandleVarRows - (rowsInLoop - 1) * tilingData_.rowsInUb;
        for (uint64_t loopIdx = 0; loopIdx < rowsInLoop - 1; loopIdx++) {
            CopyIntSumIn(loopIdx, tilingData_.rowsInUb);
            ComputeRValueAndDeQuantize(loopIdx, tilingData_.rowsInUb);
        }
        CopyIntSumIn(rowsInLoop - 1, tailUbLen);
        ComputeRValueAndDeQuantize(rowsInLoop - 1, tailUbLen);
    }
}
}
#endif  // SCATTER_ND_DETERMINISTIC_IMPL_H