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
 * \file scatter_add_deterministic.h
 * \brief deterministic kernel of scatter_add
 */

#ifndef SCATTER_ADD_DETERMINISTIC_IMPL_H
#define SCATTER_ADD_DETERMINISTIC_IMPL_H

#include "kernel_operator.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"

namespace ScatterAdd {
using namespace AscendC;

constexpr uint64_t DOUBLE_BUF = 2;
constexpr uint64_t BLOCK_SIZE = 32;
constexpr uint32_t DOUBLE = 2;
constexpr uint64_t THREE = 3;
constexpr uint32_t SORT_STAT_PADDING = 64;
constexpr uint64_t UB_AGLIN_VALUE = 32;

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
class ScatterAddDeterministicImpl {
public:
    __aicore__ inline ScatterAddDeterministicImpl(const ScatterAddTilingData& tilingData, TPipe& pipe) :
        tilingData_(tilingData), pipe_(pipe) {};
    __aicore__ inline void Init(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR varRef, GM_ADDR workspace);
    __aicore__ inline void InitWspZero();
    __aicore__ inline void CopyInUpdates(int64_t offset, int64_t dataLen);
    __aicore__ inline void CopyOutUpdates(int64_t offset, int64_t dataLen);
    __aicore__ inline void ExcuteAtomicAdd(int64_t varRefOffset, int64_t dataLen);
    __aicore__ inline void ProcessSingleLoopIndices(int64_t indicesOffset, int64_t indicesLen);
    __aicore__ inline void ProcessAtomicAdd();
    __aicore__ inline void CopyInIndicesAndUpdates(uint64_t loopIdNum, uint64_t rowNums);
    __aicore__ inline void ComputeUniqueIdNum(LocalTensor<U>& sortedIndicesLocal, LocalTensor<U>& uniqueIdCountLocalU, uint32_t dataLen);
    __aicore__ inline void ComputeUinqueIdTimes(LocalTensor<U>& uniqueIdCountLocalU, uint32_t uniqueIdNum);
    __aicore__ inline void ComputeSum(LocalTensor<U>& uniqueIdCountLocalU, LocalTensor<uint32_t>& updatesOriginIdexLocal,
        uint32_t uniqueIdNum, uint32_t dataLen);
    __aicore__ inline void ProcessSortAndSum(uint64_t loopIdx, uint32_t dataLen);
    __aicore__ inline void CopySumOutToWs(uint64_t loopIdx, uint64_t dataLen);
    __aicore__ inline void CopySumAndIdxIn(uint64_t loopIdx, uint64_t dataCount);
    __aicore__ inline void ComputeRValueAndQuantize(uint64_t loopIdx, uint64_t dataCount);
    __aicore__ inline void QuantizeForSum(uint64_t curRowIdx, uint32_t RCountsValue, __local_mem__ float* updateSumAddr);
    __aicore__ inline void CopyQuantizedSumOutToIntWs(uint64_t RValueOffset);
    __aicore__ inline void CopyIntSumIn(uint64_t loopIdx, uint64_t dataCount);
    __aicore__ inline void ComputeRValueAndDeQuantize(uint64_t loopIdx, uint64_t dataLen);
    __aicore__ inline void DeQuantizeForSum(uint64_t curRowIdx, uint32_t RCountsValue, __local_mem__ int32_t* updateSumIntAddr);
    __aicore__ inline void CopyOutDeQuantizedSum(uint64_t varOffset);
    __aicore__ inline void CopyIdxIn(uint64_t loopIdx, uint64_t dataCount);
    __aicore__ inline void ComputeRValueAndDeQuantizePro(uint64_t loopIdx, uint64_t dataLen);
    __aicore__ inline void Process();

private:
    AscendC::GlobalTensor<T> varGm_;
    AscendC::GlobalTensor<U> indicesGm_;
    AscendC::GlobalTensor<T> updatesGm_;
    AscendC::GlobalTensor<T> varRefGm_;
    AscendC::GlobalTensor<float> workspaceMaxValue_;
    AscendC::GlobalTensor<float> workspaceLogicCoreSumValue_;
    AscendC::GlobalTensor<U> workspaceCoreLogicCoreSumId_;
    AscendC::GlobalTensor<uint32_t> workspaceMaxValueCount_;
    AscendC::GlobalTensor<int32_t> workspaceInt32Res_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> dataQueue_;
    TQue<QuePosition::VECIN, DOUBLE_BUF> updatesQueue_;
    TBuf<QuePosition::VECCALC> indicesBuf_;

    TQue<QuePosition::VECIN, DOUBLE_BUF> indicesQue_;
    TBuf<QuePosition::VECCALC> sortedIndicesQue_;
    TBuf<QuePosition::VECCALC> updatesOriginIdexQue_;
    TQue<QuePosition::VECIN, DOUBLE_BUF> updateSumInQue_;
    TQue<QuePosition::VECOUT, DOUBLE_BUF> updateSumQue_;
    TQue<QuePosition::VECOUT, DOUBLE_BUF> updateSumIdxQueue_;
    TQue<QuePosition::VECIN, DOUBLE_BUF> RValueQue_;
    TQue<QuePosition::VECIN, DOUBLE_BUF> RCountsQue_;
    TPipe& pipe_;
    const ScatterAddTilingData& tilingData_;

    bool isDeterministic{false};
    uint64_t shiftOffset_{0};
    uint32_t uniqueIdNum_{0};
    uint64_t indicesUbLoop_{0};
    uint64_t tailIndicesUbLength_{0};
    uint64_t updatesSumUbFactor_{0};
    uint64_t indicesUbFactor_{0};
    uint64_t postVarAlignSize_{0};
    uint64_t postVarAlignSizeFp32_{0};
};

template<typename T, typename U>
__aicore__ inline void ScatterAddDeterministicImpl<T,  U>::Init(GM_ADDR var, GM_ADDR indices, GM_ADDR updates,
    GM_ADDR varRef, GM_ADDR workspace)
{
    isDeterministic = tilingData_.isDeterministic == 1 ? true : false;
    postVarAlignSize_ = ops::CeilAlign(tilingData_.varShape[1] * sizeof(T), UB_AGLIN_VALUE) / sizeof(T);
    postVarAlignSizeFp32_ = ops::CeilAlign(tilingData_.varShape[1] * sizeof(float), UB_AGLIN_VALUE) / sizeof(float);
    varRefGm_.SetGlobalBuffer((__gm__ T *)(varRef));
    workspaceMaxValueCount_.SetGlobalBuffer((__gm__ uint32_t *)(workspace));
    workspaceMaxValue_.SetGlobalBuffer((__gm__ float *)workspace + tilingData_.varShape[0]);
    workspaceInt32Res_.SetGlobalBuffer((__gm__ int32_t *)workspace + tilingData_.varShape[0] +
        tilingData_.varShape[1] * tilingData_.varShape[0]);

    if (GetBlockIdx() < tilingData_.logicCoreNum) {  
    // 非量化分支
        if (!isDeterministic) {
            pipe_.InitBuffer(dataQueue_, 1, tilingData_.updatesUbFactor * sizeof(T));
            pipe_.InitBuffer(indicesBuf_, tilingData_.indicesUbFactor * sizeof(U));
            varRefGm_.SetGlobalBuffer((__gm__ T *)(varRef) + GetBlockIdx() * tilingData_.perCoreHandleCol);
            indicesGm_.SetGlobalBuffer((__gm__ U *)(indices));
            updatesGm_.SetGlobalBuffer((__gm__ T *)(updates) + GetBlockIdx() * tilingData_.perCoreHandleCol);
            return;
        }
        // 量化分支
        shiftOffset_ = UB_AGLIN_VALUE / sizeof(U);
        indicesUbLoop_ = GetBlockIdx() == tilingData_.logicCoreNum - 1 ? tilingData_.tailCoreIndicesLoopSize : tilingData_.indicesLoopSize;

        pipe_.InitBuffer(indicesQue_, DOUBLE_BUF, ops::CeilAlign(tilingData_.indicesUbFactor * sizeof(U), UB_AGLIN_VALUE));
        pipe_.InitBuffer(sortedIndicesQue_, ops::CeilAlign(tilingData_.indicesUbFactor * sizeof(U), UB_AGLIN_VALUE) + SORT_STAT_PADDING);
        pipe_.InitBuffer(updatesOriginIdexQue_, ops::CeilAlign(tilingData_.indicesUbFactor * sizeof(uint32_t), UB_AGLIN_VALUE));
        pipe_.InitBuffer(updateSumIdxQueue_, DOUBLE_BUF, ops::CeilAlign(tilingData_.indicesUbFactor * sizeof(U), UB_AGLIN_VALUE));
        pipe_.InitBuffer(updatesQueue_, DOUBLE_BUF, tilingData_.indicesUbFactor * postVarAlignSize_ * sizeof(T));
        pipe_.InitBuffer(updateSumQue_, DOUBLE_BUF, tilingData_.indicesUbFactor * postVarAlignSizeFp32_ * sizeof(float));
        
        indicesGm_.SetGlobalBuffer((__gm__ U *)(indices) + GetBlockIdx() * tilingData_.perCoreHandleIndices);
        
        updatesGm_.SetGlobalBuffer((__gm__ T *)(updates) + GetBlockIdx() * tilingData_.perCoreHandleIndices * tilingData_.varShape[1]);
        workspaceLogicCoreSumValue_.SetGlobalBuffer((__gm__ float*)workspace + tilingData_.varShape[0] + 
        tilingData_.varShape[0] * tilingData_.varShape[1] * DOUBLE + GetBlockIdx() * tilingData_.perCoreHandleIndices * postVarAlignSizeFp32_);

        __gm__ float* coreLogicCoreSumIdStart = (__gm__ float*)workspace + tilingData_.varShape[0] + 
                                                tilingData_.varShape[0] * tilingData_.varShape[1] * DOUBLE +
                                                tilingData_.logicCoreNum * tilingData_.perCoreHandleIndices * postVarAlignSizeFp32_;
        workspaceCoreLogicCoreSumId_.SetGlobalBuffer((__gm__ U*)coreLogicCoreSumIdStart + GetBlockIdx() * tilingData_.perCoreHandleIndices);
    }
}

template<typename T, typename U>
__aicore__ inline void ScatterAddDeterministicImpl<T,  U>::InitWspZero()
{
    InitGlobalMemory(workspaceCoreLogicCoreSumId_, tilingData_.perCoreHandleIndices, (U)(-1));
    InitGlobalMemory(workspaceLogicCoreSumValue_, tilingData_.perCoreHandleIndices * postVarAlignSizeFp32_, (float)(0));
    InitGlobalMemory(workspaceMaxValueCount_, tilingData_.varShape[0], (uint32_t)(0));
    InitGlobalMemory(workspaceMaxValue_, tilingData_.varShape[1] * tilingData_.varShape[0], (float)(0));
    InitGlobalMemory(workspaceInt32Res_, tilingData_.varShape[1] * tilingData_.varShape[0], (int)(0));
}

template<typename T, typename U>
__aicore__ inline void ScatterAddDeterministicImpl<T,  U>::CopyInUpdates(int64_t offset, int64_t dataLen)
{
    DataCopyExtParams inParams = {static_cast<uint16_t>(1), static_cast<uint32_t>(dataLen * sizeof(T)), 0, 0, 0 };
    DataCopyPadExtParams<T> padParams = { false, 0, 0, 0 };
    LocalTensor<T> updatesLocal = dataQueue_.AllocTensor<T>();
    DataCopyPad(updatesLocal, updatesGm_[offset], inParams, padParams);
    dataQueue_.EnQue(updatesLocal);
}

template<typename T, typename U>
__aicore__ inline void ScatterAddDeterministicImpl<T,  U>::CopyOutUpdates(int64_t offset, int64_t dataLen)
{
    DataCopyExtParams outParams = {static_cast<uint16_t>(1), static_cast<uint32_t>(dataLen * sizeof(T)), 0, 0, 0 };
    LocalTensor<T> updatesLocal = dataQueue_.DeQue<T>();
    SetAtomicAdd<T>();
    DataCopyPad(varRefGm_[offset], updatesLocal, outParams);
    SetAtomicNone();
    dataQueue_.FreeTensor(updatesLocal);
}

template<typename T, typename U>
__aicore__ inline void ScatterAddDeterministicImpl<T,  U>::ProcessSingleLoopIndices(int64_t indicesOffset,
    int64_t indicesLen)
{
    DataCopyExtParams inParams = {static_cast<uint16_t>(1), static_cast<uint32_t>(indicesLen * sizeof(U)), 0, 0, 0 };
    DataCopyPadExtParams<U> padParams = { false, 0, 0, 0 };
    LocalTensor<U> indicesLocal = indicesBuf_.Get<U>();
    DataCopyPad(indicesLocal, indicesGm_[indicesOffset], inParams, padParams);
    auto sWaitMTEEventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(sWaitMTEEventID);
    WaitFlag<HardEvent::MTE2_S>(sWaitMTEEventID);
    for (int64_t i = 0; i < indicesLen; ++i) {
        U indicesValue = indicesLocal.GetValue(i);
        if (static_cast<int64_t>(indicesValue) >= tilingData_.varShape[0]) {
            continue;
        }
        int64_t updatesOffset = indicesOffset * tilingData_.varShape[1] + i * tilingData_.varShape[1];
        int64_t varRefOffset = static_cast<int64_t>(indicesValue * tilingData_.varShape[1]);
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
        CopyInUpdates(updatesOffset, updatesTailUbFactor);
        CopyOutUpdates(varRefOffset, updatesTailUbFactor);
    }
}

template<typename T, typename U>
__aicore__ inline void ScatterAddDeterministicImpl<T,  U>::ProcessAtomicAdd()
{
    if (GetBlockIdx() >= tilingData_.logicCoreNum) {
        return;
    }

    int64_t indicesTailUbFactor = tilingData_.indicesTailUbFactor;
    int64_t indicesLoopSize = tilingData_.indicesLoopSize;
    int64_t indicesOffset = 0;
    for (int64_t indicesLoopIdx = 0; indicesLoopIdx < indicesLoopSize - 1; ++indicesLoopIdx) {
        indicesOffset = indicesLoopIdx * tilingData_.indicesUbFactor;
        ProcessSingleLoopIndices(indicesOffset, tilingData_.indicesUbFactor);
    }
    indicesOffset = (indicesLoopSize - 1) * tilingData_.indicesUbFactor;
    ProcessSingleLoopIndices(indicesOffset, indicesTailUbFactor);
}

template<typename T, typename U>
__aicore__ inline void ScatterAddDeterministicImpl<T,  U>:: CopyInIndicesAndUpdates(uint64_t loopIdNum, uint64_t rowNums)
{
    uint64_t updatesOffset = loopIdNum * tilingData_.indicesUbFactor * tilingData_.varShape[1];
    LocalTensor<T> updatesLocal = updatesQueue_.template AllocTensor<T>();

    DataCopyExtParams dataCopyParam{static_cast<uint16_t>(1), static_cast<uint32_t>(rowNums * tilingData_.varShape[1] * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> dataPadParam{false, 0, 0, 0};
    DataCopyPad(updatesLocal, updatesGm_[updatesOffset], dataCopyParam, dataPadParam);
    updatesQueue_.EnQue(updatesLocal);
    // copy indices
    uint64_t IdOffset = loopIdNum * tilingData_.indicesUbFactor;
    LocalTensor<U> indicesLocal = indicesQue_.AllocTensor<U>();
    DataCopyExtParams idCopyParam{static_cast<uint16_t>(1), static_cast<uint32_t>(rowNums * sizeof(U)), 0, 0, 0};
    DataCopyPadExtParams<U> idPadParam{false, 0, 0, 0};
    DataCopyPad(indicesLocal, indicesGm_[IdOffset], idCopyParam, idPadParam);
    indicesQue_.EnQue(indicesLocal);
}

template<typename T, typename U>
__aicore__ inline void ScatterAddDeterministicImpl<T,  U>::ComputeUniqueIdNum(LocalTensor<U>& sortedIndicesLocal, 
    LocalTensor<U>& uniqueIdCountLocalU, uint32_t dataLen)
{
    LocalTensor<U> updateSumIdxLocal =  updateSumIdxQueue_.AllocTensor<U>();
    LocalTensor<int32_t> uniqueIdCountLocal = uniqueIdCountLocalU.template ReinterpretCast<int32_t>();


    __local_mem__ U* sortedIndicesAddr = (__local_mem__ U*)sortedIndicesLocal[shiftOffset_].GetPhyAddr();
    __local_mem__ int32_t* uniqueIdCountsAddr = (__local_mem__ int32_t*)uniqueIdCountLocal.GetPhyAddr();
    __local_mem__ U* updateSumIdxAddr = (__local_mem__ U*)updateSumIdxLocal.GetPhyAddr();

    uint32_t vfLen = platform::GetVRegSize() / sizeof(U);
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
__aicore__ inline void ScatterAddDeterministicImpl<T,  U>::ComputeUinqueIdTimes(LocalTensor<U>& uniqueIdCountLocalU, uint32_t uniqueIdNum)
{
    LocalTensor<int32_t> uniqueIdCountLocal = uniqueIdCountLocalU.template ReinterpretCast<int32_t>();
    __local_mem__ int32_t* uniqueIdCountsAddr = (__local_mem__ int32_t*)uniqueIdCountLocal.GetPhyAddr();

    // compute repeated num of each id
    uint32_t vfLen = platform::GetVRegSize() / sizeof(uint32_t);
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
__aicore__ inline void ScatterAddDeterministicImpl<T,  U>::ComputeSum(LocalTensor<U>& uniqueIdCountLocalU, LocalTensor<uint32_t>& updatesOriginIdexLocal,
    uint32_t uniqueIdNum, uint32_t dataLen)
{
    LocalTensor<uint32_t> uniqueIdCountLocal = uniqueIdCountLocalU.template ReinterpretCast<uint32_t>();
    LocalTensor<T> updatesLocal = updatesQueue_.template DeQue<T>();
    LocalTensor<float> updateSumLocal = updateSumQue_.AllocTensor<float>();
    Duplicate(updateSumLocal, static_cast<float>(0), dataLen * postVarAlignSizeFp32_);

    __local_mem__ T* updatesAddr = (__local_mem__ T*)updatesLocal.GetPhyAddr();
    __local_mem__ float* updateSumAddr = (__local_mem__ float*)updateSumLocal.GetPhyAddr();

    uint32_t vfLen = platform::GetVRegSize() / sizeof(T);
    uint16_t loopSize = (tilingData_.varShape[1] + vfLen - 1) / vfLen;
    int32_t idLocation = 0;
    uint32_t vlSize = platform::GetVRegSize();
    uint32_t maskLen = static_cast<uint32_t>((postVarAlignSizeFp32_ * sizeof(float) + vlSize - 1) / vlSize * vlSize / sizeof(float) * dataLen);
    __VEC_SCOPE__
    {
        for (uint16_t i = 0; i < static_cast<uint16_t>(uniqueIdNum); i++) {
            AscendC::MicroAPI::RegTensor<float> sumReg;
            AscendC::MicroAPI::RegTensor<T> tmpReg;
            AscendC::MicroAPI::RegTensor<T> dstReg1;
            AscendC::MicroAPI::RegTensor<T> dstReg2;
            AscendC::MicroAPI::RegTensor<float> tmpRegB32;
            AscendC::MicroAPI::MaskReg maskReg =
                AscendC::MicroAPI::CreateMask<int32_t, AscendC::MicroAPI::MaskPattern::ALL>();
            MicroAPI::UnalignReg u0;
            AscendC::MicroAPI::UnalignReg uOut;
            AscendC::MicroAPI::MaskReg maskReg1;
            uint16_t idRepeatTimes = static_cast<uint16_t>(uniqueIdCountLocal(i));
            for (uint16_t j = 0; j < loopSize; ++j) {
                maskReg1 = AscendC::MicroAPI::UpdateMask<int32_t>(maskLen);
                AscendC::MicroAPI::Duplicate(sumReg, (float)0, maskReg);
                for (uint16_t k = 0; k < idRepeatTimes; k++) {
                    auto updatesOffet = updatesOriginIdexLocal(idLocation + k) * tilingData_.varShape[1] + j * vfLen;
                    auto startAddr = updatesAddr + updatesOffet;
                    AscendC::MicroAPI::DataCopyUnAlignPre(u0, startAddr);
                    AscendC::MicroAPI::DataCopyUnAlign<T>(tmpReg, u0, startAddr, vfLen);
                    if constexpr (std::is_same<half, T>::value) {
                        Interleave(dstReg1, dstReg2, tmpReg, tmpReg);
                        Cast<float, half, castTraitB162B32>(tmpRegB32, dstReg1, maskReg1);
                        AscendC::MicroAPI::Add(sumReg, sumReg, tmpRegB32, maskReg1);
                    } else if constexpr (std::is_same<bfloat16_t, T>::value) {
                        Interleave(dstReg1, dstReg2, tmpReg, tmpReg);
                        Cast<float, bfloat16_t, castTraitB162B32>(tmpRegB32, dstReg1, maskReg1);
                        AscendC::MicroAPI::Add(sumReg, sumReg, tmpRegB32, maskReg1);
                    } else {
                        AscendC::MicroAPI::Add(sumReg, sumReg, tmpReg, maskReg1);
                    }
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
__aicore__ inline void ScatterAddDeterministicImpl<T,  U>::ProcessSortAndSum(uint64_t loopIdx, uint32_t dataLen)
{
    LocalTensor<U> indicesLocal = indicesQue_.DeQue<U>();
    LocalTensor<U> sortedIndicesLocal = sortedIndicesQue_.Get<U>();   
    
    LocalTensor<uint32_t> updatesOriginIdexLocal = updatesOriginIdexQue_.Get<uint32_t>();   
    Duplicate(sortedIndicesLocal, static_cast<U>(-1), shiftOffset_ * DOUBLE + tilingData_.indicesUbFactor);
    LocalTensor<U> shiftSortLocal = sortedIndicesLocal[shiftOffset_];
    AscendC::Sort<U, false, sortConfig>(shiftSortLocal, updatesOriginIdexLocal, indicesLocal, dataLen);

    ComputeUniqueIdNum(sortedIndicesLocal, indicesLocal, dataLen);
    event_t eventV2S =static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventV2S);
    WaitFlag<HardEvent::S_V>(eventV2S);
    ComputeUinqueIdTimes(indicesLocal, uniqueIdNum_);
    event_t eventS2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventS2V);
    WaitFlag<HardEvent::V_S>(eventS2V);
    ComputeSum(indicesLocal, updatesOriginIdexLocal, uniqueIdNum_, dataLen);
    indicesQue_.FreeTensor(indicesLocal);
}

template<typename T, typename U>
__aicore__ inline void ScatterAddDeterministicImpl<T,  U>::CopySumOutToWs(uint64_t loopIdx, uint64_t dataLen)
{
    // updatesSum和updatesSumId整块搬出去
    uint32_t updatesWspOffset = loopIdx * tilingData_.indicesUbFactor * postVarAlignSizeFp32_;
    LocalTensor<float> updateSumLocal = updateSumQue_.DeQue<float>();
    LocalTensor<U> updateSumIdxLocal = updateSumIdxQueue_.DeQue<U>();
    DataCopyExtParams outParams = {static_cast<uint16_t>(1), static_cast<uint32_t>(uniqueIdNum_ * postVarAlignSizeFp32_ * sizeof(float)), 0, 0, 0 };
    DataCopyPad(workspaceLogicCoreSumValue_[updatesWspOffset], updateSumLocal, outParams);
    outParams = {static_cast<uint16_t>(1), static_cast<uint32_t>(uniqueIdNum_ * sizeof(U)), 0, 0, 0 };
    DataCopyPad(workspaceCoreLogicCoreSumId_[loopIdx * tilingData_.indicesUbFactor], updateSumIdxLocal, outParams);
    auto vWaitMTEEventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(vWaitMTEEventID);
    WaitFlag<HardEvent::MTE3_V>(vWaitMTEEventID);
    PipeBarrier<PIPE_ALL>();

    // max逐行搬出, max计数加1
    Abs(updateSumLocal, updateSumLocal, uniqueIdNum_ * postVarAlignSizeFp32_);
    for (uint32_t i = 0; i < uniqueIdNum_; i++) {
        uint64_t maxWspOffset = static_cast<uint64_t>(updateSumIdxLocal(i) * tilingData_.varShape[1]);
        if (updateSumIdxLocal(i) < 0 || updateSumIdxLocal(i) >= static_cast<U>(tilingData_.varShape[0])) {
            continue;
        }
        outParams = {static_cast<uint16_t>(1), static_cast<uint32_t>(tilingData_.varShape[1] * sizeof(float)), 0, 0, 0 };
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
__aicore__ inline void ScatterAddDeterministicImpl<T,  U>::CopySumAndIdxIn(uint64_t loopIdx, uint64_t dataCount)
{
    LocalTensor<float> updateSumLocal = updateSumInQue_.AllocTensor<float>();
    LocalTensor<U> updateSumIdxLocal = updateSumIdxQueue_.AllocTensor<U>();
    

    uint64_t indicesUbOffset = loopIdx * tilingData_.indicesUbFactor;
    DataCopyExtParams dataCopyParam{static_cast<uint16_t>(1), static_cast<uint32_t>(dataCount * sizeof(U)), 0, 0, 0};
    DataCopyPadExtParams<U> dataPadParam{false, 0, 0, 0};
    DataCopyPad(updateSumIdxLocal, workspaceCoreLogicCoreSumId_[indicesUbOffset], dataCopyParam, dataPadParam);

    uint64_t updatesSumUboffset = loopIdx * tilingData_.indicesUbFactor * postVarAlignSizeFp32_;
    DataCopyExtParams dataCopy{static_cast<uint16_t>(1), static_cast<uint32_t>(dataCount * postVarAlignSizeFp32_ * sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> dataPad{false, 0, 0, 0};
    DataCopyPad(updateSumLocal, workspaceLogicCoreSumValue_[updatesSumUboffset], dataCopy, dataPad);

    updateSumInQue_.EnQue(updateSumLocal);
    updateSumIdxQueue_.EnQue(updateSumIdxLocal);
}

template<typename T, typename U>
__aicore__ inline void ScatterAddDeterministicImpl<T,  U>::ComputeRValueAndQuantize(uint64_t loopIdx, uint64_t dataCount) 
{
    LocalTensor<float> updateSumLocal = updateSumInQue_.DeQue<float>();
    LocalTensor<U> updateSumIdxLocal = updateSumIdxQueue_.DeQue<U>();

    __local_mem__ float* updateSumAddr = (__local_mem__ float*)updateSumLocal.GetPhyAddr();
    auto sWaitMTEEventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(sWaitMTEEventID);
    WaitFlag<HardEvent::MTE2_S>(sWaitMTEEventID);

    for (int64_t i = 0; i < dataCount; i++) {
        if (updateSumIdxLocal(i) < 0 || updateSumIdxLocal(i) >= static_cast<U>(tilingData_.varShape[0])) {
            continue;
        }
        uint64_t RValueOffset = updateSumIdxLocal(i) * tilingData_.varShape[1];
        DataCopyExtParams dataCopyParam{static_cast<uint16_t>(1), static_cast<uint32_t>(tilingData_.varShape[1] * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> dataPadParam{false, 0, 0, 0};
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
__aicore__ inline void ScatterAddDeterministicImpl<T,  U>::QuantizeForSum(uint64_t curRowIdx, uint32_t RCountsValue,
    __local_mem__ float* updateSumAddr)
{
    LocalTensor<float> updateRValueLocal = RValueQue_.DeQue<float>();
    LocalTensor<int> updateSumIntLocal = updateSumQue_.AllocTensor<int>();
    
    __local_mem__ float* updatesRValueAddr = (__local_mem__ float*)updateRValueLocal.GetPhyAddr();
    __local_mem__ int* updateSumIntAddr = (__local_mem__ int*)updateSumIntLocal.GetPhyAddr();

    uint32_t maskLen = static_cast<uint32_t>(tilingData_.varShape[1]);
    uint32_t vfLen = platform::GetVRegSize() / sizeof(float);
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
__aicore__ inline void ScatterAddDeterministicImpl<T,  U>::CopyQuantizedSumOutToIntWs(uint64_t RValueOffset)
{
    LocalTensor<int> updateSumIntLocal = updateSumQue_.DeQue<int>();
    DataCopyExtParams outParams = {static_cast<uint16_t>(1), static_cast<uint32_t>(tilingData_.varShape[1] * sizeof(int)), 0, 0, 0 };
    SetAtomicAdd<int>();
    DataCopyPad(workspaceInt32Res_[RValueOffset], updateSumIntLocal, outParams);
    SetAtomicNone();
    updateSumQue_.FreeTensor(updateSumIntLocal);
}

template<typename T, typename U>
__aicore__ inline void ScatterAddDeterministicImpl<T,  U>::CopyIntSumIn(uint64_t loopIdx, uint64_t dataCount)
{
    LocalTensor<int> updateIntSumLocal = updateSumInQue_.AllocTensor<int>();

    uint64_t updatesSumUboffset = (GetBlockIdx() * tilingData_.perCoreHandleRows + loopIdx * tilingData_.rowsInUb) * tilingData_.varShape[1];
    DataCopyExtParams dataCopyParam{static_cast<uint16_t>(dataCount), static_cast<uint32_t>(tilingData_.varShape[1] * sizeof(int)), 0, 0, 0};
    DataCopyPadExtParams<int> dataPadParam{false, 0, 0, 0};
    DataCopyPad(updateIntSumLocal, workspaceInt32Res_[updatesSumUboffset], dataCopyParam, dataPadParam);

    updateSumInQue_.EnQue(updateIntSumLocal);
}

template<typename T, typename U>
__aicore__ inline void ScatterAddDeterministicImpl<T,  U>::ComputeRValueAndDeQuantize(uint64_t loopIdx, uint64_t dataLen)
{
    LocalTensor<int> updateSumIntLocal = updateSumInQue_.DeQue<int>();
    __local_mem__ int32_t * updateSumIntAddr = (__local_mem__ int32_t*)updateSumIntLocal.GetPhyAddr();

    for (int64_t i = 0; i < dataLen; i++) {
        uint64_t wspRValueOffset = (GetBlockIdx() * tilingData_.perCoreHandleRows + loopIdx * tilingData_.rowsInUb + i) * tilingData_.varShape[1];
        uint64_t RCountsOffset = GetBlockIdx() * tilingData_.perCoreHandleRows + loopIdx * tilingData_.rowsInUb + i;
        DataCopyExtParams dataCopyParam1{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(float) * tilingData_.varShape[1]), 0, 0, 0};
        DataCopyPadExtParams<float> dataPadParam1{false, 0, 0, 0};
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
__aicore__ inline void ScatterAddDeterministicImpl<T,  U>::DeQuantizeForSum(uint64_t curRowIdx, uint32_t RCountsValue, __local_mem__ int32_t* updateSumIntAddr)
{
    LocalTensor<float> updateRValueLocal = RValueQue_.DeQue<float>();
    LocalTensor<T> updateSumLocal = updateSumQue_.AllocTensor<T>();
    
    __local_mem__ T* updateSumAddr = (__local_mem__ T*)updateSumLocal.GetPhyAddr();
    __local_mem__ float* updatesRValueAddr = (__local_mem__ float*)updateRValueLocal.GetPhyAddr();

    uint32_t maskLen = static_cast<uint32_t>(tilingData_.varShape[1]);
    uint32_t vfLen = platform::GetVRegSize() / sizeof(float);
    uint16_t loopCnt = (tilingData_.varShape[1] + vfLen - 1) / vfLen;
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
__aicore__ inline void ScatterAddDeterministicImpl<T,  U>::CopyOutDeQuantizedSum(uint64_t varOffset)
{
    LocalTensor<T> invQuantDataLoacl = updateSumQue_.DeQue<T>();
    DataCopyExtParams dataTCopyParam{static_cast<uint16_t>(1), static_cast<uint32_t>(tilingData_.varShape[1] * sizeof(T)), 0, 0, 0};
    SetAtomicAdd<T>();
    DataCopyPad(varRefGm_[varOffset], invQuantDataLoacl, dataTCopyParam);
    SetAtomicNone();
    updateSumQue_.FreeTensor(invQuantDataLoacl);
}

template<typename T, typename U>
__aicore__ inline void ScatterAddDeterministicImpl<T,  U>::CopyIdxIn(uint64_t loopIdx, uint64_t dataCount)
{
    LocalTensor<U> updateSumIdxLocal = updateSumIdxQueue_.AllocTensor<U>();
    
    uint64_t indicesUbOffset = loopIdx * tilingData_.rowsInUb;
    DataCopyExtParams dataCopyParam{static_cast<uint16_t>(1), static_cast<uint32_t>(dataCount * sizeof(U)), 0, 0, 0};
    DataCopyPadExtParams<U> dataPadParam{false, 0, 0, 0};
    DataCopyPad(updateSumIdxLocal, workspaceCoreLogicCoreSumId_[indicesUbOffset], dataCopyParam, dataPadParam);
    updateSumIdxQueue_.EnQue(updateSumIdxLocal);
}

template<typename T, typename U>
__aicore__ inline void ScatterAddDeterministicImpl<T,  U>::ComputeRValueAndDeQuantizePro(uint64_t loopIdx, uint64_t dataLen)
{
    LocalTensor<U> updateSumIdxLocal = updateSumIdxQueue_.DeQue<U>();
    DataCopyExtParams dataCopyParam1{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(float) * tilingData_.varShape[1]), 0, 0, 0};
    DataCopyPadExtParams<float> dataPadParam1{false, 0, 0, 0};
    DataCopyExtParams dataCopyParam2{static_cast<uint16_t>(1), static_cast<uint32_t>(sizeof(int) * tilingData_.varShape[1]), 0, 0, 0};
    DataCopyPadExtParams<int> dataPadParam2{false, 0, 0, 0};

    auto sWaitMTEEventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(sWaitMTEEventID);
    WaitFlag<HardEvent::MTE2_S>(sWaitMTEEventID);

    for (int64_t i = 0; i < dataLen; i++) {
        int64_t curIndex = updateSumIdxLocal(i);
        if (curIndex < 0 || curIndex >= static_cast<U>(tilingData_.varShape[0])){
            continue;
        }

        uint32_t RCountsValue = AscendC::AtomicExch(const_cast<__gm__ uint32_t *>(workspaceMaxValueCount_[curIndex].GetPhyAddr()), uint32_t(0));
        if (RCountsValue == 0){
            continue;
        }
        uint64_t wspRValueAndIntOffset = curIndex * tilingData_.varShape[1];
        event_t eventS2Mte = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
        SetFlag<HardEvent::S_MTE2>(eventS2Mte);
        WaitFlag<HardEvent::S_MTE2>(eventS2Mte);
        
        LocalTensor<float> updateRValueLocal = RValueQue_.AllocTensor<float>();
        DataCopyPad(updateRValueLocal, workspaceMaxValue_[wspRValueAndIntOffset], dataCopyParam1, dataPadParam1);
        RValueQue_.EnQue(updateRValueLocal);

        LocalTensor<int> updateSumIntLocal = updateSumInQue_.AllocTensor<int>();
        DataCopyPad(updateSumIntLocal, workspaceInt32Res_[wspRValueAndIntOffset], dataCopyParam2, dataPadParam2);
        updateSumInQue_.EnQue(updateSumIntLocal);
        
        updateSumIntLocal = updateSumInQue_.DeQue<int>();
        __local_mem__ int32_t * updateSumIntAddr = (__local_mem__ int32_t*)updateSumIntLocal.GetPhyAddr();
        DeQuantizeForSum(0, RCountsValue, updateSumIntAddr);
        updateSumInQue_.FreeTensor(updateSumIntLocal);
        CopyOutDeQuantizedSum(wspRValueAndIntOffset);
    }
    updateSumIdxQueue_.FreeTensor(updateSumIdxLocal);
}

template<typename T, typename U>
__aicore__ inline void ScatterAddDeterministicImpl<T,  U>::Process()
{
    if ((GetBlockIdx() >= tilingData_.logicCoreNum) && !isDeterministic) {
        return;
    }
    if (GetBlockIdx() < tilingData_.logicCoreNum) {  
        if (!isDeterministic) {
            ProcessAtomicAdd();
            return;
        }
        
        // 初始化workspace内存
        InitWspZero();
    }
    SyncAll();
    
    if (GetBlockIdx() < tilingData_.logicCoreNum) {
        // 核内sort求和,并搬出到wsp
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

        // 量化
        for (uint64_t loopIdx = 0; loopIdx < indicesUbLoop_ - 1; loopIdx++) {
            CopySumAndIdxIn(loopIdx, tilingData_.indicesUbFactor);
            ComputeRValueAndQuantize(loopIdx, tilingData_.indicesUbFactor);
        }
        CopySumAndIdxIn(indicesUbLoop_ - 1, tailIndicesUbLength_);
        ComputeRValueAndQuantize(indicesUbLoop_ - 1, tailIndicesUbLength_);
    }

    SyncAll();

    // 反量化
    if (GetBlockIdx() >= tilingData_.deQuantizeCoreNum) {
        return;
    }

    pipe_.Reset();
    pipe_.InitBuffer(RValueQue_, DOUBLE_BUF, tilingData_.rowsInUb * postVarAlignSizeFp32_ * sizeof(float));
    pipe_.InitBuffer(updateSumQue_, DOUBLE_BUF, postVarAlignSize_ * sizeof(T));
    pipe_.InitBuffer(updateSumInQue_, DOUBLE_BUF, tilingData_.rowsInUb * postVarAlignSizeFp32_ * sizeof(float));

    uint64_t curCoreHandleRows = GetBlockIdx() == tilingData_.deQuantizeCoreNum - 1 ? tilingData_.tailCoreHandleRows : tilingData_.perCoreHandleRows;
    uint64_t rowsInLoop = ops::CeilDiv(curCoreHandleRows, tilingData_.rowsInUb);
    uint64_t tailUbLen = curCoreHandleRows - (rowsInLoop - 1) * tilingData_.rowsInUb;

    if (tilingData_.dequantizeBranch == 1) {
        // 反量化分支1，按indices分核
        pipe_.InitBuffer(updateSumIdxQueue_, DOUBLE_BUF, ops::CeilAlign(tilingData_.rowsInUb * sizeof(U), UB_AGLIN_VALUE));
        for (uint64_t loopIdx = 0; loopIdx < rowsInLoop - 1; loopIdx++) {
            CopyIdxIn(loopIdx, tilingData_.rowsInUb);
            ComputeRValueAndDeQuantizePro(loopIdx, tilingData_.rowsInUb);
        }
        CopyIdxIn(rowsInLoop - 1, tailUbLen);
        ComputeRValueAndDeQuantizePro(rowsInLoop - 1, tailUbLen);
    } else {
        // 反量化分支2，按var[0]分核
        for (uint64_t loopIdx = 0; loopIdx < rowsInLoop - 1; loopIdx++) {
            CopyIntSumIn(loopIdx, tilingData_.rowsInUb);
            ComputeRValueAndDeQuantize(loopIdx, tilingData_.rowsInUb);
        }
        CopyIntSumIn(rowsInLoop - 1, tailUbLen);
        ComputeRValueAndDeQuantize(rowsInLoop - 1, tailUbLen);
    }
}
}
#endif  // SCATTER_ADD_DETERMINISTIC_IMPL_H