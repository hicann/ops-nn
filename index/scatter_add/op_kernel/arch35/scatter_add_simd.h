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
 * \file scatter_add_simd.h
 * \brief simd kernel of scatter_add
 */

#ifndef SCATTER_ADD_SIMD_IMPL_H
#define SCATTER_ADD_SIMD_IMPL_H

#include "scatter_add_common.h"

namespace ScatterAdd {
using namespace AscendC;
using namespace ScatterAddCommon;

constexpr uint64_t DOUBLE_BUFFER = 2;

template<bool supportAtomicAdd, bool updatesIsScalar>
struct UpdatesSrcPosSelector { constexpr static TPosition pos = TPosition::GM; }; // 不支持atomic_add，搬入后cast到其他buffer
template<> struct UpdatesSrcPosSelector<true, false> { constexpr static TPosition pos = TPosition::VECIN; }; // 支持atomic_add且updates不是标量,直接搬入搬出
template<> struct UpdatesSrcPosSelector<true, true> { constexpr static TPosition pos = TPosition::VECOUT; }; // 支持atomic_add且updates是标量,duplicate后搬出

template<bool supportAtomicAdd, bool updatesIsScalar>
struct UpdatesDstPosSelector { constexpr static TPosition pos = TPosition::VECIN; }; // 不支持atomic_add，搬入后cast到其他buffer
template<> struct UpdatesDstPosSelector<true, false> { constexpr static TPosition pos = TPosition::VECOUT; }; // 支持atomic_add且updates不是标量,直接搬入搬出
template<> struct UpdatesDstPosSelector<true, true> { constexpr static TPosition pos = TPosition::GM; }; // 支持atomic_add且updates是标量,duplicate后搬出

template<typename T, typename U, bool updatesIsScalar, uint32_t scatterOp>
class ScatterAddSIMDImpl {
public:
    __aicore__ inline ScatterAddSIMDImpl(const ScatterAddTilingData& tilingData, TPipe& pipe) :
        tilingData_(tilingData), pipe_(pipe) {};
    __aicore__ inline void Init(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR varRef, GM_ADDR workspace);
    __aicore__ inline void CopyInUpdates(int64_t offset, int64_t dataLen);
    __aicore__ inline void CopyOutUpdates(int64_t offset, int64_t dataLen);
    __aicore__ inline void CopyInVar(int64_t offset, int64_t dataLen);
    __aicore__ inline void CopyOutVarToWS(int64_t offset, int64_t dataLen);
    __aicore__ inline void CopyOutUpdatesToWS(int64_t offset, int64_t dataLen);
    __aicore__ inline void CopyInVarFromWS(int64_t offset, int64_t dataLen);
    __aicore__ inline void CopyOutVar(int64_t offset, int64_t dataLen);
    __aicore__ inline void ExcuteAtomicAdd(int64_t varRefOffset, int64_t dataLen);
    __aicore__ inline void ProcessSingleLoopIndices(int64_t indicesOffset, int64_t indicesLen);
    __aicore__ inline void ProcessAtomicAdd();
    __aicore__ inline void ProcessVarToWS();
    __aicore__ inline void ProcessVarFromWS();
    __aicore__ inline void Process();

private:
    AscendC::GlobalTensor<T> varGm_;
    AscendC::GlobalTensor<U> indicesGm_;
    AscendC::GlobalTensor<T> updatesGm_;
    AscendC::GlobalTensor<T> varRefGm_;
    AscendC::GlobalTensor<int32_t> varCastGm_;
    AscendC::GlobalTensor<int32_t> varCastAtomicAddGm_;
    TQueBind<UpdatesSrcPosSelector<platform::IsSupportAtomicAddTypeSIMD<T>(), updatesIsScalar>::pos,
        UpdatesDstPosSelector<platform::IsSupportAtomicAddTypeSIMD<T>(), updatesIsScalar>::pos, DOUBLE_BUFFER>
        updatesQueue_;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> updatesCastQueue_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> varInQueue_;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> varCastOutQueue_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> varCastInQueue_;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> varOutQueue_;
    TBuf<QuePosition::VECCALC> indicesBuf_;
    TPipe& pipe_;
    const ScatterAddTilingData& tilingData_;
};

template<typename T, typename U, bool updatesIsScalar, uint32_t scatterOp>
__aicore__ inline void ScatterAddSIMDImpl<T, U, updatesIsScalar, scatterOp>::Init(GM_ADDR var, GM_ADDR indices, GM_ADDR updates,
    GM_ADDR varRef, GM_ADDR workspace)
{
    if (GetBlockIdx() >= GetBlockNum()) {
        return;
    }
    if constexpr (platform::IsSupportAtomicAddTypeSIMD<T>()) {
        pipe_.InitBuffer(updatesQueue_, DOUBLE_BUFFER, tilingData_.updatesUbFactor * sizeof(T));
        pipe_.InitBuffer(indicesBuf_, tilingData_.indicesUbFactor * sizeof(U));
        varRefGm_.SetGlobalBuffer((__gm__ T *)(varRef));
    } else {
        pipe_.InitBuffer(varInQueue_, DOUBLE_BUFFER, tilingData_.ubFactor * sizeof(T));
        pipe_.InitBuffer(varCastOutQueue_, DOUBLE_BUFFER, tilingData_.ubFactor * sizeof(int32_t));
        if (GetBlockIdx() < tilingData_.copyCoreNum) {
            varGm_.SetGlobalBuffer((__gm__ T *)(var) + GetBlockIdx() * tilingData_.perCoreHandleVar);
            varCastGm_.SetGlobalBuffer((__gm__ int32_t *)(workspace) + GetBlockIdx() * tilingData_.perCoreHandleVar);
        }
        varRefGm_.SetGlobalBuffer((__gm__ T *)(varRef) + GetBlockIdx() * tilingData_.perCoreHandleVar);
        varCastAtomicAddGm_.SetGlobalBuffer((__gm__ int32_t *)(workspace));
    }
    if (GetBlockIdx() < tilingData_.atomicAddCoreNum) {
        indicesGm_.SetGlobalBuffer((__gm__ U *)(indices) + GetBlockIdx() * tilingData_.perCoreHandleIndices);
        if constexpr (updatesIsScalar) {
            updatesGm_.SetGlobalBuffer((__gm__ T *)(updates));
        } else {
            updatesGm_.SetGlobalBuffer((__gm__ T *)(updates) +
                GetBlockIdx() * tilingData_.perCoreHandleIndices * tilingData_.postAxisSize);
        }
    }
}

template<typename T, typename U, bool updatesIsScalar, uint32_t scatterOp>
__aicore__ inline void ScatterAddSIMDImpl<T, U, updatesIsScalar, scatterOp>::CopyInUpdates(int64_t offset, int64_t dataLen)
{
    DataCopyExtParams inParams = { 1, static_cast<uint32_t>(dataLen * sizeof(T)), 0, 0, 0 };
    DataCopyPadExtParams<T> padParams = { false, 0, 0, 0 };
    LocalTensor<T> updatesLocal = updatesQueue_.template AllocTensor<T>();
    DataCopyPad(updatesLocal, updatesGm_[offset], inParams, padParams);
    updatesQueue_.EnQue(updatesLocal);
}

template<typename T, typename U, bool updatesIsScalar, uint32_t scatterOp>
__aicore__ inline void ScatterAddSIMDImpl<T, U, updatesIsScalar, scatterOp>::CopyOutUpdates(int64_t offset, int64_t dataLen)
{
    DataCopyExtParams outParams = { 1, static_cast<uint32_t>(dataLen * sizeof(T)), 0, 0, 0 };
    LocalTensor<T> updatesLocal = updatesQueue_.template DeQue<T>();
    SetAtomicAdd<T>();
    DataCopyPad(varRefGm_[offset], updatesLocal, outParams);
    SetAtomicNone();
    if constexpr (updatesIsScalar) {
        updatesQueue_.EnQue(updatesLocal);
    } else {
        updatesQueue_.FreeTensor(updatesLocal);
    }
}

template<typename T, typename U, bool updatesIsScalar, uint32_t scatterOp>
__aicore__ inline void ScatterAddSIMDImpl<T, U, updatesIsScalar, scatterOp>::CopyInVar(int64_t offset, int64_t dataLen)
{
    DataCopyExtParams inParams = { 1, static_cast<uint32_t>(dataLen * sizeof(T)), 0, 0, 0 };
    DataCopyPadExtParams<T> padParams = { false, 0, 0, 0 };
    LocalTensor<T> varLocal = varInQueue_.AllocTensor<T>();
    DataCopyPad(varLocal, varGm_[offset], inParams, padParams);
    varInQueue_.EnQue(varLocal);
}

template<typename T, typename U, bool updatesIsScalar, uint32_t scatterOp>
__aicore__ inline void ScatterAddSIMDImpl<T, U, updatesIsScalar, scatterOp>::CopyOutVarToWS(int64_t offset, int64_t dataLen)
{
    DataCopyExtParams outParams = { 1, static_cast<uint32_t>(dataLen * sizeof(int32_t)), 0, 0, 0 };
    LocalTensor<int32_t> varCastLocal = varCastOutQueue_.DeQue<int32_t>();
    DataCopyPad(varCastGm_[offset], varCastLocal, outParams);
    varCastOutQueue_.FreeTensor(varCastLocal);
}

template<typename T, typename U, bool updatesIsScalar, uint32_t scatterOp>
__aicore__ inline void ScatterAddSIMDImpl<T, U, updatesIsScalar, scatterOp>::CopyOutUpdatesToWS(int64_t offset, int64_t dataLen)
{
    DataCopyExtParams outParams = { 1, static_cast<uint32_t>(dataLen * sizeof(int32_t)), 0, 0, 0 };
    LocalTensor<int32_t> updatesCastLocal = updatesCastQueue_.DeQue<int32_t>();
    if constexpr (scatterOp == SUB && !updatesIsScalar) {
        NegateUpdate<int32_t>(updatesCastLocal, static_cast<uint32_t>(dataLen));
        auto MTE3WaitVEventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(MTE3WaitVEventID);
        WaitFlag<HardEvent::V_MTE3>(MTE3WaitVEventID);
    }
    SetAtomicAdd<int32_t>();
    DataCopyPad(varCastAtomicAddGm_[offset], updatesCastLocal, outParams);
    SetAtomicNone();
    if constexpr (updatesIsScalar) {
        updatesCastQueue_.EnQue(updatesCastLocal);
    } else {
        updatesCastQueue_.FreeTensor(updatesCastLocal);
    }
}

template<typename T, typename U, bool updatesIsScalar, uint32_t scatterOp>
__aicore__ inline void ScatterAddSIMDImpl<T, U, updatesIsScalar, scatterOp>::CopyInVarFromWS(int64_t offset, int64_t dataLen)
{
    DataCopyExtParams inParams = { 1, static_cast<uint32_t>(dataLen * sizeof(int32_t)), 0, 0, 0 };
    DataCopyPadExtParams<int32_t> padParams = { false, 0, 0, 0 };
    LocalTensor<int32_t> varCastLocal = varCastInQueue_.AllocTensor<int32_t>();
    DataCopyPad(varCastLocal, varCastGm_[offset], inParams, padParams);
    varCastInQueue_.EnQue(varCastLocal);
}

template<typename T, typename U, bool updatesIsScalar, uint32_t scatterOp>
__aicore__ inline void ScatterAddSIMDImpl<T, U, updatesIsScalar, scatterOp>::CopyOutVar(int64_t offset, int64_t dataLen)
{
    DataCopyExtParams outParams = { 1, static_cast<uint32_t>(dataLen * sizeof(T)), 0, 0, 0 };
    LocalTensor<T> varLocal = varOutQueue_.DeQue<T>();
    DataCopyPad(varRefGm_[offset], varLocal, outParams);
    varOutQueue_.FreeTensor(varLocal);
}

template<typename T, typename U, bool updatesIsScalar, uint32_t scatterOp>
__aicore__ inline void ScatterAddSIMDImpl<T, U, updatesIsScalar, scatterOp>::ExcuteAtomicAdd(int64_t varRefOffset,
    int64_t dataLen)
{
    if constexpr (platform::IsSupportAtomicAddTypeSIMD<T>()) {
        CopyOutUpdates(varRefOffset, dataLen);
    } else {
        if constexpr (!updatesIsScalar) {
            LocalTensor<T> updatesLocal = updatesQueue_.template DeQue<T>();
            LocalTensor<int32_t> updatesCastLocal = updatesCastQueue_.AllocTensor<int32_t>();
            CastToInt32<T>(updatesCastLocal, updatesLocal, dataLen);
            updatesQueue_.FreeTensor(updatesLocal);
            updatesCastQueue_.EnQue(updatesCastLocal);
        }
        CopyOutUpdatesToWS(varRefOffset, dataLen);
    }
}

template<typename T, typename U, bool updatesIsScalar, uint32_t scatterOp>
__aicore__ inline void ScatterAddSIMDImpl<T, U, updatesIsScalar, scatterOp>::ProcessSingleLoopIndices(int64_t indicesOffset,
    int64_t indicesLen)
{
    DataCopyExtParams inParams = { 1, static_cast<uint32_t>(indicesLen * sizeof(U)), 0, 0, 0 };
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
        int64_t updatesOffset = indicesOffset * tilingData_.postAxisSize + i * tilingData_.postAxisSize;
        int64_t varRefOffset = indicesValue * tilingData_.postAxisSize;
        for (int64_t updatesLoopIdx = 0; updatesLoopIdx < tilingData_.updatesLoopSize - 1; ++updatesLoopIdx) {
            if constexpr (!updatesIsScalar) {
                CopyInUpdates(updatesOffset, tilingData_.updatesUbFactor);
                updatesOffset += tilingData_.updatesUbFactor;
            }
            ExcuteAtomicAdd(varRefOffset, tilingData_.updatesUbFactor);
            varRefOffset += tilingData_.updatesUbFactor;
        }
        if constexpr (!updatesIsScalar) {
            CopyInUpdates(updatesOffset, tilingData_.updatesTailUbFactor);
        }
        ExcuteAtomicAdd(varRefOffset, tilingData_.updatesTailUbFactor);
    }
}

template<typename T, typename U, bool updatesIsScalar, uint32_t scatterOp>
__aicore__ inline void ScatterAddSIMDImpl<T, U, updatesIsScalar, scatterOp>::ProcessAtomicAdd()
{
    if (GetBlockIdx() >= tilingData_.atomicAddCoreNum) {
        return;
    }
    if constexpr (updatesIsScalar) {
        T updatesValue = updatesGm_.GetValue(0);
        auto vWaitSEventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(vWaitSEventID);
        WaitFlag<HardEvent::S_V>(vWaitSEventID);
        if constexpr (platform::IsSupportAtomicAddTypeSIMD<T>()) {
            LocalTensor<T> updatesLocal = updatesQueue_.template AllocTensor<T>();
            if constexpr (scatterOp == ADD) {
                Duplicate(updatesLocal, updatesValue, tilingData_.updatesUbFactor);
            } else if constexpr (scatterOp == SUB) {
                updatesValue = -updatesValue;
                Duplicate(updatesLocal, updatesValue, tilingData_.updatesUbFactor);
            }
            updatesQueue_.EnQue(updatesLocal);
        } else {
            LocalTensor<int32_t> updatesCastLocal = updatesCastQueue_.AllocTensor<int32_t>();
            if constexpr (scatterOp == ADD) {
                Duplicate(updatesCastLocal, static_cast<int32_t>(updatesValue), tilingData_.updatesUbFactor);
            } else if constexpr (scatterOp == SUB) {
                int32_t updatesValueInt32 = -static_cast<int32_t>(updatesValue);
                Duplicate(updatesCastLocal, updatesValueInt32, tilingData_.updatesUbFactor);
            }
            updatesCastQueue_.EnQue(updatesCastLocal);
        }
    }
    int64_t indicesTailUbFactor = tilingData_.indicesTailUbFactor;
    int64_t indicesLoopSize = tilingData_.indicesLoopSize;
    if (GetBlockIdx() == tilingData_.atomicAddCoreNum - 1) {
        indicesLoopSize = tilingData_.tailCoreIndicesLoopSize;
        indicesTailUbFactor = tilingData_.tailCoreIndicesTailUbFactor;
    }
    int64_t indicesOffset = 0;
    for (int64_t indicesLoopIdx = 0; indicesLoopIdx < indicesLoopSize - 1; ++indicesLoopIdx) {
        indicesOffset = indicesLoopIdx * tilingData_.indicesUbFactor;
        ProcessSingleLoopIndices(indicesOffset, tilingData_.indicesUbFactor);
    }
    indicesOffset = (indicesLoopSize - 1) * tilingData_.indicesUbFactor;
    ProcessSingleLoopIndices(indicesOffset, indicesTailUbFactor);
    if constexpr (updatesIsScalar) {
        if constexpr (platform::IsSupportAtomicAddTypeSIMD<T>()) {
            LocalTensor<T> updatesLocal = updatesQueue_.template DeQue<T>();
            updatesQueue_.FreeTensor(updatesLocal);
        } else {
            LocalTensor<int32_t> updatesCastLocal = updatesCastQueue_.DeQue<int32_t>();
            updatesCastQueue_.FreeTensor(updatesCastLocal);
        }
    }
}

template<typename T, typename U, bool updatesIsScalar, uint32_t scatterOp>
__aicore__ inline void ScatterAddSIMDImpl<T, U, updatesIsScalar, scatterOp>::ProcessVarToWS()
{
    if (GetBlockIdx() >= tilingData_.copyCoreNum) {
        return;
    }
    int64_t tailUbFactor = tilingData_.tailUbFactor;
    int64_t loopSize = tilingData_.blockFactor;
    if (GetBlockIdx() == tilingData_.copyCoreNum - 1) {
        loopSize = tilingData_.tailBlockFactor;
        tailUbFactor = tilingData_.tailCoreTailUbFactor;
    }
    int64_t varOffset = 0;
    for (int64_t varLoopIdx = 0; varLoopIdx < loopSize - 1; ++varLoopIdx) {
        varOffset = varLoopIdx * tilingData_.ubFactor;
        CopyInVar(varOffset, tilingData_.ubFactor);
        LocalTensor<T> varLocal = varInQueue_.DeQue<T>();
        LocalTensor<int32_t> varCastLocal = varCastOutQueue_.AllocTensor<int32_t>();
        CastToInt32<T>(varCastLocal, varLocal, tilingData_.ubFactor);
        varCastOutQueue_.EnQue(varCastLocal);
        varInQueue_.FreeTensor(varLocal);
        CopyOutVarToWS(varOffset, tilingData_.ubFactor);
    }
    varOffset = (loopSize - 1) * tilingData_.ubFactor;
    CopyInVar(varOffset, tailUbFactor);
    LocalTensor<T> varLocal = varInQueue_.DeQue<T>();
    LocalTensor<int32_t> varCastLocal = varCastOutQueue_.AllocTensor<int32_t>();
    CastToInt32<T>(varCastLocal, varLocal, tailUbFactor);
    varCastOutQueue_.EnQue(varCastLocal);
    varInQueue_.FreeTensor(varLocal);
    CopyOutVarToWS(varOffset, tailUbFactor);
}

template<typename T, typename U, bool updatesIsScalar, uint32_t scatterOp>
__aicore__ inline void ScatterAddSIMDImpl<T, U, updatesIsScalar, scatterOp>::ProcessVarFromWS()
{
    if (GetBlockIdx() >= tilingData_.copyCoreNum) {
        return;
    }
    int64_t tailUbFactor = tilingData_.tailUbFactor;
    int64_t loopNum = tilingData_.blockFactor;
    if (GetBlockIdx() == tilingData_.copyCoreNum - 1) {
        loopNum = tilingData_.tailBlockFactor;
        tailUbFactor = tilingData_.tailCoreTailUbFactor;
    }
    int64_t varOffset = 0;
    for (int64_t varLoopIdx = 0; varLoopIdx < loopNum - 1; ++varLoopIdx) {
        varOffset = varLoopIdx * tilingData_.ubFactor;
        CopyInVarFromWS(varOffset, tilingData_.ubFactor);
        LocalTensor<int32_t> varCastLocal = varCastInQueue_.DeQue<int32_t>();
        LocalTensor<T> varLocal = varOutQueue_.AllocTensor<T>();
        CastToOrigin<T>(varLocal, varCastLocal, tilingData_.ubFactor);
        varCastInQueue_.FreeTensor(varCastLocal);
        varOutQueue_.EnQue(varLocal);
        CopyOutVar(varOffset, tilingData_.ubFactor);
    }
    varOffset = (loopNum - 1) * tilingData_.ubFactor;
    CopyInVarFromWS(varOffset, tailUbFactor);
    LocalTensor<int32_t> varCastLocal = varCastInQueue_.DeQue<int32_t>();
    LocalTensor<T> varLocal = varOutQueue_.AllocTensor<T>();
    CastToOrigin<T>(varLocal, varCastLocal, tailUbFactor);
    varCastInQueue_.FreeTensor(varCastLocal);
    varOutQueue_.EnQue(varLocal);
    CopyOutVar(varOffset, tailUbFactor);
}

template<typename T, typename U, bool updatesIsScalar, uint32_t scatterOp>
__aicore__ inline void ScatterAddSIMDImpl<T, U, updatesIsScalar, scatterOp>::Process()
{
    if (GetBlockIdx() >= GetBlockNum()) {
        return;
    }
    if constexpr (platform::IsSupportAtomicAddTypeSIMD<T>()) {
        ProcessAtomicAdd();
    } else {
        ProcessVarToWS();
        SyncAll();
        pipe_.Reset();
        if constexpr (!updatesIsScalar) {
            pipe_.InitBuffer(updatesQueue_, DOUBLE_BUFFER, tilingData_.updatesUbFactor * sizeof(T));
        }
        pipe_.InitBuffer(updatesCastQueue_, DOUBLE_BUFFER, tilingData_.updatesUbFactor * sizeof(int32_t));
        pipe_.InitBuffer(indicesBuf_, tilingData_.indicesUbFactor * sizeof(U));
        ProcessAtomicAdd();
        SyncAll();
        pipe_.Reset();
        pipe_.InitBuffer(varOutQueue_, DOUBLE_BUFFER, tilingData_.ubFactor * sizeof(T));
        pipe_.InitBuffer(varCastInQueue_, DOUBLE_BUFFER, tilingData_.ubFactor * sizeof(int32_t));
        ProcessVarFromWS();
    }
}
}
#endif  // SCATTER_ADD_SIMD_IMPL_H