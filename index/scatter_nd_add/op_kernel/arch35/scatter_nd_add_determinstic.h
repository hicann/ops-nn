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
 * \file scatter_nd_add_deterministic.h
 * \brief deterministic kernel of scatter_nd_add
 */

#ifndef SCATTER_ND_ADD_DETERMINISTIC_H
#define SCATTER_ND_ADD_DETERMINISTIC_H

#include "scatter_nd_add_common.h"
#include "kernel_operator.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"

namespace ScatterNdAdd {
using namespace AscendC;

static constexpr MicroAPI::CastTrait castTraitFp32ToInt32 = {
    MicroAPI::RegLayout::UNKNOWN, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
static constexpr MicroAPI::CastTrait castTraitInt32ToFp32 = {
    MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
static constexpr MicroAPI::CastTrait castTraitFp32ToVarT = {
    MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::SAT, MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

template <typename T, typename U>
class ScatterNdAddDeterministic : public ScatterNdAddBase<T, U> {
public:
    __aicore__ inline ScatterNdAddDeterministic(const ScatterNdAddRegBaseTilingData& tilingData, TPipe& pipe) :
        tilingData_(tilingData), pipe_(pipe) {};
    __aicore__ inline void Init(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR y, GM_ADDR workspace);
    __aicore__ inline void ProcessSplitAfter();

    __aicore__ inline void CopyIndicesAndUpdatesIn(int64_t rowIdx, int64_t colIdx,
                                                   int64_t rowLen, int64_t colLen);
    __aicore__ inline void ComputeSumForSameIndex(int64_t rowLen, int64_t colLen);
    __aicore__ inline void CopySumOutToWs(int64_t rowIdx, int64_t colIdx,
                                          int64_t rowLen, int64_t colLen);
    __aicore__ inline void CopySumAndRValueIn(int64_t rowIdx, int64_t colIdx,
                                              int64_t rowLen, int64_t colLen);
    __aicore__ inline void QuantizeForSum(int64_t rowLen, int64_t colLen);
    __aicore__ inline void CopyQuantizedSumOutToIntWs(int64_t rowIdx, int64_t colIdx,
                                                      int64_t rowLen, int64_t colLen);
    __aicore__ inline void CopyRValueAndIntWsIn(int64_t rowIdx, int64_t colIdx,
                                                int64_t rowLen, int64_t colLen);
    __aicore__ inline void CopyRValueAndIntWsAndIndexInOpti(int64_t rowIdx, int64_t colIdx,
                                                int64_t rowLen, int64_t colLen);
    __aicore__ inline void InverseQuantize(int64_t rowLen, int64_t colLen);
    __aicore__ inline void CopyInverseQuantizedValueOutOpti(int64_t colLen);
    __aicore__ inline void CopyInverseQuantizedValueOut(int64_t rowIdx, int64_t colIdx,
                                                        int64_t rowLen, int64_t colLen);
    __aicore__ inline void ProcessFirstStep();
    __aicore__ inline void ProcessSecondStep();
    __aicore__ inline void ProcessThirdStep();
    __aicore__ inline void Process();

private:
    AscendC::GlobalTensor<T> varGm_;
    AscendC::GlobalTensor<U> indicesGm_;
    AscendC::GlobalTensor<T> updatesGm_;
    AscendC::GlobalTensor<T> yGm_;

    AscendC::GlobalTensor<float> workspaceInitGm_;
    AscendC::GlobalTensor<float> updateSumWsGm_;
    AscendC::GlobalTensor<U> updateSumIdxWsGm_;
    AscendC::GlobalTensor<float> updateRValueWsGm_;
    AscendC::GlobalTensor<uint32_t> sameIdxCountWsGm_;
    AscendC::GlobalTensor<int32_t> sumQuanToIntWsGm_;

    TQue<QuePosition::VECIN, DOUBLE_BUFFER> updatesQue_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> sumQue_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> sumIdxQue_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> rValueQue_;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> quantaSumQue_;

    TQue<QuePosition::VECIN, DOUBLE_BUFFER> sumQuanToIntQue_;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> invQuanDataQue_;

    TPipe& pipe_;
    const ScatterNdAddRegBaseTilingData& tilingData_;

    uint64_t indicesUbLoop_{0};
    int64_t curCoreIndexCount_{0};
    int64_t curCoreVarCount_{0};

    uint64_t strideList[MAX_RANK_COUNT];
    int64_t varNumel_{0};
    int64_t sumWsSize_{0};
    int64_t rValueWsSize_{0};

    uint64_t afterAxisAlignSize_{0};
    uint64_t afterAxisAlignFp32_{0};
    uint64_t uniqueSumIdNum_{0};
};

template <typename T, typename U>
__aicore__ inline void ScatterNdAddDeterministic<T, U>::Init(GM_ADDR var, GM_ADDR indices, GM_ADDR updates,
    GM_ADDR y, GM_ADDR workspace)
{
    varGm_.SetGlobalBuffer((__gm__ T *)(var));
    indicesGm_.SetGlobalBuffer((__gm__ U *)(indices));
    updatesGm_.SetGlobalBuffer((__gm__ T *)(updates));
    yGm_.SetGlobalBuffer((__gm__ T *)(y));
    
    /* 非量化分支 */ 
    this->indicesFactor_ = tilingData_.indicesFactor;
    this->afterAxisFactor_ = tilingData_.afterAxisFactor;
    this->afterAxis_ = tilingData_.afterAxis;
    this->indexRankSize_ = tilingData_.indexRankSize;
    this->eachCoreAfterAxisCount_ = tilingData_.eachCoreAfterAxisCount;
    this->InitBaseBuffer(pipe_, tilingData_.indicesFactor, indices, updates, y);
    if (tilingData_.isSplitAfterAxis == 1) {
        return;
    }

    /* 量化分支 */ 
    afterAxisAlignSize_ = ops::CeilAlign(tilingData_.afterAxis * sizeof(T), UB_AGLIN_VALUE) / sizeof(T);
    afterAxisAlignFp32_ = ops::CeilAlign(tilingData_.afterAxis * sizeof(float), UB_AGLIN_VALUE) / sizeof(float);
    /* 量化分支afterAxisFactor大小是整个尾轴 */
    this->afterAxisFactor_ = afterAxisAlignSize_;

    varNumel_ = tilingData_.varInAxis * tilingData_.afterAxis;
    curCoreIndexCount_ = 
        (GetBlockIdx() != (tilingData_.usedCoreNumBefore - 1) ? tilingData_.eachCoreIndexCount : tilingData_.tailCoreIndexCount);
    curCoreVarCount_ = 
        (GetBlockIdx() != (tilingData_.usedCoreNumAfter - 1) ? tilingData_.eachCoreVarCount : tilingData_.tailCoreVarCount);
    sumWsSize_ = tilingData_.eachCoreIndexCount * tilingData_.afterAxis;

    /* one more col to store counter */
    int64_t varRowCount = tilingData_.varInAxis;
    rValueWsSize_ = varNumel_ + varRowCount;

    /* init workspace for multi cores */
    int64_t wsInitLen = rValueWsSize_ + varNumel_;
    int64_t mainCoreLen = wsInitLen / GetBlockNum();
    int64_t tailCoreLen = wsInitLen - (GetBlockNum() - 1) * mainCoreLen;
    int64_t curCoreLen = (GetBlockIdx() == (GetBlockNum() - 1)) ? tailCoreLen : mainCoreLen;
    int64_t wsInitOfset = GetBlockIdx() * mainCoreLen;
    workspaceInitGm_.SetGlobalBuffer((__gm__ float *)workspace + wsInitOfset, curCoreLen);
    InitGlobalMemory(workspaceInitGm_, curCoreLen, (float)0);

    event_t eventIdMte3ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
    SetFlag<HardEvent::MTE3_S>(eventIdMte3ToS);
    WaitFlag<HardEvent::MTE3_S>(eventIdMte3ToS);
    sameIdxCountWsGm_.SetGlobalBuffer((__gm__ uint32_t *)workspace, varRowCount);
    updateRValueWsGm_.SetGlobalBuffer((__gm__ float *)workspace + varRowCount, varNumel_);
    sumQuanToIntWsGm_.SetGlobalBuffer((__gm__ int32_t *)workspace + rValueWsSize_, varNumel_);
    updateSumWsGm_.SetGlobalBuffer((__gm__ float *)workspace + rValueWsSize_ + varNumel_ +
                                  GetBlockIdx() * sumWsSize_, sumWsSize_);
    auto updateSumStartAddr = (__gm__ float *)workspace + rValueWsSize_ + varNumel_ + GetBlockNum() * sumWsSize_;
    updateSumIdxWsGm_.SetGlobalBuffer((__gm__ U *)updateSumStartAddr +
                                    GetBlockIdx() * tilingData_.eachCoreIndexCount, tilingData_.eachCoreIndexCount);
    
    InitGlobalMemory(updateSumWsGm_, sumWsSize_, (float)(0));
    InitGlobalMemory(updateSumIdxWsGm_, tilingData_.eachCoreIndexCount, (U)(-1));
    AscendC::SyncAll();
}

template <typename T, typename U>
__aicore__ inline void ScatterNdAddDeterministic<T,  U>::ProcessSplitAfter()
{
    if (GetBlockIdx() >= tilingData_.usedCoreNumBefore) {
        return;
    }

    int64_t rowLoopNum = tilingData_.indicesLoopSize;
    int64_t rowMainDataLen = tilingData_.indicesFactor;
    int64_t rowTailDataLen = tilingData_.indiceTailNum;
     
    int64_t colLoopNum = (GetBlockIdx() == tilingData_.usedCoreNumBefore - 1) ? tilingData_.tailUpdateLoopSize :
            tilingData_.updateLoopSize;
    int64_t colMainDataLen = tilingData_.afterAxisFactor;
    int64_t colTailDataLen = (GetBlockIdx() == tilingData_.usedCoreNumBefore - 1) ? tilingData_.tailUpdateAxisNum :
            tilingData_.updateTailNum;

    for (int64_t rowIdx = 0; rowIdx < rowLoopNum - 1; rowIdx++) {
        for (int64_t colIdx = 0; colIdx < colLoopNum; colIdx++) {
            int64_t colDataLen = (colIdx == colLoopNum - 1) ? colTailDataLen : colMainDataLen;
            this->ProcessAfterSingleLoop(rowIdx, colIdx, rowMainDataLen, colDataLen);
        }
    }
    for (int64_t colIdx = 0; colIdx < colLoopNum; colIdx++) {
        int64_t colDataLen = (colIdx == colLoopNum - 1) ? colTailDataLen : colMainDataLen;
        this->ProcessAfterSingleLoop(rowLoopNum - 1, colIdx, rowTailDataLen, colDataLen);
    }
}

template <typename T, typename U>
__aicore__ inline void ScatterNdAddDeterministic<T, U>::CopyIndicesAndUpdatesIn(
    int64_t rowIdx, int64_t colIdx, int64_t rowLen, int64_t colLen)
{
    int64_t indicesOfst = GetBlockIdx() * tilingData_.eachCoreIndexCount + rowIdx * tilingData_.indicesFactor;
    int64_t offset = indicesOfst * tilingData_.afterAxis;

    DataCopyExtParams copyParams = {static_cast<uint16_t>(rowLen), static_cast<uint32_t>(colLen * sizeof(T)),
                                    static_cast<uint32_t>(0),
                                    static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPadExtParams<T> padParams = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0), static_cast<T>(0)};

    LocalTensor<T> updatesLocal = updatesQue_.AllocTensor<T>();
    DataCopyPad(updatesLocal, updatesGm_[offset], copyParams, padParams);
    updatesQue_.EnQue(updatesLocal);

    LocalTensor<U> indicesLocal = this->indicesBuf_.template Get<U>();
    offset = GetBlockIdx() * tilingData_.eachCoreIndexCount * tilingData_.indexRankSize + rowIdx * tilingData_.indicesFactor * tilingData_.indexRankSize;
    this->template CopyIn<U>(indicesLocal, indicesGm_[offset], rowLen * tilingData_.indexRankSize);
}

template <typename T, typename U>
__aicore__ inline void ScatterNdAddDeterministic<T, U>::ComputeSumForSameIndex(int64_t rowLen, int64_t colLen)
{
    LocalTensor<U> indicesLocal = this->indicesBuf_.template Get<U>();
    LocalTensor<U> outOfstLocal = this->outOfstBuf_.template Get<U>();
    LocalTensor<T> updatesLocal = updatesQue_.DeQue<T>();

    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    this->ComputeOutOfset(indicesLocal, outOfstLocal, rowLen, tilingData_.indexRankSize);
    this->ComputeSumForUniqueIdx(outOfstLocal, updatesLocal, rowLen, colLen);

    updatesQue_.FreeTensor(updatesLocal);
}

template <typename T, typename U>
__aicore__ void ScatterNdAddDeterministic<T, U>::CopySumOutToWs(
    int64_t rowIdx, int64_t colIdx, int64_t rowLen, int64_t colLen)
{
    int64_t rowOfset = rowIdx * tilingData_.indicesFactor;
    LocalTensor<U> updateSumIdxLocal = this->updateSumIdxQue_.template DeQue<U>();
    this->template CopyOut<U>(updateSumIdxWsGm_[rowOfset], updateSumIdxLocal, this->uniqueIdNum_);

    int64_t outOfset = rowOfset * tilingData_.afterAxis;
    LocalTensor<float> updateSumLocal = this->updateSumQue_.template DeQue<float>();
    DataCopyExtParams copyParams = {static_cast<uint16_t>(this->uniqueIdNum_),
                                    static_cast<uint32_t>(colLen * sizeof(float)),
                                    static_cast<uint32_t>(0),
                                    static_cast<uint32_t>(0),
                                    static_cast<uint32_t>(0)};
    DataCopyPad(updateSumWsGm_[outOfset], updateSumLocal, copyParams);
    event_t eventIdMte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
    WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);

    Abs(updateSumLocal, updateSumLocal, this->uniqueIdNum_ * afterAxisAlignFp32_);
    for (int32_t i = 0; i < this->uniqueIdNum_; i++) {
        int64_t sumIdx = updateSumIdxLocal(i);
        event_t eventIdSToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(eventIdSToMte3);
        WaitFlag<HardEvent::S_MTE3>(eventIdSToMte3);

        outOfset = static_cast<int64_t>(sumIdx * tilingData_.afterAxis);
        event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
        SetAtomicMax<float>();
        this->template CopyOut<float>(updateRValueWsGm_[outOfset], updateSumLocal[i * afterAxisAlignFp32_], colLen);
        SetAtomicNone();
        AscendC::AtomicAdd(const_cast<__gm__ uint32_t *>(sameIdxCountWsGm_[sumIdx].GetPhyAddr()), uint32_t(1));
    }

    this->updateSumQue_.template FreeTensor(updateSumLocal);
    this->updateSumIdxQue_.template FreeTensor(updateSumIdxLocal);
}

template <typename T, typename U>
__aicore__ inline void ScatterNdAddDeterministic<T, U>::CopySumAndRValueIn(
    int64_t rowIdx, int64_t colIdx, int64_t rowLen, int64_t colLen)
{
    LocalTensor<float> sumLocal = sumQue_.AllocTensor<float>();
    LocalTensor<U> sumIdxLocal = sumIdxQue_.AllocTensor<U>();
    LocalTensor<float> rValueLocal = rValueQue_.AllocTensor<float>();  /* reuse updatelocal for Rvalue */

    int32_t rowOfset = rowIdx * tilingData_.ubQuantaIndxFactor;
    this->template CopyIn<U>(sumIdxLocal, updateSumIdxWsGm_[rowOfset], rowLen);
    int32_t inOfset = rowOfset * tilingData_.afterAxis;

    DataCopyExtParams copyParams = {static_cast<uint16_t>(rowLen), static_cast<uint32_t>(colLen * sizeof(float)),
                                    static_cast<uint32_t>(0),
                                    static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPadExtParams<float> padParams = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0), static_cast<float>(0)};
    DataCopyPad(sumLocal, updateSumWsGm_[inOfset], copyParams, padParams);
    event_t eventIdMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
    WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
    for (int64_t i = 0; i < rowLen; i++) {
        int64_t sumIdx = sumIdxLocal(i);
        if (sumIdx < 0 || sumIdx >= tilingData_.varInAxis) {
            sumIdx = 0;
        }        
        uint32_t count = sameIdxCountWsGm_(sumIdx);
        int64_t rowOfset = sumIdx * tilingData_.afterAxis;
        int64_t localOfset = i * afterAxisAlignFp32_;

        event_t eventIdSToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
        SetFlag<HardEvent::S_MTE2>(eventIdSToMte2);
        WaitFlag<HardEvent::S_MTE2>(eventIdSToMte2);
        this->template CopyIn<float>(rValueLocal[localOfset], updateRValueWsGm_[rowOfset], colLen);

        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        AscendC::Muls(rValueLocal[localOfset], rValueLocal[localOfset], static_cast<float>(count), colLen);
    }

    sumQue_.EnQue(sumLocal);
    sumIdxQue_.EnQue(sumIdxLocal);
    rValueQue_.EnQue(rValueLocal);
}

template <typename T, typename U>
__aicore__ inline void ScatterNdAddDeterministic<T, U>::QuantizeForSum(int64_t rowLen, int64_t colLen)
{
    LocalTensor<float> sumLocal = sumQue_.DeQue<float>();
    LocalTensor<float> rValueLocal = rValueQue_.DeQue<float>();
    LocalTensor<int32_t> quantaSumLocal = quantaSumQue_.AllocTensor<int32_t>();

    __local_mem__ float* sumAddr = (__local_mem__ float*)sumLocal.GetPhyAddr();
    __local_mem__ float* rValueAddr = (__local_mem__ float*)rValueLocal.GetPhyAddr();
    __local_mem__ int32_t* quantaSumAddr = (__local_mem__ int32_t*)quantaSumLocal.GetPhyAddr();

    uint32_t vfLen = platform::GetVRegSize() / sizeof(float);
    uint16_t loopCnt = (colLen + vfLen - 1) / vfLen;
    float scaling = static_cast<float>(1 << 30);
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<float> dataReg;
        AscendC::MicroAPI::RegTensor<float> rValueReg;
        AscendC::MicroAPI::RegTensor<float> resReg;
        AscendC::MicroAPI::RegTensor<float> oneReg;
        AscendC::MicroAPI::RegTensor<int32_t> scaleReg;
        AscendC::MicroAPI::MaskReg maskReg;
        AscendC::MicroAPI::MaskReg cmpReg;
        for (uint16_t rowIdx = 0; rowIdx < static_cast<uint16_t>(rowLen); rowIdx++) {
            uint32_t maskLen = static_cast<uint32_t>(tilingData_.afterAxis);
            AscendC::MicroAPI::Duplicate(oneReg, (float)1);
            for (uint16_t i = 0; i < loopCnt; i++) {
                maskReg = AscendC::MicroAPI::UpdateMask<float>(maskLen);
                auto rowAlignOfst = rowIdx * afterAxisAlignFp32_ + i * vfLen;
                auto sumAddrOfst = sumAddr + rowAlignOfst;
                auto rValueAddrOfst = rValueAddr + rowAlignOfst;
                auto quantaSumAddrOfst = quantaSumAddr + rowAlignOfst;
                DataCopy(dataReg, sumAddrOfst);
                DataCopy(rValueReg, rValueAddrOfst);
                /* 防止除0操作 */
                CompareScalar<float, CMPMODE::EQ>(cmpReg, rValueReg, (float)0, maskReg);
                Select(rValueReg, oneReg, rValueReg, cmpReg);
                Div(resReg, dataReg, rValueReg, maskReg);
                Muls(resReg, resReg, scaling, maskReg);
                AscendC::MicroAPI::Cast<int32_t, float, castTraitFp32ToInt32>(scaleReg, resReg, maskReg);
                DataCopy(quantaSumAddrOfst, scaleReg, maskReg);
            }
        }
    }
    sumQue_.FreeTensor(sumLocal);
    rValueQue_.FreeTensor(rValueLocal);
    quantaSumQue_.EnQue(quantaSumLocal);
}

template <typename T, typename U>
__aicore__ void ScatterNdAddDeterministic<T, U>::CopyQuantizedSumOutToIntWs(
    int64_t rowIdx, int64_t colIdx, int64_t rowLen, int64_t colLen)
{
    LocalTensor<int32_t> quantaSumLocal = quantaSumQue_.DeQue<int32_t>();
    LocalTensor<U> sumIdxLocal = sumIdxQue_.DeQue<U>();

    SetAtomicAdd<int32_t>();
    for (int32_t i = 0; i < rowLen; i++) {
        int64_t sumIdx = sumIdxLocal(i);
        if (sumIdx < 0 || sumIdx >= tilingData_.varInAxis) {
            continue;
        }
        int64_t rowOfset = sumIdxLocal(i) * tilingData_.afterAxis;
        int64_t outOfset = rowOfset;
        event_t eventIdSToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(eventIdSToMte3);
        WaitFlag<HardEvent::S_MTE3>(eventIdSToMte3);
        this->template CopyOut<int32_t>(sumQuanToIntWsGm_[outOfset], quantaSumLocal[i * afterAxisAlignFp32_], colLen);
    }
    SetAtomicNone();

    quantaSumQue_.FreeTensor(quantaSumLocal);
    sumIdxQue_.FreeTensor(sumIdxLocal);
}

template <typename T, typename U>
__aicore__ inline void ScatterNdAddDeterministic<T, U>::CopyRValueAndIntWsIn(
    int64_t rowIdx, int64_t colIdx, int64_t rowLen, int64_t colLen)
{
    LocalTensor<int32_t> sumQuanToIntLocal = sumQuanToIntQue_.AllocTensor<int32_t>();
    LocalTensor<float> rValueLocal = rValueQue_.AllocTensor<float>();

    int64_t rowOfset = GetBlockIdx() * tilingData_.eachCoreVarCount + rowIdx * tilingData_.ubRowFactor;
    int64_t inOfset = rowOfset * tilingData_.afterAxis;

    DataCopyExtParams copyParams = {static_cast<uint16_t>(rowLen), static_cast<uint32_t>(colLen * sizeof(int32_t)),
                                    static_cast<uint32_t>(0),
                                    static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPadExtParams<int32_t> padParams = {false, static_cast<uint8_t>(0),
                                               static_cast<uint8_t>(0), static_cast<int32_t>(0)};
    DataCopyPad(sumQuanToIntLocal, sumQuanToIntWsGm_[inOfset], copyParams, padParams);

    copyParams.blockLen = static_cast<uint32_t>(colLen * sizeof(float));
    DataCopyPadExtParams<float> rValuepadParams = {false, static_cast<uint8_t>(0),
                                                static_cast<uint8_t>(0), static_cast<float>(0)};
    DataCopyPad(rValueLocal, updateRValueWsGm_[inOfset], copyParams, rValuepadParams);

    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

    for (int64_t i = 0; i < rowLen; i++) {
        int64_t localOfset = i * afterAxisAlignFp32_;
        uint64_t gmOfset = rowOfset + i;
        AscendC::DataCacheCleanAndInvalid<uint32_t, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(sameIdxCountWsGm_[gmOfset]);
        uint32_t count = sameIdxCountWsGm_(gmOfset);
        AscendC::Muls(rValueLocal[localOfset], rValueLocal[localOfset], (float)count, colLen);
    }

    sumQuanToIntQue_.EnQue(sumQuanToIntLocal);
    rValueQue_.EnQue(rValueLocal);
}

template <typename T, typename U>
__aicore__ inline void ScatterNdAddDeterministic<T, U>::CopyRValueAndIntWsAndIndexInOpti(
    int64_t rowIdx, int64_t colIdx, int64_t rowLen, int64_t colLen)
{
    LocalTensor<U> sumIdxLocal = sumIdxQue_.AllocTensor<U>();
    LocalTensor<float> rValueLocal = rValueQue_.AllocTensor<float>();
    LocalTensor<int32_t> sumQuanToIntLocal = sumQuanToIntQue_.AllocTensor<int32_t>();

    int64_t outOfset = rowIdx * tilingData_.ubRowOptiFactor;
    this->template CopyIn<U>(sumIdxLocal, updateSumIdxWsGm_[outOfset], rowLen);
    event_t eventIdMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
    WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);

    bool zeroFlag = false;
    uint64_t leftFlag = 0;
    for (int64_t i = 0; i < rowLen; i++) {
        int64_t sumIdx = sumIdxLocal(i);  
        if (sumIdx < 0 || sumIdx >= tilingData_.varInAxis) {
            continue;
        }
        uint32_t count = AscendC::AtomicExch(const_cast<__gm__ uint32_t *>(sameIdxCountWsGm_[sumIdx].GetPhyAddr()), uint32_t(0));
        if (count == 0) {
            continue;
        }
        sumIdxLocal(leftFlag) = sumIdxLocal(i);
        int64_t rowOfset = sumIdx * tilingData_.afterAxis;
        int64_t localOfset = (leftFlag++) * afterAxisAlignFp32_;
        this->template CopyIn<float>(rValueLocal[localOfset], updateRValueWsGm_[rowOfset], colLen);
        this->template CopyIn<int32_t>(sumQuanToIntLocal[localOfset], sumQuanToIntWsGm_[rowOfset], colLen);
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        AscendC::Muls(rValueLocal[localOfset], rValueLocal[localOfset], (float)count, colLen);
    }
    uniqueSumIdNum_ = leftFlag;
    sumIdxQue_.EnQue(sumIdxLocal);
    rValueQue_.EnQue(rValueLocal);
    sumQuanToIntQue_.EnQue(sumQuanToIntLocal);
}

template <typename T, typename U>
__aicore__ inline void ScatterNdAddDeterministic<T, U>::InverseQuantize(int64_t rowLen, int64_t colLen)
{
    LocalTensor<int32_t> sumQuanToIntLocal = sumQuanToIntQue_.DeQue<int32_t>();
    LocalTensor<float> rValueLocal = rValueQue_.DeQue<float>();
    LocalTensor<T> inverseQuantData = invQuanDataQue_.AllocTensor<T>();

    __local_mem__ int32_t* sumQuanToIntAddr = (__ubuf__ int32_t*)sumQuanToIntLocal.GetPhyAddr();
    __local_mem__ float* rValueAddr = (__ubuf__ float*)rValueLocal.GetPhyAddr();
    __local_mem__ T* invQuantDataAddr = (__ubuf__ T*)inverseQuantData.GetPhyAddr();

    uint32_t vfLen = platform::GetVRegSize() / sizeof(int32_t);
    uint16_t loopCnt = (colLen + vfLen - 1) / vfLen;
    float scaleValue = static_cast<float>(1 << 30);
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<int32_t> dataReg;
        AscendC::MicroAPI::RegTensor<float> rReg;
        AscendC::MicroAPI::RegTensor<float> resReg;
        AscendC::MicroAPI::RegTensor<float> scaleReg;
        AscendC::MicroAPI::RegTensor<T> varTReg;
        AscendC::MicroAPI::MaskReg pregLoop;
        AscendC::MicroAPI::MaskReg maskReg;
        maskReg = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::Duplicate(scaleReg, scaleValue, maskReg);
        for (uint16_t rowIdx = 0; rowIdx < static_cast<uint16_t>(rowLen); rowIdx++) {
            uint32_t maskLen = static_cast<uint32_t>(colLen);
            for (uint16_t i = 0; i < loopCnt; ++i) {
                pregLoop = AscendC::MicroAPI::UpdateMask<uint32_t>(maskLen);
                auto rowAlignOfst = rowIdx * afterAxisAlignFp32_ + i * vfLen;
                DataCopy(dataReg, sumQuanToIntAddr + rowAlignOfst);
                DataCopy(rReg, rValueAddr + rowAlignOfst);
                Cast<float, int32_t, castTraitInt32ToFp32>(resReg, dataReg, pregLoop);
                Mul(resReg, resReg, rReg, pregLoop);
                Div(resReg, resReg, scaleReg, pregLoop);

                auto outOfset = invQuantDataAddr + rowIdx * afterAxisAlignSize_ + i * vfLen;
                if constexpr (!std::is_same<float, T>::value) {
                    Cast<T, float, castTraitFp32ToVarT>(varTReg, resReg, pregLoop);
                    DataCopy<T, MicroAPI::StoreDist::DIST_PACK_B32>(outOfset, varTReg, pregLoop);
                } else {
                    DataCopy(outOfset, resReg, pregLoop);
                }
            }
        }
    }
    invQuanDataQue_.EnQue(inverseQuantData);
    sumQuanToIntQue_.FreeTensor(sumQuanToIntLocal);
    rValueQue_.FreeTensor(rValueLocal);
}

template <typename T, typename U>
__aicore__ void ScatterNdAddDeterministic<T, U>::CopyInverseQuantizedValueOutOpti(int64_t colLen)
{
    LocalTensor<T> inverseQuantData = invQuanDataQue_.DeQue<T>();
    LocalTensor<U> sumIdxLocal = sumIdxQue_.DeQue<U>();

    SetAtomicAdd<T>();                                          
    for (uint64_t i = 0; i < this->uniqueSumIdNum_; i++) {
        int64_t sumIdx = sumIdxLocal(i);                            
        int64_t rowOfset = sumIdx * tilingData_.afterAxis;
        event_t eventIdSToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(eventIdSToMte3);
        WaitFlag<HardEvent::S_MTE3>(eventIdSToMte3);
        this->template CopyOut<T>(yGm_[rowOfset], inverseQuantData[i * afterAxisAlignSize_], colLen);
    }
    SetAtomicNone();
    invQuanDataQue_.FreeTensor(inverseQuantData);
    sumIdxQue_.FreeTensor(sumIdxLocal); 
}

template <typename T, typename U>
__aicore__ void ScatterNdAddDeterministic<T, U>::CopyInverseQuantizedValueOut(
    int64_t rowIdx, int64_t colIdx, int64_t rowLen, int64_t colLen)
{
    LocalTensor<T> inverseQuantData = invQuanDataQue_.DeQue<T>();

    int64_t rowOfset = GetBlockIdx() * tilingData_.eachCoreVarCount + rowIdx * tilingData_.ubRowFactor;
    int64_t outOfset = rowOfset * tilingData_.afterAxis;
    DataCopyExtParams copyParams = {static_cast<uint16_t>(rowLen), static_cast<uint32_t>(colLen * sizeof(T)),
                                    static_cast<uint32_t>(0),
                                    static_cast<uint32_t>(0),
                                    static_cast<uint32_t>(0)};
    SetAtomicAdd<T>();
    DataCopyPad(yGm_[outOfset], inverseQuantData, copyParams);
    SetAtomicNone();
    invQuanDataQue_.FreeTensor(inverseQuantData);
}

template <typename T, typename U>
__aicore__ inline void ScatterNdAddDeterministic<T, U>::ProcessFirstStep()
{
    if (GetBlockIdx() >= tilingData_.usedCoreNumBefore) {
        return;
    }
    /* step 1: compute R value */
    pipe_.InitBuffer(updatesQue_, DOUBLE_BUFFER, tilingData_.indicesFactor * afterAxisAlignSize_ * sizeof(T));

    int64_t rowLoopNum = ops::CeilDiv(curCoreIndexCount_, tilingData_.indicesFactor);
    int64_t rowMainDataLen = tilingData_.indicesFactor;
    int64_t rowTailDataLen = curCoreIndexCount_ - tilingData_.indicesFactor * (rowLoopNum - 1);

    for (int32_t rowIdx = 0; rowIdx < rowLoopNum - 1; rowIdx++) {
        CopyIndicesAndUpdatesIn(rowIdx, 0, rowMainDataLen, tilingData_.afterAxis);
        ComputeSumForSameIndex(rowMainDataLen, tilingData_.afterAxis);
        CopySumOutToWs(rowIdx, 0, rowMainDataLen, tilingData_.afterAxis);
    }
    CopyIndicesAndUpdatesIn(rowLoopNum - 1, 0, rowTailDataLen, tilingData_.afterAxis);
    ComputeSumForSameIndex(rowTailDataLen, tilingData_.afterAxis);
    CopySumOutToWs(rowLoopNum - 1, 0, rowTailDataLen, tilingData_.afterAxis);
}

template <typename T, typename U>
__aicore__ inline void ScatterNdAddDeterministic<T, U>::ProcessSecondStep()
{
    if (GetBlockIdx() >= tilingData_.usedCoreNumBefore) {
        return;
    }
    /* step 2: quantize value in sumworkspace */
    pipe_.Reset();
    pipe_.InitBuffer(sumIdxQue_, DOUBLE_BUFFER, tilingData_.ubQuantaIndxFactor * sizeof(U));
    pipe_.InitBuffer(sumQue_, DOUBLE_BUFFER, tilingData_.ubQuantaIndxFactor * afterAxisAlignFp32_ * sizeof(float));
    pipe_.InitBuffer(quantaSumQue_, DOUBLE_BUFFER, tilingData_.ubQuantaIndxFactor * afterAxisAlignFp32_ * sizeof(int32_t));
    /* actual useful len is less equal ubQuantaIndxFactor */
    pipe_.InitBuffer(rValueQue_, DOUBLE_BUFFER, tilingData_.ubQuantaIndxFactor * afterAxisAlignFp32_ * sizeof(float));

    int64_t rowLoopNum = ops::CeilDiv(curCoreIndexCount_, tilingData_.ubQuantaIndxFactor);
    int64_t rowMainDataLen = tilingData_.ubQuantaIndxFactor;
    int64_t rowTailDataLen = curCoreIndexCount_ - tilingData_.ubQuantaIndxFactor * (rowLoopNum - 1);

    for (int32_t rowIdx = 0; rowIdx < rowLoopNum - 1; rowIdx++) {
        CopySumAndRValueIn(rowIdx, 0, rowMainDataLen, tilingData_.afterAxis);
        QuantizeForSum(rowMainDataLen, tilingData_.afterAxis);
        CopyQuantizedSumOutToIntWs(rowIdx, 0, rowMainDataLen, tilingData_.afterAxis);
    }
    CopySumAndRValueIn(rowLoopNum - 1, 0, rowTailDataLen, tilingData_.afterAxis);
    QuantizeForSum(rowTailDataLen, tilingData_.afterAxis);
    CopyQuantizedSumOutToIntWs(rowLoopNum - 1, 0, rowTailDataLen, tilingData_.afterAxis);
}

template <typename T, typename U>
__aicore__ inline void ScatterNdAddDeterministic<T, U>::ProcessThirdStep()
{
    int64_t usedCoreNum = 0;
    if (tilingData_.isOpti == 1) {
        usedCoreNum = tilingData_.usedCoreNumBefore;
    } else {
        usedCoreNum = tilingData_.usedCoreNumAfter;
    }
    if (GetBlockIdx() >= usedCoreNum) {
        return;
    }
    /* step 3: invers quantization and add to var */
    int64_t rowMainDataLen = tilingData_.isOpti == 1 ? tilingData_.ubRowOptiFactor : tilingData_.ubRowFactor;
    int64_t curCoreProcessCount = tilingData_.isOpti == 1 ? curCoreIndexCount_ : curCoreVarCount_;
    int64_t rowLoopNum = ops::CeilDiv(curCoreProcessCount, rowMainDataLen);
    int64_t rowTailDataLen = curCoreProcessCount - rowMainDataLen * (rowLoopNum - 1);
    pipe_.Reset();
    pipe_.InitBuffer(sumQuanToIntQue_, DOUBLE_BUFFER, rowMainDataLen * afterAxisAlignFp32_ * sizeof(int32_t));
    pipe_.InitBuffer(rValueQue_, DOUBLE_BUFFER, rowMainDataLen * afterAxisAlignFp32_ * sizeof(float));
    pipe_.InitBuffer(invQuanDataQue_, DOUBLE_BUFFER, rowMainDataLen * afterAxisAlignSize_ * sizeof(T));
    if (tilingData_.isOpti == 1) {
        pipe_.InitBuffer(sumIdxQue_, DOUBLE_BUFFER, rowMainDataLen * sizeof(U));
        for (uint64_t rowIdx = 0; rowIdx < rowLoopNum - 1; rowIdx++) {
            CopyRValueAndIntWsAndIndexInOpti(rowIdx, 0, rowMainDataLen, tilingData_.afterAxis);
            InverseQuantize(this->uniqueSumIdNum_, tilingData_.afterAxis);
            CopyInverseQuantizedValueOutOpti(tilingData_.afterAxis);
        }
        CopyRValueAndIntWsAndIndexInOpti(rowLoopNum - 1, 0, rowTailDataLen, tilingData_.afterAxis);
        InverseQuantize(this->uniqueSumIdNum_, tilingData_.afterAxis);
        CopyInverseQuantizedValueOutOpti(tilingData_.afterAxis);
    } else {
        for (uint64_t rowIdx = 0; rowIdx < rowLoopNum - 1; rowIdx++) {
            CopyRValueAndIntWsIn(rowIdx, 0, rowMainDataLen, tilingData_.afterAxis);
            InverseQuantize(rowMainDataLen, tilingData_.afterAxis);
            CopyInverseQuantizedValueOut(rowIdx, 0, rowMainDataLen, tilingData_.afterAxis);
        }
        CopyRValueAndIntWsIn(rowLoopNum - 1, 0, rowTailDataLen, tilingData_.afterAxis);
        InverseQuantize(rowTailDataLen, tilingData_.afterAxis);
        CopyInverseQuantizedValueOut(rowLoopNum - 1, 0, rowTailDataLen, tilingData_.afterAxis);
    }
}

template <typename T, typename U>
__aicore__ inline void ScatterNdAddDeterministic<T, U>::Process()
{
    LocalTensor<U> calcLocal = this->strideBuf_.template Get<U>();
    for (int32_t i = 0; i < MAX_RANK_COUNT; i++) {
        calcLocal(i) = tilingData_.strideList[i];
    }

    if (tilingData_.isSplitAfterAxis == 1) {
        ProcessSplitAfter();
        return;
    }

    ProcessFirstStep();
    AscendC::SyncAll();

    ProcessSecondStep();
    AscendC::SyncAll();

    ProcessThirdStep();
}
}
#endif  // SCATTER_ND_ADD_DETERMINISTIC_H