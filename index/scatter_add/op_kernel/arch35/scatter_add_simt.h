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
 * \file scatter_add_simt.h
 * \brief scatter_add
 */
#ifndef ASCENDC_SCATTER_ADD_H_
#define ASCENDC_SCATTER_ADD_H_

#include "kernel_operator.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"
#include "scatter_add_common.h"

namespace ScatterAdd
{
using namespace AscendC;
using namespace ScatterAddCommon;

constexpr uint32_t VECTOR_LENGTH = platform::GetVRegSize();
constexpr uint32_t VL_B32 = VECTOR_LENGTH / sizeof(uint32_t);
constexpr int64_t DB_BUFFER = 2;
constexpr int64_t MIN_BLOCK_SIZE = 1024;
#ifdef __DAV_FPGA__
constexpr uint32_t THREAD_NUM = 256;
constexpr uint32_t THREAD_NUM_LAUNCH_BOUND = 256;
#else
constexpr uint32_t THREAD_NUM = 1024;
constexpr uint32_t THREAD_NUM_LAUNCH_BOUND = 1024;
#endif

static constexpr MicroAPI::CastTrait castTraitB8B162B32 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                           MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

template <typename IDX_T, typename VAR_T, typename CAST_T, typename ADDR_T, bool isUpdateScalar, uint32_t scatterOp>
class ScatterAddSimt
{
public:
    __aicore__ inline ScatterAddSimt(const ScatterAddTilingData& tilingData, TPipe& pipe)
        : td_(tilingData), pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR var, GM_ADDR indices, GM_ADDR updates, GM_ADDR workspace);
    __aicore__ inline void CopyInputGmToWs(GlobalTensor<VAR_T>& inGm, GlobalTensor<CAST_T>& outGm, int64_t offset,
                                           int64_t dataLen);
    __aicore__ inline void CopyWsToOutputGm(int64_t offset, int64_t dataLen);
    __aicore__ inline void CastToInt32(LocalTensor<CAST_T>& dstLocal, LocalTensor<VAR_T>& srcLocal, uint32_t dataLen);
    __aicore__ inline void CastToOrigin(LocalTensor<VAR_T>& dstLocal, LocalTensor<CAST_T>& srcLocal, uint32_t dataLen);
    __aicore__ inline void CopyVarToWs();
    __aicore__ inline void CopyWsToVar();
    __aicore__ inline void Process();

private:
    GlobalTensor<VAR_T> var_;
    GlobalTensor<IDX_T> indices_;
    GlobalTensor<VAR_T> updates_;
    GlobalTensor<CAST_T> varWorkspaceGm_;
    TQue<QuePosition::VECIN, DB_BUFFER> varQue_;
    TQue<QuePosition::VECIN, DB_BUFFER> varCastQue_;
    TPipe& pipe_;
    const ScatterAddTilingData& td_;

    ADDR_T blockIdx_{0};
    ADDR_T blockNum_{0};
    uint64_t perCoreHandleSize_{0};
    uint64_t usedCoreNum_{0};
    uint64_t curLoopNum_{0};
    uint64_t tailLoopLength_{0};
};

template <typename IDX_T, typename VAR_T, typename CAST_T, typename ADDR_T, bool isUpdateScalar, uint32_t scatterOp>
__aicore__ inline void ScatterAddSimt<IDX_T, VAR_T, CAST_T, ADDR_T, isUpdateScalar, scatterOp>::Init(GM_ADDR var, GM_ADDR indices,
                                                                                          GM_ADDR updates,
                                                                                          GM_ADDR workspace)
{
    blockIdx_ = GetBlockIdx();
    blockNum_ = GetBlockNum();
    var_.SetGlobalBuffer((__gm__ VAR_T*)(var));
    indices_.SetGlobalBuffer((__gm__ IDX_T*)(indices));
    updates_.SetGlobalBuffer((__gm__ VAR_T*)(updates));
    if constexpr (IsSameType<VAR_T, int8_t>::value || IsSameType<VAR_T, uint8_t>::value) {
        varWorkspaceGm_.SetGlobalBuffer((__gm__ CAST_T*)workspace);
        pipe_.InitBuffer(varQue_, DB_BUFFER, td_.ubFactor * sizeof(VAR_T));
        pipe_.InitBuffer(varCastQue_, DB_BUFFER, td_.ubFactor * sizeof(CAST_T));
        perCoreHandleSize_ = td_.perCoreHandleVar;
        usedCoreNum_ = td_.copyCoreNum;
        if (blockIdx_ != usedCoreNum_ - 1) {
            curLoopNum_ = td_.blockFactor;
            tailLoopLength_ = td_.tailUbFactor;
        } else {
            curLoopNum_ = td_.tailBlockFactor;
            tailLoopLength_ = td_.tailCoreTailUbFactor;
        }
    }
}

template <typename IDX_T, typename VAR_T, typename CAST_T, typename ADDR_T, bool isUpdateScalar, uint32_t scatterOp>
__aicore__ inline void ScatterAddSimt<IDX_T, VAR_T, CAST_T, ADDR_T, isUpdateScalar, scatterOp>::CopyInputGmToWs(
    GlobalTensor<VAR_T>& inGm, GlobalTensor<CAST_T>& outGm, int64_t offset, int64_t dataLen)
{
    DataCopyExtParams copyParams = {static_cast<uint16_t>(1), static_cast<uint32_t>(dataLen * sizeof(VAR_T)),
                                    static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPadExtParams<VAR_T> padParams = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
                                             static_cast<VAR_T>(0)};
    LocalTensor<VAR_T> xLocal = varQue_.AllocTensor<VAR_T>();
    DataCopyPad(xLocal, inGm[offset], copyParams, padParams);
    varQue_.EnQue(xLocal);

    event_t eventMte3toV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eventMte3toV);
    WaitFlag<HardEvent::MTE3_V>(eventMte3toV);

    LocalTensor<VAR_T> yLocal = varQue_.DeQue<VAR_T>();
    LocalTensor<CAST_T> castDst = varCastQue_.AllocTensor<CAST_T>();
    CastToInt32(castDst, yLocal, static_cast<uint32_t>(dataLen));
    varCastQue_.EnQue(castDst);

    event_t eventVtoMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventVtoMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventVtoMTE3);

    LocalTensor<CAST_T> dstLocal = varCastQue_.DeQue<CAST_T>();
    copyParams.blockLen = dataLen * sizeof(CAST_T);
    DataCopyPad(outGm[offset], dstLocal, copyParams);
    varQue_.FreeTensor(yLocal);
    varCastQue_.FreeTensor(dstLocal);
}

template <typename IDX_T, typename VAR_T, typename CAST_T, typename ADDR_T, bool isUpdateScalar, uint32_t scatterOp>
__aicore__ inline void ScatterAddSimt<IDX_T, VAR_T, CAST_T, ADDR_T, isUpdateScalar, scatterOp>::CopyWsToOutputGm(int64_t offset,
                                                                                                      int64_t dataLen)
{
    DataCopyExtParams copyParams = {static_cast<uint16_t>(1), static_cast<uint32_t>(dataLen * sizeof(CAST_T)),
                                    static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPadExtParams<CAST_T> padParams = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
                                              static_cast<CAST_T>(0)};
    LocalTensor<CAST_T> xLocal = varCastQue_.AllocTensor<CAST_T>();
    DataCopyPad(xLocal, varWorkspaceGm_[offset], copyParams, padParams);
    varCastQue_.EnQue(xLocal);

    event_t eventMte3toV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eventMte3toV);
    WaitFlag<HardEvent::MTE3_V>(eventMte3toV);

    LocalTensor<CAST_T> yLocal = varCastQue_.DeQue<CAST_T>();
    LocalTensor<VAR_T> castDst = varQue_.AllocTensor<VAR_T>();
    CastToOrigin(castDst, yLocal, static_cast<uint32_t>(dataLen));
    varQue_.EnQue(castDst);

    event_t eventVtoMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventVtoMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventVtoMTE3);

    LocalTensor<VAR_T> dstLocal = varQue_.DeQue<VAR_T>();
    copyParams.blockLen = dataLen * sizeof(VAR_T);
    DataCopyPad(var_[offset], dstLocal, copyParams);
    varCastQue_.FreeTensor(yLocal);
    varQue_.FreeTensor(dstLocal);
}

template <typename IDX_T, typename VAR_T, typename CAST_T, typename ADDR_T, bool isUpdateScalar, uint32_t scatterOp>
__aicore__ inline void ScatterAddSimt<IDX_T, VAR_T, CAST_T, ADDR_T, isUpdateScalar, scatterOp>::CastToInt32(
    LocalTensor<CAST_T>& dstLocal, LocalTensor<VAR_T>& srcLocal, uint32_t dataLen)
{
    __local_mem__ VAR_T* srcAddr = (__local_mem__ VAR_T*)srcLocal.GetPhyAddr();
    __local_mem__ CAST_T* dstAddr = (__local_mem__ CAST_T*)dstLocal.GetPhyAddr();

    uint16_t loopTimes = ops::CeilDiv(dataLen, VL_B32);

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<VAR_T> srcValue;
        MicroAPI::RegTensor<CAST_T> dstValue;
        MicroAPI::MaskReg preg;
        uint32_t sregMask = dataLen;
        for (uint16_t j = 0; j < loopTimes; j++) {
            preg = MicroAPI::UpdateMask<uint32_t>(sregMask);
            MicroAPI::DataCopy<VAR_T, MicroAPI::LoadDist::DIST_UNPACK4_B8>(srcValue, srcAddr + VL_B32 * j);
            MicroAPI::Cast<CAST_T, VAR_T, castTraitB8B162B32>(dstValue, srcValue, preg);
            MicroAPI::DataCopy<CAST_T, MicroAPI::StoreDist::DIST_NORM>(dstAddr + VL_B32 * j, dstValue, preg);
        }
    }
}

template <typename IDX_T, typename VAR_T, typename CAST_T, typename ADDR_T, bool isUpdateScalar, uint32_t scatterOp>
__aicore__ inline void ScatterAddSimt<IDX_T, VAR_T, CAST_T, ADDR_T, isUpdateScalar, scatterOp>::CastToOrigin(
    LocalTensor<VAR_T>& dstLocal, LocalTensor<CAST_T>& srcLocal, uint32_t dataLen)
{
    __local_mem__ CAST_T* srcAddr = (__local_mem__ CAST_T*)srcLocal.GetPhyAddr();
    __local_mem__ VAR_T* dstAddr = (__local_mem__ VAR_T*)dstLocal.GetPhyAddr();

    uint16_t loopTimes = ops::CeilDiv(dataLen, VL_B32);

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<CAST_T> srcValue;
        MicroAPI::MaskReg preg;
        uint32_t sregMask = dataLen;
        for (uint16_t j = 0; j < loopTimes; j++) {
            preg = MicroAPI::UpdateMask<uint32_t>(sregMask);
            MicroAPI::DataCopy<CAST_T, MicroAPI::LoadDist::DIST_NORM>(srcValue, srcAddr + VL_B32 * j);
            MicroAPI::DataCopy<VAR_T, MicroAPI::StoreDist::DIST_PACK4_B32>(dstAddr + VL_B32 * j,
                                                                           (MicroAPI::RegTensor<VAR_T>&)srcValue, preg);
        }
    }
}

template <typename IDX_T, typename VAR_T, typename CAST_T, typename ADDR_T, bool isUpdateScalar, uint32_t scatterOp>
__aicore__ inline void ScatterAddSimt<IDX_T, VAR_T, CAST_T, ADDR_T, isUpdateScalar, scatterOp>::CopyVarToWs()
{
    uint64_t offset = 0;
    for (uint64_t idx = 0; idx < static_cast<uint64_t>(curLoopNum_ - 1); idx++) {
        offset = static_cast<uint64_t>(blockIdx_) * perCoreHandleSize_ + idx * td_.ubFactor;
        CopyInputGmToWs(var_, varWorkspaceGm_, offset, td_.ubFactor);
    }

    if (tailLoopLength_ > 0) {
        offset = static_cast<uint64_t>(blockIdx_) * perCoreHandleSize_ + (curLoopNum_ - 1) * td_.ubFactor;
        CopyInputGmToWs(var_, varWorkspaceGm_, offset, tailLoopLength_);
    }
}

template <typename IDX_T, typename VAR_T, typename CAST_T, typename ADDR_T, bool isUpdateScalar, uint32_t scatterOp>
__aicore__ inline void ScatterAddSimt<IDX_T, VAR_T, CAST_T, ADDR_T, isUpdateScalar, scatterOp>::CopyWsToVar()
{
    uint64_t offset = 0;
    for (uint64_t idx = 0; idx < static_cast<uint64_t>(curLoopNum_ - 1); idx++) {
        offset = static_cast<uint64_t>(blockIdx_) * perCoreHandleSize_ + idx * td_.ubFactor;
        CopyWsToOutputGm(offset, td_.ubFactor);
    }

    if (tailLoopLength_ > 0) {
        offset = static_cast<uint64_t>(blockIdx_) * perCoreHandleSize_ + (curLoopNum_ - 1) * td_.ubFactor;
        CopyWsToOutputGm(offset, tailLoopLength_);
    }
}

template <typename IDX_T, typename VAR_T, typename CAST_T, typename ADDR_T, bool isUpdateScalar, uint32_t scatterOp>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void ScatterAddSimtCompute(
    ADDR_T totalCol, ADDR_T indicesSize, ADDR_T varFirstDimSize, ADDR_T magic, ADDR_T shift, __gm__ VAR_T* var,
    __gm__ IDX_T* indices, __gm__ VAR_T* updates, __gm__ CAST_T* varWorkspaceGm, ADDR_T blockIdx, ADDR_T blockNum)
{
    ADDR_T totalSize = indicesSize * totalCol;
    for (ADDR_T i = blockIdx * Simt::GetThreadNum() + Simt::GetThreadIdx(); i < totalSize;
         i += blockNum * Simt::GetThreadNum()) {
        ADDR_T indiceRow = Simt::UintDiv(i, magic, shift);
        IDX_T varRow = indices[indiceRow];
        if (!(varRow >= 0 && varRow < varFirstDimSize)) {
            continue;
        }

        ADDR_T tailRowIdx = i - indiceRow * totalCol;
        ADDR_T varDataIdx = varRow * totalCol + tailRowIdx;
        if constexpr (scatterOp == ADD) {
            if constexpr (IsSameType<VAR_T, int8_t>::value || IsSameType<VAR_T, uint8_t>::value) {
                if constexpr (isUpdateScalar) {
                    Simt::AtomicAdd(varWorkspaceGm + varDataIdx, static_cast<CAST_T>(updates[0]));
                } else {
                    Simt::AtomicAdd(varWorkspaceGm + varDataIdx, static_cast<CAST_T>(updates[i]));
                }
            } else {
                if constexpr (isUpdateScalar) {
                    Simt::AtomicAdd(var + varDataIdx, static_cast<VAR_T>(updates[0]));
                } else {
                    Simt::AtomicAdd(var + varDataIdx, static_cast<VAR_T>(updates[i]));
                }
            }
        } else if constexpr (scatterOp == SUB) {
            if constexpr (IsSameType<VAR_T, int8_t>::value || IsSameType<VAR_T, uint8_t>::value) {
                if constexpr (isUpdateScalar) {
                    Simt::AtomicAdd(varWorkspaceGm + varDataIdx, static_cast<CAST_T>(-updates[0]));
                } else {
                    Simt::AtomicAdd(varWorkspaceGm + varDataIdx, static_cast<CAST_T>(-updates[i]));
                }
            } else {
                if constexpr (isUpdateScalar) {
                    Simt::AtomicAdd(var + varDataIdx, static_cast<VAR_T>(-updates[0]));
                } else {
                    Simt::AtomicAdd(var + varDataIdx, static_cast<VAR_T>(-updates[i]));
                }
            }
        }
    }
}

template <typename IDX_T, typename VAR_T, typename CAST_T, typename ADDR_T, bool isUpdateScalar, uint32_t scatterOp>
__aicore__ inline void ScatterAddSimt<IDX_T, VAR_T, CAST_T, ADDR_T, isUpdateScalar, scatterOp>::Process()
{
    if constexpr (IsSameType<VAR_T, int8_t>::value || IsSameType<VAR_T, uint8_t>::value) {
        if (blockIdx_ < usedCoreNum_) {
            CopyVarToWs();
        }
        SyncAll();
    }

    ADDR_T totalCol = static_cast<ADDR_T>(td_.varShape[1]);
    ADDR_T indicesSize = static_cast<ADDR_T>(td_.indicesSize);
    ADDR_T varFirstDimSize = static_cast<ADDR_T>(td_.varShape[0]);
    ADDR_T magic = 0;
    ADDR_T shift = 0;
    GetUintDivMagicAndShift(magic, shift, totalCol);

    Simt::VF_CALL<ScatterAddSimtCompute<IDX_T, VAR_T, CAST_T, ADDR_T, isUpdateScalar, scatterOp>>(Simt::Dim3(THREAD_NUM), 
            totalCol, indicesSize, varFirstDimSize, magic, shift, (__gm__ VAR_T*)(var_.GetPhyAddr()),
            (__gm__ IDX_T*)(indices_.GetPhyAddr()), (__gm__ VAR_T*)(updates_.GetPhyAddr()),
            (__gm__ CAST_T*)(varWorkspaceGm_.GetPhyAddr()), blockIdx_, blockNum_);

    if constexpr (IsSameType<VAR_T, int8_t>::value || IsSameType<VAR_T, uint8_t>::value) {
        SyncAll();
        if (blockIdx_ < usedCoreNum_) {
            CopyWsToVar();
        }
    }
}
}  // namespace ScatterAdd

#endif