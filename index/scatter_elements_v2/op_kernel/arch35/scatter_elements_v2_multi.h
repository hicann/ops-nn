/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file scatter_elements_v2_multi.h
 * \brief scatter_elements
 */
#ifndef ASCENDC_SCATTER_ELEMENTS_V2_MULTI_H_
#define ASCENDC_SCATTER_ELEMENTS_V2_MULTI_H_

#include "scatter_elements_v2_common.h"
#include "scatter_elements_v2_scalar.h"

namespace ScatterElements {
using namespace AscendC;

template <typename DATA_T, typename IDX_T, typename COMP_T, typename CAST_T, uint32_t REDU, uint16_t RANK, uint16_t DIM>
__simt_callee__ __aicore__ inline void ComputeTensorUpdates(__gm__ IDX_T* indices, __gm__ DATA_T* updates,
                                                            __gm__ DATA_T* y, __gm__ CAST_T* updatesWorkspace,
                                                            __gm__ CAST_T* dataWorkspace, COMP_T count,
                                                            const ScatterOffsetParams<COMP_T, RANK, true>& params)
{
    for (COMP_T i = blockIdx.x * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x) {
        auto offsets = CalcScatterOffsets<RANK, DIM, true>(i, static_cast<COMP_T>(indices[i]), params);
        if constexpr (SCATTER_USE_WORKSPACE<DATA_T, REDU>) {
            UpdateScatterValue<REDU>(dataWorkspace + offsets.output, updatesWorkspace[offsets.updates]);
        } else {
            UpdateScatterValue<REDU>(y + offsets.output, updates[offsets.updates]);
        }
    }
}

template <typename DATA_T, typename IDX_T, typename COMP_T, typename CAST_T, const uint32_t REDU,
          const uint32_t THREAD_USED>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_USED) inline void SimtComputeDim1(__gm__ IDX_T* indices,
                                                                             __gm__ DATA_T* updates, __gm__ DATA_T* y,
                                                                             __gm__ CAST_T* updatesWorkspaceGm,
                                                                             __gm__ CAST_T* xWorkspaceGm,
                                                                             COMP_T allAxis);

template <typename DATA_T, typename IDX_T, typename COMP_T, typename CAST_T, const uint32_t REDU, const uint16_t DIM,
          const uint32_t THREAD_USED>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_USED) inline void SimtComputeDim2(
    __gm__ IDX_T* indices, __gm__ DATA_T* updates, __gm__ DATA_T* y, __gm__ CAST_T* updatesWorkspaceGm,
    __gm__ CAST_T* xWorkspaceGm, __ubuf__ uint64_t* TilingUint64Ub, COMP_T allAxis, COMP_T m0, COMP_T shift0);

template <typename DATA_T, typename IDX_T, typename COMP_T, typename CAST_T, const uint32_t REDU, const uint16_t DIM,
          const uint32_t THREAD_USED>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_USED) inline void SimtComputeDim3(
    __gm__ IDX_T* indices, __gm__ DATA_T* updates, __gm__ DATA_T* y, __gm__ CAST_T* updatesWorkspaceGm,
    __gm__ CAST_T* xWorkspaceGm, __ubuf__ uint64_t* TilingUint64Ub, COMP_T allAxis, COMP_T m0, COMP_T shift0, COMP_T m1,
    COMP_T shift1);

template <typename DATA_T, typename IDX_T, typename COMP_T, typename CAST_T, const uint32_t REDU, const uint16_t DIM>
__simt_vf__ __aicore__ LAUNCH_BOUND(USED_THREAD) inline void SimtComputeDim4(
    __gm__ IDX_T* indices, __gm__ DATA_T* updates, __gm__ DATA_T* y, __gm__ CAST_T* updatesWorkspaceGm,
    __gm__ CAST_T* xWorkspaceGm, __ubuf__ uint64_t* TilingUint64Ub, COMP_T allAxis, COMP_T m0, COMP_T shift0, COMP_T m1,
    COMP_T shift1, COMP_T m2, COMP_T shift2);

template <typename DATA_T, typename IDX_T, typename COMP_T, typename CAST_T, const uint32_t REDU, const uint16_t DIM>
__simt_vf__ __aicore__ LAUNCH_BOUND(USED_THREAD) inline void SimtComputeDim5(
    __gm__ IDX_T* indices, __gm__ DATA_T* updates, __gm__ DATA_T* y, __gm__ CAST_T* updatesWorkspaceGm,
    __gm__ CAST_T* xWorkspaceGm, __ubuf__ uint64_t* TilingUint64Ub, COMP_T allAxis, __ubuf__ COMP_T* params);

template <typename DATA_T, typename IDX_T, typename COMP_T, typename CAST_T, const uint32_t REDU, const uint16_t DIM>
__simt_vf__ __aicore__ LAUNCH_BOUND(USED_THREAD) inline void SimtComputeDim6(
    __gm__ IDX_T* indices, __gm__ DATA_T* updates, __gm__ DATA_T* y, __gm__ CAST_T* updatesWorkspaceGm,
    __gm__ CAST_T* xWorkspaceGm, __ubuf__ uint64_t* TilingUint64Ub, COMP_T allAxis, __ubuf__ COMP_T* params);

template <typename DATA_T, typename IDX_T, typename COMP_T, typename CAST_T, const uint32_t REDU, const uint16_t DIM>
__simt_vf__ __aicore__ LAUNCH_BOUND(USED_THREAD) inline void SimtComputeDim7(
    __gm__ IDX_T* indices, __gm__ DATA_T* updates, __gm__ DATA_T* y, __gm__ CAST_T* updatesWorkspaceGm,
    __gm__ CAST_T* xWorkspaceGm, __ubuf__ uint64_t* TilingUint64Ub, COMP_T allAxis, __ubuf__ COMP_T* params);

template <typename DATA_T, typename IDX_T, typename COMP_T, typename CAST_T, const uint32_t REDU, const uint16_t DIM>
__simt_vf__ __aicore__ LAUNCH_BOUND(USED_THREAD) inline void SimtComputeDim8(
    __gm__ IDX_T* indices, __gm__ DATA_T* updates, __gm__ DATA_T* y, __gm__ CAST_T* updatesWorkspaceGm,
    __gm__ CAST_T* xWorkspaceGm, __ubuf__ uint64_t* TilingUint64Ub, COMP_T allAxis, __ubuf__ COMP_T* params);

template <typename DATA_T, typename IDX_T, typename COMP_T, typename CAST_T, const uint32_t REDU,
          const uint32_t TEMPLATE_V2 = 0>
class KernelScatterElements {
public:
    __aicore__ inline KernelScatterElements(TPipe& pipe) : pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR indices, GM_ADDR updates, GM_ADDR y, GM_ADDR workspace,
                                const ScatterElementsV2AscTilingData* tilingData);
    __aicore__ inline void CopyToWs(GlobalTensor<DATA_T>& inGm, GlobalTensor<CAST_T>& outGm, int64_t offset,
                                    int64_t dataLen);
    __aicore__ inline void CopyWsToY(int64_t offset, int64_t dataLen);
    __aicore__ inline void CastToInt32(LocalTensor<CAST_T>& dstLocal, LocalTensor<DATA_T>& srcLocal, uint32_t dataLen);
    __aicore__ inline void CastToOrigin(LocalTensor<DATA_T>& dstLocal, LocalTensor<CAST_T>& srcLocal, uint32_t dataLen);
    __aicore__ inline void CopyDataToWs();
    __aicore__ inline void CopyUpdatesToWs();
    __aicore__ inline void CopyResToY();
    __aicore__ inline void Process();

private:
    template <auto TENSOR_KERNEL, uint16_t RANK, uint16_t DIM, uint32_t THREADS, typename... DIV_PARAMS>
    __aicore__ inline void LaunchCompute(LocalTensor<uint64_t>& strides, DIV_PARAMS... divisionParams);

    GlobalTensor<DATA_T> y_;
    GlobalTensor<DATA_T> x_;
    GlobalTensor<DATA_T> updates_;
    GlobalTensor<IDX_T> indices_;
    GlobalTensor<CAST_T> xWorkspaceGm_;
    GlobalTensor<CAST_T> updatesWorkspaceGm_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, DB_BUFFER> dataQueue_;
    TQue<QuePosition::VECIN, DB_BUFFER> outQueue_;
    TBuf<TPosition::VECCALC> tilingDataUint64Buf_;
    TBuf<TPosition::VECCALC> paramDim5Buf_;
    TBuf<TPosition::VECCALC> paramDim6Buf_;
    TBuf<TPosition::VECCALC> paramDim7Buf_;
    TBuf<TPosition::VECCALC> paramDim8Buf_;
    TPipe& pipe_;
    const ScatterElementsV2AscTilingData* tilingData_;

    COMP_T allAxis_{1};
    COMP_T blockIdx_;
    COMP_T blockNum_;

    int64_t dataAxis_{1};
    int64_t updatesAxis_{1};

    int64_t loopLength_{0};
    int64_t normBlockData_{0};
    int64_t usedCoreNum_{0};
    int64_t loopNum_{0};
    int64_t tailLoopLength_{0};

    int64_t normBlockData2_{0};
    int64_t usedCoreNum2_{0};
    int64_t loopNum2_{0};
    int64_t tailLoopLength2_{0};
};

template <typename DATA_T, typename IDX_T, typename COMP_T, typename CAST_T, const uint32_t REDU,
          const uint32_t TEMPLATE_V2>
__aicore__ inline void KernelScatterElements<DATA_T, IDX_T, COMP_T, CAST_T, REDU, TEMPLATE_V2>::Init(
    GM_ADDR x, GM_ADDR indices, GM_ADDR updates, GM_ADDR y, GM_ADDR workspace,
    const ScatterElementsV2AscTilingData* tilingData)
{
    tilingData_ = tilingData;
    loopLength_ = tilingData->loopLength;
    allAxis_ = tilingData->allAxis;
    dataAxis_ = tilingData->dataAxis;
    updatesAxis_ = tilingData->updatesAxis;

    x_.SetGlobalBuffer((__gm__ DATA_T*)(x));
    y_.SetGlobalBuffer((__gm__ DATA_T*)(y));
    updates_.SetGlobalBuffer((__gm__ DATA_T*)(updates));
    indices_.SetGlobalBuffer((__gm__ IDX_T*)(indices));
    if constexpr (SCATTER_USE_WORKSPACE<DATA_T, REDU>) {
        xWorkspaceGm_.SetGlobalBuffer((__gm__ CAST_T*)workspace);
        if (!tilingData_->isScalarUpdates) {
            updatesWorkspaceGm_.SetGlobalBuffer(
                (__gm__ CAST_T*)workspace + ops::Aligned(dataAxis_, static_cast<int64_t>(GM_ALIGN / sizeof(CAST_T))));
        }
    }

    blockIdx_ = GetBlockIdx();
    blockNum_ = GetBlockNum();

    normBlockData_ = ops::CeilDiv(dataAxis_, static_cast<int64_t>(blockNum_));
    usedCoreNum_ = ops::CeilDiv(dataAxis_, normBlockData_);
    int64_t tailBlockData = dataAxis_ - (usedCoreNum_ - 1) * normBlockData_;
    int64_t curCoreData = blockIdx_ != (usedCoreNum_ - 1) ? normBlockData_ : tailBlockData;
    loopNum_ = curCoreData / loopLength_;
    tailLoopLength_ = curCoreData - loopNum_ * loopLength_;

    if (!tilingData_->isScalarUpdates) {
        normBlockData2_ = ops::CeilDiv(updatesAxis_, static_cast<int64_t>(blockNum_));
        usedCoreNum2_ = ops::CeilDiv(updatesAxis_, normBlockData2_);
        int64_t tailBlockData2 = updatesAxis_ - (usedCoreNum2_ - 1) * normBlockData2_;
        int64_t curCoreData2 = blockIdx_ != (usedCoreNum2_ - 1) ? normBlockData2_ : tailBlockData2;
        loopNum2_ = curCoreData2 / loopLength_;
        tailLoopLength2_ = curCoreData2 - loopNum2_ * loopLength_;
    }

    if constexpr (SCATTER_USE_WORKSPACE<DATA_T, REDU>) {
        pipe_.InitBuffer(dataQueue_, DB_BUFFER, loopLength_ * sizeof(DATA_T));
        pipe_.InitBuffer(outQueue_, DB_BUFFER, loopLength_ * sizeof(CAST_T));
    }
    pipe_.InitBuffer(tilingDataUint64Buf_, TILING_DATA_UB_NUM * sizeof(uint64_t));
    pipe_.InitBuffer(paramDim5Buf_, PARAM_DIM5_NUM * sizeof(COMP_T));
    pipe_.InitBuffer(paramDim6Buf_, PARAM_UB_NUM * sizeof(COMP_T));
    pipe_.InitBuffer(paramDim7Buf_, PARAM_UB_NUM * sizeof(COMP_T));
    pipe_.InitBuffer(paramDim8Buf_, PARAM_UB_NUM * sizeof(COMP_T));
}

template <typename DATA_T, typename IDX_T, typename COMP_T, typename CAST_T, const uint32_t REDU,
          const uint32_t TEMPLATE_V2>
__aicore__ inline void KernelScatterElements<DATA_T, IDX_T, COMP_T, CAST_T, REDU, TEMPLATE_V2>::CopyToWs(
    GlobalTensor<DATA_T>& inGm, GlobalTensor<CAST_T>& outGm, int64_t offset, int64_t dataLen)
{
    DataCopyExtParams copyParams = {static_cast<uint16_t>(1), static_cast<uint32_t>(dataLen * sizeof(DATA_T)),
                                    static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPadExtParams<DATA_T> padParams = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
                                              static_cast<DATA_T>(0)};
    LocalTensor<DATA_T> xLocal = dataQueue_.AllocTensor<DATA_T>();
    DataCopyPad(xLocal, inGm[offset], copyParams, padParams);
    dataQueue_.EnQue(xLocal);

    event_t eventMte2toV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventMte2toV);
    WaitFlag<HardEvent::MTE2_V>(eventMte2toV);

    LocalTensor<DATA_T> yLocal = dataQueue_.DeQue<DATA_T>();
    LocalTensor<CAST_T> castDst = outQueue_.AllocTensor<CAST_T>();
    if constexpr (IsSameType<CAST_T, half>::value) {
        Cast(castDst, yLocal, RoundMode::CAST_NONE, dataLen);
    } else {
        CastToInt32(castDst, yLocal, static_cast<uint32_t>(dataLen));
    }
    outQueue_.EnQue(castDst);

    event_t eventVtoMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventVtoMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventVtoMTE3);

    LocalTensor<CAST_T> dstLocal = outQueue_.DeQue<CAST_T>();
    copyParams.blockLen = dataLen * sizeof(CAST_T);
    DataCopyPad(outGm[offset], dstLocal, copyParams);
    dataQueue_.FreeTensor(yLocal);
    outQueue_.FreeTensor(dstLocal);
}

template <typename DATA_T, typename IDX_T, typename COMP_T, typename CAST_T, const uint32_t REDU,
          const uint32_t TEMPLATE_V2>
__aicore__ inline void KernelScatterElements<DATA_T, IDX_T, COMP_T, CAST_T, REDU, TEMPLATE_V2>::CopyWsToY(
    int64_t offset, int64_t dataLen)
{
    DataCopyExtParams copyParams = {static_cast<uint16_t>(1), static_cast<uint32_t>(dataLen * sizeof(CAST_T)),
                                    static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPadExtParams<CAST_T> padParams = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
                                              static_cast<CAST_T>(0)};
    LocalTensor<CAST_T> xLocal = outQueue_.AllocTensor<CAST_T>();
    DataCopyPad(xLocal, xWorkspaceGm_[offset], copyParams, padParams);
    outQueue_.EnQue(xLocal);

    event_t eventMte3toV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eventMte3toV);
    WaitFlag<HardEvent::MTE3_V>(eventMte3toV);

    LocalTensor<CAST_T> yLocal = outQueue_.DeQue<CAST_T>();
    LocalTensor<DATA_T> castDst = dataQueue_.AllocTensor<DATA_T>();
    if constexpr (IsSameType<CAST_T, half>::value) {
        Cast(castDst, yLocal, RoundMode::CAST_RINT, dataLen);
    } else {
        CastToOrigin(castDst, yLocal, static_cast<uint32_t>(dataLen));
    }
    dataQueue_.EnQue(castDst);

    event_t eventVtoMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventVtoMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventVtoMTE3);

    LocalTensor<DATA_T> dstLocal = dataQueue_.DeQue<DATA_T>();
    copyParams.blockLen = dataLen * sizeof(DATA_T);
    DataCopyPad(y_[offset], dstLocal, copyParams);
    outQueue_.FreeTensor(yLocal);
    dataQueue_.FreeTensor(dstLocal);
}

template <typename DATA_T, typename IDX_T, typename COMP_T, typename CAST_T, const uint32_t REDU,
          const uint32_t TEMPLATE_V2>
__aicore__ inline void KernelScatterElements<DATA_T, IDX_T, COMP_T, CAST_T, REDU, TEMPLATE_V2>::CastToInt32(
    LocalTensor<CAST_T>& dstLocal, LocalTensor<DATA_T>& srcLocal, uint32_t dataLen)
{
    uint16_t loopTimes = ops::CeilDiv(dataLen, VL_B32);
    CastToInt32Vf<DATA_T, CAST_T>((__ubuf__ DATA_T*)srcLocal.GetPhyAddr(), (__ubuf__ CAST_T*)dstLocal.GetPhyAddr(),
                                  dataLen, loopTimes);
}

template <typename DATA_T, typename IDX_T, typename COMP_T, typename CAST_T, const uint32_t REDU,
          const uint32_t TEMPLATE_V2>
__aicore__ inline void KernelScatterElements<DATA_T, IDX_T, COMP_T, CAST_T, REDU, TEMPLATE_V2>::CastToOrigin(
    LocalTensor<DATA_T>& dstLocal, LocalTensor<CAST_T>& srcLocal, uint32_t dataLen)
{
    uint16_t loopTimes = ops::CeilDiv(dataLen, VL_B32);
    CastToOriginVf<DATA_T, CAST_T>((__ubuf__ CAST_T*)srcLocal.GetPhyAddr(), (__ubuf__ DATA_T*)dstLocal.GetPhyAddr(),
                                   dataLen, loopTimes);
}

template <typename DATA_T, typename IDX_T, typename COMP_T, typename CAST_T, const uint32_t REDU,
          const uint32_t TEMPLATE_V2>
__aicore__ inline void KernelScatterElements<DATA_T, IDX_T, COMP_T, CAST_T, REDU, TEMPLATE_V2>::CopyDataToWs()
{
    int64_t offset = 0;
    for (int64_t idx = 0; idx < loopNum_; idx++) {
        offset = static_cast<int64_t>(blockIdx_) * normBlockData_ + idx * loopLength_;
        CopyToWs(x_, xWorkspaceGm_, offset, loopLength_);
    }

    if (tailLoopLength_ > 0) {
        offset = static_cast<int64_t>(blockIdx_) * normBlockData_ + loopNum_ * loopLength_;
        CopyToWs(x_, xWorkspaceGm_, offset, tailLoopLength_);
    }
}

template <typename DATA_T, typename IDX_T, typename COMP_T, typename CAST_T, const uint32_t REDU,
          const uint32_t TEMPLATE_V2>
__aicore__ inline void KernelScatterElements<DATA_T, IDX_T, COMP_T, CAST_T, REDU, TEMPLATE_V2>::CopyUpdatesToWs()
{
    int64_t offset = 0;
    for (int64_t idx = 0; idx < loopNum2_; idx++) {
        offset = static_cast<int64_t>(blockIdx_) * normBlockData2_ + idx * loopLength_;
        CopyToWs(updates_, updatesWorkspaceGm_, offset, loopLength_);
    }

    if (tailLoopLength2_ > 0) {
        offset = static_cast<int64_t>(blockIdx_) * normBlockData2_ + loopNum2_ * loopLength_;
        CopyToWs(updates_, updatesWorkspaceGm_, offset, tailLoopLength2_);
    }
}

template <typename DATA_T, typename IDX_T, typename COMP_T, typename CAST_T, const uint32_t REDU,
          const uint32_t TEMPLATE_V2>
__aicore__ inline void KernelScatterElements<DATA_T, IDX_T, COMP_T, CAST_T, REDU, TEMPLATE_V2>::CopyResToY()
{
    int64_t offset = 0;
    for (int64_t idx = 0; idx < loopNum_; idx++) {
        offset = static_cast<int64_t>(blockIdx_) * normBlockData_ + idx * loopLength_;
        CopyWsToY(offset, loopLength_);
    }

    if (tailLoopLength_ > 0) {
        offset = static_cast<int64_t>(blockIdx_) * normBlockData_ + loopNum_ * loopLength_;
        CopyWsToY(offset, tailLoopLength_);
    }
}

template <typename DATA_T, typename IDX_T, typename COMP_T, typename CAST_T, const uint32_t REDU,
          const uint32_t THREAD_USED>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_USED) inline void SimtComputeDim1(__gm__ IDX_T* indices,
                                                                             __gm__ DATA_T* updates, __gm__ DATA_T* y,
                                                                             __gm__ CAST_T* updatesWorkspaceGm,
                                                                             __gm__ CAST_T* xWorkspaceGm,
                                                                             COMP_T allAxis)
{
    ScatterOffsetParams<COMP_T, 1, true> offsetParams;
    LoadScatterStrides(offsetParams, nullptr);
    ComputeTensorUpdates<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 1, 0>(indices, updates, y, updatesWorkspaceGm,
                                                                    xWorkspaceGm, allAxis, offsetParams);
}

template <typename DATA_T, typename IDX_T, typename COMP_T, typename CAST_T, const uint32_t REDU, const uint16_t DIM,
          const uint32_t THREAD_USED>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_USED) inline void SimtComputeDim2(
    __gm__ IDX_T* indices, __gm__ DATA_T* updates, __gm__ DATA_T* y, __gm__ CAST_T* updatesWorkspaceGm,
    __gm__ CAST_T* xWorkspaceGm, __ubuf__ uint64_t* TilingUint64Ub, COMP_T allAxis, COMP_T m0, COMP_T shift0)
{
    ScatterOffsetParams<COMP_T, 2, true> offsetParams;
    LoadScatterStrides(offsetParams, TilingUint64Ub);
    offsetParams.magic[0] = m0;
    offsetParams.shift[0] = shift0;
    ComputeTensorUpdates<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 2, DIM>(indices, updates, y, updatesWorkspaceGm,
                                                                      xWorkspaceGm, allAxis, offsetParams);
}

template <typename DATA_T, typename IDX_T, typename COMP_T, typename CAST_T, const uint32_t REDU, const uint16_t DIM,
          const uint32_t THREAD_USED>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_USED) inline void SimtComputeDim3(
    __gm__ IDX_T* indices, __gm__ DATA_T* updates, __gm__ DATA_T* y, __gm__ CAST_T* updatesWorkspaceGm,
    __gm__ CAST_T* xWorkspaceGm, __ubuf__ uint64_t* TilingUint64Ub, COMP_T allAxis, COMP_T m0, COMP_T shift0, COMP_T m1,
    COMP_T shift1)
{
    ScatterOffsetParams<COMP_T, 3, true> offsetParams;
    LoadScatterStrides(offsetParams, TilingUint64Ub);
    offsetParams.magic[0] = m0;
    offsetParams.shift[0] = shift0;
    offsetParams.magic[1] = m1;
    offsetParams.shift[1] = shift1;
    ComputeTensorUpdates<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 3, DIM>(indices, updates, y, updatesWorkspaceGm,
                                                                      xWorkspaceGm, allAxis, offsetParams);
}

template <typename DATA_T, typename IDX_T, typename COMP_T, typename CAST_T, const uint32_t REDU, const uint16_t DIM>
__simt_vf__ __aicore__ LAUNCH_BOUND(USED_THREAD) inline void SimtComputeDim4(
    __gm__ IDX_T* indices, __gm__ DATA_T* updates, __gm__ DATA_T* y, __gm__ CAST_T* updatesWorkspaceGm,
    __gm__ CAST_T* xWorkspaceGm, __ubuf__ uint64_t* TilingUint64Ub, COMP_T allAxis, COMP_T m0, COMP_T shift0, COMP_T m1,
    COMP_T shift1, COMP_T m2, COMP_T shift2)
{
    ScatterOffsetParams<COMP_T, 4, true> offsetParams;
    LoadScatterStrides(offsetParams, TilingUint64Ub);
    offsetParams.magic[0] = m0;
    offsetParams.shift[0] = shift0;
    offsetParams.magic[1] = m1;
    offsetParams.shift[1] = shift1;
    offsetParams.magic[2] = m2;
    offsetParams.shift[2] = shift2;
    ComputeTensorUpdates<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 4, DIM>(indices, updates, y, updatesWorkspaceGm,
                                                                      xWorkspaceGm, allAxis, offsetParams);
}

template <typename DATA_T, typename IDX_T, typename COMP_T, typename CAST_T, const uint32_t REDU, const uint16_t DIM>
__simt_vf__ __aicore__ LAUNCH_BOUND(USED_THREAD) inline void SimtComputeDim5(
    __gm__ IDX_T* indices, __gm__ DATA_T* updates, __gm__ DATA_T* y, __gm__ CAST_T* updatesWorkspaceGm,
    __gm__ CAST_T* xWorkspaceGm, __ubuf__ uint64_t* TilingUint64Ub, COMP_T allAxis, __ubuf__ COMP_T* params)
{
    ScatterOffsetParams<COMP_T, 5, true> offsetParams;
    LoadScatterStrides(offsetParams, TilingUint64Ub);
#pragma unroll
    for (uint16_t d = 0; d < 4; ++d) {
        offsetParams.magic[d] = params[d];
        offsetParams.shift[d] = params[4 + d];
    }
    ComputeTensorUpdates<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 5, DIM>(indices, updates, y, updatesWorkspaceGm,
                                                                      xWorkspaceGm, allAxis, offsetParams);
}

template <typename DATA_T, typename IDX_T, typename COMP_T, typename CAST_T, const uint32_t REDU, const uint16_t DIM>
__simt_vf__ __aicore__ LAUNCH_BOUND(USED_THREAD) inline void SimtComputeDim6(
    __gm__ IDX_T* indices, __gm__ DATA_T* updates, __gm__ DATA_T* y, __gm__ CAST_T* updatesWorkspaceGm,
    __gm__ CAST_T* xWorkspaceGm, __ubuf__ uint64_t* TilingUint64Ub, COMP_T allAxis, __ubuf__ COMP_T* params)
{
    ScatterOffsetParams<COMP_T, 6, true> offsetParams;
    LoadScatterStrides(offsetParams, TilingUint64Ub);
#pragma unroll
    for (uint16_t d = 0; d < 5; ++d) {
        offsetParams.magic[d] = params[d];
        offsetParams.shift[d] = params[5 + d];
    }
    ComputeTensorUpdates<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 6, DIM>(indices, updates, y, updatesWorkspaceGm,
                                                                      xWorkspaceGm, allAxis, offsetParams);
}

template <typename DATA_T, typename IDX_T, typename COMP_T, typename CAST_T, const uint32_t REDU, const uint16_t DIM>
__simt_vf__ __aicore__ LAUNCH_BOUND(USED_THREAD) inline void SimtComputeDim7(
    __gm__ IDX_T* indices, __gm__ DATA_T* updates, __gm__ DATA_T* y, __gm__ CAST_T* updatesWorkspaceGm,
    __gm__ CAST_T* xWorkspaceGm, __ubuf__ uint64_t* TilingUint64Ub, COMP_T allAxis, __ubuf__ COMP_T* params)
{
    ScatterOffsetParams<COMP_T, 7, true> offsetParams;
    LoadScatterStrides(offsetParams, TilingUint64Ub);
#pragma unroll
    for (uint16_t d = 0; d < 6; ++d) {
        offsetParams.magic[d] = params[d];
        offsetParams.shift[d] = params[6 + d];
    }
    ComputeTensorUpdates<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 7, DIM>(indices, updates, y, updatesWorkspaceGm,
                                                                      xWorkspaceGm, allAxis, offsetParams);
}

template <typename DATA_T, typename IDX_T, typename COMP_T, typename CAST_T, const uint32_t REDU, const uint16_t DIM>
__simt_vf__ __aicore__ LAUNCH_BOUND(USED_THREAD) inline void SimtComputeDim8(
    __gm__ IDX_T* indices, __gm__ DATA_T* updates, __gm__ DATA_T* y, __gm__ CAST_T* updatesWorkspaceGm,
    __gm__ CAST_T* xWorkspaceGm, __ubuf__ uint64_t* TilingUint64Ub, COMP_T allAxis, __ubuf__ COMP_T* params)
{
    ScatterOffsetParams<COMP_T, 8, true> offsetParams;
    LoadScatterStrides(offsetParams, TilingUint64Ub);
#pragma unroll
    for (uint16_t d = 0; d < 7; ++d) {
        offsetParams.magic[d] = params[d];
        offsetParams.shift[d] = params[7 + d];
    }
    ComputeTensorUpdates<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 8, DIM>(indices, updates, y, updatesWorkspaceGm,
                                                                      xWorkspaceGm, allAxis, offsetParams);
}

template <typename DATA_T, typename IDX_T, typename COMP_T, typename CAST_T, const uint32_t REDU,
          const uint32_t TEMPLATE_V2>
template <auto TENSOR_KERNEL, uint16_t RANK, uint16_t DIM, uint32_t THREADS, typename... DIV_PARAMS>
__aicore__ inline void KernelScatterElements<DATA_T, IDX_T, COMP_T, CAST_T, REDU, TEMPLATE_V2>::LaunchCompute(
    LocalTensor<uint64_t>& strides, DIV_PARAMS... divisionParams)
{
    auto indices = (__gm__ IDX_T*)indices_.GetPhyAddr();
    auto updates = (__gm__ DATA_T*)updates_.GetPhyAddr();
    auto output = (__gm__ DATA_T*)y_.GetPhyAddr();
    auto strideAddr = (__ubuf__ uint64_t*)strides.GetPhyAddr();
    __gm__ CAST_T* dataWorkspace = nullptr;
    if constexpr (SCATTER_USE_WORKSPACE<DATA_T, REDU>) {
        dataWorkspace = (__gm__ CAST_T*)xWorkspaceGm_.GetPhyAddr();
    }
    if (tilingData_->isScalarUpdates) {
        __ubuf__ COMP_T* scalarParams = nullptr;
        if constexpr (RANK >= DIM_5) {
            __ubuf__ COMP_T* packed[] = {divisionParams...};
            scalarParams = packed[0];
        } else if constexpr (RANK > DIM_1) {
            COMP_T packed[] = {divisionParams...};
            auto paramsUb = paramDim8Buf_.Get<COMP_T>();
            for (uint16_t d = 0; d < RANK - 1; ++d) {
                paramsUb.SetValue(d, packed[2 * d]);
                paramsUb.SetValue(RANK - 1 + d, packed[2 * d + 1]);
            }
            DataSyncBarrier<MemDsbT::UB>();
            scalarParams = (__ubuf__ COMP_T*)paramsUb.GetPhyAddr();
        }
        // Materialize dimensions at the call site to avoid reusing a hoisted VF launch configuration.
        volatile uint32_t scalarThreadCount = SCALAR_THREAD_NUM;
        asc_vf_call<SimtComputeScalar<DATA_T, IDX_T, COMP_T, CAST_T, REDU, RANK, DIM>>(
            dim3(scalarThreadCount), indices, updates, output, dataWorkspace, strideAddr, allAxis_, scalarParams);
    } else {
        __gm__ CAST_T* updatesWorkspace = nullptr;
        if constexpr (SCATTER_USE_WORKSPACE<DATA_T, REDU>) {
            updatesWorkspace = (__gm__ CAST_T*)updatesWorkspaceGm_.GetPhyAddr();
        }
        if constexpr (RANK == DIM_1) {
            asc_vf_call<TENSOR_KERNEL>(dim3(THREADS), indices, updates, output, updatesWorkspace, dataWorkspace,
                                       allAxis_);
        } else {
            asc_vf_call<TENSOR_KERNEL>(dim3(THREADS), indices, updates, output, updatesWorkspace, dataWorkspace,
                                       strideAddr, allAxis_, divisionParams...);
        }
    }
}

template <typename DATA_T, typename IDX_T, typename COMP_T, typename CAST_T, const uint32_t REDU,
          const uint32_t TEMPLATE_V2>
__aicore__ inline void KernelScatterElements<DATA_T, IDX_T, COMP_T, CAST_T, REDU, TEMPLATE_V2>::Process()
{
    if (tilingData_->allAxis == 0) {
        return;
    }

    if constexpr (SCATTER_USE_WORKSPACE<DATA_T, REDU>) {
        if (blockIdx_ < usedCoreNum_) {
            CopyDataToWs();
        }
        if (!tilingData_->isScalarUpdates && blockIdx_ < usedCoreNum2_) {
            CopyUpdatesToWs();
        }
    }
    SyncAll();

    LocalTensor<uint64_t> TilingUint64Ub = tilingDataUint64Buf_.Get<uint64_t>();
    LocalTensor<COMP_T> ParamDim5Ub = paramDim5Buf_.Get<COMP_T>();
    LocalTensor<COMP_T> ParamDim6Ub = paramDim6Buf_.Get<COMP_T>();
    LocalTensor<COMP_T> ParamDim7Ub = paramDim7Buf_.Get<COMP_T>();
    LocalTensor<COMP_T> ParamDim8Ub = paramDim8Buf_.Get<COMP_T>();

    const uint64_t* tilingUint64 = reinterpret_cast<const uint64_t*>(tilingData_);
    for (uint32_t i = 0; i < TILING_DATA_UINT64_NUM; i++) {
        TilingUint64Ub.SetValue(i, tilingUint64[i]);
    }
    DataSyncBarrier<MemDsbT::UB>();

    // Limit the wider launch to the verified 32-bit payload/offset path.
    // 64-bit offsets spill registers at 2048 threads in the rank-2/3 kernels.
    // Keep the launch bound and runtime thread count identical.
    constexpr bool useWideNone = TEMPLATE_V2 == 1 && REDU == REDU_NONE && sizeof(DATA_T) == sizeof(uint32_t) &&
                                 sizeof(IDX_T) == sizeof(int64_t) && sizeof(COMP_T) == sizeof(uint32_t);
    constexpr uint32_t wideThreadCount = useWideNone ? 2048 : USED_THREAD1024;
    bool useWideThreads = false;
    if (tilingData_->rank >= DIM_1 && tilingData_->rank <= DIM_3) {
        if (tilingData_->rank == DIM_1) {
            if constexpr (useWideNone) {
                useWideThreads = true;
            } else {
                useWideThreads = tilingData_->rank == tilingData_->dim + 1 && dataAxis_ > THREAD_SUM;
            }
        } else {
            // Keep the existing 512/1024 policy for rank-2 and rank-3 kernels.
            useWideThreads = tilingData_->rank == tilingData_->dim + 1 && dataAxis_ > THREAD_SUM;
        }
    }

    if (tilingData_->rank == DIM_1) {
        if (useWideThreads) {
            LaunchCompute<SimtComputeDim1<DATA_T, IDX_T, COMP_T, CAST_T, REDU, wideThreadCount>, DIM_1, 0,
                          wideThreadCount>(TilingUint64Ub);
        } else {
            LaunchCompute<SimtComputeDim1<DATA_T, IDX_T, COMP_T, CAST_T, REDU, USED_THREAD>, DIM_1, 0, USED_THREAD>(
                TilingUint64Ub);
        }
    } else if (tilingData_->rank == DIM_2) {
        COMP_T m0 = 1;
        COMP_T shift0 = 1;
        GetUintDivMagicAndShift(m0, shift0, static_cast<COMP_T>(tilingData_->indicesStride[0]));
        if (tilingData_->dim == 0) {
            if (useWideThreads) {
                LaunchCompute<SimtComputeDim2<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 0, USED_THREAD1024>, DIM_2, 0,
                              USED_THREAD1024>(TilingUint64Ub, m0, shift0);
            } else {
                LaunchCompute<SimtComputeDim2<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 0, USED_THREAD>, DIM_2, 0,
                              USED_THREAD>(TilingUint64Ub, m0, shift0);
            }
        } else {
            if (useWideThreads) {
                LaunchCompute<SimtComputeDim2<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 1, USED_THREAD1024>, DIM_2, 1,
                              USED_THREAD1024>(TilingUint64Ub, m0, shift0);
            } else {
                LaunchCompute<SimtComputeDim2<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 1, USED_THREAD>, DIM_2, 1,
                              USED_THREAD>(TilingUint64Ub, m0, shift0);
            }
        }
    } else if (tilingData_->rank == DIM_3) {
        COMP_T m_[2] = {1, 1};
        COMP_T shift_[2] = {1, 1};
        GetUintDivMagicAndShift(m_[0], shift_[0], static_cast<COMP_T>(tilingData_->indicesStride[0]));
        GetUintDivMagicAndShift(m_[DIM_1], shift_[DIM_1], static_cast<COMP_T>(tilingData_->indicesStride[DIM_1]));
        if (tilingData_->dim == 0) {
            if (useWideThreads) {
                LaunchCompute<SimtComputeDim3<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 0, USED_THREAD1024>, DIM_3, 0,
                              USED_THREAD1024>(TilingUint64Ub, m_[0], shift_[0], m_[DIM_1], shift_[DIM_1]);
            } else {
                LaunchCompute<SimtComputeDim3<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 0, USED_THREAD>, DIM_3, 0,
                              USED_THREAD>(TilingUint64Ub, m_[0], shift_[0], m_[DIM_1], shift_[DIM_1]);
            }
        } else if (tilingData_->dim == 1) {
            if (useWideThreads) {
                LaunchCompute<SimtComputeDim3<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 1, USED_THREAD1024>, DIM_3, 1,
                              USED_THREAD1024>(TilingUint64Ub, m_[0], shift_[0], m_[DIM_1], shift_[DIM_1]);
            } else {
                LaunchCompute<SimtComputeDim3<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 1, USED_THREAD>, DIM_3, 1,
                              USED_THREAD>(TilingUint64Ub, m_[0], shift_[0], m_[DIM_1], shift_[DIM_1]);
            }
        } else {
            if (useWideThreads) {
                LaunchCompute<SimtComputeDim3<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 2, USED_THREAD1024>, DIM_3, 2,
                              USED_THREAD1024>(TilingUint64Ub, m_[0], shift_[0], m_[DIM_1], shift_[DIM_1]);
            } else {
                LaunchCompute<SimtComputeDim3<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 2, USED_THREAD>, DIM_3, 2,
                              USED_THREAD>(TilingUint64Ub, m_[0], shift_[0], m_[DIM_1], shift_[DIM_1]);
            }
        }
    } else if (tilingData_->rank == DIM_4) {
        COMP_T m_[3] = {1, 1, 1};
        COMP_T shift_[3] = {1, 1, 1};
        GetUintDivMagicAndShift(m_[0], shift_[0], static_cast<COMP_T>(tilingData_->indicesStride[0]));
        GetUintDivMagicAndShift(m_[DIM_1], shift_[DIM_1], static_cast<COMP_T>(tilingData_->indicesStride[DIM_1]));
        GetUintDivMagicAndShift(m_[DIM_2], shift_[DIM_2], static_cast<COMP_T>(tilingData_->indicesStride[DIM_2]));
        if (tilingData_->dim == 0) {
            LaunchCompute<SimtComputeDim4<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 0>, DIM_4, 0, USED_THREAD>(
                TilingUint64Ub, m_[0], shift_[0], m_[DIM_1], shift_[DIM_1], m_[DIM_2], shift_[DIM_2]);
        } else if (tilingData_->dim == 1) {
            LaunchCompute<SimtComputeDim4<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 1>, DIM_4, 1, USED_THREAD>(
                TilingUint64Ub, m_[0], shift_[0], m_[DIM_1], shift_[DIM_1], m_[DIM_2], shift_[DIM_2]);
        } else if (tilingData_->dim == 2) {
            LaunchCompute<SimtComputeDim4<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 2>, DIM_4, 2, USED_THREAD>(
                TilingUint64Ub, m_[0], shift_[0], m_[DIM_1], shift_[DIM_1], m_[DIM_2], shift_[DIM_2]);
        } else {
            LaunchCompute<SimtComputeDim4<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 3>, DIM_4, 3, USED_THREAD>(
                TilingUint64Ub, m_[0], shift_[0], m_[DIM_1], shift_[DIM_1], m_[DIM_2], shift_[DIM_2]);
        }
    } else if (tilingData_->rank == DIM_5) {
        ScatterElementsQuickDivParamDim5<COMP_T> params;
        GetUintDivMagicAndShift(params.m0, params.shift0, static_cast<COMP_T>(tilingData_->indicesStride[0]));
        GetUintDivMagicAndShift(params.m1, params.shift1, static_cast<COMP_T>(tilingData_->indicesStride[DIM_1]));
        GetUintDivMagicAndShift(params.m2, params.shift2, static_cast<COMP_T>(tilingData_->indicesStride[DIM_2]));
        GetUintDivMagicAndShift(params.m3, params.shift3, static_cast<COMP_T>(tilingData_->indicesStride[DIM_3]));
        const COMP_T* paramsDim5 = reinterpret_cast<const COMP_T*>(&params);
        for (uint32_t i = 0; i < PARAM_DIM5_NUM; i++) {
            ParamDim5Ub.SetValue(i, paramsDim5[i]);
        }
        DataSyncBarrier<MemDsbT::UB>();
        if (tilingData_->dim == 0) {
            LaunchCompute<SimtComputeDim5<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 0>, DIM_5, 0, USED_THREAD>(
                TilingUint64Ub, (__ubuf__ COMP_T*)(ParamDim5Ub.GetPhyAddr()));
        } else if (tilingData_->dim == 1) {
            LaunchCompute<SimtComputeDim5<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 1>, DIM_5, 1, USED_THREAD>(
                TilingUint64Ub, (__ubuf__ COMP_T*)(ParamDim5Ub.GetPhyAddr()));
        } else if (tilingData_->dim == 2) {
            LaunchCompute<SimtComputeDim5<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 2>, DIM_5, 2, USED_THREAD>(
                TilingUint64Ub, (__ubuf__ COMP_T*)(ParamDim5Ub.GetPhyAddr()));
        } else if (tilingData_->dim == 3) {
            LaunchCompute<SimtComputeDim5<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 3>, DIM_5, 3, USED_THREAD>(
                TilingUint64Ub, (__ubuf__ COMP_T*)(ParamDim5Ub.GetPhyAddr()));
        } else {
            LaunchCompute<SimtComputeDim5<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 4>, DIM_5, 4, USED_THREAD>(
                TilingUint64Ub, (__ubuf__ COMP_T*)(ParamDim5Ub.GetPhyAddr()));
        }
    } else if (tilingData_->rank == DIM_6) {
        ScatterElementsQuickDivParamDim6<COMP_T> params;
        GetUintDivMagicAndShift(params.m0, params.shift0, static_cast<COMP_T>(tilingData_->indicesStride[0]));
        GetUintDivMagicAndShift(params.m1, params.shift1, static_cast<COMP_T>(tilingData_->indicesStride[DIM_1]));
        GetUintDivMagicAndShift(params.m2, params.shift2, static_cast<COMP_T>(tilingData_->indicesStride[DIM_2]));
        GetUintDivMagicAndShift(params.m3, params.shift3, static_cast<COMP_T>(tilingData_->indicesStride[DIM_3]));
        GetUintDivMagicAndShift(params.m4, params.shift4, static_cast<COMP_T>(tilingData_->indicesStride[DIM_4]));
        const COMP_T* paramsDim6 = reinterpret_cast<const COMP_T*>(&params);
        for (uint32_t i = 0; i < PARAM_DIM6_NUM; i++) {
            ParamDim6Ub.SetValue(i, paramsDim6[i]);
        }
        DataSyncBarrier<MemDsbT::UB>();
        if (tilingData_->dim == 0) {
            LaunchCompute<SimtComputeDim6<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 0>, DIM_6, 0, USED_THREAD>(
                TilingUint64Ub, (__ubuf__ COMP_T*)(ParamDim6Ub.GetPhyAddr()));
        } else if (tilingData_->dim == 1) {
            LaunchCompute<SimtComputeDim6<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 1>, DIM_6, 1, USED_THREAD>(
                TilingUint64Ub, (__ubuf__ COMP_T*)(ParamDim6Ub.GetPhyAddr()));
        } else if (tilingData_->dim == 2) {
            LaunchCompute<SimtComputeDim6<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 2>, DIM_6, 2, USED_THREAD>(
                TilingUint64Ub, (__ubuf__ COMP_T*)(ParamDim6Ub.GetPhyAddr()));
        } else if (tilingData_->dim == 3) {
            LaunchCompute<SimtComputeDim6<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 3>, DIM_6, 3, USED_THREAD>(
                TilingUint64Ub, (__ubuf__ COMP_T*)(ParamDim6Ub.GetPhyAddr()));
        } else if (tilingData_->dim == 4) {
            LaunchCompute<SimtComputeDim6<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 4>, DIM_6, 4, USED_THREAD>(
                TilingUint64Ub, (__ubuf__ COMP_T*)(ParamDim6Ub.GetPhyAddr()));
        } else {
            LaunchCompute<SimtComputeDim6<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 5>, DIM_6, 5, USED_THREAD>(
                TilingUint64Ub, (__ubuf__ COMP_T*)(ParamDim6Ub.GetPhyAddr()));
        }
    } else if (tilingData_->rank == DIM_7) {
        ScatterElementsQuickDivParamDim7<COMP_T> params;
        GetUintDivMagicAndShift(params.m0, params.shift0, static_cast<COMP_T>(tilingData_->indicesStride[0]));
        GetUintDivMagicAndShift(params.m1, params.shift1, static_cast<COMP_T>(tilingData_->indicesStride[DIM_1]));
        GetUintDivMagicAndShift(params.m2, params.shift2, static_cast<COMP_T>(tilingData_->indicesStride[DIM_2]));
        GetUintDivMagicAndShift(params.m3, params.shift3, static_cast<COMP_T>(tilingData_->indicesStride[DIM_3]));
        GetUintDivMagicAndShift(params.m4, params.shift4, static_cast<COMP_T>(tilingData_->indicesStride[DIM_4]));
        GetUintDivMagicAndShift(params.m5, params.shift5, static_cast<COMP_T>(tilingData_->indicesStride[DIM_5]));
        const COMP_T* paramsDim7 = reinterpret_cast<const COMP_T*>(&params);
        for (uint32_t i = 0; i < PARAM_DIM7_NUM; i++) {
            ParamDim7Ub.SetValue(i, paramsDim7[i]);
        }
        DataSyncBarrier<MemDsbT::UB>();
        if (tilingData_->dim == 0) {
            LaunchCompute<SimtComputeDim7<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 0>, DIM_7, 0, USED_THREAD>(
                TilingUint64Ub, (__ubuf__ COMP_T*)(ParamDim7Ub.GetPhyAddr()));
        } else if (tilingData_->dim == 1) {
            LaunchCompute<SimtComputeDim7<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 1>, DIM_7, 1, USED_THREAD>(
                TilingUint64Ub, (__ubuf__ COMP_T*)(ParamDim7Ub.GetPhyAddr()));
        } else if (tilingData_->dim == 2) {
            LaunchCompute<SimtComputeDim7<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 2>, DIM_7, 2, USED_THREAD>(
                TilingUint64Ub, (__ubuf__ COMP_T*)(ParamDim7Ub.GetPhyAddr()));
        } else if (tilingData_->dim == 3) {
            LaunchCompute<SimtComputeDim7<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 3>, DIM_7, 3, USED_THREAD>(
                TilingUint64Ub, (__ubuf__ COMP_T*)(ParamDim7Ub.GetPhyAddr()));
        } else if (tilingData_->dim == 4) {
            LaunchCompute<SimtComputeDim7<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 4>, DIM_7, 4, USED_THREAD>(
                TilingUint64Ub, (__ubuf__ COMP_T*)(ParamDim7Ub.GetPhyAddr()));
        } else if (tilingData_->dim == 5) {
            LaunchCompute<SimtComputeDim7<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 5>, DIM_7, 5, USED_THREAD>(
                TilingUint64Ub, (__ubuf__ COMP_T*)(ParamDim7Ub.GetPhyAddr()));
        } else {
            LaunchCompute<SimtComputeDim7<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 6>, DIM_7, 6, USED_THREAD>(
                TilingUint64Ub, (__ubuf__ COMP_T*)(ParamDim7Ub.GetPhyAddr()));
        }
    } else if (tilingData_->rank == DIM_8) {
        ScatterElementsQuickDivParamDim8<COMP_T> params;
        GetUintDivMagicAndShift(params.m0, params.shift0, static_cast<COMP_T>(tilingData_->indicesStride[0]));
        GetUintDivMagicAndShift(params.m1, params.shift1, static_cast<COMP_T>(tilingData_->indicesStride[DIM_1]));
        GetUintDivMagicAndShift(params.m2, params.shift2, static_cast<COMP_T>(tilingData_->indicesStride[DIM_2]));
        GetUintDivMagicAndShift(params.m3, params.shift3, static_cast<COMP_T>(tilingData_->indicesStride[DIM_3]));
        GetUintDivMagicAndShift(params.m4, params.shift4, static_cast<COMP_T>(tilingData_->indicesStride[DIM_4]));
        GetUintDivMagicAndShift(params.m5, params.shift5, static_cast<COMP_T>(tilingData_->indicesStride[DIM_5]));
        GetUintDivMagicAndShift(params.m6, params.shift6, static_cast<COMP_T>(tilingData_->indicesStride[DIM_6]));
        const COMP_T* paramsDim8 = reinterpret_cast<const COMP_T*>(&params);
        for (uint32_t i = 0; i < PARAM_DIM8_NUM; i++) {
            ParamDim8Ub.SetValue(i, paramsDim8[i]);
        }
        DataSyncBarrier<MemDsbT::UB>();
        if (tilingData_->dim == 0) {
            LaunchCompute<SimtComputeDim8<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 0>, DIM_8, 0, USED_THREAD>(
                TilingUint64Ub, (__ubuf__ COMP_T*)(ParamDim8Ub.GetPhyAddr()));
        } else if (tilingData_->dim == 1) {
            LaunchCompute<SimtComputeDim8<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 1>, DIM_8, 1, USED_THREAD>(
                TilingUint64Ub, (__ubuf__ COMP_T*)(ParamDim8Ub.GetPhyAddr()));
        } else if (tilingData_->dim == 2) {
            LaunchCompute<SimtComputeDim8<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 2>, DIM_8, 2, USED_THREAD>(
                TilingUint64Ub, (__ubuf__ COMP_T*)(ParamDim8Ub.GetPhyAddr()));
        } else if (tilingData_->dim == 3) {
            LaunchCompute<SimtComputeDim8<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 3>, DIM_8, 3, USED_THREAD>(
                TilingUint64Ub, (__ubuf__ COMP_T*)(ParamDim8Ub.GetPhyAddr()));
        } else if (tilingData_->dim == 4) {
            LaunchCompute<SimtComputeDim8<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 4>, DIM_8, 4, USED_THREAD>(
                TilingUint64Ub, (__ubuf__ COMP_T*)(ParamDim8Ub.GetPhyAddr()));
        } else if (tilingData_->dim == 5) {
            LaunchCompute<SimtComputeDim8<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 5>, DIM_8, 5, USED_THREAD>(
                TilingUint64Ub, (__ubuf__ COMP_T*)(ParamDim8Ub.GetPhyAddr()));
        } else if (tilingData_->dim == 6) {
            LaunchCompute<SimtComputeDim8<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 6>, DIM_8, 6, USED_THREAD>(
                TilingUint64Ub, (__ubuf__ COMP_T*)(ParamDim8Ub.GetPhyAddr()));
        } else {
            LaunchCompute<SimtComputeDim8<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 7>, DIM_8, 7, USED_THREAD>(
                TilingUint64Ub, (__ubuf__ COMP_T*)(ParamDim8Ub.GetPhyAddr()));
        }
    }

    if constexpr (SCATTER_USE_WORKSPACE<DATA_T, REDU>) {
        SyncAll();
        if (blockIdx_ < usedCoreNum_) {
            CopyResToY();
        }
    }
}
} // namespace ScatterElements

#endif // ASCENDC_SCATTER_ELEMENTS_V2_MULTI_H_
