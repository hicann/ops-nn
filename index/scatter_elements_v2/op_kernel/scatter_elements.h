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
 * \file scatter_elements.h
 * \brief scatter_elements
 */
#ifndef ASCENDC_SCATTER_ELEMENTS_V2_H_
#define ASCENDC_SCATTER_ELEMENTS_V2_H_

#include "kernel_operator.h"

#include "../inc/platform.h"
#include "../inc/kernel_utils.h"

namespace ScatterElements
{
using namespace AscendC;

constexpr int16_t DIM_1 = 1;
constexpr int16_t DIM_2 = 2;
constexpr int16_t DIM_3 = 3;
constexpr int16_t DIM_4 = 4;
constexpr int16_t DIM_5 = 5;
constexpr int16_t DIM_6 = 6;
constexpr int16_t DIM_7 = 7;
constexpr int16_t DIM_8 = 8;
constexpr uint32_t REDU_NONE = 0;
constexpr uint32_t REDU_ADD = 1;
constexpr uint32_t REDU_MUL = 2;
constexpr uint32_t USED_THREAD = 512;
constexpr uint32_t USED_THREAD1024 = 1024;
constexpr uint32_t VECTOR_LENGTH = platform::GetVRegSize();
constexpr uint32_t VL_B32 = VECTOR_LENGTH / sizeof(uint32_t);
constexpr uint32_t PARAM_DIM5_NUM = 8;
constexpr uint32_t PARAM_DIM6_NUM = 10;
constexpr uint32_t PARAM_DIM7_NUM = 12;
constexpr uint32_t PARAM_DIM8_NUM = 14;
constexpr uint32_t PARAM_UB_NUM = 16;
constexpr uint32_t TILING_DATA_UINT64_NUM = 21;
constexpr uint32_t TILING_DATA_UB_NUM = 24;
constexpr int64_t DB_BUFFER = 1;
constexpr int64_t GM_ALIGN = 512;
constexpr int16_t TILING_ARRAY_LEN = 7;
constexpr int16_t TWO_TILING_ARRAY_LEN = 14;
constexpr int16_t THREE_TILING_ARRAY_LEN = 21;
constexpr int64_t THREAD_SUM = 32768;

static constexpr MicroAPI::CastTrait castTraitB8B162B32 = {MicroAPI::RegLayout::ZERO, MicroAPI::SatMode::UNKNOWN,
                                                           MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

template <typename COMP_T>
struct ScatterElementsQuickDivParamDim8 {
    COMP_T m0{1};
    COMP_T m1{1};
    COMP_T m2{1};
    COMP_T m3{1};
    COMP_T m4{1};
    COMP_T m5{1};
    COMP_T m6{1};
    COMP_T shift0{1};
    COMP_T shift1{1};
    COMP_T shift2{1};
    COMP_T shift3{1};
    COMP_T shift4{1};
    COMP_T shift5{1};
    COMP_T shift6{1};
};

template <typename COMP_T>
struct ScatterElementsQuickDivParamDim7 {
    COMP_T m0{1};
    COMP_T m1{1};
    COMP_T m2{1};
    COMP_T m3{1};
    COMP_T m4{1};
    COMP_T m5{1};
    COMP_T shift0{1};
    COMP_T shift1{1};
    COMP_T shift2{1};
    COMP_T shift3{1};
    COMP_T shift4{1};
    COMP_T shift5{1};
};

template <typename COMP_T>
struct ScatterElementsQuickDivParamDim6 {
    COMP_T m0{1};
    COMP_T m1{1};
    COMP_T m2{1};
    COMP_T m3{1};
    COMP_T m4{1};
    COMP_T shift0{1};
    COMP_T shift1{1};
    COMP_T shift2{1};
    COMP_T shift3{1};
    COMP_T shift4{1};
};

template <typename COMP_T>
struct ScatterElementsQuickDivParamDim5 {
    COMP_T m0{1};
    COMP_T m1{1};
    COMP_T m2{1};
    COMP_T m3{1};
    COMP_T shift0{1};
    COMP_T shift1{1};
    COMP_T shift2{1};
    COMP_T shift3{1};
};

template <typename DATA_T, typename IDX_T, typename COMP_T, typename CAST_T, const uint32_t REDU, const uint32_t THREAD_USED>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_USED) inline void SimtComputeDim1(
    __gm__ IDX_T* indices, __gm__ DATA_T* updates, __gm__ DATA_T* y, __gm__ CAST_T* updatesWorkspaceGm,
    __gm__ CAST_T* xWorkspaceGm, COMP_T allAxis);

template <typename DATA_T, typename IDX_T, typename COMP_T, typename CAST_T, const uint32_t REDU, const uint16_t DIM, const uint32_t THREAD_USED>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_USED) inline void SimtComputeDim2(
    __gm__ IDX_T* indices, __gm__ DATA_T* updates, __gm__ DATA_T* y, __gm__ CAST_T* updatesWorkspaceGm,
    __gm__ CAST_T* xWorkspaceGm, __ubuf__ uint64_t* TilingUint64Ub, COMP_T allAxis, COMP_T m0, COMP_T shift0);

template <typename DATA_T, typename IDX_T, typename COMP_T, typename CAST_T, const uint32_t REDU, const uint16_t DIM, const uint32_t THREAD_USED>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_USED) inline void SimtComputeDim3(
    __gm__ IDX_T* indices, __gm__ DATA_T* updates, __gm__ DATA_T* y, __gm__ CAST_T* updatesWorkspaceGm,
    __gm__ CAST_T* xWorkspaceGm, __ubuf__ uint64_t* TilingUint64Ub, COMP_T allAxis, COMP_T m0,
    COMP_T shift0, COMP_T m1, COMP_T shift1);

template <typename DATA_T, typename IDX_T, typename COMP_T, typename CAST_T, const uint32_t REDU, const uint16_t DIM>
__simt_vf__ __aicore__ LAUNCH_BOUND(USED_THREAD) inline void SimtComputeDim4(
    __gm__ IDX_T* indices, __gm__ DATA_T* updates, __gm__ DATA_T* y, __gm__ CAST_T* updatesWorkspaceGm,
    __gm__ CAST_T* xWorkspaceGm, __ubuf__ uint64_t* TilingUint64Ub, COMP_T allAxis, COMP_T m0,
    COMP_T shift0, COMP_T m1, COMP_T shift1, COMP_T m2, COMP_T shift2);

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

template <typename DATA_T, typename COMP_T, typename CAST_T, const uint32_t REDU>
__aicore__ inline void ReplaceOut(__gm__ DATA_T* updates, __gm__ DATA_T* y, __gm__ CAST_T* updatesWorkspaceGm,
                                  __gm__ CAST_T* xWorkspaceGm, COMP_T yOffset, COMP_T updatesOffset);

template <typename DATA_T, typename IDX_T, typename COMP_T, typename CAST_T, const uint32_t REDU, const uint32_t TEMPLATE_V2=0>
class KernelScatterElements
{
public:
    __aicore__ inline KernelScatterElements(TPipe& pipe) : pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR indices, GM_ADDR updates, GM_ADDR y, GM_ADDR workspace,
                                const ScatterElementsV2AscTilingData* tilingData);
    __aicore__ inline void CopyToY(int64_t offset, int64_t dataLen);
    __aicore__ inline void CopyToWs(GlobalTensor<DATA_T>& inGm, GlobalTensor<CAST_T>& outGm, int64_t offset,
                                    int64_t dataLen);
    __aicore__ inline void CopyWsToY(int64_t offset, int64_t dataLen);
    __aicore__ inline void CastToInt32(LocalTensor<CAST_T>& dstLocal, LocalTensor<DATA_T>& srcLocal, uint32_t dataLen);
    __aicore__ inline void CastToOrigin(LocalTensor<DATA_T>& dstLocal, LocalTensor<CAST_T>& srcLocal, uint32_t dataLen);
    __aicore__ inline void CopyDataToY();
    __aicore__ inline void CopyDataToWs();
    __aicore__ inline void CopyUpdatesToWs();
    __aicore__ inline void CopyResToY();
    __aicore__ inline void Process();

private:
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

template <typename DATA_T, typename IDX_T, typename COMP_T, typename CAST_T, const uint32_t REDU, const uint32_t TEMPLATE_V2>
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
    if constexpr ((REDU == REDU_ADD) && (IsSameType<DATA_T, int8_t>::value || IsSameType<DATA_T, uint8_t>::value ||
                                         IsSameType<DATA_T, int16_t>::value)) {
        xWorkspaceGm_.SetGlobalBuffer((__gm__ CAST_T*)workspace);
        updatesWorkspaceGm_.SetGlobalBuffer((__gm__ CAST_T*)workspace +
                                            ops::Aligned(dataAxis_, static_cast<int64_t>(GM_ALIGN / sizeof(CAST_T))));
    }

    blockIdx_ = GetBlockIdx();
    blockNum_ = GetBlockNum();

    normBlockData_ = ops::CeilDiv(dataAxis_, static_cast<int64_t>(blockNum_));
    usedCoreNum_ = ops::CeilDiv(dataAxis_, normBlockData_);
    int64_t tailBlockData = dataAxis_ - (usedCoreNum_ - 1) * normBlockData_;
    int64_t curCoreData = blockIdx_ != (usedCoreNum_ - 1) ? normBlockData_ : tailBlockData;
    loopNum_ = curCoreData / loopLength_;
    tailLoopLength_ = curCoreData - loopNum_ * loopLength_;

    normBlockData2_ = ops::CeilDiv(updatesAxis_, static_cast<int64_t>(blockNum_));
    usedCoreNum2_ = ops::CeilDiv(updatesAxis_, normBlockData2_);
    int64_t tailBlockData2 = updatesAxis_ - (usedCoreNum2_ - 1) * normBlockData2_;
    int64_t curCoreData2 = blockIdx_ != (usedCoreNum2_ - 1) ? normBlockData2_ : tailBlockData2;
    loopNum2_ = curCoreData2 / loopLength_;
    tailLoopLength2_ = curCoreData2 - loopNum2_ * loopLength_;

    if constexpr ((REDU == REDU_ADD) && (IsSameType<DATA_T, int8_t>::value || IsSameType<DATA_T, uint8_t>::value ||
                                         IsSameType<DATA_T, int16_t>::value)) {
        pipe_.InitBuffer(dataQueue_, DB_BUFFER, loopLength_ * sizeof(DATA_T));
        pipe_.InitBuffer(outQueue_, DB_BUFFER, loopLength_ * sizeof(CAST_T));
    } else {
        if constexpr (TEMPLATE_V2 == 0) {
            pipe_.InitBuffer(dataQueue_, DB_BUFFER, loopLength_ * sizeof(DATA_T));
        }
    }
    pipe_.InitBuffer(tilingDataUint64Buf_, TILING_DATA_UB_NUM * sizeof(uint64_t));
    pipe_.InitBuffer(paramDim5Buf_, PARAM_DIM5_NUM * sizeof(COMP_T));
    pipe_.InitBuffer(paramDim6Buf_, PARAM_UB_NUM * sizeof(COMP_T));
    pipe_.InitBuffer(paramDim7Buf_, PARAM_UB_NUM * sizeof(COMP_T));
    pipe_.InitBuffer(paramDim8Buf_, PARAM_UB_NUM * sizeof(COMP_T));
}

template <typename DATA_T, typename IDX_T, typename COMP_T, typename CAST_T, const uint32_t REDU, const uint32_t TEMPLATE_V2>
__aicore__ inline void KernelScatterElements<DATA_T, IDX_T, COMP_T, CAST_T, REDU, TEMPLATE_V2>::CopyToY(int64_t offset,
                                                                                           int64_t dataLen)
{
    DataCopyExtParams copyParams = {static_cast<uint16_t>(1), static_cast<uint32_t>(dataLen * sizeof(DATA_T)),
                                    static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    DataCopyPadExtParams<DATA_T> padParams = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
                                              static_cast<DATA_T>(0)};
    LocalTensor<DATA_T> xLocal = dataQueue_.AllocTensor<DATA_T>();
    DataCopyPad(xLocal, x_[offset], copyParams, padParams);
    dataQueue_.EnQue(xLocal);

    LocalTensor<DATA_T> yLocal = dataQueue_.DeQue<DATA_T>();
    DataCopyPad(y_[offset], yLocal, copyParams);
    dataQueue_.FreeTensor(yLocal);
}

template <typename DATA_T, typename IDX_T, typename COMP_T, typename CAST_T, const uint32_t REDU, const uint32_t TEMPLATE_V2>
__aicore__ inline void KernelScatterElements<DATA_T, IDX_T, COMP_T, CAST_T, REDU, TEMPLATE_V2>::CopyToWs(GlobalTensor<DATA_T>& inGm,
                                                                                            GlobalTensor<CAST_T>& outGm,
                                                                                            int64_t offset,
                                                                                            int64_t dataLen)
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

template <typename DATA_T, typename IDX_T, typename COMP_T, typename CAST_T, const uint32_t REDU, const uint32_t TEMPLATE_V2>
__aicore__ inline void KernelScatterElements<DATA_T, IDX_T, COMP_T, CAST_T, REDU, TEMPLATE_V2>::CopyWsToY(int64_t offset,
                                                                                             int64_t dataLen)
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

template <typename DATA_T, typename IDX_T, typename COMP_T, typename CAST_T, const uint32_t REDU, const uint32_t TEMPLATE_V2>
__aicore__ inline void KernelScatterElements<DATA_T, IDX_T, COMP_T, CAST_T, REDU, TEMPLATE_V2>::CastToInt32(
    LocalTensor<CAST_T>& dstLocal, LocalTensor<DATA_T>& srcLocal, uint32_t dataLen)
{
    __local_mem__ DATA_T* srcAddr = (__local_mem__ DATA_T*)srcLocal.GetPhyAddr();
    __local_mem__ CAST_T* dstAddr = (__local_mem__ CAST_T*)dstLocal.GetPhyAddr();

    uint16_t loopTimes = ops::CeilDiv(dataLen, VL_B32);

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<DATA_T> srcValue;
        MicroAPI::RegTensor<CAST_T> dstValue;
        MicroAPI::MaskReg preg;
        uint32_t sregMask = dataLen;
        for (uint16_t j = 0; j < loopTimes; j++) {
            preg = MicroAPI::UpdateMask<uint32_t>(sregMask);
            if constexpr (IsSameType<DATA_T, int16_t>::value) {
                MicroAPI::DataCopy<DATA_T, MicroAPI::LoadDist::DIST_UNPACK_B16>(srcValue, srcAddr + VL_B32 * j);
            } else {
                MicroAPI::DataCopy<DATA_T, MicroAPI::LoadDist::DIST_UNPACK4_B8>(srcValue, srcAddr + VL_B32 * j);
            }
            MicroAPI::Cast<CAST_T, DATA_T, castTraitB8B162B32>(dstValue, srcValue, preg);
            MicroAPI::DataCopy<CAST_T, MicroAPI::StoreDist::DIST_NORM>(dstAddr + VL_B32 * j, dstValue, preg);
        }
    }
}

template <typename DATA_T, typename IDX_T, typename COMP_T, typename CAST_T, const uint32_t REDU, const uint32_t TEMPLATE_V2>
__aicore__ inline void KernelScatterElements<DATA_T, IDX_T, COMP_T, CAST_T, REDU, TEMPLATE_V2>::CastToOrigin(
    LocalTensor<DATA_T>& dstLocal, LocalTensor<CAST_T>& srcLocal, uint32_t dataLen)
{
    __local_mem__ CAST_T* srcAddr = (__local_mem__ CAST_T*)srcLocal.GetPhyAddr();
    __local_mem__ DATA_T* dstAddr = (__local_mem__ DATA_T*)dstLocal.GetPhyAddr();

    uint16_t loopTimes = ops::CeilDiv(dataLen, VL_B32);

    __VEC_SCOPE__
    {
        MicroAPI::RegTensor<CAST_T> srcValue;
        MicroAPI::MaskReg preg;
        uint32_t sregMask = dataLen;
        for (uint16_t j = 0; j < loopTimes; j++) {
            preg = MicroAPI::UpdateMask<uint32_t>(sregMask);
            MicroAPI::DataCopy<CAST_T, MicroAPI::LoadDist::DIST_NORM>(srcValue, srcAddr + VL_B32 * j);
            if constexpr (IsSameType<DATA_T, int16_t>::value) {
                MicroAPI::DataCopy<DATA_T, MicroAPI::StoreDist::DIST_PACK_B32>(
                    dstAddr + VL_B32 * j, (MicroAPI::RegTensor<DATA_T>&)srcValue, preg);
            } else {
                MicroAPI::DataCopy<DATA_T, MicroAPI::StoreDist::DIST_PACK4_B32>(
                    dstAddr + VL_B32 * j, (MicroAPI::RegTensor<DATA_T>&)srcValue, preg);
            }
        }
    }
}

template <typename DATA_T, typename IDX_T, typename COMP_T, typename CAST_T, const uint32_t REDU, const uint32_t TEMPLATE_V2>
__aicore__ inline void KernelScatterElements<DATA_T, IDX_T, COMP_T, CAST_T, REDU, TEMPLATE_V2>::CopyDataToY()
{
    int64_t offset = 0;
    for (int64_t idx = 0; idx < loopNum_; idx++) {
        offset = static_cast<int64_t>(blockIdx_) * normBlockData_ + idx * loopLength_;
        CopyToY(offset, loopLength_);
    }

    if (tailLoopLength_ > 0) {
        offset = static_cast<int64_t>(blockIdx_) * normBlockData_ + loopNum_ * loopLength_;
        CopyToY(offset, tailLoopLength_);
    }
}

template <typename DATA_T, typename IDX_T, typename COMP_T, typename CAST_T, const uint32_t REDU, const uint32_t TEMPLATE_V2>
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

template <typename DATA_T, typename IDX_T, typename COMP_T, typename CAST_T, const uint32_t REDU, const uint32_t TEMPLATE_V2>
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

template <typename DATA_T, typename IDX_T, typename COMP_T, typename CAST_T, const uint32_t REDU, const uint32_t TEMPLATE_V2>
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

template <typename DATA_T, typename COMP_T, typename CAST_T, const uint32_t REDU>
__aicore__ inline void ReplaceOut(__gm__ DATA_T* updates, __gm__ DATA_T* y, __gm__ CAST_T* updatesWorkspaceGm,
                                  __gm__ CAST_T* xWorkspaceGm, COMP_T yOffset, COMP_T updatesOffset)
{
    if constexpr (REDU == REDU_ADD) {
        if constexpr (IsSameType<DATA_T, int8_t>::value || IsSameType<DATA_T, uint8_t>::value ||
                      IsSameType<DATA_T, int16_t>::value) {
            Simt::AtomicAdd(xWorkspaceGm + yOffset, updatesWorkspaceGm[updatesOffset]);
        } else {
            Simt::AtomicAdd(y + yOffset, updates[updatesOffset]);
        }
    } else if constexpr (REDU == REDU_MUL) {
        y[yOffset] = y[yOffset] * updates[updatesOffset];
    } else {
        y[yOffset] = updates[updatesOffset];
    }
}

template <typename DATA_T, typename IDX_T, typename COMP_T, typename CAST_T, const uint32_t REDU, const uint32_t THREAD_USED>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_USED) inline void SimtComputeDim1(
    __gm__ IDX_T* indices, __gm__ DATA_T* updates, __gm__ DATA_T* y, __gm__ CAST_T* updatesWorkspaceGm,
    __gm__ CAST_T* xWorkspaceGm, COMP_T allAxis)
{
    for (COMP_T i = Simt::GetBlockIdx() * Simt::GetThreadNum() + Simt::GetThreadIdx(); i < allAxis;
         i += Simt::GetBlockNum() * Simt::GetThreadNum()) {
        COMP_T yOffset = static_cast<COMP_T>(indices[i]);
        ReplaceOut<DATA_T, COMP_T, CAST_T, REDU>(updates, y, updatesWorkspaceGm, xWorkspaceGm, yOffset, i);
    }
}

template <typename DATA_T, typename IDX_T, typename COMP_T, typename CAST_T, const uint32_t REDU, const uint16_t DIM, const uint32_t THREAD_USED>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_USED) inline void SimtComputeDim2(
    __gm__ IDX_T* indices, __gm__ DATA_T* updates, __gm__ DATA_T* y, __gm__ CAST_T* updatesWorkspaceGm,
    __gm__ CAST_T* xWorkspaceGm, __ubuf__ uint64_t* TilingUint64Ub, COMP_T allAxis, COMP_T m0, COMP_T shift0)
{
    uint64_t dataStride[TILING_ARRAY_LEN] = {};
    uint64_t indicesStride[TILING_ARRAY_LEN] = {};
    uint64_t updatesStride[TILING_ARRAY_LEN] = {};
    for (uint32_t i = 0; i < TILING_DATA_UINT64_NUM; i++) {
        if (i < TILING_ARRAY_LEN) {
            dataStride[i] = TilingUint64Ub[i];
        } else if (i < TWO_TILING_ARRAY_LEN) {
            indicesStride[i - TILING_ARRAY_LEN] = TilingUint64Ub[i];
        } else if (i < THREE_TILING_ARRAY_LEN) {
            updatesStride[i - TWO_TILING_ARRAY_LEN] = TilingUint64Ub[i];
        }
    }

    for (COMP_T i = Simt::GetBlockIdx() * Simt::GetThreadNum() + Simt::GetThreadIdx(); i < allAxis;
         i += Simt::GetBlockNum() * Simt::GetThreadNum()) {
        COMP_T dim0Idx = Simt::UintDiv(i, m0, shift0);
        COMP_T dim1Idx = i - dim0Idx * indicesStride[0];

        COMP_T yOffset = 0;
        if constexpr (DIM == 0) {
            yOffset = static_cast<COMP_T>(indices[i]) * dataStride[0] + dim1Idx;
        } else {
            yOffset = dim0Idx * dataStride[0] + static_cast<COMP_T>(indices[i]);
        }
        COMP_T updatesOffset = dim0Idx * updatesStride[0] + dim1Idx;
        ReplaceOut<DATA_T, COMP_T, CAST_T, REDU>(updates, y, updatesWorkspaceGm, xWorkspaceGm, yOffset, updatesOffset);
    }
}

template <typename DATA_T, typename IDX_T, typename COMP_T, typename CAST_T, const uint32_t REDU, const uint16_t DIM, const uint32_t THREAD_USED>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_USED) inline void SimtComputeDim3(
    __gm__ IDX_T* indices, __gm__ DATA_T* updates, __gm__ DATA_T* y, __gm__ CAST_T* updatesWorkspaceGm,
    __gm__ CAST_T* xWorkspaceGm, __ubuf__ uint64_t* TilingUint64Ub, COMP_T allAxis, COMP_T m0,
    COMP_T shift0, COMP_T m1, COMP_T shift1)
{
    uint64_t dataStride[TILING_ARRAY_LEN] = {};
    uint64_t indicesStride[TILING_ARRAY_LEN] = {};
    uint64_t updatesStride[TILING_ARRAY_LEN] = {};
    for (uint32_t i = 0; i < TILING_DATA_UINT64_NUM; i++) {
        if (i < TILING_ARRAY_LEN) {
            dataStride[i] = TilingUint64Ub[i];
        } else if (i < TWO_TILING_ARRAY_LEN) {
            indicesStride[i - TILING_ARRAY_LEN] = TilingUint64Ub[i];
        } else if (i < THREE_TILING_ARRAY_LEN) {
            updatesStride[i - TWO_TILING_ARRAY_LEN] = TilingUint64Ub[i];
        }
    }

    for (COMP_T i = Simt::GetBlockIdx() * Simt::GetThreadNum() + Simt::GetThreadIdx(); i < allAxis;
         i += Simt::GetBlockNum() * Simt::GetThreadNum()) {
        COMP_T dim0Idx = Simt::UintDiv(i, m0, shift0);
        COMP_T dim0Rem = i - dim0Idx * indicesStride[0];

        COMP_T dim1Idx = Simt::UintDiv(dim0Rem, m1, shift1);
        COMP_T dim2Idx = dim0Rem - dim1Idx * indicesStride[DIM_1];

        COMP_T yOffset = 0;
        if constexpr (DIM == 0) {
            yOffset =
                static_cast<COMP_T>(indices[i]) * dataStride[0] + dim1Idx * dataStride[DIM_1] + dim2Idx;
        } else if constexpr (DIM == 1) {
            yOffset =
                dim0Idx * dataStride[0] + static_cast<COMP_T>(indices[i]) * dataStride[DIM_1] + dim2Idx;
        } else {
            yOffset =
                dim0Idx * dataStride[0] + dim1Idx * dataStride[DIM_1] + static_cast<COMP_T>(indices[i]);
        }
        COMP_T updatesOffset = dim0Idx * updatesStride[0] + dim1Idx * updatesStride[DIM_1] + dim2Idx;
        ReplaceOut<DATA_T, COMP_T, CAST_T, REDU>(updates, y, updatesWorkspaceGm, xWorkspaceGm, yOffset, updatesOffset);
    }
}

template <typename DATA_T, typename IDX_T, typename COMP_T, typename CAST_T, const uint32_t REDU, const uint16_t DIM>
__simt_vf__ __aicore__ LAUNCH_BOUND(USED_THREAD) inline void SimtComputeDim4(
    __gm__ IDX_T* indices, __gm__ DATA_T* updates, __gm__ DATA_T* y, __gm__ CAST_T* updatesWorkspaceGm,
    __gm__ CAST_T* xWorkspaceGm, __ubuf__ uint64_t* TilingUint64Ub, COMP_T allAxis, COMP_T m0,
    COMP_T shift0, COMP_T m1, COMP_T shift1, COMP_T m2, COMP_T shift2)
{
    uint64_t dataStride[TILING_ARRAY_LEN] = {};
    uint64_t indicesStride[TILING_ARRAY_LEN] = {};
    uint64_t updatesStride[TILING_ARRAY_LEN] = {};
    for (uint32_t i = 0; i < TILING_DATA_UINT64_NUM; i++) {
        if (i < TILING_ARRAY_LEN) {
            dataStride[i] = TilingUint64Ub[i];
        } else if (i < TWO_TILING_ARRAY_LEN) {
            indicesStride[i - TILING_ARRAY_LEN] = TilingUint64Ub[i];
        } else if (i < THREE_TILING_ARRAY_LEN) {
            updatesStride[i - TWO_TILING_ARRAY_LEN] = TilingUint64Ub[i];
        }
    }

    for (COMP_T i = Simt::GetBlockIdx() * Simt::GetThreadNum() + Simt::GetThreadIdx(); i < allAxis;
         i += Simt::GetBlockNum() * Simt::GetThreadNum()) {
        COMP_T dim0Idx = Simt::UintDiv(i, m0, shift0);
        COMP_T dim0Rem = i - dim0Idx * indicesStride[0];

        COMP_T dim1Idx = Simt::UintDiv(dim0Rem, m1, shift1);
        COMP_T dim1Rem = dim0Rem - dim1Idx * indicesStride[DIM_1];

        COMP_T dim2Idx = Simt::UintDiv(dim1Rem, m2, shift2);
        COMP_T dim3Idx = dim1Rem - dim2Idx * indicesStride[DIM_2];

        COMP_T yOffset = 0;
        if constexpr (DIM == 0) {
            yOffset = static_cast<COMP_T>(indices[i]) * dataStride[0] + dim1Idx * dataStride[DIM_1] +
                      dim2Idx * dataStride[DIM_2] + dim3Idx;
        } else if constexpr (DIM == 1) {
            yOffset = dim0Idx * dataStride[0] + static_cast<COMP_T>(indices[i]) * dataStride[DIM_1] +
                      dim2Idx * dataStride[DIM_2] + dim3Idx;
        } else if constexpr (DIM == 2) {
            yOffset = dim0Idx * dataStride[0] + dim1Idx * dataStride[DIM_1] +
                      static_cast<COMP_T>(indices[i]) * dataStride[DIM_2] + dim3Idx;
        } else {
            yOffset = dim0Idx * dataStride[0] + dim1Idx * dataStride[DIM_1] +
                      dim2Idx * dataStride[DIM_2] + static_cast<COMP_T>(indices[i]);
        }
        COMP_T updatesOffset = dim0Idx * updatesStride[0] + dim1Idx * updatesStride[DIM_1] +
                               dim2Idx * updatesStride[DIM_2] + dim3Idx;
        ReplaceOut<DATA_T, COMP_T, CAST_T, REDU>(updates, y, updatesWorkspaceGm, xWorkspaceGm, yOffset, updatesOffset);
    }
}

template <typename DATA_T, typename IDX_T, typename COMP_T, typename CAST_T, const uint32_t REDU, const uint16_t DIM>
__simt_vf__ __aicore__ LAUNCH_BOUND(USED_THREAD) inline void SimtComputeDim5(
    __gm__ IDX_T* indices, __gm__ DATA_T* updates, __gm__ DATA_T* y, __gm__ CAST_T* updatesWorkspaceGm,
    __gm__ CAST_T* xWorkspaceGm, __ubuf__ uint64_t* TilingUint64Ub, COMP_T allAxis, __ubuf__ COMP_T* params)
{
    uint64_t dataStride[TILING_ARRAY_LEN] = {};
    uint64_t indicesStride[TILING_ARRAY_LEN] = {};
    uint64_t updatesStride[TILING_ARRAY_LEN] = {};
    for (uint32_t i = 0; i < TILING_DATA_UINT64_NUM; i++) {
        if (i < TILING_ARRAY_LEN) {
            dataStride[i] = TilingUint64Ub[i];
        } else if (i < TWO_TILING_ARRAY_LEN) {
            indicesStride[i - TILING_ARRAY_LEN] = TilingUint64Ub[i];
        } else if (i < THREE_TILING_ARRAY_LEN) {
            updatesStride[i - TWO_TILING_ARRAY_LEN] = TilingUint64Ub[i];
        }
    }

    COMP_T m0 = params[0];
    COMP_T m1 = params[1];
    COMP_T m2 = params[2];
    COMP_T m3 = params[3];
    COMP_T shift0 = params[4];
    COMP_T shift1 = params[5];
    COMP_T shift2 = params[6];
    COMP_T shift3 = params[7];

    for (COMP_T i = Simt::GetBlockIdx() * Simt::GetThreadNum() + Simt::GetThreadIdx(); i < allAxis;
         i += Simt::GetBlockNum() * Simt::GetThreadNum()) {
        COMP_T dim0Idx = Simt::UintDiv(i, m0, shift0);
        COMP_T dim0Rem = i - dim0Idx * indicesStride[0];

        COMP_T dim1Idx = Simt::UintDiv(dim0Rem, m1, shift1);
        COMP_T dim1Rem = dim0Rem - dim1Idx * indicesStride[DIM_1];

        COMP_T dim2Idx = Simt::UintDiv(dim1Rem, m2, shift2);
        COMP_T dim2Rem = dim1Rem - dim2Idx * indicesStride[DIM_2];

        COMP_T dim3Idx = Simt::UintDiv(dim2Rem, m3, shift3);
        COMP_T dim4Idx = dim2Rem - dim3Idx * indicesStride[DIM_3];

        COMP_T yOffset = 0;
        if constexpr (DIM == 0) {
            yOffset = static_cast<COMP_T>(indices[i]) * dataStride[0] + dim1Idx * dataStride[DIM_1] +
                      dim2Idx * dataStride[DIM_2] + dim3Idx * dataStride[DIM_3] + dim4Idx;
        } else if constexpr (DIM == 1) {
            yOffset = dim0Idx * dataStride[0] + static_cast<COMP_T>(indices[i]) * dataStride[DIM_1] +
                      dim2Idx * dataStride[DIM_2] + dim3Idx * dataStride[DIM_3] + dim4Idx;
        } else if constexpr (DIM == 2) {
            yOffset = dim0Idx * dataStride[0] + dim1Idx * dataStride[DIM_1] +
                      static_cast<COMP_T>(indices[i]) * dataStride[DIM_2] +
                      dim3Idx * dataStride[DIM_3] + dim4Idx;
        } else if constexpr (DIM == 3) {
            yOffset = dim0Idx * dataStride[0] + dim1Idx * dataStride[DIM_1] +
                      dim2Idx * dataStride[DIM_2] +
                      static_cast<COMP_T>(indices[i]) * dataStride[DIM_3] + dim4Idx;
        } else {
            yOffset = dim0Idx * dataStride[0] + dim1Idx * dataStride[DIM_1] +
                      dim2Idx * dataStride[DIM_2] + dim3Idx * dataStride[DIM_3] +
                      static_cast<COMP_T>(indices[i]);
        }
        COMP_T updatesOffset = dim0Idx * updatesStride[0] + dim1Idx * updatesStride[DIM_1] +
                               dim2Idx * updatesStride[DIM_2] + dim3Idx * updatesStride[DIM_3] +
                               dim4Idx;
        ReplaceOut<DATA_T, COMP_T, CAST_T, REDU>(updates, y, updatesWorkspaceGm, xWorkspaceGm, yOffset, updatesOffset);
    }
}

template <typename DATA_T, typename IDX_T, typename COMP_T, typename CAST_T, const uint32_t REDU, const uint16_t DIM>
__simt_vf__ __aicore__ LAUNCH_BOUND(USED_THREAD) inline void SimtComputeDim6(
    __gm__ IDX_T* indices, __gm__ DATA_T* updates, __gm__ DATA_T* y, __gm__ CAST_T* updatesWorkspaceGm,
    __gm__ CAST_T* xWorkspaceGm, __ubuf__ uint64_t* TilingUint64Ub, COMP_T allAxis, __ubuf__ COMP_T* params)
{
    uint64_t dataStride[TILING_ARRAY_LEN] = {};
    uint64_t indicesStride[TILING_ARRAY_LEN] = {};
    uint64_t updatesStride[TILING_ARRAY_LEN] = {};
    for (uint32_t i = 0; i < TILING_DATA_UINT64_NUM; i++) {
        if (i < TILING_ARRAY_LEN) {
            dataStride[i] = TilingUint64Ub[i];
        } else if (i < TWO_TILING_ARRAY_LEN) {
            indicesStride[i - TILING_ARRAY_LEN] = TilingUint64Ub[i];
        } else if (i < THREE_TILING_ARRAY_LEN) {
            updatesStride[i - TWO_TILING_ARRAY_LEN] = TilingUint64Ub[i];
        }
    }

    COMP_T m0 = params[0];
    COMP_T m1 = params[1];
    COMP_T m2 = params[2];
    COMP_T m3 = params[3];
    COMP_T m4 = params[4];
    COMP_T shift0 = params[5];
    COMP_T shift1 = params[6];
    COMP_T shift2 = params[7];
    COMP_T shift3 = params[8];
    COMP_T shift4 = params[9];

    for (COMP_T i = Simt::GetBlockIdx() * Simt::GetThreadNum() + Simt::GetThreadIdx(); i < allAxis;
         i += Simt::GetBlockNum() * Simt::GetThreadNum()) {
        COMP_T dim0Idx = Simt::UintDiv(i, m0, shift0);
        COMP_T dim0Rem = i - dim0Idx * indicesStride[0];

        COMP_T dim1Idx = Simt::UintDiv(dim0Rem, m1, shift1);
        COMP_T dim1Rem = dim0Rem - dim1Idx * indicesStride[DIM_1];

        COMP_T dim2Idx = Simt::UintDiv(dim1Rem, m2, shift2);
        COMP_T dim2Rem = dim1Rem - dim2Idx * indicesStride[DIM_2];

        COMP_T dim3Idx = Simt::UintDiv(dim2Rem, m3, shift3);
        COMP_T dim3Rem = dim2Rem - dim3Idx * indicesStride[DIM_3];

        COMP_T dim4Idx = Simt::UintDiv(dim3Rem, m4, shift4);
        COMP_T dim5Idx = dim3Rem - dim4Idx * indicesStride[DIM_4];

        COMP_T yOffset = 0;
        if constexpr (DIM == 0) {
            yOffset = static_cast<COMP_T>(indices[i]) * dataStride[0] + dim1Idx * dataStride[DIM_1] +
                      dim2Idx * dataStride[DIM_2] + dim3Idx * dataStride[DIM_3] +
                      dim4Idx * dataStride[DIM_4] + dim5Idx;
        } else if constexpr (DIM == 1) {
            yOffset = dim0Idx * dataStride[0] + static_cast<COMP_T>(indices[i]) * dataStride[DIM_1] +
                      dim2Idx * dataStride[DIM_2] + dim3Idx * dataStride[DIM_3] +
                      dim4Idx * dataStride[DIM_4] + dim5Idx;
        } else if constexpr (DIM == 2) {
            yOffset = dim0Idx * dataStride[0] + dim1Idx * dataStride[DIM_1] +
                      static_cast<COMP_T>(indices[i]) * dataStride[DIM_2] +
                      dim3Idx * dataStride[DIM_3] + dim4Idx * dataStride[DIM_4] + dim5Idx;
        } else if constexpr (DIM == 3) {
            yOffset = dim0Idx * dataStride[0] + dim1Idx * dataStride[DIM_1] +
                      dim2Idx * dataStride[DIM_2] +
                      static_cast<COMP_T>(indices[i]) * dataStride[DIM_3] +
                      dim4Idx * dataStride[DIM_4] + dim5Idx;
        } else if constexpr (DIM == 4) {
            yOffset = dim0Idx * dataStride[0] + dim1Idx * dataStride[DIM_1] +
                      dim2Idx * dataStride[DIM_2] + dim3Idx * dataStride[DIM_3] +
                      static_cast<COMP_T>(indices[i]) * dataStride[DIM_4] + dim5Idx;
        } else {
            yOffset = dim0Idx * dataStride[0] + dim1Idx * dataStride[DIM_1] +
                      dim2Idx * dataStride[DIM_2] + dim3Idx * dataStride[DIM_3] +
                      dim4Idx * dataStride[DIM_4] + static_cast<COMP_T>(indices[i]);
        }
        COMP_T updatesOffset = dim0Idx * updatesStride[0] + dim1Idx * updatesStride[DIM_1] +
                               dim2Idx * updatesStride[DIM_2] + dim3Idx * updatesStride[DIM_3] +
                               dim4Idx * updatesStride[DIM_4] + dim5Idx;
        ReplaceOut<DATA_T, COMP_T, CAST_T, REDU>(updates, y, updatesWorkspaceGm, xWorkspaceGm, yOffset, updatesOffset);
    }
}

template <typename DATA_T, typename IDX_T, typename COMP_T, typename CAST_T, const uint32_t REDU, const uint16_t DIM>
__simt_vf__ __aicore__ LAUNCH_BOUND(USED_THREAD) inline void SimtComputeDim7(
    __gm__ IDX_T* indices, __gm__ DATA_T* updates, __gm__ DATA_T* y, __gm__ CAST_T* updatesWorkspaceGm,
    __gm__ CAST_T* xWorkspaceGm, __ubuf__ uint64_t* TilingUint64Ub, COMP_T allAxis, __ubuf__ COMP_T* params)
{
    uint64_t dataStride[TILING_ARRAY_LEN] = {};
    uint64_t indicesStride[TILING_ARRAY_LEN] = {};
    uint64_t updatesStride[TILING_ARRAY_LEN] = {};
    for (uint32_t i = 0; i < TILING_DATA_UINT64_NUM; i++) {
        if (i < TILING_ARRAY_LEN) {
            dataStride[i] = TilingUint64Ub[i];
        } else if (i < TWO_TILING_ARRAY_LEN) {
            indicesStride[i - TILING_ARRAY_LEN] = TilingUint64Ub[i];
        } else if (i < THREE_TILING_ARRAY_LEN) {
            updatesStride[i - TWO_TILING_ARRAY_LEN] = TilingUint64Ub[i];
        }
    }

    COMP_T m0 = params[0];
    COMP_T m1 = params[1];
    COMP_T m2 = params[2];
    COMP_T m3 = params[3];
    COMP_T m4 = params[4];
    COMP_T m5 = params[5];
    COMP_T shift0 = params[6];
    COMP_T shift1 = params[7];
    COMP_T shift2 = params[8];
    COMP_T shift3 = params[9];
    COMP_T shift4 = params[10];
    COMP_T shift5 = params[11];

    for (COMP_T i = Simt::GetBlockIdx() * Simt::GetThreadNum() + Simt::GetThreadIdx(); i < allAxis;
         i += Simt::GetBlockNum() * Simt::GetThreadNum()) {
        COMP_T dim0Idx = Simt::UintDiv(i, m0, shift0);
        COMP_T dim0Rem = i - dim0Idx * indicesStride[0];

        COMP_T dim1Idx = Simt::UintDiv(dim0Rem, m1, shift1);
        COMP_T dim1Rem = dim0Rem - dim1Idx * indicesStride[DIM_1];

        COMP_T dim2Idx = Simt::UintDiv(dim1Rem, m2, shift2);
        COMP_T dim2Rem = dim1Rem - dim2Idx * indicesStride[DIM_2];

        COMP_T dim3Idx = Simt::UintDiv(dim2Rem, m3, shift3);
        COMP_T dim3Rem = dim2Rem - dim3Idx * indicesStride[DIM_3];

        COMP_T dim4Idx = Simt::UintDiv(dim3Rem, m4, shift4);
        COMP_T dim4Rem = dim3Rem - dim4Idx * indicesStride[DIM_4];

        COMP_T dim5Idx = Simt::UintDiv(dim4Rem, m5, shift5);
        COMP_T dim6Idx = dim4Rem - dim5Idx * indicesStride[DIM_5];

        COMP_T yOffset = 0;
        if constexpr (DIM == 0) {
            yOffset = static_cast<COMP_T>(indices[i]) * dataStride[0] + dim1Idx * dataStride[DIM_1] +
                      dim2Idx * dataStride[DIM_2] + dim3Idx * dataStride[DIM_3] +
                      dim4Idx * dataStride[DIM_4] + dim5Idx * dataStride[DIM_5] + dim6Idx;
        } else if constexpr (DIM == 1) {
            yOffset = dim0Idx * dataStride[0] + static_cast<COMP_T>(indices[i]) * dataStride[DIM_1] +
                      dim2Idx * dataStride[DIM_2] + dim3Idx * dataStride[DIM_3] +
                      dim4Idx * dataStride[DIM_4] + dim5Idx * dataStride[DIM_5] + dim6Idx;
        } else if constexpr (DIM == 2) {
            yOffset = dim0Idx * dataStride[0] + dim1Idx * dataStride[DIM_1] +
                      static_cast<COMP_T>(indices[i]) * dataStride[DIM_2] +
                      dim3Idx * dataStride[DIM_3] + dim4Idx * dataStride[DIM_4] +
                      dim5Idx * dataStride[DIM_5] + dim6Idx;
        } else if constexpr (DIM == 3) {
            yOffset = dim0Idx * dataStride[0] + dim1Idx * dataStride[DIM_1] +
                      dim2Idx * dataStride[DIM_2] +
                      static_cast<COMP_T>(indices[i]) * dataStride[DIM_3] +
                      dim4Idx * dataStride[DIM_4] + dim5Idx * dataStride[DIM_5] + dim6Idx;
        } else if constexpr (DIM == 4) {
            yOffset = dim0Idx * dataStride[0] + dim1Idx * dataStride[DIM_1] +
                      dim2Idx * dataStride[DIM_2] + dim3Idx * dataStride[DIM_3] +
                      static_cast<COMP_T>(indices[i]) * dataStride[DIM_4] +
                      dim5Idx * dataStride[DIM_5] + dim6Idx;
        } else if constexpr (DIM == 5) {
            yOffset = dim0Idx * dataStride[0] + dim1Idx * dataStride[DIM_1] +
                      dim2Idx * dataStride[DIM_2] + dim3Idx * dataStride[DIM_3] +
                      dim4Idx * dataStride[DIM_4] +
                      static_cast<COMP_T>(indices[i]) * dataStride[DIM_5] + dim6Idx;
        } else {
            yOffset = dim0Idx * dataStride[0] + dim1Idx * dataStride[DIM_1] +
                      dim2Idx * dataStride[DIM_2] + dim3Idx * dataStride[DIM_3] +
                      dim4Idx * dataStride[DIM_4] + dim5Idx * dataStride[DIM_5] +
                      static_cast<COMP_T>(indices[i]);
        }
        COMP_T updatesOffset = dim0Idx * updatesStride[0] + dim1Idx * updatesStride[DIM_1] +
                               dim2Idx * updatesStride[DIM_2] + dim3Idx * updatesStride[DIM_3] +
                               dim4Idx * updatesStride[DIM_4] + dim5Idx * updatesStride[DIM_5] +
                               dim6Idx;
        ReplaceOut<DATA_T, COMP_T, CAST_T, REDU>(updates, y, updatesWorkspaceGm, xWorkspaceGm, yOffset, updatesOffset);
    }
}

template <typename DATA_T, typename IDX_T, typename COMP_T, typename CAST_T, const uint32_t REDU, const uint16_t DIM>
__simt_vf__ __aicore__ LAUNCH_BOUND(USED_THREAD) inline void SimtComputeDim8(
    __gm__ IDX_T* indices, __gm__ DATA_T* updates, __gm__ DATA_T* y, __gm__ CAST_T* updatesWorkspaceGm,
    __gm__ CAST_T* xWorkspaceGm, __ubuf__ uint64_t* TilingUint64Ub, COMP_T allAxis, __ubuf__ COMP_T* params)
{
    uint64_t dataStride[TILING_ARRAY_LEN] = {};
    uint64_t indicesStride[TILING_ARRAY_LEN] = {};
    uint64_t updatesStride[TILING_ARRAY_LEN] = {};
    for (uint32_t i = 0; i < TILING_DATA_UINT64_NUM; i++) {
        if (i < TILING_ARRAY_LEN) {
            dataStride[i] = TilingUint64Ub[i];
        } else if (i < TWO_TILING_ARRAY_LEN) {
            indicesStride[i - TILING_ARRAY_LEN] = TilingUint64Ub[i];
        } else if (i < THREE_TILING_ARRAY_LEN) {
            updatesStride[i - TWO_TILING_ARRAY_LEN] = TilingUint64Ub[i];
        }
    }

    COMP_T m0 = params[0];
    COMP_T m1 = params[1];
    COMP_T m2 = params[2];
    COMP_T m3 = params[3];
    COMP_T m4 = params[4];
    COMP_T m5 = params[5];
    COMP_T m6 = params[6];
    COMP_T shift0 = params[7];
    COMP_T shift1 = params[8];
    COMP_T shift2 = params[9];
    COMP_T shift3 = params[10];
    COMP_T shift4 = params[11];
    COMP_T shift5 = params[12];
    COMP_T shift6 = params[13];

    for (COMP_T i = Simt::GetBlockIdx() * Simt::GetThreadNum() + Simt::GetThreadIdx(); i < allAxis;
         i += Simt::GetBlockNum() * Simt::GetThreadNum()) {
        COMP_T dim0Idx = Simt::UintDiv(i, m0, shift0);
        COMP_T dim0Rem = i - dim0Idx * indicesStride[0];

        COMP_T dim1Idx = Simt::UintDiv(dim0Rem, m1, shift1);
        COMP_T dim1Rem = dim0Rem - dim1Idx * indicesStride[DIM_1];

        COMP_T dim2Idx = Simt::UintDiv(dim1Rem, m2, shift2);
        COMP_T dim2Rem = dim1Rem - dim2Idx * indicesStride[DIM_2];

        COMP_T dim3Idx = Simt::UintDiv(dim2Rem, m3, shift3);
        COMP_T dim3Rem = dim2Rem - dim3Idx * indicesStride[DIM_3];

        COMP_T dim4Idx = Simt::UintDiv(dim3Rem, m4, shift4);
        COMP_T dim4Rem = dim3Rem - dim4Idx * indicesStride[DIM_4];

        COMP_T dim5Idx = Simt::UintDiv(dim4Rem, m5, shift5);
        COMP_T dim5Rem = dim4Rem - dim5Idx * indicesStride[DIM_5];

        COMP_T dim6Idx = Simt::UintDiv(dim5Rem, m6, shift6);
        COMP_T dim7Idx = dim5Rem - dim6Idx * indicesStride[DIM_6];

        COMP_T yOffset = 0;
        if constexpr (DIM == 0) {
            yOffset = static_cast<COMP_T>(indices[i]) * dataStride[0] + dim1Idx * dataStride[DIM_1] +
                      dim2Idx * dataStride[DIM_2] + dim3Idx * dataStride[DIM_3] +
                      dim4Idx * dataStride[DIM_4] + dim5Idx * dataStride[DIM_5] +
                      dim6Idx * dataStride[DIM_6] + dim7Idx;
        } else if constexpr (DIM == 1) {
            yOffset = dim0Idx * dataStride[0] + static_cast<COMP_T>(indices[i]) * dataStride[DIM_1] +
                      dim2Idx * dataStride[DIM_2] + dim3Idx * dataStride[DIM_3] +
                      dim4Idx * dataStride[DIM_4] + dim5Idx * dataStride[DIM_5] +
                      dim6Idx * dataStride[DIM_6] + dim7Idx;
        } else if constexpr (DIM == 2) {
            yOffset = dim0Idx * dataStride[0] + dim1Idx * dataStride[DIM_1] +
                      static_cast<COMP_T>(indices[i]) * dataStride[DIM_2] +
                      dim3Idx * dataStride[DIM_3] + dim4Idx * dataStride[DIM_4] +
                      dim5Idx * dataStride[DIM_5] + dim6Idx * dataStride[DIM_6] + dim7Idx;
        } else if constexpr (DIM == 3) {
            yOffset = dim0Idx * dataStride[0] + dim1Idx * dataStride[DIM_1] +
                      dim2Idx * dataStride[DIM_2] +
                      static_cast<COMP_T>(indices[i]) * dataStride[DIM_3] +
                      dim4Idx * dataStride[DIM_4] + dim5Idx * dataStride[DIM_5] +
                      dim6Idx * dataStride[DIM_6] + dim7Idx;
        } else if constexpr (DIM == 4) {
            yOffset = dim0Idx * dataStride[0] + dim1Idx * dataStride[DIM_1] +
                      dim2Idx * dataStride[DIM_2] + dim3Idx * dataStride[DIM_3] +
                      static_cast<COMP_T>(indices[i]) * dataStride[DIM_4] +
                      dim5Idx * dataStride[DIM_5] + dim6Idx * dataStride[DIM_6] + dim7Idx;
        } else if constexpr (DIM == 5) {
            yOffset = dim0Idx * dataStride[0] + dim1Idx * dataStride[DIM_1] +
                      dim2Idx * dataStride[DIM_2] + dim3Idx * dataStride[DIM_3] +
                      dim4Idx * dataStride[DIM_4] +
                      static_cast<COMP_T>(indices[i]) * dataStride[DIM_5] +
                      dim6Idx * dataStride[DIM_6] + dim7Idx;
        } else if constexpr (DIM == 6) {
            yOffset = dim0Idx * dataStride[0] + dim1Idx * dataStride[DIM_1] +
                      dim2Idx * dataStride[DIM_2] + dim3Idx * dataStride[DIM_3] +
                      dim4Idx * dataStride[DIM_4] + dim5Idx * dataStride[DIM_5] +
                      static_cast<COMP_T>(indices[i]) * dataStride[DIM_6] + dim7Idx;
        } else {
            yOffset = dim0Idx * dataStride[0] + dim1Idx * dataStride[DIM_1] +
                      dim2Idx * dataStride[DIM_2] + dim3Idx * dataStride[DIM_3] +
                      dim4Idx * dataStride[DIM_4] + dim5Idx * dataStride[DIM_5] +
                      dim6Idx * dataStride[DIM_6] + static_cast<COMP_T>(indices[i]);
        }
        COMP_T updatesOffset = dim0Idx * updatesStride[0] + dim1Idx * updatesStride[DIM_1] +
                               dim2Idx * updatesStride[DIM_2] + dim3Idx * updatesStride[DIM_3] +
                               dim4Idx * updatesStride[DIM_4] + dim5Idx * updatesStride[DIM_5] +
                               dim6Idx * updatesStride[DIM_6] + dim7Idx;
        ReplaceOut<DATA_T, COMP_T, CAST_T, REDU>(updates, y, updatesWorkspaceGm, xWorkspaceGm, yOffset, updatesOffset);
    }
}

template <typename DATA_T, typename IDX_T, typename COMP_T, typename CAST_T, const uint32_t REDU, const uint32_t TEMPLATE_V2>
__aicore__ inline void KernelScatterElements<DATA_T, IDX_T, COMP_T, CAST_T, REDU, TEMPLATE_V2>::Process()
{
    if (tilingData_->allAxis == 0) {
        if constexpr (TEMPLATE_V2 == 0) {
            CopyDataToY();
        }
        return;
    }
    
    if constexpr ((REDU == REDU_ADD) && (IsSameType<DATA_T, int8_t>::value || IsSameType<DATA_T, uint8_t>::value ||
                                         IsSameType<DATA_T, int16_t>::value)) {
        if (blockIdx_ < usedCoreNum_) {
            CopyDataToWs();
        }
        if (blockIdx_ < usedCoreNum2_) {
            CopyUpdatesToWs();
        }
    } else {
        if (blockIdx_ < usedCoreNum_) {
            if constexpr (TEMPLATE_V2 == 0) {
                CopyDataToY();
            }
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

    bool useThread1024 = false;
    if (tilingData_->rank >= DIM_1 && tilingData_->rank <=DIM_3 && tilingData_->rank == tilingData_->dim + 1 && dataAxis_ > THREAD_SUM) {
         useThread1024 = true;
    }

    if (tilingData_->rank == DIM_1) {
        if (useThread1024) {
            Simt::VF_CALL<SimtComputeDim1<DATA_T, IDX_T, COMP_T, CAST_T, REDU, USED_THREAD1024>>(
            Simt::Dim3(USED_THREAD1024), (__gm__ IDX_T*)(indices_.GetPhyAddr()), (__gm__ DATA_T*)(updates_.GetPhyAddr()),
            (__gm__ DATA_T*)(y_.GetPhyAddr()), (__gm__ CAST_T*)(updatesWorkspaceGm_.GetPhyAddr()),
            (__gm__ CAST_T*)(xWorkspaceGm_.GetPhyAddr()), allAxis_);
        } else {
            Simt::VF_CALL<SimtComputeDim1<DATA_T, IDX_T, COMP_T, CAST_T, REDU, USED_THREAD>>(
            Simt::Dim3(USED_THREAD), (__gm__ IDX_T*)(indices_.GetPhyAddr()), (__gm__ DATA_T*)(updates_.GetPhyAddr()),
            (__gm__ DATA_T*)(y_.GetPhyAddr()), (__gm__ CAST_T*)(updatesWorkspaceGm_.GetPhyAddr()),
            (__gm__ CAST_T*)(xWorkspaceGm_.GetPhyAddr()), allAxis_);
        }
    } else if (tilingData_->rank == DIM_2) {
        COMP_T m0 = 1;
        COMP_T shift0 = 1;
        GetUintDivMagicAndShift(m0, shift0, static_cast<COMP_T>(tilingData_->indicesStride[0]));
        if (tilingData_->dim == 0) {
            if (useThread1024) {
                Simt::VF_CALL<SimtComputeDim2<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 0, USED_THREAD1024>>(
                Simt::Dim3(USED_THREAD1024), (__gm__ IDX_T*)(indices_.GetPhyAddr()),
                (__gm__ DATA_T*)(updates_.GetPhyAddr()), (__gm__ DATA_T*)(y_.GetPhyAddr()),
                (__gm__ CAST_T*)(updatesWorkspaceGm_.GetPhyAddr()), (__gm__ CAST_T*)(xWorkspaceGm_.GetPhyAddr()),
                (__ubuf__ uint64_t*)(TilingUint64Ub.GetPhyAddr()), allAxis_, m0, shift0);
            } else {
                Simt::VF_CALL<SimtComputeDim2<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 0, USED_THREAD>>(
                Simt::Dim3(USED_THREAD), (__gm__ IDX_T*)(indices_.GetPhyAddr()),
                (__gm__ DATA_T*)(updates_.GetPhyAddr()), (__gm__ DATA_T*)(y_.GetPhyAddr()),
                (__gm__ CAST_T*)(updatesWorkspaceGm_.GetPhyAddr()), (__gm__ CAST_T*)(xWorkspaceGm_.GetPhyAddr()),
                (__ubuf__ uint64_t*)(TilingUint64Ub.GetPhyAddr()), allAxis_, m0, shift0);
            }
        } else {
            if (useThread1024) {
                Simt::VF_CALL<SimtComputeDim2<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 1, USED_THREAD1024>>(
                Simt::Dim3(USED_THREAD1024), (__gm__ IDX_T*)(indices_.GetPhyAddr()),
                (__gm__ DATA_T*)(updates_.GetPhyAddr()), (__gm__ DATA_T*)(y_.GetPhyAddr()),
                (__gm__ CAST_T*)(updatesWorkspaceGm_.GetPhyAddr()), (__gm__ CAST_T*)(xWorkspaceGm_.GetPhyAddr()),
                (__ubuf__ uint64_t*)(TilingUint64Ub.GetPhyAddr()), allAxis_, m0, shift0);
            } else {
                Simt::VF_CALL<SimtComputeDim2<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 1, USED_THREAD>>(
                Simt::Dim3(USED_THREAD), (__gm__ IDX_T*)(indices_.GetPhyAddr()),
                (__gm__ DATA_T*)(updates_.GetPhyAddr()), (__gm__ DATA_T*)(y_.GetPhyAddr()),
                (__gm__ CAST_T*)(updatesWorkspaceGm_.GetPhyAddr()), (__gm__ CAST_T*)(xWorkspaceGm_.GetPhyAddr()),
                (__ubuf__ uint64_t*)(TilingUint64Ub.GetPhyAddr()), allAxis_, m0, shift0);
            }
        }
    } else if (tilingData_->rank == DIM_3) {
        COMP_T m_[2] = {1, 1};
        COMP_T shift_[2] = {1, 1};
        GetUintDivMagicAndShift(m_[0], shift_[0], static_cast<COMP_T>(tilingData_->indicesStride[0]));
        GetUintDivMagicAndShift(m_[DIM_1], shift_[DIM_1], static_cast<COMP_T>(tilingData_->indicesStride[DIM_1]));
        if (tilingData_->dim == 0) {
            if (useThread1024) {
                Simt::VF_CALL<SimtComputeDim3<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 0, USED_THREAD1024>>(
                Simt::Dim3(USED_THREAD1024), (__gm__ IDX_T*)(indices_.GetPhyAddr()),
                (__gm__ DATA_T*)(updates_.GetPhyAddr()), (__gm__ DATA_T*)(y_.GetPhyAddr()),
                (__gm__ CAST_T*)(updatesWorkspaceGm_.GetPhyAddr()), (__gm__ CAST_T*)(xWorkspaceGm_.GetPhyAddr()),
                (__ubuf__ uint64_t*)(TilingUint64Ub.GetPhyAddr()), allAxis_, m_[0], shift_[0], m_[DIM_1], shift_[DIM_1]);
            } else {
                Simt::VF_CALL<SimtComputeDim3<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 0, USED_THREAD>>(
                Simt::Dim3(USED_THREAD), (__gm__ IDX_T*)(indices_.GetPhyAddr()),
                (__gm__ DATA_T*)(updates_.GetPhyAddr()), (__gm__ DATA_T*)(y_.GetPhyAddr()),
                (__gm__ CAST_T*)(updatesWorkspaceGm_.GetPhyAddr()), (__gm__ CAST_T*)(xWorkspaceGm_.GetPhyAddr()),
                (__ubuf__ uint64_t*)(TilingUint64Ub.GetPhyAddr()), allAxis_, m_[0], shift_[0], m_[DIM_1], shift_[DIM_1]);
            }
        } else if (tilingData_->dim == 1) {
            if (useThread1024) {
                Simt::VF_CALL<SimtComputeDim3<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 1, USED_THREAD1024>>(
                Simt::Dim3(USED_THREAD1024), (__gm__ IDX_T*)(indices_.GetPhyAddr()),
                (__gm__ DATA_T*)(updates_.GetPhyAddr()), (__gm__ DATA_T*)(y_.GetPhyAddr()),
                (__gm__ CAST_T*)(updatesWorkspaceGm_.GetPhyAddr()), (__gm__ CAST_T*)(xWorkspaceGm_.GetPhyAddr()),
                (__ubuf__ uint64_t*)(TilingUint64Ub.GetPhyAddr()), allAxis_, m_[0], shift_[0], m_[DIM_1], shift_[DIM_1]);
            } else {
                Simt::VF_CALL<SimtComputeDim3<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 1, USED_THREAD>>(
                Simt::Dim3(USED_THREAD), (__gm__ IDX_T*)(indices_.GetPhyAddr()),
                (__gm__ DATA_T*)(updates_.GetPhyAddr()), (__gm__ DATA_T*)(y_.GetPhyAddr()),
                (__gm__ CAST_T*)(updatesWorkspaceGm_.GetPhyAddr()), (__gm__ CAST_T*)(xWorkspaceGm_.GetPhyAddr()),
                (__ubuf__ uint64_t*)(TilingUint64Ub.GetPhyAddr()), allAxis_, m_[0], shift_[0], m_[DIM_1], shift_[DIM_1]);
            }
        } else {
            if (useThread1024) {
                Simt::VF_CALL<SimtComputeDim3<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 2, USED_THREAD1024>>(
                Simt::Dim3(USED_THREAD1024), (__gm__ IDX_T*)(indices_.GetPhyAddr()),
                (__gm__ DATA_T*)(updates_.GetPhyAddr()), (__gm__ DATA_T*)(y_.GetPhyAddr()),
                (__gm__ CAST_T*)(updatesWorkspaceGm_.GetPhyAddr()), (__gm__ CAST_T*)(xWorkspaceGm_.GetPhyAddr()),
                (__ubuf__ uint64_t*)(TilingUint64Ub.GetPhyAddr()), allAxis_, m_[0], shift_[0], m_[DIM_1], shift_[DIM_1]);
            } else {
                Simt::VF_CALL<SimtComputeDim3<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 2, USED_THREAD>>(
                Simt::Dim3(USED_THREAD), (__gm__ IDX_T*)(indices_.GetPhyAddr()),
                (__gm__ DATA_T*)(updates_.GetPhyAddr()), (__gm__ DATA_T*)(y_.GetPhyAddr()),
                (__gm__ CAST_T*)(updatesWorkspaceGm_.GetPhyAddr()), (__gm__ CAST_T*)(xWorkspaceGm_.GetPhyAddr()),
                (__ubuf__ uint64_t*)(TilingUint64Ub.GetPhyAddr()), allAxis_, m_[0], shift_[0], m_[DIM_1], shift_[DIM_1]);
            }
        }
    } else if (tilingData_->rank == DIM_4) {
        COMP_T m_[3] = {1, 1, 1};
        COMP_T shift_[3] = {1, 1, 1};
        GetUintDivMagicAndShift(m_[0], shift_[0], static_cast<COMP_T>(tilingData_->indicesStride[0]));
        GetUintDivMagicAndShift(m_[DIM_1], shift_[DIM_1], static_cast<COMP_T>(tilingData_->indicesStride[DIM_1]));
        GetUintDivMagicAndShift(m_[DIM_2], shift_[DIM_2], static_cast<COMP_T>(tilingData_->indicesStride[DIM_2]));
        if (tilingData_->dim == 0) {
            Simt::VF_CALL<SimtComputeDim4<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 0>>(
                Simt::Dim3(USED_THREAD), (__gm__ IDX_T*)(indices_.GetPhyAddr()),
                (__gm__ DATA_T*)(updates_.GetPhyAddr()), (__gm__ DATA_T*)(y_.GetPhyAddr()),
                (__gm__ CAST_T*)(updatesWorkspaceGm_.GetPhyAddr()), (__gm__ CAST_T*)(xWorkspaceGm_.GetPhyAddr()),
                (__ubuf__ uint64_t*)(TilingUint64Ub.GetPhyAddr()), allAxis_, m_[0], shift_[0], m_[DIM_1], shift_[DIM_1],
                m_[DIM_2], shift_[DIM_2]);
        } else if (tilingData_->dim == 1) {
            Simt::VF_CALL<SimtComputeDim4<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 1>>(
                Simt::Dim3(USED_THREAD), (__gm__ IDX_T*)(indices_.GetPhyAddr()),
                (__gm__ DATA_T*)(updates_.GetPhyAddr()), (__gm__ DATA_T*)(y_.GetPhyAddr()),
                (__gm__ CAST_T*)(updatesWorkspaceGm_.GetPhyAddr()), (__gm__ CAST_T*)(xWorkspaceGm_.GetPhyAddr()),
                (__ubuf__ uint64_t*)(TilingUint64Ub.GetPhyAddr()), allAxis_, m_[0], shift_[0], m_[DIM_1], shift_[DIM_1],
                m_[DIM_2], shift_[DIM_2]);
        } else if (tilingData_->dim == 2) {
            Simt::VF_CALL<SimtComputeDim4<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 2>>(
                Simt::Dim3(USED_THREAD), (__gm__ IDX_T*)(indices_.GetPhyAddr()),
                (__gm__ DATA_T*)(updates_.GetPhyAddr()), (__gm__ DATA_T*)(y_.GetPhyAddr()),
                (__gm__ CAST_T*)(updatesWorkspaceGm_.GetPhyAddr()), (__gm__ CAST_T*)(xWorkspaceGm_.GetPhyAddr()),
                (__ubuf__ uint64_t*)(TilingUint64Ub.GetPhyAddr()), allAxis_, m_[0], shift_[0], m_[DIM_1], shift_[DIM_1],
                m_[DIM_2], shift_[DIM_2]);
        } else {
            Simt::VF_CALL<SimtComputeDim4<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 3>>(
                Simt::Dim3(USED_THREAD), (__gm__ IDX_T*)(indices_.GetPhyAddr()),
                (__gm__ DATA_T*)(updates_.GetPhyAddr()), (__gm__ DATA_T*)(y_.GetPhyAddr()),
                (__gm__ CAST_T*)(updatesWorkspaceGm_.GetPhyAddr()), (__gm__ CAST_T*)(xWorkspaceGm_.GetPhyAddr()),
                (__ubuf__ uint64_t*)(TilingUint64Ub.GetPhyAddr()), allAxis_, m_[0], shift_[0], m_[DIM_1], shift_[DIM_1],
                m_[DIM_2], shift_[DIM_2]);
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
            Simt::VF_CALL<SimtComputeDim5<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 0>>(
                Simt::Dim3(USED_THREAD), (__gm__ IDX_T*)(indices_.GetPhyAddr()),
                (__gm__ DATA_T*)(updates_.GetPhyAddr()), (__gm__ DATA_T*)(y_.GetPhyAddr()),
                (__gm__ CAST_T*)(updatesWorkspaceGm_.GetPhyAddr()), (__gm__ CAST_T*)(xWorkspaceGm_.GetPhyAddr()),
                (__ubuf__ uint64_t*)(TilingUint64Ub.GetPhyAddr()), allAxis_, (__ubuf__ COMP_T*)(ParamDim5Ub.GetPhyAddr()));
        } else if (tilingData_->dim == 1) {
            Simt::VF_CALL<SimtComputeDim5<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 1>>(
                Simt::Dim3(USED_THREAD), (__gm__ IDX_T*)(indices_.GetPhyAddr()),
                (__gm__ DATA_T*)(updates_.GetPhyAddr()), (__gm__ DATA_T*)(y_.GetPhyAddr()),
                (__gm__ CAST_T*)(updatesWorkspaceGm_.GetPhyAddr()), (__gm__ CAST_T*)(xWorkspaceGm_.GetPhyAddr()),
                (__ubuf__ uint64_t*)(TilingUint64Ub.GetPhyAddr()), allAxis_, (__ubuf__ COMP_T*)(ParamDim5Ub.GetPhyAddr()));
        } else if (tilingData_->dim == 2) {
            Simt::VF_CALL<SimtComputeDim5<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 2>>(
                Simt::Dim3(USED_THREAD), (__gm__ IDX_T*)(indices_.GetPhyAddr()),
                (__gm__ DATA_T*)(updates_.GetPhyAddr()), (__gm__ DATA_T*)(y_.GetPhyAddr()),
                (__gm__ CAST_T*)(updatesWorkspaceGm_.GetPhyAddr()), (__gm__ CAST_T*)(xWorkspaceGm_.GetPhyAddr()),
                (__ubuf__ uint64_t*)(TilingUint64Ub.GetPhyAddr()), allAxis_, (__ubuf__ COMP_T*)(ParamDim5Ub.GetPhyAddr()));
        } else if (tilingData_->dim == 3) {
            Simt::VF_CALL<SimtComputeDim5<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 3>>(
                Simt::Dim3(USED_THREAD), (__gm__ IDX_T*)(indices_.GetPhyAddr()),
                (__gm__ DATA_T*)(updates_.GetPhyAddr()), (__gm__ DATA_T*)(y_.GetPhyAddr()),
                (__gm__ CAST_T*)(updatesWorkspaceGm_.GetPhyAddr()), (__gm__ CAST_T*)(xWorkspaceGm_.GetPhyAddr()),
                (__ubuf__ uint64_t*)(TilingUint64Ub.GetPhyAddr()), allAxis_, (__ubuf__ COMP_T*)(ParamDim5Ub.GetPhyAddr()));
        } else {
            Simt::VF_CALL<SimtComputeDim5<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 4>>(
                Simt::Dim3(USED_THREAD), (__gm__ IDX_T*)(indices_.GetPhyAddr()),
                (__gm__ DATA_T*)(updates_.GetPhyAddr()), (__gm__ DATA_T*)(y_.GetPhyAddr()),
                (__gm__ CAST_T*)(updatesWorkspaceGm_.GetPhyAddr()), (__gm__ CAST_T*)(xWorkspaceGm_.GetPhyAddr()),
                (__ubuf__ uint64_t*)(TilingUint64Ub.GetPhyAddr()), allAxis_, (__ubuf__ COMP_T*)(ParamDim5Ub.GetPhyAddr()));
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
            Simt::VF_CALL<SimtComputeDim6<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 0>>(
                Simt::Dim3(USED_THREAD), (__gm__ IDX_T*)(indices_.GetPhyAddr()),
                (__gm__ DATA_T*)(updates_.GetPhyAddr()), (__gm__ DATA_T*)(y_.GetPhyAddr()),
                (__gm__ CAST_T*)(updatesWorkspaceGm_.GetPhyAddr()), (__gm__ CAST_T*)(xWorkspaceGm_.GetPhyAddr()),
                (__ubuf__ uint64_t*)(TilingUint64Ub.GetPhyAddr()), allAxis_, (__ubuf__ COMP_T*)(ParamDim6Ub.GetPhyAddr()));
        } else if (tilingData_->dim == 1) {
            Simt::VF_CALL<SimtComputeDim6<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 1>>(
                Simt::Dim3(USED_THREAD), (__gm__ IDX_T*)(indices_.GetPhyAddr()),
                (__gm__ DATA_T*)(updates_.GetPhyAddr()), (__gm__ DATA_T*)(y_.GetPhyAddr()),
                (__gm__ CAST_T*)(updatesWorkspaceGm_.GetPhyAddr()), (__gm__ CAST_T*)(xWorkspaceGm_.GetPhyAddr()),
                (__ubuf__ uint64_t*)(TilingUint64Ub.GetPhyAddr()), allAxis_, (__ubuf__ COMP_T*)(ParamDim6Ub.GetPhyAddr()));
        } else if (tilingData_->dim == 2) {
            Simt::VF_CALL<SimtComputeDim6<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 2>>(
                Simt::Dim3(USED_THREAD), (__gm__ IDX_T*)(indices_.GetPhyAddr()),
                (__gm__ DATA_T*)(updates_.GetPhyAddr()), (__gm__ DATA_T*)(y_.GetPhyAddr()),
                (__gm__ CAST_T*)(updatesWorkspaceGm_.GetPhyAddr()), (__gm__ CAST_T*)(xWorkspaceGm_.GetPhyAddr()),
                (__ubuf__ uint64_t*)(TilingUint64Ub.GetPhyAddr()), allAxis_, (__ubuf__ COMP_T*)(ParamDim6Ub.GetPhyAddr()));
        } else if (tilingData_->dim == 3) {
            Simt::VF_CALL<SimtComputeDim6<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 3>>(
                Simt::Dim3(USED_THREAD), (__gm__ IDX_T*)(indices_.GetPhyAddr()),
                (__gm__ DATA_T*)(updates_.GetPhyAddr()), (__gm__ DATA_T*)(y_.GetPhyAddr()),
                (__gm__ CAST_T*)(updatesWorkspaceGm_.GetPhyAddr()), (__gm__ CAST_T*)(xWorkspaceGm_.GetPhyAddr()),
                (__ubuf__ uint64_t*)(TilingUint64Ub.GetPhyAddr()), allAxis_, (__ubuf__ COMP_T*)(ParamDim6Ub.GetPhyAddr()));
        } else if (tilingData_->dim == 4) {
            Simt::VF_CALL<SimtComputeDim6<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 4>>(
                Simt::Dim3(USED_THREAD), (__gm__ IDX_T*)(indices_.GetPhyAddr()),
                (__gm__ DATA_T*)(updates_.GetPhyAddr()), (__gm__ DATA_T*)(y_.GetPhyAddr()),
                (__gm__ CAST_T*)(updatesWorkspaceGm_.GetPhyAddr()), (__gm__ CAST_T*)(xWorkspaceGm_.GetPhyAddr()),
                (__ubuf__ uint64_t*)(TilingUint64Ub.GetPhyAddr()), allAxis_, (__ubuf__ COMP_T*)(ParamDim6Ub.GetPhyAddr()));
        } else {
            Simt::VF_CALL<SimtComputeDim6<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 5>>(
                Simt::Dim3(USED_THREAD), (__gm__ IDX_T*)(indices_.GetPhyAddr()),
                (__gm__ DATA_T*)(updates_.GetPhyAddr()), (__gm__ DATA_T*)(y_.GetPhyAddr()),
                (__gm__ CAST_T*)(updatesWorkspaceGm_.GetPhyAddr()), (__gm__ CAST_T*)(xWorkspaceGm_.GetPhyAddr()),
                (__ubuf__ uint64_t*)(TilingUint64Ub.GetPhyAddr()), allAxis_, (__ubuf__ COMP_T*)(ParamDim6Ub.GetPhyAddr()));
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
            Simt::VF_CALL<SimtComputeDim7<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 0>>(
                Simt::Dim3(USED_THREAD), (__gm__ IDX_T*)(indices_.GetPhyAddr()),
                (__gm__ DATA_T*)(updates_.GetPhyAddr()), (__gm__ DATA_T*)(y_.GetPhyAddr()),
                (__gm__ CAST_T*)(updatesWorkspaceGm_.GetPhyAddr()), (__gm__ CAST_T*)(xWorkspaceGm_.GetPhyAddr()),
                (__ubuf__ uint64_t*)(TilingUint64Ub.GetPhyAddr()), allAxis_, (__ubuf__ COMP_T*)(ParamDim7Ub.GetPhyAddr()));
        } else if (tilingData_->dim == 1) {
            Simt::VF_CALL<SimtComputeDim7<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 1>>(
                Simt::Dim3(USED_THREAD), (__gm__ IDX_T*)(indices_.GetPhyAddr()),
                (__gm__ DATA_T*)(updates_.GetPhyAddr()), (__gm__ DATA_T*)(y_.GetPhyAddr()),
                (__gm__ CAST_T*)(updatesWorkspaceGm_.GetPhyAddr()), (__gm__ CAST_T*)(xWorkspaceGm_.GetPhyAddr()),
                (__ubuf__ uint64_t*)(TilingUint64Ub.GetPhyAddr()), allAxis_, (__ubuf__ COMP_T*)(ParamDim7Ub.GetPhyAddr()));
        } else if (tilingData_->dim == 2) {
            Simt::VF_CALL<SimtComputeDim7<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 2>>(
                Simt::Dim3(USED_THREAD), (__gm__ IDX_T*)(indices_.GetPhyAddr()),
                (__gm__ DATA_T*)(updates_.GetPhyAddr()), (__gm__ DATA_T*)(y_.GetPhyAddr()),
                (__gm__ CAST_T*)(updatesWorkspaceGm_.GetPhyAddr()), (__gm__ CAST_T*)(xWorkspaceGm_.GetPhyAddr()),
                (__ubuf__ uint64_t*)(TilingUint64Ub.GetPhyAddr()), allAxis_, (__ubuf__ COMP_T*)(ParamDim7Ub.GetPhyAddr()));
        } else if (tilingData_->dim == 3) {
            Simt::VF_CALL<SimtComputeDim7<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 3>>(
                Simt::Dim3(USED_THREAD), (__gm__ IDX_T*)(indices_.GetPhyAddr()),
                (__gm__ DATA_T*)(updates_.GetPhyAddr()), (__gm__ DATA_T*)(y_.GetPhyAddr()),
                (__gm__ CAST_T*)(updatesWorkspaceGm_.GetPhyAddr()), (__gm__ CAST_T*)(xWorkspaceGm_.GetPhyAddr()),
                (__ubuf__ uint64_t*)(TilingUint64Ub.GetPhyAddr()), allAxis_, (__ubuf__ COMP_T*)(ParamDim7Ub.GetPhyAddr()));
        } else if (tilingData_->dim == 4) {
            Simt::VF_CALL<SimtComputeDim7<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 4>>(
                Simt::Dim3(USED_THREAD), (__gm__ IDX_T*)(indices_.GetPhyAddr()),
                (__gm__ DATA_T*)(updates_.GetPhyAddr()), (__gm__ DATA_T*)(y_.GetPhyAddr()),
                (__gm__ CAST_T*)(updatesWorkspaceGm_.GetPhyAddr()), (__gm__ CAST_T*)(xWorkspaceGm_.GetPhyAddr()),
                (__ubuf__ uint64_t*)(TilingUint64Ub.GetPhyAddr()), allAxis_, (__ubuf__ COMP_T*)(ParamDim7Ub.GetPhyAddr()));
        } else if (tilingData_->dim == 5) {
            Simt::VF_CALL<SimtComputeDim7<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 5>>(
                Simt::Dim3(USED_THREAD), (__gm__ IDX_T*)(indices_.GetPhyAddr()),
                (__gm__ DATA_T*)(updates_.GetPhyAddr()), (__gm__ DATA_T*)(y_.GetPhyAddr()),
                (__gm__ CAST_T*)(updatesWorkspaceGm_.GetPhyAddr()), (__gm__ CAST_T*)(xWorkspaceGm_.GetPhyAddr()),
                (__ubuf__ uint64_t*)(TilingUint64Ub.GetPhyAddr()), allAxis_, (__ubuf__ COMP_T*)(ParamDim7Ub.GetPhyAddr()));
        } else {
            Simt::VF_CALL<SimtComputeDim7<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 6>>(
                Simt::Dim3(USED_THREAD), (__gm__ IDX_T*)(indices_.GetPhyAddr()),
                (__gm__ DATA_T*)(updates_.GetPhyAddr()), (__gm__ DATA_T*)(y_.GetPhyAddr()),
                (__gm__ CAST_T*)(updatesWorkspaceGm_.GetPhyAddr()), (__gm__ CAST_T*)(xWorkspaceGm_.GetPhyAddr()),
                (__ubuf__ uint64_t*)(TilingUint64Ub.GetPhyAddr()), allAxis_, (__ubuf__ COMP_T*)(ParamDim7Ub.GetPhyAddr()));
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
            Simt::VF_CALL<SimtComputeDim8<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 0>>(
                Simt::Dim3(USED_THREAD), (__gm__ IDX_T*)(indices_.GetPhyAddr()),
                (__gm__ DATA_T*)(updates_.GetPhyAddr()), (__gm__ DATA_T*)(y_.GetPhyAddr()),
                (__gm__ CAST_T*)(updatesWorkspaceGm_.GetPhyAddr()), (__gm__ CAST_T*)(xWorkspaceGm_.GetPhyAddr()),
                (__ubuf__ uint64_t*)(TilingUint64Ub.GetPhyAddr()), allAxis_, (__ubuf__ COMP_T*)(ParamDim8Ub.GetPhyAddr()));
        } else if (tilingData_->dim == 1) {
            Simt::VF_CALL<SimtComputeDim8<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 1>>(
                Simt::Dim3(USED_THREAD), (__gm__ IDX_T*)(indices_.GetPhyAddr()),
                (__gm__ DATA_T*)(updates_.GetPhyAddr()), (__gm__ DATA_T*)(y_.GetPhyAddr()),
                (__gm__ CAST_T*)(updatesWorkspaceGm_.GetPhyAddr()), (__gm__ CAST_T*)(xWorkspaceGm_.GetPhyAddr()),
                (__ubuf__ uint64_t*)(TilingUint64Ub.GetPhyAddr()), allAxis_, (__ubuf__ COMP_T*)(ParamDim8Ub.GetPhyAddr()));
        } else if (tilingData_->dim == 2) {
            Simt::VF_CALL<SimtComputeDim8<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 2>>(
                Simt::Dim3(USED_THREAD), (__gm__ IDX_T*)(indices_.GetPhyAddr()),
                (__gm__ DATA_T*)(updates_.GetPhyAddr()), (__gm__ DATA_T*)(y_.GetPhyAddr()),
                (__gm__ CAST_T*)(updatesWorkspaceGm_.GetPhyAddr()), (__gm__ CAST_T*)(xWorkspaceGm_.GetPhyAddr()),
                (__ubuf__ uint64_t*)(TilingUint64Ub.GetPhyAddr()), allAxis_, (__ubuf__ COMP_T*)(ParamDim8Ub.GetPhyAddr()));
        } else if (tilingData_->dim == 3) {
            Simt::VF_CALL<SimtComputeDim8<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 3>>(
                Simt::Dim3(USED_THREAD), (__gm__ IDX_T*)(indices_.GetPhyAddr()),
                (__gm__ DATA_T*)(updates_.GetPhyAddr()), (__gm__ DATA_T*)(y_.GetPhyAddr()),
                (__gm__ CAST_T*)(updatesWorkspaceGm_.GetPhyAddr()), (__gm__ CAST_T*)(xWorkspaceGm_.GetPhyAddr()),
                (__ubuf__ uint64_t*)(TilingUint64Ub.GetPhyAddr()), allAxis_, (__ubuf__ COMP_T*)(ParamDim8Ub.GetPhyAddr()));
        } else if (tilingData_->dim == 4) {
            Simt::VF_CALL<SimtComputeDim8<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 4>>(
                Simt::Dim3(USED_THREAD), (__gm__ IDX_T*)(indices_.GetPhyAddr()),
                (__gm__ DATA_T*)(updates_.GetPhyAddr()), (__gm__ DATA_T*)(y_.GetPhyAddr()),
                (__gm__ CAST_T*)(updatesWorkspaceGm_.GetPhyAddr()), (__gm__ CAST_T*)(xWorkspaceGm_.GetPhyAddr()),
                (__ubuf__ uint64_t*)(TilingUint64Ub.GetPhyAddr()), allAxis_, (__ubuf__ COMP_T*)(ParamDim8Ub.GetPhyAddr()));
        } else if (tilingData_->dim == 5) {
            Simt::VF_CALL<SimtComputeDim8<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 5>>(
                Simt::Dim3(USED_THREAD), (__gm__ IDX_T*)(indices_.GetPhyAddr()),
                (__gm__ DATA_T*)(updates_.GetPhyAddr()), (__gm__ DATA_T*)(y_.GetPhyAddr()),
                (__gm__ CAST_T*)(updatesWorkspaceGm_.GetPhyAddr()), (__gm__ CAST_T*)(xWorkspaceGm_.GetPhyAddr()),
                (__ubuf__ uint64_t*)(TilingUint64Ub.GetPhyAddr()), allAxis_, (__ubuf__ COMP_T*)(ParamDim8Ub.GetPhyAddr()));
        } else if (tilingData_->dim == 6) {
            Simt::VF_CALL<SimtComputeDim8<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 6>>(
                Simt::Dim3(USED_THREAD), (__gm__ IDX_T*)(indices_.GetPhyAddr()),
                (__gm__ DATA_T*)(updates_.GetPhyAddr()), (__gm__ DATA_T*)(y_.GetPhyAddr()),
                (__gm__ CAST_T*)(updatesWorkspaceGm_.GetPhyAddr()), (__gm__ CAST_T*)(xWorkspaceGm_.GetPhyAddr()),
                (__ubuf__ uint64_t*)(TilingUint64Ub.GetPhyAddr()), allAxis_, (__ubuf__ COMP_T*)(ParamDim8Ub.GetPhyAddr()));
        } else {
            Simt::VF_CALL<SimtComputeDim8<DATA_T, IDX_T, COMP_T, CAST_T, REDU, 7>>(
                Simt::Dim3(USED_THREAD), (__gm__ IDX_T*)(indices_.GetPhyAddr()),
                (__gm__ DATA_T*)(updates_.GetPhyAddr()), (__gm__ DATA_T*)(y_.GetPhyAddr()),
                (__gm__ CAST_T*)(updatesWorkspaceGm_.GetPhyAddr()), (__gm__ CAST_T*)(xWorkspaceGm_.GetPhyAddr()),
                (__ubuf__ uint64_t*)(TilingUint64Ub.GetPhyAddr()), allAxis_, (__ubuf__ COMP_T*)(ParamDim8Ub.GetPhyAddr()));
        }
    }

    if constexpr ((REDU == REDU_ADD) && (IsSameType<DATA_T, int8_t>::value || IsSameType<DATA_T, uint8_t>::value ||
                                         IsSameType<DATA_T, int16_t>::value)) {
        SyncAll();
        if (blockIdx_ < usedCoreNum_) {
            CopyResToY();
        }
    }
}
}  // namespace ScatterElements

#endif