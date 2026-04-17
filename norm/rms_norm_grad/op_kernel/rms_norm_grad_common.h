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
 * \file rms_norm_grad_common.h
 * \brief RmsNormGrad Common File
 */
#ifndef RMS_NORM_GRAD_BASE_H
#define RMS_NORM_GRAD_BASE_H
#include "kernel_operator.h"
#include "impl/dav_c220/kernel_operator_reg_others_impl.h"
#include "reduce_common.h"

using namespace AscendC;
static volatile __gm__ uint32_t fixed_output_sync[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
constexpr int32_t BUFFER_NUM = 1;
constexpr int32_t BUFFER_NUM_DB = 2;
constexpr uint32_t FLOAT_DTYPE = 0;
constexpr uint32_t FLOAT16_DTYPE = 1;
constexpr uint32_t BFLOAT16_DTYPE = 2;
constexpr uint32_t ALIGN_32 = 8;
constexpr uint32_t ALIGN_16 = 16;
constexpr uint32_t CORE_NUM = 50;
constexpr int32_t BLOCK_SIZE = 32;
constexpr int32_t EACH_CORE_HANDLE_NUM = BLOCK_SIZE / sizeof(int32_t);
const int32_t CAL_ONE_BLOCK_FP32 = 8;
const uint32_t REDUCESUMTHRESHOLD = 64;
const uint32_t SMALLD_THRESHOLD = 640;
constexpr uint32_t ROW_FACTOR_SPLIT_D = 32;
constexpr int32_t DIM_NUM = 2;
constexpr int32_t DIM_N = 0;
constexpr int32_t DIM_D = 1;

constexpr int64_t DOUBLE_BUFFER = 2;
constexpr int64_t ROW_TEMPLATE = 180;
constexpr int64_t COL_TEMPLATE = 64;  // 按64列进行分核
constexpr int64_t UB_SIZE = 180 * 1024 + 2 * DOUBLE_BUFFER * COL_TEMPLATE * sizeof(float);
constexpr int64_t FLOAT_ALIGN = 8;
constexpr uint64_t B32_REPEAT_STRIDE = 8;

template <typename Tp, Tp v>
struct integral_constant {
    static constexpr Tp value = v;
};

namespace RmsNormGrad{
struct deterministic_struct {
    LocalTensor<float> buffer1_;
    LocalTensor<float> buffer2_;
    GlobalTensor<float> workspaceGmOri_;
    GlobalTensor<float> dgammaGm_;
};
}
using namespace RmsNormGrad;

using true_type = integral_constant<bool, true>;
using false_type = integral_constant<bool, false>;
template <typename, typename>
struct is_same : public false_type {};
template <typename Tp>
struct is_same<Tp, Tp> : public true_type {};

__aicore__ inline void ReduceSumFP32(
    uint32_t idx, LocalTensor<float>& dst_local, const LocalTensor<float>& src_local,
    const LocalTensor<float>& work_local, int32_t count, uint32_t col_val_align)
{
    if (g_coreType == AIV) {
        int32_t elementNumPerRep = ONE_REPEAT_BYTE_SIZE / sizeof(float);
        uint64_t mask = elementNumPerRep;
        int32_t repeatTimes = count / elementNumPerRep;
        int32_t tailCount = count % elementNumPerRep;
        int32_t bodyCount = repeatTimes * elementNumPerRep;
        BinaryRepeatParams repeatParams;
        repeatParams.src0RepStride = ONE_REPEAT_BYTE_SIZE / ONE_BLK_SIZE;
        repeatParams.src0BlkStride = 1;
        repeatParams.src1RepStride = 0;
        repeatParams.src1BlkStride = 1;
        repeatParams.dstRepStride = 0;
        repeatParams.dstBlkStride = 1;
        Duplicate(work_local, 0.0f, elementNumPerRep);
        PipeBarrier<PIPE_V>();
        int32_t start_addr = idx * col_val_align;
        if (likely(repeatTimes > 0)) {
            Add(work_local, src_local[start_addr], work_local, mask, repeatTimes, repeatParams);
            PipeBarrier<PIPE_V>();
        }
        if (unlikely(tailCount != 0)) {
            Add(work_local, src_local[start_addr + bodyCount], work_local, tailCount, 1, repeatParams);
            PipeBarrier<PIPE_V>();
        }
        AscendCUtils::SetMask<float>(elementNumPerRep);
        ReduceSum(dst_local[start_addr], work_local, work_local, elementNumPerRep);
        PipeBarrier<PIPE_V>();
    }
}

/*
 * only support count <= 255 * 64 = 16320
 */
__aicore__ inline float ReduceSumFP32_V2(const LocalTensor<float>& src_local, int32_t count)
{
    int32_t elementNumPerRep = ONE_REPEAT_BYTE_SIZE / sizeof(float);
    int32_t repeatTimes = count / elementNumPerRep;
    int32_t tailCount = count % elementNumPerRep;
    int32_t bodyCount = repeatTimes * elementNumPerRep;
#ifdef __CCE_KT_TEST__
    assert(count <= MAX_REPEAT_TIMES * elementNumPerRep);
#endif
    float value = 0.0;
    if (g_coreType == AIV) {
        if (likely(repeatTimes > 0)) {
            AscendCUtils::SetMask<float>(elementNumPerRep);
            ReduceSum(src_local, src_local, src_local, elementNumPerRep);
            event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventId);
            WaitFlag<HardEvent::V_S>(eventId);
#ifdef __CCE_KT_TEST__
            uint64_t acc_val = GetAccVal();
#else
            uint64_t acc_val = GetAccVal();
#endif
            value = *reinterpret_cast<float*>(&acc_val);
        }
        if (unlikely(tailCount != 0)) {
            AscendCUtils::SetMask<float>(tailCount);
            ReduceSum(src_local[bodyCount], src_local[bodyCount], src_local[bodyCount], tailCount);
            event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventId);
            WaitFlag<HardEvent::V_S>(eventId);
#ifdef __CCE_KT_TEST__
            uint64_t acc_val = GetAccVal();
#else
            uint64_t acc_val = GetAccVal();
#endif
            value += *reinterpret_cast<float*>(&acc_val);
        }
    }
    return value;
}

template <typename T>
__aicore__ inline void Cast2FloatIf(LocalTensor<float>& castLocal, uint32_t srcOffset, uint32_t calcCount)
{
    if constexpr (!is_same<T, float>::value) {
        LocalTensor<T> castLocalB16 = castLocal.ReinterpretCast<T>();
        Cast(castLocal, castLocalB16[srcOffset], RoundMode::CAST_NONE, calcCount);
        PipeBarrier<PIPE_V>();
    }
}

__aicore__ inline uint32_t ROUND_UP(uint32_t x, uint32_t block_number)
{
    if (block_number > 0) {
        return (x + block_number - 1) / block_number * block_number;
    }
    return 0;
}

template <typename T>
__aicore__ inline void DataCopyCustom(
    GlobalTensor<T>& dstTensor, LocalTensor<T>& srcTensor, const uint32_t dstOffset, const uint32_t srcOffset,
    const uint32_t blockCount)
{
    uint32_t alignLen_ = ALIGN_32;
    if constexpr (!is_same<T, float>::value) {
        alignLen_ = ALIGN_16;
    }
    uint32_t calcLenAlign32 = (blockCount / alignLen_) * alignLen_;
    if (calcLenAlign32 > 0) {
        DataCopy(dstTensor[dstOffset], srcTensor[srcOffset], calcLenAlign32);
    }
    uint32_t calcLenTail32 = blockCount % alignLen_;

    if (calcLenTail32 > 0) {
        SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
        for (uint32_t i = 0; i < calcLenTail32; i++) {
            dstTensor.SetValue(dstOffset + calcLenAlign32 + i, srcTensor.GetValue(srcOffset + calcLenAlign32 + i));
        }
        DataCacheCleanAndInvalid<T, CacheLine::ENTIRE_DATA_CACHE>(dstTensor);
        PipeBarrier<PIPE_ALL>();
        SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
    }
}

template <typename T>
__aicore__ inline void InitGmZero(
    GlobalTensor<T>& outGm, TBuf<TPosition::VECCALC>& TmpZeroTBuf, const uint32_t zeroLen, const uint32_t outOffset)
{
    uint32_t alignLen_ = BLOCK_SIZE / sizeof(T);
    LocalTensor<T> temp_zero_tensor = TmpZeroTBuf.Get<T>();

    Duplicate(temp_zero_tensor, (T)0.0, zeroLen);
    PipeBarrier<PIPE_ALL>();
    SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);

    DataCopy(outGm[outOffset], temp_zero_tensor, ROUND_UP(zeroLen, alignLen_));
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);

    PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void dCopyIn(int64_t colIndex, int64_t colSize, int64_t rowSize, int64_t colAlignV_, int64_t chunkSize, deterministic_struct& deterministicStruct) {
    uint8_t rightPad = 0;
    bool isPad = false;
    int64_t colSizeMod = colSize % FLOAT_ALIGN;
    // 尾核补齐对齐
    if (colSizeMod != 0) {
        rightPad = FLOAT_ALIGN - colSizeMod;
        isPad = true;
    } 
#if __CCE_AICORE__ == 220
    if ASCEND_IS_AIV {
#endif
    TEventID eventID = GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>();
    SetFlag<HardEvent::V_MTE2>(eventID);
    WaitFlag<HardEvent::V_MTE2>(eventID);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE2>(eventID);
    int64_t offset = colIndex * COL_TEMPLATE;
#if __CCE_AICORE__ == 220   
    DataCopyExtParams copyInParams{
        static_cast<uint16_t>(rowSize),
        (uint32_t)(colSize * sizeof(float)),
        (uint32_t)((chunkSize - colSize) * sizeof(float)),
        static_cast<uint32_t>((COL_TEMPLATE - colSize) / FLOAT_ALIGN),
        0
    };
    DataCopyPadExtParams<float> padParams{
        false,
        0,
        static_cast<uint8_t>(rightPad),
        0
    };
    DataCopyPad(deterministicStruct.buffer1_, deterministicStruct.workspaceGmOri_[offset], copyInParams, padParams);
#else
     DataCopyParams intriParams; 
     intriParams.blockCount = rowSize; 
     intriParams.blockLen   = (colSize + FLOAT_ALIGN - 1) / FLOAT_ALIGN; 
     intriParams.srcStride  = (chunkSize - colSize) / FLOAT_ALIGN; 
     intriParams.dstStride  = (COL_TEMPLATE - (colSize + rightPad)) / FLOAT_ALIGN; 
     DataCopy(deterministicStruct.buffer1_, deterministicStruct.workspaceGmOri_[offset], intriParams);
#endif
#if __CCE_AICORE__ == 220
    }
#endif
}

__aicore__ inline void dCompute(int64_t colIndex, int64_t rowIndex, int64_t colSize, int64_t rowSize, deterministic_struct& deterministicStruct) {
    TEventID eventId = GetTPipePtr()->AllocEventID<HardEvent::MTE2_V>();
    SetFlag<HardEvent::MTE2_V>(eventId);
    WaitFlag<HardEvent::MTE2_V>(eventId);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE2_V>(eventId);
    TEventID eventId1 = GetTPipePtr()->AllocEventID<HardEvent::MTE3_V>();
    SetFlag<HardEvent::MTE3_V>(eventId1);
    WaitFlag<HardEvent::MTE3_V>(eventId1);
    GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_V>(eventId1);
    Duplicate(deterministicStruct.buffer2_, static_cast<float>(0.0), COL_TEMPLATE);
    PipeBarrier<PIPE_V>();
    uint64_t maskVal = colSize;
    uint8_t repeatTimes = MAX_REPEAT_TIMES;
    BinaryRepeatParams binaryRepeatParams;
    binaryRepeatParams.dstBlkStride = 1;
    binaryRepeatParams.src0BlkStride = 1;
    binaryRepeatParams.src1BlkStride = 1;
    binaryRepeatParams.dstRepStride = 0;
    binaryRepeatParams.src0RepStride = B32_REPEAT_STRIDE;
    binaryRepeatParams.src1RepStride = 0;
    int64_t rowRepeatTimes = (rowSize + MAX_REPEAT_TIMES - 1) / MAX_REPEAT_TIMES;
    for (int64_t i = 0; i < rowRepeatTimes; i++) {
        if (i == rowRepeatTimes - 1) {
            repeatTimes = rowSize - (rowRepeatTimes - 1) * MAX_REPEAT_TIMES;
        }
        Add(deterministicStruct.buffer2_,deterministicStruct.buffer1_[i * MAX_REPEAT_TIMES],deterministicStruct.buffer2_,maskVal,repeatTimes,binaryRepeatParams);
        PipeBarrier<PIPE_V>();
    }
}

__aicore__ inline void dCopyOut(int64_t colIndex, int64_t colSize, int64_t outOffset, deterministic_struct& deterministicStruct)
{
    DataCopyParams intriParams;
    intriParams.blockCount = 1;
    intriParams.blockLen = colSize * sizeof(float);
    intriParams.srcStride = 0;
    intriParams.dstStride = 0;
    TEventID eventID = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
    SetFlag<HardEvent::V_MTE3>(eventID);
    WaitFlag<HardEvent::V_MTE3>(eventID);
    GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(eventID);
    int64_t offset = colIndex * COL_TEMPLATE;
    DataCopyPad(deterministicStruct.dgammaGm_[outOffset + offset], deterministicStruct.buffer2_, intriParams);
}

__aicore__ inline void FinalProcessDeterministic(int64_t tcolAlignV, int64_t tblockNum, int64_t tcol, int64_t outOffset, int64_t chunkSize, deterministic_struct& deterministicStruct) {
    int64_t colAlignV_ = tcolAlignV;
    int64_t row_ = tblockNum;
    int64_t col_ = tcol;
    uint32_t coreNum = tblockNum;
    int64_t colcycleCount = (colAlignV_ + COL_TEMPLATE - 1) / COL_TEMPLATE;
    int64_t colcyclePerBlockCount = (colcycleCount + coreNum - 1) / coreNum;
    int64_t colSize = COL_TEMPLATE;
    int64_t taskId = 0;
    for (int64_t blocktaskId = 0; blocktaskId < colcyclePerBlockCount; blocktaskId++) {
        taskId = blocktaskId * coreNum + GetBlockIdx();
        if (taskId < colcycleCount) {
            if (taskId == colcycleCount - 1) {
                colSize = col_ - COL_TEMPLATE * taskId;
            }
            dCopyIn(taskId, colSize, row_, colAlignV_, chunkSize, deterministicStruct);
            dCompute(taskId, 0, colSize, row_, deterministicStruct);
            dCopyOut(taskId, colSize, outOffset, deterministicStruct); 
        } else {
            break;
        }
    }
}

// Get chunk range (start position and length) for workspace optimization
__aicore__ inline void GetChunkRange(uint32_t chunkId, uint32_t chunkSize, uint32_t chunkNum, 
    uint32_t chunkTail, uint32_t& chunkStart, uint32_t& currentChunkLen)
{
    chunkStart = chunkId * chunkSize;
    currentChunkLen = (chunkId == chunkNum - 1) ? chunkTail : chunkSize;
}

// Helper function for event synchronization
__aicore__ inline void SyncEventV_S()
{
    event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventVS);
    WaitFlag<HardEvent::V_S>(eventVS);
}

__aicore__ inline void SyncEventS_V()
{
    event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventSV);
    WaitFlag<HardEvent::S_V>(eventSV);
}

__aicore__ inline void SyncEventMTE2_V()
{
    event_t eventMte2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventMte2V);
    WaitFlag<HardEvent::MTE2_V>(eventMte2V);
}

__aicore__ inline void SyncEventMTE3_V()
{
    event_t eventMte3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eventMte3V);
    WaitFlag<HardEvent::MTE3_V>(eventMte3V);
}

__aicore__ inline void SyncEventS_MTE3()
{
    SetFlag<HardEvent::S_MTE3>(EVENT_ID0);
    WaitFlag<HardEvent::S_MTE3>(EVENT_ID0);
}

__aicore__ inline void SyncEventMTE3_S()
{
    SetFlag<HardEvent::MTE3_S>(EVENT_ID0);
    WaitFlag<HardEvent::MTE3_S>(EVENT_ID0);
}

// Helper function to pad tensor for small length
template<typename T>
__aicore__ inline void PadTensorForSmallLength(LocalTensor<T>& tensor, uint32_t calcLen, uint32_t alignLen)
{
    if (calcLen < alignLen) {
        for (uint32_t i = 0; i < alignLen; i++) {
            if (i >= calcLen) {
                tensor.SetValue(i, (T)0.0f);
            }
        }
    }
}

// Helper function to handle tail data copy for dgamma
__aicore__ inline void HandleDgammaTailCopy(
    LocalTensor<float>& dgamma_out, uint32_t calcLen, uint32_t calcLenAlign, uint32_t calcLenTail, uint32_t alignLen)
{
    if (calcLenTail > 0) {
        for (uint32_t i = 0; i < alignLen; i++) {
            if (i < (alignLen - calcLenTail)) {
                dgamma_out.SetValue(i, 0.0f);
            } else {
                uint32_t tailOffset = calcLenAlign + i - (alignLen - calcLenTail);
                dgamma_out.SetValue(i, dgamma_out.GetValue(tailOffset));
            }
        }
    }
}

// Helper function to copy tail data for dx output
template<typename T>
__aicore__ inline void CopyTailDataToGm(
    GlobalTensor<T>& gmTensor, LocalTensor<T>& localTensor, 
    uint32_t gmOffset, uint32_t calcLenAlign, uint32_t calcLenTail)
{
    if (calcLenTail > 0) {
        SyncEventMTE3_S();
        for (uint32_t i = 0; i < calcLenTail; i++) {
            gmTensor.SetValue(gmOffset + calcLenAlign + i, localTensor.GetValue(calcLenAlign + i));
        }
        DataCacheCleanAndInvalid<T, CacheLine::ENTIRE_DATA_CACHE>(gmTensor);
        PipeBarrier<PIPE_ALL>();
        SyncEventS_MTE3();
    }
}

// Helper function to process loop with main and tail
template<typename Func>
__aicore__ inline void ProcessLoopWithTail(uint32_t loopMain, uint32_t tail, Func func)
{
    for (uint32_t i = 0; i < loopMain; i++) {
        func(i, false);
    }
    if (tail > 0) {
        func(loopMain, true);
    }
}

// Helper function to cast and prepare dx output
template<typename T>
__aicore__ inline void CastAndPrepareDxOutput(
    LocalTensor<float>& dxLocal, LocalTensor<T>& dxLocalB16, uint32_t calcLen)
{
    if constexpr (is_same<T, half>::value) {
        dxLocalB16 = dxLocal.ReinterpretCast<T>();
        Cast(dxLocalB16, dxLocal, RoundMode::CAST_NONE, calcLen);
        PipeBarrier<PIPE_V>();
    }
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 220)
    else if constexpr (is_same<T, bfloat16_t>::value) {
        dxLocalB16 = dxLocal.ReinterpretCast<T>();
        Cast(dxLocalB16, dxLocal, RoundMode::CAST_RINT, calcLen);
        PipeBarrier<PIPE_V>();
    }
#endif
}

// Helper function to initialize workspace buffers
__aicore__ inline void InitWorkspaceBuffers(
    GlobalTensor<float>& workspaceGm, GlobalTensor<float>& workspaceBlockFactorGm, 
    GlobalTensor<float>& workspaceGmOri, GM_ADDR usrWorkspace,
    uint32_t blockIdx, uint32_t blockDim, uint32_t workspaceColSize, 
    uint32_t blockFactor, bool needChunk)
{
    workspaceGm.SetGlobalBuffer((__gm__ float*)usrWorkspace + blockIdx * workspaceColSize);
    InitOutput<float>(workspaceGm, workspaceColSize, 0);
    if(needChunk) {
        workspaceBlockFactorGm.SetGlobalBuffer(
            (__gm__ float*)usrWorkspace + blockDim * workspaceColSize + blockIdx * blockFactor, blockFactor);
        InitOutput<float>(workspaceBlockFactorGm, blockFactor, 0);
    }
    workspaceGmOri.SetGlobalBuffer((__gm__ float*)usrWorkspace);
}

// Helper function to copy data with conditional offset
template<typename T>
__aicore__ inline void DataCopyWithOffset(
    LocalTensor<T>& dst, GlobalTensor<T>& src, uint32_t offset, 
    uint32_t calcLen, uint32_t ubFactor, uint32_t alignLen)
{
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 220)
    DataCopyExtParams dataCopyParams{1, (uint32_t)(calcLen * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
    if constexpr (!is_same<T, float>::value) {
        DataCopyPad(dst[ubFactor], src[offset], dataCopyParams, padParams);
    } else {
        DataCopyPad(dst, src[offset], dataCopyParams, padParams);
    }
#else
    uint32_t calcLen_align = ROUND_UP(calcLen, alignLen);
    if constexpr (!is_same<T, float>::value) {
        DataCopy(dst[ubFactor], src[offset], calcLen_align);
    } else {
        DataCopy(dst, src[offset], calcLen_align);
    }
#endif
}

// Initialize dgamma output to zero
__aicore__ inline void InitDgammaOutCommon(
    GlobalTensor<float>& outGm, TBuf<TPosition::VECCALC>& outZeroTmpBuf,
    uint32_t loopMainCol, uint32_t ubFactor, uint32_t tailCol)
{
    for (uint32_t iOuter = 0; iOuter < loopMainCol; iOuter++) {
        InitGmZero<float>(outGm, outZeroTmpBuf, ubFactor, iOuter * ubFactor);
    }
    if (tailCol > 0) {
        InitGmZero<float>(outGm, outZeroTmpBuf, tailCol, loopMainCol * ubFactor);
    }
}

// Copy mean output for chunk processing
__aicore__ inline void CopyMeanOutChunkCommon(
    TBuf<TPosition::VECCALC>& meanTmpBuf, GlobalTensor<float>& workspaceBlockFactorGm,
    uint32_t iOuter, uint32_t calcRow, uint32_t rowFactor)
{
    LocalTensor<float> meanTmpLocal = meanTmpBuf.Get<float>();
    SetAtomicAdd<float>();
    uint32_t gmOffset = iOuter * rowFactor;
    DataCopyExtParams dataCopyParams{1, (uint32_t)(calcRow * sizeof(float)), 0, 0, 0};
    DataCopyPad(workspaceBlockFactorGm[gmOffset], meanTmpLocal, dataCopyParams);
    SetAtomicNone();
    SyncEventMTE3_V();
}

// Copy dgamma output with atomic add
__aicore__ inline void CopyDgammaOutCommon(
    TQue<QuePosition::VECOUT, BUFFER_NUM>& outQueDgamma, GlobalTensor<float>& outGm,
    uint32_t dIdx, uint32_t calcLen, uint32_t ubFactor)
{
    LocalTensor<float> dgamma_out = outQueDgamma.DeQue<float>();
    uint32_t gmOffset = dIdx * ubFactor;
    SetAtomicAdd<float>();
#if defined(__CCE_AICORE__) && (__CCE_AICORE__ == 220)
    DataCopyExtParams dataCopyParams{1, (uint32_t)(calcLen * sizeof(float)), 0, 0, 0};
    DataCopyPad(outGm[gmOffset], dgamma_out, dataCopyParams);
#else
    if (calcLen < ALIGN_32) {
        PadTensorForSmallLength(dgamma_out, calcLen, ALIGN_32);
        DataCopy(outGm[gmOffset], dgamma_out, ALIGN_32);
        SyncEventS_MTE3();
    } else {
        uint32_t calcLenAlign = (calcLen / ALIGN_32) * ALIGN_32;
        uint32_t calcLenTail = calcLen - calcLenAlign;
        DataCopy(outGm[gmOffset], dgamma_out, calcLenAlign);
        if (calcLenTail > 0) {
            SyncEventMTE3_S();
            HandleDgammaTailCopy(dgamma_out, calcLen, calcLenAlign, calcLenTail, ALIGN_32);
            DataCopy(outGm[gmOffset + calcLen - ALIGN_32], dgamma_out, ALIGN_32);
            SyncEventS_MTE3();
        }
    }
#endif
    SetAtomicNone();
    outQueDgamma.FreeTensor(dgamma_out);
}

#endif // RMS_NORM_GRAD_BASE_H