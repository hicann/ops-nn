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
 * \file rotate_quant.h
 * \brief Unified kernel implementation for RotateQuant operator (AIC + AIV 1C2V mode)
 */
#ifndef ROTATE_QUANT_H
#define ROTATE_QUANT_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "rotate_quant_tiling_data.h"

namespace RotateQuantOpt {
using namespace AscendC;
using namespace matmul;

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t MAX_VALUE_NUM = 8;
constexpr uint32_t SYNC_MODE = 0x2;
constexpr uint32_t SYNC_C2V_0 = 0x1;
constexpr uint32_t SYNC_V2C_0 = 0x3;
constexpr float DYNAMIC_QUANT_INT8_SYM_SCALE = 127.0;
constexpr float DYNAMIC_QUANT_INT4_SYM_SCALE = 7.0;
constexpr int32_t DIMENSION_2 = 2;
constexpr int32_t CONST_SCALE_ELEMENTS = 8;
constexpr int32_t SCALE_BUFFER_SIZE = 192;
constexpr int32_t ALIGN_FACTOR_32 = 32;
constexpr int32_t UB_RESERVED_SIZE = 1024;
constexpr int32_t STRIDE_8 = 8;
constexpr int32_t BLOCK_SIZE_64 = 64;
constexpr int32_t SPEED_N = 15000;
constexpr int32_t SPEED_STEP_LOOP = 12;
constexpr int32_t MAX_REG_COUNT = 15;

struct InitParams {
    GM_ADDR x;
    GM_ADDR rot;
    GM_ADDR y;
    GM_ADDR scale;
    GM_ADDR workspace;
    TPipe* pipe;
    const RotateQuantTilingData* tilingData;
};

template <typename xDtype, typename yDtype, typename MT>
class RotateQuant {
public:
    __aicore__ inline RotateQuant(MT& mm) : mm_(mm)
    {}

    __aicore__ inline void Init(const InitParams& params);
    __aicore__ inline void Process();

private:
    __aicore__ inline void AicProcess();
    __aicore__ inline void CubeProcessRows(int32_t mStart, int32_t mSize);
    __aicore__ inline void AivProcess();
    __aicore__ inline void AllocateLocalBuffers();
    __aicore__ inline void ComputeVector(
        uint32_t ubRows, uint32_t ubNums, uint32_t srcShape1[DIMENSION_2], uint32_t dstShape1[DIMENSION_2]);
    __aicore__ inline void CopyOutVector(
        uint32_t idx, uint32_t ubRows, uint32_t ubNums, event_t event_mte3_v, uint64_t offsetLoop,
        uint64_t offsetScale);
    __aicore__ inline void ReduceMaxInplace(const LocalTensor<float>& src_local, uint32_t count);
    __aicore__ inline void ParseTilingData(const RotateQuantTilingData* __restrict tilingData);
    __aicore__ inline void ComputeReduceMax(uint32_t ubRows);
    __aicore__ inline void CopyInVector(
        GlobalTensor<xDtype> srcGm, DataCopyExtParams& copyInParams, DataCopyPadExtParams<xDtype>& padParams,
        uint32_t ubRows, uint32_t ubNums);

    GlobalTensor<xDtype> xGm_;
    GlobalTensor<xDtype> rotGm_;
    GlobalTensor<yDtype> yGm_;
    GlobalTensor<float> scaleGm_;
    GlobalTensor<xDtype> workspaceGm_;
    MT& mm_;
    TBuf<QuePosition::VECCALC> bufQueue;
    LocalTensor<float> constScale, constInvScale;
    LocalTensor<float> castTmp, absTmp, scaleTmp, quantScaleTmp;
    LocalTensor<yDtype> outLocal;
    LocalTensor<xDtype> inLocal;
    LocalTensor<float> scaleLocal;
    TPipe* pipe_;
    RotateQuantTilingData tilingData_;
    int32_t M_;
    int32_t N_;
    int32_t K_;
    int32_t numBlocks_;
    int32_t stepLoop_;
    int32_t aicCoreNum_;
    int32_t aivCoreNum_;
    uint32_t loopM_;
    uint32_t rowPerHeadCore_;
    uint32_t rowPerCubeTailCore_;
    uint32_t rowPerCubeHeadCore_;
    uint32_t rowPerVectorTailCore_;
    uint32_t rowPerVectorLastCore_;
    uint32_t ubSize_;
    uint32_t nN_;
    uint32_t singleLoopRows = 0;
    uint32_t singleLoopEle = 0;
    uint32_t startRow = 0;
    uint32_t totalRows = 0;
    uint32_t coreIdx_;
    uint32_t lastUbRows = 0;
};

template <typename xDtype, typename yDtype, typename MT>
__aicore__ inline void RotateQuant<xDtype, yDtype, MT>::ParseTilingData(
    const RotateQuantTilingData* __restrict tilingData)
{
    tilingData_ = *tilingData;
    M_ = tilingData_.M;
    N_ = tilingData_.N;
    K_ = tilingData_.K;
    aicCoreNum_ = tilingData_.aicCoreNum;
    aivCoreNum_ = tilingData_.aivCoreNum;
    numBlocks_ = tilingData_.numBlocks;
    stepLoop_ = tilingData_.stepLoop;
    rowPerHeadCore_ = tilingData_.rowPerHeadCore;
    rowPerCubeHeadCore_ = tilingData_.rowPerCubeHeadCore;
    rowPerCubeTailCore_ = tilingData_.rowPerCubeTailCore;
    rowPerVectorTailCore_ = tilingData_.rowPerVectorTailCore;
    rowPerVectorLastCore_ = tilingData_.rowPerVectorLastCore;
    lastUbRows = tilingData_.lastUbRows;
    singleLoopRows = tilingData->multiRowNumHeadCore;
    ubSize_ = tilingData_.ubSize;
}

template <typename xDtype, typename yDtype, typename MT>
__aicore__ inline void RotateQuant<xDtype, yDtype, MT>::AllocateLocalBuffers()
{
    pipe_->InitBuffer(bufQueue, ubSize_ - UB_RESERVED_SIZE);
    LocalTensor<uint8_t> tmp = bufQueue.Get<uint8_t>();
    uint32_t offsetBytes = 0;
    uint32_t singleLoopEleAlign = CeilAlign(singleLoopEle, ALIGN_FACTOR_32);
    uint32_t rowLenAlign = CeilAlign(nN_, ALIGN_FACTOR_32);
    // inLocal用于将matmul计算的结果拷贝到local，大小 （N * 最小循环行数 32字节对齐） * sizeof(xDtype)
    inLocal = tmp.ReinterpretCast<xDtype>();
    offsetBytes += singleLoopEleAlign * sizeof(xDtype);
    // castTmp用于暂存vector输入转float32的数据，大小 （N * 最小循环行数 32字节对齐） * sizeof(float)
    castTmp = tmp[offsetBytes].ReinterpretCast<float>();
    offsetBytes += singleLoopEleAlign * sizeof(float);
    // absTmp用于暂存vector输入转float32的绝对值数据，大小 （N * 最小循环行数 32字节对齐） * sizeof(float)
    absTmp = tmp[offsetBytes].ReinterpretCast<float>();
    offsetBytes += singleLoopEleAlign * sizeof(float);
    // outLocal用于暂存y输出数据，大小 （N * 最小循环行数 32字节对齐） * sizeof(yDtype)
    outLocal = tmp[offsetBytes].ReinterpretCast<yDtype>();
    offsetBytes += singleLoopEleAlign * sizeof(yDtype);
    // scaleLocal用于暂存scale输出数据，大小 192 * sizeof(float)
    scaleLocal = tmp[offsetBytes].ReinterpretCast<float>();
    offsetBytes += SCALE_BUFFER_SIZE * sizeof(float);
    // scaleTmp用于暂存多行取最大值得到的被整除的scale，大小 192 * sizeof(float)
    scaleTmp = tmp[offsetBytes].ReinterpretCast<float>();
    offsetBytes += SCALE_BUFFER_SIZE * sizeof(float);
    // quantScaleTmp用于暂存多行取最大值得到的整除后的scale，大小 192 * sizeof(float)
    quantScaleTmp = tmp[offsetBytes].ReinterpretCast<float>();
    offsetBytes += SCALE_BUFFER_SIZE * sizeof(float);
    // constScale用于暂存constscale，大小 8 * sizeof(float)
    constScale = tmp[offsetBytes].ReinterpretCast<float>();
    offsetBytes += CONST_SCALE_ELEMENTS * sizeof(float);
    constInvScale = tmp[offsetBytes].ReinterpretCast<float>();
}

template <typename xDtype, typename yDtype, typename MT>
__aicore__ inline void RotateQuant<xDtype, yDtype, MT>::Init(const InitParams& params)
{
    ParseTilingData(params.tilingData);
    pipe_ = params.pipe;
    coreIdx_ = GetBlockIdx();
    nN_ = static_cast<uint32_t>(N_);
    if ASCEND_IS_AIC {
        startRow = coreIdx_ * rowPerCubeHeadCore_;
        if (coreIdx_ < aicCoreNum_ - 1) {
            totalRows = rowPerCubeHeadCore_;
        } else {
            totalRows = rowPerCubeTailCore_;
        }
        xGm_.SetGlobalBuffer((__gm__ xDtype*)(params.x) + static_cast<int64_t>(startRow) * nN_);
        rotGm_.SetGlobalBuffer((__gm__ xDtype*)(params.rot));
        workspaceGm_.SetGlobalBuffer((__gm__ xDtype*)(params.workspace) + static_cast<int64_t>(startRow) * nN_);
        singleLoopRows = singleLoopRows * 2;
        singleLoopEle = singleLoopRows * nN_;
        loopM_ = CeilDiv(totalRows, singleLoopRows);
    }

    if ASCEND_IS_AIV {
        startRow = (coreIdx_ / 2) * rowPerHeadCore_ * 2;
        if (coreIdx_ / 2 < aicCoreNum_ - 1) {
            totalRows = rowPerHeadCore_;
        } else if (coreIdx_ == aivCoreNum_ - 1) {
            totalRows = rowPerVectorLastCore_;
        } else {
            totalRows = rowPerVectorTailCore_;
        }
        singleLoopEle = singleLoopRows * nN_;
        loopM_ = CeilDiv(totalRows, singleLoopRows);
        workspaceGm_.SetGlobalBuffer((__gm__ xDtype*)(params.workspace) + static_cast<int64_t>(startRow) * nN_);
        scaleGm_.SetGlobalBuffer((__gm__ float*)(params.scale) + startRow);
        AllocateLocalBuffers();
        if constexpr (IsSameType<yDtype, int4b_t>::value) {
            yGm_.SetGlobalBuffer((__gm__ yDtype*)(params.y) + (static_cast<int64_t>(startRow) * nN_ >> 1));
            Duplicate<float>(constScale, DYNAMIC_QUANT_INT4_SYM_SCALE, CONST_SCALE_ELEMENTS);
            Duplicate<float>(constInvScale, float(1) / DYNAMIC_QUANT_INT4_SYM_SCALE, CONST_SCALE_ELEMENTS);
        } else {
            yGm_.SetGlobalBuffer((__gm__ yDtype*)(params.y) + static_cast<int64_t>(startRow) * nN_);
            Duplicate<float>(constScale, DYNAMIC_QUANT_INT8_SYM_SCALE, CONST_SCALE_ELEMENTS);
            Duplicate<float>(constInvScale, float(1) / DYNAMIC_QUANT_INT8_SYM_SCALE, CONST_SCALE_ELEMENTS);
        }
    }
}

template <typename xDtype, typename yDtype, typename MT>
__aicore__ inline void RotateQuant<xDtype, yDtype, MT>::CubeProcessRows(int32_t mStart, int32_t mSize)
{
    if (mSize <= 0) {
        return;
    }
    uint64_t cubeOffset = static_cast<uint64_t>(mStart) * static_cast<uint64_t>(N_);
    uint64_t totalVirtualRows = mSize * numBlocks_;
    mm_.SetOrgShape(totalVirtualRows, K_, K_);
    mm_.SetSingleShape(totalVirtualRows, K_, K_);
    mm_.SetTensorA(xGm_[cubeOffset]);
    mm_.SetTensorB(rotGm_[0]);
    mm_.IterateAll(workspaceGm_[cubeOffset], 0);
    mm_.End();
}

template <typename xDtype, typename yDtype, typename MT>
__aicore__ inline void RotateQuant<xDtype, yDtype, MT>::AicProcess()
{
    if (coreIdx_ >= aicCoreNum_) {
        return;
    }
    for (int32_t i = 0; i < loopM_; i += stepLoop_) {
        if ((i % (MAX_REG_COUNT * stepLoop_) == stepLoop_) && (i != stepLoop_)) {
            CrossCoreWaitFlag(SYNC_V2C_0);
        }
        int32_t blockStart = i * singleLoopRows;
        int32_t remainingRows = totalRows - blockStart;
        if (remainingRows < 0) {
            return;
        }
        int32_t cubeRows = singleLoopRows * stepLoop_;
        int32_t currentBlockSize = (cubeRows < remainingRows) ? cubeRows : remainingRows;
        if (currentBlockSize > 0) {
            CubeProcessRows(blockStart, currentBlockSize);
        }
        CrossCoreSetFlag<SYNC_MODE, PIPE_FIX>(SYNC_C2V_0);
    }
}

template <typename xDtype, typename yDtype, typename MT>
__aicore__ inline void RotateQuant<xDtype, yDtype, MT>::ReduceMaxInplace(
    const LocalTensor<float>& src_local, uint32_t count)
{
    uint64_t repsFp32 = count >> 6;
    uint64_t offsetsFp32 = repsFp32 << 6;
    uint64_t remsFp32 = count & 0x3f;

    if (likely(repsFp32 > 1)) {
        if (repsFp32 - 1 > 255) {
            Max(src_local, src_local[64], src_local, 64, 255, {1, 1, 1, 0, 8, 0});
            PipeBarrier<PIPE_V>();
            Max(src_local, src_local[64 * 255], src_local, 64, repsFp32 - 255 - 1, {1, 1, 1, 0, 8, 0});
        } else {
            Max(src_local, src_local[64], src_local, 64, repsFp32 - 1, {1, 1, 1, 0, 8, 0});
        }
        PipeBarrier<PIPE_V>();
    }
    if (unlikely(remsFp32 > 0) && unlikely(offsetsFp32 > 0)) {
        Max(src_local, src_local[offsetsFp32], src_local, remsFp32, 1, {1, 1, 1, 0, 8, 0});
        PipeBarrier<PIPE_V>();
    }
    uint32_t mask = repsFp32 > 0 ? 64 : count;
    WholeReduceMax(src_local, src_local, mask, 1, 8, 1, 8);
}

template <typename xDtype, typename yDtype, typename MT>
__aicore__ inline void RotateQuant<xDtype, yDtype, MT>::CopyInVector(
    GlobalTensor<xDtype> srcGm, DataCopyExtParams& copyInParams, DataCopyPadExtParams<xDtype>& padParams,
    uint32_t ubRows, uint32_t ubNums)
{
    DataCopyPad(inLocal, srcGm, copyInParams, padParams);
    event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventId);
    WaitFlag<HardEvent::MTE2_V>(eventId);
    Cast(castTmp, inLocal, RoundMode::CAST_NONE, ubNums);
    PipeBarrier<PIPE_V>();
    event_t eventId1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(eventId1);
    WaitFlag<HardEvent::V_MTE2>(eventId1);
}

template <typename xDtype, typename yDtype, typename MT>
__aicore__ inline void RotateQuant<xDtype, yDtype, MT>::ComputeVector(
    uint32_t ubRows, uint32_t ubNums, uint32_t srcShape1[DIMENSION_2], uint32_t dstShape1[DIMENSION_2])
{
    Abs(absTmp, castTmp, ubNums);
    PipeBarrier<PIPE_V>();
    ComputeReduceMax(ubRows);
    PipeBarrier<PIPE_V>();
    Div(quantScaleTmp, constScale, scaleTmp, BLOCK_SIZE_64, CeilDiv(ubRows, BLOCK_SIZE_64),
        {1, 0, 1, STRIDE_8, 0, STRIDE_8});
    PipeBarrier<PIPE_V>();
    BroadCast<float, DIMENSION_2, 1>(absTmp, quantScaleTmp, dstShape1, srcShape1);
    PipeBarrier<PIPE_V>();
    Mul(absTmp, castTmp, absTmp, ubNums);
    PipeBarrier<PIPE_V>();
    Cast(absTmp.ReinterpretCast<int16_t>(), absTmp, RoundMode::CAST_RINT, ubNums);
    PipeBarrier<PIPE_V>();
    Cast(absTmp.ReinterpretCast<half>(), absTmp.ReinterpretCast<int16_t>(), RoundMode::CAST_ROUND, ubNums);
    PipeBarrier<PIPE_V>();
}

template <typename xDtype, typename yDtype, typename MT>
__aicore__ inline void RotateQuant<xDtype, yDtype, MT>::CopyOutVector(
    uint32_t idx, uint32_t ubRows, uint32_t ubNums, event_t event_mte3_v, uint64_t offsetLoop, uint64_t offsetScale)
{
    if (idx != 0) {
        SetFlag<HardEvent::MTE3_V>(event_mte3_v);
        WaitFlag<HardEvent::MTE3_V>(event_mte3_v);
    }
    Mul(scaleLocal, scaleTmp, constInvScale, BLOCK_SIZE_64, CeilDiv(ubRows, BLOCK_SIZE_64),
        {1, 1, 0, STRIDE_8, STRIDE_8, 0});
    Cast(outLocal, absTmp.ReinterpretCast<half>(), RoundMode::CAST_TRUNC, ubNums);
    PipeBarrier<PIPE_V>();
    event_t event_v_mte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(event_v_mte3);
    WaitFlag<HardEvent::V_MTE3>(event_v_mte3);
    DataCopyParams outScaleParams{1, static_cast<uint16_t>(ubRows * sizeof(float)), 0, 0};
    DataCopyPad(scaleGm_[offsetScale], scaleLocal, outScaleParams);
    DataCopyParams copyOutParams{1, static_cast<uint16_t>(ubNums * sizeof(yDtype)), 0, 0};
    if constexpr (IsSameType<yDtype, int4b_t>::value) {
        copyOutParams.blockLen = copyOutParams.blockLen >> 1;
    }
    DataCopyPad(yGm_[offsetLoop], outLocal, copyOutParams);
}

template <typename xDtype, typename yDtype, typename MT>
__aicore__ inline void RotateQuant<xDtype, yDtype, MT>::AivProcess()
{
    if (coreIdx_ >= aivCoreNum_) {
        return;
    }
    uint32_t srcShape1[DIMENSION_2] = {singleLoopRows, 1};
    uint32_t dstShape1[DIMENSION_2] = {singleLoopRows, nN_};
    DataCopyExtParams copyInParams{1, 1, 0, 0, 0};
    DataCopyPadExtParams<xDtype> padParams{false, 0, 0, 0};
    event_t event_mte3_v = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    uint32_t ubRows = singleLoopRows;
    uint32_t ubNums = singleLoopEle;
    copyInParams.blockLen = singleLoopEle * sizeof(xDtype);
    uint32_t vecCoreIdx = coreIdx_ % 2;
    for (uint32_t idx = 0; idx < loopM_; idx += stepLoop_) {
        CrossCoreWaitFlag(SYNC_C2V_0);
        int32_t remainingRows = loopM_ - idx;
        int32_t currentLoop = (stepLoop_ < remainingRows) ? stepLoop_ : remainingRows;
        for (uint32_t j = 0; j < currentLoop; j++) {
            int32_t currIdx = idx + j;
            uint64_t offsetLoop = currIdx * singleLoopEle * 2 + vecCoreIdx * ubNums;
            uint64_t offsetScale = currIdx * 2 * singleLoopRows + vecCoreIdx * ubRows;
            if (currIdx == loopM_ - 1) {
                ubRows = totalRows - currIdx * singleLoopRows;
                ubNums = ubRows * nN_;
                srcShape1[0] = ubRows;
                dstShape1[0] = ubRows;
                copyInParams.blockLen = ubNums * sizeof(xDtype);
                if (coreIdx_ == aivCoreNum_ - 1) {
                    offsetLoop = currIdx * singleLoopEle * 2 + lastUbRows * nN_;
                    offsetScale = currIdx * 2 * singleLoopRows + lastUbRows;
                }
            }
            PipeBarrier<PIPE_V>();
            CopyInVector(workspaceGm_[offsetLoop], copyInParams, padParams, ubRows, ubNums);
            ComputeVector(ubRows, ubNums, srcShape1, dstShape1);
            CopyOutVector(currIdx, ubRows, ubNums, event_mte3_v, offsetLoop, offsetScale);
        }
        if ((idx % (MAX_REG_COUNT * stepLoop_) == 0) && (idx != 0)) {
            CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(SYNC_V2C_0);
        }
    }
}

template <typename xDtype, typename yDtype, typename MT>
__aicore__ inline void RotateQuant<xDtype, yDtype, MT>::ComputeReduceMax(uint32_t ubRows)
{
    LocalTensor<float> calcTensor;
    BinaryRepeatParams repeatParams{1, 1, 1, 0, STRIDE_8, 0};

    for (uint32_t i = 0; i < ubRows; i++) {
        calcTensor = absTmp[i * nN_];
        uint32_t repeatTimes = nN_ / BLOCK_SIZE_64;
        uint32_t remainElements = nN_ % BLOCK_SIZE_64;

        if (remainElements > 0) {
            Max(calcTensor, calcTensor, calcTensor[repeatTimes * BLOCK_SIZE_64], remainElements, 1, repeatParams);
            PipeBarrier<PIPE_V>();
        }
        if (repeatTimes > 1) {
            Max(calcTensor, calcTensor[BLOCK_SIZE_64], calcTensor, BLOCK_SIZE_64, repeatTimes - 1, repeatParams);
            PipeBarrier<PIPE_V>();
        }
    }
    WholeReduceMax(scaleTmp, absTmp, BLOCK_SIZE_64, ubRows, 1, 1, nN_ / STRIDE_8, ReduceOrder::ORDER_ONLY_VALUE);
}

template <typename xDtype, typename yDtype, typename MT>
__aicore__ inline void RotateQuant<xDtype, yDtype, MT>::Process()
{
    if ASCEND_IS_AIC {
        AicProcess();
    }

    if ASCEND_IS_AIV {
        AivProcess();
    }
}

} // namespace RotateQuantOpt

#endif // ROTATE_QUANT_H
