/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the 'License').
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dynamic_quant_regbase_pertoken_large_multicore.h
 * \brief pertoken large-shape multicore kernel: split tail axis across cores,
 *        cross-core reduction per token
 */
#ifndef DYNAMIC_QUANT_REGBASE_PERTEN_LARGE_MULTICORE_H
#define DYNAMIC_QUANT_REGBASE_PERTEN_LARGE_MULTICORE_H

#include <cmath>
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "dynamic_quant_regbase_base.h"

namespace DynamicQuantPertenLargeMc {
using namespace AscendC;

constexpr uint32_t NUM_ONE = 1;
constexpr uint32_t DOUBLE_BUFFER_NUM = 2;
constexpr uint32_t FIFTEEN = 15;
constexpr uint32_t SIXTEEN = 16;
constexpr uint32_t THIRTY_ONE = 31;
constexpr uint32_t THIRTY_TWO = 32;
constexpr uint32_t SIXTY_THREE = 63;
constexpr uint32_t SIXTY_FOUR = 64;
constexpr uint32_t TWO_FIVE_SIX = 256;
constexpr float FP8_E5M2_MAX_VALUE = 57344.0f;
constexpr float FP8_E4M3FN_MAX_VALUE = 448.0f;
constexpr float HIFLOAT8_MAX_VALUE = 32768.0f;
constexpr float INT8_MAX_VALUE = 127.0f;
constexpr float INT4_MAX_VALUE = 7.0f;
constexpr float FP8_E5M2_MAX_VALUE_NO_SYM = 114688.0f;
constexpr float FP8_E4M3FN_MAX_VALUE_NO_SYM = 896.0f;
constexpr float HIFLOAT8_MAX_VALUE_NO_SYM = 65536.0f;
constexpr float INT8_MAX_VALUE_NO_SYM = 255.0f;
constexpr float INT4_MAX_VALUE_NO_SYM = 15.0f;
constexpr float MIN_FLOAT_VALUE = -INFINITY;
constexpr float MAX_FLOAT_VALUE = INFINITY;

template <typename T, typename yDtype, int64_t hasSmooth, bool isSymmetrical>
class DynamicQuantPertenLargeMulticore {
public:
    using yCopyDtype = std::conditional_t<IsSameType<yDtype, int4b_t>::value, uint8_t, yDtype>;
    __aicore__ inline DynamicQuantPertenLargeMulticore(TPipe* pipe) { pPipe = pipe; }
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR smooth_scales, GM_ADDR y, GM_ADDR scale, GM_ADDR offset,
                                GM_ADDR workSpace, const DynamicQuantTilingDataArch35& tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void SetMaxValue();
    __aicore__ inline void ProcessScale();
    __aicore__ inline void ProcessY();
    __aicore__ inline void ProcessScaleRow();
    __aicore__ inline void ProcessScaleCol();
    __aicore__ inline void ProcessScaleRowLoop(uint32_t i, uint32_t j);
    __aicore__ inline void ProcessYRow(uint32_t i, uint32_t j, __local_mem__ float* scaleAddr,
                                       __local_mem__ float* offsetAddr);
    __aicore__ inline void CopyInByEle(int64_t offset, uint32_t loopIndex, uint32_t elementNum, uint8_t rightPadding);
    __aicore__ inline void CopyInScaleByEle(int64_t offset, uint32_t elementNum);
    __aicore__ inline void ComputeMaxRowScale(uint32_t elementNum);
    __aicore__ inline void ComputeMaxColScale(uint32_t elementNum, __local_mem__ float* maxAddr,
                                              __local_mem__ float* minAddr, __local_mem__ float* scaleAddr,
                                              __local_mem__ float* offsetAddr);
    __aicore__ inline void ComputeMaxRowScaleVF(__local_mem__ T* inLocalAddr, __local_mem__ T* smoothLocalAddr,
                                                __local_mem__ float* scaleLocalAddr, __local_mem__ float* maxLocalAddr,
                                                __local_mem__ float* minLocalAddr, uint32_t elementNum);
    __aicore__ inline void ComputeMaxColScaleVF(__local_mem__ float* scaleLocalAddr,
                                                __local_mem__ float* scaleOutLocalAddr,
                                                __local_mem__ float* maxLocalAddr, __local_mem__ float* maxOutLocalAddr,
                                                __local_mem__ float* minLocalAddr, __local_mem__ float* minOutLocalAddr,
                                                uint32_t elementNum);
    __aicore__ inline void ComputeY(uint32_t elementNum, __local_mem__ float* scaleAddr,
                                    __local_mem__ float* offsetAddr);
    __aicore__ inline void ComputeScaleSymVF(__local_mem__ float* maxLocalAddr, __local_mem__ float* minLocalAddr,
                                             __local_mem__ float* scaleLocalAddr, uint32_t elementNum);
    __aicore__ inline void ComputeOffsetSymVF(__local_mem__ float* maxLocalAddr, __local_mem__ float* scaleLocalAddr,
                                              __local_mem__ float* offsetLocalAddr, uint32_t elementNum);
    __aicore__ inline void ComputeYVF(__local_mem__ T* inLocalAddr, __local_mem__ T* smoothLocalAddr,
                                      __local_mem__ yCopyDtype* outAddr, __local_mem__ float* scaleLocalAddr,
                                      __local_mem__ float* offsetLocalAddr, uint32_t elementNum);
    __aicore__ inline void ParseTilingData(const DynamicQuantTilingDataArch35& tilingData);
    __aicore__ inline void CopyOutY(int64_t offset, uint32_t element);
    __aicore__ inline void CopyUB2Workspace(int64_t size);

private:
    TPipe* pPipe = nullptr;
    DynamicQuantTilingDataArch35 tilingData_;

    TQue<QuePosition::VECIN, NUM_ONE> inQueue;
    TQue<QuePosition::VECIN, NUM_ONE> smoothQueue;
    TQue<QuePosition::VECOUT, NUM_ONE> scaleToWorkSpaceQueue;
    TQue<QuePosition::VECIN, NUM_ONE> scaleFromWorkSpaceQueue;
    TQue<QuePosition::VECOUT, NUM_ONE> MaxToWorkSpaceQueue;
    TQue<QuePosition::VECIN, NUM_ONE> MaxFromWorkSpaceQueue;
    TQue<QuePosition::VECOUT, NUM_ONE> MinToWorkSpaceQueue;
    TQue<QuePosition::VECIN, NUM_ONE> MinFromWorkSpaceQueue;
    TQue<QuePosition::VECOUT, NUM_ONE> MaxOutQueue;
    TQue<QuePosition::VECOUT, NUM_ONE> MinOutQueue;
    TQue<QuePosition::VECOUT, NUM_ONE> scaleOutQueue;
    TQue<QuePosition::VECOUT, NUM_ONE> offsetQueue;
    TQue<QuePosition::VECOUT, NUM_ONE> outQueue;

    GlobalTensor<T> inGm;
    GlobalTensor<T> smoothGm;
    GlobalTensor<yCopyDtype> outGm;
    GlobalTensor<float> scaleGm;
    GlobalTensor<float> offsetGm;
    GlobalTensor<float> workspaceTmp1;
    GlobalTensor<float> workspaceTmp2;

    uint32_t blockIdx;
    uint32_t tokenIdx;
    uint32_t coreLocalIdx;
    uint32_t coreNumPerToken;
    uint32_t sizeFloatLen;
    uint32_t outAlignLen;
    uint32_t elePerHeadCore;
    uint32_t elePerTailCore;
    uint32_t eleCount;
    uint32_t headSlices;
    uint32_t innerLoopCnt;
    uint32_t innerLoopFull;
    uint32_t innerLoopRem;
    uint32_t lenHead;
    uint32_t lenTail;
    uint32_t outLenHead;
    uint32_t outLenTail;
    uint32_t loopCnt;
    uint32_t loopCntHead;
    uint32_t loopCntTail;
    uint32_t totalTokenNum;
    uint8_t rightPadding = 0;

    int64_t offsetBase = 0;
    int64_t srcOffset = 0;
    int64_t scaleOffset = 0;
    int64_t coreStartOffset = 0;
    float maxValue = 0.0;
    float maxValueNoSym = 0.0;

    constexpr static AscendC::Reg::CastTrait castTrait0 = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::UNKNOWN};
    constexpr static AscendC::Reg::CastTrait castTraitF32toI16 = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_RINT};
    constexpr static AscendC::Reg::CastTrait castTraitI16toF16 = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_ROUND};
    constexpr static AscendC::Reg::CastTrait castTraitF16toI8 = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_TRUNC};
    constexpr static AscendC::Reg::CastTrait castTrait32tofp8 = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::SAT, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_RINT};
    constexpr static AscendC::Reg::CastTrait castTrait32toh8 = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::SAT, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_ROUND};
};

template <typename T, typename yDtype, int64_t hasSmooth, bool isSymmetrical>
__aicore__ inline void DynamicQuantPertenLargeMulticore<T, yDtype, hasSmooth, isSymmetrical>::Init(
    GM_ADDR x, GM_ADDR smooth_scales, GM_ADDR y, GM_ADDR scale, GM_ADDR offset, GM_ADDR workSpace,
    const DynamicQuantTilingDataArch35& tilingData)
{
    DynamicQuantNDOpt::SetFloatOverflowModeForRegbase<yDtype>();
    blockIdx = GetBlockIdx();

    ParseTilingData(tilingData);
    coreNumPerToken = tilingData_.headCoreNum;

    totalTokenNum = tilingData_.coreNum / coreNumPerToken;
    tokenIdx = blockIdx / coreNumPerToken;
    coreLocalIdx = blockIdx % coreNumPerToken;
    headSlices = tilingData_.multiRowNumHeadCore;

    elePerHeadCore = tilingData_.rowPerHeadCore;
    elePerTailCore = tilingData_.rowPerTailCore;
    eleCount = (coreLocalIdx < headSlices) ? elePerHeadCore : elePerTailCore;

    if (coreLocalIdx < headSlices) {
        coreStartOffset = static_cast<int64_t>(coreLocalIdx) * static_cast<int64_t>(elePerHeadCore);
    } else {
        coreStartOffset = static_cast<int64_t>(headSlices) * static_cast<int64_t>(elePerHeadCore) +
                          static_cast<int64_t>(coreLocalIdx - headSlices) * static_cast<int64_t>(elePerTailCore);
    }

    innerLoopCnt = (eleCount + tilingData_.innerLoopEle - 1) / tilingData_.innerLoopEle;
    innerLoopRem = (eleCount % tilingData_.innerLoopEle == 0) ? tilingData_.innerLoopEle :
                                                                eleCount % tilingData_.innerLoopEle;
    innerLoopFull = (eleCount == innerLoopRem) ? 0 : (eleCount - innerLoopRem) / tilingData_.innerLoopEle;

    uint32_t sizeTailLen = (innerLoopRem + FIFTEEN) / SIXTEEN * SIXTEEN;
    rightPadding = sizeTailLen - innerLoopRem;

    loopCnt = innerLoopCnt; // 每个核在尾轴上需要搬运的次数
    loopCntHead = loopCnt / THIRTY_TWO;
    loopCntTail = loopCnt % THIRTY_TWO;
    lenHead = totalTokenNum * tilingData_.rowLen;
    lenTail = lenHead;

    if constexpr (IsSameType<yDtype, int4b_t>::value) {
        outAlignLen = (tilingData_.innerLoopEle + SIXTY_THREE) / SIXTY_FOUR * SIXTY_FOUR;
        outLenHead = lenHead >> 1;
        outLenTail = outLenHead;
    } else {
        outAlignLen = tilingData_.innerLoopEle;
        outLenHead = lenHead;
        outLenTail = outLenHead;
    }

    inGm.SetGlobalBuffer((__gm__ T*)x, lenHead);
    outGm.SetGlobalBuffer((__gm__ yCopyDtype*)y, outLenHead);
    scaleGm.SetGlobalBuffer((__gm__ float*)scale, totalTokenNum);

    if constexpr (isSymmetrical == false) {
        workspaceTmp1.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workSpace), tilingData_.coreNum);
        workspaceTmp2.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workSpace + tilingData_.coreNum * sizeof(float)),
                                      tilingData_.coreNum);
    } else {
        workspaceTmp1.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workSpace), tilingData_.coreNum);
    }

    if constexpr (hasSmooth == 1) {
        smoothGm.SetGlobalBuffer((__gm__ T*)smooth_scales);
        pPipe->InitBuffer(smoothQueue, DOUBLE_BUFFER_NUM, tilingData_.innerLoopEle * sizeof(T));
    }
    if constexpr (isSymmetrical == false) {
        offsetGm.SetGlobalBuffer((__gm__ float*)offset);
        pPipe->InitBuffer(offsetQueue, DOUBLE_BUFFER_NUM, TWO_FIVE_SIX);
        pPipe->InitBuffer(MaxOutQueue, DOUBLE_BUFFER_NUM, TWO_FIVE_SIX);
        pPipe->InitBuffer(MinOutQueue, DOUBLE_BUFFER_NUM, TWO_FIVE_SIX);
        pPipe->InitBuffer(MaxToWorkSpaceQueue, DOUBLE_BUFFER_NUM, TWO_FIVE_SIX);
        pPipe->InitBuffer(MaxFromWorkSpaceQueue, DOUBLE_BUFFER_NUM, TWO_FIVE_SIX);
        pPipe->InitBuffer(MinToWorkSpaceQueue, DOUBLE_BUFFER_NUM, TWO_FIVE_SIX);
        pPipe->InitBuffer(MinFromWorkSpaceQueue, DOUBLE_BUFFER_NUM, TWO_FIVE_SIX);
    } else {
        pPipe->InitBuffer(scaleToWorkSpaceQueue, DOUBLE_BUFFER_NUM, TWO_FIVE_SIX);
        pPipe->InitBuffer(scaleFromWorkSpaceQueue, DOUBLE_BUFFER_NUM, TWO_FIVE_SIX);
    }
    pPipe->InitBuffer(inQueue, DOUBLE_BUFFER_NUM, tilingData_.innerLoopEle * sizeof(T));
    pPipe->InitBuffer(outQueue, DOUBLE_BUFFER_NUM, tilingData_.innerLoopEle * sizeof(yCopyDtype));
    pPipe->InitBuffer(scaleOutQueue, DOUBLE_BUFFER_NUM, TWO_FIVE_SIX);

    SetMaxValue();
}

template <typename T, typename yDtype, int64_t hasSmooth, bool isSymmetrical>
__aicore__ inline void DynamicQuantPertenLargeMulticore<T, yDtype, hasSmooth, isSymmetrical>::SetMaxValue()
{
    if constexpr (IsSameType<yDtype, int8_t>::value) {
        maxValue = INT8_MAX_VALUE;
        maxValueNoSym = INT8_MAX_VALUE_NO_SYM;
    } else if constexpr (IsSameType<yDtype, int4b_t>::value) {
        maxValue = INT4_MAX_VALUE;
        maxValueNoSym = INT4_MAX_VALUE_NO_SYM;
    } else if constexpr (IsSameType<yDtype, fp8_e5m2_t>::value) {
        maxValue = FP8_E5M2_MAX_VALUE;
        maxValueNoSym = FP8_E5M2_MAX_VALUE_NO_SYM;
    } else if constexpr (IsSameType<yDtype, fp8_e4m3fn_t>::value) {
        maxValue = FP8_E4M3FN_MAX_VALUE;
        maxValueNoSym = FP8_E4M3FN_MAX_VALUE_NO_SYM;
    } else if constexpr (IsSameType<yDtype, hifloat8_t>::value) {
        maxValue = tilingData_.dstTypeMax;
        maxValueNoSym = tilingData_.dstTypeMax * DynamicQuantNDOpt::SYM_RANGE_MULTI;
    }
}

template <typename T, typename yDtype, int64_t hasSmooth, bool isSymmetrical>
__aicore__ inline void DynamicQuantPertenLargeMulticore<T, yDtype, hasSmooth, isSymmetrical>::ParseTilingData(
    const DynamicQuantTilingDataArch35& tilingData)
{
    tilingData_.coreNum = tilingData.coreNum;
    tilingData_.rowLen = tilingData.rowLen;
    tilingData_.headCoreNum = tilingData.headCoreNum;
    tilingData_.rowPerHeadCore = tilingData.rowPerHeadCore;
    tilingData_.rowPerTailCore = tilingData.rowPerTailCore;
    tilingData_.multiRowNumTailCore = tilingData.multiRowNumTailCore;
    tilingData_.multiRowNumHeadCore = tilingData.multiRowNumHeadCore;
    tilingData_.innerLoopEle = tilingData.innerLoopEle;
    tilingData_.innerLoopTimes = tilingData.innerLoopTimes;
    tilingData_.innerLoopTail = tilingData.innerLoopTail;
    tilingData_.groupNum = tilingData.groupNum;
    tilingData_.alignGroupNum = tilingData.alignGroupNum;
    tilingData_.hasSmooth = tilingData.hasSmooth;
    tilingData_.dstTypeMax = tilingData.dstTypeMax;
}

template <typename T, typename yDtype, int64_t hasSmooth, bool isSymmetrical>
__aicore__ inline void DynamicQuantPertenLargeMulticore<T, yDtype, hasSmooth, isSymmetrical>::Process()
{
    if (blockIdx >= tilingData_.coreNum) {
        return;
    }
    ProcessScale();
    ProcessY();
}

template <typename T, typename yDtype, int64_t hasSmooth, bool isSymmetrical>
__aicore__ inline void DynamicQuantPertenLargeMulticore<T, yDtype, hasSmooth, isSymmetrical>::ProcessScale()
{
    ProcessScaleRow();
    SyncAll();
    ProcessScaleCol();
}

template <typename T, typename yDtype, int64_t hasSmooth, bool isSymmetrical>
__aicore__ inline void DynamicQuantPertenLargeMulticore<T, yDtype, hasSmooth, isSymmetrical>::ProcessY()
{
    LocalTensor<float> scaleOutLocal = scaleOutQueue.DeQue<float>();
    __local_mem__ float* scaleOutLocalAddr = (__local_mem__ float*)scaleOutLocal.GetPhyAddr();

    LocalTensor<float> offsetLocal;
    __local_mem__ float* offsetLocalAddr;

    if constexpr (isSymmetrical == false) {
        offsetLocal = offsetQueue.DeQue<float>();
        offsetLocalAddr = (__local_mem__ float*)offsetLocal.GetPhyAddr();
    }

    ProcessYRow(0, 0, scaleOutLocalAddr, offsetLocalAddr);

    if (coreLocalIdx == 0) {
        DataCopyParams copyParams{1, (uint16_t)(sizeof(float)), 0, 0};
        DataCopyPad(scaleGm[tokenIdx], scaleOutLocal, copyParams);
        if constexpr (isSymmetrical == false) {
            DataCopyPad(offsetGm[tokenIdx], offsetLocal, copyParams);
        }
    }
    scaleOutQueue.FreeTensor(scaleOutLocal);
    if constexpr (isSymmetrical == false) {
        offsetQueue.FreeTensor(offsetLocal);
    }
}

template <typename T, typename yDtype, int64_t hasSmooth, bool isSymmetrical>
__aicore__ inline void DynamicQuantPertenLargeMulticore<T, yDtype, hasSmooth, isSymmetrical>::ProcessScaleRowLoop(
    uint32_t i, uint32_t j)
{
    offsetBase = i * THIRTY_TWO + j;
    srcOffset = tokenIdx * tilingData_.rowLen + coreStartOffset + offsetBase * tilingData_.innerLoopEle;
    for (uint32_t innerLoopIndex = 0; innerLoopIndex < innerLoopFull; innerLoopIndex++) {
        CopyInByEle(srcOffset, innerLoopIndex, tilingData_.innerLoopEle, 0);
        ComputeMaxRowScale(tilingData_.innerLoopEle);
        srcOffset += tilingData_.innerLoopEle;
    }
    if (innerLoopRem > 0) {
        CopyInByEle(srcOffset, innerLoopFull, innerLoopRem, rightPadding);
        ComputeMaxRowScale(innerLoopRem);
    }
}

template <typename T, typename yDtype, int64_t hasSmooth, bool isSymmetrical>
__aicore__ inline void DynamicQuantPertenLargeMulticore<T, yDtype, hasSmooth, isSymmetrical>::ProcessScaleRow()
{
    LocalTensor<float> MaxToWorkSpaceLocal;
    LocalTensor<float> MinToWorkSpaceLocal;
    LocalTensor<float> scaleToWorkSpaceLocal;

    if constexpr (isSymmetrical == false) {
        MaxToWorkSpaceLocal = MaxToWorkSpaceQueue.AllocTensor<float>();
        AscendC::Duplicate(MaxToWorkSpaceLocal, MIN_FLOAT_VALUE, 64, 1, 1, 8);
        MaxToWorkSpaceQueue.EnQue(MaxToWorkSpaceLocal);

        MinToWorkSpaceLocal = MinToWorkSpaceQueue.AllocTensor<float>();
        AscendC::Duplicate(MinToWorkSpaceLocal, MAX_FLOAT_VALUE, 64, 1, 1, 8);
        MinToWorkSpaceQueue.EnQue(MinToWorkSpaceLocal);
    } else {
        scaleToWorkSpaceLocal = scaleToWorkSpaceQueue.AllocTensor<float>();
        AscendC::Duplicate(scaleToWorkSpaceLocal, (float)0.0, 64, 1, 1, 8);
        scaleToWorkSpaceQueue.EnQue(scaleToWorkSpaceLocal);
    }

    ProcessScaleRowLoop(0, 0);

    CopyUB2Workspace(1);
}

template <typename T, typename yDtype, int64_t hasSmooth, bool isSymmetrical>
__aicore__ inline void DynamicQuantPertenLargeMulticore<T, yDtype, hasSmooth, isSymmetrical>::ProcessScaleCol()
{
    LocalTensor<float> MaxOutLocal;
    LocalTensor<float> MinOutLocal;
    LocalTensor<float> scaleOutLocal;
    LocalTensor<float> offsetLocal;
    __local_mem__ float* MaxOutLocalAddr;
    __local_mem__ float* MinOutLocalAddr;
    __local_mem__ float* scaleOutLocalAddr;
    __local_mem__ float* offsetLocalAddr;

    if constexpr (isSymmetrical == false) {
        MaxOutLocal = MaxOutQueue.AllocTensor<float>();
        AscendC::Duplicate(MaxOutLocal, MIN_FLOAT_VALUE, 64, 1, 1, 8);
        MaxOutLocalAddr = (__local_mem__ float*)MaxOutLocal.GetPhyAddr();

        MinOutLocal = MinOutQueue.AllocTensor<float>();
        AscendC::Duplicate(MinOutLocal, MAX_FLOAT_VALUE, 64, 1, 1, 8);
        MinOutLocalAddr = (__local_mem__ float*)MinOutLocal.GetPhyAddr();

        scaleOutLocal = scaleOutQueue.AllocTensor<float>();
        AscendC::Duplicate(scaleOutLocal, (float)0.0, 64, 1, 1, 8);
        scaleOutLocalAddr = (__local_mem__ float*)scaleOutLocal.GetPhyAddr();

        offsetLocal = offsetQueue.AllocTensor<float>();
        AscendC::Duplicate(offsetLocal, (float)0.0, 64, 1, 1, 8);
        offsetLocalAddr = (__local_mem__ float*)offsetLocal.GetPhyAddr();
    } else {
        scaleOutLocal = scaleOutQueue.AllocTensor<float>();
        AscendC::Duplicate(scaleOutLocal, (float)0.0, 64, 1, 1, 8);
        scaleOutLocalAddr = (__local_mem__ float*)scaleOutLocal.GetPhyAddr();
    }
    scaleOffset = 0;
    CopyInScaleByEle(static_cast<int64_t>(tokenIdx) * static_cast<int64_t>(coreNumPerToken), coreNumPerToken);
    ComputeMaxColScale(coreNumPerToken, MaxOutLocalAddr, MinOutLocalAddr, scaleOutLocalAddr, offsetLocalAddr);
    if constexpr (isSymmetrical == false) {
        MaxOutQueue.FreeTensor(MaxOutLocal);
        MinOutQueue.FreeTensor(MinOutLocal);
        scaleOutQueue.EnQue(scaleOutLocal);
        offsetQueue.EnQue(offsetLocal);
    } else {
        scaleOutQueue.EnQue(scaleOutLocal);
    }
}

template <typename T, typename yDtype, int64_t hasSmooth, bool isSymmetrical>
__aicore__ inline void DynamicQuantPertenLargeMulticore<T, yDtype, hasSmooth, isSymmetrical>::CopyInByEle(
    int64_t offset, uint32_t loopIndex, uint32_t elementNum, uint8_t rightPadding)
{
    DataCopyParams copyParams{1, (uint16_t)(elementNum * sizeof(T)), 0, 0};
    DataCopyPadParams padParams{true, 0, rightPadding, 0};

    LocalTensor<T> inLocal = inQueue.AllocTensor<T>();
    DataCopyPad(inLocal, inGm[offset], copyParams, padParams);
    inQueue.EnQue(inLocal);

    if constexpr (hasSmooth == 1) {
        int64_t smoothOffset = offset - static_cast<int64_t>(tokenIdx) * static_cast<int64_t>(tilingData_.rowLen);
        LocalTensor<T> smoothLocal = smoothQueue.AllocTensor<T>();
        DataCopyPad(smoothLocal, smoothGm[smoothOffset], copyParams, padParams);
        smoothQueue.EnQue(smoothLocal);
    }
}

template <typename T, typename yDtype, int64_t hasSmooth, bool isSymmetrical>
__aicore__ inline void DynamicQuantPertenLargeMulticore<T, yDtype, hasSmooth, isSymmetrical>::CopyInScaleByEle(
    int64_t offset, uint32_t elementNum)
{
    DataCopyParams copyParams{1, (uint16_t)(elementNum * sizeof(float)), 0, 0};
    if constexpr (isSymmetrical == false) {
        LocalTensor<float> MaxFromWorkSpaceLocal = MaxFromWorkSpaceQueue.AllocTensor<float>();
        DataCopyPad(MaxFromWorkSpaceLocal, workspaceTmp1[offset], {1, (uint16_t)(elementNum * sizeof(float)), 0, 0},
                    {false, 0, 0, 0});
        MaxFromWorkSpaceQueue.EnQue(MaxFromWorkSpaceLocal);

        LocalTensor<float> MinFromWorkSpaceLocal = MinFromWorkSpaceQueue.AllocTensor<float>();
        DataCopyPad(MinFromWorkSpaceLocal, workspaceTmp2[offset], {1, (uint16_t)(elementNum * sizeof(float)), 0, 0},
                    {false, 0, 0, 0});
        MinFromWorkSpaceQueue.EnQue(MinFromWorkSpaceLocal);
    } else {
        LocalTensor<float> scaleFromWorkSpaceLocal = scaleFromWorkSpaceQueue.AllocTensor<float>();
        DataCopyPad(scaleFromWorkSpaceLocal, workspaceTmp1[offset], {1, (uint16_t)(elementNum * sizeof(float)), 0, 0},
                    {false, 0, 0, 0});
        scaleFromWorkSpaceQueue.EnQue(scaleFromWorkSpaceLocal);
    }
}

template <typename T, typename yDtype, int64_t hasSmooth, bool isSymmetrical>
__aicore__ inline void DynamicQuantPertenLargeMulticore<T, yDtype, hasSmooth, isSymmetrical>::CopyUB2Workspace(
    int64_t size)
{
    if constexpr (isSymmetrical == false) {
        auto tmp1 = MaxToWorkSpaceQueue.DeQue<float>();
        DataCopyPad(workspaceTmp1[static_cast<int64_t>(tokenIdx) * static_cast<int64_t>(coreNumPerToken) +
                                  static_cast<int64_t>(coreLocalIdx)],
                    tmp1, {1, (uint16_t)(size * sizeof(float)), 0, 0});
        MaxToWorkSpaceQueue.FreeTensor(tmp1);

        auto tmp2 = MinToWorkSpaceQueue.DeQue<float>();
        DataCopyPad(workspaceTmp2[static_cast<int64_t>(tokenIdx) * static_cast<int64_t>(coreNumPerToken) +
                                  static_cast<int64_t>(coreLocalIdx)],
                    tmp2, {1, (uint16_t)(size * sizeof(float)), 0, 0});
        MinToWorkSpaceQueue.FreeTensor(tmp2);
    } else {
        auto tmp = scaleToWorkSpaceQueue.DeQue<float>();
        DataCopyPad(workspaceTmp1[static_cast<int64_t>(tokenIdx) * static_cast<int64_t>(coreNumPerToken) +
                                  static_cast<int64_t>(coreLocalIdx)],
                    tmp, {1, (uint16_t)(size * sizeof(float)), 0, 0});
        scaleToWorkSpaceQueue.FreeTensor(tmp);
    }
}

template <typename T, typename yDtype, int64_t hasSmooth, bool isSymmetrical>
__aicore__ inline void DynamicQuantPertenLargeMulticore<T, yDtype, hasSmooth, isSymmetrical>::ComputeMaxRowScale(
    uint32_t elementNum)
{
    LocalTensor<T> inLocal = inQueue.DeQue<T>();
    __local_mem__ T* inLocalAddr = (__local_mem__ T*)inLocal.GetPhyAddr();
    LocalTensor<T> smoothLocal;
    __local_mem__ T* smoothLocalAddr;

    LocalTensor<float> maxToWorkSpaceLocal;
    __local_mem__ float* maxToWorkSpaceLocalAddr;
    LocalTensor<float> minToWorkSpaceLocal;
    __local_mem__ float* minToWorkSpaceLocalAddr;
    LocalTensor<float> scaleToWorkSpaceLocal;
    __local_mem__ float* scaleToWorkSpaceLocalAddr;

    if constexpr (hasSmooth == 1) {
        smoothLocal = smoothQueue.DeQue<T>();
        smoothLocalAddr = (__local_mem__ T*)smoothLocal.GetPhyAddr();
    }
    if constexpr (isSymmetrical == false) {
        maxToWorkSpaceLocal = MaxToWorkSpaceQueue.DeQue<float>();
        maxToWorkSpaceLocalAddr = (__local_mem__ float*)maxToWorkSpaceLocal.GetPhyAddr();

        minToWorkSpaceLocal = MinToWorkSpaceQueue.DeQue<float>();
        minToWorkSpaceLocalAddr = (__local_mem__ float*)minToWorkSpaceLocal.GetPhyAddr();
    } else {
        scaleToWorkSpaceLocal = scaleToWorkSpaceQueue.DeQue<float>();
        scaleToWorkSpaceLocalAddr = (__local_mem__ float*)scaleToWorkSpaceLocal.GetPhyAddr();
    }

    ComputeMaxRowScaleVF(inLocalAddr, smoothLocalAddr, scaleToWorkSpaceLocalAddr, maxToWorkSpaceLocalAddr,
                         minToWorkSpaceLocalAddr, elementNum);
    if constexpr (hasSmooth == 1) {
        smoothQueue.FreeTensor(smoothLocal);
    }
    if constexpr (isSymmetrical == false) {
        MaxToWorkSpaceQueue.EnQue(maxToWorkSpaceLocal);
        MinToWorkSpaceQueue.EnQue(minToWorkSpaceLocal);
    } else {
        scaleToWorkSpaceQueue.EnQue(scaleToWorkSpaceLocal);
    }
    inQueue.FreeTensor(inLocal);
}

template <typename T, typename yDtype, int64_t hasSmooth, bool isSymmetrical>
__aicore__ inline void DynamicQuantPertenLargeMulticore<T, yDtype, hasSmooth, isSymmetrical>::ComputeMaxColScale(
    uint32_t elementNum, __local_mem__ float* maxAddr, __local_mem__ float* minAddr, __local_mem__ float* scaleAddr,
    __local_mem__ float* offsetAddr)
{
    LocalTensor<float> scaleFromWorkSpaceLocal;
    __local_mem__ float* scaleFromWorkSpaceLocalAddr;

    LocalTensor<float> maxFromWorkSpaceLocal;
    __local_mem__ float* maxFromWorkSpaceLocalAddr;

    LocalTensor<float> minFromWorkSpaceLocal;
    __local_mem__ float* minFromWorkSpaceLocalAddr;

    if constexpr (isSymmetrical == false) {
        maxFromWorkSpaceLocal = MaxFromWorkSpaceQueue.DeQue<float>();
        maxFromWorkSpaceLocalAddr = (__local_mem__ float*)maxFromWorkSpaceLocal.GetPhyAddr();

        minFromWorkSpaceLocal = MinFromWorkSpaceQueue.DeQue<float>();
        minFromWorkSpaceLocalAddr = (__local_mem__ float*)minFromWorkSpaceLocal.GetPhyAddr();

        ComputeMaxColScaleVF(scaleFromWorkSpaceLocalAddr, scaleAddr, maxFromWorkSpaceLocalAddr, maxAddr,
                             minFromWorkSpaceLocalAddr, minAddr, elementNum);
        ComputeScaleSymVF(maxAddr, minAddr, scaleAddr, 1);
        ComputeOffsetSymVF(maxAddr, scaleAddr, offsetAddr, 1);

        MaxFromWorkSpaceQueue.FreeTensor(maxFromWorkSpaceLocal);
        MinFromWorkSpaceQueue.FreeTensor(minFromWorkSpaceLocal);
    } else {
        scaleFromWorkSpaceLocal = scaleFromWorkSpaceQueue.DeQue<float>();
        scaleFromWorkSpaceLocalAddr = (__local_mem__ float*)scaleFromWorkSpaceLocal.GetPhyAddr();

        ComputeMaxColScaleVF(scaleFromWorkSpaceLocalAddr, scaleAddr, maxFromWorkSpaceLocalAddr, maxAddr,
                             minFromWorkSpaceLocalAddr, minAddr, elementNum);
        scaleFromWorkSpaceQueue.FreeTensor(scaleFromWorkSpaceLocal);
    }
}

template <typename T, typename yDtype, int64_t hasSmooth, bool isSymmetrical>
__aicore__ inline void DynamicQuantPertenLargeMulticore<T, yDtype, hasSmooth, isSymmetrical>::ComputeMaxRowScaleVF(
    __local_mem__ T* inLocalAddr, __local_mem__ T* smoothLocalAddr, __local_mem__ float* scaleLocalAddr,
    __local_mem__ float* maxLocalAddr, __local_mem__ float* minLocalAddr, uint32_t elementNum)
{
    uint32_t dtypeSize = sizeof(float);
    uint32_t VL = AscendC::VECTOR_REG_WIDTH / dtypeSize;
    uint16_t vfLoopNum = (elementNum + VL - 1) / VL;

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<T> vreg1;
        AscendC::Reg::RegTensor<T> vreg2;
        AscendC::Reg::RegTensor<float> vreg3;
        AscendC::Reg::RegTensor<float> vreg4;
        AscendC::Reg::RegTensor<float> vreg5;
        AscendC::Reg::RegTensor<float> vreg6_1;
        AscendC::Reg::RegTensor<float> vreg6_2;
        AscendC::Reg::RegTensor<float> vreg7_1;
        AscendC::Reg::RegTensor<float> vreg7_2;
        AscendC::Reg::RegTensor<float> vreg8_1;
        AscendC::Reg::RegTensor<float> vreg8_2;
        AscendC::Reg::RegTensor<float> vreg9_1;
        AscendC::Reg::RegTensor<float> vreg9_2;

        AscendC::Reg::MaskReg preg0;
        AscendC::Reg::MaskReg preg1 = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();

        if constexpr (isSymmetrical == false) {
            AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg9_1, maxLocalAddr);
            AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg9_2, minLocalAddr);
        } else {
            AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg9_1, scaleLocalAddr);
        }

        for (uint16_t i = 0; i < vfLoopNum; i++) {
            uint32_t sreg0 = elementNum - i * VL;
            preg0 = AscendC::Reg::UpdateMask<float>(sreg0);
            AscendC::Reg::DataCopy<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(vreg1, inLocalAddr + i * VL);
            AscendC::Reg::Cast<float, T, castTrait0>(vreg3, vreg1, preg0);
            if constexpr (hasSmooth == 1) {
                AscendC::Reg::DataCopy<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(vreg2, smoothLocalAddr + i * VL);
                AscendC::Reg::Cast<float, T, castTrait0>(vreg4, vreg2, preg0);
                AscendC::Reg::Mul(vreg3, vreg3, vreg4, preg0);
            }
            if constexpr (isSymmetrical == false) {
                AscendC::Reg::Max(vreg9_1, vreg3, vreg9_1, preg0);
                AscendC::Reg::Min(vreg9_2, vreg3, vreg9_2, preg0);
            } else {
                AscendC::Reg::Abs(vreg6_1, vreg3, preg0);
                AscendC::Reg::Muls(vreg6_1, vreg6_1, float(1.0) / maxValue, preg0);
                AscendC::Reg::Max(vreg9_1, vreg6_1, vreg9_1, preg0);
            }
        }

        if constexpr (isSymmetrical == false) {
            AscendC::Reg::ReduceMax(vreg8_1, vreg9_1, preg1);
            AscendC::Reg::ReduceMin(vreg8_2, vreg9_2, preg1);
            AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(maxLocalAddr, vreg8_1,
                                                                                           preg0);
            AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(minLocalAddr, vreg8_2,
                                                                                           preg0);
        } else {
            AscendC::Reg::ReduceMax(vreg8_1, vreg9_1, preg1);
            AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(scaleLocalAddr, vreg8_1,
                                                                                           preg0);
        }
    }
}

template <typename T, typename yDtype, int64_t hasSmooth, bool isSymmetrical>
__aicore__ inline void DynamicQuantPertenLargeMulticore<T, yDtype, hasSmooth, isSymmetrical>::ComputeMaxColScaleVF(
    __local_mem__ float* scaleLocalAddr, __local_mem__ float* scaleOutLocalAddr, __local_mem__ float* maxLocalAddr,
    __local_mem__ float* maxOutLocalAddr, __local_mem__ float* minLocalAddr, __local_mem__ float* minOutLocalAddr,
    uint32_t elementNum)
{
    uint32_t dtypeSize = sizeof(float);
    uint32_t VL = AscendC::VECTOR_REG_WIDTH / dtypeSize;
    uint16_t vfLoopNum = (elementNum + VL - 1) / VL;
    uint32_t maskNum;

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<float, AscendC::Reg::RegTraitNumOne> vreg0_1;
        AscendC::Reg::RegTensor<float, AscendC::Reg::RegTraitNumOne> vreg0_2;
        AscendC::Reg::RegTensor<float, AscendC::Reg::RegTraitNumOne> vreg1_1;
        AscendC::Reg::RegTensor<float, AscendC::Reg::RegTraitNumOne> vreg1_2;
        AscendC::Reg::RegTensor<float, AscendC::Reg::RegTraitNumOne> vreg2_1;
        AscendC::Reg::RegTensor<float, AscendC::Reg::RegTraitNumOne> vreg2_2;
        AscendC::Reg::RegTensor<float, AscendC::Reg::RegTraitNumOne> vreg3_1;
        AscendC::Reg::RegTensor<float, AscendC::Reg::RegTraitNumOne> vreg3_2;

        AscendC::Reg::MaskReg mask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();

        if constexpr (isSymmetrical == false) {
            AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg1_1, maxOutLocalAddr);
            AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg1_2, minOutLocalAddr);
        } else {
            AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg1_1, scaleOutLocalAddr);
        }

        for (uint16_t i = 0; i < static_cast<uint16_t>(vfLoopNum - 1); i++) {
            maskNum = elementNum - i * VL;
            mask = AscendC::Reg::UpdateMask<float>(maskNum);
            if constexpr (isSymmetrical == false) {
                AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(vreg0_1, maxLocalAddr + i * VL);
                AscendC::Reg::Max(vreg1_1, vreg0_1, vreg1_1, mask);
                AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(vreg0_2, minLocalAddr + i * VL);
                AscendC::Reg::Min(vreg1_2, vreg0_2, vreg1_2, mask);
            } else {
                AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(vreg0_1, scaleLocalAddr + i * VL);
                AscendC::Reg::Max(vreg1_1, vreg0_1, vreg1_1, mask);
            }
        }
        {
            if constexpr (isSymmetrical == false) {
                AscendC::Reg::ReduceMax<float>(vreg2_1, vreg1_1, mask);
                AscendC::Reg::ReduceMin<float>(vreg2_2, vreg1_2, mask);
            } else {
                AscendC::Reg::ReduceMax<float>(vreg2_1, vreg1_1, mask);
            }
        }
        for (uint16_t i = vfLoopNum - 1; i < vfLoopNum; i++) {
            maskNum = elementNum - i * VL;
            mask = AscendC::Reg::UpdateMask<float>(maskNum);
            if constexpr (isSymmetrical == false) {
                AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(vreg0_1, maxLocalAddr + i * VL);
                AscendC::Reg::Max(vreg1_1, vreg0_1, vreg1_1, mask);
                AscendC::Reg::ReduceMax<float>(vreg3_1, vreg1_1, mask);
                AscendC::Reg::Max(vreg3_1, vreg2_1, vreg3_1, mask);

                AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(vreg0_2, minLocalAddr + i * VL);
                AscendC::Reg::Min(vreg1_2, vreg0_2, vreg1_2, mask);
                AscendC::Reg::ReduceMin<float>(vreg3_2, vreg1_2, mask);
                AscendC::Reg::Min(vreg3_2, vreg2_2, vreg3_2, mask);
            } else {
                AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(vreg0_1, scaleLocalAddr + i * VL);
                AscendC::Reg::Max(vreg1_1, vreg0_1, vreg1_1, mask);
                AscendC::Reg::ReduceMax<float>(vreg3_1, vreg1_1, mask);
                AscendC::Reg::Max(vreg3_1, vreg2_1, vreg3_1, mask);
            }
        }
        if constexpr (isSymmetrical == false) {
            AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(maxOutLocalAddr, vreg3_1,
                                                                                           mask);
            AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(minOutLocalAddr, vreg3_2,
                                                                                           mask);
        } else {
            AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(scaleOutLocalAddr, vreg3_1,
                                                                                           mask);
        }
    }
}

template <typename T, typename yDtype, int64_t hasSmooth, bool isSymmetrical>
__aicore__ inline void DynamicQuantPertenLargeMulticore<T, yDtype, hasSmooth, isSymmetrical>::ComputeScaleSymVF(
    __local_mem__ float* maxLocalAddr, __local_mem__ float* minLocalAddr, __local_mem__ float* scaleLocalAddr,
    uint32_t elementNum)
{
    uint32_t dtypeSize = sizeof(float);
    uint32_t VL = AscendC::VECTOR_REG_WIDTH / dtypeSize;
    uint16_t vfLoopNum = (elementNum + VL - 1) / VL;
    uint32_t maskNum;

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<float, AscendC::Reg::RegTraitNumOne> vreg0;
        AscendC::Reg::RegTensor<float, AscendC::Reg::RegTraitNumOne> vreg1;
        AscendC::Reg::MaskReg mask;

        for (uint16_t i = 0; i < vfLoopNum; i++) {
            maskNum = elementNum - i * VL;
            mask = AscendC::Reg::UpdateMask<float>(maskNum);

            AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(vreg0, maxLocalAddr + i * VL);
            AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(vreg1, minLocalAddr + i * VL);
            AscendC::Reg::Sub(vreg1, vreg0, vreg1, mask);
            AscendC::Reg::Muls(vreg1, vreg1, float(1.0) / maxValueNoSym, mask);
            AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(scaleLocalAddr, vreg1, mask);
        }
    }
}

template <typename T, typename yDtype, int64_t hasSmooth, bool isSymmetrical>
__aicore__ inline void DynamicQuantPertenLargeMulticore<T, yDtype, hasSmooth, isSymmetrical>::ComputeOffsetSymVF(
    __local_mem__ float* maxLocalAddr, __local_mem__ float* scaleLocalAddr, __local_mem__ float* offsetLocalAddr,
    uint32_t elementNum)
{
    uint32_t dtypeSize = sizeof(float);
    uint32_t VL = AscendC::VECTOR_REG_WIDTH / dtypeSize;
    uint16_t vfLoopNum = (elementNum + VL - 1) / VL;
    uint32_t maskNum;

    __VEC_SCOPE__
    {
        static constexpr AscendC::Reg::DivSpecificMode mode = {AscendC::Reg::MaskMergeMode::ZEROING, true};
        AscendC::Reg::RegTensor<float, AscendC::Reg::RegTraitNumOne> vreg0;
        AscendC::Reg::RegTensor<float, AscendC::Reg::RegTraitNumOne> vreg1;
        AscendC::Reg::MaskReg mask;

        for (uint16_t i = 0; i < vfLoopNum; i++) {
            maskNum = elementNum - i * VL;
            mask = AscendC::Reg::UpdateMask<float>(maskNum);

            AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(vreg0, maxLocalAddr + i * VL);
            AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(vreg1, scaleLocalAddr + i * VL);
            AscendC::Reg::Div<float, &mode>(vreg1, vreg0, vreg1, mask);
            AscendC::Reg::Muls(vreg1, vreg1, float(-1.0), mask);
            AscendC::Reg::Adds(vreg1, vreg1, maxValue, mask);

            AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_FIRST_ELEMENT_B32>(offsetLocalAddr, vreg1,
                                                                                           mask);
        }
    }
}

template <typename T, typename yDtype, int64_t hasSmooth, bool isSymmetrical>
__aicore__ inline void DynamicQuantPertenLargeMulticore<T, yDtype, hasSmooth, isSymmetrical>::ComputeY(
    uint32_t elementNum, __local_mem__ float* scaleAddr, __local_mem__ float* offsetAddr)
{
    LocalTensor<T> inLocal = inQueue.DeQue<T>();
    __local_mem__ T* inLocalAddr = (__local_mem__ T*)inLocal.GetPhyAddr();
    LocalTensor<yCopyDtype> outLocal = outQueue.AllocTensor<yCopyDtype>();
    __local_mem__ yCopyDtype* outAddr = (__local_mem__ yCopyDtype*)outLocal.GetPhyAddr();
    LocalTensor<T> smoothLocal;
    __local_mem__ T* smoothLocalAddr;

    if constexpr (hasSmooth == 1) {
        smoothLocal = smoothQueue.DeQue<T>();
        smoothLocalAddr = (__local_mem__ T*)smoothLocal.GetPhyAddr();
    }

    ComputeYVF(inLocalAddr, smoothLocalAddr, outAddr, scaleAddr, offsetAddr, elementNum);

    if constexpr (hasSmooth == 1) {
        smoothQueue.FreeTensor(smoothLocal);
    }
    inQueue.FreeTensor(inLocal);
    outQueue.EnQue(outLocal);
}

template <typename T, typename yDtype, int64_t hasSmooth, bool isSymmetrical>
__aicore__ inline void DynamicQuantPertenLargeMulticore<T, yDtype, hasSmooth, isSymmetrical>::ComputeYVF(
    __local_mem__ T* inLocalAddr, __local_mem__ T* smoothLocalAddr, __local_mem__ yCopyDtype* outAddr,
    __local_mem__ float* scaleLocalAddr, __local_mem__ float* offsetLocalAddr, uint32_t elementNum)
{
    uint32_t dtypeSize = sizeof(float);
    uint32_t VL = AscendC::VECTOR_REG_WIDTH / dtypeSize;
    uint16_t vfLoopNum = (elementNum + VL - 1) / VL;

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<float> vreg_scale;
        AscendC::Reg::RegTensor<float> vreg_offset;

        AscendC::Reg::RegTensor<T> vreg0;
        AscendC::Reg::RegTensor<T> vreg2;
        AscendC::Reg::RegTensor<float> vreg1;
        AscendC::Reg::RegTensor<float> vreg3;
        AscendC::Reg::RegTensor<float> vreg4;
        AscendC::Reg::RegTensor<float> vreg5;
        AscendC::Reg::RegTensor<int16_t> vreg6;
        AscendC::Reg::RegTensor<half> vreg7;
        AscendC::Reg::RegTensor<yCopyDtype> vreg8;

        AscendC::Reg::MaskReg mask;
        AscendC::Reg::MaskReg mask2 = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::H>();

        AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_scale, scaleLocalAddr);
        if constexpr (isSymmetrical == false) {
            AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_BRC_B32>(vreg_offset, offsetLocalAddr);
        }

        for (uint16_t i = 0; i < vfLoopNum; i++) {
            auto addr = outAddr + i * VL;
            mask = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
            AscendC::Reg::DataCopy<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(vreg0, inLocalAddr + i * VL);
            AscendC::Reg::Cast<float, T, castTrait0>(vreg1, vreg0, mask);
            if constexpr (hasSmooth == 1) {
                AscendC::Reg::DataCopy<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(vreg2, smoothLocalAddr + i * VL);
                AscendC::Reg::Cast<float, T, castTrait0>(vreg3, vreg2, mask);
                AscendC::Reg::Mul(vreg4, vreg1, vreg3, mask);
                AscendC::Reg::Div(vreg5, vreg4, vreg_scale, mask);
            } else {
                AscendC::Reg::Div(vreg5, vreg1, vreg_scale, mask);
            }
            if constexpr (isSymmetrical == false) {
                AscendC::Reg::Add(vreg5, vreg5, vreg_offset, mask);
            }

            if constexpr (IsSameType<yDtype, int8_t>::value) {
                AscendC::Reg::Cast<int16_t, float, castTraitF32toI16>(vreg6, vreg5, mask);
                AscendC::Reg::Cast<half, int16_t, castTraitI16toF16>(vreg7, vreg6, mask);
                AscendC::Reg::Cast<yDtype, half, castTraitF16toI8>(vreg8, vreg7, mask);
            } else if constexpr (IsSameType<yDtype, int4b_t>::value) {
                AscendC::Reg::RegTensor<uint16_t> vreg20;
                AscendC::Reg::Cast<int16_t, float, castTraitF32toI16>(vreg6, vreg5, mask);
                AscendC::Reg::Cast<half, int16_t, castTraitI16toF16>(vreg7, vreg6, mask);
                AscendC::Reg::Pack(vreg20, (AscendC::Reg::RegTensor<uint32_t>&)vreg7);
                AscendC::Reg::Cast<int4x2_t, half, castTraitF16toI8>((AscendC::Reg::RegTensor<int4x2_t>&)vreg8,
                                                                     (AscendC::Reg::RegTensor<half>&)vreg20, mask);
                addr = outAddr + (i * VL) / 2;
            } else if constexpr (IsSameType<yDtype, hifloat8_t>::value) {
                AscendC::Reg::Cast<yDtype, float, castTrait32toh8>(vreg8, vreg5, mask);
            } else if constexpr (IsSameType<yDtype, fp8_e4m3fn_t>::value || IsSameType<yDtype, fp8_e5m2_t>::value) {
                AscendC::Reg::Cast<yDtype, float, castTrait32tofp8>(vreg8, vreg5, mask);
            }

            if constexpr (IsSameType<yDtype, int4b_t>::value) {
                AscendC::Reg::DataCopy<yCopyDtype, AscendC::Reg::StoreDist::DIST_PACK4_B32>(addr, vreg8, mask2);
            } else {
                AscendC::Reg::DataCopy<yCopyDtype, AscendC::Reg::StoreDist::DIST_PACK4_B32>(addr, vreg8, mask);
            }
        }
    }
}

template <typename T, typename yDtype, int64_t hasSmooth, bool isSymmetrical>
__aicore__ inline void DynamicQuantPertenLargeMulticore<T, yDtype, hasSmooth, isSymmetrical>::CopyOutY(int64_t offset,
                                                                                                       uint32_t element)
{
    LocalTensor<yCopyDtype> outLocal = outQueue.DeQue<yCopyDtype>();
    DataCopyExtParams copyExtParams{(uint16_t)1, (uint16_t)(element * sizeof(yCopyDtype)), 0, 0, 0};
    if constexpr (IsSameType<yDtype, int4b_t>::value) {
        copyExtParams.blockLen = copyExtParams.blockLen >> 1;
        uint32_t offset2 = offset / 2;
        DataCopyPad(outGm[offset2], outLocal, copyExtParams);
    } else {
        DataCopyPad(outGm[offset], outLocal, copyExtParams);
    }
    outQueue.FreeTensor(outLocal);
}

template <typename T, typename yDtype, int64_t hasSmooth, bool isSymmetrical>
__aicore__ inline void DynamicQuantPertenLargeMulticore<T, yDtype, hasSmooth, isSymmetrical>::ProcessYRow(
    uint32_t i, uint32_t j, __local_mem__ float* scaleAddr, __local_mem__ float* offsetAddr)
{
    offsetBase = i * THIRTY_TWO + j;
    srcOffset = tokenIdx * tilingData_.rowLen + coreStartOffset + offsetBase * tilingData_.innerLoopEle;
    uint32_t loopIdx = 0;
    for (uint32_t innerLoopIndex = 0; innerLoopIndex < innerLoopFull; innerLoopIndex++) {
        CopyInByEle(srcOffset, innerLoopIndex, tilingData_.innerLoopEle, 0);
        ComputeY(tilingData_.innerLoopEle, scaleAddr, offsetAddr);
        CopyOutY(srcOffset, tilingData_.innerLoopEle);
        srcOffset += tilingData_.innerLoopEle;
    }
    if (innerLoopRem > 0) {
        CopyInByEle(srcOffset, innerLoopFull, innerLoopRem, rightPadding);
        ComputeY(innerLoopRem, scaleAddr, offsetAddr);
        CopyOutY(srcOffset, innerLoopRem);
    }
}

} // namespace DynamicQuantPertenLargeMc
#endif // DYNAMIC_QUANT_REGBASE_PERTEN_LARGE_MULTICORE_H
