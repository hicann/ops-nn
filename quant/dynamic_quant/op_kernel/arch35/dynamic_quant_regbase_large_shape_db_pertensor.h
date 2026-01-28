/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the 'License').
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
/*!
 * \file dynamic_quant_large_shape_db_pertensor.h
 * \brief
 */
#ifndef DYNAMIC_QUANT_REGBASE_LARGE_SHAPE_DB_PERTENSOR_H
#define DYNAMIC_QUANT_REGBASE_LARGE_SHAPE_DB_PERTENSOR_H

#include <cmath>
#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "dynamic_quant_regbase_base.h"

namespace DynamicQuantNDPerTensorOpt2 {
using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 1;
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

template <typename T, typename yDtype, int64_t hasSmooth, bool isSymmertrical>
class DynamicQuantLargeShapeDbPertensor {
public:
    using yCopyDtype = std::conditional_t<IsSameType<yDtype, int4b_t>::value, uint8_t, yDtype>;
    __aicore__ inline DynamicQuantLargeShapeDbPertensor(TPipe *pipe) { pPipe = pipe; }
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR smooth_scales, GM_ADDR y, GM_ADDR scale, GM_ADDR offset,
                                GM_ADDR workSpace, const DynamicQuantTilingData& tilingData);
    __aicore__ inline void Process();
private:
    __aicore__ inline void SetMaxValue();
    __aicore__ inline void ProcessScale();
    __aicore__ inline void ProcessY();
    __aicore__ inline void ProcessScaleRowLoop(uint32_t i, uint32_t j);
    __aicore__ inline void ProcessScaleRow();
    __aicore__ inline void ProcessScaleCol();
    __aicore__ inline void ProcessYRow(uint32_t i, uint32_t j,__local_mem__ float* scaleAddr,__local_mem__ float* offsetAddr);
    __aicore__ inline void CopyInByEle(int64_t offset, uint32_t loopIndex, uint32_t elementNum, uint8_t rightPadding);
    __aicore__ inline void CopyInScaleByEle(int64_t offset, uint32_t elementNum);
    __aicore__ inline void ComputeMaxRowScale(uint32_t elementNum);
    __aicore__ inline void ComputeMaxColScale(uint32_t elementNum,__local_mem__ float* maxAddr,__local_mem__ float* minAddr,__local_mem__ float* scaleAddr,__local_mem__ float* offsetAddr);
    __aicore__ inline void ComputeMaxRowScaleVF(__local_mem__ T* inLocalAddr, __local_mem__ T* smoothLocalAddr, __local_mem__ float* scaleLocalAddr, __local_mem__ float* maxLocalAddr, __local_mem__ float* minLocalAddr, uint32_t elementNum);
    __aicore__ inline void ComputeMaxColScaleVF(__local_mem__ float* scaleLocalAddr,__local_mem__ float* scaleOutLocalAddr,__local_mem__ float* maxLocalAddr,__local_mem__ float* maxOutLocalAddr,__local_mem__ float* minLocalAddr,__local_mem__ float* minOutLocalAddr,uint32_t elementNum);
    __aicore__ inline void ComputeY(uint32_t elementNum,__local_mem__ float* scaleAddr,__local_mem__ float* offsetAddr);
    __aicore__ inline void ComputeScaleSymVF(__local_mem__ float* maxLocalAddr, __local_mem__ float* minLocalAddr, __local_mem__ float* scaleLocalAddr,uint32_t elementNum);
    __aicore__ inline void ComputeOffsetSymVF(__local_mem__ float* maxLocalAddr, __local_mem__ float* scaleLocalAddr, __local_mem__ float* offsetLocalAddr,uint32_t elementNum);
    __aicore__ inline void ComputeYVF(__local_mem__ T* inLocalAddr, __local_mem__ T* smoothLocalAddr,__local_mem__ yCopyDtype* outAddr, __local_mem__ float* scaleLocalAddr, __local_mem__ float* offsetLocalAddr, uint32_t elementNum);
    __aicore__ inline void ParseTilingData(const DynamicQuantTilingData& tilingData);
    __aicore__ inline void CopyOutY(int64_t offset, uint32_t element);
    __aicore__ inline void CopyUB2Workspace(int64_t size);
private:
    TPipe* pPipe = nullptr;
    /* tiling data */
    DynamicQuantTilingData tilingData_;

    /* ascendc variable */
    TQue<QuePosition::VECIN, DOUBLE_BUFFER_NUM> inQueue;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER_NUM> smoothQueue;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER_NUM> scaleToWorkSpaceQueue;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER_NUM> scaleFromWorkSpaceQueue;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER_NUM> MaxToWorkSpaceQueue;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER_NUM> MaxFromWorkSpaceQueue;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER_NUM> MinToWorkSpaceQueue;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER_NUM> MinFromWorkSpaceQueue;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER_NUM> MaxOutQueue;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER_NUM> MinOutQueue;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER_NUM> scaleOutQueue;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER_NUM> offsetQueue;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER_NUM> outQueue;

    /* global memory address */
    GlobalTensor<T> inGm;
    GlobalTensor<T> smoothGm;
    GlobalTensor<yCopyDtype> outGm;
    GlobalTensor<float> scaleGm;
    GlobalTensor<float> offsetGm;
    GlobalTensor<float> workspaceTmp1;
    GlobalTensor<float> workspaceTmp2;

    uint32_t blockIdx;
    uint32_t sizeFloatLen;
    uint32_t outAlignLen;
    uint32_t rowPerHeadCore;
    uint32_t rowPerTailCore;
    uint32_t lenHead;
    uint32_t lenTail;
    uint32_t outLen;
    uint32_t outLenHead;
    uint32_t outLenTail;
    uint32_t loopCnt = 0;
    uint32_t loopCntHead;
    uint32_t loopCntTail;
    uint8_t rightPadding = 0;

    int64_t offsetBase = 0;
    int64_t srcOffset = 0;
    int64_t scaleOffset = 0;
    float maxValue = 0.0;
    float maxValueNoSym = 0.0;

    constexpr static AscendC::MicroAPI::CastTrait castTrait0 = {
        AscendC::MicroAPI::RegLayout::ZERO,
        AscendC::MicroAPI::SatMode::UNKNOWN,
        AscendC::MicroAPI::MaskMergeMode::ZEROING,
        AscendC::RoundMode::UNKNOWN};
    constexpr static AscendC::MicroAPI::CastTrait castTraitF32toI16 = {
        AscendC::MicroAPI::RegLayout::ZERO,
        AscendC::MicroAPI::SatMode::NO_SAT,
        AscendC::MicroAPI::MaskMergeMode::ZEROING,
        RoundMode::CAST_RINT};
    constexpr static AscendC::MicroAPI::CastTrait castTraitI16toF16 = {
        AscendC::MicroAPI::RegLayout::ZERO,
        AscendC::MicroAPI::SatMode::NO_SAT,
        AscendC::MicroAPI::MaskMergeMode::ZEROING,
        RoundMode::CAST_ROUND};
    constexpr static AscendC::MicroAPI::CastTrait castTraitF16toI8 = {
        AscendC::MicroAPI::RegLayout::ZERO,
        AscendC::MicroAPI::SatMode::NO_SAT,
        AscendC::MicroAPI::MaskMergeMode::ZEROING,
        RoundMode::CAST_TRUNC};
    constexpr static AscendC::MicroAPI::CastTrait castTrait32tofp8 = {
        AscendC::MicroAPI::RegLayout::ZERO,
        AscendC::MicroAPI::SatMode::SAT,
        AscendC::MicroAPI::MaskMergeMode::ZEROING,
        RoundMode::CAST_RINT};
    constexpr static AscendC::MicroAPI::CastTrait castTrait32toh8 = {
        AscendC::MicroAPI::RegLayout::ZERO,
        AscendC::MicroAPI::SatMode::SAT,
        AscendC::MicroAPI::MaskMergeMode::ZEROING,
        RoundMode::CAST_ROUND};
};

template <typename T, typename yDtype, int64_t hasSmooth, bool isSymmertrical>
__aicore__ inline void DynamicQuantLargeShapeDbPertensor<T, yDtype, hasSmooth, isSymmertrical>::Init(GM_ADDR x, GM_ADDR smooth_scales, GM_ADDR y, GM_ADDR scale, GM_ADDR offset,
                                                                            GM_ADDR workSpace, const DynamicQuantTilingData& tilingData) {
    DynamicQuantNDOpt::SetFloatOverflowModeForRegbase<yDtype>();                                                                                
    blockIdx = GetBlockIdx();

    ParseTilingData(tilingData);

    rowPerHeadCore = tilingData_.rowPerHeadCore;
    rowPerTailCore = tilingData_.rowPerTailCore;

    // 如果是大shape，一次只算一行，所以不存在尾行
    if (blockIdx < tilingData_.headCoreNum) {
        loopCnt = rowPerHeadCore;
    }
    if (blockIdx >= tilingData_.headCoreNum && blockIdx < tilingData_.coreNum) {
        loopCnt = rowPerTailCore;
    }

    // 尾轴切分，最后剩余的部分
    uint32_t sizeTailLen = (tilingData_.innerLoopTail + FIFTEEN) / SIXTEEN * SIXTEEN;
    rightPadding = sizeTailLen - tilingData_.innerLoopTail;
    loopCntHead = loopCnt / THIRTY_TWO;
    loopCntTail = loopCnt % THIRTY_TWO;
    lenHead = rowPerHeadCore * tilingData_.rowLen;
    lenTail = rowPerTailCore * tilingData_.rowLen;

    if constexpr (IsSameType<yDtype, int4b_t>::value) {
        outAlignLen = (tilingData_.innerLoopEle + SIXTY_THREE) / SIXTY_FOUR * SIXTY_FOUR;
        outLenHead = lenHead >> 1;
        outLenTail = lenTail >> 1;
    } else {
        outAlignLen = tilingData_.innerLoopEle;
        outLenHead = lenHead;
        outLenTail = lenTail;
    }

    if (blockIdx < tilingData_.headCoreNum) {
        inGm.SetGlobalBuffer((__gm__ T*)x + blockIdx * lenHead, lenHead);
        outGm.SetGlobalBuffer((__gm__ yCopyDtype*)y + blockIdx * outLenHead, outLenHead);
    } else {
        inGm.SetGlobalBuffer(
            (__gm__ T*)x + tilingData_.headCoreNum * lenHead + (blockIdx - tilingData_.headCoreNum) * lenTail, lenTail);
        outGm.SetGlobalBuffer(
            (__gm__ yCopyDtype*)y + tilingData_.headCoreNum * outLenHead + (blockIdx - tilingData_.headCoreNum) * outLenTail,
            outLenTail);
    }
    if( blockIdx == 0 )
    {
        scaleGm.SetGlobalBuffer((__gm__ float*)scale,1);
    }

    if constexpr (isSymmertrical == false)
    {
        workspaceTmp1.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workSpace),tilingData_.coreNum);
        workspaceTmp2.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workSpace + tilingData_.coreNum*sizeof(float)),tilingData_.coreNum);
    }
    else
    {
        workspaceTmp1.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workSpace),tilingData_.coreNum);
    }
    // innerLoopEle已经保证了32Bytes对齐
    if constexpr (hasSmooth == 1) {
        smoothGm.SetGlobalBuffer((__gm__ T*)smooth_scales);
        pPipe->InitBuffer(smoothQueue, DOUBLE_BUFFER_NUM, tilingData_.innerLoopEle * sizeof(T));
    }
    if constexpr (isSymmertrical == false) {
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

template <typename T, typename yDtype, int64_t hasSmooth, bool isSymmertrical>
__aicore__ inline void DynamicQuantLargeShapeDbPertensor<T, yDtype, hasSmooth, isSymmertrical>::SetMaxValue() {
    if constexpr (IsSameType<yDtype, int8_t>::value) {
        maxValue = INT8_MAX_VALUE;
        maxValueNoSym= INT8_MAX_VALUE_NO_SYM;
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
        maxValue = HIFLOAT8_MAX_VALUE;
        maxValueNoSym = HIFLOAT8_MAX_VALUE_NO_SYM;
    }
}

template <typename T, typename yDtype, int64_t hasSmooth, bool isSymmertrical>
__aicore__ inline void DynamicQuantLargeShapeDbPertensor<T, yDtype, hasSmooth, isSymmertrical>::ParseTilingData(const DynamicQuantTilingData& tilingData) {
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
}

template <typename T, typename yDtype, int64_t hasSmooth, bool isSymmertrical>
__aicore__ inline void DynamicQuantLargeShapeDbPertensor<T, yDtype, hasSmooth, isSymmertrical>::ProcessScale()
{
    ProcessScaleRow();
    SyncAll();
    ProcessScaleCol();
}

template <typename T, typename yDtype, int64_t hasSmooth, bool isSymmertrical>
__aicore__ inline void DynamicQuantLargeShapeDbPertensor<T, yDtype, hasSmooth, isSymmertrical>::ProcessY()
{
    LocalTensor<float> scaleOutLocal = scaleOutQueue.DeQue<float>();
    __local_mem__ float* scaleOutLocalAddr = (__local_mem__ float*)scaleOutLocal.GetPhyAddr();

    LocalTensor<float> offsetLocal;
    __local_mem__ float* offsetLocalAddr;

    if constexpr (isSymmertrical == false)
    {
        offsetLocal = offsetQueue.DeQue<float>();
        offsetLocalAddr = (__local_mem__ float*)offsetLocal.GetPhyAddr();
    }

    for (uint32_t i = 0; i < loopCntHead; i++) {
        for (uint32_t j = 0; j < THIRTY_TWO; j++) {
            ProcessYRow(i, j,scaleOutLocalAddr,offsetLocalAddr);
        }
    }

    for (uint32_t i = 0; i < loopCntTail; i++) {
        ProcessYRow(loopCntHead, i,scaleOutLocalAddr,offsetLocalAddr);
    }
    if(blockIdx == 0)
    {
        DataCopyParams copyParams{1, (uint16_t)(sizeof(float)), 0, 0};
        DataCopyPad(scaleGm[0], scaleOutLocal, copyParams);
        if constexpr (isSymmertrical == false)
        {
            DataCopyPad(offsetGm[0], offsetLocal, copyParams);
        }
    }
    scaleOutQueue.FreeTensor(scaleOutLocal);
    if constexpr (isSymmertrical == false)
    {
        offsetQueue.FreeTensor(offsetLocal);
    }
}

template <typename T, typename yDtype, int64_t hasSmooth, bool isSymmertrical>
__aicore__ inline void DynamicQuantLargeShapeDbPertensor<T, yDtype, hasSmooth, isSymmertrical>::Process()
{
    if (blockIdx >= tilingData_.coreNum) {
        return;
    }
    ProcessScale();
    ProcessY();
}

template <typename T, typename yDtype, int64_t hasSmooth, bool isSymmertrical>
__aicore__ inline void DynamicQuantLargeShapeDbPertensor<T, yDtype, hasSmooth, isSymmertrical>::ProcessScaleRowLoop(uint32_t i, uint32_t j)
{
    offsetBase = i * THIRTY_TWO + j;
    srcOffset = offsetBase * tilingData_.rowLen;
    for (uint32_t innerLoopIndex = 0; innerLoopIndex < tilingData_.innerLoopTimes; innerLoopIndex++) {
        CopyInByEle(srcOffset, innerLoopIndex, tilingData_.innerLoopEle, 0);
        ComputeMaxRowScale(tilingData_.innerLoopEle);
        srcOffset += tilingData_.innerLoopEle;
    }
    if(tilingData_.innerLoopTail > 0) { // 如果核内切的有尾块
        CopyInByEle(srcOffset, tilingData_.innerLoopTimes, tilingData_.innerLoopTail, rightPadding);
        ComputeMaxRowScale(tilingData_.innerLoopTail);
    }
}

template <typename T, typename yDtype, int64_t hasSmooth, bool isSymmertrical>
__aicore__ inline void DynamicQuantLargeShapeDbPertensor<T, yDtype, hasSmooth, isSymmertrical>::ProcessScaleRow()
{
    LocalTensor<float> MaxToWorkSpaceLocal;
    LocalTensor<float> MinToWorkSpaceLocal;
    LocalTensor<float> scaleToWorkSpaceLocal;

    if constexpr (isSymmertrical == false)
    {
        MaxToWorkSpaceLocal = MaxToWorkSpaceQueue.AllocTensor<float>();
        AscendC::Duplicate(MaxToWorkSpaceLocal, MIN_FLOAT_VALUE, 64, 1, 1, 8);
        MaxToWorkSpaceQueue.EnQue(MaxToWorkSpaceLocal);

        MinToWorkSpaceLocal = MinToWorkSpaceQueue.AllocTensor<float>();
        AscendC::Duplicate(MinToWorkSpaceLocal, MAX_FLOAT_VALUE, 64, 1, 1, 8);
        MinToWorkSpaceQueue.EnQue(MinToWorkSpaceLocal);
    }
    else
    {
        scaleToWorkSpaceLocal = scaleToWorkSpaceQueue.AllocTensor<float>();
        AscendC::Duplicate(scaleToWorkSpaceLocal, (float)0.0, 64, 1, 1, 8);
        scaleToWorkSpaceQueue.EnQue(scaleToWorkSpaceLocal);
    }

    for (uint32_t i = 0; i < loopCntHead; i++) {
        for (uint32_t j = 0; j < THIRTY_TWO; j++) {
            ProcessScaleRowLoop(i, j);
        }
    }

    for (uint32_t i = 0; i < loopCntTail; i++) {
        ProcessScaleRowLoop(loopCntHead, i);
    }
    CopyUB2Workspace(1);
}

template <typename T, typename yDtype, int64_t hasSmooth, bool isSymmertrical>
__aicore__ inline void DynamicQuantLargeShapeDbPertensor<T, yDtype, hasSmooth, isSymmertrical>::ProcessScaleCol()
{
    LocalTensor<float> MaxOutLocal;
    LocalTensor<float> MinOutLocal;
    LocalTensor<float> scaleOutLocal;
    LocalTensor<float> offsetLocal;
    __local_mem__ float* MaxOutLocalAddr;
    __local_mem__ float* MinOutLocalAddr;
    __local_mem__ float* scaleOutLocalAddr;
    __local_mem__ float* offsetLocalAddr;

    if constexpr (isSymmertrical == false)
    {
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
    }
    else
    {
        scaleOutLocal = scaleOutQueue.AllocTensor<float>();
        AscendC::Duplicate(scaleOutLocal, (float)0.0, 64, 1, 1, 8);
        scaleOutLocalAddr = (__local_mem__ float*)scaleOutLocal.GetPhyAddr();
    }
    scaleOffset = 0;
    CopyInScaleByEle(scaleOffset, tilingData_.coreNum);
    ComputeMaxColScale(tilingData_.coreNum,MaxOutLocalAddr,MinOutLocalAddr,scaleOutLocalAddr,offsetLocalAddr);
    if constexpr (isSymmertrical == false)
    {
        MaxOutQueue.FreeTensor(MaxOutLocal);
        MinOutQueue.FreeTensor(MinOutLocal);
        scaleOutQueue.EnQue(scaleOutLocal);
        offsetQueue.EnQue(offsetLocal);
    }
    else
    {
        scaleOutQueue.EnQue(scaleOutLocal);
    }
}

template <typename T, typename yDtype, int64_t hasSmooth, bool isSymmertrical>
__aicore__ inline void DynamicQuantLargeShapeDbPertensor<T, yDtype, hasSmooth, isSymmertrical>::ProcessYRow(uint32_t i, uint32_t j,__local_mem__ float* scaleAddr,__local_mem__ float* offsetAddr)
{
    offsetBase = i * THIRTY_TWO + j;
    srcOffset = offsetBase * tilingData_.rowLen;

    for (uint32_t innerLoopIndex = 0; innerLoopIndex < tilingData_.innerLoopTimes; innerLoopIndex++) {
        CopyInByEle(srcOffset, innerLoopIndex, tilingData_.innerLoopEle, 0);
        ComputeY(tilingData_.innerLoopEle,scaleAddr,offsetAddr);
        CopyOutY(srcOffset, tilingData_.innerLoopEle);

        srcOffset += tilingData_.innerLoopEle;
    }
    if(tilingData_.innerLoopTail > 0) { // 如果核内切的有尾块
        CopyInByEle(srcOffset, tilingData_.innerLoopTimes, tilingData_.innerLoopTail, rightPadding);
        ComputeY(tilingData_.innerLoopTail,scaleAddr,offsetAddr);
        CopyOutY(srcOffset, tilingData_.innerLoopTail);
    }
}

template <typename T, typename yDtype, int64_t hasSmooth, bool isSymmertrical>
__aicore__ inline void DynamicQuantLargeShapeDbPertensor<T, yDtype, hasSmooth, isSymmertrical>::CopyInByEle(int64_t offset, uint32_t loopIndex, uint32_t elementNum, uint8_t rightPadding) {
    DataCopyParams copyParams{1, (uint16_t)(elementNum * sizeof(T)), 0, 0};
    DataCopyPadParams padParams{true, 0, rightPadding, 0};

    LocalTensor<T> inLocal = inQueue.AllocTensor<T>();
    DataCopyPad(inLocal, inGm[offset], copyParams, padParams);
    inQueue.EnQue(inLocal);

    if constexpr (hasSmooth == 1) {
        LocalTensor<T> smoothLocal = smoothQueue.AllocTensor<T>();
        DataCopyPad(smoothLocal, smoothGm[loopIndex * tilingData_.innerLoopEle], copyParams, padParams);
        smoothQueue.EnQue(smoothLocal);
    }
}

template <typename T, typename yDtype, int64_t hasSmooth, bool isSymmertrical>
__aicore__ inline void DynamicQuantLargeShapeDbPertensor<T, yDtype, hasSmooth, isSymmertrical>::CopyInScaleByEle(int64_t offset,uint32_t elementNum) {
    DataCopyParams copyParams{1, (uint16_t)(elementNum * sizeof(float)), 0, 0};
    if constexpr (isSymmertrical == false)
    {
        LocalTensor<float> MaxFromWorkSpaceLocal = MaxFromWorkSpaceQueue.AllocTensor<float>();
        DataCopyPad(MaxFromWorkSpaceLocal, workspaceTmp1[offset], {1, (uint16_t)(elementNum * sizeof(float)), 0, 0}, {false, 0, 0, 0});
        MaxFromWorkSpaceQueue.EnQue(MaxFromWorkSpaceLocal);

        LocalTensor<float> MinFromWorkSpaceLocal = MinFromWorkSpaceQueue.AllocTensor<float>();
        DataCopyPad(MinFromWorkSpaceLocal, workspaceTmp2[offset], {1, (uint16_t)(elementNum * sizeof(float)), 0, 0}, {false, 0, 0, 0});
        MinFromWorkSpaceQueue.EnQue(MinFromWorkSpaceLocal);
    }
    else
    {
        LocalTensor<float> scaleFromWorkSpaceLocal = scaleFromWorkSpaceQueue.AllocTensor<float>();
        DataCopyPad(scaleFromWorkSpaceLocal, workspaceTmp1[offset], {1, (uint16_t)(elementNum * sizeof(float)), 0, 0}, {false, 0, 0, 0});
        scaleFromWorkSpaceQueue.EnQue(scaleFromWorkSpaceLocal);
    }
}

template <typename T, typename yDtype, int64_t hasSmooth, bool isSymmertrical>
__aicore__ inline void DynamicQuantLargeShapeDbPertensor<T, yDtype, hasSmooth, isSymmertrical>::ComputeMaxRowScale(uint32_t elementNum)
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
    if constexpr (isSymmertrical == false)
    {
        maxToWorkSpaceLocal = MaxToWorkSpaceQueue.DeQue<float>();
        maxToWorkSpaceLocalAddr = (__local_mem__ float*)maxToWorkSpaceLocal.GetPhyAddr();

        minToWorkSpaceLocal = MinToWorkSpaceQueue.DeQue<float>();
        minToWorkSpaceLocalAddr = (__local_mem__ float*)minToWorkSpaceLocal.GetPhyAddr();
    }
    else
    {
        scaleToWorkSpaceLocal = scaleToWorkSpaceQueue.DeQue<float>();
        scaleToWorkSpaceLocalAddr = (__local_mem__ float*)scaleToWorkSpaceLocal.GetPhyAddr();
    }
    ComputeMaxRowScaleVF(inLocalAddr, smoothLocalAddr, scaleToWorkSpaceLocalAddr,maxToWorkSpaceLocalAddr,minToWorkSpaceLocalAddr,elementNum);
    if constexpr (hasSmooth == 1) {
        smoothQueue.FreeTensor(smoothLocal);
    }
    if constexpr (isSymmertrical == false)
    {
        MaxToWorkSpaceQueue.EnQue(maxToWorkSpaceLocal);
        MinToWorkSpaceQueue.EnQue(minToWorkSpaceLocal);
    }
    else
    {
        scaleToWorkSpaceQueue.EnQue(scaleToWorkSpaceLocal);
    }
    inQueue.FreeTensor(inLocal);
}

template <typename T, typename yDtype, int64_t hasSmooth, bool isSymmertrical>
__aicore__ inline void DynamicQuantLargeShapeDbPertensor<T, yDtype, hasSmooth, isSymmertrical>::ComputeMaxColScale(uint32_t elementNum,__local_mem__ float* maxAddr,__local_mem__ float* minAddr,__local_mem__ float* scaleAddr,__local_mem__ float* offsetAddr)
{
    LocalTensor<float> scaleFromWorkSpaceLocal;
    __local_mem__ float* scaleFromWorkSpaceLocalAddr;

    LocalTensor<float> maxFromWorkSpaceLocal;
    __local_mem__ float* maxFromWorkSpaceLocalAddr;
    LocalTensor<float> minFromWorkSpaceLocal;
    __local_mem__ float* minFromWorkSpaceLocalAddr;

    if constexpr (isSymmertrical == false)
    {
        maxFromWorkSpaceLocal = MaxFromWorkSpaceQueue.DeQue<float>();
        maxFromWorkSpaceLocalAddr = (__local_mem__ float*)maxFromWorkSpaceLocal.GetPhyAddr();
        minFromWorkSpaceLocal = MinFromWorkSpaceQueue.DeQue<float>();
        minFromWorkSpaceLocalAddr = (__local_mem__ float*)minFromWorkSpaceLocal.GetPhyAddr();
        ComputeMaxColScaleVF(scaleFromWorkSpaceLocalAddr,scaleAddr,maxFromWorkSpaceLocalAddr,maxAddr,minFromWorkSpaceLocalAddr,minAddr, elementNum);
        ComputeScaleSymVF(maxAddr,minAddr,scaleAddr, 1);
        ComputeOffsetSymVF(maxAddr,scaleAddr,offsetAddr,1);

        MaxFromWorkSpaceQueue.FreeTensor(maxFromWorkSpaceLocal);
        MinFromWorkSpaceQueue.FreeTensor(minFromWorkSpaceLocal);
    }
    else
    {
        scaleFromWorkSpaceLocal = scaleFromWorkSpaceQueue.DeQue<float>();
        scaleFromWorkSpaceLocalAddr = (__local_mem__ float*)scaleFromWorkSpaceLocal.GetPhyAddr();
        ComputeMaxColScaleVF(scaleFromWorkSpaceLocalAddr,scaleAddr,maxFromWorkSpaceLocalAddr,maxAddr,minFromWorkSpaceLocalAddr,minAddr, elementNum);
        scaleFromWorkSpaceQueue.FreeTensor(scaleFromWorkSpaceLocal);
    }
}

template <typename T, typename yDtype, int64_t hasSmooth, bool isSymmertrical>
__aicore__ inline void DynamicQuantLargeShapeDbPertensor<T, yDtype, hasSmooth, isSymmertrical>::ComputeMaxRowScaleVF(__local_mem__ T* inLocalAddr, __local_mem__ T* smoothLocalAddr,
                                                                                          __local_mem__ float* scaleLocalAddr, __local_mem__ float* maxLocalAddr, __local_mem__ float* minLocalAddr, uint32_t elementNum)
{
    uint32_t dtypeSize = sizeof(float);
    uint32_t VL = AscendC::VECTOR_REG_WIDTH / dtypeSize;
    uint16_t vfLoopNum = (elementNum + VL - 1) / VL;
    uint32_t maskNum;

    __VEC_SCOPE__ {
        AscendC::MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> vreg0;
        AscendC::MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vreg1;
        AscendC::MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> vreg2;
        AscendC::MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vreg3;
        AscendC::MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vreg5;
        AscendC::MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vreg6_1;
        AscendC::MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vreg6_2;
        AscendC::MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vreg7_1;
        AscendC::MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vreg7_2;
        AscendC::MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vreg8_1;
        AscendC::MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vreg8_2;

        if constexpr (isSymmertrical == false)
        {
            AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(vreg6_1, maxLocalAddr);
            AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(vreg6_2, minLocalAddr);
        }
        else
        {
            AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(vreg6_1, scaleLocalAddr);
        }
        AscendC::MicroAPI::MaskReg mask;

        for (uint16_t i = 0; i < static_cast<uint16_t>(vfLoopNum - 1); i++)
        {
            maskNum = elementNum - i * VL;
            mask = AscendC::MicroAPI::UpdateMask<float>(maskNum);
            AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vreg0, inLocalAddr + i * VL);
            AscendC::MicroAPI::Cast<float, T, castTrait0>(vreg1, vreg0, mask);
            if constexpr (hasSmooth == 1) {
                AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vreg2, smoothLocalAddr + i * VL);
                AscendC::MicroAPI::Cast<float, T, castTrait0>(vreg3, vreg2, mask);
                AscendC::MicroAPI::Mul(vreg1, vreg1, vreg3, mask);
            }

            if constexpr (isSymmertrical == false)
            {
                AscendC::MicroAPI::Max(vreg6_1, vreg1, vreg6_1, mask);
                AscendC::MicroAPI::Min(vreg6_2, vreg1, vreg6_2, mask);
            }
            else
            {
                AscendC::MicroAPI::Abs(vreg5, vreg1, mask);
                AscendC::MicroAPI::Muls(vreg5, vreg5, float(1.0)/maxValue, mask);
                AscendC::MicroAPI::Max(vreg6_1, vreg5, vreg6_1, mask);
            }
        }

        AscendC::MicroAPI::ReduceMax<float>(vreg7_1, vreg6_1, mask);
        if constexpr (isSymmertrical == false)
        {
            AscendC::MicroAPI::ReduceMin<float>(vreg7_2, vreg6_2, mask);
        }

        for (uint16_t i = vfLoopNum - 1; i < vfLoopNum; i++)
        {
            maskNum = elementNum - i * VL;
            mask = AscendC::MicroAPI::UpdateMask<float>(maskNum);
            AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vreg0, inLocalAddr + i * VL);
            AscendC::MicroAPI::Cast<float, T, castTrait0>(vreg1, vreg0, mask);
            if constexpr (hasSmooth == 1) {
                AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vreg2, smoothLocalAddr + i * VL);
                AscendC::MicroAPI::Cast<float, T, castTrait0>(vreg3, vreg2, mask);
                AscendC::MicroAPI::Mul(vreg1, vreg1, vreg3, mask);
            }
            if constexpr (isSymmertrical == false)
            {
                AscendC::MicroAPI::Max(vreg6_1, vreg1, vreg6_1, mask);
                AscendC::MicroAPI::Min(vreg6_2, vreg1, vreg6_2, mask);
            }
            else
            {
                AscendC::MicroAPI::Abs(vreg5, vreg1, mask);
                AscendC::MicroAPI::Muls(vreg5, vreg5, float(1.0)/maxValue, mask);
                AscendC::MicroAPI::Max(vreg6_1, vreg5, vreg6_1, mask);
            }
            AscendC::MicroAPI::ReduceMax<float>(vreg8_1, vreg6_1, mask);
            if constexpr (isSymmertrical == false)
            {
                AscendC::MicroAPI::ReduceMin<float>(vreg8_2, vreg6_2, mask);
            }

            AscendC::MicroAPI::Max(vreg8_1, vreg7_1, vreg8_1, mask);
            if constexpr (isSymmertrical == false)
            {
                AscendC::MicroAPI::Min(vreg8_2, vreg7_2, vreg8_2, mask);
            }
        }
        if constexpr (isSymmertrical == false)
        {
            AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(maxLocalAddr, vreg8_1, mask);
            AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(minLocalAddr, vreg8_2, mask);
        }
        else
        {
            AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(scaleLocalAddr, vreg8_1, mask);
        }
    }
}

template <typename T, typename yDtype, int64_t hasSmooth, bool isSymmertrical>
__aicore__ inline void DynamicQuantLargeShapeDbPertensor<T, yDtype, hasSmooth, isSymmertrical>::ComputeMaxColScaleVF(__local_mem__ float* scaleLocalAddr,__local_mem__ float* scaleOutLocalAddr,__local_mem__ float* maxLocalAddr,__local_mem__ float* maxOutLocalAddr,__local_mem__ float* minLocalAddr,__local_mem__ float* minOutLocalAddr,uint32_t elementNum)
{
    uint32_t dtypeSize = sizeof(float);
    uint32_t VL = AscendC::VECTOR_REG_WIDTH / dtypeSize;
    uint16_t vfLoopNum = (elementNum + VL - 1) / VL;
    uint32_t maskNum;

    __VEC_SCOPE__ {
        AscendC::MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vreg0_1;
        AscendC::MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vreg0_2;
        AscendC::MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vreg1_1;
        AscendC::MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vreg1_2;
        AscendC::MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vreg2_1;
        AscendC::MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vreg2_2;
        AscendC::MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vreg3_1;
        AscendC::MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vreg3_2;

        if constexpr (isSymmertrical == false)
        {
            AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(vreg1_1, maxOutLocalAddr);
            AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(vreg1_2, minOutLocalAddr);
        }
        else
        {
            AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(vreg1_1, scaleOutLocalAddr);
        }

        AscendC::MicroAPI::MaskReg mask = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();

        for (uint16_t i = 0; i < static_cast<uint16_t>(vfLoopNum - 1); i++)
        {
            maskNum = elementNum - i * VL;
            mask = AscendC::MicroAPI::UpdateMask<float>(maskNum);

            if constexpr (isSymmertrical == false)
            {
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(vreg0_1, maxLocalAddr + i * VL);
                AscendC::MicroAPI::Max(vreg1_1, vreg0_1, vreg1_1, mask);
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(vreg0_2, minLocalAddr + i * VL);
                AscendC::MicroAPI::Min(vreg1_2, vreg0_2, vreg1_2, mask);
            }
            else
            {
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(vreg0_1, scaleLocalAddr + i * VL);
                AscendC::MicroAPI::Max(vreg1_1, vreg0_1, vreg1_1, mask);
            }
        }
        if constexpr (isSymmertrical == false)
        {
            AscendC::MicroAPI::ReduceMax<float>(vreg2_1, vreg1_1, mask);
            AscendC::MicroAPI::ReduceMin<float>(vreg2_2, vreg1_2, mask);
        }
        else
        {
            AscendC::MicroAPI::ReduceMax<float>(vreg2_1, vreg1_1, mask);
        }

        for (uint16_t i = vfLoopNum - 1; i < vfLoopNum; i++)
        {
            maskNum = elementNum - i * VL;
            mask = AscendC::MicroAPI::UpdateMask<float>(maskNum);
            if constexpr (isSymmertrical == false)
            {
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(vreg0_1, maxLocalAddr + i * VL);
                AscendC::MicroAPI::Max(vreg1_1, vreg0_1, vreg1_1, mask);
                AscendC::MicroAPI::ReduceMax<float>(vreg3_1, vreg1_1, mask);
                AscendC::MicroAPI::Max(vreg3_1, vreg2_1, vreg3_1, mask);

                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(vreg0_2, minLocalAddr + i * VL);
                AscendC::MicroAPI::Min(vreg1_2, vreg0_2, vreg1_2, mask);
                AscendC::MicroAPI::ReduceMin<float>(vreg3_2, vreg1_2, mask);
                AscendC::MicroAPI::Min(vreg3_2, vreg2_2, vreg3_2, mask);
            }
            else
            {
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(vreg0_1, scaleLocalAddr + i * VL);
                AscendC::MicroAPI::Max(vreg1_1, vreg0_1, vreg1_1, mask);
                AscendC::MicroAPI::ReduceMax<float>(vreg3_1, vreg1_1, mask);
                AscendC::MicroAPI::Max(vreg3_1, vreg2_1, vreg3_1, mask);
            }
        }
        if constexpr (isSymmertrical == false)
        {
            AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(maxOutLocalAddr, vreg3_1, mask);
            AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(minOutLocalAddr, vreg3_2, mask);
        }
        else
        {
            AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(scaleOutLocalAddr, vreg3_1, mask);
        }
    }
}

template <typename T, typename yDtype, int64_t hasSmooth, bool isSymmertrical>
__aicore__ inline void DynamicQuantLargeShapeDbPertensor<T, yDtype, hasSmooth, isSymmertrical>::ComputeY(uint32_t elementNum,__local_mem__ float* scaleAddr,__local_mem__ float* offsetAddr)
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

    ComputeYVF(inLocalAddr, smoothLocalAddr, outAddr, scaleAddr,offsetAddr, elementNum);

    if constexpr (hasSmooth == 1) {
        smoothQueue.FreeTensor(smoothLocal);
    }
    inQueue.FreeTensor(inLocal);
    outQueue.EnQue(outLocal);
}

template <typename T, typename yDtype, int64_t hasSmooth, bool isSymmertrical>
__aicore__ inline void DynamicQuantLargeShapeDbPertensor<T, yDtype, hasSmooth, isSymmertrical>::ComputeScaleSymVF(__local_mem__ float* maxLocalAddr, __local_mem__ float* minLocalAddr, __local_mem__ float* scaleLocalAddr,uint32_t elementNum)
{
    uint32_t dtypeSize = sizeof(float);
    uint32_t VL = AscendC::VECTOR_REG_WIDTH / dtypeSize;
    uint16_t vfLoopNum = (elementNum + VL - 1) / VL;
    uint32_t maskNum;

    __VEC_SCOPE__ {
        AscendC::MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vreg0;
        AscendC::MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vreg1;

        AscendC::MicroAPI::MaskReg mask;

        for (uint16_t i = 0; i < vfLoopNum; i++)
        {
            maskNum = elementNum - i * VL;
            mask = AscendC::MicroAPI::UpdateMask<float>(maskNum);

            AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(vreg0, maxLocalAddr + i * VL);
            AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(vreg1, minLocalAddr + i * VL);
            AscendC::MicroAPI::Sub(vreg1, vreg0, vreg1, mask);
            AscendC::MicroAPI::Muls(vreg1, vreg1, float(1.0)/maxValueNoSym, mask);
            AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(scaleLocalAddr, vreg1, mask);
        }
    }
}

template <typename T, typename yDtype, int64_t hasSmooth, bool isSymmertrical>
__aicore__ inline void DynamicQuantLargeShapeDbPertensor<T, yDtype, hasSmooth, isSymmertrical>::ComputeOffsetSymVF(__local_mem__ float* maxLocalAddr, __local_mem__ float* scaleLocalAddr, __local_mem__ float* offsetLocalAddr,uint32_t elementNum)
{
    uint32_t dtypeSize = sizeof(float);
    uint32_t VL = AscendC::VECTOR_REG_WIDTH / dtypeSize;
    uint16_t vfLoopNum = (elementNum + VL - 1) / VL;
    uint32_t maskNum;

    __VEC_SCOPE__ {
        static constexpr AscendC::MicroAPI::DivSpecificMode mode = {AscendC::MicroAPI::MaskMergeMode::ZEROING, true};
        AscendC::MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vreg0;
        AscendC::MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vreg1;

        AscendC::MicroAPI::MaskReg mask;

        for (uint16_t i = 0; i < vfLoopNum; i++)
        {
            maskNum = elementNum - i * VL;
            mask = AscendC::MicroAPI::UpdateMask<float>(maskNum);

            AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(vreg0, maxLocalAddr + i * VL);
            AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_NORM>(vreg1, scaleLocalAddr + i * VL);
            AscendC::MicroAPI::Div<float,&mode>(vreg1, vreg0, vreg1, mask);
            AscendC::MicroAPI::Muls(vreg1, vreg1, float(-1.0), mask);
            AscendC::MicroAPI::Adds(vreg1, vreg1, maxValue,  mask);

            AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(offsetLocalAddr, vreg1, mask);
        }
    }
}


template <typename T, typename yDtype, int64_t hasSmooth, bool isSymmertrical>
__aicore__ inline void DynamicQuantLargeShapeDbPertensor<T, yDtype, hasSmooth, isSymmertrical>::ComputeYVF(__local_mem__ T* inLocalAddr, __local_mem__ T* smoothLocalAddr,
                                                                                   __local_mem__ yCopyDtype* outAddr, __local_mem__ float* scaleLocalAddr, __local_mem__ float* offsetLocalAddr, uint32_t elementNum)
{
    uint32_t dtypeSize = sizeof(float);
    uint32_t VL = AscendC::VECTOR_REG_WIDTH / dtypeSize;
    uint16_t vfLoopNum = (elementNum + VL - 1) / VL;

    __VEC_SCOPE__ {
        AscendC::MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> vreg0;
        AscendC::MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vreg1;
        AscendC::MicroAPI::RegTensor<T, MicroAPI::RegTraitNumOne> vreg2;
        AscendC::MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vreg3;
        AscendC::MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vreg4;
        AscendC::MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vreg5;
        AscendC::MicroAPI::RegTensor<int16_t, MicroAPI::RegTraitNumOne> vreg6;
        AscendC::MicroAPI::RegTensor<half, MicroAPI::RegTraitNumOne> vreg7;
        AscendC::MicroAPI::RegTensor<yCopyDtype, MicroAPI::RegTraitNumOne> vreg8;
        AscendC::MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vreg9;
        AscendC::MicroAPI::RegTensor<float, MicroAPI::RegTraitNumOne> vreg_offset;

        AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(vreg9, scaleLocalAddr);
        if constexpr (isSymmertrical == false)
        {
            AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(vreg_offset, offsetLocalAddr);
        }
        AscendC::MicroAPI::MaskReg mask = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg mask2 = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::H>();
        for (uint16_t i = 0; i < vfLoopNum; i++)
        {
            auto addr = outAddr + i * VL;
            AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vreg0, inLocalAddr + i * VL);
            AscendC::MicroAPI::Cast<float, T, castTrait0>(vreg1, vreg0, mask);

            if constexpr (hasSmooth == 1) {
                AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vreg2, smoothLocalAddr + i * VL);
                AscendC::MicroAPI::Cast<float, T, castTrait0>(vreg3, vreg2, mask);
                AscendC::MicroAPI::Mul(vreg4, vreg1, vreg3, mask);
                AscendC::MicroAPI::Div(vreg5, vreg4, vreg9, mask);
            } else {
                AscendC::MicroAPI::Div(vreg5, vreg1, vreg9, mask);
            }
            if constexpr (isSymmertrical == false)
            {
                AscendC::MicroAPI::Add(vreg5, vreg5, vreg_offset, mask);
            }

            if constexpr (IsSameType<yDtype, int8_t>::value) {
                AscendC::MicroAPI::Cast<int16_t, float, castTraitF32toI16>(vreg6, vreg5, mask);
                AscendC::MicroAPI::Cast<half, int16_t, castTraitI16toF16>(vreg7, vreg6, mask);
                AscendC::MicroAPI::Cast<yDtype, half, castTraitF16toI8>(vreg8, vreg7, mask);
            } else if constexpr (IsSameType<yDtype, int4b_t>::value) {
                AscendC::MicroAPI::RegTensor<uint16_t, MicroAPI::RegTraitNumOne> vreg20;
                AscendC::MicroAPI::Cast<int16_t, float, castTraitF32toI16>(vreg6, vreg5, mask);
                AscendC::MicroAPI::Cast<half, int16_t, castTraitI16toF16>(vreg7, vreg6, mask);
                AscendC::MicroAPI::Pack(vreg20, (AscendC::MicroAPI::RegTensor<uint32_t>&)vreg7);
                AscendC::MicroAPI::Cast<int4x2_t, half, castTraitF16toI8>(
                    (AscendC::MicroAPI::RegTensor<int4x2_t>&)vreg8,
                    (AscendC::MicroAPI::RegTensor<half>&)vreg20, mask);
                addr = outAddr + (i * VL) / 2;
            } else if constexpr (IsSameType<yDtype, hifloat8_t>::value) {
                AscendC::MicroAPI::Cast<yDtype, float, castTrait32toh8>(vreg8, vreg5, mask);
            } else if constexpr (IsSameType<yDtype, fp8_e4m3fn_t>::value || IsSameType<yDtype, fp8_e5m2_t>::value)  {
                AscendC::MicroAPI::Cast<yDtype, float, castTrait32tofp8>(vreg8, vreg5, mask);
            }

            if constexpr (IsSameType<yDtype, int4b_t>::value) {
                AscendC::MicroAPI::DataCopy<yCopyDtype, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(addr, vreg8, mask2);
            } else {
                AscendC::MicroAPI::DataCopy<yCopyDtype, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(addr, vreg8, mask);
            }
        }
    }
}

template <typename T, typename yDtype, int64_t hasSmooth, bool isSymmertrical>
__aicore__ inline void DynamicQuantLargeShapeDbPertensor<T, yDtype, hasSmooth, isSymmertrical>::CopyOutY(int64_t offset, uint32_t element) {
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

template <typename T, typename yDtype, int64_t hasSmooth, bool isSymmertrical>
__aicore__ inline void DynamicQuantLargeShapeDbPertensor<T, yDtype, hasSmooth, isSymmertrical>::CopyUB2Workspace(int64_t size) {
    if constexpr (isSymmertrical == false)
    {
        LocalTensor<float> maxToWorkSpaceLocal = MaxToWorkSpaceQueue.DeQue<float>();
        DataCopyPad(workspaceTmp1[blockIdx],
                  maxToWorkSpaceLocal,
                  {1, (uint16_t)(size * sizeof(float)), 0, 0});
        MaxToWorkSpaceQueue.FreeTensor(maxToWorkSpaceLocal);
        LocalTensor<float> minToWorkSpaceLocal = MinToWorkSpaceQueue.DeQue<float>();
        DataCopyPad(workspaceTmp2[blockIdx],
          minToWorkSpaceLocal,
          {1, (uint16_t)(size * sizeof(float)), 0, 0});
        MinToWorkSpaceQueue.FreeTensor(minToWorkSpaceLocal);
    }
    else
    {
        LocalTensor<float> scaleToWorkSpaceLocal = scaleToWorkSpaceQueue.DeQue<float>();
        DataCopyPad(workspaceTmp1[blockIdx],
                  scaleToWorkSpaceLocal,
                  {1, (uint16_t)(size * sizeof(float)), 0, 0});
        scaleToWorkSpaceQueue.FreeTensor(scaleToWorkSpaceLocal);
    }
}

}  // namespace DynamicQuantNDOpt2
#endif  // DYNAMIC_QUANT_REGBASE_LARGE_SHAPE_DB_H
