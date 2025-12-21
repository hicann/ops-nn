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
 * \file ctc_loss_v3.h
 * \brief
 */
#ifndef CTC_LOSS_V3_H_
#define CTC_LOSS_V3_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "impl/dav_c220/kernel_operator_reg_others_impl.h"

#ifndef INFINITY
#define INFINITY (__builtin_inff())
#endif

namespace CTCLossV3NS {
using namespace AscendC;
constexpr int64_t BLOCK_BYTES = 32;
constexpr int64_t INT64_SIZE = 8;
constexpr int64_t FLOAT_SIZE = 4;
constexpr float GATHER_FLOAT_SIZE = 4.0f;
constexpr int64_t FLOAT_NUM_PER_BLOCK = 8;
constexpr int64_t NUM_PER_REPEAT = 64;
constexpr int64_t HALF_NUM_PER_REPEAT = 32;
constexpr int64_t MASK_NUM_PER_BYTES = 8;
constexpr int64_t DOUBLE_BLOCK_FLOAT = 16;
constexpr int64_t DOUBLE = 2;
constexpr int64_t NUM_THREE = 3;
constexpr int64_t NUM_SEVEN = 7;
constexpr int64_t NUM_EIGHT = 8;
constexpr int64_t LARGE_BATCH = 1500;

template <typename T1, typename T2>
__aicore__ inline T1 CeilAlign(T1 a, T2 b)
{
    return b == 0 ? 0 : (a + b - 1) / b * b;
};

template <typename TProb, typename TIndices>
class CTCLossV3
{
private:
    TPipe* pipe;
    GlobalTensor<TProb> logProbsGm;
    GlobalTensor<TIndices> targetsGm;
    GlobalTensor<TIndices> inputLengthsGm;
    GlobalTensor<TIndices> targetLengthsGm;
    GlobalTensor<TProb> negLogLikelihoodGm;
    GlobalTensor<TProb> logAlphaGm;
    GlobalTensor<float> tempAlphaGm;
    GlobalTensor<int32_t> tempTargetsGm;
    GlobalTensor<int64_t> targetOffsetGm;

    TBuf<TPosition::VECCALC> inTargetLengthsBuf;
    TBuf<TPosition::VECCALC> inInputLengthsBuf;
    TBuf<TPosition::VECCALC> targetOffsetBuf;
    TBuf<TPosition::VECCALC> inTargetsBuf;
    TBuf<TPosition::VECCALC> inTargetsPanBuf;
    TBuf<TPosition::VECCALC> inTargetsCopyBuf;
    TBuf<TPosition::VECCALC> inlogAlphaBuf;
    TBuf<TPosition::VECCALC> inlogProbBlankBuf;
    TBuf<TPosition::VECCALC> inlogProbFirstBuf;
    TBuf<TPosition::VECCALC> outLogAlphaBuf;
    TBuf<TPosition::VECCALC> logAlpha1Buf;
    TBuf<TPosition::VECCALC> logAlpha2Buf;
    TBuf<TPosition::VECCALC> logAlpha3Buf;

    TBuf<TPosition::VECCALC> ifHaveThreeMaskBuf;
    TBuf<TPosition::VECCALC> allNeginfBuf;
    TBuf<TPosition::VECCALC> gtAlpha1MaskBuf;
    TBuf<TPosition::VECCALC> maxAlphaBuf;
    TBuf<TPosition::VECCALC> ifNeginfMaskBuf;
    TBuf<TPosition::VECCALC> allZeroBuf;
    TBuf<TPosition::VECCALC> castFloatTargetBuf;
    TBuf<TPosition::VECCALC> changedTargetsBuf;
    TBuf<TPosition::VECCALC> downMaskBuf;
    TBuf<TPosition::VECCALC> upMaskBuf;
    TBuf<TPosition::VECCALC> castIntTargetBuf;
    TBuf<TPosition::VECCALC> logProbSliceBuf;
    TBuf<TPosition::VECCALC> resPartProbBuf;
    TBuf<TPosition::VECCALC> resProbBuf;
    TBuf<TPosition::VECCALC> intResProbBuf;
    TBuf<TPosition::VECCALC> evenMaskBuf;
    TBuf<TPosition::VECCALC> allOneBuf;
    TBuf<TPosition::VECCALC> expLogBuf;
    TBuf<TPosition::VECCALC> lossBuf;

    int64_t coreNum = 0;
    int64_t curBlockIdx = 0;
    int64_t sliceLength = 0;
    int64_t sliceLengthTail = 0;
    int64_t sliceLengthAlign = 0;
    int64_t probSliceNum = 1;
    int64_t maxInputLength = 0;
    int64_t maxTargetLength = 0;
    int64_t targetsDimLength = 0;
    int64_t targetsDimNum = 1;
    int64_t targetsNum = 0;
    int64_t symbleSet = 0;
    int64_t batchSize = 0;
    int64_t taskPerCore = 0;
    int64_t taskTailCore = 0;
    int64_t BLANK = 0;
    uint64_t targetDsize = 0;
    uint64_t logProbDsize = 0;
    int64_t zeroInfinity = 0;

    int64_t curCoreTaskNum = 0;
    int64_t startTaskId = 0;
    int64_t alphaLength = 0;
    int64_t alphaTailSizeAlign = 0;
    int64_t alphaLengthAlign = 0;
    int64_t alphaLengthBlockAlign = 0;
    int64_t maxTargetLengthBlockAlign = 0;
    int64_t maxTargetLengthAlign = 0;
    int64_t maskNumAlpha = 0;
    int64_t maskNumTarget = 0;
    int64_t doubleSi = 0;
    int64_t doubleSiAlign = 0;
    int64_t doubleSiBlockAlign = 0;
    int64_t siAlign = 0;
    int64_t siBlockAlign = 0;
    int64_t targetNumPerBlock = 0;
    int64_t targetNumPerRepeat = 0;
    int64_t logProbNumPerBlock = 0;
    int64_t logProbNumPerRepeat = 0;

    float logProbBlank = 0.0f;
    float logProbFirstChar = 0.0f;

    DataCopyPadParams padParams = {false, 0, 0, 0};
    DataCopyParams copyParamsOneC = {1, sizeof(TProb), 0, 0};
    DataCopyParams copyParamsOneI = {1, sizeof(TIndices), 0, 0};
    DataCopyPadExtParams<TProb> padParamsProb = {false, 0, 0, 0};
    DataCopyPadExtParams<TIndices> padParamsIndices = {false, 0, 0, 0};

    LocalTensor<TIndices> targetLengthsTensor;
    LocalTensor<TIndices> inputLengthsTensor;
    LocalTensor<int64_t> targetOffsetTensor;
    LocalTensor<int64_t> targetsTensor;
    LocalTensor<int64_t> targetsPanTensor;
    LocalTensor<int32_t> targetsCopyTensor;
    LocalTensor<uint8_t> ifHaveThreeMaskTensor;
    LocalTensor<float> logAlphaTensor;
    LocalTensor<TProb> logProbBlankTensor;
    LocalTensor<TProb> logProbFirstTensor;
    LocalTensor<TProb> outLogAlphaTensor;
    LocalTensor<float> allNeginfTensor;
    LocalTensor<float> logAlpha1Tensor;
    LocalTensor<float> logAlpha2Tensor;
    LocalTensor<float> logAlpha3Tensor;
    LocalTensor<uint8_t> gtAlpha1MaskTensor;
    LocalTensor<float> maxAlphaTensor;
    LocalTensor<uint8_t> ifNeginfMaskTensor;
    LocalTensor<float> allZeroTensor;
    LocalTensor<float> castFloatTargetsTensor;
    LocalTensor<float> changedTargetsTensor;
    LocalTensor<uint8_t> upMaskTensor;
    LocalTensor<uint8_t> downMaskTensor;
    LocalTensor<int32_t> castIntTargetTensor;
    LocalTensor<float> logProbSliceTensor;
    LocalTensor<float> resPartProbTensor;
    LocalTensor<float> resProbTensor;
    LocalTensor<int64_t> intResProbTensor;
    LocalTensor<uint8_t> evenMaskTensor;
    LocalTensor<int64_t> allOneTensor;
    LocalTensor<float> expLogTensor;
    LocalTensor<TProb> lossTensor;

public:
    __aicore__ inline CTCLossV3(){};
    __aicore__ inline void Init(
        const CTCLossV3TilingData* __restrict tilingData, GM_ADDR logProbs, GM_ADDR targets, GM_ADDR inputLengths,
        GM_ADDR targetLengths, GM_ADDR negLogLikelihood, GM_ADDR logAlpha, GM_ADDR workspace, TPipe* inputPipe)
    {
        pipe = inputPipe;
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        coreNum = GetBlockNum();
        curBlockIdx = GetBlockIdx();
        sliceLength = tilingData->sliceLength;
        sliceLengthTail = tilingData->sliceLengthTail;
        probSliceNum = tilingData->probSliceNum;
        maxInputLength = tilingData->timeStep;
        maxTargetLength = tilingData->maxTargetLength;
        targetsDimLength = tilingData->targetsDimLength;
        targetsDimNum = tilingData->targetsDimNum;
        targetsNum = tilingData->targetsNum;
        symbleSet = tilingData->symbleSet;
        batchSize = tilingData->batchSize;
        taskPerCore = tilingData->taskPerCore;
        taskTailCore = tilingData->taskTailCore;
        BLANK = tilingData->blank;
        targetDsize = sizeof(TIndices);
        logProbDsize = sizeof(TProb);
        zeroInfinity = tilingData->zeroInfinity;
        alphaLength = DOUBLE * maxTargetLength + 1;                             // alphaLength = 2 * maxTargetLength + 1
        alphaTailSizeAlign = (alphaLength + NUM_SEVEN) / NUM_EIGHT * NUM_EIGHT; // alphaLength 8对齐
        sliceLengthAlign = CeilAlign(sliceLength, NUM_PER_REPEAT);
        logProbNumPerBlock = BLOCK_BYTES / logProbDsize;
        alphaLengthAlign = CeilAlign(alphaTailSizeAlign, NUM_PER_REPEAT);
        alphaLengthBlockAlign = CeilAlign(alphaTailSizeAlign, logProbNumPerBlock);
        maskNumAlpha = CeilAlign(alphaLengthAlign / MASK_NUM_PER_BYTES, BLOCK_BYTES);
        maxTargetLengthAlign = CeilAlign(maxTargetLength, NUM_PER_REPEAT);
        maxTargetLengthBlockAlign = CeilAlign((maxTargetLength + 1), FLOAT_NUM_PER_BLOCK);
        maskNumTarget = CeilAlign(maxTargetLengthAlign / MASK_NUM_PER_BYTES, BLOCK_BYTES);

        if (curBlockIdx < taskTailCore) {
            curCoreTaskNum = taskPerCore + 1;
            startTaskId = curCoreTaskNum * curBlockIdx;
        } else {
            curCoreTaskNum = taskPerCore;
            startTaskId = curCoreTaskNum * curBlockIdx + taskTailCore;
        }

        logProbsGm.SetGlobalBuffer((__gm__ TProb*)logProbs, symbleSet * batchSize * maxInputLength);
        targetsGm.SetGlobalBuffer((__gm__ TIndices*)targets, targetsNum);
        inputLengthsGm.SetGlobalBuffer((__gm__ TIndices*)inputLengths, batchSize);
        targetLengthsGm.SetGlobalBuffer((__gm__ TIndices*)targetLengths, batchSize);
        negLogLikelihoodGm.SetGlobalBuffer((__gm__ TProb*)negLogLikelihood, batchSize);
        logAlphaGm.SetGlobalBuffer((__gm__ TProb*)logAlpha, batchSize * alphaTailSizeAlign * maxInputLength);
        tempAlphaGm.SetGlobalBuffer((__gm__ float*)workspace, batchSize * (alphaTailSizeAlign + FLOAT_NUM_PER_BLOCK));
        tempTargetsGm.SetGlobalBuffer(
            (__gm__ int32_t*)workspace + batchSize * (alphaTailSizeAlign + FLOAT_NUM_PER_BLOCK),
            (alphaTailSizeAlign + FLOAT_NUM_PER_BLOCK) * DOUBLE * batchSize);
        if (batchSize > LARGE_BATCH) {
            targetOffsetGm.SetGlobalBuffer(
                (__gm__ int64_t*)workspace +
                    (batchSize * (alphaTailSizeAlign + FLOAT_NUM_PER_BLOCK) * NUM_THREE) / DOUBLE,
                batchSize);
        }
        GetTargetOffsetLength(batchSize);
        InitNewTargetsBuffer(batchSize);
        InitTargetLengthBuffer();
        InitAlphaLengthBuffer();
        InitTensor();
    }

    __aicore__ inline void Process()
    {
        if (curCoreTaskNum == 0) {
            return;
        }
        CreateDupliacteTensor();
        for (int32_t taskLoopId = 0; taskLoopId < this->curCoreTaskNum; taskLoopId++) {
            int64_t batchId = startTaskId + taskLoopId;
            CopyInInputLengths(batchId);
            if (batchSize > LARGE_BATCH) {
                DataCopyPad(targetLengthsTensor, targetLengthsGm[batchId], copyParamsOneI, padParams);
                MTE2ToSSync();
            }
            int64_t targetLength =
                batchSize > LARGE_BATCH ? targetLengthsTensor.GetValue(0) : targetLengthsTensor.GetValue(batchId);
            int64_t inputLength = inputLengthsTensor.GetValue(0);
            if (inputLength < 0 || targetLength < 0) {
                continue;
            }
            int64_t batchOffsetProb = batchId * symbleSet;
            int64_t batchOffsetTargets = 0;
            if (targetsDimNum == DOUBLE) {
                batchOffsetTargets = batchId * targetsDimLength;
            } else if (batchSize <= LARGE_BATCH) {
                batchOffsetTargets = targetOffsetTensor.GetValue(batchId);
            } else {
                MTE3ToMTE2Sync();
                DataCopyParams copyParamsOneInt64 = {1, 8, 0, 0};
                DataCopyPad(targetOffsetTensor, targetOffsetGm[batchId], copyParamsOneInt64, padParams);
                MTE2ToSSync();
                batchOffsetTargets = targetOffsetTensor.GetValue(0);
            }
            SToMTE2Sync();
            if (taskLoopId > 0) {
                VToMTE2Sync();
            }
            doubleSi = DOUBLE * targetLength + 1;
            doubleSiAlign = CeilAlign(doubleSi, NUM_PER_REPEAT);
            siAlign = CeilAlign(targetLength, NUM_PER_REPEAT);
            CopyInTargetTensor(targetLength, batchOffsetTargets, batchId);
            for (int32_t t = 0; t < inputLength; t++) {
                Compute(batchOffsetProb, targetLength, inputLength, batchId, t);
            }
            PipeBarrier<PIPE_ALL>();
        }
    }

private:
    __aicore__ inline void CopyInInputLengths(int64_t batchId)
    {
        DataCopyParams copyParamsBatch{1, static_cast<uint16_t>(targetDsize), 0, 0};
        DataCopyPad(inputLengthsTensor, inputLengthsGm[batchId], copyParamsBatch, padParams);
        MTE2ToSSync();
    }

    __aicore__ inline void GetTargetOffsetLength(int64_t batchSize)
    {
        pipe->InitBuffer(inTargetLengthsBuf, CeilAlign(batchSize * targetDsize, BLOCK_BYTES));
        pipe->InitBuffer(targetOffsetBuf, CeilAlign(batchSize * INT64_SIZE, BLOCK_BYTES));
        targetLengthsTensor = inTargetLengthsBuf.template Get<TIndices>();
        targetOffsetTensor = targetOffsetBuf.Get<int64_t>();
        DataCopyExtParams copyParamsBatch = {1, static_cast<uint32_t>(batchSize * targetDsize), 0, 0, 0};
        DataCopyPad(targetLengthsTensor, targetLengthsGm, copyParamsBatch, padParamsIndices);
        MTE2ToSSync();
        targetOffsetTensor.SetValue(0, 0);
        int64_t offsetSum = 0;
        for (int32_t batchId = 0; batchId < batchSize; batchId++) {
            int64_t offsetTmp = targetLengthsTensor.GetValue(batchId);
            offsetSum += offsetTmp;
            if (targetsDimNum == 1) {
                targetOffsetTensor.SetValue(batchId + 1, offsetSum);
            }
        }
    }

    __aicore__ inline void InitNewTargetsBuffer(int64_t batchSize)
    {
        if (batchSize > LARGE_BATCH) {
            DataCopyExtParams copyParamsOffset = {1, static_cast<uint32_t>(batchSize * INT64_SIZE), 0, 0, 0};
            SToMTE3Sync();
            DataCopyPad(targetOffsetGm, targetOffsetTensor, copyParamsOffset);
            pipe->Reset();
            pipe->InitBuffer(inTargetLengthsBuf, BLOCK_BYTES);
            pipe->InitBuffer(targetOffsetBuf, BLOCK_BYTES);
            targetOffsetTensor = targetOffsetBuf.Get<int64_t>();
            targetLengthsTensor = inTargetLengthsBuf.template Get<TIndices>();
        }
    }

    __aicore__ inline void CopyInTargetTensor(int64_t targetLength, int64_t batchOffsetTargets, int64_t batchId)
    {
        DataCopyPadParams padParams = {false, 0, 0, 0};
        int64_t copy_bytes = targetLength * targetDsize;
        DataCopyParams copyParamsSi = {1, static_cast<uint16_t>(copy_bytes), 0, 0};

        if constexpr (std::is_same<TIndices, int32_t>::value) {
            DataCopyPad(
                targetsTensor.ReinterpretCast<int32_t>()[maxTargetLengthBlockAlign], targetsGm[batchOffsetTargets],
                copyParamsSi, padParams);
            MTE2ToSSync();
            Cast(
                targetsTensor, targetsTensor.ReinterpretCast<int32_t>()[maxTargetLengthBlockAlign],
                RoundMode::CAST_NONE, targetLength);
        } else {
            DataCopyPad(targetsTensor, targetsGm[batchOffsetTargets], copyParamsSi, padParams);
            MTE2ToSSync();
        }
        DataCopyParams copyParamsDoubleSiOut = {1, static_cast<uint16_t>((doubleSi - 1) * sizeof(int32_t)), 0, 0};
        DataCopyParams copyParamsDoubleSiOut2 = {
            1, static_cast<uint16_t>((doubleSi - NUM_THREE) * sizeof(int32_t)), 0, 0};
        DataCopyParams copyParamsDoubleSiIn = {1, static_cast<uint16_t>((doubleSi) * sizeof(int32_t)), 0, 0};

        DataCopyPad(
            tempTargetsGm[batchId * alphaTailSizeAlign * DOUBLE + 1], targetsTensor.ReinterpretCast<int32_t>(),
            copyParamsDoubleSiOut);
        DataCopyPad(
            tempTargetsGm[batchId * alphaTailSizeAlign * DOUBLE + alphaTailSizeAlign + NUM_THREE],
            targetsTensor.ReinterpretCast<int32_t>(), copyParamsDoubleSiOut2);
        MTE3ToMTE2Sync();

        DataCopyPad(
            targetsCopyTensor, tempTargetsGm[batchId * alphaTailSizeAlign * DOUBLE + 0], copyParamsDoubleSiIn,
            padParams);
        DataCopyPad(
            targetsPanTensor.ReinterpretCast<int32_t>(),
            tempTargetsGm[batchId * alphaTailSizeAlign * DOUBLE + alphaTailSizeAlign], copyParamsDoubleSiIn, padParams);
        MTE2ToSSync();
        targetsCopyTensor.SetValue(0, 0);
        targetsPanTensor.ReinterpretCast<int32_t>().SetValue(0, 0);
        targetsPanTensor.ReinterpretCast<int32_t>().SetValue(1, 0);
        targetsPanTensor.ReinterpretCast<int32_t>().SetValue(DOUBLE, 0);
        SToVSync();
        Compare(
            ifHaveThreeMaskTensor, targetsCopyTensor, targetsPanTensor.template ReinterpretCast<int32_t>(), CMPMODE::EQ,
            doubleSiAlign);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CreateDupliacteTensor()
    {
        Duplicate<float>(allNeginfTensor, -INFINITY, INT64_SIZE);
        Duplicate<float>(allZeroTensor, 0.0f, INT64_SIZE);
        Duplicate<int32_t>(
            allOneTensor.ReinterpretCast<int32_t>()[CeilAlign(maxTargetLength + 1, BLOCK_BYTES)], 1,
            maxTargetLength + 1);
        PipeBarrier<PIPE_V>();
        Cast(
            allOneTensor, allOneTensor.ReinterpretCast<int32_t>()[CeilAlign(maxTargetLength + 1, BLOCK_BYTES)],
            RoundMode::CAST_NONE, maxTargetLength + 1);
        PipeBarrier<PIPE_V>();
        CompareScalar(
            evenMaskTensor, allOneTensor.ReinterpretCast<int32_t>(), 1, CMPMODE::EQ,
            CeilAlign(maxTargetLength + 1, BLOCK_BYTES) * DOUBLE);
    }

    __aicore__ inline void InitTargetLengthBuffer()
    {
        pipe->InitBuffer(inTargetsBuf, maxTargetLengthAlign * INT64_SIZE);
        pipe->InitBuffer(castFloatTargetBuf, maxTargetLengthAlign * FLOAT_SIZE);
        pipe->InitBuffer(changedTargetsBuf, maxTargetLengthAlign * FLOAT_SIZE);
        pipe->InitBuffer(castIntTargetBuf, maxTargetLengthAlign * FLOAT_SIZE);
        pipe->InitBuffer(resPartProbBuf, maxTargetLengthAlign * FLOAT_SIZE);
        pipe->InitBuffer(resProbBuf, maxTargetLengthAlign * FLOAT_SIZE);

        pipe->InitBuffer(inTargetsPanBuf, CeilAlign(maxTargetLength + 1, NUM_PER_REPEAT) * INT64_SIZE);
        pipe->InitBuffer(inTargetsCopyBuf, CeilAlign(maxTargetLength + 1, NUM_PER_REPEAT) * INT64_SIZE);
        pipe->InitBuffer(intResProbBuf, CeilAlign(maxTargetLength + 1, NUM_PER_REPEAT) * INT64_SIZE);
        pipe->InitBuffer(allOneBuf, CeilAlign(maxTargetLength + 1, HALF_NUM_PER_REPEAT) * INT64_SIZE);
        pipe->InitBuffer(downMaskBuf, maskNumTarget);
        pipe->InitBuffer(upMaskBuf, maskNumTarget);
    }

    __aicore__ inline void InitAlphaLengthBuffer()
    {
        pipe->InitBuffer(inInputLengthsBuf, BLOCK_BYTES);
        pipe->InitBuffer(inlogProbBlankBuf, BLOCK_BYTES);
        pipe->InitBuffer(inlogProbFirstBuf, BLOCK_BYTES);
        pipe->InitBuffer(logAlpha1Buf, alphaLengthAlign * FLOAT_SIZE);
        pipe->InitBuffer(logAlpha2Buf, alphaLengthAlign * FLOAT_SIZE);
        pipe->InitBuffer(logAlpha3Buf, alphaLengthAlign * FLOAT_SIZE);
        pipe->InitBuffer(inlogAlphaBuf, alphaTailSizeAlign * FLOAT_SIZE);
        pipe->InitBuffer(outLogAlphaBuf, alphaTailSizeAlign * sizeof(TProb));
        pipe->InitBuffer(gtAlpha1MaskBuf, maskNumAlpha);
        pipe->InitBuffer(maxAlphaBuf, alphaLengthAlign * FLOAT_SIZE);
        pipe->InitBuffer(allZeroBuf, BLOCK_BYTES);
        pipe->InitBuffer(ifHaveThreeMaskBuf, maskNumAlpha);
        pipe->InitBuffer(allNeginfBuf, BLOCK_BYTES);
        pipe->InitBuffer(ifNeginfMaskBuf, maskNumAlpha);
        pipe->InitBuffer(logProbSliceBuf, sliceLengthAlign * FLOAT_SIZE);
        pipe->InitBuffer(evenMaskBuf, maskNumAlpha);
        pipe->InitBuffer(expLogBuf, NUM_PER_REPEAT);
        pipe->InitBuffer(lossBuf, HALF_NUM_PER_REPEAT);
    }

    __aicore__ inline void SToMTE2Sync()
    {
        event_t eventIDSToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
        SetFlag<HardEvent::S_MTE2>(eventIDSToMTE2);
        WaitFlag<HardEvent::S_MTE2>(eventIDSToMTE2);
    }

    __aicore__ inline void MTE2ToSSync()
    {
        event_t eventIDMTE2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
        WaitFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
    }

    __aicore__ inline void SToMTE3Sync()
    {
        event_t eventIDSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
        WaitFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
    }

    __aicore__ inline void MTE3ToSSync()
    {
        event_t eventIDMTE3ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
        SetFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);
        WaitFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);
    }

    __aicore__ inline void SToVSync()
    {
        event_t eventIDSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventIDSToV);
        WaitFlag<HardEvent::S_V>(eventIDSToV);
    }

    __aicore__ inline void VToSSync()
    {
        event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIDVToS);
        WaitFlag<HardEvent::V_S>(eventIDVToS);
    }

    __aicore__ inline void MTE3ToVSync()
    {
        event_t eventIDMTE3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
        WaitFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
    }

    __aicore__ inline void VToMTE3Sync()
    {
        event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    }
    __aicore__ inline void MTE3ToMTE2Sync()
    {
        event_t eventIDMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
    }

    __aicore__ inline void MTE2ToVSync()
    {
        event_t eventIDMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    }

    __aicore__ inline void VToMTE2Sync()
    {
        event_t eventIDVToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventIDVToMTE2);
        WaitFlag<HardEvent::V_MTE2>(eventIDVToMTE2);
    }

    __aicore__ inline void MTE2ToMTE3Sync()
    {
        event_t eventIDMTE2ToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
        SetFlag<HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);
        WaitFlag<HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);
    }

    __aicore__ inline void InitTensor()
    {
        inputLengthsTensor = inInputLengthsBuf.template Get<TIndices>();
        targetsTensor = inTargetsBuf.Get<int64_t>();
        targetsPanTensor = inTargetsPanBuf.Get<int64_t>();
        targetsCopyTensor = inTargetsCopyBuf.Get<int32_t>();
        logAlphaTensor = inlogAlphaBuf.Get<float>();
        logProbBlankTensor = inlogProbBlankBuf.template Get<TProb>();
        logProbFirstTensor = inlogProbFirstBuf.template Get<TProb>();
        outLogAlphaTensor = outLogAlphaBuf.template Get<TProb>();
        logAlpha1Tensor = logAlpha1Buf.Get<float>();
        logAlpha2Tensor = logAlpha2Buf.Get<float>();
        logAlpha3Tensor = logAlpha3Buf.Get<float>();

        ifHaveThreeMaskTensor = ifHaveThreeMaskBuf.Get<uint8_t>();
        allNeginfTensor = allNeginfBuf.Get<float>();
        gtAlpha1MaskTensor = gtAlpha1MaskBuf.Get<uint8_t>();
        maxAlphaTensor = maxAlphaBuf.Get<float>();
        ifNeginfMaskTensor = ifNeginfMaskBuf.Get<uint8_t>();
        allZeroTensor = allZeroBuf.Get<float>();
        castFloatTargetsTensor = castFloatTargetBuf.Get<float>();
        changedTargetsTensor = changedTargetsBuf.Get<float>();
        downMaskTensor = downMaskBuf.Get<uint8_t>();
        upMaskTensor = upMaskBuf.Get<uint8_t>();
        castIntTargetTensor = castIntTargetBuf.Get<int32_t>();
        logProbSliceTensor = logProbSliceBuf.Get<float>();
        resPartProbTensor = resPartProbBuf.Get<float>();
        resProbTensor = resProbBuf.Get<float>();
        intResProbTensor = intResProbBuf.Get<int64_t>();
        evenMaskTensor = evenMaskBuf.Get<uint8_t>();
        allOneTensor = allOneBuf.Get<int64_t>();
        expLogTensor = expLogBuf.Get<float>();
        lossTensor = lossBuf.Get<TProb>();
    }

    __aicore__ inline void CopyInlogProbBlank(int64_t batchOffsetProb, int64_t t)
    {
        DataCopyPadParams padParams = {false, 0, 0, 0};
        DataCopyParams copyParamsOneC = {1, sizeof(TProb), 0, 0};
        DataCopyPad(
            logProbBlankTensor, logProbsGm[batchOffsetProb + t * symbleSet * batchSize + BLANK], copyParamsOneC,
            padParams);
        MTE2ToSSync();
    }

    __aicore__ inline void CopyInlogProbFirst(int64_t batchOffsetProb, int64_t t, int64_t firstChar)
    {
        DataCopyPadParams padParams = {false, 0, 0, 0};
        DataCopyParams copyParamsOneC = {1, sizeof(TProb), 0, 0};
        DataCopyPad(
            logProbFirstTensor, logProbsGm[batchOffsetProb + t * symbleSet * batchSize + firstChar], copyParamsOneC,
            padParams);
        MTE2ToSSync();
    }

    __aicore__ inline void CopyInAlpha123Tensor(int64_t batchId, int64_t t)
    {
        DataCopyPadParams padParams = {false, 0, 0, 0};
        DataCopyParams copyParamsDoubleSi0 = {1, static_cast<uint16_t>(doubleSi * FLOAT_SIZE), 0, 0};
        DataCopyPad(logAlpha1Tensor, tempAlphaGm[batchId * alphaTailSizeAlign + 0], copyParamsDoubleSi0, padParams);
        DataCopyPad(logAlpha2Tensor, tempAlphaGm[batchId * alphaTailSizeAlign - 1], copyParamsDoubleSi0, padParams);
        DataCopyPad(
            logAlpha3Tensor, tempAlphaGm[batchId * alphaTailSizeAlign - DOUBLE], copyParamsDoubleSi0, padParams);
        MTE2ToSSync();
        logAlpha2Tensor.SetValue(0, -INFINITY);
        logAlpha3Tensor.SetValue(0, -INFINITY);
        logAlpha3Tensor.SetValue(1, -INFINITY);
        SToVSync();
    }

    __aicore__ inline void CopyInlogProbSliceTensor(
        int64_t batchOffsetProb, int64_t t, int64_t sliceId, int64_t sliceLengthCur, int64_t sliceLengthAlign)
    {
        DataCopyExtParams copyParamsProb = {1, static_cast<uint32_t>(sliceLengthCur * sizeof(TProb)), 0, 0, 0};
        DataCopyPadExtParams<TProb> padParamsProb = {false, 0, 0, 0};
        if constexpr (std::is_same<TProb, float>::value) {
            DataCopyPad(
                logProbSliceTensor, logProbsGm[batchOffsetProb + t * symbleSet * batchSize + sliceLength * sliceId],
                copyParamsProb, padParamsProb);
            MTE2ToSSync();
        } else {
            DataCopyPad(
                logProbSliceTensor.ReinterpretCast<TProb>()[sliceLengthAlign],
                logProbsGm[batchOffsetProb + t * symbleSet * batchSize + sliceLength * sliceId], copyParamsProb,
                padParamsProb);
            MTE2ToSSync();
            Cast(
                logProbSliceTensor, logProbSliceTensor.ReinterpretCast<TProb>()[sliceLengthAlign], RoundMode::CAST_NONE,
                sliceLengthCur);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline void CopyOutNegLogLikelihood(int64_t batchId)
    {
        DataCopyParams copyParamsOne = {1, static_cast<uint16_t>(sizeof(TProb)), 0, 0};
        if constexpr (std::is_same<TProb, half>::value) {
            PipeBarrier<PIPE_V>();
            Cast(lossTensor, expLogTensor, RoundMode::CAST_NONE, NUM_EIGHT);
            SToMTE3Sync();
            DataCopyPad(negLogLikelihoodGm[batchId], lossTensor, copyParamsOne);
        } else if constexpr (std::is_same<TProb, bfloat16_t>::value) {
            PipeBarrier<PIPE_V>();
            Cast(lossTensor, expLogTensor, RoundMode::CAST_RINT, NUM_EIGHT);
            SToMTE3Sync();
            DataCopyPad(negLogLikelihoodGm[batchId], lossTensor, copyParamsOne);
        } else if constexpr (std::is_same<TProb, float>::value) {
            DataCopyPad(negLogLikelihoodGm[batchId], expLogTensor, copyParamsOne);
        }
        MTE3ToSSync();
    }

    __aicore__ inline void CopyOutAlphaTensor(int64_t batchId, int64_t t)
    {
        DataCopyParams copyParamsDouble = {1, static_cast<uint16_t>(sizeof(float) * alphaTailSizeAlign), 0, 0};
        DataCopyPad(tempAlphaGm[batchId * alphaTailSizeAlign], logAlphaTensor, copyParamsDouble);
        DataCopyParams copyParamsTProb = {1, static_cast<uint16_t>(sizeof(TProb) * alphaTailSizeAlign), 0, 0};
        if constexpr (std::is_same<TProb, half>::value) {
            PipeBarrier<PIPE_V>();
            Cast(outLogAlphaTensor, logAlphaTensor, RoundMode::CAST_NONE, alphaTailSizeAlign);
            SToMTE3Sync();
            DataCopyPad(
                logAlphaGm[batchId * alphaTailSizeAlign * maxInputLength + t * alphaTailSizeAlign], outLogAlphaTensor,
                copyParamsTProb);
        } else if constexpr (std::is_same<TProb, bfloat16_t>::value) {
            PipeBarrier<PIPE_V>();
            Cast(outLogAlphaTensor, logAlphaTensor, RoundMode::CAST_RINT, alphaTailSizeAlign);
            SToMTE3Sync();
            DataCopyPad(
                logAlphaGm[batchId * alphaTailSizeAlign * maxInputLength + t * alphaTailSizeAlign], outLogAlphaTensor,
                copyParamsTProb);
        } else if constexpr (std::is_same<TProb, float>::value) {
            DataCopyPad(
                logAlphaGm[batchId * alphaTailSizeAlign * maxInputLength + t * alphaTailSizeAlign], logAlphaTensor,
                copyParamsDouble);
        }
        MTE3ToSSync();
    }

    __aicore__ inline void Compute(
        int64_t batchOffsetProb, int64_t targetLength, int64_t inputLength, int64_t batchId, int64_t t)
    {
        VToSSync();
        CopyInlogProbBlank(batchOffsetProb, t);
        if (std::is_same<TProb, bfloat16_t>::value) {
            logProbBlank = ToFloat(logProbBlankTensor.GetValue(0));
        } else {
            logProbBlank = logProbBlankTensor.GetValue(0);
        }
        if (t == 0) {
            int64_t firstChar = targetsTensor.GetValue(0);
            SToMTE2Sync();
            CopyInlogProbFirst(batchOffsetProb, t, firstChar);
            Duplicate<float>(logAlphaTensor, -INFINITY, alphaTailSizeAlign);
            VToSSync();
            if (std::is_same<TProb, bfloat16_t>::value) {
                logProbFirstChar = ToFloat(logProbFirstTensor.GetValue(0));
            } else {
                logProbFirstChar = logProbFirstTensor.GetValue(0);
            }
            if (targetLength > 0) {
                logAlphaTensor.SetValue(1, logProbFirstChar);
            }
            logAlphaTensor.SetValue(0, logProbBlank);
            SToMTE3Sync();
        } else if (t < inputLength) {
            Duplicate<float>(logAlphaTensor, -INFINITY, alphaTailSizeAlign);
            CopyInAlpha123Tensor(batchId, t);
            Compare(gtAlpha1MaskTensor, logAlpha2Tensor, logAlpha1Tensor, CMPMODE::GT, doubleSiAlign);
            PipeBarrier<PIPE_V>();
            Select(
                maxAlphaTensor, gtAlpha1MaskTensor, logAlpha2Tensor, logAlpha1Tensor, SELMODE::VSEL_TENSOR_TENSOR_MODE,
                doubleSiAlign); // Take the maximum value of logAlpha1 and logAlpha2
            BinaryRepeatParams repeatNeginfSelect = {1, 0, 1, 8, 0, 8};
            Select(
                logAlpha3Tensor, ifHaveThreeMaskTensor, allNeginfTensor, logAlpha3Tensor,
                SELMODE::VSEL_TENSOR_TENSOR_MODE, NUM_PER_REPEAT, doubleSiAlign / NUM_PER_REPEAT, repeatNeginfSelect);
            // Correct the position of -inf in logAlpha3
            PipeBarrier<PIPE_V>();
            Compare(gtAlpha1MaskTensor, logAlpha3Tensor, maxAlphaTensor, CMPMODE::GT, doubleSiAlign);
            PipeBarrier<PIPE_V>();
            Select(
                maxAlphaTensor, gtAlpha1MaskTensor, logAlpha3Tensor, maxAlphaTensor, // Correct maxAlphaTensor
                SELMODE::VSEL_TENSOR_TENSOR_MODE, doubleSiAlign);
            PipeBarrier<PIPE_V>();
            CompareScalar(ifNeginfMaskTensor, maxAlphaTensor, -INFINITY, CMPMODE::EQ, doubleSiAlign);
            PipeBarrier<PIPE_V>();
            Select(
                maxAlphaTensor, ifNeginfMaskTensor, allZeroTensor, maxAlphaTensor, SELMODE::VSEL_TENSOR_TENSOR_MODE,
                NUM_PER_REPEAT, doubleSiAlign / NUM_PER_REPEAT, repeatNeginfSelect);

            PipeBarrier<PIPE_V>();
            Sub(logAlpha1Tensor, logAlpha1Tensor, maxAlphaTensor, doubleSi);
            Sub(logAlpha2Tensor, logAlpha2Tensor, maxAlphaTensor, doubleSi);
            Sub(logAlpha3Tensor, logAlpha3Tensor, maxAlphaTensor, doubleSi);
            PipeBarrier<PIPE_V>();
            Exp(logAlpha1Tensor, logAlpha1Tensor, doubleSi);
            Exp(logAlpha2Tensor, logAlpha2Tensor, doubleSi);
            Exp(logAlpha3Tensor, logAlpha3Tensor, doubleSi);
            PipeBarrier<PIPE_V>();
            Add(logAlpha2Tensor, logAlpha1Tensor, logAlpha2Tensor, doubleSi);
            PipeBarrier<PIPE_V>();
            Add(logAlpha3Tensor, logAlpha2Tensor, logAlpha3Tensor, doubleSi);
            PipeBarrier<PIPE_V>();
            Log(logAlphaTensor, logAlpha3Tensor, doubleSi);
            PipeBarrier<PIPE_V>();
            Add(logAlphaTensor, logAlphaTensor, maxAlphaTensor, doubleSi);
            PipeBarrier<PIPE_V>();
            Cast(castFloatTargetsTensor, targetsTensor, RoundMode::CAST_ROUND, targetLength);
            PipeBarrier<PIPE_V>();
            Muls(castFloatTargetsTensor, castFloatTargetsTensor, GATHER_FLOAT_SIZE, targetLength);
            PipeBarrier<PIPE_V>();
            int64_t sliceLengthCur = sliceLength;
            for (int32_t sliceId = 0; sliceId < probSliceNum; sliceId++) {
                float upLimit =
                    sliceId >= (probSliceNum - 1) ?
                        static_cast<float>(symbleSet) * GATHER_FLOAT_SIZE :
                        static_cast<float>(sliceLengthCur * (1L + static_cast<int64_t>(sliceId))) * GATHER_FLOAT_SIZE;
                float downLimit = sliceLengthCur * sliceId * GATHER_FLOAT_SIZE;
                float negDownLimit = -sliceLengthCur * sliceId * GATHER_FLOAT_SIZE;
                if (sliceId >= probSliceNum - 1) {
                    sliceLengthCur = sliceLengthTail;
                }
                PipeBarrier<PIPE_V>();
                CompareScalar(downMaskTensor, castFloatTargetsTensor, downLimit, CMPMODE::GE, siAlign);
                CompareScalar(upMaskTensor, castFloatTargetsTensor, upLimit, CMPMODE::LT, siAlign);
                // Determine the positions of valid data through comparison.
                Adds(changedTargetsTensor, castFloatTargetsTensor, negDownLimit, targetLength);
                // Subtract the threshold from uintTargetsTensor
                PipeBarrier<PIPE_V>();
                Select(
                    changedTargetsTensor, downMaskTensor, changedTargetsTensor, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE,
                    siAlign);
                PipeBarrier<PIPE_V>();
                Select(
                    changedTargetsTensor, upMaskTensor, changedTargetsTensor, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE,
                    siAlign);
                // Correct invalid data
                PipeBarrier<PIPE_V>();
                Cast(castIntTargetTensor, changedTargetsTensor, RoundMode::CAST_ROUND, targetLength);
                PipeBarrier<PIPE_V>();
                LocalTensor<uint32_t> uintTargetsTensor = castIntTargetTensor.ReinterpretCast<uint32_t>();
                CopyInlogProbSliceTensor(batchOffsetProb, t, sliceId, sliceLengthCur, sliceLengthAlign);
                VToMTE2Sync();
                Gather(resPartProbTensor, logProbSliceTensor, uintTargetsTensor, 0, targetLength);
                VToMTE2Sync();
                PipeBarrier<PIPE_V>();
                Select(
                    resPartProbTensor, downMaskTensor, resPartProbTensor, resProbTensor,
                    SELMODE::VSEL_TENSOR_TENSOR_MODE, siAlign);
                PipeBarrier<PIPE_V>();
                Select(
                    resProbTensor, upMaskTensor, resPartProbTensor, resProbTensor, SELMODE::VSEL_TENSOR_TENSOR_MODE,
                    siAlign);
            }
            Cast(intResProbTensor, resProbTensor.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, targetLength);
            PipeBarrier<PIPE_V>();
            Select(
                intResProbTensor.ReinterpretCast<float>(), evenMaskTensor, intResProbTensor.ReinterpretCast<float>(),
                logProbBlank, SELMODE::VSEL_TENSOR_SCALAR_MODE, doubleSiAlign);
            VToMTE3Sync();
            DataCopyParams copyParamsResProbOut = {1, static_cast<uint16_t>(DOUBLE * targetLength * FLOAT_SIZE), 0, 0};
            DataCopyPad(
                tempAlphaGm[1 + batchId * alphaTailSizeAlign], intResProbTensor.ReinterpretCast<float>(),
                copyParamsResProbOut);
            MTE3ToMTE2Sync();
            DataCopyParams copyParamsResProbIn = {
                1, static_cast<uint16_t>((DOUBLE * targetLength + 1) * FLOAT_SIZE), 0, 0};
            DataCopyPad(
                intResProbTensor.ReinterpretCast<float>(), tempAlphaGm[0 + batchId * alphaTailSizeAlign],
                copyParamsResProbIn, padParams);
            MTE2ToSSync();
            intResProbTensor.ReinterpretCast<float>().SetValue(0, logProbBlank);
            SToVSync();
            Add(logAlphaTensor, logAlphaTensor, intResProbTensor.ReinterpretCast<float>(), doubleSi);
            PipeBarrier<PIPE_V>();
            VToMTE3Sync();
        }
        if (t == inputLength - 1) {
            if (targetLength > 0) {
                VToSSync();
                float l1 = logAlphaTensor.GetValue(DOUBLE * targetLength);
                float l2 = targetLength > 0 ? logAlphaTensor.GetValue(DOUBLE * targetLength - 1) : -INFINITY;
                float m = (l1 > l2) ? l1 : l2;
                m = ((m == -INFINITY) ? 0 : m);
                expLogTensor.SetValue(0, l1 - m);
                expLogTensor.SetValue(FLOAT_NUM_PER_BLOCK, l2 - m);
                SToVSync();
                Exp(expLogTensor, expLogTensor, DOUBLE_BLOCK_FLOAT);
                PipeBarrier<PIPE_V>();
                Add(expLogTensor, expLogTensor, expLogTensor[FLOAT_NUM_PER_BLOCK], FLOAT_NUM_PER_BLOCK);
                PipeBarrier<PIPE_V>();
                Log(expLogTensor[FLOAT_NUM_PER_BLOCK], expLogTensor, FLOAT_NUM_PER_BLOCK);
                VToSSync();
                float log_likelihood = expLogTensor.GetValue(FLOAT_NUM_PER_BLOCK) + m;
                expLogTensor.SetValue(0, (zeroInfinity && log_likelihood == -INFINITY) ? 0.0f : -log_likelihood);
            } else if (targetLength == 0) {
                float log_likelihood = logAlphaTensor.GetValue(0);
                expLogTensor.SetValue(0, (zeroInfinity && log_likelihood == -INFINITY) ? 0.0f : -log_likelihood);
            }
            PipeBarrier<PIPE_ALL>();
            CopyOutNegLogLikelihood(batchId);
        }
        CopyOutAlphaTensor(batchId, t);
    }
};
} // namespace CTCLossV3NS
#endif // CTC_LOSS_V3_H_