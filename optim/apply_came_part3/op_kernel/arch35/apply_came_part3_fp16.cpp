/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "apply_came_part3_fp16.h"

using namespace AscendC;

template <typename T>
__aicore__ inline int64_t ApplyCamePart3FP16<T>::Ceil(int64_t a, int64_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b * b;
}

template <typename T>
__aicore__ inline int64_t ApplyCamePart3FP16<T>::DivCeil(int64_t a, int64_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

template <typename T>
__aicore__ inline void ApplyCamePart3FP16<T>::Init(CamePart3InOut camePart3InOut, GM_ADDR workspace,
                                                   const ApplyCamePart3TilingData* __restrict cameTiling)
{
    ParseTilingData(cameTiling);
    InitBuffers(camePart3InOut, workspace);
    InitVars();
    ClearAcculateMatrix();
}

template <typename T>
__aicore__ inline void ApplyCamePart3FP16<T>::ParseTilingData(const ApplyCamePart3TilingData* __restrict tilingData)
{
    const auto& data = *tilingData;
    usedCoreNum = data.usedCoreNum;
    curN = data.curN;
    curM = data.curM;
    rNumCalc = data.rNumCalc;
    cNumCalc = data.cNumCalc;
    baseN = data.baseN;
    baseM = data.baseM;
    rCoreNum = data.rCoreNum;
    cCoreNum = data.cCoreNum;
    isGlobalShape = data.isGlobalShape;
    useFirstMoment = data.useFirstMoment;

    maxMLoop = DivCeil(cNumCalc, baseM);
    maxNLoop = DivCeil(rNumCalc, baseN);
}

template <typename T>
__aicore__ inline void ApplyCamePart3FP16<T>::ClearAcculateMatrix()
{
    constexpr float scalarValue = 0;

    Duplicate(ubLocal2, scalarValue, bufSize);
    Duplicate(ubLocal3, scalarValue, bufSize);
    Duplicate(ubLocal4, scalarValue, bufSize);
}

template <typename T>
__aicore__ inline void ApplyCamePart3FP16<T>::InitBuffers(CamePart3InOut camePart3InOut, GM_ADDR workspace)
{
    CalcGMOffset();
    bufSize = Ceil(baseM, FP16_ONE_BLOCK_COUNT) * Ceil(baseN, FP16_ONE_BLOCK_COUNT);
    tailBlockStride = Ceil(baseM, FP32_ONE_BLOCK_COUNT) * sizeof(float) / CAME_ONE_BLOCK_SIZE;
    rowBlockStride = tailBlockStride < REP_BLOCK_STRIDE ? tailBlockStride : REP_BLOCK_STRIDE;
    InitInBuffers(camePart3InOut);
    InitOutBuffers(camePart3InOut, workspace);
    // Init Local Tensors
    pipe.InitBuffer(inQueue, 1, bufSize * sizeof(float));
    pipe.InitBuffer(outQueue, 1, bufSize * sizeof(float));

    pipe.InitBuffer(calcBuf, BUFFER_SIZE * bufSize * sizeof(float));
    ubLocal2 = calcBuf.Get<float>(BUFFER_SIZE * bufSize);
    ubLocal3 = ubLocal2[bufSize];
    ubLocal4 = ubLocal3[bufSize];

    pipe.InitBuffer(detBuf, DET_WORKSPACE_BYTE);
    ubDetWorkspace = detBuf.Get<int32_t>(DET_WORKSPACE_SIZE);

#if defined(DETERMINISTIC_MODE) && DETERMINISTIC_MODE == 1
    InitDetermineComputeWorkspace(gmDetWorkspace, ubDetWorkspace);
#endif
}

template <typename T>
__aicore__ inline void ApplyCamePart3FP16<T>::InitInBuffers(CamePart3InOut camePart3InOut)
{
    GM_ADDR u = camePart3InOut.u;
    GM_ADDR mIn = camePart3InOut.mIn;
    GM_ADDR eps = camePart3InOut.eps;
    GM_ADDR beta1 = camePart3InOut.beta1;
    GM_ADDR clipThreshold = camePart3InOut.clipThreshold;
    GM_ADDR sumSquareU = camePart3InOut.sumSquareU;
    GM_ADDR globalShape = camePart3InOut.globalShape;

    uGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(u + blockOffset * sizeof(float)), curN * curM);

    mInputGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(mIn + blockOffset * sizeof(T)), curN * curM);

    epsGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(eps), SCALAR_INPUT_SIZE);
    beta1Gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(beta1), SCALAR_INPUT_SIZE);
    clipThresholdGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(clipThreshold), SCALAR_INPUT_SIZE);
    sumSquareUGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sumSquareU), SCALAR_INPUT_SIZE);
    globalShapeGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(globalShape), SCALAR_INPUT_SIZE);
}

template <typename T>
__aicore__ inline void ApplyCamePart3FP16<T>::InitOutBuffers(CamePart3InOut camePart3InOut, GM_ADDR workspace)
{
    GM_ADDR mOut = camePart3InOut.mOut;
    GM_ADDR sumUR = camePart3InOut.sumUR;
    GM_ADDR sumUC = camePart3InOut.sumUC;
    GM_ADDR sumURC = camePart3InOut.sumURC;

    sumURGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sumUR + nOffset * sizeof(float)), curN);
    sumUCGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sumUC + mOffset * sizeof(float)), curM);
    sumURCGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sumURC), 1 * sizeof(float));
    if (useFirstMoment == 1) {
        mOutputGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(mOut + blockOffset * sizeof(T)), curN * curM);
    }

    workspaceSumGradRC_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace + DET_WORKSPACE_BYTE));
    workspaceSumGradC_.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(workspace + workspaceRCSize * sizeof(float) + DET_WORKSPACE_BYTE));
    gmDetWorkspace.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspace));

    float initValue = 0.0;
    if (GetBlockIdx() == 0) {
        InitOutput<float>(sumURGm, curN, initValue);
        InitOutput<float>(sumUCGm, curM, initValue);
        InitOutput<float>(sumURCGm, 1, initValue);
    }
    SyncAll();
}

template <typename T>
__aicore__ inline void ApplyCamePart3FP16<T>::InitVars()
{
    mLoop = DivCeil(cNumCalc, baseM);
    nLoop = DivCeil(rNumCalc, baseN);
}

template <typename T>
__aicore__ inline void ApplyCamePart3FP16<T>::CalcGMOffset()
{
    auto temp0 = DivCeil(curM, cNumCalc);
    if (temp0 == 0) {
        temp0 = 1;
    }

    auto nCoreIndx = GetBlockIdx() / temp0;
    auto mCoreIndx = GetBlockIdx() % temp0;

    // workspace gm offset
    int64_t cTailNumCalc = curM - cNumCalc * (cCoreNum - 1);
    int64_t cOneCoreBaseNum = DivCeil(cNumCalc, baseM);
    int64_t cTailCoreBaseNum = DivCeil(cTailNumCalc, baseM);
    int64_t rOneCoreBaseNum = DivCeil(rNumCalc, baseN);

    tilingBaseM = baseM;
    cBlockNum = cOneCoreBaseNum * (cCoreNum - 1) + cTailCoreBaseNum;
    // workspace 中RC的空间, 尾核也按照整核算，大于实际的值
    workspaceRCSize = cOneCoreBaseNum * rCoreNum * rOneCoreBaseNum * cCoreNum;
    // workspace中RC和C的核偏移
    workspaceRCOffset = nCoreIndx * rOneCoreBaseNum * cBlockNum + mCoreIndx * cOneCoreBaseNum;
    workspaceCOffset = nCoreIndx * rOneCoreBaseNum * curM + mCoreIndx * cNumCalc;

    // gm nd format
    mOffset = mCoreIndx * cNumCalc;
    oriMOffset = mOffset;
    nOffset = nCoreIndx * rNumCalc;
    oriNOffset = nOffset;
    blockOffset = nCoreIndx * rNumCalc * curM + mCoreIndx * cNumCalc;

    uint64_t gmUserM = curM - mCoreIndx * cNumCalc;
    cNumCalc = gmUserM < cNumCalc ? gmUserM : cNumCalc;
    uint64_t gmUserN = curN - nCoreIndx * rNumCalc;
    rNumCalc = gmUserN < rNumCalc ? gmUserN : rNumCalc;

    baseM = baseM < cNumCalc ? baseM : cNumCalc;
    baseN = baseN < rNumCalc ? baseN : rNumCalc;
    oriBaseM = baseM;
    oriBaseN = baseN;
}

template <typename T>
__aicore__ inline void ApplyCamePart3FP16<T>::Process()
{
    if (GetBlockIdx() < usedCoreNum) {
        ProcessNormal();
        ApplyCamePart3WaitMte3ToS();
        SyncAll();
    }
}

template <typename T>
__aicore__ inline void ApplyCamePart3FP16<T>::CalcOneOffset(int64_t mIdx, int64_t nIdx)
{
    mOffset = mIdx * oriBaseM;
    nOffset = nIdx * oriBaseN;
    outMOffset = nIdx * oriBaseN * curM + mIdx * oriBaseM;
    if (cNumCalc % oriBaseM && mIdx == mLoop - 1) {
        baseM = cNumCalc % oriBaseM;
    } else {
        baseM = oriBaseM;
    }
    tailBlockStride = Ceil(baseM, FP32_ONE_BLOCK_COUNT) * sizeof(float) / CAME_ONE_BLOCK_SIZE;
    rowBlockStride = tailBlockStride < REP_BLOCK_STRIDE ? tailBlockStride : REP_BLOCK_STRIDE;
    if (rNumCalc % oriBaseN && nIdx == nLoop - 1) {
        baseN = rNumCalc % oriBaseN;
    } else {
        baseN = oriBaseN;
    }

    // workspace offset of C/RC
    workspaceRCOffsetBase = workspaceRCOffset + nIdx * cBlockNum + mIdx;
    workspaceCOffsetBase = workspaceCOffset + nIdx * curM + mIdx * tilingBaseM;
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline void ApplyCamePart3FP16<T>::ProcessNormal()
{
    CalcScalar();
    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
    for (int64_t nIdx = 0; nIdx < nLoop; nIdx++) {
        for (int64_t mIdx = 0; mIdx < mLoop; mIdx++) {
            ProcessOneLoop(mIdx, nIdx);
        }
    }

#if defined(DETERMINISTIC_MODE) && DETERMINISTIC_MODE == 1
    int64_t detLoop = maxMLoop * maxNLoop - nLoop * mLoop;
    for (int64_t detIdx = 0; detIdx < detLoop; detIdx++) {
        WaitPreBlock(gmDetWorkspace, ubDetWorkspace);
        NotifyNextBlock(gmDetWorkspace, ubDetWorkspace);
    }
#endif
}

template <typename T>
__aicore__ inline void ApplyCamePart3FP16<T>::CopyInU(LocalTensor<float>& ubLocal, GlobalTensor<float>& tensorGm)
{
    LocalTensor<float> ubLocalIn = inQueue.AllocTensor<float>();
    if (baseM % FP32_ONE_BLOCK_COUNT || baseN % FP32_ONE_BLOCK_COUNT) {
        Duplicate<float>(ubLocalIn, 0.0, bufSize);
        event_t eventIdVToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
    }

    if (baseM % FP32_ONE_BLOCK_COUNT) {
        DataCopyParams copyParamsLast;
        copyParamsLast.blockCount = baseN;
        copyParamsLast.blockLen = baseM * sizeof(float);
        copyParamsLast.srcStride = (curM - baseM) * sizeof(float);
        copyParamsLast.dstStride = 0;

        DataCopyPadParams padParamsLast;
        padParamsLast.isPad = true;
        padParamsLast.leftPadding = 0;
        padParamsLast.paddingValue = 0;
        padParamsLast.rightPadding = (baseM / FP32_ONE_BLOCK_COUNT + 1) * FP32_ONE_BLOCK_COUNT - baseM;

        DataCopyPad(ubLocalIn, tensorGm[outMOffset], copyParamsLast, padParamsLast);
    } else {
        DataCopyParams intriParams;
        intriParams.blockCount = baseN;
        intriParams.blockLen = baseM * sizeof(float);
        intriParams.srcStride = (curM - baseM) * sizeof(float);
        intriParams.dstStride = 0;
        DataCopyPadParams padParamsNormal{false, 0, 0, 0};
        DataCopyPad(ubLocalIn, tensorGm[outMOffset], intriParams, padParamsNormal);
    }

    inQueue.EnQue(ubLocalIn);
    ubLocalIn = inQueue.DeQue<float>();
    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    Muls(ubLocal, ubLocalIn, static_cast<float>(1.0), bufSize);
    PipeBarrier<PIPE_V>();
    inQueue.FreeTensor(ubLocalIn);
}

template <typename T>
__aicore__ inline void ApplyCamePart3FP16<T>::CopyInM(LocalTensor<float>& ubLocal, GlobalTensor<T>& tensorGm)
{
    LocalTensor<T> ubLocalIn = inQueue.AllocTensor<T>();
    if (baseM % FP16_ONE_BLOCK_COUNT || baseN % FP16_ONE_BLOCK_COUNT) {
        Duplicate<T>(ubLocalIn, 0.0, bufSize);
        event_t eventIdVToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
    }
    if (baseM % FP16_ONE_BLOCK_COUNT) {
        DataCopyParams copyParamsLast;
        copyParamsLast.blockCount = baseN;
        copyParamsLast.blockLen = baseM * sizeof(T);
        copyParamsLast.srcStride = (curM - baseM) * sizeof(T);
        copyParamsLast.dstStride = 0;

        DataCopyPadParams padParamsLast;
        padParamsLast.isPad = true;
        padParamsLast.leftPadding = 0;
        padParamsLast.paddingValue = 0;
        padParamsLast.rightPadding = (baseM / FP16_ONE_BLOCK_COUNT + 1) * FP16_ONE_BLOCK_COUNT - baseM;

        DataCopyPad(ubLocalIn, tensorGm[outMOffset], copyParamsLast, padParamsLast);
    } else {
        DataCopyParams intriParams;
        intriParams.blockCount = baseN;
        intriParams.blockLen = baseM * sizeof(T);
        intriParams.srcStride = (curM - baseM) * sizeof(T);
        intriParams.dstStride = 0;
        DataCopyPadParams padParamsNormal{false, 0, 0, 0};
        DataCopyPad(ubLocalIn, tensorGm[outMOffset], intriParams, padParamsNormal);
    }

    inQueue.EnQue(ubLocalIn);
    ubLocalIn = inQueue.DeQue<T>();
    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
    CastIn(ubLocal, ubLocalIn);
    inQueue.FreeTensor(ubLocalIn);
}

template <typename T>
__aicore__ inline void ApplyCamePart3FP16<T>::CastIn(LocalTensor<float>& ubLocal, LocalTensor<T>& ubLocalIn)
{
    if (baseM % FP16_ONE_BLOCK_COUNT && baseM < ONE_VECTOR_FP32_SIZE) {
        int64_t curRepeatTimes = baseN;
        int64_t overMaxRepeat = 0;
        int64_t totalRepeatTimes = baseN;
        if (curRepeatTimes > MAX_REPEAT_TIME) {
            curRepeatTimes = MAX_REPEAT_TIME;
            overMaxRepeat = 1;
        }
        uint8_t srcRepStride = Ceil(baseM, FP16_ONE_BLOCK_COUNT) * sizeof(T) / CAME_ONE_BLOCK_SIZE;
        uint64_t mask = Ceil(baseM, FP32_ONE_BLOCK_COUNT);
        Cast(ubLocal, ubLocalIn, RoundMode::CAST_NONE, mask, curRepeatTimes,
             {1, 1, static_cast<uint8_t>(rowBlockStride), srcRepStride});
        int64_t fp32Offset = curRepeatTimes * mask;
        int64_t fp16Offset = curRepeatTimes * Ceil(baseM, FP16_ONE_BLOCK_COUNT);
        totalRepeatTimes -= curRepeatTimes;
        curRepeatTimes = totalRepeatTimes > MAX_REPEAT_TIME ? MAX_REPEAT_TIME : totalRepeatTimes;
        while (overMaxRepeat && curRepeatTimes > 0) {
            Cast(ubLocal[fp32Offset], ubLocalIn[fp16Offset], RoundMode::CAST_NONE, mask, curRepeatTimes,
                 {1, 1, static_cast<uint8_t>(rowBlockStride), srcRepStride});
            fp32Offset += curRepeatTimes * mask;
            fp16Offset += curRepeatTimes * Ceil(baseM, FP16_ONE_BLOCK_COUNT);
            totalRepeatTimes -= curRepeatTimes;
            curRepeatTimes = totalRepeatTimes > MAX_REPEAT_TIME ? MAX_REPEAT_TIME : totalRepeatTimes;
        }
    } else if (baseM % FP16_ONE_BLOCK_COUNT && baseM > ONE_VECTOR_FP32_SIZE) {
        uint64_t repeatTimes = baseN;
        int64_t tLoop = baseM / ONE_VECTOR_FP32_SIZE;
        uint8_t srcRepStride = Ceil(baseM, FP16_ONE_BLOCK_COUNT) * sizeof(T) / CAME_ONE_BLOCK_SIZE;
        for (int64_t tIdx = 0; tIdx < tLoop; tIdx++) {
            int64_t offset = tIdx * ONE_VECTOR_FP32_SIZE;
            Cast(ubLocal[offset], ubLocalIn[offset], RoundMode::CAST_NONE, ONE_VECTOR_FP32_SIZE, repeatTimes,
                 {1, 1, static_cast<uint8_t>(tailBlockStride), srcRepStride});
        }
        uint64_t mask = baseM % ONE_VECTOR_FP32_SIZE;
        int64_t tailBlockOffset = baseM / ONE_VECTOR_FP32_SIZE * ONE_VECTOR_FP32_SIZE;
        Cast(ubLocal[tailBlockOffset], ubLocalIn[tailBlockOffset], RoundMode::CAST_NONE, mask, repeatTimes,
             {1, 1, static_cast<uint8_t>(tailBlockStride), srcRepStride});
    } else {
        Cast(ubLocal, ubLocalIn, RoundMode::CAST_NONE, bufSize);
    }
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void ApplyCamePart3FP16<T>::CalcOutM(LocalTensor<float>& ubLocal2, LocalTensor<float>& ubLocal3,
                                                       LocalTensor<float>& ubLocal4)
{
    float max1 = static_cast<float>(1.0) / maxValue;
    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
    Muls(ubLocal2, ubLocal2, max1, bufSize);
    PipeBarrier<PIPE_V>();
    Muls(ubLocal4, ubLocal2, beta2, bufSize);
    Muls(ubLocal3, ubLocal3, beta1, bufSize);
    PipeBarrier<PIPE_V>();
    Add(ubLocal3, ubLocal4, ubLocal3, bufSize);
    PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void ApplyCamePart3FP16<T>::CalcSumUCTailBlock(LocalTensor<float>& ubLocal4,
                                                                 LocalTensor<float>& ubLocal3, int64_t rowNum,
                                                                 int64_t calcSize)
{
    uint64_t tailMask = baseM % ONE_VECTOR_FP32_SIZE;
    uint64_t lastOffset = baseM / ONE_VECTOR_FP32_SIZE * ONE_VECTOR_FP32_SIZE;
    int64_t curRepeatTimes = rowNum;
    int64_t overMaxRepeat = 0;
    int64_t totalRepeatTimes = rowNum;
    if (curRepeatTimes > MAX_REPEAT_TIME) {
        curRepeatTimes = MAX_REPEAT_TIME;
        overMaxRepeat = 1;
    }
    Add(ubLocal4[lastOffset], ubLocal3[lastOffset], ubLocal4[calcSize + lastOffset], tailMask, curRepeatTimes,
        {1, 1, 1, static_cast<uint8_t>(tailBlockStride), static_cast<uint8_t>(tailBlockStride),
         static_cast<uint8_t>(tailBlockStride)});

    int64_t overOffset = lastOffset + tailBlockStride * FP32_ONE_BLOCK_COUNT * curRepeatTimes;
    totalRepeatTimes -= curRepeatTimes;
    curRepeatTimes = totalRepeatTimes > MAX_REPEAT_TIME ? MAX_REPEAT_TIME : totalRepeatTimes;
    while (overMaxRepeat && curRepeatTimes > 0) {
        Add(ubLocal4[overOffset], ubLocal3[overOffset], ubLocal4[calcSize + overOffset], tailMask, curRepeatTimes,
            {1, 1, 1, static_cast<uint8_t>(tailBlockStride), static_cast<uint8_t>(tailBlockStride),
             static_cast<uint8_t>(tailBlockStride)});
        overOffset += tailBlockStride * FP32_ONE_BLOCK_COUNT * curRepeatTimes;
        totalRepeatTimes -= curRepeatTimes;
        curRepeatTimes = totalRepeatTimes > MAX_REPEAT_TIME ? MAX_REPEAT_TIME : totalRepeatTimes;
    }
}

template <typename T>
__aicore__ inline void ApplyCamePart3FP16<T>::CalcSumUC(LocalTensor<float>& ubLocal2, LocalTensor<float>& ubLocal4)
{
    ApplyCamePart3CalcSumUC(ubLocal2, ubLocal3, ubLocal4, baseM, baseN, bufSize, tailBlockStride);
}

template <typename T>
__aicore__ inline void ApplyCamePart3FP16<T>::CalcSumURReduce(LocalTensor<float>& ubLocal3,
                                                              LocalTensor<float>& ubLocal4, uint8_t repStride)
{
    int64_t repeatTimes = baseN;
    int64_t overMaxRepeat = 0;
    int64_t totalRepeatTimes = baseN;
    if (repeatTimes > MAX_REPEAT_TIME) {
        repeatTimes = MAX_REPEAT_TIME;
        overMaxRepeat = 1;
    }

    uint64_t mask = baseM < ONE_VECTOR_FP32_SIZE ? Ceil(baseM, FP32_ONE_BLOCK_COUNT) : ONE_VECTOR_FP32_SIZE;
    uint64_t realmask = baseM < ONE_VECTOR_FP32_SIZE ? baseM : ONE_VECTOR_FP32_SIZE;
    WholeReduceSum<float>(ubLocal4, ubLocal3, realmask, repeatTimes, 1, 1, repStride);

    int64_t offset = repeatTimes;
    totalRepeatTimes -= repeatTimes;
    repeatTimes = totalRepeatTimes > MAX_REPEAT_TIME ? MAX_REPEAT_TIME : totalRepeatTimes;
    while (overMaxRepeat && repeatTimes > 0) {
        WholeReduceSum<float>(ubLocal4[offset], ubLocal3[mask * offset], realmask, repeatTimes, 1, 1, repStride);
        offset += repeatTimes;
        totalRepeatTimes -= repeatTimes;
        repeatTimes = totalRepeatTimes > MAX_REPEAT_TIME ? MAX_REPEAT_TIME : totalRepeatTimes;
    }
    PipeBarrier<PIPE_V>();
    return;
}

template <typename T>
__aicore__ inline void ApplyCamePart3FP16<T>::CalcSumURAddBlock(LocalTensor<float>& ubLocal2, int64_t repeatTimes)
{
    int64_t vectorNum = baseM / ONE_VECTOR_FP32_SIZE;
    int64_t loopSize = 1;
    while (loopSize * SPLIT_PART <= vectorNum) {
        loopSize = loopSize * SPLIT_PART;
    }
    int64_t vectorSize = loopSize * ONE_VECTOR_FP32_SIZE;
    int64_t vectorTail = Ceil(baseM - vectorSize, FP32_ONE_BLOCK_COUNT);

    for (int64_t idx = 0; idx < repeatTimes; ++idx) {
        int64_t offset = idx * Ceil(baseM, FP32_ONE_BLOCK_COUNT);
        if (vectorTail) {
            Add(ubLocal2[offset], ubLocal2[offset + vectorSize], ubLocal2[offset], vectorTail);
            PipeBarrier<PIPE_V>();
        }

        for (int64_t j = 1; j < loopSize; j *= SPLIT_PART) {
            Add(ubLocal2[offset], ubLocal2[offset + vectorSize / SPLIT_PART / j], ubLocal2[offset],
                vectorSize / SPLIT_PART / j);
            PipeBarrier<PIPE_V>();
        }
    }
}

template <typename T>
__aicore__ inline void ApplyCamePart3FP16<T>::CalcSumUR(LocalTensor<float>& ubLocal2, LocalTensor<float>& ubLocal3,
                                                        LocalTensor<float>& ubLocal4)
{
    int64_t tLoop = DivCeil(baseM, ONE_VECTOR_FP32_SIZE) - 2;
    if (baseM % ONE_VECTOR_FP32_SIZE) {
        tLoop -= 1;
    }
    uint64_t mask = baseM < ONE_VECTOR_FP32_SIZE ? Ceil(baseM, FP32_ONE_BLOCK_COUNT) : ONE_VECTOR_FP32_SIZE;
    int64_t repeatTimes = baseN;
    int64_t overMaxRepeat = 0;
    if (repeatTimes > MAX_REPEAT_TIME) {
        repeatTimes = MAX_REPEAT_TIME;
        overMaxRepeat = 1;
    }
    int64_t loopOffset = ONE_VECTOR_FP32_SIZE;
    uint8_t src0RepStride = Ceil(baseM, FP32_ONE_BLOCK_COUNT) / (CAME_ONE_BLOCK_SIZE / sizeof(float));
    if (baseM == 1) {
        Muls(ubLocal3, ubLocal2, static_cast<float>(1.0), bufSize);
        PipeBarrier<PIPE_V>();
        return CalcSumURReduce(ubLocal3, ubLocal4, static_cast<uint8_t>(rowBlockStride));
    }

    PipeBarrier<PIPE_V>();
    if (baseM > ONE_VECTOR_FP32_SIZE) {
        CalcSumURAddBlock(ubLocal2, repeatTimes);
    }
    CalcSumURReduce(ubLocal2, ubLocal4, src0RepStride);
}

template <typename T>
__aicore__ inline void ApplyCamePart3FP16<T>::CalcSumURC(LocalTensor<float>& ubLocal4, LocalTensor<float>& ubLocal3)
{
    ApplyCamePart3ReduceVector(ubLocal3, ubLocal4, baseM);
}

template <typename T>
__aicore__ inline void ApplyCamePart3FP16<T>::CalcAddEps(LocalTensor<float>& ubLocal2)
{
    if (baseM % FP32_ONE_BLOCK_COUNT && baseM < ONE_VECTOR_FP32_SIZE) {
        int64_t curRepeatTimes = baseN;
        int64_t overMaxRepeat = 0;
        int64_t totalRepeatTimes = baseN;
        if (curRepeatTimes > MAX_REPEAT_TIME) {
            curRepeatTimes = MAX_REPEAT_TIME;
            overMaxRepeat = 1;
        }
        Adds(ubLocal2, ubLocal2, eps, baseM, curRepeatTimes,
             {1, 1, static_cast<uint8_t>(rowBlockStride), static_cast<uint8_t>(rowBlockStride)});
        int64_t offset = rowBlockStride * FP32_ONE_BLOCK_COUNT * curRepeatTimes;
        totalRepeatTimes -= curRepeatTimes;
        curRepeatTimes = totalRepeatTimes > MAX_REPEAT_TIME ? MAX_REPEAT_TIME : totalRepeatTimes;
        while (overMaxRepeat && curRepeatTimes > 0) {
            Adds(ubLocal2[offset], ubLocal2[offset], eps, baseM, curRepeatTimes,
                 {1, 1, static_cast<uint8_t>(rowBlockStride), static_cast<uint8_t>(rowBlockStride)});
            offset += rowBlockStride * FP32_ONE_BLOCK_COUNT * curRepeatTimes;
            totalRepeatTimes -= curRepeatTimes;
            curRepeatTimes = totalRepeatTimes > MAX_REPEAT_TIME ? MAX_REPEAT_TIME : totalRepeatTimes;
        }
    } else if (baseM % FP32_ONE_BLOCK_COUNT && baseM > ONE_VECTOR_FP32_SIZE) {
        int64_t tLoop = baseM / ONE_VECTOR_FP32_SIZE;
        for (int64_t tIdx = 0; tIdx < tLoop; tIdx++) {
            int64_t offset = tIdx * ONE_VECTOR_FP32_SIZE;
            Adds(ubLocal2[offset], ubLocal2[offset], eps, ONE_VECTOR_FP32_SIZE, baseN,
                 {1, 1, static_cast<uint8_t>(tailBlockStride), static_cast<uint8_t>(tailBlockStride)});
        }
        uint64_t mask = baseM % ONE_VECTOR_FP32_SIZE;
        int64_t tailBlockOffset = baseM / ONE_VECTOR_FP32_SIZE * ONE_VECTOR_FP32_SIZE;
        Adds(ubLocal2[tailBlockOffset], ubLocal2[tailBlockOffset], eps, mask, baseN,
             {1, 1, static_cast<uint8_t>(tailBlockStride), static_cast<uint8_t>(tailBlockStride)});
    } else {
        Adds(ubLocal2, ubLocal2, eps, bufSize);
    }
}

template <typename T>
__aicore__ inline void ApplyCamePart3FP16<T>::ProcessOneLoop(int64_t mIdx, int64_t nIdx)
{
    ClearAcculateMatrix();
    PipeBarrier<PIPE_V>();
    CalcOneOffset(mIdx, nIdx);

    CopyInU(ubLocal2, uGm);
    CopyInM(ubLocal3, mInputGm);
    CalcOutM(ubLocal2, ubLocal3, ubLocal4);
    if (useFirstMoment) {
        CopyOutM(mOutputGm, ubLocal3, outMOffset);
    }

    Sub(ubLocal2, ubLocal2, ubLocal3, bufSize);
    PipeBarrier<PIPE_V>();
    Mul(ubLocal2, ubLocal2, ubLocal2, bufSize);
    PipeBarrier<PIPE_V>();
    CalcAddEps(ubLocal2);
    PipeBarrier<PIPE_V>();

    CalcSumUC(ubLocal2, ubLocal4);
    CopyUB2Workspace(workspaceSumGradC_, ubLocal4, workspaceCOffsetBase, baseM);

    CalcSumURC(ubLocal4, ubLocal3);
    CopyUB2Workspace(workspaceSumGradRC_, ubLocal3, workspaceRCOffsetBase, 1);

    CalcSumUR(ubLocal2, ubLocal3, ubLocal4);
    CopyUB2Out(sumURGm, ubLocal4, nOffset, baseN);

    event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
}

template <typename T>
__aicore__ inline void ApplyCamePart3FP16<T>::CopyScalar(GlobalTensor<float>& scaleGm, float& scaleValue)
{
    LocalTensor<float> ubLocalIn = inQueue.AllocTensor<float>();

    DataCopyPad(ubLocalIn, scaleGm, {1, sizeof(float), 0, 0, 0}, {false, 0, 0, 0});
    inQueue.EnQue(ubLocalIn);
    ubLocalIn = inQueue.DeQue<float>();
    event_t eventIdMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    SetFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
    WaitFlag<HardEvent::MTE2_S>(eventIdMte2ToS);
    scaleValue = ubLocalIn.GetValue(0);
    inQueue.FreeTensor(ubLocalIn);
}

template <typename T>
__aicore__ inline void ApplyCamePart3FP16<T>::SetNM()
{
    if (isGlobalShape) {
        LocalTensor<int64_t> ubLocalIn = inQueue.AllocTensor<int64_t>();
        DataCopy(ubLocalIn, globalShapeGm, INT64_ONE_BLOCK_COUNT);
        inQueue.EnQue(ubLocalIn);
        ubLocalIn = inQueue.DeQue<int64_t>();

        Cast(ubLocal2, ubLocalIn, RoundMode::CAST_ROUND, INT64_ONE_BLOCK_COUNT);
        event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);
        globalN = ubLocal2.GetValue(0);
        globalM = ubLocal2.GetValue(1);
        inQueue.FreeTensor(ubLocalIn);
    } else {
        globalN = static_cast<float>(curN);
        globalM = static_cast<float>(curM);
    }
}

template <typename T>
__aicore__ inline void ApplyCamePart3FP16<T>::CalcScalar()
{
    CopyScalar(epsGm, eps);
    CopyScalar(beta1Gm, beta1);
    CopyScalar(clipThresholdGm, clipThreshold);
    CopyScalar(sumSquareUGm, sumSquareU);
    SetNM();
    float scaleRes = sumSquareU / (globalM * globalN) / clipThreshold;
    if (scaleRes > 1) {
        maxValue = scaleRes;
    }
    beta2 = 1 - beta1;
}

// 输出的接口
template <typename T>
__aicore__ inline void ApplyCamePart3FP16<T>::CopyOut(GlobalTensor<float>& outGlobal, int64_t offset, int64_t size)
{
    auto tmp = outQueue.DeQue<float>();

#if defined(DETERMINISTIC_MODE) && DETERMINISTIC_MODE == 1
    WaitPreBlock(gmDetWorkspace, ubDetWorkspace);
#endif
    if (size == 1) {
        event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIdVToS);
        WaitFlag<HardEvent::V_S>(eventIdVToS);
        AtomicAdd(outGlobal.GetPhyAddr(static_cast<uint64_t>(offset)), tmp.GetValue(0));
        event_t eventIdSToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(eventIdSToMte3);
        WaitFlag<HardEvent::S_MTE3>(eventIdSToMte3);
    } else {
        SetAtomicAdd<float>();
        if (size % FP32_ONE_BLOCK_COUNT == 0) {
            DataCopy(outGlobal[offset], tmp, size);
        } else {
            DataCopyParams copyParams{1, static_cast<uint16_t>(size * sizeof(float)), 0, 0};
            DataCopyPad(outGlobal[offset], tmp, copyParams);
        }
        SetAtomicNone();
    }
#if defined(DETERMINISTIC_MODE) && DETERMINISTIC_MODE == 1
    NotifyNextBlock(gmDetWorkspace, ubDetWorkspace);
#endif

    outQueue.FreeTensor(tmp);
}

template <typename T>
__aicore__ inline void ApplyCamePart3FP16<T>::MoveOut(LocalTensor<float>& ubLocal, int64_t size)
{
    LocalTensor<float> tmp = outQueue.AllocTensor<float>();
    PipeBarrier<PIPE_V>();
    Muls(tmp, ubLocal, static_cast<float>(1.0), size);
    outQueue.EnQue(tmp);
}

template <typename T>
__aicore__ inline void ApplyCamePart3FP16<T>::CopyUB2Out(GlobalTensor<float>& outGlobal, LocalTensor<float>& ubLocal,
                                                         int64_t offset, int64_t size)
{
    // 将要输出的数据搬到outque
    MoveOut(ubLocal, size);
    CopyOut(outGlobal, offset, size);
}

template <typename T>
__aicore__ inline void ApplyCamePart3FP16<T>::CopyUB2Workspace(GlobalTensor<float>& outGlobal,
                                                               LocalTensor<float>& ubLocal, int64_t offset,
                                                               int64_t size)
{
    // 将要输出的数据搬到outque
    MoveOut(ubLocal, size);

    auto tmp = outQueue.DeQue<float>();
    DataCopyPad(outGlobal[offset], tmp, {1, static_cast<uint16_t>(size * sizeof(float)), 0, 0});
    outQueue.FreeTensor(tmp);
}

template <typename T>
__aicore__ inline void ApplyCamePart3FP16<T>::CopyOutM(GlobalTensor<T>& outGlobal, LocalTensor<float>& ubLocal,
                                                       int64_t offset)
{
    LocalTensor<T> tmp = outQueue.AllocTensor<T>();

    if (baseM % FP16_ONE_BLOCK_COUNT && baseM < ONE_VECTOR_FP32_SIZE) {
        int64_t curRepeatTimes = baseN;
        int64_t overMaxRepeat = 0;
        int64_t totalRepeatTimes = baseN;
        if (curRepeatTimes > MAX_REPEAT_TIME) {
            curRepeatTimes = MAX_REPEAT_TIME;
            overMaxRepeat = 1;
        }

        uint8_t dstRepStride = Ceil(baseM, FP16_ONE_BLOCK_COUNT) * sizeof(T) / CAME_ONE_BLOCK_SIZE;
        uint64_t mask = Ceil(baseM, FP32_ONE_BLOCK_COUNT);
        Cast(tmp, ubLocal, RoundMode::CAST_RINT, mask, curRepeatTimes,
             {1, 1, dstRepStride, static_cast<uint8_t>(rowBlockStride)});

        int64_t fp32Offset = curRepeatTimes * mask;
        int64_t fp16Offset = curRepeatTimes * Ceil(baseM, FP16_ONE_BLOCK_COUNT);
        totalRepeatTimes -= curRepeatTimes;
        curRepeatTimes = totalRepeatTimes > MAX_REPEAT_TIME ? MAX_REPEAT_TIME : totalRepeatTimes;
        while (overMaxRepeat && curRepeatTimes > 0) {
            Cast(tmp[fp16Offset], ubLocal[fp32Offset], RoundMode::CAST_RINT, mask, curRepeatTimes,
                 {1, 1, dstRepStride, static_cast<uint8_t>(rowBlockStride)});
            fp32Offset += curRepeatTimes * mask;
            fp16Offset += curRepeatTimes * Ceil(baseM, FP16_ONE_BLOCK_COUNT);
            totalRepeatTimes -= curRepeatTimes;
            curRepeatTimes = totalRepeatTimes > MAX_REPEAT_TIME ? MAX_REPEAT_TIME : totalRepeatTimes;
        }
    } else if (baseM % FP16_ONE_BLOCK_COUNT && baseM > ONE_VECTOR_FP32_SIZE) {
        uint64_t repeatTimes = baseN;
        int64_t tLoop = baseM / ONE_VECTOR_FP32_SIZE;
        uint8_t dstRepStride = Ceil(baseM, FP16_ONE_BLOCK_COUNT) * sizeof(T) / CAME_ONE_BLOCK_SIZE;
        for (int64_t tIdx = 0; tIdx < tLoop; tIdx++) {
            int64_t offset = tIdx * ONE_VECTOR_FP32_SIZE;
            Cast(tmp[offset], ubLocal[offset], RoundMode::CAST_RINT, ONE_VECTOR_FP32_SIZE, repeatTimes,
                 {1, 1, dstRepStride, static_cast<uint8_t>(tailBlockStride)});
        }
        uint64_t mask = baseM % ONE_VECTOR_FP32_SIZE;
        int64_t tailBlockOffset = baseM / ONE_VECTOR_FP32_SIZE * ONE_VECTOR_FP32_SIZE;
        Cast(tmp[tailBlockOffset], ubLocal[tailBlockOffset], RoundMode::CAST_RINT, mask, repeatTimes,
             {1, 1, dstRepStride, static_cast<uint8_t>(tailBlockStride)});
    } else {
        Cast(tmp, ubLocal, RoundMode::CAST_RINT, bufSize);
    }
    outQueue.EnQue(tmp);
    tmp = outQueue.DeQue<T>();

    if (baseM < FP16_ONE_BLOCK_COUNT) {
        event_t eventVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventVToS);
        WaitFlag<HardEvent::V_S>(eventVToS);
        const int64_t sourceRowStride = Ceil(baseM, FP16_ONE_BLOCK_COUNT);
        for (int64_t row = 0; row < baseN; ++row) {
            for (int64_t column = 0; column < baseM; ++column) {
                outGlobal.SetValue(offset + row * curM + column, tmp.GetValue(row * sourceRowStride + column));
            }
        }
        event_t eventSToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(eventSToMte3);
        WaitFlag<HardEvent::S_MTE3>(eventSToMte3);
        outQueue.FreeTensor(tmp);
        return;
    }

    if (baseM % FP16_ONE_BLOCK_COUNT != 0) {
        const int64_t sourceRowStride = Ceil(baseM, FP16_ONE_BLOCK_COUNT);
        DataCopyParams rowCopyParams{1, static_cast<uint16_t>(baseM * sizeof(T)), 0, 0};
        for (int64_t row = 0; row < baseN; ++row) {
            DataCopyPad(outGlobal[offset + row * curM], tmp[row * sourceRowStride], rowCopyParams);
        }
        event_t eventMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2);
        outQueue.FreeTensor(tmp);
        return;
    }

    DataCopyParams intriParams;
    intriParams.blockCount = baseN;
    intriParams.blockLen = baseM * sizeof(T);
    intriParams.srcStride = 0;
    intriParams.dstStride = (curM - baseM) * sizeof(T);
    DataCopyPad(outGlobal[offset], tmp, intriParams);

    // The output may alias m input. Ensure the GM write is visible before the
    // tensor is released and a following tile can read the same buffer.
    event_t eventMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2);
    WaitFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2);
    outQueue.FreeTensor(tmp);
}
// -------------- ApplyCamePart3FP16 -----------------
