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
 * \file ifmr.h
 * \brief
 */
#ifndef IFMR_H
#define IFMR_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "ifmr_tiling_data.h"

using namespace AscendC;
namespace Ifmr {

constexpr uint64_t ALIGN_LENGTH = 32;
constexpr uint64_t RESERVED_LENGTH = 160;
constexpr float EPSILON = 1.192092896e-7f;

struct IfmrAttrs {
    float minPercentile;
    float maxPercentile;
    float searchRange[2];
    float searchStep;
    bool withOffset;
    uint32_t quantBits;
    uint32_t dataNums;
    uint32_t cumsumNums;
    uint32_t stepNums;
    uint32_t stepAlignedBytes;
};

template <typename T>
class KernelIfmr {
public:
    __aicore__ inline KernelIfmr() {}
    __aicore__ inline void ParseTilingData(const IfmrTilingData& tilingData) {
        attrs_.minPercentile = tilingData.minPercentile;
        attrs_.maxPercentile = tilingData.maxPercentile;
        attrs_.searchRange[0] = tilingData.searchRange[0];
        attrs_.searchRange[1] = tilingData.searchRange[1];
        attrs_.searchStep = tilingData.searchStep;
        attrs_.withOffset = tilingData.withOffset;
        attrs_.quantBits = tilingData.quantBits;
        attrs_.dataNums = tilingData.dataLength;
        attrs_.cumsumNums = tilingData.cumsumLength;
        attrs_.stepNums = Round((attrs_.searchRange[1] - attrs_.searchRange[0]) / attrs_.searchStep) + 1;
        attrs_.stepAlignedBytes = AlignB(attrs_.stepNums * sizeof(float));
    }

    __aicore__ inline void Init(GM_ADDR data, GM_ADDR data_min, GM_ADDR data_max, GM_ADDR cumsum,
        GM_ADDR scale, GM_ADDR offset, GM_ADDR workSpace, const IfmrTilingData& tiling)
    {
        ParseTilingData(tiling);
        dataGm_.SetGlobalBuffer((__gm__ T *)data, AlignB(attrs_.dataNums * sizeof(T)) / sizeof(T));
        dataMinGm_.SetGlobalBuffer((__gm__ T *)data_min, ALIGN_LENGTH / sizeof(T));
        dataMaxGm_.SetGlobalBuffer((__gm__ T *)data_max, ALIGN_LENGTH / sizeof(T));
        cumsumGm_.SetGlobalBuffer((__gm__ int32_t*)cumsum,
                                  AlignB(attrs_.cumsumNums * sizeof(int32_t)) / sizeof(int32_t));
        scaleGm_.SetGlobalBuffer((__gm__ float *)scale, 1);
        offsetGm_.SetGlobalBuffer((__gm__ float *)offset, 1);

        pipe_.InitBuffer(dataMaxTQue_, 1, ALIGN_LENGTH);
        pipe_.InitBuffer(dataMinTQue_, 1, ALIGN_LENGTH);
        uint64_t cumsumAlignedSize = AlignB(attrs_.cumsumNums * sizeof(int32_t));
        pipe_.InitBuffer(cumsumTQue_, 1, cumsumAlignedSize);
        pipe_.InitBuffer(maxListTQue_, 1, attrs_.stepNums * sizeof(float));
        pipe_.InitBuffer(minListTQue_, 1, attrs_.stepNums * sizeof(float));

        pipe_.InitBuffer(scaleListQueue_, 1, attrs_.stepAlignedBytes);
        pipe_.InitBuffer(offsetListQueue_, 1, attrs_.stepAlignedBytes);

        clipMax_ = (1 << (attrs_.quantBits - 1)) - 1;
        clipMin_ = -1 << (attrs_.quantBits - 1);
        coreNum_ = GetBlockNum();
        ASSERT(coreNum_ != 0 && "block dim can not be zero!");
        coreIdx_ = GetBlockIdx();
        dataEachBlock_ = ALIGN_LENGTH / sizeof(T);
        eachCoreDataNum_ = attrs_.dataNums / dataEachBlock_ / coreNum_ * dataEachBlock_;
        lastCoreDataNum_ = attrs_.dataNums - (eachCoreDataNum_ * (coreNum_ - 1));
        uint32_t ubFactor;
        if constexpr (std::is_same<T, half>::value) {
            ubFactor = 4; // max((orignal(1) + cast_to_f32(2)), (f32(2) + quant(2))) = 4
        } else {
            ubFactor = 2; // orignal(1) + quant(1) = 2
        }
        // 6 means maxList+minList+scale+offset+lossOut+lossIn
        eachLoopDataNum_ = (TOTAL_UB_SIZE - attrs_.stepAlignedBytes * 6 - cumsumAlignedSize -
            coreNum_ * ALIGN_LENGTH - RESERVED_LENGTH) / sizeof(T) / ubFactor / dataEachBlock_ * dataEachBlock_;
        eachCoreLossNum_ = attrs_.stepAlignedBytes / sizeof(float);
        mseLossGm_.SetGlobalBuffer((__gm__ float *)workSpace, eachCoreLossNum_ * coreNum_);
        pipe_.InitBuffer(dataQueue_, 1, eachLoopDataNum_ * sizeof(float));
        pipe_.InitBuffer(lossOutQueue_, 1, eachCoreLossNum_ * sizeof(float));
        pipe_.InitBuffer(lossInQueue_, 1, eachCoreLossNum_ * sizeof(float));
        syncGm_.SetGlobalBuffer((__gm__ int32_t*)(workSpace) + eachCoreLossNum_ * coreNum_,
            coreNum_ * ALIGN_LENGTH / sizeof(int32_t));
        pipe_.InitBuffer(syncGmQue_, 1, coreNum_ * ALIGN_LENGTH);
        pipe_.InitBuffer(tempBuf_, eachLoopDataNum_ * sizeof(float));
    }

    __aicore__ inline int32_t Round(float x)
    {
        return static_cast<int32_t>(x + 0.5f);
    }

    __aicore__ inline uint64_t AlignB(uint64_t a, uint64_t b = ALIGN_LENGTH)
    {
        if (b == 0) {
            return a;
        }
        return ((a + b - 1) / b) * b;
    }

    __aicore__ inline void CopyIn()
    {
        LocalTensor<T> dataMax = dataMaxTQue_.AllocTensor<T>();
        LocalTensor<T> dataMin = dataMinTQue_.AllocTensor<T>();
        LocalTensor<int32_t> cumsum = cumsumTQue_.AllocTensor<int32_t>();

        uint64_t cumsumAlignedSize = AlignB(attrs_.cumsumNums * sizeof(int32_t));
        DataCopy<int32_t>(cumsum, cumsumGm_, cumsumAlignedSize / sizeof(uint32_t));
        DataCopy<T>(dataMax, dataMaxGm_, ALIGN_LENGTH / sizeof(T));
        DataCopy<T>(dataMin, dataMinGm_, ALIGN_LENGTH / sizeof(T));

        dataMaxTQue_.EnQue(dataMax);
        dataMinTQue_.EnQue(dataMin);
        cumsumTQue_.EnQue(cumsum);
    }

    __aicore__ inline void Process()
    {
        CopyIn();
        event_t eventIdMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMTE2ToV);

        LocalTensor<T> dataMax = dataMaxTQue_.DeQue<T>();
        LocalTensor<T> dataMin = dataMinTQue_.DeQue<T>();
        LocalTensor<int32_t> cumsum = cumsumTQue_.DeQue<int32_t>();
        LocalTensor<float> maxList = maxListTQue_.AllocTensor<float>();
        LocalTensor<float> minList = minListTQue_.AllocTensor<float>();
        GenerateMaxMinCandidates(maxList, minList, cumsum, dataMax, dataMin);

        dataMaxTQue_.FreeTensor(dataMax);
        dataMinTQue_.FreeTensor(dataMin);
        cumsumTQue_.FreeTensor(cumsum);

        LocalTensor<float> scaleList = scaleListQueue_.AllocTensor<float>();
        LocalTensor<float> offsetList = offsetListQueue_.AllocTensor<float>();
        CalScaleOffset(scaleList, offsetList, maxList, minList);
        maxListTQue_.FreeTensor(maxList);
        minListTQue_.FreeTensor(minList);

        ComputeMSELoss(scaleList, offsetList);
        if (coreNum_ > 1) {
            SyncAll();
        }

        if (coreIdx_ == 0) {
            ReduceAndOutput(scaleList, offsetList);
        }

        scaleListQueue_.FreeTensor(scaleList);
        offsetListQueue_.FreeTensor(offsetList);
    }

private:
    TPipe pipe_;
    GlobalTensor<T> dataGm_;
    GlobalTensor<T> dataMinGm_;
    GlobalTensor<T> dataMaxGm_;
    GlobalTensor<int32_t> cumsumGm_;

    GlobalTensor<float> scaleGm_;
    GlobalTensor<float> offsetGm_;

    TQue<TPosition::VECIN, 1> dataMaxTQue_;
    TQue<TPosition::VECIN, 1> dataMinTQue_;
    TQue<TPosition::VECIN, 1> cumsumTQue_;
    TQue<TPosition::VECOUT, 1> maxListTQue_;
    TQue<TPosition::VECOUT, 1> minListTQue_;
    TQue<TPosition::VECOUT, 1> scaleListQueue_;
    TQue<TPosition::VECOUT, 1> offsetListQueue_;

    GlobalTensor<int32_t> syncGm_;
    TQue<TPosition::VECOUT, 1> syncGmQue_;
    GlobalTensor<float> mseLossGm_;
    TQue<TPosition::VECIN, 1> dataQueue_;
    TQue<TPosition::VECOUT, 1> lossOutQueue_;
    TQue<TPosition::VECIN, 1> lossInQueue_;
    TBuf<TPosition::VECCALC> tempBuf_;

    int64_t coreNum_{0};
    int64_t coreIdx_{0};
    uint32_t dataEachBlock_{0};
    uint32_t eachCoreDataNum_{0};
    uint32_t lastCoreDataNum_{0};
    uint32_t eachLoopDataNum_{0};
    uint32_t eachCoreLossNum_{0};
    float clipMax_{0.0f};
    float clipMin_{0.0f};

    IfmrAttrs attrs_;

    __aicore__ inline void CalScaleOffset(LocalTensor<float>& scaleList, LocalTensor<float>& offsetList, 
        const LocalTensor<float>& maxList, const LocalTensor<float>& minList);

    __aicore__ inline void GetMinMaxIndex(LocalTensor<float>& minIndex, LocalTensor<float>& maxIndex,
        const LocalTensor<int32_t>& cumsumLocal);

    __aicore__ inline void CalcMinInitMaxInit(LocalTensor<float>& minIndex, LocalTensor<float>& maxIndex, 
        const LocalTensor<T>& dataMin, const LocalTensor<T>& dataMax);

    __aicore__ inline void GenerateMaxMinCandidates(LocalTensor<float>& maxList, LocalTensor<float>& minList,
        const LocalTensor<int32_t>& cumsumLocal, const LocalTensor<T>& dataMax, const LocalTensor<T>& dataMin);

    __aicore__ inline void GetInputData(uint32_t loopIdx, uint32_t calNum);

    __aicore__ inline void ComputeEachMSELoss(const LocalTensor<float>& scaleList,
        const LocalTensor<float>& offsetList, LocalTensor<float>& eachCoreLoss, uint32_t calNum);

    __aicore__ inline void ComputeMSELoss(const LocalTensor<float>& scaleList, const LocalTensor<float>& offsetList);

    __aicore__ inline void ReduceAndOutput(const LocalTensor<float>& scaleList, const LocalTensor<float>& offsetList);
};

template <typename T>
__aicore__ inline void KernelIfmr<T>::CalScaleOffset(LocalTensor<float>& scaleList, LocalTensor<float>& offsetList,
    const LocalTensor<float>& maxList, const LocalTensor<float>& minList)
{
    LocalTensor<float> tmp = tempBuf_.Get<float>(attrs_.stepNums);
    if (attrs_.withOffset) {
        Sub(tmp, maxList, minList, attrs_.stepNums);
        float divisor = 1 / static_cast<float>((1ULL << attrs_.quantBits) - 1);

        PipeBarrier<PIPE_ALL>();
        Muls(scaleList, tmp, divisor, attrs_.stepNums);
        PipeBarrier<PIPE_ALL>();

        Div(tmp, minList, scaleList, attrs_.stepNums);
        PipeBarrier<PIPE_ALL>();
        Cast(tmp, tmp, RoundMode::CAST_RINT, attrs_.stepNums);
        PipeBarrier<PIPE_ALL>();
        Muls(tmp, tmp, -1.0f, attrs_.stepNums);
        PipeBarrier<PIPE_ALL>();
        Adds(offsetList, tmp, clipMin_, attrs_.stepNums);
    } else {
        float divisor = 1 / clipMax_;
        Muls(scaleList, maxList, divisor, attrs_.stepNums);
        PipeBarrier<PIPE_ALL>();
        Duplicate(offsetList, 0.0f, attrs_.stepNums);
    }
    PipeBarrier<PIPE_ALL>();
    return;
}

template <typename T>
__aicore__ inline void KernelIfmr<T>::GetMinMaxIndex(LocalTensor<float>& minIndex, LocalTensor<float>& maxIndex, 
    const LocalTensor<int32_t>& cumsumLocal)
{
    LocalTensor<float> cdf = tempBuf_.Get<float>(attrs_.cumsumNums);
    Cast(cdf, cumsumLocal, RoundMode::CAST_NONE, attrs_.cumsumNums);
    PipeBarrier<PIPE_ALL>();

    Muls(cdf, cdf , 1 / static_cast<float>(attrs_.dataNums), attrs_.cumsumNums);
    PipeBarrier<PIPE_ALL>();

    int32_t compareResultSize = AlignB(attrs_.cumsumNums, AscendCUtils::GetBitSize(sizeof(uint8_t))) /
                                AscendCUtils::GetBitSize(sizeof(uint8_t));
    LocalTensor<uint8_t> compareResult = tempBuf_.GetWithOffset<uint8_t>(compareResultSize,
        AlignB(attrs_.cumsumNums * sizeof(float)));
    CompareScalar(compareResult, cdf, attrs_.maxPercentile, CMPMODE::LT, attrs_.cumsumNums);

    LocalTensor<float> selectResult = tempBuf_.GetWithOffset<float>(attrs_.cumsumNums,
        AlignB(attrs_.cumsumNums * sizeof(float)) + AlignB(compareResultSize * sizeof(uint8_t)));
    Duplicate(selectResult, 1.0f, attrs_.cumsumNums);
    PipeBarrier<PIPE_ALL>();

    Select(selectResult, compareResult, selectResult, 0.0f,
           SELMODE::VSEL_TENSOR_SCALAR_MODE, attrs_.cumsumNums);
    PipeBarrier<PIPE_ALL>();

    // get max index by reduce sum max percentile comp result
    TBuf<TPosition::VECCALC> workSpaceTbuf;
    pipe_.InitBuffer(workSpaceTbuf, 1 * sizeof(float));
    LocalTensor<float> workSpaceTensor = workSpaceTbuf.Get<float>();

    ReduceSum(maxIndex, selectResult, workSpaceTensor, attrs_.cumsumNums);
    PipeBarrier<PIPE_ALL>();

    // get min index
    CompareScalar(compareResult, cdf, 1 - attrs_.minPercentile, CMPMODE::LT, attrs_.cumsumNums);
    Duplicate(selectResult, 1.0f, attrs_.cumsumNums);
    PipeBarrier<PIPE_ALL>();

    Select(selectResult, compareResult, selectResult, 0.0f,
           SELMODE::VSEL_TENSOR_SCALAR_MODE, attrs_.cumsumNums);
    PipeBarrier<PIPE_ALL>();
    ReduceSum(minIndex, selectResult, workSpaceTensor, attrs_.cumsumNums);
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline void KernelIfmr<T>::CalcMinInitMaxInit(LocalTensor<float>& minIndex, LocalTensor<float>& maxIndex,
    const LocalTensor<T>& dataMin, const LocalTensor<T>& dataMax)
{
    TBuf<TPosition::VECCALC> dataMaxFloat32Buf;
    TBuf<TPosition::VECCALC> dataMinFloat32Buf;

    pipe_.InitBuffer(dataMaxFloat32Buf, sizeof(float));
    pipe_.InitBuffer(dataMinFloat32Buf, sizeof(float));

    // calc max init
    LocalTensor<float> dataMaxFloat32 = dataMaxFloat32Buf.Get<float>();
    LocalTensor<float> dataMinFloat32 = dataMinFloat32Buf.Get<float>();
    if constexpr (std::is_same<T, half>::value) {
        Cast(dataMaxFloat32, dataMax, RoundMode::CAST_NONE, 1);
        Cast(dataMinFloat32, dataMin, RoundMode::CAST_NONE, 1);
    } else {
        Adds(dataMaxFloat32, dataMax, 0.0f, 1);
        Adds(dataMinFloat32, dataMin, 0.0f, 1);
    }
    PipeBarrier<PIPE_ALL>();

    Muls(maxIndex, maxIndex,  1 / static_cast<float>(attrs_.cumsumNums), 1);
    Sub(dataMaxFloat32, dataMaxFloat32, dataMinFloat32, 1);
    PipeBarrier<PIPE_ALL>();
    Mul(maxIndex, maxIndex, dataMaxFloat32, 1);
    Add(maxIndex, maxIndex, dataMinFloat32, 1);

    // calc min init 
    Muls(minIndex, minIndex, 1 / static_cast<float>(attrs_.cumsumNums), 1);
    Mul(minIndex, minIndex, dataMaxFloat32, 1);
    Add(minIndex, minIndex, dataMinFloat32, 1);
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline void KernelIfmr<T>::GenerateMaxMinCandidates(LocalTensor<float>& maxList,
    LocalTensor<float>& minList, const LocalTensor<int32_t>& cumsumLocal,
    const LocalTensor<T>& dataMax, const LocalTensor<T>& dataMin)
{
    // apply local tensor
    TQue<TPosition::VECOUT, 1> maxIndexQueue;
    TQue<TPosition::VECOUT, 1> minIndexQueue;

    // init buffer for vars in calc
    pipe_.InitBuffer(minIndexQueue, 1, sizeof(float));
    pipe_.InitBuffer(maxIndexQueue, 1, sizeof(float));

    LocalTensor<float> minIndex = minIndexQueue.AllocTensor<float>();
    LocalTensor<float> maxIndex = maxIndexQueue.AllocTensor<float>();
    GetMinMaxIndex(minIndex, maxIndex, cumsumLocal);
    maxIndexQueue.EnQue(maxIndex);
    minIndexQueue.EnQue(minIndex);

    maxIndex = maxIndexQueue.DeQue<float>();
    minIndex = minIndexQueue.DeQue<float>();
    CalcMinInitMaxInit(minIndex, maxIndex, dataMin, dataMax);

    if (attrs_.withOffset) {
        Maxs(maxIndex, maxIndex, 0.0f, 1);
        Mins(minIndex, minIndex, 0.0f, 1);
        PipeBarrier<PIPE_ALL>();

        Duplicate(minList, minIndex.GetValue(0), attrs_.stepNums);
        PipeBarrier<PIPE_ALL>();
    } else {
        Abs(minIndex, minIndex, 1);
        PipeBarrier<PIPE_ALL>();
        Max(maxIndex, maxIndex, minIndex, 1);
        PipeBarrier<PIPE_ALL>();
    }

    float maxInit = maxIndex.GetValue(0);
    PipeBarrier<PIPE_ALL>();
    minIndexQueue.FreeTensor(minIndex);
    maxIndexQueue.FreeTensor(maxIndex);
    for (int32_t i = 0; i < attrs_.stepNums; i++) {
        maxList.SetValue(i, (attrs_.searchRange[0] + attrs_.searchStep * i) * maxInit);
    }
    PipeBarrier<PIPE_ALL>();
    return;
}

template <typename T>
__aicore__ inline void KernelIfmr<T>::GetInputData(uint32_t loopIdx, uint32_t calNum)
{
    LocalTensor<float> data = dataQueue_.AllocTensor<float>();
    if constexpr (std::is_same<T, half>::value) {
        LocalTensor<half> tempData = tempBuf_.Get<half>(calNum);
        DataCopy(tempData, dataGm_[eachCoreDataNum_ * coreIdx_ + eachLoopDataNum_ * loopIdx], calNum);
        event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventID);
        WaitFlag<HardEvent::MTE2_V>(eventID);
        Cast(data, tempData, RoundMode::CAST_NONE, calNum);
    } else {
        DataCopy(data, dataGm_[eachCoreDataNum_ * coreIdx_ + eachLoopDataNum_ * loopIdx], calNum);
    }
    dataQueue_.EnQue(data);
}

template <typename T>
__aicore__ inline void KernelIfmr<T>::ComputeEachMSELoss(const LocalTensor<float>& scaleList,
    const LocalTensor<float>& offsetList, LocalTensor<float>& eachCoreLoss, uint32_t calNum)
{
    LocalTensor<float> data = dataQueue_.DeQue<float>();
    LocalTensor<float> tempData = tempBuf_.Get<float>(calNum);
    for (uint32_t i = 0; i < attrs_.stepNums; i++) {
        // quant
        Muls(tempData, data, 1 / scaleList.GetValue(i), calNum);
        Adds(tempData, tempData, offsetList.GetValue(i), calNum);
        // round
        Cast(tempData, tempData, RoundMode::CAST_RINT, calNum);
        // clip
        Mins(tempData, tempData, clipMax_, calNum);
        Maxs(tempData, tempData, clipMin_, calNum);
        // dequant
        Adds(tempData, tempData, -offsetList.GetValue(i), calNum);
        Muls(tempData, tempData, scaleList.GetValue(i), calNum);
        // calculate mse loss
        Sub(tempData, tempData, data, calNum);
        Mul(tempData, tempData, tempData, calNum);
        ReduceSum(tempData, tempData, tempData, calNum);
        event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventID);
        WaitFlag<HardEvent::V_S>(eventID);
        eachCoreLoss.SetValue(i, eachCoreLoss.GetValue(i) + tempData.GetValue(0));
    }
    dataQueue_.FreeTensor(data);
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline void KernelIfmr<T>::ComputeMSELoss(const LocalTensor<float>& scaleList,
    const LocalTensor<float>& offsetList)
{
    LocalTensor<float> eachCoreLoss = lossOutQueue_.AllocTensor<float>();
    Duplicate(eachCoreLoss, 0.0f, eachCoreLossNum_);
    uint32_t inputNum = eachCoreDataNum_;
    if (coreIdx_ == (coreNum_ - 1)) {
        inputNum = lastCoreDataNum_;
    }
    uint32_t loopNum = inputNum / eachLoopDataNum_;
    for (uint32_t i = 0; i < loopNum; i++) {
        GetInputData(i, eachLoopDataNum_);
        ComputeEachMSELoss(scaleList, offsetList, eachCoreLoss, eachLoopDataNum_);
    }
    uint32_t leftDataNum = inputNum % eachLoopDataNum_;
    if (leftDataNum > 0) {
        GetInputData(loopNum, AlignB(leftDataNum, dataEachBlock_));
        ComputeEachMSELoss(scaleList, offsetList, eachCoreLoss, leftDataNum);
    }
    DataCopy(mseLossGm_[eachCoreLossNum_ * coreIdx_], eachCoreLoss, eachCoreLossNum_);
    lossOutQueue_.EnQue(eachCoreLoss);
    lossOutQueue_.DeQue<float>();
    lossOutQueue_.FreeTensor(eachCoreLoss);
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline void KernelIfmr<T>::ReduceAndOutput(const LocalTensor<float>& scaleList,
    const LocalTensor<float>& offsetList)
{
    LocalTensor<float> localLoss = lossInQueue_.AllocTensor<float>();
    LocalTensor<float> tempLoss = tempBuf_.Get<float>(eachCoreLossNum_);
    Duplicate(tempLoss, 0.0f, eachCoreLossNum_);
    for (uint32_t i = 0; i < coreNum_; i++) {
        event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventID);
        WaitFlag<HardEvent::V_MTE2>(eventID);
        DataCopy(localLoss, mseLossGm_[eachCoreLossNum_ * i], eachCoreLossNum_);
        eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventID);
        WaitFlag<HardEvent::MTE2_V>(eventID);
        Add(tempLoss, tempLoss, localLoss, attrs_.stepNums);
    }
    ReduceMin(tempLoss, tempLoss, tempLoss, attrs_.stepNums, true);
    event_t eventID = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventID);
    WaitFlag<HardEvent::V_S>(eventID);
    float tempMinLossIndex = tempLoss.GetValue(1);
    uint32_t minIndex = *reinterpret_cast<uint32_t*>(&tempMinLossIndex);
    float scale = scaleList.GetValue(minIndex);
    float offset = offsetList.GetValue(minIndex);
    if (scale < EPSILON) {
        scale = 1.0f;
        offset = attrs_.withOffset ? clipMin_ : 0.0f;
    }
    scaleGm_.SetValue(0, scale);
    offsetGm_.SetValue(0, offset);
    DataCacheCleanAndInvalid<float, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(scaleGm_);
    DataCacheCleanAndInvalid<float, AscendC::CacheLine::SINGLE_CACHE_LINE, AscendC::DcciDst::CACHELINE_OUT>(offsetGm_);
    lossInQueue_.FreeTensor(localLoss);
    PipeBarrier<PIPE_ALL>();
}
}
#endif // IFMR_H