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
 * \file softmax_cross_entropy_with_logits_split_r.h
 * \brief softmax_cross_entropy_with_logits_split_r
 */

#ifndef SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_SPLIT_R_H
#define SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_SPLIT_R_H

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "softmax_cross_entropy_with_logits_tiling_data.h"

namespace SoftmaxCrossEntropyWithLogits {
using namespace AscendC;
using namespace AscendC::Reg;

template <typename T, uint64_t schId, uint64_t featuresBrc, uint64_t labelsBrc, uint64_t db>
class SoftmaxCrossEntropyWithLogitsSplitR {
public:
    __aicore__ inline SoftmaxCrossEntropyWithLogitsSplitR(){};
    __aicore__ inline void Init(GM_ADDR features, GM_ADDR labels, GM_ADDR loss, GM_ADDR backProp, GM_ADDR workspace,
                                const SoftmaxCrossEntropyWithLogitsTilingData* tilingData, TPipe* pipe);
    __aicore__ inline void Process();

protected:
    __aicore__ inline void InitTilingData();
    __aicore__ inline void ProcessEachCore(int64_t loopTimes, int64_t tailNum);
    __aicore__ inline void Compute(int64_t tileNum, int64_t offset);
    __aicore__ inline void DataCopyFeaturePad(int32_t nTailNum, int32_t onceR, int32_t alignR, int64_t offset);
    __aicore__ inline void DataCopyFeature(int32_t tailNum, int32_t rOnceNum, int64_t offset);
    __aicore__ inline void DataCopyLabel(int32_t tailNum, int32_t rOnceNum, int64_t offset);
    __aicore__ inline void CopyOutBackProb(int32_t tailNum, int32_t rOnceNum, int64_t offset);
    __aicore__ inline void ComputeMaxLast(int32_t nNum, LocalTensor<T>& onceMaxUb);
    __aicore__ inline void GetRowMax(int32_t nTailNum, int64_t aOffset);
    __aicore__ inline void CleanMaxUb(int32_t tailNSize, LocalTensor<float>& clearUb);
    __aicore__ inline void UpdateCache(LocalTensor<float>& srcUb, int32_t index, int32_t dimA);
    __aicore__ inline void ComputeSubExpReduceSum(int32_t tailNum, int32_t aOffset);
    __aicore__ inline void ComputeLog(int32_t nSize, LocalTensor<float>& extraUb);
    __aicore__ inline void DataCopyOutLoss(int32_t nTailNum, int64_t offset);
    __aicore__ inline void CopyOutLoss(int32_t nSize, LocalTensor<float>& extraUb, int32_t aOffset);
    __aicore__ inline void ComputeRst(int32_t tailNum, int32_t aOffset);
    __aicore__ inline void DataCopyFeatureNDDMA(int64_t nTailNum, int64_t offset, int64_t onceR, int64_t onceRAlign,
                                                T padValue);
    __aicore__ inline void DataCopyLabelsNDDMA(int64_t nTailNum, int64_t offset, int64_t onceR, int64_t onceRAlign,
                                               T padValue);
    __aicore__ inline void ComputeSubExpMainBlock(int32_t nSize, int32_t rOnceNum, LocalTensor<float>& outUb);
    __aicore__ inline void ComputeSubExpTailBlock(int32_t nSize, int32_t rOnceNumAlign, int32_t rOnceNumT,
                                                  int32_t rOnceNumTAlign, LocalTensor<float>& outUb);
    __aicore__ inline void ComputeSubExpAlignBlock(int32_t nSize, int32_t rOnceNum, int32_t rOnceNumAlign,
                                                   LocalTensor<float>& outUb);
    __aicore__ inline void ComputeLossAndBropMainBlock(int32_t nSize, int32_t rOnceNum, LocalTensor<float>& outUb);
    __aicore__ inline void ComputeLossAndBropTailBlock(int32_t nSize, int32_t rOnceNumAlign, int32_t rOnceNumT,
                                                       int32_t rOnceNumTAlign, LocalTensor<float>& outUb);
    __aicore__ inline void ComputeLossAndBropAlignBlock(int32_t nSize, int32_t rOnceNum, int32_t rOnceNumAlign,
                                                        LocalTensor<float>& outUb);
    __aicore__ inline void PipeM2V();

protected:
    constexpr static AscendC::Reg::CastTrait castB32ToB16 = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::NO_SAT, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::CAST_RINT};
    constexpr static AscendC::Reg::CastTrait castB16ToB32 = {
        AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN, AscendC::Reg::MaskMergeMode::ZEROING,
        AscendC::RoundMode::UNKNOWN};

    constexpr static uint32_t DOUBLE_BUFFER = 1;
    constexpr static uint32_t BLOCK_SIZE = Ops::Base::GetUbBlockSize(); // 32B
    constexpr static uint32_t MAX_A_NUM = 8;                            // 32B
    constexpr static uint32_t VL_FP32 = static_cast<int64_t>(Ops::Base::GetVRegSize()) / sizeof(float);

private:
    GlobalTensor<T> featuresGm_;
    GlobalTensor<T> labelsGm_;
    GlobalTensor<T> lossGm_;
    GlobalTensor<T> backPropGm_;

    TPipe* pipe_ = nullptr;
    TQue<QuePosition::VECIN, 1> featuresQueue_;
    TQue<QuePosition::VECOUT, 1> labelsQueue_;
    TQue<QuePosition::VECOUT, 1> lossQueue_;
    TQue<QuePosition::VECOUT, 1> backPropQueue_;

    TBuf<QuePosition::VECCALC> maxBuf_;
    TBuf<QuePosition::VECCALC> sumBuf_;
    TBuf<QuePosition::VECCALC> temp1Buf_;
    TBuf<QuePosition::VECCALC> cacheBuf_;
    TBuf<QuePosition::VECCALC> logBuf_;

    int64_t blockFactor_ = 0;
    int64_t blockIdx_ = 0;
    bool isLastCore{false};
    const SoftmaxCrossEntropyWithLogitsTilingData* tilingData_ = nullptr;
    int64_t realCoreNum;
    int64_t a;               // a轴
    int64_t r;               // r轴
    int64_t blockFactor;     // a轴分核，主核数据量
    int64_t tailBlockFactor; // a轴分核，尾核数据量
    int64_t rUbNumFactor;    // R轴切分，一次UB可以放下的数据量，全载模板下等于r，注意32b对齐
    int64_t aUbNumFactor; // A轴切分，一次UB可以放下的数据量，非全载模板下等于1，注意32b对齐
    int64_t aLoopTimes;   // 主核A方向循环搬移数据的次数
    int64_t aLoopTimesT;  // 尾核A方向循环搬移数据的次数
    int64_t aLoopTail;    // 主核A方向尾块的数据量
    int64_t aLoopTailT;   // 尾核A方向尾块的数据量
    int64_t rLoopTime;    // 不能全载时，R轴反向的循环次数
    int64_t rLoopTile;    // 不能全载时，R轴反向的尾块数据量
    int64_t kTimesTail;   // 不能全载时，完全二分累加，存在主尾块相加的次数
    int64_t kTimes;       // 不能全载时，完全二分累加，2的k次方内循环次数
    int64_t updateStart_;
    int64_t rLoopTileAlign;
    int64_t featuresBrcDim;
    int64_t labelsBrcDim;
    int64_t coreStartOffset;
};

template <typename T, uint64_t schId, uint64_t featuresBrc, uint64_t labelsBrc, uint64_t db>
__aicore__ inline void SoftmaxCrossEntropyWithLogitsSplitR<T, schId, featuresBrc, labelsBrc, db>::InitTilingData()
{
    realCoreNum = tilingData_->realCoreNum;
    a = tilingData_->a;
    r = tilingData_->r;
    blockFactor = tilingData_->blockFactor;
    tailBlockFactor = tilingData_->tailBlockFactor;
    rUbNumFactor = tilingData_->rUbNumFactor;
    aUbNumFactor = tilingData_->aUbNumFactor;
    aLoopTimes = tilingData_->aLoopTimes;
    aLoopTimesT = tilingData_->aLoopTimesT;
    aLoopTail = tilingData_->aLoopTail;
    aLoopTailT = tilingData_->aLoopTailT;
    rLoopTime = tilingData_->rLoopTime;
    rLoopTile = tilingData_->rLoopTile;
    rLoopTileAlign = tilingData_->rLoopTileAlign;
    kTimesTail = tilingData_->kTimesTail;
    kTimes = tilingData_->kTimes;
    featuresBrcDim = tilingData_->featuresBrcDim;
    labelsBrcDim = tilingData_->labelsBrcDim;
}

template <typename T, uint64_t schId, uint64_t featuresBrc, uint64_t labelsBrc, uint64_t db>
__aicore__ inline void SoftmaxCrossEntropyWithLogitsSplitR<T, schId, featuresBrc, labelsBrc, db>::Init(
    GM_ADDR features, GM_ADDR labels, GM_ADDR loss, GM_ADDR backProp, GM_ADDR workspace,
    const SoftmaxCrossEntropyWithLogitsTilingData* tilingData, TPipe* pipe)
{
    tilingData_ = tilingData;
    InitTilingData();
    pipe_ = pipe;

    blockFactor_ = (AscendC::GetBlockIdx() == tilingData_->realCoreNum - 1) ? tailBlockFactor : blockFactor;

    featuresGm_.SetGlobalBuffer((__gm__ T*)features);
    labelsGm_.SetGlobalBuffer((__gm__ T*)labels);
    lossGm_.SetGlobalBuffer((__gm__ T*)loss);
    backPropGm_.SetGlobalBuffer((__gm__ T*)backProp);

    int64_t ubBufferSize = rUbNumFactor * aUbNumFactor;
    pipe_->InitBuffer(featuresQueue_, DOUBLE_BUFFER, ubBufferSize * sizeof(T));
    pipe_->InitBuffer(labelsQueue_, DOUBLE_BUFFER, ubBufferSize * sizeof(T));
    pipe_->InitBuffer(lossQueue_, DOUBLE_BUFFER, ubBufferSize * sizeof(float));
    pipe_->InitBuffer(backPropQueue_, DOUBLE_BUFFER, ubBufferSize * sizeof(T));

    pipe_->InitBuffer(maxBuf_, aUbNumFactor * sizeof(float));
    pipe_->InitBuffer(logBuf_, aUbNumFactor * sizeof(float));
    pipe_->InitBuffer(sumBuf_, aUbNumFactor * sizeof(float));
    pipe_->InitBuffer(temp1Buf_, aUbNumFactor * sizeof(float));
    updateStart_ = ScalarGetCountOfValue<1>((kTimes - 1) ^ (kTimes)) - 1;
    pipe_->InitBuffer(cacheBuf_, (updateStart_ * VL_FP32 + aUbNumFactor) * sizeof(float));

    blockIdx_ = AscendC::GetBlockIdx();
    isLastCore = blockIdx_ == tilingData_->realCoreNum - 1;
    coreStartOffset = blockFactor * blockIdx_;
}

template <typename T, uint64_t schId, uint64_t featuresBrc, uint64_t labelsBrc, uint64_t db>
__aicore__ inline void SoftmaxCrossEntropyWithLogitsSplitR<T, schId, featuresBrc, labelsBrc, db>::Process()
{
    if (unlikely(isLastCore)) {
        ProcessEachCore(aLoopTimesT, aLoopTailT);
    } else {
        ProcessEachCore(aLoopTimes, aLoopTail);
    }
}

template <typename T, uint64_t schId, uint64_t featuresBrc, uint64_t labelsBrc, uint64_t db>
__aicore__ inline void SoftmaxCrossEntropyWithLogitsSplitR<T, schId, featuresBrc, labelsBrc, db>::ProcessEachCore(
    int64_t loopTimes, int64_t tailNum)
{
    for (int64_t aLoopIndex = 0; aLoopIndex < loopTimes; aLoopIndex++) {
        Compute(aUbNumFactor, aLoopIndex * aUbNumFactor + coreStartOffset);
    }
    if (unlikely(tailNum != 0)) {
        Compute(tailNum, aUbNumFactor * loopTimes + coreStartOffset);
    }
}

template <typename T, uint64_t schId, uint64_t featuresBrc, uint64_t labelsBrc, uint64_t db>
__aicore__ inline void SoftmaxCrossEntropyWithLogitsSplitR<T, schId, featuresBrc, labelsBrc, db>::DataCopyFeatureNDDMA(
    int64_t nTailNum, int64_t offset, int64_t onceR, int64_t onceRAlign, T padValue)
{
    LocalTensor<T> xInUb = featuresQueue_.AllocTensor<T>();
    T constValue = padValue;
    static constexpr NdDmaConfig config = {false};
    NdDmaLoopInfo<2> loopInfo;

    loopInfo.loopSize[0] = onceR;
    loopInfo.loopSize[1] = nTailNum;
    loopInfo.loopLpSize[0] = 0;
    loopInfo.loopRpSize[0] = onceRAlign - onceR;
    loopInfo.loopLpSize[1] = 0;
    loopInfo.loopRpSize[1] = 0;
    if constexpr (featuresBrc == 2) {
        loopInfo.loopSrcStride[0] = 0;
        loopInfo.loopSrcStride[1] = 0;
        offset = 0;
    } else {
        loopInfo.loopSrcStride[0] = 1 - featuresBrcDim;
        loopInfo.loopSrcStride[1] = featuresBrcDim;
    }
    loopInfo.loopDstStride[0] = 1;
    loopInfo.loopDstStride[1] = onceRAlign;
    NdDmaParams<T, 2> paramsMain = {loopInfo, constValue};
    DataCopy<T, 2, config>(xInUb, featuresGm_[offset], paramsMain);
    featuresQueue_.EnQue(xInUb);
}

template <typename T, uint64_t schId, uint64_t featuresBrc, uint64_t labelsBrc, uint64_t db>
__aicore__ inline void SoftmaxCrossEntropyWithLogitsSplitR<T, schId, featuresBrc, labelsBrc, db>::DataCopyLabelsNDDMA(
    int64_t nTailNum, int64_t offset, int64_t onceR, int64_t onceRAlign, T padValue)
{
    LocalTensor<T> xInUb = labelsQueue_.AllocTensor<T>();
    T constValue = padValue;
    static constexpr NdDmaConfig config = {false};
    NdDmaLoopInfo<2> loopInfo;

    loopInfo.loopSize[0] = onceR;
    loopInfo.loopSize[1] = nTailNum;
    loopInfo.loopLpSize[0] = 0;
    loopInfo.loopRpSize[0] = onceRAlign - onceR;
    loopInfo.loopLpSize[1] = 0;
    loopInfo.loopRpSize[1] = 0;
    if constexpr (labelsBrc == 2) {
        loopInfo.loopSrcStride[0] = 0;
        loopInfo.loopSrcStride[1] = 0;
        offset = 0;
    } else {
        loopInfo.loopSrcStride[0] = 1 - labelsBrcDim;
        loopInfo.loopSrcStride[1] = labelsBrcDim;
    }
    loopInfo.loopDstStride[0] = 1;
    loopInfo.loopDstStride[1] = onceRAlign;
    NdDmaParams<T, 2> paramsMain = {loopInfo, constValue};
    DataCopy<T, 2, config>(xInUb, labelsGm_[offset], paramsMain);
    labelsQueue_.EnQue(xInUb);
}

template <typename T, uint64_t schId, uint64_t featuresBrc, uint64_t labelsBrc, uint64_t db>
__aicore__ inline void SoftmaxCrossEntropyWithLogitsSplitR<T, schId, featuresBrc, labelsBrc, db>::DataCopyFeaturePad(
    int32_t nTailNum, int32_t onceR, int32_t alignR, int64_t offset)
{
    LocalTensor<T> featureUB = featuresQueue_.AllocTensor<T>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = nTailNum;
    copyParams.blockLen = onceR * sizeof(T);
    copyParams.srcStride = (r - onceR) * sizeof(T);
    copyParams.dstStride = 0;
    DataCopyPadExtParams<T> padParams;
    T minValue = -3.402823466e+38;
    if constexpr (IsSameType<T, half>::value) {
        minValue = -65504;
    }
    padParams.isPad = true;
    padParams.leftPadding = 0;
    padParams.rightPadding = alignR - onceR;
    padParams.paddingValue = minValue;
    AscendC::DataCopyPad(featureUB, featuresGm_[offset], copyParams, padParams);

    featuresQueue_.EnQue(featureUB);
}
template <typename T, uint64_t schId, uint64_t featuresBrc, uint64_t labelsBrc, uint64_t db>
__aicore__ inline void SoftmaxCrossEntropyWithLogitsSplitR<T, schId, featuresBrc, labelsBrc, db>::DataCopyFeature(
    int32_t tailNum, int32_t rOnceNum, int64_t offset)
{
    LocalTensor<T> featureUB = featuresQueue_.AllocTensor<T>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = tailNum;
    copyParams.blockLen = rOnceNum * sizeof(T);
    copyParams.srcStride = (r - rOnceNum) * sizeof(T);
    copyParams.dstStride = 0;
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    AscendC::DataCopyPad(featureUB, featuresGm_[offset], copyParams, padParams);
    featuresQueue_.EnQue(featureUB);
}
template <typename T, uint64_t schId, uint64_t featuresBrc, uint64_t labelsBrc, uint64_t db>
__aicore__ inline void SoftmaxCrossEntropyWithLogitsSplitR<T, schId, featuresBrc, labelsBrc, db>::DataCopyLabel(
    int32_t tailNum, int32_t rOnceNum, int64_t offset)
{
    LocalTensor<T> labelUB = labelsQueue_.AllocTensor<T>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = tailNum;
    copyParams.blockLen = rOnceNum * sizeof(T);
    copyParams.srcStride = (r - rOnceNum) * sizeof(T);
    copyParams.dstStride = 0;
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    AscendC::DataCopyPad(labelUB, labelsGm_[offset], copyParams, padParams);
    labelsQueue_.EnQue(labelUB);
}
template <typename T, uint64_t schId, uint64_t featuresBrc, uint64_t labelsBrc, uint64_t db>
__aicore__ inline void SoftmaxCrossEntropyWithLogitsSplitR<T, schId, featuresBrc, labelsBrc, db>::CopyOutBackProb(
    int32_t tailNum, int32_t rOnceNum, int64_t offset)
{
    LocalTensor<T> srcUb = backPropQueue_.DeQue<T>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = tailNum;
    copyParams.blockLen = rOnceNum * sizeof(T);
    copyParams.dstStride = (r - rOnceNum) * sizeof(T);
    copyParams.srcStride = 0;
    AscendC::DataCopyPad(backPropGm_[offset], srcUb, copyParams);
    backPropQueue_.FreeTensor(srcUb);
}
template <typename T, uint64_t schId, uint64_t featuresBrc, uint64_t labelsBrc, uint64_t db>
__aicore__ inline void SoftmaxCrossEntropyWithLogitsSplitR<T, schId, featuresBrc, labelsBrc, db>::ComputeMaxLast(
    int32_t nNum, LocalTensor<T>& onceMaxUb)
{
    uint32_t nTimes = nNum;

    uint32_t vfLenfp32 = VL_FP32;
    uint16_t repeatTimes1 = CeilDivision(nTimes, vfLenfp32);
    LocalTensor<float> maxUb = maxBuf_.Get<float>();

    auto maxUbAddr = (__ubuf__ float*)maxUb.GetPhyAddr();
    auto maxUbOnceAddr = (__ubuf__ T*)onceMaxUb.GetPhyAddr();
    auto maxUbOnceAddrB32 = (__ubuf__ float*)onceMaxUb.GetPhyAddr();
    __VEC_SCOPE__
    {
        AscendC::Reg::MaskReg preg;
        AscendC::Reg::RegTensor<float> srcReg1;
        AscendC::Reg::RegTensor<float> srcReg2;
        AscendC::Reg::RegTensor<T> srcReg2B16;
        AscendC::Reg::RegTensor<float> srcReg3;

        for (uint16_t j = 0; j < repeatTimes1; j++) {
            preg = AscendC::Reg::UpdateMask<float>(nTimes);
            AscendC::Reg::AddrReg srcOffset = AscendC::Reg::CreateAddrReg<float>(j, vfLenfp32);
            AscendC::Reg::LoadAlign(srcReg1, maxUbAddr, srcOffset);
            if constexpr (sizeof(T) == 4) {
                AscendC::Reg::LoadAlign(srcReg2, maxUbOnceAddrB32, srcOffset);
            } else {
                AscendC::Reg::AddrReg srcOffset1 = AscendC::Reg::CreateAddrReg<T>(j, vfLenfp32);
                AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(srcReg2B16, maxUbOnceAddr,
                                                                                    srcOffset1);
                AscendC::Reg::Cast<float, T, castB16ToB32>(srcReg2, srcReg2B16, preg);
            }
            AscendC::Reg::Max(srcReg3, srcReg1, srcReg2, preg);
            AscendC::Reg::StoreAlign(maxUbAddr, srcReg3, srcOffset, preg);
        }
    }
}
template <typename T, uint64_t schId, uint64_t featuresBrc, uint64_t labelsBrc, uint64_t db>
__aicore__ inline void SoftmaxCrossEntropyWithLogitsSplitR<T, schId, featuresBrc, labelsBrc, db>::GetRowMax(
    int32_t nTailNum, int64_t aOffset)
{
    LocalTensor<T> onceMaxUb = temp1Buf_.Get<T>();
    int32_t ctimes = static_cast<int32_t>(rLoopTime);
    for (int32_t ci = 0; ci < ctimes; ++ci) {
        int32_t onceR = ci == ctimes - 1 ? int32_t(rLoopTile) : int32_t(rUbNumFactor);
        int32_t rTileAlign = rLoopTileAlign;
        int32_t alignR = ci == ctimes - 1 ? int32_t(rTileAlign) : int32_t(rUbNumFactor);
        int64_t offset = aOffset * r + rUbNumFactor * ci;
        if constexpr (featuresBrc != 0) {
            T padValue = -3.402823466e+38;
            if constexpr (IsSameType<T, half>::value) {
                padValue = -65504;
            }
            int64_t nddmaOffset = featuresBrcDim == 1 ? aOffset : rUbNumFactor * ci;
            DataCopyFeatureNDDMA(nTailNum, nddmaOffset, onceR, alignR, padValue);
        } else {
            DataCopyFeaturePad(nTailNum, onceR, alignR, offset);
        }
        LocalTensor<T> featureUb = featuresQueue_.DeQue<T>();
        uint32_t srcShape[2] = {static_cast<uint32_t>(nTailNum), static_cast<uint32_t>(alignR)};
        AscendC::ReduceMax<T, AscendC::Pattern::Reduce::AR, true>(onceMaxUb, featureUb, srcShape, false);
        featuresQueue_.FreeTensor(featureUb);
        ComputeMaxLast(nTailNum, onceMaxUb);
    }
}

template <typename T, uint64_t schId, uint64_t featuresBrc, uint64_t labelsBrc, uint64_t db>
__aicore__ inline void SoftmaxCrossEntropyWithLogitsSplitR<T, schId, featuresBrc, labelsBrc, db>::CleanMaxUb(
    int32_t tailNSize, LocalTensor<float>& clearUb)
{
    uint32_t nSize = tailNSize;
    float minValueFp32_ = static_cast<float>(-3.402823466e+38); // fp32可以表示的最小值
    int32_t vfLen = VL_FP32;
    uint16_t repeatTimes1 = CeilDivision(nSize, vfLen);
    auto maxUbAddr = (__ubuf__ float*)clearUb.GetPhyAddr();

    __VEC_SCOPE__
    {
        AscendC::Reg::MaskReg preg;
        AscendC::Reg::RegTensor<float> srcReg0;
        AscendC::Reg::RegTensor<float> srcReg1;
        for (uint16_t j = 0; j < repeatTimes1; j++) {
            preg = AscendC::Reg::UpdateMask<float>(nSize);
            AscendC::Reg::AddrReg srcOffset = AscendC::Reg::CreateAddrReg<float>(j, vfLen);
            AscendC::Reg::Duplicate(srcReg0, minValueFp32_);
            AscendC::Reg::StoreAlign(maxUbAddr, srcReg0, srcOffset, preg);
        }
    }
}

template <typename T, uint64_t schId, uint64_t featuresBrc, uint64_t labelsBrc, uint64_t db>
__aicore__ inline void SoftmaxCrossEntropyWithLogitsSplitR<T, schId, featuresBrc, labelsBrc, db>::UpdateCache(
    LocalTensor<float>& srcUb, int32_t index, int32_t dimA)
{
    const int64_t cacheID = ScalarGetCountOfValue<1>(index ^ (index + 1)) - 1;
    int32_t elementOneRepeat = VL_FP32;
    uint16_t outerLoopTimes = CeilDivision(dimA, elementOneRepeat);
    int32_t stride = outerLoopTimes * elementOneRepeat;

    LocalTensor<float> cacheUb = cacheBuf_.Get<float>();
    auto dstUbAddr = (__ubuf__ float*)cacheUb.GetPhyAddr();
    auto cahUbAddr = (__ubuf__ float*)cacheUb.GetPhyAddr() + cacheID * stride;
    auto srcUbAddr = (__ubuf__ float*)srcUb.GetPhyAddr();

    const uint16_t innerLoopTimes = static_cast<uint16_t>(cacheID);
    uint32_t sreg = dimA;

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<float> aReg, bReg;
        AscendC::Reg::MaskReg pMask;
        for (uint16_t i = 0; i < outerLoopTimes; ++i) {
            pMask = AscendC::Reg::UpdateMask<float>(sreg);
            AscendC::Reg::AddrReg srcOffset = AscendC::Reg::CreateAddrReg<float>(i, elementOneRepeat);
            AscendC::Reg::LoadAlign(aReg, srcUbAddr, srcOffset);
            for (uint16_t j = 0; j < innerLoopTimes; ++j) {
                AscendC::Reg::AddrReg srcOffsetJ = AscendC::Reg::CreateAddrReg<float>(i, elementOneRepeat, j, stride);
                AscendC::Reg::LoadAlign(bReg, dstUbAddr, srcOffsetJ);
                AscendC::Reg::Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(aReg, aReg, bReg, pMask);
            }
            AscendC::Reg::StoreAlign(cahUbAddr, aReg, srcOffset, pMask);
        }
    }
}

template <typename T, uint64_t schId, uint64_t featuresBrc, uint64_t labelsBrc, uint64_t db>
__aicore__ inline void SoftmaxCrossEntropyWithLogitsSplitR<T, schId, featuresBrc, labelsBrc,
                                                           db>::ComputeSubExpMainBlock(int32_t nSize, int32_t rOnceNum,
                                                                                       LocalTensor<float>& outUb)
{
    LocalTensor<T> featureUb = featuresQueue_.DeQue<T>();
    uint16_t nTimes = nSize;
    uint32_t vfLen = VL_FP32;

    uint16_t repeatTimes1 = rOnceNum / vfLen;
    LocalTensor<float> maxUb = maxBuf_.Get<float>();

    auto outUbAddr = (__ubuf__ float*)outUb.GetPhyAddr();
    auto inputUbAddr = (__ubuf__ T*)featureUb.GetPhyAddr();
    auto maxUbAddr = (__ubuf__ float*)maxUb.GetPhyAddr();
    __VEC_SCOPE__
    {
        AscendC::Reg::MaskReg copyOutReg = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::RegTensor<float> maxReg;
        AscendC::Reg::RegTensor<T> srcReg0;
        AscendC::Reg::RegTensor<float> srcRegfp32;
        AscendC::Reg::RegTensor<float> subReg;
        AscendC::Reg::RegTensor<float> expReg;
        for (uint16_t i = 0; i < nTimes; i++) {
            AscendC::Reg::LoadAlign<float, LoadDist::DIST_BRC_B32>(
                maxReg,
                maxUbAddr + i); // mask通过data_copy的brc功能实现broadcast
            for (uint16_t j = 0; j < repeatTimes1; j++) {
                AscendC::Reg::AddrReg outOffset = AscendC::Reg::CreateAddrReg<float>(i, rOnceNum, j, vfLen);
                if constexpr (sizeof(T) == 2) {
                    AscendC::Reg::AddrReg srcOffset = AscendC::Reg::CreateAddrReg<T>(i, rOnceNum, j, vfLen);
                    AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(srcReg0, inputUbAddr,
                                                                                        srcOffset);
                    AscendC::Reg::Cast<float, T, castB16ToB32>(srcRegfp32, srcReg0, copyOutReg);
                } else {
                    AscendC::Reg::AddrReg srcOffset1 = AscendC::Reg::CreateAddrReg<float>(i, rOnceNum, j, vfLen);
                    AscendC::Reg::LoadAlign(srcRegfp32, inputUbAddr, srcOffset1);
                }
                AscendC::Reg::Sub(subReg, srcRegfp32, maxReg, copyOutReg);
                AscendC::Reg::Exp(expReg, subReg, copyOutReg);
                AscendC::Reg::StoreAlign(outUbAddr, expReg, outOffset, copyOutReg);
            }
        }
    }
    featuresQueue_.FreeTensor(featureUb);
}

template <typename T, uint64_t schId, uint64_t featuresBrc, uint64_t labelsBrc, uint64_t db>
__aicore__ inline void
SoftmaxCrossEntropyWithLogitsSplitR<T, schId, featuresBrc, labelsBrc, db>::ComputeSubExpTailBlock(
    int32_t nSize, int32_t rOnceNumAlign, int32_t rOnceNumT, int32_t rOnceNumTAlign, LocalTensor<float>& outUb)
{
    uint16_t nTimes = nSize;
    uint32_t vfLen = VL_FP32;

    uint16_t repeatTimes1 = rOnceNumT / vfLen;
    uint32_t tailNum = rOnceNumT % vfLen;
    uint16_t tailLoopTimes = tailNum != 0 ? 1 : 0;
    LocalTensor<float> maxUb = maxBuf_.Get<float>();

    LocalTensor<T> featureUbT = featuresQueue_.DeQue<T>();

    auto outUbAddr = (__ubuf__ float*)outUb.GetPhyAddr();
    auto inputUbAddr = (__ubuf__ T*)featureUbT.GetPhyAddr();
    auto maxUbAddr = (__ubuf__ float*)maxUb.GetPhyAddr();
    __VEC_SCOPE__
    {
        AscendC::Reg::MaskReg copyOutReg = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::RegTensor<float> maxReg;
        AscendC::Reg::RegTensor<T> srcReg0;
        AscendC::Reg::RegTensor<float> srcRegfp32;
        AscendC::Reg::RegTensor<float> subReg;
        AscendC::Reg::RegTensor<float> expReg;
        AscendC::Reg::RegTensor<float> outReg;
        AscendC::Reg::RegTensor<float> outReg1;
        AscendC::Reg::MaskReg preg = AscendC::Reg::UpdateMask<float>(tailNum);
        for (uint16_t i = 0; i < nTimes; i++) {
            AscendC::Reg::LoadAlign<float, LoadDist::DIST_BRC_B32>(maxReg, maxUbAddr + i);
            for (uint16_t j = 0; j < repeatTimes1; j++) {
                AscendC::Reg::AddrReg srcOffset = AscendC::Reg::CreateAddrReg<T>(i, rOnceNumTAlign, j, vfLen);
                AscendC::Reg::AddrReg outOffset = AscendC::Reg::CreateAddrReg<float>(i, rOnceNumAlign, j, vfLen);
                if constexpr (sizeof(T) == 2) {
                    AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(srcReg0, inputUbAddr,
                                                                                        srcOffset);
                    AscendC::Reg::Cast<float, T, castB16ToB32>(srcRegfp32, srcReg0, copyOutReg);
                } else {
                    AscendC::Reg::LoadAlign(srcRegfp32, inputUbAddr, srcOffset);
                }
                AscendC::Reg::Sub(subReg, srcRegfp32, maxReg, copyOutReg);
                AscendC::Reg::Exp(expReg, subReg, copyOutReg);
                AscendC::Reg::LoadAlign(outReg, outUbAddr, outOffset);
                AscendC::Reg::Add(outReg1, expReg, outReg, copyOutReg);
                AscendC::Reg::StoreAlign(outUbAddr, outReg1, outOffset, copyOutReg);
            }
            for (uint16_t k = 0; k < tailLoopTimes; k++) {
                if constexpr (sizeof(T) == 2) {
                    AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
                        srcReg0, inputUbAddr + i * rOnceNumTAlign + repeatTimes1 * vfLen);
                    AscendC::Reg::Cast<float, T, castB16ToB32>(srcRegfp32, srcReg0, preg);
                } else {
                    AscendC::Reg::LoadAlign(srcRegfp32, inputUbAddr + i * rOnceNumTAlign + repeatTimes1 * vfLen);
                }
                AscendC::Reg::Sub(subReg, srcRegfp32, maxReg, preg);
                AscendC::Reg::Exp(expReg, subReg, preg);
                AscendC::Reg::LoadAlign(outReg, outUbAddr + i * rOnceNumAlign + repeatTimes1 * vfLen);
                AscendC::Reg::Add(outReg1, expReg, outReg, preg);
                AscendC::Reg::StoreAlign(outUbAddr + i * rOnceNumAlign + repeatTimes1 * vfLen, outReg1, preg);
            }
        }
    }
    featuresQueue_.FreeTensor(featureUbT);
}

template <typename T, uint64_t schId, uint64_t featuresBrc, uint64_t labelsBrc, uint64_t db>
__aicore__ inline void SoftmaxCrossEntropyWithLogitsSplitR<T, schId, featuresBrc, labelsBrc,
                                                           db>::ComputeSubExpAlignBlock(int32_t nSize, int32_t rOnceNum,
                                                                                        int32_t rOnceNumAlign,
                                                                                        LocalTensor<float>& outUb)
{
    uint16_t nTimes = nSize;
    uint32_t vfLen = VL_FP32;

    uint16_t repeatTimes1 = rOnceNum / vfLen;
    uint32_t tailNum = rOnceNum % vfLen;
    uint32_t tailNumAlign = rOnceNumAlign - repeatTimes1 * vfLen;
    uint16_t tailLoopTimes = tailNum != 0 ? 1 : 0;
    LocalTensor<float> maxUb = maxBuf_.Get<float>();
    LocalTensor<T> featureUb = featuresQueue_.DeQue<T>();

    auto outUbAddr = (__ubuf__ float*)outUb.GetPhyAddr();
    auto inputUbAddr = (__ubuf__ T*)featureUb.GetPhyAddr();
    auto maxUbAddr = (__ubuf__ float*)maxUb.GetPhyAddr();
    __VEC_SCOPE__
    {
        AscendC::Reg::MaskReg copyOutReg = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::RegTensor<float> maxReg;
        AscendC::Reg::RegTensor<T> srcReg0;
        AscendC::Reg::RegTensor<float> srcRegfp32;
        AscendC::Reg::RegTensor<float> subReg;
        AscendC::Reg::RegTensor<float> expReg;
        AscendC::Reg::MaskReg preg = AscendC::Reg::UpdateMask<float>(tailNum);
        AscendC::Reg::MaskReg preg1 = AscendC::Reg::UpdateMask<float>(tailNumAlign);
        for (uint16_t i = 0; i < nTimes; i++) {
            AscendC::Reg::LoadAlign<float, LoadDist::DIST_BRC_B32>(
                maxReg,
                maxUbAddr + i); // mask通过data_copy的brc功能实现broadcast
            for (uint16_t j = 0; j < repeatTimes1; j++) {
                AscendC::Reg::AddrReg srcOffset = AscendC::Reg::CreateAddrReg<T>(i, rOnceNumAlign, j, vfLen);
                AscendC::Reg::AddrReg outOffset = AscendC::Reg::CreateAddrReg<float>(i, rOnceNumAlign, j, vfLen);
                if constexpr (sizeof(T) == 2) {
                    AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(srcReg0, inputUbAddr,
                                                                                        srcOffset);
                    AscendC::Reg::Cast<float, T, castB16ToB32>(srcRegfp32, srcReg0, copyOutReg);
                } else {
                    AscendC::Reg::LoadAlign(srcRegfp32, inputUbAddr, srcOffset);
                }
                AscendC::Reg::Sub(subReg, srcRegfp32, maxReg, copyOutReg);
                AscendC::Reg::Exp(expReg, subReg, copyOutReg);
                AscendC::Reg::StoreAlign(outUbAddr, expReg, outOffset, copyOutReg);
            }
            for (uint16_t k = 0; k < tailLoopTimes; k++) {
                if constexpr (sizeof(T) == 2) {
                    AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
                        srcReg0, inputUbAddr + i * rOnceNumAlign + repeatTimes1 * vfLen);
                    AscendC::Reg::Cast<float, T, castB16ToB32>(srcRegfp32, srcReg0, preg);
                } else {
                    AscendC::Reg::LoadAlign(srcRegfp32, inputUbAddr + i * rOnceNumAlign + repeatTimes1 * vfLen);
                }
                AscendC::Reg::Sub(subReg, srcRegfp32, maxReg, preg);
                AscendC::Reg::Exp(expReg, subReg, preg);
                AscendC::Reg::StoreAlign(outUbAddr + i * rOnceNumAlign + repeatTimes1 * vfLen, expReg, preg1);
            }
        }
    }
    featuresQueue_.FreeTensor(featureUb);
}

template <typename T, uint64_t schId, uint64_t featuresBrc, uint64_t labelsBrc, uint64_t db>
__aicore__ inline void
SoftmaxCrossEntropyWithLogitsSplitR<T, schId, featuresBrc, labelsBrc, db>::ComputeSubExpReduceSum(int32_t tailNum,
                                                                                                  int32_t aOffset)
{
    LocalTensor<float> outUbFp32 = lossQueue_.AllocTensor<float>();
    LocalTensor<float> cacheUb = cacheBuf_.Get<float>();
    int32_t tail = static_cast<int32_t>(kTimesTail);
    int32_t loopTimes = static_cast<int32_t>(kTimes);
    T padValue = 0;
    for (int32_t i = 0; i < tail; ++i) {
        LocalTensor<float> sumUb = temp1Buf_.Get<float>();
        int32_t once = static_cast<int32_t>(rUbNumFactor);
        int32_t tailC = i == tail - 1 ? static_cast<int32_t>(rLoopTile) : once;
        int32_t tailAlign = rLoopTileAlign;
        int32_t align = i == tail - 1 ? static_cast<int32_t>(tailAlign) : once;
        int64_t offset = aOffset * r + once * i;
        if constexpr (featuresBrc != 0) {
            int64_t nddmaOffset = featuresBrcDim == 1 ? aOffset : rUbNumFactor * i;
            DataCopyFeatureNDDMA(tailNum, nddmaOffset, once, once, padValue);
        } else {
            DataCopyFeature(tailNum, once, offset);
        }
        ComputeSubExpMainBlock(tailNum, once, outUbFp32);

        int64_t offset1 = aOffset * r + once * (i + kTimes);

        if constexpr (featuresBrc != 0) {
            int64_t nddmaOffset1 = featuresBrcDim == 1 ? aOffset : once * (i + kTimes);
            DataCopyFeatureNDDMA(tailNum, nddmaOffset1, tailC, align, padValue);
        } else {
            DataCopyFeature(tailNum, tailC, offset1);
        }
        ComputeSubExpTailBlock(tailNum, once, tailC, align, outUbFp32);

        uint32_t srcShape[2] = {static_cast<uint32_t>(tailNum), static_cast<uint32_t>(rUbNumFactor)};
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(sumUb, outUbFp32, srcShape, false);
        UpdateCache(sumUb, i, tailNum);
    }
    for (int32_t ci = tail; ci < loopTimes; ++ci) {
        LocalTensor<float> sumUb = temp1Buf_.Get<float>();
        int32_t tailC = static_cast<int32_t>(rUbNumFactor);
        int32_t align = static_cast<int32_t>(rUbNumFactor);
        if ((ci == loopTimes - 1) && tail == 0) {
            tailC = static_cast<int32_t>(rLoopTile);
            align = static_cast<int32_t>(rLoopTileAlign);
        }
        int64_t offset = aOffset * r + rUbNumFactor * ci;
        if constexpr (featuresBrc != 0) {
            int64_t nddmaOffset = featuresBrcDim == 1 ? aOffset : rUbNumFactor * ci;
            DataCopyFeatureNDDMA(tailNum, nddmaOffset, tailC, align, padValue);
        } else {
            DataCopyFeature(tailNum, tailC, offset);
        }
        ComputeSubExpAlignBlock(tailNum, tailC, align, outUbFp32);
        uint32_t srcShape[2] = {static_cast<uint32_t>(tailNum), static_cast<uint32_t>(align)};
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(sumUb, outUbFp32, srcShape, false);
        UpdateCache(sumUb, ci, tailNum);
    }
    lossQueue_.FreeTensor(outUbFp32);
    ComputeLog(tailNum, cacheUb);
}

template <typename T, uint64_t schId, uint64_t featuresBrc, uint64_t labelsBrc, uint64_t db>
__aicore__ inline void SoftmaxCrossEntropyWithLogitsSplitR<T, schId, featuresBrc, labelsBrc, db>::ComputeLog(
    int32_t nSize, LocalTensor<float>& extraUb)
{
    uint32_t nSize1 = nSize;
    uint32_t vfLen = VL_FP32;
    uint16_t repeatTimes1 = CeilDivision(nSize, vfLen);
    LocalTensor<float> logUb = logBuf_.Get<float>();
    LocalTensor<float> sumUb = sumBuf_.Get<float>();
    auto logUbAddr = (__ubuf__ float*)logUb.GetPhyAddr();
    auto sumUbAddr = (__ubuf__ float*)sumUb.GetPhyAddr();
    int32_t cacheStart = repeatTimes1 * vfLen * updateStart_;
    auto cacheUbAddr = (__ubuf__ float*)extraUb.GetPhyAddr() + cacheStart;
    __VEC_SCOPE__
    {
        AscendC::Reg::MaskReg preg;
        AscendC::Reg::RegTensor<float> srcReg0;
        AscendC::Reg::RegTensor<float> logReg0;
        AscendC::Reg::RegTensor<float> sumReg0;
        AscendC::Reg::RegTensor<float> addReg0;
        for (uint16_t j = 0; j < repeatTimes1; j++) {
            preg = AscendC::Reg::UpdateMask<float>(nSize1);
            AscendC::Reg::AddrReg srcOffset = AscendC::Reg::CreateAddrReg<float>(j, vfLen);
            AscendC::Reg::LoadAlign(srcReg0, cacheUbAddr, srcOffset);
            AscendC::Reg::Log(logReg0, srcReg0, preg);
            AscendC::Reg::Move(sumReg0, srcReg0, preg);
            AscendC::Reg::StoreAlign(logUbAddr, logReg0, srcOffset, preg);
            AscendC::Reg::StoreAlign(sumUbAddr, sumReg0, srcOffset, preg);
        }
    }
}

template <typename T, uint64_t schId, uint64_t featuresBrc, uint64_t labelsBrc, uint64_t db>
__aicore__ inline void SoftmaxCrossEntropyWithLogitsSplitR<T, schId, featuresBrc, labelsBrc,
                                                           db>::ComputeLossAndBropMainBlock(int32_t nSize,
                                                                                            int32_t rOnceNum,
                                                                                            LocalTensor<float>& outUb)
{
    uint16_t nTimes = nSize;
    uint32_t vfLen = VL_FP32;

    uint16_t repeatTimes1 = rOnceNum / vfLen;
    LocalTensor<T> backPropUb = backPropQueue_.AllocTensor<T>();
    LocalTensor<float> sumUb = sumBuf_.Get<float>();
    LocalTensor<float> maxUb = maxBuf_.Get<float>();
    LocalTensor<float> logUb = logBuf_.Get<float>();
    LocalTensor<T> featureUb = featuresQueue_.DeQue<T>();
    LocalTensor<T> labelsUb = labelsQueue_.DeQue<T>();

    auto outUbAddr = (__ubuf__ float*)outUb.GetPhyAddr();
    auto inputUbAddr = (__ubuf__ T*)featureUb.GetPhyAddr();
    auto labelsUbAddr = (__ubuf__ T*)labelsUb.GetPhyAddr();
    auto maxUbAddr = (__ubuf__ float*)maxUb.GetPhyAddr();
    auto sumUbAddr = (__ubuf__ float*)sumUb.GetPhyAddr();
    auto logUbAddr = (__ubuf__ float*)logUb.GetPhyAddr();
    auto backProbAddr = (__ubuf__ T*)backPropUb.GetPhyAddr();
    __VEC_SCOPE__
    {
        AscendC::Reg::MaskReg copyOutReg = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::RegTensor<float> maxReg;
        AscendC::Reg::RegTensor<float> sumReg;
        AscendC::Reg::RegTensor<float> logReg;
        AscendC::Reg::RegTensor<T> srcReg0;
        AscendC::Reg::RegTensor<T> srcReg1;
        AscendC::Reg::RegTensor<float> srcRegfp32;
        AscendC::Reg::RegTensor<float> srcReg1fp32;
        AscendC::Reg::RegTensor<float> subReg;
        AscendC::Reg::RegTensor<float> expReg;
        AscendC::Reg::RegTensor<float> tmpReg;

        AscendC::Reg::RegTensor<T> backProbReg;
        for (uint16_t i = 0; i < nTimes; i++) {
            AscendC::Reg::LoadAlign<float, LoadDist::DIST_BRC_B32>(
                maxReg,
                maxUbAddr + i); // mask通过data_copy的brc功能实现broadcast
            AscendC::Reg::LoadAlign<float, LoadDist::DIST_BRC_B32>(
                sumReg,
                sumUbAddr + i); // mask通过data_copy的brc功能实现broadcast
            AscendC::Reg::LoadAlign<float, LoadDist::DIST_BRC_B32>(
                logReg,
                logUbAddr + i); // mask通过data_copy的brc功能实现broadcast
            for (uint16_t j = 0; j < repeatTimes1; j++) {
                AscendC::Reg::AddrReg srcOffset = AscendC::Reg::CreateAddrReg<T>(i, rOnceNum, j, vfLen);
                AscendC::Reg::AddrReg outOffset = AscendC::Reg::CreateAddrReg<float>(i, rOnceNum, j, vfLen);
                if constexpr (sizeof(T) == 2) {
                    AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(srcReg0, inputUbAddr,
                                                                                        srcOffset);
                    AscendC::Reg::Cast<float, T, castB16ToB32>(srcRegfp32, srcReg0, copyOutReg);
                    AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(srcReg1, labelsUbAddr,
                                                                                        srcOffset);
                    AscendC::Reg::Cast<float, T, castB16ToB32>(srcReg1fp32, srcReg1, copyOutReg);
                } else {
                    AscendC::Reg::LoadAlign(srcRegfp32, inputUbAddr, srcOffset);
                    AscendC::Reg::LoadAlign(srcReg1fp32, labelsUbAddr, srcOffset);
                }
                AscendC::Reg::Sub(subReg, srcRegfp32, maxReg, copyOutReg);
                AscendC::Reg::Exp(expReg, subReg, copyOutReg);
                AscendC::Reg::Div(expReg, expReg, sumReg, copyOutReg);
                AscendC::Reg::Sub(expReg, expReg, srcReg1fp32, copyOutReg);
                AscendC::Reg::Sub(tmpReg, logReg, subReg, copyOutReg);
                AscendC::Reg::Mul(tmpReg, tmpReg, srcReg1fp32, copyOutReg);
                AscendC::Reg::StoreAlign(outUbAddr, tmpReg, outOffset, copyOutReg);
                if constexpr (sizeof(T) == 2) {
                    AscendC::Reg::AddrReg outOffset1 = AscendC::Reg::CreateAddrReg<T>(i, rOnceNum, j, vfLen);
                    AscendC::Reg::Cast<T, float, castB32ToB16>(backProbReg, expReg, copyOutReg);
                    AscendC::Reg::StoreAlign<T, StoreDist::DIST_PACK_B32>(backProbAddr, backProbReg, outOffset1,
                                                                          copyOutReg);
                } else {
                    AscendC::Reg::StoreAlign(backProbAddr, expReg, outOffset, copyOutReg);
                }
            }
        }
    }
    featuresQueue_.FreeTensor(featureUb);
    labelsQueue_.FreeTensor(labelsUb);
    backPropQueue_.EnQue<T>(backPropUb);
}

template <typename T, uint64_t schId, uint64_t featuresBrc, uint64_t labelsBrc, uint64_t db>
__aicore__ inline void
SoftmaxCrossEntropyWithLogitsSplitR<T, schId, featuresBrc, labelsBrc, db>::ComputeLossAndBropTailBlock(
    int32_t nSize, int32_t rOnceNumAlign, int32_t rOnceNumT, int32_t rOnceNumTAlign, LocalTensor<float>& outUb)
{
    LocalTensor<T> backPropUb = backPropQueue_.AllocTensor<T>();
    LocalTensor<T> featureUb = featuresQueue_.DeQue<T>();
    LocalTensor<T> labelsUb = labelsQueue_.DeQue<T>();
    LocalTensor<float> sumUb = sumBuf_.Get<float>();
    LocalTensor<float> maxUb = maxBuf_.Get<float>();
    LocalTensor<float> logUb = logBuf_.Get<float>();

    uint16_t nTimes = nSize;
    uint32_t vfLen = VL_FP32;

    uint16_t repeatTimes1 = rOnceNumT / vfLen;
    uint32_t tailNum = rOnceNumT % vfLen;
    uint32_t tailNumAlign = rOnceNumTAlign - repeatTimes1 * vfLen;
    uint16_t tailLoopTimes = tailNum != 0 ? 1 : 0;

    auto outUbAddr = (__ubuf__ float*)outUb.GetPhyAddr();
    auto inputUbAddr = (__ubuf__ T*)featureUb.GetPhyAddr();
    auto labelsUbAddr = (__ubuf__ T*)labelsUb.GetPhyAddr();
    auto maxUbAddr = (__ubuf__ float*)maxUb.GetPhyAddr();
    auto sumUbAddr = (__ubuf__ float*)sumUb.GetPhyAddr();
    auto logUbAddr = (__ubuf__ float*)logUb.GetPhyAddr();
    auto backProbAddr = (__ubuf__ T*)backPropUb.GetPhyAddr();
    __VEC_SCOPE__
    {
        AscendC::Reg::MaskReg copyOutReg = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::RegTensor<float> maxReg;
        AscendC::Reg::RegTensor<float> sumReg;
        AscendC::Reg::RegTensor<float> logReg;
        AscendC::Reg::RegTensor<T> srcReg0;
        AscendC::Reg::RegTensor<T> srcReg1;
        AscendC::Reg::RegTensor<float> srcRegfp32;
        AscendC::Reg::RegTensor<float> srcReg1fp32;
        AscendC::Reg::RegTensor<float> subReg;
        AscendC::Reg::RegTensor<float> expReg;
        AscendC::Reg::RegTensor<float> tmpReg;

        AscendC::Reg::RegTensor<float> outReg;
        AscendC::Reg::RegTensor<float> outReg1;
        AscendC::Reg::RegTensor<T> backPropReg;
        AscendC::Reg::MaskReg preg = AscendC::Reg::UpdateMask<float>(tailNum);
        for (uint16_t i = 0; i < nTimes; i++) {
            AscendC::Reg::LoadAlign<float, LoadDist::DIST_BRC_B32>(
                maxReg,
                maxUbAddr + i); // mask通过data_copy的brc功能实现broadcast
            AscendC::Reg::LoadAlign<float, LoadDist::DIST_BRC_B32>(
                sumReg,
                sumUbAddr + i); // mask通过data_copy的brc功能实现broadcast
            AscendC::Reg::LoadAlign<float, LoadDist::DIST_BRC_B32>(
                logReg,
                logUbAddr + i); // mask通过data_copy的brc功能实现broadcast
            for (uint16_t j = 0; j < repeatTimes1; j++) {
                AscendC::Reg::AddrReg srcOffset = AscendC::Reg::CreateAddrReg<T>(i, rOnceNumTAlign, j, vfLen);
                AscendC::Reg::AddrReg outOffset = AscendC::Reg::CreateAddrReg<float>(i, rOnceNumAlign, j, vfLen);
                if constexpr (sizeof(T) == 2) {
                    AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(srcReg0, inputUbAddr,
                                                                                        srcOffset);
                    AscendC::Reg::Cast<float, T, castB16ToB32>(srcRegfp32, srcReg0, copyOutReg);
                    AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(srcReg1, labelsUbAddr,
                                                                                        srcOffset);
                    AscendC::Reg::Cast<float, T, castB16ToB32>(srcReg1fp32, srcReg1, copyOutReg);
                } else {
                    AscendC::Reg::LoadAlign(srcRegfp32, inputUbAddr, srcOffset);
                    AscendC::Reg::LoadAlign(srcReg1fp32, labelsUbAddr, srcOffset);
                }
                AscendC::Reg::Sub(subReg, srcRegfp32, maxReg, copyOutReg);
                AscendC::Reg::Exp(expReg, subReg, copyOutReg);
                AscendC::Reg::Div(expReg, expReg, sumReg, copyOutReg);
                AscendC::Reg::Sub(expReg, expReg, srcReg1fp32, copyOutReg);
                AscendC::Reg::Sub(tmpReg, logReg, subReg, copyOutReg);
                AscendC::Reg::Mul(tmpReg, tmpReg, srcReg1fp32, copyOutReg);
                AscendC::Reg::LoadAlign(outReg, outUbAddr, outOffset);
                AscendC::Reg::Add(outReg1, tmpReg, outReg, copyOutReg);
                AscendC::Reg::StoreAlign(outUbAddr, outReg1, outOffset, copyOutReg);
                if constexpr (sizeof(T) == 2) {
                    AscendC::Reg::Cast<T, float, castB32ToB16>(backPropReg, expReg, copyOutReg);
                    AscendC::Reg::StoreAlign<T, StoreDist::DIST_PACK_B32>(backProbAddr, backPropReg, srcOffset,
                                                                          copyOutReg);
                } else {
                    AscendC::Reg::StoreAlign(backProbAddr, expReg, srcOffset, copyOutReg);
                }
            }
            for (uint16_t k = 0; k < tailLoopTimes; k++) {
                if constexpr (sizeof(T) == 2) {
                    AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
                        srcReg0, inputUbAddr + i * rOnceNumTAlign + repeatTimes1 * vfLen);
                    AscendC::Reg::Cast<float, T, castB16ToB32>(srcRegfp32, srcReg0, preg);
                    AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
                        srcReg1, labelsUbAddr + i * rOnceNumTAlign + repeatTimes1 * vfLen);
                    AscendC::Reg::Cast<float, T, castB16ToB32>(srcReg1fp32, srcReg1, preg);
                } else {
                    AscendC::Reg::LoadAlign(srcRegfp32, inputUbAddr + i * rOnceNumTAlign + repeatTimes1 * vfLen);
                    AscendC::Reg::LoadAlign(srcReg1fp32, labelsUbAddr + i * rOnceNumTAlign + repeatTimes1 * vfLen);
                }
                AscendC::Reg::Sub(subReg, srcRegfp32, maxReg, preg);
                AscendC::Reg::Exp(expReg, subReg, preg);
                AscendC::Reg::Div(expReg, expReg, sumReg, preg);
                AscendC::Reg::Sub(expReg, expReg, srcReg1fp32, preg);
                AscendC::Reg::Sub(logReg, logReg, subReg, preg);
                AscendC::Reg::Mul(logReg, logReg, srcReg1fp32, preg);
                AscendC::Reg::LoadAlign(outReg, outUbAddr + i * rOnceNumAlign + repeatTimes1 * vfLen);
                AscendC::Reg::Add(outReg1, logReg, outReg, preg);
                AscendC::Reg::StoreAlign(outUbAddr + i * rOnceNumAlign + repeatTimes1 * vfLen, outReg1, preg);
                if constexpr (sizeof(T) == 2) {
                    AscendC::Reg::Cast<T, float, castB32ToB16>(backPropReg, expReg, preg);
                    AscendC::Reg::StoreAlign<T, StoreDist::DIST_PACK_B32>(
                        backProbAddr + i * rOnceNumTAlign + repeatTimes1 * vfLen, backPropReg, preg);
                } else {
                    AscendC::Reg::StoreAlign(backProbAddr + i * rOnceNumTAlign + repeatTimes1 * vfLen, expReg, preg);
                }
            }
        }
    }
    featuresQueue_.FreeTensor(featureUb);
    labelsQueue_.FreeTensor(labelsUb);
    backPropQueue_.EnQue<T>(backPropUb);
}

template <typename T, uint64_t schId, uint64_t featuresBrc, uint64_t labelsBrc, uint64_t db>
__aicore__ inline void
SoftmaxCrossEntropyWithLogitsSplitR<T, schId, featuresBrc, labelsBrc, db>::ComputeLossAndBropAlignBlock(
    int32_t nSize, int32_t rOnceNum, int32_t rOnceNumAlign, LocalTensor<float>& outUb)
{
    LocalTensor<T> backPropUb = backPropQueue_.AllocTensor<T>();
    LocalTensor<float> sumUb = sumBuf_.Get<float>();
    LocalTensor<float> maxUb = maxBuf_.Get<float>();
    LocalTensor<float> logUb = logBuf_.Get<float>();
    LocalTensor<T> featureUb = featuresQueue_.DeQue<T>();
    LocalTensor<T> labelsUb = labelsQueue_.DeQue<T>();
    uint16_t nTimes = nSize;
    uint32_t vfLen = VL_FP32;

    uint16_t repeatTimes1 = rOnceNum / vfLen;
    uint32_t tailNum = rOnceNum % vfLen;
    uint32_t tailNumAlign = rOnceNumAlign - repeatTimes1 * vfLen;
    uint16_t tailLoopTimes = tailNum != 0 ? 1 : 0;

    auto outUbAddr = (__ubuf__ float*)outUb.GetPhyAddr();
    auto inputUbAddr = (__ubuf__ T*)featureUb.GetPhyAddr();
    auto labelsUbAddr = (__ubuf__ T*)labelsUb.GetPhyAddr();
    auto maxUbAddr = (__ubuf__ float*)maxUb.GetPhyAddr();
    auto sumUbAddr = (__ubuf__ float*)sumUb.GetPhyAddr();
    auto logUbAddr = (__ubuf__ float*)logUb.GetPhyAddr();
    auto backProbAddr = (__ubuf__ T*)backPropUb.GetPhyAddr();
    __VEC_SCOPE__
    {
        AscendC::Reg::MaskReg copyOutReg = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
        AscendC::Reg::RegTensor<float> maxReg;
        AscendC::Reg::RegTensor<float> sumReg;
        AscendC::Reg::RegTensor<float> logReg;
        AscendC::Reg::RegTensor<T> srcReg0;
        AscendC::Reg::RegTensor<T> srcReg1;
        AscendC::Reg::RegTensor<float> srcRegfp32;
        AscendC::Reg::RegTensor<float> srcReg1fp32;
        AscendC::Reg::RegTensor<float> subReg;
        AscendC::Reg::RegTensor<float> expReg;
        AscendC::Reg::RegTensor<float> tmpReg;
        AscendC::Reg::RegTensor<T> backProbReg;
        AscendC::Reg::MaskReg preg = AscendC::Reg::UpdateMask<float>(tailNum);
        AscendC::Reg::MaskReg preg1 = AscendC::Reg::UpdateMask<float>(tailNumAlign);
        for (uint16_t i = 0; i < nTimes; i++) {
            AscendC::Reg::LoadAlign<float, LoadDist::DIST_BRC_B32>(
                maxReg,
                maxUbAddr + i); // mask通过data_copy的brc功能实现broadcast
            AscendC::Reg::LoadAlign<float, LoadDist::DIST_BRC_B32>(
                sumReg,
                sumUbAddr + i); // mask通过data_copy的brc功能实现broadcast
            AscendC::Reg::LoadAlign<float, LoadDist::DIST_BRC_B32>(
                logReg,
                logUbAddr + i); // mask通过data_copy的brc功能实现broadcast
            for (uint16_t j = 0; j < repeatTimes1; j++) {
                AscendC::Reg::AddrReg srcOffset = AscendC::Reg::CreateAddrReg<T>(i, rOnceNumAlign, j, vfLen);
                AscendC::Reg::AddrReg outOffset = AscendC::Reg::CreateAddrReg<float>(i, rOnceNumAlign, j, vfLen);
                if constexpr (sizeof(T) == 2) {
                    AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(srcReg0, inputUbAddr,
                                                                                        srcOffset);
                    AscendC::Reg::Cast<float, T, castB16ToB32>(srcRegfp32, srcReg0, copyOutReg);
                    AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(srcReg1, labelsUbAddr,
                                                                                        srcOffset);
                    AscendC::Reg::Cast<float, T, castB16ToB32>(srcReg1fp32, srcReg1, copyOutReg);
                } else {
                    AscendC::Reg::LoadAlign(srcRegfp32, inputUbAddr, srcOffset);
                    AscendC::Reg::LoadAlign(srcReg1fp32, labelsUbAddr, srcOffset);
                }
                AscendC::Reg::Sub(subReg, srcRegfp32, maxReg, copyOutReg);
                AscendC::Reg::Exp(expReg, subReg, copyOutReg);
                AscendC::Reg::Div(expReg, expReg, sumReg, copyOutReg);
                AscendC::Reg::Sub(expReg, expReg, srcReg1fp32, copyOutReg);
                AscendC::Reg::Sub(tmpReg, logReg, subReg, copyOutReg);
                AscendC::Reg::Mul(tmpReg, tmpReg, srcReg1fp32, copyOutReg);
                AscendC::Reg::StoreAlign(outUbAddr, tmpReg, outOffset, copyOutReg);
                if constexpr (sizeof(T) == 2) {
                    AscendC::Reg::AddrReg outOffset1 = AscendC::Reg::CreateAddrReg<T>(i, rOnceNumAlign, j, vfLen);
                    AscendC::Reg::Cast<T, float, castB32ToB16>(backProbReg, expReg, copyOutReg);
                    AscendC::Reg::StoreAlign<T, StoreDist::DIST_PACK_B32>(backProbAddr, backProbReg, outOffset1,
                                                                          copyOutReg);
                } else {
                    AscendC::Reg::StoreAlign(backProbAddr, expReg, outOffset, copyOutReg);
                }
            }
            for (uint16_t k = 0; k < tailLoopTimes; k++) {
                if constexpr (sizeof(T) == 2) {
                    AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
                        srcReg0, inputUbAddr + i * rOnceNumAlign + repeatTimes1 * vfLen);
                    AscendC::Reg::Cast<float, T, castB16ToB32>(srcRegfp32, srcReg0, preg);
                    AscendC::Reg::LoadAlign<T, AscendC::Reg::LoadDist::DIST_UNPACK_B16>(
                        srcReg1, labelsUbAddr + i * rOnceNumAlign + repeatTimes1 * vfLen);
                    AscendC::Reg::Cast<float, T, castB16ToB32>(srcReg1fp32, srcReg1, preg);
                } else {
                    AscendC::Reg::LoadAlign(srcRegfp32, inputUbAddr + i * rOnceNumAlign + repeatTimes1 * vfLen);
                    AscendC::Reg::LoadAlign(srcReg1fp32, labelsUbAddr + i * rOnceNumAlign + repeatTimes1 * vfLen);
                }
                AscendC::Reg::Sub(subReg, srcRegfp32, maxReg, preg);
                AscendC::Reg::Exp(expReg, subReg, preg);
                AscendC::Reg::Div(expReg, expReg, sumReg, preg);
                AscendC::Reg::Sub(expReg, expReg, srcReg1fp32, preg);
                AscendC::Reg::Sub(tmpReg, logReg, subReg, preg);
                AscendC::Reg::Mul(tmpReg, tmpReg, srcReg1fp32, preg);
                AscendC::Reg::StoreAlign(outUbAddr + i * rOnceNumAlign + repeatTimes1 * vfLen, tmpReg, preg1);
                if constexpr (sizeof(T) == 2) {
                    AscendC::Reg::Cast<T, float, castB32ToB16>(backProbReg, expReg, preg);
                    AscendC::Reg::StoreAlign<T, StoreDist::DIST_PACK_B32>(
                        backProbAddr + i * rOnceNumAlign + repeatTimes1 * vfLen, backProbReg, preg);
                } else {
                    AscendC::Reg::StoreAlign(backProbAddr + i * rOnceNumAlign + repeatTimes1 * vfLen, expReg, preg1);
                }
            }
        }
    }
    featuresQueue_.FreeTensor(featureUb);
    labelsQueue_.FreeTensor(labelsUb);
    backPropQueue_.EnQue<T>(backPropUb);
}

template <typename T, uint64_t schId, uint64_t featuresBrc, uint64_t labelsBrc, uint64_t db>
__aicore__ inline void SoftmaxCrossEntropyWithLogitsSplitR<T, schId, featuresBrc, labelsBrc, db>::DataCopyOutLoss(
    int32_t nTailNum, int64_t offset)
{
    LocalTensor<T> lossUb = lossQueue_.DeQue<T>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = nTailNum * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;

    AscendC::DataCopyPad(lossGm_[offset], lossUb, copyParams);
    lossQueue_.FreeTensor(lossUb);
}
template <typename T, uint64_t schId, uint64_t featuresBrc, uint64_t labelsBrc, uint64_t db>
__aicore__ inline void SoftmaxCrossEntropyWithLogitsSplitR<T, schId, featuresBrc, labelsBrc, db>::CopyOutLoss(
    int32_t nSize, LocalTensor<float>& extraUb, int32_t aOffset)
{
    uint32_t nSize1 = nSize;
    uint32_t vfLen = VL_FP32;
    uint16_t repeatTimes1 = CeilDivision(nSize, vfLen);
    LocalTensor<T> lossUb = lossQueue_.AllocTensor<T>();
    int32_t cacheStart = repeatTimes1 * vfLen * updateStart_;
    auto cacheUbAddr = (__ubuf__ float*)extraUb.GetPhyAddr() + cacheStart;
    auto lossUbAddr = (__ubuf__ T*)lossUb.GetPhyAddr();
    __VEC_SCOPE__
    {
        AscendC::Reg::MaskReg preg;
        AscendC::Reg::RegTensor<float> srcReg0;
        AscendC::Reg::RegTensor<float> logReg0;
        AscendC::Reg::RegTensor<T> lossReg0;
        AscendC::Reg::RegTensor<float> addReg0;
        for (uint16_t j = 0; j < repeatTimes1; j++) {
            preg = AscendC::Reg::UpdateMask<float>(nSize1);
            AscendC::Reg::AddrReg srcOffset = AscendC::Reg::CreateAddrReg<float>(j, vfLen);
            AscendC::Reg::MaskReg regAllFp32 = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
            AscendC::Reg::LoadAlign(srcReg0, cacheUbAddr, srcOffset);
            if constexpr (sizeof(T) == 2) {
                AscendC::Reg::Cast<T, float, castB32ToB16>(lossReg0, srcReg0, preg);
                AscendC::Reg::StoreAlign<T, StoreDist::DIST_PACK_B32>(lossUbAddr, lossReg0, preg);
            } else {
                AscendC::Reg::Move(lossReg0, srcReg0, preg);
                AscendC::Reg::StoreAlign(lossUbAddr, lossReg0, srcOffset, preg);
            }
        }
    }
    lossQueue_.EnQue<T>(lossUb);
    DataCopyOutLoss(nSize, aOffset);
}

template <typename T, uint64_t schId, uint64_t featuresBrc, uint64_t labelsBrc, uint64_t db>
__aicore__ inline void SoftmaxCrossEntropyWithLogitsSplitR<T, schId, featuresBrc, labelsBrc, db>::ComputeRst(
    int32_t tailNum, int32_t aOffset)
{
    LocalTensor<float> lossUbFp32 = lossQueue_.AllocTensor<float>();
    T padValue = 0;
    LocalTensor<float> cacheUb = cacheBuf_.Get<float>();
    int32_t tail = static_cast<int32_t>(kTimesTail);
    int32_t loopTimes = static_cast<int32_t>(kTimes);
    for (int32_t i = 0; i < tail; ++i) {
        LocalTensor<float> sumTmpUb = temp1Buf_.Get<float>();
        int32_t tail = static_cast<int32_t>(kTimesTail);
        int32_t once = static_cast<int32_t>(rUbNumFactor);
        int32_t tailC = i == tail - 1 ? static_cast<int32_t>(rLoopTile) : once;
        int32_t tailAlign = rLoopTileAlign;
        int32_t align = i == tail - 1 ? static_cast<int32_t>(tailAlign) : once;
        int64_t offset = aOffset * r + once * i;
        if constexpr (featuresBrc != 0) {
            int64_t nddmaOffset = featuresBrcDim == 1 ? aOffset : once * i;
            DataCopyFeatureNDDMA(tailNum, nddmaOffset, once, once, padValue);
        } else {
            DataCopyFeature(tailNum, once, offset);
        }

        if constexpr (labelsBrc != 0) {
            int64_t nddmaOffset = labelsBrcDim == 1 ? aOffset : once * i;
            DataCopyLabelsNDDMA(tailNum, nddmaOffset, once, once, padValue);
        } else {
            DataCopyLabel(tailNum, once, offset);
        }
        PipeM2V();
        ComputeLossAndBropMainBlock(tailNum, once, lossUbFp32);

        CopyOutBackProb(tailNum, once, offset);

        int64_t offset1 = aOffset * r + once * (i + kTimes);
        if constexpr (featuresBrc != 0) {
            int64_t nddmaOffset1 = featuresBrcDim == 1 ? aOffset : once * (i + kTimes);
            DataCopyFeatureNDDMA(tailNum, nddmaOffset1, tailC, align, padValue);
        } else {
            DataCopyFeature(tailNum, tailC, offset1);
        }
        if constexpr (labelsBrc != 0) {
            int64_t nddmaOffset1 = labelsBrcDim == 1 ? aOffset : once * (i + kTimes);
            DataCopyLabelsNDDMA(tailNum, nddmaOffset1, tailC, align, padValue);
        } else {
            DataCopyLabel(tailNum, tailC, offset1);
        }
        PipeM2V();
        ComputeLossAndBropTailBlock(tailNum, once, tailC, align, lossUbFp32);
        CopyOutBackProb(tailNum, tailC, offset1);

        uint32_t srcShape[2] = {static_cast<uint32_t>(tailNum), static_cast<uint32_t>(rUbNumFactor)};
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(sumTmpUb, lossUbFp32, srcShape, false);
        UpdateCache(sumTmpUb, i, tailNum);
    }
    for (int32_t ci = tail; ci < loopTimes; ++ci) {
        LocalTensor<float> sumTmpUb = temp1Buf_.Get<float>();
        int32_t tailC = static_cast<int32_t>(rUbNumFactor);
        int32_t align = static_cast<int32_t>(rUbNumFactor);
        if ((ci == loopTimes - 1) && tail == 0) {
            tailC = static_cast<int32_t>(rLoopTile);
            align = static_cast<int32_t>(rLoopTileAlign);
        }
        int64_t offset = aOffset * r + rUbNumFactor * ci;
        if constexpr (featuresBrc != 0) {
            int64_t nddmaOffset = featuresBrcDim == 1 ? aOffset : rUbNumFactor * ci;
            DataCopyFeatureNDDMA(tailNum, nddmaOffset, tailC, align, padValue);
        } else {
            DataCopyFeature(tailNum, tailC, offset);
        }
        if constexpr (labelsBrc != 0) {
            int64_t nddmaOffset = labelsBrcDim == 1 ? aOffset : rUbNumFactor * ci;
            DataCopyLabelsNDDMA(tailNum, nddmaOffset, tailC, align, padValue);
        } else {
            DataCopyLabel(tailNum, tailC, offset);
        }
        PipeM2V();
        ComputeLossAndBropAlignBlock(tailNum, tailC, align, lossUbFp32);
        CopyOutBackProb(tailNum, tailC, offset);
        uint32_t srcShape[2] = {static_cast<uint32_t>(tailNum), static_cast<uint32_t>(align)};
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(sumTmpUb, lossUbFp32, srcShape, false);
        UpdateCache(sumTmpUb, ci, tailNum);
    }
    lossQueue_.FreeTensor(lossUbFp32);
    CopyOutLoss(tailNum, cacheUb, aOffset);
}

template <typename T, uint64_t schId, uint64_t featuresBrc, uint64_t labelsBrc, uint64_t db>
__aicore__ inline void SoftmaxCrossEntropyWithLogitsSplitR<T, schId, featuresBrc, labelsBrc, db>::PipeM2V()
{
    event_t eventMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventMTE2ToV);
}

template <typename T, uint64_t schId, uint64_t featuresBrc, uint64_t labelsBrc, uint64_t db>
__aicore__ inline void SoftmaxCrossEntropyWithLogitsSplitR<T, schId, featuresBrc, labelsBrc, db>::Compute(
    int64_t tileNum, int64_t offset)
{
    LocalTensor<float> maxUb = maxBuf_.Get<float>();
    CleanMaxUb(tileNum, maxUb);
    GetRowMax(tileNum, offset);
    ComputeSubExpReduceSum(tileNum, offset);
    ComputeRst(tileNum, offset);
}

} // namespace SoftmaxCrossEntropyWithLogits
#endif // SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_SPLIT_R_H
