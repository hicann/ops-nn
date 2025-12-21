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
 * \file sparse_softmax_cross_entropy_with_logits_split_r.h
 * \brief sparse_softmax_cross_entropy_with_logits_split_r
 */

#ifndef SPARSE_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_SPLIT_R_H
#define SPARSE_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_SPLIT_R_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "../inc/platform.h"
#include "sparse_softmax_cross_entropy_with_logits_tiling_data.h"

namespace SparseSoftmaxCrossEntropyWithLogits {
using namespace AscendC;
using namespace AscendC::MicroAPI;

template <typename T1, typename T2, uint64_t schId, uint64_t db>
__simt_vf__ __aicore__ LAUNCH_BOUND(1024) inline void UbSimtComputeLoopRSplit(__ubuf__ T1* backPropAddr, __ubuf__ float* temp2Addr, __ubuf__ T2* labelsAddr, __ubuf__ float* gatherAddr, const int32_t tileNum, const int32_t rOnceNum, const int32_t rOnceNumAlign, const int64_t rMainNum, const int32_t ci, const int64_t cMax)
{
    for (int64_t index = static_cast<int64_t>(Simt::GetThreadIdx()); index < tileNum; index += static_cast<int64_t>(Simt::GetThreadNum<0>())) {
		ASSERT((0 <= labelsAddr[index] && labelsAddr[index] < cMax) && "labels is not in [0, C)");
        if (labelsAddr[index] >= rMainNum * ci && labelsAddr[index] < (rMainNum * ci + rOnceNum)) {
            int64_t offset = index * rOnceNumAlign + labelsAddr[index] - rMainNum * ci;
            backPropAddr[offset] -= 1.0f;
            gatherAddr[index] = temp2Addr[offset];
        }
    }
}

template <typename T1, typename T2, uint64_t schId, uint64_t db>
class SparseSoftmaxCrossEntropyWithLogitsSplitR {
public:
    __aicore__ inline SparseSoftmaxCrossEntropyWithLogitsSplitR() {};
    __aicore__ inline void Init(GM_ADDR features, GM_ADDR labels, GM_ADDR loss, GM_ADDR backProp, GM_ADDR workspace,  const SparseSoftmaxCrossEntropyWithLogitsTilingData *tilingData, TPipe *pipe);
    __aicore__ inline void Process();

protected:
    __aicore__ inline void InitTilingData();
    __aicore__ inline void ProcessEachCore(int64_t loopTimes, int64_t tailNum);
    __aicore__ inline void Compute(int64_t tileNum, int64_t offset);
    __aicore__ inline void DataCopyFeaturePad(int32_t nTailNum, int32_t onceR, int32_t alignR, int64_t offset);
    __aicore__ inline void DataCopyFeature(int32_t tailNum, int32_t rOnceNum, int64_t offset);
    __aicore__ inline void DataCopyLabel(int32_t tailNum, int64_t offset);
    __aicore__ inline void CopyOutBackProb(int32_t tailNum, int32_t rOnceNum, int64_t offset);
    __aicore__ inline void ComputeMaxLast(int32_t nNum, LocalTensor<T1> &onceMaxUb);
    __aicore__ inline void GetRowMax(int32_t nTailNum, int64_t aOffset);
    __aicore__ inline void CleanMaxUb(int32_t tailNSize, LocalTensor<float> &clearUb);
    __aicore__ inline void UpdateCache(LocalTensor<float> &srcUb, int32_t index, int32_t dimA);
    __aicore__ inline void ComputeSubExpReduceSum(int32_t tailNum, int32_t aOffset);
    __aicore__ inline void ComputeLog(int32_t nSize, LocalTensor<float> &extraUb);
    __aicore__ inline void DataCopyOutLoss(int32_t nTailNum, int64_t offset);
    __aicore__ inline void CopyOutLoss(int32_t nSize, LocalTensor<float> &onceMaxUb, int32_t aOffset);
    __aicore__ inline void ComputeRst(int32_t tailNum, int32_t aOffset);
    __aicore__ inline void ComputeSubExpMainBlock(int32_t nSize, int32_t rOnceNum, LocalTensor<float> &outUb);
    __aicore__ inline void ComputeSubExpTailBlock(int32_t nSize, int32_t rOnceNumAlign, int32_t rOnceNumT, int32_t rOnceNumTAlign, LocalTensor<float> &outUb);
    __aicore__ inline void ComputeSubExpAlignBlock(int32_t nSize, int32_t rOnceNum, int32_t rOnceNumAlign, LocalTensor<float> &outUb);
    __aicore__ inline void ComputeBackpropAndLoss(LocalTensor<T1> &featureUb, LocalTensor<T1> &backPropUb, LocalTensor<float> &outUb, int32_t nSize, int32_t rOnceNum, int32_t rOnceNumAlign, int32_t ci);
    __aicore__ inline void PipeM2V();

protected:
    constexpr static AscendC::MicroAPI::CastTrait castB32ToB16 = { AscendC::MicroAPI::RegLayout::ZERO,
        AscendC::MicroAPI::SatMode::NO_SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT };
    constexpr static AscendC::MicroAPI::CastTrait castB16ToB32 = { AscendC::MicroAPI::RegLayout::ZERO,
        AscendC::MicroAPI::SatMode::UNKNOWN, AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN };

    constexpr static uint32_t DOUBLE_BUFFER = 1;
    constexpr static uint32_t BLOCK_SIZE = platform::GetUbBlockSize();
    constexpr static uint32_t VL_FP32 = static_cast<int64_t>(platform::GetVRegSize()) / sizeof(float);
private:
    GlobalTensor<T1> featuresGm_;
    GlobalTensor<T2> labelsGm_;
    GlobalTensor<T1> lossGm_;
    GlobalTensor<T1> backPropGm_;

    TPipe* pipe_ = nullptr;
    TQue<QuePosition::VECIN, 1> featuresQueue_;
    TQue<QuePosition::VECIN, 1> labelsQueue_;
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
    const SparseSoftmaxCrossEntropyWithLogitsTilingData* tilingData_ = nullptr;
    int64_t realCoreNum;
    int64_t a; //a轴
    int64_t r; //r轴
    int64_t blockFactor; //a轴分核，主核数据量
    int64_t tailBlockFactor; //a轴分核，尾核数据量
    int64_t rUbNumFactor; //R轴切分，一次UB可以放下的数据量，全载模板下等于r，注意32b对齐
    int64_t aUbNumFactor; //A轴切分，一次UB可以放下的数据量，非全载模板下等于1，注意32b对齐
    int64_t aLoopTimes; //主核A方向循环搬移数据的次数
    int64_t aLoopTimesT; //尾核A方向循环搬移数据的次数
    int64_t aLoopTail; //主核A方向尾块的数据量
    int64_t aLoopTailT; //尾核A方向尾块的数据量
    int64_t rLoopTime; //不能全载时，R轴反向的循环次数
    int64_t rLoopTile; //不能全载时，R轴反向的尾块数据量
    int64_t kTimesTail; //不能全载时，完全二分累加，存在主尾块相加的次数
    int64_t kTimes; //不能全载时，完全二分累加，2的k次方内循环次数
    int64_t updateStart_;
    int64_t rLoopTileAlign;
    int64_t coreStartOffset;
};

template <typename T1, typename T2, uint64_t schId, uint64_t db>
__aicore__ inline void SparseSoftmaxCrossEntropyWithLogitsSplitR<T1, T2, schId, db>::InitTilingData()
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
}

template <typename T1, typename T2, uint64_t schId, uint64_t db>
__aicore__ inline void SparseSoftmaxCrossEntropyWithLogitsSplitR<T1, T2, schId, db>::Init(GM_ADDR features, GM_ADDR labels, GM_ADDR loss, GM_ADDR backProp, GM_ADDR workspace,  const SparseSoftmaxCrossEntropyWithLogitsTilingData *tilingData, TPipe *pipe)
{
    tilingData_ = tilingData;
    InitTilingData();
    pipe_ = pipe;

    blockFactor_ = (AscendC::GetBlockIdx() == tilingData_->realCoreNum - 1) ? tailBlockFactor :  blockFactor;

    featuresGm_.SetGlobalBuffer((__gm__ T1 *)features);
    labelsGm_.SetGlobalBuffer((__gm__ T2 *)labels);
    lossGm_.SetGlobalBuffer((__gm__ T1 *)loss);
    backPropGm_.SetGlobalBuffer((__gm__ T1 *)backProp);

    int64_t ubBufferSize = rUbNumFactor * aUbNumFactor;
    pipe_->InitBuffer(featuresQueue_, DOUBLE_BUFFER, ubBufferSize * sizeof(T1));
    pipe_->InitBuffer(labelsQueue_, DOUBLE_BUFFER, aUbNumFactor * sizeof(T2));
    pipe_->InitBuffer(lossQueue_, DOUBLE_BUFFER, ubBufferSize * sizeof(float));
    pipe_->InitBuffer(backPropQueue_, DOUBLE_BUFFER, ubBufferSize * sizeof(T1));

    pipe_->InitBuffer(maxBuf_, aUbNumFactor * sizeof(float));
    pipe_->InitBuffer(logBuf_, aUbNumFactor * sizeof(float));
    pipe_->InitBuffer(sumBuf_, aUbNumFactor * sizeof(float));
    pipe_->InitBuffer(temp1Buf_, aUbNumFactor * sizeof(float));
    updateStart_ = bcnt1((kTimes - 1) ^ (kTimes)) - 1;
    pipe_->InitBuffer(cacheBuf_, (updateStart_ * VL_FP32 + aUbNumFactor) * sizeof(float));

    blockIdx_ = AscendC::GetBlockIdx();
    isLastCore = blockIdx_ == tilingData_->realCoreNum - 1;
    coreStartOffset = blockFactor * blockIdx_;
}

template <typename T1, typename T2, uint64_t schId, uint64_t db>
__aicore__ inline void SparseSoftmaxCrossEntropyWithLogitsSplitR<T1, T2, schId, db>::Process()
{
    if (unlikely(isLastCore)) {
        ProcessEachCore(aLoopTimesT, aLoopTailT);
    } else {
        ProcessEachCore(aLoopTimes, aLoopTail);
    }
}

template <typename T1, typename T2, uint64_t schId, uint64_t db>
__aicore__ inline void SparseSoftmaxCrossEntropyWithLogitsSplitR<T1, T2, schId, db>::ProcessEachCore(int64_t loopTimes, int64_t tailNum)
{
    for (int64_t aLoopIndex = 0; aLoopIndex < loopTimes; aLoopIndex++) {
        Compute(aUbNumFactor, aLoopIndex * aUbNumFactor + coreStartOffset);
    }
    if (unlikely(tailNum != 0)) {
        Compute(tailNum, aUbNumFactor * loopTimes + coreStartOffset);
    }
}

template <typename T1, typename T2, uint64_t schId, uint64_t db>
__aicore__ inline void SparseSoftmaxCrossEntropyWithLogitsSplitR<T1, T2, schId, db>::DataCopyFeaturePad(int32_t nTailNum,
    int32_t onceR, int32_t alignR, int64_t offset)
{   
    LocalTensor<T1> featureUB = featuresQueue_.AllocTensor<T1>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = nTailNum;
    copyParams.blockLen = onceR * sizeof(T1);
    copyParams.srcStride = (r - onceR) * sizeof(T1);
    copyParams.dstStride = 0;
    DataCopyPadExtParams<T1> padParams;
    T1 minValue = -3.402823466e+38;
    if constexpr (IsSameType<T1, half>::value) {
        minValue = -65504;
    }
    padParams.isPad = true;
    padParams.leftPadding = 0;
    padParams.rightPadding = alignR - onceR;
    padParams.paddingValue = minValue;
    AscendC::DataCopyPad(featureUB, featuresGm_[offset], copyParams, padParams);
    
    featuresQueue_.EnQue(featureUB);
}
template <typename T1, typename T2, uint64_t schId, uint64_t db>
__aicore__ inline void SparseSoftmaxCrossEntropyWithLogitsSplitR<T1, T2, schId, db>::DataCopyFeature(int32_t tailNum,
    int32_t rOnceNum, int64_t offset)
{
    LocalTensor<T1> featureUB = featuresQueue_.AllocTensor<T1>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = tailNum;
    copyParams.blockLen = rOnceNum * sizeof(T1);
    copyParams.srcStride = (r - rOnceNum) * sizeof(T1);
    copyParams.dstStride = 0;
    DataCopyPadExtParams<T1> padParams{ false, 0, 0, 0 };
    AscendC::DataCopyPad(featureUB, featuresGm_[offset], copyParams, padParams);
    featuresQueue_.EnQue(featureUB);
}
template <typename T1, typename T2, uint64_t schId, uint64_t db>
__aicore__ inline void SparseSoftmaxCrossEntropyWithLogitsSplitR<T1, T2, schId, db>::DataCopyLabel(int32_t tailNum, int64_t offset)
{
    LocalTensor<T2> labelUB = labelsQueue_.AllocTensor<T2>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = tailNum * sizeof(T2);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPadExtParams<T2> padParams{ false, 0, 0, 0 };
    AscendC::DataCopyPad(labelUB, labelsGm_[offset], copyParams, padParams);
    labelsQueue_.EnQue(labelUB);
}
template <typename T1, typename T2, uint64_t schId, uint64_t db>
__aicore__ inline void SparseSoftmaxCrossEntropyWithLogitsSplitR<T1, T2, schId, db>::CopyOutBackProb(int32_t tailNum,
    int32_t rOnceNum, int64_t offset)
{
    LocalTensor<T1> srcUb = backPropQueue_.DeQue<T1>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = tailNum;
    copyParams.blockLen = rOnceNum * sizeof(T1);
    copyParams.dstStride = (r - rOnceNum) * sizeof(T1);
    copyParams.srcStride = 0;
    AscendC::DataCopyPad(backPropGm_[offset], srcUb, copyParams);
    backPropQueue_.FreeTensor(srcUb);
}
template <typename T1, typename T2, uint64_t schId, uint64_t db>
__aicore__ inline void SparseSoftmaxCrossEntropyWithLogitsSplitR<T1, T2, schId, db>::ComputeMaxLast(int32_t nNum, LocalTensor<T1> &onceMaxUb)
{
    uint32_t nTimes = nNum;

    uint32_t vfLenfp32 = VL_FP32;
    uint16_t repeatTimes1 = CeilDivision(nTimes, vfLenfp32);
    LocalTensor<float> maxUb = maxBuf_.Get<float>();

    auto maxUbAddr = (__ubuf__ float *)maxUb.GetPhyAddr();
    auto maxUbOnceAddr = (__ubuf__ T1 *)onceMaxUb.GetPhyAddr();
    auto maxUbOnceAddrB32 = (__ubuf__ float *)onceMaxUb.GetPhyAddr();
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::MaskReg preg;
        AscendC::MicroAPI::RegTensor<float> srcReg1;
        AscendC::MicroAPI::RegTensor<float> srcReg2;
        AscendC::MicroAPI::RegTensor<T1> srcReg2B16;
        AscendC::MicroAPI::RegTensor<float> srcReg3;

        for (uint16_t j = 0; j < repeatTimes1; j++) {
            preg = AscendC::MicroAPI::UpdateMask<float>(nTimes);
            AscendC::MicroAPI::AddrReg srcOffset = AscendC::MicroAPI::CreateAddrReg<float>(j, vfLenfp32);
            AscendC::MicroAPI::DataCopy(srcReg1, maxUbAddr, srcOffset);
            if constexpr (sizeof(T1) == 4) {
                AscendC::MicroAPI::DataCopy(srcReg2, maxUbOnceAddrB32, srcOffset);
            } else {
                AscendC::MicroAPI::AddrReg srcOffset1 = AscendC::MicroAPI::CreateAddrReg<T1>(j, vfLenfp32);
                AscendC::MicroAPI::DataCopy<T1, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(srcReg2B16, maxUbOnceAddr,
                    srcOffset1);
                AscendC::MicroAPI::Cast<float, T1, castB16ToB32>(srcReg2, srcReg2B16, preg);
            }
            AscendC::MicroAPI::Max(srcReg3, srcReg1, srcReg2, preg);
            AscendC::MicroAPI::DataCopy(maxUbAddr, srcReg3, srcOffset, preg);
        }
    }
}
template <typename T1, typename T2, uint64_t schId, uint64_t db>
__aicore__ inline void SparseSoftmaxCrossEntropyWithLogitsSplitR<T1, T2, schId, db>::GetRowMax(int32_t nTailNum, int64_t aOffset)
{
    LocalTensor<T1> onceMaxUb = temp1Buf_.Get<T1>();
    int32_t ctimes = static_cast<int32_t>(rLoopTime);
    for (int32_t ci = 0; ci < ctimes; ++ci) {
        int32_t onceR = ci == ctimes - 1 ? int32_t(rLoopTile) : int32_t(rUbNumFactor);
        int32_t rTileAlign = rLoopTileAlign;
        int32_t alignR = ci == ctimes - 1 ? int32_t(rTileAlign) : int32_t(rUbNumFactor);
        int64_t offset = aOffset * r + rUbNumFactor * ci;
        DataCopyFeaturePad(nTailNum, onceR, alignR, offset);
        LocalTensor<T1> featureUb = featuresQueue_.DeQue<T1>();
        uint32_t srcShape[2] = {static_cast<uint32_t>(nTailNum), static_cast<uint32_t>(alignR)};
        AscendC::ReduceMax<T1, AscendC::Pattern::Reduce::AR, true>(onceMaxUb, featureUb, srcShape, false);
        featuresQueue_.FreeTensor(featureUb);
        ComputeMaxLast(nTailNum, onceMaxUb);
    }
}

template <typename T1, typename T2, uint64_t schId, uint64_t db>
__aicore__ inline void SparseSoftmaxCrossEntropyWithLogitsSplitR<T1, T2, schId, db>::CleanMaxUb(int32_t tailNSize,
    LocalTensor<float> &clearUb)
{
    uint32_t nSize = tailNSize;
    float minValueFp32_ = static_cast<float>(-3.402823466e+38); // fp32可以表示的最小值
    int32_t vfLen = VL_FP32;
    uint16_t repeatTimes1 = CeilDivision(nSize, vfLen);
    auto maxUbAddr = (__ubuf__ float *)clearUb.GetPhyAddr();

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::MaskReg preg;
        AscendC::MicroAPI::RegTensor<float> srcReg0;
        AscendC::MicroAPI::RegTensor<float> srcReg1;
        for (uint16_t j = 0; j < repeatTimes1; j++) {
            preg = AscendC::MicroAPI::UpdateMask<float>(nSize);
            AscendC::MicroAPI::AddrReg srcOffset = AscendC::MicroAPI::CreateAddrReg<float>(j, vfLen);
            AscendC::MicroAPI::Duplicate(srcReg0, minValueFp32_);
            AscendC::MicroAPI::DataCopy(maxUbAddr, srcReg0, srcOffset, preg);
        }
    }
}

template <typename T1, typename T2, uint64_t schId, uint64_t db>
__aicore__ inline void SparseSoftmaxCrossEntropyWithLogitsSplitR<T1, T2, schId, db>::UpdateCache(LocalTensor<float> &srcUb,
    int32_t index, int32_t dimA)
{
    const int32_t cacheID = bcnt1(index ^ (index + 1)) - 1;
    int32_t elementOneRepeat = VL_FP32;
    uint16_t outerLoopTimes = CeilDivision(dimA, elementOneRepeat);
    int32_t stride = outerLoopTimes * elementOneRepeat;

    LocalTensor<float> cacheUb = cacheBuf_.Get<float>();
    auto dstUbAddr = (__ubuf__ float *)cacheUb.GetPhyAddr();
    auto cahUbAddr = (__ubuf__ float *)cacheUb.GetPhyAddr() + cacheID * stride;
    auto srcUbAddr = (__ubuf__ float *)srcUb.GetPhyAddr();

    const uint16_t innerLoopTimes = static_cast<uint16_t>(cacheID);
    uint32_t sreg = dimA;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<float> aReg, bReg;
        AscendC::MicroAPI::MaskReg pMask;
        for (uint16_t i = 0; i < outerLoopTimes; ++i) {
            pMask = AscendC::MicroAPI::UpdateMask<float>(sreg);
            AscendC::MicroAPI::AddrReg srcOffset = AscendC::MicroAPI::CreateAddrReg<float>(i, elementOneRepeat);
            AscendC::MicroAPI::DataCopy(aReg, srcUbAddr, srcOffset);
            for (uint16_t j = 0; j < innerLoopTimes; ++j) {
                AscendC::MicroAPI::AddrReg srcOffsetJ =
                    AscendC::MicroAPI::CreateAddrReg<float>(i, elementOneRepeat, j, stride);
                AscendC::MicroAPI::DataCopy(bReg, dstUbAddr, srcOffsetJ);
                AscendC::MicroAPI::Add<float, AscendC::MicroAPI::MaskMergeMode::ZEROING>(aReg, aReg, bReg, pMask);
            }
            AscendC::MicroAPI::DataCopy(cahUbAddr, aReg, srcOffset, pMask);
        }
    }
}

template <typename T1, typename T2, uint64_t schId, uint64_t db>
__aicore__ inline void SparseSoftmaxCrossEntropyWithLogitsSplitR<T1, T2, schId, db>::ComputeSubExpMainBlock(int32_t nSize, int32_t rOnceNum,
    LocalTensor<float> &outUb)
{
    LocalTensor<T1> featureUb = featuresQueue_.DeQue<T1>();
    uint16_t nTimes = nSize;
    uint32_t vfLen = VL_FP32;

    uint16_t repeatTimes1 = rOnceNum / vfLen;
    LocalTensor<float> maxUb = maxBuf_.Get<float>();

    auto outUbAddr = (__ubuf__ float *)outUb.GetPhyAddr();
    auto inputUbAddr = (__ubuf__ T1 *)featureUb.GetPhyAddr();
    auto maxUbAddr = (__ubuf__ float *)maxUb.GetPhyAddr();
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::MaskReg copyOutReg =
            AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::RegTensor<float> maxReg;
        AscendC::MicroAPI::RegTensor<T1> srcReg0;
        AscendC::MicroAPI::RegTensor<float> srcRegfp32;
        AscendC::MicroAPI::RegTensor<float> subReg;
        AscendC::MicroAPI::RegTensor<float> expReg;
        for (uint16_t i = 0; i < nTimes; i++) {
            AscendC::MicroAPI::DataCopy<float, LoadDist::DIST_BRC_B32>(maxReg, maxUbAddr + i);
            for (uint16_t j = 0; j < repeatTimes1; j++) {                                                                     
                AscendC::MicroAPI::AddrReg outOffset = AscendC::MicroAPI::CreateAddrReg<float>(i, rOnceNum, j, vfLen); 
                if constexpr (sizeof(T1) == 2) {
                    AscendC::MicroAPI::AddrReg srcOffset = AscendC::MicroAPI::CreateAddrReg<T1>(i, rOnceNum, j, vfLen);
                    AscendC::MicroAPI::DataCopy<T1, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(srcReg0, inputUbAddr,
                        srcOffset);                                                                                           
                    AscendC::MicroAPI::Cast<float, T1, castB16ToB32>(srcRegfp32, srcReg0, copyOutReg);
                } else {                                                                                                      
                    AscendC::MicroAPI::AddrReg srcOffset1 =                                                                   
                        AscendC::MicroAPI::CreateAddrReg<float>(i, rOnceNum, j, vfLen);                                 
                    AscendC::MicroAPI::DataCopy(srcRegfp32, inputUbAddr, srcOffset1);                                      
                }                                                                                                             
                AscendC::MicroAPI::Sub(subReg, srcRegfp32, maxReg, copyOutReg);                                               
                AscendC::MicroAPI::Exp(expReg, subReg, copyOutReg);                                                           
                AscendC::MicroAPI::DataCopy(outUbAddr, expReg, outOffset, copyOutReg);                                             
            }
        }
    }
    featuresQueue_.FreeTensor(featureUb);
}

template <typename T1, typename T2, uint64_t schId, uint64_t db>
__aicore__ inline void SparseSoftmaxCrossEntropyWithLogitsSplitR<T1, T2, schId, db>::ComputeSubExpTailBlock(int32_t nSize, int32_t rOnceNumAlign,
 int32_t rOnceNumT, int32_t rOnceNumTAlign, LocalTensor<float> &outUb)
{
    uint16_t nTimes = nSize;
    uint32_t vfLen = VL_FP32;

    uint16_t repeatTimes1 = rOnceNumT / vfLen;
    uint32_t tailNum = rOnceNumT % vfLen;
    uint16_t tailLoopTimes = tailNum != 0 ? 1 : 0;
    LocalTensor<float> maxUb = maxBuf_.Get<float>();

    LocalTensor<T1> featureUbT = featuresQueue_.DeQue<T1>();

    auto outUbAddr = (__ubuf__ float *)outUb.GetPhyAddr();
    auto inputUbAddr = (__ubuf__ T1 *)featureUbT.GetPhyAddr();
    auto maxUbAddr = (__ubuf__ float *)maxUb.GetPhyAddr();
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::MaskReg copyOutReg =
            AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::RegTensor<float> maxReg;
        AscendC::MicroAPI::RegTensor<T1> srcReg0;
        AscendC::MicroAPI::RegTensor<float> srcRegfp32;
        AscendC::MicroAPI::RegTensor<float> subReg;
        AscendC::MicroAPI::RegTensor<float> expReg;
        AscendC::MicroAPI::RegTensor<float> outReg;
        AscendC::MicroAPI::RegTensor<float> outReg1;
        AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::UpdateMask<float>(tailNum);
        for (uint16_t i = 0; i < nTimes; i++) {
            AscendC::MicroAPI::DataCopy<float, LoadDist::DIST_BRC_B32>(maxReg, maxUbAddr + i);
            for (uint16_t j = 0; j < repeatTimes1; j++) {
                AscendC::MicroAPI::AddrReg srcOffset =
                    AscendC::MicroAPI::CreateAddrReg<T1>(i, rOnceNumTAlign, j, vfLen);
                AscendC::MicroAPI::AddrReg outOffset =
                    AscendC::MicroAPI::CreateAddrReg<float>(i, rOnceNumAlign, j, vfLen);                                                 
                if constexpr (sizeof(T1) == 2) {
                    AscendC::MicroAPI::DataCopy<T1, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(srcReg0, inputUbAddr,
                        srcOffset);                                                                                               
                    AscendC::MicroAPI::Cast<float, T1, castB16ToB32>(srcRegfp32, srcReg0, copyOutReg);
                } else {
                    AscendC::MicroAPI::DataCopy(srcRegfp32, inputUbAddr, srcOffset);                                          
                }                                                                                                                 
                AscendC::MicroAPI::Sub(subReg, srcRegfp32, maxReg, copyOutReg);                                                   
                AscendC::MicroAPI::Exp(expReg, subReg, copyOutReg);                                                               
                AscendC::MicroAPI::DataCopy(outReg, outUbAddr, outOffset);                                                        
                AscendC::MicroAPI::Add(outReg1, expReg, outReg, copyOutReg);                                                      
                AscendC::MicroAPI::DataCopy(outUbAddr, outReg1, outOffset, copyOutReg);
            }
            for (uint16_t k = 0; k < tailLoopTimes; k++) {                                                                           
                if constexpr (sizeof(T1) == 2) {
                    AscendC::MicroAPI::DataCopy<T1, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(srcReg0,
                        inputUbAddr + i * rOnceNumTAlign + repeatTimes1 * vfLen);                                            
                    AscendC::MicroAPI::Cast<float, T1, castB16ToB32>(srcRegfp32, srcReg0, preg);
                } else {                                                                                                     
                    AscendC::MicroAPI::DataCopy(srcRegfp32, inputUbAddr + i * rOnceNumTAlign + repeatTimes1 * vfLen);     
                }                                                                                                            
                AscendC::MicroAPI::Sub(subReg, srcRegfp32, maxReg, preg);                                              
                AscendC::MicroAPI::Exp(expReg, subReg, preg);                                                          
                AscendC::MicroAPI::DataCopy(outReg, outUbAddr + i * rOnceNumAlign + repeatTimes1 * vfLen);                  
                AscendC::MicroAPI::Add(outReg1, expReg, outReg, preg);                                                 
                AscendC::MicroAPI::DataCopy(outUbAddr + i * rOnceNumAlign + repeatTimes1 * vfLen, outReg1, preg);
            }
        }
    }
    featuresQueue_.FreeTensor(featureUbT);
}

template <typename T1, typename T2, uint64_t schId, uint64_t db>
__aicore__ inline void SparseSoftmaxCrossEntropyWithLogitsSplitR<T1, T2, schId, db>::ComputeSubExpAlignBlock(int32_t nSize, int32_t rOnceNum, int32_t rOnceNumAlign,
    LocalTensor<float> &outUb)
{
    uint16_t nTimes = nSize;
    uint32_t vfLen = VL_FP32;

    uint16_t repeatTimes1 = rOnceNum / vfLen;
    uint32_t tailNum = rOnceNum % vfLen;
    uint32_t tailNumAlign = rOnceNumAlign - repeatTimes1 * vfLen;
    uint16_t tailLoopTimes = tailNum != 0 ? 1 : 0;
    LocalTensor<float> maxUb = maxBuf_.Get<float>();
    LocalTensor<T1> featureUb = featuresQueue_.DeQue<T1>();

    auto outUbAddr = (__ubuf__ float *)outUb.GetPhyAddr();
    auto inputUbAddr = (__ubuf__ T1 *)featureUb.GetPhyAddr();
    auto maxUbAddr = (__ubuf__ float *)maxUb.GetPhyAddr();
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::MaskReg copyOutReg =
            AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::RegTensor<float> maxReg;
        AscendC::MicroAPI::RegTensor<T1> srcReg0;
        AscendC::MicroAPI::RegTensor<float> srcRegfp32;
        AscendC::MicroAPI::RegTensor<float> subReg;
        AscendC::MicroAPI::RegTensor<float> expReg;
        AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::UpdateMask<float>(tailNum);
        AscendC::MicroAPI::MaskReg preg1 = AscendC::MicroAPI::UpdateMask<float>(tailNumAlign);
        for (uint16_t i = 0; i < nTimes; i++) {
            AscendC::MicroAPI::DataCopy<float, LoadDist::DIST_BRC_B32>(maxReg, maxUbAddr + i);
            for (uint16_t j = 0; j < repeatTimes1; j++) {                                                                     
                AscendC::MicroAPI::AddrReg srcOffset = AscendC::MicroAPI::CreateAddrReg<T1>(i, rOnceNumAlign, j, vfLen);
                AscendC::MicroAPI::AddrReg outOffset = AscendC::MicroAPI::CreateAddrReg<float>(i, rOnceNumAlign, j, vfLen); 
                if constexpr (sizeof(T1) == 2) {
                    AscendC::MicroAPI::DataCopy<T1, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(srcReg0, inputUbAddr,
                        srcOffset);                                                                                           
                    AscendC::MicroAPI::Cast<float, T1, castB16ToB32>(srcRegfp32, srcReg0, copyOutReg);
                } else {                                
                    AscendC::MicroAPI::DataCopy(srcRegfp32, inputUbAddr, srcOffset);                                      
                }                                                                                                             
                AscendC::MicroAPI::Sub(subReg, srcRegfp32, maxReg, copyOutReg);                                               
                AscendC::MicroAPI::Exp(expReg, subReg, copyOutReg);                                                           
                AscendC::MicroAPI::DataCopy(outUbAddr, expReg, outOffset, copyOutReg);                                             
            }              
            for (uint16_t k = 0; k < tailLoopTimes; k++) {                                                                        
                if constexpr (sizeof(T1) == 2) {
                    AscendC::MicroAPI::DataCopy<T1, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(srcReg0,
                        inputUbAddr + i * rOnceNumAlign + repeatTimes1 * vfLen);                                        
                    AscendC::MicroAPI::Cast<float, T1, castB16ToB32>(srcRegfp32, srcReg0, preg);
                } else {                                                                                                 
                    AscendC::MicroAPI::DataCopy(srcRegfp32, inputUbAddr + i * rOnceNumAlign + repeatTimes1 * vfLen); 
                }                                                                                                        
                AscendC::MicroAPI::Sub(subReg, srcRegfp32, maxReg, preg);                                          
                AscendC::MicroAPI::Exp(expReg, subReg, preg);                                                      
                AscendC::MicroAPI::DataCopy(outUbAddr + i * rOnceNumAlign + repeatTimes1 * vfLen, expReg, preg1);       
            }
        }
    }
    featuresQueue_.FreeTensor(featureUb);
}

template <typename T1, typename T2, uint64_t schId, uint64_t db>
__aicore__ inline void SparseSoftmaxCrossEntropyWithLogitsSplitR<T1, T2, schId, db>::ComputeSubExpReduceSum(int32_t tailNum, int32_t aOffset)
{
    LocalTensor<float> outUbFp32 = lossQueue_.AllocTensor<float>();
    LocalTensor<float> cacheUb = cacheBuf_.Get<float>();
    int32_t tail = static_cast<int32_t>(kTimesTail);
    int32_t loopTimes = static_cast<int32_t>(kTimes);
    T1 padValue = 0;
    for (int32_t i = 0; i < tail; ++i) {
        LocalTensor<float> sumUb = temp1Buf_.Get<float>();
        int32_t once = static_cast<int32_t>(rUbNumFactor);
        int32_t tailC = i == tail - 1 ? static_cast<int32_t>(rLoopTile) : once;
        int32_t tailAlign = rLoopTileAlign;
        int32_t align = i == tail - 1 ? static_cast<int32_t>(tailAlign) : once;
        int64_t offset = aOffset * r + once * i;
        DataCopyFeature(tailNum, once, offset);
        ComputeSubExpMainBlock(tailNum, once, outUbFp32);

        int64_t offset1 = aOffset * r + once * (i + kTimes);

        DataCopyFeature(tailNum, tailC, offset1);
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
        DataCopyFeature(tailNum, tailC, offset);
        ComputeSubExpAlignBlock(tailNum, tailC, align, outUbFp32);
        uint32_t srcShape[2] = {static_cast<uint32_t>(tailNum), static_cast<uint32_t>(align)};
        AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(sumUb, outUbFp32, srcShape, false);
        UpdateCache(sumUb, ci, tailNum);
    }
    lossQueue_.FreeTensor(outUbFp32);
    ComputeLog(tailNum, cacheUb);
}

template <typename T1, typename T2, uint64_t schId, uint64_t db>
__aicore__ inline void SparseSoftmaxCrossEntropyWithLogitsSplitR<T1, T2, schId, db>::ComputeLog(int32_t nSize, LocalTensor<float> &extraUb)
{
    uint32_t nSize1 = nSize;
    uint32_t vfLen = VL_FP32;
    uint16_t repeatTimes1 = CeilDivision(nSize, vfLen);
    LocalTensor<float> logUb = logBuf_.Get<float>();
    LocalTensor<float> sumUb = sumBuf_.Get<float>();
    auto logUbAddr = (__ubuf__ float *)logUb.GetPhyAddr();
    auto sumUbAddr = (__ubuf__ float *)sumUb.GetPhyAddr();
    int32_t cacheStart = repeatTimes1 * vfLen * updateStart_;
    auto cacheUbAddr = (__ubuf__ float *)extraUb.GetPhyAddr() + cacheStart;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::MaskReg preg;
        AscendC::MicroAPI::RegTensor<float> srcReg0;
        AscendC::MicroAPI::RegTensor<float> logReg0;
        AscendC::MicroAPI::RegTensor<float> sumReg0;
        AscendC::MicroAPI::RegTensor<float> addReg0;
        for (uint16_t j = 0; j < repeatTimes1; j++) {
            preg = AscendC::MicroAPI::UpdateMask<float>(nSize1);
            AscendC::MicroAPI::AddrReg srcOffset = AscendC::MicroAPI::CreateAddrReg<float>(j, vfLen);
            AscendC::MicroAPI::DataCopy(srcReg0, cacheUbAddr, srcOffset);
            AscendC::MicroAPI::Log(logReg0, srcReg0, preg);
            AscendC::MicroAPI::Copy(sumReg0, srcReg0, preg);
            AscendC::MicroAPI::DataCopy(logUbAddr, logReg0, srcOffset, preg);
            AscendC::MicroAPI::DataCopy(sumUbAddr, sumReg0, srcOffset, preg);
        }
    }
}

template <typename T1, typename T2, uint64_t schId, uint64_t db>
__aicore__ inline void SparseSoftmaxCrossEntropyWithLogitsSplitR<T1, T2, schId, db>::DataCopyOutLoss(int32_t nTailNum, int64_t offset)
{
    LocalTensor<T1> lossUb = lossQueue_.DeQue<T1>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = nTailNum * sizeof(T1);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;

    AscendC::DataCopyPad(lossGm_[offset], lossUb, copyParams);
    lossQueue_.FreeTensor(lossUb);
}
template <typename T1, typename T2, uint64_t schId, uint64_t db>
__aicore__ inline void SparseSoftmaxCrossEntropyWithLogitsSplitR<T1, T2, schId, db>::CopyOutLoss(int32_t nSize, LocalTensor<float> &onceMaxUb, int32_t aOffset)
{
    uint32_t nSize1 = nSize;
    uint32_t vfLen = VL_FP32;
    uint16_t repeatTimes1 = CeilDivision(nSize, vfLen);
    LocalTensor<T1> lossUb = lossQueue_.AllocTensor<T1>();
    int32_t cacheStart = repeatTimes1 * vfLen * updateStart_;
    auto cacheUbAddr = (__ubuf__ float *)onceMaxUb.GetPhyAddr();
    auto lossUbAddr = (__ubuf__ T1 *)lossUb.GetPhyAddr();
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::MaskReg preg;
        AscendC::MicroAPI::RegTensor<float> srcReg0;
        AscendC::MicroAPI::RegTensor<float> logReg0;
        AscendC::MicroAPI::RegTensor<T1> lossReg0;
        AscendC::MicroAPI::RegTensor<float> addReg0;
        for (uint16_t j = 0; j < repeatTimes1; j++) {
            preg = AscendC::MicroAPI::UpdateMask<float>(nSize1);
            AscendC::MicroAPI::AddrReg srcOffset = AscendC::MicroAPI::CreateAddrReg<float>(j, vfLen);
            AscendC::MicroAPI::MaskReg regAllFp32 =
                AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
            AscendC::MicroAPI::DataCopy(srcReg0, cacheUbAddr, srcOffset);
            if constexpr (sizeof(T1) == 2) {
                AscendC::MicroAPI::Cast<T1, float, castB32ToB16>(lossReg0, srcReg0, preg);
                AscendC::MicroAPI::DataCopy<T1, StoreDist::DIST_PACK_B32>(lossUbAddr, lossReg0, preg);
            } else {
                AscendC::MicroAPI::Copy(lossReg0, srcReg0, preg);
                AscendC::MicroAPI::DataCopy(lossUbAddr, lossReg0, srcOffset, preg);
            }
        }
    }
    lossQueue_.EnQue<T1>(lossUb);
    DataCopyOutLoss(nSize, aOffset);
}

template <typename T1, typename T2, uint64_t schId, uint64_t db>
__aicore__ inline void SparseSoftmaxCrossEntropyWithLogitsSplitR<T1, T2, schId, db>::ComputeRst(int32_t tailNum, int32_t aOffset)
{
    LocalTensor<float> lossUbFp32 = lossQueue_.AllocTensor<float>();
    LocalTensor<float> onceMaxUb = temp1Buf_.Get<float>();
    int32_t ctimes = static_cast<int32_t>(rLoopTime);
	int64_t cMax = r;
	int64_t rFactor = rUbNumFactor;
    DataCopyLabel(tailNum, aOffset);
    LocalTensor<T2> labelsUb = labelsQueue_.DeQue<T2>();
    for (int32_t ci = 0; ci < ctimes; ++ci) {
        int32_t onceR = ci == ctimes - 1 ? int32_t(rLoopTile) : int32_t(rUbNumFactor);
        int32_t rTileAlign = rLoopTileAlign;
        int32_t alignR = ci == ctimes - 1 ? int32_t(rTileAlign) : int32_t(rUbNumFactor);
        int64_t offset = aOffset * r + rUbNumFactor * ci;
        DataCopyFeature(tailNum, onceR, offset);
        LocalTensor<T1> featureUb = featuresQueue_.DeQue<T1>();
		LocalTensor<T1> backPropUb = backPropQueue_.AllocTensor<T1>();
        ComputeBackpropAndLoss(featureUb, backPropUb, lossUbFp32, tailNum, onceR, alignR, ci);
		Simt::VF_CALL<UbSimtComputeLoopRSplit<T1, T2, schId, db>>(Simt::Dim3{1024}, (__ubuf__ T1 *)backPropUb.GetPhyAddr(), (__ubuf__ float *)lossUbFp32.GetPhyAddr(), (__ubuf__ T2 *)labelsUb.GetPhyAddr(), (__ubuf__ float *)onceMaxUb.GetPhyAddr(), tailNum, onceR, alignR, rFactor, ci, cMax);
        featuresQueue_.FreeTensor(featureUb);
		backPropQueue_.EnQue<T1>(backPropUb);
        CopyOutBackProb(tailNum, onceR, offset);
    }
    labelsQueue_.FreeTensor(labelsUb);
    lossQueue_.FreeTensor(lossUbFp32);
    CopyOutLoss(tailNum, onceMaxUb, aOffset);
}

template <typename T1, typename T2, uint64_t schId, uint64_t db>
__aicore__ inline void SparseSoftmaxCrossEntropyWithLogitsSplitR<T1, T2, schId, db>::ComputeBackpropAndLoss(LocalTensor<T1> &featureUb, LocalTensor<T1> &backPropUb, LocalTensor<float> &outUb, int32_t nSize, int32_t rOnceNum, int32_t rOnceNumAlign, int32_t ci)
{
    LocalTensor<float> sumUb = sumBuf_.Get<float>();
    LocalTensor<float> maxUb = maxBuf_.Get<float>();
    LocalTensor<float> logUb = logBuf_.Get<float>();
    LocalTensor<float> tempUb = temp1Buf_.Get<float>();
    uint16_t nTimes = nSize;
    uint32_t vfLen = VL_FP32;

    uint16_t repeatTimes1 = rOnceNum / vfLen;
    uint32_t tailNum = rOnceNum % vfLen;
    uint32_t tailNumAlign = rOnceNumAlign - repeatTimes1 * vfLen;
    uint16_t tailLoopTimes = tailNum != 0 ? 1 : 0;

    auto outUbAddr = (__ubuf__ float *)outUb.GetPhyAddr();
    auto tempUbAddr = (__ubuf__ float *)tempUb.GetPhyAddr();
    auto inputUbAddr = (__ubuf__ T1 *)featureUb.GetPhyAddr();
    auto maxUbAddr = (__ubuf__ float *)maxUb.GetPhyAddr();
    auto sumUbAddr = (__ubuf__ float *)sumUb.GetPhyAddr();
    auto logUbAddr = (__ubuf__ float *)logUb.GetPhyAddr();
    auto backProbAddr = (__ubuf__ T1 *)backPropUb.GetPhyAddr();
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::MaskReg copyOutReg =
            AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::RegTensor<float> maxReg;
        AscendC::MicroAPI::RegTensor<float> sumReg;
        AscendC::MicroAPI::RegTensor<float> logReg;
        AscendC::MicroAPI::RegTensor<T1> srcReg0;
        AscendC::MicroAPI::RegTensor<T1> srcReg1;
        AscendC::MicroAPI::RegTensor<float> srcRegfp32;
        AscendC::MicroAPI::RegTensor<float> srcReg1fp32;
        AscendC::MicroAPI::RegTensor<float> subReg;
        AscendC::MicroAPI::RegTensor<float> expReg;
        AscendC::MicroAPI::RegTensor<float> tmpReg;
        AscendC::MicroAPI::RegTensor<T1> backProbReg;
        AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::UpdateMask<float>(tailNum);
        AscendC::MicroAPI::MaskReg preg1 = AscendC::MicroAPI::UpdateMask<float>(tailNumAlign);
        for (uint16_t i = 0; i < nTimes; i++) {
            AscendC::MicroAPI::DataCopy<float, LoadDist::DIST_BRC_B32>(maxReg, maxUbAddr + i);
            AscendC::MicroAPI::DataCopy<float, LoadDist::DIST_BRC_B32>(sumReg, sumUbAddr + i);
            AscendC::MicroAPI::DataCopy<float, LoadDist::DIST_BRC_B32>(logReg, logUbAddr + i);
            for (uint16_t j = 0; j < repeatTimes1; j++) {
                AscendC::MicroAPI::AddrReg srcOffset = AscendC::MicroAPI::CreateAddrReg<T1>(i, rOnceNumAlign, j, vfLen);
                AscendC::MicroAPI::AddrReg outOffset = AscendC::MicroAPI::CreateAddrReg<float>(i, rOnceNumAlign, j, vfLen);
                if constexpr (sizeof(T1) == 2) {
                    AscendC::MicroAPI::DataCopy<T1, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(srcReg0, inputUbAddr,
                        srcOffset);
                    AscendC::MicroAPI::Cast<float, T1, castB16ToB32>(srcRegfp32, srcReg0, copyOutReg);
                } else {
                    AscendC::MicroAPI::DataCopy(srcRegfp32, inputUbAddr, srcOffset);
                }
                AscendC::MicroAPI::Sub(subReg, srcRegfp32, maxReg, copyOutReg);
                AscendC::MicroAPI::Exp(expReg, subReg, copyOutReg);
                AscendC::MicroAPI::Div(expReg, expReg, sumReg, copyOutReg);
                AscendC::MicroAPI::Sub(tmpReg, logReg, subReg, copyOutReg);
                AscendC::MicroAPI::DataCopy(outUbAddr, tmpReg, outOffset, copyOutReg);
                if constexpr (sizeof(T1) == 2) {
                    AscendC::MicroAPI::AddrReg outOffset1 = AscendC::MicroAPI::CreateAddrReg<T1>(i, rOnceNumAlign, j, vfLen);
                    AscendC::MicroAPI::Cast<T1, float, castB32ToB16>(backProbReg, expReg, copyOutReg);
                    AscendC::MicroAPI::DataCopy<T1, StoreDist::DIST_PACK_B32>(backProbAddr, backProbReg, outOffset1, copyOutReg);
                } else {
                    AscendC::MicroAPI::DataCopy(backProbAddr, expReg, outOffset, copyOutReg);
                }
            }
			for (uint16_t k = 0; k < tailLoopTimes; k++) {
                if constexpr (sizeof(T1) == 2) {
                    AscendC::MicroAPI::DataCopy<T1, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(srcReg0, inputUbAddr+i*rOnceNumAlign+repeatTimes1*vfLen);
                    AscendC::MicroAPI::Cast<float, T1, castB16ToB32>(srcRegfp32, srcReg0, preg);
                } else {
                    AscendC::MicroAPI::DataCopy(srcRegfp32, inputUbAddr+i*rOnceNumAlign+repeatTimes1*vfLen);
                }
                AscendC::MicroAPI::Sub(subReg, srcRegfp32, maxReg, preg);
                AscendC::MicroAPI::Exp(expReg, subReg, preg);
                AscendC::MicroAPI::Div(expReg, expReg, sumReg, preg);
                AscendC::MicroAPI::Sub(tmpReg, logReg, subReg, preg);
                AscendC::MicroAPI::DataCopy(outUbAddr+i*rOnceNumAlign+repeatTimes1*vfLen, tmpReg, preg);
                if constexpr (sizeof(T1) == 2) {
                    AscendC::MicroAPI::Cast<T1, float, castB32ToB16>(backProbReg, expReg, preg);
                    AscendC::MicroAPI::DataCopy<T1, StoreDist::DIST_PACK_B32>(backProbAddr+i*rOnceNumAlign+repeatTimes1*vfLen, backProbReg, preg);
                } else {
                    AscendC::MicroAPI::DataCopy(backProbAddr+i*rOnceNumAlign+repeatTimes1*vfLen, expReg, preg);
                }
            }
        }
    }
}

template <typename T1, typename T2, uint64_t schId, uint64_t db>
__aicore__ inline void SparseSoftmaxCrossEntropyWithLogitsSplitR<T1, T2, schId, db>::PipeM2V() {
    event_t eventMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventMTE2ToV);
}

template <typename T1, typename T2, uint64_t schId, uint64_t db>
__aicore__ inline void SparseSoftmaxCrossEntropyWithLogitsSplitR<T1, T2, schId, db>::Compute(int64_t tileNum, int64_t offset)
{   LocalTensor<float> maxUb = maxBuf_.Get<float>();
    CleanMaxUb(tileNum, maxUb);
    GetRowMax(tileNum, offset);
    ComputeSubExpReduceSum(tileNum, offset);
    ComputeRst(tileNum, offset);
}

}
#endif //SPARSE_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_SPLIT_R_H