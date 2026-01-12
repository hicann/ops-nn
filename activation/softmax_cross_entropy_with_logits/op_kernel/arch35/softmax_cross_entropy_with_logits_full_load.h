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
 * \file softmax_cross_entropy_with_logits_full_load.h
 * \brief
 */

#ifndef SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_H
#define SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_H

#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "softmax_cross_entropy_with_logits_tiling_data.h"

namespace SoftmaxCrossEntropyWithLogits {
using namespace AscendC;
using AscendC::MicroAPI::LoadDist;
using AscendC::MicroAPI::MaskPattern;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::MaskUnPack;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::StoreDist;
using AscendC::MicroAPI::UpdateMask;

template <typename T, uint64_t schId, uint64_t featuresBrc, uint64_t labelsBrc, uint64_t db>
class SoftmaxCrossEntropyWithLogitsFullLoad {
public:
    __aicore__ inline SoftmaxCrossEntropyWithLogitsFullLoad() {};
    __aicore__ inline void Init(GM_ADDR features, GM_ADDR labels, GM_ADDR loss, GM_ADDR backProp, GM_ADDR workspace,  const SoftmaxCrossEntropyWithLogitsTilingData *tilingData, TPipe *pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void VfSubExp(int64_t tileNum, int64_t r, int64_t rAlign, LocalTensor<float> maxBuf, LocalTensor<T> featuresBuf, LocalTensor<float> subBuf, LocalTensor<float> temp1Buf);
    __aicore__ inline void VfBackProp(int64_t tileNum, int64_t r, int64_t rAlign, LocalTensor<float> sumBuf, LocalTensor<float> temp1Buf, LocalTensor<float> logBuf, LocalTensor<T> labelsBuf, LocalTensor<T> backPropBuf, LocalTensor<float> temp2Buf, LocalTensor<float> subBuf);
    __aicore__ inline void ProcessEachCore(int64_t tileNum, int64_t tailNum, int64_t loopTimes);
    __aicore__ inline void CopyInPad(TQue<QuePosition::VECIN, 1>& dstQueue, GlobalTensor<T> &srcTensor, int64_t tileNum, int64_t offset);
    __aicore__ inline void CopyOutPadBackProp(GlobalTensor<T> &dstTensor, TQue<QuePosition::VECOUT, 1>& srcQueue, int64_t tileNum, int64_t offset, int64_t rNum);
    __aicore__ inline void CopyOutPadLoss(GlobalTensor<T> &dstTensor, TQue<QuePosition::VECOUT, 1>& srcQueue, int64_t nTailNum, int64_t offset);
    __aicore__ inline void Compute(int64_t tileNum, int64_t aOffset);
	__aicore__ inline void CopyInNDDMA(TQue<QuePosition::VECIN, 1>& dstQueue, GlobalTensor<T> &srcTensor, int64_t tileNum, int64_t aOffset, int64_t brcDim, uint64_t brcNum);
    __aicore__ inline void VfReduceMax(int64_t tileNum, int64_t r, int64_t rAlign, LocalTensor<float> maxBuf, LocalTensor<T> featuresBuf);

protected:
    constexpr static AscendC::MicroAPI::CastTrait castB32ToB16 = { AscendC::MicroAPI::RegLayout::ZERO,
        AscendC::MicroAPI::SatMode::NO_SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT };
    constexpr static AscendC::MicroAPI::CastTrait castB16ToB32 = { AscendC::MicroAPI::RegLayout::ZERO,
        AscendC::MicroAPI::SatMode::UNKNOWN, AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN };

    const static int32_t FP32_DTYPE = 4;
    constexpr static uint32_t DOUBLE_BUFFER = 2;
    constexpr static uint32_t BLOCK_SIZE = Ops::Base::GetUbBlockSize();
    constexpr static uint32_t VL_FP32 = static_cast<int64_t>(Ops::Base::GetVRegSize()) / sizeof(float);

private:
    /* global memory address */
    GlobalTensor<T> featuresGm_;
    GlobalTensor<T> labelsGm_;
    GlobalTensor<T> lossGm_;
    GlobalTensor<T> backPropGm_;

    /* ascendc variable */
    TPipe* pipe_ = nullptr;
    TQue<QuePosition::VECIN, 1> featuresQueue_;
    TQue<QuePosition::VECIN, 1> labelsQueue_;
    TQue<QuePosition::VECOUT, 1> lossQueue_;
    TQue<QuePosition::VECOUT, 1> backPropQueue_;

    TBuf<QuePosition::VECCALC> maxBuf_;
    TBuf<QuePosition::VECCALC> subBuf_;
    TBuf<QuePosition::VECCALC> temp1Buf_;
    TBuf<QuePosition::VECCALC> temp2Buf_;
    TBuf<QuePosition::VECCALC> logBuf_;
    TBuf<QuePosition::VECCALC> sumBuf_;

    int64_t blockIdx_ = 0;
    int64_t dataTypeSize_ = 0;
    bool isLastCore_;
    const SoftmaxCrossEntropyWithLogitsTilingData* tilingData_ = nullptr;
    int32_t vfLenFp32_ = Ops::Base::GetVRegSize() / FP32_DTYPE;
    float MIN_FLOAT = -3.402823466e+38;

	int64_t realCoreNum_;
	int64_t a_;
	int64_t r_;
	int64_t blockFactor_;
	int64_t tailBlockFactor_;
	int64_t rUbNumFactor_;
	int64_t aUbNumFactor_;
	int64_t aLoopTimes_;
	int64_t aLoopTimesT_;
	int64_t aLoopTail_;
	int64_t aLoopTailT_;
	int64_t featuresBrcDim_;
    int64_t labelsBrcDim_;
};

template <typename T, uint64_t schId, uint64_t featuresBrc, uint64_t labelsBrc, uint64_t db>
__aicore__ inline void SoftmaxCrossEntropyWithLogitsFullLoad<T, schId, featuresBrc, labelsBrc, db>::Init(GM_ADDR features, GM_ADDR labels, GM_ADDR loss, GM_ADDR backProp, GM_ADDR workspace,  const SoftmaxCrossEntropyWithLogitsTilingData *tilingData, TPipe *pipe)
{
	realCoreNum_ = tilingData->realCoreNum;
	a_ = tilingData->a;
	r_ = tilingData->r;
	blockFactor_ = tilingData->blockFactor;
	tailBlockFactor_ = tilingData->tailBlockFactor;
	rUbNumFactor_ = tilingData->rUbNumFactor;
	aUbNumFactor_ = tilingData->aUbNumFactor;
	aLoopTimes_ = tilingData->aLoopTimes;
	aLoopTimesT_ = tilingData->aLoopTimesT;
	aLoopTail_ = tilingData->aLoopTail;
	aLoopTailT_ = tilingData->aLoopTailT;
	featuresBrcDim_ = tilingData->featuresBrcDim;
	labelsBrcDim_ = tilingData->labelsBrcDim;

    dataTypeSize_ = sizeof(T);
    pipe_ = pipe;

    featuresGm_.SetGlobalBuffer((__gm__ T *)features);
    labelsGm_.SetGlobalBuffer((__gm__ T *)labels);
    lossGm_.SetGlobalBuffer((__gm__ T *)loss);
    backPropGm_.SetGlobalBuffer((__gm__ T *)backProp);

    int64_t ubBufferSize = rUbNumFactor_ * aUbNumFactor_;
    pipe_->InitBuffer(featuresQueue_, DOUBLE_BUFFER, ubBufferSize * sizeof(T));
    pipe_->InitBuffer(labelsQueue_, DOUBLE_BUFFER, ubBufferSize * sizeof(T));
    pipe_->InitBuffer(lossQueue_, DOUBLE_BUFFER, aUbNumFactor_ * sizeof(T));
    pipe_->InitBuffer(backPropQueue_, DOUBLE_BUFFER, ubBufferSize * sizeof(T));

    pipe_->InitBuffer(maxBuf_, aUbNumFactor_ * sizeof(float));
    pipe_->InitBuffer(subBuf_, ubBufferSize * sizeof(float));
    pipe_->InitBuffer(temp1Buf_, ubBufferSize * sizeof(float));
    pipe_->InitBuffer(temp2Buf_, ubBufferSize * sizeof(float));
    pipe_->InitBuffer(logBuf_, aUbNumFactor_ * sizeof(float));
    pipe_->InitBuffer(sumBuf_, aUbNumFactor_ * sizeof(float));

    blockIdx_ = AscendC::GetBlockIdx();
    isLastCore_ = (blockIdx_ == realCoreNum_ - 1) && (tailBlockFactor_ != 0);
}

template <typename T, uint64_t schId, uint64_t featuresBrc, uint64_t labelsBrc, uint64_t db>
__aicore__ inline void SoftmaxCrossEntropyWithLogitsFullLoad<T, schId, featuresBrc, labelsBrc, db>::Process()
{
	if (blockIdx_ >= realCoreNum_) {
		return;
	}

    if (schId == 2) {
        int64_t processNumA;
        if (blockIdx_ < realCoreNum_ - 1) {
            processNumA = blockFactor_;
        } else {
            processNumA = tailBlockFactor_;
        }
        int64_t startOffset = blockIdx_ * processNumA;
        InitOutput<T>(lossGm_[startOffset], processNumA, T(0));
        return;
    }

    if (isLastCore_) {
        ProcessEachCore(aUbNumFactor_, aLoopTailT_, aLoopTimesT_);
    } else {
        ProcessEachCore(aUbNumFactor_, aLoopTail_, aLoopTimes_);
    }
}

template <typename T, uint64_t schId, uint64_t featuresBrc, uint64_t labelsBrc, uint64_t db>
__aicore__ inline void SoftmaxCrossEntropyWithLogitsFullLoad<T, schId, featuresBrc, labelsBrc, db>::ProcessEachCore(int64_t tileNum, int64_t tailNum, int64_t loopTimes)
{
    int64_t aGmOffset = blockFactor_ * blockIdx_;
    for (int64_t aLoopIndex = 0; aLoopIndex < loopTimes; aLoopIndex++) {
        Compute(tileNum, aLoopIndex * tileNum + aGmOffset);
    }
    if (tailNum > 0) {
        Compute(tailNum, loopTimes * tileNum  + aGmOffset);
    }
}

template <typename T, uint64_t schId, uint64_t featuresBrc, uint64_t labelsBrc, uint64_t db>
__aicore__ inline void SoftmaxCrossEntropyWithLogitsFullLoad<T, schId, featuresBrc, labelsBrc, db>::Compute(int64_t tileNum, int64_t aOffset)
{
	if constexpr(featuresBrc != 0) {
        CopyInNDDMA(featuresQueue_, featuresGm_, tileNum, aOffset, featuresBrcDim_, featuresBrc);
    } else {
		CopyInPad(featuresQueue_, featuresGm_, tileNum, aOffset * r_);
	}
	LocalTensor<T> featuresBuf = featuresQueue_.DeQue<T>();
	LocalTensor<float> maxBuf = maxBuf_.Get<float>();

	uint32_t srcShape[2] = {static_cast<uint32_t>(tileNum), static_cast<uint32_t>(rUbNumFactor_)};
    VfReduceMax(tileNum, r_, rUbNumFactor_, maxBuf, featuresBuf);

	LocalTensor<float> subBuf = subBuf_.Get<float>();
	LocalTensor<float> temp1Buf = temp1Buf_.Get<float>();
	VfSubExp(tileNum, r_, rUbNumFactor_, maxBuf, featuresBuf, subBuf, temp1Buf);
	featuresQueue_.FreeTensor(featuresBuf);

	LocalTensor<float> sumBuf = sumBuf_.Get<float>();
	AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, false>(sumBuf, temp1Buf, srcShape, false);

	if constexpr(labelsBrc != 0) {
        CopyInNDDMA(labelsQueue_, labelsGm_, tileNum, aOffset, labelsBrcDim_, labelsBrc);
    } else {
		CopyInPad(labelsQueue_, labelsGm_, tileNum, aOffset * r_);
 	}
	LocalTensor<T> labelsBuf = labelsQueue_.DeQue<T>();
	LocalTensor<float> logBuf = logBuf_.Get<float>();
    LocalTensor<float> temp2Buf = temp2Buf_.Get<float>();
	LocalTensor<T> backPropBuf = backPropQueue_.AllocTensor<T>();
	VfBackProp(tileNum, r_, rUbNumFactor_, sumBuf, temp1Buf, logBuf, labelsBuf, backPropBuf, temp2Buf, subBuf);
	backPropQueue_.EnQue<T>(backPropBuf);
	CopyOutPadBackProp(backPropGm_, backPropQueue_, tileNum, aOffset * r_, r_);

	LocalTensor<T> lossBuf = lossQueue_.AllocTensor<T>();
	AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(temp1Buf, temp2Buf, srcShape, false);
    if constexpr (sizeof(T) == 2) {
		AscendC::Cast(lossBuf, temp1Buf, AscendC::RoundMode::CAST_RINT, tileNum);
	} else {
		AscendC::Copy(lossBuf, temp1Buf, tileNum);
	}
	lossQueue_.EnQue<T>(lossBuf);

	CopyOutPadLoss(lossGm_, lossQueue_, tileNum, aOffset);
	featuresQueue_.FreeTensor(featuresBuf);
	labelsQueue_.FreeTensor(labelsBuf);
}

template <typename T, uint64_t schId, uint64_t featuresBrc, uint64_t labelsBrc, uint64_t db>
__aicore__ inline void SoftmaxCrossEntropyWithLogitsFullLoad<T, schId, featuresBrc, labelsBrc, db>::CopyInPad(TQue<QuePosition::VECIN, 1>& dstQueue, GlobalTensor<T> &srcTensor, int64_t tileNum, int64_t offset)
{
    LocalTensor<T> dstBuf = dstQueue.AllocTensor<T>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = tileNum;
    copyParams.blockLen = r_ * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;

	T minVal = -3.402823466e+38;
	if constexpr (IsSameType<T, half>::value) {
		minVal = -65504;
	}
	int64_t rNumAlign = rUbNumFactor_;
	DataCopyPadExtParams<T> padParams;
	padParams.isPad = true;
	padParams.leftPadding = 0;
	padParams.rightPadding = rNumAlign - r_;
	padParams.paddingValue = minVal;
    AscendC::DataCopyPad(dstBuf, srcTensor[offset], copyParams, padParams);
    dstQueue.EnQue<T>(dstBuf);
}

template <typename T, uint64_t schId, uint64_t featuresBrc, uint64_t labelsBrc, uint64_t db>
__aicore__ inline void SoftmaxCrossEntropyWithLogitsFullLoad<T, schId, featuresBrc, labelsBrc, db>::CopyInNDDMA(TQue<QuePosition::VECIN, 1>& dstQueue, GlobalTensor<T> &srcTensor, int64_t tileNum, int64_t aOffset, int64_t brcDim, uint64_t brcNum)
{
	int64_t rNumAlign = rUbNumFactor_;
    LocalTensor<T> dstBuf = dstQueue.AllocTensor<T>();
    T constValue = -3.402823466e+38;
	if constexpr (IsSameType<T, half>::value) {
		constValue = -65504;
	}
    static constexpr MultiCopyConfig config = { false };
    MultiCopyLoopInfo<2> loopInfo;
    loopInfo.loopSize[0] = r_;
    loopInfo.loopSize[1] = tileNum;
    loopInfo.loopLpSize[0] = 0;
    loopInfo.loopRpSize[0] = rNumAlign - r_;
    loopInfo.loopLpSize[1] = 0;
    loopInfo.loopRpSize[1] = 0;
    int64_t offset = brcDim == 1 ? aOffset : 0;
    if (brcNum == 2) {
        loopInfo.loopSrcStride[0] = 0;
        loopInfo.loopSrcStride[1] = 0;
        offset = 0;
    } else {
        loopInfo.loopSrcStride[0] = 1 - brcDim;
        loopInfo.loopSrcStride[1] = brcDim;
    }
    loopInfo.loopDstStride[0] = 1;
    loopInfo.loopDstStride[1] = rNumAlign;
    MultiCopyParams<T, 2> paramsMain = { loopInfo, constValue };
    DataCopy<T, 2, config>(dstBuf, srcTensor[offset], paramsMain);
    dstQueue.EnQue<T>(dstBuf);
}

template <typename T, uint64_t schId, uint64_t featuresBrc, uint64_t labelsBrc, uint64_t db>
__aicore__ inline void SoftmaxCrossEntropyWithLogitsFullLoad<T, schId, featuresBrc, labelsBrc, db>::CopyOutPadBackProp(GlobalTensor<T> &dstTensor, TQue<QuePosition::VECOUT, 1>& srcQueue, int64_t tileNum, int64_t offset, int64_t rNum)
{
    LocalTensor<T> srcBuf = srcQueue.DeQue<T>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = tileNum;
    copyParams.blockLen = rNum * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPad(dstTensor[offset], srcBuf, copyParams);
    srcQueue.FreeTensor(srcBuf);
}

template <typename T, uint64_t schId, uint64_t featuresBrc, uint64_t labelsBrc, uint64_t db>
__aicore__ inline void SoftmaxCrossEntropyWithLogitsFullLoad<T, schId, featuresBrc, labelsBrc, db>::CopyOutPadLoss(GlobalTensor<T> &dstTensor, TQue<QuePosition::VECOUT, 1>& srcQueue, int64_t nTailNum, int64_t offset)
{
    LocalTensor<T> srcBuf = srcQueue.DeQue<T>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = nTailNum * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPad(dstTensor[offset], srcBuf, copyParams);
    srcQueue.FreeTensor(srcBuf);
}

template <typename T, uint64_t schId, uint64_t featuresBrc, uint64_t labelsBrc, uint64_t db>
__aicore__ inline void SoftmaxCrossEntropyWithLogitsFullLoad<T, schId, featuresBrc, labelsBrc, db>::VfReduceMax(int64_t tileNum, int64_t r, int64_t rAlign, LocalTensor<float> maxBuf, LocalTensor<T> featuresBuf)
{
    uint16_t aTimes = tileNum;
	uint32_t vfLen = Ops::Base::GetVRegSize() / sizeof(T);
	uint16_t repeatTimes = r / vfLen;
	uint32_t tailNum = r % vfLen;
	uint16_t tailLoop = tailNum != 0 ? 1 : 0;

    auto maxAddr = (__ubuf__ float *)maxBuf.GetPhyAddr();
    auto featuresAddr = (__ubuf__ T *)featuresBuf.GetPhyAddr();
    auto featuresAddr1 = (__ubuf__ T *)featuresBuf.GetPhyAddr();
    T minValue = MIN_FLOAT;
    if constexpr (IsSameType<T, half>::value) {
        minValue = -65504;
    }

    __VEC_SCOPE__
	{
		AscendC::MicroAPI::RegTensor<T> featuresReg1;
		AscendC::MicroAPI::RegTensor<float> maxReg;
        AscendC::MicroAPI::RegTensor<T> featuresReg;
        AscendC::MicroAPI::RegTensor<T> featuresRegLowest;
        AscendC::MicroAPI::RegTensor<T> featuresRegHighest;
        AscendC::MicroAPI::RegTensor<float> featuresRegLowest32;
        AscendC::MicroAPI::RegTensor<float> featuresRegHighest32;
        AscendC::MicroAPI::RegTensor<float> maxRegTemp;

		AscendC::MicroAPI::MaskReg pregMain = AscendC::MicroAPI::CreateMask<T, AscendC::MicroAPI::MaskPattern::ALL>();
		AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::UpdateMask<T>(tailNum);
		AscendC::MicroAPI::MaskReg pregReduce = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg mergePreg = AscendC::MicroAPI::CreateMask<float, MaskPattern::VL1>();
		AscendC::MicroAPI::MaskReg comparePreg;

		for(uint16_t i = 0; i < aTimes; i++) {
			AscendC::MicroAPI::Duplicate(featuresReg, minValue);
            AscendC::MicroAPI::DataCopy(featuresReg1, featuresAddr + i * rAlign + repeatTimes * vfLen);
            AscendC::MicroAPI::Max(featuresReg1, featuresReg, featuresReg1, preg);
            AscendC::MicroAPI::Copy<T, AscendC::MicroAPI::MaskMergeMode::MERGING>(featuresReg, featuresReg1, preg);
            for(uint16_t j = 0; j < repeatTimes; j++) {
                AscendC::MicroAPI::AddrReg offset = AscendC::MicroAPI::CreateAddrReg<T>(i, rAlign, j, vfLen);
                AscendC::MicroAPI::DataCopy(featuresReg1, featuresAddr1, offset);
                AscendC::MicroAPI::Max(featuresReg, featuresReg1, featuresReg, pregMain);
            }
            if constexpr (sizeof(T) == 2) {
                AscendC::MicroAPI::UnPack<int32_t, int16_t, AscendC::MicroAPI::HighLowPart::LOWEST>(
                    (AscendC::MicroAPI::RegTensor<int32_t>&)featuresRegLowest,
                    (AscendC::MicroAPI::RegTensor<int16_t>&)featuresReg);
                AscendC::MicroAPI::UnPack<int32_t, int16_t, AscendC::MicroAPI::HighLowPart::HIGHEST>(
                    (AscendC::MicroAPI::RegTensor<int32_t>&)featuresRegHighest,
                    (AscendC::MicroAPI::RegTensor<int16_t>&)featuresReg);
				AscendC::MicroAPI::Cast<float, T, castB16ToB32>(featuresRegLowest32, featuresRegLowest, pregReduce);
				AscendC::MicroAPI::Cast<float, T, castB16ToB32>(featuresRegHighest32, featuresRegHighest, pregReduce);
                AscendC::MicroAPI::Max(maxRegTemp, featuresRegLowest32, featuresRegHighest32, pregReduce);
			    AscendC::MicroAPI::ReduceMax(maxReg, maxRegTemp, pregReduce);
		    } else {
			    AscendC::MicroAPI::ReduceMax(maxReg, featuresReg, pregReduce);
		    }
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(maxAddr + i, maxReg, mergePreg);
		}
	}
}

template <typename T, uint64_t schId, uint64_t featuresBrc, uint64_t labelsBrc, uint64_t db>
__aicore__ inline void SoftmaxCrossEntropyWithLogitsFullLoad<T, schId, featuresBrc, labelsBrc, db>::VfSubExp(int64_t tileNum, int64_t r, int64_t rAlign, LocalTensor<float> maxBuf, LocalTensor<T> featuresBuf, LocalTensor<float> subBuf, LocalTensor<float> temp1Buf)
{
    auto subAddr = (__ubuf__ float *)subBuf.GetPhyAddr();
    auto temp1Addr = (__ubuf__ float *)temp1Buf.GetPhyAddr();
    auto maxAddr = (__ubuf__ float *)maxBuf.GetPhyAddr();
    auto featuresAddr = (__ubuf__ T *)featuresBuf.GetPhyAddr();

	uint16_t aTimes = tileNum;
	uint32_t vfLen = vfLenFp32_;
	uint16_t repeatTimes = r / vfLen;
	uint32_t tailNum = r % vfLen;
	uint16_t tailLoop = tailNum != 0 ? 1 : 0;
	uint32_t tailNumAlign = rAlign - repeatTimes * vfLen;

	__VEC_SCOPE__
	{
		AscendC::MicroAPI::RegTensor<float> temp1Reg;
		AscendC::MicroAPI::RegTensor<float> subReg;

		AscendC::MicroAPI::RegTensor<T> featuresReg;
		AscendC::MicroAPI::RegTensor<float> featuresReg32;
		AscendC::MicroAPI::RegTensor<float> maxReg32;

		AscendC::MicroAPI::MaskReg pregMain = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
		AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::UpdateMask<float>(tailNum);
		AscendC::MicroAPI::MaskReg pregAlign = AscendC::MicroAPI::UpdateMask<float>(tailNumAlign);

		for(uint16_t i = 0; i < aTimes; i++) {
			AscendC::MicroAPI::DataCopy<float, LoadDist::DIST_BRC_B32>(maxReg32, maxAddr + i);
			for(uint16_t j = 0; j < repeatTimes; j++) {
				AscendC::MicroAPI::AddrReg offsetT = AscendC::MicroAPI::CreateAddrReg<T>(i, rAlign, j, vfLen);
				AscendC::MicroAPI::AddrReg offset = AscendC::MicroAPI::CreateAddrReg<float>(i, rAlign, j, vfLen);
				if constexpr (sizeof(T) == 2) {
					AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(featuresReg, featuresAddr, offsetT);
					AscendC::MicroAPI::Cast<float, T, castB16ToB32>(featuresReg32, featuresReg, pregMain);
				} else {
					AscendC::MicroAPI::DataCopy(featuresReg32, featuresAddr, offset);
				}
				AscendC::MicroAPI::Sub(subReg, featuresReg32, maxReg32, pregMain);
				AscendC::MicroAPI::Exp(temp1Reg, subReg, pregMain);
				AscendC::MicroAPI::DataCopy(temp1Addr, temp1Reg, offset, pregMain);
				AscendC::MicroAPI::DataCopy(subAddr, subReg, offset, pregMain);
			}

			for(uint16_t k = 0; k < tailLoop; k++) {
				if constexpr (sizeof(T) == 2) {
					AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(featuresReg, featuresAddr+i*rAlign+repeatTimes*vfLen);
					AscendC::MicroAPI::Cast<float, T, castB16ToB32>(featuresReg32, featuresReg, preg);
				} else {
					AscendC::MicroAPI::DataCopy(featuresReg32, featuresAddr+i*rAlign+repeatTimes*vfLen);
				}
				AscendC::MicroAPI::Sub(subReg, featuresReg32, maxReg32, preg);
				AscendC::MicroAPI::Exp(temp1Reg, subReg, preg);
				AscendC::MicroAPI::DataCopy(temp1Addr+i*rAlign+repeatTimes*vfLen, temp1Reg, pregAlign);
				AscendC::MicroAPI::DataCopy(subAddr+i*rAlign+repeatTimes*vfLen, subReg, preg);
			}
		}
	}
}

template <typename T, uint64_t schId, uint64_t featuresBrc, uint64_t labelsBrc, uint64_t db>
__aicore__ inline void SoftmaxCrossEntropyWithLogitsFullLoad<T, schId, featuresBrc, labelsBrc, db>::VfBackProp(int64_t tileNum, int64_t r, int64_t rAlign, LocalTensor<float> sumBuf, LocalTensor<float> temp1Buf, LocalTensor<float> logBuf, LocalTensor<T> labelsBuf, LocalTensor<T> backPropBuf, LocalTensor<float> temp2Buf, LocalTensor<float> subBuf)
{
    auto sumAddr = (__ubuf__ float *)sumBuf.GetPhyAddr();
    auto temp1Addr = (__ubuf__ float *)temp1Buf.GetPhyAddr();
    auto logAddr = (__ubuf__ float *)logBuf.GetPhyAddr();
    auto temp2Addr = (__ubuf__ float *)temp2Buf.GetPhyAddr();
    auto subAddr = (__ubuf__ float *)subBuf.GetPhyAddr();
    auto labelsAddr = (__ubuf__ T *)labelsBuf.GetPhyAddr();
    auto backPropAddr = (__ubuf__ T *)backPropBuf.GetPhyAddr();

	uint16_t aTimes = tileNum;
	uint32_t vfLen = vfLenFp32_;
	uint16_t repeatTimes = r / vfLen;
	uint32_t tailNum = r % vfLen;
	uint16_t tailLoop = tailNum != 0 ? 1 : 0;
	uint32_t tailNumAlign = rAlign - repeatTimes * vfLen;

	__VEC_SCOPE__
	{
		AscendC::MicroAPI::RegTensor<float> sumReg;
		AscendC::MicroAPI::RegTensor<float> temp1Reg;
		AscendC::MicroAPI::RegTensor<float> logReg;
		AscendC::MicroAPI::RegTensor<float> subReg;
		AscendC::MicroAPI::RegTensor<float> temp2Reg;
		AscendC::MicroAPI::RegTensor<T> backPropReg;
		AscendC::MicroAPI::RegTensor<float> backPropReg32;

		AscendC::MicroAPI::RegTensor<T> labelsReg;
		AscendC::MicroAPI::RegTensor<float> labelsReg32;

		AscendC::MicroAPI::MaskReg pregMain = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
		AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::UpdateMask<float>(tailNum);
        AscendC::MicroAPI::MaskReg pregAlign = AscendC::MicroAPI::UpdateMask<float>(tailNumAlign);

		for(uint16_t i = 0; i < aTimes; i++) {
			AscendC::MicroAPI::DataCopy<float, LoadDist::DIST_BRC_B32>(sumReg, sumAddr + i);
			for(uint16_t j = 0; j < repeatTimes; j++) {
				AscendC::MicroAPI::AddrReg offsetT = AscendC::MicroAPI::CreateAddrReg<T>(i, rAlign, j, vfLen);
				AscendC::MicroAPI::AddrReg offset = AscendC::MicroAPI::CreateAddrReg<float>(i, rAlign, j, vfLen);
				AscendC::MicroAPI::DataCopy(temp1Reg, temp1Addr, offset);
				AscendC::MicroAPI::Div(temp1Reg, temp1Reg, sumReg, pregMain);
				if constexpr (sizeof(T) == 2) {
					AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(labelsReg, labelsAddr, offsetT);
					AscendC::MicroAPI::Cast<float, T, castB16ToB32>(labelsReg32, labelsReg, pregMain);
				} else {
					AscendC::MicroAPI::DataCopy(labelsReg32, labelsAddr, offset);
				}
				AscendC::MicroAPI::Sub(backPropReg32, temp1Reg, labelsReg32, pregMain);
                if constexpr (sizeof(T) == 2) {
                    AscendC::MicroAPI::Cast<T, float, castB32ToB16>(backPropReg, backPropReg32, pregMain);
					AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(backPropAddr, backPropReg, offsetT, pregMain);
				} else {
					AscendC::MicroAPI::DataCopy(backPropAddr, backPropReg32, offset, pregMain);
				}
                AscendC::MicroAPI::DataCopy(subReg, subAddr, offset);
                AscendC::MicroAPI::Log(logReg, sumReg, pregMain);
				AscendC::MicroAPI::Sub(temp2Reg, logReg, subReg, pregMain);
				AscendC::MicroAPI::Mul(temp2Reg, temp2Reg, labelsReg32, pregMain);
				AscendC::MicroAPI::DataCopy(temp2Addr, temp2Reg, offset, pregMain);
			}

			for(uint16_t k = 0; k < tailLoop; k++) {
				AscendC::MicroAPI::DataCopy(temp1Reg, temp1Addr+i*rAlign+repeatTimes*vfLen);
				AscendC::MicroAPI::Div(temp1Reg, temp1Reg, sumReg, preg);
				if constexpr (sizeof(T) == 2) {
					AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(labelsReg, labelsAddr+i*rAlign+repeatTimes*vfLen);
					AscendC::MicroAPI::Cast<float, T, castB16ToB32>(labelsReg32, labelsReg, preg);
				} else {
					AscendC::MicroAPI::DataCopy(labelsReg32, labelsAddr+i*rAlign+repeatTimes*vfLen);
				}
				AscendC::MicroAPI::Sub(backPropReg32, temp1Reg, labelsReg32, preg);

				if constexpr (sizeof(T) == 2) {
                    AscendC::MicroAPI::Cast<T, float, castB32ToB16>(backPropReg, backPropReg32, preg);
					AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(backPropAddr+i*rAlign+repeatTimes*vfLen, backPropReg, preg);
				} else {
					AscendC::MicroAPI::DataCopy(backPropAddr+i*rAlign+repeatTimes*vfLen, backPropReg32, pregAlign);
				}
                AscendC::MicroAPI::DataCopy(subReg, subAddr+i*rAlign+repeatTimes*vfLen);
                AscendC::MicroAPI::Log(logReg, sumReg, preg);
				AscendC::MicroAPI::Sub(temp2Reg, logReg, subReg, preg);
				AscendC::MicroAPI::Mul(temp2Reg, temp2Reg, labelsReg32, preg);
				AscendC::MicroAPI::DataCopy(temp2Addr+i*rAlign+repeatTimes*vfLen, temp2Reg, pregAlign);
			}
		}
	}
}

}
#endif