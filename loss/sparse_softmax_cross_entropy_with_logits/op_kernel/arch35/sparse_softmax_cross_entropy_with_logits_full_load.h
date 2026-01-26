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
 * \file sparse_softmax_cross_entropy_with_logits_full_load.h
 * \brief
 */

#ifndef SPARSE_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_H
#define SPARSE_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "../inc/platform.h"
#include "sparse_softmax_cross_entropy_with_logits_tiling_data.h"

namespace SparseSoftmaxCrossEntropyWithLogits {
using namespace AscendC;
using AscendC::MicroAPI::LoadDist;
using AscendC::MicroAPI::MaskPattern;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::MaskUnPack;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::StoreDist;
using AscendC::MicroAPI::UpdateMask;

template <typename T1, typename T2, uint64_t schId, uint64_t db>
__simt_vf__ __aicore__ LAUNCH_BOUND(1024) inline void UbSimtComputeLoopRFullLoad(__ubuf__ float* temp1Addr, __ubuf__ float* temp2Addr, __ubuf__ T2* labelsAddr, __ubuf__ float* gatherAddr, const int64_t tileNum, const int64_t rAlign, const int64_t rMax)
{
    for (int64_t index = static_cast<int64_t>(Simt::GetThreadIdx()); index < tileNum; index += static_cast<int64_t>(Simt::GetThreadNum<0>())) {
		ASSERT((0 <= labelsAddr[index] && labelsAddr[index] < rMax) && "lable is not in [0, C)");
        int64_t offset = index * rAlign + labelsAddr[index];
        temp1Addr[offset] -= 1.0f;
        gatherAddr[index] = temp2Addr[offset];
    }
}

template <typename T1, typename T2, uint64_t schId, uint64_t db>
class SparseSoftmaxCrossEntropyWithLogitsFullLoad {
public:
    __aicore__ inline SparseSoftmaxCrossEntropyWithLogitsFullLoad() {};
    __aicore__ inline void Init(GM_ADDR features, GM_ADDR labels, GM_ADDR loss, GM_ADDR backProp, GM_ADDR workspace,  const SparseSoftmaxCrossEntropyWithLogitsTilingData *tilingData, TPipe *pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void VfSubExp(int64_t tileNum, int64_t r, int64_t rAlign, LocalTensor<float> &maxBuf, LocalTensor<T1> &featuresBuf, LocalTensor<float> &subBuf, LocalTensor<float> &temp1Buf);
    __aicore__ inline void VfBackProp(int64_t tileNum, int64_t r, int64_t rAlign, LocalTensor<float> &sumBuf, LocalTensor<float> &temp1Buf, LocalTensor<float> &logBuf, LocalTensor<T2> &labelsBuf, LocalTensor<T1> &backPropBuf, LocalTensor<float> &temp2Buf, LocalTensor<float> &subBuf);
    __aicore__ inline void ProcessEachCore(int64_t tileNum, int64_t tailNum, int64_t loopTimes);
    __aicore__ inline void CopyInPadFeatures(TQue<QuePosition::VECIN, 1>& dstQueue, GlobalTensor<T1> &srcTensor, int64_t tileNum, int64_t offset);
    __aicore__ inline void CopyInPadLabels(TQue<QuePosition::VECIN, 1>& dstQueue, GlobalTensor<T2> &srcTensor, int64_t tileNum, int64_t offset);
    __aicore__ inline void CopyOutPadBackProp(GlobalTensor<T1> &dstTensor, TQue<QuePosition::VECOUT, 1>& srcQueue, int64_t tileNum, int64_t offset, int64_t rNum);
    __aicore__ inline void CopyOutPadLoss(GlobalTensor<T1> &dstTensor, TQue<QuePosition::VECOUT, 1>& srcQueue, int64_t nTailNum, int64_t offset);
    __aicore__ inline void Compute(int64_t tileNum, int64_t aOffset);
    __aicore__ inline void VfReduceMax(int64_t tileNum, int64_t r, int64_t rAlign, LocalTensor<float> &maxBuf, LocalTensor<T1> &featuresBuf);

protected:
    constexpr static AscendC::MicroAPI::CastTrait castB32ToB16 = { AscendC::MicroAPI::RegLayout::ZERO,
        AscendC::MicroAPI::SatMode::NO_SAT, AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT };
    constexpr static AscendC::MicroAPI::CastTrait castB16ToB32 = { AscendC::MicroAPI::RegLayout::ZERO,
        AscendC::MicroAPI::SatMode::UNKNOWN, AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN };

    const static int32_t FP32_DTYPE = 4;
    constexpr static uint32_t DOUBLE_BUFFER_1 = 1;
    constexpr static uint32_t DOUBLE_BUFFER_2 = 2;
    constexpr static uint32_t BLOCK_SIZE = platform::GetUbBlockSize();
    constexpr static uint32_t VL_FP32 = static_cast<int64_t>(platform::GetVRegSize()) / sizeof(float);

private:
    /* global memory address */
    GlobalTensor<T1> featuresGm_;
    GlobalTensor<T2> labelsGm_;
    GlobalTensor<T1> lossGm_;
    GlobalTensor<T1> backPropGm_;

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
    const SparseSoftmaxCrossEntropyWithLogitsTilingData* tilingData_ = nullptr;
    int32_t vfLenFp32_ = platform::GetVRegSize() / FP32_DTYPE;

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
    int64_t ubBufferSize_;
};

template <typename T1, typename T2, uint64_t schId, uint64_t db>
__aicore__ inline void SparseSoftmaxCrossEntropyWithLogitsFullLoad<T1, T2, schId, db>::Init(GM_ADDR features, GM_ADDR labels, GM_ADDR loss, GM_ADDR backProp, GM_ADDR workspace,  const SparseSoftmaxCrossEntropyWithLogitsTilingData *tilingData, TPipe *pipe)
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

    dataTypeSize_ = sizeof(T1);
    pipe_ = pipe;

    featuresGm_.SetGlobalBuffer((__gm__ T1 *)features);
    labelsGm_.SetGlobalBuffer((__gm__ T2 *)labels);
    lossGm_.SetGlobalBuffer((__gm__ T1 *)loss);
    backPropGm_.SetGlobalBuffer((__gm__ T1 *)backProp);

    ubBufferSize_ = rUbNumFactor_ * aUbNumFactor_;
    if constexpr (db == 0) {
        pipe_->InitBuffer(featuresQueue_, DOUBLE_BUFFER_1, ubBufferSize_ * sizeof(T1));
    } else {
        pipe_->InitBuffer(featuresQueue_, DOUBLE_BUFFER_2, ubBufferSize_ * sizeof(T1));
    }
    pipe_->InitBuffer(labelsQueue_, DOUBLE_BUFFER_1, aUbNumFactor_ * sizeof(T2));
    pipe_->InitBuffer(lossQueue_, DOUBLE_BUFFER_1, aUbNumFactor_ * sizeof(T1));
    pipe_->InitBuffer(backPropQueue_, DOUBLE_BUFFER_1, ubBufferSize_ * sizeof(T1));

    pipe_->InitBuffer(maxBuf_, aUbNumFactor_ * sizeof(float));
    pipe_->InitBuffer(subBuf_, ubBufferSize_ * sizeof(float));
    pipe_->InitBuffer(temp1Buf_, ubBufferSize_ * sizeof(float));
    pipe_->InitBuffer(logBuf_, aUbNumFactor_ * sizeof(float));
    pipe_->InitBuffer(sumBuf_, aUbNumFactor_ * sizeof(float));
	pipe_->InitBuffer(temp2Buf_, ubBufferSize_ * sizeof(float));

    blockIdx_ = AscendC::GetBlockIdx();
    isLastCore_ = (blockIdx_ == realCoreNum_ - 1) && (tailBlockFactor_ != 0);
}

template <typename T1, typename T2, uint64_t schId, uint64_t db>
__aicore__ inline void SparseSoftmaxCrossEntropyWithLogitsFullLoad<T1, T2, schId, db>::Process()
{
	if (blockIdx_ >= realCoreNum_) {
		return;
	}

    if (isLastCore_) {
        ProcessEachCore(aUbNumFactor_, aLoopTailT_, aLoopTimesT_);
    } else {
        ProcessEachCore(aUbNumFactor_, aLoopTail_, aLoopTimes_);
    }
}

template <typename T1, typename T2, uint64_t schId, uint64_t db>
__aicore__ inline void SparseSoftmaxCrossEntropyWithLogitsFullLoad<T1, T2, schId, db>::ProcessEachCore(int64_t tileNum, int64_t tailNum, int64_t loopTimes)
{
    int64_t aGmOffset = blockFactor_ * blockIdx_;
    for (int64_t aLoopIndex = 0; aLoopIndex < loopTimes; aLoopIndex++) {
        Compute(tileNum, aLoopIndex * tileNum + aGmOffset);
    }
    if (tailNum > 0) {
        Compute(tailNum, loopTimes * tileNum  + aGmOffset);
    }
}

template <typename T1, typename T2, uint64_t schId, uint64_t db>
__aicore__ inline void SparseSoftmaxCrossEntropyWithLogitsFullLoad<T1, T2, schId, db>::Compute(int64_t tileNum, int64_t aOffset)
{
	CopyInPadFeatures(featuresQueue_, featuresGm_, tileNum, aOffset * r_);
	LocalTensor<T1> featuresBuf = featuresQueue_.DeQue<T1>();
	LocalTensor<float> maxBuf = maxBuf_.Get<float>();

	uint32_t srcShape[2] = {static_cast<uint32_t>(tileNum), static_cast<uint32_t>(rUbNumFactor_)};
    VfReduceMax(tileNum, r_, rUbNumFactor_, maxBuf, featuresBuf);

	LocalTensor<float> subBuf = subBuf_.Get<float>();
	LocalTensor<float> temp1Buf = temp1Buf_.Get<float>();
	VfSubExp(tileNum, r_, rUbNumFactor_, maxBuf, featuresBuf, subBuf, temp1Buf);
	featuresQueue_.FreeTensor(featuresBuf);

	LocalTensor<float> sumBuf = sumBuf_.Get<float>();
	AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, false>(sumBuf, temp1Buf, srcShape, false);

	CopyInPadLabels(labelsQueue_, labelsGm_, tileNum, aOffset);
	LocalTensor<T2> labelsBuf = labelsQueue_.DeQue<T2>();
	LocalTensor<float> logBuf = logBuf_.Get<float>();
    LocalTensor<float> temp2Buf = temp2Buf_.Get<float>();
	LocalTensor<T1> backPropBuf = backPropQueue_.AllocTensor<T1>();
	VfBackProp(tileNum, r_, rUbNumFactor_, sumBuf, temp1Buf, logBuf, labelsBuf, backPropBuf, temp2Buf, subBuf);
    int64_t simtParam1 = rUbNumFactor_;
    int64_t simtParam2 = r_;
    Simt::VF_CALL<UbSimtComputeLoopRFullLoad<T1, T2, schId, db>>(Simt::Dim3{1024}, (__ubuf__ float*)temp1Buf.GetPhyAddr(), (__ubuf__ float*)temp2Buf.GetPhyAddr(), (__ubuf__ T2*)labelsBuf.GetPhyAddr(), (__ubuf__ float*)maxBuf.GetPhyAddr(), tileNum, simtParam1, simtParam2);
    if constexpr (sizeof(T1) == 2) {
        AscendC::Cast(backPropBuf, temp1Buf, AscendC::RoundMode::CAST_RINT, tileNum * rUbNumFactor_);
    } else {
        AscendC::Copy(backPropBuf, temp1Buf, tileNum * rUbNumFactor_);
    }
    backPropQueue_.EnQue<T1>(backPropBuf);
	CopyOutPadBackProp(backPropGm_, backPropQueue_, tileNum, aOffset * r_, r_);

	LocalTensor<T1> lossBuf = lossQueue_.AllocTensor<T1>();
    if constexpr (sizeof(T1) == 2) {
		AscendC::Cast(lossBuf, maxBuf, AscendC::RoundMode::CAST_RINT, tileNum);
	} else {
		AscendC::Copy(lossBuf, maxBuf, tileNum);
	}
	lossQueue_.EnQue<T1>(lossBuf);
	CopyOutPadLoss(lossGm_, lossQueue_, tileNum, aOffset);
	labelsQueue_.FreeTensor(labelsBuf);
}

template <typename T1, typename T2, uint64_t schId, uint64_t db>
__aicore__ inline void SparseSoftmaxCrossEntropyWithLogitsFullLoad<T1, T2, schId, db>::CopyInPadFeatures(TQue<QuePosition::VECIN, 1>& dstQueue, GlobalTensor<T1> &srcTensor, int64_t tileNum, int64_t offset)
{
    LocalTensor<T1> dstBuf = dstQueue.AllocTensor<T1>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = tileNum;
    copyParams.blockLen = r_ * sizeof(T1);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;

	T1 minVal = -3.402823466e+38;
	if constexpr (IsSameType<T1, half>::value) {
		minVal = -65504;
	}
	int64_t rNumAlign = rUbNumFactor_;
	DataCopyPadExtParams<T1> padParams;
	padParams.isPad = true;
	padParams.leftPadding = 0;
	padParams.rightPadding = rNumAlign - r_;
	padParams.paddingValue = minVal;
    AscendC::DataCopyPad(dstBuf, srcTensor[offset], copyParams, padParams);
    dstQueue.EnQue<T1>(dstBuf);
}

template <typename T1, typename T2, uint64_t schId, uint64_t db>
__aicore__ inline void SparseSoftmaxCrossEntropyWithLogitsFullLoad<T1, T2, schId, db>::CopyInPadLabels(TQue<QuePosition::VECIN, 1>& dstQueue, GlobalTensor<T2> &srcTensor, int64_t tileNum, int64_t offset)
{
    LocalTensor<T2> dstBuf = dstQueue.AllocTensor<T2>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = tileNum * sizeof(T2);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPadExtParams<T2> padParams;
    padParams.isPad = false;
    padParams.leftPadding = 0;
    padParams.rightPadding = 0;
    padParams.paddingValue = 0;
    AscendC::DataCopyPad(dstBuf, srcTensor[offset], copyParams, padParams);
    dstQueue.EnQue<T2>(dstBuf);
}

template <typename T1, typename T2, uint64_t schId, uint64_t db>
__aicore__ inline void SparseSoftmaxCrossEntropyWithLogitsFullLoad<T1, T2, schId, db>::CopyOutPadBackProp(GlobalTensor<T1> &dstTensor, TQue<QuePosition::VECOUT, 1>& srcQueue, int64_t tileNum, int64_t offset, int64_t rNum)
{
    LocalTensor<T1> srcBuf = srcQueue.DeQue<T1>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = tileNum;
    copyParams.blockLen = rNum * sizeof(T1);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPad(dstTensor[offset], srcBuf, copyParams);
    srcQueue.FreeTensor(srcBuf);
}

template <typename T1, typename T2, uint64_t schId, uint64_t db>
__aicore__ inline void SparseSoftmaxCrossEntropyWithLogitsFullLoad<T1, T2, schId, db>::CopyOutPadLoss(GlobalTensor<T1> &dstTensor, TQue<QuePosition::VECOUT, 1>& srcQueue, int64_t nTailNum, int64_t offset)
{
    LocalTensor<T1> srcBuf = srcQueue.DeQue<T1>();
    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = nTailNum * sizeof(T1);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPad(dstTensor[offset], srcBuf, copyParams);
    srcQueue.FreeTensor(srcBuf);
}

template <typename T1, typename T2, uint64_t schId, uint64_t db>
__aicore__ inline void SparseSoftmaxCrossEntropyWithLogitsFullLoad<T1, T2, schId, db>::VfReduceMax(int64_t tileNum, int64_t r, int64_t rAlign, LocalTensor<float> &maxBuf, LocalTensor<T1> &featuresBuf)
{
    uint16_t aTimes = tileNum;
	uint32_t vfLen = platform::GetVRegSize() / sizeof(T1);
	uint16_t repeatTimes = r / vfLen;
	uint32_t tailNum = r % vfLen;
	uint16_t tailLoop = tailNum != 0 ? 1 : 0;
	T1 minVal = -3.402823466e+38;
	if constexpr (IsSameType<T1, half>::value) {
		minVal = -65504;
	}
    auto maxAddr = (__ubuf__ float *)maxBuf.GetPhyAddr();
    auto featuresAddr = (__ubuf__ T1 *)featuresBuf.GetPhyAddr();
    auto featuresAddr1 = (__ubuf__ T1 *)featuresBuf.GetPhyAddr();

	__VEC_SCOPE__
	{
		AscendC::MicroAPI::RegTensor<T1> featuresReg1;
		AscendC::MicroAPI::RegTensor<float> maxReg;
        AscendC::MicroAPI::RegTensor<T1> featuresReg;
        AscendC::MicroAPI::RegTensor<T1> featuresRegLowest;
        AscendC::MicroAPI::RegTensor<T1> featuresRegHighest;
        AscendC::MicroAPI::RegTensor<float> featuresRegLowest32;
        AscendC::MicroAPI::RegTensor<float> featuresRegHighest32;
        AscendC::MicroAPI::RegTensor<float> maxRegTemp;

		AscendC::MicroAPI::MaskReg pregMain = AscendC::MicroAPI::CreateMask<T1, AscendC::MicroAPI::MaskPattern::ALL>();
		AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::UpdateMask<T1>(tailNum);
		AscendC::MicroAPI::MaskReg pregReduce = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg mergePreg = AscendC::MicroAPI::CreateMask<float, MaskPattern::VL1>();
		AscendC::MicroAPI::MaskReg comparePreg;

		for(uint16_t i = 0; i < aTimes; i++) {
			AscendC::MicroAPI::Duplicate(featuresReg, minVal);
            AscendC::MicroAPI::DataCopy(featuresReg1, featuresAddr + i * rAlign + repeatTimes * vfLen);
            AscendC::MicroAPI::Max(featuresReg1, featuresReg, featuresReg1, preg);
            AscendC::MicroAPI::Copy<T1, AscendC::MicroAPI::MaskMergeMode::MERGING>(featuresReg, featuresReg1, preg);
            for(uint16_t j = 0; j < repeatTimes; j++) {
                AscendC::MicroAPI::AddrReg offset = AscendC::MicroAPI::CreateAddrReg<T1>(i, rAlign, j, vfLen);
                AscendC::MicroAPI::DataCopy(featuresReg1, featuresAddr1, offset);
                AscendC::MicroAPI::Max(featuresReg, featuresReg1, featuresReg, pregMain);
            }
            if constexpr (sizeof(T1) == 2) {
                AscendC::MicroAPI::UnPack<int32_t, int16_t, AscendC::MicroAPI::HighLowPart::LOWEST>(
                    (AscendC::MicroAPI::RegTensor<int32_t>&)featuresRegLowest,
                    (AscendC::MicroAPI::RegTensor<int16_t>&)featuresReg);
                AscendC::MicroAPI::UnPack<int32_t, int16_t, AscendC::MicroAPI::HighLowPart::HIGHEST>(
                    (AscendC::MicroAPI::RegTensor<int32_t>&)featuresRegHighest,
                    (AscendC::MicroAPI::RegTensor<int16_t>&)featuresReg);
				AscendC::MicroAPI::Cast<float, T1, castB16ToB32>(featuresRegLowest32, featuresRegLowest, pregReduce);
				AscendC::MicroAPI::Cast<float, T1, castB16ToB32>(featuresRegHighest32, featuresRegHighest, pregReduce);
                AscendC::MicroAPI::Max(maxRegTemp, featuresRegLowest32, featuresRegHighest32, pregReduce);
			    AscendC::MicroAPI::ReduceMax(maxReg, maxRegTemp, pregReduce);
		    } else {
			    AscendC::MicroAPI::ReduceMax(maxReg, featuresReg, pregReduce);
		    }
            DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(maxAddr + i, maxReg, mergePreg);
		}
	}
}

template <typename T1, typename T2, uint64_t schId, uint64_t db>
__aicore__ inline void SparseSoftmaxCrossEntropyWithLogitsFullLoad<T1, T2, schId, db>::VfSubExp(int64_t tileNum, int64_t r, int64_t rAlign, LocalTensor<float> &maxBuf, LocalTensor<T1> &featuresBuf, LocalTensor<float> &subBuf, LocalTensor<float> &temp1Buf)
{
    auto subAddr = (__ubuf__ float *)subBuf.GetPhyAddr();
    auto temp1Addr = (__ubuf__ float *)temp1Buf.GetPhyAddr();
    auto maxAddr = (__ubuf__ float *)maxBuf.GetPhyAddr();
    auto featuresAddr = (__ubuf__ T1 *)featuresBuf.GetPhyAddr();

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

		AscendC::MicroAPI::RegTensor<T1> featuresReg;
		AscendC::MicroAPI::RegTensor<float> featuresReg32;
		AscendC::MicroAPI::RegTensor<float> maxReg32;

		AscendC::MicroAPI::MaskReg pregMain = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
		AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::UpdateMask<float>(tailNum);
		AscendC::MicroAPI::MaskReg pregAlign = AscendC::MicroAPI::UpdateMask<float>(tailNumAlign);

		for(uint16_t i = 0; i < aTimes; i++) {
			AscendC::MicroAPI::DataCopy<float, LoadDist::DIST_BRC_B32>(maxReg32, maxAddr + i);
			for(uint16_t j = 0; j < repeatTimes; j++) {
				AscendC::MicroAPI::AddrReg offsetT = AscendC::MicroAPI::CreateAddrReg<T1>(i, rAlign, j, vfLen);
				AscendC::MicroAPI::AddrReg offset = AscendC::MicroAPI::CreateAddrReg<float>(i, rAlign, j, vfLen);
				if constexpr (sizeof(T1) == 2) {
					AscendC::MicroAPI::DataCopy<T1, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(featuresReg, featuresAddr, offsetT);
					AscendC::MicroAPI::Cast<float, T1, castB16ToB32>(featuresReg32, featuresReg, pregMain);
				} else {
					AscendC::MicroAPI::DataCopy(featuresReg32, featuresAddr, offset);
				}
				AscendC::MicroAPI::Sub(subReg, featuresReg32, maxReg32, pregMain);
				AscendC::MicroAPI::Exp(temp1Reg, subReg, pregMain);
				AscendC::MicroAPI::DataCopy(temp1Addr, temp1Reg, offset, pregMain);
				AscendC::MicroAPI::DataCopy(subAddr, subReg, offset, pregMain);
			}

			for(uint16_t k = 0; k < tailLoop; k++) {
				if constexpr (sizeof(T1) == 2) {
					AscendC::MicroAPI::DataCopy<T1, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(featuresReg, featuresAddr+i*rAlign+repeatTimes*vfLen);
					AscendC::MicroAPI::Cast<float, T1, castB16ToB32>(featuresReg32, featuresReg, preg);
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

template <typename T1, typename T2, uint64_t schId, uint64_t db>
__aicore__ inline void SparseSoftmaxCrossEntropyWithLogitsFullLoad<T1, T2, schId, db>::VfBackProp(int64_t tileNum, int64_t r, int64_t rAlign, LocalTensor<float> &sumBuf, LocalTensor<float> &temp1Buf, LocalTensor<float> &logBuf, LocalTensor<T2> &labelsBuf, LocalTensor<T1> &backPropBuf, LocalTensor<float> &temp2Buf, LocalTensor<float> &subBuf)
{
    auto sumAddr = (__ubuf__ float *)sumBuf.GetPhyAddr();
    auto temp1Addr = (__ubuf__ float *)temp1Buf.GetPhyAddr();
    auto logAddr = (__ubuf__ float *)logBuf.GetPhyAddr();
    auto temp2Addr = (__ubuf__ float *)temp2Buf.GetPhyAddr();
    auto subAddr = (__ubuf__ float *)subBuf.GetPhyAddr();
    auto labelsAddr = (__ubuf__ T2 *)labelsBuf.GetPhyAddr();
    auto backPropAddr = (__ubuf__ T1 *)backPropBuf.GetPhyAddr();

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
		AscendC::MicroAPI::RegTensor<T1> backPropReg;
		AscendC::MicroAPI::RegTensor<float> backPropReg32;

		AscendC::MicroAPI::RegTensor<T2> labelsReg;
		AscendC::MicroAPI::RegTensor<float> labelsReg32;

		AscendC::MicroAPI::MaskReg pregMain = AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
		AscendC::MicroAPI::MaskReg preg = AscendC::MicroAPI::UpdateMask<float>(tailNum);
        AscendC::MicroAPI::MaskReg pregAlign = AscendC::MicroAPI::UpdateMask<float>(tailNumAlign);

		for(uint16_t i = 0; i < aTimes; i++) {
			AscendC::MicroAPI::DataCopy<float, LoadDist::DIST_BRC_B32>(sumReg, sumAddr + i);
			for(uint16_t j = 0; j < repeatTimes; j++) {
				AscendC::MicroAPI::AddrReg offsetT = AscendC::MicroAPI::CreateAddrReg<T1>(i, rAlign, j, vfLen);
				AscendC::MicroAPI::AddrReg offset = AscendC::MicroAPI::CreateAddrReg<float>(i, rAlign, j, vfLen);
				AscendC::MicroAPI::DataCopy(temp1Reg, temp1Addr, offset);
			    AscendC::MicroAPI::DataCopy(subReg, subAddr, offset);
				AscendC::MicroAPI::Div(temp1Reg, temp1Reg, sumReg, pregMain);
                AscendC::MicroAPI::DataCopy(temp1Addr, temp1Reg, offset, pregMain);
                AscendC::MicroAPI::Log(logReg, sumReg, pregMain);
				AscendC::MicroAPI::Sub(temp2Reg, logReg, subReg, pregMain);
				AscendC::MicroAPI::DataCopy(temp2Addr, temp2Reg, offset, pregMain);
			}

			for(uint16_t k = 0; k < tailLoop; k++) {
				AscendC::MicroAPI::DataCopy(temp1Reg, temp1Addr+i*rAlign+repeatTimes*vfLen);
			    AscendC::MicroAPI::DataCopy(subReg, subAddr+i*rAlign+repeatTimes*vfLen);
				AscendC::MicroAPI::Div(temp1Reg, temp1Reg, sumReg, preg);
				AscendC::MicroAPI::DataCopy(temp1Addr+i*rAlign+repeatTimes*vfLen, temp1Reg, preg);
                AscendC::MicroAPI::Log(logReg, sumReg, preg);
				AscendC::MicroAPI::Sub(temp2Reg, logReg, subReg, preg);
				AscendC::MicroAPI::DataCopy(temp2Addr+i*rAlign+repeatTimes*vfLen, temp2Reg, preg);
			}
		}
	}
}

}
#endif