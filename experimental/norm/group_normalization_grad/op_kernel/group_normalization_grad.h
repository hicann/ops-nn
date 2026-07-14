/**
 * This file is part of the OpenBOAT project at Harbin Institute of Technology (HIT)
 * and is contributed to the CANN Open Software.
 *
 * Copyright (c) 2026 AISS Group, Harbin Institute of Technology (HIT).
 * All Rights Reserved.
 *
 * Authors (accounts):
 * - Zhou Jianhua <@LePenseur>
 * - Su Tonghua <@sutonghua>
 *
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file group_normalization_grad.h
 * \brief
 */
#ifndef GROUP_NORMALIZATION_GRAD_H
#define GROUP_NORMALIZATION_GRAD_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "group_normalization_grad_tiling_data.h"
#include "group_normalization_grad_tiling_key.h"

namespace NsGroupNormalizationGrad {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename TYPE_X>
class KernelGroupNormalizationGrad {
public:
    __aicore__ inline KernelGroupNormalizationGrad(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR dy, GM_ADDR gamma, GM_ADDR mean, GM_ADDR rstd, GM_ADDR dx,
                                uint64_t groupElemNum, uint64_t groupCount, uint64_t smallCoreGroupNum,
                                uint64_t bigCoreGroupNum, uint64_t finalGroupTileNum, uint64_t tileDataNum,
                                uint64_t alignedTileDataNum, uint64_t tailDataNum, uint64_t tailBlockNum,
                                float groupElemNumFloat, float groupElemNumReciprocal);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(uint32_t groupIdx, uint32_t segmentIdx, uint32_t currentDataNum);
    __aicore__ inline void CopyOut(uint32_t groupIdx, uint32_t segmentIdx, uint32_t currentDataNum);
    __aicore__ inline void ProcessSingleGroup(uint32_t groupIdx);

    __aicore__ inline float ReadScalar(AscendC::GlobalTensor<TYPE_X>& tensor, uint32_t index);
    __aicore__ inline void CastToFloat(AscendC::LocalTensor<float> dst, AscendC::LocalTensor<TYPE_X> src,
                                       uint32_t dataNum);
    __aicore__ inline void CastFromFloat(AscendC::LocalTensor<TYPE_X> dst, AscendC::LocalTensor<float> src,
                                         uint32_t dataNum);
    __aicore__ inline uint32_t GetCurrentDataNum(uint32_t segmentIdx) const;

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueDY;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueGamma;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueDX;

    AscendC::TBuf<AscendC::TPosition::VECCALC> xFloatBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> dyFloatBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> gammaFloatBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> xhatBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmp0Buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmp1Buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmp2Buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tmp3Buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> reduceBuf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> scalarBuf;

    AscendC::GlobalTensor<TYPE_X> xGm;
    AscendC::GlobalTensor<TYPE_X> dyGm;
    AscendC::GlobalTensor<TYPE_X> gammaGm;
    AscendC::GlobalTensor<TYPE_X> meanGm;
    AscendC::GlobalTensor<TYPE_X> rstdGm;
    AscendC::GlobalTensor<TYPE_X> dxGm;

    uint64_t groupElemNum;
    uint64_t groupCount;
    uint64_t coreGroupNum;
    uint64_t groupOffset;
    uint64_t groupTileNum;
    uint64_t tileDataNum;
    uint64_t alignedTileDataNum;
    uint64_t tailDataNum;
    float groupElemNumFloat;
    float groupElemNumReciprocal;
};

template <typename TYPE_X>
__aicore__ inline void KernelGroupNormalizationGrad<TYPE_X>::Init(GM_ADDR x, GM_ADDR dy, GM_ADDR gamma, GM_ADDR mean,
                                                                  GM_ADDR rstd, GM_ADDR dx, uint64_t groupElemNum,
                                                                  uint64_t groupCount, uint64_t smallCoreGroupNum,
                                                                  uint64_t bigCoreGroupNum, uint64_t finalGroupTileNum,
                                                                  uint64_t tileDataNum, uint64_t alignedTileDataNum,
                                                                  uint64_t tailDataNum, uint64_t tailBlockNum,
                                                                  float groupElemNumFloat, float groupElemNumReciprocal)
{
    ASSERT(AscendC::GetBlockNum() != 0 && "block dim can not be zero!");
    this->groupElemNum = groupElemNum;
    this->groupCount = groupCount;
    this->tileDataNum = tileDataNum;
    this->alignedTileDataNum = alignedTileDataNum;
    this->tailDataNum = tailDataNum;
    this->groupElemNumFloat = groupElemNumFloat;
    this->groupElemNumReciprocal = groupElemNumReciprocal;

    uint64_t blockIdx = AscendC::GetBlockIdx();
    uint64_t groupOffsetVal = bigCoreGroupNum * blockIdx;
    if (blockIdx < tailBlockNum) {
        this->coreGroupNum = bigCoreGroupNum;
    } else {
        this->coreGroupNum = smallCoreGroupNum;
        groupOffsetVal -= (bigCoreGroupNum - smallCoreGroupNum) * (blockIdx - tailBlockNum);
    }
    this->groupOffset = groupOffsetVal;
    this->groupTileNum = finalGroupTileNum;

    xGm.SetGlobalBuffer((__gm__ TYPE_X*)x, groupCount * groupElemNum);
    dyGm.SetGlobalBuffer((__gm__ TYPE_X*)dy, groupCount * groupElemNum);
    gammaGm.SetGlobalBuffer((__gm__ TYPE_X*)gamma, groupCount * groupElemNum);
    meanGm.SetGlobalBuffer((__gm__ TYPE_X*)mean, groupCount);
    rstdGm.SetGlobalBuffer((__gm__ TYPE_X*)rstd, groupCount);
    dxGm.SetGlobalBuffer((__gm__ TYPE_X*)dx, groupCount * groupElemNum);

    pipe.InitBuffer(inQueueX, BUFFER_NUM, this->alignedTileDataNum * sizeof(TYPE_X));
    pipe.InitBuffer(inQueueDY, BUFFER_NUM, this->alignedTileDataNum * sizeof(TYPE_X));
    pipe.InitBuffer(inQueueGamma, BUFFER_NUM, this->alignedTileDataNum * sizeof(TYPE_X));
    pipe.InitBuffer(outQueueDX, BUFFER_NUM, this->alignedTileDataNum * sizeof(TYPE_X));

    pipe.InitBuffer(xFloatBuf, this->alignedTileDataNum * sizeof(float));
    pipe.InitBuffer(dyFloatBuf, this->alignedTileDataNum * sizeof(float));
    pipe.InitBuffer(gammaFloatBuf, this->alignedTileDataNum * sizeof(float));
    pipe.InitBuffer(xhatBuf, this->alignedTileDataNum * sizeof(float));
    pipe.InitBuffer(tmp0Buf, this->alignedTileDataNum * sizeof(float));
    pipe.InitBuffer(tmp1Buf, this->alignedTileDataNum * sizeof(float));
    pipe.InitBuffer(tmp2Buf, this->alignedTileDataNum * sizeof(float));
    pipe.InitBuffer(tmp3Buf, this->alignedTileDataNum * sizeof(float));
    pipe.InitBuffer(reduceBuf, this->alignedTileDataNum * sizeof(float));
    pipe.InitBuffer(scalarBuf, 8 * sizeof(float));
}

template <typename TYPE_X>
__aicore__ inline void KernelGroupNormalizationGrad<TYPE_X>::Process()
{
    uint32_t groupIndex = 0;
    while (groupIndex < this->coreGroupNum) {
        ProcessSingleGroup(this->groupOffset + groupIndex);
        ++groupIndex;
    }
}

template <typename TYPE_X>
__aicore__ inline void KernelGroupNormalizationGrad<TYPE_X>::CopyIn(uint32_t groupIdx, uint32_t segmentIdx,
                                                                    uint32_t currentDataNum)
{
    uint64_t baseOffset = groupIdx * this->groupElemNum + segmentIdx * this->tileDataNum;
    AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.AllocTensor<TYPE_X>();
    AscendC::LocalTensor<TYPE_X> dyLocal = inQueueDY.AllocTensor<TYPE_X>();
    AscendC::LocalTensor<TYPE_X> gammaLocal = inQueueGamma.AllocTensor<TYPE_X>();
    AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentDataNum * sizeof(TYPE_X)), 0, 0, 0};
    AscendC::DataCopyPadExtParams<TYPE_X> padParams{true, 0, 0, 0};
    AscendC::DataCopyPad(xLocal, xGm[baseOffset], copyParams, padParams);
    AscendC::DataCopyPad(dyLocal, dyGm[baseOffset], copyParams, padParams);
    AscendC::DataCopyPad(gammaLocal, gammaGm[baseOffset], copyParams, padParams);
    inQueueX.EnQue(xLocal);
    inQueueDY.EnQue(dyLocal);
    inQueueGamma.EnQue(gammaLocal);
}

template <typename TYPE_X>
__aicore__ inline void KernelGroupNormalizationGrad<TYPE_X>::CopyOut(uint32_t groupIdx, uint32_t segmentIdx,
                                                                     uint32_t currentDataNum)
{
    uint64_t baseOffset = groupIdx * this->groupElemNum + segmentIdx * this->tileDataNum;
    AscendC::LocalTensor<TYPE_X> dxLocal = outQueueDX.DeQue<TYPE_X>();
    AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentDataNum * sizeof(TYPE_X)), 0, 0, 0};
    AscendC::DataCopyPad(dxGm[baseOffset], dxLocal, copyParams);
    outQueueDX.FreeTensor(dxLocal);
}

template <typename TYPE_X>
__aicore__ inline float KernelGroupNormalizationGrad<TYPE_X>::ReadScalar(AscendC::GlobalTensor<TYPE_X>& tensor,
                                                                         uint32_t index)
{
    if constexpr (AscendC::Std::is_same<TYPE_X, float>::value) {
        return tensor.GetValue(index);
    } else if constexpr (AscendC::Std::is_same<TYPE_X, bfloat16_t>::value) {
        AscendC::LocalTensor<TYPE_X> tempSrc = scalarBuf.Get<TYPE_X>();
        AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(sizeof(TYPE_X)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<TYPE_X> padParams{true, 0, 0, 0};
        AscendC::DataCopyPad(tempSrc, tensor[index], copyParams, padParams);
        AscendC::PipeBarrier<PIPE_ALL>();
        AscendC::LocalTensor<float> tempDst = scalarBuf.Get<float>();
        AscendC::Cast(tempDst, tempSrc, AscendC::RoundMode::CAST_NONE, 1);
        return tempDst.GetValue(0);
    } else {
        return static_cast<float>(tensor.GetValue(index));
    }
}

template <typename TYPE_X>
__aicore__ inline void KernelGroupNormalizationGrad<TYPE_X>::CastToFloat(AscendC::LocalTensor<float> dst,
                                                                         AscendC::LocalTensor<TYPE_X> src,
                                                                         uint32_t dataNum)
{
    if constexpr (AscendC::Std::is_same<TYPE_X, float>::value) {
        AscendC::Adds(dst, src, 0.0f, dataNum);
    } else {
        AscendC::Cast(dst, src, AscendC::RoundMode::CAST_NONE, dataNum);
    }
}

template <typename TYPE_X>
__aicore__ inline void KernelGroupNormalizationGrad<TYPE_X>::CastFromFloat(AscendC::LocalTensor<TYPE_X> dst,
                                                                           AscendC::LocalTensor<float> src,
                                                                           uint32_t dataNum)
{
    if constexpr (AscendC::Std::is_same<TYPE_X, float>::value) {
        AscendC::Adds(dst, src, 0.0f, dataNum);
    } else if constexpr (AscendC::Std::is_same<TYPE_X, half>::value) {
        AscendC::Cast(dst, src, AscendC::RoundMode::CAST_ROUND, dataNum);
    } else if constexpr (AscendC::Std::is_same<TYPE_X, bfloat16_t>::value) {
        AscendC::Cast(dst, src, AscendC::RoundMode::CAST_ROUND, dataNum);
    } else {
        AscendC::Cast(dst, src, AscendC::RoundMode::CAST_RINT, dataNum);
    }
}

template <typename TYPE_X>
__aicore__ inline uint32_t KernelGroupNormalizationGrad<TYPE_X>::GetCurrentDataNum(uint32_t segmentIdx) const
{
    return (segmentIdx + 1 == this->groupTileNum) ? this->tailDataNum : this->tileDataNum;
}

template <typename TYPE_X>
__aicore__ inline void KernelGroupNormalizationGrad<TYPE_X>::ProcessSingleGroup(uint32_t groupIdx)
{
    float meanValue = ReadScalar(meanGm, groupIdx);
    float rstdValue = ReadScalar(rstdGm, groupIdx);
    float sumDyGamma = 0.0f;
    float sumDyGammaXhat = 0.0f;

    // Pass 1: compute reduction sums
    uint32_t segmentIdx = 0;
    while (segmentIdx < this->groupTileNum) {
        uint32_t currentDataNum = GetCurrentDataNum(segmentIdx);
        CopyIn(groupIdx, segmentIdx, currentDataNum);

        AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.DeQue<TYPE_X>();
        AscendC::LocalTensor<TYPE_X> dyLocal = inQueueDY.DeQue<TYPE_X>();
        AscendC::LocalTensor<TYPE_X> gammaLocal = inQueueGamma.DeQue<TYPE_X>();

        AscendC::LocalTensor<float> xFloat = xFloatBuf.Get<float>();
        AscendC::LocalTensor<float> dyFloat = dyFloatBuf.Get<float>();
        AscendC::LocalTensor<float> gammaFloat = gammaFloatBuf.Get<float>();
        AscendC::LocalTensor<float> xhat = xhatBuf.Get<float>();
        AscendC::LocalTensor<float> tmp0 = tmp0Buf.Get<float>();
        AscendC::LocalTensor<float> tmp1 = tmp1Buf.Get<float>();
        AscendC::LocalTensor<float> reduceTensor = reduceBuf.Get<float>();
        AscendC::LocalTensor<float> scalarTensor = scalarBuf.Get<float>();

        CastToFloat(xFloat, xLocal, currentDataNum);
        CastToFloat(dyFloat, dyLocal, currentDataNum);
        CastToFloat(gammaFloat, gammaLocal, currentDataNum);

        // x_hat = (x - mean) * rstd
        AscendC::Adds(xhat, xFloat, -meanValue, currentDataNum);
        AscendC::Muls(xhat, xhat, rstdValue, currentDataNum);

        // s1 = sum(dy * gamma)
        AscendC::Mul(tmp0, dyFloat, gammaFloat, currentDataNum);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ReduceSum(scalarTensor, tmp0, reduceTensor, currentDataNum);
        AscendC::PipeBarrier<PIPE_V>();
        sumDyGamma += scalarTensor.GetValue(0);

        // s2 = sum(dy * gamma * x_hat)
        AscendC::Mul(tmp1, tmp0, xhat, currentDataNum);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::ReduceSum(scalarTensor, tmp1, reduceTensor, currentDataNum);
        AscendC::PipeBarrier<PIPE_V>();
        sumDyGammaXhat += scalarTensor.GetValue(0);

        inQueueX.FreeTensor(xLocal);
        inQueueDY.FreeTensor(dyLocal);
        inQueueGamma.FreeTensor(gammaLocal);
        ++segmentIdx;
    }

    AscendC::PipeBarrier<PIPE_V>();

    float coeff = rstdValue * this->groupElemNumReciprocal;

    // Pass 2: compute dx
    segmentIdx = 0;
    while (segmentIdx < this->groupTileNum) {
        uint32_t currentDataNum = GetCurrentDataNum(segmentIdx);
        CopyIn(groupIdx, segmentIdx, currentDataNum);

        AscendC::LocalTensor<TYPE_X> xLocal = inQueueX.DeQue<TYPE_X>();
        AscendC::LocalTensor<TYPE_X> dyLocal = inQueueDY.DeQue<TYPE_X>();
        AscendC::LocalTensor<TYPE_X> gammaLocal = inQueueGamma.DeQue<TYPE_X>();
        AscendC::LocalTensor<TYPE_X> dxLocal = outQueueDX.AllocTensor<TYPE_X>();

        AscendC::LocalTensor<float> xFloat = xFloatBuf.Get<float>();
        AscendC::LocalTensor<float> dyFloat = dyFloatBuf.Get<float>();
        AscendC::LocalTensor<float> gammaFloat = gammaFloatBuf.Get<float>();
        AscendC::LocalTensor<float> xhat = xhatBuf.Get<float>();
        AscendC::LocalTensor<float> tmp0 = tmp0Buf.Get<float>();
        AscendC::LocalTensor<float> tmp1 = tmp1Buf.Get<float>();
        AscendC::LocalTensor<float> tmp2 = tmp2Buf.Get<float>();
        AscendC::LocalTensor<float> tmp3 = tmp3Buf.Get<float>();

        CastToFloat(xFloat, xLocal, currentDataNum);
        CastToFloat(dyFloat, dyLocal, currentDataNum);
        CastToFloat(gammaFloat, gammaLocal, currentDataNum);

        // x_hat = (x - mean) * rstd
        AscendC::Adds(xhat, xFloat, -meanValue, currentDataNum);
        AscendC::Muls(xhat, xhat, rstdValue, currentDataNum);

        // dx = (rstd / M) * gamma * (M * dy - s1 - x_hat * s2)
        AscendC::Muls(tmp0, dyFloat, this->groupElemNumFloat, currentDataNum);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Adds(tmp0, tmp0, -sumDyGamma, currentDataNum);
        AscendC::Muls(tmp1, xhat, sumDyGammaXhat, currentDataNum);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Sub(tmp2, tmp0, tmp1, currentDataNum);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Mul(tmp3, gammaFloat, tmp2, currentDataNum);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::Muls(tmp3, tmp3, coeff, currentDataNum);

        CastFromFloat(dxLocal, tmp3, currentDataNum);
        outQueueDX.EnQue(dxLocal);
        CopyOut(groupIdx, segmentIdx, currentDataNum);

        inQueueX.FreeTensor(xLocal);
        inQueueDY.FreeTensor(dyLocal);
        inQueueGamma.FreeTensor(gammaLocal);
        ++segmentIdx;
    }
}

} // namespace NsGroupNormalizationGrad
#endif // GROUP_NORMALIZATION_GRAD_H
