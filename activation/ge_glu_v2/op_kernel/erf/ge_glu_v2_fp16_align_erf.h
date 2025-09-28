/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file ge_glu_v2_fp16_align_erf.h
 * \brief
 */

#ifndef GeGluV2_HALF_ALIGN_ERF_H
#define GeGluV2_HALF_ALIGN_ERF_H

#include "../ge_glu_v2_base.h"

namespace GeGluV2 {
using namespace AscendC;

template <typename T>
class GeGluV2Fp16AlignErf : public GeGluV2Base<T>
{
public:
    __aicore__ inline GeGluV2Fp16AlignErf(){};
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, GM_ADDR gelu, GM_ADDR workspace, const GeGluV2TilingData* tilingData);
    __aicore__ inline void Process();

    constexpr static int32_t bufferNum = 2;
    constexpr static int64_t bufferSize = 6144;

private:
    __aicore__ inline void CopyInX(const int64_t& index, const int64_t& blockCount);
    __aicore__ inline void ComputeGelu(const int64_t& ub_num);
    __aicore__ inline void CopyOutGelu(const int64_t& index, const int64_t& ub_num, const int64_t& group);
    __aicore__ inline void ComputeMul(const int64_t& ub_num);
    __aicore__ inline void CopyOutMul(const int64_t& index, const int64_t& ub_num, const int64_t& group);
    __aicore__ inline void ProcessPerCore();
    __aicore__ inline void ProcessLastCore();

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, bufferNum> inQueueX1;
    TQue<QuePosition::VECIN, bufferNum> inQueueX2;
    TQue<QuePosition::VECOUT, bufferNum> outQueueGelu;
    TQue<QuePosition::VECOUT, bufferNum> outQueueMul;

    TBuf<QuePosition::VECCALC> resultTempBuf1;
    TBuf<QuePosition::VECCALC> resultTempBuf2;
    TBuf<QuePosition::VECCALC> resultTempBuf3;
    TBuf<QuePosition::VECCALC> resultTempBuf4;
};

template <typename T>
__aicore__ inline void GeGluV2Fp16AlignErf<T>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR gelu, GM_ADDR workspace, const GeGluV2TilingData* tilingData)
{
    this->BaseInit(x, y, gelu, tilingData);
    pipe.InitBuffer(inQueueX1, bufferNum, bufferSize * sizeof(T));
    pipe.InitBuffer(inQueueX2, bufferNum, bufferSize * sizeof(T));
    pipe.InitBuffer(outQueueGelu, 1, bufferSize * sizeof(T));
    pipe.InitBuffer(outQueueMul, 1, bufferSize * sizeof(T));

    pipe.InitBuffer(resultTempBuf1, bufferSize * sizeof(float));
    pipe.InitBuffer(resultTempBuf2, bufferSize * sizeof(float));
    pipe.InitBuffer(resultTempBuf3, bufferSize * sizeof(float));
    pipe.InitBuffer(resultTempBuf4, bufferSize * sizeof(float));
}

template <typename T>
__aicore__ inline void GeGluV2Fp16AlignErf<T>::Process()
{
    if (this->blockIdx >= this->m_tilingData.realCoreNum) {
        return;
    }

    if (this->isLastCore) { // process last core
        ProcessLastCore();
    } else {
        ProcessPerCore();
    }
}

template <typename T>
__aicore__ inline void GeGluV2Fp16AlignErf<T>::ProcessPerCore()
{
    // process core
    for (int64_t idx = 0; idx < this->m_tilingData.loopNum; idx++) {
        CopyInX(idx, this->m_tilingData.group);
        ComputeGelu(this->group_ub_num);
        CopyOutGelu(idx, this->group_ub_num, this->m_tilingData.group);
        ComputeMul(this->group_ub_num);
        CopyOutMul(idx, this->group_ub_num, this->m_tilingData.group);
    }

    if (this->m_tilingData.nLastTailGroup > 0) {
        CopyInX(this->m_tilingData.loopNum, this->m_tilingData.nLastTailGroup);
        ComputeGelu(this->nlast_tail_ub_num);
        CopyOutGelu(this->m_tilingData.loopNum, this->nlast_tail_ub_num, this->m_tilingData.nLastTailGroup);
        ComputeMul(this->nlast_tail_ub_num);
        CopyOutMul(this->m_tilingData.loopNum, this->nlast_tail_ub_num, this->m_tilingData.nLastTailGroup);
    }
}

template <typename T>
__aicore__ inline void GeGluV2Fp16AlignErf<T>::ProcessLastCore()
{
    for (int64_t idx = 0; idx < this->m_tilingData.tailLoopNum; idx++) {
        CopyInX(idx, this->m_tilingData.group);
        ComputeGelu(this->group_ub_num);
        CopyOutGelu(idx, this->group_ub_num, this->m_tilingData.group);
        ComputeMul(this->group_ub_num);
        CopyOutMul(idx, this->group_ub_num, this->m_tilingData.group);
    }
    if (this->m_tilingData.lastTailGroup > 0) {
        CopyInX(this->m_tilingData.tailLoopNum, this->m_tilingData.lastTailGroup);
        ComputeGelu(this->last_tail_ub_num);
        CopyOutGelu(this->m_tilingData.tailLoopNum, this->last_tail_ub_num, this->m_tilingData.lastTailGroup);
        ComputeMul(this->last_tail_ub_num);
        CopyOutMul(this->m_tilingData.tailLoopNum, this->last_tail_ub_num, this->m_tilingData.lastTailGroup);
    }
}

template <typename T>
__aicore__ inline void GeGluV2Fp16AlignErf<T>::CopyInX(const int64_t& index, const int64_t& blockCount)
{
    LocalTensor<T> ubX1 = inQueueX1.AllocTensor<T>();
    LocalTensor<T> ubX2 = inQueueX2.AllocTensor<T>();

    this->CopyInXAlign(index, blockCount, ubX1, ubX2);

    inQueueX1.EnQue(ubX1);
    inQueueX2.EnQue(ubX2);
}

template <typename T>
__aicore__ inline void GeGluV2Fp16AlignErf<T>::ComputeGelu(const int64_t& ub_num)
{
    LocalTensor<T> ubX2 = inQueueX2.DeQue<T>();
    LocalTensor<float> ubx2_fp32 = resultTempBuf2.Get<float>();
    Cast(ubx2_fp32, ubX2, RoundMode::CAST_NONE, ub_num);
    inQueueX2.FreeTensor(ubX2);

    // after cast to fp32 , input buffer release, to use as tmp buffer wihle do geluv2 compute.
    LocalTensor<float> tmpBuf = resultTempBuf1.Get<float>();
    LocalTensor<float> tmpBuf1 = resultTempBuf3.Get<float>();
    LocalTensor<float> tmpBuf2 = resultTempBuf4.Get<float>();
    this->ComputeGeluErf(ubx2_fp32, tmpBuf, tmpBuf1, tmpBuf2, ub_num);

    LocalTensor<T> gelu_out = outQueueGelu.AllocTensor<T>();
    Cast(gelu_out, tmpBuf, RoundMode::CAST_RINT, ub_num);
    outQueueGelu.EnQue(gelu_out);
}

template <typename T>
__aicore__ inline void GeGluV2Fp16AlignErf<T>::ComputeMul(const int64_t& ub_num)
{
    LocalTensor<T> ubX1 = inQueueX1.DeQue<T>();
    LocalTensor<T> gelu_out = outQueueGelu.DeQue<T>();
    LocalTensor<T> mul_out = outQueueMul.AllocTensor<T>();
    PipeBarrier<PIPE_V>();
    Mul(mul_out, gelu_out, ubX1, ub_num);
    outQueueMul.EnQue(mul_out);

    outQueueGelu.FreeTensor(gelu_out);
    inQueueX1.FreeTensor(ubX1);
}

template <typename T>
__aicore__ inline void GeGluV2Fp16AlignErf<T>::CopyOutGelu(
    const int64_t& index, const int64_t& ub_num, const int64_t& group)
{
    LocalTensor<T> outLocalGelu = outQueueGelu.DeQue<T>();
    this->CopyOutGeluBase(index, ub_num, group, outLocalGelu);
    outQueueGelu.EnQue(outLocalGelu);
}

template <typename T>
__aicore__ inline void GeGluV2Fp16AlignErf<T>::CopyOutMul(
    const int64_t& index, const int64_t& ub_num, const int64_t& group)
{
    LocalTensor<T> outLocalMul = outQueueMul.DeQue<T>();
    this->CopyOutMulBase(index, ub_num, group, outLocalMul);
    outQueueMul.FreeTensor(outLocalMul);
}
} // namespace GeGluV2
#endif // GeGluV2_HALF_ALIGN_ERF_H