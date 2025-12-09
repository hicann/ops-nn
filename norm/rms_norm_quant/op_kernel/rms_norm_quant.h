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
 * \file kernel_common.h
 * \brief
 */

#ifndef RMS_NORM_QUANT_KERNEL_COMMON_H
#define RMS_NORM_QUANT_KERNEL_COMMON_H
#include "kernel_operator.h"
#include "../rms_norm/rms_norm_base.h"

using AscendC::HardEvent;
using namespace AscendC;
using namespace RmsNorm;

template <typename T, typename Q>
__aicore__ inline void CopyIn(const AscendC::GlobalTensor<T>& gm, Q& queue, uint64_t offset, uint32_t count)
{
    AscendC::LocalTensor<T> local = queue.template AllocTensor<T>();
    DataCopy(local, gm[offset], count);
    queue.EnQue(local);
}

template <typename T>
__aicore__ inline void CopyInXResIn(
    AscendC::LocalTensor<T>& fp16_x, AscendC::LocalTensor<T>& fp16_res_in, AscendC::GlobalTensor<T>& gm_x_,
    AscendC::GlobalTensor<T>& gm_res_in_, uint64_t& gmOffset, uint32_t& numel)
{
    DataCopy(fp16_x, gm_x_[gmOffset], numel);
    DataCopy(fp16_res_in, gm_res_in_[gmOffset], numel); // res_in
}

template <typename T>
__aicore__ inline void CopyInG(
    AscendC::LocalTensor<T>& fp16_gamma, AscendC::GlobalTensor<T>& gm_g_, uint32_t offset, uint32_t& numelAlignFp16)
{
    DataCopy(fp16_gamma, gm_g_[offset], numelAlignFp16);
}

template <typename T>
__aicore__ inline void BiasIn(
    AscendC::LocalTensor<T>& fp16_x, AscendC::LocalTensor<T>& fp16_bias, AscendC::LocalTensor<float>& fp32_bias,
    AscendC::GlobalTensor<T>& gm_bias, uint32_t countAlign, uint32_t gmOffset = 0)
{
    // do copyIn and Cast
    DataCopy(fp16_bias, gm_bias[gmOffset], countAlign);
    AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID1);
    AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID1);
    Cast(fp32_bias, fp16_bias, AscendC::RoundMode::CAST_NONE, countAlign);
    AscendC::SetFlag<HardEvent::V_MTE2>(EVENT_ID1);
    AscendC::WaitFlag<HardEvent::V_MTE2>(EVENT_ID1);
}

template <typename T, typename Q>
__aicore__ inline void CopyOut(const AscendC::GlobalTensor<T>& gm, Q& queue, uint64_t offset, uint32_t count)
{
    AscendC::LocalTensor<T> local = queue.template DeQue<T>();
    DataCopy(gm[offset], local, count);
    queue.FreeTensor(local);
}

template <typename T>
__aicore__ inline void CastFrom16To32(
    const AscendC::LocalTensor<float>& out, const AscendC::LocalTensor<T>& in, uint32_t count)
{
    Cast(out, in, AscendC::RoundMode::CAST_NONE, count);
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void CastFrom32To16(
    const AscendC::LocalTensor<T>& out, const AscendC::LocalTensor<float>& in, uint32_t count)
{
    if constexpr (AscendC::IsSameType<T, half>::value) {
        Cast(out, in, AscendC::RoundMode::CAST_NONE, count);
    } else { // bf16
        Cast(out, in, AscendC::RoundMode::CAST_RINT, count);
    }
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline void CastFromF16ToI8(
    const AscendC::LocalTensor<int8_t>& out, const AscendC::LocalTensor<half>& in, half quantMin, uint32_t count)
{
    Maxs(in, in, quantMin, count);
    AscendC::PipeBarrier<PIPE_V>();
    Mins(in, in, (half)127, count); // 127: limit
    AscendC::PipeBarrier<PIPE_V>();
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    Cast(out, in, AscendC::RoundMode::CAST_RINT, count);
#else
    Cast(out, in, AscendC::RoundMode::CAST_NONE, count);
#endif
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename T, typename Q>
__aicore__ inline void CopyInAndCastF32(
    const AscendC::LocalTensor<float>& out, const AscendC::GlobalTensor<T>& gm, Q& queue, uint64_t offset,
    uint32_t count)
{
    CopyIn(gm, queue, offset, count);
    AscendC::LocalTensor<T> local = queue.template DeQue<T>();
    Cast(out, local, AscendC::RoundMode::CAST_NONE, count);
    queue.FreeTensor(local);
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename T, typename Q>
__aicore__ inline void Cast16AndCopyOut(
    const AscendC::LocalTensor<float>& in, const AscendC::GlobalTensor<T>& gm, Q& queue, uint64_t offset,
    uint32_t count)
{
    AscendC::LocalTensor<T> local = queue.template AllocTensor<T>();
    CastFrom32To16(local, in, count);
    queue.EnQue(local);
    CopyOut(gm, queue, offset, count);
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline T ComputeSum(
    const AscendC::LocalTensor<T>& in, const AscendC::LocalTensor<T>& tmp, const AscendC::LocalTensor<T>& workLocal,
    uint32_t count)
{
#if __CCE_AICORE__ == 100
    float sum = 0;
    int64_t elementNumPerRep = AscendC::ONE_REPEAT_BYTE_SIZE / sizeof(T);
    AscendC::LocalTensor<T> src = in;
    while (count > elementNumPerRep) {
        int64_t repeatTimes = count / elementNumPerRep;
        int64_t tailCount = count % elementNumPerRep;
        int64_t bodyCount = repeatTimes * elementNumPerRep;
        if (repeatTimes > 0) {
            AscendC::AscendCUtils::SetMask<T>(elementNumPerRep);
            WholeReduceSum<T, false>(tmp, src, MASK_PLACEHOLDER, repeatTimes, 1, 1, 8);
            AscendC::SetFlag<HardEvent::V_S>(EVENT_ID0); // PipeBarrier(PIPE_V)?
            AscendC::WaitFlag<HardEvent::V_S>(EVENT_ID0);
        }

        if (tailCount != 0) {
            AscendC::AscendCUtils::SetMask<T>(tailCount);
            WholeReduceSum<T, false>(tmp[bodyCount], src[bodyCount], MASK_PLACEHOLDER, 1, 1, 1, 8);
            AscendC::SetFlag<HardEvent::V_S>(EVENT_ID0);
            AscendC::WaitFlag<HardEvent::V_S>(EVENT_ID0);
            sum += tmp.GetValue(bodyCount);
        }

        count = repeatTimes;
        src = tmp;
    }

    if (count > 1) {
        AscendC::AscendCUtils::SetMask<T>(count);
        WholeReduceSum<T, false>(tmp, tmp, MASK_PLACEHOLDER, 1, 1, 1, 8);
        AscendC::SetFlag<HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::V_S>(EVENT_ID0);
    }

    sum += tmp.GetValue(0);
    return sum;
#else
    ReduceSum(tmp, in, workLocal, count);
    AscendC::SetFlag<HardEvent::V_S>(EVENT_ID0);
    AscendC::WaitFlag<HardEvent::V_S>(EVENT_ID0);
    return tmp.GetValue(0);
#endif
}

__aicore__ inline float ComputeSliceSquareSum(
    const AscendC::LocalTensor<float>& in, const AscendC::LocalTensor<float>& tmp,
    const AscendC::LocalTensor<float>& workLocal, uint32_t count)
{
    Mul(tmp, in, in, count);
    AscendC::PipeBarrier<PIPE_V>();
    return ComputeSum(tmp, tmp, workLocal, count);
}
template <typename T>
__aicore__ inline void ComputeRmsNorm(
    const AscendC::LocalTensor<T>& out, const AscendC::LocalTensor<float>& in, float rms,
    const AscendC::LocalTensor<T>& gamma, uint32_t count, uint32_t precisionMode, uint32_t gemmaMode,
    const AscendC::LocalTensor<float>& tmp)
{
    float value = 1.0;
    Duplicate(tmp, rms, count);
    AscendC::PipeBarrier<PIPE_V>();
    Div(tmp, in, tmp, count);
    AscendC::PipeBarrier<PIPE_V>();

    if (precisionMode == 0) {
        CastFrom16To32(in, gamma, count);
        AscendC::PipeBarrier<PIPE_V>();
        if (gemmaMode == 1) {
            Adds(in, in, value, count);
            AscendC::PipeBarrier<PIPE_V>();
        }
        Mul(in, in, tmp, count);
        AscendC::PipeBarrier<PIPE_V>();
        CastFrom32To16(out, in, count);
        return;
    }
    if constexpr (std::is_same<T, half>::value) {
        CastFrom32To16(out, tmp, count);
        Mul(out, out, gamma, count);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

template <typename T, uint32_t gemmaMode>
__aicore__ inline void CastGAndIsGemmaMode(
    const AscendC::LocalTensor<float>& out, const AscendC::LocalTensor<T>& gamma, uint32_t count)
{
    Cast(out, gamma, AscendC::RoundMode::CAST_NONE, count);
    AscendC::PipeBarrier<PIPE_V>();
    float value = 1.0;
    if constexpr (gemmaMode == 1) {
        Adds(out, out, value, count);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

template <typename T, uint32_t precisionMode>
__aicore__ inline void ComputeRmsNormFast(
    const AscendC::LocalTensor<T>& out, const AscendC::LocalTensor<float>& in, float rms,
    const AscendC::LocalTensor<T>& gamma, uint32_t count, const AscendC::LocalTensor<float>& tmp,
    const AscendC::LocalTensor<float>& fp32_g)
{
    float value = 1.0;
    Duplicate(tmp, rms, count);
    AscendC::PipeBarrier<PIPE_V>();
    Div(tmp, in, tmp, count);
    AscendC::PipeBarrier<PIPE_V>();
    if constexpr (precisionMode == 0) {
        Mul(in, fp32_g, tmp, count);
        AscendC::PipeBarrier<PIPE_V>();
        CastFrom32To16(out, in, count);
        return;
    }
    if constexpr (std::is_same<T, half>::value) {
        CastFrom32To16(out, tmp, count);
        Mul(out, out, gamma, count);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

template <bool WITH_BETA = true>
__aicore__ inline void ComputeRmsNorm(
    const AscendC::LocalTensor<float>& out, const AscendC::LocalTensor<float>& in, float rms,
    const AscendC::LocalTensor<half>& gamma, const AscendC::LocalTensor<half>& beta,
    const AscendC::LocalTensor<float>& tmp, uint32_t count)
{
    Duplicate(tmp, rms, count);
    AscendC::PipeBarrier<PIPE_V>();
    Div(out, in, tmp, count);
    AscendC::PipeBarrier<PIPE_V>();
    CastFrom16To32(tmp, gamma, count);
    Mul(out, out, tmp, count);
    AscendC::PipeBarrier<PIPE_V>();
    if constexpr (WITH_BETA) {
        CastFrom16To32(tmp, beta, count);
        Add(out, out, tmp, count);
        AscendC::PipeBarrier<PIPE_V>();
    }
}

template <typename T>
__aicore__ inline void ComputeRmsNorm(
    const AscendC::LocalTensor<float>& out, const AscendC::LocalTensor<float>& in, float reciprocal_of_rms,
    const AscendC::LocalTensor<T>& gamma, const AscendC::LocalTensor<float>& tmp,
    const AscendC::LocalTensor<T>& res_out, uint32_t count)
{
    Duplicate(tmp, reciprocal_of_rms, count);
    AscendC::PipeBarrier<PIPE_V>();
    Mul(out, in, tmp, count);
    AscendC::PipeBarrier<PIPE_V>();
    CastFrom16To32(tmp, gamma, count);
    Mul(out, out, tmp, count);
    AscendC::PipeBarrier<PIPE_V>();
    CastFrom32To16(res_out, out, count);
}

template <typename T>
__aicore__ inline void ComputeResidualAdd(
    const AscendC::LocalTensor<T>& out, const AscendC::LocalTensor<T>& in, const AscendC::LocalTensor<T>& resIn,
    uint32_t count)
{
    Add(out, in, resIn, count);
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline void ComputeFp16ToI8Quant(
    const AscendC::LocalTensor<int8_t>& out, const AscendC::LocalTensor<half>& in,
    const AscendC::LocalTensor<half>& tmp, half scale, half offset, half quantMin, uint32_t count)
{
    Muls(tmp, in, scale, count);
    AscendC::PipeBarrier<PIPE_V>();
    Adds(tmp, tmp, offset, count);
    AscendC::PipeBarrier<PIPE_V>();
    CastFromF16ToI8(out, tmp, quantMin, count);
}

__aicore__ inline void ComputeFp32ToI8Quant(
    const AscendC::LocalTensor<int8_t>& out, const AscendC::LocalTensor<float>& in,
    const AscendC::LocalTensor<half>& tmp, half scale, half offset, half quantMin, uint32_t count)
{
    CastFrom32To16(tmp, in, count);
    AscendC::PipeBarrier<PIPE_V>();
    ComputeFp16ToI8Quant(out, tmp, tmp, scale, offset, quantMin, count);
}

template <typename T>
__aicore__ inline void MultiplyGamma(
    AscendC::LocalTensor<T>& fp16_gamma, AscendC::LocalTensor<float>& sqx, AscendC::LocalTensor<float>& fp32_xy,
    AscendC::LocalTensor<T>& out_buf, uint32_t numCol_, uint32_t& numColAlignFp32, uint32_t& numColAlignFp16,
    uint32_t& precisionMode_)
{
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    if constexpr (std::is_same<T, bfloat16_t>::value) {
        Cast(sqx, fp16_gamma, AscendC::RoundMode::CAST_NONE, numColAlignFp16); // g
        AscendC::PipeBarrier<PIPE_V>();
        Mul(fp32_xy, fp32_xy, sqx, numColAlignFp32);
        AscendC::PipeBarrier<PIPE_V>();
        CastFrom32To16(out_buf, fp32_xy, numCol_);
    }
#endif
    if constexpr (std::is_same<T, half>::value) {
        if (precisionMode_ == 1) {
            AscendC::PipeBarrier<PIPE_V>();
            CastFrom32To16(out_buf, fp32_xy, numCol_);
            AscendC::PipeBarrier<PIPE_V>();
            Mul(out_buf, out_buf, fp16_gamma, numColAlignFp16);
        } else {
            AscendC::PipeBarrier<PIPE_V>();
            Cast(sqx, fp16_gamma, AscendC::RoundMode::CAST_NONE, numColAlignFp16); // g
            AscendC::PipeBarrier<PIPE_V>();
            Mul(fp32_xy, sqx, fp32_xy, numColAlignFp32);
            AscendC::PipeBarrier<PIPE_V>();
            CastFrom32To16(out_buf, fp32_xy, numCol_);
        }
    }
}

template <typename T>
__aicore__ inline void AddResAndCast(
    AscendC::LocalTensor<T>& fp16_x, AscendC::LocalTensor<T>& fp16_res_in, AscendC::LocalTensor<float>& fp32_xy,
    AscendC::LocalTensor<float>& buf, uint32_t count)
{
    Cast(fp32_xy, fp16_x, AscendC::RoundMode::CAST_NONE, count);
    Cast(buf, fp16_res_in, AscendC::RoundMode::CAST_NONE, count); // res_in
    AscendC::PipeBarrier<PIPE_V>();
    Add(fp32_xy, fp32_xy, buf, count);
    AscendC::PipeBarrier<PIPE_V>();
}

template <typename T>
__aicore__ inline void AddResBiasAndCast(
    AscendC::LocalTensor<T>& fp16_x, AscendC::LocalTensor<T>& fp16_res_in, AscendC::LocalTensor<T>& fp16_bias,
    AscendC::LocalTensor<float>& fp32_xy, AscendC::LocalTensor<float>& buf, AscendC::LocalTensor<float>& buf_bias,
    uint32_t count)
{
    Cast(fp32_xy, fp16_x, AscendC::RoundMode::CAST_NONE, count);
    Cast(buf, fp16_res_in, AscendC::RoundMode::CAST_NONE, count); // res_in
    AscendC::PipeBarrier<PIPE_V>();
    Add(fp32_xy, fp32_xy, buf, count);
    AscendC::PipeBarrier<PIPE_V>();
    Add(fp32_xy, fp32_xy, buf_bias, count);
    AscendC::PipeBarrier<PIPE_V>();
}

__aicore__ inline void FigureOutNorm(
    AscendC::LocalTensor<float>& sqx, AscendC::LocalTensor<float>& fp32_xy,
    AscendC::LocalTensor<float>& fp32_reduce_workspace, float& avgFactor_, float& epsilon_, uint32_t& count,
    uint32_t& countAlign)
{
    // x / sqrt(sum(x^2) * 1 / len(sum) + eps) --> norm operation
    Mul(sqx, fp32_xy, fp32_xy, countAlign);
    AscendC::PipeBarrier<PIPE_V>();
    ReduceSumCustom(sqx, sqx, fp32_reduce_workspace, count);

    Muls(sqx, sqx, avgFactor_, 1);
    AscendC::PipeBarrier<PIPE_V>();
    Adds(sqx, sqx, epsilon_, 1);
    AscendC::PipeBarrier<PIPE_V>();

    Sqrt(sqx, sqx, 1);
    AscendC::SetFlag<HardEvent::V_S>(EVENT_ID2);

    AscendC::WaitFlag<HardEvent::V_S>(EVENT_ID2);
    float factor = 1 / sqx.GetValue(0);
    Muls(fp32_xy, fp32_xy, factor, countAlign);
    AscendC::PipeBarrier<PIPE_V>();
}
#endif

using AscendC::HardEvent;

template <typename S, typename O>
__aicore__ inline void GetQuantInfo(
    S& varScale, O& varOffset, __gm__ uint8_t* scale, __gm__ uint8_t* offset,
    AscendC::TBuf<AscendC::TPosition::VECCALC>& buf)
{
    AscendC::GlobalTensor<half> gm_s;
    AscendC::GlobalTensor<int8_t> gm_o;
    gm_s.SetGlobalBuffer((__gm__ half*)scale);
    gm_o.SetGlobalBuffer((__gm__ int8_t*)offset);

    AscendC::LocalTensor<half> scale_buffer = buf.Get<half>();
    DataCopy(scale_buffer, gm_s, BLOCK_SIZE / sizeof(half));
    AscendC::SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    AscendC::WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    varScale = 1 / static_cast<S>(scale_buffer.GetValue(0));

    AscendC::LocalTensor<int8_t> offset_buffer = buf.Get<int8_t>();
    AscendC::SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
    DataCopy(offset_buffer, gm_o, BLOCK_SIZE / sizeof(int8_t));
    AscendC::SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    AscendC::WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    varOffset = static_cast<O>(offset_buffer.GetValue(0));
    AscendC::SetFlag<HardEvent::S_MTE2>(EVENT_ID0);
    AscendC::WaitFlag<HardEvent::S_MTE2>(EVENT_ID0);
}

template <typename T>
__aicore__ inline void GetScaleAndOffset(
    float& varScale, float& varOffset, GM_ADDR scale, GM_ADDR offset, AscendC::TBuf<AscendC::TPosition::VECCALC>& buf)
{
    AscendC::GlobalTensor<T> gm_s;
    gm_s.SetGlobalBuffer((__gm__ T*)scale);
    AscendC::LocalTensor<T> scale_buffer = buf.Get<T>();
    DataCopy(scale_buffer, gm_s, BLOCK_SIZE / sizeof(T));
    if constexpr (AscendC::IsSameType<T, half>::value) {
        AscendC::SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        varScale = 1 / (float)(scale_buffer.GetValue(0));
    } else {
        AscendC::LocalTensor<float> tmpFp32 = buf.Get<float>();
        AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Cast(tmpFp32, scale_buffer, AscendC::RoundMode::CAST_NONE, 1);
        AscendC::SetFlag<HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::V_S>(EVENT_ID0);
        varScale = 1 / (float)(tmpFp32.GetValue(0));
    }
    AscendC::GlobalTensor<int8_t> gm_o;
    gm_o.SetGlobalBuffer((__gm__ int8_t*)offset);
    AscendC::LocalTensor<int8_t> tmpInt8 = buf.Get<int8_t>();
    DataCopy(tmpInt8, gm_o, BLOCK_SIZE / sizeof(int8_t));
    AscendC::SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
    AscendC::WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
    varOffset = (float)(tmpInt8.GetValue(0));
}