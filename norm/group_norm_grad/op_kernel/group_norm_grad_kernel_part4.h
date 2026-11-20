/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* Shared helpers and second-stage workspace reduction.
 * Include only from group_norm_grad_kernel.h. */

__aicore__ inline void Get_Mean_Rstd(uint64_t task_idx)
{
    if constexpr (IsSameType<T, bfloat16_t>::value) {
        this->mean_scalar = ToFloat(mean_gm.GetValue(task_idx));
        this->rstd_scalar = ToFloat(rstd_gm.GetValue(task_idx));
    } else {
        this->mean_scalar = float(mean_gm.GetValue(task_idx));
        this->rstd_scalar = float(rstd_gm.GetValue(task_idx));
    }
    PipeBarrier<PIPE_ALL>();
}

__aicore__ inline void Custom_DataCopy_In(const LocalTensor<float>& _in, TBuf<TPosition::VECCALC>& tbuf,
                                          const GlobalTensor<T>& gm, const uint64_t offset, const uint32_t count)
{
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    if constexpr (IsSameType<T, float>::value) {
        DataCopyPad(_in, gm[offset], copyParams, padParams);
    } else {
        LocalTensor<T> temp = tbuf.Get<T>(count);
        DataCopyPad(temp, gm[offset], copyParams, padParams);
        PipeBarrier<PIPE_ALL>();
        Cast(_in, temp, RoundMode::CAST_NONE, count);
    }
}

__aicore__ inline void Custom_DataCopy_In(LocalTensor<float>& _in, const uint32_t ub_offset,
                                          TBuf<TPosition::VECCALC>& tbuf, GlobalTensor<T>& gm, const uint64_t gm_offset,
                                          const uint32_t count)
{
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    if constexpr (IsSameType<T, float>::value) {
        DataCopyPad(_in[ub_offset], gm[gm_offset], copyParams, padParams);
    } else {
        LocalTensor<T> temp = tbuf.Get<T>(count);
        DataCopyPad(temp, gm[gm_offset], copyParams, padParams);
        PipeBarrier<PIPE_ALL>();
        Cast(_in[ub_offset], temp, RoundMode::CAST_NONE, count);
    }
}

__aicore__ inline void Custom_DataCopy_Out(LocalTensor<float>& _out, const uint64_t gm_offset,
                                           TBuf<TPosition::VECCALC>& tbuf, const uint32_t count)
{
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
    if constexpr (IsSameType<T, float>::value) {
        DataCopyPad(dx_gm[gm_offset], _out, copyParams);
    } else {
        LocalTensor<T> temp = tbuf.Get<T>(count);
        Cast(temp, _out, RoundMode::CAST_ROUND, count);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyPad(dx_gm[gm_offset], temp, copyParams);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }
}

__aicore__ inline void Custom_DataCopy_Out(LocalTensor<float>& _out, const uint32_t ub_offset, const uint64_t gm_offset,
                                           TBuf<TPosition::VECCALC>& tbuf, const uint32_t count)
{
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
    if constexpr (IsSameType<T, float>::value) {
        DataCopyPad(dx_gm[gm_offset], _out[ub_offset], copyParams);
    } else {
        LocalTensor<T> temp = tbuf.Get<T>(count);
        Cast(temp, _out[ub_offset], RoundMode::CAST_ROUND, count);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyPad(dx_gm[gm_offset], temp, copyParams);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }
}

__aicore__ inline void Custom_DataCopy_Out(LocalTensor<float>& _out, GlobalTensor<T>& gm_out, const uint64_t gm_offset,
                                           TBuf<TPosition::VECCALC>& tbuf, const uint32_t count)
{
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
    if constexpr (IsSameType<T, float>::value) {
        DataCopyPad(gm_out[gm_offset], _out, copyParams);
    } else {
        LocalTensor<T> temp = tbuf.Get<T>(count);
        Cast(temp, _out, RoundMode::CAST_ROUND, count);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyPad(gm_out[gm_offset], temp, copyParams);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
    }
}

__aicore__ inline void fp32_dgamma_dbeta2GM(uint64_t channel_idx, GlobalTensor<float>& dgamma_out,
                                            const LocalTensor<float>& dgamma_ub, GlobalTensor<float>& dbeta_out,
                                            const LocalTensor<float>& dbeta_ub)
{
    event_t eventIDSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
    WaitFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
    SetAtomicAdd<float>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(this->C_G * sizeof(float)), 0, 0, 0};
    if (dbeta_is_require) {
        DataCopyPad(dbeta_out[channel_idx], dbeta_ub, copyParams);
        event_t eventIDMTE3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
        WaitFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
    }
    if (dgamma_is_require) {
        DataCopyPad(dgamma_out[channel_idx], dgamma_ub, copyParams);
        event_t eventIDMTE3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
        WaitFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
    }
    SetAtomicNone();
}

__aicore__ inline void non_fp32_dgamma_dbeta2GM(uint64_t channel_idx, GlobalTensor<T>& dgamma_out,
                                                const LocalTensor<float>& dgamma_ub, GlobalTensor<T>& dbeta_out,
                                                const LocalTensor<float>& dbeta_ub,
                                                TBuf<TPosition::VECCALC>& dgamma_tbuf,
                                                TBuf<TPosition::VECCALC>& dbeta_tbuf)
{
    event_t eventIDSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    SetFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
    WaitFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(this->C_G * sizeof(T)), 0, 0, 0};
    if (dbeta_is_require) {
        LocalTensor<T> dbeta_temp = dbeta_tbuf.Get<T>(this->C_G);
        Cast(dbeta_temp, dbeta_ub, RoundMode::CAST_ROUND, this->C_G);
        event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
        DataCopyPad(dbeta_out[channel_idx], dbeta_temp, copyParams);
    }
    if (dgamma_is_require) {
        LocalTensor<T> dgamma_temp = dgamma_tbuf.Get<T>(this->C_G);
        Cast(dgamma_temp, dgamma_ub, RoundMode::CAST_ROUND, this->C_G);
        event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
        DataCopyPad(dgamma_out[channel_idx], dgamma_temp, copyParams);
    }
}

__aicore__ inline void Mode_Selection(uint64_t task_idx, const LocalTensor<float>& dgamma_ub,
                                      const LocalTensor<float>& dbeta_ub)
{
    uint64_t outChannelIdx;
    constexpr uint32_t SPLIT_COUNT = 2;
    if constexpr (!isDeterministic && IsSameType<T, float>::value) {
        outChannelIdx = (task_idx % this->G) * this->C_G;
        fp32_dgamma_dbeta2GM(outChannelIdx, dgamma_gm, dgamma_ub, dbeta_gm, dbeta_ub);
    } else if constexpr (!isDeterministic && !IsSameType<T, float>::value) {
        outChannelIdx = (task_idx % this->G) * this->C_G;
        if (N == 1) {
            non_fp32_dgamma_dbeta2GM(outChannelIdx, dgamma_gm, dgamma_ub, dbeta_gm, dbeta_ub, out_tbuf_dgamma_T,
                                     out_tbuf_dbeta_T);
        } else {
            fp32_dgamma_dbeta2GM(outChannelIdx, dgamma_workspace, dgamma_ub, dbeta_workspace, dbeta_ub);
        }
    } else if constexpr (isDeterministic) {
        outChannelIdx = ((task_idx / this->G) / SPLIT_COUNT * this->G + (task_idx % this->G)) * this->C_G;
        fp32_dgamma_dbeta2GM(outChannelIdx, dgamma_workspace, dgamma_ub, dbeta_workspace, dbeta_ub);
    }
}

__aicore__ inline uint64_t divceil(uint64_t a, uint64_t b)
{
    if (b != 0) {
        return (a - 1) / b + 1;
    } else {
        return 0;
    }
}

__aicore__ inline uint64_t ceil(uint64_t a, uint64_t b)
{
    if (b != 0) {
        return ((a - 1) / b + 1) * b;
    } else {
        return 0;
    }
}

__aicore__ inline uint64_t floor(uint64_t a, uint64_t b)
{
    if (b != 0) {
        return (a / b * b);
    } else {
        return 0;
    }
}

__aicore__ inline void Adds_Muls_template(const LocalTensor<float>& dstLocal, const LocalTensor<float>& srcLocal,
                                          float adds_scalarValue, float muls_scalarValue, const int32_t calCount)
{
    Muls(dstLocal, srcLocal, muls_scalarValue, calCount);
    Adds(dstLocal, srcLocal, adds_scalarValue, calCount);
}

__aicore__ inline void Adds_Muls_template(const LocalTensor<float>& mul_dstLocal,
                                          const LocalTensor<float>& mul_srcLocal,
                                          const LocalTensor<float>& add_dstLocal,
                                          const LocalTensor<float>& add_srcLocal, float adds_scalarValue,
                                          float muls_scalarValue, const int32_t calCount)
{
    Muls(mul_dstLocal, mul_srcLocal, muls_scalarValue, calCount);
    Adds(add_dstLocal, add_srcLocal, adds_scalarValue, calCount);
}

__aicore__ inline float ReduceSum_template(const LocalTensor<float>& dstLocal, const LocalTensor<float>& srcLocal,
                                           const LocalTensor<float>& workLocal, uint32_t calCount, bool mode)
{
    float temp_value = 0;
    const int32_t MASK = 64;
    const int32_t MAX_LEN = 128;
    const uint32_t STRIDE_BLOCK = 1;
    const uint8_t PROCESS_BLOCK = 8;
    if (mode && calCount > MASK) {
        bool firstAdd = true;
        if (calCount > MASK) {
            for (int32_t start = MAX_LEN; start >= 1; start >>= 1) {
                const int32_t innerOffset = start * MASK;
                if (calCount > innerOffset && firstAdd) {
                    Copy(dstLocal, srcLocal, MASK, start, {STRIDE_BLOCK, STRIDE_BLOCK, PROCESS_BLOCK, PROCESS_BLOCK});
                    Add<float>(dstLocal, srcLocal[innerOffset], dstLocal, calCount - innerOffset);
                    calCount = innerOffset;
                    firstAdd = false;
                } else if (calCount > innerOffset) {
                    Add<float, false>(
                        dstLocal, dstLocal[innerOffset], dstLocal, MASK, start,
                        {STRIDE_BLOCK, STRIDE_BLOCK, STRIDE_BLOCK, PROCESS_BLOCK, PROCESS_BLOCK, PROCESS_BLOCK});
                    calCount = innerOffset;
                }
            }
            ReduceSum<float>(dstLocal, dstLocal, workLocal, MASK);
            temp_value = dstLocal.GetValue(0);
        }
    } else {
        ReduceSum<float>(dstLocal, srcLocal, workLocal, calCount);
        temp_value = dstLocal.GetValue(0);
    }
    return temp_value;
}

__aicore__ inline void Mul_ReduceSum_template(const LocalTensor<float>& dstLocal1, const LocalTensor<float>& dstLocal2,
                                              const LocalTensor<float>& srcLocal, const int32_t calCount, float& res1,
                                              float& res2)
{
    Mul(dstLocal1, srcLocal, dstLocal1, calCount);
    Mul(dstLocal2, srcLocal, dstLocal2, calCount);
    res1 = ReduceSum_template(dstLocal1, dstLocal1, dstLocal1, calCount, false) / this->ele_num_per_group;
    res2 = ReduceSum_template(dstLocal2, dstLocal2, dstLocal2, calCount, false) / this->ele_num_per_group;
}

__aicore__ inline void Initbuffer4stage2_mode0()
{
    pipe.InitBuffer(in_queue_dgamma_channel, 1, castEleNum * sizeof(float));
    pipe.InitBuffer(in_queue_dbeta_channel, 1, castEleNum * sizeof(float));
    pipe.InitBuffer(out_tbuf_dgamma_channel_T, castEleNum * sizeof(T));
    pipe.InitBuffer(out_tbuf_dbeta_channel_T, castEleNum * sizeof(T));
}

__aicore__ inline void Initbuffer4stage2_mode1()
{
    pipe.InitBuffer(cal_queue_dgamma_reduce, 1, castEleNum * sizeof(float));
    pipe.InitBuffer(cal_queue_dbeta_reduce, 1, castEleNum * sizeof(float));
    pipe.InitBuffer(in_queue_dgamma_channel, 1, coreBatchParts * castEleNum * sizeof(float));
    pipe.InitBuffer(in_queue_dbeta_channel, 1, coreBatchParts * castEleNum * sizeof(float));
}

__aicore__ inline void Initbuffer4stage2_mode2()
{
    pipe.InitBuffer(cal_queue_dgamma_reduce, 1, castEleNum * sizeof(float));
    pipe.InitBuffer(cal_queue_dbeta_reduce, 1, castEleNum * sizeof(float));
    pipe.InitBuffer(in_queue_dgamma_channel, 1, coreBatchParts * castEleNum * sizeof(float));
    pipe.InitBuffer(in_queue_dbeta_channel, 1, coreBatchParts * castEleNum * sizeof(float));
    pipe.InitBuffer(out_tbuf_dgamma_channel_T, castEleNum * sizeof(T));
    pipe.InitBuffer(out_tbuf_dbeta_channel_T, castEleNum * sizeof(T));
}

__aicore__ inline void cast_dgamma_dbeta_WSP2GM(uint64_t channel_idx, uint32_t count)
{
    DataCopyExtParams copyParams_fp16{1, static_cast<uint32_t>(count * sizeof(half)), 0, 0, 0};
    DataCopyExtParams copyParams_fp32{1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0};
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    if (dgamma_is_require) {
        LocalTensor<float> dgamma_Local = in_queue_dgamma_channel.AllocTensor<float>();
        DataCopyPad(dgamma_Local, dgamma_workspace[channel_idx], copyParams_fp32, padParams);
        LocalTensor<T> dgamma_temp = out_tbuf_dgamma_channel_T.Get<T>(count);
        event_t eventIDMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        Cast(dgamma_temp, dgamma_Local, RoundMode::CAST_ROUND, count);
        in_queue_dgamma_channel.FreeTensor(dgamma_Local);
        event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
        DataCopyPad(dgamma_gm[channel_idx], dgamma_temp, copyParams_fp16);
    }
    if (dbeta_is_require) {
        LocalTensor<float> dbeta_Local = in_queue_dbeta_channel.AllocTensor<float>();
        DataCopyPad(dbeta_Local, dbeta_workspace[channel_idx], copyParams_fp32, padParams);
        LocalTensor<T> dbeta_temp = out_tbuf_dbeta_channel_T.Get<T>(count);
        event_t eventIDMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        Cast(dbeta_temp, dbeta_Local, RoundMode::CAST_ROUND, count);
        in_queue_dbeta_channel.FreeTensor(dbeta_Local);
        event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
        DataCopyPad(dbeta_gm[channel_idx], dbeta_temp, copyParams_fp16);
    }
}

__aicore__ inline void reduce_axis_n_WSP2UB(TQue<QuePosition::VECIN, 1>& vecInQue, const GlobalTensor<float>& workspace,
                                            uint64_t workSpaceOffset, uint64_t repeatTime, uint32_t count,
                                            const LocalTensor<float>& dstLocal)
{
    LocalTensor<float> vecInLocal = vecInQue.AllocTensor<float>();
    DataCopyExtParams copyParams_in{static_cast<uint16_t>(repeatTime), static_cast<uint32_t>(count * sizeof(float)),
                                    static_cast<uint32_t>((C - count) * sizeof(float)),
                                    static_cast<uint32_t>((castEleNum - count) / float_per_block), 0};
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
    DataCopyPad(vecInLocal, workspace[workSpaceOffset], copyParams_in, padParams);
    event_t eventIDMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    uint64_t end = repeatTime;
    uint64_t mid = repeatTime;
    constexpr int MIN_LOOP_THRESHOLD = 2;
    while (mid >= MIN_LOOP_THRESHOLD) {
        mid >>= 1;
        end -= mid;
        PipeBarrier<PIPE_V>();
        Add(vecInLocal, vecInLocal[end * castEleNum], vecInLocal, mid * castEleNum);
        PipeBarrier<PIPE_V>();
        mid = end;
    }
    PipeBarrier<PIPE_V>();
    Add(dstLocal, vecInLocal, dstLocal, count);
    vecInQue.FreeTensor(vecInLocal);
}

__aicore__ inline void reduce_dgamma_dbeta_WSP2GM(uint64_t start_offset, uint64_t channel_idx, uint32_t count,
                                                  uint64_t reduce_axis_num)
{
    uint64_t repeatTime = 0;
    LocalTensor<float> dbeta_sum_Local = cal_queue_dbeta_reduce.AllocTensor<float>();
    LocalTensor<float> dgamma_sum_Local = cal_queue_dgamma_reduce.AllocTensor<float>();
    Duplicate<float>(dbeta_sum_Local, 0.0f, castEleNum);
    Duplicate<float>(dgamma_sum_Local, 0.0f, castEleNum);
    PipeBarrier<PIPE_V>();
    for (uint64_t loop_idx = 0; loop_idx < divceil(reduce_axis_num, coreBatchParts); loop_idx++) {
        uint64_t loop_offset = loop_idx * coreBatchParts * C;
        if (loop_idx < divceil(reduce_axis_num, coreBatchParts) - 1) {
            repeatTime = coreBatchParts;
        } else if (loop_idx == divceil(reduce_axis_num, coreBatchParts) - 1) {
            repeatTime = reduce_axis_num % coreBatchParts == 0 ? coreBatchParts : reduce_axis_num % coreBatchParts;
        }
        if (dgamma_is_require) {
            reduce_axis_n_WSP2UB(in_queue_dgamma_channel, dgamma_workspace, start_offset + loop_offset + channel_idx,
                                 repeatTime, count, dgamma_sum_Local);
        }
        if (dbeta_is_require) {
            reduce_axis_n_WSP2UB(in_queue_dbeta_channel, dbeta_workspace, start_offset + loop_offset + channel_idx,
                                 repeatTime, count, dbeta_sum_Local);
        }
    }
    event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
    DataCopyExtParams copyParams_out{1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0};
    if (dgamma_is_require) {
        SetAtomicAdd<float>();
        DataCopyPad(dgamma_gm[channel_idx], dgamma_sum_Local, copyParams_out);
        SetAtomicNone();
    }
    cal_queue_dgamma_reduce.FreeTensor(dgamma_sum_Local);
    if (dbeta_is_require) {
        SetAtomicAdd<float>();
        DataCopyPad(dbeta_gm[channel_idx], dbeta_sum_Local, copyParams_out);
        SetAtomicNone();
    }
    cal_queue_dbeta_reduce.FreeTensor(dbeta_sum_Local);
}

__aicore__ inline void reduce_cast_dgamma_dbeta_WSP2GM(uint64_t channel_idx, uint32_t count, uint64_t reduce_axis_num)
{
    uint64_t repeatTime = 0;
    LocalTensor<float> dbeta_sum_Local = cal_queue_dbeta_reduce.AllocTensor<float>();
    LocalTensor<float> dgamma_sum_Local = cal_queue_dgamma_reduce.AllocTensor<float>();
    Duplicate<float>(dbeta_sum_Local, 0.0f, castEleNum);
    Duplicate<float>(dgamma_sum_Local, 0.0f, castEleNum);
    PipeBarrier<PIPE_V>();
    for (uint64_t loop_idx = 0; loop_idx < divceil(reduce_axis_num, coreBatchParts); loop_idx++) {
        uint64_t loop_offset = loop_idx * coreBatchParts * C;
        if (loop_idx < divceil(reduce_axis_num, coreBatchParts) - 1) {
            repeatTime = coreBatchParts;
        } else if (loop_idx == divceil(reduce_axis_num, coreBatchParts) - 1) {
            repeatTime = reduce_axis_num % coreBatchParts == 0 ? coreBatchParts : reduce_axis_num % coreBatchParts;
        }
        if (dgamma_is_require) {
            reduce_axis_n_WSP2UB(in_queue_dgamma_channel, dgamma_workspace, loop_offset + channel_idx, repeatTime,
                                 count, dgamma_sum_Local);
        }
        if (dbeta_is_require) {
            reduce_axis_n_WSP2UB(in_queue_dbeta_channel, dbeta_workspace, loop_offset + channel_idx, repeatTime, count,
                                 dbeta_sum_Local);
        }
    }
    PipeBarrier<PIPE_ALL>();
    DataCopyExtParams copyParams_out{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
    if (dgamma_is_require) {
        Custom_DataCopy_Out(dgamma_sum_Local, dgamma_gm, channel_idx, out_tbuf_dgamma_channel_T, count);
    }
    cal_queue_dgamma_reduce.FreeTensor(dgamma_sum_Local);
    if (dbeta_is_require) {
        Custom_DataCopy_Out(dbeta_sum_Local, dbeta_gm, channel_idx, out_tbuf_dbeta_channel_T, count);
    }
    cal_queue_dbeta_reduce.FreeTensor(dbeta_sum_Local);
}
