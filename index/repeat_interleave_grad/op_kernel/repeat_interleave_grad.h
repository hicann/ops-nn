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
 * \file repeat_interleave_grad.h
 * \brief
 */
#ifndef _REPEAT_INTERLEAVE_GRAD_H_
#define _REPEAT_INTERLEAVE_GRAD_H_
#include "kernel_operator.h"

using namespace AscendC;

template <typename GRAD_T, typename INDEX_T>
class KernelRepeatInterleaveGrad {
public:
  __aicore__ inline KernelRepeatInterleaveGrad() {}
  __aicore__ inline void Init(GM_ADDR input_grad, GM_ADDR repeats, GM_ADDR output_grad, GM_ADDR workspace, const RepeatInterleaveGradTilingData *__restrict tiling)
  {
    InitComputeInfo(tiling);

    // get start index for current core, core parallel
    input_grad_gm.SetGlobalBuffer((__gm__ GRAD_T*)input_grad);
    repeats_gm.SetGlobalBuffer((__gm__ INDEX_T*)repeats);
    output_grad_gm.SetGlobalBuffer((__gm__ GRAD_T*)output_grad);

    // pipe alloc memory to queue, the unit is Bytes
    pipe.InitBuffer(input_grad_in_queue, BUFFER_NUM, element_num_each_loop * sizeof(GRAD_T));
    pipe.InitBuffer(output_grad_out_queue, BUFFER_NUM, element_num_each_loop * sizeof(GRAD_T));

    pipe.InitBuffer(input_grad_fp32_buf, BUFFER_NUM * element_num_each_loop * sizeof(float));
    pipe.InitBuffer(output_grad_fp32_buf, BUFFER_NUM * element_num_each_loop * sizeof(float));
  }

  __aicore__ inline void Process()
  {
    for (int64_t batch_dim_index = 0; batch_dim_index < batch_dim_num_core; batch_dim_index++) {
        batch_dim_index_gm = batch_index_start_core + batch_dim_index;
        input_grad_repeat_offset_gm = batch_dim_index_gm * repeats_i_grad_dim_num * data_dim_num + loop_offset_start_core;

        for (int64_t repeats_index=0; repeats_index < repeats_o_grad_dim_num; repeats_index++) {
            repeats_num = repeats_gm.GetValue(repeats_index);

            for (int64_t loop_index = 0; loop_index < loop_times; loop_index++) {
                ComputeEachLoop(repeats_index, loop_index);
            }

            input_grad_repeat_offset_gm = input_grad_repeat_offset_gm + repeats_num * data_dim_num;
        }
    }
  }

private:
    __aicore__ inline void InitComputeInfo(const RepeatInterleaveGradTilingData *__restrict tiling)
    {
        batch_dim_num = tiling->batch_dim_num;
        repeats_i_grad_dim_num = tiling->repeats_i_grad_dim_num;
        repeats_o_grad_dim_num = tiling->repeats_o_grad_dim_num;
        data_dim_num = tiling->data_dim_num;

        core_index = GetBlockIdx();
        core_num = tiling->core_num;
        if (core_index < (core_num - 1)) {
            batch_dim_num_core = tiling->batch_dim_num_each_core;
        } else {
            batch_dim_num_core = tiling->batch_dim_num_last_core;
        }
        int64_t core_num_each_batch = tiling->core_num_each_batch;
        batch_index_start_core = core_index / core_num_each_batch * tiling->batch_dim_num_each_core;
        loop_offset_start_core = core_index % core_num_each_batch * tiling->element_num_each_core;

        int64_t element_num;
        if ((core_index % core_num_each_batch) == (core_num_each_batch - 1)) {
            element_num = tiling->element_num_last_core;
        } else {
            element_num = tiling->element_num_each_core;
        }
        element_num_each_loop = tiling->element_num_each_loop;

        loop_times = (element_num + element_num_each_loop - 1) / element_num_each_loop;
        element_num_last_loop = element_num - element_num_each_loop * (loop_times - 1);
    }

    __aicore__ inline void ComputeEachLoop(int64_t repeats_index, int64_t loop_index)
    {
        if (loop_index < (loop_times - 1)) {
            element_num_loop = element_num_each_loop;
        } else {
            element_num_loop = element_num_last_loop;
        }

        input_grad_fp32 = input_grad_fp32_buf.Get<float>();
        output_grad_fp32 = output_grad_fp32_buf.Get<float>();

        Duplicate(output_grad_fp32, (float)0.0, element_num_loop);
        PipeBarrier<PIPE_V>();
        DataCopyExtParams copy_params{1, element_num_loop * (uint32_t)sizeof(GRAD_T), 0, 0, 0};
        DataCopyPadExtParams<GRAD_T> pad_params{false, 0, 0, 0};

        for (int64_t repeats_num_index = 0; repeats_num_index < repeats_num; repeats_num_index++) {
            input_grad_offset_gm = input_grad_repeat_offset_gm + loop_index * element_num_each_loop + repeats_num_index * data_dim_num;
            input_grad_local = input_grad_in_queue.AllocTensor<GRAD_T>();
            DataCopyPad(input_grad_local, input_grad_gm[input_grad_offset_gm], copy_params, pad_params);

            input_grad_in_queue.EnQue(input_grad_local);
            input_grad_local = input_grad_in_queue.DeQue<GRAD_T>();
            if constexpr (std::is_same<GRAD_T, half>::value || std::is_same<GRAD_T, bfloat16_t>::value) {
                Cast(input_grad_fp32, input_grad_local, RoundMode::CAST_NONE, element_num_loop);
            } else if constexpr (std::is_same<GRAD_T, float>::value) {
                uint64_t mask = 64;
                Copy(input_grad_fp32, input_grad_local, mask, (element_num_loop + mask -1) / mask, {1, 1, 8, 8});
            }
            PipeBarrier<PIPE_V>();
            Add(output_grad_fp32, output_grad_fp32, input_grad_fp32, element_num_loop);
            PipeBarrier<PIPE_V>();
            input_grad_in_queue.FreeTensor(input_grad_local);
        }
        output_grad_local = output_grad_out_queue.AllocTensor<GRAD_T>();
        if constexpr (std::is_same<GRAD_T, half>::value) {
            Cast(output_grad_local, output_grad_fp32, RoundMode::CAST_NONE, element_num_loop);
        } else if constexpr (std::is_same<GRAD_T, bfloat16_t>::value) {
            Cast(output_grad_local, output_grad_fp32, RoundMode::CAST_RINT, element_num_loop);
        } else if constexpr (std::is_same<GRAD_T, float>::value) {
            uint64_t mask = 64;
            Copy(output_grad_local, output_grad_fp32, mask, (element_num_loop + mask -1) / mask, {1, 1, 8, 8});
        }
        PipeBarrier<PIPE_V>();
        output_grad_out_queue.EnQue<GRAD_T>(output_grad_local);
        output_grad_local = output_grad_out_queue.DeQue<GRAD_T>();
        output_grad_offset_gm = batch_dim_index_gm * repeats_o_grad_dim_num * data_dim_num + repeats_index * data_dim_num + loop_offset_start_core + loop_index * element_num_each_loop;
        DataCopyPad(output_grad_gm[output_grad_offset_gm], output_grad_local, copy_params);
        output_grad_out_queue.FreeTensor(output_grad_local);
    }

    // buffer num: 1 or 2
    constexpr static int32_t BUFFER_NUM = 2;

    // define pipe
    TPipe pipe;

    // define gm tensor
    GlobalTensor<GRAD_T> input_grad_gm;
    GlobalTensor<INDEX_T> repeats_gm;
    GlobalTensor<GRAD_T> output_grad_gm;

    // define mte tensor
    TQue<QuePosition::VECIN, BUFFER_NUM> input_grad_in_queue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> output_grad_out_queue;
    LocalTensor<GRAD_T> input_grad_local;
    LocalTensor<GRAD_T> output_grad_local;

    // define vector tensor
    TBuf<TPosition::VECCALC> input_grad_fp32_buf;
    TBuf<TPosition::VECCALC> output_grad_fp32_buf;
    LocalTensor<float> input_grad_fp32;
    LocalTensor<float> output_grad_fp32;

    // data shape info
    int64_t batch_dim_num;
    int64_t repeats_i_grad_dim_num;
    int64_t repeats_o_grad_dim_num;
    int64_t data_dim_num;

    // kernel info
    int64_t core_index;
    int64_t core_num;
    int64_t batch_dim_num_core;
    int64_t batch_index_start_core;
    int64_t loop_offset_start_core;

    // loop info
    int64_t element_num_each_loop;
    int64_t loop_times;
    int64_t element_num_last_loop;
    uint32_t element_num_loop;

    // loop offset
    int64_t batch_dim_index_gm;
    int64_t input_grad_repeat_offset_gm;
    int64_t input_grad_offset_gm;
    int64_t output_grad_offset_gm;

    INDEX_T repeats_num;
};
#endif // _REPEAT_INTERLEAVE_GRAD_H_