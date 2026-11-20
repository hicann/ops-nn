/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* Mode 0, 1, and 2 kernels.
 * Include only from group_norm_grad_kernel.h. */

//----------------------------------------------MODE0--------------------------------------//
__aicore__ inline void CopyIn_Mode_0(uint64_t task_idx)
{
    // alloc tensor from queue memory
    uint64_t offset = task_idx * this->ele_num_per_group;
    uint64_t once_process_ele_num = this->ele_num_per_group;
    LocalTensor<float> dy_local = in_queue_dy.AllocTensor<float>();
    Custom_DataCopy_In(dy_local, in_tbuf_dy_T, dy_gm, offset, once_process_ele_num);
    in_queue_dy.EnQue(dy_local);
    LocalTensor<float> x_local = in_queue_x.AllocTensor<float>();
    Custom_DataCopy_In(x_local, in_tbuf_x_T, x_gm, offset, once_process_ele_num);
    event_t eventIDMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    Get_Mean_Rstd(task_idx);
    event_t eventIDSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIDSToV);
    WaitFlag<HardEvent::S_V>(eventIDSToV);
    WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
    Adds_Muls_template(x_local, x_local, (float(-1.0) * this->mean_scalar * this->rstd_scalar), this->rstd_scalar,
                       this->ele_num_per_group);
    in_queue_x.EnQue(x_local);
}

__aicore__ inline void Compute_Mode_0(uint64_t task_idx)
{
    // deque input tensors from VECIN queue
    LocalTensor<float> dy_local = in_queue_dy.DeQue<float>();
    LocalTensor<float> x_local = in_queue_x.DeQue<float>();
    LocalTensor<float> temp_1_local = out_queue_dbeta.AllocTensor<float>();
    LocalTensor<float> temp_2_local = out_queue_dgamma.AllocTensor<float>();
    LocalTensor<float> temp_hxw_local = temp_queue_HXW.AllocTensor<float>();

    for (uint64_t C_G_idx = 0; C_G_idx < this->C_G; C_G_idx++) {
        uint64_t is_align = (C_G_idx * this->ele_num_per_channel) % this->float_per_block;
        if (is_align == 0) {
            float temp1I = ReduceSum_template(temp_hxw_local, dy_local[C_G_idx * this->ele_num_per_channel],
                                              temp_hxw_local, this->ele_num_per_channel, true);
            temp_1_local.SetValue(C_G_idx, temp1I);
            event_t eventIDSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(eventIDSToV);
            WaitFlag<HardEvent::S_V>(eventIDSToV);
            Mul(temp_hxw_local, x_local[C_G_idx * this->ele_num_per_channel],
                dy_local[C_G_idx * this->ele_num_per_channel], this->ele_num_per_channel);
            float temp2I = ReduceSum_template(temp_hxw_local, temp_hxw_local, temp_hxw_local, this->ele_num_per_channel,
                                              true);
            temp_2_local.SetValue(C_G_idx, temp2I);
        } else {
            uint64_t gm_offset = task_idx * this->ele_num_per_group + C_G_idx * this->ele_num_per_channel;
            LocalTensor<float> unalign_channel_x = in_temp_x.Get<float>(this->ele_num_per_channel);
            LocalTensor<float> unalign_channel_dy = in_temp_dy.Get<float>(this->ele_num_per_channel);
            Custom_DataCopy_In(unalign_channel_x, in_tbuf_x_T, x_gm, gm_offset, this->ele_num_per_channel);
            Custom_DataCopy_In(unalign_channel_dy, in_tbuf_dy_T, dy_gm, gm_offset, this->ele_num_per_channel);
            event_t eventIDMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
            Adds_Muls_template(unalign_channel_x, unalign_channel_x,
                               (float(-1.0) * this->mean_scalar * this->rstd_scalar), this->rstd_scalar,
                               this->ele_num_per_channel);
            float temp1I = ReduceSum_template(temp_hxw_local, unalign_channel_dy, temp_hxw_local,
                                              this->ele_num_per_channel, true);
            Mul(temp_hxw_local, unalign_channel_x, unalign_channel_dy, this->ele_num_per_channel);
            float temp2I = ReduceSum_template(temp_hxw_local, temp_hxw_local, temp_hxw_local, this->ele_num_per_channel,
                                              true);
            PipeBarrier<PIPE_ALL>();
            temp_1_local.SetValue(C_G_idx, temp1I);
            temp_2_local.SetValue(C_G_idx, temp2I);
        }
    }
    temp_queue_HXW.FreeTensor(temp_hxw_local);
    Mode_Selection(task_idx, temp_2_local, temp_1_local);
    if (dx_is_require) {
        uint64_t channel_idx = (task_idx % this->G) * this->C_G;
        LocalTensor<float> gamma_local = in_queue_gamma.AllocTensor<float>();
        Custom_DataCopy_In(gamma_local, in_tbuf_gamma_T, gamma_gm, channel_idx, this->C_G);
        event_t eventIDMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        float gamma = 0;
        float c1 = 0;
        float c2 = 0;
        Mul_ReduceSum_template(temp_1_local, temp_2_local, gamma_local, C_G, c1, c2);
        for (uint64_t C_G_idx = 0; C_G_idx < this->C_G; C_G_idx++) {
            uint64_t is_align = (C_G_idx * this->ele_num_per_channel) % this->T_per_block;
            uint64_t gm_offset = task_idx * this->ele_num_per_group + C_G_idx * this->ele_num_per_channel;
            gamma = gamma_local.GetValue(C_G_idx);
            event_t eventIDSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(eventIDSToV);
            if (is_align == 0) {
                uint64_t offset = C_G_idx * this->ele_num_per_channel;
                Adds_Muls_template(x_local[offset], x_local[offset], (this->rstd_scalar * c1), (this->rstd_scalar * c2),
                                   this->ele_num_per_channel);
                WaitFlag<HardEvent::S_V>(eventIDSToV);
                Muls(dy_local[offset], dy_local[offset], (this->rstd_scalar * gamma), this->ele_num_per_channel);
                LocalTensor<float> temp_hxw_local = temp_queue_HXW.AllocTensor<float>();
                Sub(temp_hxw_local, dy_local[offset], x_local[offset], this->ele_num_per_channel);
                event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
                SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
                WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
                Custom_DataCopy_Out(temp_hxw_local, gm_offset, in_tbuf_dy_T, ele_num_per_channel);
                temp_queue_HXW.FreeTensor(temp_hxw_local);
            } else {
                uint64_t offset = floor(C_G_idx * this->ele_num_per_channel, this->T_per_block);
                PipeBarrier<PIPE_ALL>();
                Custom_DataCopy_In(x_local, offset, in_tbuf_x_T, x_gm, gm_offset, this->ele_num_per_channel);
                Custom_DataCopy_In(dy_local, offset, in_tbuf_dy_T, dy_gm, gm_offset, this->ele_num_per_channel);
                event_t eventIDMTE2ToV0 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV0);
                WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV0);
                Adds_Muls_template(
                    x_local[offset], x_local[offset],
                    (this->rstd_scalar * c1 - this->mean_scalar * this->rstd_scalar * this->rstd_scalar * c2),
                    (this->rstd_scalar * this->rstd_scalar * c2), this->ele_num_per_channel);
                WaitFlag<HardEvent::S_V>(eventIDSToV);
                Muls(dy_local[offset], dy_local[offset], (this->rstd_scalar * gamma), this->ele_num_per_channel);
                LocalTensor<float> temp_hxw_local = temp_queue_HXW.AllocTensor<float>();
                Sub(temp_hxw_local, dy_local[offset], x_local[offset], this->ele_num_per_channel);
                PipeBarrier<PIPE_ALL>();
                event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
                SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
                WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
                Custom_DataCopy_Out(temp_hxw_local, gm_offset, in_tbuf_dy_T, ele_num_per_channel);
                temp_queue_HXW.FreeTensor(temp_hxw_local);
            }
        }
        in_queue_gamma.FreeTensor(gamma_local);
    }
    out_queue_dbeta.FreeTensor(temp_1_local);
    out_queue_dgamma.FreeTensor(temp_2_local);
    in_queue_dy.FreeTensor(dy_local);
    in_queue_x.FreeTensor(x_local);
}
//----------------------------------------------MODE1--------------------------------------//
__aicore__ inline void Compute_Mode_1(uint64_t task_idx)
{
    LocalTensor<float> temp_1_local = out_queue_dbeta.AllocTensor<float>();
    LocalTensor<float> temp_2_local = out_queue_dgamma.AllocTensor<float>();
    uint64_t base_offset = task_idx * this->ele_num_per_group;
    uint64_t once_process_ele_num = this->ele_num_per_channel;
    LocalTensor<T> dy_T_local;
    LocalTensor<T> x_T_local;
    LocalTensor<float> dy_local;
    LocalTensor<float> x_local;
    LocalTensor<float> temp_hxw_local;
    for (uint64_t C_G_idx = 0; C_G_idx < this->C_G; C_G_idx++) {
        uint64_t offset = base_offset + C_G_idx * this->ele_num_per_channel;
        dy_local = in_queue_dy.AllocTensor<float>();
        Custom_DataCopy_In(dy_local, in_tbuf_dy_T, dy_gm, offset, once_process_ele_num);
        x_local = in_queue_x.AllocTensor<float>();
        Custom_DataCopy_In(x_local, in_tbuf_x_T, x_gm, offset, once_process_ele_num);
        event_t eventIDMTE2ToV0 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV0);
        WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV0);
        temp_hxw_local = temp_queue_HXW.AllocTensor<float>();
        float temp1I = ReduceSum_template(temp_hxw_local, dy_local, temp_hxw_local, this->ele_num_per_channel, true);
        Adds_Muls_template(x_local, x_local, (-1.0f * this->mean_scalar * this->rstd_scalar), (this->rstd_scalar),
                           this->ele_num_per_channel);
        Mul(temp_hxw_local, x_local, dy_local, this->ele_num_per_channel);
        in_queue_x.FreeTensor(x_local);
        in_queue_dy.FreeTensor(dy_local);
        float temp2I = ReduceSum_template(temp_hxw_local, temp_hxw_local, temp_hxw_local, this->ele_num_per_channel,
                                          true);
        temp_queue_HXW.FreeTensor(temp_hxw_local);
        temp_1_local.SetValue(C_G_idx, temp1I);
        temp_2_local.SetValue(C_G_idx, temp2I);
    }
    Mode_Selection(task_idx, temp_2_local, temp_1_local);
    if (dx_is_require) {
        float c1 = 0;
        float c2 = 0;
        float gamma = 0;
        uint64_t channel_idx = (task_idx % this->G) * this->C_G;
        LocalTensor<float> gamma_local = in_queue_gamma.AllocTensor<float>();
        Custom_DataCopy_In(gamma_local, in_tbuf_gamma_T, gamma_gm, channel_idx, this->C_G);
        event_t eventIDMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        Mul_ReduceSum_template(temp_1_local, temp_2_local, gamma_local, C_G, c1, c2);
        out_queue_dbeta.FreeTensor(temp_1_local);
        out_queue_dgamma.FreeTensor(temp_2_local);
        for (uint64_t C_G_idx = 0; C_G_idx < this->C_G; C_G_idx++) {
            uint64_t offset = base_offset + C_G_idx * this->ele_num_per_channel;
            gamma = (float)gamma_local.GetValue(C_G_idx);
            event_t eventIDSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(eventIDSToV);
            x_local = in_queue_x.AllocTensor<float>();
            Custom_DataCopy_In(x_local, in_tbuf_x_T, x_gm, offset, once_process_ele_num);
            dy_local = in_queue_dy.AllocTensor<float>();
            Custom_DataCopy_In(dy_local, in_tbuf_dy_T, dy_gm, offset, once_process_ele_num);
            event_t eventIDMTE2ToV0 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV0);
            WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV0);
            WaitFlag<HardEvent::S_V>(eventIDSToV);
            Muls(dy_local, dy_local, gamma, this->ele_num_per_channel);
            Adds_Muls_template(x_local, x_local, (-1.0f * (this->mean_scalar * this->rstd_scalar * c2 - c1)),
                               (this->rstd_scalar * c2), this->ele_num_per_channel);
            Sub(dy_local, dy_local, x_local, this->ele_num_per_channel);
            in_queue_x.FreeTensor(x_local);
            temp_hxw_local = temp_queue_HXW.AllocTensor<float>();
            Muls(temp_hxw_local, dy_local, (this->rstd_scalar), this->ele_num_per_channel);
            in_queue_dy.FreeTensor(dy_local);
            event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
            WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
            Custom_DataCopy_Out(temp_hxw_local, offset, in_tbuf_dy_T, ele_num_per_channel);
            temp_queue_HXW.FreeTensor(temp_hxw_local);
        }
        in_queue_gamma.FreeTensor(gamma_local);
    } else {
        out_queue_dbeta.FreeTensor(temp_1_local);
        out_queue_dgamma.FreeTensor(temp_2_local);
    }
}
//----------------------------------------------MODE2--------------------------------------//
__aicore__ inline void Compute_Mode_2(uint64_t task_idx)
{
    // deque input tensors from VECIN queue
    LocalTensor<float> x_local[2] = {in_queue_x.AllocTensor<float>(), in_queue_x.AllocTensor<float>()};
    LocalTensor<float> dy_local[2] = {in_queue_dy.AllocTensor<float>(), in_queue_dy.AllocTensor<float>()};
    LocalTensor<float> temp_hxw_local[2] = {temp_queue_HXW.AllocTensor<float>(), temp_queue_HXW.AllocTensor<float>()};
    LocalTensor<T> dy_T_local[2];
    LocalTensor<T> x_T_local[2];
    const int DOUBLE_BUFFER_ = 2;
    int restChannelStartIdx = C_G / DOUBLE_BUFFER_ * DOUBLE_BUFFER_;
    uint64_t channel_idx = (task_idx % this->G) * this->C_G;
    LocalTensor<float> temp_1_local = out_queue_dbeta.AllocTensor<float>();
    LocalTensor<float> temp_2_local = out_queue_dgamma.AllocTensor<float>();
    uint64_t offset[2];
    float temp1I[2];
    float temp2I[2];

    uint64_t base_offset = task_idx * this->ele_num_per_group;
    uint64_t once_process_ele_num = this->ele_num_per_channel;
    for (uint64_t C_G_idx = 0; C_G_idx < this->C_G / DOUBLE_BUFFER_; C_G_idx++) {
        for (int i = 0; i < DOUBLE_BUFFER_; i++) {
            offset[i] = base_offset + (C_G_idx * DOUBLE_BUFFER_ + i) * this->ele_num_per_channel;
        }
        for (int i = 0; i < DOUBLE_BUFFER_; i++) {
            Custom_DataCopy_In(dy_local[i], in_tbuf_dy_T, dy_gm, offset[i], once_process_ele_num);
            Custom_DataCopy_In(x_local[i], in_tbuf_x_T, x_gm, offset[i], once_process_ele_num);
        }
        event_t eventIDMTE2ToV0 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV0);
        WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV0);
        for (int i = 0; i < DOUBLE_BUFFER_; i++) {
            temp1I[i] = ReduceSum_template(temp_hxw_local[i], dy_local[i], temp_hxw_local[i], this->ele_num_per_channel,
                                           true);
        }
        for (int i = 0; i < DOUBLE_BUFFER_; i++) {
            Adds_Muls_template(x_local[i], x_local[i], (-1.0f * this->mean_scalar * this->rstd_scalar),
                               this->rstd_scalar, this->ele_num_per_channel);
        }
        for (int i = 0; i < DOUBLE_BUFFER_; i++) {
            Mul(temp_hxw_local[i], x_local[i], dy_local[i], this->ele_num_per_channel);
        }
        for (int i = 0; i < DOUBLE_BUFFER_; i++) {
            temp2I[i] = ReduceSum_template(temp_hxw_local[i], temp_hxw_local[i], temp_hxw_local[i],
                                           this->ele_num_per_channel, true);
        }
        for (int i = 0; i < DOUBLE_BUFFER_; i++) {
            temp_1_local.SetValue(C_G_idx * DOUBLE_BUFFER_ + i, temp1I[i]);
            temp_2_local.SetValue(C_G_idx * DOUBLE_BUFFER_ + i, temp2I[i]);
        }
    }
    if (C_G != restChannelStartIdx) {
        offset[0] = base_offset + restChannelStartIdx * this->ele_num_per_channel;
        Custom_DataCopy_In(dy_local[0], in_tbuf_dy_T, dy_gm, offset[0], once_process_ele_num);
        Custom_DataCopy_In(x_local[0], in_tbuf_x_T, x_gm, offset[0], once_process_ele_num);
        event_t eventIDMTE2ToV0 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV0);
        WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV0);
        temp1I[0] = ReduceSum_template(temp_hxw_local[0], dy_local[0], temp_hxw_local[0], this->ele_num_per_channel,
                                       true);
        Adds_Muls_template(x_local[0], x_local[0], (-1.0f * this->mean_scalar * this->rstd_scalar), this->rstd_scalar,
                           this->ele_num_per_channel);
        Mul(temp_hxw_local[0], x_local[0], dy_local[0], this->ele_num_per_channel);
        temp2I[0] = ReduceSum_template(temp_hxw_local[0], temp_hxw_local[0], temp_hxw_local[0],
                                       this->ele_num_per_channel, true);
        temp_2_local.SetValue(restChannelStartIdx, temp2I[0]);
        temp_1_local.SetValue(restChannelStartIdx, temp1I[0]);
    }
    temp_queue_HXW.FreeTensor(temp_hxw_local[0]);
    temp_queue_HXW.FreeTensor(temp_hxw_local[1]);
    in_queue_dy.FreeTensor(dy_local[0]);
    in_queue_dy.FreeTensor(dy_local[1]);
    in_queue_x.FreeTensor(x_local[0]);
    in_queue_x.FreeTensor(x_local[1]);
    Mode_Selection(task_idx, temp_2_local, temp_1_local);
    if (dx_is_require) {
        float gamma[2] = {0, 0};
        float c1 = 0;
        float c2 = 0;
        LocalTensor<float> gamma_local = in_queue_gamma.AllocTensor<float>();
        Custom_DataCopy_In(gamma_local, in_tbuf_gamma_T, gamma_gm, channel_idx, this->C_G);
        event_t eventIDMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        Mul_ReduceSum_template(temp_1_local, temp_2_local, gamma_local, C_G, c1, c2);
        out_queue_dgamma.FreeTensor(temp_2_local);
        out_queue_dbeta.FreeTensor(temp_1_local);
        DataCopyParams copyParams{1, (uint16_t)(this->ele_num_per_channel * sizeof(float)), 0, 0};
        for (uint64_t C_G_idx = 0; C_G_idx < this->C_G / DOUBLE_BUFFER_; C_G_idx++) {
            for (int i = 0; i < DOUBLE_BUFFER_; i++) {
                offset[i] = base_offset + (C_G_idx * DOUBLE_BUFFER_ + i) * this->ele_num_per_channel;
                gamma[i] = (float)gamma_local.GetValue(C_G_idx * DOUBLE_BUFFER_ + i);
            }
            event_t eventIDSToV0 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(eventIDSToV0);
            for (int i = 0; i < DOUBLE_BUFFER_; i++) {
                x_local[i] = in_queue_x.AllocTensor<float>();
                dy_local[i] = in_queue_dy.AllocTensor<float>();
                Custom_DataCopy_In(x_local[i], in_tbuf_x_T, x_gm, offset[i], once_process_ele_num);
                Custom_DataCopy_In(dy_local[i], in_tbuf_dy_T, dy_gm, offset[i], once_process_ele_num);
            }
            event_t eventIDMTE2ToV0 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV0);
            WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV0);
            WaitFlag<HardEvent::S_V>(eventIDSToV0);
            for (int i = 0; i < DOUBLE_BUFFER_; i++) {
                Muls(dy_local[i], dy_local[i], gamma[i], this->ele_num_per_channel);
                Adds_Muls_template(x_local[i], x_local[i], (-1.0f * (this->mean_scalar * this->rstd_scalar * c2 - c1)),
                                   (this->rstd_scalar * c2), this->ele_num_per_channel);
                Sub(dy_local[i], dy_local[i], x_local[i], this->ele_num_per_channel);
                in_queue_x.FreeTensor(x_local[i]);
                temp_hxw_local[i] = temp_queue_HXW.AllocTensor<float>();
                Muls(temp_hxw_local[i], dy_local[i], this->rstd_scalar, this->ele_num_per_channel);
                in_queue_dy.FreeTensor(dy_local[i]);
                event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
                SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
                WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
                Custom_DataCopy_Out(temp_hxw_local[i], offset[i], in_tbuf_dy_T, ele_num_per_channel);
                temp_queue_HXW.FreeTensor(temp_hxw_local[i]);
            }
        }
        if (C_G != restChannelStartIdx) {
            offset[0] = base_offset + restChannelStartIdx * this->ele_num_per_channel;
            gamma[0] = (float)gamma_local.GetValue(restChannelStartIdx);
            event_t eventIDSToV0 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(eventIDSToV0);
            x_local[0] = in_queue_x.AllocTensor<float>();
            dy_local[0] = in_queue_dy.AllocTensor<float>();
            Custom_DataCopy_In(x_local[0], in_tbuf_x_T, x_gm, offset[0], once_process_ele_num);
            Custom_DataCopy_In(dy_local[0], in_tbuf_dy_T, dy_gm, offset[0], once_process_ele_num);
            event_t eventIDMTE2ToV0 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV0);
            WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV0);
            WaitFlag<HardEvent::S_V>(eventIDSToV0);
            Muls(dy_local[0], dy_local[0], gamma[0], this->ele_num_per_channel);
            Adds_Muls_template(x_local[0], x_local[0], (-1.0f * (this->mean_scalar * this->rstd_scalar * c2 - c1)),
                               (this->rstd_scalar * c2), this->ele_num_per_channel);
            Sub(dy_local[0], dy_local[0], x_local[0], this->ele_num_per_channel);
            in_queue_x.FreeTensor(x_local[0]);
            temp_hxw_local[0] = temp_queue_HXW.AllocTensor<float>();
            Muls(temp_hxw_local[0], dy_local[0], this->rstd_scalar, this->ele_num_per_channel);
            in_queue_dy.FreeTensor(dy_local[0]);
            event_t eventIDVToMTE30 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eventIDVToMTE30);
            WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE30);
            Custom_DataCopy_Out(temp_hxw_local[0], offset[0], in_tbuf_dy_T, ele_num_per_channel);
            temp_queue_HXW.FreeTensor(temp_hxw_local[0]);
        }
        in_queue_gamma.FreeTensor(gamma_local);
    } else {
        out_queue_dgamma.FreeTensor(temp_2_local);
        out_queue_dbeta.FreeTensor(temp_1_local);
    }
}
