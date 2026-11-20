/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* Mode 3 and 5 kernels.
 * Include only from group_norm_grad_kernel.h. */

//----------------------------------------------MODE3--------------------------------------//
__aicore__ inline void CopyIn_x(float& corr1, float& temp1I, uint64_t offset, uint64_t process_num)
{
    LocalTensor<float> x_local = in_queue_x.AllocTensor<float>();
    LocalTensor<float> dy_local = in_queue_dy.AllocTensor<float>();
    Custom_DataCopy_In(dy_local, in_tbuf_dy_T, dy_gm, offset, process_num);
    Custom_DataCopy_In(x_local, in_tbuf_x_T, x_gm, offset, process_num);
    in_queue_x.EnQue(x_local);

    event_t eventIDMTE2ToV0 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV0);
    WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV0);
    LocalTensor<float> temp_hxw_local = temp_queue_HXW.AllocTensor<float>();
    float adjustedSum = ReduceSum_template(temp_hxw_local, dy_local, temp_hxw_local, process_num, true) - corr1;
    float tempSum = adjustedSum + temp1I;
    corr1 = (tempSum - temp1I) - adjustedSum;
    temp1I = tempSum;
    temp_queue_HXW.FreeTensor(temp_hxw_local);
    in_queue_dy.EnQue(dy_local);
}

__aicore__ inline void compute_dgamma(uint64_t offset, uint64_t process_num)
{
    LocalTensor<float> x_local = in_queue_x.DeQue<float>();
    Adds_Muls_template(x_local, x_local, (-1.0f * this->mean_scalar * this->rstd_scalar), (this->rstd_scalar),
                       this->mode2_ub_capacity_ele);
    LocalTensor<float> temp_hxw_local = temp_queue_HXW.AllocTensor<float>();
    LocalTensor<float> dy_local = in_queue_dy.DeQue<float>();
    Mul(temp_hxw_local, x_local, dy_local, process_num);
    temp_queue_HXW.EnQue(temp_hxw_local);
    in_queue_dy.FreeTensor(dy_local);
    in_queue_x.FreeTensor(x_local);
}

__aicore__ inline void Copyout_dgamma(float& corr2, float& temp2I, uint64_t offset, uint64_t process_num)
{
    LocalTensor<float> temp_hxw_local = temp_queue_HXW.DeQue<float>();
    float adjustedSum = ReduceSum_template(temp_hxw_local, temp_hxw_local, temp_hxw_local, process_num, true) - corr2;
    float tempSum = adjustedSum + temp2I;
    corr2 = (tempSum - temp2I) - adjustedSum;
    temp2I = tempSum;
    temp_queue_HXW.FreeTensor(temp_hxw_local);
}

__aicore__ inline void Compute_Mode_3(uint64_t task_idx)
{
    // deque input tensors from VECIN queue
    LocalTensor<float> temp_1_local = out_queue_dbeta.AllocTensor<float>();
    LocalTensor<float> temp_2_local = out_queue_dgamma.AllocTensor<float>();
    float temp1I = 0;
    float temp2I = 0;
    float corr1 = 0;
    float corr2 = 0;
    for (uint64_t iter_C_G_idx = 0; iter_C_G_idx < this->C_G * this->mode2_ub_iteration_num; iter_C_G_idx++) {
        uint64_t offset = task_idx * this->ele_num_per_group +
                          iter_C_G_idx / this->mode2_ub_iteration_num * this->ele_num_per_channel +
                          iter_C_G_idx % this->mode2_ub_iteration_num * this->mode2_ub_capacity_ele;
        if (iter_C_G_idx % this->mode2_ub_iteration_num != this->mode2_ub_iteration_num - 1) {
            CopyIn_x(corr1, temp1I, offset, this->mode2_ub_capacity_ele);
            compute_dgamma(offset, this->mode2_ub_capacity_ele);
            Copyout_dgamma(corr2, temp2I, offset, this->mode2_ub_capacity_ele);
        } else {
            CopyIn_x(corr1, temp1I, offset, this->mode2_ub_tail_num);
            compute_dgamma(offset, this->mode2_ub_tail_num);
            Copyout_dgamma(corr2, temp2I, offset, this->mode2_ub_tail_num);
            temp_1_local.SetValue((int)(iter_C_G_idx / this->mode2_ub_iteration_num), temp1I);
            temp_2_local.SetValue((int)(iter_C_G_idx / this->mode2_ub_iteration_num), temp2I);
            corr1 = 0;
            corr2 = 0;
            temp1I = 0;
            temp2I = 0;
        }
    }
    Mode_Selection(task_idx, temp_2_local, temp_1_local);
    if (dx_is_require) {
        float c1 = 0;
        float c2 = 0;
        uint64_t channel_idx = (task_idx % this->G) * this->C_G;
        LocalTensor<float> gamma_local = in_queue_gamma.AllocTensor<float>();
        Custom_DataCopy_In(gamma_local, in_tbuf_gamma_T, gamma_gm, channel_idx, this->C_G);
        event_t eventIDMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV);
        Mul_ReduceSum_template(temp_1_local, temp_2_local, gamma_local, C_G, c1, c2);
        out_queue_dbeta.FreeTensor(temp_1_local);
        out_queue_dgamma.FreeTensor(temp_2_local);
        float gamma = 0;
        for (uint64_t iter_C_G_idx = 0; iter_C_G_idx < this->C_G * this->mode2_ub_iteration_num; iter_C_G_idx++) {
            uint64_t offset = task_idx * this->ele_num_per_group +
                              iter_C_G_idx / this->mode2_ub_iteration_num * this->ele_num_per_channel +
                              iter_C_G_idx % this->mode2_ub_iteration_num * this->mode2_ub_capacity_ele;
            gamma = (float)gamma_local.GetValue(iter_C_G_idx / this->mode2_ub_iteration_num);
            event_t eventIDSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(eventIDSToV);
            if (iter_C_G_idx % this->mode2_ub_iteration_num != this->mode2_ub_iteration_num - 1) {
                LocalTensor<float> x_local = in_queue_x.AllocTensor<float>();
                LocalTensor<float> dy_local = in_queue_dy.AllocTensor<float>();
                Custom_DataCopy_In(x_local, in_tbuf_x_T, x_gm, offset, this->mode2_ub_capacity_ele);
                Custom_DataCopy_In(dy_local, in_tbuf_dy_T, dy_gm, offset, this->mode2_ub_capacity_ele);
                event_t eventIDMTE2ToV0 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV0);
                WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV0);
                WaitFlag<HardEvent::S_V>(eventIDSToV);
                Muls(dy_local, dy_local, gamma, this->mode2_ub_capacity_ele);
                Adds_Muls_template(x_local, x_local, (-1.0f * (this->mean_scalar * this->rstd_scalar * c2 - c1)),
                                   (this->rstd_scalar * c2), this->mode2_ub_capacity_ele);
                Sub(dy_local, dy_local, x_local, this->mode2_ub_capacity_ele);
                LocalTensor<float> temp_hxw_local = temp_queue_HXW.AllocTensor<float>();
                Muls(temp_hxw_local, dy_local, this->rstd_scalar, this->mode2_ub_capacity_ele);
                in_queue_x.FreeTensor(x_local);
                in_queue_dy.FreeTensor(dy_local);
                event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
                SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
                WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
                Custom_DataCopy_Out(temp_hxw_local, offset, in_tbuf_dy_T, mode2_ub_capacity_ele);
                temp_queue_HXW.FreeTensor(temp_hxw_local);
            } else {
                LocalTensor<float> x_local = in_queue_x.AllocTensor<float>();
                LocalTensor<float> dy_local = in_queue_dy.AllocTensor<float>();
                Custom_DataCopy_In(x_local, in_tbuf_x_T, x_gm, offset, this->mode2_ub_tail_num);
                Custom_DataCopy_In(dy_local, in_tbuf_dy_T, dy_gm, offset, this->mode2_ub_tail_num);
                event_t eventIDMTE2ToV0 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
                SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV0);
                WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV0);
                WaitFlag<HardEvent::S_V>(eventIDSToV);
                Muls(dy_local, dy_local, gamma, this->mode2_ub_tail_num);
                Adds_Muls_template(x_local, x_local, (-1.0f * (this->mean_scalar * this->rstd_scalar * c2 - c1)),
                                   (this->rstd_scalar * c2), this->mode2_ub_tail_num);
                Sub(dy_local, dy_local, x_local, this->mode2_ub_tail_num);
                LocalTensor<float> temp_hxw_local = temp_queue_HXW.AllocTensor<float>();
                Muls(temp_hxw_local, dy_local, this->rstd_scalar, this->mode2_ub_tail_num);
                in_queue_x.FreeTensor(x_local);
                in_queue_dy.FreeTensor(dy_local);
                event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
                SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
                WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
                Custom_DataCopy_Out(temp_hxw_local, offset, in_tbuf_dy_T, mode2_ub_tail_num);
                temp_queue_HXW.FreeTensor(temp_hxw_local);
            }
        }
        in_queue_gamma.FreeTensor(gamma_local);
    } else {
        out_queue_dbeta.FreeTensor(temp_1_local);
        out_queue_dgamma.FreeTensor(temp_2_local);
    }
}
//----------------------------------------------MODE5--------------------------------------//
__aicore__ inline void Compute_Mode_5(uint64_t task_idx)
{
    // deque input tensors from VECIN queue
    LocalTensor<float> dy_local = in_queue_dy.DeQue<float>();
    LocalTensor<float> x_local = in_queue_x.DeQue<float>();
    LocalTensor<float> temp_1_local = out_queue_dbeta.AllocTensor<float>();
    LocalTensor<float> temp_2_local = out_queue_dgamma.AllocTensor<float>();
    DataCopy(temp_1_local, dy_local, ceil(this->C_G, this->float_per_block));
    PipeBarrier<PIPE_ALL>();
    Mul(temp_2_local, x_local, dy_local, this->C_G);
    Mode_Selection(task_idx, temp_2_local, temp_1_local);
    if (dx_is_require) {
        uint64_t channel_idx = (task_idx % this->G) * this->C_G;
        LocalTensor<float> gamma_local = in_queue_gamma.AllocTensor<float>();
        Custom_DataCopy_In(gamma_local, in_tbuf_gamma_T, gamma_gm, channel_idx, this->C_G);
        event_t eventIDMTE2ToV1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIDMTE2ToV1);
        WaitFlag<HardEvent::MTE2_V>(eventIDMTE2ToV1);
        Mul(dy_local, dy_local, gamma_local, this->C_G);
        float sumC1 = 0;
        float sumC2 = 0;
        Mul_ReduceSum_template(temp_1_local, temp_2_local, gamma_local, C_G, sumC1, sumC2);
        out_queue_dbeta.FreeTensor(temp_1_local);
        out_queue_dgamma.FreeTensor(temp_2_local);
        in_queue_gamma.FreeTensor(gamma_local);
        Adds_Muls_template(x_local, x_local, this->rstd_scalar * sumC1, this->rstd_scalar * sumC2, this->C_G);
        Muls(dy_local, dy_local, this->rstd_scalar, this->C_G);
        LocalTensor<float> temp_hxw_local = temp_queue_HXW.AllocTensor<float>();
        Sub(temp_hxw_local, dy_local, x_local, this->C_G);
        in_queue_dy.FreeTensor(dy_local);
        in_queue_x.FreeTensor(x_local);
        uint64_t gm_offset = task_idx * this->C_G;
        event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
        Custom_DataCopy_Out(temp_hxw_local, gm_offset, in_tbuf_dy_T, this->C_G);
        event_t eventIDMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
        temp_queue_HXW.FreeTensor(temp_hxw_local);
    } else {
        out_queue_dbeta.FreeTensor(temp_1_local);
        out_queue_dgamma.FreeTensor(temp_2_local);
        in_queue_dy.FreeTensor(dy_local);
        in_queue_x.FreeTensor(x_local);
    }
}
