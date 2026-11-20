/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* Output and unified buffer initialization.
 * Include only from group_norm_grad_kernel.h. */

//----------------------------------------------tools_func--------------------------------------//
__aicore__ inline void Init_Output_GM()
{
    if constexpr (!isDeterministic && IsSameType<T, float>::value) {
        if (this->curBlockIdx == 0) {
            InitOutput<float>(dgamma_gm, C, 0.0f);
            InitOutput<float>(dbeta_gm, C, 0.0f);
        }
    } else if constexpr (!isDeterministic && !IsSameType<T, float>::value) {
        if (this->curBlockIdx == 0) {
            InitOutput<float>(dgamma_workspace, C, 0.0f);
            InitOutput<float>(dbeta_workspace, C, 0.0f);
        }
    } else if constexpr (isDeterministic && IsSameType<T, float>::value) {
        if (this->curBlockIdx == 0) {
            InitOutput<float>(dgamma_workspace, workSpaceSize, 0.0f);
            InitOutput<float>(dbeta_workspace, workSpaceSize, 0.0f);
            InitOutput<float>(dgamma_gm, C, 0.0f);
            InitOutput<float>(dbeta_gm, C, 0.0f);
        }
    } else if constexpr (isDeterministic && !IsSameType<T, float>::value) {
        if (this->curBlockIdx == 0) {
            InitOutput<float>(dgamma_workspace, workSpaceSize, 0.0f);
            InitOutput<float>(dbeta_workspace, workSpaceSize, 0.0f);
        }
    }
}

__aicore__ inline void Init_Unified_Buffer_FP32_MODE_0(const GroupNormGradTilingData* tiling_data)
{
    // VEC_IN
    uint32_t HxW_alloc_space = ceil(this->ele_num_per_channel, T_per_block) * sizeof(T);
    uint32_t C_GxHxW_alloc_space = ceil(this->ele_num_per_group, T_per_block) * sizeof(T);
    uint32_t NxG_alloc_space = T_per_block * sizeof(T);
    uint32_t C_G_alloc_space = ceil(this->C_G, T_per_block) * sizeof(T);
    pipe.InitBuffer(in_queue_dy, 1, C_GxHxW_alloc_space);
    pipe.InitBuffer(in_queue_x, 1, C_GxHxW_alloc_space);
    pipe.InitBuffer(in_queue_gamma, 1, C_G_alloc_space);
    // VEC_OUT
    pipe.InitBuffer(out_queue_dgamma, 1, C_G_alloc_space);
    pipe.InitBuffer(out_queue_dbeta, 1, C_G_alloc_space);
    pipe.InitBuffer(temp_queue_HXW, 1, C_GxHxW_alloc_space);
    if (!(this->ele_num_per_channel % float_per_block == 0 || this->C_G == 1)) {
        pipe.InitBuffer(in_temp_x, HxW_alloc_space);
        pipe.InitBuffer(in_temp_dy, HxW_alloc_space);
    }
}

__aicore__ inline void Init_Unified_Buffer_FP32_MODE_1(const GroupNormGradTilingData* tiling_data)
{
    // VEC_IN
    uint32_t HxW_alloc_space = ceil(this->ele_num_per_channel, T_per_block) * sizeof(float);
    uint32_t NxG_alloc_space = T_per_block * sizeof(float);
    uint32_t C_G_alloc_space = ceil(this->C_G, T_per_block) * sizeof(float);
    pipe.InitBuffer(in_queue_dy, 1, HxW_alloc_space);
    pipe.InitBuffer(in_queue_x, 1, HxW_alloc_space);
    pipe.InitBuffer(in_queue_gamma, 1, C_G_alloc_space);
    // VEC_OUT
    pipe.InitBuffer(out_queue_dgamma, 1, C_G_alloc_space);
    pipe.InitBuffer(out_queue_dbeta, 1, C_G_alloc_space);
    pipe.InitBuffer(temp_queue_HXW, 1, HxW_alloc_space);
}

__aicore__ inline void Init_Unified_Buffer_FP32_MODE_2(const GroupNormGradTilingData* tiling_data)
{
    // VEC_IN
    uint32_t HxW_alloc_space = ceil(this->ele_num_per_channel, T_per_block) * sizeof(float);
    uint32_t NxG_alloc_space = T_per_block * sizeof(float);
    uint32_t C_G_alloc_space = ceil(this->C_G, T_per_block) * sizeof(float);
    pipe.InitBuffer(in_queue_dy, DOUBLE_BUFFER, HxW_alloc_space);
    pipe.InitBuffer(in_queue_x, DOUBLE_BUFFER, HxW_alloc_space);
    pipe.InitBuffer(in_queue_gamma, 1, C_G_alloc_space);
    // VEC_OUT
    pipe.InitBuffer(out_queue_dgamma, 1, C_G_alloc_space);
    pipe.InitBuffer(out_queue_dbeta, 1, C_G_alloc_space);
    pipe.InitBuffer(temp_queue_HXW, DOUBLE_BUFFER, HxW_alloc_space);
}

__aicore__ inline void Init_Unified_Buffer_FP32_MODE_3(const GroupNormGradTilingData* tiling_data)
{
    this->mode2_ub_capacity_ele = tiling_data->mode2_ub_capacity_ele;
    this->mode2_ub_iteration_num = tiling_data->mode2_ub_iteration_num;
    this->mode2_ub_tail_num = tiling_data->mode2_ub_tail_num;
    // VEC_IN
    uint32_t capacity_ele_alloc_space = ceil(this->mode2_ub_capacity_ele, T_per_block) * sizeof(float);
    uint32_t NxG_alloc_space = T_per_block * sizeof(float);
    uint32_t C_G_alloc_space = ceil(this->C_G, T_per_block) * sizeof(float);
    pipe.InitBuffer(in_queue_dy, 1, capacity_ele_alloc_space);
    pipe.InitBuffer(in_queue_x, 1, capacity_ele_alloc_space);
    pipe.InitBuffer(in_queue_gamma, 1, C_G_alloc_space);
    pipe.InitBuffer(temp_queue_HXW, 1, capacity_ele_alloc_space);
    pipe.InitBuffer(out_queue_dgamma, 1, C_G_alloc_space);
    pipe.InitBuffer(out_queue_dbeta, 1, C_G_alloc_space);
}

__aicore__ inline void Init_Unified_Buffer_FP32_MODE_5(const GroupNormGradTilingData* tiling_data)
{
    // VEC_IN
    uint32_t C_G_alloc_space = ceil(this->C_G, T_per_block) * sizeof(T);
    pipe.InitBuffer(in_queue_dy, 1, C_G_alloc_space);
    pipe.InitBuffer(in_queue_x, 1, C_G_alloc_space);
    pipe.InitBuffer(in_queue_gamma, 1, C_G_alloc_space);
    // VEC_OUT
    pipe.InitBuffer(out_queue_dgamma, 1, C_G_alloc_space);
    pipe.InitBuffer(out_queue_dbeta, 1, C_G_alloc_space);
    pipe.InitBuffer(temp_queue_HXW, 1, C_G_alloc_space);
}

__aicore__ inline void Init_Unified_Buffer_FP16_BF16_MODE_0(const GroupNormGradTilingData* tiling_data)
{
    uint32_t HxW_alloc_space = ceil(this->ele_num_per_channel, T_per_block);
    uint32_t C_GxHxW_alloc_space = ceil(this->ele_num_per_group, T_per_block);
    uint32_t NxG_alloc_space = T_per_block;
    uint32_t C_G_alloc_space = ceil(this->C_G, T_per_block);
    // VEC_IN
    pipe.InitBuffer(in_queue_dy, 1, C_GxHxW_alloc_space * sizeof(float));
    pipe.InitBuffer(in_tbuf_dy_T, C_GxHxW_alloc_space * sizeof(T));
    pipe.InitBuffer(in_queue_x, 1, C_GxHxW_alloc_space * sizeof(float));
    if (!(this->ele_num_per_channel % float_per_block == 0 || this->C_G == 1)) {
        pipe.InitBuffer(in_temp_x, HxW_alloc_space * sizeof(float));
        pipe.InitBuffer(in_temp_dy, HxW_alloc_space * sizeof(float));
    }
    pipe.InitBuffer(in_tbuf_x_T, C_GxHxW_alloc_space * sizeof(T));
    pipe.InitBuffer(in_queue_gamma, 1, C_G_alloc_space * sizeof(float));
    pipe.InitBuffer(in_tbuf_gamma_T, C_G_alloc_space * sizeof(T));
    pipe.InitBuffer(out_tbuf_dgamma_T, C_G_alloc_space * sizeof(T));
    pipe.InitBuffer(out_tbuf_dbeta_T, C_G_alloc_space * sizeof(T));
    // VEC_OUT
    pipe.InitBuffer(out_queue_dgamma, 1, C_G_alloc_space * sizeof(float));
    pipe.InitBuffer(out_queue_dbeta, 1, C_G_alloc_space * sizeof(float));
    pipe.InitBuffer(temp_queue_HXW, 1, C_GxHxW_alloc_space * sizeof(float));
}

__aicore__ inline void Init_Unified_Buffer_FP16_BF16_MODE_1(const GroupNormGradTilingData* tiling_data)
{
    // VEC_IN
    uint32_t HxW_alloc_space = ceil(this->ele_num_per_channel, T_per_block);
    uint32_t NxG_alloc_space = T_per_block;
    uint32_t C_G_alloc_space = ceil(this->C_G, T_per_block);
    pipe.InitBuffer(in_queue_dy, 1, HxW_alloc_space * sizeof(float));
    pipe.InitBuffer(in_tbuf_dy_T, HxW_alloc_space * sizeof(T));
    pipe.InitBuffer(in_queue_x, 1, HxW_alloc_space * sizeof(float));
    pipe.InitBuffer(in_tbuf_x_T, HxW_alloc_space * sizeof(T));
    pipe.InitBuffer(in_queue_gamma, 1, C_G_alloc_space * sizeof(float));
    pipe.InitBuffer(in_tbuf_gamma_T, C_G_alloc_space * sizeof(T));
    pipe.InitBuffer(out_tbuf_dgamma_T, C_G_alloc_space * sizeof(T));
    pipe.InitBuffer(out_tbuf_dbeta_T, C_G_alloc_space * sizeof(T));
    // VEC_OUT
    pipe.InitBuffer(out_queue_dgamma, 1, C_G_alloc_space * sizeof(float));
    pipe.InitBuffer(out_queue_dbeta, 1, C_G_alloc_space * sizeof(float));
    pipe.InitBuffer(temp_queue_HXW, 1, HxW_alloc_space * sizeof(float));
}

__aicore__ inline void Init_Unified_Buffer_FP16_BF16_MODE_2(const GroupNormGradTilingData* tiling_data)
{
    // VEC_IN
    uint32_t HxW_alloc_space = ceil(this->ele_num_per_channel, T_per_block);
    uint32_t NxG_alloc_space = T_per_block;
    uint32_t C_G_alloc_space = ceil(this->C_G, T_per_block);
    pipe.InitBuffer(in_queue_dy, DOUBLE_BUFFER, HxW_alloc_space * sizeof(float));
    pipe.InitBuffer(in_tbuf_dy_T, HxW_alloc_space * sizeof(T));
    pipe.InitBuffer(in_queue_x, DOUBLE_BUFFER, HxW_alloc_space * sizeof(float));
    pipe.InitBuffer(in_tbuf_x_T, HxW_alloc_space * sizeof(T));
    pipe.InitBuffer(in_queue_gamma, 1, C_G_alloc_space * sizeof(float));
    pipe.InitBuffer(in_tbuf_gamma_T, C_G_alloc_space * sizeof(T));
    pipe.InitBuffer(out_tbuf_dgamma_T, C_G_alloc_space * sizeof(T));
    pipe.InitBuffer(out_tbuf_dbeta_T, C_G_alloc_space * sizeof(T));
    // VEC_OUT
    pipe.InitBuffer(out_queue_dgamma, 1, C_G_alloc_space * sizeof(float));
    pipe.InitBuffer(out_queue_dbeta, 1, C_G_alloc_space * sizeof(float));
    pipe.InitBuffer(temp_queue_HXW, DOUBLE_BUFFER, HxW_alloc_space * sizeof(float));
}

__aicore__ inline void Init_Unified_Buffer_FP16_BF16_MODE_3(const GroupNormGradTilingData* tiling_data)
{
    this->mode2_ub_capacity_ele = tiling_data->mode2_ub_capacity_ele;
    this->mode2_ub_iteration_num = tiling_data->mode2_ub_iteration_num;
    this->mode2_ub_tail_num = tiling_data->mode2_ub_tail_num;
    uint32_t capacity_ele_alloc_space = ceil(this->mode2_ub_capacity_ele, T_per_block);
    uint32_t NxG_alloc_space = T_per_block;
    uint32_t C_G_alloc_space = ceil(this->C_G, T_per_block);
    // VEC_IN
    pipe.InitBuffer(in_queue_dy, 1, capacity_ele_alloc_space * sizeof(float));
    pipe.InitBuffer(in_tbuf_dy_T, capacity_ele_alloc_space * sizeof(T));
    pipe.InitBuffer(in_queue_x, 1, capacity_ele_alloc_space * sizeof(float));
    pipe.InitBuffer(in_tbuf_x_T, capacity_ele_alloc_space * sizeof(T));
    pipe.InitBuffer(in_queue_gamma, 1, C_G_alloc_space * sizeof(float));
    pipe.InitBuffer(in_tbuf_gamma_T, C_G_alloc_space * sizeof(T));
    pipe.InitBuffer(out_tbuf_dgamma_T, C_G_alloc_space * sizeof(T));
    pipe.InitBuffer(out_tbuf_dbeta_T, C_G_alloc_space * sizeof(T));
    // VEC_OUT
    pipe.InitBuffer(out_queue_dgamma, 1, C_G_alloc_space * sizeof(float));
    pipe.InitBuffer(out_queue_dbeta, 1, C_G_alloc_space * sizeof(float));
    pipe.InitBuffer(temp_queue_HXW, 1, capacity_ele_alloc_space * sizeof(float));
}

__aicore__ inline void Init_Unified_Buffer_FP16_BF16_MODE_5(const GroupNormGradTilingData* tiling_data)
{
    // VEC_IN
    uint32_t C_G_alloc_space = ceil(this->C_G, T_per_block);
    pipe.InitBuffer(in_queue_dy, 1, C_G_alloc_space * sizeof(float));
    pipe.InitBuffer(in_tbuf_dy_T, C_G_alloc_space * sizeof(T));
    pipe.InitBuffer(in_queue_x, 1, C_G_alloc_space * sizeof(float));
    pipe.InitBuffer(in_tbuf_x_T, C_G_alloc_space * sizeof(T));
    pipe.InitBuffer(in_queue_gamma, 1, C_G_alloc_space * sizeof(float));
    pipe.InitBuffer(in_tbuf_gamma_T, C_G_alloc_space * sizeof(T));
    pipe.InitBuffer(out_tbuf_dgamma_T, C_G_alloc_space * sizeof(T));
    pipe.InitBuffer(out_tbuf_dbeta_T, C_G_alloc_space * sizeof(T));
    // VEC_OUT
    pipe.InitBuffer(out_queue_dgamma, 1, C_G_alloc_space * sizeof(float));
    pipe.InitBuffer(out_queue_dbeta, 1, C_G_alloc_space * sizeof(float));
    pipe.InitBuffer(temp_queue_HXW, 1, C_G_alloc_space * sizeof(float));
}

__aicore__ inline void Init_Unified_Buffer(const GroupNormGradTilingData* tiling_data)
{
    if constexpr (IsSameType<T, float>::value) {
        if (Tiling_key == MODE_0) {
            Init_Unified_Buffer_FP32_MODE_0(tiling_data);
        } else if (Tiling_key == MODE_1) {
            Init_Unified_Buffer_FP32_MODE_1(tiling_data);
        } else if (Tiling_key == MODE_2) {
            Init_Unified_Buffer_FP32_MODE_2(tiling_data);
        } else if (Tiling_key == MODE_3) {
            Init_Unified_Buffer_FP32_MODE_3(tiling_data);
        } else if (Tiling_key == MODE_5) {
            Init_Unified_Buffer_FP32_MODE_5(tiling_data);
        }
    } else if constexpr (IsSameType<T, half>::value || IsSameType<T, bfloat16_t>::value) {
        if (Tiling_key == MODE_0) {
            Init_Unified_Buffer_FP16_BF16_MODE_0(tiling_data);
        } else if (Tiling_key == MODE_1) {
            Init_Unified_Buffer_FP16_BF16_MODE_1(tiling_data);
        } else if (Tiling_key == MODE_2) {
            Init_Unified_Buffer_FP16_BF16_MODE_2(tiling_data);
        } else if (Tiling_key == MODE_3) {
            Init_Unified_Buffer_FP16_BF16_MODE_3(tiling_data);
        } else if (Tiling_key == MODE_5) {
            Init_Unified_Buffer_FP16_BF16_MODE_5(tiling_data);
        }
    }
}
