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
 * \file group_norm_grad_kernel.h
 * \brief
 */
#ifndef GROUP_NORM_GRAD_KERNEL_H
#define GROUP_NORM_GRAD_KERNEL_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "kernel_type.h"
using namespace AscendC;
template <typename T, bool isDeterministic>
class GroupNormGrad {
private:
    // Pipe object
    TPipe pipe;
    // Que object
    TQue<QuePosition::VECIN, 1> in_queue_dy;
    TQue<QuePosition::VECIN, 1> in_queue_x;
    TQue<QuePosition::VECIN, 1> in_queue_gamma;
    TQue<QuePosition::VECIN, 1> in_queue_dgamma_channel;
    TQue<QuePosition::VECIN, 1> in_queue_dbeta_channel;
    TQue<QuePosition::VECIN, 1> cal_queue_dgamma_reduce;
    TQue<QuePosition::VECIN, 1> cal_queue_dbeta_reduce;
    TBuf<TPosition::VECCALC> in_tbuf_dy_T;
    TBuf<TPosition::VECCALC> in_tbuf_x_T;
    TBuf<TPosition::VECCALC> in_temp_x;
    TBuf<TPosition::VECCALC> in_temp_dy;
    TBuf<TPosition::VECCALC> in_tbuf_gamma_T;
    TBuf<TPosition::VECCALC> out_tbuf_dgamma_T;
    TBuf<TPosition::VECCALC> out_tbuf_dbeta_T;
    TBuf<TPosition::VECCALC> out_tbuf_dgamma_channel_T;
    TBuf<TPosition::VECCALC> out_tbuf_dbeta_channel_T;
    // Que for output
    TQue<QuePosition::VECOUT, 1> out_queue_dgamma;
    TQue<QuePosition::VECOUT, 1> out_queue_dbeta;
    TQue<QuePosition::VECOUT, 1> temp_queue_HXW;
    // Global Memory
    GlobalTensor<T> dy_gm;
    GlobalTensor<T> mean_gm;
    GlobalTensor<T> rstd_gm;
    GlobalTensor<T> x_gm;
    GlobalTensor<T> gamma_gm;
    GlobalTensor<T> beta_gm;
    GlobalTensor<float> dgamma_workspace;
    GlobalTensor<float> dbeta_workspace;
    GlobalTensor<T> dx_gm;
    GlobalTensor<T> dgamma_gm;
    GlobalTensor<T> dbeta_gm;
    uint32_t Tiling_key;
    bool isGammaEqual = true;
    uint64_t N;
    uint64_t C;
    uint64_t G;
    uint64_t HXW;
    uint64_t NXG;
    uint64_t C_G;
    uint64_t all_ele_num;
    // number of calculations on each core
    uint64_t ele_num_per_group;
    // number of tiles on each core
    uint64_t ele_num_per_channel;
    uint64_t task_idx;
    // number of calculations in each tile
    uint64_t tileLength;
    uint64_t task_num_per_core;
    uint64_t task_num_per_tail_core;
    uint64_t tail_core;
    uint64_t cur_core_task_num;
    uint64_t workSpaceSize;
    uint64_t stage2CoreUsed;
    uint64_t castEleNum;
    uint64_t tailCastNum;
    int32_t curBlockIdx;
    uint64_t start_task_id;
    float mean_scalar;
    float rstd_scalar;
    uint64_t mode2_ub_capacity_ele;
    uint64_t mode2_ub_iteration_num;
    uint64_t mode2_ub_tail_num;
    uint64_t coreBatchParts = 0;
    bool dx_is_require;
    bool dgamma_is_require;
    bool dbeta_is_require;
    uint64_t T_per_block;
    const uint32_t float_per_block = 8;
    const uint16_t DST_BLK_STRIDE = 1;
    const uint16_t SRC_BLK_STRIDE = 1;
    const uint8_t DST_REP_STRIDE = 8;
    const uint8_t SRC_REP_STRIDE = 8;
    const uint32_t BLOCK_BYTES = 32;
    const uint32_t MAX_REPEAT_TIMES = 255;
    const uint32_t COM_REPEAT_TIMES = 64;
    const int32_t MODE_0 = 0;
    const int32_t MODE_1 = 1;
    const int32_t MODE_2 = 2;
    const int32_t MODE_3 = 3;
    const int32_t MODE_4 = 4;
    const int32_t MODE_5 = 5;
    const int DOUBLE_BUFFER = 2;
    const int32_t VECTOR_ONCE_BYTES = 256;

public:
    __aicore__ inline GroupNormGrad(GM_ADDR dy, GM_ADDR mean, GM_ADDR rstd, GM_ADDR x, GM_ADDR gamma, GM_ADDR dx,
                                    GM_ADDR dgamma, GM_ADDR dbeta, GM_ADDR workspace,
                                    const GroupNormGradTilingData* tiling_data)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->curBlockIdx = GetBlockIdx();
        this->Tiling_key = tiling_data->Tiling_key;
        this->N = tiling_data->N;
        this->C = tiling_data->C;
        this->G = tiling_data->G;
        this->HXW = tiling_data->HXW;
        this->NXG = tiling_data->NXG;
        this->C_G = tiling_data->C_G;
        this->task_num_per_core = tiling_data->task_num_per_core;
        this->task_num_per_tail_core = tiling_data->task_num_per_tail_core;
        this->tail_core = tiling_data->tail_core;
        this->workSpaceSize = tiling_data->workSpaceSize;
        this->all_ele_num = N * C * HXW;
        this->ele_num_per_group = C_G * HXW;
        this->ele_num_per_channel = HXW;
        this->dx_is_require = tiling_data->dx_is_require;
        this->dgamma_is_require = tiling_data->dgamma_is_require;
        this->dbeta_is_require = tiling_data->dbeta_is_require;
        this->stage2CoreUsed = tiling_data->stage2CoreUsed;
        this->castEleNum = tiling_data->castEleNum;
        this->tailCastNum = tiling_data->tailCastNum;
        this->coreBatchParts = tiling_data->coreBatchParts;
        this->cur_core_task_num = this->task_num_per_core;
        this->start_task_id = this->task_num_per_core * this->curBlockIdx;

        if (this->curBlockIdx >= this->tail_core) {
            this->cur_core_task_num = this->task_num_per_tail_core;
            this->start_task_id = this->task_num_per_tail_core * this->curBlockIdx + this->tail_core;
        }
        this->T_per_block = BLOCK_BYTES / sizeof(T);
        dy_gm.SetGlobalBuffer((__gm__ T*)dy, this->all_ele_num);
        mean_gm.SetGlobalBuffer((__gm__ T*)mean, this->NXG);
        rstd_gm.SetGlobalBuffer((__gm__ T*)rstd, this->NXG);
        x_gm.SetGlobalBuffer((__gm__ T*)x, this->all_ele_num);
        gamma_gm.SetGlobalBuffer((__gm__ T*)gamma, this->C);
        dx_gm.SetGlobalBuffer((__gm__ T*)dx, this->all_ele_num);
        dgamma_gm.SetGlobalBuffer((__gm__ T*)dgamma, this->C);
        dbeta_gm.SetGlobalBuffer((__gm__ T*)dbeta, this->C);
        dgamma_workspace.SetGlobalBuffer((__gm__ float*)workspace, this->workSpaceSize);
        dbeta_workspace.SetGlobalBuffer((__gm__ float*)workspace + this->workSpaceSize, this->workSpaceSize);
        // init output or workspace
        Init_Output_GM();
        // init unified buffer
        Init_Unified_Buffer(tiling_data);
#ifndef __CCE_KT_TEST__
        // wait core
        SyncAll();
#endif
    }

    __aicore__ inline void Process()
    {
        // tiling strategy, pipeline parallel
        if (Tiling_key == MODE_0) {
            for (uint64_t task_idx = 0; task_idx < this->cur_core_task_num; task_idx++) {
                CopyIn_Mode_0(task_idx + this->start_task_id);
                Compute_Mode_0(task_idx + this->start_task_id);
            }
        } else if (Tiling_key == MODE_1) {
            for (uint64_t task_idx = 0; task_idx < this->cur_core_task_num; task_idx++) {
                Get_Mean_Rstd(task_idx + this->start_task_id);
                Compute_Mode_1(task_idx + this->start_task_id);
            }
        } else if (Tiling_key == MODE_2) {
            for (uint64_t task_idx = 0; task_idx < this->cur_core_task_num; task_idx++) {
                Get_Mean_Rstd(task_idx + this->start_task_id);
                Compute_Mode_2(task_idx + this->start_task_id);
            }
        } else if (Tiling_key == MODE_3) {
            for (uint64_t task_idx = 0; task_idx < this->cur_core_task_num; task_idx++) {
                Get_Mean_Rstd(task_idx + this->start_task_id);
                Compute_Mode_3(task_idx + this->start_task_id);
            }
        } else if (Tiling_key == MODE_5) {
            for (uint64_t task_idx = 0; task_idx < this->cur_core_task_num; task_idx++) {
                CopyIn_Mode_0(task_idx + this->start_task_id);
                Compute_Mode_5(task_idx + this->start_task_id);
            }
        }
        pipe.Reset();
        constexpr uint32_t SPLIT_COUNT = 2;
        if constexpr (!isDeterministic && !IsSameType<T, float>::value) {
#ifndef __CCE_KT_TEST__
            // wait core
            SyncAll();
#endif
            if (N != 1 && curBlockIdx < stage2CoreUsed) {
                Initbuffer4stage2_mode0();
                // cast TO GM
                if (curBlockIdx < stage2CoreUsed - 1) {
                    cast_dgamma_dbeta_WSP2GM(curBlockIdx * castEleNum, castEleNum);
                } else if (curBlockIdx == stage2CoreUsed - 1) {
                    cast_dgamma_dbeta_WSP2GM(curBlockIdx * castEleNum, tailCastNum);
                }
            }
        } else if constexpr (isDeterministic && IsSameType<T, float>::value) {
#ifndef __CCE_KT_TEST__
            // wait core
            SyncAll();
#endif
            Initbuffer4stage2_mode1();
            // reduceSUM TO GM
            //  0 ~ stage2CoreUsed - 2
            if (curBlockIdx < stage2CoreUsed - 1) {
                reduce_dgamma_dbeta_WSP2GM(0, curBlockIdx * castEleNum, castEleNum,
                                           divceil(divceil(N, SPLIT_COUNT), SPLIT_COUNT));
            }
            // stage2CoreUsed ~ 2 * stage2CoreUsed -1
            else if (stage2CoreUsed <= curBlockIdx && curBlockIdx < SPLIT_COUNT * stage2CoreUsed - 1) {
                reduce_dgamma_dbeta_WSP2GM(divceil(divceil(N, SPLIT_COUNT), SPLIT_COUNT) * C,
                                           (curBlockIdx % stage2CoreUsed) * castEleNum, castEleNum,
                                           (divceil(N, SPLIT_COUNT) - divceil(divceil(N, SPLIT_COUNT), SPLIT_COUNT)));
            }
            // stage2CoreUsed -1
            else if (curBlockIdx == stage2CoreUsed - 1) {
                reduce_dgamma_dbeta_WSP2GM(0, curBlockIdx * castEleNum, tailCastNum,
                                           divceil(divceil(N, SPLIT_COUNT), SPLIT_COUNT));
            }
            // 2 * stage2CoreUsed -1
            else if (curBlockIdx == SPLIT_COUNT * stage2CoreUsed - 1) {
                reduce_dgamma_dbeta_WSP2GM(divceil(divceil(N, SPLIT_COUNT), SPLIT_COUNT) * C,
                                           (curBlockIdx % stage2CoreUsed) * castEleNum, tailCastNum,
                                           (divceil(N, SPLIT_COUNT) - divceil(divceil(N, SPLIT_COUNT), SPLIT_COUNT)));
            }
        } else if constexpr (isDeterministic && !IsSameType<T, float>::value) {
#ifndef __CCE_KT_TEST__
            // wait core
            SyncAll();
#endif
            Initbuffer4stage2_mode2();
            // reduceSUM And Cast TO GM
            if (curBlockIdx < stage2CoreUsed - 1) {
                reduce_cast_dgamma_dbeta_WSP2GM(curBlockIdx * castEleNum, castEleNum, divceil(N, SPLIT_COUNT));
            } else if (curBlockIdx == stage2CoreUsed - 1) {
                reduce_cast_dgamma_dbeta_WSP2GM(curBlockIdx * castEleNum, tailCastNum, divceil(N, SPLIT_COUNT));
            }
        }
    }

private:
#include "group_norm_grad_kernel_part1.h"
#include "group_norm_grad_kernel_part2.h"
#include "group_norm_grad_kernel_part3.h"

#include "group_norm_grad_kernel_part4.h"
};
#endif // GROUP_NORM_GRAD_KERNEL_H
