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
 * \file chamfer_distance_grad.h
 * \brief
 */
#ifndef CHAMFER_DISTANCE_GRAD_H
#define CHAMFER_DISTANCE_GRAD_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
using namespace AscendC;

template <typename T>
class ChamferDistanceGrad {
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> in_queue_xyz, in_queue_grad_dist, in_queue_id;
    TQue<QuePosition::VECOUT, 1> out_queue_d, out_queue_xy_zero_block, out_queue_block;
    GlobalTensor<T> xyz1_gm, xyz2_gm, grad_dist1_gm, grad_dist2_gm;
    GlobalTensor<int32_t> idx1_gm, idx2_gm;
    GlobalTensor<T> grad_xyz1_gm, grad_xyz2_gm;
    uint32_t num;
    uint32_t core_num;
    uint32_t cur_block_idx;
    uint32_t cur_core_task_num;
    uint32_t xyz_ele_num;
    uint32_t half_xyz_ele_num;
    uint32_t t_per_block;
    uint32_t per_ub_size;
    uint32_t start_task_id;

public:
    __aicore__ inline ChamferDistanceGrad(
        GM_ADDR xyz1, GM_ADDR xyz2, GM_ADDR grad_dist1, GM_ADDR grad_dist2, GM_ADDR idx1, GM_ADDR idx2,
        GM_ADDR grad_xyz1, GM_ADDR grad_xyz2, ChamferDistanceGradTilingData* tiling_data)
    {
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->core_num = GetBlockNum();
        this->cur_block_idx = GetBlockIdx();
        this->num = tiling_data->num;
        auto batch_size = tiling_data->batch_size;
        this->half_xyz_ele_num = this->num * batch_size;
        uint32_t doub = 2;
        this->xyz_ele_num = this->num * batch_size * doub;

        this->cur_core_task_num = tiling_data->task_per_core;
        this->start_task_id = tiling_data->task_per_core * this->cur_block_idx;
        if (this->cur_block_idx == tiling_data->core_used - 1) {
            this->cur_core_task_num = tiling_data->task_tail_core;
        }
        uint32_t block_bytes = 32;
        this->t_per_block = block_bytes / sizeof(T);
        xyz1_gm.SetGlobalBuffer((__gm__ T*)xyz1, this->xyz_ele_num);
        xyz2_gm.SetGlobalBuffer((__gm__ T*)xyz2, this->xyz_ele_num);
        grad_dist1_gm.SetGlobalBuffer((__gm__ T*)grad_dist1, this->half_xyz_ele_num);
        grad_dist2_gm.SetGlobalBuffer((__gm__ T*)grad_dist2, this->half_xyz_ele_num);
        idx1_gm.SetGlobalBuffer((__gm__ int32_t*)(idx1), this->half_xyz_ele_num);
        idx2_gm.SetGlobalBuffer((__gm__ int32_t*)(idx2), this->half_xyz_ele_num);
        grad_xyz1_gm.SetGlobalBuffer((__gm__ T*)grad_xyz1, this->xyz_ele_num);
        grad_xyz2_gm.SetGlobalBuffer((__gm__ T*)grad_xyz2, this->xyz_ele_num);
#ifndef __CCE_KT_TEST__
        if (this->cur_block_idx == 0) {
            InitOutput<T>(grad_xyz1_gm, this->num * batch_size * doub, 0);
            InitOutput<T>(grad_xyz2_gm, this->num * batch_size * doub, 0);
        }
        SyncAll();
#endif
        uint32_t ub_num = 6;
        uint32_t per_ub_size = tiling_data->ub_size / ub_num / sizeof(float) / block_bytes * block_bytes;
        this->per_ub_size = per_ub_size;
        pipe.InitBuffer(in_queue_xyz, 1, doub * per_ub_size * sizeof(T));
        pipe.InitBuffer(in_queue_grad_dist, 1, per_ub_size * sizeof(T));
        pipe.InitBuffer(in_queue_id, 1, per_ub_size * sizeof(int32_t));
        pipe.InitBuffer(out_queue_d, 1, doub * per_ub_size * sizeof(T));
        pipe.InitBuffer(out_queue_xy_zero_block, 1, this->t_per_block * sizeof(T));
        pipe.InitBuffer(out_queue_block, 1, this->t_per_block * sizeof(T));
    }

    __aicore__ inline void process()
    {
        main_fuc_of_mode_zero(this->start_task_id);
    }

private:
    __aicore__ inline void init_ub_mode_zero()
    {
        LocalTensor<T> xyz_local = in_queue_xyz.AllocTensor<T>();
        LocalTensor<T> grad_dist_local = in_queue_grad_dist.AllocTensor<T>();
        LocalTensor<int32_t> id_local = in_queue_id.AllocTensor<int32_t>();
        LocalTensor<T> d_local = out_queue_d.AllocTensor<T>();
        LocalTensor<T> xy_zero_block_local = out_queue_xy_zero_block.AllocTensor<T>();
        LocalTensor<T> block_local = out_queue_block.AllocTensor<T>();
        in_queue_xyz.EnQue(xyz_local);
        in_queue_grad_dist.EnQue(grad_dist_local);
        in_queue_id.EnQue(id_local);
        out_queue_d.EnQue(d_local);
        out_queue_xy_zero_block.EnQue(xy_zero_block_local);
        out_queue_block.EnQue(block_local);
    }

    template <AscendC::HardEvent hardEvent>
    __aicore__ inline void PipeSync()
    {
        int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(hardEvent));
        AscendC::SetFlag<hardEvent>(eventID);
        AscendC::WaitFlag<hardEvent>(eventID);
    }

    __aicore__ inline void compute_mode_zero_with_d2(
        int32_t task_idx, T db, float neg, T fill_value, const LocalTensor<T>& xyz_local,
        const LocalTensor<T>& grad_dist_local, const LocalTensor<int32_t>& id_local, const LocalTensor<T>& d_local,
        const LocalTensor<T>& xy_zero_block_local, const LocalTensor<T>& block_local, int32_t cur_task_idx,
        int32_t cur_task_batch, int32_t cur_task_num)
    {
        uint32_t doub = 2;
        if (this->cur_core_task_num > this->per_ub_size) {
            int32_t loop = this->cur_core_task_num / this->per_ub_size;
            for (uint32_t loop_ind = 0; loop_ind < loop; loop_ind++) {
                cur_task_idx = task_idx + loop_ind * this->per_ub_size;
                cur_task_batch = cur_task_idx / this->num;
                cur_task_num = cur_task_idx % this->num;
                main_cal_func_d2(
                    cur_task_batch, db, neg, xyz_local, grad_dist_local, id_local, d_local, xy_zero_block_local,
                    block_local, cur_task_num, this->per_ub_size, cur_task_idx);
                PipeSync<AscendC::HardEvent::V_MTE3>();
                PipeSync<AscendC::HardEvent::MTE2_MTE3>();
                SetAtomicAdd<T>();
                DataCopyPad(
                    grad_xyz2_gm[cur_task_batch * this->num * doub + cur_task_num * doub], d_local,
                    {1, static_cast<uint32_t>(this->per_ub_size * doub * sizeof(T)), 0, 0, 0});
                SetAtomicNone();
                PipeSync<AscendC::HardEvent::MTE3_S>();
            }
        }
        if (this->cur_core_task_num % this->per_ub_size) {
            int32_t num = this->cur_core_task_num % this->per_ub_size;
            int32_t loop_num = this->cur_core_task_num / this->per_ub_size;
            cur_task_idx = task_idx + loop_num * this->per_ub_size;
            cur_task_batch = cur_task_idx / this->num;
            cur_task_num = cur_task_idx % this->num;
            main_cal_func_d2(
                cur_task_batch, db, neg, xyz_local, grad_dist_local, id_local, d_local, xy_zero_block_local,
                block_local, cur_task_num, num, cur_task_idx);
            PipeSync<AscendC::HardEvent::V_S>();
            PipeSync<AscendC::HardEvent::MTE2_S>();
            PipeSync<AscendC::HardEvent::MTE3_S>();
            auto end = ceil(num * doub, this->t_per_block);
            PipeSync<AscendC::HardEvent::S_V>();
            if (end > num * doub) {
                for (auto ind = num * doub; ind < end; ind++) {
                    d_local.SetValue(ind, fill_value);
                }
            }
            PipeSync<AscendC::HardEvent::V_MTE3>();
            SetAtomicAdd<T>();
            DataCopyPad(
                grad_xyz2_gm[cur_task_batch * this->num * doub + cur_task_num * doub], d_local,
                {1, static_cast<uint32_t>(num * doub * sizeof(T)), 0, 0, 0});
            SetAtomicNone();
            PipeSync<AscendC::HardEvent::MTE3_S>();
        }
    }

    __aicore__ inline void main_cal_func_d2(
        int32_t cur_task_batch, T db, float neg, const LocalTensor<T>& xyz_local, const LocalTensor<T>& grad_dist_local,
        const LocalTensor<int32_t>& id_local, const LocalTensor<T>& d_local, const LocalTensor<T>& xy_zero_block_local,
        const LocalTensor<T>& block_local, int32_t cur_task_num, int32_t num, int32_t cur_task_idx)
    {
        uint32_t doub = 2;
        size_t fp32_type = 4;
        PipeSync<AscendC::HardEvent::S_MTE2>();
        DataCopyPad(
            xyz_local, xyz2_gm[cur_task_batch * this->num * doub + cur_task_num * doub],
            {1, static_cast<uint32_t>(num * doub * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
        PipeBarrier<PIPE_MTE2>();
        DataCopyPad(
            grad_dist_local, grad_dist2_gm[cur_task_batch * this->num + cur_task_num],
            {1, static_cast<uint32_t>(num * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
        PipeBarrier<PIPE_MTE2>();
        DataCopyPad(
            id_local, idx2_gm[cur_task_batch * this->num + cur_task_num],
            {1, static_cast<uint32_t>(num * sizeof(int32_t)), 0, 0, 0}, {false, 0, 0, 0});
        PipeSync<AscendC::HardEvent::MTE2_V>();
        muls_template(grad_dist_local, grad_dist_local, db, ceil(num, this->t_per_block));
        PipeSync<AscendC::HardEvent::V_S>();

        for (int32_t ind = 0; ind < num; ind++) {
            auto cur_batch = (cur_task_idx + ind) / this->num;
            auto cur_ind = id_local.GetValue(ind);
            PipeSync<AscendC::HardEvent::V_MTE2>();
            DataCopyPad(
                block_local, xyz1_gm[cur_batch * this->num * doub + cur_ind * doub],
                {1, static_cast<uint32_t>(doub * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
            PipeSync<AscendC::HardEvent::MTE2_V>();
            float x2 = (float)block_local.GetValue(0);
            float y2 = (float)block_local.GetValue(1);
            float x1 = (float)xyz_local.GetValue(doub * ind);
            float y1 = (float)xyz_local.GetValue(doub * ind + 1);
            float g2 = (float)grad_dist_local.GetValue(ind);
            float d1 = (x1 - x2) * g2;
            float d2 = (y1 - y2) * g2;
            PipeSync<AscendC::HardEvent::S_V>();
            if (sizeof(T) == fp32_type) {
                d_local.SetValue(doub * ind, d1);
                d_local.SetValue(doub * ind + 1, d2);
            } else {
                d_local.SetValue(doub * ind, (half)d1);
                d_local.SetValue(doub * ind + 1, (half)d2);
            }
            PipeSync<AscendC::HardEvent::V_S>();
            float neg_d1 = neg * d1;
            float neg_d2 = neg * d2;
            PipeSync<AscendC::HardEvent::S_V>();
            if (sizeof(T) == fp32_type) {
                xy_zero_block_local.SetValue(0, neg_d1);
                xy_zero_block_local.SetValue(1, neg_d2);
            } else {
                xy_zero_block_local.SetValue(0, (half)neg_d1);
                xy_zero_block_local.SetValue(1, (half)neg_d2);
            }
            PipeSync<AscendC::HardEvent::V_MTE3>();
            SetAtomicAdd<T>();
            DataCopyPad(
                grad_xyz1_gm[cur_batch * this->num * doub + cur_ind * doub], xy_zero_block_local,
                {1, static_cast<uint32_t>(doub * sizeof(T)), 0, 0, 0});
            SetAtomicNone();
            PipeSync<AscendC::HardEvent::MTE3_S>();
        }
    }

    __aicore__ inline void compute_mode_zero(int32_t task_idx)
    {
        LocalTensor<T> xyz_local = in_queue_xyz.DeQue<T>();
        LocalTensor<T> grad_dist_local = in_queue_grad_dist.DeQue<T>();
        LocalTensor<int32_t> id_local = in_queue_id.DeQue<int32_t>();
        LocalTensor<T> d_local = out_queue_d.DeQue<T>();
        LocalTensor<T> xy_zero_block_local = out_queue_xy_zero_block.DeQue<T>();
        LocalTensor<T> block_local = out_queue_block.DeQue<T>();
        uint32_t block_bytes = 32;
        uint32_t doub = 2;
        T db = 2.0;
        float neg = -1.0f;
        T fill_value = 0.0;
        PipeSync<AscendC::HardEvent::S_V>();
        Duplicate<T>(xy_zero_block_local, fill_value, this->t_per_block);
        PipeSync<AscendC::HardEvent::V_S>();
        int32_t cur_task_idx = 0;
        int32_t cur_task_batch = 0;
        int32_t cur_task_num = 0;
        if (this->cur_core_task_num > this->per_ub_size) {
            auto loop = this->cur_core_task_num / this->per_ub_size;
            for (uint32_t loop_ind = 0; loop_ind < loop; loop_ind++) {
                cur_task_idx = task_idx + loop_ind * this->per_ub_size;
                cur_task_batch = cur_task_idx / this->num;
                cur_task_num = cur_task_idx % this->num;
                main_cal_func_d1(
                    cur_task_batch, db, neg, xyz_local, grad_dist_local, id_local, d_local, xy_zero_block_local,
                    block_local, cur_task_num, this->per_ub_size, cur_task_idx);
                PipeSync<AscendC::HardEvent::MTE2_S>();
                PipeSync<AscendC::HardEvent::S_MTE3>();
                PipeSync<AscendC::HardEvent::V_MTE3>();
                SetAtomicAdd<T>();
                DataCopyPad(
                    grad_xyz1_gm[cur_task_batch * this->num * doub + cur_task_num * doub], d_local,
                    {1, static_cast<uint32_t>(this->per_ub_size * doub * sizeof(T)), 0, 0, 0});
                SetAtomicNone();
                PipeSync<AscendC::HardEvent::MTE3_S>();
            }
        }
        if (this->cur_core_task_num % this->per_ub_size) {
            auto num = this->cur_core_task_num % this->per_ub_size;
            auto loop_num = this->cur_core_task_num / this->per_ub_size;
            cur_task_idx = task_idx + loop_num * this->per_ub_size;
            cur_task_batch = cur_task_idx / this->num;
            cur_task_num = cur_task_idx % this->num;
            main_cal_func_d1(
                cur_task_batch, db, neg, xyz_local, grad_dist_local, id_local, d_local, xy_zero_block_local,
                block_local, cur_task_num, num, cur_task_idx);
            PipeSync<AscendC::HardEvent::MTE2_S>();
            PipeSync<AscendC::HardEvent::MTE3_S>();
            PipeSync<AscendC::HardEvent::V_S>();
            auto end = ceil(num * doub, this->t_per_block);
            PipeSync<AscendC::HardEvent::S_V>();
            if (end > num * doub) {
                for (auto ind = num * doub; ind < end; ind++) {
                    d_local.SetValue(ind, fill_value);
                }
            }
            PipeSync<AscendC::HardEvent::V_MTE3>();
            SetAtomicAdd<T>();
            DataCopyPad(
                grad_xyz1_gm[cur_task_batch * this->num * doub + cur_task_num * doub], d_local,
                {1, static_cast<uint32_t>(num * doub * sizeof(T)), 0, 0, 0});
            SetAtomicNone();
            PipeSync<AscendC::HardEvent::MTE3_S>();
        }
        compute_mode_zero_with_d2(
            task_idx, db, neg, fill_value, xyz_local, grad_dist_local, id_local, d_local, xy_zero_block_local,
            block_local, cur_task_idx, cur_task_batch, cur_task_num);
        PipeSync<AscendC::HardEvent::V_S>();
        PipeSync<AscendC::HardEvent::MTE2_S>();
        PipeSync<AscendC::HardEvent::MTE3_S>();
        out_queue_d.FreeTensor(d_local);
        out_queue_xy_zero_block.FreeTensor(xy_zero_block_local);
        out_queue_block.FreeTensor(block_local);
        in_queue_grad_dist.FreeTensor(grad_dist_local);
        in_queue_xyz.FreeTensor(xyz_local);
        in_queue_id.FreeTensor(id_local);
    }

    __aicore__ inline void main_cal_func_d1(
        int32_t cur_task_batch, T db, float neg, const LocalTensor<T>& xyz_local, const LocalTensor<T>& grad_dist_local,
        const LocalTensor<int32_t>& id_local, const LocalTensor<T>& d_local, const LocalTensor<T>& xy_zero_block_local,
        const LocalTensor<T>& block_local, int32_t cur_task_num, int32_t num, int32_t cur_task_idx)
    {
        uint32_t doub = 2;
        size_t fp32_type = 4;
        PipeSync<AscendC::HardEvent::S_MTE2>();
        DataCopyPad(
            xyz_local, xyz1_gm[cur_task_batch * this->num * doub + cur_task_num * doub],
            {1, static_cast<uint32_t>(num * doub * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
        PipeBarrier<PIPE_MTE2>();
        DataCopyPad(
            grad_dist_local, grad_dist1_gm[cur_task_batch * this->num + cur_task_num],
            {1, static_cast<uint32_t>(num * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
        PipeBarrier<PIPE_MTE2>();
        DataCopyPad(
            id_local, idx1_gm[cur_task_batch * this->num + cur_task_num],
            {1, static_cast<uint32_t>(num * sizeof(int32_t)), 0, 0, 0}, {false, 0, 0, 0});
        PipeSync<AscendC::HardEvent::MTE2_V>();
        muls_template(grad_dist_local, grad_dist_local, db, ceil(num, this->t_per_block));
        PipeSync<AscendC::HardEvent::V_S>();

        for (int32_t ind = 0; ind < num; ind++) {
            int32_t cur_batch = (cur_task_idx + ind) / this->num;
            int32_t cur_ind = id_local.GetValue(ind);
            PipeSync<AscendC::HardEvent::V_MTE2>();
            PipeSync<AscendC::HardEvent::MTE3_MTE2>();
            DataCopyPad(
                block_local, xyz2_gm[cur_batch * this->num * doub + cur_ind * doub],
                {1, static_cast<uint32_t>(doub * sizeof(T)), 0, 0, 0}, {false, 0, 0, 0});
            PipeBarrier<PIPE_MTE2>();
            float x2 = (float)block_local.GetValue(0);
            float y2 = (float)block_local.GetValue(1);
            float x1 = (float)xyz_local.GetValue(doub * ind);
            float y1 = (float)xyz_local.GetValue(doub * ind + 1);
            float g1 = (float)grad_dist_local.GetValue(ind);
            float d1 = (x1 - x2) * g1;
            float d2 = (y1 - y2) * g1;
            PipeSync<AscendC::HardEvent::S_V>();
            if (sizeof(T) == fp32_type) {
                d_local.SetValue(doub * ind, d1);
                d_local.SetValue(doub * ind + 1, d2);
            } else {
                d_local.SetValue(doub * ind, (half)d1);
                d_local.SetValue(doub * ind + 1, (half)d2);
            }
            PipeSync<AscendC::HardEvent::V_S>();
            float neg_d1 = neg * d1;
            float neg_d2 = neg * d2;
            PipeSync<AscendC::HardEvent::S_V>();
            if (sizeof(T) == fp32_type) {
                xy_zero_block_local.SetValue(0, neg_d1);
                xy_zero_block_local.SetValue(1, neg_d2);
            } else {
                xy_zero_block_local.SetValue(0, (half)neg_d1);
                xy_zero_block_local.SetValue(1, (half)neg_d2);
            }
            PipeSync<AscendC::HardEvent::V_MTE3>();
            SetAtomicAdd<T>();
            DataCopyPad(
                grad_xyz2_gm[cur_batch * this->num * doub + cur_ind * doub], xy_zero_block_local,
                {1, static_cast<uint32_t>(doub * sizeof(T)), 0, 0, 0});
            SetAtomicNone();
            PipeBarrier<PIPE_MTE2>();
            PipeBarrier<PIPE_MTE3>();
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline uint32_t ceil(uint32_t a, uint32_t b)
    {
        if (b == 0) {
            return 0;
        }
        return ((a - 1) / b + 1) * b;
    }

    __aicore__ inline void muls_template(
        const LocalTensor<T>& dstLocal, const LocalTensor<T>& srcLocal, T scalarValue, const int32_t calCount)
    {
        int32_t unit = 256;
        int32_t max_repeat = 255;
        int32_t mask = unit / sizeof(T);
        int32_t repeats = calCount / mask;
        int32_t loop = repeats / max_repeat;
        int32_t repeats_tail = repeats % max_repeat;
        int32_t tail = calCount % mask;
        int32_t tensor_offset = 0;
        PipeBarrier<PIPE_V>();
        for (int32_t loop_idx = 0; loop_idx < loop; loop_idx++) {
            Muls(
                dstLocal[loop_idx * max_repeat * mask], srcLocal[loop_idx * max_repeat * mask], scalarValue, mask,
                max_repeat, {1, 1, 8, 8});
            PipeBarrier<PIPE_V>();
        }
        tensor_offset = loop * max_repeat * mask;
        if (repeats_tail >= 1) {
            Muls(dstLocal[tensor_offset], srcLocal[tensor_offset], scalarValue, mask, repeats_tail, {1, 1, 8, 8});
            PipeBarrier<PIPE_V>();
        }
        tensor_offset += repeats_tail * mask;
        if (tail >= 1) {
            Muls(dstLocal[tensor_offset], srcLocal[tensor_offset], scalarValue, tail);
            PipeBarrier<PIPE_V>();
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void main_fuc_of_mode_zero(int32_t task_idx)
    {
        init_ub_mode_zero();
        compute_mode_zero(task_idx);
    }
};
#endif // CHAMFER_DISTANCE_GRAD