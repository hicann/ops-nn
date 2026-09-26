/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file npu_scatter_add.h
 * \brief NpuScatterAdd kernel 实现：按 sort_idx 排序遍历，workspace 缓存同一目标行的累加中间结果
 */
#ifndef _NPU_SCATTER_ADD_H_
#define _NPU_SCATTER_ADD_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

using namespace AscendC;

namespace NpuScatterAddKernel {

template <typename DataType, bool WithScale = true, bool UseHighPrecision = false>
class NpuScatterAdd {
public:
    __aicore__ NpuScatterAdd(const NpuScatterAddTilingData& tiling, GM_ADDR x, GM_ADDR y, GM_ADDR s, GM_ADDR indices,
                             GM_ADDR sortIdx, GM_ADDR validTokenNum, GM_ADDR workspace)
        : tiling_(tiling)
    {
        // Init
        total_rows_ = tiling.totalRows;
        hidden_state_ = tiling.hiddenState;
        align_hidden_state_ = tiling.alignHiddenState;
        use_core_num_ = tiling.usedCoreNum;
        reduce_row_id_ = -1;

        if (tiling_.withValid != 0) {
            AscendC::GlobalTensor<int> validNumTensor;
            validNumTensor.SetGlobalBuffer((__gm__ int*)(validTokenNum));
            total_rows_ = validNumTensor.GetValue(0);
        }
        rows_per_core_ = (total_rows_ + use_core_num_ - 1) / use_core_num_;

        // init global tensor
        x_gm_.SetGlobalBuffer((__gm__ DataType*)(x));
        y_gm_.SetGlobalBuffer((__gm__ DataType*)(y));
        if constexpr (WithScale) {
            s_gm_.SetGlobalBuffer((__gm__ DataType*)(s));
        }
        indices_gm_.SetGlobalBuffer((__gm__ int*)(indices));
        sort_idx_gm_.SetGlobalBuffer((__gm__ int*)sortIdx);

        const uint64_t reduceOffset = static_cast<uint64_t>(GetBlockIdx()) * sizeof(DataType) * align_hidden_state_;
        reduce_gm_.SetGlobalBuffer((__gm__ DataType*)(workspace + reduceOffset));
        token_idx_gm_.SetGlobalBuffer(
            (__gm__ int*)(workspace + sizeof(DataType) * align_hidden_state_ * use_core_num_));

        // init ub
        pipe_.InitBuffer(x_buf_, sizeof(DataType) * align_hidden_state_);
        pipe_.InitBuffer(y_buf_, sizeof(DataType) * align_hidden_state_);
        pipe_.InitBuffer(reduce_buf_, sizeof(DataType) * align_hidden_state_);
        pipe_.InitBuffer(x_fp32_buf_, sizeof(float) * align_hidden_state_);
        pipe_.InitBuffer(y_fp32_buf_, sizeof(float) * align_hidden_state_);
        pipe_.InitBuffer(reduce_fp32_buf_, sizeof(float) * align_hidden_state_);

        pipe_.InitBuffer(reduce_row_id_buf_, 32); // NPU至少32B
    }

    __aicore__ void ProcessTask(int startRow, int endRow)
    {
        // process [startRow, endRow]
        if (startRow > endRow) {
            return;
        }

        bool hasIdxChange = false;
        reduce_row_id_ = indices_gm_.GetValue(sort_idx_gm_.GetValue(endRow));

        for (int i = endRow; i >= startRow; --i) {
            if (i < endRow) {
                WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
            }
            const auto srcIdx = sort_idx_gm_.GetValue(i);
            const auto dstIdx = indices_gm_.GetValue(srcIdx);

            if (dstIdx != reduce_row_id_) {
                // block间隔部分处理结束
                hasIdxChange = true;
            }

            // copy x & y & scale
            x_local_ = x_buf_.Get<DataType>(align_hidden_state_);
            y_local_ = y_buf_.Get<DataType>(align_hidden_state_);

            DataCopyPad(x_local_, x_gm_[srcIdx * hidden_state_],
                        {static_cast<uint16_t>(1), static_cast<uint32_t>(hidden_state_ * sizeof(DataType)), 0, 0, 0},
                        {false, 0, 0, 0});

            if (i == endRow) {
            } else if (!hasIdxChange) {
                DataCopyPad(
                    y_local_, reduce_gm_,
                    {static_cast<uint16_t>(1), static_cast<uint32_t>(hidden_state_ * sizeof(DataType)), 0, 0, 0},
                    {true, 0, static_cast<uint8_t>(align_hidden_state_ - hidden_state_), static_cast<DataType>(0)});
            } else {
                // 一般的case
                DataCopyPad(
                    y_local_, y_gm_[dstIdx * hidden_state_],
                    {static_cast<uint16_t>(1), static_cast<uint32_t>(hidden_state_ * sizeof(DataType)), 0, 0, 0},
                    {true, 0, static_cast<uint8_t>(align_hidden_state_ - hidden_state_), static_cast<DataType>(0)});
            }

            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

            x_fp32_local_ = x_fp32_buf_.Get<float>(align_hidden_state_);
            y_fp32_local_ = y_fp32_buf_.Get<float>(align_hidden_state_);
            Cast(x_fp32_local_, x_local_, RoundMode::CAST_NONE, align_hidden_state_);
            Cast(y_fp32_local_, y_local_, RoundMode::CAST_NONE, align_hidden_state_);
            if (i == endRow) {
                // 第一次对于y还是手动zero化
                Duplicate(y_fp32_local_, static_cast<float>(0), align_hidden_state_);
            }
            PipeBarrier<PIPE_V>();

            float scaleFp32 = 0.;
            if constexpr (WithScale) {
                if constexpr (std::is_same<DataType, bfloat16_t>::value) {
                    scaleFp32 = ToFloat(s_gm_[srcIdx].GetValue(0));
                } else {
                    scaleFp32 = static_cast<float>(s_gm_[srcIdx].GetValue(0));
                }
                Muls(x_fp32_local_, x_fp32_local_, scaleFp32, align_hidden_state_);
                PipeBarrier<PIPE_V>();
                if constexpr (!UseHighPrecision) {
                    Cast(x_local_, x_fp32_local_, RoundMode::CAST_RINT, align_hidden_state_);
                    PipeBarrier<PIPE_V>();
                    Cast(x_fp32_local_, x_local_, RoundMode::CAST_NONE, align_hidden_state_);
                    PipeBarrier<PIPE_V>();
                }
            }

            Add(y_fp32_local_, x_fp32_local_, y_fp32_local_, align_hidden_state_);
            PipeBarrier<PIPE_V>();

            // write to final result
            Cast(y_local_, y_fp32_local_, RoundMode::CAST_RINT, align_hidden_state_);
            PipeBarrier<PIPE_V>();

            SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

            if (hasIdxChange) {
                DataCopyPad(
                    y_gm_[dstIdx * hidden_state_], y_local_,
                    {static_cast<uint16_t>(1),                                                             // blockCount
                     static_cast<uint32_t>(hidden_state_ * sizeof(DataType)),                              // blockLen
                     static_cast<uint32_t>((align_hidden_state_ - hidden_state_) * sizeof(DataType) / 32), // srcStride
                     0,                                                                                    // dstStride
                     0});
            } else {
                // write to reduce workspace
                DataCopyPad(
                    reduce_gm_, y_local_,
                    {static_cast<uint16_t>(1),                                                             // blockCount
                     static_cast<uint32_t>(hidden_state_ * sizeof(DataType)),                              // blockLen
                     static_cast<uint32_t>((align_hidden_state_ - hidden_state_) * sizeof(DataType) / 32), // srcStride
                     0,                                                                                    // dstStride
                     0});
            }

            if (i >= startRow + 1) {
                SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
            }
        }
    }

    __aicore__ void Process()
    {
        if ASCEND_IS_AIV {
            int startRow = GetBlockIdx() * rows_per_core_;

            auto endRow = startRow + rows_per_core_;
            if (endRow > total_rows_) {
                endRow = total_rows_;
            }
            endRow = endRow - 1;

            ProcessTask(startRow, endRow);

            // 将reduce_row_id_写入 tmp workspace
            reduce_row_id_local_ = reduce_row_id_buf_.Get<int>(1);
            reduce_row_id_local_.SetValue(0, reduce_row_id_);
            WaitFlag<HardEvent::S_MTE3>(eventIdSToMte3);
            SetFlag<HardEvent::S_MTE3>(eventIdSToMte3);
            DataCopyPad(token_idx_gm_[GetBlockIdx()], reduce_row_id_local_,
                        {static_cast<uint16_t>(1), static_cast<uint32_t>(1 * sizeof(int)), 0,
                         static_cast<uint32_t>(32 - sizeof(int)), 0});

            // sync all aiv core
            SyncAll();
        }

        // reduce final result
        if ASCEND_IS_AIV {
            if (GetBlockIdx() != 0) {
                return;
            }

            SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);

            for (int i = 0; i < use_core_num_; ++i) {
                const auto dstIdx = token_idx_gm_.GetValue(i);
                if (dstIdx == -1) {
                    continue;
                }

                // wait last times finish
                WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
                // copy reduce to final result
                y_local_ = y_buf_.Get<DataType>(align_hidden_state_);
                reduce_local_ = reduce_buf_.Get<DataType>(align_hidden_state_);
                DataCopyPad(
                    y_local_, y_gm_[dstIdx * hidden_state_],
                    {static_cast<uint16_t>(1), static_cast<uint32_t>(hidden_state_ * sizeof(DataType)), 0, 0, 0},
                    {false, 0, 0, 0});
                const auto reduceGmOffset = i * align_hidden_state_;
                DataCopyPad(
                    reduce_local_, reduce_gm_[reduceGmOffset],
                    {static_cast<uint16_t>(1), static_cast<uint32_t>(hidden_state_ * sizeof(DataType)), 0, 0, 0},
                    {false, 0, 0, 0});
                SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
                WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

                // cast to fp32
                x_fp32_local_ = x_fp32_buf_.Get<float>(align_hidden_state_);
                y_fp32_local_ = y_fp32_buf_.Get<float>(align_hidden_state_);

                Cast(x_fp32_local_, reduce_local_, RoundMode::CAST_NONE, align_hidden_state_);
                Cast(y_fp32_local_, y_local_, RoundMode::CAST_NONE, align_hidden_state_);
                PipeBarrier<PIPE_V>();

                Add(y_fp32_local_, x_fp32_local_, y_fp32_local_, align_hidden_state_);
                PipeBarrier<PIPE_V>();
                Cast(y_local_, y_fp32_local_, RoundMode::CAST_RINT, align_hidden_state_);
                PipeBarrier<PIPE_V>();

                SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
                WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);
                // copy to gm
                DataCopyPad(
                    y_gm_[dstIdx * hidden_state_], y_local_,
                    {static_cast<uint16_t>(1), static_cast<uint32_t>(hidden_state_ * sizeof(DataType)),
                     static_cast<uint32_t>((align_hidden_state_ - hidden_state_) * sizeof(DataType) / 32), 0, 0});
                SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
            }

            WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        }
    }

private:
    TPipe pipe_;
    const NpuScatterAddTilingData& tiling_;

    // gm tensor
    GlobalTensor<DataType> x_gm_;
    GlobalTensor<DataType> y_gm_;
    GlobalTensor<DataType> s_gm_;
    GlobalTensor<int> indices_gm_;
    GlobalTensor<int> sort_idx_gm_;

    GlobalTensor<DataType> reduce_gm_;
    GlobalTensor<int> token_idx_gm_;

    // ub tensor
    TBuf<> x_buf_;
    TBuf<> y_buf_;
    TBuf<> reduce_buf_;
    TBuf<> x_fp32_buf_; // bf16/fp16 cast to fp32 buffer
    TBuf<> y_fp32_buf_;
    TBuf<> reduce_fp32_buf_;
    TBuf<> reduce_row_id_buf_;

    // local tensor
    LocalTensor<DataType> x_local_;
    LocalTensor<DataType> y_local_;
    LocalTensor<DataType> reduce_local_;
    LocalTensor<float> x_fp32_local_;
    LocalTensor<float> y_fp32_local_;
    LocalTensor<int> reduce_row_id_local_;

    uint32_t rows_per_core_;
    uint32_t total_rows_;
    uint32_t hidden_state_;
    uint32_t align_hidden_state_;
    uint32_t use_core_num_;

    int reduce_row_id_;

    // sync events
    event_t eventIdMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID<HardEvent::MTE2_S>());
    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID<HardEvent::MTE2_V>());
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID<HardEvent::V_MTE3>());
    event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID<HardEvent::MTE3_MTE2>());
    event_t eventIdSToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID<HardEvent::S_MTE3>());
    event_t eventIdMte3ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID<HardEvent::MTE3_S>());
};

} // namespace NpuScatterAddKernel

#endif // _NPU_SCATTER_ADD_H_
