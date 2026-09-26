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
 * \file npu_scatter_add_bwd.h
 * \brief NpuScatterAddBwd kernel 实现：按行计算 x_grad 与 s_grad
 */
#ifndef _NPU_SCATTER_ADD_BWD_H_
#define _NPU_SCATTER_ADD_BWD_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

using namespace AscendC;

namespace NpuScatterAddBwdKernel {

template <typename DataType>
class NpuScatterAddBwd {
public:
    __aicore__ NpuScatterAddBwd(const NpuScatterAddBwdTilingData& tiling, GM_ADDR yGrad, GM_ADDR x, GM_ADDR s,
                                GM_ADDR indices, GM_ADDR xGrad, GM_ADDR sGrad, GM_ADDR workspace)
        : tiling_(tiling)
    {
        // Init
        rowsPerCore_ = tiling.rowsPerCore;
        totalRows_ = tiling.totalRows;
        hiddenState_ = tiling.hiddenState;
        alignHiddenState_ = tiling.alignHiddenState;
        useCoreNum_ = tiling.usedCoreNum;

        // init global tensor
        yGradGm_.SetGlobalBuffer((__gm__ DataType*)(yGrad));
        xGm_.SetGlobalBuffer((__gm__ DataType*)(x));
        sGm_.SetGlobalBuffer((__gm__ DataType*)(s));
        indicesGm_.SetGlobalBuffer((__gm__ int*)(indices));
        xGradGm_.SetGlobalBuffer((__gm__ DataType*)xGrad);
        sGradGm_.SetGlobalBuffer((__gm__ DataType*)(sGrad));

        // init ub
        pipe_.InitBuffer(yGradBuf_, sizeof(DataType) * alignHiddenState_);
        pipe_.InitBuffer(xBuf_, sizeof(DataType) * alignHiddenState_);
        pipe_.InitBuffer(yGradFp32Buf_, sizeof(float) * alignHiddenState_);
        pipe_.InitBuffer(xFp32Buf_, sizeof(float) * alignHiddenState_);
        pipe_.InitBuffer(xGradFp32Buf_, sizeof(float) * alignHiddenState_);
        pipe_.InitBuffer(sGradFp32Buf_, sizeof(float) * alignHiddenState_);
        pipe_.InitBuffer(xGradBuf_, sizeof(DataType) * alignHiddenState_);
        pipe_.InitBuffer(sGradBuf_, sizeof(DataType) * alignHiddenState_);
        pipe_.InitBuffer(workBuf_, 32);
    }

    __aicore__ void ProcessTask(int startRow, int endRow)
    {
        // process [startRow, endRow)
        if (startRow >= endRow) {
            return;
        }

        for (int i = startRow; i < endRow; ++i) {
            if (i > startRow) {
                WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
            }

            const auto xIdx = i;
            const auto sIdx = xIdx;
            const auto yIdx = indicesGm_.GetValue(i);

            // copy y_grad & x to ub
            yGradLocal_ = yGradBuf_.Get<DataType>(alignHiddenState_);
            DataCopyPad(yGradLocal_, yGradGm_[yIdx * hiddenState_],
                        {static_cast<uint16_t>(1), static_cast<uint32_t>(hiddenState_ * sizeof(DataType)), 0, 0, 0},
                        {false, 0, 0, 0});
            xLocal_ = xBuf_.Get<DataType>(alignHiddenState_);
            DataCopyPad(xLocal_, xGm_[xIdx * hiddenState_],
                        {static_cast<uint16_t>(1), static_cast<uint32_t>(hiddenState_ * sizeof(DataType)), 0, 0, 0},
                        {false, 0, 0, 0});

            float sFp32 = 0.;
            if constexpr (std::is_same<DataType, bfloat16_t>::value) {
                sFp32 = ToFloat(sGm_[sIdx].GetValue(0));
            } else {
                sFp32 = static_cast<float>(sGm_[sIdx].GetValue(0));
            }

            WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
            SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);

            yGradFp32Local_ = yGradFp32Buf_.Get<float>(alignHiddenState_);
            xFp32Local_ = xFp32Buf_.Get<float>(alignHiddenState_);
            Cast(yGradFp32Local_, yGradLocal_, RoundMode::CAST_NONE, alignHiddenState_);
            Cast(xFp32Local_, xLocal_, RoundMode::CAST_NONE, alignHiddenState_);
            PipeBarrier<PIPE_V>();

            // compute s_grad_fp32 = sum(x * y_grad)
            Mul(xFp32Local_, xFp32Local_, yGradFp32Local_, alignHiddenState_);
            PipeBarrier<PIPE_V>();

            sGradFp32Local_ = sGradFp32Buf_.Get<float>(alignHiddenState_);
            workLocal_ = workBuf_.Get<float>(1);
            ReduceSum(sGradFp32Local_, xFp32Local_, workLocal_, hiddenState_);
            PipeBarrier<PIPE_V>();

            // compute x_grad_fp32 = y_grad * s
            xGradFp32Local_ = xGradFp32Buf_.Get<float>(alignHiddenState_);
            Muls(xGradFp32Local_, yGradFp32Local_, sFp32, hiddenState_);
            PipeBarrier<PIPE_V>();

            // cast
            xGradLocal_ = xGradBuf_.Get<DataType>(alignHiddenState_);
            sGradLocal_ = xGradFp32Buf_.Get<DataType>(alignHiddenState_);
            Cast(xGradLocal_, xGradFp32Local_, RoundMode::CAST_RINT, alignHiddenState_);
            Cast(sGradLocal_, sGradFp32Local_, RoundMode::CAST_RINT, 1);
            PipeBarrier<PIPE_V>();

            SetFlag<HardEvent::V_MTE3>(eventIdVToMte3);
            WaitFlag<HardEvent::V_MTE3>(eventIdVToMte3);

            // write to gm
            DataCopyPad(xGradGm_[xIdx * hiddenState_], xGradLocal_,
                        {static_cast<uint16_t>(1),                               // blockCount
                         static_cast<uint32_t>(hiddenState_ * sizeof(DataType)), // blockLen
                         static_cast<uint32_t>((alignHiddenState_ - hiddenState_) * sizeof(DataType) / 32), // srcStride
                         0,                                                                                 // dstStride
                         0});
            DataCopyPad(sGradGm_[sIdx], sGradLocal_,
                        {static_cast<uint16_t>(1), static_cast<uint32_t>(1 * sizeof(DataType)), 0, 0, 0});

            if (i < endRow - 1) {
                SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
            }
        }
    }

    __aicore__ void Process()
    {
        if ASCEND_IS_AIV {
            int startRow = GetBlockIdx() * rowsPerCore_;
            auto endRow = startRow + rowsPerCore_;
            if (endRow > totalRows_) {
                endRow = totalRows_;
            }
            ProcessTask(startRow, endRow);
        }
    }

private:
    TPipe pipe_;
    const NpuScatterAddBwdTilingData& tiling_;

    // gm tensor
    GlobalTensor<DataType> yGradGm_;
    GlobalTensor<DataType> xGm_;
    GlobalTensor<DataType> sGm_;
    GlobalTensor<int> indicesGm_;
    GlobalTensor<DataType> xGradGm_;
    GlobalTensor<DataType> sGradGm_;

    // ub tensor
    TBuf<> yGradBuf_;
    TBuf<> xBuf_;
    TBuf<> yGradFp32Buf_; // bf16/fp16 cast to fp32 buffer
    TBuf<> xFp32Buf_;
    TBuf<> xGradFp32Buf_;
    TBuf<> sGradFp32Buf_;
    TBuf<> xGradBuf_;
    TBuf<> sGradBuf_;
    TBuf<> workBuf_;

    // local tensor
    LocalTensor<DataType> yGradLocal_;
    LocalTensor<DataType> xLocal_;
    LocalTensor<float> yGradFp32Local_;
    LocalTensor<float> xFp32Local_;
    LocalTensor<float> xGradFp32Local_;
    LocalTensor<float> sGradFp32Local_;
    LocalTensor<DataType> xGradLocal_;
    LocalTensor<DataType> sGradLocal_;
    LocalTensor<float> workLocal_;

    uint32_t rowsPerCore_;
    uint32_t totalRows_;
    uint32_t hiddenState_;
    uint32_t alignHiddenState_;
    uint32_t useCoreNum_;

    // sync events
    event_t eventIdMte2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID<HardEvent::MTE2_S>());
    event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID<HardEvent::MTE2_V>());
    event_t eventIdVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID<HardEvent::V_MTE3>());
    event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID<HardEvent::MTE3_MTE2>());
    event_t eventIdSToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID<HardEvent::S_MTE3>());
    event_t eventIdMte3ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID<HardEvent::MTE3_S>());
};

} // namespace NpuScatterAddBwdKernel

#endif // _NPU_SCATTER_ADD_BWD_H_
