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
 * \file layer_norm_quant_normal.h
 * \brief
 */
#include "kernel_operator.h"
#include "layer_norm_quant_helper.h"

using AscendC::HardEvent;

template <typename T>
class KernelLayerNormQuant {
public:
    __aicore__ inline KernelLayerNormQuant() {}

    __aicore__ inline uint32_t ROUND_UP_FUNC(uint32_t x) { return (x + BLOCK_NUM - 1) / BLOCK_NUM * BLOCK_NUM; }

    __aicore__ inline void Init(__gm__ uint8_t *x, __gm__ uint8_t *gamma, __gm__ uint8_t *beta, __gm__ uint8_t *scale,
                                __gm__ uint8_t *offset, __gm__ uint8_t *z, LayerNormQuantTilingData tilingData)
    {
        num_core = tilingData.numCore;
        num_last_dim = tilingData.numLastDim;
        colsAligned = tilingData.colsAligned;
        nl_first_dim_per_core = tilingData.nlFirstdimPerCore;
        l_first_dim_per_core = tilingData.lFirstdimPerCore;
        first_dim_per_times = tilingData.firstDimPerTimes;
        eps = tilingData.epsStr;
        aveNum = tilingData.aveStr;
        slice_size = tilingData.sliceSize;
        half_num = 2;

        if (AscendC::GetBlockIdx() != num_core - 1) {
            row_work = nl_first_dim_per_core;
            row_step = first_dim_per_times;
        } else {
            row_work = l_first_dim_per_core;
            row_step = MIN(first_dim_per_times, row_work);
        }
        
        copyLenAlign = BLOCK_ALIGN(num_last_dim * sizeof(T), 32) / sizeof(T);  // 向上对齐
        row_tail_ = (row_work % first_dim_per_times == 0) ? first_dim_per_times : (row_work % first_dim_per_times);
        gm_offset_ = static_cast<uint64_t>(nl_first_dim_per_core) * num_last_dim;

        x_gm.SetGlobalBuffer((__gm__ T *)x + AscendC::GetBlockIdx() * gm_offset_);
        z_gm.SetGlobalBuffer((__gm__ int8_t *)z + AscendC::GetBlockIdx() * gm_offset_);
        beta_gm.SetGlobalBuffer((__gm__ T *)beta);
        gamma_gm.SetGlobalBuffer((__gm__ T *)gamma);

#if __CCE_AICORE__ == 220
#else
        pipe.InitBuffer(tensor_buf, 32);
#endif
        pipe.InitBuffer(x_que, BUFFER_NUM, row_step * ROUND_UP_FUNC(slice_size) * DATA_BYTE);
        pipe.InitBuffer(z_que, BUFFER_NUM, row_step * ROUND_UP_FUNC(slice_size) * INT8_DATA_BYTE);
        pipe.InitBuffer(beta_que, BUFFER_NUM, 1 * ROUND_UP_FUNC(slice_size) * DATA_BYTE);
        pipe.InitBuffer(gamma_que, BUFFER_NUM, 1 * ROUND_UP_FUNC(slice_size) * DATA_BYTE);
        pipe.InitBuffer(var_buf, BLOCK_NUM * DATA_BYTE);
        pipe.InitBuffer(ave_buf, BLOCK_NUM * DATA_BYTE);
        pipe.InitBuffer(x_buf_fp32, half_num * ROUND_UP_FUNC(slice_size) * DATA_BYTE);
        pipe.InitBuffer(y_buf_fp32, half_num * ROUND_UP_FUNC(slice_size) * DATA_BYTE);
        pipe.InitBuffer(z_buf_fp32, half_num * ROUND_UP_FUNC(slice_size) * DATA_BYTE);

        GetScaleAndOffset(scale, offset);
    }

    __aicore__ inline void Process()
    {
        uint32_t move_cnt = CEIL_DIV(row_work, row_step);
        for (uint32_t i = 0; i < move_cnt; ++i) {
            if (i < move_cnt - 1) {
                FastCopyIn(i, row_step);
                FastCopyInGammaBeta();

                AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                FastPrecisionCompute(row_step);
                AscendC::SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                AscendC::WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);

                FastCopyOut(i, row_step);
            } else {
                FastCopyIn(i, row_tail_);
                FastCopyInGammaBeta();

                AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                FastPrecisionCompute(row_tail_);
                AscendC::SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                AscendC::WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);

                FastCopyOut(i, row_tail_);
            }
        }
    }

private:
    __aicore__ inline uint32_t BLOCK_ALIGN(uint32_t a, uint32_t blockSize)
    {
        return (blockSize > 0) ? (a + blockSize - 1) / blockSize * blockSize : 0;
    }

    __aicore__ inline void FastCopyIn(uint64_t proc_id, uint64_t size)
    {
        AscendC::LocalTensor<T> x_local = x_que.AllocTensor<T>();
        uint64_t offset = proc_id * row_step * num_last_dim;

        DataCopyPadExtParams<T> padParams;
        padParams.isPad = true;
        padParams.paddingValue = static_cast<T>(0.0);
        padParams.rightPadding = copyLenAlign - num_last_dim;

#if __CCE_AICORE__ == 220
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = size;  // 搬多少块
        dataCopyParams.blockLen = num_last_dim * sizeof(T);  // 搬多长
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;

        DataCopyPad(x_local, x_gm[offset], dataCopyParams, padParams);
#else
        AscendC::LocalTensor<T> tensor_local = tensor_buf.Get<T>();
        for (auto i = 0; i < size; i++) {
            DataCopyEx(x_local[i * copyLenAlign], x_gm[offset + i * num_last_dim], tensor_local, num_last_dim, 1);
        }
#endif
        x_que.EnQue(x_local);
    }

    __aicore__ inline void FastCopyInGammaBeta(){
        AscendC::LocalTensor<T> beta_local = beta_que.AllocTensor<T>();
        AscendC::LocalTensor<T> gamma_local = gamma_que.AllocTensor<T>();

        DataCopyPadExtParams<T> padParams;
        padParams.isPad = true;
        padParams.paddingValue = static_cast<T>(0.0);
        padParams.rightPadding = copyLenAlign - num_last_dim;
#if __CCE_AICORE__ == 220
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = num_last_dim * sizeof(T);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;

        DataCopyPad(beta_local, beta_gm[0], dataCopyParams, padParams);
        DataCopyPad(gamma_local, gamma_gm[0], dataCopyParams, padParams);
#else
        AscendC::LocalTensor<T> tensor_local = tensor_buf.Get<T>();
        DataCopyEx(beta_local, beta_gm[0], tensor_local, num_last_dim, 1);
        DataCopyEx(gamma_local, gamma_gm[0], tensor_local, num_last_dim, 1);
#endif
        beta_que.EnQue(beta_local);
        gamma_que.EnQue(gamma_local);
    }

    __aicore__ inline float ComputeMeanValue(LocalTensor<float>& x_local_fp32)
    {
        AscendC::LocalTensor<float> y_local_fp32 = y_buf_fp32.Get<float>();
        AscendC::LocalTensor<float> z_fp32_local = z_buf_fp32.Get<float>();
        AscendC::LocalTensor<float> ave_local = ave_buf.Get<float>();

        Duplicate(y_local_fp32, aveNum, num_last_dim);
        AscendC::PipeBarrier<PIPE_V>();
        Mul(z_fp32_local, x_local_fp32, y_local_fp32, num_last_dim);
        AscendC::PipeBarrier<PIPE_V>();
        ReduceSum(ave_local, z_fp32_local, y_local_fp32, num_last_dim);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::V_S>(EVENT_ID0);
        float ave_local_temp = ave_local.GetValue(0);
        return ave_local_temp;
    }

    __aicore__ inline float ComputeVarValue(LocalTensor<float>& x_local_fp32, LocalTensor<float>& y_local_fp32, float ave_local_temp)
    {
        AscendC::LocalTensor<float> var_local = var_buf.Get<float>();
        AscendC::LocalTensor<float> z_fp32_local = z_buf_fp32.Get<float>();
        Duplicate(y_local_fp32, ave_local_temp, num_last_dim);
        AscendC::PipeBarrier<PIPE_V>();
        Sub(z_fp32_local, x_local_fp32, y_local_fp32, num_last_dim);
        AscendC::PipeBarrier<PIPE_V>();
        Mul(x_local_fp32, z_fp32_local, z_fp32_local, num_last_dim);
        AscendC::PipeBarrier<PIPE_V>();
        Muls(y_local_fp32, x_local_fp32, aveNum, num_last_dim);
        AscendC::PipeBarrier<PIPE_V>();
        ReduceSum(var_local, y_local_fp32, x_local_fp32, num_last_dim);
        AscendC::PipeBarrier<PIPE_V>();
        Adds(var_local, var_local, eps, BLOCK_NUMEL / half_num);
        AscendC::PipeBarrier<PIPE_V>();
        Sqrt(var_local, var_local, BLOCK_NUMEL / half_num);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<HardEvent::V_S>(EVENT_ID2);
        AscendC::WaitFlag<HardEvent::V_S>(EVENT_ID2);
        float var_local_temp = var_local.GetValue(0);
        return var_local_temp;
    }

    __aicore__ inline void FastPrecisionCompute(uint64_t nums)
    {
        AscendC::LocalTensor<T> x_local = x_que.DeQue<T>();
        AscendC::LocalTensor<T> gamma_local = gamma_que.DeQue<T>();
        AscendC::LocalTensor<T> beta_local = beta_que.DeQue<T>();
        AscendC::LocalTensor<int8_t> z_local = z_que.AllocTensor<int8_t>();
        AscendC::LocalTensor<float> x_local_fp32 = x_buf_fp32.Get<float>();
        AscendC::LocalTensor<float> z_fp32_local = z_buf_fp32.Get<float>();
        AscendC::LocalTensor<float> y_local_fp32 = y_buf_fp32.Get<float>();
        for (uint64_t rid = 0; rid < nums; ++rid) {
            Cast(x_local_fp32, x_local[rid * copyLenAlign], AscendC::RoundMode::CAST_NONE, num_last_dim);
            AscendC::PipeBarrier<PIPE_V>();
            float ave_local_temp = ComputeMeanValue(x_local_fp32);
            AscendC::SetFlag<HardEvent::S_V>(EVENT_ID0);
            AscendC::WaitFlag<HardEvent::S_V>(EVENT_ID0);
            AscendC::PipeBarrier<PIPE_V>();
            float var_local_temp = ComputeVarValue(x_local_fp32, y_local_fp32, ave_local_temp);
            AscendC::SetFlag<HardEvent::S_V>(EVENT_ID2);
            AscendC::WaitFlag<HardEvent::S_V>(EVENT_ID2);
            AscendC::PipeBarrier<PIPE_V>();
            Duplicate(x_local_fp32, var_local_temp, num_last_dim);
            Duplicate(y_local_fp32, (float)1, num_last_dim);
            AscendC::PipeBarrier<PIPE_V>();
            Div(y_local_fp32, y_local_fp32, x_local_fp32, num_last_dim);
            AscendC::PipeBarrier<PIPE_V>();
            Mul(x_local_fp32, z_fp32_local, y_local_fp32, num_last_dim);
            Cast(z_fp32_local, gamma_local, AscendC::RoundMode::CAST_NONE, num_last_dim);
            AscendC::PipeBarrier<PIPE_V>();
            Mul(y_local_fp32, x_local_fp32, z_fp32_local, num_last_dim);
            Cast(x_local_fp32, beta_local, AscendC::RoundMode::CAST_NONE, num_last_dim);
            AscendC::PipeBarrier<PIPE_V>();
            Add(z_fp32_local, y_local_fp32, x_local_fp32, num_last_dim);
            AscendC::PipeBarrier<PIPE_V>();
            Muls(z_fp32_local, z_fp32_local, inputScale, num_last_dim);
            AscendC::PipeBarrier<PIPE_V>();
            Adds(z_fp32_local, z_fp32_local, inputOffset, num_last_dim);
            AscendC::PipeBarrier<PIPE_V>();
            AscendC::LocalTensor<half> tmpFp16 = x_buf_fp32.Get<half>();
            CastFrom32To16(tmpFp16, z_fp32_local, num_last_dim);
            AscendC::PipeBarrier<PIPE_V>();
            CastFromF16ToI8(z_local[rid * colsAligned], tmpFp16, QUANT_MIN, num_last_dim);
            AscendC::PipeBarrier<PIPE_V>();
        }
        x_que.FreeTensor(x_local);
        beta_que.FreeTensor(beta_local);
        gamma_que.FreeTensor(gamma_local);
        z_que.EnQue(z_local);
    }

    __aicore__ inline void FastCopyOut(uint64_t proc_id, uint64_t size)
    {
        AscendC::LocalTensor<int8_t> z = z_que.DeQue<int8_t>();
        uint64_t offset = proc_id * row_step * num_last_dim;

#if __CCE_AICORE__ == 220
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = size;  // 搬多少块
        dataCopyParams.blockLen = num_last_dim * 1;  // 搬多长
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;

        DataCopyPad(z_gm[offset], z, dataCopyParams);
#else
        AscendC::LocalTensor<int8_t> tensor_local = tensor_buf.Get<int8_t>();
        for (auto i = 0; i < size; i++) {
            DataCopyEx(z_gm[offset + i * num_last_dim], z[i * colsAligned], tensor_local, num_last_dim, 1);
        }
#endif
        z_que.FreeTensor(z);
    }

    __aicore__ inline void GetScaleAndOffset(__gm__ uint8_t *scale, __gm__ uint8_t *offset)
    {
        AscendC::GlobalTensor<T> gm_s;
        gm_s.SetGlobalBuffer((__gm__ T *)scale);
        AscendC::LocalTensor<T> tempFp16 = x_buf_fp32.Get<T>();
        DataCopy(tempFp16, gm_s, BLOCK_SIZE / sizeof(T));
        if constexpr (AscendC::IsSameType<T, half>::value) {
            AscendC::SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            AscendC::WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
            inputScale = 1 / ((float)(tempFp16.GetValue(0)) == 0 ? 1 : (float)(tempFp16.GetValue(0)));
        } else {
            AscendC::LocalTensor<float> tempFp32 = y_buf_fp32.Get<float>();
            AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            Cast(tempFp32, tempFp16, AscendC::RoundMode::CAST_NONE, 1);
            AscendC::SetFlag<HardEvent::V_S>(EVENT_ID0);
            AscendC::WaitFlag<HardEvent::V_S>(EVENT_ID0);
            inputScale = 1 / ((float)(tempFp32.GetValue(0)) == 0 ? 1 : (float)(tempFp32.GetValue(0)));
        }
        AscendC::GlobalTensor<int8_t> gm_o;
        gm_o.SetGlobalBuffer((__gm__ int8_t *)offset);
        AscendC::LocalTensor<int8_t> tempInt8 = x_buf_fp32.Get<int8_t>();
        DataCopy(tempInt8, gm_o, BLOCK_SIZE / sizeof(int8_t));
        AscendC::SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        inputOffset = (float)(tempInt8.GetValue(0));
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> x_que;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> z_que;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> gamma_que;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> beta_que;
    AscendC::TBuf<AscendC::TPosition::VECCALC> ave_buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tensor_buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> var_buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> x_buf_fp32;
    AscendC::TBuf<AscendC::TPosition::VECCALC> y_buf_fp32;
    AscendC::TBuf<AscendC::TPosition::VECCALC> z_buf_fp32;
    AscendC::GlobalTensor<T> x_gm;
    AscendC::GlobalTensor<T> gamma_gm;
    AscendC::GlobalTensor<T> beta_gm;
    AscendC::GlobalTensor<int8_t> z_gm;
    uint32_t copyLenAlign{0};
    uint32_t colsAligned{0};
    uint32_t half_num{0};
    uint32_t num_core{0};
    uint32_t num_last_dim{0};
    uint32_t row_step{0};
    uint32_t row_work{0};
    uint32_t row_tail_{0};
    uint64_t gm_offset_{0};
    uint32_t first_dim_per_times{0};
    uint32_t nl_first_dim_per_core{0};
    uint32_t l_first_dim_per_core{0};
    uint32_t slice_size{0};
    float inputScale{0};
    float inputOffset{0};
    float eps{0};
    float aveNum{0};
};
