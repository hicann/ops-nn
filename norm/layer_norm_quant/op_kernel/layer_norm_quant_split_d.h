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
 * \file layer_norm_quant_split_d.h
 * \brief
 */
#include "kernel_operator.h"
#include "layer_norm_quant_helper.h"

using AscendC::HardEvent;

template <typename T>
class LayerNormQuantSplitD {
public:
    __aicore__ inline LayerNormQuantSplitD() {}

    __aicore__ inline uint32_t ROUND_UP_FUNC(uint32_t x) { return (x + BLOCK_NUM - 1) / BLOCK_NUM * BLOCK_NUM; }

    __aicore__ inline void Init(__gm__ uint8_t *x, __gm__ uint8_t *gamma, __gm__ uint8_t *beta, __gm__ uint8_t *scale,
                                __gm__ uint8_t *offset, __gm__ uint8_t *z, LayerNormQuantTilingData tilingData)
    {
        half_num = 2;
        num_core = tilingData.numCore;
        num_last_dim = tilingData.numLastDim;
        nl_first_dim_per_core = tilingData.nlFirstdimPerCore;
        l_first_dim_per_core = tilingData.lFirstdimPerCore;
        first_dim_per_times = tilingData.firstDimPerTimes;
        eps = tilingData.epsStr;
        aveNum = tilingData.aveStr;

        if (AscendC::GetBlockIdx() != num_core - 1) {
            row_work = nl_first_dim_per_core;
            row_step = first_dim_per_times;
        } else {
            row_work = l_first_dim_per_core;
            row_step = MIN(first_dim_per_times, row_work);
        }

        slice_size = tilingData.sliceSize;
        num_slice = tilingData.sliceNum;
        tail_slice_size = tilingData.tailSliceSize;
        gm_offset_ = static_cast<uint64_t>(nl_first_dim_per_core) * num_last_dim;

        x_gm.SetGlobalBuffer((__gm__ T *)x + AscendC::GetBlockIdx() * gm_offset_);
        z_gm.SetGlobalBuffer((__gm__ int8_t *)z + AscendC::GetBlockIdx() * gm_offset_);
        gamma_gm.SetGlobalBuffer((__gm__ T *)gamma);
        beta_gm.SetGlobalBuffer((__gm__ T *)beta);

        pipe.InitBuffer(x_que, BUFFER_NUM, row_step * ROUND_UP_FUNC(slice_size) * DATA_BYTE);
        pipe.InitBuffer(z_que, BUFFER_NUM, row_step * ROUND_UP_FUNC(slice_size) * INT8_DATA_BYTE);
        pipe.InitBuffer(beta_que, BUFFER_NUM, 1 * ROUND_UP_FUNC(slice_size) * DATA_BYTE);
        pipe.InitBuffer(gamma_que, BUFFER_NUM, 1 * ROUND_UP_FUNC(slice_size) * DATA_BYTE);
        pipe.InitBuffer(ave_buf, BLOCK_NUM * DATA_BYTE);
        pipe.InitBuffer(var_buf, BLOCK_NUM * DATA_BYTE);
        pipe.InitBuffer(x_buf_fp32, half_num * ROUND_UP_FUNC(slice_size) * DATA_BYTE);
        pipe.InitBuffer(y_buf_fp32, half_num * ROUND_UP_FUNC(slice_size) * DATA_BYTE);
        pipe.InitBuffer(z_buf_fp32, half_num * ROUND_UP_FUNC(slice_size) * DATA_BYTE);
#if __CCE_AICORE__ == 220
#else
        pipe.InitBuffer(tensor_buf, 32);
#endif

        GetScaleAndOffset(scale, offset);
    }

    __aicore__ inline void Process()
    {
        for (uint32_t pid = 0; pid < row_work; pid++) {
            uint32_t row_offset = pid * num_last_dim;

            float mean = ComputeMean(row_offset);
            float variance = ComputeVariance(row_offset, mean);

            for (uint32_t sid = 0; sid < num_slice; sid++) {
                uint32_t eleNum = 0;
                uint64_t col_offset = 0;
                uint32_t slice_offset = 0;
                GetSliceOffsetAndSize(row_offset, sid, slice_offset, col_offset, eleNum);

                SliceCopyIn(slice_offset, col_offset, eleNum);

                AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                SlicePrecisionCompute(eleNum, mean, variance);
                AscendC::SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                AscendC::WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);

                SliceCopyOut(col_offset, eleNum);
            }
        }
    }

private:
    __aicore__ inline void GetScaleAndOffset(__gm__ uint8_t *scale, __gm__ uint8_t *offset)
    {
        AscendC::GlobalTensor<T> gm_s;
        gm_s.SetGlobalBuffer((__gm__ T *)scale);
        AscendC::LocalTensor<T> tmpFp16 = x_buf_fp32.Get<T>();
        DataCopy(tmpFp16, gm_s, BLOCK_SIZE / sizeof(T));
        if constexpr (AscendC::IsSameType<T, half>::value) {
            AscendC::SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            AscendC::WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
            inputScale = 1 / ((float)(tmpFp16.GetValue(0)) == 0 ? 1 : (float)(tmpFp16.GetValue(0)));
        } else {
            AscendC::LocalTensor<float> tmpFp32 = y_buf_fp32.Get<float>();
            AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            Cast(tmpFp32, tmpFp16, AscendC::RoundMode::CAST_NONE, 1);
            AscendC::SetFlag<HardEvent::V_S>(EVENT_ID0);
            AscendC::WaitFlag<HardEvent::V_S>(EVENT_ID0);
            inputScale = 1 / ((float)(tmpFp32.GetValue(0)) == 0 ? 1 : (float)(tmpFp32.GetValue(0)));
        }
        AscendC::GlobalTensor<int8_t> gm_o;
        gm_o.SetGlobalBuffer((__gm__ int8_t *)offset);
        AscendC::LocalTensor<int8_t> tmpInt8 = x_buf_fp32.Get<int8_t>();
        DataCopy(tmpInt8, gm_o, BLOCK_SIZE / sizeof(int8_t));
        AscendC::SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        inputOffset = (float)(tmpInt8.GetValue(0));
    }

    __aicore__ inline void GetSliceOffsetAndSize(uint32_t row_offset, uint32_t sid, uint32_t &slice_offset,
                                                 uint64_t &col_offset, uint32_t &eleNum)
    {
        slice_offset = sid * slice_size;
        col_offset = row_offset + slice_offset;
        eleNum = (sid == (num_slice - 1)) ? tail_slice_size : slice_size;
    }

    __aicore__ inline float ComputeMean(uint32_t row_offset)
    {
        //  num_last_dim -> num_slice/tail_slice
        float temp_sum = 0;
        for (uint32_t sid = 0; sid < num_slice; sid++) {
            uint32_t slice_offset = 0;
            uint64_t col_offset = 0;
            uint32_t eleNum = 0;
            GetSliceOffsetAndSize(row_offset, sid, slice_offset, col_offset, eleNum);
            temp_sum += ComputeSliceMean(col_offset, eleNum);
        }
        return temp_sum * aveNum;
    }

    __aicore__ inline float ComputeVariance(uint32_t row_offset, float mean)
    {
        float ssd = 0;
        for (uint32_t sid = 0; sid < num_slice; sid++) {
            uint32_t slice_offset = 0;
            uint64_t col_offset = 0;
            uint32_t eleNum = 0;
            GetSliceOffsetAndSize(row_offset, sid, slice_offset, col_offset, eleNum);
            ssd += ComputeSliceSSD(col_offset, eleNum, mean);
        }
        return ssd * aveNum + eps;
    }

    __aicore__ inline float ComputeSliceMean(uint64_t col_offset, uint32_t size)
    {
        AscendC::LocalTensor<T> x_local = x_que.AllocTensor<T>();

#if __CCE_AICORE__ == 220
        DataCopyPadExtParams<T> padParams;
        padParams.isPad = false;
        padParams.paddingValue = static_cast<T>(0.0);

        DataCopyExtParams dtCopyParams;
        dtCopyParams.blockCount = 1;
        dtCopyParams.blockLen = size * sizeof(T);
        dtCopyParams.srcStride = 0;
        dtCopyParams.dstStride = 0;

        DataCopyPad(x_local, x_gm[col_offset], dtCopyParams, padParams);
#else
        AscendC::LocalTensor<T> tensor_local = tensor_buf.Get<T>();
        DataCopyEx(x_local, x_gm[col_offset], tensor_local, size, 1);
#endif

        x_que.EnQue(x_local);
        AscendC::LocalTensor<T> x_local2 = x_que.DeQue<T>();
        AscendC::LocalTensor<float> x_local_fp32 = x_buf_fp32.Get<float>();
        AscendC::LocalTensor<float> y_local_fp32 = y_buf_fp32.Get<float>();
        AscendC::LocalTensor<float> z_fp32_local = z_buf_fp32.Get<float>();
        AscendC::PipeBarrier<PIPE_V>();
        Cast(x_local_fp32, x_local2, AscendC::RoundMode::CAST_NONE, size);
        AscendC::PipeBarrier<PIPE_V>();
        ReduceSum(z_fp32_local, x_local_fp32, y_local_fp32, size);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::V_S>(EVENT_ID0);
        float x_sum = z_fp32_local.GetValue(0);
        AscendC::SetFlag<HardEvent::S_V>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::S_V>(EVENT_ID0);
        x_que.FreeTensor(x_local2);
        return x_sum;
    }

    __aicore__ inline float ComputeSliceSSD(uint64_t col_offset, uint32_t size, float mean)
    {
        AscendC::LocalTensor<T> x_local = x_que.AllocTensor<T>();

        DataCopyPadExtParams<T> padParams;
        padParams.isPad = false;
        padParams.paddingValue = static_cast<T>(0.0);

#if __CCE_AICORE__ == 220
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = size * sizeof(T);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;

        DataCopyPad(x_local, x_gm[col_offset], dataCopyParams, padParams);
#else
        AscendC::LocalTensor<T> tensor_local = tensor_buf.Get<T>();
        DataCopyEx(x_local, x_gm[col_offset], tensor_local, size, 1);
#endif

        x_que.EnQue(x_local);
        AscendC::LocalTensor<T> x_local2 = x_que.DeQue<T>();
        AscendC::LocalTensor<float> var_local = var_buf.Get<float>();
        AscendC::LocalTensor<float> x_local_fp32 = x_buf_fp32.Get<float>();
        AscendC::LocalTensor<float> y_local_fp32 = y_buf_fp32.Get<float>();
        AscendC::LocalTensor<float> z_fp32_local = z_buf_fp32.Get<float>();
        AscendC::PipeBarrier<PIPE_V>();
        Cast(x_local_fp32, x_local2, AscendC::RoundMode::CAST_NONE, size);
        Duplicate(y_local_fp32, mean, size);
        AscendC::PipeBarrier<PIPE_V>();
        Sub(z_fp32_local, x_local_fp32, y_local_fp32, size);
        AscendC::PipeBarrier<PIPE_V>();
        Mul(x_local_fp32, z_fp32_local, z_fp32_local, size);
        AscendC::PipeBarrier<PIPE_V>();
        ReduceSum(var_local, x_local_fp32, y_local_fp32, size);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::SetFlag<HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::V_S>(EVENT_ID0);
        float var_local_temp = var_local.GetValue(0);
        AscendC::SetFlag<HardEvent::S_V>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::S_V>(EVENT_ID0);
        x_que.FreeTensor(x_local2);
        return var_local_temp;
    }

    __aicore__ inline void SliceCopyIn(uint32_t slice_offset, uint64_t col_offset, uint32_t size)
    {
        AscendC::LocalTensor<T> x_local = x_que.AllocTensor<T>();
        AscendC::LocalTensor<T> beta_local = beta_que.AllocTensor<T>();
        AscendC::LocalTensor<T> gamma_local = gamma_que.AllocTensor<T>();

        DataCopyPadExtParams<T> dtpadParams;
        dtpadParams.isPad = false;
        dtpadParams.paddingValue = static_cast<T>(0.0);
#if __CCE_AICORE__ == 220
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = size * sizeof(T);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;

        DataCopyPad(x_local, x_gm[col_offset], dataCopyParams, dtpadParams);
        DataCopyPad(beta_local, beta_gm[slice_offset], dataCopyParams, dtpadParams);
        DataCopyPad(gamma_local, gamma_gm[slice_offset], dataCopyParams, dtpadParams);
#else
        AscendC::LocalTensor<T> tensor_local = tensor_buf.Get<T>();
        DataCopyEx(x_local, x_gm[col_offset], tensor_local, size, 1);
        DataCopyEx(beta_local, beta_gm[slice_offset], tensor_local, size, 1);
        DataCopyEx(gamma_local, gamma_gm[slice_offset], tensor_local, size, 1);
#endif

        x_que.EnQue(x_local);
        beta_que.EnQue(beta_local);
        gamma_que.EnQue(gamma_local);
    }

    __aicore__ inline void SlicePrecisionCompute(uint32_t nums, float mean, float variance)
    {
        AscendC::LocalTensor<T> x_local = x_que.DeQue<T>();
        AscendC::LocalTensor<T> beta_local = beta_que.DeQue<T>();
        AscendC::LocalTensor<T> gamma_local = gamma_que.DeQue<T>();
        AscendC::LocalTensor<int8_t> z_local = z_que.AllocTensor<int8_t>();
        AscendC::LocalTensor<float> x_local_fp32 = x_buf_fp32.Get<float>();
        AscendC::LocalTensor<float> y_local_fp32 = y_buf_fp32.Get<float>();
        AscendC::LocalTensor<float> z_fp32_local = z_buf_fp32.Get<float>();
        AscendC::PipeBarrier<PIPE_V>();
        Cast(x_local_fp32, x_local, AscendC::RoundMode::CAST_NONE, nums);
        Duplicate(y_local_fp32, mean, nums);
        AscendC::PipeBarrier<PIPE_V>();
        Sub(z_fp32_local, x_local_fp32, y_local_fp32, nums);
        Duplicate(y_local_fp32, variance, nums);
        AscendC::PipeBarrier<PIPE_V>();
        Sqrt(y_local_fp32, y_local_fp32, nums);
        Duplicate(x_local_fp32, (float)1, nums);
        AscendC::PipeBarrier<PIPE_V>();
        Div(y_local_fp32, x_local_fp32, y_local_fp32, nums);
        AscendC::PipeBarrier<PIPE_V>();
        Mul(x_local_fp32, z_fp32_local, y_local_fp32, nums);
        Cast(z_fp32_local, gamma_local, AscendC::RoundMode::CAST_NONE, nums);
        AscendC::PipeBarrier<PIPE_V>();
        Mul(y_local_fp32, x_local_fp32, z_fp32_local, nums);
        Cast(x_local_fp32, beta_local, AscendC::RoundMode::CAST_NONE, nums);
        AscendC::PipeBarrier<PIPE_V>();
        Add(z_fp32_local, y_local_fp32, x_local_fp32, nums);
        AscendC::SetFlag<HardEvent::V_S>(EVENT_ID2);
        AscendC::WaitFlag<HardEvent::V_S>(EVENT_ID2);
        AscendC::PipeBarrier<PIPE_V>();
        Muls(z_fp32_local, z_fp32_local, inputScale, nums);
        AscendC::PipeBarrier<PIPE_V>();
        Adds(z_fp32_local, z_fp32_local, inputOffset, nums);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::LocalTensor<half> tmpFp16 = x_buf_fp32.Get<half>();
        CastFrom32To16(tmpFp16, z_fp32_local, nums);
        AscendC::PipeBarrier<PIPE_V>();
        CastFromF16ToI8(z_local, tmpFp16, QUANT_MIN, nums);
        AscendC::PipeBarrier<PIPE_V>();
        x_que.FreeTensor(x_local);
        beta_que.FreeTensor(beta_local);
        gamma_que.FreeTensor(gamma_local);
        z_que.EnQue(z_local);
    }

    __aicore__ inline void SliceCopyOut(uint64_t offset, uint32_t size)
    {
        AscendC::LocalTensor<int8_t> z = z_que.DeQue<int8_t>();
#if __CCE_AICORE__ == 220
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;  // 搬多少块
        dataCopyParams.blockLen = size;  // 搬多长
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;

        DataCopyPad(z_gm[offset], z, dataCopyParams);
#else
        AscendC::LocalTensor<int8_t> tensor_local = tensor_buf.Get<int8_t>();
        DataCopyEx(z_gm[offset], z, tensor_local, size);
#endif
        z_que.FreeTensor(z);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> x_que;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> z_que;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> gamma_que;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> beta_que;
    AscendC::TBuf<AscendC::TPosition::VECCALC> ave_buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> var_buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> tensor_buf;
    AscendC::TBuf<AscendC::TPosition::VECCALC> x_buf_fp32;
    AscendC::TBuf<AscendC::TPosition::VECCALC> y_buf_fp32;
    AscendC::TBuf<AscendC::TPosition::VECCALC> z_buf_fp32;
    AscendC::GlobalTensor<T> x_gm;
    AscendC::GlobalTensor<T> gamma_gm;
    AscendC::GlobalTensor<T> beta_gm;
    AscendC::GlobalTensor<int8_t> z_gm;
    uint32_t half_num{0};
    uint32_t num_last_dim{0};
    uint32_t row_step{0};
    uint32_t row_work{0};
    uint32_t num_core{0};
    uint64_t gm_offset_{0};
    uint32_t first_dim_per_times{0};
    uint32_t nl_first_dim_per_core{0};
    uint32_t l_first_dim_per_core{0};
    uint32_t num_slice{0};
    uint32_t slice_size{0};
    uint32_t tail_slice_size{0};
    float inputScale{0};
    float inputOffset{0};
    float eps{0};
    float aveNum{0};
};
