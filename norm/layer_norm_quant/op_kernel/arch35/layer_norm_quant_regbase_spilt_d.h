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
 * \file layer_norm_quant_regbase_spilt_d.h
 * \brief
 */
#include "kernel_operator.h"
#include "utils.h"

template <typename T>
class KernelLayerNormQuantSplitD {
public:
    __aicore__ inline KernelLayerNormQuantSplitD() {}

    __aicore__ inline void Init(__gm__ uint8_t *x, __gm__ uint8_t *gamma, __gm__ uint8_t *beta, __gm__ uint8_t *scale,
                                __gm__ uint8_t *offset, __gm__ uint8_t *z, LayerNormQuantRegTilingData tilingData)
    {
        half_num = 2;
        num_core = tilingData.numCore;
        num_last_dim = tilingData.numLastDim;
        nl_first_dim_per_core = tilingData.nlFirstdimPerCore;
        l_first_dim_per_core = tilingData.lFirstdimPerCore;
        first_dim_per_times = tilingData.firstDimPerTimes;
        aveNum = tilingData.aveStr;
        eps = tilingData.epsStr;

        if (AscendC::GetBlockIdx() != num_core - 1) {
            row_work = nl_first_dim_per_core;
            row_step = first_dim_per_times;
        } else {
            row_work = l_first_dim_per_core;
            row_step = MIN(first_dim_per_times, row_work);
        }

        num_slice = tilingData.sliceNum;
        slice_size = tilingData.sliceSize;
        tail_slice_size = tilingData.tailSliceSize;
        gm_offset_ = static_cast<uint64_t>(nl_first_dim_per_core) * num_last_dim;

        x_gm.SetGlobalBuffer((__gm__ T *)x + AscendC::GetBlockIdx() * gm_offset_);
        z_gm.SetGlobalBuffer((__gm__ int8_t *)z + AscendC::GetBlockIdx() * gm_offset_);
        gamma_gm.SetGlobalBuffer((__gm__ T *)gamma);
        beta_gm.SetGlobalBuffer((__gm__ T *)beta);

        pipe.InitBuffer(x_que, BUFFER_NUM, row_step * RoundUp(slice_size) * sizeof(T));
        pipe.InitBuffer(z_que, BUFFER_NUM, row_step * RoundUp(slice_size) * INT8_DATA_BYTE);
        pipe.InitBuffer(beta_que, BUFFER_NUM, 1 * RoundUp(slice_size) * sizeof(T));
        pipe.InitBuffer(gamma_que, BUFFER_NUM, 1 * RoundUp(slice_size) * sizeof(T));
        pipe.InitBuffer(ave_buf, BLOCK_NUMEL * sizeof(T));
        pipe.InitBuffer(var_buf, BLOCK_NUMEL * sizeof(T));
        pipe.InitBuffer(x_buf_fp32, half_num * RoundUp(slice_size) * 2);
        pipe.InitBuffer(y_buf_fp32, half_num * RoundUp(slice_size) * 2);
        pipe.InitBuffer(z_buf_fp32, half_num * RoundUp(slice_size) * 2);

        GetScaleAndOffset(scale, offset);
    }

    __aicore__ inline void Process()
    {
        for (uint32_t pid = 0; pid < row_work; pid++) {
            uint32_t row_offset = pid * num_last_dim;

            float mean = ComputeMeanRegBase(row_offset);
            float variance = ComputeVarianceRegBase(row_offset, mean);

            for (uint32_t sid = 0; sid < num_slice; sid++) {
                uint32_t slice_offset = 0;
                uint64_t col_offset = 0;
                uint32_t eleNum = 0;
                GetSliceOffsetAndSizeRegBase(row_offset, sid, slice_offset, col_offset, eleNum);

                SliceCopyInRegBase(slice_offset, col_offset, eleNum);

                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                SlicePrecisionComputeRegBase(eleNum, mean, variance);
                SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);

                SliceCopyOut(col_offset, eleNum);
            }
        }
    }

private:

    __aicore__ inline void GetScaleAndOffset(__gm__ uint8_t *scale, __gm__ uint8_t *offset)
    {
        AscendC::GlobalTensor<T> gm_s;
        gm_s.SetGlobalBuffer((__gm__ T *)scale);
        LocalTensor<T> tmpFp16 = x_buf_fp32.Get<T>();
        DataCopy(tmpFp16, gm_s, BLOCK_SIZE / sizeof(T));
        if constexpr (AscendC::IsSameType<T, half>::value || AscendC::IsSameType<T, float>::value) {
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
            inputScale = 1 / (static_cast<float>(tmpFp16.GetValue(0)) == 0 ? 1 : static_cast<float>(tmpFp16.GetValue(0)));
        } else if constexpr (AscendC::IsSameType<T, bfloat16_t>::value){
            LocalTensor<float> tmpFp32 = y_buf_fp32.Get<float>();
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            Cast(tmpFp32, tmpFp16, AscendC::RoundMode::CAST_NONE, 1);
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            inputScale = 1 / (static_cast<float>(tmpFp32.GetValue(0)) == 0 ? 1 : static_cast<float>(tmpFp32.GetValue(0)));
        }
        AscendC::GlobalTensor<int8_t> gm_o;
        gm_o.SetGlobalBuffer((__gm__ int8_t *)offset);
        LocalTensor<int8_t> tmpInt8 = x_buf_fp32.Get<int8_t>();
        DataCopy(tmpInt8, gm_o, BLOCK_SIZE / sizeof(int8_t));
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        inputOffset = static_cast<float>(tmpInt8.GetValue(0));
    }

    __aicore__ inline void GetSliceOffsetAndSizeRegBase(uint32_t row_offset, uint32_t sid, uint32_t &slice_offset,
                                                 uint64_t &col_offset, uint32_t &eleNum)
    {
        slice_offset = sid * slice_size;
        col_offset = row_offset + slice_offset;
        eleNum = (sid == (num_slice - 1)) ? tail_slice_size : slice_size;
    }

    __aicore__ inline float ComputeMeanRegBase(uint32_t row_offset)
    {
        //  num_last_dim -> num_slice/tail_slice
        float temp_sum = 0;
        for (uint32_t sid = 0; sid < num_slice; sid++) {
            uint32_t slice_offset = 0;
            uint64_t col_offset = 0;
            uint32_t eleNum = 0;
            GetSliceOffsetAndSizeRegBase(row_offset, sid, slice_offset, col_offset, eleNum);
            temp_sum += ComputeSliceMeanRegBase(col_offset, eleNum);
        }
        return temp_sum * aveNum;
    }

    __aicore__ inline float ComputeVarianceRegBase(uint32_t row_offset, float mean)
    {
        float ssd = 0;
        for (uint32_t sid = 0; sid < num_slice; sid++) {
            uint32_t slice_offset = 0;
            uint64_t col_offset = 0;
            uint32_t eleNum = 0;
            GetSliceOffsetAndSizeRegBase(row_offset, sid, slice_offset, col_offset, eleNum);
            ssd += ComputeSliceSSD(col_offset, eleNum, mean);
        }
        return ssd * aveNum + eps;
    }

    __aicore__ inline float ComputeSliceMeanRegBase(uint64_t col_offset, uint32_t size)
    {
        LocalTensor<T> x_local = x_que.AllocTensor<T>();

        DataCopyPadExtParams<T> padParams;
        padParams.isPad = false;
        padParams.paddingValue = static_cast<T>(0.0);

        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = size * sizeof(T);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;

        DataCopyPad(x_local, x_gm[col_offset], dataCopyParams, padParams);

        x_que.EnQue(x_local);
        LocalTensor<T> x_local2 = x_que.DeQue<T>();
        LocalTensor<float> x_local_fp32 = x_buf_fp32.Get<float>();
        LocalTensor<float> z_local_fp32 = z_buf_fp32.Get<float>();
        LocalTensor<float> y_local_fp32 = y_buf_fp32.Get<float>();
        PipeBarrier<PIPE_V>();
        if constexpr (IsSameType<T, half>::value ||IsSameType<T, bfloat16_t>::value){
                Cast(x_local_fp32, x_local2, AscendC::RoundMode::CAST_NONE, size);
        } else {
            Adds(x_local_fp32, x_local2, 0, size);
        }
        
        PipeBarrier<PIPE_V>();
        ReduceSum(z_local_fp32, x_local_fp32, y_local_fp32, size);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        float x_sum = z_local_fp32.GetValue(0);
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        x_que.FreeTensor(x_local2);
        return x_sum;
    }

    __aicore__ inline float ComputeSliceSSD(uint64_t col_offset, uint32_t size, float mean)
    {
        LocalTensor<T> x_local = x_que.AllocTensor<T>();

        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = size * sizeof(T);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;

        DataCopyPadExtParams<T> padParams;
        padParams.isPad = false;
        padParams.paddingValue = static_cast<T>(0.0);

        DataCopyPad(x_local, x_gm[col_offset], dataCopyParams, padParams);

        x_que.EnQue(x_local);
        LocalTensor<T> x_local2 = x_que.DeQue<T>();
        LocalTensor<float> var_local = var_buf.Get<float>();
        LocalTensor<float> x_local_fp32 = x_buf_fp32.Get<float>();
        LocalTensor<float> y_local_fp32 = y_buf_fp32.Get<float>();
        LocalTensor<float> z_local_fp32 = z_buf_fp32.Get<float>();
        PipeBarrier<PIPE_V>();
        if constexpr (IsSameType<T, half>::value ||IsSameType<T, bfloat16_t>::value){
            Cast(x_local_fp32, x_local2, AscendC::RoundMode::CAST_NONE, size);
        } else {
            Adds(x_local_fp32, x_local2, 0, size);
        }
        PipeBarrier<PIPE_V>();
        Duplicate(y_local_fp32, mean, size);
        PipeBarrier<PIPE_V>();
        Sub(z_local_fp32, x_local_fp32, y_local_fp32, size);
        PipeBarrier<PIPE_V>();
        Mul(x_local_fp32, z_local_fp32, z_local_fp32, size);
        PipeBarrier<PIPE_V>();
        ReduceSum(var_local, x_local_fp32, y_local_fp32, size);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        float var_local_temp = var_local.GetValue(0);
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        x_que.FreeTensor(x_local2);
        return var_local_temp;
    }

    __aicore__ inline void SliceCopyInRegBase(uint32_t slice_offset, uint64_t col_offset, uint32_t size)
    {
        LocalTensor<T> x_local = x_que.AllocTensor<T>();
        LocalTensor<T> beta_local = beta_que.AllocTensor<T>();
        LocalTensor<T> gamma_local = gamma_que.AllocTensor<T>();

        DataCopyPadExtParams<T> padParams;
        padParams.isPad = false;
        padParams.paddingValue = static_cast<T>(0.0);

        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = size * sizeof(T);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;

        DataCopyPad(x_local, x_gm[col_offset], dataCopyParams, padParams);
        DataCopyPad(beta_local, beta_gm[slice_offset], dataCopyParams, padParams);
        DataCopyPad(gamma_local, gamma_gm[slice_offset], dataCopyParams, padParams);

        x_que.EnQue(x_local);
        beta_que.EnQue(beta_local);
        gamma_que.EnQue(gamma_local);
    }

    __aicore__ inline void SlicePrecisionComputeRegBase(uint32_t nums, float mean, float var)
    {
        LocalTensor<T> x_local = x_que.DeQue<T>();
        LocalTensor<T> beta_local = beta_que.DeQue<T>();
        LocalTensor<T> gamma_local = gamma_que.DeQue<T>();
        LocalTensor<int8_t> z_local = z_que.AllocTensor<int8_t>();
        LocalTensor<float> x_local_fp32 = x_buf_fp32.Get<float>();
        LocalTensor<float> y_local_fp32 = y_buf_fp32.Get<float>();
        LocalTensor<float> z_local_fp32 = z_buf_fp32.Get<float>();
        PipeBarrier<PIPE_V>();
        if constexpr (IsSameType<T, half>::value ||IsSameType<T, bfloat16_t>::value){
            Cast(x_local_fp32, x_local, AscendC::RoundMode::CAST_NONE, nums);
        } else {
            Adds(x_local_fp32, x_local, 0, nums);
        }
        
        PipeBarrier<PIPE_V>();
        Duplicate(y_local_fp32, mean, nums);
        PipeBarrier<PIPE_V>();
        Sub(z_local_fp32, x_local_fp32, y_local_fp32, nums);
        PipeBarrier<PIPE_V>();
        Duplicate(y_local_fp32, var, nums);
        PipeBarrier<PIPE_V>();
        Sqrt(y_local_fp32, y_local_fp32, nums);
        PipeBarrier<PIPE_V>();
        Duplicate(x_local_fp32, (float)1, nums);
        PipeBarrier<PIPE_V>();
        Div(y_local_fp32, x_local_fp32, y_local_fp32, nums);
        PipeBarrier<PIPE_V>();
        ComputeNormalLayerNorm(x_local_fp32, y_local_fp32, gamma_local, beta_local, z_local_fp32, nums);
        SetFlag<HardEvent::V_S>(EVENT_ID2);
        WaitFlag<HardEvent::V_S>(EVENT_ID2);
        PipeBarrier<PIPE_V>();
        Muls(z_local_fp32, z_local_fp32, inputScale, nums);
        PipeBarrier<PIPE_V>();
        Adds(z_local_fp32, z_local_fp32, inputOffset, nums);
        PipeBarrier<PIPE_V>();
        LocalTensor<half> tmpFp16 = x_buf_fp32.Get<half>();
        CastFrom32To16(tmpFp16, z_local_fp32, nums);
        PipeBarrier<PIPE_V>();
        CastFromF16ToI8(z_local, tmpFp16, QUANT_MIN, nums);
        PipeBarrier<PIPE_V>();
        x_que.FreeTensor(x_local);
        beta_que.FreeTensor(beta_local);
        gamma_que.FreeTensor(gamma_local);
        z_que.EnQue(z_local);
    }

    __aicore__ inline void SliceCopyOut(uint64_t offset, uint32_t size)
    {
        LocalTensor<int8_t> z = z_que.DeQue<int8_t>();
        DataCopyPadExtParams<T> padParams;
        padParams.isPad = false;
        padParams.paddingValue = static_cast<T>(0.0);

        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;  // 搬多少块
        dataCopyParams.blockLen = size;  // 搬多长
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;

        DataCopyPad(z_gm[offset], z, dataCopyParams);
        z_que.FreeTensor(z);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> x_que;
    TQue<QuePosition::VECOUT, BUFFER_NUM> z_que;
    TQue<TPosition::VECIN, BUFFER_NUM> gamma_que;
    TQue<TPosition::VECIN, BUFFER_NUM> beta_que;
    
    TBuf<TPosition::VECCALC> x_buf_fp32;
    TBuf<TPosition::VECCALC> y_buf_fp32;
    TBuf<TPosition::VECCALC> z_buf_fp32;

    TBuf<TPosition::VECCALC> ave_buf;
    TBuf<TPosition::VECCALC> var_buf;
    GlobalTensor<T> x_gm;
    GlobalTensor<T> gamma_gm;
    GlobalTensor<T> beta_gm;
    GlobalTensor<int8_t> z_gm;
    uint32_t half_num{0};
    uint32_t num_core{0};
    uint32_t num_last_dim{0};
    uint32_t row_step{0};
    uint32_t row_work{0};
    uint64_t gm_offset_{0};
    uint32_t first_dim_per_times{0};
    uint32_t nl_first_dim_per_core{0};
    uint32_t l_first_dim_per_core{0};
    uint32_t slice_size{0};
    uint32_t num_slice{0};
    uint32_t tail_slice_size{0};
    float eps{0};
    float aveNum{0};
    float inputScale{0};
    float inputOffset{0};
};
