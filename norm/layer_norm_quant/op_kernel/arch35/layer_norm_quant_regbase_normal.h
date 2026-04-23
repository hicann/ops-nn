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
 * \file layer_norm_quant_regbase_normal.h
 * \brief
 */
#include "kernel_operator.h"
#include "utils.h"

template <typename T>
class KernelLayerNormQuantNormal {
public:
    __aicore__ inline KernelLayerNormQuantNormal() {}

    __aicore__ inline void Init(__gm__ uint8_t *x, __gm__ uint8_t *gamma, __gm__ uint8_t *beta, __gm__ uint8_t *scale,
                                __gm__ uint8_t *offset, __gm__ uint8_t *z, LayerNormQuantRegTilingData tilingData)
    {
        half_num = 2;
        num_core = tilingData.numCore;
        num_last_dim = tilingData.numLastDim;
        num_first_dim = tilingData.numFirstDim;
        nl_first_dim_per_core = tilingData.nlFirstdimPerCore;
        l_first_dim_per_core = tilingData.lFirstdimPerCore;
        first_dim_per_times = tilingData.firstDimPerTimes;
        colsAligned = tilingData.colsAligned;
        aveNum = tilingData.aveStr;
        eps = tilingData.epsStr;

        if (AscendC::GetBlockIdx() != num_core - 1) {
            row_work = nl_first_dim_per_core;
            row_step = first_dim_per_times;
        } else {
            row_work = l_first_dim_per_core;
            row_step = MIN(first_dim_per_times, row_work);
        }

        slice_size = tilingData.sliceSize;
        row_tail_ = (row_work % first_dim_per_times == 0) ? first_dim_per_times : (row_work % first_dim_per_times);
        gm_offset_ = static_cast<uint64_t>(nl_first_dim_per_core) * num_last_dim;
        copyLenAlign = BLOCK_ALIGN(num_last_dim * sizeof(T), 32) / sizeof(T);  // 向上对齐

        x_gm.SetGlobalBuffer((__gm__ T *)x + AscendC::GetBlockIdx() * gm_offset_);
        z_gm.SetGlobalBuffer((__gm__ int8_t *)z + AscendC::GetBlockIdx() * gm_offset_);
        gamma_gm.SetGlobalBuffer((__gm__ T *)gamma);
        beta_gm.SetGlobalBuffer((__gm__ T *)beta);

        pipe.InitBuffer(x_que, BUFFER_NUM, row_step * RoundUp(slice_size) * sizeof(T));
        pipe.InitBuffer(z_que, BUFFER_NUM, row_step * RoundUp(slice_size) * INT8_DATA_BYTE);
        pipe.InitBuffer(gamma_que, BUFFER_NUM, 1 * RoundUp(slice_size) * sizeof(T));
        pipe.InitBuffer(beta_que, BUFFER_NUM, 1 * RoundUp(slice_size) * sizeof(T));
        pipe.InitBuffer(ave_buf, BLOCK_NUMEL * sizeof(T));
        pipe.InitBuffer(var_buf, BLOCK_NUMEL * sizeof(T));
        pipe.InitBuffer(x_buf_fp32, half_num * RoundUp(slice_size) * 2);
        pipe.InitBuffer(y_buf_fp32, half_num * RoundUp(slice_size) * 2);
        pipe.InitBuffer(z_buf_fp32, half_num * RoundUp(slice_size) * 2);

        GetScaleAndOffset(scale, offset);
    }

    __aicore__ inline void Process()
    {
        uint32_t move_cnt = CEIL_DIV(row_work, row_step);
        for (uint32_t i = 0; i < move_cnt; ++i) {
            if (i < move_cnt - 1) {
                FastCopyIn(i, row_step);
                FastCopyInGammaBeta();

                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                FastPrecisionComputeRegbase(row_step);
                SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);

                FastCopyOut(i, row_step);
            } else {
                FastCopyIn(i, row_tail_);
                FastCopyInGammaBeta();

                SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
                WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
                FastPrecisionComputeRegbase(row_tail_);
                SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
                WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);

                FastCopyOut(i, row_tail_);
            }
        }
    }

private:

    __aicore__ inline uint32_t BLOCK_ALIGN(uint32_t x, uint32_t blockSize)
    {
        return (blockSize > 0) ? (x + blockSize - 1) / blockSize * blockSize : 0;
    }

    __aicore__ inline void FastCopyIn(uint64_t proc_id, uint64_t size)
    {
        LocalTensor<T> x_local = x_que.AllocTensor<T>();
        uint64_t offset = proc_id * row_step * num_last_dim;

        DataCopyPadExtParams<T> padParams;
        padParams.isPad = true;
        padParams.paddingValue = static_cast<T>(0.0);
        padParams.rightPadding = copyLenAlign - num_last_dim;

        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = size;  // 搬多少块
        dataCopyParams.blockLen = num_last_dim * sizeof(T);  // 搬多长
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;

        DataCopyPad(x_local, x_gm[offset], dataCopyParams, padParams);
        x_que.EnQue(x_local);
    }

    __aicore__ inline void FastCopyInGammaBeta(){
        LocalTensor<T> beta_local = beta_que.AllocTensor<T>();
        LocalTensor<T> gamma_local = gamma_que.AllocTensor<T>();

        DataCopyPadExtParams<T> padParams;
        padParams.isPad = true;
        padParams.paddingValue = static_cast<T>(0.0);
        padParams.rightPadding = copyLenAlign - num_last_dim;

        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = num_last_dim * sizeof(T);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;

        DataCopyPad(beta_local, beta_gm[0], dataCopyParams, padParams);
        DataCopyPad(gamma_local, gamma_gm[0], dataCopyParams, padParams);
        beta_que.EnQue(beta_local);
        gamma_que.EnQue(gamma_local);
    }

    __aicore__ inline float ComputeNormalMean(LocalTensor<float>& x_local_fp32, uint64_t rid)
    {
        LocalTensor<float> y_local_fp32 = y_buf_fp32.Get<float>();
        LocalTensor<float> z_local_fp32 = z_buf_fp32.Get<float>();
        LocalTensor<float> ave_local = ave_buf.Get<float>();

        Duplicate(y_local_fp32, aveNum, num_last_dim);
        PipeBarrier<PIPE_V>();
        Mul(z_local_fp32, x_local_fp32, y_local_fp32, num_last_dim);
        PipeBarrier<PIPE_V>();
        ReduceSum(ave_local, z_local_fp32, y_local_fp32, num_last_dim);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        float ave_local_temp = ave_local.GetValue(0);
        return ave_local_temp;
    }

    __aicore__ inline float ComputeNormalVariance(LocalTensor<float>& x_local_fp32,float ave_local_temp)
    {
        LocalTensor<float> y_local_fp32 = y_buf_fp32.Get<float>();
        LocalTensor<float> z_local_fp32 = z_buf_fp32.Get<float>();
        LocalTensor<float> var_local = var_buf.Get<float>();
        Duplicate(y_local_fp32, ave_local_temp, num_last_dim);
        PipeBarrier<PIPE_V>();
        Sub(z_local_fp32, x_local_fp32, y_local_fp32, num_last_dim);
        PipeBarrier<PIPE_V>();
        Mul(x_local_fp32, z_local_fp32, z_local_fp32, num_last_dim);
        PipeBarrier<PIPE_V>();
        Muls(y_local_fp32, x_local_fp32, aveNum, num_last_dim);
        PipeBarrier<PIPE_V>();
        ReduceSum(var_local, y_local_fp32, x_local_fp32, num_last_dim);
        PipeBarrier<PIPE_V>();
        Adds(var_local, var_local, eps, BLOCK_NUMEL / half_num);
        PipeBarrier<PIPE_V>();
        Sqrt(var_local, var_local, BLOCK_NUMEL / half_num);
        PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_S>(EVENT_ID2);
        WaitFlag<HardEvent::V_S>(EVENT_ID2);
        float var_local_temp = var_local.GetValue(0);
        return var_local_temp;
    }

    __aicore__ inline void FastPrecisionComputeRegbase(uint64_t nums)
    {
        LocalTensor<T> x_local = x_que.DeQue<T>();
        LocalTensor<T> beta_local = beta_que.DeQue<T>();
        LocalTensor<T> gamma_local = gamma_que.DeQue<T>();
        LocalTensor<float> y_local_fp32 = y_buf_fp32.Get<float>();
        LocalTensor<int8_t> z_local = z_que.AllocTensor<int8_t>();
        LocalTensor<float> x_local_fp32 = x_buf_fp32.Get<float>();
        LocalTensor<float> z_local_fp32 = z_buf_fp32.Get<float>();
        for (uint64_t rid = 0; rid < nums; ++rid) {
            if constexpr (IsSameType<T, half>::value ||IsSameType<T, bfloat16_t>::value){
                Cast(x_local_fp32, x_local[rid * copyLenAlign], AscendC::RoundMode::CAST_NONE, num_last_dim);
            } else {
                Adds(x_local_fp32, x_local[rid * copyLenAlign], 0, num_last_dim);
            }
            PipeBarrier<PIPE_V>();

            float ave_local_temp = ComputeNormalMean(x_local_fp32, rid);
            SetFlag<HardEvent::S_V>(EVENT_ID0);
            WaitFlag<HardEvent::S_V>(EVENT_ID0);
            PipeBarrier<PIPE_V>();
            
            float var_local_temp = ComputeNormalVariance(x_local_fp32, ave_local_temp);
            SetFlag<HardEvent::S_V>(EVENT_ID2);
            WaitFlag<HardEvent::S_V>(EVENT_ID2);
            PipeBarrier<PIPE_V>();
            Duplicate(x_local_fp32, var_local_temp, num_last_dim);
            PipeBarrier<PIPE_V>();
            Duplicate(y_local_fp32, (float)1, num_last_dim);
            PipeBarrier<PIPE_V>();
            Div(y_local_fp32, y_local_fp32, x_local_fp32, num_last_dim);
            PipeBarrier<PIPE_V>();
            ComputeNormalLayerNorm(x_local_fp32, y_local_fp32, gamma_local, beta_local, z_local_fp32, num_last_dim);
            
            PipeBarrier<PIPE_V>();
            Muls(z_local_fp32, z_local_fp32, inputScale, num_last_dim);
            PipeBarrier<PIPE_V>();
            Adds(z_local_fp32, z_local_fp32, inputOffset, num_last_dim);
            PipeBarrier<PIPE_V>();
            LocalTensor<half> tmpFp16 = x_buf_fp32.Get<half>();
            CastFrom32To16(tmpFp16, z_local_fp32, num_last_dim);
            PipeBarrier<PIPE_V>();
            CastFromF16ToI8(z_local[rid * colsAligned], tmpFp16, QUANT_MIN, num_last_dim);
            PipeBarrier<PIPE_V>();
        }
        x_que.FreeTensor(x_local);
        beta_que.FreeTensor(beta_local);
        gamma_que.FreeTensor(gamma_local);
        z_que.EnQue(z_local);
    }

    __aicore__ inline void FastCopyOut(uint64_t proc_id, uint64_t size)
    {
        LocalTensor<int8_t> z = z_que.DeQue<int8_t>();
        uint64_t offset = proc_id * row_step * num_last_dim;

        DataCopyPadExtParams<T> padParams;
        padParams.isPad = false;
        padParams.paddingValue = static_cast<T>(0.0);
        padParams.rightPadding = copyLenAlign - num_last_dim;

        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = size;  // 搬多少块
        dataCopyParams.blockLen = num_last_dim * 1;  // 搬多长
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;

        DataCopyPad(z_gm[offset], z, dataCopyParams);

        z_que.FreeTensor(z);
    }

    __aicore__ inline void GetScaleAndOffset(__gm__ uint8_t *scale, __gm__ uint8_t *offset)
    {
        AscendC::GlobalTensor<T> gm_s;
        gm_s.SetGlobalBuffer((__gm__ T *)scale);
        LocalTensor<T> tmpFp16 = x_buf_fp32.Get<T>();
        DataCopy(tmpFp16, gm_s, BLOCK_SIZE / sizeof(T));
        if constexpr (AscendC::IsSameType<T, half>::value || AscendC::IsSameType<T, float>::value) {
            SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
            inputScale = 1 / ((float)(tmpFp16.GetValue(0)) == 0 ? 1 : (float)(tmpFp16.GetValue(0)));
        } else if constexpr (AscendC::IsSameType<T, bfloat16_t>::value){
            LocalTensor<float> tmpFp32 = y_buf_fp32.Get<float>();
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            Cast(tmpFp32, tmpFp16, AscendC::RoundMode::CAST_NONE, 1);
            SetFlag<HardEvent::V_S>(EVENT_ID0);
            WaitFlag<HardEvent::V_S>(EVENT_ID0);
            inputScale = 1 / ((float)(tmpFp32.GetValue(0)) == 0 ? 1 : (float)(tmpFp32.GetValue(0)));
        }
        AscendC::GlobalTensor<int8_t> gm_o;
        gm_o.SetGlobalBuffer((__gm__ int8_t *)offset);
        LocalTensor<int8_t> tmpInt8 = x_buf_fp32.Get<int8_t>();
        DataCopy(tmpInt8, gm_o, BLOCK_SIZE / sizeof(int8_t));
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);
        inputOffset = (float)(tmpInt8.GetValue(0));
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> x_que;
    TQue<QuePosition::VECOUT, BUFFER_NUM> z_que;
    TQue<TPosition::VECIN, BUFFER_NUM> gamma_que;
    TQue<TPosition::VECIN, BUFFER_NUM> beta_que;
    TBuf<TPosition::VECCALC> ave_buf;
    TBuf<TPosition::VECCALC> var_buf;
    TBuf<TPosition::VECCALC> y_buf_fp32;
    TBuf<TPosition::VECCALC> x_buf_fp32;
    TBuf<TPosition::VECCALC> z_buf_fp32;
    GlobalTensor<T> x_gm;
    GlobalTensor<T> beta_gm;
    GlobalTensor<T> gamma_gm;
    GlobalTensor<int8_t> z_gm;
    uint32_t half_num{0};
    uint32_t num_core{0};
    uint32_t colsAligned{0};
    uint32_t copyLenAlign{0};
    uint32_t num_first_dim{0};
    uint32_t num_last_dim{0};
    uint32_t row_step{0};
    uint32_t row_work{0};
    uint64_t gm_offset_{0};
    uint32_t row_tail_{0};
    uint32_t first_dim_per_times{0};
    uint32_t nl_first_dim_per_core{0};
    uint32_t l_first_dim_per_core{0};
    uint32_t slice_size{0};
    float eps{0};
    float aveNum{0};
    float inputScale{0};
    float inputOffset{0};
};
