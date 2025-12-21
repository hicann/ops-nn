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
 * \file rms_norm_quant.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "rms_norm_quant.h"

namespace {                                     // 匿名命名空间
static constexpr uint32_t BUF_FACTOR = 3;       // 1(g) + 1(sqx) + 1(sum) = 3
static constexpr uint32_t OFFSET_GAMMA = 0;     // the offset of gamma is 0
static constexpr uint32_t OFFSET_SQX = 1;       // the offset of sqx is 1
static constexpr uint32_t OFFSET_SUM = 2;       // the offset of sum is 2
static constexpr uint32_t OFFSET_WORKSPACE = 3; // the offset of workspace is 3
static constexpr uint32_t SIZE_INT4 = 2;        // 用于sizeof(int4) / 2
static constexpr uint32_t DIM_2 = 2;
static constexpr uint32_t REPEAT_TIME_256 = 256; // 256 default stride
static constexpr uint32_t REPEAT_TIME_16 = 16;   // 16 default stride
static constexpr uint32_t REPEAT_TIME_64 = 64;   // 64 default stride
} // namespace

template <typename T, typename yDtype,  bool EN_BETA, bool FastComputeMode = false>
class RmsNormQuant {
public:
    __aicore__ inline RmsNormQuant(
        GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR scale, GM_ADDR offset, GM_ADDR y,
        const NormCommonTilingData1& tiling_data)
    {
        num_core_ = tiling_data.numCore;
        num_col_ = tiling_data.numCol;
        avg_factor_ = tiling_data.avgFactor;
        epsilon_ = tiling_data.epsilon;
        slice_size_ = tiling_data.sliceSize;
        uint32_t num_row = tiling_data.numRow;
        uint32_t row_work = (num_row + num_core_ - 1) / num_core_;
        if (AscendC::GetBlockIdx() != num_core_ - 1) {
            row_work_ = row_work;
        } else {
            row_work_ = num_row - (num_core_ - 1) * row_work;
        }
        gm_offset_ = static_cast<uint64_t>(row_work) * num_col_;
        if (num_col_ <= slice_size_) {
            num_col_align_int8 = (num_col_ + REPEAT_TIME_256 - 1) / REPEAT_TIME_256 * REPEAT_TIME_256;
            num_col_align_int4 = (num_col_ + REPEAT_TIME_256 - 1) / REPEAT_TIME_256 * REPEAT_TIME_256;
            num_col_align_f16 = (num_col_ + REPEAT_TIME_16 - 1) / REPEAT_TIME_16 * REPEAT_TIME_16;
            num_col_align_f32 = (num_col_ + REPEAT_TIME_64 - 1) / REPEAT_TIME_64 * REPEAT_TIME_64;
        } else {
            num_col_align_int8 = slice_size_;
            num_col_align_int4 = slice_size_;
            num_col_align_f16 = slice_size_;
            num_col_align_f32 = slice_size_;
            num_col_align_f32_long = (num_col_ + REPEAT_TIME_64 - 1) / REPEAT_TIME_64 * REPEAT_TIME_64;
        }
        quantMin_ = tiling_data.quantMin;
        dstType = tiling_data.dstType;
        gm_x_.SetGlobalBuffer((__gm__ T*)x + AscendC::GetBlockIdx() * gm_offset_);
        gm_g_.SetGlobalBuffer((__gm__ T*)gamma);
        gm_b_.SetGlobalBuffer((__gm__ T*)beta);
        gm_y_.SetGlobalBuffer((__gm__ int8_t*)y + AscendC::GetBlockIdx() * gm_offset_);
        gm_y_int4.SetGlobalBuffer((__gm__ int4b_t*)y + AscendC::GetBlockIdx() * gm_offset_ / 2);

        pipe.InitBuffer(fp16_x_que_, BUFFER_NUM, num_col_align_f16 * sizeof(T));
        pipe.InitBuffer(int8_y_que_, BUFFER_NUM, num_col_align_int8 * sizeof(int8_t)); // quant output
        pipe.InitBuffer(int4_y_que_, BUFFER_NUM, num_col_align_int4 * sizeof(int8_t) / SIZE_INT4); // quant output
        pipe.InitBuffer(fp32_xy_buf_, num_col_align_f32 * sizeof(float));
        pipe.InitBuffer(fp16_buf_, num_col_align_f16 * sizeof(T));
        pipe.InitBuffer(calc_buf_, BUF_FACTOR * num_col_align_f32 * sizeof(float) + 32); // 32 for sum
        GetScaleAndOffset<T>(input_scale_, input_offset_, scale, offset, fp16_buf_);
    }

    __aicore__ inline void Launch()
    {
        if constexpr (FastComputeMode) {
            FastCompute();
        } else {
            SliceCompute();
        }
    }

private:
    __aicore__ inline void CopyIn(uint64_t offset, uint32_t numel)
    {
        AscendC::LocalTensor<T> fp16_x = fp16_x_que_.AllocTensor<T>();
        DataCopyCustom<T>(fp16_x, gm_x_[offset], numel);
        fp16_x_que_.EnQue(fp16_x);
    }

    __aicore__ inline void FastCompute()
    {
        AscendC::LocalTensor<T> fp16_g = fp32_xy_buf_.Get<T>(num_col_align_f32);
        AscendC::LocalTensor<float> fp32_g = calc_buf_.Get<float>(num_col_align_f16);
        DataCopyCustom<T>(fp16_g, gm_g_, num_col_);
        AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Cast(fp32_g[OFFSET_GAMMA * num_col_align_f32], fp16_g, AscendC::RoundMode::CAST_NONE, num_col_);
        AscendC::SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
        if constexpr (EN_BETA) {
            AscendC::LocalTensor<T> fp16_buffer = fp16_buf_.Get<T>();
            DataCopyCustom<T>(fp16_buffer, gm_b_, num_col_);
        }
        uint64_t pid = 0;
        while (pid < row_work_) {
            uint64_t offset = pid * num_col_;
            CopyIn(offset, num_col_);
            Compute();
            CopyOut(offset, num_col_);
            ++pid;
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void SliceCompute()
    {
        num_slice_ = (num_col_align_f32_long + slice_size_ - 1) / slice_size_;
        tail_copy_ = num_col_ - (num_slice_ - 1) * slice_size_;
        for (uint64_t pid = 0; pid < row_work_; pid++) {
            uint64_t row_offset = pid * num_col_;
            float squareSum = 0.0f;
            for (uint64_t sid = 0; sid < num_slice_; sid++) {
                uint64_t col_offset = row_offset + sid * slice_size_;
                uint32_t copyNum = (sid == (num_slice_ - 1)) ? tail_copy_ : slice_size_;
                CopyIn(col_offset, copyNum);
                AscendC::PipeBarrier<PIPE_ALL>();
                squareSum += ComputeSquareSum(copyNum);
            }
            float avg = squareSum * avg_factor_;
            float rms = sqrt(avg + epsilon_);
            float factor = (rms != 0) ? 1 / rms : 1.0f; // 避免除0问题
            for (uint64_t sid = 0; sid < num_slice_; sid++) {
                uint64_t sliceOffset = sid * slice_size_;
                uint32_t copyNum = (sid == (num_slice_ - 1)) ? tail_copy_ : slice_size_;
                uint64_t totalOffset = row_offset + sliceOffset;
                if constexpr (EN_BETA) {
                    AscendC::LocalTensor<T> fp16_buffer = fp16_buf_.Get<T>();
                    DataCopyCustom<T>(fp16_buffer, gm_b_[sliceOffset], copyNum);
                }
                CopyInGama(sliceOffset, copyNum);
                AscendC::PipeBarrier<PIPE_ALL>();
                CopyIn(totalOffset, copyNum);
                AscendC::PipeBarrier<PIPE_ALL>();
                ComputeNormandQuant(factor, copyNum);
                AscendC::PipeBarrier<PIPE_ALL>();
                CopyOut(totalOffset, copyNum);
                AscendC::PipeBarrier<PIPE_ALL>();
            }
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }

    __aicore__ inline void CopyInGama(int64_t sliceOffset, uint32_t numel)
    {
        AscendC::LocalTensor<T> fp16_g = fp32_xy_buf_.Get<T>(slice_size_);
        AscendC::LocalTensor<float> fp32_g = calc_buf_.Get<float>(slice_size_);
        DataCopyCustom<T>(fp16_g, gm_g_[sliceOffset], numel);
        AscendC::SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Cast(fp32_g[OFFSET_GAMMA * slice_size_], fp16_g, AscendC::RoundMode::CAST_NONE, numel);
        AscendC::SetFlag<HardEvent::V_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::V_MTE2>(EVENT_ID0);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<T> fp16_x = fp16_x_que_.DeQue<T>();
        AscendC::LocalTensor<float> fp32_xy = fp32_xy_buf_.Get<float>();
        AscendC::LocalTensor<float> buf = calc_buf_.Get<float>();
        AscendC::LocalTensor<float> g = buf[OFFSET_GAMMA * num_col_align_f32];       // 0
        AscendC::LocalTensor<float> sqx = buf[OFFSET_SQX * num_col_align_f32];       // 1
        AscendC::LocalTensor<float> work = buf[OFFSET_SUM * num_col_align_f32];      // 2
        AscendC::LocalTensor<float> sum = buf[OFFSET_WORKSPACE * num_col_align_f32]; // 4
        Cast(fp32_xy, fp16_x, AscendC::RoundMode::CAST_NONE, num_col_);
        AscendC::PipeBarrier<PIPE_V>();
        Mul(sqx, fp32_xy, fp32_xy, num_col_);
        AscendC::PipeBarrier<PIPE_V>();
        Muls(sqx, sqx, avg_factor_, num_col_);
        AscendC::PipeBarrier<PIPE_V>();
        ReduceSumCustom(sum, sqx, work, num_col_);
        AscendC::PipeBarrier<PIPE_V>();
        Adds(sum, sum, epsilon_, 1);
        AscendC::PipeBarrier<PIPE_V>();
        Sqrt(sum, sum, 1);
        AscendC::SetFlag<HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::V_S>(EVENT_ID0);
        float factor = 1 / sum.GetValue(0);
        AscendC::SetFlag<HardEvent::S_V>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::S_V>(EVENT_ID0);
        Muls(fp32_xy, fp32_xy, factor, num_col_);
        AscendC::PipeBarrier<PIPE_V>();
        Mul(fp32_xy, fp32_xy, g, num_col_);
        AscendC::PipeBarrier<PIPE_V>();
        if constexpr (EN_BETA) { // quant的beta是fp16加的
            AscendC::LocalTensor<T> b = fp16_buf_.Get<T>();
            Cast(work, b, AscendC::RoundMode::CAST_NONE, num_col_);
            AscendC::PipeBarrier<PIPE_V>();
            Add(fp32_xy, fp32_xy, work, num_col_);
            AscendC::PipeBarrier<PIPE_V>();
        }
        Muls(fp32_xy, fp32_xy, input_scale_, num_col_);
        AscendC::PipeBarrier<PIPE_V>();
        Adds(fp32_xy, fp32_xy, input_offset_, num_col_);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::LocalTensor<half> tmpfp16Buf = calc_buf_.Get<half>();
        AscendC::LocalTensor<half> tmpfp16 =
            tmpfp16Buf[OFFSET_SUM * num_col_align_f32 * DIM_2]; // 2：work,float偏移到half
        CastFrom32To16(tmpfp16, fp32_xy, num_col_);
        if constexpr(IsSameType<yDtype, int4b_t>::value) {
            AscendC::LocalTensor<int4b_t> int4_y = int4_y_que_.AllocTensor<int4b_t>();
            Cast(int4_y, tmpfp16, RoundMode::CAST_RINT, num_col_);
            int4_y_que_.EnQue(int4_y);
        } else {
            AscendC::LocalTensor<int8_t> int8_y = int8_y_que_.AllocTensor<int8_t>();
            CastFromF16ToI8(int8_y, tmpfp16, quantMin_, num_col_);
            int8_y_que_.EnQue(int8_y);
        }
        fp16_x_que_.FreeTensor(fp16_x);
    }

    __aicore__ inline float ComputeSquareSum(uint32_t numel)
    {
        AscendC::LocalTensor<T> fp16_x = fp16_x_que_.DeQue<T>();
        AscendC::LocalTensor<float> fp32_xy = fp32_xy_buf_.Get<float>();
        AscendC::LocalTensor<float> buf = calc_buf_.Get<float>();
        AscendC::LocalTensor<float> sqx = buf[OFFSET_SQX * slice_size_];       // 1
        AscendC::LocalTensor<float> work = buf[OFFSET_SUM * slice_size_];      // 2
        AscendC::LocalTensor<float> sum = buf[OFFSET_WORKSPACE * slice_size_]; // 4
        Cast(fp32_xy, fp16_x, AscendC::RoundMode::CAST_NONE, numel);
        AscendC::PipeBarrier<PIPE_V>();
        Mul(sqx, fp32_xy, fp32_xy, numel);
        AscendC::PipeBarrier<PIPE_V>();
#if __CCE_AICORE__ == 100
        ReduceSumCustom(sum, sqx, work, numel);
#else
        ReduceSum(sum, sqx, work, numel);
#endif
        AscendC::SetFlag<HardEvent::V_S>(EVENT_ID0);
        AscendC::WaitFlag<HardEvent::V_S>(EVENT_ID0);
        fp16_x_que_.FreeTensor(fp16_x);
        float sumall = sum.GetValue(0);
        return sumall;
    }

    __aicore__ inline void ComputeNormandQuant(float factor, uint32_t num)
    {
        AscendC::LocalTensor<T> fp16_x = fp16_x_que_.DeQue<T>();
        AscendC::LocalTensor<float> fp32_xy = fp32_xy_buf_.Get<float>();
        AscendC::LocalTensor<float> buf = calc_buf_.Get<float>();
        AscendC::LocalTensor<float> g = buf[OFFSET_GAMMA * slice_size_];  // 0
        AscendC::LocalTensor<float> sqx = buf[OFFSET_SQX * slice_size_];  // 1
        AscendC::LocalTensor<float> work = buf[OFFSET_SUM * slice_size_]; // 2
        Cast(fp32_xy, fp16_x, AscendC::RoundMode::CAST_NONE, num);
        AscendC::PipeBarrier<PIPE_V>();
        Muls(fp32_xy, fp32_xy, factor, num);
        AscendC::PipeBarrier<PIPE_V>();
        Mul(fp32_xy, fp32_xy, g, num);
        AscendC::PipeBarrier<PIPE_V>();
        if constexpr (EN_BETA) { // quant的beta是fp16加的
            AscendC::LocalTensor<T> b = fp16_buf_.Get<T>();
            Cast(work, b, AscendC::RoundMode::CAST_NONE, num);
            AscendC::PipeBarrier<PIPE_V>();
            Add(fp32_xy, fp32_xy, work, num);
            AscendC::PipeBarrier<PIPE_V>();
        }
        Muls(fp32_xy, fp32_xy, input_scale_, num);
        AscendC::PipeBarrier<PIPE_V>();
        Adds(fp32_xy, fp32_xy, input_offset_, num);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::LocalTensor<half> tmpfp16Buf = calc_buf_.Get<half>();
        AscendC::LocalTensor<half> tmpfp16 = tmpfp16Buf[OFFSET_SUM * slice_size_ * DIM_2]; // 2：work,float偏移到half
        CastFrom32To16(tmpfp16, fp32_xy, num);
        if constexpr(IsSameType<yDtype, int4b_t>::value) {
            AscendC::LocalTensor<int4b_t> int4_y = int4_y_que_.AllocTensor<int4b_t>();
            Cast(int4_y, tmpfp16, RoundMode::CAST_RINT, num);
            int4_y_que_.EnQue(int4_y);
        } else {
            AscendC::LocalTensor<int8_t> int8_y = int8_y_que_.AllocTensor<int8_t>();
            CastFromF16ToI8(int8_y, tmpfp16, quantMin_, num);
            int8_y_que_.EnQue(int8_y);
        }
        fp16_x_que_.FreeTensor(fp16_x);
    }

    __aicore__ inline void CopyOut(int64_t offset, uint32_t numel)
    {
        if constexpr(IsSameType<yDtype, int4b_t>::value) {
            AscendC::LocalTensor<int4b_t> int4_y = int4_y_que_.DeQue<int4b_t>();
            DataCopyParams copyParams;
            copyParams.blockLen = numel * sizeof(int4b_t) / SIZE_INT4;
            copyParams.blockCount = 1;
            DataCopyPad(gm_y_int4[offset], int4_y, copyParams);
            int4_y_que_.FreeTensor(int4_y);
        } else {
            AscendC::LocalTensor<int8_t> int8_y = int8_y_que_.DeQue<int8_t>();
            DataCopyCustom<int8_t>(gm_y_[offset], int8_y, numel);
            int8_y_que_.FreeTensor(int8_y);
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> fp16_x_que_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> int8_y_que_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> int4_y_que_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> fp32_xy_buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> calc_buf_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> fp16_buf_;

    AscendC::GlobalTensor<T> gm_x_;
    AscendC::GlobalTensor<T> gm_g_;
    AscendC::GlobalTensor<T> gm_b_;
    AscendC::GlobalTensor<int8_t> gm_y_;
    AscendC::GlobalTensor<int4b_t> gm_y_int4;
    uint32_t num_core_{0};   // 一共激活多少AICORE
    uint32_t num_col_{0};    // 输入的列数
    uint32_t row_work_{0};   // 需要计算多少行
    uint32_t row_step_{0};   // 除最后一次，每次搬入多少行
    uint32_t row_tail_{0};   // 最后一次搬入多少行数据
    uint64_t gm_offset_{0};  // GM数据起始位置偏移量
    float avg_factor_{1.0};  // num_col_的倒数
    float input_scale_{1.0}; // 非对称量化系数
    float input_offset_{0};  // 非对称量化偏移适配高精度
    float epsilon_{1e-12f};  // norm平滑参数
    uint32_t num_col_align_int8{0};
    uint32_t num_col_align_int4{0};
    uint32_t num_col_align_f16{0};
    uint32_t num_col_align_f32{0};
    uint32_t num_col_align_f32_long{0};
    uint32_t num_col_temp;
    half quantMin_{-128};
    uint32_t slice_size_{0};
    uint32_t num_slice_{0};
    uint32_t tail_copy_{0};
    int32_t dstType{29};
};

extern "C" __global__ __aicore__ void rms_norm_quant(
    GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR scale, GM_ADDR offset, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    if (TILING_KEY_IS(257)) { // fp16, beta, gamma, beta, no slice 0b0_1_0_0_000001
        RmsNormQuant<half, DTYPE_Y, true, true> kernel(x, gamma, beta, scale, offset, y, tiling_data);
        kernel.Launch();
    }
    if (TILING_KEY_IS(1)) { // fp16, empty beta, gamma, beta, no slice  0b0_0_0_0_000001
        RmsNormQuant<half, DTYPE_Y, false, true> kernel(x, gamma, beta, scale, offset, y, tiling_data);
        kernel.Launch();
    }
    if (TILING_KEY_IS(321)) { // fp16, beta, gamma, beta, use slice  0b0_1_0_1_000001
        RmsNormQuant<half, DTYPE_Y, true, false> kernel(x, gamma, beta, scale, offset, y, tiling_data);
        kernel.Launch();
    }
    if (TILING_KEY_IS(65)) { // fp16, empty beta, gamma, beta, use slice  0b0_0_0_1_000001
        RmsNormQuant<half, DTYPE_Y, false, false> kernel(x, gamma, beta, scale, offset, y, tiling_data);
        kernel.Launch();
    }
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    if (TILING_KEY_IS(283)) { // bf16, beta, gamma, beta, no slice  0b0_1_0_0_011011
        RmsNormQuant<bfloat16_t, DTYPE_Y, true, true> kernel(x, gamma, beta, scale, offset, y, tiling_data);
        kernel.Launch();
    }
    if (TILING_KEY_IS(27)) { // bf16, empty beta, gamma, beta, no slice  0b0_0_0_0_011011
        RmsNormQuant<bfloat16_t, DTYPE_Y, false, true> kernel(x, gamma, beta, scale, offset, y, tiling_data);
        kernel.Launch();
    }
    if (TILING_KEY_IS(347)) { // bf16, beta, gamma, beta, use slice  0b0_1_0_1_011011
        RmsNormQuant<bfloat16_t, DTYPE_Y, true, false> kernel(x, gamma, beta, scale, offset, y, tiling_data);
        kernel.Launch();
    }
    if (TILING_KEY_IS(91)) { // bf16, empty beta, gamma, beta, use slice  0b0_0_0_1_011011
        RmsNormQuant<bfloat16_t, DTYPE_Y, false, false> kernel(x, gamma, beta, scale, offset, y, tiling_data);
        kernel.Launch();
    }
#endif
}