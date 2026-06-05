/*
 * Copyright (c) 2026 联通（广东）产业互联网有限公司.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "kernel_operator.h"
#include "intern_vl_add_rms_norm_tiling.h"
#include <type_traits>

using namespace AscendC;

constexpr uint32_t NUM_PER_REP_FP32 = 64;
constexpr uint32_t REDUCE_BLOCK_ELEMS_FP32 = 8;

template <typename T>
class KernelInternVlAddRmsNorm {
public:
    __aicore__ inline KernelInternVlAddRmsNorm() {}

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR residual, GM_ADDR gamma,
        GM_ADDR y, GM_ADDR residual_out,
        __gm__ optiling::InternVlAddRmsNormTilingData* tilingData) {
        hidden_size = tilingData->hidden_size;
        eps = tilingData->eps;
        tile_size = tilingData->tile_size;
        tile_num = tilingData->tile_num;

        uint32_t core_idx = GetBlockIdx();
        core_start_token = core_idx * tilingData->tokens_per_core;
        core_end_token = min(core_start_token + tilingData->tokens_per_core,
                             tilingData->batch);
        core_token_num = (core_end_token > core_start_token) ?
                         (core_end_token - core_start_token) : 0;

        x_gm.SetGlobalBuffer((__gm__ T*)x);
        residual_gm.SetGlobalBuffer((__gm__ T*)residual);
        gamma_gm.SetGlobalBuffer((__gm__ T*)gamma);
        y_gm.SetGlobalBuffer((__gm__ T*)y);
        residual_out_gm.SetGlobalBuffer((__gm__ T*)residual_out);

        type_buf_size = ((tile_size * sizeof(T) + 31) / 32) * 32;
        type_buf_elems = type_buf_size / sizeof(T);
        float_buf_size = ((tile_size * sizeof(float) + 31) / 32) * 32;
        float_buf_elems = float_buf_size / sizeof(float);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, type_buf_size);
        pipe.InitBuffer(inQueueResidual, BUFFER_NUM, type_buf_size);
        pipe.InitBuffer(inQueueGamma, BUFFER_NUM, type_buf_size);
        pipe.InitBuffer(outQueueResOut, BUFFER_NUM, type_buf_size);
        pipe.InitBuffer(outQueueY, BUFFER_NUM, type_buf_size);
        pipe.InitBuffer(tmpBuf, float_buf_size * 3);
    }

    __aicore__ inline void Process() {
        if (core_token_num == 0) {
            return;
        }
        if (tile_num == 1) {
            ProcessTileNumOne();
            return;
        }
        ProcessMultiTile();
    }

private:
    __aicore__ inline uint32_t GetCurTileSize(uint32_t tile_idx) const {
        return tile_idx == tile_num - 1 ? hidden_size - tile_idx * tile_size : tile_size;
    }

    __aicore__ inline void ComputeResidualSquare(LocalTensor<T>& res_out_local,
        LocalTensor<T>& x_local, LocalTensor<T>& res_local, LocalTensor<float>& f_local,
        LocalTensor<float>& sq_local, LocalTensor<float>& f_res, uint32_t cur_tile_size) {
        if constexpr (std::is_same<T, bfloat16_t>::value) {
            Cast(f_local, x_local, RoundMode::CAST_NONE, cur_tile_size);
            Cast(f_res, res_local, RoundMode::CAST_NONE, cur_tile_size);
            PipeBarrier<PIPE_V>();
            Add(f_local, f_local, f_res, cur_tile_size);
            PipeBarrier<PIPE_V>();
            Cast(res_out_local, f_local, RoundMode::CAST_RINT, cur_tile_size);
            PipeBarrier<PIPE_V>();
            Cast(f_local, res_out_local, RoundMode::CAST_NONE, cur_tile_size);
        } else if constexpr (std::is_same<T, float>::value) {
            Add(res_out_local, x_local, res_local, cur_tile_size);
        } else if constexpr (std::is_same<T, half>::value) {
            Add(res_out_local, x_local, res_local, cur_tile_size);
            PipeBarrier<PIPE_V>();
            Cast(f_local, res_out_local, RoundMode::CAST_NONE, cur_tile_size);
        }
        PipeBarrier<PIPE_V>();
        if constexpr (std::is_same<T, float>::value) {
            Mul(sq_local, res_out_local, res_out_local, cur_tile_size);
        } else {
            Mul(sq_local, f_local, f_local, cur_tile_size);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeOutputFromInput(LocalTensor<T>& y_local,
        LocalTensor<T>& x_local, LocalTensor<T>& res_local, LocalTensor<T>& gamma_local,
        LocalTensor<float>& f_res, LocalTensor<float>& f_gamma, LocalTensor<float>& f_y,
        uint32_t cur_tile_size, float inv_rms) {
        if constexpr (std::is_same<T, float>::value) {
            Add(y_local, x_local, res_local, cur_tile_size);
            PipeBarrier<PIPE_V>();
            Muls(y_local, y_local, inv_rms, cur_tile_size);
            PipeBarrier<PIPE_V>();
            Mul(y_local, y_local, gamma_local, cur_tile_size);
            return;
        }
        if constexpr (std::is_same<T, bfloat16_t>::value) {
            Cast(f_res, x_local, RoundMode::CAST_NONE, cur_tile_size);
            Cast(f_y, res_local, RoundMode::CAST_NONE, cur_tile_size);
            PipeBarrier<PIPE_V>();
            Add(f_res, f_res, f_y, cur_tile_size);
        } else if constexpr (std::is_same<T, half>::value) {
            Add(y_local, x_local, res_local, cur_tile_size);
            PipeBarrier<PIPE_V>();
            Cast(f_res, y_local, RoundMode::CAST_NONE, cur_tile_size);
        }
        Cast(f_gamma, gamma_local, RoundMode::CAST_NONE, cur_tile_size);
        PipeBarrier<PIPE_V>();
        Muls(f_y, f_res, inv_rms, cur_tile_size);
        PipeBarrier<PIPE_V>();
        Mul(f_y, f_y, f_gamma, cur_tile_size);
        PipeBarrier<PIPE_V>();
        Cast(y_local, f_y, std::is_same<T, bfloat16_t>::value ? RoundMode::CAST_RINT : RoundMode::CAST_NONE,
            cur_tile_size);
    }

    __aicore__ inline void ComputeOutputFromResidual(LocalTensor<T>& y_local,
        LocalTensor<T>& res_out_local, LocalTensor<T>& gamma_local, LocalTensor<float>& f_local,
        LocalTensor<float>& f_gamma, LocalTensor<float>& f_y, uint32_t cur_tile_size, float inv_rms) {
        if constexpr (std::is_same<T, float>::value) {
            Muls(y_local, res_out_local, inv_rms, cur_tile_size);
            PipeBarrier<PIPE_V>();
            Mul(y_local, y_local, gamma_local, cur_tile_size);
            return;
        }
        Cast(f_gamma, gamma_local, RoundMode::CAST_NONE, cur_tile_size);
        PipeBarrier<PIPE_V>();
        Muls(f_y, f_local, inv_rms, cur_tile_size);
        PipeBarrier<PIPE_V>();
        Mul(f_y, f_y, f_gamma, cur_tile_size);
        PipeBarrier<PIPE_V>();
        Cast(y_local, f_y, std::is_same<T, bfloat16_t>::value ? RoundMode::CAST_RINT : RoundMode::CAST_NONE,
            cur_tile_size);
    }

    __aicore__ inline float ProcessResidualPass(uint64_t token_offset) {
        float token_sum_sq = 0.0f;
        for (uint32_t t = 0; t < tile_num; ++t) {
            uint32_t cur_tile_size = GetCurTileSize(t);
            uint64_t cur_offset = token_offset + t * tile_size;
            LocalTensor<T> x_in = inQueueX.AllocTensor<T>();
            LocalTensor<T> res_in = inQueueResidual.AllocTensor<T>();
            DataCopy(x_in, x_gm[cur_offset], cur_tile_size);
            DataCopy(res_in, residual_gm[cur_offset], cur_tile_size);
            inQueueX.EnQue(x_in);
            inQueueResidual.EnQue(res_in);
            LocalTensor<T> x_local = inQueueX.DeQue<T>();
            LocalTensor<T> res_local = inQueueResidual.DeQue<T>();
            LocalTensor<T> res_out_local = outQueueResOut.AllocTensor<T>();
            LocalTensor<float> tmp_local = tmpBuf.Get<float>();
            LocalTensor<float> f_local = tmp_local[0];
            LocalTensor<float> sq_local = tmp_local[float_buf_elems];
            LocalTensor<float> f_res = tmp_local[float_buf_elems * 2];
            ComputeResidualSquare(res_out_local, x_local, res_local, f_local, sq_local, f_res, cur_tile_size);
            token_sum_sq += VectorReduceSum(sq_local, f_res, cur_tile_size);
            outQueueResOut.EnQue(res_out_local);
            LocalTensor<T> res_out = outQueueResOut.DeQue<T>();
            DataCopy(residual_out_gm[cur_offset], res_out, cur_tile_size);
            inQueueX.FreeTensor(x_local);
            inQueueResidual.FreeTensor(res_local);
            outQueueResOut.FreeTensor(res_out);
        }
        return token_sum_sq;
    }

    __aicore__ inline void ProcessOutputPass(uint64_t token_offset, float inv_rms) {
        for (uint32_t t = 0; t < tile_num; ++t) {
            uint32_t cur_tile_size = GetCurTileSize(t);
            uint64_t cur_offset = token_offset + t * tile_size;

            LocalTensor<T> res_out_in = inQueueResidual.AllocTensor<T>();
            LocalTensor<T> gamma_in = inQueueGamma.AllocTensor<T>();
            DataCopy(res_out_in, residual_out_gm[cur_offset], cur_tile_size);
            DataCopy(gamma_in, gamma_gm[t * tile_size], cur_tile_size);

            inQueueResidual.EnQue(res_out_in);
            inQueueGamma.EnQue(gamma_in);

            LocalTensor<T> res_out_local = inQueueResidual.DeQue<T>();
            LocalTensor<T> gamma_local = inQueueGamma.DeQue<T>();
            LocalTensor<T> y_local = outQueueY.AllocTensor<T>();

            LocalTensor<float> tmp_local = tmpBuf.Get<float>();
            LocalTensor<float> f_res = tmp_local[0];
            LocalTensor<float> f_gamma = tmp_local[float_buf_elems];
            LocalTensor<float> f_y = tmp_local[float_buf_elems * 2];

            if constexpr (!std::is_same<T, float>::value) {
                Cast(f_res, res_out_local, RoundMode::CAST_NONE, cur_tile_size);
                PipeBarrier<PIPE_V>();
            }

            ComputeOutputFromResidual(y_local, res_out_local, gamma_local, f_res, f_gamma, f_y, cur_tile_size,
                inv_rms);

            PipeBarrier<PIPE_V>();
            outQueueY.EnQue(y_local);
            LocalTensor<T> y_out = outQueueY.DeQue<T>();
            DataCopy(y_gm[cur_offset], y_out, cur_tile_size);

            inQueueResidual.FreeTensor(res_out_local);
            inQueueGamma.FreeTensor(gamma_local);
            outQueueY.FreeTensor(y_out);
        }
    }

    __aicore__ inline void ProcessMultiTile() {
        for (uint32_t token = 0; token < core_token_num; ++token) {
            uint32_t token_idx = core_start_token + token;
            uint64_t token_offset = static_cast<uint64_t>(token_idx) * hidden_size;
            float token_sum_sq = ProcessResidualPass(token_offset);
            float inv_rms = 1.0f / sqrt(token_sum_sq / hidden_size + eps);
            ProcessOutputPass(token_offset, inv_rms);
        }
    }

private:
    __aicore__ inline void ProcessTileNumOne() {
        for (uint32_t token = 0; token < core_token_num; ++token) {
            uint32_t token_idx = core_start_token + token;
            ProcessSingleTileToken(static_cast<uint64_t>(token_idx) * hidden_size);
        }
    }

    __aicore__ inline void ProcessSingleTileToken(uint64_t token_offset) {
        uint32_t cur_tile_size = hidden_size;
        LocalTensor<T> x_in = inQueueX.AllocTensor<T>();
        LocalTensor<T> res_in = inQueueResidual.AllocTensor<T>();
        LocalTensor<T> gamma_in = inQueueGamma.AllocTensor<T>();
        DataCopy(x_in, x_gm[token_offset], cur_tile_size);
        DataCopy(res_in, residual_gm[token_offset], cur_tile_size);
        DataCopy(gamma_in, gamma_gm[0], cur_tile_size);
        inQueueX.EnQue(x_in);
        inQueueResidual.EnQue(res_in);
        inQueueGamma.EnQue(gamma_in);
        LocalTensor<T> x_local = inQueueX.DeQue<T>();
        LocalTensor<T> res_local = inQueueResidual.DeQue<T>();
        LocalTensor<T> gamma_local = inQueueGamma.DeQue<T>();
        LocalTensor<T> res_out_local = outQueueResOut.AllocTensor<T>();
        LocalTensor<T> y_local = outQueueY.AllocTensor<T>();
        LocalTensor<float> tmp_local = tmpBuf.Get<float>();
        LocalTensor<float> f_local = tmp_local[0];
        LocalTensor<float> sq_local = tmp_local[float_buf_elems];
        LocalTensor<float> f_res = tmp_local[float_buf_elems * 2];
        ComputeResidualSquare(res_out_local, x_local, res_local, f_local, sq_local, f_res, cur_tile_size);
        float token_sum_sq = VectorReduceSum(sq_local, f_res, cur_tile_size);
        float inv_rms = 1.0f / sqrt(token_sum_sq / hidden_size + eps);
        ComputeOutputFromResidual(y_local, res_out_local, gamma_local, f_local, sq_local, f_res, cur_tile_size,
            inv_rms);
        outQueueResOut.EnQue(res_out_local);
        LocalTensor<T> res_out = outQueueResOut.DeQue<T>();
        DataCopy(residual_out_gm[token_offset], res_out, cur_tile_size);
        outQueueY.EnQue(y_local);
        LocalTensor<T> y_out = outQueueY.DeQue<T>();
        DataCopy(y_gm[token_offset], y_out, cur_tile_size);
        inQueueX.FreeTensor(x_local);
        inQueueResidual.FreeTensor(res_local);
        inQueueGamma.FreeTensor(gamma_local);
        outQueueResOut.FreeTensor(res_out);
        outQueueY.FreeTensor(y_out);
    }

private:
    __aicore__ inline float VectorReduceSum(LocalTensor<float>& vec, LocalTensor<float>& work, uint32_t len) {
        uint32_t repeat_times = len / NUM_PER_REP_FP32;
        uint32_t tail_count = len % NUM_PER_REP_FP32;
        uint32_t reduce_elems = repeat_times * REDUCE_BLOCK_ELEMS_FP32;

        if (repeat_times > 0) {
            BlockReduceSum(work, vec, repeat_times, NUM_PER_REP_FP32, 1, 1, DEFAULT_REPEAT_STRIDE);
            PipeBarrier<PIPE_V>();
        }

        if (tail_count != 0) {
            uint32_t src_offset = repeat_times * NUM_PER_REP_FP32;
            Duplicate(work[reduce_elems], 0.0f, REDUCE_BLOCK_ELEMS_FP32);
            PipeBarrier<PIPE_V>();
            BlockReduceSum(work[reduce_elems], vec[src_offset], 1, tail_count, 1, 1, DEFAULT_REPEAT_STRIDE);
            PipeBarrier<PIPE_V>();
            reduce_elems += REDUCE_BLOCK_ELEMS_FP32;
        }

        event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventVS);
        WaitFlag<HardEvent::V_S>(eventVS);

        float sum = 0.0f;
        for (uint32_t i = 0; i < reduce_elems; ++i) {
            sum += work.GetValue(i);
        }

        event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventSV);
        WaitFlag<HardEvent::S_V>(eventSV);

        return sum;
    }

    GlobalTensor<T> x_gm;
    GlobalTensor<T> residual_gm;
    GlobalTensor<T> gamma_gm;
    GlobalTensor<T> y_gm;
    GlobalTensor<T> residual_out_gm;

    TPipe pipe;
    TQue<QuePosition::VECIN, 2> inQueueX;
    TQue<QuePosition::VECIN, 2> inQueueResidual;
    TQue<QuePosition::VECIN, 2> inQueueGamma;
    TQue<QuePosition::VECOUT, 2> outQueueResOut;
    TQue<QuePosition::VECOUT, 2> outQueueY;
    TBuf<TPosition::VECCALC> tmpBuf;

    uint32_t hidden_size;
    float eps;
    uint32_t tile_size, tile_num;
    uint32_t core_start_token, core_end_token, core_token_num;
    uint32_t type_buf_size, type_buf_elems;
    uint32_t float_buf_size, float_buf_elems;

    static constexpr uint32_t BUFFER_NUM = 2;
};

extern "C" __global__ __aicore__ void intern_vl_add_rms_norm(
    GM_ADDR x, GM_ADDR residual, GM_ADDR gamma,
    GM_ADDR y, GM_ADDR residual_out,
    GM_ADDR workspace, GM_ADDR tiling) {
    auto tilingData = reinterpret_cast<__gm__ optiling::InternVlAddRmsNormTilingData*>(tiling);

    if (TILING_KEY_IS(10)) {
        KernelInternVlAddRmsNorm<half> op;
        op.Init(x, residual, gamma, y, residual_out, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(20)) {
        KernelInternVlAddRmsNorm<float> op;
        op.Init(x, residual, gamma, y, residual_out, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(30)) {
#if !(defined(__NPU_ARCH__) && __NPU_ARCH__ == 3003)
        KernelInternVlAddRmsNorm<bfloat16_t> op;
        op.Init(x, residual, gamma, y, residual_out, tilingData);
        op.Process();
#endif
    }
}
