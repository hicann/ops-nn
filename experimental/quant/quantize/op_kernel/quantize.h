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
 * \file quantize.h
 * \brief Quantize kernel base + per-tensor / per-channel impl, ascend910b (DAV_2201) standard model.
 *
 * y = saturate_cast_to(DTYPE_Y)( round_to_nearest( x / scale + zero_point ) )
 * fixed modes: div + round-to-nearest(rint) + no-sqrt.
 *
 * Unified fp32 compute path (avoids the dtype cartesian blow-up):
 *   x(DTYPE_X)   -> fp32 (Cast, or direct when already fp32)
 *   scale        -> fp32 scalar (per row), broadcast into a buffer, true Div
 *   zero_point   -> fp32 scalar (per row), Adds  (skipped when absent)
 *   fp32         -> DTYPE_Y with round-to-nearest + saturation (CastOut ladder)
 *
 * zero_points dtype note: kernel selection keys on (x, scales, y) dtypes ONLY; the optional
 * zero_points dtype is NOT part of the selection key, so one dispatched binary must serve every
 * zero_points dtype. The binary's compile-time Z is therefore unreliable for reading zero_points
 * (an int8-compiled binary would misread an int32 buffer with the wrong element size/stride). We
 * instead read zero_points by its real runtime dtype (carried in tiling.zpDtype), reinterpreting
 * the raw GM buffer accordingly — see ReadOffsetFp32.
 */

#ifndef QUANTIZE_KERNEL_H
#define QUANTIZE_KERNEL_H

#include "kernel_operator.h"
#include "kernel_operator_intf.h"

namespace QuantizeOp {
using namespace AscendC;

// T = x dtype, S = scales dtype, Z = zero_points dtype (compile-time hint only, see ReadOffsetFp32), Y = y dtype
template <typename T, typename S, typename Z, typename Y>
class QuantizeBase {
public:
    __aicore__ inline QuantizeBase() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR scales, GM_ADDR zeroPoints, GM_ADDR y,
                                const QuantizeTilingData* tilingData)
    {
        blockIdx_ = GetBlockIdx();
        tiling_ = *tilingData;

        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
        scaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ S*>(scales));
        // Keep the raw zero_points address; it is reinterpreted by its runtime dtype in ReadOffsetFp32.
        zpBase_ = zeroPoints;
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ Y*>(y));

        int64_t baseLen = tiling_.baseLen;
        pipe_.InitBuffer(inQueueX_, BUFFER_NUM, baseLen * sizeof(T));
        pipe_.InitBuffer(outQueueY_, BUFFER_NUM, baseLen * sizeof(Y));
        pipe_.InitBuffer(calcBuf_, baseLen * sizeof(float));

        // The int8/uint8 saturation ladder's int32->half cast uses the deq-scale register with a fixed
        // unit scale. It is constant across every tile, so set it ONCE here instead of per tile — this
        // removes a SetDeqScale + two PipeBarrier<PIPE_V> from the inner loop of every int8/uint8 case.
        // Harmless (unused) for int32 / fp outputs.
        SetDeqScale(static_cast<half>(1.0f));
    }

protected:
    // ge::DataType codes carried in tiling.zpDtype for the runtime zero_points read.
    static constexpr uint32_t ZP_DT_FLOAT = 0;
    static constexpr uint32_t ZP_DT_FLOAT16 = 1;
    static constexpr uint32_t ZP_DT_INT8 = 2;
    static constexpr uint32_t ZP_DT_INT32 = 3;
    static constexpr uint32_t ZP_DT_UINT8 = 4;
    static constexpr uint32_t ZP_DT_BF16 = 27;

    __aicore__ inline float ScaleToFp32(S v)
    {
        if constexpr (IsSameType<S, bfloat16_t>::value) {
            return ToFloat(v);
        } else {
            return static_cast<float>(v);
        }
    }

    // Read zero_points[index] as fp32 by its REAL runtime dtype (tiling.zpDtype), reinterpreting the raw
    // GM buffer. This is correct regardless of which (x,scales,y) binary was dispatched, because that
    // binary's compile-time Z does not necessarily match the actual zero_points dtype (the selection key
    // omits zero_points). Reading with the wrong element type would use the wrong stride at index > 0 —
    // exactly the per-channel int32-zero_points corruption this fixes.
    __aicore__ inline float ReadOffsetFp32(int64_t index)
    {
        if (tiling_.hasZeroPoint == 0) {
            return 0.0f;
        }
        switch (tiling_.zpDtype) {
            case ZP_DT_INT8: {
                GlobalTensor<int8_t> g;
                g.SetGlobalBuffer(reinterpret_cast<__gm__ int8_t*>(zpBase_));
                return static_cast<float>(g.GetValue(index));
            }
            case ZP_DT_UINT8: {
                GlobalTensor<uint8_t> g;
                g.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(zpBase_));
                // A direct unsigned-scalar -> float cast is rejected on AICore; widen through signed int32
                // first (uint8 is 0..255, exact in both int32 and float).
                return static_cast<float>(static_cast<int32_t>(g.GetValue(index)));
            }
            case ZP_DT_INT32: {
                GlobalTensor<int32_t> g;
                g.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(zpBase_));
                return static_cast<float>(g.GetValue(index));
            }
            case ZP_DT_BF16: {
                GlobalTensor<bfloat16_t> g;
                g.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(zpBase_));
                return ToFloat(g.GetValue(index));
            }
            case ZP_DT_FLOAT16: {
                GlobalTensor<half> g;
                g.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(zpBase_));
                return static_cast<float>(g.GetValue(index));
            }
            case ZP_DT_FLOAT: {
                GlobalTensor<float> g;
                g.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(zpBase_));
                return g.GetValue(index);
            }
            default:
                return 0.0f;
        }
    }

    // fp32 -> Y with round-to-nearest (rint) + saturation, in place on fpLocal.
    // Writes the final result into outLocal; the caller's outQueueY_ EnQue/DeQue provides the V->MTE3
    // sync, so NO trailing PipeBarrier<PIPE_V> is needed here (it would only stall the vector pipe).
    // The deq-scale register is set once in Init(), not per tile.
    __aicore__ inline void CastOut(LocalTensor<Y>& outLocal, LocalTensor<float>& fpLocal, int64_t count)
    {
        if constexpr (IsSameType<Y, int32_t>::value) {
            Cast(outLocal, fpLocal, RoundMode::CAST_RINT, count);
        } else {
            // int8_t / uint8_t : fp32 -> int32(rint) -> half -> int8/uint8(rint, saturating).
            // Barriers between the casts are true RAW hazards (each reads the prior cast's output).
            Cast(fpLocal.ReinterpretCast<int32_t>(), fpLocal, RoundMode::CAST_RINT, count);
            PipeBarrier<PIPE_V>();
            Cast(fpLocal.ReinterpretCast<half>(), fpLocal.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, count);
            PipeBarrier<PIPE_V>();
            Cast(outLocal, fpLocal.ReinterpretCast<half>(), RoundMode::CAST_RINT, count);
        }
    }

    // Reciprocal of the quantization scale, applied as a scalar Muls in place of a vector Div.
    // A zero scale is degenerate (no meaningful quantization step), so fall back to 1.0f instead of
    // dividing by zero — same convention as quant/dequant_swiglu_quant.
    __aicore__ inline float ReciprocalOfScale(float scaleF) { return (scaleF == 0.0f) ? 1.0f : (1.0f / scaleF); }

    // process one contiguous tile [off, off+count) with a scalar reciprocal-scale/offset (fp32).
    // x / scale is computed as x * (1/scale): the reciprocal is an exact IEEE fp32 scalar division
    // (<=0.5 ULP), and the scalar Muls avoids materializing a scale-broadcast buffer + a vector Div.
    // Verified against the stand_quantize (|diff|<=1) golden across the full 208-case ATK set.
    __aicore__ inline void ProcessOneTile(int64_t off, int64_t count, float recipScaleF, float offsetF)
    {
        LocalTensor<T> xLocal = inQueueX_.AllocTensor<T>();
        DataCopyExtParams copyInParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(xLocal, xGm_[off], copyInParams, padParams);
        inQueueX_.EnQue(xLocal);
        xLocal = inQueueX_.DeQue<T>();

        LocalTensor<Y> outLocal = outQueueY_.AllocTensor<Y>();
        LocalTensor<float> calc = calcBuf_.Get<float>();

        if constexpr (IsSameType<T, float>::value) {
            Muls(calc, xLocal, recipScaleF, count);
            PipeBarrier<PIPE_V>();
        } else {
            Cast(calc, xLocal, RoundMode::CAST_NONE, count);
            PipeBarrier<PIPE_V>();
            Muls(calc, calc, recipScaleF, count);
            PipeBarrier<PIPE_V>();
        }
        if (tiling_.hasZeroPoint != 0) {
            Adds(calc, calc, offsetF, count);
            PipeBarrier<PIPE_V>();
        }
        CastOut(outLocal, calc, count);

        inQueueX_.FreeTensor(xLocal);
        outQueueY_.EnQue(outLocal);
        outLocal = outQueueY_.DeQue<Y>();
        DataCopyExtParams copyOutParams{1, static_cast<uint32_t>(count * sizeof(Y)), 0, 0, 0};
        DataCopyPad(yGm_[off], outLocal, copyOutParams);
        outQueueY_.FreeTensor(outLocal);
    }

    static constexpr int32_t BUFFER_NUM = 2;
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY_;
    TBuf<TPosition::VECCALC> calcBuf_;
    GlobalTensor<T> xGm_;
    GlobalTensor<S> scaleGm_;
    GM_ADDR zpBase_ = nullptr;
    GlobalTensor<Y> yGm_;
    QuantizeTilingData tiling_;
    int32_t blockIdx_ = 0;
};

template <typename T, typename S, typename Z, typename Y>
class QuantizePerTensor : public QuantizeBase<T, S, Z, Y> {
public:
    __aicore__ inline QuantizePerTensor() {}

    __aicore__ inline void Process()
    {
        if (static_cast<uint32_t>(this->blockIdx_) >= this->tiling_.numCore) {
            return;
        }
        int64_t blockFactor = this->tiling_.blockFactor;
        int64_t myCount = (static_cast<uint32_t>(this->blockIdx_) == this->tiling_.numCore - 1) ?
                              this->tiling_.blockTailFactor :
                              blockFactor;
        if (myCount <= 0) {
            return;
        }
        int64_t gmOffset = static_cast<int64_t>(this->blockIdx_) * blockFactor;

        float scaleF = this->ScaleToFp32(this->scaleGm_.GetValue(0));
        float recipScaleF = this->ReciprocalOfScale(scaleF);
        float offsetF = this->ReadOffsetFp32(0);

        int64_t baseLen = this->tiling_.baseLen;
        int64_t loopNum = myCount / baseLen;
        int64_t loopTail = myCount % baseLen;
        for (int64_t i = 0; i < loopNum; ++i) {
            this->ProcessOneTile(gmOffset + i * baseLen, baseLen, recipScaleF, offsetF);
        }
        if (loopTail != 0) {
            this->ProcessOneTile(gmOffset + loopNum * baseLen, loopTail, recipScaleF, offsetF);
        }
    }
};

template <typename T, typename S, typename Z, typename Y>
class QuantizePerChannel : public QuantizeBase<T, S, Z, Y> {
public:
    __aicore__ inline QuantizePerChannel() {}

    __aicore__ inline void Process()
    {
        if (static_cast<uint32_t>(this->blockIdx_) >= this->tiling_.numCore) {
            return;
        }
        int64_t blockFactor = this->tiling_.blockFactor;
        int64_t myRows = (static_cast<uint32_t>(this->blockIdx_) == this->tiling_.numCore - 1) ?
                             this->tiling_.blockTailFactor :
                             blockFactor;
        if (myRows <= 0) {
            return;
        }
        int64_t rowStart = static_cast<int64_t>(this->blockIdx_) * blockFactor;
        int64_t channelNum = this->tiling_.channelNum;
        int64_t rowLen = this->tiling_.rowLen;
        int64_t baseLen = this->tiling_.baseLen;

        for (int64_t k = 0; k < myRows; ++k) {
            int64_t r = rowStart + k;
            int64_t channel = (channelNum > 0) ? (r % channelNum) : 0;
            float scaleF = this->ScaleToFp32(this->scaleGm_.GetValue(channel));
            float recipScaleF = this->ReciprocalOfScale(scaleF);
            float offsetF = this->ReadOffsetFp32(channel);

            int64_t base = r * rowLen;
            int64_t loopNum = rowLen / baseLen;
            int64_t loopTail = rowLen % baseLen;
            for (int64_t i = 0; i < loopNum; ++i) {
                this->ProcessOneTile(base + i * baseLen, baseLen, recipScaleF, offsetF);
            }
            if (loopTail != 0) {
                this->ProcessOneTile(base + loopNum * baseLen, loopTail, recipScaleF, offsetF);
            }
        }
    }
};

} // namespace QuantizeOp

#endif // QUANTIZE_KERNEL_H
