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
 * \file add_layer_norm_dual_static_quant_single_row_kernel.h
 * \brief
 */
#ifndef ADD_LAYER_NORM_DUAL_STATIC_QUANT_SINGLE_ROW_KERNEL_H_
#define ADD_LAYER_NORM_DUAL_STATIC_QUANT_SINGLE_ROW_KERNEL_H_

#include "add_layer_norm_quant_base.h"
#include "kernel_operator.h"

template <typename T, int TILING_KEY, int BUFFER_NUM = 1>
class KernelAddLayerNormDualStaticQuantSingleRowV2 : public KernelAddLayerNormQuantBase<T, TILING_KEY, BUFFER_NUM> {
public:
    __aicore__ inline KernelAddLayerNormDualStaticQuantSingleRowV2(TPipe* pipe)
    {
        Ppipe = pipe;
    }

    __aicore__ inline void Init(
        __gm__ uint8_t* x1, __gm__ uint8_t* x2, __gm__ uint8_t* gamma, __gm__ uint8_t* beta, __gm__ uint8_t* bias,
        __gm__ uint8_t* scales1, __gm__ uint8_t* scales2, __gm__ uint8_t* zeroPoints1, __gm__ uint8_t* zeroPoints2,
        __gm__ uint8_t* y1, __gm__ uint8_t* y2, __gm__ uint8_t* x, __gm__ uint8_t* layernormRes, __gm__ uint8_t* fakeOut1, __gm__ uint8_t* fakeOut2,
        __gm__ uint8_t* workspace, const AddLayerNormQuantV2TilingData* tiling)
    {
        this->InitBaseParams(tiling);
        this->InitInGlobalTensors(x1, x2, gamma, beta, bias);
        this->InitOutGlobalTensors(y1, y2, x);
        layernormGm.SetGlobalBuffer((__gm__ T*)(layernormRes) + block_idx * this->gmOffset_);

        isZeroPoint1Exist = tiling->scaleOffsetMode % 100 >= 10;
        isZeroPoint2Exist = tiling->scaleOffsetMode % 10 >= 1;

        scales1Gm.SetGlobalBuffer((__gm__ T*)scales1);
        scales2Gm.SetGlobalBuffer((__gm__ T*)scales2);
        if (isZeroPoint1Exist) {
            zeroPoints1Gm.SetGlobalBuffer((__gm__ T*)zeroPoints1);
        }
        if (isZeroPoint2Exist) {
            zeroPoints2Gm.SetGlobalBuffer((__gm__ T*)zeroPoints2);
        }
        Ppipe->InitBuffer(tensor_buf, 32);
        Ppipe->InitBuffer(rowInQue, BUFFER_NUM, this->numLastDimAligned * sizeof(T));
        Ppipe->InitBuffer(rowOutQue, BUFFER_NUM, this->numLastDimAligned * sizeof(T));
        Ppipe->InitBuffer(quantizeOutQue, BUFFER_NUM, this->numLastDimAligned * sizeof(int8_t));
        Ppipe->InitBuffer(tmpBuf, 3 * this->numLastDimAligned * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        uint64_t gm_offset = 0;
        for (uint32_t row_idx = 0; row_idx < this->rowWork; ++row_idx) {
            CopyInAddSingleRow(gm_offset);
            precision_compute_single_row(gm_offset);
            gm_offset += static_cast<uint64_t>(this->numLastDim);
        }
    }

private:
    __aicore__ inline void CopyInAddSingleRowFp32(uint64_t gm_offset, LocalTensor<float> tmpTensors,
        LocalTensor<T> biasIn, LocalTensor<T> tensor_local)
    {
        LocalTensor<float> x1Fp32 = tmpTensors[0];
        LocalTensor<float> x2Fp32 = tmpTensors[this->numLastDimAligned];
        Add(x1Fp32, x1Fp32, x2Fp32, this->numLastDim);
        PipeBarrier<PIPE_V>();
        if constexpr (IS_BIAS_ELEWISE || IS_BIAS_BROADCAST) {
            rowInQue.EnQue(biasIn);
            auto biastensor = rowInQue.template DeQue<T>();
            Add(x1Fp32, x1Fp32, biastensor, this->numLastDim);
            PipeBarrier<PIPE_V>();
            rowInQue.FreeTensor(biastensor);
        }
        if (this->isXOut) {
            auto xOutTensor = rowOutQue.template AllocTensor<T>();
            Adds(xOutTensor, x1Fp32, ZERO, this->numLastDim);
            rowOutQue.template EnQue<T>(xOutTensor);
            auto xOut = rowOutQue.template DeQue<T>();
            DataCopyExV2(this->xGm[gm_offset], xOut, tensor_local, this->numLastDim);
            rowOutQue.FreeTensor(xOut);
        }
    }

    __aicore__ inline void CopyInAddSingleRowFp16(uint64_t gm_offset, LocalTensor<float> tmpTensors,
        LocalTensor<T> x1In, LocalTensor<T> x2In, LocalTensor<T> biasIn, LocalTensor<T> tensor_local)
    {
        LocalTensor<float> xTensor = tmpTensors[0];
        LocalTensor<float> yTensor = tmpTensors[this->numLastDimAligned];
        LocalTensor<float> zTensor = tmpTensors[this->numLastDimAligned * 2];

        Cast(yTensor, x1In, RoundMode::CAST_NONE, this->numLastDimAligned);
        PipeBarrier<PIPE_V>();
        Cast(zTensor, x2In, RoundMode::CAST_NONE, this->numLastDimAligned);
        PipeBarrier<PIPE_V>();
        Add(yTensor, yTensor, zTensor, this->numLastDim);
        PipeBarrier<PIPE_V>();
        if constexpr (IS_BIAS_ELEWISE || IS_BIAS_BROADCAST) {
            rowInQue.EnQue(biasIn);
            auto biastensor = rowInQue.template DeQue<T>();
            Cast(xTensor, biastensor, RoundMode::CAST_NONE, this->numLastDimAligned);
            PipeBarrier<PIPE_V>();
            Add(xTensor, xTensor, yTensor, this->numLastDim);
            rowInQue.FreeTensor(biastensor);
        } else {
            Adds(xTensor, yTensor, ZERO, this->numLastDim);
        }
        if (this->isXOut) {
            auto xOutTensor = rowOutQue.template AllocTensor<T>();
            PipeBarrier<PIPE_V>();
            Cast(xOutTensor, xTensor, RoundMode::CAST_RINT, this->numLastDimAligned);
            rowOutQue.template EnQue<T>(xOutTensor);
            auto xOut = rowOutQue.template DeQue<T>();
            DataCopyExV2(this->xGm[gm_offset], xOut, tensor_local, this->numLastDim);
            rowOutQue.FreeTensor(xOut);
        }
    }

    __aicore__ inline void CopyInAddSingleRow(uint64_t gm_offset)
    {
        LocalTensor<float> tmpTensors = tmpBuf.Get<float>();
        LocalTensor<T> x1x2Fp32 = tmpTensors.ReinterpretCast<T>()[0];
        LocalTensor<T> x1In = x1x2Fp32[0];
        LocalTensor<T> x2In = x1x2Fp32[this->numLastDimAligned];
        LocalTensor<T> biasIn = rowInQue.template AllocTensor<T>();

        LocalTensor<T> tensor_local = tensor_buf.Get<T>();
        DataCopyExV2(x1In, this->x1Gm[gm_offset], tensor_local, this->numLastDim);
        PipeBarrier<PIPE_V>();
        DataCopyExV2(x2In, this->x2Gm[gm_offset], tensor_local, this->numLastDim);
        PipeBarrier<PIPE_V>();
        DataCopyExV2(biasIn, this->biasGm, tensor_local, this->numLastDim);

        rowInQue.EnQue(biasIn);
        auto biasTensor = rowInQue.template DeQue<T>();

        if constexpr (is_same<float, T>::value) {
            CopyInAddSingleRowFp32(gm_offset, tmpTensors, biasIn, tensor_local);
        } else {
            CopyInAddSingleRowFp16(gm_offset, tmpTensors, x1In, x2In, biasIn, tensor_local);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline float ComputeMeanVarRstd(LocalTensor<float> xTensor, LocalTensor<float> tmpTensor)
    {
        Muls(tmpTensor, xTensor, this->aveNum, this->numLastDim);
        float aveLocalTemp = ReduceSumFP32(tmpTensor, this->numLastDim);
        event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventSV);
        WaitFlag<HardEvent::S_V>(eventSV);
        Adds(xTensor, xTensor, -1 * aveLocalTemp, this->numLastDim);
        PipeBarrier<PIPE_V>();
        Mul(tmpTensor, xTensor, xTensor, this->numLastDim);
        PipeBarrier<PIPE_V>();
        Muls(tmpTensor, tmpTensor, this->aveNum, this->numLastDim);
        PipeBarrier<PIPE_V>();
        float varLocalTemp = ReduceSumFP32(tmpTensor, this->numLastDim);
        PipeBarrier<PIPE_V>();
        float rstdLocalTemp = 1 / sqrt(varLocalTemp + this->eps);
        event_t eventSV1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventSV1);
        WaitFlag<HardEvent::S_V>(eventSV1);
        Muls(xTensor, xTensor, rstdLocalTemp, this->numLastDim);
        return rstdLocalTemp;
    }

    __aicore__ inline void ApplyGammaBeta(
        LocalTensor<float> xTensor, LocalTensor<float> tmpTensor, LocalTensor<T> gammaLocal, LocalTensor<T> betaLocal)
    {
        if constexpr (is_same<T, float>::value) {
            Adds(tmpTensor, gammaLocal, ZERO, this->numLastDim);
        } else {
            Cast(tmpTensor, gammaLocal, RoundMode::CAST_NONE, this->numLastDimAligned);
        }
        PipeBarrier<PIPE_V>();
        Mul(xTensor, xTensor, tmpTensor, this->numLastDim);
        PipeBarrier<PIPE_V>();
        if constexpr (is_same<T, float>::value) {
            Adds(tmpTensor, betaLocal, ZERO, this->numLastDim);
        } else {
            Cast(tmpTensor, betaLocal, RoundMode::CAST_NONE, this->numLastDimAligned);
        }
        PipeBarrier<PIPE_V>();
        Add(xTensor, xTensor, tmpTensor, this->numLastDim);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void QuantizeToInt8(LocalTensor<float> fp32Tensor, LocalTensor<int8_t> yLocal)
    {
        Cast(fp32Tensor.ReinterpretCast<int32_t>(), fp32Tensor, RoundMode::CAST_RINT, this->numLastDimAligned);
        PipeBarrier<PIPE_V>();
        SetDeqScale((half)1.000000e+00f);
        PipeBarrier<PIPE_V>();
        Cast(
            fp32Tensor.ReinterpretCast<half>(), fp32Tensor.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE,
            this->numLastDimAligned);
        PipeBarrier<PIPE_V>();
        Cast(yLocal, fp32Tensor.ReinterpretCast<half>(), RoundMode::CAST_TRUNC, this->numLastDimAligned);
    }

    __aicore__ inline void QuantizeY1(uint64_t gm_offset, LocalTensor<float> xTensor, LocalTensor<float> tmpTensor,
        LocalTensor<float> constsTensor, LocalTensor<T> tensor_local, LocalTensor<int8_t> tensor_localY,
        LocalTensor<T> scales01Tensor)
    {
        if constexpr (is_same<T, float>::value) {
            Adds(tmpTensor, scales01Tensor, ZERO, this->numLastDim);
        } else {
            Cast(tmpTensor, scales01Tensor, RoundMode::CAST_NONE, this->numLastDimAligned);
        }
        rowInQue.FreeTensor(scales01Tensor);

        LocalTensor<T> offsets01In;
        if (isZeroPoint1Exist) {
            offsets01In = rowInQue.template AllocTensor<T>();
            DataCopyExV2(offsets01In, zeroPoints1Gm, tensor_local, this->numLastDim);
            event_t eventMTE2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventMTE2V);
            WaitFlag<HardEvent::MTE2_V>(eventMTE2V);
        }

        Mul(constsTensor, xTensor, tmpTensor, this->numLastDim);
        PipeBarrier<PIPE_V>();

        if (isZeroPoint1Exist) {
            rowInQue.EnQue(offsets01In);
            auto offsets01Local = rowInQue.template DeQue<T>();
            if constexpr (is_same<T, float>::value) {
                Adds(tmpTensor, offsets01Local, ZERO, this->numLastDim);
            } else {
                Cast(tmpTensor, offsets01Local, RoundMode::CAST_NONE, this->numLastDimAligned);
            }
            PipeBarrier<PIPE_V>();
            rowInQue.FreeTensor(offsets01Local);
            Add(constsTensor, constsTensor, tmpTensor, this->numLastDim);
            PipeBarrier<PIPE_V>();
        }

        LocalTensor<int8_t> y1Local = quantizeOutQue.template AllocTensor<int8_t>();
        QuantizeToInt8(constsTensor, y1Local);
        quantizeOutQue.EnQue(y1Local);
        auto y1Out = quantizeOutQue.template DeQue<int8_t>();
        DataCopyExV2(this->y1Gm[gm_offset], y1Out, tensor_localY, this->numLastDim);
        event_t eventMTE3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventMTE3V);
        WaitFlag<HardEvent::MTE3_V>(eventMTE3V);
        quantizeOutQue.FreeTensor(y1Out);
    }

    __aicore__ inline void CopyInAndCastScales2Offsets2(LocalTensor<float> constsTensor,
        LocalTensor<float> tmpTensor, LocalTensor<T> tensor_local)
    {
        event_t eventMTE2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        LocalTensor<T> scales2In = rowInQue.template AllocTensor<T>();
        DataCopyExV2(scales2In, scales2Gm, tensor_local, this->numLastDim);
        SetFlag<HardEvent::MTE2_V>(eventMTE2V);
        WaitFlag<HardEvent::MTE2_V>(eventMTE2V);
        LocalTensor<T> offsets2Tensor;
        if (isZeroPoint2Exist) {
            offsets2Tensor = rowOutQue.template AllocTensor<T>();
            DataCopyExV2(offsets2Tensor, zeroPoints2Gm, tensor_local, this->numLastDim);
            SetFlag<HardEvent::MTE2_V>(eventMTE2V);
            WaitFlag<HardEvent::MTE2_V>(eventMTE2V);
        }
        rowInQue.EnQue(scales2In);
        auto scales2Tensor = rowInQue.template DeQue<T>();

        if constexpr (is_same<T, float>::value) {
            Adds(constsTensor, scales2Tensor, ZERO, this->numLastDim);
        } else {
            Cast(constsTensor, scales2Tensor, RoundMode::CAST_NONE, this->numLastDimAligned);
        }
        if (isZeroPoint2Exist) {
            if constexpr (is_same<T, float>::value) {
                Adds(tmpTensor, offsets2Tensor, ZERO, this->numLastDim);
            } else {
                Cast(tmpTensor, offsets2Tensor, RoundMode::CAST_NONE, this->numLastDimAligned);
            }
            rowOutQue.FreeTensor(offsets2Tensor);
        }
        PipeBarrier<PIPE_V>();
        rowInQue.FreeTensor(scales2Tensor);
    }

    __aicore__ inline void QuantizeY2(uint64_t gm_offset, LocalTensor<float> xTensor, LocalTensor<float> tmpTensor,
        LocalTensor<float> constsTensor, LocalTensor<T> tensor_local, LocalTensor<int8_t> tensor_localY)
    {
        CopyInAndCastScales2Offsets2(constsTensor, tmpTensor, tensor_local);

        PipeBarrier<PIPE_V>();
        Mul(constsTensor, xTensor, constsTensor, this->numLastDim);
        PipeBarrier<PIPE_V>();
        if (isZeroPoint2Exist) {
            Add(constsTensor, constsTensor, tmpTensor, this->numLastDim);
            PipeBarrier<PIPE_V>();
        }

        LocalTensor<int8_t> y2Local = quantizeOutQue.template AllocTensor<int8_t>();
        QuantizeToInt8(constsTensor, y2Local);
        quantizeOutQue.EnQue(y2Local);
        auto y2Out = quantizeOutQue.template DeQue<int8_t>();
        event_t eventVMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventVMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventVMTE3);
        DataCopyExV2(this->y2Gm[gm_offset], y2Out, tensor_localY, this->numLastDim);
        event_t eventMTE3V1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventMTE3V1);
        WaitFlag<HardEvent::MTE3_V>(eventMTE3V1);
        quantizeOutQue.FreeTensor(y2Out);
    }

    __aicore__ inline void precision_compute_single_row(uint64_t gm_offset)
    {
        LocalTensor<float> tmpTensors = tmpBuf.Get<float>();
        LocalTensor<float> xTensor = tmpTensors[0];
        LocalTensor<float> tmpTensor = tmpTensors[this->numLastDimAligned];
        LocalTensor<float> constsTensor = tmpTensors[this->numLastDimAligned * 2];

        LocalTensor<T> scales01In = rowInQue.template AllocTensor<T>();
        LocalTensor<T> gammaLocal = constsTensor.ReinterpretCast<T>()[0];
        LocalTensor<T> betaLocal = constsTensor.ReinterpretCast<T>()[this->numLastDimAligned];
        LocalTensor<T> tensor_local = tensor_buf.Get<T>();

        DataCopyExV2(gammaLocal, this->gammaGm, tensor_local, this->numLastDim);
        PipeBarrier<PIPE_V>();
        DataCopyExV2(betaLocal, this->betaGm, tensor_local, this->numLastDim);
        PipeBarrier<PIPE_V>();
        DataCopyExV2(scales01In, scales1Gm, tensor_local, this->numLastDim);
        PipeBarrier<PIPE_V>();

        event_t eventMTE2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventMTE2V);
        WaitFlag<HardEvent::MTE2_V>(eventMTE2V);

        ComputeMeanVarRstd(xTensor, tmpTensor);

        rowInQue.EnQue(scales01In);
        auto scales01Tensor = rowInQue.template DeQue<T>();
        ApplyGammaBeta(xTensor, tmpTensor, gammaLocal, betaLocal);
        CopyOutLayernormRes(gm_offset);
        PipeBarrier<PIPE_V>();

        LocalTensor<int8_t> tensor_localY = tensor_buf.Get<int8_t>();
        QuantizeY1(gm_offset, xTensor, tmpTensor, constsTensor, tensor_local, tensor_localY, scales01Tensor);
        QuantizeY2(gm_offset, xTensor, tmpTensor, constsTensor, tensor_local, tensor_localY);
    }

    __aicore__ inline void CopyOutLayernormRes(uint32_t gm_offset)
    {
        LocalTensor<float> tmpTensors = tmpBuf.Get<float>();
        LocalTensor<float> xTensor = tmpTensors[0];
        LocalTensor<T> tensor_local = tensor_buf.Get<T>();
        LocalTensor<T> resLocal = rowOutQue.template AllocTensor<T>();

        if constexpr (is_same<T, float>::value) {
            Adds(resLocal, xTensor, 0.0f, this->numLastDim);
        } else if constexpr (is_same<T, half>::value) {
            Cast(resLocal, xTensor, RoundMode::CAST_NONE, this->numLastDimAligned);
        } else{
            Cast(resLocal, xTensor, RoundMode::CAST_RINT, this->numLastDimAligned);
        }
        PipeBarrier<PIPE_V>();
        rowOutQue.template EnQue<T>(resLocal);
        LocalTensor<T> resDeq = rowOutQue.template DeQue<T>();
        DataCopyExV2(this->layernormGm[gm_offset], resDeq, tensor_local, this->numLastDim);
        rowOutQue.FreeTensor(resDeq);
    }

private:
    TPipe* Ppipe = nullptr;
    TQue<QuePosition::VECIN, BUFFER_NUM> rowInQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> rowOutQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> quantizeOutQue;

    TBuf<TPosition::VECCALC> tmpBuf;
    TBuf<TPosition::VECCALC> tensor_buf;

    GlobalTensor<T> scales1Gm;
    GlobalTensor<T> scales2Gm;
    GlobalTensor<T> zeroPoints1Gm;
    GlobalTensor<T> zeroPoints2Gm;
    GlobalTensor<T> layernormGm;

    GlobalTensor<float> workspaceGm;

    bool isZeroPoint1Exist;
    bool isZeroPoint2Exist;
};

#endif // __ADD_LAYER_NORM_DUAL_STATIC_QUANT_SINGLE_ROW_KERNEL_H_