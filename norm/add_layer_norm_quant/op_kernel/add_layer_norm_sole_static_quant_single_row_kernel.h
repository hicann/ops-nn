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
 * \file add_layer_norm_sole_static_quant_single_row_kernel.h
 * \brief
 */
#ifndef ADD_LAYER_NORM_SOLE_STATIC_QUANT_SINGLE_ROW_KERNEL_H_
#define ADD_LAYER_NORM_SOLE_STATIC_QUANT_SINGLE_ROW_KERNEL_H_

#include "add_layer_norm_quant_base.h"
#include "kernel_operator.h"

template <typename T, int TILING_KEY, int BUFFER_NUM = 1>
class KernelAddLayerNormSoleStaticQuantSingleRowV2 : public KernelAddLayerNormQuantBase<T, TILING_KEY, BUFFER_NUM> {
public:
    __aicore__ inline KernelAddLayerNormSoleStaticQuantSingleRowV2(TPipe* pipe)
    {
        Ppipe = pipe;
    }

    __aicore__ inline void Init(
        __gm__ uint8_t* x1, __gm__ uint8_t* x2, __gm__ uint8_t* gamma, __gm__ uint8_t* beta, __gm__ uint8_t* bias,
        __gm__ uint8_t* scales, __gm__ uint8_t* fakeScale, __gm__ uint8_t* offsets, __gm__ uint8_t* fakeOffsets,
        __gm__ uint8_t* y, __gm__ uint8_t* fakeY, __gm__ uint8_t* x, __gm__ uint8_t* layernormRes, __gm__ uint8_t* fakeOut1, __gm__ uint8_t* fakeOut2,
        __gm__ uint8_t* workspace, const AddLayerNormQuantV2TilingData* tiling)
    {
        this->InitBaseParams(tiling);
        this->InitInGlobalTensors(x1, x2, gamma, beta, bias);
        this->InitOutGlobalTensors(y, fakeY, x);
        layernormGm.SetGlobalBuffer((__gm__ T*)(layernormRes) + block_idx * this->gmOffset_);

        isOffsetExist = tiling->scaleOffsetMode % 100 >= 10;

        scalesGm.SetGlobalBuffer((__gm__ T*)scales);
        offsetsGm.SetGlobalBuffer((__gm__ T*)offsets);
        uint64_t workspaceOffset = static_cast<uint64_t>(block_idx) * 2 * this->numLastDim;
        workspaceGm.SetGlobalBuffer((__gm__ float*)workspace + workspaceOffset);

        Ppipe->InitBuffer(rowInQue, BUFFER_NUM, this->numLastDimAligned * sizeof(T));
        Ppipe->InitBuffer(rowOutQue, BUFFER_NUM, this->numLastDimAligned * sizeof(T));
        Ppipe->InitBuffer(quantizeOutQue, BUFFER_NUM, this->numLastDimAligned * sizeof(int8_t));
        Ppipe->InitBuffer(tensor_buf, 32);
        Ppipe->InitBuffer(tmpBuf, 3 * this->numLastDimAligned * sizeof(float));
        Ppipe->InitBuffer(gammaBetaBuf, 2 * this->numLastDimAligned * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        CopyInBetaGammaBias();
        uint64_t gmOffset = 0;
        for (int32_t row_idx = 0; row_idx < this->rowWork; ++row_idx) {
            CopyInAddSingleRow(gmOffset);
            precision_compute_single_row(gmOffset);
            gmOffset += static_cast<uint64_t>(this->numLastDim);
        }
    }

private:
    __aicore__ inline void CopyInAddSingleRowFp32(uint64_t gmOffset, LocalTensor<float> tmpTensors,
        LocalTensor<T> biasIn, LocalTensor<T> tensor_local)
    {
        LocalTensor<float> x1Fp32 = tmpTensors[0];
        LocalTensor<float> x2Fp32 = tmpTensors[this->numLastDimAligned];
        Add(x1Fp32, x1Fp32, x2Fp32, this->numLastDim);
        PipeBarrier<PIPE_V>();
        if constexpr (IS_BIAS_ELEWISE || IS_BIAS_BROADCAST) {
            rowInQue.EnQue(biasIn);
            auto biasOptionalTensor = rowInQue.template DeQue<T>();
            Add(x1Fp32, x1Fp32, biasOptionalTensor, this->numLastDim);
            PipeBarrier<PIPE_V>();
            rowInQue.FreeTensor(biasOptionalTensor);
        }
        if (this->isXOut) {
            auto xOutputTensor = rowOutQue.template AllocTensor<T>();
            Adds(xOutputTensor, x1Fp32, ZERO, this->numLastDim);
            rowOutQue.template EnQue<T>(xOutputTensor);
            auto xOut = rowOutQue.template DeQue<T>();
            DataCopyExV2(this->xGm[gmOffset], xOut, tensor_local, this->numLastDim);
            event_t eventMTE3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
            SetFlag<HardEvent::MTE3_V>(eventMTE3V);
            WaitFlag<HardEvent::MTE3_V>(eventMTE3V);
            rowOutQue.FreeTensor(xOut);
        }
    }

    __aicore__ inline void CopyInAddSingleRowFp16(uint64_t gmOffset, LocalTensor<float> tmpTensors,
        LocalTensor<T> x1In, LocalTensor<T> x2In, LocalTensor<T> biasIn, LocalTensor<T> tensor_local)
    {
        LocalTensor<float> xTensor = tmpTensors[0];
        LocalTensor<float> yTensor = tmpTensors[this->numLastDimAligned];
        LocalTensor<float> zTensor = tmpTensors[this->numLastDimAligned * 2];

        Cast(yTensor, x1In, RoundMode::CAST_NONE, this->numLastDimAligned);
        Cast(zTensor, x2In, RoundMode::CAST_NONE, this->numLastDimAligned);
        PipeBarrier<PIPE_V>();
        Add(yTensor, yTensor, zTensor, this->numLastDim);
        PipeBarrier<PIPE_V>();
        if constexpr (IS_BIAS_ELEWISE || IS_BIAS_BROADCAST) {
            rowInQue.EnQue(biasIn);
            auto biasOptionalTensor = rowInQue.template DeQue<T>();
            Cast(xTensor, biasOptionalTensor, RoundMode::CAST_NONE, this->numLastDimAligned);
            PipeBarrier<PIPE_V>();
            Add(xTensor, xTensor, yTensor, this->numLastDim);
            rowInQue.FreeTensor(biasOptionalTensor);
        } else {
            Adds(xTensor, yTensor, ZERO, this->numLastDim);
        }
        if (this->isXOut) {
            auto xOutTensor = rowOutQue.template AllocTensor<T>();
            PipeBarrier<PIPE_V>();
            Cast(xOutTensor, xTensor, RoundMode::CAST_NONE, this->numLastDimAligned);
            rowOutQue.template EnQue<T>(xOutTensor);
            auto xOut = rowOutQue.template DeQue<T>();
            DataCopyExV2(this->xGm[gmOffset], xOut, tensor_local, this->numLastDim);
            event_t eventMTE3V1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
            SetFlag<HardEvent::MTE3_V>(eventMTE3V1);
            WaitFlag<HardEvent::MTE3_V>(eventMTE3V1);
            rowOutQue.FreeTensor(xOut);
        }
    }

    __aicore__ inline void CopyInAddSingleRow(uint64_t gmOffset)
    {
        LocalTensor<float> tmpTensors = tmpBuf.Get<float>();
        LocalTensor<T> x1x2In = tmpTensors.ReinterpretCast<T>()[0];
        LocalTensor<T> x1In = x1x2In[0];
        LocalTensor<T> x2In = x1x2In[this->numLastDimAligned];

        LocalTensor<T> tensor_local = tensor_buf.Get<T>();
        DataCopyExV2(x1In, this->x1Gm[gmOffset], tensor_local, this->numLastDim);
        DataCopyExV2(x2In, this->x2Gm[gmOffset], tensor_local, this->numLastDim);

        LocalTensor<T> biasIn;
        if constexpr (IS_BIAS_ELEWISE || IS_BIAS_BROADCAST) {
            biasIn = rowInQue.template AllocTensor<T>();
            if constexpr (IS_BIAS_ELEWISE) {
                DataCopyExV2(biasIn, this->biasGm[gmOffset], tensor_local, this->numLastDim);
            } else {
                DataCopyExV2(biasIn, this->biasGm, tensor_local, this->numLastDim);
            }
        }
        event_t eventMTE2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventMTE2V);
        WaitFlag<HardEvent::MTE2_V>(eventMTE2V);

        if constexpr (is_same<float, T>::value) {
            CopyInAddSingleRowFp32(gmOffset, tmpTensors, biasIn, tensor_local);
        } else {
            CopyInAddSingleRowFp16(gmOffset, tmpTensors, x1In, x2In, biasIn, tensor_local);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline float ComputeMeanVarRstd(LocalTensor<float> xTensor, LocalTensor<float> tmpTensor)
    {
        Muls(tmpTensor, xTensor, this->aveNum, this->numLastDim);
        PipeBarrier<PIPE_V>();
        float aveLocalTemp = ReduceSumFP32(tmpTensor, this->numLastDim);
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        Adds(xTensor, xTensor, -1 * aveLocalTemp, this->numLastDim);
        PipeBarrier<PIPE_V>();
        Mul(tmpTensor, xTensor, xTensor, this->numLastDim);
        PipeBarrier<PIPE_V>();
        Muls(tmpTensor, tmpTensor, this->aveNum, this->numLastDim);
        PipeBarrier<PIPE_V>();
        float varLocalTemp = ReduceSumFP32(tmpTensor, this->numLastDim);
        float rstdLocalTemp = 1 / sqrt(varLocalTemp + this->eps);
        SetFlag<HardEvent::S_V>(EVENT_ID0);
        WaitFlag<HardEvent::S_V>(EVENT_ID0);
        Muls(xTensor, xTensor, rstdLocalTemp, this->numLastDim);
        PipeBarrier<PIPE_V>();
        return rstdLocalTemp;
    }

    __aicore__ inline void ApplyGammaBeta(
        LocalTensor<float> xTensor, LocalTensor<float> tmpTensor, LocalTensor<T> gammaTensor, LocalTensor<T> betaTensor)
    {
        CastToFloat<T>(tmpTensor, gammaTensor, this->numLastDimAligned);
        Mul(xTensor, tmpTensor, xTensor, this->numLastDim);
        PipeBarrier<PIPE_V>();
        CastToFloat<T>(tmpTensor, betaTensor, this->numLastDimAligned);
        PipeBarrier<PIPE_V>();
        Add(xTensor, tmpTensor, xTensor, this->numLastDim);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void QuantizeToInt8(LocalTensor<float> floatTensor, LocalTensor<int8_t> yLocal)
    {
        Cast(floatTensor.ReinterpretCast<int32_t>(), floatTensor, RoundMode::CAST_RINT, this->numLastDimAligned);
        PipeBarrier<PIPE_V>();
        SetDeqScale((half)1.000000e+00f);
        PipeBarrier<PIPE_V>();
        Cast(
            floatTensor.ReinterpretCast<half>(), floatTensor.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE,
            this->numLastDimAligned);
        PipeBarrier<PIPE_V>();
        Cast(yLocal, floatTensor.ReinterpretCast<half>(), RoundMode::CAST_TRUNC, this->numLastDimAligned);
    }

    __aicore__ inline void ApplyScaleOffsetAndQuantize(uint64_t gmOffset, LocalTensor<float> xTensor,
        LocalTensor<float> tmpTensor, LocalTensor<T> tensor_local)
    {
        LocalTensor<T> inTensor = rowOutQue.template AllocTensor<T>();
        DataCopyExV2(inTensor, scalesGm, tensor_local, this->numLastDim);
        event_t eventMTE2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventMTE2V);
        WaitFlag<HardEvent::MTE2_V>(eventMTE2V);

        CastToFloat<T>(tmpTensor, inTensor, this->numLastDimAligned);
        PipeBarrier<PIPE_V>();
        Mul(xTensor, xTensor, tmpTensor, this->numLastDim);
        PipeBarrier<PIPE_V>();

        event_t eventVMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        if (isOffsetExist) {
            SetFlag<HardEvent::V_MTE2>(eventVMTE2);
            WaitFlag<HardEvent::V_MTE2>(eventVMTE2);
            DataCopyExV2(inTensor, offsetsGm, tensor_local, this->numLastDim);
            event_t eventMTE2V1 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventMTE2V1);
            WaitFlag<HardEvent::MTE2_V>(eventMTE2V1);
            CastToFloat<T>(tmpTensor, inTensor, this->numLastDimAligned);
            PipeBarrier<PIPE_V>();
            Add(xTensor, xTensor, tmpTensor, this->numLastDim);
            PipeBarrier<PIPE_V>();
        }
        rowOutQue.FreeTensor(inTensor);

        LocalTensor<int8_t> yLocal = quantizeOutQue.template AllocTensor<int8_t>();
        QuantizeToInt8(xTensor, yLocal);
        SetFlag<HardEvent::V_MTE2>(eventVMTE2);
        WaitFlag<HardEvent::V_MTE2>(eventVMTE2);
        quantizeOutQue.EnQue(yLocal);
        auto yOut = quantizeOutQue.template DeQue<int8_t>();
        LocalTensor<int8_t> tensor_localY = tensor_buf.Get<int8_t>();
        DataCopyExV2(this->y1Gm[gmOffset], yOut, tensor_localY, this->numLastDim);
        quantizeOutQue.FreeTensor(yOut);
    }

    __aicore__ inline void precision_compute_single_row(uint64_t gmOffset)
    {
        LocalTensor<float> tmpTensors = tmpBuf.Get<float>();
        LocalTensor<T> gammaBetaTensor = gammaBetaBuf.Get<T>();

        LocalTensor<float> xTensor = tmpTensors[0];
        LocalTensor<float> tmpTensor = tmpTensors[this->numLastDimAligned];
        LocalTensor<T> gammaTensor = gammaBetaTensor[0];
        LocalTensor<T> betaTensor = gammaBetaTensor[this->numLastDimAligned];
        LocalTensor<T> tensor_local = tensor_buf.Get<T>();

        ComputeMeanVarRstd(xTensor, tmpTensor);
        ApplyGammaBeta(xTensor, tmpTensor, gammaTensor, betaTensor);

        CopyOutLayernormRes(gmOffset);
        event_t eventMTE3MTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventMTE3MTE2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventMTE3MTE2);
        PipeBarrier<PIPE_V>();

        ApplyScaleOffsetAndQuantize(gmOffset, xTensor, tmpTensor, tensor_local);
    }

    __aicore__ inline void CopyInBetaGammaBias()
    {
        LocalTensor<T> gammaBetaTensor = gammaBetaBuf.Get<T>();

        LocalTensor<T> gammaIn = gammaBetaTensor[0];
        LocalTensor<T> betaIn = gammaBetaTensor[this->numLastDimAligned];

        LocalTensor<T> tensor_local = tensor_buf.Get<T>();
        DataCopyExV2(gammaIn, this->gammaGm, tensor_local, this->numLastDim);
        DataCopyExV2(betaIn, this->betaGm, tensor_local, this->numLastDim);

        event_t eventMTE2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventMTE2V);
        WaitFlag<HardEvent::MTE2_V>(eventMTE2V);
    }

    __aicore__ inline void CopyOutLayernormRes(uint32_t gm_offset)
    {
        LocalTensor<float> tmpTensors = tmpBuf.Get<float>();
        LocalTensor<float> xTensor = tmpTensors[0];
        LocalTensor<T> resLocal = rowOutQue.template AllocTensor<T>();
        LocalTensor<T> tensor_local = tensor_buf.Get<T>();

        if constexpr (is_same<T, float>::value) {
            Adds(resLocal, xTensor, ZERO, this->numLastDim);
        } else if constexpr (is_same<T, half>::value){
            Cast(resLocal, xTensor, RoundMode::CAST_NONE, this->numLastDimAligned);
        } else {
            Cast(resLocal, xTensor, RoundMode::CAST_RINT, this->numLastDimAligned);
        }

        rowOutQue.template EnQue<T>(resLocal);
        LocalTensor<T> resDeq = rowOutQue.template DeQue<T>();
        DataCopyExV2(this->layernormGm[gm_offset], resDeq, tensor_local, this->numLastDim);
        rowOutQue.FreeTensor(resDeq);
    }

private:
    TPipe* Ppipe = nullptr;
    TQue<QuePosition::VECIN, BUFFER_NUM> rowInQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> quantizeOutQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> rowOutQue;

    TBuf<TPosition::VECCALC> tmpBuf;
    TBuf<TPosition::VECCALC> gammaBetaBuf;
    TBuf<TPosition::VECCALC> tensor_buf;

    GlobalTensor<T> scalesGm;
    GlobalTensor<T> offsetsGm;
    GlobalTensor<T> layernormGm;

    GlobalTensor<float> workspaceGm;

    bool isOffsetExist;
};

#endif // __ADD_LAYER_NORM_SOLE_STATIC_QUANT_SINGLE_ROW_KERNEL_H_