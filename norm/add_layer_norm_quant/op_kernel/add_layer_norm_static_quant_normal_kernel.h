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
 * \file add_layer_norm_static_quant_normal_kernel.h
 * \brief
 */

#ifndef ADD_LAYER_NORM_STATIC_QUANT_NORMAL_KERNEL_H_
#define ADD_LAYER_NORM_STATIC_QUANT_NORMAL_KERNEL_H_

#include "add_layer_norm_quant_base.h"

template <typename T, typename S, int TILING_KEY, int BUFFER_NUM = 1>
class KernelAddLayerNormStaticQuantNormal : public KernelAddLayerNormQuantBase<T, TILING_KEY, BUFFER_NUM> {
public:
    __aicore__ inline KernelAddLayerNormStaticQuantNormal(TPipe* pipe) { Ppipe = pipe; }

    template <typename TilingDataT>
    __aicore__ inline void Init(__gm__ uint8_t* x1, __gm__ uint8_t* x2, __gm__ uint8_t* gamma, __gm__ uint8_t* beta,
                                __gm__ uint8_t* bias, __gm__ uint8_t* scales1, __gm__ uint8_t* scales2,
                                __gm__ uint8_t* zeroPoints1, __gm__ uint8_t* zeroPoints2, __gm__ uint8_t* y1,
                                __gm__ uint8_t* y2, __gm__ uint8_t* x, __gm__ uint8_t* layernormRes,
                                __gm__ uint8_t* outScale1, __gm__ uint8_t* outScale2, __gm__ uint8_t* workspace,
                                const TilingDataT* tiling)
    {
        this->InitBaseParams(tiling);
        this->InitInGlobalTensors(x1, x2, gamma, beta, bias);
        this->InitOutGlobalTensors(y1, y2, x);
        if (layernormRes == nullptr) {
            // 不输出layernormRes，走v1
            layernormResExist = false;
        } else {
            layernormResExist = true;
            resGm.SetGlobalBuffer((__gm__ T*)(layernormRes) + block_idx * this->gmOffset_);
        }

        this->scales1Exist = tiling->scaleOffsetMode >= 100;
        this->scales2Exist = tiling->scaleOffsetMode >= 200;
        this->isZeroPoint1Exist = tiling->scaleOffsetMode % 100 >= 10;
        this->isZeroPoint2Exist = tiling->scaleOffsetMode % 10 >= 1;

        scales1Gm.SetGlobalBuffer((__gm__ S*)scales1);
        scales2Gm.SetGlobalBuffer((__gm__ S*)scales2);
        zeroPoints1Gm.SetGlobalBuffer((__gm__ S*)zeroPoints1);
        zeroPoints2Gm.SetGlobalBuffer((__gm__ S*)zeroPoints2);

        /*
          UB = 3 * this->rowStep * alignedCol * sizeof(T)
              + 2 * this->rowStep * alignedCol * sizeof(float)
              + Count(gamma,beta,bias) * alignedCol * sizeof(T)
              + 512Bytes(256 + reduceOut)
        */
        Ppipe->InitBuffer(tensor_buf, 32);
        alignedStride = layernormResExist ? this->numLastDimRoundUp32 : this->numLastDimAligned;

        Ppipe->InitBuffer(inRowsQue, BUFFER_NUM, ROUND_UP32(2 * this->rowStep * alignedStride * sizeof(T)));
        Ppipe->InitBuffer(outRowQue, BUFFER_NUM, ROUND_UP32(this->rowStep * alignedStride * sizeof(T)));
        Ppipe->InitBuffer(xBufFp32, ROUND_UP32(this->rowStep * alignedStride * sizeof(float)));
        Ppipe->InitBuffer(yBufFp32, ROUND_UP32(this->rowStep * alignedStride * sizeof(float)));
        Ppipe->InitBuffer(gammaBuf, ROUND_UP32(alignedStride * sizeof(T)));
        Ppipe->InitBuffer(betaBuf, ROUND_UP32(alignedStride * sizeof(T)));
        if constexpr (IS_BIAS_BROADCAST) {
            Ppipe->InitBuffer(biasBuf, ROUND_UP32(alignedStride * sizeof(T)));
        }

        if (this->isPerTensor && this->scales1Exist) {
            GetPerTensorScaleOffset();
        }
    }

    __aicore__ inline void Process()
    {
        int32_t rowMoveCnt = CEIL_DIV(this->rowWork, this->rowStep);
        CopyInGammaBeta();

        LocalTensor<T> betaLocal = betaBuf.template Get<T>();
        LocalTensor<T> gammaLocal = gammaBuf.template Get<T>();

        DataCopyPadParams padParams;
        if (this->lastDimPad) {
            padParams.isPad = true;
            padParams.paddingValue = 0;
            padParams.rightPadding = this->numLastDimAligned - this->numLastDim;
        }

        uint64_t gmOffset = 0;
        uint64_t gmOffsetScale = 0;
        int32_t elementCount = this->rowStep * alignedStride;

        for (int32_t rowIdx = 0; rowIdx < rowMoveCnt - 1; ++rowIdx) {
            CopyInX1X2(gmOffset, this->rowStep, elementCount, padParams);
            AddX1X2Bias(elementCount, this->rowStep);
            CopyInScaleOffset(scales1Gm, zeroPoints1Gm, isZeroPoint1Exist);
            CopyOutAdditionalOutput(gmOffset, this->rowStep, elementCount);
            ComputeLayerNorm(gmOffset, this->rowStep, elementCount, gammaLocal, betaLocal);
            ComputeStaticQuant(this->rowStep, elementCount);
            CopyOut(gmOffset, gmOffsetScale, this->rowStep);
            gmOffset += static_cast<uint64_t>(this->rowStep) * this->numLastDim;
            gmOffsetScale += this->rowStep;
        }
        {
            elementCount = alignedStride * this->rowTail_;
            int32_t rowIdx = rowMoveCnt - 1;
            CopyInX1X2(gmOffset, this->rowTail_, elementCount, padParams);
            AddX1X2Bias(elementCount, this->rowTail_);
            CopyInScaleOffset(scales1Gm, zeroPoints1Gm, isZeroPoint1Exist);
            CopyOutAdditionalOutput(gmOffset, this->rowTail_, elementCount);
            ComputeLayerNorm(gmOffset, this->rowTail_, elementCount, gammaLocal, betaLocal);
            ComputeStaticQuant(this->rowTail_, elementCount);
            CopyOut(gmOffset, gmOffsetScale, this->rowTail_);
        }
    }

private:
    __aicore__ inline void GetPerTensorScaleOffset()
    {
        if constexpr (is_same<S, float>::value) {
            perTensorScale1 = scales1Gm.GetValue(0);
        } else if constexpr (is_same<S, half>::value) {
            perTensorScale1 = static_cast<float>(scales1Gm.GetValue(0));
        } else {
            perTensorScale1 = Cast(scales1Gm.GetValue(0));
        }

        if (isZeroPoint1Exist) {
            if constexpr (is_same<S, float>::value) {
                perTensorOffset1 = zeroPoints1Gm.GetValue(0);
            } else if constexpr (is_same<S, half>::value) {
                perTensorOffset1 = static_cast<float>(zeroPoints1Gm.GetValue(0));
            } else {
                perTensorOffset1 = Cast(zeroPoints1Gm.GetValue(0));
            }
        }
    }

    __aicore__ inline void CopyInX1X2(uint64_t gmOffset, int32_t rowCount, int32_t elementCount,
                                      DataCopyPadParams& padParams)
    {
        LocalTensor<T> x1x2LocalIn = inRowsQue.template AllocTensor<T>();

        if (layernormResExist) {
            LocalTensor<T> tensor_local = tensor_buf.Get<T>();
#if __NPU_ARCH__ == 2201
            uint32_t stride_ = (this->numLastDimRoundUp32 - this->numLastDimAligned) * sizeof(T) / 32;
            DataCopyExAlign(x1x2LocalIn[0], this->x1Gm[gmOffset], tensor_local, this->numLastDim, stride_, rowCount,
                            padParams);
            DataCopyExAlign(x1x2LocalIn[elementCount], this->x2Gm[gmOffset], tensor_local, this->numLastDim, stride_,
                            rowCount, padParams);
            if constexpr (IS_BIAS_ELEWISE) {
                LocalTensor<T> biasLocal = yBufFp32.template Get<T>();
                if constexpr (is_same<float, T>::value) {
                    DataCopyExAlign(biasLocal, this->biasGm[gmOffset], tensor_local, this->numLastDim, stride_,
                                    rowCount, padParams);
                } else {
                    DataCopyExAlign(biasLocal[elementCount], this->biasGm[gmOffset], tensor_local, this->numLastDim,
                                    stride_, rowCount, padParams);
                }
            }
#else
            LocalTensor<T> biasLocal = yBufFp32.template Get<T>();
            for (auto i = 0; i < rowCount; i++) {
                DataCopyExV2(x1x2LocalIn[i * this->numLastDimRoundUp32], this->x1Gm[gmOffset + i * this->numLastDim],
                             tensor_local, this->numLastDim, 1);
                DataCopyExV2(x1x2LocalIn[elementCount + i * this->numLastDimRoundUp32],
                             this->x2Gm[gmOffset + i * this->numLastDim], tensor_local, this->numLastDim, 1);
                if constexpr (IS_BIAS_ELEWISE) {
                    if constexpr (is_same<float, T>::value) {
                        DataCopyExV2(biasLocal[i * this->numLastDimRoundUp32],
                                     this->biasGm[gmOffset + i * this->numLastDim], tensor_local, this->numLastDim);
                    } else {
                        DataCopyExV2(biasLocal[elementCount + i * this->numLastDimRoundUp32],
                                     this->biasGm[gmOffset + i * this->numLastDim], tensor_local, this->numLastDim);
                    }
                }
            }
#endif
        } else {
            DataCopyEx(x1x2LocalIn[0], this->x1Gm[gmOffset], this->numLastDim, rowCount, padParams);
            DataCopyEx(x1x2LocalIn[elementCount], this->x2Gm[gmOffset], this->numLastDim, rowCount, padParams);
            if constexpr (IS_BIAS_ELEWISE) {
                LocalTensor<T> biasLocal = yBufFp32.template Get<T>();
                if constexpr (is_same<float, T>::value) {
                    DataCopyEx(biasLocal, this->biasGm[gmOffset], this->numLastDim, rowCount, padParams);
                } else {
                    DataCopyEx(biasLocal[elementCount], this->biasGm[gmOffset], this->numLastDim, rowCount, padParams);
                }
            }
        }
        inRowsQue.EnQue(x1x2LocalIn);
    }

    __aicore__ inline void AddX1X2BiasFp32(int32_t elementCount, int32_t rowCount, LocalTensor<float> xLocalFp32,
                                           LocalTensor<T> x1Local, LocalTensor<T> x2Local)
    {
        Add(xLocalFp32, x1Local, x2Local, elementCount);
        PipeBarrier<PIPE_V>();
        if constexpr (IS_BIAS_BROADCAST) {
            auto biasLocal = biasBuf.template Get<T>();
            for (int i = 0; i < rowCount; i++) {
                Add(xLocalFp32[i * alignedStride], biasLocal, xLocalFp32[i * alignedStride], this->numLastDim);
            }
        } else if constexpr (IS_BIAS_ELEWISE) {
            LocalTensor<float> yLocalFp32 = yBufFp32.Get<float>();
            Add(xLocalFp32, yLocalFp32, xLocalFp32, elementCount);
        }
    }

    __aicore__ inline void AddX1X2BiasFp16(int32_t elementCount, int32_t rowCount, LocalTensor<float> xLocalFp32,
                                           LocalTensor<float> yLocalFp32, LocalTensor<T> x1Local,
                                           LocalTensor<T> x2Local, LocalTensor<T> x1x2Local)
    {
        if constexpr (IS_BIAS_BROADCAST) {
            auto biasLocal = biasBuf.template Get<T>();
            Cast(xLocalFp32, x1Local, RoundMode::CAST_NONE, elementCount);
            Cast(yLocalFp32, x2Local, RoundMode::CAST_NONE, elementCount);
            PipeBarrier<PIPE_V>();
            Add(xLocalFp32, xLocalFp32, yLocalFp32, elementCount);
            Cast(x1x2Local.template ReinterpretCast<float>(), biasLocal, RoundMode::CAST_NONE, this->numLastDim);
            PipeBarrier<PIPE_V>();
            for (int i = 0; i < rowCount; i++) {
                Add(xLocalFp32[i * alignedStride], x1x2Local.template ReinterpretCast<float>(),
                    xLocalFp32[i * alignedStride], this->numLastDim);
            }
        } else if constexpr (IS_BIAS_ELEWISE) {
            auto biaslocal = yLocalFp32.ReinterpretCast<T>()[elementCount];
            Cast(xLocalFp32, x1Local, RoundMode::CAST_NONE, elementCount);
            PipeBarrier<PIPE_V>();
            Cast(yLocalFp32, biaslocal, RoundMode::CAST_NONE, elementCount);
            PipeBarrier<PIPE_V>();
            Add(xLocalFp32, xLocalFp32, yLocalFp32, elementCount);
            PipeBarrier<PIPE_V>();
            Cast(yLocalFp32, x2Local, RoundMode::CAST_NONE, elementCount);
            PipeBarrier<PIPE_V>();
            Add(xLocalFp32, xLocalFp32, yLocalFp32, elementCount);
        } else {
            Cast(xLocalFp32, x1Local, RoundMode::CAST_NONE, elementCount);
            PipeBarrier<PIPE_V>();
            Cast(yLocalFp32, x2Local, RoundMode::CAST_NONE, elementCount);
            PipeBarrier<PIPE_V>();
            Add(xLocalFp32, xLocalFp32, yLocalFp32, elementCount);
        }
    }

    __aicore__ inline void AddX1X2Bias(int32_t elementCount, int32_t rowCount)
    {
        LocalTensor<float> xLocalFp32 = xBufFp32.Get<float>();
        LocalTensor<float> yLocalFp32 = yBufFp32.Get<float>();
        LocalTensor<T> x1x2Local = inRowsQue.template DeQue<T>();
        auto x1Local = x1x2Local[0];
        auto x2Local = x1x2Local[elementCount];

        if constexpr (is_same<float, T>::value) {
            AddX1X2BiasFp32(elementCount, rowCount, xLocalFp32, x1Local, x2Local);
        } else {
            AddX1X2BiasFp16(elementCount, rowCount, xLocalFp32, yLocalFp32, x1Local, x2Local, x1x2Local);
        }
        PipeBarrier<PIPE_V>();
        inRowsQue.FreeTensor(x1x2Local);
    }

    __aicore__ inline void CopyOutAdditionalOutput(uint64_t gmOffset, int32_t rowCount, int32_t elementCount)
    {
        if (this->isXOut) {
            LocalTensor<float> addBufLocal = xBufFp32.Get<float>();
            auto xLocal = outRowQue.template AllocTensor<T>();
            if constexpr (is_same<T, float>::value) {
                Adds(xLocal, addBufLocal, ZERO, elementCount);
            } else if constexpr (is_same<T, half>::value) {
                Cast(xLocal, addBufLocal, RoundMode::CAST_NONE, elementCount);
            } else {
                Cast(xLocal, addBufLocal, RoundMode::CAST_RINT, elementCount);
            }
            PipeBarrier<PIPE_V>();
            outRowQue.template EnQue<T>(xLocal);

            auto x = outRowQue.template DeQue<T>();
            PipeBarrier<PIPE_V>();

            if (layernormResExist) {
                LocalTensor<T> tensor_local = tensor_buf.Get<T>();
                uint32_t stride_ = (this->numLastDimRoundUp32 - this->numLastDimAligned) * sizeof(T) / 32;
#if __NPU_ARCH__ == 2201
                DataCopyExAlign(this->xGm[gmOffset], x, tensor_local, this->numLastDim, stride_, rowCount);
#else
                for (auto i = 0; i < rowCount; i++) {
                    DataCopyExV2(this->xGm[gmOffset + i * this->numLastDim], x[i * this->numLastDimRoundUp32],
                                 tensor_local, this->numLastDim, 1);
                }
#endif
            } else {
                DataCopyEx(this->xGm[gmOffset], x, this->numLastDim, rowCount);
            }
            outRowQue.FreeTensor(x);
        }
    }

    __aicore__ inline void ApplyGammaBeta(int32_t roundOffset, int32_t elementCount, LocalTensor<float>& xLocalFp32,
                                          LocalTensor<float>& yLocalFp32, LocalTensor<T>& gammaLocal,
                                          LocalTensor<T>& betaLocal)
    {
        if constexpr (!is_same<T, float>::value) {
            Cast(yLocalFp32, gammaLocal, RoundMode::CAST_NONE, this->numLastDim);
            PipeBarrier<PIPE_V>();
            Mul(xLocalFp32[roundOffset], yLocalFp32, xLocalFp32[roundOffset], this->numLastDim);
            PipeBarrier<PIPE_V>();
            Cast(yLocalFp32, betaLocal, RoundMode::CAST_NONE, this->numLastDim);
            PipeBarrier<PIPE_V>();
            Add(xLocalFp32[roundOffset], yLocalFp32, xLocalFp32[roundOffset], this->numLastDim);
        } else {
            Mul(yLocalFp32, xLocalFp32[roundOffset], gammaLocal, this->numLastDim);
            PipeBarrier<PIPE_V>();
            Add(xLocalFp32[roundOffset], yLocalFp32, betaLocal, this->numLastDim);
        }
    }

    __aicore__ inline void ComputeLayerNorm(int32_t gmOffset, int32_t nums, int32_t elementCount,
                                            LocalTensor<T>& gammaLocal, LocalTensor<T>& betaLocal)
    {
        LocalTensor<float> xLocalFp32 = xBufFp32.Get<float>(); // xLocalFp32 <-- x1 + x2 + bias
        LocalTensor<float> yLocalFp32 = yBufFp32.Get<float>();

        Muls(yLocalFp32, xLocalFp32, this->aveNum, elementCount); // yLocalFp32 <-- x / N
        PipeBarrier<PIPE_V>();

        // reduce#1 for mean
        for (int32_t rid = 0; rid < nums; ++rid) {
            auto roundOffset = rid * alignedStride;
            auto aveLocalTemp = ReduceSumFP32(yLocalFp32[roundOffset], this->numLastDim); // aveLocalTemp <-- E(x)

            event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(eventSV);
            WaitFlag<HardEvent::S_V>(eventSV);
            Adds(xLocalFp32[roundOffset], xLocalFp32[roundOffset], aveLocalTemp * -1,
                 this->numLastDim); // xLocalFp32 <-- x - E(x)
        }
        PipeBarrier<PIPE_V>();

        Mul(yLocalFp32, xLocalFp32, xLocalFp32, elementCount); // yLocalFp32 <-- (x - E(x))**2
        PipeBarrier<PIPE_V>();
        Muls(yLocalFp32, yLocalFp32, this->aveNum, elementCount); // yLocalFp32 <-- (x - E(x))**2 / N
        PipeBarrier<PIPE_V>();

        // reduce#2 for var
        for (int32_t rid = 0; rid < nums; ++rid) {
            auto roundOffset = rid * alignedStride;
            float varLocalTemp = ReduceSumFP32(yLocalFp32[roundOffset], this->numLastDim); // varLocalTemp <-- Var(x)
            float rstdLocalTemp = 1 / sqrt(varLocalTemp + this->eps);                      // rstdLocalTemp <-- rstd

            event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(eventSV);
            WaitFlag<HardEvent::S_V>(eventSV);
            Muls(xLocalFp32[roundOffset], xLocalFp32[roundOffset], rstdLocalTemp,
                 this->numLastDim); // xLocalFp32 <-- (x - E(x)) * rstd
            PipeBarrier<PIPE_V>();
            ApplyGammaBeta(roundOffset, elementCount, xLocalFp32, yLocalFp32, gammaLocal, betaLocal);
        }
        PipeBarrier<PIPE_V>();
#if __NPU_ARCH__ != 2002
        if (!layernormResExist) {
            if constexpr (is_same<T, half>::value) {
                Cast(yLocalFp32.template ReinterpretCast<T>(), xLocalFp32, RoundMode::CAST_NONE, elementCount);
                PipeBarrier<PIPE_V>();
                Cast(xLocalFp32, yLocalFp32.template ReinterpretCast<T>(), RoundMode::CAST_NONE, elementCount);
                PipeBarrier<PIPE_V>();
            } else if constexpr (is_same<T, bfloat16_t>::value) {
                Cast(yLocalFp32.template ReinterpretCast<T>(), xLocalFp32, RoundMode::CAST_RINT, elementCount);
                PipeBarrier<PIPE_V>();
                Cast(xLocalFp32, yLocalFp32.template ReinterpretCast<T>(), RoundMode::CAST_NONE, elementCount);
                PipeBarrier<PIPE_V>();
            }
        }
#endif
        CopyOutLayernormRes(gmOffset, nums, elementCount);
    }

    __aicore__ inline void CopyOutLayernormRes(int32_t gmOffset, int32_t nums, int32_t elementCount)
    {
        if (!layernormResExist)
            return;

        LocalTensor<float> xLocalFp32 = xBufFp32.Get<float>();
        LocalTensor<T> resLocal = outRowQue.template AllocTensor<T>();
        LocalTensor<T> tensor_local = tensor_buf.Get<T>();

        if constexpr (is_same<T, float>::value) {
            Adds(resLocal, xLocalFp32, ZERO, elementCount);
        } else if constexpr (is_same<T, half>::value) {
            Cast(resLocal, xLocalFp32, RoundMode::CAST_NONE, elementCount);
        } else {
            Cast(resLocal, xLocalFp32, RoundMode::CAST_RINT, elementCount);
        }
        PipeBarrier<PIPE_V>();
#include "add_layer_norm_static_quant_normal_kernel_part1.h"
