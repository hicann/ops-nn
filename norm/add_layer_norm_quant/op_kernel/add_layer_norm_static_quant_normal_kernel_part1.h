/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* Internal continuation of add_layer_norm_static_quant_normal_kernel.h. */
outRowQue.template EnQue<T>(resLocal);
LocalTensor<T> resDeq = outRowQue.template DeQue<T>();

#if __NPU_ARCH__ == 2201
uint32_t stride_ = (this->numLastDimRoundUp32 - this->numLastDimAligned) * sizeof(T) / 32;
DataCopyExAlign(this->resGm[gmOffset], resDeq, tensor_local, this->numLastDim, stride_, nums);
#else
for (auto i = 0; i < nums; i++) {
    DataCopyExV2(this->resGm[gmOffset + i * this->numLastDim], resDeq[i * this->numLastDimRoundUp32], tensor_local,
                 this->numLastDim, 1);
}
#endif
outRowQue.FreeTensor(resDeq);
}

__aicore__ inline void ComputeStaticQuant(int32_t nums, int32_t elementCount)
{
    if (!this->scales2Exist) {
        ComputeSoleStaticQuant(nums, elementCount);
    } else {
        ComputeDualStaticQuant(nums, elementCount);
    }
}

__aicore__ inline void ComputeSoleStaticQuant(int32_t nums, int32_t elementCount)
{
    LocalTensor<float> xLocalFp32 = xBufFp32.Get<float>(); // xLocalFp32 <-- y
    LocalTensor<float> yLocalFp32 = yBufFp32.Get<float>();

    if (this->isPerTensor && layernormResExist) {
        Muls(xLocalFp32, xLocalFp32, perTensorScale1, elementCount);
        PipeBarrier<PIPE_V>();
        if (isZeroPoint1Exist) {
            Adds(xLocalFp32, xLocalFp32, perTensorOffset1, elementCount);
            PipeBarrier<PIPE_V>();
        }
    } else {
        LocalTensor<S> scalesOffsetLocal = inRowsQue.template DeQue<S>();
        LocalTensor<float> tmpLocal = scalesOffsetLocal.template ReinterpretCast<float>();
        auto scalesLocal = scalesOffsetLocal[0];
        auto offsetsLocal = scalesOffsetLocal[alignedStride];

        CastToFloat<T>(yLocalFp32, scalesLocal, this->numLastDim);
        PipeBarrier<PIPE_V>();
        for (int32_t rid = 0; rid < nums; ++rid) {
            if (layernormResExist) {
                Mul(xLocalFp32[rid * alignedStride], xLocalFp32[rid * alignedStride], yLocalFp32,
                    this->numLastDim); // xLocalFp32 <-- y * scales1
            } else {
                Div(xLocalFp32[rid * alignedStride], xLocalFp32[rid * alignedStride], yLocalFp32,
                    this->numLastDim); // xLocalFp32 <-- y / scales1
            }
        }
        PipeBarrier<PIPE_V>();
        if (isZeroPoint1Exist) {
            CastToFloat<T>(tmpLocal, offsetsLocal, this->numLastDim);
            PipeBarrier<PIPE_V>();
            for (int32_t rid = 0; rid < nums; ++rid) {
                Add(xLocalFp32[rid * alignedStride], xLocalFp32[rid * alignedStride], tmpLocal, this->numLastDim);
            }
        }
        inRowsQue.FreeTensor(scalesOffsetLocal);
        PipeBarrier<PIPE_V>();
    }
    LocalTensor<int8_t> yLocal = outRowQue.template AllocTensor<int8_t>();
    RoundFloat2Int8(yLocal, xLocalFp32, elementCount);
    PipeBarrier<PIPE_V>();
    outRowQue.EnQue(yLocal);
}

__aicore__ inline void ApplyScales2AndOffsets(int32_t nums, int32_t alignedStride, LocalTensor<float>& xLocalFp32,
                                              LocalTensor<float>& yLocalFp32, LocalTensor<S>& scalesOffsetLocal,
                                              LocalTensor<float>& tmpLocal)
{
    auto scalesLocal = scalesOffsetLocal[0];
    auto offsetsLocal = scalesOffsetLocal[alignedStride];

    CastToFloat<S>(yLocalFp32, scalesLocal, this->numLastDim);
    PipeBarrier<PIPE_V>();
    for (int32_t rid = 0; rid < nums; ++rid) {
        if (layernormResExist) {
            Mul(xLocalFp32[rid * alignedStride], xLocalFp32[rid * alignedStride], yLocalFp32, this->numLastDim);
        } else {
            Div(xLocalFp32[rid * alignedStride], xLocalFp32[rid * alignedStride], yLocalFp32, this->numLastDim);
        }
    }
    PipeBarrier<PIPE_V>();
    if (isZeroPoint2Exist) {
        CastToFloat<S>(tmpLocal, offsetsLocal, this->numLastDim);
        PipeBarrier<PIPE_V>();
        for (int32_t rid = 0; rid < nums; ++rid) {
            Add(xLocalFp32[rid * alignedStride], xLocalFp32[rid * alignedStride], tmpLocal, this->numLastDim);
        }
    }
    inRowsQue.FreeTensor(scalesOffsetLocal);
}

__aicore__ inline void ComputeDualStaticQuant(int32_t nums, int32_t elementCount)
{
    LocalTensor<float> xLocalFp32 = xBufFp32.Get<float>(); // xLocalFp32 <-- y
    LocalTensor<float> yLocalFp32 = yBufFp32.Get<float>();

    if (this->isPerTensor && layernormResExist) {
        Muls(yLocalFp32, xLocalFp32, perTensorScale1, elementCount);
        PipeBarrier<PIPE_V>();
        if (isZeroPoint1Exist) {
            Adds(yLocalFp32, yLocalFp32, perTensorOffset1, elementCount);
            PipeBarrier<PIPE_V>();
        }
    } else {
        LocalTensor<S> scalesOffsetLocal = inRowsQue.template DeQue<S>();
        LocalTensor<float> tmpLocal = scalesOffsetLocal.template ReinterpretCast<float>();
        auto scalesLocal = scalesOffsetLocal[0];
        auto offsetsLocal = scalesOffsetLocal[alignedStride];

        auto scales1Fp32 = yLocalFp32[(nums - 1) * alignedStride];
        CastToFloat<S>(scales1Fp32, scalesLocal, this->numLastDim);
        PipeBarrier<PIPE_V>();
        for (int32_t rid = 0; rid < nums; ++rid) {
            if (layernormResExist) {
                Mul(yLocalFp32[rid * alignedStride], xLocalFp32[rid * alignedStride], scales1Fp32, this->numLastDim);
            } else {
                Div(yLocalFp32[rid * alignedStride], xLocalFp32[rid * alignedStride], scales1Fp32, this->numLastDim);
            }
        }
        PipeBarrier<PIPE_V>();
        if (isZeroPoint1Exist) {
            CastToFloat<S>(tmpLocal, offsetsLocal, this->numLastDim);
            PipeBarrier<PIPE_V>();
            for (int32_t rid = 0; rid < nums; ++rid) {
                Add(yLocalFp32[rid * alignedStride], yLocalFp32[rid * alignedStride], tmpLocal, this->numLastDim);
            }
        }
        inRowsQue.FreeTensor(scalesOffsetLocal);
    }
    PipeBarrier<PIPE_V>();

    CopyInScaleOffset(scales2Gm, zeroPoints2Gm, isZeroPoint2Exist);

    LocalTensor<int8_t> y12Local = outRowQue.template AllocTensor<int8_t>();
    auto y1Local = y12Local[0];
    auto y2Local = y12Local[nums * alignedStride];
    RoundFloat2Int8(y1Local, yLocalFp32, elementCount);
    PipeBarrier<PIPE_V>();

    LocalTensor<S> scalesOffsetLocal = inRowsQue.template DeQue<S>();
    LocalTensor<float> tmpLocal = scalesOffsetLocal.template ReinterpretCast<float>();
    ApplyScales2AndOffsets(nums, alignedStride, xLocalFp32, yLocalFp32, scalesOffsetLocal, tmpLocal);
    PipeBarrier<PIPE_V>();

    RoundFloat2Int8(y2Local, xLocalFp32, elementCount);
    PipeBarrier<PIPE_V>();

    outRowQue.EnQue(y12Local);
}

__aicore__ inline void CopyOut(uint64_t gmOffset, uint64_t gmOffsetScale, int32_t rowCount)
{
    LocalTensor<int8_t> outY12 = outRowQue.template DeQue<int8_t>();
    auto outY1 = outY12[0];
    auto outY2 = outY12[this->rowStep * alignedStride];

    if (layernormResExist) {
        LocalTensor<int8_t> tensor_local = tensor_buf.Get<int8_t>();
#if __NPU_ARCH__ == 2201
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = rowCount;
        dataCopyParams.blockLen = this->numLastDim * 1;
        DataCopyPad(this->y1Gm[gmOffset], outY1, dataCopyParams);
#else
        for (auto i = 0; i < rowCount; i++) {
            DataCopyExV2(this->y1Gm[gmOffset + i * this->numLastDim], outY1[i * this->numLastDimRoundUp32],
                         tensor_local, this->numLastDim, 1);
        }
#endif
        if (this->scales2Exist) {
            DataCopyExV2(this->y2Gm[gmOffset], outY2, tensor_local, this->numLastDim, rowCount);
        }
    } else {
        DataCopyEx(this->y1Gm[gmOffset], outY1, this->numLastDim, rowCount);
        if (this->scales2Exist) {
            DataCopyEx(this->y2Gm[gmOffset], outY2, this->numLastDim, rowCount);
        }
    }
    outRowQue.FreeTensor(outY12);
}

__aicore__ inline void CopyInGammaBeta()
{
    LocalTensor<T> gammaLocal = gammaBuf.template Get<T>();
    LocalTensor<T> betaLocal = betaBuf.template Get<T>();
    if (layernormResExist) {
        LocalTensor<T> tensor_local = tensor_buf.Get<T>();
        DataCopyExV2(gammaLocal, this->gammaGm, tensor_local, this->numLastDim);
        PipeBarrier<PIPE_V>();
        DataCopyExV2(betaLocal, this->betaGm, tensor_local, this->numLastDim);
        PipeBarrier<PIPE_V>();
        if constexpr (IS_BIAS_BROADCAST) {
            LocalTensor<T> biasLocal = biasBuf.template Get<T>();
            DataCopyExV2(biasLocal, this->biasGm, tensor_local, this->numLastDim);
        }
    } else {
        DataCopyEx(gammaLocal, this->gammaGm, this->numLastDim);
        DataCopyEx(betaLocal, this->betaGm, this->numLastDim);
        if constexpr (IS_BIAS_BROADCAST) {
            LocalTensor<T> biasLocal = biasBuf.template Get<T>();
            DataCopyEx(biasLocal, this->biasGm, this->numLastDim);
        }
    }
}

__aicore__ inline void CopyInScaleOffset(GlobalTensor<S> scalesGM, GlobalTensor<S> offsetsGM, bool hasOffset)
{
    if (this->isPerTensor) {
        return;
    }
    LocalTensor<S> scalesOffsetCopyIn = inRowsQue.template AllocTensor<S>();
    if (layernormResExist) {
        LocalTensor<S> tensor_local = tensor_buf.Get<S>();
        DataCopyExV2(scalesOffsetCopyIn[0], scalesGM, tensor_local, this->numLastDim);
        if (hasOffset) {
            DataCopyExV2(scalesOffsetCopyIn[alignedStride], offsetsGM, tensor_local, this->numLastDim);
        }
    } else {
        DataCopyEx(scalesOffsetCopyIn[0], scalesGM, this->numLastDim);
        if (hasOffset) {
            DataCopyEx(scalesOffsetCopyIn[alignedStride], offsetsGM, this->numLastDim);
        }
    }
    inRowsQue.EnQue(scalesOffsetCopyIn);
}

private:
TPipe* Ppipe = nullptr;
TQue<QuePosition::VECIN, BUFFER_NUM> inRowsQue;
TQue<QuePosition::VECOUT, BUFFER_NUM> outRowQue;

TBuf<TPosition::VECCALC> xBufFp32;
TBuf<TPosition::VECCALC> yBufFp32;
TBuf<TPosition::VECCALC> betaBuf;
TBuf<TPosition::VECCALC> gammaBuf;
TBuf<TPosition::VECCALC> biasBuf;
TBuf<TPosition::VECCALC> tensor_buf;

GlobalTensor<S> scales1Gm;
GlobalTensor<S> scales2Gm;
GlobalTensor<S> zeroPoints1Gm;
GlobalTensor<S> zeroPoints2Gm;
GlobalTensor<T> resGm;

bool scales1Exist = false;
bool scales2Exist = false;
bool isZeroPoint1Exist = false;
bool isZeroPoint2Exist = false;
bool layernormResExist = false;
int32_t alignedStride = 0;

float perTensorScale1 = 1.0f;
float perTensorOffset1 = 0.0f;
}
;

#endif // __ADD_LAYER_NORM_STATIC_QUANT_NORMAL_KERNEL_H_
