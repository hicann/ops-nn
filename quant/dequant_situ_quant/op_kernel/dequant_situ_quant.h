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
 * \file dequant_situ_quant.h
 * \brief DequantSituQuant kernel: Dequant -> Situ -> Quant
 */

#ifndef DEQUANT_SITU_QUANT_H
#define DEQUANT_SITU_QUANT_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include <type_traits>

#define TEMPLATE_DSQ_DECLARE template <bool hasDequantBias>
#define TEMPLATE_DSQ_ARGS hasDequantBias

namespace DequantSituQuantOps {
using namespace AscendC;

constexpr static int64_t DB_BUFFER = 1;
constexpr static int64_t BLOCK_SIZE = 32;
constexpr static int64_t BLOCK_ELEM = BLOCK_SIZE / sizeof(float);
constexpr static int64_t MASK_NUM_T32 = 256 / sizeof(float);
constexpr static int64_t MASK_BLK_STRIDE = 8;
constexpr static int64_t ELEM_PER_REP_FP32 = 64;
constexpr static int64_t MAX_REPEAT = 255;
constexpr static int64_t SWI_FACTOR = 2;
constexpr static float DYNAMIC_QUANT_FACTOR = 1.0 / 127.0;

TEMPLATE_DSQ_DECLARE
class DequantSituQuantKernel {
public:
    __aicore__ inline DequantSituQuantKernel(TPipe* pipe) { pipe_ = pipe; }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR dequantScale, GM_ADDR dequantBias, GM_ADDR quantScale,
                                GM_ADDR quantOffset, GM_ADDR y, GM_ADDR yScale, GM_ADDR workspace,
                                const DequantSituQuantTilingData* tilingData)
    {
        tl_ = tilingData;
        blockIdx_ = GetBlockIdx();

        rowLen_ = tl_->rowLen;
        colLen_ = tl_->colLen;
        inDimy_ = colLen_ * SWI_FACTOR;
        outDimy_ = colLen_;
        baseRowLen_ = tl_->baseRowLen;
        baseColLen_ = tl_->baseColLen < colLen_ ? tl_->baseColLen : colLen_;
        usedCoreNum_ = tl_->usedCoreNum;
        activateLeft_ = tl_->activateLeft;
        quantType_ = tl_->quantType;
        quantIsOne_ = tl_->quantIsOne;
        dequantScaleIsOne_ = tl_->dequantScaleIsOne;
        dequantBiasIsOne_ = tl_->dequantBiasIsOne;
        quantScaleIsEmpty_ = tl_->quantScaleIsEmpty;
        quantOffsetIsEmpty_ = tl_->quantOffsetIsEmpty;
        quantOffsetIsOne_ = static_cast<bool>(tl_->quantOffsetIsOne);
        beta_ = tl_->beta;
        linearBeta_ = tl_->linearBeta;

        if (rowLen_ < usedCoreNum_) {
            usedCoreNum_ = rowLen_;
        }
        int64_t perRoundCnt = usedCoreNum_ == 0 ? 0 : rowLen_ / usedCoreNum_;
        int64_t remainCnt = rowLen_ - usedCoreNum_ * perRoundCnt;
        curCoreRowNum_ = perRoundCnt;
        if (blockIdx_ < remainCnt) {
            curCoreRowNum_ = perRoundCnt + 1;
            inputCopyOffset_ = blockIdx_ * curCoreRowNum_;
        } else {
            inputCopyOffset_ = remainCnt * (perRoundCnt + 1) + (blockIdx_ - remainCnt) * perRoundCnt;
        }

        xGm_.SetGlobalBuffer((__gm__ int8_t*)x + inputCopyOffset_ * inDimy_, curCoreRowNum_ * inDimy_);
        dequantScaleGm_.SetGlobalBuffer((__gm__ float*)dequantScale);
        if constexpr (hasDequantBias) {
            dequantBiasGm_.SetGlobalBuffer((__gm__ float*)dequantBias);
        }
        if (quantScaleIsEmpty_ == 0) {
            quantScaleGm_.SetGlobalBuffer((__gm__ float*)quantScale);
        }
        if (quantOffsetIsEmpty_ == 0) {
            quantOffsetGm_.SetGlobalBuffer((__gm__ float*)quantOffset);
        }
        yGm_.SetGlobalBuffer((__gm__ int8_t*)y + inputCopyOffset_ * outDimy_, curCoreRowNum_ * outDimy_);
        yScaleGm_.SetGlobalBuffer((__gm__ float*)yScale + inputCopyOffset_, curCoreRowNum_);

        if (quantScaleIsEmpty_ == 0 && quantIsOne_) {
            quantScaleVal_ = quantScaleGm_.GetValue(0);
            if (quantScaleVal_ == 0.0f) {
                quantScaleVal_ = 1.0f;
            } else if (quantType_ != 1) {
                // Static mode: quant_scale is a divisor (y = situ_out / quant_scale).
                // Take reciprocal so Muls(situOut, situOut, quantScaleVal_) performs division.
                quantScaleVal_ = 1.0f / quantScaleVal_;
            }
            // Dynamic mode (quantType_ == 1): quant_scale is a smoothScale multiplier
            // (situ_out *= quant_scale before absmax). Keep raw value — this matches
            // the !quantIsOne_ path which uses Mul(situOut, situOut, quantLocal_).
        }
        if (quantOffsetIsEmpty_ == 0 && quantOffsetIsOne_) {
            quantOffsetVal_ = quantOffsetGm_.GetValue(0);
        }

        // dequantScaleIsOne_/dequantBiasIsOne_ are host-authoritative (delivered
        // via tiling). GlobalTensor::GetSize() must NOT be used here: SetGlobalBuffer
        // is called with a bare aclnn GM_ADDR and no explicit size, so GetSize()
        // returns 0 and a real scalar (shape [1]) would be misdetected as vector,
        // causing the gate-half CopyIn to read out of bounds (index outDimy_).
        if (dequantScaleIsOne_) {
            dequantScaleVal_ = dequantScaleGm_.GetValue(0);
        }
        if constexpr (hasDequantBias) {
            if (dequantBiasIsOne_) {
                dequantBiasVal_ = dequantBiasGm_.GetValue(0);
            }
        }

        curColNum_ = baseColLen_;
        InitUbBuffer();
    }

    __aicore__ inline void Process()
    {
        if (blockIdx_ >= usedCoreNum_) {
            return;
        }
        processCompute();
    }

protected:
    __aicore__ inline void InitUbBuffer()
    {
        int64_t alignColNum = curColNum_ == Align(curColNum_, sizeof(int8_t)) ? curColNum_ :
                                                                                Align(curColNum_, sizeof(int8_t));
        int64_t alignInDimy = alignColNum * SWI_FACTOR;

        pipe_->InitBuffer(inQueueX_, DB_BUFFER, alignInDimy * sizeof(int8_t));
        pipe_->InitBuffer(dequantScaleBuf_, alignInDimy * sizeof(float));
        if constexpr (hasDequantBias) {
            pipe_->InitBuffer(dequantBiasBuf_, alignInDimy * sizeof(float));
        }
        // quantBuf_ is needed when quant_scale is vector (!quantIsOne_), or when
        // quant_offset is vector (!quantOffsetIsOne_) while quant_scale is scalar.
        // In the latter case only the offset half [H] is used; quant_scale uses Muls.
        if (quantScaleIsEmpty_ == 0 && (!quantIsOne_ || (quantOffsetIsEmpty_ == 0 && !quantOffsetIsOne_))) {
            int64_t quantBufElems = (quantType_ == 1) ? alignColNum : alignInDimy;
            pipe_->InitBuffer(quantBuf_, quantBufElems * sizeof(float));
        }
        // outQueue_ must hold max(int8 output, float situ output) + scale + padding
        // Situ computation uses outQueue as float buffer [H] floats = H*4 bytes
        // Final output is [H] int8 = H bytes + scale [1] float = 4 bytes
        pipe_->InitBuffer(outQueue_, 1, alignColNum * sizeof(float) + sizeof(float) + BLOCK_SIZE);

        // temp buffers for compute: dequantOut[2H] + situTemp[2H] = 4H floats
        pipe_->InitBuffer(tmpBuf_, alignInDimy * SWI_FACTOR * sizeof(float));
        // cast buffer for int8<->float conversion intermediates
        pipe_->InitBuffer(castBuf_, alignInDimy * SWI_FACTOR * sizeof(float));
    }

    __aicore__ inline void CopyInDequantParams(int64_t colOffset)
    {
        // Sync V→MTE2: ensure previous tile's V operations finish before
        // overwriting dequantScaleBuf_ (TBuf has no automatic pipeline sync)
        event_t eventV2MTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventV2MTE2);
        WaitFlag<HardEvent::V_MTE2>(eventV2MTE2);

        if (!dequantScaleIsOne_) {
            uint8_t rPad = FloatRightPadding();
            DataCopyExtParams params = {1, static_cast<uint32_t>(curColNum_ * sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> padParams{true, 0, rPad, 0.0f};
            LocalTensor<float> dequantScaleLocal = dequantScaleBuf_.template Get<float>();
            DataCopyPad(dequantScaleLocal, dequantScaleGm_[colOffset], params, padParams);
            PadFloatTail(dequantScaleLocal, 0);
            DataCopyPad(dequantScaleLocal[alignColNum_], dequantScaleGm_[outDimy_ + colOffset], params, padParams);
            PadFloatTail(dequantScaleLocal, alignColNum_);
            dequantScaleLocal_ = dequantScaleLocal;
        }

        if constexpr (hasDequantBias) {
            if (!dequantBiasIsOne_) {
                uint8_t rPad = FloatRightPadding();
                DataCopyExtParams params = {1, static_cast<uint32_t>(curColNum_ * sizeof(float)), 0, 0, 0};
                DataCopyPadExtParams<float> padParams{true, 0, rPad, 0.0f};
                LocalTensor<float> dequantBiasLocal = dequantBiasBuf_.template Get<float>();
                DataCopyPad(dequantBiasLocal, dequantBiasGm_[colOffset], params, padParams);
                PadFloatTail(dequantBiasLocal, 0);
                DataCopyPad(dequantBiasLocal[alignColNum_], dequantBiasGm_[outDimy_ + colOffset], params, padParams);
                PadFloatTail(dequantBiasLocal, alignColNum_);
                dequantBiasLocal_ = dequantBiasLocal;
            }
        }

        // Sync MTE2→V: ensure TBuf DataCopyPad completes before vector compute reads it
        event_t eventMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventMTE2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventMTE2ToV);
    }

    __aicore__ inline void CopyInQuantParams(int64_t colOffset)
    {
        // Load quant_scale vector and/or quant_offset vector into quantBuf_.
        // quantBuf_ layout: [0:H] = quant_scale (if !quantIsOne_), [H:2H] = quant_offset (if !quantOffsetIsOne_).
        // When quantIsOne_ but quant_offset is vector, only [H:2H] is loaded and used.
        bool needLoadScale = (quantScaleIsEmpty_ == 0 && !quantIsOne_);
        bool needLoadOffset = (quantOffsetIsEmpty_ == 0 && !quantOffsetIsOne_);
        if (needLoadScale || needLoadOffset) {
            uint8_t rPad = FloatRightPadding();
            DataCopyExtParams params = {1, static_cast<uint32_t>(curColNum_ * sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> padParams{true, 0, rPad, 0.0f};
            LocalTensor<float> quantLocal = quantBuf_.template Get<float>();
            if (needLoadScale) {
                DataCopyPad(quantLocal, quantScaleGm_[colOffset], params, padParams);
                PadFloatTail(quantLocal, 0);
            }
            if (needLoadOffset) {
                DataCopyPad(quantLocal[alignColNum_], quantOffsetGm_[colOffset], params, padParams);
                PadFloatTail(quantLocal, alignColNum_);
            }
            quantLocal_ = quantLocal;

            // Sync MTE2→V: ensure TBuf DataCopyPad completes before vector compute reads it
            event_t eventMTE2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
            SetFlag<HardEvent::MTE2_V>(eventMTE2ToV);
            WaitFlag<HardEvent::MTE2_V>(eventMTE2ToV);
        }
    }

    __aicore__ inline void CopyIn(int64_t rowIdx, int64_t colOffset)
    {
        // x layout: [up(0:H), gate(H:2H)] — load up and gate separately
        // isPad=true + rightPadding fills tail with paddingValue=0 to 32B boundary
        uint8_t padLen = static_cast<uint8_t>(alignColNum_ - curColNum_);
        DataCopyExtParams params = {1, static_cast<uint32_t>(curColNum_ * sizeof(int8_t)), 0, 0, 0};
        DataCopyPadExtParams<int8_t> padParams{true, 0, padLen, static_cast<int8_t>(0)};

        LocalTensor<int8_t> xLocal = inQueueX_.template AllocTensor<int8_t>();
        DataCopyPad(xLocal, xGm_[rowIdx * inDimy_ + colOffset], params, padParams);
        DataCopyPad(xLocal[alignColNum_], xGm_[rowIdx * inDimy_ + outDimy_ + colOffset], params, padParams);
        inQueueX_.EnQue(xLocal);
    }

    __aicore__ inline void ComputeDequant(int64_t rowIdx)
    {
        LocalTensor<int8_t> xLocalI8 = inQueueX_.template DeQue<int8_t>();

        LocalTensor<float> tmpF32 = tmpBuf_.template Get<float>();
        LocalTensor<float> dequantOut = tmpF32;
        LocalTensor<float> situTemp = tmpF32[alignTileLen_];

        // Step 1: Cast int8 -> half -> fp32
        LocalTensor<half> tmpHalf = castBuf_.template Get<half>();
        Cast(tmpHalf, xLocalI8, RoundMode::CAST_NONE, alignTileLen_);
        PipeBarrier<PIPE_V>();
        Cast(dequantOut, tmpHalf, RoundMode::CAST_NONE, alignTileLen_);
        PipeBarrier<PIPE_V>();
        inQueueX_.FreeTensor(xLocalI8);

        // Step 2: Mul dequant_scale
        if (dequantScaleIsOne_) {
            Muls(dequantOut, dequantOut, dequantScaleVal_, alignTileLen_);
        } else {
            Mul(dequantOut, dequantOut, dequantScaleLocal_, alignTileLen_);
        }
        PipeBarrier<PIPE_V>();

        // Step 3: Add dequant_bias (if exists)
        if constexpr (hasDequantBias) {
            if (dequantBiasIsOne_) {
                Adds(dequantOut, dequantOut, dequantBiasVal_, alignTileLen_);
                PipeBarrier<PIPE_V>();
            } else {
                Add(dequantOut, dequantOut, dequantBiasLocal_, alignTileLen_);
                PipeBarrier<PIPE_V>();
            }
        }

        // Store dequantOut for Situ computation
        dequantOut_ = dequantOut;
        situTemp_ = situTemp;
    }

    __aicore__ inline void ComputeSitu()
    {
        int64_t H = alignColNum_;
        LocalTensor<float> dequantOut = dequantOut_;
        LocalTensor<float> tmp = situTemp_;

        // gate and up: activateLeft=0 means gate=right half, up=left half
        // activateLeft=1 means gate=left half, up=right half
        int64_t gateOffset = (activateLeft_ == 1) ? 0 : H;
        int64_t upOffset = (activateLeft_ == 1) ? H : 0;

        LocalTensor<float> gate = dequantOut[gateOffset];
        LocalTensor<float> up = dequantOut[upOffset];

        // tmpBuf_ layout: [0:2H] = dequantOut (no longer needed), [2H:4H] = situTemp
        // Reuse situTemp for Situ computation:
        // tmp[0:H] = tanh result (beta * tanh(gate/beta))
        // tmp[H:2H] = sigmoid result + sigmoid denom
        LocalTensor<float> tanhResult = tmp;
        LocalTensor<float> sigmoidResult = tmp[H];

        // Step 1: tanh(gate / beta) * beta
        float invBeta = 1.0f / beta_;
        Muls(tanhResult, gate, invBeta, H);
        PipeBarrier<PIPE_V>();
        Tanh(tanhResult, tanhResult, H);
        PipeBarrier<PIPE_V>();

        Muls(tanhResult, tanhResult, beta_, H);
        PipeBarrier<PIPE_V>();

        // Step 2: sigmoid(gate) = 1 / (1 + exp(-gate))
        // Numerically stable: avoids positive-input exp overflow.
        LocalTensor<float> denomTmp = dequantOut[gateOffset];
        Muls(sigmoidResult, gate, -1.0f, H);
        PipeBarrier<PIPE_V>();
        Exp(sigmoidResult, sigmoidResult, H);
        PipeBarrier<PIPE_V>();

        Adds(denomTmp, sigmoidResult, 1.0f, H); // 1 + exp(-gate)
        PipeBarrier<PIPE_V>();

        // sigmoid = 1 / (1 + exp(-gate))
        // Use Level 0 Div instead of Reciprocal for better precision.
        // src0 is a single datablock of 1.0f, reused across all repeats via
        // src0BlkStride=0 and src0RepStride=0.
        LocalTensor<float> onesBlock = castBuf_.template Get<float>();
        Duplicate<float>(onesBlock, 1.0f, 8);
        PipeBarrier<PIPE_V>();

        constexpr uint64_t maskFp32 = static_cast<uint64_t>(ELEM_PER_REP_FP32);
        uint32_t fullReps = static_cast<uint32_t>(H / maskFp32);
        uint32_t remainder = static_cast<uint32_t>(H % maskFp32);
        BinaryRepeatParams divParams(1, 0, 1, 8, 0, 8);

        if (fullReps > 0) {
            Div(sigmoidResult, onesBlock, denomTmp, maskFp32, static_cast<uint8_t>(fullReps), divParams);
            PipeBarrier<PIPE_V>();
        }
        if (remainder > 0) {
            Div(sigmoidResult[fullReps * maskFp32], onesBlock, denomTmp[fullReps * maskFp32], remainder, 1, divParams);
            PipeBarrier<PIPE_V>();
        }

        // Step 3: situ_a = tanhResult * sigmoidResult
        Mul(tanhResult, tanhResult, sigmoidResult, H);
        PipeBarrier<PIPE_V>();

        // Step 4: if linear_beta > 0: up = linear_beta * tanh(up / linear_beta)
        if (linearBeta_ > 0.0f) {
            float invLinearBeta = 1.0f / linearBeta_;
            Muls(up, up, invLinearBeta, H);
            PipeBarrier<PIPE_V>();
            Tanh(up, up, H);
            PipeBarrier<PIPE_V>();
            Muls(up, up, linearBeta_, H);
            PipeBarrier<PIPE_V>();
        }

        // Step 5: output = situ_a * up = tanhResult * up
        // Write to gate buffer (no longer needed, avoids aliasing with up)
        LocalTensor<float> situOut = dequantOut[gateOffset];
        Mul(situOut, tanhResult, up, H);
        PipeBarrier<PIPE_V>();

        // situOut now holds the Situ output [H] in fp32, stored in gate buffer region
        situOut_ = situOut;
    }

    __aicore__ inline void ComputeQuant()
    {
        int64_t H = alignColNum_;
        LocalTensor<float> situOut = situOut_;

        if (quantType_ == 1) {
            // Dynamic quant
            DynamicQuant(situOut);
        } else {
            // Static quant
            StaticQuant(situOut);
        }
    }

    __aicore__ inline void StaticQuant(LocalTensor<float>& situOut)
    {
        int64_t H = alignColNum_;

        if (quantScaleIsEmpty_ == 0) {
            if (quantIsOne_) {
                Muls(situOut, situOut, quantScaleVal_, H);
                PipeBarrier<PIPE_V>();
                if (quantOffsetIsEmpty_ == 0) {
                    if (quantOffsetIsOne_) {
                        Adds(situOut, situOut, quantOffsetVal_, H);
                    } else {
                        // quant_offset is vector [H], loaded into quantLocal_[H]
                        Add(situOut, situOut, quantLocal_[H], H);
                    }
                    PipeBarrier<PIPE_V>();
                }
            } else {
                Div(situOut, situOut, quantLocal_, H);
                PipeBarrier<PIPE_V>();
                if (quantOffsetIsEmpty_ == 0) {
                    if (quantOffsetIsOne_) {
                        Adds(situOut, situOut, quantOffsetVal_, H);
                    } else {
                        Add(situOut, situOut, quantLocal_[H], H);
                    }
                    PipeBarrier<PIPE_V>();
                }
            }
        }

        // Allocate outQueue and cast fp32 -> int8
        LocalTensor<float> outLocal = outQueue_.template AllocTensor<float>();
        LocalTensor<int8_t> yOut = outLocal.template ReinterpretCast<int8_t>();
        CastFloatToInt8(situOut, yOut, H);
        outQueue_.EnQue<float>(outLocal);
    }

    __aicore__ inline void DynamicQuant(LocalTensor<float>& situOut)
    {
        int64_t H = alignColNum_;

        // Apply smooth scale (same semantics as ApplySmoothScale in the
        // column-tiling path). When quantIsOne_, quantLocal_ / quantBuf_ are
        // never initialized (InitBuffer and CopyInQuantParams are both gated
        // on !quantIsOne_), so accessing quantLocal_ here would read an
        // uninitialised UB address → "VEC instruction not aligned" crash.
        // Use the scalar Muls path in that case.
        if (quantScaleIsEmpty_ == 0) {
            if (quantIsOne_) {
                Muls(situOut, situOut, quantScaleVal_, H);
            } else {
                Mul(situOut, situOut, quantLocal_, H);
            }
            PipeBarrier<PIPE_V>();
        }

        // Compute per-row abs max using situTemp_ (no longer needed after Situ)
        LocalTensor<float> absBuf = situTemp_;
        Abs(absBuf, situOut, H);
        PipeBarrier<PIPE_V>();

        // Zero out padding region [curColNum_, alignColNum_) via multiply-mask:
        // pad holds dequant(bias) values which would pollute the absmax.
        // maskTensor[i] = (i < curColNum_) ? 1 : 0 — built with an aligned
        // full-zero Duplicate plus a masked Adds over the valid elements only.
        if (alignColNum_ > curColNum_) {
            LocalTensor<float> maskTensor = castBuf_.template Get<float>();
            Duplicate<float>(maskTensor, 0.0f, alignColNum_);
            PipeBarrier<PIPE_V>();
            Adds(maskTensor, maskTensor, 1.0f, curColNum_);
            PipeBarrier<PIPE_V>();
            Mul(absBuf, absBuf, maskTensor, alignColNum_);
            PipeBarrier<PIPE_V>();
        }

        // Allocate outQueue for final output: [H int8][1 float scale]
        LocalTensor<float> outLocal = outQueue_.template AllocTensor<float>();
        LocalTensor<float> yScaleOut = outLocal[H];
        LocalTensor<int8_t> yOut = outLocal.template ReinterpretCast<int8_t>();

        // Padding is zeroed above, so reducing over the full aligned width
        // is equivalent to reducing valid elements only
        ComputeReduceMax(absBuf, static_cast<int32_t>(alignColNum_));
        PipeBarrier<PIPE_V>();

        Muls(yScaleOut, absBuf, DYNAMIC_QUANT_FACTOR, 1);
        PipeBarrier<PIPE_V>();

        event_t eventV2S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventV2S);
        WaitFlag<HardEvent::V_S>(eventV2S);
        float scaleVal = yScaleOut.GetValue(0);
        if (scaleVal == 0.0f) {
            scaleVal = 1.0f;
        }
        float invScale = 1.0f / scaleVal;
        Muls(situOut, situOut, invScale, H);
        PipeBarrier<PIPE_V>();

        CastFloatToInt8(situOut, yOut, H);
        outQueue_.EnQue<float>(outLocal);
    }

    __aicore__ inline void CastFloatToInt8(const LocalTensor<float>& src, LocalTensor<int8_t>& dst, int64_t count)
    {
        // FP32 -> INT32 (rint)
        LocalTensor<int32_t> tmpI32 = castBuf_.template Get<int32_t>();
        Cast(tmpI32, src, RoundMode::CAST_RINT, count);
        PipeBarrier<PIPE_V>();
        SetDeqScale((half)1.000000e+00f);

        // INT32 -> FP16 (round)
        LocalTensor<float> tmpF32 = castBuf_.template Get<float>();
        LocalTensor<half> tmpF16 = tmpF32.ReinterpretCast<half>();
        Cast(tmpF16, tmpI32, RoundMode::CAST_ROUND, count);
        PipeBarrier<PIPE_V>();

        // FP16 -> INT8 (trunc)
        Cast(dst, tmpF16, RoundMode::CAST_TRUNC, count);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeReduceMax(const LocalTensor<float>& tempRes, int32_t calCount)
    {
        uint32_t repsFp32 = static_cast<uint32_t>(calCount >> 6);
        uint32_t offsetsFp32 = repsFp32 << 6;
        uint32_t remsFp32 = static_cast<uint32_t>(calCount & 0x3f);

        if (likely(repsFp32 > 1)) {
            if (repsFp32 - 1 > MAX_REPEAT) {
                Max(tempRes, tempRes[ELEM_PER_REP_FP32], tempRes, ELEM_PER_REP_FP32, MAX_REPEAT, {1, 1, 1, 0, 8, 0});
                PipeBarrier<PIPE_V>();
                Max(tempRes, tempRes[ELEM_PER_REP_FP32 * MAX_REPEAT], tempRes, ELEM_PER_REP_FP32,
                    repsFp32 - MAX_REPEAT - 1, {1, 1, 1, 0, 8, 0});
            } else {
                Max(tempRes, tempRes[ELEM_PER_REP_FP32], tempRes, ELEM_PER_REP_FP32, repsFp32 - 1, {1, 1, 1, 0, 8, 0});
            }
            PipeBarrier<PIPE_V>();
        }
        if (unlikely(remsFp32 > 0) && unlikely(offsetsFp32 > 0)) {
            Max(tempRes, tempRes[offsetsFp32], tempRes, remsFp32, 1, {1, 1, 1, 0, 8, 0});
            PipeBarrier<PIPE_V>();
        }
        uint32_t mask = repsFp32 > 0 ? ELEM_PER_REP_FP32 : calCount;
        WholeReduceMax(tempRes, tempRes, mask, 1, 8, 1, 8);
    }

    __aicore__ inline void CopyOut(int64_t rowIdx, int64_t colOffset)
    {
        LocalTensor<float> outLocal = outQueue_.template DeQue<float>();
        LocalTensor<int8_t> yOut = outLocal.template ReinterpretCast<int8_t>();

        DataCopyExtParams dataCopyYParams{1, static_cast<uint32_t>(curColNum_ * sizeof(int8_t)), 0, 0, 0};
        DataCopyPad(yGm_[rowIdx * outDimy_ + colOffset], yOut, dataCopyYParams);

        if (quantType_ == 1) {
            LocalTensor<float> yScaleOut = outLocal[alignColNum_];
            DataCopyExtParams dataCopyScaleParams{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
            DataCopyPad(yScaleGm_[rowIdx], yScaleOut, dataCopyScaleParams);
        }

        outQueue_.FreeTensor(outLocal);
    }

    __aicore__ inline void processCompute()
    {
        int64_t lastColNum = baseColLen_;
        int64_t colLoops = 1;
        if (baseColLen_ < colLen_) {
            colLoops = (colLen_ + baseColLen_ - 1) / baseColLen_;
            lastColNum = colLen_ - (colLoops - 1) * baseColLen_;
        }

        if (quantType_ == 1 && colLoops > 1) {
            // Dynamic mode with column tiling: two-pass (recompute) approach
            // Pass 1: compute per-row absmax across all column tiles
            // Pass 2: re-compute Situ, quantize with global scale, output
            for (int64_t i = 0; i < curCoreRowNum_; i++) {
                float scaleVal = DynamicComputeRowScale(i, colLoops, lastColNum);
                // Sync MTE3→V between Pass 1 and Pass 2
                event_t eventMTE3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
                SetFlag<HardEvent::MTE3_V>(eventMTE3ToV);
                WaitFlag<HardEvent::MTE3_V>(eventMTE3ToV);
                DynamicQuantizeAndOutput(i, scaleVal, colLoops, lastColNum);
            }
        } else {
            // Single-pass approach (no column tiling or static mode)
            // Row-major order: process all tiles for each row before moving to next row
            for (int64_t i = 0; i < curCoreRowNum_; i++) {
                for (int64_t colLoop = 0; colLoop < colLoops; colLoop++) {
                    curColNum_ = (colLoop == colLoops - 1) ? lastColNum : baseColLen_;
                    curColNum_ = (curColNum_ == 0) ? baseColLen_ : curColNum_;
                    UpdateAlignColNum();

                    CopyInDequantParams(colLoop * baseColLen_);
                    if (quantScaleIsEmpty_ == 0 && (!quantIsOne_ || (quantOffsetIsEmpty_ == 0 && !quantOffsetIsOne_))) {
                        CopyInQuantParams(colLoop * baseColLen_);
                    }
                    CopyIn(i, colLoop * baseColLen_);
                    ComputeDequant(i);
                    ComputeSitu();
                    ComputeQuant();
                    CopyOut(i, colLoop * baseColLen_);
                }
            }
        }
    }

    __aicore__ inline void ApplySmoothScale(LocalTensor<float>& situOut)
    {
        if (quantScaleIsEmpty_ == 0) {
            if (quantIsOne_) {
                Muls(situOut, situOut, quantScaleVal_, alignColNum_);
            } else {
                Mul(situOut, situOut, quantLocal_, alignColNum_);
            }
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline float DynamicComputeRowScale(int64_t rowIdx, int64_t colLoops, int64_t lastColNum)
    {
        float rowAbsMax = 0.0f;
        for (int64_t colLoop = 0; colLoop < colLoops; colLoop++) {
            curColNum_ = (colLoop == colLoops - 1) ? lastColNum : baseColLen_;
            curColNum_ = (curColNum_ == 0) ? baseColLen_ : curColNum_;
            UpdateAlignColNum();

            CopyInDequantParams(colLoop * baseColLen_);
            if (quantScaleIsEmpty_ == 0 && (!quantIsOne_ || (quantOffsetIsEmpty_ == 0 && !quantOffsetIsOne_))) {
                CopyInQuantParams(colLoop * baseColLen_);
            }
            CopyIn(rowIdx, colLoop * baseColLen_);
            ComputeDequant(rowIdx);
            ComputeSitu();
            ApplySmoothScale(situOut_);

            // Compute absmax for this tile
            LocalTensor<float> absBuf = situTemp_;
            Abs(absBuf, situOut_, alignColNum_);
            PipeBarrier<PIPE_V>();
            // Zero out padding region [curColNum_, alignColNum_) via multiply-mask:
            // pad holds dequant(bias) values which would pollute the absmax
            if (alignColNum_ > curColNum_) {
                LocalTensor<float> maskTensor = castBuf_.template Get<float>();
                Duplicate<float>(maskTensor, 0.0f, alignColNum_);
                PipeBarrier<PIPE_V>();
                Adds(maskTensor, maskTensor, 1.0f, curColNum_);
                PipeBarrier<PIPE_V>();
                Mul(absBuf, absBuf, maskTensor, alignColNum_);
                PipeBarrier<PIPE_V>();
            }
            // Padding is zeroed above, so reducing over the full aligned width
            // is equivalent to reducing valid elements only
            ComputeReduceMax(absBuf, static_cast<int32_t>(alignColNum_));
            PipeBarrier<PIPE_V>();

            event_t eventV2S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventV2S);
            WaitFlag<HardEvent::V_S>(eventV2S);
            float tileMax = absBuf.GetValue(0);
            if (tileMax > rowAbsMax) {
                rowAbsMax = tileMax;
            }
        }

        float scaleVal = rowAbsMax * DYNAMIC_QUANT_FACTOR;
        // Return the original (unclamped) scale for output. The caller handles
        // the division-by-zero guard separately so the output y_scale matches
        // the CPU golden (0.0 when absmax == 0).
        return scaleVal;
    }

    __aicore__ inline void DynamicQuantizeAndOutput(int64_t rowIdx, float scaleVal, int64_t colLoops,
                                                    int64_t lastColNum)
    {
        // Guard against division by zero: use 1.0 for invScale computation
        // but preserve the original scaleVal (possibly 0.0) for output.
        float invScaleVal = (scaleVal == 0.0f) ? 1.0f : scaleVal;
        float invScale = 1.0f / invScaleVal;

        for (int64_t colLoop = 0; colLoop < colLoops; colLoop++) {
            curColNum_ = (colLoop == colLoops - 1) ? lastColNum : baseColLen_;
            curColNum_ = (curColNum_ == 0) ? baseColLen_ : curColNum_;
            UpdateAlignColNum();

            CopyInDequantParams(colLoop * baseColLen_);
            if (quantScaleIsEmpty_ == 0 && (!quantIsOne_ || (quantOffsetIsEmpty_ == 0 && !quantOffsetIsOne_))) {
                CopyInQuantParams(colLoop * baseColLen_);
            }
            CopyIn(rowIdx, colLoop * baseColLen_);
            ComputeDequant(rowIdx);
            ComputeSitu();
            ApplySmoothScale(situOut_);

            // Quantize with global scale
            Muls(situOut_, situOut_, invScale, alignColNum_);
            PipeBarrier<PIPE_V>();

            // Cast to int8 and output
            LocalTensor<float> outLocal = outQueue_.template AllocTensor<float>();
            LocalTensor<int8_t> yOut = outLocal.template ReinterpretCast<int8_t>();
            CastFloatToInt8(situOut_, yOut, alignColNum_);

            // Write scale for first tile only
            if (colLoop == 0) {
                LocalTensor<float> yScaleOut = outLocal[alignColNum_];
                Duplicate<float>(yScaleOut, scaleVal, 1);
                PipeBarrier<PIPE_V>();
            }

            outQueue_.EnQue<float>(outLocal);

            // CopyOut y (always) and scale (first tile only)
            LocalTensor<float> outLocalDeq = outQueue_.template DeQue<float>();
            LocalTensor<int8_t> yOutDeq = outLocalDeq.template ReinterpretCast<int8_t>();
            DataCopyExtParams dataCopyYParams{1, static_cast<uint32_t>(curColNum_ * sizeof(int8_t)), 0, 0, 0};
            DataCopyPad(yGm_[rowIdx * outDimy_ + colLoop * baseColLen_], yOutDeq, dataCopyYParams);

            if (colLoop == 0) {
                LocalTensor<float> yScaleOutDeq = outLocalDeq[alignColNum_];
                DataCopyExtParams dataCopyScaleParams{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
                DataCopyPad(yScaleGm_[rowIdx], yScaleOutDeq, dataCopyScaleParams);
            }

            outQueue_.FreeTensor(outLocalDeq);

            // Sync MTE3→MTE2 between tiles: ensure copy-out completes before
            // next tile's CopyInDequantParams overwrites TBuf buffers
            if (colLoop + 1 < colLoops) {
                event_t eventMTE3ToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
                SetFlag<HardEvent::MTE3_MTE2>(eventMTE3ToMTE2);
                WaitFlag<HardEvent::MTE3_MTE2>(eventMTE3ToMTE2);
            }
        }
    }

    __aicore__ inline void UpdateAlignColNum()
    {
        // INT8 path: x is int8, cast chain goes int8→half→float.
        // All Vector counts/offsets must be 32B-aligned for int8 (32 elements),
        // which also satisfies half (16) and float (8) alignment.
        alignColNum_ = Align(curColNum_, sizeof(int8_t));
        alignTileLen_ = alignColNum_ * SWI_FACTOR;
    }

    // For float-type DataCopyPad: rightPadding (in bytes) must not exceed 32B.
    // This means at most 8 float elements. When alignColNum_ (based on int8)
    // exceeds the float 32B boundary, we split padding into:
    //   1. DataCopyPad rightPadding to the next float 32B boundary
    //   2. Duplicate<float> to zero-fill the rest up to alignColNum_
    __aicore__ inline uint8_t FloatRightPadding()
    {
        int64_t alignFloatElems = Align(curColNum_, sizeof(float));
        return static_cast<uint8_t>(alignFloatElems - curColNum_);
    }

    __aicore__ inline void PadFloatTail(LocalTensor<float>& buf, int64_t baseOffset)
    {
        int64_t alignFloatElems = Align(curColNum_, sizeof(float));
        int64_t remainZeros = alignColNum_ - alignFloatElems;
        if (remainZeros > 0) {
            Duplicate<float>(buf[baseOffset + alignFloatElems], 0.0f, remainZeros);
            PipeBarrier<PIPE_V>();
        }
    }

    __aicore__ inline int64_t Align(int64_t elementNum, int64_t bytes)
    {
        constexpr int64_t BLOCK_BYTES = 32;
        if (bytes == 0) {
            return 0;
        }
        return (elementNum * bytes + BLOCK_BYTES - 1) / BLOCK_BYTES * BLOCK_BYTES / bytes;
    }

protected:
    TPipe* pipe_ = nullptr;
    const DequantSituQuantTilingData* tl_ = nullptr;

    int64_t blockIdx_ = 0;
    int64_t rowLen_ = 0;
    int64_t colLen_ = 0;
    int64_t inDimy_ = 0;
    int64_t outDimy_ = 0;
    int64_t baseRowLen_ = 0;
    int64_t baseColLen_ = 0;
    int64_t curColNum_ = 0;
    int64_t alignColNum_ = 0;
    int64_t alignTileLen_ = 0;
    int64_t usedCoreNum_ = 0;
    int64_t curCoreRowNum_ = 0;
    int64_t inputCopyOffset_ = 0;
    int64_t activateLeft_ = 0;
    int64_t quantType_ = 0;
    bool quantIsOne_ = false;
    int64_t quantScaleIsEmpty_ = 1;
    int64_t quantOffsetIsEmpty_ = 1;
    bool quantOffsetIsOne_ = false;
    float beta_ = 1.0f;
    float linearBeta_ = 0.0f;
    float quantScaleVal_ = 1.0f;
    float quantOffsetVal_ = 0.0f;
    bool dequantScaleIsOne_ = false;
    float dequantScaleVal_ = 1.0f;
    bool dequantBiasIsOne_ = false;
    float dequantBiasVal_ = 0.0f;

    GlobalTensor<int8_t> xGm_;
    GlobalTensor<float> dequantScaleGm_;
    GlobalTensor<float> dequantBiasGm_;
    GlobalTensor<float> quantScaleGm_;
    GlobalTensor<float> quantOffsetGm_;
    GlobalTensor<int8_t> yGm_;
    GlobalTensor<float> yScaleGm_;

    TQue<QuePosition::VECIN, DB_BUFFER> inQueueX_;
    TBuf<TPosition::VECCALC> dequantScaleBuf_;
    TBuf<TPosition::VECCALC> dequantBiasBuf_;
    TBuf<TPosition::VECCALC> quantBuf_;
    TQue<QuePosition::VECOUT, 1> outQueue_;
    TBuf<TPosition::VECCALC> tmpBuf_;
    TBuf<TPosition::VECCALC> castBuf_;

    // Intermediate results passed between compute stages
    LocalTensor<float> dequantScaleLocal_;
    LocalTensor<float> dequantBiasLocal_;
    LocalTensor<float> quantLocal_;
    LocalTensor<float> dequantOut_;
    LocalTensor<float> situTemp_;
    LocalTensor<float> situOut_;
};

// ---------------------------------------------------------------------------
// K3 Kernel: INT32/BF16 path with MoE routing and per-row dynamic quant
// ---------------------------------------------------------------------------

constexpr int64_t K3_MASK_FP32 = 256 / sizeof(float);
constexpr int64_t K3_MASK_BLK_STRIDE = 8;
constexpr float K3_DYNAMIC_QUANT_FACTOR = 1.0f / 127.0f;
constexpr int32_t K3_BUFFER_NUM = 2;
// Vector instructions require 32B-aligned UB addresses. The int8 y data region
// in outQueue_ is aligned up to a whole 32B block, and the per-row y_scale
// (one float) is packed at the next 32B boundary. Its index in float elements
// is computed in Init as scaleIdx_ = (32B-aligned y data bytes) / sizeof(float),
// so DynamicQuant (writer) and CopyOutRow (reader) share one offset.
constexpr int64_t K3_BLOCK_BYTES = 32;

template <typename XType>
class DequantSituQuantK3Kernel {
public:
    __aicore__ inline explicit DequantSituQuantK3Kernel(TPipe* pipe) : pipe_(pipe) {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weightScale, GM_ADDR activationScale, GM_ADDR bias,
                                GM_ADDR groupIndex, GM_ADDR y, GM_ADDR yScale,
                                const DequantSituQuantTilingData* tilingData)
    {
        tilingData_ = tilingData;
        blockIdx_ = GetBlockIdx();
        rowLen_ = static_cast<int64_t>(tilingData_->rowLen);
        inputWidth_ = static_cast<int64_t>(tilingData_->inputWidth);
        outputWidth_ = static_cast<int64_t>(tilingData_->outputWidth);
        expertNum_ = static_cast<int64_t>(tilingData_->expertNum);
        usedCoreNum_ = static_cast<int64_t>(tilingData_->usedCoreNum);
        hasBias_ = tilingData_->dequantBiasIsEmpty == 0;
        hasGroupIndex_ = tilingData_->hasGroupIndex != 0;
        activateLeft_ = tilingData_->activateLeft;
        beta_ = tilingData_->beta;
        linearBeta_ = tilingData_->linearBeta;

        // Align to 32B for Vector instructions based on actual dtype size.
        // Use the smaller element size to guarantee 32B alignment for both
        // XType and FP32 (the cast target).
        constexpr int64_t BLOCK_BYTES = 32;
        constexpr int64_t xTypeSize = sizeof(XType);
        constexpr int64_t fp32Size = sizeof(float);
        constexpr int64_t minElemSize = (xTypeSize < fp32Size) ? xTypeSize : fp32Size;
        constexpr int64_t alignUnit = BLOCK_BYTES / minElemSize;
        alignOutputWidth_ = (outputWidth_ + alignUnit - 1) / alignUnit * alignUnit;
        alignInputWidth_ = (inputWidth_ + alignUnit - 1) / alignUnit * alignUnit;
        // ComputeRow splits the FP32 buffer into gate[0:alignOutputWidth_] and
        // up[alignOutputWidth_:2*alignOutputWidth_], so the buffer must hold at
        // least 2*alignOutputWidth_ elements. When H is tiny, alignment rounding
        // amplifies alignOutputWidth_ (e.g. BF16 H=1 -> alignOutputWidth_=16) so
        // that 2*alignOutputWidth_=32 exceeds alignInputWidth_=align(2H)=16.
        // bufferWidth_ is the actual allocation/operation width for FP32-side
        // buffers (dequantBuf_, xQueue_ INT32 in-place, tmpBuf_, weight/bias
        // queues): it covers both gate and up halves. alignInputWidth_ stays
        // unchanged because CopyInRow's DataCopyPad rightPadding (in XType
        // elements, bytes <= 32B) is bounded by alignInputWidth_ - inputWidth_.
        constexpr int64_t NUM_TWO = 2;
        bufferWidth_ = (alignInputWidth_ >= NUM_TWO * alignOutputWidth_) ? alignInputWidth_ :
                                                                           (NUM_TWO * alignOutputWidth_);

        xGm_.SetGlobalBuffer((__gm__ XType*)x, rowLen_ * inputWidth_);
        if constexpr (std::is_same_v<XType, int32_t>) {
            weightScaleGm_.SetGlobalBuffer((__gm__ float*)weightScale, expertNum_ * inputWidth_);
            activationScaleGm_.SetGlobalBuffer((__gm__ float*)activationScale, rowLen_);
            if (hasBias_) {
                biasGm_.SetGlobalBuffer((__gm__ float*)bias, expertNum_ * inputWidth_);
            }
            if (hasGroupIndex_) {
                groupIndexGm_.SetGlobalBuffer((__gm__ int64_t*)groupIndex, expertNum_);
            }
        }
        yGm_.SetGlobalBuffer((__gm__ int8_t*)y, rowLen_ * outputWidth_);
        yScaleGm_.SetGlobalBuffer((__gm__ float*)yScale, rowLen_);

        // Buffer allocation uses aligned sizes to ensure 32B-aligned Vector access.
        // FP32-side buffers (dequantBuf_, INT32 xQueue_ in-place, weight/bias
        // queues, tmpBuf_) are sized by bufferWidth_ (>= 2*alignOutputWidth_) so
        // that ComputeRow's up half at xLocalF32[alignOutputWidth_] is in-bounds.
        // xQueue_ for BF16/FP16 still only needs alignInputWidth_ XType elements
        // (CopyInRow pads to that width), but the cast target dequantBuf_ uses
        // bufferWidth_ floats.
        const int64_t bufferBytes = bufferWidth_ * static_cast<int64_t>(sizeof(float));
        // outQueue_ packs [int8 y data (32B-aligned)] [y_scale (1 float at 32B boundary)].
        // alignOutputWidth_ is aligned to XType/float's 32B boundary (e.g. 16 for
        // BF16, 8 for INT32), but the int8 y region uses alignOutputWidth_ * 1 byte,
        // which is NOT necessarily a 32B multiple (BF16 x=[1,2] -> 16 bytes). The
        // scale is written by Duplicate<float>, whose UB address must be 32B-aligned,
        // so the int8 region is explicitly aligned up to a whole 32B block and the
        // scale sits at the next 32B boundary (scaleIdx_ computed below).
        const int64_t yDataBytes = (alignOutputWidth_ * static_cast<int64_t>(sizeof(int8_t)) + K3_BLOCK_BYTES - 1) /
                                   K3_BLOCK_BYTES * K3_BLOCK_BYTES;
        const int64_t outputBytes = yDataBytes + K3_BLOCK_BYTES;
        // scaleIdx_ = yDataBytes / sizeof(float): yDataBytes is a 32B multiple, so
        // this is a whole number of floats and the byte offset yDataBytes is
        // 32B-aligned — satisfying Duplicate<float>'s alignment requirement.
        scaleIdx_ = yDataBytes / static_cast<int64_t>(sizeof(float));
        if constexpr (std::is_same_v<XType, int32_t>) {
            // INT32 path reuses xQueue_ in-place as FP32 (same 4-byte width), so the
            // queue must hold bufferWidth_ elements, not just alignInputWidth_.
            pipe_->InitBuffer(xQueue_, K3_BUFFER_NUM, bufferBytes);
            pipe_->InitBuffer(weightScaleQueue_, 1, bufferBytes);
            if (hasBias_) {
                pipe_->InitBuffer(biasQueue_, 1, bufferBytes);
            }
        } else {
            // BF16/FP16: xQueue_ now holds two separately-loaded halves
            // [first(alignOutputWidth_) | second(alignOutputWidth_)] = bufferWidth_
            // XType elements; dequantBuf_ holds the FP32 cast result (same count).
            pipe_->InitBuffer(xQueue_, K3_BUFFER_NUM, bufferWidth_ * static_cast<int64_t>(sizeof(XType)));
            pipe_->InitBuffer(dequantBuf_, bufferBytes);
        }
        pipe_->InitBuffer(outQueue_, K3_BUFFER_NUM, outputBytes);
        // tmpBuf_ holds: sigmoid[0:alignOutputWidth_] + sigmoidTmp workspace
        // [alignOutputWidth_:2*alignOutputWidth_] (ComputeRow), then later
        // Abs[0:alignOutputWidth_] + maskTensor[bufferWidth_:] + tempInt32
        // [alignOutputWidth_:] (DynamicQuant/CastFloatToInt8). The mask region
        // starts at bufferWidth_ (>= alignInputWidth_), so size for
        // bufferWidth_ + alignOutputWidth_ floats to cover mask + tempInt32.
        pipe_->InitBuffer(tmpBuf_, (bufferWidth_ + alignOutputWidth_) * static_cast<int64_t>(sizeof(float)));
    }

    __aicore__ inline void Process()
    {
        if (usedCoreNum_ <= 0 || blockIdx_ >= usedCoreNum_) {
            return;
        }

        if constexpr (!std::is_same_v<XType, int32_t>) {
            ProcessGroup(0, rowLen_, 0);
            return;
        }

        if (!hasGroupIndex_) {
            ProcessGroup(0, rowLen_, 0);
            return;
        }

        int64_t groupOffset = 0;
        for (int64_t expertIdx = 0; expertIdx < expertNum_ && groupOffset < rowLen_; ++expertIdx) {
            const int64_t requestedRows = groupIndexGm_.GetValue(expertIdx);
            const int64_t remainingRows = rowLen_ - groupOffset;
            const int64_t groupRows = requestedRows <= 0 ?
                                          0 :
                                          (requestedRows > remainingRows ? remainingRows : requestedRows);
            if (groupRows > 0) {
                ProcessGroup(expertIdx, groupRows, groupOffset);
                groupOffset += groupRows;
            }
        }
    }

private:
    __aicore__ inline void ProcessGroup(int64_t expertIdx, int64_t groupRows, int64_t groupOffset)
    {
        const int64_t rowsPerCore = (groupRows + usedCoreNum_ - 1) / usedCoreNum_;
        const int64_t localGroupOffset = blockIdx_ * rowsPerCore;
        if (localGroupOffset >= groupRows) {
            return;
        }
        const int64_t localRows = groupRows - localGroupOffset < rowsPerCore ? groupRows - localGroupOffset :
                                                                               rowsPerCore;
        const int64_t firstRow = groupOffset + localGroupOffset;

        if constexpr (std::is_same_v<XType, int32_t>) {
            CopyInExpertParams(expertIdx);
            weightScaleLocal_ = weightScaleQueue_.DeQue<float>();
            if (hasBias_) {
                biasLocal_ = biasQueue_.DeQue<float>();
            }
        }

        for (int64_t localRow = 0; localRow < localRows; ++localRow) {
            const int64_t rowIdx = firstRow + localRow;
            CopyInRow(rowIdx);
            ComputeRow(rowIdx);
            CopyOutRow(rowIdx);
        }

        if constexpr (std::is_same_v<XType, int32_t>) {
            weightScaleQueue_.FreeTensor(weightScaleLocal_);
            if (hasBias_) {
                biasQueue_.FreeTensor(biasLocal_);
            }
        }
    }

    __aicore__ inline void CopyInExpertParams(int64_t expertIdx)
    {
        // Load weight_scale and bias in two halves to match xLocalF32's
        // [first_half(alignOutputWidth_) | second_half(alignOutputWidth_)] layout
        // produced by CopyInRow. GM stores them contiguously as [first(H)|second(H)].
        // Float DataCopyPad rightPadding (in float elems) bytes must not exceed 32B:
        // max padding = alignOutputWidth_ - outputWidth_ <= alignUnit-1, and
        // (alignUnit-1)*4 <= 28 < 32B, so a single DataCopyPad per half suffices.
        const int64_t paramOffset = expertIdx * inputWidth_;
        uint8_t rPad = static_cast<uint8_t>(alignOutputWidth_ - outputWidth_);
        const uint32_t halfBytes = static_cast<uint32_t>(outputWidth_ * sizeof(float));
        DataCopyExtParams params{1, halfBytes, 0, 0, 0};
        DataCopyPadExtParams<float> padParams{true, 0, rPad, 0.0f};

        LocalTensor<float> weightScaleLocal = weightScaleQueue_.AllocTensor<float>();
        // First half: weight_scale[0:H] -> [0:alignOutputWidth_]
        DataCopyPad(weightScaleLocal, weightScaleGm_[paramOffset], params, padParams);
        // Second half: weight_scale[H:2H] -> [alignOutputWidth_:2*alignOutputWidth_]
        DataCopyPad(weightScaleLocal[alignOutputWidth_], weightScaleGm_[paramOffset + outputWidth_], params, padParams);
        weightScaleQueue_.EnQue(weightScaleLocal);

        if (hasBias_) {
            LocalTensor<float> biasLocal = biasQueue_.AllocTensor<float>();
            DataCopyPad(biasLocal, biasGm_[paramOffset], params, padParams);
            DataCopyPad(biasLocal[alignOutputWidth_], biasGm_[paramOffset + outputWidth_], params, padParams);
            biasQueue_.EnQue(biasLocal);
        }
    }

    __aicore__ inline void CopyInRow(int64_t rowIdx)
    {
        // Load gate and up halves SEPARATELY into aligned buffer regions, matching
        // the INT8 path. The input row in GM is [first_half(H) | second_half(H)];
        // loading it contiguously and splitting at alignOutputWidth_ is WRONG when
        // H < alignOutputWidth_ (e.g. BF16 H=1: data at indices [0,1] but split at
        // 16 puts up value -0.3 into the gate region and leaves up all-zeros).
        // Buffer layout: [first_half(alignOutputWidth_) | second_half(alignOutputWidth_)].
        // activateLeft_ only swaps which half is called gate vs up in ComputeRow,
        // not the GM/buffer layout.
        LocalTensor<XType> xLocal = xQueue_.AllocTensor<XType>();
        uint8_t padLen = static_cast<uint8_t>(alignOutputWidth_ - outputWidth_);
        const uint32_t halfBytes = static_cast<uint32_t>(outputWidth_ * sizeof(XType));
        DataCopyExtParams params{1, halfBytes, 0, 0, 0};
        DataCopyPadExtParams<XType> padParams{true, 0, padLen, static_cast<XType>(0)};

        // First half (x[:, 0:H]) -> xLocal[0 : alignOutputWidth_]
        DataCopyPad(xLocal, xGm_[rowIdx * inputWidth_], params, padParams);
        // Second half (x[:, H:2H]) -> xLocal[alignOutputWidth_ : 2*alignOutputWidth_]
        DataCopyPad(xLocal[alignOutputWidth_], xGm_[rowIdx * inputWidth_ + outputWidth_], params, padParams);
        xQueue_.EnQue(xLocal);
    }

    __aicore__ inline void ComputeRow(int64_t rowIdx)
    {
        LocalTensor<XType> xLocal = xQueue_.DeQue<XType>();
        LocalTensor<float> xLocalF32;
        if constexpr (std::is_same_v<XType, int32_t>) {
            xLocalF32 = xLocal.template ReinterpretCast<float>();
        } else {
            xLocalF32 = dequantBuf_.Get<float>();
        }
        // Cast the full bufferWidth_ range: CopyInRow already loaded both halves
        // (gate + up) each padded to alignOutputWidth_ with zeros, so the entire
        // bufferWidth_ = 2*alignOutputWidth_ range is valid data + clean padding.
        Cast(xLocalF32, xLocal, RoundMode::CAST_NONE, bufferWidth_);
        PipeBarrier<PIPE_V>();
        // BF16 path: xLocal is no longer needed after cast — release early
        // so the double-buffered xQueue can start loading the next row.
        // INT32 path: xLocal is reused in-place as FP32, must keep until end.
        if constexpr (!std::is_same_v<XType, int32_t>) {
            xQueue_.FreeTensor(xLocal);
        }
        if constexpr (std::is_same_v<XType, int32_t>) {
            const float activationScale = activationScaleGm_.GetValue(rowIdx);
            Mul(xLocalF32, xLocalF32, weightScaleLocal_, bufferWidth_);
            PipeBarrier<PIPE_V>();
            Muls(xLocalF32, xLocalF32, activationScale, bufferWidth_);
            PipeBarrier<PIPE_V>();
            if (hasBias_) {
                Add(xLocalF32, xLocalF32, biasLocal_, bufferWidth_);
                PipeBarrier<PIPE_V>();
            }
        }

        LocalTensor<float> temp = tmpBuf_.Get<float>();
        int64_t gateOffset = (activateLeft_ == 1) ? 0 : alignOutputWidth_;
        int64_t upOffset = (activateLeft_ == 1) ? alignOutputWidth_ : 0;

        LocalTensor<float> gate = xLocalF32[gateOffset];
        LocalTensor<float> up = xLocalF32[upOffset];
        LocalTensor<float> sigmoid = temp;
        LocalTensor<uint8_t> sigmoidTmp = temp[alignOutputWidth_].template ReinterpretCast<uint8_t>();

        // Sigmoid(gate) — result stored in sigmoid buffer, gate unchanged
        Sigmoid(sigmoid, gate, sigmoidTmp, alignOutputWidth_);
        PipeBarrier<PIPE_V>();

        // gate tanh: beta * tanh(gate / beta)
        Muls(gate, gate, 1.0f / beta_, alignOutputWidth_);
        PipeBarrier<PIPE_V>();
        Tanh(gate, gate, alignOutputWidth_);
        PipeBarrier<PIPE_V>();
        Muls(gate, gate, beta_, alignOutputWidth_);
        PipeBarrier<PIPE_V>();

        // up tanh (if linear_beta > 0): linear_beta * tanh(up / linear_beta)
        if (linearBeta_ > 0.0f) {
            Muls(up, up, 1.0f / linearBeta_, alignOutputWidth_);
            PipeBarrier<PIPE_V>();
            Tanh(up, up, alignOutputWidth_);
            PipeBarrier<PIPE_V>();
            Muls(up, up, linearBeta_, alignOutputWidth_);
            PipeBarrier<PIPE_V>();
        }

        // situ_a = (beta * tanh(gate/beta) * sigmoid(gate)) * up
        Mul(gate, gate, sigmoid, alignOutputWidth_);
        PipeBarrier<PIPE_V>();
        Mul(gate, gate, up, alignOutputWidth_);
        PipeBarrier<PIPE_V>();

        DynamicQuant(gate, temp);
        if constexpr (std::is_same_v<XType, int32_t>) {
            xQueue_.FreeTensor(xLocal);
        }
    }

    __aicore__ inline void DynamicQuant(LocalTensor<float>& situ, LocalTensor<float>& temp)
    {
        Abs(temp, situ, alignOutputWidth_);
        PipeBarrier<PIPE_V>();
        // Zero out padding region [outputWidth_, alignOutputWidth_) via multiply-mask:
        // pad holds dequant(bias) values which would pollute the absmax.
        // Mask tensor lives at temp[bufferWidth_:], tmpBuf_ sized for it in Init.
        if (alignOutputWidth_ > outputWidth_) {
            LocalTensor<float> maskTensor = temp[bufferWidth_];
            Duplicate<float>(maskTensor, 0.0f, alignOutputWidth_);
            PipeBarrier<PIPE_V>();
            Adds(maskTensor, maskTensor, 1.0f, outputWidth_);
            PipeBarrier<PIPE_V>();
            Mul(temp, temp, maskTensor, alignOutputWidth_);
            PipeBarrier<PIPE_V>();
        }
        // Padding is zeroed above, so reducing over the full aligned width
        // is equivalent to reducing valid elements only
        ComputeReduceMax(temp);
        PipeBarrier<PIPE_V>();

        event_t eventVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventVToS);
        WaitFlag<HardEvent::V_S>(eventVToS);
        float scaleValue = temp.GetValue(0) * K3_DYNAMIC_QUANT_FACTOR;
        // Save the original output scale before clamping. When absmax == 0,
        // the true y_scale is 0.0 (matching the CPU golden), but we need
        // invScale = 1.0 to avoid division by zero during quantization.
        // Writing the clamped 1.0 to the output would mismatch the CPU.
        float outputScale = scaleValue;
        if (scaleValue <= 0.0f) {
            scaleValue = 1.0f;
        }
        Muls(situ, situ, 1.0f / scaleValue, alignOutputWidth_);
        PipeBarrier<PIPE_V>();

        LocalTensor<float> outLocal = outQueue_.AllocTensor<float>();
        LocalTensor<int8_t> yLocal = outLocal.ReinterpretCast<int8_t>();
        // Scale is packed after the 32B-aligned int8 y data region; scaleIdx_
        // (computed in Init) points to the next 32B boundary in float elements.
        LocalTensor<float> yScaleLocal = outLocal[scaleIdx_];
        Duplicate<float>(yScaleLocal, outputScale, 1);
        PipeBarrier<PIPE_V>();
        CastFloatToInt8(situ, temp, yLocal);
        outQueue_.EnQue<float>(outLocal);
    }

    __aicore__ inline void ComputeReduceMax(const LocalTensor<float>& temp)
    {
        // Padding is zeroed by the caller's multiply-mask, so reduce over the
        // full aligned width
        const uint32_t vectorCycles = static_cast<uint32_t>(alignOutputWidth_ / K3_MASK_FP32);
        const uint32_t remainder = static_cast<uint32_t>(alignOutputWidth_ % K3_MASK_FP32);

        if (vectorCycles > 1) {
            BinaryRepeatParams repeatParams;
            repeatParams.dstBlkStride = 1;
            repeatParams.src0BlkStride = 1;
            repeatParams.src1BlkStride = 1;
            repeatParams.dstRepStride = 0;
            repeatParams.src0RepStride = K3_MASK_BLK_STRIDE;
            repeatParams.src1RepStride = 0;
            Max(temp, temp[K3_MASK_FP32], temp, K3_MASK_FP32, static_cast<uint8_t>(vectorCycles - 1), repeatParams);
            PipeBarrier<PIPE_V>();
        }
        if (remainder > 0 && vectorCycles > 0) {
            Max(temp, temp[vectorCycles * K3_MASK_FP32], temp, remainder, 1, {1, 1, 1, 0, 8, 0});
            PipeBarrier<PIPE_V>();
        }
        uint32_t mask = vectorCycles > 0 ? K3_MASK_FP32 : static_cast<uint32_t>(alignOutputWidth_);
        WholeReduceMax(temp, temp, mask, 1, K3_MASK_BLK_STRIDE, 1, K3_MASK_BLK_STRIDE, ReduceOrder::ORDER_ONLY_VALUE);
    }

    __aicore__ inline void CastFloatToInt8(const LocalTensor<float>& src, LocalTensor<float>& temp,
                                           LocalTensor<int8_t>& dst)
    {
        LocalTensor<int32_t> tempInt32 = temp[alignOutputWidth_].ReinterpretCast<int32_t>();
        Cast(tempInt32, src, RoundMode::CAST_RINT, alignOutputWidth_);
        PipeBarrier<PIPE_V>();
        SetDeqScale((half)1.0f);

        LocalTensor<half> tempHalf = temp.ReinterpretCast<half>();
        Cast(tempHalf, tempInt32, RoundMode::CAST_ROUND, alignOutputWidth_);
        PipeBarrier<PIPE_V>();
        Cast(dst, tempHalf, RoundMode::CAST_TRUNC, alignOutputWidth_);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CopyOutRow(int64_t rowIdx)
    {
        LocalTensor<float> outLocal = outQueue_.DeQue<float>();
        LocalTensor<int8_t> yLocal = outLocal.ReinterpretCast<int8_t>();
        // Must match DynamicQuant's writer offset (scaleIdx_, computed in Init).
        LocalTensor<float> yScaleLocal = outLocal[scaleIdx_];

        DataCopyExtParams yParams{1, static_cast<uint32_t>(outputWidth_ * sizeof(int8_t)), 0, 0, 0};
        DataCopyPad(yGm_[rowIdx * outputWidth_], yLocal, yParams);
        DataCopyExtParams scaleParams{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        DataCopyPad(yScaleGm_[rowIdx], yScaleLocal, scaleParams);
        outQueue_.FreeTensor(outLocal);
    }

    TPipe* pipe_ = nullptr;
    const DequantSituQuantTilingData* tilingData_ = nullptr;
    int64_t blockIdx_ = 0;
    int64_t rowLen_ = 0;
    int64_t inputWidth_ = 0;
    int64_t outputWidth_ = 0;
    int64_t alignInputWidth_ = 0;
    int64_t alignOutputWidth_ = 0;
    // FP32-side buffer width covering both gate and up halves (>= 2 *
    // alignOutputWidth_). Used for dequantBuf_/xQueue_(INT32)/tmpBuf_/weight-bias
    // queue sizing and the Cast/Mul/Add operation length, so that up lives at
    // xLocalF32[alignOutputWidth_] inside the allocated buffer.
    int64_t bufferWidth_ = 0;
    // Index (in float elements) of y_scale inside outQueue_, computed in Init so
    // that DynamicQuant (writer) and CopyOutRow (reader) share one formula. The
    // int8 y data region is aligned up to a 32B block, so the scale float lands
    // on the next 32B boundary — a 32B-aligned UB address required by Duplicate.
    int64_t scaleIdx_ = 0;
    int64_t expertNum_ = 0;
    int64_t usedCoreNum_ = 0;
    uint32_t activateLeft_ = 1;
    bool hasBias_ = false;
    bool hasGroupIndex_ = false;
    float beta_ = 4.0f;
    float linearBeta_ = 25.0f;

    GlobalTensor<XType> xGm_;
    GlobalTensor<float> weightScaleGm_;
    GlobalTensor<float> activationScaleGm_;
    GlobalTensor<float> biasGm_;
    GlobalTensor<int64_t> groupIndexGm_;
    GlobalTensor<int8_t> yGm_;
    GlobalTensor<float> yScaleGm_;

    TQue<QuePosition::VECIN, K3_BUFFER_NUM> xQueue_;
    TQue<QuePosition::VECIN, 1> weightScaleQueue_;
    TQue<QuePosition::VECIN, 1> biasQueue_;
    TQue<QuePosition::VECOUT, K3_BUFFER_NUM> outQueue_;
    TBuf<TPosition::VECCALC> tmpBuf_;
    TBuf<TPosition::VECCALC> dequantBuf_;
    LocalTensor<float> weightScaleLocal_;
    LocalTensor<float> biasLocal_;
};

} // namespace DequantSituQuantOps
#endif // DEQUANT_SITU_QUANT_H
