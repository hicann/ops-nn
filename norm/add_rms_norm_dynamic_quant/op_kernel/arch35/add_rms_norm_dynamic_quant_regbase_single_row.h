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
 * \file add_rms_norm_dynamic_quant_regbase_single_row.h
 * \brief
 */

#ifndef ADD_RMS_NORM_DYNAMIC_QUANT_REGBASE_SINGLE_ROW_KERNEL_H_
#define ADD_RMS_NORM_DYNAMIC_QUANT_REGBASE_SINGLE_ROW_KERNEL_H_

#include <cstdint>
#include "add_rms_norm_dynamic_quant_regbase_common.h"
#include "../../norm_common/reduce_common_regbase.h"
namespace AddRmsNormDynamicQuant {
constexpr uint32_t BUFFER_NUM = 1;
constexpr int32_t ROW_FACTOR = 128;
constexpr uint32_t ELEM_PER_REP_FP32 = 64;
constexpr int32_t HALf_INTERVAL = 2;

template <typename T_X, typename T_Y>
class KernelAddRmsNormDynamicQuantRegbaseSingleRow {
public:
    __aicore__ inline KernelAddRmsNormDynamicQuantRegbaseSingleRow(TPipe* pipe)
    {
        Ppipe = pipe;
    }

    __aicore__ inline void Init(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR smooth1, GM_ADDR smooth2, GM_ADDR beta, GM_ADDR y1, GM_ADDR y2,
        GM_ADDR x, GM_ADDR outScale1, GM_ADDR outScale2, const AddRmsNormDynamicQuantRegbaseTilingData* tiling)
    {
        this->InitBaseParams(tiling);
        this->InitInGlobalTensors(x1, x2, gamma, smooth1, smooth2, beta);
        this->InitOutGlobalTensors(y1, y2, x, outScale1, outScale2);
        Ppipe->InitBuffer(inRowsQue, BUFFER_NUM, 2 * this->numLastDimAligned * sizeof(T_X)); // 2 * D * 2
        Ppipe->InitBuffer(yQue, BUFFER_NUM, this->numLastDimAligned * sizeof(T_X));          // D * 2
        Ppipe->InitBuffer(xBufFp32, this->numLastDimAligned * sizeof(float)); // D * 4
        Ppipe->InitBuffer(yBufFp32, this->numLastDimAligned * sizeof(float)); // D * 4
        Ppipe->InitBuffer(smoothBuf, this->numLastDimAligned * sizeof(T_X));    // D * 2
        // 2 dynamic quant operator required 2 scale buffer.
        Ppipe->InitBuffer(scalesQue, BUFFER_NUM, 2 * ROW_FACTOR * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (this->smooth1Exist) {
            AscendC::LocalTensor<T_X> smooth1Local = smoothBuf.Get<T_X>();
            DataCopyPad(smooth1Local, this->smooth1Gm, {1, static_cast<uint16_t>(this->numLastDim * sizeof(T_X)), 0, 0}, {});
        }
        int32_t outLoopTail = this->rowWork % ROW_FACTOR;
        int32_t outLoopCount = this->rowWork / ROW_FACTOR;
        uint64_t gmOffset = 0;
        uint64_t gmOffsetReduce = 0;

        int64_t oriOverflowMode = GetOverflowMode<T_Y>();
        SetOverflowMode<T_Y>(0);

        for (int32_t loopIdx = 0; loopIdx < outLoopCount; ++loopIdx) {
            AscendC::LocalTensor<float> scalesLocalOut1 = scalesQue.template AllocTensor<float>();
            for (int32_t innerIdx = 0; innerIdx < ROW_FACTOR; ++innerIdx) {
                CopyInX1X2(gmOffset);
                AddSingleRow(gmOffset);
                CopyInGamma();
                ComputeRmsNorm();
                CopyInSmooth();
                ComputeDynamicQuant(innerIdx, scalesLocalOut1);
                CopyOut(gmOffset);
                gmOffset += this->numLastDim;
            }
            scalesQue.EnQue(scalesLocalOut1);
            CopyOutScale(gmOffsetReduce, ROW_FACTOR);
            gmOffsetReduce += ROW_FACTOR;
        }
        {
            AscendC::LocalTensor<float> scalesLocalOut1 = scalesQue.template AllocTensor<float>();
            for (int32_t innerIdx1 = 0; innerIdx1 < outLoopTail; ++innerIdx1) {
                CopyInX1X2(gmOffset);
                AddSingleRow(gmOffset);
                CopyInGamma();
                ComputeRmsNorm();
                CopyInSmooth();
                ComputeDynamicQuant(innerIdx1, scalesLocalOut1);
                CopyOut(gmOffset);
                gmOffset += this->numLastDim;
            }
            scalesQue.EnQue(scalesLocalOut1);
            CopyOutScale(gmOffsetReduce, outLoopTail);
        }

        SetOverflowMode<T_Y>(oriOverflowMode);
    }

private:
    __aicore__ inline void CopyOutScale(uint64_t gmOffset, int32_t copyInNums)
    {
        AscendC::LocalTensor<float> outScalesLocal = scalesQue.template DeQue<float>();
        AscendC::LocalTensor<float> outScales1Local = outScalesLocal[0];
        AscendC::LocalTensor<float> outScales2Local = outScalesLocal[ROW_FACTOR];
        AscendC::DataCopyPad(this->outScale1Gm[gmOffset], outScales1Local, {1, static_cast<uint16_t>(copyInNums * sizeof(float)), 0, 0});
        if (this->smooth2Exist) {
            AscendC::DataCopyPad(this->outScale2Gm[gmOffset], outScales2Local, {1, static_cast<uint16_t>(copyInNums * sizeof(float)), 0, 0});
        }
        scalesQue.FreeTensor(outScalesLocal);
    }

    __aicore__ inline void CopyOut(uint64_t gmOffset)
    {
        AscendC::LocalTensor<T_Y> res12 = yQue.template DeQue<T_Y>();
        auto res1 = res12[0];
        auto res2 = res12[this->numLastDimAligned];
        AscendC::DataCopyPad(this->y1Gm[gmOffset], res1, {1, static_cast<uint16_t>(this->numLastDim * sizeof(T_Y)), 0, 0});
        if (this->smooth2Exist) {
            AscendC::DataCopyPad(this->y2Gm[gmOffset], res2, {1, static_cast<uint16_t>(this->numLastDim * sizeof(T_Y)), 0, 0});
        }
        yQue.FreeTensor(res12);
    }

    __aicore__ inline void RoundFloat2Quant(AscendC::LocalTensor<T_Y>& dstTensor, AscendC::LocalTensor<float>& srcTensor, uint32_t count)
    {
        uint16_t repeatTimes = static_cast<uint16_t>(CeilDivision(count, V_LENGTH));
        __local_mem__ T_Y* dstAddr = (__ubuf__ T_Y*)dstTensor.GetPhyAddr();
        __local_mem__ float* srcAddr = (__ubuf__ float*)srcTensor.GetPhyAddr();

        __VEC_SCOPE__
        {
            RegTensor<float> srcReg;
            RegTensor<float> tmpReg;
            RegTensor<half> halfReg;
            RegTensor<T_Y> dstReg;
            MaskReg maskReg;

            for (uint16_t idx = 0; idx < repeatTimes; idx++) {
                maskReg = UpdateMask<float>(count);
                DataCopy<float>(srcReg, srcAddr + idx * V_LENGTH);

                if constexpr (IsSameType<T_Y, int8_t>::value) {
                    Truncate<float, RoundMode::CAST_RINT>(tmpReg, srcReg, maskReg);
                    Cast<half, float, castTraitFp322Fp16>(halfReg, tmpReg, maskReg);
                    Cast<T_Y, half, castTraitFp162Int8>(dstReg, halfReg, maskReg);
                } else if constexpr (IsSameType<T_Y, fp8_e4m3fn_t>::value || IsSameType<T_Y, fp8_e5m2_t>::value) {
                    Cast<T_Y, float, castTraitFp322Fp8>(dstReg, srcReg, maskReg);
                } else if constexpr (IsSameType<T_Y, hifloat8_t>::value) {
                    Cast<T_Y, float, castTraitFp322Hifp8>(dstReg, srcReg, maskReg);
                }

                DataCopy<T_Y, StoreDist::DIST_PACK4_B32>(dstAddr + idx * V_LENGTH, dstReg, maskReg);
            }
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ReduceMaxInplace(const AscendC::LocalTensor<float>& srcLocal1, int32_t count)
    {
        uint64_t repsFp32 = count >> 6;       // 6 is cound / ELEM_PER_REP_FP32
        uint64_t offsetsFp32 = repsFp32 << 6; // 6 is repsFp32 * ELEM_PER_REP_FP32
        uint64_t remsFp32 = count & 0x3f;     // 0x3f 63, count % ELEM_PER_REP_FP32

        if (likely(repsFp32 > 1)) {
            // 8 is rep stride
            AscendC::Max(srcLocal1, srcLocal1[ELEM_PER_REP_FP32], srcLocal1, ELEM_PER_REP_FP32, repsFp32 - 1, {1, 1, 1, 0, 8, 0});
            PipeBarrier<PIPE_V>();
        }
        if (unlikely(remsFp32 > 0)) {
            AscendC::Max(srcLocal1, srcLocal1[offsetsFp32], srcLocal1, remsFp32, 1, {1, 1, 1, 0, 8, 0});
            PipeBarrier<PIPE_V>();
        }
        uint32_t mask = (repsFp32 > 0) ? ELEM_PER_REP_FP32 : count;
        // 8 is rep stride
        AscendC::WholeReduceMax(srcLocal1, srcLocal1, mask, 1, 8, 1, 8);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ScaleTensor(
        AscendC::LocalTensor<float>& srcTensor, AscendC::LocalTensor<float>& tmpTensor1, AscendC::LocalTensor<float>& scaleTensor, int32_t idx)
    {
        AscendC::Abs(tmpTensor1, srcTensor, this->numLastDim); // tmpLocal <-- |y * smooth|
        PipeBarrier<PIPE_V>();
        ReduceMaxInplace(tmpTensor1, this->numLastDim);
        event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventVS);
        WaitFlag<HardEvent::V_S>(eventVS);
        float maxTemp = tmpTensor1.GetValue(0);
        float scaleTemp = ((maxTemp == 0) ? (1.0f) : (this->quantMax / maxTemp));
        scaleTensor.SetValue(idx, 1 / scaleTemp);
        event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventSV);
        WaitFlag<HardEvent::S_V>(eventSV);
        Muls(srcTensor, srcTensor, scaleTemp, this->numLastDim);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ComputeDynamicQuant(int32_t idx, LocalTensor<float>& scalesLocalOut)
    {
        LocalTensor<float> xLocalFp32 = xBufFp32.Get<float>();
        LocalTensor<float> yLocalFp32 = yBufFp32.Get<float>();
        AscendC::LocalTensor<T_Y> yLocal = yQue.template AllocTensor<T_Y>();
        auto y1Local = yLocal[0];
        auto y2Local = yLocal[this->numLastDimAligned];

        if (this->smooth2Exist) {
            AscendC::LocalTensor<T_X> smooth2Local = inRowsQue.template DeQue<T_X>();
            AscendC::LocalTensor<float> tmpTensor = smooth2Local.template ReinterpretCast<float>();
            AscendC::Cast(yLocalFp32, smooth2Local, RoundMode::CAST_NONE, this->numLastDim); // yLocalFp32 <-- smooth2
            PipeBarrier<PIPE_V>();
            AscendC::Mul(yLocalFp32, xLocalFp32, yLocalFp32, this->numLastDim); // yLocalFp32 <-- y * smooth2
            PipeBarrier<PIPE_V>();
            ScaleTensor(
                yLocalFp32, tmpTensor, scalesLocalOut,
                idx + ROW_FACTOR); // yLocalFp32 <-- yLocalFp32 / max(abs(yLocalFp32))
            PipeBarrier<PIPE_V>();
            inRowsQue.FreeTensor(smooth2Local);
            RoundFloat2Quant(y2Local, yLocalFp32, static_cast<uint32_t>(this->numLastDim));
        }
        if (this->smooth1Exist) {
            AscendC::LocalTensor<T_X> smooth1Local = smoothBuf.template Get<T_X>();
            AscendC::Cast(yLocalFp32, smooth1Local, RoundMode::CAST_NONE, this->numLastDim); // yLocalFp32 <-- smooth1
            PipeBarrier<PIPE_V>();
            AscendC::Mul(yLocalFp32, xLocalFp32, yLocalFp32, this->numLastDim); // yLocalFp32 <-- y * smooth1
            PipeBarrier<PIPE_V>();
        } else {
            Muls(yLocalFp32, xLocalFp32, (float)1.0, this->numLastDim); // yLocalFp32 <-- y * smooth1
            PipeBarrier<PIPE_V>();
        }
        ScaleTensor(
            yLocalFp32, xLocalFp32, scalesLocalOut, idx); // yLocalFp32 <-- yLocalFp32 / max(abs(yLocalFp32))
        PipeBarrier<PIPE_V>();
        RoundFloat2Quant(y1Local, yLocalFp32, static_cast<uint32_t>(this->numLastDim));
        PipeBarrier<PIPE_V>();
        yQue.EnQue(yLocal);
    }

    __aicore__ inline void CopyInSmooth()
    {
        if (this->smooth2Exist) {
            AscendC::LocalTensor<T_X> smoothCopyIn1 = inRowsQue.template AllocTensor<T_X>();
            AscendC::DataCopyPad(smoothCopyIn1[0], this->smooth2Gm, {1, static_cast<uint16_t>(this->numLastDim * sizeof(T_X)), 0, 0}, {});
            inRowsQue.EnQue(smoothCopyIn1);
        }
    }

    __aicore__ inline void CopyInBeta()
    {
        AscendC::LocalTensor<T_X> betaCopyIn = inRowsQue.template AllocTensor<T_X>();
        AscendC::DataCopyPad(betaCopyIn[0], this->betaGm, {1, static_cast<uint16_t>(this->numLastDim * sizeof(T_X)), 0, 0}, {});
        inRowsQue.EnQue(betaCopyIn);
    }

    __aicore__ inline float ReduceSumHalfInterval(const LocalTensor<float>& src_local1, int32_t count)
    {
        if (likely(count > ELEM_PER_REP_FP32)) {
            int32_t bodyCount = static_cast<int32_t>(this->powerSplit_);
            int32_t tailCount = count - bodyCount;
            if (tailCount > 0) {
                AscendC::Add(src_local1, src_local1, src_local1[bodyCount], tailCount);
                PipeBarrier<PIPE_V>();
            }
            while (bodyCount > ELEM_PER_REP_FP32) {
                bodyCount = bodyCount / HALf_INTERVAL;
                Add(src_local1, src_local1, src_local1[bodyCount], bodyCount);
                PipeBarrier<PIPE_V>();
            }

            AscendCUtils::SetMask<float>(ELEM_PER_REP_FP32);
        } else {
            AscendCUtils::SetMask<float>(count);
        }
        AscendC::WholeReduceSum<float, false>(src_local1, src_local1, MASK_PLACEHOLDER, 1, 1, 1, DEFAULT_REPEAT_STRIDE);
        event_t event_v_s = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(event_v_s);
        WaitFlag<HardEvent::V_S>(event_v_s);
        return src_local1.GetValue(0);
    }

    __aicore__ inline void ComputeRmsNorm()
    {
        AscendC::LocalTensor<float> xLocalFp32 = xBufFp32.Get<float>();
        AscendC::LocalTensor<float> yLocalFp32V1 = yBufFp32.Get<float>();

        AscendC::Mul(yLocalFp32V1, xLocalFp32, xLocalFp32, this->numLastDim); // yLocalFp32 <- x ** 2
        PipeBarrier<PIPE_V>();

        float squareSumTemp = ReduceSumHalfInterval(yLocalFp32V1, this->numLastDim);
        float rstdLocalTemp = 1 / sqrt(squareSumTemp * this->aveNum + this->eps);
        event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
        SetFlag<HardEvent::S_V>(eventSV);
        WaitFlag<HardEvent::S_V>(eventSV);
        Muls(xLocalFp32, xLocalFp32, rstdLocalTemp, this->numLastDim); // xLocalFp32 <- x * rstd
        PipeBarrier<PIPE_V>();
        LocalTensor<T_X> gammaLocal = inRowsQue.template DeQue<T_X>();

        AscendC::Cast(yLocalFp32V1, gammaLocal, RoundMode::CAST_NONE, this->numLastDim); // yLocalB16 <- Cast(gamma)
        inRowsQue.FreeTensor(gammaLocal);
        AscendC::Mul(xLocalFp32, xLocalFp32, yLocalFp32V1, this->numLastDim); // xLocalFp32 <- x * rstd * gamma
        PipeBarrier<PIPE_V>();
        if (this->betaExist) {
            CopyInBeta();
            LocalTensor<T_X> betaLocal = inRowsQue.template DeQue<T_X>();
            AscendC::Cast(yLocalFp32V1, betaLocal, RoundMode::CAST_NONE, this->numLastDim); // yLocalB16 <- Cast(beta)
            PipeBarrier<PIPE_V>();
            AscendC::Add(xLocalFp32, xLocalFp32, yLocalFp32V1, this->numLastDim);
            PipeBarrier<PIPE_V>();
            inRowsQue.FreeTensor(betaLocal);
        }
    }

    __aicore__ inline void CopyInGamma()
    {
        AscendC::LocalTensor<T_X> gammaCopyIn1 = inRowsQue.template AllocTensor<T_X>();
        AscendC::DataCopyPad(gammaCopyIn1[0], this->gammaGm, {1, static_cast<uint16_t>(this->numLastDim * sizeof(T_X)), 0, 0}, {});
        inRowsQue.EnQue(gammaCopyIn1);
    }

    __aicore__ inline void AddSingleRow(uint64_t gmOffset1)
    {
        auto x1x2Local = inRowsQue.template DeQue<T_X>();
        auto x1Local = x1x2Local[0];
        auto x2Local = x1x2Local[this->numLastDimAligned];

        auto xBufLocal = xBufFp32.Get<float>();
        auto yBufLocal1 = yBufFp32.Get<float>();

        // never have fp32 input here. All fp16/bf16 should cast to fp32 before Add
        AscendC::Cast(xBufLocal, x1Local, RoundMode::CAST_NONE, this->numLastDim);
        Cast(yBufLocal1, x2Local, RoundMode::CAST_NONE, this->numLastDim);
        inRowsQue.FreeTensor(x1x2Local);
        PipeBarrier<PIPE_V>();
        AscendC::Add(xBufLocal, yBufLocal1, xBufLocal, this->numLastDim);
        PipeBarrier<PIPE_V>();
        auto xLocal = yQue.template AllocTensor<T_X>();
        AscendC::Cast(xLocal, xBufLocal, RoundMode::CAST_RINT, this->numLastDim);
        yQue.template EnQue<T_X>(xLocal);

        PipeBarrier<PIPE_V>();
        auto x = yQue.template DeQue<T_X>();
        AscendC::DataCopyPad(this->xGm[gmOffset1], x, {1, static_cast<uint16_t>(this->numLastDim * sizeof(T_X)), 0, 0});
        yQue.FreeTensor(x);
    }

    __aicore__ inline void CopyInX1X2(uint64_t gmOffset1)
    {
        AscendC::LocalTensor<T_X> x1x2LocalIn = inRowsQue.template AllocTensor<T_X>();
        AscendC::DataCopyPad(x1x2LocalIn[0], this->x1Gm[gmOffset1], {1, static_cast<uint16_t>(this->numLastDim * sizeof(T_X)), 0, 0}, {});
        AscendC::DataCopyPad(x1x2LocalIn[this->numLastDimAligned], this->x2Gm[gmOffset1], {1, static_cast<uint16_t>(this->numLastDim * sizeof(T_X)), 0, 0}, {});
        inRowsQue.EnQue(x1x2LocalIn);
    }
    
    __aicore__ inline void InitBaseParams(const AddRmsNormDynamicQuantRegbaseTilingData* tilingData)
    {
        this->numCore = GetBlockNum();
        this->numLastDim = tilingData->numN;
        this->numLastDimAligned = CeilAlign(this->numLastDim, BLOCK_SIZE / sizeof(T_Y));

        this->firstDimPerCore = tilingData->mPerCore;
        this->firstDimPerCoreTail = tilingData->mLastCore;

        this->aveNum = tilingData->avgFactor;
        this->eps = tilingData->epsilon;

        this->blockIdx_ = GetBlockIdx();
        if (this->blockIdx_ == (this->numCore - 1)) {
            this->rowWork = this->firstDimPerCoreTail;
        } else {
            this->rowWork = this->firstDimPerCore;
        }
        this->gmOffset_ = this->firstDimPerCore * this->numLastDim;
        
        this->smooth1Exist = tilingData->hasSmoothScale1;
        this->smooth2Exist = tilingData->hasSmoothScale2;
        this->betaExist = tilingData->hasBeta;
        this->powerSplit_ = tilingData->powerSplit;
        if constexpr (IsSameType<T_Y, int8_t>::value) {
            this->quantMax = 1.0f / DIV_FACTOR_INT8;
        } else if constexpr (IsSameType<T_Y, fp8_e4m3fn_t>::value) {
            this->quantMax = 1.0f / DIV_FACTOR_FP8E4M3FN;
        } else if constexpr (IsSameType<T_Y, fp8_e5m2_t>::value) {
            this->quantMax = 1.0f / DIV_FACTOR_FP8E5M2;
        } else if constexpr (IsSameType<T_Y, hifloat8_t>::value) {
            this->quantMax = 1.0f / DIV_FACTOR_HIFP8;
        }
    }

    __aicore__ inline void InitInGlobalTensors(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR smooth1, GM_ADDR smooth2, GM_ADDR beta)
    {
        uint64_t gmLen = this->rowWork * this->numLastDim;
        x1Gm.SetGlobalBuffer((__gm__ T_X*)(x1) + this->blockIdx_ * this->gmOffset_, gmLen);
        x2Gm.SetGlobalBuffer((__gm__ T_X*)(x2) + this->blockIdx_ * this->gmOffset_, gmLen);
        gammaGm.SetGlobalBuffer((__gm__ T_X*)(gamma), this->numLastDim);
        if (this->smooth1Exist) {
            smooth1Gm.SetGlobalBuffer((__gm__ T_X*)(smooth1), this->numLastDim);
        }
        if (this->smooth2Exist) {
            smooth2Gm.SetGlobalBuffer((__gm__ T_X*)(smooth2), this->numLastDim);
        }
        if (this->betaExist) {
            betaGm.SetGlobalBuffer((__gm__ T_X*)(beta), this->numLastDim);
        }
    }

    __aicore__ inline void InitOutGlobalTensors(
        GM_ADDR y1, GM_ADDR y2, GM_ADDR x, GM_ADDR outScale1, GM_ADDR outScale2)
    {
        uint64_t gmLen = this->rowWork * this->numLastDim;
        y1Gm.SetGlobalBuffer((__gm__ T_Y*)(y1) + this->blockIdx_ * this->gmOffset_, gmLen);
        if (this->smooth2Exist) {
            y2Gm.SetGlobalBuffer((__gm__ T_Y*)(y2) + this->blockIdx_ * this->gmOffset_, gmLen);
        }
        xGm.SetGlobalBuffer((__gm__ T_X*)(x) + this->blockIdx_ * this->gmOffset_, gmLen);
        outScale1Gm.SetGlobalBuffer((__gm__ float*)(outScale1) + this->blockIdx_ * this->firstDimPerCore, this->rowWork);
        if (this->smooth2Exist) {
            outScale2Gm.SetGlobalBuffer((__gm__ float*)(outScale2) + this->blockIdx_ * this->firstDimPerCore, this->rowWork);
        }
    }

private:
    TPipe* Ppipe = nullptr;
    TQue<QuePosition::VECIN, BUFFER_NUM> inRowsQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> yQue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> scalesQue;
    TBuf<TPosition::VECCALC> xBufFp32;
    TBuf<TPosition::VECCALC> yBufFp32;
    TBuf<TPosition::VECCALC> smoothBuf;

    GlobalTensor<T_X> x2Gm;
    GlobalTensor<T_X> x1Gm;
    GlobalTensor<T_X> smooth1Gm;
    GlobalTensor<T_X> gammaGm;
    GlobalTensor<T_X> smooth2Gm;
    GlobalTensor<T_X> betaGm;
    GlobalTensor<T_Y> y1Gm;
    GlobalTensor<T_Y> y2Gm;
    GlobalTensor<T_X> xGm;
    GlobalTensor<float> outScale1Gm;
    GlobalTensor<float> outScale2Gm;

    uint64_t numCore;
    uint64_t numLastDim;
    uint64_t firstDimPerCore;
    uint64_t numLastDimAligned;
    uint64_t firstDimPerCoreTail;

    float aveNum;
    float eps;
    float quantMax;

    uint64_t gmOffset_;
    uint64_t blockIdx_;
    uint64_t rowWork;
    uint64_t powerSplit_{0};

    bool smooth1Exist;
    bool smooth2Exist;
    bool betaExist;
};

} // namespace AddRmsNormDynamicQuant

#endif // __ADD_RMS_NORM_DYNAMIC_QUANT_SINGLE_ROW_KERNEL_H_
