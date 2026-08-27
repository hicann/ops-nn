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
 * \file fake_quant_with_min_max_vars_per_channel_gradient_regbase.h
 * \brief regbase / Reg implementation for FakeQuantWithMinMaxVarsPerChannelGradient (FP32).
 *
 * D-chunked design: every UB-resident buffer is sized by dTileLen (not by D), so the
 * kernel scales to arbitrarily large channel count. Outer loop iterates d-chunks;
 * inner loop iterates rows; bp_min / bp_max are chunk-locally accumulated then
 * atomic-added to the corresponding GM slice.
 */

#ifndef FAKE_QUANT_WITH_MIN_MAX_VARS_PER_CHANNEL_GRADIENT_REGBASE_H
#define FAKE_QUANT_WITH_MIN_MAX_VARS_PER_CHANNEL_GRADIENT_REGBASE_H

#include "kernel_operator.h"

namespace FakeQuantWMMVPCG {
using namespace AscendC;

constexpr uint32_t FP32_VL = 64; // 256B / 4B per FP32 vector reg
constexpr uint32_t SPLIT_MODE_ROWS = 0;
constexpr uint32_t SPLIT_MODE_DCHUNKS = 1;

static constexpr Reg::CastTrait castTraitFp32ToInt32 = {Reg::RegLayout::UNKNOWN, Reg::SatMode::SAT,
                                                        Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
static constexpr Reg::CastTrait castTraitInt32ToFp32 = {Reg::RegLayout::ZERO, Reg::SatMode::SAT,
                                                        Reg::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};

template <typename T>
class Kernel {
public:
    __aicore__ inline Kernel() {}

    __aicore__ inline void Init(GM_ADDR gradients, GM_ADDR x, GM_ADDR minIn, GM_ADDR maxIn, GM_ADDR bpx, GM_ADDR bpMin,
                                GM_ADDR bpMax, GM_ADDR workspace,
                                const FakeQuantWithMinMaxVarsPerChannelGradientTilingData& tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void PrepareNudgedChunk(uint32_t dOffset, uint32_t dCount);
    __aicore__ inline void ZeroAccChunk(uint32_t dCount);
    __aicore__ inline void ProcessRowChunk(uint32_t absRowIdx, uint32_t dOffset, uint32_t dCount, bool writeBpx,
                                           bool accumulate);
    __aicore__ inline void ProcessRowBatch(uint32_t startRowIdx, uint32_t dOffset, uint32_t dCount, uint32_t batchRows);
    __aicore__ inline void ZeroWorkspaceSlot();
    __aicore__ inline void WriteAccToWorkspace(uint32_t dOffset, uint32_t dCount);
    __aicore__ inline void MergePhase();
    __aicore__ inline void ComputeChunkRegbase(__ubuf__ T* gPtr, __ubuf__ T* xPtr, __ubuf__ T* loPtr, __ubuf__ T* hiPtr,
                                               __ubuf__ T* bpxPtr, __ubuf__ T* bpMinAccPtr, __ubuf__ T* bpMaxAccPtr,
                                               __ubuf__ T* bpMinCompPtr, __ubuf__ T* bpMaxCompPtr, uint32_t calCount,
                                               bool accumulate);

    TPipe pipe_;
    GlobalTensor<T> gradGm_;
    GlobalTensor<T> xGm_;
    GlobalTensor<T> minGm_;
    GlobalTensor<T> maxGm_;
    GlobalTensor<T> bpxGm_;
    GlobalTensor<T> bpMinGm_;
    GlobalTensor<T> bpMaxGm_;

    // All sized by dTileLen.
    TBuf<TPosition::VECCALC> nudgedMinBuf_;
    TBuf<TPosition::VECCALC> nudgedMaxBuf_;
    TBuf<TPosition::VECCALC> bpMinAccBuf_;
    TBuf<TPosition::VECCALC> bpMaxAccBuf_;
    TBuf<TPosition::VECCALC> bpMinCompBuf_;
    TBuf<TPosition::VECCALC> bpMaxCompBuf_;
    TBuf<TPosition::VECCALC> gBuf_;
    TBuf<TPosition::VECCALC> xBuf_;
    TBuf<TPosition::VECCALC> bpxBuf_;

    uint32_t blockIdx_ = 0;
    uint32_t usedCoreNum_ = 0;
    uint32_t totalRows_ = 0;
    uint32_t channelNum_ = 0;
    uint32_t dTileLen_ = 0;
    uint32_t numDChunks_ = 0;
    uint32_t splitMode_ = 0;
    int32_t quantMinI_ = 0;
    int32_t quantMaxI_ = 0;

    uint32_t rowStart_ = 0;
    uint32_t rowEnd_ = 0;
    uint32_t chunkStart_ = 0;
    uint32_t chunkEnd_ = 0;
    uint32_t rowsPerBatch_ = 1;
    uint32_t wsStride_ = 0;
    GlobalTensor<T> workspaceGm_;
};

template <typename T>
__aicore__ inline void Kernel<T>::Init(GM_ADDR gradients, GM_ADDR x, GM_ADDR minIn, GM_ADDR maxIn, GM_ADDR bpx,
                                       GM_ADDR bpMin, GM_ADDR bpMax, GM_ADDR workspace,
                                       const FakeQuantWithMinMaxVarsPerChannelGradientTilingData& tilingData)
{
    blockIdx_ = GetBlockIdx();
    usedCoreNum_ = tilingData.usedCoreNum;
    channelNum_ = tilingData.channelNum;
    totalRows_ = tilingData.totalRows;
    dTileLen_ = tilingData.dTileLen;
    numDChunks_ = tilingData.numDChunks;
    splitMode_ = tilingData.splitMode;
    quantMinI_ = tilingData.quantMin;
    quantMaxI_ = tilingData.quantMax;
    rowsPerBatch_ = (channelNum_ > 0) ? (dTileLen_ / channelNum_) : 1;
    if (rowsPerBatch_ == 0) {
        rowsPerBatch_ = 1;
    }
    wsStride_ = ((channelNum_ + FP32_VL - 1) / FP32_VL) * FP32_VL;

    gradGm_.SetGlobalBuffer((__gm__ T*)gradients);
    xGm_.SetGlobalBuffer((__gm__ T*)x);
    minGm_.SetGlobalBuffer((__gm__ T*)minIn);
    maxGm_.SetGlobalBuffer((__gm__ T*)maxIn);
    bpxGm_.SetGlobalBuffer((__gm__ T*)bpx);
    bpMinGm_.SetGlobalBuffer((__gm__ T*)bpMin);
    bpMaxGm_.SetGlobalBuffer((__gm__ T*)bpMax);
    // WHY: skip the CANN system-reserved workspace head (used by SyncAll); writing user
    // accumulators from the raw base collided with the reserved region and overran the
    // tiling-declared workspace, tripping TTK's OOB sentinel.
    workspaceGm_.SetGlobalBuffer((__gm__ T*)AscendC::GetUserWorkspace(workspace));

    if (blockIdx_ >= usedCoreNum_) {
        return;
    }

    if (splitMode_ == SPLIT_MODE_ROWS) {
        uint32_t rowsPerCore = tilingData.rowsPerCore;
        uint32_t tailRows = tilingData.tailRows;
        if (blockIdx_ < tailRows) {
            rowStart_ = blockIdx_ * (rowsPerCore + 1);
            rowEnd_ = rowStart_ + rowsPerCore + 1;
        } else {
            rowStart_ = tailRows * (rowsPerCore + 1) + (blockIdx_ - tailRows) * rowsPerCore;
            rowEnd_ = rowStart_ + rowsPerCore;
        }
        chunkStart_ = 0;
        chunkEnd_ = numDChunks_;
    } else {
        uint32_t chunksPerCore = tilingData.chunksPerCore;
        uint32_t tailChunks = tilingData.tailChunks;
        if (blockIdx_ < tailChunks) {
            chunkStart_ = blockIdx_ * (chunksPerCore + 1);
            chunkEnd_ = chunkStart_ + chunksPerCore + 1;
        } else {
            chunkStart_ = tailChunks * (chunksPerCore + 1) + (blockIdx_ - tailChunks) * chunksPerCore;
            chunkEnd_ = chunkStart_ + chunksPerCore;
        }
        rowStart_ = 0;
        rowEnd_ = totalRows_;
    }

    uint32_t tileBytes = dTileLen_ * sizeof(T);
    pipe_.InitBuffer(nudgedMinBuf_, tileBytes);
    pipe_.InitBuffer(nudgedMaxBuf_, tileBytes);
    pipe_.InitBuffer(bpMinAccBuf_, tileBytes);
    pipe_.InitBuffer(bpMaxAccBuf_, tileBytes);
    pipe_.InitBuffer(bpMinCompBuf_, tileBytes);
    pipe_.InitBuffer(bpMaxCompBuf_, tileBytes);
    uint32_t batchBytes = rowsPerBatch_ * dTileLen_ * sizeof(T);
    pipe_.InitBuffer(gBuf_, batchBytes);
    pipe_.InitBuffer(xBuf_, batchBytes);
    pipe_.InitBuffer(bpxBuf_, tileBytes);
}

template <typename T>
__aicore__ inline void Kernel<T>::PrepareNudgedChunk(uint32_t dOffset, uint32_t dCount)
{
    LocalTensor<T> minLocal = nudgedMinBuf_.template Get<T>();
    LocalTensor<T> maxLocal = nudgedMaxBuf_.template Get<T>();

    DataCopyExtParams cp{1, static_cast<uint32_t>(dCount * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
    DataCopyPad(minLocal, minGm_[dOffset], cp, pad);
    DataCopyPad(maxLocal, maxGm_[dOffset], cp, pad);
    PipeBarrier<PIPE_ALL>();

    const float qRange = static_cast<float>(quantMaxI_ - quantMinI_);
    const float qminF = static_cast<float>(quantMinI_);
    const float qmaxF = static_cast<float>(quantMaxI_);

#ifdef __CCE_AICORE__
    uint16_t loopNum = static_cast<uint16_t>((dCount + FP32_VL - 1) / FP32_VL);
    __ubuf__ T* minPtr = (__ubuf__ T*)minLocal.GetPhyAddr();
    __ubuf__ T* maxPtr = (__ubuf__ T*)maxLocal.GetPhyAddr();

    __VEC_SCOPE__
    {
        // WHY high-precision Div: TF reference path uses zp = qmin - min/scale.
        // Default vdiv on dav_c310 only achieves ~14-bit precision and round-mismatches
        // when min ~= -max, causing zp to land on the wrong int (off by 1). The
        // 0 ULP IEEE 754 Div (PRECISION_0ULP_FTZ_FALSE) gives bit-equal results
        // with CPU/TF for fp32 division. See zp-precision-trap.md.
        static constexpr AscendC::Reg::DivSpecificMode DIV_HP_ZP_MODE = {
            .mrgMode = AscendC::Reg::MaskMergeMode::ZEROING,
            .precisionMode = false,
            .algo = AscendC::DivAlgo::PRECISION_0ULP_FTZ_FALSE};
        AscendC::Reg::RegTensor<float> vMin;
        AscendC::Reg::RegTensor<float> vMax;
        AscendC::Reg::RegTensor<float> vScale;
        AscendC::Reg::RegTensor<float> vZpf;
        AscendC::Reg::RegTensor<float> vNzpRound;
        AscendC::Reg::RegTensor<int32_t> vNzpInt;
        AscendC::Reg::RegTensor<float> vNzp;
        AscendC::Reg::RegTensor<float> vNudgedMin;
        AscendC::Reg::RegTensor<float> vNudgedMax;
        AscendC::Reg::RegTensor<float> vQminVec;
        AscendC::Reg::RegTensor<float> vQmaxVec;
        AscendC::Reg::RegTensor<float> vQRange;
        AscendC::Reg::RegTensor<float> vZero;
        AscendC::Reg::RegTensor<float> vNegInf;
        AscendC::Reg::RegTensor<float> vPosInf;
        AscendC::Reg::MaskReg pMask;
        AscendC::Reg::MaskReg cmpLo;
        AscendC::Reg::MaskReg cmpHi;
        AscendC::Reg::MaskReg cmpDegenerate;

        AscendC::Reg::Duplicate(vQminVec, qminF);
        AscendC::Reg::Duplicate(vQmaxVec, qmaxF);
        AscendC::Reg::Duplicate(vQRange, qRange);
        AscendC::Reg::Duplicate(vZero, 0.0f);
        AscendC::Reg::Duplicate(vNegInf, -3.4e38f);
        AscendC::Reg::Duplicate(vPosInf, 3.4e38f);

        uint32_t remain = dCount;
        for (uint16_t i = 0; i < loopNum; ++i) {
            pMask = AscendC::Reg::UpdateMask<float>(remain);
            AscendC::Reg::DataCopy(vMin, minPtr + i * FP32_VL);
            AscendC::Reg::DataCopy(vMax, maxPtr + i * FP32_VL);

            AscendC::Reg::Sub(vScale, vMax, vMin, pMask);
            AscendC::Reg::Compare<float, CMPMODE::LE>(cmpDegenerate, vScale, vZero, pMask);
            AscendC::Reg::Div<float, &DIV_HP_ZP_MODE>(vScale, vScale, vQRange, pMask);

            // zp = qmin - min/scale (TF original formula). High-precision 0 ULP Div
            // is required because min ~= -max would otherwise round wrong (see DIV_HP_ZP_MODE above).
            // degenerate (scale==0) may produce inf/NaN here, but is overwritten via cmpDegenerate
            // Select on vNudgedMin/vNudgedMax below (zp itself is not stored).
            AscendC::Reg::Div<float, &DIV_HP_ZP_MODE>(vZpf, vMin, vScale, pMask);
            AscendC::Reg::Sub(vZpf, vQminVec, vZpf, pMask);

            AscendC::Reg::Cast<int32_t, float, castTraitFp32ToInt32>(vNzpInt, vZpf, pMask);
            AscendC::Reg::Cast<float, int32_t, castTraitInt32ToFp32>(vNzpRound, vNzpInt, pMask);

            AscendC::Reg::Compare<float, CMPMODE::LT>(cmpLo, vZpf, vQminVec, pMask);
            AscendC::Reg::Compare<float, CMPMODE::GT>(cmpHi, vZpf, vQmaxVec, pMask);

            AscendC::Reg::Select(vNzp, vQminVec, vNzpRound, cmpLo);
            AscendC::Reg::Select(vNzp, vQmaxVec, vNzp, cmpHi);

            AscendC::Reg::Sub(vNudgedMin, vQminVec, vNzp, pMask);
            AscendC::Reg::Mul(vNudgedMin, vNudgedMin, vScale, pMask);
            AscendC::Reg::Sub(vNudgedMax, vQmaxVec, vNzp, pMask);
            AscendC::Reg::Mul(vNudgedMax, vNudgedMax, vScale, pMask);

            AscendC::Reg::Select(vNudgedMin, vNegInf, vNudgedMin, cmpDegenerate);
            AscendC::Reg::Select(vNudgedMax, vPosInf, vNudgedMax, cmpDegenerate);

            AscendC::Reg::DataCopy(minPtr + i * FP32_VL, vNudgedMin, pMask);
            AscendC::Reg::DataCopy(maxPtr + i * FP32_VL, vNudgedMax, pMask);
        }
    }
#endif
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline void Kernel<T>::ZeroAccChunk(uint32_t dCount)
{
    LocalTensor<T> bpMinAccLocal = bpMinAccBuf_.template Get<T>();
    LocalTensor<T> bpMaxAccLocal = bpMaxAccBuf_.template Get<T>();
    LocalTensor<T> bpMinCompLocal = bpMinCompBuf_.template Get<T>();
    LocalTensor<T> bpMaxCompLocal = bpMaxCompBuf_.template Get<T>();
#ifdef __CCE_AICORE__
    __ubuf__ T* accMinPtr = (__ubuf__ T*)bpMinAccLocal.GetPhyAddr();
    __ubuf__ T* accMaxPtr = (__ubuf__ T*)bpMaxAccLocal.GetPhyAddr();
    __ubuf__ T* compMinPtr = (__ubuf__ T*)bpMinCompLocal.GetPhyAddr();
    __ubuf__ T* compMaxPtr = (__ubuf__ T*)bpMaxCompLocal.GetPhyAddr();
    uint16_t loopNum = static_cast<uint16_t>((dCount + FP32_VL - 1) / FP32_VL);
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<float> vZero;
        AscendC::Reg::MaskReg pMask;
        AscendC::Reg::Duplicate(vZero, 0.0f);
        uint32_t remain = dCount;
        for (uint16_t i = 0; i < loopNum; ++i) {
            pMask = AscendC::Reg::UpdateMask<float>(remain);
            AscendC::Reg::DataCopy(accMinPtr + i * FP32_VL, vZero, pMask);
            AscendC::Reg::DataCopy(accMaxPtr + i * FP32_VL, vZero, pMask);
            AscendC::Reg::DataCopy(compMinPtr + i * FP32_VL, vZero, pMask);
            AscendC::Reg::DataCopy(compMaxPtr + i * FP32_VL, vZero, pMask);
        }
    }
#endif
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline void Kernel<T>::ComputeChunkRegbase(__ubuf__ T* gPtr, __ubuf__ T* xPtr, __ubuf__ T* loPtr,
                                                      __ubuf__ T* hiPtr, __ubuf__ T* bpxPtr, __ubuf__ T* bpMinAccPtr,
                                                      __ubuf__ T* bpMaxAccPtr, __ubuf__ T* bpMinCompPtr,
                                                      __ubuf__ T* bpMaxCompPtr, uint32_t calCount, bool accumulate)
{
#ifdef __CCE_AICORE__
    uint16_t loopNum = static_cast<uint16_t>((calCount + FP32_VL - 1) / FP32_VL);

    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<float> vG;
        AscendC::Reg::RegTensor<float> vX;
        AscendC::Reg::RegTensor<float> vLo;
        AscendC::Reg::RegTensor<float> vHi;
        AscendC::Reg::RegTensor<float> vBpx;
        AscendC::Reg::RegTensor<float> vBpMinAcc;
        AscendC::Reg::RegTensor<float> vBpMaxAcc;
        AscendC::Reg::RegTensor<float> vAddMin;
        AscendC::Reg::RegTensor<float> vAddMax;
        AscendC::Reg::RegTensor<float> vCMin;
        AscendC::Reg::RegTensor<float> vCMax;
        AscendC::Reg::RegTensor<float> vY;
        AscendC::Reg::RegTensor<float> vT;
        AscendC::Reg::RegTensor<float> vZero;
        AscendC::Reg::MaskReg pMask;
        AscendC::Reg::MaskReg cmpLo;
        AscendC::Reg::MaskReg cmpHi;

        AscendC::Reg::Duplicate(vZero, 0.0f);

        uint32_t remain = calCount;
        for (uint16_t i = 0; i < loopNum; ++i) {
            pMask = AscendC::Reg::UpdateMask<float>(remain);

            AscendC::Reg::DataCopy(vG, gPtr + i * FP32_VL);
            AscendC::Reg::DataCopy(vX, xPtr + i * FP32_VL);
            AscendC::Reg::DataCopy(vLo, loPtr + i * FP32_VL);
            AscendC::Reg::DataCopy(vHi, hiPtr + i * FP32_VL);
            if (accumulate) {
                AscendC::Reg::DataCopy(vBpMinAcc, bpMinAccPtr + i * FP32_VL);
                AscendC::Reg::DataCopy(vBpMaxAcc, bpMaxAccPtr + i * FP32_VL);
                AscendC::Reg::DataCopy(vCMin, bpMinCompPtr + i * FP32_VL);
                AscendC::Reg::DataCopy(vCMax, bpMaxCompPtr + i * FP32_VL);
            }

            AscendC::Reg::Compare<float, CMPMODE::LT>(cmpLo, vX, vLo, pMask);
            AscendC::Reg::Compare<float, CMPMODE::GT>(cmpHi, vX, vHi, pMask);

            AscendC::Reg::RegTensor<float> vOne;
            AscendC::Reg::RegTensor<float> vMask;
            AscendC::Reg::Duplicate(vOne, 1.0f);
            AscendC::Reg::Select(vMask, vZero, vOne, cmpLo);  // 0 if x<lo else 1
            AscendC::Reg::Select(vMask, vZero, vMask, cmpHi); // 0 if x>hi else keep
            AscendC::Reg::Mul(vBpx, vG, vMask, pMask);
            AscendC::Reg::DataCopy(bpxPtr + i * FP32_VL, vBpx, pMask);

            if (accumulate) {
                // Kahan compensated summation for bp_min:
                //   y = input - c;  t = acc + y;  c = (t - acc) - y;  acc = t
                AscendC::Reg::Select(vMask, vOne, vZero, cmpLo);
                AscendC::Reg::Mul(vAddMin, vG, vMask, pMask);
                AscendC::Reg::Sub(vY, vAddMin, vCMin, pMask);
                AscendC::Reg::Add(vT, vBpMinAcc, vY, pMask);
                AscendC::Reg::Sub(vCMin, vT, vBpMinAcc, pMask);
                AscendC::Reg::Sub(vCMin, vCMin, vY, pMask);
                AscendC::Reg::DataCopy(bpMinCompPtr + i * FP32_VL, vCMin, pMask);
                AscendC::Reg::DataCopy(bpMinAccPtr + i * FP32_VL, vT, pMask);

                // Kahan compensated summation for bp_max
                AscendC::Reg::Select(vMask, vOne, vZero, cmpHi);
                AscendC::Reg::Mul(vAddMax, vG, vMask, pMask);
                AscendC::Reg::Sub(vY, vAddMax, vCMax, pMask);
                AscendC::Reg::Add(vT, vBpMaxAcc, vY, pMask);
                AscendC::Reg::Sub(vCMax, vT, vBpMaxAcc, pMask);
                AscendC::Reg::Sub(vCMax, vCMax, vY, pMask);
                AscendC::Reg::DataCopy(bpMaxCompPtr + i * FP32_VL, vCMax, pMask);
                AscendC::Reg::DataCopy(bpMaxAccPtr + i * FP32_VL, vT, pMask);
            }
        }
    }
#endif
}

template <typename T>
__aicore__ inline void Kernel<T>::ProcessRowChunk(uint32_t absRowIdx, uint32_t dOffset, uint32_t dCount, bool writeBpx,
                                                  bool accumulate)
{
    LocalTensor<T> nudgedMinLocal = nudgedMinBuf_.template Get<T>();
    LocalTensor<T> nudgedMaxLocal = nudgedMaxBuf_.template Get<T>();
    LocalTensor<T> bpMinAccLocal = bpMinAccBuf_.template Get<T>();
    LocalTensor<T> bpMaxAccLocal = bpMaxAccBuf_.template Get<T>();
    LocalTensor<T> bpMinCompLocal = bpMinCompBuf_.template Get<T>();
    LocalTensor<T> bpMaxCompLocal = bpMaxCompBuf_.template Get<T>();
    LocalTensor<T> gLocal = gBuf_.template Get<T>();
    LocalTensor<T> xLocal = xBuf_.template Get<T>();
    LocalTensor<T> bpxLocal = bpxBuf_.template Get<T>();

    uint64_t gmOffset = static_cast<uint64_t>(absRowIdx) * channelNum_ + dOffset;
    DataCopyExtParams cp{1, static_cast<uint32_t>(dCount * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
    DataCopyPad(gLocal, gradGm_[gmOffset], cp, pad);
    DataCopyPad(xLocal, xGm_[gmOffset], cp, pad);
    PipeBarrier<PIPE_ALL>();

#ifdef __CCE_AICORE__
    __ubuf__ T* loPtr = (__ubuf__ T*)nudgedMinLocal.GetPhyAddr();
    __ubuf__ T* hiPtr = (__ubuf__ T*)nudgedMaxLocal.GetPhyAddr();
    __ubuf__ T* bpMinAccPtr = (__ubuf__ T*)bpMinAccLocal.GetPhyAddr();
    __ubuf__ T* bpMaxAccPtr = (__ubuf__ T*)bpMaxAccLocal.GetPhyAddr();
    __ubuf__ T* bpMinCompPtr = (__ubuf__ T*)bpMinCompLocal.GetPhyAddr();
    __ubuf__ T* bpMaxCompPtr = (__ubuf__ T*)bpMaxCompLocal.GetPhyAddr();
    __ubuf__ T* gPtr = (__ubuf__ T*)gLocal.GetPhyAddr();
    __ubuf__ T* xPtr = (__ubuf__ T*)xLocal.GetPhyAddr();
    __ubuf__ T* bpxPtr = (__ubuf__ T*)bpxLocal.GetPhyAddr();
    ComputeChunkRegbase(gPtr, xPtr, loPtr, hiPtr, bpxPtr, bpMinAccPtr, bpMaxAccPtr, bpMinCompPtr, bpMaxCompPtr, dCount,
                        accumulate);
#endif

    PipeBarrier<PIPE_ALL>();
    if (writeBpx) {
        DataCopyPad(bpxGm_[gmOffset], bpxLocal, cp);
        PipeBarrier<PIPE_ALL>();
    }
}

template <typename T>
__aicore__ inline void Kernel<T>::ProcessRowBatch(uint32_t startRowIdx, uint32_t dOffset, uint32_t dCount,
                                                  uint32_t batchRows)
{
    LocalTensor<T> nudgedMinLocal = nudgedMinBuf_.template Get<T>();
    LocalTensor<T> nudgedMaxLocal = nudgedMaxBuf_.template Get<T>();
    LocalTensor<T> bpMinAccLocal = bpMinAccBuf_.template Get<T>();
    LocalTensor<T> bpMaxAccLocal = bpMaxAccBuf_.template Get<T>();
    LocalTensor<T> bpMinCompLocal = bpMinCompBuf_.template Get<T>();
    LocalTensor<T> bpMaxCompLocal = bpMaxCompBuf_.template Get<T>();
    LocalTensor<T> gLocal = gBuf_.template Get<T>();
    LocalTensor<T> xLocal = xBuf_.template Get<T>();
    LocalTensor<T> bpxLocal = bpxBuf_.template Get<T>();

    uint32_t totalElems = dCount * batchRows;
    DataCopyExtParams cpBatch{1, static_cast<uint32_t>(totalElems * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};
    DataCopyExtParams cpRow{1, static_cast<uint32_t>(dCount * sizeof(T)), 0, 0, 0};

    uint64_t gmOffset = static_cast<uint64_t>(startRowIdx) * channelNum_ + dOffset;
    DataCopyPad(gLocal, gradGm_[gmOffset], cpBatch, pad);
    DataCopyPad(xLocal, xGm_[gmOffset], cpBatch, pad);
    PipeBarrier<PIPE_ALL>();

#ifdef __CCE_AICORE__
    __ubuf__ T* loPtr = (__ubuf__ T*)nudgedMinLocal.GetPhyAddr();
    __ubuf__ T* hiPtr = (__ubuf__ T*)nudgedMaxLocal.GetPhyAddr();
    __ubuf__ T* bpMinAccPtr = (__ubuf__ T*)bpMinAccLocal.GetPhyAddr();
    __ubuf__ T* bpMaxAccPtr = (__ubuf__ T*)bpMaxAccLocal.GetPhyAddr();
    __ubuf__ T* bpMinCompPtr = (__ubuf__ T*)bpMinCompLocal.GetPhyAddr();
    __ubuf__ T* bpMaxCompPtr = (__ubuf__ T*)bpMaxCompLocal.GetPhyAddr();
    __ubuf__ T* gPtr = (__ubuf__ T*)gLocal.GetPhyAddr();
    __ubuf__ T* xPtr = (__ubuf__ T*)xLocal.GetPhyAddr();
    __ubuf__ T* bpxPtr = (__ubuf__ T*)bpxLocal.GetPhyAddr();

    for (uint32_t r = 0; r < batchRows; ++r) {
        ComputeChunkRegbase(gPtr + r * dCount, xPtr + r * dCount, loPtr, hiPtr, bpxPtr, bpMinAccPtr, bpMaxAccPtr,
                            bpMinCompPtr, bpMaxCompPtr, dCount, true);
        PipeBarrier<PIPE_ALL>();
        uint64_t rowOffset = static_cast<uint64_t>(startRowIdx + r) * channelNum_ + dOffset;
        DataCopyPad(bpxGm_[rowOffset], bpxLocal, cpRow);
        PipeBarrier<PIPE_ALL>();
    }
#endif
}

template <typename T>
__aicore__ inline void Kernel<T>::ZeroWorkspaceSlot()
{
    LocalTensor<T> bpMinAccLocal = bpMinAccBuf_.template Get<T>();
    LocalTensor<T> bpMaxAccLocal = bpMaxAccBuf_.template Get<T>();
#ifdef __CCE_AICORE__
    __ubuf__ T* minPtr = (__ubuf__ T*)bpMinAccLocal.GetPhyAddr();
    __ubuf__ T* maxPtr = (__ubuf__ T*)bpMaxAccLocal.GetPhyAddr();
    uint16_t loopNum = static_cast<uint16_t>((channelNum_ + FP32_VL - 1) / FP32_VL);
    __VEC_SCOPE__
    {
        AscendC::Reg::RegTensor<float> vZero;
        AscendC::Reg::MaskReg pMask;
        AscendC::Reg::Duplicate(vZero, 0.0f);
        uint32_t remain = channelNum_;
        for (uint16_t i = 0; i < loopNum; ++i) {
            pMask = AscendC::Reg::UpdateMask<float>(remain);
            AscendC::Reg::DataCopy(minPtr + i * FP32_VL, vZero, pMask);
            AscendC::Reg::DataCopy(maxPtr + i * FP32_VL, vZero, pMask);
        }
    }
#endif
    PipeBarrier<PIPE_ALL>();
    DataCopyExtParams cp{1, static_cast<uint32_t>(channelNum_ * sizeof(T)), 0, 0, 0};
    uint64_t minWsOffset = static_cast<uint64_t>(blockIdx_) * wsStride_;
    uint64_t maxWsOffset = static_cast<uint64_t>(usedCoreNum_) * wsStride_ + minWsOffset;
    DataCopyPad(workspaceGm_[minWsOffset], bpMinAccLocal, cp);
    DataCopyPad(workspaceGm_[maxWsOffset], bpMaxAccLocal, cp);
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline void Kernel<T>::WriteAccToWorkspace(uint32_t dOffset, uint32_t dCount)
{
    LocalTensor<T> bpMinAccLocal = bpMinAccBuf_.template Get<T>();
    LocalTensor<T> bpMaxAccLocal = bpMaxAccBuf_.template Get<T>();
    PipeBarrier<PIPE_ALL>();
    DataCopyExtParams cp{1, static_cast<uint32_t>(dCount * sizeof(T)), 0, 0, 0};
    uint64_t minWsOffset = static_cast<uint64_t>(blockIdx_) * wsStride_ + dOffset;
    uint64_t maxWsOffset = static_cast<uint64_t>(usedCoreNum_) * wsStride_ + minWsOffset;
    DataCopyPad(workspaceGm_[minWsOffset], bpMinAccLocal, cp);
    DataCopyPad(workspaceGm_[maxWsOffset], bpMaxAccLocal, cp);
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline void Kernel<T>::MergePhase()
{
    uint32_t channelsPerCore = channelNum_ / usedCoreNum_;
    uint32_t tailChannels = channelNum_ % usedCoreNum_;
    uint32_t myChStart, myChEnd;
    if (blockIdx_ < tailChannels) {
        myChStart = blockIdx_ * (channelsPerCore + 1);
        myChEnd = myChStart + channelsPerCore + 1;
    } else {
        myChStart = tailChannels * (channelsPerCore + 1) + (blockIdx_ - tailChannels) * channelsPerCore;
        myChEnd = myChStart + channelsPerCore;
    }
    if (myChStart >= myChEnd) {
        return;
    }

    for (uint32_t c = chunkStart_; c < chunkEnd_; ++c) {
        uint32_t dOffset = c * dTileLen_;
        uint32_t dEnd = dOffset + dTileLen_;
        if (dEnd > channelNum_) {
            dEnd = channelNum_;
        }
        uint32_t mergeStart = (dOffset > myChStart) ? dOffset : myChStart;
        uint32_t mergeEnd = (dEnd < myChEnd) ? dEnd : myChEnd;
        if (mergeStart >= mergeEnd) {
            continue;
        }
        uint32_t dCount = mergeEnd - mergeStart;

        LocalTensor<T> bpMinAccLocal = bpMinAccBuf_.template Get<T>();
        LocalTensor<T> bpMaxAccLocal = bpMaxAccBuf_.template Get<T>();

        DataCopyExtParams cp{1, static_cast<uint32_t>(dCount * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> pad{false, 0, 0, static_cast<T>(0)};

        uint64_t minBase = static_cast<uint64_t>(0) * wsStride_ + mergeStart;
        uint64_t maxBase = static_cast<uint64_t>(usedCoreNum_) * wsStride_ + minBase;
        DataCopyPad(bpMinAccLocal, workspaceGm_[minBase], cp, pad);
        DataCopyPad(bpMaxAccLocal, workspaceGm_[maxBase], cp, pad);
        PipeBarrier<PIPE_ALL>();

        for (uint32_t core = 1; core < usedCoreNum_; ++core) {
            uint64_t minOffset = static_cast<uint64_t>(core) * wsStride_ + mergeStart;
            uint64_t maxOffset = static_cast<uint64_t>(usedCoreNum_) * wsStride_ + minOffset;

            LocalTensor<T> minLocal = nudgedMinBuf_.template Get<T>();
            LocalTensor<T> maxLocal = nudgedMaxBuf_.template Get<T>();
            DataCopyPad(minLocal, workspaceGm_[minOffset], cp, pad);
            DataCopyPad(maxLocal, workspaceGm_[maxOffset], cp, pad);
            PipeBarrier<PIPE_ALL>();

#ifdef __CCE_AICORE__
            __ubuf__ T* accMinPtr = (__ubuf__ T*)bpMinAccLocal.GetPhyAddr();
            __ubuf__ T* accMaxPtr = (__ubuf__ T*)bpMaxAccLocal.GetPhyAddr();
            __ubuf__ T* srcMinPtr = (__ubuf__ T*)minLocal.GetPhyAddr();
            __ubuf__ T* srcMaxPtr = (__ubuf__ T*)maxLocal.GetPhyAddr();
            uint16_t loopNum = static_cast<uint16_t>((dCount + FP32_VL - 1) / FP32_VL);
            __VEC_SCOPE__
            {
                AscendC::Reg::RegTensor<float> vAcc;
                AscendC::Reg::RegTensor<float> vSrc;
                AscendC::Reg::MaskReg pMask;
                uint32_t remain = dCount;
                for (uint16_t i = 0; i < loopNum; ++i) {
                    pMask = AscendC::Reg::UpdateMask<float>(remain);
                    AscendC::Reg::DataCopy(vAcc, accMinPtr + i * FP32_VL);
                    AscendC::Reg::DataCopy(vSrc, srcMinPtr + i * FP32_VL);
                    AscendC::Reg::Add(vAcc, vAcc, vSrc, pMask);
                    AscendC::Reg::DataCopy(accMinPtr + i * FP32_VL, vAcc, pMask);

                    AscendC::Reg::DataCopy(vAcc, accMaxPtr + i * FP32_VL);
                    AscendC::Reg::DataCopy(vSrc, srcMaxPtr + i * FP32_VL);
                    AscendC::Reg::Add(vAcc, vAcc, vSrc, pMask);
                    AscendC::Reg::DataCopy(accMaxPtr + i * FP32_VL, vAcc, pMask);
                }
            }
#endif
            PipeBarrier<PIPE_ALL>();
        }

        DataCopyPad(bpMinGm_[mergeStart], bpMinAccLocal, cp);
        DataCopyPad(bpMaxGm_[mergeStart], bpMaxAccLocal, cp);
        PipeBarrier<PIPE_ALL>();
    }
}

template <typename T>
__aicore__ inline void Kernel<T>::Process()
{
    if (blockIdx_ >= usedCoreNum_) {
        return;
    }

    // WHY: split_mode=1 partitions disjoint d-chunks across cores, so per-core channel ranges
    // never overlap — no cross-core merge / workspace / SyncAll required, write each chunk's
    // accumulator directly to bp_min/bp_max GM. Avoids GB-scale workspace at huge channelNum.
    if (splitMode_ == SPLIT_MODE_DCHUNKS) {
        for (uint32_t c = chunkStart_; c < chunkEnd_; ++c) {
            uint32_t dOffset = c * dTileLen_;
            uint32_t dCount = (channelNum_ - dOffset >= dTileLen_) ? dTileLen_ : (channelNum_ - dOffset);

            PrepareNudgedChunk(dOffset, dCount);
            ZeroAccChunk(dCount);

            uint32_t numRows = rowEnd_ - rowStart_;
            uint32_t numBatches = numRows / rowsPerBatch_;
            uint32_t remainderRows = numRows % rowsPerBatch_;
            for (uint32_t b = 0; b < numBatches; ++b) {
                ProcessRowBatch(rowStart_ + b * rowsPerBatch_, dOffset, dCount, rowsPerBatch_);
            }
            for (uint32_t r = rowEnd_ - remainderRows; r < rowEnd_; ++r) {
                ProcessRowChunk(r, dOffset, dCount, /*writeBpx=*/true, /*accumulate=*/true);
            }

            LocalTensor<T> bpMinAccLocal = bpMinAccBuf_.template Get<T>();
            LocalTensor<T> bpMaxAccLocal = bpMaxAccBuf_.template Get<T>();
            PipeBarrier<PIPE_ALL>();
            DataCopyExtParams cp{1, static_cast<uint32_t>(dCount * sizeof(T)), 0, 0, 0};
            DataCopyPad(bpMinGm_[dOffset], bpMinAccLocal, cp);
            DataCopyPad(bpMaxGm_[dOffset], bpMaxAccLocal, cp);
            PipeBarrier<PIPE_ALL>();
        }
        return;
    }

    ZeroWorkspaceSlot();

    for (uint32_t c = chunkStart_; c < chunkEnd_; ++c) {
        uint32_t dOffset = c * dTileLen_;
        uint32_t dCount = (channelNum_ - dOffset >= dTileLen_) ? dTileLen_ : (channelNum_ - dOffset);

        PrepareNudgedChunk(dOffset, dCount);
        ZeroAccChunk(dCount);

        uint32_t numRows = rowEnd_ - rowStart_;
        uint32_t numBatches = numRows / rowsPerBatch_;
        uint32_t remainderRows = numRows % rowsPerBatch_;
        for (uint32_t b = 0; b < numBatches; ++b) {
            ProcessRowBatch(rowStart_ + b * rowsPerBatch_, dOffset, dCount, rowsPerBatch_);
        }
        for (uint32_t r = rowEnd_ - remainderRows; r < rowEnd_; ++r) {
            ProcessRowChunk(r, dOffset, dCount, /*writeBpx=*/true, /*accumulate=*/true);
        }

        WriteAccToWorkspace(dOffset, dCount);
    }

    SyncAll();
    MergePhase();
}

} // namespace FakeQuantWMMVPCG

#endif // FAKE_QUANT_WITH_MIN_MAX_VARS_PER_CHANNEL_GRADIENT_REGBASE_H
