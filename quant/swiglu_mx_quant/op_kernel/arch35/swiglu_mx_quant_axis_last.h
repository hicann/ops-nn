/* *
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file swiglu_mx_quant_last_last.h
 * \brief Regbase implementation for Swiglu + MX quantization (activate_dim=-1, axis=-1)
 */

#ifndef SWIGLU_MX_QUANT_AXIS_LAST_H
#define SWIGLU_MX_QUANT_AXIS_LAST_H

#include "swiglu_mx_quant_common.h"
#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#include "op_kernel/platform_util.h"
#include "kernel_tiling/kernel_tiling.h"

namespace SwigluMxQuant {
using namespace AscendC;
// Regbase class for SwiGLU + MX Quantization
// T: Input data type (half, bfloat16_t)
// ComputeT: Computation data type (float)
// QuantT: Quantized output type (fp8_e4m3fn_t, fp8_e5m2_t, etc.)
// IsTailAxis: Whether axis is the last axis (true for axis=-1)
template <typename T, typename U, typename T_IDX, bool isGroupIndex, RoundMode roundMode, bool isLast>
class SwigluMxQuantAxisLast {
public:
    __aicore__ inline SwigluMxQuantAxisLast(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR group_index, GM_ADDR y, GM_ADDR mxscale, GM_ADDR workspace,
        const SwigluMxQuantTilingData *__restrict tilingData, AscendC::TPipe *pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitGroupIndex(int64_t &allNum);
    __aicore__ inline void Compute(int64_t dim0Size, int64_t dim1Size, int64_t dim1AlignSize);
    __aicore__ inline void ComputeF8(__local_mem__ T *swigluUbAddr, __local_mem__ uint16_t *maxExpUbAddr,
        __local_mem__ uint16_t *halfScaleLocalAddr, int64_t dim0Size, int64_t dim1AlignSize);
    __aicore__ inline void ComputeVfSwigluV2(__local_mem__ T *x1UbAddr, __local_mem__ T *x2UbAddr,
        __local_mem__ T *swigluUbAddr, int64_t dim0OnceSize, int64_t dim1OnceSize, int64_t dim1AlignSize);

    __aicore__ inline void CopyIn(int64_t rowOffset, int64_t colBlockStart, int64_t dim0OnceSize, int64_t dim1OnceSize);
    __aicore__ inline void CopyOut(int64_t rowOffset, int64_t colBlockStart, int64_t dim0OnceSize, int64_t dim1OnceSize,
        int64_t dim1OnceSizeAlgin);

private:
    GlobalTensor<T> xGm_;
    GlobalTensor<T_IDX> groupIndexGm_;
    GlobalTensor<uint8_t> yGm_;
    GlobalTensor<uint8_t> scaleGm_;
    const SwigluMxQuantTilingData *tiling_;
    AscendC::TPipe *pipe_;
    int32_t blockIdx_ = 0;

    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQuex_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQuey_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueScale_;

    TBuf<QuePosition::VECCALC> reduceSumBuffer_;
    TBuf<QuePosition::VECCALC> swigluBuffer_;
    TBuf<QuePosition::VECCALC> maxExpBuffer_;
    TBuf<QuePosition::VECCALC> halfScaleBuffer_;

    int64_t realCoreNum_ = 0;
    int64_t activateLeft_ = 0;
    int64_t scaleAlg_ = 0;
    int64_t bStart_ = 0; // 本核 batch 起始
    int64_t bEnd_ = 1;

    float clampLimit_ = 0.0f;
    float gluBias_ = 0.0f;
    float gluAlpha_ = 0.0f;
    int64_t swigluMode_ = 0;

    int64_t dimM_ = 0;
    int64_t dim2N_ = 0;
    int64_t dimN_ = 0;
    int64_t factorDim0Size_ = 0;
    int64_t factorDim1Size_ = 0;

    // 3D core grid (B=1, M-split, N-split)
    int64_t mStart_ = 0;
    int64_t nStart_ = 0;
    int64_t loopTimesPerBatch_ = 0;
    int64_t tailPerBatch_ = 0;
    int64_t loopTimesN_ = 0;
    int64_t tailN_ = 0;
    uint16_t f4Emax_ = 0;
    uint32_t dtypeMax_ = 0;
    uint16_t f8Emax_ = 0;
    int64_t outputScaleRowBytes_ = 0;
};

template <typename T, typename U, typename T_IDX, bool isGroupIndex, RoundMode roundMode, bool isLast>
__aicore__ inline void SwigluMxQuantAxisLast<T, U, T_IDX, isGroupIndex, roundMode, isLast>::InitGroupIndex(
    int64_t &allNum)
{
    int64_t groupIndexNum = tiling_->groupIndexNum;
    LocalTensor<T_IDX> reduceSumUb = reduceSumBuffer_.Get<T_IDX>();
    LocalTensor<T_IDX> groupIndexUb = inQuex_.AllocTensor<T_IDX>();
    DataCopyExtParams copyInParam = { 1, 0, 0, 0, 0 };
    DataCopyPadExtParams<T_IDX> copyPadParams = { false, 0, 0, 0 };
    copyInParam.blockLen = groupIndexNum * sizeof(T_IDX);
    DataCopyPad(groupIndexUb, groupIndexGm_[0], copyInParam, copyPadParams);
    inQuex_.EnQue(groupIndexUb);
    groupIndexUb = inQuex_.DeQue<T_IDX>();
    ReduceAllVf<T_IDX>(reduceSumUb, groupIndexUb, groupIndexNum);
    inQuex_.FreeTensor(groupIndexUb);
    event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventVS);
    WaitFlag<HardEvent::V_S>(eventVS);
    allNum = static_cast<int64_t>(reduceSumUb.GetValue(0));
}

template <typename T, typename U, typename T_IDX, bool isGroupIndex, RoundMode roundMode, bool isLast>
__aicore__ inline void SwigluMxQuantAxisLast<T, U, T_IDX, isGroupIndex, roundMode, isLast>::Init(GM_ADDR x,
    GM_ADDR group_index, GM_ADDR y, GM_ADDR mxscale, GM_ADDR workspace,
    const SwigluMxQuantTilingData *__restrict tilingData, AscendC::TPipe *pipe)
{
#if (__NPU_ARCH__ == 3510)
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
#endif
    tiling_ = tilingData;
    pipe_ = pipe;
    blockIdx_ = GetBlockIdx();
    xGm_.SetGlobalBuffer((__gm__ T *)x);
    yGm_.SetGlobalBuffer((__gm__ uint8_t *)y);
    scaleGm_.SetGlobalBuffer((__gm__ uint8_t *)mxscale);
    if constexpr (isGroupIndex) {
        groupIndexGm_.SetGlobalBuffer((__gm__ T_IDX *)group_index);
        pipe_->InitBuffer(reduceSumBuffer_, ONE_BLOCK_UB);
    }
    // 开始get tilingData
    dimM_ = tiling_->inputDim1;
    int64_t batch = 1;
    if constexpr (!isLast) {
        batch = tiling_->inputDim0;
    }
    dimN_ = tiling_->inputDim2;
    dim2N_ = dimN_ * CONST_2;
    if constexpr (isLast) {
        swigluMode_ = tiling_->swigluMode;
        clampLimit_ = tiling_->clampLimit;
        gluBias_ = tiling_->gluBias;
        gluAlpha_ = tiling_->gluAlpha;
    }

    int64_t dimNBlockNum = tiling_->dimNBlockNum;
    realCoreNum_ = tiling_->usedCoreNum;
    factorDim0Size_ = tiling_->maxBasicNumUbDim1;
    factorDim1Size_ = tiling_->maxBasicNumUbDim2;

    // Initialize pipe buffers
    int32_t factorSize = factorDim0Size_ * factorDim1Size_;
    pipe_->InitBuffer(inQuex_, CONST_2, factorSize * X_ONCE_NUM * sizeof(T));
    if constexpr (IsSame<U, fp4x2_e2m1_t>::value || IsSame<U, fp4x2_e1m2_t>::value) {
        pipe_->InitBuffer(outQuey_, CONST_2, factorSize * QUANT_ONCE_NUM_FP4 * sizeof(uint8_t));
    } else {
        pipe_->InitBuffer(outQuey_, CONST_2, (factorSize * QUANT_ONCE_NUM) * sizeof(uint8_t));
    }
    int32_t scaleUbSize = factorSize * SCALE_ONCE_NUM;
    scaleUbSize = ((scaleUbSize + CONST_64 - 1) / CONST_64) * CONST_64;
    pipe_->InitBuffer(outQueScale_, CONST_2, scaleUbSize);

    // Initialize tmp buffers according to new UB allocation scheme
    // group_reserve: 32 bytes (fixed size, pre-reserved for future inputs)
    pipe_->InitBuffer(swigluBuffer_, factorSize * QUANT_ONCE_NUM * sizeof(T));
    int32_t maxExpUbSize = factorSize * SCALE_ONCE_NUM * sizeof(uint16_t);
    maxExpUbSize = ((maxExpUbSize + ONE_BLOCK_UB - 1) / ONE_BLOCK_UB) * ONE_BLOCK_UB;
    // reciprocal_scale: 8 × sizeof(half) = 16 bytes (8 quantization blocks, stored as FP16)
    pipe_->InitBuffer(maxExpBuffer_, maxExpUbSize);

    // max_exp: 8 × sizeof(int16_t) = 16 bytes (max exponent for OCP algorithm)
    pipe_->InitBuffer(halfScaleBuffer_, maxExpUbSize);

    activateLeft_ = tiling_->activateLeft;

    scaleAlg_ = tiling_->scaleAlg;
    // 3D core grid (B=1 always for last_last)
    int64_t mCorePerB = tiling_->mCorePerB;
    int64_t nCoreNum = tiling_->nCoreNum;
    int64_t bCoreNum = 1;
    if constexpr ((!isLast) && (!isGroupIndex)) {
        bCoreNum = tiling_->bCoreNum;
    }
    int64_t dimMNow = dimM_;
    if constexpr (isGroupIndex) {
        int64_t groupTotalRows = 0;
        InitGroupIndex(groupTotalRows);
        int64_t bmCores = realCoreNum_;
        if (groupTotalRows < bmCores)
            bmCores = groupTotalRows;
        nCoreNum = 1;
        int64_t nCore = realCoreNum_ / bmCores;
        if (nCore > 1) {
            nCoreNum = nCore;
            if (dimNBlockNum < nCoreNum)
                nCoreNum = dimNBlockNum;
        }
        realCoreNum_ = bmCores * nCoreNum;
        mCorePerB = bmCores;
        dimMNow = groupTotalRows;
    }
    int64_t nIdx = 0;
    int64_t mIdx = 0;
    if constexpr (!isLast) {
        int64_t bmCores = bCoreNum * mCorePerB;
        int64_t bmIdx = blockIdx_ / nCoreNum;
        nIdx = blockIdx_ % nCoreNum;
        int64_t bIdx = bmIdx / mCorePerB;
        mIdx = bmIdx % mCorePerB;
        // B split: divide batch dimension among bCoreNum cores
        int64_t bHeadCores = batch % bCoreNum;
        int64_t bBase = batch / bCoreNum;
        bStart_ = (bIdx < bHeadCores) ? bIdx * (bBase + 1) : bHeadCores * (bBase + 1) + (bIdx - bHeadCores) * bBase;
        bEnd_ = bStart_ + ((bIdx < bHeadCores) ? bBase + 1 : bBase);
    } else {
        nIdx = blockIdx_ % nCoreNum;
        mIdx = blockIdx_ / nCoreNum;
    }
    int64_t mHeadCores = dimMNow % mCorePerB;
    int64_t mBase = dimMNow / mCorePerB;
    mStart_ = (mIdx < mHeadCores) ? mIdx * (mBase + 1) : mHeadCores * (mBase + 1) + (mIdx - mHeadCores) * mBase;
    int64_t mRows = (mIdx < mHeadCores) ? mBase + 1 : mBase;

    loopTimesPerBatch_ = Ops::Base::CeilDiv(mRows, factorDim0Size_);
    tailPerBatch_ = mRows - (loopTimesPerBatch_ - 1) * factorDim0Size_;
    int64_t nHeadCores = dimNBlockNum % nCoreNum;
    int64_t blockPerNCore = dimNBlockNum / nCoreNum;
    int64_t nFrontCore = blockPerNCore + 1;
    nStart_ = (nIdx < nHeadCores) ? nIdx * nFrontCore : nHeadCores * nFrontCore + (nIdx - nHeadCores) * blockPerNCore;
    int64_t loopPerCoreN = (nIdx < nHeadCores) ? nFrontCore : blockPerNCore;
    loopTimesN_ = Ops::Base::CeilDiv(loopPerCoreN, factorDim1Size_);
    if (nIdx < nCoreNum - 1) {
        tailN_ = loopPerCoreN * 256 - (loopTimesN_ - 1) * factorDim1Size_ * 256;
    } else {
        tailN_ = dimN_ - nStart_ * 256 - (loopTimesN_ - 1) * factorDim1Size_ * 256;
    }

    outputScaleRowBytes_ = ((dimN_ + 64 - 1) / 64) * 2;
    if constexpr (IsSame<U, fp4x2_e2m1_t>::value) {
        f4Emax_ = FP4_E2M1_BF16_MAX_EXP;
    }
    if constexpr (IsSame<U, fp4x2_e1m2_t>::value) {
        f4Emax_ = FP4_E1M2_MAX_EXP;
    }
    if constexpr (IsSame<U, fp8_e4m3fn_t>::value) {
        f8Emax_ = FP8_E4M3_MAX_EXP;
        dtypeMax_ = FP8_E4M3_MAX;
    }
    if constexpr (IsSame<U, fp8_e5m2_t>::value) {
        f8Emax_ = FP8_E5M2_MAX_EXP;
        dtypeMax_ = FP8_E5M2_MAX;
    }
}

template <typename T, typename U, typename T_IDX, bool isGroupIndex, RoundMode roundMode, bool isLast>
__aicore__ inline void SwigluMxQuantAxisLast<T, U, T_IDX, isGroupIndex, roundMode, isLast>::Process()
{
    if (blockIdx_ >= realCoreNum_) {
        return;
    }
    int64_t dim1Size = factorDim1Size_ * QUANT_ONCE_NUM;
    int64_t dim1AlignSize = ((tailN_ + CONST_64 - 1) / CONST_64) * CONST_64;
    if constexpr (isLast) {
        for (int64_t mGroup = 0; mGroup < loopTimesPerBatch_; mGroup++) {
            int64_t dim0Size = (mGroup == loopTimesPerBatch_ - 1) ? tailPerBatch_ : factorDim0Size_;
            int64_t rowOffset = mStart_ + mGroup * factorDim0Size_;
            for (int64_t nLoop = 0; nLoop < loopTimesN_; nLoop++) {
                int64_t colOffset = nStart_ + nLoop * factorDim1Size_;
                bool isTailDim1 = (nLoop == loopTimesN_ - 1);
                int64_t dim1SizeNow = isTailDim1 ? tailN_ : dim1Size;
                int64_t dim1AlignSizeNow = isTailDim1 ? dim1AlignSize : dim1Size;
                CopyIn(rowOffset, colOffset, dim0Size, dim1SizeNow);
                Compute(dim0Size, dim1SizeNow, dim1AlignSizeNow);
                CopyOut(rowOffset, colOffset, dim0Size, dim1SizeNow, dim1AlignSizeNow);
            }
        }
    } else {
        for (int64_t b = bStart_; b < bEnd_; b++) {
            for (int64_t mGroup = 0; mGroup < loopTimesPerBatch_; mGroup++) {
                int64_t dim0Size = (mGroup == loopTimesPerBatch_ - 1) ? tailPerBatch_ : factorDim0Size_;
                // Input: (B, 2M, N), output: (B, M, N)
                // act at (b, mStart_+mGroup*factorDim0Size_), gate at (b, mStart_+mGroup*factorDim0Size_ + M)
                int64_t rowOffset = b * dimM_ * CONST_2 + mStart_ + mGroup * factorDim0Size_;
                int64_t rowOffsetY = b * dimM_ + mStart_ + mGroup * factorDim0Size_;
                for (int64_t dim2LoopIdx = 0; dim2LoopIdx < loopTimesN_; dim2LoopIdx++) {
                    int64_t curDim2Size = (dim2LoopIdx == loopTimesN_ - 1) ? tailN_ : dim1Size;
                    int64_t curDim2Align = (dim2LoopIdx == loopTimesN_ - 1) ? dim1AlignSize : dim1Size;

                    int64_t colBlockStart = nStart_ + dim2LoopIdx * factorDim1Size_;
                    int64_t colOffset = colBlockStart * QUANT_ONCE_NUM;

                    CopyIn(rowOffset, colOffset, dim0Size, curDim2Size);
                    Compute(dim0Size, curDim2Size, curDim2Align);

                    CopyOut(rowOffsetY, colBlockStart, dim0Size, curDim2Size, curDim2Align);
                }
            }
        }
    }
}

template <typename T, typename U, typename T_IDX, bool isGroupIndex, RoundMode roundMode, bool isLast>
__aicore__ inline void SwigluMxQuantAxisLast<T, U, T_IDX, isGroupIndex, roundMode, isLast>::ComputeF8(
    __local_mem__ T *swigluUbAddr, __local_mem__ uint16_t *maxExpUbAddr, __local_mem__ uint16_t *halfScaleLocalAddr,
    int64_t dim0OnceSize, int64_t dim1AlignSize)
{
    if (scaleAlg_ == 0) {
        ComputeVfMaxExpVfLast<T>(swigluUbAddr, maxExpUbAddr, dim0OnceSize, dim1AlignSize);
        LocalTensor<uint16_t> mxScaleLocal = outQueScale_.AllocTensor<uint16_t>();
        auto mxScaleLocalAddr = (__ubuf__ uint16_t *)mxScaleLocal.GetPhyAddr();
        ComputeScaleLast<T>(f8Emax_, maxExpUbAddr, mxScaleLocalAddr, halfScaleLocalAddr, dim0OnceSize, dim1AlignSize);
        outQueScale_.EnQue(mxScaleLocal);
    } else {
        ComputeVfMaxExpVfBLASLast<T>(swigluUbAddr, maxExpUbAddr, dim0OnceSize, dim1AlignSize);
        LocalTensor<uint16_t> mxScaleLocal = outQueScale_.AllocTensor<uint16_t>();
        auto mxScaleLocalAddr = (__ubuf__ uint16_t *)mxScaleLocal.GetPhyAddr();
        ComputeScaleBLASLast<T>(dtypeMax_, maxExpUbAddr, mxScaleLocalAddr, halfScaleLocalAddr, dim0OnceSize,
            dim1AlignSize);
        outQueScale_.EnQue(mxScaleLocal);
    }
}

template <typename T, typename U, typename T_IDX, bool isGroupIndex, RoundMode roundMode, bool isLast>
__aicore__ inline void SwigluMxQuantAxisLast<T, U, T_IDX, isGroupIndex, roundMode, isLast>::Compute(
    int64_t dim0OnceSize, int64_t dim1OnceSize, int64_t dim1AlignSize)
{
    LocalTensor<T> xlocal = inQuex_.DeQue<T>();
    auto x1UbAddr = (__ubuf__ T *)xlocal.GetPhyAddr();
    auto x2UbAddr = (__ubuf__ T *)xlocal[factorDim0Size_ * factorDim1Size_ * QUANT_ONCE_NUM].GetPhyAddr();
    LocalTensor<T> swigluUb = swigluBuffer_.Get<T>();
    auto swigluUbAddr = (__ubuf__ T *)swigluUb.GetPhyAddr();
    if constexpr (isLast) {
        if (swigluMode_ == 0 && activateLeft_ == 0) {
            x1UbAddr = (__ubuf__ T *)xlocal[factorDim0Size_ * factorDim1Size_ * QUANT_ONCE_NUM].GetPhyAddr();
            x2UbAddr = (__ubuf__ T *)xlocal.GetPhyAddr();
        }
        if (swigluMode_ == 1) {
            x1UbAddr = (__ubuf__ T *)xlocal.GetPhyAddr();
            x2UbAddr = (__ubuf__ T *)xlocal[CONST_64].GetPhyAddr();
        }
    } else {
        if (activateLeft_ == 0) {
            x1UbAddr = (__ubuf__ T *)xlocal[factorDim0Size_ * factorDim1Size_ * QUANT_ONCE_NUM].GetPhyAddr();
            x2UbAddr = (__ubuf__ T *)xlocal.GetPhyAddr();
        }
    }
    if constexpr (isLast) {
        if (swigluMode_ == 0) {
            ComputeVfSwigluV1<T>(x1UbAddr, x2UbAddr, swigluUbAddr, dim0OnceSize, dim1OnceSize, dim1AlignSize);
        } else {
            ComputeVfSwigluV2(x1UbAddr, x2UbAddr, swigluUbAddr, dim0OnceSize, dim1OnceSize, dim1AlignSize);
        }
    } else {
        ComputeVfSwigluV1<T>(x1UbAddr, x2UbAddr, swigluUbAddr, dim0OnceSize, dim1OnceSize, dim1AlignSize);
    }
    inQuex_.FreeTensor(xlocal);
    LocalTensor<uint16_t> maxExpUb = maxExpBuffer_.Get<uint16_t>();
    auto maxExpUbAddr = (__ubuf__ uint16_t *)maxExpUb.GetPhyAddr();
    LocalTensor<uint16_t> halfScaleLocal = halfScaleBuffer_.Get<uint16_t>();
    auto halfScaleLocalAddr = reinterpret_cast<__ubuf__ uint16_t *>(halfScaleLocal.GetPhyAddr());
    if constexpr (IsSame<U, fp4x2_e2m1_t>::value || IsSame<U, fp4x2_e1m2_t>::value) {
        ComputeVfMaxExpVfLast<T>(swigluUbAddr, maxExpUbAddr, dim0OnceSize, dim1AlignSize);
        LocalTensor<uint16_t> mxScaleLocal = outQueScale_.AllocTensor<uint16_t>();
        auto mxScaleLocalAddr = (__ubuf__ uint16_t *)mxScaleLocal.GetPhyAddr();
        ComputeScaleLast<T>(f4Emax_, maxExpUbAddr, mxScaleLocalAddr, halfScaleLocalAddr, dim0OnceSize, dim1AlignSize);
        outQueScale_.EnQue(mxScaleLocal);
    } else {
        ComputeF8(swigluUbAddr, maxExpUbAddr, halfScaleLocalAddr, dim0OnceSize, dim1AlignSize);
    }
    LocalTensor<int8_t> outLocal = outQuey_.AllocTensor<int8_t>();
    auto outLocalAddr = (__ubuf__ int8_t *)outLocal.GetPhyAddr();
    if constexpr (IsSame<U, fp4x2_e2m1_t>::value || IsSame<U, fp4x2_e1m2_t>::value) {
        if constexpr (roundMode == RoundMode::CAST_RINT) {
            ComputeDataF4Last<T, U, RoundMode::CAST_TRUNC, RoundMode::CAST_RINT>(swigluUbAddr, halfScaleLocalAddr,
                outLocalAddr, dim0OnceSize, dim1AlignSize);
        }
        if constexpr (roundMode == RoundMode::CAST_ROUND) {
            ComputeDataF4Last<T, U, RoundMode::CAST_TRUNC, RoundMode::CAST_ROUND>(swigluUbAddr, halfScaleLocalAddr,
                outLocalAddr, dim0OnceSize, dim1AlignSize);
        }
        if constexpr (roundMode == RoundMode::CAST_FLOOR) {
            ComputeDataF4Last<T, U, RoundMode::CAST_FLOOR, RoundMode::CAST_FLOOR>(swigluUbAddr, halfScaleLocalAddr,
                outLocalAddr, dim0OnceSize, dim1AlignSize);
        }
    } else {
        ComputeDataF8Last<T, U>(swigluUbAddr, halfScaleLocalAddr, outLocalAddr, dim0OnceSize, dim1AlignSize);
    }
    outQuey_.EnQue(outLocal);
}

template <typename T, typename U, typename T_IDX, bool isGroupIndex, RoundMode roundMode, bool isLast>
__aicore__ inline void SwigluMxQuantAxisLast<T, U, T_IDX, isGroupIndex, roundMode, isLast>::ComputeVfSwigluV2(
    __local_mem__ T *x1UbAddr, __local_mem__ T *x2UbAddr, __local_mem__ T *swigluUbAddr, int64_t dim0OnceSize,
    int64_t dim1OnceSize, int64_t dim1AlignSize)
{
    // 在计算swiglu时需把pad 0做了
    uint32_t alignDim1In = dim1OnceSize * CONST_2;
    uint32_t alignDim1Out = dim1AlignSize;
    uint16_t dim1VfTimes = alignDim1In / VF_LEN_T;
    uint16_t dim0VfTimes = static_cast<uint16_t>(dim0OnceSize);
    uint32_t dim1Tail = alignDim1In % VF_LEN_T; // 17
    uint16_t dim1TailTimes = 0;
    uint16_t dim1Tail2 = 0;
    uint32_t mask1Num = 0;
    uint32_t mask2Num = 0;
    uint32_t mask3Num = 0;
    T numZero = 0;
    float negScalarOne = -1.0f;
    float clampLimit = clampLimit_;
    float negClampLimit = -clampLimit_;
    float negAplha = -gluAlpha_;
    float scalarOne = 1.0f;
    float gluBias = gluBias_;
    auto x1UbAddr1 = x1UbAddr;
    auto x2UbAddr1 = x2UbAddr;
    auto swigluUbAddr1 = swigluUbAddr;
    auto swigluUbAddr2 = swigluUbAddr;
    if (dim1Tail > 0) {
        alignDim1In = ((alignDim1In + ONE_BLOCK_NUM - 1) / ONE_BLOCK_NUM) * ONE_BLOCK_NUM;
        dim1TailTimes = 1;
        mask1Num = dim1Tail / CONST_2; //  搬出只有一半的数
        uint32_t padNum = alignDim1Out - dim1VfTimes * VF_LEN_FP32;
        if (padNum <= VF_LEN_FP32) {
            mask2Num = padNum;
        } else {
            dim1Tail2 = 1;
            mask2Num = VF_LEN_FP32;
            mask3Num = padNum - VF_LEN_FP32;
        }
        x1UbAddr1 = x1UbAddr + dim1VfTimes * VF_LEN_T;
        x2UbAddr1 = x2UbAddr + dim1VfTimes * VF_LEN_T;
        swigluUbAddr1 = swigluUbAddr + dim1VfTimes * VF_LEN_FP32;
        swigluUbAddr2 = swigluUbAddr + dim1VfTimes * VF_LEN_FP32 + dim1TailTimes * VF_LEN_FP32;
    }
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<T> vregX1;
        AscendC::MicroAPI::RegTensor<T> vregX2;
        AscendC::MicroAPI::RegTensor<float> vregX1F;
        AscendC::MicroAPI::RegTensor<float> vregX2F;
        AscendC::MicroAPI::RegTensor<float> vregX1DeF;
        AscendC::MicroAPI::RegTensor<float> vregX2DeF;
        AscendC::MicroAPI::RegTensor<float> minsReg;
        AscendC::MicroAPI::RegTensor<float> mulsReg;
        AscendC::MicroAPI::RegTensor<float> expReg;
        AscendC::MicroAPI::RegTensor<float> addsReg;
        AscendC::MicroAPI::RegTensor<float> sigmoidReg;
        AscendC::MicroAPI::RegTensor<float> outFReg;
        AscendC::MicroAPI::RegTensor<T> outTReg;
        AscendC::MicroAPI::MaskReg mask = AscendC::MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg mask1 = AscendC::MicroAPI::UpdateMask<float>(mask1Num);
        AscendC::MicroAPI::MaskReg mask2 = AscendC::MicroAPI::UpdateMask<float>(mask2Num);
        AscendC::MicroAPI::MaskReg mask3 = AscendC::MicroAPI::UpdateMask<T>(mask3Num);
        for (uint16_t dim0vfLoopIdx = 0; dim0vfLoopIdx < dim0VfTimes; dim0vfLoopIdx++) {
            for (uint16_t dim1vfLoopIdx = 0; dim1vfLoopIdx < dim1VfTimes; dim1vfLoopIdx++) {
                AscendC::MicroAPI::AddrReg srcIdxOffset =
                    AscendC::MicroAPI::CreateAddrReg<T>(dim0vfLoopIdx, alignDim1In, dim1vfLoopIdx, 128);
                AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vregX1, x1UbAddr,
                    srcIdxOffset);
                AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vregX2, x2UbAddr,
                    srcIdxOffset);
                AscendC::MicroAPI::Cast<float, T, CAST_ZERO>(vregX1F, vregX1, mask);
                AscendC::MicroAPI::Cast<float, T, CAST_ZERO>(vregX2F, vregX2, mask);

                AscendC::MicroAPI::DeInterleave(vregX1DeF, vregX2DeF, vregX1F, vregX2F);
                AscendC::MicroAPI::Mins(minsReg, vregX1DeF, clampLimit, mask);
                AscendC::MicroAPI::Muls(mulsReg, minsReg, negAplha, mask);
                AscendC::MicroAPI::Exp(expReg, mulsReg, mask);
                AscendC::MicroAPI::Adds(addsReg, expReg, scalarOne, mask);
                AscendC::MicroAPI::Div(sigmoidReg, minsReg, addsReg, mask);

                AscendC::MicroAPI::Mins(vregX2DeF, vregX2DeF, clampLimit, mask);
                AscendC::MicroAPI::Maxs(vregX2DeF, vregX2DeF, negClampLimit, mask);
                AscendC::MicroAPI::Adds(vregX2DeF, vregX2DeF, gluBias, mask);

                AscendC::MicroAPI::Mul(outFReg, sigmoidReg, vregX2DeF, mask);


                AscendC::MicroAPI::Cast<T, float, CAST_FP32_TO_FP16_BF16>(outTReg, outFReg, mask);
                AscendC::MicroAPI::AddrReg outOffset =
                    AscendC::MicroAPI::CreateAddrReg<T>(dim0vfLoopIdx, alignDim1Out, dim1vfLoopIdx, 64);
                DataCopy<T, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(swigluUbAddr, outTReg, outOffset, mask);
            }
            AscendC::MicroAPI::AddrReg srcIdxOffset1 = AscendC::MicroAPI::CreateAddrReg<T>(dim0vfLoopIdx, alignDim1In);
            AscendC::MicroAPI::AddrReg outOffset1 = AscendC::MicroAPI::CreateAddrReg<T>(dim0vfLoopIdx, alignDim1Out);
            for (uint16_t aa = 0; aa < dim1TailTimes; aa++) {
                AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vregX1, x1UbAddr1,
                    srcIdxOffset1);
                AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vregX2, x2UbAddr1,
                    srcIdxOffset1);
                AscendC::MicroAPI::Cast<float, T, CAST_ZERO>(vregX1F, vregX1, mask);
                AscendC::MicroAPI::Cast<float, T, CAST_ZERO>(vregX2F, vregX2, mask);

                AscendC::MicroAPI::DeInterleave(vregX1DeF, vregX2DeF, vregX1F, vregX2F);
                AscendC::MicroAPI::Mins(minsReg, vregX1DeF, clampLimit, mask1);
                AscendC::MicroAPI::Muls(mulsReg, minsReg, negAplha, mask1);
                AscendC::MicroAPI::Exp(expReg, mulsReg, mask1);
                AscendC::MicroAPI::Adds(addsReg, expReg, scalarOne, mask1);
                AscendC::MicroAPI::Div(sigmoidReg, minsReg, addsReg, mask1);

                AscendC::MicroAPI::Mins(vregX2DeF, vregX2DeF, clampLimit, mask1);
                AscendC::MicroAPI::Maxs(vregX2DeF, vregX2DeF, negClampLimit, mask1);
                AscendC::MicroAPI::Adds(vregX2DeF, vregX2DeF, gluBias, mask1);

                AscendC::MicroAPI::Mul(outFReg, sigmoidReg, vregX2DeF, mask1);
                AscendC::MicroAPI::Cast<T, float, CAST_FP32_TO_FP16_BF16>(outTReg, outFReg, mask1);
                DataCopy<T, AscendC::MicroAPI::StoreDist::DIST_PACK_B32>(swigluUbAddr1, outTReg, outOffset1, mask2);
            }
            for (uint16_t cc = 0; cc < dim1Tail2; cc++) {
                Duplicate<T>(vregX1, numZero);
                DataCopy<T>(swigluUbAddr2, vregX1, outOffset1, mask3);
            }
        }
    }
}

template <typename T, typename U, typename T_IDX, bool isGroupIndex, RoundMode roundMode, bool isLast>
__aicore__ inline void SwigluMxQuantAxisLast<T, U, T_IDX, isGroupIndex, roundMode, isLast>::CopyIn(int64_t rowOffset,
    int64_t colBlockStart, int64_t dim0OnceSize, int64_t dim1OnceSize)
{
    LocalTensor<T> xlocal = inQuex_.AllocTensor<T>();
    DataCopyExtParams copyInParam = { 0, 0, 0, 0, 0 };
    DataCopyPadExtParams<T> copyPadParams = { false, 0, 0, 0 };
    if constexpr (isLast) {
        copyInParam.blockCount = dim0OnceSize;
        copyInParam.blockLen = dim1OnceSize * sizeof(T);
        if (swigluMode_ == 0) {
            int64_t offset = rowOffset * dim2N_ + colBlockStart * QUANT_ONCE_NUM;
            copyInParam.srcStride = (dim2N_ - dim1OnceSize) * sizeof(T);
            DataCopyPad(xlocal, xGm_[offset], copyInParam, copyPadParams);
            DataCopyPad(xlocal[factorDim0Size_ * factorDim1Size_ * QUANT_ONCE_NUM], xGm_[offset + dimN_], copyInParam,
                copyPadParams);
        } else {
            int64_t offset = rowOffset * dim2N_ + colBlockStart * X_ONCE_NUM;
            copyInParam.blockCount = dim0OnceSize;
            copyInParam.blockLen = dim1OnceSize * CONST_2 * sizeof(T);
            copyInParam.srcStride = (dim2N_ - dim1OnceSize * CONST_2) * sizeof(T);
            DataCopyPad(xlocal, xGm_[offset], copyInParam, copyPadParams);
        }
    } else {
        // -2轴激活无奇偶切分
        copyInParam.blockCount = static_cast<uint16_t>(dim0OnceSize);
        copyInParam.blockLen = static_cast<uint64_t>(dim1OnceSize * sizeof(T));

        copyInParam.srcStride = static_cast<uint16_t>((dimN_ - dim1OnceSize) * sizeof(T));
        copyInParam.dstStride = 0;
        DataCopyPad(xlocal, xGm_[rowOffset * dimN_ + colBlockStart], copyInParam, copyPadParams);
        DataCopyPad(xlocal[factorDim0Size_ * factorDim1Size_ * QUANT_ONCE_NUM],
            xGm_[(rowOffset + dimM_) * dimN_ + colBlockStart], copyInParam, copyPadParams);
    }
    inQuex_.EnQue(xlocal);
}

template <typename T, typename U, typename T_IDX, bool isGroupIndex, RoundMode roundMode, bool isLast>
__aicore__ inline void SwigluMxQuantAxisLast<T, U, T_IDX, isGroupIndex, roundMode, isLast>::CopyOut(int64_t rowOffset,
    int64_t colBlockStart, int64_t dim0OnceSize, int64_t dim1OnceSize, int64_t dim1OnceSizeAlgin)
{
    LocalTensor<uint8_t> mxScaleLocal = outQueScale_.DeQue<uint8_t>();
    LocalTensor<uint8_t> outLocal = outQuey_.DeQue<uint8_t>();
    DataCopyExtParams copyOutParamData = { 0, 0, 0, 0, 0 };
    copyOutParamData.blockCount = dim0OnceSize;
    int64_t offset = rowOffset * dimN_ + colBlockStart * 256;
    if constexpr (IsSame<U, fp4x2_e2m1_t>::value || IsSame<U, fp4x2_e1m2_t>::value) {
        copyOutParamData.blockLen = dim1OnceSize / CONST_2;
        copyOutParamData.srcStride = (dim1OnceSizeAlgin / CONST_2 - copyOutParamData.blockLen) / ONE_BLOCK_UB;
        copyOutParamData.dstStride = (dimN_ - dim1OnceSize) / CONST_2;
        offset = offset / CONST_2;
    } else {
        copyOutParamData.blockLen = dim1OnceSize;
        copyOutParamData.srcStride = (dim1OnceSizeAlgin - copyOutParamData.blockLen) / ONE_BLOCK_UB;
        copyOutParamData.dstStride = dimN_ - copyOutParamData.blockLen;
    }
    DataCopyPad(yGm_[offset], outLocal, copyOutParamData);

    DataCopyExtParams copyOutParamScale = { 0, 0, 0, 0, 0 };
    uint32_t usedFactorDim1 = dim1OnceSizeAlgin / ONE_BLOCK_UB;
    copyOutParamScale.blockCount = dim0OnceSize;
    copyOutParamScale.blockLen = usedFactorDim1;
    copyOutParamScale.srcStride = 0;
    copyOutParamScale.dstStride = outputScaleRowBytes_ - copyOutParamScale.blockLen;
    int64_t offsetScale = rowOffset * outputScaleRowBytes_ + colBlockStart * SCALE_ONCE_NUM;
    DataCopyPad<uint8_t, PaddingMode::Compact>(scaleGm_[offsetScale], mxScaleLocal, copyOutParamScale);
    outQuey_.FreeTensor(outLocal);
    outQueScale_.FreeTensor(mxScaleLocal);
}
} // namespace SwigluMxQuant
#endif // SWIGLU_MX_QUANT_AXIS_LAST_H
