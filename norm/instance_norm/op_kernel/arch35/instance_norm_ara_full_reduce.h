/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file instance_norm_ara_full_reduce.h
 * \brief
 */

#ifndef INSTANCE_NORM_ARA_FULL_REDUCE_REGBASE_H
#define INSTANCE_NORM_ARA_FULL_REDUCE_REGBASE_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "instance_norm_common.h"
namespace InstanceNormOps {
using namespace AscendC;
using namespace AscendC::MicroAPI;

using AscendC::MicroAPI::LoadDist;
using AscendC::MicroAPI::MaskMergeMode;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::StoreDist;

constexpr int64_t SCALE_COEF_TWO = 2;
constexpr int64_t SCALE_COEF_FOUR = 4;
constexpr int64_t SCALE_COEF_EIGHT = 8;

constexpr int64_t BIN_ADD_THRES = 8;
constexpr int64_t BIN_ADD_PADDING = 1;

constexpr uint16_t ROW_ZERO = 0;
constexpr uint16_t ROW_ONE = 1;
constexpr uint16_t ROW_TWO = 2;
constexpr uint16_t ROW_THREE = 3;
constexpr uint16_t ROW_FOUR = 4;
constexpr uint16_t ROW_FIVE = 5;
constexpr uint16_t ROW_SIX = 6;
constexpr uint16_t ROW_SEVEN = 7;

constexpr uint32_t ROW_TWO_OFFSET = 2;
constexpr uint32_t ROW_THREE_OFFSET = 3;
constexpr uint32_t ROW_FOUR_OFFSET = 4;
constexpr uint32_t ROW_FIVE_OFFSET = 5;
constexpr uint32_t ROW_SIX_OFFSET = 6;
constexpr uint32_t ROW_SEVEN_OFFSET = 7;

static constexpr int32_t BUFFER_NUM = 2;

static constexpr float RMS_POS_INF = 3.40282366920938E+38;
static constexpr float RMS_ZERO = 0.0f;

constexpr uint64_t VECTOR_REG_WIDTH = platform::GetVRegSize();

template <typename T_X, typename T_B, typename T_M>
class InstanceNormARAFullReduce {
public:
    __aicore__ inline InstanceNormARAFullReduce(const InstanceNormARAFullReduceTilingData* tilingDataIn)
    {
        tilingData_ = tilingDataIn;
        totalTiles_ = tilingData_->totalTiles;
        tilesPerCore_ = tilingData_->tilesPerCore;
        usedCoreNum_ = tilingData_->usedCoreNum;
        totalA1Len_ = tilingData_->totalA1Len;
        totalRLen_ = tilingData_->totalRLen;
        totalA0Len_ = tilingData_->totalA0Len;
        a0Outer_ = tilingData_->a0Outer;
        tileA0Len_ = tilingData_->tileA0Len;
        tileA0Tail_ = tilingData_->tileA0Tail;

        powerOfTwoForR_ = tilingData_->powerOfTwoForR;
        r1Quotient_ = tilingData_->binaryAddQuotient;
        binaryAddK_ = tilingData_->binaryAddK;
        binaryAddLast_ = tilingData_->binaryAddLast;

        epsilon_ = tilingData_->epsilon;
        if (powerOfTwoForR_ != 0) {
            nFactor_ = static_cast<float>(1) / static_cast<float>(powerOfTwoForR_);
            nCorrectionFactor_ = static_cast<float>(powerOfTwoForR_) / static_cast<float>(totalRLen_);
        } else {
            nFactor_ = 1;
            nCorrectionFactor_ = 1;
        }
    }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR beta, GM_ADDR y, GM_ADDR mean_out, GM_ADDR var_out)
    {
        xGm_.SetGlobalBuffer((__gm__ T_X*)x);
        betaGm_.SetGlobalBuffer((__gm__ T_B*)beta);
        gammaGm_.SetGlobalBuffer((__gm__ T_B*)gamma);
        yGm_.SetGlobalBuffer((__gm__ T_X*)y);
        meanGm_.SetGlobalBuffer((__gm__ T_M*)mean_out);
        varianceGm_.SetGlobalBuffer((__gm__ T_M*)var_out);

        int64_t xShapeLen = tileA0Len_ * totalRLen_;
        if constexpr (IsSameType<T_X, float>::value) {
            pipe_.InitBuffer(xQueue_, BUFFER_NUM, tileA0Len_ * (totalRLen_ + 1) * sizeof(T_X));
        } else {
            pipe_.InitBuffer(xQueue_, BUFFER_NUM, tileA0Len_ * (totalRLen_ + 1) * sizeof(T_X));
            pipe_.InitBuffer(castBuf_, tileA0Len_ * (totalRLen_ + 1) * sizeof(float));
        }
        pipe_.InitBuffer(gammaQueue_, 1, tileA0Len_ * sizeof(T_B));
        pipe_.InitBuffer(betaQueue_, 1, tileA0Len_ * sizeof(T_B));
        pipe_.InitBuffer(yQueue_, BUFFER_NUM, xShapeLen * sizeof(T_X));
        pipe_.InitBuffer(meanQueue_, BUFFER_NUM, tileA0Len_ * sizeof(float));
        pipe_.InitBuffer(varianceQueue_, BUFFER_NUM, tileA0Len_ * sizeof(float));
        pipe_.InitBuffer(rstdBuf_, tileA0Len_ * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        int64_t blockIdx = GetBlockIdx();
        int64_t beginIdx = blockIdx * tilesPerCore_;
        int64_t endIdx = beginIdx + tilesPerCore_;
        endIdx = endIdx > totalTiles_ ? totalTiles_ : endIdx;

        for (int64_t curIdx = beginIdx; curIdx < endIdx; curIdx++) {
            int64_t curA0Idx = curIdx % a0Outer_;
            int64_t curA1Idx = curIdx / a0Outer_;

            uint32_t curTileA0Len = curA0Idx == (a0Outer_ - 1) ? tileA0Tail_ : tileA0Len_;

            int64_t xOffset = curA1Idx * totalRLen_ * totalA0Len_ + // a1 offset
                              curA0Idx * tileA0Len_;                // a0 offset
            int64_t GammaBetaOffset = curA0Idx * tileA0Len_;        // a0 offset
            int64_t MeanVarOffset = curA1Idx * totalA0Len_ +        // a1 offset
                                    curA0Idx * tileA0Len_;          // a0 offset
            CopyInX(xOffset, totalRLen_, curTileA0Len);
            Compute(totalRLen_, curTileA0Len, GammaBetaOffset);
            CopyOutY(xOffset, totalRLen_, curTileA0Len);
            CopyOutMeanVar(MeanVarOffset, curTileA0Len);
        }
    }

    __aicore__ inline void Compute(int64_t curTileRLen, uint32_t curTileA0Len, int64_t GammaBetaOffset)
    {
        LocalTensor<T_X> xInUb = xQueue_.template DeQue<T_X>();
        LocalTensor<T_X> yOutUb = yQueue_.AllocTensor<T_X>();
        LocalTensor<float> meanOutUb = meanQueue_.AllocTensor<float>();
        LocalTensor<float> varianceOutUb = varianceQueue_.AllocTensor<float>();
        __local_mem__ T_X* xInUbAddr = (__local_mem__ T_X*)xInUb.GetPhyAddr();
        __local_mem__ float* xFp32InUbAddr = (__local_mem__ float*)xInUbAddr;
        __local_mem__ T_X* yOutUbAddr = (__local_mem__ T_X*)yOutUb.GetPhyAddr();
        __local_mem__ float* meanOutUbAddr = (__local_mem__ float*)meanOutUb.GetPhyAddr();
        __local_mem__ float* varianceOutUbAddr = (__local_mem__ float*)varianceOutUb.GetPhyAddr();
        if constexpr (IsSameType<T_X, half>::value || IsSameType<T_X, bfloat16_t>::value) {
            LocalTensor<float> castInUb = castBuf_.Get<float>();
            xFp32InUbAddr = (__local_mem__ float*)castInUb.GetPhyAddr();
            CastToFp32(xInUbAddr, xFp32InUbAddr, curTileA0Len);
            CalculateMean(xFp32InUbAddr, yOutUbAddr, meanOutUbAddr, curTileA0Len);
            CalculateVar(xFp32InUbAddr, yOutUbAddr, meanOutUbAddr, varianceOutUbAddr, curTileA0Len);
        } else {
            CalculateMean(xFp32InUbAddr, yOutUbAddr, meanOutUbAddr, curTileA0Len);
            CalculateVar(xFp32InUbAddr, yOutUbAddr, meanOutUbAddr, varianceOutUbAddr, curTileA0Len);
        }

        meanQueue_.EnQue(meanOutUb);
        varianceQueue_.EnQue(varianceOutUb);
        meanQueue_.DeQue<float>();
        varianceQueue_.DeQue<float>();

        LocalTensor<float> rstdUb = rstdBuf_.Get<float>();
        __local_mem__ float* rstdUbAddr = (__local_mem__ float*)rstdUb.GetPhyAddr();
        ComputeRstd(varianceOutUbAddr, rstdUbAddr, curTileA0Len, epsilon_);

        CopyInGammaBeta(GammaBetaOffset, curTileA0Len);
        LocalTensor<T_B> betaInUb = betaQueue_.template DeQue<T_B>();
        LocalTensor<T_B> gammaInUb = gammaQueue_.template DeQue<T_B>();
        __local_mem__ T_B* betaInUbAddr = (__local_mem__ T_B*)betaInUb.GetPhyAddr();
        __local_mem__ T_B* gammaInUbAddr = (__local_mem__ T_B*)gammaInUb.GetPhyAddr();

        CalculateY(xFp32InUbAddr, yOutUbAddr, betaInUbAddr, gammaInUbAddr, meanOutUbAddr, rstdUbAddr, curTileA0Len);
        if constexpr (!IsSameType<T_M, float>::value) {
            CastMeanRstd(meanOutUb, varianceOutUb, curTileA0Len);
        }
        xQueue_.FreeTensor(xInUb);
        betaQueue_.FreeTensor(betaInUb);
        gammaQueue_.FreeTensor(gammaInUb);
        yQueue_.EnQue(yOutUb);
        meanQueue_.EnQue(meanOutUb);
        varianceQueue_.EnQue(varianceOutUb);
    }

    __aicore__ inline uint32_t CEIL_DIV(uint32_t x, uint32_t y)
    {
        if (y > 0) {
            return (x + y - 1) / y;
        }
        return 0;
    }

    __aicore__ inline void CastToFp32(__local_mem__ T_X* xInUb, __local_mem__ float* castInUb, uint32_t curTileA0Len)
    {
        uint32_t tileA0Len = tileA0Len_;
        uint16_t curTileRLen = totalRLen_;
        uint16_t loopA0Num = ops::CeilDiv(curTileA0Len, VL_FP32);
        __VEC_SCOPE__
        {
            RegTensor<float> x;
            MaskReg pregMask;
            uint32_t sreg = curTileA0Len;

            for (uint16_t k = 0; k < loopA0Num; k++) {
                pregMask = UpdateMask<float>(sreg);
                // get max
                for (uint16_t i = 0; i < curTileRLen; i++) {
                    uint32_t xOffset = i * tileA0Len + k * VL_FP32;
                    // load x
                    LoadOneTensorForDtypeT(xInUb, x, pregMask, xOffset);
                    DataCopy(((__local_mem__ float*)castInUb) + xOffset, x, pregMask);
                }
            }
        }
    }

    __aicore__ inline void CopyInX(int64_t xGmOffset, int64_t curTileRLen, uint32_t curTileA0Len)
    {
        LocalTensor<T_X> xLocal = xQueue_.AllocTensor<T_X>();
        DataCopyPadExtParams<T_X> dataCopyPadExtParams;
        dataCopyPadExtParams.isPad = false;
        dataCopyPadExtParams.leftPadding = 0;
        dataCopyPadExtParams.rightPadding = 0;
        dataCopyPadExtParams.paddingValue = 0;
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = curTileRLen;
        copyInParams.blockLen = curTileA0Len * sizeof(T_X);
        copyInParams.srcStride = (totalA0Len_ - curTileA0Len) * sizeof(T_X);
        copyInParams.dstStride = (tileA0Len_ - curTileA0Len) * sizeof(T_X) / BLOCK_SIZE;
        DataCopyPad<T_X, PaddingMode::Normal>(xLocal, xGm_[xGmOffset], copyInParams, dataCopyPadExtParams);
        xQueue_.EnQue(xLocal);
    }

    __aicore__ inline void CopyInGammaBeta(int64_t offset, int64_t curTileA0Len)
    {
        LocalTensor<T_B> betaInUb = betaQueue_.AllocTensor<T_B>();
        LocalTensor<T_B> gammaInUb = gammaQueue_.AllocTensor<T_B>();
        DataCopyPadExtParams<T_B> dataCopyPadExtParams;
        dataCopyPadExtParams.isPad = false;
        dataCopyPadExtParams.leftPadding = 0;
        dataCopyPadExtParams.rightPadding = 0;
        dataCopyPadExtParams.paddingValue = 0;
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = 1;
        copyInParams.blockLen = curTileA0Len * sizeof(T_B);
        copyInParams.srcStride = 0;
        copyInParams.dstStride = 0;
        DataCopyPad(betaInUb, betaGm_[offset], copyInParams, dataCopyPadExtParams);
        DataCopyPad(gammaInUb, gammaGm_[offset], copyInParams, dataCopyPadExtParams);
        betaQueue_.EnQue(betaInUb);
        gammaQueue_.EnQue(gammaInUb);
    }

    template <typename T_SRC>
    __aicore__ inline void LoadTwoTensorForDtypeT(
        __local_mem__ T_SRC* src1, __local_mem__ T_SRC* src2, RegTensor<float>& dst1, RegTensor<float>& dst2,
        MaskReg& dst1Preg, MaskReg& dst2Preg, uint32_t src1Offset, uint32_t src2Offset)
    {
        if constexpr (IsSameType<T_SRC, half>::value) {
            RegTensor<half> xFp16Q;
            RegTensor<half> xFp16R;
            DataCopy<half, LoadDist::DIST_UNPACK_B16>(xFp16Q, ((__local_mem__ half*)(src1) + (src1Offset)));
            DataCopy<half, LoadDist::DIST_UNPACK_B16>(xFp16R, ((__local_mem__ half*)(src2) + (src2Offset)));
            Cast<float, half, castTraitB162B32>(dst1, xFp16Q, dst1Preg);
            Cast<float, half, castTraitB162B32>(dst2, xFp16R, dst2Preg);
        } else if constexpr (IsSameType<T_SRC, bfloat16_t>::value) {
            RegTensor<bfloat16_t> xFp16Q;
            RegTensor<bfloat16_t> xFp16R;
            DataCopy<bfloat16_t, LoadDist::DIST_UNPACK_B16>(xFp16Q, ((__local_mem__ bfloat16_t*)(src1) + (src1Offset)));
            DataCopy<bfloat16_t, LoadDist::DIST_UNPACK_B16>(xFp16R, ((__local_mem__ bfloat16_t*)(src2) + (src2Offset)));
            Cast<float, bfloat16_t, castTraitB162B32>(dst1, xFp16Q, dst1Preg);
            Cast<float, bfloat16_t, castTraitB162B32>(dst2, xFp16R, dst2Preg);
        } else {
            DataCopy(dst1, ((__local_mem__ float*)(src1) + (src1Offset)));
            DataCopy(dst2, ((__local_mem__ float*)(src2) + (src2Offset)));
        }
    }

    __aicore__ inline void LoadOneTensorForDtypeT(
        __local_mem__ T_X* input, RegTensor<float>& dst, MaskReg& preg, uint32_t offset)
    {
        if constexpr (IsSameType<T_X, half>::value) {
            RegTensor<half> xFp16;
            DataCopy<half, LoadDist::DIST_UNPACK_B16>(xFp16, ((__local_mem__ half*)(input) + (offset)));
            Cast<float, half, castTraitB162B32>(dst, xFp16, preg);
        } else if constexpr (IsSameType<T_X, bfloat16_t>::value) {
            RegTensor<bfloat16_t> xBf16;
            DataCopy<bfloat16_t, LoadDist::DIST_UNPACK_B16>(xBf16, ((__local_mem__ bfloat16_t*)(input) + (offset)));
            Cast<float, bfloat16_t, castTraitB162B32>(dst, xBf16, preg);
        } else {
            DataCopy(dst, ((__local_mem__ float*)(input) + (offset)));
        }
    }

    __aicore__ inline void TwoRowAddForMeanWithTail(
        RegTensor<float>& dst, __local_mem__ float* input, MaskReg& preg, uint32_t offset1, uint32_t offset2,
        uint32_t offset3, uint32_t offset4, RegTensor<float>& rem, RegTensor<float>& nextRow,
        RegTensor<float>& remNextRow, float n)
    {
        DataCopy(dst, ((__local_mem__ float*)(input) + (offset1)));
        DataCopy(rem, ((__local_mem__ float*)(input) + (offset2)));
        Muls(dst, dst, n, preg);
        Muls(rem, rem, n, preg);
        Add(dst, dst, rem, preg);
        DataCopy(nextRow, ((__local_mem__ float*)(input) + (offset3)));
        DataCopy(remNextRow, ((__local_mem__ float*)(input) + (offset4)));
        Muls(nextRow, nextRow, n, preg);
        Muls(remNextRow, remNextRow, n, preg);
        Add(nextRow, nextRow, remNextRow, preg);
        Add(dst, dst, nextRow, preg);
    }

    __aicore__ inline void TwoRowAddForMean(
        RegTensor<float>& dst, __local_mem__ float* input, MaskReg& preg, uint32_t offset1, uint32_t offset2,
        RegTensor<float>& nextRow, float n)
    {
        DataCopy(dst, ((__local_mem__ float*)(input) + (offset1)));
        DataCopy(nextRow, ((__local_mem__ float*)(input) + (offset2)));
        Muls(dst, dst, n, preg);
        Muls(nextRow, nextRow, n, preg);
        Add(dst, dst, nextRow, preg);
    }

    __aicore__ inline void TwoRowAddForVarWithTail(
        RegTensor<float>& dst, __local_mem__ float* input, MaskReg& preg, uint32_t offset1, uint32_t offset2,
        uint32_t offset3, uint32_t offset4, RegTensor<float>& mean, RegTensor<float>& rem, RegTensor<float>& nextRow,
        RegTensor<float>& remNextRow, float n)
    {
        DataCopy(dst, ((__local_mem__ float*)(input) + (offset1)));
        DataCopy(rem, ((__local_mem__ float*)(input) + (offset2)));
        Sub(dst, dst, mean, preg);
        Sub(rem, rem, mean, preg);
        Mul(dst, dst, dst, preg);
        Mul(rem, rem, rem, preg);
        Muls(dst, dst, n, preg);
        Muls(rem, rem, n, preg);
        Add(dst, dst, rem, preg);
        DataCopy(nextRow, ((__local_mem__ float*)(input) + (offset3)));
        DataCopy(remNextRow, ((__local_mem__ float*)(input) + (offset4)));
        Sub(nextRow, nextRow, mean, preg);
        Sub(remNextRow, remNextRow, mean, preg);
        Mul(nextRow, nextRow, nextRow, preg);
        Mul(remNextRow, remNextRow, remNextRow, preg);
        Muls(nextRow, nextRow, n, preg);
        Muls(remNextRow, remNextRow, n, preg);
        Add(nextRow, nextRow, remNextRow, preg);
        Add(dst, dst, nextRow, preg);
    }

    __aicore__ inline void TwoRowAddForVar(
        RegTensor<float>& dst, __local_mem__ float* input, MaskReg& preg, uint32_t offset1, uint32_t offset2,
        RegTensor<float>& mean, RegTensor<float>& nextRow, float n)
    {
        DataCopy(dst, ((__local_mem__ float*)(input) + (offset1)));
        DataCopy(nextRow, ((__local_mem__ float*)(input) + (offset2)));
        Sub(dst, dst, mean, preg);
        Sub(nextRow, nextRow, mean, preg);
        Mul(dst, dst, dst, preg);
        Mul(nextRow, nextRow, nextRow, preg);
        Muls(dst, dst, n, preg);
        Muls(nextRow, nextRow, n, preg);
        Add(dst, dst, nextRow, preg);
    }

    __aicore__ inline void BinaryAddVF(
        __local_mem__ float* binaryAddTmpAddr, uint32_t rLoopStride, uint32_t offset, uint16_t binaryAddKLoop,
        uint16_t binaryAddInnerLoop, uint16_t binaryAddLastLoop, MaskReg& pregLoop, RegTensor<float>& x1,
        RegTensor<float>& x2, RegTensor<float>& x3, RegTensor<float>& x4)
    {
        uint16_t curBinaryAddInnerLoop = binaryAddInnerLoop;
        for (uint16_t i = 0; i < binaryAddKLoop; i++) {
            curBinaryAddInnerLoop = curBinaryAddInnerLoop / ROW_FOUR_OFFSET;
            for (uint16_t j = 0; j < curBinaryAddInnerLoop; j++) {
                DataCopy(x1, ((__local_mem__ float*)binaryAddTmpAddr + (j * ROW_FOUR_OFFSET) * rLoopStride + offset));
                DataCopy(
                    x2, ((__local_mem__ float*)binaryAddTmpAddr + (j * ROW_FOUR_OFFSET + 1) * rLoopStride + offset));
                Add(x1, x1, x2, pregLoop);
                DataCopy(
                    x3, ((__local_mem__ float*)binaryAddTmpAddr + (j * ROW_FOUR_OFFSET + ROW_TWO_OFFSET) * rLoopStride +
                         offset));
                DataCopy(
                    x4, ((__local_mem__ float*)binaryAddTmpAddr +
                         (j * ROW_FOUR_OFFSET + ROW_THREE_OFFSET) * rLoopStride + offset));
                Add(x3, x3, x4, pregLoop);
                Add(x1, x1, x3, pregLoop);
                DataCopy(((__local_mem__ float*)binaryAddTmpAddr + j * rLoopStride + offset), x1, pregLoop);
            }
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        }
        for (uint16_t i = 0; i < binaryAddLastLoop; i++) {
            DataCopy(x1, ((__local_mem__ float*)binaryAddTmpAddr + offset));
            DataCopy(x2, ((__local_mem__ float*)binaryAddTmpAddr + rLoopStride + offset));
            Add(x1, x1, x2, pregLoop);
            DataCopy(((__local_mem__ float*)binaryAddTmpAddr + offset), x1, pregLoop);
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
        }
    }

    __aicore__ inline void CalculateMean(
        __local_mem__ float* xInUb, __local_mem__ T_X* yOutUb, __local_mem__ float* meanOutUbAddr, int64_t curTileA0Len)
    {
        if (totalRLen_ <= SCALE_COEF_TWO) {
            CalculateMeanRLessThan2(xInUb, meanOutUbAddr, curTileA0Len);
        } else if (totalRLen_ <= SCALE_COEF_FOUR) {
            CalculateMeanRLessThan4(xInUb, meanOutUbAddr, curTileA0Len);
        } else if (totalRLen_ <= SCALE_COEF_EIGHT) {
            CalculateMeanRLessThan8(xInUb, meanOutUbAddr, curTileA0Len);
        } else {
            CalculateMeanRMoreThan8(xInUb, yOutUb, meanOutUbAddr, curTileA0Len);
        }
    }

    __aicore__ inline void CalculateMeanRLessThan2(
        __local_mem__ float* xInUb, __local_mem__ float* meanOutUbAddr, int64_t curTileA0Len)
    {
        uint32_t rStride = tileA0Len_;
        uint16_t rLoopCount = totalRLen_;
        float n = static_cast<float>(1) / static_cast<float>(totalRLen_);
        uint16_t aLoopCount = CEIL_DIV(curTileA0Len, VL_FP32);

        __VEC_SCOPE__
        {
            RegTensor<float> xld;
            RegTensor<float> xmuls;
            RegTensor<float> sum;

            MaskReg pregLoop;
            uint32_t sreg0 = curTileA0Len;
            for (uint16_t k = 0; k < aLoopCount; k++) {
                pregLoop = UpdateMask<float>(sreg0);
                Duplicate(sum, 0.0, pregLoop);
                for (uint16_t i = 0; i < rLoopCount; i++) {
                    DataCopy(xld, ((__local_mem__ float*)xInUb + i * rStride + k * VL_FP32));
                    Muls(xmuls, xld, n, pregLoop);
                    Add(sum, sum, xmuls, pregLoop);
                }
                DataCopy(((__local_mem__ float*)meanOutUbAddr + k * VL_FP32), sum, pregLoop);
            }
        }
    }

    __aicore__ inline void CalculateMeanRLessThan4(
        __local_mem__ float* xInUb, __local_mem__ float* meanOutUbAddr, int64_t curTileA0Len)
    {
        uint32_t remainderOffset = SCALE_COEF_TWO * tileA0Len_;
        uint32_t aLength = tileA0Len_;
        uint32_t validNumInXUb = totalRLen_ * tileA0Len_;

        float n = nFactor_;
        float nCorrection = nCorrectionFactor_;

        uint16_t remainderTailCount = totalRLen_ - SCALE_COEF_TWO;
        uint32_t remainderTailOffset0 = (ROW_ZERO > remainderTailCount) ? validNumInXUb : remainderOffset;
        uint32_t remainderTailOffset1 = (ROW_ONE > remainderTailCount) ? validNumInXUb : remainderOffset + aLength;

        uint16_t aLoopCount = CEIL_DIV(curTileA0Len, VL_FP32);
        __VEC_SCOPE__
        {
            RegTensor<float> x1;
            RegTensor<float> nextRow;
            RegTensor<float> rem;
            RegTensor<float> remNextRow;

            MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
            RegTensor<float> zero;
            Duplicate(zero, 0.0, pregMain);

            MaskReg pregLoop;
            uint32_t sreg0 = curTileA0Len;
            for (uint16_t k = 0; k < aLoopCount; k++) {
                pregLoop = UpdateMask<float>(sreg0);
                uint32_t aLoopOffset = k * VL_FP32;
                DataCopy(((__local_mem__ float*)xInUb + validNumInXUb + aLoopOffset), zero, pregLoop);
                LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
                TwoRowAddForMeanWithTail(
                    x1, xInUb, pregLoop, aLoopOffset, remainderTailOffset0 + aLoopOffset, aLength + aLoopOffset,
                    remainderTailOffset1 + aLoopOffset, rem, nextRow, remNextRow, n);
                Muls(x1, x1, nCorrection, pregLoop);
                DataCopy(((__local_mem__ float*)meanOutUbAddr + aLoopOffset), x1, pregLoop);
            }
        }
    }

    __aicore__ inline void CalculateMeanRLessThan8(
        __local_mem__ float* xInUb, __local_mem__ float* meanOutUbAddr, int64_t curTileA0Len)
    {
        uint32_t remainderOffset = SCALE_COEF_FOUR * tileA0Len_;
        uint32_t aLength = tileA0Len_;
        uint32_t validNumInXUb = totalRLen_ * tileA0Len_;

        float n = nFactor_;
        float nCorrection = nCorrectionFactor_;

        uint16_t remainderTailCount = totalRLen_ - SCALE_COEF_FOUR;
        uint32_t remainderTailOffset0 = (ROW_ZERO > remainderTailCount) ? validNumInXUb : remainderOffset;
        uint32_t remainderTailOffset1 = (ROW_ONE > remainderTailCount) ? validNumInXUb : remainderOffset + aLength;
        uint32_t remainderTailOffset2 =
            (ROW_TWO > remainderTailCount) ? validNumInXUb : remainderOffset + ROW_TWO_OFFSET * aLength;
        uint32_t remainderTailOffset3 =
            (ROW_THREE > remainderTailCount) ? validNumInXUb : remainderOffset + ROW_THREE_OFFSET * aLength;

        uint16_t aLoopCount = CEIL_DIV(curTileA0Len, VL_FP32);
        __VEC_SCOPE__
        {
            RegTensor<float> x1;
            RegTensor<float> x2;
            RegTensor<float> nextRow;
            RegTensor<float> rem;
            RegTensor<float> remNextRow;

            MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
            RegTensor<float> zero;
            Duplicate(zero, 0.0, pregMain);

            MaskReg pregLoop;
            uint32_t sreg0 = curTileA0Len;
            for (uint16_t k = 0; k < aLoopCount; k++) {
                pregLoop = UpdateMask<float>(sreg0);
                uint32_t aLoopOffset = k * VL_FP32;
                DataCopy(((__local_mem__ float*)xInUb + validNumInXUb + aLoopOffset), zero, pregLoop);
                LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
                TwoRowAddForMeanWithTail(
                    x1, xInUb, pregLoop, aLoopOffset, remainderTailOffset0 + aLoopOffset, aLength + aLoopOffset,
                    remainderTailOffset1 + aLoopOffset, rem, nextRow, remNextRow, n);
                TwoRowAddForMeanWithTail(
                    x2, xInUb, pregLoop, ROW_TWO_OFFSET * aLength + aLoopOffset, remainderTailOffset2 + aLoopOffset,
                    ROW_THREE_OFFSET * aLength + aLoopOffset, remainderTailOffset3 + aLoopOffset, rem, nextRow,
                    remNextRow, n);
                Add(x1, x1, x2, pregLoop);
                Muls(x1, x1, nCorrection, pregLoop);
                DataCopy(((__local_mem__ float*)meanOutUbAddr + aLoopOffset), x1, pregLoop);
            }
        }
    }

    __aicore__ inline void CalculateMeanRMoreThan8(
        __local_mem__ float* xInUb, __local_mem__ T_X* yOutUb, __local_mem__ float* meanOutUbAddr, int64_t curTileA0Len)
    {
        uint16_t remainderLoopCount = (totalRLen_ - r1Quotient_ + SCALE_COEF_EIGHT - 1) / SCALE_COEF_EIGHT;
        uint16_t quotientLoopCount = (r1Quotient_ / SCALE_COEF_EIGHT) - remainderLoopCount;
        uint32_t remainderOffset = r1Quotient_ * tileA0Len_;

        uint32_t baseLineOffset = SCALE_COEF_EIGHT * tileA0Len_;
        uint32_t aLength = tileA0Len_;

        uint16_t binaryAddKLoop = binaryAddK_;
        uint16_t binaryAddInnerLoop = r1Quotient_ / SCALE_COEF_EIGHT;
        uint16_t binaryAddLastLoop = binaryAddLast_;

        uint32_t validNumInXUb = totalRLen_ * tileA0Len_;

        float n = nFactor_;
        float nCorrection = nCorrectionFactor_;

        uint16_t remainderTailCount = (totalRLen_ - r1Quotient_) - (remainderLoopCount - 1) * SCALE_COEF_EIGHT;
        uint32_t quotientTailOffset = (remainderLoopCount - 1) * baseLineOffset;
        uint32_t remainderTailOffset = quotientTailOffset + remainderOffset;
        uint32_t remainderTailOffset0 = (ROW_ZERO > remainderTailCount) ? validNumInXUb : remainderTailOffset;
        uint32_t remainderTailOffset1 = (ROW_ONE > remainderTailCount) ? validNumInXUb : remainderTailOffset + aLength;
        uint32_t remainderTailOffset2 =
            (ROW_TWO > remainderTailCount) ? validNumInXUb : remainderTailOffset + ROW_TWO_OFFSET * aLength;
        uint32_t remainderTailOffset3 =
            (ROW_THREE > remainderTailCount) ? validNumInXUb : remainderTailOffset + ROW_THREE_OFFSET * aLength;
        uint32_t remainderTailOffset4 =
            (ROW_FOUR > remainderTailCount) ? validNumInXUb : remainderTailOffset + ROW_FOUR_OFFSET * aLength;
        uint32_t remainderTailOffset5 =
            (ROW_FIVE > remainderTailCount) ? validNumInXUb : remainderTailOffset + ROW_FIVE_OFFSET * aLength;
        uint32_t remainderTailOffset6 =
            (ROW_SIX > remainderTailCount) ? validNumInXUb : remainderTailOffset + ROW_SIX_OFFSET * aLength;
        uint32_t remainderTailOffset7 =
            (ROW_SEVEN > remainderTailCount) ? validNumInXUb : remainderTailOffset + ROW_SEVEN_OFFSET * aLength;

        uint16_t aLoopCount = CEIL_DIV(curTileA0Len, VL_FP32);
        __VEC_SCOPE__
        {
            RegTensor<float> x1;
            RegTensor<float> x2;
            RegTensor<float> x3;
            RegTensor<float> x4;

            RegTensor<float> nextRow;
            RegTensor<float> rem;
            RegTensor<float> remNextRow;

            MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
            RegTensor<float> zero;
            Duplicate(zero, 0.0, pregMain);

            MaskReg pregLoop;
            uint32_t sreg0 = curTileA0Len;
            for (uint16_t k = 0; k < aLoopCount; k++) {
                pregLoop = UpdateMask<float>(sreg0);
                uint32_t aLoopOffset = k * VL_FP32;
                DataCopy(((__local_mem__ float*)xInUb + validNumInXUb + aLoopOffset), zero, pregLoop);
                LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
                // 前半部分与后半部分中，都为8行的部分
                for (uint16_t i = 0; i < static_cast<uint16_t>(remainderLoopCount - 1); i++) {
                    uint32_t quotOffset = i * baseLineOffset + aLoopOffset;
                    uint32_t remOffset = i * baseLineOffset + remainderOffset + aLoopOffset;
                    TwoRowAddForMeanWithTail(
                        x1, xInUb, pregLoop, quotOffset, remOffset, quotOffset + aLength, remOffset + aLength, rem,
                        nextRow, remNextRow, n);
                    TwoRowAddForMeanWithTail(
                        x2, xInUb, pregLoop, quotOffset + ROW_TWO_OFFSET * aLength,
                        remOffset + ROW_TWO_OFFSET * aLength, quotOffset + ROW_THREE_OFFSET * aLength,
                        remOffset + ROW_THREE_OFFSET * aLength, rem, nextRow, remNextRow, n);
                    Add(x1, x1, x2, pregLoop);
                    TwoRowAddForMeanWithTail(
                        x3, xInUb, pregLoop, quotOffset + ROW_FOUR_OFFSET * aLength,
                        remOffset + ROW_FOUR_OFFSET * aLength, quotOffset + ROW_FIVE_OFFSET * aLength,
                        remOffset + ROW_FIVE_OFFSET * aLength, rem, nextRow, remNextRow, n);
                    TwoRowAddForMeanWithTail(
                        x4, xInUb, pregLoop, quotOffset + ROW_SIX_OFFSET * aLength,
                        remOffset + ROW_SIX_OFFSET * aLength, quotOffset + ROW_SEVEN_OFFSET * aLength,
                        remOffset + ROW_SEVEN_OFFSET * aLength, rem, nextRow, remNextRow, n);
                    Add(x3, x3, x4, pregLoop);
                    Add(x1, x1, x3, pregLoop);
                    DataCopy(((__local_mem__ float*)yOutUb + i * aLength + aLoopOffset), x1, pregLoop);
                }
                // 前半部分为8行，后半部分可能不足8行
                {
                    TwoRowAddForMeanWithTail(
                        x1, xInUb, pregLoop, quotientTailOffset + aLoopOffset, remainderTailOffset0 + aLoopOffset,
                        quotientTailOffset + aLength + aLoopOffset, remainderTailOffset1 + aLoopOffset, rem, nextRow,
                        remNextRow, n);
                    TwoRowAddForMeanWithTail(
                        x2, xInUb, pregLoop, quotientTailOffset + ROW_TWO_OFFSET * aLength + aLoopOffset,
                        remainderTailOffset2 + aLoopOffset,
                        quotientTailOffset + ROW_THREE_OFFSET * aLength + aLoopOffset,
                        remainderTailOffset3 + aLoopOffset, rem, nextRow, remNextRow, n);
                    Add(x1, x1, x2, pregLoop);
                    TwoRowAddForMeanWithTail(
                        x3, xInUb, pregLoop, quotientTailOffset + ROW_FOUR_OFFSET * aLength + aLoopOffset,
                        remainderTailOffset4 + aLoopOffset,
                        quotientTailOffset + ROW_FIVE_OFFSET * aLength + aLoopOffset,
                        remainderTailOffset5 + aLoopOffset, rem, nextRow, remNextRow, n);
                    TwoRowAddForMeanWithTail(
                        x4, xInUb, pregLoop, quotientTailOffset + ROW_SIX_OFFSET * aLength + aLoopOffset,
                        remainderTailOffset6 + aLoopOffset,
                        quotientTailOffset + ROW_SEVEN_OFFSET * aLength + aLoopOffset,
                        remainderTailOffset7 + aLoopOffset, rem, nextRow, remNextRow, n);
                    Add(x3, x3, x4, pregLoop);
                    Add(x1, x1, x3, pregLoop);
                    DataCopy(
                        ((__local_mem__ float*)yOutUb + (remainderLoopCount - 1) * aLength + aLoopOffset), x1,
                        pregLoop);
                }
                // 剩余的前半部分，一次for循环，处理8行
                for (uint16_t i = 0; i < quotientLoopCount; i++) {
                    uint32_t baseOffset = (remainderLoopCount + i) * baseLineOffset + aLoopOffset;
                    TwoRowAddForMean(x1, xInUb, pregLoop, baseOffset, baseOffset + aLength, nextRow, n);
                    TwoRowAddForMean(
                        x2, xInUb, pregLoop, baseOffset + ROW_TWO_OFFSET * aLength,
                        baseOffset + ROW_THREE_OFFSET * aLength, nextRow, n);
                    Add(x1, x1, x2, pregLoop);
                    TwoRowAddForMean(
                        x3, xInUb, pregLoop, baseOffset + ROW_FOUR_OFFSET * aLength,
                        baseOffset + ROW_FIVE_OFFSET * aLength, nextRow, n);
                    TwoRowAddForMean(
                        x4, xInUb, pregLoop, baseOffset + ROW_SIX_OFFSET * aLength,
                        baseOffset + ROW_SEVEN_OFFSET * aLength, nextRow, n);
                    Add(x3, x3, x4, pregLoop);
                    Add(x1, x1, x3, pregLoop);
                    DataCopy(
                        ((__local_mem__ float*)yOutUb + (remainderLoopCount + i) * aLength + aLoopOffset), x1,
                        pregLoop);
                }
                LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
                BinaryAddVF(
                    (__local_mem__ float*)yOutUb, aLength, aLoopOffset, binaryAddKLoop, binaryAddInnerLoop,
                    binaryAddLastLoop, pregLoop, x1, x2, x3, x4);
                DataCopy(x1, ((__local_mem__ float*)yOutUb + aLoopOffset));
                Muls(x1, x1, nCorrection, pregLoop);
                DataCopy(((__local_mem__ float*)meanOutUbAddr + aLoopOffset), x1, pregLoop);
            }
        }
    }

    __aicore__ inline void CalculateVar(
        __local_mem__ float* xInUb, __local_mem__ T_X* yOutUb, __local_mem__ float* meanOutUbAddr,
        __local_mem__ float* varianceOutUbAddr, int64_t curTileA0Len)
    {
        if (totalRLen_ <= SCALE_COEF_TWO) {
            CalculateVarRLessThan2(xInUb, meanOutUbAddr, varianceOutUbAddr, curTileA0Len);
        } else if (totalRLen_ <= SCALE_COEF_FOUR) {
            CalculateVarRLessThan4(xInUb, meanOutUbAddr, varianceOutUbAddr, curTileA0Len);
        } else if (totalRLen_ <= SCALE_COEF_EIGHT) {
            CalculateVarRLessThan8(xInUb, meanOutUbAddr, varianceOutUbAddr, curTileA0Len);
        } else {
            CalculateVarRMoreThan8(xInUb, yOutUb, meanOutUbAddr, varianceOutUbAddr, curTileA0Len);
        }
    }

    __aicore__ inline void CalculateVarRLessThan2(
        __local_mem__ float* xInUb, __local_mem__ float* meanOutUbAddr, __local_mem__ float* varianceOutUbAddr,
        int64_t curTileA0Len)
    {
        uint32_t rStride = tileA0Len_;
        uint16_t rLoopCount = totalRLen_;
        float n = static_cast<float>(1) / static_cast<float>(totalRLen_);
        uint16_t aLoopCount = CEIL_DIV(curTileA0Len, VL_FP32);

        __VEC_SCOPE__
        {
            RegTensor<float> xld;
            RegTensor<float> xsub;
            RegTensor<float> xpow;
            RegTensor<float> xmuls;
            RegTensor<float> sum;
            RegTensor<float> mean;

            MaskReg pregLoop;
            uint32_t sreg0 = curTileA0Len;
            for (uint16_t k = 0; k < aLoopCount; k++) {
                pregLoop = UpdateMask<float>(sreg0);
                Duplicate(sum, 0.0, pregLoop);
                DataCopy(mean, ((__local_mem__ float*)meanOutUbAddr + k * VL_FP32));
                for (uint16_t i = 0; i < rLoopCount; i++) {
                    DataCopy(xld, ((__local_mem__ float*)xInUb + i * rStride + k * VL_FP32));
                    Sub(xsub, xld, mean, pregLoop);
                    Mul(xpow, xsub, xsub, pregLoop);
                    Muls(xmuls, xpow, n, pregLoop);
                    Add(sum, sum, xmuls, pregLoop);
                }
                DataCopy(((__local_mem__ float*)varianceOutUbAddr + k * VL_FP32), sum, pregLoop);
            }
        }
    }

    __aicore__ inline void CalculateVarRLessThan4(
        __local_mem__ float* xInUb, __local_mem__ float* meanOutUbAddr, __local_mem__ float* varianceOutUbAddr,
        int64_t curTileA0Len)
    {
        uint32_t remainderOffset = SCALE_COEF_TWO * tileA0Len_;
        uint32_t aLength = tileA0Len_;
        uint32_t validNumInXUb = totalRLen_ * tileA0Len_;

        float n = nFactor_;
        float nCorrection = nCorrectionFactor_;

        uint16_t remainderTailCount = totalRLen_ - SCALE_COEF_TWO;
        uint32_t remainderTailOffset0 = (ROW_ZERO > remainderTailCount) ? validNumInXUb : remainderOffset;
        uint32_t remainderTailOffset1 = (ROW_ONE > remainderTailCount) ? validNumInXUb : remainderOffset + aLength;

        uint16_t aLoopCount = CEIL_DIV(curTileA0Len, VL_FP32);
        __VEC_SCOPE__
        {
            RegTensor<float> x1;
            RegTensor<float> nextRow;
            RegTensor<float> rem;
            RegTensor<float> remNextRow;
            RegTensor<float> mean;

            MaskReg pregLoop;
            uint32_t sreg0 = curTileA0Len;
            for (uint16_t k = 0; k < aLoopCount; k++) {
                pregLoop = UpdateMask<float>(sreg0);
                uint32_t aLoopOffset = k * VL_FP32;
                DataCopy(mean, ((__local_mem__ float*)meanOutUbAddr + aLoopOffset));
                DataCopy(((__local_mem__ float*)xInUb + validNumInXUb + aLoopOffset), mean, pregLoop);
                LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
                TwoRowAddForVarWithTail(
                    x1, xInUb, pregLoop, aLoopOffset, remainderTailOffset0 + aLoopOffset, aLength + aLoopOffset,
                    remainderTailOffset1 + aLoopOffset, mean, rem, nextRow, remNextRow, n);
                Muls(x1, x1, nCorrection, pregLoop);
                DataCopy(((__local_mem__ float*)varianceOutUbAddr + aLoopOffset), x1, pregLoop);
            }
        }
    }

    __aicore__ inline void CalculateVarRLessThan8(
        __local_mem__ float* xInUb, __local_mem__ float* meanOutUbAddr, __local_mem__ float* varianceOutUbAddr,
        int64_t curTileA0Len)
    {
        uint32_t remainderOffset = SCALE_COEF_FOUR * tileA0Len_;
        uint32_t aLength = tileA0Len_;
        uint32_t validNumInXUb = totalRLen_ * tileA0Len_;

        float n = nFactor_;
        float nCorrection = nCorrectionFactor_;

        uint16_t remainderTailCount = totalRLen_ - SCALE_COEF_FOUR;
        uint32_t remainderTailOffset0 = (ROW_ZERO > remainderTailCount) ? validNumInXUb : remainderOffset;
        uint32_t remainderTailOffset1 = (ROW_ONE > remainderTailCount) ? validNumInXUb : remainderOffset + aLength;
        uint32_t remainderTailOffset2 =
            (ROW_TWO > remainderTailCount) ? validNumInXUb : remainderOffset + ROW_TWO_OFFSET * aLength;
        uint32_t remainderTailOffset3 =
            (ROW_THREE > remainderTailCount) ? validNumInXUb : remainderOffset + ROW_THREE_OFFSET * aLength;

        uint16_t aLoopCount = CEIL_DIV(curTileA0Len, VL_FP32);
        __VEC_SCOPE__
        {
            RegTensor<float> x1;
            RegTensor<float> x2;
            RegTensor<float> nextRow;
            RegTensor<float> rem;
            RegTensor<float> remNextRow;
            RegTensor<float> mean;

            MaskReg pregLoop;
            uint32_t sreg0 = curTileA0Len;
            for (uint16_t k = 0; k < aLoopCount; k++) {
                pregLoop = UpdateMask<float>(sreg0);
                uint32_t aLoopOffset = k * VL_FP32;
                DataCopy(mean, ((__local_mem__ float*)meanOutUbAddr + aLoopOffset));
                DataCopy(((__local_mem__ float*)xInUb + validNumInXUb + aLoopOffset), mean, pregLoop);
                LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
                TwoRowAddForVarWithTail(
                    x1, xInUb, pregLoop, aLoopOffset, remainderTailOffset0 + aLoopOffset, aLength + aLoopOffset,
                    remainderTailOffset1 + aLoopOffset, mean, rem, nextRow, remNextRow, n);
                TwoRowAddForVarWithTail(
                    x2, xInUb, pregLoop, ROW_TWO_OFFSET * aLength + aLoopOffset, remainderTailOffset2 + aLoopOffset,
                    ROW_THREE_OFFSET * aLength + aLoopOffset, remainderTailOffset3 + aLoopOffset, mean, rem, nextRow,
                    remNextRow, n);
                Add(x1, x1, x2, pregLoop);
                Muls(x1, x1, nCorrection, pregLoop);
                DataCopy(((__local_mem__ float*)varianceOutUbAddr + aLoopOffset), x1, pregLoop);
            }
        }
    }

    __aicore__ inline void CalculateVarRMoreThan8(
        __local_mem__ float* xInUb, __local_mem__ T_X* yOutUb, __local_mem__ float* meanOutUbAddr,
        __local_mem__ float* varianceOutUbAddr, int64_t curTileA0Len)
    {
        uint16_t remainderLoopCount = (totalRLen_ - r1Quotient_ + SCALE_COEF_EIGHT - 1) / SCALE_COEF_EIGHT;
        uint16_t quotientLoopCount = (r1Quotient_ / SCALE_COEF_EIGHT) - remainderLoopCount;
        uint32_t remainderOffset = r1Quotient_ * tileA0Len_;

        uint32_t baseLineOffset = SCALE_COEF_EIGHT * tileA0Len_;
        uint32_t aLength = tileA0Len_;

        uint16_t binaryAddKLoop = binaryAddK_;
        uint16_t binaryAddInnerLoop = r1Quotient_ / SCALE_COEF_EIGHT;
        uint16_t binaryAddLastLoop = binaryAddLast_;

        uint32_t validNumInXUb = totalRLen_ * tileA0Len_;

        float n = nFactor_;
        float nCorrection = nCorrectionFactor_;

        uint16_t remainderTailCount = (totalRLen_ - r1Quotient_) - (remainderLoopCount - 1) * SCALE_COEF_EIGHT;
        uint32_t quotientTailOffset = (remainderLoopCount - 1) * baseLineOffset;
        uint32_t remainderTailOffset = quotientTailOffset + remainderOffset;
        uint32_t remainderTailOffset0 = (ROW_ZERO > remainderTailCount) ? validNumInXUb : remainderTailOffset;
        uint32_t remainderTailOffset1 = (ROW_ONE > remainderTailCount) ? validNumInXUb : remainderTailOffset + aLength;
        uint32_t remainderTailOffset2 =
            (ROW_TWO > remainderTailCount) ? validNumInXUb : remainderTailOffset + ROW_TWO_OFFSET * aLength;
        uint32_t remainderTailOffset3 =
            (ROW_THREE > remainderTailCount) ? validNumInXUb : remainderTailOffset + ROW_THREE_OFFSET * aLength;
        uint32_t remainderTailOffset4 =
            (ROW_FOUR > remainderTailCount) ? validNumInXUb : remainderTailOffset + ROW_FOUR_OFFSET * aLength;
        uint32_t remainderTailOffset5 =
            (ROW_FIVE > remainderTailCount) ? validNumInXUb : remainderTailOffset + ROW_FIVE_OFFSET * aLength;
        uint32_t remainderTailOffset6 =
            (ROW_SIX > remainderTailCount) ? validNumInXUb : remainderTailOffset + ROW_SIX_OFFSET * aLength;
        uint32_t remainderTailOffset7 =
            (ROW_SEVEN > remainderTailCount) ? validNumInXUb : remainderTailOffset + ROW_SEVEN_OFFSET * aLength;

        uint16_t aLoopCount = CEIL_DIV(curTileA0Len, VL_FP32);
        __VEC_SCOPE__
        {
            RegTensor<float> x1;
            RegTensor<float> x2;
            RegTensor<float> x3;
            RegTensor<float> x4;

            RegTensor<float> nextRow;
            RegTensor<float> rem;
            RegTensor<float> remNextRow;
            RegTensor<float> mean;

            MaskReg pregLoop;
            uint32_t sreg0 = curTileA0Len;
            for (uint16_t k = 0; k < aLoopCount; k++) {
                pregLoop = UpdateMask<float>(sreg0);
                uint32_t aLoopOffset = k * VL_FP32;
                DataCopy(mean, ((__local_mem__ float*)meanOutUbAddr + aLoopOffset));
                DataCopy(((__local_mem__ float*)xInUb + validNumInXUb + aLoopOffset), mean, pregLoop);
                LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
                // 前半部分与后半部分中，都为8行的部分
                for (uint16_t i = 0; i < static_cast<uint16_t>(remainderLoopCount - 1); i++) {
                    uint32_t quotOffset = i * baseLineOffset + aLoopOffset;
                    uint32_t remOffset = i * baseLineOffset + remainderOffset + aLoopOffset;
                    TwoRowAddForVarWithTail(
                        x1, xInUb, pregLoop, quotOffset, remOffset, quotOffset + aLength, remOffset + aLength, mean,
                        rem, nextRow, remNextRow, n);
                    TwoRowAddForVarWithTail(
                        x2, xInUb, pregLoop, quotOffset + ROW_TWO_OFFSET * aLength,
                        remOffset + ROW_TWO_OFFSET * aLength, quotOffset + ROW_THREE_OFFSET * aLength,
                        remOffset + ROW_THREE_OFFSET * aLength, mean, rem, nextRow, remNextRow, n);
                    Add(x1, x1, x2, pregLoop);
                    TwoRowAddForVarWithTail(
                        x3, xInUb, pregLoop, quotOffset + ROW_FOUR_OFFSET * aLength,
                        remOffset + ROW_FOUR_OFFSET * aLength, quotOffset + ROW_FIVE_OFFSET * aLength,
                        remOffset + ROW_FIVE_OFFSET * aLength, mean, rem, nextRow, remNextRow, n);
                    TwoRowAddForVarWithTail(
                        x4, xInUb, pregLoop, quotOffset + ROW_SIX_OFFSET * aLength,
                        remOffset + ROW_SIX_OFFSET * aLength, quotOffset + ROW_SEVEN_OFFSET * aLength,
                        remOffset + ROW_SEVEN_OFFSET * aLength, mean, rem, nextRow, remNextRow, n);
                    Add(x3, x3, x4, pregLoop);
                    Add(x1, x1, x3, pregLoop);
                    DataCopy(((__local_mem__ float*)yOutUb + i * aLength + aLoopOffset), x1, pregLoop);
                }
                // 前半部分为8行，后半部分可能不足8行
                {
                    TwoRowAddForVarWithTail(
                        x1, xInUb, pregLoop, quotientTailOffset + aLoopOffset, remainderTailOffset0 + aLoopOffset,
                        quotientTailOffset + aLength + aLoopOffset, remainderTailOffset1 + aLoopOffset, mean, rem,
                        nextRow, remNextRow, n);
                    TwoRowAddForVarWithTail(
                        x2, xInUb, pregLoop, quotientTailOffset + ROW_TWO_OFFSET * aLength + aLoopOffset,
                        remainderTailOffset2 + aLoopOffset,
                        quotientTailOffset + ROW_THREE_OFFSET * aLength + aLoopOffset,
                        remainderTailOffset3 + aLoopOffset, mean, rem, nextRow, remNextRow, n);
                    Add(x1, x1, x2, pregLoop);
                    TwoRowAddForVarWithTail(
                        x3, xInUb, pregLoop, quotientTailOffset + ROW_FOUR_OFFSET * aLength + aLoopOffset,
                        remainderTailOffset4 + aLoopOffset,
                        quotientTailOffset + ROW_FIVE_OFFSET * aLength + aLoopOffset,
                        remainderTailOffset5 + aLoopOffset, mean, rem, nextRow, remNextRow, n);
                    TwoRowAddForVarWithTail(
                        x4, xInUb, pregLoop, quotientTailOffset + ROW_SIX_OFFSET * aLength + aLoopOffset,
                        remainderTailOffset6 + aLoopOffset,
                        quotientTailOffset + ROW_SEVEN_OFFSET * aLength + aLoopOffset,
                        remainderTailOffset7 + aLoopOffset, mean, rem, nextRow, remNextRow, n);
                    Add(x3, x3, x4, pregLoop);
                    Add(x1, x1, x3, pregLoop);
                    DataCopy(
                        ((__local_mem__ float*)yOutUb + (remainderLoopCount - 1) * aLength + aLoopOffset), x1,
                        pregLoop);
                }
                // 剩余的前半部分，一次for循环，处理8行
                for (uint16_t i = 0; i < quotientLoopCount; i++) {
                    uint32_t baseOffset = (remainderLoopCount + i) * baseLineOffset + aLoopOffset;
                    TwoRowAddForVar(x1, xInUb, pregLoop, baseOffset, baseOffset + aLength, mean, nextRow, n);
                    TwoRowAddForVar(
                        x2, xInUb, pregLoop, baseOffset + ROW_TWO_OFFSET * aLength,
                        baseOffset + ROW_THREE_OFFSET * aLength, mean, nextRow, n);
                    Add(x1, x1, x2, pregLoop);
                    TwoRowAddForVar(
                        x3, xInUb, pregLoop, baseOffset + ROW_FOUR_OFFSET * aLength,
                        baseOffset + ROW_FIVE_OFFSET * aLength, mean, nextRow, n);
                    TwoRowAddForVar(
                        x4, xInUb, pregLoop, baseOffset + ROW_SIX_OFFSET * aLength,
                        baseOffset + ROW_SEVEN_OFFSET * aLength, mean, nextRow, n);
                    Add(x3, x3, x4, pregLoop);
                    Add(x1, x1, x3, pregLoop);
                    DataCopy(
                        ((__local_mem__ float*)yOutUb + (remainderLoopCount + i) * aLength + aLoopOffset), x1,
                        pregLoop);
                }
                LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
                BinaryAddVF(
                    (__local_mem__ float*)yOutUb, aLength, aLoopOffset, binaryAddKLoop, binaryAddInnerLoop,
                    binaryAddLastLoop, pregLoop, x1, x2, x3, x4);
                DataCopy(x1, ((__local_mem__ float*)yOutUb + aLoopOffset));
                Muls(x1, x1, nCorrection, pregLoop);
                DataCopy(((__local_mem__ float*)varianceOutUbAddr + aLoopOffset), x1, pregLoop);
            }
        }
    }

    __aicore__ inline void ComputeRstd(
        __local_mem__ float* varianceOutUbAddr, __local_mem__ float* rstdUbAddr, uint64_t curTileA0Len, float epsilon)
    {
        uint16_t aLoop = static_cast<uint16_t>((curTileA0Len + VL_FP32 - 1) / VL_FP32);
        __VEC_SCOPE__
        {
            MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
            RegTensor<float> var;
            RegTensor<float> one;
            RegTensor<float> r;
            RegTensor<float> y;
            RegTensor<float> s;
            RegTensor<float> t;
            RegTensor<float> scalar1;
            RegTensor<float> scalarInf;
            RegTensor<float> scalarZero;
            RegTensor<float> t1;
            RegTensor<float> t2;
            RegTensor<float> t3;
            RegTensor<float> t4;
            RegTensor<float> rstd;

            MaskReg cmpRegZero;
            MaskReg cmpRegInf;
            MaskReg pregLoop;

            Duplicate(one, 1.0, pregMain);
            uint32_t sreg0 = static_cast<uint32_t>(curTileA0Len);
            for (uint16_t a = 0; a < aLoop; a++) {
                pregLoop = UpdateMask<float>(sreg0);
                Duplicate(scalar1, float(0.5), pregLoop);
                Duplicate(scalarInf, RMS_POS_INF, pregLoop);
                Duplicate(scalarZero, RMS_ZERO, pregLoop);
                Duplicate(t1, float(1.5), pregLoop);
                Duplicate(s, float(1.0), pregLoop);

                // rstd
                DataCopy(var, varianceOutUbAddr + a * VL_FP32);
                Adds(var, var, epsilon, pregLoop);
                Div(r, one, var, pregLoop);
                Sqrt(y, r, pregLoop);
                Muls(t, var, float(-0.5), pregLoop);
                Mul(t, t, y, pregLoop);                // -0.5 * x * y
                Mula(t1, t, y, pregLoop);              // 1.5 + (-0.5 * x * y) * y
                Mul(rstd, y, t1, pregLoop);            // y = y * (1.5 - 0.5 * x * y)
                Muls(t3, var, float(-1.0), pregLoop);  // -1 * x
                Mula(s, t3, r, pregLoop);              // 1 + (-1) * x * r
                Muls(t4, rstd, float(-1.0), pregLoop); // (-1) * y
                Mula(r, t4, rstd, pregLoop);           // r + (-1) * y * y
                Mula(s, var, r, pregLoop);             // s + x * t
                Mul(s, s, rstd, pregLoop);             // e * y
                Mula(rstd, s, scalar1, pregLoop);      // y + y * e * 0.5
                CompareScalar(cmpRegZero, var, RMS_POS_INF, pregLoop);
                Select(rstd, scalarZero, rstd, cmpRegZero);
                CompareScalar(cmpRegInf, var, RMS_ZERO, pregLoop);
                Select(rstd, scalarInf, rstd, cmpRegInf);
                DataCopy(rstdUbAddr + a * VL_FP32, rstd, pregLoop);
            }
        }
    }

    __aicore__ inline void CalculateY(
        __local_mem__ float* xInUb, __local_mem__ T_X* yOutUb, __local_mem__ T_B* betaInUb,
        __local_mem__ T_B* gammaInUb, __local_mem__ float* meanOutUbAddr, __local_mem__ float* rstdUbAddr,
        uint16_t curTileA0Len)
    {
        uint16_t rLoopCount = totalRLen_;
        uint16_t aLoopCount = CEIL_DIV(curTileA0Len, VL_FP32);
        uint32_t rStride = tileA0Len_;
        __VEC_SCOPE__
        {
            RegTensor<float> mean;

            RegTensor<float> x2;
            RegTensor<float> y2;
            RegTensor<float> rsqrtVar;

            RegTensor<float> beta;
            RegTensor<float> gamma;

            MaskReg pregLoop;
            uint32_t sreg2 = curTileA0Len;
            for (uint16_t k = 0; k < aLoopCount; k++) {
                pregLoop = UpdateMask<float>(sreg2);
                LoadTwoTensorForDtypeT(betaInUb, gammaInUb, beta, gamma, pregLoop, pregLoop, k * VL_FP32, k * VL_FP32);
                DataCopy(mean, ((__local_mem__ float*)meanOutUbAddr + k * VL_FP32));
                DataCopy(rsqrtVar, ((__local_mem__ float*)rstdUbAddr + k * VL_FP32));
                for (uint16_t r = 0; r < rLoopCount; r++) {
                    DataCopy(x2, ((__local_mem__ float*)xInUb + r * rStride + k * VL_FP32));
                    Sub(x2, x2, mean, pregLoop);
                    Mul(y2, x2, rsqrtVar, pregLoop);
                    Mul(y2, y2, gamma, pregLoop);
                    Add(y2, y2, beta, pregLoop);
                    if constexpr (IsSameType<T_X, half>::value) {
                        RegTensor<half> yFp16;
                        Cast<half, float, castTraitB322B16>(yFp16, y2, pregLoop);
                        DataCopy<half, StoreDist::DIST_PACK_B32>(
                            ((__local_mem__ half*)yOutUb + r * rStride + k * VL_FP32), yFp16, pregLoop);
                    } else if constexpr (IsSameType<T_X, bfloat16_t>::value) {
                        RegTensor<bfloat16_t> xBf16;
                        Cast<bfloat16_t, float, castTraitB322B16>(xBf16, y2, pregLoop);
                        DataCopy<bfloat16_t, StoreDist::DIST_PACK_B32>(
                            ((__local_mem__ bfloat16_t*)yOutUb + r * rStride + k * VL_FP32), xBf16, pregLoop);
                    } else {
                        DataCopy(((__local_mem__ float*)yOutUb + r * rStride + k * VL_FP32), y2, pregLoop);
                    }
                }
            }
        }
    }
    __aicore__ inline void CastMeanRstd(
        LocalTensor<float> meanOutUb, LocalTensor<float> varianceOutUb, uint64_t curTileA0Len)
    {
        __local_mem__ float* meanInAddr = (__local_mem__ float*)meanOutUb.GetPhyAddr();
        __local_mem__ float* varianceInAddr = (__local_mem__ float*)varianceOutUb.GetPhyAddr();
        __local_mem__ T_M* meanOutAddr = (__local_mem__ T_M*)meanOutUb.GetPhyAddr();
        __local_mem__ T_M* varianceOutAddr = (__local_mem__ T_M*)varianceOutUb.GetPhyAddr();

        uint32_t castCount = static_cast<uint32_t>(curTileA0Len);
        uint16_t castLoops = static_cast<uint32_t>((castCount + VL_FP32 - 1) / VL_FP32);
        __VEC_SCOPE__
        {
            RegTensor<float> input_mean;
            RegTensor<float> input_rstd;
            RegTensor<T_M> output_mean;
            RegTensor<T_M> output_rstd;
            MicroAPI::MaskReg pregLoop;
            for (uint16_t i = 0; i < castLoops; i++) {
                pregLoop = MicroAPI::UpdateMask<float>(castCount);
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(input_mean, meanInAddr + VL_FP32 * i);
                MicroAPI::DataCopy<float, MicroAPI::LoadDist::DIST_NORM>(input_rstd, varianceInAddr + VL_FP32 * i);
                Cast<T_M, float, castTraitB322B16>(output_mean, input_mean, pregLoop);
                Cast<T_M, float, castTraitB322B16>(output_rstd, input_rstd, pregLoop);
                DataCopy<T_M, StoreDist::DIST_PACK_B32>(
                    ((__local_mem__ T_M*)meanOutAddr + i * VL_MEAN), output_mean, pregLoop);
                DataCopy<T_M, StoreDist::DIST_PACK_B32>(
                    ((__local_mem__ T_M*)varianceOutAddr + i * VL_MEAN), output_rstd, pregLoop);
            }
        }
    }

    __aicore__ inline void CopyOutMeanVar(int64_t offset, int64_t curTileA0Len)
    {
        if constexpr (!IsSameType<T_M, float>::value) {
            // float to bfloat16 or float16, input continue and output each repeat have only half value
            LocalTensor<float> MeanInUb = meanQueue_.template DeQue<float>();
            LocalTensor<float> varianceUb = varianceQueue_.template DeQue<float>();
            // VL_FP32
            uint32_t castDmaCount = static_cast<uint32_t>(curTileA0Len);
            uint32_t castDmaLoops = static_cast<uint32_t>(castDmaCount / VL_FP32);
            if (castDmaLoops > 0) {
                DataCopyExtParams copyInParams;
                copyInParams.blockCount = castDmaLoops;
                copyInParams.blockLen = VL_FP32 * sizeof(T_M);
                copyInParams.srcStride = (VECTOR_REG_WIDTH - VL_FP32 * sizeof(T_M)) / BLOCK_SIZE;
                copyInParams.dstStride = 0;
                DataCopyPad(meanGm_[offset], MeanInUb.ReinterpretCast<T_M>(), copyInParams);
                DataCopyPad(varianceGm_[offset], varianceUb.ReinterpretCast<T_M>(), copyInParams);
            }

            // tail
            uint32_t tailSize = static_cast<uint32_t>(castDmaCount % VL_FP32);
            if (tailSize > 0) {
                DataCopyExtParams copyInParamsTail;
                copyInParamsTail.blockCount = 1;
                copyInParamsTail.blockLen = tailSize * sizeof(T_M);
                copyInParamsTail.srcStride = 0;
                copyInParamsTail.dstStride = 0;
                DataCopyPad(
                    meanGm_[offset + castDmaLoops * VL_FP32], MeanInUb[castDmaLoops * VL_FP32].ReinterpretCast<T_M>(),
                    copyInParamsTail);
                DataCopyPad(
                    varianceGm_[offset + castDmaLoops * VL_FP32],
                    varianceUb[castDmaLoops * VL_FP32].ReinterpretCast<T_M>(), copyInParamsTail);
            }
            meanQueue_.FreeTensor(MeanInUb);
            varianceQueue_.FreeTensor(varianceUb);
        } else {
            LocalTensor<float> meanOutUb = meanQueue_.template DeQue<float>();
            LocalTensor<float> VarOutUb = varianceQueue_.template DeQue<float>();
            DataCopyExtParams copyInParams;
            copyInParams.blockCount = 1;
            copyInParams.blockLen = curTileA0Len * sizeof(float);
            copyInParams.srcStride = 0;
            copyInParams.dstStride = 0;
            DataCopyPad(meanGm_[offset], meanOutUb, copyInParams);
            DataCopyPad(varianceGm_[offset], VarOutUb, copyInParams);
            meanQueue_.FreeTensor(meanOutUb);
            varianceQueue_.FreeTensor(VarOutUb);
        }
    }

    __aicore__ inline void CopyOutY(int64_t yGmOffset, int64_t curTileRLen, uint32_t curTileA0Len)
    {
        LocalTensor<T_X> y = yQueue_.DeQue<T_X>();
        DataCopyExtParams copyInParams;
        copyInParams.blockCount = curTileRLen;
        copyInParams.blockLen = curTileA0Len * sizeof(T_X);
        copyInParams.srcStride = (tileA0Len_ - curTileA0Len) * sizeof(T_X) / BLOCK_SIZE;
        copyInParams.dstStride = (totalA0Len_ - curTileA0Len) * sizeof(T_X);
        DataCopyPad<T_X, PaddingMode::Normal>(yGm_[yGmOffset], y, copyInParams);
        yQueue_.FreeTensor(y);
    }

    // global memory address
    GlobalTensor<T_X> xGm_;
    GlobalTensor<T_B> betaGm_;
    GlobalTensor<T_B> gammaGm_;
    GlobalTensor<T_X> yGm_;
    GlobalTensor<T_M> meanGm_;
    GlobalTensor<T_M> varianceGm_;

    // variable
    const InstanceNormARAFullReduceTilingData* tilingData_;

    int64_t totalTiles_;
    int64_t tilesPerCore_;
    int64_t usedCoreNum_;
    int64_t totalA1Len_;
    int64_t totalRLen_;
    int64_t totalA0Len_;
    int64_t tileA0Len_;
    int64_t tileA0Tail_;
    int64_t powerOfTwoForR_;
    int64_t a0Outer_;
    int64_t r1Quotient_;
    int64_t binaryAddK_;
    int64_t binaryAddLast_;

    float epsilon_;

    float nFactor_;
    float nCorrectionFactor_;

    // ascendc variable
    TPipe pipe_;
    TQue<QuePosition::VECIN, 1> xQueue_, gammaQueue_, betaQueue_;
    TQue<QuePosition::VECOUT, 1> yQueue_, meanQueue_, varianceQueue_;

    TBuf<TPosition::VECCALC> castBuf_;
    TBuf<TPosition::VECCALC> rstdBuf_;

    constexpr static uint32_t VL_MEAN = platform::GetVRegSize() / sizeof(T_M);
};
} // namespace InstanceNormOps
#endif
