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
 * \file add_rms_norm_cast_regbase_high_performance.h
 * \brief AddRmsNormCast regbase high_performance template.
 */
#ifndef OPS_BUILT_IN_TBE_IMPL_ASCENDC_ADD_RMS_NORM_CAST_REGBASE_HIGH_PERFORMANCE_H
#define OPS_BUILT_IN_TBE_IMPL_ASCENDC_ADD_RMS_NORM_CAST_REGBASE_HIGH_PERFORMANCE_H
#include "add_rms_norm_cast_regbase_common.h"
#include "../../rms_norm/rms_norm_base.h"
#include "../inc/platform.h"

namespace AddRmsNormCast {

constexpr int32_t NUM_ONE = 1;
constexpr int32_t NUM_TWO = 2;

using RmsNorm::DataCopyCustom;
using RmsNorm::DataCopyImpl;

using AscendC::MicroAPI::LoadDist;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::RegTensor;

constexpr static uint32_t VL_FP32 = platform::GetVRegSize() / sizeof(float);
constexpr static uint32_t BLK_B32 = BLOCK_SIZE / sizeof(float);

template <typename T>
__aicore__ inline T Min(T a, T b)
{
    return a > b ? b : a;
}
template <typename T>
class KernelAddRmsNormCastRegBaseHighPerformance {
public:
    __aicore__ inline KernelAddRmsNormCastRegBaseHighPerformance(TPipe* pipe)
    {
        pPipe = pipe;
    }
    __aicore__ inline void Init(
        GM_ADDR x1, GM_ADDR x2, GM_ADDR gamma, GM_ADDR y1, GM_ADDR y2, GM_ADDR rstd, GM_ADDR x, GM_ADDR workspace,
        const AddRmsNormCastRegbaseTilingData* tiling)
    {
        ASSERT(GetBlockNum() != 0 && "Block dim can not be zero!");
        numRow = tiling->numM;
        numCol = tiling->baseN;
        blockFactor = tiling->mPerCore;
        binAddQuotient = tiling->powerSplit;
        rowFactor = tiling->baseM;
        epsilon = tiling->epsilon;
        numColAlign = tiling->baseNDtypeAlign;
        avgFactor = tiling->avgFactor;
        rowWork = (GetBlockIdx() < GetBlockNum() - 1) ? blockFactor : numRow - (GetBlockNum() - 1) * blockFactor;
        numColB32Align = CeilAlign(numCol, AddRmsNormCast::B32_BLOCK_NUM);
        uint64_t rstdUbSizeAlignSize = CeilAlign(rowFactor, static_cast<uint64_t>(VL_FP32)) * sizeof(float);
        uint16_t binaryAddQuotientLoop = (binAddQuotient + VL_FP32 - 1) / VL_FP32;
        uint32_t binaryAddBufLen =
            (binaryAddQuotientLoop + BLK_B32 - 1) / BLK_B32 * BLK_B32 * sizeof(float) * rowFactor;

        xGm1.SetGlobalBuffer((__gm__ T*)x1 + GetBlockIdx() * blockFactor * numCol, rowWork * numCol);
        xGm2.SetGlobalBuffer((__gm__ T*)x2 + GetBlockIdx() * blockFactor * numCol, rowWork * numCol);
        gammaGm.SetGlobalBuffer((__gm__ T*)gamma, numCol);
        y1Gm.SetGlobalBuffer((__gm__ float*)y1 + GetBlockIdx() * blockFactor * numCol, rowWork * numCol);
        y2Gm.SetGlobalBuffer((__gm__ T*)y2 + GetBlockIdx() * blockFactor * numCol, rowWork * numCol);
        rstdGm.SetGlobalBuffer((__gm__ float*)rstd + GetBlockIdx() * blockFactor, blockFactor);
        xOutGm.SetGlobalBuffer((__gm__ T*)x + GetBlockIdx() * blockFactor * numCol, rowWork * numCol);

        pPipe->InitBuffer(inQueueX1, DOUBLE_BUFFER_NUM, numColAlign * sizeof(T) * rowFactor);
        pPipe->InitBuffer(inQueueX2, DOUBLE_BUFFER_NUM, numColAlign * sizeof(T) * rowFactor);
        pPipe->InitBuffer(inQueueGamma, 1, numColAlign * sizeof(T));
        pPipe->InitBuffer(outQueueY1, DOUBLE_BUFFER_NUM, numColAlign * sizeof(float) * rowFactor);
        pPipe->InitBuffer(outQueueY2, DOUBLE_BUFFER_NUM, numColAlign * sizeof(T) * rowFactor);
        pPipe->InitBuffer(outQueueX, DOUBLE_BUFFER_NUM, numColAlign * sizeof(T) * rowFactor);
        pPipe->InitBuffer(outQueueRstd, DOUBLE_BUFFER_NUM, rstdUbSizeAlignSize);
        pPipe->InitBuffer(xReduceBuff, rstdUbSizeAlignSize);
        pPipe->InitBuffer(xFp32Buff, numColAlign * sizeof(float) * rowFactor);
        pPipe->InitBuffer(binaryAddBuf, binaryAddBufLen);
    }

    __aicore__ inline void Process()
    {
        CopyInGamma();
        LocalTensor<T> gammaLocal = inQueueGamma.DeQue<T>();
        uint32_t repeatTimes = CeilDiv(rowWork, rowFactor);
        for (uint32_t repeat = 0; repeat < repeatTimes; repeat++) {
            uint64_t offset = repeat * rowFactor * numCol;
            uint32_t curRows = Min(rowWork - repeat * rowFactor, rowFactor);
            Compute(repeat, gammaLocal, curRows, offset);
        }
        inQueueGamma.FreeTensor(gammaLocal);
    }

private:
    __aicore__ inline void Compute(
        uint32_t rowRepeat, LocalTensor<T> gammaLocal, uint32_t curRows, uint64_t offset)
    {
        CopyInXMutiMoveAlign(offset, numColAlign, curRows);
        LocalTensor<T> xLocal1 = inQueueX1.DeQue<T>();
        LocalTensor<T> xLocal2 = inQueueX2.DeQue<T>();
        LocalTensor<T> xOutLocal = outQueueX.AllocTensor<T>();
        LocalTensor<float> xFp32Local = xFp32Buff.Get<float>();

        CalculateXAdd(xLocal1, xLocal2, xOutLocal, xFp32Local, curRows, numColAlign);
        inQueueX1.FreeTensor(xLocal1);
        inQueueX2.FreeTensor(xLocal2);
        outQueueX.EnQue<T>(xOutLocal);
        CopyOutX(offset, curRows, numColAlign);

        LocalTensor<float> rstdLocal = outQueueRstd.AllocTensor<float>();
        LocalTensor<float> xReduceLocal = xReduceBuff.Get<float>();
        CalculateSquareReduceSum(xFp32Local, xReduceLocal, curRows, numColAlign, numCol);
        CalculateRstd(xReduceLocal, rstdLocal, curRows, avgFactor, epsilon);
        outQueueRstd.EnQue<float>(rstdLocal);

        rstdLocal = outQueueRstd.DeQue<float>();
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(1),
            static_cast<uint32_t>(curRows * sizeof(float)),
            static_cast<uint32_t>(0),
            static_cast<uint32_t>(0),
            0
        };
        DataCopyPad(rstdGm[rowRepeat * rowFactor], rstdLocal, copyParams);

        LocalTensor<float> y1Local = outQueueY1.AllocTensor<float>();
        LocalTensor<T> y2Local = outQueueY2.AllocTensor<T>();
        CalculateY1Y2(xFp32Local, gammaLocal, y1Local, y2Local, rstdLocal, curRows, numColAlign, numCol);
        outQueueRstd.FreeTensor(rstdLocal);
        outQueueY1.EnQue<float>(y1Local);
        outQueueY2.EnQue<T>(y2Local);
        CopyOutY1Y2(offset, curRows, numColAlign);
    }

    __aicore__ inline void CalculateXAdd(
        LocalTensor<T>& xLocal1, LocalTensor<T>& xLocal2, LocalTensor<T>& xOutLocal, LocalTensor<float>& xFp32Local,
        uint32_t curRows, uint32_t numColAlign)
    {
        __local_mem__ T* x1InUb = (__local_mem__ T*)xLocal1.GetPhyAddr();
        __local_mem__ T* x2InUb = (__local_mem__ T*)xLocal2.GetPhyAddr();
        __local_mem__ T* xOutInUb = (__local_mem__ T*)xOutLocal.GetPhyAddr();
        __local_mem__ float* xFp32Tmp = (__local_mem__ float*)xFp32Local.GetPhyAddr();

        uint32_t sreg = curRows * numColAlign;
        uint16_t loopCount = (sreg + VL_FP32 - 1) / VL_FP32;

        __VEC_SCOPE__
        {
            RegTensor<float> x1;
            RegTensor<float> x2;
            RegTensor<float> xSum;
            MaskReg pregLoop;
            for (uint16_t i = 0; i < loopCount; ++i) {
                uint32_t offset = i * VL_FP32;
                pregLoop = UpdateMask<float>(sreg);
                LoadTensorForDtypeTIn<T>(x1InUb, x1, pregLoop, offset);
                LoadTensorForDtypeTIn<T>(x2InUb, x2, pregLoop, offset);
                Add(xSum, x1, x2, pregLoop);
                StoreTensorForDtypeTOut<T>(xOutInUb, xSum, pregLoop, offset);
                DataCopy<float, StoreDist::DIST_NORM_B32>(xFp32Tmp + offset, xSum, pregLoop);
            }
        }
    }

    __aicore__ inline void CalculateY1Y2(
        LocalTensor<float>& xFp32Local, LocalTensor<T>& gammaLocal, LocalTensor<float>& y1Local, LocalTensor<T>& y2Local,
        LocalTensor<float>& rstdLocal, uint32_t curRows, uint32_t numColAlign, uint32_t reduceNum)
    {
        __local_mem__ float* xFp32Tmp = (__local_mem__ float*)xFp32Local.GetPhyAddr();
        __local_mem__ T* gammaInUb = (__local_mem__ T*)gammaLocal.GetPhyAddr();
        __local_mem__ float* y1InUb = (__local_mem__ float*)y1Local.GetPhyAddr();
        __local_mem__ T* y2InUb = (__local_mem__ T*)y2Local.GetPhyAddr();
        __local_mem__ float* rstdInUb = (__local_mem__ float*)rstdLocal.GetPhyAddr();

        uint16_t loopRows = static_cast<uint16_t>(curRows);
        uint16_t loopCols = static_cast<uint16_t>((reduceNum + VL_FP32 - 1) / VL_FP32);
        uint16_t loopRowsFold = loopRows / 2;
        uint16_t loopRowsHasLast = loopRows % 2;

        __VEC_SCOPE__ {
            RegTensor<float> x1Reg;
            RegTensor<float> x2Reg;
            RegTensor<float> gammaReg;
            RegTensor<float> rstd1Reg;
            RegTensor<float> rstd2Reg;
            RegTensor<float> mul1Reg;
            RegTensor<float> mul1UnrollReg;
            RegTensor<float> mul2Reg;
            RegTensor<float> mul2UnrollReg;

            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
            for (uint16_t i = 0; i < loopRowsFold; ++i) {
                uint32_t sregCount = reduceNum;
                DataCopy<float, LoadDist::DIST_BRC_B32>(rstd1Reg, rstdInUb + NUM_TWO * i);
                DataCopy<float, LoadDist::DIST_BRC_B32>(rstd2Reg, rstdInUb + (NUM_TWO * i + NUM_ONE));
                for (uint16_t r = 0; r < loopCols; ++r) {
                    uint32_t offset1 = (NUM_TWO * i) * numColAlign + r * VL_FP32;
                    uint32_t offset2 = (NUM_TWO * i + NUM_ONE) * numColAlign + r * VL_FP32;
                    MaskReg regCurLoop = UpdateMask<float>(sregCount);
                    LoadTensorForDtypeTIn<float>(xFp32Tmp, x1Reg, regCurLoop, offset1);
                    LoadTensorForDtypeTIn<float>(xFp32Tmp, x2Reg, regCurLoop, offset2);
                    Mul(mul1Reg, x1Reg, rstd1Reg, regCurLoop);
                    Mul(mul1UnrollReg, x2Reg, rstd2Reg, regCurLoop);
                    LoadTensorForDtypeTIn<T>(gammaInUb, gammaReg, regCurLoop, r * VL_FP32);
                    Mul(mul2Reg, mul1Reg, gammaReg, regCurLoop);
                    Mul(mul2UnrollReg, mul1UnrollReg, gammaReg, regCurLoop);
                    StoreTensorForDtypeTOut<float>(y1InUb, mul2Reg, regCurLoop, offset1);
                    StoreTensorForDtypeTOut<float>(y1InUb, mul2UnrollReg, regCurLoop, offset2);
                    StoreTensorForDtypeTOut<T>(y2InUb, mul2Reg, regCurLoop, offset1);
                    StoreTensorForDtypeTOut<T>(y2InUb, mul2UnrollReg, regCurLoop, offset2);
                }
            }
            for (uint16_t i = 0; i < loopRowsHasLast; ++i) {
                uint32_t sregCount = reduceNum;
                DataCopy<float, LoadDist::DIST_BRC_B32>(rstd1Reg, rstdInUb + NUM_TWO * loopRowsFold);
                for (uint16_t r = 0; r < loopCols; ++r) {
                    uint32_t offset = (NUM_TWO * loopRowsFold) * numColAlign + r * VL_FP32;
                    MaskReg regCurLoop = UpdateMask<float>(sregCount);
                    LoadTensorForDtypeTIn<float>(xFp32Tmp, x1Reg, regCurLoop, offset);
                    Mul(mul1Reg, x1Reg, rstd1Reg, regCurLoop);
                    LoadTensorForDtypeTIn<T>(gammaInUb, gammaReg, regCurLoop, r * VL_FP32);
                    Mul(mul2Reg, mul1Reg, gammaReg, regCurLoop);
                    StoreTensorForDtypeTOut<float>(y1InUb, mul2Reg, regCurLoop, offset);
                    StoreTensorForDtypeTOut<T>(y2InUb, mul2Reg, regCurLoop, offset);
                }
            }
        }
    }

    __aicore__ inline void CalculateSquareReduceSum(
        LocalTensor<float>& xFp32Local, LocalTensor<float>& xReduceLocal, uint32_t curRows, uint32_t numColAlign,
        uint32_t reduceNum)
    {
        LocalTensor<float> binaryAddBuffTmp = binaryAddBuf.Get<float>();
        __local_mem__ float* xReduceUb = (__local_mem__ float*)xReduceLocal.GetPhyAddr();
        __local_mem__ float* tmpUb = (__local_mem__ float*)binaryAddBuffTmp.GetPhyAddr();
        __local_mem__ float* xFp32Tmp = (__local_mem__ float*)xFp32Local.GetPhyAddr();

        if (reduceNum <= VL_FP32) {
            CalculateSquareReduceSumLessThanVL(xFp32Tmp, xReduceUb, curRows, numColAlign, reduceNum);
        } else if (reduceNum <= VL_FP32 + VL_FP32) {
            CalculateSquareReduceSumLessThanTwoVL(xFp32Tmp, xReduceUb, curRows, numColAlign, reduceNum);
        } else if (reduceNum <= VL_FP32 * VL_FP32 * NUM_TWO) {
            CalculateSquareReduceSumCommon<NUM_ONE>(xFp32Tmp, xReduceUb, tmpUb, curRows, numColAlign, reduceNum);
        } else {
            CalculateSquareReduceSumCommon<NUM_TWO>(xFp32Tmp, xReduceUb, tmpUb, curRows, numColAlign, reduceNum);
        }
    }

    __aicore__ inline void CalculateSquareReduceSumLessThanVL(
        __local_mem__ float* xFp32Tmp, __local_mem__ float* xReduceUb, uint16_t curRows, uint32_t numColAlign,
        uint32_t reduceNum)
    {
        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> vMean;
            RegTensor<float> onesReg;

            uint32_t sreg0 = reduceNum;
            MaskReg pregLoop = UpdateMask<float>(sreg0);
            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
            Duplicate(onesReg, float(1.0), pregOne);

            for (uint16_t i = 0; i < curRows; i++) {
                LoadTensorForDtypeTIn<float>(xFp32Tmp, x, pregLoop, i * numColAlign);
                Mul(x, x, x, pregLoop);
                ReduceSum(vMean, x, pregLoop);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(xReduceUb + i, vMean, pregOne);
            }
        }
    }

    __aicore__ inline void CalculateSquareReduceSumLessThanTwoVL(
        __local_mem__ float* xFp32Tmp, __local_mem__ float* xReduceUb, uint16_t curRows, uint32_t numColAlign,
        uint32_t reduceNum)
    {
        uint32_t tailLen = reduceNum - VL_FP32;
        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> xFold;
            RegTensor<float> sumReg;
            RegTensor<float> vMean;
            RegTensor<float> onesReg;

            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
            MaskReg pregTail = UpdateMask<float>(tailLen);
            Duplicate(onesReg, float(1.0), pregOne);

            for (uint16_t i = 0; i < curRows; ++i) {
                LoadTensorForDtypeTIn<float>(xFp32Tmp, x, pregFull, i * numColAlign);
                LoadTensorForDtypeTIn<float>(xFp32Tmp + VL_FP32, xFold, pregTail, i * numColAlign);
                Mul(x, x, x, pregFull);
                Mul(xFold, xFold, xFold, pregTail);
                ShiftLefts((RegTensor<uint32_t> &)xFold, (RegTensor<uint32_t> &)xFold, static_cast<int16_t>(0), pregTail);
                Add(sumReg, x, xFold, pregFull);
                ReduceSum(vMean, sumReg, pregFull);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(xReduceUb + i, vMean, pregOne);
            }
        }
    }

    template <int32_t LAST_LOOP_NUMS>
    __aicore__ inline void CalculateSquareReduceSumCommon(
        __local_mem__ float* xFp32Tmp, __local_mem__ float* xReduceUb, __local_mem__ float* tmpUb, uint16_t curRows,
        uint32_t numColAlign, uint32_t reduceNum)
    {
        uint32_t binaryAddQuotient = binAddQuotient; 
        uint16_t binaryAddQuotientLoop = (binaryAddQuotient + VL_FP32 - 1) / VL_FP32; 

        uint32_t lastBinaryAddNum = binaryAddQuotient / VL_FP32; 
        uint32_t lastBinaryAddNumAlign = (binaryAddQuotientLoop + BLK_B32 - 1) / BLK_B32 * BLK_B32; 

        uint32_t binaryAddRemainder = reduceNum - binaryAddQuotient; 
        uint16_t binaryAddRemainderCeilLoop = (binaryAddRemainder + VL_FP32 - 1) / VL_FP32; 
        uint16_t binaryAddRemainderFloorLoop = binaryAddRemainder / VL_FP32; 
        __VEC_SCOPE__
        {
            RegTensor<float> x;
            RegTensor<float> xFold;
            RegTensor<float> sumReg;
            RegTensor<float> vMean;
            RegTensor<float> onesReg;

            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
            MaskReg pregLoop;
            Duplicate(onesReg, float(1.0), pregOne);

            for (uint16_t i = 0; i < curRows; ++i) {
                uint32_t baseOffset = i * numColAlign;
                for (uint16_t r = 0; r < binaryAddRemainderFloorLoop; ++r) {
                    uint32_t offset = r * VL_FP32 + baseOffset;
                    LoadTensorForDtypeTIn<float>(xFp32Tmp, x, pregFull, offset);
                    LoadTensorForDtypeTIn<float>(xFp32Tmp + binaryAddQuotient, xFold, pregFull, offset);
                    Mul(x, x, x, pregFull);
                    Mul(xFold, xFold, xFold, pregFull);
                    Add(sumReg, x, xFold, pregFull);
                    ReduceSum(vMean, sumReg, pregFull);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        tmpUb + static_cast<uint32_t>(i * lastBinaryAddNumAlign + r), vMean, pregOne);
                }
                uint32_t sregRemainder = binaryAddRemainder - binaryAddRemainderFloorLoop * VL_FP32;
                for (uint16_t r = 0;
                     r < static_cast<uint16_t>(binaryAddRemainderCeilLoop - binaryAddRemainderFloorLoop); r++) {
                    uint16_t offset = baseOffset;
                    pregLoop = UpdateMask<float>(sregRemainder);
                    LoadTensorForDtypeTIn<float>(xFp32Tmp + binaryAddRemainderFloorLoop * VL_FP32, x, pregFull, offset);
                    LoadTensorForDtypeTIn<float>(
                        xFp32Tmp + binaryAddRemainderFloorLoop * VL_FP32 + binaryAddQuotient, xFold, pregLoop, offset);
                    Mul(x, x, x, pregFull);
                    Mul(xFold, xFold, xFold, pregLoop);
                    ShiftLefts((RegTensor<uint32_t> &)xFold, (RegTensor<uint32_t> &)xFold, static_cast<int16_t>(0), pregLoop);
                    Add(sumReg, x, xFold, pregFull);
                    ReduceSum(vMean, sumReg, pregFull);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        tmpUb + static_cast<uint32_t>(i * lastBinaryAddNumAlign + binaryAddRemainderFloorLoop), vMean,
                        pregOne);
                }
                for (uint16_t r = 0; r < static_cast<uint16_t>(binaryAddQuotientLoop - binaryAddRemainderCeilLoop);
                     r++) {
                    uint32_t offset = r * VL_FP32 + baseOffset;
                    LoadTensorForDtypeTIn<float>(xFp32Tmp + binaryAddRemainderCeilLoop * VL_FP32, x, pregFull, offset);
                    Mul(x, x, x, pregFull);
                    ReduceSum(vMean, x, pregFull);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                        tmpUb + static_cast<uint32_t>(i * lastBinaryAddNumAlign + binaryAddRemainderCeilLoop + r),
                        vMean, pregOne);
                }
            }
            LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
            if constexpr (LAST_LOOP_NUMS == 1) {
                MaskReg pregLast = UpdateMask<float>(lastBinaryAddNum);
                for (uint16_t i = 0; i < curRows; ++i) {
                    DataCopy(x, tmpUb + static_cast<uint32_t>(i * lastBinaryAddNumAlign));
                    ReduceSum(vMean, x, pregLast);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(xReduceUb + i, vMean, pregOne);
                }
            } else if constexpr (LAST_LOOP_NUMS == 2) {
                lastBinaryAddNum -= VL_FP32;
                MaskReg pregLast = UpdateMask<float>(lastBinaryAddNum);
                for (uint16_t i = 0; i < curRows; ++i) {
                    DataCopy(x, tmpUb + static_cast<uint32_t>(i * lastBinaryAddNumAlign));
                    DataCopy(xFold, tmpUb + static_cast<uint32_t>(i * lastBinaryAddNumAlign + VL_FP32));
                    ShiftLefts((RegTensor<uint32_t> &)xFold, (RegTensor<uint32_t> &)xFold, static_cast<int16_t>(0), pregLast);
                    Add(sumReg, x, xFold, pregFull);
                    ReduceSum(vMean, sumReg, pregFull);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(xReduceUb + i, vMean, pregOne);
                }
            }
        }
    }

    __aicore__ inline void CalculateRstd(
        LocalTensor<float>& xReduceLocal, LocalTensor<float>& rstdLocal, uint32_t curRows, float avgFactor,
        float epsilon)
    {
        static constexpr float POS_INF = 3.40282366920938E+38;
        static constexpr float SCALAR1 = -0.5;
        static constexpr float SCALAR2 = 1.5;
        static constexpr float SCALAR3 = 0.5;
        static constexpr float SCALAR0 = -99.99;

        __local_mem__ float* rstdInUb = (__local_mem__ float*)rstdLocal.GetPhyAddr();
        __local_mem__ float* xReduceUb = (__local_mem__ float*)xReduceLocal.GetPhyAddr();
        uint16_t loopRows = static_cast<uint16_t>((curRows + VL_FP32 - 1) / VL_FP32);
        __VEC_SCOPE__
        {
            RegTensor<float> var;
            RegTensor<float> rstd;
            RegTensor<float> r;
            RegTensor<float> y;
            RegTensor<float> s;
            RegTensor<float> t;
            RegTensor<float> one;
            RegTensor<float> scalar1;
            RegTensor<float> t1;
            RegTensor<float> t2;
            RegTensor<float> t3;
            RegTensor<float> t4;
            RegTensor<float> scalarInf;
            RegTensor<float> scalarZero;
            MaskReg cmpRegZero;
            MaskReg cmpRegInf;
            MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
            MaskReg pregLoop;

            uint32_t sreg = static_cast<uint32_t>(curRows);
            for (uint16_t i = 0; i < loopRows; ++i) {
                pregLoop = UpdateMask<float>(sreg);
                Duplicate(scalarInf, POS_INF, pregLoop);
                Duplicate(scalarZero, float(0.0), pregLoop);
                Duplicate(one, float(1.0), pregLoop);
                Duplicate(scalar1, SCALAR3, pregLoop);
                Duplicate(t1, SCALAR2, pregLoop);
                Duplicate(s, float(1.0), pregLoop);
                // rstd
                DataCopy(var, xReduceUb + i * VL_FP32);
                Muls(var, var, avgFactor, pregLoop);
                Adds(var, var, epsilon, pregLoop);
                Maxs(var, var, SCALAR0, pregLoop);
                Div(r, one, var, pregLoop);
                Sqrt(y, r, pregLoop);
                Muls(t, var, SCALAR1, pregLoop);
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
                CompareScalar(cmpRegZero, var, POS_INF, pregLoop);
                Select(rstd, scalarZero, rstd, cmpRegZero);
                CompareScalar(cmpRegInf, var, float(0.0), pregLoop);
                Select(rstd, scalarInf, rstd, cmpRegInf);
                DataCopy(rstdInUb + i * VL_FP32, rstd, pregLoop);
            }
        }
    }

    __aicore__ inline void CopyInXMutiMoveAlign(uint64_t offset, uint32_t curCols, uint32_t curRows = 0)
    {
        LocalTensor<T> xLocal1 = inQueueX1.AllocTensor<T>();
        LocalTensor<T> xLocal2 = inQueueX2.AllocTensor<T>();
        DataCopyExtParams extParams{
            static_cast<uint16_t>(curRows),                                               // blockCount
            static_cast<uint32_t>(numCol * sizeof(T)),                                    // blockLen
            static_cast<uint32_t>(0),                                                     // srcStride
            static_cast<uint32_t>((numColAlign - curCols) * sizeof(T) / ALIGN_32_FACTOR), // dstStride
            0                                                                             // rsv
        };
        DataCopyPadExtParams<T> padParams{
            false,                   // isPad
            static_cast<uint8_t>(0), // leftPadding
            static_cast<uint8_t>(0), // rightPadding
            static_cast<T>(0.0)      // paddingValue
        };
        DataCopyPad(xLocal1, xGm1[offset], extParams, padParams);
        DataCopyPad(xLocal2, xGm2[offset], extParams, padParams);
        inQueueX1.EnQue(xLocal1);
        inQueueX2.EnQue(xLocal2);
    }

    __aicore__ inline void CopyInGamma()
    {
        LocalTensor<T> gammaLocal = inQueueGamma.AllocTensor<T>();
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(1),                  // blockCount
            static_cast<uint32_t>(numCol * sizeof(T)), // blockLen
            static_cast<uint32_t>(0),                  // srcStride
            static_cast<uint32_t>(0),                  // dstStride
            0                                          // rsv
        };
        DataCopyPadExtParams<T> padParams{
            false,                   // isPad
            static_cast<uint8_t>(0), // leftPadding
            static_cast<uint8_t>(0), // rightPadding
            static_cast<T>(0.0)      // paddingValue
        };
        DataCopyPad(gammaLocal, gammaGm, copyParams, padParams);
        inQueueGamma.EnQue(gammaLocal);
    }

    __aicore__ inline void CopyOutY1Y2(uint64_t offset, uint32_t curRows, uint32_t colAlign)
    {
        LocalTensor<float> y1Local = outQueueY1.DeQue<float>();
        LocalTensor<T> y2Local = outQueueY2.DeQue<T>();
        uint32_t srcStride1 = (numColAlign - numColB32Align) * sizeof(float) / ALIGN_32_FACTOR;
        uint32_t srcStride2 = (numColAlign - colAlign) * sizeof(T) / ALIGN_32_FACTOR;
        DataCopyExtParams copyParams1{
            static_cast<uint16_t>(curRows),            // blockCount
            static_cast<uint32_t>(numCol * sizeof(float)), // blockLen
            static_cast<uint32_t>(srcStride1),         // srcStride
            static_cast<uint32_t>(0),                  // dstStride
            0                                          // rsv
        };
        DataCopyExtParams copyParams2{
            static_cast<uint16_t>(curRows),            // blockCount
            static_cast<uint32_t>(numCol * sizeof(T)), // blockLen
            static_cast<uint32_t>(srcStride2),         // srcStride
            static_cast<uint32_t>(0),                  // dstStride
            0                                          // rsv
        };
        DataCopyPad(y1Gm[offset], y1Local, copyParams1);
        DataCopyPad(y2Gm[offset], y2Local, copyParams2);
        outQueueY1.FreeTensor(y1Local);
        outQueueY2.FreeTensor(y2Local);
    }

    __aicore__ inline void CopyOutX(uint64_t offset, uint32_t curRows, uint32_t colAlign)
    {
        LocalTensor<T> xLocal = outQueueX.DeQue<T>();
        uint32_t srcStride = (numColAlign - colAlign) * sizeof(T) / ALIGN_32_FACTOR;
        DataCopyExtParams copyParams{
            static_cast<uint16_t>(curRows),            // blockCount
            static_cast<uint32_t>(numCol * sizeof(T)), // blockLen
            static_cast<uint32_t>(srcStride),          // srcStride
            static_cast<uint32_t>(0),                  // dstStride
            0                                          // rsv
        };
        DataCopyPad(xOutGm[offset], xLocal, copyParams);
        outQueueX.FreeTensor(xLocal);
    }

private:
    TPipe* pPipe = nullptr;
    TQue<QuePosition::VECIN, 1> inQueueX1;
    TQue<QuePosition::VECIN, 1> inQueueX2;
    TQue<QuePosition::VECIN, 1> inQueueGamma;
    TQue<QuePosition::VECOUT, 1> outQueueY1;
    TQue<QuePosition::VECOUT, 1> outQueueY2;
    TQue<QuePosition::VECOUT, 1> outQueueRstd;
    TQue<QuePosition::VECOUT, 1> outQueueX;
    TBuf<TPosition::VECCALC> xReduceBuff;
    TBuf<TPosition::VECCALC> xFp32Buff;
    TBuf<TPosition::VECCALC> binaryAddBuf;

    GlobalTensor<T> xGm1;
    GlobalTensor<T> xGm2;
    GlobalTensor<T> gammaGm;
    GlobalTensor<float> y1Gm;
    GlobalTensor<T> y2Gm;
    GlobalTensor<float> rstdGm;
    GlobalTensor<T> xOutGm;
    uint64_t numRow;
    uint64_t numCol;
    uint64_t numColAlign;
    uint64_t blockFactor;
    uint64_t rowFactor;
    uint64_t binAddQuotient;
    float epsilon;
    float avgFactor;
    uint64_t numColB32Align;
    uint64_t rowWork{1};
};
} // namespace AddRmsNormCast
#endif // OPS_BUILT_IN_TBE_IMPL_ASCENDC_ADD_RMS_NORM_CAST_REGBASE_HIGH_PERFORMANCE_H
