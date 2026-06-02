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
#include "../../norm_common/reduce_common_regbase.h"

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
        pPipe->InitBuffer(outQueueX, DOUBLE_BUFFER_NUM, numColAlign * sizeof(T) * rowFactor);
        pPipe->InitBuffer(outQueueY2, DOUBLE_BUFFER_NUM, numColAlign * sizeof(T) * rowFactor);
        pPipe->InitBuffer(outQueueY1, DOUBLE_BUFFER_NUM, numColAlign * sizeof(float) * rowFactor);
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
        NormCommon::NormCommonRegbase::CalculateSquareReduceSum<float>(
            xFp32Local, xReduceLocal, binaryAddBuf, static_cast<uint16_t>(curRows), numColAlign, numCol,
            static_cast<uint32_t>(binAddQuotient), static_cast<uint32_t>(BLK_B32));
        NormCommon::ComputeRstdNewtonRaphson<true, true>(
            xReduceLocal, rstdLocal, curRows, epsilon, avgFactor, VL_FP32);
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

        uint32_t tileLen = curRows * numColAlign;
        uint16_t loopCount = (tileLen + VL_FP32 - 1) / VL_FP32;

        __VEC_SCOPE__
        {
            RegTensor<float> x1;
            RegTensor<float> x2;
            RegTensor<float> xSum;
            MaskReg pregLoop;
            for (uint16_t i = 0; i < loopCount; ++i) {
                uint32_t offset = i * VL_FP32;
                pregLoop = UpdateMask<float>(tileLen);
                LoadRegForDtype<T>(x1InUb, x1, pregLoop, offset);
                LoadRegForDtype<T>(x2InUb, x2, pregLoop, offset);
                Add(xSum, x1, x2, pregLoop);
                StoreRegForDtype<T>(xOutInUb, xSum, pregLoop, offset);
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
            RegTensor<float> rstd1Reg;
            RegTensor<float> rstd2Reg;
            RegTensor<float> gammaReg;
            RegTensor<float> mul1Reg;
            RegTensor<float> mul2Reg;
            RegTensor<float> mul1UnrollReg;
            RegTensor<float> mul2UnrollReg;

            for (uint16_t i = 0; i < loopRowsFold; ++i) {
                uint32_t sregCount = reduceNum;
                DataCopy<float, LoadDist::DIST_BRC_B32>(rstd1Reg, rstdInUb + NUM_TWO * i);
                DataCopy<float, LoadDist::DIST_BRC_B32>(rstd2Reg, rstdInUb + (NUM_TWO * i + NUM_ONE));
                for (uint16_t r = 0; r < loopCols; ++r) {
                    uint32_t offset1 = (NUM_TWO * i) * numColAlign + r * VL_FP32;
                    uint32_t offset2 = (NUM_TWO * i + NUM_ONE) * numColAlign + r * VL_FP32;
                    MaskReg regCurLoop = UpdateMask<float>(sregCount);
                    LoadRegForDtype<float>(xFp32Tmp, x1Reg, regCurLoop, offset1);
                    LoadRegForDtype<float>(xFp32Tmp, x2Reg, regCurLoop, offset2);
                    Mul(mul1Reg, x1Reg, rstd1Reg, regCurLoop);
                    Mul(mul1UnrollReg, x2Reg, rstd2Reg, regCurLoop);
                    LoadRegForDtype<T>(gammaInUb, gammaReg, regCurLoop, r * VL_FP32);
                    Mul(mul2Reg, mul1Reg, gammaReg, regCurLoop);
                    Mul(mul2UnrollReg, mul1UnrollReg, gammaReg, regCurLoop);
                    StoreRegForDtype<float>(y1InUb, mul2Reg, regCurLoop, offset1);
                    StoreRegForDtype<float>(y1InUb, mul2UnrollReg, regCurLoop, offset2);
                    StoreRegForDtype<T>(y2InUb, mul2Reg, regCurLoop, offset1);
                    StoreRegForDtype<T>(y2InUb, mul2UnrollReg, regCurLoop, offset2);
                }
            }
            for (uint16_t i = 0; i < loopRowsHasLast; ++i) {
                uint32_t sregCount = reduceNum;
                DataCopy<float, LoadDist::DIST_BRC_B32>(rstd1Reg, rstdInUb + NUM_TWO * loopRowsFold);
                for (uint16_t r = 0; r < loopCols; ++r) {
                    uint32_t offset = (NUM_TWO * loopRowsFold) * numColAlign + r * VL_FP32;
                    MaskReg regCurLoop = UpdateMask<float>(sregCount);
                    LoadRegForDtype<float>(xFp32Tmp, x1Reg, regCurLoop, offset);
                    Mul(mul1Reg, x1Reg, rstd1Reg, regCurLoop);
                    LoadRegForDtype<T>(gammaInUb, gammaReg, regCurLoop, r * VL_FP32);
                    Mul(mul2Reg, mul1Reg, gammaReg, regCurLoop);
                    StoreRegForDtype<float>(y1InUb, mul2Reg, regCurLoop, offset);
                    StoreRegForDtype<T>(y2InUb, mul2Reg, regCurLoop, offset);
                }
            }
        }
    }

    __aicore__ inline void CopyInXMutiMoveAlign(uint64_t offset, uint32_t curCols, uint32_t curRows = 0)
    {
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
        LocalTensor<T> xLocal1 = inQueueX1.AllocTensor<T>();
        LocalTensor<T> xLocal2 = inQueueX2.AllocTensor<T>();
        DataCopyPad(xLocal1, xGm1[offset], extParams, padParams);
        DataCopyPad(xLocal2, xGm2[offset], extParams, padParams);
        inQueueX1.EnQue(xLocal1);
        inQueueX2.EnQue(xLocal2);
    }

    __aicore__ inline void CopyInGamma()
    {
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
        LocalTensor<T> gammaLocal = inQueueGamma.AllocTensor<T>();
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
