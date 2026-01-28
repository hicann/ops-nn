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
 * \file rms_norm_regbase_perf.h
 * \brief RmsNorm regbase perf schedule
 */
#ifndef OPS_BUILT_IN_TBE_IMPL_ASCENDC_RMS_NORM_REGBASE_PERF_H
#define OPS_BUILT_IN_TBE_IMPL_ASCENDC_RMS_NORM_REGBASE_PERF_H
#include "rms_norm_regbase_common.h"

namespace RmsNorm {
    using namespace AscendC;

    template <typename DX, typename DG>
    class KernelRmsNormRegBasePerf {
    public:
        __aicore__ inline KernelRmsNormRegBasePerf()
        {}
        __aicore__ inline void Init(GM_ADDR x, GM_ADDR gamma, GM_ADDR y, GM_ADDR rstd, const RMSNormArch35TilingData* tiling)
        {
            // Tilingdata init
            curBlockIdx = GetBlockIdx();
            coreNum = GetBlockNum();
            numRow = tiling->num_row;
            numCol = tiling->num_col;
            numColAlign = tiling->num_col_align;
            ubFactor = tiling->ub_factor;
            blockFactor = tiling->block_factor;
            blockFactorTail = tiling->last_block_factor;

            epsilon = tiling->epsilon;
            avgFactor = tiling->avg_factor;
            colFlodFactor = tiling->col_flod_factor;

            uint64_t rstdBlockCount = blockFactor;
            uint64_t xBlockCount = rstdBlockCount * numCol;
            uint64_t curRstdBlockCount = rstdBlockCount;
            uint64_t curXBlockCount = xBlockCount;
            curBlockFactor = blockFactor;
            if (curBlockIdx == (coreNum - 1)) {
                curBlockFactor = blockFactorTail;
                curRstdBlockCount = blockFactorTail;
                curXBlockCount = curRstdBlockCount * numCol;
            }

            curBlockLoops = (curBlockFactor + ubFactor - 1) / ubFactor;
            curUbTails = curBlockFactor - (curBlockLoops - 1) * ubFactor;

            // init gm buffer
            xGm.SetGlobalBuffer((__gm__ DX*)x + curBlockIdx * xBlockCount, curXBlockCount);
            gammaGm.SetGlobalBuffer((__gm__ DG*)gamma, numCol);
            yGm.SetGlobalBuffer((__gm__ DX*)y + curBlockIdx * xBlockCount, curXBlockCount);
            rstdGm.SetGlobalBuffer((__gm__ float*)rstd + curBlockIdx * rstdBlockCount, curRstdBlockCount);

            // init que buffer
            pipe.InitBuffer(inQueueGamma, 1, numColAlign * sizeof(DG));
            pipe.InitBuffer(inQueueX, DOUBLE_BUFFER_NUM, ubFactor * numColAlign * sizeof(DX));
            pipe.InitBuffer(outQueueY, DOUBLE_BUFFER_NUM, ubFactor * numColAlign * sizeof(DX));
            uint64_t rstdUbSizeAlign =
                (ubFactor + blockSizeB32 - 1) / blockSizeB32 * blockSizeB32 * sizeof(float); // vector length align
            pipe.InitBuffer(outQueueRstd, DOUBLE_BUFFER_NUM, rstdUbSizeAlign);
            // init tmp buffer
            uint64_t firstVcaddResult =
                ubFactor *
                (((colFlodFactor + VectorLenB32 - 1) / VectorLenB32 + blockSizeB32 - 1) / blockSizeB32 * blockSizeB32) *
                sizeof(float);
            pipe.InitBuffer(xTmpBuf, firstVcaddResult);
            // branch 4 need 512B tmpBuffer
            pipe.InitBuffer(xReduceTmpBuf, rstdUbSizeAlign);
        }

        __aicore__ inline void Process()
        {
            // gamma copy params
            DataCopyPadExtParams<DG> dataCopyPadExtParamsGamma;
            dataCopyPadExtParamsGamma.isPad = false;
            dataCopyPadExtParamsGamma.leftPadding = 0;
            dataCopyPadExtParamsGamma.rightPadding = 0;
            dataCopyPadExtParamsGamma.paddingValue = 0;
            DataCopyExtParams copyInParamsGamma;
            copyInParamsGamma.blockCount = 1;
            copyInParamsGamma.blockLen = numCol * sizeof(DG);
            copyInParamsGamma.srcStride = 0;
            copyInParamsGamma.dstStride = 0;

            LocalTensor<DG> gammaLocal = inQueueGamma.AllocTensor<DG>();
            DataCopyPad(gammaLocal, gammaGm, copyInParamsGamma, dataCopyPadExtParamsGamma);

            inQueueGamma.EnQue(gammaLocal);
            inQueueGamma.DeQue<DG>();

            for (uint64_t i = 0; i < curBlockLoops; i++) {
                curUbFactor = (i == (curBlockLoops - 1)) ? curUbTails : ubFactor;

                DataCopyPadExtParams<DX> dataCopyPadExtParamsX;
                dataCopyPadExtParamsX.isPad = false;
                dataCopyPadExtParamsX.leftPadding = 0;
                dataCopyPadExtParamsX.rightPadding = 0;
                dataCopyPadExtParamsX.paddingValue = 0;
                DataCopyExtParams copyInParamsX;
                copyInParamsX.blockCount = curUbFactor;
                copyInParamsX.blockLen =  numCol * sizeof(DX);
                copyInParamsX.srcStride = 0;
                copyInParamsX.dstStride = (numColAlign - numCol) * sizeof(DX) / blockSize;
                LocalTensor<DX> xLocal = inQueueX.AllocTensor<DX>();
                DataCopyPad(xLocal, xGm[i * ubFactor * numCol], copyInParamsX, dataCopyPadExtParamsX);
                inQueueX.EnQue(xLocal);
                inQueueX.DeQue<DX>();
                LocalTensor<float> rstdLocal = outQueueRstd.AllocTensor<float>();
                LocalTensor<float> xTmpLocal = xTmpBuf.Get<float>();
                LocalTensor<float> xReduceTmpLocal = xReduceTmpBuf.Get<float>();
                ComputeSquareReduceSum(xLocal, xTmpLocal, xReduceTmpLocal, curUbFactor);
                ComputeRstd(xReduceTmpLocal, rstdLocal, curUbFactor, epsilon, avgFactor);

                outQueueRstd.EnQue(rstdLocal);
                outQueueRstd.DeQue<float>();

                DataCopyExtParams copyInParamsRstd;
                copyInParamsRstd.blockCount = 1;
                copyInParamsRstd.blockLen =  curUbFactor * sizeof(float);
                copyInParamsRstd.srcStride = 0;
                copyInParamsRstd.dstStride = 0;
                DataCopyPad(rstdGm[i * ubFactor], rstdLocal, copyInParamsRstd);

                LocalTensor<DX> yLocal = outQueueY.AllocTensor<DX>();
                ComputeY(xLocal, gammaLocal, rstdLocal, yLocal, curUbFactor);
                inQueueX.FreeTensor(xLocal);
                outQueueRstd.FreeTensor(rstdLocal);
                outQueueY.EnQue(yLocal);
                outQueueY.DeQue<DX>();

                DataCopyExtParams copyInParamsY;
                copyInParamsY.blockCount = curUbFactor;
                copyInParamsY.blockLen =  numCol * sizeof(DX);
                copyInParamsY.srcStride = (numColAlign - numCol) * sizeof(DX) / blockSize;
                copyInParamsY.dstStride = 0;
                DataCopyPad(yGm[i * ubFactor * numCol], yLocal, copyInParamsY);
                outQueueY.FreeTensor(yLocal);
            }
            inQueueGamma.FreeTensor(gammaLocal);
        }

    private:

        __aicore__ inline void ComputeSquareReduceSum(LocalTensor<DX> xLocal, LocalTensor<float> xTmpLocal, 
            LocalTensor<float> xReduceTmpLocal, uint64_t curUbFactor)
        {
            __local_mem__ DX* xLocalAddr = (__local_mem__ DX*)xLocal.GetPhyAddr();
            __local_mem__ float* xTmpLocalUbAddr = (__local_mem__ float*)xTmpLocal.GetPhyAddr();
            __local_mem__ float* xReduceTmpLocalUbAddr = (__local_mem__ float*)xReduceTmpLocal.GetPhyAddr();

            if (numColAlign <= VectorLenB32) {
                CalculateSquareReduceSumRLessThanVL(xLocalAddr, xReduceTmpLocalUbAddr, curUbFactor, numCol, numColAlign);
            } else if (numColAlign <= (VectorLenB32 + VectorLenB32)) {
                CalculateSquareReduceSumRLessThanTwoVL(xLocalAddr, xReduceTmpLocalUbAddr, curUbFactor, numCol, numColAlign);
            } else if (numColAlign <= VectorLenB32 * VectorLenB32 * NUM_TWO){
                CalculateSquareReduceSumRCommon<NUM_ONE>(xLocalAddr, xTmpLocalUbAddr, xReduceTmpLocalUbAddr, curUbFactor, numCol, numColAlign,colFlodFactor);
            } else {
                CalculateSquareReduceSumRCommon<NUM_TWO>(xLocalAddr, xTmpLocalUbAddr, xReduceTmpLocalUbAddr, curUbFactor, numCol, numColAlign,colFlodFactor);
            }
        }

        __aicore__ inline void ComputeRstd(LocalTensor<float> xReduceTmpLocal, LocalTensor<float> rstdLocal, uint64_t curUbFactor, float epsilon, float avgFactor)
        {
            __local_mem__ float* rstdLocalUbAddr = (__local_mem__ float*)rstdLocal.GetPhyAddr();
            __local_mem__ float* xReduceTmpLocalUbAddr = (__local_mem__ float*)xReduceTmpLocal.GetPhyAddr();
            uint16_t aLoop = static_cast<uint16_t>((curUbFactor + VectorLenB32 - 1) / VectorLenB32);
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
                uint32_t sreg0 = static_cast<uint32_t>(curUbFactor);
                for (uint16_t a = 0; a < aLoop; a++) {
                    pregLoop = UpdateMask<float>(sreg0);
                    Duplicate(scalar1, float(0.5), pregLoop);
                    Duplicate(scalarInf, RMS_POS_INF, pregLoop);
                    Duplicate(scalarZero, RMS_ZERO, pregLoop);
                    Duplicate(t1, float(1.5), pregLoop);
                    Duplicate(s, float(1.0), pregLoop);

                    // rstd
                    DataCopy(var, xReduceTmpLocalUbAddr + a * VectorLenB32);
                    Muls(var, var, avgFactor, pregLoop);
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
                    DataCopy(rstdLocalUbAddr + a * VectorLenB32, rstd, pregLoop);
                }
            }
        }

        __aicore__ inline void ComputeY(LocalTensor<DX> xLocal, LocalTensor<DG> gammaLocal, LocalTensor<float> rstdLocal, LocalTensor<DX> yLocal, uint64_t curUbFactor)
        {
            __local_mem__ DX* xLocalAddr = (__local_mem__ DX*)xLocal.GetPhyAddr();
            __local_mem__ DG* gammaLocalUbAddr = (__local_mem__ DG*)gammaLocal.GetPhyAddr();
            __local_mem__ float* rstdLocalUbAddr = (__local_mem__ float*)rstdLocal.GetPhyAddr();
            __local_mem__ DX* yLocalUbAddr = (__local_mem__ DX*)yLocal.GetPhyAddr();

            uint32_t colNum = static_cast<uint32_t>(numCol);
            uint16_t curAloops = static_cast<uint16_t>(curUbFactor);
            uint16_t colLoops = static_cast<uint16_t>((numColAlign + VectorLenB32-1) / VectorLenB32);
            uint32_t colNumAlign = static_cast<uint32_t>(numColAlign);
            __VEC_SCOPE__
            {
                RegTensor<float> RstdReg;
                RegTensor<float> gammaReg;
                RegTensor<float> xReg;
                RegTensor<float> mul1Reg;
                RegTensor<float> mul2Reg;

                MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
                MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();

                for (uint16_t i = 0; i < curAloops; i++) {
                    uint32_t sregElewiseNum = numCol;
                    DataCopy<float, LoadDist::DIST_BRC_B32>(RstdReg, rstdLocalUbAddr + i);
                    for (uint16_t j = 0; j < colLoops; j++)  {
                        MaskReg pregCurLoop = UpdateMask<float>(sregElewiseNum);
                        LoadTensorForDtypeTIn(xLocalAddr, xReg, pregCurLoop, (i * colNumAlign + j * VectorLenB32));
                        Mul(mul1Reg, xReg, RstdReg, pregCurLoop);
                        LoadTensorForDtypeTIn(gammaLocalUbAddr, gammaReg, pregCurLoop, (j * VectorLenB32));
                        Mul(mul2Reg, mul1Reg, gammaReg, pregCurLoop);
                        StoreTensorForDtypeTOut(yLocalUbAddr, mul2Reg, pregCurLoop, (i * colNumAlign + j * VectorLenB32));
                    }
                }
            }
        }

        __aicore__ inline void CalculateSquareReduceSumRLessThanVL(__local_mem__ DX* xLocalAddr,__local_mem__ float* xReduceTmpLocalUbAddr,
            uint64_t curUbFactor, uint64_t numCol, uint64_t numColAlign)
        {
            uint32_t colNum = static_cast<uint32_t>(numCol);
            uint16_t curAloops = static_cast<uint16_t>(curUbFactor);
            uint32_t colNumAlign = static_cast<uint32_t>(numColAlign);
            __VEC_SCOPE__
            {
                RegTensor<float> xReg;
                RegTensor<float> squareReg;
                RegTensor<float> sumReg;

                // rstd cal
                MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
                uint32_t colCountSreg = colNum;
                MaskReg pregLoop = UpdateMask<float>(colCountSreg);
                for (uint16_t i = 0; i < curAloops; i++) {
                    LoadTensorForDtypeTIn(xLocalAddr, xReg, pregLoop, (i * colNumAlign));
                    Mul(squareReg, xReg, xReg, pregLoop);
                    ReduceSum(sumReg, squareReg, pregLoop);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(xReduceTmpLocalUbAddr + i, sumReg, pregOne);
                }
            }
        }

        __aicore__ inline void CalculateSquareReduceSumRLessThanTwoVL(__local_mem__ DX* xLocalAddr, 
            __local_mem__ float* xReduceTmpLocalUbAddr,
            uint64_t curUbFactor, uint64_t numCol, uint64_t numColAlign)
        {
            uint32_t colNum = static_cast<uint32_t>(numCol);
            uint32_t colNumAlign = static_cast<uint32_t>(numColAlign);
            uint16_t curAloops = static_cast<uint16_t>(curUbFactor);
            
            __VEC_SCOPE__
            {
                RegTensor<float> xReg;
                RegTensor<float> xTailReg;
                RegTensor<float> squareReg;
                RegTensor<float> squareTailReg;
                RegTensor<float> addReg;
                RegTensor<float> sumReg;

                MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
                MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
                // 64 CHANGE dtype need check
                uint32_t colTailSreg = colNum - VectorLenB32;
                MaskReg pregTail = UpdateMask<float>(colTailSreg);
                for (uint16_t i = 0; i < curAloops; i++) {
                    LoadTensorForDtypeTIn(xLocalAddr, xReg, pregFull, (i * colNumAlign));
                    Mul(squareReg, xReg, xReg, pregFull);
                    LoadTensorForDtypeTIn(xLocalAddr + VectorLenB32, xTailReg, pregTail, (i * colNumAlign));
                    Mul(squareTailReg, xTailReg, xTailReg, pregTail);
                    Add(addReg, squareReg, squareTailReg, pregFull);
                    ReduceSum(sumReg, addReg, pregFull);
                    DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(xReduceTmpLocalUbAddr + i, sumReg, pregOne);
                }
            }
        }

        template <int32_t LAST_LOOP_NUMS>
        __aicore__ inline void CalculateSquareReduceSumRCommon(__local_mem__ DX* xLocalAddr,
            __local_mem__ float* xTmpLocalUbAddr, __local_mem__ float* xReduceTmpLocalUbAddr, uint64_t curUbFactor, 
            uint64_t numCol, uint64_t numColAlign, uint64_t colFlodFactor)
        {
            uint32_t colNum = static_cast<uint32_t>(numCol);
            uint32_t colNumAlign = static_cast<uint32_t>(numColAlign);
            uint32_t colFlodNum = static_cast<uint32_t>(colFlodFactor);
            uint16_t curAloops = static_cast<uint16_t>(curUbFactor);

            // first flod
            uint32_t firstFlodTial = static_cast<uint32_t>(colNum - colFlodFactor);
            uint16_t firstFlodAddLoops = static_cast<uint16_t>((firstFlodTial + VectorLenB32-1) / VectorLenB32);
            uint16_t firstFlodWithOutAddLoops = static_cast<uint16_t>((colNumAlign + VectorLenB32-1) / VectorLenB32) - firstFlodAddLoops;

            // first vcadd
            uint32_t firstVcaddNum = static_cast<uint32_t>((colFlodFactor + VectorLenB32-1 )/ VectorLenB32);
            uint32_t firstVcaddNumCeilAlign = static_cast<uint32_t>((firstVcaddNum + blockSizeB32-1) / blockSizeB32 * blockSizeB32);

            // second flod
            // rstd cal
            uint16_t elewiseLoop = static_cast<uint16_t>((curUbFactor + VectorLenB32-1) / VectorLenB32);

            __VEC_SCOPE__
            {
                RegTensor<float> xReg1;
                RegTensor<float> xReg2;
                RegTensor<float> squareReg1;
                RegTensor<float> squareReg2;
                RegTensor<float> addReg;
                RegTensor<float> sumReg;

                RegTensor<float> xReg3;
                RegTensor<float> squareReg3;
                RegTensor<float> sumReg3;

                RegTensor<float> mulsReg;
                RegTensor<float> addsReg;
                RegTensor<float> sqrtReg;

                MaskReg pregFull = CreateMask<float, MaskPattern::ALL>();
                MaskReg pregOne = CreateMask<float, MaskPattern::VL1>();
                MaskReg pregLoop;

                for (uint16_t i = 0; i < curAloops; i++) {
                    uint32_t sregfirstFlodTial = firstFlodTial;
                    for (uint16_t j = 0; j < firstFlodAddLoops; j++) {
                        pregLoop = UpdateMask<float>(sregfirstFlodTial);
                        LoadTensorForDtypeTIn(xLocalAddr, xReg1, pregFull, (i * colNumAlign + j * VectorLenB32));
                        Mul(squareReg1, xReg1, xReg1, pregFull);
                        LoadTensorForDtypeTIn(xLocalAddr + colFlodNum, xReg2, pregFull, (i * colNumAlign + j * VectorLenB32));
                        Mul(squareReg2, xReg2, xReg2, pregLoop);
                        Add(addReg, squareReg1, squareReg2, pregFull);
                        ReduceSum(sumReg, addReg, pregFull);
                        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                            xTmpLocalUbAddr + static_cast<uint32_t>(i * firstVcaddNumCeilAlign + j), sumReg, pregOne);
                    }
                    for (uint16_t j = 0; j < static_cast<uint16_t>(firstFlodWithOutAddLoops); j++) {
                        LoadTensorForDtypeTIn(xLocalAddr + firstFlodAddLoops * VectorLenB32,  xReg3, pregFull, 
                            (i * colNumAlign + j * VectorLenB32));
                        Mul(squareReg3, xReg3, xReg3, pregFull);
                        ReduceSum(sumReg3, squareReg3, pregFull);
                        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(
                            xTmpLocalUbAddr + static_cast<uint32_t>(i * firstVcaddNumCeilAlign + firstFlodAddLoops + j),sumReg3, pregOne);
                    }
                }

                // if need a add to last repeat
                LocalMemBar<MemType::VEC_STORE, MemType::VEC_LOAD>();
                if constexpr (LAST_LOOP_NUMS == 1) {
                    uint32_t sregSecondReduce = firstVcaddNum;
                    MaskReg pregLast = UpdateMask<float>(sregSecondReduce);
                    for (uint16_t i = 0; i < curAloops; i++) {
                        DataCopy(xReg1, xTmpLocalUbAddr + static_cast<uint32_t>(i * firstVcaddNumCeilAlign));
                        ReduceSum(sumReg, xReg1, pregLast);
                        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(xReduceTmpLocalUbAddr + i, sumReg, pregOne);
                    }
                } else if constexpr (LAST_LOOP_NUMS == 2) {
                    uint32_t sregSecondReduce = firstVcaddNum - VectorLenB32;
                    MaskReg pregLast = UpdateMask<float>(sregSecondReduce);
                    RegTensor<float> shiftLeft;
                    for (uint16_t i = 0; i < curAloops; i++) {
                        DataCopy(xReg1, xTmpLocalUbAddr + static_cast<uint32_t>(i * firstVcaddNumCeilAlign));
                        DataCopy(xReg2, xTmpLocalUbAddr + static_cast<uint32_t>(i * firstVcaddNumCeilAlign + VectorLenB32));
                        ShiftLefts((RegTensor<uint32_t> &)shiftLeft, (RegTensor<uint32_t> &)xReg2, static_cast<int16_t>(0), pregLast);
                        Add(addReg, xReg1, shiftLeft, pregFull);
                        ReduceSum(sumReg, addReg, pregFull);
                        DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(xReduceTmpLocalUbAddr + i, sumReg, pregOne);
                    }
                }
            }
        }

        template <typename T_IN>
        __aicore__ inline void LoadTensorForDtypeTIn(__local_mem__ T_IN* src, RegTensor<float>& dst,
                                                    MaskReg& preg, uint32_t offset)
        {
            if constexpr (IsSameType<T_IN, float>::value) {
                DataCopy<float, LoadDist::DIST_NORM>(dst, src + offset);
            } else {
                RegTensor<T_IN> xIn;
                DataCopy<T_IN, LoadDist::DIST_UNPACK_B16>(xIn, src + offset);
                Cast<float, T_IN, castTraitB162B32>(dst, xIn, preg);
            }
        }

        template <typename T_OUT>
        __aicore__ inline void StoreTensorForDtypeTOut(__local_mem__ T_OUT* dst, RegTensor<float>& src,
                                                    MaskReg& preg, uint32_t offset)
        {
            if constexpr (IsSameType<T_OUT, float>::value) {
                DataCopy<T_OUT, StoreDist::DIST_NORM>(dst + offset, src, preg);
            } else {
                RegTensor<T_OUT> xOut;
                Cast<T_OUT, float, castTraitB322B16>(xOut, src, preg);
                DataCopy<T_OUT, StoreDist::DIST_PACK_B32>(dst + offset, xOut, preg);
            }
        }

    private:
        TPipe pipe;
        // QUE
        TQue<QuePosition::VECIN, DOUBLE_BUFFER_NUM> inQueueX;
        TQue<QuePosition::VECIN, 1> inQueueGamma;
        TQue<QuePosition::VECOUT, DOUBLE_BUFFER_NUM> outQueueY;
        TQue<QuePosition::VECOUT, DOUBLE_BUFFER_NUM> outQueueRstd;
        TBuf<TPosition::VECCALC> xTmpBuf;
        TBuf<TPosition::VECCALC> xReduceTmpBuf;

        // GM tensor
        GlobalTensor<DX> xGm;
        GlobalTensor<DG> gammaGm;
        GlobalTensor<DX> yGm;
        GlobalTensor<float> rstdGm;

        uint64_t curBlockIdx;
        uint64_t coreNum;
        uint64_t numRow;
        uint64_t numCol;
        uint64_t numColAlign;
        uint64_t blockFactor;
        uint64_t ubFactor;
        uint64_t colFlodFactor;
        uint64_t blockFactorTail;
        float epsilon;
        float avgFactor;
        uint64_t curBlockFactor;
        uint64_t curUbFactor;
        uint64_t curBlockLoops;
        uint64_t curUbTails;
        uint32_t blockSize = GetUbBlockSize();
        uint32_t blockSizeB32 = GetUbBlockSize() / sizeof(float);
        uint32_t VectorLenB32 = GetVRegSize() / sizeof(float);
        static constexpr float RMS_POS_INF = 3.40282366920938E+38;
        static constexpr float RMS_ZERO = 0.0f;
        static constexpr int32_t NUM_ONE = 1;
        static constexpr int32_t NUM_TWO = 2;
};
} // namespace RmsNorm
#endif // OPS_BUILT_IN_TBE_IMPL_ASCENDC_RMS_NORM_REGBASE_PERF_H