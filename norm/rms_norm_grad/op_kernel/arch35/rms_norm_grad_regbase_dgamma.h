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
 * \file rms_norm_grad_regbase_dgamma.h
 * \brief RmsNorm regbase dgamma
 */

#ifndef RMS_NORM_GRAD_REGBASE_DGAMMA_H
#define RMS_NORM_GRAD_REGBASE_DGAMMA_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "rms_norm_grad_dgamma_regbase_helper.h"

namespace RmsNormGrad {
using namespace AscendC;
template <typename DY_TYPE, typename X_TYPE, typename RSTD_TYPE, int32_t TILING_KEY, int32_t BUFFER_NUM = 2>
class RegbaseDgamma {
public:
    __aicore__ inline RegbaseDgamma(TPipe* pipe, const RmsNormGradRegbaseTilingData* tilingData)
        : Ppipe_(pipe), tiling_(tilingData)
    {}

    __aicore__ inline void Init(
        __gm__ uint8_t* dy, __gm__ uint8_t* x, __gm__ uint8_t* rstd, __gm__ uint8_t* gamma, __gm__ uint8_t* dx,
        __gm__ uint8_t* dgamma)
    {
        coreIdx_ = AscendC::GetBlockIdx();
        if (coreIdx_ >= tiling_->usedCoreNumDG) {
            return;
        }
        blockSize_ = tiling_->blockSize;
        vlFp32_ = VECTOR_REG_WIDTH / sizeof(float);
        colsPerUB_ = vlFp32_;
        cols_ = tiling_->dxTilingData.cols;
        rows_ = tiling_->dxTilingData.rows;
        binaryAddK_ = tiling_->binaryAddKDG;
        colsPerCore_ = tiling_->colsPerCoreDG;
        rowsPerUB_ = tiling_->rowsPerUBDG;
        gmOffset_ = colsPerCore_ * coreIdx_;
        colsUbLoopCount_ = CEIL_DIV(colsPerCore_, colsPerUB_);
        dyGm_.SetGlobalBuffer((__gm__ DY_TYPE*)dy);
        xGm_.SetGlobalBuffer((__gm__ X_TYPE*)x);
        rstdGm_.SetGlobalBuffer((__gm__ RSTD_TYPE*)rstd);
        dgammaGm_.SetGlobalBuffer((__gm__ float*)dgamma);

        colsPerLoopAlign_ = BLOCK_ALIGN(colsPerUB_ * sizeof(float), blockSize_) / sizeof(float);
        int64_t rowsPerUbAligned = BLOCK_ALIGN(rowsPerUB_ * sizeof(float), blockSize_) / sizeof(float);
        Ppipe_->InitBuffer(rstdQueue_, BUFFER_NUM, (rowsPerUbAligned * sizeof(float)));
        Ppipe_->InitBuffer(dyQueue_, BUFFER_NUM, (rowsPerUB_ * colsPerLoopAlign_ * sizeof(float)));
        Ppipe_->InitBuffer(xQueue_, BUFFER_NUM, (rowsPerUB_ * colsPerLoopAlign_ * sizeof(float)));
        Ppipe_->InitBuffer(dgammaQueue_, BUFFER_NUM, ((rowsPerUB_ + 1) * colsPerLoopAlign_ * sizeof(float)));

        if (TILING_KEY == 7001) {
            Ppipe_->InitBuffer(
                binaryAddCacheQueue_, BUFFER_NUM,
                ((tiling_->binaryAddKDG + RESERVESIZE) * colsPerLoopAlign_ * sizeof(float)));
            Ppipe_->InitBuffer(dgammaQueue1_, BUFFER_NUM, ((rowsPerUB_ + 1) * colsPerLoopAlign_ * sizeof(float)));
        }
    }

    __aicore__ inline void CopyInputsToUB(
        LocalTensor<DY_TYPE> dyLocal, LocalTensor<X_TYPE> xLocal, LocalTensor<RSTD_TYPE> rstdLocal, int64_t inputOffset,
        int32_t copyLen, int32_t curRowsNum, int32_t rstdOffset)
    {
        // Datacopy Params for input_x & input_dy
        DataCopyPadExtParams<X_TYPE> padParams_x;
        padParams_x.isPad = false;
        DataCopyExtParams dataCopyParams_x;
        dataCopyParams_x.blockCount = curRowsNum;
        dataCopyParams_x.blockLen = copyLen * sizeof(X_TYPE);
        dataCopyParams_x.srcStride = (cols_ - copyLen) * sizeof(X_TYPE);
        dataCopyParams_x.dstStride = (colsPerUB_ - copyLen) * sizeof(X_TYPE) / blockSize_;
        // Datacopy Params for input_rstd
        DataCopyPadExtParams<RSTD_TYPE> padParams_rstd;
        padParams_rstd.isPad = false;
        DataCopyExtParams dataCopyParams_rstd;
        dataCopyParams_rstd.blockCount = 1;
        dataCopyParams_rstd.blockLen = curRowsNum * sizeof(RSTD_TYPE);
        dataCopyParams_rstd.srcStride = 0;
        dataCopyParams_rstd.dstStride = 0;

        DataCopyPad(xLocal, xGm_[inputOffset], dataCopyParams_x, padParams_x);
        DataCopyPad(dyLocal, dyGm_[inputOffset], dataCopyParams_x, padParams_x);
        DataCopyPad(rstdLocal, rstdGm_[rstdOffset], dataCopyParams_rstd, padParams_rstd);
    }

    __aicore__ inline void CopyDgammaToGm(
        LocalTensor<float> outLocal, uint32_t dgammaGmOffset, int32_t curCols, int32_t dgammaUBOffset)
    {
        DataCopyExtParams dataCopyParams;
        dataCopyParams.blockCount = 1;
        dataCopyParams.blockLen = curCols * sizeof(float);
        dataCopyParams.srcStride = 0;
        dataCopyParams.dstStride = 0;

        DataCopyPad(dgammaGm_[dgammaGmOffset], outLocal[dgammaUBOffset], dataCopyParams);
    }

    __aicore__ inline void VFCalcPreDgamma(
        __local_mem__ DY_TYPE* dyAddr, __local_mem__ X_TYPE* xAddr, __local_mem__ RSTD_TYPE* rstdAddr,
        __local_mem__ float* dgammaOutAddr, uint16_t curUBLoopColsCount, int32_t curRowsNum)
    {
        uint16_t colsRegLoopCount = CEIL_DIV(curUBLoopColsCount, vlFp32_);
        uint32_t colsPerLoop = colsPerUB_;
        uint32_t colsPerLoopAlign = colsPerLoopAlign_;

        __VEC_SCOPE__
        {
            MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
            // 计算二分累加前的乘法计算
            for (uint16_t k = 0; k < static_cast<uint16_t>(curRowsNum); k++) {
                CalcMulRes<DY_TYPE, X_TYPE, RSTD_TYPE, TILING_KEY>(
                    dyAddr, xAddr, rstdAddr, dgammaOutAddr, pregMain, (k * colsRegLoopCount) * vlFp32_, k);
            }
        }
    }

    __aicore__ inline void VFDuplicateRows(__local_mem__ float* srcAddr, uint32_t padRowsLen, uint64_t rowsBoundLine)
    {
        __VEC_SCOPE__
        {
            RegTensor<float> tempReg;
            MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
            uint32_t sreg0 = padRowsLen;
            MaskReg pregLoop = UpdateMask<float>(sreg0);
            // 填充数据
            AscendC::MicroAPI::Duplicate(tempReg, 0);
            AscendC::MicroAPI::DataCopy(srcAddr + static_cast<uint32_t>(rowsBoundLine), tempReg, pregLoop);
        }
    }

    __aicore__ inline void VFBinaryReduceSumWithoutTail(
        __local_mem__ float* dgammaOutAddr, uint16_t curUbLoopColsCount, int64_t rows)
    {
        uint32_t BinaryAddNumLevel2 = rows / REDUCEBY8ELENUM;
        uint32_t BinaryAddNumLevel1 = BinaryAddNumLevel2 <= REDUCEBY8ELENUM ? 1 : BinaryAddNumLevel2 / 16;
        if (rows <= REDUCEBY8ELENUM) {
            BinaryAddNumLevel1 = 0;
        }
        __VEC_SCOPE__
        {
            MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
            // 计算二分累加level2
            for (uint16_t i = 0; i < static_cast<uint16_t>(BinaryAddNumLevel2); i++) {
                reduceSumCompressedBy8(dgammaOutAddr, pregMain, vlFp32_, i * vlFp32_);
            }
            MicroAPI::LocalMemBar<MicroAPI::MemType::VEC_STORE, MicroAPI::MemType::VEC_LOAD>();
            // 计算二分累加level1
            if (BinaryAddNumLevel1 == 1) {
                if (BinaryAddNumLevel2 == REDUCEBY8ELENUM) {
                    reduceSumCompressedBy8(dgammaOutAddr, pregMain, vlFp32_, 0);
                } else if (BinaryAddNumLevel2 == REDUCEBY4ELENUM) {
                    reduceSumCompressedBy4(dgammaOutAddr, pregMain, vlFp32_, 0);
                } else if (BinaryAddNumLevel2 == REDUCEBY2ELENUM) {
                    reduceSumCompressedBy2(dgammaOutAddr, pregMain, vlFp32_, 0);
                } else if (BinaryAddNumLevel2 == REDUCEBY1ELENUM) {
                    reduceSumCompressedBy1(dgammaOutAddr, pregMain, vlFp32_);
                }
            } else {
                for (uint16_t i = 0; i < BinaryAddNumLevel1; i++) {
                    reduceSumCompressedBy8(dgammaOutAddr, pregMain, vlFp32_, i * vlFp32_);
                }
                // 计算二分累加level0
                if (BinaryAddNumLevel1 == REDUCEBY8ELENUM) {
                    reduceSumCompressedBy8(dgammaOutAddr, pregMain, vlFp32_, 0);
                } else if (BinaryAddNumLevel1 == REDUCEBY4ELENUM) {
                    reduceSumCompressedBy4(dgammaOutAddr, pregMain, vlFp32_, 0);
                } else if (BinaryAddNumLevel1 == REDUCEBY2ELENUM) {
                    reduceSumCompressedBy2(dgammaOutAddr, pregMain, vlFp32_, 0);
                } else if (BinaryAddNumLevel1 == REDUCEBY1ELENUM) {
                    reduceSumCompressedBy1(dgammaOutAddr, pregMain, vlFp32_);
                }
            }
            if (rows == REDUCEBY4ELENUM) {
                reduceSumCompressedBy4(dgammaOutAddr, pregMain, vlFp32_, 0);
            } else if (rows == REDUCEBY2ELENUM) {
                reduceSumCompressedBy2(dgammaOutAddr, pregMain, vlFp32_, 0);
            } else if (rows == REDUCEBY1ELENUM) {
                reduceSumCompressedBy1(dgammaOutAddr, pregMain, vlFp32_);
            }
        }
    }

    __aicore__ inline void VFHandleTailRows(
        __local_mem__ float* dgammaOutAddr, uint16_t rowsTail, uint64_t tailDataOffset)
    {
        uint32_t BinaryAddTailNum = (rowsTail + COMPRESSBY8ELENUM - 1) / COMPRESSBY8ELENUM;

        __VEC_SCOPE__
        {
            MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
            uint64_t rowsBoundLine = rows_ * vlFp32_;
            for (uint16_t i = 0; i < static_cast<uint16_t>(BinaryAddTailNum - 1); i++) {
                reduceSumCompressedBy8WithOutPad(
                    dgammaOutAddr, dgammaOutAddr + tailDataOffset, pregMain, i * vlFp32_ * COMPRESSBY8ELENUM, vlFp32_);
            }
            reduceSumCompressedBy8WithPad(
                dgammaOutAddr, dgammaOutAddr, pregMain, (BinaryAddTailNum - 1) * vlFp32_ * COMPRESSBY8ELENUM,
                rowsBoundLine, vlFp32_, tailDataOffset);
        }
    }

    __aicore__ inline void VFHandleTailRowsWithTwoBuffer(
        __local_mem__ float* dgammaOutAddr, __local_mem__ float* dgammaOutAddr1, uint64_t tailRowsNum)
    {
        uint32_t BinaryAddTailNum = (tailRowsNum + COMPRESSBY8ELENUM - 1) / COMPRESSBY8ELENUM;
        uint32_t tailDataOffset = 0;

        __VEC_SCOPE__
        {
            MaskReg pregMain = CreateMask<float, MaskPattern::ALL>();
            uint64_t rowsBoundLine = tailRowsNum * vlFp32_;
            for (uint16_t i = 0; i < static_cast<uint16_t>(BinaryAddTailNum - 1); i++) {
                reduceSumCompressedBy8WithOutPad(
                    dgammaOutAddr, dgammaOutAddr1, pregMain, i * vlFp32_ * COMPRESSBY8ELENUM, vlFp32_);
            }
            reduceSumCompressedBy8WithPad(
                dgammaOutAddr, dgammaOutAddr1, pregMain, (BinaryAddTailNum - 1) * vlFp32_ * COMPRESSBY8ELENUM,
                rowsBoundLine, vlFp32_, tailDataOffset);
        }
    }

    __aicore__ inline void CalcDgamma(uint32_t inputOffset, uint32_t currentCols, bool isWithPad)
    {
        LocalTensor<RSTD_TYPE> rstdLocal = rstdQueue_.template AllocTensor<RSTD_TYPE>();
        LocalTensor<DY_TYPE> dyLocal = dyQueue_.template AllocTensor<DY_TYPE>();
        LocalTensor<X_TYPE> xLocal = xQueue_.template AllocTensor<X_TYPE>();
        LocalTensor<float> dgammaOutLocal = dgammaQueue_.template AllocTensor<float>();

        CopyInputsToUB(dyLocal, xLocal, rstdLocal, inputOffset, currentCols, rowsPerUB_, 0);
        xQueue_.EnQue(xLocal);
        rstdQueue_.EnQue(rstdLocal);
        dyQueue_.EnQue(dyLocal);

        dyLocal = dyQueue_.template DeQue<DY_TYPE>();
        xLocal = xQueue_.template DeQue<X_TYPE>();
        rstdLocal = rstdQueue_.template DeQue<RSTD_TYPE>();

        __local_mem__ DY_TYPE* dyAddr = (__local_mem__ DY_TYPE*)dyLocal[0].GetPhyAddr();
        __local_mem__ X_TYPE* xAddr = (__local_mem__ X_TYPE*)xLocal[0].GetPhyAddr();
        __local_mem__ RSTD_TYPE* rstdAddr = (__local_mem__ RSTD_TYPE*)rstdLocal[0].GetPhyAddr();
        __local_mem__ float* dgammaOutAddr = (__local_mem__ float*)dgammaOutLocal[0].GetPhyAddr();

        VFCalcPreDgamma(dyAddr, xAddr, rstdAddr, dgammaOutAddr, currentCols, rowsPerUB_);
        dyQueue_.FreeTensor(dyLocal);
        xQueue_.FreeTensor(xLocal);
        rstdQueue_.FreeTensor(rstdLocal);
        if (isWithPad) {
            uint32_t mainRows = rows_ - tiling_->rowsTailDG;
            VFDuplicateRows(dgammaOutAddr, vlFp32_, rows_ * vlFp32_);
            tailDataOffset_ = mainRows * vlFp32_;
            VFHandleTailRows(dgammaOutAddr, tiling_->rowsTailDG, tailDataOffset_);
            VFBinaryReduceSumWithoutTail(dgammaOutAddr, currentCols, mainRows);
        } else {
            VFBinaryReduceSumWithoutTail(dgammaOutAddr, currentCols, rows_);
        }

        dgammaQueue_.EnQue(dgammaOutLocal);
        dgammaOutLocal = dgammaQueue_.template DeQue<float>();
        CopyDgammaToGm(dgammaOutLocal, inputOffset, currentCols, 0);

        dgammaQueue_.FreeTensor(dgammaOutLocal);
    }

    __aicore__ inline void CalcLargeRowsDgamma(
        uint32_t inputOffset, uint32_t currentCols, uint32_t i, LocalTensor<float> binaryAddCacheLocal)
    {
        LocalTensor<RSTD_TYPE> rstdLocal = rstdQueue_.template AllocTensor<RSTD_TYPE>();
        LocalTensor<DY_TYPE> dyLocal = dyQueue_.template AllocTensor<DY_TYPE>();
        LocalTensor<X_TYPE> xLocal = xQueue_.template AllocTensor<X_TYPE>();
        LocalTensor<float> dgammaOutLocal = dgammaQueue_.template AllocTensor<float>();
        int64_t cacheID = GetCacheID(i);
        uint32_t rstdOffset = i * rowsPerUB_;

        CopyInputsToUB(dyLocal, xLocal, rstdLocal, inputOffset, currentCols, rowsPerUB_, rstdOffset);
        xQueue_.EnQue(xLocal);
        rstdQueue_.EnQue(rstdLocal);
        dyQueue_.EnQue(dyLocal);

        dyLocal = dyQueue_.template DeQue<DY_TYPE>();
        xLocal = xQueue_.template DeQue<X_TYPE>();
        rstdLocal = rstdQueue_.template DeQue<RSTD_TYPE>();

        __local_mem__ DY_TYPE* dyAddr = (__local_mem__ DY_TYPE*)dyLocal[0].GetPhyAddr();
        __local_mem__ X_TYPE* xAddr = (__local_mem__ X_TYPE*)xLocal[0].GetPhyAddr();
        __local_mem__ RSTD_TYPE* rstdAddr = (__local_mem__ RSTD_TYPE*)rstdLocal[0].GetPhyAddr();
        __local_mem__ float* dgammaOutAddr = (__local_mem__ float*)dgammaOutLocal[0].GetPhyAddr();

        VFCalcPreDgamma(dyAddr, xAddr, rstdAddr, dgammaOutAddr, currentCols, rowsPerUB_);
        dyQueue_.FreeTensor(dyLocal);
        xQueue_.FreeTensor(xLocal);
        rstdQueue_.FreeTensor(rstdLocal);

        VFBinaryReduceSumWithoutTail(dgammaOutAddr, currentCols, rowsPerUB_);
        UpdateCache(binaryAddCacheLocal, dgammaOutAddr, cacheID, vlFp32_);
        dgammaQueue_.FreeTensor(dgammaOutLocal);
    }

    __aicore__ inline void CalcLargeRowsDgammaWithPad(
        uint32_t inputOffset, uint32_t tailDyXOffset, uint32_t tailRstdOffset, uint32_t currentCols, uint32_t i,
        LocalTensor<float> binaryAddCacheLocal)
    {
        LocalTensor<RSTD_TYPE> rstdLocal = rstdQueue_.template AllocTensor<RSTD_TYPE>();
        LocalTensor<DY_TYPE> dyLocal = dyQueue_.template AllocTensor<DY_TYPE>();
        LocalTensor<X_TYPE> xLocal = xQueue_.template AllocTensor<X_TYPE>();
        LocalTensor<float> dgammaOutLocal = dgammaQueue_.template AllocTensor<float>();
        int64_t cacheID = GetCacheID(i);
        uint32_t rstdOffset = i * rowsPerUB_;

        CopyInputsToUB(dyLocal, xLocal, rstdLocal, inputOffset, currentCols, rowsPerUB_, rstdOffset);
        xQueue_.EnQue(xLocal);
        rstdQueue_.EnQue(rstdLocal);
        dyQueue_.EnQue(dyLocal);

        dyLocal = dyQueue_.template DeQue<DY_TYPE>();
        xLocal = xQueue_.template DeQue<X_TYPE>();
        rstdLocal = rstdQueue_.template DeQue<RSTD_TYPE>();

        __local_mem__ DY_TYPE* dyAddr = (__local_mem__ DY_TYPE*)dyLocal[0].GetPhyAddr();
        __local_mem__ X_TYPE* xAddr = (__local_mem__ X_TYPE*)xLocal[0].GetPhyAddr();
        __local_mem__ RSTD_TYPE* rstdAddr = (__local_mem__ RSTD_TYPE*)rstdLocal[0].GetPhyAddr();
        __local_mem__ float* dgammaOutAddr = (__local_mem__ float*)dgammaOutLocal[0].GetPhyAddr();

        VFCalcPreDgamma(dyAddr, xAddr, rstdAddr, dgammaOutAddr, currentCols, rowsPerUB_);
        dyQueue_.FreeTensor(dyLocal);
        xQueue_.FreeTensor(xLocal);
        rstdQueue_.FreeTensor(rstdLocal);

        // 处理累加的尾块
        LocalTensor<RSTD_TYPE> rstdLocal1 = rstdQueue_.template AllocTensor<RSTD_TYPE>();
        LocalTensor<DY_TYPE> dyLocal1 = dyQueue_.template AllocTensor<DY_TYPE>();
        LocalTensor<X_TYPE> xLocal1 = xQueue_.template AllocTensor<X_TYPE>();
        LocalTensor<float> dgammaOutLocal1 = dgammaQueue1_.template AllocTensor<float>();
        int32_t tailRowsNum = i == tiling_->tailBlockCountWithoutPadDG ? rows_ % rowsPerUB_ : rowsPerUB_;
        CopyInputsToUB(dyLocal1, xLocal1, rstdLocal1, tailDyXOffset, currentCols, tailRowsNum, tailRstdOffset);
        xQueue_.EnQue(xLocal1);
        rstdQueue_.EnQue(rstdLocal1);
        dyQueue_.EnQue(dyLocal1);

        dyLocal1 = dyQueue_.template DeQue<DY_TYPE>();
        xLocal1 = xQueue_.template DeQue<X_TYPE>();
        rstdLocal1 = rstdQueue_.template DeQue<RSTD_TYPE>();

        __local_mem__ DY_TYPE* dyAddr1 = (__local_mem__ DY_TYPE*)dyLocal1[0].GetPhyAddr();
        __local_mem__ X_TYPE* xAddr1 = (__local_mem__ X_TYPE*)xLocal1[0].GetPhyAddr();
        __local_mem__ RSTD_TYPE* rstdAddr1 = (__local_mem__ RSTD_TYPE*)rstdLocal1[0].GetPhyAddr();
        __local_mem__ float* dgammaOutAddr1 = (__local_mem__ float*)dgammaOutLocal1[0].GetPhyAddr();

        VFCalcPreDgamma(dyAddr1, xAddr1, rstdAddr1, dgammaOutAddr1, currentCols, rowsPerUB_);
        dyQueue_.FreeTensor(dyLocal1);
        xQueue_.FreeTensor(xLocal1);
        rstdQueue_.FreeTensor(rstdLocal1);

        if (i == tiling_->tailBlockCountWithoutPadDG) {
            VFDuplicateRows(dgammaOutAddr1, vlFp32_, tailRowsNum * vlFp32_);
            VFHandleTailRowsWithTwoBuffer(dgammaOutAddr, dgammaOutAddr1, tailRowsNum);
        } else {
            VFHandleTailRowsWithTwoBuffer(dgammaOutAddr, dgammaOutAddr1, rowsPerUB_);
        }
        dgammaQueue1_.FreeTensor(dgammaOutLocal1);

        VFBinaryReduceSumWithoutTail(dgammaOutAddr, currentCols, rowsPerUB_);
        UpdateCache(binaryAddCacheLocal, dgammaOutAddr, cacheID, vlFp32_);
        dgammaQueue_.FreeTensor(dgammaOutLocal);
    }

    __aicore__ inline void HandlingLargeRows2KAlign(uint32_t startOffset, uint32_t currentCols)
    {
        LocalTensor<float> binaryAddCacheLocal = binaryAddCacheQueue_.template AllocTensor<float>();
        for (int32_t i = 0; i < tiling_->mainBlockCountDG; i++) {
            uint32_t inputOffset = startOffset + i * cols_ * rowsPerUB_;
            CalcLargeRowsDgamma(inputOffset, currentCols, i, binaryAddCacheLocal);
        }
        binaryAddCacheQueue_.EnQue(binaryAddCacheLocal);
        binaryAddCacheLocal = binaryAddCacheQueue_.template DeQue<float>();
        CopyDgammaToGm(binaryAddCacheLocal, startOffset, currentCols, binaryAddK_ * vlFp32_);
        binaryAddCacheQueue_.FreeTensor(binaryAddCacheLocal);
    }

    __aicore__ inline void HandlingLargeRows2KUnAlign(uint32_t startOffset, uint32_t currentCols)
    {
        LocalTensor<float> binaryAddCacheLocal = binaryAddCacheQueue_.template AllocTensor<float>();
        // 处理row对齐尾块
        for (uint32_t i = 0; i < tiling_->tailBlockCountWithoutPadDG; i++) {
            uint32_t loopOffset = i * cols_ * rowsPerUB_;
            uint32_t inputOffset = startOffset + loopOffset;
            uint32_t tailDyXOffset = inputOffset + tiling_->powerOfTwoBlockCountDG * rowsPerUB_ * cols_;
            uint32_t tailRstdOffset = i * rowsPerUB_ + tiling_->powerOfTwoBlockCountDG * rowsPerUB_;
            CalcLargeRowsDgammaWithPad(inputOffset, tailDyXOffset, tailRstdOffset, currentCols, i, binaryAddCacheLocal);
        }
        // 处理row不对齐尾块
        for (uint32_t i = tiling_->tailBlockCountWithoutPadDG;
             i < tiling_->tailBlockCountWithoutPadDG + tiling_->tailBlockCountwithPadDG; i++) {
            uint32_t loopOffset = i * cols_ * rowsPerUB_;
            uint32_t inputOffset = startOffset + loopOffset;
            uint32_t tailDyXOffset = startOffset + tiling_->mainBlockCountDG * rowsPerUB_ * cols_;
            uint32_t tailRstdOffset = tiling_->mainBlockCountDG * rowsPerUB_;
            CalcLargeRowsDgammaWithPad(inputOffset, tailDyXOffset, tailRstdOffset, currentCols, i, binaryAddCacheLocal);
        }
        // 处理正常主块
        for (uint32_t i = tiling_->tailBlockCountWithoutPadDG + tiling_->tailBlockCountwithPadDG;
             i < tiling_->powerOfTwoBlockCountDG; i++) {
            uint32_t loopOffset = i * cols_ * rowsPerUB_;
            uint32_t inputOffset = startOffset + loopOffset;
            CalcLargeRowsDgamma(inputOffset, currentCols, i, binaryAddCacheLocal);
        }
        binaryAddCacheQueue_.EnQue(binaryAddCacheLocal);
        binaryAddCacheLocal = binaryAddCacheQueue_.template DeQue<float>();
        CopyDgammaToGm(binaryAddCacheLocal, startOffset, currentCols, binaryAddK_ * vlFp32_);
        binaryAddCacheQueue_.FreeTensor(binaryAddCacheLocal);
    }

    __aicore__ inline void Process()
    {
        if (coreIdx_ >= tiling_->usedCoreNumDG) {
            return;
        }
        int64_t inputOffset = 0;
        curCols_ = colsPerUB_;
        if ((coreIdx_ == tiling_->usedCoreNumDG - 1) && !tiling_->isMultiColset) {
            curCols_ = tiling_->colsPerTailCoreDG;
        }
        // main core calc
        bool isWithPad = tiling_->rowsTailDG == 0 ? false : true;
        for (int64_t curLoop = 0; curLoop < colsUbLoopCount_; curLoop++) {
            inputOffset = curLoop * colsPerUB_ + gmOffset_;
            CalcDgamma(inputOffset, curCols_, isWithPad);
        }
        // Handle tail
        inputOffset = tiling_->usedCoreNumDG * colsPerCore_ + coreIdx_ * vlFp32_;
        if (coreIdx_ < tiling_->tailCoreNumDG - 1 && tiling_->isMultiColset) {
            CalcDgamma(inputOffset, curCols_, isWithPad);
        } else if (coreIdx_ == tiling_->tailCoreNumDG - 1 && tiling_->isMultiColset) {
            CalcDgamma(inputOffset, tiling_->colsLastCoreDG, isWithPad);
        }
    }

    __aicore__ inline void ProcessWithLargeRows()
    {
        if (coreIdx_ >= tiling_->usedCoreNumDG) {
            return;
        }
        curCols_ = colsPerUB_;
        if ((coreIdx_ == tiling_->usedCoreNumDG - 1) && !tiling_->isMultiColset) {
            curCols_ = tiling_->colsPerTailCoreDG;
        }
        int64_t inputOffset = 0;
        int64_t outputOffset = 0;
        bool isPowerofTwoRows =
            (tiling_->tailBlockCountwithPadDG + tiling_->tailBlockCountWithoutPadDG) == 0 ? true : false;
        if (isPowerofTwoRows) {
            for (uint32_t curLoop = 0; curLoop < colsUbLoopCount_; curLoop++) {
                outputOffset = curLoop * colsPerUB_ + gmOffset_;
                HandlingLargeRows2KAlign(outputOffset, curCols_);
            }
            outputOffset = tiling_->usedCoreNumDG * colsPerCore_ + coreIdx_ * vlFp32_;
            if (coreIdx_ < tiling_->tailCoreNumDG - 1 && tiling_->isMultiColset) {
                HandlingLargeRows2KAlign(outputOffset, curCols_);
            } else if (coreIdx_ == tiling_->tailCoreNumDG - 1 && tiling_->isMultiColset) {
                HandlingLargeRows2KAlign(outputOffset, tiling_->colsLastCoreDG);
            }
        } else {
            // 核间累加，2k不对齐场景
            for (uint32_t curLoop = 0; curLoop < colsUbLoopCount_; curLoop++) {
                uint32_t startOffset = curLoop * colsPerUB_ + gmOffset_;
                HandlingLargeRows2KUnAlign(startOffset, curCols_);
            }
            uint32_t startOffset = tiling_->usedCoreNumDG * colsPerCore_ + coreIdx_ * vlFp32_;
            if (coreIdx_ < tiling_->tailCoreNumDG - 1 && tiling_->isMultiColset) {
                HandlingLargeRows2KUnAlign(startOffset, curCols_);
            } else if (coreIdx_ == tiling_->tailCoreNumDG - 1 && tiling_->isMultiColset) {
                HandlingLargeRows2KUnAlign(startOffset, tiling_->colsLastCoreDG);
            }
        }
    }

private:
    TQue<QuePosition::VECIN, 1> dyQueue_;
    TQue<QuePosition::VECIN, 1> xQueue_;
    TQue<QuePosition::VECIN, 1> rstdQueue_;
    TQue<QuePosition::VECOUT, 1> dgammaQueue_;
    TQue<QuePosition::VECOUT, 1> dgammaQueue1_;
    TQue<QuePosition::VECOUT, 1> binaryAddCacheQueue_;

    GlobalTensor<DTYPE_DY> dyGm_;
    GlobalTensor<DTYPE_X> xGm_;
    GlobalTensor<DTYPE_RSTD> rstdGm_;
    GlobalTensor<float> dgammaGm_;

    uint32_t blockSize_;
    uint32_t vlFp32_;
    uint32_t coreIdx_;
    int64_t cols_;
    int64_t rows_;
    uint32_t colsPerUB_;
    uint32_t colsPerCore_;
    uint32_t rowsPerUB_;
    uint32_t colsUbLoopCount_;
    uint32_t colsPerLoopAlign_;
    uint32_t binaryAddK_;
    uint32_t tailDataOffset_;
    uint64_t gmOffset_;
    uint32_t curCols_;
    TPipe* Ppipe_ = nullptr;

    const RmsNormGradRegbaseTilingData* tiling_;
};
} // namespace RmsNormGrad
#endif // RMS_NORM_GRAD_REGBASE_DGAMMA_H