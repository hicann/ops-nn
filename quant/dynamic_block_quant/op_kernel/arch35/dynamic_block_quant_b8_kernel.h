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
 * \file dynamic_block_quant_b8_kernel.h
 * \brief
 */

#ifndef DYNAMIC_BLOCK_QUANT_B8_H
#define DYNAMIC_BLOCK_QUANT_B8_H
#include "kernel_operator.h"
#include "../inc/platform.h"

#include "dynamic_block_quant_common.h"
namespace DynamicBlockQuant {
using namespace AscendC;

template <typename T, typename U, int64_t RMode>
class DynamicBlockQuantB8 {
public:
    __aicore__ inline DynamicBlockQuantB8(TPipe* pipe)
    {
        Ppipe = pipe;
    }
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR scale, const DynamicBlockQuantTilingData& tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(
        int32_t blockRow, int32_t blockCol, int32_t blockRowNum, int32_t blockColNum, int64_t baseXGmOffset);
    __aicore__ inline void CopyOut(
        int32_t blockRow, int32_t blockCol, int32_t blockRowNum, int32_t blockColNum, int64_t baseXGmOffset,
        int64_t baseScaleOffset);
    __aicore__ inline void Compute(int32_t blockRow, int32_t blockCol, int32_t blockRowNum, int32_t blockColNum);
    __aicore__ inline void ParseTilingData(const DynamicBlockQuantTilingData& tilingData);
    __aicore__ inline void ComputeXTmpVF(
        __local_mem__ T* xLocal, __local_mem__ T* xLocalTmp, int32_t blockRowNum, int32_t blockColNum,
        int32_t blockColNumAlign);

    __aicore__ inline void ComputeScaleVF(
        __local_mem__ T* scaleLocalTmp, __local_mem__ float* scaleLocal, int32_t blockRow, int32_t blockCol);
    __aicore__ inline void ComputeOutVF(
        __local_mem__ T* xLocal, __local_mem__ float* scaleLocal, __local_mem__ U* outLocal, int32_t blockRow,
        int32_t blockCol, int32_t blockRowNum, int32_t blockColNum, int32_t blockColNumAlign);

private:
    TPipe* Ppipe = nullptr;
    TQue<QuePosition::VECIN, 1> inQueue_;
    GlobalTensor<T> xGm_;
    TQue<QuePosition::VECOUT, 1> outQueue_;
    GlobalTensor<U> yGm_;
    TQue<QuePosition::VECOUT, 1> scaleQueue_;
    GlobalTensor<float> scaleGm_;

    TBuf<QuePosition::VECCALC> xLocalBuffer_;
    TBuf<QuePosition::VECCALC> scaleBuffer_;

    int64_t blockIdx_ = 0;
    float fpMaxValue_ = 0.0f;

    int64_t totalCoreNum_ = 0;
    int64_t ubSize_ = 0;
    int64_t vfLen_ = 0;
    float minScale_ = 0.0;
    int64_t roundMode_ = 0;
    int64_t dstType_ = 0;
    int64_t blockSizeRow_ = 0;
    int64_t blockSizeCol_ = 0;
    int64_t rowNum_ = 0;
    int64_t colNum_ = 0;
    int64_t rowBlockLoopNum_ = 0;
    int64_t colBlockLoopNum_ = 0;
    int64_t rowUbBlockLoopNum_ = 0;
    int64_t colUbBlockLoopNum_ = 0;
    int64_t rowUbFactor_ = 0;
    int64_t colUbFactor_ = 0;
    int64_t usedCoreNum_ = 0;
    int64_t rowTileNum_ = 0;
    int64_t colTileNum_ = 0;
    int64_t normalCoreRowTileNum_ = 0;
    int64_t normalCoreColTileNum_ = 0;
    int64_t tailCoreRowTileNum_ = 0;
    int64_t tailCoreColTileNum_ = 0;
    int64_t rowNormalCoreNum_ = 0;
    int64_t colNormalCoreNum_ = 0;
    int64_t rowTailCoreNum_ = 0;
    int64_t colTailCoreNum_ = 0;

    int64_t rowCoreIdx_ = 0;
    int64_t colCoreIdx_ = 0;

    bool isRowTailCore_ = false;
    bool isColTailCore_ = false;

    int64_t rowCoreTileNum_ = 0;
    int64_t colCoreTileNum_ = 0;
    int64_t rowUbLoop_ = 0;
    int64_t colUbLoop_ = 0;
    int64_t coreRowNum_ = 0;
    int64_t coreColNum_ = 0;

    int64_t blockSizeColAlign_ = 0;

    static constexpr AscendC::MicroAPI::CastTrait castTrait32toh8 = []() {
        if constexpr (RMode == 1 || RMode == 4) {
            return AscendC::MicroAPI::CastTrait{
                AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
                AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};
        } else if constexpr (RMode == 7) {
            return AscendC::MicroAPI::CastTrait{
                AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
                AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_HYBRID};
        }
    }();
};

template <typename T, typename U, int64_t RMode>
__aicore__ inline void DynamicBlockQuantB8<T, U, RMode>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR scale, const DynamicBlockQuantTilingData& tilingData)
{
    ParseTilingData(tilingData);

    blockIdx_ = GetBlockIdx();
    if (blockIdx_ >= usedCoreNum_) {
        return;
    }

    Ppipe->InitBuffer(inQueue_, 1, rowUbFactor_ * colUbFactor_ * sizeof(T));
    Ppipe->InitBuffer(outQueue_, 1, rowUbFactor_ * colUbFactor_ * sizeof(U));
    Ppipe->InitBuffer(scaleQueue_, 1, (colUbBlockLoopNum_ + 64 - 1) / 64 * 64 * rowUbBlockLoopNum_ * sizeof(float));

    Ppipe->InitBuffer(xLocalBuffer_, rowUbFactor_ * colUbFactor_ * sizeof(T));
    Ppipe->InitBuffer(scaleBuffer_, (rowUbBlockLoopNum_ * colUbBlockLoopNum_) * 8 * sizeof(T));

    rowCoreIdx_ = blockIdx_ / colTileNum_;
    colCoreIdx_ = blockIdx_ % colTileNum_;
    isRowTailCore_ = (rowCoreIdx_ >= rowNormalCoreNum_);
    isColTailCore_ = (colCoreIdx_ >= colNormalCoreNum_);
    blockSizeColAlign_ = (blockSizeCol_ + 128 - 1) / 128 * 128;

    int64_t xRowGmOffset = 0;
    int64_t xColGmOffset = 0;
    int64_t xGmOffset = 0;
    int64_t scaleGmRowOffset = 0;
    int64_t scaleGmColOffset = 0;
    int64_t scaleGmOffset = 0;

    if constexpr (IsSameType<U, fp8_e5m2_t>::value) {
        fpMaxValue_ = FP8_E5M2_MAX_VALUE;
    } else if constexpr (IsSameType<U, fp8_e4m3fn_t>::value) {
        fpMaxValue_ = FP8_E4M3_MAX_VALUE;
    } else if constexpr (IsSameType<U, hifloat8_t>::value) {
        fpMaxValue_ = HIFP8_MAX_VALUE;
    }

    if (isRowTailCore_) {
        rowCoreTileNum_ = tailCoreRowTileNum_;
        rowUbBlockLoopNum_ = rowUbBlockLoopNum_ > rowCoreTileNum_ ? rowCoreTileNum_ : rowUbBlockLoopNum_;
        rowUbLoop_ = tailCoreRowTileNum_ / rowUbBlockLoopNum_;
        xRowGmOffset = rowNormalCoreNum_ * normalCoreRowTileNum_ * colNum_ +
                       (rowCoreIdx_ - rowNormalCoreNum_) * tailCoreRowTileNum_ * colNum_;
        scaleGmRowOffset =
            rowNormalCoreNum_ * normalCoreRowTileNum_ + (rowCoreIdx_ - rowNormalCoreNum_) * tailCoreRowTileNum_;
        coreRowNum_ = rowCoreIdx_ + 1 == rowTileNum_ ?
                          rowNum_ - (rowNormalCoreNum_ * normalCoreRowTileNum_ +
                                     (rowCoreIdx_ - rowNormalCoreNum_) * tailCoreRowTileNum_) *
                                        blockSizeRow_ :
                          tailCoreRowTileNum_ * blockSizeRow_;
    } else {
        rowCoreTileNum_ = normalCoreRowTileNum_;
        rowUbLoop_ = rowCoreTileNum_ / rowUbBlockLoopNum_;
        xRowGmOffset = rowCoreIdx_ * normalCoreRowTileNum_ * colNum_;
        scaleGmRowOffset = rowCoreIdx_ * normalCoreRowTileNum_;
        coreRowNum_ = rowCoreIdx_ + 1 == rowTileNum_ ? rowNum_ - rowCoreIdx_ * normalCoreRowTileNum_ * blockSizeRow_ :
                                                       normalCoreRowTileNum_ * blockSizeRow_;
    }
    if (isColTailCore_) {
        colCoreTileNum_ = tailCoreColTileNum_;
        colUbBlockLoopNum_ = colUbBlockLoopNum_ > colCoreTileNum_ ? colCoreTileNum_ : colUbBlockLoopNum_;
        colUbLoop_ = colCoreTileNum_ / colUbBlockLoopNum_;
        xColGmOffset = colNormalCoreNum_ * normalCoreColTileNum_ * blockSizeCol_ +
                       (colCoreIdx_ - colNormalCoreNum_) * tailCoreColTileNum_ * blockSizeCol_;
        scaleGmColOffset =
            colNormalCoreNum_ * normalCoreColTileNum_ + (colCoreIdx_ - colNormalCoreNum_) * tailCoreColTileNum_;
        coreColNum_ = colCoreIdx_ + 1 == colTileNum_ ?
                          colNum_ - (colNormalCoreNum_ * normalCoreColTileNum_ +
                                     (colCoreIdx_ - colNormalCoreNum_) * tailCoreColTileNum_) *
                                        blockSizeCol_ :
                          tailCoreColTileNum_ * blockSizeCol_;
    } else {
        colCoreTileNum_ = normalCoreColTileNum_;
        colUbLoop_ = colCoreTileNum_ / colUbBlockLoopNum_;
        xColGmOffset = colCoreIdx_ * normalCoreColTileNum_ * blockSizeCol_;
        scaleGmColOffset = colCoreIdx_ * normalCoreColTileNum_;
        coreColNum_ = colCoreIdx_ + 1 == colTileNum_ ? colNum_ - colCoreIdx_ * normalCoreColTileNum_ * blockSizeCol_ :
                                                       normalCoreColTileNum_ * blockSizeCol_;
    }
    xGmOffset = xRowGmOffset * blockSizeCol_ + xColGmOffset;
    scaleGmOffset = scaleGmRowOffset * colBlockLoopNum_ + scaleGmColOffset;

    xGm_.SetGlobalBuffer((__gm__ T*)x + xGmOffset);
    yGm_.SetGlobalBuffer((__gm__ U*)y + xGmOffset);
    scaleGm_.SetGlobalBuffer((__gm__ float*)scale + scaleGmOffset);
}

template <typename T, typename U, int64_t RMode>
__aicore__ inline void DynamicBlockQuantB8<T, U, RMode>::ParseTilingData(const DynamicBlockQuantTilingData& tilingData)
{
    totalCoreNum_ = tilingData.totalCoreNum;
    ubSize_ = tilingData.ubSize;
    vfLen_ = tilingData.vfLen;
    minScale_ = tilingData.minScale;
    roundMode_ = tilingData.roundMode;
    dstType_ = tilingData.dstType;
    blockSizeRow_ = tilingData.blockSizeRow;
    blockSizeCol_ = tilingData.blockSizeCol;
    rowNum_ = tilingData.rowNum;
    colNum_ = tilingData.colNum;
    rowBlockLoopNum_ = tilingData.rowBlockLoopNum;
    colBlockLoopNum_ = tilingData.colBlockLoopNum;
    rowUbBlockLoopNum_ = tilingData.rowUbBlockLoopNum;
    colUbBlockLoopNum_ = tilingData.colUbBlockLoopNum;
    rowUbFactor_ = tilingData.rowUbFactor;
    colUbFactor_ = tilingData.colUbFactor;
    usedCoreNum_ = tilingData.usedCoreNum;
    rowTileNum_ = tilingData.rowTileNum;
    colTileNum_ = tilingData.colTileNum;
    normalCoreRowTileNum_ = tilingData.normalCoreRowTileNum;
    normalCoreColTileNum_ = tilingData.normalCoreColTileNum;
    tailCoreRowTileNum_ = tilingData.tailCoreRowTileNum;
    tailCoreColTileNum_ = tilingData.tailCoreColTileNum;
    rowNormalCoreNum_ = tilingData.rowNormalCoreNum;
    colNormalCoreNum_ = tilingData.colNormalCoreNum;
    rowTailCoreNum_ = tilingData.rowTailCoreNum;
    colTailCoreNum_ = tilingData.colTailCoreNum;
}

template <typename T, typename U, int64_t RMode>
__aicore__ inline void DynamicBlockQuantB8<T, U, RMode>::Process()
{
    if (blockIdx_ >= usedCoreNum_) {
        return;
    }

    for (int64_t rowUbLoopIdx = 0; rowUbLoopIdx < rowUbLoop_; rowUbLoopIdx++) {
        for (int64_t colUbLoopIdx = 0; colUbLoopIdx < colUbLoop_; colUbLoopIdx++) {
            int32_t blockRow = rowUbLoopIdx == rowUbLoop_ - 1 ? rowCoreTileNum_ - rowUbLoopIdx * rowUbBlockLoopNum_ :
                                                                rowUbBlockLoopNum_;
            int32_t blockCol = colUbLoopIdx == colUbLoop_ - 1 ? colCoreTileNum_ - colUbLoopIdx * colUbBlockLoopNum_ :
                                                                colUbBlockLoopNum_;
            int32_t blockRowNum = rowUbLoopIdx == rowUbLoop_ - 1 ?
                                      coreRowNum_ - rowUbLoopIdx * rowUbBlockLoopNum_ * blockSizeRow_ :
                                      blockSizeRow_;
            int32_t blockColNum = colUbLoopIdx == colUbLoop_ - 1 ?
                                      coreColNum_ - colUbLoopIdx * colUbBlockLoopNum_ * blockSizeCol_ :
                                      blockSizeCol_;
            int64_t baseXGmOffset = rowUbLoopIdx * rowUbBlockLoopNum_ * blockSizeRow_ * colNum_ +
                                    colUbLoopIdx * colUbBlockLoopNum_ * blockSizeCol_;
            int64_t baseScaleOffset =
                rowUbLoopIdx * rowUbBlockLoopNum_ * colBlockLoopNum_ + colUbLoopIdx * colUbBlockLoopNum_;

            CopyIn(blockRow, blockCol, blockRowNum, blockColNum, baseXGmOffset);
            Compute(blockRow, blockCol, blockRowNum, blockColNum);
            CopyOut(blockRow, blockCol, blockRowNum, blockColNum, baseXGmOffset, baseScaleOffset);
        }
    }
}

template <typename T, typename U, int64_t RMode>
__aicore__ inline void DynamicBlockQuantB8<T, U, RMode>::CopyIn(
    int32_t blockRow, int32_t blockCol, int32_t blockRowNum, int32_t blockColNum, int64_t baseXGmOffset)
{
    LocalTensor<T> inLocal = inQueue_.AllocTensor<T>();

    int64_t gmOffset = 0;
    int64_t ubOffset = 0;

    for (int32_t rowIdx = 0; rowIdx < blockRow; rowIdx++) {
        for (int32_t colIdx = 0; colIdx < blockCol; colIdx++) {
            ubOffset = (rowIdx * blockCol + colIdx) * blockSizeColAlign_ * blockSizeRow_;
            gmOffset =
                baseXGmOffset + (rowIdx * rowBlockLoopNum_ + colIdx * colBlockLoopNum_) * blockSizeCol_ * blockSizeRow_;

            int32_t rowBlockSize = rowIdx == blockRow - 1 ? blockRowNum - rowIdx * blockSizeRow_ : blockSizeRow_;
            int32_t cowBlockSize = colIdx == blockCol - 1 ? blockColNum - colIdx * blockSizeCol_ : blockSizeCol_;
            int32_t cowBlockSizeAlign = (cowBlockSize + 16 - 1) / 16 * 16;

            DataCopyExtParams copyParams = {
                static_cast<uint16_t>(rowBlockSize), static_cast<uint32_t>(cowBlockSize * sizeof(T)),
                static_cast<uint32_t>((colNum_ - cowBlockSize) * sizeof(T)), 0, 0};
            DataCopyPadExtParams<T> padParams{true, 0, static_cast<uint8_t>(cowBlockSizeAlign - cowBlockSize), 0};
            DataCopyPad(inLocal[ubOffset], xGm_[gmOffset], copyParams, padParams);
        }
    }
    inQueue_.EnQue(inLocal);
}

template <typename T, typename U, int64_t RMode>
__aicore__ inline void DynamicBlockQuantB8<T, U, RMode>::Compute(
    int32_t blockRow, int32_t blockCol, int32_t blockRowNum, int32_t blockColNum)
{
    int32_t blockColNumAlign = (blockColNum + 16 - 1) / 16 * 16;
    LocalTensor<T> inLocal = inQueue_.DeQue<T>();
    LocalTensor<T> xLocalTmp = xLocalBuffer_.Get<T>();
    LocalTensor<T> scaleLocalTmp = scaleBuffer_.Get<T>();

    __local_mem__ T* xLocalAddr = (__local_mem__ T*)inLocal.GetPhyAddr();
    __local_mem__ T* xLocalTmpAddr = (__local_mem__ T*)xLocalTmp.GetPhyAddr();

    ComputeXTmpVF(xLocalAddr, xLocalTmpAddr, blockRowNum, blockColNum, blockColNumAlign);

    for (int32_t rowIdx = 0; rowIdx < blockRow; rowIdx++) {
        for (int32_t colIdx = 0; colIdx < blockCol; colIdx++) {
            int32_t ubOffset = (rowIdx * blockCol + colIdx) * blockSizeColAlign_ * blockSizeRow_;
            int32_t scaleUbOffset = (rowIdx * blockCol + colIdx) * 16;
            int32_t rowBlockSize = rowIdx == blockRow - 1 ? blockRowNum - rowIdx * blockSizeRow_ : blockSizeRow_;
            int32_t cowBlockSize = colIdx == blockCol - 1 ? blockColNum - colIdx * blockSizeCol_ : blockSizeCol_;
            int32_t cowBlockSizeAlign = (cowBlockSize + 16 - 1) / 16 * 16;
            AscendC::ReduceMax<half>(
                scaleLocalTmp[scaleUbOffset], xLocalTmp[ubOffset], xLocalTmp[ubOffset],
                rowBlockSize * cowBlockSizeAlign);
        }
    }

    LocalTensor<float> scaleLocal = scaleQueue_.AllocTensor<float>();
    __local_mem__ T* scaleLocalTmpAddr = (__local_mem__ T*)scaleLocalTmp.GetPhyAddr();
    __local_mem__ float* scaleLocalAddr = (__local_mem__ float*)scaleLocal.GetPhyAddr();
    ComputeScaleVF(scaleLocalTmpAddr, scaleLocalAddr, blockRow, blockCol);
    LocalTensor<U> outLocal = outQueue_.AllocTensor<U>();
    __local_mem__ U* outLocalAddr = (__local_mem__ U*)outLocal.GetPhyAddr();

    ComputeOutVF(
        xLocalAddr, scaleLocalAddr, outLocalAddr, blockRow, blockCol, blockRowNum, blockColNum, blockColNumAlign);
    inQueue_.FreeTensor(inLocal);
    outQueue_.EnQue(outLocal);
    scaleQueue_.EnQue(scaleLocal);
}
template <typename T, typename U, int64_t RMode>
__aicore__ inline void DynamicBlockQuantB8<T, U, RMode>::CopyOut(
    int32_t blockRow, int32_t blockCol, int32_t blockRowNum, int32_t blockColNum, int64_t baseXGmOffset,
    int64_t baseScaleOffset)
{
    LocalTensor<U> outLocal = outQueue_.DeQue<U>();
    LocalTensor<float> scaleLocal = scaleQueue_.DeQue<float>();

    int64_t outGmOffset = 0;
    int64_t outUbOffset = 0;
    int64_t scaleGmOffset = 0;
    int64_t scaleUbOffset = 0;

    for (int32_t rowIdx = 0; rowIdx < blockRow; rowIdx++) {
        for (int32_t colIdx = 0; colIdx < blockCol; colIdx++) {
            outUbOffset = (rowIdx * blockSizeColAlign_ + colIdx) * blockSizeColAlign_ * blockSizeRow_;
            outGmOffset =
                baseXGmOffset + (rowIdx * rowBlockLoopNum_ + colIdx * colBlockLoopNum_) * blockSizeCol_ * blockSizeRow_;
            scaleUbOffset = (rowIdx * blockCol + colIdx) * 8;
            scaleGmOffset = baseScaleOffset + (rowIdx * rowBlockLoopNum_ + colIdx);

            int32_t rowBlockSize = rowIdx == blockRow - 1 ? blockRowNum - rowIdx * blockSizeRow_ : blockSizeRow_;
            int32_t cowBlockSize = colIdx == blockCol - 1 ? blockColNum - colIdx * blockSizeCol_ : blockSizeCol_;
            int32_t cowBlockSizeAlign = (cowBlockSize + 32 - 1) / 32 * 32;

            DataCopyExtParams outCopyParams = {
                static_cast<uint16_t>(rowBlockSize), static_cast<uint32_t>(cowBlockSize * sizeof(U)),
                static_cast<uint32_t>((128 - cowBlockSizeAlign) / 32),
                static_cast<uint32_t>((colNum_ - cowBlockSize) * sizeof(U)), 0};
            DataCopyExtParams scaleCopyParams = {1, static_cast<uint32_t>(1 * sizeof(float)), 0, 0, 0};
            DataCopyPad(yGm_[outGmOffset], outLocal[outUbOffset], outCopyParams);
            DataCopyPad(scaleGm_[scaleGmOffset], scaleLocal[scaleUbOffset], scaleCopyParams);
        }
    }

    outQueue_.FreeTensor(outLocal);
    scaleQueue_.FreeTensor(scaleLocal);
}

template <typename T, typename U, int64_t RMode>
__aicore__ inline void DynamicBlockQuantB8<T, U, RMode>::ComputeXTmpVF(
    __local_mem__ T* xLocal, __local_mem__ T* xLocalTmp, int32_t blockRowNum, int32_t blockColNum,
    int32_t blockColNumAlign)
{
    uint32_t dtypeSize = sizeof(T);
    uint16_t VL = AscendC::VECTOR_REG_WIDTH / dtypeSize;
    uint32_t count = blockRowNum * blockColNumAlign;
    uint16_t vfLoop = (blockRowNum * blockColNumAlign + VL - 1) / VL;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<T> vreg1;
        AscendC::MicroAPI::RegTensor<T> vreg2;
        AscendC::MicroAPI::RegTensor<T> vreg3;

        AscendC::MicroAPI::MaskReg preg0;
        AscendC::MicroAPI::MaskReg preg1;
        AscendC::MicroAPI::MaskReg preg2;

        for (uint16_t i = 0; i < vfLoop; i++) {
            preg0 = AscendC::MicroAPI::UpdateMask<T>(count);
            AscendC::MicroAPI::DataCopy(vreg1, xLocal + i * VL);
            AscendC::MicroAPI::Muls(vreg2, vreg1, static_cast<T>(0), preg0);
            AscendC::MicroAPI::Compare<T, CMPMODE::NE>(preg1, vreg2, vreg2, preg0);
            AscendC::MicroAPI::MaskNot(preg2, preg1, preg0);
            AscendC::MicroAPI::Abs(vreg3, vreg1, preg2);
            AscendC::MicroAPI::DataCopy(xLocalTmp + i * VL, vreg3, preg0);
        }
    }
}

template <typename T, typename U, int64_t RMode>
__aicore__ inline void DynamicBlockQuantB8<T, U, RMode>::ComputeScaleVF(
    __local_mem__ T* scaleLocalTmp, __local_mem__ float* scaleLocal, int32_t blockRow, int32_t blockCol)
{
    uint32_t size = blockRow * blockCol * 16;
    uint32_t dtypeSize = sizeof(float);
    uint16_t VL = AscendC::VECTOR_REG_WIDTH / dtypeSize;
    uint16_t vfLoopNum = (size + VL - 1) / VL;

    static constexpr AscendC::MicroAPI::DivSpecificMode mode = {AscendC::MicroAPI::MaskMergeMode::ZEROING, false};

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<T> vreg1;
        AscendC::MicroAPI::RegTensor<float> vreg2;
        AscendC::MicroAPI::RegTensor<float> vreg3;
        AscendC::MicroAPI::RegTensor<float> vreg4;
        AscendC::MicroAPI::RegTensor<float> vreg5;
        AscendC::MicroAPI::RegTensor<float> vreg6;

        AscendC::MicroAPI::MaskReg preg0;
        AscendC::MicroAPI::MaskReg preg1;

        preg0 = AscendC::MicroAPI::CreateMask<float>();

        AscendC::MicroAPI::Duplicate(vreg3, fpMaxValue_);

        for (uint16_t i = 0; i < vfLoopNum; i++) {
            preg0 = AscendC::MicroAPI::UpdateMask<float>(size);
            AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vreg1, scaleLocalTmp + i * VL);
            AscendC::MicroAPI::Cast<float, T, castTrait0>(vreg2, vreg1, preg0);
            AscendC::MicroAPI::CompareScalar<float, CMPMODE::NE>(preg1, vreg2, (float)0.0, preg0);
            AscendC::MicroAPI::Div<float, &mode>(vreg5, vreg3, vreg2, preg1);
            AscendC::MicroAPI::Maxs(vreg6, vreg5, minScale_, preg0);
            AscendC::MicroAPI::DataCopy(scaleLocal + i * VL, vreg6, preg0);
        }
    }
}

template <typename T, typename U, int64_t RMode>
__aicore__ inline void DynamicBlockQuantB8<T, U, RMode>::ComputeOutVF(
    __local_mem__ T* xLocal, __local_mem__ float* scaleLocal, __local_mem__ U* outLocal, int32_t blockRow,
    int32_t blockCol, int32_t blockRowNum, int32_t blockColNum, int32_t blockColNumAlign)
{
    uint32_t dtypeSize = sizeof(float);
    uint16_t VL = AscendC::VECTOR_REG_WIDTH / dtypeSize;
    uint16_t blockNum = blockRow * blockCol;
    uint16_t rowVfLoopNum = blockRowNum;
    uint16_t colVfLoopNum = (blockColNumAlign + VL - 1) / VL;
    uint32_t fp8BlockColNumAlign = (blockColNum + 128 - 1) / 128 * 128;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<T> vreg1;
        AscendC::MicroAPI::RegTensor<float> vreg2;
        AscendC::MicroAPI::RegTensor<float> vreg3;
        AscendC::MicroAPI::RegTensor<float> vreg4;
        AscendC::MicroAPI::RegTensor<float> vreg5;
        AscendC::MicroAPI::RegTensor<float> vreg7;
        AscendC::MicroAPI::RegTensor<U> vreg6;

        AscendC::MicroAPI::MaskReg preg0;
        AscendC::MicroAPI::MaskReg preg1;

        preg0 = AscendC::MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();

        for (uint16_t scaleIdx = 0; scaleIdx < blockNum; scaleIdx++) {
            AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
                vreg2, scaleLocal + scaleIdx * 8);
            for (uint16_t i = 0; i < rowVfLoopNum; i++) {
                for (uint16_t j = 0; j < colVfLoopNum; j++) {
                    AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                        vreg1, xLocal + i * blockColNumAlign + j * VL);
                    AscendC::MicroAPI::Cast<float, T, castTrait0>(vreg4, vreg1, preg0);
                    AscendC::MicroAPI::Mul(vreg5, vreg4, vreg2, preg0);

                    AscendC::MicroAPI::Muls(vreg3, vreg4, 0.0f, preg0);
                    AscendC::MicroAPI::Compare<float, CMPMODE::NE>(preg1, vreg3, vreg3, preg0);
                    AscendC::MicroAPI::Select(vreg7, vreg4, vreg5, preg1);

                    if constexpr (IsSameType<U, hifloat8_t>::value) {
                        AscendC::MicroAPI::Cast<U, float, castTrait32toh8>(vreg6, vreg7, preg0);
                    } else {
                        AscendC::MicroAPI::Cast<U, float, castTrait32tofp8>(vreg6, vreg7, preg0);
                    }
                    MicroAPI::DataCopy<U, MicroAPI::StoreDist::DIST_PACK4_B32>(
                        outLocal + i * fp8BlockColNumAlign + j * VL, vreg6, preg0);
                }
            }
        }
    }
}

} // namespace DynamicBlockQuant
#endif // DYNAMIC_BLOCK_QUANT_B8_H