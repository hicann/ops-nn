/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dynamic_block_quant_small_blocksize_kernel.h
 * \brief
 */

#ifndef DYNAMIC_BLOCK_QUANT_SMALL_BLOCKSIZE_H
#define DYNAMIC_BLOCK_QUANT_SMALL_BLOCKSIZE_H
#include "kernel_operator.h"
#include "../inc/platform.h"

#include "dynamic_block_quant_common.h"
namespace DynamicBlockQuant {
#define FLOAT_OVERFLOW_MODE_CTRL 60
using namespace AscendC;

template <typename T, typename U, int64_t RMode>
class DynamicBlockQuantSmallBlockSize {
public:
    __aicore__ inline DynamicBlockQuantSmallBlockSize(TPipe* pipe) { Ppipe = pipe; }
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR scale, const DynamicBlockQuantTilingData& tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t blockRow, int32_t blockCol, int64_t blockRowNum, int64_t blockColNum,
                                  int64_t baseXGmOffset, int64_t nowRowBlock);
    __aicore__ inline void CopyOut(int32_t blockRow, int32_t blockCol, int64_t blockRowNum, int64_t blockColNum,
                                   int64_t baseXGmOffset, int64_t baseScaleOffset, int64_t nowRowBlock);
    __aicore__ inline void Compute(int32_t blockRow, int32_t blockCol, int64_t blockRowNum, int64_t blockColNum,
                                   int64_t nowRowBlock);
    __aicore__ inline void ComputeVF(int64_t rowNum, int64_t colNum, __ubuf__ T* xAddr, __ubuf__ float* scaleOutAddr,
                                     __ubuf__ U* yOutAddr);
    __aicore__ inline void ParseTilingData(const DynamicBlockQuantTilingData& tilingData);

private:
    TPipe* Ppipe = nullptr;
    TQue<QuePosition::VECIN, DB_BUFFER> inQueue_;
    GlobalTensor<T> xGm_;
    TQue<QuePosition::VECOUT, DB_BUFFER> outQueue_;
    GlobalTensor<U> yGm_;
    TQue<QuePosition::VECOUT, DB_BUFFER> scaleQueue_;
    GlobalTensor<float> scaleGm_;

    int64_t blockIdx_ = 0;
    uint16_t maxValue_ = 0x7fff;       // FP16/BF16 使用，取绝对值掩码
    uint32_t maxValue32_ = 0x7fffffff; // float32 使用，取绝对值掩码
    uint32_t infValue_ = 0;
    float fp8MaxValue_ = 0;

    int64_t totalCoreNum_ = 0;
    int64_t ubSize_ = 0;
    int64_t vfLen_ = 0;
    float minScale_ = 0.0;
    int64_t roundMode_ = 0;
    int64_t dstType_ = 0;
    int64_t blockSizeRow_ = 0;
    int64_t blockSizeCol_ = 0;
    int64_t batchNum_ = 0;
    int64_t rowNum_ = 0;
    int64_t colNum_ = 0;
    int64_t singleBatchRowBlockLoopNum_ = 0;
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

    int32_t rowCoreIdx_ = 0;
    int32_t colCoreIdx_ = 0;

    bool isRowTailCore_ = false;
    bool isColTailCore_ = false;

    int32_t rowCoreTileNum_ = 0;
    int32_t colCoreTileNum_ = 0;
    int32_t rowUbLoop_ = 0;
    int32_t colUbLoop_ = 0;
    int32_t coreRowNum_ = 0;
    int32_t coreColNum_ = 0;

    // 所在block
    int64_t preRowBlockNum_ = 0;
    // 所在batch
    int64_t preBatch_ = 0;
    int64_t preSingleBatchBlock_ = 0;
    // 行尾块行数
    int64_t tailBlockSizeRow_ = 0;
    // 目标数据类型的最大值
    float dstTypeMax_ = 0;
    static constexpr AscendC::MicroAPI::CastTrait castTrait32toh8 = []() {
        if constexpr (RMode == 1 || RMode == 4) {
            return AscendC::MicroAPI::CastTrait{AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT,
                                                AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};
        } else if constexpr (RMode == 7) {
            return AscendC::MicroAPI::CastTrait{AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT,
                                                AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_HYBRID};
        }
    }();
};

template <typename T, typename U, int64_t RMode>
__aicore__ inline void DynamicBlockQuantSmallBlockSize<T, U, RMode>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR scale,
                                                                          const DynamicBlockQuantTilingData& tilingData)
{
#if (__NPU_ARCH__ == 3510) && (DTYPE_Y != DT_INT8)
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
#endif
    ParseTilingData(tilingData);

    blockIdx_ = GetBlockIdx();
    if (blockIdx_ >= usedCoreNum_) {
        return;
    }

    Ppipe->InitBuffer(inQueue_, DB_BUFFER, rowUbFactor_ * colUbFactor_ * sizeof(T));
    Ppipe->InitBuffer(outQueue_, DB_BUFFER, rowUbFactor_ * colUbFactor_ * sizeof(U));
    Ppipe->InitBuffer(scaleQueue_, DB_BUFFER, colUbBlockLoopNum_ * rowUbBlockLoopNum_ * BLOCK_BYTE_32);

    rowCoreIdx_ = blockIdx_ / colTileNum_;
    colCoreIdx_ = blockIdx_ % colTileNum_;
    isRowTailCore_ = (rowCoreIdx_ >= rowNormalCoreNum_);
    isColTailCore_ = (colCoreIdx_ >= colNormalCoreNum_);

    int64_t xRowGmOffset = 0;
    int64_t xColGmOffset = 0;
    int64_t xGmOffset = 0;
    int64_t scaleGmRowOffset = 0;
    int64_t scaleGmColOffset = 0;
    int64_t scaleGmOffset = 0;

    infValue_ = FP32_INF_VALUE;
    fp8MaxValue_ = GetDstTypeMaxValue<U>(dstTypeMax_);

    tailBlockSizeRow_ = rowNum_ % blockSizeRow_;
    if (tailBlockSizeRow_ == 0) {
        tailBlockSizeRow_ = blockSizeRow_;
    }

    if (isRowTailCore_) {
        rowCoreTileNum_ = tailCoreRowTileNum_;
        rowUbBlockLoopNum_ = rowUbBlockLoopNum_ > rowCoreTileNum_ ? rowCoreTileNum_ : rowUbBlockLoopNum_;
        rowUbLoop_ = Ceil(tailCoreRowTileNum_, rowUbBlockLoopNum_);
        preRowBlockNum_ = rowNormalCoreNum_ * normalCoreRowTileNum_ +
                          (rowCoreIdx_ - rowNormalCoreNum_) * tailCoreRowTileNum_;
    } else {
        rowCoreTileNum_ = normalCoreRowTileNum_;
        rowUbLoop_ = Ceil(rowCoreTileNum_, rowUbBlockLoopNum_);
        preRowBlockNum_ = rowCoreIdx_ * normalCoreRowTileNum_;
    }

    preBatch_ = preRowBlockNum_ / singleBatchRowBlockLoopNum_;
    preSingleBatchBlock_ = preRowBlockNum_ % singleBatchRowBlockLoopNum_;
    xRowGmOffset = preBatch_ * rowNum_ * colNum_ + preSingleBatchBlock_ * blockSizeRow_ * colNum_;
    scaleGmRowOffset = preRowBlockNum_;

    int64_t endRowBlockNum_ = preRowBlockNum_ + rowCoreTileNum_;
    int64_t endSingleBatchBlock = endRowBlockNum_ % singleBatchRowBlockLoopNum_;
    int64_t endBath = endRowBlockNum_ / singleBatchRowBlockLoopNum_;
    coreRowNum_ = (endBath - preBatch_) * rowNum_ + (endSingleBatchBlock - preSingleBatchBlock_) * blockSizeRow_;

    if (isColTailCore_) {
        colCoreTileNum_ = tailCoreColTileNum_;
        colUbBlockLoopNum_ = colUbBlockLoopNum_ > colCoreTileNum_ ? colCoreTileNum_ : colUbBlockLoopNum_;
        colUbLoop_ = Ceil(colCoreTileNum_, colUbBlockLoopNum_);
        xColGmOffset = colNormalCoreNum_ * normalCoreColTileNum_ * blockSizeCol_ +
                       (colCoreIdx_ - colNormalCoreNum_) * tailCoreColTileNum_ * blockSizeCol_;
        scaleGmColOffset = colNormalCoreNum_ * normalCoreColTileNum_ +
                           (colCoreIdx_ - colNormalCoreNum_) * tailCoreColTileNum_;
        coreColNum_ = colCoreIdx_ + 1 == colTileNum_ ?
                          colNum_ - (colNormalCoreNum_ * normalCoreColTileNum_ +
                                     (colCoreIdx_ - colNormalCoreNum_) * tailCoreColTileNum_) *
                                        blockSizeCol_ :
                          tailCoreColTileNum_ * blockSizeCol_;
    } else {
        colCoreTileNum_ = normalCoreColTileNum_;
        colUbLoop_ = Ceil(colCoreTileNum_, colUbBlockLoopNum_);
        xColGmOffset = colCoreIdx_ * normalCoreColTileNum_ * blockSizeCol_;
        scaleGmColOffset = colCoreIdx_ * normalCoreColTileNum_;
        coreColNum_ = colCoreIdx_ + 1 == colTileNum_ ? colNum_ - colCoreIdx_ * normalCoreColTileNum_ * blockSizeCol_ :
                                                       normalCoreColTileNum_ * blockSizeCol_;
    }
    xGmOffset = xRowGmOffset + xColGmOffset;
    scaleGmOffset = scaleGmRowOffset * colBlockLoopNum_ + scaleGmColOffset;

    xGm_.SetGlobalBuffer((__gm__ T*)x + xGmOffset);
    yGm_.SetGlobalBuffer((__gm__ U*)y + xGmOffset);
    scaleGm_.SetGlobalBuffer((__gm__ float*)scale + scaleGmOffset);
}

template <typename T, typename U, int64_t RMode>
__aicore__ inline void DynamicBlockQuantSmallBlockSize<T, U, RMode>::ParseTilingData(
    const DynamicBlockQuantTilingData& tilingData)
{
    totalCoreNum_ = tilingData.totalCoreNum;
    ubSize_ = tilingData.ubSize;
    vfLen_ = tilingData.vfLen;
    minScale_ = tilingData.minScale;
    roundMode_ = tilingData.roundMode;
    dstType_ = tilingData.dstType;
    blockSizeRow_ = tilingData.blockSizeRow;
    blockSizeCol_ = tilingData.blockSizeCol;
    dstTypeMax_ = tilingData.dstTypeMax;
    batchNum_ = tilingData.batchNum;
    rowNum_ = tilingData.rowNum;
    colNum_ = tilingData.colNum;
    singleBatchRowBlockLoopNum_ = tilingData.singleBatchRowBlockLoopNum;
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
__aicore__ inline void DynamicBlockQuantSmallBlockSize<T, U, RMode>::Process()
{
    if (blockIdx_ >= usedCoreNum_) {
        return;
    }
    for (int32_t rowUbLoopIdx = 0; rowUbLoopIdx < rowUbLoop_; rowUbLoopIdx++) {
        // 行block数
        int32_t blockRow = rowUbLoopIdx == rowUbLoop_ - 1 ? rowCoreTileNum_ - rowUbLoopIdx * rowUbBlockLoopNum_ :
                                                            rowUbBlockLoopNum_;
        int64_t nowRowBlockNum = preRowBlockNum_ + rowUbLoopIdx * rowUbBlockLoopNum_;
        int64_t nowBatch = nowRowBlockNum / singleBatchRowBlockLoopNum_;
        int64_t nowSingleBatchBlock = nowRowBlockNum % singleBatchRowBlockLoopNum_;

        int64_t endRowBlockNum = nowRowBlockNum + blockRow;
        int64_t endBatch = endRowBlockNum / singleBatchRowBlockLoopNum_;
        int64_t endSingleBatchBlock = endRowBlockNum % singleBatchRowBlockLoopNum_;

        int64_t blockRowNum = (endBatch - nowBatch) * rowNum_ +
                              (endSingleBatchBlock - nowSingleBatchBlock) * blockSizeRow_;
        for (int32_t colUbLoopIdx = 0; colUbLoopIdx < colUbLoop_; colUbLoopIdx++) {
            int32_t blockCol = colUbLoopIdx == colUbLoop_ - 1 ? colCoreTileNum_ - colUbLoopIdx * colUbBlockLoopNum_ :
                                                                colUbBlockLoopNum_;
            int64_t baseXGmOffset = ((nowBatch - preBatch_) * rowNum_ +
                                     (nowSingleBatchBlock - preSingleBatchBlock_) * blockSizeRow_) *
                                        colNum_ +
                                    colUbLoopIdx * colUbBlockLoopNum_ * blockSizeCol_;
            int64_t blockColNum = colUbLoopIdx == colUbLoop_ - 1 ?
                                      coreColNum_ - colUbLoopIdx * colUbBlockLoopNum_ * blockSizeCol_ :
                                      colUbBlockLoopNum_ * blockSizeCol_;
            int64_t baseScaleOffset = rowUbLoopIdx * rowUbBlockLoopNum_ * colBlockLoopNum_ +
                                      colUbLoopIdx * colUbBlockLoopNum_;
            CopyIn(blockRow, blockCol, blockRowNum, blockColNum, baseXGmOffset, nowSingleBatchBlock);
            Compute(blockRow, blockCol, blockRowNum, blockColNum, nowSingleBatchBlock);
            CopyOut(blockRow, blockCol, blockRowNum, blockColNum, baseXGmOffset, baseScaleOffset, nowSingleBatchBlock);
        }
    }
}

template <typename T, typename U, int64_t RMode>
__aicore__ inline void DynamicBlockQuantSmallBlockSize<T, U, RMode>::CopyIn(int32_t blockRow, int32_t blockCol,
                                                                            int64_t blockRowNum, int64_t blockColNum,
                                                                            int64_t baseXGmOffset, int64_t nowRowBlock)
{
    LocalTensor<T> inLocal = inQueue_.AllocTensor<T>();

    int64_t gmOffset = 0;
    int64_t gmRowOffset = 0;
    int64_t ubOffset = 0;
    int32_t rowBlockSize = 0;
    uint16_t inputColAlign = BLOCK_BYTE_32 / sizeof(T);
    for (int32_t rowIdx = 0; rowIdx < blockRow; rowIdx++) {
        int64_t tmpBlock = nowRowBlock + rowIdx;
        if (tmpBlock % singleBatchRowBlockLoopNum_ == singleBatchRowBlockLoopNum_ - 1) {
            rowBlockSize = tailBlockSizeRow_;
        } else {
            rowBlockSize = blockSizeRow_;
        }
        for (int32_t colIdx = 0; colIdx < blockCol; colIdx++) {
            gmOffset = baseXGmOffset + gmRowOffset + colIdx * blockSizeCol_;

            int32_t colBlockSize = colIdx == blockCol - 1 ? blockColNum - colIdx * blockSizeCol_ : blockSizeCol_;
            DataCopyExtParams copyParams = {static_cast<uint16_t>(rowBlockSize),
                                            static_cast<uint32_t>(colBlockSize * sizeof(T)),
                                            static_cast<uint32_t>((colNum_ - colBlockSize) * sizeof(T)), 0, 0};
            DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
            DataCopyPad<T, PaddingMode::Compact>(inLocal[ubOffset], xGm_[gmOffset], copyParams, padParams);
            int64_t blockTotalNum = rowBlockSize * colBlockSize;
            ubOffset = ubOffset + (blockTotalNum + inputColAlign - 1) / inputColAlign * inputColAlign;
        }
        gmRowOffset = gmRowOffset + rowBlockSize * colNum_;
    }
    inQueue_.EnQue(inLocal);
}

template <typename T, typename U, int64_t RMode>
__aicore__ inline void DynamicBlockQuantSmallBlockSize<T, U, RMode>::ComputeVF(int64_t rowNum, int64_t colNum,
                                                                               __ubuf__ T* xAddr,
                                                                               __ubuf__ float* scaleOutAddr,
                                                                               __ubuf__ U* yOutAddr)
{
    constexpr uint32_t vfLen = platform::GetVRegSize() / sizeof(T);
    constexpr uint32_t vfNum = platform::GetVRegSize() / sizeof(float);
    uint32_t xTotalNum = rowNum * colNum;
    uint16_t inputVfLoop = (xTotalNum + vfLen - 1) / vfLen;
    uint16_t vfLoop = (xTotalNum + vfNum - 1) / vfNum;
    uint32_t inputNum = xTotalNum;
    uint32_t outputNum = xTotalNum;
    uint32_t scaleNum = 1;
    static constexpr AscendC::MicroAPI::DivSpecificMode mode = {AscendC::MicroAPI::MaskMergeMode::ZEROING, false};
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<T> vreg1;
        AscendC::MicroAPI::RegTensor<T> vreg2;
        AscendC::MicroAPI::RegTensor<T> vreg3;
        AscendC::MicroAPI::RegTensor<T> yReg1;
        AscendC::MicroAPI::RegTensor<float> scaleRegTensor;
        AscendC::MicroAPI::RegTensor<float> fp8MaxReg;
        AscendC::MicroAPI::RegTensor<float> reciprocalScale;
        AscendC::MicroAPI::RegTensor<float> minScaleReg;
        AscendC::MicroAPI::RegTensor<float> yReg2;
        AscendC::MicroAPI::RegTensor<float> yReg3;
        AscendC::MicroAPI::RegTensor<float> yReg4;
        AscendC::MicroAPI::RegTensor<float> yReg5;
        AscendC::MicroAPI::RegTensor<int16_t> int16Reg;
        AscendC::MicroAPI::RegTensor<half> halfReg;
        AscendC::MicroAPI::RegTensor<U> outReg;
        AscendC::MicroAPI::MaskReg preg0;
        AscendC::MicroAPI::MaskReg scaleMaskReg;
        AscendC::MicroAPI::MaskReg yMaskReg0;
        AscendC::MicroAPI::MaskReg yMaskReg1;

        if constexpr (IsSameType<T, float>::value) {
            // ===== float32 类型处理 =====
            AscendC::MicroAPI::RegTensor<uint32_t> expMaxRegTensor;
            AscendC::MicroAPI::Duplicate((AscendC::MicroAPI::RegTensor<uint32_t>&)vreg2, maxValue32_);
            AscendC::MicroAPI::Duplicate(expMaxRegTensor, 0);

            AscendC::MicroAPI::MaskReg
                maskAll = AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();
            AscendC::MicroAPI::MaskReg
                ymaskAll = AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();

            // comput Max(x) - float32 使用 uint32_t 位运算
            for (uint16_t i = 0; i < inputVfLoop; i++) {
                preg0 = AscendC::MicroAPI::UpdateMask<T>(inputNum);
                AscendC::MicroAPI::DataCopy(vreg1, xAddr + i * vfLen);
                AscendC::MicroAPI::And((AscendC::MicroAPI::RegTensor<uint32_t>&)vreg3,
                                       (AscendC::MicroAPI::RegTensor<uint32_t>&)vreg1,
                                       (AscendC::MicroAPI::RegTensor<uint32_t>&)vreg2, preg0);
                AscendC::MicroAPI::Max<uint32_t, AscendC::MicroAPI::MaskMergeMode::MERGING>(
                    expMaxRegTensor, expMaxRegTensor, (AscendC::MicroAPI::RegTensor<uint32_t>&)vreg3, preg0);
            }

            // comput scale - float32 直接用位模式参与计算
            AscendC::MicroAPI::ReduceMax<uint32_t>(expMaxRegTensor, expMaxRegTensor, maskAll);
            AscendC::MicroAPI::Duplicate(expMaxRegTensor, expMaxRegTensor, maskAll);
            AscendC::MicroAPI::Duplicate(fp8MaxReg, fp8MaxValue_);
            AscendC::MicroAPI::Duplicate(reciprocalScale, 1.0f);
            AscendC::MicroAPI::Duplicate(minScaleReg, minScale_);

            AscendC::MicroAPI::Div<float, &mode>(reciprocalScale, reciprocalScale, minScaleReg, maskAll);

            // 位模式本身就是 float32 格式，直接参与计算
            AscendC::MicroAPI::Div<float, &mode>(scaleRegTensor, (AscendC::MicroAPI::RegTensor<float>&)expMaxRegTensor,
                                                 fp8MaxReg, ymaskAll);

            AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::LT>(
                scaleMaskReg, (AscendC::MicroAPI::RegTensor<uint32_t>&)scaleRegTensor, infValue_, ymaskAll);
            AscendC::MicroAPI::Min<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(scaleRegTensor, scaleRegTensor,
                                                                                     reciprocalScale, scaleMaskReg);
            AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                scaleOutAddr, scaleRegTensor, ymaskAll);

            // compute y - float32 直接加载，不需要 UNPACK
            for (uint16_t i = 0; i < vfLoop; i++) {
                yMaskReg0 = AscendC::MicroAPI::UpdateMask<float>(outputNum);
                AscendC::MicroAPI::DataCopy(yReg1, xAddr + i * vfNum);
                AscendC::MicroAPI::Div<float, &mode>(yReg3, yReg1, scaleRegTensor, yMaskReg0);

                if constexpr (IsSameType<U, hifloat8_t>::value) {
                    AscendC::MicroAPI::Cast<U, float, castTrait32toh8>(outReg, yReg3, yMaskReg0);
                } else if constexpr (IsSameType<U, int8_t>::value) {
                    AscendC::MicroAPI::Cast<int16_t, float, castTraitF32ToI16>(int16Reg, yReg3, yMaskReg0);
                    AscendC::MicroAPI::Cast<half, int16_t, castTraitI16ToF16>(halfReg, int16Reg, yMaskReg0);
                    AscendC::MicroAPI::Cast<U, half, castTraitF16ToI8>(outReg, halfReg, yMaskReg0);
                } else {
                    AscendC::MicroAPI::Cast<U, float, castTrait32tofp8>(outReg, yReg3, yMaskReg0);
                }
                AscendC::MicroAPI::DataCopy<U, MicroAPI::StoreDist::DIST_PACK4_B32>(yOutAddr + i * vfNum, outReg,
                                                                                    yMaskReg0);
            }
        } else {
            // ===== FP16/BF16 类型处理 =====
            AscendC::MicroAPI::RegTensor<uint16_t> expMaxRegTensor;
            AscendC::MicroAPI::Duplicate((AscendC::MicroAPI::RegTensor<uint16_t>&)vreg2, maxValue_);
            AscendC::MicroAPI::Duplicate(expMaxRegTensor, 0);

            AscendC::MicroAPI::MaskReg
                maskAll = AscendC::MicroAPI::CreateMask<uint16_t, AscendC::MicroAPI::MaskPattern::ALL>();
            AscendC::MicroAPI::MaskReg
                ymaskAll = AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();

            // comput Max(x) - FP16/BF16 使用 uint16_t 位运算
            for (uint16_t i = 0; i < inputVfLoop; i++) {
                preg0 = AscendC::MicroAPI::UpdateMask<T>(inputNum);
                AscendC::MicroAPI::DataCopy(vreg1, xAddr + i * vfLen);
                AscendC::MicroAPI::And((AscendC::MicroAPI::RegTensor<uint16_t>&)vreg3,
                                       (AscendC::MicroAPI::RegTensor<uint16_t>&)vreg1,
                                       (AscendC::MicroAPI::RegTensor<uint16_t>&)vreg2, preg0);
                AscendC::MicroAPI::Max<uint16_t, AscendC::MicroAPI::MaskMergeMode::MERGING>(
                    expMaxRegTensor, expMaxRegTensor, (AscendC::MicroAPI::RegTensor<uint16_t>&)vreg3, preg0);
            }

            // comput scale
            AscendC::MicroAPI::ReduceMax<uint16_t>(expMaxRegTensor, expMaxRegTensor, maskAll);
            AscendC::MicroAPI::Duplicate(expMaxRegTensor, expMaxRegTensor, maskAll);
            AscendC::MicroAPI::Duplicate(fp8MaxReg, fp8MaxValue_);
            AscendC::MicroAPI::Duplicate(reciprocalScale, 1.0f);
            AscendC::MicroAPI::Duplicate(minScaleReg, minScale_);

            AscendC::MicroAPI::Div<float, &mode>(reciprocalScale, reciprocalScale, minScaleReg, maskAll);
            AscendC::MicroAPI::Cast<float, T, castTrait0>(scaleRegTensor,
                                                          (AscendC::MicroAPI::RegTensor<T>&)expMaxRegTensor, maskAll);

            AscendC::MicroAPI::Div<float, &mode>(scaleRegTensor, scaleRegTensor, fp8MaxReg, ymaskAll);
            AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::LT>(
                scaleMaskReg, (AscendC::MicroAPI::RegTensor<uint32_t>&)scaleRegTensor, infValue_, ymaskAll);
            AscendC::MicroAPI::Min<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(scaleRegTensor, scaleRegTensor,
                                                                                     reciprocalScale, scaleMaskReg);
            AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                scaleOutAddr, scaleRegTensor, ymaskAll);

            // compute y
            for (uint16_t i = 0; i < vfLoop; i++) {
                yMaskReg0 = AscendC::MicroAPI::UpdateMask<float>(outputNum);
                AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(yReg1, xAddr + i * vfNum);
                AscendC::MicroAPI::Cast<float, T, castTrait0>(yReg2, yReg1, yMaskReg0);
                AscendC::MicroAPI::Div<float, &mode>(yReg3, yReg2, scaleRegTensor, yMaskReg0);

                if constexpr (IsSameType<U, hifloat8_t>::value) {
                    AscendC::MicroAPI::Cast<U, float, castTrait32toh8>(outReg, yReg3, yMaskReg0);
                } else if constexpr (IsSameType<U, int8_t>::value) {
                    AscendC::MicroAPI::Cast<int16_t, float, castTraitF32ToI16>(int16Reg, yReg3, yMaskReg0);
                    AscendC::MicroAPI::Cast<half, int16_t, castTraitI16ToF16>(halfReg, int16Reg, yMaskReg0);
                    AscendC::MicroAPI::Cast<U, half, castTraitF16ToI8>(outReg, halfReg, yMaskReg0);
                } else {
                    AscendC::MicroAPI::Cast<U, float, castTrait32tofp8>(outReg, yReg3, yMaskReg0);
                }
                AscendC::MicroAPI::DataCopy<U, MicroAPI::StoreDist::DIST_PACK4_B32>(yOutAddr + i * vfNum, outReg,
                                                                                    yMaskReg0);
            }
        }
    }
}

template <typename T, typename U, int64_t RMode>
__aicore__ inline void DynamicBlockQuantSmallBlockSize<T, U, RMode>::Compute(int32_t blockRow, int32_t blockCol,
                                                                             int64_t blockRowNum, int64_t blockColNum,
                                                                             int64_t nowRowBlock)
{
    LocalTensor<T> inLocal = inQueue_.DeQue<T>();
    __local_mem__ T* xLocalAddr = (__local_mem__ T*)inLocal.GetPhyAddr();

    LocalTensor<float> scaleLocal = scaleQueue_.AllocTensor<float>();
    __local_mem__ float* scaleLocalAddr = (__local_mem__ float*)scaleLocal.GetPhyAddr();

    LocalTensor<U> outLocal = outQueue_.AllocTensor<U>();
    __local_mem__ U* outLocalAddr = (__local_mem__ U*)outLocal.GetPhyAddr();
    int64_t ubOffset = 0;
    int64_t yOffset = 0;
    int32_t inputColAlign = BLOCK_BYTE_32 / sizeof(T);
    int32_t outputColAlign = BLOCK_BYTE_32 / sizeof(U);

    for (int32_t rowIdx = 0; rowIdx < blockRow; rowIdx++) {
        int64_t tmpBlock = nowRowBlock + rowIdx;
        // 当前block的行数
        int64_t rowBlockSize = 0;
        if (tmpBlock % singleBatchRowBlockLoopNum_ == singleBatchRowBlockLoopNum_ - 1) {
            rowBlockSize = tailBlockSizeRow_;
        } else {
            rowBlockSize = blockSizeRow_;
        }
        for (int32_t colIdx = 0; colIdx < blockCol; colIdx++) {
            int64_t colBlockSize = colIdx == blockCol - 1 ? blockColNum - colIdx * blockSizeCol_ : blockSizeCol_;
            int64_t scaleOffset = (rowIdx * blockCol + colIdx) * 8;
            ComputeVF(rowBlockSize, colBlockSize, xLocalAddr + ubOffset, scaleLocalAddr + scaleOffset,
                      outLocalAddr + yOffset);
            int64_t xTotalNum = rowBlockSize * colBlockSize;
            ubOffset = ubOffset + (xTotalNum + inputColAlign - 1) / inputColAlign * inputColAlign;
            yOffset = yOffset + (xTotalNum + outputColAlign - 1) / outputColAlign * outputColAlign;
        }
    }

    inQueue_.FreeTensor(inLocal);
    outQueue_.EnQue(outLocal);
    scaleQueue_.EnQue(scaleLocal);
}

template <typename T, typename U, int64_t RMode>
__aicore__ inline void DynamicBlockQuantSmallBlockSize<T, U, RMode>::CopyOut(int32_t blockRow, int32_t blockCol,
                                                                             int64_t blockRowNum, int64_t blockColNum,
                                                                             int64_t baseXGmOffset,
                                                                             int64_t baseScaleOffset,
                                                                             int64_t nowRowBlock)
{
    LocalTensor<U> outLocal = outQueue_.DeQue<U>();
    LocalTensor<float> scaleLocal = scaleQueue_.DeQue<float>();
    int32_t outputColAlign = BLOCK_BYTE_32 / sizeof(U);
    // yGm偏移地址
    int64_t outGmOffset = 0;
    // yUb偏移地址
    int64_t outUbOffset = 0;
    // scaleGm偏移地址
    int64_t scaleGmOffset = 0;
    // scaleUb偏移地址
    int64_t scaleUbOffset = 0;
    int32_t rowBlockSize = 0;
    int64_t outGmRowOffset = 0;
    for (int32_t rowIdx = 0; rowIdx < blockRow; rowIdx++) {
        int64_t tmpBlock = nowRowBlock + rowIdx;
        if (tmpBlock % singleBatchRowBlockLoopNum_ == singleBatchRowBlockLoopNum_ - 1) {
            rowBlockSize = tailBlockSizeRow_;
        } else {
            rowBlockSize = blockSizeRow_;
        }
        for (int32_t colIdx = 0; colIdx < blockCol; colIdx++) {
            outGmOffset = baseXGmOffset + outGmRowOffset + colIdx * blockSizeCol_;
            scaleUbOffset = (rowIdx * blockCol + colIdx) * 8;
            scaleGmOffset = baseScaleOffset + (rowIdx * colBlockLoopNum_ + colIdx);
            int64_t colBlockSize = colIdx == blockCol - 1 ? blockColNum - colIdx * blockSizeCol_ : blockSizeCol_;

            DataCopyExtParams outCopyParams = {static_cast<uint16_t>(rowBlockSize),
                                               static_cast<uint32_t>(colBlockSize * sizeof(U)), 0,
                                               static_cast<uint32_t>((colNum_ - colBlockSize) * sizeof(U)), 0};
            DataCopyExtParams scaleCopyParams = {1, static_cast<uint32_t>(1 * sizeof(float)), 0, 0, 0};

            DataCopyPad<U, PaddingMode::Compact>(yGm_[outGmOffset], outLocal[outUbOffset], outCopyParams);
            DataCopyPad(scaleGm_[scaleGmOffset], scaleLocal[scaleUbOffset], scaleCopyParams);
            outUbOffset = outUbOffset +
                          (rowBlockSize * colBlockSize + outputColAlign - 1) / outputColAlign * outputColAlign;
        }
        outGmRowOffset = outGmRowOffset + rowBlockSize * colNum_;
    }

    outQueue_.FreeTensor(outLocal);
    scaleQueue_.FreeTensor(scaleLocal);
}

} // namespace DynamicBlockQuant
#endif // DYNAMIC_BLOCK_QUANT_SMALL_BLOCKSIZE_H
