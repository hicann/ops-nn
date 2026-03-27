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
 * \file dynamic_block_mx_quant_base.h
 * \brief
 */

#ifndef DYNAMIC_BLOCK_MX_QUANT_BASE_H
#define DYNAMIC_BLOCK_MX_QUANT_BASE_H
#include "kernel_operator.h"
#include "../inc/platform.h"
#include "dynamic_block_mx_quant_tilingdata.h"

namespace DynamicBlockMxQuant {
#define FLOAT_OVERFLOW_MODE_CTRL 60
using namespace AscendC;

constexpr int64_t DB_BUFFER = 2;
constexpr int64_t DIGIT_TWO = 2;
constexpr int64_t OUT_ELE_NUM_ONE_BLK = 64;
constexpr uint16_t INF_FOR_BF16 = 0x7f80;          // 0111 1111 1000 0000
constexpr uint16_t INF_FOR_FP16 = 0x7c00;          // 0111 1100 0000 0000
constexpr uint16_t NAN_FOR_FP8_E8M0 = 0x00ff;      // 0000 0000 1111 1111
constexpr uint16_t BF16_EXP_BIAS = 0x7f00;         // 0111 1111 0000 0000
constexpr uint16_t NAN_CUSTOMIZATION = 0x7f81;     // 0111 1111 1000 0001
constexpr uint16_t SPECIAL_EXP_THRESHOLD = 0x0040; // 0000 0000 0100 0000
constexpr int16_t SHR_NUM_FOR_BF16 = 7;
constexpr uint16_t FP8_E4M3_MAX_EXP = 0x0400; // elem_emax右移7位(BF16E8M7) 0 00001000 0000000
constexpr uint16_t FP8_E5M2_MAX_EXP = 0x0780; // 0 00001111 0000000
constexpr uint32_t FP8_E5M2_MAX = 0x37924925; // 1/57344的float32表示 57334是E5M2所能表示的最大值
constexpr uint32_t FP8_E4M3_MAX = 0x3b124925; // 1/448的float32表示 448是E4M3所能表示的最大值
constexpr uint16_t SPECIAL_VALUE_E2M1 = 0x00ff;
constexpr uint16_t SPECIAL_VALUE_E1M2 = 0x007f;
constexpr uint16_t FP4_E2M1_BF16_MAX_EXP = 0x0100;
constexpr uint32_t MAX_EXP_FOR_FP32 = 0x7f800000;
constexpr int32_t NEG_ZERO = 0x80000000;
constexpr uint16_t EXP_MASK_BF16 = 0x7f80; // 0111 1111 1000 0000
constexpr uint16_t EXP_MASK_FP16 = 0x7c00; // 0111 1100 0000 0000
constexpr uint16_t ABS_FOR_UINT16 = 0x7fff;

constexpr uint16_t ONE_BIT_FOR_MANTISSA = 0x00c1; // 0000 0000 0011 1111
constexpr uint16_t TWO_BIT_FOR_MANTISSA = 0x00e1; // 0000 0000 0001 1111

constexpr float FOUR = 4.0;
constexpr float ONE_FOURTH = 0.25;
constexpr float FLOAT_ZERO = 0.0;
constexpr float FLOAT_SIX = 6.0;
constexpr float FLOAT_SEVEN = 7.0;
constexpr int32_t NEG_ONE = -1;
constexpr int16_t SHR_NUM_FOR_FP32 = 23;
constexpr int32_t FP32_BIAS = 127;
constexpr int32_t FP32_BIAS_NEG = -127;
constexpr int32_t UbBlockRow = 64;
constexpr int32_t UbBlockCol = 256;
constexpr int32_t ScaleGatherIndex = 8;

template <typename T, typename U, AscendC::RoundMode roundMode, uint64_t scaleAlg>
class DynamicBlockMxQuantBase {
public:
    __aicore__ inline DynamicBlockMxQuantBase(const DynamicBlockMxQuantTilingData* tilingData, TPipe* pipe)
        : tilingData_(tilingData), pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR mxScale1, GM_ADDR mxScale2);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitParams();
    __aicore__ inline void CopyIn(
        int32_t blockRow, int32_t blockCol, int32_t blockRowNum, int32_t blockColNum, int64_t baseXGmOffset,
        int64_t nowRowBlock);
    __aicore__ inline void CopyOut(
        int32_t blockRow, int32_t blockCol, int32_t blockRowNum, int32_t blockColNum, int64_t baseXGmOffset,
        int64_t baseScale1Offset, int64_t baseScale2Offset, int64_t nowRowBlock);
    __aicore__ inline void Compute(
        int32_t blockRow, int32_t blockCol, int32_t blockRowNum, int32_t blockColNum, int64_t nowRowBlock);
    __aicore__ inline void ComputeOcp(
        uint16_t rowNum, int64_t colNum, __ubuf__ T* xAddr, __ubuf__ uint8_t* scaleOutAddr,
        __ubuf__ uint8_t* scale2OutAddr, __ubuf__ uint16_t* mxScaleReciprocalAddr, __ubuf__ uint8_t* yOutAddr,
        __ubuf__ uint8_t* gatherIndexAddr);
    __aicore__ inline void ComputeDdr(
        uint16_t rowNum, int64_t colNum, __ubuf__ T* xAddr, __ubuf__ uint8_t* scaleOutAddr,
        __ubuf__ uint8_t* scale2OutAddr, __ubuf__ uint16_t* mxScaleReciprocalAddr, __ubuf__ uint8_t* yOutAddr,
        __ubuf__ uint8_t* gatherIndexAddr);
    __aicore__ inline void ComputeYVf(
        uint16_t dataLen, uint16_t blockCount, __ubuf__ T* xAddr, __ubuf__ uint16_t* mxScaleReciprocalAddr,
        __ubuf__ uint8_t* yAddr);
    __aicore__ inline void ComputeY1ToFP4(
        uint16_t dataLen, uint16_t blockCount, __ubuf__ T* xAddr, __ubuf__ uint16_t* mxScale1ReciprocalAddr,
        __ubuf__ uint8_t* y1Addr);
    __aicore__ inline void ComputeY1ToFP8(
        uint16_t dataLen, uint16_t blockCount, __ubuf__ T* xAddr, __ubuf__ uint16_t* mxScale1ReciprocalAddr,
        __ubuf__ uint8_t* y1Addr);
    __aicore__ inline void ComputeFP4FromHalf(Reg::RegTensor<float>& Reg);

protected:
    static constexpr Reg::CastTrait castTraitHalf2BF16 = {
        Reg::RegLayout::UNKNOWN, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_TRUNC};
    static constexpr Reg::CastTrait castTraitHalf2BF16Ddr = {
        Reg::RegLayout::UNKNOWN, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
    static constexpr Reg::CastTrait castTraitXdtypetoFp32Zero = {
        Reg::RegLayout::ZERO, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};
    static constexpr Reg::CastTrait castTraitXdtypetoFp32One = {
        Reg::RegLayout::ONE, Reg::SatMode::UNKNOWN, Reg::MaskMergeMode::ZEROING, AscendC::RoundMode::UNKNOWN};

    static constexpr Reg::CastTrait castTraitFp32toYdtype = {
        Reg::RegLayout::ZERO, Reg::SatMode::SAT, Reg::MaskMergeMode::ZEROING, roundMode};

    static constexpr Reg::CastTrait castTraitFp32toBF16 = {
        Reg::RegLayout::ZERO, Reg::SatMode::NO_SAT, Reg::MaskMergeMode::ZEROING, roundMode};
    static constexpr Reg::CastTrait castTraitBF16toFp4 = {
        Reg::RegLayout::ZERO, Reg::SatMode::SAT, Reg::MaskMergeMode::ZEROING, roundMode};

private:
    // tiling data
    const DynamicBlockMxQuantTilingData* tilingData_;
    TPipe* pipe_ = nullptr;
    TQue<QuePosition::VECIN, DB_BUFFER> inQueue_;
    GlobalTensor<T> xGm_;
    TQue<QuePosition::VECOUT, DB_BUFFER> outQueue_;
    GlobalTensor<uint8_t> yGm_;
    TBuf<TPosition::VECCALC> mxScaleReciprocalBuf_;
    TBuf<TPosition::VECCALC> tempIndexBuf_;

    TQue<QuePosition::VECOUT, DB_BUFFER> scale1Queue_;
    TQue<QuePosition::VECOUT, DB_BUFFER> scale2Queue_;
    GlobalTensor<uint8_t> mxScale1Gm_;
    GlobalTensor<uint8_t> mxScale2Gm_;

    int64_t blockIdx_ = 0;
    uint16_t infValue_ = 0;
    float fp8MaxValue_ = 0;
    int64_t ubBlockSize_ = platform::GetUbBlockSize();
    // dType
    uint32_t invDtypeMax_ = 0;
    uint16_t dtypeYMaxExp_ = 0;
    uint16_t fp4SpecialValue_ = 0;

    uint16_t bitForMantissa_ = 0;

    int64_t totalCoreNum_ = 0;
    int64_t ubSize_ = 0;
    int64_t blockSizeRow_ = 0;
    int64_t blockSizeCol_ = 0;
    int64_t batchNum_ = 0;
    int64_t rowNum_ = 0;
    int64_t colNum_ = 0;
    int64_t singleBatchRowBlockLoopNum_ = 0;
    int64_t rowBlockLoopNum_ = 0;
    int64_t colBlockLoopNum_ = 0;
    int64_t rowScaleNum_ = 0;
    int64_t colScaleNum_ = 0;
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
    float dstTypeMax_ = 0.0;

    int64_t blockW_ = 256;
    int64_t blockH_ = 64;

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
    int64_t vlForHalfNumber_ = platform::GetVRegSize() / sizeof(uint16_t);
    int64_t vlForFp8 = platform::GetVRegSize() / sizeof(uint8_t);
    int64_t oneBlockCountB16_ = ubBlockSize_ / sizeof(T);
    int64_t oneBlockCountB8_ = ubBlockSize_ / sizeof(uint8_t);
};

template <typename T, typename U, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void DynamicBlockMxQuantBase<T, U, roundMode, scaleAlg>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR mxScale1, GM_ADDR mxScale2)
{
#if (__NPU_ARCH__ == 3510)
    SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
#endif
    InitParams();

    if (blockIdx_ >= usedCoreNum_) {
        return;
    }

    pipe_->InitBuffer(inQueue_, DB_BUFFER, rowUbFactor_ * colUbFactor_ * sizeof(T));
    pipe_->InitBuffer(outQueue_, DB_BUFFER, rowUbFactor_ * colUbFactor_);
    pipe_->InitBuffer(scale1Queue_, DB_BUFFER, rowUbFactor_ * colUbBlockLoopNum_ * ubBlockSize_);
    pipe_->InitBuffer(scale2Queue_, DB_BUFFER, rowUbBlockLoopNum_ * DIGIT_TWO * colUbBlockLoopNum_ * blockW_);
    pipe_->InitBuffer(
        mxScaleReciprocalBuf_, rowUbBlockLoopNum_ * DIGIT_TWO * colUbBlockLoopNum_ * ubBlockSize_ * sizeof(T));
    pipe_->InitBuffer(tempIndexBuf_, vlForFp8);

    // 计算核在行列的位置，是否是尾核
    rowCoreIdx_ = blockIdx_ / colTileNum_;
    colCoreIdx_ = blockIdx_ % colTileNum_;
    isRowTailCore_ = (rowCoreIdx_ >= rowNormalCoreNum_);
    isColTailCore_ = (colCoreIdx_ >= colNormalCoreNum_);

    int64_t xRowGmOffset = 0;
    int64_t xColGmOffset = 0;
    int64_t xGmOffset = 0;
    int64_t scale1GmRowOffset = 0;
    int64_t scale1GmColOffset = 0;
    int64_t scale1GmOffset = 0;
    int64_t scale2GmRowOffset = 0;
    int64_t scale2GmColOffset = 0;
    int64_t scale2GmOffset = 0;

    tailBlockSizeRow_ = rowNum_ % blockH_;
    if (tailBlockSizeRow_ == 0) {
        tailBlockSizeRow_ = blockH_;
    }

    if (isRowTailCore_) {
        rowCoreTileNum_ = tailCoreRowTileNum_;
        rowUbBlockLoopNum_ = rowUbBlockLoopNum_ > rowCoreTileNum_ ? rowCoreTileNum_ : rowUbBlockLoopNum_;
        rowUbLoop_ = Ceil(tailCoreRowTileNum_, rowUbBlockLoopNum_);
        // 算前面有多少block行，来计算patch
        preRowBlockNum_ =
            rowNormalCoreNum_ * normalCoreRowTileNum_ + (rowCoreIdx_ - rowNormalCoreNum_) * tailCoreRowTileNum_;
    } else {
        rowCoreTileNum_ = normalCoreRowTileNum_;
        rowUbLoop_ = Ceil(rowCoreTileNum_, rowUbBlockLoopNum_);
        preRowBlockNum_ = rowCoreIdx_ * normalCoreRowTileNum_;
    }

    // 计算当前核前面有多少个batch，当前核的自己batch数，多出来的block数
    preBatch_ = preRowBlockNum_ / singleBatchRowBlockLoopNum_;
    preSingleBatchBlock_ = preRowBlockNum_ % singleBatchRowBlockLoopNum_;
    xRowGmOffset = preBatch_ * rowNum_ * colNum_ + preSingleBatchBlock_ * blockH_ * colNum_;

    scale1GmRowOffset = preBatch_ * rowNum_ * colScaleNum_ + preSingleBatchBlock_ * blockH_ * colScaleNum_;
    scale2GmRowOffset = preRowBlockNum_ * DIGIT_TWO * colNum_;

    int64_t endRowBlockNum_ = preRowBlockNum_ + rowCoreTileNum_;
    int64_t endBath = endRowBlockNum_ / singleBatchRowBlockLoopNum_;
    int64_t endSingleBatchBlock = endRowBlockNum_ % singleBatchRowBlockLoopNum_;
    // 首尾做差得到当前核处理的行数
    coreRowNum_ = (endBath - preBatch_) * rowNum_ + (endSingleBatchBlock - preSingleBatchBlock_) * blockH_;

    if (isColTailCore_) {
        colCoreTileNum_ = tailCoreColTileNum_;
        colUbBlockLoopNum_ = colUbBlockLoopNum_ > colCoreTileNum_ ? colCoreTileNum_ : colUbBlockLoopNum_;
        colUbLoop_ = Ceil(colCoreTileNum_, colUbBlockLoopNum_);
        xColGmOffset = colNormalCoreNum_ * normalCoreColTileNum_ * blockW_ +
                       (colCoreIdx_ - colNormalCoreNum_) * tailCoreColTileNum_ * blockW_;
        scale1GmColOffset =
            (colNormalCoreNum_ * normalCoreColTileNum_ + (colCoreIdx_ - colNormalCoreNum_) * tailCoreColTileNum_) *
            blockW_ / blockSizeCol_;
        scale2GmColOffset = colNormalCoreNum_ * normalCoreColTileNum_ * blockW_ * DIGIT_TWO +
                            (colCoreIdx_ - colNormalCoreNum_) * tailCoreColTileNum_ * blockW_ * DIGIT_TWO;

        coreColNum_ = colCoreIdx_ + 1 == colTileNum_ ?
                          colNum_ - (colNormalCoreNum_ * normalCoreColTileNum_ +
                                     (colCoreIdx_ - colNormalCoreNum_) * tailCoreColTileNum_) *
                                        blockW_ :
                          tailCoreColTileNum_ * blockW_;
    } else {
        colCoreTileNum_ = normalCoreColTileNum_;
        colUbLoop_ = Ceil(colCoreTileNum_, colUbBlockLoopNum_);
        xColGmOffset = colCoreIdx_ * normalCoreColTileNum_ * blockW_;

        scale1GmColOffset = colCoreIdx_ * normalCoreColTileNum_ * blockW_ / blockSizeCol_;
        scale2GmColOffset = colCoreIdx_ * normalCoreColTileNum_ * blockW_ * DIGIT_TWO;

        coreColNum_ = colCoreIdx_ + 1 == colTileNum_ ? colNum_ - colCoreIdx_ * normalCoreColTileNum_ * blockW_ :
                                                       normalCoreColTileNum_ * blockW_;
    }
    xGmOffset = xRowGmOffset + xColGmOffset;
    scale1GmOffset = scale1GmRowOffset + scale1GmColOffset;
    scale2GmOffset = scale2GmRowOffset + scale2GmColOffset;

    xGm_.SetGlobalBuffer((__gm__ T*)x + xGmOffset);
    if constexpr (IsSameType<U, fp4x2_e2m1_t>::value || IsSameType<U, fp4x2_e1m2_t>::value) {
        yGm_.SetGlobalBuffer((__gm__ uint8_t*)y + xGmOffset / 2);
    } else {
        yGm_.SetGlobalBuffer((__gm__ uint8_t*)y + xGmOffset);
    }
    mxScale1Gm_.SetGlobalBuffer((__gm__ uint8_t*)mxScale1 + scale1GmOffset);
    mxScale2Gm_.SetGlobalBuffer((__gm__ uint8_t*)mxScale2 + scale2GmOffset);
}

template <typename T, typename U, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void DynamicBlockMxQuantBase<T, U, roundMode, scaleAlg>::InitParams()
{
    blockIdx_ = GetBlockIdx();
    totalCoreNum_ = tilingData_->totalCoreNum;
    ubSize_ = tilingData_->ubSize;
    blockSizeRow_ = tilingData_->blockSizeRow;
    blockSizeCol_ = tilingData_->blockSizeCol;
    batchNum_ = tilingData_->batchNum;
    rowNum_ = tilingData_->rowNum;
    colNum_ = tilingData_->colNum;
    singleBatchRowBlockLoopNum_ = tilingData_->singleBatchRowBlockLoopNum;
    rowBlockLoopNum_ = tilingData_->rowBlockLoopNum;
    colBlockLoopNum_ = tilingData_->colBlockLoopNum;
    rowScaleNum_ = tilingData_->rowScaleNum;
    colScaleNum_ = tilingData_->colScaleNum;
    rowUbBlockLoopNum_ = tilingData_->rowUbBlockLoopNum;
    colUbBlockLoopNum_ = tilingData_->colUbBlockLoopNum;
    rowUbFactor_ = tilingData_->rowUbFactor;
    colUbFactor_ = tilingData_->colUbFactor;
    usedCoreNum_ = tilingData_->usedCoreNum;
    rowTileNum_ = tilingData_->rowTileNum;
    colTileNum_ = tilingData_->colTileNum;
    normalCoreRowTileNum_ = tilingData_->normalCoreRowTileNum;
    normalCoreColTileNum_ = tilingData_->normalCoreColTileNum;
    tailCoreRowTileNum_ = tilingData_->tailCoreRowTileNum;
    tailCoreColTileNum_ = tilingData_->tailCoreColTileNum;
    rowNormalCoreNum_ = tilingData_->rowNormalCoreNum;
    colNormalCoreNum_ = tilingData_->colNormalCoreNum;
    rowTailCoreNum_ = tilingData_->rowTailCoreNum;
    colTailCoreNum_ = tilingData_->colTailCoreNum;
    dstTypeMax_ = tilingData_->dstTypeMax;

    if constexpr (IsSameType<U, fp8_e4m3fn_t>::value) {
        dtypeYMaxExp_ = FP8_E4M3_MAX_EXP;
        invDtypeMax_ = FP8_E4M3_MAX;
    } else if constexpr (IsSameType<U, fp8_e5m2_t>::value) {
        dtypeYMaxExp_ = FP8_E5M2_MAX_EXP;
        invDtypeMax_ = FP8_E5M2_MAX;
    } else if constexpr (IsSameType<U, fp4x2_e2m1_t>::value) {
        dtypeYMaxExp_ = FP4_E2M1_BF16_MAX_EXP;
        fp4SpecialValue_ = SPECIAL_VALUE_E2M1;
    } else {
        dtypeYMaxExp_ = 0;
        fp4SpecialValue_ = SPECIAL_VALUE_E1M2;
    }

    if (dstTypeMax_ == FLOAT_ZERO || dstTypeMax_ == FLOAT_SIX) {
        bitForMantissa_ = ONE_BIT_FOR_MANTISSA;
    } else {
        bitForMantissa_ = TWO_BIT_FOR_MANTISSA;
    }
}

template <typename T, typename U, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void DynamicBlockMxQuantBase<T, U, roundMode, scaleAlg>::Process()
{
    if (blockIdx_ >= usedCoreNum_) {
        return;
    }
    LocalTensor<uint8_t> tempIndexLocal = tempIndexBuf_.Get<uint8_t>();

    for (int32_t idx = 0; idx < ScaleGatherIndex; idx++) {
        int32_t offset = idx * blockSizeCol_;
        Duplicate<uint8_t>(tempIndexLocal[offset], static_cast<uint8_t>(idx), static_cast<uint8_t>(blockSizeCol_));
    }

    for (int32_t rowUbLoopIdx = 0; rowUbLoopIdx < rowUbLoop_; rowUbLoopIdx++) {
        // 行block数
        int32_t blockRow =
            rowUbLoopIdx == rowUbLoop_ - 1 ? rowCoreTileNum_ - rowUbLoopIdx * rowUbBlockLoopNum_ : rowUbBlockLoopNum_;
        int64_t nowRowBlockNum = preRowBlockNum_ + rowUbLoopIdx * rowUbBlockLoopNum_;
        int64_t nowBatch = nowRowBlockNum / singleBatchRowBlockLoopNum_;
        int64_t nowSingleBatchBlock = nowRowBlockNum % singleBatchRowBlockLoopNum_;

        int64_t endRowBlockNum = nowRowBlockNum + blockRow;
        int64_t endBatch = endRowBlockNum / singleBatchRowBlockLoopNum_;
        int64_t endSingleBatchBlock = endRowBlockNum % singleBatchRowBlockLoopNum_;

        int32_t blockRowNum = (endBatch - nowBatch) * rowNum_ + (endSingleBatchBlock - nowSingleBatchBlock) * blockH_;
        for (int32_t colUbLoopIdx = 0; colUbLoopIdx < colUbLoop_; colUbLoopIdx++) {
            int32_t blockCol = colUbLoopIdx == colUbLoop_ - 1 ? colCoreTileNum_ - colUbLoopIdx * colUbBlockLoopNum_ :
                                                                colUbBlockLoopNum_;
            int64_t baseXGmOffset =
                ((nowBatch - preBatch_) * rowNum_ + (nowSingleBatchBlock - preSingleBatchBlock_) * blockH_) * colNum_ +
                colUbLoopIdx * colUbBlockLoopNum_ * blockW_;
            int32_t blockColNum = colUbLoopIdx == colUbLoop_ - 1 ?
                                      coreColNum_ - colUbLoopIdx * colUbBlockLoopNum_ * blockW_ :
                                      colUbBlockLoopNum_ * blockW_;

            int64_t baseScale1Offset =
                ((nowBatch - preBatch_) * rowNum_ + (nowSingleBatchBlock - preSingleBatchBlock_) * blockH_) *
                    colScaleNum_ +
                colUbLoopIdx * colUbBlockLoopNum_ * blockW_ / blockSizeCol_;

            int64_t baseScale2Offset = rowUbLoopIdx * rowUbBlockLoopNum_ * DIGIT_TWO * colNum_ +
                                       colUbLoopIdx * colUbBlockLoopNum_ * blockW_ * DIGIT_TWO;

            CopyIn(blockRow, blockCol, blockRowNum, blockColNum, baseXGmOffset, nowSingleBatchBlock);
            Compute(blockRow, blockCol, blockRowNum, blockColNum, nowSingleBatchBlock);
            CopyOut(
                blockRow, blockCol, blockRowNum, blockColNum, baseXGmOffset, baseScale1Offset, baseScale2Offset,
                nowSingleBatchBlock);
        }
    }
}

template <typename T, typename U, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void DynamicBlockMxQuantBase<T, U, roundMode, scaleAlg>::CopyIn(
    int32_t blockRow, int32_t blockCol, int32_t blockRowNum, int32_t blockColNum, int64_t baseXGmOffset,
    int64_t nowRowBlock)
{
    LocalTensor<T> inLocal = inQueue_.AllocTensor<T>();

    int64_t gmOffset = 0;
    int64_t gmRowOffset = 0;
    int64_t ubOffset = 0;
    int32_t rowBlockSize = 0;
    int64_t inputColAlign = ubBlockSize_ / sizeof(T);

    if (blockColNum % blockW_ != 0) {
        Duplicate<T>(inLocal, static_cast<T>(0), rowUbFactor_ * colUbFactor_);
        event_t eventIDVToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventIDVToMTE2);
        WaitFlag<HardEvent::V_MTE2>(eventIDVToMTE2);
    }

    for (int32_t rowIdx = 0; rowIdx < blockRow; rowIdx++) {
        int64_t tmpBlock = nowRowBlock + rowIdx;
        if (tmpBlock % singleBatchRowBlockLoopNum_ == singleBatchRowBlockLoopNum_ - 1) {
            rowBlockSize = tailBlockSizeRow_;
        } else {
            rowBlockSize = blockH_;
        }
        for (int32_t colIdx = 0; colIdx < blockCol; colIdx++) {
            gmOffset = baseXGmOffset + gmRowOffset + colIdx * blockW_;
            int32_t colBlockSize = colIdx == blockCol - 1 ? blockColNum - colIdx * blockW_ : blockW_;
            int64_t rightPadding =
                ops::CeilAlign(static_cast<int64_t>(colBlockSize * sizeof(T)), ubBlockSize_) / sizeof(T) - colBlockSize;
            DataCopyExtParams copyParams = {
                static_cast<uint16_t>(rowBlockSize), static_cast<uint32_t>(colBlockSize * sizeof(T)),
                static_cast<uint32_t>((colNum_ - colBlockSize) * sizeof(T)),
                static_cast<uint32_t>((blockW_ - colBlockSize) * sizeof(T) / ubBlockSize_), 0};
            DataCopyPadExtParams<T> padParams{true, 0, static_cast<uint8_t>(rightPadding), 0};
            DataCopyPad(inLocal[ubOffset], xGm_[gmOffset], copyParams, padParams);
            int64_t blockTotalNum = rowBlockSize * blockW_;
            ubOffset = ubOffset + (blockTotalNum + inputColAlign - 1) / inputColAlign * inputColAlign;
        }
        gmRowOffset = gmRowOffset + rowBlockSize * colNum_;
    }
    inQueue_.EnQue(inLocal);
}

template <typename T, typename U, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void DynamicBlockMxQuantBase<T, U, roundMode, scaleAlg>::ComputeOcp(
    uint16_t rowNum, int64_t colNum, __ubuf__ T* xAddr, __ubuf__ uint8_t* mxScale1Addr, __ubuf__ uint8_t* mxScale2Addr,
    __ubuf__ uint16_t* mxScaleReciprocalAddr, __ubuf__ uint8_t* yOutAddr, __ubuf__ uint8_t* gatherIndexAddr)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<T> x0;
        Reg::RegTensor<T> x1;
        Reg::RegTensor<uint16_t> x0ExpFP16;
        Reg::RegTensor<uint16_t> x1ExpFP16;
        Reg::RegTensor<bfloat16_t> x0BF16;
        Reg::RegTensor<bfloat16_t> x1BF16;
        Reg::RegTensor<uint16_t> x0ExpBF16;
        Reg::RegTensor<uint16_t> x1ExpBF16;
        Reg::RegTensor<uint16_t> expMaskBF16;
        Reg::RegTensor<uint16_t> expMaskFP16;
        Reg::RegTensor<uint16_t> expMaxDim;
        Reg::RegTensor<uint16_t> expMaxDim1;
        Reg::RegTensor<uint16_t> expMax1Dim2;
        Reg::RegTensor<uint16_t> expMax2Dim2;
        Reg::RegTensor<uint16_t> yMaxExp;
        Reg::RegTensor<uint16_t> nanE8M0;
        Reg::RegTensor<uint16_t> biasE8M0;
        Reg::RegTensor<uint16_t> zero;
        Reg::RegTensor<uint16_t> nanBF16;
        Reg::RegTensor<uint16_t> specialExp;
        Reg::RegTensor<uint16_t> mxScaleB16;
        Reg::RegTensor<uint8_t> mxScale1B8;
        Reg::RegTensor<uint16_t> reversedShareExp;

        Reg::RegTensor<uint8_t> mxScale2B8;
        Reg::RegTensor<uint8_t> gatherIndex;

        Reg::MaskReg infMask;
        Reg::MaskReg zeroMask;
        Reg::MaskReg invalidDataMask;
        Reg::MaskReg infNanDataMask0;
        Reg::MaskReg infNanDataMask1;
        Reg::MaskReg maskAll = Reg::CreateMask<T, Reg::MaskPattern::ALL>();
        Reg::MaskReg maskB8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg maskReduceB8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::VL8>();
        Reg::MaskReg maskReduceB16 = Reg::CreateMask<uint8_t, Reg::MaskPattern::VL16>();

        Reg::Duplicate(expMaskBF16, EXP_MASK_BF16);
        Reg::Duplicate(expMaskFP16, EXP_MASK_FP16);
        Reg::Duplicate(expMax1Dim2, 0);
        Reg::Duplicate(expMax2Dim2, 0);
        Reg::Duplicate(yMaxExp, dtypeYMaxExp_);
        Reg::Duplicate(nanE8M0, NAN_FOR_FP8_E8M0);
        Reg::Duplicate(biasE8M0, BF16_EXP_BIAS);
        Reg::Duplicate(zero, 0);
        Reg::Duplicate(nanBF16, NAN_CUSTOMIZATION);
        Reg::Duplicate(specialExp, SPECIAL_EXP_THRESHOLD);

        // comput Max(x)
        for (uint16_t i = 0; i < rowNum; i++) {
            // 交织搬运，一次搬256个B16
            Reg::LoadAlign<T, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
                x0, x1, xAddr, vlForHalfNumber_ * DIGIT_TWO);

            if constexpr (IsSameType<T, half>::value) {
                // 提取指数位
                Reg::And(x0ExpFP16, (Reg::RegTensor<uint16_t>&)x0, expMaskFP16, maskAll);
                Reg::And(x1ExpFP16, (Reg::RegTensor<uint16_t>&)x1, expMaskFP16, maskAll);
                // 比较INF/NAN数据
                Reg::Compare<uint16_t, CMPMODE::NE>(infNanDataMask0, x0ExpFP16, expMaskFP16, maskAll);
                Reg::Compare<uint16_t, CMPMODE::NE>(infNanDataMask1, x1ExpFP16, expMaskFP16, maskAll);
                // 原始数据转成bf16
                Reg::Cast<bfloat16_t, T, castTraitHalf2BF16>(x0BF16, x0, maskAll);
                Reg::Cast<bfloat16_t, T, castTraitHalf2BF16>(x1BF16, x1, maskAll);
                // 提取指数位
                Reg::And(x0ExpBF16, (Reg::RegTensor<uint16_t>&)x0BF16, expMaskBF16, maskAll);
                Reg::And(x1ExpBF16, (Reg::RegTensor<uint16_t>&)x1BF16, expMaskBF16, maskAll);
                // 选择数据，INF/NAN数据时设成BF的INF/NAN
                Reg::Select<uint16_t>(x0ExpBF16, x0ExpBF16, expMaskBF16, infNanDataMask0);
                Reg::Select<uint16_t>(x1ExpBF16, x1ExpBF16, expMaskBF16, infNanDataMask1);
            } else {
                // 提取指数位
                Reg::And(x0ExpBF16, (Reg::RegTensor<uint16_t>&)x0, expMaskBF16, maskAll);
                Reg::And(x1ExpBF16, (Reg::RegTensor<uint16_t>&)x1, expMaskBF16, maskAll);
            }
            Reg::Max(expMax1Dim2, expMax1Dim2, x0ExpBF16, maskAll);
            Reg::Max(expMax2Dim2, expMax2Dim2, x1ExpBF16, maskAll);
        }

        // 计算x0和x1的最大值，相当于计算原始相邻两个数据的最大值
        Reg::Max(expMaxDim, expMax1Dim2, expMax2Dim2, maskAll);
        // ReduceMax一个block，即16个数，配合上一步，可以计算出每32*32个数的最大值，一共256/32个
        Reg::ReduceMaxWithDataBlock(expMaxDim, expMaxDim, maskAll);

        // inf/nan值单独处理，结果为E8M0的nan
        Reg::Compare<uint16_t, CMPMODE::NE>(infMask, expMaxDim, expMaskBF16, maskAll);
        // 0值单独处理，结果为0
        Reg::Compare<uint16_t, CMPMODE::NE>(zeroMask, expMaxDim, zero, maskAll);
        // 指数位不足被量化类型的ele_max时，为subnormal场景，结果为0
        Reg::Compare<uint16_t, CMPMODE::LE>(invalidDataMask, expMaxDim, yMaxExp, maskAll);
        Reg::Select<uint16_t>(expMaxDim, yMaxExp, expMaxDim, invalidDataMask);
        // 指数位减去expMax，按照BF16的格式处理，例：E5M2的expMax为15，即需要减去0 00001111 0000000
        Reg::Sub(expMaxDim, expMaxDim, yMaxExp, maskAll);
        // 右移7位，BF16的指数位移到了末8位
        Reg::ShiftRights(mxScaleB16, expMaxDim, SHR_NUM_FOR_BF16, maskAll);
        Reg::Select<uint16_t>(mxScaleB16, mxScaleB16, nanE8M0, infMask);
        Reg::Select<uint16_t>(mxScaleB16, mxScaleB16, zero, zeroMask);

        Reg::Pack<uint8_t, uint16_t, Reg::HighLowPart::LOWEST>(mxScale1B8, mxScaleB16);

        // -1轴scale循环搬出
        for (uint16_t i = 0; i < rowNum; i++) {
            Reg::StoreAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE>(
                mxScale1Addr, mxScale1B8, ubBlockSize_ / sizeof(uint8_t), maskReduceB8);
        }

        // -2轴scale扩展搬出
        Reg::LoadAlign(gatherIndex, gatherIndexAddr);
        Reg::Gather(mxScale2B8, mxScale1B8, gatherIndex);

        Reg::StoreAlign(mxScale2Addr, mxScale2B8, maskB8);

        // compute  1/scale
        // 只有在E1M2时，yMaxExp=0，xTmp0ExpBF16可能会等于biasE8M0
        Reg::Compare<uint16_t, CMPMODE::EQ>(invalidDataMask, expMaxDim, biasE8M0, maskAll);
        Reg::Sub(reversedShareExp, biasE8M0, expMaxDim, maskAll);
        Reg::Select<uint16_t>(reversedShareExp, reversedShareExp, nanBF16, infMask);
        Reg::Select<uint16_t>(reversedShareExp, reversedShareExp, zero, zeroMask);
        Reg::Select<uint16_t>(reversedShareExp, specialExp, reversedShareExp, invalidDataMask);

        Reg::StoreAlign(mxScaleReciprocalAddr, reversedShareExp, maskReduceB16);
    }
}

template <typename T, typename U, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void DynamicBlockMxQuantBase<T, U, roundMode, scaleAlg>::ComputeDdr(
    uint16_t rowNum, int64_t colNum, __ubuf__ T* xAddr, __ubuf__ uint8_t* mxScale1Addr, __ubuf__ uint8_t* mxScale2Addr,
    __ubuf__ uint16_t* mxScaleReciprocalAddr, __ubuf__ uint8_t* yOutAddr, __ubuf__ uint8_t* gatherIndexAddr)
{
    __VEC_SCOPE__
    {
        Reg::RegTensor<T> x0;
        Reg::RegTensor<T> x1;
        Reg::RegTensor<bfloat16_t> x0BF16;
        Reg::RegTensor<bfloat16_t> x1BF16;
        Reg::RegTensor<uint16_t> x0ExpFP16;
        Reg::RegTensor<uint16_t> x1ExpFP16;
        Reg::RegTensor<uint16_t> x0ExpBF16;
        Reg::RegTensor<uint16_t> x1ExpBF16;
        Reg::RegTensor<uint16_t> expMaxDim;
        Reg::RegTensor<uint16_t> expTmpMaxDim;
        Reg::RegTensor<uint16_t> expMaxDim1;
        Reg::RegTensor<uint16_t> expMax1Dim2;
        Reg::RegTensor<uint16_t> expMax2Dim2;
        Reg::RegTensor<uint16_t> expMaskBF16;
        Reg::RegTensor<uint16_t> expMaskFP16;

        Reg::RegTensor<uint16_t> nanE8M0;
        Reg::RegTensor<uint16_t> biasE8M0;
        Reg::RegTensor<uint16_t> zero;
        Reg::RegTensor<uint16_t> specialExp;
        Reg::RegTensor<uint16_t> nanBF16;
        Reg::RegTensor<uint16_t> mxScaleB16;
        Reg::RegTensor<uint8_t> mxScale1B8;
        Reg::RegTensor<uint8_t> mxScale2B8;
        Reg::RegTensor<uint16_t> yMaxExp;
        Reg::RegTensor<uint16_t> reversedExp;
        Reg::RegTensor<uint16_t> absForX;
        Reg::RegTensor<uint16_t> mantissaForMax;
        Reg::RegTensor<uint8_t> gatherIndex;

        Reg::MaskReg infMask;
        Reg::MaskReg zeroMask;
        Reg::MaskReg infNanDataMask0;
        Reg::MaskReg infNanDataMask1;
        Reg::MaskReg invalidDataMask;

        Reg::MaskReg maskB8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();
        Reg::MaskReg maskReduceB8 = Reg::CreateMask<uint8_t, Reg::MaskPattern::VL8>();
        Reg::MaskReg maskReduceB16 = Reg::CreateMask<uint8_t, Reg::MaskPattern::VL16>();
        Reg::MaskReg maskAll = Reg::CreateMask<T, Reg::MaskPattern::ALL>();

        Reg::Duplicate(zero, 0);
        Reg::Duplicate(expMax1Dim2, 0);
        Reg::Duplicate(expMax2Dim2, 0);
        Reg::Duplicate(expMaskBF16, EXP_MASK_BF16);
        Reg::Duplicate(expMaskFP16, EXP_MASK_FP16);
        Reg::Duplicate(nanE8M0, NAN_FOR_FP8_E8M0);
        Reg::Duplicate(nanBF16, NAN_CUSTOMIZATION);
        Reg::Duplicate(biasE8M0, BF16_EXP_BIAS);
        Reg::Duplicate(absForX, ABS_FOR_UINT16);
        Reg::Duplicate(yMaxExp, dtypeYMaxExp_);
        Reg::Duplicate(specialExp, SPECIAL_EXP_THRESHOLD);
        Reg::Duplicate(mantissaForMax, bitForMantissa_);

        // comput Max(x)
        for (uint16_t i = 0; i < rowNum; i++) {
            // 交织搬运，一次搬256个B16
            Reg::LoadAlign<T, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
                x0, x1, xAddr, vlForHalfNumber_ * DIGIT_TWO);

            if constexpr (IsSameType<T, half>::value) {
                // 求绝对值
                Reg::And(x0ExpFP16, (Reg::RegTensor<uint16_t>&)x0, absForX, maskAll);
                Reg::And(x1ExpFP16, (Reg::RegTensor<uint16_t>&)x1, absForX, maskAll);
                // 比较INF/NAN数据
                Reg::Compare<uint16_t, CMPMODE::LT>(infNanDataMask0, x0ExpFP16, expMaskFP16, maskAll);
                Reg::Compare<uint16_t, CMPMODE::LT>(infNanDataMask1, x1ExpFP16, expMaskFP16, maskAll);
                // 原始数据转成bf16
                Reg::Cast<bfloat16_t, T, castTraitHalf2BF16Ddr>(x0BF16, x0, maskAll);
                Reg::Cast<bfloat16_t, T, castTraitHalf2BF16Ddr>(x1BF16, x1, maskAll);
                // 提取指数位
                Reg::And(x0ExpBF16, (Reg::RegTensor<uint16_t>&)x0BF16, absForX, maskAll);
                Reg::And(x1ExpBF16, (Reg::RegTensor<uint16_t>&)x1BF16, absForX, maskAll);
                // 选择数据，INF/NAN数据时设成BF的INF/NAN
                Reg::Select<uint16_t>(x0ExpBF16, x0ExpBF16, expMaskBF16, infNanDataMask0);
                Reg::Select<uint16_t>(x1ExpBF16, x1ExpBF16, expMaskBF16, infNanDataMask1);
            } else {
                // 求绝对值
                Reg::And(x0ExpBF16, (AscendC::Reg::RegTensor<uint16_t>&)x0, absForX, maskAll);
                Reg::And(x1ExpBF16, (AscendC::Reg::RegTensor<uint16_t>&)x1, absForX, maskAll);
            }
            Reg::Max(expMax1Dim2, expMax1Dim2, x0ExpBF16, maskAll);
            Reg::Max(expMax2Dim2, expMax2Dim2, x1ExpBF16, maskAll);
        }

        // 计算x0和x1的最大值，相当于计算原始相邻两个数据的最大值
        Reg::Max(expMaxDim, expMax1Dim2, expMax2Dim2, maskAll);
        // ReduceMax一个block，即16个数，配合上一步，可以计算出每32*32个数的最大值，一共256/32个
        Reg::ReduceMaxWithDataBlock(expMaxDim, expMaxDim, maskAll);

        Reg::And(expTmpMaxDim, expMaxDim, expMaskBF16, maskAll);

        // inf/nan值单独处理，结果为E8M0的nan
        Reg::Compare<uint16_t, CMPMODE::NE>(infMask, expTmpMaxDim, expMaskBF16, maskAll);
        // 0值单独处理，结果为0
        Reg::Compare<uint16_t, CMPMODE::NE>(zeroMask, expTmpMaxDim, zero, maskAll);
        // 指数位不足被量化类型的ele_max时，为invalid场景，结果为0
        Reg::Compare<uint16_t, CMPMODE::LT>(invalidDataMask, expTmpMaxDim, yMaxExp, maskAll);
        Reg::Select<uint16_t>(expMaxDim, mantissaForMax, expMaxDim, invalidDataMask);

        // 尾数位进位
        Reg::Sub(expMaxDim, expMaxDim, mantissaForMax, maskAll);
        // 右移7位，BF16的指数位移到了末8位
        Reg::ShiftRights(mxScaleB16, expMaxDim, SHR_NUM_FOR_BF16, maskAll);
        Reg::Select<uint16_t>(mxScaleB16, mxScaleB16, nanE8M0, infMask);
        Reg::Select<uint16_t>(mxScaleB16, mxScaleB16, zero, zeroMask);

        Reg::Pack<uint8_t, uint16_t, Reg::HighLowPart::LOWEST>(mxScale1B8, mxScaleB16);
        Reg::LoadAlign(gatherIndex, gatherIndexAddr);
        Reg::Gather(mxScale2B8, mxScale1B8, gatherIndex);
        Reg::StoreAlign(mxScale2Addr, mxScale2B8, maskB8);
        Reg::And(expTmpMaxDim, expMaxDim, expMaskBF16, maskAll);
        // compute  1/scale
        // 只有在E1M2时，yMaxExp=0，xTmp0ExpBF16可能会等于biasE8M0
        Reg::Compare<uint16_t, CMPMODE::EQ>(invalidDataMask, expTmpMaxDim, biasE8M0, maskAll);
        Reg::Sub(reversedExp, biasE8M0, expTmpMaxDim, maskAll);
        Reg::Select<uint16_t>(reversedExp, reversedExp, nanBF16, infMask);
        Reg::Select<uint16_t>(reversedExp, reversedExp, zero, zeroMask);
        Reg::Select<uint16_t>(reversedExp, specialExp, reversedExp, invalidDataMask);
        // -1轴scale循环搬出
        for (uint16_t i = 0; i < rowNum; i++) {
            Reg::StoreAlign<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE>(
                mxScale1Addr, mxScale1B8, ubBlockSize_ / sizeof(uint8_t), maskReduceB8);
        }
        Reg::StoreAlign(mxScaleReciprocalAddr, reversedExp, maskReduceB16);
    }
}

template <typename T, typename U, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void DynamicBlockMxQuantBase<T, U, roundMode, scaleAlg>::ComputeYVf(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ T* xAddr, __ubuf__ uint16_t* mxScaleReciprocalAddr,
    __ubuf__ uint8_t* yAddr)
{
    if constexpr (IsSameType<U, fp4x2_e2m1_t>::value || IsSameType<U, fp4x2_e1m2_t>::value) {
        // 算Y1是交织处理
        ComputeY1ToFP4(dataLen, blockCount, xAddr, mxScaleReciprocalAddr, yAddr);
    } else {
        ComputeY1ToFP8(dataLen, blockCount, xAddr, mxScaleReciprocalAddr, yAddr);
    }
}

template <typename T, typename U, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void DynamicBlockMxQuantBase<T, U, roundMode, scaleAlg>::ComputeY1ToFP4(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ T* xAddr, __ubuf__ uint16_t* mxScale1ReciprocalAddr,
    __ubuf__ uint8_t* y1Addr)
{
    __VEC_SCOPE__
    {
        Reg::MaskReg dataMaskB8 = Reg::CreateMask<uint8_t>();
        Reg::MaskReg dataMaskB16 = Reg::CreateMask<half>();
        Reg::MaskReg dataMaskB32 = Reg::CreateMask<float>();
        Reg::RegTensor<uint16_t> scaleForMulFP16;
        Reg::RegTensor<T> x0;
        Reg::RegTensor<T> x1;

        Reg::RegTensor<float> x0ZeroFP32;
        Reg::RegTensor<float> x0OneFP32;
        Reg::RegTensor<float> x1ZeroFP32;
        Reg::RegTensor<float> x1OneFP32;
        Reg::RegTensor<float> scaleForMulZeroFP32;
        Reg::RegTensor<float> scaleForMulOneFP32;

        Reg::RegTensor<bfloat16_t> x0ZeroBF16;
        Reg::RegTensor<bfloat16_t> x0OneBF16;
        Reg::RegTensor<bfloat16_t> x1ZeroBF16;
        Reg::RegTensor<bfloat16_t> x1OneBF16;

        Reg::RegTensor<U> x0FP4;
        Reg::RegTensor<U> x1FP4;

        Reg::DataCopy<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_E2B_B16>(
            scaleForMulFP16, mxScale1ReciprocalAddr, ubBlockSize_ / sizeof(uint16_t));
        for (uint16_t i = 0; i < blockCount; i++) {
            Reg::DataCopy<T, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
                x0, x1, xAddr, vlForHalfNumber_ * DIGIT_TWO);
            if constexpr (IsSameType<T, half>::value) {
                Reg::Cast<float, bfloat16_t, castTraitXdtypetoFp32Zero>(
                    scaleForMulZeroFP32, (Reg::RegTensor<bfloat16_t>&)scaleForMulFP16, dataMaskB16);

                // x0 cast to bf16
                Reg::Cast<float, T, castTraitXdtypetoFp32Zero>(x0ZeroFP32, x0, dataMaskB16);
                Reg::Cast<float, T, castTraitXdtypetoFp32One>(x0OneFP32, x0, dataMaskB16);

                Reg::Mul(x0ZeroFP32, scaleForMulZeroFP32, x0ZeroFP32, dataMaskB32);
                Reg::Mul(x0OneFP32, scaleForMulZeroFP32, x0OneFP32, dataMaskB32);
                ComputeFP4FromHalf(x0ZeroFP32);
                ComputeFP4FromHalf(x0OneFP32);
                Reg::Cast<bfloat16_t, float, castTraitFp32toBF16>(x0ZeroBF16, x0ZeroFP32, dataMaskB32);
                Reg::Cast<bfloat16_t, float, castTraitFp32toBF16>(x0OneBF16, x0OneFP32, dataMaskB32);
                Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(
                    (Reg::RegTensor<uint16_t>&)x0ZeroBF16, (Reg::RegTensor<uint32_t>&)x0ZeroBF16);
                Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(
                    (Reg::RegTensor<uint16_t>&)x0OneBF16, (Reg::RegTensor<uint32_t>&)x0OneBF16);
                Reg::Interleave(x0ZeroBF16, x0OneBF16, x0ZeroBF16, x0OneBF16);

                // x1 cast to bf16
                Reg::Cast<float, T, castTraitXdtypetoFp32Zero>(x1ZeroFP32, x1, dataMaskB16);
                Reg::Cast<float, T, castTraitXdtypetoFp32One>(x1OneFP32, x1, dataMaskB16);

                Reg::Mul(x1ZeroFP32, scaleForMulZeroFP32, x1ZeroFP32, dataMaskB32);
                Reg::Mul(x1OneFP32, scaleForMulZeroFP32, x1OneFP32, dataMaskB32);
                ComputeFP4FromHalf(x1ZeroFP32);
                ComputeFP4FromHalf(x1OneFP32);
                Reg::Cast<bfloat16_t, float, castTraitFp32toBF16>(x1ZeroBF16, x1ZeroFP32, dataMaskB32);
                Reg::Cast<bfloat16_t, float, castTraitFp32toBF16>(x1OneBF16, x1OneFP32, dataMaskB32);
                Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(
                    (Reg::RegTensor<uint16_t>&)x1ZeroBF16, (Reg::RegTensor<uint32_t>&)x1ZeroBF16);
                Reg::Pack<uint16_t, uint32_t, Reg::HighLowPart::LOWEST>(
                    (Reg::RegTensor<uint16_t>&)x1OneBF16, (Reg::RegTensor<uint32_t>&)x1OneBF16);
                Reg::Interleave(x1ZeroBF16, x1OneBF16, x1ZeroBF16, x1OneBF16);

                // interleave x0 and x1
                Reg::Interleave(x0ZeroBF16, x1ZeroBF16, x0ZeroBF16, x1ZeroBF16);
                Reg::Cast<U, bfloat16_t, castTraitBF16toFp4>(x0FP4, x0ZeroBF16, dataMaskB16);
                Reg::Cast<U, bfloat16_t, castTraitBF16toFp4>(x1FP4, x1ZeroBF16, dataMaskB16);
            } else {
                Reg::Mul(x0, x0, (Reg::RegTensor<T>&)scaleForMulFP16, dataMaskB16);
                Reg::Mul(x1, x1, (Reg::RegTensor<T>&)scaleForMulFP16, dataMaskB16);
                Reg::Interleave(x0, x1, x0, x1);
                Reg::Cast<U, T, castTraitBF16toFp4>(x0FP4, x0, dataMaskB16);
                Reg::Cast<U, T, castTraitBF16toFp4>(x1FP4, x1, dataMaskB16);
            }

            // copy to ub
            Reg::DataCopy<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK4_B32>(
                y1Addr, (Reg::RegTensor<uint8_t>&)x0FP4, OUT_ELE_NUM_ONE_BLK, dataMaskB8);
            Reg::DataCopy<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK4_B32>(
                y1Addr, (Reg::RegTensor<uint8_t>&)x1FP4, OUT_ELE_NUM_ONE_BLK, dataMaskB8);
        }
    }
    return;
}

template <typename T, typename U, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void DynamicBlockMxQuantBase<T, U, roundMode, scaleAlg>::ComputeY1ToFP8(
    uint16_t dataLen, uint16_t blockCount, __ubuf__ T* xAddr, __ubuf__ uint16_t* mxScale1ReciprocalAddr,
    __ubuf__ uint8_t* y1Addr)
{
    __VEC_SCOPE__
    {
        Reg::MaskReg maskAll = Reg::CreateMask<uint16_t, Reg::MaskPattern::ALL>();
        Reg::RegTensor<uint16_t> scaleForMulFP16;
        Reg::RegTensor<float> scaleForMulFP32;
        Reg::RegTensor<T> x0;
        Reg::RegTensor<T> x1;
        Reg::RegTensor<bfloat16_t> x0BF16;
        Reg::RegTensor<bfloat16_t> x1BF16;
        Reg::RegTensor<float> x0ZeroFP32;
        Reg::RegTensor<float> x0OneFP32;
        Reg::RegTensor<float> x1ZeroFP32;
        Reg::RegTensor<float> x1OneFP32;
        Reg::RegTensor<U> x0ZeroFP8;
        Reg::RegTensor<U> x0OneFP8;
        Reg::RegTensor<U> x1ZeroFP8;
        Reg::RegTensor<U> x1OneFP8;

        Reg::DataCopy<uint16_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_E2B_B16>(
            scaleForMulFP16, mxScale1ReciprocalAddr, ubBlockSize_ / sizeof(uint16_t));
        for (uint16_t i = 0; i < blockCount; i++) {
            Reg::DataCopy<T, Reg::PostLiteral::POST_MODE_UPDATE, Reg::LoadDist::DIST_DINTLV_B16>(
                x0, x1, xAddr, vlForHalfNumber_ * DIGIT_TWO);
            if constexpr (IsSameType<T, half>::value) {
                Reg::Cast<float, T, castTraitXdtypetoFp32Zero>(x0ZeroFP32, x0, maskAll);
                Reg::Cast<float, T, castTraitXdtypetoFp32One>(x0OneFP32, x0, maskAll);
                Reg::Cast<float, bfloat16_t, castTraitXdtypetoFp32Zero>(
                    scaleForMulFP32, (Reg::RegTensor<bfloat16_t>&)scaleForMulFP16, maskAll);
                Reg::Mul(x0ZeroFP32, x0ZeroFP32, scaleForMulFP32, maskAll);
                Reg::Mul(x0OneFP32, x0OneFP32, scaleForMulFP32, maskAll);
                Reg::Interleave(x0ZeroFP32, x0OneFP32, x0ZeroFP32, x0OneFP32);
                Reg::Cast<float, T, castTraitXdtypetoFp32Zero>(x1ZeroFP32, x1, maskAll);
                Reg::Cast<float, T, castTraitXdtypetoFp32One>(x1OneFP32, x1, maskAll);
                Reg::Mul(x1ZeroFP32, x1ZeroFP32, scaleForMulFP32, maskAll);
                Reg::Mul(x1OneFP32, x1OneFP32, scaleForMulFP32, maskAll);
                Reg::Interleave(x1ZeroFP32, x1OneFP32, x1ZeroFP32, x1OneFP32);
                Reg::Interleave(x0ZeroFP32, x1ZeroFP32, x0ZeroFP32, x1ZeroFP32);
                Reg::Interleave(x0OneFP32, x1OneFP32, x0OneFP32, x1OneFP32);
                Reg::Cast<U, float, castTraitFp32toYdtype>(x0ZeroFP8, x0ZeroFP32, maskAll);
                Reg::Cast<U, float, castTraitFp32toYdtype>(x0OneFP8, x1ZeroFP32, maskAll);
                Reg::Cast<U, float, castTraitFp32toYdtype>(x1ZeroFP8, x0OneFP32, maskAll);
                Reg::Cast<U, float, castTraitFp32toYdtype>(x1OneFP8, x1OneFP32, maskAll);
            } else {
                Reg::Mul(x0, x0, (Reg::RegTensor<T>&)scaleForMulFP16, maskAll);
                Reg::Mul(x1, x1, (Reg::RegTensor<T>&)scaleForMulFP16, maskAll);
                Reg::Interleave(x0, x1, x0, x1);
                Reg::Cast<float, T, castTraitXdtypetoFp32Zero>(x0ZeroFP32, x0, maskAll);
                Reg::Cast<float, T, castTraitXdtypetoFp32One>(x0OneFP32, x0, maskAll);
                Reg::Interleave(x0ZeroFP32, x0OneFP32, x0ZeroFP32, x0OneFP32);
                Reg::Cast<U, float, castTraitFp32toYdtype>(x0ZeroFP8, x0ZeroFP32, maskAll);
                Reg::Cast<U, float, castTraitFp32toYdtype>(x0OneFP8, x0OneFP32, maskAll);
                Reg::Cast<float, T, castTraitXdtypetoFp32Zero>(x1ZeroFP32, x1, maskAll);
                Reg::Cast<float, T, castTraitXdtypetoFp32One>(x1OneFP32, x1, maskAll);
                Reg::Interleave(x1ZeroFP32, x1OneFP32, x1ZeroFP32, x1OneFP32);
                Reg::Cast<U, float, castTraitFp32toYdtype>(x1ZeroFP8, x1ZeroFP32, maskAll);
                Reg::Cast<U, float, castTraitFp32toYdtype>(x1OneFP8, x1OneFP32, maskAll);
            }
            Reg::DataCopy<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK4_B32>(
                y1Addr, (Reg::RegTensor<uint8_t>&)x0ZeroFP8, OUT_ELE_NUM_ONE_BLK, maskAll);
            Reg::DataCopy<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK4_B32>(
                y1Addr, (Reg::RegTensor<uint8_t>&)x0OneFP8, OUT_ELE_NUM_ONE_BLK, maskAll);
            Reg::DataCopy<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK4_B32>(
                y1Addr, (Reg::RegTensor<uint8_t>&)x1ZeroFP8, OUT_ELE_NUM_ONE_BLK, maskAll);
            Reg::DataCopy<uint8_t, Reg::PostLiteral::POST_MODE_UPDATE, Reg::StoreDist::DIST_PACK4_B32>(
                y1Addr, (Reg::RegTensor<uint8_t>&)x1OneFP8, OUT_ELE_NUM_ONE_BLK, maskAll);
        }
    }
    return;
}

template <typename T, typename U, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void DynamicBlockMxQuantBase<T, U, roundMode, scaleAlg>::ComputeFP4FromHalf(
    Reg::RegTensor<float>& Reg)
{
    Reg::MaskReg pregAll32 = Reg::CreateMask<uint32_t, Reg::MaskPattern::ALL>();
    Reg::MaskReg zeroMask;
    Reg::MaskReg specialMask;
    Reg::MaskReg negInfMask;

    Reg::RegTensor<int32_t> negZero;
    Reg::RegTensor<int32_t> maxExpFP32;
    Reg::RegTensor<int32_t> exp0FP32;
    Reg::RegTensor<int32_t> exp1FP32;

    Reg::Duplicate(negZero, NEG_ZERO);

    Reg::Compare<int32_t, CMPMODE::EQ>(negInfMask, (Reg::RegTensor<int32_t>&)Reg, negZero, pregAll32);
    if constexpr (IsSameType<U, fp4x2_e1m2_t>::value) {
        Reg::Muls(Reg, Reg, FOUR, pregAll32);
        Reg::CompareScalar<float, CMPMODE::LT>(specialMask, Reg, 0, pregAll32);
        Reg::Truncate<float, roundMode>(Reg, Reg, pregAll32);
        Reg::Muls(Reg, Reg, ONE_FOURTH, pregAll32);
    } else {
        // fp4x2_e2m1
        Reg::Duplicate(maxExpFP32, MAX_EXP_FOR_FP32);
        Reg::And(exp0FP32, (Reg::RegTensor<int32_t>&)Reg, maxExpFP32, pregAll32);
        Reg::ShiftRights(exp0FP32, exp0FP32, SHR_NUM_FOR_FP32, pregAll32);
        Reg::Adds(exp0FP32, exp0FP32, FP32_BIAS_NEG, pregAll32);
        Reg::Maxs(exp0FP32, exp0FP32, 0, pregAll32);
        Reg::Adds(exp0FP32, exp0FP32, NEG_ONE, pregAll32);
        Reg::Muls(exp1FP32, exp0FP32, NEG_ONE, pregAll32);
        Reg::Adds(exp1FP32, exp1FP32, FP32_BIAS, pregAll32);
        Reg::ShiftLefts(exp1FP32, exp1FP32, SHR_NUM_FOR_FP32, pregAll32);

        Reg::Mul(Reg, Reg, (Reg::RegTensor<float>&)exp1FP32, pregAll32);
        Reg::Adds(exp0FP32, exp0FP32, FP32_BIAS, pregAll32);
        Reg::ShiftLefts(exp0FP32, exp0FP32, SHR_NUM_FOR_FP32, pregAll32);
        Reg::CompareScalar<float, CMPMODE::LT>(specialMask, Reg, 0, pregAll32);
        Reg::Truncate<float, roundMode>(Reg, Reg, pregAll32);
        Reg::Mul(Reg, Reg, (Reg::RegTensor<float>&)exp0FP32, pregAll32);
    }
    Reg::CompareScalar<float, CMPMODE::EQ>(zeroMask, Reg, 0, pregAll32);
    Reg::MaskAnd(zeroMask, specialMask, zeroMask, pregAll32);
    Reg::MaskOr(zeroMask, negInfMask, zeroMask, pregAll32);
    Reg::Select<int32_t>((Reg::RegTensor<int32_t>&)Reg, negZero, (Reg::RegTensor<int32_t>&)Reg, zeroMask);
}

template <typename T, typename U, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void DynamicBlockMxQuantBase<T, U, roundMode, scaleAlg>::Compute(
    int32_t blockRow, int32_t blockCol, int32_t blockRowNum, int32_t blockColNum, int64_t nowRowBlock)
{
    LocalTensor<T> inLocal = inQueue_.DeQue<T>();
    __local_mem__ T* xLocalAddr = (__local_mem__ T*)inLocal.GetPhyAddr();

    LocalTensor<uint8_t> scale1Local = scale1Queue_.AllocTensor<uint8_t>();
    __local_mem__ uint8_t* scale1LocalAddr = (__local_mem__ uint8_t*)scale1Local.GetPhyAddr();

    LocalTensor<uint8_t> scale2Local = scale2Queue_.AllocTensor<uint8_t>();
    __local_mem__ uint8_t* scale2LocalAddr = (__local_mem__ uint8_t*)scale2Local.GetPhyAddr();

    LocalTensor<uint8_t> outLocal = outQueue_.AllocTensor<uint8_t>();
    __local_mem__ uint8_t* outLocalAddr = (__local_mem__ uint8_t*)outLocal.GetPhyAddr();

    LocalTensor<uint16_t> mxScaleReciprocalLocal = mxScaleReciprocalBuf_.Get<uint16_t>();
    auto mxScaleReciprocalAddr = (__local_mem__ uint16_t*)mxScaleReciprocalLocal.GetPhyAddr();

    LocalTensor<uint8_t> tempIndexLocal = tempIndexBuf_.Get<uint8_t>();
    auto tempIndexLocalAddr = (__local_mem__ uint8_t*)tempIndexLocal.GetPhyAddr();

    int64_t ubOffset = 0;
    int64_t yOffset = 0;
    int64_t inputColAlign = ubBlockSize_ / sizeof(T);
    int64_t outputColAlign = ubBlockSize_ / sizeof(uint8_t);
    int64_t scale1Offset = 0;
    int64_t scale2Offset = 0;
    int64_t scaleReciprocalOffset = 0;

    // yoffset偏移因子，fp4要除2
    int64_t yOffsetFactor = 1;
    if constexpr ((IsSameType<U, fp8_e4m3fn_t>::value) || (IsSameType<U, fp8_e5m2_t>::value)) {
        yOffsetFactor = 1;
    } else {
        // 两个fp4合成一个fp8输出，所以要/2
        yOffsetFactor = 2;
    }

    for (int32_t rowIdx = 0; rowIdx < blockRow; rowIdx++) {
        int64_t tmpBlock = nowRowBlock + rowIdx;
        // 当前block的行数
        int32_t rowBlockSize = 0;

        int32_t row1BlockSize = 0;
        int32_t row2BlockSize = 0;
        if (tmpBlock % singleBatchRowBlockLoopNum_ == singleBatchRowBlockLoopNum_ - 1) {
            rowBlockSize = tailBlockSizeRow_;
            row1BlockSize = tailBlockSizeRow_ < blockSizeRow_ ? tailBlockSizeRow_ : blockSizeRow_;
            row2BlockSize = tailBlockSizeRow_ < blockSizeRow_ ? 0 : tailBlockSizeRow_ - blockSizeRow_;
        } else {
            rowBlockSize = blockH_;
            row1BlockSize = blockSizeRow_;
            row2BlockSize = blockSizeRow_;
        }
        for (int32_t colIdx = 0; colIdx < blockCol; colIdx++) {
            int32_t colBlockSize = colIdx == blockCol - 1 ? blockColNum - colIdx * blockW_ : blockW_;

            if constexpr (scaleAlg == 2 && (IsSameType<U, fp4x2_e2m1_t>::value)) {
                ComputeDdr(
                    row1BlockSize, colBlockSize, xLocalAddr + ubOffset, scale1LocalAddr + scale1Offset,
                    scale2LocalAddr + scale2Offset, mxScaleReciprocalAddr + scaleReciprocalOffset,
                    outLocalAddr + yOffset, tempIndexLocalAddr);
            } else {
                // 计算基本块中的第一行的block32*256
                ComputeOcp(
                    row1BlockSize, colBlockSize, xLocalAddr + ubOffset, scale1LocalAddr + scale1Offset,
                    scale2LocalAddr + scale2Offset, mxScaleReciprocalAddr + scaleReciprocalOffset,
                    outLocalAddr + yOffset, tempIndexLocalAddr);
            }

            if constexpr ((IsSameType<U, fp8_e4m3fn_t>::value) || (IsSameType<U, fp8_e5m2_t>::value)) {
                yOffsetFactor = 1;
            } else {
                // 两个fp4合成一个fp8输出，所以要/2
                yOffsetFactor = 2;
            }
            ComputeYVf(
                colBlockSize, row1BlockSize, xLocalAddr + ubOffset, mxScaleReciprocalAddr + scaleReciprocalOffset,
                outLocalAddr + yOffset);
            scale1Offset = scale1Offset + row1BlockSize * oneBlockCountB8_;
            scale2Offset = scale2Offset + blockW_;
            ubOffset = ubOffset + row1BlockSize * blockW_;
            scaleReciprocalOffset = scaleReciprocalOffset + oneBlockCountB16_;
            yOffset = yOffset + row1BlockSize * blockW_ / yOffsetFactor;
            if (row2BlockSize == 0) {
                Duplicate<uint8_t>(scale2Local[scale2Offset], (uint8_t)0, blockW_);
                scale2Offset = scale2Offset + blockW_;
            } else {
                if constexpr (scaleAlg == 2 && (IsSameType<U, fp4x2_e2m1_t>::value)) {
                    ComputeDdr(
                        row2BlockSize, colBlockSize, xLocalAddr + ubOffset, scale1LocalAddr + scale1Offset,
                        scale2LocalAddr + scale2Offset, mxScaleReciprocalAddr + scaleReciprocalOffset,
                        outLocalAddr + yOffset, tempIndexLocalAddr);
                } else {
                    ComputeOcp(
                        row2BlockSize, colBlockSize, xLocalAddr + ubOffset, scale1LocalAddr + scale1Offset,
                        scale2LocalAddr + scale2Offset, mxScaleReciprocalAddr + scaleReciprocalOffset,
                        outLocalAddr + yOffset, tempIndexLocalAddr);
                }

                ComputeYVf(
                    colBlockSize, row2BlockSize, xLocalAddr + ubOffset, mxScaleReciprocalAddr + scaleReciprocalOffset,
                    outLocalAddr + yOffset);
                scale1Offset = scale1Offset + row2BlockSize * oneBlockCountB8_;
                scale2Offset = scale2Offset + blockW_;
                ubOffset = ubOffset + row2BlockSize * blockW_;
                scaleReciprocalOffset = scaleReciprocalOffset + oneBlockCountB16_;
                yOffset = yOffset + row2BlockSize * blockW_ / yOffsetFactor;
            }
            // -2轴交织
            Interleave(
                scale2Local[scale2Offset - (DIGIT_TWO * blockW_)], scale2Local[scale2Offset - blockW_],
                scale2Local[scale2Offset - (DIGIT_TWO * blockW_)], scale2Local[scale2Offset - blockW_], blockW_);
        }
    }

    inQueue_.FreeTensor(inLocal);
    outQueue_.EnQue(outLocal);
    scale1Queue_.EnQue(scale1Local);
    scale2Queue_.EnQue(scale2Local);
}

template <typename T, typename U, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void DynamicBlockMxQuantBase<T, U, roundMode, scaleAlg>::CopyOut(
    int32_t blockRow, int32_t blockCol, int32_t blockRowNum, int32_t blockColNum, int64_t baseXGmOffset,
    int64_t baseScale1Offset, int64_t baseScale2Offset, int64_t nowRowBlock)
{
    LocalTensor<uint8_t> outLocal = outQueue_.DeQue<uint8_t>();
    LocalTensor<uint8_t> scale1Local = scale1Queue_.DeQue<uint8_t>();
    LocalTensor<uint8_t> scale2Local = scale2Queue_.DeQue<uint8_t>();
    int64_t outputColAlign = ubBlockSize_ / sizeof(uint8_t);
    // yGm偏移地址
    int64_t outGmOffset = 0;
    // yUb偏移地址
    int64_t outUbOffset = 0;
    // scaleGm偏移地址
    int64_t scale1GmOffset = 0;
    int64_t scale2GmOffset = 0;
    // scaleUb偏移地址
    int64_t scale1UbOffset = 0;
    int64_t scale2UbOffset = 0;
    int32_t rowBlockSize = 0;
    int64_t outGmRowOffset = 0;
    int64_t outScale1GmRowOffset = 0;
    int64_t outScale1UbRowOffset = 0;
    uint32_t outBlockLen = 0;
    uint32_t srcStride = 0;
    uint32_t dstStride = 0;
    int64_t yOffset = 0;
    for (int32_t rowIdx = 0; rowIdx < blockRow; rowIdx++) {
        int64_t tmpBlock = nowRowBlock + rowIdx;
        if (tmpBlock % singleBatchRowBlockLoopNum_ == singleBatchRowBlockLoopNum_ - 1) {
            rowBlockSize = tailBlockSizeRow_;
        } else {
            rowBlockSize = blockH_;
        }

        for (int32_t colIdx = 0; colIdx < blockCol; colIdx++) {
            outGmOffset = baseXGmOffset + outGmRowOffset + colIdx * blockW_;
            scale1UbOffset = (outScale1UbRowOffset * blockCol + rowBlockSize * colIdx) * ubBlockSize_;
            scale1GmOffset = baseScale1Offset + outScale1GmRowOffset + colIdx * blockW_ / blockSizeCol_;
            int32_t colBlockSize = colIdx == blockCol - 1 ? blockColNum - colIdx * blockW_ : blockW_;
            if constexpr (IsSameType<U, fp4x2_e2m1_t>::value || IsSameType<U, fp4x2_e1m2_t>::value) {
                outBlockLen = colBlockSize / DIGIT_TWO * sizeof(uint8_t);
                srcStride = ((blockW_ - colBlockSize) / DIGIT_TWO * sizeof(uint8_t) / ubBlockSize_);
                dstStride = (colNum_ - colBlockSize) / DIGIT_TWO * sizeof(uint8_t);
                yOffset = outGmOffset / DIGIT_TWO;
            } else {
                outBlockLen = colBlockSize * sizeof(uint8_t);
                srcStride = ((blockW_ - colBlockSize) * sizeof(uint8_t) / ubBlockSize_);
                dstStride = (colNum_ - colBlockSize) * sizeof(uint8_t);
                yOffset = outGmOffset;
            }

            uint32_t scale1OutLen =
                ops::CeilAlign(static_cast<int64_t>(colBlockSize), blockSizeCol_ * DIGIT_TWO) / blockSizeCol_;
            DataCopyExtParams scale1CopyParams = {
                static_cast<uint16_t>(rowBlockSize), static_cast<uint32_t>(scale1OutLen), 0,
                colScaleNum_ - scale1OutLen, 0};
            DataCopyPad(mxScale1Gm_[scale1GmOffset], scale1Local[scale1UbOffset], scale1CopyParams);

            DataCopyExtParams outCopyParams = {
                static_cast<uint16_t>(rowBlockSize), static_cast<uint32_t>(outBlockLen), srcStride,
                static_cast<uint32_t>(dstStride), 0};

            DataCopyPad(yGm_[yOffset], outLocal[outUbOffset], outCopyParams);
            if constexpr ((IsSameType<U, fp8_e4m3fn_t>::value) || (IsSameType<U, fp8_e5m2_t>::value)) {
                outUbOffset = outUbOffset + rowBlockSize * blockW_;
            } else {
                // 两个fp4合成一个fp8输出，所以要/2
                outUbOffset = outUbOffset + rowBlockSize * blockW_ / 2;
            }
        }

        outGmRowOffset = outGmRowOffset + rowBlockSize * colNum_;
        outScale1GmRowOffset = outScale1GmRowOffset + rowBlockSize * colScaleNum_;
        outScale1UbRowOffset = outScale1UbRowOffset + rowBlockSize;
    }

    uint32_t scale2SrcStride = DIGIT_TWO * ops::CeilDiv(static_cast<int64_t>(blockW_), ubBlockSize_) -
                               ops::CeilDiv(static_cast<int64_t>(DIGIT_TWO * blockColNum), ubBlockSize_);

    DataCopyExtParams scale2CopyParams = {
        static_cast<uint16_t>(blockRow), static_cast<uint32_t>(blockColNum * DIGIT_TWO), scale2SrcStride,
        DIGIT_TWO * (colNum_ - blockColNum), 0};
    DataCopyPad(mxScale2Gm_[baseScale2Offset], scale2Local, scale2CopyParams);

    outQueue_.FreeTensor(outLocal);
    scale1Queue_.FreeTensor(scale1Local);
    scale2Queue_.FreeTensor(scale2Local);
}

} // namespace DynamicBlockMxQuant
#endif // DYNAMIC_BLOCK_MX_QUANT_BASE_H