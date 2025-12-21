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
 * \file dynamic_block_quant_single_row_kernel.h
 * \brief
 */

#ifndef DYNAMIC_BLOCK_QUNANT_SINGLE_ROW_KERNEL_H
#define DYNAMIC_BLOCK_QUNANT_SINGLE_ROW_KERNEL_H

#include "kernel_operator.h"
#include "../inc/platform.h"
#include "dynamic_block_quant_common.h"

namespace DynamicBlockQuant {

using namespace AscendC;

template <typename IN_TYPE, typename OUT_TYPE, int64_t ROUND_MODE>
class DynamicBlockQuantSingleRow {
public:
    __aicore__ inline DynamicBlockQuantSingleRow(TPipe* pipe)
    {
        tPipe_ = pipe;
    }
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR scale, const DynamicBlockQuantTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessRow(uint64_t rowBaseOffset, uint64_t rowScaleBaseOffset, uint64_t rowSize);
    __aicore__ inline void ProcessCol(
        uint64_t colBaseOffset, uint64_t colScaleBaseOffset, uint64_t rowSize, uint64_t colSize);
    __aicore__ inline void CopyIn(uint64_t offset, uint64_t rowSize, uint64_t colSize);
    __aicore__ inline void Compute(uint64_t colSize, uint64_t rowBlockSize, uint64_t colBlockSize);
    __aicore__ inline void CopyOut(uint64_t offset, uint64_t rowSize, uint64_t colSize);
    __aicore__ inline void CopyOutScale(uint64_t offset, uint64_t rowSize, uint64_t colSize);
    __aicore__ inline void ComputeVF(
        __local_mem__ OUT_TYPE* outLocal, __local_mem__ float* scaleLocal, __local_mem__ IN_TYPE* xLocal,
        uint64_t colSize, uint64_t rowBlockSize, uint64_t colBlockSize);
    __aicore__ inline void ComputeVFBF16(
        __local_mem__ OUT_TYPE* outLocal, __local_mem__ float* scaleLocal, __local_mem__ IN_TYPE* xLocal,
        uint64_t colSize, uint64_t rowBlockSize, uint64_t colBlockSize);

private:
    TPipe* tPipe_ = nullptr;
    const DynamicBlockQuantTilingData* tilingData_ = nullptr;

    TQue<QuePosition::VECIN, 1> inQueue_;
    GlobalTensor<IN_TYPE> xGm_;
    TQue<QuePosition::VECOUT, 1> outQueue_;
    GlobalTensor<OUT_TYPE> yGm_;
    TQue<QuePosition::VECOUT, 1> scaleQueue_;
    GlobalTensor<float> scaleGm_;

    int64_t blockIdx_ = 0;

    int64_t rowCoreNum_ = 0; // 核内行数
    int64_t colCoreNum_ = 0; // 核内列数

    int64_t rowUbNum_ = 0; // 单次UB行数
    int64_t colUbNum_ = 0; // 单次UB列数

    int64_t rowUbBlockNum_ = 0; // UB内行方向循环次数
    int64_t colUbBlockNum_ = 0; // UB内列方向循环次数

    static constexpr AscendC::MicroAPI::CastTrait castTrait32toh8Zero = []() {
        if constexpr (ROUND_MODE == 1 || ROUND_MODE == 4) {
            return AscendC::MicroAPI::CastTrait{
                AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
                AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};
        } else if constexpr (ROUND_MODE == 7) {
            return AscendC::MicroAPI::CastTrait{
                AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
                AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_HYBRID};
        }
    }();
};

template <typename IN_TYPE, typename OUT_TYPE, int64_t ROUND_MODE>
__aicore__ inline void DynamicBlockQuantSingleRow<IN_TYPE, OUT_TYPE, ROUND_MODE>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR scale, const DynamicBlockQuantTilingData* tilingData)
{
    tilingData_ = tilingData;
    blockIdx_ = GetBlockIdx();
    if (blockIdx_ > tilingData_->usedCoreNum) {
        return;
    }

    // Compute BlockOffset
    int64_t coreRowIdx = blockIdx_ / tilingData_->colTileNum;
    int64_t coreColIdx = blockIdx_ % tilingData_->colTileNum;

    int64_t rowBlockOffset = 0; // 行方向偏移的块数
    int64_t colBlockOffset = 0; // 列方向偏移的块数

    if (coreRowIdx < tilingData_->rowNormalCoreNum) {
        // normal core
        rowBlockOffset = coreRowIdx * tilingData_->normalCoreRowTileNum;
        rowCoreNum_ =
            Min(tilingData_->normalCoreRowTileNum * tilingData_->blockSizeRow,
                tilingData_->rowNum - rowBlockOffset * tilingData_->blockSizeRow);
    } else {
        rowBlockOffset = tilingData_->rowNormalCoreNum * tilingData_->normalCoreRowTileNum +
                         (coreRowIdx - tilingData_->rowNormalCoreNum) * tilingData_->tailCoreRowTileNum;
        rowCoreNum_ =
            Min(tilingData_->tailCoreRowTileNum * tilingData_->blockSizeRow,
                tilingData_->rowNum - rowBlockOffset * tilingData_->blockSizeRow);
    }

    if (coreColIdx < tilingData_->colNormalCoreNum) {
        colBlockOffset = coreColIdx * tilingData_->normalCoreColTileNum;
        colCoreNum_ =
            Min(tilingData_->normalCoreColTileNum * tilingData_->blockSizeCol,
                tilingData_->colNum - colBlockOffset * tilingData_->blockSizeCol);
    } else {
        colBlockOffset = tilingData_->colNormalCoreNum * tilingData_->normalCoreColTileNum +
                         (coreColIdx - tilingData_->colNormalCoreNum) * tilingData_->tailCoreColTileNum;
        colCoreNum_ =
            Min(tilingData_->tailCoreColTileNum * tilingData_->blockSizeCol,
                tilingData_->colNum - colBlockOffset * tilingData_->blockSizeCol);
    }

    int64_t blockOffset = rowBlockOffset * tilingData_->colNum + colBlockOffset * tilingData_->blockSizeCol;
    int64_t blockScaleOffset = rowBlockOffset * tilingData_->colBlockLoopNum + colBlockOffset;

    xGm_.SetGlobalBuffer((__gm__ IN_TYPE*)x + blockOffset);
    yGm_.SetGlobalBuffer((__gm__ OUT_TYPE*)y + blockOffset);
    scaleGm_.SetGlobalBuffer((__gm__ float*)scale + blockScaleOffset);

    // Compute real ub size
    rowUbNum_ = Min(tilingData_->rowUbBlockLoopNum * tilingData_->blockSizeRow, rowCoreNum_);
    colUbNum_ = Min(tilingData_->colUbBlockLoopNum * tilingData_->blockSizeCol, colCoreNum_);

    // Compute ub loop num
    rowUbBlockNum_ = Ceil(rowUbNum_, tilingData_->blockSizeRow);
    colUbBlockNum_ = Ceil(colUbNum_, tilingData_->blockSizeCol);

    // Init Buffer
    tPipe_->InitBuffer(inQueue_, 1, tilingData_->rowUbFactor * tilingData_->colUbFactor * sizeof(IN_TYPE));
    tPipe_->InitBuffer(outQueue_, 1, tilingData_->rowUbFactor * tilingData_->colUbFactor * sizeof(OUT_TYPE));
    tPipe_->InitBuffer(
        scaleQueue_, 1, Ceil(tilingData_->colUbBlockLoopNum, 64) * 64 * tilingData_->rowUbBlockLoopNum * sizeof(float));
}

template <typename IN_TYPE, typename OUT_TYPE, int64_t ROUND_MODE>
inline __aicore__ void DynamicBlockQuantSingleRow<IN_TYPE, OUT_TYPE, ROUND_MODE>::Process()
{
    if (blockIdx_ > tilingData_->usedCoreNum) {
        return;
    }

    uint64_t rowBaseOffset = 0;
    uint64_t rowScaleBaseOffset = 0;
    uint64_t rowLoopNum = rowCoreNum_ / rowUbNum_;
    for (uint64_t rowIdx = 0; rowIdx < rowLoopNum; rowIdx++) {
        ProcessRow(rowBaseOffset, rowScaleBaseOffset, rowUbNum_);
        rowBaseOffset += rowUbNum_ * tilingData_->colNum;
        rowScaleBaseOffset += rowUbBlockNum_ * tilingData_->colBlockLoopNum;
    }
    if (rowLoopNum * rowUbNum_ < rowCoreNum_) {
        ProcessRow(rowBaseOffset, rowScaleBaseOffset, rowCoreNum_ - rowLoopNum * rowUbNum_);
    }
}

template <typename IN_TYPE, typename OUT_TYPE, int64_t ROUND_MODE>
inline __aicore__ void DynamicBlockQuantSingleRow<IN_TYPE, OUT_TYPE, ROUND_MODE>::ProcessRow(
    uint64_t rowBaseOffset, uint64_t rowScaleBaseOffset, uint64_t rowSize)
{
    uint64_t colBaseOffset = 0;
    uint64_t colScaleBaseOffset = rowScaleBaseOffset;
    uint64_t colLoopNum = colCoreNum_ / colUbNum_;
    for (uint64_t colIdx = 0; colIdx < colLoopNum; colIdx++) {
        ProcessCol(rowBaseOffset + colBaseOffset, colScaleBaseOffset, rowSize, colUbNum_);
        colBaseOffset += colUbNum_;
        colScaleBaseOffset += colUbBlockNum_;
    }
    if (colLoopNum * colUbNum_ < colCoreNum_) {
        ProcessCol(rowBaseOffset + colBaseOffset, colScaleBaseOffset, rowSize, colCoreNum_ - colLoopNum * colUbNum_);
    }
}

template <typename IN_TYPE, typename OUT_TYPE, int64_t ROUND_MODE>
inline __aicore__ void DynamicBlockQuantSingleRow<IN_TYPE, OUT_TYPE, ROUND_MODE>::ProcessCol(
    uint64_t colBaseOffset, uint64_t colScaleBaseOffset, uint64_t rowSize, uint64_t colSize)
{
    uint64_t rowBlockSize = Ceil(rowSize, tilingData_->blockSizeRow);
    uint64_t colBlockSize = Ceil(colSize, tilingData_->blockSizeCol);
    CopyIn(colBaseOffset, rowSize, colSize);
    Compute(colSize, rowBlockSize, colBlockSize);
    CopyOut(colBaseOffset, rowSize, colSize);
    CopyOutScale(colScaleBaseOffset, rowBlockSize, colBlockSize);
}

template <typename IN_TYPE, typename OUT_TYPE, int64_t ROUND_MODE>
inline __aicore__ void DynamicBlockQuantSingleRow<IN_TYPE, OUT_TYPE, ROUND_MODE>::CopyIn(
    uint64_t offset, uint64_t rowSize, uint64_t colSize)
{
    LocalTensor<IN_TYPE> xLocal = inQueue_.AllocTensor<IN_TYPE>();
    uint32_t dstStride = colSize % 128 == 0 ? 0 : (128 - colSize % 128) * sizeof(IN_TYPE) / 32;
    DataCopyExtParams dataCopyExtParams = {
        static_cast<uint16_t>(rowSize), static_cast<uint32_t>(colSize * sizeof(IN_TYPE)),
        static_cast<uint32_t>((tilingData_->colNum - colSize) * sizeof(IN_TYPE)), static_cast<uint32_t>(dstStride), 0};
    DataCopyPadExtParams<IN_TYPE> dataCopyPadExtParams = {false, 0, 0, 0};
    DataCopyPad(xLocal, xGm_[offset], dataCopyExtParams, dataCopyPadExtParams);
    inQueue_.EnQue(xLocal);
}

template <typename IN_TYPE, typename OUT_TYPE, int64_t ROUND_MODE>
inline __aicore__ void DynamicBlockQuantSingleRow<IN_TYPE, OUT_TYPE, ROUND_MODE>::Compute(
    uint64_t colSize, uint64_t rowBlockSize, uint64_t colBlockSize)
{
    LocalTensor<IN_TYPE> xLocal = inQueue_.DeQue<IN_TYPE>();
    LocalTensor<OUT_TYPE> yLocal = outQueue_.AllocTensor<OUT_TYPE>();
    LocalTensor<float> scaleLocal = scaleQueue_.AllocTensor<float>();

    __local_mem__ IN_TYPE* xLocalPtr = (__local_mem__ IN_TYPE*)xLocal.GetPhyAddr();
    __local_mem__ OUT_TYPE* yLocalPtr = (__local_mem__ OUT_TYPE*)yLocal.GetPhyAddr();
    __local_mem__ float* scaleLocalPtr = (__local_mem__ float*)scaleLocal.GetPhyAddr();

    if constexpr (IsSameType<IN_TYPE, half>::value) {
        ComputeVF(yLocalPtr, scaleLocalPtr, xLocalPtr, colSize, rowBlockSize, colBlockSize);
    } else {
        ComputeVFBF16(yLocalPtr, scaleLocalPtr, xLocalPtr, colSize, rowBlockSize, colBlockSize);
    }

    inQueue_.FreeTensor(xLocal);
    outQueue_.EnQue(yLocal);
    scaleQueue_.EnQue(scaleLocal);
}

template <typename IN_TYPE, typename OUT_TYPE, int64_t ROUND_MODE>
inline __aicore__ void DynamicBlockQuantSingleRow<IN_TYPE, OUT_TYPE, ROUND_MODE>::CopyOut(
    uint64_t offset, uint64_t rowSize, uint64_t colSize)
{
    LocalTensor<OUT_TYPE> yLocal = outQueue_.DeQue<OUT_TYPE>();
    uint32_t srcStride = colSize % 128 == 0 ? 0 : (128 - colSize % 128) * sizeof(OUT_TYPE) / 32;
    DataCopyExtParams dataCopyExtParams = {
        static_cast<uint16_t>(rowSize), static_cast<uint32_t>(colSize * sizeof(OUT_TYPE)), srcStride,
        static_cast<uint32_t>((tilingData_->colNum - colSize) * sizeof(OUT_TYPE)), 0};
    DataCopyPad(yGm_[offset], yLocal, dataCopyExtParams);
    outQueue_.FreeTensor(yLocal);
}

template <typename IN_TYPE, typename OUT_TYPE, int64_t ROUND_MODE>
inline __aicore__ void DynamicBlockQuantSingleRow<IN_TYPE, OUT_TYPE, ROUND_MODE>::CopyOutScale(
    uint64_t offset, uint64_t rowSize, uint64_t colSize)
{
    LocalTensor<float> scaleLocal = scaleQueue_.DeQue<float>();
    uint32_t srcStride = colSize % 64 == 0 ? 0 : (8 - Ceil(colSize * sizeof(float), 32));
    DataCopyExtParams dataCopyExtParams = {
        static_cast<uint16_t>(rowSize), static_cast<uint32_t>(colSize * sizeof(float)), srcStride,
        static_cast<uint32_t>((tilingData_->colBlockLoopNum - colSize) * sizeof(float)), 0};
    DataCopyPad(scaleGm_[offset], scaleLocal, dataCopyExtParams);
    scaleQueue_.FreeTensor(scaleLocal);
}

template <typename IN_TYPE, typename OUT_TYPE, int64_t ROUND_MODE>
inline __aicore__ void DynamicBlockQuantSingleRow<IN_TYPE, OUT_TYPE, ROUND_MODE>::ComputeVF(
    __local_mem__ OUT_TYPE* outLocal, __local_mem__ float* scaleLocal, __local_mem__ IN_TYPE* xLocal, uint64_t colSize,
    uint64_t rowBlockSize, uint64_t colBlockSize)
{
    IN_TYPE zero = 0.0;
    uint32_t vfColSize = colSize;
    uint16_t vfRowBlockSize = rowBlockSize;
    uint16_t vfColBlockSize = colBlockSize;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<IN_TYPE> vReg0;
        AscendC::MicroAPI::RegTensor<IN_TYPE> vReg1;
        AscendC::MicroAPI::RegTensor<IN_TYPE> vReg2;
        AscendC::MicroAPI::RegTensor<IN_TYPE> vReg3;
        AscendC::MicroAPI::RegTensor<IN_TYPE> vReg4;
        AscendC::MicroAPI::RegTensor<float> vReg5;
        AscendC::MicroAPI::RegTensor<float> vReg6;
        AscendC::MicroAPI::RegTensor<float> vReg7;
        AscendC::MicroAPI::RegTensor<float> vReg8;
        AscendC::MicroAPI::RegTensor<float> vReg9;
        AscendC::MicroAPI::RegTensor<float> vReg10;
        AscendC::MicroAPI::RegTensor<float> vReg11;
        AscendC::MicroAPI::RegTensor<float> vReg12;
        AscendC::MicroAPI::RegTensor<float> vReg13;
        AscendC::MicroAPI::RegTensor<float> vReg14;
        AscendC::MicroAPI::RegTensor<float> vReg17;
        AscendC::MicroAPI::RegTensor<float> vReg18;
        AscendC::MicroAPI::RegTensor<OUT_TYPE> vReg15;
        AscendC::MicroAPI::RegTensor<OUT_TYPE> vReg16;
        AscendC::MicroAPI::MaskReg maskReg0;
        AscendC::MicroAPI::MaskReg maskReg1;
        AscendC::MicroAPI::MaskReg maskReg2;
        AscendC::MicroAPI::MaskReg maskReg3;
        AscendC::MicroAPI::MaskReg notZeroMaskReg;
        AscendC::MicroAPI::MaskReg defaultMaskReg = AscendC::MicroAPI::CreateMask<IN_TYPE>();
        AscendC::MicroAPI::MaskReg inputMaskReg = AscendC::MicroAPI::CreateMask<IN_TYPE>();

        AscendC::MicroAPI::RegTensor<float> vRegFPMax;
        // FP_MAX
        if constexpr (IsSameType<OUT_TYPE, fp8_e5m2_t>::value) {
            AscendC::MicroAPI::Duplicate(vRegFPMax, FP8_E5M2_MAX_VALUE);
        } else if constexpr (IsSameType<OUT_TYPE, fp8_e4m3fn_t>::value) {
            AscendC::MicroAPI::Duplicate(vRegFPMax, FP8_E4M3_MAX_VALUE);
        } else {
            AscendC::MicroAPI::Duplicate(vRegFPMax, HIFP8_MAX_VALUE);
        }

        uint32_t curSize = vfColSize;
        static constexpr AscendC::MicroAPI::CastTrait castTraitZero = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
        static constexpr AscendC::MicroAPI::CastTrait castTraitOne = {
            AscendC::MicroAPI::RegLayout::ONE, AscendC::MicroAPI::SatMode::NO_SAT,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
        static constexpr AscendC::MicroAPI::DivSpecificMode mode = {AscendC::MicroAPI::MaskMergeMode::ZEROING, false};

        for (uint16_t rowIdx = 0; rowIdx < vfRowBlockSize; rowIdx++) {
            curSize = vfColSize;
            for (uint16_t colIdx = 0; colIdx < vfColBlockSize; colIdx++) {
                inputMaskReg = AscendC::MicroAPI::UpdateMask<IN_TYPE>(curSize);
                AscendC::MicroAPI::DataCopy(vReg0, xLocal + (rowIdx * vfColBlockSize + colIdx) * 128);
                AscendC::MicroAPI::Muls(vReg1, vReg0, zero, inputMaskReg);
                AscendC::MicroAPI::Compare<IN_TYPE, CMPMODE::EQ>(maskReg0, vReg1, vReg1, inputMaskReg);
                AscendC::MicroAPI::MaskNot(maskReg1, maskReg0, inputMaskReg);

                // abs(x)
                AscendC::MicroAPI::Abs(vReg2, vReg0, maskReg0);

                // reduce_max(abs(x))
                AscendC::MicroAPI::ReduceMax(vReg3, vReg2, maskReg0);

                // brc(input_max)
                AscendC::MicroAPI::Duplicate(vReg4, vReg3, defaultMaskReg);

                // cast_to_f32
                AscendC::MicroAPI::Cast<float, IN_TYPE, castTraitZero>(vReg5, vReg4, defaultMaskReg);

                AscendC::MicroAPI::CompareScalar<float, CMPMODE::NE>(notZeroMaskReg, vReg5, (float)0.0, defaultMaskReg);
                // FP_MAX / input_max
                AscendC::MicroAPI::Div<float, &mode>(vReg7, vRegFPMax, vReg5, notZeroMaskReg);

                // max(FP_MAX / input_max, minScale)
                AscendC::MicroAPI::Maxs(vReg8, vReg7, tilingData_->minScale, defaultMaskReg);

                // copy out scale
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    scaleLocal + rowIdx * Ceil(vfColBlockSize, 64) * 64 + colIdx, vReg8, defaultMaskReg);

                AscendC::MicroAPI::Cast<float, IN_TYPE, castTraitZero>(vReg9, vReg0, defaultMaskReg);
                AscendC::MicroAPI::Cast<float, IN_TYPE, castTraitOne>(vReg10, vReg0, defaultMaskReg);

                AscendC::MicroAPI::Interleave(vReg11, vReg12, vReg9, vReg10);
                AscendC::MicroAPI::Mul(vReg13, vReg11, vReg8, defaultMaskReg);
                AscendC::MicroAPI::Mul(vReg14, vReg12, vReg8, defaultMaskReg);
                AscendC::MicroAPI::MaskUnPack<AscendC::MicroAPI::HighLowPart::LOWEST>(maskReg2, maskReg1);
                AscendC::MicroAPI::MaskUnPack<AscendC::MicroAPI::HighLowPart::HIGHEST>(maskReg3, maskReg1);
                AscendC::MicroAPI::Select(vReg17, vReg11, vReg13, maskReg2);
                AscendC::MicroAPI::Select(vReg18, vReg12, vReg14, maskReg3);

                if constexpr (IsSameType<OUT_TYPE, hifloat8_t>::value) {
                    AscendC::MicroAPI::Cast<OUT_TYPE, float, castTrait32toh8Zero>(vReg15, vReg17, defaultMaskReg);
                    AscendC::MicroAPI::Cast<OUT_TYPE, float, castTrait32toh8Zero>(vReg16, vReg18, defaultMaskReg);
                } else {
                    AscendC::MicroAPI::Cast<OUT_TYPE, float, castTrait32tofp8>(vReg15, vReg17, defaultMaskReg);
                    AscendC::MicroAPI::Cast<OUT_TYPE, float, castTrait32tofp8>(vReg16, vReg18, defaultMaskReg);
                }

                AscendC::MicroAPI::DataCopy<OUT_TYPE, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                    outLocal + (rowIdx * vfColBlockSize + colIdx) * 128, vReg15, defaultMaskReg);
                AscendC::MicroAPI::DataCopy<OUT_TYPE, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                    outLocal + (rowIdx * vfColBlockSize + colIdx) * 128 + 64, vReg16, defaultMaskReg);
            }
        }
    }
}

template <typename IN_TYPE, typename OUT_TYPE, int64_t ROUND_MODE>
inline __aicore__ void DynamicBlockQuantSingleRow<IN_TYPE, OUT_TYPE, ROUND_MODE>::ComputeVFBF16(
    __local_mem__ OUT_TYPE* outLocal, __local_mem__ float* scaleLocal, __local_mem__ IN_TYPE* xLocal, uint64_t colSize,
    uint64_t rowBlockSize, uint64_t colBlockSize)
{
    float zero = 0.0;
    uint32_t vfColSize = colSize;
    uint16_t vfRowBlockSize = rowBlockSize;
    uint16_t vfColBlockSize = colBlockSize;

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<IN_TYPE> vReg0;
        AscendC::MicroAPI::RegTensor<float> vReg1;
        AscendC::MicroAPI::RegTensor<float> vReg2;
        AscendC::MicroAPI::RegTensor<float> vReg3;
        AscendC::MicroAPI::RegTensor<float> vReg4;
        AscendC::MicroAPI::RegTensor<float> vReg5;
        AscendC::MicroAPI::RegTensor<float> vReg6;
        AscendC::MicroAPI::RegTensor<float> vReg7;
        AscendC::MicroAPI::RegTensor<float> vReg8;
        AscendC::MicroAPI::RegTensor<float> vReg9;
        AscendC::MicroAPI::RegTensor<float> vReg10;
        AscendC::MicroAPI::RegTensor<float> vReg11;
        AscendC::MicroAPI::RegTensor<float> vReg12;
        AscendC::MicroAPI::RegTensor<float> vReg13;
        AscendC::MicroAPI::RegTensor<float> vReg14;
        AscendC::MicroAPI::RegTensor<float> vReg15;
        AscendC::MicroAPI::RegTensor<float> vReg16;
        AscendC::MicroAPI::RegTensor<float> vReg17;
        AscendC::MicroAPI::RegTensor<float> vReg18;
        AscendC::MicroAPI::RegTensor<float> vReg19;
        AscendC::MicroAPI::RegTensor<OUT_TYPE> vReg20;
        AscendC::MicroAPI::RegTensor<OUT_TYPE> vReg21;
        AscendC::MicroAPI::MaskReg maskReg0;
        AscendC::MicroAPI::MaskReg maskReg1;
        AscendC::MicroAPI::MaskReg maskReg2;
        AscendC::MicroAPI::MaskReg maskReg3;
        AscendC::MicroAPI::MaskReg notZeroMaskReg;
        AscendC::MicroAPI::MaskReg f32MaskReg =
            AscendC::MicroAPI::CreateMask<float, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg inputMaskReg = AscendC::MicroAPI::CreateMask<IN_TYPE>();

        AscendC::MicroAPI::RegTensor<float> vRegFPMax;
        // FP_MAX
        if constexpr (IsSameType<OUT_TYPE, fp8_e5m2_t>::value) {
            AscendC::MicroAPI::Duplicate(vRegFPMax, FP8_E5M2_MAX_VALUE);
        } else if constexpr (IsSameType<OUT_TYPE, fp8_e4m3fn_t>::value) {
            AscendC::MicroAPI::Duplicate(vRegFPMax, FP8_E4M3_MAX_VALUE);
        } else {
            AscendC::MicroAPI::Duplicate(vRegFPMax, HIFP8_MAX_VALUE);
        }

        uint32_t curSize = vfColSize;
        static constexpr AscendC::MicroAPI::CastTrait castTraitZero = {
            AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::NO_SAT,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
        static constexpr AscendC::MicroAPI::CastTrait castTraitOne = {
            AscendC::MicroAPI::RegLayout::ONE, AscendC::MicroAPI::SatMode::NO_SAT,
            AscendC::MicroAPI::MaskMergeMode::ZEROING, AscendC::RoundMode::CAST_RINT};
        static constexpr AscendC::MicroAPI::DivSpecificMode mode = {AscendC::MicroAPI::MaskMergeMode::ZEROING, false};

        for (uint16_t rowIdx = 0; rowIdx < vfRowBlockSize; rowIdx++) {
            curSize = vfColSize;
            for (uint16_t colIdx = 0; colIdx < vfColBlockSize; colIdx++) {
                inputMaskReg = AscendC::MicroAPI::UpdateMask<IN_TYPE>(curSize);
                AscendC::MicroAPI::DataCopy(vReg0, xLocal + (rowIdx * vfColBlockSize + colIdx) * 128);
                AscendC::MicroAPI::Cast<float, IN_TYPE, castTraitZero>(vReg1, vReg0, inputMaskReg);
                AscendC::MicroAPI::Cast<float, IN_TYPE, castTraitOne>(vReg2, vReg0, inputMaskReg);
                AscendC::MicroAPI::Interleave(vReg3, vReg4, vReg1, vReg2);
                AscendC::MicroAPI::Muls(vReg5, vReg3, zero, f32MaskReg);
                AscendC::MicroAPI::Muls(vReg6, vReg4, zero, f32MaskReg);
                AscendC::MicroAPI::Compare<float, CMPMODE::NE>(maskReg0, vReg5, vReg5, f32MaskReg);
                AscendC::MicroAPI::Compare<float, CMPMODE::NE>(maskReg1, vReg6, vReg6, f32MaskReg);
                AscendC::MicroAPI::MaskNot(maskReg2, maskReg0, f32MaskReg);
                AscendC::MicroAPI::MaskNot(maskReg3, maskReg1, f32MaskReg);

                // abs(x)
                AscendC::MicroAPI::Abs(vReg7, vReg3, maskReg2);
                AscendC::MicroAPI::Abs(vReg8, vReg4, maskReg3);
                // reduce_max(abs(x))
                AscendC::MicroAPI::ReduceMax(vReg9, vReg7, f32MaskReg);
                AscendC::MicroAPI::ReduceMax(vReg10, vReg8, f32MaskReg);

                // brc(input_max)
                AscendC::MicroAPI::Duplicate(vReg11, vReg9, f32MaskReg);
                AscendC::MicroAPI::Duplicate(vReg12, vReg10, f32MaskReg);
                // input_max
                AscendC::MicroAPI::Max(vReg13, vReg11, vReg12, f32MaskReg);

                AscendC::MicroAPI::CompareScalar<float, CMPMODE::NE>(notZeroMaskReg, vReg13, (float)0.0, f32MaskReg);
                // FP_MAX / input_max
                AscendC::MicroAPI::Div<float, &mode>(vReg14, vRegFPMax, vReg13, notZeroMaskReg);

                // max(FP_MAX / input_max, minScale)
                AscendC::MicroAPI::Maxs(vReg15, vReg14, tilingData_->minScale, f32MaskReg);

                // copy out scale
                AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                    scaleLocal + rowIdx * Ceil(vfColBlockSize, 64) * 64 + colIdx, vReg15, f32MaskReg);

                AscendC::MicroAPI::Mul(vReg16, vReg3, vReg15, f32MaskReg);
                AscendC::MicroAPI::Mul(vReg17, vReg4, vReg15, f32MaskReg);

                AscendC::MicroAPI::Select(vReg18, vReg3, vReg16, maskReg0);
                AscendC::MicroAPI::Select(vReg19, vReg4, vReg17, maskReg1);

                if constexpr (IsSameType<OUT_TYPE, hifloat8_t>::value) {
                    AscendC::MicroAPI::Cast<OUT_TYPE, float, castTrait32toh8Zero>(vReg20, vReg18, f32MaskReg);
                    AscendC::MicroAPI::Cast<OUT_TYPE, float, castTrait32toh8Zero>(vReg21, vReg19, f32MaskReg);
                } else {
                    AscendC::MicroAPI::Cast<OUT_TYPE, float, castTrait32tofp8>(vReg20, vReg18, f32MaskReg);
                    AscendC::MicroAPI::Cast<OUT_TYPE, float, castTrait32tofp8>(vReg21, vReg19, f32MaskReg);
                }

                AscendC::MicroAPI::DataCopy<OUT_TYPE, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                    outLocal + (rowIdx * vfColBlockSize + colIdx) * 128, vReg20, f32MaskReg);
                AscendC::MicroAPI::DataCopy<OUT_TYPE, AscendC::MicroAPI::StoreDist::DIST_PACK4_B32>(
                    outLocal + (rowIdx * vfColBlockSize + colIdx) * 128 + 64, vReg21, f32MaskReg);
            }
        }
    }
}

} // namespace DynamicBlockQuant
#endif // DYNAMIC_BLOCK_QUNANT_SINGLE_ROW_KERNEL_H