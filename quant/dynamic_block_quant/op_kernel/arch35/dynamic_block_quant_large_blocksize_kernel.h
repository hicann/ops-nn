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
 * \file dynamic_block_quant_large_blocksize_kernel.h
 * \brief
 */

#ifndef DYNAMIC_BLOCK_QUANT_LARGE_BLOCKSIZE_H
#define DYNAMIC_BLOCK_QUANT_LARGE_BLOCKSIZE_H
#include "kernel_operator.h"
#include "../inc/platform.h"

#include "dynamic_block_quant_common.h"

#define FLOAT_OVERFLOW_MODE_CTRL 60
namespace DynamicBlockQuant {
using namespace AscendC;

template <typename T, typename U, int64_t RMode>
class DynamicBlockQuantLargeBlockSize {
public:
    __aicore__ inline DynamicBlockQuantLargeBlockSize(TPipe* pipe)
    {
        Ppipe = pipe;
    }
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR scale, const DynamicBlockQuantTilingData& tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(
        int64_t rowNum, int64_t blockColNum, int64_t baseXGmOffset);
    __aicore__ inline void CopyOutY(
        int64_t rowNum, int64_t colNum, int64_t baseXGmOffset);
    __aicore__ inline void CopyOutScale(
        int64_t baseScaleOffset);
    __aicore__ inline void ParseTilingData(const DynamicBlockQuantTilingData& tilingData);
    __aicore__ inline void ComputeXTmpMax(
        int64_t rowNum, int64_t blockColNum, __local_mem__ T* xLocalAddr, __local_mem__ T* xLocalMaxTmp);
    __aicore__ inline void ComputeScaleVF(
        __local_mem__ float* scaleLocalTmp, __local_mem__ T* xLocalMaxTmp);
    __aicore__ inline void ComputeOutVF(
        int64_t rowNum, int64_t colNum, __local_mem__ T* xLocalAddr, __local_mem__ float* scaleLocal, __local_mem__ U* outLocal);
    __aicore__ inline void InitBuffer();
    __aicore__ inline void InitGmOffset(GM_ADDR x, GM_ADDR y, GM_ADDR scale);
    __aicore__ inline void ProcessBlock(
        int64_t blockRowNum, int64_t blockColNum, int64_t baseXGmOffset, int64_t baseScaleOffset);

private:
    TPipe* Ppipe = nullptr;
    TQue<QuePosition::VECIN, DB_BUFFER> inQueue_;
    GlobalTensor<T> xGm_;
    TQue<QuePosition::VECOUT, DB_BUFFER> outQueue_;
    GlobalTensor<U> yGm_;
    TQue<QuePosition::VECOUT, DB_BUFFER> scaleQueue_;
    GlobalTensor<float> scaleGm_;

    TBuf<QuePosition::VECCALC> xLocalMaxBuffer_;

    int64_t blockIdx_ = 0;
    float fp8MaxValue_ = 0;
    uint16_t infValue_;
    uint16_t maxValue_ = 0x7fff;

    int64_t totalCoreNum_ = 0;
    int64_t ubSize_ = 0;
    int64_t vfLen_ = 0;
    int64_t roundMode_ = 0;
    int64_t dstType_ = 0;
    int64_t blockSizeRow_ = 0;
    int64_t blockSizeCol_ = 0;
    int64_t singleBatchRowBlockLoopNum_ = 0;
    int64_t rowBlockLoopNum_ = 0;
    int64_t colBlockLoopNum_ = 0;
    int64_t rowUbBlockLoopNum_ = 0;
    int64_t colUbBlockLoopNum_ = 0;
    int64_t usedCoreNum_ = 0;
    int64_t batchNum_ = 0;
    int64_t rowNum_ = 0;
    int64_t colNum_ = 0;
    int64_t rowTileNum_ = 0;
    int64_t colTileNum_ = 0;
    int64_t rowUbFactor_ = 0;
    int64_t colUbFactor_ = 0;
    int64_t normalCoreRowTileNum_ = 0;
    int64_t normalCoreColTileNum_ = 0;
    int64_t tailCoreRowTileNum_ = 0;
    int64_t tailCoreColTileNum_ = 0;
    int64_t rowNormalCoreNum_ = 0;
    int64_t colNormalCoreNum_ = 0;
    int64_t rowTailCoreNum_ = 0;
    int64_t colTailCoreNum_ = 0;
    float minScale_ = 0.0;

    int64_t largeShapeUbRowLoopNum_ = 0;
    int64_t largeShapeNormalUbRowNum_ = 0;
    int64_t largeShapeTailUbRowNum_ = 0;
    int64_t singleUbBufferHandleNum_ = 0;

    static constexpr AscendC::MicroAPI::CastTrait castTrait32toh8 = []() {
        if constexpr (RMode == 1 || RMode == 4) {
            return AscendC::MicroAPI::CastTrait {
                AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT,
                AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};
        } else if constexpr (RMode == 7) {
            return AscendC::MicroAPI::CastTrait {
                AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT,
                AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_HYBRID};
        }
    }();

    int64_t rowCoreIdx_ = 0;
    int64_t colCoreIdx_ = 0;
    int64_t rowUbLoop_ = 0;
    int64_t colUbLoop_ = 0;

    int64_t preRowBlockNum_ = 0;
    int64_t rowCoreTileNum_ = 0;
    int64_t colCoreTileNum_ = 0;
    bool isRowTailCore_ = false;
    bool isColTailCore_ = false;
    int64_t coreRowNum_ = 0;
    int64_t coreColNum_ = 0;

    // 当前所在batch
    int64_t preBatch_ = 0;
    int64_t preSingleBatchBlock_ = 0;
    // 行尾块行数
    int64_t tailBlockSizeRow_ = 0;
};

template <typename T, typename U, int64_t RMode>
__aicore__ inline void DynamicBlockQuantLargeBlockSize<T, U, RMode>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR scale, const DynamicBlockQuantTilingData& tilingData)
{
    ParseTilingData(tilingData);

    blockIdx_ = GetBlockIdx();
    if (blockIdx_ >= usedCoreNum_) {
        return;
    }
    #if (__NPU_ARCH__ == 3510)
        AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
    #endif
    
    rowCoreIdx_ = blockIdx_ / colTileNum_;
    colCoreIdx_ = blockIdx_ % colTileNum_;
    isRowTailCore_ = (rowCoreIdx_ >= rowNormalCoreNum_);
    isColTailCore_ = (colCoreIdx_ >= colNormalCoreNum_);
    InitGmOffset(x, y, scale);
    InitBuffer();
}

template <typename T, typename U, int64_t RMode>
__aicore__ inline void DynamicBlockQuantLargeBlockSize<T, U, RMode>::InitGmOffset(GM_ADDR x, GM_ADDR y, GM_ADDR scale) {
    int64_t xRowGmOffset = 0;
    int64_t xColGmOffset = 0;
    int64_t xGmOffset = 0;
    int64_t scaleGmRowOffset = 0;
    int64_t scaleGmColOffset = 0;
    int64_t scaleGmOffset = 0;

    tailBlockSizeRow_ = rowNum_ % blockSizeRow_;
    if (tailBlockSizeRow_ == 0) {
        tailBlockSizeRow_ = blockSizeRow_;
    }
    if (isRowTailCore_) {
        rowCoreTileNum_ = tailCoreRowTileNum_;
        rowUbBlockLoopNum_ = rowUbBlockLoopNum_ > rowCoreTileNum_ ? rowCoreTileNum_ : rowUbBlockLoopNum_;
        preRowBlockNum_ = rowNormalCoreNum_ * normalCoreRowTileNum_ + (rowCoreIdx_ - rowNormalCoreNum_) * tailCoreRowTileNum_;
        rowUbLoop_ = tailCoreRowTileNum_ / rowUbBlockLoopNum_;
    } else {
        rowCoreTileNum_ = normalCoreRowTileNum_;
        preRowBlockNum_ = rowCoreIdx_ * normalCoreRowTileNum_;
        rowUbLoop_ = rowCoreTileNum_ / rowUbBlockLoopNum_;
    }
    preBatch_ = preRowBlockNum_ / singleBatchRowBlockLoopNum_;
    preSingleBatchBlock_ = preRowBlockNum_ % singleBatchRowBlockLoopNum_;
    xRowGmOffset = preBatch_ * rowNum_ * colNum_ + preSingleBatchBlock_ * blockSizeRow_ * colNum_;
    scaleGmRowOffset = preRowBlockNum_;

    int64_t endRowBlockNum_ = preRowBlockNum_ + rowCoreTileNum_;
    int64_t endBath = endRowBlockNum_ / singleBatchRowBlockLoopNum_;
    int64_t endSingleBatchBlock = endRowBlockNum_ % singleBatchRowBlockLoopNum_;
    coreRowNum_ = (endBath - preBatch_) * rowNum_ + (endSingleBatchBlock - preSingleBatchBlock_) * blockSizeRow_;

    if (isColTailCore_) {
        colCoreTileNum_ = tailCoreColTileNum_;
        colUbBlockLoopNum_ = colUbBlockLoopNum_ > colCoreTileNum_ ? colCoreTileNum_ : colUbBlockLoopNum_;
        xColGmOffset = colNormalCoreNum_ * normalCoreColTileNum_ * blockSizeCol_ +
                       (colCoreIdx_ - colNormalCoreNum_) * tailCoreColTileNum_ * blockSizeCol_;
        scaleGmColOffset =
            colNormalCoreNum_ * normalCoreColTileNum_ + (colCoreIdx_ - colNormalCoreNum_) * tailCoreColTileNum_;
        colUbLoop_ = colCoreTileNum_ / colUbBlockLoopNum_;
        coreColNum_ = colCoreIdx_ + 1 == colTileNum_ ?
                          colNum_ - (colNormalCoreNum_ * normalCoreColTileNum_ +
                                     (colCoreIdx_ - colNormalCoreNum_) * tailCoreColTileNum_) * blockSizeCol_ :  tailCoreColTileNum_ * blockSizeCol_;
    } else {
        colCoreTileNum_ = normalCoreColTileNum_;
        xColGmOffset = colCoreIdx_ * normalCoreColTileNum_ * blockSizeCol_;
        scaleGmColOffset = colCoreIdx_ * normalCoreColTileNum_;
        colUbLoop_ = colCoreTileNum_ / colUbBlockLoopNum_;
        coreColNum_ = colCoreIdx_ + 1 == colTileNum_ ? colNum_ - colCoreIdx_ * normalCoreColTileNum_ * blockSizeCol_ : normalCoreColTileNum_ * blockSizeCol_;
    }
    xGmOffset = xRowGmOffset + xColGmOffset;
    scaleGmOffset = scaleGmRowOffset * colBlockLoopNum_ + scaleGmColOffset;

    xGm_.SetGlobalBuffer((__gm__ T*)x + xGmOffset);
    yGm_.SetGlobalBuffer((__gm__ U*)y + xGmOffset);
    scaleGm_.SetGlobalBuffer((__gm__ float*)scale + scaleGmOffset);
}

template <typename T, typename U, int64_t RMode>
__aicore__ inline void DynamicBlockQuantLargeBlockSize<T, U, RMode>::InitBuffer() {
    int64_t perBlockSize = 0;
    if constexpr (IsSameType<U, fp8_e5m2_t>::value) {
        fp8MaxValue_ = FP8_E5M2_MAX_VALUE;
    } else if constexpr (IsSameType<U, fp8_e4m3fn_t>::value) {
        fp8MaxValue_ = FP8_E4M3_MAX_VALUE;
    } else if constexpr (IsSameType<U, hifloat8_t>::value) {
        fp8MaxValue_ = HIFP8_MAX_VALUE;
    }

    if constexpr (IsSameType<T, half>::value) {
        infValue_ = FP16_INF_VALUE;    
    } else {
        infValue_ = BF16_INF_VALUE;
    }
    // 单个block最大的col列数
    int64_t normalBlockCol = coreColNum_ < blockSizeCol_ ? coreColNum_ : blockSizeCol_;
    int64_t normalBlockRow = coreRowNum_ < blockSizeRow_ ? coreRowNum_ : blockSizeRow_;

    // single block need ubsize
    perBlockSize = DB_BUFFER * normalBlockRow * (normalBlockCol * (sizeof(U) + sizeof(T))) + DB_BUFFER * 32;
    
    largeShapeUbRowLoopNum_ = Ceil(perBlockSize, (ubSize_ - AscendC::VECTOR_REG_WIDTH));
    largeShapeNormalUbRowNum_ = Ceil(normalBlockRow, largeShapeUbRowLoopNum_);

    // need buffer
    int64_t inQueueBuffer = largeShapeNormalUbRowNum_ * normalBlockCol * sizeof(T);

    int64_t outQueueBuffer = largeShapeNormalUbRowNum_ * normalBlockCol * sizeof(U);

    singleUbBufferHandleNum_ = largeShapeNormalUbRowNum_ * normalBlockCol;

    Ppipe->InitBuffer(inQueue_, DB_BUFFER, inQueueBuffer);
    Ppipe->InitBuffer(outQueue_, DB_BUFFER, outQueueBuffer);
    Ppipe->InitBuffer(scaleQueue_, DB_BUFFER, 32);
    Ppipe->InitBuffer(xLocalMaxBuffer_, AscendC::VECTOR_REG_WIDTH);
}

template <typename T, typename U, int64_t RMode>
__aicore__ inline void DynamicBlockQuantLargeBlockSize<T, U, RMode>::ParseTilingData(const DynamicBlockQuantTilingData& tilingData)
{
    totalCoreNum_ = tilingData.totalCoreNum;
    ubSize_ = tilingData.ubSize;
    vfLen_ = tilingData.vfLen;
    blockSizeRow_ = tilingData.blockSizeRow;
    blockSizeCol_ = tilingData.blockSizeCol;
    minScale_ = tilingData.minScale;
    roundMode_ = tilingData.roundMode;
    dstType_ = tilingData.dstType;
    batchNum_ = tilingData.batchNum;
    rowNum_ = tilingData.rowNum;
    colNum_ = tilingData.colNum;
    singleBatchRowBlockLoopNum_ = tilingData.singleBatchRowBlockLoopNum;
    rowBlockLoopNum_ = tilingData.rowBlockLoopNum;
    colBlockLoopNum_ = tilingData.colBlockLoopNum;
    rowUbBlockLoopNum_ = tilingData.rowUbBlockLoopNum;
    colUbBlockLoopNum_ = tilingData.colUbBlockLoopNum;
    normalCoreRowTileNum_ = tilingData.normalCoreRowTileNum;
    normalCoreColTileNum_ = tilingData.normalCoreColTileNum;
    tailCoreRowTileNum_ = tilingData.tailCoreRowTileNum;
    tailCoreColTileNum_ = tilingData.tailCoreColTileNum;
    rowUbFactor_ = tilingData.rowUbFactor;
    colUbFactor_ = tilingData.colUbFactor;
    usedCoreNum_ = tilingData.usedCoreNum;
    rowTileNum_ = tilingData.rowTileNum;
    colTileNum_ = tilingData.colTileNum;
    rowNormalCoreNum_ = tilingData.rowNormalCoreNum;
    colNormalCoreNum_ = tilingData.colNormalCoreNum;
    rowTailCoreNum_ = tilingData.rowTailCoreNum;
    colTailCoreNum_ = tilingData.colTailCoreNum;
}

template <typename T, typename U, int64_t RMode>
__aicore__ inline void DynamicBlockQuantLargeBlockSize<T, U, RMode>::Process()
{
    if (blockIdx_ >= usedCoreNum_) {
        return;
    }
    // 当前ub需要处理的行块数
    int64_t blockRow = 1;
    // 当前ub需要处理的列块数
    int64_t blockCol = 1;
    for (int64_t rowUbLoopIdx = 0; rowUbLoopIdx < rowUbLoop_; rowUbLoopIdx++) {
        for (int64_t colUbLoopIdx = 0; colUbLoopIdx < colUbLoop_; colUbLoopIdx++) {
            int64_t blockRowNum = blockSizeRow_;
            int64_t blockColNum = colUbLoopIdx == colUbLoop_ - 1 ? coreColNum_ - colUbLoopIdx *  blockSizeCol_ : blockSizeCol_;
            int64_t nowRowBlockNum = preRowBlockNum_ + rowUbLoopIdx;
            if (nowRowBlockNum % singleBatchRowBlockLoopNum_ == singleBatchRowBlockLoopNum_ - 1) {
                blockRowNum = tailBlockSizeRow_;
            }
            int64_t nowBatch = nowRowBlockNum / singleBatchRowBlockLoopNum_;
            int64_t nowSingleBatchBlock = nowRowBlockNum % singleBatchRowBlockLoopNum_;
            int64_t baseXGmOffset = ((nowBatch - preBatch_) * rowNum_ + (nowSingleBatchBlock - preSingleBatchBlock_) * blockSizeRow_) * colNum_ + colUbLoopIdx * blockSizeCol_;
            
            int64_t baseScaleOffset = rowUbLoopIdx * colBlockLoopNum_ + colUbLoopIdx;
            // 处理单block
            ProcessBlock(blockRowNum, blockColNum, baseXGmOffset, baseScaleOffset);
        }
    }
}

template <typename T, typename U, int64_t RMode>
__aicore__ inline void DynamicBlockQuantLargeBlockSize<T, U, RMode>::ProcessBlock(
    int64_t blockRowNum, int64_t blockColNum, int64_t baseXGmOffset, int64_t baseScaleOffset) 
{
    // 当前block单次ub处理最大行数
    int64_t singleLoopHandleRow = singleUbBufferHandleNum_ / blockColNum;
    // 当前block需要ub循环次数
    int64_t singleBlockLoopUbNum = Ceil(blockRowNum, singleLoopHandleRow);
    int64_t singleLoopHandleRowNum;
    LocalTensor<T> xLocalMaxTmp = xLocalMaxBuffer_.Get<T>();
    AscendC::Duplicate(xLocalMaxTmp, static_cast<T>(0), xLocalMaxTmp.GetSize());
    __local_mem__ T* xLocalMaxTmpAddr = (__local_mem__ T*)xLocalMaxTmp.GetPhyAddr();
    LocalTensor<T> inLocal;
    __local_mem__ T* xLocalAddr;
    int64_t nowXGmOffset;

    // 分段搬入，计算max(input)
    for(int64_t rowLoopIdx = singleBlockLoopUbNum - 1; rowLoopIdx >= 0; rowLoopIdx--) {
        singleLoopHandleRowNum = rowLoopIdx == singleBlockLoopUbNum - 1 ? blockRowNum - singleLoopHandleRow * rowLoopIdx : singleLoopHandleRow;
        nowXGmOffset = baseXGmOffset + rowLoopIdx * singleLoopHandleRow * colNum_;
        CopyIn(singleLoopHandleRowNum, blockColNum, nowXGmOffset);
        inLocal = inQueue_.DeQue<T>();
        xLocalAddr = (__local_mem__ T*)inLocal.GetPhyAddr();
        ComputeXTmpMax(singleLoopHandleRowNum, blockColNum, xLocalAddr, xLocalMaxTmpAddr);
        if (rowLoopIdx != 0) {
            inQueue_.FreeTensor(inLocal);
        }
    }
    AscendC::ReduceMax<uint16_t>((AscendC::LocalTensor<uint16_t>&)xLocalMaxTmp, 
    (AscendC::LocalTensor<uint16_t>&)xLocalMaxTmp, (AscendC::LocalTensor<uint16_t>&)xLocalMaxTmp, xLocalMaxTmp.GetSize());

    LocalTensor<float> scaleLocal = scaleQueue_.AllocTensor<float>();
    __local_mem__ float* scaleLocalAddr = (__local_mem__ float*)scaleLocal.GetPhyAddr();
    ComputeScaleVF(scaleLocalAddr, xLocalMaxTmpAddr);

    LocalTensor<U> outLocal = outQueue_.AllocTensor<U>();
    __local_mem__ U* outLocalAddr = (__local_mem__ U*)outLocal.GetPhyAddr();
    
    // 分段计算Y
    ComputeOutVF(singleLoopHandleRowNum, blockColNum, xLocalAddr, scaleLocalAddr, outLocalAddr);
    inQueue_.FreeTensor(inLocal);
    outQueue_.EnQue(outLocal);
    CopyOutY(singleLoopHandleRowNum, blockColNum, nowXGmOffset);

    for(int64_t rowLoopIdx = 1; rowLoopIdx < singleBlockLoopUbNum; rowLoopIdx++) {
        singleLoopHandleRowNum = rowLoopIdx == singleBlockLoopUbNum - 1 ? blockRowNum - singleLoopHandleRow * rowLoopIdx : singleLoopHandleRow;
        nowXGmOffset = baseXGmOffset + rowLoopIdx * singleLoopHandleRow * colNum_;
        CopyIn(singleLoopHandleRowNum, blockColNum, nowXGmOffset);

        inLocal = inQueue_.DeQue<T>();
        xLocalAddr = (__local_mem__ T*)inLocal.GetPhyAddr();

        outLocal = outQueue_.AllocTensor<U>();
        outLocalAddr = (__local_mem__ U*)outLocal.GetPhyAddr();
        ComputeOutVF(singleLoopHandleRowNum, blockColNum, xLocalAddr, scaleLocalAddr, outLocalAddr);
        
        inQueue_.FreeTensor(inLocal); 
        outQueue_.EnQue(outLocal);
        CopyOutY(singleLoopHandleRowNum, blockColNum, nowXGmOffset);
    }
    scaleQueue_.EnQue(scaleLocal);
    CopyOutScale(baseScaleOffset);
}

template <typename T, typename U, int64_t RMode>
__aicore__ inline void DynamicBlockQuantLargeBlockSize<T, U, RMode>::CopyIn(
    int64_t rowNum, int64_t blockColNum, int64_t baseXGmOffset)
{
    LocalTensor<T> inLocal = inQueue_.AllocTensor<T>();
    DataCopyExtParams copyParams = {
        static_cast<uint16_t>(rowNum), static_cast<uint32_t>(blockColNum * sizeof(T)),
        static_cast<uint32_t>((colNum_ - blockColNum) * sizeof(T)), 0, 0};

    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
    DataCopyPad<T, PaddingMode::Compact>(inLocal, xGm_[baseXGmOffset], copyParams, padParams);
    inQueue_.EnQue(inLocal);
}

template <typename T, typename U, int64_t RMode>
__aicore__ inline void DynamicBlockQuantLargeBlockSize<T, U, RMode>::ComputeXTmpMax(
    int64_t rowNum, int64_t blockColNum, __local_mem__ T* xLocalAddr, __local_mem__ T* xLocalMaxTmp) {

    uint32_t xTotalNum = rowNum * blockColNum;
    uint32_t dtypeSize = sizeof(T);
    uint16_t VL = AscendC::VECTOR_REG_WIDTH / dtypeSize;
    uint16_t vfLoop = Ceil(xTotalNum, VL);
    __VEC_SCOPE__ 
    {
        AscendC::MicroAPI::RegTensor<T> vreg1;
        AscendC::MicroAPI::RegTensor<T> vreg2;
        AscendC::MicroAPI::RegTensor<T> vreg3;
        AscendC::MicroAPI::RegTensor<T> vLocalTmpMaxReg;

        AscendC::MicroAPI::Duplicate((AscendC::MicroAPI::RegTensor<uint16_t>&)vreg2, maxValue_);
        AscendC::MicroAPI::MaskReg preg0;
        AscendC::MicroAPI::MaskReg preg1;
        AscendC::MicroAPI::MaskReg maskAll = AscendC::MicroAPI::CreateMask<uint16_t, AscendC::MicroAPI::MaskPattern::ALL>();

        AscendC::MicroAPI::DataCopy(vLocalTmpMaxReg, xLocalMaxTmp);
        
        for(uint16_t i = 0; i < vfLoop; i++) {
            preg0 = AscendC::MicroAPI::UpdateMask<T>(xTotalNum);
            AscendC::MicroAPI::DataCopy(vreg1, xLocalAddr + i * VL);
            AscendC::MicroAPI::And((AscendC::MicroAPI::RegTensor<uint16_t>&)vreg3, (AscendC::MicroAPI::RegTensor<uint16_t>&)vreg1,
            (AscendC::MicroAPI::RegTensor<uint16_t>&)vreg2, preg0);
            AscendC::MicroAPI::CompareScalar<uint16_t, CMPMODE::LT>(preg1, (AscendC::MicroAPI::RegTensor<uint16_t>&)vreg3, infValue_, preg0);
            AscendC::MicroAPI::Max<T, AscendC::MicroAPI::MaskMergeMode::MERGING>(vLocalTmpMaxReg, vLocalTmpMaxReg, vreg3, preg1);
        }
        AscendC::MicroAPI::DataCopy(xLocalMaxTmp, vLocalTmpMaxReg, maskAll);
    }
}

template <typename T, typename U, int64_t RMode>
__aicore__ inline void DynamicBlockQuantLargeBlockSize<T, U, RMode>::ComputeScaleVF(
    __local_mem__ float* scaleLocalTmp, __local_mem__ T* xLocalMaxTmp)
{
    static constexpr AscendC::MicroAPI::DivSpecificMode mode = {AscendC::MicroAPI::MaskMergeMode::ZEROING, false};
    uint32_t scaleNum = 1;
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<T> vreg1;
        AscendC::MicroAPI::RegTensor<float> vreg2;
        AscendC::MicroAPI::RegTensor<float> vreg3;
        AscendC::MicroAPI::RegTensor<float> reciprocalScale;
        AscendC::MicroAPI::RegTensor<float> minScaleReg;
        AscendC::MicroAPI::RegTensor<float> vreg5;
        AscendC::MicroAPI::RegTensor<float> vreg6;

        AscendC::MicroAPI::MaskReg preg0;

        preg0 = AscendC::MicroAPI::UpdateMask<T>(scaleNum);

        AscendC::MicroAPI::Duplicate(vreg3, fp8MaxValue_);
        AscendC::MicroAPI::Duplicate(reciprocalScale, 1.0f);
        AscendC::MicroAPI::Duplicate(minScaleReg, minScale_);
        AscendC::MicroAPI::Div<float, &mode>(reciprocalScale, reciprocalScale, minScaleReg, preg0);

        AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vreg1, xLocalMaxTmp);
        AscendC::MicroAPI::Cast<float, T, castTrait0>(vreg2, vreg1, preg0);

        AscendC::MicroAPI::Div<float, &mode>(vreg5, vreg2, vreg3, preg0);
        AscendC::MicroAPI::Min(vreg6, vreg5, reciprocalScale, preg0);

        AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(scaleLocalTmp, vreg6, preg0);
    }
}

template <typename T, typename U, int64_t RMode>
__aicore__ inline void DynamicBlockQuantLargeBlockSize<T, U, RMode>::ComputeOutVF(
    int64_t rowNum, int64_t colNum, __local_mem__ T* xLocalAddr, __local_mem__ float* scaleLocal, __local_mem__ U* outLocal)
{
    uint32_t xTotalNum = rowNum * colNum;
    uint32_t dtypeSize = sizeof(float);
    uint16_t VL = AscendC::VECTOR_REG_WIDTH / dtypeSize;
    uint16_t vfLoop = (xTotalNum + VL - 1) / VL;

    static constexpr AscendC::MicroAPI::DivSpecificMode mode = {AscendC::MicroAPI::MaskMergeMode::ZEROING, false};
    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<T> vreg1;
        AscendC::MicroAPI::RegTensor<float> vreg2;
        AscendC::MicroAPI::RegTensor<float> vreg3;
        AscendC::MicroAPI::RegTensor<float> vreg4;
        AscendC::MicroAPI::RegTensor<float> vreg5;
        AscendC::MicroAPI::RegTensor<float> vreg7;
        AscendC::MicroAPI::RegTensor<U> outReg;

        AscendC::MicroAPI::MaskReg preg0;
        AscendC::MicroAPI::MaskReg preg1;

        preg0 = AscendC::MicroAPI::CreateMask<float, MicroAPI::MaskPattern::ALL>();

        AscendC::MicroAPI::DataCopy<float, AscendC::MicroAPI::LoadDist::DIST_BRC_B32>(
            vreg2, scaleLocal);

        for (uint16_t i = 0; i < static_cast<uint16_t>(vfLoop); i++) {
            preg0 = AscendC::MicroAPI::UpdateMask<float>(xTotalNum);
            AscendC::MicroAPI::DataCopy<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(vreg1, xLocalAddr + i * VL);
            AscendC::MicroAPI::Cast<float, T, castTrait0>(vreg4, vreg1, preg0);
            AscendC::MicroAPI::Div<float, &mode>(vreg5, vreg4, vreg2, preg0);

            AscendC::MicroAPI::Muls(vreg3, vreg4, 0.0f, preg0);
            AscendC::MicroAPI::Compare<float, CMPMODE::NE>(preg1, vreg3, vreg3, preg0);
            AscendC::MicroAPI::Select(vreg7, vreg4, vreg5, preg1);

            if constexpr (IsSameType<U, hifloat8_t>::value) {
                AscendC::MicroAPI::Cast<U, float, castTrait32toh8>(outReg, vreg7, preg0);
            } else {
                AscendC::MicroAPI::Cast<U, float, castTrait32tofp8>(outReg, vreg7, preg0);
            }
            MicroAPI::DataCopy<U, MicroAPI::StoreDist::DIST_PACK4_B32>(
                outLocal + i * VL, outReg, preg0);
        }
    }
}
template <typename T, typename U, int64_t RMode>
__aicore__ inline void DynamicBlockQuantLargeBlockSize<T, U, RMode>::CopyOutScale(int64_t baseScaleOffset) {
    LocalTensor<float> scaleLocal = scaleQueue_.DeQue<float>();

    DataCopyExtParams scaleCopyParams = {1, static_cast<uint32_t>(1 * sizeof(float)), 0, 0, 0};
    DataCopyPad(scaleGm_[baseScaleOffset], scaleLocal, scaleCopyParams);
    scaleQueue_.FreeTensor(scaleLocal);
}

template <typename T, typename U, int64_t RMode>
__aicore__ inline void DynamicBlockQuantLargeBlockSize<T, U, RMode>::CopyOutY(int64_t rowNum, int64_t colNum, int64_t baseXGmOffset)
{
    LocalTensor<U> outLocal = outQueue_.DeQue<U>();
    DataCopyExtParams outCopyParams = {
        static_cast<uint16_t>(rowNum), static_cast<uint32_t>(colNum * sizeof(U)), 
        0, static_cast<uint32_t>((colNum_ - colNum) * sizeof(U)), 0};
    DataCopyPad<U, PaddingMode::Compact>(yGm_[baseXGmOffset], outLocal, outCopyParams);
    outQueue_.FreeTensor(outLocal);
}

} // namespace DynamicBlockQuant
#endif // DYNAMIC_BLOCK_QUANT_LARGE_BLOCKSIZE_H