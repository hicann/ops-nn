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
 * \file grouped_dynamic_block_quant_small_block.h
 * \brief
 */

#ifndef GROUPED_DYNAMIC_BLOCK_QUANT_SMALL_BLOCK_H
#define GROUPED_DYNAMIC_BLOCK_QUANT_SMALL_BLOCK_H
#define FLOAT_OVERFLOW_MODE_CTRL 60
#include "kernel_operator.h"
#include "../inc/platform.h"

namespace GroupedDynamicBlockQuant {
using namespace AscendC;

template <typename T, typename U, int64_t RMode>
class GroupedDynamicBlockQuantSmallBlock {
public:
    __aicore__ inline GroupedDynamicBlockQuantSmallBlock(){};
    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR groupList, GM_ADDR yOut, GM_ADDR scaleOut,
        const GroupedDynamicBlockQuantTilingData& tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseTilingData(const GroupedDynamicBlockQuantTilingData& tilingData);
    __aicore__ inline void CopyIn(int64_t offset, int64_t blockCount, int64_t dataLen);
    __aicore__ inline void CopyOut(int64_t xOffset, int64_t scaleOffset, int64_t blockCount, int64_t dataLen);
    __aicore__ inline void SplitMaxUbRowCompute(int64_t blockCount, int64_t dataLen);
    __aicore__ inline void ComputeAlign(
        int64_t blockCount, int64_t dataLen, __ubuf__ T* xAddr, __ubuf__ uint32_t* scaleOutAddr, __ubuf__ U* yOutAddr);

private:
    int64_t totalCoreNum_ = 0;
    int64_t ubSize_ = 0;
    int64_t vfLen_ = 0;
    int64_t roundMode_ = 0;
    int64_t dstType_ = 0;
    int64_t batchNum_ = 0;
    int64_t blockIdx_ = 0;    // 核ID
    int64_t usedCoreNum_ = 0; // 实际使用的核数
    int64_t headCoreNum_ = 0;
    int64_t groupBlockNumHeadCore_ = 0;
    int64_t groupBlockNumTailCore_ = 0;
    int64_t blockFactor_ = 0;
    int64_t tailBlockFactor_ = 0;
    int64_t blockLoopOffset_ = 0; // 当前核起始偏移
    int64_t uo_ = 0;              // N轴计算cacheline次数
    int64_t maxUbRow_ = 0;
    int64_t rowNum_ = 0;
    int64_t colNum_ = 0;
    int64_t scaleRowNum_ = 0;
    int64_t scaleColNum_ = 0;
    int64_t groupNum_ = 0;
    int64_t rowBlockSize_ = 0;
    int64_t colBlockSize_ = 0;
    float minScale_ = 0;
    int64_t fullBlockNum = 0;
    int64_t resBlockRowNum = 0;
    float fp8MaxExpValue = 0.0;
    constexpr static int64_t DB_BUFFER = 2;
    constexpr static int64_t DIGIT_ONE = 1;
    constexpr static int64_t ADDR_PAD_OFFSET = 16;
    constexpr static float FP8_E5M2_MAX_VALUE = 57344.0f;
    constexpr static float FP8_E4M3_MAX_VALUE = 448.0f;
    constexpr static float HIFP8_MAX_VALUE = 32768.0f;
    constexpr static uint16_t ABS_MASK_FOR_16 = 0x7fff;
    bool isDataLenAlign32 = false;

    TPipe pipe_;
    TQue<QuePosition::VECIN, DB_BUFFER> inQueue_;
    TQue<QuePosition::VECOUT, DB_BUFFER> scaleQueue_;
    TQue<QuePosition::VECOUT, DB_BUFFER> outQueue_;
    GlobalTensor<T> xGm_;
    GlobalTensor<int32_t> groupListGm_;
    GlobalTensor<U> yOutGm_;
    GlobalTensor<uint32_t> scaleOutGm_;

    static constexpr AscendC::MicroAPI::CastTrait castTrait32to8 = []() {
        if constexpr (RMode == 1) {
            return AscendC::MicroAPI::CastTrait{
                AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT,
                AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        } else if constexpr (RMode == 4) {
            return AscendC::MicroAPI::CastTrait{
                AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT,
                AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};
        } else if constexpr (RMode == 7) {
            return AscendC::MicroAPI::CastTrait{
                AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT,
                AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_HYBRID};
        }
    }();
    static constexpr AscendC::MicroAPI::CastTrait castTraitT2Float = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    static constexpr AscendC::MicroAPI::CastTrait castTraitZero = {
        AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::UNKNOWN,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
    static constexpr AscendC::MicroAPI::CastTrait castTraitOne = {
        AscendC::MicroAPI::RegLayout::ONE, AscendC::MicroAPI::SatMode::UNKNOWN,
        AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
};

template <typename T, typename U, int64_t RMode>
__aicore__ inline void GroupedDynamicBlockQuantSmallBlock<T, U, RMode>::ParseTilingData(
    const GroupedDynamicBlockQuantTilingData& tilingData)
{
    totalCoreNum_ = tilingData.totalCoreNum;
    usedCoreNum_ = tilingData.usedCoreNum;
    headCoreNum_ = tilingData.headCoreNum;
    groupBlockNumHeadCore_ = tilingData.groupBlockNumHeadCore;
    groupBlockNumTailCore_ = tilingData.groupBlockNumTailCore;
    ubSize_ = tilingData.ubSize;
    vfLen_ = tilingData.vfLen;
    minScale_ = tilingData.minScale;
    roundMode_ = tilingData.roundMode;
    dstType_ = tilingData.dstType;
    rowBlockSize_ = tilingData.rowBlockSize;
    colBlockSize_ = tilingData.colBlockSize;
    batchNum_ = tilingData.batchNum;
    rowNum_ = tilingData.rowNum;
    colNum_ = tilingData.colNum;
    scaleRowNum_ = tilingData.scaleRowNum;
    scaleColNum_ = tilingData.scaleColNum;
    uo_ = tilingData.uo;
    blockFactor_ = tilingData.blockFactor;
    tailBlockFactor_ = tilingData.tailBlockFactor;
    groupNum_ = tilingData.groupNum;
    maxUbRow_ = tilingData.maxUbRow;
}

template <typename T, typename U, int64_t RMode>
__aicore__ inline void GroupedDynamicBlockQuantSmallBlock<T, U, RMode>::Init(
    GM_ADDR x, GM_ADDR groupList, GM_ADDR yOut, GM_ADDR scaleOut, const GroupedDynamicBlockQuantTilingData& tilingData)
{
#if (__NPU_ARCH__ == 3101)
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
#endif
    ParseTilingData(tilingData);
    blockIdx_ = GetBlockIdx();
    if (this->blockIdx_ >= this->usedCoreNum_) {
        return;
    }

    this->xGm_.SetGlobalBuffer((__gm__ T*)(x));
    this->groupListGm_.SetGlobalBuffer((__gm__ int32_t*)(groupList));
    this->yOutGm_.SetGlobalBuffer((__gm__ U*)(yOut));
    this->scaleOutGm_.SetGlobalBuffer((__gm__ uint32_t*)(scaleOut));

    this->pipe_.InitBuffer(this->inQueue_, DB_BUFFER, colBlockSize_ * maxUbRow_ * sizeof(T));
    this->pipe_.InitBuffer(this->outQueue_, DB_BUFFER, colBlockSize_ * maxUbRow_ * sizeof(uint8_t));
    this->pipe_.InitBuffer(this->scaleQueue_, DB_BUFFER, ops::Ceil(maxUbRow_, rowBlockSize_) * 8 * sizeof(float));
}

template <typename T, typename U, int64_t RMode>
__aicore__ inline void GroupedDynamicBlockQuantSmallBlock<T, U, RMode>::CopyIn(
    int64_t offset, int64_t blockCount, int64_t dataLen)
{
    DataCopyExtParams inCopyParams_ = {
        static_cast<uint16_t>(blockCount), static_cast<uint32_t>(dataLen * sizeof(T)),
        static_cast<uint32_t>((colNum_ - dataLen) * sizeof(T)), static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    int64_t padLeftNum = isDataLenAlign32 ? 0 : 16;
    int64_t padRightNum = (dataLen % 16 == 0) ? 0 : (16 - dataLen % 16);
    DataCopyPadExtParams<T> padParams_ = {
        true, static_cast<uint8_t>(padLeftNum), static_cast<uint8_t>(padRightNum), static_cast<T>(0)};
    LocalTensor<T> x = inQueue_.AllocTensor<T>();
    DataCopyPad(x, xGm_[offset], inCopyParams_, padParams_);
    inQueue_.EnQue(x);
}

template <typename T, typename U, int64_t RMode>
__aicore__ inline void GroupedDynamicBlockQuantSmallBlock<T, U, RMode>::CopyOut(
    int64_t xOffset, int64_t scaleOffset, int64_t blockCount, int64_t dataLen)
{
    DataCopyExtParams outCopyParams_ = {
        static_cast<uint16_t>(blockCount), static_cast<uint32_t>(dataLen * sizeof(uint8_t)), static_cast<uint32_t>(0),
        static_cast<uint32_t>((colNum_ - dataLen) * sizeof(uint8_t)), static_cast<uint32_t>(0)};
    DataCopyExtParams scaleCopyParams_ = {
        static_cast<uint16_t>(ops::Ceil(blockCount, rowBlockSize_)), static_cast<uint32_t>(1 * sizeof(float)),
        static_cast<uint32_t>(0),
        static_cast<uint32_t>((ops::Ceil(colNum_, colBlockSize_) - DIGIT_ONE) * sizeof(uint32_t)),
        static_cast<uint32_t>(0)};
    LocalTensor<U> yOut = outQueue_.DeQue<U>();
    DataCopyPad(yOutGm_[xOffset], yOut, outCopyParams_);
    outQueue_.FreeTensor(yOut);
    LocalTensor<uint32_t> scaleOut = scaleQueue_.DeQue<uint32_t>();
    DataCopyPad(scaleOutGm_[scaleOffset], scaleOut, scaleCopyParams_);
    scaleQueue_.FreeTensor(scaleOut);
}

template <typename T, typename U, int64_t RMode>
__aicore__ inline void GroupedDynamicBlockQuantSmallBlock<T, U, RMode>::Process()
{
    bool isTailBlock = blockIdx_ >= headCoreNum_;
    int64_t loopNum = isTailBlock ? groupBlockNumTailCore_ : groupBlockNumHeadCore_;
    blockLoopOffset_ = isTailBlock ?
                           headCoreNum_ * groupBlockNumHeadCore_ + (blockIdx_ - headCoreNum_) * groupBlockNumTailCore_ :
                           blockIdx_ * groupBlockNumHeadCore_;

    for (int64_t loopIdx = 0; loopIdx < loopNum; loopIdx++) {
        int64_t curLoopIdxInAllCore = loopIdx + blockLoopOffset_;
        int64_t bIdx = curLoopIdxInAllCore / (uo_ * groupNum_);
        int64_t nIdx = (curLoopIdxInAllCore - bIdx * groupNum_ * uo_) / groupNum_;
        int64_t gIdx = curLoopIdxInAllCore % groupNum_;
        int64_t dataLen = (nIdx == uo_ - 1) ? tailBlockFactor_ : blockFactor_;
        isDataLenAlign32 = (dataLen % 32) == 0 || (dataLen % 32) > 16;

        int64_t xBatchGmOffset = bIdx * rowNum_ * colNum_;
        int64_t scaleBatchGmOffset = bIdx * scaleRowNum_ * scaleColNum_;
        int64_t xTileGmOffset = nIdx * blockFactor_;
        int64_t scaleTileGmOffset = nIdx;

        int64_t gIdxValueStart = (gIdx > 0) ? groupListGm_.GetValue(gIdx - 1) : 0;
        int64_t gIdxValueEnd = groupListGm_.GetValue(gIdx);
        assert(
            (0 <= gIdxValueEnd && gIdxValueEnd <= rowNum_), "groupList %lld = %lld out of range[0 %lld]!\n", gIdx,
            gIdxValueEnd, rowNum_);
        assert(
            (gIdxValueStart <= gIdxValueEnd), "groupList %lld = %lld less than previous groupIdx = %lld!\n", gIdx,
            gIdxValueEnd, gIdxValueStart);
        int64_t groupSizeIdx = gIdxValueEnd - gIdxValueStart;
        int64_t inLoopNum = ops::Ceil(groupSizeIdx, maxUbRow_);

        int64_t xGroupOffset = xBatchGmOffset + xTileGmOffset + gIdxValueStart * colNum_;
        int64_t scaleGroupOffset =
            scaleBatchGmOffset + scaleTileGmOffset + (gIdxValueStart / rowBlockSize_ + gIdx) * scaleColNum_;
        for (int64_t i = 0; i < inLoopNum; i++) {
            int64_t blockCount = (i == inLoopNum - 1) ? groupSizeIdx - i * maxUbRow_ : maxUbRow_;
            int64_t xGmOffset = xGroupOffset + i * maxUbRow_ * colNum_;
            int64_t scaleGmOffset = scaleGroupOffset + i * (maxUbRow_ / rowBlockSize_) * scaleColNum_;
            CopyIn(xGmOffset, blockCount, dataLen);
            SplitMaxUbRowCompute(blockCount, dataLen);
            CopyOut(xGmOffset, scaleGmOffset, blockCount, dataLen);
        }
    }
}

template <typename T, typename U, int64_t RMode>
__aicore__ inline void GroupedDynamicBlockQuantSmallBlock<T, U, RMode>::SplitMaxUbRowCompute(
    int64_t blockCount, int64_t dataLen)
{
    LocalTensor<T> x = this->inQueue_.template DeQue<T>();
    LocalTensor<uint32_t> scaleOut = this->scaleQueue_.template AllocTensor<uint32_t>();
    LocalTensor<U> yOut = this->outQueue_.template AllocTensor<U>();

    auto xAddr = (__ubuf__ T*)x.GetPhyAddr();
    auto scaleOutAddr = (__ubuf__ uint32_t*)scaleOut.GetPhyAddr();
    auto yOutAddr = (__ubuf__ U*)yOut.GetPhyAddr();

    if (isDataLenAlign32) {
        ComputeAlign(blockCount, dataLen, xAddr, scaleOutAddr, yOutAddr);
    } else {
        ComputeAlign(blockCount, dataLen, xAddr + ADDR_PAD_OFFSET, scaleOutAddr, yOutAddr);
    }

    this->scaleQueue_.template EnQue(scaleOut);
    this->outQueue_.template EnQue(yOut);
    this->inQueue_.template FreeTensor(x);
}

template <typename T, typename U, int64_t RMode>
__aicore__ inline void GroupedDynamicBlockQuantSmallBlock<T, U, RMode>::ComputeAlign(
    int64_t blockCount, int64_t dataLen, __ubuf__ T* xAddr, __ubuf__ uint32_t* scaleOutAddr, __ubuf__ U* yOutAddr)
{
    constexpr uint32_t vfLen = platform::GetVRegSize() / sizeof(T);     // 寄存器单次能处理的长度
    constexpr uint32_t vfNum = platform::GetVRegSize() / sizeof(float); // cast到FP32后单个寄存器中的元素个数
    int64_t blockNum = ops::Ceil(blockCount, rowBlockSize_);
    int64_t tailBlockIdx = blockNum - 1;
    int64_t outDataLenAlign = 0;
    int64_t tailBlockDataNum = 0;
    int64_t normalBlockDataNum = 0;
    int64_t blockDataNumOffset = 0;

    if (isDataLenAlign32) {
        outDataLenAlign = (dataLen + 15) / 16 * 16;
        tailBlockDataNum = (blockCount - tailBlockIdx * rowBlockSize_) * outDataLenAlign;
        normalBlockDataNum = rowBlockSize_ * outDataLenAlign;
        blockDataNumOffset = normalBlockDataNum;
    } else {
        outDataLenAlign = (dataLen + 15) / 16 * 16 + ADDR_PAD_OFFSET;
        tailBlockDataNum = (blockCount - tailBlockIdx * rowBlockSize_) * outDataLenAlign - ADDR_PAD_OFFSET;
        normalBlockDataNum = rowBlockSize_ * outDataLenAlign - ADDR_PAD_OFFSET;
        blockDataNumOffset = rowBlockSize_ * outDataLenAlign;
    }

    uint32_t tailStep128 = ops::Ceil(static_cast<uint32_t>(tailBlockDataNum), vfLen);
    uint32_t normalStep128 = ops::Ceil(static_cast<uint32_t>(normalBlockDataNum), vfLen);
    uint32_t tailStep64 = ops::Ceil(static_cast<uint32_t>(tailBlockDataNum), vfNum);
    uint32_t normalStep64 = ops::Ceil(static_cast<uint32_t>(normalBlockDataNum), vfNum);

    if constexpr (ops::IsSame<U, fp8_e4m3fn_t>::value) {
        fp8MaxExpValue = FP8_E4M3_MAX_VALUE;
    } else if constexpr (ops::IsSame<U, fp8_e5m2_t>::value) {
        fp8MaxExpValue = FP8_E5M2_MAX_VALUE;
    } else if constexpr (ops::IsSame<U, hifloat8_t>::value) {
        fp8MaxExpValue = HIFP8_MAX_VALUE;
    }

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<T> xRegTensor;
        AscendC::MicroAPI::RegTensor<T> xRegTensorZero;
        AscendC::MicroAPI::RegTensor<T> zeroRegTensor16;
        AscendC::MicroAPI::RegTensor<uint32_t> zeroRegTensor32;
        AscendC::MicroAPI::RegTensor<uint32_t> xRegTensorFp32;
        AscendC::MicroAPI::RegTensor<uint16_t> expMaxRegTensor;
        AscendC::MicroAPI::RegTensor<float> expMaxRegTensorZero;
        AscendC::MicroAPI::RegTensor<float> expMaxRegTensorOne;
        AscendC::MicroAPI::RegTensor<uint16_t> xAbsRegTensor;
        AscendC::MicroAPI::RegTensor<uint16_t> absMaskRegTensor;
        AscendC::MicroAPI::RegTensor<float> expMaxRegTensorFp32;
        AscendC::MicroAPI::RegTensor<float> oneRegTensor32;
        AscendC::MicroAPI::RegTensor<float> reverseMinScaleRegTensor;
        AscendC::MicroAPI::RegTensor<float> minScaleRegTensor;
        AscendC::MicroAPI::RegTensor<float> fp8MaxRegTensor;
        AscendC::MicroAPI::RegTensor<uint32_t> scaleRegTensor;
        AscendC::MicroAPI::RegTensor<uint32_t> yRegTensorFp32;
        AscendC::MicroAPI::RegTensor<uint32_t> y;
        AscendC::MicroAPI::RegTensor<U> yRegTensorFp8;

        AscendC::MicroAPI::MaskReg p1;
        AscendC::MicroAPI::MaskReg p2;
        AscendC::MicroAPI::MaskReg dataLenMask16;
        AscendC::MicroAPI::MaskReg dataLenMask32;
        AscendC::MicroAPI::MaskReg scaleMask32 =
            AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();

        AscendC::MicroAPI::Duplicate(absMaskRegTensor, ABS_MASK_FOR_16);
        AscendC::MicroAPI::Duplicate(oneRegTensor32, DIGIT_ONE);
        AscendC::MicroAPI::Duplicate(fp8MaxRegTensor, fp8MaxExpValue);
        AscendC::MicroAPI::Duplicate(zeroRegTensor16, 0);
        AscendC::MicroAPI::Duplicate(zeroRegTensor32, 0x00000000);
        AscendC::MicroAPI::Duplicate(minScaleRegTensor, minScale_);
        AscendC::MicroAPI::Div(minScaleRegTensor, oneRegTensor32, minScaleRegTensor, scaleMask32);

        for (uint16_t stepBlock = 0; stepBlock < static_cast<uint16_t>(tailBlockIdx); stepBlock++) {
            uint32_t pnum16 = normalBlockDataNum;
            AscendC::MicroAPI::Duplicate(expMaxRegTensor, 0x0000);
            for (uint16_t step128 = 0; step128 < static_cast<uint16_t>(normalStep128); step128++) {
                dataLenMask16 = AscendC::MicroAPI::UpdateMask<uint16_t>(pnum16);
                AscendC::MicroAPI::LoadAlign(xRegTensor, xAddr + stepBlock * blockDataNumOffset + step128 * vfLen);
                AscendC::MicroAPI::Mul(xRegTensorZero, xRegTensor, zeroRegTensor16, dataLenMask16);
                AscendC::MicroAPI::Compare<T, CMPMODE::NE>(p1, xRegTensorZero, xRegTensorZero, dataLenMask16);
                AscendC::MicroAPI::Select(xRegTensor, zeroRegTensor16, xRegTensor, p1);
                AscendC::MicroAPI::And(
                    xAbsRegTensor, (AscendC::MicroAPI::RegTensor<uint16_t>&)xRegTensor, absMaskRegTensor,
                    dataLenMask16);
                AscendC::MicroAPI::Max<uint16_t, AscendC::MicroAPI::MaskMergeMode::MERGING>(
                    expMaxRegTensor, expMaxRegTensor, xAbsRegTensor, dataLenMask16);
            }
            pnum16 = normalBlockDataNum;
            dataLenMask16 = AscendC::MicroAPI::UpdateMask<uint16_t>(pnum16);
            AscendC::MicroAPI::ReduceMax<uint16_t>(expMaxRegTensor, expMaxRegTensor, dataLenMask16);
            AscendC::MicroAPI::Cast<float, T, castTraitT2Float>(
                expMaxRegTensorFp32, (AscendC::MicroAPI::RegTensor<T>&)expMaxRegTensor, dataLenMask16);

            // scale
            AscendC::MicroAPI::Duplicate(expMaxRegTensorFp32, expMaxRegTensorFp32, scaleMask32);
            AscendC::MicroAPI::Div(
                (AscendC::MicroAPI::RegTensor<float>&)scaleRegTensor, expMaxRegTensorFp32, fp8MaxRegTensor,
                scaleMask32);
            AscendC::MicroAPI::Min(
                (AscendC::MicroAPI::RegTensor<float>&)scaleRegTensor,
                (AscendC::MicroAPI::RegTensor<float>&)scaleRegTensor, minScaleRegTensor, scaleMask32);
            AscendC::MicroAPI::StoreAlign<uint32_t, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                scaleOutAddr + stepBlock * 8, scaleRegTensor, scaleMask32);

            // data value
            uint32_t pnum32 = normalBlockDataNum;
            for (uint16_t step64 = 0; step64 < static_cast<uint16_t>(normalStep64); step64++) {
                dataLenMask32 = AscendC::MicroAPI::UpdateMask<uint32_t>(pnum32);
                AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    xRegTensor, xAddr + stepBlock * blockDataNumOffset + step64 * vfNum);
                AscendC::MicroAPI::Cast<float, T, castTraitT2Float>(
                    (AscendC::MicroAPI::RegTensor<float>&)xRegTensorFp32, xRegTensor, dataLenMask32);
                AscendC::MicroAPI::Muls(
                    (AscendC::MicroAPI::RegTensor<float>&)y, (AscendC::MicroAPI::RegTensor<float>&)xRegTensorFp32, 0.0f,
                    dataLenMask32);
                AscendC::MicroAPI::Compare<float, CMPMODE::NE>(
                    p2, (AscendC::MicroAPI::RegTensor<float>&)y, (AscendC::MicroAPI::RegTensor<float>&)y,
                    dataLenMask32);
                AscendC::MicroAPI::Select(yRegTensorFp32, zeroRegTensor32, xRegTensorFp32, p2);
                AscendC::MicroAPI::Div(
                    (AscendC::MicroAPI::RegTensor<float>&)yRegTensorFp32,
                    (AscendC::MicroAPI::RegTensor<float>&)yRegTensorFp32,
                    (AscendC::MicroAPI::RegTensor<float>&)scaleRegTensor, dataLenMask32);
                AscendC::MicroAPI::Select(yRegTensorFp32, xRegTensorFp32, yRegTensorFp32, p2);
                AscendC::MicroAPI::Cast<U, float, castTrait32to8>(
                    yRegTensorFp8, (AscendC::MicroAPI::RegTensor<float>&)yRegTensorFp32, dataLenMask32);
                AscendC::MicroAPI::StoreAlign<U, MicroAPI::StoreDist::DIST_PACK4_B32>(
                    yOutAddr + stepBlock * blockDataNumOffset + step64 * vfNum, yRegTensorFp8, dataLenMask32);
            }
        }

        // tail process
        uint32_t pnum16 = tailBlockDataNum;
        AscendC::MicroAPI::Duplicate(expMaxRegTensor, 0x0000);
        for (uint16_t step128 = 0; step128 < static_cast<uint16_t>(tailStep128); step128++) {
            dataLenMask16 = AscendC::MicroAPI::UpdateMask<uint16_t>(pnum16);
            AscendC::MicroAPI::LoadAlign(xRegTensor, xAddr + tailBlockIdx * blockDataNumOffset + step128 * vfLen);
            AscendC::MicroAPI::Mul(xRegTensorZero, xRegTensor, zeroRegTensor16, dataLenMask16);
            AscendC::MicroAPI::Compare<T, CMPMODE::NE>(p1, xRegTensorZero, xRegTensorZero, dataLenMask16);
            AscendC::MicroAPI::Select(xRegTensor, zeroRegTensor16, xRegTensor, p1);
            AscendC::MicroAPI::And(
                xAbsRegTensor, (AscendC::MicroAPI::RegTensor<uint16_t>&)xRegTensor, absMaskRegTensor, dataLenMask16);
            AscendC::MicroAPI::Max<uint16_t, AscendC::MicroAPI::MaskMergeMode::MERGING>(
                expMaxRegTensor, expMaxRegTensor, xAbsRegTensor, dataLenMask16);
        }
        pnum16 = tailBlockDataNum;
        dataLenMask16 = AscendC::MicroAPI::UpdateMask<uint16_t>(pnum16);
        AscendC::MicroAPI::ReduceMax<uint16_t>(expMaxRegTensor, expMaxRegTensor, dataLenMask16);
        AscendC::MicroAPI::Cast<float, T, castTraitT2Float>(
            expMaxRegTensorFp32, (AscendC::MicroAPI::RegTensor<T>&)expMaxRegTensor, dataLenMask16);

        // scale
        AscendC::MicroAPI::Duplicate(expMaxRegTensorFp32, expMaxRegTensorFp32, scaleMask32);
        AscendC::MicroAPI::Div(
            (AscendC::MicroAPI::RegTensor<float>&)scaleRegTensor, expMaxRegTensorFp32, fp8MaxRegTensor, scaleMask32);
        AscendC::MicroAPI::Min(
            (AscendC::MicroAPI::RegTensor<float>&)scaleRegTensor, (AscendC::MicroAPI::RegTensor<float>&)scaleRegTensor,
            minScaleRegTensor, scaleMask32);
        AscendC::MicroAPI::StoreAlign<uint32_t, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
            scaleOutAddr + tailBlockIdx * 8, scaleRegTensor, scaleMask32);

        // data value
        uint32_t pnum32 = tailBlockDataNum;
        for (uint16_t step64 = 0; step64 < static_cast<uint16_t>(tailStep64); step64++) {
            dataLenMask32 = AscendC::MicroAPI::UpdateMask<uint32_t>(pnum32);
            AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                xRegTensor, xAddr + tailBlockIdx * blockDataNumOffset + step64 * vfNum);
            AscendC::MicroAPI::Cast<float, T, castTraitT2Float>(
                (AscendC::MicroAPI::RegTensor<float>&)xRegTensorFp32, xRegTensor, dataLenMask32);
            AscendC::MicroAPI::Muls(
                (AscendC::MicroAPI::RegTensor<float>&)y, (AscendC::MicroAPI::RegTensor<float>&)xRegTensorFp32, 0.0f,
                dataLenMask32);
            AscendC::MicroAPI::Compare<float, CMPMODE::NE>(
                p2, (AscendC::MicroAPI::RegTensor<float>&)y, (AscendC::MicroAPI::RegTensor<float>&)y, dataLenMask32);
            AscendC::MicroAPI::Select(yRegTensorFp32, zeroRegTensor32, xRegTensorFp32, p2);
            AscendC::MicroAPI::Div(
                (AscendC::MicroAPI::RegTensor<float>&)yRegTensorFp32,
                (AscendC::MicroAPI::RegTensor<float>&)yRegTensorFp32,
                (AscendC::MicroAPI::RegTensor<float>&)scaleRegTensor, dataLenMask32);
            AscendC::MicroAPI::Select(yRegTensorFp32, xRegTensorFp32, yRegTensorFp32, p2);
            AscendC::MicroAPI::Cast<U, float, castTrait32to8>(
                yRegTensorFp8, (AscendC::MicroAPI::RegTensor<float>&)yRegTensorFp32, dataLenMask32);
            AscendC::MicroAPI::StoreAlign<U, MicroAPI::StoreDist::DIST_PACK4_B32>(
                yOutAddr + tailBlockIdx * blockDataNumOffset + step64 * vfNum, yRegTensorFp8, dataLenMask32);
        }
    }
}

} // namespace GroupedDynamicBlockQuant
#endif // GROUPED_DYNAMIC_BLOCK_QUANT_SMALL_BLOCK_H