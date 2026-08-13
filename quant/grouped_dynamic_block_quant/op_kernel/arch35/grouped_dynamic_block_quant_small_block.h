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
#include "grouped_dynamic_block_quant_split.h"

namespace GroupedDynamicBlockQuant {
using namespace AscendC;

template <typename T, typename U, int64_t RMode>
class GroupedDynamicBlockQuantSmallBlock : public GroupedSplitLocal<GroupedDynamicBlockQuantSmallBlock<T, U, RMode>> {
public:
    __aicore__ inline GroupedDynamicBlockQuantSmallBlock(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR groupList, GM_ADDR yOut, GM_ADDR scaleOut,
                                const GroupedDynamicBlockQuantTilingData& tilingData);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessOneLoop(const int64_t bIdx, const int64_t curBlockRowSize,
                                          const int64_t curBlockColSize, const int64_t blockRowIdx,
                                          const int64_t blockColIdx, const int64_t groupStart, const int64_t groupIdx);

private:
    // 独立noinline调度：隔离非wide-N的ColBand决策代码，避免扰动Process()主体(wide-N)代码生成
    __aicore__ __attribute__((noinline)) void DispatchNonWide(int64_t usedCoreNum, int64_t blockIdx, int64_t groupNum,
                                                              int64_t rowNum, int64_t uo, int64_t batchNum,
                                                              int64_t maxUbRow, int64_t blockFactor,
                                                              int64_t tailBlockFactor)
    {
        int64_t groupNumLoc = (groupNum > 0) ? groupNum : 1;
        int64_t realBatch = (batchNum > 0) ? batchNum : 1;
        int64_t mPerGroup = ops::CeilDiv(rowNum, groupNumLoc);
        int64_t totalUnits = realBatch * groupNumLoc * uo;
        bool useColBand = (rowNum >= groupNum) && (mPerGroup >= COLBAND_MIN_M_PER_GROUP) &&
                          (totalUnits >= usedCoreNum) &&
                          ((mPerGroup <= COLBAND_MAX_M_PER_GROUP) || (uo >= COLBAND_MIN_UO));
        if (useColBand) {
            this->ProcessColBand(usedCoreNum, blockIdx, groupNum, uo, batchNum, maxUbRow, blockFactor, tailBlockFactor);
        } else {
            this->ProcessBase(usedCoreNum, blockIdx, groupNum, maxUbRow, blockFactor, tailBlockFactor, uo, batchNum);
        }
    }

    __aicore__ inline void ParseTilingData(const GroupedDynamicBlockQuantTilingData& tilingData);
    __aicore__ inline void CopyIn(int64_t offset, int64_t blockCount, int64_t dataLen);
    __aicore__ inline void CopyOut(int64_t xOffset, int64_t scaleOffset, int64_t blockCount, int64_t dataLen);
    __aicore__ inline void SplitMaxUbRowCompute(int64_t blockCount, int64_t dataLen);
    __aicore__ inline void ComputeAlign(int64_t blockCount, int64_t dataLen, __ubuf__ T* xAddr,
                                        __ubuf__ uint32_t* scaleOutAddr, __ubuf__ U* yOutAddr);

private:
    int64_t totalCoreNum_ = 0;
    int64_t ubSize_ = 0;
    int64_t vfLen_ = 0;
    int64_t roundMode_ = 0;
    int64_t dstType_ = 0;
    int64_t batchNum_ = 0;
    int64_t blockIdx_ = 0;    // 核ID
    int64_t usedCoreNum_ = 0; // 实际使用的核数
    int64_t blockFactor_ = 0;
    int64_t tailBlockFactor_ = 0;
    int64_t uo_ = 0; // N轴计算cacheline次数
    int64_t maxUbRow_ = 0;
    int64_t nBatch_ = 0; // wide-N模式下每次CopyIn处理的N轴sub-block数，0表示非wide-N
    int64_t rowNum_ = 0;
    int64_t colNum_ = 0;
    int64_t scaleRowNum_ = 0;
    int64_t scaleColNum_ = 0;
    int64_t groupNum_ = 0;
    int64_t rowBlockSize_ = 0;
    int64_t colBlockSize_ = 0;
    float minScale_ = 0;
    // 目标数据类型的最大值
    float dstTypeMax_ = 0.0f;
    int64_t fullBlockNum = 0;
    int64_t resBlockRowNum = 0;
    float fp8MaxExpValue = 0.0;
    uint32_t infValue_ = 0;
    constexpr static int64_t DB_BUFFER = 2;
    constexpr static int64_t DIGIT_ONE = 1;
    constexpr static int64_t ADDR_PAD_OFFSET = 16;
    constexpr static int64_t SCALE_ALIGN_NUM = 8;
    // 列带流式(COLBAND)自适应判据：M轴每group行数不超过该阈值时，整组行在核内流式的代价可接受；
    // 否则2D切分跨核并行M，避免M过大时把整组行串行在单核导致并行度不足。
    // 对 uo<COLBAND_MIN_UO 的分支收紧到 150（small block 受益用例此分支 mpg 均<150，
    // 高 mpg 的受益用例 uo>=COLBAND_MIN_UO 走逃生阀，不受影响）。
    constexpr static int64_t COLBAND_MAX_M_PER_GROUP = 150;
    // COLBAND 的意义是"在 group 内流式 M 行"获取局部性：若每 group 仅 1 行(M轴无行可流式)，
    // ColBand 不再有收益、只剩调度开销，退回 2D ProcessBase。
    constexpr static int64_t COLBAND_MIN_M_PER_GROUP = 2;
    // rowBlockSize=1(small block)时，若N轴并行度(uo)足够大，即使M较大也保持列带流式以利用流式局部性。
    constexpr static int64_t COLBAND_MIN_UO = 100;
    constexpr static int64_t COLBAND_MAX_TOTAL_UNITS = 3000;
    constexpr static uint32_t FP32_INF_VALUE = 0x7f800000;
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
    GlobalTensor<U> yOutGm_;
    GlobalTensor<uint32_t> scaleOutGm_;

    static constexpr AscendC::MicroAPI::CastTrait castTrait32to8 = []() {
        if constexpr (RMode == 1) {
            return AscendC::MicroAPI::CastTrait{AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT,
                                                AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_RINT};
        } else if constexpr (RMode == 4) {
            return AscendC::MicroAPI::CastTrait{AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT,
                                                AscendC::MicroAPI::MaskMergeMode::ZEROING, RoundMode::CAST_ROUND};
        } else if constexpr (RMode == 7) {
            return AscendC::MicroAPI::CastTrait{AscendC::MicroAPI::RegLayout::ZERO, AscendC::MicroAPI::SatMode::SAT,
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
    ubSize_ = tilingData.ubSize;
    vfLen_ = tilingData.vfLen;
    minScale_ = tilingData.minScale;
    roundMode_ = tilingData.roundMode;
    dstType_ = tilingData.dstType;
    rowBlockSize_ = tilingData.rowBlockSize;
    colBlockSize_ = tilingData.colBlockSize;
    dstTypeMax_ = tilingData.dstTypeMax;
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
    nBatch_ = tilingData.headCoreNum;
}

template <typename T, typename U, int64_t RMode>
__aicore__ inline void GroupedDynamicBlockQuantSmallBlock<T, U, RMode>::Init(
    GM_ADDR x, GM_ADDR groupList, GM_ADDR yOut, GM_ADDR scaleOut, const GroupedDynamicBlockQuantTilingData& tilingData)
{
#if (__NPU_ARCH__ == 3510)
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
#endif
    ParseTilingData(tilingData);
    blockIdx_ = GetBlockIdx();
    if (this->blockIdx_ >= this->usedCoreNum_) {
        return;
    }
    infValue_ = FP32_INF_VALUE;
    this->InitGroup(groupList);
    this->xGm_.SetGlobalBuffer((__gm__ T*)(x));
    this->yOutGm_.SetGlobalBuffer((__gm__ U*)(yOut));
    this->scaleOutGm_.SetGlobalBuffer((__gm__ uint32_t*)(scaleOut));

    this->pipe_.InitBuffer(this->inQueue_, DB_BUFFER, colBlockSize_ * maxUbRow_ * sizeof(T));
    this->pipe_.InitBuffer(this->outQueue_, DB_BUFFER, colBlockSize_ * maxUbRow_ * sizeof(uint8_t));
    this->pipe_.InitBuffer(this->scaleQueue_, DB_BUFFER, ops::Ceil(maxUbRow_, rowBlockSize_) * 8 * sizeof(float));
}

template <typename T, typename U, int64_t RMode>
__aicore__ inline void GroupedDynamicBlockQuantSmallBlock<T, U, RMode>::CopyIn(int64_t offset, int64_t blockCount,
                                                                               int64_t dataLen)
{
    DataCopyExtParams inCopyParams_ = {static_cast<uint16_t>(blockCount), static_cast<uint32_t>(dataLen * sizeof(T)),
                                       static_cast<uint32_t>((colNum_ - dataLen) * sizeof(T)), static_cast<uint32_t>(0),
                                       static_cast<uint32_t>(0)};
    int64_t padLeftNum = isDataLenAlign32 ? 0 : 16;
    int64_t padRightNum = (dataLen % 16 == 0) ? 0 : (16 - dataLen % 16);
    DataCopyPadExtParams<T> padParams_ = {true, static_cast<uint8_t>(padLeftNum), static_cast<uint8_t>(padRightNum),
                                          static_cast<T>(0)};
    LocalTensor<T> x = inQueue_.AllocTensor<T>();
    DataCopyPad(x, xGm_[offset], inCopyParams_, padParams_);
    inQueue_.EnQue(x);
}

template <typename T, typename U, int64_t RMode>
__aicore__ inline void GroupedDynamicBlockQuantSmallBlock<T, U, RMode>::CopyOut(int64_t xOffset, int64_t scaleOffset,
                                                                                int64_t blockCount, int64_t dataLen)
{
    DataCopyExtParams outCopyParams_ = {
        static_cast<uint16_t>(blockCount), static_cast<uint32_t>(dataLen * sizeof(uint8_t)), static_cast<uint32_t>(0),
        static_cast<uint32_t>((colNum_ - dataLen) * sizeof(uint8_t)), static_cast<uint32_t>(0)};
    DataCopyExtParams scaleCopyParams_;
    if (nBatch_ > 1) {
        // wide-N: 一次搬出nBatch个scale，UB中每个scale间距SCALE_ALIGN_NUM*float，GM中连续
        int64_t scaleCount = dataLen / colBlockSize_;
        scaleCopyParams_ = {static_cast<uint16_t>(scaleCount), static_cast<uint32_t>(sizeof(float)),
                            static_cast<uint32_t>(SCALE_ALIGN_NUM * sizeof(float) - sizeof(float)),
                            static_cast<uint32_t>(0), static_cast<uint32_t>(0)};
    } else {
        scaleCopyParams_ = {static_cast<uint16_t>(ops::Ceil(blockCount, rowBlockSize_)),
                            static_cast<uint32_t>(1 * sizeof(float)), static_cast<uint32_t>(0),
                            static_cast<uint32_t>((ops::Ceil(colNum_, colBlockSize_) - DIGIT_ONE) * sizeof(uint32_t)),
                            static_cast<uint32_t>(0)};
    }
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
    if (this->blockIdx_ >= this->usedCoreNum_) {
        return;
    }
    int64_t tuTotal = (batchNum_ > 0 ? batchNum_ : 1) * (groupNum_ > 0 ? groupNum_ : 1) * uo_;
    if (tuTotal > COLBAND_MAX_TOTAL_UNITS) {
        if (nBatch_ > 1) {
            this->ProcessBase(usedCoreNum_, blockIdx_, groupNum_, 1, blockFactor_, tailBlockFactor_, uo_, batchNum_);
        } else {
            this->ProcessBase(usedCoreNum_, blockIdx_, groupNum_, maxUbRow_, blockFactor_, tailBlockFactor_, uo_,
                              batchNum_);
        }
        return;
    }
    if (nBatch_ > 1) {
        // wide-N: M轴每次1行，N轴每次nBatch个sub-block
        this->ProcessBase(usedCoreNum_, blockIdx_, groupNum_, 1, blockFactor_, tailBlockFactor_, uo_, batchNum_);
    } else {
        // 自适应调度交给独立DispatchNonWide，保持Process()主体(wide-N)代码生成不受扰动
        this->DispatchNonWide(usedCoreNum_, blockIdx_, groupNum_, rowNum_, uo_, batchNum_, maxUbRow_, blockFactor_,
                              tailBlockFactor_);
    }
}

template <typename T, typename U, int64_t RMode>
__aicore__ inline void GroupedDynamicBlockQuantSmallBlock<T, U, RMode>::ProcessOneLoop(
    const int64_t bIdx, const int64_t curBlockRowSize, const int64_t curBlockColSize, const int64_t blockRowIdx,
    const int64_t blockColIdx, const int64_t groupStart, const int64_t groupIdx)
{
    int64_t dataLen = curBlockRowSize;
    int64_t blockCount = curBlockColSize;
    isDataLenAlign32 = (dataLen % 32) == 0 || (dataLen % 32) > 16;

    int64_t xGmOffset;
    int64_t scaleGmOffset;
    if (nBatch_ > 1) {
        // wide-N: blockRowIdx是N轴大block索引，blockColIdx是M轴行索引
        xGmOffset = bIdx * rowNum_ * colNum_ + blockRowIdx * blockFactor_ + (groupStart + blockColIdx) * colNum_;
        scaleGmOffset = bIdx * scaleRowNum_ * scaleColNum_ + blockRowIdx * nBatch_ +
                        (groupStart / rowBlockSize_ + groupIdx + blockColIdx) * scaleColNum_;
    } else {
        xGmOffset = bIdx * rowNum_ * colNum_ + blockRowIdx * blockFactor_ +
                    (groupStart + blockColIdx * maxUbRow_) * colNum_;
        scaleGmOffset = bIdx * scaleRowNum_ * scaleColNum_ + blockRowIdx +
                        (groupStart / rowBlockSize_ + groupIdx + blockColIdx * (maxUbRow_ / rowBlockSize_)) *
                            scaleColNum_;
    }

    CopyIn(xGmOffset, blockCount, dataLen);
    SplitMaxUbRowCompute(blockCount, dataLen);
    CopyOut(xGmOffset, scaleGmOffset, blockCount, dataLen);
}

template <typename T, typename U, int64_t RMode>
__aicore__ inline void GroupedDynamicBlockQuantSmallBlock<T, U, RMode>::SplitMaxUbRowCompute(int64_t blockCount,
                                                                                             int64_t dataLen)
{
    LocalTensor<T> x = this->inQueue_.template DeQue<T>();
    LocalTensor<uint32_t> scaleOut = this->scaleQueue_.template AllocTensor<uint32_t>();
    LocalTensor<U> yOut = this->outQueue_.template AllocTensor<U>();

    auto xAddr = (__ubuf__ T*)x.GetPhyAddr();
    auto scaleOutAddr = (__ubuf__ uint32_t*)scaleOut.GetPhyAddr();
    auto yOutAddr = (__ubuf__ U*)yOut.GetPhyAddr();

    if (nBatch_ > 1) {
        // wide-N: 将nBatch个N轴sub-block视为nBatch个"虚拟行"，每行colBlockSize个元素
        int64_t subBlockCount = dataLen / colBlockSize_;
        ComputeAlign(subBlockCount, colBlockSize_, xAddr, scaleOutAddr, yOutAddr);
    } else if (isDataLenAlign32) {
        ComputeAlign(blockCount, dataLen, xAddr, scaleOutAddr, yOutAddr);
    } else {
        ComputeAlign(blockCount, dataLen, xAddr + ADDR_PAD_OFFSET, scaleOutAddr, yOutAddr);
    }

    this->scaleQueue_.template EnQue(scaleOut);
    this->outQueue_.template EnQue(yOut);
    this->inQueue_.template FreeTensor(x);
}

template <typename T, typename U, int64_t RMode>
__aicore__ inline void GroupedDynamicBlockQuantSmallBlock<T, U, RMode>::ComputeAlign(int64_t blockCount,
                                                                                     int64_t dataLen, __ubuf__ T* xAddr,
                                                                                     __ubuf__ uint32_t* scaleOutAddr,
                                                                                     __ubuf__ U* yOutAddr)
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
        if (dstTypeMax_ != 0) {
            fp8MaxExpValue = dstTypeMax_;
        } else {
            fp8MaxExpValue = HIFP8_MAX_VALUE;
        }
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
        AscendC::MicroAPI::MaskReg scaleMaskReg;
        AscendC::MicroAPI::MaskReg
            scaleMask32 = AscendC::MicroAPI::CreateMask<uint32_t, AscendC::MicroAPI::MaskPattern::ALL>();

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
                AscendC::MicroAPI::And(xAbsRegTensor, (AscendC::MicroAPI::RegTensor<uint16_t>&)xRegTensor,
                                       absMaskRegTensor, dataLenMask16);
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
            AscendC::MicroAPI::Div((AscendC::MicroAPI::RegTensor<float>&)scaleRegTensor, expMaxRegTensorFp32,
                                   fp8MaxRegTensor, scaleMask32);
            AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::LT>(
                scaleMaskReg, (AscendC::MicroAPI::RegTensor<uint32_t>&)scaleRegTensor, infValue_, scaleMask32);
            // Min(input_max / FP_MAX, 1 / minScale)
            AscendC::MicroAPI::Min<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(
                (AscendC::MicroAPI::RegTensor<float>&)scaleRegTensor,
                (AscendC::MicroAPI::RegTensor<float>&)scaleRegTensor, minScaleRegTensor, scaleMaskReg);
            AscendC::MicroAPI::StoreAlign<uint32_t, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
                scaleOutAddr + stepBlock * 8, scaleRegTensor, scaleMask32);

            // data value
            uint32_t pnum32 = normalBlockDataNum;
            for (uint16_t step64 = 0; step64 < static_cast<uint16_t>(normalStep64); step64++) {
                dataLenMask32 = AscendC::MicroAPI::UpdateMask<uint32_t>(pnum32);
                AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                    xRegTensor, xAddr + stepBlock * blockDataNumOffset + step64 * vfNum);
                AscendC::MicroAPI::Cast<float, T, castTraitT2Float>(
                    (AscendC::MicroAPI::RegTensor<float>&)yRegTensorFp32, xRegTensor, dataLenMask32);
                AscendC::MicroAPI::Div((AscendC::MicroAPI::RegTensor<float>&)yRegTensorFp32,
                                       (AscendC::MicroAPI::RegTensor<float>&)yRegTensorFp32,
                                       (AscendC::MicroAPI::RegTensor<float>&)scaleRegTensor, dataLenMask32);
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
            AscendC::MicroAPI::And(xAbsRegTensor, (AscendC::MicroAPI::RegTensor<uint16_t>&)xRegTensor, absMaskRegTensor,
                                   dataLenMask16);
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
        AscendC::MicroAPI::Div((AscendC::MicroAPI::RegTensor<float>&)scaleRegTensor, expMaxRegTensorFp32,
                               fp8MaxRegTensor, scaleMask32);
        AscendC::MicroAPI::CompareScalar<uint32_t, CMPMODE::LT>(
            scaleMaskReg, (AscendC::MicroAPI::RegTensor<uint32_t>&)scaleRegTensor, infValue_, scaleMask32);
        // Min(input_max / FP_MAX, 1 / minScale)
        AscendC::MicroAPI::Min<float, AscendC::MicroAPI::MaskMergeMode::MERGING>(
            (AscendC::MicroAPI::RegTensor<float>&)scaleRegTensor, (AscendC::MicroAPI::RegTensor<float>&)scaleRegTensor,
            minScaleRegTensor, scaleMaskReg);
        AscendC::MicroAPI::StoreAlign<uint32_t, AscendC::MicroAPI::StoreDist::DIST_FIRST_ELEMENT_B32>(
            scaleOutAddr + tailBlockIdx * 8, scaleRegTensor, scaleMask32);

        // data value
        uint32_t pnum32 = tailBlockDataNum;
        for (uint16_t step64 = 0; step64 < static_cast<uint16_t>(tailStep64); step64++) {
            dataLenMask32 = AscendC::MicroAPI::UpdateMask<uint32_t>(pnum32);
            AscendC::MicroAPI::LoadAlign<T, AscendC::MicroAPI::LoadDist::DIST_UNPACK_B16>(
                xRegTensor, xAddr + tailBlockIdx * blockDataNumOffset + step64 * vfNum);
            AscendC::MicroAPI::Cast<float, T, castTraitT2Float>((AscendC::MicroAPI::RegTensor<float>&)yRegTensorFp32,
                                                                xRegTensor, dataLenMask32);
            AscendC::MicroAPI::Div((AscendC::MicroAPI::RegTensor<float>&)yRegTensorFp32,
                                   (AscendC::MicroAPI::RegTensor<float>&)yRegTensorFp32,
                                   (AscendC::MicroAPI::RegTensor<float>&)scaleRegTensor, dataLenMask32);
            AscendC::MicroAPI::Cast<U, float, castTrait32to8>(
                yRegTensorFp8, (AscendC::MicroAPI::RegTensor<float>&)yRegTensorFp32, dataLenMask32);
            AscendC::MicroAPI::StoreAlign<U, MicroAPI::StoreDist::DIST_PACK4_B32>(
                yOutAddr + tailBlockIdx * blockDataNumOffset + step64 * vfNum, yRegTensorFp8, dataLenMask32);
        }
    }
}

} // namespace GroupedDynamicBlockQuant
#endif // GROUPED_DYNAMIC_BLOCK_QUANT_SMALL_BLOCK_H
