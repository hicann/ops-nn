/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef GROUPED_DYNAMIC_MX_QUANT_WITH_DUAL_AXIS_REGBASE_H
#define GROUPED_DYNAMIC_MX_QUANT_WITH_DUAL_AXIS_REGBASE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../inc/platform.h"
#include "grouped_dynamic_mx_quant_with_dual_axis_tiling_key.h"
#include "grouped_dynamic_mx_quant_with_dual_axis_tilingdata.h"

namespace GroupedDynamicMxQuantWithDualAxis {
using namespace AscendC;
#define FLOAT_OVERFLOW_MODE_CTRL 60

template <typename T>
using RegTensor = AscendC::Reg::RegTensor<T>;
using MaskReg = AscendC::Reg::MaskReg;

constexpr int64_t MX_BLOCK_SIZE = 32;
constexpr int64_t DOUBLE_MX_BLOCK_SIZE = 64;
constexpr uint16_t MX_BLOCK_SIZE_U16 = 32;
constexpr uint16_t DIGIT_TWO = 2;
constexpr uint16_t SCALE2_ROW_SLOT_COUNT = 2;
constexpr int32_t QUEUE_DEPTH = 2;
constexpr int64_t DTYPE_FLOAT8_E5M2 = 35;
constexpr int64_t DTYPE_FLOAT8_E4M3FN = 36;
constexpr uint32_t REG_VF_LEN_FP32 = platform::GetVRegSize() / sizeof(float);
constexpr uint32_t REG_VF_LEN_B8 = platform::GetVRegSize() / sizeof(uint8_t);
constexpr uint32_t REG_VF_LEN_B16 = platform::GetVRegSize() / sizeof(uint16_t);
constexpr uint32_t REG_VF_LEN_B16_DOUBLE = REG_VF_LEN_B16 * DIGIT_TWO;
constexpr uint16_t ELEMENT_AFTER_REDUCE = 8;
constexpr int64_t UB_BLOCK_SIZE = platform::GetUbBlockSize();
constexpr uint16_t ABS_FOR_UINT16 = 0x7fff;
constexpr uint32_t MAN_FOR_FP32 = 0x007fffff;
constexpr uint32_t MAX_EXP_FOR_FP8_IN_FP32 = 0x000000ff;
constexpr uint32_t FP32_EXP_BIAS_CEIL_ALG = 0x00007f00;
constexpr uint32_t NAN_CUSTOMIZATION_PACK = 0x00007f81;
constexpr int16_t SHR_NUM_FOR_BF16 = 7;
constexpr int16_t SHR_NUM_FOR_FP32 = 23;
constexpr uint32_t MAX_EXP_FOR_FP32 = 0x7f800000;
constexpr uint32_t NUMBER_ZERO = 0x00000000;
constexpr uint32_t NUMBER_TWO_FIVE_FOUR = 0x000000fe;
constexpr uint32_t NUMBER_HALF = 0x00400000;
constexpr uint16_t BF16_ONE = 0x3f80;
constexpr float FP8_E4M3_INV_MAX = 0.002232142857f; // 1 / 448
constexpr float FP8_E5M2_INV_MAX = 0.000017438616f; // 1 / 57344

constexpr AscendC::Reg::CastTrait CAST_TO_FP32_ZERO = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
                                                       AscendC::Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
constexpr AscendC::Reg::CastTrait CAST_TO_FP32_ONE = {AscendC::Reg::RegLayout::ONE, AscendC::Reg::SatMode::UNKNOWN,
                                                      AscendC::Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};
constexpr AscendC::Reg::CastTrait CAST_ABS_TO_FP32 = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::UNKNOWN,
                                                      AscendC::Reg::MaskMergeMode::ZEROING, RoundMode::UNKNOWN};

template <AscendC::RoundMode castRoundMode>
constexpr AscendC::Reg::CastTrait CAST_FP32_TO_Y_ZERO = {AscendC::Reg::RegLayout::ZERO, AscendC::Reg::SatMode::SAT,
                                                         AscendC::Reg::MaskMergeMode::ZEROING, castRoundMode};
template <AscendC::RoundMode castRoundMode>
constexpr AscendC::Reg::CastTrait CAST_FP32_TO_Y_ONE = {AscendC::Reg::RegLayout::ONE, AscendC::Reg::SatMode::SAT,
                                                        AscendC::Reg::MaskMergeMode::ZEROING, castRoundMode};
template <AscendC::RoundMode castRoundMode>
constexpr AscendC::Reg::CastTrait CAST_FP32_TO_Y_TWO = {AscendC::Reg::RegLayout::TWO, AscendC::Reg::SatMode::SAT,
                                                        AscendC::Reg::MaskMergeMode::ZEROING, castRoundMode};
template <AscendC::RoundMode castRoundMode>
constexpr AscendC::Reg::CastTrait CAST_FP32_TO_Y_THREE = {AscendC::Reg::RegLayout::THREE, AscendC::Reg::SatMode::SAT,
                                                          AscendC::Reg::MaskMergeMode::ZEROING, castRoundMode};

static __aicore__ inline int64_t CeilDiv(int64_t value, int64_t factor) { return (value + factor - 1) / factor; }

static __aicore__ inline int64_t Min(int64_t lhs, int64_t rhs) { return (lhs < rhs) ? lhs : rhs; }

static __aicore__ inline int64_t AlignB16CountToUbBlock(int64_t count)
{
    return CeilDiv(count * static_cast<int64_t>(sizeof(uint16_t)), UB_BLOCK_SIZE) *
           (UB_BLOCK_SIZE / static_cast<int64_t>(sizeof(uint16_t)));
}

template <typename XType, typename YType, AscendC::RoundMode roundMode, uint64_t scaleAlg>
class GroupedDynamicMxQuantWithDualAxisBase {
public:
    __aicore__ inline GroupedDynamicMxQuantWithDualAxisBase(
        const GroupedDynamicMxQuantWithDualAxisTilingData* tilingData, TPipe* pipe);

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR groupIndex, GM_ADDR y1, GM_ADDR y1Scale, GM_ADDR y2,
                                GM_ADDR y2Scale);

    __aicore__ inline void Process();

private:
    __aicore__ inline void InitParams();

    __aicore__ inline void ProcessBlock(int64_t groupId, int64_t groupStart, int64_t groupEnd, int64_t rowBlockIdx,
                                        int64_t colBlockIdx, int64_t rowStart, int64_t rowEnd, int64_t colStart,
                                        int64_t colEnd);

    __aicore__ inline void CopyIn(int64_t rowStart, int64_t rowEnd, int64_t colStart, int64_t colEnd);

    __aicore__ inline void ComputeDualAxisMxMaxAndY1(int64_t rowStart, int64_t rowEnd, int64_t colStart, int64_t colEnd,
                                                     LocalTensor<XType>& xLocal);

    __aicore__ inline void ComputeY2(int64_t rowStart, int64_t rowEnd, int64_t colStart, int64_t colEnd,
                                     LocalTensor<XType>& xLocal);

    __aicore__ inline void ComputeDualAxisMxMaxAndY1ByReg(int64_t calcRow, int64_t calcCol, __ubuf__ XType* xAddr,
                                                          __ubuf__ uint16_t* scale1MaxExpAddr,
                                                          __ubuf__ uint16_t* scale2MaxExpAddr,
                                                          __ubuf__ uint8_t* scale1Addr,
                                                          __ubuf__ uint16_t* reciprocalAddr, __ubuf__ uint8_t* y1Addr);

    __aicore__ inline void ComputeScale1MaxAndScale2MaxByReg(uint16_t regRow, uint16_t regCol, __ubuf__ XType* xAddr,
                                                             __ubuf__ uint16_t* scale1MaxExpAddr,
                                                             __ubuf__ uint16_t* scale2MaxExpAddr);

    __aicore__ inline void ComputeScale1ByReg(uint16_t totalMxBlockCount, __ubuf__ uint16_t* scale1MaxExpAddr,
                                              __ubuf__ uint8_t* scale1Addr, __ubuf__ uint16_t* reciprocalAddr);

    __aicore__ inline void ComputeY1DataByReg(uint16_t totalMxBlockCount, __ubuf__ XType* xAddr,
                                              __ubuf__ uint16_t* reciprocalAddr, __ubuf__ uint8_t* y1Addr);

    __aicore__ inline void ComputeScale2ByReg(uint16_t calcCol, __ubuf__ uint16_t* scale2MaxExpAddr,
                                              __ubuf__ uint8_t* packedScale2Addr);

    __aicore__ inline void ComputeScale2ForSlotByReg(
        __ubuf__ uint16_t* maxReadAddr, __ubuf__ uint16_t* reciprocalWriteAddr, RegTensor<uint8_t>& outRegTensor,
        RegTensor<uint32_t>& zeroRegTensor32, RegTensor<uint32_t>& fp32MantissaMaskRegTensor,
        RegTensor<uint32_t>& fp8NanRegTensor32, RegTensor<uint32_t>& nanRegTensor32,
        RegTensor<uint32_t>& biasRegTensor32, MaskReg& dataMask, MaskReg& scaleMask, MaskReg& reciprocalMask);

    __aicore__ inline void ComputeY2DataByReg(uint16_t calcRow, uint16_t calcCol, __ubuf__ XType* xAddr,
                                              __ubuf__ uint16_t* reciprocalAddr, __ubuf__ uint8_t* y2Addr);

    __aicore__ inline void ComputeY2DataFullBlockByReg(__ubuf__ XType* xAddr, __ubuf__ uint16_t* reciprocalAddr,
                                                       __ubuf__ uint8_t* y2Addr);

    __aicore__ inline void ComputeY2DataTailBlockByReg(uint16_t calcRow, uint16_t calcCol, __ubuf__ XType* xAddr,
                                                       __ubuf__ uint16_t* reciprocalAddr, __ubuf__ uint8_t* y2Addr);

    __aicore__ inline void CopyOutScale2(int64_t groupId, int64_t groupStart, int64_t groupEnd, int64_t rowBlockIdx,
                                         int64_t colStart, int64_t calcCol);

    __aicore__ inline void CopyOutY1(int64_t rowStart, int64_t rowEnd, int64_t colStart, int64_t colEnd);

    __aicore__ inline void CopyOutY2(int64_t rowStart, int64_t rowEnd, int64_t colStart, int64_t colEnd);

    __aicore__ inline void CopyOutScale1(int64_t rowStart, int64_t rowEnd, int64_t colStart, int64_t colEnd);

    const GroupedDynamicMxQuantWithDualAxisTilingData* tilingData_{nullptr};
    TPipe* pipe_{nullptr};

    GlobalTensor<XType> xGm_;
    GlobalTensor<int64_t> groupIndexGm_;
    GlobalTensor<uint8_t> y1Gm_;
    GlobalTensor<uint8_t> y1ScaleGm_;
    GlobalTensor<uint8_t> y2Gm_;
    GlobalTensor<uint8_t> y2ScaleGm_;
    TQue<QuePosition::VECIN, QUEUE_DEPTH> inQueue_;
    TQue<QuePosition::VECOUT, QUEUE_DEPTH> y1OutQueue_;
    TQue<QuePosition::VECOUT, QUEUE_DEPTH> y2OutQueue_;
    TQue<QuePosition::VECOUT, QUEUE_DEPTH> y1ScaleOutQueue_;
    TQue<QuePosition::VECOUT, QUEUE_DEPTH> y2ScaleOutQueue_;
    TBuf<TPosition::VECCALC> y1ScaleMaxExpBuf_;
    TBuf<TPosition::VECCALC> y1ScaleReciprocalBuf_;
    TBuf<TPosition::VECCALC> y2ScaleMaxExpBuf_;

    int64_t blockIdx_{0};
    int64_t totalCoreNum_{1};
    int64_t usedCoreNum_{1};
    int64_t m_{0};
    int64_t n_{0};
    int64_t groupNum_{0};
    int64_t rowBlockSize_{0};
    int64_t colBlockSize_{0};
    int64_t colBlocksNum_{0};
    int64_t scale1ColPairs_{0};
    int64_t dstDtype_{DTYPE_FLOAT8_E4M3FN};
    float invDtypeMax_{FP8_E4M3_INV_MAX};
};

template <typename XType, typename YType, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline GroupedDynamicMxQuantWithDualAxisBase<XType, YType, roundMode, scaleAlg>::
    GroupedDynamicMxQuantWithDualAxisBase(const GroupedDynamicMxQuantWithDualAxisTilingData* tilingData, TPipe* pipe)
    : tilingData_(tilingData), pipe_(pipe)
{}

template <typename XType, typename YType, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void GroupedDynamicMxQuantWithDualAxisBase<XType, YType, roundMode, scaleAlg>::Init(
    GM_ADDR x, GM_ADDR groupIndex, GM_ADDR y1, GM_ADDR y1Scale, GM_ADDR y2, GM_ADDR y2Scale)
{
#if (__NPU_ARCH__ == 3510)
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
#endif
    InitParams();

    int64_t scale2UbColSize = (colBlockSize_ < REG_VF_LEN_B16_DOUBLE) ? REG_VF_LEN_B16_DOUBLE : colBlockSize_;
    int64_t scale2StoreLoopCount = CeilDiv(scale2UbColSize, static_cast<int64_t>(REG_VF_LEN_FP32));
    int64_t scale2StoreStride = SCALE2_ROW_SLOT_COUNT * REG_VF_LEN_FP32;
    int64_t scale2FullInterleaveStoreSize = SCALE2_ROW_SLOT_COUNT * REG_VF_LEN_B8;
    // DIST_INTLV_B8 每轮固定写 2 * VL 字节；按最后一轮的起点和完整写入长度申请 scale2 UB
    int64_t scale2UbBufferSize = (scale2StoreLoopCount - 1) * scale2StoreStride + scale2FullInterleaveStoreSize;
    pipe_->InitBuffer(inQueue_, QUEUE_DEPTH, rowBlockSize_ * colBlockSize_ * sizeof(XType));
    pipe_->InitBuffer(y1OutQueue_, QUEUE_DEPTH, rowBlockSize_ * colBlockSize_ * sizeof(uint8_t));
    pipe_->InitBuffer(y2OutQueue_, QUEUE_DEPTH, rowBlockSize_ * colBlockSize_ * sizeof(uint8_t));
    pipe_->InitBuffer(y1ScaleOutQueue_, QUEUE_DEPTH, rowBlockSize_ * (colBlockSize_ / MX_BLOCK_SIZE) * sizeof(uint8_t));
    pipe_->InitBuffer(y2ScaleOutQueue_, QUEUE_DEPTH, scale2UbBufferSize * sizeof(uint8_t));
    pipe_->InitBuffer(y1ScaleMaxExpBuf_, rowBlockSize_ * (colBlockSize_ / MX_BLOCK_SIZE) * sizeof(uint16_t));
    pipe_->InitBuffer(y1ScaleReciprocalBuf_,
                      rowBlockSize_ * AlignB16CountToUbBlock(colBlockSize_ / MX_BLOCK_SIZE) * sizeof(uint16_t));
    pipe_->InitBuffer(y2ScaleMaxExpBuf_, SCALE2_ROW_SLOT_COUNT * scale2UbColSize * sizeof(uint16_t));

    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ XType*>(x));
    groupIndexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(groupIndex));
    y1Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y1));
    y1ScaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y1Scale));
    y2Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y2));
    y2ScaleGm_.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(y2Scale));

    xGm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
    groupIndexGm_.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
}

template <typename XType, typename YType, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void GroupedDynamicMxQuantWithDualAxisBase<XType, YType, roundMode, scaleAlg>::InitParams()
{
    blockIdx_ = AscendC::GetBlockIdx();
    totalCoreNum_ = tilingData_->totalCoreNum;
    usedCoreNum_ = tilingData_->usedCoreNum;
    m_ = tilingData_->m;
    n_ = tilingData_->n;
    groupNum_ = tilingData_->groupNum;
    rowBlockSize_ = tilingData_->rowBlockSize;
    colBlockSize_ = tilingData_->colBlockSize;
    colBlocksNum_ = tilingData_->colBlocksNum;
    scale1ColPairs_ = tilingData_->scale1ColPairs;
    dstDtype_ = tilingData_->dstDtype;
    invDtypeMax_ = (dstDtype_ == DTYPE_FLOAT8_E5M2) ? FP8_E5M2_INV_MAX : FP8_E4M3_INV_MAX;
}

template <typename XType, typename YType, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void GroupedDynamicMxQuantWithDualAxisBase<XType, YType, roundMode, scaleAlg>::Process()
{
    if (usedCoreNum_ <= 0 || blockIdx_ >= usedCoreNum_) {
        return;
    }

    int64_t globalBlockBase = 0;
    for (int64_t groupId = 0; groupId < groupNum_; ++groupId) {
        int64_t groupStart = (groupId > 0) ? groupIndexGm_.GetValue(groupId - 1) : 0;
        int64_t groupEnd = groupIndexGm_.GetValue(groupId);
        int64_t groupRows = groupEnd - groupStart;
        if (groupRows <= 0) {
            continue;
        }

        int64_t rowBlocks = CeilDiv(groupRows, rowBlockSize_);
        int64_t blockCount = rowBlocks * colBlocksNum_;
        int64_t firstBlockOffset = (blockIdx_ - (globalBlockBase % usedCoreNum_) + usedCoreNum_) % usedCoreNum_;
        for (int64_t blockOffset = firstBlockOffset; blockOffset < blockCount; blockOffset += usedCoreNum_) {
            int64_t rowBlockIdx = blockOffset / colBlocksNum_;
            int64_t colBlockIdx = blockOffset % colBlocksNum_;
            int64_t rowStart = groupStart + rowBlockIdx * rowBlockSize_;
            int64_t rowEnd = Min(rowStart + rowBlockSize_, groupEnd);
            int64_t colStart = colBlockIdx * colBlockSize_;
            int64_t colEnd = Min(colStart + colBlockSize_, n_);

            ProcessBlock(groupId, groupStart, groupEnd, rowBlockIdx, colBlockIdx, rowStart, rowEnd, colStart, colEnd);
        }
        globalBlockBase += blockCount;
    }
}

template <typename XType, typename YType, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void GroupedDynamicMxQuantWithDualAxisBase<XType, YType, roundMode, scaleAlg>::ProcessBlock(
    int64_t groupId, int64_t groupStart, int64_t groupEnd, int64_t rowBlockIdx, int64_t colBlockIdx, int64_t rowStart,
    int64_t rowEnd, int64_t colStart, int64_t colEnd)
{
    CopyIn(rowStart, rowEnd, colStart, colEnd);
    LocalTensor<XType> xLocal = inQueue_.template DeQue<XType>();
    int64_t regRowEnd = rowStart + rowBlockSize_;
    ComputeDualAxisMxMaxAndY1(rowStart, regRowEnd, colStart, colEnd, xLocal);
    ComputeY2(rowStart, regRowEnd, colStart, colEnd, xLocal);
    inQueue_.FreeTensor(xLocal);
    CopyOutY1(rowStart, rowEnd, colStart, colEnd);
    CopyOutY2(rowStart, rowEnd, colStart, colEnd);
    CopyOutScale1(rowStart, rowEnd, colStart, colEnd);
    CopyOutScale2(groupId, groupStart, groupEnd, rowBlockIdx, colStart, colEnd - colStart);
}

template <typename XType, typename YType, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void GroupedDynamicMxQuantWithDualAxisBase<XType, YType, roundMode, scaleAlg>::CopyIn(
    int64_t rowStart, int64_t rowEnd, int64_t colStart, int64_t colEnd)
{
    int64_t calcRow = rowEnd - rowStart;
    int64_t calcCol = colEnd - colStart;
    LocalTensor<XType> xLocal = inQueue_.template AllocTensor<XType>();
    if (calcRow < rowBlockSize_) {
        int64_t paddingStart = calcRow * calcCol;
        int64_t paddedElementCount = (rowBlockSize_ - calcRow) * calcCol;
        Duplicate<XType>(xLocal[paddingStart], 0, static_cast<int32_t>(paddedElementCount));
    }

    DataCopyExtParams copyParams = {0, 0, 0, 0, 0};
    DataCopyPadExtParams<XType> padParams = {false, 0, 0, 0};
    copyParams.blockCount = static_cast<uint16_t>(calcRow);
    copyParams.blockLen = static_cast<uint32_t>(calcCol * static_cast<int64_t>(sizeof(XType)));
    copyParams.srcStride = static_cast<uint32_t>((n_ - calcCol) * static_cast<int64_t>(sizeof(XType)));

    int64_t gmOffset = rowStart * n_ + colStart;
    DataCopyPad(xLocal, xGm_[gmOffset], copyParams, padParams);
    inQueue_.template EnQue(xLocal);
}

template <typename XType, typename YType, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void
GroupedDynamicMxQuantWithDualAxisBase<XType, YType, roundMode, scaleAlg>::ComputeDualAxisMxMaxAndY1(
    int64_t rowStart, int64_t rowEnd, int64_t colStart, int64_t colEnd, LocalTensor<XType>& xLocal)
{
    int64_t calcRow = rowEnd - rowStart;
    int64_t calcCol = colEnd - colStart;
    LocalTensor<uint8_t> y1Local = y1OutQueue_.template AllocTensor<uint8_t>();
    LocalTensor<uint8_t> scale1Local = y1ScaleOutQueue_.template AllocTensor<uint8_t>();
    LocalTensor<uint16_t> maxExpLocal = y1ScaleMaxExpBuf_.template Get<uint16_t>();
    LocalTensor<uint16_t> scale2MaxExpLocal = y2ScaleMaxExpBuf_.template Get<uint16_t>();
    LocalTensor<uint16_t> reciprocalLocal = y1ScaleReciprocalBuf_.template Get<uint16_t>();

    auto xAddr = reinterpret_cast<__ubuf__ XType*>(xLocal.GetPhyAddr());
    auto scale1MaxExpAddr = reinterpret_cast<__ubuf__ uint16_t*>(maxExpLocal.GetPhyAddr());
    auto scale2MaxExpAddr = reinterpret_cast<__ubuf__ uint16_t*>(scale2MaxExpLocal.GetPhyAddr());
    auto scale1Addr = reinterpret_cast<__ubuf__ uint8_t*>(scale1Local.GetPhyAddr());
    auto reciprocalAddr = reinterpret_cast<__ubuf__ uint16_t*>(reciprocalLocal.GetPhyAddr());
    auto y1Addr = reinterpret_cast<__ubuf__ uint8_t*>(y1Local.GetPhyAddr());
    ComputeDualAxisMxMaxAndY1ByReg(calcRow, calcCol, xAddr, scale1MaxExpAddr, scale2MaxExpAddr, scale1Addr,
                                   reciprocalAddr, y1Addr);

    y1ScaleOutQueue_.template EnQue(scale1Local);
    y1OutQueue_.template EnQue(y1Local);
}

template <typename XType, typename YType, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void GroupedDynamicMxQuantWithDualAxisBase<XType, YType, roundMode, scaleAlg>::ComputeY2(
    int64_t rowStart, int64_t rowEnd, int64_t colStart, int64_t colEnd, LocalTensor<XType>& xLocal)
{
    int64_t calcRow = rowEnd - rowStart;
    int64_t calcCol = colEnd - colStart;
    LocalTensor<uint8_t> y2Local = y2OutQueue_.template AllocTensor<uint8_t>();
    LocalTensor<uint8_t> scale2Local = y2ScaleOutQueue_.template AllocTensor<uint8_t>();
    LocalTensor<uint16_t> scale2MaxExpLocal = y2ScaleMaxExpBuf_.template Get<uint16_t>();

    auto xAddr = reinterpret_cast<__ubuf__ XType*>(xLocal.GetPhyAddr());
    auto scale2Addr = reinterpret_cast<__ubuf__ uint8_t*>(scale2Local.GetPhyAddr());
    auto scale2MaxExpAddr = reinterpret_cast<__ubuf__ uint16_t*>(scale2MaxExpLocal.GetPhyAddr());
    auto y2Addr = reinterpret_cast<__ubuf__ uint8_t*>(y2Local.GetPhyAddr());
    uint16_t calcRows = static_cast<uint16_t>(calcRow);
    uint16_t calcCols = static_cast<uint16_t>(calcCol);
    ComputeScale2ByReg(calcCols, scale2MaxExpAddr, scale2Addr);
    auto scale2ReciprocalAddr = scale2MaxExpAddr;
    ComputeY2DataByReg(calcRows, calcCols, xAddr, scale2ReciprocalAddr, y2Addr);

    y2ScaleOutQueue_.template EnQue(scale2Local);
    y2OutQueue_.template EnQue(y2Local);
}

template <typename XType, typename YType, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void
GroupedDynamicMxQuantWithDualAxisBase<XType, YType, roundMode, scaleAlg>::ComputeDualAxisMxMaxAndY1ByReg(
    int64_t calcRow, int64_t calcCol, __ubuf__ XType* xAddr, __ubuf__ uint16_t* scale1MaxExpAddr,
    __ubuf__ uint16_t* scale2MaxExpAddr, __ubuf__ uint8_t* scale1Addr, __ubuf__ uint16_t* reciprocalAddr,
    __ubuf__ uint8_t* y1Addr)
{
    uint16_t regRow = static_cast<uint16_t>(calcRow);
    uint16_t regCol = static_cast<uint16_t>(calcCol);
    uint16_t scaleCount = static_cast<uint16_t>(regCol / MX_BLOCK_SIZE);
    uint16_t totalMxBlockCount = static_cast<uint16_t>(regRow * scaleCount);
    ComputeScale1MaxAndScale2MaxByReg(regRow, regCol, xAddr, scale1MaxExpAddr, scale2MaxExpAddr);
    ComputeScale1ByReg(totalMxBlockCount, scale1MaxExpAddr, scale1Addr, reciprocalAddr);
    ComputeY1DataByReg(totalMxBlockCount, xAddr, reciprocalAddr, y1Addr);
}

template <typename XType, typename YType, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void
GroupedDynamicMxQuantWithDualAxisBase<XType, YType, roundMode, scaleAlg>::ComputeScale1MaxAndScale2MaxByReg(
    uint16_t regRow, uint16_t regCol, __ubuf__ XType* xAddr, __ubuf__ uint16_t* scale1MaxExpAddr,
    __ubuf__ uint16_t* scale2MaxExpAddr)
{
    uint16_t scaleCount = static_cast<uint16_t>(regCol / MX_BLOCK_SIZE);
    __VEC_SCOPE__
    {
        RegTensor<XType> x0;
        RegTensor<XType> x1;
        RegTensor<uint16_t> x0Abs;
        RegTensor<uint16_t> x1Abs;
        RegTensor<uint16_t> y1MaxExp;
        RegTensor<uint16_t> scale2Slot0Part0;
        RegTensor<uint16_t> scale2Slot0Part1;
        RegTensor<uint16_t> scale2Slot1Part0;
        RegTensor<uint16_t> scale2Slot1Part1;
        RegTensor<uint16_t> absMaskRegTensor;
        AscendC::Reg::UnalignReg maxExpUreg;
        uint32_t dataMaskLen = static_cast<uint32_t>(regCol / DIGIT_TWO);
        MaskReg dataMask = AscendC::Reg::UpdateMask<uint16_t>(dataMaskLen);
        __ubuf__ XType* xMaxCursor = xAddr;
        __ubuf__ uint16_t* scale1MaxCursor = scale1MaxExpAddr;

        AscendC::Reg::Duplicate(absMaskRegTensor, ABS_FOR_UINT16);
        AscendC::Reg::Duplicate(scale2Slot0Part0, 0);
        AscendC::Reg::Duplicate(scale2Slot0Part1, 0);
        AscendC::Reg::Duplicate(scale2Slot1Part0, 0);
        AscendC::Reg::Duplicate(scale2Slot1Part1, 0);

        for (uint16_t row = 0; row < MX_BLOCK_SIZE_U16; ++row) {
            AscendC::Reg::DataCopy<XType, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                   AscendC::Reg::LoadDist::DIST_DINTLV_B16>(x0, x1, xMaxCursor, regCol);
            AscendC::Reg::And(x0Abs, (RegTensor<uint16_t>&)x0, absMaskRegTensor, dataMask);
            AscendC::Reg::And(x1Abs, (RegTensor<uint16_t>&)x1, absMaskRegTensor, dataMask);
            AscendC::Reg::Max(y1MaxExp, x0Abs, x1Abs, dataMask);
            AscendC::Reg::ReduceMaxWithDataBlock(y1MaxExp, y1MaxExp, dataMask);
            AscendC::Reg::StoreUnAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(scale1MaxCursor, y1MaxExp,
                                                                                              maxExpUreg, scaleCount);
            AscendC::Reg::Max(scale2Slot0Part0, scale2Slot0Part0, x0Abs, dataMask);
            AscendC::Reg::Max(scale2Slot0Part1, scale2Slot0Part1, x1Abs, dataMask);
        }
        for (uint16_t row = MX_BLOCK_SIZE_U16; row < regRow; ++row) {
            AscendC::Reg::DataCopy<XType, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                   AscendC::Reg::LoadDist::DIST_DINTLV_B16>(x0, x1, xMaxCursor, regCol);
            AscendC::Reg::And(x0Abs, (RegTensor<uint16_t>&)x0, absMaskRegTensor, dataMask);
            AscendC::Reg::And(x1Abs, (RegTensor<uint16_t>&)x1, absMaskRegTensor, dataMask);
            AscendC::Reg::Max(y1MaxExp, x0Abs, x1Abs, dataMask);
            AscendC::Reg::ReduceMaxWithDataBlock(y1MaxExp, y1MaxExp, dataMask);
            AscendC::Reg::StoreUnAlign<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(scale1MaxCursor, y1MaxExp,
                                                                                              maxExpUreg, scaleCount);
            AscendC::Reg::Max(scale2Slot1Part0, scale2Slot1Part0, x0Abs, dataMask);
            AscendC::Reg::Max(scale2Slot1Part1, scale2Slot1Part1, x1Abs, dataMask);
        }
        AscendC::Reg::StoreUnAlignPost(scale1MaxCursor, maxExpUreg, 0);
        AscendC::Reg::DataCopy<uint16_t, AscendC::Reg::StoreDist::DIST_INTLV_B16>(scale2MaxExpAddr, scale2Slot0Part0,
                                                                                  scale2Slot0Part1, dataMask);
        AscendC::Reg::DataCopy<uint16_t, AscendC::Reg::StoreDist::DIST_INTLV_B16>(
            scale2MaxExpAddr + regCol, scale2Slot1Part0, scale2Slot1Part1, dataMask);
    }
}

template <typename XType, typename YType, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void GroupedDynamicMxQuantWithDualAxisBase<XType, YType, roundMode, scaleAlg>::ComputeScale1ByReg(
    uint16_t totalMxBlockCount, __ubuf__ uint16_t* scale1MaxExpAddr, __ubuf__ uint8_t* scale1Addr,
    __ubuf__ uint16_t* reciprocalAddr)
{
    uint16_t scaleLoopNum = static_cast<uint16_t>(CeilDiv(totalMxBlockCount, REG_VF_LEN_FP32));
    __VEC_SCOPE__
    {
        RegTensor<uint16_t> maxAbsRegTensor;
        RegTensor<uint32_t> zeroRegTensor32;
        RegTensor<uint32_t> fp32MantissaMaskRegTensor;
        RegTensor<uint32_t> fp8NanRegTensor32;
        RegTensor<uint32_t> nanRegTensor32;
        RegTensor<uint32_t> biasRegTensor32;
        RegTensor<uint16_t> scalePackedU16RegTensor;
        RegTensor<uint32_t> scaleRegTensor;
        RegTensor<uint32_t> expRegTensor;
        RegTensor<uint32_t> mantissaRegTensor;
        RegTensor<uint8_t> outRegTensor;
        MaskReg isFiniteMask;
        MaskReg isNonZeroMask;
        MaskReg p0;
        MaskReg p1;
        uint32_t dataMaskLen = REG_VF_LEN_FP32;
        uint32_t outMaskLen = REG_VF_LEN_FP32;
        uint32_t reciprocalMaskLen = REG_VF_LEN_FP32;
        MaskReg dataMask = AscendC::Reg::UpdateMask<float>(dataMaskLen);
        MaskReg outMask = AscendC::Reg::UpdateMask<uint8_t>(outMaskLen);
        MaskReg reciprocalMask = AscendC::Reg::UpdateMask<uint16_t>(reciprocalMaskLen);
        __ubuf__ uint16_t* scale1MaxReadCursor = scale1MaxExpAddr;
        AscendC::Reg::Duplicate(zeroRegTensor32, NUMBER_ZERO);
        AscendC::Reg::Duplicate(fp32MantissaMaskRegTensor, MAN_FOR_FP32);
        AscendC::Reg::Duplicate(fp8NanRegTensor32, MAX_EXP_FOR_FP8_IN_FP32);
        AscendC::Reg::Duplicate(nanRegTensor32, NAN_CUSTOMIZATION_PACK);
        AscendC::Reg::Duplicate(biasRegTensor32, FP32_EXP_BIAS_CEIL_ALG);

        for (uint16_t i = 0; i < scaleLoopNum; ++i) {
            AscendC::Reg::DataCopy<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                   AscendC::Reg::LoadDist::DIST_UNPACK_B16>(maxAbsRegTensor, scale1MaxReadCursor,
                                                                            REG_VF_LEN_FP32);
            AscendC::Reg::Cast<float, XType, CAST_ABS_TO_FP32>((RegTensor<float>&)scaleRegTensor,
                                                               (RegTensor<XType>&)maxAbsRegTensor, dataMask);
            AscendC::Reg::CompareScalar<uint32_t, CMPMODE::LT>(isFiniteMask, scaleRegTensor, MAX_EXP_FOR_FP32,
                                                               dataMask);
            AscendC::Reg::Compare<uint32_t, CMPMODE::NE>(isNonZeroMask, scaleRegTensor, zeroRegTensor32, dataMask);
            AscendC::Reg::Muls((RegTensor<float>&)scaleRegTensor, (RegTensor<float>&)scaleRegTensor, invDtypeMax_,
                               dataMask);
            AscendC::Reg::ShiftRights(expRegTensor, scaleRegTensor, SHR_NUM_FOR_FP32, dataMask);
            AscendC::Reg::And(mantissaRegTensor, scaleRegTensor, fp32MantissaMaskRegTensor, dataMask);
            AscendC::Reg::CompareScalar<uint32_t, CMPMODE::GT>(p0, expRegTensor, NUMBER_ZERO, dataMask);
            AscendC::Reg::CompareScalar<uint32_t, CMPMODE::LT>(p0, expRegTensor, NUMBER_TWO_FIVE_FOUR, p0);
            AscendC::Reg::CompareScalar<uint32_t, CMPMODE::GT>(p0, mantissaRegTensor, NUMBER_ZERO, p0);
            AscendC::Reg::CompareScalar<uint32_t, CMPMODE::EQ>(p1, expRegTensor, NUMBER_ZERO, dataMask);
            AscendC::Reg::CompareScalar<uint32_t, CMPMODE::GT>(p1, mantissaRegTensor, NUMBER_HALF, p1);
            AscendC::Reg::MaskOr(p0, p0, p1, dataMask);
            AscendC::Reg::Adds(scaleRegTensor, expRegTensor, 1, dataMask);
            AscendC::Reg::Select(scaleRegTensor, scaleRegTensor, expRegTensor, p0);
            AscendC::Reg::Select<uint32_t>(scaleRegTensor, scaleRegTensor, fp8NanRegTensor32, isFiniteMask);
            AscendC::Reg::Select<uint32_t>(scaleRegTensor, scaleRegTensor, zeroRegTensor32, isNonZeroMask);
            AscendC::Reg::Pack<uint16_t, uint32_t, AscendC::Reg::HighLowPart::LOWEST>(scalePackedU16RegTensor,
                                                                                      scaleRegTensor);
            AscendC::Reg::Pack<uint8_t, uint16_t, AscendC::Reg::HighLowPart::LOWEST>(outRegTensor,
                                                                                     scalePackedU16RegTensor);
            AscendC::Reg::DataCopy<uint8_t, AscendC::Reg::StoreDist::DIST_NORM>(scale1Addr + i * REG_VF_LEN_FP32,
                                                                                outRegTensor, outMask);

            AscendC::Reg::ShiftLefts(scaleRegTensor, scaleRegTensor, SHR_NUM_FOR_BF16, dataMask);
            AscendC::Reg::Sub(expRegTensor, biasRegTensor32, scaleRegTensor, dataMask);
            AscendC::Reg::Select<uint32_t>(expRegTensor, expRegTensor, nanRegTensor32, isFiniteMask);
            AscendC::Reg::Select<uint32_t>(expRegTensor, expRegTensor, zeroRegTensor32, isNonZeroMask);
            AscendC::Reg::Pack<uint16_t, uint32_t, AscendC::Reg::HighLowPart::LOWEST>(maxAbsRegTensor, expRegTensor);
            AscendC::Reg::DataCopy<uint16_t, AscendC::Reg::StoreDist::DIST_NORM>(reciprocalAddr + i * REG_VF_LEN_FP32,
                                                                                 maxAbsRegTensor, reciprocalMask);
        }
    }
}

template <typename XType, typename YType, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void GroupedDynamicMxQuantWithDualAxisBase<XType, YType, roundMode, scaleAlg>::ComputeY1DataByReg(
    uint16_t totalMxBlockCount, __ubuf__ XType* xAddr, __ubuf__ uint16_t* reciprocalAddr, __ubuf__ uint8_t* y1Addr)
{
    __VEC_SCOPE__
    {
        RegTensor<XType> x0;
        RegTensor<XType> x1;
        RegTensor<uint16_t> zeroU16;
        RegTensor<uint16_t> scaleForMulRegTensor;
        RegTensor<float> reciprocalFp32;
        RegTensor<float> xEvenFp32Layout0;
        RegTensor<float> xEvenFp32Layout1;
        RegTensor<float> xOddFp32Layout0;
        RegTensor<float> xOddFp32Layout1;
        RegTensor<YType> yEvenFp8Layout0;
        RegTensor<YType> yEvenFp8Layout2;
        RegTensor<YType> yOddFp8Layout1;
        RegTensor<YType> yOddFp8Layout3;
        MaskReg maskAllB16 = AscendC::Reg::CreateMask<uint16_t, AscendC::Reg::MaskPattern::ALL>();
        MaskReg maskB32 = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();
        MaskReg validScaleMask;
        AscendC::Reg::Duplicate(zeroU16, static_cast<uint16_t>(0));
        __ubuf__ XType* xDataCursor = xAddr;
        __ubuf__ uint8_t* yCursor = y1Addr;
        uint16_t dataLoopNum = static_cast<uint16_t>(CeilDiv(totalMxBlockCount, ELEMENT_AFTER_REDUCE));
        for (uint16_t i = 0; i < dataLoopNum; ++i) {
            uint16_t processedBlocks = static_cast<uint16_t>(i * ELEMENT_AFTER_REDUCE);
            uint16_t chunkBlocks = static_cast<uint16_t>(totalMxBlockCount - processedBlocks);
            if (chunkBlocks > ELEMENT_AFTER_REDUCE) {
                chunkBlocks = ELEMENT_AFTER_REDUCE;
            }
            uint16_t chunkElements = static_cast<uint16_t>(chunkBlocks * MX_BLOCK_SIZE);
            uint32_t xMaskLen = static_cast<uint32_t>(chunkElements / DIGIT_TWO);
            MaskReg xMask = AscendC::Reg::UpdateMask<XType>(xMaskLen);
            uint32_t yMaskLen = chunkElements;
            MaskReg yMask = AscendC::Reg::UpdateMask<uint8_t>(yMaskLen);
            __ubuf__ uint16_t* reciprocalDataCursor = reciprocalAddr + processedBlocks;

            AscendC::Reg::DataCopy<XType, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                   AscendC::Reg::LoadDist::DIST_DINTLV_B16>(x0, x1, xDataCursor, chunkElements);
            AscendC::Reg::DataCopy<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                   AscendC::Reg::LoadDist::DIST_E2B_B16>(scaleForMulRegTensor, reciprocalDataCursor,
                                                                         chunkBlocks);

            AscendC::Reg::Compare<uint16_t, CMPMODE::NE>(validScaleMask, scaleForMulRegTensor, zeroU16, xMask);
            if constexpr (IsSameType<XType, half>::value) {
                AscendC::Reg::Cast<float, bfloat16_t, CAST_TO_FP32_ZERO>(
                    reciprocalFp32, (RegTensor<bfloat16_t>&)scaleForMulRegTensor, maskAllB16);
                AscendC::Reg::Cast<float, XType, CAST_TO_FP32_ZERO>(xEvenFp32Layout0, x0, validScaleMask);
                AscendC::Reg::Cast<float, XType, CAST_TO_FP32_ONE>(xEvenFp32Layout1, x0, validScaleMask);
                AscendC::Reg::Cast<float, XType, CAST_TO_FP32_ZERO>(xOddFp32Layout0, x1, validScaleMask);
                AscendC::Reg::Cast<float, XType, CAST_TO_FP32_ONE>(xOddFp32Layout1, x1, validScaleMask);
                AscendC::Reg::Mul(xEvenFp32Layout0, xEvenFp32Layout0, reciprocalFp32, maskB32);
                AscendC::Reg::Mul(xEvenFp32Layout1, xEvenFp32Layout1, reciprocalFp32, maskB32);
                AscendC::Reg::Mul(xOddFp32Layout0, xOddFp32Layout0, reciprocalFp32, maskB32);
                AscendC::Reg::Mul(xOddFp32Layout1, xOddFp32Layout1, reciprocalFp32, maskB32);
            } else {
                AscendC::Reg::Mul(x0, x0, (RegTensor<XType>&)scaleForMulRegTensor, validScaleMask);
                AscendC::Reg::Mul(x1, x1, (RegTensor<XType>&)scaleForMulRegTensor, validScaleMask);
                AscendC::Reg::Cast<float, XType, CAST_TO_FP32_ZERO>(xEvenFp32Layout0, x0, validScaleMask);
                AscendC::Reg::Cast<float, XType, CAST_TO_FP32_ONE>(xEvenFp32Layout1, x0, validScaleMask);
                AscendC::Reg::Cast<float, XType, CAST_TO_FP32_ZERO>(xOddFp32Layout0, x1, validScaleMask);
                AscendC::Reg::Cast<float, XType, CAST_TO_FP32_ONE>(xOddFp32Layout1, x1, validScaleMask);
            }

            AscendC::Reg::Cast<YType, float, CAST_FP32_TO_Y_ZERO<roundMode>>(yEvenFp8Layout0, xEvenFp32Layout0,
                                                                             maskB32);
            AscendC::Reg::Cast<YType, float, CAST_FP32_TO_Y_TWO<roundMode>>(yEvenFp8Layout2, xEvenFp32Layout1, maskB32);
            AscendC::Reg::Cast<YType, float, CAST_FP32_TO_Y_ONE<roundMode>>(yOddFp8Layout1, xOddFp32Layout0, maskB32);
            AscendC::Reg::Cast<YType, float, CAST_FP32_TO_Y_THREE<roundMode>>(yOddFp8Layout3, xOddFp32Layout1, maskB32);
            AscendC::Reg::Add((RegTensor<uint8_t>&)yEvenFp8Layout0, (RegTensor<uint8_t>&)yEvenFp8Layout0,
                              (RegTensor<uint8_t>&)yEvenFp8Layout2, yMask);
            AscendC::Reg::Add((RegTensor<uint8_t>&)yOddFp8Layout1, (RegTensor<uint8_t>&)yOddFp8Layout1,
                              (RegTensor<uint8_t>&)yOddFp8Layout3, yMask);
            AscendC::Reg::Add((RegTensor<uint8_t>&)yEvenFp8Layout0, (RegTensor<uint8_t>&)yEvenFp8Layout0,
                              (RegTensor<uint8_t>&)yOddFp8Layout1, yMask);
            AscendC::Reg::DataCopy<uint8_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                   AscendC::Reg::StoreDist::DIST_NORM_B8>(yCursor, (RegTensor<uint8_t>&)yEvenFp8Layout0,
                                                                          chunkElements, yMask);
        }
    }
}

template <typename XType, typename YType, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void
GroupedDynamicMxQuantWithDualAxisBase<XType, YType, roundMode, scaleAlg>::ComputeScale2ForSlotByReg(
    __ubuf__ uint16_t* maxReadAddr, __ubuf__ uint16_t* reciprocalWriteAddr, RegTensor<uint8_t>& outRegTensor,
    RegTensor<uint32_t>& zeroRegTensor32, RegTensor<uint32_t>& fp32MantissaMaskRegTensor,
    RegTensor<uint32_t>& fp8NanRegTensor32, RegTensor<uint32_t>& nanRegTensor32, RegTensor<uint32_t>& biasRegTensor32,
    MaskReg& dataMask, MaskReg& scaleMask, MaskReg& reciprocalMask)
{
    RegTensor<uint16_t> maxAbsRegTensor;
    RegTensor<uint16_t> scalePackedU16RegTensor;
    RegTensor<uint32_t> scaleRegTensor;
    RegTensor<uint32_t> expRegTensor;
    RegTensor<uint32_t> mantissaRegTensor;
    MaskReg isFiniteMask;
    MaskReg isNonZeroMask;
    MaskReg p0;
    MaskReg p1;
    AscendC::Reg::DataCopy<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                           AscendC::Reg::LoadDist::DIST_UNPACK_B16>(maxAbsRegTensor, maxReadAddr, REG_VF_LEN_FP32);
    AscendC::Reg::Cast<float, XType, CAST_ABS_TO_FP32>((RegTensor<float>&)scaleRegTensor,
                                                       (RegTensor<XType>&)maxAbsRegTensor, dataMask);
    AscendC::Reg::CompareScalar<uint32_t, CMPMODE::LT>(isFiniteMask, scaleRegTensor, MAX_EXP_FOR_FP32, scaleMask);
    AscendC::Reg::Compare<uint32_t, CMPMODE::NE>(isNonZeroMask, scaleRegTensor, zeroRegTensor32, scaleMask);
    AscendC::Reg::Muls((RegTensor<float>&)scaleRegTensor, (RegTensor<float>&)scaleRegTensor, invDtypeMax_, scaleMask);
    AscendC::Reg::ShiftRights(expRegTensor, scaleRegTensor, SHR_NUM_FOR_FP32, scaleMask);
    AscendC::Reg::And(mantissaRegTensor, scaleRegTensor, fp32MantissaMaskRegTensor, scaleMask);
    AscendC::Reg::CompareScalar<uint32_t, CMPMODE::GT>(p0, expRegTensor, NUMBER_ZERO, scaleMask);
    AscendC::Reg::CompareScalar<uint32_t, CMPMODE::LT>(p0, expRegTensor, NUMBER_TWO_FIVE_FOUR, p0);
    AscendC::Reg::CompareScalar<uint32_t, CMPMODE::GT>(p0, mantissaRegTensor, NUMBER_ZERO, p0);
    AscendC::Reg::CompareScalar<uint32_t, CMPMODE::EQ>(p1, expRegTensor, NUMBER_ZERO, scaleMask);
    AscendC::Reg::CompareScalar<uint32_t, CMPMODE::GT>(p1, mantissaRegTensor, NUMBER_HALF, p1);
    AscendC::Reg::MaskOr(p0, p0, p1, scaleMask);
    AscendC::Reg::Adds(scaleRegTensor, expRegTensor, 1, scaleMask);
    AscendC::Reg::Select(scaleRegTensor, scaleRegTensor, expRegTensor, p0);
    AscendC::Reg::Select<uint32_t>(scaleRegTensor, scaleRegTensor, fp8NanRegTensor32, isFiniteMask);
    AscendC::Reg::Select<uint32_t>(scaleRegTensor, scaleRegTensor, zeroRegTensor32, isNonZeroMask);
    AscendC::Reg::Pack<uint16_t, uint32_t, AscendC::Reg::HighLowPart::LOWEST>(scalePackedU16RegTensor, scaleRegTensor);
    AscendC::Reg::Pack<uint8_t, uint16_t, AscendC::Reg::HighLowPart::LOWEST>(outRegTensor, scalePackedU16RegTensor);
    AscendC::Reg::ShiftLefts(scaleRegTensor, scaleRegTensor, SHR_NUM_FOR_BF16, scaleMask);
    AscendC::Reg::Sub(expRegTensor, biasRegTensor32, scaleRegTensor, scaleMask);
    AscendC::Reg::Select<uint32_t>(expRegTensor, expRegTensor, nanRegTensor32, isFiniteMask);
    AscendC::Reg::Select<uint32_t>(expRegTensor, expRegTensor, zeroRegTensor32, isNonZeroMask);
    AscendC::Reg::Pack<uint16_t, uint32_t, AscendC::Reg::HighLowPart::LOWEST>(maxAbsRegTensor, expRegTensor);
    AscendC::Reg::DataCopy<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE>(reciprocalWriteAddr, maxAbsRegTensor,
                                                                                  REG_VF_LEN_FP32, reciprocalMask);
}

template <typename XType, typename YType, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void GroupedDynamicMxQuantWithDualAxisBase<XType, YType, roundMode, scaleAlg>::ComputeScale2ByReg(
    uint16_t calcCol, __ubuf__ uint16_t* scale2MaxExpAddr, __ubuf__ uint8_t* packedScale2Addr)
{
    uint16_t colLoopCount = static_cast<uint16_t>(CeilDiv(static_cast<int64_t>(calcCol), REG_VF_LEN_FP32));
    __VEC_SCOPE__
    {
        RegTensor<uint32_t> zeroRegTensor32;
        RegTensor<uint32_t> fp32MantissaMaskRegTensor;
        RegTensor<uint32_t> fp8NanRegTensor32;
        RegTensor<uint32_t> nanRegTensor32;
        RegTensor<uint32_t> biasRegTensor32;
        RegTensor<uint8_t> outRegTensor0;
        RegTensor<uint8_t> outRegTensor1;
        MaskReg dataMask = AscendC::Reg::CreateMask<XType, AscendC::Reg::MaskPattern::ALL>();
        MaskReg scaleMask = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();
        MaskReg interleaveMask = AscendC::Reg::CreateMask<uint8_t, AscendC::Reg::MaskPattern::ALL>();
        MaskReg reciprocalMask = AscendC::Reg::CreateMask<uint16_t, AscendC::Reg::MaskPattern::VL64>();

        AscendC::Reg::Duplicate(zeroRegTensor32, NUMBER_ZERO);
        AscendC::Reg::Duplicate(fp32MantissaMaskRegTensor, MAN_FOR_FP32);
        AscendC::Reg::Duplicate(fp8NanRegTensor32, MAX_EXP_FOR_FP8_IN_FP32);
        AscendC::Reg::Duplicate(nanRegTensor32, NAN_CUSTOMIZATION_PACK);
        AscendC::Reg::Duplicate(biasRegTensor32, FP32_EXP_BIAS_CEIL_ALG);

        for (uint16_t colBlockIdx = 0; colBlockIdx < colLoopCount; ++colBlockIdx) {
            uint16_t colOffset = static_cast<uint16_t>(colBlockIdx * REG_VF_LEN_FP32);
            __ubuf__ uint16_t* maxReadAddr0 = scale2MaxExpAddr + colOffset;
            __ubuf__ uint16_t* reciprocalWriteAddr0 = scale2MaxExpAddr + colOffset;
            __ubuf__ uint16_t* maxReadAddr1 = scale2MaxExpAddr + calcCol + colOffset;
            __ubuf__ uint16_t* reciprocalWriteAddr1 = scale2MaxExpAddr + calcCol + colOffset;

            ComputeScale2ForSlotByReg(maxReadAddr0, reciprocalWriteAddr0, outRegTensor0, zeroRegTensor32,
                                      fp32MantissaMaskRegTensor, fp8NanRegTensor32, nanRegTensor32, biasRegTensor32,
                                      dataMask, scaleMask, reciprocalMask);
            ComputeScale2ForSlotByReg(maxReadAddr1, reciprocalWriteAddr1, outRegTensor1, zeroRegTensor32,
                                      fp32MantissaMaskRegTensor, fp8NanRegTensor32, nanRegTensor32, biasRegTensor32,
                                      dataMask, scaleMask, reciprocalMask);

            AscendC::Reg::DataCopy<uint8_t, AscendC::Reg::StoreDist::DIST_INTLV_B8>(
                packedScale2Addr + SCALE2_ROW_SLOT_COUNT * colOffset, outRegTensor0, outRegTensor1, interleaveMask);
        }
    }
}
template <typename XType, typename YType, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void GroupedDynamicMxQuantWithDualAxisBase<XType, YType, roundMode, scaleAlg>::ComputeY2DataByReg(
    uint16_t calcRow, uint16_t calcCol, __ubuf__ XType* xAddr, __ubuf__ uint16_t* reciprocalAddr,
    __ubuf__ uint8_t* y2Addr)
{
    if (calcRow == rowBlockSize_ && calcCol == static_cast<uint16_t>(colBlockSize_)) {
        ComputeY2DataFullBlockByReg(xAddr, reciprocalAddr, y2Addr);
        return;
    }
    ComputeY2DataTailBlockByReg(calcRow, calcCol, xAddr, reciprocalAddr, y2Addr);
}

template <typename XType, typename YType, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void
GroupedDynamicMxQuantWithDualAxisBase<XType, YType, roundMode, scaleAlg>::ComputeY2DataFullBlockByReg(
    __ubuf__ XType* xAddr, __ubuf__ uint16_t* reciprocalAddr, __ubuf__ uint8_t* y2Addr)
{
    __VEC_SCOPE__
    {
        RegTensor<XType> xEven;
        RegTensor<XType> xOdd;
        RegTensor<uint16_t> reciprocalEven;
        RegTensor<uint16_t> reciprocalOdd;
        RegTensor<float> reciprocalEvenFp32Layout0;
        RegTensor<float> reciprocalEvenFp32Layout1;
        RegTensor<float> reciprocalOddFp32Layout0;
        RegTensor<float> reciprocalOddFp32Layout1;
        RegTensor<float> xEvenFp32Layout0;
        RegTensor<float> xEvenFp32Layout1;
        RegTensor<float> xOddFp32Layout0;
        RegTensor<float> xOddFp32Layout1;
        RegTensor<float> zeroFp32;
        RegTensor<YType> yEvenFp8Layout0;
        RegTensor<YType> yEvenFp8Layout2;
        RegTensor<YType> yOddFp8Layout1;
        RegTensor<YType> yOddFp8Layout3;
        MaskReg maskX = AscendC::Reg::CreateMask<XType>();
        MaskReg maskAllB16 = AscendC::Reg::CreateMask<uint16_t, AscendC::Reg::MaskPattern::ALL>();
        MaskReg maskY = AscendC::Reg::CreateMask<YType>();
        MaskReg maskAllB8 = AscendC::Reg::CreateMask<uint8_t, AscendC::Reg::MaskPattern::ALL>();
        MaskReg maskB32 = AscendC::Reg::CreateMask<uint32_t, AscendC::Reg::MaskPattern::ALL>();
        MaskReg validScaleEven0;
        MaskReg validScaleEven1;
        MaskReg validScaleOdd0;
        MaskReg validScaleOdd1;
        AscendC::Reg::Duplicate(zeroFp32, 0.0f);

        for (uint16_t scaleSlot = 0; scaleSlot < SCALE2_ROW_SLOT_COUNT; ++scaleSlot) {
            uint16_t localRowStart = static_cast<uint16_t>(scaleSlot * MX_BLOCK_SIZE);
            uint16_t localRowEnd = static_cast<uint16_t>(localRowStart + static_cast<uint16_t>(MX_BLOCK_SIZE));
            __ubuf__ uint16_t* reciprocalCursor = reciprocalAddr + scaleSlot * REG_VF_LEN_B16_DOUBLE;
            AscendC::Reg::DataCopy<uint16_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                   AscendC::Reg::LoadDist::DIST_DINTLV_B16>(reciprocalEven, reciprocalOdd,
                                                                            reciprocalCursor, REG_VF_LEN_B16_DOUBLE);
            AscendC::Reg::Cast<float, bfloat16_t, CAST_TO_FP32_ZERO>(
                reciprocalEvenFp32Layout0, (RegTensor<bfloat16_t>&)reciprocalEven, maskAllB16);
            AscendC::Reg::Cast<float, bfloat16_t, CAST_TO_FP32_ONE>(reciprocalEvenFp32Layout1,
                                                                    (RegTensor<bfloat16_t>&)reciprocalEven, maskAllB16);
            AscendC::Reg::Cast<float, bfloat16_t, CAST_TO_FP32_ZERO>(reciprocalOddFp32Layout0,
                                                                     (RegTensor<bfloat16_t>&)reciprocalOdd, maskAllB16);
            AscendC::Reg::Cast<float, bfloat16_t, CAST_TO_FP32_ONE>(reciprocalOddFp32Layout1,
                                                                    (RegTensor<bfloat16_t>&)reciprocalOdd, maskAllB16);
            AscendC::Reg::Compare<float, CMPMODE::NE>(validScaleEven0, reciprocalEvenFp32Layout0, zeroFp32, maskB32);
            AscendC::Reg::Compare<float, CMPMODE::NE>(validScaleEven1, reciprocalEvenFp32Layout1, zeroFp32, maskB32);
            AscendC::Reg::Compare<float, CMPMODE::NE>(validScaleOdd0, reciprocalOddFp32Layout0, zeroFp32, maskB32);
            AscendC::Reg::Compare<float, CMPMODE::NE>(validScaleOdd1, reciprocalOddFp32Layout1, zeroFp32, maskB32);
            for (uint16_t localRow = localRowStart; localRow < localRowEnd; ++localRow) {
                __ubuf__ XType* xCursor = xAddr + localRow * REG_VF_LEN_B16_DOUBLE;
                AscendC::Reg::DataCopy<XType, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                       AscendC::Reg::LoadDist::DIST_DINTLV_B16>(xEven, xOdd, xCursor,
                                                                                REG_VF_LEN_B16_DOUBLE);
                if constexpr (IsSameType<XType, half>::value) {
                    AscendC::Reg::Cast<float, XType, CAST_TO_FP32_ZERO>(xEvenFp32Layout0, xEven, maskX);
                    AscendC::Reg::Cast<float, XType, CAST_TO_FP32_ONE>(xEvenFp32Layout1, xEven, maskX);
                    AscendC::Reg::Cast<float, XType, CAST_TO_FP32_ZERO>(xOddFp32Layout0, xOdd, maskX);
                    AscendC::Reg::Cast<float, XType, CAST_TO_FP32_ONE>(xOddFp32Layout1, xOdd, maskX);
                    AscendC::Reg::Mul(xEvenFp32Layout0, xEvenFp32Layout0, reciprocalEvenFp32Layout0, maskB32);
                    AscendC::Reg::Mul(xEvenFp32Layout1, xEvenFp32Layout1, reciprocalEvenFp32Layout1, maskB32);
                    AscendC::Reg::Mul(xOddFp32Layout0, xOddFp32Layout0, reciprocalOddFp32Layout0, maskB32);
                    AscendC::Reg::Mul(xOddFp32Layout1, xOddFp32Layout1, reciprocalOddFp32Layout1, maskB32);
                } else {
                    AscendC::Reg::Mul(xEven, xEven, (RegTensor<XType>&)reciprocalEven, maskX);
                    AscendC::Reg::Mul(xOdd, xOdd, (RegTensor<XType>&)reciprocalOdd, maskX);
                    AscendC::Reg::Cast<float, XType, CAST_TO_FP32_ZERO>(xEvenFp32Layout0, xEven, maskX);
                    AscendC::Reg::Cast<float, XType, CAST_TO_FP32_ONE>(xEvenFp32Layout1, xEven, maskX);
                    AscendC::Reg::Cast<float, XType, CAST_TO_FP32_ZERO>(xOddFp32Layout0, xOdd, maskX);
                    AscendC::Reg::Cast<float, XType, CAST_TO_FP32_ONE>(xOddFp32Layout1, xOdd, maskX);
                }

                AscendC::Reg::Cast<YType, float, CAST_FP32_TO_Y_ZERO<roundMode>>(yEvenFp8Layout0, xEvenFp32Layout0,
                                                                                 validScaleEven0);
                AscendC::Reg::Cast<YType, float, CAST_FP32_TO_Y_TWO<roundMode>>(yEvenFp8Layout2, xEvenFp32Layout1,
                                                                                validScaleEven1);
                AscendC::Reg::Cast<YType, float, CAST_FP32_TO_Y_ONE<roundMode>>(yOddFp8Layout1, xOddFp32Layout0,
                                                                                validScaleOdd0);
                AscendC::Reg::Cast<YType, float, CAST_FP32_TO_Y_THREE<roundMode>>(yOddFp8Layout3, xOddFp32Layout1,
                                                                                  validScaleOdd1);
                AscendC::Reg::Add((RegTensor<uint8_t>&)yEvenFp8Layout0, (RegTensor<uint8_t>&)yEvenFp8Layout0,
                                  (RegTensor<uint8_t>&)yEvenFp8Layout2, maskY);
                AscendC::Reg::Add((RegTensor<uint8_t>&)yOddFp8Layout1, (RegTensor<uint8_t>&)yOddFp8Layout1,
                                  (RegTensor<uint8_t>&)yOddFp8Layout3, maskY);
                AscendC::Reg::Add((RegTensor<uint8_t>&)yEvenFp8Layout0, (RegTensor<uint8_t>&)yEvenFp8Layout0,
                                  (RegTensor<uint8_t>&)yOddFp8Layout1, maskY);
                __ubuf__ int8_t* yCursor = reinterpret_cast<__ubuf__ int8_t*>(y2Addr +
                                                                              localRow * REG_VF_LEN_B16_DOUBLE);
                AscendC::Reg::StoreAlign<int8_t, AscendC::Reg::PostLiteral::POST_MODE_UPDATE,
                                         AscendC::Reg::StoreDist::DIST_NORM_B8>(
                    yCursor, (RegTensor<int8_t>&)yEvenFp8Layout0, REG_VF_LEN_B16_DOUBLE, maskY);
            }
        }
    }
}

template <typename XType, typename YType, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void
GroupedDynamicMxQuantWithDualAxisBase<XType, YType, roundMode, scaleAlg>::ComputeY2DataTailBlockByReg(
    uint16_t calcRow, uint16_t calcCol, __ubuf__ XType* xAddr, __ubuf__ uint16_t* reciprocalAddr,
    __ubuf__ uint8_t* y2Addr)
{
    uint16_t colLoopCount = static_cast<uint16_t>(CeilDiv(static_cast<int64_t>(calcCol), REG_VF_LEN_FP32));
    __VEC_SCOPE__
    {
        RegTensor<XType> xRegTensor;
        RegTensor<uint16_t> reciprocalRegTensor;
        RegTensor<bfloat16_t> valueRegTensor;
        RegTensor<float> yZero;
        RegTensor<float> yOne;
        RegTensor<float> yInterleaveZero;
        RegTensor<float> yInterleaveOne;
        RegTensor<float> reciprocalZero;
        RegTensor<float> reciprocalOne;
        RegTensor<float> zeroFp32;
        RegTensor<YType> yZeroFP8;
        RegTensor<uint16_t> yRegTensor;
        RegTensor<uint8_t> outRegTensor;
        MaskReg maskAll = AscendC::Reg::CreateMask<uint16_t, AscendC::Reg::MaskPattern::ALL>();
        MaskReg validScaleZero;
        MaskReg validScaleOne;
        MaskReg validScaleInterleaveZero;
        MaskReg validScaleInterleaveOne;
        AscendC::Reg::Duplicate(zeroFp32, 0.0f);
        for (uint16_t scaleSlot = 0; scaleSlot < SCALE2_ROW_SLOT_COUNT; ++scaleSlot) {
            uint16_t localRowStart = static_cast<uint16_t>(scaleSlot * MX_BLOCK_SIZE);
            uint16_t localRowEnd = static_cast<uint16_t>(
                Min(static_cast<int64_t>(localRowStart) + MX_BLOCK_SIZE, calcRow));
            for (uint16_t colBlockIdx = 0; colBlockIdx < colLoopCount; ++colBlockIdx) {
                uint16_t colOffset = static_cast<uint16_t>(colBlockIdx * REG_VF_LEN_FP32);
                uint16_t loopLen = static_cast<uint16_t>(
                    Min(REG_VF_LEN_FP32, static_cast<int64_t>(calcCol - colOffset)));
                uint32_t dataLoopLen = loopLen;
                uint32_t outLoopLen = loopLen;
                MaskReg dataMask = AscendC::Reg::UpdateMask<XType>(dataLoopLen);
                MaskReg outMask = AscendC::Reg::UpdateMask<uint8_t>(outLoopLen);
                AscendC::Reg::DataCopy<uint16_t, AscendC::Reg::LoadDist::DIST_NORM>(
                    reciprocalRegTensor, reciprocalAddr + scaleSlot * calcCol + colOffset);
                AscendC::Reg::Cast<float, bfloat16_t, CAST_TO_FP32_ZERO>(
                    reciprocalZero, (RegTensor<bfloat16_t>&)reciprocalRegTensor, dataMask);
                AscendC::Reg::Cast<float, bfloat16_t, CAST_TO_FP32_ONE>(
                    reciprocalOne, (RegTensor<bfloat16_t>&)reciprocalRegTensor, dataMask);
                AscendC::Reg::Compare<float, CMPMODE::NE>(validScaleZero, reciprocalZero, zeroFp32, maskAll);
                AscendC::Reg::Compare<float, CMPMODE::NE>(validScaleOne, reciprocalOne, zeroFp32, maskAll);
                AscendC::Reg::MaskInterleave<float>(validScaleInterleaveZero, validScaleInterleaveOne, validScaleZero,
                                                    validScaleOne);
                for (uint16_t localRow = localRowStart; localRow < localRowEnd; ++localRow) {
                    AscendC::Reg::DataCopy<XType, AscendC::Reg::LoadDist::DIST_NORM>(
                        xRegTensor, xAddr + localRow * calcCol + colOffset);
                    if constexpr (IsSameType<XType, half>::value) {
                        AscendC::Reg::Cast<float, XType, CAST_TO_FP32_ZERO>(yZero, xRegTensor, dataMask);
                        AscendC::Reg::Cast<float, XType, CAST_TO_FP32_ONE>(yOne, xRegTensor, dataMask);
                        AscendC::Reg::Mul(yZero, yZero, reciprocalZero, maskAll);
                        AscendC::Reg::Mul(yOne, yOne, reciprocalOne, maskAll);
                    } else {
                        AscendC::Reg::Mul(valueRegTensor, xRegTensor, (RegTensor<bfloat16_t>&)reciprocalRegTensor,
                                          dataMask);
                        AscendC::Reg::Cast<float, bfloat16_t, CAST_TO_FP32_ZERO>(yZero, valueRegTensor, maskAll);
                        AscendC::Reg::Cast<float, bfloat16_t, CAST_TO_FP32_ONE>(yOne, valueRegTensor, maskAll);
                    }
                    AscendC::Reg::Interleave(yInterleaveZero, yInterleaveOne, yZero, yOne);
                    AscendC::Reg::Cast<YType, float, CAST_FP32_TO_Y_ZERO<roundMode>>(yZeroFP8, yInterleaveZero,
                                                                                     validScaleInterleaveZero);
                    AscendC::Reg::Pack(yRegTensor, (RegTensor<uint32_t>&)yZeroFP8);
                    AscendC::Reg::Pack(outRegTensor, yRegTensor);
                    AscendC::Reg::DataCopy<uint8_t, AscendC::Reg::StoreDist::DIST_NORM>(
                        y2Addr + localRow * calcCol + colOffset, outRegTensor, outMask);
                }
            }
        }
    }
}

template <typename XType, typename YType, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void GroupedDynamicMxQuantWithDualAxisBase<XType, YType, roundMode, scaleAlg>::CopyOutScale2(
    int64_t groupId, int64_t groupStart, int64_t groupEnd, int64_t rowBlockIdx, int64_t colStart, int64_t calcCol)
{
    LocalTensor<uint8_t> scale2Local = y2ScaleOutQueue_.template DeQue<uint8_t>();
    DataCopyExtParams copyParams = {0, 0, 0, 0, 0};
    copyParams.blockCount = 1;
    copyParams.blockLen = static_cast<uint32_t>(calcCol * SCALE2_ROW_SLOT_COUNT *
                                                static_cast<int64_t>(sizeof(uint8_t)));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;

    int64_t scale2Row = groupStart / DOUBLE_MX_BLOCK_SIZE + groupId + rowBlockIdx;
    int64_t scale2Offset = (scale2Row * n_ + colStart) * SCALE2_ROW_SLOT_COUNT;
    DataCopyPad(y2ScaleGm_[scale2Offset], scale2Local, copyParams);

    int64_t rowBlocks = CeilDiv(groupEnd - groupStart, rowBlockSize_);
    int64_t nextGroupScale2Row = groupEnd / DOUBLE_MX_BLOCK_SIZE + groupId + 1;
    if (rowBlockIdx == rowBlocks - 1 && scale2Row + 1 < nextGroupScale2Row) {
        // y2_scale 的第一维为 floor(M / 64) + G。该布局可能在当前组实际 scale 行和
        // 下一组起始行之间预留一行，需要显式写 0
        PipeBarrier<PIPE_ALL>();
        Duplicate<uint8_t>(scale2Local, static_cast<uint8_t>(0), static_cast<int32_t>(calcCol * SCALE2_ROW_SLOT_COUNT));
        PipeBarrier<PIPE_ALL>();
        int64_t paddingOffset = ((scale2Row + 1) * n_ + colStart) * SCALE2_ROW_SLOT_COUNT;
        DataCopyPad(y2ScaleGm_[paddingOffset], scale2Local, copyParams);
    }
    y2ScaleOutQueue_.FreeTensor(scale2Local);
}

template <typename XType, typename YType, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void GroupedDynamicMxQuantWithDualAxisBase<XType, YType, roundMode, scaleAlg>::CopyOutY1(
    int64_t rowStart, int64_t rowEnd, int64_t colStart, int64_t colEnd)
{
    int64_t calcRow = rowEnd - rowStart;
    int64_t calcCol = colEnd - colStart;

    LocalTensor<uint8_t> y1OutLocal = y1OutQueue_.template DeQue<uint8_t>();
    DataCopyExtParams copyParams = {0, 0, 0, 0, 0};
    copyParams.blockCount = static_cast<uint16_t>(calcRow);
    copyParams.blockLen = static_cast<uint32_t>(calcCol * static_cast<int64_t>(sizeof(uint8_t)));
    copyParams.srcStride = 0;
    copyParams.dstStride = static_cast<uint32_t>((n_ - calcCol) * static_cast<int64_t>(sizeof(uint8_t)));

    int64_t yOffset = rowStart * n_ + colStart;
    DataCopyPad(y1Gm_[yOffset], y1OutLocal, copyParams);

    y1OutQueue_.FreeTensor(y1OutLocal);
}

template <typename XType, typename YType, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void GroupedDynamicMxQuantWithDualAxisBase<XType, YType, roundMode, scaleAlg>::CopyOutY2(
    int64_t rowStart, int64_t rowEnd, int64_t colStart, int64_t colEnd)
{
    int64_t calcRow = rowEnd - rowStart;
    int64_t calcCol = colEnd - colStart;

    LocalTensor<uint8_t> y2OutLocal = y2OutQueue_.template DeQue<uint8_t>();
    DataCopyExtParams copyParams = {0, 0, 0, 0, 0};
    copyParams.blockCount = static_cast<uint16_t>(calcRow);
    copyParams.blockLen = static_cast<uint32_t>(calcCol * static_cast<int64_t>(sizeof(uint8_t)));
    copyParams.srcStride = 0;
    copyParams.dstStride = static_cast<uint32_t>((n_ - calcCol) * static_cast<int64_t>(sizeof(uint8_t)));

    int64_t yOffset = rowStart * n_ + colStart;
    DataCopyPad(y2Gm_[yOffset], y2OutLocal, copyParams);

    y2OutQueue_.FreeTensor(y2OutLocal);
}

template <typename XType, typename YType, AscendC::RoundMode roundMode, uint64_t scaleAlg>
__aicore__ inline void GroupedDynamicMxQuantWithDualAxisBase<XType, YType, roundMode, scaleAlg>::CopyOutScale1(
    int64_t rowStart, int64_t rowEnd, int64_t colStart, int64_t colEnd)
{
    int64_t calcRow = rowEnd - rowStart;
    int64_t calcCol = colEnd - colStart;
    int64_t scaleCount = calcCol / MX_BLOCK_SIZE;
    LocalTensor<uint8_t> scale1Local = y1ScaleOutQueue_.template DeQue<uint8_t>();

    DataCopyExtParams copyParams = {0, 0, 0, 0, 0};
    copyParams.blockCount = static_cast<uint16_t>(calcRow);
    copyParams.blockLen = static_cast<uint32_t>(scaleCount * static_cast<int64_t>(sizeof(uint8_t)));
    copyParams.srcStride = 0;
    copyParams.dstStride = static_cast<uint32_t>((scale1ColPairs_ * DIGIT_TWO - scaleCount) *
                                                 static_cast<int64_t>(sizeof(uint8_t)));

    int64_t scale1Offset = rowStart * scale1ColPairs_ * DIGIT_TWO + (colStart / DOUBLE_MX_BLOCK_SIZE) * DIGIT_TWO;
    DataCopyPad<uint8_t, PaddingMode::Compact>(y1ScaleGm_[scale1Offset], scale1Local, copyParams);
    y1ScaleOutQueue_.FreeTensor(scale1Local);
}
} // namespace GroupedDynamicMxQuantWithDualAxis

#endif // GROUPED_DYNAMIC_MX_QUANT_WITH_DUAL_AXIS_REGBASE_H
