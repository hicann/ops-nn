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
 * \file grouped_dynamic_mx_quant_combine.h
 * \brief
 */

#ifndef GROUPED_DYNAMIC_MX_QUANT_COMBINE_H
#define GROUPED_DYNAMIC_MX_QUANT_COMBINE_H
#include "grouped_dynamic_mx_quant_common.h"
#include "../../quant_common/mx_quant_common.h"
#include "grouped_dynamic_mx_quant_tilingdata.h"
#include "grouped_dynamic_mx_quant_struct.h"
#include "../../quant_common/grouped_split_base.h"
#include "kernel_operator.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"

namespace GroupedDynamicMxQuant {
using namespace AscendC;
#define FLOAT_OVERFLOW_MODE_CTRL 60

constexpr int64_t NUM_TWO = 2;

template <typename T, typename U, const uint64_t scaleAlg, const uint64_t dstTypeMax, RoundMode roundMode>
class GroupedDynamicMxQuantCombine
    : public GroupedSplitBase::GroupedSplit<GroupedDynamicMxQuantCombine<T, U, scaleAlg, dstTypeMax, roundMode>> {
public:
    __aicore__ inline GroupedDynamicMxQuantCombine(const GroupedDynamicMxQuantTilingData* tilingData, TPipe* pipe)
        : tilingData_(tilingData), pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR groupIndex, GM_ADDR y, GM_ADDR mxScale);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessOneLoop(const int64_t curBlockRowSize, const int64_t curBlockColSize,
                                          const int64_t blockRowIdx, const int64_t blockColIdx,
                                          const int64_t groupStart, const int64_t groupIdx);

private:
    __aicore__ inline void InitTilingData();
    __aicore__ inline void Compute(int64_t dataLen, int64_t blockCount);
    __aicore__ inline void CopyIn(int64_t offset, int64_t blockCount, int64_t dataLen);
    __aicore__ inline void CopyOut(int64_t xOffset, int64_t scaleOffset, int64_t blockCount, int64_t dataLen);
    template <const int64_t padMode>
    __aicore__ inline void ComputeAll(int64_t dataLen, uint16_t loop0, uint16_t loop1, __ubuf__ T* xAddr,
                                      __ubuf__ uint8_t* mxScaleAddr, __ubuf__ uint8_t* yAddr);

private:
    TPipe* pipe_;
    TQue<QuePosition::VECIN, DB_BUFFER> inQueue_;
    TQue<QuePosition::VECOUT, DB_BUFFER> mxScaleQueue_;
    TQue<QuePosition::VECOUT, DB_BUFFER> outQueue_;
    GlobalTensor<T> xGm_;
    GlobalTensor<int32_t> groupIndexGm_;
    GlobalTensor<uint8_t> yGm_;
    GlobalTensor<uint8_t> mxScaleGm_;

    const GroupedDynamicMxQuantTilingData* tilingData_{};

    int64_t coreIdx_ = 0;
    int64_t rowSize_ = 0;
    int64_t colSize_ = 0;
    int64_t blockRowSize_ = 0;
    int64_t blockColSize_ = 0;
    int64_t blockRowTailSize_ = 0;
    int64_t blockRowCount_ = 0;
    int64_t groupNum_ = 0;
    int64_t totalCoreNum_ = 0;
    float invDstTypeMax_ = 0.0;
};

template <typename T, typename U, const uint64_t scaleAlg, const uint64_t dstTypeMax, RoundMode roundMode>
__aicore__ inline void GroupedDynamicMxQuantCombine<T, U, scaleAlg, dstTypeMax, roundMode>::Init(GM_ADDR x,
                                                                                                 GM_ADDR groupIndex,
                                                                                                 GM_ADDR y,
                                                                                                 GM_ADDR mxScale)
{
#if (__NPU_ARCH__ == 3510)
    SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
#endif

    coreIdx_ = GetBlockIdx();
    if (this->coreIdx_ >= tilingData_->usedCoreNum) {
        return;
    }
    this->InitGroup(groupIndex);
    InitTilingData();

    int64_t bufferSize = blockColSize_ * blockRowSize_ * sizeof(T);

    this->xGm_.SetGlobalBuffer((__gm__ T*)(x));
    this->yGm_.SetGlobalBuffer((__gm__ uint8_t*)(y));
    this->mxScaleGm_.SetGlobalBuffer((__gm__ uint8_t*)(mxScale));

    bufferSize = Ops::Base::CeilAlign(bufferSize, static_cast<int64_t>(Ops::Base::GetUbBlockSize()));
    this->pipe_->InitBuffer(this->inQueue_, DB_BUFFER, bufferSize);
    this->pipe_->InitBuffer(this->mxScaleQueue_, DB_BUFFER, bufferSize);
    this->pipe_->InitBuffer(this->outQueue_, DB_BUFFER, bufferSize);
}

template <typename T, typename U, const uint64_t scaleAlg, const uint64_t dstTypeMax, RoundMode roundMode>
__aicore__ inline void GroupedDynamicMxQuantCombine<T, U, scaleAlg, dstTypeMax, roundMode>::InitTilingData()
{
    blockRowCount_ = tilingData_->blockRowCount;
    blockRowTailSize_ = tilingData_->blockRowTailSize;
    groupNum_ = tilingData_->groupNum;
    totalCoreNum_ = tilingData_->totalCoreNum;
    rowSize_ = tilingData_->rowSize;
    colSize_ = tilingData_->colSize;
    blockRowSize_ = tilingData_->blockRowSize;
    blockColSize_ = tilingData_->blockColSize;
    invDstTypeMax_ = tilingData_->invDstTypeMax;
}

template <typename T, typename U, const uint64_t scaleAlg, const uint64_t dstTypeMax, RoundMode roundMode>
__aicore__ inline void GroupedDynamicMxQuantCombine<T, U, scaleAlg, dstTypeMax, roundMode>::Process()
{
    // 调用grouped算子基本切分方式，传入使用的核数，当前核id，group数量，基本块的高、宽，-1轴方向的基本切分数据
    this->ProcessBase(tilingData_->usedCoreNum, coreIdx_, groupNum_, blockColSize_, blockRowSize_, blockRowTailSize_,
                      blockRowCount_);
}

template <typename T, typename U, const uint64_t scaleAlg, const uint64_t dstTypeMax, RoundMode roundMode>
__aicore__ inline void GroupedDynamicMxQuantCombine<T, U, scaleAlg, dstTypeMax, roundMode>::ProcessOneLoop(
    const int64_t curBlockRowSize, const int64_t curBlockColSize, const int64_t blockRowIdx, const int64_t blockColIdx,
    const int64_t groupStart, const int64_t groupIdx)
{
    int64_t xOffset = (groupStart + blockColIdx * blockColSize_) * rowSize_ + blockRowIdx * blockRowSize_;
    int64_t scaleOffset = (groupStart / 64 + groupIdx + blockColIdx) * rowSize_ * NUM_TWO +
                          blockRowIdx * blockRowSize_ * NUM_TWO;

    CopyIn(xOffset, curBlockColSize, curBlockRowSize);
    Compute(curBlockColSize, curBlockRowSize);
    CopyOut(xOffset, scaleOffset, curBlockColSize, curBlockRowSize);
}

template <typename T, typename U, const uint64_t scaleAlg, const uint64_t dstTypeMax, RoundMode roundMode>
__aicore__ inline void GroupedDynamicMxQuantCombine<T, U, scaleAlg, dstTypeMax, roundMode>::CopyIn(int64_t offset,
                                                                                                   int64_t blockCount,
                                                                                                   int64_t dataLen)
{
    DataCopyExtParams inCopyParams_ = {static_cast<uint16_t>(blockCount), static_cast<uint32_t>(dataLen * sizeof(T)),
                                       static_cast<uint32_t>((rowSize_ - dataLen) * sizeof(T)),
                                       static_cast<uint32_t>(0), static_cast<uint32_t>(0)};

    DataCopyPadExtParams<T> padParams_ = {false, static_cast<uint8_t>(0), static_cast<uint8_t>(0), static_cast<T>(0)};
    LocalTensor<T> x = inQueue_.AllocTensor<T>();
    DataCopyPad(x, xGm_[offset], inCopyParams_, padParams_);
    inQueue_.EnQue(x);
}

template <typename T, typename U, const uint64_t scaleAlg, const uint64_t dstTypeMax, RoundMode roundMode>
__aicore__ inline void GroupedDynamicMxQuantCombine<T, U, scaleAlg, dstTypeMax, roundMode>::CopyOut(int64_t xOffset,
                                                                                                    int64_t scaleOffset,
                                                                                                    int64_t blockCount,
                                                                                                    int64_t dataLen)
{
    uint16_t outBurst = 0;
    uint32_t outBlockLen = 0;
    uint32_t srcStride = 0;
    uint32_t dstStride = 0;
    int64_t YOffset = 0;
    if constexpr (IsSameType<U, fp4x2_e2m1_t>::value || IsSameType<U, fp4x2_e1m2_t>::value) {
        outBurst = blockCount;
        outBlockLen = dataLen / 2 * sizeof(uint8_t);
        srcStride = 0;
        dstStride = (rowSize_ - dataLen) / 2 * sizeof(uint8_t);
        YOffset = xOffset / 2;
    } else {
        outBurst = blockCount;
        outBlockLen = dataLen * sizeof(uint8_t);
        srcStride = 0;
        dstStride = (rowSize_ - dataLen) * sizeof(uint8_t);
        YOffset = xOffset;
    }
    DataCopyExtParams outCopyParams_ = {static_cast<uint16_t>(outBurst), static_cast<uint32_t>(outBlockLen),
                                        static_cast<uint32_t>(srcStride), static_cast<uint32_t>(dstStride),
                                        static_cast<uint32_t>(0)};

    DataCopyExtParams scaleCopyOutParams = {static_cast<uint16_t>(1),
                                            static_cast<uint32_t>(dataLen * 2 * sizeof(uint8_t)),
                                            static_cast<uint32_t>(0), // 搬入pad做完交织后变成2倍pad
                                            static_cast<uint32_t>(0), static_cast<uint32_t>(0)};

    LocalTensor<uint8_t> y = outQueue_.DeQue<uint8_t>();
    DataCopyPad(yGm_[YOffset], y, outCopyParams_);
    outQueue_.FreeTensor(y);

    LocalTensor<uint8_t> mxScale = mxScaleQueue_.DeQue<uint8_t>();
    DataCopyPad(mxScaleGm_[scaleOffset], mxScale, scaleCopyOutParams);
    mxScaleQueue_.FreeTensor(mxScale);
}

template <typename T, typename U, const uint64_t scaleAlg, const uint64_t dstTypeMax, RoundMode roundMode>
__aicore__ inline void GroupedDynamicMxQuantCombine<T, U, scaleAlg, dstTypeMax, roundMode>::Compute(int64_t blockCount,
                                                                                                    int64_t dataLen)
{
    LocalTensor<T> x = this->inQueue_.template DeQue<T>();
    LocalTensor<uint8_t> mxScale = this->mxScaleQueue_.template AllocTensor<uint8_t>();
    LocalTensor<uint8_t> y = this->outQueue_.template AllocTensor<uint8_t>();

    auto xAddr = (__ubuf__ T*)x.GetPhyAddr();
    auto mxScaleAddr = (__ubuf__ uint8_t*)mxScale.GetPhyAddr();
    auto yAddr = (__ubuf__ uint8_t*)y.GetPhyAddr();

    uint16_t loop0 = 0;
    uint16_t loop1 = 0;

    if (blockCount <= 32) {
        loop0 = static_cast<uint16_t>(blockCount);
        loop1 = 0;
        ComputeAll<0>(dataLen, loop0, loop1, xAddr, mxScaleAddr, yAddr);
    } else {
        loop0 = static_cast<uint16_t>(BLOCK_SIZE);
        loop1 = static_cast<uint16_t>(blockCount - BLOCK_SIZE);
        ComputeAll<1>(dataLen, loop0, loop1, xAddr, mxScaleAddr, yAddr);
    }

    this->mxScaleQueue_.template EnQue(mxScale);
    this->outQueue_.template EnQue(y);
    this->inQueue_.template FreeTensor(x);
}

template <typename T, typename U, const uint64_t scaleAlg, const uint64_t dstTypeMax, RoundMode roundMode>
template <const int64_t padMode>
__aicore__ inline void GroupedDynamicMxQuantCombine<T, U, scaleAlg, dstTypeMax, roundMode>::ComputeAll(
    int64_t dataLen, uint16_t loop0, uint16_t loop1, __ubuf__ T* xAddr, __ubuf__ uint8_t* mxScaleAddr,
    __ubuf__ uint8_t* yAddr)
{
    int64_t inDataLen = (dataLen + 15) & (~15);
    int64_t xOffset = inDataLen * BLOCK_SIZE;
    int64_t yOffset = 0;
    int64_t outDataLen = 0;
    if constexpr (IsSameType<U, fp4x2_e2m1_t>::value || IsSameType<U, fp4x2_e1m2_t>::value) {
        outDataLen = ((dataLen + 63) & (~63)) / 2;
        yOffset = outDataLen * BLOCK_SIZE;
    } else {
        outDataLen = (dataLen + 31) & (~31);
        yOffset = outDataLen * BLOCK_SIZE;
    }
    __VEC_SCOPE__
    {
        Reg::RegTensor<uint8_t> scaleReg0;
        Reg::RegTensor<uint16_t> reversedScaleReg0;
        Reg::RegTensor<uint8_t> scaleReg1;
        Reg::RegTensor<uint16_t> reversedScaleReg1;

        Reg::MaskReg maskAll = Reg::CreateMask<uint8_t, Reg::MaskPattern::ALL>();

        MxQuantCommon::ComputeMxScale<T, U, scaleAlg, dstTypeMax>(inDataLen, loop0, xAddr, scaleReg0, reversedScaleReg0,
                                                                  invDstTypeMax_);
        MxQuantCommon::ComputeData<T, U, scaleAlg, dstTypeMax, roundMode>(inDataLen, outDataLen, loop0, xAddr, yAddr,
                                                                          reversedScaleReg0);
        if constexpr (padMode == 0) {
            Reg::Duplicate(scaleReg1, 0);
        } else {
            MxQuantCommon::ComputeMxScale<T, U, scaleAlg, dstTypeMax>(inDataLen, loop1, xAddr + xOffset, scaleReg1,
                                                                      reversedScaleReg1, invDstTypeMax_);
            MxQuantCommon::ComputeData<T, U, scaleAlg, dstTypeMax, roundMode>(
                inDataLen, outDataLen, loop1, xAddr + xOffset, yAddr + yOffset, reversedScaleReg1);
        }
        Reg::StoreAlign<uint8_t, Reg::StoreDist::DIST_INTLV_B8>(mxScaleAddr, scaleReg0, scaleReg1, maskAll);
    }
}

} // namespace GroupedDynamicMxQuant
#endif // GROUPED_DYNAMIC_MX_QUANT_COMBINE_H
