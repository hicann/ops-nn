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
 * \file grouped_dynamic_hi_f4_quant_split.h
 * \brief 使用 GroupedSplitBase 统一切分的 kernel 类。
 *        切分（逐组动态配核 + 核窗口轮转 + 组内等大 tile 均分）全部在 kernel 内完成，
 *        host 侧 tiling 仅提供 tile 几何（blockColSize×blockRowSize，blockColSize = ubMaxblockNum×64）。
 *        替代原 mode0/mode1 两套切核模式。
 */

#ifndef OPS_NN_GROUPED_DYNAMIC_HIF4_QUANT_SPLIT_H
#define OPS_NN_GROUPED_DYNAMIC_HIF4_QUANT_SPLIT_H

#define FLOAT_OVERFLOW_MODE_CTRL 60

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../inc/platform.h"
#include "../inc/kernel_utils.h"
#include "grouped_dynamic_hi_f4_quant_tilingdata.h"
#include "grouped_dynamic_hi_f4_quant_base.h"
#include "../../quant_common/grouped_split_base.h"

namespace GroupedDynamicHiF4Quant {
using namespace AscendC;

template <typename xDtype, typename yDtype>
class GroupedDynamicHiF4QuantSplit
    : public GroupedDynamicHiF4QuantBase<xDtype, yDtype>,
      public GroupedSplitBase::GroupedSplit<GroupedDynamicHiF4QuantSplit<xDtype, yDtype>, int32_t> {
public:
    __aicore__ inline GroupedDynamicHiF4QuantSplit(const GroupedDynamicHiF4QuantTilingData* tilingData, TPipe* pipe)
        : tilingData_(tilingData), pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR group_index, GM_ADDR y, GM_ADDR scale);
    __aicore__ inline void Process();
    __aicore__ inline void ProcessOneLoop(const int64_t curBlockRowSize, const int64_t curBlockColSize,
                                          const int64_t blockRowIdx, const int64_t blockColIdx,
                                          const int64_t groupStart, const int64_t groupIdx);

private:
    __aicore__ inline void CopyIn(int64_t blockCount, int64_t blockLen, int64_t xGmOffset);
    __aicore__ inline void Compute(int64_t blockCount, int64_t blockLen);
    __aicore__ inline void CopyOut(int64_t blockCount, int64_t blockLen, int64_t xGmOffset, int64_t scaleGmOffset);
    const GroupedDynamicHiF4QuantTilingData* tilingData_{};
    TPipe* pipe_;

    TQue<QuePosition::VECIN, DB_BUFFER> xInQueue_;
    TQue<QuePosition::VECOUT, DB_BUFFER> yOutQueue_;
    TQue<QuePosition::VECOUT, DB_BUFFER> scaleOutFrontQueue_;
    TQue<QuePosition::VECOUT, DB_BUFFER> scaleOutBackQueue_;

    TBuf<TPosition::VECCALC> tmpScaleBuf_;
    TBuf<TPosition::VECCALC> maxPer4Buf_;
    TBuf<TPosition::VECCALC> maxPer8Buf_;
    TBuf<TPosition::VECCALC> eaBuf_;

    GlobalTensor<xDtype> xGm_;
    GlobalTensor<uint8_t> yGm_;
    GlobalTensor<uint16_t> scaleGm_;

private:
    int64_t coreIdx_ = 0;
    // 断言缓存：groupIndex 边界只需每组校验一次（避免每 tile 重复 GetValue）
    int64_t lastAssertGroupIdx_ = -1;
};

template <typename xDtype, typename yDtype>
__aicore__ inline void GroupedDynamicHiF4QuantSplit<xDtype, yDtype>::Init(GM_ADDR x, GM_ADDR group_index, GM_ADDR y,
                                                                          GM_ADDR scale)
{
#if (__NPU_ARCH__ == 3101)
    AscendC::SetCtrlSpr<FLOAT_OVERFLOW_MODE_CTRL, FLOAT_OVERFLOW_MODE_CTRL>(0);
#endif
    coreIdx_ = GetBlockIdx();
    if (coreIdx_ >= tilingData_->usedCoreNum) {
        return;
    }

    xGm_.SetGlobalBuffer((__gm__ xDtype*)(x));
    this->InitGroup(group_index);
    yGm_.SetGlobalBuffer((__gm__ uint8_t*)(y));
    scaleGm_.SetGlobalBuffer((__gm__ uint16_t*)(scale));
    this->BaseInitData(tilingData_);

    int64_t scaleBlockNum = this->blockColSize_ / BLOCK_SIZE;
    int64_t bufferSize = this->blockRowSize_ * this->blockColSize_;
    int64_t xBufferSize = bufferSize * BYTE_OF_INPUT_X;
    int64_t yBufferSize = bufferSize / Y_NUM_OF_ONE_BYTE;
    int64_t scaleFrontBufferSize = scaleBlockNum * this->blockRowSize_ * BYTE_OF_INPUT_X;
    int64_t scaleBackBufferSize = scaleFrontBufferSize;
    int64_t tmpScaleBufferSize = bufferSize / NUM_FOUR * BYTE_OF_INPUT_X;
    int64_t maxPer4BufSize = tmpScaleBufferSize;
    int64_t maxPer8BufSize = bufferSize / NUM_EIGHT * BYTE_OF_INPUT_X;
    int64_t eaBufSize = scaleBlockNum * this->blockRowSize_ * BYTE_OF_INPUT_X;

    pipe_->InitBuffer(xInQueue_, DB_BUFFER, xBufferSize);
    pipe_->InitBuffer(yOutQueue_, DB_BUFFER, yBufferSize);
    pipe_->InitBuffer(scaleOutFrontQueue_, DB_BUFFER, scaleFrontBufferSize);
    pipe_->InitBuffer(scaleOutBackQueue_, DB_BUFFER, scaleBackBufferSize);
    pipe_->InitBuffer(tmpScaleBuf_, tmpScaleBufferSize);
    pipe_->InitBuffer(maxPer4Buf_, maxPer4BufSize);
    pipe_->InitBuffer(maxPer8Buf_, maxPer8BufSize);
    pipe_->InitBuffer(eaBuf_, eaBufSize);
}

template <typename xDtype, typename yDtype>
__aicore__ inline void GroupedDynamicHiF4QuantSplit<xDtype, yDtype>::Process()
{
    if (coreIdx_ >= tilingData_->usedCoreNum) {
        return;
    }
    // 切分全部交由 GroupedSplitBase：逐组动态配核 + 核窗口轮转 + 组内等大 tile 均分
    this->ProcessBase(tilingData_->usedCoreNum, coreIdx_, this->groupNum_, this->blockColSize_, this->blockRowSize_,
                      this->tailBlockRowSize_, this->blockRowNum_);
}

template <typename xDtype, typename yDtype>
__aicore__ inline void GroupedDynamicHiF4QuantSplit<xDtype, yDtype>::ProcessOneLoop(
    const int64_t curBlockRowSize, const int64_t curBlockColSize, const int64_t blockRowIdx, const int64_t blockColIdx,
    const int64_t groupStart, const int64_t groupIdx)
{
    // groupIndex 边界断言：每组每核仅一次，避免逐 tile 重复 GetValue
    if (lastAssertGroupIdx_ != groupIdx) {
        int64_t groupEnd = this->groupIndexGm_.GetValue(groupIdx);
        assert((0 <= groupEnd && groupEnd <= this->colSize_), "groupIndex %lld = %lld out of range[0 %lld]!\n",
               groupIdx, groupEnd, this->colSize_);
        assert((groupStart <= groupEnd), "groupIndex %lld = %lld less than previous groupIdx = %lld!\n", groupIdx,
               groupEnd, groupStart);
        lastAssertGroupIdx_ = groupIdx;
    }

    int64_t xGmOffset = (groupStart + blockColIdx * this->blockColSize_) * this->rowSize_ +
                        blockRowIdx * this->blockRowSize_;
    int64_t scaleGmOffset = (groupStart / BLOCK_SIZE + groupIdx + blockColIdx * (this->blockColSize_ / BLOCK_SIZE)) *
                                this->rowSize_ * NUM_TWO +
                            blockRowIdx * this->blockRowSize_;

    CopyIn(curBlockColSize, curBlockRowSize, xGmOffset);
    Compute(curBlockColSize, curBlockRowSize);
    CopyOut(curBlockColSize, curBlockRowSize, xGmOffset, scaleGmOffset);
}

template <typename xDtype, typename yDtype>
__aicore__ inline void GroupedDynamicHiF4QuantSplit<xDtype, yDtype>::CopyIn(int64_t blockCount, int64_t blockLen,
                                                                            int64_t xGmOffset)
{
    DataCopyExtParams copyInParams = {
        static_cast<uint16_t>(blockCount), static_cast<uint32_t>(blockLen * BYTE_OF_INPUT_X),
        static_cast<uint32_t>((this->rowSize_ - blockLen) * BYTE_OF_INPUT_X),
        static_cast<uint32_t>((this->blockRowSize_ - blockLen) * BYTE_OF_INPUT_X / UB_BLOCK_SIZE),
        static_cast<uint32_t>(0)};
    DataCopyPadExtParams<xDtype> padParams{false, 0, 0, 0};

    LocalTensor<xDtype> xLocal = xInQueue_.template AllocTensor<xDtype>();
    DataCopyPad(xLocal, xGm_[xGmOffset], copyInParams, padParams);
    xInQueue_.template EnQue(xLocal);
}

template <typename xDtype, typename yDtype>
__aicore__ inline void GroupedDynamicHiF4QuantSplit<xDtype, yDtype>::Compute(int64_t blockCount, int64_t blockLen)
{
    LocalTensor<xDtype> x = xInQueue_.template DeQue<xDtype>();
    LocalTensor<uint8_t> y = yOutQueue_.template AllocTensor<uint8_t>();
    LocalTensor<uint16_t> scaleFront = scaleOutFrontQueue_.template AllocTensor<uint16_t>();
    LocalTensor<uint16_t> scaleBack = scaleOutBackQueue_.template AllocTensor<uint16_t>();

    LocalTensor<uint16_t> ea = eaBuf_.Get<uint16_t>();
    LocalTensor<uint16_t> tmpScale = tmpScaleBuf_.Get<uint16_t>();
    LocalTensor<uint16_t> maxPer4 = maxPer4Buf_.Get<uint16_t>();
    LocalTensor<uint16_t> maxPer8 = maxPer8Buf_.Get<uint16_t>();

    auto xAddr = (__ubuf__ xDtype*)x.GetPhyAddr();
    auto yAddr = (__ubuf__ uint32_t*)y.GetPhyAddr();
    auto scaleFrontAddr = (__ubuf__ uint16_t*)scaleFront.GetPhyAddr();
    auto scaleBackAddr = (__ubuf__ uint16_t*)scaleBack.GetPhyAddr();
    auto eaAddr16 = (__ubuf__ uint16_t*)ea.GetPhyAddr();
    auto tmpScaleAddr = (__ubuf__ uint16_t*)tmpScale.GetPhyAddr();
    auto maxPer4Addr = (__ubuf__ uint16_t*)maxPer4.GetPhyAddr();
    auto maxPer8Addr = (__ubuf__ uint16_t*)maxPer8.GetPhyAddr();

    int64_t maxPer4Num = ops::CeilDiv(blockCount, NUM_FOUR);
    uint16_t maxPer4LoopNum = static_cast<uint16_t>(maxPer4Num - 1);
    uint16_t tailMaxper4 = static_cast<uint16_t>(blockCount - maxPer4LoopNum * NUM_FOUR);

    int64_t maxPer8Num = ops::CeilDiv(maxPer4Num, NUM_TWO);
    uint16_t maxPer8LoopNum = static_cast<uint16_t>(maxPer8Num - 1);
    uint16_t tailMaxper8 = static_cast<uint16_t>(maxPer4Num - maxPer8LoopNum * NUM_TWO);

    int64_t maxPer64Num = ops::CeilDiv(maxPer8Num, NUM_EIGHT);
    uint16_t maxPer64LoopNum = static_cast<uint16_t>(maxPer64Num - 1);
    uint16_t tail8Maxper64 = static_cast<uint16_t>(maxPer8Num - maxPer64LoopNum * NUM_EIGHT);
    uint16_t tail4Maxper64 = static_cast<uint16_t>(maxPer4Num - maxPer64LoopNum * NUM_SIXTENN);
    // 尾组（最后一个 64 行组）拆解为整 8 行块数与残 4 行微块数，供 ComputeTmpScale 的 0/1 趟循环
    uint16_t tailFullLoop = static_cast<uint16_t>(tail4Maxper64 / NUM_TWO);
    uint16_t tailPartM4 = static_cast<uint16_t>(tail4Maxper64 % NUM_TWO);
    // 尾 4 行组拆解为成对行数与残行数，供 ComputeOut 的 0/1 趟循环
    uint16_t tailPairLoop = static_cast<uint16_t>(tailMaxper4 / NUM_TWO);
    uint16_t tailOdd = static_cast<uint16_t>(tailMaxper4 % NUM_TWO);

    this->ComputeMaxPer4(maxPer4LoopNum, tailMaxper4, xAddr, maxPer4Addr);
    this->ComputeMaxPer8(maxPer8LoopNum, tailMaxper8, maxPer4Addr, maxPer8Addr);
    this->ComputeE6M2(maxPer64LoopNum, tail8Maxper64, maxPer8Addr, eaAddr16);
    this->ComputeTmpScale(maxPer64LoopNum, tailFullLoop, tailPartM4, maxPer4Addr, maxPer8Addr, eaAddr16, tmpScaleAddr,
                          scaleFrontAddr, scaleBackAddr);
    this->ComputeOut(maxPer4LoopNum, tailPairLoop, tailOdd, xAddr, tmpScaleAddr, yAddr);

    xInQueue_.template FreeTensor(x);
    yOutQueue_.template EnQue(y);
    scaleOutFrontQueue_.template EnQue(scaleFront);
    scaleOutBackQueue_.template EnQue(scaleBack);
}

template <typename xDtype, typename yDtype>
__aicore__ inline void GroupedDynamicHiF4QuantSplit<xDtype, yDtype>::CopyOut(int64_t blockCount, int64_t blockLen,
                                                                             int64_t xGmOffset, int64_t scaleGmOffset)
{
    int64_t ySrcStride = (this->blockRowSize_ - blockLen) / NUM_TWO / UB_BLOCK_SIZE;
    int64_t yDstStride = (this->rowSize_ - blockLen) / NUM_TWO;
    int64_t scaleBlockCount = ops::CeilDiv(blockCount, BLOCK_SIZE);
    int64_t scaleSrcStride = (this->blockRowSize_ - blockLen) * NUM_TWO / UB_BLOCK_SIZE;
    int64_t scaleDstStride = (NUM_TWO * this->rowSize_ - blockLen) * NUM_TWO;

    DataCopyExtParams yCopyOutParams = {static_cast<uint16_t>(blockCount), static_cast<uint32_t>(blockLen / NUM_TWO),
                                        static_cast<uint32_t>(ySrcStride), static_cast<uint32_t>(yDstStride),
                                        static_cast<uint32_t>(0)};
    DataCopyExtParams scaleCopyOutParams = {
        static_cast<uint16_t>(scaleBlockCount), static_cast<uint32_t>(blockLen * NUM_TWO),
        static_cast<uint32_t>(scaleSrcStride), static_cast<uint32_t>(scaleDstStride), static_cast<uint32_t>(0)};

    auto yLocal = yOutQueue_.template DeQue<uint8_t>();
    DataCopyPad(yGm_[xGmOffset / 2], yLocal, yCopyOutParams);
    yOutQueue_.template FreeTensor(yLocal);

    auto scaleFrontLocal = scaleOutFrontQueue_.template DeQue<uint16_t>();
    DataCopyPad(scaleGm_[scaleGmOffset], scaleFrontLocal, scaleCopyOutParams);
    scaleOutFrontQueue_.template FreeTensor(scaleFrontLocal);

    auto scaleBackLocal = scaleOutBackQueue_.template DeQue<uint16_t>();
    DataCopyPad(scaleGm_[scaleGmOffset + this->rowSize_], scaleBackLocal, scaleCopyOutParams);
    scaleOutBackQueue_.template FreeTensor(scaleBackLocal);
}

} // namespace GroupedDynamicHiF4Quant
#endif // OPS_NN_GROUPED_DYNAMIC_HIF4_QUANT_SPLIT_H
