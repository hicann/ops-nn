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
 * \file map_index.h
 * \brief
 */
#ifndef MAP_INDEX_H_
#define MAP_INDEX_H_

#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#include "kernel_tiling/kernel_tiling.h"

namespace MapIndexOp {

using namespace AscendC;
constexpr uint32_t BLOCK_INT8 = 8;
constexpr uint32_t VF_LEN_INT32 = 64;
constexpr uint32_t ONE_BLOCK_NUM = 32;
constexpr uint8_t MASK_RESULT = 17; // 00010001
class MapIndex {
public:
    __aicore__ inline MapIndex(){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR dataSeq, GM_ADDR y, GM_ADDR workspace, const MapIndexTilingData &tilingData, TPipe* inputPipe);
    __aicore__ inline void ParseTilingData(const MapIndexTilingData &tilingData);
    __aicore__ inline void CopyInX(int64_t calCount);
    __aicore__ inline void CopyInDataSeq(int64_t offset, int64_t calCount);
    __aicore__ inline void ProcessOneRow(LocalTensor<int32_t> &xLocalRes, int64_t offset, int32_t rowCount, LocalTensor<uint8_t> &compareRes, LocalTensor<uint8_t> &compareResultMask);
    __aicore__ inline void Process();
    __aicore__ inline void ComputeOneRowMask(LocalTensor<int32_t> &xLocalRes, LocalTensor<int32_t> &dataSeqLocalRes, LocalTensor<uint8_t> &maskRes, int32_t calCount);
    __aicore__ inline void CopyOut();

private:
    TPipe *pipe;
    /* global memory address */
    GlobalTensor<int32_t> xGm_;
    GlobalTensor<int32_t> dataSeqGm_;
    GlobalTensor<int32_t> yGm_;

    TQue<QuePosition::VECIN, 1> inXQueue_;
    TQue<QuePosition::VECIN, 1> inDataSeqQueue_;
    TQue<QuePosition::VECOUT, 1> outYQueue_;
    TBuf<TPosition::VECCALC> maskBuf_;
    TBuf<TPosition::VECCALC> compareBuf_;
    TBuf<TPosition::VECCALC> compareResultBuf_;

    // 变量区
    uint32_t usedCoreNum_ = 0;
    int64_t curCoreProcessNum_ = 0;
    int64_t normalCoreProcessNum_ = 0;
    int64_t tailCoreProcessNum_ = 0;
    int64_t Dim1Size_ = 0;
    int64_t Dim1SizeAlign_ = 0;
    int64_t CopyInDim0_ = 0;
    int64_t CopyInDim0Times_ = 0;
    int64_t tailCopyInDim0Times_ = 0;
    int64_t doubleBuffNum_ = 0;
    uint32_t blockIdx_ = 0;

    /* variable */
    int64_t loopNum_ = 0;
};

__aicore__ inline void MapIndex::ParseTilingData(const MapIndexTilingData &tilingData)
{
    usedCoreNum_ = tilingData.usedCoreNum;
    normalCoreProcessNum_ = tilingData.normalCoreProcessNum;
    tailCoreProcessNum_ = tilingData.tailCoreProcessNum;
    Dim1Size_ = tilingData.Dim1Size;
    Dim1SizeAlign_ = tilingData.Dim1SizeAlign;
    CopyInDim0_ = tilingData.CopyInDim0;
    CopyInDim0Times_ = tilingData.CopyInDim0Times;
    tailCopyInDim0Times_ = tilingData.tailCopyInDim0Times;
    doubleBuffNum_ = tilingData.doubleBuffNum;
}

__aicore__ inline void MapIndex::Init(GM_ADDR x, GM_ADDR dataSeq, GM_ADDR y, GM_ADDR workspace, const MapIndexTilingData &tilingData, TPipe* inputPipe)
{
    pipe = inputPipe;
    ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
    blockIdx_ = GetBlockIdx();
    ParseTilingData(tilingData);
    // shield global memory address between different core
    uint64_t intraCoreOffset = blockIdx_ * normalCoreProcessNum_ * Dim1Size_;

    if (blockIdx_ + 1 == usedCoreNum_) {
        curCoreProcessNum_ = tailCoreProcessNum_;
    } else {
        curCoreProcessNum_ = normalCoreProcessNum_;
    }

    xGm_.SetGlobalBuffer((__gm__ int32_t *)x);
    dataSeqGm_.SetGlobalBuffer((__gm__ int32_t *)dataSeq);
    yGm_.SetGlobalBuffer((__gm__ int32_t *)y);

    if (blockIdx_ == 0) {
        InitOutput<int32_t>(yGm_, 1, -1);
        PipeBarrier<PIPE_ALL>();
    }
    SyncAll();

    pipe->InitBuffer(inXQueue_, 1, Dim1SizeAlign_ * sizeof(int32_t));
    pipe->InitBuffer(inDataSeqQueue_, static_cast<uint8_t>(doubleBuffNum_), Dim1SizeAlign_ * sizeof(int32_t));
    pipe->InitBuffer(maskBuf_, BLOCK_INT8 * sizeof(int32_t));
    pipe->InitBuffer(compareBuf_, BLOCK_INT8 * sizeof(int32_t));
    pipe->InitBuffer(compareResultBuf_, BLOCK_INT8 * sizeof(int32_t));
    pipe->InitBuffer(outYQueue_, 1, BLOCK_INT8 * sizeof(int32_t));
}

__aicore__ inline void MapIndex::CopyInX(int64_t calCount)
{
    LocalTensor<int32_t> xLocal = inXQueue_.AllocTensor<int32_t>();
    Duplicate<int32_t>(xLocal[Dim1SizeAlign_ - VF_LEN_INT32], static_cast<int32_t>(0), VF_LEN_INT32);
    event_t eventIDVToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(eventIDVToMTE2);
    WaitFlag<HardEvent::V_MTE2>(eventIDVToMTE2);
    DataCopyExtParams copyParams{ static_cast<uint16_t>(1), static_cast<uint32_t>(calCount * sizeof(int32_t)),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0) };

    DataCopyPadExtParams<int32_t> padParams{ false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
        static_cast<int32_t>(0) };
    DataCopyPad(xLocal, xGm_, copyParams, padParams);
    inXQueue_.EnQue(xLocal);
}

__aicore__ inline void MapIndex::CopyInDataSeq(int64_t offset, int64_t calCount)
{
    LocalTensor<int32_t> dataSeqLocal = inDataSeqQueue_.AllocTensor<int32_t>();
    Duplicate<int32_t>(dataSeqLocal[Dim1SizeAlign_ - VF_LEN_INT32], static_cast<int32_t>(0), VF_LEN_INT32);
    event_t eventIDVToMTE2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(eventIDVToMTE2);
    WaitFlag<HardEvent::V_MTE2>(eventIDVToMTE2);
    DataCopyExtParams copyParams{ static_cast<uint16_t>(1), static_cast<uint32_t>(calCount * sizeof(int32_t)),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0) };
    uint64_t intraCoreOffset = blockIdx_ * normalCoreProcessNum_ * Dim1Size_;
    DataCopyPadExtParams<int32_t> padParams{ false, static_cast<uint8_t>(0), static_cast<uint8_t>(0),
        static_cast<int32_t>(0) };
    DataCopyPad(dataSeqLocal, dataSeqGm_[offset+intraCoreOffset], copyParams, padParams);
    inDataSeqQueue_.EnQue(dataSeqLocal);
}

__aicore__ inline void MapIndex::ProcessOneRow(LocalTensor<int32_t> &xLocalRes, int64_t offset, int32_t rowCount, LocalTensor<uint8_t> &compareRes, LocalTensor<uint8_t> &compareResultMask)
{
    CopyInDataSeq(offset, rowCount);
    LocalTensor<int32_t> dataSeqLocalRes = inDataSeqQueue_.DeQue<int32_t>();
    LocalTensor<uint8_t> maskRes = maskBuf_.Get<uint8_t>();
    ComputeOneRowMask(xLocalRes, dataSeqLocalRes, maskRes, Dim1SizeAlign_);
    inDataSeqQueue_.FreeTensor(dataSeqLocalRes);
    Compare(compareResultMask, compareRes, maskRes, CMPMODE::EQ, ONE_BLOCK_NUM);
}

__aicore__ inline void MapIndex::Process()
{
    if (blockIdx_ >= usedCoreNum_) {
        return;
    }
    CopyInX(Dim1Size_);
    LocalTensor<uint8_t> compareRes = compareBuf_.Get<uint8_t>();
    LocalTensor<uint8_t> compareResultMask = compareResultBuf_.Get<uint8_t>();
    Duplicate<uint8_t>(compareRes, static_cast<uint8_t>(MASK_RESULT), ONE_BLOCK_NUM); // 00010001
    LocalTensor<int32_t> xLocalRes = inXQueue_.DeQue<int32_t>();
    int64_t loopNum = curCoreProcessNum_;
    for (int64_t loopIndex = 0; loopIndex < loopNum; loopIndex++) {
        ProcessOneRow(xLocalRes, Dim1Size_ * loopIndex, Dim1Size_, compareRes,  compareResultMask);
        event_t eventIDVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eventIDVToS);
        WaitFlag<HardEvent::V_S>(eventIDVToS);
        uint32_t a = compareResultMask.ReinterpretCast<uint32_t>().GetValue(0);
        if(a == static_cast<uint32_t>(0xFFFFFFFF)) {
            LocalTensor<int32_t> outYLocal = outYQueue_.AllocTensor<int32_t>();
            Duplicate<int32_t>(outYLocal, static_cast<int32_t>(loopIndex + blockIdx_ * normalCoreProcessNum_), BLOCK_INT8);
            outYQueue_.EnQue(outYLocal);
            CopyOut();
            break;
        }
    }
    inXQueue_.FreeTensor(xLocalRes);
}

__aicore__ inline void MapIndex::ComputeOneRowMask(LocalTensor<int32_t> &xLocalRes, LocalTensor<int32_t> &dataSeqLocalRes, LocalTensor<uint8_t> &maskRes, int32_t calCount)
{
    uint32_t dtypeSize = sizeof(int32_t);
    uint32_t vl = VECTOR_REG_WIDTH / dtypeSize;
    uint16_t loopNum = (calCount + vl - 1) / vl;
    uint32_t vlSize = vl;
    __ubuf__ int32_t* xAddr = (__ubuf__ int32_t*)xLocalRes.GetPhyAddr();
    __ubuf__ int32_t* dataSeqAddr = (__ubuf__ int32_t*)dataSeqLocalRes.GetPhyAddr();
    __ubuf__ uint8_t* maskAddr = (__ubuf__ uint8_t*)maskRes.GetPhyAddr();

    __VEC_SCOPE__
    {
        AscendC::MicroAPI::RegTensor<int32_t> xReg;
        AscendC::MicroAPI::RegTensor<int32_t> dataSeqReg;
        AscendC::MicroAPI::MaskReg preg0;
        AscendC::MicroAPI::MaskReg preg1 = AscendC::MicroAPI::CreateMask<int32_t, AscendC::MicroAPI::MaskPattern::ALL>();
        AscendC::MicroAPI::MaskReg preg2;
        AscendC::MicroAPI::MaskReg pregResultMask = AscendC::MicroAPI::CreateMask<int32_t, AscendC::MicroAPI::MaskPattern::ALL>();
        
        uint32_t sreg0 = calCount;
        for (uint16_t i = 0; i < loopNum; i++) { // 256B
            preg0 = AscendC::MicroAPI::UpdateMask<int32_t>(sreg0);
            AscendC::MicroAPI::DataCopy(xReg, xAddr + i * vl);
            AscendC::MicroAPI::DataCopy(dataSeqReg, dataSeqAddr + i * vl);
            AscendC::MicroAPI::Compare<int32_t, CMPMODE::EQ>(preg2, xReg, dataSeqReg, preg0);
            AscendC::MicroAPI::MaskAnd(pregResultMask, pregResultMask, preg2, preg0);
        }
        AscendC::MicroAPI::DataCopy(maskAddr, pregResultMask);
    }
}

__aicore__ inline void MapIndex::CopyOut()
{
    LocalTensor<int32_t> yOutLocal = outYQueue_.DeQue<int32_t>();
    DataCopyExtParams copyParams{ static_cast<uint16_t>(1), static_cast<uint32_t>(1 * sizeof(int32_t)),
        static_cast<uint32_t>(0), static_cast<uint32_t>(0), static_cast<uint32_t>(0) };
    DataCopyPad(yGm_, yOutLocal, copyParams);
    outYQueue_.FreeTensor(yOutLocal);
}

}
#endif 