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
 * \file dequant_bias_multi.h
 * \brief dequant_bias_multi head file
 */

#ifndef DEQUANT_BIAS_MULTI_IMPL_H
#define DEQUANT_BIAS_MULTI_IMPL_H

namespace DequantBias {

using namespace AscendC;

template <typename XTYPE, typename WSTYPE, typename BIASTYPE, typename YTYPE, bool IFBIAS>
class DequantBiasMultiImpl {
public:
    __aicore__ inline DequantBiasMultiImpl() {};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weightScale, GM_ADDR activateScale, GM_ADDR bias, GM_ADDR y,
                                const DequantBiasTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ParseBaseTilingData(const DequantBiasTilingData* tilingData);
    __aicore__ inline void CopyIn(LocalTensor<XTYPE>& xLocal, int64_t rowOffset, int64_t inRows);
    __aicore__ inline void CopyOut(LocalTensor<XTYPE>& xLocal, int64_t rowOffset, int64_t outRows);
    __aicore__ inline void ComputeDequant(LocalTensor<XTYPE>& xLocal, int64_t rowsOffset, int64_t groupOffset, int64_t nProcess);
    __aicore__ inline void ComputeDequantWithBiasFloat(LocalTensor<XTYPE>& xLocal, int64_t rowsOffset, int64_t groupOffset, int64_t nProcess);
    __aicore__ inline void ComputeDequantWithBiasInt32(LocalTensor<XTYPE>& xLocal, int64_t rowsOffset, int64_t groupOffset, int64_t nProcess);
    __aicore__ inline void ComputeDequantWithoutBias(LocalTensor<XTYPE>& xLocal, int64_t rowsOffset, int64_t groupOffset, int64_t nProcess);
    __aicore__ inline void DataCopyInActivateScale();
    __aicore__ inline void DataCopyInBias(int64_t offset, int64_t nProcess);
    __aicore__ inline void DataCopyInWeightScale(int64_t offset, int64_t nProcess);
    __aicore__ inline void DataCopyInBiasFloat(int64_t offset, int64_t nProcess);

private:
    TPipe pipe_;
    GlobalTensor<XTYPE> xGM_;
    GlobalTensor<WSTYPE> weightScaleGM_;
    GlobalTensor<BIASTYPE> biasGM_;
    GlobalTensor<float> activeScaleGM_;
    GlobalTensor<YTYPE> yGM_;
    TBuf<> pingBuf_;
    TBuf<> pongBuf_;
    TBuf<> wsBuf_;
    TBuf<> asBuf_;
    TBuf<> biasBuf_;

    LocalTensor<float> wsLocal_;
    LocalTensor<BIASTYPE> biasLocal_;
    LocalTensor<float> asLocal_;

    event_t pongId_{EVENT_ID7};
    event_t pingId_{EVENT_ID6};

protected:
    const int64_t BLOCK_SIZE = 32;
    const int64_t PART_LEN = 8192;
    uint32_t blockIdx_ = 0;
    // tiling data
    uint32_t N_ = 0;
    uint32_t nAlign_ = 0;
    uint32_t asExist_ = 0;   // active scale是否存在
    uint32_t needCoreNum_ = 0;
    uint32_t mainCoreRow_ = 0;
    uint32_t curCoreRow_ = 0; // 当前核处理的总行数
    uint32_t inBufferSize_ = 0; // bytes
    uint32_t biasBufferSize_ = 0;
    uint32_t wsBufferSize_ = 0;
    uint32_t curCoreLoopRow_ = 0; // 当前核每次循环能处理的行数
    uint32_t curCoreTailLoopRow_ = 0; // 尾行
    uint32_t curCoreLoops_ = 0; // 当前核的总循环次数

    bool oneBlockMore_ = false;
};

template <typename XTYPE, typename WSTYPE, typename BIASTYPE, typename YTYPE, bool IFBIAS>
__aicore__ inline void DequantBiasMultiImpl<XTYPE, WSTYPE, BIASTYPE, YTYPE, IFBIAS>::ParseBaseTilingData(
    const DequantBiasTilingData* tilingData)
{
    N_ = tilingData->N;
    nAlign_ = tilingData->nAlign;
    asExist_ = tilingData->asExist;
    needCoreNum_ = tilingData->needCoreNum;
    mainCoreRow_ = tilingData->perCoreRow;
    inBufferSize_ = 65536;
    wsBufferSize_ = tilingData->wsBufferSize;
    biasBufferSize_ = tilingData->biasBufferSize;
    if (blockIdx_ == needCoreNum_ - 1) {
        curCoreRow_ = tilingData->tailCoreRow;
        curCoreLoopRow_ = 1;
        curCoreTailLoopRow_ = 1;
        curCoreLoops_ = tilingData->tailCoreLoops;
    } else {
        curCoreRow_ = tilingData->perCoreRow;
        curCoreLoopRow_ = 1;
        curCoreTailLoopRow_ = 1;
        curCoreLoops_ = tilingData->perCoreLoops;
    }

    if ((nAlign_ - N_) * sizeof(XTYPE) >= BLOCK_SIZE) {
        oneBlockMore_ = true;
    }
}

template <typename XTYPE, typename WSTYPE, typename BIASTYPE, typename YTYPE, bool IFBIAS>
__aicore__ inline void DequantBiasMultiImpl<XTYPE, WSTYPE, BIASTYPE, YTYPE, IFBIAS>::Init(GM_ADDR x,
    GM_ADDR weightScale, GM_ADDR activateScale,
    GM_ADDR bias, GM_ADDR y, const DequantBiasTilingData* tilingData)
{
    // assert tilingData->N > 8192
    blockIdx_ = GetBlockIdx();
    ParseBaseTilingData(tilingData);
    if (blockIdx_ >= needCoreNum_) {
        return;
    }

    xGM_.SetGlobalBuffer((__gm__ XTYPE*)x);
    weightScaleGM_.SetGlobalBuffer((__gm__ WSTYPE*)weightScale);
    yGM_.SetGlobalBuffer((__gm__ YTYPE*)y);

    pipe_.InitBuffer(wsBuf_, wsBufferSize_);
    pipe_.InitBuffer(pingBuf_, inBufferSize_ / 2);
    pipe_.InitBuffer(pongBuf_, inBufferSize_ / 2);

    if constexpr(IFBIAS == true) {
        biasGM_.SetGlobalBuffer((__gm__ BIASTYPE*)bias);
        pipe_.InitBuffer(biasBuf_, biasBufferSize_);
    }

    if (asExist_ != 0) {
        activeScaleGM_.SetGlobalBuffer((__gm__ float*)activateScale);
        pipe_.InitBuffer(asBuf_, tilingData->asBufferSize);
    }
}

template <typename XTYPE, typename WSTYPE, typename BIASTYPE, typename YTYPE, bool IFBIAS>
__aicore__ inline void DequantBiasMultiImpl<XTYPE, WSTYPE, BIASTYPE, YTYPE, IFBIAS>::Process()
{
    if (blockIdx_ >= needCoreNum_) {
        return;
    }

    // activate scale
    DataCopyInActivateScale();
    PipeBarrier<PIPE_ALL>();

    int64_t startRowOffset = blockIdx_ * mainCoreRow_;
    for (int64_t colIdx = 0; colIdx < (N_ + PART_LEN - 1) / PART_LEN; colIdx++) {
        int64_t nFullParts = N_ / PART_LEN;
        int64_t nProcess = colIdx == nFullParts ? N_ % PART_LEN : PART_LEN;
        int64_t weightOffset = colIdx * PART_LEN;

        // update weight_scale and bias
        PipeBarrier<PIPE_ALL>();
        DataCopyInBias(weightOffset, nProcess);
        DataCopyInWeightScale(weightOffset, nProcess);
        PipeBarrier<PIPE_ALL>();

        if (curCoreLoops_ > 0) {
            SetFlag<HardEvent::MTE3_MTE2>(pingId_);
            if (curCoreLoops_ > 1) {
                SetFlag<HardEvent::MTE3_MTE2>(pongId_);
            }
        }

        for (uint32_t idx = 0; idx < curCoreLoops_; idx++) {
            // process one line per loop
            auto pipeId = (idx % 2 == 0) ? pingId_ : pongId_;
            LocalTensor<XTYPE> xLocal = (idx % 2 == 0) ? pingBuf_.Get<XTYPE>() : pongBuf_.Get<XTYPE>();
            int64_t rowsOffset = idx;
            int64_t groupOffset = (startRowOffset + idx * curCoreLoopRow_) * N_ + colIdx * PART_LEN;

            WaitFlag<HardEvent::MTE3_MTE2>(pipeId);
            CopyIn(xLocal, groupOffset, nProcess);
            SetFlag<HardEvent::MTE2_V>(pipeId);

            WaitFlag<HardEvent::MTE2_V>(pipeId);
            ComputeDequant(xLocal, rowsOffset, groupOffset, nProcess);
            SetFlag<HardEvent::V_MTE3>(pipeId);

            WaitFlag<HardEvent::V_MTE3>(pipeId);
            CopyOut(xLocal, groupOffset, nProcess);
            if (idx + 2 < curCoreLoops_) {
                SetFlag<HardEvent::MTE3_MTE2>(pipeId);
            }
        }
    }
}

template <typename XTYPE, typename WSTYPE, typename BIASTYPE, typename YTYPE, bool IFBIAS>
__aicore__ inline void DequantBiasMultiImpl<XTYPE, WSTYPE, BIASTYPE, YTYPE, IFBIAS>::CopyIn(LocalTensor<XTYPE>& xLocal,
    int64_t groupOffset, int64_t nProcess)
{
    DataCopyExtParams extParams;
    DataCopyPadExtParams<XTYPE> padParams {false, 0, 0, 0};
    extParams.blockCount = 1;
    extParams.blockLen = nProcess * sizeof(XTYPE);
    extParams.srcStride = 0;
    extParams.dstStride = 0;
    DataCopyPad(xLocal, xGM_[groupOffset], extParams, padParams);
}

template <typename XTYPE, typename WSTYPE, typename BIASTYPE, typename YTYPE, bool IFBIAS>
__aicore__ inline void DequantBiasMultiImpl<XTYPE, WSTYPE, BIASTYPE, YTYPE, IFBIAS>::CopyOut(LocalTensor<XTYPE>& xLocal,
    int64_t groupOffset, int64_t nProcess)
{
    LocalTensor<YTYPE> yLocal = xLocal.template ReinterpretCast<YTYPE>();
    DataCopyExtParams extParams;
    extParams.blockCount = 1;
    extParams.blockLen = nProcess * sizeof(YTYPE);
    extParams.srcStride = 0; // oneBlockMore_ ? 1 : 0;
    extParams.dstStride = 0;
    DataCopyPad(yGM_[groupOffset], yLocal, extParams);
}

template <typename XTYPE, typename WSTYPE, typename BIASTYPE, typename YTYPE, bool IFBIAS>
__aicore__ inline void DequantBiasMultiImpl<XTYPE, WSTYPE, BIASTYPE, YTYPE, IFBIAS>::ComputeDequant(
    LocalTensor<XTYPE>& xLocal, int64_t rowsOffset, int64_t groupOffset, int64_t nProcess)
{
    if constexpr(IFBIAS == true) {
        if constexpr(IsSameType<BIASTYPE, int32_t>::value) {
            ComputeDequantWithBiasInt32(xLocal, rowsOffset, groupOffset, nProcess);
        } else {
            ComputeDequantWithBiasFloat(xLocal, rowsOffset, groupOffset, nProcess);
        }
    } else {
        ComputeDequantWithoutBias(xLocal, rowsOffset, groupOffset, nProcess);
    }
}

template <typename XTYPE, typename WSTYPE, typename BIASTYPE, typename YTYPE, bool IFBIAS>
__aicore__ inline void DequantBiasMultiImpl<XTYPE, WSTYPE, BIASTYPE, YTYPE, IFBIAS>::ComputeDequantWithBiasFloat(
    LocalTensor<XTYPE>& xLocal, int64_t rowsOffset, int64_t groupOffset, int64_t nProcess)
{
    int64_t nAlign = ((nProcess * sizeof(float)) + 63) / 64 * 64 / sizeof(float);

    LocalTensor<float> xLocalFp32 = xLocal.template ReinterpretCast<float>();
    Cast(xLocalFp32, xLocal, RoundMode::CAST_NONE, nAlign * 1);
    PipeBarrier<PIPE_V>();

    Mul(xLocalFp32, xLocalFp32, wsLocal_, nAlign);

    PipeBarrier<PIPE_V>();

    if (asExist_ != 0) {
        Muls(xLocalFp32, xLocalFp32, asLocal_.GetValue(rowsOffset), nAlign);
        PipeBarrier<PIPE_V>();
    }
    PipeBarrier<PIPE_V>();
    Add(xLocalFp32, xLocalFp32, biasLocal_.template ReinterpretCast<float>(), nAlign);
    PipeBarrier<PIPE_V>();
    LocalTensor<YTYPE> yLocal = xLocal.template ReinterpretCast<YTYPE>();
    Cast(yLocal, xLocalFp32, RoundMode::CAST_RINT, nAlign * 1);
    PipeBarrier<PIPE_V>();
}

template <typename XTYPE, typename WSTYPE, typename BIASTYPE, typename YTYPE, bool IFBIAS>
__aicore__ inline void DequantBiasMultiImpl<XTYPE, WSTYPE, BIASTYPE, YTYPE, IFBIAS>::ComputeDequantWithBiasInt32(
    LocalTensor<XTYPE>& xLocal, int64_t rowsOffset, int64_t groupOffset, int64_t nProcess)
{
    int64_t nAlign = ((nProcess * sizeof(float)) + 63) / 64 * 64 / sizeof(float);

    Add(xLocal, xLocal, biasLocal_, nAlign);
    PipeBarrier<PIPE_V>();
    LocalTensor<float> xLocalFp32 = xLocal.template ReinterpretCast<float>();
    Cast(xLocalFp32, xLocal, RoundMode::CAST_NONE, 1 * nAlign);
    PipeBarrier<PIPE_V>();
    Mul(xLocalFp32, xLocalFp32, wsLocal_, nAlign);
    PipeBarrier<PIPE_V>();
    if (asExist_ != 0) {
        Muls(xLocalFp32, xLocalFp32, asLocal_.GetValue(rowsOffset), nAlign);
        PipeBarrier<PIPE_V>();
    }
    PipeBarrier<PIPE_V>();
    LocalTensor<YTYPE> yLocal = xLocal.template ReinterpretCast<YTYPE>();
    Cast(yLocal, xLocalFp32, RoundMode::CAST_RINT, 1 * nAlign);
    PipeBarrier<PIPE_V>();
}

template <typename XTYPE, typename WSTYPE, typename BIASTYPE, typename YTYPE, bool IFBIAS>
__aicore__ inline void DequantBiasMultiImpl<XTYPE, WSTYPE, BIASTYPE, YTYPE, IFBIAS>::ComputeDequantWithoutBias(
    LocalTensor<XTYPE>& xLocal, int64_t rowsOffset, int64_t groupOffset, int64_t nProcess)
{
    int64_t nAlign = ((nProcess * sizeof(float)) + 63) / 64 * 64 / sizeof(float);

    LocalTensor<float> xLocalFp32 = xLocal.template ReinterpretCast<float>();
    Cast(xLocalFp32, xLocal, RoundMode::CAST_NONE, 1 * nAlign);
    PipeBarrier<PIPE_V>();
    Mul(xLocalFp32, xLocalFp32, wsLocal_, nAlign);
    PipeBarrier<PIPE_V>();
    if (asExist_ != 0) {
        Muls(xLocalFp32, xLocalFp32, asLocal_.GetValue(rowsOffset), nAlign * 1);
        PipeBarrier<PIPE_V>();
    }
    PipeBarrier<PIPE_V>();
    LocalTensor<YTYPE> yLocal = xLocal.template ReinterpretCast<YTYPE>();
    Cast(yLocal, xLocalFp32, RoundMode::CAST_RINT, nAlign * 1);
    PipeBarrier<PIPE_V>();
}


template <typename XTYPE, typename WSTYPE, typename BIASTYPE, typename YTYPE, bool IFBIAS>
__aicore__ inline void DequantBiasMultiImpl<XTYPE, WSTYPE, BIASTYPE, YTYPE, IFBIAS>::DataCopyInActivateScale()
{
    // activate scale 是 fp32
    if (asExist_ != 0) {
        asLocal_ = asBuf_.Get<float>();
        DataCopyPadExtParams<float> asPadParams {false, 0, 0, 0};
        DataCopyExtParams extParams {1, static_cast<uint32_t>(curCoreRow_ * sizeof(float)), 0, 0, 0};
        DataCopyPad(asLocal_, activeScaleGM_[mainCoreRow_ * blockIdx_], extParams, asPadParams);
    }
}

template <typename XTYPE, typename WSTYPE, typename BIASTYPE, typename YTYPE, bool IFBIAS>
__aicore__ inline void DequantBiasMultiImpl<XTYPE, WSTYPE, BIASTYPE, YTYPE, IFBIAS>::DataCopyInBias(int64_t offset, int64_t nProcess)
{
    if constexpr(IFBIAS == true) {
        // bias是int32
        if constexpr(IsSameType<BIASTYPE, int32_t>::value) {
            DataCopyPadExtParams<BIASTYPE> biasPadParams {false, 0, 0, 0};
            DataCopyExtParams biasParams {1, static_cast<uint32_t>(nProcess * sizeof(BIASTYPE)), 0, 0, 0};
            biasLocal_ = biasBuf_.Get<BIASTYPE>();
            DataCopyPad(biasLocal_, biasGM_[offset], biasParams, biasPadParams);
        } else {
            // bias是fp16/bf16/fp32
            DataCopyInBiasFloat(offset, nProcess);
        }
    }
}

template <typename XTYPE, typename WSTYPE, typename BIASTYPE, typename YTYPE, bool IFBIAS>
__aicore__ inline void DequantBiasMultiImpl<XTYPE, WSTYPE, BIASTYPE, YTYPE, IFBIAS>::DataCopyInWeightScale(int64_t offset, int64_t nProcess)
{
    wsLocal_ = wsBuf_.Get<float>();
    int64_t wsTypeNum = 0;
    DataCopyExtParams wsParams {1, static_cast<uint32_t>(nProcess * sizeof(WSTYPE)), 0, 0, 0};
    if constexpr(!IsSameType<WSTYPE, float>::value) {
        DataCopyPadExtParams<WSTYPE> padParams {false, 0, 0, 0};
        wsTypeNum = wsBufferSize_ / sizeof(WSTYPE) / 2;
        LocalTensor<WSTYPE> tmpBuf = wsBuf_.GetWithOffset<WSTYPE>(wsTypeNum, wsBufferSize_ / 2);
        DataCopyPad(tmpBuf, weightScaleGM_[offset], wsParams, padParams);
        event_t eventMTE2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventMTE2V);
        WaitFlag<HardEvent::MTE2_V>(eventMTE2V);
        Cast(wsLocal_, tmpBuf, RoundMode::CAST_NONE, nProcess);
        PipeBarrier<PIPE_V>();
    } else {
        DataCopyPadExtParams<float> padParams {false, 0, 0, 0};
        DataCopyPad(wsLocal_, weightScaleGM_[offset], wsParams, padParams);
    }
}

template <typename XTYPE, typename WSTYPE, typename BIASTYPE, typename YTYPE, bool IFBIAS>
__aicore__ inline void DequantBiasMultiImpl<XTYPE, WSTYPE, BIASTYPE, YTYPE, IFBIAS>::DataCopyInBiasFloat(int64_t offset, int64_t nProcess)
{
    DataCopyExtParams biasParams {1, static_cast<uint32_t>(nProcess * sizeof(BIASTYPE)), 0, 0, 0};
    biasLocal_ = biasBuf_.Get<BIASTYPE>();
    if constexpr(!IsSameType<BIASTYPE, float>::value) {
        DataCopyPadExtParams<BIASTYPE> biasPadParams {false, 0, 0, 0};
        int64_t biasTypeNum = biasBufferSize_ / sizeof(BIASTYPE) / 2;
        LocalTensor<BIASTYPE> tmpBuf = biasBuf_.GetWithOffset<BIASTYPE>(biasTypeNum, biasBufferSize_ / 2);
        DataCopyPad(tmpBuf, biasGM_[offset], biasParams, biasPadParams);
        event_t eventMTE2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventMTE2V);
        WaitFlag<HardEvent::MTE2_V>(eventMTE2V);
        Cast(biasLocal_.template ReinterpretCast<float>(), tmpBuf, RoundMode::CAST_NONE, nProcess);
        PipeBarrier<PIPE_V>();
    } else {
        DataCopyPadExtParams<float> biasPadParams {false, 0, 0, 0};
        DataCopyPad(biasLocal_, biasGM_[offset], biasParams, biasPadParams);
    }
}

} // namespace

#endif // DEQUANT_BIAS_IMPL_H