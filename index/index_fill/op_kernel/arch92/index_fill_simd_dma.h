/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License")
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file index_fill_simd.h
 * \brief simd kernel of index_fill
 */

#ifndef INDEX_FILL_SIMD_DMA_IMPL_H
#define INDEX_FILL_SIMD_DMA_IMPL_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "../inc/kernel_utils.h"
#include "../arch35/index_fill_struct.h"
#include "simt_api/asc_simt.h"
#include "../inc/platform.h"
#include "op_kernel/platform_util.h"
#include "op_kernel/math_util.h"
#include "../inc/load_store_utils.h"

namespace IndexFill {
using namespace AscendC;

template <typename T, typename INDEX_TYPE, typename COM_T>
class IndexFillSimdDmaImpl {
public:
    __aicore__ inline IndexFillSimdDmaImpl(const IndexFillSimdDmaTilingData& tilingData, TPipe& pipe)
        : tilingData_(tilingData), pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR indices, GM_ADDR value, GM_ADDR y, GM_ADDR workspace);
    __aicore__ inline void Process();

private:
    static __simt_vf__ __aicore__ inline void IndexFillSimtMaskFill(uint64_t blockIdx, uint64_t blockNum,
                                                                    __gm__ INDEX_TYPE* indices, __gm__ int8_t* mask,
                                                                    COM_T indicesNum, uint64_t n);
    __aicore__ inline void InitMask(uint64_t length);
    __aicore__ inline void ProcessCopy(T valScalar);
    __aicore__ inline int8_t GetMask(uint64_t startIdx, uint64_t endIdx);

private:
    AscendC::GlobalTensor<T> xGm_;
    AscendC::GlobalTensor<INDEX_TYPE> indicesGm_;
    AscendC::GlobalTensor<T> varGm_;
    AscendC::GlobalTensor<T> yGm_;
    AscendC::GlobalTensor<int8_t> maskGm_;

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, DOUBLE_BUFFER> xQueue_;
    TQue<QuePosition::VECIN, DOUBLE_BUFFER> indicesQueue_;
    TBuf<QuePosition::VECCALC> indicesBuf_;
    TPipe& pipe_;
    const IndexFillSimdDmaTilingData& tilingData_;
    uint64_t blockIdx_ = 0;
    uint64_t blockNum_ = 0;
    uint64_t indicesOffsetBase_ = 0;
    uint64_t curIndexSize_ = 0;
    uint64_t blockQNumLoop_ = 0;
    uint64_t blockQNumtail_ = 0;
    uint64_t blockQNum_ = 0;
};

template <typename T, typename INDEX_TYPE, typename COM_T>
__aicore__ inline void IndexFillSimdDmaImpl<T, INDEX_TYPE, COM_T>::InitMask(uint64_t length)
{
    __gm__ int8_t* basePtr = (__gm__ int8_t*)maskGm_.GetPhyAddr();

    int8_t zero = 0;
    if (length <= static_cast<uint64_t>(MASK_THRESHPLD) && blockIdx_ == 0) {
        AscendC::Fill(maskGm_, length, zero);
        return;
    }

    uint64_t usedCoreNum = blockNum_;
    uint64_t numelPerCore = length / usedCoreNum;
    uint64_t tailBlockNum = length - numelPerCore * usedCoreNum;
    if (numelPerCore <= 0) {
        usedCoreNum = tailBlockNum;
    }

    if (blockIdx_ >= usedCoreNum) {
        return;
    }

    uint64_t offset = 0;
    uint64_t len = 0;
    GlobalTensor<int8_t> subMask;
    if (blockIdx_ < tailBlockNum) {
        len = numelPerCore + 1;
        offset = blockIdx_ * (numelPerCore + 1);
    } else {
        len = numelPerCore;
        offset = tailBlockNum * (numelPerCore + 1) + (blockIdx_ - tailBlockNum) * numelPerCore;
    }

    subMask.SetGlobalBuffer(basePtr + offset, len);
    AscendC::Fill(subMask, len, zero);
}

template <typename T, typename INDEX_TYPE, typename COM_T>
__aicore__ inline void IndexFillSimdDmaImpl<T, INDEX_TYPE, COM_T>::Init(GM_ADDR x, GM_ADDR indices, GM_ADDR value,
                                                                        GM_ADDR y, GM_ADDR workspace)
{
    if (GetBlockIdx() >= GetBlockNum()) {
        return;
    }
    xGm_.SetGlobalBuffer((__gm__ T*)(x));
    indicesGm_.SetGlobalBuffer((__gm__ INDEX_TYPE*)(indices));
    varGm_.SetGlobalBuffer((__gm__ T*)(value));
    yGm_.SetGlobalBuffer((__gm__ T*)(y));
    maskGm_.SetGlobalBuffer((__gm__ int8_t*)(workspace), tilingData_.n * sizeof(int8_t));
    blockIdx_ = GetBlockIdx();
    blockNum_ = GetBlockNum();
    indicesOffsetBase_ = 0;
    curIndexSize_ = 0;

    pipe_.InitBuffer(indicesQueue_, DOUBLE_BUFFER, tilingData_.indicesUbFactorSizeAlign);
    pipe_.InitBuffer(xQueue_, DOUBLE_BUFFER, tilingData_.blockFactorAlignQ * sizeof(T));
    pipe_.InitBuffer(indicesBuf_, tilingData_.blockFactorUbBufferMask * sizeof(int8_t)); // 分配固定大小索引缓冲区

    InitMask(tilingData_.n);
    SyncAll();
}

template <typename T, typename INDEX_TYPE, typename COM_T>
__simt_vf__ __aicore__
LAUNCH_BOUND(THREAD_NUM) inline void IndexFillSimdDmaImpl<T, INDEX_TYPE, COM_T>::IndexFillSimtMaskFill(
    uint64_t blockIdx, uint64_t blockNum, __gm__ INDEX_TYPE* indices, __gm__ int8_t* mask, COM_T indicesNum, uint64_t n)
{
    COM_T threadIdxLocal = static_cast<COM_T>(blockIdx * blockDim.x + threadIdx.x);
    COM_T threadNum = static_cast<COM_T>(blockNum * blockDim.x);
    for (COM_T idx = threadIdxLocal; idx < indicesNum; idx += threadNum) {
        INDEX_TYPE nIdx = static_cast<INDEX_TYPE>(indices[idx]);
        nIdx = nIdx >= 0 ? nIdx : nIdx + n;
        if (nIdx < 0 || nIdx >= n) {
            continue;
        }
        mask[nIdx] = static_cast<int8_t>(1);
    }
}

template <typename T, typename INDEX_TYPE, typename COM_T>
__aicore__ inline int8_t IndexFillSimdDmaImpl<T, INDEX_TYPE, COM_T>::GetMask(uint64_t startIdx, uint64_t endIdx)
{
    uint64_t startNIdx = startIdx % tilingData_.n;
    // UB中没有所需索引，需重新拷贝
    if (startNIdx >= indicesOffsetBase_ + curIndexSize_ || startNIdx < indicesOffsetBase_) {
        uint64_t startPIdx = startIdx / tilingData_.n;
        uint64_t endPIdx = endIdx / tilingData_.n;
        uint64_t endNIdx = endIdx % tilingData_.n;
        uint64_t copyLen = 0;
        if (tilingData_.blockFactorUbBufferMask >= tilingData_.n) {
            // 拷贝全部索引
            if (endPIdx == startPIdx) {
                copyLen = endNIdx - startNIdx + 1;
                indicesOffsetBase_ = startNIdx;
            } else if ((endPIdx == startPIdx + 1) && (endNIdx < startNIdx)) {
                copyLen = tilingData_.n - startNIdx;
                indicesOffsetBase_ = startNIdx;
            } else {
                copyLen = tilingData_.n;
                indicesOffsetBase_ = 0;
            }
        } else {
            // 拷贝部分索引
            if (endPIdx == startPIdx) {
                copyLen = min((endNIdx - startNIdx + 1), tilingData_.blockFactorUbBufferMask);
                indicesOffsetBase_ = startNIdx;
            } else {
                copyLen = min((tilingData_.n - startNIdx), tilingData_.blockFactorUbBufferMask);
                indicesOffsetBase_ = startNIdx;
            }
        }

        curIndexSize_ = copyLen;
        LocalTensor<int8_t> tmpLocal = indicesBuf_.Get<int8_t>();
        CommonCopyIn<int8_t>(tmpLocal, maskGm_, indicesOffsetBase_, copyLen);

        event_t eventIdMTE2toS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(eventIdMTE2toS);
        WaitFlag<HardEvent::MTE2_S>(eventIdMTE2toS);
    }
    LocalTensor<int8_t> indicesLocal = indicesBuf_.Get<int8_t>();
    return indicesLocal.GetValue(startNIdx - indicesOffsetBase_);
}

template <typename T, typename INDEX_TYPE, typename COM_T>
__aicore__ inline void IndexFillSimdDmaImpl<T, INDEX_TYPE, COM_T>::ProcessCopy(T valScalar)
{
    uint64_t endIdx = 0;
    uint64_t startIdx = 0;
    if (blockIdx_ < tilingData_.tailBlockNumPN) {
        startIdx = blockIdx_ * (tilingData_.blockFactorPN + 1);
        endIdx = startIdx + tilingData_.blockFactorPN + 1;
        blockQNumLoop_ = tilingData_.normBlockQNumLoop;
        blockQNumtail_ = tilingData_.normBlockQNumTail;
    } else {
        startIdx = tilingData_.tailBlockNumPN + blockIdx_ * tilingData_.blockFactorPN;
        endIdx = startIdx + tilingData_.blockFactorPN;
        blockQNumLoop_ = tilingData_.tailBlockQNumLoop;
        blockQNumtail_ = tilingData_.tailBlockQNumTail;
    }

    for (uint64_t loop = 0; loop < blockQNumLoop_; loop++) {
        blockQNum_ = loop == blockQNumLoop_ - 1 ? blockQNumtail_ : tilingData_.indicesUbFactor;
        LocalTensor<int64_t> dstIndicesLocal = indicesQueue_.AllocTensor<int64_t>();
        uint64_t dstOffset = (startIdx + loop * tilingData_.indicesUbFactor) * tilingData_.q;
        LocalTensor<T> xLocal = xQueue_.AllocTensor<T>();
        Duplicate(xLocal, valScalar, tilingData_.blockFactorAlignQ);
        for (uint64_t idx = startIdx + loop * tilingData_.indicesUbFactor;
             idx < startIdx + loop * tilingData_.indicesUbFactor + blockQNum_; idx++) {
            int64_t srcOffset = static_cast<int64_t>(idx) * static_cast<int64_t>(tilingData_.q) * sizeof(T);
            int8_t needReplace = GetMask(idx, endIdx);
            uint64_t indicesLocalOffset = idx - (startIdx + loop * tilingData_.indicesUbFactor);
            dstIndicesLocal(indicesLocalOffset) = srcOffset;
            if (needReplace) {
                dstIndicesLocal(indicesLocalOffset) = -1;
            }
        }
        // 这里 eventIdSToMte2 是搬入前需要等待 s计算完所有的地址
        event_t eventIdSToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
        SetFlag<HardEvent::S_MTE2>(eventIdSToMte2);
        WaitFlag<HardEvent::S_MTE2>(eventIdSToMte2);
        // 这里的 eventIdVToMte2 是搬入之前要等待 vector 计算完所有内容
        event_t eventIdVToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);
        indicesQueue_.EnQue<int64_t>(dstIndicesLocal);
        dstIndicesLocal = indicesQueue_.DeQue<int64_t>();
        GatherParams gatherParams;
        gatherParams.offsetNum = blockQNum_;
        gatherParams.burstLen = tilingData_.q * sizeof(T);
        gatherParams.dstStride = tilingData_.q * sizeof(T);
        GatherPadParams<T> gatherPadparams(GatherPaddingMode::CONSTANT_PADDING_MODE, 0);
        pipe_barrier(PIPE_ALL);
        GatherDataCopy<T, int64_t>(xLocal, xGm_, dstIndicesLocal, gatherParams, gatherPadparams);
        xQueue_.EnQue<T>(xLocal);
        xLocal = xQueue_.DeQue<T>();
        CommonCopyOut(yGm_, xLocal, dstOffset, blockQNum_ * tilingData_.q);
        // 这里的 eventIdMte3ToV 是 vector 在duplicate 之前要等待 mte3 全部搬出所有内容
        event_t eventIdMte3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        WaitFlag<HardEvent::MTE3_V>(eventIdMte3ToV);
        xQueue_.FreeTensor(xLocal);
        indicesQueue_.FreeTensor(dstIndicesLocal);
    }
}

template <typename T, typename INDEX_TYPE, typename COM_T>
__aicore__ inline void IndexFillSimdDmaImpl<T, INDEX_TYPE, COM_T>::Process()
{
    if (GetBlockIdx() >= GetBlockNum()) {
        return;
    }
    uint64_t n = tilingData_.n;
    COM_T indicesNum = static_cast<COM_T>(tilingData_.indicesNum);
    asc_vf_call<IndexFillSimdDmaImpl<T, INDEX_TYPE, COM_T>::IndexFillSimtMaskFill>(
        dim3(THREAD_NUM), blockIdx_, blockNum_, (__gm__ INDEX_TYPE*)(indicesGm_.GetPhyAddr()),
        (__gm__ int8_t*)(maskGm_.GetPhyAddr()), indicesNum, n);
    T valScalar = varGm_.GetValue(0);
    if (tilingData_.fp4Mode) {
        valScalar = valScalar | ((valScalar << FOUR_BIT) & HIGH_FOUR_BIT);
    }
    SyncAll();

    ProcessCopy(valScalar);
}
} // namespace IndexFill
#endif // INDEX_FILL_SIMD_DMA_IMPL_H
