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

#ifndef INDEX_FILL_SIMD_IMPL_H
#define INDEX_FILL_SIMD_IMPL_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "../inc/kernel_utils.h"
#include "index_fill_struct.h"

namespace IndexFill {
using namespace AscendC;

constexpr int32_t DOUBLE_BUFFER = 2;
constexpr int32_t MASK_THRESHPLD = 512;

template<typename T, typename INDEX_TYPE, typename COM_T>
class IndexFillSimdImpl {
public:
    __aicore__ inline IndexFillSimdImpl(const IndexFillSimdTilingData& tilingData, TPipe& pipe) :
        tilingData_(tilingData), pipe_(pipe) {};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR indices, GM_ADDR value, GM_ADDR y, GM_ADDR workspace);
    __aicore__ inline void Process();

private:
    static __simt_vf__ __aicore__ inline void IndexFillSimtMaskFill(uint64_t blockIdx, uint64_t blockNum, __gm__ INDEX_TYPE* indices, __gm__ int8_t* mask, COM_T indicesNum, uint64_t n);
    __aicore__ inline void InitMask(uint64_t length);
    __aicore__ inline void ProcessCopy();
    __aicore__ inline int8_t GetMask(uint64_t startIdx, uint64_t endIdx);

private:
    AscendC::GlobalTensor<T> xGm_;
    AscendC::GlobalTensor<INDEX_TYPE> indicesGm_;
    AscendC::GlobalTensor<T> varGm_;
    AscendC::GlobalTensor<T> yGm_;
    AscendC::GlobalTensor<int8_t> maskGm_;

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, DOUBLE_BUFFER> xQueue_;
    TQue<QuePosition::VECOUT, DOUBLE_BUFFER> valueQueue_;
    TBuf<QuePosition::VECCALC> indicesBuf_;
    TPipe& pipe_;
    const IndexFillSimdTilingData& tilingData_;
    uint64_t blockIdx_;
    uint64_t blockNum_;
    uint64_t indicesOffsetBase_;
    uint64_t curIndexSize_;
};

__aicore__ inline static uint64_t Div(uint64_t value, uint64_t factor)
{
    if (factor == 0) {
        return 0;
    }
    return value / factor;
}

__aicore__ inline static uint64_t Mod(uint64_t value, uint64_t factor)
{
    if (factor == 0) {
        return 0;
    }
    return value % factor;
}

template<typename T, typename INDEX_TYPE, typename COM_T>
__aicore__ inline void IndexFillSimdImpl<T, INDEX_TYPE, COM_T>::InitMask(uint64_t length)
{
    __gm__ int8_t* basePtr = (__gm__ int8_t*)maskGm_.GetPhyAddr();

    int8_t zero = 0;
    if (length <= MASK_THRESHPLD) {
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

template<typename T, typename INDEX_TYPE, typename COM_T>
__aicore__ inline void IndexFillSimdImpl<T, INDEX_TYPE, COM_T>::Init(GM_ADDR x, GM_ADDR indices, GM_ADDR value, GM_ADDR y, GM_ADDR workspace)
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
    indicesOffsetBase_ = -1;
    curIndexSize_ = 0;

    pipe_.InitBuffer(xQueue_, DOUBLE_BUFFER, tilingData_.blockFactorUbFactorQ * sizeof(T));
    pipe_.InitBuffer(valueQueue_, 1, tilingData_.blockFactorUbFactorQ * sizeof(T));
    pipe_.InitBuffer(indicesBuf_, tilingData_.blockFactorUbBufferMask * sizeof(int8_t));  //分配固定大小索引缓冲区

    InitMask(tilingData_.n);
    SyncAll();
}

template<typename T, typename INDEX_TYPE, typename COM_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void IndexFillSimdImpl<T, INDEX_TYPE, COM_T>::IndexFillSimtMaskFill(uint64_t blockIdx, uint64_t blockNum,
__gm__ INDEX_TYPE* indices, __gm__ int8_t* mask, COM_T indicesNum, uint64_t n)
{
    COM_T threadIdx = static_cast<COM_T>(blockIdx * Simt::GetThreadNum() + Simt::GetThreadIdx());
    COM_T threadNum = static_cast<COM_T>(blockNum * Simt::GetThreadNum());
    for (COM_T idx = threadIdx; idx < indicesNum; idx += threadNum) {
        INDEX_TYPE nIdx = static_cast<INDEX_TYPE>(indices[idx]);
        nIdx = nIdx >= 0 ? nIdx : nIdx + n;
        if (nIdx < 0 || nIdx >= n) {
            continue;
        }
        mask[nIdx] = static_cast<int8_t>(1);
    }
}

template <typename T>
__aicore__ inline void CommonCopyIn(LocalTensor<T> dstLocal, GlobalTensor<T> srcGm, uint64_t offset, uint64_t copyLen)
{
    DataCopyExtParams dataCoptExtParams = {1, static_cast<uint32_t>(copyLen * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> dataCopyPadExtParams = {false, 0, 0, 0};
    DataCopyPad(dstLocal, srcGm[offset], dataCoptExtParams, dataCopyPadExtParams);
}

template <typename T>
__aicore__ inline void CommonCopyOut(GlobalTensor<T> dstGm, LocalTensor<T> srcLocal, uint64_t offset, uint64_t copyLen)
{
    DataCopyExtParams dataCoptExtParams = {1, static_cast<uint32_t>(copyLen * sizeof(T)), 0, 0, 0};
    DataCopyPad(dstGm[offset], srcLocal, dataCoptExtParams);
}

template<typename T, typename INDEX_TYPE, typename COM_T>
__aicore__ inline int8_t IndexFillSimdImpl<T, INDEX_TYPE, COM_T>::GetMask(uint64_t startIdx, uint64_t endIdx)
{
    uint64_t startNIdx = startIdx % tilingData_.n;
    //UB中没有所需索引，需重新拷贝
    if (startNIdx >= indicesOffsetBase_ + curIndexSize_ || startNIdx < indicesOffsetBase_) {
        uint64_t startPIdx = Div(startIdx, tilingData_.n);
        uint64_t endPIdx = Div(endIdx, tilingData_.n);
        uint64_t endNIdx = endIdx % tilingData_.n;
        uint64_t copyLen = 0;
        if (tilingData_.blockFactorUbBufferMask >= tilingData_.n) {
            //拷贝全部索引
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
            //拷贝部分索引
            if (endPIdx == startPIdx) {
                copyLen = min(static_cast<uint64_t>(endNIdx - startNIdx + 1), static_cast<uint64_t>(tilingData_.blockFactorUbBufferMask));
                indicesOffsetBase_ = startNIdx;
            } else {
                copyLen = min(static_cast<uint64_t>(tilingData_.n - startNIdx), static_cast<uint64_t>(tilingData_.blockFactorUbBufferMask));
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

template<typename T, typename INDEX_TYPE, typename COM_T>
__aicore__ inline void IndexFillSimdImpl<T, INDEX_TYPE, COM_T>::ProcessCopy()
{
    T valScalar = varGm_.GetValue(0);
    event_t eventIdStoV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdStoV);
    WaitFlag<HardEvent::S_V>(eventIdStoV);

    LocalTensor<T> valLocal = valueQueue_.AllocTensor<T>();
    Duplicate(valLocal, valScalar, tilingData_.blockFactorUbFactorQ);
    valueQueue_.EnQue<T>(valLocal);
    valLocal = valueQueue_.DeQue<T>();

    if (tilingData_.splitQ) {
        uint64_t qCoreId = blockIdx_ % tilingData_.usedCoreNumQ;
        uint64_t pnIdx = Div(blockIdx_, tilingData_.usedCoreNumQ);
        uint64_t nIdx = Mod(pnIdx, tilingData_.n);
        uint64_t dstOffset = pnIdx * tilingData_.q + qCoreId * tilingData_.blockFactorQ;

        int64_t tileNumQ = tilingData_.blockFactorTileNumQ;
        int64_t ubFactorQ = tilingData_.blockFactorUbFactorQ;
        int64_t ubFactorTailQ = tilingData_.blockFactorUbTailQ;
        if (qCoreId == tilingData_.usedCoreNumQ - 1) {
            // 如果是Q的尾核
            tileNumQ = tilingData_.blockTailTileNumQ;
            ubFactorQ = tilingData_.blockTailUbFactorQ;
            ubFactorTailQ = tilingData_.blockTailUbTailQ;
        }

        if (maskGm_.GetValue(nIdx) == 1) {
            for (uint64_t i = 0; i < tileNumQ - 1; i++) {
                CommonCopyOut(yGm_, valLocal, dstOffset, ubFactorQ);
                dstOffset += ubFactorQ;
            }
            CommonCopyOut(yGm_, valLocal, dstOffset, ubFactorTailQ);
        } else {
            for (uint64_t i = 0; i < tileNumQ - 1; i++) {
                LocalTensor<T> xLocal = xQueue_.AllocTensor<T>();
                CommonCopyIn(xLocal, xGm_, dstOffset, ubFactorQ);
                xQueue_.EnQue<T>(xLocal);

                xLocal = xQueue_.DeQue<T>();
                CommonCopyOut(yGm_, xLocal, dstOffset, ubFactorQ);
                xQueue_.FreeTensor(xLocal);

                dstOffset += ubFactorQ;
            }

            LocalTensor<T> xLocal = xQueue_.AllocTensor<T>();
            CommonCopyIn(xLocal, xGm_, dstOffset, ubFactorTailQ);
            xQueue_.EnQue<T>(xLocal);

            xLocal = xQueue_.DeQue<T>();
            CommonCopyOut(yGm_, xLocal, dstOffset, ubFactorTailQ);
            xQueue_.FreeTensor(xLocal);
        }
    } else {
        uint64_t endIdx = 0;
        uint64_t startIdx = 0;
        if (blockIdx_ < tilingData_.tailBlockNumPN) {
            startIdx = blockIdx_ * (tilingData_.blockFactorPN + 1);
            endIdx = startIdx + tilingData_.blockFactorPN + 1;
        } else {
            startIdx = tilingData_.tailBlockNumPN * (tilingData_.blockFactorPN + 1) + (blockIdx_ - tilingData_.tailBlockNumPN) * tilingData_.blockFactorPN;
            endIdx = startIdx + tilingData_.blockFactorPN;
        }

        for (uint64_t idx = startIdx; idx < endIdx; idx++) {
            uint64_t pnIdx = Div(idx, tilingData_.n);
            uint64_t nIdx = idx % tilingData_.n;
            uint64_t dstOffset = idx * tilingData_.q;
            int8_t needReplace = GetMask(idx, endIdx);
            if (needReplace) {
                for (uint64_t i = 0; i < tilingData_.blockFactorTileNumQ - 1; i++) {
                    CommonCopyOut(yGm_, valLocal, dstOffset, tilingData_.blockFactorUbFactorQ);
                    dstOffset += tilingData_.blockFactorUbFactorQ;
                }
                // 尾块处理
                CommonCopyOut(yGm_, valLocal, dstOffset, tilingData_.blockFactorUbTailQ);
            } else {
                for (uint64_t i = 0; i < tilingData_.blockFactorTileNumQ - 1; i++) {
                    LocalTensor<T> xLocal = xQueue_.AllocTensor<T>();
                    CommonCopyIn(xLocal, xGm_, dstOffset, tilingData_.blockFactorUbFactorQ);
                    xQueue_.EnQue<T>(xLocal);

                    xLocal = xQueue_.DeQue<T>();
                    CommonCopyOut(yGm_, xLocal, dstOffset, tilingData_.blockFactorUbFactorQ);
                    xQueue_.FreeTensor(xLocal);

                    dstOffset += tilingData_.blockFactorUbFactorQ;
                }

                // 尾块处理
                LocalTensor<T> xLocal = xQueue_.AllocTensor<T>();
                CommonCopyIn(xLocal, xGm_, dstOffset, tilingData_.blockFactorUbTailQ);
                xQueue_.EnQue<T>(xLocal);

                xLocal = xQueue_.DeQue<T>();
                CommonCopyOut(yGm_, xLocal, dstOffset, tilingData_.blockFactorUbTailQ);
                xQueue_.FreeTensor(xLocal);
            }
        }
    }
}

template<typename T, typename INDEX_TYPE, typename COM_T>
__aicore__ inline void IndexFillSimdImpl<T, INDEX_TYPE, COM_T>::Process()
{
    if (GetBlockIdx() >= GetBlockNum()) {
        return;
    }
    uint64_t n = tilingData_.n;
    COM_T indicesNum = static_cast<COM_T>(tilingData_.indicesNum);
    AscendC::Simt::VF_CALL<IndexFillSimdImpl<T, INDEX_TYPE, COM_T>::IndexFillSimtMaskFill>(
        AscendC::Simt::Dim3(THREAD_NUM), blockIdx_, blockNum_, (__gm__ INDEX_TYPE*)(indicesGm_.GetPhyAddr()), (__gm__ int8_t*)(maskGm_.GetPhyAddr()), indicesNum, n);
    SyncAll();

    ProcessCopy();
}
}
#endif  // INDEX_FILL_SIMD_IMPL_H