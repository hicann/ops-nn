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
 * \file index_fill_simt.h
 * \brief
 */

#ifndef CANN_INDEX_FILL_SIMT_H
#define CANN_INDEX_FILL_SIMT_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "index_fill_struct.h"

namespace IndexFill {
using namespace AscendC;
constexpr uint32_t THREAD_NUM = 2048;
constexpr uint8_t DB_BUFFER = 2;

template <typename T, typename INDEX_TYPE, typename COM_T>
 __simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void IndexFillSimtCompute(__gm__ T* y, __gm__ INDEX_TYPE* indices, T value, COM_T total_num, COM_T p, COM_T n, COM_T q, 
     COM_T slice_size, COM_T shift, COM_T magic, COM_T shift_q, COM_T magic_q);

template <typename T, typename INDEX_TYPE, typename COM_T>
class IndexFillSimtImpl {
public:
    __aicore__ inline IndexFillSimtImpl(const IndexFillSimtTilingData* __restrict tilingData, TPipe *pipe): pipe_(pipe), tilingData_(tilingData), blockIdx_(GetBlockIdx()), blockNum_(GetBlockNum()) {
    }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR indices, GM_ADDR value);
    __aicore__ inline void Process(__gm__ T* y);

private:
    __aicore__ inline void CopyTensor();
    __aicore__ inline void CopyIn(int64_t offset, int64_t dataLen);
    __aicore__ inline void CopyOut(int64_t offset, int64_t dataLen);

private:
    TPipe *pipe_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, DB_BUFFER> dataQueue_;
    AscendC::GlobalTensor<T> xGm_;
    AscendC::GlobalTensor<T> yGm_;
    AscendC::GlobalTensor<T> valueGm_;
    AscendC::GlobalTensor<INDEX_TYPE> indices_;
    const IndexFillSimtTilingData* tilingData_;
    uint32_t blockIdx_ = 0;
    uint64_t offset_ = 0;
    uint32_t blockNum_ = 0;
    T fillValue;
};

template <typename T, typename INDEX_TYPE, typename COM_T>
__aicore__ inline void IndexFillSimtImpl<T, INDEX_TYPE, COM_T>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR indices, GM_ADDR value)
{
    valueGm_.SetGlobalBuffer((__gm__ T*)(value));
    indices_.SetGlobalBuffer((__gm__ INDEX_TYPE*)(indices));
    fillValue = valueGm_.GetValue(0);

    // 只有当前核的id位于[0, simdUsedCoreNum)内时，才需下面搬运相关的初始化逻辑.
    if (blockIdx_ >= tilingData_->simdUsedCoreNum) {
        return;
    }

    if (blockIdx_ < tilingData_->frontCoreNum) {
        offset_ = blockIdx_ * tilingData_->loopsPerFrontCore * tilingData_->blockSize;
    } else {
        offset_ = (tilingData_->frontCoreNum * tilingData_->loopsPerFrontCore * tilingData_->blockSize) + ((blockIdx_ - tilingData_->frontCoreNum) * tilingData_->loopsPerTailCore * tilingData_->blockSize);
    }

    xGm_.SetGlobalBuffer((__gm__ T*)(x) + offset_);
    yGm_.SetGlobalBuffer((__gm__ T*)(y) + offset_);
    pipe_->InitBuffer(dataQueue_, DB_BUFFER, tilingData_->blockSize * sizeof(T));
}

template <typename T, typename INDEX_TYPE, typename COM_T>
__aicore__ inline void IndexFillSimtImpl<T, INDEX_TYPE, COM_T>::CopyIn(int64_t offset, int64_t dataLen)
{
  DataCopyExtParams inParams = {
    static_cast<uint16_t>(1),
    static_cast<uint32_t>(dataLen * sizeof(T)),
    static_cast<uint32_t>(0),
    static_cast<uint32_t>(0),
    static_cast<uint32_t>(0)
  };
  DataCopyPadExtParams<T> padParams = { false, static_cast<uint8_t>(0), static_cast<uint8_t>(0), static_cast<T>(0) };
  LocalTensor<T> xLocal = dataQueue_.AllocTensor<T>();
  DataCopyPad(xLocal, xGm_[offset], inParams, padParams);
  dataQueue_.EnQue(xLocal);
}

template <typename T, typename INDEX_TYPE, typename COM_T>
__aicore__ inline void IndexFillSimtImpl<T, INDEX_TYPE, COM_T>::CopyOut(int64_t offset, int64_t dataLen)
{
  DataCopyExtParams outParams = {
    static_cast<uint16_t>(1),
    static_cast<uint32_t>(dataLen * sizeof(T)),
    static_cast<uint32_t>(0),
    static_cast<uint32_t>(0),
    static_cast<uint32_t>(0)
  };
  LocalTensor<T> yLocal = dataQueue_.DeQue<T>();
  DataCopyPad(yGm_[offset], yLocal, outParams);
  dataQueue_.FreeTensor(yLocal);
}

template <typename T, typename INDEX_TYPE, typename COM_T>
__aicore__ inline void IndexFillSimtImpl<T, INDEX_TYPE, COM_T>::CopyTensor()
{
    if (blockIdx_ >= tilingData_->simdUsedCoreNum) {
        return;
    }

    int64_t loopSize = blockIdx_ < tilingData_->frontCoreNum ? tilingData_->loopsPerFrontCore : tilingData_->loopsPerTailCore;
    int64_t offset = 0;
    for (int64_t idx = 0; idx < loopSize - 1; idx++) {
        offset = idx * tilingData_->blockSize;
        CopyIn(offset, tilingData_->blockSize);
        CopyOut(offset, tilingData_->blockSize);
    }

    // 尾块搬运
    offset = (loopSize - 1) * tilingData_->blockSize;
    int64_t dataLen = tilingData_->blockSize;
    if (blockIdx_ == tilingData_->simdUsedCoreNum - 1) {
        dataLen = tilingData_->tailBlockSize;
    }
    CopyIn(offset, dataLen);
    CopyOut(offset, dataLen);
}

template <typename T, typename INDEX_TYPE, typename COM_T>
__aicore__ inline void IndexFillSimtImpl<T, INDEX_TYPE, COM_T>::Process(__gm__ T* y)
{
    // 将x复制搬运到y上
    CopyTensor();

    // 同步等待
    SyncAll();

    COM_T p = static_cast<COM_T>(tilingData_->p);
    COM_T q = static_cast<COM_T>(tilingData_->q);
    COM_T n = static_cast<COM_T>(tilingData_->n);
    COM_T process_num = static_cast<COM_T>(tilingData_->indicesNum * p * q);

    COM_T slice_size = static_cast<COM_T>(p * q);
    COM_T shift = 0;
    COM_T magic = 0;
    GetUintDivMagicAndShift(magic, shift, slice_size);

    COM_T shift_q = 0;
    COM_T magic_q = 0;
    GetUintDivMagicAndShift(magic_q, shift_q, q);

    // 执行index_fill核心逻辑
    AscendC::Simt::VF_CALL<IndexFillSimtCompute<T, INDEX_TYPE, COM_T>>(AscendC::Simt::Dim3{THREAD_NUM}, y,
        (__gm__ INDEX_TYPE*)indices_.GetPhyAddr(), fillValue, process_num, p, n, q, slice_size, shift, magic, shift_q, magic_q);
}

template <typename T, typename INDEX_TYPE, typename COM_T>
 __simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void IndexFillSimtCompute(__gm__ T* y, __gm__ INDEX_TYPE* indices, T value, COM_T total_num, COM_T p, COM_T n, COM_T q, 
    COM_T slice_size, COM_T shift, COM_T magic, COM_T shift_q, COM_T magic_q)
{
    COM_T threadIdx = static_cast<COM_T>(Simt::GetBlockIdx() * Simt::GetThreadNum() + Simt::GetThreadIdx());
    COM_T threadNum = static_cast<COM_T>(Simt::GetBlockNum() * Simt::GetThreadNum());

    for (COM_T i = threadIdx; i < total_num; i += threadNum) {
        COM_T sliceId = Simt::UintDiv(i, magic, shift);
        COM_T innerId = i - (sliceId * slice_size);
        INDEX_TYPE nIdx = static_cast<INDEX_TYPE>(indices[sliceId]);
        nIdx = nIdx >= 0 ? nIdx : nIdx + n;
        if (nIdx < 0 || nIdx >= n) {
            continue;
        }
        COM_T pIdx = Simt::UintDiv(innerId, magic_q, shift_q);
        COM_T qOffset =  innerId - (pIdx * q);
        COM_T offset = pIdx * n * q + nIdx * q + qOffset;
        y[offset] = value;
    }
}

} // namespace IndexFill

#endif // CANN_INDEX_FILL_SIMT_H
