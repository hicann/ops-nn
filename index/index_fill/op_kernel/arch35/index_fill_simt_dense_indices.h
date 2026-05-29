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
 * \file index_fill_simt_dense_indices.h
 * \brief
 */

#ifndef CANN_INDEX_FILL_SIMT_DENSE_INDICES_H
#define CANN_INDEX_FILL_SIMT_DENSE_INDICES_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "index_fill_common.h"

namespace IndexFill {
using namespace AscendC;

template <typename T, typename INDEX_TYPE, typename COM_T>
 __simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void SetMaskSimt(__local_mem__ INDEX_TYPE* indicesLocalAddr, __local_mem__ int32_t* uniqueIndicesAddr, __gm__ int8_t* mask, COM_T processNum, COM_T n, int64_t length);

template <typename T, typename INDEX_TYPE, typename COM_T>
 __simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void IndexFillSimtDenseIndicesCompute(__gm__ T* y, __gm__ int8_t* mask, T value, COM_T total_num, COM_T n,
     COM_T shift_n, COM_T magic_n, COM_T shift_q, COM_T magic_q);

template <typename T, typename INDEX_TYPE, typename COM_T>
class IndexFillSimtDenseIndicesImpl {
public:
    __aicore__ inline IndexFillSimtDenseIndicesImpl(const IndexFillSimtDenseIndicesTilingData* __restrict tilingData, TPipe *pipe): pipe_(pipe), tilingData_(tilingData), blockIdx_(GetBlockIdx()), blockNum_(GetBlockNum()) {
    }

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR indices, GM_ADDR value, GM_ADDR workspace);
    __aicore__ inline void Process(__gm__ T* y);

private:
    __aicore__ inline void CopyTensor();
    __aicore__ inline void CopyIn(int64_t offset, int64_t dataLen);
    __aicore__ inline void CopyOut(int64_t offset, int64_t dataLen);
    __aicore__ inline void BuildIndicesMask();
    __aicore__ inline void CopyInIndices(int64_t offset, int64_t dataLen);
    __aicore__ inline void FillMaskWithUniqueIndices(int64_t indicesNum);

private:
    TPipe *pipe_;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, DB_BUFFER> dataQueue_;
    TQue<QuePosition::VECIN, DB_BUFFER> indicesQueue_;
    TBuf<QuePosition::VECCALC> indicesBuff_;
    TBuf<QuePosition::VECCALC> uniqueIdBuff_;

    AscendC::GlobalTensor<T> xGm_;
    AscendC::GlobalTensor<T> yGm_;
    AscendC::GlobalTensor<T> valueGm_;
    AscendC::GlobalTensor<INDEX_TYPE> indices_;
    AscendC::GlobalTensor<int8_t> maskGm_;

    const IndexFillSimtDenseIndicesTilingData* tilingData_;
    uint32_t blockIdx_ = 0;
    uint64_t offset_ = 0;
    uint32_t blockNum_ = 0;
    T fillValue;
};

template <typename T, typename INDEX_TYPE, typename COM_T>
__aicore__ inline void IndexFillSimtDenseIndicesImpl<T, INDEX_TYPE, COM_T>::CopyInIndices(int64_t offset, int64_t dataLen)
{
    DataCopyExtParams inParams = {
        static_cast<uint16_t>(1),
        static_cast<uint32_t>(dataLen * sizeof(INDEX_TYPE)),
        static_cast<uint32_t>(0),
        static_cast<uint32_t>(0),
        static_cast<uint32_t>(0)
    };
    DataCopyPadExtParams<INDEX_TYPE> padParams = { false, static_cast<uint8_t>(0), static_cast<uint8_t>(0), static_cast<INDEX_TYPE>(0) };
    LocalTensor<INDEX_TYPE> indicesLocal = indicesQueue_.AllocTensor<INDEX_TYPE>();
    DataCopyPad(indicesLocal, indices_[offset], inParams, padParams);
    indicesQueue_.EnQue(indicesLocal);
}

template <typename T, typename INDEX_TYPE, typename COM_T>
 __simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void SetMaskSimt(__local_mem__ INDEX_TYPE* indicesLocalAddr,
    __local_mem__ int32_t* uniqueIndicesAddr, __gm__ int8_t* mask, COM_T processNum, COM_T n, int64_t length)
{
    COM_T threadIdx = static_cast<COM_T>(Simt::GetThreadIdx());
    COM_T threadNum = static_cast<COM_T>(Simt::GetThreadNum());
    for (COM_T i = threadIdx; i < processNum; i += threadNum) {
        int32_t id = uniqueIndicesAddr[i];
        if (id < 0 || id >= length) {
            continue;
        }
        INDEX_TYPE index = indicesLocalAddr[id];
        index = index < 0 ? index + n : index;
        if (index < 0 || index >= n) {
            continue;
        }
        mask[index] = 1;
    }
}

template <typename T, typename INDEX_TYPE, typename COM_T>
__aicore__ inline void IndexFillSimtDenseIndicesImpl<T, INDEX_TYPE, COM_T>::FillMaskWithUniqueIndices(int64_t length)
{
    // sortedOut输出结果的ub布局: 32字节前哨兵区 + length * sizeof(INDEX_TYPE) + 1个sizeof(INDEX_TYPE)后哨兵区
    int64_t shiftOffset = UB_AGLIN_VALUE / sizeof(INDEX_TYPE);
    LocalTensor<INDEX_TYPE> sortedOut = indicesBuff_.Get<INDEX_TYPE>();
    LocalTensor<int32_t> uniqueIndicesOut = uniqueIdBuff_.Get<int32_t>();
    LocalTensor<INDEX_TYPE> indicesLocal = indicesQueue_.DeQue<INDEX_TYPE>();
    uint32_t uniqueNum = SortAndUnique(sortedOut, uniqueIndicesOut, indicesLocal, length);

    COM_T n = static_cast<COM_T>(tilingData_->n);
    __local_mem__ INDEX_TYPE* actualSortOut = ((__local_mem__ INDEX_TYPE*)sortedOut.GetPhyAddr()) + shiftOffset;
    AscendC::Simt::VF_CALL<SetMaskSimt<T, INDEX_TYPE, COM_T>>(AscendC::Simt::Dim3{THREAD_NUM}, actualSortOut,
        (__local_mem__ int32_t*)uniqueIndicesOut.GetPhyAddr(), (__gm__ int8_t*)maskGm_.GetPhyAddr(), uniqueNum, n, length);

    indicesQueue_.FreeTensor(indicesLocal);
}

template<typename T, typename INDEX_TYPE, typename COM_T>
__aicore__ inline void IndexFillSimtDenseIndicesImpl<T, INDEX_TYPE, COM_T>::BuildIndicesMask()
{
    int64_t indicesUbFactor = tilingData_->indicesUbFactor;
    int64_t totalBlocks = ops::Ceil(tilingData_->indicesNum, indicesUbFactor);
    int64_t usedCoreNum = AscendC::Std::min(totalBlocks, static_cast<int64_t>(blockNum_));
    int64_t tailIndicesUbFactor = tilingData_->indicesNum - (totalBlocks - 1) * indicesUbFactor; // 得到尾块大小
    if (blockIdx_ >= usedCoreNum) {
        return;
    }

    int64_t tailBlocksPerCore = totalBlocks / usedCoreNum;
    int64_t frontBlocksPerCore = totalBlocks % usedCoreNum == 0 ? tailBlocksPerCore : tailBlocksPerCore + 1;
    int64_t frontCoreNum = totalBlocks % usedCoreNum;
    int64_t tailCoreNum = usedCoreNum - frontCoreNum;

    // 表示当前核要处理的块数, 处理的块的索引范围: [start, end)
    uint64_t loop = 0;
    uint64_t start = 0;
    uint64_t end = 0;
    if (blockIdx_ < frontCoreNum) {
        loop = frontBlocksPerCore;
        start = frontBlocksPerCore * blockIdx_;
        end = start + loop;
    } else {
        loop = tailBlocksPerCore;
        start = frontBlocksPerCore * frontCoreNum + (blockIdx_ - frontCoreNum) * tailBlocksPerCore;
        end = start + loop;
    }

    int64_t vfLen = platform::GetVRegSize() / sizeof(INDEX_TYPE);
    int64_t indicesBuffSize = ops::CeilDiv(indicesUbFactor + 1, vfLen) * vfLen * sizeof(INDEX_TYPE) + UB_AGLIN_VALUE;
    pipe_->Reset();
    pipe_->InitBuffer(indicesQueue_, DB_BUFFER, indicesUbFactor * sizeof(INDEX_TYPE));
    pipe_->InitBuffer(indicesBuff_, indicesBuffSize);
    pipe_->InitBuffer(uniqueIdBuff_, indicesUbFactor * sizeof(int32_t));

    int64_t offset = 0;
    for (int64_t i = start; i < end; i++) {
        offset = i * indicesUbFactor;
        // 尾块场景
        if (i == totalBlocks - 1) {
            CopyInIndices(offset, tailIndicesUbFactor);
            FillMaskWithUniqueIndices(tailIndicesUbFactor);
            break;
        }

        CopyInIndices(offset, indicesUbFactor);
        FillMaskWithUniqueIndices(indicesUbFactor);
    }
}

template <typename T, typename INDEX_TYPE, typename COM_T>
__aicore__ inline void IndexFillSimtDenseIndicesImpl<T, INDEX_TYPE, COM_T>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR indices, GM_ADDR value, GM_ADDR workspace)
{
    valueGm_.SetGlobalBuffer((__gm__ T*)(value));
    fillValue = valueGm_.GetValue(0);
    indices_.SetGlobalBuffer((__gm__ INDEX_TYPE*)(indices));
    maskGm_.SetGlobalBuffer((__gm__ int8_t*)(workspace), tilingData_->n * sizeof(int8_t));
    ZeroMemory(maskGm_, tilingData_->n);
    SyncAll();

    // 只有当前核的id位于[0, simdUsedCoreNum)内时，才需下面搬运相关的初始化逻辑.
    if (blockIdx_ >= tilingData_->simdUsedCoreNum) {
        return;
    }

    if (blockIdx_ < tilingData_->frontCoreNum) {
        offset_ = blockIdx_ * tilingData_->loopsPerFrontCore * tilingData_->blockSize;
    } else {
        offset_ = (tilingData_->frontCoreNum * tilingData_->loopsPerFrontCore * tilingData_->blockSize) + ((blockIdx_ - tilingData_->frontCoreNum) * tilingData_->loopsPerTailCore * tilingData_->blockSize);
    }

    pipe_->InitBuffer(dataQueue_, DB_BUFFER, tilingData_->blockSize * sizeof(T));
    xGm_.SetGlobalBuffer((__gm__ T*)(x) + offset_);
    yGm_.SetGlobalBuffer((__gm__ T*)(y) + offset_);
}

template <typename T, typename INDEX_TYPE, typename COM_T>
__aicore__ inline void IndexFillSimtDenseIndicesImpl<T, INDEX_TYPE, COM_T>::CopyIn(int64_t offset, int64_t dataLen)
{
  DataCopyPadExtParams<T> padParams = { false, static_cast<uint8_t>(0), static_cast<uint8_t>(0), static_cast<T>(0) };
  DataCopyExtParams inParams = {
    static_cast<uint16_t>(1),
    static_cast<uint32_t>(dataLen * sizeof(T)),
    static_cast<uint32_t>(0),
    static_cast<uint32_t>(0),
    static_cast<uint32_t>(0)
  };
  LocalTensor<T> xLocal = dataQueue_.AllocTensor<T>();
  DataCopyPad(xLocal, xGm_[offset], inParams, padParams);
  dataQueue_.EnQue(xLocal);
}

template <typename T, typename INDEX_TYPE, typename COM_T>
__aicore__ inline void IndexFillSimtDenseIndicesImpl<T, INDEX_TYPE, COM_T>::CopyOut(int64_t offset, int64_t dataLen)
{
  LocalTensor<T> yLocal = dataQueue_.DeQue<T>();
  DataCopyExtParams outParams = {
    static_cast<uint16_t>(1),
    static_cast<uint32_t>(dataLen * sizeof(T)),
    static_cast<uint32_t>(0),
    static_cast<uint32_t>(0),
    static_cast<uint32_t>(0)
  };
  DataCopyPad(yGm_[offset], yLocal, outParams);
  dataQueue_.FreeTensor(yLocal);
}

template <typename T, typename INDEX_TYPE, typename COM_T>
__aicore__ inline void IndexFillSimtDenseIndicesImpl<T, INDEX_TYPE, COM_T>::CopyTensor()
{
    if (blockIdx_ >= tilingData_->simdUsedCoreNum) {
        return;
    }

    int64_t offset = 0;
    int64_t loopSize = blockIdx_ < tilingData_->frontCoreNum ? tilingData_->loopsPerFrontCore : tilingData_->loopsPerTailCore;
    for (int64_t idx = 0; idx < loopSize - 1; idx++) {
        offset = idx * tilingData_->blockSize;
        CopyIn(offset, tilingData_->blockSize);
        CopyOut(offset, tilingData_->blockSize);
    }

    // 尾块搬运
    int64_t dataLen = tilingData_->blockSize;
    offset = (loopSize - 1) * tilingData_->blockSize;
    if (blockIdx_ == tilingData_->simdUsedCoreNum - 1) {
        dataLen = tilingData_->tailBlockSize;
    }
    CopyIn(offset, dataLen);
    CopyOut(offset, dataLen);
}

template <typename T, typename INDEX_TYPE, typename COM_T>
__aicore__ inline void IndexFillSimtDenseIndicesImpl<T, INDEX_TYPE, COM_T>::Process(__gm__ T* y)
{
    // 将x复制搬运到y上
    CopyTensor();

    SyncAll();

    // ub按照indicesNum重新切分，构建maskGm位图
    BuildIndicesMask();

    // 同步等待
    SyncAll();

    COM_T p = static_cast<COM_T>(tilingData_->p);
    COM_T q = static_cast<COM_T>(tilingData_->q);
    COM_T n = static_cast<COM_T>(tilingData_->n);
    COM_T process_num = static_cast<COM_T>(tilingData_->n * p * q);

    COM_T shift_n = 0;
    COM_T magic_n = 0;
    GetUintDivMagicAndShift(magic_n, shift_n, n);

    COM_T shift_q = 0;
    COM_T magic_q = 0;
    GetUintDivMagicAndShift(magic_q, shift_q, q);

    // 执行index_fill核心逻辑
    AscendC::Simt::VF_CALL<IndexFillSimtDenseIndicesCompute<T, INDEX_TYPE, COM_T>>(AscendC::Simt::Dim3{THREAD_NUM}, y,
        (__gm__ int8_t*)maskGm_.GetPhyAddr(), fillValue, process_num, n, shift_n, magic_n, shift_q, magic_q);
}

template <typename T, typename INDEX_TYPE, typename COM_T>
 __simt_vf__ __aicore__ LAUNCH_BOUND(THREAD_NUM) inline void IndexFillSimtDenseIndicesCompute(__gm__ T* y, __gm__ int8_t* maskAddr, T value, COM_T total_num, COM_T n,
    COM_T shift_n, COM_T magic_n, COM_T shift_q, COM_T magic_q)
{
    COM_T threadNum = static_cast<COM_T>(Simt::GetBlockNum() * Simt::GetThreadNum());
    COM_T threadIdx = static_cast<COM_T>(Simt::GetBlockIdx() * Simt::GetThreadNum() + Simt::GetThreadIdx());
    for (COM_T i = threadIdx; i < total_num; i += threadNum) {
        COM_T pnIdx = Simt::UintDiv(i, magic_q, shift_q);
        COM_T pIdx = Simt::UintDiv(pnIdx, magic_n, shift_n);
        COM_T nIdx = pnIdx - (pIdx * n);
        if (maskAddr[nIdx] == 0) {
            continue;
        }
        y[i] = value;
    }
}

} // namespace IndexFill

#endif // CANN_INDEX_FILL_SIMT_DENSE_INDICES_H
