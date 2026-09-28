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
 * \file cum_sum_split_batch_simt.h
 * \brief
 */

#ifndef CUM_SUM_SPLIT_BATCH_SIMT_H
#define CUM_SUM_SPLIT_BATCH_SIMT_H

#include "op_kernel/platform_util.h"
#include "repeat_interleave_base.h"
#include "op_kernel/math_util.h"

namespace RepeatInterleave {
using namespace AscendC;

template <typename T, typename U, typename V, typename AddrType, bool isCumSumInUb = false>
class SplitBatchSimt {
public:
    __aicore__ inline SplitBatchSimt(const RepeatInterleaveCumSumTilingData& tilingData, TPipe& pipe)
        : tilingData_(tilingData), pipe_(pipe){};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR repeats, GM_ADDR y, GM_ADDR workspace);

    __aicore__ inline void Process();
    __aicore__ inline void CopyInRepeat(int64_t dataLen);
    __aicore__ inline void ComputePrefixSumAndRepeat(int64_t dataLen);
    __aicore__ inline void ComputeSingleLoopPrefixSum(LocalTensor<U> repeatsLocal, uint32_t dataLen,
                                                      __ubuf__ V* prefixSumAddr, __ubuf__ V* tmpAddr);

private:
    AscendC::GlobalTensor<T> xGm_;
    AscendC::GlobalTensor<U> repeatsGm_;
    AscendC::GlobalTensor<T> yGm_;
    AscendC::GlobalTensor<V> prefixSumGm_;

    TQue<QuePosition::VECIN, CUMSUM_BUFFER> repeatsQueue_;
    TBuf<QuePosition::VECCALC> tmpBuf_;
    TQue<QuePosition::VECOUT, 1> prefixSumQueue_;

    TPipe& pipe_;
    const RepeatInterleaveCumSumTilingData& tilingData_;

    AddrType curBatchCount_ = 0;
    AddrType inputBatchOffset_ = 0;
    AddrType outputBatchOffset_ = 0;
    int64_t customOffset_ = 0;
    __local_mem__ V* cumSumAddr_ = nullptr;
};

template <typename T, typename U, typename V, typename AddrType, bool isCumSumInUb>
__aicore__ inline void SplitBatchSimt<T, U, V, AddrType, isCumSumInUb>::Init(GM_ADDR x, GM_ADDR repeats, GM_ADDR y,
                                                                             GM_ADDR workspace)
{
    xGm_.SetGlobalBuffer((__gm__ T*)x + AscendC::GetBlockIdx() * tilingData_.eachCoreBatchCount *
                                            tilingData_.mergedDims[1] * tilingData_.mergedDims[2]);
    repeatsGm_.SetGlobalBuffer((__gm__ U*)repeats);
    yGm_.SetGlobalBuffer((__gm__ T*)y + AscendC::GetBlockIdx() * tilingData_.eachCoreBatchCount *
                                            tilingData_.totalRepeatSum * tilingData_.mergedDims[2]);
    if constexpr (!isCumSumInUb) {
        prefixSumGm_.SetGlobalBuffer((__gm__ V*)workspace);
    }

    if constexpr (isCumSumInUb) {
        pipe_.InitBuffer(repeatsQueue_, CUMSUM_BUFFER,
                         tilingData_.mergedDims[1] * sizeof(V) + platform::GetUbBlockSize());
        if constexpr (!std::is_same_v<U, V>) {
            pipe_.InitBuffer(tmpBuf_, tilingData_.mergedDims[1] * sizeof(V) + platform::GetUbBlockSize());
        }
        pipe_.InitBuffer(prefixSumQueue_, 1, tilingData_.mergedDims[1] * sizeof(V));
        customOffset_ = platform::GetUbBlockSize() / sizeof(V);
    }

    curBatchCount_ = (AscendC::GetBlockIdx() == tilingData_.usedCoreNum - 1) ? tilingData_.tailCoreBatchCount :
                                                                               tilingData_.eachCoreBatchCount;
    inputBatchOffset_ = tilingData_.mergedDims[1] * tilingData_.mergedDims[2];
    outputBatchOffset_ = tilingData_.totalRepeatSum * tilingData_.mergedDims[2];
}

template <typename T, typename U, typename V, typename AddrType, bool isCumSumInUb>
__aicore__ inline void SplitBatchSimt<T, U, V, AddrType, isCumSumInUb>::CopyInRepeat(int64_t dataLen)
{
    DataCopyPadExtParams<U> padParams;
    padParams.isPad = false;
    padParams.leftPadding = 0;
    padParams.rightPadding = 0;
    padParams.paddingValue = 0;

    DataCopyExtParams inParams;
    inParams.blockCount = 1;
    inParams.blockLen = dataLen * sizeof(U);
    inParams.srcStride = 0;
    inParams.dstStride = 0;

    LocalTensor<U> repeatsLocal = repeatsQueue_.AllocTensor<U>();
    DataCopyPad(repeatsLocal[customOffset_], repeatsGm_, inParams, padParams);
    repeatsQueue_.EnQue(repeatsLocal);
}

template <typename T, typename U, typename V, typename AddrType, bool isCumSumInUb>
__aicore__ inline void SplitBatchSimt<T, U, V, AddrType, isCumSumInUb>::ComputeSingleLoopPrefixSum(
    LocalTensor<U> repeatsLocal, uint32_t dataLen, __ubuf__ V* prefixSumAddr, __ubuf__ V* tmpAddr)
{
    auto repeatsAddr = (__ubuf__ U*)repeatsLocal.GetPhyAddr() + customOffset_;
    uint32_t vfLen = Ops::Base::GetVRegSize() / sizeof(int32_t);
    uint32_t vfLenB64 = Ops::Base::GetVRegSize() / sizeof(int64_t);
    uint32_t rows = vfLen;
    uint32_t cols = (dataLen + vfLen - 1) / vfLen;
    uint32_t loopB64 = (dataLen + vfLenB64 - 1) / vfLenB64;

    uint16_t size0 = cols;
    uint16_t size1 = cols / vfLen;
    uint16_t tailSize1 = cols - size1 * vfLen;
    if (tailSize1 == 0) {
        size1--;
        tailSize1 = vfLen;
    }
    ComputeSingleLoopPrefixSumVf<U, V>(repeatsAddr, prefixSumAddr, tmpAddr, vfLen, rows, cols, size0, size1, tailSize1,
                                       vfLenB64, loopB64);
}

template <typename T, typename U, typename V, typename AddrType, bool isCumSumInUb>
__aicore__ inline void SplitBatchSimt<T, U, V, AddrType, isCumSumInUb>::ComputePrefixSumAndRepeat(int64_t dataLen)
{
    LocalTensor<V> prefixSumLocal = prefixSumQueue_.AllocTensor<V>();
    auto prefixSumAddr = (__ubuf__ V*)prefixSumLocal.GetPhyAddr();

    LocalTensor<V> tmpLocal;
    __ubuf__ V* tmpAddr = nullptr;
    if constexpr (!std::is_same_v<U, V>) {
        tmpLocal = tmpBuf_.Get<V>();
        tmpAddr = (__ubuf__ V*)tmpLocal.GetPhyAddr() + customOffset_;
    }

    LocalTensor<U> repeatsLocal = repeatsQueue_.DeQue<U>();
    ComputeSingleLoopPrefixSum(repeatsLocal, dataLen, tmpAddr, prefixSumAddr);

    if constexpr (!std::is_same_v<U, V>) {
        Duplicate(tmpLocal, (V)0, customOffset_);
        cumSumAddr_ = (__ubuf__ V*)tmpLocal.GetPhyAddr() + customOffset_ - 1;
    } else {
        Duplicate(repeatsLocal, (V)0, customOffset_);
        cumSumAddr_ = (__ubuf__ V*)repeatsLocal.GetPhyAddr() + customOffset_ - 1;
    }

    uint32_t threadNumX = static_cast<uint32_t>(tilingData_.threadNumX);
    uint32_t threadNumY = static_cast<uint32_t>(tilingData_.threadNumY);
    uint32_t threadNumZ = static_cast<uint32_t>(tilingData_.threadNumZ);

    // simt vf with __ubuf__ cumSumAddr_
    asc_vf_call<SimtRepeatSplitBatchCumSumUb<T, U, V, AddrType>>(
        dim3{threadNumX, threadNumY, threadNumZ}, curBatchCount_, tilingData_.mergedDims[1], inputBatchOffset_,
        outputBatchOffset_, tilingData_.mergedDims[2], (__gm__ T*)(xGm_.GetPhyAddr()),
        (__gm__ U*)(repeatsGm_.GetPhyAddr()), (__gm__ T*)(yGm_.GetPhyAddr()), cumSumAddr_);

    prefixSumQueue_.FreeTensor(prefixSumLocal);
    repeatsQueue_.FreeTensor(repeatsLocal);
}

template <typename T, typename U, typename V, typename AddrType, bool isCumSumInUb>
__aicore__ inline void SplitBatchSimt<T, U, V, AddrType, isCumSumInUb>::Process()
{
    if (AscendC::GetBlockIdx() >= tilingData_.usedCoreNum) {
        return;
    }

    if constexpr (isCumSumInUb) {
        int64_t dataLen = tilingData_.mergedDims[1];
        CopyInRepeat(dataLen);
        ComputePrefixSumAndRepeat(dataLen);
    } else {
        uint32_t threadNumX = static_cast<uint32_t>(tilingData_.threadNumX);
        uint32_t threadNumY = static_cast<uint32_t>(tilingData_.threadNumY);
        uint32_t threadNumZ = static_cast<uint32_t>(tilingData_.threadNumZ);

        // simt vf with __gm__ prefixSumGm_
        asc_vf_call<SimtRepeatSplitBatch<T, U, V, AddrType>>(
            dim3{threadNumX, threadNumY, threadNumZ}, curBatchCount_, tilingData_.mergedDims[1], inputBatchOffset_,
            outputBatchOffset_, tilingData_.mergedDims[2], (__gm__ T*)(xGm_.GetPhyAddr()),
            (__gm__ U*)(repeatsGm_.GetPhyAddr()), (__gm__ T*)(yGm_.GetPhyAddr()),
            (__gm__ V*)(prefixSumGm_.GetPhyAddr()));
    }
}

} // namespace RepeatInterleave

#endif
