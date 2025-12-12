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
 * \file linear_index_v2.h
 * \brief
 */
#ifndef LINEAR_INDEX_V2_H
#define LINEAR_INDEX_V2_H
#include "kernel_operator.h"
#define IS_CAST_INT (isSame<T, int64_t>::value)
using namespace AscendC;

template <typename Tp, Tp v>
struct integralConstant {
    static constexpr Tp value = v;
};
using true_type = integralConstant<bool, true>;
using false_type = integralConstant<bool, false>;
template <typename, typename>
struct isSame : public false_type {
};
template <typename Tp>
struct isSame<Tp, Tp> : public true_type {
};

constexpr uint32_t BUFFER_NUM = 1;
constexpr uint32_t BLOCK_SIZE = 32;

template <typename T>
class LinearIndexKernelV2
{
public:
    __aicore__ inline LinearIndexKernelV2() = delete;
    __aicore__ inline LinearIndexKernelV2(
        GM_ADDR indexList, GM_ADDR stride, GM_ADDR valueSize, GM_ADDR output, GM_ADDR workSpace,
        const LinearIndexV2TilingData& tiling, TPipe& pipe)
    {
        InitParams(tiling, indexList);
        SetGmAddr(stride, valueSize, output, tiling);
        InitBuffers(pipe);
    }

    __aicore__ inline void Process()
    {
        for (int i = 0; i < tensorId_; i++) {
            indexGm_.SetGlobalBuffer(GetTensorAddr(indexList_, i) + idxAddrOffset_);
            for (int j = 0; j < formerTime_; j++) {
                CopyIn(i, j, true);
                Compute(true);
                CopyOut(j, true);
            }
            for (int j = 0; j < tailTime_; j++) {
                CopyIn(i, j, false);
                Compute(false);
                CopyOut(j, false);
            }
        }
    }

private:
    __aicore__ inline void InitParams(const LinearIndexV2TilingData& tiling, GM_ADDR indexList)
    {
        indexList_ = indexList;
        blockIdx_ = GetBlockIdx();
        formerCoreNum_ = tiling.params.formerCoreNum;
        tensorId_ = tiling.params.tensorId;
        if (blockIdx_ < formerCoreNum_) {
            dataNum_ = tiling.params.formerCoreDataNum;
            formerDataNum_ = tiling.params.formerCoreFormerDataNum;
            tailDataNum_ = tiling.params.formerCoreTailDataNum;
            formerTime_ = tiling.params.formerCoreFormerTime;
            tailTime_ = tiling.params.formerCoreTailTime;
            idxAddrOffset_ = tiling.params.formerCoreDataNum * blockIdx_;
        } else {
            dataNum_ = tiling.params.tailCoreDataNum;
            formerDataNum_ = tiling.params.tailCoreFormerDataNum;
            tailDataNum_ = tiling.params.tailCoreTailDataNum;
            formerTime_ = tiling.params.tailCoreFormerTime;
            tailTime_ = tiling.params.tailCoreTailTime;
            idxAddrOffset_ = tiling.params.formerCoreDataNum * formerCoreNum_ +
                             tiling.params.tailCoreDataNum * (blockIdx_ - formerCoreNum_);
        }
    }

    __aicore__ inline void InitBuffers(TPipe& pipe)
    {
        uint32_t strideAlign = BLOCK_SIZE / sizeof(int);
        uint32_t idxInBlock = BLOCK_SIZE / sizeof(T);
        uint32_t idxAlignNum = ((formerDataNum_ + idxInBlock - 1) / idxInBlock) * idxInBlock;
        pipe.InitBuffer(strideQue_, BUFFER_NUM, strideAlign * sizeof(int));
        pipe.InitBuffer(valueSizeQue_, BUFFER_NUM, strideAlign * sizeof(int));
        pipe.InitBuffer(indexInQue_, BUFFER_NUM, idxAlignNum * sizeof(T));
        pipe.InitBuffer(indexOutQue_, BUFFER_NUM, idxAlignNum * sizeof(int));
        pipe.InitBuffer(remainQue_, BUFFER_NUM, idxAlignNum * sizeof(float));
    }

    __aicore__ inline void SetGmAddr(
        GM_ADDR stride, GM_ADDR valueSize, GM_ADDR output, const LinearIndexV2TilingData& tiling)
    {
        valueSizeGm_.SetGlobalBuffer((__gm__ int*)valueSize);
        strideGm_.SetGlobalBuffer((__gm__ int*)stride);
        outputGm_.SetGlobalBuffer((__gm__ int*)output + idxAddrOffset_, dataNum_);
        InitGlobalMemory(outputGm_, dataNum_, 0);
        SyncAll();
    }

    __aicore__ inline void CopyIn(const int progress, const int curTimes, const bool formerFlag)
    {
        LocalTensor<T> indexLocal = indexInQue_.AllocTensor<T>();
        LocalTensor<int> strideLocal = strideQue_.AllocTensor<int>();
        LocalTensor<int> valueSizeLocal = valueSizeQue_.AllocTensor<int>();
        uint32_t copyNum = formerFlag ? formerDataNum_ : tailDataNum_;
        uint32_t addrOffset =
            formerFlag ? formerDataNum_ * curTimes : formerDataNum_ * formerTime_ + tailDataNum_ * curTimes;
        DataCopyExtParams idxCopyParams{1, static_cast<uint32_t>(copyNum * sizeof(T)), 0, 0, 0};
        DataCopyExtParams strideCopyParams{1, static_cast<uint32_t>(sizeof(int)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> padParams{true, 0, 0, 0};
        if constexpr (IS_CAST_INT) {
            DataCopyPadExtParams<uint32_t> uPadParams{true, 0, 0, 0};
            DataCopyPadGm2UBImpl(
                (__ubuf__ uint32_t*)indexLocal.GetPhyAddr(), (__gm__ uint32_t*)indexGm_[addrOffset].GetPhyAddr(),
                idxCopyParams, uPadParams);
        } else {
            DataCopyPadExtParams<T> uPadParams{true, 0, 0, 0};
            DataCopyPad(indexLocal, indexGm_[addrOffset], idxCopyParams, uPadParams);
        }
        DataCopyPad(valueSizeLocal, valueSizeGm_[progress], strideCopyParams, padParams);
        DataCopyPad(strideLocal, strideGm_[progress], strideCopyParams, padParams);
        valueSizeQue_.EnQue<int>(valueSizeLocal);
        indexInQue_.EnQue<T>(indexLocal);
        strideQue_.EnQue<int>(strideLocal);
    }

    __aicore__ inline void Compute(const bool formerFlag)
    {
        LocalTensor<int> indexOutLocal = indexOutQue_.AllocTensor<int>();
        LocalTensor<T> indexLocal = indexInQue_.DeQue<T>();
        LocalTensor<int> strideLocal = strideQue_.DeQue<int>();
        LocalTensor<int> valueSizeLocal = valueSizeQue_.DeQue<int>();
        uint32_t dataNum = formerFlag ? formerDataNum_ : tailDataNum_;
        int32_t stride = strideLocal.GetValue(0);
        int32_t size = valueSizeLocal.GetValue(0);
        float sizeDiv = size == 0 ? 1 : 1.0f / size;
        if constexpr (IS_CAST_INT) {
            LocalTensor<float> remainLocal = remainQue_.AllocTensor<float>();
            Cast<float, T>(remainLocal, indexLocal, RoundMode::CAST_RINT, dataNum);
            Muls(remainLocal, remainLocal, sizeDiv, dataNum);
            Cast<int, float>(indexOutLocal, remainLocal, RoundMode::CAST_FLOOR, dataNum);
            Muls(indexOutLocal, indexOutLocal, size, dataNum);
            remainQue_.FreeTensor<float>(remainLocal);

            LocalTensor<int> castLocal = remainQue_.AllocTensor<int>();
            Cast<int, T>(castLocal, indexLocal, RoundMode::CAST_NONE, dataNum);
            Sub(indexOutLocal, castLocal, indexOutLocal, dataNum);
            Muls(indexOutLocal, indexOutLocal, stride, dataNum);
            remainQue_.FreeTensor<int>(castLocal);
        } else {
            LocalTensor<float> remainLocal = remainQue_.AllocTensor<float>();
            Cast<float, T>(remainLocal, indexLocal, RoundMode::CAST_NONE, dataNum);
            Muls(remainLocal, remainLocal, sizeDiv, dataNum);
            Cast<int, float>(indexOutLocal, remainLocal, RoundMode::CAST_FLOOR, dataNum);
            Muls(indexOutLocal, indexOutLocal, size, dataNum);
            Sub(indexOutLocal, indexLocal, indexOutLocal, dataNum);
            Muls(indexOutLocal, indexOutLocal, stride, dataNum);
            remainQue_.FreeTensor<float>(remainLocal);
        }
        indexInQue_.FreeTensor<T>(indexLocal);
        strideQue_.FreeTensor<int>(strideLocal);
        valueSizeQue_.FreeTensor<int>(valueSizeLocal);
        indexOutQue_.EnQue<int>(indexOutLocal);
    }

    __aicore__ inline void CopyOut(const int curTimes, const bool formerFlag)
    {
        LocalTensor<int> indexOutLocal = indexOutQue_.DeQue<int>();
        uint32_t copyNum = formerFlag ? formerDataNum_ : tailDataNum_;
        uint32_t addrOffset =
            formerFlag ? formerDataNum_ * curTimes : formerDataNum_ * formerTime_ + tailDataNum_ * curTimes;
        DataCopyExtParams idxCopyParams{1, static_cast<uint32_t>(copyNum * sizeof(int)), 0, 0, 0};
        SetAtomicAdd<int32_t>();
        DataCopyPad(outputGm_[addrOffset], indexOutLocal, idxCopyParams);
        SetAtomicNone();
        indexOutQue_.FreeTensor<int>(indexOutLocal);
    }

    __aicore__ inline __gm__ T* GetTensorAddr(GM_ADDR indexListPtr, const int offset)
    {
        // 获取tensorList中tensor的首地址
        __gm__ uint64_t* dataAddr = reinterpret_cast<__gm__ uint64_t*>(indexListPtr);
        uint64_t tensorPtrOffset = *dataAddr;
        __gm__ uint64_t* tensorPtr = dataAddr + (tensorPtrOffset >> 3);
        return reinterpret_cast<__gm__ T*>(*(tensorPtr + offset));
    }

private:
    GM_ADDR indexList_;
    GlobalTensor<T> indexGm_;
    GlobalTensor<int> valueSizeGm_;
    GlobalTensor<int> strideGm_;
    GlobalTensor<int> outputGm_;

    TQue<TPosition::VECIN, BUFFER_NUM> valueSizeQue_;
    TQue<TPosition::VECIN, BUFFER_NUM> strideQue_;
    TQue<TPosition::VECIN, BUFFER_NUM> remainQue_;
    TQue<TPosition::VECIN, BUFFER_NUM> indexInQue_;
    TQue<TPosition::VECOUT, BUFFER_NUM> indexOutQue_;

    uint64_t idxAddrOffset_;
    uint64_t blockIdx_;
    uint64_t tensorId_;
    uint64_t dataNum_;
    uint64_t formerCoreNum_;
    uint64_t formerDataNum_;
    uint64_t tailDataNum_;
    uint64_t formerTime_;
    uint64_t tailTime_;
};

#endif // LINEAR_INDEX_V2_H