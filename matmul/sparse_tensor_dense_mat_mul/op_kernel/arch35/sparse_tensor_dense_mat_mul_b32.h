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
 * \file sparse_tensor_dense_mat_mul_b16.h
 * \brief
 */
#ifndef SPARSE_TENSOR_DENSE_MAT_MUL_B32_H
#define SPARSE_TENSOR_DENSE_MAT_MUL_B32_H

#include "sparse_tensor_dense_mat_mul_base.h"

namespace SparseTensorDenseMatMul {
using namespace AscendC;

template <typename Tindices, typename Tvalues, bool ADJ_A, bool ADJ_B>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void ComputeB32(
    const int32_t usedCoreNum, const int32_t currCoreIdx, const int32_t elemNum, const int32_t m, const int32_t n,
    const int32_t p, __gm__ Tindices* x1Indices, __gm__ Tvalues* x1Values, __gm__ Tvalues* x2, __gm__ Tvalues* y)
{
    // 总共有usedCoreNum*ThreadNum个线程，每个线程所在的位置为currCoreIdx*ThreadNum+LocalThreadIdx
    for (int32_t elemIdx = currCoreIdx * Simt::GetThreadNum() + Simt::GetThreadIdx(); elemIdx < elemNum;
         elemIdx += usedCoreNum * Simt::GetThreadNum()) {
        // 计算索引i、j、k，用于找到v1=x1(i,k)和v2=x2(k,j)
        // i、j、k 统一转成int32类型
        int32_t x1VecIdx = elemIdx / p;
        int32_t j = elemIdx % p;
        int32_t i, k;
        if constexpr (ADJ_A) {
            i = static_cast<int32_t>(x1Indices[2 * x1VecIdx + 1]); // 转置前x1_indices存的每对元素是[列, 行]
            k = static_cast<int32_t>(x1Indices[2 * x1VecIdx]);
        } else {
            i = static_cast<int32_t>(x1Indices[2 * x1VecIdx]);
            k = static_cast<int32_t>(x1Indices[2 * x1VecIdx + 1]);
        }
        // 读取v1和v2
        Tvalues v1 = x1Values[x1VecIdx];
        Tvalues v2;
        if constexpr (ADJ_B) {
            v2 = x2[j * n + k]; // 转置后x2(k, j)的元素位置，在转置前的x2(j, k)，且转置前的shape=(p, n)
        } else {
            v2 = x2[k * p + j];
        }
        // 累加到对应位置
        __gm__ Tvalues* outAddr = y + i * p + j;
        Simt::AtomicAdd(outAddr, v1 * v2);
    }
}

template <typename Tindices, typename Tvalues, bool ADJ_A, bool ADJ_B>
class SparseTensorDenseMatMulB32 {
public:
    __aicore__ inline SparseTensorDenseMatMulB32(TPipe* pipe, const SparseTensorDenseMatMulTilingData* tiling)
        : pipe_(pipe), tilingData_(tiling){};
    __aicore__ inline void Init(GM_ADDR x1Indices, GM_ADDR x1Values, GM_ADDR x2, GM_ADDR y);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitGlobalTensor(GM_ADDR x1Indices, GM_ADDR x1Values, GM_ADDR x2, GM_ADDR y);
    __aicore__ inline void InitOutputGm();

private:
    TPipe* pipe_;
    const SparseTensorDenseMatMulTilingData* tilingData_;
    uint32_t currCoreIdx_{0};

    GlobalTensor<Tindices> x1IndicesGm_;
    GlobalTensor<Tvalues> x1ValuesGm_;
    GlobalTensor<Tvalues> x2Gm_;
    GlobalTensor<Tvalues> yGm_;       // y的原始位置
    GlobalTensor<Tvalues> yOutputGm_; // y分配到每个核心的数据，偏移后的位置
};

template <typename Tindices, typename Tvalues, bool ADJ_A, bool ADJ_B>
__aicore__ inline void SparseTensorDenseMatMulB32<Tindices, Tvalues, ADJ_A, ADJ_B>::Init(
    GM_ADDR x1Indices, GM_ADDR x1Values, GM_ADDR x2, GM_ADDR y)
{
    currCoreIdx_ = static_cast<uint32_t>(GetBlockIdx());
    InitGlobalTensor(x1Indices, x1Values, x2, y);
    InitOutputGm(); //! InitOutput/Workspace要在pipe->InitBuffer之前执行
}

template <typename Tindices, typename Tvalues, bool ADJ_A, bool ADJ_B>
__aicore__ inline void SparseTensorDenseMatMulB32<Tindices, Tvalues, ADJ_A, ADJ_B>::InitGlobalTensor(
    GM_ADDR x1Indices, GM_ADDR x1Values, GM_ADDR x2, GM_ADDR y)
{
    x1IndicesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ Tindices*>(x1Indices));
    x1ValuesGm_.SetGlobalBuffer(reinterpret_cast<__gm__ Tvalues*>(x1Values));
    x2Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ Tvalues*>(x2));
    yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ Tvalues*>(y));
    yOutputGm_.SetGlobalBuffer(
        reinterpret_cast<__gm__ Tvalues*>(y) + currCoreIdx_ * tilingData_->initAndOutFormerCoreElemNum);
}

template <typename Tindices, typename Tvalues, bool ADJ_A, bool ADJ_B>
__aicore__ inline void SparseTensorDenseMatMulB32<Tindices, Tvalues, ADJ_A, ADJ_B>::InitOutputGm()
{
    if (currCoreIdx_ < tilingData_->initAndOutUsedCoreNum) {
        uint64_t initEleNumPerCore = (currCoreIdx_ == tilingData_->initAndOutUsedCoreNum - 1) ?
                                         tilingData_->initAndOutTailCoreElemNum :
                                         tilingData_->initAndOutFormerCoreElemNum;
        InitGlobalMemory(yOutputGm_, initEleNumPerCore, static_cast<Tvalues>(0));
    }
    SyncAll();
}

template <typename Tindices, typename Tvalues, bool ADJ_A, bool ADJ_B>
__aicore__ inline void SparseTensorDenseMatMulB32<Tindices, Tvalues, ADJ_A, ADJ_B>::Process()
{
    if (currCoreIdx_ < tilingData_->computeUsedCoreNum) {
        __gm__ Tindices* x1IndicesGmAddr = (__gm__ Tindices*)x1IndicesGm_.GetPhyAddr();
        __gm__ Tvalues* x1ValuesGmAddr = (__gm__ Tvalues*)x1ValuesGm_.GetPhyAddr();
        __gm__ Tvalues* x2GmAddr = (__gm__ Tvalues*)x2Gm_.GetPhyAddr();
        __gm__ Tvalues* yGmAddr = (__gm__ Tvalues*)yGm_.GetPhyAddr();
        Simt::VF_CALL<ComputeB32<Tindices, Tvalues, ADJ_A, ADJ_B>>(
            Simt::Dim3{SIMT_MAX_THREAD_NUM, 1, 1}, tilingData_->computeUsedCoreNum, currCoreIdx_,
            tilingData_->computeTotalElemNum, tilingData_->computeM, tilingData_->computeN,
            tilingData_->computeP, x1IndicesGmAddr, x1ValuesGmAddr, x2GmAddr, yGmAddr);
    }
}

} // namespace SparseTensorDenseMatMul

#endif // SPARSE_TENSOR_DENSE_MAT_MUL_B32_H
