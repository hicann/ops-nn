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
#ifndef SPARSE_TENSOR_DENSE_MAT_MUL_B16_H
#define SPARSE_TENSOR_DENSE_MAT_MUL_B16_H

#include "sparse_tensor_dense_mat_mul_base.h"
#include "../inc/platform.h"
#include "kernel_operator.h"
#include "../inc/kernel_utils.h"

namespace SparseTensorDenseMatMul {
using namespace AscendC;
constexpr uint32_t BLOCK_SIZE = platform::GetUbBlockSize();

template <typename TIdx, typename T, typename TSum, bool ADJ_A, bool ADJ_B>
class SparseTensorDenseMatMulB16 {
public:
    __aicore__ inline SparseTensorDenseMatMulB16(TPipe* pipe, const SparseTensorDenseMatMulTilingData* tiling)
        : pipe_(pipe), tilingData_(tiling){};
    __aicore__ inline void Init(GM_ADDR x1Indices, GM_ADDR x1Values, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitGlobalBuffer(
        GM_ADDR x1Indices, GM_ADDR x1Values, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace);
    __aicore__ inline void InitWorkspaceGm();
    __aicore__ inline void InitTQueBuffer();
    __aicore__ inline void ProcessYResult();
    __aicore__ inline void CopyInYWorkspace(const uint64_t& resultGmOffset, const DataCopyExtParams& dataCopyParams);
    __aicore__ inline void CastY(const uint64_t& processNumPerLoop);
    __aicore__ inline void CopyOutY(const uint64_t& resultGmOffset, const DataCopyExtParams& dataCopyParams);

private:
    TPipe* pipe_;
    const SparseTensorDenseMatMulTilingData* tilingData_;

    TQue<TPosition::VECIN, 1> wsInQue_;
    TQue<TPosition::VECOUT, 1> yOutQue_;

    GlobalTensor<TIdx> x1IndicesGm_;
    GlobalTensor<T> x1ValuesGm_;
    GlobalTensor<T> x2Gm_;
    GlobalTensor<T> yGm_;
    GlobalTensor<TSum> workspaceGm_;
    GlobalTensor<TSum> workspaceInitGm_;

    uint32_t blockIdx_ = 0;
};

template <typename TIdx, typename T, typename TSum, bool ADJ_A, bool ADJ_B>
__aicore__ inline void SparseTensorDenseMatMulB16<TIdx, T, TSum, ADJ_A, ADJ_B>::Init(
    GM_ADDR x1Indices, GM_ADDR x1Values, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace)
{
    blockIdx_ = static_cast<uint32_t>(GetBlockIdx());
    InitGlobalBuffer(x1Indices, x1Values, x2, y, workspace);
    InitWorkspaceGm();
    InitTQueBuffer();
}

template <typename TIdx, typename T, typename TSum, bool ADJ_A, bool ADJ_B>
__aicore__ inline void SparseTensorDenseMatMulB16<TIdx, T, TSum, ADJ_A, ADJ_B>::InitGlobalBuffer(
    GM_ADDR x1Indices, GM_ADDR x1Values, GM_ADDR x2, GM_ADDR y, GM_ADDR workspace)
{
    x1IndicesGm_.SetGlobalBuffer((__gm__ TIdx*)x1Indices);
    x1ValuesGm_.SetGlobalBuffer((__gm__ T*)x1Values);
    x2Gm_.SetGlobalBuffer((__gm__ T*)x2);
    yGm_.SetGlobalBuffer((__gm__ T*)y);
    workspaceGm_.SetGlobalBuffer((__gm__ TSum*)workspace);
    workspaceInitGm_.SetGlobalBuffer((__gm__ TSum*)workspace + blockIdx_ * tilingData_->initAndOutFormerCoreElemNum);
}

template <typename TIdx, typename T, typename TSum, bool ADJ_A, bool ADJ_B>
__aicore__ inline void SparseTensorDenseMatMulB16<TIdx, T, TSum, ADJ_A, ADJ_B>::InitWorkspaceGm()
{
    if (blockIdx_ < tilingData_->initAndOutUsedCoreNum) {
        uint64_t initEleNumPerCore = (blockIdx_ == tilingData_->initAndOutUsedCoreNum - 1) ?
                                         tilingData_->initAndOutTailCoreElemNum :
                                         tilingData_->initAndOutFormerCoreElemNum;
        InitGlobalMemory(workspaceInitGm_, initEleNumPerCore, static_cast<TSum>(0));
    }
    SyncAll();
}

template <typename TIdx, typename T, typename TSum, bool ADJ_A, bool ADJ_B>
__aicore__ inline void SparseTensorDenseMatMulB16<TIdx, T, TSum, ADJ_A, ADJ_B>::InitTQueBuffer()
{
    int64_t ubFactorNum = (blockIdx_ == tilingData_->initAndOutUsedCoreNum - 1) ? tilingData_->outTailCoreUbFactor :
                                                                                  tilingData_->outFormerCoreUbFactor;
    int64_t onceBlockAlignNum = static_cast<int64_t>(BLOCK_SIZE / sizeof(TSum));
    int64_t UbFactorNumAlign = ops::CeilAlign(ubFactorNum, onceBlockAlignNum);
    pipe_->InitBuffer(wsInQue_, BUFFER_NUM, UbFactorNumAlign * sizeof(TSum));
    pipe_->InitBuffer(yOutQue_, BUFFER_NUM, UbFactorNumAlign * sizeof(T));
}

template <typename TIdx, typename T, typename TSum, bool ADJ_A, bool ADJ_B>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void ComputeB16(
    const int32_t usedCoreNum, const int32_t currCoreIdx, const int32_t elemNum, const int32_t m, const int32_t n,
    const int32_t p, __gm__ TIdx* x1IndicesGmAddr, __gm__ T* x1ValuesGmAddr, __gm__ T* x2GmAddr,
    __gm__ TSum* workspaceGmAddr)
{
    // 总共有usedCoreNum*ThreadNum个线程，每个线程所在的位置为currCoreIdx*ThreadNum+LocalThreadIdx
    for (int32_t elemIdx = currCoreIdx * Simt::GetThreadNum() + Simt::GetThreadIdx(); elemIdx < elemNum;
         elemIdx += usedCoreNum * Simt::GetThreadNum()) {
        // 计算索引i、j、k，用于找到v1=x1(i,k)和v2=x2GmAddr(k,j)
        // i、j、k 统一转成int32类型
        int32_t x1VecIdx = elemIdx / p;
        int32_t j = elemIdx % p;
        int32_t i = static_cast<int32_t>(x1IndicesGmAddr[INDICES_DIM_1 * x1VecIdx]);
        int32_t k = static_cast<int32_t>(x1IndicesGmAddr[INDICES_DIM_1 * x1VecIdx + 1]);
        if constexpr (ADJ_A) {
            i = static_cast<int32_t>(
                x1IndicesGmAddr[INDICES_DIM_1 * x1VecIdx + 1]); // 转置前x1_indices存的每对元素是[列, 行]
            k = static_cast<int32_t>(x1IndicesGmAddr[INDICES_DIM_1 * x1VecIdx]);
        }
        // 读取v1和v2
        TSum v1 = static_cast<TSum>(x1ValuesGmAddr[x1VecIdx]);
        TSum v2 = static_cast<TSum>(x2GmAddr[k * p + j]);
        if constexpr (ADJ_B) {
            // 转置后x2(k, j)的元素位置，在转置前的x2(j, k)，且转置前的shape=(p, n)
            v2 = static_cast<TSum>(x2GmAddr[j * n + k]);
        }
        // 累加到对应位置
        __gm__ TSum* outAddr = workspaceGmAddr + i * p + j;
        Simt::AtomicAdd(outAddr, v1 * v2);
    }
}

template <typename TIdx, typename T, typename TSum, bool ADJ_A, bool ADJ_B>
__aicore__ inline void SparseTensorDenseMatMulB16<TIdx, T, TSum, ADJ_A, ADJ_B>::Process()
{
    if (blockIdx_ < tilingData_->computeUsedCoreNum) {
        __gm__ TIdx* x1IndicesGmAddr = (__gm__ TIdx*)x1IndicesGm_.GetPhyAddr();
        __gm__ T* x1ValuesGmAddr = (__gm__ T*)x1ValuesGm_.GetPhyAddr();
        __gm__ T* x2GmAddr = (__gm__ T*)x2Gm_.GetPhyAddr();
        __gm__ TSum* workspaceGmAddr = (__gm__ TSum*)workspaceGm_.GetPhyAddr();
        Simt::VF_CALL<ComputeB16<TIdx, T, TSum, ADJ_A, ADJ_B>>(
            Simt::Dim3{SIMT_MAX_THREAD_NUM, 1, 1}, tilingData_->computeUsedCoreNum, blockIdx_,
            tilingData_->computeTotalElemNum, tilingData_->computeM, tilingData_->computeN,
            tilingData_->computeP, x1IndicesGmAddr, x1ValuesGmAddr, x2GmAddr, workspaceGmAddr);
    }
    SyncAll();
    if (blockIdx_ < tilingData_->initAndOutUsedCoreNum) {
        ProcessYResult();
    }
}

template <typename TIdx, typename T, typename TSum, bool ADJ_A, bool ADJ_B>
__aicore__ inline void SparseTensorDenseMatMulB16<TIdx, T, TSum, ADJ_A, ADJ_B>::ProcessYResult()
{
    uint64_t ubLoopTime = (blockIdx_ == tilingData_->initAndOutUsedCoreNum - 1) ? tilingData_->outTailCoreUbLoopTimes :
                                                                                  tilingData_->outFormerCoreUbLoopTimes;
    uint64_t processNumPerLoop = (blockIdx_ == tilingData_->initAndOutUsedCoreNum - 1) ?
                                     tilingData_->outTailCoreUbFactor :
                                     tilingData_->outFormerCoreUbFactor;

    for (uint64_t ubLoopNum = 0; ubLoopNum < ubLoopTime; ++ubLoopNum) {
        if (ubLoopNum == ubLoopTime - 1) {
            processNumPerLoop = (blockIdx_ == tilingData_->initAndOutUsedCoreNum - 1) ?
                                    tilingData_->outTailCoreUbTailFactor :
                                    tilingData_->outFormerCoreUbTailFactor;
        }
        DataCopyExtParams dataCopyInParams = {1, static_cast<uint32_t>(processNumPerLoop * sizeof(TSum)), 0, 0, 0};
        DataCopyExtParams dataCopyOutParams = {1, static_cast<uint32_t>(processNumPerLoop * sizeof(T)), 0, 0, 0};
        uint64_t resultGmOffset =
            blockIdx_ * tilingData_->initAndOutFormerCoreElemNum + ubLoopNum * tilingData_->outFormerCoreUbFactor;
        CopyInYWorkspace(resultGmOffset, dataCopyInParams);
        CastY(processNumPerLoop);
        CopyOutY(resultGmOffset, dataCopyOutParams);
    }
}

template <typename TIdx, typename T, typename TSum, bool ADJ_A, bool ADJ_B>
__aicore__ inline void SparseTensorDenseMatMulB16<TIdx, T, TSum, ADJ_A, ADJ_B>::CopyInYWorkspace(
    const uint64_t& resultGmOffset, const DataCopyExtParams& dataCopyParams)
{
    LocalTensor<TSum> tempResultTensor = wsInQue_.AllocTensor<TSum>();
    DataCopyPadExtParams dataCopyPadParams{false, 0, 0, static_cast<TSum>(0)};
    DataCopyPad(tempResultTensor, workspaceGm_[resultGmOffset], dataCopyParams, dataCopyPadParams);
    wsInQue_.EnQue(tempResultTensor);
}

template <typename TIdx, typename T, typename TSum, bool ADJ_A, bool ADJ_B>
__aicore__ inline void SparseTensorDenseMatMulB16<TIdx, T, TSum, ADJ_A, ADJ_B>::CastY(const uint64_t& processNumPerLoop)
{
    LocalTensor<TSum> tempResultTensor = wsInQue_.DeQue<TSum>();
    LocalTensor<T> resultTensor = yOutQue_.AllocTensor<T>();
    Cast(resultTensor, tempResultTensor, RoundMode::CAST_RINT, processNumPerLoop);
    wsInQue_.FreeTensor(tempResultTensor);
    yOutQue_.EnQue(resultTensor);
}

template <typename TIdx, typename T, typename TSum, bool ADJ_A, bool ADJ_B>
__aicore__ inline void SparseTensorDenseMatMulB16<TIdx, T, TSum, ADJ_A, ADJ_B>::CopyOutY(
    const uint64_t& resultGmOffset, const DataCopyExtParams& dataCopyParams)
{
    LocalTensor<T> resultTensor = yOutQue_.DeQue<T>();
    DataCopyPad(yGm_[resultGmOffset], resultTensor, dataCopyParams);
    yOutQue_.FreeTensor(resultTensor);
}

} // namespace SparseTensorDenseMatMul

#endif // SPARSE_TENSOR_DENSE_MAT_MUL_B16_H
