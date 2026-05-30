/**
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file inplace_index_fill_simt_dense_indices.h
 * \brief DenseIndices kernel: Sort+Unique deduplication + mask + sequential write
 *        适用于 indicesNum >= N*5 的极端稠密场景
 */
#ifndef INPLACE_INDEX_FILL_SIMT_DENSE_INDICES_H_
#define INPLACE_INDEX_FILL_SIMT_DENSE_INDICES_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "inplace_index_fill_common.h"
#include "inplace_index_fill_struct.h"
#include "simt_api/asc_simt.h"

namespace InplaceIndexFillDenseIndices {
using namespace AscendC;
using namespace InplaceIndexFill;

// ============================================================================
// 前向声明
// ============================================================================

// SetMaskSimt: 使用 Sort+Unique 去重后的结果填充 mask
template <typename T_X, typename INDEX_TYPE, typename COM_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_THREAD_NUM) inline void SetMaskSimt(
    __local_mem__ INDEX_TYPE* indicesLocalAddr,
    __local_mem__ int32_t* uniqueIndicesAddr,
    __gm__ int8_t* mask,
    COM_T processNum, COM_T n, int64_t length);

// ============================================================================
// 主类定义
// ============================================================================
template <typename T_X, typename INDEX_TYPE, typename COM_T>
class InplaceIndexFillDenseIndicesImpl {
public:
    __aicore__ inline InplaceIndexFillDenseIndicesImpl(
        const InplaceIndexFill::InplaceIndexFillSimtDenseIndicesTilingData* tilingData, TPipe* pipe)
        : pipe_(pipe), tilingData_(tilingData),
          blockIdx_(GetBlockIdx()), blockNum_(GetBlockNum()) {}

    __aicore__ inline void Init(GM_ADDR indices, GM_ADDR value, GM_ADDR workspace);
    __aicore__ inline void Process(__gm__ T_X* x);

private:
    __aicore__ inline void InitMask(GM_ADDR workspace);
    __aicore__ inline void BuildIndicesMask();
    __aicore__ inline void CopyInIndices(int64_t offset, int64_t dataLen);
    __aicore__ inline void FillMaskWithUniqueIndices(int64_t length);

private:
    TPipe* pipe_;
    // indices 搬运队列
    TQue<QuePosition::VECIN, SIMT_DB_BUFFER> indicesQueue_;
    // Sort 输出 buffer
    TBuf<QuePosition::VECCALC> indicesBuff_;
    // Unique ID 输出 buffer
    TBuf<QuePosition::VECCALC> uniqueIdBuff_;

    GlobalTensor<T_X> valueGm_;
    GlobalTensor<INDEX_TYPE> indicesGm_;
    GlobalTensor<int8_t> maskGm_;

    const InplaceIndexFill::InplaceIndexFillSimtDenseIndicesTilingData* tilingData_;
    uint32_t blockIdx_ = 0;
    uint32_t blockNum_ = 0;
    T_X fillValue;
};

// ============================================================================
// InitMask：初始化 mask 位图为全 0（多核并行）
// ============================================================================
template <typename T_X, typename INDEX_TYPE, typename COM_T>
__aicore__ inline void InplaceIndexFillDenseIndicesImpl<T_X, INDEX_TYPE, COM_T>::InitMask(GM_ADDR workspace)
{
    int64_t n = tilingData_->n;
    InplaceIndexFillCommon::InitMaskZero(maskGm_, workspace, n, blockIdx_, blockNum_);
}

// ============================================================================
// CopyInIndices：将 indices 的一个分片搬入 UB
// ============================================================================
template <typename T_X, typename INDEX_TYPE, typename COM_T>
__aicore__ inline void InplaceIndexFillDenseIndicesImpl<T_X, INDEX_TYPE, COM_T>::CopyInIndices(
    int64_t offset, int64_t dataLen)
{
    DataCopyExtParams inParams = {
        static_cast<uint16_t>(1),
        static_cast<uint32_t>(dataLen * sizeof(INDEX_TYPE)),
        static_cast<uint32_t>(0),
        static_cast<uint32_t>(0),
        static_cast<uint32_t>(0)
    };
    DataCopyPadExtParams<INDEX_TYPE> padParams = {
        false, static_cast<uint8_t>(0), static_cast<uint8_t>(0), static_cast<INDEX_TYPE>(0)
    };
    LocalTensor<INDEX_TYPE> indicesLocal = indicesQueue_.AllocTensor<INDEX_TYPE>();
    DataCopyPad(indicesLocal, indicesGm_[offset], inParams, padParams);
    indicesQueue_.EnQue(indicesLocal);
}

// ============================================================================
// SetMaskSimt：使用去重后的唯一索引填充 mask
// ============================================================================
template <typename T_X, typename INDEX_TYPE, typename COM_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_THREAD_NUM) inline void SetMaskSimt(
    __local_mem__ INDEX_TYPE* indicesLocalAddr,
    __local_mem__ int32_t* uniqueIndicesAddr,
    __gm__ int8_t* mask,
    COM_T processNum, COM_T n, int64_t length)
{
    COM_T threadIdx = static_cast<COM_T>(Simt::GetThreadIdx());
    COM_T threadNum = static_cast<COM_T>(Simt::GetThreadNum());

    // 遍历去重后的唯一索引
    for (COM_T i = threadIdx; i < processNum; i += threadNum) {
        // uniqueIndicesAddr[i] 是排序后数组中的位置索引
        int32_t id = uniqueIndicesAddr[i];
        if (id < 0 || id >= length) {
            continue;
        }
        // 通过位置索引找到实际的 indices 值
        INDEX_TYPE index = indicesLocalAddr[id];
        // 处理负索引
        index = index < 0 ? index + n : index;
        if (index < 0 || index >= n) {
            continue;
        }
        // 填充 mask
        mask[index] = 1;
    }
}

// ============================================================================
// FillMaskWithUniqueIndices：Sort+Unique 去重后填充 mask
// ============================================================================
template <typename T_X, typename INDEX_TYPE, typename COM_T>
__aicore__ inline void InplaceIndexFillDenseIndicesImpl<T_X, INDEX_TYPE, COM_T>::FillMaskWithUniqueIndices(
    int64_t length)
{
    // SortAndUnique 输出布局：32字节前哨兵区 + length*sizeof(INDEX_TYPE) + 后哨兵区
    int64_t shiftOffset = UB_AGLIN_VALUE / sizeof(INDEX_TYPE);
    LocalTensor<INDEX_TYPE> sortedOut = indicesBuff_.Get<INDEX_TYPE>();
    LocalTensor<int32_t> uniqueIndicesOut = uniqueIdBuff_.Get<int32_t>();
    LocalTensor<INDEX_TYPE> indicesLocal = indicesQueue_.DeQue<INDEX_TYPE>();

    // 执行 Sort+Unique：返回去重后的唯一元素个数
    uint32_t uniqueNum = SortAndUnique(sortedOut, uniqueIndicesOut, indicesLocal, length);

    COM_T n = static_cast<COM_T>(tilingData_->n);
    // sortedOut 的实际数据起始地址需要跳过前哨兵区
    __local_mem__ INDEX_TYPE* actualSortOut =
        ((__local_mem__ INDEX_TYPE*)sortedOut.GetPhyAddr()) + shiftOffset;

    // 用去重后的唯一索引填充 mask
    Simt::VF_CALL<SetMaskSimt<T_X, INDEX_TYPE, COM_T>>(
        Simt::Dim3{SIMT_THREAD_NUM},
        actualSortOut,
        (__local_mem__ int32_t*)uniqueIndicesOut.GetPhyAddr(),
        (__gm__ int8_t*)maskGm_.GetPhyAddr(),
        uniqueNum, n, length);

    indicesQueue_.FreeTensor(indicesLocal);
}

// ============================================================================
// BuildIndicesMask：分批加载 indices 并去重构建 mask
// ============================================================================
template <typename T_X, typename INDEX_TYPE, typename COM_T>
__aicore__ inline void InplaceIndexFillDenseIndicesImpl<T_X, INDEX_TYPE, COM_T>::BuildIndicesMask()
{
    int64_t indicesUbFactor = tilingData_->indicesUbFactor;
    int64_t indicesNum = tilingData_->indicesNum;
    // 计算总块数
    int64_t totalBlocks = ops::Ceil(indicesNum, indicesUbFactor);
    int64_t usedCoreNum = AscendC::Std::min(totalBlocks, static_cast<int64_t>(blockNum_));
    // 尾块大小
    int64_t tailIndicesUbFactor = indicesNum - (totalBlocks - 1) * indicesUbFactor;

    // 超出范围的核不参与 mask 构建
    if (blockIdx_ >= static_cast<uint32_t>(usedCoreNum)) {
        return;
    }

    // 将 totalBlocks 分配给各核
    int64_t tailBlocksPerCore = totalBlocks / usedCoreNum;
    int64_t frontBlocksPerCore = totalBlocks % usedCoreNum == 0
        ? tailBlocksPerCore : tailBlocksPerCore + 1;
    int64_t frontCoreNum = totalBlocks % usedCoreNum;

    // 计算当前核的块范围 [start, end)
    uint64_t loop = 0;
    uint64_t start = 0;
    uint64_t end = 0;
    if (blockIdx_ < static_cast<uint32_t>(frontCoreNum)) {
        loop = frontBlocksPerCore;
        start = frontBlocksPerCore * blockIdx_;
        end = start + loop;
    } else {
        loop = tailBlocksPerCore;
        start = frontBlocksPerCore * frontCoreNum +
                (blockIdx_ - frontCoreNum) * tailBlocksPerCore;
        end = start + loop;
    }

    // 初始化 UB buffer
    int64_t vfLen = platform::GetVRegSize() / sizeof(INDEX_TYPE);
    int64_t indicesBuffSize = ops::CeilDiv(indicesUbFactor + 1, vfLen) * vfLen
                            * sizeof(INDEX_TYPE) + UB_AGLIN_VALUE;
    pipe_->Reset();
    pipe_->InitBuffer(indicesQueue_, SIMT_DB_BUFFER, indicesUbFactor * sizeof(INDEX_TYPE));
    pipe_->InitBuffer(indicesBuff_, indicesBuffSize);
    pipe_->InitBuffer(uniqueIdBuff_, indicesUbFactor * sizeof(int32_t));

    // 逐块处理
    int64_t offset = 0;
    for (int64_t i = start; i < static_cast<int64_t>(end); i++) {
        offset = i * indicesUbFactor;
        if (i == totalBlocks - 1) {
            // 尾块
            CopyInIndices(offset, tailIndicesUbFactor);
            FillMaskWithUniqueIndices(tailIndicesUbFactor);
            break;
        }
        CopyInIndices(offset, indicesUbFactor);
        FillMaskWithUniqueIndices(indicesUbFactor);
    }
}

// ============================================================================
// Init
// ============================================================================
template <typename T_X, typename INDEX_TYPE, typename COM_T>
__aicore__ inline void InplaceIndexFillDenseIndicesImpl<T_X, INDEX_TYPE, COM_T>::Init(
    GM_ADDR indices, GM_ADDR value, GM_ADDR workspace)
{
    valueGm_.SetGlobalBuffer((__gm__ T_X*)(value));
    indicesGm_.SetGlobalBuffer((__gm__ INDEX_TYPE*)(indices));
    InitMask(workspace);
    fillValue = valueGm_.GetValue(0);
}

// ============================================================================
// Process：主流程
// ============================================================================
template <typename T_X, typename INDEX_TYPE, typename COM_T>
__aicore__ inline void InplaceIndexFillDenseIndicesImpl<T_X, INDEX_TYPE, COM_T>::Process(__gm__ T_X* x)
{
    // Sort+Unique 去重构建 mask
    BuildIndicesMask();
    SyncAll();

    // 顺序遍历 N*P*Q，按 mask 填充
    COM_T p = static_cast<COM_T>(tilingData_->p);
    COM_T q = static_cast<COM_T>(tilingData_->q);
    COM_T n = static_cast<COM_T>(tilingData_->n);

    InplaceIndexFillCommon::CallMaskBasedFill<T_X, INDEX_TYPE, COM_T>(
        x, (__gm__ int8_t*)maskGm_.GetPhyAddr(), fillValue, n, p, q);
}

} // namespace InplaceIndexFillDenseIndices

#endif // INPLACE_INDEX_FILL_SIMT_DENSE_INDICES_H_