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
 * \file inplace_index_fill_simt.h
 * \brief SIMT kernel implementation with dual-path optimization:
 *        Path A (sparse): iterate over indicesNum*P*Q, random write
 *        Path B (dense):  build mask bitmap, iterate over N*P*Q, sequential write
 */
#ifndef INPLACE_INDEX_FILL_SIMT_H_
#define INPLACE_INDEX_FILL_SIMT_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "inplace_index_fill_struct.h"
#include "inplace_index_fill_common.h"
#include "simt_api/asc_simt.h"

namespace InplaceIndexFillSimt {
using namespace AscendC;
using namespace InplaceIndexFill;

// ============================================================================
// 前向声明：SIMT 核函数
// ============================================================================

// MaskFill: 多线程并行构建 mask 位图
// 每个线程处理 indices 数组的一部分，将对应 mask[nIdx] 置 1
template <typename T_X, typename INDEX_TYPE, typename COM_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_THREAD_NUM) inline void InplaceIndexFillSimtMaskFill(
    uint64_t blockIdx, uint64_t blockNum,
    __gm__ INDEX_TYPE* indices, __gm__ int8_t* mask,
    COM_T indicesNum, uint64_t n);

// ComputeByIndices (路径A): 稀疏场景，按 indicesNum*P*Q 遍历
// 每个线程从线性索引反算 (sliceId, pIdx, qIdx)，读 indices[sliceId] 得到 nIdx，随机写 x[offset]
template <typename T_X, typename INDEX_TYPE, typename COM_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_THREAD_NUM) inline void InplaceIndexFillSimtComputeByIndices(
    __gm__ T_X* x, __gm__ INDEX_TYPE* indices, T_X value,
    COM_T total_num, COM_T p, COM_T n, COM_T q,
    COM_T slice_size, COM_T shift, COM_T magic,
    COM_T shift_q, COM_T magic_q);

// ============================================================================
// 主类定义
// ============================================================================
template <typename T_X, typename INDEX_TYPE, typename COM_T>
class InplaceIndexFillSimtImpl {
public:
    __aicore__ inline InplaceIndexFillSimtImpl(
        const InplaceIndexFill::InplaceIndexFillSimtTilingData* tilingData)
        : tilingData_(tilingData),
          blockIdx_(GetBlockIdx()), blockNum_(GetBlockNum()) {}

    __aicore__ inline void Init(GM_ADDR indices, GM_ADDR value, GM_ADDR workspace);
    __aicore__ inline void Process(__gm__ T_X* x, GM_ADDR workspace);

private:
    __aicore__ inline void BuildIndicesMask(GM_ADDR workspace);

private:
    GlobalTensor<T_X> valueGm_;     // value 标量的 GM 地址
    GlobalTensor<INDEX_TYPE> indicesGm_;  // indices 数组的 GM 地址
    GlobalTensor<int8_t> maskGm_;   // mask 位图的 GM 地址（workspace 上）
    const InplaceIndexFill::InplaceIndexFillSimtTilingData* tilingData_;
    uint32_t blockIdx_ = 0;         // 当前核的 block ID
    uint32_t blockNum_ = 0;         // 总核数
    T_X fillValue;                  // 填充值（从 GM 读取后缓存）
};

// ============================================================================
// MaskFill 实现：构建 mask 位图
// ============================================================================
template <typename T_X, typename INDEX_TYPE, typename COM_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_THREAD_NUM) inline void InplaceIndexFillSimtMaskFill(
    uint64_t blockIdx, uint64_t blockNum,
    __gm__ INDEX_TYPE* indices, __gm__ int8_t* mask,
    COM_T indicesNum, uint64_t n)
{
    // 计算全局线程 ID：blockIdx * 每核线程数 + 核内线程 ID
    COM_T threadIdx = static_cast<COM_T>(blockIdx * Simt::GetThreadNum() + Simt::GetThreadIdx());
    // 计算全局线程总数：总核数 * 每核线程数
    COM_T threadNum = static_cast<COM_T>(blockNum * Simt::GetThreadNum());

    // 每个线程以 stride=threadNum 遍历 indices 数组
    for (COM_T idx = threadIdx; idx < indicesNum; idx += threadNum) {
        // 读取 indices[idx]，处理负索引
        INDEX_TYPE nIdx = static_cast<INDEX_TYPE>(indices[idx]);
        nIdx = nIdx >= 0 ? nIdx : nIdx + n;
        // 越界检查：跳过无效索引
        if (nIdx < 0 || nIdx >= static_cast<INDEX_TYPE>(n)) {
            continue;
        }
        // 将 mask[nIdx] 置 1，表示该位置需要填充
        // 多线程可能同时写同一位置（indices 有重复），但写入值相同，结果正确
        mask[nIdx] = static_cast<int8_t>(1);
    }
}

// ============================================================================
// BuildIndicesMask：初始化 mask 并调用 MaskFill
// ============================================================================
template <typename T_X, typename INDEX_TYPE, typename COM_T>
__aicore__ inline void InplaceIndexFillSimtImpl<T_X, INDEX_TYPE, COM_T>::BuildIndicesMask(GM_ADDR workspace)
{
    int64_t n = tilingData_->n;
    InplaceIndexFillCommon::InitMaskZero(maskGm_, workspace, n, blockIdx_, blockNum_);

    // 所有核并行填充 mask
    COM_T indicesNum = static_cast<COM_T>(tilingData_->indicesNum);
    Simt::VF_CALL<InplaceIndexFillSimtMaskFill<T_X, INDEX_TYPE, COM_T>>(
        Simt::Dim3(SIMT_THREAD_NUM),
        blockIdx_, blockNum_,
        (__gm__ INDEX_TYPE*)(indicesGm_.GetPhyAddr()),
        (__gm__ int8_t*)(maskGm_.GetPhyAddr()),
        indicesNum, static_cast<uint64_t>(n));

    // 核间同步：确保所有核的 mask 填充完毕后再进入计算阶段
    SyncAll();
}

// ============================================================================
// Init：初始化 GM 地址
// ============================================================================
template <typename T_X, typename INDEX_TYPE, typename COM_T>
__aicore__ inline void InplaceIndexFillSimtImpl<T_X, INDEX_TYPE, COM_T>::Init(
    GM_ADDR indices, GM_ADDR value, GM_ADDR workspace)
{
    // 设置 value 和 indices 的 GM 地址
    valueGm_.SetGlobalBuffer((__gm__ T_X*)(value));
    indicesGm_.SetGlobalBuffer((__gm__ INDEX_TYPE*)(indices));
    // 从 GM 读取 fillValue 标量并缓存到寄存器
    fillValue = valueGm_.GetValue(0);
}

// ============================================================================
// Process：主流程，选择路径A或路径B执行
// ============================================================================
template <typename T_X, typename INDEX_TYPE, typename COM_T>
__aicore__ inline void InplaceIndexFillSimtImpl<T_X, INDEX_TYPE, COM_T>::Process(
    __gm__ T_X* x, GM_ADDR workspace)
{
    // 读取 tiling 参数
    COM_T p = static_cast<COM_T>(tilingData_->p);
    COM_T q = static_cast<COM_T>(tilingData_->q);
    COM_T n = static_cast<COM_T>(tilingData_->n);

    // 第二层运行时分支：根据 indicesNum 与 N 的比值选择路径
    if (tilingData_->indicesNum * 5 <= tilingData_->n) {
        // ================================================================
        // 路径A：稀疏场景（indicesNum * 5 <= N）
        // 遍历量 = indicesNum * P * Q（较小）
        // 缺点：随机读 indices + 随机写 x，但因为遍历量小，总体仍然快
        // ================================================================
        COM_T process_num = static_cast<COM_T>(tilingData_->indicesNum * p * q);
        // slice_size = P * Q，用于从线性索引反算 sliceId
        COM_T slice_size = static_cast<COM_T>(p * q);

        // 预计算 magic number 除法参数（替代昂贵的整数除法）
        COM_T shift = 0;
        COM_T magic = 0;
        GetUintDivMagicAndShift(magic, shift, slice_size);

        COM_T shift_q = 0;
        COM_T magic_q = 0;
        GetUintDivMagicAndShift(magic_q, shift_q, q);

        // 调用路径A的 SIMT 核函数
        Simt::VF_CALL<InplaceIndexFillSimtComputeByIndices<T_X, INDEX_TYPE, COM_T>>(
            Simt::Dim3{SIMT_THREAD_NUM},
            x, (__gm__ INDEX_TYPE*)indicesGm_.GetPhyAddr(),
            fillValue, process_num, p, n, q,
            slice_size, shift, magic, shift_q, magic_q);
    } else {
        // ================================================================
        // 路径B：稠密场景（indicesNum * 5 > N）
        // 先构建 mask 位图，再按 N*P*Q 顺序遍历
        // 优点：顺序写 x[i]，100% 缓存命中，零写冲突
        // ================================================================
        BuildIndicesMask(workspace);

        // 调用路径B的 SIMT 核函数
        InplaceIndexFillCommon::CallMaskBasedFill<T_X, INDEX_TYPE, COM_T>(
            x, (__gm__ int8_t*)maskGm_.GetPhyAddr(), fillValue, n, p, q);
    }
}

// ============================================================================
// 路径A 核函数：ComputeByIndices
// 适用于稀疏场景（indicesNum 远小于 N）
// 遍历空间：indicesNum * P * Q
// ============================================================================
template <typename T_X, typename INDEX_TYPE, typename COM_T>
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_THREAD_NUM) inline void InplaceIndexFillSimtComputeByIndices(
    __gm__ T_X* x, __gm__ INDEX_TYPE* indices, T_X value,
    COM_T total_num, COM_T p, COM_T n, COM_T q,
    COM_T slice_size, COM_T shift, COM_T magic,
    COM_T shift_q, COM_T magic_q)
{
    // 计算全局线程 ID 和总线程数
    COM_T threadIdx = static_cast<COM_T>(Simt::GetBlockIdx() * Simt::GetThreadNum() + Simt::GetThreadIdx());
    COM_T threadNum = static_cast<COM_T>(Simt::GetBlockNum() * Simt::GetThreadNum());

    for (COM_T i = threadIdx; i < total_num; i += threadNum) {
        // 从线性索引 i 反算三维坐标：
        // i 的布局是 [indicesNum, P, Q]，即 i = sliceId * (P*Q) + pIdx * Q + qIdx
        // sliceId = i / slice_size（slice_size = P*Q）
        COM_T sliceId = Simt::UintDiv(i, magic, shift);
        // innerId = i - sliceId * slice_size = pIdx * Q + qIdx
        COM_T innerId = i - (sliceId * slice_size);

        // 从 indices 数组读取 nIdx（随机读）
        INDEX_TYPE nIdx = static_cast<INDEX_TYPE>(indices[sliceId]);
        // 处理负索引：Python 风格的负索引转换
        nIdx = nIdx >= 0 ? nIdx : nIdx + n;
        // 越界检查
        if (nIdx < 0 || nIdx >= static_cast<INDEX_TYPE>(n)) {
            continue;
        }

        // 从 innerId 反算 pIdx 和 qIdx
        // pIdx = innerId / Q
        COM_T pIdx = Simt::UintDiv(innerId, magic_q, shift_q);
        // qIdx = innerId - pIdx * Q = innerId % Q
        COM_T qOffset = innerId - (pIdx * q);

        // 计算目标地址：x 的布局是 [P, N, Q]
        // offset = pIdx * N * Q + nIdx * Q + qIdx
        COM_T offset = pIdx * n * q + nIdx * q + qOffset;
        // 写入填充值（随机写）
        x[offset] = value;
    }
}

} // namespace InplaceIndexFillSimt

#endif // INPLACE_INDEX_FILL_SIMT_H_