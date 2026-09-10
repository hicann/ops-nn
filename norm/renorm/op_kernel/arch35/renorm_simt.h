/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _RENORM_SIMT_H_
#define _RENORM_SIMT_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "simt_api/asc_simt.h"
#include "renorm_tiling_data.h"
#include "renorm_tiling_key.h"

// ============================================================
// 纯 SIMT 路径小白解读
// ============================================================
// SIMT (Single Instruction Multiple Threads) 模式类似 GPU 的 CUDA 编程:
//   一次性启动 512 个线程并行，每个线程独立处理若干个 slice，互不干扰。
//
// 与 SIMD (Template E 等) 模式的关键区别:
//   - SIMD: 一条向量指令同时处理一个 block 的多个元素，需要 UB buffer 管理、
//           流水线同步(SetFlag/WaitFlag)、双缓冲等复杂调度，性能高但代码复杂。
//   - SIMT: 每个线程用标量代码(powf/sqrtf/+-*/)直接读写 GM，不需要 UB buffer、
//           不需要流水线同步、不需要 ReduceSum。代码非常简单直观。
//
// 优缺点:
//   - 优点: 代码极简，易于理解和维护，适合 blockSize 较大或形状不规则的场景。
//   - 缺点: 标量访存和计算效率不如 SIMD 向量化，性能通常低于 SIMD 路径。
//
// 本文件的 RenormSimtProcess 是真正跑在线程上的"向量函数(VF)"，
// RenormSimt 类只是它的外层包装(负责读 tiling、切分 slice 给本核、发起 VF 调用)。

namespace NsRenorm {

using namespace AscendC;

// SIMT 线程数（待定，后续根据性能数据调整）
constexpr uint32_t SIMT_THREAD_NUM = 512;

// normMode 常量（与 tiling 中保持一致）
constexpr int32_t SIMT_NORM_MODE_P_POSITIVE = 0;
constexpr int32_t SIMT_NORM_MODE_P_ZERO = 1;
constexpr int32_t SIMT_NORM_MODE_P_INF = 2;
constexpr int32_t SIMT_NORM_MODE_MAXNORM_ZERO = 3;

// ============================================================
// 纯 SIMT VF 函数：每个线程独立完成 norm 计算 -> scale -> 写 output
// 无需 GM workspace，无需 DCache 同步，无需 SIMD/SIMT 跨范式同步
// ============================================================
template <typename T>
// __simt_vf__: 声明这是一个 SIMT "向量函数(Vector Function)"，会被 512 个线程并行执行，
//             函数内的标量代码在每个线程上各跑一份，threadIdx.x 用来区分线程身份。
// LAUNCH_BOUND(SIMT_THREAD_NUM): 告诉编译器本函数最大启动 512 个线程，用于寄存器分配优化。
__simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_THREAD_NUM) inline void RenormSimtProcess(
    __gm__ T* inputGm, __gm__ T* outputGm, int64_t sliceCount, int64_t blockSize, int64_t numBlocks, int64_t startSlice,
    int64_t slicesThisCore, float p, float maxNorm, float eps, int32_t normMode)
{
    // tid = 当前线程编号(0..511)。所有线程执行同一份代码，但用 tid 区分各自处理的数据。
    uint32_t tid = threadIdx.x;

    // 线程网格循环: 每个线程处理 slicesThisCore/SIMT_THREAD_NUM 个 slice(向上取整)。
    // 线程 tid 处理 s = tid, tid+512, tid+1024, ... 直到超过本核负责的 slice 数。
    // 这就是 SIMT 的"网格步进(stride)"循环模式，保证所有 slice 被均匀分配给 512 个线程。
    for (int64_t s = static_cast<int64_t>(tid); s < slicesThisCore; s += SIMT_THREAD_NUM) {
        int64_t sliceIdx = startSlice + s;

        // maxNorm=0: 直接写零
        if (normMode == SIMT_NORM_MODE_MAXNORM_ZERO) {
            for (int64_t b = 0; b < numBlocks; ++b) {
                int64_t baseOffset = b * sliceCount * blockSize + sliceIdx * blockSize;
                for (int64_t e = 0; e < blockSize; ++e) {
                    outputGm[baseOffset + e] = static_cast<T>(0.0f);
                }
            }
            continue;
        }

        // Step 1: 标量累加范数。遍历这个 slice 在所有 block 中的元素，直接从 GM 标量读取。
        // 与 SIMD 路径不同: 这里用标量 for 循环 + 标量运算(+=/powf)累加，结果存在线程局部变量 norm 中。
        // 三种 normMode 都在这一个循环里处理:
        //   P_POSITIVE: 累加 |x|^p (p=2 用 val*val，p=1 用 absVal，其他用 powf)
        //   P_ZERO:     统计非零个数(每遇非零 norm+=1)
        //   P_INF:      取最大绝对值(每遇更大值则更新 norm)
        // 此时 norm 是"原始累加值"，尚未做开根号等后处理(见 Step 2)。
        // Step 1: 计算 raw norm（直接 GM 标量读）
        float norm = 0.0f;
        for (int64_t b = 0; b < numBlocks; ++b) {
            int64_t baseOffset = b * sliceCount * blockSize + sliceIdx * blockSize;
            for (int64_t e = 0; e < blockSize; ++e) {
                float val = static_cast<float>(inputGm[baseOffset + e]);
                float absVal = val < 0.0f ? -val : val;

                if (normMode == SIMT_NORM_MODE_P_POSITIVE) {
                    if (p == 2.0f) {
                        norm += val * val;
                    } else if (p == 1.0f) {
                        norm += absVal;
                    } else {
                        norm += powf(absVal, p);
                    }
                } else if (normMode == SIMT_NORM_MODE_P_ZERO) {
                    if (val != 0.0f) {
                        norm += 1.0f;
                    }
                } else if (normMode == SIMT_NORM_MODE_P_INF) {
                    if (absVal > norm) {
                        norm = absVal;
                    }
                }
            }
        }

        // Step 2: 标量后处理，把 Step1 的累加值变成真正的范数。
        // 仅 P_POSITIVE 且 p≠1 时需要: p=1 时范数就是 Σ|x| 无需处理; p=2 时开根号 sqrtf;
        // 通用 p 时 norm = (Σ|x|^p)^(1/p) = powf(norm, 1/p)，先用 safeNorm 防 0。
        // P_INF 和 P_ZERO 在 Step1 已得到最终值(最大值/计数)，无需后处理。
        // Step 2: 标量变换（sqrtf/powf，线程局部变量）
        if (normMode == SIMT_NORM_MODE_P_POSITIVE && p != 1.0f) {
            if (p == 2.0f) {
                norm = sqrtf(norm);
            } else {
                float safeNorm = norm > eps ? norm : eps;
                norm = powf(safeNorm, 1.0f / p);
            }
        }

        // Step 3: 计算 scale。这里用标量 if-else(每个线程独立判断自己的 slice):
        //   若 norm > maxNorm: scale = maxNorm / max(norm, eps)  ← 需要缩放回 maxNorm
        //   否则:             scale = 1.0                        ← 不超限，原样输出
        // (SIMD 路径用 Compare+Select 向量化实现同样的条件，这里直接标量 if 更简单)
        // Step 3: 计算 scale（R-01 修复：与 SIMD 路径 ComputeScale 逻辑一致）
        float scale;
        if (norm > maxNorm) {
            float denom = norm > eps ? norm : eps;
            scale = maxNorm / denom;
        } else {
            scale = 1.0f;
        }

        // Step 4: 应用 scale 并写回输出。再次遍历所有 block，从 GM 标量读入元素，
        // 乘以 scale 后转回原 dtype 写到 outputGm 对应位置。
        // 注意: 这里重新读一遍输入(SIMD 路径也是两遍 Pass)，因为算范数和写输出是两个独立阶段。
        // Step 4: 应用 scale，写 output（直接 GM 标量读写）
        for (int64_t b = 0; b < numBlocks; ++b) {
            int64_t baseOffset = b * sliceCount * blockSize + sliceIdx * blockSize;
            for (int64_t e = 0; e < blockSize; ++e) {
                float val = static_cast<float>(inputGm[baseOffset + e]);
                outputGm[baseOffset + e] = static_cast<T>(val * scale);
            }
        }
    }
}

// ============================================================
// RenormSimt 类：纯 SIMT 包装
// ============================================================
template <typename D_T_X>
class RenormSimt {
public:
    __aicore__ inline RenormSimt() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const RenormTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __gm__ D_T_X* inputGm_;
    __gm__ D_T_X* outputGm_;

    int64_t totalElements_ = 0;
    int64_t sliceCount_ = 0;
    int64_t blockSize_ = 0;
    int64_t numBlocks_ = 0;
    int64_t slicesPerCore_ = 0;
    int64_t startSlice_ = 0;
    int64_t endSlice_ = 0;
    int64_t slicesThisCore_ = 0;
    float p_ = 0.0f;
    float maxNorm_ = 0.0f;
    float eps_ = 0.0f;
    int32_t normMode_ = 0;
};

template <typename D_T_X>
__aicore__ inline void RenormSimt<D_T_X>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace,
                                               const RenormTilingData* tilingData)
{
    totalElements_ = tilingData->totalElements;
    sliceCount_ = tilingData->sliceCount;
    blockSize_ = tilingData->blockSize;
    numBlocks_ = tilingData->numBlocks;
    slicesPerCore_ = tilingData->slicesPerCore;
    p_ = tilingData->p;
    maxNorm_ = tilingData->maxNorm;
    eps_ = tilingData->eps;
    normMode_ = tilingData->normMode;

    if (totalElements_ == 0 || sliceCount_ == 0 || slicesPerCore_ == 0) {
        return;
    }

    int64_t blockIdx = GetBlockIdx();
    startSlice_ = blockIdx * slicesPerCore_;
    endSlice_ = startSlice_ + slicesPerCore_;
    if (endSlice_ > sliceCount_) {
        endSlice_ = sliceCount_;
    }
    slicesThisCore_ = endSlice_ - startSlice_;
    if (slicesThisCore_ <= 0) {
        return;
    }

    inputGm_ = (__gm__ D_T_X*)x;
    outputGm_ = (__gm__ D_T_X*)y;
}

template <typename D_T_X>
__aicore__ inline void RenormSimt<D_T_X>::Process()
{
    if (slicesThisCore_ <= 0) {
        return;
    }

    Simt::VF_CALL<RenormSimtProcess<D_T_X>>(Simt::Dim3(SIMT_THREAD_NUM), inputGm_, outputGm_, sliceCount_, blockSize_,
                                            numBlocks_, startSlice_, slicesThisCore_, p_, maxNorm_, eps_, normMode_);
}

} // namespace NsRenorm

#endif // _RENORM_SIMT_H_
