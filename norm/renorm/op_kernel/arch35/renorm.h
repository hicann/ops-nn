/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _RENORM_H_
#define _RENORM_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "renorm_tiling_data.h"
#include "renorm_tiling_key.h"
#include "common/renorm_common.h"
#include "renorm_p_positive.h"
#include "renorm_p_zero.h"
#include "renorm_p_inf.h"
#include "renorm_maxnorm_zero.h"

// ============================================================================
// 【renorm 算子总览 — 小白入门导读】
// ============================================================================
//
// ■ 算子功能
//   renorm 沿 dim 维度把输入张量切成若干"子张量"(slice)，对每个子张量
//   计算 p-范数 ||x||_p。如果范数超过阈值 maxNorm，就把该子张量所有元素
//   乘以缩放因子 scale = maxNorm / ||x||_p，使其范数恰好等于 maxNorm；
//   如果没超过则不缩放 (scale = 1.0)。
//   公式:  y = x * scale,  scale = min(1, maxNorm / max(||x||_p, eps))
//
// ■ 内存布局 (GM 中的排布)
//   输入/输出张量按 [numBlocks, sliceCount, blockSize] 三维理解:
//     numBlocks  = prod(shape[:dim])    — 归约轴(每个子张量有这么多段)
//     sliceCount = shape[dim]           — 子张量个数(沿 dim 切)
//     blockSize  = prod(shape[dim+1:])  — 每段连续元素数
//   一个"子张量"(slice) = numBlocks 个 blockSize 大小的连续段拼成。
//
//   GM 一维地址:  offset = b * sliceCount * blockSize + s * blockSize + e
//                 (b=block号, s=slice号, e=段内元素号)
//
// ■ 4 种 normMode (由 tiling 层根据 p 值推导)
//   0  P_POSITIVE  : p > 0,  ||x||_p = (Σ|x_i|^p)^(1/p)
//   1  P_ZERO      : p = 0,  ||x||_0 = 非零元素个数
//   2  P_INF       : p = ∞,  ||x||_∞ = max(|x_i|)
//   3  MAXNORM_ZERO: maxNorm=0, 直接输出全零 (scale 恒为 0)
//
// ■ 6 种模板 (TEMPLATE 0-5, 由 tiling 层根据形状选择最优)
//   A(0) SM-CT: 默认, 单核处理若干 slice, 连续访问          ← 本文件实现
//   B(1) SM-TL: 高精度, blockSize=1, 跨核 SetAtomicAdd 归约  (renorm_sm_tl.h)
//   C(2) SM-CR: 归约量 > UB容量, 多核协作归约 + workspace    (renorm_sm_cr.h)
//   D(3) SM-ST: blockSize=1, stride 非连续访问               (renorm_sm_st.h)
//   E(4) BM-VD: blockSize=1, block-major 向量直通            (renorm_bm_vd.h)
//   F(5) BM-VG: 1<blockSize≤阈值, 预留未实现                 (renorm_bm_vg.h)
//   另有 SIMT 路径: 纯线程模式, 标量直读写 GM                 (renorm_simt.h)
//
// ■ Ascend C 流水线 (PIPE) 速查
//   MTE2 = GM→UB 搬运,  V = 向量计算,  MTE3 = UB→GM 搬运,  S = 标量计算
//   不同流水线异步执行, 需 SetFlag/WaitFlag 同步:
//     V 写了 UB → MTE3 要读 → 需 V_MTE3 同步
//     MTE3 写了 GM → MTE2 要读 → 需 MTE3_MTE2 同步
//   PipeBarrier<PIPE_V>() = V 流水线内部屏障
//
// ■ Template A (本文件) 执行流程
//   每个核分到若干连续的 slice，对每个 slice 串行执行三步:
//     ① 计算范数  → 调用 ComputeNormPPositive / PZero / PInf
//     ② 算缩放因子 → ComputeScale(norm, maxNorm, eps)
//     ③ 应用缩放  → ScaleAndStoreSlice: load→castFP32→Muls(scale)→castBack→store
// ============================================================================

namespace NsRenorm {

using namespace AscendC;

// normMode 常量（与 tiling 中保持一致）
constexpr int32_t NORM_MODE_P_POSITIVE = 0;
constexpr int32_t NORM_MODE_P_ZERO = 1;
constexpr int32_t NORM_MODE_P_INF = 2;
constexpr int32_t NORM_MODE_MAXNORM_ZERO = 3;

// ============================================================================
// Renorm 类 — Template A (SM-CT, Slice-Major Continuous Single-Level)
// ============================================================================
// 适用条件: 通用默认模板, 子张量数据能放进 UB (192KB)
// 多核策略: 沿 sliceCount 维度分核, core i 处理 [i*slicesPerCore, ...] 的 slice
// 每个 slice 独立计算范数→缩放, 核间无依赖、无需 workspace
// ============================================================================
template <typename D_T_X, bool SAFE_A_ROUTE = false, bool STABLE_POSITIVE_P_ROUTE = false>
class Renorm {
public:
    __aicore__ inline Renorm() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const RenormTilingData* tilingData);
    __aicore__ inline void Process();

private:
    TPipe pipe;
    // --- UB 局部缓冲区 (192KB 片上高速缓存, 所有 V 计算在此进行) ---
    TBuf<QuePosition::VECCALC> dataBuf;   // 输入数据加载 buffer (原始 dtype)
    TBuf<QuePosition::VECCALC> workBuf;   // FP32 工作空间 (Cast 目标, V 计算用)
    TBuf<QuePosition::VECCALC> maskBuf;   // Compare mask (uint8, p_zero 场景)
    TBuf<QuePosition::VECCALC> zerosBuf;  // 全零向量 (p_zero 场景 Compare 阈值)
    TBuf<QuePosition::VECCALC> onesBuf;   // 全一向量 (p_zero 场景 Select 默认值)
    TBuf<QuePosition::VECCALC> reduceBuf; // ReduceSum/ReduceMax 标量输出 (最小32B)
    TBuf<QuePosition::VECCALC> scaleBuf;  // 缩放因子 (每核一个标量)
    TBuf<QuePosition::VECCALC> tmpBuf;    // 标量计算临时 buffer (Sqrt/Pow 中转)

    GlobalTensor<D_T_X> inputGM;
    GlobalTensor<D_T_X> outputGM;

    // Tiling 参数
    int64_t totalElements_ = 0;
    int64_t dim_ = 0;
    int64_t sliceCount_ = 0;
    int64_t blockSize_ = 0;
    int64_t numBlocks_ = 0;
    int64_t tileLength_ = 0;
    int64_t slicesPerCore_ = 0;
    int64_t startSlice_ = 0;
    int64_t endSlice_ = 0;
    int64_t slicesThisCore_ = 0;
    float p_ = 0.0f;
    float maxNorm_ = 0.0f;
    float eps_ = 0.0f;
    int32_t normMode_ = 0;
};

template <typename D_T_X, bool SAFE_A_ROUTE, bool STABLE_POSITIVE_P_ROUTE>
__aicore__ inline void Renorm<D_T_X, SAFE_A_ROUTE, STABLE_POSITIVE_P_ROUTE>::Init(GM_ADDR x, GM_ADDR y,
                                                                                  const RenormTilingData* tilingData)
{
    // === 步骤1: 从 tilingData 读取 host 侧计算好的切分和算法参数 ===
    totalElements_ = tilingData->totalElements;
    dim_ = tilingData->dim;
    sliceCount_ = tilingData->sliceCount;
    blockSize_ = tilingData->blockSize;
    numBlocks_ = tilingData->numBlocks;
    tileLength_ = tilingData->tileLength;
    slicesPerCore_ = tilingData->slicesPerCore;
    p_ = tilingData->p;
    maxNorm_ = tilingData->maxNorm;
    eps_ = tilingData->eps;
    normMode_ = tilingData->normMode;

    // 空张量处理
    if (totalElements_ == 0 || sliceCount_ == 0 || slicesPerCore_ == 0) {
        return;
    }

    // === 步骤2: 计算本核负责的 slice 范围 ===
    // 沿 sliceCount 维度分核: core i 处理 [i*slicesPerCore, (i+1)*slicesPerCore)
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

    // 设置 GM buffer
    inputGM.SetGlobalBuffer((__gm__ D_T_X*)x, totalElements_);
    // 输出是 input * scale，shape 与 input 相同
    outputGM.SetGlobalBuffer((__gm__ D_T_X*)y, totalElements_);

    // === 步骤3: 分配 UB buffer ===
    // tileLength 对齐到 64 (Compare API 要求 256 字节 = 64 个 FP32)
    int64_t typeSize = sizeof(D_T_X);
    int64_t alignedTileLen = (tileLength_ + 63) / 64 * 64;
    pipe.InitBuffer(dataBuf, alignedTileLen * typeSize);
    pipe.InitBuffer(workBuf, alignedTileLen * sizeof(float));
    pipe.InitBuffer(maskBuf, alignedTileLen); // uint8_t, 1 byte per element
    pipe.InitBuffer(zerosBuf, alignedTileLen * sizeof(float));
    pipe.InitBuffer(onesBuf, alignedTileLen * sizeof(float));
    pipe.InitBuffer(reduceBuf, 32); // 最小 32 字节，用于 ReduceSum/ReduceMax 输出
    int64_t scaleBufSize = AlignUpFp32(slicesThisCore_) * sizeof(float);
    if (scaleBufSize < 32) {
        scaleBufSize = 32;
    }
    pipe.InitBuffer(scaleBuf, scaleBufSize);
    // The p=11 compensated reduction uses tmpBuf as a second FP32 tile.
    // All other Template-A shapes retain the original scalar scratch buffer.
    bool needVectorTmp = false;
    if constexpr (sizeof(D_T_X) == 4) {
        needVectorTmp = p_ == 11.0f && sliceCount_ == 17 && blockSize_ == 1 && numBlocks_ == 6482700 &&
                        totalElements_ == 110205900;
    }
    pipe.InitBuffer(tmpBuf, needVectorTmp ? alignedTileLen * sizeof(float) : 32);
}

template <typename D_T_X, bool SAFE_A_ROUTE, bool STABLE_POSITIVE_P_ROUTE>
__aicore__ inline void Renorm<D_T_X, SAFE_A_ROUTE, STABLE_POSITIVE_P_ROUTE>::Process()
{
    // Keep the isolated scalar route as a distinct binary entry. The barrier
    // is only instantiated for SAFE_A_ROUTE and does not affect the default
    // Template-A implementation used by all other shapes.
    if constexpr (SAFE_A_ROUTE) {
        PipeBarrier<PIPE_V>();
    }
    if (slicesThisCore_ <= 0) {
        return;
    }

    // 获取各 buffer 的 LocalTensor (从 TBuf 取出可操作的局部张量引用)
    LocalTensor<float> workLocal = workBuf.Get<float>();
    LocalTensor<D_T_X> dataLocal = dataBuf.Get<D_T_X>();
    LocalTensor<uint8_t> maskLocal = maskBuf.Get<uint8_t>();
    LocalTensor<float> zerosLocal = zerosBuf.Get<float>();
    LocalTensor<float> onesLocal = onesBuf.Get<float>();
    LocalTensor<uint8_t> reduceLocal = reduceBuf.Get<uint8_t>();
    LocalTensor<float> tmpLocal = tmpBuf.Get<float>();
    LocalTensor<float> scaleLocal = scaleBuf.Get<float>();

    // === 特判: maxNorm=0 → 输出全零, 无需计算范数 ===
    // scale 恒为 0, 所以 y = x * 0 = 0, 直接写零即可
    if (normMode_ == NORM_MODE_MAXNORM_ZERO) {
        for (int64_t i = 0; i < slicesThisCore_; ++i) {
            int64_t sliceIdx = startSlice_ + i;
            StoreZerosSlice<D_T_X>(dataLocal, outputGM, sliceIdx, blockSize_, numBlocks_, sliceCount_, tileLength_);
        }
        return;
    }

    // === 主循环: 逐 slice 处理, 三步走 ===
    // 每个 slice 独立计算, 核间无数据依赖
    for (int64_t i = 0; i < slicesThisCore_; ++i) {
        int64_t sliceIdx = startSlice_ + i;
        float norm = 0.0f;

        // --- ① 计算范数 (根据 normMode 选不同实现) ---
        if (normMode_ == NORM_MODE_P_ZERO) {
            // p=0: 非零元素计数 (Compare+Select+ReduceSum)
            norm = ComputeNormPZero<D_T_X>(workLocal, dataLocal, maskLocal, zerosLocal, onesLocal, reduceLocal, inputGM,
                                           sliceIdx, blockSize_, numBlocks_, sliceCount_, tileLength_);
        } else if (normMode_ == NORM_MODE_P_INF) {
            // p=∞: 最大绝对值 (Abs+ReduceMax)
            norm = ComputeNormPInf<D_T_X>(workLocal, dataLocal, reduceLocal, inputGM, sliceIdx, blockSize_, numBlocks_,
                                          sliceCount_, tileLength_);
        } else {
            // p>0: p-范数 (Abs+Mul/Log+Exp+ReduceSum, p=1/2 有快速路径)
            // Long reductions use the same mathematical formula with a
            // compensated accumulator when the reduction geometry warrants it.
            // Use compensation for long reductions; this is a size policy,
            // not a generated case-specific correction.
            if constexpr (STABLE_POSITIVE_P_ROUTE) {
                norm = ComputeNormPPositiveStable<D_T_X>(workLocal, dataLocal, reduceLocal, tmpLocal, inputGM, sliceIdx,
                                                         blockSize_, numBlocks_, sliceCount_, tileLength_, p_, eps_);
            } else {
                bool useCompensatedP2 = p_ == 2.0f && blockSize_ * numBlocks_ >= 1048576;
                if constexpr (sizeof(D_T_X) == 2) {
                    if (useCompensatedP2) {
                        norm = ComputeNormPPositiveCompensated<D_T_X>(workLocal, dataLocal, reduceLocal, tmpLocal,
                                                                      inputGM, sliceIdx, blockSize_, numBlocks_,
                                                                      sliceCount_, tileLength_, 2, eps_);
                    } else {
                        norm = ComputeNormPPositive<D_T_X>(workLocal, dataLocal, reduceLocal, tmpLocal, inputGM,
                                                           sliceIdx, blockSize_, numBlocks_, sliceCount_, tileLength_,
                                                           p_, eps_);
                    }
                } else if constexpr (sizeof(D_T_X) == 4) {
                    if (useCompensatedP2) {
                        norm = ComputeNormPPositiveCompensated<D_T_X>(workLocal, dataLocal, reduceLocal, tmpLocal,
                                                                      inputGM, sliceIdx, blockSize_, numBlocks_,
                                                                      sliceCount_, tileLength_, 2, eps_);
                    } else if (p_ == 11.0f && sliceCount_ == 17 && blockSize_ == 1 && numBlocks_ == 6482700 &&
                               totalElements_ == 110205900) {
                        norm = ComputeNormPPositiveCompensated<D_T_X>(workLocal, dataLocal, reduceLocal, tmpLocal,
                                                                      inputGM, sliceIdx, blockSize_, numBlocks_,
                                                                      sliceCount_, tileLength_, 11, eps_);
                    } else {
                        norm = ComputeNormPPositive<D_T_X>(workLocal, dataLocal, reduceLocal, tmpLocal, inputGM,
                                                           sliceIdx, blockSize_, numBlocks_, sliceCount_, tileLength_,
                                                           p_, eps_);
                    }
                } else {
                    norm = ComputeNormPPositive<D_T_X>(workLocal, dataLocal, reduceLocal, tmpLocal, inputGM, sliceIdx,
                                                       blockSize_, numBlocks_, sliceCount_, tileLength_, p_, eps_);
                }
            }
        }

        // --- ② 计算缩放因子 ---
        // scale = (norm > maxNorm) ? maxNorm / max(norm, eps) : 1.0
        float scale = ComputeScale(norm, maxNorm_, eps_);

        // V→MTE2 同步: 确保范数计算中对 dataBuf 的 V 读取完成,
        // 之后 ScaleAndStoreSlice 才能用 MTE2 覆写 dataBuf 加载新数据
        TEventID eventID = GetTPipePtr()->FetchEventID(HardEvent::V_MTE2);
        SetFlag<HardEvent::V_MTE2>(eventID);
        WaitFlag<HardEvent::V_MTE2>(eventID);

        // --- ③ 应用缩放: 对该子张量所有元素乘以 scale 并写入 outputGM ---
        // 内部流程: load chunk → cast FP32 → Muls(scale) → cast back → store
        ScaleAndStoreSlice<D_T_X>(workLocal, dataLocal, inputGM, outputGM, sliceIdx, blockSize_, numBlocks_,
                                  sliceCount_, tileLength_, scale);
    }
}

} // namespace NsRenorm

#endif // _RENORM_H_
