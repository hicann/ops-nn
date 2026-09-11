/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Template C: Slice-Major Cross-Core Reduction (SM-CR)
// 适用条件: blockSize × numBlocks > UB 容量 (归约量太大，单核无法完成)
// 沿归约轴 (numBlocks) 切分，多核协作归约，通过 workspace + SyncAll 聚合
// 对应 CCE: G5/G10/G11/G16 (reduce_rf + sync_0)

#ifndef _RENORM_SM_CR_H_
#define _RENORM_SM_CR_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "renorm_tiling_data.h"
#include "renorm_tiling_key.h"
#include "common/renorm_common.h"
#include "renorm_p_positive.h"
#include "renorm_p_zero.h"
#include "renorm_p_inf.h"
#include "renorm_maxnorm_zero.h"

namespace NsRenormSmCr {

using namespace AscendC;

constexpr int32_t NORM_MODE_P_POSITIVE = 0;
constexpr int32_t NORM_MODE_P_ZERO = 1;
constexpr int32_t NORM_MODE_P_INF = 2;
constexpr int32_t NORM_MODE_MAXNORM_ZERO = 3;

template <typename D_T_X>
class RenormSmCr {
public:
    __aicore__ inline RenormSmCr() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const RenormTilingData* tilingData,
                                TPipe* pipeIn);
    __aicore__ inline void Process();

private:
    TPipe* pipe = nullptr;
    TBuf<QuePosition::VECCALC> dataBuf;
    TBuf<QuePosition::VECCALC> workBuf;
    TBuf<QuePosition::VECCALC> maskBuf;
    TBuf<QuePosition::VECCALC> zerosBuf;
    TBuf<QuePosition::VECCALC> onesBuf;
    TBuf<QuePosition::VECCALC> reduceBuf;
    TBuf<QuePosition::VECCALC> scaleBuf;
    TBuf<QuePosition::VECCALC> tmpBuf;
    TBuf<QuePosition::VECCALC> partialBuf; // 跨核归约: 部分和 buffer

    GlobalTensor<D_T_X> inputGM;
    GlobalTensor<D_T_X> outputGM;
    GlobalTensor<float> workspaceGM; // 跨核归约 workspace

    int64_t totalElements_ = 0;
    int64_t sliceCount_ = 0;
    int64_t blockSize_ = 0;
    int64_t alignedBlockSize_ = 0; // blockSize 对齐到 32B (UB 对齐)
    int64_t numBlocks_ = 0;
    int64_t tileLength_ = 0;
    int64_t reduceSplitsPerCore_ = 0;
    int64_t batchBlocks_ = 0; // 每次迭代处理的 block 数
    int64_t coreNum_ = 0;
    float p_ = 0.0f;
    float maxNorm_ = 0.0f;
    float eps_ = 0.0f;
    int32_t normMode_ = 0;
    int64_t wsStride_ = 0; // workspace slot 步长 (64B 对齐, 16 FP32)
};

template <typename D_T_X>
__aicore__ inline void RenormSmCr<D_T_X>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace,
                                               const RenormTilingData* tilingData, TPipe* pipeIn)
{
    pipe = pipeIn;
    totalElements_ = tilingData->totalElements;
    sliceCount_ = tilingData->sliceCount;
    blockSize_ = tilingData->blockSize;
    numBlocks_ = tilingData->numBlocks;
    tileLength_ = tilingData->tileLength;
    reduceSplitsPerCore_ = tilingData->reduceSplitsPerCore;
    batchBlocks_ = tilingData->blockFactor;
    if (batchBlocks_ <= 0) {
        batchBlocks_ = 1;
    }
    p_ = tilingData->p;
    maxNorm_ = tilingData->maxNorm;
    eps_ = tilingData->eps;
    normMode_ = tilingData->normMode;

    if (totalElements_ == 0 || sliceCount_ == 0 || reduceSplitsPerCore_ == 0) {
        return;
    }

    inputGM.SetGlobalBuffer((__gm__ D_T_X*)x, totalElements_);
    outputGM.SetGlobalBuffer((__gm__ D_T_X*)y, totalElements_);

    // workspace: 前 16MB 为系统 workspace (SyncAll), 之后为用户 workspace
    SetSysWorkspace(workspace);
    GM_ADDR userWs = GetUserWorkspace(workspace);
    if (userWs == nullptr) {
        return;
    }
    workspaceGM.SetGlobalBuffer((__gm__ float*)userWs, tilingData->workspaceSize / sizeof(float));

    // wsStride: 64B 对齐 (16 FP32), 避免多核写同一条 cache line 竞态
    constexpr int64_t ATOMIC_ALIGN = 16;
    wsStride_ = (sliceCount_ + ATOMIC_ALIGN - 1) / ATOMIC_ALIGN * ATOMIC_ALIGN;

    // alignedBlockSize_: 对齐到 32B, 确保逐 slice 加载时每行起始地址 32B 对齐
    // FP32: 8 元素, FP16/BF16: 16 元素
    constexpr int64_t UB_ALIGN_BYTES = 32;
    int64_t typeSize = sizeof(D_T_X);
    int64_t dataAlignElements = UB_ALIGN_BYTES / typeSize;
    alignedBlockSize_ = (blockSize_ + dataAlignElements - 1) / dataAlignElements * dataAlignElements;

    int64_t alignedTileLen = (tileLength_ + 63) / 64 * 64;
    pipe->InitBuffer(dataBuf, alignedTileLen * typeSize);
    pipe->InitBuffer(workBuf, alignedTileLen * sizeof(float));
    pipe->InitBuffer(maskBuf, alignedTileLen);
    pipe->InitBuffer(zerosBuf, alignedTileLen * sizeof(float));
    pipe->InitBuffer(onesBuf, alignedTileLen * sizeof(float));
    pipe->InitBuffer(reduceBuf, 32);
    // scaleBuf: 每个 slice 一个 scale 值
    int64_t scaleBufSize = NsRenorm::AlignUpFp32(sliceCount_) * sizeof(float);
    if (scaleBufSize < 32) {
        scaleBufSize = 32;
    }
    pipe->InitBuffer(scaleBuf, scaleBufSize);
    // tmpBuf: 需要容纳 max(wsStride_, tileLength_) 个 FP32
    // Phase 1: Pattern ReduceSum 的临时空间 + dst 输出
    // Phase 2/3: workspace 读写 (wsStride_ 个 FP32)
    int64_t tmpBufSize = wsStride_ * sizeof(float);
    if (tileLength_ > wsStride_) {
        tmpBufSize = tileLength_ * sizeof(float);
    }
    if (tmpBufSize < 32) {
        tmpBufSize = 32;
    }
    pipe->InitBuffer(tmpBuf, tmpBufSize);
    pipe->InitBuffer(partialBuf, wsStride_ * sizeof(float) < 32 ? 32 : wsStride_ * sizeof(float));
}

template <typename D_T_X>
__aicore__ inline void RenormSmCr<D_T_X>::Process()
{
    if (totalElements_ == 0 || sliceCount_ == 0 || reduceSplitsPerCore_ == 0) {
        return;
    }

    int64_t blockIdx = GetBlockIdx();
    LocalTensor<float> workLocal = workBuf.Get<float>();
    LocalTensor<D_T_X> dataLocal = dataBuf.Get<D_T_X>();
    LocalTensor<uint8_t> maskLocal = maskBuf.Get<uint8_t>();
    LocalTensor<float> zerosLocal = zerosBuf.Get<float>();
    LocalTensor<float> onesLocal = onesBuf.Get<float>();
    LocalTensor<uint8_t> reduceLocal = reduceBuf.Get<uint8_t>();
    LocalTensor<float> tmpLocal = tmpBuf.Get<float>();
    LocalTensor<float> scaleLocal = scaleBuf.Get<float>();
    LocalTensor<float> partialLocal = partialBuf.Get<float>();

    if (normMode_ == NORM_MODE_MAXNORM_ZERO) {
        // maxNorm=0: 直接输出全零，不需要跨核归约
        for (int64_t sliceIdx = 0; sliceIdx < sliceCount_; ++sliceIdx) {
            NsRenorm::StoreZerosSlice<D_T_X>(dataLocal, outputGM, sliceIdx, blockSize_, numBlocks_, sliceCount_,
                                             tileLength_);
        }
        return;
    }

    // === Pre-Pass 1: Core 0 清零 workspace norm slot ===
    // SetAtomicAdd 模式: 所有核原子累加到同一个 slot, 需要先清零
    // P_INF 模式用 SetAtomicMax, 初始值应为 0 (|x| >= 0, 0 是安全下界)
    if (blockIdx == 0) {
        Duplicate(partialLocal, 0.0f, static_cast<int32_t>(wsStride_));
        PipeBarrier<PIPE_V>();

        TEventID eventIDVMte3 = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
        SetFlag<HardEvent::V_MTE3>(eventIDVMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIDVMte3);

        DataCopyExtParams zeroParams;
        zeroParams.blockCount = 1;
        zeroParams.blockLen = static_cast<uint32_t>(wsStride_ * sizeof(float));
        zeroParams.srcStride = 0;
        zeroParams.dstStride = 0;
        DataCopyPad(workspaceGM[0], partialLocal, zeroParams);

        // Flush L1 DCache to ensure zeroing is visible to all clusters
        DataCacheCleanAndInvalid<float, CacheLine::ENTIRE_DATA_CACHE, DcciDst::CACHELINE_OUT>(workspaceGM);
    }

    // 确保所有核看到清零后的 workspace, 然后才开始 AtomicAdd
    SyncAll();

    // === Phase 1: 每个 core 计算自己负责的 block 范围的部分归约 ===
    // 批量优化: 每次迭代处理 batchBlocks_ 个 block, UB 布局 [sliceCount, batchBlocks*alignedBlockSize]
    // 一次 MTE2→V 同步 + Cast + |x|^p + Pattern ReduceSum, 大幅减少循环开销
    int64_t blockStart = blockIdx * reduceSplitsPerCore_;
    int64_t blockEnd = blockStart + reduceSplitsPerCore_;
    if (blockEnd > numBlocks_) {
        blockEnd = numBlocks_;
    }

    // 初始化 partialLocal 为 0
    Duplicate(partialLocal, 0.0f, static_cast<int32_t>(wsStride_));
    PipeBarrier<PIPE_V>();

    int64_t blockLen = sliceCount_ * blockSize_; // 一个 block 的总元素数 (GM 连续)
    LocalTensor<uint8_t> patternTmpLocal = tmpBuf.Get<uint8_t>();

    for (int64_t b = blockStart; b < blockEnd; b += batchBlocks_) {
        int64_t batchEnd = b + batchBlocks_;
        if (batchEnd > blockEnd) {
            batchEnd = blockEnd;
        }
        int64_t actualBatch = batchEnd - b;
        int64_t actualRowStride = actualBatch * alignedBlockSize_;
        int64_t totalElements = sliceCount_ * actualRowStride;

        // 批间同步: 确保上一轮 V 流水线对 dataLocal 的读取完成
        if (b > blockStart) {
            TEventID eventIDVMte2 = GetTPipePtr()->FetchEventID(HardEvent::V_MTE2);
            SetFlag<HardEvent::V_MTE2>(eventIDVMte2);
            WaitFlag<HardEvent::V_MTE2>(eventIDVMte2);
        }

        // 1. 加载 actualBatch 个 block 到 [sliceCount, actualBatch * alignedBlockSize] 布局
        uint8_t rightPad = static_cast<uint8_t>(alignedBlockSize_ - blockSize_);
        constexpr int64_t UB_ALIGN_BYTES = 32;
        bool blockSizeAligned = (blockSize_ * sizeof(D_T_X)) % UB_ALIGN_BYTES == 0;

        if (blockSizeAligned) {
            // 2D 模式: 每个 slice 一次 DataCopyPad 加载所有 actualBatch 个 block
            DataCopyExtParams batchLoadParams;
            batchLoadParams.blockCount = static_cast<uint32_t>(actualBatch);
            batchLoadParams.blockLen = static_cast<uint32_t>(blockSize_ * sizeof(D_T_X));
            batchLoadParams.srcStride = static_cast<uint32_t>((sliceCount_ - 1) * blockSize_ * sizeof(D_T_X));
            batchLoadParams.dstStride = 0;
            DataCopyPadExtParams<D_T_X> batchLoadPadParams = {true, 0, rightPad, 0};

            for (int64_t sliceIdx = 0; sliceIdx < sliceCount_; ++sliceIdx) {
                int64_t srcOffset = b * blockLen + sliceIdx * blockSize_;
                DataCopyPad(dataLocal[sliceIdx * actualRowStride], inputGM[srcOffset], batchLoadParams,
                            batchLoadPadParams);
            }
        } else {
            // 1D 模式: blockSize * typeSize 非 32B 对齐时, 2D DataCopyPad 的 GM 地址
            // 会触发 MTE 对齐错误. 改用逐 block 1D DataCopyPad (blockCount=1) 规避.
            DataCopyExtParams singleLoadParams;
            singleLoadParams.blockCount = 1;
            singleLoadParams.blockLen = static_cast<uint32_t>(blockSize_ * sizeof(D_T_X));
            singleLoadParams.srcStride = 0;
            singleLoadParams.dstStride = 0;
            DataCopyPadExtParams<D_T_X> singlePadParams = {true, 0, rightPad, 0};

            for (int64_t sliceIdx = 0; sliceIdx < sliceCount_; ++sliceIdx) {
                for (int64_t bi = 0; bi < actualBatch; ++bi) {
                    int64_t srcOffset = (b + bi) * blockLen + sliceIdx * blockSize_;
                    DataCopyPad(dataLocal[sliceIdx * actualRowStride + bi * alignedBlockSize_], inputGM[srcOffset],
                                singleLoadParams, singlePadParams);
                }
            }
        }

        // 2. MTE2→V 同步 (整个 batch 的 DataCopyPad 完成后才能 Cast)
        TEventID eventIDMte2V = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
        SetFlag<HardEvent::MTE2_V>(eventIDMte2V);
        WaitFlag<HardEvent::MTE2_V>(eventIDMte2V);

        // 3. 一次性 Cast 所有元素到 workLocal [sliceCount, actualRowStride]
        if constexpr (sizeof(D_T_X) == sizeof(float)) {
            DataCopy(workLocal, dataLocal, static_cast<int32_t>(totalElements));
        } else {
            Cast(workLocal, dataLocal, RoundMode::CAST_NONE, static_cast<int32_t>(totalElements));
        }
        PipeBarrier<PIPE_V>();

        // 4. 计算 |x|^p (padding 区域为 0, 不影响归约)
        Abs(workLocal, workLocal, static_cast<int32_t>(totalElements));
        PipeBarrier<PIPE_V>();

        if (normMode_ == NORM_MODE_P_ZERO) {
            Duplicate(zerosLocal, 0.0f, static_cast<int32_t>(totalElements));
            Duplicate(onesLocal, 1.0f, static_cast<int32_t>(totalElements));
            PipeBarrier<PIPE_V>();
            Compare(maskLocal, workLocal, zerosLocal, CMPMODE::GT, static_cast<int32_t>(totalElements));
            PipeBarrier<PIPE_V>();
            Select(workLocal, maskLocal, onesLocal, zerosLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE,
                   static_cast<int32_t>(totalElements));
            PipeBarrier<PIPE_V>();
        } else if (normMode_ == NORM_MODE_P_POSITIVE) {
            if (p_ == 2.0f) {
                Mul(workLocal, workLocal, workLocal, static_cast<int32_t>(totalElements));
                PipeBarrier<PIPE_V>();
            } else if (p_ != 1.0f) {
                Maxs(workLocal, workLocal, eps_, static_cast<int32_t>(totalElements));
                PipeBarrier<PIPE_V>();
                Log(workLocal, workLocal, static_cast<int32_t>(totalElements));
                PipeBarrier<PIPE_V>();
                Muls(workLocal, workLocal, p_, static_cast<int32_t>(totalElements));
                PipeBarrier<PIPE_V>();
                Exp(workLocal, workLocal, static_cast<int32_t>(totalElements));
                PipeBarrier<PIPE_V>();
            }
        }

        // 5. Pattern ReduceSum/Max: [sliceCount, actualRowStride] → [sliceCount]
        //    srcInnerPad=false: 整行参与归约, padding 零不影响 sum/max
        LocalTensor<float> reduceResultLocal = scaleLocal;
        uint32_t srcShape[2] = {static_cast<uint32_t>(sliceCount_), static_cast<uint32_t>(actualRowStride)};
        if (normMode_ == NORM_MODE_P_INF) {
            ReduceMax<float, Pattern::Reduce::AR, false>(reduceResultLocal, workLocal, patternTmpLocal, srcShape,
                                                         false);
        } else {
            ReduceSum<float, Pattern::Reduce::AR, false>(reduceResultLocal, workLocal, patternTmpLocal, srcShape,
                                                         false);
        }
        PipeBarrier<PIPE_V>();

        // 6. 向量化累加/取Max 到 partialLocal
        if (normMode_ == NORM_MODE_P_INF) {
            Max(partialLocal, partialLocal, reduceResultLocal, static_cast<int32_t>(wsStride_));
        } else {
            Add(partialLocal, partialLocal, reduceResultLocal, static_cast<int32_t>(wsStride_));
        }
        PipeBarrier<PIPE_V>();
    }

    // 存入 partialLocal (不做后处理, 聚合后才做 Sqrt/Pow)

    // === Phase 2: SetAtomicAdd/Max 部分归约到 workspace, 跨核同步 ===
    // 所有核将 partialLocal 原子累加/取Max 到 workspace[0] (单个 slot)
    // SetAtomicAdd/Max 直接写入 GM/L2, 硬件保证跨 cluster 可见性 (参考 lp_norm_v3)
    TEventID eventIDSV = GetTPipePtr()->FetchEventID(HardEvent::S_V);
    SetFlag<HardEvent::S_V>(eventIDSV);
    WaitFlag<HardEvent::S_V>(eventIDSV);

    TEventID eventIDVMte3 = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
    SetFlag<HardEvent::V_MTE3>(eventIDVMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIDVMte3);

    if (normMode_ == NORM_MODE_P_INF) {
        SetAtomicMax<float>();
    } else {
        SetAtomicAdd<float>();
    }
    DataCopyExtParams atomicParams;
    atomicParams.blockCount = 1;
    atomicParams.blockLen = static_cast<uint32_t>(wsStride_ * sizeof(float));
    atomicParams.srcStride = 0;
    atomicParams.dstStride = 0;
    DataCopyPad(workspaceGM[0], partialLocal, atomicParams);
    SetAtomicNone();

    // 跨核屏障: 确保所有核的 AtomicAdd 完成
    SyncAll();

    // === Phase 3: 读取聚合 norm, 计算 scale ===
    int64_t alignedLen = NsRenorm::AlignUpFp32(sliceCount_);

    // 读取聚合后的 norm (单次读取, 替代原来的 coreNum 次循环)
    LocalTensor<float> normLocal = partialLocal;
    DataCopyExtParams readParams;
    readParams.blockCount = 1;
    readParams.blockLen = static_cast<uint32_t>(wsStride_ * sizeof(float));
    readParams.srcStride = 0;
    readParams.dstStride = 0;
    DataCopyPadExtParams<float> readPadParams{false, 0, 0, 0.0f};
    DataCopyPad(normLocal, workspaceGM[0], readParams, readPadParams);

    TEventID eventIDMte2V = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
    SetFlag<HardEvent::MTE2_V>(eventIDMte2V);
    WaitFlag<HardEvent::MTE2_V>(eventIDMte2V);

    // 后处理: Sqrt/Pow (向量化)
    if (normMode_ == NORM_MODE_P_POSITIVE) {
        if (p_ == 2.0f) {
            Sqrt(normLocal, normLocal, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
        } else if (p_ != 1.0f) {
            Maxs(normLocal, normLocal, eps_, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
            Log(normLocal, normLocal, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
            Muls(normLocal, normLocal, 1.0f / p_, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
            Exp(normLocal, normLocal, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
        }
    }

    // 计算 scale (向量化): scale = maxNorm/max(norm,eps) if norm>maxNorm else 1.0
    Duplicate(onesLocal, 1.0f, static_cast<int32_t>(alignedLen));
    Duplicate(tmpLocal, maxNorm_, static_cast<int32_t>(alignedLen));
    PipeBarrier<PIPE_V>();

    Maxs(scaleLocal, normLocal, eps_, static_cast<int32_t>(alignedLen));
    PipeBarrier<PIPE_V>();
    Reciprocal(scaleLocal, scaleLocal, static_cast<int32_t>(alignedLen));
    PipeBarrier<PIPE_V>();
    Muls(scaleLocal, scaleLocal, maxNorm_, static_cast<int32_t>(alignedLen));
    PipeBarrier<PIPE_V>();

    Compare(maskLocal, normLocal, tmpLocal, CMPMODE::GT, static_cast<int32_t>(alignedLen));
    PipeBarrier<PIPE_V>();
    Select(scaleLocal, maskLocal, scaleLocal, onesLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE,
           static_cast<int32_t>(alignedLen));
    PipeBarrier<PIPE_V>();

    // V→S 同步: 确保 scaleLocal 可读
    TEventID eventIDVS2 = GetTPipePtr()->FetchEventID(HardEvent::V_S);
    SetFlag<HardEvent::V_S>(eventIDVS2);
    WaitFlag<HardEvent::V_S>(eventIDVS2);

    // === Phase 4: 批量应用 scale 并写回 GM ===
    // 批量优化: 每次迭代处理 batchBlocks_ 个 block (与 Phase 1 一致)
    // 1. 批量 strided 加载 + 批量 Cast + Mul(scaleTensor)
    // 2. 逐 slice 逐 block 写回 (batch store 受 UB 32B 对齐限制)
    LocalTensor<float> scaleTensor = zerosLocal;

    for (int64_t b = blockStart; b < blockEnd; b += batchBlocks_) {
        int64_t batchEnd = b + batchBlocks_;
        if (batchEnd > blockEnd) {
            batchEnd = blockEnd;
        }
        int64_t actualBatch = batchEnd - b;
        int64_t actualRowStride = actualBatch * alignedBlockSize_;
        int64_t totalElements = sliceCount_ * actualRowStride;
        uint8_t rightPad = static_cast<uint8_t>(alignedBlockSize_ - blockSize_);

        // V→MTE2 同步 (确保上一轮 V 读 dataLocal 完成)
        TEventID eventIDVMte2 = GetTPipePtr()->FetchEventID(HardEvent::V_MTE2);
        SetFlag<HardEvent::V_MTE2>(eventIDVMte2);
        WaitFlag<HardEvent::V_MTE2>(eventIDVMte2);

        // MTE2: 加载 [sliceCount, actualBatch * alignedBlockSize]
        constexpr int64_t UB_ALIGN_BYTES_P4 = 32;
        bool blockSizeAlignedP4 = (blockSize_ * sizeof(D_T_X)) % UB_ALIGN_BYTES_P4 == 0;

        if (blockSizeAlignedP4) {
            // 2D 模式: 每个 slice 一次 DataCopyPad 加载所有 actualBatch 个 block
            DataCopyExtParams batchLoadParams;
            batchLoadParams.blockCount = static_cast<uint32_t>(actualBatch);
            batchLoadParams.blockLen = static_cast<uint32_t>(blockSize_ * sizeof(D_T_X));
            batchLoadParams.srcStride = static_cast<uint32_t>((sliceCount_ - 1) * blockSize_ * sizeof(D_T_X));
            batchLoadParams.dstStride = 0;
            DataCopyPadExtParams<D_T_X> batchLoadPadParams = {true, 0, rightPad, 0};

            for (int64_t sliceIdx = 0; sliceIdx < sliceCount_; ++sliceIdx) {
                int64_t srcOffset = b * blockLen + sliceIdx * blockSize_;
                DataCopyPad(dataLocal[sliceIdx * actualRowStride], inputGM[srcOffset], batchLoadParams,
                            batchLoadPadParams);
            }
        } else {
            // 1D 模式: blockSize 非 32B 对齐, 逐 block 1D DataCopyPad 规避 MTE 对齐错误
            DataCopyExtParams singleLoadParams;
            singleLoadParams.blockCount = 1;
            singleLoadParams.blockLen = static_cast<uint32_t>(blockSize_ * sizeof(D_T_X));
            singleLoadParams.srcStride = 0;
            singleLoadParams.dstStride = 0;
            DataCopyPadExtParams<D_T_X> singlePadParams = {true, 0, rightPad, 0};

            for (int64_t sliceIdx = 0; sliceIdx < sliceCount_; ++sliceIdx) {
                for (int64_t bi = 0; bi < actualBatch; ++bi) {
                    int64_t srcOffset = (b + bi) * blockLen + sliceIdx * blockSize_;
                    DataCopyPad(dataLocal[sliceIdx * actualRowStride + bi * alignedBlockSize_], inputGM[srcOffset],
                                singleLoadParams, singlePadParams);
                }
            }
        }

        // V: 构建 scaleTensor [sliceCount, actualRowStride] (与 MTE2 并行)
        for (int64_t sliceIdx = 0; sliceIdx < sliceCount_; ++sliceIdx) {
            float scale = scaleLocal.GetValue(sliceIdx);
            Duplicate(scaleTensor[sliceIdx * actualRowStride], scale, static_cast<int32_t>(actualRowStride));
        }
        PipeBarrier<PIPE_V>();

        // MTE2→V 同步
        TEventID eventIDMte2V = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
        SetFlag<HardEvent::MTE2_V>(eventIDMte2V);
        WaitFlag<HardEvent::MTE2_V>(eventIDMte2V);

        // V: Cast to FP32
        if constexpr (sizeof(D_T_X) == sizeof(float)) {
            DataCopy(workLocal, dataLocal, static_cast<int32_t>(totalElements));
        } else {
            Cast(workLocal, dataLocal, RoundMode::CAST_NONE, static_cast<int32_t>(totalElements));
        }
        PipeBarrier<PIPE_V>();

        // V: Mul with scaleTensor
        Mul(workLocal, workLocal, scaleTensor, static_cast<int32_t>(totalElements));
        PipeBarrier<PIPE_V>();

        // V: Cast back to dtype
        NsRenorm::CastBackToDtype<D_T_X>(dataLocal, workLocal, totalElements);
        PipeBarrier<PIPE_V>();

        // V→MTE3 同步
        TEventID eventIDVMte3 = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
        SetFlag<HardEvent::V_MTE3>(eventIDVMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIDVMte3);

        // MTE3: 逐 slice 逐 block 写回 GM (batch store 受 UB padding 非 32B 对齐限制)
        DataCopyExtParams storeParams;
        storeParams.blockCount = 1;
        storeParams.blockLen = static_cast<uint32_t>(blockSize_ * sizeof(D_T_X));
        storeParams.srcStride = 0;
        storeParams.dstStride = 0;

        for (int64_t sliceIdx = 0; sliceIdx < sliceCount_; ++sliceIdx) {
            for (int64_t bi = 0; bi < actualBatch; ++bi) {
                int64_t gmOffset = (b + bi) * blockLen + sliceIdx * blockSize_;
                int64_t ubOffset = sliceIdx * actualRowStride + bi * alignedBlockSize_;
                DataCopyPad(outputGM[gmOffset], dataLocal[ubOffset], storeParams);
            }
        }

        // MTE3→MTE2 同步
        TEventID eventIDMte3Mte2 = GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2);
        SetFlag<HardEvent::MTE3_MTE2>(eventIDMte3Mte2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventIDMte3Mte2);
    }
}

} // namespace NsRenormSmCr

#endif // _RENORM_SM_CR_H_
