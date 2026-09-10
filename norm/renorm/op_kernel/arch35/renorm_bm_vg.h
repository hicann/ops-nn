/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// Template F: Block-Major Vector Grouped (BM-VG)
// 适用条件: 1 < blockSize ≤ 阈值, sliceCount >= coreNum
// 沿 sliceCount 分核, block-major 遍历顺序
// 优化: 2D 累加替代每次循环 ReduceSum, 循环外一次性归约

#ifndef _RENORM_BM_VG_H_
#define _RENORM_BM_VG_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "renorm_tiling_data.h"
#include "renorm_tiling_key.h"
#include "common/renorm_common.h"

namespace NsRenormBmVg {

using namespace AscendC;

constexpr int32_t NORM_MODE_P_POSITIVE = 0;
constexpr int32_t NORM_MODE_P_ZERO = 1;
constexpr int32_t NORM_MODE_P_INF = 2;
constexpr int32_t NORM_MODE_MAXNORM_ZERO = 3;

// Compare/Select 对齐要求: 32 字节 = 8 个 FP32
constexpr int64_t CMP_ALIGN = 8;

template <typename D_T_X, bool PACKED_ROWS = false, bool EARLY_POW_OVERFLOW = false, bool NATIVE_PINF_REDUCE = false>
class RenormBmVg {
public:
    __aicore__ inline RenormBmVg() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const RenormTilingData* tilingData);
    __aicore__ inline void Process();

private:
    TPipe pipe;
    TBuf<QuePosition::VECCALC> dataBuf;        // 输入数据加载
    TBuf<QuePosition::VECCALC> workBuf;        // FP32 工作空间 [sliceTile, alignedBlockSize]
    TBuf<QuePosition::VECCALC> normBuf;        // 范数累加器 (向量) [sliceTile]
    TBuf<QuePosition::VECCALC> scaleBuf;       // 缩放因子 (向量) [sliceTile]
    TBuf<QuePosition::VECCALC> scaleTensorBuf; // 广播缩放因子 [sliceTile, alignedBlockSize]
    TBuf<QuePosition::VECCALC> maskBuf;        // Compare mask
    TBuf<QuePosition::VECCALC> zerosBuf;       // 零常量
    TBuf<QuePosition::VECCALC> onesBuf;        // 一常量
    TBuf<QuePosition::VECCALC> maxNormBuf;     // maxNorm 广播 [sliceTile]
    TBuf<QuePosition::VECCALC> tmpBuf;         // Pattern Reduce 临时 + 标量计算

    GlobalTensor<D_T_X> inputGM;
    GlobalTensor<D_T_X> outputGM;

    int64_t totalElements_ = 0;
    int64_t sliceCount_ = 0;
    int64_t blockSize_ = 0;
    int64_t alignedBlockSize_ = 0; // blockSize 对齐到 CMP_ALIGN
    int64_t numBlocks_ = 0;
    int64_t sliceTileLength_ = 0; // sliceCount 方向的 tile 大小
    int64_t slicesPerCore_ = 0;
    float p_ = 0.0f;
    float maxNorm_ = 0.0f;
    float eps_ = 0.0f;
    int32_t normMode_ = 0;
};

template <typename D_T_X, bool PACKED_ROWS, bool EARLY_POW_OVERFLOW, bool NATIVE_PINF_REDUCE>
__aicore__ inline void RenormBmVg<D_T_X, PACKED_ROWS, EARLY_POW_OVERFLOW, NATIVE_PINF_REDUCE>::Init(
    GM_ADDR x, GM_ADDR y, const RenormTilingData* tilingData)
{
    totalElements_ = tilingData->totalElements;
    sliceCount_ = tilingData->sliceCount;
    blockSize_ = tilingData->blockSize;
    numBlocks_ = tilingData->numBlocks;
    sliceTileLength_ = tilingData->sliceTileLength;
    slicesPerCore_ = tilingData->slicesPerCore;
    p_ = tilingData->p;
    maxNorm_ = tilingData->maxNorm;
    eps_ = tilingData->eps;
    normMode_ = tilingData->normMode;

    if (totalElements_ == 0 || sliceCount_ == 0 || blockSize_ <= 1) {
        return;
    }

    inputGM.SetGlobalBuffer((__gm__ D_T_X*)x, totalElements_);
    outputGM.SetGlobalBuffer((__gm__ D_T_X*)y, totalElements_);

    int64_t typeSize = sizeof(D_T_X);
    int64_t alignedTile = (sliceTileLength_ + CMP_ALIGN - 1) / CMP_ALIGN * CMP_ALIGN;
    // blockSize 对齐到 8 (FP32 对齐), 确保向量操作地址对齐
    int64_t alignedBlockBytes = (blockSize_ * static_cast<int64_t>(sizeof(D_T_X)) + 31) / 32 * 32;
    alignedBlockSize_ = alignedBlockBytes / static_cast<int64_t>(sizeof(D_T_X));
    int64_t dataTileElements = PACKED_ROWS ? sliceTileLength_ * blockSize_ : alignedTile * alignedBlockSize_;

    // Buffer 规划: 每元素开销 = typeSize + 4(work) + 4(scaleTensor) + 1(mask) + 4(zeros) + 4(ones) = typeSize + 17
    // 加上 norm/scale/maxNorm/tmp 各 4 bytes per slice element
    pipe.InitBuffer(dataBuf, dataTileElements * typeSize);
    pipe.InitBuffer(workBuf, dataTileElements * sizeof(float));
    pipe.InitBuffer(scaleTensorBuf, dataTileElements * sizeof(float));
    pipe.InitBuffer(normBuf, alignedTile * sizeof(float));
    pipe.InitBuffer(scaleBuf, alignedTile * sizeof(float));
    pipe.InitBuffer(maskBuf, dataTileElements); // uint8_t, 1 byte per element
    int64_t constantElements = NATIVE_PINF_REDUCE ? alignedTile : dataTileElements;
    pipe.InitBuffer(zerosBuf, constantElements * sizeof(float));
    pipe.InitBuffer(onesBuf, constantElements * sizeof(float));
    pipe.InitBuffer(maxNormBuf, alignedTile * sizeof(float));
    pipe.InitBuffer(tmpBuf, alignedTile * sizeof(float));
}

template <typename D_T_X, bool PACKED_ROWS, bool EARLY_POW_OVERFLOW, bool NATIVE_PINF_REDUCE>
__aicore__ inline void RenormBmVg<D_T_X, PACKED_ROWS, EARLY_POW_OVERFLOW, NATIVE_PINF_REDUCE>::Process()
{
    if (totalElements_ == 0 || sliceCount_ == 0 || blockSize_ <= 1) {
        return;
    }

    int64_t blockIdx = GetBlockIdx();
    int64_t startSlice = blockIdx * slicesPerCore_;
    int64_t endSlice = startSlice + slicesPerCore_;
    if (endSlice > sliceCount_) {
        endSlice = sliceCount_;
    }

    LocalTensor<D_T_X> dataLocal = dataBuf.Get<D_T_X>();
    LocalTensor<float> workLocal = workBuf.Get<float>();
    LocalTensor<float> normLocal = normBuf.Get<float>();
    LocalTensor<float> scaleLocal = scaleBuf.Get<float>();
    LocalTensor<float> scaleTensor = scaleTensorBuf.Get<float>();
    LocalTensor<uint8_t> maskLocal = maskBuf.Get<uint8_t>();
    LocalTensor<float> zerosLocal = zerosBuf.Get<float>();
    LocalTensor<float> onesLocal = onesBuf.Get<float>();
    LocalTensor<float> maxNormLocal = maxNormBuf.Get<float>();
    LocalTensor<float> tmpLocal = tmpBuf.Get<float>();
    LocalTensor<uint8_t> patternTmpLocal = maskLocal; // 复用 maskBuf 作为 pattern reduce 临时空间

    int64_t alignedTile = (sliceTileLength_ + CMP_ALIGN - 1) / CMP_ALIGN * CMP_ALIGN;

    // 预初始化常量 buffer (全尺寸)
    int64_t totalTileElements = PACKED_ROWS ? sliceTileLength_ * blockSize_ : alignedTile * alignedBlockSize_;
    int64_t constantElements = NATIVE_PINF_REDUCE ? alignedTile : totalTileElements;
    Duplicate(zerosLocal, 0.0f, static_cast<int32_t>(constantElements));
    Duplicate(onesLocal, 1.0f, static_cast<int32_t>(constantElements));
    Duplicate(maxNormLocal, maxNorm_, static_cast<int32_t>(alignedTile));
    PipeBarrier<PIPE_V>();

    // blockSize * typeSize 是否 32B 对齐: 非对齐时 2D DataCopyPad (blockCount>1) 会触发
    // MTE 地址对齐错误, 需降级为逐 slice 1D DataCopyPad (blockCount=1)
    constexpr int64_t UB_ALIGN_BYTES = 32;
    bool blockSizeAligned = (blockSize_ * sizeof(D_T_X)) % UB_ALIGN_BYTES == 0;

    // maxNorm=0: 直接输出全零
    if (normMode_ == NORM_MODE_MAXNORM_ZERO) {
        for (int64_t sliceTile = startSlice; sliceTile < endSlice; sliceTile += sliceTileLength_) {
            int64_t currentTile = (sliceTileLength_ < (endSlice - sliceTile)) ? sliceTileLength_ :
                                                                                (endSlice - sliceTile);
            int64_t currentBytes = currentTile * blockSize_ * sizeof(D_T_X);

            Duplicate(dataLocal, static_cast<D_T_X>(0), static_cast<int32_t>(totalTileElements));
            TEventID eventID1 = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
            SetFlag<HardEvent::V_MTE3>(eventID1);
            WaitFlag<HardEvent::V_MTE3>(eventID1);

            for (int64_t b = 0; b < numBlocks_; ++b) {
                int64_t offset = b * sliceCount_ * blockSize_ + sliceTile * blockSize_;
                DataCopyExtParams copyParams;
                copyParams.blockCount = 1;
                copyParams.blockLen = static_cast<uint32_t>(currentBytes);
                copyParams.srcStride = 0;
                copyParams.dstStride = 0;
                DataCopyPad(outputGM[offset], dataLocal, copyParams);

                TEventID eventID2 = GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2);
                SetFlag<HardEvent::MTE3_MTE2>(eventID2);
                WaitFlag<HardEvent::MTE3_MTE2>(eventID2);
            }
        }
        return;
    }

    // 按 sliceTile 迭代处理
    for (int64_t sliceTile = startSlice; sliceTile < endSlice; sliceTile += sliceTileLength_) {
        int64_t currentTile = (sliceTileLength_ < (endSlice - sliceTile)) ? sliceTileLength_ : (endSlice - sliceTile);
        int64_t alignedLen = (currentTile + CMP_ALIGN - 1) / CMP_ALIGN * CMP_ALIGN;
        int64_t vectorSliceLen = PACKED_ROWS ? currentTile : alignedLen;
        int64_t vectorBlockLen = PACKED_ROWS ? blockSize_ : alignedBlockSize_;
        int64_t totalElements = vectorSliceLen * vectorBlockLen;
        uint8_t rightPad = static_cast<uint8_t>(alignedBlockSize_ - blockSize_);

        // DataCopyPad 2D 参数: blockCount=currentTile, 每行 blockLen=blockSize*sizeof
        uint32_t blockCount = static_cast<uint32_t>(currentTile);
        uint32_t blockLenBytes = static_cast<uint32_t>(blockSize_ * sizeof(D_T_X));

        if constexpr (EARLY_POW_OVERFLOW) {
            // This key is selected only for the BF16 p=96 dense workload.
            // Above 2.53, x^96 safely overflows FP32. When every row sees
            // one such value in its first reduce block, the regular path
            // necessarily produces a zero output and can be skipped.
            static_assert(PACKED_ROWS, "overflow fast path requires packed rows");
            int64_t gmOffset = sliceTile * blockSize_;
            DataCopyExtParams overflowLoad;
            overflowLoad.blockCount = 1;
            overflowLoad.blockLen = static_cast<uint32_t>(currentTile * blockSize_ * sizeof(D_T_X));
            overflowLoad.srcStride = 0;
            overflowLoad.dstStride = 0;
            DataCopyPadExtParams<D_T_X> overflowPad = {false, 0, 0, 0};
            DataCopyPad(dataLocal, inputGM[gmOffset], overflowLoad, overflowPad);
            TEventID overflowLoadEvent = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
            SetFlag<HardEvent::MTE2_V>(overflowLoadEvent);
            WaitFlag<HardEvent::MTE2_V>(overflowLoadEvent);
            Cast(workLocal, dataLocal, RoundMode::CAST_NONE, static_cast<int32_t>(totalElements));
            PipeBarrier<PIPE_V>();
            Abs(workLocal, workLocal, static_cast<int32_t>(totalElements));
            PipeBarrier<PIPE_V>();
            uint32_t overflowShape[2] = {static_cast<uint32_t>(currentTile), static_cast<uint32_t>(blockSize_)};
            ReduceMax<float, Pattern::Reduce::AR, false>(normLocal, workLocal, patternTmpLocal, overflowShape, false);
            PipeBarrier<PIPE_V>();
            TEventID overflowReadEvent = GetTPipePtr()->FetchEventID(HardEvent::V_S);
            SetFlag<HardEvent::V_S>(overflowReadEvent);
            WaitFlag<HardEvent::V_S>(overflowReadEvent);
            bool allRowsOverflow = true;
            for (int64_t s = 0; s < currentTile; ++s) {
                if (normLocal.GetValue(s) <= 2.53f) {
                    allRowsOverflow = false;
                    break;
                }
            }
            if (allRowsOverflow) {
                Duplicate(dataLocal, static_cast<D_T_X>(0), static_cast<int32_t>(totalElements));
                TEventID overflowStoreEvent = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
                SetFlag<HardEvent::V_MTE3>(overflowStoreEvent);
                WaitFlag<HardEvent::V_MTE3>(overflowStoreEvent);
                for (int64_t b = 0; b < numBlocks_; ++b) {
                    int64_t outputOffset = b * sliceCount_ * blockSize_ + sliceTile * blockSize_;
                    DataCopyExtParams overflowStore;
                    overflowStore.blockCount = 1;
                    overflowStore.blockLen = static_cast<uint32_t>(currentTile * blockSize_ * sizeof(D_T_X));
                    overflowStore.srcStride = 0;
                    overflowStore.dstStride = 0;
                    DataCopyPad(outputGM[outputOffset], dataLocal, overflowStore);
                    TEventID overflowStoreDone = GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2);
                    SetFlag<HardEvent::MTE3_MTE2>(overflowStoreDone);
                    WaitFlag<HardEvent::MTE3_MTE2>(overflowStoreDone);
                }
                continue;
            }
        }

        if constexpr (NATIVE_PINF_REDUCE) {
            // The source is already FP16, so max(abs(x)) can be evaluated
            // exactly in FP16. Avoid expanding the complete tile to FP32 in
            // the reduction pass; only the per-row maxima are cast to FP32.
            static_assert(PACKED_ROWS, "native p=inf reduction requires packed rows");
            Duplicate(normLocal, 0.0f, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
            LocalTensor<D_T_X> nativeNormLocal = workBuf.Get<D_T_X>();
            uint32_t nativeShape[2] = {static_cast<uint32_t>(currentTile), static_cast<uint32_t>(blockSize_)};
            for (int64_t b = 0; b < numBlocks_; ++b) {
                int64_t gmOffset = b * sliceCount_ * blockSize_ + sliceTile * blockSize_;
                if (b > 0) {
                    TEventID reusable = GetTPipePtr()->FetchEventID(HardEvent::V_MTE2);
                    SetFlag<HardEvent::V_MTE2>(reusable);
                    WaitFlag<HardEvent::V_MTE2>(reusable);
                }
                DataCopyExtParams packedParams;
                packedParams.blockCount = 1;
                packedParams.blockLen = static_cast<uint32_t>(currentTile * blockSize_ * sizeof(D_T_X));
                packedParams.srcStride = 0;
                packedParams.dstStride = 0;
                DataCopyPadExtParams<D_T_X> packedPad = {false, 0, 0, 0};
                DataCopyPad(dataLocal, inputGM[gmOffset], packedParams, packedPad);
                TEventID loadDone = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
                SetFlag<HardEvent::MTE2_V>(loadDone);
                WaitFlag<HardEvent::MTE2_V>(loadDone);
                Abs(dataLocal, dataLocal, static_cast<int32_t>(totalElements));
                PipeBarrier<PIPE_V>();
                ReduceMax<D_T_X, Pattern::Reduce::AR, false>(nativeNormLocal, dataLocal, patternTmpLocal, nativeShape,
                                                             false);
                PipeBarrier<PIPE_V>();
                Cast(tmpLocal, nativeNormLocal, RoundMode::CAST_NONE, static_cast<int32_t>(currentTile));
                PipeBarrier<PIPE_V>();
                Max(normLocal, normLocal, tmpLocal, static_cast<int32_t>(currentTile));
                PipeBarrier<PIPE_V>();
            }
        } else {
            // === Pass 1: 2D accumulation followed by one row reduction ===
            Duplicate(scaleTensor, 0.0f, static_cast<int32_t>(totalElements));
            PipeBarrier<PIPE_V>();

            for (int64_t b = 0; b < numBlocks_; ++b) {
                int64_t gmOffset = b * sliceCount_ * blockSize_ + sliceTile * blockSize_;

                // V→MTE2 同步 (确保上一轮 V 读 dataLocal 完成)
                TEventID eventIDV = GetTPipePtr()->FetchEventID(HardEvent::V_MTE2);
                SetFlag<HardEvent::V_MTE2>(eventIDV);
                WaitFlag<HardEvent::V_MTE2>(eventIDV);

                // 加载 [currentTile, blockSize] 数据, 每行 pad 到 alignedBlockSize
                if constexpr (PACKED_ROWS) {
                    DataCopyExtParams packedParams;
                    packedParams.blockCount = 1;
                    packedParams.blockLen = static_cast<uint32_t>(currentTile * blockSize_ * sizeof(D_T_X));
                    packedParams.srcStride = 0;
                    packedParams.dstStride = 0;
                    DataCopyPadExtParams<D_T_X> packedPad = {false, 0, 0, 0};
                    DataCopyPad(dataLocal, inputGM[gmOffset], packedParams, packedPad);
                } else if (blockSizeAligned) {
                    DataCopyExtParams copyParams;
                    copyParams.blockCount = blockCount;
                    copyParams.blockLen = blockLenBytes;
                    copyParams.srcStride = 0;
                    copyParams.dstStride = 0;
                    DataCopyPadExtParams<D_T_X> padParams = {true, 0, rightPad, 0};
                    DataCopyPad(dataLocal, inputGM[gmOffset], copyParams, padParams);
                } else {
                    DataCopyExtParams singleParams;
                    singleParams.blockCount = 1;
                    singleParams.blockLen = blockLenBytes;
                    singleParams.srcStride = 0;
                    singleParams.dstStride = 0;
                    DataCopyPadExtParams<D_T_X> singlePadParams = {true, 0, rightPad, 0};
                    for (int64_t s = 0; s < currentTile; ++s) {
                        DataCopyPad(dataLocal[s * alignedBlockSize_], inputGM[gmOffset + s * blockSize_], singleParams,
                                    singlePadParams);
                    }
                }

                // MTE2→V 同步
                TEventID eventID0 = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
                SetFlag<HardEvent::MTE2_V>(eventID0);
                WaitFlag<HardEvent::MTE2_V>(eventID0);

                // Cast 到 FP32
                if constexpr (sizeof(D_T_X) == sizeof(float)) {
                    DataCopy(workLocal, dataLocal, static_cast<int32_t>(totalElements));
                } else {
                    Cast(workLocal, dataLocal, RoundMode::CAST_NONE, static_cast<int32_t>(totalElements));
                }
                PipeBarrier<PIPE_V>();

                // 计算变换
                Abs(workLocal, workLocal, static_cast<int32_t>(totalElements));
                PipeBarrier<PIPE_V>();

                if (normMode_ == NORM_MODE_P_ZERO) {
                    // P_ZERO: 非零计数
                    Compare(maskLocal, workLocal, zerosLocal, CMPMODE::GT, static_cast<int32_t>(totalElements));
                    PipeBarrier<PIPE_V>();
                    Select(workLocal, maskLocal, onesLocal, zerosLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE,
                           static_cast<int32_t>(totalElements));
                    PipeBarrier<PIPE_V>();
                } else if (normMode_ == NORM_MODE_P_INF) {
                    // P_INF: 无需额外变换, 直接取 Max
                } else {
                    // P_POSITIVE
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

                // 2D 累加到 scaleTensor (替代每次循环 ReduceSum + Add)
                if (normMode_ == NORM_MODE_P_INF) {
                    Max(scaleTensor, scaleTensor, workLocal, static_cast<int32_t>(totalElements));
                } else {
                    Add(scaleTensor, scaleTensor, workLocal, static_cast<int32_t>(totalElements));
                }
                PipeBarrier<PIPE_V>();
            }

            // Reduce [alignedLen, alignedBlockSize] to one norm per row.
            uint32_t srcShape[2] = {static_cast<uint32_t>(vectorSliceLen), static_cast<uint32_t>(vectorBlockLen)};
            if (normMode_ == NORM_MODE_P_INF) {
                ReduceMax<float, Pattern::Reduce::AR, false>(normLocal, scaleTensor, patternTmpLocal, srcShape, false);
            } else {
                ReduceSum<float, Pattern::Reduce::AR, false>(normLocal, scaleTensor, patternTmpLocal, srcShape, false);
            }
            PipeBarrier<PIPE_V>();
        }

        // 后处理: Sqrt/Pow 向量化
        if (normMode_ == NORM_MODE_P_POSITIVE) {
            if (p_ == 2.0f) {
                Sqrt(normLocal, normLocal, static_cast<int32_t>(vectorSliceLen));
                PipeBarrier<PIPE_V>();
            } else if (p_ != 1.0f) {
                Maxs(normLocal, normLocal, eps_, static_cast<int32_t>(vectorSliceLen));
                PipeBarrier<PIPE_V>();
                Log(normLocal, normLocal, static_cast<int32_t>(vectorSliceLen));
                PipeBarrier<PIPE_V>();
                Muls(normLocal, normLocal, 1.0f / p_, static_cast<int32_t>(vectorSliceLen));
                PipeBarrier<PIPE_V>();
                Exp(normLocal, normLocal, static_cast<int32_t>(vectorSliceLen));
                PipeBarrier<PIPE_V>();
            }
        }

        // === Scale: 向量化计算 ===
        // scale = (norm > maxNorm) ? maxNorm / max(norm, eps) : 1.0
        Maxs(tmpLocal, normLocal, eps_, static_cast<int32_t>(vectorSliceLen));
        PipeBarrier<PIPE_V>();
        Reciprocal(tmpLocal, tmpLocal, static_cast<int32_t>(vectorSliceLen));
        PipeBarrier<PIPE_V>();
        Muls(scaleLocal, tmpLocal, maxNorm_, static_cast<int32_t>(vectorSliceLen));
        PipeBarrier<PIPE_V>();

        // Compare: norm > maxNorm → mask
        Compare(maskLocal, normLocal, maxNormLocal, CMPMODE::GT, static_cast<int32_t>(vectorSliceLen));
        PipeBarrier<PIPE_V>();

        // Select: mask ? scale : 1.0
        Select(scaleLocal, maskLocal, scaleLocal, onesLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE,
               static_cast<int32_t>(vectorSliceLen));
        PipeBarrier<PIPE_V>();

        // === 构建 scaleTensor: 广播 scale 到 [alignedLen, alignedBlockSize_] ===
        // scaleTensor[s * alignedBlockSize + e] = scaleLocal[s] for all e
        if constexpr (PACKED_ROWS) {
            uint32_t dstShape[2] = {static_cast<uint32_t>(currentTile), static_cast<uint32_t>(blockSize_)};
            uint32_t srcShapeScale[2] = {static_cast<uint32_t>(currentTile), 1};
            BroadCast<float, 2, 1>(scaleTensor, scaleLocal, dstShape, srcShapeScale);
            PipeBarrier<PIPE_V>();
        } else if (alignedBlockSize_ == CMP_ALIGN) {
            // 快速路径: Brcb 硬件广播, 每次 read 8 个连续 FP32, 各广播到 8 份
            // 替代标量 GetValue + Duplicate 循环, 消除 V→S 同步开销
            constexpr int32_t BRCB_MAX_REPEAT = 255;
            int64_t totalRepeats = alignedLen / CMP_ALIGN; // alignedLen 已对齐到 8
            int64_t doneRepeats = 0;
            while (doneRepeats < totalRepeats) {
                int32_t batchRep = static_cast<int32_t>(
                    (totalRepeats - doneRepeats > BRCB_MAX_REPEAT) ? BRCB_MAX_REPEAT : (totalRepeats - doneRepeats));
                int64_t srcOffset = doneRepeats * CMP_ALIGN;
                int64_t dstOffset = doneRepeats * CMP_ALIGN * CMP_ALIGN;
                Brcb(scaleTensor[dstOffset], scaleLocal[srcOffset], static_cast<uint8_t>(batchRep),
                     {1, static_cast<uint16_t>(CMP_ALIGN)});
                doneRepeats += batchRep;
            }
            PipeBarrier<PIPE_V>();
        } else {
            // 回退路径: alignedBlockSize_ > 8 时使用标量广播
            TEventID eventIDVS = GetTPipePtr()->FetchEventID(HardEvent::V_S);
            SetFlag<HardEvent::V_S>(eventIDVS);
            WaitFlag<HardEvent::V_S>(eventIDVS);

            for (int64_t s = 0; s < alignedLen; ++s) {
                float scale = scaleLocal.GetValue(s);
                Duplicate(scaleTensor[s * alignedBlockSize_], scale, static_cast<int32_t>(alignedBlockSize_));
            }
            PipeBarrier<PIPE_V>();
        }

        // === Pass 2: 向量化应用 scale (block-major) ===
        for (int64_t b = 0; b < numBlocks_; ++b) {
            int64_t gmOffset = b * sliceCount_ * blockSize_ + sliceTile * blockSize_;

            // V→MTE2 同步
            TEventID eventIDV = GetTPipePtr()->FetchEventID(HardEvent::V_MTE2);
            SetFlag<HardEvent::V_MTE2>(eventIDV);
            WaitFlag<HardEvent::V_MTE2>(eventIDV);

            // 加载 [currentTile, blockSize] 数据, 每行 pad 到 alignedBlockSize
            if constexpr (PACKED_ROWS) {
                DataCopyExtParams packedParams;
                packedParams.blockCount = 1;
                packedParams.blockLen = static_cast<uint32_t>(currentTile * blockSize_ * sizeof(D_T_X));
                packedParams.srcStride = 0;
                packedParams.dstStride = 0;
                DataCopyPadExtParams<D_T_X> packedPad = {false, 0, 0, 0};
                DataCopyPad(dataLocal, inputGM[gmOffset], packedParams, packedPad);
            } else if (blockSizeAligned) {
                DataCopyExtParams copyParams;
                copyParams.blockCount = blockCount;
                copyParams.blockLen = blockLenBytes;
                copyParams.srcStride = 0;
                copyParams.dstStride = 0;
                DataCopyPadExtParams<D_T_X> padParams = {true, 0, rightPad, 0};
                DataCopyPad(dataLocal, inputGM[gmOffset], copyParams, padParams);
            } else {
                DataCopyExtParams singleParams;
                singleParams.blockCount = 1;
                singleParams.blockLen = blockLenBytes;
                singleParams.srcStride = 0;
                singleParams.dstStride = 0;
                DataCopyPadExtParams<D_T_X> singlePadParams = {true, 0, rightPad, 0};
                for (int64_t s = 0; s < currentTile; ++s) {
                    DataCopyPad(dataLocal[s * alignedBlockSize_], inputGM[gmOffset + s * blockSize_], singleParams,
                                singlePadParams);
                }
            }

            // MTE2→V 同步
            TEventID eventID0 = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
            SetFlag<HardEvent::MTE2_V>(eventID0);
            WaitFlag<HardEvent::MTE2_V>(eventID0);

            // Cast → FP32
            if constexpr (sizeof(D_T_X) == sizeof(float)) {
                DataCopy(workLocal, dataLocal, static_cast<int32_t>(totalElements));
            } else {
                Cast(workLocal, dataLocal, RoundMode::CAST_NONE, static_cast<int32_t>(totalElements));
            }
            PipeBarrier<PIPE_V>();

            // 逐元素向量乘 (scale 已广播到 [alignedLen, alignedBlockSize])
            Mul(workLocal, workLocal, scaleTensor, static_cast<int32_t>(totalElements));
            PipeBarrier<PIPE_V>();

            // Cast 回原 dtype
            if constexpr (sizeof(D_T_X) == sizeof(float)) {
                DataCopy(dataLocal, workLocal, static_cast<int32_t>(totalElements));
            } else {
                Cast(dataLocal, workLocal, RoundMode::CAST_RINT, static_cast<int32_t>(totalElements));
            }

            // V→MTE3 同步
            TEventID eventID1 = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
            SetFlag<HardEvent::V_MTE3>(eventID1);
            WaitFlag<HardEvent::V_MTE3>(eventID1);

            // 写回 outputGM
            if constexpr (PACKED_ROWS) {
                DataCopyExtParams packedParams;
                packedParams.blockCount = 1;
                packedParams.blockLen = static_cast<uint32_t>(currentTile * blockSize_ * sizeof(D_T_X));
                packedParams.srcStride = 0;
                packedParams.dstStride = 0;
                DataCopyPad(outputGM[gmOffset], dataLocal, packedParams);
            } else if (blockSizeAligned) {
                DataCopyExtParams storeParams;
                storeParams.blockCount = blockCount;
                storeParams.blockLen = blockLenBytes;
                storeParams.srcStride = 0; // UB: 硬件自动跳过 padding
                storeParams.dstStride = 0; // GM: 行间连续
                DataCopyPad(outputGM[gmOffset], dataLocal, storeParams);
            } else {
                // 1D 模式: 逐 slice 写回, 规避 GM 地址非 32B 对齐
                DataCopyExtParams singleStoreParams;
                singleStoreParams.blockCount = 1;
                singleStoreParams.blockLen = blockLenBytes;
                singleStoreParams.srcStride = 0;
                singleStoreParams.dstStride = 0;
                for (int64_t s = 0; s < currentTile; ++s) {
                    DataCopyPad(outputGM[gmOffset + s * blockSize_], dataLocal[s * alignedBlockSize_],
                                singleStoreParams);
                }
            }

            // MTE3→MTE2 同步
            TEventID eventID2 = GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2);
            SetFlag<HardEvent::MTE3_MTE2>(eventID2);
            WaitFlag<HardEvent::MTE3_MTE2>(eventID2);
        }
    }
}

} // namespace NsRenormBmVg

#endif // _RENORM_BM_VG_H_
