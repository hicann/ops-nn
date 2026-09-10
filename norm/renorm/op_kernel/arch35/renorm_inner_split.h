/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _RENORM_INNER_SPLIT_H_
#define _RENORM_INNER_SPLIT_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "renorm_tiling_data.h"
#include "common/renorm_common.h"

namespace NsRenormInnerSplit {

using namespace AscendC;

constexpr int32_t NORM_MODE_P_POSITIVE = 0;
constexpr int32_t NORM_MODE_P_ZERO = 1;
constexpr int32_t NORM_MODE_P_INF = 2;
constexpr int32_t NORM_MODE_MAXNORM_ZERO = 3;
constexpr int64_t FP32_ALIGN = 8;
constexpr int64_t ATOMIC_ALIGN = 16;

template <typename D_T_X, bool COPY_IF_UNSCALED = false, bool NATIVE_PINF_REDUCE = false,
          bool SHORT_CIRCUIT_OVERFLOW = false, bool COMPACT_POSITIVE = false, bool INTEGER_POWER = false>
class RenormInnerSplit {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const RenormTilingData* tilingData,
                                TPipe* pipeIn);
    __aicore__ inline void Process();

private:
    __aicore__ inline void TransformForNorm(LocalTensor<float>& workLocal, LocalTensor<uint8_t>& maskLocal,
                                            LocalTensor<float>& zerosLocal, LocalTensor<float>& onesLocal,
                                            LocalTensor<float>& tmpLocal, int64_t alignedLen);

    TPipe* pipe = nullptr;
    TBuf<QuePosition::VECCALC> dataBuf;
    TBuf<QuePosition::VECCALC> nativeReduceBuf;
    TBuf<QuePosition::VECCALC> workBuf;
    TBuf<QuePosition::VECCALC> maskBuf;
    // Reduce AR uses a FP32-sized pattern scratch tensor.  It must not share
    // the byte mask buffer used by Compare/Select, otherwise the reduction
    // helper can issue misaligned/out-of-bounds UB accesses.
    TBuf<QuePosition::VECCALC> patternBuf;
    TBuf<QuePosition::VECCALC> zerosBuf;
    TBuf<QuePosition::VECCALC> onesBuf;
    TBuf<QuePosition::VECCALC> reduceBuf;
    TBuf<QuePosition::VECCALC> tmpBuf;
    TBuf<QuePosition::VECCALC> partialBuf;
    TBuf<QuePosition::VECCALC> scaleBuf;

    GlobalTensor<D_T_X> inputGM;
    GlobalTensor<D_T_X> outputGM;
    GlobalTensor<float> workspaceGM;

    int64_t totalElements_ = 0;
    int64_t sliceCount_ = 0;
    int64_t blockSize_ = 0;
    int64_t tileLength_ = 0;
    int64_t blockStart_ = 0;
    int64_t blockEnd_ = 0;
    int64_t wsStride_ = 0;
    int64_t coreNum_ = 0;
    float p_ = 0.0f;
    float maxNorm_ = 0.0f;
    float eps_ = 0.0f;
    int32_t normMode_ = 0;
};

template <typename D_T_X, bool COPY_IF_UNSCALED, bool NATIVE_PINF_REDUCE, bool SHORT_CIRCUIT_OVERFLOW,
          bool COMPACT_POSITIVE, bool INTEGER_POWER>
__aicore__ inline void RenormInnerSplit<D_T_X, COPY_IF_UNSCALED, NATIVE_PINF_REDUCE, SHORT_CIRCUIT_OVERFLOW,
                                        COMPACT_POSITIVE, INTEGER_POWER>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR workspace,
                                                                               const RenormTilingData* tilingData,
                                                                               TPipe* pipeIn)
{
    pipe = pipeIn;
    totalElements_ = tilingData->totalElements;
    sliceCount_ = tilingData->sliceCount;
    blockSize_ = tilingData->blockSize;
    tileLength_ = tilingData->tileLength;
    p_ = tilingData->p;
    maxNorm_ = tilingData->maxNorm;
    eps_ = tilingData->eps;
    normMode_ = tilingData->normMode;
    coreNum_ = GetBlockNum();

    int64_t blockIdx = GetBlockIdx();
    int64_t blocksPerCore = tilingData->reduceSplitsPerCore;
    blockStart_ = blockIdx * blocksPerCore;
    blockEnd_ = blockStart_ + blocksPerCore;
    if (blockEnd_ > blockSize_) {
        blockEnd_ = blockSize_;
    }

    if (totalElements_ == 0 || sliceCount_ == 0 || blockSize_ == 0 || tileLength_ == 0) {
        return;
    }
    inputGM.SetGlobalBuffer((__gm__ D_T_X*)x, totalElements_);
    outputGM.SetGlobalBuffer((__gm__ D_T_X*)y, totalElements_);
    SetSysWorkspace(workspace);
    GM_ADDR userWs = GetUserWorkspace(workspace);
    if (userWs == nullptr) {
        return;
    }
    workspaceGM.SetGlobalBuffer((__gm__ float*)userWs, tilingData->workspaceSize / sizeof(float));

    wsStride_ = (sliceCount_ + ATOMIC_ALIGN - 1) / ATOMIC_ALIGN * ATOMIC_ALIGN;
    int64_t typeSize = sizeof(D_T_X);
    int64_t alignElems = 32 / typeSize;
    int64_t alignedTile = (tileLength_ + alignElems - 1) / alignElems * alignElems;
    int64_t alignedTileFp32 = (tileLength_ + FP32_ALIGN - 1) / FP32_ALIGN * FP32_ALIGN;
    if (alignedTileFp32 > alignedTile) {
        alignedTile = alignedTileFp32;
    }

    if constexpr (COMPACT_POSITIVE) {
        // Positive-p H4 only needs input, FP32 work and reduction scratch
        // during pass 1; the remaining vectors are slice-sized. Keeping the
        // pattern scratch byte-sized leaves room for a substantially larger
        // contiguous DMA tile than the general H allocation.
        int64_t smallVectorBytes = wsStride_ * sizeof(float);
        if (smallVectorBytes < 32) {
            smallVectorBytes = 32;
        }
        pipe->InitBuffer(dataBuf, alignedTile * typeSize);
        pipe->InitBuffer(workBuf, alignedTile * sizeof(float));
        pipe->InitBuffer(patternBuf, alignedTile);
        pipe->InitBuffer(maskBuf, 32);
        pipe->InitBuffer(zerosBuf, smallVectorBytes);
        pipe->InitBuffer(onesBuf, smallVectorBytes);
        pipe->InitBuffer(reduceBuf, 32);
        pipe->InitBuffer(tmpBuf, INTEGER_POWER ? alignedTile * sizeof(float) : smallVectorBytes);
        pipe->InitBuffer(partialBuf, smallVectorBytes);
        pipe->InitBuffer(scaleBuf, smallVectorBytes);
    } else {
        pipe->InitBuffer(dataBuf, alignedTile * typeSize);
        if constexpr (NATIVE_PINF_REDUCE) {
            // The native p=inf path only needs the full tile for input and the
            // optional scale pass. Keep scalar/reduction helpers at slice width so
            // the isolated key can use a much larger GM tile.
            pipe->InitBuffer(nativeReduceBuf, 32);
            pipe->InitBuffer(workBuf, alignedTile * sizeof(float));
            pipe->InitBuffer(maskBuf, wsStride_ < 32 ? 32 : wsStride_);
            // Native p=inf can batch several slices into one RA reduction.  The
            // pattern helper then needs byte scratch proportional to the tile.
            pipe->InitBuffer(patternBuf, alignedTile);
            pipe->InitBuffer(zerosBuf, wsStride_ * sizeof(float));
            pipe->InitBuffer(onesBuf, wsStride_ * sizeof(float));
        } else {
            pipe->InitBuffer(workBuf, alignedTile * sizeof(float));
            pipe->InitBuffer(maskBuf, alignedTile);
            pipe->InitBuffer(patternBuf, alignedTile * sizeof(float));
            pipe->InitBuffer(zerosBuf, alignedTile * sizeof(float));
            pipe->InitBuffer(onesBuf, alignedTile * sizeof(float));
        }
        pipe->InitBuffer(reduceBuf, 32);
        int64_t tmpElements = NATIVE_PINF_REDUCE ? wsStride_ : (alignedTile > wsStride_ ? alignedTile : wsStride_);
        pipe->InitBuffer(tmpBuf, tmpElements * sizeof(float));
        pipe->InitBuffer(partialBuf, wsStride_ * sizeof(float));
        pipe->InitBuffer(scaleBuf, wsStride_ * sizeof(float));
    }
}

template <typename D_T_X, bool COPY_IF_UNSCALED, bool NATIVE_PINF_REDUCE, bool SHORT_CIRCUIT_OVERFLOW,
          bool COMPACT_POSITIVE, bool INTEGER_POWER>
__aicore__ inline void
RenormInnerSplit<D_T_X, COPY_IF_UNSCALED, NATIVE_PINF_REDUCE, SHORT_CIRCUIT_OVERFLOW, COMPACT_POSITIVE,
                 INTEGER_POWER>::TransformForNorm(LocalTensor<float>& workLocal, LocalTensor<uint8_t>& maskLocal,
                                                  LocalTensor<float>& zerosLocal, LocalTensor<float>& onesLocal,
                                                  LocalTensor<float>& tmpLocal, int64_t alignedLen)
{
    Abs(workLocal, workLocal, static_cast<int32_t>(alignedLen));
    PipeBarrier<PIPE_V>();
    if (normMode_ == NORM_MODE_P_ZERO) {
        Compare(maskLocal, workLocal, zerosLocal, CMPMODE::GT, static_cast<int32_t>(alignedLen));
        PipeBarrier<PIPE_V>();
        Select(workLocal, maskLocal, onesLocal, zerosLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE,
               static_cast<int32_t>(alignedLen));
        PipeBarrier<PIPE_V>();
    } else if (normMode_ == NORM_MODE_P_POSITIVE) {
        if constexpr (INTEGER_POWER) {
            // The isolated 4601 route uses p=5.  Keep the original base in
            // tmpLocal and form x^5 with vector multiplies, matching the
            // CPU reference's direct FP32 power without Log/Exp drift.
            if (p_ == 5.0f) {
                DataCopy(tmpLocal, workLocal, static_cast<int32_t>(alignedLen));
                Mul(workLocal, workLocal, workLocal, static_cast<int32_t>(alignedLen));
                PipeBarrier<PIPE_V>();
                Mul(workLocal, workLocal, workLocal, static_cast<int32_t>(alignedLen));
                PipeBarrier<PIPE_V>();
                Mul(workLocal, workLocal, tmpLocal, static_cast<int32_t>(alignedLen));
                PipeBarrier<PIPE_V>();
            } else if (p_ == 2.0f) {
                Mul(workLocal, workLocal, workLocal, static_cast<int32_t>(alignedLen));
                PipeBarrier<PIPE_V>();
            } else if (p_ != 1.0f) {
                Maxs(workLocal, workLocal, eps_, static_cast<int32_t>(alignedLen));
                PipeBarrier<PIPE_V>();
                Log(workLocal, workLocal, static_cast<int32_t>(alignedLen));
                PipeBarrier<PIPE_V>();
                Muls(workLocal, workLocal, p_, static_cast<int32_t>(alignedLen));
                PipeBarrier<PIPE_V>();
                Exp(workLocal, workLocal, static_cast<int32_t>(alignedLen));
                PipeBarrier<PIPE_V>();
            }
        } else if (p_ == 2.0f) {
            Mul(workLocal, workLocal, workLocal, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
        } else if (p_ != 1.0f) {
            Maxs(workLocal, workLocal, eps_, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
            Log(workLocal, workLocal, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
            Muls(workLocal, workLocal, p_, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
            Exp(workLocal, workLocal, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
        }
    }
}

template <typename D_T_X, bool COPY_IF_UNSCALED, bool NATIVE_PINF_REDUCE, bool SHORT_CIRCUIT_OVERFLOW,
          bool COMPACT_POSITIVE, bool INTEGER_POWER>
__aicore__ inline void RenormInnerSplit<D_T_X, COPY_IF_UNSCALED, NATIVE_PINF_REDUCE, SHORT_CIRCUIT_OVERFLOW,
                                        COMPACT_POSITIVE, INTEGER_POWER>::Process()
{
    if (totalElements_ == 0 || sliceCount_ == 0 || blockSize_ == 0 || tileLength_ == 0) {
        return;
    }

    LocalTensor<D_T_X> dataLocal = dataBuf.Get<D_T_X>();
    LocalTensor<D_T_X> nativeReduceLocal;
    if constexpr (NATIVE_PINF_REDUCE) {
        nativeReduceLocal = nativeReduceBuf.Get<D_T_X>();
    }
    LocalTensor<float> workLocal = workBuf.Get<float>();
    LocalTensor<uint8_t> maskLocal = maskBuf.Get<uint8_t>();
    LocalTensor<float> zerosLocal = zerosBuf.Get<float>();
    LocalTensor<float> onesLocal = onesBuf.Get<float>();
    LocalTensor<uint8_t> patternTmpLocal = patternBuf.Get<uint8_t>();
    LocalTensor<float> reduceLocal = reduceBuf.Get<float>();
    LocalTensor<float> tmpLocal = tmpBuf.Get<float>();
    LocalTensor<float> partialLocal = partialBuf.Get<float>();
    LocalTensor<float> scaleLocal = scaleBuf.Get<float>();
    // Long inner-axis reductions use compensated accumulation.  The policy is
    // based on reduction geometry, so newly generated shapes get the same
    // numerically stable formula without a shape-specific scale factor.
    bool useCompensatedAccumulation = normMode_ == NORM_MODE_P_POSITIVE && blockSize_ >= 1048576 && p_ > 1.0f;

    if constexpr (SHORT_CIRCUIT_OVERFLOW && COMPACT_POSITIVE) {
        // T50 is selected only for the FP16 p=77 long-row shape.  If any
        // element of a slice is greater than 3.25, that element's direct
        // FP32 77th power is already above FLT_MAX, hence the reference norm
        // is +inf and its output scale is exactly zero.  Probe maxima first;
        // if the proof is incomplete, continue through the full H4 path.
        constexpr float DIRECT_P77_OVERFLOW_BOUND = 3.25f;
        constexpr int64_t TYPE_ALIGN_BYTES_PRE = 32;
        int64_t typeSizePre = sizeof(D_T_X);
        int64_t alignElemsPre = TYPE_ALIGN_BYTES_PRE / typeSizePre;
        Duplicate(partialLocal, 0.0f, static_cast<int32_t>(wsStride_));
        PipeBarrier<PIPE_V>();

        for (int64_t s = 0; s < sliceCount_; ++s) {
            float localMax = 0.0f;
            for (int64_t off = blockStart_; off < blockEnd_; off += tileLength_) {
                int64_t cur = tileLength_;
                if (cur > blockEnd_ - off) {
                    cur = blockEnd_ - off;
                }
                int64_t alignedLen = (cur + alignElemsPre - 1) / alignElemsPre * alignElemsPre;
                int64_t alignedLenFp32 = (cur + FP32_ALIGN - 1) / FP32_ALIGN * FP32_ALIGN;
                if (alignedLenFp32 > alignedLen) {
                    alignedLen = alignedLenFp32;
                }
                DataCopyExtParams copyParams;
                copyParams.blockCount = 1;
                copyParams.blockLen = static_cast<uint32_t>(cur * typeSizePre);
                copyParams.srcStride = 0;
                copyParams.dstStride = 0;
                DataCopyPadExtParams<D_T_X> padParams = {true, 0, static_cast<uint8_t>(alignedLen - cur), 0};
                DataCopyPad(dataLocal, inputGM[s * blockSize_ + off], copyParams, padParams);
                TEventID loadDone = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
                SetFlag<HardEvent::MTE2_V>(loadDone);
                WaitFlag<HardEvent::MTE2_V>(loadDone);
                Cast(workLocal, dataLocal, RoundMode::CAST_NONE, static_cast<int32_t>(alignedLen));
                PipeBarrier<PIPE_V>();
                Abs(workLocal, workLocal, static_cast<int32_t>(alignedLen));
                PipeBarrier<PIPE_V>();
                uint32_t probeShape[2] = {1, static_cast<uint32_t>(alignedLen)};
                ReduceMax<float, Pattern::Reduce::AR, false>(reduceLocal, workLocal, patternTmpLocal, probeShape,
                                                             false);
                PipeBarrier<PIPE_V>();
                TEventID reduceToScalar = GetTPipePtr()->FetchEventID(HardEvent::V_S);
                SetFlag<HardEvent::V_S>(reduceToScalar);
                WaitFlag<HardEvent::V_S>(reduceToScalar);
                float tileMax = reduceLocal.GetValue(0);
                if (tileMax > localMax) {
                    localMax = tileMax;
                }
                TEventID reusable = GetTPipePtr()->FetchEventID(HardEvent::V_MTE2);
                SetFlag<HardEvent::V_MTE2>(reusable);
                WaitFlag<HardEvent::V_MTE2>(reusable);
                if (localMax > DIRECT_P77_OVERFLOW_BOUND) {
                    break;
                }
            }
            partialLocal.SetValue(s, localMax);
        }

        // Publish one conservative maximum vector per core, then merge it on
        // every core.  The existing H workspace is per-core and cache-line
        // aligned, so this does not alter the fallback path's contract.
        TEventID scalarToVector = GetTPipePtr()->FetchEventID(HardEvent::S_V);
        SetFlag<HardEvent::S_V>(scalarToVector);
        WaitFlag<HardEvent::S_V>(scalarToVector);
        Muls(partialLocal, partialLocal, 1.0f, static_cast<int32_t>(wsStride_));
        PipeBarrier<PIPE_V>();
        DataCopyExtParams probeWsParams;
        probeWsParams.blockCount = 1;
        probeWsParams.blockLen = static_cast<uint32_t>(wsStride_ * sizeof(float));
        probeWsParams.srcStride = 0;
        probeWsParams.dstStride = 0;
        TEventID readyToStore = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
        SetFlag<HardEvent::V_MTE3>(readyToStore);
        WaitFlag<HardEvent::V_MTE3>(readyToStore);
        DataCopyPad(workspaceGM[GetBlockIdx() * wsStride_], partialLocal, probeWsParams);
        TEventID probeStored = GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2);
        SetFlag<HardEvent::MTE3_MTE2>(probeStored);
        WaitFlag<HardEvent::MTE3_MTE2>(probeStored);
        SyncAll();

        Duplicate(partialLocal, 0.0f, static_cast<int32_t>(wsStride_));
        PipeBarrier<PIPE_V>();
        DataCopyPadExtParams<float> probeReadPad{false, 0, 0, 0.0f};
        for (int64_t core = 0; core < coreNum_; ++core) {
            DataCopyPad(tmpLocal, workspaceGM[core * wsStride_], probeWsParams, probeReadPad);
            TEventID probeRead = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
            SetFlag<HardEvent::MTE2_V>(probeRead);
            WaitFlag<HardEvent::MTE2_V>(probeRead);
            Max(partialLocal, partialLocal, tmpLocal, static_cast<int32_t>(wsStride_));
            PipeBarrier<PIPE_V>();
            if (core + 1 < coreNum_) {
                TEventID tmpReusable = GetTPipePtr()->FetchEventID(HardEvent::V_MTE2);
                SetFlag<HardEvent::V_MTE2>(tmpReusable);
                WaitFlag<HardEvent::V_MTE2>(tmpReusable);
            }
        }
        TEventID probeToScalar = GetTPipePtr()->FetchEventID(HardEvent::V_S);
        SetFlag<HardEvent::V_S>(probeToScalar);
        WaitFlag<HardEvent::V_S>(probeToScalar);
        bool allSlicesOverflow = true;
        for (int64_t s = 0; s < sliceCount_; ++s) {
            if (!(partialLocal.GetValue(s) > DIRECT_P77_OVERFLOW_BOUND)) {
                allSlicesOverflow = false;
                break;
            }
        }
        if (allSlicesOverflow) {
            uint32_t ownedElements = static_cast<uint32_t>(blockEnd_ - blockStart_);
            for (int64_t s = 0; s < sliceCount_; ++s) {
                InitOutput<D_T_X>(outputGM[s * blockSize_ + blockStart_], ownedElements, static_cast<D_T_X>(0));
            }
            return;
        }
    }

    Duplicate(partialLocal, 0.0f, static_cast<int32_t>(wsStride_));
    if (useCompensatedAccumulation) {
        Duplicate(scaleLocal, 0.0f, static_cast<int32_t>(wsStride_));
    }
    int64_t constantLength = (NATIVE_PINF_REDUCE || COMPACT_POSITIVE) ? wsStride_ : tileLength_;
    Duplicate(zerosLocal, 0.0f, static_cast<int32_t>(constantLength));
    Duplicate(onesLocal, 1.0f, static_cast<int32_t>(constantLength));
    PipeBarrier<PIPE_V>();

    constexpr int64_t TYPE_ALIGN_BYTES = 32;
    int64_t typeSize = sizeof(D_T_X);
    int64_t alignElems = TYPE_ALIGN_BYTES / typeSize;
    bool usedNativeBatch = false;
    if constexpr (NATIVE_PINF_REDUCE) {
        if (normMode_ == NORM_MODE_P_INF && blockEnd_ - blockStart_ <= tileLength_) {
            constexpr int64_t NATIVE_BATCH = 8;
            int64_t cur = blockEnd_ - blockStart_;
            int64_t alignedLen = (cur + alignElems - 1) / alignElems * alignElems;
            DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = static_cast<uint32_t>(cur * typeSize);
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            DataCopyPadExtParams<D_T_X> padParams = {true, 0, static_cast<uint8_t>(alignedLen - cur), 0};
            TEventID loadDone = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
            TEventID reuseDone = GetTPipePtr()->FetchEventID(HardEvent::V_MTE2);
            for (int64_t s = 0; s < sliceCount_; s += NATIVE_BATCH) {
                int64_t actualBatch = sliceCount_ - s;
                if (actualBatch > NATIVE_BATCH) {
                    actualBatch = NATIVE_BATCH;
                }
                for (int64_t bi = 0; bi < actualBatch; ++bi) {
                    DataCopyPad(dataLocal[bi * alignedLen], inputGM[(s + bi) * blockSize_ + blockStart_], copyParams,
                                padParams);
                }
                SetFlag<HardEvent::MTE2_V>(loadDone);
                WaitFlag<HardEvent::MTE2_V>(loadDone);
                int64_t denseLen = actualBatch * alignedLen;
                Abs(dataLocal, dataLocal, static_cast<int32_t>(denseLen));
                PipeBarrier<PIPE_V>();
                uint32_t batchShape[2] = {static_cast<uint32_t>(actualBatch), static_cast<uint32_t>(alignedLen)};
                ReduceMax<D_T_X, Pattern::Reduce::RA, false>(nativeReduceLocal, dataLocal, patternTmpLocal, batchShape,
                                                             false);
                PipeBarrier<PIPE_V>();
                TEventID reduceToScalar = GetTPipePtr()->FetchEventID(HardEvent::V_S);
                SetFlag<HardEvent::V_S>(reduceToScalar);
                WaitFlag<HardEvent::V_S>(reduceToScalar);
                for (int64_t bi = 0; bi < actualBatch; ++bi) {
                    partialLocal.SetValue(s + bi, static_cast<float>(nativeReduceLocal.GetValue(bi)));
                }
                if (s + NATIVE_BATCH < sliceCount_) {
                    SetFlag<HardEvent::V_MTE2>(reuseDone);
                    WaitFlag<HardEvent::V_MTE2>(reuseDone);
                }
            }
            usedNativeBatch = true;
        }
    }
    if (!usedNativeBatch) {
        for (int64_t s = 0; s < sliceCount_; ++s) {
            for (int64_t off = blockStart_; off < blockEnd_; off += tileLength_) {
                int64_t cur = tileLength_;
                if (cur > blockEnd_ - off) {
                    cur = blockEnd_ - off;
                }
                int64_t alignedLen = (cur + alignElems - 1) / alignElems * alignElems;
                int64_t alignedLenFp32 = (cur + FP32_ALIGN - 1) / FP32_ALIGN * FP32_ALIGN;
                if (alignedLenFp32 > alignedLen) {
                    alignedLen = alignedLenFp32;
                }
                DataCopyExtParams copyParams;
                copyParams.blockCount = 1;
                copyParams.blockLen = static_cast<uint32_t>(cur * typeSize);
                copyParams.srcStride = 0;
                copyParams.dstStride = 0;
                DataCopyPadExtParams<D_T_X> padParams = {true, 0, static_cast<uint8_t>(alignedLen - cur), 0};
                DataCopyPad(dataLocal, inputGM[s * blockSize_ + off], copyParams, padParams);
                TEventID loadDone = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
                SetFlag<HardEvent::MTE2_V>(loadDone);
                WaitFlag<HardEvent::MTE2_V>(loadDone);

                uint32_t shape[2] = {1, static_cast<uint32_t>(alignedLen)};
                if constexpr (NATIVE_PINF_REDUCE) {
                    if (normMode_ == NORM_MODE_P_INF) {
                        Abs(dataLocal, dataLocal, static_cast<int32_t>(alignedLen));
                        PipeBarrier<PIPE_V>();
                        ReduceMax<D_T_X>(nativeReduceLocal, dataLocal, dataLocal, static_cast<uint32_t>(alignedLen));
                        PipeBarrier<PIPE_V>();
                        TEventID reduceToScalar = GetTPipePtr()->FetchEventID(HardEvent::V_S);
                        SetFlag<HardEvent::V_S>(reduceToScalar);
                        WaitFlag<HardEvent::V_S>(reduceToScalar);
                        float tileReduce = static_cast<float>(nativeReduceLocal.GetValue(0));
                        float partial = partialLocal.GetValue(s);
                        partialLocal.SetValue(s, partial > tileReduce ? partial : tileReduce);
                    } else {
                        if constexpr (sizeof(D_T_X) == sizeof(float)) {
                            DataCopy(workLocal, dataLocal, static_cast<int32_t>(alignedLen));
                        } else {
                            Cast(workLocal, dataLocal, RoundMode::CAST_NONE, static_cast<int32_t>(alignedLen));
                        }
                        PipeBarrier<PIPE_V>();
                        TransformForNorm(workLocal, maskLocal, zerosLocal, onesLocal, tmpLocal, alignedLen);
                        ReduceSum<float, Pattern::Reduce::AR, false>(reduceLocal, workLocal, patternTmpLocal, shape,
                                                                     false);
                        PipeBarrier<PIPE_V>();
                        float tileReduce = reduceLocal.GetValue(0);
                        float partial = partialLocal.GetValue(s);
                        if (useCompensatedAccumulation) {
                            float compensation = scaleLocal.GetValue(s);
                            float corrected = tileReduce - compensation;
                            float next = partial + corrected;
                            scaleLocal.SetValue(s, (next - partial) - corrected);
                            partialLocal.SetValue(s, next);
                        } else {
                            partialLocal.SetValue(s, partial + tileReduce);
                        }
                    }
                } else if (normMode_ == NORM_MODE_P_INF) {
                    if constexpr (sizeof(D_T_X) == sizeof(float)) {
                        DataCopy(workLocal, dataLocal, static_cast<int32_t>(alignedLen));
                    } else {
                        Cast(workLocal, dataLocal, RoundMode::CAST_NONE, static_cast<int32_t>(alignedLen));
                    }
                    PipeBarrier<PIPE_V>();
                    TransformForNorm(workLocal, maskLocal, zerosLocal, onesLocal, tmpLocal, alignedLen);
                    ReduceMax<float, Pattern::Reduce::AR, false>(reduceLocal, workLocal, patternTmpLocal, shape, false);
                    PipeBarrier<PIPE_V>();
                    // A scalar FP32 element at partialLocal[s] is not guaranteed
                    // to be 32-byte aligned for s > 0.  VEC instructions require
                    // an aligned destination even for a one-element operation;
                    // use the scalar pipe for the per-slice accumulator instead.
                    float tileReduce = reduceLocal.GetValue(0);
                    float partial = partialLocal.GetValue(s);
                    partialLocal.SetValue(s, partial > tileReduce ? partial : tileReduce);
                } else {
                    if constexpr (sizeof(D_T_X) == sizeof(float)) {
                        DataCopy(workLocal, dataLocal, static_cast<int32_t>(alignedLen));
                    } else {
                        Cast(workLocal, dataLocal, RoundMode::CAST_NONE, static_cast<int32_t>(alignedLen));
                    }
                    PipeBarrier<PIPE_V>();
                    TransformForNorm(workLocal, maskLocal, zerosLocal, onesLocal, tmpLocal, alignedLen);
                    ReduceSum<float, Pattern::Reduce::AR, false>(reduceLocal, workLocal, patternTmpLocal, shape, false);
                    PipeBarrier<PIPE_V>();
                    float tileReduce = reduceLocal.GetValue(0);
                    float partial = partialLocal.GetValue(s);
                    if (useCompensatedAccumulation) {
                        float compensation = scaleLocal.GetValue(s);
                        float corrected = tileReduce - compensation;
                        float next = partial + corrected;
                        scaleLocal.SetValue(s, (next - partial) - corrected);
                        partialLocal.SetValue(s, next);
                    } else {
                        partialLocal.SetValue(s, partial + tileReduce);
                    }
                }
                PipeBarrier<PIPE_V>();
                TEventID vToMte2 = GetTPipePtr()->FetchEventID(HardEvent::V_MTE2);
                SetFlag<HardEvent::V_MTE2>(vToMte2);
                WaitFlag<HardEvent::V_MTE2>(vToMte2);
            }
        }
    }

    TEventID readyToStore = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
    SetFlag<HardEvent::V_MTE3>(readyToStore);
    WaitFlag<HardEvent::V_MTE3>(readyToStore);
    DataCopyExtParams wsParams;
    wsParams.blockCount = 1;
    wsParams.blockLen = static_cast<uint32_t>(wsStride_ * sizeof(float));
    wsParams.srcStride = 0;
    wsParams.dstStride = 0;
    DataCopyPad(workspaceGM[GetBlockIdx() * wsStride_], partialLocal, wsParams);
    TEventID wsDone = GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2);
    SetFlag<HardEvent::MTE3_MTE2>(wsDone);
    WaitFlag<HardEvent::MTE3_MTE2>(wsDone);
    SyncAll();

    Duplicate(partialLocal, 0.0f, static_cast<int32_t>(wsStride_));
    if (useCompensatedAccumulation) {
        Duplicate(scaleLocal, 0.0f, static_cast<int32_t>(wsStride_));
    }
    PipeBarrier<PIPE_V>();
    DataCopyPadExtParams<float> readPad{false, 0, 0, 0.0f};
    for (int64_t core = 0; core < coreNum_; ++core) {
        DataCopyPad(tmpLocal, workspaceGM[core * wsStride_], wsParams, readPad);
        TEventID readDone = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
        SetFlag<HardEvent::MTE2_V>(readDone);
        WaitFlag<HardEvent::MTE2_V>(readDone);
        if (normMode_ == NORM_MODE_P_INF) {
            Max(partialLocal, partialLocal, tmpLocal, static_cast<int32_t>(wsStride_));
        } else if (useCompensatedAccumulation) {
            for (int64_t s = 0; s < sliceCount_; ++s) {
                float partial = partialLocal.GetValue(s);
                float value = tmpLocal.GetValue(s);
                float compensation = scaleLocal.GetValue(s);
                float corrected = value - compensation;
                float next = partial + corrected;
                scaleLocal.SetValue(s, (next - partial) - corrected);
                partialLocal.SetValue(s, next);
            }
        } else {
            Add(partialLocal, partialLocal, tmpLocal, static_cast<int32_t>(wsStride_));
        }
        PipeBarrier<PIPE_V>();
        if (core + 1 < coreNum_) {
            TEventID tmpReusable = GetTPipePtr()->FetchEventID(HardEvent::V_MTE2);
            SetFlag<HardEvent::V_MTE2>(tmpReusable);
            WaitFlag<HardEvent::V_MTE2>(tmpReusable);
        }
    }

    if constexpr (SHORT_CIRCUIT_OVERFLOW) {
        // The normal H path evaluates abs(x)^p in FP32.  If every slice sum
        // has already overflowed to +inf, the later root/scale pass produces
        // an exact zero scale.  Skip that vector work and the input reload;
        // each core initializes only its owned contiguous inner ranges.
        constexpr float FP32_MAX_VALUE = 3.402823466e38F;
        bool allSlicesOverflow = normMode_ == NORM_MODE_P_POSITIVE;
        TEventID partialToScalar = GetTPipePtr()->FetchEventID(HardEvent::V_S);
        SetFlag<HardEvent::V_S>(partialToScalar);
        WaitFlag<HardEvent::V_S>(partialToScalar);
        for (int64_t s = 0; s < sliceCount_ && allSlicesOverflow; ++s) {
            if (!(partialLocal.GetValue(s) > FP32_MAX_VALUE)) {
                allSlicesOverflow = false;
            }
        }
        if (allSlicesOverflow) {
            uint32_t ownedElements = static_cast<uint32_t>(blockEnd_ - blockStart_);
            for (int64_t s = 0; s < sliceCount_; ++s) {
                InitOutput<D_T_X>(outputGM[s * blockSize_ + blockStart_], ownedElements, static_cast<D_T_X>(0));
            }
            return;
        }
    }

    int64_t alignedSlice = (sliceCount_ + FP32_ALIGN - 1) / FP32_ALIGN * FP32_ALIGN;
    if (normMode_ == NORM_MODE_P_POSITIVE) {
        if (p_ == 2.0f) {
            Sqrt(partialLocal, partialLocal, static_cast<int32_t>(alignedSlice));
            PipeBarrier<PIPE_V>();
        } else if (p_ != 1.0f) {
            Maxs(partialLocal, partialLocal, eps_, static_cast<int32_t>(alignedSlice));
            PipeBarrier<PIPE_V>();
            Log(partialLocal, partialLocal, static_cast<int32_t>(alignedSlice));
            PipeBarrier<PIPE_V>();
            Muls(partialLocal, partialLocal, 1.0f / p_, static_cast<int32_t>(alignedSlice));
            PipeBarrier<PIPE_V>();
            Exp(partialLocal, partialLocal, static_cast<int32_t>(alignedSlice));
            PipeBarrier<PIPE_V>();
        }
    }
    Duplicate(onesLocal, 1.0f, static_cast<int32_t>(alignedSlice));
    Duplicate(tmpLocal, maxNorm_, static_cast<int32_t>(alignedSlice));
    PipeBarrier<PIPE_V>();
    Maxs(scaleLocal, partialLocal, eps_, static_cast<int32_t>(alignedSlice));
    PipeBarrier<PIPE_V>();
    Reciprocal(scaleLocal, scaleLocal, static_cast<int32_t>(alignedSlice));
    PipeBarrier<PIPE_V>();
    Muls(scaleLocal, scaleLocal, maxNorm_, static_cast<int32_t>(alignedSlice));
    PipeBarrier<PIPE_V>();
    Compare(maskLocal, partialLocal, tmpLocal, CMPMODE::GT, static_cast<int32_t>(alignedSlice));
    PipeBarrier<PIPE_V>();
    Select(scaleLocal, maskLocal, scaleLocal, onesLocal, SELMODE::VSEL_TENSOR_TENSOR_MODE,
           static_cast<int32_t>(alignedSlice));
    PipeBarrier<PIPE_V>();

    TEventID scaleToScalar = GetTPipePtr()->FetchEventID(HardEvent::V_S);
    SetFlag<HardEvent::V_S>(scaleToScalar);
    WaitFlag<HardEvent::V_S>(scaleToScalar);
    if constexpr (COPY_IF_UNSCALED) {
        bool copyOnly = true;
        for (int64_t s = 0; s < sliceCount_; ++s) {
            if (scaleLocal.GetValue(s) != 1.0f) {
                copyOnly = false;
                break;
            }
        }
        if (copyOnly) {
            // The reduction pass has proven that every scale is one. Avoid
            // a second FP16->FP32->FP16 vector pass and preserve input bytes.
            for (int64_t s = 0; s < sliceCount_; ++s) {
                for (int64_t off = blockStart_; off < blockEnd_; off += tileLength_) {
                    int64_t cur = tileLength_;
                    if (cur > blockEnd_ - off) {
                        cur = blockEnd_ - off;
                    }
                    DataCopyExtParams copyParams;
                    copyParams.blockCount = 1;
                    copyParams.blockLen = static_cast<uint32_t>(cur * typeSize);
                    copyParams.srcStride = 0;
                    copyParams.dstStride = 0;
                    DataCopyPadExtParams<D_T_X> copyPad = {false, 0, 0, 0};
                    DataCopyPad(dataLocal, inputGM[s * blockSize_ + off], copyParams, copyPad);
                    TEventID loadDone = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
                    SetFlag<HardEvent::MTE2_V>(loadDone);
                    WaitFlag<HardEvent::MTE2_V>(loadDone);
                    TEventID storeReady = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
                    SetFlag<HardEvent::V_MTE3>(storeReady);
                    WaitFlag<HardEvent::V_MTE3>(storeReady);
                    DataCopyPad(outputGM[s * blockSize_ + off], dataLocal, copyParams);
                    TEventID storeDone = GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2);
                    SetFlag<HardEvent::MTE3_MTE2>(storeDone);
                    WaitFlag<HardEvent::MTE3_MTE2>(storeDone);
                }
            }
            return;
        }
    }
    for (int64_t s = 0; s < sliceCount_; ++s) {
        float scaleValue = scaleLocal.GetValue(s);
        TEventID scalarDone = GetTPipePtr()->FetchEventID(HardEvent::S_V);
        SetFlag<HardEvent::S_V>(scalarDone);
        WaitFlag<HardEvent::S_V>(scalarDone);
        for (int64_t off = blockStart_; off < blockEnd_; off += tileLength_) {
            int64_t cur = tileLength_;
            if (cur > blockEnd_ - off) {
                cur = blockEnd_ - off;
            }
            int64_t alignedLen = (cur + alignElems - 1) / alignElems * alignElems;
            int64_t alignedLenFp32 = (cur + FP32_ALIGN - 1) / FP32_ALIGN * FP32_ALIGN;
            if (alignedLenFp32 > alignedLen) {
                alignedLen = alignedLenFp32;
            }
            DataCopyExtParams copyParams;
            copyParams.blockCount = 1;
            copyParams.blockLen = static_cast<uint32_t>(cur * typeSize);
            copyParams.srcStride = 0;
            copyParams.dstStride = 0;
            DataCopyPadExtParams<D_T_X> padParams = {true, 0, static_cast<uint8_t>(alignedLen - cur), 0};
            DataCopyPad(dataLocal, inputGM[s * blockSize_ + off], copyParams, padParams);
            TEventID loadDone = GetTPipePtr()->FetchEventID(HardEvent::MTE2_V);
            SetFlag<HardEvent::MTE2_V>(loadDone);
            WaitFlag<HardEvent::MTE2_V>(loadDone);
            if constexpr (sizeof(D_T_X) == sizeof(float)) {
                DataCopy(workLocal, dataLocal, static_cast<int32_t>(alignedLen));
            } else {
                Cast(workLocal, dataLocal, RoundMode::CAST_NONE, static_cast<int32_t>(alignedLen));
            }
            PipeBarrier<PIPE_V>();
            Muls(workLocal, workLocal, scaleValue, static_cast<int32_t>(alignedLen));
            PipeBarrier<PIPE_V>();
            NsRenorm::CastBackToDtype<D_T_X>(dataLocal, workLocal, alignedLen);
            PipeBarrier<PIPE_V>();
            TEventID storeReady = GetTPipePtr()->FetchEventID(HardEvent::V_MTE3);
            SetFlag<HardEvent::V_MTE3>(storeReady);
            WaitFlag<HardEvent::V_MTE3>(storeReady);
            DataCopyPad(outputGM[s * blockSize_ + off], dataLocal, copyParams);
            TEventID storeDone = GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2);
            SetFlag<HardEvent::MTE3_MTE2>(storeDone);
            WaitFlag<HardEvent::MTE3_MTE2>(storeDone);
        }
        TEventID nextScalar = GetTPipePtr()->FetchEventID(HardEvent::V_S);
        SetFlag<HardEvent::V_S>(nextScalar);
        WaitFlag<HardEvent::V_S>(nextScalar);
    }
}

} // namespace NsRenormInnerSplit

#endif // _RENORM_INNER_SPLIT_H_
