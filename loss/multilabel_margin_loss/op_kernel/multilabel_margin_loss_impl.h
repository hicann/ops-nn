/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */
#ifndef __MULTILABEL_MARGIN_LOSS_H__
#define __MULTILABEL_MARGIN_LOSS_H__

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "multilabel_margin_loss_tiling_data.h"
#include "multilabel_margin_loss_tiling_key.h"
#include <type_traits>

namespace NsMultilabelMarginLoss {
using namespace AscendC;

constexpr int32_t RED_NONE = 0;
constexpr int32_t RED_MEAN = 1;
constexpr int32_t RED_SUM = 2;

template <typename T>
class KernelMultilabelMarginLoss {
private:
    TPipe pipe;

    TQue<TPosition::VECIN, 1> inputQueue;
    TQue<TPosition::VECIN, 1> targetQueue;
    TQue<TPosition::VECIN, 1> partialsInQueue;
    TQue<TPosition::VECOUT, 1> isTargetOutQueue;
    TQue<TPosition::VECOUT, 1> workspaceOutQueue;
    TQue<TPosition::VECOUT, 1> outputQueue;

    TBuf<TPosition::VECCALC> xRowBuf;
    TBuf<TPosition::VECCALC> isPosBuf;
    TBuf<TPosition::VECCALC> reduceBuf;
    TBuf<TPosition::VECCALC> workBuf;
    TBuf<TPosition::VECCALC> outCastBuf;
    TBuf<TPosition::VECCALC> rowLossBuf;   // this core's row losses (float), staged before atomic add
    TBuf<TPosition::VECCALC> gatherBuf;    // core 0: full float workspace read-back (<= N floats)
    TBuf<TPosition::VECCALC> gatherOutBuf; // core 0: cast-to-T output vector (<= N elems)

    GlobalTensor<T> inputGm;
    GlobalTensor<int32_t> targetGm;
    GlobalTensor<T> outputGm;
    GlobalTensor<int32_t> isTargetGm;
    GlobalTensor<float> workspaceGm;

    uint32_t N;
    uint32_t C;
    uint32_t basePerCore;
    uint32_t pivot;
    uint32_t usedCoreNum;
    int32_t reduction;
    uint32_t myRows;
    uint32_t myStartRow;
    uint32_t programId;

public:
    __aicore__ inline KernelMultilabelMarginLoss() {}

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR target, GM_ADDR y, GM_ADDR isTarget, GM_ADDR workspace,
                                const MultilabelMarginLossTilingData* tilingData)
    {
        this->N = tilingData->N;
        this->C = tilingData->C;
        this->basePerCore = tilingData->basePerCore;
        this->pivot = tilingData->pivot;
        this->usedCoreNum = tilingData->usedCoreNum;
        this->reduction = tilingData->reduction;
        this->programId = static_cast<uint32_t>(GetBlockIdx());

        this->myRows = basePerCore + (this->programId < pivot ? 1u : 0u);
        this->myStartRow = this->programId * basePerCore + (this->programId < pivot ? this->programId : pivot);

        InitGlobalBuffers(input, target, y, isTarget, workspace);
        InitLocalBuffers();
    }

    // Multi-core-safe output write.
    // Each y element (per-row for reduction=none, the single scalar for mean/sum) is produced by
    // exactly one core. Writing it directly with a sub-32B DataCopyPad races across cores, because
    // several cores land in the same 32B GM block and the block-granular RMW clobbers neighbours.
    // Fix (same pattern as l2_loss): accumulate into a zero-initialised FLOAT workspace via atomic
    // add (atomic RMW is race-free; add-to-zero == value), then core 0 casts to T and writes the
    // whole y tensor contiguously (single writer -> no race, and float staging avoids fp16/bf16
    // atomic which is unsupported).
    __aicore__ inline void Process()
    {
        uint32_t wsElems = (this->reduction == RED_NONE) ? this->N : 1u;

        if (this->programId == 0u && wsElems > 0u) {
            InitGlobalMemory(workspaceGm, wsElems, 0.0f);
        }
        SyncAll();

        if (this->reduction == RED_NONE) {
            if (this->myRows > 0u) {
                LocalTensor<float> lossVec = rowLossBuf.Get<float>();
                for (uint32_t r = 0; r < this->myRows; r++) {
                    lossVec.SetValue(r, ProcessRow(this->myStartRow + r));
                }
                PipeBarrier<HardEvent::S_MTE3>();
                SetAtomicAdd<float>();
                DataCopyExtParams cpWs{1, this->myRows * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
                DataCopyPad(workspaceGm[this->myStartRow], lossVec, cpWs);
                SetAtomicNone();
            }
        } else {
            float coreSum = 0.0f;
            for (uint32_t r = 0; r < this->myRows; r++) {
                coreSum += ProcessRow(this->myStartRow + r);
            }
            // Fold mean's 1/N into each core's contribution so sum-of-partials == mean; FinalizeOutput
            // then only casts+writes (sum(coreSum_c / N) == total / N).
            float coreVal = (this->reduction == RED_MEAN && this->N != 0u) ?
                                (coreSum / static_cast<float>(static_cast<int32_t>(this->N))) :
                                coreSum;
            LocalTensor<float> one = rowLossBuf.Get<float>();
            one.SetValue(0, coreVal);
            PipeBarrier<HardEvent::S_MTE3>();
            SetAtomicAdd<float>();
            DataCopyExtParams cpWs{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
            DataCopyPad(workspaceGm[0], one, cpWs);
            SetAtomicNone();
        }

        SyncAll();

        if (this->programId == 0u) {
            FinalizeOutput(wsElems);
        }
    }

private:
    // Per-row byte size: max of 32B-aligned pad length and 16-element-aligned cast length.
    template <typename U>
    __aicore__ inline uint32_t RowBytes()
    {
        uint32_t padBytes = ((C * sizeof(U) + 31u) / 32u) * 32u;
        uint32_t castBytes = ((C + 15u) / 16u) * 16u * sizeof(U);
        return (padBytes > castBytes) ? padBytes : castBytes;
    }

    __aicore__ inline void InitGlobalBuffers(GM_ADDR input, GM_ADDR target, GM_ADDR y, GM_ADDR isTarget,
                                             GM_ADDR workspace)
    {
        uint64_t outputElems = (reduction == RED_NONE) ? static_cast<uint64_t>(N) : 1ULL;
        uint64_t nc = static_cast<uint64_t>(N) * static_cast<uint64_t>(C);
        inputGm.SetGlobalBuffer((__gm__ T*)input, nc);
        targetGm.SetGlobalBuffer((__gm__ int32_t*)target, nc);
        outputGm.SetGlobalBuffer((__gm__ T*)y, outputElems);
        isTargetGm.SetGlobalBuffer((__gm__ int32_t*)isTarget, nc);
        // Float accumulation workspace: N slots for reduction=none (per-row loss), else 1 (scalar).
        uint64_t wsElems = (reduction == RED_NONE) ? static_cast<uint64_t>(N) : 1ULL;
        if (wsElems == 0ULL) {
            wsElems = 1ULL;
        }
        workspaceGm.SetGlobalBuffer((__gm__ float*)workspace, wsElems);
    }

    __aicore__ inline void InitLocalBuffers()
    {
        uint32_t inputRowBytes = RowBytes<T>();
        uint32_t intRowBytes = RowBytes<int32_t>();
        uint32_t fRowBytes = RowBytes<float>();

        uint32_t partialsBytes = ((usedCoreNum * sizeof(float) + 31u) / 32u) * 32u;
        if (partialsBytes < 32u)
            partialsBytes = 32u;
        uint32_t scalarBytes = 32u;

        pipe.InitBuffer(inputQueue, 1, inputRowBytes);
        pipe.InitBuffer(targetQueue, 1, intRowBytes);
        pipe.InitBuffer(partialsInQueue, 1, partialsBytes);
        pipe.InitBuffer(isTargetOutQueue, 1, intRowBytes);
        pipe.InitBuffer(workspaceOutQueue, 1, scalarBytes);
        pipe.InitBuffer(outputQueue, 1, scalarBytes);

        pipe.InitBuffer(xRowBuf, fRowBytes);
        pipe.InitBuffer(isPosBuf, fRowBytes);
        pipe.InitBuffer(reduceBuf, fRowBytes);
        pipe.InitBuffer(workBuf, fRowBytes);
        pipe.InitBuffer(outCastBuf, scalarBytes);

        // This core stages its (basePerCore+1) row losses before one atomic add to the workspace.
        uint32_t rowLossBytes = (((this->basePerCore + 1u + 7u) / 8u) * 8u) * sizeof(float);
        if (rowLossBytes < 32u)
            rowLossBytes = 32u;
        // Core 0 reads back up to N float slots and casts them to N T elements for the final write.
        uint32_t nFloatBytes = (((this->N + 7u) / 8u) * 8u) * sizeof(float);
        if (nFloatBytes < 32u)
            nFloatBytes = 32u;
        uint32_t nTBytes = (((this->N + 15u) / 16u) * 16u) * sizeof(T);
        if (nTBytes < 32u)
            nTBytes = 32u;
        pipe.InitBuffer(rowLossBuf, rowLossBytes);
        pipe.InitBuffer(gatherBuf, nFloatBytes);
        pipe.InitBuffer(gatherOutBuf, nTBytes);
    }

    __aicore__ inline void CastInputToFloat(LocalTensor<float>& dst, LocalTensor<T>& src, uint32_t cnt)
    {
        if constexpr (std::is_same<T, float>::value) {
            uint32_t cnt8 = ((cnt + 7u) / 8u) * 8u;
            DataCopy(dst, src, cnt8);
        } else {
            Cast(dst, src, RoundMode::CAST_NONE, cnt);
        }
    }

    __aicore__ inline void StoreScalarAsInput(LocalTensor<T>& outLocal, float value)
    {
        if constexpr (std::is_same<T, float>::value) {
            outLocal.SetValue(0, value);
        } else {
            LocalTensor<float> stage = outCastBuf.Get<float>();
            stage.SetValue(0, value);
            PipeBarrier<HardEvent::S_V>();
            if constexpr (std::is_same<T, bfloat16_t>::value) {
                Cast(outLocal, stage, RoundMode::CAST_RINT, 1);
            } else {
                Cast(outLocal, stage, RoundMode::CAST_NONE, 1);
            }
            PipeBarrier<HardEvent::V_S>();
        }
    }

    // Single-event pipe barrier (e.g. HardEvent::V_S) to order scalar/vector accesses.
    template <HardEvent evt>
    __aicore__ inline void PipeBarrier()
    {
        event_t ev = static_cast<event_t>(GetTPipePtr()->FetchEventID(evt));
        SetFlag<evt>(ev);
        WaitFlag<evt>(ev);
    }

    // Copy one row of input and target from GM into local tensors (already DeQue'd).
    __aicore__ inline void CopyInRow(uint32_t row, uint32_t cnt, LocalTensor<T>& xRowIn, LocalTensor<int32_t>& tgtIn)
    {
        LocalTensor<T> inputLocal = inputQueue.AllocTensor<T>();
        LocalTensor<int32_t> targetLocal = targetQueue.AllocTensor<int32_t>();

        uint64_t rowOff = static_cast<uint64_t>(row) * static_cast<uint64_t>(cnt);

        // Explicit cast: cnt * sizeof(...) is unsigned long; cast to uint32_t avoids the
        // brace-init narrowing (-Wc++11-narrowing) on DataCopyExtParams.blockLen.
        DataCopyExtParams cpInExt{1, static_cast<uint32_t>(cnt * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padInExt{false, 0, 0, 0};
        DataCopyPad(inputLocal, inputGm[rowOff], cpInExt, padInExt);

        DataCopyExtParams cpTgtExt{1, static_cast<uint32_t>(cnt * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> padTgtExt{false, 0, 0, 0};
        DataCopyPad(targetLocal, targetGm[rowOff], cpTgtExt, padTgtExt);

        inputQueue.EnQue(inputLocal);
        targetQueue.EnQue(targetLocal);
        xRowIn = inputQueue.DeQue<T>();
        tgtIn = targetQueue.DeQue<int32_t>();
    }

    // Build is_pos from the target row and copy out the is_target row to GM.
    __aicore__ inline void BuildMasks(uint32_t row, uint32_t cnt, LocalTensor<int32_t>& tgtIn,
                                      LocalTensor<float>& isPos)
    {
        Duplicate(isPos, 0.0f, cnt);
        LocalTensor<int32_t> isTargetLocal = isTargetOutQueue.AllocTensor<int32_t>();
        Duplicate(isTargetLocal, static_cast<int32_t>(0), cnt);

        // V->S: scalar GetValue/SetValue below depends on Duplicate to isPos.
        PipeBarrier<HardEvent::V_S>();

        // Walk target with -1 sentinel break (PyTorch MultiLabelMarginLoss semantics).
        // tt is the class index; guard against out-of-range labels before using it as offset.
        for (uint32_t t = 0; t < cnt; t++) {
            int32_t tt = tgtIn.GetValue(t);
            if (tt == -1)
                break;
            if (tt < 0 || static_cast<uint32_t>(tt) >= cnt)
                continue;
            isPos.SetValue(static_cast<uint32_t>(tt), 1.0f);
            isTargetLocal.SetValue(static_cast<uint32_t>(tt), static_cast<int32_t>(1));
        }

        // CopyOut is_target row to GM (every reduction mode).
        isTargetOutQueue.EnQue(isTargetLocal);
        LocalTensor<int32_t> isTargetDeq = isTargetOutQueue.DeQue<int32_t>();
        uint64_t rowOff = static_cast<uint64_t>(row) * static_cast<uint64_t>(cnt);
        DataCopyExtParams cpIsTgt{1, static_cast<uint32_t>(cnt * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(isTargetGm[rowOff], isTargetDeq, cpIsTgt);
        isTargetOutQueue.FreeTensor(isTargetDeq);
    }

    // Sum over each valid target label position t: sum_{k: not pos} max(0, 1 - xRow[label] + xRow[k]).
    // Scalar inner loop over k gated by is_pos (only non-target classes contribute), matching PyTorch
    // MultiLabelMarginLoss. Doing it per-element (instead of clipped*neg_mask) avoids inf*0 = NaN when
    // an input element is +/-inf at a target position, so +inf/nan inputs match torch's IEEE result.
    __aicore__ inline float AccumulateRowLoss(uint32_t cnt, LocalTensor<float>& xRow, LocalTensor<int32_t>& tgtIn,
                                              LocalTensor<float>& isPos)
    {
        // Walk target positions with -1 sentinel break; guard out-of-range labels.
        float rowLoss = 0.0f;
        for (uint32_t t = 0; t < cnt; t++) {
            int32_t tt = tgtIn.GetValue(t);
            if (tt == -1)
                break;
            if (tt < 0 || static_cast<uint32_t>(tt) >= cnt)
                continue;
            float xj = xRow.GetValue(static_cast<uint32_t>(tt));
            for (uint32_t k = 0; k < cnt; k++) {
                if (isPos.GetValue(k) != 0.0f)
                    continue; // skip target classes
                float pairLoss = (1.0f - xj) + xRow.GetValue(k);
                // Accumulate max(0, pairLoss). Reject NaN via integer bit pattern (exponent all
                // ones + non-zero mantissa) instead of float compare, which the AICore scalar unit
                // / -O3 may not treat per-IEEE. torch keeps such pairs (e.g. (1-inf)+inf) out of sum.
                uint32_t bits = *reinterpret_cast<uint32_t*>(&pairLoss);
                bool isNan = ((bits & 0x7F800000u) == 0x7F800000u) && ((bits & 0x007FFFFFu) != 0u);
                if (!isNan && pairLoss > 0.0f)
                    rowLoss += pairLoss;
            }
        }
        return rowLoss;
    }

    __aicore__ inline float ProcessRow(uint32_t row)
    {
        const uint32_t cnt = this->C;

        LocalTensor<T> xRowIn;
        LocalTensor<int32_t> tgtIn;
        CopyInRow(row, cnt, xRowIn, tgtIn);

        LocalTensor<float> xRow = xRowBuf.Get<float>();
        LocalTensor<float> isPos = isPosBuf.Get<float>();

        CastInputToFloat(xRow, xRowIn, cnt);
        BuildMasks(row, cnt, tgtIn, isPos);

        float rowLoss = AccumulateRowLoss(cnt, xRow, tgtIn, isPos);
        rowLoss = (this->C == 0u) ? 0.0f : (rowLoss / static_cast<float>(static_cast<int32_t>(this->C)));

        inputQueue.FreeTensor(xRowIn);
        targetQueue.FreeTensor(tgtIn);
        return rowLoss;
    }

    // Core 0 only: read the accumulated float workspace, apply mean division, cast to T, and write
    // the whole y tensor in one contiguous copy (single writer -> no multi-core race).
    // wsElems == N for reduction=none (per-row losses), or 1 for mean/sum (the reduced scalar).
    __aicore__ inline void FinalizeOutput(uint32_t wsElems)
    {
        if (wsElems == 0u) {
            return;
        }
        // Invalidate core 0's cached view of the workspace it zero-initialised, so the read below
        // sees the values other cores atomic-added (stride 8 floats = 32B covers any cache line).
        for (uint32_t off = 0; off < wsElems; off += 8u) {
            DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(workspaceGm[off]);
        }
        LocalTensor<float> acc = gatherBuf.Get<float>();
        DataCopyExtParams cpIn{1, wsElems * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padIn{false, 0, 0, 0};
        DataCopyPad(acc, workspaceGm[0], cpIn, padIn);
        PipeBarrier<HardEvent::MTE2_V>();

        LocalTensor<T> outVec = gatherOutBuf.Get<T>();
        if constexpr (std::is_same<T, float>::value) {
            Adds(outVec, acc, 0.0f, wsElems);
        } else if constexpr (std::is_same<T, bfloat16_t>::value) {
            Cast(outVec, acc, RoundMode::CAST_RINT, wsElems);
        } else {
            Cast(outVec, acc, RoundMode::CAST_NONE, wsElems);
        }
        PipeBarrier<HardEvent::V_MTE3>();

        DataCopyExtParams cpOut{1, wsElems * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        DataCopyPad(outputGm[0], outVec, cpOut);
    }
};

} // namespace NsMultilabelMarginLoss
#endif
