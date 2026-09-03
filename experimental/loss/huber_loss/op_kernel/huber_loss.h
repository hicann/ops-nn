/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file huber_loss.h
 * \brief HuberLoss vector core
 *
 * One formula, one pipeline skeleton, parameterised by dtype and schedule
 * mode. dtype paths differ only by the cast layer; mean and sum differ only by
 * the value of TilingData::divisor. Adding a dtype or a mode must never copy
 * the formula.
 */
#ifndef HUBER_LOSS_H_
#define HUBER_LOSS_H_

#include "kernel_operator.h"
#include "huber_loss_tiling_data.h"

namespace NsHuberLoss {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2; // double buffering: CopyIn(i+1) overlaps Compute(i)
constexpr int32_t ACC_LEN = HUBER_LOSS_ACC_LEN;
// One 32B reduction slot is 8 float lanes; only lane 0 carries the partial sum.
constexpr int32_t SLOT_FLOATS = HUBER_LOSS_SLOT_BYTES / static_cast<int32_t>(sizeof(float));

template <typename T>
struct IsFp32 {
    static constexpr bool value = false;
};
template <>
struct IsFp32<float> {
    static constexpr bool value = true;
};

__aicore__ inline uint32_t AlignUp(uint32_t v, uint32_t a) { return (a == 0) ? 0 : ((v + a - 1) / a) * a; }

// The formula, up to the final multiply, which each path issues itself:
// loss = m * (|e| - 0.5*m), m = min(|e|, delta), e = input - target
// Always evaluated in fp32 regardless of storage dtype.
/* Every write lands on a tensor whose remaining readers are already done, in
 * the form the API documents as supported (dst == src0). The caller relies on
 * it: tmpAbs may be the same tensor as `in`, tmpMin the same as `tgt`. Live
 * ranges, in instruction order:
 *
 *     in, tgt   read only by 1
 *     tmpAbs    written by 1, last read by the caller's multiply
 *     tmpMin    written by 3, last read by the caller's multiply
 */
__aicore__ inline void ComputeHuberFactors(const LocalTensor<float>& in, const LocalTensor<float>& tgt,
                                           const LocalTensor<float>& tmpAbs, const LocalTensor<float>& tmpMin,
                                           float delta, int32_t len)
{
    Sub(tmpAbs, in, tgt, len);        // 1  e
    Abs(tmpAbs, tmpAbs, len);         // 2  |e|
    Mins(tmpMin, tmpAbs, delta, len); // 3  m = min(|e|, delta)
    Axpy(tmpAbs, tmpMin, -0.5f, len); // 4  |e| - 0.5m  (dst = src*scalar + dst)
}

template <typename T, uint32_t SCH_MODE>
class HuberLoss {
public:
    __aicore__ inline HuberLoss() {}

    // Shape comes entirely from TilingData; the kernel owns no shape constant.
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR target, GM_ADDR loss, GM_ADDR workspace,
                                const HuberLossTilingData& tiling)
    {
        InitCoreSlice(tiling);
        InitGlobalTensors(input, target, loss);
        InitUbBuffers(workspace, tiling);
    }

    __aicore__ inline void Process()
    {
        if constexpr (SCH_MODE == HUBER_LOSS_SCH_MODE_REDUCE) {
            acc = accBuf.Get<float>();
            Duplicate(acc, 0.0f, static_cast<int32_t>(tileDataNum));
        }

        // coreNumel == 0 is legitimate: an empty tensor, or a core past
        // usedCoreNum. tileDataNum == 0 would make the tile loop advance by
        // zero forever, and on device there is no assert and no error channel
        // -- a spinning AICore takes the whole job down with a timeout.
        const bool hasWork = (coreNumel > 0 && tileDataNum > 0);

        if constexpr (SCH_MODE == HUBER_LOSS_SCH_MODE_NONE) {
            // No barrier on this path, so leaving early is free.
            if (!hasWork) {
                return;
            }
            TileLoop();
        } else {
            // Reduce path: no return is permitted before SyncAll. A core that
            // skips the barrier leaves every other core waiting forever, so a
            // core with nothing to do still writes its (zero) slot and still
            // enters the barrier. The only early exit is after it.
            if (hasWork) {
                TileLoop();
            }
            Epilogue();
        }
    }

private:
    // usedCoreNum is a tiling-only field: this core's slice is derived from
    // GetBlockIdx() and the front/tail split, never from GetBlockNum().
    __aicore__ inline void InitCoreSlice(const HuberLossTilingData& tiling)
    {
        this->delta = tiling.delta;
        this->divisor = tiling.divisor;
        this->tileDataNum = tiling.tileDataNum;
        this->usedCoreNum = tiling.usedCoreNum;
        this->blockIdx = static_cast<uint32_t>(GetBlockIdx());

        // Remainder goes to the front cores. A core past usedCoreNum owns
        // nothing. This is not hypothetical: the vector cores come in pairs
        // per AIC, so GetBlockIdx() ranges wider than the core count tiling
        // planned for, and an unguarded index walks the front/tail formula
        // straight past the end of the tensor.
        if (blockIdx >= tiling.usedCoreNum) {
            this->coreNumel = 0;
            this->coreOffset = 0;
        } else if (blockIdx < tiling.frontCoreNum) {
            this->coreNumel = tiling.coreDataNum;
            this->coreOffset = static_cast<uint64_t>(blockIdx) * tiling.coreDataNum;
        } else {
            this->coreNumel = tiling.tailCoreDataNum;
            this->coreOffset = static_cast<uint64_t>(tiling.frontCoreNum) * tiling.coreDataNum +
                               static_cast<uint64_t>(blockIdx - tiling.frontCoreNum) * tiling.tailCoreDataNum;
        }
    }

    __aicore__ inline void InitGlobalTensors(GM_ADDR input, GM_ADDR target, GM_ADDR loss)
    {
        inGm.SetGlobalBuffer((__gm__ T*)input + coreOffset, coreNumel);
        targetGm.SetGlobalBuffer((__gm__ T*)target + coreOffset, coreNumel);
        if constexpr (SCH_MODE == HUBER_LOSS_SCH_MODE_NONE) {
            outGm.SetGlobalBuffer((__gm__ T*)loss + coreOffset, coreNumel);
        } else {
            outGm.SetGlobalBuffer((__gm__ T*)loss, 1);
        }
    }

    __aicore__ inline void InitUbBuffers(GM_ADDR workspace, const HuberLossTilingData& tiling)
    {
        // Every sub-tensor handed to a vector op must start on a 32B boundary.
        // InitBuffer aligns the buffers themselves, but GetWithOffset offsets
        // are ours to align: a tile of 300 floats is 1200B and would misalign
        // every tensor after the first.
        this->bufStride = AlignUp(tileDataNum * static_cast<uint32_t>(sizeof(float)), 32u);
        this->slotStride = bufStride + HUBER_LOSS_BANK_PAD_BYTES;

        // Aligned for the same reason bufStride is. In production this is
        // the identity -- tiling emits tiles that are multiples of ACC_LEN
        // (256), so tile * sizeof(T) is at least 512 bytes and already a
        // multiple of 32 -- so it costs no UB and does not change the tile
        // size. It matters if that granularity is ever relaxed: BUFFER_NUM is
        // 2, so the second buffer of each queue starts at +len, and an
        // unaligned len would silently misalign it while bufStride stayed
        // correct.
        const uint32_t queueBytes = AlignUp(tileDataNum * static_cast<uint32_t>(sizeof(T)), 32u);
        pipe.InitBuffer(inQueueInput, BUFFER_NUM, queueBytes);
        pipe.InitBuffer(bankPadQ, HUBER_LOSS_BANK_PAD_BYTES);
        pipe.InitBuffer(inQueueTarget, BUFFER_NUM, queueBytes);
        if constexpr (SCH_MODE == HUBER_LOSS_SCH_MODE_NONE) {
            pipe.InitBuffer(bankPadO, HUBER_LOSS_BANK_PAD_BYTES);
            pipe.InitBuffer(outQueue, BUFFER_NUM, queueBytes);
        }
        /* Two tile-wide fp32 slots serve every fp32 role.
         *
         * The chain only ever writes into tensors whose sources are already
         * dead, so the upcast inputs, the compute temporaries and the result
         * share storage. Slot 0 is `in` then tmpAbs; slot 1 is `tgt`, then
         * tmpMin, then the result. On the fp32 path the queue tensors are
         * already float, so the slots hold only the temporaries.
         *
         * The saving is what lets the tile grow, and the tile is what puts a
         * half-precision GM transfer past the 16 KB the DMA engine needs to
         * reach peak bandwidth.
         */
        // Slot 1 starts 256B past slot 0's end so Axpy/Mul's two srcs are not
        // in the same 220x bank group. A tile of 9472 floats is 37888B, which
        // is 74 * 512B -- without the pad the two slots alias exactly.
        pipe.InitBuffer(wsBuf, slotStride + bufStride);
        if constexpr (SCH_MODE == HUBER_LOSS_SCH_MODE_REDUCE) {
            InitReduceBuffers(workspace, tiling);
        }
    }

    __aicore__ inline void InitReduceBuffers(GM_ADDR workspace, const HuberLossTilingData& tiling)
    {
        // The cross-core slot region starts at the offset tiling computed.
        // Never recompute it here from usedCoreNum: the sync region layout
        // would then exist in two places, and a change on the host side
        // that the kernel did not follow reads the wrong address in
        // silence.
        slotGm.SetGlobalBuffer((__gm__ float*)(workspace + tiling.slotRegionOffset),
                               static_cast<uint32_t>(usedCoreNum) * SLOT_FLOATS);

        const uint32_t crossLen = static_cast<uint32_t>(usedCoreNum) * SLOT_FLOATS;

        // The accumulator is a whole tile wide, not a fixed 256 lanes: a
        // fixed-width accumulator forces the per-tile accumulation into
        // tile/256 short instructions whose issue overhead does not
        // amortise. One Add per tile instead. Costs 4B per element of UB,
        // and fans the accumulation out over a tile of lanes, so
        // cross-tile error falls.
        pipe.InitBuffer(bankPadA, HUBER_LOSS_BANK_PAD_BYTES);
        pipe.InitBuffer(accBuf, bufStride);
        pipe.InitBuffer(resBuf, ACC_LEN * sizeof(float));
        // One 32B block each, for narrowing the scalar result through the
        // vector unit. See Epilogue.
        pipe.InitBuffer(scalarBuf, 32);
        if constexpr (!IsFp32<T>::value) {
            pipe.InitBuffer(scalarOutBuf, 32);
        }
        if (usedCoreNum > 1) {
            pipe.InitBuffer(slotBuf, SLOT_FLOATS * sizeof(float));
            pipe.InitBuffer(crossBuf, crossLen * sizeof(float));
        }
    }

    /* Tile loop in prefetch order.
     *
     * CopyIn(i+1) before Compute(i) is what makes the second input buffer
     * earn its UB. On the none path, CopyOut is one tile behind so MTE3 of
     * i-1 overlaps Compute(i) and the next MTE2; the out queue is already
     * BUFFER_NUM=2 for that. CopyOut in the same iteration as Compute would
     * serialise MTE3 after Vector; running one tile behind hides that stall.
     */
    __aicore__ inline void TileLoop()
    {
        uint64_t done = 0;
        uint32_t valid = (coreNumel > tileDataNum) ? tileDataNum : static_cast<uint32_t>(coreNumel);
        CopyIn(done, valid);

        uint64_t outOff = 0;
        uint32_t outValid = 0;
        bool haveOut = false;

        while (done < coreNumel) {
            const uint64_t next = done + valid;
            uint32_t nextValid = 0;
            if (next < coreNumel) {
                const uint64_t left = coreNumel - next;
                nextValid = (left > tileDataNum) ? tileDataNum : static_cast<uint32_t>(left);
                CopyIn(next, nextValid);
            }
            Compute(valid);
            if constexpr (SCH_MODE == HUBER_LOSS_SCH_MODE_NONE) {
                if (haveOut) {
                    CopyOut(outOff, outValid);
                }
                outOff = done;
                outValid = valid;
                haveOut = true;
            }
            done = next;
            valid = nextValid;
        }
        if constexpr (SCH_MODE == HUBER_LOSS_SCH_MODE_NONE) {
            if (haveOut) {
                CopyOut(outOff, outValid);
            }
        }
    }

public:
private:
    // GM -> UB. DataCopyPad handles any tail length; rounding the copy length
    // up past the tensor end reads out of bounds on GM.
    // Note the default DataCopyPadExtParams does not zero-fill the padding --
    // it repeats the first element. Nothing here may rely on the pad value;
    // every vector op below is issued with the valid length.
    __aicore__ inline void CopyGmToUb(const LocalTensor<T>& dst, const GlobalTensor<T>& src, uint32_t valid)
    {
        const uint32_t bytes = valid * static_cast<uint32_t>(sizeof(T));
        if ((bytes & 31u) == 0) {
            DataCopy(dst, src, valid);
        } else {
            DataCopyExtParams copyParams{1, bytes, 0, 0, 0};
            DataCopyPadExtParams<T> padParams;
            DataCopyPad(dst, src, copyParams, padParams);
        }
    }

    __aicore__ inline void CopyIn(uint64_t offset, uint32_t valid)
    {
        LocalTensor<T> in = inQueueInput.template AllocTensor<T>();
        LocalTensor<T> tgt = inQueueTarget.template AllocTensor<T>();
        CopyGmToUb(in, inGm[offset], valid);
        CopyGmToUb(tgt, targetGm[offset], valid);
        inQueueInput.EnQue(in);
        inQueueTarget.EnQue(tgt);
    }

    __aicore__ inline void Compute(uint32_t valid)
    {
        LocalTensor<T> inLocal = inQueueInput.template DeQue<T>();
        LocalTensor<T> tgtLocal = inQueueTarget.template DeQue<T>();

        LocalTensor<float> tmpAbs = wsBuf.template GetWithOffset<float>(tileDataNum, 0);
        LocalTensor<float> tmpMin = wsBuf.template GetWithOffset<float>(tileDataNum, slotStride);

        LocalTensor<float> inF;
        LocalTensor<float> tgtF;
        if constexpr (IsFp32<T>::value) {
            inF = inLocal.template ReinterpretCast<float>();
            tgtF = tgtLocal.template ReinterpretCast<float>();
        } else {
            // The upcast lands in the two slots. Instruction 1 of the chain
            // then overwrites slot 0, which is the last read of either.
            inF = tmpAbs;
            tgtF = tmpMin;
            Cast(inF, inLocal, RoundMode::CAST_NONE, valid);
            Cast(tgtF, tgtLocal, RoundMode::CAST_NONE, valid);
        }

        ComputeHuberFactors(inF, tgtF, tmpAbs, tmpMin, delta, static_cast<int32_t>(valid));

        inQueueInput.FreeTensor(inLocal);
        inQueueTarget.FreeTensor(tgtLocal);

        if constexpr (SCH_MODE == HUBER_LOSS_SCH_MODE_REDUCE) {
            // acc += m * (|e| - 0.5m). Lanes past `valid` keep whatever they
            // accumulated from earlier tiles: the tail contributes nothing.
            MulAddDst(acc, tmpMin, tmpAbs, static_cast<int32_t>(valid));
        } else {
            LocalTensor<T> outLocal = outQueue.template AllocTensor<T>();
            LocalTensor<float> dstF;
            if constexpr (IsFp32<T>::value) {
                dstF = outLocal.template ReinterpretCast<float>();
            } else {
                // Slot 1: tmpMin is the multiply's own src0, which the API
                // allows to be the destination.
                dstF = tmpMin;
            }
            Mul(dstF, tmpMin, tmpAbs, static_cast<int32_t>(valid)); // 5  m * (|e| - 0.5m)
            if constexpr (!IsFp32<T>::value) {
                Cast(outLocal, dstF, RoundMode::CAST_RINT, valid);
            }
            // The queue's MTE3 wait is taken at EnQue time: nothing may sit
            // between the result and its EnQue, or the copy out starves the
            // tile loop.
            outQueue.EnQue(outLocal);
        }
    }

    // UB -> GM, tail-safe: writes exactly `valid` elements.
    __aicore__ inline void CopyOut(uint64_t offset, uint32_t valid)
    {
        LocalTensor<T> outLocal = outQueue.template DeQue<T>();
        const uint32_t bytes = valid * static_cast<uint32_t>(sizeof(T));
        if ((bytes & 31u) == 0) {
            DataCopy(outGm[offset], outLocal, valid);
        } else {
            DataCopyExtParams copyParams{1, bytes, 0, 0, 0};
            DataCopyPad(outGm[offset], outLocal, copyParams);
        }
        outQueue.FreeTensor(outLocal);
    }

    __aicore__ inline void Epilogue()
    {
        // The compute loop is over, so slot 0 is free and a tile wide; it
        // serves as the reduction scratch instead of a buffer of its own.
        LocalTensor<float> work = wsBuf.GetWithOffset<float>(tileDataNum, 0);
        LocalTensor<float> res = resBuf.Get<float>();
        ReduceSum(res, acc, work, static_cast<int32_t>(tileDataNum));
        // Vector result read by a scalar load: barrier first.
        PipeBarrier<PIPE_ALL>();
        float partial = res.GetValue(0);

        if (usedCoreNum > 1) {
            // Publish this core's partial sum into its slot. Only cores inside
            // the planned range write: slotGm is mapped for usedCoreNum slots
            // and core0 reads only [0, usedCoreNum), so a core past the
            // planned count (the premise guarded in Init) skips the store but
            // still enters the barrier below.
            if (blockIdx < usedCoreNum) {
                LocalTensor<float> slot = slotBuf.Get<float>();
                // Lanes 1..7 are zeroed so core0 can sum the whole region in
                // one vector pass instead of striding over lane 0 with a
                // scalar loop.
                Duplicate(slot, 0.0f, SLOT_FLOATS);
                // Vector write then scalar write to the same buffer: without a
                // barrier between them the scalar store can be overtaken by the
                // vector one, which would zero lane 0 and drop this core's
                // share from the cross-core sum.
                PipeBarrier<PIPE_ALL>();
                slot.SetValue(0, partial);
                PipeBarrier<PIPE_ALL>();
                DataCopy(slotGm[static_cast<uint64_t>(blockIdx) * SLOT_FLOATS], slot, SLOT_FLOATS);
            }

            SyncAll();

            // The one permitted early exit, and it is after the barrier.
            if (blockIdx != 0) {
                return;
            }

            // Reading [0, usedCoreNum) covers every partial sum by
            // construction: the block-index-to-data mapping is the front/tail
            // split in tiling, so the cores holding data are exactly that
            // contiguous range. Slots above it are neither written nor read,
            // which is why no pre-zeroing pass is needed -- and such a pass
            // would have cost a second barrier.
            const uint32_t crossLen = static_cast<uint32_t>(usedCoreNum) * SLOT_FLOATS;
            LocalTensor<float> all = crossBuf.Get<float>();
            DataCopy(all, slotGm, crossLen);
            PipeBarrier<PIPE_ALL>();
            ReduceSum(res, all, work, static_cast<int32_t>(crossLen));
            PipeBarrier<PIPE_ALL>();
            partial = res.GetValue(0);
        }

        // The guard above returns inside the multi-core branch, after the
        // barrier. It is repeated here for the single-core case, which never
        // enters that branch: with usedCoreNum == 1 every launched core
        // reaches the store below, and a core past the planned count (the
        // premise guarded in Init and in the slot region sizing) carries a
        // zeroed accumulator, so it would write 0 over the real result.
        if (blockIdx != 0) {
            return;
        }

        // Branchless: divisor is numel for mean and 1.0 for sum. The division
        // happens in fp32, before the single narrowing conversion -- casting
        // first would cost a second rounding. Empty tensors need no special
        // case: 0/0 is NaN for mean, 0/1 is 0 for sum.
        const float total = partial / divisor;

        // The narrowing conversion and the store both go through the vector
        // path: `outGm.SetValue(0, static_cast<T>(x))` fails to compile for
        // bfloat16 -- the device backend has no scalar bf16 conversion, only a
        // vector one.
        LocalTensor<float> scalarF = scalarBuf.template Get<float>();
        Duplicate(scalarF, total, 8);
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        if constexpr (IsFp32<T>::value) {
            DataCopyPad(outGm, scalarF.template ReinterpretCast<T>(), copyParams);
        } else {
            LocalTensor<T> scalarT = scalarOutBuf.template Get<T>();
            Cast(scalarT, scalarF, RoundMode::CAST_RINT, 8);
            DataCopyPad(outGm, scalarT, copyParams);
        }
    }

    float delta = 1.0f;
    float divisor = 1.0f;
    uint64_t coreNumel = 0;
    uint64_t coreOffset = 0;
    uint32_t tileDataNum = 0;
    uint32_t bufStride = 0;  // per-sub-tensor byte stride, 32B aligned
    uint32_t slotStride = 0; // bufStride + bank pad, start of ws slot 1
    uint32_t usedCoreNum = 1;
    uint32_t blockIdx = 0;

    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueInput, inQueueTarget;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
    TBuf<TPosition::VECCALC> wsBuf, accBuf, resBuf, scalarBuf, scalarOutBuf, slotBuf, crossBuf;
    TBuf<TPosition::VECCALC> bankPadQ, bankPadO, bankPadA;
    LocalTensor<float> acc;
    GlobalTensor<T> inGm, targetGm, outGm;
    GlobalTensor<float> slotGm; // cross-core reduction slots, fp32 throughout
};

} // namespace NsHuberLoss

#endif // HUBER_LOSS_H_
