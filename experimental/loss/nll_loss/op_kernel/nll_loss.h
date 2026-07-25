/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NLLLOSS_H
#define NLLLOSS_H

#include <type_traits>
#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "nll_loss_tiling_data.h"

namespace NsNllLoss {
using namespace AscendC;

constexpr uint32_t BLK_ELEM = 8u;
constexpr uint32_t VEC_ALIGN = 64u;
constexpr uint32_t BLK_BYTES = 32u;

__aicore__ inline uint32_t CeilAlign(uint32_t v, uint32_t a) { return a == 0u ? v : (v + a - 1u) / a * a; }

__aicore__ inline uint16_t FloatToHalfBits(float f)
{
    union {
        float f;
        uint32_t u;
    } c;
    c.f = f;
    uint32_t bits = c.u;
    uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = (bits >> 13) & 0x3ffu;
    if (exp <= 0) {
        return static_cast<uint16_t>(sign);
    }
    if (exp >= 31) {
        return static_cast<uint16_t>(sign | 0x7c00u);
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | mant);
}

__aicore__ inline uint16_t FloatToBf16Bits(float f)
{
    union {
        float f;
        uint32_t u;
    } c;
    c.f = f;
    uint32_t bits = c.u;
    uint32_t lsb = (bits >> 16) & 1u;
    bits += 0x7fffu + lsb;
    return static_cast<uint16_t>(bits >> 16);
}

__aicore__ inline float Bf16BitsToFloat(uint16_t b)
{
    union {
        uint32_t u;
        float f;
    } c;
    c.u = static_cast<uint32_t>(b) << 16;
    return c.f;
}

__aicore__ inline float HalfBitsToFloat(uint16_t h)
{
    uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t exp = (static_cast<uint32_t>(h) >> 10) & 0x1fu;
    uint32_t mant = static_cast<uint32_t>(h) & 0x3ffu;
    union {
        uint32_t u;
        float f;
    } c;
    if (exp == 0u) {
        if (mant == 0u) {
            c.u = sign;
        } else {
            uint32_t e = 127u - 15u + 1u;
            while ((mant & 0x400u) == 0u) {
                mant <<= 1;
                e--;
            }
            mant &= 0x3ffu;
            c.u = sign | (e << 23) | (mant << 13);
        }
    } else if (exp == 0x1fu) {
        c.u = sign | 0x7f800000u | (mant << 13);
    } else {
        c.u = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    return c.f;
}

__aicore__ inline int32_t ClampIdx(int32_t t, int32_t Ci)
{
    int32_t r = (t < 0) ? 0 : t;
    r = (r >= Ci) ? (Ci - 1) : r;
    return r;
}

template <typename T>
__aicore__ inline float ReadXVal(const LocalTensor<T>& xr, uint32_t idx)
{
    if constexpr (std::is_same_v<T, float>) {
        return xr.GetValue(idx);
    } else if constexpr (std::is_same_v<T, half>) {
        return HalfBitsToFloat(xr.template ReinterpretCast<uint16_t>().GetValue(idx));
    } else {
        return Bf16BitsToFloat(xr.template ReinterpretCast<uint16_t>().GetValue(idx));
    }
}

template <typename T>
__aicore__ inline void StoreScalar(GlobalTensor<uint16_t>& gU, GlobalTensor<float>& gF, uint32_t idx, float v)
{
    if constexpr (std::is_same_v<T, float>) {
        gF.SetValue(idx, v);
    } else if constexpr (std::is_same_v<T, half>) {
        gU.SetValue(idx, FloatToHalfBits(v));
    } else {
        gU.SetValue(idx, FloatToBf16Bits(v));
    }
}

template <typename T>
class NllLossKernel {
public:
    __aicore__ inline NllLossKernel() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR target, GM_ADDR weight, GM_ADDR y, GM_ADDR tw, GM_ADDR workspace,
                                TPipe* pipeIn, const NllLossTilingData* td)
    {
        this->pipe = pipeIn;
        this->N = td->rowNum;
        this->C = td->classNum;
        this->reduction = td->reduction;
        this->ignoreIndex = td->ignoreIndex;
        this->hasWeight = (td->hasWeight != 0u);
        this->targetIsInt64 = (td->targetIsInt64 != 0u);
        this->usedCore = td->usedCoreNum;
        this->rowsPerCore = td->rowsPerCore;
        this->tileRows = (td->tileRows == 0u) ? 1u : td->tileRows;
        this->useVector = (td->useVector != 0u);

        this->coreIdx = static_cast<uint64_t>(GetBlockIdx());
        this->rowStart = this->coreIdx * this->rowsPerCore;
        uint64_t re = this->rowStart + this->rowsPerCore;
        this->rowEnd = (re < this->N) ? re : this->N;
        this->myRows = (this->rowEnd > this->rowStart) ? (this->rowEnd - this->rowStart) : 0u;

        this->xGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x));
        this->yGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y));
        this->yGmU.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(y));
        this->yGmF.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y));
        this->twGmU.SetGlobalBuffer(reinterpret_cast<__gm__ uint16_t*>(tw));
        this->twGmF.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(tw));
        if (this->hasWeight) {
            this->wGm.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(weight));
        }
        if (this->targetIsInt64) {
            this->tgtGmI64.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(target));
        } else {
            this->tgtGmI32.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(target));
        }
        this->wsGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace + BLK_BYTES * this->usedCore));

        uint32_t tr = static_cast<uint32_t>(this->tileRows);
        this->trA = CeilAlign(tr, VEC_ALIGN);
        this->blkA = CeilAlign(tr * static_cast<uint32_t>(this->C), VEC_ALIGN);
        this->cA = CeilAlign(static_cast<uint32_t>(this->C), VEC_ALIGN);
        this->segLen = CeilAlign(static_cast<uint32_t>(this->usedCore) * BLK_ELEM, VEC_ALIGN);
        this->InitBuffers();
    }

    __aicore__ inline void Process()
    {
        LocalTensor<float> wF32 = this->wF32Buf.template Get<float>();
        if (this->hasWeight) {
            this->LoadWeight(wF32);
        }

        float accLossVal = 0.0f;
        float accWVal = 0.0f;
        if (this->useVector) {
            this->RunVector(wF32, accLossVal, accWVal);
        } else {
            for (uint64_t off = 0u; off < this->myRows; off += this->tileRows) {
                uint64_t cur = this->myRows - off;
                if (cur > this->tileRows) {
                    cur = this->tileRows;
                }
                this->ProcessTileScalar(this->rowStart + off, static_cast<uint32_t>(cur), wF32, accLossVal, accWVal);
            }
        }

        if (this->usedCore == 1u) {
            this->WriteReduced(accLossVal, accWVal);
        } else {
            this->CrossCoreReduce(accLossVal, accWVal);
        }
    }

private:
    __aicore__ inline void InitBuffers()
    {
        this->pipe->InitBuffer(this->xInQue, 1, this->blkA * sizeof(T));
        this->pipe->InitBuffer(this->tInQue, 1, this->trA * sizeof(int64_t));
        if (this->hasWeight) {
            this->pipe->InitBuffer(this->wInQue, 1, this->cA * sizeof(T));
        }
        this->pipe->InitBuffer(this->yOutQue, 1, this->trA * sizeof(T));

        this->pipe->InitBuffer(this->wF32Buf, this->cA * sizeof(float));
        this->pipe->InitBuffer(this->yF32Buf, this->trA * sizeof(float));

        if (this->useVector) {
            this->pipe->InitBuffer(this->rowBaseBuf, this->trA * sizeof(int32_t));
            this->pipe->InitBuffer(this->accLossBuf, this->trA * sizeof(float));
            this->pipe->InitBuffer(this->accWBuf, this->trA * sizeof(float));
            this->pipe->InitBuffer(this->tFBuf, this->trA * sizeof(float));
            this->pipe->InitBuffer(this->idxBuf, this->trA * sizeof(int32_t));
            this->pipe->InitBuffer(this->offBuf, this->trA * sizeof(uint32_t));
            this->pipe->InitBuffer(this->xGathBuf, this->trA * sizeof(float));
            this->pipe->InitBuffer(this->gRawBuf, this->trA * sizeof(T));
            this->pipe->InitBuffer(this->wGathBuf, this->trA * sizeof(float));
            this->pipe->InitBuffer(this->maskBuf, this->trA);
        }

        uint32_t redWork = (this->trA > this->segLen) ? this->trA : this->segLen;
        this->pipe->InitBuffer(this->redTmpBuf, redWork * sizeof(float));
        this->pipe->InitBuffer(this->redOutBuf, VEC_ALIGN * sizeof(float));
        this->pipe->InitBuffer(this->partialOutQue, 1, 2u * BLK_BYTES);
        this->pipe->InitBuffer(this->finalInQue, 1, this->segLen * sizeof(float));
    }

    __aicore__ inline void LoadWeight(const LocalTensor<float>& wF32)
    {
        LocalTensor<T> wRaw = this->wInQue.template AllocTensor<T>();
        DataCopyExtParams wp{1u, static_cast<uint32_t>(this->C * sizeof(T)), 0u, 0u, 0u};
        DataCopyPadExtParams<T> pad{false, 0u, 0u, 0};
        DataCopyPad(wRaw, this->wGm, wp, pad);
        this->wInQue.EnQue(wRaw);
        wRaw = this->wInQue.template DeQue<T>();
        if constexpr (std::is_same_v<T, float>) {
            Adds(wF32, wRaw.template ReinterpretCast<float>(), 0.0f, static_cast<int32_t>(this->C));
        } else {
            Cast(wF32, wRaw, RoundMode::CAST_NONE, static_cast<int32_t>(this->C));
        }
        this->wInQue.FreeTensor(wRaw);
        if (!this->useVector) {
            event_t eidVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eidVS);
            WaitFlag<HardEvent::V_S>(eidVS);
        }
    }

    // load one row-tile of x and target into UB (shared by vector and scalar paths).
    __aicore__ inline void LoadInputTile(uint64_t globalRow0, uint32_t cur, LocalTensor<T>& xRaw,
                                         LocalTensor<int32_t>& t32, LocalTensor<int64_t>& t64)
    {
        uint32_t Cu = static_cast<uint32_t>(this->C);
        xRaw = this->xInQue.template AllocTensor<T>();
        DataCopyExtParams xp{1u, static_cast<uint32_t>(cur * Cu * sizeof(T)), 0u, 0u, 0u};
        DataCopyPadExtParams<T> xpad{false, 0u, 0u, 0};
        DataCopyPad(xRaw, this->xGm[globalRow0 * this->C], xp, xpad);
        this->xInQue.EnQue(xRaw);
        xRaw = this->xInQue.template DeQue<T>();

        if (this->targetIsInt64) {
            t64 = this->tInQue.template AllocTensor<int64_t>();
            DataCopyExtParams tp{1u, static_cast<uint32_t>(cur * sizeof(int64_t)), 0u, 0u, 0u};
            DataCopyPadExtParams<int64_t> tpad{false, 0u, 0u, 0};
            DataCopyPad(t64, this->tgtGmI64[globalRow0], tp, tpad);
            this->tInQue.EnQue(t64);
            t64 = this->tInQue.template DeQue<int64_t>();
        } else {
            t32 = this->tInQue.template AllocTensor<int32_t>();
            DataCopyExtParams tp{1u, static_cast<uint32_t>(cur * sizeof(int32_t)), 0u, 0u, 0u};
            DataCopyPadExtParams<int32_t> tpad{false, 0u, 0u, 0};
            DataCopyPad(t32, this->tgtGmI32[globalRow0], tp, tpad);
            this->tInQue.EnQue(t32);
            t32 = this->tInQue.template DeQue<int32_t>();
        }
    }

    __aicore__ inline void FreeInputTile(LocalTensor<T>& xRaw, LocalTensor<int32_t>& t32, LocalTensor<int64_t>& t64)
    {
        this->xInQue.FreeTensor(xRaw);
        if (this->targetIsInt64) {
            this->tInQue.FreeTensor(t64);
        } else {
            this->tInQue.FreeTensor(t32);
        }
    }

    // cast a float loss tile to T and copy out to y (reduction == none).
    __aicore__ inline void WriteRowsOutput(uint64_t globalRow0, uint32_t cur, const LocalTensor<float>& srcF)
    {
        LocalTensor<T> yOut = this->yOutQue.template AllocTensor<T>();
        if constexpr (std::is_same_v<T, float>) {
            Adds(yOut.template ReinterpretCast<float>(), srcF, 0.0f, static_cast<int32_t>(cur));
        } else {
            Cast(yOut, srcF, RoundMode::CAST_RINT, static_cast<int32_t>(cur));
        }
        this->yOutQue.EnQue(yOut);
        yOut = this->yOutQue.template DeQue<T>();
        DataCopyExtParams yp{1u, static_cast<uint32_t>(cur * sizeof(T)), 0u, 0u, 0u};
        DataCopyPad(this->yGm[globalRow0], yOut, yp);
        this->yOutQue.FreeTensor(yOut);
    }

    __aicore__ inline void RunVector(const LocalTensor<float>& wF32, float& accLossVal, float& accWVal)
    {
        LocalTensor<int32_t> rowBase = this->rowBaseBuf.template Get<int32_t>();
        CreateVecIndex(rowBase, 0, static_cast<int32_t>(this->trA));
        Muls(rowBase, rowBase, static_cast<int32_t>(this->C), static_cast<int32_t>(this->trA));

        LocalTensor<float> accLoss = this->accLossBuf.template Get<float>();
        LocalTensor<float> accW = this->accWBuf.template Get<float>();
        Duplicate(accLoss, 0.0f, static_cast<int32_t>(this->trA));
        Duplicate(accW, 0.0f, static_cast<int32_t>(this->trA));

        for (uint64_t off = 0u; off < this->myRows; off += this->tileRows) {
            uint64_t cur = this->myRows - off;
            if (cur > this->tileRows) {
                cur = this->tileRows;
            }
            this->ProcessTileVector(this->rowStart + off, static_cast<uint32_t>(cur), wF32, rowBase, accLoss, accW);
        }

        LocalTensor<float> redTmp = this->redTmpBuf.template Get<float>();
        LocalTensor<float> redOut = this->redOutBuf.template Get<float>();
        if (this->reduction != 0) {
            ReduceSum(redOut, accLoss, redTmp, static_cast<int32_t>(this->trA));
        }
        ReduceSum(redOut[BLK_ELEM], accW, redTmp, static_cast<int32_t>(this->trA));
        event_t eRS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eRS);
        WaitFlag<HardEvent::V_S>(eRS);
        accLossVal = (this->reduction != 0) ? redOut.GetValue(0) : 0.0f;
        accWVal = redOut.GetValue(BLK_ELEM);
    }

    // gather the target-class x value (and weight) for every row of the tile.
    __aicore__ inline void GatherTileValues(const LocalTensor<T>& xRaw, const LocalTensor<float>& wF32,
                                            const LocalTensor<int32_t>& idx, const LocalTensor<int32_t>& offI,
                                            const LocalTensor<int32_t>& rowBase, const LocalTensor<float>& xGathF,
                                            const LocalTensor<float>& wGathF, uint32_t cur, int32_t curI)
    {
        Add(offI, rowBase, idx, curI);
        Muls(offI, offI, static_cast<int32_t>(sizeof(T)), curI);
        if constexpr (std::is_same_v<T, float>) {
            Gather(xGathF, xRaw, offI.template ReinterpretCast<uint32_t>(), 0u, cur);
        } else {
            LocalTensor<T> gRaw = this->gRawBuf.template Get<T>();
            Gather(gRaw, xRaw, offI.template ReinterpretCast<uint32_t>(), 0u, cur);
            Cast(xGathF, gRaw, RoundMode::CAST_NONE, curI);
        }

        if (this->hasWeight) {
            Muls(offI, idx, static_cast<int32_t>(sizeof(float)), curI);
            Gather(wGathF, wF32, offI.template ReinterpretCast<uint32_t>(), 0u, cur);
        } else {
            Duplicate(wGathF, 1.0f, curI);
        }
    }

    __aicore__ inline void ProcessTileVector(uint64_t globalRow0, uint32_t cur, const LocalTensor<float>& wF32,
                                             const LocalTensor<int32_t>& rowBase, const LocalTensor<float>& accLoss,
                                             const LocalTensor<float>& accW)
    {
        int32_t curI = static_cast<int32_t>(cur);
        LocalTensor<T> xRaw;
        LocalTensor<int32_t> t32;
        LocalTensor<int64_t> t64;
        this->LoadInputTile(globalRow0, cur, xRaw, t32, t64);

        LocalTensor<float> tF = this->tFBuf.template Get<float>();
        LocalTensor<int32_t> idx = this->idxBuf.template Get<int32_t>();
        LocalTensor<int32_t> offI = this->offBuf.template Get<int32_t>();
        LocalTensor<float> xGathF = this->xGathBuf.template Get<float>();
        LocalTensor<float> wGathF = this->wGathBuf.template Get<float>();
        LocalTensor<float> lossF = this->yF32Buf.template Get<float>();
        LocalTensor<uint8_t> mask = this->maskBuf.template Get<uint8_t>();

        if (this->targetIsInt64) {
            Cast(idx, t64, RoundMode::CAST_NONE, curI);
            Cast(tF, idx, RoundMode::CAST_RINT, curI);
        } else {
            Cast(tF, t32, RoundMode::CAST_RINT, curI);
        }

        uint32_t cmpCount = CeilAlign(cur, VEC_ALIGN);
        float ignoreF = static_cast<float>(this->ignoreIndex);
        CompareScalar(mask, tF, ignoreF, CMPMODE::NE, cmpCount);

        Maxs(tF, tF, 0.0f, curI);
        Mins(tF, tF, static_cast<float>(static_cast<int32_t>(this->C) - 1), curI);
        Cast(idx, tF, RoundMode::CAST_RINT, curI);

        this->GatherTileValues(xRaw, wF32, idx, offI, rowBase, xGathF, wGathF, cur, curI);

        Mul(lossF, wGathF, xGathF, curI);
        Muls(lossF, lossF, -1.0f, curI);
        Select(lossF, mask, lossF, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, cur);
        Select(wGathF, mask, wGathF, 0.0f, SELMODE::VSEL_TENSOR_SCALAR_MODE, cur);

        if (this->reduction != 0) {
            Add(accLoss, accLoss, lossF, curI);
        }
        Add(accW, accW, wGathF, curI);

        if (this->reduction == 0) {
            this->WriteRowsOutput(globalRow0, cur, lossF);
        }

        this->FreeInputTile(xRaw, t32, t64);
    }

    __aicore__ inline void ProcessTileScalar(uint64_t globalRow0, uint32_t cur, const LocalTensor<float>& wF32,
                                             float& accLoss, float& accW)
    {
        LocalTensor<T> xRaw;
        LocalTensor<int32_t> t32;
        LocalTensor<int64_t> t64;
        this->LoadInputTile(globalRow0, cur, xRaw, t32, t64);

        event_t eidMS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(eidMS);
        WaitFlag<HardEvent::MTE2_S>(eidMS);

        LocalTensor<float> yF = this->yF32Buf.template Get<float>();
        bool isNone = (this->reduction == 0);
        int32_t Ci = static_cast<int32_t>(this->C);
        int32_t ig = static_cast<int32_t>(this->ignoreIndex);
        uint32_t Cu = static_cast<uint32_t>(this->C);
        for (uint32_t j = 0u; j < cur; j++) {
            int32_t t = this->targetIsInt64 ? static_cast<int32_t>(t64.GetValue(j)) : t32.GetValue(j);
            float loss = 0.0f;
            float w = 0.0f;
            if (t != ig) {
                int32_t tc = ClampIdx(t, Ci);
                float xv = ReadXVal<T>(xRaw, j * Cu + static_cast<uint32_t>(tc));
                w = this->hasWeight ? wF32.GetValue(static_cast<uint32_t>(tc)) : 1.0f;
                loss = -w * xv;
            }
            if (isNone) {
                yF.SetValue(j, loss);
            }
            accLoss += loss;
            accW += w;
        }

        this->FreeInputTile(xRaw, t32, t64);

        if (isNone) {
            event_t eidSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(eidSV);
            WaitFlag<HardEvent::S_V>(eidSV);
            this->WriteRowsOutput(globalRow0, cur, yF);
        }
    }

    __aicore__ inline void CrossCoreReduce(float accLossVal, float accWVal)
    {
        LocalTensor<float> partUb = this->partialOutQue.template AllocTensor<float>();
        Duplicate(partUb, 0.0f, static_cast<int32_t>(2u * BLK_ELEM));
        PipeBarrier<PIPE_V>();
        Duplicate(partUb, accLossVal, 1);
        Duplicate(partUb[BLK_ELEM], accWVal, 1);
        PipeBarrier<PIPE_V>();
        this->partialOutQue.EnQue(partUb);
        partUb = this->partialOutQue.template DeQue<float>();
        DataCopy(this->wsGm[this->coreIdx * BLK_ELEM], partUb, BLK_ELEM);
        DataCopy(this->wsGm[this->usedCore * BLK_ELEM + this->coreIdx * BLK_ELEM], partUb[BLK_ELEM], BLK_ELEM);
        this->partialOutQue.FreeTensor(partUb);
        AscendC::SyncAll();

        if (this->coreIdx == 0u) {
            this->FinalReduceAndWrite();
        }
    }

    __aicore__ inline void FinalReduceAndWrite()
    {
        uint32_t cnt = static_cast<uint32_t>(this->usedCore) * BLK_ELEM;

        LocalTensor<float> redTmp = this->redTmpBuf.template Get<float>();
        LocalTensor<float> redOut = this->redOutBuf.template Get<float>();

        float totalLoss = 0.0f;
        if (this->reduction != 0) {
            LocalTensor<float> lossAll = this->finalInQue.template AllocTensor<float>();
            DataCopy(lossAll, this->wsGm, cnt);
            this->finalInQue.EnQue(lossAll);
            lossAll = this->finalInQue.template DeQue<float>();
            if (this->segLen > cnt) {
                Duplicate(lossAll[cnt], 0.0f, static_cast<int32_t>(this->segLen - cnt));
                PipeBarrier<PIPE_V>();
            }
            ReduceSum(redOut, lossAll, redTmp, static_cast<int32_t>(this->segLen));
            event_t eidFL = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eidFL);
            WaitFlag<HardEvent::V_S>(eidFL);
            totalLoss = redOut.GetValue(0);
            this->finalInQue.FreeTensor(lossAll);
        }

        LocalTensor<float> wAll = this->finalInQue.template AllocTensor<float>();
        DataCopy(wAll, this->wsGm[this->usedCore * BLK_ELEM], cnt);
        this->finalInQue.EnQue(wAll);
        wAll = this->finalInQue.template DeQue<float>();
        if (this->segLen > cnt) {
            Duplicate(wAll[cnt], 0.0f, static_cast<int32_t>(this->segLen - cnt));
            PipeBarrier<PIPE_V>();
        }
        ReduceSum(redOut, wAll, redTmp, static_cast<int32_t>(this->segLen));
        event_t eidFW = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
        SetFlag<HardEvent::V_S>(eidFW);
        WaitFlag<HardEvent::V_S>(eidFW);
        float totalWeight = redOut.GetValue(0);
        this->finalInQue.FreeTensor(wAll);

        this->WriteReduced(totalLoss, totalWeight);
    }

    __aicore__ inline void WriteReduced(float totalLoss, float totalWeight)
    {
        StoreScalar<T>(this->twGmU, this->twGmF, 0, totalWeight);
        if (this->reduction == 2) {
            StoreScalar<T>(this->yGmU, this->yGmF, 0, totalLoss);
        } else if (this->reduction == 1) {
            float m = (totalWeight != 0.0f) ? (totalLoss / totalWeight) : 0.0f;
            StoreScalar<T>(this->yGmU, this->yGmF, 0, m);
        }
    }

    TPipe* pipe = nullptr;
    TQue<QuePosition::VECIN, 1> xInQue;
    TQue<QuePosition::VECIN, 1> tInQue;
    TQue<QuePosition::VECIN, 1> wInQue;
    TQue<QuePosition::VECOUT, 1> yOutQue;
    TQue<QuePosition::VECOUT, 1> partialOutQue;
    TQue<QuePosition::VECIN, 1> finalInQue;

    TBuf<QuePosition::VECCALC> wF32Buf;
    TBuf<QuePosition::VECCALC> yF32Buf;
    TBuf<QuePosition::VECCALC> rowBaseBuf;
    TBuf<QuePosition::VECCALC> accLossBuf;
    TBuf<QuePosition::VECCALC> accWBuf;
    TBuf<QuePosition::VECCALC> tFBuf;
    TBuf<QuePosition::VECCALC> idxBuf;
    TBuf<QuePosition::VECCALC> offBuf;
    TBuf<QuePosition::VECCALC> xGathBuf;
    TBuf<QuePosition::VECCALC> gRawBuf;
    TBuf<QuePosition::VECCALC> wGathBuf;
    TBuf<QuePosition::VECCALC> maskBuf;
    TBuf<QuePosition::VECCALC> redTmpBuf;
    TBuf<QuePosition::VECCALC> redOutBuf;

    GlobalTensor<T> xGm;
    GlobalTensor<T> wGm;
    GlobalTensor<T> yGm;
    GlobalTensor<uint16_t> yGmU;
    GlobalTensor<float> yGmF;
    GlobalTensor<uint16_t> twGmU;
    GlobalTensor<float> twGmF;
    GlobalTensor<int32_t> tgtGmI32;
    GlobalTensor<int64_t> tgtGmI64;
    GlobalTensor<float> wsGm;

    uint64_t N = 0u;
    uint64_t C = 0u;
    int64_t reduction = 1;
    int64_t ignoreIndex = -100;
    bool hasWeight = false;
    bool targetIsInt64 = false;
    bool useVector = true;
    uint64_t usedCore = 1u;
    uint64_t rowsPerCore = 0u;
    uint64_t tileRows = 0u;
    uint64_t coreIdx = 0u;
    uint64_t rowStart = 0u;
    uint64_t rowEnd = 0u;
    uint64_t myRows = 0u;
    uint32_t trA = 0u;
    uint32_t blkA = 0u;
    uint32_t cA = 0u;
    uint32_t segLen = 0u;
};

template <typename T>
__aicore__ inline void Run(GM_ADDR x, GM_ADDR target, GM_ADDR weight, GM_ADDR y, GM_ADDR tw, GM_ADDR workspace,
                           const NllLossTilingData* tilingData)
{
    TPipe pipe;
    NllLossKernel<T> op;
    op.Init(x, target, weight, y, tw, workspace, &pipe, tilingData);
    op.Process();
}

} // namespace NsNllLoss

#endif
