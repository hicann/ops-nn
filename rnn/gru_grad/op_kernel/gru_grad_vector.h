/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file gru_grad_vector.h
 * \brief
 */

__aicore__ inline void SyncM2toV()
{
    event_t eventId = static_cast<event_t>(this->pipe.FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventId);
    WaitFlag<HardEvent::MTE2_V>(eventId);
};

__aicore__ inline void SyncVtoM3()
{
    event_t eventId = static_cast<event_t>(this->pipe.FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventId);
    WaitFlag<HardEvent::V_MTE3>(eventId);
};

__aicore__ inline void SyncVtoM2()
{
    event_t eventId = static_cast<event_t>(this->pipe.FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(eventId);
    WaitFlag<HardEvent::V_MTE2>(eventId);
};

__aicore__ inline void SyncM3toV()
{
    event_t eventId = static_cast<event_t>(this->pipe.FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eventId);
    WaitFlag<HardEvent::MTE3_V>(eventId);
}

__aicore__ inline void SyncM3toM2()
{
    event_t eventId = static_cast<event_t>(this->pipe.FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventId);
    WaitFlag<HardEvent::MTE3_MTE2>(eventId);
};

__aicore__ inline void SyncM2toM3()
{
    event_t eventId = static_cast<event_t>(this->pipe.FetchEventID(HardEvent::MTE2_MTE3));
    SetFlag<HardEvent::MTE2_MTE3>(eventId);
    WaitFlag<HardEvent::MTE2_MTE3>(eventId);
};

template <typename T>
__aicore__ inline void CopyInRow(GlobalTensor<T>& gm, LocalTensor<float>& ub, int64_t gmOffset, int64_t ubOffset,
                                 int64_t rows, int64_t length, int64_t alignedLen, int64_t gmStride)
{
    if constexpr (sizeof(T) == 2) {
        auto tmpUb = ub.template ReinterpretCast<T>();
        DataCopyExtParams cp(static_cast<uint16_t>(rows), static_cast<uint32_t>(length * sizeof(T)),
                             static_cast<uint32_t>(gmStride * sizeof(T)), static_cast<uint32_t>(0), 0);
        DataCopyPadExtParams<T> pp(true, 0, 0, 0);
        DataCopyPad(tmpUb[allocLength], gm[gmOffset], cp, pp);
        SyncM2toV();
        AscendC::Cast(ub, tmpUb[allocLength], AscendC::RoundMode::CAST_NONE, allocLength);
        PipeBarrier<PIPE_V>();
    } else {
        DataCopyExtParams cp(static_cast<uint16_t>(rows), static_cast<uint32_t>(length * sizeof(T)),
                             static_cast<uint32_t>(gmStride * sizeof(T)), static_cast<uint32_t>(0), 0);
        DataCopyPadExtParams<T> pp(true, 0, 0, 0);
        DataCopyPad(ub, gm[gmOffset], cp, pp);
    }
}

template <typename T>
__aicore__ inline void CopyOutRow(GlobalTensor<T>& gm, LocalTensor<float>& ub, int64_t gmOffset, int64_t ubOffset,
                                  int64_t rows, int64_t length, int64_t gmStride, int64_t ubRowStride)
{
    if constexpr (sizeof(T) == 2) {
        auto tmpUb = this->ubTmp2.template ReinterpretCast<T>();
        AscendC::Cast(tmpUb, ub, AscendC::RoundMode::CAST_RINT, allocLength);
        SyncVtoM3();
        DataCopyExtParams cp(static_cast<uint16_t>(rows), static_cast<uint32_t>(length * sizeof(T)),
                             static_cast<uint32_t>(0), static_cast<uint32_t>(gmStride * sizeof(T)),
                             static_cast<uint32_t>(ubOffset * sizeof(T)));
        DataCopyPad(gm[gmOffset], tmpUb, cp);
    } else {
        int64_t ubGapBytes = (ubRowStride - length) * static_cast<int64_t>(sizeof(float));
        DataCopyExtParams cp(static_cast<uint16_t>(rows), static_cast<uint32_t>(length * sizeof(T)),
                             static_cast<uint32_t>(0), static_cast<uint32_t>(gmStride * sizeof(T)),
                             static_cast<uint32_t>(ubOffset * sizeof(T)));
        DataCopyPad(gm[gmOffset], ub, cp);
    }
}

__aicore__ inline void ReadHPrev(int64_t tIdx, int64_t mStart, int64_t bOff, int64_t bRows, int64_t hOff, int64_t hLen,
                                 int64_t blkAligned, LocalTensor<float>& ub)
{
    const bool isReverse = (this->tiling->direction != 0);
    if (tIdx == 0) {
        CopyInRow(this->inputGm.initHGm, ub, (mStart + bOff) * this->H + hOff, 0, bRows, hLen, blkAligned, 0);
        return;
    }
    int64_t prevIdx = isReverse ? (this->T - tIdx) : (tIdx - 1);
    if (!this->tiling->isSeqLength) {
        CopyInRow(this->inputGm.outputHGm, ub, prevIdx * this->B * this->H + (mStart + bOff) * this->H + hOff, 0, bRows,
                  hLen, blkAligned, 0);
        return;
    }
    int64_t compactOutputRow = this->GetCompactRowOffset(prevIdx);
    int64_t bsPrev = this->inputGm.batchSizesGm.GetValue(prevIdx);
    int64_t rowStart = mStart + bOff; // 全局 batch 起点
    int64_t hAligned = ((hLen + ALIGN_32B_FP32_MASK) / ALIGN_32B_FP32) * ALIGN_32B_FP32;
    if (rowStart >= bsPrev) {
        CopyInRow(this->inputGm.initHGm, ub, rowStart * this->H + hOff, 0, bRows, hLen, blkAligned, 0);
    } else if (rowStart + bRows <= bsPrev) {
        CopyInRow(this->inputGm.outputHGm, ub, (compactOutputRow + rowStart) * this->H + hOff, 0, bRows, hLen,
                  blkAligned, 0);
    } else {
        int64_t nActive = bsPrev - rowStart;
        int64_t nNew = bRows - nActive;
        CopyInRow(this->inputGm.outputHGm, ub, (compactOutputRow + rowStart) * this->H + hOff, 0, nActive, hLen,
                  nActive * hAligned, 0);
        auto ubNewPart = ub[nActive * hAligned];
        CopyInRow(this->inputGm.initHGm, ubNewPart, (rowStart + nActive) * this->H + hOff, 0, nNew, hLen,
                  nNew * hAligned, 0);
    }
}

// grad_h_t  = dy[t] + dh_next
// dn(d_new) = grad_h_t * (1 - z)
// dz_raw    = grad_h_t * (hp - n)
// dh_prev_from_h = grad_h_t * z
// d_i_new   = dn * (1 - n^2)
// d_reset   = (d_i_new * h_n) * r * (1-r)
// d_update  = dz_raw * z * (1 - z)
// d_gi = [d_reset, d_update, d_i_new]
// d_gh = [d_reset, d_update, d_i_new*r]

// grad_h_t = dy[t] + dh_next (最后一步 dh_next 读 dhGm, 其余读 dhPrevWsGm)
__aicore__ inline void LoadGradH(int64_t tIdx, int64_t ghOff, int64_t batchHOff, int64_t bRows, int64_t hLen,
                                 int64_t blkAligned)
{
    CopyInRow(this->inputGm.dyGm, this->ubGradH, ghOff, 0, bRows, hLen, blkAligned, 0);
    if (tIdx == T - 1) {
        CopyInRow(this->inputGm.dhGm, this->ubTmp, batchHOff, 0, bRows, hLen, blkAligned, 0);
    } else {
        CopyInRow(this->workGm.dhPrevWsGm, this->ubTmp, batchHOff, 0, bRows, hLen, blkAligned, 0);
    }
    SyncM2toV();
    Add(this->ubGradH, this->ubGradH, this->ubTmp, blkAligned);
}

__aicore__ inline void LoadGatesAndHPrev(int64_t tIdx, int64_t ghOff, int64_t mStart, int64_t bOff, int64_t bRows,
                                         int64_t hOff, int64_t hLen, int64_t blkAligned)
{
    CopyInRow(this->inputGm.updateGateGm, this->ubUpdate, ghOff, 0, bRows, hLen, blkAligned, 0);
    CopyInRow(this->inputGm.newGateGm, this->ubNew, ghOff, 0, bRows, hLen, blkAligned, 0);
    CopyInRow(this->inputGm.resetGateGm, this->ubReset, ghOff, 0, bRows, hLen, blkAligned, 0);
    CopyInRow(this->inputGm.hNGm, this->ubHN, ghOff, 0, bRows, hLen, blkAligned, 0);
    this->ReadHPrev(tIdx, mStart, bOff, bRows, hOff, hLen, blkAligned, this->ubHPrev);
}

// d_i_new = grad_h_t * (1-z) * (1-n^2)  -> dGi[new]
__aicore__ inline void ComputeDINew(int64_t gmOffGi, int64_t bRows, int64_t hLen, int64_t gmGateStride,
                                    int64_t blkAligned)
{
    SyncM2toV();
    Duplicate(this->ubGate, 1.0f, blkAligned);
    Sub(this->ubTmp, this->ubGate, this->ubUpdate, blkAligned);
    Mul(this->ubTmp2, this->ubGradH, this->ubTmp, blkAligned);
    Mul(this->ubTmp, this->ubNew, this->ubNew, blkAligned);
    Sub(this->ubTmp, this->ubGate, this->ubTmp, blkAligned);
    Mul(this->ubDHPrevFromH, this->ubTmp2, this->ubTmp, blkAligned);
    SyncVtoM3();
    CopyOutRow(this->workGm.dGiGm, this->ubDHPrevFromH, gmOffGi + GATE_IDX_NEW * H, 0, bRows, hLen, gmGateStride,
               blkAligned);
    SyncM3toV();
}

// d_reset = (d_i_new * h_n) * r * (1-r)  -> dGi[reset], dGh[reset]
__aicore__ inline void ComputeDReset(int64_t gmOffGi, int64_t bRows, int64_t hLen, int64_t gmGateStride,
                                     int64_t blkAligned)
{
    Mul(this->ubTmp, this->ubDHPrevFromH, this->ubHN, blkAligned);
    Mul(this->ubTmp2, this->ubTmp, this->ubReset, blkAligned);
    Sub(this->ubTmp, this->ubGate, this->ubReset, blkAligned);
    Mul(this->ubTmp3, this->ubTmp2, this->ubTmp, blkAligned);
    SyncVtoM3();
    CopyOutRow(this->workGm.dGiGm, this->ubTmp3, gmOffGi, 0, bRows, hLen, gmGateStride, blkAligned);
    CopyOutRow(this->workGm.dGhGm, this->ubTmp3, gmOffGi, 0, bRows, hLen, gmGateStride, blkAligned);
    SyncM3toV();
}

// d_update = grad_h_t * (hp-n) * z * (1-z)  -> dGi[update], dGh[update]
__aicore__ inline void ComputeDUpdate(int64_t gmOffGi, int64_t bRows, int64_t hLen, int64_t gmGateStride,
                                      int64_t blkAligned)
{
    Sub(this->ubTmp, this->ubHPrev, this->ubNew, blkAligned);
    Mul(this->ubTmp2, this->ubGradH, this->ubTmp, blkAligned);
    Mul(this->ubTmp, this->ubTmp2, this->ubUpdate, blkAligned);
    Sub(this->ubTmp2, this->ubGate, this->ubUpdate, blkAligned);
    Mul(this->ubTmp3, this->ubTmp, this->ubTmp2, blkAligned);
    SyncVtoM3();
    CopyOutRow(this->workGm.dGiGm, this->ubTmp3, gmOffGi + H, 0, bRows, hLen, gmGateStride, blkAligned);
    CopyOutRow(this->workGm.dGhGm, this->ubTmp3, gmOffGi + H, 0, bRows, hLen, gmGateStride, blkAligned);
    SyncM3toV();
}

// d_gh[new] = d_i_new * r;  dh_prev_from_h = grad_h_t * z  -> dhFromH
__aicore__ inline void ComputeDGhNewAndDhFromH(int64_t gmOffGi, int64_t batchHOff, int64_t bRows, int64_t hLen,
                                               int64_t gmGateStride, int64_t blkAligned)
{
    Mul(this->ubTmp3, this->ubDHPrevFromH, this->ubReset, blkAligned);
    SyncVtoM3();
    CopyOutRow(this->workGm.dGhGm, this->ubTmp3, gmOffGi + GATE_IDX_NEW * H, 0, bRows, hLen, gmGateStride, blkAligned);
    SyncM3toV();
    Mul(this->ubDHPrevFromH, this->ubGradH, this->ubUpdate, blkAligned);
    SyncVtoM3();
    CopyOutRow(this->workGm.dhFromHGm, this->ubDHPrevFromH, batchHOff, 0, bRows, hLen, 0, blkAligned);
    SyncM3toV();
}

__aicore__ inline void ProcessVectorHTile(int64_t tIdx, int64_t gateIdx, int64_t mStart, int64_t bOff, int64_t bRows,
                                          int64_t ht, int64_t compactGateRow, int64_t gm3H, int64_t gmGateStride,
                                          int64_t hTiles)
{
    int64_t hLen = this->vecHTile_;
    int64_t hOff = ht * this->vecHTile_;
    if (ht == hTiles - 1) {
        hLen = H - hOff;
    }
    int64_t hAligned = ((hLen + ALIGN_32B_FP32_MASK) / ALIGN_32B_FP32) * ALIGN_32B_FP32;
    int64_t blkAligned = bRows * hAligned;

    int64_t ghOff, gmOffGi;
    if (this->tiling->isSeqLength) {
        ghOff = (compactGateRow + mStart + bOff) * H + hOff;
        gmOffGi = (compactGateRow + mStart + bOff) * gm3H + hOff;
    } else {
        ghOff = gateIdx * B * H + (mStart + bOff) * H + hOff;
        gmOffGi = gateIdx * B * gm3H + (mStart + bOff) * gm3H + hOff;
    }
    int64_t batchHOff = (mStart + bOff) * H + hOff;

    this->LoadGradH(tIdx, ghOff, batchHOff, bRows, hLen, blkAligned);
    this->LoadGatesAndHPrev(tIdx, ghOff, mStart, bOff, bRows, hOff, hLen, blkAligned);
    this->ComputeDINew(gmOffGi, bRows, hLen, gmGateStride, blkAligned);
    this->ComputeDReset(gmOffGi, bRows, hLen, gmGateStride, blkAligned);
    this->ComputeDUpdate(gmOffGi, bRows, hLen, gmGateStride, blkAligned);
    this->ComputeDGhNewAndDhFromH(gmOffGi, batchHOff, bRows, hLen, gmGateStride, blkAligned);
}

__aicore__ inline void StoreHPrevHTile(int64_t tIdx, int64_t gateIdx, int64_t mStart, int64_t bOff, int64_t bRows,
                                       int64_t ht, int64_t compactGateRow, int64_t hTiles)
{
    int64_t hLen = this->vecHTile_;
    int64_t hOff = ht * this->vecHTile_;
    if (ht == hTiles - 1) {
        hLen = H - hOff;
    }
    int64_t hAligned = ((hLen + ALIGN_32B_FP32_MASK) / ALIGN_32B_FP32) * ALIGN_32B_FP32;
    int64_t blkAligned = bRows * hAligned;

    Duplicate(this->ubTmp2, 0.0f, blkAligned);
    SyncVtoM2();
    this->ReadHPrev(tIdx, mStart, bOff, bRows, hOff, hLen, blkAligned, this->ubTmp2);
    SyncM2toM3();
    int64_t hPrevWsOff;
    if (this->tiling->isSeqLength) {
        hPrevWsOff = (compactGateRow + mStart + bOff) * H + hOff;
    } else {
        hPrevWsOff = (gateIdx * B + mStart + bOff) * H + hOff;
    }
    CopyOutRow(this->workGm.hPrevWsGm, this->ubTmp2, hPrevWsOff, 0, bRows, hLen, 0, blkAligned);
    SyncM3toV();
}

__aicore__ inline void ProcessVector(int64_t tIdx)
{
    const int64_t bTile = this->vecBTile_, hTile = this->vecHTile_;
    const bool isReverse = (this->tiling->direction != 0);
    const int64_t gateIdx = isReverse ? (this->T - 1 - tIdx) : tIdx;
    const int64_t gm3H = this->H * GRU_GATE_SIZE;
    const int64_t gmGateStride = this->H * GATE_IDX_NEW;

    int64_t compactGateRow = 0, curBatch = this->B;
    if (this->tiling->isSeqLength) {
        compactGateRow = this->GetCompactRowOffset(gateIdx);
        curBatch = this->inputGm.batchSizesGm.GetValue(gateIdx);
    }

    int64_t mStart = 0, mCnt = 0;
    if (!this->GetCoreRows(mStart, mCnt)) {
        return;
    }
    if (this->tiling->isSeqLength) {
        if (mStart >= curBatch) {
            return;
        }
        if (mStart + mCnt > curBatch) {
            mCnt = curBatch - mStart;
        }
    }

    int64_t bTiles = (mCnt + bTile - 1) / bTile;
    int64_t hTiles = (H + hTile - 1) / hTile;
    for (int64_t bt = 0; bt < bTiles; bt++) {
        int64_t bRows = bTile;
        int64_t bOff = bt * bTile;
        if (bt == bTiles - 1) {
            bRows = mCnt - bOff;
        }
        for (int64_t ht = 0; ht < hTiles; ht++) {
            this->ProcessVectorHTile(tIdx, gateIdx, mStart, bOff, bRows, ht, compactGateRow, gm3H, gmGateStride,
                                     hTiles);
        }
        for (int64_t ht = 0; ht < hTiles; ht++) {
            this->StoreHPrevHTile(tIdx, gateIdx, mStart, bOff, bRows, ht, compactGateRow, hTiles);
        }
    }
}

__aicore__ inline void InitDhPrev()
{
    if (!this->tiling->isSeqLength) {
        return;
    }
    const int64_t bTile = this->vecBTile_, hTile = this->vecHTile_;
    int64_t mStart = 0, mCnt = 0;
    if (!this->GetCoreRows(mStart, mCnt)) {
        return;
    }
    int64_t bTiles = (mCnt + bTile - 1) / bTile;
    int64_t hTiles = (H + hTile - 1) / hTile;
    for (int64_t bT = 0; bT < bTiles; bT++) {
        int64_t bOff = bT * bTile, bRows = bTile;
        if (bT == bTiles - 1)
            bRows = mCnt - bOff;
        for (int64_t hT = 0; hT < hTiles; hT++) {
            int64_t hOff = hT * hTile, hLen = hTile;
            if (hT == hTiles - 1)
                hLen = H - hOff;
            int64_t hAligned = ((hLen + ALIGN_32B_FP32_MASK) / ALIGN_32B_FP32) * ALIGN_32B_FP32;
            CopyInRow(this->inputGm.dhGm, this->ubTmp, (mStart + bOff) * H + hOff, 0, bRows, hLen, hAligned, 0);
            SyncM2toM3();
            CopyOutRow(this->workGm.dhPrevWsGm, this->ubTmp, (mStart + bOff) * H + hOff, 0, bRows, hLen, 0, hAligned);
            SyncM3toM2();
        }
    }
}

// grad_h_prev = dgateMM_result + dh_prev_from_h
__aicore__ inline void AccumulateDhPrev(int64_t tIdx)
{
    const int64_t bTile = this->vecBTile_, hTile = this->vecHTile_;
    int64_t mStart = 0, mCnt = 0;
    if (!this->GetCoreRows(mStart, mCnt)) {
        return;
    }
    if (this->tiling->isSeqLength) {
        const bool isReverse = (this->tiling->direction != 0);
        int64_t gateIdx = isReverse ? (this->T - 1 - tIdx) : tIdx;
        int64_t curBatch = this->inputGm.batchSizesGm.GetValue(gateIdx);
        if (mStart >= curBatch) {
            return;
        }
        if (mStart + mCnt > curBatch) {
            mCnt = curBatch - mStart;
        }
    }
    int64_t bTiles = (mCnt + bTile - 1) / bTile;
    int64_t hTiles = (H + hTile - 1) / hTile;
    for (int64_t bT = 0; bT < bTiles; bT++) {
        int64_t bOff = bT * bTile, bRows = bTile;
        if (bT == bTiles - 1)
            bRows = mCnt - bOff;
        for (int64_t hT = 0; hT < hTiles; hT++) {
            int64_t hOff = hT * hTile, hLen = hTile;
            if (hT == hTiles - 1)
                hLen = H - hOff;
            hAligned = ((hLen + ALIGN_32B_FP32_MASK) / ALIGN_32B_FP32) * ALIGN_32B_FP32;
            CopyInRow(this->workGm.dhPrevWsGm, this->ubTmp, (mStart + bOff) * H + hOff, 0, bRows, hLen, hAligned, 0);
            CopyInRow(this->workGm.dhFromHGm, this->ubTmp2, (mStart + bOff) * H + hOff, 0, bRows, hLen, hAligned, 0);
            SyncM2toV();
            Add(this->ubTmp3, this->ubTmp, this->ubTmp2, bRows * hAligned);
            SyncVtoM3();
            CopyOutRow(this->workGm.dhPrevWsGm, this->ubTmp3, (mStart + bOff) * H + hOff, 0, bRows, hLen, 0, hAligned);
            SyncM3toM2();
        }
    }
}
__aicore__ inline void StoreDhPrev()
{
    const int64_t bTile = this->vecBTile_, hTile = this->vecHTile_;
    int64_t mStart = 0, mCnt = 0;
    if (!this->GetCoreRows(mStart, mCnt)) {
        return;
    }
    int64_t bTiles = (mCnt + bTile - 1) / bTile;
    int64_t hTiles = (H + hTile - 1) / hTile;
    for (int64_t bT = 0; bT < bTiles; bT++) {
        int64_t bOff = bT * bTile, bRows = bTile;
        if (bT == bTiles - 1)
            bRows = mCnt - bOff;
        for (int64_t hT = 0; hT < hTiles; hT++) {
            int64_t hOff = hT * hTile, hLen = hTile;
            if (hT == hTiles - 1)
                hLen = H - hOff;

            CopyInRow(this->workGm.dhPrevWsGm, this->ubTmp, (mStart + bOff) * H + hOff, 0, bRows, hLen, hAligned, 0);
            SyncM2toM3();
            CopyOutRow(this->outputGm.dhPrevGm, this->ubTmp, (mStart + bOff) * H + hOff, 0, bRows, hLen, 0, hAligned);
            SyncM3toM2();
        }
    }
}

__aicore__ inline void ProcessBiasReduce(GlobalTensor<DTYPE>& srcGm, GlobalTensor<DTYPE>& dstGm, int64_t rows,
                                         int64_t cols)
{
    SyncM3toV();
    int64_t nReduceCnt = this->tiling->nReduceCnt;
    int64_t singleCoreReduceN = this->tiling->singleCoreReduceN;
    if (nReduceCnt <= 0)
        return;
    if (GetBlockIdx() >= nReduceCnt)
        return;
    int64_t nIdx = GetBlockIdx();
    int64_t nStart = nIdx * singleCoreReduceN;
    int64_t nCnt = (nIdx == nReduceCnt - 1) ? this->tiling->singleCoreReduceNTail : singleCoreReduceN;
    if (nStart >= cols || nCnt <= 0)
        return;
    if (nStart + nCnt > cols)
        nCnt = cols - nStart;

    int64_t maxNOnce = allocLength;

    for (int64_t cStart = 0; cStart < nCnt; cStart += maxNOnce) {
        int64_t cCnt = (cStart + maxNOnce > nCnt) ? (nCnt - cStart) : maxNOnce;
        int64_t cAligned = ((cCnt + ALIGN_32B_FP32_MASK) / ALIGN_32B_FP32) * ALIGN_32B_FP32;
        int64_t cGlobalOff = nStart + cStart;

        Duplicate(this->ubTmp, 0.0f, cAligned);
        SyncVtoM2();
        for (int64_t r = 0; r < rows; ++r) {
            CopyInRow(srcGm, this->ubTmp2, r * cols + cGlobalOff, 0, 1, cCnt, cAligned, 0);
            SyncM2toV();
            Add(this->ubTmp, this->ubTmp, this->ubTmp2, cAligned);
            SyncVtoM2();
        }
        SyncVtoM3();
        CopyOutRow(dstGm, this->ubTmp, cGlobalOff, 0, 1, cCnt, 0, cAligned);
        SyncM3toV();
    }
}
