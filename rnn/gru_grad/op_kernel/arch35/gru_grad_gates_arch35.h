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
 * \file gru_grad_gates_arch35.h
 * \brief 逐门梯度计算与时间逆向逐步处理的类外定义。
 */
#ifndef _GRU_GRAD_GATES_ARCH35_
#define _GRU_GRAD_GATES_ARCH35_

template <typename DTYPE>
__aicore__ inline void GruGradKernel<DTYPE>::ComputeDINew(int64_t gmOffGi, int64_t bRows, int64_t hLen,
                                                          int64_t gmGateStride, int64_t hAligned)
{
    SyncM2toV();
    Duplicate(this->ubGate, 1.0f, bRows * hAligned);
    Sub(this->ubTmp, this->ubGate, this->ubUpdate, bRows * hAligned);
    Mul(this->ubTmp2, this->ubGradH, this->ubTmp, bRows * hAligned);
    Mul(this->ubTmp, this->ubNew, this->ubNew, bRows * hAligned);
    Sub(this->ubTmp, this->ubGate, this->ubTmp, bRows * hAligned);
    Mul(this->ubDHPrevFromH, this->ubTmp2, this->ubTmp, bRows * hAligned);
    SyncVtoM3();
    CopyOutRow(this->workGm.dGiGm, this->ubDHPrevFromH, gmOffGi + GATE_IDX_NEW * H, bRows, hLen,
               this->H * GRU_GATE_SIZE - hLen, hAligned);
    SyncM3toV();
}

template <typename DTYPE>
__aicore__ inline void GruGradKernel<DTYPE>::ComputeDReset(int64_t gmOffGi, int64_t bRows, int64_t hLen,
                                                           int64_t gmGateStride, int64_t hAligned)
{
    Mul(this->ubTmp, this->ubDHPrevFromH, this->ubHN, bRows * hAligned);
    Mul(this->ubTmp2, this->ubTmp, this->ubReset, bRows * hAligned);
    Sub(this->ubTmp, this->ubGate, this->ubReset, bRows * hAligned);
    Mul(this->ubTmp3, this->ubTmp2, this->ubTmp, bRows * hAligned);
    SyncVtoM3();
    CopyOutRow(this->workGm.dGiGm, this->ubTmp3, gmOffGi, bRows, hLen, this->H * GRU_GATE_SIZE - hLen, hAligned);
    CopyOutRow(this->workGm.dGhGm, this->ubTmp3, gmOffGi, bRows, hLen, this->H * GRU_GATE_SIZE - hLen, hAligned);
    SyncM3toV();
}

template <typename DTYPE>
__aicore__ inline void GruGradKernel<DTYPE>::ComputeDUpdate(int64_t gmOffGi, int64_t bRows, int64_t hLen,
                                                            int64_t gmGateStride, int64_t hAligned)
{
    Sub(this->ubTmp, this->ubHPrev, this->ubNew, bRows * hAligned);
    Mul(this->ubTmp2, this->ubGradH, this->ubTmp, bRows * hAligned);
    Mul(this->ubTmp, this->ubTmp2, this->ubUpdate, bRows * hAligned);
    Sub(this->ubTmp2, this->ubGate, this->ubUpdate, bRows * hAligned);
    Mul(this->ubTmp3, this->ubTmp, this->ubTmp2, bRows * hAligned);
    SyncVtoM3();
    CopyOutRow(this->workGm.dGiGm, this->ubTmp3, gmOffGi + H, bRows, hLen, this->H * GRU_GATE_SIZE - hLen, hAligned);
    CopyOutRow(this->workGm.dGhGm, this->ubTmp3, gmOffGi + H, bRows, hLen, this->H * GRU_GATE_SIZE - hLen, hAligned);
    SyncM3toV();
}

template <typename DTYPE>
__aicore__ inline void GruGradKernel<DTYPE>::ComputeDGhNewAndDhFromH(int64_t gmOffGi, int64_t batchHOff, int64_t bRows,
                                                                     int64_t hLen, int64_t gmGateStride,
                                                                     int64_t hAligned)
{
    Mul(this->ubTmp3, this->ubDHPrevFromH, this->ubReset, bRows * hAligned);
    SyncVtoM3();
    CopyOutRow(this->workGm.dGhGm, this->ubTmp3, gmOffGi + GATE_IDX_NEW * H, bRows, hLen, gmGateStride, hAligned);
    SyncM3toV();
    Mul(this->ubDHPrevFromH, this->ubGradH, this->ubUpdate, bRows * hAligned);
    SyncVtoM3();
    CopyOutRow(this->workGm.dhFromHGm, this->ubDHPrevFromH, batchHOff, bRows, hLen, 0, hAligned);
    SyncM3toV();
}

template <typename DTYPE>
__aicore__ inline void GruGradKernel<DTYPE>::ProcessVectorHTile(int64_t tIdx, int64_t gateIdx, int64_t mStart,
                                                                int64_t bOff, int64_t bRows, int64_t ht,
                                                                int64_t compactGateRow, int64_t gm3H,
                                                                int64_t gmGateStride, int64_t hTiles)
{
    int64_t hLen = this->vecHTile_;
    int64_t hOff = ht * this->vecHTile_;
    if (ht == hTiles - 1) {
        hLen = H - hOff;
    }
    int64_t hAligned = ((hLen + this->alignMask) / this->alignPara) * this->alignPara; // dtype 感知行距
    int64_t blkAligned = bRows * hAligned;
    // 复位脏标记——Accumulate 的 SyncM2toV（上迭代末）已覆盖此前全部 MTE2（含
    // 上步 Tail 预取，MTE2 管内序）；此后仅 LoadGates/ReadHPrev 的 MTE2 分支重标脏。
    this->vfMte2Dirty_ = false;

    // ⚠ hTiles>=2（H>4096）时的跨 tile WAR：本 tile 的 MTE2 装载（写 ubGradH/ubTmp/ubUpdate 等）
    // 与上一 tile 仍在 V 管线排队的计算（读/写同名 UB）无序——V 标志链只护 MTE3 不护 MTE2。
    // （H4097/4104 实测）：上一 tile 的 V 写晚到覆盖本 tile 的 dh 装载（ubTmp 写写冲突）
    // → tile-1 全域输出错；上一 tile 的 V 读被本 tile 装载提前覆盖 → tile-0 区 h=0..7 散点错。
    // 单 tile（H<=4096）不跨 tile 故从未触发；步进循环有 SyncAll 全栅栏护住。
    SyncVtoM2(); // 恢复无条件（单 tile 跳过依赖的传递序推理不可靠）

    int64_t ghOff = 0, gmOffGi = 0;
    if (this->tiling->isSeqLength) {
        ghOff = (compactGateRow + mStart + bOff) * H + hOff;
        gmOffGi = (compactGateRow + mStart + bOff) * gm3H + hOff;
    } else {
        ghOff = gateIdx * B * H + (mStart + bOff) * H + hOff;
        gmOffGi = gateIdx * B * gm3H + (mStart + bOff) * gm3H + hOff;
    }
    int64_t batchHOff = (mStart + bOff) * H + hOff;

    this->LoadGradH(tIdx, ghOff, batchHOff, bRows, hLen, hAligned);
    this->LoadGatesAndHPrev(tIdx, ghOff, mStart, bOff, bRows, hOff, hLen, hAligned);
    const bool mergedGrads = this->GateGradsMergedOk(bRows, hLen, hAligned);
    this->dhFromHInHp_ = mergedGrads; // dhFromH 常驻缓冲按路径记录（供 Accumulate 快路径）
    if (this->GateGradsScatterOk(bRows, hLen, hAligned)) {
        // VF scatter 直写 staging（hA%VL==0 整拍域）——装配 MTE2 与双同步出链
        this->ComputeGateGradsScatter(gmOffGi, batchHOff, bRows, hLen, hAligned);
    } else if (mergedGrads) {
        // 合并路径——单 tile hOff==0，gmOffGi 即整行 [0,3H) 起点
        this->ComputeGateGradsMerged(gmOffGi, batchHOff, bRows, hLen, hAligned);
    } else {
        this->ComputeDINew(gmOffGi, bRows, hLen, gmGateStride, hAligned);
        this->ComputeDReset(gmOffGi, bRows, hLen, gmGateStride, hAligned);
        this->ComputeDUpdate(gmOffGi, bRows, hLen, gmGateStride, hAligned);
        this->ComputeDGhNewAndDhFromH(gmOffGi, batchHOff, bRows, hLen, gmGateStride, hAligned);
    }
}

template <typename DTYPE>
__aicore__ inline void GruGradKernel<DTYPE>::StoreHPrevHTile(int64_t tIdx, int64_t gateIdx, int64_t mStart,
                                                             int64_t bOff, int64_t bRows, int64_t ht,
                                                             int64_t compactGateRow, int64_t hTiles, bool hpInUb)
{
    int64_t hLen = this->vecHTile_;
    int64_t hOff = ht * this->vecHTile_;
    if (ht == hTiles - 1) {
        hLen = H - hOff;
    }
    int64_t hAligned = ((hLen + this->alignMask) / this->alignPara) * this->alignPara;

    // hpInUb 分支无 V 写/无 MTE2（仅 MTE3 读 V 写过的 ubDHPrevFromH，其可见性由
    // scatter 内 SyncVtoM3 已建立；MTE3 同管读读内序）——顶部 M3→V 防护仅为非 hpInUb
    // 分支的 Duplicate(V 写 ubTmp2) 所需，移入该分支。
    int64_t hPrevWsOff = 0;
    if (this->tiling->isSeqLength) {
        hPrevWsOff = (compactGateRow + mStart + bOff) * H + hOff;
    } else {
        hPrevWsOff = (gateIdx * B + mStart + bOff) * H + hOff;
    }
    if (hpInUb) {
        // hp 已由 VF scatter 保留至 ubDHPrevFromH（本步 scatter 单 tile 域）——
        // 免 ReadHPrev GM 重载。V→M3 序：scatter 内 SyncVtoM3 已建立（其后 MTE3 同序）。
        CopyOutRow(this->workGm.hPrevWsGm, this->ubDHPrevFromH, hPrevWsOff, bRows, hLen, 0, hAligned);
    } else {
        // 跨 tile WAR：本 tile Duplicate(V 写 ubTmp2) 需等上一 tile CopyOutRow(MTE3 读 ubTmp2) 排空
        SyncM3toV();
        Duplicate(this->ubTmp2, 0.0f, bRows * hAligned);
        SyncVtoM2();
        this->ReadHPrev(tIdx, mStart, bOff, bRows, hOff, hLen, hAligned, this->ubTmp2);
        SyncM2toM3();
        CopyOutRow(this->workGm.hPrevWsGm, this->ubTmp2, hPrevWsOff, bRows, hLen, 0, hAligned);
    }
    SyncM3toM2(); // 下一 hTile 的 ReadHPrev（MTE2 写 ubTmp2）与本写（MTE3 读）防同型 WAR
}

template <typename DTYPE>
__aicore__ inline void GruGradKernel<DTYPE>::ProcessVector(int64_t tIdx)
{
    const int64_t bTile = this->vecBTile_, hTile = this->vecHTile_;
    const bool isReverse = (this->tiling->direction != 0);
    const int64_t gateIdx = isReverse ? (this->T - 1 - tIdx) : tIdx;
    const int64_t gm3H = this->H * GRU_GATE_SIZE;
    const int64_t gmGateStride = this->H * GATE_IDX_NEW; // 弃用: 消费点已改 3H-hLen(GAP 正确值)

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
    }
}

template <typename DTYPE>
__aicore__ inline void GruGradKernel<DTYPE>::ProcessVectorTail(int64_t tIdx)
{
    const int64_t bTile = this->vecBTile_, hTile = this->vecHTile_;
    const bool isReverse = (this->tiling->direction != 0);
    const int64_t gateIdx = isReverse ? (this->T - 1 - tIdx) : tIdx;

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
    {
        int64_t bTiles = (mCnt + bTile - 1) / bTile;
        int64_t hTiles = (H + hTile - 1) / hTile;
        // hp 保留消费仅限单 tile 域——多 bTile 时 VF 保留面被逐 tile 覆写,
        // 仅末 tile 数据有效; 回退 ReadHPrev GM 重载
        const bool hpInUb = this->hpPreserved_ && bTiles == 1 && hTiles == 1;
        // 单次消费即复位（与 gatesPrefetched 同语义）：变长模式下 mCnt 逐步变化，
        // scatter 适配可能跨 stagingCapRows_ 翻转——翻转后的逐门路径已把 ubDHPrevFromH
        // 覆写为 dhFromH，历史标志若不清零会误将 dhFromH 当 hp 直写 hPrevWs
        this->hpPreserved_ = false;
        for (int64_t bt = 0; bt < bTiles; bt++) {
            int64_t bRows = bTile;
            int64_t bOff = bt * bTile;
            if (bt == bTiles - 1) {
                bRows = mCnt - bOff;
            }
            for (int64_t ht = 0; ht < hTiles; ht++) {
                this->StoreHPrevHTile(tIdx, gateIdx, mStart, bOff, bRows, ht, compactGateRow, hTiles, hpInUb);
            }
        }
        // hp DCCI(自 DcciStepSlices 移入窗口: hPrevWs 的跨簇读者仅终轮转置, SyncAll+V2C#2 兜底,
        // 逐步刷新仅为写缓冲落地; 置于本写之后语义不变)
        {
            GlobalTensor<DTYPE> hp;
            if (this->tiling->isSeqLength) {
                hp.SetGlobalBuffer(this->workGm.hPrevWsGm.GetPhyAddr((compactGateRow + mStart) * this->H),
                                   mCnt * this->H);
            } else {
                hp.SetGlobalBuffer(this->workGm.hPrevWsGm.GetPhyAddr((gateIdx * this->B + mStart) * this->H),
                                   mCnt * this->H);
            }
            DataCacheCleanAndInvalid<DTYPE, AscendC::CacheLine::ENTIRE_DATA_CACHE>(hp);
        }
    }
    // 门平面预取: 下一拍(tIdx-1)的 r/z/n/hn/dy 的 MTE2 装载(乒乓目标 = 非别名缓冲)
    if (tIdx > 0) {
        const int64_t nextGateIdx = isReverse ? (this->T - tIdx) : (tIdx - 1);
        int64_t nextGhOff = 0;
        if (this->tiling->isSeqLength) {
            // packed 模式预取从简(逐 timestep 偏移不同, 预取意义有限) —— 不预取
        } else {
            nextGhOff = nextGateIdx * B * H;
        }
        if (!this->tiling->isSeqLength) {
            // ⚠ 预取仅限单 bTile 且整行落 UB 预算:
            //   1) pfBTiles>=2 时各 tile 重复写 ubN*[0] 互踩——消费端(LoadGatesAndHPrev/LoadGradH)
            //      恒从缓冲头读 bRows 行, 后一 tile 覆盖前一 tile 首行 → 本 AIV 首行门控错
            //      (H);
            //   2) bRows*H 超 allocLength 时 MTE2 越界写 UB(原 预算门)。
            // 容量/单tile判定缓存; 7 面装载共享一份 cp（逐步恒定, 原 7×15-20
            // 条标量装配 → 1×; fp32 + 块长 32B 对齐域, 与 CopyInRow 快路径逐参一致）。
            const int64_t hAlignedFull = ((this->H + this->alignMask) / this->alignPara) * this->alignPara;
            if (this->pfCapOk_) {
                const int64_t pitchBytes = hAlignedFull * static_cast<int64_t>(sizeof(float));
                if constexpr (sizeof(DTYPE) == FLOAT_BYTES) {
                    // fp32 共参快路径（块长 32B 对齐前提与 CopyInRow 内部一致）
                    if ((this->H * FLOAT_BYTES) % BLOCK_BYTES == 0) {
                        DataCopyExtParams cp(static_cast<uint16_t>(mCnt), static_cast<uint32_t>(this->H * FLOAT_BYTES),
                                             0, (pitchBytes - this->H * FLOAT_BYTES) / BLOCK_BYTES, 0);
                        DataCopyPadExtParams<DTYPE> pp(true, 0, 0, 0);
                        DataCopyPad(this->ubNUpdate, this->inputGm.updateGateGm[nextGhOff + mStart * H], cp, pp);
                        DataCopyPad(this->ubNNew, this->inputGm.newGateGm[nextGhOff + mStart * H], cp, pp);
                        DataCopyPad(this->ubNReset, this->inputGm.resetGateGm[nextGhOff + mStart * H], cp, pp);
                        DataCopyPad(this->ubNHN, this->inputGm.hNGm[nextGhOff + mStart * H], cp, pp);
                        if (this->pfDyCapOk_) { // ubNDy 槽位 8KB，超域不预取（GM 回退）
                            DataCopyPad(this->ubNDy, this->inputGm.dyGm[nextGhOff + mStart * H], cp, pp);
                            this->dyPrefetched = true;
                        }
                        this->gatesPrefetched = true;
                        if (this->hAlign8Ok_) {
                            const int64_t nextT = tIdx - 1;
                            if (nextT == 0) {
                                DataCopyPad(this->ubGate, this->inputGm.initHGm[mStart * this->H], cp, pp);
                            } else {
                                const int64_t prevIdx = isReverse ? (this->T - nextT) : (nextT - 1);
                                DataCopyPad(this->ubGate,
                                            this->inputGm.outputHGm[prevIdx * this->B * this->H + mStart * this->H], cp,
                                            pp);
                            }
                            this->hpPrefetched = true;
                        }
                    } else {
                        this->PrefetchFallback(tIdx, isReverse, nextGhOff, mStart, mCnt, hAlignedFull);
                    }
                } else {
                    this->PrefetchFallback(tIdx, isReverse, nextGhOff, mStart, mCnt, hAlignedFull);
                }
            }
        }
    }
}

template <typename DTYPE>
__aicore__ inline void GruGradKernel<DTYPE>::PrefetchFallback(int64_t tIdx, bool isReverse, int64_t nextGhOff,
                                                              int64_t mStart, int64_t mCnt, int64_t hAlignedFull)
{
    CopyInRow(this->inputGm.updateGateGm, this->ubNUpdate, nextGhOff + mStart * H, mCnt, this->H, hAlignedFull, 0);
    CopyInRow(this->inputGm.newGateGm, this->ubNNew, nextGhOff + mStart * H, mCnt, this->H, hAlignedFull, 0);
    CopyInRow(this->inputGm.resetGateGm, this->ubNReset, nextGhOff + mStart * H, mCnt, this->H, hAlignedFull, 0);
    CopyInRow(this->inputGm.hNGm, this->ubNHN, nextGhOff + mStart * H, mCnt, this->H, hAlignedFull, 0);
    // dy 同窗预取(下一拍 LoadGradH UB 直取)；8KB 槽位容量门
    if (this->pfDyCapOk_) {
        CopyInRow(this->inputGm.dyGm, this->ubNDy, nextGhOff + mStart * H, mCnt, this->H, hAlignedFull, 0);
        this->dyPrefetched = true;
    }
    this->gatesPrefetched = true;
    // hp(t-1) 同窗预取至 ubGate（合并路径域——ubGate 的 1.0 常量仅逐门
    // 遗留路径消费, 其 ComputeDINew 入口自重填）。hp 源与 ReadHPrev 同构:
    // nextT==0 → initHGm, 否则 outputHGm[prevIdx(nextT)]。
    if (this->hAlign8Ok_) {
        const int64_t nextT = tIdx - 1;
        if (nextT == 0) {
            CopyInRow(this->inputGm.initHGm, this->ubGate, mStart * this->H, mCnt, this->H, hAlignedFull, 0);
        } else {
            const int64_t prevIdx = isReverse ? (this->T - nextT) : (nextT - 1);
            CopyInRow(this->inputGm.outputHGm, this->ubGate, prevIdx * this->B * this->H + mStart * this->H, mCnt,
                      this->H, hAlignedFull, 0);
        }
        this->hpPrefetched = true;
    }
}

template <typename DTYPE>
__aicore__ inline void GruGradKernel<DTYPE>::ConsumePrefetchSwaps()
{
    LocalTensor<float> tv = this->ubUpdate;
    this->ubUpdate = this->ubNUpdate;
    this->ubNUpdate = tv;
    tv = this->ubNew;
    this->ubNew = this->ubNNew;
    this->ubNNew = tv;
    tv = this->ubReset;
    this->ubReset = this->ubNReset;
    this->ubNReset = tv;
    tv = this->ubHN;
    this->ubHN = this->ubNHN;
    this->ubNHN = tv;
    this->gatesPrefetched = false;
    LocalTensor<float> th = this->ubHPrev;
    this->ubHPrev = this->ubGate;
    this->ubGate = th;
    this->hpPrefetched = false;
}

#endif // _GRU_GRAD_GATES_ARCH35_
