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
 * \file gru_grad_io_arch35.h
 * \brief 核内事件原语、GM<->UB 行搬运、门/梯度/hPrev 装载与预取乒乓的类外定义。
 */
#ifndef _GRU_GRAD_IO_ARCH35_
#define _GRU_GRAD_IO_ARCH35_

template <typename DTYPE>
__aicore__ inline void GruGradKernel<DTYPE>::SyncM2toV()
{
    event_t eventId = static_cast<event_t>(this->pipe.FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventId);
    WaitFlag<HardEvent::MTE2_V>(eventId);
};

template <typename DTYPE>
__aicore__ inline void GruGradKernel<DTYPE>::SyncVtoM3()
{
    event_t eventId = static_cast<event_t>(this->pipe.FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventId);
    WaitFlag<HardEvent::V_MTE3>(eventId);
};

template <typename DTYPE>
__aicore__ inline void GruGradKernel<DTYPE>::SyncVtoM2()
{
    event_t eventId = static_cast<event_t>(this->pipe.FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(eventId);
    WaitFlag<HardEvent::V_MTE2>(eventId);
};

template <typename DTYPE>
__aicore__ inline void GruGradKernel<DTYPE>::SyncM3toV()
{
    event_t eventId = static_cast<event_t>(this->pipe.FetchEventID(HardEvent::MTE3_V));
    SetFlag<HardEvent::MTE3_V>(eventId);
    WaitFlag<HardEvent::MTE3_V>(eventId);
}

template <typename DTYPE>
__aicore__ inline void GruGradKernel<DTYPE>::SyncM3toM2()
{
    event_t eventId = static_cast<event_t>(this->pipe.FetchEventID(HardEvent::MTE3_MTE2));
    SetFlag<HardEvent::MTE3_MTE2>(eventId);
    WaitFlag<HardEvent::MTE3_MTE2>(eventId);
};

template <typename DTYPE>
__aicore__ inline void GruGradKernel<DTYPE>::SyncM2toM3()
{
    event_t eventId = static_cast<event_t>(this->pipe.FetchEventID(HardEvent::MTE2_MTE3));
    SetFlag<HardEvent::MTE2_MTE3>(eventId);
    WaitFlag<HardEvent::MTE2_MTE3>(eventId);
};

template <typename DTYPE>
template <typename T>
__aicore__ inline void GruGradKernel<DTYPE>::CopyInRow(GlobalTensor<T>& gm, const LocalTensor<float>& ub,
                                                       int64_t gmOffset, int64_t rows, int64_t length, int64_t rowAlign,
                                                       int64_t gmStride)
{
    constexpr int64_t dSize = static_cast<int64_t>(sizeof(T));
    const int64_t blockLen = length * dSize;
    const int64_t pitchBytes = rowAlign * static_cast<int64_t>(sizeof(float)); // fp32 布局行距（字节）
    if constexpr (sizeof(T) == FP16_BYTES) {
        LocalTensor<float> ubW = ub; // 视图拷贝：ReinterpretCast 需要非 const 接收者
        // half 暂存区基址：跳过前 allocLength 个 fp32（= allocLength*4 字节，换算成 T 元素）——16KB buffer 后半
        auto tmpUb = ubW.template ReinterpretCast<T>()[allocLength * FLOAT_BYTES / dSize];
        if (blockLen % BLOCK_BYTES == 0) {
            // 快路径：块长 32B 对齐，UB 行距可精确表达（(行距-块长)/32 个 32B 块）
            DataCopyExtParams cp(static_cast<uint16_t>(rows), static_cast<uint32_t>(blockLen),
                                 static_cast<int64_t>(gmStride * dSize), (pitchBytes - blockLen) / BLOCK_BYTES, 0);
            DataCopyPadExtParams<T> pp(true, 0, 0, 0);
            DataCopyPad(tmpUb, gm[gmOffset], cp, pp);
        } else {
            // 尾块非对齐：逐行搬运（行距缺陷只在多行 + 非对齐时存在，逐行构造闭合）
            for (int64_t r = 0; r < rows; ++r) {
                DataCopyExtParams cp(1, static_cast<uint32_t>(blockLen), 0, 0, 0);
                DataCopyPadExtParams<T> pp(true, 0, 0, 0);
                DataCopyPad(tmpUb[r * (pitchBytes / dSize)], gm[gmOffset + r * (gmStride + length)], cp, pp);
            }
        }
        SyncM2toV();
        // 布局保持的位置逐元素 Cast：fp32 tile 行距 pitchBytes 与 half 暂存行距一致
        AscendC::Cast(ub, tmpUb, AscendC::RoundMode::CAST_NONE, static_cast<uint32_t>(rows * rowAlign));
        PipeBarrier<PIPE_V>();
    } else {
        if (blockLen % BLOCK_BYTES == 0) {
            DataCopyExtParams cp(static_cast<uint16_t>(rows), static_cast<uint32_t>(blockLen),
                                 static_cast<int64_t>(gmStride * dSize), (pitchBytes - blockLen) / BLOCK_BYTES, 0);
            DataCopyPadExtParams<T> pp(true, 0, 0, 0);
            DataCopyPad(ub, gm[gmOffset], cp, pp);
        } else {
            for (int64_t r = 0; r < rows; ++r) {
                DataCopyExtParams cp(1, static_cast<uint32_t>(blockLen), 0, 0, 0);
                DataCopyPadExtParams<T> pp(true, 0, 0, 0);
                DataCopyPad(ub[rowAlign * r], gm[gmOffset + r * (gmStride + length)], cp, pp);
            }
        }
    }
}

template <typename DTYPE>
template <typename T>
__aicore__ inline void GruGradKernel<DTYPE>::CopyOutRow(GlobalTensor<T>& gm, const LocalTensor<float>& ub,
                                                        int64_t gmOffset, int64_t rows, int64_t length,
                                                        int64_t gmStride, int64_t rowAlign, bool vSyncPreDone)
{
    constexpr int64_t dSize = static_cast<int64_t>(sizeof(T));
    const int64_t blockLen = length * dSize;
    const int64_t pitchBytes = rowAlign * static_cast<int64_t>(sizeof(float));
    if constexpr (sizeof(T) == FP16_BYTES) {
        LocalTensor<float> ubW = ub;
        auto tmpUb = ubW.template ReinterpretCast<T>();
        AscendC::Cast(tmpUb, ub, AscendC::RoundMode::CAST_RINT, static_cast<uint32_t>(rows * rowAlign));
        SyncVtoM3();
        const int64_t halfPitch = pitchBytes / dSize; // half 元素行距
        if (blockLen % BLOCK_BYTES == 0) {
            DataCopyExtParams cp(static_cast<uint16_t>(rows), static_cast<uint32_t>(blockLen),
                                 (pitchBytes - blockLen) / BLOCK_BYTES, static_cast<int64_t>(gmStride * dSize), 0);
            DataCopyPad(gm[gmOffset], tmpUb, cp);
        } else {
            for (int64_t r = 0; r < rows; ++r) {
                DataCopyExtParams cp(1, static_cast<uint32_t>(blockLen), 0, 0, 0);
                DataCopyPad(gm[gmOffset + r * (gmStride + length)], tmpUb[r * halfPitch], cp);
            }
        }
    } else {
        // 调用方已建立 V->MTE2->MTE3 传递序时（合并装配路径），内部 V->MTE3 同步为空转，可跳过
        if (!vSyncPreDone) {
            SyncVtoM3();
        }
        if (blockLen % BLOCK_BYTES == 0) {
            DataCopyExtParams cp(static_cast<uint16_t>(rows), static_cast<uint32_t>(blockLen),
                                 (pitchBytes - blockLen) / BLOCK_BYTES, static_cast<int64_t>(gmStride * dSize), 0);
            DataCopyPad(gm[gmOffset], ub, cp);
        } else {
            for (int64_t r = 0; r < rows; ++r) {
                DataCopyExtParams cp(1, static_cast<uint32_t>(blockLen), 0, 0, 0);
                DataCopyPad(gm[gmOffset + r * (gmStride + length)], ub[rowAlign * r], cp);
            }
        }
    }
}

template <typename DTYPE>
__aicore__ inline void GruGradKernel<DTYPE>::ReadHPrev(int64_t tIdx, int64_t mStart, int64_t bOff, int64_t bRows,
                                                       int64_t hOff, int64_t hLen, int64_t hAligned,
                                                       LocalTensor<float>& ub)
{
    this->vfMte2Dirty_ = true; // 本函数 MTE2 装载喂 VF/后续消费——scatter 需同步
                               // （Tail 侧调用为伪标记, 下步入口复位, 安全）
    const bool isReverse = (this->tiling->direction != 0);
    if (tIdx == 0) {
        CopyInRow(this->inputGm.initHGm, ub, (mStart + bOff) * this->H + hOff, bRows, hLen, hAligned, 0);
        return;
    }
    int64_t prevIdx = isReverse ? (this->T - tIdx) : (tIdx - 1);
    if (!this->tiling->isSeqLength) {
        CopyInRow(this->inputGm.outputHGm, ub, prevIdx * this->B * this->H + (mStart + bOff) * this->H + hOff, bRows,
                  hLen, hAligned, 0);
        return;
    }
    int64_t compactOutputRow = this->GetCompactRowOffset(prevIdx);
    int64_t bsPrev = this->inputGm.batchSizesGm.GetValue(prevIdx);
    int64_t rowStart = mStart + bOff; // 全局 batch 起点
    if (rowStart >= bsPrev) {
        CopyInRow(this->inputGm.initHGm, ub, rowStart * this->H + hOff, bRows, hLen, hAligned, 0);
    } else if (rowStart + bRows <= bsPrev) {
        CopyInRow(this->inputGm.outputHGm, ub, (compactOutputRow + rowStart) * this->H + hOff, bRows, hLen, hAligned,
                  0);
    } else {
        int64_t nActive = bsPrev - rowStart;
        int64_t nNew = bRows - nActive;
        CopyInRow(this->inputGm.outputHGm, ub, (compactOutputRow + rowStart) * this->H + hOff, nActive, hLen, hAligned,
                  0);
        auto ubNewPart = ub[nActive * hAligned];
        CopyInRow(this->inputGm.initHGm, ubNewPart, (rowStart + nActive) * this->H + hOff, nNew, hLen, hAligned, 0);
    }
}

template <typename DTYPE>
__aicore__ inline void GruGradKernel<DTYPE>::InitFastPathCache()
{
    if (this->fastPathInit_) {
        return;
    }
    this->hAlign8Ok_ = (this->H % this->alignPara) == 0;
    this->mergedShapeOk_ = this->hAlign8Ok_;
    const int64_t hAFull = ((this->H + this->alignMask) / this->alignPara) * this->alignPara;
    const int64_t vfVL = static_cast<int64_t>(AscendC::GetVecLen() / sizeof(float));
    this->scatterVOk_ = (hAFull % vfVL) == 0;
    this->stagingCapRows_ = this->allocLength / (GRU_GATE_SIZE * hAFull); // staging [rows, 3hA] 双面容量
    int64_t mStart = 0, mCnt = 0;
    if (this->GetCoreRows(mStart, mCnt) && mCnt > 0) {
        const int64_t hTiles = (this->H + this->vecHTile_ - 1) / this->vecHTile_;
        const int64_t bTiles = (mCnt + this->vecBTile_ - 1) / this->vecBTile_;
        const int64_t pfRows = (mCnt < this->vecBTile_) ? mCnt : this->vecBTile_;
        this->dhPrevUbPath_ = !this->tiling->isSeqLength && hTiles == 1 && bTiles == 1;
        this->pfCapOk_ = bTiles == 1 && pfRows * hAFull <= this->allocLength;
        // ubNDy 槽位仅 8KB——dy 预取超域时由 LoadGradH 回退 GM 直读
        this->pfDyCapOk_ = this->pfCapOk_ && pfRows * hAFull <= this->allocLength / DBUF_SLOTS; // 半池 8KB 上限
        // 带化分派——scatter 域之上加 T>=2（首步无 GEMM 前驱走全 H 路径）与 H%128==0
        // （带宽 hAb=H/2 需 %VL==0 且 >=VL）。预取命中由 pfCapOk_ && mCnt>0 && !seq && T>=2
        // 构造性保证（Tail(t) 对 t>0 恒预取 t-1）。
        this->bandPath_ = this->dhPrevUbPath_ && this->pfCapOk_ && this->mergedShapeOk_ && this->scatterVOk_ &&
                          this->T >= BAND_NUM && (this->H % BAND_H_ALIGN) == 0 && mCnt > 0 &&
                          mCnt <= this->stagingCapRows_; // T>=2: 首步无 GEMM 前驱走全 H 路径
    }
    this->fastPathInit_ = true;
}

template <typename DTYPE>
__aicore__ inline bool GruGradKernel<DTYPE>::DhPrevUbPath()
{
    if constexpr (sizeof(DTYPE) == FLOAT_BYTES) {
        return this->dhPrevUbPath_;
    } else {
        return false;
    }
}

template <typename DTYPE>
__aicore__ inline void GruGradKernel<DTYPE>::LoadGradH(int64_t tIdx, int64_t ghOff, int64_t batchHOff, int64_t bRows,
                                                       int64_t hLen, int64_t hAligned)
{
    if (tIdx != T - 1 && this->DhPrevUbPath()) {
        if (this->dyPrefetched) {
            this->dyPrefetched = false; // 单次消费（与 gatesPrefetched 同语义）
            Add(this->ubGradH, this->ubGradH, this->ubNDy, bRows * hAligned);
        } else {
            CopyInRow(this->inputGm.dyGm, this->ubTmp, ghOff, bRows, hLen, hAligned, 0);
            SyncM2toV();
            Add(this->ubGradH, this->ubGradH, this->ubTmp, bRows * hAligned);
        }
        return;
    }
    CopyInRow(this->inputGm.dyGm, this->ubGradH, ghOff, bRows, hLen, hAligned, 0);
    if (tIdx == T - 1) {
        CopyInRow(this->inputGm.dhGm, this->ubTmp, batchHOff, bRows, hLen, hAligned, 0);
    } else {
        CopyInRow(this->workGm.dhPrevWsGm, this->ubTmp, batchHOff, bRows, hLen, hAligned, 0);
    }
    SyncM2toV();
    Add(this->ubGradH, this->ubGradH, this->ubTmp, bRows * hAligned);
}

template <typename DTYPE>
__aicore__ inline void GruGradKernel<DTYPE>::LoadGatesAndHPrev(int64_t tIdx, int64_t ghOff, int64_t mStart,
                                                               int64_t bOff, int64_t bRows, int64_t hOff, int64_t hLen,
                                                               int64_t hAligned)
{
    if (this->gatesPrefetched && !this->tiling->isSeqLength && this->H <= this->vecHTile_ &&
        (this->H % this->alignPara) == 0) {
        // 视图乒乓——计算直读预取缓冲, 免 4 面 UB->UB 中转 DataCopy + 同步。
        // 条件: 单 h/b tile 且 H 对齐(预取 rowAlign=H 与计算 hAligned 间距一致);
        // 不满足回退原中转路径。乒乓: 交换后 ubN* 指向旧计算缓冲, 供下步预取写入。
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
        this->gatesPrefetched = false; // 单次消费语义保持
    } else if (this->gatesPrefetched && !this->tiling->isSeqLength) {
        // 消费预取: UB->UB 拷贝(MTE2 发射, 低于 GM 访存开销), r/z/n/hn 四面
        // MTE2 喂 VF——标记脏（scatter 处同步）
        this->vfMte2Dirty_ = true;
        const int64_t cpCount = bRows * hAligned;
        if ((cpCount * FLOAT_BYTES) % BLOCK_BYTES == 0) {
            DataCopyParams dp(1, static_cast<uint16_t>(cpCount * FLOAT_BYTES / BLOCK_BYTES), 0, 0);
            DataCopy(this->ubUpdate, this->ubNUpdate, dp);
            DataCopy(this->ubNew, this->ubNNew, dp);
            DataCopy(this->ubReset, this->ubNReset, dp);
            DataCopy(this->ubHN, this->ubNHN, dp);
        } else {
            for (int64_t i = 0; i < cpCount; ++i) {
                this->ubUpdate.SetValue(i, this->ubNUpdate.GetValue(i));
                this->ubNew.SetValue(i, this->ubNNew.GetValue(i));
                this->ubReset.SetValue(i, this->ubNReset.GetValue(i));
                this->ubHN.SetValue(i, this->ubNHN.GetValue(i));
            }
        }
        this->gatesPrefetched = false; // 单次消费
    } else {
        // GM->UB 门装载 MTE2 喂 VF——标记脏（scatter 处同步）
        this->vfMte2Dirty_ = true;
        CopyInRow(this->inputGm.updateGateGm, this->ubUpdate, ghOff, bRows, hLen, hAligned, 0);
        CopyInRow(this->inputGm.newGateGm, this->ubNew, ghOff, bRows, hLen, hAligned, 0);
        CopyInRow(this->inputGm.resetGateGm, this->ubReset, ghOff, bRows, hLen, hAligned, 0);
        CopyInRow(this->inputGm.hNGm, this->ubHN, ghOff, bRows, hLen, hAligned, 0);
    }
    // hp 预取消费——乒乓换入（与门控视图乒乓同构）。换出的旧 ubHPrev 缓冲
    // （持上步 dhFromH，Accumulate 已消费）成为 ubGate，供 Tail 预取下一拍 hp。
    // 预取仅在合并路径域发射（ubGate 的 1.0 常量仅逐门遗留路径消费）。
    if (this->hpPrefetched && !this->tiling->isSeqLength && this->H <= this->vecHTile_ &&
        (this->H % this->alignPara) == 0) {
        LocalTensor<float> tv = this->ubHPrev;
        this->ubHPrev = this->ubGate;
        this->ubGate = tv;
        this->hpPrefetched = false; // 单次消费语义保持
    } else {
        this->ReadHPrev(tIdx, mStart, bOff, bRows, hOff, hLen, hAligned, this->ubHPrev);
    }
}

#endif // _GRU_GRAD_IO_ARCH35_
