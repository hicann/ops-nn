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
 * \file gru_grad_band_arch35.h
 * \brief 带化主循环、门梯度合并/散写直排、dhPrev 物化的类外定义。
 */
#ifndef _GRU_GRAD_BAND_ARCH35_
#define _GRU_GRAD_BAND_ARCH35_

template <typename DTYPE>
__aicore__ inline void GruGradKernel<DTYPE>::VecProcessBand()
{
    this->dhFromHInHp_ = true;
    const int64_t mStart = this->mStart_, mCnt = this->mCnt_;
    const int64_t hA = ((this->H + this->alignMask) / this->alignPara) * this->alignPara;
    const int64_t hAb = this->H / BAND_NUM; // 带宽 = H 二分
    const uint32_t vfVL = static_cast<uint32_t>(AscendC::GetVecLen() / sizeof(float));
    // 偏置输出恒走 ProcessBiasReduce GM 归约：原 biasInline UB 部分和全宽直写
    // 在多 AIV 有行时构成多写竞争且缺他核行贡献（db=全 (T,B) 行求和），已弃用。
    // 全宽 M3 的 cp 跨迭代恒定——外提（逐步免 ~10 条标量装配）
    DataCopyExtParams cpBand(static_cast<uint16_t>(mCnt), static_cast<uint32_t>(GRU_GATE_SIZE * this->H * FLOAT_BYTES),
                             0, 0, 0); // [rows,3H] 全宽
    // 首步: 现行全 H 路径（dy+dh 首步装载 + 非预取门装载 + scatter + M3）
    this->ProcessVector(this->T - 1);
    this->DcciStepSlices(this->T - 1);
    if (this->s1Usable_) {
        // 带化主循环——B0/B1/C2V 信号替换为 SyncAll<false>×2 + 规约。
        // 迭代序: #2(t+1) → Reduce(t+1)[dhPrevF32 物化] → 带VF/M3/Dcci → #1(t) → Tail(t)。
        AscendC::SyncAll<false>();
        this->ProcessVectorTail(this->T - 1);
        for (int64_t tIdx = this->T - 2; tIdx >= 0;
             tIdx--) { // 自次末步 T-2 起倒推（首步 T-1 无 GEMM 前驱，已单独处理）
            const bool isReverse = (this->tiling->direction != 0);
            const int64_t gateIdx = isReverse ? (this->T - 1 - tIdx) : tIdx;
            const int64_t gmOffGi = (gateIdx * this->B + mStart) * this->H * GRU_GATE_SIZE;
            AscendC::SyncAll<false>(); // #2(t+1): 部分和全簇就绪
            this->ReduceS1();
            if (mCnt > 0) {
                this->ConsumePrefetchSwaps();
                CopyInRow(this->dhPrevF32Gm, this->ubTmp3, mStart * this->H, mCnt, hAb, hAb, this->H - hAb);
                SyncM2toV();
                asc_vf_call<GateGradsBandVF<float>>(
                    (__ubuf__ float*)this->ubTmp.GetPhyAddr(), (__ubuf__ float*)this->ubTmp2.GetPhyAddr(),
                    (__ubuf__ float*)this->ubHPrev.GetPhyAddr(), (__ubuf__ float*)this->ubDHPrevFromH.GetPhyAddr(),
                    (__ubuf__ float*)this->ubTmp3.GetPhyAddr(), (__ubuf__ float*)this->ubNDy.GetPhyAddr(),
                    (__ubuf__ float*)this->ubGate.GetPhyAddr(), (__ubuf__ float*)this->ubUpdate.GetPhyAddr(),
                    (__ubuf__ float*)this->ubNew.GetPhyAddr(), (__ubuf__ float*)this->ubHPrev.GetPhyAddr(),
                    (__ubuf__ float*)this->ubHN.GetPhyAddr(), (__ubuf__ float*)this->ubReset.GetPhyAddr(),
                    static_cast<uint32_t>(mCnt), static_cast<uint32_t>(hA), static_cast<uint32_t>(hAb), 0, vfVL);
                SyncVtoM2();
                CopyInRow(this->dhPrevF32Gm, this->ubTmp3, mStart * this->H + hAb, mCnt, hAb, hAb, this->H - hAb);
                SyncM2toV();
                asc_vf_call<GateGradsBandVF<float>>(
                    (__ubuf__ float*)this->ubTmp.GetPhyAddr(), (__ubuf__ float*)this->ubTmp2.GetPhyAddr(),
                    (__ubuf__ float*)this->ubHPrev.GetPhyAddr(), (__ubuf__ float*)this->ubDHPrevFromH.GetPhyAddr(),
                    (__ubuf__ float*)this->ubTmp3.GetPhyAddr(), (__ubuf__ float*)this->ubNDy.GetPhyAddr(),
                    (__ubuf__ float*)this->ubGate.GetPhyAddr(), (__ubuf__ float*)this->ubUpdate.GetPhyAddr(),
                    (__ubuf__ float*)this->ubNew.GetPhyAddr(), (__ubuf__ float*)this->ubHPrev.GetPhyAddr(),
                    (__ubuf__ float*)this->ubHN.GetPhyAddr(), (__ubuf__ float*)this->ubReset.GetPhyAddr(),
                    static_cast<uint32_t>(mCnt), static_cast<uint32_t>(hA), static_cast<uint32_t>(hAb),
                    static_cast<uint32_t>(hAb), vfVL);
                this->hpPreserved_ = true;
                SyncVtoM3();
                if constexpr (sizeof(DTYPE) == FLOAT_BYTES) {
                    DataCopyPad(this->workGm.dGiGm[gmOffGi], this->ubTmp, cpBand);
                    DataCopyPad(this->workGm.dGhGm[gmOffGi], this->ubTmp2, cpBand);
                }
                SyncM3toV();
            }
            this->DcciStepSlices(tIdx);
            AscendC::SyncAll<false>(); // #1(t): dGh(t) 全簇就绪 → AIC 切片 GEMM
            this->ProcessVectorTail(tIdx);
        }
        AscendC::SyncAll<false>(); // #2(0)
        this->ReduceS1();
        this->AccumulateDhPrev(0);
        this->VecProcessFinalize();
        return;
    }
    ggsync::VecSignalCube();
    this->ProcessVectorTail(this->T - 1);
    for (int64_t tIdx = this->T - 2; tIdx >= 0; tIdx--) { // 自次末步 T-2 起倒推（首步 T-1 无 GEMM 前驱，已单独处理）
        const bool isReverse = (this->tiling->direction != 0);
        const int64_t gateIdx = isReverse ? (this->T - 1 - tIdx) : tIdx;
        const int64_t gmOffGi = (gateIdx * this->B + mStart) * this->H * GRU_GATE_SIZE;
        ggsync::VecWaitCubeBand(); // B0: GEMM(t+1) 带0 就绪
        if (mCnt > 0) {
            this->ConsumePrefetchSwaps();
            // 带 0: gemmRes M2 + 折叠 VF（与 AIC 带1 GEMM 重叠窗口）
            CopyInRow(this->dhPrevF32Gm, this->ubTmp3, mStart * this->H, mCnt, hAb, hAb, this->H - hAb);
            SyncM2toV();
            asc_vf_call<GateGradsBandVF<float>>(
                (__ubuf__ float*)this->ubTmp.GetPhyAddr(),         // dGi staging 全宽
                (__ubuf__ float*)this->ubTmp2.GetPhyAddr(),        // dGh staging 全宽
                (__ubuf__ float*)this->ubHPrev.GetPhyAddr(),       // dhFromH(t) 带写出（就地）
                (__ubuf__ float*)this->ubDHPrevFromH.GetPhyAddr(), // hp 保留带写
                (__ubuf__ float*)this->ubTmp3.GetPhyAddr(),        // gemmRes 带紧凑
                (__ubuf__ float*)this->ubNDy.GetPhyAddr(),         // dy(t) 带
                (__ubuf__ float*)this->ubGate.GetPhyAddr(),        // dhFromH(t+1) 带（换出缓冲）
                (__ubuf__ float*)this->ubUpdate.GetPhyAddr(),      // z
                (__ubuf__ float*)this->ubNew.GetPhyAddr(),         // n
                (__ubuf__ float*)this->ubHPrev.GetPhyAddr(),       // hp(t)
                (__ubuf__ float*)this->ubHN.GetPhyAddr(),          // hn
                (__ubuf__ float*)this->ubReset.GetPhyAddr(),       // r
                static_cast<uint32_t>(mCnt), static_cast<uint32_t>(hA), static_cast<uint32_t>(hAb), 0, vfVL);
            SyncVtoM2(); // 带0 VF 的 ubTmp3 读 → 带1 M2 写 WAR（V→M2 定向）
        }
        ggsync::VecWaitCubeBand1(); // B1——末带 Fixpipe 落地（早于 C2V）
        if (mCnt > 0) {
            // 带 1 gemmRes M2 预发射：装载延迟藏进 B1→C2V 间隔（band1 关键段缩短）
            CopyInRow(this->dhPrevF32Gm, this->ubTmp3, mStart * this->H + hAb, mCnt, hAb, hAb, this->H - hAb);
        }
        ggsync::VecWaitCube(); // C2V: GEMM(t+1) 全就绪
        if (mCnt > 0) {
            // 带 1: 折叠 VF + 全宽 M3（staging 两带列区均已写齐）
            SyncM2toV();
            asc_vf_call<GateGradsBandVF<float>>(
                (__ubuf__ float*)this->ubTmp.GetPhyAddr(), (__ubuf__ float*)this->ubTmp2.GetPhyAddr(),
                (__ubuf__ float*)this->ubHPrev.GetPhyAddr(), (__ubuf__ float*)this->ubDHPrevFromH.GetPhyAddr(),
                (__ubuf__ float*)this->ubTmp3.GetPhyAddr(), (__ubuf__ float*)this->ubNDy.GetPhyAddr(),
                (__ubuf__ float*)this->ubGate.GetPhyAddr(), (__ubuf__ float*)this->ubUpdate.GetPhyAddr(),
                (__ubuf__ float*)this->ubNew.GetPhyAddr(), (__ubuf__ float*)this->ubHPrev.GetPhyAddr(),
                (__ubuf__ float*)this->ubHN.GetPhyAddr(), (__ubuf__ float*)this->ubReset.GetPhyAddr(),
                static_cast<uint32_t>(mCnt), static_cast<uint32_t>(hA), static_cast<uint32_t>(hAb),
                static_cast<uint32_t>(hAb), vfVL);
            this->hpPreserved_ = true;
            SyncVtoM3();
            if constexpr (sizeof(DTYPE) == FLOAT_BYTES) {
                // 全宽 staging 单 M3（H%8==0 ⟹ hA==H，块长=3H 连续）
                DataCopyPad(this->workGm.dGiGm[gmOffGi], this->ubTmp, cpBand);
                DataCopyPad(this->workGm.dGhGm[gmOffGi], this->ubTmp2, cpBand);
            }
            SyncM3toV();
        }
        // bias 累加——从 staging 直积到 UB 累加器(ubGate 尾部别名)。
        // 位于 SyncM3toV(MTE3 已读完 staging)与 V2C(S 管不等 V 管)之间,
        // V 管空闲窗口内零关键路径; 管内序保证 Add 在 scatter 后执行。
        this->DcciStepSlices(tIdx);
        ggsync::VecSignalCube();
        this->ProcessVectorTail(tIdx);
    }
    // 尾步 epilogue: 消费 GEMM(0) 的 B0/B1/C2V（flag 平衡）+ dhPrev(0) 物化（现行 Accumulate 复用，
    // 读 ubHPrev 的 dhFromH(0) + dhPrevF32(0) 全量 M2）
    ggsync::VecWaitCubeBand();
    ggsync::VecWaitCubeBand1();
    ggsync::VecWaitCube();
    this->AccumulateDhPrev(0);
    this->VecProcessFinalize();
}

template <typename DTYPE>
__aicore__ inline void GruGradKernel<DTYPE>::InitDhPrev()
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
            int64_t hAligned = ((hLen + this->alignMask) / this->alignPara) * this->alignPara;
            CopyInRow(this->inputGm.dhGm, this->ubTmp, (mStart + bOff) * H + hOff, bRows, hLen, hAligned, 0);
            SyncM2toM3();
            CopyOutRow(this->workGm.dhPrevWsGm, this->ubTmp, (mStart + bOff) * H + hOff, bRows, hLen, 0, hAligned);
            SyncM3toM2();
        }
    }
}

template <typename DTYPE>
__aicore__ inline bool GruGradKernel<DTYPE>::GateGradsMergedOk(int64_t bRows, int64_t hLen, int64_t hAligned)
{
    if constexpr (sizeof(DTYPE) != FLOAT_BYTES) {
        return false;
    } else {
        // 仅单 hTile（hLen==H）：多 tile 时 r/z/n 门列区带间隔不连续，装配布局不成立。
        // 且要求 3H 行距 32B 对齐（H%8==0）：整行 [rows,3H] 写出走多行快路径；
        // 非对齐 H 的 per-row DataCopyPad 尾块会越界写破坏下一行首元素（H=85 实证，
        // GM 行距 3H 无对齐余量），回退原逐门路径（其尾块溢出落在本行 padding 内无害）。
        // 形状侧判定 Init 缓存；容量比较改预除上界（hLen==H 时 hAligned==hiddenSizeAligned）。
        return hLen == this->H && this->mergedShapeOk_ && bRows <= this->stagingCapRows_;
    }
}

template <typename DTYPE>
__aicore__ inline void GruGradKernel<DTYPE>::ComputeGateGradsMerged(int64_t gmOffGi, int64_t batchHOff, int64_t bRows,
                                                                    int64_t hLen, int64_t hAligned)
{
    const int64_t cnt = bRows * hAligned;
    SyncM2toV(); // 门/梯度 MTE2 装载完成
    // VF 寄存器级重写——基线多组 UB 往返 V 指令收敛为少数载入/存储 + 寄存器算子
    // 中间量全程驻留向量寄存器不落 UB。
    // 数值与 基线链逐位一致（GateGradsVF 内同指令序同操作数配对）；ubHN(hn→dGh_n)与
    // ubHPrev(hp→dhFromH) 就地覆写，下游 staging/快路径消费契约不变。
    // 可用性判据：asc_vf_call 若在本编译目标未展开（非 3510），输出缓冲持旧值 →
    {
        const uint32_t vfCnt = static_cast<uint32_t>(cnt);
        const uint32_t vfVL = static_cast<uint32_t>(AscendC::GetVecLen() / sizeof(float));
        const uint16_t vfRep = static_cast<uint16_t>(
            AscendC::CeilDivision(static_cast<int32_t>(vfCnt), static_cast<int32_t>(vfVL)));
        asc_vf_call<GateGradsVF<float>>((__ubuf__ float*)this->ubTmp2.GetPhyAddr(),        // drp = dhn·hn·r·(1−r)
                                        (__ubuf__ float*)this->ubTmp3.GetPhyAddr(),        // dzp = A·(hp−n)·z
                                        (__ubuf__ float*)this->ubDHPrevFromH.GetPhyAddr(), // dhn = A·(1−n²)
                                        (__ubuf__ float*)this->ubHN.GetPhyAddr(), // dGh_n = r·dhn（就地覆写 hn）
                                        (__ubuf__ float*)this->ubHPrev.GetPhyAddr(), // dhFromH = gradH·z（就地覆写 hp）
                                        (__ubuf__ float*)this->ubGradH.GetPhyAddr(),  // gradH
                                        (__ubuf__ float*)this->ubUpdate.GetPhyAddr(), // z
                                        (__ubuf__ float*)this->ubNew.GetPhyAddr(),    // n
                                        (__ubuf__ float*)this->ubHPrev.GetPhyAddr(),  // hp
                                        (__ubuf__ float*)this->ubHN.GetPhyAddr(),     // hn
                                        (__ubuf__ float*)this->ubReset.GetPhyAddr(),  // r
                                        vfCnt, vfVL, vfRep);
    }
    SyncVtoM2();
    // UB 装配: staging_dGi(ubTmp) = [drp|dzp|dhn], staging_dGh(ubReset) = [drp|dzp|dGh_n]
    // （staging_dGh 由 ubGate 迁至 ubReset：ubGate 现专持 1.0 常量不被覆写）
    const uint16_t blkLen = static_cast<uint16_t>(hAligned * FLOAT_BYTES / BLOCK_BYTES);
    // dstGap：z/n 两门列区跨距 = (GRU_GATE_SIZE-1)*hAligned 个 fp32（32B 块单位）
    const uint16_t dstGap = static_cast<uint16_t>((GRU_GATE_SIZE - 1) * hAligned * FLOAT_BYTES / BLOCK_BYTES);
    const DataCopyParams dp(static_cast<uint16_t>(bRows), blkLen, 0, dstGap);
    DataCopy(this->ubTmp[GATE_IDX_RESET * hAligned], this->ubTmp2, dp);      // 门 r
    DataCopy(this->ubTmp[GATE_IDX_UPDATE * hAligned], this->ubTmp3, dp);     // 门 z
    DataCopy(this->ubTmp[GATE_IDX_NEW * hAligned], this->ubDHPrevFromH, dp); // 门 n
    DataCopy(this->ubReset[GATE_IDX_RESET * hAligned], this->ubTmp2, dp);    // 门 r
    DataCopy(this->ubReset[GATE_IDX_UPDATE * hAligned], this->ubTmp3, dp);   // 门 z
    DataCopy(this->ubReset[GATE_IDX_NEW * hAligned], this->ubHN, dp);        // 门 n
    SyncM2toM3();
    // 整行连续写出: [rows, 3H] 平面一次 MTE3（列偏移 3×hOff 已含在 gmOffGi）。
    // V->MTE2->MTE3 传递序已建立，CopyOutRow 内部同步经 vSyncPreDone 跳过。
    CopyOutRow(this->workGm.dGiGm, this->ubTmp, gmOffGi, bRows, GRU_GATE_SIZE * hLen,
               this->H * GRU_GATE_SIZE - GRU_GATE_SIZE * hLen, GRU_GATE_SIZE * hAligned);
    CopyOutRow(this->workGm.dGhGm, this->ubReset, gmOffGi, bRows, GRU_GATE_SIZE * hLen,
               this->H * GRU_GATE_SIZE - GRU_GATE_SIZE * hLen, GRU_GATE_SIZE * hAligned);
    CopyOutRow(this->workGm.dhFromHGm, this->ubHPrev, batchHOff, bRows, hLen, 0, hAligned);
    SyncM3toV();
}

template <typename DTYPE>
__aicore__ inline bool GruGradKernel<DTYPE>::GateGradsScatterOk(int64_t bRows, int64_t hLen, int64_t hAligned)
{
    if constexpr (sizeof(DTYPE) != FLOAT_BYTES) {
        return false;
    } else {
        // scatter 前置（单 hTile + H%8==0 + staging 容量）之上加整拍约束 hA%VL==0：
        // scatter 循环无尾拍掩码（CreateMask 全满），StoreAlign 地址恒 32B 对齐。
        // 非整拍域回退逐门路径。整拍判定 Init 缓存。
        return GateGradsMergedOk(bRows, hLen, hAligned) && this->scatterVOk_;
    }
}

template <typename DTYPE>
__aicore__ inline void GruGradKernel<DTYPE>::ComputeGateGradsScatter(int64_t gmOffGi, int64_t batchHOff, int64_t bRows,
                                                                     int64_t hLen, int64_t hAligned)
{
    // 快路径（预取视图乒乓 + hp 乒乓消费）下 Accumulate 同步后无新 MTE2——
    // 其 SyncM2toV 传递覆盖本 VF 全部输入（gradH=V 链内序、门/hp=上步预取 MTE2 管内序），
    // 免重复事件对；LoadGates/ReadHPrev 的 MTE2 分支已标脏，此处补同步。
    if (this->vfMte2Dirty_) {
        SyncM2toV(); // 门/梯度 MTE2 装载完成
    }
    {
        const uint32_t vfVL = static_cast<uint32_t>(AscendC::GetVecLen() / sizeof(float));
        asc_vf_call<GateGradsScatterVF<float>>(
            (__ubuf__ float*)this->ubTmp.GetPhyAddr(),         // dGi staging [drp|dzp|dhn]
            (__ubuf__ float*)this->ubTmp2.GetPhyAddr(),        // dGh staging [drp|dzp|dGh_n]
            (__ubuf__ float*)this->ubHPrev.GetPhyAddr(),       // dhFromH（就地覆写 hp）
            (__ubuf__ float*)this->ubDHPrevFromH.GetPhyAddr(), // hp 保留面（Tail 消费）
            (__ubuf__ float*)this->ubGradH.GetPhyAddr(),       // gradH
            (__ubuf__ float*)this->ubUpdate.GetPhyAddr(),      // z
            (__ubuf__ float*)this->ubNew.GetPhyAddr(),         // n
            (__ubuf__ float*)this->ubHPrev.GetPhyAddr(),       // hp
            (__ubuf__ float*)this->ubHN.GetPhyAddr(),          // hn
            (__ubuf__ float*)this->ubReset.GetPhyAddr(),       // r
            static_cast<uint32_t>(bRows), static_cast<uint32_t>(hAligned), vfVL);
    }
    this->hpPreserved_ = true; // 本步 hp 已保留（Tail 单 tile 域消费）
    SyncVtoM3();               // VF scatter → MTE3（替代 V→M2→M3 传递链）
    // 整行连续写出: [rows, 3H] 平面一次 MTE3；vSyncPreDone——上方已建立 V→M3 序。
    // dGi/dGh 同参共享一份 cp（bRows/块长/行距逐步恒定, 原 2× 装配 → 1×）。
    if constexpr (sizeof(DTYPE) == FLOAT_BYTES) {
        const int64_t blockLen = GRU_GATE_SIZE * hLen * FLOAT_BYTES;
        const int64_t pitchBytes = GRU_GATE_SIZE * hAligned * FLOAT_BYTES;
        DataCopyExtParams cp(static_cast<uint16_t>(bRows), static_cast<uint32_t>(blockLen),
                             (pitchBytes - blockLen) / BLOCK_BYTES,
                             static_cast<int64_t>(this->H * GRU_GATE_SIZE - GRU_GATE_SIZE * hLen) * FLOAT_BYTES, 0);
        DataCopyPad(this->workGm.dGiGm[gmOffGi], this->ubTmp, cp);
        DataCopyPad(this->workGm.dGhGm[gmOffGi], this->ubTmp2, cp);
    } else {
        CopyOutRow(this->workGm.dGiGm, this->ubTmp, gmOffGi, bRows, GRU_GATE_SIZE * hLen,
                   this->H * GRU_GATE_SIZE - GRU_GATE_SIZE * hLen, GRU_GATE_SIZE * hAligned, true);
        CopyOutRow(this->workGm.dGhGm, this->ubTmp2, gmOffGi, bRows, GRU_GATE_SIZE * hLen,
                   this->H * GRU_GATE_SIZE - GRU_GATE_SIZE * hLen, GRU_GATE_SIZE * hAligned, true);
    }
    // DhPrevUbPath 快路径下 dhFromHGm 无 GM 读者（Accumulate 直取 ubHPrev、
    // DcciStepSlices 已跳过）——MTE3 盲写出链。慢路径（多 bTile/变长）保留 GM 写。
    if (!this->DhPrevUbPath()) {
        CopyOutRow(this->workGm.dhFromHGm, this->ubHPrev, batchHOff, bRows, hLen, 0, hAligned, true);
    }
    SyncM3toV();
}

template <typename DTYPE>
__aicore__ inline void GruGradKernel<DTYPE>::AccumulateDhPrev(int64_t tIdx)
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
    if (this->DhPrevUbPath()) {
        const int64_t hAligned = ((this->H + this->alignMask) / this->alignPara) * this->alignPara;
        CopyInRow(this->dhPrevF32Gm, this->ubTmp, mStart * this->H, mCnt, this->H, hAligned, 0);
        SyncM2toV();
        // dhFromH 常驻缓冲按计算路径选择（合并路径在 ubHPrev，旧路径在 ubDHPrevFromH）
        Add(this->ubGradH, this->dhFromHInHp_ ? this->ubHPrev : this->ubDHPrevFromH, this->ubTmp, mCnt * hAligned);
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
            int64_t hAligned = ((hLen + this->alignMask) / this->alignPara) * this->alignPara;
            // dhPrevF32 为 fp32 workspace：直接 fp32 搬入（rowAlign 为 fp32 元素距）
            CopyInRow(this->dhPrevF32Gm, this->ubTmp, (mStart + bOff) * H + hOff, bRows, hLen, hAligned, 0);
            CopyInRow(this->workGm.dhFromHGm, this->ubTmp2, (mStart + bOff) * H + hOff, bRows, hLen, hAligned, 0);
            SyncM2toV();
            Add(this->ubTmp3, this->ubTmp, this->ubTmp2, bRows * hAligned);
            SyncVtoM3();
            CopyOutRow(this->workGm.dhPrevWsGm, this->ubTmp3, (mStart + bOff) * H + hOff, bRows, hLen, 0, hAligned);
            SyncM3toM2();
        }
    }
}

template <typename DTYPE>
__aicore__ inline void GruGradKernel<DTYPE>::StoreDhPrev()
{
    const int64_t bTile = this->vecBTile_, hTile = this->vecHTile_;
    int64_t mStart = 0, mCnt = 0;
    if (!this->GetCoreRows(mStart, mCnt)) {
        return;
    }
    if (this->DhPrevUbPath()) {
        const int64_t hAligned = ((this->H + this->alignMask) / this->alignPara) * this->alignPara;
        CopyOutRow(this->outputGm.dhPrevGm, this->ubGradH, mStart * this->H, mCnt, this->H, 0, hAligned);
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
            int64_t hAligned = ((hLen + this->alignMask) / this->alignPara) * this->alignPara;
            CopyInRow(this->workGm.dhPrevWsGm, this->ubTmp, (mStart + bOff) * H + hOff, bRows, hLen, hAligned, 0);
            SyncM2toM3();
            CopyOutRow(this->outputGm.dhPrevGm, this->ubTmp, (mStart + bOff) * H + hOff, bRows, hLen, 0, hAligned);
            SyncM3toM2();
        }
    }
}

#endif // _GRU_GRAD_BAND_ARCH35_
