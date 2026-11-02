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
 * \file gru_grad_aic_arch35.h
 * \brief AIC 主流程、终轮三路批量 GEMM、S1 规约/DCCI/wNz 写像的类外定义。
 */
#ifndef _GRU_GRAD_AIC_ARCH35_
#define _GRU_GRAD_AIC_ARCH35_

template <typename DTYPE>
__aicore__ inline int64_t GruGradKernel<DTYPE>::VecRowStart(int64_t aivIdx)
{
    const int64_t blockDim = static_cast<int64_t>(GetBlockNum()) * AIV_PER_AIC;
    const int64_t bPerCore = (this->B + blockDim - 1) / blockDim;
    const int64_t rem = this->B % blockDim;
    if (rem == 0) {
        return aivIdx * bPerCore;
    }
    if (aivIdx < rem) {
        return aivIdx * bPerCore;
    }
    return rem * bPerCore + (aivIdx - rem) * (bPerCore - 1);
}

template <typename DTYPE>
__aicore__ inline int64_t GruGradKernel<DTYPE>::VecRowCount(int64_t aivIdx)
{
    const int64_t blockDim = static_cast<int64_t>(GetBlockNum()) * AIV_PER_AIC;
    const int64_t bPerCore = (this->B + blockDim - 1) / blockDim;
    const int64_t rem = this->B % blockDim;
    int64_t cnt = (rem == 0) ? bPerCore : ((aivIdx < rem) ? bPerCore : bPerCore - 1);
    int64_t start = VecRowStart(aivIdx);
    if (start >= this->B) {
        return 0;
    }
    if (start + cnt > this->B) {
        cnt = this->B - start;
    }
    return cnt;
}

template <typename DTYPE>
__aicore__ inline void GruGradKernel<DTYPE>::DgateStep(int64_t tIdx)
{
    const int64_t cluster = static_cast<int64_t>(GetBlockIdx());
    const int64_t aiv0 = cluster * AIV_PER_AIC;
    const int64_t r0 = VecRowStart(aiv0);
    const int64_t r1 = VecRowStart(aiv0 + 1) + VecRowCount(aiv0 + 1);
    int64_t mExt = r1 - r0;
    if (r0 >= this->B || mExt <= 0) {
        return;
    }
    if (r0 + mExt > this->B) {
        mExt = this->B - r0;
    }

    const bool isReverse = (this->tiling->direction != 0);
    const int64_t gateIdx = isReverse ? (this->T - 1 - tIdx) : tIdx;
    int64_t aBase = 0;
    if (this->tiling->isSeqLength) {
        int64_t compactRow = this->GetCompactRowOffset(gateIdx);
        aBase = compactRow * this->H * GRU_GATE_SIZE;
        int64_t curBatch = this->inputGm.batchSizesGm.GetValue(gateIdx);
        if (r0 >= curBatch) {
            return;
        }
        if (r0 + mExt > curBatch) {
            mExt = curBatch - r0;
        }
    } else {
        aBase = gateIdx * this->B * this->H * GRU_GATE_SIZE;
    }

    GlobalTensor<DTYPE> aGm = this->workGm.dGhGm[aBase + r0 * this->H * GRU_GATE_SIZE];
    GlobalTensor<float> cGm = this->dhPrevF32Gm[r0 * this->H];
    // 已回退: dgate 批装载(SUPER>=2)在三组实验中确定性坏(输入源/工作区中转源均然,
    // dW 同码独过, 根因未定位——kChunk>=64 或 T 循环语境相关); T 循环 GEMM 维持逐 chunk。
    // w_hh B 侧驻留路径优先(单 m-tile 小 M 场景), 大 mExt 回退 深度-1 流水。
    if (this->dgResident_ > 0 && mExt <= ggcube::CUBE_BLOCK) {
        this->RunDgateResident(aGm, cGm, mExt);
        return;
    }
    RunGemmTiles(this->dgatePlan, aGm, this->H * GRU_GATE_SIZE, this->inputGm.wHiddenGm, this->H, cGm, this->H, 0, 0,
                 mExt, this->H, false, true); // 深度-1 流水
}

template <typename DTYPE>
__aicore__ inline void GruGradKernel<DTYPE>::RunDxGemm()
{
    const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    const int64_t numClusters = static_cast<int64_t>(GetBlockNum());
    const int64_t TS = this->totalSteps_;
    const int64_t bandTS = (TS + numClusters - 1) / numClusters;
    const int64_t dx0 = blockIdx * bandTS;
    if (dx0 < TS) {
        const int64_t dxExt = (TS - dx0 < bandTS) ? TS - dx0 : bandTS;
        if constexpr (sizeof(DTYPE) == FLOAT_BYTES) {
            // fp32 时 GEMM 直写输出张量, 免 dxF32 中转 + ConvertOnePlane 回投拷贝
            // （m0=dx0 同时索引 a 侧 dGiGm 与 c 侧, dxOut 传全量视图）
            GlobalTensor<float> dxOut;
            dxOut.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(this->outputGm.dxGm.GetPhyAddr(0)),
                                  this->totalSteps_ * this->I);
            RunGemmTiles(this->dxPlan, this->workGm.dGiGm, this->H * GRU_GATE_SIZE, this->inputGm.wInputGm, this->I,
                         dxOut, this->I, dx0, 0, dxExt, this->dxPlan.N, false);
        } else {
            RunGemmTiles(this->dxPlan, this->workGm.dGiGm, this->H * GRU_GATE_SIZE, this->inputGm.wInputGm, this->I,
                         this->dxF32Gm, this->I, dx0, 0, dxExt, this->dxPlan.N, false);
        }
    }
}

template <typename DTYPE>
__aicore__ inline void GruGradKernel<DTYPE>::RunDwGemm2D(const GGemmPlan& p, GlobalTensor<DTYPE>& aGm,
                                                         int64_t aRowPitch, GlobalTensor<DTYPE>& bGm, int64_t bRowPitch,
                                                         GlobalTensor<float>& cGm, int64_t cRowPitch)
{
    if (p.M <= 0 || p.N <= 0) {
        return;
    }
    const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    const int64_t numClusters = static_cast<int64_t>(GetBlockNum());
    const int64_t nBand = p.nStep;
    const int64_t nBands = (p.N + nBand - 1) / nBand;
    int64_t mBand = (p.M * nBands + numClusters - 1) / numClusters;
    mBand = (mBand + ggcube::CUBE_BLOCK - 1) / ggcube::CUBE_BLOCK * ggcube::CUBE_BLOCK;
    if (mBand < ggcube::CUBE_BLOCK) {
        mBand = ggcube::CUBE_BLOCK;
    }
    const int64_t mBands = (p.M + mBand - 1) / mBand;
    const int64_t totalTiles = mBands * nBands;
    for (int64_t tile = blockIdx; tile < totalTiles; tile += numClusters) {
        const int64_t mb = tile / nBands;
        const int64_t nb = tile % nBands;
        const int64_t m0 = mb * mBand;
        const int64_t mExt = (p.M - m0 < mBand) ? (p.M - m0) : mBand;
        const int64_t n0 = nb * nBand;
        const int64_t nExt = (p.N - n0 < nBand) ? (p.N - n0) : nBand;
        RunGemmTiles(p, aGm, aRowPitch, bGm, bRowPitch, cGm, cRowPitch, m0, n0, mExt, nExt, true);
    }
}

template <typename DTYPE>
__aicore__ inline void GruGradKernel<DTYPE>::RunDwGmms()
{
    // dW^T 重排(不变量 #11 B 侧): dwIh^T[I,3H] = xT[I,TS] @ dGi[TS,3H]
    //                            dwHh^T[H,3H] = hPrevT[H,TS] @ dGh[TS,3H]
    // B 侧 dGi/dGh ND 直载白拿转置; A 侧为小转置 xT/hPrevT(6x/3x 体积↓)。
    const int64_t TS = this->totalSteps_;
    this->RunDwGemm2D(this->dwIhPlan, this->xTGm, TS, this->workGm.dGiGm, this->H * GRU_GATE_SIZE, this->dwIF32Gm,
                      this->H * GRU_GATE_SIZE);
    this->RunDwGemm2D(this->dwHhPlan, this->hPrevTGm, TS, this->workGm.dGhGm, this->H * GRU_GATE_SIZE, this->dwHF32Gm,
                      this->H * GRU_GATE_SIZE);
}

template <typename DTYPE>
__aicore__ inline void GruGradKernel<DTYPE>::CubeProcess()
{
    const bool hasRows = VecRowCount(static_cast<int64_t>(GetBlockIdx()) * AIV_PER_AIC) > 0 ||
                         VecRowCount(static_cast<int64_t>(GetBlockIdx()) * AIV_PER_AIC + 1) > 0;
    // w_hh 驻留槽一次性预装（T 循环内不再重复装载）
    if (hasRows && this->dgResident_ > 0) {
        this->PreloadDgateB();
    }
    // K 切分主循环——dgate 全行×本簇 K 切片, 部分和跨簇规约(AIV 侧)。
    // 逐步协议: SyncAll#1(dGh 全簇就绪) → 切片 GEMM → PIPE_ALL(fixpipe 落地, E2/E4 纪律)
    //           → SyncAll#2(部分和发布)。B0/B1/C2V 逐步信号在本域全部旁路(Finals 的
    //           V2C#1/#2 由 AIV Finalize 正常发射, 不受影响)。
    if (this->s1Usable_) {
        const GGemmPlan& p = this->dgatePlan;
        const int64_t chunks = p.K / p.kChunk;
        const int64_t aicCnt = static_cast<int64_t>(GetBlockNum());
        const int64_t bIdx = static_cast<int64_t>(GetBlockIdx());
        // 平衡 chunk 分配: 簇 i 得 [chunks*i/C, chunks*(i+1)/C)——无一空切片
        // (朴素 ceil 均分曾使尾部簇空转, 其 (mt,nt) 循环仍 fixpipe 未初始化 L0C 垃圾进 partial)
        const int64_t c0 = chunks * bIdx / aicCnt;
        const int64_t c1 = chunks * (bIdx + 1) / aicCnt;
        const int64_t kFrom = c0 * p.kChunk;
        const int64_t kTo = c1 * p.kChunk;
        for (int64_t tIdx = this->T - 1; tIdx >= 0; tIdx--) {
            AscendC::SyncAll<false>();
            const int64_t gateIdx = (this->tiling->direction != 0) ? (this->T - 1 - tIdx) : tIdx;
            const int64_t aBase = gateIdx * this->B * this->H * GRU_GATE_SIZE;
            this->RunDgateSliceGEMM(aBase, kFrom, kTo);
            PipeBarrier<PIPE_ALL>();
            AscendC::SyncAll<false>();
        }
        ggsync::CubeWaitVec();
        this->RunDxGemm();      // dx = dGi x w_ih —— 与 AIV 转置重叠(V2C#1)
        ggsync::CubeWaitVec2(); // V2C#2: 本簇 xT/hPrevT 就绪
        this->RunDwGmms();      // dwIh/dwHh —— 需转置产物
        PipeBarrier<PIPE_ALL>();
        ggsync::CubeSignalVec();
        return;
    }
    // 时间逆向循环：每步一轮 V->C（消费一次）+ 一轮 C->V（set 一次）——轮次静态配对
    for (int64_t tIdx = this->T - 1; tIdx >= 0; tIdx--) {
        ggsync::CubeWaitVec();
        this->band0Early_ = false; // 每步复位（RunDgate 的带内置位）
        this->band1Early_ = false; // A1-P3a
        if (hasRows) {
            this->DgateStep(tIdx);
        }
        // 纪律 1——每步每块恰好一次 B0 set：无行块/非带路径（mExt>CUBE_BLOCK 回退、
        // dgResident_==0、单 tile 域）未在 GEMM 内发射者由此补发，与 AIV 的 VecWaitCubeBand
        // 保持轮次静态配对（B0 恒先于 FLAG_C2V 到达，AIV 计算时机不变）。
        if (!this->band0Early_) {
            ggsync::CubeSignalVecBand();
        }
        if (!this->band1Early_) {
            ggsync::CubeSignalVecBand1();
        }
        // Fixpipe 的 GM 写在指令退休后仍可能在 write buffer 中未落地；C2V 置位（PIPE_FIX）
        // 只等指令退休。置位前全停排空，保证消费者（AIV MTE2 读）看到最终数据。
        // ⚠ 消融实证(E2/E4): 本屏障必须 PIPE_ALL——收窄为 PIPE_FIX 或移除均挂死
        //   (flag 协议承重, 非仅数据定序)。
        PipeBarrier<PIPE_ALL>();
        ggsync::CubeSignalVec();
    }
    // 终轮拆两段（dx 重叠转置）:
    //   V2C #1 —— T 循环全部 dGi/dGh 已齐(SyncAll 后), 先跑 dx GEMM(只读 dGi 原布局,
    //             不依赖转置)与 AIV 的 TransposeGiGhT 并行
    //   V2C #2 —— 转置完成后跑 dW 两路(需 dGiT/dGhT)
    ggsync::CubeWaitVec();
    this->RunDxGemm();      // dx = dGi x w_ih —— 与 AIV 转置重叠(V2C#1)
    ggsync::CubeWaitVec2(); // V2C#2: 本簇 xT/hPrevT 就绪
    this->RunDwGmms();      // dwIh/dwHh —— 需转置产物
    PipeBarrier<PIPE_ALL>();
    ggsync::CubeSignalVec();
}

template <typename DTYPE>
__aicore__ inline void GruGradKernel<DTYPE>::ReduceS1()
{
    int64_t mStart = 0, mCnt = 0;
    if (!this->GetCoreRows(mStart, mCnt) || mCnt <= 0) {
        return;
    }
    const int64_t aicCnt = static_cast<int64_t>(GetBlockNum());
    const int64_t segMax = this->allocLength;
    for (int64_t r = 0; r < mCnt; ++r) {
        for (int64_t h0 = 0; h0 < this->H; h0 += segMax) {
            const int64_t seg = (this->H - h0 < segMax) ? (this->H - h0) : segMax;
            DataCopy(this->ubTmp, this->s1PartGm[(mStart + r) * this->H + h0], static_cast<uint32_t>(seg));
            SyncM2toV();
            for (int64_t c = 1; c < aicCnt; ++c) {
                DataCopy(this->ubTmp2, this->s1PartGm[c * this->B * this->H + (mStart + r) * this->H + h0],
                         static_cast<uint32_t>(seg));
                SyncM2toV();
                Add(this->ubTmp, this->ubTmp, this->ubTmp2, static_cast<uint32_t>(seg));
                SyncVtoM2(); // WAR: Add(V 读 ubTmp2) → 下一轮 MTE2 重写 ubTmp2
            }
            SyncVtoM3();
            DataCopy(this->dhPrevF32Gm[(mStart + r) * this->H + h0], this->ubTmp, static_cast<uint32_t>(seg));
            SyncM3toV();
        }
    }
    // MTE3 写(dhPrevF32) → 同核后续 MTE2 读跨管定序：SyncM3toV 仅排 V 管，
    // 不覆盖 MTE2 读者，须全停排空。
    PipeBarrier<PIPE_ALL>();
    GlobalTensor<float> dp;
    dp.SetGlobalBuffer(this->dhPrevF32Gm.GetPhyAddr(mStart * this->H * FLOAT_BYTES), mCnt * this->H);
    DataCacheCleanAndInvalid<float, AscendC::CacheLine::ENTIRE_DATA_CACHE>(dp);
}

template <typename DTYPE>
__aicore__ inline void GruGradKernel<DTYPE>::DcciStepSlices(int64_t tIdx)
{
    const bool isReverse = (this->tiling->direction != 0);
    const int64_t gateIdx = isReverse ? (this->T - 1 - tIdx) : tIdx;
    int64_t mStart = 0, mCnt = 0;
    if (!this->GetCoreRows(mStart, mCnt)) {
        return;
    }
    const int64_t cols3H = this->H * GRU_GATE_SIZE;
    // dGi 的跨核读者只有终轮 dx GEMM——逐步 DCCI 延迟到 T 循环后一次性全量刷新
    // （VecProcess 的 V2C#1 前），此处仅刷 dGh（dgate 每步经 V2C 消费）。
    if (this->tiling->isSeqLength) {
        int64_t compactRow = this->GetCompactRowOffset(gateIdx);
        int64_t curBatch = this->inputGm.batchSizesGm.GetValue(gateIdx);
        if (mStart >= curBatch) {
            return;
        }
        if (mStart + mCnt > curBatch) {
            mCnt = curBatch - mStart;
        }
        GlobalTensor<DTYPE> gh;
        gh.SetGlobalBuffer(this->workGm.dGhGm.GetPhyAddr((compactRow + mStart) * cols3H), mCnt * cols3H);
        DataCacheCleanAndInvalid<DTYPE, AscendC::CacheLine::ENTIRE_DATA_CACHE>(gh);
        // hp DCCI 移入 ProcessVectorTail(GEMM 窗口)
    } else {
        GlobalTensor<DTYPE> gh;
        gh.SetGlobalBuffer(this->workGm.dGhGm.GetPhyAddr((gateIdx * this->B + mStart) * cols3H), mCnt * cols3H);
        DataCacheCleanAndInvalid<DTYPE, AscendC::CacheLine::ENTIRE_DATA_CACHE>(gh);
        // hp DCCI 移入 ProcessVectorTail(GEMM 窗口)
    }
    if (!this->DhPrevUbPath()) { // 快路径下 dhFromHGm 无 GM 读者(Accumulate 直取 UB)
        GlobalTensor<DTYPE> dhf;
        dhf.SetGlobalBuffer(this->workGm.dhFromHGm.GetPhyAddr(mStart * this->H), mCnt * this->H);
        DataCacheCleanAndInvalid<DTYPE, AscendC::CacheLine::ENTIRE_DATA_CACHE>(dhf);
    }
}

template <typename DTYPE>
__aicore__ inline void GruGradKernel<DTYPE>::WriteWNzImage()
{
    const GGemmPlan& p = this->dgatePlan;
    const int64_t aivCnt = static_cast<int64_t>(GetBlockNum()) * AIV_PER_AIC;
    const int64_t aivIdx = static_cast<int64_t>(GetBlockIdx());
    const int64_t totalSlots = this->dgTileCnt_ * this->dgChunkCnt_;
    const int64_t kFrac = p.kChunk / ggcube::CUBE_BLOCK;
    const int64_t c0e = ggcube::C0<DTYPE>();
    const int64_t nBlocks = p.nStep / c0e;
    const int64_t sliceElems = p.kChunk * c0e;
    const int64_t dSize = static_cast<int64_t>(sizeof(DTYPE));
    LocalTensor<DTYPE> bufs[WNZ_STAGE_BUFFS] = {
        this->ubGate.template ReinterpretCast<DTYPE>(), this->ubTmp.template ReinterpretCast<DTYPE>(),
        this->ubTmp2.template ReinterpretCast<DTYPE>(), this->ubTmp3.template ReinterpretCast<DTYPE>()};
    const int64_t ubElems = this->allocLength * FLOAT_BYTES / dSize;
    const int64_t perBuf = ubElems / sliceElems; // 每 TBuf 可容切片数(fp32 4KB 切片 → 4)
    DataCopyPadParams pp;
    pp.isPad = false;
    pp.leftPadding = 0;
    pp.rightPadding = 0;
    pp.paddingValue = 0;
    for (int64_t s = aivIdx; s < totalSlots; s += aivCnt) {
        const int64_t nt = s / this->dgChunkCnt_;
        const int64_t cIdx = s % this->dgChunkCnt_;
        const int64_t k0 = cIdx * p.kChunk;
        const int64_t n0 = nt * p.nStep;
        const int64_t slotBase = s * p.l1BElems;
        for (int64_t g = 0; g < nBlocks;) {
            int64_t done = 0;
            for (int64_t b = 0; b < WNZ_STAGE_BUFFS && g < nBlocks; ++b, g += perBuf) {
                const int64_t cnt = (nBlocks - g < perBuf) ? (nBlocks - g) : perBuf;
                for (int64_t j = 0; j < cnt; ++j) {
                    DataCopyParams cp;
                    cp.blockCount = static_cast<uint32_t>(p.kChunk);
                    cp.blockLen = static_cast<uint32_t>(c0e * dSize);
                    cp.srcStride = static_cast<uint32_t>((this->H - c0e) * dSize);
                    cp.dstStride = 0;
                    DataCopyPad(bufs[b][j * sliceElems], this->inputGm.wHiddenGm[k0 * this->H + n0 + (g + j) * c0e], cp,
                                pp);
                }
                done += cnt;
            }
            PipeBarrier<PIPE_ALL>(); // MTE2 批完成(MTE2→MTE3 边)
            int64_t gBase = g - done;
            for (int64_t b = 0; b < WNZ_STAGE_BUFFS && gBase < nBlocks; ++b) {
                const int64_t cnt = (nBlocks - gBase < perBuf) ? (nBlocks - gBase) : perBuf;
                for (int64_t j = 0; j < cnt; ++j) {
                    const int64_t gg = gBase + j;
                    DataCopy(this->wNzGm[slotBase + gg * kFrac * (ggcube::CUBE_BLOCK * c0e)], bufs[b][j * sliceElems],
                             static_cast<uint32_t>(sliceElems));
                }
                gBase += cnt;
            }
            PipeBarrier<PIPE_ALL>(); // MTE3 批排空(下一轮 MTE2 重写 TBuf 的 WAR 边)
        }
        // 纪律 5: 跨核读者(AIC MTE2)可见性
        GlobalTensor<DTYPE> view;
        view.SetGlobalBuffer(this->wNzGm.GetPhyAddr(slotBase * dSize), p.l1BElems);
        DataCacheCleanAndInvalid<DTYPE, AscendC::CacheLine::ENTIRE_DATA_CACHE>(view);
    }
}

template <typename DTYPE>
__aicore__ inline void GruGradKernel<DTYPE>::VecProcess()
{
    // 巨型 shape 时先一次性写 w_hh NZ 像(预循环 TBuf 全空闲可借用), SyncAll 全 AIV 背书——
    // 首步 V2C 在其后, AIC 经 CubeWaitVec 消费时像已全局完备。
    if (this->wNzUsable_) {
        this->WriteWNzImage();
        SyncAll();
    }
    // ubGate 常驻 1.0（每 kernel 一次），合并路径免逐步 Duplicate
    Duplicate(this->ubGate, 1.0f, this->allocLength);
    this->InitFastPathCache(); // 快路径判定/行带划分一次算齐（AIV 侧全部消费者）
    if (this->s1Usable_) {     // K 切分主循环（逐步双全局屏障 + 规约）
        for (int64_t tIdx = this->T - 1; tIdx >= 0; tIdx--) {
            this->ProcessVector(tIdx);
            this->DcciStepSlices(tIdx);
            AscendC::SyncAll<false>();     // #1: dGh(tIdx) 全簇就绪 → AIC 切片 GEMM
            this->ProcessVectorTail(tIdx); // GEMM 窗口重叠
            AscendC::SyncAll<false>();     // #2: 部分和全簇就绪
            this->ReduceS1();              // → dhPrevF32[own rows] + DCCI
            this->AccumulateDhPrev(tIdx);
        }
        this->VecProcessFinalize();
        return;
    }
    if (this->bandPath_) { // 带化主循环（首步全 H + 带循环 + 尾步 epilogue）
        this->VecProcessBand();
        return;
    }
    if (this->tiling->isSeqLength) {
        this->InitDhPrev();
    }
    // 时间逆向循环：每步一轮 V->C + 一轮 C->V（双 AIV 都 set / AIC 消费一次；轮次静态配对）
    for (int64_t tIdx = this->T - 1; tIdx >= 0; tIdx--) {
        this->ProcessVector(tIdx);
        this->DcciStepSlices(tIdx); // 纪律 5：dGi/dGh 刷新(hp 已移入 Tail 的 GEMM 窗口)
        ggsync::VecSignalCube();    // V->C（栅栏语义：双 AIV set，AIC 消费一次）
        // GEMM 窗口——StoreHPrev + hp DCCI + 门/dy 预取移入(与 GEMM 重叠);
        // 其 MTE2/MTE3 在 C2V 等待期间异步排空
        this->ProcessVectorTail(tIdx);
        ggsync::VecWaitCubeBand(); // 带 0 提前信号（恒先于 C2V 到达，计算时机不变）
        ggsync::VecWaitCube();     // C->V（dhPrevF32 就绪）。非模板 wait 默认 PIPE_S＝发射流，
                               // 等待本身阻塞本核后续全部发射，原随后的 PIPE_ALL 冗余，移除。
        this->AccumulateDhPrev(tIdx);
        // 逐步 SyncAll 移除(洁净重试版)。原失败与 vSyncPredone 毒混淆(三处
        // 同步同缺); 毒源修复后分离变量重验: 48 大形状 + 16 边界形状 golden 零失败。
        // WAR 覆盖(单跳 S 阻塞序, 非多跳传递): 读(k) 经 Accumulate 内 SyncM2toV 完成后,
        // S 方发射 V2C(k-1) 的 MTE3 flag → AIC 写(k-1) 不与读(k) 竞争; R6b 时代失败疑因
        // 其基线先于 Tail 同步链引入。栅栏前 AIV 已被握手链近似同步。
        // ⚠ D6(hPrevWs 直读消中转) 曾尝试移除 StoreHPrev 并以 M2→M3/M3→M2 双向窄同步补边——
        // 大形状间歇损坏(1/5~5/5), 存在超出两条已识别 WAR 的隐性耦合(疑乒乓预取与 staging
        // 的跨轮交互), 已整体回退。教训: StoreHPrev 的同步链是 T 循环管序的隐性承重墙。
    }
    this->VecProcessFinalize();
}

template <typename DTYPE>
__aicore__ inline void GruGradKernel<DTYPE>::VecProcessFinalize()
{
    // ⚠ T 循环后必须一道全栅栏：TransposeGiGhT 按列带转置、读全部 TS 行——跨簇消费
    // 所有 AIV 的逐步 dGi/dGh 写出（小形状删除后 dW 两路错、
    // ）。逐步 DCCI 已保证可见性，此处只补完成序。
    SyncAll();
    // dGi 一次性全量 DCCI——其唯一跨核读者是终轮 dx GEMM（V2C#1 后），逐步刷新无必要
    {
        GlobalTensor<DTYPE> gi;
        gi.SetGlobalBuffer(this->workGm.dGiGm.GetPhyAddr(0), this->totalSteps_ * this->H * GRU_GATE_SIZE);
        DataCacheCleanAndInvalid<DTYPE, AscendC::CacheLine::ENTIRE_DATA_CACHE>(gi);
    }
    ggsync::VecSignalCube(); // V2C#1(FLAG 9): dGi/dGh 全局齐, AIC 先跑 dx(与转置重叠)
    this->TransposeGiGhT();
    // ⚠ 转置尾部 DCCI(缓存清刷)不在 V2C#2 的 PIPE_MTE3 定序内——flag 抢跑 cache clean,
    // AIC 读到未刷写的 xT/hPrevT。
    PipeBarrier<PIPE_ALL>();
    // 2D 分核使 dW 的 A 侧(xT/hPrevT)跨簇消费——V2C#2 仅保证本簇转置完成，
    // 置位前须网格栅栏为全部簇的转置产物背书（沿用终轮 C2V 后 SyncAll 的 AIV 间背书模式；
    // dGi/dGh 侧已由 T 循环后 SyncAll + 全量 DCCI 覆盖）。
    SyncAll();
    ggsync::VecSignalCube2(); // V2C#2(FLAG 10, 独立 ID): 转置全局完成, AIC 跑 dW GEMMs
    this->StoreDhPrev();
    // bias 输出——恒走 ProcessBiasReduce GM 归约（dGi/dGh [TS,3H] 列和 = 全 (T,B) 行贡献）。
    // 原 band+biasInline 域（H∈{128,256}）的 UB 部分和全宽直写在 ≥2 AIV 有行时
    // 构成多写竞争且缺他核行贡献（db 终值不确定），已弃用（原优化自评性能中性）。
    if (this->tiling->isBias == 1) {
        int64_t rows = this->totalSteps_;
        int64_t cols = this->H * GRU_GATE_SIZE;
        this->ProcessBiasReduce(this->workGm.dGiGm, this->outputGm.dbInputGm, rows, cols);
        this->ProcessBiasReduce(this->workGm.dGhGm, this->outputGm.dbHiddenGm, rows, cols);
    }
    ggsync::VecWaitCube(); // 终轮 C->V（本簇 AIC 的 GEMM 完成）
    PipeBarrier<PIPE_ALL>();
    // ⚠ dW 两路 convert 行带在 N 维（cols3H）而 GEMM M 带在 M 维——簇 c 的 convert 读 dwIF32/
    // dwHF32 全部 M 行（跨簇产物），仅本簇 C2V 不足。AIV 间 SyncAll：各 AIV 为本簇 AIC 完成背书，
    // 全员通过即全簇 GEMM 完成（周期 28 零行实证：GEMM 输出正确但 convert 读到未写区）。
    SyncAll();
    this->ConvertOutputs();
}

template <typename DTYPE>
__aicore__ inline int64_t GruGradKernel<DTYPE>::GetCompactRowOffset(int64_t seqIdx)
{
    int64_t offset = 0;
    for (int64_t i = 0; i < seqIdx; i++) {
        offset += this->inputGm.batchSizesGm.GetValue(i);
    }
    return offset;
}

template <typename DTYPE>
__aicore__ inline bool GruGradKernel<DTYPE>::GetCoreRows(int64_t& mStart, int64_t& mCnt)
{
    // 行带划分每核恒定（B/blockDim/块序 kernel 内不变）——首调缓存，消逐步
    // 4-5 处重复的整除/取模/分支（ESL: AIV 段标量风暴成分之一）。
    if (!this->coreRowsInit_) {
        int64_t blockDim = static_cast<int64_t>(GetBlockNum()) * AIV_PER_AIC;
        int64_t B_per_core = (this->B + blockDim - 1) / blockDim;
        int64_t rem = this->B % blockDim;
        int64_t blockIdx = GetBlockIdx();
        if (rem == 0 || blockIdx < rem) {
            this->mCnt_ = B_per_core;
            this->mStart_ = blockIdx * B_per_core;
        } else {
            this->mCnt_ = B_per_core - 1;
            this->mStart_ = rem * B_per_core + (blockIdx - rem) * (B_per_core - 1);
        }
        this->coreRowsOk_ = (this->mCnt_ > 0 && this->mStart_ < this->B);
        this->coreRowsInit_ = true;
    }
    mStart = this->mStart_;
    mCnt = this->mCnt_;
    return this->coreRowsOk_;
}

#endif // _GRU_GRAD_AIC_ARCH35_
