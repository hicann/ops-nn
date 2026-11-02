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
 * \file gru_grad_dgate_arch35.h
 * \brief dgate L1 驻留路径、通用 GEMM tile 执行器、S1 K 切片 GEMM 的类外定义。
 */
#ifndef _GRU_GRAD_DGATE_ARCH35_
#define _GRU_GRAD_DGATE_ARCH35_

template <typename DTYPE>
__aicore__ inline void GruGradKernel<DTYPE>::PlanDgateResident()
{
    const GGemmPlan& p = this->dgatePlan;
    const int64_t dSize = static_cast<int64_t>(sizeof(DTYPE));
    this->dgSlotBBytes_ = AlignUp512(p.l1BElems * dSize);
    this->dgSlotABytes_ = AlignUp512(p.l1AElems * dSize);
    this->dgChunkCnt_ = (p.K > 0 && p.kChunk > 0) ? (p.K + p.kChunk - 1) / p.kChunk : 0;
    this->dgTileCnt_ = (p.N > 0 && p.nStep > 0) ? (p.N + p.nStep - 1) / p.nStep : 0;
    this->dgResident_ = 0;
    if (this->tiling->isSeqLength != 0 || this->dgChunkCnt_ <= 1 || this->dgTileCnt_ < 1) {
        return;
    }
    const int64_t aBurstBytes = this->dgChunkCnt_ * this->dgSlotABytes_;
    if (aBurstBytes + this->dgSlotBBytes_ >= static_cast<int64_t>(ggcube::L1_BUDGET)) {
        return;
    }
    const int64_t avail = static_cast<int64_t>(ggcube::L1_BUDGET) - aBurstBytes - this->dgSlotBBytes_;
    int64_t r = avail / this->dgSlotBBytes_;
    const int64_t total = this->dgChunkCnt_ * this->dgTileCnt_;
    if (r > total) {
        r = total;
    }
    if (r < RESIDENT_MIN_SLOTS || r * DG_RESIDENT_MIN_RATIO < total) { // 驻留率不足则不值得切换
        return;
    }
    this->dgResident_ = r;
}

template <typename DTYPE>
__aicore__ inline void GruGradKernel<DTYPE>::PlanWNz()
{
    this->wNzUsable_ = false;
    if (!this->wNzGmValid_ || this->tiling->wNzEnabled == 0 || this->tiling->isSeqLength != 0) {
        return;
    }
    const GGemmPlan& p = this->dgatePlan;
    if (this->dgResident_ > 0 || p.K <= 0 || p.kChunk <= 0 || p.N <= 0 || p.nStep <= 0) {
        return;
    }
    if (p.K % p.kChunk != 0 || p.N % p.nStep != 0 || p.kChunk % ggcube::CUBE_BLOCK != 0) {
        return; // 尾块(k/n)与 kChunk<16 分形形态走原 ND2NZ 路径
    }
    this->wNzUsable_ = true;
    // 复核: K 切片数(chunk 数)须 >= AIC 核数(空切片簇会产生未初始化部分和)
    const int64_t chunks = p.K / p.kChunk;
    const int64_t aicCnt = static_cast<int64_t>(GetBlockNum());
    this->s1Usable_ = this->s1SegValid_ && this->tiling->s1Enabled != 0 && chunks >= aicCnt;
}

template <typename DTYPE>
__aicore__ inline void GruGradKernel<DTYPE>::PreloadDgateB()
{
    const GGemmPlan& p = this->dgatePlan;
    int64_t loaded = 0;
    for (int64_t nt = 0; nt < this->dgTileCnt_ && loaded < this->dgResident_; ++nt) {
        const int64_t n0 = nt * p.nStep;
        const int64_t nSub = (p.N - n0 < p.nStep) ? (p.N - n0) : p.nStep;
        const int64_t nLd0 = (nSub + ggcube::CUBE_BLOCK - 1) / ggcube::CUBE_BLOCK * ggcube::CUBE_BLOCK;
        for (int64_t cIdx = 0; cIdx < this->dgChunkCnt_ && loaded < this->dgResident_; ++cIdx) {
            const int64_t k0 = cIdx * p.kChunk;
            const int64_t kSub = (p.K - k0 < p.kChunk) ? (p.K - k0) : p.kChunk;
            const int64_t kLoad = (kSub > ggcube::CUBE_BLOCK) ?
                                      (kSub + ggcube::CUBE_BLOCK - 1) / ggcube::CUBE_BLOCK * ggcube::CUBE_BLOCK :
                                      kSub;
            AscendC::LocalTensor<DTYPE> b1(TPosition::B1, loaded * this->dgSlotBBytes_, p.l1BElems);
            ggcube::CopyInNd2Nz<DTYPE>(b1, this->inputGm.wHiddenGm[k0 * this->H + n0], kLoad, nLd0, this->H, kLoad);
            ++loaded;
        }
    }
}

template <typename DTYPE>
__aicore__ inline void GruGradKernel<DTYPE>::RunDgateResident(GlobalTensor<DTYPE>& aGm, GlobalTensor<float>& cGm,
                                                              int64_t mExt)
{
    const GGemmPlan& p = this->dgatePlan;
    const int64_t mSub = ggcube::CUBE_BLOCK; // mExt <= 16，分形整装
    const int64_t aRowPitch = this->H * GRU_GATE_SIZE;
    const int64_t aScrBase = this->dgResident_ * this->dgSlotBBytes_;
    const int64_t bScrOff = aScrBase + this->dgChunkCnt_ * this->dgSlotABytes_;
    const int64_t total = this->dgTileCnt_ * this->dgChunkCnt_;
    AscendC::LocalTensor<DTYPE> b2(TPosition::B2, 0, this->l0BElems_);
    // A4: L0C 双带偏移——每带独立 C tile 区域，消除带间 WAIT_FLAG_FIXP（L0C WAR）。
    // L0C 256KB >> 每带 [16, nStep]×4B，双带仅 ~16KB。带 1 的 MMAD 与带 0 的 Fixpipe
    // 完全并行（不同 L0C 地址无 WAR）。偏移按 512B 对齐（L0C 分形粒度）。
    const int64_t cTileBytes = AlignUp512(mSub * p.nStep * static_cast<int64_t>(sizeof(float)));
    // a 侧全 K 单次 ND2NZ——[mSub, K] 一次入 L1（单 op 替代 dgChunkCnt_ 次 4KB 窄行突发）。
    // 同一全 K 像一次 SplitA 整装 L0A(48KB≤64KB, kStep 跨 k 分形), 各 Mmad 取
    // cIdx×l1AElems 偏移视图——12 次 SplitA/chunk 合并为 1 次/步, Mmad 序不变(数值
    // 逐位一致), 消除 L0A 逐 chunk WAR 与 11 次串行发射(屏障仍由 L0B 的 B 侧 WAR 决定,
    // B 侧 w_hh 远大于 L0B 不可合并)。a1Full 提升作用域供 SplitA 消费。
    const int64_t aFullElems = this->dgChunkCnt_ * p.l1AElems;
    AscendC::LocalTensor<DTYPE> a1Full(TPosition::A1, aScrBase, aFullElems);
    ggcube::CopyInNd2Nz<DTYPE>(a1Full, aGm[0], mSub, p.K, aRowPitch, mSub);
    // 首个 scratch chunk（flat == dgResident_，存在时）随突发一并预发射
    bool scratchInFlight = false;
    if (this->dgResident_ < total) {
        const int64_t nt = this->dgResident_ / this->dgChunkCnt_;
        const int64_t cIdx = this->dgResident_ % this->dgChunkCnt_;
        const int64_t k0 = cIdx * p.kChunk;
        const int64_t kSub = (p.K - k0 < p.kChunk) ? (p.K - k0) : p.kChunk;
        const int64_t kLoad = (kSub > ggcube::CUBE_BLOCK) ?
                                  (kSub + ggcube::CUBE_BLOCK - 1) / ggcube::CUBE_BLOCK * ggcube::CUBE_BLOCK :
                                  kSub;
        const int64_t n0 = nt * p.nStep;
        const int64_t nSub = (p.N - n0 < p.nStep) ? (p.N - n0) : p.nStep;
        const int64_t nLd0 = (nSub + ggcube::CUBE_BLOCK - 1) / ggcube::CUBE_BLOCK * ggcube::CUBE_BLOCK;
        AscendC::LocalTensor<DTYPE> b1(TPosition::B1, bScrOff, p.l1BElems);
        ggcube::CopyInNd2Nz<DTYPE>(b1, this->inputGm.wHiddenGm[k0 * this->H + n0], kLoad, nLd0, this->H, kLoad);
        scratchInFlight = true;
    }
    ggsync::WaitMte2ToMte1(); // 覆盖 a 突发 +（首步）B 预装 + 首个 scratch 预发射
    {
        AscendC::LocalTensor<DTYPE> a2Full(TPosition::A2, 0, aFullElems);
        ggcube::SplitA<DTYPE>(a2Full, a1Full, mSub, p.K);
    }
    for (int64_t nt = 0; nt < this->dgTileCnt_; ++nt) {
        const int64_t n0 = nt * p.nStep;
        const int64_t nSub = (p.N - n0 < p.nStep) ? (p.N - n0) : p.nStep;
        const int64_t nLd0 = (nSub + ggcube::CUBE_BLOCK - 1) / ggcube::CUBE_BLOCK * ggcube::CUBE_BLOCK;
        // A4: 每带独立 L0C 偏移——带间无 WAR，MMAD(带1) 与 Fixpipe(带0) 并行
        AscendC::LocalTensor<float> c(TPosition::CO1, nt * cTileBytes, this->l0CElems_);
        bool firstK = true;
        for (int64_t cIdx = 0; cIdx < this->dgChunkCnt_; ++cIdx) {
            const int64_t flat = nt * this->dgChunkCnt_ + cIdx;
            const int64_t k0 = cIdx * p.kChunk;
            const int64_t kSub = (p.K - k0 < p.kChunk) ? (p.K - k0) : p.kChunk;
            const int64_t kLoad = (kSub > ggcube::CUBE_BLOCK) ?
                                      (kSub + ggcube::CUBE_BLOCK - 1) / ggcube::CUBE_BLOCK * ggcube::CUBE_BLOCK :
                                      kSub;
            if (scratchInFlight) {
                ggsync::WaitMte2ToMte1(); // scratch(c) 装载完成（须在 PIPE_ALL 前）
                scratchInFlight = false;
            }
            if (flat > 0) {
                PipeBarrier<PIPE_ALL>(); // Mmad(c-1) 退休，L0A/L0B/scratch 可重写
            }
            AscendC::LocalTensor<DTYPE> b1(
                TPosition::B1, (flat < this->dgResident_) ? flat * this->dgSlotBBytes_ : bScrOff, p.l1BElems);
            AscendC::LocalTensor<DTYPE> a2c(TPosition::A2, cIdx * p.l1AElems * static_cast<int64_t>(sizeof(DTYPE)),
                                            p.l1AElems);
            ggcube::SplitBV2<DTYPE>(b2, b1, kLoad, nSub, kLoad); // 单次调用转置(singleK=槽行数)
            ggsync::WaitMte1ToM();
            if (firstK) {
                ggcube::MmadPlain<float, DTYPE, DTYPE>(c, a2c, b2, mSub, nSub, kSub);
                firstK = false;
            } else {
                ggcube::MmadAccum<float, DTYPE, DTYPE>(c, a2c, b2, mSub, nSub, kSub);
            }
            // 预发射下一 chunk（若为 scratch）：与 Mmad 同管外重叠
            if (flat + 1 < total && flat + 1 >= this->dgResident_) {
                const int64_t nnt = (flat + 1) / this->dgChunkCnt_;
                const int64_t nc = (flat + 1) % this->dgChunkCnt_;
                const int64_t nk0 = nc * p.kChunk;
                const int64_t nkSub = (p.K - nk0 < p.kChunk) ? (p.K - nk0) : p.kChunk;
                const int64_t nkLoad = (nkSub > ggcube::CUBE_BLOCK) ?
                                           (nkSub + ggcube::CUBE_BLOCK - 1) / ggcube::CUBE_BLOCK * ggcube::CUBE_BLOCK :
                                           nkSub;
                const int64_t nn0 = nnt * p.nStep;
                const int64_t nnSub = (p.N - nn0 < p.nStep) ? (p.N - nn0) : p.nStep;
                const int64_t nnLd0 = (nnSub + ggcube::CUBE_BLOCK - 1) / ggcube::CUBE_BLOCK * ggcube::CUBE_BLOCK;
                AscendC::LocalTensor<DTYPE> nb1(TPosition::B1, bScrOff, p.l1BElems);
                ggcube::CopyInNd2Nz<DTYPE>(nb1, this->inputGm.wHiddenGm[nk0 * this->H + nn0], nkLoad, nnLd0, this->H,
                                           nkLoad);
                scratchInFlight = true;
            }
        }
        ggsync::WaitMToFix();
        ggcube::FixpipeL0CToGm(cGm[n0], c, mExt, nSub, this->H);
        // A4: WaitFixToM 移除——每带独立 L0C 偏移，带间无 L0C WAR。
        // 带 0 的 Fixpipe 排空不再阻塞带 1 的 MMAD（不同 L0C 区域）。
        // 末带的 WaitFixToM 由 CubeProcess 的 PIPE_ALL + C2V 兜底（L0C 生命周期
        // 跨步复用仍需排空，但排空与带间并行已解耦）。
        // 带 0 提前信号——Fixpipe 发射后即发 B0（不等排空，PIPE_ALL 兜底落地）。
        // 末带追加 B1——早于 FLAG_C2V 一个 PIPE_ALL+set，AIV 预发射 band1 M2。
        if (nt == 0 && this->dgTileCnt_ >= BAND_NUM) { // 仅双带形态存在 band0 提前信号
            PipeBarrier<PIPE_ALL>();
            ggsync::CubeSignalVecBand();
            this->band0Early_ = true;
        } else if (nt == this->dgTileCnt_ - 1 && this->dgTileCnt_ >= BAND_NUM) { // 末带 B1 信号
            PipeBarrier<PIPE_ALL>();
            ggsync::CubeSignalVecBand1();
            this->band1Early_ = true;
        }
    }
}

template <typename DTYPE>
__aicore__ inline void GruGradKernel<DTYPE>::RunGemmTiles(const GGemmPlan& p, GlobalTensor<DTYPE>& aGm,
                                                          int64_t aRowPitch, GlobalTensor<DTYPE>& bGm,
                                                          int64_t bRowPitch, GlobalTensor<float>& cGm,
                                                          int64_t cRowPitch, int64_t m0, int64_t n0, int64_t mExt,
                                                          int64_t nExt, bool batched, bool pipelined)
{
    if (mExt <= 0 || nExt <= 0) {
        return;
    }
    // 批路径的 L1 足迹 = SUPER×(chunkA+chunkB), 超预算回退逐 chunk 路径
    constexpr int64_t SUPER = 4;
    if (batched) {
        const int64_t dSize = static_cast<int64_t>(sizeof(DTYPE));
        const int64_t aBytes = AlignUp512(SUPER * p.l1AElems * dSize);
        if (aBytes + SUPER * p.l1BElems * dSize > static_cast<int64_t>(ggcube::L1_BUDGET)) {
            batched = false;
        }
    }
    const int64_t a1Cap = batched ? (SUPER * p.l1AElems) : this->l1AElems_;
    const int64_t b1Cap = batched ? (SUPER * p.l1BElems) : this->l1BElems_;
    AscendC::LocalTensor<DTYPE> a1(TPosition::A1, 0, a1Cap);
    // ⚠ A1/B1 是同一块 L1 的逻辑视角，LocalTensor 构造函数第二参为 L1 内偏移、单位是【字节】
    // （b1 形参: b1(B1, aElems * sizeof(Op), bElems)）。两者都取 0 会互踩；第一版修复
    // 误传元素数（fp32 差 4 倍）仍互踩—— A=0 仍产出非零。按 512B 对齐排 A 之后。
    const int64_t b1OffBytes = this->AlignUp512(a1Cap * static_cast<int64_t>(sizeof(DTYPE)));
    AscendC::LocalTensor<DTYPE> b1(TPosition::B1, b1OffBytes, b1Cap);
    AscendC::LocalTensor<DTYPE> a2(TPosition::A2, 0, this->l0AElems_);
    AscendC::LocalTensor<DTYPE> b2(TPosition::B2, 0, this->l0BElems_);
    AscendC::LocalTensor<float> c(TPosition::CO1, 0, this->l0CElems_);

    const int64_t mTiles = (mExt + p.mStep - 1) / p.mStep;
    const int64_t nTiles = (nExt + p.nStep - 1) / p.nStep;
    for (int64_t mt = 0; mt < mTiles; ++mt) {
        const int64_t mSub0 = mt * p.mStep;
        const int64_t mSubRaw = (mExt - mSub0 < p.mStep) ? (mExt - mSub0) : p.mStep;
        // fp32 Mmad 的 m<16 分形语义未实测（已实测 fp16 m=2 与 m=16/64），为落在实测包络内，
        // 装载/Mmad 恒按 16 行分形整装（GM pad 行读越界只进入 workspace 相邻段，合法且不被泄出），
        // Fixpipe 仅泄出 mSubRaw 个真实行。
        const int64_t mSub = (mSubRaw + ggcube::CUBE_BLOCK - 1) / ggcube::CUBE_BLOCK * ggcube::CUBE_BLOCK;
        for (int64_t nt = 0; nt < nTiles; ++nt) {
            const int64_t nSub0 = nt * p.nStep;
            const int64_t nSub = (nExt - nSub0 < p.nStep) ? (nExt - nSub0) : p.nStep;
            // ⚠ N 侧补满 16 对齐: nSub 非 16 整除时 Nd2Nz 装出半块 NZ n-block, 后续 Mmad 按
            // CeilAlign(n,16) 计算并经 Fixpipe 泄出陈旧 L1 列（读旧值）。
            const int64_t nLd0 = (nSub + ggcube::CUBE_BLOCK - 1) / ggcube::CUBE_BLOCK * ggcube::CUBE_BLOCK;
            if (pipelined && !batched && p.K > p.kChunk && this->H <= PIPELINE_MAX_HIDDEN) {
                // 该门: H>256 L1 预算超限
                // ⚠ 门禁承重：H>256 时 L1 预算超限，勿移除。
                // 深度-1 流水——L1 双缓冲(消 MTE1→MTE2 WAR 边) + CopyIn(c+1) 预发射(与
                // Mmad(c) 重叠, MTE2/M 不同管不同存) + Mmad 后窄事件 WaitMToMte1(dW 路径
                // 已证安全)。单 MTE2 在飞——避开 批装载多描述符突发的未解管线 bug。
                const int64_t aBuf = p.l1AElems, bBuf = p.l1BElems;
                const int64_t dS = static_cast<int64_t>(sizeof(DTYPE));
                const int64_t bOff2 = this->AlignUp512(DBUF_SLOTS * aBuf * dS);
                if (bOff2 + DBUF_SLOTS * bBuf * dS <= static_cast<int64_t>(ggcube::L1_BUDGET)) {
                    LocalTensor<DTYPE> a1p(TPosition::A1, 0, DBUF_SLOTS * aBuf);
                    LocalTensor<DTYPE> b1p(TPosition::B1, bOff2, DBUF_SLOTS * bBuf);
                    int64_t chunkIdx = 0;
                    bool firstK = true;
                    {
                        const int64_t kSub = (p.K < p.kChunk) ? p.K : p.kChunk;
                        const int64_t kLoad = (kSub > ggcube::CUBE_BLOCK) ?
                                                  (kSub + ggcube::CUBE_BLOCK - 1) / ggcube::CUBE_BLOCK *
                                                      ggcube::CUBE_BLOCK :
                                                  kSub;
                        ggcube::CopyInNd2Nz<DTYPE>(a1p[0], aGm[(m0 + mSub0) * aRowPitch], mSub, kSub, aRowPitch, mSub);
                        ggcube::CopyInNd2Nz<DTYPE>(b1p[0], bGm[n0 + nSub0], kLoad, nLd0, bRowPitch, kLoad);
                    }
                    for (int64_t k0 = 0; k0 < p.K; k0 += p.kChunk) {
                        const int64_t kSub = (p.K - k0 < p.kChunk) ? (p.K - k0) : p.kChunk;
                        const int64_t kLoad = (kSub > ggcube::CUBE_BLOCK) ?
                                                  (kSub + ggcube::CUBE_BLOCK - 1) / ggcube::CUBE_BLOCK *
                                                      ggcube::CUBE_BLOCK :
                                                  kSub;
                        ggsync::WaitMte2ToMte1();
                        if (chunkIdx > 0) {
                            // Mmad(c-1) 完成后 Split(c) 才可写 L0(L0B 无双缓冲, 64KB 满)。
                            // PIPE_ALL 在此处不抵消预发射: CopyIn(c) 已于上迭代底部发出,
                            // 与 Mmad(c-1) 在 MTE2/M 两管上重叠完成。
                            // ⚠ 窄事件 WaitMToMte1 于本位置实证
                            PipeBarrier<PIPE_ALL>();
                        }
                        const int64_t cur = (chunkIdx % DBUF_SLOTS) * aBuf; // 乒乓当前槽
                        const int64_t curB = (chunkIdx % DBUF_SLOTS) * bBuf;
                        ggcube::SplitA<DTYPE>(a2, a1p[cur], mSub, kSub);
                        ggcube::SplitB<DTYPE>(b2, b1p[curB], kLoad, nSub);
                        ggsync::WaitMte1ToM();
                        if (firstK) {
                            ggcube::MmadPlain<float, DTYPE, DTYPE>(c, a2, b2, mSub, nSub, kSub);
                            firstK = false;
                        } else {
                            ggcube::MmadAccum<float, DTYPE, DTYPE>(c, a2, b2, mSub, nSub, kSub);
                        }
                        if (k0 + p.kChunk < p.K) {
                            const int64_t kSubN = (p.K - k0 - p.kChunk < p.kChunk) ? (p.K - k0 - p.kChunk) : p.kChunk;
                            const int64_t kLoadN = (kSubN > ggcube::CUBE_BLOCK) ?
                                                       (kSubN + ggcube::CUBE_BLOCK - 1) / ggcube::CUBE_BLOCK *
                                                           ggcube::CUBE_BLOCK :
                                                       kSubN;
                            const int64_t nxt = ((chunkIdx + 1) % DBUF_SLOTS) * aBuf; // 预发射下一槽
                            const int64_t nxtB = ((chunkIdx + 1) % DBUF_SLOTS) * bBuf;
                            ggcube::CopyInNd2Nz<DTYPE>(a1p[nxt], aGm[(m0 + mSub0) * aRowPitch + k0 + p.kChunk], mSub,
                                                       kSubN, aRowPitch, mSub);
                            ggcube::CopyInNd2Nz<DTYPE>(b1p[nxtB], bGm[(k0 + p.kChunk) * bRowPitch + n0 + nSub0], kLoadN,
                                                       nLd0, bRowPitch, kLoadN);
                        }
                        ++chunkIdx;
                    }
                } else {
                    goto vanillaKLoop;
                }
                goto gemmDone;
            }
            if (batched) {
                // 超块批量装载 —— SUPER 个 chunk 的操作数 A/B 一次连发(MTE2 发射共享延迟) +
                // 单次 WaitMte2ToMte1, 子循环内零 MTE2/零 MTE2 等待; Mmad 后窄事件(M→MTE1),
                // 超块尾补 MTE1→MTE2(护下一超块 L1 重写)。终轮 GEMM(K=TS, kChunk
                // L0B 压至 16 → 400 次串行 MTE2 等待)的喂给瓶颈解法。
                bool firstK = true;
                for (int64_t sc0 = 0; sc0 < p.K; sc0 += p.kChunk * SUPER) {
                    int64_t nSc = (p.K - sc0 + p.kChunk - 1) / p.kChunk;
                    nSc = (nSc < SUPER) ? nSc : SUPER;
                    for (int64_t ci = 0; ci < nSc; ++ci) {
                        const int64_t k0 = sc0 + ci * p.kChunk;
                        const int64_t kSub = (p.K - k0 < p.kChunk) ? (p.K - k0) : p.kChunk;
                        const int64_t kLoad = (kSub > ggcube::CUBE_BLOCK) ?
                                                  (kSub + ggcube::CUBE_BLOCK - 1) / ggcube::CUBE_BLOCK *
                                                      ggcube::CUBE_BLOCK :
                                                  kSub;
                        ggcube::CopyInNd2Nz<DTYPE>(a1[ci * p.l1AElems], aGm[(m0 + mSub0) * aRowPitch + k0], mSub, kSub,
                                                   aRowPitch, mSub);
                        ggcube::CopyInNd2Nz<DTYPE>(b1[ci * p.l1BElems], bGm[k0 * bRowPitch + n0 + nSub0], kLoad, nLd0,
                                                   bRowPitch, kLoad);
                    }
                    ggsync::WaitMte2ToMte1();
                    for (int64_t ci = 0; ci < nSc; ++ci) {
                        const int64_t k0 = sc0 + ci * p.kChunk;
                        const int64_t kSub = (p.K - k0 < p.kChunk) ? (p.K - k0) : p.kChunk;
                        const int64_t kLoad = (kSub > ggcube::CUBE_BLOCK) ?
                                                  (kSub + ggcube::CUBE_BLOCK - 1) / ggcube::CUBE_BLOCK *
                                                      ggcube::CUBE_BLOCK :
                                                  kSub;
                        ggcube::SplitA<DTYPE>(a2, a1[ci * p.l1AElems], mSub, kSub);
                        ggcube::SplitB<DTYPE>(b2, b1[ci * p.l1BElems], kLoad, nSub);
                        ggsync::WaitMte1ToM();
                        if (firstK) {
                            ggcube::MmadPlain<float, DTYPE, DTYPE>(c, a2, b2, mSub, nSub, kSub);
                            firstK = false;
                        } else {
                            ggcube::MmadAccum<float, DTYPE, DTYPE>(c, a2, b2, mSub, nSub, kSub);
                        }
                        // ⚠ M→MTE1 窄事件在 MIX AIC 语境不可靠
                        // dgate 批装载: 换全栅栏。
                        PipeBarrier<PIPE_ALL>();
                    }
                    PipeBarrier<PIPE_ALL>(); // 超块尾全栅栏(MTE1→MTE2 边: 窄事件在此 MIX AIC 语境不可靠,
                                             // dx 复现)
                }
            } else {
            vanillaKLoop:
                for (int64_t k0 = 0; k0 < p.K; k0 += p.kChunk) {
                    const int64_t kSub = (p.K - k0 < p.kChunk) ? (p.K - k0) : p.kChunk;
                    // B 侧装载 K 填充到 16 倍数：SplitB 的 L1 布局公式仅在被 16 整除的 K 上实测
                    // （k=32/64/128/256）；非整除 K（如 3H=24、TS=21）的分组跨步落在未测区。
                    // Mmad 仍用真实 kSub——"k 恰好按值采用，pad 通道不贡献"（真机实测）。
                    // ⚠ 仅 k>16 时填充：k≤16 的单分形形态已被小 TS 用例实测通过；若强制 16 对齐，
                    // 大 N 时 L0B 足迹 n16*k16*dsize 会
                    // 突破 64KB——）。
                    const int64_t kLoad = (kSub > ggcube::CUBE_BLOCK) ? (kSub + ggcube::CUBE_BLOCK - 1) /
                                                                            ggcube::CUBE_BLOCK * ggcube::CUBE_BLOCK :
                                                                        kSub;
                    // GM -> L1（NZ）：srcDValue = 行起点距（元素，实现内部转 gap）
                    ggcube::CopyInNd2Nz<DTYPE>(a1, aGm[(m0 + mSub0) * aRowPitch + k0], mSub, kSub, aRowPitch, mSub);
                    if (pipelined && this->wNzUsable_) {
                        // dgate B 侧(w_hh, 跨 t 不变)读预暂存 NZ 像——线性 GM->L1(消每步
                        // ND2NZ 变换)。槽址 = (nt*chunkCnt + cIdx) * l1Belems, 与 AIV writer 一致。
                        // 位级等价: 像内容 = CopyInNd2Nz 同参输出(一次变换), SplitB/Mmad 不变。
                        const int64_t wnFlat = nt * this->dgChunkCnt_ + k0 / p.kChunk;
                        const int64_t wnOff = wnFlat * p.l1BElems;
                        DataCopy(b1, this->wNzGm[wnOff], static_cast<uint32_t>(kLoad * nLd0));
                    } else {
                        ggcube::CopyInNd2Nz<DTYPE>(b1, bGm[k0 * bRowPitch + n0 + nSub0], kLoad, nLd0, bRowPitch, kLoad);
                    }
                    ggsync::WaitMte2ToMte1();
                    // L1 -> L0A/L0B
                    ggcube::SplitA<DTYPE>(a2, a1, mSub, kSub);
                    ggcube::SplitB<DTYPE>(b2, b1, kLoad, nSub);
                    ggsync::WaitMte1ToM();
                    if (k0 == 0) {
                        ggcube::MmadPlain<float, DTYPE, DTYPE>(c, a2, b2, mSub, nSub, kSub);
                    } else {
                        ggcube::MmadAccum<float, DTYPE, DTYPE>(c, a2, b2, mSub, nSub, kSub);
                    }
                    PipeBarrier<PIPE_ALL>();
                }
            }
        gemmDone:
            ggsync::WaitMToFix();
            ggcube::FixpipeL0CToGm(cGm[(m0 + mSub0) * cRowPitch + n0 + nSub0], c, mSubRaw, nSub, cRowPitch);
            ggsync::WaitFixToM();
        }
    }
}

template <typename DTYPE>
__aicore__ inline void GruGradKernel<DTYPE>::RunDgateSliceGEMM(int64_t aBase, int64_t kFrom, int64_t kTo)
{
    const GGemmPlan& p = this->dgatePlan;
    const int64_t aRowPitch = this->H * GRU_GATE_SIZE;
    const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    GlobalTensor<DTYPE> aGm = this->workGm.dGhGm[aBase];
    GlobalTensor<float> cGm = this->s1PartGm[blockIdx * this->B * this->H];
    AscendC::LocalTensor<DTYPE> a1(TPosition::A1, 0, this->l1AElems_);
    const int64_t b1OffBytes = this->AlignUp512(this->l1AElems_ * static_cast<int64_t>(sizeof(DTYPE)));
    AscendC::LocalTensor<DTYPE> b1(TPosition::B1, b1OffBytes, this->l1BElems_);
    AscendC::LocalTensor<DTYPE> a2(TPosition::A2, 0, this->l0AElems_);
    AscendC::LocalTensor<DTYPE> b2(TPosition::B2, 0, this->l0BElems_);
    AscendC::LocalTensor<float> c(TPosition::CO1, 0, this->l0CElems_);
    const int64_t mTiles = (this->B + p.mStep - 1) / p.mStep;
    const int64_t nTiles = (p.N + p.nStep - 1) / p.nStep;
    for (int64_t mt = 0; mt < mTiles; ++mt) {
        const int64_t mSub0 = mt * p.mStep;
        const int64_t mSubRaw = (this->B - mSub0 < p.mStep) ? (this->B - mSub0) : p.mStep;
        const int64_t mSub = (mSubRaw + ggcube::CUBE_BLOCK - 1) / ggcube::CUBE_BLOCK * ggcube::CUBE_BLOCK;
        for (int64_t nt = 0; nt < nTiles; ++nt) {
            const int64_t nSub0 = nt * p.nStep;
            const int64_t nSub = (p.N - nSub0 < p.nStep) ? (p.N - nSub0) : p.nStep;
            const int64_t nLd0 = (nSub + ggcube::CUBE_BLOCK - 1) / ggcube::CUBE_BLOCK * ggcube::CUBE_BLOCK;
            for (int64_t k0 = kFrom; k0 < kTo; k0 += p.kChunk) {
                const int64_t kSub = p.kChunk; // K 切分 域门保证 K%kChunk==0
                ggcube::CopyInNd2Nz<DTYPE>(a1, aGm[mSub0 * aRowPitch + k0], mSub, kSub, aRowPitch, mSub);
                const int64_t wnFlat = nt * this->dgChunkCnt_ + k0 / p.kChunk;
                DataCopy(b1, this->wNzGm[wnFlat * p.l1BElems], static_cast<uint32_t>(kSub * nLd0));
                ggsync::WaitMte2ToMte1();
                ggcube::SplitA<DTYPE>(a2, a1, mSub, kSub);
                ggcube::SplitB<DTYPE>(b2, b1, kSub, nSub);
                ggsync::WaitMte1ToM();
                if (k0 == kFrom) {
                    ggcube::MmadPlain<float, DTYPE, DTYPE>(c, a2, b2, mSub, nSub, kSub);
                } else {
                    ggcube::MmadAccum<float, DTYPE, DTYPE>(c, a2, b2, mSub, nSub, kSub);
                }
                PipeBarrier<PIPE_ALL>();
            }
            ggsync::WaitMToFix();
            ggcube::FixpipeL0CToGm(cGm[mSub0 * this->H + nSub0], c, mSubRaw, nSub, this->H);
            ggsync::WaitFixToM();
        }
    }
}

#endif // _GRU_GRAD_DGATE_ARCH35_
