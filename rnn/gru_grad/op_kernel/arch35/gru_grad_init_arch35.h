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
 * \file gru_grad_init_arch35.h
 * \brief Init/缓冲初始化/GEMM 计划构建等基础成员的类外定义。
 */
#ifndef _GRU_GRAD_INIT_ARCH35_
#define _GRU_GRAD_INIT_ARCH35_

template <typename DTYPE>
__aicore__ inline int64_t GruGradKernel<DTYPE>::AlignUp512(int64_t bytes)
{
    return (bytes + L1_ALIGN_BYTES - 1) / L1_ALIGN_BYTES * L1_ALIGN_BYTES;
}

template <typename DTYPE>
__aicore__ inline void GruGradKernel<DTYPE>::Init(GM_ADDR x, GM_ADDR w_input, GM_ADDR w_hidden, GM_ADDR init_h,
                                                  GM_ADDR output_h, GM_ADDR reset_gate, GM_ADDR update_gate,
                                                  GM_ADDR new_gate, GM_ADDR h_n, GM_ADDR dy, GM_ADDR dh,
                                                  GM_ADDR batch_sizes, GM_ADDR dx, GM_ADDR dh_prev, GM_ADDR dw_input,
                                                  GM_ADDR dw_hidden, GM_ADDR db_input, GM_ADDR db_hidden,
                                                  const GruGradTilingData* tiling, GM_ADDR workspace)
{
    this->tiling = tiling;
    this->T = tiling->timeStep;
    this->B = tiling->batch;
    this->H = tiling->hiddenSize;
    this->I = tiling->inputSize;
    this->totalSteps_ = tiling->totalSteps;
    this->vecBTile_ = tiling->singleCoreM;
    this->vecHTile_ = tiling->singleCoreN;

    InitGlobalBuffers(x, w_input, w_hidden, init_h, output_h, reset_gate, update_gate, new_gate, h_n, dy, dh,
                      batch_sizes, dx, dh_prev, dw_input, dw_hidden, db_input, db_hidden, workspace);
    BuildPlans();
    InitVectorBuf();
    this->PlanDgateResident();
    this->PlanWNz();
}

template <typename DTYPE>
__aicore__ inline void GruGradKernel<DTYPE>::Process()
{
    if ASCEND_IS_AIC {
        this->CubeProcess();
    }
    if ASCEND_IS_AIV {
        this->VecProcess();
    }
    PipeBarrier<PIPE_ALL>();
}

template <typename DTYPE>
__aicore__ inline void GruGradKernel<DTYPE>::InitWorkBuffers(__gm__ uint8_t* ws)
{
    constexpr int64_t dSize = static_cast<int64_t>(sizeof(DTYPE));
    int64_t TS = totalSteps_;
    int64_t off = 0;
    // 注：原实现用 lambda 封装 SetGlobalBuffer；CCE 中 __aicore__ 展开为 [aicore] lambda 属性，
    // 与 setD 的 aicore 调用域冲突导致 fp32/fp16 编译失败（host/aicore 调用域不匹配），改为
    // 私有 __aicore__ 成员函数 SetWorkBuf。
    setD(workGm.dGhGm, ws, off, TS * H * GRU_GATE_SIZE);
    off += TS * H * GRU_GATE_SIZE * dSize;
    setD(workGm.dGiGm, ws, off, TS * H * GRU_GATE_SIZE);
    off += TS * H * GRU_GATE_SIZE * dSize;
    setD(workGm.hPrevWsGm, ws, off, TS * H);
    off += TS * H * dSize;
    // xRevWs [TS,I] 段已删除：本 kernel 零消费（deprecated），host 侧同步移除预留
    setD(workGm.dhPrevWsGm, ws, off, B * H);
    off += B * H * dSize;
    setD(workGm.dhFromHGm, ws, off, B * H);
    off += B * H * dSize;
    // ---- 扩展段（host 侧 GetWorkspaceSize 同序）----
    // dW^T 重排(不变量 #11 B 侧): 只物化 xT[I,TS]/hPrevT[H,TS](体积 6x/3x↓),
    // 复用原 dGiT/dGhT 段(TS*3H ≥ TS*I 和 TS*H)
    setD(xTGm, ws, off, TS * I);
    off += TS * I * dSize;
    setD(hPrevTGm, ws, off, TS * H);
    off += TS * H * dSize;
    dhPrevF32Gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ws + off), B * H);
    off += B * H * FLOAT_BYTES;
    dwIF32Gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ws + off), H * GRU_GATE_SIZE * I);
    off += H * GRU_GATE_SIZE * I * FLOAT_BYTES;
    dwHF32Gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ws + off), H * GRU_GATE_SIZE * H);
    off += H * GRU_GATE_SIZE * H * FLOAT_BYTES;
    dxF32Gm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ws + off), TS * I);
    off += TS * I * FLOAT_BYTES;
    // wNz 段——w_hh [3H,H] 的 NZ 像(输入 dtype)。host wNzEnabled 时分配 3H*H;
    // kernel 侧 wNzUsable_ 复核失败则不消费(段位闲置)。
    if (this->tiling->wNzEnabled != 0) {
        setD(wNzGm, ws, off, static_cast<int64_t>(H) * GRU_GATE_SIZE * H);
        this->wNzGmValid_ = true;
        off += static_cast<int64_t>(H) * GRU_GATE_SIZE * H * dSize;
    }
    // dgate K 切分部分和段 [aicoreCnt, B, H] fp32——宿主 s1Enabled 时分配
    if (this->tiling->s1Enabled != 0) {
        this->s1PartGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(ws + off),
                                       static_cast<int64_t>(GetBlockNum()) * this->B * this->H);
        this->s1SegValid_ = true;
    }
}

template <typename DTYPE>
__aicore__ inline void GruGradKernel<DTYPE>::setD(GlobalTensor<DTYPE>& g, __gm__ uint8_t* base, int64_t off,
                                                  int64_t elems)
{
    g.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE*>(base + off), elems);
}

template <typename DTYPE>
__aicore__ inline void GruGradKernel<DTYPE>::InitGlobalBuffers(GM_ADDR x, GM_ADDR w_input, GM_ADDR w_hidden,
                                                               GM_ADDR init_h, GM_ADDR output_h, GM_ADDR reset_gate,
                                                               GM_ADDR update_gate, GM_ADDR new_gate, GM_ADDR h_n,
                                                               GM_ADDR dy, GM_ADDR dh, GM_ADDR batch_sizes, GM_ADDR dx,
                                                               GM_ADDR dh_prev, GM_ADDR dw_input, GM_ADDR dw_hidden,
                                                               GM_ADDR db_input, GM_ADDR db_hidden, GM_ADDR workspace)
{
    this->inputGm.xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE*>(x), totalSteps_ * I);
    this->inputGm.wInputGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE*>(w_input), I * H * GRU_GATE_SIZE);
    this->inputGm.wHiddenGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE*>(w_hidden), H * H * GRU_GATE_SIZE);
    this->inputGm.outputHGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE*>(output_h), totalSteps_ * H);
    this->inputGm.dyGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE*>(dy), totalSteps_ * H);
    this->inputGm.dhGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE*>(dh), B * H);
    this->inputGm.initHGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE*>(init_h), B * H);
    this->inputGm.resetGateGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE*>(reset_gate), totalSteps_ * H);
    this->inputGm.updateGateGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE*>(update_gate), totalSteps_ * H);
    this->inputGm.newGateGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE*>(new_gate), totalSteps_ * H);
    this->inputGm.hNGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE*>(h_n), totalSteps_ * H);
    this->outputGm.dxGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE*>(dx), totalSteps_ * I);
    this->outputGm.dhPrevGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE*>(dh_prev), B * H);
    this->outputGm.dwInputGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE*>(dw_input), I * H * GRU_GATE_SIZE);
    this->outputGm.dwHiddenGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE*>(dw_hidden), H * H * GRU_GATE_SIZE);
    if (this->tiling->isBias == 1) {
        this->outputGm.dbInputGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE*>(db_input), H * GRU_GATE_SIZE);
        this->outputGm.dbHiddenGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE*>(db_hidden), H * GRU_GATE_SIZE);
    }
    if (this->tiling->isSeqLength) {
        this->inputGm.batchSizesGm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(batch_sizes), T);
    }
    this->InitWorkBuffers(reinterpret_cast<__gm__ uint8_t*>(workspace));
}

template <typename DTYPE>
__aicore__ inline void GruGradKernel<DTYPE>::InitVectorBuf()
{
    this->pipe.InitBuffer(this->vbGradH, allocLength * FLOAT_BYTES);
    this->pipe.InitBuffer(this->vbHPrev, allocLength * FLOAT_BYTES);
    this->pipe.InitBuffer(this->vbReset, allocLength * FLOAT_BYTES);
    this->pipe.InitBuffer(this->vbUpdate, allocLength * FLOAT_BYTES);
    this->pipe.InitBuffer(this->vbNew, allocLength * FLOAT_BYTES);
    this->pipe.InitBuffer(this->vbHN, allocLength * FLOAT_BYTES);
    this->pipe.InitBuffer(this->vbDHPrevFromH, allocLength * FLOAT_BYTES);
    this->pipe.InitBuffer(this->vbTmp, allocLength * FLOAT_BYTES);
    this->pipe.InitBuffer(this->vbTmp2, allocLength * FLOAT_BYTES);
    this->pipe.InitBuffer(this->vbTmp3, allocLength * FLOAT_BYTES);
    this->pipe.InitBuffer(this->vbGate, allocLength * FLOAT_BYTES);
    this->pipe.InitBuffer(this->vbNReset, allocLength * FLOAT_BYTES);
    this->pipe.InitBuffer(this->vbNUpdate, allocLength * FLOAT_BYTES);
    this->pipe.InitBuffer(this->vbNNew, allocLength * FLOAT_BYTES);
    this->pipe.InitBuffer(this->vbNHN, allocLength * FLOAT_BYTES);
    // VECCALC 池 15×16KB + 8KB = 248KB（ascend950 UB 上限）。
    // 原 16×16KB=256KB 时第 16 块 [248KB,256KB) 与框架保留临时区（counter-mask 模式
    // 的 Duplicate/Cast 掩码暂存）重叠——条件性数据损坏。本缓冲仅承载 dy 预取
    // （原 ubNHPrev 无消费者，复用为 ubNDy），预取超 8KB 域由 pfDyCapOk_ 关闭并
    // 由 LoadGradH 回退 GM 直读。
    this->pipe.InitBuffer(this->vbNHPrev, (allocLength / DBUF_SLOTS) * FLOAT_BYTES); // 半池 8KB=allocLength/2

    this->ubGradH = this->vbGradH.template Get<float>();
    this->ubHPrev = this->vbHPrev.template Get<float>();
    this->ubReset = this->vbReset.template Get<float>();
    this->ubUpdate = this->vbUpdate.template Get<float>();
    this->ubNew = this->vbNew.template Get<float>();
    this->ubHN = this->vbHN.template Get<float>();
    this->ubDHPrevFromH = this->vbDHPrevFromH.template Get<float>();
    this->ubTmp = this->vbTmp.template Get<float>();
    this->ubTmp2 = this->vbTmp2.template Get<float>();
    this->ubTmp3 = this->vbTmp3.template Get<float>();
    this->ubGate = this->vbGate.template Get<float>();
    this->ubNReset = this->vbNReset.template Get<float>();
    this->ubNUpdate = this->vbNUpdate.template Get<float>();
    this->ubNNew = this->vbNNew.template Get<float>();
    this->ubNHN = this->vbNHN.template Get<float>();
    this->ubNHPrev = this->vbNHPrev.template Get<float>();
    // 复用闲置的预取 hPrev 槽位作 dy 预取视图(原 ubNHPrev 无消费者)
    this->ubNDy = this->ubNHPrev;
}

template <typename DTYPE>
__aicore__ inline int64_t GruGradKernel<DTYPE>::AlignDown16(int64_t v)
{
    const int64_t blk = static_cast<int64_t>(ggcube::CUBE_BLOCK);
    return v < blk ? blk : (v / blk) * blk;
}

template <typename DTYPE>
__aicore__ inline void GruGradKernel<DTYPE>::SwapTilingMN(TCubeTiling& t)
{
    int64_t v = t.M;
    t.M = t.N;
    t.N = v;
    v = t.singleCoreM;
    t.singleCoreM = t.singleCoreN;
    t.singleCoreN = v;
    v = t.baseM;
    t.baseM = t.baseN;
    t.baseN = v;
}

template <typename DTYPE>
__aicore__ inline void GruGradKernel<DTYPE>::BuildOnePlan(GGemmPlan& p, const TCubeTiling& t)
{
    p.M = t.M;
    p.N = t.N;
    p.K = t.Ka;
    p.singleCoreM = t.singleCoreM > 0 ? t.singleCoreM : t.M;
    p.singleCoreN = t.singleCoreN > 0 ? t.singleCoreN : t.N;
    p.usedCoreNum = t.usedCoreNum > 0 ? t.usedCoreNum : 1;
    p.mCoreLoop = (p.M + p.singleCoreM - 1) / p.singleCoreM;
    p.nCoreLoop = (p.N + p.singleCoreN - 1) / p.singleCoreN;

    int64_t mStep = (t.baseM > 0 && t.baseM <= p.singleCoreM) ? t.baseM : p.singleCoreM;
    int64_t nStep = (t.baseN > 0 && t.baseN <= p.singleCoreN) ? t.baseN : p.singleCoreN;
    // L0C 预算：mStep*nStep <= 64K 元素（fp32）
    nStep = nStep < ggcube::L0C_ELEMS / mStep ? nStep : ggcube::L0C_ELEMS / mStep;
    nStep = AlignDown16(nStep);
    mStep = mStep < ggcube::L0C_ELEMS / nStep ? mStep : ggcube::L0C_ELEMS / nStep;
    mStep = AlignDown16(mStep);
    p.mStep = mStep;
    p.nStep = nStep;

    constexpr int64_t dSize = static_cast<int64_t>(sizeof(DTYPE));
    constexpr int64_t c0 = static_cast<int64_t>(32 / sizeof(DTYPE));
    // ⚠ 片上预算封顶（L0A/L0B 各 64KB）——kChunk 推导需以 L0B 为准。
    // L0B 足迹按【分形数上取整】计：kFrac×nRep×16x16×dsize，其中 kFrac=CeilDiv(k,16)、
    // nRep=CeilDiv(n,16)——kChunk<16 仍占满一个 16 行 k 分形块（）。
    // 故 nStep 先按"kFrac=1 时"封顶：nStep16 ≤ L0B/(256×dsize)（fp32≤1024/fp16≤2048），
    // 再按剩余 kFrac 名额封 kChunk。
    const int64_t nRepCap = static_cast<int64_t>(ggcube::L0B_BYTES) /
                            (static_cast<int64_t>(ggcube::FRACT_ELEMS) * dSize); // 每分形 16x16 元素
    if (nStep > nRepCap * ggcube::CUBE_BLOCK) {
        nStep = nRepCap * ggcube::CUBE_BLOCK;
    }
    // nStep 下限 64（受 N/L0C/nRepCap 封顶）——窄 nStep 使 b 侧 ND2NZ 行宽降至 64B
    // （GM 突发效率差），且 nTile 数与 b 侧搬运 op 数等比膨胀（dx: nStep=16 → 8 nTile × 8 chunk
    // = 64 次 26KB；提升后 2 nTile × 8 chunk = 16 次 64KB）。
    if (nStep < ggcube::CUBE_BLOCK * 4) {
        int64_t nFloor = (p.N < ggcube::CUBE_BLOCK * 4) ?
                             ((p.N + ggcube::CUBE_BLOCK - 1) / ggcube::CUBE_BLOCK * ggcube::CUBE_BLOCK) :
                             ggcube::CUBE_BLOCK * 4;
        nFloor = nFloor < ggcube::L0C_ELEMS / mStep ? nFloor : ggcube::L0C_ELEMS / mStep;
        nFloor = nFloor < nRepCap * ggcube::CUBE_BLOCK ? nFloor : nRepCap * ggcube::CUBE_BLOCK;
        if (nFloor > nStep) {
            nStep = nFloor;
        }
    }
    p.nStep = nStep;
    const int64_t cubeBlk = static_cast<int64_t>(ggcube::CUBE_BLOCK);
    const int64_t mAl = (mStep + cubeBlk - 1) / cubeBlk * cubeBlk; // M 侧分形对齐
    const int64_t nAl = (nStep + cubeBlk - 1) / cubeBlk * cubeBlk; // N 侧分形对齐
    const int64_t nRep = (nStep + ggcube::CUBE_BLOCK - 1) / ggcube::CUBE_BLOCK;
    const int64_t kFracCap = static_cast<int64_t>(ggcube::L0B_BYTES) /
                             (static_cast<int64_t>(ggcube::FRACT_ELEMS) * dSize * nRep);
    int64_t kC = ggcube::L1_BUDGET / (dSize * (mAl + nAl));
    // L0A: kC * mAl * dSize <= 64KB（足迹对 kC 线性）
    kC = kC < ggcube::L0A_BYTES / (dSize * mAl) ? kC : ggcube::L0A_BYTES / (dSize * mAl);
    // L0B: kFrac(kC) * nRep * 256 * dSize <= 64KB
    kC = kC < kFracCap * ggcube::CUBE_BLOCK ? kC : kFracCap * ggcube::CUBE_BLOCK;
    kC = (kC / c0) * c0;
    if (kC < c0) {
        kC = c0;
    }
    if (kC > p.K) {
        kC = p.K;
    }
    if (kC < p.K) {
        kC = (kC / cubeBlk) * cubeBlk; // 尾块单独处理，非尾块保持分形对齐（兼 c0 对齐）
        if (kC < cubeBlk) {
            kC = cubeBlk;
        }
    }
    p.kChunk = kC;
    p.l1AElems = ggcube::CeilDiv(static_cast<uint32_t>(kC), ggcube::C0<DTYPE>()) *
                 ggcube::CeilAlign(static_cast<uint32_t>(mStep), ggcube::CUBE_BLOCK) * ggcube::C0<DTYPE>();
    p.l1BElems = ggcube::CeilDiv(static_cast<uint32_t>(nStep), ggcube::C0<DTYPE>()) *
                 ggcube::CeilAlign(static_cast<uint32_t>(kC), ggcube::CUBE_BLOCK) * ggcube::C0<DTYPE>();
    p.l0CElems = ggcube::CeilAlign(static_cast<uint32_t>(mStep), ggcube::CUBE_BLOCK) *
                 ggcube::CeilAlign(static_cast<uint32_t>(nStep), ggcube::CUBE_BLOCK);
}

template <typename DTYPE>
__aicore__ inline void GruGradKernel<DTYPE>::BuildPlans()
{
    // 带域强制 N 二分——fp32 定长且 H%128==0 时 dgatePlan 的 baseN/singleCoreN
    // 夹到 H/2（dgTileCnt_=2，per-band Fixpipe 提供 B0 提前信号点）。仅触 dgate 一路。
    // H/2≥64 满足 nStep 下限；H/2%16==0 满足对齐；kChunk 由预算自适应（数值容差内）。
    TCubeTiling dgateT = this->tiling->dgateMMParam;
    if constexpr (sizeof(DTYPE) == FLOAT_BYTES) {
        if (!this->tiling->isSeqLength && (this->H % BAND_H_ALIGN) == 0 && this->H >= BAND_H_ALIGN) {
            const int64_t halfN = this->H / BAND_NUM; // 带宽 = H 二分
            if (dgateT.baseN > halfN) {
                dgateT.baseN = halfN;
            }
            if (dgateT.singleCoreN > halfN) {
                dgateT.singleCoreN = halfN;
            }
        }
    }
    BuildOnePlan(this->dgatePlan, dgateT);
    { // dW^T = xT @ dGi: M/N 交换消费 host tiling
        TCubeTiling t = this->tiling->dwIhMMParam;
        SwapTilingMN(t);
        BuildOnePlan(this->dwIhPlan, t);
    }
    {
        TCubeTiling t = this->tiling->dwHhMMParam;
        SwapTilingMN(t);
        BuildOnePlan(this->dwHhPlan, t);
    }
    BuildOnePlan(this->dxPlan, this->tiling->dxMMParam);

    // 片上 buffer 取四路计划的最大值，一次分配
    int64_t l1A = dgatePlan.l1AElems;
    l1A = l1A > dwIhPlan.l1AElems ? l1A : dwIhPlan.l1AElems;
    l1A = l1A > dwHhPlan.l1AElems ? l1A : dwHhPlan.l1AElems;
    l1A = l1A > dxPlan.l1AElems ? l1A : dxPlan.l1AElems;
    int64_t l1B = dgatePlan.l1BElems;
    l1B = l1B > dwIhPlan.l1BElems ? l1B : dwIhPlan.l1BElems;
    l1B = l1B > dwHhPlan.l1BElems ? l1B : dwHhPlan.l1BElems;
    l1B = l1B > dxPlan.l1BElems ? l1B : dxPlan.l1BElems;
    int64_t l0C = dgatePlan.l0CElems;
    l0C = l0C > dwIhPlan.l0CElems ? l0C : dwIhPlan.l0CElems;
    l0C = l0C > dwHhPlan.l0CElems ? l0C : dwHhPlan.l0CElems;
    l0C = l0C > dxPlan.l0CElems ? l0C : dxPlan.l0CElems;

    this->l1AElems_ = l1A;
    this->l1BElems_ = l1B;
    this->l0CElems_ = l0C;
    this->l0AElems_ = l1A;
    this->l0BElems_ = l1B;
}

#endif // _GRU_GRAD_INIT_ARCH35_
