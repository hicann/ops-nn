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
 * \file gru_grad.h
 * \brief
 */

#ifndef _GRU_GRAD_KERNEL_H_
#define _GRU_GRAD_KERNEL_H_

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "gru_grad_tiling_data.h"
using namespace AscendC;

constexpr int64_t GRU_GATE_SIZE = 3;
constexpr int64_t FLOAT_BYTES = 4;
constexpr int64_t DEFAULT_UB_BUF_ELEMENTS = 4096;
constexpr int64_t ALIGN_32B_FP32_MASK = 7;
constexpr int64_t ALIGN_32B_FP32 = 8;
constexpr int64_t GATE_IDX_NEW = 2;
constexpr int64_t AIV_PER_AIC = 2;
// 跨核事件同步: mode 2 = sub-block (AIC↔AIV 配对), flag 为硬件事件 ID
constexpr uint8_t SYNC_MODE2 = 2;
constexpr uint16_t SYNC_AIV_AIC_FLAG = 6;
constexpr uint16_t SYNC_AIC_AIV_FLAG = 8;

struct GRnnOffsets {
    int64_t AOffset{0};
    int64_t BOffset{0};
    int64_t COffset{0};
};

struct GRnnTail {
    int64_t tailSingleCoreM{0};
    int64_t tailSingleCoreN{0};
    int64_t notTailMCoreCount{0};
    int64_t notTailNCoreCount{0};
    int32_t mCoreLoop{0};
    int32_t nCoreLoop{0};
    int64_t mCoreIndx{0};
    int64_t nCoreIndx{0};
};

template <typename DTYPE>
class GruGradKernel {
public:
    __aicore__ inline GruGradKernel() = default;

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR w_input, GM_ADDR w_hidden, GM_ADDR init_h, GM_ADDR output_h,
                                GM_ADDR reset_gate, GM_ADDR update_gate, GM_ADDR new_gate, GM_ADDR h_n, GM_ADDR dy,
                                GM_ADDR dh, GM_ADDR batch_sizes, GM_ADDR dx, GM_ADDR dh_prev, GM_ADDR dw_input,
                                GM_ADDR dw_hidden, GM_ADDR db_input, GM_ADDR db_hidden,
                                const GruGradTilingData* __restrict tiling, GM_ADDR workspace)
    {
        this->tiling = tiling;
        this->T = tiling->timeStep;
        this->B = tiling->batch;
        this->H = tiling->hiddenSize;
        this->I = tiling->inputSize;
        this->dgateMMTiling = tiling->dgateMMParam;
        this->dwIhMMTiling = tiling->dwIhMMParam;
        this->dwHhMMTiling = tiling->dwHhMMParam;
        this->dxMMTiling = tiling->dxMMParam;

        this->InitGlobalBuffers(x, w_input, w_hidden, init_h, output_h, reset_gate, update_gate, new_gate, h_n, dy, dh,
                                batch_sizes, dx, dh_prev, dw_input, dw_hidden, db_input, db_hidden, workspace);
        this->InitVectorBuf();
    }

    __aicore__ inline void Process()
    {
        if (GetBlockIdx() < this->dgateMMTiling.usedCoreNum) {
            this->dgateMM.SetTensorB(this->inputGm.wHiddenGm[this->dgateOffsets.BOffset], false);
            if (!this->tiling->isSeqLength) {
                this->ApplyTail(this->dgateMM, this->dgateMMTiling, this->dgateTail);
            }
        }

        this->InitDhPrev();

        for (int64_t tIdx = this->T - 1; tIdx >= 0; tIdx--) {
            this->ProcessVector(tIdx);
            SyncAll();
            this->ProcessDgateMM(tIdx);
            SyncAll();
            this->AccumulateDhPrev(tIdx);
        }
        this->StoreDhPrev();
        this->ProcessDwIhMM();
        this->ProcessDwHhMM();
        this->ProcessDxMM();
        if (this->tiling->isBias == 1) {
            int64_t rows = this->totalSteps_;
            int64_t cols = this->H * GRU_GATE_SIZE;
            this->ProcessBiasReduce(this->workGm.dGiGm, this->outputGm.dbInputGm, rows, cols);
            this->ProcessBiasReduce(this->workGm.dGhGm, this->outputGm.dbHiddenGm, rows, cols);
        }
    }

    TPipe pipe;

    matmul::Matmul<matmul::MatmulType<TPosition::GM, CubeFormat::ND, DTYPE>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, DTYPE>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, DTYPE>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>>
        dgateMM;

    matmul::Matmul<matmul::MatmulType<TPosition::GM, CubeFormat::ND, DTYPE, true>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, DTYPE>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, DTYPE>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>>
        dwIhMM;

    matmul::Matmul<matmul::MatmulType<TPosition::GM, CubeFormat::ND, DTYPE, true>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, DTYPE>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, DTYPE>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>>
        dwHhMM;

    // MM3: dx = d_gi × w_input^T  [T*B,3H]×[3H,I]=[T*B,I]
    matmul::Matmul<matmul::MatmulType<TPosition::GM, CubeFormat::ND, DTYPE>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, DTYPE>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, DTYPE>,
                   matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>>
        dxMM;

private:
    struct InputGm {
        __aicore__ inline InputGm() = default;
        GlobalTensor<DTYPE> xGm;
        GlobalTensor<DTYPE> wInputGm;
        GlobalTensor<DTYPE> wHiddenGm;
        GlobalTensor<DTYPE> outputHGm;
        GlobalTensor<DTYPE> dyGm;
        GlobalTensor<DTYPE> dhGm;
        GlobalTensor<DTYPE> initHGm;
        GlobalTensor<DTYPE> resetGateGm;
        GlobalTensor<DTYPE> updateGateGm;
        GlobalTensor<DTYPE> newGateGm;
        GlobalTensor<DTYPE> hNGm;
        GlobalTensor<int64_t> batchSizesGm;
    };
    struct OutputGm {
        __aicore__ inline OutputGm() = default;
        GlobalTensor<DTYPE> dxGm;
        GlobalTensor<DTYPE> dhPrevGm;
        GlobalTensor<DTYPE> dwInputGm;
        GlobalTensor<DTYPE> dwHiddenGm;
        GlobalTensor<DTYPE> dbInputGm;
        GlobalTensor<DTYPE> dbHiddenGm;
    };
    struct WorkGm {
        __aicore__ inline WorkGm() = default;
        GlobalTensor<DTYPE> dGhGm;
        GlobalTensor<DTYPE> dGiGm;
        GlobalTensor<DTYPE> hPrevWsGm;
        GlobalTensor<DTYPE> xRevWsGm;
        GlobalTensor<DTYPE> dhPrevWsGm;
        GlobalTensor<DTYPE> dhFromHGm;
    };

    InputGm inputGm;
    OutputGm outputGm;
    WorkGm workGm;

    TBuf<TPosition::VECCALC> vbGradH, vbHPrev, vbReset, vbUpdate, vbNew, vbHN, vbDHPrevFromH;
    TBuf<TPosition::VECCALC> vbTmp, vbTmp2, vbTmp3;
    TBuf<TPosition::VECCALC> vbGate;
    LocalTensor<float> ubGradH, ubHPrev, ubReset, ubUpdate, ubNew, ubHN, ubDHPrevFromH;
    LocalTensor<float> ubTmp, ubTmp2, ubTmp3;
    LocalTensor<float> ubGate;

    GRnnOffsets dgateOffsets;
    GRnnOffsets dwIhOffsets;
    GRnnOffsets dwHhOffsets;
    GRnnOffsets dxOffsets;
    GRnnTail dgateTail;
    GRnnTail dwIhTail;
    GRnnTail dwHhTail;
    GRnnTail dxTail;

    const GruGradTilingData* __restrict tiling{nullptr};
    int64_t T{0};
    int64_t B{0};
    int64_t H{0};
    int64_t I{0};
    TCubeTiling dgateMMTiling;
    TCubeTiling dwIhMMTiling;
    TCubeTiling dwHhMMTiling;
    TCubeTiling dxMMTiling;
    int64_t vecBTile_{0};
    int64_t vecHTile_{0};
    int64_t allocLength{DEFAULT_UB_BUF_ELEMENTS};
    int64_t hAligned{0};
    int64_t totalSteps_{0};
    __aicore__ inline int64_t Ceil(int64_t x, int64_t y) { return (y == 0) ? x : (x + y - 1) / y; }

    __aicore__ inline int64_t GetCompactRowOffset(int64_t seqIdx)
    {
        int64_t offset = 0;
        for (int64_t i = 0; i < seqIdx; i++) {
            offset += this->inputGm.batchSizesGm.GetValue(i);
        }
        return offset;
    }

    __aicore__ inline void CalcGMOffsetForTransA(TCubeTiling& param, GRnnOffsets& off, GRnnTail& t, int64_t kSize)
    {
        t.mCoreLoop = static_cast<int32_t>(this->Ceil(param.M, param.singleCoreM));
        t.nCoreLoop = static_cast<int32_t>(this->Ceil(param.N, param.singleCoreN));

        int64_t blockIdx = GetBlockIdx();
        t.mCoreIndx = (t.mCoreLoop <= 1) ? 0 : (blockIdx / t.nCoreLoop);
        t.nCoreIndx = (t.nCoreLoop <= 1) ? 0 : (blockIdx % t.nCoreLoop);

        t.tailSingleCoreM = param.M - (t.mCoreLoop - 1) * param.singleCoreM;
        t.tailSingleCoreN = param.N - (t.nCoreLoop - 1) * param.singleCoreN;
        t.notTailMCoreCount = t.mCoreLoop - 1;
        t.notTailNCoreCount = t.nCoreLoop - 1;

        off.AOffset = t.mCoreIndx * param.singleCoreM;
        off.BOffset = t.nCoreIndx * param.singleCoreN;
        off.COffset = t.mCoreIndx * param.N * param.singleCoreM + t.nCoreIndx * param.singleCoreN;
    }

    __aicore__ inline void CalcGMOffset(TCubeTiling& param, GRnnOffsets& off, GRnnTail& t, int64_t kSize)
    {
        t.mCoreLoop = static_cast<int32_t>(this->Ceil(param.M, param.singleCoreM));
        t.nCoreLoop = static_cast<int32_t>(this->Ceil(param.N, param.singleCoreN));

        int64_t blockIdx = GetBlockIdx();
        t.mCoreIndx = (t.mCoreLoop <= 1) ? 0 : (blockIdx / t.nCoreLoop);
        t.nCoreIndx = (t.nCoreLoop <= 1) ? 0 : (blockIdx % t.nCoreLoop);

        t.tailSingleCoreM = param.M - (t.mCoreLoop - 1) * param.singleCoreM;
        t.tailSingleCoreN = param.N - (t.nCoreLoop - 1) * param.singleCoreN;
        t.notTailMCoreCount = t.mCoreLoop - 1;
        t.notTailNCoreCount = t.nCoreLoop - 1;

        off.AOffset = t.mCoreIndx * param.singleCoreM * kSize;
        off.BOffset = t.nCoreIndx * param.singleCoreN;
        off.COffset = t.mCoreIndx * param.N * param.singleCoreM + t.nCoreIndx * param.singleCoreN;
    }

    __aicore__ inline void ApplyTail(matmul::Matmul<matmul::MatmulType<TPosition::GM, CubeFormat::ND, DTYPE>,
                                                    matmul::MatmulType<TPosition::GM, CubeFormat::ND, DTYPE>,
                                                    matmul::MatmulType<TPosition::GM, CubeFormat::ND, DTYPE>,
                                                    matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>>& mm,
                                     TCubeTiling& param, GRnnTail& t)
    {
        if (t.nCoreIndx == t.notTailNCoreCount && t.mCoreIndx == t.notTailMCoreCount) {
            mm.SetTail(t.tailSingleCoreM, t.tailSingleCoreN);
        } else if (t.nCoreIndx == t.notTailNCoreCount) {
            mm.SetTail(param.singleCoreM, t.tailSingleCoreN);
        } else if (t.mCoreIndx == t.notTailMCoreCount) {
            mm.SetTail(t.tailSingleCoreM, param.singleCoreN);
        }
    }

    __aicore__ inline void ApplyTailTrans(matmul::Matmul<matmul::MatmulType<TPosition::GM, CubeFormat::ND, DTYPE, true>,
                                                         matmul::MatmulType<TPosition::GM, CubeFormat::ND, DTYPE>,
                                                         matmul::MatmulType<TPosition::GM, CubeFormat::ND, DTYPE>,
                                                         matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>>& mm,
                                          TCubeTiling& param, GRnnTail& t)
    {
        if (t.nCoreIndx == t.notTailNCoreCount && t.mCoreIndx == t.notTailMCoreCount) {
            mm.SetTail(t.tailSingleCoreM, t.tailSingleCoreN);
        } else if (t.nCoreIndx == t.notTailNCoreCount) {
            mm.SetTail(param.singleCoreM, t.tailSingleCoreN);
        } else if (t.mCoreIndx == t.notTailMCoreCount) {
            mm.SetTail(t.tailSingleCoreM, param.singleCoreN);
        }
    }

    __aicore__ inline void InitWorkBuffers(__gm__ DTYPE* ws)
    {
        int64_t off = 0;
        int64_t TS = totalSteps_;
        this->workGm.dGhGm.SetGlobalBuffer(ws + off, TS * H * GRU_GATE_SIZE);
        off += TS * H * GRU_GATE_SIZE;
        this->workGm.dGiGm.SetGlobalBuffer(ws + off, TS * H * GRU_GATE_SIZE);
        off += TS * H * GRU_GATE_SIZE;
        this->workGm.hPrevWsGm.SetGlobalBuffer(ws + off, TS * H);
        off += TS * H;
        this->workGm.xRevWsGm.SetGlobalBuffer(ws + off, TS * I);
        off += TS * I;
        this->workGm.dhPrevWsGm.SetGlobalBuffer(ws + off, B * H);
        off += B * H;
        this->workGm.dhFromHGm.SetGlobalBuffer(ws + off, B * H);
    }

    __aicore__ inline void InitGlobalBuffers(GM_ADDR x, GM_ADDR w_input, GM_ADDR w_hidden, GM_ADDR init_h,
                                             GM_ADDR output_h, GM_ADDR reset_gate, GM_ADDR update_gate,
                                             GM_ADDR new_gate, GM_ADDR h_n, GM_ADDR dy, GM_ADDR dh, GM_ADDR batch_sizes,
                                             GM_ADDR dx, GM_ADDR dh_prev, GM_ADDR dw_input, GM_ADDR dw_hidden,
                                             GM_ADDR db_input, GM_ADDR db_hidden, GM_ADDR workspace)
    {
        auto blockSize = 32 / sizeof(DTYPE);
        hAligned = ((H + blockSize - 1)) / blockSize * blockSize;

        this->vecBTile_ = this->tiling->singleCoreM;
        this->vecHTile_ = this->tiling->singleCoreN;

        this->totalSteps_ = this->tiling->totalSteps;

        this->CalcGMOffset(this->dgateMMTiling, this->dgateOffsets, this->dgateTail, H * GRU_GATE_SIZE);
        this->CalcGMOffsetForTransA(this->dwIhMMTiling, this->dwIhOffsets, this->dwIhTail, totalSteps_);
        this->CalcGMOffsetForTransA(this->dwHhMMTiling, this->dwHhOffsets, this->dwHhTail, totalSteps_);
        this->CalcGMOffset(this->dxMMTiling, this->dxOffsets, this->dxTail, H * GRU_GATE_SIZE);

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

        auto ws = reinterpret_cast<__gm__ DTYPE*>(workspace);
        this->InitWorkBuffers(ws);
    }

    // MM1: grad_h_prev = d_gh × w_hh  [curBatch,3H]×[3H,H]=[curBatch,H]
    __aicore__ inline void ProcessDgateMM(int64_t tIdx)
    {
        if (GetBlockIdx() >= this->dgateMMTiling.usedCoreNum) {
            return;
        }

        bool isReverse = (this->tiling->direction != 0);
        int64_t gateIdx = isReverse ? (this->T - 1 - tIdx) : tIdx;
        int64_t tAOff = 0;

        if (this->tiling->isSeqLength) {
            int64_t compactRow = this->GetCompactRowOffset(gateIdx);
            tAOff = compactRow * this->H * GRU_GATE_SIZE;
        } else {
            tAOff = gateIdx * this->B * this->H * GRU_GATE_SIZE;
        }

        this->dgateMM.SetTensorA(this->workGm.dGhGm[tAOff + this->dgateOffsets.AOffset], false);

        if (this->tiling->isSeqLength) {
            int64_t curBatch = this->inputGm.batchSizesGm.GetValue(gateIdx);
            int64_t coreStartM = this->dgateTail.mCoreIndx * this->dgateMMTiling.singleCoreM;
            if (coreStartM >= curBatch) {
                return; // 该核完全越界, 直接退出
            }
            int64_t actualM = this->dgateMMTiling.singleCoreM;
            if (coreStartM + actualM > curBatch) {
                actualM = curBatch - coreStartM;
            }
            int64_t tailN = (this->dgateTail.nCoreIndx == this->dgateTail.notTailNCoreCount) ?
                                this->dgateTail.tailSingleCoreN :
                                this->dgateMMTiling.singleCoreN;
            this->dgateMM.SetTail(actualM, tailN);
        }

        this->dgateMM.IterateAll(this->workGm.dhPrevWsGm[this->dgateOffsets.COffset], false);
    }

    // MM2a: dw_ih = d_gi^T @  x [3H,TB]×[TB, I]=[3H, I]
    __aicore__ inline void ProcessDwIhMM()
    {
        if (GetBlockIdx() >= this->dwIhMMTiling.usedCoreNum) {
            return;
        }
        this->dwIhMM.SetTensorA(this->workGm.dGiGm[this->dwIhOffsets.AOffset], true);
        this->dwIhMM.SetTensorB(this->inputGm.xGm[this->dwIhOffsets.BOffset], false);
        this->ApplyTailTrans(this->dwIhMM, this->dwIhMMTiling, this->dwIhTail);
        this->dwIhMM.IterateAll(this->outputGm.dwInputGm[this->dwIhOffsets.COffset], false);
    }

    // MM2b: dw_hh = d_gh^T @ h_prev  [3H,TB]×[TB,H]=[3H,H]
    __aicore__ inline void ProcessDwHhMM()
    {
        if (GetBlockIdx() >= this->dwHhMMTiling.usedCoreNum) {
            return;
        }
        this->dwHhMM.SetTensorA(this->workGm.dGhGm[this->dwHhOffsets.AOffset], true);
        this->dwHhMM.SetTensorB(this->workGm.hPrevWsGm[this->dwHhOffsets.BOffset], false);
        this->ApplyTailTrans(this->dwHhMM, this->dwHhMMTiling, this->dwHhTail);
        this->dwHhMM.IterateAll(this->outputGm.dwHiddenGm[this->dwHhOffsets.COffset], false);
    }

    // MM3: dx = d_gi @ w_ih^T  [TB,3H]×[3H,I]=[TB,I]
    __aicore__ inline void ProcessDxMM()
    {
        if (GetBlockIdx() >= this->dxMMTiling.usedCoreNum) {
            return;
        }
        this->dxMM.SetTensorA(this->workGm.dGiGm[this->dxOffsets.AOffset], false);
        this->dxMM.SetTensorB(this->inputGm.wInputGm[this->dxOffsets.BOffset], false);
        this->ApplyTail(this->dxMM, this->dxMMTiling, this->dxTail);
        this->dxMM.IterateAll(this->outputGm.dxGm[this->dxOffsets.COffset], false);
    }

    __aicore__ inline void InitVectorBuf()
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
    }

    __aicore__ inline bool GetCoreRows(int64_t& mStart, int64_t& mCnt)
    {
        int64_t blockDim = GetBlockNum() * AIV_PER_AIC;
        int64_t B_per_core = (this->B + blockDim - 1) / blockDim;
        int64_t rem = this->B % blockDim;
        int64_t blockIdx = GetBlockIdx();
        if (rem == 0) {
            mCnt = B_per_core;
            mStart = blockIdx * B_per_core;
        } else if (blockIdx < rem) {
            mCnt = B_per_core;
            mStart = blockIdx * B_per_core;
        } else {
            mCnt = B_per_core - 1;
            mStart = rem * B_per_core + (blockIdx - rem) * (B_per_core - 1);
        }
        return (mCnt > 0 && mStart < this->B);
    }

#include "gru_grad_vector.h"
};

#endif
