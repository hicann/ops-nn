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
        for (int64_t tIdx = this->T - 1; tIdx >= 0; tIdx--) {
            SyncAll();
            this->ProcessVector(tIdx);
            SyncAll();
            this->ProcessDgateMM(tIdx);
            SyncAll();
            this->AccumulateDhPrev();
        }
        SyncAll();
        this->StoreDhPrev();
        SyncAll();
        this->ProcessDwIhMM();
        SyncAll();
        this->ProcessDwHhMM();
        SyncAll();
        this->ProcessDxMM();
        SyncAll();
        if (this->tiling->isBias == 1) {
            int64_t TB = this->T * this->B;
            int64_t cols = this->H * GRU_GATE_SIZE;
            this->ProcessBiasReduce(this->workGm.dGiGm, this->outputGm.dbInputGm, TB, cols);
            SyncAll();
            this->ProcessBiasReduce(this->workGm.dGhGm, this->outputGm.dbHiddenGm, TB, cols);
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

    // MM3: dx = d_gi × w_input^T  [T*B,3H]×[3H,I]=[T*B,I]  -> 直接写回 dx 输出
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
    __aicore__ inline int64_t Ceil(int64_t x, int64_t y) { return (y == 0) ? x : (x + y - 1) / y; }

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

        this->CalcGMOffset(this->dgateMMTiling, this->dgateOffsets, this->dgateTail, H * GRU_GATE_SIZE);
        this->CalcGMOffsetForTransA(this->dwIhMMTiling, this->dwIhOffsets, this->dwIhTail, T * B);
        this->CalcGMOffsetForTransA(this->dwHhMMTiling, this->dwHhOffsets, this->dwHhTail, T * B);
        this->CalcGMOffset(this->dxMMTiling, this->dxOffsets, this->dxTail, H * GRU_GATE_SIZE);

        this->inputGm.xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE*>(x), T * B * I);
        this->inputGm.wInputGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE*>(w_input), I * H * GRU_GATE_SIZE);
        this->inputGm.wHiddenGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE*>(w_hidden), H * H * GRU_GATE_SIZE);
        this->inputGm.outputHGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE*>(output_h), T * B * H);
        this->inputGm.dyGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE*>(dy), T * B * H);
        this->inputGm.dhGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE*>(dh), B * H);
        this->inputGm.initHGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE*>(init_h), B * H);
        this->inputGm.resetGateGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE*>(reset_gate), T * B * H);
        this->inputGm.updateGateGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE*>(update_gate), T * B * H);
        this->inputGm.newGateGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE*>(new_gate), T * B * H);
        this->inputGm.hNGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE*>(h_n), T * B * H);

        this->outputGm.dxGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE*>(dx), T * B * I);
        this->outputGm.dhPrevGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE*>(dh_prev), B * H);
        this->outputGm.dwInputGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE*>(dw_input), I * H * GRU_GATE_SIZE);
        this->outputGm.dwHiddenGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE*>(dw_hidden), H * H * GRU_GATE_SIZE);
        if (this->tiling->isBias == 1) {
            this->outputGm.dbInputGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE*>(db_input), H * GRU_GATE_SIZE);
            this->outputGm.dbHiddenGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE*>(db_hidden), H * GRU_GATE_SIZE);
        }

        auto ws = reinterpret_cast<__gm__ DTYPE*>(workspace);
        int64_t off = 0;
        int64_t TB = T * B;
        this->workGm.dGhGm.SetGlobalBuffer(ws + off, TB * H * GRU_GATE_SIZE);
        off += TB * H * GRU_GATE_SIZE;
        this->workGm.dGiGm.SetGlobalBuffer(ws + off, TB * H * GRU_GATE_SIZE);
        off += TB * H * GRU_GATE_SIZE;
        this->workGm.hPrevWsGm.SetGlobalBuffer(ws + off, TB * H);
        off += TB * H;
        this->workGm.xRevWsGm.SetGlobalBuffer(ws + off, TB * I);
        off += TB * I;
        this->workGm.dhPrevWsGm.SetGlobalBuffer(ws + off, B * H);
        off += B * H;
        this->workGm.dhFromHGm.SetGlobalBuffer(ws + off, B * H);
    }

    // MM1: grad_h_prev = d_gh × w_hh  [B,3H]×[3H,H]=[B,H]
    __aicore__ inline void ProcessDgateMM(int64_t tIdx)
    {
        if (GetBlockIdx() >= this->dgateMMTiling.usedCoreNum) {
            return;
        }

        bool isReverse = (this->tiling->direction != 0);
        int64_t gateIdx = isReverse ? (this->T - 1 - tIdx) : tIdx;
        int64_t tAOff = gateIdx * this->B * this->H * GRU_GATE_SIZE;

        this->dgateMM.SetTensorA(this->workGm.dGhGm[tAOff + this->dgateOffsets.AOffset], false);
        this->dgateMM.SetTensorB(this->inputGm.wHiddenGm[this->dgateOffsets.BOffset], false);
        this->ApplyTail(this->dgateMM, this->dgateMMTiling, this->dgateTail);

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
            PipeBarrier<PIPE_ALL>();
            AscendC::Cast(ub, tmpUb[allocLength], AscendC::RoundMode::CAST_NONE, allocLength);
            PipeBarrier<PIPE_ALL>();
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
            PipeBarrier<PIPE_ALL>();
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

    // grad_h_t  = dy[t] + dh_next
    // dn(d_new) = grad_h_t * (1 - z)
    // dz_raw    = grad_h_t * (hp - n)
    // dh_prev_from_h = grad_h_t * z
    // d_i_new   = dn * (1 - n^2)
    // d_reset   = (d_i_new * h_n) * r * (1-r)
    // d_update  = dz_raw * z * (1 - z)
    // d_gi = [d_reset, d_update, d_i_new]
    // d_gh = [d_reset, d_update, d_i_new*r]
    __aicore__ inline void ProcessVector(int64_t tIdx)
    {
        const int64_t bTile = this->vecBTile_, hTile = this->vecHTile_;

        const bool isReverse = (this->tiling->direction != 0);
        const int64_t gateIdx = isReverse ? (this->T - 1 - tIdx) : tIdx;
        const int64_t revT = this->T - 1 - tIdx;
        const int64_t gm3H = this->H * GRU_GATE_SIZE;
        const int64_t gmGateStride = this->H * GATE_IDX_NEW;

        int64_t mStart = 0, mCnt = 0;
        if (!this->GetCoreRows(mStart, mCnt)) {
            return;
        }
        int64_t bTiles = (mCnt + bTile - 1) / bTile;
        int64_t hTiles = (H + hTile - 1) / hTile;
        int64_t bOff = 0, hOff = 0, iOff = 0;
        for (int64_t bt = 0; bt < bTiles; bt++) {
            int64_t bRows = bTile;
            bOff = bt * bTile;
            if (bt == bTiles - 1)
                bRows = mCnt - bOff;

            for (int64_t ht = 0; ht < hTiles; ht++) {
                int64_t hLen = hTile;
                hOff = ht * hTile;
                if (ht == hTiles - 1)
                    hLen = H - hOff;
                int64_t hAligned = ((hLen + ALIGN_32B_FP32_MASK) / ALIGN_32B_FP32) * ALIGN_32B_FP32;
                int64_t blkAligned = bRows * hAligned;
                int64_t ghOff = gateIdx * B * H + (mStart + bOff) * H + hOff;
                int64_t gmOffGi = gateIdx * B * gm3H + (mStart + bOff) * gm3H + hOff;

                CopyInRow(this->inputGm.dyGm, this->ubGradH, ghOff, 0, bRows, hLen, blkAligned, 0);
                if (tIdx == T - 1) {
                    CopyInRow(this->inputGm.dhGm, this->ubTmp, (mStart + bOff) * H + hOff, 0, bRows, hLen, blkAligned,
                              0);
                } else {
                    CopyInRow(this->workGm.dhPrevWsGm, this->ubTmp, (mStart + bOff) * H + hOff, 0, bRows, hLen,
                              blkAligned, 0);
                }

                PipeBarrier<PIPE_ALL>();
                Add(this->ubGradH, this->ubGradH, this->ubTmp, blkAligned);

                CopyInRow(this->inputGm.updateGateGm, this->ubUpdate, ghOff, 0, bRows, hLen, blkAligned, 0);

                CopyInRow(this->inputGm.newGateGm, this->ubNew, ghOff, 0, bRows, hLen, blkAligned, 0);

                CopyInRow(this->inputGm.resetGateGm, this->ubReset, ghOff, 0, bRows, hLen, blkAligned, 0);

                CopyInRow(this->inputGm.hNGm, this->ubHN, ghOff, 0, bRows, hLen, blkAligned, 0);

                if (tIdx == 0) {
                    CopyInRow(this->inputGm.initHGm, this->ubHPrev, (mStart + bOff) * H + hOff, 0, bRows, hLen,
                              blkAligned, 0);
                } else {
                    int64_t prevIdx = isReverse ? (this->T - tIdx) : (tIdx - 1);
                    CopyInRow(this->inputGm.outputHGm, this->ubHPrev, prevIdx * B * H + (mStart + bOff) * H + hOff, 0,
                              bRows, hLen, blkAligned, 0);
                }

                PipeBarrier<PIPE_ALL>();
                Duplicate(this->ubGate, 1.0f, blkAligned);
                Sub(this->ubTmp, this->ubGate, this->ubUpdate, blkAligned);

                Mul(this->ubTmp2, this->ubGradH, this->ubTmp, blkAligned);

                Mul(this->ubTmp, this->ubNew, this->ubNew, blkAligned);

                Sub(this->ubTmp, this->ubGate, this->ubTmp, blkAligned);

                Mul(this->ubDHPrevFromH, this->ubTmp2, this->ubTmp, blkAligned);
                PipeBarrier<PIPE_ALL>();

                CopyOutRow(this->workGm.dGiGm, this->ubDHPrevFromH, gmOffGi + GATE_IDX_NEW * H, 0, bRows, hLen,
                           gmGateStride, blkAligned);
                PipeBarrier<PIPE_ALL>();

                Mul(this->ubTmp, this->ubDHPrevFromH, this->ubHN, blkAligned);

                Mul(this->ubTmp2, this->ubTmp, this->ubReset, blkAligned);

                Sub(this->ubTmp, this->ubGate, this->ubReset, blkAligned);

                Mul(this->ubTmp3, this->ubTmp2, this->ubTmp, blkAligned);

                PipeBarrier<PIPE_ALL>();

                CopyOutRow(this->workGm.dGiGm, this->ubTmp3, gmOffGi, 0, bRows, hLen, gmGateStride, blkAligned);

                CopyOutRow(this->workGm.dGhGm, this->ubTmp3, gmOffGi, 0, bRows, hLen, gmGateStride, blkAligned);
                PipeBarrier<PIPE_ALL>();

                Sub(this->ubTmp, this->ubHPrev, this->ubNew, blkAligned);

                Mul(this->ubTmp2, this->ubGradH, this->ubTmp, blkAligned);

                Mul(this->ubTmp, this->ubTmp2, this->ubUpdate, blkAligned);

                Sub(this->ubTmp2, this->ubGate, this->ubUpdate, blkAligned);

                Mul(this->ubTmp3, this->ubTmp, this->ubTmp2, blkAligned);

                PipeBarrier<PIPE_ALL>();

                CopyOutRow(this->workGm.dGiGm, this->ubTmp3, gmOffGi + H, 0, bRows, hLen, gmGateStride, blkAligned);

                CopyOutRow(this->workGm.dGhGm, this->ubTmp3, gmOffGi + H, 0, bRows, hLen, gmGateStride, blkAligned);
                PipeBarrier<PIPE_ALL>();

                Mul(this->ubTmp3, this->ubDHPrevFromH, this->ubReset, blkAligned);

                PipeBarrier<PIPE_ALL>();

                CopyOutRow(this->workGm.dGhGm, this->ubTmp3, gmOffGi + GATE_IDX_NEW * H, 0, bRows, hLen, gmGateStride,
                           blkAligned);
                PipeBarrier<PIPE_ALL>();

                Mul(this->ubDHPrevFromH, this->ubGradH, this->ubUpdate, blkAligned);
                PipeBarrier<PIPE_ALL>();
                CopyOutRow(this->workGm.dhFromHGm, this->ubDHPrevFromH, (mStart + bOff) * H + hOff, 0, bRows, hLen, 0,
                           blkAligned);
                PipeBarrier<PIPE_ALL>();
            }

            for (int64_t ht = 0; ht < hTiles; ht++) {
                int64_t hLen = hTile;
                hOff = ht * hTile;
                if (ht == hTiles - 1)
                    hLen = H - hOff;
                int64_t hAligned = ((hLen + ALIGN_32B_FP32_MASK) / ALIGN_32B_FP32) * ALIGN_32B_FP32;
                int64_t blkAligned = bRows * hAligned;

                Duplicate(this->ubTmp2, 0.0f, blkAligned);
                PipeBarrier<PIPE_ALL>();
                if (tIdx == 0) {
                    CopyInRow(this->inputGm.initHGm, this->ubTmp2, (mStart + bOff) * H + hOff, 0, bRows, hLen,
                              blkAligned, 0);
                } else {
                    int64_t prevIdx = isReverse ? (this->T - tIdx) : (tIdx - 1);
                    CopyInRow(this->inputGm.outputHGm, this->ubTmp2, prevIdx * B * H + (mStart + bOff) * H + hOff, 0,
                              bRows, hLen, blkAligned, 0);
                }

                PipeBarrier<PIPE_ALL>();
                CopyOutRow(this->workGm.hPrevWsGm, this->ubTmp2, (gateIdx * B + mStart + bOff) * H + hOff, 0, bRows,
                           hLen, 0, blkAligned);
                PipeBarrier<PIPE_ALL>();
            }
        }
    }

    // grad_h_prev = dgateMM_result + dh_prev_from_h
    __aicore__ inline void AccumulateDhPrev()
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
                hAligned = ((hLen + ALIGN_32B_FP32_MASK) / ALIGN_32B_FP32) * ALIGN_32B_FP32;
                PipeBarrier<PIPE_ALL>();
                CopyInRow(this->workGm.dhPrevWsGm, this->ubTmp, (mStart + bOff) * H + hOff, 0, bRows, hLen, hAligned,
                          0);
                CopyInRow(this->workGm.dhFromHGm, this->ubTmp2, (mStart + bOff) * H + hOff, 0, bRows, hLen, hAligned,
                          0);
                PipeBarrier<PIPE_ALL>();
                Add(this->ubTmp3, this->ubTmp, this->ubTmp2, bRows * hAligned);
                PipeBarrier<PIPE_ALL>();
                CopyOutRow(this->workGm.dhPrevWsGm, this->ubTmp3, (mStart + bOff) * H + hOff, 0, bRows, hLen, 0,
                           hAligned);
                PipeBarrier<PIPE_ALL>();
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

                CopyInRow(this->workGm.dhPrevWsGm, this->ubTmp, (mStart + bOff) * H + hOff, 0, bRows, hLen, hAligned,
                          0);
                PipeBarrier<PIPE_ALL>();
                CopyOutRow(this->outputGm.dhPrevGm, this->ubTmp, (mStart + bOff) * H + hOff, 0, bRows, hLen, 0,
                           hAligned);
                PipeBarrier<PIPE_ALL>();
            }
        }
    }

    __aicore__ inline void ProcessBiasReduce(GlobalTensor<DTYPE>& srcGm, GlobalTensor<DTYPE>& dstGm, int64_t rows,
                                             int64_t cols)
    {
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
            for (int64_t r = 0; r < rows; ++r) {
                CopyInRow(srcGm, this->ubTmp2, r * cols + cGlobalOff, 0, 1, cCnt, cAligned, 0);
                PipeBarrier<PIPE_ALL>();
                Add(this->ubTmp, this->ubTmp, this->ubTmp2, cAligned);
                PipeBarrier<PIPE_ALL>();
            }
            CopyOutRow(dstGm, this->ubTmp, cGlobalOff, 0, 1, cCnt, 0, cAligned);
            PipeBarrier<PIPE_ALL>();
        }
    }
};

#endif
