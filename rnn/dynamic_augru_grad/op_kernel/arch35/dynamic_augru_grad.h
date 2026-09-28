/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the License).
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file dynamic_augru_grad.h
 * \brief DynamicAUGRUGrad kernel实现（ascend950/arch35，regbase范式）
 *
 * 单kernel按时间步倒序（T-1 -> 0）执行BPTT：每步向量阶段计算各门梯度并写出
 * dGi/dGh/hPrev/dhFromH，cube阶段做 dhPrevWs = dGh[t] @ w_hidden^T；
 * seq_length掩码在kernel内在线生成（不物化mask张量）。循环结束后收尾：
 * dh_prev输出、dw_input/dw_hidden/dx三个matmul、db_input/db_hidden列归约。
 * cube路径统一fp32，fp16在kernel边界做Cast；workspace按HPad=Ceil(H,16)*16
 * padded布局，pad区恒0对matmul无贡献（支持任意H/I）。
 */

#ifndef __DYNAMIC_AUGRU_GRAD_H__
#define __DYNAMIC_AUGRU_GRAD_H__

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "kernel_tiling/kernel_tiling.h"
#include "dynamic_augru_grad_tiling_data.h"

namespace NsDynamicAUGRUGrad {

using namespace AscendC;

constexpr int64_t GATE_NUM = 3;
constexpr int64_t FP32_BYTES = 4;
constexpr int64_t B32_BYTES = 32;        // MTE搬运32B对齐粒度（多块blockLen须为其倍数）
constexpr int64_t FP32_ALIGN = 8;        // fp32向量32B块对齐元素数
constexpr int64_t B32_REPEAT_ELEMS = 64; // 一个向量repeat的fp32元素数(256B)
constexpr int64_t MAX_REDUCE_REPEAT = 248;
constexpr int64_t MAX_ROW_ACC = 512; // bTile上限
constexpr int64_t REDUCE_TMP_ROWS = 64;
constexpr int64_t MAX_UB_SEQ_BYTES = 65536; // seq_length本核分片UB上限（超限回退GM直读）
// n门（候选门）槽位：zrh/rzh下z/r互换，n恒为第3段（与canndev参考实现一致）
constexpr int64_t SLOT_NEW_GATE = 2;

struct MMOffsets {
    int64_t aOffset = 0;
    int64_t bOffset = 0;
    int64_t cOffset = 0;
};

struct MMTail {
    int64_t mLoop = 1;
    int64_t nLoop = 1;
    int64_t mIdx = 0;
    int64_t nIdx = 0;
    int64_t tailM = 0;
    int64_t tailN = 0;
};

struct VectorTileParams {
    int64_t t;
    int64_t bStart;
    int64_t bRows;
    int64_t hOff;
    int64_t hLen;
    int64_t hAligned;
    int64_t tbhOff;
    int64_t bhOff;
    int64_t giRowOff;
};

// fp32统一cube类型
using MMAtype = matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>;
using MMAtypeTrans = matmul::MatmulType<TPosition::GM, CubeFormat::ND, float, true>;
using MMBiasType = matmul::MatmulType<TPosition::GM, CubeFormat::ND, float>;

template <typename DTYPE>
class DynamicAUGRUGradKernel {
public:
    static constexpr bool IS_FP16 = sizeof(DTYPE) == sizeof(half);
    static constexpr bool IS_FP32 = sizeof(DTYPE) == sizeof(float);

    __aicore__ inline DynamicAUGRUGradKernel() = default;

    TPipe pipe;

    // MM1: dh = dgate_h[t] @ w_hidden^T, A=[B,3H]不转置, B=[H,3H]转置
    matmul::Matmul<MMAtype, matmul::MatmulType<TPosition::GM, CubeFormat::ND, float, true>, MMAtype, MMBiasType>
        dgateMM;

    // MM2: dw_input = x^T @ dgate_x, A=x[TB,I]转置, B=dGi[TB,3H]不转置
    matmul::Matmul<MMAtypeTrans, MMAtype, MMAtype, MMBiasType> dwInputMM;

    // MM3: dw_hidden = h_prev^T @ dgate_h, A=hPrev[TB,H]转置, B=dGh[TB,3H]不转置
    matmul::Matmul<MMAtypeTrans, MMAtype, MMAtype, MMBiasType> dwHiddenMM;

    // MM4: dx = dgate_x @ w_input^T, A=dGi[TB,3H]不转置, B=wInput[I,3H]转置
    matmul::Matmul<MMAtype, matmul::MatmulType<TPosition::GM, CubeFormat::ND, float, true>, MMAtype, MMBiasType> dxMM;

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weightInput, GM_ADDR weightHidden, GM_ADDR weightAtt, GM_ADDR y,
                                GM_ADDR initH, GM_ADDR h, GM_ADDR dy, GM_ADDR dh, GM_ADDR update, GM_ADDR updateAtt,
                                GM_ADDR reset, GM_ADDR newGate, GM_ADDR hiddenNew, GM_ADDR seqLength, GM_ADDR mask,
                                GM_ADDR dwInput, GM_ADDR dwHidden, GM_ADDR dbInput, GM_ADDR dbHidden, GM_ADDR dx,
                                GM_ADDR dhPrev, GM_ADDR dwAtt, GM_ADDR workspace,
                                const DynamicAUGRUGradTilingData* tilingData)
    {
        (void)y;
        (void)mask;
        InitTilingParameters(tilingData);
        BindInputBuffers(x, weightInput, weightHidden, weightAtt, initH, h, dy, dh, update, updateAtt, reset, newGate,
                         hiddenNew, seqLength);
        BindOutputBuffers(dwInput, dwHidden, dbInput, dbHidden, dx, dhPrev, dwAtt);
        InitWorkspaceBuffers(x, weightInput, weightHidden, dwInput, dwHidden, dx, workspace, tilingData);
        CalcMMOffsets();
        InitUbBuffers();
    }

    // K维pad行清零：dGh/dGi的[tb,tbPad)行、hPrev的[tb,tbPad)行
    __aicore__ inline void ZeroKPadRows()
    {
        int64_t padRows = tbPad_ - tb_;
        if (padRows <= 0) {
            return;
        }
        ZeroKPadRegion(dGhGm_, padRows, threeHPad_);
        ZeroKPadRegion(dGiGm_, padRows, threeHPad_);
        ZeroKPadRegion(hPrevGm_, padRows, hPad_);
    }

    // 单区域清零：[start+tb*stride起)的rows行×stride列，按核分片
    __aicore__ inline void ZeroKPadRegion(GlobalTensor<float>& gm, int64_t rows, int64_t stride)
    {
        int64_t total = rows * stride;
        int64_t start = static_cast<int64_t>(GetBlockIdx()) * ubLength_;
        int64_t startBase = tb_ * stride;
        int64_t step = static_cast<int64_t>(GetBlockNum()) * ubLength_;
        for (; start < total; start += step) {
            int64_t cnt = Min(ubLength_, total - start);
            int64_t cntAligned = Ceil(cnt, FP32_ALIGN) * FP32_ALIGN;
            Duplicate(ubTmp2_, 0.0f, static_cast<int32_t>(cntAligned));
            PipeBarrier<PIPE_V>();
            CopyOutTileF(gm, ubTmp2_, start + startBase, 1, cnt, cnt, cntAligned);
        }
    }

    // seq_length本核batch分片GM->UB（一次性），后续按行UB标量读（替代GM标量读）
    __aicore__ inline void LoadSeqToUb()
    {
        if (!useUbSeq_) {
            return;
        }
        DataCopyExtParams cp(1, static_cast<uint32_t>(coreMCnt_ * static_cast<int64_t>(sizeof(int32_t))), 0, 0, 0);
        DataCopyPadExtParams<int32_t> pp(true, 0, 0, 0);
        DataCopyPad(ubSeq_, seqLenGm_[coreMStart_], cp, pp);
        PipeBarrier<PIPE_ALL>(); // 标量读可见性：等MTE2落UB
    }

    __aicore__ inline void Process()
    {
        LoadSeqToUb();
        ZeroKPadRows();
        ZeroSlotPadColumns();
        if (dbInline_) {
            ZeroDbPartials();
            Duplicate(ubDbGi_, 0.0f, static_cast<int32_t>(GATE_NUM * hPad_));
            Duplicate(ubDbGh_, 0.0f, static_cast<int32_t>(GATE_NUM * hPad_));
            Duplicate(ubDbGiCorrection_, 0.0f, static_cast<int32_t>(GATE_NUM * hPad_));
            Duplicate(ubDbGhCorrection_, 0.0f, static_cast<int32_t>(GATE_NUM * hPad_));
            PipeBarrier<PIPE_V>();
        }
        if (needWsCopy_ || ((mmChunkMask_ & MM_CHUNK_DW_INPUT) != 0)) {
            // fp16或H非对齐：先做权重/输入副本的fp32化+pad（cube路径统一fp32/padded）；
            // fp32+MM2分块：x的padded副本（含pad行清零）
            CastInputToFp32();
            SyncAll();
        }
        // 尾MM的SetTensor必须在BPTT写出dGi/dGh之后（写入前SetTensor会读到陈旧数据）
        if (GetBlockIdx() < dgateMMTiling_.usedCoreNum) {
            dgateMM.SetTensorB(wHiddenFp32Gm_[dgateOff_.bOffset], true);
        }

        InitDhPrev();
        for (int64_t t = T_ - 1; t >= 0; t--) {
            // 流水模式下除首步外，操作数已在上一时间步MM窗口预取
            ProcessVector(t, pipeline_ && t < T_ - 1);
            SyncAll();
            if (pipeline_) {
                ProcessDgateMMPipeline(t);
            } else {
                ProcessDgateMM(t);
            }
            SyncAll();
        }
        // dhFromH折叠后循环内不再累加：收尾一次性写出dh_prev = mm(0) + dhFromH(0)
        FinalStoreDhPrev();
        // bias归约先于尾MM（向量读dGi/dGh，避开cube读后的潜在一致性问题）
        SyncAll();
        if (dbInline_) {
            // db内联：各核partial落盘 -> 跨核可见 -> 小表列归约（替代全量重读dGi/dGh）
            FinalDbPartial();
            SyncAll();
            FinalDbReduce(dbPartialGiGm_, dbInputGm_);
            FinalDbReduce(dbPartialGhGm_, dbHiddenGm_);
        } else {
            ProcessBiasReduce(dGiGm_, dbInputGm_);
            ProcessBiasReduce(dGhGm_, dbHiddenGm_);
        }
        ProcessDwInputMM();
        ProcessDwHiddenMM();
        ProcessDxMM();

        SyncAll();
        WriteBackOutputs();
        // 结尾排空MTE3：fp16队列路径的写出无逐次完成等待，退出前须等全部落盘
        SyncM3toV();
        PipeBarrier<PIPE_ALL>();
    }

private:
    __aicore__ inline void InitTilingParameters(const DynamicAUGRUGradTilingData* tilingData)
    {
        tiling_ = tilingData;
        T_ = tilingData->timeStep;
        B_ = tilingData->batchSize;
        H_ = tilingData->hiddenSize;
        I_ = tilingData->inputSize;
        hPad_ = tilingData->hPad;
        bTile_ = tilingData->bTile;
        hTile_ = tilingData->hTile;
        ubLength_ = tilingData->ubLength;
        threeH_ = GATE_NUM * H_;
        threeHPad_ = GATE_NUM * hPad_;
        tb_ = T_ * B_;
        padH_ = (hPad_ != H_);
        mmChunkK_ = tilingData->mmChunkK;
        mmChunkMask_ = tilingData->mmChunkMask;
        tbPad_ = ((mmChunkMask_ & (MM_CHUNK_DW_INPUT | MM_CHUNK_DW_HIDDEN)) != 0) ?
                     Ceil(tb_, mmChunkK_) * mmChunkK_ :
                     Ceil(tb_, MM_DIM_ALIGN) * MM_DIM_ALIGN;
        zSlot_ = (tilingData->gateOrder == DYNAMIC_AUGRU_GRAD_GATE_ZRH) ? DYNAMIC_AUGRU_GRAD_GATE_ZRH :
                                                                          DYNAMIC_AUGRU_GRAD_GATE_RZH;
        rSlot_ = DYNAMIC_AUGRU_GRAD_GATE_ZRH + DYNAMIC_AUGRU_GRAD_GATE_RZH - zSlot_;
        hTiles_ = Ceil(H_, hTile_);
        coreValid_ = GetCoreRows(coreMStart_, coreMCnt_);
        pipeline_ = (tilingData->enablePipeline == 1) && IS_FP32 && (hTiles_ == 1) &&
                    (bTile_ >= Ceil(B_, static_cast<int64_t>(GetBlockNum()))) &&
                    (threeHPad_ <= Min(hPad_ / 2, MM_RECURRENT_CHUNK));
        fromHInUb_ = !pipeline_ && (hTiles_ == 1) && (coreMCnt_ <= bTile_);
    }

    __aicore__ inline void BindInputBuffers(GM_ADDR x, GM_ADDR weightInput, GM_ADDR weightHidden, GM_ADDR weightAtt,
                                            GM_ADDR initH, GM_ADDR h, GM_ADDR dy, GM_ADDR dh, GM_ADDR update,
                                            GM_ADDR updateAtt, GM_ADDR reset, GM_ADDR newGate, GM_ADDR hiddenNew,
                                            GM_ADDR seqLength)
    {
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE*>(x), tb_ * I_);
        wInputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE*>(weightInput), I_ * threeH_);
        wHiddenGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE*>(weightHidden), H_ * threeH_);
        attGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE*>(weightAtt), tb_ * H_);
        initHGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE*>(initH), B_ * H_);
        hGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE*>(h), tb_ * H_);
        dyGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE*>(dy), tb_ * H_);
        dhGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE*>(dh), B_ * H_);
        updateGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE*>(update), tb_ * H_);
        uAttGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE*>(updateAtt), tb_ * H_);
        resetGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE*>(reset), tb_ * H_);
        newGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE*>(newGate), tb_ * H_);
        hiddenNewGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE*>(hiddenNew), tb_ * H_);
        if (tiling_->isSeqLength == 1) {
            seqLenGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(seqLength), B_);
        }
    }

    __aicore__ inline void BindOutputBuffers(GM_ADDR dwInput, GM_ADDR dwHidden, GM_ADDR dbInput, GM_ADDR dbHidden,
                                             GM_ADDR dx, GM_ADDR dhPrev, GM_ADDR dwAtt)
    {
        dwInputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE*>(dwInput), I_ * threeH_);
        dwHiddenGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE*>(dwHidden), H_ * threeH_);
        dbInputGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE*>(dbInput), threeH_);
        dbHiddenGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE*>(dbHidden), threeH_);
        dxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE*>(dx), tb_ * I_);
        dhPrevGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE*>(dhPrev), B_ * H_);
        dwAttGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE*>(dwAtt), tb_);
    }

    __aicore__ inline void InitBaseWorkspace(__gm__ float* ws, int64_t& off)
    {
        dGhGm_.SetGlobalBuffer(ws + off, tbPad_ * threeHPad_);
        off += tbPad_ * threeHPad_;
        dGiGm_.SetGlobalBuffer(ws + off, tbPad_ * threeHPad_);
        off += tbPad_ * threeHPad_;
        hPrevGm_.SetGlobalBuffer(ws + off, tbPad_ * hPad_);
        off += tbPad_ * hPad_;
        dhPrevWsGm_.SetGlobalBuffer(ws + off, B_ * hPad_);
        off += B_ * hPad_;
        dhFromHGm_.SetGlobalBuffer(ws + off, B_ * H_);
        off += B_ * H_;
    }

    __aicore__ inline void InitInputWorkspace(__gm__ float* ws, int64_t& off, GM_ADDR x, GM_ADDR weightInput,
                                              GM_ADDR weightHidden)
    {
        needWsCopy_ = IS_FP16 || padH_;
        if (needWsCopy_) {
            wHiddenFp32Gm_.SetGlobalBuffer(ws + off, hPad_ * threeHPad_);
            off += hPad_ * threeHPad_;
            wInputFp32Gm_.SetGlobalBuffer(ws + off, I_ * threeHPad_);
            off += I_ * threeHPad_;
        } else {
            wHiddenFp32Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(weightHidden), H_ * threeH_);
            wInputFp32Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(weightInput), I_ * threeH_);
        }
        if constexpr (IS_FP16) {
            xFp32Gm_.SetGlobalBuffer(ws + off, tbPad_ * I_);
            off += tbPad_ * I_;
        } else if ((mmChunkMask_ & MM_CHUNK_DW_INPUT) != 0) {
            xFp32Gm_.SetGlobalBuffer(ws + off, tbPad_ * I_);
            off += tbPad_ * I_;
        } else {
            xFp32Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x), tb_ * I_);
        }
    }

    __aicore__ inline void InitOutputWorkspace(__gm__ float* ws, int64_t& off, GM_ADDR dwInput, GM_ADDR dwHidden,
                                               GM_ADDR dx)
    {
        if constexpr (IS_FP16) {
            dwInputFp32Gm_.SetGlobalBuffer(ws + off, I_ * threeHPad_);
            off += I_ * threeHPad_;
            dwHiddenFp32Gm_.SetGlobalBuffer(ws + off, hPad_ * threeHPad_);
            off += hPad_ * threeHPad_;
            dxFp32Gm_.SetGlobalBuffer(ws + off, tb_ * I_);
            off += tb_ * I_;
        } else {
            if (padH_) {
                dwInputFp32Gm_.SetGlobalBuffer(ws + off, I_ * threeHPad_);
                off += I_ * threeHPad_;
                dwHiddenFp32Gm_.SetGlobalBuffer(ws + off, hPad_ * threeHPad_);
                off += hPad_ * threeHPad_;
            } else {
                dwInputFp32Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dwInput), I_ * threeH_);
                dwHiddenFp32Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dwHidden), H_ * threeH_);
            }
            dxFp32Gm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dx), tb_ * I_);
        }
    }

    __aicore__ inline void InitReductionWorkspace(__gm__ float* ws, int64_t& off,
                                                  const DynamicAUGRUGradTilingData* tilingData)
    {
        dbInline_ = (tilingData->enableDbInline == 1);
        if (dbInline_) {
            int64_t dbRows = static_cast<int64_t>(GetBlockNum());
            dbPartialGiGm_.SetGlobalBuffer(ws + off, dbRows * threeH_);
            off += dbRows * threeH_;
            dbPartialGhGm_.SetGlobalBuffer(ws + off, dbRows * threeH_);
            off += dbRows * threeH_;
        }
        if (mmChunkMask_ & MM_CHUNK_DW_INPUT) {
            dwInputPartialGm_.SetGlobalBuffer(ws + off, I_ * threeHPad_);
            off += I_ * threeHPad_;
        }
        if (mmChunkMask_ & MM_CHUNK_DW_HIDDEN) {
            dwHiddenPartialGm_.SetGlobalBuffer(ws + off, hPad_ * threeHPad_);
            off += hPad_ * threeHPad_;
        }
        int64_t correctionRows = (mmChunkMask_ & MM_CHUNK_DW_INPUT) ? I_ : 0;
        if ((mmChunkMask_ & MM_CHUNK_DW_HIDDEN) && hPad_ > correctionRows) {
            correctionRows = hPad_;
        }
        if (correctionRows > 0) {
            weightCorrectionGm_.SetGlobalBuffer(ws + off, correctionRows * threeHPad_);
            off += correctionRows * threeHPad_;
        }
        int64_t projectionSize = (B_ * hPad_ > tb_ * I_) ? B_ * hPad_ : tb_ * I_;
        projectionPartialGm_.SetGlobalBuffer(ws + off, projectionSize);
        off += projectionSize;
        projectionCorrectionGm_.SetGlobalBuffer(ws + off, projectionSize);
    }

    __aicore__ inline void InitWorkspaceBuffers(GM_ADDR x, GM_ADDR weightInput, GM_ADDR weightHidden, GM_ADDR dwInput,
                                                GM_ADDR dwHidden, GM_ADDR dx, GM_ADDR workspace,
                                                const DynamicAUGRUGradTilingData* tilingData)
    {
        __gm__ float* ws = reinterpret_cast<__gm__ float*>(GetUserWorkspace(workspace));
        int64_t off = 0;
        InitBaseWorkspace(ws, off);
        InitInputWorkspace(ws, off, x, weightInput, weightHidden);
        InitOutputWorkspace(ws, off, dwInput, dwHidden, dx);
        InitReductionWorkspace(ws, off, tilingData);
    }

    __aicore__ inline void InitUbBuffers()
    {
        pipe.InitBuffer(vbGradH_, ubLength_ * FP32_BYTES);
        pipe.InitBuffer(vbHPrev_, ubLength_ * FP32_BYTES);
        pipe.InitBuffer(vbUpdate_, ubLength_ * FP32_BYTES);
        pipe.InitBuffer(vbUAtt_, ubLength_ * FP32_BYTES);
        pipe.InitBuffer(vbAtt_, ubLength_ * FP32_BYTES);
        pipe.InitBuffer(vbReset_, ubLength_ * FP32_BYTES);
        pipe.InitBuffer(vbNew_, ubLength_ * FP32_BYTES);
        pipe.InitBuffer(vbHN_, ubLength_ * FP32_BYTES);
        pipe.InitBuffer(vbTmp_, ubLength_ * FP32_BYTES);
        pipe.InitBuffer(vbFromH_, ubLength_ * FP32_BYTES);
        pipe.InitBuffer(vbDNT_, ubLength_ * FP32_BYTES);
        pipe.InitBuffer(vbTmp2_, ubLength_ * FP32_BYTES);
        pipe.InitBuffer(vbTmp3_, ubLength_ * FP32_BYTES);
        pipe.InitBuffer(vbDz_, ubLength_ * FP32_BYTES);
        pipe.InitBuffer(vbWa_, ubLength_ * FP32_BYTES);
        pipe.InitBuffer(vbRowAcc_, MAX_ROW_ACC * FP32_BYTES);
        pipe.InitBuffer(vbRowSum_, MAX_ROW_ACC * FP32_BYTES);
        pipe.InitBuffer(vbReduceTmp_, REDUCE_TMP_ROWS * B32_REPEAT_ELEMS * FP32_BYTES);
        if (dbInline_) {
            pipe.InitBuffer(vbDbGi_, GATE_NUM * hPad_ * FP32_BYTES);
            pipe.InitBuffer(vbDbGh_, GATE_NUM * hPad_ * FP32_BYTES);
            pipe.InitBuffer(vbDbGiCorrection_, GATE_NUM * hPad_ * FP32_BYTES);
            pipe.InitBuffer(vbDbGhCorrection_, GATE_NUM * hPad_ * FP32_BYTES);
            ubDbGi_ = vbDbGi_.Get<float>();
            ubDbGh_ = vbDbGh_.Get<float>();
            ubDbGiCorrection_ = vbDbGiCorrection_.Get<float>();
            ubDbGhCorrection_ = vbDbGhCorrection_.Get<float>();
        }
        if constexpr (IS_FP16) {
            pipe.InitBuffer(stagingInQueue_, 1, ubLength_ * static_cast<int64_t>(sizeof(DTYPE)));
            pipe.InitBuffer(stagingOutQueue_, 1, ubLength_ * static_cast<int64_t>(sizeof(DTYPE)));
        }
        InitSeqBuffer();
        AssignUbBuffers();
    }

    __aicore__ inline void AssignUbBuffers()
    {
        ubGradH_ = vbGradH_.Get<float>();
        ubHPrev_ = vbHPrev_.Get<float>();
        ubUpdate_ = vbUpdate_.Get<float>();
        ubUAtt_ = vbUAtt_.Get<float>();
        ubAtt_ = vbAtt_.Get<float>();
        ubReset_ = vbReset_.Get<float>();
        ubNew_ = vbNew_.Get<float>();
        ubHN_ = vbHN_.Get<float>();
        ubTmp_ = vbTmp_.Get<float>();
        ubFromH_ = vbFromH_.Get<float>();
        ubDNT_ = vbDNT_.Get<float>();
        ubTmp2_ = vbTmp2_.Get<float>();
        ubTmp3_ = vbTmp3_.Get<float>();
        ubDz_ = vbDz_.Get<float>();
        ubWa_ = vbWa_.Get<float>();
        ubRowAcc_ = vbRowAcc_.Get<float>();
        ubRowSum_ = vbRowSum_.Get<float>();
        ubReduceTmp_ = vbReduceTmp_.Get<float>();
    }

    __aicore__ inline void InitSeqBuffer()
    {
        useUbSeq_ = false;
        if (tiling_->isSeqLength != 1 || !coreValid_) {
            return;
        }
        int64_t bPerCore = Ceil(B_, static_cast<int64_t>(GetBlockNum()));
        if (bPerCore * static_cast<int64_t>(sizeof(int32_t)) <= MAX_UB_SEQ_BYTES) {
            pipe.InitBuffer(vbSeq_, Ceil(bPerCore, FP32_ALIGN) * FP32_ALIGN * static_cast<int64_t>(sizeof(int32_t)));
            ubSeq_ = vbSeq_.Get<int32_t>();
            useUbSeq_ = true;
        }
    }

    __aicore__ inline int64_t Ceil(int64_t a, int64_t b) { return (a + b - 1) / b; }

    __aicore__ inline int64_t Min(int64_t a, int64_t b) { return (a < b) ? a : b; }

    __aicore__ inline void SyncM2toV()
    {
        event_t eventId = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);
    }

    __aicore__ inline void SyncVtoM2Wait()
    {
        // MTE2等待V与MTE3完成：防止UB缓冲上一轮消费未结束时被新一轮载入覆盖
        event_t evV = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(evV);
        WaitFlag<HardEvent::V_MTE2>(evV);
        event_t evM3 = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(evM3);
        WaitFlag<HardEvent::MTE3_MTE2>(evM3);
    }

    __aicore__ inline void SyncVtoM3()
    {
        event_t eventId = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);
    }

    __aicore__ inline void SyncM3toV()
    {
        event_t eventId = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventId);
        WaitFlag<HardEvent::MTE3_V>(eventId);
    }

    // ---- GM(DTYPE, 可为fp16) [rows,len] -> UB fp32 ----
    // 注：950 MTE多块搬运在blockLen非32B对齐时异常（载入丢块/写出错位，真机实测），
    // 故多块非对齐时逐行拆分为单块拷贝
    __aicore__ inline void CopyInTileHalf(GlobalTensor<DTYPE>& gm, LocalTensor<float>& ub, int64_t gmOffset,
                                          int64_t rows, int64_t len, int64_t gmRowStride, int64_t ubRowStride)
    {
        if (rows > 1 && (len * sizeof(DTYPE)) % B32_BYTES != 0) {
            DataCopyExtParams cp(1, static_cast<uint32_t>(len * sizeof(DTYPE)),
                                 static_cast<uint32_t>((gmRowStride - len) * sizeof(DTYPE)),
                                 static_cast<uint32_t>((ubRowStride - len) * sizeof(DTYPE)), 0);
            DataCopyPadExtParams<DTYPE> pp(true, 0, 0, 0);
            for (int64_t r = 0; r < rows; r++) {
                SyncVtoM2Wait();
                LocalTensor<DTYPE> staging = stagingInQueue_.template AllocTensor<DTYPE>();
                DataCopyPad(staging, gm[gmOffset + r * gmRowStride], cp, pp);
                stagingInQueue_.EnQue(staging);
                LocalTensor<DTYPE> stagingReady = stagingInQueue_.template DeQue<DTYPE>();
                Cast(ub[r * ubRowStride], stagingReady, RoundMode::CAST_NONE, static_cast<int32_t>(ubRowStride));
                PipeBarrier<PIPE_V>();
                stagingInQueue_.FreeTensor(stagingReady);
            }
            return;
        }
        SyncVtoM2Wait();
        LocalTensor<DTYPE> staging = stagingInQueue_.template AllocTensor<DTYPE>();
        DataCopyExtParams cp(static_cast<uint16_t>(rows), static_cast<uint32_t>(len * sizeof(DTYPE)),
                             static_cast<uint32_t>((gmRowStride - len) * sizeof(DTYPE)),
                             static_cast<uint32_t>((ubRowStride - len) * sizeof(DTYPE)), 0);
        DataCopyPadExtParams<DTYPE> pp(true, 0, 0, 0);
        DataCopyPad(staging, gm[gmOffset], cp, pp);
        stagingInQueue_.EnQue(staging);
        LocalTensor<DTYPE> stagingReady = stagingInQueue_.template DeQue<DTYPE>();
        Cast(ub, stagingReady, RoundMode::CAST_NONE, static_cast<int32_t>(rows * ubRowStride));
        PipeBarrier<PIPE_V>();
        stagingInQueue_.FreeTensor(stagingReady);
    }

    __aicore__ inline void CopyInTileFloat(GlobalTensor<DTYPE>& gm, LocalTensor<float>& ub, int64_t gmOffset,
                                           int64_t rows, int64_t len, int64_t gmRowStride, int64_t ubRowStride)
    {
        if (rows > 1 && (len * FP32_BYTES) % B32_BYTES != 0) {
            DataCopyExtParams cp(1, static_cast<uint32_t>(len * FP32_BYTES),
                                 static_cast<uint32_t>((gmRowStride - len) * FP32_BYTES),
                                 static_cast<uint32_t>((ubRowStride - len) * FP32_BYTES), 0);
            DataCopyPadExtParams<DTYPE> pp(true, 0, 0, 0);
            for (int64_t r = 0; r < rows; r++) {
                SyncVtoM2Wait();
                DataCopyPad(ub[r * ubRowStride], gm[gmOffset + r * gmRowStride], cp, pp);
                SyncM2toV();
            }
            return;
        }
        SyncVtoM2Wait();
        DataCopyExtParams cp(static_cast<uint16_t>(rows), static_cast<uint32_t>(len * sizeof(DTYPE)),
                             static_cast<uint32_t>((gmRowStride - len) * sizeof(DTYPE)),
                             static_cast<uint32_t>((ubRowStride - len) * sizeof(DTYPE)), 0);
        DataCopyPadExtParams<DTYPE> pp(true, 0, 0, 0);
        DataCopyPad(ub, gm[gmOffset], cp, pp);
        SyncM2toV();
    }

    __aicore__ inline void CopyInTile(GlobalTensor<DTYPE>& gm, LocalTensor<float>& ub, int64_t gmOffset, int64_t rows,
                                      int64_t len, int64_t gmRowStride, int64_t ubRowStride)
    {
        if constexpr (IS_FP16) {
            CopyInTileHalf(gm, ub, gmOffset, rows, len, gmRowStride, ubRowStride);
        } else {
            CopyInTileFloat(gm, ub, gmOffset, rows, len, gmRowStride, ubRowStride);
        }
    }

    // ---- GM fp32 [rows,len] -> UB fp32（无同步，供批量装载组包） ----
    __aicore__ inline void CopyInTileFRaw(GlobalTensor<float>& gm, LocalTensor<float>& ub, int64_t gmOffset,
                                          int64_t rows, int64_t len, int64_t gmRowStride, int64_t ubRowStride)
    {
        if (rows > 1 && (len * FP32_BYTES) % B32_BYTES != 0) {
            DataCopyExtParams cp(1, static_cast<uint32_t>(len * FP32_BYTES),
                                 static_cast<uint32_t>((gmRowStride - len) * FP32_BYTES),
                                 static_cast<uint32_t>((ubRowStride - len) * FP32_BYTES), 0);
            DataCopyPadExtParams<float> pp(true, 0, 0, 0);
            for (int64_t r = 0; r < rows; r++) {
                DataCopyPad(ub[r * ubRowStride], gm[gmOffset + r * gmRowStride], cp, pp);
            }
            return;
        }
        DataCopyExtParams cp(static_cast<uint16_t>(rows), static_cast<uint32_t>(len * FP32_BYTES),
                             static_cast<uint32_t>((gmRowStride - len) * FP32_BYTES),
                             static_cast<uint32_t>((ubRowStride - len) * FP32_BYTES), 0);
        DataCopyPadExtParams<float> pp(true, 0, 0, 0);
        DataCopyPad(ub, gm[gmOffset], cp, pp);
    }

    // ---- GM fp32 [rows,len] -> UB fp32 ----
    __aicore__ inline void CopyInTileF(GlobalTensor<float>& gm, LocalTensor<float>& ub, int64_t gmOffset, int64_t rows,
                                       int64_t len, int64_t gmRowStride, int64_t ubRowStride)
    {
        SyncVtoM2Wait();
        CopyInTileFRaw(gm, ub, gmOffset, rows, len, gmRowStride, ubRowStride);
        SyncM2toV();
    }

    // ---- GM fp32 [rows,len] -> UB fp32[ubOffset起] ----
    __aicore__ inline void CopyInTileFOff(GlobalTensor<float>& gm, LocalTensor<float>& ub, int64_t ubOffset,
                                          int64_t gmOffset, int64_t rows, int64_t len, int64_t gmRowStride,
                                          int64_t ubRowStride)
    {
        SyncVtoM2Wait();
        if (rows > 1 && (len * FP32_BYTES) % B32_BYTES != 0) {
            DataCopyExtParams cp(1, static_cast<uint32_t>(len * FP32_BYTES),
                                 static_cast<uint32_t>((gmRowStride - len) * FP32_BYTES),
                                 static_cast<uint32_t>((ubRowStride - len) * FP32_BYTES), 0);
            DataCopyPadExtParams<float> pp(true, 0, 0, 0);
            for (int64_t r = 0; r < rows; r++) {
                DataCopyPad(ub[ubOffset + r * ubRowStride], gm[gmOffset + r * gmRowStride], cp, pp);
            }
            SyncM2toV();
            return;
        }
        DataCopyExtParams cp(static_cast<uint16_t>(rows), static_cast<uint32_t>(len * FP32_BYTES),
                             static_cast<uint32_t>((gmRowStride - len) * FP32_BYTES),
                             static_cast<uint32_t>((ubRowStride - len) * FP32_BYTES), 0);
        DataCopyPadExtParams<float> pp(true, 0, 0, 0);
        DataCopyPad(ub[ubOffset], gm[gmOffset], cp, pp);
        SyncM2toV();
    }

    // ---- GM(DTYPE) [rows,len] -> UB fp32[ubOffset起]（fp16经staging队列转） ----
    __aicore__ inline void CopyInTileOffT(GlobalTensor<DTYPE>& gm, LocalTensor<float>& ub, int64_t ubOffset,
                                          int64_t gmOffset, int64_t rows, int64_t len, int64_t gmRowStride,
                                          int64_t ubRowStride)
    {
        if (rows > 1 && (len * sizeof(DTYPE)) % B32_BYTES != 0) {
            DataCopyExtParams cp(1, static_cast<uint32_t>(len * sizeof(DTYPE)),
                                 static_cast<uint32_t>((gmRowStride - len) * sizeof(DTYPE)),
                                 static_cast<uint32_t>((ubRowStride - len) * sizeof(DTYPE)), 0);
            DataCopyPadExtParams<DTYPE> pp(true, 0, 0, 0);
            for (int64_t r = 0; r < rows; r++) {
                if constexpr (IS_FP16) {
                    SyncVtoM2Wait();
                    LocalTensor<DTYPE> staging = stagingInQueue_.template AllocTensor<DTYPE>();
                    DataCopyPad(staging, gm[gmOffset + r * gmRowStride], cp, pp);
                    stagingInQueue_.EnQue(staging);
                    LocalTensor<DTYPE> stagingReady = stagingInQueue_.template DeQue<DTYPE>();
                    Cast(ub[ubOffset + r * ubRowStride], stagingReady, RoundMode::CAST_NONE, static_cast<int32_t>(len));
                    PipeBarrier<PIPE_V>();
                    stagingInQueue_.FreeTensor(stagingReady);
                } else {
                    SyncVtoM2Wait();
                    DataCopyPad(ub[ubOffset + r * ubRowStride], gm[gmOffset + r * gmRowStride], cp, pp);
                    SyncM2toV();
                }
            }
            return;
        }
        if constexpr (IS_FP16) {
            SyncVtoM2Wait();
            LocalTensor<DTYPE> staging = stagingInQueue_.template AllocTensor<DTYPE>();
            DataCopyExtParams cp(static_cast<uint16_t>(rows), static_cast<uint32_t>(len * sizeof(DTYPE)),
                                 static_cast<uint32_t>((gmRowStride - len) * sizeof(DTYPE)),
                                 static_cast<uint32_t>((ubRowStride - len) * sizeof(DTYPE)), 0);
            DataCopyPadExtParams<DTYPE> pp(true, 0, 0, 0);
            DataCopyPad(staging, gm[gmOffset], cp, pp);
            stagingInQueue_.EnQue(staging);
            LocalTensor<DTYPE> stagingReady = stagingInQueue_.template DeQue<DTYPE>();
            Cast(ub[ubOffset], stagingReady, RoundMode::CAST_NONE, static_cast<int32_t>(rows * len));
            PipeBarrier<PIPE_V>();
            stagingInQueue_.FreeTensor(stagingReady);
        } else {
            DataCopyExtParams cp(static_cast<uint16_t>(rows), static_cast<uint32_t>(len * sizeof(DTYPE)),
                                 static_cast<uint32_t>((gmRowStride - len) * sizeof(DTYPE)),
                                 static_cast<uint32_t>((ubRowStride - len) * sizeof(DTYPE)), 0);
            SyncVtoM2Wait();
            DataCopyPadExtParams<DTYPE> pp(true, 0, 0, 0);
            DataCopyPad(ub[ubOffset], gm[gmOffset], cp, pp);
            SyncM2toV();
        }
    }

    // ---- UB fp32 -> GM(DTYPE, 可为fp16) [rows,len] ----
    __aicore__ inline void CopyOutTile(GlobalTensor<DTYPE>& gm, LocalTensor<float>& ub, int64_t gmOffset, int64_t rows,
                                       int64_t len, int64_t gmRowStride, int64_t ubRowStride)
    {
        if (rows > 1 && (len * sizeof(DTYPE)) % B32_BYTES != 0) {
            DataCopyExtParams cp(1, static_cast<uint32_t>(len * sizeof(DTYPE)),
                                 static_cast<uint32_t>((ubRowStride - len) * sizeof(DTYPE)),
                                 static_cast<uint32_t>((gmRowStride - len) * sizeof(DTYPE)), 0);
            for (int64_t r = 0; r < rows; r++) {
                if constexpr (IS_FP16) {
                    LocalTensor<DTYPE> staging = stagingOutQueue_.template AllocTensor<DTYPE>();
                    Cast(staging, ub[r * ubRowStride], RoundMode::CAST_RINT, static_cast<int32_t>(ubRowStride));
                    PipeBarrier<PIPE_V>();
                    stagingOutQueue_.EnQue(staging);
                    LocalTensor<DTYPE> stagingReady = stagingOutQueue_.template DeQue<DTYPE>();
                    DataCopyPad(gm[gmOffset + r * gmRowStride], stagingReady, cp);
                    stagingOutQueue_.FreeTensor(stagingReady);
                } else {
                    SyncVtoM3();
                    DataCopyPad(gm[gmOffset + r * gmRowStride], ub[r * ubRowStride], cp);
                    SyncM3toV();
                }
            }
            return;
        }
        if constexpr (IS_FP16) {
            LocalTensor<DTYPE> staging = stagingOutQueue_.template AllocTensor<DTYPE>();
            Cast(staging, ub, RoundMode::CAST_RINT, static_cast<int32_t>(rows * ubRowStride));
            PipeBarrier<PIPE_V>();
            stagingOutQueue_.EnQue(staging);
            LocalTensor<DTYPE> stagingReady = stagingOutQueue_.template DeQue<DTYPE>();
            DataCopyExtParams cp(static_cast<uint16_t>(rows), static_cast<uint32_t>(len * sizeof(DTYPE)),
                                 static_cast<uint32_t>((ubRowStride - len) * sizeof(DTYPE)),
                                 static_cast<uint32_t>((gmRowStride - len) * sizeof(DTYPE)), 0);
            DataCopyPad(gm[gmOffset], stagingReady, cp);
            stagingOutQueue_.FreeTensor(stagingReady);
        } else {
            SyncVtoM3();
            DataCopyExtParams cp(static_cast<uint16_t>(rows), static_cast<uint32_t>(len * sizeof(DTYPE)),
                                 static_cast<uint32_t>((ubRowStride - len) * sizeof(DTYPE)),
                                 static_cast<uint32_t>((gmRowStride - len) * sizeof(DTYPE)), 0);
            DataCopyPad(gm[gmOffset], ub, cp);
            SyncM3toV();
        }
    }

    // ---- UB fp32 -> GM fp32 [rows,len]（无同步，供批量写出组包） ----
    __aicore__ inline void CopyOutTileFRaw(GlobalTensor<float>& gm, LocalTensor<float>& ub, int64_t gmOffset,
                                           int64_t rows, int64_t len, int64_t gmRowStride, int64_t ubRowStride)
    {
        if (rows > 1 && (len * FP32_BYTES) % B32_BYTES != 0) {
            DataCopyExtParams cp(1, static_cast<uint32_t>(len * FP32_BYTES),
                                 static_cast<uint32_t>((ubRowStride - len) * FP32_BYTES),
                                 static_cast<uint32_t>((gmRowStride - len) * FP32_BYTES), 0);
            for (int64_t r = 0; r < rows; r++) {
                DataCopyPad(gm[gmOffset + r * gmRowStride], ub[r * ubRowStride], cp);
            }
            return;
        }
        DataCopyExtParams cp(static_cast<uint16_t>(rows), static_cast<uint32_t>(len * FP32_BYTES),
                             static_cast<uint32_t>((ubRowStride - len) * FP32_BYTES),
                             static_cast<uint32_t>((gmRowStride - len) * FP32_BYTES), 0);
        DataCopyPad(gm[gmOffset], ub, cp);
    }

    // ---- UB fp32 -> GM fp32 [rows,len] ----
    __aicore__ inline void CopyOutTileF(GlobalTensor<float>& gm, LocalTensor<float>& ub, int64_t gmOffset, int64_t rows,
                                        int64_t len, int64_t gmRowStride, int64_t ubRowStride)
    {
        SyncVtoM3();
        CopyOutTileFRaw(gm, ub, gmOffset, rows, len, gmRowStride, ubRowStride);
        SyncM3toV();
    }

    __aicore__ inline bool GetCoreRows(int64_t& mStart, int64_t& mCnt)
    {
        int64_t blockDim = GetBlockNum();
        int64_t bPerCore = Ceil(B_, blockDim);
        int64_t usedCores = Ceil(B_, bPerCore);
        int64_t blockIdx = GetBlockIdx();
        if (blockIdx >= usedCores) {
            mCnt = 0;
            return false;
        }
        mStart = blockIdx * bPerCore;
        mCnt = Min(bPerCore, B_ - mStart);
        return mCnt > 0;
    }

    // 行归约: src[rows, colAlign] -> dst[rows]
    // 容量约束：调用方保证rows*colAlign = blkAligned <= ubLength，故colAlign >
    // B32_REPEAT_ELEMS时rows <= ubLength/colAlign < REDUCE_TMP_ROWS，ubReduceTmp_
    // （REDUCE_TMP_ROWS行x64列）恒不越界；colAlign <= 64时走WholeReduceSum路径无此约束
    __aicore__ inline void ReduceRows(LocalTensor<float>& dst, LocalTensor<float>& src, int64_t rows, int64_t colAlign)
    {
        PipeBarrier<PIPE_V>();
        if (colAlign <= B32_REPEAT_ELEMS) {
            int32_t colBlkStride = static_cast<int32_t>(colAlign / FP32_ALIGN);
            int64_t loops = rows / MAX_REDUCE_REPEAT;
            int64_t rem = rows - loops * MAX_REDUCE_REPEAT;
            for (int64_t i = 0; i < loops; i++) {
                WholeReduceSum(dst[i * MAX_REDUCE_REPEAT], src[i * MAX_REDUCE_REPEAT * colAlign],
                               static_cast<int32_t>(colAlign), static_cast<int32_t>(MAX_REDUCE_REPEAT), 1, 1,
                               colBlkStride);
            }
            if (rem != 0) {
                WholeReduceSum(dst[loops * MAX_REDUCE_REPEAT], src[loops * MAX_REDUCE_REPEAT * colAlign],
                               static_cast<int32_t>(colAlign), static_cast<int32_t>(rem), 1, 1, colBlkStride);
            }
        } else {
            DataCopyParams copyParams;
            copyParams.blockCount = static_cast<uint16_t>(rows);
            copyParams.blockLen = static_cast<uint16_t>(B32_REPEAT_ELEMS / FP32_ALIGN);
            copyParams.srcStride = static_cast<uint16_t>(colAlign / FP32_ALIGN - B32_REPEAT_ELEMS / FP32_ALIGN);
            copyParams.dstStride = 0;
            DataCopy(ubReduceTmp_, src, copyParams);
            PipeBarrier<PIPE_V>();
            BinaryRepeatParams addParams;
            addParams.dstBlkStride = 1;
            addParams.src0BlkStride = 1;
            addParams.src1BlkStride = 1;
            addParams.dstRepStride = B32_REPEAT_ELEMS / FP32_ALIGN;
            addParams.src0RepStride = B32_REPEAT_ELEMS / FP32_ALIGN;
            addParams.src1RepStride = colAlign / FP32_ALIGN;
            int64_t colLoops = colAlign / B32_REPEAT_ELEMS;
            int64_t remCol = colAlign - colLoops * B32_REPEAT_ELEMS;
            for (int64_t i = 1; i < colLoops; i++) {
                Add(ubReduceTmp_, ubReduceTmp_, src[i * B32_REPEAT_ELEMS], B32_REPEAT_ELEMS, static_cast<int32_t>(rows),
                    addParams);
                PipeBarrier<PIPE_V>();
            }
            if (remCol != 0) {
                Add(ubReduceTmp_, ubReduceTmp_, src[colLoops * B32_REPEAT_ELEMS], remCol, static_cast<int32_t>(rows),
                    addParams);
                PipeBarrier<PIPE_V>();
            }
            WholeReduceSum(dst, ubReduceTmp_, B32_REPEAT_ELEMS, static_cast<int32_t>(rows), 1, 1,
                           B32_REPEAT_ELEMS / FP32_ALIGN);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void CalcMMTail(TCubeTiling& param, MMTail& t)
    {
        t.mLoop = Ceil(param.M, param.singleCoreM);
        t.nLoop = Ceil(param.N, param.singleCoreN);
        int64_t blockIdx = GetBlockIdx();
        t.mIdx = (t.mLoop <= 1) ? 0 : (blockIdx / t.nLoop);
        t.nIdx = (t.nLoop <= 1) ? 0 : (blockIdx % t.nLoop);
        t.tailM = param.M - (t.mLoop - 1) * param.singleCoreM;
        t.tailN = param.N - (t.nLoop - 1) * param.singleCoreN;
    }

    template <typename MMT>
    __aicore__ inline void ApplyMMTail(MMT& mm, TCubeTiling& param, MMTail& t)
    {
        if (t.nIdx == t.nLoop - 1 && t.mIdx == t.mLoop - 1) {
            mm.SetTail(t.tailM, t.tailN);
        } else if (t.nIdx == t.nLoop - 1) {
            mm.SetTail(param.singleCoreM, t.tailN);
        } else if (t.mIdx == t.mLoop - 1) {
            mm.SetTail(t.tailM, param.singleCoreN);
        }
    }

    __aicore__ inline void CalcMMOffsets()
    {
        dgateMMTiling_ = tiling_->dgateMMParam;
        dwInputMMTiling_ = tiling_->dwInputMMParam;
        dwHiddenMMTiling_ = tiling_->dwHiddenMMParam;
        dxMMTiling_ = tiling_->dxMMParam;

        CalcMMTail(dgateMMTiling_, dgateTail_);
        CalcMMTail(dwInputMMTiling_, dwInputTail_);
        CalcMMTail(dwHiddenMMTiling_, dwHiddenTail_);
        CalcMMTail(dxMMTiling_, dxTail_);

        // MM1: A=dGh[B,3HPad]不转置(行偏移*3HPad), B=wHiddenFp32[HPad,3HPad]转置(行偏移*3HPad，
        // fp32对齐模式别名原始[H,3H]张量), C=dhPrevWs[B,HPad](尾列垃圾，仅[0,H)列有效)
        dgateOff_.aOffset = dgateTail_.mIdx * dgateMMTiling_.singleCoreM * threeHPad_;
        dgateOff_.bOffset = dgateTail_.nIdx * dgateMMTiling_.singleCoreN * threeHPad_;
        dgateOff_.cOffset = dgateTail_.mIdx * hPad_ * dgateMMTiling_.singleCoreM +
                            dgateTail_.nIdx * dgateMMTiling_.singleCoreN;
        // MM2: A=xFp32[TB,I]转置(列偏移), B=dGi[TB,3HPad]不转置(列偏移), C=dwInputFp32[I,3HPad]
        dwInputOff_.aOffset = dwInputTail_.mIdx * dwInputMMTiling_.singleCoreM;
        dwInputOff_.bOffset = dwInputTail_.nIdx * dwInputMMTiling_.singleCoreN;
        dwInputOff_.cOffset = dwInputTail_.mIdx * threeHPad_ * dwInputMMTiling_.singleCoreM +
                              dwInputTail_.nIdx * dwInputMMTiling_.singleCoreN;
        // MM3: A=hPrev[TB,HPad]转置(列偏移), B=dGh[TB,3HPad]不转置(列偏移), C=dwHiddenFp32[HPad,3HPad](尾行垃圾)
        dwHiddenOff_.aOffset = dwHiddenTail_.mIdx * dwHiddenMMTiling_.singleCoreM;
        dwHiddenOff_.bOffset = dwHiddenTail_.nIdx * dwHiddenMMTiling_.singleCoreN;
        dwHiddenOff_.cOffset = dwHiddenTail_.mIdx * threeHPad_ * dwHiddenMMTiling_.singleCoreM +
                               dwHiddenTail_.nIdx * dwHiddenMMTiling_.singleCoreN;
        // MM4: A=dGi[TB,3HPad]不转置(行偏移*3HPad), B=wInputFp32[I,3HPad]转置(行偏移*3HPad), C=dxFp32[TB,I]
        dxOff_.aOffset = dxTail_.mIdx * dxMMTiling_.singleCoreM * threeHPad_;
        dxOff_.bOffset = dxTail_.nIdx * dxMMTiling_.singleCoreN * threeHPad_;
        dxOff_.cOffset = dxTail_.mIdx * I_ * dxMMTiling_.singleCoreM + dxTail_.nIdx * dxMMTiling_.singleCoreN;
    }

    // 权重padded副本：[rows,3H] -> [rowsPad,3HPad]，分槽分块拷入（支持任意H），
    // 槽尾与尾行pad区清零（matmul的N/K维pad区恒0无贡献）
    __aicore__ inline void CastWeightPadded(GlobalTensor<DTYPE>& src, GlobalTensor<float>& dst, int64_t dstRows,
                                            int64_t padRows)
    {
        int64_t hTail = hPad_ - H_;
        int64_t blockNum = GetBlockNum();
        for (int64_t r = GetBlockIdx(); r < padRows; r += blockNum) {
            if (r < dstRows) {
                int64_t srcRow = r * threeH_;
                int64_t dstRow = r * threeHPad_;
                for (int64_t slot = 0; slot < GATE_NUM; slot++) {
                    int64_t srcSlot = srcRow + slot * H_;
                    int64_t dstSlot = dstRow + slot * hPad_;
                    for (int64_t c0 = 0; c0 < H_; c0 += ubLength_) {
                        int64_t cnt = Min(ubLength_, H_ - c0);
                        int64_t cntAligned = Ceil(cnt, FP32_ALIGN) * FP32_ALIGN;
                        CopyInTileOffT(src, ubTmp3_, 0, srcSlot + c0, 1, cnt, threeH_, cntAligned);
                        CopyOutTileF(dst, ubTmp3_, dstSlot + c0, 1, cnt, threeHPad_, cntAligned);
                    }
                    if (hTail > 0) {
                        Duplicate(ubTmp3_, 0.0f, static_cast<int32_t>(hTail));
                        PipeBarrier<PIPE_V>();
                        CopyOutTileF(dst, ubTmp3_, dstSlot + H_, 1, hTail, hTail, hTail);
                    }
                }
            } else {
                int64_t dstRow = r * threeHPad_;
                for (int64_t c0 = 0; c0 < threeHPad_; c0 += ubLength_) {
                    int64_t cnt = Min(ubLength_, threeHPad_ - c0);
                    int64_t cntAligned = Ceil(cnt, FP32_ALIGN) * FP32_ALIGN;
                    Duplicate(ubTmp3_, 0.0f, static_cast<int32_t>(cntAligned));
                    PipeBarrier<PIPE_V>();
                    CopyOutTileF(dst, ubTmp3_, dstRow + c0, 1, cnt, threeHPad_, cntAligned);
                }
            }
        }
    }

    // fp16/padH模式：x/w_input/w_hidden -> fp32 padded副本；fp32+MM2分块时x亦需
    // padded副本（尾块K读到[tb,tbPad)清零行，不能别名输入张量）
    __aicore__ inline void CastInputToFp32()
    {
        int64_t total = tb_ * I_;
        int64_t chunk = bTile_ * hTile_;
        int64_t rows = Ceil(total, chunk);
        int64_t blockNum = GetBlockNum();
        bool copyX = false;
        if constexpr (IS_FP16) {
            copyX = true;
        } else {
            copyX = ((mmChunkMask_ & MM_CHUNK_DW_INPUT) != 0); // fp32仅分块时需副本
        }
        if (copyX) {
            for (int64_t blk = GetBlockIdx(); blk < rows; blk += blockNum) {
                int64_t start = blk * chunk;
                int64_t cnt = Min(chunk, total - start);
                if (cnt <= 0) {
                    continue;
                }
                int64_t cntAligned = Ceil(cnt, FP32_ALIGN) * FP32_ALIGN;
                if constexpr (IS_FP16) {
                    CopyInTile(xGm_, ubTmp_, start, 1, cnt, cnt, cntAligned);
                } else {
                    CopyInTileF(xGm_, ubTmp_, start, 1, cnt, cnt, cntAligned);
                }
                CopyOutTileF(xFp32Gm_, ubTmp_, start, 1, cnt, cnt, cntAligned);
            }
            // x副本pad行清零：[tb*I, tbPad*I)（分块尾块读到，恒0贡献）
            int64_t padTotal = tbPad_ * I_ - total;
            for (int64_t blk = GetBlockIdx(); blk < Ceil(padTotal, ubLength_); blk += blockNum) {
                int64_t start = total + blk * ubLength_;
                int64_t cnt = Min(ubLength_, padTotal - blk * ubLength_);
                int64_t cntAligned = Ceil(cnt, FP32_ALIGN) * FP32_ALIGN;
                Duplicate(ubTmp_, 0.0f, static_cast<int32_t>(cntAligned));
                PipeBarrier<PIPE_V>();
                CopyOutTileF(xFp32Gm_, ubTmp_, start, 1, cnt, cnt, cntAligned);
            }
        }
        if (needWsCopy_) {
            CastWeightPadded(wHiddenGm_, wHiddenFp32Gm_, H_, hPad_);
            CastWeightPadded(wInputGm_, wInputFp32Gm_, I_, I_);
        }
    }

    // 行回写：padded中转[rowsN,3HPad] -> 紧凑输出[rowsN,3H]。padH时逐槽拷贝
    // （整行连续拷会把pad区带进输出）。validRows为有效行数：dw_hidden的C按
    // M=HPad分配尾行垃圾不写回；dw_input的C按M=I分配全行有效（不可共用H_作
    // 守卫——I>H时会把dw_input的行[H,I)误跳过）
    __aicore__ inline void WriteBackRows(GlobalTensor<DTYPE>& dstGm, GlobalTensor<float>& srcGm, int64_t rowsN,
                                         int64_t validRows)
    {
        int64_t blockNum = GetBlockNum();
        if (!padH_) {
            for (int64_t r = GetBlockIdx(); r < rowsN; r += blockNum) {
                int64_t srcRow = r * threeHPad_;
                int64_t dstRow = r * threeH_;
                for (int64_t c0 = 0; c0 < threeH_; c0 += ubLength_) {
                    int64_t cnt = Min(ubLength_, threeH_ - c0);
                    int64_t cntAligned = Ceil(cnt, FP32_ALIGN) * FP32_ALIGN;
                    CopyInTileF(srcGm, ubTmp_, srcRow + c0, 1, cnt, threeHPad_, cntAligned);
                    CopyOutTile(dstGm, ubTmp_, dstRow + c0, 1, cnt, threeH_, cntAligned);
                }
            }
            return;
        }
        for (int64_t r = GetBlockIdx(); r < rowsN; r += blockNum) {
            if (r >= validRows) {
                break; // M=HPad时尾行[validRows,rowsN)为垃圾，不写回
            }
            int64_t srcRow = r * threeHPad_;
            int64_t dstRow = r * threeH_;
            for (int64_t slot = 0; slot < GATE_NUM; slot++) {
                int64_t srcSlot = srcRow + slot * hPad_;
                int64_t dstSlot = dstRow + slot * H_;
                for (int64_t c0 = 0; c0 < H_; c0 += ubLength_) {
                    int64_t cnt = Min(ubLength_, H_ - c0);
                    int64_t cntAligned = Ceil(cnt, FP32_ALIGN) * FP32_ALIGN;
                    CopyInTileF(srcGm, ubTmp_, srcSlot + c0, 1, cnt, threeHPad_, cntAligned);
                    CopyOutTile(dstGm, ubTmp_, dstSlot + c0, 1, cnt, threeH_, cntAligned);
                }
            }
        }
    }

    // 输出中转回写：fp16或padH时dw_input/dw_hidden从padded中转拷回紧凑输出，
    // fp16额外做fp32->half转换；fp32+H对齐时C直写输出张量无需回写
    __aicore__ inline void WriteBackOutputs()
    {
        if (needWsCopy_) {
            WriteBackRows(dwInputGm_, dwInputFp32Gm_, I_, I_);      // C按M=I分配，全行有效
            WriteBackRows(dwHiddenGm_, dwHiddenFp32Gm_, hPad_, H_); // C按M=HPad分配，尾行垃圾
        }
        if constexpr (IS_FP16) {
            int64_t total = tb_ * I_;
            int64_t chunk = bTile_ * hTile_;
            int64_t rows = Ceil(total, chunk);
            int64_t blockNum = GetBlockNum();
            for (int64_t blk = GetBlockIdx(); blk < rows; blk += blockNum) {
                int64_t start = blk * chunk;
                int64_t cnt = Min(chunk, total - start);
                if (cnt <= 0) {
                    continue;
                }
                int64_t cntAligned = Ceil(cnt, FP32_ALIGN) * FP32_ALIGN;
                CopyInTileF(dxFp32Gm_, ubTmp_, start, 1, cnt, cnt, cntAligned);
                CopyOutTile(dxGm_, ubTmp_, start, 1, cnt, cnt, cntAligned);
            }
        }
    }

    __aicore__ inline void InitDhPrev()
    {
        if (!coreValid_) {
            return;
        }
        int64_t mStart = coreMStart_;
        int64_t mCnt = coreMCnt_;
        for (int64_t bOff = 0; bOff < mCnt; bOff += bTile_) {
            int64_t bRows = Min(bTile_, mCnt - bOff);
            int64_t row = mStart + bOff;
            for (int64_t ht = 0; ht < hTiles_; ht++) {
                int64_t hOff = ht * hTile_;
                int64_t hLen = Min(hTile_, H_ - hOff);
                int64_t hAligned = Ceil(hLen, FP32_ALIGN) * FP32_ALIGN;
                CopyInTile(dhGm_, ubTmp_, row * H_ + hOff, bRows, hLen, H_, hAligned);
                CopyOutTileF(dhPrevWsGm_, ubTmp_, row * hPad_ + hOff, bRows, hLen, hPad_, hAligned);
            }
        }
    }

    // db内联：两张partial表各自按核分片清零（无行核的partial恒0）
    __aicore__ inline void ZeroDbPartials()
    {
        int64_t total = static_cast<int64_t>(GetBlockNum()) * threeH_;
        int64_t blockNum = GetBlockNum();
        for (int64_t start = static_cast<int64_t>(GetBlockIdx()) * ubLength_; start < total;
             start += blockNum * ubLength_) {
            int64_t cnt = Min(ubLength_, total - start);
            int64_t cntAligned = Ceil(cnt, FP32_ALIGN) * FP32_ALIGN;
            Duplicate(ubTmp2_, 0.0f, static_cast<int32_t>(cntAligned));
            PipeBarrier<PIPE_V>();
            CopyOutTileF(dbPartialGiGm_, ubTmp2_, start, 1, cnt, cnt, cntAligned);
            CopyOutTileF(dbPartialGhGm_, ubTmp2_, start, 1, cnt, cnt, cntAligned);
        }
    }

    // db内联：src块[rows,len]逐行列入累加器（列和；V内按序执行，无冒险）
    __aicore__ inline void DbAccumulate(LocalTensor<float>& acc, LocalTensor<float>& correction,
                                        LocalTensor<float>& src, int64_t accOff, int64_t rows, int64_t len)
    {
        for (int64_t r = 0; r < rows; r++) {
            AccumulatePartialVf(acc[accOff], src[r * len], correction[accOff], static_cast<int32_t>(len));
        }
    }

    // db内联：每核累加器[3槽×HPad] -> partial行[3H]紧凑（逐槽拷，跳过槽尾pad；
    // 无行核partial已在ZeroDbPartials清零）
    __aicore__ inline void FinalDbPartial()
    {
        if (!coreValid_) {
            return;
        }
        int64_t rowGi = static_cast<int64_t>(GetBlockIdx()) * threeH_;
        int64_t rowGh = rowGi; // 两表基址不同（各自SetGlobalBuffer），行内偏移一致
        for (int64_t slot = 0; slot < GATE_NUM; slot++) {
            LocalTensor<float> srcGi = ubDbGi_[slot * hPad_];
            LocalTensor<float> srcGh = ubDbGh_[slot * hPad_];
            CopyOutTileF(dbPartialGiGm_, srcGi, rowGi + slot * H_, 1, H_, H_, H_);
            CopyOutTileF(dbPartialGhGm_, srcGh, rowGh + slot * H_, 1, H_, H_, H_);
        }
    }

    // db内联收尾：各核partial[blockDim,3H]列归约 -> dst[3H]（紧凑无pad，列块切核）
    __aicore__ inline void FinalDbReduce(GlobalTensor<float>& srcGm, GlobalTensor<DTYPE>& dstGm)
    {
        int64_t rows = static_cast<int64_t>(GetBlockNum());
        int64_t cols = threeH_;
        int64_t blockDim = GetBlockNum();
        int64_t reduceChunk = tiling_->singleCoreReduceN;
        int64_t blockCnt = Ceil(cols, reduceChunk);
        for (int64_t blk = GetBlockIdx(); blk < blockCnt; blk += blockDim) {
            int64_t nStart = blk * reduceChunk;
            int64_t nCnt = Min(reduceChunk, cols - nStart);
            if (nCnt <= 0) {
                continue;
            }
            for (int64_t cStart = 0; cStart < nCnt; cStart += ubLength_) {
                int64_t cCnt = Min(ubLength_, nCnt - cStart);
                int64_t cAligned = Ceil(cCnt, FP32_ALIGN) * FP32_ALIGN;
                int64_t colOff = nStart + cStart;
                Duplicate(ubTmp_, 0.0f, static_cast<int32_t>(cAligned));
                Duplicate(ubTmp3_, 0.0f, static_cast<int32_t>(cAligned));
                PipeBarrier<PIPE_V>();
                int64_t maxRows = ubLength_ / cAligned;
                for (int64_t r0 = 0; r0 < rows; r0 += maxRows) {
                    int64_t rCnt = Min(maxRows, rows - r0);
                    CopyInTileF(srcGm, ubTmp2_, r0 * cols + colOff, rCnt, cCnt, cols, cAligned);
                    for (int64_t rr = 0; rr < rCnt; rr++) {
                        AccumulatePartialVf(ubTmp_, ubTmp2_[rr * cAligned], ubTmp3_, static_cast<int32_t>(cAligned));
                    }
                }
                CopyOutTile(dstGm, ubTmp_, colOff, 1, cCnt, cCnt, cAligned);
            }
        }
    }

    // 槽尾pad列一次性前置清零：dGi/dGh每行三槽尾[H,HPad)、hPrev行尾[H,HPad)。
    // BPTT循环内写出恒只覆盖[0,H)有效列，pad列清零后全程保持
    __aicore__ inline void ZeroSlotPadColumns()
    {
        if (!padH_) {
            return;
        }
        int64_t hTail = hPad_ - H_;
        int64_t tailAligned = Ceil(hTail, FP32_ALIGN) * FP32_ALIGN;
        int64_t maxRows = ubLength_ / tailAligned;
        int64_t perCore = Ceil(tb_, GetBlockNum());
        int64_t rowStart = GetBlockIdx() * perCore;
        int64_t rowCnt = Min(perCore, tb_ - rowStart);
        for (int64_t r0 = 0; r0 < rowCnt; r0 += maxRows) {
            int64_t cnt = Min(maxRows, rowCnt - r0);
            Duplicate(ubTmp2_, 0.0f, static_cast<int32_t>(cnt * tailAligned));
            PipeBarrier<PIPE_V>();
            for (int64_t slot = 0; slot < GATE_NUM; slot++) {
                CopyOutTileF(dGiGm_, ubTmp2_, (rowStart + r0) * threeHPad_ + slot * hPad_ + H_, cnt, hTail, threeHPad_,
                             tailAligned);
                CopyOutTileF(dGhGm_, ubTmp2_, (rowStart + r0) * threeHPad_ + slot * hPad_ + H_, cnt, hTail, threeHPad_,
                             tailAligned);
            }
            CopyOutTileF(hPrevGm_, ubTmp2_, (rowStart + r0) * hPad_ + H_, cnt, hTail, hPad_, tailAligned);
        }
    }

    __aicore__ inline void ProcessVector(int64_t t, bool prefetched)
    {
        if (!coreValid_) {
            return;
        }
        int64_t mStart = coreMStart_;
        int64_t mCnt = coreMCnt_;
        for (int64_t bOff = 0; bOff < mCnt; bOff += bTile_) {
            int64_t bRows = Min(bTile_, mCnt - bOff);
            Duplicate(ubRowAcc_, 0.0f, static_cast<int32_t>(bRows));
            PipeBarrier<PIPE_V>();
            for (int64_t ht = 0; ht < hTiles_; ht++) {
                ProcessVectorHTile(t, mStart + bOff, bRows, ht, prefetched);
            }
            StoreDwAtt(t, mStart + bOff, bRows);
        }
    }

    // VF计算体（regbase范式核心）：门梯度全链寄存器驻留计算——从9个UB操作数
    // 缓冲装载RegTensor，约18步计算不落UB，结果存回6个独立结果缓冲（操作数/
    // 结果缓冲分离，无同址读写）。UB操作数恒为fp32，VF体无dtype分支。
    // 这里有意保留单一__VEC_SCOPE__：门梯度链共享rGr/rNew/rGhn等寄存器；拆成
    // 多个函数会迫使中间结果落UB再重载，增加同步和舍入点并造成已测性能回退。
    __aicore__ inline void ComputeGateGradsVf(int64_t blkAligned)
    {
        __ubuf__ float* aDy = (__ubuf__ float*)ubGradH_.GetPhyAddr();
        __ubuf__ float* aDhp = (__ubuf__ float*)ubTmp_.GetPhyAddr();
        __ubuf__ float* aUpd = (__ubuf__ float*)ubUpdate_.GetPhyAddr();
        __ubuf__ float* aUAtt = (__ubuf__ float*)ubUAtt_.GetPhyAddr();
        __ubuf__ float* aAtt = (__ubuf__ float*)ubAtt_.GetPhyAddr();
        __ubuf__ float* aRst = (__ubuf__ float*)ubReset_.GetPhyAddr();
        __ubuf__ float* aNew = (__ubuf__ float*)ubNew_.GetPhyAddr();
        __ubuf__ float* aHN = (__ubuf__ float*)ubHN_.GetPhyAddr();
        __ubuf__ float* aHP = (__ubuf__ float*)ubHPrev_.GetPhyAddr();
        __ubuf__ float* aFromH = (__ubuf__ float*)ubFromH_.GetPhyAddr();
        __ubuf__ float* aDnt = (__ubuf__ float*)ubDNT_.GetPhyAddr();
        __ubuf__ float* aGhN = (__ubuf__ float*)ubTmp2_.GetPhyAddr();
        __ubuf__ float* aDr = (__ubuf__ float*)ubTmp3_.GetPhyAddr();
        __ubuf__ float* aDz = (__ubuf__ float*)ubDz_.GetPhyAddr();
        __ubuf__ float* aWa = (__ubuf__ float*)ubWa_.GetPhyAddr();

        constexpr uint32_t vfVl = static_cast<uint32_t>(AscendC::VECTOR_REG_WIDTH / FP32_BYTES);
        uint32_t count = static_cast<uint32_t>(blkAligned);
        uint16_t vfLoopNum = static_cast<uint16_t>((count + vfVl - 1) / vfVl);
        __VEC_SCOPE__
        {
            AscendC::Reg::RegTensor<float> rGr, rNew, rX, rY, rZ, rGhn, rOne, rFallback;
            AscendC::Reg::MaskReg preg, valid;
            AscendC::Reg::MaskReg pregAll = AscendC::Reg::CreateMask<float, AscendC::Reg::MaskPattern::ALL>();
            AscendC::Reg::Duplicate<float, AscendC::Reg::MaskMergeMode::ZEROING, float>(rOne, 1.0f, pregAll);
            for (uint16_t i = 0; i < vfLoopNum; i++) {
                preg = AscendC::Reg::UpdateMask<float>(count);
                uint32_t off = i * vfVl;
                __ubuf__ float* pDy = aDy + off;
                __ubuf__ float* pDhp = aDhp + off;
                __ubuf__ float* pUpd = aUpd + off;
                __ubuf__ float* pUAtt = aUAtt + off;
                __ubuf__ float* pAtt = aAtt + off;
                __ubuf__ float* pRst = aRst + off;
                __ubuf__ float* pNew = aNew + off;
                __ubuf__ float* pHN = aHN + off;
                __ubuf__ float* pHP = aHP + off;
                __ubuf__ float* pFromH = aFromH + off;
                __ubuf__ float* pDnt = aDnt + off;
                __ubuf__ float* pGhN = aGhN + off;
                __ubuf__ float* pDr = aDr + off;
                __ubuf__ float* pDz = aDz + off;
                __ubuf__ float* pWa = aWa + off;

                // gradH = dy + dhp（dhp已施加行级掩码）
                AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(rGr, pDy);
                AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(rX, pDhp);
                AscendC::Reg::Add<float, AscendC::Reg::MaskMergeMode::ZEROING>(rGr, rGr, rX, preg);
                // dhFromH = gradH * update_att
                AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(rX, pUAtt);
                AscendC::Reg::Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(rY, rGr, rX, preg);
                AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(pFromH, rY, preg);
                // dnt = gradH * (1-ua) * (1-n^2)
                AscendC::Reg::Sub<float, AscendC::Reg::MaskMergeMode::ZEROING>(rY, rOne, rX, preg);
                AscendC::Reg::Mul(rZ, rGr, rY, preg);
                // Fuse gradH * (1-ua), preserving the unfused special-value result.
                AscendC::Reg::Muls(rY, rGr, -1.0f, preg);
                AscendC::Reg::FusedMulDstAdd(rY, rX, rGr, preg);
                AscendC::Reg::Compare<float, CMPMODE::EQ>(valid, rY, rY, preg);
                AscendC::Reg::Select(rY, rY, rZ, valid);
                AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(rNew, pNew);
                // Fuse 1 - n*n so that a rounded square does not lose the
                // small derivative when the candidate gate is near +/-1.
                AscendC::Reg::Muls<float, float, AscendC::Reg::MaskMergeMode::ZEROING>(rX, rNew, -1.0f, preg);
                AscendC::Reg::FusedMulDstAdd(rX, rNew, rOne, preg);
                AscendC::Reg::Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(rY, rY, rX, preg);
                AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(pDnt, rY, preg);
                // dGh[n] = dnt * r
                AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(rX, pRst);
                AscendC::Reg::Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(rZ, rY, rX, preg);
                AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(pGhN, rZ, preg);
                // dr = dnt * r * (1-r) * hidden_new
                AscendC::Reg::Sub<float, AscendC::Reg::MaskMergeMode::ZEROING>(rZ, rOne, rX, preg);
                AscendC::Reg::Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(rZ, rZ, rX, preg);
                // Fuse r - r*r; retain the unfused result for non-finite inputs.
                AscendC::Reg::Muls(rFallback, rX, -1.0f, preg);
                AscendC::Reg::FusedMulDstAdd(rFallback, rX, rX, preg);
                AscendC::Reg::Compare<float, CMPMODE::EQ>(valid, rFallback, rFallback, preg);
                AscendC::Reg::Select(rZ, rFallback, rZ, valid);
                AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(rX, pHN);
                AscendC::Reg::Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(rZ, rZ, rX, preg);
                AscendC::Reg::Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(rZ, rY, rZ, preg);
                AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(pDr, rZ, preg);
                // ghn = gradH * (hp - n)
                AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(rX, pHP);
                AscendC::Reg::Sub<float, AscendC::Reg::MaskMergeMode::ZEROING>(rX, rX, rNew, preg);
                AscendC::Reg::Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(rGhn, rGr, rX, preg);
                // dz = ghn * (1-att) * update * (1-update)（update=z_t，与README的z_t一致；
                // README的u_t=z_t*(1-att_t)指update_att，勿混淆）
                AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(rX, pAtt);
                AscendC::Reg::Sub<float, AscendC::Reg::MaskMergeMode::ZEROING>(rX, rOne, rX, preg);
                AscendC::Reg::Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(rX, rGhn, rX, preg);
                AscendC::Reg::DataCopy<float, AscendC::Reg::LoadDist::DIST_NORM>(rY, pUpd);
                AscendC::Reg::Sub<float, AscendC::Reg::MaskMergeMode::ZEROING>(rZ, rOne, rY, preg);
                AscendC::Reg::Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(rZ, rZ, rY, preg);
                // Fuse z - z*z for the update-gate derivative as well.
                AscendC::Reg::Muls(rFallback, rY, -1.0f, preg);
                AscendC::Reg::FusedMulDstAdd(rFallback, rY, rY, preg);
                AscendC::Reg::Compare<float, CMPMODE::EQ>(valid, rFallback, rFallback, preg);
                AscendC::Reg::Select(rZ, rFallback, rZ, valid);
                AscendC::Reg::Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(rX, rX, rZ, preg);
                AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(pDz, rX, preg);
                // dw_att因子 = -(ghn * update)（gap列恒0，行归约无污染）
                AscendC::Reg::Mul<float, AscendC::Reg::MaskMergeMode::ZEROING>(rX, rGhn, rY, preg);
                AscendC::Reg::Muls<float, float, AscendC::Reg::MaskMergeMode::ZEROING>(rX, rX, -1.0f, preg);
                AscendC::Reg::DataCopy<float, AscendC::Reg::StoreDist::DIST_NORM_B32>(pWa, rX, preg);
            }
        }
    }

    __aicore__ inline void LoadVectorTile(const VectorTileParams& p, bool prefetched, bool loadFromH)
    {
        if (prefetched) {
            // 流水模式仅装载本步MM完成后才就绪的递归项。
            CopyInTileFRaw(dhPrevWsGm_, ubTmp_, p.bStart * hPad_ + p.hOff, p.bRows, p.hLen, hPad_, p.hAligned);
            SyncM2toV();
            return;
        }
        if constexpr (IS_FP16) {
            CopyInTile(dyGm_, ubGradH_, p.tbhOff, p.bRows, p.hLen, H_, p.hAligned);
            CopyInTileF(dhPrevWsGm_, ubTmp_, p.bStart * hPad_ + p.hOff, p.bRows, p.hLen, hPad_, p.hAligned);
            if (loadFromH && !fromHInUb_) {
                CopyInTileF(dhFromHGm_, ubFromH_, p.bhOff, p.bRows, p.hLen, H_, p.hAligned);
            }
            CopyInTile(updateGm_, ubUpdate_, p.tbhOff, p.bRows, p.hLen, H_, p.hAligned);
            CopyInTile(uAttGm_, ubUAtt_, p.tbhOff, p.bRows, p.hLen, H_, p.hAligned);
            CopyInTile(attGm_, ubAtt_, p.tbhOff, p.bRows, p.hLen, H_, p.hAligned);
            CopyInTile(resetGm_, ubReset_, p.tbhOff, p.bRows, p.hLen, H_, p.hAligned);
            CopyInTile(newGm_, ubNew_, p.tbhOff, p.bRows, p.hLen, H_, p.hAligned);
            CopyInTile(hiddenNewGm_, ubHN_, p.tbhOff, p.bRows, p.hLen, H_, p.hAligned);
            if (p.t == 0) {
                CopyInTile(initHGm_, ubHPrev_, p.bhOff, p.bRows, p.hLen, H_, p.hAligned);
            } else {
                CopyInTile(hGm_, ubHPrev_, (p.t - 1) * B_ * H_ + p.bStart * H_ + p.hOff, p.bRows, p.hLen, H_,
                           p.hAligned);
            }
        } else {
            SyncVtoM2Wait();
            CopyInTileFRaw(dyGm_, ubGradH_, p.tbhOff, p.bRows, p.hLen, H_, p.hAligned);
            CopyInTileFRaw(dhPrevWsGm_, ubTmp_, p.bStart * hPad_ + p.hOff, p.bRows, p.hLen, hPad_, p.hAligned);
            if (loadFromH && !fromHInUb_) {
                CopyInTileFRaw(dhFromHGm_, ubFromH_, p.bhOff, p.bRows, p.hLen, H_, p.hAligned);
            }
            CopyInTileFRaw(updateGm_, ubUpdate_, p.tbhOff, p.bRows, p.hLen, H_, p.hAligned);
            CopyInTileFRaw(uAttGm_, ubUAtt_, p.tbhOff, p.bRows, p.hLen, H_, p.hAligned);
            CopyInTileFRaw(attGm_, ubAtt_, p.tbhOff, p.bRows, p.hLen, H_, p.hAligned);
            CopyInTileFRaw(resetGm_, ubReset_, p.tbhOff, p.bRows, p.hLen, H_, p.hAligned);
            CopyInTileFRaw(newGm_, ubNew_, p.tbhOff, p.bRows, p.hLen, H_, p.hAligned);
            CopyInTileFRaw(hiddenNewGm_, ubHN_, p.tbhOff, p.bRows, p.hLen, H_, p.hAligned);
            if (p.t == 0) {
                CopyInTileFRaw(initHGm_, ubHPrev_, p.bhOff, p.bRows, p.hLen, H_, p.hAligned);
            } else {
                CopyInTileFRaw(hGm_, ubHPrev_, (p.t - 1) * B_ * H_ + p.bStart * H_ + p.hOff, p.bRows, p.hLen, H_,
                               p.hAligned);
            }
            SyncM2toV();
        }
    }

    __aicore__ inline void ApplyRecurrentGradient(const VectorTileParams& p, bool loadFromH)
    {
        if (loadFromH) {
            Add(ubTmp_, ubTmp_, ubFromH_, static_cast<int32_t>(p.bRows * p.hAligned));
            PipeBarrier<PIPE_V>();
        }
        if (tiling_->isSeqLength != 1) {
            return;
        }
        for (int64_t r = 0; r < p.bRows; r++) {
            int64_t b = p.bStart + r;
            int64_t seqV = useUbSeq_ ? static_cast<int64_t>(ubSeq_.GetValue(static_cast<int32_t>(b - coreMStart_))) :
                                       static_cast<int64_t>(seqLenGm_.GetValue(b));
            float mask = (p.t < seqV) ? 1.0f : 0.0f;
            if (mask != 1.0f) {
                Muls(ubTmp_[r * p.hAligned], ubTmp_[r * p.hAligned], mask, static_cast<int32_t>(p.hAligned));
            }
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void StoreVectorTile(const VectorTileParams& p)
    {
        SyncVtoM3();
        if (!fromHInUb_) {
            CopyOutTileFRaw(dhFromHGm_, ubFromH_, p.bhOff, p.bRows, p.hLen, H_, p.hAligned);
        }
        CopyOutTileFRaw(hPrevGm_, ubHPrev_, (p.t * B_ + p.bStart) * hPad_ + p.hOff, p.bRows, p.hLen, hPad_, p.hAligned);
        CopyOutTileFRaw(dGiGm_, ubDNT_, p.giRowOff + SLOT_NEW_GATE * hPad_, p.bRows, p.hLen, threeHPad_, p.hAligned);
        CopyOutTileFRaw(dGhGm_, ubTmp2_, p.giRowOff + SLOT_NEW_GATE * hPad_, p.bRows, p.hLen, threeHPad_, p.hAligned);
        CopyOutTileFRaw(dGiGm_, ubTmp3_, p.giRowOff + rSlot_ * hPad_, p.bRows, p.hLen, threeHPad_, p.hAligned);
        CopyOutTileFRaw(dGhGm_, ubTmp3_, p.giRowOff + rSlot_ * hPad_, p.bRows, p.hLen, threeHPad_, p.hAligned);
        CopyOutTileFRaw(dGiGm_, ubDz_, p.giRowOff + zSlot_ * hPad_, p.bRows, p.hLen, threeHPad_, p.hAligned);
        CopyOutTileFRaw(dGhGm_, ubDz_, p.giRowOff + zSlot_ * hPad_, p.bRows, p.hLen, threeHPad_, p.hAligned);
        SyncM3toV();

        if (dbInline_) {
            DbAccumulate(ubDbGi_, ubDbGiCorrection_, ubDz_, zSlot_ * hPad_ + p.hOff, p.bRows, p.hAligned);
            DbAccumulate(ubDbGh_, ubDbGhCorrection_, ubDz_, zSlot_ * hPad_ + p.hOff, p.bRows, p.hAligned);
            DbAccumulate(ubDbGi_, ubDbGiCorrection_, ubTmp3_, rSlot_ * hPad_ + p.hOff, p.bRows, p.hAligned);
            DbAccumulate(ubDbGh_, ubDbGhCorrection_, ubTmp3_, rSlot_ * hPad_ + p.hOff, p.bRows, p.hAligned);
            DbAccumulate(ubDbGi_, ubDbGiCorrection_, ubDNT_, SLOT_NEW_GATE * hPad_ + p.hOff, p.bRows, p.hAligned);
            DbAccumulate(ubDbGh_, ubDbGhCorrection_, ubTmp2_, SLOT_NEW_GATE * hPad_ + p.hOff, p.bRows, p.hAligned);
        }
        ReduceRows(ubRowSum_, ubWa_, p.bRows, p.hAligned);
        Add(ubRowAcc_, ubRowAcc_, ubRowSum_, static_cast<int32_t>(p.bRows));
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ProcessVectorHTile(int64_t t, int64_t bStart, int64_t bRows, int64_t ht, bool prefetched)
    {
        int64_t hOff = ht * hTile_;
        int64_t hLen = Min(hTile_, H_ - hOff);
        int64_t hAligned = Ceil(hLen, FP32_ALIGN) * FP32_ALIGN;
        int64_t blkAligned = bRows * hAligned;

        int64_t tbhOff = t * B_ * H_ + bStart * H_ + hOff;
        int64_t giRowOff = (t * B_ + bStart) * threeHPad_ + hOff; // dGi/dGh行距3HPad
        int64_t bhOff = bStart * H_ + hOff;
        // dhFromH折叠：t<T-1时装载上一步的dhFromH并与dhPrevWs相加，
        // 省去每步一段独立的GM读写与同步
        const bool loadFromH = (t < T_ - 1);

        VectorTileParams p = {t, bStart, bRows, hOff, hLen, hAligned, tbhOff, bhOff, giRowOff};
        LoadVectorTile(p, prefetched, loadFromH);

        ApplyRecurrentGradient(p, loadFromH);

        // 3. VF计算体：gradH起点的门梯度全链寄存器驻留
        ComputeGateGradsVf(blkAligned);

        StoreVectorTile(p);
    }

    __aicore__ inline void StoreDwAtt(int64_t t, int64_t bStart, int64_t bRows)
    {
        if constexpr (IS_FP16) {
            LocalTensor<DTYPE> staging = stagingOutQueue_.template AllocTensor<DTYPE>();
            Cast(staging, ubRowAcc_, RoundMode::CAST_RINT, static_cast<int32_t>(bRows));
            PipeBarrier<PIPE_V>();
            stagingOutQueue_.EnQue(staging);
            LocalTensor<DTYPE> stagingReady = stagingOutQueue_.template DeQue<DTYPE>();
            DataCopyExtParams cp(1, static_cast<uint32_t>(bRows * sizeof(DTYPE)), 0, 0, 0);
            DataCopyPad(dwAttGm_[t * B_ + bStart], stagingReady, cp);
            stagingOutQueue_.FreeTensor(stagingReady);
        } else {
            SyncVtoM3();
            DataCopyExtParams cp(1, static_cast<uint32_t>(bRows * sizeof(DTYPE)), 0, 0, 0);
            DataCopyPad(dwAttGm_[t * B_ + bStart], ubRowAcc_, cp);
            SyncM3toV();
        }
    }

    // dhPrevWs = dGh[t] @ w_hidden^T. Even small H uses short blocks
    // to limit cancellation error in the recurrent projection.
    __aicore__ inline void ProcessDgateMM(int64_t t)
    {
        if (threeHPad_ > Min(hPad_ / 2, MM_RECURRENT_CHUNK)) {
            ProcessRecurrentChunks(t);
            return;
        }
        if (GetBlockIdx() >= dgateMMTiling_.usedCoreNum) {
            return;
        }
        dgateMM.SetTensorA(dGhGm_[t * B_ * threeHPad_ + dgateOff_.aOffset], false);
        ApplyMMTail(dgateMM, dgateMMTiling_, dgateTail_);
        dgateMM.IterateAll(dhPrevWsGm_[dgateOff_.cOffset], false);
    }

    // Each Add/Sub preserves the FP32 rounding required by compensated summation.
    __aicore__ inline void AccumulatePartialVf(int32_t count) { AccumulatePartialVf(ubTmp_, ubTmp2_, ubTmp3_, count); }

    __aicore__ inline void AccumulatePartialVf(LocalTensor<float> sumTensor, LocalTensor<float> partialTensor,
                                               LocalTensor<float> correctionTensor, int32_t count)
    {
        __ubuf__ float* total = (__ubuf__ float*)sumTensor.GetPhyAddr();
        __ubuf__ float* partial = (__ubuf__ float*)partialTensor.GetPhyAddr();
        __ubuf__ float* correction = (__ubuf__ float*)correctionTensor.GetPhyAddr();
        uint32_t remaining = static_cast<uint32_t>(count);
        constexpr uint32_t lanes = AscendC::VECTOR_REG_WIDTH / FP32_BYTES;
        uint16_t loops = (remaining + lanes - 1) / lanes;
        __VEC_SCOPE__
        {
            Reg::RegTensor<float> sum, part, residual, adjusted, updated, zero;
            Reg::MaskReg active, valid;
            for (uint16_t i = 0; i < loops; ++i) {
                active = Reg::UpdateMask<float>(remaining);
                Reg::LoadAlign(sum, total + i * lanes);
                Reg::LoadAlign(part, partial + i * lanes);
                Reg::LoadAlign(residual, correction + i * lanes);
                Reg::Sub(adjusted, part, residual, active);
                Reg::Add(updated, sum, adjusted, active);
                Reg::Sub(residual, updated, sum, active);
                Reg::Sub(residual, residual, adjusted, active);
                Reg::Compare<float, CMPMODE::EQ>(valid, residual, residual, active);
                Reg::Duplicate(zero, 0.0f);
                Reg::Select(residual, residual, zero, valid);
                Reg::StoreAlign(total + i * lanes, updated, active);
                Reg::StoreAlign(correction + i * lanes, residual, active);
            }
        }
    }

    __aicore__ inline void AccumulateProjectionPartial(GlobalTensor<float>& outputGm, int64_t stride, int64_t mLo,
                                                       int64_t rows, int64_t nLo, int64_t cols, bool first)
    {
        for (int64_t col = 0; col < cols; col += ubLength_) {
            int64_t len = Min(ubLength_, cols - col);
            int64_t aligned = Ceil(len, FP32_ALIGN) * FP32_ALIGN;
            for (int64_t row = 0; row < rows; row += ubLength_ / aligned) {
                int64_t countRows = Min(ubLength_ / aligned, rows - row);
                int64_t off = (mLo + row) * stride + nLo + col;
                int32_t count = static_cast<int32_t>(countRows * aligned);
                CopyInTileF(projectionPartialGm_, ubTmp2_, off, countRows, len, stride, aligned);
                if (first) {
                    Duplicate(ubTmp3_, 0.0f, count);
                    CopyOutTileF(outputGm, ubTmp2_, off, countRows, len, stride, aligned);
                } else {
                    CopyInTileF(outputGm, ubTmp_, off, countRows, len, stride, aligned);
                    CopyInTileF(projectionCorrectionGm_, ubTmp3_, off, countRows, len, stride, aligned);
                    AccumulatePartialVf(count);
                    CopyOutTileF(outputGm, ubTmp_, off, countRows, len, stride, aligned);
                }
                CopyOutTileF(projectionCorrectionGm_, ubTmp3_, off, countRows, len, stride, aligned);
            }
        }
    }

    __aicore__ inline void ProcessRecurrentChunks(int64_t time)
    {
        bool active = GetBlockIdx() < dgateMMTiling_.usedCoreNum;
        int64_t rows = (dgateTail_.mIdx == dgateTail_.mLoop - 1) ? dgateTail_.tailM : dgateMMTiling_.singleCoreM;
        int64_t cols = (dgateTail_.nIdx == dgateTail_.nLoop - 1) ? dgateTail_.tailN : dgateMMTiling_.singleCoreN;
        int64_t mLo = dgateTail_.mIdx * dgateMMTiling_.singleCoreM;
        int64_t nLo = dgateTail_.nIdx * dgateMMTiling_.singleCoreN;
        if (active) {
            dgateMM.SetHF32(false, 0);
            dgateMM.SetOrgShape(B_, hPad_, threeHPad_);
        }
        int64_t blockK = Min(hPad_ / 2, MM_RECURRENT_CHUNK);
        for (int64_t k = 0; k < threeHPad_; k += blockK) {
            if (active) {
                dgateMM.SetTensorA(dGhGm_[time * B_ * threeHPad_ + dgateOff_.aOffset + k], false);
                dgateMM.SetTensorB(wHiddenFp32Gm_[dgateOff_.bOffset + k], true);
                ApplyMMTail(dgateMM, dgateMMTiling_, dgateTail_);
                dgateMM.IterateAll(projectionPartialGm_[dgateOff_.cOffset], false);
                dgateMM.End();
            }
            // IterateAll(false) completes this core's tile. Each core owns a
            // disjoint output tile, so no cross-core barrier is needed here.
            if (active) {
                AccumulateProjectionPartial(dhPrevWsGm_, hPad_, mLo, rows, nLo, cols, k == 0);
            }
        }
    }

    // 单tile流水版dgateMM：异步IterateAll后在等待窗口内预取下一时间步操作数。
    // waitIterateAll必须显式传true（否则cube master不置完成标志，真机永挂）
    __aicore__ inline void ProcessDgateMMPipeline(int64_t t)
    {
        if (GetBlockIdx() < dgateMMTiling_.usedCoreNum) {
            dgateMM.SetTensorA(dGhGm_[t * B_ * threeHPad_ + dgateOff_.aOffset], false);
            ApplyMMTail(dgateMM, dgateMMTiling_, dgateTail_);
            dgateMM.IterateAll<false>(dhPrevWsGm_[dgateOff_.cOffset], 0, false, true);
        }
        if (t > 0) {
            PrefetchOperands(t - 1);
        }
        if (GetBlockIdx() < dgateMMTiling_.usedCoreNum) {
            dgateMM.WaitIterateAll();
        }
    }

    // 预取下一时间步tNext的9个非递归操作数到UB（递归的dhPrevWs在本步MM完成后
    // 才就绪，不可预取）。调用点本核V/MTE3均已排空、上一批MTE2已等待完成，
    // 操作数缓冲可被MTE2覆写
    __aicore__ inline void PrefetchOperands(int64_t tNext)
    {
        // 流水仅fp32（host侧门控）；fp16实例化下编译为空
        if constexpr (IS_FP32) {
            if (!coreValid_) {
                return;
            }
            int64_t bRows = coreMCnt_;
            int64_t hLen = Min(hTile_, H_);
            int64_t hAligned = Ceil(hLen, FP32_ALIGN) * FP32_ALIGN;
            int64_t tbhOff = tNext * B_ * H_ + coreMStart_ * H_;
            int64_t bhOff = coreMStart_ * H_;
            CopyInTileFRaw(dyGm_, ubGradH_, tbhOff, bRows, hLen, H_, hAligned);
            CopyInTileFRaw(dhFromHGm_, ubFromH_, bhOff, bRows, hLen, H_, hAligned);
            CopyInTileFRaw(updateGm_, ubUpdate_, tbhOff, bRows, hLen, H_, hAligned);
            CopyInTileFRaw(uAttGm_, ubUAtt_, tbhOff, bRows, hLen, H_, hAligned);
            CopyInTileFRaw(attGm_, ubAtt_, tbhOff, bRows, hLen, H_, hAligned);
            CopyInTileFRaw(resetGm_, ubReset_, tbhOff, bRows, hLen, H_, hAligned);
            CopyInTileFRaw(newGm_, ubNew_, tbhOff, bRows, hLen, H_, hAligned);
            CopyInTileFRaw(hiddenNewGm_, ubHN_, tbhOff, bRows, hLen, H_, hAligned);
            if (tNext == 0) {
                CopyInTileFRaw(initHGm_, ubHPrev_, bhOff, bRows, hLen, H_, hAligned);
            } else {
                CopyInTileFRaw(hGm_, ubHPrev_, (tNext - 1) * B_ * H_ + bhOff, bRows, hLen, H_, hAligned);
            }
        }
    }

    // 收尾：dh_prev = mm(0) + dhFromH(0)（dhFromH折叠后循环内dhPrevWs恒为mm原始结果）
    __aicore__ inline void FinalStoreDhPrev()
    {
        if (!coreValid_) {
            return;
        }
        int64_t mStart = coreMStart_;
        int64_t mCnt = coreMCnt_;
        for (int64_t bOff = 0; bOff < mCnt; bOff += bTile_) {
            int64_t bRows = Min(bTile_, mCnt - bOff);
            for (int64_t ht = 0; ht < hTiles_; ht++) {
                int64_t hOff = ht * hTile_;
                int64_t hLen = Min(hTile_, H_ - hOff);
                int64_t hAligned = Ceil(hLen, FP32_ALIGN) * FP32_ALIGN;
                int64_t off = (mStart + bOff) * hPad_ + hOff;
                int64_t offH = (mStart + bOff) * H_ + hOff;
                CopyInTileF(dhPrevWsGm_, ubTmp_, off, bRows, hLen, hPad_, hAligned);
                if (fromHInUb_) {
                    // 单tile：ubFromH_仍驻留dhFromH[0]，免GM读回
                    Add(ubTmp3_, ubTmp_, ubFromH_, static_cast<int32_t>(bRows * hAligned));
                } else {
                    CopyInTileF(dhFromHGm_, ubTmp2_, offH, bRows, hLen, H_, hAligned);
                    Add(ubTmp3_, ubTmp_, ubTmp2_, static_cast<int32_t>(bRows * hAligned));
                }
                CopyOutTile(dhPrevGm_, ubTmp3_, offH, bRows, hLen, H_, hAligned);
            }
        }
    }

    // dw_input = x^T @ dGi（SetTensor在BPTT写出dGi之后）
    __aicore__ inline void ProcessDwInputMM()
    {
        if ((mmChunkMask_ & MM_CHUNK_DW_INPUT) != 0) {
            RunChunkedMM(dwInputMM, dwInputMMTiling_, dwInputTail_, xFp32Gm_, dwInputOff_.aOffset, I_, true, dGiGm_,
                         dwInputOff_.bOffset, threeHPad_, false, dwInputFp32Gm_, dwInputPartialGm_, threeHPad_, tbPad_);
            return;
        }
        if (GetBlockIdx() >= dwInputMMTiling_.usedCoreNum) {
            return;
        }
        dwInputMM.SetTensorA(xFp32Gm_[dwInputOff_.aOffset], true);
        dwInputMM.SetTensorB(dGiGm_[dwInputOff_.bOffset], false);
        ApplyMMTail(dwInputMM, dwInputMMTiling_, dwInputTail_);
        dwInputMM.IterateAll(dwInputFp32Gm_[dwInputOff_.cOffset], false);
    }

    // dw_hidden = hPrev^T @ dGh（SetTensor在BPTT写出dGh/hPrev之后）
    __aicore__ inline void ProcessDwHiddenMM()
    {
        if ((mmChunkMask_ & MM_CHUNK_DW_HIDDEN) != 0) {
            RunChunkedMM(dwHiddenMM, dwHiddenMMTiling_, dwHiddenTail_, hPrevGm_, dwHiddenOff_.aOffset, hPad_, true,
                         dGhGm_, dwHiddenOff_.bOffset, threeHPad_, false, dwHiddenFp32Gm_, dwHiddenPartialGm_,
                         threeHPad_, tbPad_);
            return;
        }
        if (GetBlockIdx() >= dwHiddenMMTiling_.usedCoreNum) {
            return;
        }
        dwHiddenMM.SetTensorA(hPrevGm_[dwHiddenOff_.aOffset], true);
        dwHiddenMM.SetTensorB(dGhGm_[dwHiddenOff_.bOffset], false);
        ApplyMMTail(dwHiddenMM, dwHiddenMMTiling_, dwHiddenTail_);
        dwHiddenMM.IterateAll(dwHiddenFp32Gm_[dwHiddenOff_.cOffset], false);
    }

    // GM fp32同布局tile累加：acc[mLo..mLo+mRows, nLo..nLo+nCols] += part同区域
    // （两侧行距相同均为stride；按ubLength列块×行块经UB搬运累加）
    __aicore__ inline void AccumulateTileFp32(GlobalTensor<float>& accGm, GlobalTensor<float>& partGm, int64_t mLo,
                                              int64_t mRows, int64_t nLo, int64_t nCols, int64_t stride, bool first)
    {
        for (int64_t c0 = 0; c0 < nCols; c0 += ubLength_) {
            int64_t cCnt = Min(ubLength_, nCols - c0);
            int64_t cAligned = Ceil(cCnt, FP32_ALIGN) * FP32_ALIGN;
            int64_t maxRows = ubLength_ / cAligned;
            for (int64_t r0 = 0; r0 < mRows; r0 += maxRows) {
                int64_t rCnt = Min(maxRows, mRows - r0);
                int64_t off = (mLo + r0) * stride + nLo + c0;
                CopyInTileF(accGm, ubTmp_, off, rCnt, cCnt, stride, cAligned);
                CopyInTileF(partGm, ubTmp2_, off, rCnt, cCnt, stride, cAligned);
                int32_t count = static_cast<int32_t>(rCnt * cAligned);
                if (first) {
                    Duplicate(ubTmp3_, 0.0f, count);
                } else {
                    CopyInTileF(weightCorrectionGm_, ubTmp3_, off, rCnt, cCnt, stride, cAligned);
                }
                AccumulatePartialVf(count);
                PipeBarrier<PIPE_V>();
                CopyOutTileF(accGm, ubTmp_, off, rCnt, cCnt, stride, cAligned);
                CopyOutTileF(weightCorrectionGm_, ubTmp3_, off, rCnt, cCnt, stride, cAligned);
            }
        }
    }

    // 通用K分块matmul：大K单次cube累加误差随K系统性增长（T=1隔离实测：K=96≈
    // fp32-BLAS底线，K=1024约1.5倍，K=8192约4.6倍），按mmChunkK_分块多次cube
    // 调用、首块直写C、其余各块落partial后向量fp32累加，恢复底线精度。
    // kTotal为分块对齐后的K（=Ceil(K,mmChunkK_)*mmChunkK_）：所有块均为16对齐
    // 全长、无K尾块（SetTail的K维语义为"最后一个K迭代长度"，会致越界累计），
    // [K,kTotal)的pad行恒0贡献。aBaseOff/bBaseOff为该MM M/N分块的基偏移
    // （CalcMMOffsets语义），块间K向平移为k0*aKStride/k0*bKStride。
    // SyncAll为全核路障：非参与核也须到达，故active仅掩护cube/向量段。
    template <typename MMT>
    __aicore__ inline void RunChunkedMM(MMT& mm, TCubeTiling& param, MMTail& t, GlobalTensor<float>& aGm,
                                        int64_t aBaseOff, int64_t aKStride, bool aTrans, GlobalTensor<float>& bGm,
                                        int64_t bBaseOff, int64_t bKStride, bool bTrans, GlobalTensor<float>& cGm,
                                        GlobalTensor<float>& partialGm, int64_t cStride, int64_t kTotal)
    {
        bool active = GetBlockIdx() < param.usedCoreNum;
        int64_t mRows = (t.mIdx == t.mLoop - 1) ? t.tailM : param.singleCoreM;
        int64_t nCols = (t.nIdx == t.nLoop - 1) ? t.tailN : param.singleCoreN;
        int64_t mLo = t.mIdx * param.singleCoreM;
        int64_t nLo = t.nIdx * param.singleCoreN;
        int64_t cOff = mLo * cStride + nLo;
        for (int64_t k0 = 0; k0 < kTotal; k0 += mmChunkK_) {
            if (active) {
                mm.SetTensorA(aGm[aBaseOff + k0 * aKStride], aTrans);
                mm.SetTensorB(bGm[bBaseOff + k0 * bKStride], bTrans);
                ApplyMMTail(mm, param, t);
                if (k0 == 0) {
                    mm.IterateAll(cGm[cOff], false);
                } else {
                    mm.IterateAll(partialGm[cOff], false);
                }
            }
            if (k0 > 0) {
                SyncAll();
                if (active) {
                    AccumulateTileFp32(cGm, partialGm, mLo, mRows, nLo, nCols, cStride, k0 == mmChunkK_);
                }
                SyncAll();
            }
        }
    }

    // Preserve the complete ND row stride while reducing short K blocks.
    // Small hidden dimensions use one padded gate per block to reduce
    // cancellation error between gate contributions to near-zero dx values.
    __aicore__ inline void ProcessDxMM()
    {
        if (GetBlockIdx() >= dxMMTiling_.usedCoreNum) {
            return;
        }
        int64_t rows = (dxTail_.mIdx == dxTail_.mLoop - 1) ? dxTail_.tailM : dxMMTiling_.singleCoreM;
        int64_t cols = (dxTail_.nIdx == dxTail_.nLoop - 1) ? dxTail_.tailN : dxMMTiling_.singleCoreN;
        int64_t mLo = dxTail_.mIdx * dxMMTiling_.singleCoreM;
        int64_t nLo = dxTail_.nIdx * dxMMTiling_.singleCoreN;
        int64_t blockK = Min(hPad_, MM_RECURRENT_CHUNK);
        dxMM.SetHF32(false, 0);
        dxMM.SetOrgShape(tb_, I_, threeHPad_);
        for (int64_t k = 0; k < threeHPad_; k += blockK) {
            dxMM.SetTensorA(dGiGm_[dxOff_.aOffset + k], false);
            dxMM.SetTensorB(wInputFp32Gm_[dxOff_.bOffset + k], true);
            ApplyMMTail(dxMM, dxMMTiling_, dxTail_);
            dxMM.IterateAll(projectionPartialGm_[dxOff_.cOffset], false);
            dxMM.End();
            AccumulateProjectionPartial(dxFp32Gm_, I_, mLo, rows, nLo, cols, k == 0);
        }
    }

    // padded布局逐槽列归约：[TS,3HPad](z/pad/r/pad/n) -> [3H]紧凑输出
    // 按槽×槽内列块切核（槽内段不跨槽），归约结果写dst对应槽内偏移
    __aicore__ inline void ProcessBiasReduceSlot(GlobalTensor<float>& srcGm, GlobalTensor<DTYPE>& dstGm)
    {
        int64_t rows = tb_;
        int64_t blockDim = GetBlockNum();
        int64_t chunk = tiling_->singleCoreReduceN; // 槽内列块宽度
        // 全局块号 = slot * Ceil(H,chunk) + 槽内块号
        int64_t blocksPerSlot = Ceil(H_, chunk);
        int64_t blockCnt = GATE_NUM * blocksPerSlot;
        for (int64_t blk = GetBlockIdx(); blk < blockCnt; blk += blockDim) {
            int64_t slot = blk / blocksPerSlot;
            int64_t sub = blk % blocksPerSlot;
            int64_t inSlotStart = sub * chunk;
            int64_t nCnt = Min(chunk, H_ - inSlotStart);
            if (nCnt <= 0) {
                continue;
            }
            int64_t srcSlot = slot * hPad_ + inSlotStart;
            int64_t dstSlot = slot * H_ + inSlotStart;
            for (int64_t cStart = 0; cStart < nCnt; cStart += ubLength_) {
                int64_t cCnt = Min(ubLength_, nCnt - cStart);
                int64_t cAligned = Ceil(cCnt, FP32_ALIGN) * FP32_ALIGN;
                int64_t srcCol = srcSlot + cStart;
                Duplicate(ubTmp_, 0.0f, static_cast<int32_t>(cAligned));
                Duplicate(ubTmp3_, 0.0f, static_cast<int32_t>(cAligned));
                PipeBarrier<PIPE_V>();
                int64_t maxRows = ubLength_ / cAligned;
                for (int64_t r0 = 0; r0 < rows; r0 += maxRows) {
                    int64_t rCnt = Min(maxRows, rows - r0);
                    // 2D批量搬入该槽该列段（源行距3HPad、槽距hPad，对齐拆分由helper内部处理）
                    CopyInTileFOff(srcGm, ubTmp2_, 0, r0 * threeHPad_ + srcCol, rCnt, cCnt, threeHPad_, cAligned);
                    for (int64_t rr = 0; rr < rCnt; rr++) {
                        AccumulatePartialVf(ubTmp_, ubTmp2_[rr * cAligned], ubTmp3_, static_cast<int32_t>(cAligned));
                    }
                }
                CopyOutTile(dstGm, ubTmp_, dstSlot + cStart, 1, cCnt, cCnt, cAligned);
            }
        }
    }

    // 对[TS, cols]沿TS列归约得[cols]；列块按blockDim跨核循环，任意blockDim均覆盖全部列
    __aicore__ inline void ProcessBiasReduce(GlobalTensor<float>& srcGm, GlobalTensor<DTYPE>& dstGm)
    {
        int64_t rows = tb_;
        // padH时源为槽间含pad的padded布局，按槽归约后拼接；对齐时整行归约即可
        if (padH_) {
            ProcessBiasReduceSlot(srcGm, dstGm);
            return;
        }
        int64_t cols = threeH_;
        int64_t srcStride = threeHPad_;
        int64_t blockDim = GetBlockNum();
        int64_t reduceChunk = tiling_->singleCoreReduceN;
        int64_t blockCnt = Ceil(cols, reduceChunk);
        for (int64_t blk = GetBlockIdx(); blk < blockCnt; blk += blockDim) {
            int64_t nStart = blk * reduceChunk;
            int64_t nCnt = Min(reduceChunk, cols - nStart);
            if (nCnt <= 0) {
                continue;
            }
            for (int64_t cStart = 0; cStart < nCnt; cStart += ubLength_) {
                int64_t cCnt = Min(ubLength_, nCnt - cStart);
                int64_t cAligned = Ceil(cCnt, FP32_ALIGN) * FP32_ALIGN;
                int64_t colOff = nStart + cStart;
                Duplicate(ubTmp_, 0.0f, static_cast<int32_t>(cAligned));
                Duplicate(ubTmp3_, 0.0f, static_cast<int32_t>(cAligned));
                PipeBarrier<PIPE_V>();
                // 按行块读入累加（行数受UB容量限制）
                int64_t maxRows = ubLength_ / cAligned;
                for (int64_t r0 = 0; r0 < rows; r0 += maxRows) {
                    int64_t rCnt = Min(maxRows, rows - r0);
                    CopyInTileF(srcGm, ubTmp2_, r0 * srcStride + colOff, rCnt, cCnt, srcStride, cAligned);
                    for (int64_t rr = 0; rr < rCnt; rr++) {
                        AccumulatePartialVf(ubTmp_, ubTmp2_[rr * cAligned], ubTmp3_, static_cast<int32_t>(cAligned));
                    }
                }
                CopyOutTile(dstGm, ubTmp_, colOff, 1, cCnt, cCnt, cAligned);
            }
        }
    }

private:
    GlobalTensor<DTYPE> xGm_;
    GlobalTensor<DTYPE> wInputGm_;
    GlobalTensor<DTYPE> wHiddenGm_;
    GlobalTensor<DTYPE> attGm_;
    GlobalTensor<DTYPE> initHGm_;
    GlobalTensor<DTYPE> hGm_;
    GlobalTensor<DTYPE> dyGm_;
    GlobalTensor<DTYPE> dhGm_;
    GlobalTensor<DTYPE> updateGm_;
    GlobalTensor<DTYPE> uAttGm_;
    GlobalTensor<DTYPE> resetGm_;
    GlobalTensor<DTYPE> newGm_;
    GlobalTensor<DTYPE> hiddenNewGm_;
    GlobalTensor<int32_t> seqLenGm_;

    GlobalTensor<DTYPE> dwInputGm_;
    GlobalTensor<DTYPE> dwHiddenGm_;
    GlobalTensor<DTYPE> dbInputGm_;
    GlobalTensor<DTYPE> dbHiddenGm_;
    GlobalTensor<DTYPE> dxGm_;
    GlobalTensor<DTYPE> dhPrevGm_;
    GlobalTensor<DTYPE> dwAttGm_;

    GlobalTensor<float> dGhGm_;
    GlobalTensor<float> dGiGm_;
    GlobalTensor<float> hPrevGm_;
    GlobalTensor<float> dhPrevWsGm_;
    GlobalTensor<float> dhFromHGm_;
    GlobalTensor<float> wHiddenFp32Gm_;
    GlobalTensor<float> wInputFp32Gm_;
    GlobalTensor<float> xFp32Gm_;
    GlobalTensor<float> dwInputFp32Gm_;
    GlobalTensor<float> dwHiddenFp32Gm_;
    GlobalTensor<float> dxFp32Gm_;
    GlobalTensor<float> dbPartialGiGm_;
    GlobalTensor<float> dbPartialGhGm_;
    GlobalTensor<float> dwInputPartialGm_;
    GlobalTensor<float> dwHiddenPartialGm_;
    GlobalTensor<float> projectionPartialGm_;
    GlobalTensor<float> projectionCorrectionGm_;
    GlobalTensor<float> weightCorrectionGm_;

    TBuf<TPosition::VECCALC> vbGradH_;
    TBuf<TPosition::VECCALC> vbHPrev_;
    TBuf<TPosition::VECCALC> vbUpdate_;
    TBuf<TPosition::VECCALC> vbUAtt_;
    TBuf<TPosition::VECCALC> vbAtt_;
    TBuf<TPosition::VECCALC> vbReset_;
    TBuf<TPosition::VECCALC> vbNew_;
    TBuf<TPosition::VECCALC> vbHN_;
    TBuf<TPosition::VECCALC> vbTmp_;
    TBuf<TPosition::VECCALC> vbFromH_;
    TBuf<TPosition::VECCALC> vbDNT_;
    TBuf<TPosition::VECCALC> vbTmp2_;
    TBuf<TPosition::VECCALC> vbTmp3_;
    TBuf<TPosition::VECCALC> vbDz_;
    TBuf<TPosition::VECCALC> vbWa_;
    TBuf<TPosition::VECCALC> vbRowAcc_;
    TBuf<TPosition::VECCALC> vbRowSum_;
    TBuf<TPosition::VECCALC> vbReduceTmp_;
    TBuf<TPosition::VECCALC> vbDbGi_;
    TBuf<TPosition::VECCALC> vbDbGh_;
    TBuf<TPosition::VECCALC> vbDbGiCorrection_;
    TBuf<TPosition::VECCALC> vbDbGhCorrection_;
    // fp16暂存走TQue：队列框架保证跨流水staging复用安全
    TQue<QuePosition::VECIN, 1> stagingInQueue_;
    TQue<QuePosition::VECOUT, 1> stagingOutQueue_;

    LocalTensor<float> ubGradH_;
    LocalTensor<float> ubHPrev_;
    LocalTensor<float> ubUpdate_;
    LocalTensor<float> ubUAtt_;
    LocalTensor<float> ubAtt_;
    LocalTensor<float> ubReset_;
    LocalTensor<float> ubNew_;
    LocalTensor<float> ubHN_;
    LocalTensor<float> ubTmp_;
    LocalTensor<float> ubFromH_;
    LocalTensor<float> ubDNT_;
    LocalTensor<float> ubTmp2_;
    LocalTensor<float> ubTmp3_;
    LocalTensor<float> ubDz_;
    LocalTensor<float> ubWa_;
    LocalTensor<float> ubRowAcc_;
    LocalTensor<float> ubRowSum_;
    LocalTensor<float> ubReduceTmp_;
    LocalTensor<float> ubDbGi_;
    LocalTensor<float> ubDbGh_;
    LocalTensor<float> ubDbGiCorrection_;
    LocalTensor<float> ubDbGhCorrection_;

    const DynamicAUGRUGradTilingData* tiling_ = nullptr;
    int64_t T_ = 0;
    int64_t B_ = 0;
    int64_t H_ = 0;
    int64_t I_ = 0;
    int64_t hPad_ = 0;
    int64_t tbPad_ = 0;
    int64_t threeHPad_ = 0;
    bool padH_ = false;
    bool needWsCopy_ = false;
    int64_t threeH_ = 0;
    int64_t tb_ = 0;
    int64_t bTile_ = 0;
    int64_t hTile_ = 0;
    int64_t ubLength_ = 0;
    int64_t zSlot_ = 0;
    int64_t rSlot_ = 1;
    bool coreValid_ = false;
    int64_t coreMStart_ = 0;
    int64_t coreMCnt_ = 0;
    int64_t hTiles_ = 0;
    bool pipeline_ = false;
    bool dbInline_ = false;
    bool fromHInUb_ = false;
    bool useUbSeq_ = false;
    int64_t mmChunkK_ = 0;
    int64_t mmChunkMask_ = 0;
    TBuf<TPosition::VECCALC> vbSeq_;
    LocalTensor<int32_t> ubSeq_;

    TCubeTiling dgateMMTiling_;
    TCubeTiling dwInputMMTiling_;
    TCubeTiling dwHiddenMMTiling_;
    TCubeTiling dxMMTiling_;
    MMOffsets dgateOff_;
    MMOffsets dwInputOff_;
    MMOffsets dwHiddenOff_;
    MMOffsets dxOff_;
    MMTail dgateTail_;
    MMTail dwInputTail_;
    MMTail dwHiddenTail_;
    MMTail dxTail_;
};

} // namespace NsDynamicAUGRUGrad
#endif // __DYNAMIC_AUGRU_GRAD_H__
