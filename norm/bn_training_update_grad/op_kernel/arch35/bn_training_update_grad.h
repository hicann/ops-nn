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
 * \file bn_training_update_grad.h
 * \brief BNTrainingUpdateGrad arch35(regbase/MicroAPI) kernel
 *        rstd[c]        = 1 / sqrt(batch_variance[c] + epsilon)
 *        diff_scale[c]  = sum_{n,r} grads[n,c,r] * (x[n,c,r] - batch_mean[c]) * rstd[c]
 *        diff_offset[c] = sum_{n,r} grads[n,c,r]
 *        与 910b TBE（dynamic/bn_training_update_grad.py）语义与逐元素运算顺序一致；
 *        fp16/bf16 输入 reg 内解包升 fp32 计算，输出恒 fp32。
 *
 *        结构要点：
 *        - 切核：channel 主切分（前多后少，blockIdx = cRangeIdx）；每 channel 的完整
 *          归约由唯一归属核完成，零核间通信、无 workspace、无原子加，相同输入必得
 *          相同输出（确定性）；
 *        - 核内三层循环：channel chunk(cLenCap) × R 分片(sliceR) × N 行 tile(rowsPerTile)；
 *          每 (chunk,rSlice) 一次向量化展开 mean/rstd 系数到 [cLen*rEff]（rEff%8==0 走
 *          DIST_BRC_B32 广播 + mask store；否则标量 SetValue 兜底 + V_S/S_V 事件同步）；
 *        - 每 tile：2D DataCopyPad 搬 grads/x [rows, cLen*rEff]（行距 C*R），VF 逐元素
 *          t=(x-meanExp)*rstdExp、p=grads*t（运算序对齐 A2），并按元素位置向量累加进
 *          核内二维累加器 accRowG/accRowP[cLen*rEff]（每 n 行一趟；VL 块在外、行在内 +
 *          寄存器累加器，UB 读改写每块仅一次、不同块不同地址，规避 __VEC_SCOPE__ 软流水
 *          把下行 load 排到上行 store 前的丢行问题）；
 *        - 每 (chunk,rSlice) 末仅一次 ReduceSum<float, Pattern::Reduce::AR, isReuseSource=true>
 *          [cLen, rEff]→[cLen]（单发调用形态）；rEff==1 时 accRow 即逐 channel 结果，
 *          跳过归约；归约结果跨 rSlice 累加进 accOff/accScale 后写出。
 *          ReduceSum 后 PipeBarrier<PIPE_V> 排干 V pipe（实测：adv_api 非对齐
 *          LoadUnAlign 序列与后续 VF/MTE 混排会污染地址寄存器，致后续 DMA 错误访存）。
 *
 *        小 C 深归约快路（cLenCap==1，即 innerSize > qCap 或 C<=核数）：
 *        - 每 (n,c) 段为连续 1D：大块 1D DataCopyPad（sliceR 字段下发 chunk 元素数），
 *          mean/rstd 退化为标量（Adds/Muls 标量-向量指令，免展开系数）；
 *        - VF 纯 elementwise 逐位置写 scratch（accRowG_/accRowP_ 复用；零 RMW 零寄存器
 *          跨块携带——实测本平台寄存器跨块持有累加会被 __VEC_SCOPE__ 软流水偶发丢块，
 *          掩码 load+累加组合亦不可靠），每 chunk 两发位置归约 [1,eff64]→[1] 累加进
 *          accOff/accScale（慢路每 slice 同款已验证形态）；
 *        - 尾部 eff%64(<64 元素)：独立 64 槽尾缓冲(expMean_/expRstd_ 复用)，标量
 *          SetValue 写全 64 槽(值+显式补零) + MTE2_S/S_V 事件对 + ReduceSum [1,64]，
 *          全程无掩码无隐式填充——修掉慢路每片 ~19us 固定开销(BuildExpanded/ZeroBuf/
 *          每片归约)在小 C 深 R 形态把单核带宽压到 ~1.7GB/s 的问题。
 *
 *        NHWC 路径（isNhwc=1，含 ND C==1 大规模/ND R==1 巨 C 两条 reroute，布局同构）：
 *        - [rows, C] 行主序（C=最后一维）；每核 (行范围/通道段, c 窗口) 二维职责；
 *        - channelSplit（mode=1，C 大）：每核独占通道段扫全部行，零通信零 workspace；
 *          段长 > 窗预算时 c 窗口化（窗口外层 × 行 tile 内层，元素恰读一次，C 无上限）；
 *        - rowSplit（mode=2，C 小）：每核独占行段×全部 C；缺省 0 号核覆盖写输出 + SyncAll +
 *          其余核浮点原子加直写（零 ws——原子和顺序不定，误差 O(eps)）；ge.deterministic=1 时
 *          切换两段确定性归约（各核部分和落用户 ws 固定槽 → SyncAll → 固定布局 gather+RA 归约，
 *          相同输入逐位复现；ws 取 GetUserWorkspace 用户区，直绑原始地址会踩 16MB 系统区）
 *          ——并行度恒=核数（G22 C==1 单核 DMA-bound 的解法）；
 *        - 计算体与慢路 ComputeTile 同构（VL 块外层 × 行内层 + 块内寄存器累加器，每
 *          tile 每块仅一次 acc UB RMW——实测安全的唯一累加器结构），mean/rstd 按 64
 *          chunk 驻留 UB 直加载（免展开）；Kahan 向量化（寄存器源变体）保深 rows 精度；
 *        - 窗口宽 W 恒 64 对齐：UB 行基址 32B 对齐 + 行尾 64 元向量无掩码读不越 pitch；
 *          尾窗 cnt<W 的死 lane 垃圾随 tile 驻留残渣进入 acc 死 lane，最终只写 [cnt]
 *          元素（WritePartials 的 blockLen=cnt*4 原样复用），全程无掩码 load/累加。
 */
#ifndef BN_TRAINING_UPDATE_GRAD_H
#define BN_TRAINING_UPDATE_GRAD_H

#include <type_traits>
#include "kernel_operator.h"
#include "adv_api/reduce/reduce.h"
#include "../inc/platform.h"
#include "bn_training_update_grad_tiling_data.h"

namespace BNTrainingUpdateGradOps {
using namespace AscendC;
using AscendC::MicroAPI::CreateMask;
using AscendC::MicroAPI::LoadDist;
using AscendC::MicroAPI::LocalMemBar;
using AscendC::MicroAPI::MaskPattern;
using AscendC::MicroAPI::MaskReg;
using AscendC::MicroAPI::MemType;
using AscendC::MicroAPI::RegTensor;
using AscendC::MicroAPI::StoreDist;
using AscendC::MicroAPI::UpdateMask;

constexpr uint32_t VL_FP32 = platform::GetVRegSize() / sizeof(float);

constexpr AscendC::MicroAPI::CastTrait castTraitB16ToFp32 = {
    AscendC::MicroAPI::RegLayout::ZERO,
    AscendC::MicroAPI::SatMode::UNKNOWN,
    AscendC::MicroAPI::MaskMergeMode::ZEROING,
    AscendC::RoundMode::UNKNOWN,
};

// tile 从 UB load 进 fp32 寄存器（fp16/bf16 解包升精度）；offset 以元素计
template <typename T_IN>
__aicore__ inline void LoadToFp32(__ubuf__ T_IN* src, RegTensor<float>& dst, MaskReg& preg, uint32_t offset)
{
    if constexpr (std::is_same<T_IN, float>::value) {
        DataCopy<float, LoadDist::DIST_NORM>(dst, src + offset);
    } else {
        RegTensor<T_IN> xIn;
        DataCopy<T_IN, LoadDist::DIST_UNPACK_B16>(xIn, src + offset);
        Cast<float, T_IN, castTraitB16ToFp32>(dst, xIn, preg);
    }
}

// 满块 for + 尾块 0/1 次 for 的循环骨架参数（VF 内无 if）
struct VfLoops {
    uint16_t fullLoops;
    uint16_t totalLoops;
    uint32_t tailCount;
};

__aicore__ inline VfLoops MakeVfLoops(int64_t count)
{
    VfLoops lp;
    lp.fullLoops = static_cast<uint16_t>(count / static_cast<int64_t>(VL_FP32));
    lp.totalLoops = static_cast<uint16_t>((count + static_cast<int64_t>(VL_FP32) - 1) / static_cast<int64_t>(VL_FP32));
    lp.tailCount = static_cast<uint32_t>(count) - lp.fullLoops * VL_FP32;
    return lp;
}

template <typename T>
class BNTrainingUpdateGradKernel {
public:
    __aicore__ inline BNTrainingUpdateGradKernel() = default;

    __aicore__ inline void Init(GM_ADDR grads, GM_ADDR x, GM_ADDR batchMean, GM_ADDR batchVar, GM_ADDR diffScale,
                                GM_ADDR diffOffset, GM_ADDR workspace, const BNTrainingUpdateGradTilingData* tilingData,
                                TPipe* pipe)
    {
        pipe_ = pipe;
        tl_ = tilingData;
        // blockIdx 即切分维块号（ND/NHWC-channelSplit=通道段；NHWC-rowSplit=行段）：
        // ND/NHWC-channelSplit 下每 channel 的完整归约由唯一归属核完成（零核间通信）
        int64_t cRangeIdx = static_cast<int64_t>(GetBlockIdx());
        if (cRangeIdx < tl_->cFormerCoreNum) {
            cRangeLen_ = tl_->cFormerLen;
            cStart_ = cRangeIdx * tl_->cFormerLen;
        } else {
            cRangeLen_ = tl_->cLatterLen;
            cStart_ = tl_->cFormerCoreNum * tl_->cFormerLen + (cRangeIdx - tl_->cFormerCoreNum) * tl_->cLatterLen;
        }
        epsilon_ = tl_->epsilon;

        qCap_ = tl_->cLenCap * tl_->sliceR; // channel chunk 元素上限
        pitchElems_ = (qCap_ * static_cast<int64_t>(sizeof(T)) + 31) / 32 * 32 / static_cast<int64_t>(sizeof(T));
        cLenPad_ = (tl_->cLenCap + 7) / 8 * 8; // 归约 dst 行距上限（32B 对齐）

        int64_t gmLen = tl_->numN * tl_->numC * tl_->innerSize;
        if (tl_->isNhwc != 0) {
            gmLen = tl_->numN * tl_->numC; // NHWC：numN 即 rows，innerSize 恒 1
        }
        gradsGm_.SetGlobalBuffer((__gm__ T*)grads, gmLen);
        xGm_.SetGlobalBuffer((__gm__ T*)x, gmLen);
        meanGm_.SetGlobalBuffer((__gm__ float*)batchMean, tl_->numC);
        varGm_.SetGlobalBuffer((__gm__ float*)batchVar, tl_->numC);
        diffScaleGm_.SetGlobalBuffer((__gm__ float*)diffScale, tl_->numC);
        diffOffsetGm_.SetGlobalBuffer((__gm__ float*)diffOffset, tl_->numC);
        // 确定性 rowSplit：用户区 ws [核数][2][cLenCap] fp32。必须经 GetUserWorkspace 取用户区
        // 起始——950 运行时 workspace 头部保留 16MB 系统区（ffts 跨核同步 SyncAll 信箱所在），
        // 直绑原始地址写入会踩信箱触发 VECTOR_CORE_EXCEPTION（bn_training_reduce/lp_norm_reduce
        // 同款先例；当初 ws 尝试即败于此，非 ws 通路本身不可用）。绑定为 bn3d 同款单参形态。
        if (tl_->isNhwc != 0 && tl_->nhwcSplitMode == 2 && tl_->isDeterministic != 0) {
            wsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(GetUserWorkspace(workspace)));
        }

        if (tl_->isNhwc != 0) {
            InitNhwcBuffer();
            return;
        }
        // 全部 VF 读缓冲统一 +VL_FP32 槽位：非对齐尾块的前整 VL load 允许越入槽位（mask 屏蔽，不越界）；
        // 缓冲尺寸与 host tiling 的 perRowBytes/fixedBytes 预算公式严格一致
        int64_t rowPitch = tl_->rowsPerTile * pitchElems_ + VL_FP32;
        pipe_->InitBuffer(gQue_, DOUBLE_BUFFER, rowPitch * sizeof(T));
        pipe_->InitBuffer(xQue_, DOUBLE_BUFFER, rowPitch * sizeof(T));
        pipe_->InitBuffer(statMeanQue_, 1, (tl_->cLenCap + VL_FP32) * sizeof(float));
        pipe_->InitBuffer(statVarQue_, 1, (tl_->cLenCap + VL_FP32) * sizeof(float));
        pipe_->InitBuffer(outQue_, 1, (tl_->cLenCap + VL_FP32) * sizeof(float));
        if (tl_->cLenCap == 1) {
            // 快路:accRowG/P 复用为位置化 scratch(chunk+2VL 槽);expMean/expRstd 复用为
            // 尾部 64 槽;expMean2/expRstd2 为 Kahan 补偿缓冲;无 qCap 级展开系数/二维累加器
            pipe_->InitBuffer(accRowG_, (tl_->sliceR + 2 * VL_FP32) * sizeof(float));
            pipe_->InitBuffer(accRowP_, (tl_->sliceR + 2 * VL_FP32) * sizeof(float));
            pipe_->InitBuffer(expMean_, 2 * VL_FP32 * sizeof(float));
            pipe_->InitBuffer(expRstd_, 2 * VL_FP32 * sizeof(float));
            pipe_->InitBuffer(kahanOff_, 2 * VL_FP32 * sizeof(float));
            pipe_->InitBuffer(kahanScale_, 2 * VL_FP32 * sizeof(float));
        } else {
            pipe_->InitBuffer(accRowG_, (qCap_ + VL_FP32) * sizeof(float));
            pipe_->InitBuffer(accRowP_, (qCap_ + VL_FP32) * sizeof(float));
            pipe_->InitBuffer(expMean_, (qCap_ + VL_FP32) * sizeof(float));
            pipe_->InitBuffer(expRstd_, (qCap_ + VL_FP32) * sizeof(float));
        }
        pipe_->InitBuffer(dstG_, (cLenPad_ + VL_FP32) * sizeof(float));
        pipe_->InitBuffer(dstP_, (cLenPad_ + VL_FP32) * sizeof(float));
        pipe_->InitBuffer(accOff_, (tl_->cLenCap + VL_FP32) * sizeof(float));
        pipe_->InitBuffer(accScale_, (tl_->cLenCap + VL_FP32) * sizeof(float));
        pipe_->InitBuffer(rstdBuf_, (tl_->cLenCap + VL_FP32) * sizeof(float));
        pipe_->InitBuffer(reduceTmp_, REDUCE_TMP_BYTES);
    }

    __aicore__ inline void Process()
    {
        if (tl_->isNhwc != 0) {
            ProcessNhwc();
            return;
        }
        if (tl_->cLenCap == 1) {
            ProcessFast(); // 小 C 深归约快路(单 channel chunk,1D 连续段 + 位置化 scratch)
        } else {
            ProcessSlow();
        }
    }

private:
    // 慢路(通用):channel chunk × R 分片 × N 行 tile 三层循环
    __aicore__ inline void ProcessSlow()
    {
        // channel chunk 主循环：核内累加归约后直接写输出 GM（channel 归属唯一，零核间通信）
        int64_t cEnd = cStart_ + cRangeLen_;
        for (int64_t cPos = cStart_; cPos < cEnd; cPos += tl_->cLenCap) {
            int64_t len = (cEnd - cPos < tl_->cLenCap) ? (cEnd - cPos) : tl_->cLenCap;
            LocalTensor<float> meanUb = StageStat(statMeanQue_, meanGm_, cPos, len);
            LocalTensor<float> varUb = StageStat(statVarQue_, varGm_, cPos, len);
            ComputeRstd((__ubuf__ float*)varUb.GetPhyAddr(), len);
            ZeroBuf(accOff_, len);
            ZeroBuf(accScale_, len);
            LocalTensor<float> rstdTensor = rstdBuf_.Get<float>();
            for (int64_t r0 = 0; r0 < tl_->innerSize; r0 += tl_->sliceR) {
                int64_t rEff = (tl_->innerSize - r0 < tl_->sliceR) ? (tl_->innerSize - r0) : tl_->sliceR;
                BuildExpanded(meanUb, rstdTensor, len, rEff);
                ZeroBuf(accRowG_, len * rEff);
                ZeroBuf(accRowP_, len * rEff);
                ProcessRSlices(cPos, r0, len, rEff);
                ReduceSlice(len, rEff);
            }
            WritePartials(cPos, len);
            statMeanQue_.FreeTensor(meanUb);
            statVarQue_.FreeTensor(varUb);
        }
    }

    // 快路(cLenCap==1):逐 channel,逐 (n,c) 连续段 1D 大块 DMA + 标量系数 + 位置化 scratch;
    // channel 归属唯一,直写输出(零核间通信)
    __aicore__ inline void ProcessFast()
    {
        for (int64_t cPos = cStart_; cPos < cStart_ + cRangeLen_; cPos++) {
            LocalTensor<float> meanUb = StageStat(statMeanQue_, meanGm_, cPos, 1);
            LocalTensor<float> varUb = StageStat(statVarQue_, varGm_, cPos, 1);
            ComputeRstd((__ubuf__ float*)varUb.GetPhyAddr(), 1);
            // VF 写(rstd)→ 标量读的 V_S 同步
            event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventVS);
            WaitFlag<HardEvent::V_S>(eventVS);
            float meanVal = meanUb.GetValue(0);
            float rstdVal = rstdBuf_.Get<float>().GetValue(0);
            statMeanQue_.FreeTensor(meanUb);
            statVarQue_.FreeTensor(varUb);

            ZeroBuf(accOff_, 1);
            ZeroBuf(accScale_, 1);
            ZeroBuf(kahanOff_, VL_FP32); // Kahan 补偿位清零(每 channel)
            ZeroBuf(kahanScale_, VL_FP32);
            if (tl_->numC == 1) {
                // C==1:全平面 [N*R] 连续,摊平为单段 1D 循环(免逐段尾部开销)
                for (int64_t r0 = 0; r0 < tl_->numN * tl_->innerSize; r0 += tl_->sliceR) {
                    int64_t total = tl_->numN * tl_->innerSize;
                    int64_t eff = (total - r0 < tl_->sliceR) ? (total - r0) : tl_->sliceR;
                    AccumChunk(r0, eff, meanVal, rstdVal);
                }
            } else if (tl_->innerSize == 1 && tl_->numN > 1 && tl_->numN * tl_->numC <= tl_->sliceR) {
                // R==1 且 C>1 的微形态(如 256x3):channel 数据跨步 C,整平面一次 1D 搬入,
                // 标量按步长 C 抽取(标量读 DMA 数据 MTE2_S 同步,已验证形态)
                ProcessScalarTinyR(cPos, meanVal, rstdVal);
            } else if (tl_->innerSize < static_cast<int64_t>(VL_FP32) && tl_->numN > 1) {
                // 小 R 大 N 批处理:2D DataCopyPad 按 [nRows,R] 整批搬入(行距 C*R)
                ProcessSmallR(cPos, meanVal, rstdVal);
            } else {
                for (int64_t n = 0; n < tl_->numN; n++) {
                    int64_t segBase = (n * tl_->numC + cPos) * tl_->innerSize;
                    for (int64_t r0 = 0; r0 < tl_->innerSize; r0 += tl_->sliceR) {
                        int64_t eff = (tl_->innerSize - r0 < tl_->sliceR) ? (tl_->innerSize - r0) : tl_->sliceR;
                        AccumChunk(segBase + r0, eff, meanVal, rstdVal);
                    }
                }
            }
            WritePartials(cPos, 1);
        }
    }

    // ===================== NHWC 路径（isNhwc=1） =====================

    // NHWC 专属 buffer：g/x 双队列(tileRows×W×dsize×2) + fp32 常驻 8 条 (W+VL)
    // （statMean/statVar/rstd/accOff/accScale/kahanOff/kahanScale/out，与 host
    // SolveNhwcWindowMax/SolveNhwcTileRows 预算公式严格一致）+ rowSplit ws 读 tile
    __aicore__ inline void InitNhwcBuffer()
    {
        int64_t window = tl_->cLenCap;                  // W（64 对齐）
        int64_t tileRows = tl_->sliceR;                 // 行 tile 行数
        int64_t rowPitch = tileRows * window + VL_FP32; // +VL 槽位（尾块前整 VL load 越入）
        if (tl_->numC == 1) {
            // C==1 reroute：sliceR 语义为 1D chunk 元素数，g/x 为连续段缓冲（无行距 W）
            rowPitch = tileRows + VL_FP32;
        }
        pipe_->InitBuffer(gQue_, DOUBLE_BUFFER, rowPitch * sizeof(T));
        pipe_->InitBuffer(xQue_, DOUBLE_BUFFER, rowPitch * sizeof(T));
        pipe_->InitBuffer(statMeanQue_, 1, (window + VL_FP32) * sizeof(float));
        pipe_->InitBuffer(statVarQue_, 1, (window + VL_FP32) * sizeof(float));
        pipe_->InitBuffer(outQue_, 1, (window + VL_FP32) * sizeof(float));
        pipe_->InitBuffer(accOff_, (window + VL_FP32) * sizeof(float));
        pipe_->InitBuffer(accScale_, (window + VL_FP32) * sizeof(float));
        pipe_->InitBuffer(rstdBuf_, (window + VL_FP32) * sizeof(float));
        pipe_->InitBuffer(kahanOff_, (window + VL_FP32) * sizeof(float));
        pipe_->InitBuffer(kahanScale_, (window + VL_FP32) * sizeof(float));
        pipe_->InitBuffer(reduceTmp_, REDUCE_TMP_BYTES);
        if (tl_->nhwcSplitMode == 2 && tl_->isDeterministic != 0) {
            // 确定性 Phase2 收集暂存：[VL] fp32（逐核 1D 收集 + 固定序向量累加）；
            // host 侧 p2StageBytes_ 已计入 tileRows/chunk 预算，UB 不越界
            pipe_->InitBuffer(p2Que_, 1, VL_FP32 * sizeof(float));
        }
        if (tl_->numC == 1) {
            // C==1 reroute：AccumChunk 的 scratch/尾槽/归约 dst（ND 快路同款尺寸口径）
            pipe_->InitBuffer(accRowG_, (tileRows + 2 * VL_FP32) * sizeof(float));
            pipe_->InitBuffer(accRowP_, (tileRows + 2 * VL_FP32) * sizeof(float));
            pipe_->InitBuffer(expMean_, 2 * VL_FP32 * sizeof(float));
            pipe_->InitBuffer(expRstd_, 2 * VL_FP32 * sizeof(float));
            pipe_->InitBuffer(dstG_, (8 + VL_FP32) * sizeof(float));
            pipe_->InitBuffer(dstP_, (8 + VL_FP32) * sizeof(float));
        }
    }

    __aicore__ inline void ProcessNhwc()
    {
        if (tl_->nhwcSplitMode == 2) {
            ProcessNhwcRowSplit();
        } else {
            ProcessNhwcChannelSplit();
        }
    }

    // channelSplit：本核通道段 [cStart_, cStart_+cRangeLen_) 扫全部行；段长 > 窗预算时
    // c 窗口化（元素恰读一次，C 无上限）；acc 即逐通道结果，直写 GM（零核间通信）
    __aicore__ inline void ProcessNhwcChannelSplit()
    {
        int64_t cEnd = cStart_ + cRangeLen_;
        for (int64_t cw0 = cStart_; cw0 < cEnd; cw0 += tl_->cLenCap) {
            int64_t cnt = (cEnd - cw0 < tl_->cLenCap) ? (cEnd - cw0) : tl_->cLenCap;
            ProcessNhwcWindow(cw0, cnt, 0, tl_->numN);
            WritePartials(cw0, cnt);
        }
    }

    // rowSplit：本核行段 [cStart_, cStart_+cRangeLen_)×全部 C（单窗）
    // isDeterministic=1（ge.deterministic=1）：两段确定性归约（bn_training_reduce 确定性变体同款）
    //   Phase1 各核部分和写用户 ws 固定槽（槽序=核序）→ 全核 SyncAll → Phase2 全核按 C 段并行
    //   2D gather [核数,64] + ReduceSum(RA) 固定布局归约 → 直写输出；相同输入逐位复现。
    // isDeterministic=0（默认）：0 号核覆盖写输出 + SyncAll + 其余核浮点原子加直写（零 ws；
    // SetAtomicAdd + DataCopyPad + SetAtomicNone，bn_training_reduce 同款已验证形态）。原子和
    // 顺序不定但误差 O(eps)（部分和数量 = 核数 ≤64）——性能优先的缺省形态。
    // ws 注意：两段归约的 ws 必须经 GetUserWorkspace 取用户区（Init 已绑定）——直写原始 workspace
    // 指针会踩头部 16MB 系统区（ffts SyncAll 信箱）触发 VECTOR_CORE_EXCEPTION，当初"弃 ws"的根因。
    __aicore__ inline void ProcessNhwcRowSplit()
    {
        ProcessNhwcWindow(0, tl_->numC, cStart_, cStart_ + cRangeLen_);
        if (tl_->isDeterministic != 0) {
            RowSplitWritePartialsToWs();
            SyncAll();
            RowSplitGatherReduce();
            return;
        }
        if (GetBlockIdx() == 0) {
            // 0 号核覆盖写（初值无关）→ 全核栅障 → 其余核原子加：总和等价于"清零后全
            // 原子加"，输出对调用方的初始内容零依赖（TTK 输出预填全 1 亦正确）。
            // SyncAll 裸调用（kernel 侧零 ws 零 flag；batch mode 由 tiling 侧 rowSplit 分支设置）。
            WriteAccToGm(diffOffsetGm_, (__ubuf__ float*)accOff_.Get<float>().GetPhyAddr(), tl_->numC);
            WriteAccToGm(diffScaleGm_, (__ubuf__ float*)accScale_.Get<float>().GetPhyAddr(), tl_->numC);
            SyncAll();
            return;
        }
        SyncAll();
        SetAtomicAdd<float>();
        WriteAccToGm(diffOffsetGm_, (__ubuf__ float*)accOff_.Get<float>().GetPhyAddr(), tl_->numC);
        WriteAccToGm(diffScaleGm_, (__ubuf__ float*)accScale_.Get<float>().GetPhyAddr(), tl_->numC);
        SetAtomicNone();
    }

    // 确定性 Phase1：本核部分和 [C] 落用户 ws 固定槽（槽序=核序，Phase2 gather 布局由此固定）。
    // 槽布局 [核][2][cLenCap]（off 在前、scale 在后）；经 WriteAccToGm 的 outQue EnQue/DeQue
    // 完成 V→MTE3 同步，随后 SyncAll 栅障（跨核 GM 可见性由 barrier 保证，bn_training_reduce 同款）。
    __aicore__ inline void RowSplitWritePartialsToWs()
    {
        int64_t core = static_cast<int64_t>(GetBlockIdx());
        int64_t slot = tl_->cLenCap; // ws 槽宽（64 对齐，host/kernel 共用 cLenCap）
        // For C=1 the aligned workspace slot has room for the reduction residual.
        const int64_t count = tl_->numC == 1 ? RESIDUAL_INDEX + 1 : tl_->numC;
        WriteAccToGm(wsGm_[core * 2 * slot], (__ubuf__ float*)accOff_.Get<float>().GetPhyAddr(), count);
        WriteAccToGm(wsGm_[core * 2 * slot + slot], (__ubuf__ float*)accScale_.Get<float>().GetPhyAddr(), count);
    }

    // 确定性 Phase2：全核按 C 段并行（前多后少均分），每段固定布局收集全部核的部分和并归约。
    // 输出只由唯一归属核覆写（无原子），段内求和序 = ws 槽序 = 核序（跨段不交换）——逐位确定。
    __aicore__ inline void RowSplitGatherReduce()
    {
        int64_t cores = tl_->channelCores;
        int64_t cTotal = tl_->numC;
        int64_t core = static_cast<int64_t>(GetBlockIdx());
        int64_t cPer = (cTotal + cores - 1) / cores;
        int64_t c0 = core * cPer;
        int64_t c1 = (c0 + cPer < cTotal) ? (c0 + cPer) : cTotal;
        for (int64_t base = c0; base < c1; base += static_cast<int64_t>(VL_FP32)) {
            int64_t w = (c1 - base < static_cast<int64_t>(VL_FP32)) ? (c1 - base) : static_cast<int64_t>(VL_FP32);
            GatherReduceChunk(base, w, 0, diffOffsetGm_);
            GatherReduceChunk(base, w, 1, diffScaleGm_);
        }
    }

    // 单段确定性归约：逐核 1D DataCopyPad 整槽收集（blockLen 恒 VL*4=256B ≥ 32B 且无 padding，
    // 规避 rightPadding≤32B/sizeof(T) 的 MTE 约束——补零式收集的 rightPad=63 会触发 AI Core
    // MTE 异常 78；槽内 [w,64) 为 Phase1 未写的驻留垃圾，KahanAccumulateFrom 按 w 截断不进累加器）
    // + 固定核序 Kahan 向量累加（KahanAddReg 已验证形态，含 ±Inf 部和 NaN 防护），求和序 =
    // ws 槽序 = 核序——逐位确定。跨核部分和 ~cores 次裸 fp32 顺序加在 |acc|≫|partial| 的深
    // 归约下每步 0.5 ULP(|acc|) 截断（139 万行例 24 部分和偏 1 ULP 致 rmse 比值超限），Kahan
    // 压回 O(eps)；补偿位复用 Phase2 空闲的 kahanScale_。
    __aicore__ inline void GatherReduceChunk(int64_t c0, int64_t w, int64_t outIdx, GlobalTensor<float>& outGm)
    {
        int64_t cores = tl_->channelCores;
        int64_t slot = tl_->cLenCap;
        __ubuf__ float* accAddr = (__ubuf__ float*)kahanOff_.Get<float>().GetPhyAddr();    // Phase2 空闲复用
        __ubuf__ float* compAddr = (__ubuf__ float*)kahanScale_.Get<float>().GetPhyAddr(); // Phase2 空闲(补偿位)
        ZeroBuf(kahanOff_, static_cast<int64_t>(VL_FP32));                                 // 死 lane 恒 0
        ZeroBuf(kahanScale_, static_cast<int64_t>(VL_FP32));                               // 补偿位清零
        for (int64_t g = 0; g < cores; g++) {
            LocalTensor<float> stage = p2Que_.AllocTensor<float>();
            DataCopyExtParams cpIn{1, static_cast<uint32_t>(VL_FP32 * sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> padIn{false, 0, 0, 0.0f};
            DataCopyPad(stage, wsGm_[g * 2 * slot + outIdx * slot + c0], cpIn, padIn);
            p2Que_.EnQue(stage);
            stage = p2Que_.DeQue<float>();
            __ubuf__ float* stageAddr = (__ubuf__ float*)stage.GetPhyAddr();
            if (tl_->numC == 1) {
                __VEC_SCOPE__
                {
                    RegTensor<float> acc, partial, comp, next, err, zero;
                    MaskReg mask = CreateMask<float, MaskPattern::ALL>();
                    Duplicate(zero, 0.0f);
                    DataCopy<float, LoadDist::DIST_NORM>(acc, accAddr);
                    DataCopy<float, LoadDist::DIST_NORM>(partial, stageAddr);
                    DataCopy<float, LoadDist::DIST_NORM>(comp, compAddr);
                    SumWithError(next, err, acc, partial, mask);
                    SanitizeCompReg(err, zero, mask);
                    Sub(comp, comp, err, mask);
                    DataCopy<float, StoreDist::DIST_NORM>(accAddr, next, mask);
                    DataCopy<float, StoreDist::DIST_NORM>(compAddr, comp, mask);
                }
                PipeBarrier<PIPE_V>();
            } else {
                KahanAccumulateFrom(stageAddr, accAddr, compAddr, w);
            }
            p2Que_.FreeTensor(stage);
        }
        if (tl_->numC == 1) {
            __VEC_SCOPE__
            {
                RegTensor<float> hi, comp, lo, loComp;
                MaskReg mask = CreateMask<float, MaskPattern::VL1>();
                DataCopy<float, LoadDist::DIST_NORM>(hi, accAddr);
                DataCopy<float, LoadDist::DIST_NORM>(comp, compAddr);
                DataCopy<float, LoadDist::DIST_BRC_B32>(lo, accAddr + RESIDUAL_INDEX);
                DataCopy<float, LoadDist::DIST_BRC_B32>(loComp, compAddr + RESIDUAL_INDEX);
                Sub(lo, lo, loComp, mask);
                Sub(lo, lo, comp, mask);
                Add(hi, hi, lo, mask);
                DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(accAddr, hi, mask);
            }
            PipeBarrier<PIPE_V>();
        }
        WriteAccToGm(outGm[c0], accAddr, w);
    }

    // Phase2 Kahan 向量累加：src → acc（comp 为补偿位），逐 lane 独立补偿；loads 不带 mask、
    // 算子/存储按 cnt 截断（AccumulateFrom 同款形态），死 lane 垃圾不入累加器。
    __aicore__ inline void KahanAccumulateFrom(__ubuf__ float* srcAddr, __ubuf__ float* accAddr,
                                               __ubuf__ float* compAddr, int64_t cnt)
    {
        VfLoops lp = MakeVfLoops(cnt);
        __VEC_SCOPE__
        {
            RegTensor<float> srcReg;
            MaskReg fullMask = CreateMask<float, MaskPattern::ALL>();
            for (uint16_t i = 0; i < lp.fullLoops; i++) {
                uint32_t offset = i * VL_FP32;
                DataCopy<float, LoadDist::DIST_NORM>(srcReg, srcAddr + offset);
                KahanAddReg(accAddr + offset, compAddr + offset, srcReg, fullMask);
            }
            for (uint16_t i = lp.fullLoops; i < lp.totalLoops; i++) {
                MaskReg tailMask = UpdateMask<float>(lp.tailCount);
                uint32_t offset = i * VL_FP32;
                DataCopy<float, LoadDist::DIST_NORM>(srcReg, srcAddr + offset);
                KahanAddReg(accAddr + offset, compAddr + offset, srcReg, tailMask);
            }
        }
    }

    // 一个 (c 窗口, 行范围)：staging 窗口统计量 + 清 acc/kahan + 行 tile 循环（2D
    // DataCopyPad + ComputeNhwcTile）；行 tile 数由 tiling 的 sliceR=tileRows 下发
    __aicore__ inline void ProcessNhwcWindow(int64_t cw0, int64_t cnt, int64_t r0, int64_t r1)
    {
        if (tl_->numC == 1) {
            // C==1：mean/rstd 标量化 + 整段 1D 连续 DMA（ND 快路形态），行段即元素段
            LocalTensor<float> meanUb = StageStat(statMeanQue_, meanGm_, cw0, 1);
            LocalTensor<float> varUb = StageStat(statVarQue_, varGm_, cw0, 1);
            ComputeRstd((__ubuf__ float*)varUb.GetPhyAddr(), 1);
            event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventVS);
            WaitFlag<HardEvent::V_S>(eventVS);
            float meanVal = meanUb.GetValue(0);
            float rstdVal = rstdBuf_.Get<float>().GetValue(0);
            ZeroBuf(accOff_, VL_FP32); // 位置化 Kahan 累加器：全 64 lane 需清零
            ZeroBuf(accScale_, VL_FP32);
            ZeroBuf(kahanOff_, VL_FP32);
            ZeroBuf(kahanScale_, VL_FP32);
            if (tl_->isDeterministic != 0 || tl_->numN == 1) {
                rstdLow_ = RstdResidual(varUb.GetValue(0), rstdVal);
            }
            ProcessNhwcWindowC1(r0, r1, meanVal, rstdVal);
            FinalizeLaneAcc(); // 64 lane 累加器 → 标量（lane0），下游 WritePartials/rowSplit 不变
            statMeanQue_.FreeTensor(meanUb);
            statVarQue_.FreeTensor(varUb);
            return;
        }
        LocalTensor<float> meanUb = StageStat(statMeanQue_, meanGm_, cw0, cnt);
        LocalTensor<float> varUb = StageStat(statVarQue_, varGm_, cw0, cnt);
        ComputeRstd((__ubuf__ float*)varUb.GetPhyAddr(), cnt);
        // 清零扩到窗口整宽 W（而非 cnt）：acc/kahan 死 lane 从 0 起算，rowSplit 整 pitch
        // 写 ws + reduce 核整 64 宽列段求和时死 lane 恒 0（tile pad 零 × 任何 = 0 的
        // 不变式之外再兜一层——不依赖上一窗的驻留残渣）
        ZeroBuf(accOff_, tl_->cLenCap);
        ZeroBuf(accScale_, tl_->cLenCap);
        ZeroBuf(kahanOff_, tl_->cLenCap);
        ZeroBuf(kahanScale_, tl_->cLenCap);
        for (int64_t r = r0; r < r1; r += tl_->sliceR) {
            int64_t nr = (r1 - r < tl_->sliceR) ? (r1 - r) : tl_->sliceR;
            LocalTensor<T> gUb = gQue_.AllocTensor<T>();
            LocalTensor<T> xUb = xQue_.AllocTensor<T>();
            NhwcCopyRowTile(gUb, xUb, cw0, cnt, r, nr);
            gQue_.EnQue(gUb);
            gUb = gQue_.DeQue<T>();
            xQue_.EnQue(xUb);
            xUb = xQue_.DeQue<T>();
            auto* g = (__ubuf__ T*)gUb.GetPhyAddr();
            auto* x = (__ubuf__ T*)xUb.GetPhyAddr();
            auto* mean = (__ubuf__ float*)meanUb.GetPhyAddr();
            if constexpr (std::is_same<T, float>::value) {
                if (tl_->nhwcSplitMode == 1 && tl_->isDeterministic && tl_->numN > 1) {
                    ComputeNhwcTile<true>(g, x, nr, cnt, mean, r + nr == r1);
                } else {
                    ComputeNhwcTile<false>(g, x, nr, cnt, mean, false);
                }
            } else {
                ComputeNhwcTile<false>(g, x, nr, cnt, mean, false);
            }
            gQue_.FreeTensor(gUb);
            xQue_.FreeTensor(xUb);
        }
        statMeanQue_.FreeTensor(meanUb);
        statVarQue_.FreeTensor(varUb);
    }

    // C==1 特化（ND C==1 reroute 场景）：行宽 1 元素逐行 DMA 效率 1/64；改为整段连续 1D
    // 大块 DMA + 64 lane 位置化 Kahan 累加（跨 chunk 只累加 lane，不落中间标量；最终
    // FinalizeLaneAcc 一次 64 lane 归约）。深对消下避免「整 chunk 一次 fp32 ReduceSum +
    // chunk 部分和再次舍入」的两级 ULP 损失（issue110：2M 元素 C==1 精度 FAIL 根因）。
    __aicore__ inline void ProcessNhwcWindowC1(int64_t r0, int64_t r1, float meanVal, float rstdVal)
    {
        for (int64_t r = r0; r < r1; r += tl_->sliceR) {
            int64_t eff = (r1 - r < tl_->sliceR) ? (r1 - r) : tl_->sliceR;
            AccumChunkLane(r, eff, meanVal, rstdVal);
        }
    }

    // Copy all rows with one DMA per input. GM stride is measured in bytes;
    // UB stride counts 32-byte blocks after the rounded-up payload.
    __aicore__ inline void NhwcCopyRowTile(LocalTensor<T>& gUb, LocalTensor<T>& xUb, int64_t cw0, int64_t cnt,
                                           int64_t r0, int64_t nr)
    {
        const uint32_t bytes = cnt * sizeof(T);
        const uint64_t gap = static_cast<uint64_t>(tl_->numC - cnt) * sizeof(T);
        DataCopyExtParams cp{1, bytes, 0, 0, 0};
        DataCopyPadExtParams<T> pad{false, 0, 0, 0};
        const int64_t base = r0 * tl_->numC + cw0;
        if (gap <= 0xffffffffULL) {
            cp.blockCount = static_cast<uint16_t>(nr);
            cp.srcStride = static_cast<uint32_t>(gap);
            cp.dstStride = tl_->cLenCap * sizeof(T) / 32 - (bytes + 31) / 32;
            DataCopyPad(gUb, gradsGm_[base], cp, pad);
            DataCopyPad(xUb, xGm_[base], cp, pad);
        } else {
            for (int64_t i = 0; i < nr; ++i) {
                DataCopyPad(gUb[i * tl_->cLenCap], gradsGm_[base + i * tl_->numC], cp, pad);
                DataCopyPad(xUb[i * tl_->cLenCap], xGm_[base + i * tl_->numC], cp, pad);
            }
        }
    }

    // NHWC 单 tile 主计算：VL 块外层 × 行内层 + 块内寄存器累加器（慢路 ComputeTile 已验证
    // 结构），mean/rstd 从驻留 UB 直加载（免展开系数）；每 tile 每块一次 KahanAddReg 落
    // acc（UB RMW 频次 = 行 tile 数 × 块数，比逐行 RMW 低 3 个数量级）。尾窗 cnt<W 的死
    // lane：mean/rstd 驻留残渣 × tile pad 零 → 垃圾进 acc 死 lane，最终只写 [cnt] 元素。
    template <bool compensateScale>
    __aicore__ inline void ComputeNhwcTile(__ubuf__ T* gAddr, __ubuf__ T* xAddr, int64_t nr, int64_t cnt,
                                           __ubuf__ float* meanAddr, bool finalTile)
    {
        int64_t window = tl_->cLenCap;
        __ubuf__ float* rstdAddr = (__ubuf__ float*)rstdBuf_.Get<float>().GetPhyAddr();
        __ubuf__ float* accOffAddr = (__ubuf__ float*)accOff_.Get<float>().GetPhyAddr();
        __ubuf__ float* accScaleAddr = (__ubuf__ float*)accScale_.Get<float>().GetPhyAddr();
        __ubuf__ float* kahanOffAddr = (__ubuf__ float*)kahanOff_.Get<float>().GetPhyAddr();
        __ubuf__ float* kahanScaleAddr = (__ubuf__ float*)kahanScale_.Get<float>().GetPhyAddr();
        uint16_t blocks = static_cast<uint16_t>((cnt + VL_FP32 - 1) / VL_FP32); // 含尾块（死 lane 全 mask）
        uint16_t nrU = static_cast<uint16_t>(nr);
        __VEC_SCOPE__
        {
            RegTensor<float> gReg, xReg, mReg, sReg, tReg, accGReg, accPReg;
            MaskReg fullMask = CreateMask<float, MaskPattern::ALL>();
            for (uint16_t b = 0; b < blocks; b++) {
                uint32_t offset = b * VL_FP32;
                DataCopy<float, LoadDist::DIST_NORM>(mReg, meanAddr + offset);
                DataCopy<float, LoadDist::DIST_NORM>(sReg, rstdAddr + offset);
                Duplicate(accGReg, 0.0f);
                Duplicate(accPReg, 0.0f);
                RegTensor<float> low, zero, negativeMean;
                if constexpr (compensateScale) {
                    DataCopy<float, LoadDist::DIST_NORM>(accPReg, accScaleAddr + offset);
                    DataCopy<float, LoadDist::DIST_NORM>(low, kahanScaleAddr + offset);
                    Duplicate(zero, 0.0f);
                    Sub(negativeMean, zero, mReg, fullMask);
                }
                for (uint16_t row = 0; row < nrU; row++) {
                    LoadToFp32(gAddr + row * window, gReg, fullMask, offset);
                    LoadToFp32(xAddr + row * window, xReg, fullMask, offset);
                    Add(accGReg, accGReg, gReg, fullMask);
                    if constexpr (compensateScale) {
                        // Preserve the existing high-word operation order and IEEE results.
                        // Carry the discarded product and addition bits across tile boundaries.
                        RegTensor<float> d, de, norm, ne, productError, next, addError;
                        SumWithError(d, de, xReg, negativeMean, fullMask);
                        Mul(norm, d, sReg, fullMask);
                        Sub(ne, zero, norm, fullMask);
                        MulAddDst(ne, d, sReg, fullMask);
                        MulAddDst(ne, de, sReg, fullMask);
                        Mul(tReg, gReg, norm, fullMask);
                        Sub(productError, zero, tReg, fullMask);
                        MulAddDst(productError, gReg, norm, fullMask);
                        MulAddDst(productError, gReg, ne, fullMask);
                        SanitizeCompReg(productError, zero, fullMask);
                        SumWithError(next, addError, accPReg, tReg, fullMask);
                        SanitizeCompReg(addError, zero, fullMask);
                        Add(low, low, productError, fullMask);
                        Add(low, low, addError, fullMask);
                        Adds(accPReg, next, 0.0f, fullMask);
                    } else {
                        Sub(tReg, xReg, mReg, fullMask);
                        Mul(tReg, tReg, sReg, fullMask);
                        Mul(tReg, gReg, tReg, fullMask);
                        Add(accPReg, accPReg, tReg, fullMask);
                    }
                }
                KahanAddReg(accOffAddr + offset, kahanOffAddr + offset, accGReg, fullMask);
                if constexpr (compensateScale) {
                    if (finalTile) {
                        Add(accPReg, accPReg, low, fullMask);
                    }
                    DataCopy<float, StoreDist::DIST_NORM>(accScaleAddr + offset, accPReg, fullMask);
                    DataCopy<float, StoreDist::DIST_NORM>(kahanScaleAddr + offset, low, fullMask);
                } else {
                    KahanAddReg(accScaleAddr + offset, kahanScaleAddr + offset, accPReg, fullMask);
                }
            }
        }
    }

    // Kahan 补偿累加（寄存器源变体）：tileSumReg → acc，comp 为补偿位。KahanAdd（UB 源）
    // 保留供快路；深 rows 部分和上万次顺序累加的 fp32 截断由 Kahan 压到 O(eps)。
    // 非有限防护：部分和为 ±Inf 时补偿位 c=(t-sum)-y=Inf-Inf=NaN，后续 y=src-c=NaN 会
    // 毒化累加器（三方校验实测：golden/GPU 输出 ±Inf 而 NPU 输出 NaN）。NaN lane 补偿位
    // 清零后，非有限路径退化为 IEEE 普通加法语义（对齐 golden/torch.sum）；有限路径
    // c!=c 恒 false 原值保留，Kahan 精度不受影响。
    __aicore__ inline void KahanAddReg(__ubuf__ float* accAddr, __ubuf__ float* compAddr, RegTensor<float>& srcReg,
                                       MaskReg& mask)
    {
        RegTensor<float> sumReg, cReg, yReg, tReg, zeroReg;
        MaskReg nanMask;
        DataCopy<float, LoadDist::DIST_NORM>(sumReg, accAddr);
        DataCopy<float, LoadDist::DIST_NORM>(cReg, compAddr);
        Sub(yReg, srcReg, cReg, mask); // y = src - c
        Add(tReg, sumReg, yReg, mask); // t = sum + y
        Sub(cReg, tReg, sumReg, mask); // (t - sum)
        Sub(cReg, cReg, yReg, mask);   // c = (t - sum) - y
        Duplicate(zeroReg, 0.0f);
        MicroAPI::Compare<float, CMPMODE::NE>(nanMask, cReg, cReg, mask); // c != c ⟺ NaN lane
        MicroAPI::Select(cReg, zeroReg, cReg, nanMask);                   // c = nan ? 0 : c
        DataCopy<float, StoreDist::DIST_NORM>(accAddr, tReg, mask);
        DataCopy<float, StoreDist::DIST_NORM>(compAddr, cReg, mask);
    }

    // acc → GM 直写（rowSplit 写 ws 用；WritePartials 的单趟抽取，整 pitch 拷出，
    // 死 lane 垃圾入 ws 垫不出输出；经 outQue EnQue/DeQue 完成 V→MTE3 同步）
    __aicore__ inline void WriteAccToGm(GlobalTensor<float> dstGm, __ubuf__ float* accAddr, int64_t cnt)
    {
        DataCopyExtParams cpOut{1, static_cast<uint32_t>(cnt * sizeof(float)), 0, 0, 0};
        LocalTensor<float> outUb = outQue_.AllocTensor<float>();
        CopyAccToOut(accAddr, (__ubuf__ float*)outUb.GetPhyAddr(), cnt);
        outQue_.EnQue(outUb);
        outUb = outQue_.DeQue<float>();
        DataCopyPad(dstGm, outUb, cpOut);
        outQue_.FreeTensor(outUb);
    }

    // 标量读 T 并升 fp32(bf16 标量 cast 后端不支持,按位 <<16 提升,位精确)
    __aicore__ inline float ScalarToFp32(LocalTensor<T>& t, uint32_t idx)
    {
        if constexpr (std::is_same<T, float>::value) {
            return t.GetValue(idx);
        } else if constexpr (std::is_same<T, half>::value) {
            return static_cast<float>(t.GetValue(idx));
        } else {
            bfloat16_t v = t.GetValue(idx);
            uint16_t bits = *reinterpret_cast<uint16_t*>(&v);
            uint32_t fbits = static_cast<uint32_t>(bits) << 16;
            return *reinterpret_cast<float*>(&fbits);
        }
    }

    // Kahan 补偿累加(全 lane 同算,仅 lane0 有意义):src → acc,comp 为补偿位。
    // 深 R 下 chunk 部分和上千次顺序累加的 fp32 截断是精度瓶颈(实测 1x8x18.7M 例
    // 重对消 channel 误差 3.2e-3 超联合容差);Kahan 把累加误差从 O(n·eps) 降到 O(eps)。
    // 非有限防护:同 KahanAddReg——±Inf 部分和下补偿位 c=Inf-Inf=NaN,NaN lane 清零
    // 退化为 IEEE 普通加法(对齐 golden/torch.sum),有限路径 Kahan 精度不变。
    __aicore__ inline void KahanAdd(__ubuf__ float* accAddr, __ubuf__ float* compAddr, __ubuf__ float* srcAddr)
    {
        __VEC_SCOPE__
        {
            RegTensor<float> srcReg, sumReg, cReg, yReg, tReg, zeroReg;
            MaskReg fullMask = CreateMask<float, MaskPattern::ALL>();
            MaskReg nanMask;
            DataCopy<float, LoadDist::DIST_NORM>(srcReg, srcAddr);
            DataCopy<float, LoadDist::DIST_NORM>(sumReg, accAddr);
            DataCopy<float, LoadDist::DIST_NORM>(cReg, compAddr);
            Sub(yReg, srcReg, cReg, fullMask); // y = src - c
            Add(tReg, sumReg, yReg, fullMask); // t = sum + y
            Sub(cReg, tReg, sumReg, fullMask); // (t - sum)
            Sub(cReg, cReg, yReg, fullMask);   // c = (t - sum) - y
            Duplicate(zeroReg, 0.0f);
            MicroAPI::Compare<float, CMPMODE::NE>(nanMask, cReg, cReg, fullMask); // c != c ⟺ NaN lane
            MicroAPI::Select(cReg, zeroReg, cReg, nanMask);                       // c = nan ? 0 : c
            DataCopy<float, StoreDist::DIST_NORM>(accAddr, tReg, fullMask);
            DataCopy<float, StoreDist::DIST_NORM>(compAddr, cReg, fullMask);
        }
    }

    // 快路单 chunk:1D DataCopyPad 搬 grads/x 一段连续 eff 元素;64 整除部分走 VF 纯
    // elementwise 逐位置写 scratch(accRowG_/accRowP_ 复用;不同块不同地址,零 RMW 零寄存器
    // 跨块携带——实测寄存器跨块持有累加会被 __VEC_SCOPE__ 软流水偶发丢块),随后两发位置
    // 归约 [1,eff64]→[1] 累加进 accOff/accScale(慢路每 slice 同款已验证形态);
    // 尾部 eff%64(<64 元素)走独立 64 槽尾缓冲(expMean_/expRstd_ 复用):标量 SetValue
    // 写值+显式补零满 64 槽(实测 ZeroBuf 的掩码 store 与标量写同 buffer 组合不可靠;
    // 标量读 DMA 数据前必须 MTE2_S,缺则读陈旧 UB,r32 探针确定性复现),S_V 后
    // ReduceSum [1,64] 再累加。全程无掩码。
    __aicore__ inline void AccumChunk(int64_t gmOff, int64_t eff, float meanVal, float rstdVal)
    {
        DataCopyExtParams cpIn{1, static_cast<uint32_t>(eff * sizeof(T)), 0, 0, 0};
        uint32_t misalign = cpIn.blockLen & 31U;
        uint8_t rightPad = static_cast<uint8_t>(misalign == 0U ? 0U : (32U - misalign) / sizeof(T));
        DataCopyPadExtParams<T> padIn{true, 0, rightPad, static_cast<T>(0)};

        LocalTensor<T> gUb = gQue_.AllocTensor<T>();
        DataCopyPad(gUb, gradsGm_[gmOff], cpIn, padIn);
        gQue_.EnQue(gUb);
        gUb = gQue_.DeQue<T>();
        LocalTensor<T> xUb = xQue_.AllocTensor<T>();
        DataCopyPad(xUb, xGm_[gmOff], cpIn, padIn);
        xQue_.EnQue(xUb);
        xUb = xQue_.DeQue<T>();

        int64_t eff64 = eff / static_cast<int64_t>(VL_FP32) * static_cast<int64_t>(VL_FP32);
        if (eff64 > 0) {
            __ubuf__ T* gAddr = (__ubuf__ T*)gUb.GetPhyAddr();
            __ubuf__ T* xAddr = (__ubuf__ T*)xUb.GetPhyAddr();
            __ubuf__ float* scrGAddr = (__ubuf__ float*)accRowG_.Get<float>().GetPhyAddr();
            __ubuf__ float* scrPAddr = (__ubuf__ float*)accRowP_.Get<float>().GetPhyAddr();
            uint16_t blocks = static_cast<uint16_t>(eff64 / static_cast<int64_t>(VL_FP32)); // 恒整除
            float negMean = -meanVal; // MicroAPI 无 Subs:x+(-mean) 与 x-mean IEEE 逐位一致
            __VEC_SCOPE__
            {
                RegTensor<float> gReg, xReg, tReg;
                MaskReg fullMask = CreateMask<float, MaskPattern::ALL>();
                for (uint16_t i = 0; i < blocks; i++) {
                    uint32_t offset = i * VL_FP32;
                    LoadToFp32(gAddr, gReg, fullMask, offset);
                    LoadToFp32(xAddr, xReg, fullMask, offset);
                    Adds(tReg, xReg, negMean, fullMask);
                    Muls(tReg, tReg, rstdVal, fullMask);
                    Mul(tReg, gReg, tReg, fullMask);
                    DataCopy<float, StoreDist::DIST_NORM>(scrGAddr + offset, gReg, fullMask);
                    DataCopy<float, StoreDist::DIST_NORM>(scrPAddr + offset, tReg, fullMask);
                }
            }
            // 位置归约 [1,eff64]→[1] × 2 路(ReduceSum 后 PipeBarrier 排干 V pipe,同慢路教训)
            LocalTensor<float> scrG = accRowG_.Get<float>();
            LocalTensor<float> scrP = accRowP_.Get<float>();
            LocalTensor<float> dstGTensor = dstG_.Get<float>();
            LocalTensor<float> dstPTensor = dstP_.Get<float>();
            LocalTensor<uint8_t> reduceTmpTensor = reduceTmp_.Get<uint8_t>();
            uint32_t srcShape[2] = {1, static_cast<uint32_t>(eff64)};
            AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(dstGTensor, scrG, reduceTmpTensor, srcShape,
                                                                          false);
            AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(dstPTensor, scrP, reduceTmpTensor, srcShape,
                                                                          false);
            PipeBarrier<PIPE_V>();
            KahanAdd((__ubuf__ float*)accOff_.Get<float>().GetPhyAddr(),
                     (__ubuf__ float*)kahanOff_.Get<float>().GetPhyAddr(),
                     (__ubuf__ float*)dstG_.Get<float>().GetPhyAddr());
            KahanAdd((__ubuf__ float*)accScale_.Get<float>().GetPhyAddr(),
                     (__ubuf__ float*)kahanScale_.Get<float>().GetPhyAddr(),
                     (__ubuf__ float*)dstP_.Get<float>().GetPhyAddr());
        }
        // 标量尾(<64 元素):独立 64 槽尾缓冲,全部已验证原语
        int64_t rem = eff - eff64;
        if (rem > 0) {
            // DeQue 只给 MTE2→V 同步,标量管读 UB 前必须补 MTE2_S(实测缺失读陈旧 UB)
            event_t eventMS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
            SetFlag<HardEvent::MTE2_S>(eventMS);
            WaitFlag<HardEvent::MTE2_S>(eventMS);
            // 标量覆写 tailG/tailP 前需等上一轮向量 ReduceSum 读完(与 ProcessScalarTinyR 同款
            // V→S 危险:多轮/多 channel 复用尾缓冲时标量写可与向量读并发提交)
            event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventVS);
            WaitFlag<HardEvent::V_S>(eventVS);
            LocalTensor<float> tailG = expMean_.Get<float>();
            LocalTensor<float> tailP = expRstd_.Get<float>();
            for (int64_t k = eff64; k < eff; k++) {
                float gv = ScalarToFp32(gUb, static_cast<uint32_t>(k));
                float xv = ScalarToFp32(xUb, static_cast<uint32_t>(k));
                tailG.SetValue(static_cast<uint32_t>(k - eff64), gv);
                tailP.SetValue(static_cast<uint32_t>(k - eff64), gv * ((xv - meanVal) * rstdVal)); // 运算序对齐
            }
            for (int64_t k = rem; k < static_cast<int64_t>(VL_FP32); k++) { // 显式补零(不依赖 ZeroBuf)
                tailG.SetValue(static_cast<uint32_t>(k), 0.0f);
                tailP.SetValue(static_cast<uint32_t>(k), 0.0f);
            }
            event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(eventSV);
            WaitFlag<HardEvent::S_V>(eventSV);
            LocalTensor<float> dstGTensor = dstG_.Get<float>();
            LocalTensor<float> dstPTensor = dstP_.Get<float>();
            LocalTensor<uint8_t> reduceTmpTensor = reduceTmp_.Get<uint8_t>();
            uint32_t tailShape[2] = {1, VL_FP32};
            AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(dstGTensor, tailG, reduceTmpTensor, tailShape,
                                                                          false);
            AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(dstPTensor, tailP, reduceTmpTensor, tailShape,
                                                                          false);
            PipeBarrier<PIPE_V>();
            KahanAdd((__ubuf__ float*)accOff_.Get<float>().GetPhyAddr(),
                     (__ubuf__ float*)kahanOff_.Get<float>().GetPhyAddr(),
                     (__ubuf__ float*)dstG_.Get<float>().GetPhyAddr());
            KahanAdd((__ubuf__ float*)accScale_.Get<float>().GetPhyAddr(),
                     (__ubuf__ float*)kahanScale_.Get<float>().GetPhyAddr(),
                     (__ubuf__ float*)dstP_.Get<float>().GetPhyAddr());
        }
        gQue_.FreeTensor(gUb);
        xQue_.FreeTensor(xUb);
    }

    // 块尾 Kahan 合并（同族 bn3d_training_reduce 的块补偿形态）：blk → acc，comp 为补偿位。
    // c=(t-acc)-y 在 ±Inf 下退化为 NaN，必须 SanitizeCompReg 清零（否则回灌污染 acc）。
    __aicore__ inline void KahanMergeBlock(RegTensor<float>& acc, RegTensor<float>& comp, RegTensor<float>& blk,
                                           RegTensor<float>& y, RegTensor<float>& t, RegTensor<float>& zeroReg,
                                           MaskReg& mask)
    {
        Sub(y, blk, comp, mask);
        Add(t, acc, y, mask);
        Sub(comp, t, acc, mask);
        Sub(comp, comp, y, mask);
        SanitizeCompReg(comp, zeroReg, mask);
        Adds(acc, t, 0.0f, mask); // 寄存器搬移：+0 对有限值逐位精确，inf/nan 按 IEEE 保留
    }

    // 补偿量非有限值清零（同族 SanitizeComp）：+/-Inf 参与 TwoSum/Kahan 误差式会生成 NaN，
    // 主值已是正确传播结果，误差项按 0 处理；复用一个 MaskReg 避免额外寄存器。
    __aicore__ inline void SanitizeCompReg(RegTensor<float>& comp, RegTensor<float>& zeroReg, MaskReg& mask)
    {
        constexpr float FP32_MAX_FINITE = 3.4028234663852886e38f;
        MaskReg inFiniteRange;
        AscendC::Reg::Compares<float, CMPMODE::LE>(inFiniteRange, comp, FP32_MAX_FINITE, mask);
        Select(comp, comp, zeroReg, inFiniteRange);
        AscendC::Reg::Compares<float, CMPMODE::GE>(inFiniteRange, comp, -FP32_MAX_FINITE, mask);
        Select(comp, comp, zeroReg, inFiniteRange);
    }

    // C==1 位置化 Kahan 累加（NHWC rowSplit 专供）：products 按 lane 位置跨 chunk/跨段累加进
    // accOff_/accScale_（64 lane）与补偿位 kahanOff_/kahanScale_。
    // 动机：原「整 chunk(≈9792 元素) 一次 fp32 ReduceSum → 标量 → Kahan 跨 chunk」在深对消
    // （|Σ| ≪ Σ|项|，如 2M 元素求和结果 ~1e4 而 Σ|项| ~8e6）下有两级不可补偿舍入：
    // ① chunk 内求和自身误差 ~1e-3；② 每个 chunk 部分和落到 fp32 的舍入 ~ulp(部分和)。
    // 位置化 Kahan 让每个 lane 的累加误差保持 O(eps)，chunk 边界不再产生标量舍入；
    // 最终仅由 FinalizeLaneAcc 做一次 64 lane 归约（≤~0.5 ULP）。
    // Dekker product residual for normal fp32 intermediates. Splitting by bits
    // avoids an overflowing multiplication by the usual splitter constant.
    __aicore__ inline float ProductResidual(float a, float b, float product)
    {
        uint32_t ahBits = __builtin_bit_cast(uint32_t, a) & 0xfffff000U;
        uint32_t bhBits = __builtin_bit_cast(uint32_t, b) & 0xfffff000U;
        float ah = __builtin_bit_cast(float, ahBits);
        float bh = __builtin_bit_cast(float, bhBits);
        float al = a - ah;
        float bl = b - bh;
        return (((ah * bh - product) + ah * bl) + al * bh) + al * bl;
    }

    __aicore__ inline float SumResidual(float a, float b, float sum)
    {
        float tmp = sum - a;
        return (a - (sum - tmp)) + (b - tmp);
    }

    __aicore__ inline float RstdResidual(float var, float r)
    {
        float v = var + epsilon_;
        float rr = r * r;
        float vrr = v * rr;
        float residual = ((1.0f - vrr) - ProductResidual(v, rr, vrr)) -
                         (v * ProductResidual(r, r, rr) + SumResidual(var, epsilon_, v) * rr);
        float dr = (0.5f * residual) * r;
        return __builtin_isfinite(dr) && vrr > 0.5f && vrr < 1.5f ? dr : 0.0f;
    }

    __aicore__ inline void SumWithError(RegTensor<float>& sum, RegTensor<float>& err, RegTensor<float>& a,
                                        RegTensor<float>& b, MaskReg& mask)
    {
        RegTensor<float> v, t;
        Add(sum, a, b, mask);
        Sub(v, sum, a, mask);
        Sub(t, sum, v, mask);
        Sub(err, a, t, mask);
        Sub(t, b, v, mask);
        Add(err, err, t, mask);
    }

    // Track the high word and the accumulated error separately. Existing UB
    // compensation buffers store the negative low word (Kahan convention).
    __aicore__ inline void AccumPrecise(__ubuf__ T* grads, __ubuf__ T* x, uint32_t count, float mean, float rstd)
    {
        __ubuf__ float* off = (__ubuf__ float*)accOff_.Get<float>().GetPhyAddr();
        __ubuf__ float* scale = (__ubuf__ float*)accScale_.Get<float>().GetPhyAddr();
        __ubuf__ float* offComp = (__ubuf__ float*)kahanOff_.Get<float>().GetPhyAddr();
        __ubuf__ float* scaleComp = (__ubuf__ float*)kahanScale_.Get<float>().GetPhyAddr();
        const uint16_t blocks = (count + VL_FP32 - 1) / VL_FP32;
        const float dr = rstdLow_;
        __VEC_SCOPE__
        {
            MaskReg full = CreateMask<float, MaskPattern::ALL>();
            RegTensor<float> oh, oc, sh, sc, g, xv, m, r, rl, zero;
            RegTensor<float> d, de, norm, ne, prod, pe, next, err;
            DataCopy<float, LoadDist::DIST_NORM>(oh, off);
            DataCopy<float, LoadDist::DIST_NORM>(oc, offComp);
            DataCopy<float, LoadDist::DIST_NORM>(sh, scale);
            DataCopy<float, LoadDist::DIST_NORM>(sc, scaleComp);
            Duplicate(m, -mean);
            Duplicate(r, rstd);
            Duplicate(rl, dr);
            Duplicate(zero, 0.0f);
            uint32_t remaining = count;
            for (uint16_t block = 0; block < blocks; ++block) {
                MaskReg active = UpdateMask<float>(remaining);
                uint32_t pos = block * VL_FP32;
                LoadToFp32(grads, g, active, pos);
                LoadToFp32(x, xv, active, pos);
                SumWithError(d, de, xv, m, active);
                Mul(norm, d, r, active);
                Sub(ne, zero, norm, active);
                MulAddDst(ne, d, r, active);
                MulAddDst(ne, de, r, active);
                MulAddDst(ne, d, rl, active);
                Mul(prod, g, norm, active);
                Sub(pe, zero, prod, active);
                MulAddDst(pe, g, norm, active);
                MulAddDst(pe, g, ne, active);
                Select(prod, prod, zero, active);
                Select(g, g, zero, active);
                Select(pe, pe, zero, active);
                SanitizeCompReg(pe, zero, full);
                SumWithError(next, err, sh, prod, full);
                SanitizeCompReg(err, zero, full);
                Sub(sc, sc, err, full);
                Sub(sc, sc, pe, full);
                Adds(sh, next, 0.0f, full);
                SumWithError(next, err, oh, g, full);
                SanitizeCompReg(err, zero, full);
                Sub(oc, oc, err, full);
                Adds(oh, next, 0.0f, full);
            }
            DataCopy<float, StoreDist::DIST_NORM>(off, oh, full);
            DataCopy<float, StoreDist::DIST_NORM>(offComp, oc, full);
            DataCopy<float, StoreDist::DIST_NORM>(scale, sh, full);
            DataCopy<float, StoreDist::DIST_NORM>(scaleComp, sc, full);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void AccumChunkLane(int64_t gmOff, int64_t eff, float meanVal, float rstdVal)
    {
        DataCopyExtParams cpIn{1, static_cast<uint32_t>(eff * sizeof(T)), 0, 0, 0};
        uint32_t misalign = cpIn.blockLen & 31U;
        uint8_t rightPad = static_cast<uint8_t>(misalign == 0U ? 0U : (32U - misalign) / sizeof(T));
        DataCopyPadExtParams<T> padIn{true, 0, rightPad, static_cast<T>(0)};

        LocalTensor<T> gUb = gQue_.AllocTensor<T>();
        DataCopyPad(gUb, gradsGm_[gmOff], cpIn, padIn);
        gQue_.EnQue(gUb);
        gUb = gQue_.DeQue<T>();
        LocalTensor<T> xUb = xQue_.AllocTensor<T>();
        DataCopyPad(xUb, xGm_[gmOff], cpIn, padIn);
        xQue_.EnQue(xUb);
        xUb = xQue_.DeQue<T>();

        if (tl_->isDeterministic != 0 || tl_->numN == 1) {
            AccumPrecise((__ubuf__ T*)gUb.GetPhyAddr(), (__ubuf__ T*)xUb.GetPhyAddr(), static_cast<uint32_t>(eff),
                         meanVal, rstdVal);
            gQue_.FreeTensor(gUb);
            xQue_.FreeTensor(xUb);
            return;
        }

        __ubuf__ float* accOffAddr = (__ubuf__ float*)accOff_.Get<float>().GetPhyAddr();
        __ubuf__ float* accScaleAddr = (__ubuf__ float*)accScale_.Get<float>().GetPhyAddr();
        __ubuf__ float* kahanOffAddr = (__ubuf__ float*)kahanOff_.Get<float>().GetPhyAddr();
        __ubuf__ float* kahanScaleAddr = (__ubuf__ float*)kahanScale_.Get<float>().GetPhyAddr();

        int64_t eff64 = eff / static_cast<int64_t>(VL_FP32) * static_cast<int64_t>(VL_FP32);
        if (eff64 > 0) {
            __ubuf__ T* gAddr = (__ubuf__ T*)gUb.GetPhyAddr();
            __ubuf__ T* xAddr = (__ubuf__ T*)xUb.GetPhyAddr();
            uint16_t blocks = static_cast<uint16_t>(eff64 / static_cast<int64_t>(VL_FP32));
            float negMean = -meanVal; // MicroAPI 无 Subs:x+(-mean) 与 x-mean IEEE 逐位一致
            // 同族 bn3d_training_reduce 的分块 Kahan（KAHAN_BLOCK_CHUNKS=8）：块内 8 个 VL 块
            // 朴素累加（2 条/块），块尾一次 Kahan 合并（10 条/块组），摊到每块 2+10/8 条。
            // 逐块 Kahan 会因寄存器链与指令数使该用例 16.8us→75us；K=8 与同族实测结论一致。
            // acc/comp 常驻寄存器，仅在 scope 进出各一次 UB 读写（无 RMW hazard，无需 LocalMemBar）。
            constexpr uint16_t KAHAN_BLOCK_BLOCKS = 8;
            __VEC_SCOPE__
            {
                RegTensor<float> accGReg, compGReg, accPReg, compPReg;
                RegTensor<float> blkGReg, blkPReg, gReg, xReg, yReg, tReg, zeroReg;
                MaskReg fullMask = CreateMask<float, MaskPattern::ALL>();
                Duplicate(zeroReg, 0.0f);
                DataCopy<float, LoadDist::DIST_NORM>(accGReg, accOffAddr);
                DataCopy<float, LoadDist::DIST_NORM>(compGReg, kahanOffAddr);
                DataCopy<float, LoadDist::DIST_NORM>(accPReg, accScaleAddr);
                DataCopy<float, LoadDist::DIST_NORM>(compPReg, kahanScaleAddr);
                const uint16_t fullGroups = static_cast<uint16_t>(blocks / KAHAN_BLOCK_BLOCKS);
                const uint16_t tailBlocks = static_cast<uint16_t>(blocks - fullGroups * KAHAN_BLOCK_BLOCKS);
                for (uint16_t gi = 0; gi < fullGroups; gi++) {
                    Duplicate(blkGReg, 0.0f);
                    Duplicate(blkPReg, 0.0f);
                    for (uint16_t k = 0; k < KAHAN_BLOCK_BLOCKS; k++) {
                        uint32_t offset = static_cast<uint32_t>(gi * KAHAN_BLOCK_BLOCKS + k) * VL_FP32;
                        LoadToFp32(gAddr, gReg, fullMask, offset);
                        LoadToFp32(xAddr, xReg, fullMask, offset);
                        Adds(xReg, xReg, negMean, fullMask);
                        Muls(xReg, xReg, rstdVal, fullMask);
                        Mul(xReg, gReg, xReg, fullMask); // xReg 复用为乘积 p
                        Add(blkGReg, blkGReg, gReg, fullMask);
                        Add(blkPReg, blkPReg, xReg, fullMask);
                    }
                    KahanMergeBlock(accGReg, compGReg, blkGReg, yReg, tReg, zeroReg, fullMask);
                    KahanMergeBlock(accPReg, compPReg, blkPReg, yReg, tReg, zeroReg, fullMask);
                }
                if (tailBlocks > 0) {
                    Duplicate(blkGReg, 0.0f);
                    Duplicate(blkPReg, 0.0f);
                    for (uint16_t k = 0; k < tailBlocks; k++) {
                        uint32_t offset = static_cast<uint32_t>(fullGroups * KAHAN_BLOCK_BLOCKS + k) * VL_FP32;
                        LoadToFp32(gAddr, gReg, fullMask, offset);
                        LoadToFp32(xAddr, xReg, fullMask, offset);
                        Adds(xReg, xReg, negMean, fullMask);
                        Muls(xReg, xReg, rstdVal, fullMask);
                        Mul(xReg, gReg, xReg, fullMask);
                        Add(blkGReg, blkGReg, gReg, fullMask);
                        Add(blkPReg, blkPReg, xReg, fullMask);
                    }
                    KahanMergeBlock(accGReg, compGReg, blkGReg, yReg, tReg, zeroReg, fullMask);
                    KahanMergeBlock(accPReg, compPReg, blkPReg, yReg, tReg, zeroReg, fullMask);
                }
                DataCopy<float, StoreDist::DIST_NORM>(accOffAddr, accGReg, fullMask);
                DataCopy<float, StoreDist::DIST_NORM>(kahanOffAddr, compGReg, fullMask);
                DataCopy<float, StoreDist::DIST_NORM>(accScaleAddr, accPReg, fullMask);
                DataCopy<float, StoreDist::DIST_NORM>(kahanScaleAddr, compPReg, fullMask);
            }
        }
        // 标量尾(<64 元素):独立 64 槽尾缓冲,补零后整 VL 位置化累加（与主块同 lane 布局）
        int64_t rem = eff - eff64;
        if (rem > 0) {
            event_t eventMS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
            SetFlag<HardEvent::MTE2_S>(eventMS);
            WaitFlag<HardEvent::MTE2_S>(eventMS);
            // 标量覆写 tail 缓冲前需等上一轮向量 Kahan 读完（V→S 危险,同 AccumChunk）
            event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventVS);
            WaitFlag<HardEvent::V_S>(eventVS);
            LocalTensor<float> tailG = expMean_.Get<float>();
            LocalTensor<float> tailP = expRstd_.Get<float>();
            for (int64_t k = eff64; k < eff; k++) {
                float gv = ScalarToFp32(gUb, static_cast<uint32_t>(k));
                float xv = ScalarToFp32(xUb, static_cast<uint32_t>(k));
                tailG.SetValue(static_cast<uint32_t>(k - eff64), gv);
                tailP.SetValue(static_cast<uint32_t>(k - eff64), gv * ((xv - meanVal) * rstdVal));
            }
            for (int64_t k = rem; k < static_cast<int64_t>(VL_FP32); k++) { // 显式补零
                tailG.SetValue(static_cast<uint32_t>(k), 0.0f);
                tailP.SetValue(static_cast<uint32_t>(k), 0.0f);
            }
            event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(eventSV);
            WaitFlag<HardEvent::S_V>(eventSV);
            __ubuf__ float* tailGAddr = (__ubuf__ float*)tailG.GetPhyAddr();
            __ubuf__ float* tailPAddr = (__ubuf__ float*)tailP.GetPhyAddr();
            __VEC_SCOPE__
            {
                RegTensor<float> tgReg, tpReg;
                MaskReg fullMask = CreateMask<float, MaskPattern::ALL>();
                DataCopy<float, LoadDist::DIST_NORM>(tgReg, tailGAddr);
                DataCopy<float, LoadDist::DIST_NORM>(tpReg, tailPAddr);
                KahanAddReg(accOffAddr, kahanOffAddr, tgReg, fullMask);
                KahanAddReg(accScaleAddr, kahanScaleAddr, tpReg, fullMask);
            }
        }
        gQue_.FreeTensor(gUb);
        xQue_.FreeTensor(xUb);
    }

    // 64 lane 累加器 → 标量：同族 bn3d_training_reduce 的 CompensatedReduceSum——Knuth TwoSum
    // 补偿二叉树逐层携带低位，最后 hi+lo 落一次。普通 ReduceSum/Σacc-Σcomp 会在「大输出
    // 临界舍入点」丢 1 ULP（同族注释同款结论）。
    __aicore__ inline void CompensatedReduceToScalar(__ubuf__ float* accAddr, __ubuf__ float* compAddr,
                                                     __ubuf__ float* outAddr)
    {
        __VEC_SCOPE__
        {
            RegTensor<float> hi, comp, lo, lhsHi, rhsHi, lhsLo, rhsLo, tmp, err, zeroReg;
            RegTensor<int32_t> evenIndex, oddIndex;
            MaskReg fullMask = CreateMask<float, MaskPattern::ALL>();
            Duplicate(zeroReg, 0.0f);
            DataCopy<float, LoadDist::DIST_NORM>(hi, accAddr);
            DataCopy<float, LoadDist::DIST_NORM>(comp, compAddr);
            Muls(lo, comp, -1.0f, fullMask); // 更高精度的真值 = hi - comp
            AscendC::Reg::Arange(evenIndex, static_cast<int32_t>(0));
            ShiftLefts(evenIndex, evenIndex, static_cast<int16_t>(1), fullMask);
            Adds(oddIndex, evenIndex, static_cast<int32_t>(1), fullMask);
            for (uint16_t activeCount = static_cast<uint16_t>(VL_FP32 / 2U); activeCount > 0;
                 activeCount = static_cast<uint16_t>(activeCount / 2U)) {
                uint32_t activeWidth = activeCount;
                MaskReg activeMask = UpdateMask<float>(activeWidth);
                AscendC::Reg::Gather<float, uint32_t>(lhsHi, hi, (RegTensor<uint32_t>&)evenIndex);
                AscendC::Reg::Gather<float, uint32_t>(rhsHi, hi, (RegTensor<uint32_t>&)oddIndex);
                AscendC::Reg::Gather<float, uint32_t>(lhsLo, lo, (RegTensor<uint32_t>&)evenIndex);
                AscendC::Reg::Gather<float, uint32_t>(rhsLo, lo, (RegTensor<uint32_t>&)oddIndex);
                Add(hi, lhsHi, rhsHi, activeMask); // s
                Sub(tmp, hi, lhsHi, activeMask);   // bb = s - a
                Sub(err, hi, tmp, activeMask);     // s - bb
                Sub(err, lhsHi, err, activeMask);  // a - (s - bb)
                Sub(tmp, rhsHi, tmp, activeMask);  // b - bb
                Add(err, err, tmp, activeMask);    // TwoSum 误差项
                SanitizeCompReg(err, zeroReg, activeMask);
                Add(lo, lhsLo, rhsLo, activeMask);
                Add(lo, lo, err, activeMask); // 低位随层传播
            }
            // Preserve the low word in the padding of each deterministic C=1 slot.
            Add(tmp, hi, lo, fullMask);
            Sub(err, tmp, hi, fullMask);
            Sub(lhsLo, tmp, err, fullMask);
            Sub(lhsLo, hi, lhsLo, fullMask);
            Sub(err, lo, err, fullMask);
            Add(err, lhsLo, err, fullMask);
            SanitizeCompReg(err, zeroReg, fullMask);
            MaskReg oneMask = CreateMask<float, MaskPattern::VL1>();
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(outAddr, tmp, oneMask);
            DataCopy<float, StoreDist::DIST_FIRST_ELEMENT_B32>(outAddr + RESIDUAL_INDEX, err, oneMask);
        }
    }

    // 位置化累加器 → 标量（写回 accOff_/accScale_ lane0），下游 WritePartials/rowSplit 不变。
    __aicore__ inline void FinalizeLaneAcc()
    {
        __ubuf__ float* accOffAddr = (__ubuf__ float*)accOff_.Get<float>().GetPhyAddr();
        __ubuf__ float* accScaleAddr = (__ubuf__ float*)accScale_.Get<float>().GetPhyAddr();
        __ubuf__ float* kahanOffAddr = (__ubuf__ float*)kahanOff_.Get<float>().GetPhyAddr();
        __ubuf__ float* kahanScaleAddr = (__ubuf__ float*)kahanScale_.Get<float>().GetPhyAddr();
        CompensatedReduceToScalar(accOffAddr, kahanOffAddr, accOffAddr);
        CompensatedReduceToScalar(accScaleAddr, kahanScaleAddr, accScaleAddr);
        PipeBarrier<PIPE_V>();
    }

    // R==1 且 C>1 的微形态(如 256x3):整平面 [N*C] 一次 1D 搬入(channel c 的元素在
    // 平面内步长 C),标量按步长抽取逐位置写尾缓冲,64 块归约 + Kahan;标量读 DMA 数据
    // 前 MTE2_S 同步(r32 探针实证必需),SetValue 后 S_V(全部已验证原语)
    __aicore__ inline void ProcessScalarTinyR(int64_t cPos, float meanVal, float rstdVal)
    {
        int64_t plane = tl_->numN * tl_->numC;
        DataCopyExtParams cpIn{1, static_cast<uint32_t>(plane * sizeof(T)), 0, 0, 0};
        uint32_t misalign = cpIn.blockLen & 31U;
        uint8_t rightPad = static_cast<uint8_t>(misalign == 0U ? 0U : (32U - misalign) / sizeof(T));
        DataCopyPadExtParams<T> padIn{true, 0, rightPad, static_cast<T>(0)};

        LocalTensor<T> gUb = gQue_.AllocTensor<T>();
        DataCopyPad(gUb, gradsGm_[0], cpIn, padIn);
        gQue_.EnQue(gUb);
        gUb = gQue_.DeQue<T>();
        LocalTensor<T> xUb = xQue_.AllocTensor<T>();
        DataCopyPad(xUb, xGm_[0], cpIn, padIn);
        xQue_.EnQue(xUb);
        xUb = xQue_.DeQue<T>();

        event_t eventMS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(eventMS);
        WaitFlag<HardEvent::MTE2_S>(eventMS);

        LocalTensor<float> tailG = expMean_.Get<float>();
        LocalTensor<float> tailP = expRstd_.Get<float>();
        LocalTensor<float> dstGTensor = dstG_.Get<float>();
        LocalTensor<float> dstPTensor = dstP_.Get<float>();
        LocalTensor<uint8_t> reduceTmpTensor = reduceTmp_.Get<uint8_t>();
        for (int64_t n0 = 0; n0 < tl_->numN; n0 += VL_FP32) {
            int64_t cnt = (tl_->numN - n0 < static_cast<int64_t>(VL_FP32)) ? (tl_->numN - n0) :
                                                                             static_cast<int64_t>(VL_FP32);
            // 标量即将覆写 tailG/tailP：必须先等上一轮(或上一 channel)向量 ReduceSum 读完。
            // 缺此 V_S 时标量写可与向量读并发提交，令上一块部分和混入当前块 lane（实测 N≥2·VL
            // 的两轮形态随机丢/重计 lane，误差随运行时刻变化，N<2·VL 单轮不触发）。
            event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventVS);
            WaitFlag<HardEvent::V_S>(eventVS);
            for (int64_t k = 0; k < cnt; k++) {
                int64_t idx = (n0 + k) * tl_->numC + cPos;
                float gv = ScalarToFp32(gUb, static_cast<uint32_t>(idx));
                float xv = ScalarToFp32(xUb, static_cast<uint32_t>(idx));
                tailG.SetValue(static_cast<uint32_t>(k), gv);
                tailP.SetValue(static_cast<uint32_t>(k), gv * ((xv - meanVal) * rstdVal)); // 运算序对齐
            }
            for (int64_t k = cnt; k < static_cast<int64_t>(VL_FP32); k++) {
                tailG.SetValue(static_cast<uint32_t>(k), 0.0f);
                tailP.SetValue(static_cast<uint32_t>(k), 0.0f);
            }
            event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(eventSV);
            WaitFlag<HardEvent::S_V>(eventSV);
            uint32_t tailShape[2] = {1, VL_FP32};
            AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(dstGTensor, tailG, reduceTmpTensor, tailShape,
                                                                          false);
            AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(dstPTensor, tailP, reduceTmpTensor, tailShape,
                                                                          false);
            PipeBarrier<PIPE_V>();
            KahanAdd((__ubuf__ float*)accOff_.Get<float>().GetPhyAddr(),
                     (__ubuf__ float*)kahanOff_.Get<float>().GetPhyAddr(),
                     (__ubuf__ float*)dstG_.Get<float>().GetPhyAddr());
            KahanAdd((__ubuf__ float*)accScale_.Get<float>().GetPhyAddr(),
                     (__ubuf__ float*)kahanScale_.Get<float>().GetPhyAddr(),
                     (__ubuf__ float*)dstP_.Get<float>().GetPhyAddr());
        }
        gQue_.FreeTensor(gUb);
        xQue_.FreeTensor(xUb);
    }

    // 小 R 大 N 批处理(R<VL 且 N>1):按 [nRows,R] 2D 整批搬入(行距 C*R,行间连续),
    // nRows = sliceR/R;批内 rows*R 元素与 AccumChunk 同款处理(主部满块归约+尾部独立 64 槽)
    __aicore__ inline void ProcessSmallR(int64_t cPos, float meanVal, float rstdVal)
    {
        int64_t rSize = tl_->innerSize;
        int64_t nRowsCap = tl_->sliceR / ((rSize * static_cast<int64_t>(sizeof(T)) + 31) / 32 * 32 /
                                          static_cast<int64_t>(sizeof(T))); // 按 32B 对齐行距反推
        if (nRowsCap < 1) {
            nRowsCap = 1;
        }
        // 2D DataCopyPad 的行内 rightPad 按 32B 对齐逐行填充(padValue=0),UB 行距为
        // pitchElems(慢路同款);批内元素按 rows*pitchElems 计,pad 零经 (0-mean)*rstd*0=0
        // 恒无贡献,数学等价(实测:按 rows*rSize 线性读会把 pad 当数据,256x1 例全错)
        int64_t pitchElems = (rSize * static_cast<int64_t>(sizeof(T)) + 31) / 32 * 32 / static_cast<int64_t>(sizeof(T));
        for (int64_t n0 = 0; n0 < tl_->numN; n0 += nRowsCap) {
            int64_t rows = (tl_->numN - n0 < nRowsCap) ? (tl_->numN - n0) : nRowsCap;
            int64_t elems = rows * pitchElems;
            int64_t gmOff = (n0 * tl_->numC + cPos) * rSize;
            DataCopyExtParams cpIn{static_cast<uint16_t>(rows), static_cast<uint32_t>(rSize * sizeof(T)),
                                   (tl_->numC * rSize - rSize) * static_cast<int64_t>(sizeof(T)), 0, 0};
            uint32_t misalign = cpIn.blockLen & 31U;
            uint8_t rightPad = static_cast<uint8_t>(misalign == 0U ? 0U : (32U - misalign) / sizeof(T));
            DataCopyPadExtParams<T> padIn{true, 0, rightPad, static_cast<T>(0)};

            LocalTensor<T> gUb = gQue_.AllocTensor<T>();
            DataCopyPad(gUb, gradsGm_[gmOff], cpIn, padIn);
            gQue_.EnQue(gUb);
            gUb = gQue_.DeQue<T>();
            LocalTensor<T> xUb = xQue_.AllocTensor<T>();
            DataCopyPad(xUb, xGm_[gmOff], cpIn, padIn);
            xQue_.EnQue(xUb);
            xUb = xQue_.DeQue<T>();

            int64_t eff64 = elems / static_cast<int64_t>(VL_FP32) * static_cast<int64_t>(VL_FP32);
            if (eff64 > 0) {
                __ubuf__ T* gAddr = (__ubuf__ T*)gUb.GetPhyAddr();
                __ubuf__ T* xAddr = (__ubuf__ T*)xUb.GetPhyAddr();
                __ubuf__ float* scrGAddr = (__ubuf__ float*)accRowG_.Get<float>().GetPhyAddr();
                __ubuf__ float* scrPAddr = (__ubuf__ float*)accRowP_.Get<float>().GetPhyAddr();
                uint16_t blocks = static_cast<uint16_t>(eff64 / static_cast<int64_t>(VL_FP32));
                float negMean = -meanVal;
                __VEC_SCOPE__
                {
                    RegTensor<float> gReg, xReg, tReg;
                    MaskReg fullMask = CreateMask<float, MaskPattern::ALL>();
                    for (uint16_t i = 0; i < blocks; i++) {
                        uint32_t offset = i * VL_FP32;
                        LoadToFp32(gAddr, gReg, fullMask, offset);
                        LoadToFp32(xAddr, xReg, fullMask, offset);
                        Adds(tReg, xReg, negMean, fullMask);
                        Muls(tReg, tReg, rstdVal, fullMask);
                        Mul(tReg, gReg, tReg, fullMask);
                        DataCopy<float, StoreDist::DIST_NORM>(scrGAddr + offset, gReg, fullMask);
                        DataCopy<float, StoreDist::DIST_NORM>(scrPAddr + offset, tReg, fullMask);
                    }
                }
                LocalTensor<float> scrG = accRowG_.Get<float>();
                LocalTensor<float> scrP = accRowP_.Get<float>();
                LocalTensor<float> dstGTensor = dstG_.Get<float>();
                LocalTensor<float> dstPTensor = dstP_.Get<float>();
                LocalTensor<uint8_t> reduceTmpTensor = reduceTmp_.Get<uint8_t>();
                uint32_t srcShape[2] = {1, static_cast<uint32_t>(eff64)};
                AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(dstGTensor, scrG, reduceTmpTensor,
                                                                              srcShape, false);
                AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(dstPTensor, scrP, reduceTmpTensor,
                                                                              srcShape, false);
                PipeBarrier<PIPE_V>();
                KahanAdd((__ubuf__ float*)accOff_.Get<float>().GetPhyAddr(),
                         (__ubuf__ float*)kahanOff_.Get<float>().GetPhyAddr(),
                         (__ubuf__ float*)dstG_.Get<float>().GetPhyAddr());
                KahanAdd((__ubuf__ float*)accScale_.Get<float>().GetPhyAddr(),
                         (__ubuf__ float*)kahanScale_.Get<float>().GetPhyAddr(),
                         (__ubuf__ float*)dstP_.Get<float>().GetPhyAddr());
            }
            // 尾部 elems%64(每批至多 63 元素):独立 64 槽尾缓冲(AccumChunk 同款已验证形态)
            int64_t rem = elems - eff64;
            if (rem > 0) {
                event_t eventMS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
                SetFlag<HardEvent::MTE2_S>(eventMS);
                WaitFlag<HardEvent::MTE2_S>(eventMS);
                // 标量覆写 tailG/tailP 前需等上一批向量 ReduceSum 读完(同 ProcessScalarTinyR 的
                // V→S 危险,仅当按 nRowsCap 分批且末批 rem>0 才会多轮复用尾缓冲)
                event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
                SetFlag<HardEvent::V_S>(eventVS);
                WaitFlag<HardEvent::V_S>(eventVS);
                LocalTensor<float> tailG = expMean_.Get<float>();
                LocalTensor<float> tailP = expRstd_.Get<float>();
                for (int64_t k = eff64; k < elems; k++) {
                    float gv = ScalarToFp32(gUb, static_cast<uint32_t>(k));
                    float xv = ScalarToFp32(xUb, static_cast<uint32_t>(k));
                    tailG.SetValue(static_cast<uint32_t>(k - eff64), gv);
                    tailP.SetValue(static_cast<uint32_t>(k - eff64), gv * ((xv - meanVal) * rstdVal));
                }
                for (int64_t k = rem; k < static_cast<int64_t>(VL_FP32); k++) {
                    tailG.SetValue(static_cast<uint32_t>(k), 0.0f);
                    tailP.SetValue(static_cast<uint32_t>(k), 0.0f);
                }
                event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
                SetFlag<HardEvent::S_V>(eventSV);
                WaitFlag<HardEvent::S_V>(eventSV);
                LocalTensor<float> dstGTensor = dstG_.Get<float>();
                LocalTensor<float> dstPTensor = dstP_.Get<float>();
                LocalTensor<uint8_t> reduceTmpTensor = reduceTmp_.Get<uint8_t>();
                uint32_t tailShape[2] = {1, VL_FP32};
                AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(dstGTensor, tailG, reduceTmpTensor,
                                                                              tailShape, false);
                AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(dstPTensor, tailP, reduceTmpTensor,
                                                                              tailShape, false);
                PipeBarrier<PIPE_V>();
                KahanAdd((__ubuf__ float*)accOff_.Get<float>().GetPhyAddr(),
                         (__ubuf__ float*)kahanOff_.Get<float>().GetPhyAddr(),
                         (__ubuf__ float*)dstG_.Get<float>().GetPhyAddr());
                KahanAdd((__ubuf__ float*)accScale_.Get<float>().GetPhyAddr(),
                         (__ubuf__ float*)kahanScale_.Get<float>().GetPhyAddr(),
                         (__ubuf__ float*)dstP_.Get<float>().GetPhyAddr());
            }
            gQue_.FreeTensor(gUb);
            xQue_.FreeTensor(xUb);
        }
    }

private:
    // 统计量 staging：GM [cStart, cStart+cnt) → UB，EnQue/DeQue 完成 MTE2→V 同步
    __aicore__ inline LocalTensor<float> StageStat(TQue<QuePosition::VECIN, 1>& que, GlobalTensor<float>& gm,
                                                   int64_t cStart, int64_t cnt)
    {
        LocalTensor<float> ub = que.AllocTensor<float>();
        DataCopyExtParams cpIn{1, static_cast<uint32_t>(cnt * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padIn{false, 0, 0, 0};
        DataCopyPad(ub, gm[cStart], cpIn, padIn);
        que.EnQue(ub);
        return que.DeQue<float>();
    }

    // rstd = 1 / sqrt(var + epsilon)：逐 chunk 向量化（Adds→Sqrt→Div 两步舍入，对齐 A2 vadds→vsqrt→vdiv(1,·)）
    __aicore__ inline void ComputeRstd(__ubuf__ float* varAddr, int64_t cnt)
    {
        __ubuf__ float* rstdAddr = (__ubuf__ float*)rstdBuf_.Get<float>().GetPhyAddr();
        VfLoops lp = MakeVfLoops(cnt);
        __VEC_SCOPE__
        {
            RegTensor<float> varReg, oneReg;
            MaskReg fullMask = CreateMask<float, MaskPattern::ALL>();
            Duplicate(oneReg, 1.0f);
            for (uint16_t i = 0; i < lp.fullLoops; i++) {
                uint32_t offset = i * VL_FP32;
                DataCopy<float, LoadDist::DIST_NORM>(varReg, varAddr + offset);
                Adds(varReg, varReg, epsilon_, fullMask);
                Sqrt(varReg, varReg, fullMask);
                Div(varReg, oneReg, varReg, fullMask);
                DataCopy<float, StoreDist::DIST_NORM>(rstdAddr + offset, varReg, fullMask);
            }
            for (uint16_t i = lp.fullLoops; i < lp.totalLoops; i++) { // 尾块 0 或 1 次
                uint32_t tail = lp.tailCount;
                MaskReg tailMask = UpdateMask<float>(tail);
                uint32_t offset = i * VL_FP32;
                DataCopy<float, LoadDist::DIST_NORM>(varReg, varAddr + offset);
                Adds(varReg, varReg, epsilon_, tailMask);
                Sqrt(varReg, varReg, tailMask);
                Div(varReg, oneReg, varReg, tailMask);
                DataCopy<float, StoreDist::DIST_NORM>(rstdAddr + offset, varReg, tailMask);
            }
        }
    }

    // TBuf 前 cnt 个元素清零（accOff/accScale 每 chunk 一次；accRow 每 rSlice 一次）
    __aicore__ inline void ZeroBuf(TBuf<>& buf, int64_t cnt)
    {
        __ubuf__ float* addr = (__ubuf__ float*)buf.Get<float>().GetPhyAddr();
        VfLoops lp = MakeVfLoops(cnt);
        __VEC_SCOPE__
        {
            RegTensor<float> zeroReg;
            MaskReg fullMask = CreateMask<float, MaskPattern::ALL>();
            Duplicate(zeroReg, 0.0f);
            for (uint16_t i = 0; i < lp.fullLoops; i++) {
                DataCopy<float, StoreDist::DIST_NORM>(addr + i * VL_FP32, zeroReg, fullMask);
            }
            for (uint16_t i = lp.fullLoops; i < lp.totalLoops; i++) {
                uint32_t tail = lp.tailCount;
                MaskReg tailMask = UpdateMask<float>(tail);
                DataCopy<float, StoreDist::DIST_NORM>(addr + i * VL_FP32, zeroReg, tailMask);
            }
        }
    }

    // 展开系数构建：expMean/expRstd[j*rEff + r] = mean[j]/rstd[j]（一次性预计算，主循环按
    // per-element DIST_NORM 直接向量加载，无需广播）。rEff%8==0 走向量化广播（DIST_BRC_B32
    // 广播 load + mask store，32B 对齐）；否则标量 SetValue 兜底（MicroAPI 对齐 store 要求
    // 32B 地址；SetValue/GetValue 为 arch35 实证可用的标量 UB 访问 idiom，一次性成本 ≤ qCap 次写）
    __aicore__ inline void BuildExpanded(LocalTensor<float>& meanTensor, LocalTensor<float>& rstdTensor, int64_t len,
                                         int64_t rEff)
    {
        __ubuf__ float* meanAddr = (__ubuf__ float*)meanTensor.GetPhyAddr();
        __ubuf__ float* rstdAddr = (__ubuf__ float*)rstdTensor.GetPhyAddr();
        __ubuf__ float* expMeanAddr = (__ubuf__ float*)expMean_.Get<float>().GetPhyAddr();
        __ubuf__ float* expRstdAddr = (__ubuf__ float*)expRstd_.Get<float>().GetPhyAddr();
        if ((rEff & 7) != 0) {
            // ComputeRstd 的 VF 写 → 标量 GetValue 读，先 V_S 事件同步
            event_t eventVS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventVS);
            WaitFlag<HardEvent::V_S>(eventVS);
            LocalTensor<float> expMeanTensor = expMean_.Get<float>();
            LocalTensor<float> expRstdTensor = expRstd_.Get<float>();
            for (int64_t j = 0; j < len; j++) {
                float meanVal = meanTensor.GetValue(j);
                float rstdVal = rstdTensor.GetValue(j);
                for (int64_t r = 0; r < rEff; r++) {
                    expMeanTensor.SetValue(j * rEff + r, meanVal);
                    expRstdTensor.SetValue(j * rEff + r, rstdVal);
                }
            }
            // 标量写 → VF 读的跨 pipe 同步（S_V 事件，add_rms_norm_dynamic_quant 同款 idiom）
            event_t eventSV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
            SetFlag<HardEvent::S_V>(eventSV);
            WaitFlag<HardEvent::S_V>(eventSV);
            return;
        }
        VfLoops lp = MakeVfLoops(rEff);
        uint16_t lenU = static_cast<uint16_t>(len);
        __VEC_SCOPE__
        {
            RegTensor<float> meanReg, rstdReg;
            MaskReg fullMask = CreateMask<float, MaskPattern::ALL>();
            for (uint16_t j = 0; j < lenU; j++) {
                DataCopy<float, LoadDist::DIST_BRC_B32>(meanReg, meanAddr + j);
                DataCopy<float, LoadDist::DIST_BRC_B32>(rstdReg, rstdAddr + j);
                for (uint16_t i = 0; i < lp.fullLoops; i++) {
                    uint32_t offset = static_cast<uint32_t>(j) * static_cast<uint32_t>(rEff) + i * VL_FP32;
                    DataCopy<float, StoreDist::DIST_NORM>(expMeanAddr + offset, meanReg, fullMask);
                    DataCopy<float, StoreDist::DIST_NORM>(expRstdAddr + offset, rstdReg, fullMask);
                }
                for (uint16_t i = lp.fullLoops; i < lp.totalLoops; i++) {
                    uint32_t tail = lp.tailCount;
                    MaskReg tailMask = UpdateMask<float>(tail);
                    uint32_t offset = static_cast<uint32_t>(j) * static_cast<uint32_t>(rEff) + i * VL_FP32;
                    DataCopy<float, StoreDist::DIST_NORM>(expMeanAddr + offset, meanReg, tailMask);
                    DataCopy<float, StoreDist::DIST_NORM>(expRstdAddr + offset, rstdReg, tailMask);
                }
            }
        }
    }

    // 一个 (chunk, rSlice) 内的 N 行流式处理：多行 2D DMA 摊薄搬运，tile 内 VF 计算 + 按位累加
    __aicore__ inline void ProcessRSlices(int64_t cPos, int64_t r0, int64_t len, int64_t rEff)
    {
        int64_t qEff = len * rEff;
        for (int64_t n = 0; n < tl_->numN; n += tl_->rowsPerTile) {
            int64_t nr = (tl_->numN - n < tl_->rowsPerTile) ? (tl_->numN - n) : tl_->rowsPerTile;
            int64_t gmBase = n * tl_->numC * tl_->innerSize + cPos * tl_->innerSize + r0;
            DataCopyExtParams cpIn{static_cast<uint16_t>(nr), static_cast<uint32_t>(qEff * sizeof(T)),
                                   (tl_->numC * tl_->innerSize - qEff) * static_cast<int64_t>(sizeof(T)), 0, 0};
            uint32_t misalign = cpIn.blockLen & 31U;
            uint8_t rightPad = static_cast<uint8_t>(misalign == 0U ? 0U : (32U - misalign) / sizeof(T));
            DataCopyPadExtParams<T> padIn{true, 0, rightPad, static_cast<T>(0)};

            LocalTensor<T> gUb = gQue_.AllocTensor<T>();
            DataCopyPad(gUb, gradsGm_[gmBase], cpIn, padIn);
            gQue_.EnQue(gUb);
            gUb = gQue_.DeQue<T>();
            LocalTensor<T> xUb = xQue_.AllocTensor<T>();
            DataCopyPad(xUb, xGm_[gmBase], cpIn, padIn);
            xQue_.EnQue(xUb);
            xUb = xQue_.DeQue<T>();

            // DMA 行距按本 tile 实际 qEff（len<cLenCap 的尾 chunk 与最大 pitchElems_ 不同）
            int64_t pitchElems = (qEff * static_cast<int64_t>(sizeof(T)) + 31) / 32 * 32 /
                                 static_cast<int64_t>(sizeof(T));
            ComputeTile((__ubuf__ T*)gUb.GetPhyAddr(), (__ubuf__ T*)xUb.GetPhyAddr(), nr, len, rEff, pitchElems);

            gQue_.FreeTensor(gUb);
            xQue_.FreeTensor(xUb);
        }
    }

    // 单 tile 主计算：逐元素 t=(x-expMean)*expRstd、p=grads*t（运算序对齐 A2 TBE），
    // 并按元素位置向量累加进 accRowG/accRowP[0, qEff)（跨 n 行持久，(chunk,rSlice) 末统一归约）。
    // 循环结构为【VL 块在外、n 行在内 + 寄存器累加器】：accRow 的读改写每个 VL 块仅一次
    // （不同块不同地址，无跨迭代别名），规避 __VEC_SCOPE__ 软流水把下行 load 排到上行
    // store 前导致的丢行（实测：行在外时编译器重排 accRow 读改写，仅末行生效）
    __aicore__ inline void ComputeTile(__ubuf__ T* gAddr, __ubuf__ T* xAddr, int64_t nr, int64_t len, int64_t rEff,
                                       int64_t pitchElems)
    {
        int64_t qEff = len * rEff;
        __ubuf__ float* accGAddr = (__ubuf__ float*)accRowG_.Get<float>().GetPhyAddr();
        __ubuf__ float* accPAddr = (__ubuf__ float*)accRowP_.Get<float>().GetPhyAddr();
        __ubuf__ float* expMeanAddr = (__ubuf__ float*)expMean_.Get<float>().GetPhyAddr();
        __ubuf__ float* expRstdAddr = (__ubuf__ float*)expRstd_.Get<float>().GetPhyAddr();
        VfLoops lp = MakeVfLoops(qEff);
        uint16_t nrU = static_cast<uint16_t>(nr);
        __VEC_SCOPE__
        {
            RegTensor<float> gReg, xReg, mReg, sReg, tReg, accGReg, accPReg, accReg;
            MaskReg fullMask = CreateMask<float, MaskPattern::ALL>();
            for (uint16_t i = 0; i < lp.fullLoops; i++) {
                uint32_t offset = i * VL_FP32; // 展开系数仅 qEff 项，每块加载一次（不随 n 行重复）
                DataCopy<float, LoadDist::DIST_NORM>(mReg, expMeanAddr + offset);
                DataCopy<float, LoadDist::DIST_NORM>(sReg, expRstdAddr + offset);
                Duplicate(accGReg, 0.0f);
                Duplicate(accPReg, 0.0f);
                for (uint16_t row = 0; row < nrU; row++) {
                    LoadToFp32(gAddr + row * pitchElems, gReg, fullMask, offset);
                    LoadToFp32(xAddr + row * pitchElems, xReg, fullMask, offset);
                    Sub(tReg, xReg, mReg, fullMask);
                    Mul(tReg, tReg, sReg, fullMask);
                    Mul(tReg, gReg, tReg, fullMask);
                    Add(accGReg, accGReg, gReg, fullMask);
                    Add(accPReg, accPReg, tReg, fullMask);
                }
                DataCopy<float, LoadDist::DIST_NORM>(accReg, accGAddr + offset);
                Add(accReg, accReg, accGReg, fullMask);
                DataCopy<float, StoreDist::DIST_NORM>(accGAddr + offset, accReg, fullMask);
                DataCopy<float, LoadDist::DIST_NORM>(accReg, accPAddr + offset);
                Add(accReg, accReg, accPReg, fullMask);
                DataCopy<float, StoreDist::DIST_NORM>(accPAddr + offset, accReg, fullMask);
            }
            for (uint16_t i = lp.fullLoops; i < lp.totalLoops; i++) { // 尾块 0 或 1 次
                uint32_t tail = lp.tailCount;
                MaskReg tailMask = UpdateMask<float>(tail);
                uint32_t offset = i * VL_FP32;
                DataCopy<float, LoadDist::DIST_NORM>(mReg, expMeanAddr + offset);
                DataCopy<float, LoadDist::DIST_NORM>(sReg, expRstdAddr + offset);
                Duplicate(accGReg, 0.0f);
                Duplicate(accPReg, 0.0f);
                for (uint16_t row = 0; row < nrU; row++) {
                    LoadToFp32(gAddr + row * pitchElems, gReg, tailMask, offset);
                    LoadToFp32(xAddr + row * pitchElems, xReg, tailMask, offset);
                    Sub(tReg, xReg, mReg, tailMask);
                    Mul(tReg, tReg, sReg, tailMask);
                    Mul(tReg, gReg, tReg, tailMask);
                    Add(accGReg, accGReg, gReg, tailMask);
                    Add(accPReg, accPReg, tReg, tailMask);
                }
                DataCopy<float, LoadDist::DIST_NORM>(accReg, accGAddr + offset);
                Add(accReg, accReg, accGReg, tailMask);
                DataCopy<float, StoreDist::DIST_NORM>(accGAddr + offset, accReg, tailMask);
                DataCopy<float, LoadDist::DIST_NORM>(accReg, accPAddr + offset);
                Add(accReg, accReg, accPReg, tailMask);
                DataCopy<float, StoreDist::DIST_NORM>(accPAddr + offset, accReg, tailMask);
            }
        }
    }

    // (chunk,rSlice) 末统一归约：accRowG/accRowP [len, rEff] → dstG/dstP [len]（单发 adv_api 调用），
    // 结果累加进 accOff/accScale（跨 rSlice 累加）；rEff==1 时 accRow 即结果直接累加
    __aicore__ inline void ReduceSlice(int64_t len, int64_t rEff)
    {
        __ubuf__ float* srcGAddr = (__ubuf__ float*)accRowG_.Get<float>().GetPhyAddr();
        __ubuf__ float* srcPAddr = (__ubuf__ float*)accRowP_.Get<float>().GetPhyAddr();
        if (rEff != 1) {
            LocalTensor<float> accRowGTensor = accRowG_.Get<float>();
            LocalTensor<float> accRowPTensor = accRowP_.Get<float>();
            LocalTensor<float> dstGTensor = dstG_.Get<float>();
            LocalTensor<float> dstPTensor = dstP_.Get<float>();
            LocalTensor<uint8_t> reduceTmpTensor = reduceTmp_.Get<uint8_t>();
            uint32_t srcShape[2] = {static_cast<uint32_t>(len), static_cast<uint32_t>(rEff)};
            AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(dstGTensor, accRowGTensor, reduceTmpTensor,
                                                                          srcShape, false);
            AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, true>(dstPTensor, accRowPTensor, reduceTmpTensor,
                                                                          srcShape, false);
            // adv_api 归约内部使用 LoadUnAlign/StoreUnAlign 地址寄存器序列，与后续 VF/MTE
            // 混排会污染地址寄存器（实测致后续 DMA 以错误地址访存，error 95）；PipeBarrier 排干 V pipe
            PipeBarrier<PIPE_V>();
            srcGAddr = (__ubuf__ float*)dstG_.Get<float>().GetPhyAddr();
            srcPAddr = (__ubuf__ float*)dstP_.Get<float>().GetPhyAddr();
        }
        AccumulateFrom(srcGAddr, (__ubuf__ float*)accOff_.Get<float>().GetPhyAddr(), len);
        AccumulateFrom(srcPAddr, (__ubuf__ float*)accScale_.Get<float>().GetPhyAddr(), len);
    }

    // acc += src（[cnt] 向量累加）
    __aicore__ inline void AccumulateFrom(__ubuf__ float* srcAddr, __ubuf__ float* accAddr, int64_t cnt)
    {
        VfLoops lp = MakeVfLoops(cnt);
        __VEC_SCOPE__
        {
            RegTensor<float> accReg, srcReg;
            MaskReg fullMask = CreateMask<float, MaskPattern::ALL>();
            for (uint16_t i = 0; i < lp.fullLoops; i++) {
                uint32_t offset = i * VL_FP32;
                DataCopy<float, LoadDist::DIST_NORM>(accReg, accAddr + offset);
                DataCopy<float, LoadDist::DIST_NORM>(srcReg, srcAddr + offset);
                Add(accReg, accReg, srcReg, fullMask);
                DataCopy<float, StoreDist::DIST_NORM>(accAddr + offset, accReg, fullMask);
            }
            for (uint16_t i = lp.fullLoops; i < lp.totalLoops; i++) {
                uint32_t tail = lp.tailCount;
                MaskReg tailMask = UpdateMask<float>(tail);
                uint32_t offset = i * VL_FP32;
                DataCopy<float, LoadDist::DIST_NORM>(accReg, accAddr + offset);
                DataCopy<float, LoadDist::DIST_NORM>(srcReg, srcAddr + offset);
                Add(accReg, accReg, srcReg, tailMask);
                DataCopy<float, StoreDist::DIST_NORM>(accAddr + offset, accReg, tailMask);
            }
        }
    }

    // 结果写出：accOff/accScale 经 VECOUT 队列（EnQue/DeQue 完成 V→MTE3 同步）写输出 GM
    __aicore__ inline void WritePartials(int64_t cPos, int64_t len)
    {
        DataCopyExtParams cpOut{1, static_cast<uint32_t>(len * sizeof(float)), 0, 0, 0};
        LocalTensor<float> outUb = outQue_.AllocTensor<float>();
        CopyAccToOut((__ubuf__ float*)accOff_.Get<float>().GetPhyAddr(), (__ubuf__ float*)outUb.GetPhyAddr(), len);
        outQue_.EnQue(outUb);
        outUb = outQue_.DeQue<float>();
        DataCopyPad(diffOffsetGm_[cPos], outUb, cpOut);
        outQue_.FreeTensor(outUb);

        outUb = outQue_.AllocTensor<float>();
        CopyAccToOut((__ubuf__ float*)accScale_.Get<float>().GetPhyAddr(), (__ubuf__ float*)outUb.GetPhyAddr(), len);
        outQue_.EnQue(outUb);
        outUb = outQue_.DeQue<float>();
        DataCopyPad(diffScaleGm_[cPos], outUb, cpOut);
        outQue_.FreeTensor(outUb);
    }

    // acc → outQue 缓冲的向量拷贝（写出一趟）
    __aicore__ inline void CopyAccToOut(__ubuf__ float* accAddr, __ubuf__ float* outAddr, int64_t cnt)
    {
        VfLoops lp = MakeVfLoops(cnt);
        __VEC_SCOPE__
        {
            RegTensor<float> accReg;
            MaskReg fullMask = CreateMask<float, MaskPattern::ALL>();
            for (uint16_t i = 0; i < lp.fullLoops; i++) {
                uint32_t offset = i * VL_FP32;
                DataCopy<float, LoadDist::DIST_NORM>(accReg, accAddr + offset);
                DataCopy<float, StoreDist::DIST_NORM>(outAddr + offset, accReg, fullMask);
            }
            for (uint16_t i = lp.fullLoops; i < lp.totalLoops; i++) {
                uint32_t tail = lp.tailCount;
                MaskReg tailMask = UpdateMask<float>(tail);
                uint32_t offset = i * VL_FP32;
                DataCopy<float, LoadDist::DIST_NORM>(accReg, accAddr + offset);
                DataCopy<float, StoreDist::DIST_NORM>(outAddr + offset, accReg, tailMask);
            }
        }
    }

private:
    static constexpr int64_t RESIDUAL_INDEX = 8; // 32-byte aligned, inside the padded C=1 slot
    static constexpr uint32_t DOUBLE_BUFFER = 2;
    static constexpr int64_t REDUCE_TMP_BYTES = 1024; // ReduceSum sharedTmpBuffer（reuse-source 路径实际不用）

    TPipe* pipe_ = nullptr;
    const BNTrainingUpdateGradTilingData* tl_ = nullptr;
    int64_t cStart_ = 0;
    int64_t cRangeLen_ = 0;
    int64_t qCap_ = 0;
    int64_t pitchElems_ = 0;
    int64_t cLenPad_ = 0; // 归约 dst 行距上限（align8(cLenCap)，32B 对齐）
    float rstdLow_ = 0.0f;
    float epsilon_ = 0.0f;

    GlobalTensor<T> gradsGm_;
    GlobalTensor<T> xGm_;
    GlobalTensor<float> meanGm_;
    GlobalTensor<float> varGm_;
    GlobalTensor<float> diffScaleGm_;
    GlobalTensor<float> diffOffsetGm_;

    TQue<QuePosition::VECIN, 2> gQue_;
    TQue<QuePosition::VECIN, 2> xQue_;
    TQue<QuePosition::VECIN, 1> statMeanQue_;
    TQue<QuePosition::VECIN, 1> statVarQue_;
    TQue<QuePosition::VECOUT, 1> outQue_;
    TQue<QuePosition::VECIN, 1> p2Que_; // 确定性 rowSplit Phase2 gather 暂存 [核数, VL] fp32
    GlobalTensor<float> wsGm_;          // 确定性 rowSplit 用户区 ws（GetUserWorkspace 绑定）
    TBuf<> accRowG_;                    // grads 二维累加器 [cLen*sliceR]（跨 n 行按位累加）
    TBuf<> accRowP_;                    // grads*xnorm 二维累加器
    TBuf<> dstG_;                       // ReduceSum AR 结果 [cLen]
    TBuf<> dstP_;
    TBuf<> accOff_;     // 核内 diff_offset 部分和 [cLen]（跨 rSlice 累加）
    TBuf<> accScale_;   // 核内 diff_scale 部分和 [cLen]
    TBuf<> rstdBuf_;    // 每 chunk 的 rstd [cLen]
    TBuf<> expMean_;    // 展开系数 [cLen*sliceR](快路:尾部 64 槽 G)
    TBuf<> expRstd_;    // (快路:尾部 64 槽 P)
    TBuf<> kahanOff_;   // Kahan 补偿位(diff_offset 累加,快路)
    TBuf<> kahanScale_; // Kahan 补偿位(diff_scale 累加,快路)
    TBuf<> reduceTmp_;
};

} // namespace BNTrainingUpdateGradOps

#endif // BN_TRAINING_UPDATE_GRAD_H
