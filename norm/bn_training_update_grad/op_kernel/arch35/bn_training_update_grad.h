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
using AscendC::MicroAPI::MaskPattern;
using AscendC::MicroAPI::MaskReg;
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
        (void)workspace; // 无 workspace（channel 归属唯一，零核间通信）
        pipe_ = pipe;
        tl_ = tilingData;
        // blockIdx 即 channel 块号：每 channel 的完整归约由唯一归属核完成（零核间通信）
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
        gradsGm_.SetGlobalBuffer((__gm__ T*)grads, gmLen);
        xGm_.SetGlobalBuffer((__gm__ T*)x, gmLen);
        meanGm_.SetGlobalBuffer((__gm__ float*)batchMean, tl_->numC);
        varGm_.SetGlobalBuffer((__gm__ float*)batchVar, tl_->numC);
        diffScaleGm_.SetGlobalBuffer((__gm__ float*)diffScale, tl_->numC);
        diffOffsetGm_.SetGlobalBuffer((__gm__ float*)diffOffset, tl_->numC);

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
    __aicore__ inline void KahanAdd(__ubuf__ float* accAddr, __ubuf__ float* compAddr, __ubuf__ float* srcAddr)
    {
        __VEC_SCOPE__
        {
            RegTensor<float> srcReg, sumReg, cReg, yReg, tReg;
            MaskReg fullMask = CreateMask<float, MaskPattern::ALL>();
            DataCopy<float, LoadDist::DIST_NORM>(srcReg, srcAddr);
            DataCopy<float, LoadDist::DIST_NORM>(sumReg, accAddr);
            DataCopy<float, LoadDist::DIST_NORM>(cReg, compAddr);
            Sub(yReg, srcReg, cReg, fullMask); // y = src - c
            Add(tReg, sumReg, yReg, fullMask); // t = sum + y
            Sub(cReg, tReg, sumReg, fullMask); // (t - sum)
            Sub(cReg, cReg, yReg, fullMask);   // c = (t - sum) - y
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
    static constexpr uint32_t DOUBLE_BUFFER = 2;
    static constexpr int64_t REDUCE_TMP_BYTES = 1024; // ReduceSum sharedTmpBuffer（reuse-source 路径实际不用）

    TPipe* pipe_ = nullptr;
    const BNTrainingUpdateGradTilingData* tl_ = nullptr;
    int64_t cStart_ = 0;
    int64_t cRangeLen_ = 0;
    int64_t qCap_ = 0;
    int64_t pitchElems_ = 0;
    int64_t cLenPad_ = 0; // 归约 dst 行距上限（align8(cLenCap)，32B 对齐）
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
    TBuf<> accRowG_; // grads 二维累加器 [cLen*sliceR]（跨 n 行按位累加）
    TBuf<> accRowP_; // grads*xnorm 二维累加器
    TBuf<> dstG_;    // ReduceSum AR 结果 [cLen]
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
