/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
/**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */
#ifndef __MULTILABEL_MARGIN_LOSS_H__
#define __MULTILABEL_MARGIN_LOSS_H__

#include "kernel_operator.h"
#include "multilabel_margin_loss_row_base.h"
#include "../multilabel_margin_loss_tiling_key.h"

namespace NsMultilabelMarginLoss {
// T = x/self dtype(float/half/bf16);IsTgtT = is_target 输出 dtype。
// is_target 是 target 派生的 0/1 掩码,与 self 无关:GE 图路径为 int32(对齐 A2),
// aclnn 路径为 float(免 int32->float Cast)。int32/float 同 4 字节,buffer 尺寸不变。
template <typename T, typename IsTgtT>
class KernelMultilabelMarginLoss : public MultilabelMarginLossRowBase<T, IsTgtT> {
public:
    __aicore__ inline KernelMultilabelMarginLoss() {}

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR target, GM_ADDR y, GM_ADDR isTarget, GM_ADDR workspace,
                                const MultilabelMarginLossArch35TilingData* tilingData)
    {
        this->N = tilingData->N;
        this->C = tilingData->C;
        this->basePerCore = tilingData->basePerCore;
        this->pivot = tilingData->pivot;
        this->usedCoreNum = tilingData->usedCoreNum;
        this->reduction = tilingData->reduction;
        this->ubFactor = tilingData->ubFactor;
        this->wsCoreStride = tilingData->wsCoreStride;
        this->partialUbElems = tilingData->partialUbElems;
        this->cFactor = (tilingData->cFactor == 0u) ? this->C : tilingData->cFactor;
        this->splitC = (this->cFactor < this->C);
        this->rowElems = this->splitC ? this->cFactor : this->C;
        this->programId = static_cast<uint32_t>(GetBlockIdx());

        this->myRows = this->basePerCore + (this->programId < this->pivot ? 1u : 0u);
        this->myStartRow = this->programId * this->basePerCore +
                           (this->programId < this->pivot ? this->programId : this->pivot);

        this->InitGlobalBuffers(input, target, y, isTarget, workspace);
        this->InitLocalBuffers();
    }

    // Multi-core-safe output write.
    // Each y element (per-row for reduction=none, the single scalar for mean/sum) is produced by
    // exactly one core. Writing it directly with a sub-32B DataCopyPad races across cores, because
    // several cores land in the same 32B GM block and the block-granular RMW clobbers neighbours.
    // Fix: 先落 FLOAT 工作区,再由核 0 统一 cast 成 T 连续写出(单写者 -> 无竞争,
    // 且 float 暂存规避 fp16/bf16 不支持原子加)。工作区两种布局:
    //   reduction=none —— 每行一个槽,由所属核原子加写入(add-to-zero == value,RMW 无竞争);
    //   mean/sum       —— 每核一个 32B 独占槽(this->wsCoreStride),不用原子加,
    //                     核 0 按固定的 blockIdx 顺序 Kahan 合并 -> 结果可复现且更准。
    __aicore__ inline void Process()
    {
        uint32_t wsElems = (this->reduction == RED_NONE) ? this->N : (this->usedCoreNum * this->wsCoreStride);

        if (this->programId == 0u && wsElems > 0u) {
            InitGlobalMemory(this->workspaceGm, wsElems, 0.0f);
        }
        SyncAll();

        if (this->reduction == RED_NONE) {
            StageRowLosses();
        } else {
            StageCorePartial();
        }

        SyncAll();

        if (this->programId == 0u) {
            if (this->reduction == RED_NONE) {
                FinalizeOutput(wsElems);
            } else {
                FinalizeReduced();
            }
        }
    }

private:
    // reduction=none: 逐行算 loss 暂存后原子加写进每行独占的 float 工作区槽。
    __aicore__ inline void StageRowLosses()
    {
        if (this->myRows > 0u) {
            // 按 this->ubFactor 分块暂存+写出:UB 占用定长,不随本核行数增长。
            LocalTensor<float> lossVec = this->rowLossBuf.template Get<float>();
            for (uint32_t base = 0; base < this->myRows; base += this->ubFactor) {
                uint32_t cur = this->myRows - base;
                if (cur > this->ubFactor) {
                    cur = this->ubFactor;
                }
                for (uint32_t r = 0; r < cur; r++) {
                    lossVec.SetValue(r, this->ProcessRow(this->myStartRow + base + r));
                }
                PipeBarrier<HardEvent::S_MTE3>();
                SetAtomicAdd<float>();
                DataCopyExtParams cpWs{1, cur * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
                DataCopyPad(this->workspaceGm[this->myStartRow + base], lossVec, cpWs);
                SetAtomicNone();
                PipeBarrier<HardEvent::MTE3_S>(); // 下一块 SetValue 复写前,等本块搬出完成
            }
        }
    }

    // reduction=sum/mean: 本核各行 Kahan 累加成一个 partial, 写进本核独占的 32B 槽。
    __aicore__ inline void StageCorePartial()
    {
        // Kahan 补偿累加:裸的 coreSum += p 每加一次都把加数末位舍掉,误差随行数线性累积。
        //     y = p - comp             先补上轮丢的
        //     t = coreSum + y          这一步会丢零头
        //     comp = (t - coreSum) - y 实际加进去的 减 本来要加的 = 这次丢的零头
        float coreSum = 0.0f;
        float comp = 0.0f;
        for (uint32_t r = 0; r < this->myRows; r++) {
            float y = this->ProcessRowSum(this->myStartRow + r) - comp;
            float t = coreSum + y;
            comp = (t - coreSum) - y;
            if (__isinf(t) || __isnan(t)) { // t 非有限: 补偿项 inf-inf=NaN 会污染后续行, 清零
                comp = 0.0f;
            }
            coreSum = t;
        }
        // 只写本核原始 partial,不再各自先除 N:sum(coreSum_c / N) 每核都舍一次,
        // 改由 FinalizeReduced 合并后统一除一次。独占槽位 -> 不用原子加。
        // 写整槽(coreSum + 其余补 0): 跨核 GM 写本就是 32B 粒度, 补零后 FinalizeReduced 能
        // 一次连续读入直接矢量累加(补零车道加 0 不改变结果), 不必逐核 GetValue。
        LocalTensor<float> one = this->rowLossBuf.template Get<float>();
        for (uint32_t k = 0; k < this->wsCoreStride; k++) {
            one.SetValue(k, 0.0f);
        }
        one.SetValue(0, coreSum);
        PipeBarrier<HardEvent::S_MTE3>();
        DataCopyExtParams cpWs{1, static_cast<uint32_t>(this->wsCoreStride * sizeof(float)), 0, 0, 0};
        DataCopyPad(this->workspaceGm[this->programId * this->wsCoreStride], one, cpWs);
    }
    __aicore__ inline void FinalizeReduced()
    {
        uint32_t wsElems = this->usedCoreNum * this->wsCoreStride;
        for (uint32_t off = 0; off < wsElems; off += this->wsCoreStride) {
            DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                this->workspaceGm[off]);
        }
        LocalTensor<float> partials = this->partialsInQueue.template AllocTensor<float>();
        DataCopyExtParams cpIn{1, wsElems * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padIn{false, 0, 0, 0};
        DataCopyPad(partials, this->workspaceGm[0], cpIn, padIn);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

        const uint16_t mmlRounds = static_cast<uint16_t>(this->partialUbElems / MML_VL_FP32);
        // 只清**补零区**: 矢量合并整轮读满 MML_VL_FP32 车道、收尾标量循环也按 stride 读满
        // MML_VL_FP32/this->wsCoreStride 条车道, 而 DataCopyPad 只搬入 wsElems 个元素; 余下车道若不清零
        // 读到的是 UB 残留(可能含 ±inf/NaN), 会被当作 partial 累加 —— 核数越少无效车道越多。
        // 放在搬入之后只清尾巴, 省一次 V->MTE2 同步和大部分清零量; Duplicate 与下面的矢量循环
        // 同为 V 流水, 按序发射无需额外同步。
        if (this->partialUbElems > wsElems) {
            Duplicate(partials[wsElems], 0.0f, static_cast<int32_t>(this->partialUbElems - wsElems));
        }

        // ── 跨核合并: **矢量(regbase)形态的 Kahan 补偿累加** ───────────────────────────
        // 每核占一个 this->wsCoreStride 槽(首元素有效、其余为 0), 一次寄存器载入(64 车道)覆盖
        // 64/this->wsCoreStride 个核; 车道并行做 Kahan, 补偿项覆盖各车道的顺序步数。补零车道不产生补偿量。
        // 原写法是核 0 上串行 this->usedCoreNum(最多 64)次的标量循环。
        // inf/nan: 补偿量 c = (mmlNext - sum) - mmlLaneVal 在 mmlNext 为 ±inf 时算出 NaN, 会污染后续每一轮, 把本该 inf
        // 的和变成 nan。自比 Compare<EQ>(c, c) + Select 清掉 NaN 车道的补偿量, 只清补偿、不动 sum,
        // 真 nan 仍如实传播。(同款写法见 norm/instance_norm_grad。)
        __local_mem__ float* mmlUbAddr = (__local_mem__ float*)partials.GetPhyAddr();
        {
            AscendC::Reg::RegTensor<float> mmlSum;
            AscendC::Reg::RegTensor<float> mmlComp;
            AscendC::Reg::RegTensor<float> mmlZero;
            AscendC::Reg::RegTensor<float> mmlPartialReg;
            AscendC::Reg::RegTensor<float> mmlErr;
            AscendC::Reg::RegTensor<float> mmlAcc;
            AscendC::Reg::RegTensor<float> mmlDelta;
            AscendC::Reg::MaskReg mmlMask;
            AscendC::Reg::MaskReg mmlFiniteMask;
            uint32_t mmlLaneCnt = static_cast<uint32_t>(mmlRounds) * MML_VL_FP32;
            __VEC_SCOPE__
            {
                mmlMask = AscendC::Reg::UpdateMask<float>(mmlLaneCnt);
                AscendC::Reg::Duplicate(mmlSum, 0.0f, mmlMask);
                AscendC::Reg::Duplicate(mmlComp, 0.0f, mmlMask);
                AscendC::Reg::Duplicate(mmlZero, 0.0f, mmlMask);
                for (uint16_t mmlR = 0; mmlR < mmlRounds; ++mmlR) {
                    AscendC::Reg::DataCopy(mmlPartialReg, mmlUbAddr + mmlR * MML_VL_FP32);
                    AscendC::Reg::Sub(mmlErr, mmlPartialReg, mmlComp, mmlMask);
                    AscendC::Reg::Add(mmlAcc, mmlSum, mmlErr, mmlMask);
                    AscendC::Reg::Sub(mmlDelta, mmlAcc, mmlSum, mmlMask);
                    AscendC::Reg::Sub(mmlComp, mmlDelta, mmlErr, mmlMask);
                    AscendC::Reg::Compare<float, AscendC::CMPMODE::EQ>(mmlFiniteMask, mmlComp, mmlComp, mmlMask);
                    AscendC::Reg::Select(mmlComp, mmlComp, mmlZero, mmlFiniteMask);
                    AscendC::Reg::Move(mmlSum, mmlAcc, mmlMask);
                }
                // 车道总账与欠账都写回: 收尾要按 (tot - comp) 再做一次标量 Kahan
                AscendC::Reg::Sub(mmlAcc, mmlSum, mmlComp, mmlMask);
                AscendC::Reg::DataCopy(mmlUbAddr, mmlAcc, mmlMask);
            }
        }
        // 收尾: 只对 MML_VL_FP32/this->wsCoreStride 条有效车道做标量 Kahan(核数 64 时是 8 步, 不是原来的 64 步)。
        // 车道内已补偿过各自的顺序累加, 这里再补偿车道之间 —— 实测(同一批 partial 配对比较)
        // 与原标量 Kahan 6 例中 5 例逐位相同; 若这里改用无补偿的树形规约, 会有 3/6 变差。
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        float mmlTotal = 0.0f;
        float mmlCarry = 0.0f;
        for (int32_t mmlLane = 0; mmlLane < MML_VL_FP32; mmlLane += static_cast<int32_t>(this->wsCoreStride)) {
            float mmlLaneVal = partials.GetValue(mmlLane) - mmlCarry;
            float mmlNext = mmlTotal + mmlLaneVal;
            mmlCarry = (mmlNext - mmlTotal) - mmlLaneVal;
            if (__isinf(mmlNext) || __isnan(mmlNext)) {
                mmlCarry = 0.0f;
            }
            mmlTotal = mmlNext;
        }
        this->partialsInQueue.FreeTensor(partials);

        WriteScalarOutput(ApplyReductionDivisor(mmlTotal));
    }

    // 对合并后的总和施加 reduction 除数。loss = Σ_all margin / C(sum) 或 /(C·N)(mean)。
    // 关键:mean 必须**一次**除完。拆成 /C 再 /N 会双重舍入 —— 实测 12x40、15x25 两例都因此
    // 偏正确舍入 1 格,而竞品(同为 fp32)命中;换成单次除 C·N 后离线复算即落在正确舍入上。
    // 同理不用乘倒数(1/C、1/N 在 fp32 多半存不下,又是先舍一次乘时再舍一次)。
    // C·N > 2^24 时 fp32 存不下该整数除数,除数本身先失真,此时退回两次除法反而更准。
    // C==0 或 N==0(空 tensor)不做除法,保持原语义(输出 0)。
    __aicore__ inline float ApplyReductionDivisor(float total)
    {
        if (this->C == 0u) {
            return total;
        }
        if (this->reduction != RED_MEAN) { // RED_SUM:只除 C
            return total / static_cast<float>(static_cast<int32_t>(this->C));
        }
        if (this->N == 0u) {
            // 空 tensor 的 mean 按标准是 0/0=nan(与 torch .mean() 一致), 不是 0。
            // 走 GE 图直下 kernel 时由这里保证; aclnn 通路在 L2 层空分支已按同一标准填值。
            return total / 0.0f;
        }
        uint64_t denom = static_cast<uint64_t>(this->C) * static_cast<uint64_t>(this->N);
        constexpr uint64_t FP32_EXACT_INT_MAX = 1ULL << 24;
        if (denom <= FP32_EXACT_INT_MAX) {
            return total / static_cast<float>(static_cast<int32_t>(denom));
        }
        return total / static_cast<float>(static_cast<int32_t>(this->C)) /
               static_cast<float>(static_cast<int32_t>(this->N));
    }

    // float 暂存 -> cast 成 T -> 写单元素 y(mean/sum 的输出恒为标量)。
    __aicore__ inline void WriteScalarOutput(float v)
    {
        LocalTensor<float> acc = this->gatherBuf.template Get<float>();
        acc.SetValue(0, v);
        PipeBarrier<HardEvent::S_V>();
        LocalTensor<T> outVec = this->gatherOutBuf.template Get<T>();
        if constexpr (std::is_same<T, float>::value) {
            Adds(outVec, acc, 0.0f, 1);
        } else if constexpr (std::is_same<T, bfloat16_t>::value) {
            Cast(outVec, acc, RoundMode::CAST_RINT, 1);
        } else {
            Cast(outVec, acc, RoundMode::CAST_NONE, 1);
        }
        PipeBarrier<HardEvent::V_MTE3>();
        DataCopyExtParams cpOut{1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        DataCopyPad(this->outputGm[0], outVec, cpOut);
    }

    __aicore__ inline void FinalizeOutput(uint32_t wsElems)
    {
        if (wsElems == 0u) {
            return;
        }
        LocalTensor<float> acc = this->gatherBuf.template Get<float>();
        LocalTensor<T> outVec = this->gatherOutBuf.template Get<T>();
        // 按 this->ubFactor 分块回读+写出:UB 占用定长,不随 N 增长。
        for (uint32_t base = 0; base < wsElems; base += this->ubFactor) {
            uint32_t cur = wsElems - base;
            if (cur > this->ubFactor) {
                cur = this->ubFactor;
            }
            // Invalidate core 0's cached view of the workspace it zero-initialised, so the read below
            // sees the values other cores atomic-added (stride 8 floats = 32B covers any cache line).
            for (uint32_t off = base; off < base + cur; off += 8u) {
                DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(
                    this->workspaceGm[off]);
            }
            DataCopyExtParams cpIn{1, cur * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> padIn{false, 0, 0, 0};
            DataCopyPad(acc, this->workspaceGm[base], cpIn, padIn);
            PipeBarrier<HardEvent::MTE2_V>();

            if constexpr (std::is_same<T, float>::value) {
                Adds(outVec, acc, 0.0f, cur);
            } else if constexpr (std::is_same<T, bfloat16_t>::value) {
                Cast(outVec, acc, RoundMode::CAST_RINT, cur);
            } else {
                Cast(outVec, acc, RoundMode::CAST_NONE, cur);
            }
            PipeBarrier<HardEvent::V_MTE3>();

            DataCopyExtParams cpOut{1, cur * static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
            DataCopyPad(this->outputGm[base], outVec, cpOut);
            PipeBarrier<HardEvent::MTE3_V>(); // 下一块 Cast 复写 outVec 前,等本块搬出完成
            PipeBarrier<HardEvent::V_MTE2>(); // 下一块搬入 acc 前,等本块 Cast 读完
        }
    }
};

} // namespace NsMultilabelMarginLoss
#endif
