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
#include "kernel_tiling/kernel_tiling.h"
#include "multilabel_margin_loss_tiling_data_arch35.h"
#include "../multilabel_margin_loss_tiling_key.h"
#include <type_traits>

namespace NsMultilabelMarginLoss {
using namespace AscendC;

constexpr int32_t RED_NONE = 0;
constexpr int32_t RED_MEAN = 1;
constexpr int32_t RED_SUM = 2;

// T = x/self dtype(float/half/bf16);IsTgtT = is_target 输出 dtype。
// is_target 是 target 派生的 0/1 掩码,与 self 无关:GE 图路径为 int32(对齐 A2),
// aclnn 路径为 float(免 int32->float Cast)。int32/float 同 4 字节,buffer 尺寸不变。
template <typename T, typename IsTgtT>
class KernelMultilabelMarginLoss {
private:
    TPipe pipe;

    TQue<TPosition::VECIN, 1> inputQueue;
    TQue<TPosition::VECIN, 1> targetQueue;
    TQue<TPosition::VECIN, 1> partialsInQueue;
    TQue<TPosition::VECOUT, 1> isTargetOutQueue;

    TBuf<TPosition::VECCALC> xRowBuf;
    TBuf<TPosition::VECCALC> isPosBuf;
    TBuf<TPosition::VECCALC> reduceBuf; // accVec: per-row accumulator over target labels (float)
    TBuf<TPosition::VECCALC> workBuf;   // ReduceSum work buffer
    TBuf<TPosition::VECCALC> outCastBuf;
    TBuf<TPosition::VECCALC> rowLossBuf;   // this core's row losses (float), staged before atomic add
    TBuf<TPosition::VECCALC> gatherBuf;    // core 0: full float workspace read-back (<= N floats)
    TBuf<TPosition::VECCALC> gatherOutBuf; // core 0: cast-to-T output vector (<= N elems)
    TBuf<TPosition::VECCALC> isTgtFBuf;    // C 分块路径: is_target 段回读并转 float, 作为非目标位掩码

    GlobalTensor<T> inputGm;
    GlobalTensor<int32_t> targetGm;
    GlobalTensor<T> outputGm;
    GlobalTensor<IsTgtT> isTargetGm;
    GlobalTensor<float> workspaceGm;

    uint32_t N;
    uint32_t C;
    uint32_t basePerCore;
    uint32_t pivot;
    uint32_t usedCoreNum;
    int32_t reduction;
    uint32_t myRows;
    uint32_t myStartRow;
    uint32_t programId;
    uint32_t ubFactor;     // host 侧实算下发:每轮 UB 处理的元素数(已对齐)
    uint32_t wsCoreStride; // host 侧实算下发:mean/sum 每核独占槽位跨步(float 个数)
    uint32_t cFactor;      // host 侧实算下发:C 方向每段元素数; == C 表示整行装得下, 走全行路径
    bool splitC;           // cFactor < C
    uint32_t rowElems;     // 行缓冲按它分配: splitC ? cFactor : C

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
        this->cFactor = (tilingData->cFactor == 0u) ? this->C : tilingData->cFactor;
        this->splitC = (this->cFactor < this->C);
        this->rowElems = this->splitC ? this->cFactor : this->C;
        this->programId = static_cast<uint32_t>(GetBlockIdx());

        this->myRows = basePerCore + (this->programId < pivot ? 1u : 0u);
        this->myStartRow = this->programId * basePerCore + (this->programId < pivot ? this->programId : pivot);

        InitGlobalBuffers(input, target, y, isTarget, workspace);
        InitLocalBuffers();
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
            InitGlobalMemory(workspaceGm, wsElems, 0.0f);
        }
        SyncAll();

        if (this->reduction == RED_NONE) {
            if (this->myRows > 0u) {
                // 按 this->ubFactor 分块暂存+写出:UB 占用定长,不随本核行数增长。
                LocalTensor<float> lossVec = rowLossBuf.Get<float>();
                for (uint32_t base = 0; base < this->myRows; base += this->ubFactor) {
                    uint32_t cur = this->myRows - base;
                    if (cur > this->ubFactor) {
                        cur = this->ubFactor;
                    }
                    for (uint32_t r = 0; r < cur; r++) {
                        lossVec.SetValue(r, ProcessRow(this->myStartRow + base + r));
                    }
                    PipeBarrier<HardEvent::S_MTE3>();
                    SetAtomicAdd<float>();
                    DataCopyExtParams cpWs{1, cur * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
                    DataCopyPad(workspaceGm[this->myStartRow + base], lossVec, cpWs);
                    SetAtomicNone();
                    PipeBarrier<HardEvent::MTE3_S>(); // 下一块 SetValue 复写前,等本块搬出完成
                }
            }
        } else {
            // Kahan 补偿累加:裸的 coreSum += p 每加一次都把加数末位舍掉,误差随行数线性累积。
            //     y = p - comp             先补上轮丢的
            //     t = coreSum + y          这一步会丢零头
            //     comp = (t - coreSum) - y 实际加进去的 减 本来要加的 = 这次丢的零头
            float coreSum = 0.0f;
            float comp = 0.0f;
            for (uint32_t r = 0; r < this->myRows; r++) {
                float y = ProcessRowSum(this->myStartRow + r) - comp;
                float t = coreSum + y;
                comp = (t - coreSum) - y;
                coreSum = t;
            }
            // 只写本核原始 partial,不再各自先除 N:sum(coreSum_c / N) 每核都舍一次,
            // 改由 FinalizeReduced 合并后统一除一次。独占槽位 -> 不用原子加。
            LocalTensor<float> one = rowLossBuf.Get<float>();
            one.SetValue(0, coreSum);
            PipeBarrier<HardEvent::S_MTE3>();
            DataCopyExtParams cpWs{1, static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
            DataCopyPad(workspaceGm[this->programId * this->wsCoreStride], one, cpWs);
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
    // Per-row byte size: max of 32B-aligned pad length and 16-element-aligned cast length.
    template <typename U>
    __aicore__ inline uint32_t RowBytes()
    {
        uint32_t padBytes = ((rowElems * sizeof(U) + 31u) / 32u) * 32u;
        uint32_t castBytes = ((rowElems + 15u) / 16u) * 16u * sizeof(U);
        return (padBytes > castBytes) ? padBytes : castBytes;
    }

    __aicore__ inline void InitGlobalBuffers(GM_ADDR input, GM_ADDR target, GM_ADDR y, GM_ADDR isTarget,
                                             GM_ADDR workspace)
    {
        uint64_t outputElems = (reduction == RED_NONE) ? static_cast<uint64_t>(N) : 1ULL;
        uint64_t nc = static_cast<uint64_t>(N) * static_cast<uint64_t>(C);
        inputGm.SetGlobalBuffer((__gm__ T*)input, nc);
        targetGm.SetGlobalBuffer((__gm__ int32_t*)target, nc);
        outputGm.SetGlobalBuffer((__gm__ T*)y, outputElems);
        isTargetGm.SetGlobalBuffer((__gm__ IsTgtT*)isTarget, nc);
        // Float workspace: N slots for reduction=none (per-row loss);
        // mean/sum 为每核一个 32B 独占槽(usedCoreNum * this->wsCoreStride)。
        uint64_t wsElems = (reduction == RED_NONE) ? static_cast<uint64_t>(N) :
                                                     (static_cast<uint64_t>(usedCoreNum) * this->wsCoreStride);
        if (wsElems == 0ULL) {
            wsElems = 1ULL;
        }
        workspaceGm.SetGlobalBuffer((__gm__ float*)workspace, wsElems);
    }

    __aicore__ inline void InitLocalBuffers()
    {
        uint32_t inputRowBytes = RowBytes<T>();
        uint32_t intRowBytes = RowBytes<int32_t>();
        uint32_t fRowBytes = RowBytes<float>();
        // VF 工作 buffer 按向量寄存器宽度(VLF=VECTOR_REG_WIDTH/4)对齐: 内层 full-VL load
        // DataCopy(reg, addr+i*VL) 末块读满 VL 个元素, buffer 须 >= ceil(C/VL)*VL 才不越界。
        // 这是向量化访问的正当内存布局(同 cross_entropy_loss), 非"补 pad 算垃圾"规避——
        // 尾块无效 lane 由 UpdateMask 屏蔽, 不参与计算, 归约只读 [0,C)。VLF 取自权威设备常量。
        constexpr uint32_t VLF = VECTOR_REG_WIDTH / sizeof(float);
        uint32_t vfRowBytes = (((this->rowElems + VLF - 1u) / VLF) * VLF) * sizeof(float);
        if (vfRowBytes < fRowBytes)
            vfRowBytes = fRowBytes;
        if (vfRowBytes < 32u)
            vfRowBytes = 32u;

        // 核0 读回全部 partial:每核一个 this->wsCoreStride 槽,故按跨步总量分配。
        uint32_t partialsBytes = ((usedCoreNum * this->wsCoreStride * sizeof(float) + 31u) / 32u) * 32u;
        if (partialsBytes < 32u)
            partialsBytes = 32u;
        uint32_t scalarBytes = 32u;

        pipe.InitBuffer(inputQueue, 1, inputRowBytes);
        pipe.InitBuffer(targetQueue, 1, intRowBytes);
        pipe.InitBuffer(partialsInQueue, 1, partialsBytes);
        pipe.InitBuffer(isTargetOutQueue, 1, RowBytes<IsTgtT>());

        pipe.InitBuffer(xRowBuf, vfRowBytes);
        pipe.InitBuffer(isPosBuf, vfRowBytes);
        pipe.InitBuffer(reduceBuf, vfRowBytes);
        pipe.InitBuffer(workBuf, fRowBytes);
        pipe.InitBuffer(outCastBuf, scalarBytes);
        // 分块路径要把 is_target 段回读成 float 当掩码; 全行路径用不到, 给最小块即可。
        pipe.InitBuffer(isTgtFBuf, this->splitC ? vfRowBytes : 32u);

        // 行损失的暂存(本核 myRows 行)与回读(核0 N 行)都按 min(需求, this->ubFactor) 分配:
        // 小 shape 仍按需精确分配(与原实现一致,不浪费 UB),大 N 被上限截断后走分块循环,
        // 使 UB 占用有硬上界。原实现直接按 N / (basePerCore+1) 分配,N 大时撑爆 UB,
        // 而 host tiling 侧没有任何 UB 校验兜底。mean/sum 复用 rowLossBuf 存单个 partial,32B 下限已覆盖。
        uint32_t rowTile = (this->myRows < this->ubFactor) ? this->myRows : this->ubFactor;
        uint32_t nTile = (this->N < this->ubFactor) ? this->N : this->ubFactor;
        uint32_t rowLossBytes = (((rowTile + 7u) / 8u) * 8u) * sizeof(float);
        if (rowLossBytes < 32u)
            rowLossBytes = 32u;
        uint32_t gatherFloatBytes = (((nTile + 7u) / 8u) * 8u) * sizeof(float);
        if (gatherFloatBytes < 32u)
            gatherFloatBytes = 32u;
        uint32_t gatherTBytes = (((nTile + 15u) / 16u) * 16u) * sizeof(T);
        if (gatherTBytes < 32u)
            gatherTBytes = 32u;
        pipe.InitBuffer(rowLossBuf, rowLossBytes);
        pipe.InitBuffer(gatherBuf, gatherFloatBytes);
        pipe.InitBuffer(gatherOutBuf, gatherTBytes);
    }

    __aicore__ inline void CastInputToFloat(LocalTensor<float>& dst, LocalTensor<T>& src, uint32_t cnt)
    {
        if constexpr (std::is_same<T, float>::value) {
            uint32_t cnt8 = ((cnt + 7u) / 8u) * 8u;
            DataCopy(dst, src, cnt8);
        } else {
            Cast(dst, src, RoundMode::CAST_NONE, cnt);
        }
    }

    __aicore__ inline void StoreScalarAsInput(LocalTensor<T>& outLocal, float value)
    {
        if constexpr (std::is_same<T, float>::value) {
            outLocal.SetValue(0, value);
        } else {
            LocalTensor<float> stage = outCastBuf.Get<float>();
            stage.SetValue(0, value);
            PipeBarrier<HardEvent::S_V>();
            if constexpr (std::is_same<T, bfloat16_t>::value) {
                Cast(outLocal, stage, RoundMode::CAST_RINT, 1);
            } else {
                Cast(outLocal, stage, RoundMode::CAST_NONE, 1);
            }
            PipeBarrier<HardEvent::V_S>();
        }
    }

    // Single-event pipe barrier (e.g. HardEvent::V_S) to order scalar/vector accesses.
    template <HardEvent evt>
    __aicore__ inline void PipeBarrier()
    {
        // 用固定事件 ID(仓上通行做法)而非每次 FetchEventID:本屏障在逐行路径上被调用,
        // N 大时每核累计数千次取用,固定 ID 免去反复申请。Set/Wait 严格成对且不嵌套,
        // 复用同一 ID 语义等价。
        SetFlag<evt>(EVENT_ID0);
        WaitFlag<evt>(EVENT_ID0);
    }

    // Copy one row of input and target from GM into local tensors (already DeQue'd).
    __aicore__ inline void CopyInRow(uint32_t row, uint32_t cnt, LocalTensor<T>& xRowIn, LocalTensor<int32_t>& tgtIn)
    {
        LocalTensor<T> inputLocal = inputQueue.AllocTensor<T>();
        LocalTensor<int32_t> targetLocal = targetQueue.AllocTensor<int32_t>();

        uint64_t rowOff = static_cast<uint64_t>(row) * static_cast<uint64_t>(cnt);

        // Explicit cast: cnt * sizeof(...) is unsigned long; cast to uint32_t avoids the
        // brace-init narrowing (-Wc++11-narrowing) on DataCopyExtParams.blockLen.
        DataCopyExtParams cpInExt{1, static_cast<uint32_t>(cnt * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padInExt{false, 0, 0, 0};
        DataCopyPad(inputLocal, inputGm[rowOff], cpInExt, padInExt);

        DataCopyExtParams cpTgtExt{1, static_cast<uint32_t>(cnt * sizeof(int32_t)), 0, 0, 0};
        DataCopyPadExtParams<int32_t> padTgtExt{false, 0, 0, 0};
        DataCopyPad(targetLocal, targetGm[rowOff], cpTgtExt, padTgtExt);

        inputQueue.EnQue(inputLocal);
        targetQueue.EnQue(targetLocal);
        xRowIn = inputQueue.DeQue<T>();
        tgtIn = targetQueue.DeQue<int32_t>();
    }

    // Build is_pos from the target row and copy out the is_target row to GM.
    __aicore__ inline void BuildMasks(uint32_t row, uint32_t cnt, LocalTensor<int32_t>& tgtIn,
                                      LocalTensor<float>& isPos)
    {
        Duplicate(isPos, 0.0f, cnt);
        LocalTensor<IsTgtT> isTargetLocal = isTargetOutQueue.AllocTensor<IsTgtT>();
        Duplicate(isTargetLocal, static_cast<IsTgtT>(0), cnt);

        // V->S: scalar GetValue/SetValue below depends on Duplicate to isPos.
        PipeBarrier<HardEvent::V_S>();

        // Walk target with -1 sentinel break (PyTorch MultiLabelMarginLoss semantics).
        // tt is the class index; guard against out-of-range labels before using it as offset.
        for (uint32_t t = 0; t < cnt; t++) {
            int32_t tt = tgtIn.GetValue(t);
            if (tt == -1)
                break;
            if (tt < 0 || static_cast<uint32_t>(tt) >= cnt)
                continue;
            isPos.SetValue(static_cast<uint32_t>(tt), 1.0f);
            isTargetLocal.SetValue(static_cast<uint32_t>(tt), static_cast<IsTgtT>(1));
        }

        // CopyOut is_target row to GM (every reduction mode).
        isTargetOutQueue.EnQue(isTargetLocal);
        LocalTensor<IsTgtT> isTargetDeq = isTargetOutQueue.DeQue<IsTgtT>();
        uint64_t rowOff = static_cast<uint64_t>(row) * static_cast<uint64_t>(cnt);
        DataCopyExtParams cpIsTgt{1, static_cast<uint32_t>(cnt * sizeof(IsTgtT)), 0, 0, 0};
        DataCopyPad(isTargetGm[rowOff], isTargetDeq, cpIsTgt);
        isTargetOutQueue.FreeTensor(isTargetDeq);
    }

    // Row loss = sum_{k in target labels} sum_{i not in target} max(0, 1 - x[k] + x[i]), matching PyTorch
    // MultiLabelMarginLoss. Outer scalar loop over valid target labels (data-dependent length, -1 sentinel
    // break, out-of-range guard); inner work fully VECTORISED over all C classes:
    //   margin[i] = relu((1 - x[k]) + x[i])   via Adds + CompareScalar(>0) + Select (strict >0 = torch z>0, nan-safe)
    //   drop target positions (i in T) with Select on the non-target mask, NOT a multiply by (1 - isPos):
    //     x[i] = +/-inf at a target position would give inf*0 = NaN and corrupt the sum; Select replaces
    //     the value (no arithmetic), so target slots become an exact 0 and non-target inf/nan propagate
    //     as torch's IEEE result.
    // Accumulate each label's masked margins into accVec, then one ReduceSum per row (sum_k sum_i == sum_i sum_k).
    // arch35 regbase(Reg VF)实现: 外层 target 标量循环(数据依赖,-1 哨兵 break + 越界守卫),
    // 内层按 C 用 RegTensor 硬件向量循环(VF, 尾块 UpdateMask)。语义与 A2 完全一致:
    //   margin[i] = relu((1 - x[k]) + x[i]) 严格 >0 select(nan-safe,对齐 torch `if(z>0)`),
    //   非目标位用 Select 屏蔽(非乘法,避免 target 位 inf*0=NaN)。逐 k 累加进 UB 的 accVec,
    //   最后对 accVec 做一次 ReduceSum(sum_k sum_i == sum_i sum_k)。
    __aicore__ inline float AccumulateRowLoss(uint32_t cnt, LocalTensor<float>& xRow, LocalTensor<int32_t>& tgtIn,
                                              LocalTensor<float>& isPos)
    {
        if (cnt == 0u) {
            return 0.0f;
        }
        using namespace AscendC::Reg;
        LocalTensor<float> accVec = reduceBuf.Get<float>();
        auto accAddr = (__ubuf__ float*)accVec.GetPhyAddr();
        auto xAddr = (__ubuf__ float*)xRow.GetPhyAddr();
        auto posAddr = (__ubuf__ float*)isPos.GetPhyAddr();

        // VF 必须包在 __VEC_SCOPE__ 内(否则后端 "Do not know how to split the result")。
        // 尾块用 UpdateMask 逐迭代处理(只算有效 lane, 不算 padding 垃圾); 工作 buffer 已按向量寄存器
        // 宽度(cAlign, host tiling 计算)对齐, 使 full-VL load 不越界——正式对齐, 非 kernel 侧补 pad 规避。
        constexpr uint16_t VL = VECTOR_REG_WIDTH / sizeof(float);
        uint16_t repeatTimes = static_cast<uint16_t>((cnt + VL - 1u) / VL);

        // accVec[0..cnt) = 0
        {
            uint32_t sreg = cnt;
            __VEC_SCOPE__
            {
                RegTensor<float> z;
                MaskReg preg;
                for (uint16_t i = 0; i < repeatTimes; i++) {
                    preg = UpdateMask<float>(sreg);
                    Duplicate(z, 0.0f, preg);
                    DataCopy(accAddr + i * VL, z, preg);
                }
            }
        }
        PipeBarrier<HardEvent::V_S>(); // V->S: 下面读 x[k] 标量依赖 Cast 产出的 xRow

        // 外层 target 标量循环(数据依赖, -1 哨兵 break + 越界守卫); 内层按 C 向量化累加进 accVec。
        for (uint32_t t = 0; t < cnt; t++) {
            int32_t tt = tgtIn.GetValue(t);
            if (tt == -1)
                break;
            if (tt < 0 || static_cast<uint32_t>(tt) >= cnt)
                continue;
            float s = 1.0f - xRow.GetValue(static_cast<uint32_t>(tt));
            uint32_t sreg = cnt;
            __VEC_SCOPE__
            {
                RegTensor<float> xr, posr, accr, tmp, zero;
                MaskReg preg, posM, tgtM;
                for (uint16_t i = 0; i < repeatTimes; i++) {
                    preg = UpdateMask<float>(sreg);
                    DataCopy(xr, xAddr + i * VL);
                    DataCopy(posr, posAddr + i * VL);
                    DataCopy(accr, accAddr + i * VL);
                    Duplicate(zero, 0.0f, preg);
                    Adds(tmp, xr, s, preg);                                   // (1 - x[k]) + x[i]
                    CompareScalar<float, CMPMODE::GT>(posM, tmp, 0.0f, preg); // tmp > 0 (严格)
                    Select(tmp, tmp, zero, posM); // relu 严格 >0 (nan/负 -> 0, +inf 保留)
                    CompareScalar<float, CMPMODE::GT>(tgtM, posr, 0.5f, preg); // isPos > 0.5 (目标位)
                    Select(tmp, zero, tmp, tgtM);                              // 目标位 -> 0, 非目标位保留 tmp
                    // 勿在此加 Kahan 补偿:补偿量代数恒为 0,VF 后端 -O2 重结合会整段消掉(实测无效)。
                    Add(accr, accr, tmp, preg);
                    DataCopy(accAddr + i * VL, accr, preg);
                }
            }
        }
        // 行损失 = sum_i accVec[i] (只读有效 [0,cnt))。
        float acc = LocalReduceSum(accVec, cnt);
        return acc;
    }

    // 硬件 ReduceSum(树规约)求 src[0,cnt) 之和。不用逐元素 GetValue 标量单链:单链误差随 C
    // 线性累积(~eps*C),树规约 ~eps*log2(C),且省掉 C 次标量读 UB。
    // 独立成函数以避开调用点 using namespace Reg 引入的同名重载。
    __aicore__ inline float LocalReduceSum(LocalTensor<float>& src, uint32_t cnt)
    {
        LocalTensor<float> work = workBuf.Get<float>();
        ReduceSum(work, src, work, static_cast<int32_t>(cnt));
        SetFlag<HardEvent::V_S>(EVENT_ID0);
        WaitFlag<HardEvent::V_S>(EVENT_ID0);
        return work.GetValue(0);
    }

    // ================= C 分块路径(splitC) =================
    // 整行装不下 UB 时启用。语义与全行路径逐字一致, 只是把"一行 C"拆成若干 cFactor 段:
    //   Phase 1: 产出 is_target(先整行清零, 再按有效标签散点置 1) —— 它本就是必产出的输出,
    //            这里同时充当 Phase 2 的"非目标位"掩码源, 免去在 UB 里常驻一份 C 长的 isPos;
    //   Phase 2: 标签分批(每批至多 cFactor 个, 连 x[k] 一并取回), 每批对全部 x 段各做一次向量累加。
    // 计算量与全行路径同为 O(P*C)(P = 有效标签数), 代价是 x/掩码段被重载 ceil(P/cFactor) 轮。

    // 对一段 x(长度 cnt)累加一批标签的贡献, 返回该段部分和。掩码 isTgtTile: >0.5 即目标位, 置 0 不计。
    __aicore__ inline float AccumulateTileForLabels(uint32_t cnt, LocalTensor<float>& xTile,
                                                    LocalTensor<float>& isTgtTile, LocalTensor<float>& labX,
                                                    uint32_t nLab)
    {
        if (cnt == 0u || nLab == 0u) {
            return 0.0f;
        }
        using namespace AscendC::Reg;
        LocalTensor<float> accVec = reduceBuf.Get<float>();
        auto accAddr = (__ubuf__ float*)accVec.GetPhyAddr();
        auto xAddr = (__ubuf__ float*)xTile.GetPhyAddr();
        auto tgtAddr = (__ubuf__ float*)isTgtTile.GetPhyAddr();
        constexpr uint16_t VL = VECTOR_REG_WIDTH / sizeof(float);
        uint16_t repeatTimes = static_cast<uint16_t>((cnt + VL - 1u) / VL);

        {
            uint32_t sreg = cnt;
            __VEC_SCOPE__
            {
                RegTensor<float> z;
                MaskReg preg;
                for (uint16_t i = 0; i < repeatTimes; i++) {
                    preg = UpdateMask<float>(sreg);
                    Duplicate(z, 0.0f, preg);
                    DataCopy(accAddr + i * VL, z, preg);
                }
            }
        }
        PipeBarrier<HardEvent::V_S>();

        for (uint32_t j = 0; j < nLab; j++) {
            float s = 1.0f - labX.GetValue(j);
            uint32_t sreg = cnt;
            __VEC_SCOPE__
            {
                RegTensor<float> xr, tgtr, accr, tmp, zero;
                MaskReg preg, posM, tgtM;
                for (uint16_t i = 0; i < repeatTimes; i++) {
                    preg = UpdateMask<float>(sreg);
                    DataCopy(xr, xAddr + i * VL);
                    DataCopy(tgtr, tgtAddr + i * VL);
                    DataCopy(accr, accAddr + i * VL);
                    Duplicate(zero, 0.0f, preg);
                    Adds(tmp, xr, s, preg);                                   // (1 - x[k]) + x[i]
                    CompareScalar<float, CMPMODE::GT>(posM, tmp, 0.0f, preg); // 严格 >0, nan 安全
                    Select(tmp, tmp, zero, posM);
                    CompareScalar<float, CMPMODE::GT>(tgtM, tgtr, 0.5f, preg); // 目标位
                    Select(tmp, zero, tmp, tgtM);
                    Add(accr, accr, tmp, preg);
                    DataCopy(accAddr + i * VL, accr, preg);
                }
            }
        }
        return LocalReduceSum(accVec, cnt);
    }

    __aicore__ inline float ProcessRowSumTiled(uint32_t row)
    {
        const uint64_t rowOff = static_cast<uint64_t>(row) * static_cast<uint64_t>(this->C);
        LocalTensor<float> xTile = xRowBuf.Get<float>();
        LocalTensor<float> isTgtTile = isTgtFBuf.Get<float>();
        LocalTensor<float> labX = isPosBuf.Get<float>();
        LocalTensor<T> labRaw = workBuf.Get<float>().template ReinterpretCast<T>();
        float rowSum = 0.0f;

        // 逐 x 段处理。掩码在 UB 内就地建好(不经 GM 往返):
        // 曾用"标量写 is_target 到 GM 再回读当掩码"的做法, 实测会**丢写**(9000 类的一行少了 16 个 1,
        // loss 随之偏大 1.0018 倍), 标量写 + DataCacheCleanAndInvalid 在这种散点场景下不可靠。
        for (uint32_t xoff = 0; xoff < this->C; xoff += this->cFactor) {
            uint32_t xcnt = ((this->C - xoff) < this->cFactor) ? (this->C - xoff) : this->cFactor;

            // ---- (a) 扫 target 建本段掩码, 同时产出 is_target 段 ----
            LocalTensor<IsTgtT> mLocal = isTargetOutQueue.AllocTensor<IsTgtT>();
            Duplicate(mLocal, static_cast<IsTgtT>(0), xcnt);
            SetFlag<HardEvent::V_S>(EVENT_ID2);
            WaitFlag<HardEvent::V_S>(EVENT_ID2);
            bool stop = false;
            for (uint32_t soff = 0; soff < this->C && !stop; soff += this->cFactor) {
                uint32_t scnt = ((this->C - soff) < this->cFactor) ? (this->C - soff) : this->cFactor;
                LocalTensor<int32_t> tIn = targetQueue.AllocTensor<int32_t>();
                DataCopyExtParams cpT{1, static_cast<uint32_t>(scnt * sizeof(int32_t)), 0, 0, 0};
                DataCopyPadExtParams<int32_t> padT{false, 0, 0, 0};
                DataCopyPad(tIn, targetGm[rowOff + soff], cpT, padT);
                targetQueue.EnQue(tIn);
                LocalTensor<int32_t> tDeq = targetQueue.DeQue<int32_t>();
                // DeQue 给的是 MTE2->V; 下面是**标量**逐点读, 必须自己补 MTE2->S,
                // 否则读到尚未落地的旧数据 —— 表现为同一用例多次跑掩码丢的位置/个数都不同。
                SetFlag<HardEvent::MTE2_S>(EVENT_ID4);
                WaitFlag<HardEvent::MTE2_S>(EVENT_ID4);
                for (uint32_t t = 0; t < scnt; t++) {
                    int32_t tt = tDeq.GetValue(t);
                    if (tt == -1) {
                        stop = true;
                        break;
                    }
                    if (tt < 0 || static_cast<uint32_t>(tt) >= this->C) {
                        continue;
                    }
                    uint32_t u = static_cast<uint32_t>(tt);
                    if (u >= xoff && u < xoff + xcnt) {
                        mLocal.SetValue(u - xoff, static_cast<IsTgtT>(1));
                    }
                }
                targetQueue.FreeTensor(tDeq);
            }
            // 掩码转 float 供向量比较用, 再把 is_target 段写出(它是必产出的输出)。
            // 两处同步都必须有, 且**用独立事件 ID**:类内 PipeBarrier 固定 EVENT_ID0, 在本路径里会与
            // 队列同步/LocalReduceSum 的同 ID 事件交织, 实测表现为**同一用例多次跑结果不同**
            // (is_target 一次少 7 个 1、一次少 1167 个)。
            //   S->V : 标量写的 1 要被 Cast(向量)读到;
            //   S->MTE3: DataCopyPad 从 UB 搬出前必须等标量写落地 —— EnQue 只保证 V->MTE3,
            //            而这里最后的写者是标量单元, 缺这道同步就会丢写。
            SetFlag<HardEvent::S_V>(EVENT_ID2);
            WaitFlag<HardEvent::S_V>(EVENT_ID2);
            if constexpr (std::is_same<IsTgtT, float>::value) {
                uint32_t cnt8 = ((xcnt + 7u) / 8u) * 8u;
                DataCopy(isTgtTile, mLocal, cnt8);
            } else {
                Cast(isTgtTile, mLocal, RoundMode::CAST_NONE, xcnt);
            }
            SetFlag<HardEvent::S_MTE3>(EVENT_ID3);
            WaitFlag<HardEvent::S_MTE3>(EVENT_ID3);
            isTargetOutQueue.EnQue(mLocal);
            LocalTensor<IsTgtT> mDeq = isTargetOutQueue.DeQue<IsTgtT>();
            DataCopyExtParams cpM{1, static_cast<uint32_t>(xcnt * sizeof(IsTgtT)), 0, 0, 0};
            DataCopyPad(isTargetGm[rowOff + xoff], mDeq, cpM);
            isTargetOutQueue.FreeTensor(mDeq);

            // ---- (b) 载入本段 x ----
            LocalTensor<T> xIn = inputQueue.AllocTensor<T>();
            DataCopyExtParams cpX{1, static_cast<uint32_t>(xcnt * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> padX{false, 0, 0, 0};
            DataCopyPad(xIn, inputGm[rowOff + xoff], cpX, padX);
            inputQueue.EnQue(xIn);
            LocalTensor<T> xDeq = inputQueue.DeQue<T>();
            CastInputToFloat(xTile, xDeq, xcnt);
            inputQueue.FreeTensor(xDeq);
            PipeBarrier<HardEvent::V_S>();

            // ---- (c) 标签分批 × 本段累加 ----
            uint32_t scanPos = 0;
            bool scanDone = false;
            while (!scanDone) {
                uint32_t nLab = 0;
                while (nLab < this->cFactor && scanPos < this->C && !scanDone) {
                    uint32_t cnt = ((this->C - scanPos) < this->cFactor) ? (this->C - scanPos) : this->cFactor;
                    LocalTensor<int32_t> tIn = targetQueue.AllocTensor<int32_t>();
                    DataCopyExtParams cpT{1, static_cast<uint32_t>(cnt * sizeof(int32_t)), 0, 0, 0};
                    DataCopyPadExtParams<int32_t> padT{false, 0, 0, 0};
                    DataCopyPad(tIn, targetGm[rowOff + scanPos], cpT, padT);
                    targetQueue.EnQue(tIn);
                    LocalTensor<int32_t> tDeq = targetQueue.DeQue<int32_t>();
                    SetFlag<HardEvent::MTE2_S>(EVENT_ID4);
                    WaitFlag<HardEvent::MTE2_S>(EVENT_ID4);
                    uint32_t consumed = 0;
                    for (uint32_t t = 0; t < cnt; t++) {
                        consumed++;
                        int32_t tt = tDeq.GetValue(t);
                        if (tt == -1) {
                            scanDone = true;
                            break;
                        }
                        if (tt < 0 || static_cast<uint32_t>(tt) >= this->C) {
                            continue;
                        }
                        labRaw.SetValue(nLab, inputGm.GetValue(rowOff + static_cast<uint64_t>(tt)));
                        nLab++;
                        if (nLab >= this->cFactor) {
                            break;
                        }
                    }
                    targetQueue.FreeTensor(tDeq);
                    scanPos += consumed;
                }
                if (nLab == 0u) {
                    break;
                }
                SetFlag<HardEvent::S_V>(EVENT_ID2);
                WaitFlag<HardEvent::S_V>(EVENT_ID2);
                CastInputToFloat(labX, labRaw, nLab);
                SetFlag<HardEvent::V_S>(EVENT_ID2);
                WaitFlag<HardEvent::V_S>(EVENT_ID2);
                rowSum += AccumulateTileForLabels(xcnt, xTile, isTgtTile, labX, nLab);
            }
        }
        return rowSum;
    }

    // 行原始和 = Σ_i margins,未除 C。
    __aicore__ inline float ProcessRowSum(uint32_t row)
    {
        if (this->splitC) {
            return ProcessRowSumTiled(row);
        }
        const uint32_t cnt = this->C;

        LocalTensor<T> xRowIn;
        LocalTensor<int32_t> tgtIn;
        CopyInRow(row, cnt, xRowIn, tgtIn);

        LocalTensor<float> xRow = xRowBuf.Get<float>();
        LocalTensor<float> isPos = isPosBuf.Get<float>();

        CastInputToFloat(xRow, xRowIn, cnt);
        BuildMasks(row, cnt, tgtIn, isPos);

        float rowSum = AccumulateRowLoss(cnt, xRow, tgtIn, isPos);

        inputQueue.FreeTensor(xRowIn);
        targetQueue.FreeTensor(tgtIn);
        return rowSum;
    }

    // 单行损失 = 行原始和 / C。仅 reduction=none 用:它每行单独出一个输出,必须逐行除。
    // mean/sum 走 ProcessRowSum + FinalizeReduced 末尾统一除,少 N 次中间舍入。
    __aicore__ inline float ProcessRow(uint32_t row)
    {
        float rowSum = ProcessRowSum(row);
        return (this->C == 0u) ? 0.0f : (rowSum / static_cast<float>(static_cast<int32_t>(this->C)));
    }

    // Core 0 only: read the accumulated float workspace, apply mean division, cast to T, and write
    // the whole y tensor in one contiguous copy (single writer -> no multi-core race).
    // wsElems == N for reduction=none (per-row losses), or 1 for mean/sum (the reduced scalar).
    // Core 0 only (reduction=mean/sum): 按 this->wsCoreStride 跨步读回各核 partial,
    // Kahan 合并后统一除一次 N,再 cast 写标量 y。
    __aicore__ inline void FinalizeReduced()
    {
        uint32_t wsElems = this->usedCoreNum * this->wsCoreStride;
        for (uint32_t off = 0; off < wsElems; off += this->wsCoreStride) {
            DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(workspaceGm[off]);
        }
        LocalTensor<float> partials = partialsInQueue.AllocTensor<float>();
        DataCopyExtParams cpIn{1, wsElems * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> padIn{false, 0, 0, 0};
        DataCopyPad(partials, workspaceGm[0], cpIn, padIn);
        SetFlag<HardEvent::MTE2_S>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_S>(EVENT_ID0);

        // Kahan 补偿累加(同 Process 中的核内累加)。跨核合并顺序由 blockIdx 定,与调度无关 -> 结果可复现。
        float total = 0.0f;
        float comp = 0.0f;
        for (uint32_t i = 0; i < this->usedCoreNum; i++) {
            float y = partials.GetValue(i * this->wsCoreStride) - comp;
            float t = total + y;
            comp = (t - total) - y;
            total = t;
        }
        partialsInQueue.FreeTensor(partials);

        total = ApplyReductionDivisor(total);

        WriteScalarOutput(total);
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
            return total;
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
        LocalTensor<float> acc = gatherBuf.Get<float>();
        acc.SetValue(0, v);
        PipeBarrier<HardEvent::S_V>();
        LocalTensor<T> outVec = gatherOutBuf.Get<T>();
        if constexpr (std::is_same<T, float>::value) {
            Adds(outVec, acc, 0.0f, 1);
        } else if constexpr (std::is_same<T, bfloat16_t>::value) {
            Cast(outVec, acc, RoundMode::CAST_RINT, 1);
        } else {
            Cast(outVec, acc, RoundMode::CAST_NONE, 1);
        }
        PipeBarrier<HardEvent::V_MTE3>();
        DataCopyExtParams cpOut{1, static_cast<uint32_t>(sizeof(T)), 0, 0, 0};
        DataCopyPad(outputGm[0], outVec, cpOut);
    }

    __aicore__ inline void FinalizeOutput(uint32_t wsElems)
    {
        if (wsElems == 0u) {
            return;
        }
        LocalTensor<float> acc = gatherBuf.Get<float>();
        LocalTensor<T> outVec = gatherOutBuf.Get<T>();
        // 按 this->ubFactor 分块回读+写出:UB 占用定长,不随 N 增长。
        for (uint32_t base = 0; base < wsElems; base += this->ubFactor) {
            uint32_t cur = wsElems - base;
            if (cur > this->ubFactor) {
                cur = this->ubFactor;
            }
            // Invalidate core 0's cached view of the workspace it zero-initialised, so the read below
            // sees the values other cores atomic-added (stride 8 floats = 32B covers any cache line).
            for (uint32_t off = base; off < base + cur; off += 8u) {
                DataCacheCleanAndInvalid<float, CacheLine::SINGLE_CACHE_LINE, DcciDst::CACHELINE_OUT>(workspaceGm[off]);
            }
            DataCopyExtParams cpIn{1, cur * static_cast<uint32_t>(sizeof(float)), 0, 0, 0};
            DataCopyPadExtParams<float> padIn{false, 0, 0, 0};
            DataCopyPad(acc, workspaceGm[base], cpIn, padIn);
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
            DataCopyPad(outputGm[base], outVec, cpOut);
            PipeBarrier<HardEvent::MTE3_V>(); // 下一块 Cast 复写 outVec 前,等本块搬出完成
            PipeBarrier<HardEvent::V_MTE2>(); // 下一块搬入 acc 前,等本块 Cast 读完
        }
    }
};

} // namespace NsMultilabelMarginLoss
#endif
