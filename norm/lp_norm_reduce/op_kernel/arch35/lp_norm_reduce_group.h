/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// =============================================================================
// LpNormReduce_package/op_kernel/arch35/lp_norm_reduce_group.h
// =============================================================================
//
// ROLE: LpNormReduce Group 模板（tilingKey=1）kernel 类实现。
//
//   据 docs/LpNormReduce/design/Kernel.md §9.8（Group 两阶段规格与复用表）与
//   docs/LpNormReduce/design/branches/DESIGN-BRANCH-1.md §3–§5（流水线 / 内存
//   规划 / Kernel 计算链）落码。A×R 2D 分核两阶段数据流：
//
//     ProcessGroup = Phase1Process → SyncAll → Phase2Process
//
//   - Phase 1：每核把分到的 R 区间（rCount 个 rLoop 迭代，大小核式均匀分配）
//     对自己的那一个 A chunk（aPerCore=1 恒成立）做核内局部二分缓存树累加，
//     得 1 个 fp32 partial，从 cacheBuf 局部树根 DataCopyPad 直写 workspace
//     第 rChunkIdx 行（跳过 PostElewise——partial 保持 fp32，缩位 Cast 留到
//     Phase 2；⛔ 不中转——fp16 时中转 buf 可能比 cacheBuf 小有越界风险）。
//   - SyncAll：全文件唯一一处 AscendC::SyncAll()，建立「全核 workspace 写入 →
//     全核读取」跨核依赖（Host 侧 SetScheduleMode(1) batch 调度为其前提）。
//   - Phase 2：RA mini-kernel 消解 workspace 的 rGroupCnt 维——每核按现算
//     aUbFactorP2 重切 A 方向，CopyIn(workspace) → ReduceXxx(RA) →
//     PostElewiseP2(缩位 Cast) → CopyOut(GM y)。两次 Reduce 串联合并等价
//     base 一次全 R reduce。
//
//   复用（Kernel.md §9.8 复用表 / DESIGN-BRANCH-1.md §5「不重复产出」约束）：
//   跨分支共享 VF 链与 CopyIn 机制全部经继承 LpNormReduceBaseKernel<DType>
//   引用 base 既有实现（不复制粘贴）——PreElewise / pad 清零 / MergeTmpBuf /
//   ReduceChunk / DoCaching（Kernel.md §9.1–§9.5，base §9.6 PostElewise 的 VF
//   本体 PostElewiseVfImpl 亦为 base 命名空间自由模板函数，Phase 2 复用）、
//   BuildUBAxes + DoCopyInTile + ProcessOneRChunk + UnravelALoop / UnravelRLoop
//   解码（DESIGN-BRANCH-0.md §5.2，与 base 逐字相同）。
//
//   与 base Init 的差异（DESIGN-BRANCH-1.md §5.1）：其一，多绑定 workspace（fp32
//   partial 矩阵 GM，[rGroupCnt, aTotal] 行优先 dense）；其二，二分树参数不在 Init
//   预计算——按局部 rCount 在 Phase1Process 现算（Kernel.md §9.8 局部化约束）；
//   其三，V_MTE2 取两个互异 Event ID（evVtoMTE2_ = Phase 1 循环 preInBuf WAR、
//   evVtoMTE2P2_ = 阶段过渡 preRes WAR + Phase 2 循环 WAR，§5.5 事件 ID 分配表）。
//
// =============================================================================

#ifndef LP_NORM_REDUCE_GROUP_H_
#define LP_NORM_REDUCE_GROUP_H_

#include "kernel_operator.h"            // Ascend C core framework
#include "adv_api/reduce/reduce.h"      // ReduceSum / ReduceMax / ReduceMin + Pattern
#include "lp_norm_reduce_base.h"        // Base 模板（VF 链 + CopyIn 机制复用源）
#include "lp_norm_reduce_tiling_data.h" // LpNormReduceTilingData（base/group 共用）

namespace NsLpNormReduce {

// ===========================================================================
// LpNormReduceGroupKernel<DType> —— Group 模板 kernel 类（tilingKey=1）
//
// 公共继承 LpNormReduceBaseKernel<DType>：protected 的 VF wrapper / CopyIn 机制 /
// Unravel 解码 / 二分树数学 / TBuf / Event ID 全部直接复用（「引用 base 既有实现，
// 不复制粘贴」）。base 的 Init / Process / PostElewise / CopyOut 不被本类调用
// （模板成员函数按需实例化，不引入死代码）。
// ===========================================================================
template <typename DType>
class LpNormReduceGroupKernel : public LpNormReduceBaseKernel<DType> {
public:
    using DT = DType;
    using Base = LpNormReduceBaseKernel<DT>;

    __aicore__ inline LpNormReduceGroupKernel() : Base() {}

    // Group 初始化（比 base Init 多绑定 workspace；二分树三参数【不在此预计算】，
    // 按局部 rCount 在 Phase1Process 现算——Kernel.md §9.8 局部化约束）
    __aicore__ inline void InitGroup(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const LpNormReduceTilingData* td,
                                     TPipe* pipe);
    // Group 主流程（§5.3）：Phase1Process → SyncAll → Phase2Process
    __aicore__ inline void ProcessGroup();

private:
    __aicore__ inline void Phase1Process(); // §5.3：局部归约 → workspace
    __aicore__ inline void Phase2Process(); // §5.3：RA mini-kernel → GM y
    // §5.4：cacheBuf 局部树根 → workspace 第 rChunkIdx 行（fp32，三路径同 base
    // 仅 sizeof(D_T)→sizeof(float)；⛔ 直接从 cacheBuf DataCopyPad、不中转）
    __aicore__ inline void CopyOutPhase1(int64_t rChunkIdx, int64_t localRootOff, int64_t chunkOutOff, int64_t aLen);
    // §5.2：workspace[rGroupCnt, aLen] → preReduceResult（fp32 partial，RA 布局）
    __aicore__ inline void CopyInWorkspaceP2(int64_t aOff, int64_t aLen);
    // §9.8 PostElewiseP2：树根 = cacheBuf[0]（cacheCount=1 退化、rootOff=0）、
    // 缩位 Cast → outBuf（复用 base PostElewiseVfImpl）
    __aicore__ inline void PostElewiseP2(int64_t aLen);

    // ─── 依赖基类模板的 protected 成员（模板基类名字查找需显式 using） ───
    // 数据 / buffer / Event ID
    using Base::cacheBuf_;
    using Base::evMTE2toV_;
    using Base::evMte3toV_;
    using Base::evVtoMTE2_;
    using Base::evVtoMTE3_;
    using Base::isTailR_;
    using Base::outBuf_;
    using Base::outStride_;
    using Base::pipe_;
    using Base::preInBuf_;
    using Base::preReduceResult_;
    using Base::preReduceResultTail_;
    using Base::rSplitChunkCnt_;
    using Base::td_;
    using Base::xGm_;
    using Base::yGm_;
    // 复用的 base 方法（VF 链 wrapper / CopyIn 机制 / 解码 / 二分树数学）
    using Base::CalLog2;
    using Base::DoCachingVf;
    using Base::FindNearestPower2;
    using Base::GetCacheID;
    using Base::LastAAxis;
    using Base::MergeTmpBufVf;
    using Base::ProcessOneRChunk;
    using Base::ReduceChunk;
    using Base::UnravelALoop;

    // GM：workspace 用户区 [rGroupCnt, aTotal] fp32 行优先 dense（§8；
    // 入口已经 GetUserWorkspace 换算到用户区起始——950 运行时在 workspace 头部
    // 保留 16MB 系统区（RESERVED_WORKSPACE，ffts 跨核同步 mailbox 所在），
    // 直绑原始地址会踩 SyncAll 信箱，bn3d_training_reduce / gn_training_reduce 同款）
    GlobalTensor<float> wsGm_;
    int64_t aTotal_ = 0; // ∏(全部 A 轴 axisShape) = workspace 列数 = 输出元素总数（InitGroup 现算）
    // 阶段过渡（P1 末次 V 用 preRes → P2 首次 CopyIn 覆写）+ Phase 2 跨迭代 WAR
    // （本轮 Reduce → 下轮 CopyIn）共用的 V_MTE2 ID——与 evVtoMTE2_（Phase 1 循环
    // preInBuf WAR）互异（§5.5：两阶段各用一个 V_MTE2 ID，避免 Phase 1 末轮未消费
    // 的 Set 与 Phase 2 首个 Set 连续双 Set 破坏严格交替）
    int32_t evVtoMTE2P2_ = 0;
};

// ---------------------------------------------------------------------------
// InitGroup（DESIGN-BRANCH-1.md §5.1）
// ---------------------------------------------------------------------------
template <typename DType>
__aicore__ inline void LpNormReduceGroupKernel<DType>::InitGroup(GM_ADDR x, GM_ADDR y, GM_ADDR workspace,
                                                                 const LpNormReduceTilingData* td, TPipe* pipe)
{
    td_ = td;
    isTailR_ = (td_->axisNum % AXIS_INTERVAL == 0); // 偶数轴→tail-R，奇数轴→tail-A

    // UnravelRLoop（base §5.2，经 ProcessOneRChunk 消费）依赖 rSplitChunkCnt_
    rSplitChunkCnt_ = Ops::Base::CeilDiv(td->axisShape[td->rSplitIdx], td->rUbFactor);

    // 输出 / workspace 列步长：输出 = 各 A 轴 size 顺序拼接、连续紧凑无 R 维
    // （output_strides[k_A] = ∏(更内 A 轴 size)，kernel 端从 axisShape 现算，同 base）
    {
        int64_t outStrideAcc = 1;
        for (int32_t i = td->axisNum - 1; i >= 0; --i) {
            if (i % AXIS_INTERVAL == 0) {
                outStride_[i] = outStrideAcc;
                outStrideAcc *= td->axisShape[i];
            }
        }
    }

    // workspace 列数 aTotal_ = ∏(全部 A 轴)（§5.1 唯二预计算派生量之一）
    aTotal_ = 1;
    for (int32_t k = 0; k < td_->axisNum; k += AXIS_INTERVAL) {
        aTotal_ *= td_->axisShape[k];
    }

    // GM 绑定（本分支比 base 多绑定 workspace：Phase 1 写 / Phase 2 读，§8）
    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT*>(x));
    yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT*>(y));
    wsGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));

    // TBuf 分配（§4 划分表：3 路 pre 同尺寸 + cacheBuf 恒 16KB + outBuf；全部
    // VECCALC、深度 1，范式约束统一 TBuf 不使用 TQue；Phase 1 / Phase 2 共用
    // 同一组物理槽——SyncAll 天然隔断两阶段生命周期）
    pipe_ = pipe;
    pipe_->InitBuffer(preInBuf_, td->preBufSize);
    pipe_->InitBuffer(preReduceResult_, td->preBufSize);
    pipe_->InitBuffer(preReduceResultTail_, td->preBufSize);
    pipe_->InitBuffer(cacheBuf_, td->cacheBufUbSize);
    pipe_->InitBuffer(outBuf_, td->postBufSize);

    // Event ID（§5.5 持有法则 trace 结论：4 类事件、5 个 ID——V_MTE2 取两个）。
    // ⚠ FetchEventID 只窥视不占用（同型两次调用恒返回同一 ID），而 §5.5 要求
    // evVtoMTE2_ / evVtoMTE2P2_ 为两个互异 ID，故 V_MTE2 用 AllocEventID 真实
    // 分配（0 / 1）；其余三类与 base 相同用 FetchEventID（各自独立 ID 池，恒 0）
    evMTE2toV_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    evVtoMTE3_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    evMte3toV_ = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    evVtoMTE2_ = static_cast<int32_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
    evVtoMTE2P2_ = static_cast<int32_t>(GetTPipePtr()->AllocEventID<HardEvent::V_MTE2>());
}

// ---------------------------------------------------------------------------
// ProcessGroup（DESIGN-BRANCH-1.md §5.3 / Kernel.md §9.8 三段式封装）
// ---------------------------------------------------------------------------
template <typename DType>
__aicore__ inline void LpNormReduceGroupKernel<DType>::ProcessGroup()
{
    Phase1Process();
    AscendC::SyncAll(); // 全核同步：Phase 1 workspace 写入 → Phase 2 读取的跨核依赖
                        // （全文件唯一一处；Host 侧 SetScheduleMode(1) batch 调度为其前提）
    Phase2Process();
}

// ---------------------------------------------------------------------------
// Phase1Process（DESIGN-BRANCH-1.md §5.3）—— 核内局部归约 → workspace
// 持有（§4 划分表，峰值 ≤ kPhysNodes=5）：循环内 Phase A 尾块 PreElewise 时
// preIn + preRes + preResTail（3）+ 常驻 cacheBuf（1）；收尾 cacheBuf（1）。
// ---------------------------------------------------------------------------
template <typename DType>
__aicore__ inline void LpNormReduceGroupKernel<DType>::Phase1Process()
{
    const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());

    // ── 1) 2D 网格坐标（§2 Step 2：blockIdx = aChunkIdx × rGroupCnt + rChunkIdx，
    //       aPerCore=1 恒成立——numBlocks 对齐到 aLoopCntTotal 整数倍保证） ──
    const int64_t aChunkIdx = blockIdx / td_->rGroupCnt; // 本核唯一的 A chunk
    const int64_t rChunkIdx = blockIdx % td_->rGroupCnt; // workspace 行号 ∈ [0, rGroupCnt)

    // ── 2) R 区间大小核式均匀分配（Kernel.md §9.8；⛔ 禁止 CeilDiv 截断式分配——
    //       会造出空组，Phase 2 读脏 workspace 行，静默数值错误） ──
    const int64_t rSmallGroupLoopCnt = td_->rLoopCntTotal / td_->rGroupCnt;
    const int64_t rBigGroupCnt = td_->rLoopCntTotal % td_->rGroupCnt;
    const int64_t rBigGroupLoopCnt = rSmallGroupLoopCnt + (rBigGroupCnt > 0 ? 1 : 0);
    int64_t rStart = 0;
    int64_t rCount = 0;
    if (rChunkIdx < rBigGroupCnt) {
        rStart = rChunkIdx * rBigGroupLoopCnt;
        rCount = rBigGroupLoopCnt;
    } else {
        rStart = rBigGroupCnt * rBigGroupLoopCnt + (rChunkIdx - rBigGroupCnt) * rSmallGroupLoopCnt;
        rCount = rSmallGroupLoopCnt;
    }
    if (rStart >= td_->rLoopCntTotal) {
        return; // 防御性早退（rGroupCnt ≤ rLoopCntTotal 恒成立、理论不可达；防 tiling 公式被改坏）
    }

    // ── 3) 局部二分树参数（Kernel.md §9.2 公式，入参换局部 rCount；⚠ 二分树
    //       三参数不在 Init 预计算——Kernel.md §9.8 局部化约束） ──
    const int64_t bisPos = static_cast<int64_t>(FindNearestPower2(static_cast<uint64_t>(rCount)));
    const int64_t bisTail = rCount - bisPos;

    // ── 4) A chunk 解码（每核恰 1 个，UnravelALoop 同 base §5.2；chunkGmOff 为
    //       A 侧输入 GM 偏移、chunkOutOff 为 A chunk 在输出空间的列偏移——与
    //       base Process 的推导逐字相同） ──
    int64_t aIdx[MAX_PATTERN_RANK] = {0};
    int64_t aSplitChunkIdx = 0;
    UnravelALoop(aChunkIdx, aIdx, aSplitChunkIdx);
    int64_t chunkGmOff = 0;
    int64_t chunkOutOff = 0;
    for (int32_t k = td_->aSplitIdx - AXIS_INTERVAL; k >= 0; k -= AXIS_INTERVAL) {
        chunkGmOff += aIdx[k] * td_->axisStride[k];
        chunkOutOff += aIdx[k] * outStride_[k];
    }
    const int64_t aChunkStart = aSplitChunkIdx * td_->aUbFactor; // A 切分尾块 valid 起点
    const int64_t aEnd = aChunkStart + td_->aUbFactor;
    const int64_t aLen = (aEnd > td_->axisShape[td_->aSplitIdx]) ? (td_->axisShape[td_->aSplitIdx] - aChunkStart) :
                                                                   td_->aUbFactor;
    chunkGmOff += aChunkStart * td_->axisStride[td_->aSplitIdx];
    chunkOutOff += aChunkStart * outStride_[td_->aSplitIdx];

    __ubuf__ float* preRes = reinterpret_cast<__ubuf__ float*>(preReduceResult_.template Get<float>().GetPhyAddr());
    __ubuf__ float* preResTail = reinterpret_cast<__ubuf__ float*>(
        preReduceResultTail_.template Get<float>().GetPhyAddr());

    bool firstChunkOfCore = true; // 首个 CopyIn 前跳过 WaitFlag<V_MTE2>（§5.5 首轮规则）

    // ── 5) 局部 R 主段循环 [0, bisPos)：一棵独立局部二分缓存树 ──
    for (int64_t localI = 0; localI < bisPos; ++localI) {
        // 主块：CopyIn(MTE2) + PreElewise(V) → preReduceResult（ProcessOneRChunk
        //    同 base §5.2，入参 rIdx = 全局下标 rStart + localI；公式 |x| / 1[x≠0] /
        //    |x|^p，Kernel.md §9.1；pad 清零在 ProcessOneRChunk 内完成）
        ProcessOneRChunk(chunkGmOff, aLen, rStart + localI, preRes, firstChunkOfCore);
        firstChunkOfCore = false;

        // Phase A（localI < bisTail）：配对尾块 localI + bisPos → preReduceResultTail，
        //    MergeTmpBufVf 逐元素合并进主块（公式 Σ 拼接 R：sum=Add / max=Max / min=Min
        //    按 pOrder，Kernel.md §9.3）
        if (localI < bisTail) {
            ProcessOneRChunk(chunkGmOff, aLen, rStart + localI + bisPos, preResTail, firstChunkOfCore);
            MergeTmpBufVf(preRes, preResTail);
        }

        // Reduce（Kernel.md §9.5 ReduceChunk：ReduceSum/Max/Min 按 pOrder 分发）：
        //    dst = cacheBuf[cacheID × levelStride]，sharedTmpBuffer = preResTail
        //    ⚠ cacheID 用【局部】下标 GetCacheID(localI)（Kernel.md §9.8 局部化约束）
        const uint16_t cacheID = GetCacheID(localI);
        ReduceChunk(cacheID);
        // DoCaching（V，Kernel.md §9.2：本层就地吸收全部低层；合并算子按 pOrder 分发）
        DoCachingVf(cacheID);
    }

    // ── 6) Phase 1 收尾：局部树根 → workspace 第 rChunkIdx 行（fp32） ──
    SetFlag<HardEvent::V_MTE3>(evVtoMTE3_);   // RAW：V 写局部树根（含最后一次 DoCaching）→ MTE3 读
    SetFlag<HardEvent::V_MTE2>(evVtoMTE2P2_); // WAR：R 循环最后一次 V 用 preRes → Phase 2 MTE2 覆写
    WaitFlag<HardEvent::V_MTE3>(evVtoMTE3_);  // 等局部树根落定（CopyOutPhase1 的 MTE3 读依赖）
    const uint32_t laneA = static_cast<uint32_t>(td_->aUbFactor * td_->innerAProdAlign);
    const int64_t levelStride = static_cast<int64_t>(Ops::Base::CeilAlign(laneA, UB_BLOCK_F32));
    // ⚠ 局部树根：localRoot = CalLog2(局部 bisPos) × levelStride（= (局部 cacheCount−1)
    //   × levelStride）——禁止用全局 rLoopCntTotal 的 cacheCount 定位树根（§5.3）
    const int64_t localRootOff = static_cast<int64_t>(CalLog2(static_cast<uint64_t>(bisPos))) * levelStride;
    // Phase 1 跳过 PostElewise（partial 保持 fp32，缩位 Cast 留到 Phase 2，§5.3 注）
    CopyOutPhase1(rChunkIdx, localRootOff, chunkOutOff, aLen); // §5.4：直出 cacheBuf，⛔ 不中转
    SetFlag<HardEvent::MTE3_V>(evMte3toV_);                    // WAR：MTE3 读 cacheBuf → Phase 2 V 写 cacheBuf[0]
}

// ---------------------------------------------------------------------------
// Phase2Process（DESIGN-BRANCH-1.md §5.3 / Kernel.md §9.8）—— RA mini-kernel
// 持有：循环内 Reduce 时 preRes + preResTail + cacheBuf（3）→ PostElewise 时
// cacheBuf(根) + outBuf（2），均 ≤ kPhysNodes=5。
// ---------------------------------------------------------------------------
template <typename DType>
__aicore__ inline void LpNormReduceGroupKernel<DType>::Phase2Process()
{
    const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());

    // ── 0) 阶段过渡 WAR 消费（§5.5）：Phase 1 的两个 outstanding Set 在此 Wait ──
    WaitFlag<HardEvent::V_MTE2>(evVtoMTE2P2_); // preRes：Phase 1 末次 V 用完 → 本阶段 CopyIn 覆写
    WaitFlag<HardEvent::MTE3_V>(evMte3toV_);   // cacheBuf：Phase 1 CopyOut 读完 → 本阶段 V 写 cacheBuf[0]

    // ── 1) kernel 侧现算切分参数（Phase 2 专用，不读 TilingData 切分字段——范式
    //       §4.2 约束；⚠ aUbFactorP2 ≠ TilingData 的 aUbFactor：Phase 1 UB 布局
    //       [A_bundle, R_bundle]（R 在内）、Phase 2 UB 布局 [rGroupCnt, a_len_ub]
    //       （RA，A 在内），容量切分方向不同，须重新计算） ──
    const int64_t preInElems = td_->preBufSize / static_cast<int64_t>(sizeof(float));
    constexpr int64_t bsFp32 = 8;                                            // 32B / sizeof(fp32)
    int64_t aUbFactorP2 = preInElems / static_cast<int64_t>(td_->rGroupCnt); // floor，按 R 分组均分 preBuf
    if (aUbFactorP2 >= bsFp32) {
        aUbFactorP2 = (aUbFactorP2 / bsFp32) * bsFp32; // ★ burst 尾轴 32B 对齐，小值不归零
    }
    const int64_t postCapElems = static_cast<int64_t>(td_->postBufSize) / static_cast<int64_t>(sizeof(DT));
    aUbFactorP2 = (aUbFactorP2 < postCapElems) ? aUbFactorP2 : postCapElems; // postBuf 容量上界
    aUbFactorP2 = (aUbFactorP2 < aTotal_) ? aUbFactorP2 : aTotal_;           // A 总长上界
    if (aUbFactorP2 <= 0) {
        return; // 防御（合法 tiling 下不可达：rGroupCnt ≤ rLoopCntTotal ≤ preInElems；防除零挂死）
    }
    const int64_t aSplitChunkCntP2 = Ops::Base::CeilDiv(aTotal_, aUbFactorP2);
    const int64_t aLoopCntTotalP2 = aSplitChunkCntP2; // outerAProd = 1 退化

    // 大小核均衡（usedCoreNumP2 可能 < usedCoreNum：aLoopCntTotalP2 < usedCoreNum
    // 时多余核早退——⚠ 须在阶段过渡 Wait 之后早退，保证每个启动核先消费自己的
    // 过渡 Set 再早退，不遗留未消费标志，§5.5）
    const int64_t aSmallCoreLoopCntP2 = aLoopCntTotalP2 / static_cast<int64_t>(td_->usedCoreNum);
    const int64_t aBigCoreCntP2 = aLoopCntTotalP2 % static_cast<int64_t>(td_->usedCoreNum);
    const int64_t aBigCoreLoopCntP2 = aSmallCoreLoopCntP2 + (aBigCoreCntP2 > 0 ? 1 : 0);
    const int64_t usedCoreNumP2 = (aSmallCoreLoopCntP2 > 0) ? static_cast<int64_t>(td_->usedCoreNum) : aBigCoreCntP2;
    if (blockIdx >= usedCoreNumP2) {
        return;
    }
    int64_t aLoopStart = 0;
    int64_t aLoopEnd = 0; // blockIdx → [aLoopStart, aLoopEnd)，区间不重叠（每列恰写一次）
    if (blockIdx < aBigCoreCntP2) {
        aLoopStart = blockIdx * aBigCoreLoopCntP2;
        aLoopEnd = aLoopStart + aBigCoreLoopCntP2;
    } else {
        aLoopStart = aBigCoreCntP2 * aBigCoreLoopCntP2 + (blockIdx - aBigCoreCntP2) * aSmallCoreLoopCntP2;
        aLoopEnd = aLoopStart + aSmallCoreLoopCntP2;
    }

    // ── 2) aLoop 主循环 ──
    for (int64_t aLoopIdx = aLoopStart; aLoopIdx < aLoopEnd; ++aLoopIdx) {
        const int64_t aSplitChunkIdx = aLoopIdx;           // outerAProd = 1
        const int64_t aOff = aSplitChunkIdx * aUbFactorP2; // valid 当前块起始（workspace 列 / GM y 偏移）
        const int64_t aLen = (aUbFactorP2 < aTotal_ - aOff) ? aUbFactorP2 : (aTotal_ - aOff); // valid
        const int64_t aLenUb = Ops::Base::CeilAlign(aLen, bsFp32);                            // padded UB 行步长

        // 2a) CopyIn（MTE2）：workspace[rGroupCnt, aLen] → preReduceResult
        //     （复用 Phase 1 物理槽，R=rGroupCnt 全载；fp32 partial——无 PreElewise、
        //     无 pad 清零、无二分树，A 方向 pad 不进 reduce 结果）
        if (aLoopIdx != aLoopStart) {
            WaitFlag<HardEvent::V_MTE2>(evVtoMTE2P2_); // WAR：上轮 Reduce(V 读 preRes) → 本轮 MTE2 覆写
        }
        CopyInWorkspaceP2(aOff, aLen);          // §5.2 Phase 2 搬入
        SetFlag<HardEvent::MTE2_V>(evMTE2toV_); // RAW：MTE2 写 preRes → V 读
        WaitFlag<HardEvent::MTE2_V>(evMTE2toV_);

        // 2b) Reduce（V/Reduce 硬件）：RA pattern、R=rGroupCnt 全载单 chunk、dst=cacheBuf[0]
        //     （无二分树层级）；公式 max/min/Σ 跨核 partial 合并（ReduceSum/Max/Min
        //     按 pOrder 分发）；⚠ srcShape 必须用 aLenUb（padded）——ReduceXxx 按
        //     aLenUb 行步长读
        uint32_t srcShape[REDUCE_SHAPE_DIM] = {static_cast<uint32_t>(td_->rGroupCnt), static_cast<uint32_t>(aLenUb)};
        if (td_->pOrder == P_INF_SENTINEL) {
            AscendC::ReduceMax<float, AscendC::Pattern::Reduce::RA, /*isReuseSource=*/true>(
                cacheBuf_.template Get<float>()[0], preReduceResult_.template Get<float>(),
                preReduceResultTail_.template Get<uint8_t>(), srcShape, /*srcInnerPad=*/true);
        } else if (td_->pOrder == N_INF_SENTINEL) {
            AscendC::ReduceMin<float, AscendC::Pattern::Reduce::RA, /*isReuseSource=*/true>(
                cacheBuf_.template Get<float>()[0], preReduceResult_.template Get<float>(),
                preReduceResultTail_.template Get<uint8_t>(), srcShape, /*srcInnerPad=*/true);
        } else {
            AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, /*isReuseSource=*/true>(
                cacheBuf_.template Get<float>()[0], preReduceResult_.template Get<float>(),
                preReduceResultTail_.template Get<uint8_t>(), srcShape, /*srcInnerPad=*/true);
        }
        SetFlag<HardEvent::V_MTE2>(evVtoMTE2P2_); // WAR：本轮 V 读 preRes 完成 → 下轮可覆写

        // 2c) PostElewiseP2（V，Kernel.md §9.6/§9.8：树根=cacheBuf[0]、缩位 Cast →
        //     outBuf；无算子专属后处理——不开方）
        if (aLoopIdx != aLoopStart) {
            WaitFlag<HardEvent::MTE3_V>(evMte3toV_); // WAR：上轮 CopyOut(MTE3 读 outBuf) → 本轮 V 写
        }
        PostElewiseP2(aLen);
        SetFlag<HardEvent::V_MTE3>(evVtoMTE3_); // RAW：V 写 outBuf → MTE3 读
        WaitFlag<HardEvent::V_MTE3>(evVtoMTE3_);

        // 2d) CopyOut（MTE3，单路径）：outBuf → GM_y（valid = aLen × sizeof(D_T)，§5.4；
        //     A 方向 pad 不被写出）
        DataCopyExtParams outParams;
        outParams.blockLen = static_cast<uint32_t>(aLen * static_cast<int64_t>(sizeof(DT)));
        outParams.blockCount = 1;
        outParams.srcStride = 0; // UB 侧 gap=0，HW 自动按 CeilAlign(blockLen, 32B) 读取
        outParams.dstStride = 0; // GM y dense
        outParams.rsv = 0;       // 必须显式填 0（datacopypad-rules）
        DataCopyPad(yGm_[aOff], outBuf_.template Get<DT>(), outParams);
        SetFlag<HardEvent::MTE3_V>(evMte3toV_); // WAR：跨 aLoop 迭代 outBuf 复用（末轮无害保留）
    }
}

// ---------------------------------------------------------------------------
// CopyOutPhase1（DESIGN-BRANCH-1.md §5.4）—— 三路径决策（依 isTailR_ 与
// aSplitIdx 位置；与 base CopyOut 三路径一致、仅 sizeof(D_T)→sizeof(float) 且
// srcStride 按 fp32 域重算）。写入偏移 wsOff = rChunkIdx × aTotal + chunkOutOff
// （每核写不同行 rChunkIdx 互异，无写冲突；All Reduce 场景 aTotal=1 时
// chunkOutOff 恒 0）。
// ---------------------------------------------------------------------------
template <typename DType>
__aicore__ inline void LpNormReduceGroupKernel<DType>::CopyOutPhase1(int64_t rChunkIdx, int64_t localRootOff,
                                                                     int64_t chunkOutOff, int64_t aLen)
{
    // innerAProd = ∏ axisShape[aSplitIdx+2 .. LastA]（真实乘积，kernel 端现算）
    int64_t innerAProd = 1;
    for (int32_t k = td_->aSplitIdx + AXIS_INTERVAL; k < td_->axisNum; k += AXIS_INTERVAL) {
        innerAProd *= td_->axisShape[k];
    }

    DataCopyExtParams wsParams;
    if (isTailR_) {
        // 路径 1：tail-R，A_bundle dense 单 burst
        wsParams.blockLen = static_cast<uint32_t>(aLen * innerAProd * static_cast<int64_t>(sizeof(float)));
        wsParams.blockCount = 1;
        wsParams.srcStride = 0;
    } else {
        const int32_t lastA = LastAAxis(); // 最大偶下标 A 轴（规整后恒存在）
        const int64_t lastASize = td_->axisShape[lastA];
        if (td_->aSplitIdx == lastA) {
            // 路径 2：tail-A 且切最内 A 轴，单 burst
            wsParams.blockLen = static_cast<uint32_t>(aLen * static_cast<int64_t>(sizeof(float)));
            wsParams.blockCount = 1;
            wsParams.srcStride = 0;
        } else {
            // 路径 3：tail-A 且非切最内 A 轴，多 burst（cacheBuf 树根行步距 gap 显式
            // 给出；lastASizeAlign 按输入 dtype bsElem 对齐，gap ≤ 60B → srcStride
            // 只能为 0 或 1——与 base outBuf 路径 3 的 srcStride=0 不同：cacheBuf 树根
            // 行的 A_bundle 行步距由输入域对齐决定，非 CeilAlign(blockLen,32B) 恰好重合）
            const int64_t bsElem = static_cast<int64_t>(UB_BLOCK_BYTES) / static_cast<int64_t>(sizeof(DT));
            const int64_t lastASizeAlign = Ops::Base::CeilAlign(lastASize, bsElem);
            wsParams.blockLen = static_cast<uint32_t>(lastASize * static_cast<int64_t>(sizeof(float)));
            wsParams.blockCount = static_cast<uint16_t>(aLen * innerAProd / lastASize);
            wsParams.srcStride = static_cast<uint32_t>((lastASizeAlign - lastASize) *
                                                       static_cast<int64_t>(sizeof(float)) / UB_BLOCK_BYTES);
        }
    }
    wsParams.dstStride = 0; // workspace GM dense 写出
    wsParams.rsv = 0;       // 必须显式填 0（datacopypad-rules）
    DataCopyPad(wsGm_[rChunkIdx * aTotal_ + chunkOutOff], cacheBuf_.template Get<float>()[localRootOff], wsParams);
}

// ---------------------------------------------------------------------------
// CopyInWorkspaceP2（DESIGN-BRANCH-1.md §5.2）—— workspace fp32 partial 搬入。
// ⚠ 逐项约束（Kernel.md §9.8）：blockCount = 全部 R 分组（不是 1）；
// srcStride = workspace 行间 gap（GM 侧字节单位，不是 0）；dstStride = 0（块间
// gap=0，HW 按 CeilAlign(blockLen, 32B) 自动放置下一块 → UB 行步长恰为 aLenUb）；
// padParams.isPad = false（行宽对齐由 HW 自动保证，A 方向 pad 不进 reduce 结果）。
// ---------------------------------------------------------------------------
template <typename DType>
__aicore__ inline void LpNormReduceGroupKernel<DType>::CopyInWorkspaceP2(int64_t aOff, int64_t aLen)
{
    DataCopyExtParams ext;
    ext.blockLen = static_cast<uint32_t>(aLen * static_cast<int64_t>(sizeof(float))); // valid 字节
    ext.blockCount = static_cast<uint16_t>(td_->rGroupCnt);                           // 全部 R 分组（不是 1）
    // workspace 行间 gap（字节）：3510 下 srcStride 为 int64_t 字段，int64 表达式直赋，
    // 保持 GM 侧字节 stride 全链 int64（与 base DoCopyInTile 的 srcStride 直赋一致）
    ext.srcStride = aTotal_ * static_cast<int64_t>(sizeof(float)) - aLen * static_cast<int64_t>(sizeof(float));
    ext.dstStride = 0; // 块间 gap=0，HW 按 CeilAlign(blockLen, 32B) 自动放置
    ext.rsv = 0;       // 必须显式填 0（datacopypad-rules）
    AscendC::DataCopyPadExtParams<float> padParams{/*isPad=*/false, /*leftPadding=*/0,
                                                   /*rightPadding=*/0, /*paddingValue=*/0.0f};
    DataCopyPad(preReduceResult_.template Get<float>(), wsGm_[aOff], ext, padParams);
}

// ---------------------------------------------------------------------------
// PostElewiseP2（Kernel.md §9.8）—— 复用 base PostElewiseVfImpl（Kernel.md §9.6
// 缩位 Cast：fp32 直通 / fp16 CAST_RINT + DIST_PACK_B32），树根 = cacheBuf[0]
// （cacheCount=1 退化、rootOff=0）。按 padded 整行处理（A 方向 garbage Cast 无害），
// CopyOut 只拷 valid。
// ---------------------------------------------------------------------------
template <typename DType>
__aicore__ inline void LpNormReduceGroupKernel<DType>::PostElewiseP2(int64_t aLen)
{
    const uint32_t laneN = static_cast<uint32_t>(Ops::Base::CeilAlign(static_cast<uint32_t>(aLen), UB_BLOCK_F32));
    __ubuf__ float* rootPtr = reinterpret_cast<__ubuf__ float*>(cacheBuf_.template Get<float>().GetPhyAddr());
    __ubuf__ DT* outPtr = reinterpret_cast<__ubuf__ DT*>(outBuf_.template Get<DT>().GetPhyAddr());
    const uint16_t repeatTime = static_cast<uint16_t>(Ops::Base::CeilDiv(laneN, static_cast<uint32_t>(REP_F32_U16)));
    asc_vf_call<PostElewiseVfImpl<DT>>(rootPtr, outPtr, laneN, repeatTime);
}

} // namespace NsLpNormReduce

#endif // LP_NORM_REDUCE_GROUP_H_
