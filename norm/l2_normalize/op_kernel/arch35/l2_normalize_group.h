/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software; you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// =============================================================================
// norm/l2_normalize/op_kernel/arch35/l2_normalize_group.h
// =============================================================================
//
// ROLE: Group 模板 kernel 类（TPL_SEL_2，isGroup=1 / isEmptyTensor=0）——
//   A 用不满核（aLoopCntTotal ≤ coreNum/2）且 R 有并行度（rLoopCntTotal ≥ 2）
//   时的 A×R 2D 分核三阶段实现（Host 侧 SetScheduleMode(1) 配套）。
// 三阶段数据流：
//   Phase 1（每核 1 个 A chunk × 1 段 R 分组，核号 = aChunkIdx×rGroupCnt + rChunkIdx）：
//     GM_x → CopyIn → CastSquareVf → pad 清零 → [尾块配对 Merge] → ReduceSum →
//     局部 rCount 二分缓存树 → cacheBuf[localRoot] → workspace partial 区第
//     rChunkIdx 行（fp32，跳过 PostElewise——partial 是中间量，eps 钳制只在
//     Phase 2 做一次）；
//   Phase 1→2：SyncAll() 全核同步；
//   Phase 2（RA mini-kernel，A 重切分 aUbFactorP2，kernel 侧现算不进 TilingData）：
//     ws partial 区 [rGroupCnt, aLen] → ReduceSum RA（dst=cacheBuf[0]）→
//     PostElewise（max(s,eps)→sqrt，eps 钳在平方和上）→ denom →
//     workspace denom 区本核槽位（padded 槽位布局，槽步长 slotStride）；
//   Phase 3（keepdims 广播除法，tail-A 沿用 Phase 2 自产自销；tail-R 在第二次
//     SyncAll 后按 A 槽位 × R chunk 分核，避免少量 A 槽位限制输出并行度）：
//     GM_x 二次读 → denom 广播物化（tail-A srcStride=-blockLen 广播搬入 B2 /
//     tail-R dense 装入 B3 + BroadcastDenomTailRVf 行常量）→ DivCastVf（fp16
//     含扩位/缩位 Cast）→ CopyOut y。
//
// Phase 3 的 A 子 tile 恒为 LastA 单轴连续段（§5.2 行扫描段分解）：段内 A-lane
//   序 ≡ flat-A 序 ≡ denom 槽位序，逐 lane 对齐；每 (段 × R chunk) tile 落于
//   B0/B1/B2 基址（单 tile ≤ preBufSize 构造性保证，G9 预算反解），段首偏移
//   segLaneOff 仅用于 denom 源定位（tail-R B3 源 / tail-A ws 源）。
//
// group 不消费 base 的全局二分树派生量（bisectionPos_/cacheCount_），Phase 1
//   按本核局部 rCount 现算树深与 localRoot（⛔ 禁用全局 cacheCount_）。
// =============================================================================

#ifndef OPS_NORM_L2_NORMALIZE_GROUP_H_
#define OPS_NORM_L2_NORMALIZE_GROUP_H_

#include "kernel_operator.h"            // Ascend C kernel framework
#include "adv_api/reduce/reduce.h"      // AscendC::ReduceSum (AR / RA)
#include "l2_normalize_tiling_struct.h" // L2NormalizeTilingData / MAX_PATTERN_RANK
#include "l2_normalize_base.h"          // Base 模板 kernel 类（继承复用）

namespace NsL2Normalize {

using namespace AscendC;

// Phase 3 LastA 行扫描段：槽位 flat-A 区间按
// LastA 轴行扫描，行内按 subMaxLane 切段；段 A bundle 恒单层 {LastA partial}。
struct ASeg {
    int64_t outerFixed[MAX_PATTERN_RANK] = {0}; // 外层 A 坐标（偶下标 < lastA）
    int64_t segStart = 0;                       // 行内 LastA 起点
    int64_t subLen = 0;                         // 段宽（valid lane 数）
    int64_t segLaneOff = 0;                     // 段首在槽内 flat-A 偏移（denom 源定位）
};

__aicore__ inline int64_t CeilAlignI64(int64_t a, int64_t b) { return (a + b - 1) / b * b; }

__aicore__ inline int64_t MinI64(int64_t a, int64_t b) { return (a < b) ? a : b; }

// ════════════════════════════════════════════════════════════════════════════
// L2NormalizeGroupKernel — Group 模板 kernel 类（TPL_SEL_2）
//   继承 L2NormalizeBaseKernel 复用 B0–B3+cacheBuf 物理槽、§9.1–§9.7 全部 VF
//   与 CopyIn/CopyOut 发射体；fp16 / fp32 各 1 binary（DTYPE_X 编译期实例化）。
// ════════════════════════════════════════════════════════════════════════════
template <typename DType>
class L2NormalizeGroupKernel : public L2NormalizeBaseKernel<DType> {
public:
    using Base = L2NormalizeBaseKernel<DType>;

    __aicore__ inline L2NormalizeGroupKernel() {}

    // Group 初始化：复用 Base::Init（GM/Buffer/event/outStride），再派生 aTotal
    // （A 轴总乘积 = workspace partial 区列数，G2 符号；wsGm_ 由 Base::Init 绑定）
    __aicore__ inline void InitGroup(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, const L2NormalizeTilingData* td,
                                     TPipe* pipe)
    {
        Base::Init(x, y, workspace, td, pipe);
        aTotal_ = 1;
        for (int32_t i = 0; i < Base::td_->axisNum; i += AXIS_INTERVAL) { // 偶下标 = A 轴
            aTotal_ *= Base::td_->axisShape[i];
        }
    }

    // tail-R 输出可消费其他核的 denom，因此所有核都必须参加第二次屏障。
    __aicore__ inline void ProcessGroup()
    {
        Phase1Process();
        SyncAll(); // SetScheduleMode(1) 配套
        Phase2Process();
        if (Base::isTailR_) {
            SyncAll();
        }
        Phase3Process();
    }

private:
    // ─── Phase 1：跨核 partial 归约（§5.3）───
    __aicore__ inline void Phase1Process();
    __aicore__ inline void DoOneAChunkGroup(int64_t outerGmOff, int64_t aLen, int64_t rStart, int64_t rEnd);
    __aicore__ inline void Phase1OutputToWorkspace(int64_t wsColOff, int64_t aLen, int64_t rChunkIdx, int64_t rCount);
    // ─── Phase 2：RA mini-kernel 二次归约得 denom（§5.3）───
    __aicore__ inline void Phase2Process();
    __aicore__ inline void Phase2CopyInPartial(int64_t aOff, int64_t aLen);
    // ─── Phase 3：keepdims 广播除法（§5.2–§5.3）───
    __aicore__ inline void Phase3Process();
    __aicore__ inline void UnravelAFlat(int64_t aFlat, int64_t aIdx[]) const;
    __aicore__ inline int64_t UnravelRLoopP3(int64_t rIdx, int64_t rOuterIdx[], int64_t& rChunkIdx, int64_t& rLen,
                                             int64_t rUbFactorP3) const;
    __aicore__ inline int32_t BuildUBAxesP3(const ASeg& seg, int64_t rLen, int64_t rUbFactorP3, UBAxisDesc out[]) const;
    __aicore__ inline void DoCopyInTileP3(const ASeg& seg, int64_t rOff, int64_t rLen, int64_t rUbFactorP3,
                                          AscendC::LocalTensor<DType>& preInLocal);
    __aicore__ inline void CopyInDenomSlotP3(int64_t wsSlotOff, int64_t slotStride);
    __aicore__ inline void CopyInDenomSegP3(int64_t wsSlotOff, const ASeg& seg, int64_t subLaneP3, int64_t rBundleP3);
    __aicore__ inline void DoCopyOutTileP3(const ASeg& seg, int64_t rOff, int64_t rLen, int64_t rUbFactorP3,
                                           AscendC::LocalTensor<DType>& yLocal);

    int64_t aTotal_ = 0; // ∏(所有 A 轴 axisShape) = partial 区列数
};

// ════════════════════════════════════════════════════════════════════════════
// Phase1Process（§5.3）：2D 坐标映射（G5）+ R 方向大小核式均匀分配（G6）+
//   aPerCore=1 单 A chunk + 局部 rCount 归约 + partial 写 workspace。
// ════════════════════════════════════════════════════════════════════════════
template <typename DType>
__aicore__ inline void L2NormalizeGroupKernel<DType>::Phase1Process()
{
    const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
    if (blockIdx >= static_cast<int64_t>(Base::td_->usedCoreNum)) {
        return; // idle 核早退（G4/G5）
    }

    const int64_t rOuter = Base::td_->rLoopCntTotal;
    const int64_t aChunkIdx = blockIdx / Base::td_->rGroupCnt; // G5: 核号 = aChunkIdx×rGroupCnt + rChunkIdx
    const int64_t rChunkIdx = blockIdx % Base::td_->rGroupCnt; //   = workspace partial 区行号

    // R 方向大小核式均匀分配（G6；范式 ⛔ 禁 CeilDiv 截断式：每组 ≥1 chunk 无空组）
    const int64_t rSmallGroupLoopCnt = rOuter / Base::td_->rGroupCnt;
    const int64_t rBigGroupCnt = rOuter % Base::td_->rGroupCnt;
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
    const int64_t rEnd = rStart + rCount;
    if (rStart >= rOuter) {
        return; // 防御性早退（理论上不可达）
    }

    // aPerCore=1：本核只处理 1 个 A chunk（G4）
    int64_t aIdx[MAX_PATTERN_RANK] = {0};
    int64_t aSplitChunkIdx = 0;
    Base::UnravelALoop(aChunkIdx, aIdx, aSplitChunkIdx);

    const int64_t aSplitAxisSize = Base::td_->axisShape[Base::td_->aSplitIdx];
    const int64_t aChunkStart = aSplitChunkIdx * Base::td_->aUbFactor;
    const int64_t aEndVal = aChunkStart + Base::td_->aUbFactor;
    const int64_t aLen = (aEndVal > aSplitAxisSize) ? (aSplitAxisSize - aChunkStart) // A 尾块 valid（G7）
                                                      :
                                                      Base::td_->aUbFactor;
    if (aLen <= 0) {
        return;
    }

    int64_t chunkGmOff = 0;
    int64_t chunkOutOff = 0;
    for (int32_t k = Base::td_->aSplitIdx - AXIS_INTERVAL; k >= 0; k -= AXIS_INTERVAL) {
        chunkGmOff += aIdx[k] * Base::td_->axisStride[k];
        chunkOutOff += aIdx[k] * Base::outStride_[k]; // flat-A 列偏移（Base::Init 派生）
    }
    chunkGmOff += aChunkStart * Base::td_->axisStride[Base::td_->aSplitIdx];
    chunkOutOff += aChunkStart * Base::outStride_[Base::td_->aSplitIdx];

    DoOneAChunkGroup(chunkGmOff, aLen, rStart, rEnd); // ── S1–S5：局部 s 就绪于 cacheBuf[localRoot] ──
    Phase1OutputToWorkspace(chunkOutOff, aLen, rChunkIdx, rCount); // ── S5a：partial 写 workspace ──
}

// ════════════════════════════════════════════════════════════════════════════
// DoOneAChunkGroup（§5.3）：Phase 1 单 A chunk 的 R 段归约——复用 Base 全部
//   子步骤，R 迭代区间限定 [rStart, rEnd)（局部 rCount 上的二分缓存树，
//   GetCacheID/DoCaching 序列与 Base 路径一致，保证确定性）。
// ════════════════════════════════════════════════════════════════════════════
template <typename DType>
__aicore__ inline void L2NormalizeGroupKernel<DType>::DoOneAChunkGroup(int64_t outerGmOff, int64_t aLen, int64_t rStart,
                                                                       int64_t rEnd)
{
    const int64_t rCount = rEnd - rStart;
    if (rCount <= 0) {
        return;
    }

    const int64_t bisectionPos = static_cast<int64_t>(Base::FindNearestPower2(static_cast<uint64_t>(rCount)));
    const int64_t bisectionTail = rCount - bisectionPos; // Phase A 主尾配对数（局部口径，⛔ 非全局）

    auto preInLocal = Base::preInBuf_.template Get<DType>();
    auto preResLocal = Base::preReduceResult_.template Get<float>();
    __ubuf__ DType* preIn = reinterpret_cast<__ubuf__ DType*>(preInLocal.GetPhyAddr());
    __ubuf__ float* preRes = reinterpret_cast<__ubuf__ float*>(preResLocal.GetPhyAddr());

    for (int64_t rIdx = 0; rIdx < bisectionPos; ++rIdx) {
        if (rIdx != 0) {
            WaitFlag<HardEvent::V_MTE2>(Base::evVtoMTE2_); // B0 WAR：上一轮 CastSquareVf 已读 B0
        }

        int64_t rOuterIdx[MAX_PATTERN_RANK] = {0};
        int64_t rChunkMain = 0;
        int64_t rLenMain = 0;
        const int64_t rOffMain = Base::UnravelRLoop(rStart + rIdx, rOuterIdx, rChunkMain, rLenMain);

        Base::DoCopyInTile(outerGmOff + rOffMain, aLen, rLenMain, preInLocal); // S1 主块 → B0（继承）
        SetFlag<HardEvent::MTE2_V>(Base::evMTE2toV_);
        WaitFlag<HardEvent::MTE2_V>(Base::evMTE2toV_); // B0 就绪

        Base::CastSquareVf(preIn, preRes); // S2+S3: x²（fp16: Cast↑+Mul 一条 VF）→ B1
        if (rLenMain < Base::td_->rUbFactor) {
            Base::ClearChunkExtensionVf(preRes, rLenMain); // S3a: ExtensionPad 清零（partial chunk）
        }
        if (Base::isTailR_) {
            Base::ClearInnerBurstTailPadVf(preRes, rLenMain); // S3a: BurstPad 清零（tail-R 非对齐）
        }

        if (rIdx < bisectionTail) { // Phase A：尾块配对（M = P + T）
            int64_t rOuterIdxTail[MAX_PATTERN_RANK] = {0};
            int64_t rChunkTail = 0;
            int64_t rLenTail = 0;
            const int64_t rOffTail = Base::UnravelRLoop(rStart + rIdx + bisectionPos, rOuterIdxTail, rChunkTail,
                                                        rLenTail);

            __ubuf__ float* preResTail = reinterpret_cast<__ubuf__ float*>(
                Base::preReduceResultTail_.template Get<float>().GetPhyAddr());

            SetFlag<HardEvent::V_MTE2>(Base::evVtoMTE2_);
            WaitFlag<HardEvent::V_MTE2>(Base::evVtoMTE2_); // B0 WAR：本轮 CastSquareVf 已读 B0
            Base::DoCopyInTile(outerGmOff + rOffTail, aLen, rLenTail, preInLocal); // S1 尾块（复用 B0）
            SetFlag<HardEvent::MTE2_V>(Base::evMTE2toV_);
            WaitFlag<HardEvent::MTE2_V>(Base::evMTE2toV_);
            Base::CastSquareVf(preIn, preResTail); // S2+S3 → B2
            if (rLenTail < Base::td_->rUbFactor) {
                Base::ClearChunkExtensionVf(preResTail, rLenTail);
            }
            if (Base::isTailR_) {
                Base::ClearInnerBurstTailPadVf(preResTail, rLenTail);
            }
            Base::MergeTmpBufVf(preRes, preResTail); // S3b: main ⊕ tail → main（B1）
        }

        // S4: ReduceSum（AR/RA 按 tail 运行时二选一；src=B1、sharedTmp=B2、
        //     dst=cacheBuf[cacheID 层]；srcShape 按 padded 值、srcInnerPad=true）
        const uint16_t cacheID = Base::GetCacheID(rIdx);
        const uint32_t laneA = static_cast<uint32_t>(Base::td_->aUbFactor * Base::td_->innerAProdAlign);
        const uint32_t levelStride = CeilAlignU32(laneA, UB_BLOCK_F32);
        const int32_t levelOff = static_cast<int32_t>(cacheID) * static_cast<int32_t>(levelStride);

        if (Base::isTailR_) {
            uint32_t srcShape[REDUCE_SHAPE_DIM] = {
                laneA, static_cast<uint32_t>(Base::td_->rUbFactorAlign * Base::td_->innerRProdAlign)};
            AscendC::ReduceSum<float, AscendC::Pattern::Reduce::AR, /*isReuseSource=*/true>(
                Base::cacheBuf_.template Get<float>()[levelOff], preResLocal,
                Base::preReduceResultTail_.template Get<uint8_t>(), srcShape, /*srcInnerPad=*/true);
        } else {
            uint32_t srcShape[REDUCE_SHAPE_DIM] = {
                static_cast<uint32_t>(Base::td_->rUbFactorAlign * Base::td_->innerRProdAlign), laneA};
            AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, /*isReuseSource=*/true>(
                Base::cacheBuf_.template Get<float>()[levelOff], preResLocal,
                Base::preReduceResultTail_.template Get<uint8_t>(), srcShape, /*srcInnerPad=*/true);
        }
        Base::DoCachingVf(cacheID); // S5: 二分缓存树正序吸收 + 覆盖写

        if (rIdx != bisectionPos - 1) {
            SetFlag<HardEvent::V_MTE2>(Base::evVtoMTE2_); // 为下一轮 WaitFlag 配对
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Phase1OutputToWorkspace（§5.3 / §5.4 三路径决策表）：按本核局部 rCount 现算
//   二分树根偏移（localRoot，⛔ 禁用全局 cacheCount_），cacheBuf 树根（partial
//   平方和）→ workspace partial 区第 rChunkIdx 行 [wsColOff, ...) 列（fp32；
//   范式 [7.5] 约束：sizeof(D_T) → sizeof(float)）。
// ════════════════════════════════════════════════════════════════════════════
template <typename DType>
__aicore__ inline void L2NormalizeGroupKernel<DType>::Phase1OutputToWorkspace(int64_t wsColOff, int64_t aLen,
                                                                              int64_t rChunkIdx, int64_t rCount)
{
    const int64_t bisectionPos = static_cast<int64_t>(Base::FindNearestPower2(static_cast<uint64_t>(rCount)));
    const int64_t cacheCount = static_cast<int64_t>(Base::CalLog2(static_cast<uint64_t>(bisectionPos))) + 1;
    const int64_t laneN = Base::td_->aUbFactor * Base::td_->innerAProdAlign;
    const int64_t levelStride = CeilAlignI64(laneN, static_cast<int64_t>(UB_BLOCK_F32));
    const int64_t rootOff = (cacheCount - 1) * levelStride; // localRoot（局部树深）

    auto cacheLocal = Base::cacheBuf_.template Get<float>();

    SetFlag<HardEvent::V_MTE3>(Base::evVtoMTE3_);
    WaitFlag<HardEvent::V_MTE3>(Base::evVtoMTE3_); // cacheBuf: V 写 → MTE3 读（兼 Phase 1 V 全量 drain）

    DataCopyExtParams ext;
    if (Base::isTailR_) { // 路径 1（tail-R dense，§5.4 决策表）
        int64_t innerAProd = 1;
        for (int32_t k = Base::td_->aSplitIdx + AXIS_INTERVAL; k <= Base::LastAAxis(); k += AXIS_INTERVAL) {
            innerAProd *= Base::td_->axisShape[k];
        }
        ext.blockLen = static_cast<uint32_t>(aLen * innerAProd * static_cast<int64_t>(sizeof(float)));
        ext.blockCount = 1;
        ext.srcStride = 0;
    } else {
        const int32_t lastA = Base::LastAAxis(); // 路径 2 / 3（tail-A，§5.4 决策表）
        const int64_t lastASize = Base::td_->axisShape[lastA];
        if (Base::td_->aSplitIdx == lastA) {
            ext.blockLen = static_cast<uint32_t>(aLen * static_cast<int64_t>(sizeof(float)));
            ext.blockCount = 1;
            ext.srcStride = 0;
        } else {
            int64_t innerAProd = 1;
            for (int32_t k = Base::td_->aSplitIdx + AXIS_INTERVAL; k <= lastA; k += AXIS_INTERVAL) {
                innerAProd *= Base::td_->axisShape[k];
            }
            ext.blockLen = static_cast<uint32_t>(lastASize * static_cast<int64_t>(sizeof(float)));
            ext.blockCount = static_cast<uint16_t>(aLen * innerAProd / lastASize);
            const int64_t bsElem = UB_BLOCK_BYTES / static_cast<int64_t>(sizeof(DType));
            const int64_t lastASizeAlign = CeilAlignI64(lastASize, bsElem);
            ext.srcStride = (lastASizeAlign - lastASize) * static_cast<int64_t>(sizeof(float)) / UB_BLOCK_BYTES;
        }
    }
    ext.dstStride = 0; // GM dense（partial 区行优先 [rGroupCnt, aTotal]）
    ext.rsv = 0;

    const int64_t wsOff = rChunkIdx * aTotal_ + wsColOff; // partial 区第 rChunkIdx 行本核列段
    DataCopyPad(Base::wsGm_[wsOff], cacheLocal[rootOff], ext);
}

// ════════════════════════════════════════════════════════════════════════════
// Phase2Process（§5.3）：RA mini-kernel——kernel 侧现算 A 重切分参数（G8，
//   ⚠ aUbFactorP2 ≠ TilingData 的 aUbFactor（Phase 1））+ 大小核均衡 + workspace
//   CopyIn + ReduceSum RA（R 全载无二分）+ PostElewise + denom 槽位写回。
//   与 host 侧 ComputeGroupP2 同式（workspace 布局一致性；B3 容量上界按
//   postBufSize/sizeof(float) 口径）。
// ════════════════════════════════════════════════════════════════════════════
template <typename DType>
__aicore__ inline void L2NormalizeGroupKernel<DType>::Phase2Process()
{
    const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());

    // ── 1) kernel 侧现算切分参数（G8）──
    const int64_t preInElems = Base::td_->preBufSize / static_cast<int64_t>(sizeof(float));
    constexpr int64_t bsFp32 = UB_BLOCK_BYTES / static_cast<int64_t>(sizeof(float)); // = 8
    constexpr int64_t bsElem = UB_BLOCK_BYTES / static_cast<int64_t>(sizeof(DType)); // fp32=8 / fp16=16

    int64_t aUbFactorP2 = preInElems / Base::td_->rGroupCnt; // floor：R 全载 rGroupCnt 行优先（tile 装进 B1）
    if (aUbFactorP2 >= bsFp32) {
        aUbFactorP2 = (aUbFactorP2 / bsFp32) * bsFp32; // ★ burst 尾轴对齐（8），小值不归零
    }
    aUbFactorP2 = MinI64(aUbFactorP2,
                         Base::td_->postBufSize / static_cast<int64_t>(sizeof(float))); // B3 fp32 容量
    aUbFactorP2 = MinI64(aUbFactorP2, aTotal_);        // 上界：aTotal（chunk 宽终值）
    aUbFactorP2 = (aUbFactorP2 > 1) ? aUbFactorP2 : 1; // 防御（与 host ComputeGroupP2 同式）
    const int64_t slotStride = Base::isTailR_ ? aUbFactorP2 : CeilAlignI64(aUbFactorP2, bsElem); // denom 槽位步长
    const int64_t aSplitChunkCntP2 = CeilDivI64(aTotal_, aUbFactorP2);

    // ── 2) 大小核均衡（G8；实际参与核数 usedCoreNumP2 可能 < usedCoreNum）──
    const int64_t aSmallCoreLoopCntP2 = aSplitChunkCntP2 / Base::td_->usedCoreNum;
    const int64_t aBigCoreCntP2 = aSplitChunkCntP2 % Base::td_->usedCoreNum;
    const int64_t aBigCoreLoopCntP2 = aSmallCoreLoopCntP2 + (aBigCoreCntP2 > 0 ? 1 : 0);
    const int64_t usedCoreNumP2 = (aSmallCoreLoopCntP2 > 0) ? Base::td_->usedCoreNum : aBigCoreCntP2;
    if (blockIdx >= usedCoreNumP2) {
        return; // 仅退出 Phase 2；tail-R 空闲核仍参加其后的屏障和 Phase 3
    }

    int64_t aLoopStart = 0;
    int64_t aLoopEnd = 0;
    if (blockIdx < aBigCoreCntP2) {
        aLoopStart = blockIdx * aBigCoreLoopCntP2;
        aLoopEnd = aLoopStart + aBigCoreLoopCntP2;
    } else {
        aLoopStart = aBigCoreCntP2 * aBigCoreLoopCntP2 + (blockIdx - aBigCoreCntP2) * aSmallCoreLoopCntP2;
        aLoopEnd = aLoopStart + aSmallCoreLoopCntP2;
    }

    for (int64_t aLoopIdx = aLoopStart; aLoopIdx < aLoopEnd; ++aLoopIdx) {
        if (aLoopIdx != aLoopStart) {                      // 槽位边界 WAR（§5.5）：B1（上块 ReduceSum
            WaitFlag<HardEvent::V_MTE2>(Base::evVtoMTE2_); //   V 读 → 本块 CopyIn MTE2 写）/ B3（上块
            WaitFlag<HardEvent::MTE3_V>(Base::evMTE3toV_); //   CopyOut MTE3 读 → 本块 PostElewise V 写）
        }
        const int64_t aOff = aLoopIdx * aUbFactorP2;
        int64_t aLen = aUbFactorP2;
        if (aOff + aLen > aTotal_) {
            aLen = aTotal_ - aOff; // A 尾块 valid（G8）
        }
        const int64_t aLenUb = CeilAlignI64(aLen, bsFp32); // UB padded 行宽

        // 2a) S2a: CopyIn workspace partial 区 → B1（fp32，复用 Phase 1 物理槽；§5.2）
        Phase2CopyInPartial(aOff, aLen);
        SetFlag<HardEvent::MTE2_V>(Base::evMTE2toV_);
        WaitFlag<HardEvent::MTE2_V>(Base::evMTE2toV_); // B1 就绪

        // 2b) S2b: ReduceSum RA——R 全载单 chunk，dst 写 cacheBuf[0]（无二分树层级）
        //     srcShape 必须用 aLenUb（padded）：ReduceSum 按 padded 行步长读。
        {
            auto cacheLocal = Base::cacheBuf_.template Get<float>();
            auto preReduceLocal = Base::preReduceResult_.template Get<float>();
            uint32_t srcShape[REDUCE_SHAPE_DIM] = {static_cast<uint32_t>(Base::td_->rGroupCnt),
                                                   static_cast<uint32_t>(aLenUb)};
            AscendC::ReduceSum<float, AscendC::Pattern::Reduce::RA, /*isReuseSource=*/true>(
                cacheLocal, preReduceLocal, Base::preReduceResultTail_.template Get<uint8_t>(), srcShape,
                /*srcInnerPad=*/true);
        }

        // 2c) S2c: PostElewise（复用 Base VF，树根 = cacheBuf[0]）：
        //     max(s, eps) → sqrt → denom（B3）；eps 经 TilingData 传入（禁硬编码）
        {
            __ubuf__ float* rootPtr = reinterpret_cast<__ubuf__ float*>(
                Base::cacheBuf_.template Get<float>().GetPhyAddr());
            __ubuf__ float* denomPtr = reinterpret_cast<__ubuf__ float*>(
                Base::postReduceResult_.template Get<float>().GetPhyAddr());
            const uint16_t repeatTime = static_cast<uint16_t>(
                CeilDivU32(static_cast<uint32_t>(aLenUb), static_cast<uint32_t>(REP_F32_U16)));
            asc_vf_call<PostElewiseVfImpl>(rootPtr, denomPtr, static_cast<uint32_t>(aLenUb), Base::td_->eps,
                                           repeatTime);
        }

        // 2d) S2d: denom 中转写 workspace denom 区本核槽位（denom 区起点 = rGroupCnt × aTotal）：
        //     padded 槽位布局（槽步长 slotStride ≥ chunk 宽 aUbFactorP2），
        //     整 padded 行直写（B3 尾部 stale garbage 落入槽 pad，不被 Phase 3 valid 段消费）
        SetFlag<HardEvent::V_MTE3>(Base::evVtoMTE3_);
        WaitFlag<HardEvent::V_MTE3>(Base::evVtoMTE3_); // B3: V 写 → MTE3 读
        {
            auto denomLocal = Base::postReduceResult_.template Get<float>();
            DataCopyExtParams ext;
            ext.blockLen = static_cast<uint32_t>(slotStride * static_cast<int64_t>(sizeof(float)));
            ext.blockCount = 1;
            ext.srcStride = 0;
            ext.dstStride = 0;
            ext.rsv = 0;
            DataCopyPad(Base::wsGm_[Base::td_->rGroupCnt * aTotal_ + aLoopIdx * slotStride], denomLocal, ext);
        }
        if (aLoopIdx != aLoopEnd - 1) {
            SetFlag<HardEvent::V_MTE2>(Base::evVtoMTE2_); // 为下一块 B1 WAR 配对（§5.5）
            SetFlag<HardEvent::MTE3_V>(Base::evMTE3toV_); // 为下一块 B3 WAR 配对
        }
    }
    SetFlag<HardEvent::MTE3_MTE2>(Base::evMTE3toMTE2_); // Phase 2→3 边界：ws 写→读序 + tail-R B3 覆写
    if (Base::isTailR_) {
        // 在生产核消费本地事件；空闲核不能等待从未发出的事件。
        WaitFlag<HardEvent::MTE3_MTE2>(Base::evMTE3toMTE2_);
    }
}

// S2a: workspace partial 区 [rGroupCnt, aTotal] fp32 dense 行优先 → B1（复用 Phase 1
//   物理槽）。行 g ∈ [0, rGroupCnt) = R 分组，列 = flat-A；本核读列段 [aOff, aOff+aLen)。
template <typename DType>
__aicore__ inline void L2NormalizeGroupKernel<DType>::Phase2CopyInPartial(int64_t aOff, int64_t aLen)
{
    auto preReduceLocal = Base::preReduceResult_.template Get<float>();
    DataCopyExtParams ext;
    ext.blockLen = static_cast<uint32_t>(aLen * static_cast<int64_t>(sizeof(float))); // valid 字节
    ext.blockCount = static_cast<uint16_t>(Base::td_->rGroupCnt);                     // 全部 R 分组行
    ext.srcStride = aTotal_ * static_cast<int64_t>(sizeof(float)) -
                    static_cast<int64_t>(ext.blockLen); // GM 行间 gap（byte）
    ext.dstStride = 0;                                  // UB 32B 块紧排 → 行步长 aLenUb×4B
    ext.rsv = 0;
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
    DataCopyPad(preReduceLocal, Base::wsGm_[aOff], ext, padParams);
}

// ════════════════════════════════════════════════════════════════════════════
// Phase3Process（§5.3）：keepdims 广播除法——denom 槽位布局沿用 Phase 2，
//   tail-R 按槽位 × R chunk 分核；tail-A 保持自产自销。A 子 tile = LastA 行扫描段。
//   每 (段 × R chunk) tile 落 B0/B1/B2 基址（单 tile ≤ preBufSize 构造性保证），
//   x tile / denom_bcast tile / y tile 三者布局逐 lane 对齐（tail-R [A,R] /
//   tail-A [R,A]）；段首偏移 segLaneOff 仅用于 denom 源定位。
// ════════════════════════════════════════════════════════════════════════════
template <typename DType>
__aicore__ inline void L2NormalizeGroupKernel<DType>::Phase3Process()
{
    const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());

    // ── 1) 槽位布局与 Phase 2 完全同式；tail-R 可重分配槽位消费者──
    const int64_t preInElems = Base::td_->preBufSize / static_cast<int64_t>(sizeof(float));
    constexpr int64_t bsFp32 = UB_BLOCK_BYTES / static_cast<int64_t>(sizeof(float));
    constexpr int64_t bsElem = UB_BLOCK_BYTES / static_cast<int64_t>(sizeof(DType));

    int64_t aUbFactorP2 = preInElems / Base::td_->rGroupCnt;
    if (aUbFactorP2 >= bsFp32) {
        aUbFactorP2 = (aUbFactorP2 / bsFp32) * bsFp32;
    }
    aUbFactorP2 = MinI64(aUbFactorP2, Base::td_->postBufSize / static_cast<int64_t>(sizeof(float)));
    aUbFactorP2 = MinI64(aUbFactorP2, aTotal_);
    aUbFactorP2 = (aUbFactorP2 > 1) ? aUbFactorP2 : 1;
    const int64_t slotStride = Base::isTailR_ ? aUbFactorP2 :
                                                CeilAlignI64(aUbFactorP2, bsElem); // tail-A 槽宽按 bsElem 对齐
    const int64_t aSplitChunkCntP2 = CeilDivI64(aTotal_, aUbFactorP2);
    const int64_t aSmallCoreLoopCntP2 = aSplitChunkCntP2 / Base::td_->usedCoreNum;
    const int64_t aBigCoreCntP2 = aSplitChunkCntP2 % Base::td_->usedCoreNum;
    const int64_t aBigCoreLoopCntP2 = aSmallCoreLoopCntP2 + (aBigCoreCntP2 > 0 ? 1 : 0);
    const int64_t usedCoreNumP2 = (aSmallCoreLoopCntP2 > 0) ? Base::td_->usedCoreNum : aBigCoreCntP2;
    if (!Base::isTailR_ && blockIdx >= usedCoreNumP2) {
        return; // 与 Phase 2 同界早退（自产自销，无跨核依赖）
    }
    int64_t aLoopStart = 0;
    int64_t aLoopEnd = 0;
    if (blockIdx < aBigCoreCntP2) {
        aLoopStart = blockIdx * aBigCoreLoopCntP2;
        aLoopEnd = aLoopStart + aBigCoreLoopCntP2;
    } else {
        aLoopStart = aBigCoreCntP2 * aBigCoreLoopCntP2 + (blockIdx - aBigCoreCntP2) * aSmallCoreLoopCntP2;
        aLoopEnd = aLoopStart + aSmallCoreLoopCntP2;
    }
    int64_t rLoopStart = 0;
    int64_t rLoopStep = 1;
    if (Base::isTailR_ && aSplitChunkCntP2 < Base::td_->usedCoreNum) {
        // 核号 = rRank * 槽位数 + slot。每个槽的核数允许相差 1。
        // 同槽各核只写互不重叠的 R chunk，分母及算术顺序不变。
        aLoopStart = blockIdx % aSplitChunkCntP2;
        aLoopEnd = aLoopStart + 1;
        rLoopStart = blockIdx / aSplitChunkCntP2;
        rLoopStep = CeilDivI64(Base::td_->usedCoreNum - aLoopStart, aSplitChunkCntP2);
    }

    // ── 2) R 切分独立现算（G9；Phase 1 的 rUbFactor 仅服务 reduce，不约束除法遍）──
    const int64_t lastASize = Base::td_->axisShape[Base::LastAAxis()];
    const int64_t lastR = Base::LastRAxis();
    const bool splitIsLastR = Base::td_->rSplitIdx == lastR;
    int64_t subMaxLane = MinI64(lastASize, slotStride); // 单段 A 宽上界
    // 段宽还须受 B0/B1/B2 tile 预算约束：单段 [subMaxLane × 最小 R bundle] 须装进 preInElems，
    // 否则 budget3=0 → rUbFactorP3=0 → rChunkCntP3 除零（Phase 3 整体退化）。
    // 最小 R bundle：splitIsLastR 时 chunk 轴即 UB 最内层、须 ≥ bsElem；否则内层 R 整根
    // （innerRProdAlign 已含 lastR 的 CeilAlign，32B 对齐构造性成立）。
    const int64_t minBundle = splitIsLastR ? bsElem : Base::td_->innerRProdAlign;
    const int64_t laneCap = preInElems / minBundle; // ≥ 1（preBufSize ≥ aUnit×rPadded ≥ minBundle）
    if (Base::isTailR_) {
        subMaxLane = MinI64(subMaxLane, (laneCap > 1) ? laneCap : 1);
    } else {
        // tail-A 段宽按 bsElem padded 后生效，须 CeilAlign(subMaxLane, bsElem) ≤ laneCap
        const int64_t capA = (laneCap / bsElem) * bsElem;
        subMaxLane = MinI64(subMaxLane, (capA > 1) ? capA : 1);
    }
    const int64_t laneBudget = Base::isTailR_ ? subMaxLane :
                                                CeilAlignI64(subMaxLane, bsElem);   // tail-A 段内层 padded 口径
    const int64_t budget3 = preInElems / (laneBudget * Base::td_->innerRProdAlign); // B2（fp32）binding 预算
    const int64_t rAxisSize = Base::td_->axisShape[Base::td_->rSplitIdx];
    int64_t rUbFactorP3 = MinI64(budget3, rAxisSize);
    if (Base::isTailR_ && splitIsLastR) { // chunk 轴即 UB 最内层（G9 对齐纪律；与 host isBurstTailR 同口径）
        if (rUbFactorP3 == rAxisSize && rAxisSize % bsElem != 0) {
            rUbFactorP3 = (budget3 / bsElem) * bsElem; // 整轴全载但非对齐 → 退回多 chunk
        } else if (rUbFactorP3 < rAxisSize) {
            rUbFactorP3 = (rUbFactorP3 / bsElem) * bsElem; // 多 chunk 对齐（构造性 ≥ bsElem）
        }
    } // tail-R 且 rSplitIdx!=lastR：chunk 轴非 UB 最内层，rBundleP3×4B 已由 innerRProdAlign 保证 32B
      // 对齐，无须对 rUbFactorP3 施加 bsElem 对齐（强加会把 budget3<bsElem 的形态对齐成 0 → 除零）
    const int64_t rBundleP3 = rUbFactorP3 * Base::td_->innerRProdAlign; // tile R-lane 宽（全 chunk 统一）
    int64_t rChunkCntP3 = CeilDivI64(rAxisSize, rUbFactorP3);
    for (int32_t i = Base::td_->rSplitIdx - AXIS_INTERVAL; i >= 1; i -= AXIS_INTERVAL) {
        rChunkCntP3 *= Base::td_->axisShape[i]; // × ∏ 外层 R 轴
    }

    auto xLocal = Base::preInBuf_.template Get<DType>();
    auto yLocal = Base::preReduceResult_.template Get<DType>(); // B1 复用为 y（§4 复用关系）
    __ubuf__ DType* xPtr = reinterpret_cast<__ubuf__ DType*>(xLocal.GetPhyAddr());
    __ubuf__ DType* yPtr = reinterpret_cast<__ubuf__ DType*>(yLocal.GetPhyAddr());
    __ubuf__ float* denomPtr = reinterpret_cast<__ubuf__ float*>(
        Base::postReduceResult_.template Get<float>().GetPhyAddr());
    __ubuf__ float* bcastPtr = reinterpret_cast<__ubuf__ float*>(
        Base::preReduceResultTail_.template Get<float>().GetPhyAddr());
    const uint16_t repPerRow = static_cast<uint16_t>(
        CeilDivU32(static_cast<uint32_t>(rBundleP3), static_cast<uint32_t>(REP_F32_U16)));

    bool firstMte2 = !Base::isTailR_; // tail-R 已经在生产核消费事件并完成全核屏障
    bool pendingV2M = false;          // 有未消费的 Set<V_MTE2>（上轮末尾配对发出）
    bool pendingM3V = false;          // 有未消费的 Set<MTE3_V>（上轮末尾配对发出）
    const int32_t lastAIdx = Base::LastAAxis();
    for (int64_t aLoopIdx = aLoopStart; aLoopIdx < aLoopEnd; ++aLoopIdx) {
        const int64_t aOff = aLoopIdx * aUbFactorP2;
        int64_t aLen = aUbFactorP2;
        if (aOff + aLen > aTotal_) {
            aLen = aTotal_ - aOff; // 尾槽 valid
        }
        const int64_t wsSlot = Base::td_->rGroupCnt * aTotal_ + aLoopIdx * slotStride; // denom 区本核槽位基址

        if (Base::isTailR_) {
            // S9a · 每槽一次 dense 装载：ws 槽位 → B3（MTE2）
            if (firstMte2) {
                WaitFlag<HardEvent::MTE3_MTE2>(Base::evMTE3toMTE2_); // Phase 2→3 边界（§5.5）
                firstMte2 = false;
            } else if (pendingV2M) {
                WaitFlag<HardEvent::V_MTE2>(Base::evVtoMTE2_); // B3+B0 双 WAR（上槽 Broadcast/DivCast
                pendingV2M = false;                            //   V 读 → 本槽 S9a/S8 MTE2 覆写）
            }
            CopyInDenomSlotP3(wsSlot, slotStride);
            SetFlag<HardEvent::MTE2_V>(Base::evMTE2toV_);
            WaitFlag<HardEvent::MTE2_V>(Base::evMTE2toV_); // B3 就绪（自配对）
        }

        // ── 行扫描（§5.2 DecomposeLastARows 算法流式展开：行 = 外层 A 坐标固定 +
        //    LastA 连续区间；行内按 subMaxLane 切段；段数无静态上界，故不物化段列表）──
        int64_t cur = aOff;
        const int64_t segEnd = aOff + aLen;
        int64_t segLane = 0;
        while (cur < segEnd) {
            int64_t aIdx[MAX_PATTERN_RANK] = {0};
            UnravelAFlat(cur, aIdx); // flat-A → A 轴坐标（混合进制，LastA 最内）
            const int64_t rowRemain = lastASize - aIdx[lastAIdx];
            const int64_t rowLen = MinI64(segEnd - cur, rowRemain);
            int64_t inner = 0;
            while (inner < rowLen) {
                const int64_t subLen = MinI64(subMaxLane, rowLen - inner);
                ASeg seg; // {outerFixed, segStart, subLen, segLaneOff}
                seg.segStart = aIdx[lastAIdx] + inner;
                seg.subLen = subLen;
                seg.segLaneOff = segLane;
                for (int32_t ax = lastAIdx - AXIS_INTERVAL; ax >= 0; ax -= AXIS_INTERVAL) {
                    seg.outerFixed[ax] = aIdx[ax];
                }
                const int64_t subLaneP3 = Base::isTailR_ ?
                                              seg.subLen :
                                              CeilAlignI64(seg.subLen, bsElem); // tail-A 段内层 padded 宽（G9）
                const uint32_t totalElems = static_cast<uint32_t>(Base::isTailR_ ? seg.subLen * rBundleP3 :
                                                                                   rBundleP3 * subLaneP3);
                const uint16_t repTime = static_cast<uint16_t>(
                    CeilDivU32(totalElems, static_cast<uint32_t>(REP_F32_U16)));

                for (int64_t j = rLoopStart; j < rChunkCntP3; j += rLoopStep) {
                    int64_t rOuterIdx[MAX_PATTERN_RANK] = {0};
                    int64_t rChunkIdx = 0;
                    int64_t rLen = 0;
                    const int64_t rOff = UnravelRLoopP3(j, rOuterIdx, rChunkIdx, rLen, rUbFactorP3);
                    const bool isLast = (aLoopIdx == aLoopEnd - 1) && (cur + rowLen >= segEnd) &&
                                        (inner + subLen >= rowLen) && (j + rLoopStep >= rChunkCntP3);

                    if (pendingV2M) {
                        WaitFlag<HardEvent::V_MTE2>(Base::evVtoMTE2_); // B0/B2 WAR（上轮 DivCast/Broadcast V 读）
                        pendingV2M = false;
                    } // 全局首个 S8 无 WAR（Phase 1 V drain，§5.5）
                    if (firstMte2) { // tail-A：ws 写→读边界（首段 S9 前）
                        WaitFlag<HardEvent::MTE3_MTE2>(Base::evMTE3toMTE2_);
                        firstMte2 = false;
                    }
                    DoCopyInTileP3(seg, rOff, rLen, rUbFactorP3, xLocal); // S8: x GM 二次读 → B0（段）
                    if (!Base::isTailR_) {
                        CopyInDenomSegP3(wsSlot, seg, subLaneP3, rBundleP3); // S9 tail-A: 广播 → B2（段）
                    }
                    SetFlag<HardEvent::MTE2_V>(Base::evMTE2toV_);
                    WaitFlag<HardEvent::MTE2_V>(Base::evMTE2toV_); // B0/B2(或 B3) 就绪
                    if (Base::isTailR_) {
                        // S9 tail-R: B3[segLaneOff, +subLen) → B2 行常量（共享 VF；
                        // 源偏移 = 段首槽内 flat-A 偏移）
                        asc_vf_call<BroadcastDenomTailRVfImpl>(denomPtr + seg.segLaneOff, bcastPtr,
                                                               static_cast<uint32_t>(seg.subLen),
                                                               static_cast<uint32_t>(rBundleP3), repPerRow);
                    }
                    if (pendingM3V) {
                        WaitFlag<HardEvent::MTE3_V>(Base::evMTE3toV_); // B1(y) WAR：上轮 CopyOut 已读
                        pendingM3V = false;
                    } // 全局首个 DivCast 无 WAR（B1 上一访问为 Phase 2 V 读）
                    // S10+S11: y = x / denom_bcast（fp16: Cast↑+Div+Cast↓ 一条 VF；fp32: Div 链长 1）→ B1
                    constexpr bool kFp16PrecG = !AscendC::IsSameType<DType, float>::value;
                    asc_vf_call<DivCastVfImpl<DType, kFp16PrecG>>(xPtr, bcastPtr, yPtr, totalElems, repTime);
                    SetFlag<HardEvent::V_MTE3>(Base::evVtoMTE3_);
                    WaitFlag<HardEvent::V_MTE3>(Base::evVtoMTE3_);         // B1: V 写 → MTE3 读
                    DoCopyOutTileP3(seg, rOff, rLen, rUbFactorP3, yLocal); // S12: y → GM（段，§5.4）
                    if (!isLast) {
                        SetFlag<HardEvent::MTE3_V>(Base::evMTE3toV_); // 为下一轮 DivCast 覆写 B1 配对
                        SetFlag<HardEvent::V_MTE2>(Base::evVtoMTE2_); // 为下一轮 S8/S9 覆写 B0/B2(或 B3) 配对
                        pendingM3V = true;
                        pendingV2M = true;
                    }
                }
                inner += subLen;
                segLane += subLen;
            }
            cur += rowLen;
        }
    }
}

// UnravelAFlat: flat-A 下标 → A 轴坐标（混合进制：偶下标 0,2,…,lastA 为 digit，
//   最外 → 最内，radix = axisShape[该轴]；与 Base::Init 的 outStride_ 行主序一致）
template <typename DType>
__aicore__ inline void L2NormalizeGroupKernel<DType>::UnravelAFlat(int64_t aFlat, int64_t aIdx[]) const
{
    int64_t cur = aFlat;
    for (int32_t i = Base::LastAAxis(); i >= 0; i -= AXIS_INTERVAL) {
        aIdx[i] = cur % Base::td_->axisShape[i];
        cur /= Base::td_->axisShape[i];
    }
}

// UnravelRLoopP3: Phase 3 专用 R 解码（chunk 宽 rUbFactorP3 ≠ Phase 1 的 rUbFactor；
//   算法同 base §5.3 UnravelRLoop：rIdx → (外层 R 坐标, rSplitIdx chunk, rLen)，
//   返回 R 侧 GM 偏移）
template <typename DType>
__aicore__ inline int64_t L2NormalizeGroupKernel<DType>::UnravelRLoopP3(int64_t rIdx, int64_t rOuterIdx[],
                                                                        int64_t& rChunkIdx, int64_t& rLen,
                                                                        int64_t rUbFactorP3) const
{
    const int64_t rChunksOnSplit = CeilDivI64(Base::td_->axisShape[Base::td_->rSplitIdx], rUbFactorP3);
    rChunkIdx = rIdx % rChunksOnSplit;
    int64_t cur = rIdx / rChunksOnSplit;
    int64_t gmOff = 0;
    for (int32_t i = Base::td_->rSplitIdx - AXIS_INTERVAL; i >= 1; i -= AXIS_INTERVAL) { // 外层 R 轴
        rOuterIdx[i] = cur % Base::td_->axisShape[i];
        cur /= Base::td_->axisShape[i];
        gmOff += rOuterIdx[i] * Base::td_->axisStride[i];
    }
    const int64_t start = rChunkIdx * rUbFactorP3;
    rLen = (start + rUbFactorP3 > Base::td_->axisShape[Base::td_->rSplitIdx]) ?
               (Base::td_->axisShape[Base::td_->rSplitIdx] - start) :
               rUbFactorP3; // R 尾 chunk valid
    return gmOff + start * Base::td_->axisStride[Base::td_->rSplitIdx];
}

// BuildUBAxesP3（§5.2）：单段轴描述——A bundle 恒单层 = LastA 段（行扫描纪律），
//   R 侧 chunk 宽 = rUbFactorP3（非 Phase 1 的 rUbFactor）。
//   tail-R → 内 bundle = R（最内 R CeilAlign + … + rSplitIdx chunk）、外 bundle =
//   A 段（无 pad）；tail-A → 内 bundle = A 段（段级 CeilAlign(bsElem) 尾部 pad）、
//   外 bundle = R（最内 R 整根 + … + rSplitIdx chunk）。
template <typename DType>
__aicore__ inline int32_t L2NormalizeGroupKernel<DType>::BuildUBAxesP3(const ASeg& seg, int64_t rLen,
                                                                       int64_t rUbFactorP3, UBAxisDesc out[]) const
{
    int32_t k = 0;
    const int64_t bsElem = static_cast<int64_t>(UB_BLOCK_BYTES) / static_cast<int64_t>(sizeof(DType));
    const int32_t lastA = Base::LastAAxis();
    const int32_t lastR = Base::LastRAxis();

    if (Base::isTailR_) { // 内 bundle = R（从最内 R 到 rSplitIdx）
        for (int32_t i = lastR; i >= Base::td_->rSplitIdx; i -= AXIS_INTERVAL) {
            int64_t actual = 0;
            int64_t padded = 0;
            if (i == Base::td_->rSplitIdx) {
                actual = rLen;
                padded = rUbFactorP3; // chunk 轴（G9 对齐纪律）
            } else {
                actual = Base::td_->axisShape[i]; // 内层 R 整根
                padded = (i == lastR) ? CeilAlignI64(actual, bsElem) : actual;
            }
            out[k++] = {i, actual, padded, Base::td_->axisStride[i]};
        }
        out[k++] = {lastA, seg.subLen, seg.subLen, Base::td_->axisStride[lastA]}; // 外 bundle = A（单层，无 pad）
    } else { // 内 bundle = A（单层，段级尾部 pad）
        const int64_t subPadded = CeilAlignI64(seg.subLen, bsElem);
        out[k++] = {lastA, seg.subLen, subPadded, Base::td_->axisStride[lastA]};
        for (int32_t i = lastR; i >= Base::td_->rSplitIdx; i -= AXIS_INTERVAL) { // 外 bundle = R（内→外）
            int64_t actual = 0;
            int64_t padded = 0;
            if (i == Base::td_->rSplitIdx) {
                actual = rLen;
                padded = rUbFactorP3; // chunk 轴（外层无对齐要求）
            } else {
                actual = padded = Base::td_->axisShape[i]; // R 整根
            }
            out[k++] = {i, actual, padded, Base::td_->axisStride[i]};
        }
    }
    return k;
}

// DoCopyInTileP3（§5.2）：S8 · GM_x → B0（单段发射——段由调用侧行扫描流式产出）。
//   段 GM 基址 = rOff（R 侧偏移）+ segStart×axisStride[lastA] + Σ outerFixed×axisStride；
//   tile 落 B0 基址（单 tile ≤ preBufSize，G9 预算构造性保证），经 Base::EmitTileCopyIn
//   继承发射（K 分级同 base §5.2）。
template <typename DType>
__aicore__ inline void L2NormalizeGroupKernel<DType>::DoCopyInTileP3(const ASeg& seg, int64_t rOff, int64_t rLen,
                                                                     int64_t rUbFactorP3,
                                                                     AscendC::LocalTensor<DType>& preInLocal)
{
    UBAxisDesc ubAxes[MAX_PATTERN_RANK];
    const int32_t K = BuildUBAxesP3(seg, rLen, rUbFactorP3, ubAxes);
    int64_t segGmOff = rOff + seg.segStart * Base::td_->axisStride[Base::LastAAxis()];
    for (int32_t ax = Base::LastAAxis() - AXIS_INTERVAL; ax >= 0; ax -= AXIS_INTERVAL) {
        segGmOff += seg.outerFixed[ax] * Base::td_->axisStride[ax];
    }
    Base::EmitTileCopyIn(segGmOff, ubAxes, K, preInLocal);
}

// CopyInDenomSlotP3（§5.2）：S9 · tail-R 每槽一次 dense 装载：ws denom 区本核
//   槽位 → B3（blockCount=1, blockLen=slotStride×4B；行读 ⊂ 槽内恒安全）。
template <typename DType>
__aicore__ inline void L2NormalizeGroupKernel<DType>::CopyInDenomSlotP3(int64_t wsSlotOff, int64_t slotStride)
{
    DataCopyExtParams ext{1, static_cast<uint32_t>(slotStride * static_cast<int64_t>(sizeof(float))), 0, 0, 0};
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
    DataCopyPad(Base::postReduceResult_.template Get<float>(), Base::wsGm_[wsSlotOff], ext, padParams);
}

// CopyInDenomSegP3（§5.2）：S9 · tail-A 每段一次 denom 广播搬入：ws 槽位段 →
//   B2 [rBundleP3, subLaneP3] 行复制。srcStride 为 GM 侧 gap 语义（datacopypad-rules
//   §2：stride=0 是 dense 紧排而非重读），重读同一源行须取 srcStride = −blockLen
//   （GM 侧 byte 单位支持负值，同 base CopyInDenom tail-A 分支）；行宽 subLaneP3×4B
//   恒 32B 对齐（段级 CeilAlign(bsElem)）；blockCount ∈ [1,4095]，rBundleP3 > 4095
//   分段、段内语义不变。段源偏移 = 槽基址 + segLaneOff（段首 flat-A 偏移——与 B0
//   段 A-lane 的 flat 序逐 lane 对齐，⚠ 非 LastA 坐标 segStart：segStart 只用于
//   x/y 的 GM 寻址）。GM 读越界兜底：末段读 [wsSlot+segLaneOff, +subLaneP3) 可越
//   槽尾至多 (bsElem−1) 个 fp32——由 denom 区每槽 pad 余量构造性覆盖（§8），读到的
//   pad garbage 落入 B2 garbage lane，不被 valid 消费。
template <typename DType>
__aicore__ inline void L2NormalizeGroupKernel<DType>::CopyInDenomSegP3(int64_t wsSlotOff, const ASeg& seg,
                                                                       int64_t subLaneP3, int64_t rBundleP3)
{
    const int64_t segCnt = CeilDivI64(rBundleP3, static_cast<int64_t>(BRC_BLOCKCNT_LIMIT));
    const int64_t segRows = CeilDivI64(rBundleP3, segCnt);
    auto bcastLocal = Base::preReduceResultTail_.template Get<float>();
    const int64_t srcOff = wsSlotOff + seg.segLaneOff; // 段首 flat-A 偏移（槽内）
    const int64_t blockLen = subLaneP3 * static_cast<int64_t>(sizeof(float));
    DataCopyPadExtParams<float> padParams{false, 0, 0, 0.0f};
    for (int64_t seg2 = 0; seg2 < segCnt; ++seg2) {
        const int64_t rows = MinI64(segRows, rBundleP3 - seg2 * segRows);
        DataCopyExtParams ext{static_cast<uint16_t>(rows), static_cast<uint32_t>(blockLen), -blockLen, 0, 0};
        DataCopyPad(bcastLocal[seg2 * segRows * subLaneP3], Base::wsGm_[srcOff], ext, padParams);
    }
}

// DoCopyOutTileP3（§5.4）：S12 · B1(y) → GM_y（§5.2 DoCopyInTileP3 的镜像，单段
//   发射）。y 与 x 同 shape 同布局，GM 偏移/步长直接用 td_->axisStride；MTE3 方向
//   stride 单位 src=UB datablock(32B) / dst=GM byte；按 valid 段拷出（pad/garbage
//   不出有效段）；tile 落 B1 基址，经 Base::EmitTileCopyOut 继承发射。
template <typename DType>
__aicore__ inline void L2NormalizeGroupKernel<DType>::DoCopyOutTileP3(const ASeg& seg, int64_t rOff, int64_t rLen,
                                                                      int64_t rUbFactorP3,
                                                                      AscendC::LocalTensor<DType>& yLocal)
{
    UBAxisDesc ubAxes[MAX_PATTERN_RANK];
    const int32_t K = BuildUBAxesP3(seg, rLen, rUbFactorP3, ubAxes);
    int64_t segGmOff = rOff + seg.segStart * Base::td_->axisStride[Base::LastAAxis()];
    for (int32_t ax = Base::LastAAxis() - AXIS_INTERVAL; ax >= 0; ax -= AXIS_INTERVAL) {
        segGmOff += seg.outerFixed[ax] * Base::td_->axisStride[ax];
    }
    Base::EmitTileCopyOut(segGmOff, ubAxes, K, yLocal);
}

} // namespace NsL2Normalize

#endif // OPS_NORM_L2_NORMALIZE_GROUP_H_
