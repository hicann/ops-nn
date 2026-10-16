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
 * \file bn3_d_training_update_stat.h
 * \brief 相 S:通道统计单元。
 */

#ifndef BN3_D_TRAINING_UPDATE_STAT_H
#define BN3_D_TRAINING_UPDATE_STAT_H

#include <cstdint>
#include "kernel_operator.h"
#include "bn3_d_training_update_common.h"
#include "bn3_d_training_update_tiling_data.h"

namespace Bn3dTrainingUpdateOps {

// 相 S 单元的几何与标量参数(channel-first / channel-last 两份 TilingData 的同名字段适配)。
struct Bn3dTrainingUpdateStatDesc {
    int64_t channel;
    int64_t statUnitCount;
    int64_t statChunk;
    int64_t baseStatPerCore;
    int64_t extraStatCount;
    float numRec;
    float besselScaler;
    float factor;
    float oneMinusFactor;
    float epsilon;
};

// 输入 staging 槽位(与 VFFoldChunk 的入参次序一致)。
constexpr int64_t kStatInSum = 0;
constexpr int64_t kStatInSquareSum = 1;
constexpr int64_t kStatInScale = 2;
constexpr int64_t kStatInOffset = 3;
constexpr int64_t kStatInMean = 4;
constexpr int64_t kStatInVariance = 5;
// 结果槽位:batchMean / saveVar 由折叠产出;meanOut / varOut 由 EMA 产出。
// 折叠的 mult / offset(直通)出参复用 meanOut / varOut 槽(内容不被读取),故只需 4 个结果槽。
constexpr int64_t kStatOutBatchMean = 0;
constexpr int64_t kStatOutSaveVar = 1;
constexpr int64_t kStatOutMeanOut = 2;
constexpr int64_t kStatOutVarOut = 3;

class Bn3dTrainingUpdateStat {
public:
    __aicore__ inline Bn3dTrainingUpdateStat() = default;

    __aicore__ inline void Init(GM_ADDR sum, GM_ADDR squareSum, GM_ADDR scale, GM_ADDR offset, GM_ADDR mean,
                                GM_ADDR variance, GM_ADDR meanOut, GM_ADDR varianceOut, GM_ADDR batchMean,
                                GM_ADDR batchVariance, const Bn3dTrainingUpdateStatDesc& desc, TPipe* pipe)
    {
        desc_ = desc;
        pipe_ = pipe;
        const int64_t c = desc.channel;
        sumGm_.SetGlobalBuffer((__gm__ float*)sum, c);
        squareSumGm_.SetGlobalBuffer((__gm__ float*)squareSum, c);
        scaleGm_.SetGlobalBuffer((__gm__ float*)scale, c);
        offsetGm_.SetGlobalBuffer((__gm__ float*)offset, c);
        meanGm_.SetGlobalBuffer((__gm__ float*)mean, c);
        varianceGm_.SetGlobalBuffer((__gm__ float*)variance, c);
        meanOutGm_.SetGlobalBuffer((__gm__ float*)meanOut, c);
        varianceOutGm_.SetGlobalBuffer((__gm__ float*)varianceOut, c);
        batchMeanGm_.SetGlobalBuffer((__gm__ float*)batchMean, c);
        batchVarianceGm_.SetGlobalBuffer((__gm__ float*)batchVariance, c);
        slotBytes_ = ChunkSlotBytes(static_cast<int64_t>(VL_FP32), UB_BLOCK_BYTES);
        slotElems_ = slotBytes_ / static_cast<int64_t>(sizeof(float));
        pipe_->InitBuffer(slotBuf_, (kStatInputSlots + kStatResultSlots) * slotBytes_);
    }

    __aicore__ inline void Process()
    {
        const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
        const int64_t unitCount = desc_.baseStatPerCore + (blockIdx < desc_.extraStatCount ? 1 : 0);
        int64_t unit = blockIdx * desc_.baseStatPerCore +
                       (blockIdx < desc_.extraStatCount ? blockIdx : desc_.extraStatCount);
        for (int64_t i = 0; i < unitCount; ++i, ++unit) {
            const int64_t unitBegin = unit * desc_.statChunk;
            const int64_t remain = desc_.channel - unitBegin;
            const int64_t unitLen = remain < desc_.statChunk ? remain : desc_.statChunk;
            if (unitLen > 0) {
                RunUnit(unitBegin, unitLen);
            }
        }
    }

private:
    __aicore__ inline __ubuf__ float* Slot(int64_t slot)
    {
        return (__ubuf__ float*)slotBuf_.Get<float>()[slot * slotElems_].GetPhyAddr();
    }

    __aicore__ inline void StageIn(const GlobalTensor<float>& gm, int64_t slot, int64_t gmOffset, int64_t count)
    {
        LocalTensor<float> ub = slotBuf_.Get<float>()[slot * slotElems_];
        DataCopyExtParams params{1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0};
        DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        DataCopyPad(ub, gm[gmOffset], params, pad);
    }

    __aicore__ inline void StageOut(const GlobalTensor<float>& gm, int64_t slot, int64_t gmOffset, int64_t count)
    {
        LocalTensor<float> ub = slotBuf_.Get<float>()[slot * slotElems_];
        DataCopyExtParams params{1, static_cast<uint32_t>(count * sizeof(float)), 0, 0, 0};
        DataCopyPad(gm[gmOffset], ub, params);
    }

    // 一个 S 单元:按折叠分块(kParamChunk)处理 unitLen 个通道。
    __aicore__ inline void RunUnit(int64_t unitBegin, int64_t unitLen)
    {
        const int64_t chunk = kParamChunkLen(static_cast<int64_t>(VL_FP32));
        for (int64_t c0 = 0; c0 < unitLen; c0 += chunk) {
            int64_t count = unitLen - c0;
            if (count > chunk) {
                count = chunk;
            }
            const int64_t gmOffset = unitBegin + c0;
            const uint32_t len = static_cast<uint32_t>(count);

            // 本轮的 StageIn(MTE2 写槽 0..5)与 VF(V 写槽 0..3)都要落在**上一轮 StageOut(MTE3)
            // 刚读过**的那批槽上(槽 {0,1,2,3} 与 staging 重叠)。MTE3 与 MTE2/V 分属不同 pipe,
            // 必须显式配 MTE3→MTE2 与 MTE3→V 两对事件,否则构成跨轮 WAR 竞争。后果是 4 路 (C,)
            // 输出概率性错值且**不崩** —— 正是"总体 PASS/FAIL 看不出来"的一类。
            // 逐轮自平衡(相邻 SetFlag/WaitFlag 成对),不设首/末轮特判、不引整栏 barrier。
            SyncAfterPrevStoreOut();
            // 6 路输入 staging:等上一轮 VF 读完本 staging(V 拥有数据 → MTE2 复用),再搬入。
            SyncBeforeStaging();
            const int64_t inSlots[kStatInputSlots] = {kStatInSum,    kStatInSquareSum, kStatInScale,
                                                      kStatInOffset, kStatInMean,      kStatInVariance};
            StageIn(sumGm_, inSlots[0], gmOffset, count);
            StageIn(squareSumGm_, inSlots[1], gmOffset, count);
            StageIn(scaleGm_, inSlots[2], gmOffset, count);
            StageIn(offsetGm_, inSlots[3], gmOffset, count);
            StageIn(meanGm_, inSlots[4], gmOffset, count);
            StageIn(varianceGm_, inSlots[5], gmOffset, count);
            SyncAfterStaging();
            // 折叠出参次序(非折叠式):mult / bm / offset(直通)/ saveVar。
            // 相 S 只要 bm(→ batchMean 槽,供 EMA 与写出)与 saveVar(→ saveVar 槽);
            // mult 与 offset 直通落 meanOut / varOut 槽 —— 其内容不被读取,随后被 VFStatEma 覆写
            // (结果槽仍只需 4 个)。
            asc_vf_call<VFFoldChunk>(Slot(kStatInSum), Slot(kStatInSquareSum), Slot(kStatInScale), Slot(kStatInOffset),
                                     Slot(kStatOutMeanOut), Slot(kStatOutBatchMean), Slot(kStatOutVarOut),
                                     Slot(kStatOutSaveVar), desc_.numRec, desc_.epsilon, len);
            // EMA 合并:meanOut/varOut 槽(2/3)以及折叠读的 bm/saveVar 槽(0/1)的跨轮 WAR
            // 已由循环开头的 SyncAfterPrevStoreOut() 统一配好,此处不再重复。
            asc_vf_call<VFStatEma>(Slot(kStatOutBatchMean), Slot(kStatOutSaveVar), Slot(kStatInMean),
                                   Slot(kStatInVariance), Slot(kStatOutMeanOut), Slot(kStatOutVarOut), desc_.factor,
                                   desc_.oneMinusFactor, desc_.besselScaler, len);
            SyncBeforeStoreOut();
            StageOut(batchMeanGm_, kStatOutBatchMean, gmOffset, count);
            StageOut(batchVarianceGm_, kStatOutSaveVar, gmOffset, count);
            StageOut(meanOutGm_, kStatOutMeanOut, gmOffset, count);
            StageOut(varianceOutGm_, kStatOutVarOut, gmOffset, count);
        }
    }

    // V 读完 staging → MTE2 可复用;两者相邻成对,每轮自平衡(不整栏 barrier)。
    __aicore__ inline void SyncBeforeStaging()
    {
        event_t event = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(event);
        WaitFlag<HardEvent::V_MTE2>(event);
    }

    __aicore__ inline void SyncAfterStaging()
    {
        event_t event = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(event);
        WaitFlag<HardEvent::MTE2_V>(event);
    }

    // 上一轮 StageOut(MTE3)读完本批槽位后,MTE2 与 V 方可重写。
    // MTE3→MTE2 防"MTE2 写槽 0..5 覆盖 MTE3 尚未读完的 0..3";MTE3→V 防"V 写槽 0..3 覆盖同一批"。
    // 两者分属不同 pipe,缺任一即不完整。
    __aicore__ inline void SyncAfterPrevStoreOut()
    {
        event_t eventMte2 = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE3_MTE2));
        SetFlag<HardEvent::MTE3_MTE2>(eventMte2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventMte2);
        event_t eventVec = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventVec);
        WaitFlag<HardEvent::MTE3_V>(eventVec);
    }

    __aicore__ inline void SyncBeforeStoreOut()
    {
        event_t event = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(event);
        WaitFlag<HardEvent::V_MTE3>(event);
    }

private:
    Bn3dTrainingUpdateStatDesc desc_{};
    TPipe* pipe_ = nullptr;
    int64_t slotBytes_ = 0;
    int64_t slotElems_ = 0;
    TBuf<TPosition::VECCALC> slotBuf_;
    GlobalTensor<float> sumGm_;
    GlobalTensor<float> squareSumGm_;
    GlobalTensor<float> scaleGm_;
    GlobalTensor<float> offsetGm_;
    GlobalTensor<float> meanGm_;
    GlobalTensor<float> varianceGm_;
    GlobalTensor<float> meanOutGm_;
    GlobalTensor<float> varianceOutGm_;
    GlobalTensor<float> batchMeanGm_;
    GlobalTensor<float> batchVarianceGm_;
};

} // namespace Bn3dTrainingUpdateOps

#endif // BN3_D_TRAINING_UPDATE_STAT_H
