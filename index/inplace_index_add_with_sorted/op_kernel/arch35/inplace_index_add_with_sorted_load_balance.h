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
 * \file inplace_index_add_with_sorted_load_balance.h
 * \brief A5 (ascend950) Load-Balance kernel — StageA (核内累加分流) + StageB (跨核合并)
 *
 *   结构基于老 fix.h（dLoop 外层 / rows 内层），改造点：
 *     1. accumBuf 保存 pure α×update 和（不含 var）；DIRECT 闭组时单次加载 var 并累加
 *     2. 边界冲突（前/后哨兵）的组写出走 workspace（row 0 / row 1），不写 GM
 *     3. SyncAll 后 StageB 扫描 workspace 全表 index，跨核合并相同 index 行 + 加 var → GM
 *
 *   workspace 物理分离布局（TilingData struct 零改动，布局参数 Kernel 现算）：
 *     段 1 [0, wsIndexSize)：连续 int32，usedCoreNum × 2 个，初值 -1
 *     段 2 [wsIndexSize, wsTotal)：连续 fp32，usedCoreNum × 2 × updatesOneTime 个
 */
#ifndef INPLACE_INDEX_ADD_WITH_SORTED_ARCH35_LOAD_BALANCE_H_
#define INPLACE_INDEX_ADD_WITH_SORTED_ARCH35_LOAD_BALANCE_H_

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "../inc/platform.h"
#include "inplace_index_add_with_sorted_struct.h"

using namespace AscendC;

constexpr int64_t LB_BUFFER_NUM = 2;
constexpr int64_t LB_BLOCK_SIZE = 32;
constexpr int64_t LB_NUM_TWO = 2;
constexpr int64_t LB_INDEX_UB_NUM = 1536;
constexpr int64_t LB_WS_ROWS_PER_CORE = 2; // 每核 workspace 行数：前哨兵 + 后哨兵
constexpr int32_t LB_WS_INVALID = -1;      // workspace index 段无效标记
constexpr uint32_t LB_B32_REP_SIZE = platform::GetVRegSize() / sizeof(float); // = 64

constexpr MicroAPI::CastTrait castTraitB162B32Lb = {
    MicroAPI::RegLayout::ZERO,
    MicroAPI::SatMode::UNKNOWN,
    MicroAPI::MaskMergeMode::ZEROING,
    RoundMode::UNKNOWN,
};

__aicore__ inline int64_t CeilAlignLb(int64_t val, int64_t align) { return ((val + align - 1) / align) * align; }

// ============================================================================
// MicroAPI VF helpers（pure float32, DIST_NORM）
// ============================================================================
template <typename T>
__simd_vf__ inline void LbAccumInitVf(__ubuf__ float* accumUb, __ubuf__ float* valueUb, uint32_t count,
                                      uint32_t oneRepeat, uint16_t repeat)
{
    // accum := value （组首行初始化，无 alpha）
    MicroAPI::RegTensor<float> valueReg;
    MicroAPI::MaskReg mask;
    for (uint16_t r = 0; r < repeat; ++r) {
        uint32_t offset = r * oneRepeat;
        uint32_t curCount = (r == repeat - 1) ? (count - r * oneRepeat) : oneRepeat;
        mask = MicroAPI::UpdateMask<float>(curCount);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(valueReg, valueUb + offset);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(accumUb + offset, valueReg, mask);
    }
}

template <typename T>
__simd_vf__ inline void LbAccumInitAlphaVf(__ubuf__ float* accumUb, __ubuf__ float* valueUb, uint32_t count,
                                           uint32_t oneRepeat, uint16_t repeat, float alphaVal)
{
    // accum := α × value （组首行初始化，带 alpha）
    MicroAPI::RegTensor<float> valueReg, scaledReg;
    MicroAPI::MaskReg mask;
    for (uint16_t r = 0; r < repeat; ++r) {
        uint32_t offset = r * oneRepeat;
        uint32_t curCount = (r == repeat - 1) ? (count - r * oneRepeat) : oneRepeat;
        mask = MicroAPI::UpdateMask<float>(curCount);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(valueReg, valueUb + offset);
        MicroAPI::Muls(scaledReg, valueReg, alphaVal, mask);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(accumUb + offset, scaledReg, mask);
    }
}

template <typename T>
__simd_vf__ inline void LbAccumAddVf(__ubuf__ float* accumUb, __ubuf__ float* valueUb, uint32_t count,
                                     uint32_t oneRepeat, uint16_t repeat)
{
    // accum += value （组内累加，无 alpha）
    MicroAPI::RegTensor<float> accumReg, valueReg;
    MicroAPI::MaskReg mask;
    for (uint16_t r = 0; r < repeat; ++r) {
        uint32_t offset = r * oneRepeat;
        uint32_t curCount = (r == repeat - 1) ? (count - r * oneRepeat) : oneRepeat;
        mask = MicroAPI::UpdateMask<float>(curCount);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(accumReg, accumUb + offset);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(valueReg, valueUb + offset);
        MicroAPI::Add(accumReg, accumReg, valueReg, mask);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(accumUb + offset, accumReg, mask);
    }
}

template <typename T>
__simd_vf__ inline void LbAccumAddAlphaVf(__ubuf__ float* accumUb, __ubuf__ float* valueUb, uint32_t count,
                                          uint32_t oneRepeat, uint16_t repeat, float alphaVal)
{
    // accum += α × value （组内累加，带 alpha）
    MicroAPI::RegTensor<float> accumReg, valueReg, scaledReg;
    MicroAPI::MaskReg mask;
    for (uint16_t r = 0; r < repeat; ++r) {
        uint32_t offset = r * oneRepeat;
        uint32_t curCount = (r == repeat - 1) ? (count - r * oneRepeat) : oneRepeat;
        mask = MicroAPI::UpdateMask<float>(curCount);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(accumReg, accumUb + offset);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(valueReg, valueUb + offset);
        MicroAPI::Muls(scaledReg, valueReg, alphaVal, mask);
        MicroAPI::Add(accumReg, accumReg, scaledReg, mask);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(accumUb + offset, accumReg, mask);
    }
}

template <typename T>
__simd_vf__ inline void LbAddVarVf(__ubuf__ float* accumUb, __ubuf__ float* varUb, uint32_t count, uint32_t oneRepeat,
                                   uint16_t repeat)
{
    // accum += var （DIRECT 闭组时单次加 var）
    MicroAPI::RegTensor<float> accumReg, varReg;
    MicroAPI::MaskReg mask;
    for (uint16_t r = 0; r < repeat; ++r) {
        uint32_t offset = r * oneRepeat;
        uint32_t curCount = (r == repeat - 1) ? (count - r * oneRepeat) : oneRepeat;
        mask = MicroAPI::UpdateMask<float>(curCount);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(accumReg, accumUb + offset);
        MicroAPI::LoadAlign<float, MicroAPI::LoadDist::DIST_NORM>(varReg, varUb + offset);
        MicroAPI::Add(accumReg, accumReg, varReg, mask);
        MicroAPI::StoreAlign<float, MicroAPI::StoreDist::DIST_NORM>(accumUb + offset, accumReg, mask);
    }
}

// ============================================================================
// InplaceIndexAddWithSortedLoadBalance
// ============================================================================
template <typename T>
class InplaceIndexAddWithSortedLoadBalance {
public:
    __aicore__ inline InplaceIndexAddWithSortedLoadBalance(
        TPipe* pipeIn, const InplaceIndexAddWithSortedTilingData* __restrict tilingData)
    {
        pipe_ = pipeIn;
        coreId_ = GetBlockIdx();

        usedCoreNum_ = tilingData->usedCoreNum;
        enableAlpha_ = tilingData->enableAlpha;
        eachIndexCount_ = tilingData->eachIndexCount;
        lastIndexCount_ = tilingData->lastIndexCount;
        inputCount_ = tilingData->inputCount;
        indicesCount_ = tilingData->indicesCount;
        updatesCount_ = tilingData->updatesCount;
        updatesOneTime_ = tilingData->updatesOneTime;
        maxSize_ = tilingData->maxSize;
        eachNum_ = tilingData->eachNum;
        eachLoop_ = tilingData->eachLoop;
        eachTail_ = tilingData->eachTail;
        eachUBIndexRound_ = tilingData->eachUBIndexRound;
        eachUBIndexCount_ = tilingData->eachUBIndexCount;
        eachUBIndexTail_ = tilingData->eachUBIndexTail;
        lastUBIndexRound_ = tilingData->lastUBIndexRound;
        lastUBIndexCount_ = tilingData->lastUBIndexCount;
        lastUBIndexTail_ = tilingData->lastUBIndexTail;

        isLastCore_ = (coreId_ == usedCoreNum_ - 1);
        currentUBIndexRound_ = isLastCore_ ? lastUBIndexRound_ : eachUBIndexRound_;
        currentIndexCount_ = isLastCore_ ? lastUBIndexCount_ : eachUBIndexCount_;
        currentUBIndexTail_ = isLastCore_ ? lastUBIndexTail_ : eachUBIndexTail_;

        // workspace 布局参数（Kernel 现算，不入 TilingData）
        wsIndexSize_ = usedCoreNum_ * LB_WS_ROWS_PER_CORE * sizeof(int32_t);
        wsIndexBufSize_ = CeilAlignLb(wsIndexSize_, LB_BLOCK_SIZE);
    }

    __aicore__ inline void Init(GM_ADDR var, GM_ADDR value, GM_ADDR sortedIndices, GM_ADDR pos, GM_ADDR alpha,
                                GM_ADDR workspace)
    {
        varGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(var), inputCount_);
        valueGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(value), updatesCount_);
        idxGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(sortedIndices), indicesCount_);
        posGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(pos), indicesCount_);
        dstGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(var), inputCount_); // in-place

        // workspace 物理分离布局：段 1 index（int32），段 2 data（fp32，字节偏移 = wsIndexBufSize_）
        wsIndexGm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(workspace), usedCoreNum_ * LB_WS_ROWS_PER_CORE);
        wsDataGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(reinterpret_cast<__gm__ uint8_t*>(workspace) + wsIndexBufSize_),
            usedCoreNum_ * LB_WS_ROWS_PER_CORE * updatesOneTime_);

        // alpha 标量（不占 UB）
        if (enableAlpha_ == 1) {
            alphaGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(alpha), 1);
            alphaScalar_ = alphaGm_.GetValue(0);
            if (alphaScalar_ == static_cast<float>(1.0)) {
                enableAlpha_ = 0;
            }
        }

        // 哨兵（标量 GM 读取）
        frontSentinelIdx_ = (coreId_ != 0) ? idxGm_.GetValue(coreId_ * eachIndexCount_ - 1) : LB_WS_INVALID;
        backSentinelIdx_ = (coreId_ != usedCoreNum_ - 1) ? idxGm_.GetValue((coreId_ + 1) * eachIndexCount_) :
                                                           LB_WS_INVALID;

        // ===== 单 TPipe：一次 InitBuffer 分配所有 buffer =====
        pipe_->InitBuffer(wsIndexBuf_, wsIndexBufSize_);                                 // B1: 动态
        pipe_->InitBuffer(indexQue_, 1, LB_NUM_TWO * LB_INDEX_UB_NUM * sizeof(int32_t)); // B4+B5 合并
        pipe_->InitBuffer(updateQue_, LB_BUFFER_NUM, maxSize_ * sizeof(float));          // B6: NK9 fp32
        pipe_->InitBuffer(varInQue_, LB_BUFFER_NUM, maxSize_ * sizeof(float));           // B7: NK9 fp32
        pipe_->InitBuffer(accumBuf_, maxSize_ * sizeof(float));                          // B8: 累加器
        pipe_->InitBuffer(outQue_, LB_BUFFER_NUM, maxSize_ * sizeof(T));                 // B10: T 类型

        // Publish index metadata through MTE3 so Stage B observes it after SyncAll.
        WriteWorkspaceIndex(0, LB_WS_INVALID);
        WriteWorkspaceIndex(1, LB_WS_INVALID);
    }

    __aicore__ inline void Process()
    {
        StageAProcess();
        // 确保 workspace 写入（MTE3 data + S index）全流水排空，再跨核同步
        event_t eh = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
        SetFlag<HardEvent::MTE3_S>(eh);
        WaitFlag<HardEvent::MTE3_S>(eh);
        SyncAll();
        StageBProcess();
    }

private:
    // ============ Stage A：核内累加分流 ============
    __aicore__ inline void StageAProcess()
    {
        for (int64_t dLoop = 0; dLoop < eachLoop_; ++dLoop) {
            currentEachNum_ = (dLoop == eachLoop_ - 1) ? eachTail_ : eachNum_;
            currentEachNumAlign_ = CeilAlignLb(currentEachNum_, BLOCK_UB_SIZE_);
            ProcessTile(dLoop);
        }
    }

    __aicore__ inline void ProcessTile(int64_t dLoop)
    {
        bool firstMov = true;
        int32_t lastUpdateIndex = -1;
        bool groupStartedAtFirstRow = false; // 当前累加组是否起始于核内第一行

        for (int64_t idxRound = 0; idxRound < currentUBIndexRound_; ++idxRound) {
            currentEachIndex_ = (idxRound == currentUBIndexRound_ - 1) ? currentUBIndexTail_ : currentIndexCount_;
            int64_t indexOffset = coreId_ * eachIndexCount_ + idxRound * LB_INDEX_UB_NUM;

            CopyInIndicesPos(indexOffset, currentEachIndex_);

            for (int64_t j = 0; j < currentEachIndex_; ++j) {
                int32_t curIdx = idxLocal_.GetValue(j);
                int32_t curPos = idxLocal_.GetValue(LB_INDEX_UB_NUM + j);
                bool isGroupStart = firstMov || (curIdx != lastUpdateIndex);

                // 闭前一个组（deferred close，对齐 fix.h ifSyncOut）
                if (isGroupStart && !firstMov) {
                    bool endedAtLastRow = false; // 后续还有行，非末尾
                    CloseGroup(lastUpdateIndex, dLoop, groupStartedAtFirstRow, endedAtLastRow);
                }

                if (isGroupStart) {
                    groupStartedAtFirstRow = (idxRound == 0 && j == 0);
                    AccumInit(curPos, dLoop);
                } else {
                    AccumAdd(curPos, dLoop);
                }

                lastUpdateIndex = curIdx;
                firstMov = false;
            }

            FreeIndexTensor();
        }

        // 闭最后一个组（endedAtLastRow = true，可能触发后哨兵冲突）
        if (!firstMov) {
            bool endedAtLastRow = true;
            CloseGroup(lastUpdateIndex, dLoop, groupStartedAtFirstRow, endedAtLastRow);
        }
    }

    __aicore__ inline void CopyInIndicesPos(int64_t indexOffset, int64_t count)
    {
        idxLocal_ = indexQue_.AllocTensor<int32_t>();
        DataCopyPadExtParams<int32_t> tPadParams = {false, 0, 0, static_cast<int32_t>(0)};
        DataCopyExtParams extParams = {(uint16_t)1, static_cast<uint32_t>(count * sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(idxLocal_, idxGm_[indexOffset], extParams, tPadParams);
        DataCopyPad(idxLocal_[LB_INDEX_UB_NUM], posGm_[indexOffset], extParams, tPadParams);

        event_t eventIDMTE2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
        WaitFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
    }

    __aicore__ inline void FreeIndexTensor() { indexQue_.FreeTensor(idxLocal_); }

    // ---- 累加初始化（组首行）：accum := α × value ----
    __aicore__ inline void AccumInit(int32_t pos, int64_t dLoop)
    {
        CopyValueIn(pos, dLoop, currentEachNum_);
        auto valueLocal = updateQue_.template DeQue<float>();

        auto accumLocal = accumBuf_.Get<float>();
        __ubuf__ float* accumUb = reinterpret_cast<__ubuf__ float*>(accumLocal.GetPhyAddr());
        __ubuf__ float* valueUb = reinterpret_cast<__ubuf__ float*>(valueLocal.GetPhyAddr());
        DoAccum(accumUb, valueUb, currentEachNumAlign_, /*isInit=*/true);
        // 等 VF_CALL 读完 valueUb，再释放，避免下次 CopyValueIn 的 MTE2 覆盖
        PipeBarrier<PIPE_V>();
        updateQue_.FreeTensor(valueLocal);
    }

    // ---- 累加（组内非首行）：accum += α × value ----
    __aicore__ inline void AccumAdd(int32_t pos, int64_t dLoop)
    {
        CopyValueIn(pos, dLoop, currentEachNum_);
        auto valueLocal = updateQue_.template DeQue<float>();

        auto accumLocal = accumBuf_.Get<float>();
        __ubuf__ float* accumUb = reinterpret_cast<__ubuf__ float*>(accumLocal.GetPhyAddr());
        __ubuf__ float* valueUb = reinterpret_cast<__ubuf__ float*>(valueLocal.GetPhyAddr());
        DoAccum(accumUb, valueUb, currentEachNumAlign_, /*isInit=*/false);
        // 等 VF_CALL 读完 valueUb，再释放，避免下次 CopyValueIn 的 MTE2 覆盖
        PipeBarrier<PIPE_V>();
        updateQue_.FreeTensor(valueLocal);
    }

    __aicore__ inline void DoAccum(__ubuf__ float* accumUb, __ubuf__ float* valueUb, int64_t count, bool isInit)
    {
        uint32_t cnt = static_cast<uint32_t>(count);
        uint16_t repeat = static_cast<uint16_t>(AscendC::CeilDivision(cnt, LB_B32_REP_SIZE));
        if (enableAlpha_ == 1) {
            if (isInit) {
                AscendC::VF_CALL<LbAccumInitAlphaVf<T>>(accumUb, valueUb, cnt, LB_B32_REP_SIZE, repeat, alphaScalar_);
            } else {
                AscendC::VF_CALL<LbAccumAddAlphaVf<T>>(accumUb, valueUb, cnt, LB_B32_REP_SIZE, repeat, alphaScalar_);
            }
        } else {
            if (isInit) {
                AscendC::VF_CALL<LbAccumInitVf<T>>(accumUb, valueUb, cnt, LB_B32_REP_SIZE, repeat);
            } else {
                AscendC::VF_CALL<LbAccumAddVf<T>>(accumUb, valueUb, cnt, LB_B32_REP_SIZE, repeat);
            }
        }
    }

    // ---- 闭组：判定分支，DIRECT 写 GM / WORKSPACE 写 workspace data + index ----
    __aicore__ inline void CloseGroup(int32_t closedIdx, int64_t dLoop, bool startedAtFirstRow, bool endedAtLastRow)
    {
        bool frontConflict = startedAtFirstRow && (coreId_ != 0) && (closedIdx == frontSentinelIdx_);
        bool backConflict = endedAtLastRow && (coreId_ != usedCoreNum_ - 1) && (closedIdx == backSentinelIdx_);

        int32_t wsRow; // 0 = 前哨兵行，1 = 后哨兵行
        bool toWorkspace = false;
        if (frontConflict && backConflict) {
            wsRow = 1; // 同时前后冲突 → 统一归入后哨兵行
            toWorkspace = true;
        } else if (frontConflict) {
            wsRow = 0;
            toWorkspace = true;
        } else if (backConflict) {
            wsRow = 1;
            toWorkspace = true;
        }

        if (toWorkspace) {
            WriteWorkspace(closedIdx, wsRow, dLoop);
        } else {
            WriteDirect(closedIdx, dLoop);
        }
    }

    // ---- DIRECT 闭组：单次加 var → Cast → 写 GM ----
    __aicore__ inline void WriteDirect(int32_t idx, int64_t dLoop)
    {
        // 加 var（仅 DIRECT 路径加 var）
        CopyVarIn(idx, dLoop, currentEachNum_);
        auto varLocal = varInQue_.template DeQue<float>();

        __ubuf__ float* accumUb = reinterpret_cast<__ubuf__ float*>(accumBuf_.Get<float>().GetPhyAddr());
        __ubuf__ float* varUb = reinterpret_cast<__ubuf__ float*>(varLocal.GetPhyAddr());
        uint32_t cnt = static_cast<uint32_t>(currentEachNumAlign_);
        uint16_t repeat = static_cast<uint16_t>(AscendC::CeilDivision(cnt, LB_B32_REP_SIZE));
        AscendC::VF_CALL<LbAddVarVf<T>>(accumUb, varUb, cnt, LB_B32_REP_SIZE, repeat);
        // 等 VF_CALL 读完 varUb、写完 accumUb，再释放 varLocal
        PipeBarrier<PIPE_V>();
        varInQue_.FreeTensor(varLocal);

        // Cast fp32 → T 并写回 GM
        auto accumLocal = accumBuf_.Get<float>();
        LocalTensor<T> outT = outQue_.template AllocTensor<T>();
        Cast(outT, accumLocal, RoundMode::CAST_RINT, currentEachNumAlign_);
        outQue_.EnQue(outT);
        outT = outQue_.template DeQue<T>();
        DataCopyExtParams extParams = {(uint16_t)1, static_cast<uint32_t>(currentEachNum_ * sizeof(T)), 0, 0, 0};
        DataCopyPad(dstGm_[idx * updatesOneTime_ + dLoop * maxSize_], outT, extParams);
        outQue_.FreeTensor(outT);
    }

    // ---- WORKSPACE 闭组：fp32 直写 workspace data + 标量写 index ----
    __aicore__ inline void WriteWorkspace(int32_t idx, int32_t wsRow, int64_t dLoop)
    {
        auto accumLocal = accumBuf_.Get<float>();
        // workspace data 段：fp32 直写（无 Cast）
        DataCopyExtParams extParams = {(uint16_t)1, static_cast<uint32_t>(currentEachNum_ * sizeof(float)), 0, 0, 0};
        int64_t dataOffset = (coreId_ * LB_WS_ROWS_PER_CORE + wsRow) * updatesOneTime_ + dLoop * maxSize_;

        event_t eventIDVToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIDVToMTE3);
        WaitFlag<HardEvent::V_MTE3>(eventIDVToMTE3);

        DataCopyPad(wsDataGm_[dataOffset], accumLocal, extParams);

        // accumBuf_ is reused by the next group, so wait until MTE3 has consumed it.
        event_t eventIDMTE3ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);
        WaitFlag<HardEvent::MTE3_V>(eventIDMTE3ToV);

        // Publish the index only after its data row is globally visible.
        WriteWorkspaceIndex(wsRow, idx);
    }

    __aicore__ inline void WriteWorkspaceIndex(int32_t wsRow, int32_t idx)
    {
        int64_t slot = coreId_ * LB_WS_ROWS_PER_CORE + wsRow;
        auto wsIndexLocal = wsIndexBuf_.Get<int32_t>();
        // MTE3 requires an aligned UB source; the GM slot carries the per-core offset.
        wsIndexLocal.SetValue(0, idx);

        event_t eventIDSToMTE3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
        SetFlag<HardEvent::S_MTE3>(eventIDSToMTE3);
        WaitFlag<HardEvent::S_MTE3>(eventIDSToMTE3);

        DataCopyExtParams extParams = {(uint16_t)1, static_cast<uint32_t>(sizeof(int32_t)), 0, 0, 0};
        DataCopyPad(wsIndexGm_[slot], wsIndexLocal, extParams);

        event_t eventIDMTE3ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_S));
        SetFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);
        WaitFlag<HardEvent::MTE3_S>(eventIDMTE3ToS);
    }

    // ---- NK9 Cast-in-place：T → fp32（对齐 fix.h CopyValueIn）----
    __aicore__ inline void CopyValueIn(int32_t pos, int64_t progress, int64_t dataLen)
    {
        auto valueLocal = updateQue_.template AllocTensor<float>();
        LocalTensor<T> valueLocalT = valueLocal.template ReinterpretCast<T>();
        int64_t valueOffset = CeilAlignLb(dataLen, BLOCK_UB_SIZE_);

        DataCopyPadExtParams<T> tPadParams = {false, 0, 0, static_cast<T>(0)};
        DataCopyExtParams extParams = {(uint16_t)1, static_cast<uint32_t>(dataLen * sizeof(T)), 0, 0, 0};
        DataCopyPad(valueLocalT[valueOffset], valueGm_[pos * updatesOneTime_ + progress * maxSize_], extParams,
                    tPadParams);

        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);

        CastT2Fp32(valueLocal, valueLocalT, valueOffset, dataLen);
        updateQue_.EnQue(valueLocal);
    }

    // ---- NK9 Cast-in-place：var T → fp32 ----
    __aicore__ inline void CopyVarIn(int32_t idx, int64_t progress, int64_t dataLen)
    {
        auto varLocal = varInQue_.template AllocTensor<float>();
        LocalTensor<T> varLocalT = varLocal.template ReinterpretCast<T>();
        int64_t varOffset = CeilAlignLb(dataLen, BLOCK_UB_SIZE_);

        DataCopyPadExtParams<T> tPadParams = {false, 0, 0, static_cast<T>(0)};
        DataCopyExtParams extParams = {(uint16_t)1, static_cast<uint32_t>(dataLen * sizeof(T)), 0, 0, 0};
        DataCopyPad(varLocalT[varOffset], varGm_[idx * updatesOneTime_ + progress * maxSize_], extParams, tPadParams);

        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);

        CastT2Fp32(varLocal, varLocalT, varOffset, dataLen);
        varInQue_.EnQue(varLocal);
    }

    // ---- MicroAPI VF: T staging（后半）→ fp32（前半）----
    __aicore__ inline void CastT2Fp32(LocalTensor<float>& dstLocal, LocalTensor<T>& srcLocalT, int64_t srcOffset,
                                      int64_t dataLen)
    {
        int64_t count = CeilAlignLb(dataLen, BLOCK_UB_SIZE_);
        uint16_t loops = AscendC::CeilDivision(count * sizeof(float), platform::GetVRegSize());
        uint32_t loopsStride = platform::GetVRegSize() / sizeof(float);

        __VEC_SCOPE__
        {
            __local_mem__ float* dst = reinterpret_cast<__local_mem__ float*>(dstLocal.GetPhyAddr());
            __local_mem__ T* src = reinterpret_cast<__local_mem__ T*>(srcLocalT.GetPhyAddr()) + srcOffset;
            uint32_t sreg = static_cast<uint32_t>(count);
            MicroAPI::MaskReg mask;
            MicroAPI::RegTensor<T> aReg;
            MicroAPI::RegTensor<float> bReg;
            for (uint16_t i = 0; i < loops; ++i) {
                mask = MicroAPI::UpdateMask<float>(sreg);
                MicroAPI::DataCopy<T, MicroAPI::LoadDist::DIST_UNPACK_B16>(aReg, src + i * loopsStride);
                MicroAPI::Cast<float, T, castTraitB162B32Lb>(bReg, aReg, mask);
                MicroAPI::DataCopy(dst + i * loopsStride, bReg, mask);
            }
        }
    }

    // ============ Stage B：跨核合并 ============
    __aicore__ inline void StageBProcess()
    {
        // 一次性搬入 workspace index 段（物理连续）
        CopyInWsIndex();
        auto wsIdx = wsIndexBuf_.Get<int32_t>();
        int64_t totalWsRows = usedCoreNum_ * LB_WS_ROWS_PER_CORE;

        // Each core owns only its two workspace rows. The full index table is still used
        // to determine ownership and merge consecutive rows from following cores.
        int64_t rowBegin = coreId_ * LB_WS_ROWS_PER_CORE;
        int64_t rowEnd = rowBegin + LB_WS_ROWS_PER_CORE;
        for (int64_t row = rowBegin; row < rowEnd; ++row) {
            int32_t curIdx = wsIdx.GetValue(row);
            if (curIdx == LB_WS_INVALID) {
                continue;
            }

            // 判定 curIdx 是否在 row 之前出现过（首个负责核判定）
            if (IsIndexSeenBefore(wsIdx, row, curIdx)) {
                continue;
            }

            // 本核负责：按 dLoop tile 逐块合并 + 写回
            for (int64_t dLoop = 0; dLoop < eachLoop_; ++dLoop) {
                currentEachNum_ = (dLoop == eachLoop_ - 1) ? eachTail_ : eachNum_;
                currentEachNumAlign_ = CeilAlignLb(currentEachNum_, BLOCK_UB_SIZE_);
                AccumulateRange(wsIdx, row, curIdx, totalWsRows, dLoop);
                WriteBackToDstGm(curIdx, dLoop);
            }
        }
    }

    __aicore__ inline void CopyInWsIndex()
    {
        auto wsIdx = wsIndexBuf_.Get<int32_t>();
        DataCopyPad(wsIdx, wsIndexGm_, {1, static_cast<uint32_t>(wsIndexSize_), 0, 0, 0},
                    {false, 0, 0, static_cast<int32_t>(0)});

        event_t eventIDMTE2ToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
        SetFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
        WaitFlag<HardEvent::MTE2_S>(eventIDMTE2ToS);
    }

    __aicore__ inline bool IsIndexSeenBefore(LocalTensor<int32_t>& wsIdx, int64_t row, int32_t curIdx)
    {
        for (int64_t prev = row - 1; prev >= 0; --prev) {
            int32_t prevIdx = wsIdx.GetValue(prev);
            if (prevIdx == LB_WS_INVALID) {
                continue;
            }
            if (prevIdx == curIdx) {
                return true;
            }
        }
        return false;
    }

    __aicore__ inline void AccumulateRange(LocalTensor<int32_t>& wsIdx, int64_t startRow, int32_t curIdx,
                                           int64_t totalWsRows, int64_t dLoop)
    {
        auto accumLocal = accumBuf_.Get<float>();
        // 确保上一轮（Stage A 末组 / 上一 dLoop 的 WriteBackToDstGm）对 accumBuf_ 的 V 读（Cast/VF_CALL）
        // 已完成，MTE2 才能覆盖 accumBuf_；否则尾轴分块循环时会出现 V 读与 MTE2 写的 WAR 冲突
        event_t eventIdVToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventIdVToMte2);
        WaitFlag<HardEvent::V_MTE2>(eventIdVToMte2);

        // 首行：accum := workspace data[startRow][dLoop tile]
        DataCopyExtParams extParams = {(uint16_t)1, static_cast<uint32_t>(currentEachNum_ * sizeof(float)), 0, 0, 0};
        DataCopyPad(accumLocal, wsDataGm_[startRow * updatesOneTime_ + dLoop * maxSize_], extParams,
                    {false, 0, 0, static_cast<float>(0)});

        event_t eventId = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);

        // 向后扫描相同 index 行累加
        for (int64_t next = startRow + 1; next < totalWsRows; ++next) {
            int32_t nextIdx = wsIdx.GetValue(next);
            if (nextIdx == LB_WS_INVALID) {
                continue;
            }
            if (nextIdx != curIdx) {
                break;
            }

            // Stage B starts after Stage A has drained updateQue_; reuse it for workspace data.
            auto dataIn = updateQue_.template AllocTensor<float>();
            DataCopyPad(dataIn, wsDataGm_[next * updatesOneTime_ + dLoop * maxSize_], extParams,
                        {false, 0, 0, static_cast<float>(0)});
            updateQue_.EnQue(dataIn);
            auto dataLocal = updateQue_.template DeQue<float>();

            __ubuf__ float* accumUb = reinterpret_cast<__ubuf__ float*>(accumLocal.GetPhyAddr());
            __ubuf__ float* dataUb = reinterpret_cast<__ubuf__ float*>(dataLocal.GetPhyAddr());
            uint32_t cnt = static_cast<uint32_t>(currentEachNumAlign_);
            uint16_t repeat = static_cast<uint16_t>(AscendC::CeilDivision(cnt, LB_B32_REP_SIZE));
            AscendC::VF_CALL<LbAccumAddVf<T>>(accumUb, dataUb, cnt, LB_B32_REP_SIZE, repeat);
            // 等 VF_CALL 读完 dataUb，再释放，避免下一行 MTE2 覆盖
            PipeBarrier<PIPE_V>();
            updateQue_.FreeTensor(dataLocal);
        }
    }

    __aicore__ inline void WriteBackToDstGm(int32_t curIdx, int64_t dLoop)
    {
        // 末尾单次加 var（workspace 路径 var 唯一累加点）
        CopyVarIn(curIdx, dLoop, currentEachNum_);
        auto varLocal = varInQue_.template DeQue<float>();
        auto accumLocal = accumBuf_.Get<float>();

        __ubuf__ float* accumUb = reinterpret_cast<__ubuf__ float*>(accumLocal.GetPhyAddr());
        __ubuf__ float* varUb = reinterpret_cast<__ubuf__ float*>(varLocal.GetPhyAddr());
        uint32_t cnt = static_cast<uint32_t>(currentEachNumAlign_);
        uint16_t repeat = static_cast<uint16_t>(AscendC::CeilDivision(cnt, LB_B32_REP_SIZE));
        AscendC::VF_CALL<LbAddVarVf<T>>(accumUb, varUb, cnt, LB_B32_REP_SIZE, repeat);
        // 等 VF_CALL 读完 varUb、写完 accumUb，再释放 varLocal
        PipeBarrier<PIPE_V>();
        varInQue_.FreeTensor(varLocal);

        // Cast fp32 → T 并写回 GM
        LocalTensor<T> outT = outQue_.template AllocTensor<T>();
        Cast(outT, accumLocal, RoundMode::CAST_RINT, currentEachNumAlign_);
        outQue_.EnQue(outT);
        outT = outQue_.template DeQue<T>();
        DataCopyExtParams extParams = {(uint16_t)1, static_cast<uint32_t>(currentEachNum_ * sizeof(T)), 0, 0, 0};
        DataCopyPad(dstGm_[curIdx * updatesOneTime_ + dLoop * maxSize_], outT, extParams);
        outQue_.FreeTensor(outT);
    }

private:
    TPipe* pipe_;
    int64_t coreId_;
    bool isLastCore_;

    // TilingData 副本
    int32_t usedCoreNum_;
    int32_t enableAlpha_;
    int64_t eachIndexCount_;
    int64_t lastIndexCount_;
    int64_t inputCount_;
    int64_t indicesCount_;
    int64_t updatesCount_;
    int64_t updatesOneTime_;
    int64_t maxSize_;
    int64_t eachNum_;
    int64_t eachLoop_;
    int64_t eachTail_;
    int64_t eachUBIndexRound_;
    int64_t eachUBIndexCount_;
    int64_t eachUBIndexTail_;
    int64_t lastUBIndexRound_;
    int64_t lastUBIndexCount_;
    int64_t lastUBIndexTail_;

    // per-core 派生
    int64_t currentUBIndexRound_;
    int64_t currentIndexCount_;
    int64_t currentUBIndexTail_;
    int64_t currentEachIndex_;
    int64_t currentEachNum_;
    int64_t currentEachNumAlign_;

    // workspace 布局（Kernel 现算，不入 TilingData）
    int64_t wsIndexSize_;    // 段 1 大小 = 段 2 起始字节偏移
    int64_t wsIndexBufSize_; // B1 UB 大小

    // 哨兵 / 标量
    int32_t frontSentinelIdx_;
    int32_t backSentinelIdx_;
    float alphaScalar_;

    // Buffer（单 TPipe）
    TBuf<> wsIndexBuf_;                                 // B1
    TQue<QuePosition::VECIN, 1> indexQue_;              // B4+B5 合并（indices + pos）
    TQue<QuePosition::VECIN, LB_BUFFER_NUM> updateQue_; // B6（Stage A updates / Stage B workspace data 复用）
    TQue<QuePosition::VECIN, LB_BUFFER_NUM> varInQue_;  // B7
    TBuf<> accumBuf_;                                   // B8
    TQue<QuePosition::VECOUT, LB_BUFFER_NUM> outQue_;   // B10

    // LocalTensor 句柄
    LocalTensor<int32_t> idxLocal_;

    // GM
    GlobalTensor<T> varGm_;
    GlobalTensor<T> valueGm_;
    GlobalTensor<T> dstGm_;
    GlobalTensor<int32_t> idxGm_;
    GlobalTensor<int32_t> posGm_;
    GlobalTensor<float> alphaGm_;
    GlobalTensor<int32_t> wsIndexGm_;
    GlobalTensor<float> wsDataGm_;

    static constexpr int64_t BLOCK_UB_SIZE_ = 32 / sizeof(T); // T 元素对齐（ReinterpretCast staging）
};

#endif // INPLACE_INDEX_ADD_WITH_SORTED_ARCH35_LOAD_BALANCE_H_
