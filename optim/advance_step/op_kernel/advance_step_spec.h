/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file advance_step_spec.h
 * \brief
 */

#ifndef ADVANCE_STEP_SPEC_H
#define ADVANCE_STEP_SPEC_H

#include <type_traits>
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "advance_step_common.h"

namespace AdvanceStepNs {
using namespace AscendC;

// int32 向量 Div 后端支持性：dav_3510（Ascend950，__NPU_ARCH__==3510）支持 int32 vdiv
// （v3.6 上板验证）；dav_c220（Ascend910B/910_93 家族，__NPU_ARCH__==2201，CPU 仿真同源）
// 的 vdiv 仅 half/float。仅 950 走向量除，其余后端回退标量循环——正确性与接受面不变，
// 各后端维持已验证路径。Subs<int> 在 dav_c220 同样无实现，已统一经 Adds 负标量实现。
#if defined(__NPU_ARCH__) && __NPU_ARCH__ == 3510 && !defined(ASCENDC_CPU_DEBUG)
#define ADV_VEC_INT_DIV_SUPPORTED 1
#else
#define ADV_VEC_INT_DIV_SUPPORTED 0
#endif

template <typename T>
class KernelAdvanceStepSpec {
public:
    __aicore__ inline KernelAdvanceStepSpec(){};
    __aicore__ inline void Init(GM_ADDR input_tokens, GM_ADDR sampled_token_ids, GM_ADDR input_positions,
                                GM_ADDR seq_lens, GM_ADDR slot_mapping, GM_ADDR block_tables, GM_ADDR spec_token,
                                GM_ADDR accepted_num, GM_ADDR workspace, const AdvanceStepTilingData* tilingData,
                                TPipe* tPipe);
    __aicore__ inline void Process();
    __aicore__ inline void CopyIn(int64_t loop, int64_t seqsNum);
    __aicore__ inline void Compute(int64_t loop, int64_t seqsNum);
    __aicore__ inline void CopyOut(int64_t loop, int64_t seqsNum);
    __aicore__ inline void ComputeSlotMapping(int64_t loop, int64_t seqsNum);
    __aicore__ inline void GatherRowsScalar(const LocalTensor<int32_t>& lanesTensor, int64_t blockTablesOffset,
                                            int64_t rowStart, int64_t rowCount, int64_t rowStride);
    __aicore__ inline void CopyOutSlotMapping(int64_t loop, int64_t seqsNum);
    __aicore__ inline void CopyInInputTokens(int64_t loop, int64_t seqsNum);
    __aicore__ inline void ComputeInputTokens(int64_t loop, int64_t seqsNum);
    __aicore__ inline void CopyOutInputTokens(int64_t loop, int64_t seqsNum);

private:
    TPipe* pipe_;
    GlobalTensor<T> inputTokensGm_;
    GlobalTensor<T> sampledTokenIdsGm_;
    GlobalTensor<T> inputPositionsGm_;
    GlobalTensor<T> seqLensGm_;
    GlobalTensor<T> slotMappingGm_;
    GlobalTensor<T> blockTablesGm_;
    GlobalTensor<T> specTokenGm_;
    GlobalTensor<T> acceptedNumGm_;

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> buffer1;
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> buffer2;
    TQue<QuePosition::VECIN, 1> buffer3;
    TQue<QuePosition::VECIN, 1> tempQue_;

    TBuf<QuePosition::VECCALC> sampledTokenIdsBuf_;
    TBuf<QuePosition::VECCALC> outputInt32Buf_;
    // 向量化行展开 scratch（替代逐行标量化调用——大 shape 主要耗时源，实测 ~51ns/次×行数）：
    // rowIdxBuf_（行号/展开结果）、byteOffBuf_（字节偏移/行基址），acc32Buf_（acc int32 化 + te==2 等值掩码）
    TBuf<QuePosition::VECCALC> rowIdxBuf_;
    TBuf<QuePosition::VECCALC> byteOffBuf_;
    TBuf<QuePosition::VECCALC> acc32Buf_;
    int64_t blockIdx_ = 0;
    int64_t blockTablesStride_ = 0;
    int64_t numSeqs_ = 0;
    int64_t numQueries_ = 0;
    int64_t blockSize_ = 0;
    int64_t specNum_ = 0;
    int64_t curCoreSeqs_ = 0;
    int64_t curCoreLoopNum_ = 0;
    int64_t perLoopSeqs_ = 0;
    int64_t perCoreSeqs_ = 0;
    int64_t lastLoopSeqs_ = 0;
    int64_t tokenEachReqs_ = 0;
    bool align = false;
    bool vectorGather_ = false;
};

template <typename T>
__aicore__ inline void KernelAdvanceStepSpec<T>::Init(GM_ADDR input_tokens, GM_ADDR sampled_token_ids,
                                                      GM_ADDR input_positions, GM_ADDR seq_lens, GM_ADDR slot_mapping,
                                                      GM_ADDR block_tables, GM_ADDR spec_token, GM_ADDR accepted_num,
                                                      GM_ADDR workspace, const AdvanceStepTilingData* tilingData,
                                                      TPipe* tPipe)
{
    pipe_ = tPipe;
    blockIdx_ = GetBlockIdx();
    // tiling中获取本核处理的seqs数量
    if (blockIdx_ == tilingData->needCoreNum - 1) {
        curCoreSeqs_ = tilingData->lastCoreSeqs;
    } else {
        curCoreSeqs_ = tilingData->perCoreSeqs;
    }
    specNum_ = tilingData->specNum;
    blockSize_ = tilingData->blockSize;
    numSeqs_ = tilingData->numSeqs;
    blockTablesStride_ = tilingData->blockTablesStride;

    // 计算每个loop需要处理的seqs数量、需要多少个loop、以及最后一个loop处理的seqs数量
    perCoreSeqs_ = tilingData->perCoreSeqs;
    perLoopSeqs_ = Min(tilingData->perLoopMaxSeqs, curCoreSeqs_);
    curCoreLoopNum_ = curCoreSeqs_ / perLoopSeqs_;
    lastLoopSeqs_ = curCoreSeqs_ - curCoreLoopNum_ * perLoopSeqs_;

    int64_t seqOffset = blockIdx_ * tilingData->perCoreSeqs;
    tokenEachReqs_ = specNum_ + 1;
    int64_t elementByte32 = AlignBytes(tokenEachReqs_, sizeof(T)) / 2;
    if (elementByte32 % BLOCK_BYTES != 0) {
        align = true;
    }

    inputTokensGm_.SetGlobalBuffer((__gm__ T*)input_tokens + seqOffset * tokenEachReqs_);
    sampledTokenIdsGm_.SetGlobalBuffer((__gm__ T*)sampled_token_ids + seqOffset * tokenEachReqs_);
    inputPositionsGm_.SetGlobalBuffer((__gm__ T*)input_positions + seqOffset * tokenEachReqs_);
    seqLensGm_.SetGlobalBuffer((__gm__ T*)seq_lens + seqOffset * tokenEachReqs_);
    slotMappingGm_.SetGlobalBuffer((__gm__ T*)slot_mapping + seqOffset * tokenEachReqs_);
    blockTablesGm_.SetGlobalBuffer((__gm__ T*)block_tables + seqOffset * blockTablesStride_);
    specTokenGm_.SetGlobalBuffer((__gm__ T*)spec_token + seqOffset * specNum_);
    acceptedNumGm_.SetGlobalBuffer((__gm__ T*)accepted_num + seqOffset);

    pipe_->InitBuffer(buffer1, 1, (AlignBytes(tokenEachReqs_, sizeof(T)) + BLOCK_BYTES) * perLoopSeqs_);
    pipe_->InitBuffer(buffer2, 1, (AlignBytes(tokenEachReqs_, sizeof(T)) + BLOCK_BYTES) * perLoopSeqs_);
    // buffer3 尾部加对齐余量：te==2 快速路径的 Select 按 256b 粒度计数写入 [seqs, seqsR) 垃圾 lane
    pipe_->InitBuffer(buffer3, 1,
                      (AlignBytes(tokenEachReqs_, sizeof(T)) + BLOCK_BYTES) * perLoopSeqs_ + VEC_SCRATCH_ALIGN_SLACK);
    pipe_->InitBuffer(tempQue_, 1, (AlignBytes(tokenEachReqs_, sizeof(T)) + BLOCK_BYTES) * perLoopSeqs_);

    pipe_->InitBuffer(sampledTokenIdsBuf_, AlignBytes(tokenEachReqs_ + 1, sizeof(float)));
    pipe_->InitBuffer(outputInt32Buf_, (AlignBytes(tokenEachReqs_, sizeof(int32_t)) + BLOCK_BYTES) * perLoopSeqs_);

    // 向量化 scratch：与 kernel rowStride（AlignBytes(te,8)/8 + align 时 +4 元素）一致的容量上界，
    // 尾部附加对齐槽余量（te==2 快速路径的 x0/x1/neg1/t0 子张量须 32B 对齐切片，槽距按 64B 计）
    int64_t vecRowStride = AlignBytes(tokenEachReqs_, sizeof(T)) / static_cast<int64_t>(sizeof(T));
    if (align) {
        vecRowStride += INT64_ONE_BLOCK_NUM;
    }
    pipe_->InitBuffer(rowIdxBuf_, perLoopSeqs_ * vecRowStride * sizeof(int32_t) + VEC_SCRATCH_ALIGN_SLACK);
    pipe_->InitBuffer(byteOffBuf_, perLoopSeqs_ * vecRowStride * sizeof(int32_t) + VEC_SCRATCH_ALIGN_SLACK);
    pipe_->InitBuffer(acc32Buf_, perLoopSeqs_ * sizeof(int32_t) + VEC_SCRATCH_ALIGN_SLACK);

    // 注：向量 gather 已回退标量查表（见 ComputeSlotMapping），btChunkQue_ 分块缓冲不再分配；
    // tiling 侧 vectorGather 预算仍按含分块上界估算，此处实际分配 ≤ 预留，接受面不变
    vectorGather_ = (tilingData->vectorGather != 0);
}

template <typename T>
__aicore__ inline void KernelAdvanceStepSpec<T>::CopyIn(int64_t loop, int64_t seqsNum)
{
    LocalTensor<T> inputPositionsLocal = buffer1.AllocTensor<T>();
    LocalTensor<T> seqLensLocal = buffer2.AllocTensor<T>();
    LocalTensor<T> acceptedNumLocal = buffer3.AllocTensor<T>();
    DataCopyExtParams dataCopyParams{static_cast<uint16_t>(seqsNum), static_cast<uint32_t>(tokenEachReqs_ * sizeof(T)),
                                     0, 0, 0};
    if (align) {
        dataCopyParams.dstStride = 1;
    }
    DataCopyPadExtParams<T> dataCopyPadParams{false, 0, 0, 0};
    int64_t offset = loop * perLoopSeqs_ * tokenEachReqs_;
    DataCopyPad(inputPositionsLocal, inputPositionsGm_[offset], dataCopyParams, dataCopyPadParams);
    DataCopyPad(seqLensLocal, seqLensGm_[offset], dataCopyParams, dataCopyPadParams);
    DataCopyExtParams acceptedNumDataCopyParams{static_cast<uint16_t>(1), static_cast<uint32_t>(seqsNum * sizeof(T)), 0,
                                                0, 0};
    offset = loop * perLoopSeqs_;
    DataCopyPad(acceptedNumLocal, acceptedNumGm_[offset], acceptedNumDataCopyParams, dataCopyPadParams);

    buffer1.EnQue(inputPositionsLocal);
    buffer2.EnQue(seqLensLocal);
    buffer3.EnQue(acceptedNumLocal);
}

template <typename T>
__aicore__ inline void KernelAdvanceStepSpec<T>::Compute(int64_t loop, int64_t seqsNum)
{
    LocalTensor<T> inputPositionsLocalInt64 = buffer1.DeQue<T>();
    LocalTensor<T> seqLensLocalInt64 = buffer2.DeQue<T>();
    LocalTensor<T> acceptedNumLocalInt64 = buffer3.DeQue<T>();
    LocalTensor<int32_t> bufferLocal = tempQue_.AllocTensor<int32_t>();
    LocalTensor<int32_t> seqLensLocal = outputInt32Buf_.Get<int32_t>();
    tempQue_.EnQue(bufferLocal);
    bufferLocal = tempQue_.DeQue<int32_t>();

    int bufOffset = 0;
    int elementNum = seqsNum * Align(tokenEachReqs_, sizeof(T));
    if (align) {
        elementNum = seqsNum * (Align(tokenEachReqs_, sizeof(T)) + INT64_ONE_BLOCK_NUM);
    }

    Cast(bufferLocal, inputPositionsLocalInt64, RoundMode::CAST_NONE, elementNum);
    Cast(seqLensLocal, seqLensLocalInt64, RoundMode::CAST_NONE, elementNum);
    PipeBarrier<PIPE_V>();

    if (vectorGather_) {
        // acceptedNum 行展开向量化（原逐行 Adds×seqsNum 次标量化调用为大 shape 主要耗时源）：
        // rowIdx[j]=j/rowStride → ×4 字节偏移 → Gather 展开 acc32 到整块 → 向量加到 positions
        LocalTensor<int32_t> acc32 = acc32Buf_.Get<int32_t>();
        Cast(acc32, acceptedNumLocalInt64, RoundMode::CAST_NONE, static_cast<int32_t>(seqsNum));
        PipeBarrier<PIPE_V>();
        LocalTensor<int32_t> rowIdx = rowIdxBuf_.Get<int32_t>();
        ArithProgression(rowIdx, static_cast<int32_t>(0), static_cast<int32_t>(1), static_cast<int32_t>(elementNum));
        PipeBarrier<PIPE_V>();
        LocalTensor<int32_t> accOff = byteOffBuf_.Get<int32_t>();
        Duplicate(accOff, static_cast<int32_t>(elementNum / seqsNum), elementNum);
        PipeBarrier<PIPE_V>();
#if ADV_VEC_INT_DIV_SUPPORTED
        Div(rowIdx, rowIdx, accOff, elementNum);
#else
        for (int32_t i = 0; i < elementNum; i++) {
            rowIdx.SetValue(i, rowIdx.GetValue(i) / accOff.GetValue(i));
        }
#endif
        PipeBarrier<PIPE_V>();
        Muls(accOff, rowIdx, static_cast<int32_t>(sizeof(int32_t)), static_cast<int32_t>(elementNum));
        PipeBarrier<PIPE_V>();
        // dav_3510 向量 Gather 每次按整 64 lane（256B）LoadAlign 读偏移：elementNum 非 64 整倍数时
        // 尾块 lane 读到偏移缓冲的陈旧值，作为 gather 索引即 UB 越界读（实测 bs=1 且 te≠4 组合
        // 触发 vector core exception）。尾块偏移显式补 0（索引 0 恒界内），byteOffBuf_ 尾部
        // 对齐槽覆盖 align64(elementNum)×4B 的读取范围
        const int32_t elementNumAligned = static_cast<int32_t>((elementNum + GATHER_LANES_PER_REPEAT - 1) /
                                                               GATHER_LANES_PER_REPEAT * GATHER_LANES_PER_REPEAT);
        if (elementNumAligned > elementNum) {
            Duplicate(accOff[elementNum], static_cast<int32_t>(0), elementNumAligned - elementNum);
            PipeBarrier<PIPE_V>();
        }
        Gather(rowIdx, acc32, accOff.template ReinterpretCast<uint32_t>(), 0u, static_cast<uint32_t>(elementNum));
        PipeBarrier<PIPE_V>();
        Add(bufferLocal, bufferLocal, rowIdx, elementNum);
        PipeBarrier<PIPE_V>();
    } else {
        // 非 950（含 910B/910_93）基线路径：逐行 Adds 展开acceptedNum（master 已验证）
        SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
        for (int64_t i = 0; i < seqsNum; i++) {
            int32_t acceptedNumI = static_cast<int32_t>(acceptedNumLocalInt64.GetValue(i));
            Adds(bufferLocal[i * elementNum / seqsNum], bufferLocal[i * elementNum / seqsNum], acceptedNumI,
                 tokenEachReqs_);
        }
        SetWaitFlag<HardEvent::S_V>(HardEvent::S_V);
        PipeBarrier<PIPE_V>();
    }
    // inputPositions += 1
    Adds(bufferLocal, bufferLocal, 1, elementNum);
    PipeBarrier<PIPE_V>();
    ;
    // seqLens = inputPositions + 1
    Adds(seqLensLocal, bufferLocal, 1, elementNum);
    PipeBarrier<PIPE_V>();
    ;
    Cast(inputPositionsLocalInt64, bufferLocal, RoundMode::CAST_NONE, elementNum);
    Cast(seqLensLocalInt64, seqLensLocal, RoundMode::CAST_NONE, elementNum);
    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);

    buffer1.EnQue(inputPositionsLocalInt64);
    buffer2.EnQue(seqLensLocalInt64);
    tempQue_.EnQue(bufferLocal);
    buffer3.EnQue(acceptedNumLocalInt64);
}

template <typename T>
__aicore__ inline void KernelAdvanceStepSpec<T>::CopyOut(int64_t loop, int64_t seqsNum)
{
    LocalTensor<T> inputPositionsLocal = buffer1.DeQue<T>();
    LocalTensor<T> seqLensLocal = buffer2.DeQue<T>();
    // 搬运inputPositions、seqLens、slotMapping，shape相同，用同一个dataCopyParams
    DataCopyExtParams dataCopyParams{static_cast<uint16_t>(seqsNum), static_cast<uint32_t>(tokenEachReqs_ * sizeof(T)),
                                     0, 0, 0};
    if (align) {
        dataCopyParams.srcStride = 1;
    }
    int64_t offset = loop * perLoopSeqs_ * tokenEachReqs_;
    DataCopyPad(inputPositionsGm_[offset], inputPositionsLocal, dataCopyParams);
    DataCopyPad(seqLensGm_[offset], seqLensLocal, dataCopyParams);
    SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
    buffer1.EnQue(inputPositionsLocal);
    buffer2.EnQue(seqLensLocal);
}

template <typename T>
__aicore__ inline void KernelAdvanceStepSpec<T>::ComputeSlotMapping(int64_t loop, int64_t seqsNum)
{
    LocalTensor<T> slotMappingLocalInt64 = buffer2.DeQue<T>();
    LocalTensor<int32_t> inputPositionsLocal = tempQue_.DeQue<int32_t>();
    LocalTensor<int32_t> bufferLocal = buffer3.DeQue<int32_t>();

    int64_t rowStride = Align(tokenEachReqs_, sizeof(T));
    if (align) {
        rowStride += INT64_ONE_BLOCK_NUM;
    }
    const int32_t elementNum = static_cast<int32_t>(seqsNum * rowStride);
    LocalTensor<int32_t> slotMappingLocal = outputInt32Buf_.Get<int32_t>();
    Cast(slotMappingLocal, slotMappingLocalInt64, RoundMode::CAST_NONE, elementNum);
    PipeBarrier<PIPE_V>();

    const int64_t blockTablesOffset = loop * perLoopSeqs_ * blockTablesStride_;
    if (vectorGather_) {
        // q = inputPositions / blockSize：除数广播到 slotMappingLocal 复用暂存（向量除）
        Duplicate(slotMappingLocal, static_cast<int32_t>(blockSize_), elementNum);
        PipeBarrier<PIPE_V>();
#if ADV_VEC_INT_DIV_SUPPORTED
        Div(bufferLocal, inputPositionsLocal, slotMappingLocal, elementNum);
#else
        for (int32_t i = 0; i < elementNum; i++) {
            bufferLocal.SetValue(i, inputPositionsLocal.GetValue(i) / slotMappingLocal.GetValue(i));
        }
#endif
        PipeBarrier<PIPE_V>();

        // flat 下标 = q + blockTablesStride * 行号：向量化（原逐行 Adds×seqsNum 次标量化调用为
        // 大 shape 主要耗时源）。行号张量 rowIdx[j]=j/rowStride 经 ArithProgression+Div 重建
        LocalTensor<int32_t> rowIdx = rowIdxBuf_.Get<int32_t>();
        ArithProgression(rowIdx, static_cast<int32_t>(0), static_cast<int32_t>(1), static_cast<int32_t>(elementNum));
        PipeBarrier<PIPE_V>();
        LocalTensor<int32_t> rowBase = byteOffBuf_.Get<int32_t>();
        Duplicate(rowBase, static_cast<int32_t>(rowStride), elementNum);
        PipeBarrier<PIPE_V>();
#if ADV_VEC_INT_DIV_SUPPORTED
        Div(rowIdx, rowIdx, rowBase, elementNum);
#else
        for (int32_t i = 0; i < elementNum; i++) {
            rowIdx.SetValue(i, rowIdx.GetValue(i) / rowBase.GetValue(i));
        }
#endif
        PipeBarrier<PIPE_V>();
        Muls(rowBase, rowIdx, static_cast<int32_t>(blockTablesStride_), static_cast<int32_t>(elementNum));
        PipeBarrier<PIPE_V>();
        Add(slotMappingLocal, bufferLocal, rowBase, elementNum);
        PipeBarrier<PIPE_V>();

        // 行内列掩码 qMask = 1 + (q>>31) + ((stride-1-q)>>31)：q∈[0, stride) 为 1，越列（含
        // 跨行平坦下标与负值）为 0。rowIdx/rowBase 此后不再需要，rowIdxBuf_ 转存掩码，
        // 供末段对最终 slot 整体屏蔽（块号与余数同时归零，兑现"越界防御性写 0"契约）
        LocalTensor<int32_t> qMask = rowIdx;
        ShiftRight(qMask, bufferLocal, static_cast<int32_t>(31), elementNum);
        PipeBarrier<PIPE_V>();
        LocalTensor<int32_t> tmpOff = rowBase;
        Duplicate(tmpOff, static_cast<int32_t>(blockTablesStride_ - 1), elementNum);
        PipeBarrier<PIPE_V>();
        Sub(tmpOff, tmpOff, bufferLocal, elementNum);
        PipeBarrier<PIPE_V>();
        ShiftRight(tmpOff, tmpOff, static_cast<int32_t>(31), elementNum);
        PipeBarrier<PIPE_V>();
        Add(qMask, qMask, tmpOff, elementNum);
        PipeBarrier<PIPE_V>();
        Adds(qMask, qMask, static_cast<int32_t>(1), elementNum);
        PipeBarrier<PIPE_V>();

        // mod = inputPositions - q * blockSize（q 此后不再需要，bufferLocal 原地转存 mod 供末段相加）
        Muls(bufferLocal, bufferLocal, static_cast<int32_t>(blockSize_), elementNum);
        PipeBarrier<PIPE_V>();
        Sub(bufferLocal, inputPositionsLocal, bufferLocal, elementNum);
        PipeBarrier<PIPE_V>();

        // gather blockTables[flat] 查表回退标量防护路径：dav_3510 预编译 high_performance 二进制上，
        // Level-2 向量 Gather（VfGatherApi2B32 的 LoadAlign 链路）对特定索引模式触发
        // vector core exception（507035，数据依赖：实测 bs=1 且 te≠4 的 15 例稳定复现，
        // kernel JIT 编译同源同 tiling 则通过；标量查表替换后同数据全部通过）。
        // flat/qMask/mod 仍走向量计算（批次 3 优化保留），仅查表降级为逐 lane 标量 GM 读
        // （int32 低 32 位截断，与原实现同语义）；行内列校验 0 ≤ q < stride 与原实现一致
        // （越列含跨行平坦下标与负值防御性写 0、无 GM 越界读）
        GatherRowsScalar(slotMappingLocal, blockTablesOffset, 0, seqsNum, rowStride);

        // slot = bt 值 * blockSize + mod，再经 qMask 整体屏蔽：越列 lane（含跨行平坦
        // 下标、负位置）slot=0（块号与余数同时归零）；分块包络内读到的他行块号同样被掩
        SetWaitFlag<HardEvent::S_V>(HardEvent::S_V);
        Muls(slotMappingLocal, slotMappingLocal, static_cast<int32_t>(blockSize_), elementNum);
        PipeBarrier<PIPE_V>();
        Add(slotMappingLocal, bufferLocal, slotMappingLocal, elementNum);
        PipeBarrier<PIPE_V>();
        Mul(slotMappingLocal, slotMappingLocal, qMask, elementNum);
        PipeBarrier<PIPE_V>();
    } else {
        // 非 950（910B/910_93 等）基线路径：master 已验证的逐行 flat 构造 + 逐元素查表；
        // 查表带行内列校验 0 ≤ q < stride（越列含跨行平坦下标与负值防御性 slot 写 0，
        // 块号与余数同时屏蔽，与向量路径 qMask 语义一致）
        ArithProgression<int32_t>(bufferLocal, static_cast<int32_t>(0), static_cast<int32_t>(blockTablesStride_),
                                  static_cast<int32_t>(seqsNum));
        SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
        PipeBarrier<PIPE_V>();
        for (int64_t i = 0; i < seqsNum; i++) {
            int32_t bufferLocalI = static_cast<int32_t>(bufferLocal.GetValue(i));
            Duplicate(slotMappingLocal[i * rowStride], bufferLocalI, tokenEachReqs_);
        }
        PipeBarrier<PIPE_V>();
        for (int32_t i = 0; i < elementNum; i++) {
            bufferLocal.SetValue(i, inputPositionsLocal.GetValue(i) / static_cast<int32_t>(blockSize_));
        }
        SetWaitFlag<HardEvent::S_V>(HardEvent::S_V);
        Add(bufferLocal, slotMappingLocal, bufferLocal, elementNum);
        SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
        PipeBarrier<PIPE_V>();
        for (int64_t i = 0; i < seqsNum; ++i) {
            for (int64_t j = 0; j < tokenEachReqs_; ++j) {
                const int64_t index = i * rowStride + j;
                const int64_t pos = inputPositionsLocal.GetValue(index);
                const int32_t q = static_cast<int32_t>(bufferLocal.GetValue(index)) -
                                  static_cast<int32_t>(i * blockTablesStride_);
                int32_t slot = 0;
                if (q >= 0 && q < blockTablesStride_) {
                    const int32_t v = static_cast<int32_t>(
                        blockTablesGm_[blockTablesOffset + bufferLocal.GetValue(index)].GetValue(0));
                    slot = v * static_cast<int32_t>(blockSize_) + static_cast<int32_t>(pos - q * blockSize_);
                }
                slotMappingLocal.SetValue(index, slot);
            }
        }
        SetWaitFlag<HardEvent::S_V>(HardEvent::S_V);
    }
    Cast(slotMappingLocalInt64, slotMappingLocal, RoundMode::CAST_NONE, elementNum);
    SetWaitFlag<HardEvent::V_MTE3>(HardEvent::V_MTE3);

    buffer2.EnQue(slotMappingLocalInt64);
    buffer3.EnQue(bufferLocal);
    tempQue_.FreeTensor(inputPositionsLocal);
}

template <typename T>
__aicore__ inline void KernelAdvanceStepSpec<T>::GatherRowsScalar(const LocalTensor<int32_t>& lanesTensor,
                                                                  int64_t blockTablesOffset, int64_t rowStart,
                                                                  int64_t rowCount, int64_t rowStride)
{
    // 标量防护查表（向量 gather 回退后的 V2 大 shape 主要耗时段）：行内列校验
    // 0 ≤ q < blockTablesStride（q = flat − stride×行号）——越列/负值写 0 且不读 GM；
    // 行内蕴含核内视图界内，无 GM 越界读；最终 slot 另经 qMask 整体屏蔽。
    // 全程使用带地址空间属性的张量标量 API（GetValue/SetValue）+ 行基址外提：
    // 禁止 reinterpret_cast(GetPhyAddr()) 裸指针（丢 __ubuf__ 属性，high_performance
    // 预编译实测触发 vector core exception，见 te==2 分支注释）
    SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
    const int64_t stride = blockTablesStride_;
    if (tokenEachReqs_ == 2) {
        // te==2 双 lane 特化 + 8 行批量软件流水（V2 大 shape 主流形态）：标量 GM 读为延迟主导
        // （实测 ~100ns/lane，编译器不重叠独立读），按 8 行一组先算齐 16 个地址再集中发射 16 个
        // 独立 GM 读，时延摊薄一个批次。越界 lane 以钳位地址（窗口基址）无条件读替代分支，
        // 其取值随后被 qMask 整体屏蔽归零（语义与写 0 严格一致），批次内全程无分支。
        // 全程走带地址空间属性的张量标量 API：裸指针 reinterpret_cast(GetPhyAddr()) 丢失
        // __ubuf__ 属性，high_performance 预编译下实测触发 vector core exception，禁止使用
        constexpr int64_t ROW_BATCH = 8;
        int64_t i = rowStart;
        for (; i + ROW_BATCH <= rowStart + rowCount; i += ROW_BATCH) {
            int64_t off[ROW_BATCH * 2];
            int32_t val[ROW_BATCH * 2];
            for (int64_t k = 0; k < ROW_BATCH; ++k) {
                const int64_t rowBase = (i + k) * stride;
                const uint32_t laneBase = static_cast<uint32_t>((i + k) * rowStride);
                const int64_t o0 = lanesTensor.GetValue(laneBase);
                const int64_t o1 = lanesTensor.GetValue(laneBase + 1);
                off[2 * k] = static_cast<uint64_t>(o0 - rowBase) < static_cast<uint64_t>(stride) ? o0 : 0;
                off[2 * k + 1] = static_cast<uint64_t>(o1 - rowBase) < static_cast<uint64_t>(stride) ? o1 : 0;
            }
            for (int64_t k = 0; k < ROW_BATCH * 2; ++k) {
                val[k] = static_cast<int32_t>(blockTablesGm_.GetValue(blockTablesOffset + off[k]));
            }
            for (int64_t k = 0; k < ROW_BATCH; ++k) {
                const uint32_t laneBase = static_cast<uint32_t>((i + k) * rowStride);
                lanesTensor.SetValue(laneBase, val[2 * k]);
                lanesTensor.SetValue(laneBase + 1, val[2 * k + 1]);
            }
        }
        for (; i < rowStart + rowCount; ++i) {
            const int64_t rowBase = i * stride;
            const uint32_t laneBase = static_cast<uint32_t>(i * rowStride);
            const int64_t o0 = lanesTensor.GetValue(laneBase);
            const int64_t o1 = lanesTensor.GetValue(laneBase + 1);
            const int64_t s0 = static_cast<uint64_t>(o0 - rowBase) < static_cast<uint64_t>(stride) ? o0 : 0;
            const int64_t s1 = static_cast<uint64_t>(o1 - rowBase) < static_cast<uint64_t>(stride) ? o1 : 0;
            lanesTensor.SetValue(laneBase, static_cast<int32_t>(blockTablesGm_.GetValue(blockTablesOffset + s0)));
            lanesTensor.SetValue(laneBase + 1, static_cast<int32_t>(blockTablesGm_.GetValue(blockTablesOffset + s1)));
        }
        SetWaitFlag<HardEvent::S_V>(HardEvent::S_V);
        return;
    }
    for (int64_t i = rowStart; i < rowStart + rowCount; ++i) {
        const int64_t rowBase = i * stride;
        const uint32_t laneBase = static_cast<uint32_t>(i * rowStride);
        for (int64_t j = 0; j < tokenEachReqs_; ++j) {
            const int64_t offset = lanesTensor.GetValue(laneBase + static_cast<uint32_t>(j));
            const int64_t q = offset - rowBase;
            int32_t v = 0;
            if (static_cast<uint64_t>(q) < static_cast<uint64_t>(stride)) {
                v = static_cast<int32_t>(blockTablesGm_.GetValue(blockTablesOffset + offset));
            }
            lanesTensor.SetValue(laneBase + static_cast<uint32_t>(j), v);
        }
    }
    SetWaitFlag<HardEvent::S_V>(HardEvent::S_V);
}

template <typename T>
__aicore__ inline void KernelAdvanceStepSpec<T>::CopyOutSlotMapping(int64_t loop, int64_t seqsNum)
{
    LocalTensor<T> slotMappingLocal = buffer2.DeQue<T>();
    int elementNum = seqsNum * tokenEachReqs_;
    DataCopyExtParams dataCopyParams{static_cast<uint16_t>(seqsNum), static_cast<uint32_t>(tokenEachReqs_ * sizeof(T)),
                                     0, 0, 0};
    if (align) {
        dataCopyParams.srcStride = 1;
    }
    int64_t offset = loop * perLoopSeqs_ * tokenEachReqs_;
    DataCopyPad(slotMappingGm_[offset], slotMappingLocal, dataCopyParams);
    SetWaitFlag<HardEvent::MTE3_MTE2>(HardEvent::MTE3_MTE2);
    buffer2.EnQue(slotMappingLocal);
}

template <typename T>
__aicore__ inline void KernelAdvanceStepSpec<T>::CopyInInputTokens(int64_t loop, int64_t seqsNum)
{
    LocalTensor<T> sampledTokenIdsLocal = buffer1.DeQue<T>();
    LocalTensor<T> specTokenLocal = buffer2.DeQue<T>();

    // 搬运sampledTokenIds，其第2维是specNum+1
    int64_t offset = loop * perLoopSeqs_ * tokenEachReqs_;
    DataCopyExtParams sampledTokenIdsDataCopyParams{static_cast<uint16_t>(seqsNum),
                                                    static_cast<uint32_t>(tokenEachReqs_ * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> sampledTokenIdsDataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(sampledTokenIdsLocal, sampledTokenIdsGm_[offset], sampledTokenIdsDataCopyParams,
                sampledTokenIdsDataCopyPadParams);

    // 搬运specToken，其第2维是specNum
    offset = loop * perLoopSeqs_ * specNum_;
    DataCopyExtParams specTokenDataCopyParams{static_cast<uint16_t>(seqsNum),
                                              static_cast<uint32_t>(specNum_ * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> specTokenDataCopyPadParams{false, 0, 0, 0};
    DataCopyPad(specTokenLocal, specTokenGm_[offset], specTokenDataCopyParams, specTokenDataCopyPadParams);

    buffer1.EnQue(sampledTokenIdsLocal);
    buffer2.EnQue(specTokenLocal);
}

template <typename T>
__aicore__ inline void KernelAdvanceStepSpec<T>::ComputeInputTokens(int64_t loop, int64_t seqsNum)
{
    LocalTensor<T> sampledTokenIdsLocalInt64 = buffer1.DeQue<T>();
    LocalTensor<T> lastTokensInt64 = buffer3.DeQue<T>();

    LocalTensor<float> sampledTokenIdsLocalFp32 = sampledTokenIdsBuf_.Get<float>();
    uint64_t stride = Align(tokenEachReqs_, sizeof(T));
    SetWaitFlag<HardEvent::MTE3_V>(HardEvent::MTE3_V);
    for (uint64_t i = 0; i < seqsNum; ++i) {
        // sampledTokenIds末尾补-1
        Cast(sampledTokenIdsLocalFp32, sampledTokenIdsLocalInt64[i * stride], RoundMode::CAST_RINT, tokenEachReqs_);
        SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
        sampledTokenIdsLocalFp32.SetValue(tokenEachReqs_, -1);
        LocalTensor<float> temp = sampledTokenIdsLocalFp32;
        SetWaitFlag<HardEvent::S_V>(HardEvent::S_V);
        // 取出sampledTokenIds一行的第一个-1的索引，将lastTokens存放在bufferLocal中
        ReduceMin(temp, temp, temp, specNum_ + 2, true);
        SetWaitFlag<HardEvent::V_S>(HardEvent::V_S);
        float tempIndex = temp.GetValue(1);
        uint32_t minIndex = *reinterpret_cast<uint32_t*>(&tempIndex);
        // bufferLocal存放的是lastTokens；行首即为-1表示该request无接受token，lastToken为-1
        int64_t lastToken = -1;
        if (minIndex != 0) {
            lastToken = sampledTokenIdsLocalInt64.GetValue(i * stride + minIndex - 1);
        }
        lastTokensInt64.SetValue(Align(1, sizeof(T)) * i, lastToken);
        PipeBarrier<PIPE_V>();
        ;
    }
    // lastTokens 由标量（S 管线）写入，随后在 CopyOutInputTokens 中经 MTE3 拷出，
    // 队列事件不覆盖 S→MTE3 依赖，需显式同步（参照 layer_norm_v4_single_read.h 同步范式）
    SetWaitFlag<HardEvent::S_MTE3>(HardEvent::S_MTE3);
    buffer3.EnQue(lastTokensInt64);
    buffer1.FreeTensor(sampledTokenIdsLocalInt64);
}

template <typename T>
__aicore__ inline void KernelAdvanceStepSpec<T>::CopyOutInputTokens(int64_t loop, int64_t seqsNum)
{
    LocalTensor<T> specTokenLocal = buffer2.DeQue<T>();
    LocalTensor<T> lastTokensLocal = buffer3.DeQue<T>();
    // 搬运lastToken到inputTokens的第1列
    int64_t offset = loop * perLoopSeqs_ * tokenEachReqs_;
    DataCopyExtParams lastTokenDataCopyParams{static_cast<uint16_t>(seqsNum), static_cast<uint32_t>(sizeof(T)), 0,
                                              static_cast<uint32_t>((specNum_) * sizeof(T)), 0};
    DataCopyPad(inputTokensGm_[offset], lastTokensLocal, lastTokenDataCopyParams);

    // // 搬运specToke到inputTokens的后specNum列
    DataCopyExtParams specTokenDataCopyParams{static_cast<uint16_t>(seqsNum),
                                              static_cast<uint32_t>(specNum_ * sizeof(T)), 0,
                                              static_cast<uint32_t>(sizeof(T)), 0};
    DataCopyPad(inputTokensGm_[offset + 1], specTokenLocal, specTokenDataCopyParams);

    buffer2.FreeTensor(specTokenLocal);
    buffer3.FreeTensor(lastTokensLocal);
}

template <typename T>
__aicore__ inline void KernelAdvanceStepSpec<T>::Process()
{
    // 每个循环处理perLoopSeqs_行
    int64_t i = 0;
    for (; i < curCoreLoopNum_; i++) {
        CopyIn(i, perLoopSeqs_);
        Compute(i, perLoopSeqs_);
        CopyOut(i, perLoopSeqs_);
        ComputeSlotMapping(i, perLoopSeqs_);
        CopyOutSlotMapping(i, perLoopSeqs_);
        CopyInInputTokens(i, perLoopSeqs_);
        ComputeInputTokens(i, perLoopSeqs_);
        CopyOutInputTokens(i, perLoopSeqs_);
    }
    if (lastLoopSeqs_ > 0) {
        CopyIn(i, lastLoopSeqs_);
        Compute(i, lastLoopSeqs_);
        CopyOut(i, lastLoopSeqs_);
        ComputeSlotMapping(i, lastLoopSeqs_);
        CopyOutSlotMapping(i, lastLoopSeqs_);
        CopyInInputTokens(i, lastLoopSeqs_);
        ComputeInputTokens(i, lastLoopSeqs_);
        CopyOutInputTokens(i, lastLoopSeqs_);
    }
}
} // namespace AdvanceStepNs
#endif // ADVANCE_STEP_SPEC_H
