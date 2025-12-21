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
 * \file fused_linear_cross_entropy_loss_grad_high_perf.h
 * \brief
 */
#ifndef FUSED_LINEAR_CROSS_ENTROPY_LOSS_GRAD_HIGH_PERF_H
#define FUSED_LINEAR_CROSS_ENTROPY_LOSS_GRAD_HIGH_PERF_H

#include <cstdint>
#include <type_traits>
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "fused_linear_cross_entropy_loss_grad_config.h"
#ifdef __CCE_KT_TEST__
#include "fused_linear_cross_entropy_loss_grad_tiling_data_def.h"
#endif  // __CCE_KT_TEST__

namespace FusedLinearCrossEntropyLossGradNS {
using namespace AscendC;

template <typename MM, typename HalfType, typename TargetMaskType, typename MaskedTargetType>
class FusedLinearCrossEntropyLossGradHighPerf {
public:
    __aicore__ inline FusedLinearCrossEntropyLossGradHighPerf(const FusedLinearCrossEntropyLossGradHighPerfTilingData& tilingData)
        : tilingData(tilingData) {}
    __aicore__ inline void Init(
        GM_ADDR input, GM_ADDR weight, GM_ADDR softmax, GM_ADDR target_mask, GM_ADDR masked_target, GM_ADDR grad_output,
        GM_ADDR grad_input, GM_ADDR grad_weight, GM_ADDR workspace)
    {
        if (blockIdx < tilingData.btLastCoreId) {
            btThisCore = tilingData.btPerCore;
        } else if (blockIdx == tilingData.btLastCoreId) {
            btThisCore = tilingData.btLastCore;
        } else {
            btThisCore = 0U;
        }

        if ASCEND_IS_AIV {
            pipe.Init();
        } else {
            mm1.Init(&tilingData.mm1Tiling, &pipe);
            mm2.Init(&tilingData.mm2Tiling, &pipe);
        }

        softmaxHalfGM.SetGlobalBuffer((__gm__ HalfType *)softmax, tilingData.BT * tilingData.VOut512BAlignedSize);
        auto nzGMSize = tilingData.userWorkspaceSize / tilingData.wsNum;
        for (uint32_t i = 0; i < tilingData.wsNum; i++) {
            A1NZGMs[i].SetGlobalBuffer((__gm__ HalfType *)GetUserWorkspace(workspace) + i * nzGMSize, nzGMSize);
            hasA1NZGMTask[i] = false;
        }

        if ASCEND_IS_AIV {
            mm1ProcessingBaseBlockIdx = ctrlBlockIdx;
            mm2ProcessingBaseBlockIdx = ctrlBlockIdx;

            if (btThisCore) {
                // softmax update
                softmaxForUpdateGM.SetGlobalBuffer((__gm__ float *)softmax + tilingData.V * btOffset, tilingData.V * btThisCore);
                targetMaskForUpdateGM.SetGlobalBuffer((__gm__ TargetMaskType *)target_mask, tilingData.targetMaskSize);  // false就将-1，true就pass。取全长，便于uint8时偏移计算
                maskedTarget1dForUpdateGM.SetGlobalBuffer((__gm__ MaskedTargetType *)masked_target + btOffset, btThisCore);  // 列索引
                pipe.InitBuffer(updateBuf, SIZE_32);
            }

            // mul & cast
            softmaxGM.SetGlobalBuffer((__gm__ float *)softmax, tilingData.BT * tilingData.V);
            gradOutputGM.SetGlobalBuffer((__gm__ float *)grad_output, tilingData.BT);
            pipe.InitBuffer(queBindSoftmax, BUFFER_NUM, tilingData.quebindSoftmaxByteSize);
        } else {
            // mm1
            mm1ProcessingBaseBlockIdx = blockIdx;
            weightGM.SetGlobalBuffer((__gm__ HalfType *)weight, tilingData.V * tilingData.H);
            gradInputGM.SetGlobalBuffer((__gm__ HalfType *)grad_input, tilingData.BT * tilingData.H);
            // mm2
            mm2ProcessingBaseBlockIdx = blockIdx;
            inputGM.SetGlobalBuffer((__gm__ HalfType *)input, tilingData.BT * tilingData.H);
            gradWeightGM.SetGlobalBuffer((__gm__ HalfType *)grad_weight, tilingData.V * tilingData.H);
        }
    }

    __aicore__ inline void Process()
    {
        if ASCEND_IS_AIV {
            ProcessVec();
        } else {
            ProcessCube();
        }
    }

    __aicore__ inline void ProcessVec()
    {
        InitSyncVec();

        // softmax update
        if (btThisCore) {
            ProcessUpdate();
        }
        SyncAllVecMTE3();  // vec前同步，确保update都已完成

        // mul & cast & pad
        ProcessVecFull();

        CloseSyncVec();
        pipe.Destroy();
    }

    __aicore__ inline void ProcessCube()
    {
        InitSyncCube();

        // matmul
        ProcessCubeFull();

        CloseSyncCube();
    }

private:
    __aicore__ inline void InitSyncVec()
    {
        evtIDMTE3ToS = GetTPipePtr()->AllocEventID<HardEvent::MTE3_S>();
        evtIDSToMTE3 = GetTPipePtr()->AllocEventID<HardEvent::S_MTE3>();
        evtIDVToMTE3 = GetTPipePtr()->AllocEventID<HardEvent::V_MTE3>();
    }

    __aicore__ inline void CloseSyncVec()
    {
        GetTPipePtr()->ReleaseEventID<HardEvent::MTE3_S>(evtIDMTE3ToS);
        GetTPipePtr()->ReleaseEventID<HardEvent::S_MTE3>(evtIDSToMTE3);
        GetTPipePtr()->ReleaseEventID<HardEvent::V_MTE3>(evtIDVToMTE3);
    }

    __aicore__ inline void InitSyncCube()
    {
        evtIDFixToMTE2 = GetTPipePtr()->AllocEventID<HardEvent::FIX_MTE2>();
    }

    __aicore__ inline void CloseSyncCube()
    {
        GetTPipePtr()->ReleaseEventID<HardEvent::FIX_MTE2>(evtIDFixToMTE2);
    }

    __aicore__ inline void SyncAllVecMTE2()
    {
        CrossCoreSetFlag<0, PIPE_MTE2>(VecSyncMTE2FlagId);
        CrossCoreWaitFlag(VecSyncMTE2FlagId);
    }

    __aicore__ inline void SyncAllVecMTE3()
    {
        CrossCoreSetFlag<0, PIPE_MTE3>(VecSyncMTE3FlagId);
        CrossCoreWaitFlag(VecSyncMTE3FlagId);
    }

    __aicore__ inline void SyncFixPipe()
    {
        SetFlag<HardEvent::FIX_MTE2>(evtIDFixToMTE2);
        WaitFlag<HardEvent::FIX_MTE2>(evtIDFixToMTE2);
    }

    __aicore__ inline bool GetBoolFromUint8Tensor(uint64_t bitIndex) {
        // 计算目标位所在的字节索引和位偏移
        size_t byteIndex = bitIndex / 8;
        size_t bitOffset = bitIndex % 8;
        uint8_t cachedByte = targetMaskForUpdateGM.GetValue(byteIndex);
        return (cachedByte >> bitOffset) & 0x01;
    }

    __aicore__ inline void ProcessUpdate()
    {
        auto updateUB = updateBuf.Get<float>();
        for (uint64_t i = 0; i < btThisCore; i++) {
            bool targetMask;
            uint64_t btIdx = i + btOffset;
            if constexpr (std::is_same_v<DTYPE_TARGET_MASK, uint8_t>) {
                targetMask = GetBoolFromUint8Tensor(btIdx);
            } else {
                targetMask = targetMaskForUpdateGM.GetValue(btIdx);
            }
            if (!targetMask) {
                auto maskedTarget1d = maskedTarget1dForUpdateGM.GetValue(i);
                auto idx = tilingData.V * i + maskedTarget1d;
                SetFlag<HardEvent::MTE3_S>(evtIDMTE3ToS);
                WaitFlag<HardEvent::MTE3_S>(evtIDMTE3ToS);
                updateUB.SetValue(0, softmaxForUpdateGM.GetValue(idx) - 1.0f);
                SetFlag<HardEvent::S_MTE3>(evtIDSToMTE3);
                WaitFlag<HardEvent::S_MTE3>(evtIDSToMTE3);
                DataCopyExtParams copyParams{1, IN_BYTE_SIZE,
                                             0,
                                             0,
                                             0};
                DataCopyPad(softmaxForUpdateGM[idx], updateUB, copyParams);
            }
        }
    }

    __aicore__ inline void TogetherWaitingNzGM(uint8_t numNzGM, uint16_t flagId)
    {
        if (hasA1NZGMTask[numNzGM]) {
            CrossCoreWaitFlag(flagId);  // 等待workspace的上组mm搬入完毕
            hasA1NZGMTask[numNzGM] = false;
        }
        SyncAllVecMTE2();  // 仅指令位置同步
    }

    __aicore__ inline void SuperviseMM1(uint32_t baseMIdx, uint8_t numNzGM)
    {
        uint32_t readyBaseBlock = (baseMIdx + 1) * tilingData.mm1BaseNCnt;
        SyncAllVecMTE3();  // 输出同步后进入mm1判断和启动流程
        
        while (mm1ProcessingBaseBlockIdx < readyBaseBlock) {
            hasA1NZGMTask[numNzGM] = true;
            mm1ProcessingBaseBlockIdx += tilingData.aicNum;
        }
        if (hasA1NZGMTask[numNzGM]) {
            NotifyMM();
        }
    }

    __aicore__ inline void NotifyMM()
    {
        CrossCoreSetFlag<CVModeId, PIPE_MTE3>(CubeWaitVecFlagId);  // 通知cube启动mm
    }

    __aicore__ inline void ListeningMM1()
    {
        for (uint32_t baseMIdx = 0; baseMIdx < tilingData.mm1BaseMCnt; baseMIdx++) {
            uint32_t readyBaseBlock = (baseMIdx + 1) * tilingData.mm1BaseNCnt;
            uint8_t numNzGM = baseMIdx % tilingData.wsNum;
            bool hasTask = false;

            while (mm1ProcessingBaseBlockIdx < readyBaseBlock) {
                if (not hasTask) {
                    CrossCoreWaitFlag(CubeWaitVecFlagId);  // 等待启动mm
                    hasTask = true;
                }
                LaunchMM1(baseMIdx, numNzGM);
                mm1ProcessingBaseBlockIdx += tilingData.aicNum;
            }
            if (hasTask) {
                CrossCoreSetFlag<CVModeId, PIPE_MTE2>(mmSingleAReadFlagId);  // 通知vec本块workspace搬入完毕
            }
        }
    }

    __aicore__ inline void LaunchMM1(uint32_t baseMIdx, uint8_t numNzGM)
    {
        // launch current baseblock
        uint32_t baseNIdx = mm1ProcessingBaseBlockIdx % tilingData.mm1BaseNCnt;
        auto& gmA = A1NZGMs[numNzGM];
        auto gmB = weightGM[baseNIdx * tilingData.mm1Tiling.baseN];
        auto gmC = gradInputGM[baseMIdx * tilingData.mm1Tiling.baseM * tilingData.H + baseNIdx * tilingData.mm1Tiling.baseN];
        int singleCoreM = (baseMIdx == tilingData.mm1BaseMCnt - 1) ? tilingData.mm1TailM : tilingData.mm1Tiling.singleCoreM;
        int singleCoreN = (baseNIdx == tilingData.mm1BaseNCnt - 1) ? tilingData.mm1TailN : tilingData.mm1Tiling.singleCoreN;
        mm1.SetTensorA(gmA, false);
        mm1.SetTensorB(gmB, false);
        mm1.SetSingleShape(singleCoreM, singleCoreN, tilingData.mm1Tiling.singleCoreK);
        mm1.template IterateAll<false>(gmC);
    }

    __aicore__ inline void WaitMM1()
    {
        mm1.End();
    }

    __aicore__ inline void SuperviseMM2(uint32_t baseMIdx, uint8_t numNzGM)
    {
        uint32_t readyBaseBlock = (baseMIdx + 1) * tilingData.mm2BaseNCnt;
        SyncAllVecMTE3();  // 输出同步后进入mm2判断和启动流程
        
        while (mm2ProcessingBaseBlockIdx < readyBaseBlock) {
            hasA1NZGMTask[numNzGM] = true;
            mm2ProcessingBaseBlockIdx += tilingData.aicNum;
        }
        if (hasA1NZGMTask[numNzGM]) {
            NotifyMM();
        }
    }
    
    __aicore__ inline void ListeningMM2()
    {
        uint8_t nextNumNzGM = tilingData.mm1BaseMCnt % tilingData.wsNum;
        for (uint32_t baseMIdx = 0; baseMIdx < tilingData.mm2BaseMCnt; baseMIdx++) {
            uint32_t readyBaseBlock = (baseMIdx + 1) * tilingData.mm2BaseNCnt;
            uint8_t numNzGM = (baseMIdx + nextNumNzGM) % tilingData.wsNum;
            bool hasTask = false;

            while (mm2ProcessingBaseBlockIdx < readyBaseBlock) {
                if (not hasTask) {
                    CrossCoreWaitFlag(CubeWaitVecFlagId);  // 等待启动mm
                    hasTask = true;
                }
                LaunchMM2(baseMIdx, numNzGM);
                mm2ProcessingBaseBlockIdx += tilingData.aicNum;
            }
            if (hasTask) {
                CrossCoreSetFlag<CVModeId, PIPE_MTE2>(mmSingleAReadFlagId);  // 通知vec本块workspace搬入完毕
            }
        }
    }

    __aicore__ inline void LaunchMM2(uint32_t baseMIdx, uint8_t numNzGM)
    {
        // launch current baseblock
        uint32_t baseNIdx = mm2ProcessingBaseBlockIdx % tilingData.mm2BaseNCnt;
        auto& gmA = A1NZGMs[numNzGM];
        auto gmB = inputGM[baseNIdx * tilingData.mm2Tiling.baseN];
        auto gmC = gradWeightGM[baseMIdx * tilingData.mm2Tiling.baseM * tilingData.H + baseNIdx * tilingData.mm2Tiling.baseN];
        mm2.SetTensorA(gmA, true);
        mm2.SetTensorB(gmB, false);
        int singleCoreM = (baseMIdx == tilingData.mm2BaseMCnt - 1) ? tilingData.mm2TailM : tilingData.mm2Tiling.singleCoreM;
        int singleCoreN = (baseNIdx == tilingData.mm2BaseNCnt - 1) ? tilingData.mm2TailN : tilingData.mm2Tiling.singleCoreN;
        mm2.SetSingleShape(singleCoreM, singleCoreN, tilingData.mm2Tiling.singleCoreK);
        mm2.template IterateAll<false>(gmC);
    }

// -------------------------------------------------- 每轮流水能处理1行及以上 --------------------------------------------------
    __aicore__ inline void ProcessVecFull()
    {
        // 标量准备
        C0BlockCnt = (tilingData.V - 1) / OUT_SIZE_PER_32B + 1;
        C0ZeroSize = C0BlockCnt * OUT_SIZE_PER_32B - tilingData.V;
        if (C0ZeroSize) {
            C0TailOffset = (C0BlockCnt - 1) * tilingData.mm1Tiling.baseM * OUT_SIZE_PER_32B + (OUT_SIZE_PER_32B - C0ZeroSize);
        }
        C0BlockCntPerBaseM = tilingData.mm2Tiling.baseM / OUT_SIZE_PER_32B;

        uint32_t epochCnt = tilingData.epochCnt;  // 处理当前singleA所需轮数
        int64_t realM = tilingData.mm1Tiling.baseM;  // 当前singleA的行数
        // mm1 part
        for (uint32_t baseMIdx = 0; baseMIdx < tilingData.mm1BaseMCnt; baseMIdx++) {
            uint8_t numNzGM = baseMIdx % tilingData.wsNum;  // 当前singleA要输出的workspace序号
            if (baseMIdx >= tilingData.wsNum) {
                TogetherWaitingNzGM(numNzGM, mmSingleAReadFlagId);
            }
            if (baseMIdx == tilingData.mm1BaseMCnt - 1){
                epochCnt = tilingData.lastEpochCnt;
                realM = tilingData.mm1TailM;
            }
            uint32_t FinishedSingleARowCnt = baseMIdx * tilingData.mm1Tiling.baseM;  // 已处理的singleA包含的行数
            uint32_t epochRowOffset = blockIdx * tilingData.btPerEpochSingle;  // 即将处理的起始行在当前轮内的偏移

            for (uint32_t epoch = 0; epoch < epochCnt; epoch++) {
                uint32_t singleARowOffset = epoch * tilingData.btPerEpochTotal + epochRowOffset;  // 即将处理的起始行在当前singleA内的偏移
                uint32_t startRow = FinishedSingleARowCnt + singleARowOffset;  // 当轮的起始行号
                uint32_t rowCnt = tilingData.btPerEpochSingle;  // 即将处理的行数
                if (epoch == epochCnt - 1) {
                    int64_t remainingCnt = realM - (int64_t)singleARowOffset;  // 未处理行数
                    if (remainingCnt > 0) {
                        if (remainingCnt < rowCnt) rowCnt = remainingCnt;
                        CopyInVecFull(startRow, rowCnt);
                        ComputeVecFull(startRow, rowCnt);
                        SyncAllVecMTE2();  // vec核心每个epoch同步，在确保都完成当轮GM读入后再写出，以防非法踩踏
                        CopyOutVecFull(startRow, singleARowOffset, rowCnt, realM, numNzGM);
                    } else {
                        SyncAllVecMTE2();  // 配合其他轮次
                    }
                } else {
                    CopyInVecFull(startRow, rowCnt);
                    ComputeVecFull(startRow, rowCnt);
                    SyncAllVecMTE2();  // vec核心每个epoch同步，在确保都完成当轮GM读入后再写出，以防非法踩踏
                    CopyOutVecFull(startRow, singleARowOffset, rowCnt, realM, numNzGM);
                }
            }
            SuperviseMM1(baseMIdx, numNzGM);
        }
        // mm2 part
        uint8_t nextNumNzGM = tilingData.mm1BaseMCnt % tilingData.wsNum;
        for (uint32_t baseMIdx = 0; baseMIdx < tilingData.mm2BaseMCnt; baseMIdx++) {
            uint8_t numNzGM = (baseMIdx + nextNumNzGM) % tilingData.wsNum;  // 当前singleA要输出的workspace序号
            TogetherWaitingNzGM(numNzGM, mmSingleAReadFlagId);
            uint32_t mm2BaseCurrVec = 0;
            if (blockIdx < tilingData.mm2BaseLastVecId) {
                mm2BaseCurrVec = tilingData.mm2BasePerVec;
            } else if (blockIdx == tilingData.mm2BaseLastVecId) {
                mm2BaseCurrVec = tilingData.mm2BaseLastVec;
            }
            for (int64_t innerBaseIdx = 0; innerBaseIdx < mm2BaseCurrVec; innerBaseIdx += SIZE_2) {
                // 处理最多基本块，搬入、搬出
                uint32_t baseKIdx = tilingData.mm2BasePerVec * blockIdx + innerBaseIdx;
                uint32_t mm2BaseCurrent = (mm2BaseCurrVec - innerBaseIdx > 1) ? SIZE_2 : 1;
                ProcessMM2Nd2nzFull(baseMIdx, baseKIdx, numNzGM, mm2BaseCurrent);
            }
            SuperviseMM2(baseMIdx, numNzGM);
        }
    }

    __aicore__ inline void ProcessCubeFull()
    {
        ListeningMM1();
        WaitMM1();
        SyncFixPipe();
        ListeningMM2();
    }

    __aicore__ inline void CopyInVecFull(uint32_t startRow, uint16_t rowCnt)
    {
        auto softmaxUB = queBindSoftmax.AllocTensor<float>();
        DataCopyExtParams copyParams{rowCnt, (uint32_t)(tilingData.V * IN_BYTE_SIZE),
                                     0,
                                     tilingData.copyInDstStride,
                                     0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        DataCopyPad(softmaxUB, softmaxGM[startRow * tilingData.V], copyParams, padParams);
        queBindSoftmax.EnQue<TPosition::GM, TPosition::VECIN, float>(softmaxUB);
    }

    __aicore__ inline void ComputeVecFull(uint32_t startRow, uint32_t rowCnt)
    {
        auto softmaxUB = queBindSoftmax.DeQue<TPosition::GM, TPosition::VECIN, float>();
        LocalTensor<HalfType> softmaxHalfUB = softmaxUB.ReinterpretCast<HalfType>();
        MulVecFull(startRow, rowCnt, softmaxUB);
        PipeBarrier<PIPE_V>();
        CastVecFull(rowCnt, softmaxUB, softmaxHalfUB);
        queBindSoftmax.EnQue<TPosition::VECOUT, TPosition::GM, HalfType>(softmaxHalfUB);
    }

    __aicore__ inline void MulVecFull(uint32_t startRow, uint32_t rowCnt, LocalTensor<float>& softmaxUB)
    {
        for (uint32_t i = 0; i < rowCnt; i++) {
            auto gradOutput = gradOutputGM.GetValue(startRow + i);
            Muls(softmaxUB[i * tilingData.V64BAlignedSize], softmaxUB[i * tilingData.V64BAlignedSize], gradOutput, tilingData.V);
        }
    }

    __aicore__ inline void CastVecFull(uint32_t rowCnt, LocalTensor<float>& softmaxUB, LocalTensor<HalfType> &softmaxHalfUB)
    {
        Cast(softmaxHalfUB, softmaxUB, RoundMode::CAST_RINT, rowCnt * tilingData.V64BAlignedSize);
    }

    __aicore__ inline void CopyOutVecFull(uint32_t startRow, uint32_t singleARowOffset, uint16_t rowCnt, uint32_t realM, uint8_t numNzGM)
    {
        auto softmaxHalfUB = queBindSoftmax.DeQue<TPosition::VECOUT, TPosition::GM, HalfType>();
        DataCopyExtParams copyParams{rowCnt, (uint32_t)(tilingData.V * OUT_BYTE_SIZE),
                                     0,
                                     tilingData.copyOutDstStride,
                                     0};
        DataCopyPad(softmaxHalfGM[startRow * tilingData.VOut512BAlignedSize], softmaxHalfUB, copyParams);
        
        // 转nz搬出到workspace，用于启动第一个mm
        DataCopyParams repeatParams{
            C0BlockCnt,
            1,
            0,
            (uint16_t)(tilingData.mm1Tiling.baseM - 1)
        };
        GlobalTensor<HalfType>& A1NZGM = A1NZGMs[numNzGM];
        for (uint32_t i = 0; i < rowCnt; i++) {
            uint64_t nzOffset = (singleARowOffset + i) * OUT_SIZE_PER_32B;  // 输出WorkSpace偏移
            DataCopy(A1NZGM[nzOffset], softmaxHalfUB[tilingData.VOut32BAlignedSize * i], repeatParams);
        }
        queBindSoftmax.FreeTensor(softmaxHalfUB);
    }

    __aicore__ inline void ProcessMM2Nd2nzFull(uint32_t baseMIdx, uint32_t baseKIdx, uint8_t numNzGM, uint32_t mm2BaseCurrent)
    {
        // 处理1个基本块的nd2nz
        GlobalTensor<HalfType>& A1NZGM = A1NZGMs[numNzGM];
        uint64_t gmInOffset = baseKIdx * tilingData.mm2Tiling.baseK * tilingData.VOut512BAlignedSize + baseMIdx * tilingData.mm2Tiling.baseM;
        uint64_t wsOutOffset = OUT_SIZE_PER_32B * tilingData.mm2Tiling.baseK * baseKIdx;
        uint16_t realK = (baseKIdx + mm2BaseCurrent != tilingData.mm2BaseKCnt) ? tilingData.mm2Tiling.baseK : tilingData.mm2TailK;  // 本次实际处理行数
        if (mm2BaseCurrent == SIZE_2) {
            realK += tilingData.mm2Tiling.baseK;
        }
        CopyInMM2Nd2nzFull(gmInOffset, realK);
        CopyOutMM2Nd2nzFull(wsOutOffset, A1NZGM, realK);
    }

    __aicore__ inline void CopyInMM2Nd2nzFull(uint64_t gmInOffset, uint16_t realK)
    {
        auto nd2nzUB = queBindSoftmax.AllocTensor<HalfType>();
        Nd2NzParams nd2NzParams{
            (uint16_t)1,
            realK,
            (uint16_t)tilingData.mm2Tiling.baseM,
            (uint16_t)0,
            (uint16_t)tilingData.VOut512BAlignedSize,
            realK,
            (uint16_t)1,
            (uint16_t)0
        };
        DataCopy(nd2nzUB, softmaxHalfGM[gmInOffset], nd2NzParams);
        queBindSoftmax.EnQue(nd2nzUB);
    }

    __aicore__ inline void CopyOutMM2Nd2nzFull(uint64_t wsOutOffset, GlobalTensor<HalfType>& A1NZGM, uint16_t realK)
    {
        auto nd2nzUB = queBindSoftmax.DeQue<HalfType>();
        DataCopyParams repeatParams{
            C0BlockCntPerBaseM,
            realK,
            0,
            (uint16_t)(tilingData.BT_16ALIGNED - realK)
        };
        DataCopy(A1NZGM[wsOutOffset], nd2nzUB, repeatParams);
        queBindSoftmax.FreeTensor(nd2nzUB);
    }

public:
    TPipe pipe;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, BUFFER_NUM> queBindSoftmax;
    TBuf<TPosition::VECCALC> updateBuf;
    TEventID evtIDMTE3ToS;  // S等搬出
    TEventID evtIDSToMTE3;  // 搬出等S
    TEventID evtIDVToMTE3;  // 搬出等计算
    TEventID evtIDFixToMTE2;  // 搬入等搬出
    GlobalTensor<HalfType> inputGM;  // [BT, H]
    GlobalTensor<HalfType> weightGM;  // [V, H]
    GlobalTensor<float> softmaxGM;  // [BT, V]
    GlobalTensor<float> softmaxForUpdateGM;
    GlobalTensor<HalfType> softmaxHalfGM;  // [BT, V]
    GlobalTensor<HalfType> A1NZGMs[MAX_WS_NUM];  // [baseM, max(VOut32BAlignedSize, BT)]
    GlobalTensor<TargetMaskType> targetMaskForUpdateGM;
    GlobalTensor<MaskedTargetType> maskedTarget1dForUpdateGM;
    GlobalTensor<float> gradOutputGM;  // [BT]
    GlobalTensor<HalfType> gradInputGM;
    GlobalTensor<HalfType> gradWeightGM;

    const FusedLinearCrossEntropyLossGradHighPerfTilingData& tilingData;
    const int64_t blockIdx = GetBlockIdx();
    const int64_t ctrlBlockIdx = blockIdx / 2;
    const uint64_t btOffset = tilingData.btPerCore * GetBlockIdx();
    const uint16_t VecSyncMTE2FlagId = 0;
    const uint16_t VecSyncMTE3FlagId = 1;
    const uint16_t CubeWaitVecFlagId = 2;  // 仅指令位置同步，借用MTE3
    const uint16_t mmSingleAReadFlagId = 3;  // 用于判断mm workspace释放，MTE2
    bool hasA1NZGMTask[MAX_WS_NUM];

    uint32_t btThisCore;
    uint32_t lastBaseMIdx = 2;  // 上一块已涉及的baseM序号
    uint16_t C0BlockCnt;  // 每行输出所占32B块数量
    uint32_t C0ZeroSize;  // C0向32B对齐需要填充0的数量
    uint64_t C0TailOffset;  // nzOffset距离尾块C0需要填充0的位置的间距
    uint16_t C0BlockCntPerBaseM;  // mm2的baseM分为多少个C0
    uint32_t mm1ProcessingBaseBlockIdx;
    uint32_t mm2ProcessingBaseBlockIdx;
    MM mm1;
    MM mm2;
};

}  // namespace FusedLinearCrossEntropyLossGradNS

#endif  // FUSED_LINEAR_CROSS_ENTROPY_LOSS_GRAD_HIGH_PERF_H