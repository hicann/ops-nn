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
 * \file conv_bp_sub_func_deterministic_common.h
 * \brief 确定性累加的跨核编排层: 跨核同步、切块、L0C输出与累加树寻址, 并聚合 def/simd/ub2gm
 */

#ifndef CONV3D_BP_FILTER_SUB_FUNC_DETERMINISTIC_COMMON_H
#define CONV3D_BP_FILTER_SUB_FUNC_DETERMINISTIC_COMMON_H
#include "conv_bp_sub_func_deterministic_def.h"
#include "conv_bp_sub_func_deterministic_simd.h"
#include "conv_bp_sub_func_deterministic_ub2gm.h"

namespace ConvolutionBackpropFunc {
static __aicore__ inline void BarrierVector()
{
#ifndef __CCE_KT_TEST__
    static constexpr uint16_t SYNC_AIV_BAR_FLAG = 0;
    CrossCoreSetFlag<SYNC_MODE0, PIPE_MTE3>(SYNC_AIV_BAR_FLAG);
    CrossCoreWaitFlag(SYNC_AIV_BAR_FLAG);
#endif
}

template <class Intf>
static __aicore__ inline void GetCutShape(Intf* self, DeterMinisticShape& deterShape)
{
    uint64_t splitedCin1 = ShiftCeilM0(Ceil(Ceil(self->ctx.baseUseN_, self->ctx.curSingleCoreDk_), self->ctx.hwK_),
                                       BLOCK_CUBE);
    uint64_t splitedCout1 = ShiftCeilM0(self->ctx.baseUseM_, BLOCK_CUBE);
    // 扩维场景在host侧已经保证的数据不超UBsize，因此不需要对数据做切分
    if constexpr (!Intf::conv3ddwConfig.groupEnlarge) {
        // 默认切2份，L0C不开double buffer时切4份
        DivTwoNumersInHalf(splitedCin1, splitedCout1);
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        if (self->ctx.tiling_->cl0Pbuffer <= 1) {
            DivTwoNumersInHalf(splitedCin1, splitedCout1);
        }
#endif
    }

    uint64_t cutNSize = ShiftMulM0(splitedCin1 * self->ctx.curSingleCoreDk_ * self->ctx.hwK_, BLOCK_CUBE);
    cutNSize = self->ctx.baseUseN_ < cutNSize ? self->ctx.baseUseN_ : cutNSize;
    uint64_t cutMSize = ShiftMulM0(splitedCout1, BLOCK_CUBE);
    uint64_t splitedMIter = Ceil(self->ctx.baseUseM_, cutMSize);
    uint64_t splitedNIter = Ceil(self->ctx.baseUseN_, cutNSize);
    uint64_t workspaceOffset = 0;
    uint64_t pingpongIdx = 0;

    for (uint64_t i = 0; i < splitedNIter; i++) {
        for (uint64_t j = 0; j < splitedMIter; j++) {
            uint64_t inx = pingpongIdx + i * splitedMIter + j;
            deterShape.mSize[inx] = (j == (splitedMIter - 1)) ? (self->ctx.baseUseM_ - cutMSize * j) : cutMSize;
            deterShape.nSize[inx] = (i == (splitedNIter - 1)) ? (self->ctx.baseUseN_ - cutNSize * i) : cutNSize;
            deterShape.mnSize[inx] = deterShape.mSize[inx] * deterShape.nSize[inx];
            deterShape.usedMSize[inx] = j * cutMSize;
            deterShape.usedNSize[inx] = i * cutNSize;
            deterShape.addrOffset[inx] = workspaceOffset;
            deterShape.isNTail[inx] = (self->ctx.curNIdx_ == self->ctx.nIter_ - 1) && (i == splitedNIter - 1);
            workspaceOffset += deterShape.mnSize[inx];
        }
    }
}

template <class Intf>
static __aicore__ inline void MovOutL0cForDeterministicRefactor(Intf* self, LocalTensor<typename Intf::L0cT>& l0c,
                                                                const GlobalTensor<typename Intf::DstT>& output)
{
    uint64_t splitedCin1 = ShiftCeilM0(Ceil(Ceil(self->ctx.baseUseN_, self->ctx.curSingleCoreDk_), self->ctx.hwK_),
                                       BLOCK_CUBE);
    uint64_t splitedCout1 = ShiftCeilM0(self->ctx.baseUseM_, BLOCK_CUBE);
    // 扩维场景在host侧已经保证的数据不超UBsize，因此不需要对数据做切分
    if constexpr (!Intf::conv3ddwConfig.groupEnlarge) {
        // 默认切2份，L0C不开double buffer时切4份
        DivTwoNumersInHalf(splitedCin1, splitedCout1);
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
        if (self->ctx.tiling_->cl0Pbuffer <= 1) {
            DivTwoNumersInHalf(splitedCin1, splitedCout1);
        }
#endif
    }

    uint64_t cutNSize = ShiftMulM0(static_cast<uint64_t>(splitedCin1) * self->ctx.curSingleCoreDk_ * self->ctx.hwK_,
                                   BLOCK_CUBE);

    cutNSize = self->ctx.baseUseN_ < cutNSize ? self->ctx.baseUseN_ : cutNSize;
    uint64_t cutMSize = ShiftMulM0(splitedCout1, BLOCK_CUBE);

    FixpipeParamsArch3510<CO2Layout::NZ> fixPipeParams;
    fixPipeParams.quantPre = QuantMode_t::NoQuant;
    fixPipeParams.unitFlag = 0;

    uint64_t splitedMIter = Ceil(self->ctx.baseUseM_, cutMSize);
    uint64_t splitedNIter = Ceil(self->ctx.baseUseN_, cutNSize);
    uint64_t workspaceOffset = 0;

    uint64_t alignedUseM = ShiftMulM0(ShiftCeilM0(self->ctx.baseUseM_, BLOCK_CUBE), BLOCK_CUBE);
    for (uint64_t i = 0; i < splitedNIter; i++) {
        fixPipeParams.nSize = (i == (splitedNIter - 1)) ? (self->ctx.baseUseN_ - cutNSize * i) : cutNSize;
        for (uint64_t j = 0; j < splitedMIter; j++) {
            fixPipeParams.mSize = (j == (splitedMIter - 1)) ? (self->ctx.baseUseM_ - cutMSize * j) : cutMSize;
            fixPipeParams.srcStride = alignedUseM;
            fixPipeParams.dstStride = ShiftMulM0(fixPipeParams.mSize, BLOCK_CUBE);
            // l0c: cin1 hkwk cout1 cout0 cin0, gm: cin1 hkwk cout cin0, cin0=16
            uint64_t l0cAddress = i * cutNSize * alignedUseM + ShiftMulM0(j * cutMSize, BLOCK_CUBE);
            Fixpipe<typename Intf::DstT, float, CFG_NZ>(output[workspaceOffset], l0c[l0cAddress], fixPipeParams);
            workspaceOffset += fixPipeParams.mSize * fixPipeParams.nSize;
        }
    }
}

template <class Intf>
static __aicore__ inline void DeterministicUb2Gm(Intf* self, const GlobalTensor<typename Intf::DstT>& userGm,
                                                 DeterMinisticShape& deterShape)
{
    DataCopyExtParams ub2GmParams;
    ub2GmParams.srcStride = 0;
    ub2GmParams.dstStride = 0;
    ub2GmParams.blockCount = 1;
    ub2GmParams.blockLen = deterShape.mnSize[self->ctx.subCoreInx_] * sizeof(typename Intf::DstT);

    event_t eventIdVecToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventIdVecToMte3);
    WaitFlag<HardEvent::V_MTE3>(eventIdVecToMte3);
    DataCopyPad<typename Intf::DstT, PaddingMode::Compact>(userGm, self->ctx.vecOutBuf_, ub2GmParams);
}

template <class Intf>
static __aicore__ inline void DeterministicUb2GmNoPingPong(Intf* self, const GlobalTensor<typename Intf::DstT>& userGm,
                                                           DeterMinisticShape& deterShape)
{
    DataCopyExtParams ub2GmParams;
    ub2GmParams.srcStride = 0;
    ub2GmParams.dstStride = 0;
    ub2GmParams.blockCount = 1;
    ub2GmParams.blockLen = deterShape.mnSize[self->ctx.subCoreInx_ + RELATED_CORE_NUM] * sizeof(typename Intf::DstT);

    auto vecMiddleBuf = self->ctx.vecBuf_.template GetWithOffset<float>(VECTOR_UB_SIZE_HALF >> FLOAT_SHIFT_SIZE,
                                                                        VECTOR_UB_SIZE_HALF);
    DataCopyPad<typename Intf::DstT, PaddingMode::Compact>(userGm, vecMiddleBuf, ub2GmParams);
}

static __aicore__ inline bool IsAddTreeLoopEnd(const uint64_t coreAddCnt, const uint64_t deterAddCoreNum)
{
    return (coreAddCnt << 1) >= deterAddCoreNum;
}

template <class Intf>
static __aicore__ inline bool GetIsLastPieceOut(Intf* self, const uint64_t coreAddCnt,
                                                const uint64_t coreRelatedIndexTotal)
{
    return coreAddCnt == 1 && self->ctx.tiling_->cl0Pbuffer == 1 && (self->ctx.deterAddCoreNum_ & 1) != 0 &&
           self->ctx.deterAddCoreNum_ == coreRelatedIndexTotal - self->ctx.coreStartIndexTotal_;
}

template <class Intf>
static __aicore__ inline bool GetLoopEndFlag(Intf* self, const uint64_t coreAddCnt)
{
    // 1、二叉树计算中是输出核，需要提前退出累加过程
    // 2、累加核数量为奇数时，最后一个核只需要提前输出一次，提前退出累加过程
    uint64_t doublecoreAddCnt = (coreAddCnt << 1);
    bool isLastCoreOdd = (self->ctx.deterAddCoreIndex_ == self->ctx.deterAddCoreNum_ - 1) &&
                         (self->ctx.deterAddCoreNum_ & 1) != 0;
    if (((self->ctx.deterAddCoreIndex_ >> 1) % doublecoreAddCnt > 0) || isLastCoreOdd) {
        return true;
    }
    // 3、累加核数量超过累加次数，完成计算退出
    if (IsAddTreeLoopEnd(coreAddCnt, self->ctx.deterAddCoreNum_)) {
        return true;
    }

    return false;
}

template <class Intf>
static __aicore__ inline uint64_t GetRdGmAddr(Intf* self, const uint64_t coreRelatedIndexTotal,
                                              DeterMinisticShape& deterShape)
{
    uint64_t cubeUserGmSize = GetBlockNum() * CUBE_WORKSPACE; // cube输出gm总大小
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    return cubeUserGmSize + ((coreRelatedIndexTotal << 1) + GetSubBlockIdx()) * QUARTER_CUBE_WORKSPACE;
#elif defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201)
    uint64_t addCoreRdGmAddr = 0;
    if ((coreRelatedIndexTotal - self->ctx.coreStartIndexTotal_ == self->ctx.deterAddCoreNum_ - 1) &&
        (self->ctx.deterAddCoreNum_ & 1) != 0) { // 最后一个奇数核需要从CUBE_WORKSPACE搬入
        addCoreRdGmAddr = coreRelatedIndexTotal * CUBE_WORKSPACE + deterShape.addrOffset[self->ctx.subCoreInx_];
    } else {
        addCoreRdGmAddr = cubeUserGmSize + ((coreRelatedIndexTotal << 1) + self->ctx.subCoreInx_) * HALF_CUBE_WORKSPACE;
    }
    return addCoreRdGmAddr;
#endif
}

template <class Intf>
static __aicore__ inline uint64_t GetStGmAddr(Intf* self, const uint64_t cubeUserGmSize)
{
    uint64_t coreIndexTotal = self->ctx.coreStartIndexTotal_ + self->ctx.deterAddCoreIndex_; // 当前核的索引
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510)
    return cubeUserGmSize + ((coreIndexTotal << 1) + GetSubBlockIdx()) * QUARTER_CUBE_WORKSPACE;
#elif defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201)
    return cubeUserGmSize + ((coreIndexTotal << 1) + self->ctx.subCoreInx_) * HALF_CUBE_WORKSPACE;
#endif
}
} // namespace ConvolutionBackpropFunc

#endif
