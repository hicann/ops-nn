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
 * \file conv_bp_sub_func_deterministic_calculate.h
 * \brief
 */

#ifndef CONV3D_BP_FILTER_SUB_FUNC_DETERMINISTIC_CALCULATE_H
#define CONV3D_BP_FILTER_SUB_FUNC_DETERMINISTIC_CALCULATE_H
#include "conv_bp_sub_func_deterministic_common.h"

namespace ConvolutionBackpropFunc {
template <class Intf>
static __aicore__ inline bool GetIsNeedAddFlag(Intf *self, const uint64_t coreAddCnt,
    const uint64_t coreRelatedIndexTotal)
{
#if defined(__DAV_C310__)
    if (GetIsLastPieceOut(self, coreAddCnt, coreRelatedIndexTotal)) {
        return false;
    }

    uint64_t coreRelatedIndex = coreRelatedIndexTotal - self->ctx.coreStartIndexTotal_;
    // // CV 1:2, 需按cl0Pbuffer>1和等于1的场景判断提前退出条件
    bool needAddFlag =
        (self->ctx.tiling_->cl0Pbuffer > 1 && coreRelatedIndex < self->ctx.deterAddCoreNum_) ||
        (self->ctx.tiling_->cl0Pbuffer == 1 && (((coreRelatedIndex <= self->ctx.deterAddCoreNum_) &&
        self->ctx.subCoreInx_ >= RELATED_CORE_NUM) || ((coreRelatedIndex < self->ctx.deterAddCoreNum_) &&
        self->ctx.subCoreInx_ < RELATED_CORE_NUM)));
    return needAddFlag;
#elif defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201)
    // CV 1:1, 只需判断被累加核超过累加核数量
    uint64_t coreRelatedIndex = coreRelatedIndexTotal - self->ctx.coreStartIndexTotal_;
    return (coreRelatedIndex < self->ctx.deterAddCoreNum_);
#endif
}

template <class Intf>
static __aicore__ inline uint64_t GetRelatedCore(Intf *self, const uint64_t coreAddCnt)
{
    uint64_t coreIndexTotal = self->ctx.coreStartIndexTotal_ + self->ctx.deterAddCoreIndex_;
    uint64_t coreRelatedIndexTotal = 0;
    uint64_t doublecoreAddCnt = (coreAddCnt << 1);
#if defined(__DAV_C310__)
    if (self->ctx.tiling_->cl0Pbuffer > 1) {
        // pingpong模式下：单次输出2个vec核
        // 首层累加: ping，奇数vec核继续累加，偶数vec核提前输出; pong，偶数vec核继续累加，奇数vec核提前输出。
        //          首层为同一个cube核对应的vec核累加
        // 非首层累加: 二叉累加向后寻找对应的计算核
        if (self->ctx.deterAddCoreIndex_ % doublecoreAddCnt == 0) {
            coreRelatedIndexTotal = coreIndexTotal + coreAddCnt;
        } else {
            coreRelatedIndexTotal = coreIndexTotal - coreAddCnt;
        }
    } else {
        // 非pingpong模式下，4个vec核同时输出
        // 按照4个核为基准，向前或者向后寻找对应的累加核
        if ((coreAddCnt == 1 && self->ctx.deterAddCoreIndex_ % doublecoreAddCnt == 0) ||
            (coreAddCnt != 1 && (self->ctx.deterAddCoreIndex_ >> 1) % coreAddCnt == 0)) {
            coreRelatedIndexTotal = coreIndexTotal + coreAddCnt;
        } else if ((coreAddCnt == 1 && self->ctx.deterAddCoreIndex_ % doublecoreAddCnt != 0) ||
            (coreAddCnt != 1 && (self->ctx.deterAddCoreIndex_ >> 1) % coreAddCnt != 0)) {
            coreRelatedIndexTotal = coreIndexTotal - coreAddCnt;
        }
    }
#elif defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201)
    // CV1:1无需根据cl0Pbuffer区分切块
    // 首层累加: 首层为同一个cube核对应的vec核累加
    // 非首层累加: 二叉累加向后寻找对应的计算核
    if (self->ctx.deterAddCoreIndex_ % doublecoreAddCnt == 0) {
        coreRelatedIndexTotal = coreIndexTotal + coreAddCnt;
    } else {
        coreRelatedIndexTotal = coreIndexTotal - coreAddCnt;
    }
#endif
    return coreRelatedIndexTotal;
}

template <class Intf>
static __aicore__ inline void VecAddFunc(Intf *self, const GlobalTensor<typename Intf::DstT> &userGm,
    const uint64_t coreAddCnt, uint64_t &coreRelatedIndexTotal, DeterMinisticShape &deterShape)
{
    auto vecMiddleBuf =
        self->ctx.vecBuf_.template GetWithOffset<float>(VECTOR_UB_SIZE_HALF >> FLOAT_SHIFT_SIZE, VECTOR_UB_SIZE_HALF);

    coreRelatedIndexTotal = GetRelatedCore(self, coreAddCnt);
    bool needAddFlag = GetIsNeedAddFlag(self, coreAddCnt, coreRelatedIndexTotal);

    DataCopyParams intriParams;
    intriParams.blockCount = 1;
    intriParams.blockLen = Ceil(deterShape.mnSize[self->ctx.subCoreInx_] * sizeof(typename Intf::DstT),
        AscendC::ONE_BLOCK_SIZE);
    intriParams.srcStride = 0;
    intriParams.dstStride = 0;

    if (coreAddCnt == 1) {  // 从CubeGm中读取数据
        uint64_t coreIndexTotal = self->ctx.coreStartIndexTotal_ + self->ctx.deterAddCoreIndex_; // 当前核的索引
        uint64_t addCoreRdGmAddr1 = coreIndexTotal * CUBE_WORKSPACE + deterShape.addrOffset[self->ctx.subCoreInx_];
        DataCopy(self->ctx.vecOutBuf_, userGm[addCoreRdGmAddr1], intriParams);
#if defined(__DAV_C310__)
        if (GetIsLastPieceOut(self, coreAddCnt, coreRelatedIndexTotal)) {
            uint64_t addCoreRdGmAddr2 = coreIndexTotal * CUBE_WORKSPACE +
                deterShape.addrOffset[self->ctx.subCoreInx_ + RELATED_CORE_NUM];
            DataCopy(vecMiddleBuf, userGm[addCoreRdGmAddr2], intriParams);
        }
#endif
        if (needAddFlag) {
            uint64_t addCoreRdGmAddr2 = coreRelatedIndexTotal * CUBE_WORKSPACE +
                deterShape.addrOffset[self->ctx.subCoreInx_];
            DataCopy(vecMiddleBuf, userGm[addCoreRdGmAddr2], intriParams);
        }
    } else {    // vec核读取数据并累加
        if (needAddFlag) {
            uint64_t addCoreRdGmAddr = GetRdGmAddr(self, coreRelatedIndexTotal, deterShape);
            PipeBarrier<PIPE_MTE2>();
            DataCopy(vecMiddleBuf, userGm[addCoreRdGmAddr], intriParams);
        }
    }

    if (needAddFlag) {
        event_t eventIdMte2ToVec = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToVec);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToVec);

        Add<float>(self->ctx.vecOutBuf_[0], self->ctx.vecOutBuf_[0], vecMiddleBuf[0],
            deterShape.mnSize[self->ctx.subCoreInx_]);
    } else {
        // 无需累加且只需要输出数据
        event_t eventIdMte2ToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
        SetFlag<HardEvent::MTE2_MTE3>(eventIdMte2ToMte3);
        WaitFlag<HardEvent::MTE2_MTE3>(eventIdMte2ToMte3);
    }
}

template <class Intf>
static __aicore__ inline bool IsCoreNonCalc(Intf *self, const uint64_t coreAddCnt, DeterMinisticShape &deterShape)
{
#if defined(__DAV_C310__)
    // 超出累加核数量，或者累加核shape为0，通知cube结束。
    // streamk场景会有一些核需要跳过，不参与确定性计算
    return coreAddCnt >= self->ctx.deterAddCoreNum_ ||
        deterShape.mSize[self->ctx.subCoreInx_] == 0 || deterShape.nSize[self->ctx.subCoreInx_] == 0 ||
        self->ctx.isNoDeterministic_;
#elif defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201)
    bool isLastCoreOdd = (self->ctx.deterAddCoreIndex_ == self->ctx.deterAddCoreNum_ - 1) &&
            (self->ctx.deterAddCoreNum_ & 1) != 0;
    // 超出累加核数量，或者累加核shape为0，通知vec结束。
    // 最后一个累加奇数核不搬不算
    // streamk场景会有一些核需要跳过，不参与确定性计算
    return coreAddCnt >= self->ctx.deterAddCoreNum_ ||
        deterShape.mSize[self->ctx.subCoreInx_] == 0 || deterShape.nSize[self->ctx.subCoreInx_] == 0 ||
        isLastCoreOdd || self->ctx.isNoDeterministic_;
#endif
}

template <class Intf>
static __aicore__ inline void VecUb2Gm(Intf *self, const GlobalTensor<typename Intf::DstT> &userGm,
    const uint64_t coreAddCnt, const uint64_t coreRelatedIndexTotal, DeterMinisticShape &deterShape)
{
    uint64_t cubeUserGmSize = (GetBlockNum() * CUBE_WORKSPACE); // cube输出gm总大小
    uint64_t addCoreStGmAddr = GetStGmAddr(self, cubeUserGmSize);
    DeterministicUb2Gm(self, userGm[addCoreStGmAddr], deterShape);
#if defined(__DAV_C310__)
    if (GetIsLastPieceOut(self, coreAddCnt, coreRelatedIndexTotal)) {
        uint64_t addCoreStGmAddr2 =
            cubeUserGmSize + ((coreRelatedIndexTotal << 1) + GetSubBlockIdx()) * QUARTER_CUBE_WORKSPACE;
        DeterministicUb2GmNoPingPong(self, userGm[addCoreStGmAddr2], deterShape);
    }
#endif
}

template <class Intf>
static __aicore__ inline void InitSubCoreInx(Intf *self)
{
#if defined(__DAV_C310__)
    uint64_t isOddCore = (self->ctx.deterAddCoreIndex_ & 1) == 0 ? 0 : 1;    // 标记当前核是否为奇数核
    self->ctx.subCoreInx_ = (isOddCore << 1) + GetSubBlockIdx();
#elif defined(__NPU_ARCH__) && (__NPU_ARCH__ == 9201)
    // 奇数核负责L0C切块数据的计算，调整addCoreIndex
    self->ctx.subCoreInx_ = GetBlockIdx() % PINGPONG_NUM;
    self->ctx.deterAddCoreIndex_ = self->ctx.deterAddCoreIndex_ - (self->ctx.deterAddCoreIndex_ & 1);
#endif
}

template <class Intf>
static __aicore__ inline void DeterministicAddFunc(Intf *self, const GlobalTensor<typename Intf::DstT> &output,
    const GlobalTensor<typename Intf::DstT> &userGm)
{
    uint64_t coreAddCnt = 1;     // 迭代次数
    bool loopEndFlag = false;   // 标记是否结束迭代

    DeterMinisticShape deterShape;
    GetCutShape(self, deterShape);
    InitSubCoreInx(self);
    self->ctx.vecOutBuf_ = self->ctx.vecBuf_.template Get<typename Intf::DstT>();

    if (IsCoreNonCalc(self, coreAddCnt, deterShape)) {
        loopEndFlag = true;
    }

    // 累加核数量小于总累加核数量时，继续累加
    while (coreAddCnt < self->ctx.deterAddCoreNum_) {
        if (loopEndFlag) {
            coreAddCnt = (coreAddCnt << 1);
            BarrierVector();
            continue;
        }
        uint64_t coreRelatedIndexTotal = 0; // 被累加核的索引
        VecAddFunc(self, userGm, coreAddCnt, coreRelatedIndexTotal, deterShape);
        loopEndFlag = GetLoopEndFlag(self, coreAddCnt);

        if (IsAddTreeLoopEnd(coreAddCnt, self->ctx.deterAddCoreNum_)) {   // 达到累加迭代次数，输出到GM
            UBRearrange2Gm(self, output, deterShape);
        } else if (loopEndFlag) {    // output to VEC
            VecUb2Gm(self, userGm, coreAddCnt, coreRelatedIndexTotal, deterShape);
        }

        BarrierVector();
        coreAddCnt = (coreAddCnt << 1);
    }
}
}  // namespace ConvolutionBackpropFunc

#endif