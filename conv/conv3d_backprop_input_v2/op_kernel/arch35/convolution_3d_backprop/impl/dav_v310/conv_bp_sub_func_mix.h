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
 * \file conv_bp_sub_func_mix.h
 * \brief
 */

#ifndef CONV3D_BP_INPUT_SUB_FUNC_MIX_ADVANCE_H
#define CONV3D_BP_INPUT_SUB_FUNC_MIX_ADVANCE_H

#include "../../../../../inc/platform.h"
#include "../../../../conv3d_backprop_input_v2_arch35_tiling_key.h"

using AscendC::DivCeil;
using AscendC::GlobalTensor;
using AscendC::LocalTensor;
using AscendC::MultiCopyConfig;

namespace Convolution3DBackpropFunc {
const static uint64_t DQ_SCALAR_ONE = 0x3F800000; // float 1.0
constexpr uint8_t FLAG_MTE1_ID_1 = 6;
constexpr uint8_t FLAG_MTE1_ID_2 = 7;
constexpr uint8_t FLAG_FIXP_ID = 8;
constexpr uint8_t FLAG_MTE2_VEC_ID = 9;
constexpr uint8_t SYNC_MODE = 4;
constexpr uint8_t CROSS_CORE_FLAG_ID_MAX = 16;
constexpr uint8_t GROUP_NDDMA_DIM_NUM = 4;
constexpr uint8_t SUB_KERNEL_NUM = 4;
constexpr uint8_t INDEX_0 = 0;
constexpr uint8_t INDEX_1 = 1;
constexpr uint8_t INDEX_2 = 2;
constexpr uint8_t INDEX_3 = 3;
constexpr uint8_t VEC_NUM = 2;
constexpr uint8_t ONE_BLK_SHIFT_SIZE = 5;
constexpr uint8_t C04_COUT_SIZE = 4;
constexpr uint8_t C04_SHIFT_SIZE = 2;
constexpr uint8_t MASK_REG_WIDTH = AscendC::VECTOR_REG_WIDTH >> 3; // 右移3bit: MaskReg的宽度是RegTensor的1/8
constexpr MultiCopyConfig nddmaConfig = {false};
constexpr FixpipeConfig CFG_COLUMN_MAJOR_UB = {CO2Layout::COLUMN_MAJOR, true};
constexpr uint32_t UB_SIZE = AscendC::TOTAL_UB_SIZE;
constexpr uint32_t SHIFT_BIT_4 = 4;

static __aicore__ inline uint32_t Div16(uint32_t a)
{
    return a >> SHIFT_BIT_4;
}

static __aicore__ inline uint32_t DivCeil16(uint32_t a)
{
    return (a + 15) >> SHIFT_BIT_4;
}

static __aicore__ inline uint32_t AlignUp16(uint32_t a)
{
    return DivCeil16(a) << SHIFT_BIT_4;
}

static __aicore__ inline uint32_t AlignDown(uint32_t a, uint32_t rnd)
{
    return ((a) == 0 ? 0 : ((a / rnd) * rnd));
}

static __aicore__ inline uint32_t AlignUpByDtype(uint32_t a, uint32_t dtypeBit)
{
    return ((a + ((1 << dtypeBit) - 1)) >> dtypeBit) << dtypeBit;
}

template <class Intf>
static __aicore__ inline uint32_t DivCeilC0(Intf *self, const uint32_t a)
{
    return (a + self->ctx.tiling_->c0 - 1) >> self->ctx.tiling_->c0BitsB;
}

template <class Intf>
static __aicore__ inline uint32_t AlignUpC0(Intf *self, const uint32_t a)
{
    return ((a + self->ctx.tiling_->c0 - 1) >> self->ctx.tiling_->c0BitsB) << self->ctx.tiling_->c0BitsB;
}

template <class Intf>
static __aicore__ inline uint32_t DivHkWk(Intf *self, uint32_t a)
{
    if constexpr (Intf::conv3dConfig.kernelSplitMode != TPL_NO_SPLIT_KERNEL) {
        return self->ctx.splitHkWkList_[self->ctx.splitIndex_] > 1 ?
            a / self->ctx.splitHkWkList_[self->ctx.splitIndex_] : a;
    } else {
        return self->ctx.singleShapeHWk_ > 1 ? a / self->ctx.singleShapeHWk_ : a;
    }
}

template <class Intf>
static __aicore__ inline uint32_t DivCeilHkWk(Intf *self, uint32_t a)
{
    if constexpr (Intf::conv3dConfig.kernelSplitMode != TPL_NO_SPLIT_KERNEL) {
        return self->ctx.splitHkWkList_[self->ctx.splitIndex_] > 1 ?
            (a + self->ctx.splitHkWkList_[self->ctx.splitIndex_] - 1) /
            self->ctx.splitHkWkList_[self->ctx.splitIndex_] : a;
    } else {
        return self->ctx.singleShapeHWk_ > 1 ? (a + self->ctx.singleShapeHWk_ - 1) / self->ctx.singleShapeHWk_ : a;
    }
}

template <class DType>
static __aicore__ inline uint32_t DivDtypeByte(uint32_t a)
{
    if constexpr(std::is_same<DType, bfloat16_t>::value || std::is_same<DType, half>::value || std::is_same<DType, uint16_t>::value) {
        return a >> 1; // 除以2字节
    } else if constexpr(std::is_same<DType, float>::value || std::is_same<DType, uint32_t>::value) {
        return a >> 2; // 2: 除以4字节
    } else { // hifloat8_t || fp8_e4m3fn_t
        return a;
    }
}

template <class Intf>
__aicore__ inline void WaitForVecBeforeLoadToB2(Intf *self)
{
#ifndef __CCE_KT_TEST__
    CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(FLAG_MTE1_ID_1);
    if constexpr (Intf::conv3dConfig.enableC04Flag) {
        CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE1>(FLAG_MTE1_ID_1 + CROSS_CORE_FLAG_ID_MAX);
    }
#endif
}

template <class Intf>
__aicore__ inline void NotifyVecAfterLoadToB2(Intf *self)
{
#ifndef __CCE_KT_TEST__
    CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(FLAG_MTE1_ID_2);
    if constexpr (Intf::conv3dConfig.enableC04Flag) {
        CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(FLAG_MTE1_ID_2 + CROSS_CORE_FLAG_ID_MAX);
    }
#endif
}

template <class Intf>
__aicore__ inline void WaitForCubeBeforeLoadToB1(Intf *self)
{
#ifndef __CCE_KT_TEST__
    CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE3>(FLAG_MTE1_ID_2);
#endif
}

template <class Intf>
__aicore__ inline void NotifyCubeAfterLoadToB1(Intf *self)
{
#ifndef __CCE_KT_TEST__
    CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(FLAG_MTE1_ID_1);
#endif
}

template <class Intf>
__aicore__ inline LocalTensor<typename Intf::SrcT> GetB1Tbuf(Intf *self, const uint64_t kIdx)
{
    bool b1PingPongFlag = true;
    if (self->ctx.tiling_->bl1Pbuffer > 1) {
        b1PingPongFlag = (1 + kIdx / self->ctx.tiling_->stepKb) & 1;
    }
    LocalTensor<typename Intf::SrcT> useB1Tbuf;
    if (b1PingPongFlag) {
        useB1Tbuf = self->ctx.b1UbPing_.template Get<typename Intf::SrcT>();
    } else {
        useB1Tbuf = self->ctx.b1UbPong_.template Get<typename Intf::SrcT>();
    }
    return useB1Tbuf;
}

template <class Intf>
__aicore__ inline uint32_t CalcCurCinSizeB1(Intf *self, uint32_t curCinIdx)
{
    uint32_t curCinSize = self->ctx.curStepN_ * self->ctx.tiling_->baseN;
    // consider tail
    uint32_t curCinRemain = self->ctx.singleShapeCin_ - curCinIdx;
    curCinSize = curCinSize < curCinRemain ? curCinSize : curCinRemain;
    return curCinSize;
}

template <class Intf>
__aicore__ inline void CalcCoutIndexAndSizeB1(Intf *self, uint64_t kIdx,
    uint32_t &curCoutIdx, uint32_t &curCoutSize)
{
   if constexpr (Intf::conv3dConfig.kernelSplitMode == TPL_SPLIT_KERNEL_HW) {
        if (self->ctx.kSCoutFullLoad_) {    // cout全载场景
            curCoutSize = self->ctx.tiling_->singleCoreCout;
            return;
        }
    }

    // 考虑到Preload的场景，L1的载入量要根据传入的KIdx确定，不能使用全局变量
    uint32_t kbL1Size = 0;
    if (unlikely(kIdx >= self->ctx.kIterStepKbTail)) {
        kbL1Size = (self->ctx.stepKbTail - 1) * self->ctx.tiling_->baseK + self->ctx.tailK_;
    } else {
        kbL1Size = self->ctx.curStepKb_ * self->ctx.tiling_->baseK;
    }

    if constexpr (Intf::conv3dConfig.kernelSplitMode != TPL_NO_SPLIT_KERNEL) {
        curCoutIdx = kIdx * self->ctx.tiling_->baseK / self->ctx.splitHkWkList_[self->ctx.splitIndex_];
    } else {
        curCoutIdx = DivHkWk<Intf>(self, kIdx * self->ctx.tiling_->baseK);
    }
    curCoutSize = DivHkWk<Intf>(self, kbL1Size);
    // consider tail
    uint32_t curCoutRemain = self->ctx.singleShapeCout_ - curCoutIdx;
    curCoutSize = curCoutSize < curCoutRemain ? curCoutSize : curCoutRemain;
}

template <class Intf>
static __aicore__ inline void CalcCutInWIndex(Intf *self, const uint32_t crossBlockNum)
{
    uint32_t doubleBaseUseM = self->ctx.baseUseM_ << crossBlockNum;
    uint32_t wiUsed = AlignUp(self->ctx.tiling_->wi, self->ctx.tiling_->strideW); // 奇数场景当前tiling限制wi<512，对其之后不会溢出
    uint32_t mSize = self->ctx.curML0Idx_ * (self->ctx.tiling_->baseM << crossBlockNum);
    uint32_t curWiPos = wiUsed - (mSize % wiUsed); // 上一轮baseM搬完后一行Wi还剩下未处理的长度
    if (curWiPos > doubleBaseUseM || curWiPos == wiUsed) {
        // 未处理的长度大于baseUseM 或 等于整行wi，即上一次搬运不涉及尾块
        // 则无需处理首块
        self->ctx.headWi_ = 0;
    } else {
        self->ctx.headWi_ = curWiPos;
    }
    int32_t leftBaseUseM = doubleBaseUseM - self->ctx.headWi_;
    if (leftBaseUseM < 0) {
        leftBaseUseM = 0;
    }
    self->ctx.midHi_ = static_cast<uint32_t>(leftBaseUseM) / wiUsed;
    self->ctx.tailWi_ = static_cast<uint32_t>(leftBaseUseM) - self->ctx.midHi_ * wiUsed;
}

template <class Intf>
static __aicore__ inline void LoadL0c2GMForKernelSplitInner(Intf *self, const LocalTensor<typename Intf::L0cT> &useC1Buf,
    FixpipeParamsC310<CO2Layout::COLUMN_MAJOR> &fixPipeParams, const uint32_t crossBlockNum)
{
    uint32_t align32Byte = 4; // 4: b16 需要对齐到16，2的4次方
    if constexpr(std::is_same<typename Intf::DstT, hifloat8_t>::value ||
        std::is_same<typename Intf::DstT, fp8_e4m3fn_t>::value) {
        align32Byte = 5; // 5: b8 需要对齐到32，2的5次方
    }

    uint32_t hi = DivCeil(self->ctx.tiling_->baseM, self->ctx.splitWi_);
    uint64_t hwAlign = static_cast<uint64_t>(AlignUpByDtype(self->ctx.splitWi_, align32Byte)) * hi * self->ctx.tiling_->baseN;
    uint32_t srcWi = (self->ctx.baseUseM_ < self->ctx.splitWi_) ? self->ctx.baseUseM_ : self->ctx.splitWi_;
    uint32_t realHeadWi = (self->ctx.headWi_ >> crossBlockNum);
    uint32_t realTailWi = (self->ctx.tailWi_ >> crossBlockNum);
    uint32_t alignMidWi = AlignUpByDtype(srcWi, align32Byte);
    uint64_t hwSize = AlignUpByDtype(realHeadWi, align32Byte) + static_cast<uint64_t>(self->ctx.midHi_) * alignMidWi +
        AlignUpByDtype(realTailWi, align32Byte);
    // 将每个kernel拆分的小块L0C的结果暂存在UB上，直到有完整一个W的数据
    uint64_t wsDstOffset = hwSize * self->ctx.baseUseN_ * self->ctx.rearrangeWIndex_ +
        GetAicBlockIdx() * (hwAlign << crossBlockNum);
    uint64_t srcOffset = 0;

    // loop2_dst_stride, element, c
    fixPipeParams.dstStride = hwSize; // dst N stride, loop2_dst_stride (unit: element)
    // fixpipe->ub 需要分首块，中间块，尾块分别对齐到16，然后再搬到ub
    if (self->ctx.headWi_ != 0) { // 需要首块
        uint32_t alignHeadWi = AlignUpByDtype(realHeadWi, align32Byte);
        fixPipeParams.params.dnNum = 1; // not use
        fixPipeParams.params.srcNzMatrixStride = 0; // loop3_src_stride
        fixPipeParams.params.dstDnMatrixStride = 0; // loop3_dst_stride

        fixPipeParams.mSize = realHeadWi; // M: 首块的长度, 真实的headwi
        Fixpipe<typename Intf::DstT, float, CFG_COLUMN_MAJOR>(self->ctx.l0cOutWorkspace_[wsDstOffset],
            useC1Buf[srcOffset], fixPipeParams);
        // BLOCK_CUBE: MMAD一次计算为16*16, fixpipe搬到ub的时候取L0c的数据应该固定c0为16，不能随数据类型变化
        srcOffset += realHeadWi << 4; // headWi_/2是一个子kernel首块w的长度
        wsDstOffset += alignHeadWi;
    }

    if (self->ctx.midHi_ != 0) { // 需要中间块
        fixPipeParams.params.dnNum = self->ctx.midHi_; // 循环中间块的行数
        fixPipeParams.params.srcNzMatrixStride = srcWi; // loop3_src_stride
        fixPipeParams.params.dstDnMatrixStride = alignMidWi; // loop3_dst_stride

        fixPipeParams.mSize = srcWi; // M: 中间块一行的长度, 真实的midwi
        Fixpipe<typename Intf::DstT, float, CFG_COLUMN_MAJOR>(self->ctx.l0cOutWorkspace_[wsDstOffset],
            useC1Buf[srcOffset], fixPipeParams);
        // BLOCK_CUBE: MMAD一次计算为16*16, fixpipe搬到ub的时候取L0c的数据应该固定c0为16，不能随数据类型变化
        srcOffset += self->ctx.midHi_ * srcWi << 4; // srcWi是一个子kernel中间块w的长度
        wsDstOffset += static_cast<int64_t>(self->ctx.midHi_) * alignMidWi;
    }

    if (self->ctx.tailWi_ != 0) { // 需要尾块
        fixPipeParams.params.dnNum = 1; // not use
        fixPipeParams.params.srcNzMatrixStride = 0; // loop3_src_stride
        fixPipeParams.params.dstDnMatrixStride = 0; // loop3_dst_stride

        fixPipeParams.mSize = realTailWi; // M: 尾块的长度, 真实的tailwi
        Fixpipe<typename Intf::DstT, float, CFG_COLUMN_MAJOR>(self->ctx.l0cOutWorkspace_[wsDstOffset],
            useC1Buf[srcOffset], fixPipeParams);
    }
}

template <class Intf>
static __aicore__ inline void LoadL0c2UbForKernelSplitInner(Intf *self, const LocalTensor<typename Intf::L0cT> &useC1Buf,
    FixpipeParamsC310<CO2Layout::COLUMN_MAJOR> &fixPipeParams, const uint32_t crossBlockNum)
{
    self->ctx.vecOutBuf_ = self->ctx.vecBuf_.template Get<typename Intf::DstT>();
    uint32_t align32Byte = 4; // 4: b16 需要对齐到16，2的4次方
    if constexpr(std::is_same<typename Intf::DstT, hifloat8_t>::value ||
        std::is_same<typename Intf::DstT, fp8_e4m3fn_t>::value) {
        align32Byte = 5; // 5: b8 需要对齐到32，2的5次方
    }

    uint64_t srcWi = (self->ctx.baseUseM_ < self->ctx.splitWi_) ? self->ctx.baseUseM_ : self->ctx.splitWi_;
    uint32_t realHeadWi = (self->ctx.headWi_ >> crossBlockNum);
    uint32_t realTailWi = (self->ctx.tailWi_ >> crossBlockNum);
    uint32_t alignMidWi = AlignUpByDtype(srcWi, align32Byte);
    int64_t hwSize = AlignUpByDtype(realHeadWi, align32Byte) + self->ctx.midHi_ * alignMidWi +
        AlignUpByDtype(realTailWi, align32Byte);
    // 将每个kernel拆分的小块L0C的结果暂存在UB上，直到有完整一个W的数据
    int64_t ubDstOffset = hwSize * self->ctx.baseUseN_ * self->ctx.rearrangeWIndex_;
    int64_t srcOffset = 0;

    // loop2_dst_stride, element, c
    fixPipeParams.dstStride = hwSize; // dst N stride, loop2_dst_stride (unit: element)
    // fixpipe->ub 需要分首块，中间块，尾块分别对齐到16，然后再搬到ub
    if (self->ctx.headWi_ != 0) { // 需要首块
        uint32_t alignHeadWi = AlignUpByDtype(realHeadWi, align32Byte);
        fixPipeParams.params.dnNum = 1; // not use
        fixPipeParams.params.srcNzMatrixStride = 0; // loop3_src_stride
        fixPipeParams.params.dstDnMatrixStride = 0; // loop3_dst_stride

        fixPipeParams.mSize = alignHeadWi; // M: 首块的长度
        Fixpipe<typename Intf::DstT, float, CFG_COLUMN_MAJOR_UB>(self->ctx.vecOutBuf_[ubDstOffset],
            useC1Buf[srcOffset], fixPipeParams);
        // BLOCK_CUBE: MMAD一次计算为16*16, fixpipe搬到ub的时候取L0c的数据应该固定c0为16，不能随数据类型变化
        srcOffset += realHeadWi << 4; // headWi_/2是一个子kernel首块w的长度
        ubDstOffset += alignHeadWi;
    }

    if (self->ctx.midHi_ != 0) { // 需要中间块
        fixPipeParams.params.dnNum = self->ctx.midHi_; // 循环中间块的行数
        fixPipeParams.params.srcNzMatrixStride = srcWi; // loop3_src_stride
        fixPipeParams.params.dstDnMatrixStride = alignMidWi; // loop3_dst_stride

        fixPipeParams.mSize = alignMidWi; // M: 中间块一行的长度
        Fixpipe<typename Intf::DstT, float, CFG_COLUMN_MAJOR_UB>(self->ctx.vecOutBuf_[ubDstOffset],
            useC1Buf[srcOffset], fixPipeParams);
        // BLOCK_CUBE: MMAD一次计算为16*16, fixpipe搬到ub的时候取L0c的数据应该固定c0为16，不能随数据类型变化
        srcOffset += self->ctx.midHi_ * srcWi << 4; // srcWi是一个子kernel中间块w的长度
        ubDstOffset += self->ctx.midHi_ * alignMidWi;
    }

    if (self->ctx.tailWi_ != 0) { // 需要尾块
        fixPipeParams.params.dnNum = 1; // not use
        fixPipeParams.params.srcNzMatrixStride = 0; // loop3_src_stride
        fixPipeParams.params.dstDnMatrixStride = 0; // loop3_dst_stride

        fixPipeParams.mSize = AlignUpByDtype(realTailWi, align32Byte); // M: 尾块的长度
        Fixpipe<typename Intf::DstT, float, CFG_COLUMN_MAJOR_UB>(self->ctx.vecOutBuf_[ubDstOffset],
            useC1Buf[srcOffset], fixPipeParams);
    }
}

template <class Intf>
static __aicore__ inline void SetFixPipeQuantVal(FixpipeParamsC310<CO2Layout::COLUMN_MAJOR> &fixPipeParams)
{
    if constexpr(std::is_same<typename Intf::DstT, bfloat16_t>::value) {
        fixPipeParams.quantPre = QuantMode_t::F322BF16;
    } else if constexpr(std::is_same<typename Intf::DstT, half>::value) {
        fixPipeParams.quantPre = QuantMode_t::F322F16;
    } else if constexpr(std::is_same<typename Intf::DstT, hifloat8_t>::value) {
        fixPipeParams.quantPre = QuantMode_t::QF322HIF8_PRE; // Half to Away Round
        fixPipeParams.deqScalar = DQ_SCALAR_ONE;
    } else if constexpr(std::is_same<typename Intf::DstT, fp8_e4m3fn_t>::value) {
        fixPipeParams.quantPre = QuantMode_t::QF322FP8_PRE;
        fixPipeParams.deqScalar = DQ_SCALAR_ONE;
    }
}

// kernel拆分后, 输出到Ub上进行重排
template <class Intf>
static __aicore__ inline void LoadL0c2OutForKernelSplitHW(Intf *self, const LocalTensor<typename Intf::L0cT> &useC1Buf)
{
    if (GetSubBlockIdx() > 0) {
        return;
    }
    constexpr uint32_t crossBlockNum = 1; // 1: two corss blocks shift bit is 1, use to *2 and /2
    CalcCutInWIndex<Intf>(self, crossBlockNum);

    FixpipeParamsC310<CO2Layout::COLUMN_MAJOR> fixPipeParams;
    SetFixPipeQuantVal<Intf>(fixPipeParams);
    fixPipeParams.params.srcNzC0Stride = 1; // src M stride, loop0_src_stride (unit: 32B)
    fixPipeParams.nSize = self->ctx.baseUseN_; // N: cin
    // loop1_src_stride, c0_size, cin1
    fixPipeParams.srcStride = AlignUp16(self->ctx.baseUseM_); // src N stride, loop1_src_stride (unit: 32B)
    // 由于tiling切M的时候是hi*wi一个整体对齐到32B进行切分，可能无法保证切到一个完整的wi，而ub指令又有对齐要求，所以需要分块对齐搬运
    if (self->ctx.kSUseWorkSpace_) {
        LoadL0c2GMForKernelSplitInner<Intf>(self, useC1Buf, fixPipeParams, crossBlockNum);
    } else {
        LoadL0c2UbForKernelSplitInner<Intf>(self, useC1Buf, fixPipeParams, crossBlockNum);
    }
}

template <class Intf>
static __aicore__ inline void SetDataCopyPadLoopInfo(Intf *self, int64_t hwSize, const uint32_t curUseN,
    const uint32_t crossBlockNum)
{
    LoopModeParams loopParams;
    loopParams.loop2Size = 1;   // not use loop2Size
    loopParams.loop2SrcStride = 0;
    loopParams.loop2DstStride = 0;
    loopParams.loop1Size = curUseN;
    loopParams.loop1SrcStride = (hwSize << crossBlockNum) * sizeof(typename Intf::DstT);
    loopParams.loop1DstStride = self->ctx.diHiWi_ * sizeof(typename Intf::DstT);
    SetLoopModePara(loopParams, DataCopyMVType::UB_TO_OUT);
}

template <class Intf>
static __aicore__ inline void DataCopyUbToGmForKernelSplit(Intf *self, const GlobalTensor<typename Intf::DstT> &output,
    int64_t srcOffset, int64_t dstOffset, const uint32_t crossBlockNum)
{
    uint32_t wiUsed = AlignUp(self->ctx.tiling_->wi, self->ctx.tiling_->strideW);
    uint32_t align32Byte = 4; // 4: b16 需要对齐到16，2的4次方
    if constexpr(std::is_same<typename Intf::DstT, hifloat8_t>::value ||
        std::is_same<typename Intf::DstT, fp8_e4m3fn_t>::value) {
        align32Byte = 5; // 5: b8 需要对齐到32，2的5次方
    }
    DataCopyExtParams ub2GmParams;
    ub2GmParams.srcStride = 0;
    if (self->ctx.headWi_ != 0) { // 需要首块
        uint32_t realHeadWi = self->ctx.headWi_;
        // 奇数场景下，首块的最后一个不搬
        if (self->ctx.rearrangeWIndex_ == 1 && (self->ctx.tiling_->wi & 0x1) == 1) {
            realHeadWi -= 1;
        }
        ub2GmParams.blockLen = realHeadWi * sizeof(typename Intf::DstT);
        ub2GmParams.blockCount = 1;
        ub2GmParams.dstStride = 0;
        DataCopyPad(output[dstOffset], self->ctx.vecOutBuf_[srcOffset], ub2GmParams);
        srcOffset += (AlignUpByDtype(self->ctx.headWi_ >> crossBlockNum, align32Byte) << crossBlockNum);
        dstOffset += (realHeadWi + self->ctx.tiling_->wi);
    }

    if (self->ctx.midHi_ != 0) { // 需要中间块
        uint32_t srcWi = (self->ctx.baseUseM_ < self->ctx.splitWi_) ? self->ctx.baseUseM_ : self->ctx.splitWi_;
        uint32_t alignWi = AlignUpByDtype(srcWi, align32Byte);
        ub2GmParams.srcStride = ((alignWi << crossBlockNum) - wiUsed) *
            sizeof(typename Intf::DstT) >> ONE_BLK_SHIFT_SIZE; // 32Bytes one datablock
        ub2GmParams.blockLen = self->ctx.tiling_->wi * sizeof(typename Intf::DstT);
        ub2GmParams.blockCount = self->ctx.midHi_;
        ub2GmParams.dstStride = self->ctx.tiling_->wi * sizeof(typename Intf::DstT);
        DataCopyPad(output[dstOffset], self->ctx.vecOutBuf_[srcOffset], ub2GmParams);
        srcOffset += (alignWi * self->ctx.midHi_ << crossBlockNum);
        dstOffset += (self->ctx.tiling_->wi * self->ctx.midHi_ << crossBlockNum);
    }

    if (self->ctx.tailWi_ != 0) { // 需要尾块
        ub2GmParams.blockLen = self->ctx.tailWi_ * sizeof(typename Intf::DstT);
        ub2GmParams.blockCount = 1;
        ub2GmParams.dstStride = 0;
        DataCopyPad(output[dstOffset], self->ctx.vecOutBuf_[srcOffset], ub2GmParams);
    }
}

template <class Intf>
static __aicore__ inline void LoadUbToGmForKernelSplit(Intf *self, const GlobalTensor<typename Intf::DstT> &output,
    const uint32_t curUseN, const int64_t hwSize, const int64_t dstNOffset, const uint32_t crossBlockNum)
{
    uint32_t wiUsed = AlignUp(self->ctx.tiling_->wi, self->ctx.tiling_->strideW);
    uint32_t mSize = self->ctx.curML0Idx_ * self->ctx.tiling_->baseM << crossBlockNum;
    uint32_t hiUsed = mSize / wiUsed;
    uint32_t mUseSize = hiUsed * self->ctx.tiling_->wi + mSize - hiUsed * wiUsed;
    uint32_t curSkipMSize = (mSize / wiUsed) * self->ctx.tiling_->wi; // GM上隔行需要空开的数量
    uint64_t dstOffset = static_cast<uint64_t>(self->ctx.curNL0Idx_) * self->ctx.tiling_->baseN * self->ctx.diHiWi_ + // cin offset
                    static_cast<uint64_t>(self->ctx.curDinIdx_) * self->ctx.hiWi_ + // di offset, remove useless data
                    mUseSize + curSkipMSize + // hi&wi offset
                    self->ctx.rearrangeHIndex_ * self->ctx.tiling_->wi + dstNOffset;

    SetDataCopyPadLoopInfo(self, hwSize, curUseN, crossBlockNum);
    int64_t dataLen = curUseN * hwSize;
    int64_t srcOffset = (static_cast<uint64_t>(dataLen) << crossBlockNum);
    DataCopyUbToGmForKernelSplit(self, output, srcOffset, dstOffset, crossBlockNum);
    ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
}

template <class Intf, class ReDstT>
__aicore__ inline void InterleaveUbOutForKernelSplit(Intf *self, int64_t dataLen, const uint32_t crossBlockNum)
{
    uint32_t vfLen = platform::GetVRegSize() / sizeof(ReDstT);
    uint32_t doubleVfLen = (vfLen << crossBlockNum);
    uint16_t repeatTimes = (dataLen + vfLen - 1) / vfLen;
    uint64_t twoBlockLen = (dataLen << crossBlockNum);

    auto src0Ptr = (__ubuf__ ReDstT *)self->ctx.vecOutBuf_[0].GetPhyAddr();
    auto src1Ptr = (__ubuf__ ReDstT *)self->ctx.vecOutBuf_[dataLen].GetPhyAddr();
    auto dst0Ptr = (__ubuf__ ReDstT *)self->ctx.vecOutBuf_[twoBlockLen].GetPhyAddr();
    auto dst1Ptr = (__ubuf__ ReDstT *)self->ctx.vecOutBuf_[twoBlockLen + vfLen].GetPhyAddr();

    // 取两个kernel拆分的切块(CHW)使用interleave进行交叉排布
    __VEC_SCOPE__
    {
        MicroAPI::MaskReg preg = MicroAPI::CreateMask<ReDstT, MicroAPI::MaskPattern::ALL>();
        MicroAPI::RegTensor<ReDstT> src0;
        MicroAPI::RegTensor<ReDstT> src1;
        MicroAPI::RegTensor<ReDstT> dst0;
        MicroAPI::RegTensor<ReDstT> dst1;

        for (uint16_t i = 0; i < repeatTimes; i++) {
            MicroAPI::DataCopy(src0, src0Ptr + i * vfLen);
            MicroAPI::DataCopy(src1, src1Ptr + i * vfLen);
            // Interleave指令不支持hif8，需要伪装成uint8
            MicroAPI::Interleave(dst0, dst1, src0, src1);
            MicroAPI::DataCopy(dst0Ptr + i * doubleVfLen, dst0, preg);
            MicroAPI::DataCopy(dst1Ptr + i * doubleVfLen, dst1, preg);
        }
    }
}

template <class Intf>
__aicore__ inline void LoadWorkSpaceDataToUbInner(Intf *self, int64_t srcOffset, int64_t dstOffset, const DataCopyExtParams &mte2Param)
{
    if constexpr(std::is_same<typename Intf::DstT, hifloat8_t>::value ||
            std::is_same<typename Intf::DstT, fp8_e4m3fn_t>::value) {
        DataCopyPadExtParams<uint8_t> padParams = {false, 0, 0, 0};
        DataCopyPad<uint8_t, PaddingMode::Compact>(
            self->ctx.vecOutBuf_.template ReinterpretCast<uint8_t>()[dstOffset],
                self->ctx.l0cOutWorkspace_.template ReinterpretCast<uint8_t>()[srcOffset], mte2Param, padParams);
    } else {
        DataCopyPadExtParams<typename Intf::DstT> padParams = {false, 0, 0, 0};
        DataCopyPad<typename Intf::DstT, PaddingMode::Compact>(
            self->ctx.vecOutBuf_[dstOffset], self->ctx.l0cOutWorkspace_[srcOffset], mte2Param, padParams);
    }
}

template <class Intf>
__aicore__ inline void LoadWorkSpaceDataToUb(Intf *self, const int64_t hwSize,
    const uint32_t curUseN, int64_t srcOffset)
{
    self->ctx.vecOutBuf_ = self->ctx.vecBuf_.template Get<typename Intf::DstT>();
    DataCopyExtParams mte2Param;
    mte2Param.blockCount = 1;
    mte2Param.blockLen = curUseN * hwSize * sizeof(typename Intf::DstT);
    mte2Param.srcStride = 0;
    mte2Param.dstStride = 0;
    LoadWorkSpaceDataToUbInner(self, srcOffset, 0, mte2Param);

    srcOffset += hwSize * self->ctx.baseUseN_;
    int64_t dstOffset = hwSize * curUseN;
    LoadWorkSpaceDataToUbInner(self, srcOffset, dstOffset, mte2Param);
    event_t eventIdMte2ToVec = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToVec);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToVec);
}

template <class Intf>
__aicore__ inline void Rearrange2GmForL0cToWorkSpace(Intf *self, const GlobalTensor<typename Intf::DstT> &output,
    uint32_t align32Byte, int64_t hwSize, int64_t dataLen, const uint32_t crossBlockNum)
{
    int32_t loopIter = DivCeil(dataLen * sizeof(typename Intf::DstT) * 4, UB_SIZE); // 4 : UB重排需要4块相同大小区域
    int32_t baseUseN = self->ctx.baseUseN_ / loopIter;  // 根据循环次数确定每次循环加载的baseUseN
    loopIter = DivCeil(self->ctx.baseUseN_, baseUseN);  // 重新计算loopIter, 由baseUseN_来确定
    int32_t tailUseN = self->ctx.baseUseN_ - (loopIter - 1) * baseUseN;
    int64_t baseDataLen = baseUseN * hwSize;
    int64_t tailDataLen = tailUseN * hwSize;
    int32_t hi = DivCeil(self->ctx.tiling_->baseM, self->ctx.splitWi_);
    int64_t hwAlign = static_cast<int64_t>(AlignUpByDtype(self->ctx.splitWi_, align32Byte)) * hi * self->ctx.tiling_->baseN;
    int64_t srcNOffset = GetAicBlockIdx() * hwAlign << crossBlockNum;
    int64_t dstNOffset = 0;

    event_t eventIdVecToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    event_t eventIdMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    for (int32_t i = 0; i < loopIter; i++) {
        if (likely(i != 0)) {
            WaitFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        }
        int64_t curDataLen = (i == loopIter - 1) ? tailDataLen : baseDataLen;
        int32_t curUseN = (i == loopIter - 1) ? tailUseN : baseUseN;
        LoadWorkSpaceDataToUb(self, hwSize, curUseN, srcNOffset);
        if constexpr(std::is_same<typename Intf::DstT, hifloat8_t>::value ||
            std::is_same<typename Intf::DstT, fp8_e4m3fn_t>::value) {
            InterleaveUbOutForKernelSplit<Intf, uint8_t>(self, curDataLen, crossBlockNum);
        } else {
            InterleaveUbOutForKernelSplit<Intf, typename Intf::DstT>(self, curDataLen, crossBlockNum);
        }

        SetFlag<HardEvent::V_MTE3>(eventIdVecToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVecToMte3);
        LoadUbToGmForKernelSplit(self, output, curUseN, hwSize, dstNOffset, crossBlockNum);
        srcNOffset += baseUseN * hwSize;
        dstNOffset += baseUseN * self->ctx.diHiWi_;
        if (likely(i != loopIter - 1)) {
            SetFlag<HardEvent::MTE3_MTE2>(eventIdMte3ToMte2);
        }
    }
}

template <class Intf>
__aicore__ inline void Rearrange2Gm(Intf *self, const GlobalTensor<typename Intf::DstT> &output,
                                         uint8_t enAtomic = 0, bool enSequentialWrite = false)
{
    if ASCEND_IS_AIC {
        return;
    }
    if (GetSubBlockIdx() > 0) {
        return;
    }
    if (!enSequentialWrite) {
        self->ctx.vecOutBuf_ = self->ctx.vecBuf_.template Get<typename Intf::DstT>();
        constexpr uint32_t crossBlockNum = 1; // 1: two corss blocks shift bit is 1, use to *2 and /2
        // UB->GM 重排数据，交叉排列
        uint64_t srcWi = (self->ctx.baseUseM_ < self->ctx.splitWi_) ? self->ctx.baseUseM_ : self->ctx.splitWi_;
        CalcCutInWIndex<Intf>(self, crossBlockNum);
        uint32_t align32Byte = 4; // 4: b16 需要对齐到16，2的4次方
        if constexpr(std::is_same<typename Intf::DstT, hifloat8_t>::value ||
            std::is_same<typename Intf::DstT, fp8_e4m3fn_t>::value) {
            align32Byte = 5; // 5: b8 需要对齐到32，2的5次方
        }
        int64_t hwSize = AlignUpByDtype(self->ctx.headWi_ >> crossBlockNum, align32Byte) +
            self->ctx.midHi_ * AlignUpByDtype(srcWi, align32Byte) +
            AlignUpByDtype(self->ctx.tailWi_ >> crossBlockNum, align32Byte);
        int64_t dataLen = hwSize * self->ctx.baseUseN_;
        if (self->ctx.kSUseWorkSpace_) {
            Rearrange2GmForL0cToWorkSpace(self, output, align32Byte, hwSize, dataLen, crossBlockNum);
        } else {
            if constexpr(std::is_same<typename Intf::DstT, hifloat8_t>::value ||
                std::is_same<typename Intf::DstT, fp8_e4m3fn_t>::value) {
                InterleaveUbOutForKernelSplit<Intf, uint8_t>(self, dataLen, crossBlockNum);
            } else {
                InterleaveUbOutForKernelSplit<Intf, typename Intf::DstT>(self, dataLen, crossBlockNum);
            }

            event_t eventIdVecToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
            SetFlag<HardEvent::V_MTE3>(eventIdVecToMte3);
            WaitFlag<HardEvent::V_MTE3>(eventIdVecToMte3);
            LoadUbToGmForKernelSplit(self, output, self->ctx.baseUseN_, hwSize, 0, crossBlockNum);
        }
    }
}

template <class Intf>
__aicore__ inline void CastToDstType(Intf *self, const GlobalTensor<typename Intf::DstT> &output,
                                         uint8_t enAtomic = 0, bool enSequentialWrite = false)
{
    if ASCEND_IS_AIC {
        return;
    }
    if (GetSubBlockIdx() > 0) {
        return;
    }
    // 单核切Dk时Iterate接口里DinIdx会超出范围，跳过
    if (self->ctx.curDinIdx_ >= self->ctx.curDinStartIdx_ + self->ctx.singleShapeDin_) {
        return;
    }
    if (!enSequentialWrite) {
        // workspace格式: D singleCoreCin/baseN singleCoreM/baseM baseN baseM
        uint64_t singleCoreCinAlignBaseN = AlignUp(self->ctx.tiling_->singleCoreCin, self->ctx.tiling_->baseN);
        uint64_t singleCoreMAlignBaseM = AlignUp(self->ctx.tiling_->singleCoreM, self->ctx.tiling_->baseM);
        uint64_t singleCoreWorkspaceSize = singleCoreCinAlignBaseN * singleCoreMAlignBaseM *
            self->ctx.tiling_->singleCoreDin;
        int64_t srcOffset = (self->ctx.curDinIdx_ - self->ctx.curDinStartIdx_) * singleCoreCinAlignBaseN *
            singleCoreMAlignBaseM + self->ctx.curNL0Idx_ * self->ctx.tiling_->baseN *
            singleCoreMAlignBaseM + self->ctx.curML0Idx_ * self->ctx.tiling_->baseN *
            self->ctx.tiling_->baseM + GetAicBlockIdx() * singleCoreWorkspaceSize;
        self->ctx.castVecTensor_ = self->ctx.vecBuf_.template Get<float>();
        DataCopyExtParams mte2Param;
        mte2Param.blockCount = 1;
        mte2Param.blockLen = self->ctx.baseUseM_ * self->ctx.baseUseN_ * sizeof(float);
        mte2Param.srcStride = 0;
        mte2Param.dstStride = 0;
        DataCopyPadExtParams<float> padParams {false, 0, 0, 0};
        DataCopyPad<float, PaddingMode::Compact>(self->ctx.castVecTensor_, self->ctx.l0cOutGm_[srcOffset], mte2Param, padParams);

        event_t eventIdMte2ToVec = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToVec);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToVec);

        // fp32 cast to Dst Type
        Cast(self->ctx.castVecTensor_.template ReinterpretCast<typename Intf::SrcT>(), self->ctx.castVecTensor_,
            RoundMode::CAST_RINT, self->ctx.baseUseM_ * self->ctx.baseUseN_);

        event_t eventIdVecToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventIdVecToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventIdVecToMte3);
        // ub(Dst Type, DCHW) -> GM(Dst Type, CDHW)
        uint64_t dstOffset = static_cast<uint64_t>(self->ctx.curNL0Idx_) * self->ctx.tiling_->baseN * self->ctx.diHiWi_ + // cin offset
                            static_cast<uint64_t>(self->ctx.curDinIdx_) * self->ctx.hiWi_ + // di offset
                            static_cast<uint64_t>(self->ctx.curML0Idx_) * self->ctx.tiling_->baseM; // hi&wi offset
        DataCopyExtParams mte3Param;
        mte3Param.blockCount = self->ctx.baseUseN_;
        mte3Param.blockLen = self->ctx.baseUseM_ * sizeof(typename Intf::DstT);
        mte3Param.srcStride = 0;
        mte3Param.dstStride = self->ctx.diHiWi_ * sizeof(typename Intf::DstT) - mte3Param.blockLen;
        DataCopyPad<typename Intf::DstT, PaddingMode::Compact>(output[dstOffset],
            self->ctx.castVecTensor_.template ReinterpretCast<typename Intf::SrcT>(), mte3Param);
    }
}

template <class Intf>
static __aicore__ inline void InitUbZero4Group(Intf *self)
{
    // Set ndVecBuf to zero.
    // size is cout1G * hk * wk * cinG * BLOCK_CUBE * c0
    self->ctx.ndVecTensor_ = self->ctx.ndVecBuf_.template Get<typename Intf::SrcT>();
    uint32_t groupHalfUbSize = self->ctx.tiling_->cout1G * self->ctx.hkWk_ * self->ctx.tiling_->cin1G *
        BLOCK_CUBE * sizeof(typename Intf::SrcT) << self->ctx.tiling_->c0BitsB;
    Duplicate<typename Intf::SrcT>(self->ctx.ndVecTensor_, 0, groupHalfUbSize / sizeof(typename Intf::SrcT));
}

/*
 * B matrix format: [enlarge*coutPerGroup, cinPerGroup, 1, hk, wk] -> [enlarge*coutPerGroup, enlarge*cinPerGroup, hk, wk]
 */
template <class Intf>
static __aicore__ inline void LoadUbDiag4GroupNCDHW(Intf *self, uint64_t srcGmOffset)
{
    if (unlikely(self->ctx.groupIterIdx_ == GetSubBlockIdx())) {
        uint32_t cinPerGroup = self->ctx.tiling_->cinG / self->ctx.tiling_->enlarge;
        uint32_t coutPerGroup = self->ctx.tiling_->coutG / self->ctx.tiling_->enlarge;
        // 因为扩维，所以是cinG
        uint64_t dstCoutGStride = (self->ctx.curEnlargeCin1_ * self->ctx.hkWk_) << SHIFT_BIT_4;
        // 因为对角化，所以dstEnlargeStride加上了cinPerGroup * hk * wk
        uint64_t dstEnlargeStride = coutPerGroup * dstCoutGStride + cinPerGroup * self->ctx.hkWk_;
        // NDDMA Loop0 params
        self->ctx.groupCopyParams_.loopInfo.loopSize[INDEX_0] = self->ctx.hkWk_;
        self->ctx.groupCopyParams_.loopInfo.loopSrcStride[INDEX_0] = 1;
        self->ctx.groupCopyParams_.loopInfo.loopDstStride[INDEX_0] = 1;
        // NDDMA Loop1 params
        self->ctx.groupCopyParams_.loopInfo.loopSize[INDEX_1] = cinPerGroup;
        self->ctx.groupCopyParams_.loopInfo.loopSrcStride[INDEX_1] = self->ctx.dkHkWk_;
        self->ctx.groupCopyParams_.loopInfo.loopDstStride[INDEX_1] = self->ctx.hkWk_;
        // NDDMA Loop2 params
        self->ctx.groupCopyParams_.loopInfo.loopSize[INDEX_2] = coutPerGroup;
        self->ctx.groupCopyParams_.loopInfo.loopSrcStride[INDEX_2] = cinPerGroup * self->ctx.dkHkWk_;
        self->ctx.groupCopyParams_.loopInfo.loopDstStride[INDEX_2] = dstCoutGStride;
        // NDDMA Loop3 params
        self->ctx.groupCopyParams_.loopInfo.loopSrcStride[INDEX_3] = coutPerGroup * cinPerGroup * self->ctx.dkHkWk_;
        self->ctx.groupCopyParams_.loopInfo.loopDstStride[INDEX_3] = dstEnlargeStride;
    }
    self->ctx.groupCopyParams_.loopInfo.loopSize[INDEX_3] = self->ctx.curEnlarge;
    DataCopy<typename Intf::SrcT, GROUP_NDDMA_DIM_NUM, nddmaConfig>(
        self->ctx.ndVecTensor_, self->ctx.weightGlobal_[srcGmOffset], self->ctx.groupCopyParams_);
}

template <class Intf>
static __aicore__ inline void LoadUbDiag4GroupNDHWC(Intf *self, uint64_t srcGmOffset)
{
    if (unlikely(self->ctx.groupIterIdx_ == GetSubBlockIdx())) {
        uint32_t cinPerGroup = self->ctx.tiling_->cinG / self->ctx.tiling_->enlarge;
        uint32_t coutPerGroup = self->ctx.tiling_->coutG / self->ctx.tiling_->enlarge;
        uint64_t dstCoutGStride = (self->ctx.curEnlargeCin1_ * self->ctx.hkWk_) << SHIFT_BIT_4;
        // NDDMA Loop0 params
        self->ctx.groupCopyParams_.loopInfo.loopSize[INDEX_0] = cinPerGroup;
        self->ctx.groupCopyParams_.loopInfo.loopSrcStride[INDEX_0] = 1;
        self->ctx.groupCopyParams_.loopInfo.loopDstStride[INDEX_0] = self->ctx.hkWk_;
        // NDDMA Loop1 params
        self->ctx.groupCopyParams_.loopInfo.loopSize[INDEX_1] = self->ctx.hkWk_;
        self->ctx.groupCopyParams_.loopInfo.loopSrcStride[INDEX_1] = cinPerGroup;
        self->ctx.groupCopyParams_.loopInfo.loopDstStride[INDEX_1] = 1;
        // NDDMA Loop2 params
        self->ctx.groupCopyParams_.loopInfo.loopSize[INDEX_2] = coutPerGroup;
        self->ctx.groupCopyParams_.loopInfo.loopSrcStride[INDEX_2] = cinPerGroup * self->ctx.dkHkWk_;
        self->ctx.groupCopyParams_.loopInfo.loopDstStride[INDEX_2] = dstCoutGStride;
        // NDDMA Loop3 params
        self->ctx.groupCopyParams_.loopInfo.loopSrcStride[INDEX_3] = coutPerGroup * cinPerGroup * self->ctx.dkHkWk_;
        self->ctx.groupCopyParams_.loopInfo.loopDstStride[INDEX_3] = coutPerGroup * dstCoutGStride + cinPerGroup * self->ctx.hkWk_;
    }
    self->ctx.groupCopyParams_.loopInfo.loopSize[INDEX_3] = self->ctx.curEnlarge;
    DataCopy<typename Intf::SrcT, GROUP_NDDMA_DIM_NUM, nddmaConfig>(
        self->ctx.ndVecTensor_, self->ctx.weightGlobal_[srcGmOffset], self->ctx.groupCopyParams_);
}

template <class Intf>
static __aicore__ inline void LoadUbDiag4GroupDHWCN(Intf *self, uint64_t srcGmOffset)
{
    if (unlikely(self->ctx.groupIterIdx_ == GetSubBlockIdx())) {
        uint32_t cinPerGroup = self->ctx.tiling_->cinG / self->ctx.tiling_->enlarge;
        uint32_t coutPerGroup = self->ctx.tiling_->coutG / self->ctx.tiling_->enlarge;
        uint64_t dstCoutGStride = (self->ctx.curEnlargeCin1_ * self->ctx.hkWk_) << SHIFT_BIT_4;
        // NDDMA Loop0 params
        self->ctx.groupCopyParams_.loopInfo.loopSize[INDEX_0] = coutPerGroup;
        self->ctx.groupCopyParams_.loopInfo.loopSrcStride[INDEX_0] = 1;
        self->ctx.groupCopyParams_.loopInfo.loopDstStride[INDEX_0] = dstCoutGStride;
        // NDDMA Loop1 params
        self->ctx.groupCopyParams_.loopInfo.loopSize[INDEX_1] = cinPerGroup;
        self->ctx.groupCopyParams_.loopInfo.loopSrcStride[INDEX_1] = self->ctx.tiling_->cout;
        self->ctx.groupCopyParams_.loopInfo.loopDstStride[INDEX_1] = self->ctx.hkWk_;
        // NDDMA Loop2 params
        self->ctx.groupCopyParams_.loopInfo.loopSize[INDEX_2] = self->ctx.hkWk_;
        self->ctx.groupCopyParams_.loopInfo.loopSrcStride[INDEX_2] = cinPerGroup * self->ctx.tiling_->cout;
        self->ctx.groupCopyParams_.loopInfo.loopDstStride[INDEX_2] = 1;
        // NDDMA Loop3 params
        self->ctx.groupCopyParams_.loopInfo.loopSrcStride[INDEX_3] = coutPerGroup;
        self->ctx.groupCopyParams_.loopInfo.loopDstStride[INDEX_3] = coutPerGroup * dstCoutGStride + cinPerGroup * self->ctx.hkWk_;
    }
    self->ctx.groupCopyParams_.loopInfo.loopSize[INDEX_3] = self->ctx.curEnlarge;
    DataCopy<typename Intf::SrcT, GROUP_NDDMA_DIM_NUM, nddmaConfig>(
        self->ctx.ndVecTensor_, self->ctx.weightGlobal_[srcGmOffset], self->ctx.groupCopyParams_);
}

template <class Intf>
static __aicore__ inline void SetGatherIdxDn2Nz(Intf *self)
{
    // cinG * hk * wk * 16 should be in uint16 range(ubSize constrain satisfy this)
    /* gen gather index for: [coutG, cinG, hk, wk] -> [coutG1, hk, wk, cinG1, cin0, cout0]
       [cinG * hk * wk * [0, 1, ..., c0 - 1] + 0 * hk * wk,
        cinG * hk * wk * [0, 1, ..., c0 - 1] + 1 * hk * wk,
        ...
        cinG * hk * wk * [0, 1, ..., c0 - 1] + (repeatTimes - 1) * hk * wk]
     */
    self->ctx.idxVecTensor_ = self->ctx.idxVecBuf_.template Get<typename Intf::IndexT>();
    typename Intf::IndexT idxVal = 0;
    // cinG * hk * wk * [0, 1, ..., c0 - 1]
    for (uint8_t idx = 0; idx < self->ctx.tiling_->c0; ++idx) {
        self->ctx.idxVecTensor_.SetValue(idx, idxVal);
        idxVal += static_cast<typename Intf::IndexT>((self->ctx.curEnlargeCin1_ * self->ctx.hkWk_) << SHIFT_BIT_4);
    }

    event_t eventId = static_cast<event_t>(self->ctx.pipe_.FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventId);
    WaitFlag<HardEvent::S_V>(eventId);

    auto idxAddr = (__ubuf__ typename Intf::IndexT *)self->ctx.idxVecTensor_.GetPhyAddr();
    uint16_t repeatTimes = static_cast<uint16_t>(
        AscendC::VECTOR_REG_WIDTH / (sizeof(typename Intf::IndexT) << self->ctx.tiling_->c0BitsB) - 1);
    uint16_t numPerRepeat = self->ctx.tiling_->c0;
    uint16_t dstOffset = self->ctx.tiling_->c0;
    uint32_t mask = self->ctx.tiling_->c0;
    auto cinStride = static_cast<typename Intf::IndexT>(self->ctx.hkWk_);

    __VEC_SCOPE__ {
        MicroAPI::RegTensor<typename Intf::IndexT> idxReg;
        MicroAPI::DataCopy<typename Intf::IndexT>(idxReg, idxAddr);
        MicroAPI::MaskReg maskReg = MicroAPI::UpdateMask<typename Intf::IndexT>(mask);

        for (uint16_t i = 0; i < repeatTimes; ++i) {
            // cinG * hk * wk * [0, 1, ..., c0 - 1] + (i + 1) * hk * wk
            MicroAPI::Adds<typename Intf::IndexT, typename Intf::IndexT>(idxReg, idxReg, cinStride, maskReg);
            MicroAPI::DataCopy<typename Intf::IndexT>(idxAddr + dstOffset, idxReg, maskReg);
            dstOffset += numPerRepeat;
        }
    }
}

template <class Intf>
static __aicore__ inline void Dn2Nz4Group(Intf *self)
{
    // [coutG, cinG, hk, wk] -> [cinG1, hk, wk, coutG1, cout0, cin0]
    // 其中coutG = enlarge * coutPerGroup, cinG = enlarge * cinPerGroup
    // gather index only need set once
    if (unlikely(self->ctx.groupIterIdx_ == GetSubBlockIdx())) {
        SetGatherIdxDn2Nz<Intf>(self);
        PipeBarrier<PIPE_V>();
    }

    uint32_t C0PerReg = AscendC::VECTOR_REG_WIDTH / (sizeof(typename Intf::IndexT) << self->ctx.tiling_->c0BitsB);
    uint16_t C0LoopTimes = BLOCK_CUBE / C0PerReg;
    uint32_t srcCout1GStride = (BLOCK_CUBE * self->ctx.curEnlargeCin1_ * self->ctx.hkWk_) << self->ctx.tiling_->c0BitsB;
    uint32_t srcCin1GStride = C0PerReg * self->ctx.hkWk_;
    uint32_t dstCout1GStride = (self->ctx.hkWk_ * self->ctx.curEnlargeCin1_ << self->ctx.tiling_->c0BitsB) << SHIFT_BIT_4;
    uint32_t dstKStride = (self->ctx.curEnlargeCin1_ << self->ctx.tiling_->c0BitsB) << SHIFT_BIT_4;
    uint32_t dstCin1GStride = C0PerReg << self->ctx.tiling_->c0BitsB;

    self->ctx.nzVecTensor_ = self->ctx.nzVecBuf_.template Get<typename Intf::SrcT>();

    auto idxAddr = (__ubuf__ typename Intf::IndexT *)self->ctx.idxVecTensor_.GetPhyAddr();
    auto srcAddr = (__ubuf__ typename Intf::SrcT *)self->ctx.ndVecTensor_.GetPhyAddr();
    auto dstAddr = (__ubuf__ typename Intf::SrcT *)self->ctx.nzVecTensor_.GetPhyAddr();

    // ub size is 253952, half is 126976, bfloat16 data num max is 63488, which is smaller than uint16_max.
    // that is cout1G * hk * wk * cinG * BLOCK_CUBE * c0 < 63488
    uint16_t cout1G = static_cast<uint16_t>(self->ctx.curEnlargeCout1_);
    uint16_t hkWk = static_cast<uint16_t>(self->ctx.hkWk_);
    uint16_t cin1G = static_cast<uint16_t>(self->ctx.curEnlargeCin1_);
    uint16_t cin1GIterMax = cin1G * C0LoopTimes;

    __VEC_SCOPE__ {
        MicroAPI::RegTensor<typename Intf::SrcT> gatherReg;
        MicroAPI::RegTensor<typename Intf::IndexT> idxReg;
        MicroAPI::MaskReg maskReg = MicroAPI::CreateMask<typename Intf::SrcT, MicroAPI::MaskPattern::ALL>();

        // copy index from ub to reg
        MicroAPI::DataCopy<typename Intf::IndexT>(idxReg, idxAddr);

        for (uint16_t cout1GIdx = 0; cout1GIdx < cout1G; ++cout1GIdx) {
            for (uint16_t kIdx = 0; kIdx < hkWk; ++kIdx) {
                for (uint16_t cin1GIdx = 0; cin1GIdx < cin1GIterMax; ++cin1GIdx) {
                    uint32_t srcOffset = cout1GIdx * srcCout1GStride + cin1GIdx * srcCin1GStride + kIdx;
                    uint32_t dstOffset = cout1GIdx * dstCout1GStride + kIdx * dstKStride + cin1GIdx * dstCin1GStride;

                    // gather data from ub to reg according to gather index
                    MicroAPI::DataCopyGather<typename Intf::SrcT, typename Intf::SrcT, typename Intf::IndexT>(
                        gatherReg, srcAddr + srcOffset, idxReg, maskReg);
                    // copy gather output data from reg to ub
                    MicroAPI::DataCopy<typename Intf::SrcT>(dstAddr + dstOffset, gatherReg, maskReg);
                }
            }
        }
    }
}

template <class Intf>
static __aicore__ inline void CopyUb2L14Group(Intf *self, uint32_t curCout1Size, uint32_t curCin1Cin0Size,
    uint32_t srcUbOffset, LocalTensor<typename Intf::SrcT> &useB1Buf)
{
    // from [coutG1, hk, wk, cinG1, cin0, cout0] extract [curCout1Size, hk, wk, curCin1Cin0Size, cout0],
    // copy data from ub to L1
    DataCopyParams copyParams;
    copyParams.blockCount = curCout1Size * self->ctx.hkWk_;
    copyParams.blockLen = curCin1Cin0Size;
    copyParams.srcStride = (self->ctx.curEnlargeCin1_ << SHIFT_BIT_4) - curCin1Cin0Size;
    copyParams.dstStride = 0;
    DataCopy(useB1Buf, self->ctx.nzVecTensor_[srcUbOffset], copyParams);
}

template <class Intf>
static __aicore__ inline void GroupTransdataWeightCore(Intf *self, uint32_t curCinSize, uint32_t curCoutSize,
    uint64_t srcGmOffset, uint32_t srcUbOffset, LocalTensor<typename Intf::SrcT> &useB1Buf)
{
    uint32_t curCin1Cin0Size = AlignUp16(curCinSize);
    uint32_t curCout1Size = DivCeilC0<Intf>(self, curCoutSize);
    uint32_t curCin1Size = DivCeil16(curCinSize);
#ifndef __CCE_KT_TEST__
    CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(FLAG_MTE1_ID_2);
#endif
    // vdup wait vgather
    if (self->ctx.groupIterIdx_ > GetSubBlockIdx()) {
        PipeBarrier<PIPE_V>();
    }
    InitUbZero4Group<Intf>(self); // vDup (pipe_v)

    // nddma wait vdup
    event_t eventId = static_cast<event_t>(self->ctx.pipe_.FetchEventID(HardEvent::V_MTE2));
    SetFlag<HardEvent::V_MTE2>(eventId);
    WaitFlag<HardEvent::V_MTE2>(eventId);
    if constexpr (Intf::Config::xType::format == Convolution3DBackprop::CubeFormat::NCDHW) {
        LoadUbDiag4GroupNCDHW<Intf>(self, srcGmOffset); // NDDMA (MTE2)
    } else if constexpr (Intf::Config::xType::format == Convolution3DBackprop::CubeFormat::NDHWC) {
        LoadUbDiag4GroupNDHWC<Intf>(self, srcGmOffset);
    } else {  // DHWCN
        LoadUbDiag4GroupDHWCN<Intf>(self, srcGmOffset);
    }

    // vgather wait nddma
    eventId = static_cast<event_t>(self->ctx.pipe_.FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventId);
    WaitFlag<HardEvent::MTE2_V>(eventId);
    Dn2Nz4Group<Intf>(self); // vGather (pipe_v)

    // next loop SetValue wait current loop vGather
    eventId = static_cast<event_t>(self->ctx.pipe_.FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventId);
    WaitFlag<HardEvent::V_S>(eventId);

    // ub2l1 wait vgather
    eventId = static_cast<event_t>(self->ctx.pipe_.FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(eventId);
    WaitFlag<HardEvent::V_MTE3>(eventId);
    CopyUb2L14Group<Intf>(self, curCout1Size, curCin1Cin0Size, srcUbOffset, useB1Buf); // MOV_UB_TO_L1 (MTE3)
#ifndef __CCE_KT_TEST__
    CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(FLAG_MTE1_ID_1);
#endif
}

template <class Intf>
__aicore__ inline void GroupTransdataWeight(Intf *self, uint32_t kIdx, uint32_t curDkIdx)
{
    if (GetSubBlockIdx() != 0) {
        return;
    }

    LocalTensor<typename Intf::SrcT> useB1Tbuf = GetB1Tbuf<Intf>(self, kIdx);

    uint32_t curCinIdx = self->ctx.curNL1Idx_ * self->ctx.tiling_->stepN * self->ctx.tiling_->baseN;
    uint32_t curCinSize = CalcCurCinSizeB1(self, curCinIdx);
    uint32_t curCoutIdx = 0;
    uint32_t curCoutSize = 0;
    CalcCoutIndexAndSizeB1(self, kIdx, curCoutIdx, curCoutSize);

    uint32_t curCin1Idx = (self->ctx.curCinStartIdx_ + curCinIdx) >> SHIFT_BIT_4;
    uint32_t curCout1Idx = (self->ctx.curCoutStartIdx_ + curCoutIdx) >> self->ctx.tiling_->c0BitsB;

    uint64_t srcGmOffset = 0;
    if constexpr (Intf::Config::xType::format == Convolution3DBackprop::CubeFormat::NCDHW) {
        srcGmOffset = static_cast<uint64_t>(curDkIdx) * self->ctx.hkWk_;
    } else if (Intf::Config::xType::format == Convolution3DBackprop::CubeFormat::NDHWC) {
        srcGmOffset = static_cast<uint64_t>(curDkIdx) * self->ctx.hkWk_ * self->ctx.tiling_->cinG / self->ctx.tiling_->enlarge;
    } else {  // DHWCN
        srcGmOffset = static_cast<uint64_t>(curDkIdx) * self->ctx.hkWk_ * self->ctx.tiling_->cinG / self->ctx.tiling_->enlarge *
            self->ctx.tiling_->cout;
    }
    uint32_t srcUbOffset = (curCout1Idx * self->ctx.hkWk_ * self->ctx.curEnlargeCin1_ +
        curCin1Idx) << self->ctx.tiling_->c0BitsB << SHIFT_BIT_4;
    // 调用指令不支持hif8，暂时隔离开，否则编译不通过
    if constexpr(!std::is_same<typename Intf::SrcT, hifloat8_t>::value &&
        !std::is_same<typename Intf::SrcT, fp8_e4m3fn_t>::value) {
        GroupTransdataWeightCore<Intf>(self, curCinSize, curCoutSize, srcGmOffset, srcUbOffset, useB1Tbuf);
    }
}

template <class Intf>
static __aicore__ inline void InitUbZero4C04(Intf *self, uint32_t b1CinSize)
{
    self->ctx.ndVecTensor_ = self->ctx.ndVecBuf_.template Get<typename Intf::SrcT>();
    uint32_t ubCinSize = (b1CinSize < self->ctx.vecBlockN_) ? b1CinSize : self->ctx.vecBlockN_;
    uint64_t ubPixCount = static_cast<uint64_t>(ubCinSize) * self->ctx.dkHkWk_;
    uint64_t ubOffset = DivDtypeByte<typename Intf::SrcT>(AscendC::ONE_BLOCK_SIZE);
    ubOffset += (self->ctx.tiling_->cout * self->ctx.vecBlockN_ * self->ctx.dkHkWk_);
    for (uint8_t i = self->ctx.tiling_->cout; i < C04_COUT_SIZE; i++) {
        Duplicate<typename Intf::SrcT>(self->ctx.ndVecTensor_[ubOffset], 0, ubPixCount);
        ubOffset += (self->ctx.vecBlockN_ * self->ctx.dkHkWk_);
    }
}

template <class Intf>
static __aicore__ inline void LoadUb4C04(Intf *self, uint32_t cinBlockSize, uint64_t srcGmOffset)
{
    LoopModeParams loopParams;
    loopParams.loop2Size = 1;
    loopParams.loop2SrcStride = 0;
    loopParams.loop2DstStride = 0;
    loopParams.loop1Size = 1;
    loopParams.loop1SrcStride = 0;
    loopParams.loop1DstStride = 0;
    SetLoopModePara(loopParams, DataCopyMVType::OUT_TO_UB);

    // The size of 'self->ctx.dkHkWk_ * sizeof(typename Intf::SrcT)' is not larger than the size of UB.
    // It will be checked in CheckC04Enable(), in range of uint32_t.
    uint32_t cinDataLen = static_cast<uint32_t>(self->ctx.dkHkWk_) * sizeof(typename Intf::SrcT);
    uint32_t oriBlockLen = cinBlockSize * cinDataLen;
    uint32_t alignedBlockLen = AlignUp(oriBlockLen, AscendC::ONE_BLOCK_SIZE);
    uint8_t rightPadding = DivDtypeByte<typename Intf::SrcT>(alignedBlockLen - oriBlockLen);
    DataCopyPadExtParams<typename Intf::SrcT> padParams {true, 0, rightPadding, 0};

    DataCopyExtParams gm2UbParams;
    gm2UbParams.blockLen = oriBlockLen;
    gm2UbParams.blockCount = self->ctx.tiling_->cout;
    gm2UbParams.srcStride = (self->ctx.tiling_->cin - cinBlockSize) * cinDataLen;
    gm2UbParams.dstStride = (self->ctx.vecBlockN_ * cinDataLen - alignedBlockLen) >> ONE_BLK_SHIFT_SIZE;

    uint32_t ubOffset = DivDtypeByte<typename Intf::SrcT>(AscendC::ONE_BLOCK_SIZE);
    DataCopyPad<typename Intf::SrcT, PaddingMode::Normal>(self->ctx.ndVecTensor_[ubOffset],
        self->ctx.weightGlobal_[srcGmOffset], gm2UbParams, padParams);
}

template <class Intf>
static __aicore__ inline void SetGatherIdx4C04(Intf *self)
{
    self->ctx.idxVecTensor_ = self->ctx.idxVecBuf_.template Get<typename Intf::IndexT>();
    typename Intf::IndexT idxVal = 0;
    uint8_t idx = 0;
    for (int8_t kernelIdx = self->ctx.tiling_->c0 / C04_COUT_SIZE - 1; kernelIdx >= 0; --kernelIdx) {
        idxVal = kernelIdx;
        for (uint8_t coutIdx = 0; coutIdx < C04_COUT_SIZE; ++coutIdx) {
            self->ctx.idxVecTensor_.SetValue(idx, idxVal);
            ++idx;
            idxVal += (self->ctx.vecBlockN_ * self->ctx.dkHkWk_);
        }
    }

    event_t eventId = static_cast<event_t>(self->ctx.pipe_.FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventId);
    WaitFlag<HardEvent::S_V>(eventId);

    auto idxAddr = (__ubuf__ typename Intf::IndexT *)self->ctx.idxVecTensor_.GetPhyAddr();
    uint16_t repeatTimes = static_cast<uint16_t>(
        DivDtypeByte<typename Intf::IndexT>(AscendC::VECTOR_REG_WIDTH) >> self->ctx.tiling_->c0BitsB) - 1;
    uint32_t mask = self->ctx.tiling_->c0;
    uint16_t dstOffset = self->ctx.tiling_->c0;
    auto cinStride = static_cast<typename Intf::IndexT>(self->ctx.dkHkWk_);

    __VEC_SCOPE__ {
        MicroAPI::RegTensor<typename Intf::IndexT> idxReg;
        MicroAPI::DataCopy<typename Intf::IndexT>(idxReg, idxAddr);
        MicroAPI::MaskReg maskReg = MicroAPI::UpdateMask<typename Intf::IndexT>(mask);

        for (uint16_t i = 0; i < repeatTimes; ++i) {
            MicroAPI::Adds<typename Intf::IndexT, typename Intf::IndexT>(idxReg, idxReg, cinStride, maskReg);
            MicroAPI::DataCopy<typename Intf::IndexT>(idxAddr + dstOffset, idxReg, maskReg);
            dstOffset += self->ctx.tiling_->c0;
        }
    }
}

template <class Intf>
static __aicore__ inline void SetGatherTailMask4C04(Intf *self)
{
    self->ctx.maskVecTensor_ = self->ctx.maskVecBuf_.template Get<uint32_t>();
    uint32_t maskVal = 0xffffffff;
    uint32_t kernelNumInC0 = self->ctx.tiling_->c0 >> C04_SHIFT_SIZE;
    uint32_t tmpVal = self->ctx.hkWk_ % kernelNumInC0;
    if constexpr(std::is_same<typename Intf::SrcT, float>::value) {
        if (tmpVal == 1) { // 最后一个分形，VGather只需要从UB中取hkwk_rev中的最后1个点
            maskVal = 0xffff;
        }
    } else if constexpr(std::is_same<typename Intf::SrcT, bfloat16_t>::value || std::is_same<typename Intf::SrcT, half>::value) {
        if (tmpVal == 3) { // 最后一个分形，VGather只需要从UB中取hkwk_rev中的最后3个点
            maskVal = 0xffffff;
        } else if (tmpVal == 2) { // 最后一个分形，VGather只需要从UB中取hkwk_rev中的最后2个点
            maskVal = 0xffff;
        } else if (tmpVal == 1) {
            maskVal = 0xff;
        }
    }
    uint8_t repeatTimes = static_cast<uint8_t>(
        DivDtypeByte<typename Intf::IndexT>(AscendC::VECTOR_REG_WIDTH) >> self->ctx.tiling_->c0BitsB);
    for (uint8_t idx = 0; idx < repeatTimes; ++idx) {
        self->ctx.maskVecTensor_.SetValue(idx, maskVal);
    }
}

template <class Intf>
static __aicore__ inline void SetIdxAndMask4C04(Intf *self, uint32_t loopIdx)
{
    // VGather的index只需要生成一次
    if (likely(self->ctx.c04LoadToB1IterIdx_ != GetSubBlockIdx() || loopIdx > 0)) {
        return;
    }
    SetGatherIdx4C04<Intf>(self);
    PipeBarrier<PIPE_V>();

    SetGatherTailMask4C04<Intf>(self);
    event_t eventId = static_cast<event_t>(self->ctx.pipe_.FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventId);
    WaitFlag<HardEvent::S_V>(eventId);
}

template <class Intf>
static __aicore__ inline void Dn2Nz4C04(Intf *self, uint32_t cinBlockSize, uint32_t curDkIdx, uint32_t loopIdx)
{
    SetIdxAndMask4C04<Intf>(self, loopIdx);

    uint32_t C0PerReg = DivDtypeByte<typename Intf::IndexT>(AscendC::VECTOR_REG_WIDTH) >> self->ctx.tiling_->c0BitsB;
    uint16_t C0LoopTimes = BLOCK_CUBE / C0PerReg;
    uint32_t kernelNumInC0 = self->ctx.tiling_->c0 >> C04_SHIFT_SIZE;
    uint16_t k1 = static_cast<uint16_t>(DivCeil(self->ctx.hkWk_, kernelNumInC0));
    uint16_t n1 = static_cast<uint16_t>(DivCeil16(cinBlockSize));
    uint16_t n1IterMax = n1 * C0LoopTimes;
    uint32_t srcN1Stride = C0PerReg * self->ctx.dkHkWk_;
    uint32_t srcK1Stride = kernelNumInC0;
    uint32_t dstN1Stride = C0PerReg << self->ctx.tiling_->c0BitsB;
    uint32_t dstK1Stride = AlignUp16(cinBlockSize) << self->ctx.tiling_->c0BitsB;

    self->ctx.nzVecTensor_ = self->ctx.nzVecBuf_.template Get<typename Intf::SrcT>();

    auto idxAddr = (__ubuf__ typename Intf::IndexT *)self->ctx.idxVecTensor_.GetPhyAddr();
    auto srcAddr = (__ubuf__ typename Intf::SrcT *)self->ctx.ndVecTensor_.GetPhyAddr();
    srcAddr += (DivDtypeByte<typename Intf::SrcT>(AscendC::ONE_BLOCK_SIZE) + curDkIdx * self->ctx.hkWk_);
    srcAddr -= ((k1 * kernelNumInC0) - self->ctx.hkWk_);
    auto dstAddr = (__ubuf__ typename Intf::SrcT *)self->ctx.nzVecTensor_.GetPhyAddr();
    auto maskAddr = (__ubuf__ uint32_t *)self->ctx.maskVecTensor_.GetPhyAddr();

    __VEC_SCOPE__ {
        MicroAPI::RegTensor<typename Intf::SrcT> gatherReg;
        MicroAPI::RegTensor<typename Intf::IndexT> idxReg;
        MicroAPI::MaskReg maskReg = MicroAPI::CreateMask<typename Intf::SrcT, MicroAPI::MaskPattern::ALL>();

        // copy index from ub to reg
        MicroAPI::DataCopy<typename Intf::IndexT>(idxReg, idxAddr);

        for (uint16_t k1Idx = k1; k1Idx > 1; --k1Idx) {
            for (uint16_t n1Idx = 0; n1Idx < n1IterMax; ++n1Idx) {
                uint32_t srcOffset = n1Idx * srcN1Stride + (k1Idx - 1) * srcK1Stride;
                uint32_t dstOffset = n1Idx * dstN1Stride + (k1 - k1Idx) * dstK1Stride;

                MicroAPI::DataCopyGather<typename Intf::SrcT, typename Intf::SrcT, typename Intf::IndexT>(
                    gatherReg, srcAddr + srcOffset, idxReg, maskReg);
                MicroAPI::DataCopy<typename Intf::SrcT>(dstAddr + dstOffset, gatherReg, maskReg);
            }
        }

        MicroAPI::MaskReg tailMaskReg = MicroAPI::CreateMask<typename Intf::SrcT, MicroAPI::MaskPattern::ALL>();
        // copy mask from ub to reg
        MicroAPI::DataCopy<uint32_t>(tailMaskReg, maskAddr);
        for (uint16_t n1Idx = 0; n1Idx < n1IterMax; ++n1Idx) {
            uint32_t srcOffset = n1Idx * srcN1Stride;
            uint32_t dstOffset = n1Idx * dstN1Stride + (k1 - 1) * dstK1Stride;

            MicroAPI::DataCopyGather<typename Intf::SrcT, typename Intf::SrcT, typename Intf::IndexT>(
                gatherReg, srcAddr + srcOffset, idxReg, tailMaskReg);
            MicroAPI::DataCopy<typename Intf::SrcT>(dstAddr + dstOffset, gatherReg, maskReg);
        }
    }
}

template <class Intf>
static __aicore__ inline void CopyUb2L14C04(Intf *self, uint32_t cinBlockSize, uint32_t b1CinSize,
    LocalTensor<typename Intf::SrcT> &useB1Buf, uint32_t dstB1Offset)
{
    if (cinBlockSize == b1CinSize) { // UB上的格式转换不需要切块
        DataCopyParams copyParams;
        copyParams.blockCount = 1;
        copyParams.blockLen = (AlignUp16(cinBlockSize) * AlignUp(self->ctx.hkWk_ * C04_COUT_SIZE, self->ctx.tiling_->c0) *
            sizeof(typename Intf::SrcT)) >> ONE_BLK_SHIFT_SIZE;
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        DataCopy(useB1Buf, self->ctx.nzVecTensor_, copyParams);
    } else {
        DataCopyParams copyParams;
        uint32_t kernelNumInC0 = self->ctx.tiling_->c0 >> C04_SHIFT_SIZE;
        copyParams.blockCount = DivCeil(self->ctx.hkWk_, kernelNumInC0);
        copyParams.blockLen = AlignUp16(cinBlockSize);
        copyParams.srcStride = 0;
        copyParams.dstStride = AlignUp16(b1CinSize) - AlignUp16(cinBlockSize);
        DataCopy(useB1Buf[dstB1Offset], self->ctx.nzVecTensor_, copyParams);
    }
}

template <class Intf>
static __aicore__ inline void C04TransdataWeightCore(Intf *self, uint32_t b1CinSize, uint64_t srcGmOffset,
    LocalTensor<typename Intf::SrcT> &useB1Buf, uint32_t curDkIdx)
{
    uint32_t loopCnt = DivCeil(b1CinSize, self->ctx.vecBlockN_);
    uint32_t cinRemain = b1CinSize;
    uint32_t cinBlockSize = (cinRemain < self->ctx.vecBlockN_) ? cinRemain : self->ctx.vecBlockN_;
    uint32_t dstB1Offset = 0;
    for (uint32_t i = 0; i < loopCnt; i++) {
        // DataCopy wait VDup or prev loop VGather
        event_t eventId = static_cast<event_t>(self->ctx.pipe_.FetchEventID(HardEvent::V_MTE2));
        SetFlag<HardEvent::V_MTE2>(eventId);
        WaitFlag<HardEvent::V_MTE2>(eventId);
        // DataCopy (MTE2)
        LoadUb4C04<Intf>(self, cinBlockSize, srcGmOffset);
        srcGmOffset += (static_cast<uint64_t>(cinBlockSize) * self->ctx.dkHkWk_);

        // VGather wait DataCopy
        eventId = static_cast<event_t>(self->ctx.pipe_.FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventId);
        WaitFlag<HardEvent::MTE2_V>(eventId);
        // VGather wait prev loop MOV_UB_TO_L1
        eventId = static_cast<event_t>(self->ctx.pipe_.FetchEventID(HardEvent::MTE3_V));
        SetFlag<HardEvent::MTE3_V>(eventId);
        WaitFlag<HardEvent::MTE3_V>(eventId);
        // VGather
        Dn2Nz4C04<Intf>(self, cinBlockSize, curDkIdx, i);

        // MOV_UB_TO_L1 wait prev VGather
        eventId = static_cast<event_t>(self->ctx.pipe_.FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(eventId);
        WaitFlag<HardEvent::V_MTE3>(eventId);
        // MOV_UB_TO_L1 (MTE3)
        CopyUb2L14C04<Intf>(self, cinBlockSize, b1CinSize, useB1Buf, dstB1Offset);
        dstB1Offset += (AlignUp16(cinBlockSize) << self->ctx.tiling_->c0BitsB);

        cinRemain -= cinBlockSize;
        cinBlockSize = (cinRemain < self->ctx.vecBlockN_) ? cinRemain : self->ctx.vecBlockN_;
    }
}

template <class Intf>
__aicore__ inline void C04TransdataWeight(Intf *self, const uint32_t kIdx, uint32_t curDkIdx)
{
    WaitForCubeBeforeLoadToB1<Intf>(self);
    LocalTensor<typename Intf::SrcT> useB1Tbuf = GetB1Tbuf<Intf>(self, kIdx);

    if (GetSubBlockIdx() == (self->ctx.c04LoadToB1IterIdx_ & 1)) {
        uint32_t curCinIdx = self->ctx.curNL1Idx_ * self->ctx.tiling_->stepN * self->ctx.tiling_->baseN;
        uint32_t b1CinSize = CalcCurCinSizeB1(self, curCinIdx);
        uint64_t srcGmOffset = static_cast<uint64_t>(curCinIdx) * self->ctx.dkHkWk_;

        // 调用指令不支持hif8，暂时隔离开，否则编译不通过
        if constexpr(!std::is_same<typename Intf::SrcT, hifloat8_t>::value &&
            !std::is_same<typename Intf::SrcT, fp8_e4m3fn_t>::value) {
            // 每个AIV只在第一次计算时需要清零
            if (GetSubBlockIdx() == self->ctx.c04LoadToB1IterIdx_) {
                InitUbZero4C04<Intf>(self, b1CinSize); // VDup
            }
            C04TransdataWeightCore<Intf>(self, b1CinSize, srcGmOffset, useB1Tbuf, curDkIdx);
        }
    }

    NotifyCubeAfterLoadToB1<Intf>(self);
}

template <class Intf>
__aicore__ inline void InitUbByteSize(Intf *self)
{
#if __CCE_AICORE__ == 310
    if constexpr (Intf::conv3dConfig.kernelSplitMode == TPL_SPLIT_KERNEL_HW) {
        if (GetSubBlockIdx() != 0) {
            return;
        }
        if (self->ctx.kSUseWorkSpace_) {
            if ASCEND_IS_AIV {
                self->ctx.pipe_.InitBuffer(self->ctx.vecBuf_, UB_SIZE);
            }
        } else {
            self->ctx.pipe_.InitBuffer(self->ctx.vecBuf_, UB_SIZE);
        }
    } else if constexpr (Intf::conv3dConfig.groupMode == TPL_GROUP_MODE_ENLARGE) {
        if ASCEND_IS_AIV {
            constexpr uint32_t GROUP_UB_BUF_SIZE = (UB_SIZE - AscendC::VECTOR_REG_WIDTH) / HALF_FACTOR;
            self->ctx.pipe_.InitBuffer(self->ctx.ndVecBuf_, GROUP_UB_BUF_SIZE);
            self->ctx.pipe_.InitBuffer(self->ctx.nzVecBuf_, GROUP_UB_BUF_SIZE);
            self->ctx.pipe_.InitBuffer(self->ctx.idxVecBuf_, AscendC::VECTOR_REG_WIDTH);
        }
    } else if constexpr (Intf::conv3dConfig.enableC04Flag) {
        if ASCEND_IS_AIV {
            constexpr uint32_t C04_UB_BUF_SIZE = (UB_SIZE - AscendC::VECTOR_REG_WIDTH -
                MASK_REG_WIDTH - AscendC::ONE_BLOCK_SIZE) >> 1;
            self->ctx.pipe_.InitBuffer(self->ctx.ndVecBuf_, C04_UB_BUF_SIZE);
            self->ctx.pipe_.InitBuffer(self->ctx.nzVecBuf_, C04_UB_BUF_SIZE);
            self->ctx.pipe_.InitBuffer(self->ctx.idxVecBuf_, AscendC::VECTOR_REG_WIDTH);
            self->ctx.pipe_.InitBuffer(self->ctx.maskVecBuf_, MASK_REG_WIDTH);
        }
    } else {
        if ASCEND_IS_AIV {
            self->ctx.pipe_.InitBuffer(self->ctx.vecBuf_, UB_SIZE);
        }
    }
#endif
}

template <class Intf>
__aicore__ inline bool IsL1bUseTQue(Intf *self)
{
    if constexpr (Intf::conv3dConfig.groupMode != TPL_GROUP_MODE_ENLARGE &&
        !Intf::conv3dConfig.enableC04Flag) {
        return true;
    }
    return false;
}

template <class Intf>
__aicore__ inline void CrossCoreSetHead(Intf *self)
{
#ifndef __CCE_KT_TEST__
    if ASCEND_IS_AIC {
        if constexpr (Intf::conv3dConfig.groupMode == TPL_GROUP_MODE_ENLARGE) {
            CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(FLAG_MTE1_ID_2);
        }
    }
#endif
}

template <class Intf>
__aicore__ inline void CrossCoreWaitTail(Intf *self)
{
#ifndef __CCE_KT_TEST__
    if ASCEND_IS_AIV {
        if constexpr (Intf::conv3dConfig.groupMode == TPL_GROUP_MODE_ENLARGE) {
            if (GetSubBlockIdx() == 0) {
                CrossCoreWaitFlag<SYNC_MODE, PIPE_V>(FLAG_MTE1_ID_2);
            }
        }
    }
#endif
}

template <class Intf>
__aicore__ inline void CrossCoreSetHeadForMix(Intf *self)
{
#ifndef __CCE_KT_TEST__
    if ASCEND_IS_AIV {
        if constexpr (Intf::conv3dConfig.kernelSplitMode == TPL_SPLIT_KERNEL_HW) {
            CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(FLAG_FIXP_ID);
        } else if (self->ctx.enableSplitDk_) {
            CrossCoreSetFlag<SYNC_MODE, PIPE_MTE3>(FLAG_FIXP_ID);
        }
    }
    if ASCEND_IS_AIC {
        if constexpr (Intf::conv3dConfig.enableC04Flag) {
            CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(FLAG_MTE1_ID_2);
            CrossCoreSetFlag<SYNC_MODE, PIPE_MTE1>(FLAG_MTE1_ID_2 + CROSS_CORE_FLAG_ID_MAX);
        }
    }
#endif
}

template <class Intf>
__aicore__ inline void CrossCoreWaitTailForMix(Intf *self)
{
#ifndef __CCE_KT_TEST__
    if ASCEND_IS_AIC {
        if constexpr (Intf::conv3dConfig.kernelSplitMode == TPL_SPLIT_KERNEL_HW) {
            CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(FLAG_FIXP_ID);
        } else if (self->ctx.enableSplitDk_) {
            CrossCoreWaitFlag<SYNC_MODE, PIPE_FIX>(FLAG_FIXP_ID);
        }
    }
    if ASCEND_IS_AIV {
        if constexpr (Intf::conv3dConfig.enableC04Flag) {
            CrossCoreWaitFlag<SYNC_MODE, PIPE_MTE3>(FLAG_MTE1_ID_2);
        }
    }
#endif
}
}  // namespace Convolution3DBackpropFunc

#endif
