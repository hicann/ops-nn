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
 * \file conv_bp_input_sub_func_kernelsplit_hw.h
 * \brief KernelSplit HW L0c-to-UB/GM rearrangement for conv3d backprop input
 */

#ifndef CONV3D_BP_INPUT_SUB_FUNC_KERNELSPLIT_HW_H
#define CONV3D_BP_INPUT_SUB_FUNC_KERNELSPLIT_HW_H

#include "conv_bp_input_sub_func_utils.h"
#include "conv_bp_input_sub_func_index_calc.h"
#include "conv_bp_input_sub_func_store_l0c_fixpipe.h"
#include "conv_bp_input_sub_func_vector_intrinsics.h"

using AscendC::DivCeil;
using AscendC::FixpipeParamsC310;
using AscendC::GlobalTensor;
using AscendC::LocalTensor;

namespace Convolution3DBackpropFunc {

template <class Intf>
static __aicore__ inline void LoadL0c2GMForKernelSplitInner(Intf* self,
                                                            const LocalTensor<typename Intf::L0cT>& useC1Buf,
                                                            FixpipeParamsC310<CO2Layout::COLUMN_MAJOR>& fixPipeParams,
                                                            const uint32_t crossBlockNum)
{
    uint32_t align32Byte = 4; // 4: b16 需要对齐到16，2的4次方
    if constexpr (std::is_same<typename Intf::DstT, hifloat8_t>::value ||
                  std::is_same<typename Intf::DstT, fp8_e4m3fn_t>::value ||
                  std::is_same<typename Intf::DstT, int8_t>::value) {
        align32Byte = 5; // 5: b8 需要对齐到32，2的5次方
    }

    uint32_t hi = DivCeil(self->ctx.tiling_->baseM, self->ctx.splitWi_);
    uint64_t hwAlign = static_cast<uint64_t>(AlignUpByDtype(self->ctx.splitWi_, align32Byte)) * hi *
                       self->ctx.tiling_->baseN;
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
        fixPipeParams.params.dnNum = 1;             // not use
        fixPipeParams.params.srcNzMatrixStride = 0; // loop3_src_stride
        fixPipeParams.params.dstDnMatrixStride = 0; // loop3_dst_stride

        fixPipeParams.mSize = realHeadWi; // M: 首块的长度, 真实的headwi
        LoadL0c2GMFixPipe(self, srcOffset, wsDstOffset, self->ctx.l0cOutWorkspace_, useC1Buf, fixPipeParams);
        // BLOCK_CUBE: MMAD一次计算为16*16, fixpipe搬到ub的时候取L0c的数据应该固定c0为16，不能随数据类型变化
        srcOffset += realHeadWi << 4; // headWi_/2是一个子kernel首块w的长度
        wsDstOffset += alignHeadWi;
    }

    if (self->ctx.midHi_ != 0) {                             // 需要中间块
        fixPipeParams.params.dnNum = self->ctx.midHi_;       // 循环中间块的行数
        fixPipeParams.params.srcNzMatrixStride = srcWi;      // loop3_src_stride
        fixPipeParams.params.dstDnMatrixStride = alignMidWi; // loop3_dst_stride

        fixPipeParams.mSize = srcWi; // M: 中间块一行的长度, 真实的midwi
        LoadL0c2GMFixPipe(self, srcOffset, wsDstOffset, self->ctx.l0cOutWorkspace_, useC1Buf, fixPipeParams);
        // BLOCK_CUBE: MMAD一次计算为16*16, fixpipe搬到ub的时候取L0c的数据应该固定c0为16，不能随数据类型变化
        srcOffset += self->ctx.midHi_ * srcWi << 4; // srcWi是一个子kernel中间块w的长度
        wsDstOffset += static_cast<int64_t>(self->ctx.midHi_) * alignMidWi;
    }

    if (self->ctx.tailWi_ != 0) {                   // 需要尾块
        fixPipeParams.params.dnNum = 1;             // not use
        fixPipeParams.params.srcNzMatrixStride = 0; // loop3_src_stride
        fixPipeParams.params.dstDnMatrixStride = 0; // loop3_dst_stride

        fixPipeParams.mSize = realTailWi; // M: 尾块的长度, 真实的tailwi
        LoadL0c2GMFixPipe(self, srcOffset, wsDstOffset, self->ctx.l0cOutWorkspace_, useC1Buf, fixPipeParams);
    }
}

template <class Intf>
static __aicore__ inline void LoadL0c2UbForKernelSplitInner(Intf* self,
                                                            const LocalTensor<typename Intf::L0cT>& useC1Buf,
                                                            FixpipeParamsC310<CO2Layout::COLUMN_MAJOR>& fixPipeParams,
                                                            const uint32_t crossBlockNum)
{
    self->ctx.vecOutBuf_ = self->ctx.vecBuf_.template Get<typename Intf::DstT>();
    uint32_t align32Byte = 4; // 4: b16 需要对齐到16，2的4次方
    if constexpr (std::is_same<typename Intf::DstT, hifloat8_t>::value ||
                  std::is_same<typename Intf::DstT, fp8_e4m3fn_t>::value ||
                  std::is_same<typename Intf::DstT, int8_t>::value) {
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
        fixPipeParams.params.dnNum = 1;             // not use
        fixPipeParams.params.srcNzMatrixStride = 0; // loop3_src_stride
        fixPipeParams.params.dstDnMatrixStride = 0; // loop3_dst_stride

        fixPipeParams.mSize = alignHeadWi; // M: 首块的长度
        LoadL0c2UbFixPipe(self, srcOffset, ubDstOffset, self->ctx.vecOutBuf_, useC1Buf, fixPipeParams);
        // BLOCK_CUBE: MMAD一次计算为16*16, fixpipe搬到ub的时候取L0c的数据应该固定c0为16，不能随数据类型变化
        srcOffset += realHeadWi << 4; // headWi_/2是一个子kernel首块w的长度
        ubDstOffset += alignHeadWi;
    }

    if (self->ctx.midHi_ != 0) {                             // 需要中间块
        fixPipeParams.params.dnNum = self->ctx.midHi_;       // 循环中间块的行数
        fixPipeParams.params.srcNzMatrixStride = srcWi;      // loop3_src_stride
        fixPipeParams.params.dstDnMatrixStride = alignMidWi; // loop3_dst_stride

        fixPipeParams.mSize = alignMidWi; // M: 中间块一行的长度
        LoadL0c2UbFixPipe(self, srcOffset, ubDstOffset, self->ctx.vecOutBuf_, useC1Buf, fixPipeParams);
        // BLOCK_CUBE: MMAD一次计算为16*16, fixpipe搬到ub的时候取L0c的数据应该固定c0为16，不能随数据类型变化
        srcOffset += self->ctx.midHi_ * srcWi << 4; // srcWi是一个子kernel中间块w的长度
        ubDstOffset += self->ctx.midHi_ * alignMidWi;
    }

    if (self->ctx.tailWi_ != 0) {                   // 需要尾块
        fixPipeParams.params.dnNum = 1;             // not use
        fixPipeParams.params.srcNzMatrixStride = 0; // loop3_src_stride
        fixPipeParams.params.dstDnMatrixStride = 0; // loop3_dst_stride

        fixPipeParams.mSize = AlignUpByDtype(realTailWi, align32Byte); // M: 尾块的长度
        LoadL0c2UbFixPipe(self, srcOffset, ubDstOffset, self->ctx.vecOutBuf_, useC1Buf, fixPipeParams);
    }
}

// kernel拆分后, 输出到Ub上进行重排
template <class Intf>
static __aicore__ inline void LoadL0c2OutForKernelSplitHW(Intf* self, const LocalTensor<typename Intf::L0cT>& useC1Buf)
{
    if (GetSubBlockIdx() > 0) {
        return;
    }
    constexpr uint32_t crossBlockNum = 1; // 1: two corss blocks shift bit is 1, use to *2 and /2
    CalcCutInWIndex<Intf>(self, crossBlockNum);

    FixpipeParamsC310<CO2Layout::COLUMN_MAJOR> fixPipeParams;
    SetFixPipeQuantVal<Intf>(self, fixPipeParams);
    fixPipeParams.params.srcNzC0Stride = 1;    // src M stride, loop0_src_stride (unit: 32B)
    fixPipeParams.nSize = self->ctx.baseUseN_; // N: cin
    fixPipeParams.reluEn = self->ctx.tiling_->enRelu;
#if __FIXED_POINT_ONLY_CUBE_TO_L0C__
    fixPipeParams.preReluMode = static_cast<ReluMode>(self->ctx.tiling_->enRelu);
#endif
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
static __aicore__ inline void SetDataCopyPadLoopInfo(Intf* self, int64_t hwSize, const uint32_t curUseN,
                                                     const uint32_t crossBlockNum)
{
    LoopModeParams loopParams;
    loopParams.loop2Size = 1; // not use loop2Size
    loopParams.loop2SrcStride = 0;
    loopParams.loop2DstStride = 0;
    loopParams.loop1Size = curUseN;
    loopParams.loop1SrcStride = (hwSize << crossBlockNum) * sizeof(typename Intf::DstT);
    loopParams.loop1DstStride = self->ctx.diHiWi_ * sizeof(typename Intf::DstT);
    SetLoopModePara(loopParams, DataCopyMVType::UB_TO_OUT);
}

template <class Intf>
static __aicore__ inline void DataCopyUbToGmForKernelSplit(Intf* self, const GlobalTensor<typename Intf::DstT>& output,
                                                           int64_t srcOffset, int64_t dstOffset,
                                                           const uint32_t crossBlockNum)
{
    uint32_t wiUsed = AlignUp(self->ctx.tiling_->wi, self->ctx.tiling_->strideW);
    uint32_t align32Byte = 4; // 4: b16 需要对齐到16，2的4次方
    if constexpr (std::is_same<typename Intf::DstT, hifloat8_t>::value ||
                  std::is_same<typename Intf::DstT, fp8_e4m3fn_t>::value ||
                  std::is_same<typename Intf::DstT, int8_t>::value) {
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
        ub2GmParams.srcStride = ((alignWi << crossBlockNum) - wiUsed) * sizeof(typename Intf::DstT) >>
                                ONE_BLK_SHIFT_SIZE; // 32Bytes one datablock
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
static __aicore__ inline void LoadUbToGmForKernelSplit(Intf* self, const GlobalTensor<typename Intf::DstT>& output,
                                                       const uint32_t curUseN, const int64_t hwSize,
                                                       const int64_t dstNOffset, const uint32_t crossBlockNum)
{
    uint32_t wiUsed = AlignUp(self->ctx.tiling_->wi, self->ctx.tiling_->strideW);
    uint32_t mSize = self->ctx.curMIdx_ * self->ctx.tiling_->baseM << crossBlockNum;
    uint32_t hiUsed = mSize / wiUsed;
    uint32_t mUseSize = hiUsed * self->ctx.tiling_->wi + mSize - hiUsed * wiUsed;
    uint32_t curSkipMSize = (mSize / wiUsed) * self->ctx.tiling_->wi; // GM上隔行需要空开的数量
    uint64_t dstOffset = static_cast<uint64_t>(self->ctx.curNIdx_) * self->ctx.tiling_->baseN *
                             self->ctx.diHiWi_ + // cin offset
                         static_cast<uint64_t>(self->ctx.curDinIdx_) *
                             self->ctx.hiWi_ + // di offset, remove useless data
                         mUseSize +
                         curSkipMSize + // hi&wi offset
                         self->ctx.rearrangeHIndex_ * self->ctx.tiling_->wi + dstNOffset;

    SetDataCopyPadLoopInfo(self, hwSize, curUseN, crossBlockNum);
    int64_t dataLen = curUseN * hwSize;
    int64_t srcOffset = (static_cast<uint64_t>(dataLen) << crossBlockNum);
    DataCopyUbToGmForKernelSplit(self, output, srcOffset, dstOffset, crossBlockNum);
    ResetLoopModePara(DataCopyMVType::UB_TO_OUT);
}

template <class Intf, class ReDstT>
__aicore__ inline void InterleaveUbOutForKernelSplit(Intf* self, int64_t dataLen, const uint32_t crossBlockNum)
{
    uint32_t vfLen = platform::GetVRegSize() / sizeof(ReDstT);
    uint32_t doubleVfLen = (vfLen << crossBlockNum);
    uint16_t repeatTimes = (dataLen + vfLen - 1) / vfLen;
    uint64_t twoBlockLen = (dataLen << crossBlockNum);
    bool kernelFlag1 = (self->ctx.tiling_->wk == 1 && self->ctx.tiling_->hk == 1);
    auto src0Ptr = (__ubuf__ ReDstT*)self->ctx.vecOutBuf_[0].GetPhyAddr();
    auto src1Ptr = (__ubuf__ ReDstT*)self->ctx.vecOutBuf_[dataLen].GetPhyAddr();
    auto dst0Ptr = (__ubuf__ ReDstT*)self->ctx.vecOutBuf_[twoBlockLen].GetPhyAddr();
    auto dst1Ptr = (__ubuf__ ReDstT*)self->ctx.vecOutBuf_[twoBlockLen + vfLen].GetPhyAddr();

    // 取两个kernel拆分的切块(CHW)使用interleave进行交叉排布
    if (likely(!kernelFlag1)) {
        InterleaveKernelSplitNormal<ReDstT>(src0Ptr, src1Ptr, dst0Ptr, dst1Ptr, vfLen, doubleVfLen, repeatTimes);
    } else {
        // kernel = 1*1场景下需要对第二个Reg清零
        InterleaveKernelSplit1x1<ReDstT>(src0Ptr, dst0Ptr, dst1Ptr, vfLen, doubleVfLen, repeatTimes);
    }
}

template <class Intf>
__aicore__ inline void LoadWorkSpaceDataToUbInner(Intf* self, int64_t srcOffset, int64_t dstOffset,
                                                  const DataCopyExtParams& mte2Param)
{
    if constexpr (std::is_same<typename Intf::DstT, hifloat8_t>::value ||
                  std::is_same<typename Intf::DstT, fp8_e4m3fn_t>::value ||
                  std::is_same<typename Intf::DstT, int8_t>::value) {
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
__aicore__ inline void LoadWorkSpaceDataToUb(Intf* self, const int64_t hwSize, const uint32_t curUseN,
                                             int64_t srcOffset)
{
    self->ctx.vecOutBuf_ = self->ctx.vecBuf_.template Get<typename Intf::DstT>();
    DataCopyExtParams mte2Param;
    mte2Param.blockCount = 1;
    mte2Param.blockLen = curUseN * hwSize * sizeof(typename Intf::DstT);
    mte2Param.srcStride = 0;
    mte2Param.dstStride = 0;
    LoadWorkSpaceDataToUbInner(self, srcOffset, 0, mte2Param);
    if (self->ctx.tiling_->wk != 1 || self->ctx.tiling_->hk != 1) { // kernel=1*1 只需要搬一块
        srcOffset += hwSize * self->ctx.baseUseN_;
        int64_t dstOffset = hwSize * curUseN;
        LoadWorkSpaceDataToUbInner(self, srcOffset, dstOffset, mte2Param);
    }
    event_t eventIdMte2ToVec = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(eventIdMte2ToVec);
    WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToVec);
}

template <class Intf>
__aicore__ inline void Rearrange2GmForL0cToWorkSpace(Intf* self, const GlobalTensor<typename Intf::DstT>& output,
                                                     uint32_t align32Byte, int64_t hwSize, int64_t dataLen,
                                                     const uint32_t crossBlockNum)
{
    int32_t loopIter = DivCeil(dataLen * sizeof(typename Intf::DstT) * 4, UB_SIZE); // 4 : UB重排需要4块相同大小区域
    int32_t baseUseN = self->ctx.baseUseN_ / loopIter; // 根据循环次数确定每次循环加载的baseUseN
    loopIter = DivCeil(self->ctx.baseUseN_, baseUseN); // 重新计算loopIter, 由baseUseN_来确定
    int32_t tailUseN = self->ctx.baseUseN_ - (loopIter - 1) * baseUseN;
    int64_t baseDataLen = baseUseN * hwSize;
    int64_t tailDataLen = tailUseN * hwSize;
    int32_t hi = DivCeil(self->ctx.tiling_->baseM, self->ctx.splitWi_);
    int64_t hwAlign = static_cast<int64_t>(AlignUpByDtype(self->ctx.splitWi_, align32Byte)) * hi *
                      self->ctx.tiling_->baseN;
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
        if constexpr (std::is_same<typename Intf::DstT, hifloat8_t>::value ||
                      std::is_same<typename Intf::DstT, fp8_e4m3fn_t>::value ||
                      std::is_same<typename Intf::DstT, int8_t>::value) {
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
__aicore__ inline void Rearrange2Gm(Intf* self, const GlobalTensor<typename Intf::DstT>& output, uint8_t enAtomic = 0,
                                    bool enSequentialWrite = false)
{
    if ASCEND_IS_AIC_SHOULD_RETURN {
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
        if constexpr (std::is_same<typename Intf::DstT, hifloat8_t>::value ||
                      std::is_same<typename Intf::DstT, fp8_e4m3fn_t>::value ||
                      std::is_same<typename Intf::DstT, int8_t>::value) {
            align32Byte = 5; // 5: b8 需要对齐到32，2的5次方
        }
        int64_t hwSize = AlignUpByDtype(self->ctx.headWi_ >> crossBlockNum, align32Byte) +
                         self->ctx.midHi_ * AlignUpByDtype(srcWi, align32Byte) +
                         AlignUpByDtype(self->ctx.tailWi_ >> crossBlockNum, align32Byte);
        int64_t dataLen = hwSize * self->ctx.baseUseN_;
        if (self->ctx.kSUseWorkSpace_) {
            Rearrange2GmForL0cToWorkSpace(self, output, align32Byte, hwSize, dataLen, crossBlockNum);
        } else {
            if constexpr (std::is_same<typename Intf::DstT, hifloat8_t>::value ||
                          std::is_same<typename Intf::DstT, fp8_e4m3fn_t>::value ||
                          std::is_same<typename Intf::DstT, int8_t>::value) {
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

} // namespace Convolution3DBackpropFunc

#endif
