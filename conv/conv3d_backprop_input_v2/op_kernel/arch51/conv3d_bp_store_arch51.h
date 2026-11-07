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
 * \file conv3d_bp_store_arch51.h
 * \brief
 */

#ifndef CONV3D_BP_STORE_ARCH51_H
#define CONV3D_BP_STORE_ARCH51_H

namespace Convolution3DBackpropFunc {

template <class Intf>
static __aicore__ inline void StoreL1Result(Intf* self, const GlobalTensor<typename Intf::DstT>& output)
{
    const uint8_t flag = self->ctx.l1ResPingPongFlag_ & 1;
    constexpr uint32_t halfResBytes = V516_L1_RES_HALF_BYTES;
    uint32_t tempOffset = flag * halfResBytes / sizeof(typename Intf::DstT);

    const uint64_t blockMStart = static_cast<uint64_t>(self->ctx.curMStartIdx_) +
                                 static_cast<uint64_t>(self->ctx.curMIdx_) * self->ctx.tiling_->baseM;
    const uint64_t gridFlatStart = blockMStart / self->ctx.tiling_->wi * self->ctx.tiling_->wi;
    const uint32_t hiCal = DivCeil(self->ctx.baseUseM_ + blockMStart % self->ctx.tiling_->wi, self->ctx.tiling_->wi);
    const uint32_t gridM = hiCal * self->ctx.tiling_->wi; // CONV 实际产出的 m 数
    const uint64_t mRemain = self->ctx.hiWi_ > gridFlatStart ? self->ctx.hiWi_ - gridFlatStart : 0;
    FixpipeParamsV516 instriParams;
    instriParams.mSize = static_cast<uint16_t>(gridM < mRemain ? gridM : mRemain); // CONV 网格整行（末块截断）
    instriParams.nSize = self->ctx.baseUseN_;                                      // currentNL0_（N: cin）
    instriParams.srcStride = static_cast<uint16_t>((gridM + BLOCK_CUBE - 1) / BLOCK_CUBE * BLOCK_CUBE);
    instriParams.dstStride = self->ctx.diHiWi_;
    instriParams.ndNum = 1;
    instriParams.srcNdStride = 0;
    instriParams.dstNdStride = 0;

    uint64_t dstOffset = static_cast<uint64_t>(self->ctx.curNIdx_) * self->ctx.tiling_->baseN * self->ctx.diHiWi_ +
                         static_cast<uint64_t>(self->ctx.curDinIdx_) * self->ctx.hiWi_ + gridFlatStart;

    LocalTensor<typename Intf::DstT> src = self->ctx.wholeL1ResTensor_[tempOffset];
    LocalTensor<typename Intf::DstT>& srcOut = src;

    WaitFlag<HardEvent::M_FIX>(flag);
    Fixpipe<typename Intf::DstT, typename Intf::DstT, CFG_COLUMN_MAJOR>(output[dstOffset], srcOut, instriParams);
    SetFlag<HardEvent::FIX_M>(flag);
    self->ctx.fixmEventSeen_[flag] = true;
}

template <class Intf>
static __aicore__ inline void LoadL0c2Gm(Intf* self, const GlobalTensor<typename Intf::DstT>& output)
{
    // 只有当参与了mmad运算时，才进行L0C的出队和拷贝
    if (!self->ctx.needComputeFlag_) {
        return;
    }

    self->ctx.l1ResPingPongFlag_ = (self->ctx.l1ResPingPongFlag_ - 1) & 1;
    StoreL1Result<Intf>(self, output);
    if (unlikely(self->ctx.kIter_ == 1)) {
        self->ctx.l1ResPingPongFlag_ ^= 1;
    }
}

} // namespace Convolution3DBackpropFunc

#endif // CONV3D_BP_STORE_ARCH51_H
