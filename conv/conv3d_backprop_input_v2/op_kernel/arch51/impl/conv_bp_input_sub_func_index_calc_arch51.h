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
 * \file conv_bp_input_sub_func_index_calc_arch51.h
 * \brief
 */

#ifndef CONV3D_BP_INPUT_SUB_FUNC_INDEX_CALC_ARCH51_H
#define CONV3D_BP_INPUT_SUB_FUNC_INDEX_CALC_ARCH51_H

#include "conv_bp_input_sub_func_utils_arch51.h"

namespace Convolution3DBackpropFunc {

template <class Intf>
__aicore__ inline uint32_t CalcCurCinSizeB1(Intf* self, uint32_t curCinIdx)
{
    uint32_t curCinSize = self->ctx.tiling_->baseN;
    // consider tail
    uint32_t curCinRemain = self->ctx.singleShapeCin_ - curCinIdx;
    curCinSize = curCinSize < curCinRemain ? curCinSize : curCinRemain;
    return curCinSize;
}

template <class Intf>
__aicore__ inline void CalcCoutIndexAndSizeB1(Intf* self, uint64_t kIdx, uint32_t& curCoutIdx, uint32_t& curCoutSize)
{
    // 考虑到Preload的场景，L1的载入量要根据传入的KIdx确定，不能使用全局变量
    uint32_t kbL1Size = 0;
    if (unlikely(kIdx >= self->ctx.kIterStepKbTail)) {
        kbL1Size = (self->ctx.stepKbTail - 1) * self->ctx.tiling_->baseK + self->ctx.tailK_;
    } else {
        kbL1Size = self->ctx.curStepKb_ * self->ctx.tiling_->baseK;
    }

    curCoutIdx = DivHkWk<Intf>(self, kIdx * self->ctx.tiling_->baseK);
    curCoutSize = DivHkWk<Intf>(self, kbL1Size);

    uint32_t curCoutRemain = self->ctx.singleShapeCout_ - curCoutIdx;
    curCoutSize = curCoutSize < curCoutRemain ? curCoutSize : curCoutRemain;
}

} // namespace Convolution3DBackpropFunc

#endif
