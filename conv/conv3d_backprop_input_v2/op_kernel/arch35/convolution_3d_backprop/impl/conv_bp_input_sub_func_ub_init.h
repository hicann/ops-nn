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
 * \file conv_bp_input_sub_func_ub_init.h
 * \brief UB buffer initialization for conv3d backprop input
 */

#ifndef CONV3D_BP_INPUT_SUB_FUNC_UB_INIT_H
#define CONV3D_BP_INPUT_SUB_FUNC_UB_INIT_H

#include "conv_bp_input_sub_func_utils.h"

namespace Convolution3DBackpropFunc {

template <class Intf>
__aicore__ inline void InitUbByteSize(Intf* self)
{
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ == 3510) || __DAV_35_FAMILY__
    // 切K场景非fp32需要初始化AIV
    if (self->ctx.useUbAccumForSplitK_) {
        if (GetSubBlockIdx() != 0) {
            return;
        }
        if ASCEND_IS_AIV_SCALAR {
            self->ctx.pipe_.InitBuffer(self->ctx.vecBuf_, UB_SIZE);
        }
        return;
    }

    if constexpr (Intf::conv3dConfig.kernelSplitMode == TPL_SPLIT_KERNEL_HW) {
        if (GetSubBlockIdx() != 0) {
            return;
        }
        if (self->ctx.kSUseWorkSpace_) {
            if ASCEND_IS_AIV_SCALAR {
                self->ctx.pipe_.InitBuffer(self->ctx.vecBuf_, UB_SIZE);
            }
        } else {
            self->ctx.pipe_.InitBuffer(self->ctx.vecBuf_, UB_SIZE);
        }
    } else if constexpr (Intf::conv3dConfig.groupMode == TPL_GROUP_MODE_ENLARGE) {
        if ASCEND_IS_AIV_SCALAR {
            constexpr uint32_t GROUP_UB_BUF_SIZE = (UB_SIZE - AscendC::VECTOR_REG_WIDTH) / HALF_FACTOR;
            self->ctx.pipe_.InitBuffer(self->ctx.ndVecBuf_, GROUP_UB_BUF_SIZE);
            self->ctx.pipe_.InitBuffer(self->ctx.nzVecBuf_, GROUP_UB_BUF_SIZE);
            self->ctx.pipe_.InitBuffer(self->ctx.idxVecBuf_, AscendC::VECTOR_REG_WIDTH);
        }
    } else if constexpr (Intf::conv3dConfig.enableC04Flag) {
        if ASCEND_IS_AIV_SCALAR {
            constexpr uint32_t C04_UB_BUF_SIZE = (UB_SIZE - AscendC::VECTOR_REG_WIDTH - MASK_REG_WIDTH -
                                                  AscendC::ONE_BLOCK_SIZE) >>
                                                 1;
            self->ctx.pipe_.InitBuffer(self->ctx.ndVecBuf_, C04_UB_BUF_SIZE);
            self->ctx.pipe_.InitBuffer(self->ctx.nzVecBuf_, C04_UB_BUF_SIZE);
            self->ctx.pipe_.InitBuffer(self->ctx.idxVecBuf_, AscendC::VECTOR_REG_WIDTH);
            self->ctx.pipe_.InitBuffer(self->ctx.maskVecBuf_, MASK_REG_WIDTH);
        }
    } else {
        if ASCEND_IS_AIV_SCALAR {
            self->ctx.pipe_.InitBuffer(self->ctx.vecBuf_, UB_SIZE);
        }
    }
#endif
}

} // namespace Convolution3DBackpropFunc

#endif
