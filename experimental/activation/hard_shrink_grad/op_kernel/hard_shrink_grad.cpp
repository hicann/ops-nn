 /**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

 /**
 * NOTE: Portions of this code were AI-generated and have been
 * technically reviewed for functional accuracy and security
 */

/**
 * \file hard_shrink_grad.cpp
 * \brief HardShrinkGrad kernel entry point (arch32 architecture)
 *
 * Template parameters (matching ASCENDC_TPL_ARGS_DECL in hard_shrink_grad_tiling_key.h):
 *   - D_T: Data type, from ASCENDC_TPL_DATATYPE_DECL
 *   - BUFFER_MODE: Buffer mode (0=single, 1=double), from ASCENDC_TPL_UINT_DECL
 *
 * Kernel selection:
 *   - D_T = float  -> HardShrinkGradDirect (compute directly in fp32)
 *   - D_T = half   -> HardShrinkGradCastFp32 (Cast fp16->fp32, compute, Cast fp32->fp16)
 */

#include "hard_shrink_grad.h"

template <typename D_T, int BUFFER_MODE>
__global__ __aicore__ void hard_shrink_grad(GM_ADDR grad_output, GM_ADDR self, GM_ADDR output,
                                             GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(HardShrinkGradTilingData);
    GET_TILING_DATA_WITH_STRUCT(HardShrinkGradTilingData, tilingData, tiling);

    if constexpr (sizeof(D_T) == 2) {
        // fp16/bf16: Cast to fp32 for compute to avoid incorrect Compare on arch32
        NsHardShrinkGrad::HardShrinkGradCastFp32<D_T, BUFFER_MODE> op;
        op.Init(grad_output, self, output, &tilingData);
        op.Process();
    } else {
        // fp32: compute directly in native type
        NsHardShrinkGrad::HardShrinkGradDirect<D_T, BUFFER_MODE> op;
        op.Init(grad_output, self, output, &tilingData);
        op.Process();
    }
}
