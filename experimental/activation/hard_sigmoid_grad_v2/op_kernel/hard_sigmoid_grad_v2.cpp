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
 * \file hard_sigmoid_grad_v2_arch32.cpp
 * \brief HardSigmoidGradV2 kernel entry (arch32)
 *
 * Template parameters (matching hard_sigmoid_grad_v2_tiling_key.h):
 *   - D_T_X: Data type, from ASCENDC_TPL_DATATYPE_DECL
 *   - BUFFER_MODE: Buffer mode (0=single, 1=double), from ASCENDC_TPL_UINT_DECL
 */

#include "hard_sigmoid_grad_v2.h"

template <typename D_T_X, int BUFFER_MODE>
__global__ __aicore__ void hard_sigmoid_grad_v2(
    GM_ADDR grad_output, GM_ADDR self, GM_ADDR grad_input,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(HardSigmoidGradV2TilingData);
    GET_TILING_DATA_WITH_STRUCT(HardSigmoidGradV2TilingData, tilingData, tiling);
    NsHardSigmoidGradV2::HardSigmoidGradV2<D_T_X, BUFFER_MODE> op;
    op.Init(grad_output, self, grad_input, &tilingData);
    op.Process();
}
