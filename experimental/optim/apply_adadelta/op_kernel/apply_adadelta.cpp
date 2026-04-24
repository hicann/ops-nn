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
/*!
 * \file apply_adadelta.cpp
 * \brief ApplyAdadelta kernel entry (arch35)
 *
 * Template parameters (matching apply_adadelta_tiling_key.h):
 *   - D_T_X:        Data type, from ASCENDC_TPL_DATATYPE_DECL
 *   - BUFFER_MODE:  Buffer mode (0=single, 1=double), from ASCENDC_TPL_UINT_DECL
 */

#include "apply_adadelta.h"

template <typename D_T_X, int BUFFER_MODE>
__global__ __aicore__ void apply_adadelta(
    GM_ADDR var, GM_ADDR accum, GM_ADDR accumUpdate, GM_ADDR grad,
    GM_ADDR varOut, GM_ADDR accumOut, GM_ADDR accumUpdateOut,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ApplyAdadeltaTilingData);
    GET_TILING_DATA_WITH_STRUCT(ApplyAdadeltaTilingData, tilingData, tiling);
    NsApplyAdadelta::ApplyAdadelta<D_T_X, BUFFER_MODE> op;
    op.Init(var, accum, accumUpdate, grad, varOut, accumOut, accumUpdateOut, &tilingData);
    op.Process();
}
