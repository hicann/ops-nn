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
* 我们正常的版权申明，下面是我们的备注
*
* NOTE: Portions of this code were AI-generated and have been
* technically reviewed for functional accuracy and security
*/

/**
 * \file foreach_div_scalar_arch32.cpp
 * \brief ForeachDivScalar kernel entry (arch32 / Ascend910B)
 *
 * Template parameter:
 *   - D_T_X: Data type, mapped from ASCENDC_TPL_DATATYPE_DECL
 *
 * Kernel arguments (fixed order: inputs -> outputs -> workspace -> tiling):
 *   - x:         TensorList input (contiguous GM memory)
 *   - scalar:    Scalar tensor (single float element on GM)
 *   - y:         TensorList output (contiguous GM memory)
 *   - workspace: Workspace memory
 *   - tiling:    Tiling data
 */

#include "foreach_div_scalar.h"

template <typename D_T_X>
__global__ __aicore__ void foreach_div_scalar(GM_ADDR x, GM_ADDR scalar, GM_ADDR y,
                                              GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ForeachDivScalarTilingData);
    GET_TILING_DATA_WITH_STRUCT(ForeachDivScalarTilingData, tilingData, tiling);
    NsForeachDivScalar::ForeachDivScalar<D_T_X> op;
    op.Init(x, scalar, y, &tilingData);
    op.Process();
}
