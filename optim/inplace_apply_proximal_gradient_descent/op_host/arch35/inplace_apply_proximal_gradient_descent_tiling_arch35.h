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
 * =============================================================================
 * inplace_apply_proximal_gradient_descent_package/op_host/arch35/inplace_apply_proximal_gradient_descent_tiling_arch35.h
 * =============================================================================
 * Role: Header declaring the TilingFuncInplaceApplyProximalGradientDescent
 *       callback for the InplaceApplyProximalGradientDescent operator on the
 *       ascend950 (arch35) platform.  Implements DESIGN §9.9 overall structure;
 *       registered per §9.8 via IMPL_OP_OPTILING + empty CompileInfo TilingParse.
 * =============================================================================
 */

#ifndef INPLACE_APPLY_PROXIMAL_GRADIENT_DESCENT_TILING_ARCH35_H
#define INPLACE_APPLY_PROXIMAL_GRADIENT_DESCENT_TILING_ARCH35_H

namespace optiling {

/**
 * TilingFuncInplaceApplyProximalGradientDescent: the tiling callback invoked by
 * the CANN framework before kernel launch.
 *
 * Parameters:
 *   context — [in/out] tiling context providing platform info, input shapes,
 *             input 0 descriptor datatype, and accepting the computed
 *             InplaceApplyProximalGradientDescentTilingData.
 *
 * Returns:
 *   ge::GRAPH_SUCCESS on success; GRAPH_FAILED on any §9.9 failure point
 *   (no TilingData/selector side effects beyond the documented order).
 */
ge::graphStatus TilingFuncInplaceApplyProximalGradientDescent(gert::TilingContext* context);

} // namespace optiling

#endif
