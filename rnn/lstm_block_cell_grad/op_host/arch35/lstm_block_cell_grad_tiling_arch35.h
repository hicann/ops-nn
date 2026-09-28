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
 * Header for the host-side tiling of the LSTMBlockCellGrad operator on the
 * ascend950 (arch35) platform.  Contains the LSTMBlockCellGradCompileInfo
 * struct (TilingParse carrier: coreNum / ubSize) and the declarations of the
 * runtime tiling entry and the compile-time tiling prepare functions.
 *
 * NOTE: the runtime TilingFunc does NOT consume LSTMBlockCellGradCompileInfo —
 * platform facts (coreNum / ubSize / sysWorkspaceSize) are re-queried via
 * context->GetPlatformInfo() + platform_ascendc::PlatformAscendC on every
 * call (the aclnn/UT path may carry a stale or dummy CompileInfo, so the
 * tiling math must never depend on it).
 */

#ifndef LSTM_BLOCK_CELL_GRAD_TILING_ARCH35_H
#define LSTM_BLOCK_CELL_GRAD_TILING_ARCH35_H

#include <cstdint>

#include "exe_graph/runtime/tiling_context.h"
#include "exe_graph/runtime/tiling_parse_context.h"

namespace optiling {

/**
 * LSTMBlockCellGradCompileInfo — platform information for the tiling compile
 * phase.  Populated by TilingPrepareForLSTMBlockCellGrad() with platform
 * hardware facts (number of AIV cores, UB bytes per core); read-only during
 * tiling (the runtime TilingFunc does not depend on it, see file header NOTE).
 *
 * Fields:
 *   coreNum — number of available AIV (AI Vector) cores on the NPU
 *   ubSize  — size of the Unified Buffer (UB) in bytes per core
 */
struct LSTMBlockCellGradCompileInfo {
    uint64_t coreNum; // AIV core count
    uint64_t ubSize;  // UB bytes per core
};

/**
 * TilingFuncLSTMBlockCellGrad: the runtime tiling callback for
 * LSTMBlockCellGrad on ascend950 (registered via
 * IMPL_OP_OPTILING(LSTMBlockCellGrad).Tiling(...)).
 *
 * Parameters:
 *   context — [in/out] tiling context providing input shapes/dtypes/attrs and
 *             platform info, and accepting LSTMBlockCellGradTilingData,
 *             tilingKey, blockDim, workspace sizes and schedule mode.
 *
 * Returns:
 *   ge::GRAPH_SUCCESS on success; ge::GRAPH_FAILED on negative validation
 *   (null input / rank violation / shape contract violation / dtype not
 *   supported — no tilingKey routing) or on invalid platform info /
 *   unsolvable UB budget (defensive).
 */
ge::graphStatus TilingFuncLSTMBlockCellGrad(gert::TilingContext* context);

/**
 * TilingPrepareForLSTMBlockCellGrad: compile-time preparation callback
 * (registered via IMPL_OP_OPTILING(LSTMBlockCellGrad)
 * .TilingParse<LSTMBlockCellGradCompileInfo>(...)).
 *
 * Fills LSTMBlockCellGradCompileInfo from the compile-time platform info.
 *
 * Returns:
 *   ge::GRAPH_SUCCESS on success, ge::GRAPH_FAILED on null context members.
 */
ge::graphStatus TilingPrepareForLSTMBlockCellGrad(gert::TilingParseContext* context);

} // namespace optiling

#endif // LSTM_BLOCK_CELL_GRAD_TILING_ARCH35_H
