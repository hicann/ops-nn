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
#include "multilabel_margin_loss_impl.h"

template <uint32_t schMode>
__global__ __aicore__ void multilabel_margin_loss(GM_ADDR x, GM_ADDR target, GM_ADDR y, GM_ADDR is_target,
                                                  GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(MultilabelMarginLossTilingData);
    GET_TILING_DATA_WITH_STRUCT(MultilabelMarginLossTilingData, tilingData, tiling);

    if constexpr (schMode == MULTILABEL_MARGIN_LOSS_SCH_MODE_DEFAULT) {
        // User-usable workspace starts after the framework-reserved region (sysWorkspaceSize).
        // Writing partials to the raw workspace base would clobber the lib's internal area.
        GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);
        NsMultilabelMarginLoss::KernelMultilabelMarginLoss<DTYPE_X> op;
        op.Init(x, target, y, is_target, usrWorkspace, &tilingData);
        op.Process();
    }
}
