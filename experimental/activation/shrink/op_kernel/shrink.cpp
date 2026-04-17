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
 * @file shrink.cpp
 * @brief Shrink kernel entry
 */
#include "shrink.h"

template <typename D_T_X, int BUFFER_MODE>
__global__ __aicore__ void shrink(
    GM_ADDR self, GM_ADDR out,
    [[maybe_unused]] GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ShrinkTilingData);
    GET_TILING_DATA_WITH_STRUCT(ShrinkTilingData, tilingData, tiling);
    NsShrink::Shrink<D_T_X, BUFFER_MODE> op;
    op.Init(self, out, &tilingData);
    op.Process();
}
