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
 * \file nonzero.cpp
 * \brief NonZero kernel entry — dtype decided by compile-time D_T_X (fp32/fp16/bf16/int32)
 */
// Framework build: enable the dtype-composed tiling key dispatch (see nonzero.h).
#define NONZERO_USE_TILING_KEY
#include "nonzero.h"

template <typename D_T_X>
__global__ __aicore__ void nonzero(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(NonzeroTilingData);
    GET_TILING_DATA_WITH_STRUCT(NonzeroTilingData, tilingData, tiling);

    NsNonzero::Nonzero<D_T_X> op;
    op.Init(x, workspace, y, &tilingData);
    op.Process();
}
