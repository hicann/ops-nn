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
 * \file hard_shrink_apt.cpp
 * \brief HardShrink 算子 Kernel 入口（arch35 架构，Ascend950）
 *
 * 模板参数（与 hard_shrink_tiling_key.h 中 ASCENDC_TPL_ARGS_DECL 定义对应）：
 *   - D_T: 计算数据类型 (half/float/bfloat16_t)
 *   - BUFFER_MODE: 缓冲模式 (0=单缓冲, 1=双缓冲)
 *   - IS_BF16: 是否为 bf16 输入 (0=否, 1=是，需 cast 到 float 计算)
 *
 * 注：lambd 作为 Attr 通过 TilingData 传递，不作为 kernel 参数
 */

#include "hard_shrink.h"

template <typename D_T, int BUFFER_MODE, int IS_BF16>
__global__ __aicore__ void hard_shrink(GM_ADDR self, GM_ADDR out,
    GM_ADDR workspace, GM_ADDR tiling)
{
    ENABLE_PRINTF();
    REGISTER_TILING_DEFAULT(HardShrinkTilingData);
    GET_TILING_DATA_WITH_STRUCT(HardShrinkTilingData, tilingData, tiling);

    NsHardShrink::HardShrink<D_T, BUFFER_MODE, IS_BF16> op;
    op.Init(self, out, &tilingData);
    op.Process();
}
