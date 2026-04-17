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
 * \file relu6.cpp
 * \brief Relu6 Kernel 入口（arch35 = Ascend950）
 *
 * 模板参数说明（与 relu6_tiling_key.h 中 ASCENDC_TPL_ARGS_DECL 定义对应）：
 *   - D_T: 数据类型，由 ASCENDC_TPL_DATATYPE_DECL 定义
 *
 * 迭代一：单核 float16 骨架，D_T 模板自动推导实例化
 * 核函数参数顺序（固定）：
 *   GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling
 */

#include "relu6.h"

template <typename D_T>
__global__ __aicore__ void relu6(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(Relu6TilingData);
    GET_TILING_DATA_WITH_STRUCT(Relu6TilingData, tilingData, tiling);
    NsRelu6::Relu6<D_T> op;
    op.Init(x, y, &tilingData);
    op.Process();
}
